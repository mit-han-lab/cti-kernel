# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Algorithm reference: the WY/UT form of the chunked gated delta rule (the flash-linear-attention
# papers; fla-core 0.5.2's gate, kkt-solve and recompute_w_u kernels); no kernel code is copied.

"""GDN prefill, intra-chunk part: per (64-token chunk, value head), independent of the state.

With g = -exp(A_log) softplus(a + dt_bias) and gamma its chunk-local cumulative sum (in log2
units), L_ij = beta_i k_i.k_j 2^(gamma_i - gamma_j) for i > j, and T = (I + L)^-1 (four 16x16
diagonal blocks by forward substitution, then the off-diagonal blocks, FP32 products in TF32
as fla-core's):

    w = T (beta 2^gamma k),  u = T (beta v)   (BF16 operands, FP32 accumulation, BF16 out).

Each program finds its (sequence, chunk) from ``cu_seqlens`` on the device; programs past the
batch's chunks return. Writes gamma [T, Hv] FP32 and w [T, Hv, K], u [T, Hv, V] BF16."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import Geometry, K, V

CHUNK = 64
_BC = 16  # diagonal block
# Two warps, one stage: fastest on RTX PRO 6000 at the anchor (37 vs 48 us at four, three).
_WARPS = 2
_STAGES = 1
_PRECISION = "tf32"  # the solve's FP32 products: fla-core's choice
_BN = 128  # sequences per scan step
RCP_LN2 = 1.4426950408889634


@triton.jit
def chunk_of(cu_seqlens, count, entry, CHUNK: tl.constexpr, BN: tl.constexpr):
    """The (sequence, chunk) of chunk table entry ``entry``; sequence == count past the end."""
    seq = tl.zeros((), dtype=tl.int32)
    begin = tl.zeros((), dtype=tl.int64)
    done = tl.zeros((), dtype=tl.int64)
    for start in range(0, count, BN):
        n = start + tl.arange(0, BN)
        m = n < count
        lengths = tl.load(cu_seqlens + n + 1, mask=m, other=0) - tl.load(
            cu_seqlens + n, mask=m, other=0
        )
        chunks = (lengths.to(tl.int64) + CHUNK - 1) // CHUNK
        ends = done + tl.cumsum(chunks, 0)
        before = m & (ends <= entry)
        seq += tl.sum(before.to(tl.int32), 0)
        begin += tl.sum(tl.where(before, chunks, 0), 0)
        done += tl.sum(chunks, 0)
    return seq, entry - begin


@triton.jit
def chunks_before(cu_seqlens, n, CHUNK: tl.constexpr, BN: tl.constexpr):
    """The chunk table entries of the sequences before sequence ``n``."""
    total = tl.zeros((), dtype=tl.int64)
    for start in range(0, n, BN):
        i = start + tl.arange(0, BN)
        m = i < n
        lengths = tl.load(cu_seqlens + i + 1, mask=m, other=0) - tl.load(
            cu_seqlens + i, mask=m, other=0
        )
        total += tl.sum((lengths.to(tl.int64) + CHUNK - 1) // CHUNK, 0)
    return total


@triton.jit
def _invert(L, BC: tl.constexpr):
    """(I + L)^-1 of a strictly lower [BC, BC] block, row by row (forward substitution)."""
    r = tl.arange(0, BC)
    eye = (r[:, None] == r[None, :]).to(tl.float32)
    X = eye
    for i in tl.static_range(1, BC):
        row = tl.sum(tl.where(r[:, None] == i, L, 0.0), 0)
        x = -tl.sum(row[:, None] * X, 0)
        X = tl.where(r[:, None] == i, x[None, :] + eye, X)
    return X


@triton.jit
def _dot(a, b, PRECISION: tl.constexpr):
    return tl.dot(a, b, input_precision=PRECISION)


@triton.jit
def _rows(k, a, beta, first, eos, blk, carry, decay, bias, gamma, hk, hv,
          HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BC: tl.constexpr,
          RCP_LN2: tl.constexpr):  # fmt: skip
    """Block ``blk`` of the chunk: rows t, mask, BF16 keys, beta, gamma (stored) and the
    carried sum of g."""
    t = first + blk * BC + tl.arange(0, BC)
    m = t < eos
    c = tl.arange(0, D)
    x = tl.load(a + t * HV + hv, mask=m, other=0.0).to(tl.float32) + bias
    g = tl.where(m, decay * tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x), 0.0)
    gam = (carry + tl.cumsum(g, 0)) * RCP_LN2
    tl.store(gamma + t * HV + hv, gam, mask=m)
    kk = tl.load(k + (t * HK + hk)[:, None] * D + c[None, :], mask=m[:, None], other=0.0)
    bb = tl.load(beta + t * HV + hv, mask=m, other=0.0)
    return t, m, kk, bb, gam, carry + tl.sum(g, 0)


@triton.jit
def _wu(X, ks, bs, gs, v, ts, ms, hv, HV: tl.constexpr, D: tl.constexpr):
    """X_rs (beta 2^gamma k_s) and X_rs (beta v_s) with BF16 operands, FP32 sums."""
    c = tl.arange(0, D)
    kb = (ks.to(tl.float32) * (bs * tl.exp2(gs))[:, None]).to(ks.dtype)
    vv = tl.load(v + (ts * HV + hv)[:, None] * D + c[None, :], mask=ms[:, None], other=0.0)
    vb = (vv.to(tl.float32) * bs[:, None]).to(vv.dtype)
    Xb = X.to(ks.dtype)
    return tl.dot(Xb, kb), tl.dot(Xb, vb)


@triton.jit
def _L(kr, br, gr, ks, gs, STRICT: tl.constexpr, BC: tl.constexpr):
    """L block: beta_i k_i.k_j 2^(gamma_i - gamma_j), strictly lower on the diagonal."""
    L = br[:, None] * tl.dot(kr, tl.trans(ks)) * tl.exp2(gr[:, None] - gs[None, :])
    if STRICT:
        i = tl.arange(0, BC)
        L = tl.where(i[:, None] > i[None, :], L, 0.0)
    return L


@triton.jit
def _store(w, u, t, m, hv, ww, uu, HV: tl.constexpr, D: tl.constexpr):
    out = (t * HV + hv)[:, None] * D + tl.arange(0, D)[None, :]
    tl.store(w + out, ww.to(w.dtype.element_ty), mask=m[:, None])
    tl.store(u + out, uu.to(u.dtype.element_ty), mask=m[:, None])


@triton.jit(do_not_specialize=["count"])
def _intra_kernel(
    k, v, a, beta, A_log, dt_bias, cu_seqlens, gamma, w, u, count,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr,
    CHUNK: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr, RCP_LN2: tl.constexpr,
    PRECISION: tl.constexpr,
):  # fmt: skip
    entry, hv = tl.program_id(0).to(tl.int64), tl.program_id(1)
    seq, chunk = chunk_of(cu_seqlens, count, entry, CHUNK, BN)
    if seq >= count:
        return
    first = tl.load(cu_seqlens + seq).to(tl.int64) + chunk * CHUNK
    eos = tl.load(cu_seqlens + seq + 1).to(tl.int64)
    hk = hv // (HV // HK)
    decay = -tl.exp(tl.load(A_log + hv).to(tl.float32))
    bias = tl.load(dt_bias + hv).to(tl.float32)
    t0, m0, k0, b0, g0, carry = _rows(
        k, a, beta, first, eos, 0, 0.0, decay, bias, gamma, hk, hv, HK, HV, D, BC, RCP_LN2
    )
    t1, m1, k1, b1, g1, carry = _rows(
        k, a, beta, first, eos, 1, carry, decay, bias, gamma, hk, hv, HK, HV, D, BC, RCP_LN2
    )
    t2, m2, k2, b2, g2, carry = _rows(
        k, a, beta, first, eos, 2, carry, decay, bias, gamma, hk, hv, HK, HV, D, BC, RCP_LN2
    )
    t3, m3, k3, b3, g3, carry = _rows(
        k, a, beta, first, eos, 3, carry, decay, bias, gamma, hk, hv, HK, HV, D, BC, RCP_LN2
    )
    # T = (I + L)^-1 by blocks: X_rr = (I + L_rr)^-1, X_rs = -X_rr sum_{s <= j < r} L_rj X_js.
    X00 = _invert(_L(k0, b0, g0, k0, g0, True, BC), BC)
    X11 = _invert(_L(k1, b1, g1, k1, g1, True, BC), BC)
    X22 = _invert(_L(k2, b2, g2, k2, g2, True, BC), BC)
    X33 = _invert(_L(k3, b3, g3, k3, g3, True, BC), BC)
    L10 = _L(k1, b1, g1, k0, g0, False, BC)
    L20 = _L(k2, b2, g2, k0, g0, False, BC)
    L21 = _L(k2, b2, g2, k1, g1, False, BC)
    L30 = _L(k3, b3, g3, k0, g0, False, BC)
    L31 = _L(k3, b3, g3, k1, g1, False, BC)
    L32 = _L(k3, b3, g3, k2, g2, False, BC)
    X10 = -_dot(X11, _dot(L10, X00, PRECISION), PRECISION)
    X21 = -_dot(X22, _dot(L21, X11, PRECISION), PRECISION)
    X32 = -_dot(X33, _dot(L32, X22, PRECISION), PRECISION)
    X20 = -_dot(X22, _dot(L20, X00, PRECISION) + _dot(L21, X10, PRECISION), PRECISION)
    X31 = -_dot(X33, _dot(L31, X11, PRECISION) + _dot(L32, X21, PRECISION), PRECISION)
    X30 = -_dot(
        X33,
        _dot(L30, X00, PRECISION) + _dot(L31, X10, PRECISION) + _dot(L32, X20, PRECISION),
        PRECISION,
    )
    # w = T (beta 2^gamma k), u = T (beta v), block row by block row.
    ww, uu = _wu(X00, k0, b0, g0, v, t0, m0, hv, HV, D)
    _store(w, u, t0, m0, hv, ww, uu, HV, D)
    ww, uu = _wu(X10, k0, b0, g0, v, t0, m0, hv, HV, D)
    dw, du = _wu(X11, k1, b1, g1, v, t1, m1, hv, HV, D)
    _store(w, u, t1, m1, hv, ww + dw, uu + du, HV, D)
    ww, uu = _wu(X20, k0, b0, g0, v, t0, m0, hv, HV, D)
    dw, du = _wu(X21, k1, b1, g1, v, t1, m1, hv, HV, D)
    ww, uu = ww + dw, uu + du
    dw, du = _wu(X22, k2, b2, g2, v, t2, m2, hv, HV, D)
    _store(w, u, t2, m2, hv, ww + dw, uu + du, HV, D)
    ww, uu = _wu(X30, k0, b0, g0, v, t0, m0, hv, HV, D)
    dw, du = _wu(X31, k1, b1, g1, v, t1, m1, hv, HV, D)
    ww, uu = ww + dw, uu + du
    dw, du = _wu(X32, k2, b2, g2, v, t2, m2, hv, HV, D)
    ww, uu = ww + dw, uu + du
    dw, du = _wu(X33, k3, b3, g3, v, t3, m3, hv, HV, D)
    _store(w, u, t3, m3, hv, ww + dw, uu + du, HV, D)


def intra(
    geometry: Geometry,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gamma, w, u) from the features (see the module)."""
    rows, hv = k.shape[0], geometry.v_heads
    count = cu_seqlens.shape[0] - 1
    gamma = torch.empty(rows, hv, dtype=torch.float32, device=k.device)
    w = k.new_empty(rows, hv, K)
    u = v.new_empty(rows, hv, V)
    capacity = (rows + (CHUNK - 1) * count) // CHUNK
    if capacity:
        launch(
            _intra_kernel,
            (capacity, hv, 1),
            k, v, a, beta, A_log, dt_bias, cu_seqlens, gamma, w, u, count,
            HK=geometry.k_heads, HV=hv, D=K, CHUNK=CHUNK, BC=_BC, BN=_BN, RCP_LN2=RCP_LN2,
            PRECISION=_PRECISION, num_warps=_WARPS, num_stages=_STAGES,
        )  # fmt: skip
    return gamma, w, u


__all__ = ["CHUNK", "RCP_LN2", "chunk_of", "chunks_before", "intra"]
