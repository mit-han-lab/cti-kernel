# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e (fla-core 0.5.2:
# `fla/ops/gated_delta_rule/chunk_fwd.py`, `fla/ops/gated_delta_rule/wy_fast.py`,
# `fla/ops/gated_delta_rule/gate.py`); see THIRD_PARTY_NOTICES.

"""GDN training, intra-chunk kernels: one program per (64-token chunk, value head).

- ``solve``: g = -exp(A_log) softplus(a + dt_bias), gamma its chunk-local cumulative sum
  (log2 units, FP32), and T = (I + L)^-1 with L = strict_lower(beta_i k_i.k_j
  2^(gamma_i - gamma_j)), by 16-row diagonal blocks (forward substitution) and TF32 block
  products (as fla-core's); T is stored BF16 [T, Hv, 64] (row t holds its chunk row of T).
  From the same blocks: w = T (beta 2^gamma k), u = T (beta v) (BF16 operands, FP32 sums,
  BF16 out).
- ``wy_backward``: per (chunk, value head), the gradients through w and u (dk terms, dv,
  dbeta and the decay term), then the decay gradient's reverse cumulative sum over the chunk
  with the gate and beta-sigmoid backward, written straight into the b and a columns of d
  projected.

Chunks come from a device-built table [capacity, 2] (sequence, chunk); entries past the
batch's chunks return. No atomics: every output element has one writer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import K

CHUNK = 64
BC = 16  # diagonal block rows
RCP_LN2 = 1.4426950408889634


@triton.jit
def chunk_rows(table, cu_seqlens, entry, CHUNK: tl.constexpr):
    """The first row and the end of chunk table entry ``entry``."""
    seq = tl.load(table + entry * 2)
    chunk = tl.load(table + entry * 2 + 1).to(tl.int64)
    first = tl.load(cu_seqlens + seq).to(tl.int64) + chunk * CHUNK
    return first, tl.load(cu_seqlens + seq + 1).to(tl.int64)


@triton.jit
def _softplus(x):
    """log(1 + e^x), accurate for very negative x too (Kahan's log1p), x past 20 as is."""
    e = tl.exp(tl.minimum(x, 20.0))
    u = 1.0 + e
    return tl.where(x > 20.0, x, tl.where(u == 1.0, e, tl.log(u) * (e / (u - 1.0))))


@triton.jit
def _gates(a, a_stride, first, eos, blk, carry, decay, bias, gamma, hv,
           HV: tl.constexpr, BC: tl.constexpr, RCP_LN2: tl.constexpr):  # fmt: skip
    """Rows of diagonal block ``blk``: rows, mask, gamma (stored) and the carried sum of g."""
    t = first + blk * BC + tl.arange(0, BC)
    m = t < eos
    x = tl.load(a + t * a_stride + hv, mask=m, other=0.0).to(tl.float32) + bias
    g = tl.where(m, decay * _softplus(x), 0.0)
    gam = (carry + tl.cumsum(g, 0)) * RCP_LN2
    tl.store(gamma + t * HV + hv, gam, mask=m)
    return t, m, gam, carry + tl.sum(g, 0)


@triton.jit
def _lower(G, br, gr, gs, DIAGONAL: tl.constexpr, BC: tl.constexpr):
    """Block of L from its Gram block G = k_r k_s^T: beta_i G 2^(gamma_i - gamma_j), strictly
    lower on the diagonal."""
    L = br[:, None] * G * tl.exp2(gr[:, None] - gs[None, :])
    if DIAGONAL:
        i = tl.arange(0, BC)
        L = tl.where(i[:, None] > i[None, :], L, 0.0)
    return L


@triton.jit
def _invert(L, BC: tl.constexpr):
    """(I + L)^-1 of a strictly lower block, row by row."""
    r = tl.arange(0, BC)
    eye = (r[:, None] == r[None, :]).to(tl.float32)
    X = eye
    for i in range(1, BC):
        row = tl.sum(tl.where(r[:, None] == i, L, 0.0), 0)
        X = tl.where(r[:, None] == i, (eye - tl.sum(row[:, None] * X, 0)[None, :]), X)
    return X


@triton.jit
def _tf32(a, b):
    return tl.dot(a, b, input_precision="tf32")


@triton.jit
def _store_row(A, t, m, hv, x0, x1, x2, x3, HV: tl.constexpr, CHUNK: tl.constexpr,
               BC: tl.constexpr):  # fmt: skip
    """One block row of T: four [BC, BC] blocks side by side."""
    c = tl.arange(0, BC)
    base = A + (t * HV + hv)[:, None] * CHUNK + c[None, :]
    dt = A.dtype.element_ty
    tl.store(base, x0.to(dt), mask=m[:, None])
    tl.store(base + BC, x1.to(dt), mask=m[:, None])
    tl.store(base + 2 * BC, x2.to(dt), mask=m[:, None])
    tl.store(base + 3 * BC, x3.to(dt), mask=m[:, None])


@triton.jit
def _scaled(x, offsets, m, scale):
    """Rows of x scaled per row, rounded to x's dtype (the w / u operand)."""
    rows = tl.load(x + offsets, mask=m[:, None], other=0.0)
    return (rows * scale[:, None]).to(rows.dtype)


@triton.jit
def _store_wu(out, t0, t1, t2, t3, m0, m1, m2, m3, hv, cs, X00, X10, X11, X20, X21, X22, X30,
              X31, X32, X33, y0, y1, y2, y3, HV: tl.constexpr, D: tl.constexpr):  # fmt: skip
    """T times the four row blocks y_s, block row r = sum_{s <= r} X_rs y_s, stored."""
    dt = out.dtype.element_ty
    tl.store(out + (t0 * HV + hv)[:, None] * D + cs, tl.dot(X00, y0).to(dt), mask=m0[:, None])
    r1 = tl.dot(X10, y0) + tl.dot(X11, y1)
    tl.store(out + (t1 * HV + hv)[:, None] * D + cs, r1.to(dt), mask=m1[:, None])
    r2 = tl.dot(X20, y0) + tl.dot(X21, y1) + tl.dot(X22, y2)
    tl.store(out + (t2 * HV + hv)[:, None] * D + cs, r2.to(dt), mask=m2[:, None])
    r3 = tl.dot(X30, y0) + tl.dot(X31, y1) + tl.dot(X32, y2) + tl.dot(X33, y3)
    tl.store(out + (t3 * HV + hv)[:, None] * D + cs, r3.to(dt), mask=m3[:, None])


@triton.jit
def _solve_kernel(
    k, v, a, a_stride, beta, A_log, dt_bias, table, cu_seqlens, gamma, A, w, u,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr, BW: tl.constexpr, RCP_LN2: tl.constexpr,
):  # fmt: skip
    entry, hv = tl.program_id(0).to(tl.int64), tl.program_id(1)
    first, eos = chunk_rows(table, cu_seqlens, entry, CHUNK)
    if first >= eos:
        return
    hk = hv // (HV // HK)
    decay = -tl.exp(tl.load(A_log + hv).to(tl.float32))
    bias = tl.load(dt_bias + hv).to(tl.float32)
    t0, m0, g0, carry = _gates(a, a_stride, first, eos, 0, 0.0, decay, bias, gamma, hv, HV, BC,
                               RCP_LN2)  # fmt: skip
    t1, m1, g1, carry = _gates(a, a_stride, first, eos, 1, carry, decay, bias, gamma, hv, HV, BC,
                               RCP_LN2)  # fmt: skip
    t2, m2, g2, carry = _gates(a, a_stride, first, eos, 2, carry, decay, bias, gamma, hv, HV, BC,
                               RCP_LN2)  # fmt: skip
    t3, m3, g3, carry = _gates(a, a_stride, first, eos, 3, carry, decay, bias, gamma, hv, HV, BC,
                               RCP_LN2)  # fmt: skip
    b0 = tl.load(beta + t0 * HV + hv, mask=m0, other=0.0)
    b1 = tl.load(beta + t1 * HV + hv, mask=m1, other=0.0)
    b2 = tl.load(beta + t2 * HV + hv, mask=m2, other=0.0)
    b3 = tl.load(beta + t3 * HV + hv, mask=m3, other=0.0)
    # The ten Gram blocks k_r k_s^T (r >= s), over BK-column slices of the keys.
    G00 = tl.zeros((BC, BC), dtype=tl.float32)
    G10, G11 = tl.zeros((BC, BC), dtype=tl.float32), tl.zeros((BC, BC), dtype=tl.float32)
    G20, G21 = tl.zeros((BC, BC), dtype=tl.float32), tl.zeros((BC, BC), dtype=tl.float32)
    G22, G30 = tl.zeros((BC, BC), dtype=tl.float32), tl.zeros((BC, BC), dtype=tl.float32)
    G31, G32 = tl.zeros((BC, BC), dtype=tl.float32), tl.zeros((BC, BC), dtype=tl.float32)
    G33 = tl.zeros((BC, BC), dtype=tl.float32)
    for c0 in tl.static_range(0, D, BK):
        c = c0 + tl.arange(0, BK)[None, :]
        k0 = tl.load(k + (t0 * HK + hk)[:, None] * D + c, mask=m0[:, None], other=0.0)
        k1 = tl.load(k + (t1 * HK + hk)[:, None] * D + c, mask=m1[:, None], other=0.0)
        k2 = tl.load(k + (t2 * HK + hk)[:, None] * D + c, mask=m2[:, None], other=0.0)
        k3 = tl.load(k + (t3 * HK + hk)[:, None] * D + c, mask=m3[:, None], other=0.0)
        G00 += tl.dot(k0, tl.trans(k0))
        G10 += tl.dot(k1, tl.trans(k0))
        G11 += tl.dot(k1, tl.trans(k1))
        G20 += tl.dot(k2, tl.trans(k0))
        G21 += tl.dot(k2, tl.trans(k1))
        G22 += tl.dot(k2, tl.trans(k2))
        G30 += tl.dot(k3, tl.trans(k0))
        G31 += tl.dot(k3, tl.trans(k1))
        G32 += tl.dot(k3, tl.trans(k2))
        G33 += tl.dot(k3, tl.trans(k3))
    X00 = _invert(_lower(G00, b0, g0, g0, True, BC), BC)
    X11 = _invert(_lower(G11, b1, g1, g1, True, BC), BC)
    X22 = _invert(_lower(G22, b2, g2, g2, True, BC), BC)
    X33 = _invert(_lower(G33, b3, g3, g3, True, BC), BC)
    L10 = _lower(G10, b1, g1, g0, False, BC)
    L20 = _lower(G20, b2, g2, g0, False, BC)
    L21 = _lower(G21, b2, g2, g1, False, BC)
    L30 = _lower(G30, b3, g3, g0, False, BC)
    L31 = _lower(G31, b3, g3, g1, False, BC)
    L32 = _lower(G32, b3, g3, g2, False, BC)
    # X_rs = -X_rr sum_{s <= j < r} L_rj X_js
    X10 = -_tf32(X11, _tf32(L10, X00))
    X21 = -_tf32(X22, _tf32(L21, X11))
    X32 = -_tf32(X33, _tf32(L32, X22))
    X20 = -_tf32(X22, _tf32(L20, X00) + _tf32(L21, X10))
    X31 = -_tf32(X33, _tf32(L31, X11) + _tf32(L32, X21))
    X30 = -_tf32(X33, _tf32(L30, X00) + _tf32(L31, X10) + _tf32(L32, X20))
    zero = tl.zeros((BC, BC), dtype=tl.float32)
    _store_row(A, t0, m0, hv, X00, zero, zero, zero, HV, CHUNK, BC)
    _store_row(A, t1, m1, hv, X10, X11, zero, zero, HV, CHUNK, BC)
    _store_row(A, t2, m2, hv, X20, X21, X22, zero, HV, CHUNK, BC)
    _store_row(A, t3, m3, hv, X30, X31, X32, X33, HV, CHUNK, BC)
    # w = T (beta 2^gamma k) and u = T (beta v) from the blocks above, BW columns at a time
    # (BF16 operands, FP32 sums, one rounding at the store).
    dt = k.dtype.element_ty
    X00, X10, X11 = X00.to(dt), X10.to(dt), X11.to(dt)
    X20, X21, X22 = X20.to(dt), X21.to(dt), X22.to(dt)
    X30, X31, X32, X33 = X30.to(dt), X31.to(dt), X32.to(dt), X33.to(dt)
    bg0, bg1 = b0 * tl.exp2(g0), b1 * tl.exp2(g1)
    bg2, bg3 = b2 * tl.exp2(g2), b3 * tl.exp2(g3)
    for c0 in range(0, D, BW):
        cs = c0 + tl.arange(0, BW)[None, :]
        y0 = _scaled(k, (t0 * HK + hk)[:, None] * D + cs, m0, bg0)
        y1 = _scaled(k, (t1 * HK + hk)[:, None] * D + cs, m1, bg1)
        y2 = _scaled(k, (t2 * HK + hk)[:, None] * D + cs, m2, bg2)
        y3 = _scaled(k, (t3 * HK + hk)[:, None] * D + cs, m3, bg3)
        _store_wu(w, t0, t1, t2, t3, m0, m1, m2, m3, hv, cs, X00, X10, X11, X20, X21, X22, X30,
                  X31, X32, X33, y0, y1, y2, y3, HV, D)  # fmt: skip
        y0 = _scaled(v, (t0 * HV + hv)[:, None] * D + cs, m0, b0)
        y1 = _scaled(v, (t1 * HV + hv)[:, None] * D + cs, m1, b1)
        y2 = _scaled(v, (t2 * HV + hv)[:, None] * D + cs, m2, b2)
        y3 = _scaled(v, (t3 * HV + hv)[:, None] * D + cs, m3, b3)
        _store_wu(u, t0, t1, t2, t3, m0, m1, m2, m3, hv, cs, X00, X10, X11, X20, X21, X22, X30,
                  X31, X32, X33, y0, y1, y2, y3, HV, D)  # fmt: skip


@triton.jit(do_not_specialize=["rows"])
def _wy_bwd_kernel(
    k, v, beta, gamma, A, dw, du, dk, dv, dg_parts, a, a_stride, b_raw, b_stride, A_log,
    dt_bias, grad_b, db_stride, grad_a, da_stride, table, cu_seqlens, rows,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
    BD: tl.constexpr, PARTS: tl.constexpr,
):  # fmt: skip
    entry, hv = tl.program_id(0).to(tl.int64), tl.program_id(1)
    first, eos = chunk_rows(table, cu_seqlens, entry, CHUNK)
    if first >= eos:
        return
    hk = hv // (HV // HK)
    i = tl.arange(0, CHUNK)
    t = first + i
    m = t < eos
    mm = m[:, None] & m[None, :]
    strict = (i[:, None] > i[None, :]) & mm
    key = (t * HK + hk)[:, None] * D
    head = (t * HV + hv)[:, None] * D
    c = tl.arange(0, BD)[None, :]
    # T^T: element [j, i] = T[i, j].
    Tt = tl.load(A + (t * HV + hv)[None, :] * CHUNK + i[:, None], mask=mm, other=0.0)
    b = tl.load(beta + t * HV + hv, mask=m, other=0.0)
    gam = tl.load(gamma + t * HV + hv, mask=m, other=0.0)
    bg = b * tl.exp2(gam)
    db = tl.zeros((CHUNK,), dtype=tl.float32)
    dg = tl.zeros((CHUNK,), dtype=tl.float32)
    dA = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
    for c0 in range(0, D, BD):  # dL's term through w = T (beta 2^gamma k)
        kk = tl.load(k + key + c0 + c, mask=m[:, None], other=0.0)
        ddw = tl.load(dw + head + c0 + c, mask=m[:, None], other=0.0)
        dA += tl.dot(ddw, tl.trans((kk * bg[:, None]).to(ddw.dtype)))
    for c0 in range(0, D, BD):  # through u = T (beta v)
        vv = tl.load(v + head + c0 + c, mask=m[:, None], other=0.0)
        ddu = tl.load(du + head + c0 + c, mask=m[:, None], other=0.0)
        dA += tl.dot(ddu, tl.trans((vv * b[:, None]).to(vv.dtype)))
        dvb = tl.dot(Tt, ddu)
        db += tl.sum(dvb * vv.to(tl.float32), 1)
        tl.store(dv + head + c0 + c, (dvb * b[:, None]).to(dv.dtype.element_ty), mask=m[:, None])
    # Through T = (I + L)^-1: dL = -T^T dT T^T, then L's own factors.
    dA = tl.where(strict, dA, 0.0)
    dA = tl.dot(Tt, tl.dot(dA.to(Tt.dtype), Tt).to(Tt.dtype))
    dA = tl.where(strict, -dA * tl.exp2(gam[:, None] - gam[None, :]), 0.0).to(Tt.dtype)
    Ab = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
    for c0 in range(0, D, BD):  # this value head's dk terms: w's and L's
        kk = tl.load(k + key + c0 + c, mask=m[:, None], other=0.0)
        kf = kk.to(tl.float32)
        dkbg = tl.dot(Tt, tl.load(dw + head + c0 + c, mask=m[:, None], other=0.0))
        db += tl.sum(dkbg * kf * tl.exp2(gam)[:, None], 1)
        dg += tl.sum(dkbg * kf * bg[:, None], 1)
        Ab += tl.dot(kk, tl.trans(kk))
        dkb = tl.dot(dA, kk)
        db += tl.sum(dkb * kf, 1)
        delta = dkbg * bg[:, None] + dkb * b[:, None]
        delta += tl.dot(tl.trans(dA), (kk * b[:, None]).to(kk.dtype))
        tl.store(dk + head + c0 + c, delta.to(dk.dtype.element_ty), mask=m[:, None])
    AdA = dA.to(tl.float32) * Ab * b[:, None]
    dg += tl.sum(AdA, 1) - tl.sum(AdA, 0)
    for p in tl.static_range(PARTS):
        dg += tl.load(dg_parts + p * rows * HV + t * HV + hv, mask=m, other=0.0)
    # gamma is a cumulative sum of g, and g = -exp(A_log) softplus(a + dt_bias).
    dg = tl.where(m, tl.cumsum(dg, 0, reverse=True), 0.0)
    x = tl.load(a + t * a_stride + hv, mask=m, other=0.0).to(tl.float32)
    x += tl.load(dt_bias + hv).to(tl.float32)
    da = -tl.exp(tl.load(A_log + hv).to(tl.float32)) * (dg * tl.sigmoid(x))
    tl.store(grad_a + t * da_stride + hv, da.to(grad_a.dtype.element_ty), mask=m)
    s = tl.sigmoid(tl.load(b_raw + t * b_stride + hv, mask=m, other=0.0).to(tl.float32))
    tl.store(grad_b + t * db_stride + hv, (db * s * (1 - s)).to(grad_b.dtype.element_ty), mask=m)


def solve(k, v, a, beta, A_log, dt_bias, table, cu_seqlens, config):
    """gamma [T, Hv] FP32, T [T, Hv, 64] BF16 and w [T, Hv, K], u [T, Hv, V] BF16 of keys
    k [T, Hk, K], values v [T, Hv, V], a [T, Hv] (strided rows) and beta [T, Hv] FP32."""
    rows, hk = k.shape[:2]
    v_heads = v.shape[1]
    gamma = torch.empty(rows, v_heads, dtype=torch.float32, device=k.device)
    A = k.new_empty(rows, v_heads, CHUNK)
    w, u = k.new_empty(rows, v_heads, K), torch.empty_like(v)
    _solve_kernel[(table.shape[0], v_heads)](
        k, v, a, a.stride(0), beta, A_log, dt_bias, table, cu_seqlens, gamma, A, w, u,
        HK=hk, HV=v_heads, D=K, CHUNK=CHUNK, BC=BC, RCP_LN2=RCP_LN2, **config,
    )  # fmt: skip
    return gamma, A, w, u


def wy_backward(k, v, beta, gamma, A, dw, du, dg_parts, a, b, A_log, dt_bias, da, db, table,
                cu_seqlens, config):  # fmt: skip
    """The WY backward per (chunk, value head): returns each value head's dk terms
    [T, Hv, K] BF16 and dv [T, Hv, V] BF16 (as fla-core keeps them), and writes the gradients of
    the raw gate inputs a and b (beta = sigmoid(b)) into ``da`` and ``db`` (all [T, Hv],
    strided rows)."""
    rows, hv = v.shape[:2]
    dk, dv = k.new_empty(rows, hv, K), torch.empty_like(v)
    _wy_bwd_kernel[(table.shape[0], hv)](
        k, v, beta, gamma, A, dw, du, dk, dv, dg_parts, a, a.stride(0), b, b.stride(0), A_log,
        dt_bias, db, db.stride(0), da, da.stride(0), table, cu_seqlens, rows,
        HK=k.shape[1], HV=hv, D=K, CHUNK=CHUNK, PARTS=dg_parts.shape[0], **config,
    )  # fmt: skip
    return dk, dv


__all__ = ["CHUNK", "RCP_LN2", "chunk_rows", "solve", "wy_backward"]
