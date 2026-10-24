# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e (fla-core 0.5.2:
# `fla/ops/common/chunk_delta_h.py`); see THIRD_PARTY_NOTICES.

"""GDN training, the sequential state passes: one program per (sequence, value head, value
block of BV columns) walks the sequence's chunks with S [K, BV] (or its gradient) in FP32
registers; the per-chunk work runs in the parallel kernels of ``_train_output``.

- ``states``: stores the state entering each chunk (BF16, ``h`` [chunks, Hv, K, V]),
  v' = u - w S (BF16) and the chunk's output o = scale (2^gamma q S + lower(q k^T
  2^(gamma_i - gamma_j)) v') (BF16), then S <- 2^gamma_last S + k^T (2^(gamma_last - gamma) v')
  (v' scaled and rounded to BF16 for the product). Optional initial and final states
  [N, Hv, V, K] FP32.
- ``state_backward``: in reverse, stores dS of the state leaving each chunk (BF16, ``dh``),
  dv' = dv'_local + 2^(gamma_last - gamma) k dS (BF16), with the chunk's own term dv'_local =
  lower(q k^T 2^(gamma_i - gamma_j))^T scale do formed in the step (it does not wait on dS), then
  dS <- 2^gamma_last dS + scale q^T (2^gamma do) - w^T dv'; the initial state's gradient
  (FP32) when asked, from the final state's (FP32) when given.

Chunk c of sequence n is ``h`` row ``offsets[n] + c``, the chunk table's entry order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import K, V
from ._train_intra import CHUNK


@triton.jit
def _states_kernel(
    q, k, w, u, gamma, h, v_new, o, h0, ht, cu_seqlens, offsets, scale,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    CHUNK: tl.constexpr, INITIAL: tl.constexpr, FINAL: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)  # value block fastest, then head, then sequence
    vb, hv, n = pid % (D // BV), (pid // (D // BV)) % HV, pid // (D // BV * HV)
    hk = hv // (HV // HK)
    bos = tl.load(cu_seqlens + n).to(tl.int64)
    eos = tl.load(cu_seqlens + n + 1).to(tl.int64)
    base = tl.load(offsets + n).to(tl.int64)
    i = tl.arange(0, CHUNK)
    c = tl.arange(0, D)
    vc = vb * BV + tl.arange(0, BV)
    head = (hv * D + c)[:, None] * D + vc[None, :]  # S [K, BV] within one [Hv, K, V] state
    pool = (hv * D + vc)[None, :] * D + c[:, None]  # the same within one [Hv, V, K] pool row
    if INITIAL:
        S = tl.load(h0 + n * (HV * D * D) + pool)
    else:
        S = tl.zeros((D, BV), dtype=tl.float32)
    for chunk in range(0, tl.cdiv(eos - bos, CHUNK)):
        tl.store(h + (base + chunk) * (HV * D * D) + head, S.to(h.dtype.element_ty))
        t = bos + chunk * CHUNK + i
        m = t < eos
        ww = tl.load(w + (t * HV + hv)[:, None] * D + c[None, :], mask=m[:, None], other=0.0)
        tv = (t * HV + hv)[:, None] * D + vc[None, :]
        vn = tl.load(u + tv, mask=m[:, None], other=0.0).to(tl.float32)
        vn -= tl.dot(ww, S.to(ww.dtype))
        tl.store(v_new + tv, vn.to(v_new.dtype.element_ty), mask=m[:, None])
        gam = tl.load(gamma + t * HV + hv, mask=m, other=0.0)
        last = tl.load(gamma + (tl.minimum(eos, bos + chunk * CHUNK + CHUNK) - 1) * HV + hv)
        vnb = vn.to(ww.dtype)
        kt = tl.load(k + (t * HK + hk)[None, :] * D + c[:, None], mask=m[None, :], other=0.0)
        # The chunk's output, beside the update (it does not hold the next step back):
        # o = scale (2^gamma q S + lower(q k^T 2^(gamma_i - gamma_j)) v').
        qq = tl.load(q + (t * HK + hk)[:, None] * D + c[None, :], mask=m[:, None], other=0.0)
        P = tl.dot(qq, kt)
        P = tl.where((i[:, None] >= i[None, :]) & m[:, None],
                     P * tl.exp2(gam[:, None] - gam[None, :]), 0.0)  # fmt: skip
        out = tl.dot(qq, S.to(qq.dtype)) * (tl.exp2(gam) * scale)[:, None]
        out += tl.dot(P.to(vnb.dtype), vnb) * scale
        tl.store(o + tv, out.to(o.dtype.element_ty), mask=m[:, None])
        vs = tl.where(m[:, None], vn * tl.exp2(last - gam)[:, None], 0.0).to(ww.dtype)
        S = S * tl.exp2(last) + tl.dot(kt, vs)
    if FINAL:
        tl.store(ht + n * (HV * D * D) + pool, S)


@triton.jit
def _state_bwd_kernel(
    q, k, w, gamma, do, dh, du, dht, dh0, cu_seqlens, offsets, scale,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    CHUNK: tl.constexpr, INITIAL: tl.constexpr, FINAL: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)  # value block fastest, then head, then sequence
    vb, hv, n = pid % (D // BV), (pid // (D // BV)) % HV, pid // (D // BV * HV)
    hk = hv // (HV // HK)
    bos = tl.load(cu_seqlens + n).to(tl.int64)
    eos = tl.load(cu_seqlens + n + 1).to(tl.int64)
    base = tl.load(offsets + n).to(tl.int64)
    i = tl.arange(0, CHUNK)
    c = tl.arange(0, D)
    vc = vb * BV + tl.arange(0, BV)
    head = (hv * D + c)[:, None] * D + vc[None, :]
    pool = (hv * D + vc)[None, :] * D + c[:, None]
    if FINAL:
        dS = tl.load(dht + n * (HV * D * D) + pool)
    else:
        dS = tl.zeros((D, BV), dtype=tl.float32)
    for back in range(0, tl.cdiv(eos - bos, CHUNK)):
        chunk = tl.cdiv(eos - bos, CHUNK) - 1 - back
        tl.store(dh + (base + chunk) * (HV * D * D) + head, dS.to(dh.dtype.element_ty))
        t = bos + chunk * CHUNK + i
        m = t < eos
        gam = tl.load(gamma + t * HV + hv, mask=m, other=0.0)
        last = tl.load(gamma + (tl.minimum(eos, bos + chunk * CHUNK + CHUNK) - 1) * HV + hv)
        key = (t * HK + hk)[:, None] * D + c[None, :]
        kk = tl.load(k + key, mask=m[:, None], other=0.0)
        qt = tl.load(q + (t * HK + hk)[None, :] * D + c[:, None], mask=m[None, :], other=0.0)
        tv = (t * HV + hv)[:, None] * D + vc[None, :]
        ddo = tl.load(do + tv, mask=m[:, None], other=0.0)
        # dv'_local, independent of dS: [key row j, query row i] scale k_j.q_i
        # 2^(gamma_i - gamma_j) for j <= i, times do.
        Pt = tl.dot(kk, qt) * scale * tl.exp2(gam[None, :] - gam[:, None])
        Pt = tl.where((i[:, None] <= i[None, :]) & m[:, None] & m[None, :], Pt, 0.0)
        dvn = tl.dot(kk, dS.to(kk.dtype)) * tl.where(m, tl.exp2(last - gam), 0.0)[:, None]
        dvn += tl.dot(Pt.to(ddo.dtype), ddo)
        dvn = dvn.to(du.dtype.element_ty)
        tl.store(du + tv, dvn, mask=m[:, None])
        qgt = qt.to(tl.float32) * tl.exp2(gam)[None, :]
        wt = tl.load(w + (t * HV + hv)[None, :] * D + c[:, None], mask=m[None, :], other=0.0)
        dS = dS * tl.exp2(last) + scale * tl.dot(qgt, ddo.to(tl.float32), input_precision="tf32")
        dS -= tl.dot(wt, dvn)
    if INITIAL:
        tl.store(dh0 + n * (HV * D * D) + pool, dS)


def states(q, k, w, u, gamma, cu_seqlens, offsets, chunks, scale, config, initial_state=None,
           final=False):  # fmt: skip
    """(o [T, Hv, V] BF16, h [chunks, Hv, K, V] BF16, v' [T, Hv, V] BF16, final states
    [N, Hv, V, K] FP32 or None) from q, k [T, Hk, K], w and u (``_train_intra.solve``) and
    gamma."""
    rows, hv = u.shape[:2]
    count = cu_seqlens.shape[0] - 1
    h = k.new_empty(chunks, hv, K, V)
    v_new, o = torch.empty_like(u), torch.empty_like(u)
    ht = torch.empty(count, hv, V, K, dtype=torch.float32, device=k.device) if final else None
    # Value blocks of one (sequence, head) run side by side and share its rows in L2.
    _states_kernel[(V // config["BV"] * hv * count,)](
        q, k, w, u, gamma, h, v_new, o, initial_state, ht, cu_seqlens, offsets, scale,
        HK=k.shape[1], HV=hv, D=K, CHUNK=CHUNK, INITIAL=initial_state is not None, FINAL=final,
        **config,
    )  # fmt: skip
    return o, h, v_new, ht


def state_backward(q, k, w, gamma, do, cu_seqlens, offsets, chunks, scale, config,
                   dht=None, initial=False):  # fmt: skip
    """(dh [chunks, Hv, K, V] BF16, dv' [T, Hv, V] BF16, dh0 [N, Hv, V, K] FP32 or None)."""
    rows, hv = do.shape[:2]
    count = cu_seqlens.shape[0] - 1
    dh = k.new_empty(chunks, hv, K, V)
    du = torch.empty_like(do, dtype=k.dtype)
    dh0 = torch.empty(count, hv, V, K, dtype=torch.float32, device=k.device) if initial else None
    _state_bwd_kernel[(V // config["BV"] * hv * count,)](
        q, k, w, gamma, do, dh, du, dht, dh0, cu_seqlens, offsets, scale,
        HK=k.shape[1], HV=hv, D=K, CHUNK=CHUNK, INITIAL=initial, FINAL=dht is not None,
        **config,
    )  # fmt: skip
    return dh, du, dh0


__all__ = ["state_backward", "states"]
