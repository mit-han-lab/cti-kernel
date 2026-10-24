# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e (fla-core 0.5.2:
# `fla/ops/common/chunk_o.py`); see THIRD_PARTY_NOTICES.

"""GDN training, the parallel per-chunk kernels around the state passes (``_train_state``):
one program per (chunk table entry, head[, block]).

- ``gated_norm``: y = (o r) w silu(z) with r = 1/sqrt(mean(o^2) + eps) per value head (FP32
  arithmetic on the BF16 o from ``_train_state.states``), keeping r for ``gdn._gated_norm``'s
  backward.
- ``dqkw``: per (key block, chunk, key head), the key head's value heads in turn: dq and dk
  (summed over them in FP32), dw = -dv' S^T (BF16) and the decay gradient by gamma, one FP32
  part per key block (summed, in a fixed order, by ``_train_intra.wy_backward``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import EPS, K, V
from ._train_intra import CHUNK, chunk_rows


@triton.jit(do_not_specialize=["rows"])
def _gated_norm_kernel(
    o, z, z_stride, norm_weight, y, rstd, rows,
    HV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)  # (token, value head) rows
    m = r < rows * HV
    c = tl.arange(0, D)[None, :]
    x = tl.load(o + r[:, None] * D + c, mask=m[:, None], other=0.0).to(tl.float32)
    inv = 1 / tl.sqrt(tl.sum(x * x, 1) / D + EPS)
    zz = tl.load(z + (r // HV)[:, None] * z_stride + (r % HV)[:, None] * D + c, mask=m[:, None],
                 other=0.0).to(tl.float32)  # fmt: skip
    out = x * inv[:, None] * tl.load(norm_weight + c) * (zz * tl.sigmoid(zz))
    tl.store(y + r[:, None] * D + c, out.to(y.dtype.element_ty), mask=m[:, None])
    tl.store(rstd + r, inv, mask=m)


@triton.jit(do_not_specialize=["rows"])
def _dqkw_kernel(
    q, k, v_new, gamma, h, do, dh, du, dq, dk, dw, dg_parts, table, cu_seqlens, scale, rows,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    CHUNK: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)  # key block fastest, then key head, then chunk
    kb, hk, entry = pid % (D // BK), (pid // (D // BK)) % HK, pid // (D // BK * HK)
    first, eos = chunk_rows(table, cu_seqlens, entry, CHUNK)
    if first >= eos:
        return
    i = tl.arange(0, CHUNK)
    t = first + i
    m = t < eos
    kc = kb * BK + tl.arange(0, BK)
    key = (t * HK + hk)[:, None] * D + kc[None, :]
    qq = tl.load(q + key, mask=m[:, None], other=0.0)
    kk = tl.load(k + key, mask=m[:, None], other=0.0)
    lower = (i[:, None] >= i[None, :]) & m[:, None] & m[None, :]
    is_last = t == tl.minimum(eos, first + CHUNK) - 1
    dq_sum = tl.zeros((CHUNK, BK), dtype=tl.float32)
    dk_sum = tl.zeros((CHUNK, BK), dtype=tl.float32)
    for g in range(HV // HK):  # the key head's value heads, summed in order
        hv = hk * (HV // HK) + g
        state = h + (entry * HV + hv) * (D * D) + kc[None, :] * D  # [K, V]: S^T blocks
        dstate = dh + (entry * HV + hv) * (D * D) + kc[None, :] * D
        dq_ = tl.zeros((CHUNK, BK), dtype=tl.float32)
        dk_ = tl.zeros((CHUNK, BK), dtype=tl.float32)
        dw_ = tl.zeros((CHUNK, BK), dtype=tl.float32)
        ds = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
        dlast = tl.zeros((), dtype=tl.float32)
        for v0 in range(0, D, BV):
            vc = v0 + tl.arange(0, BV)
            tv = (t * HV + hv)[:, None] * D + vc[None, :]
            vn = tl.load(v_new + tv, mask=m[:, None], other=0.0)
            ddo = tl.load(do + tv, mask=m[:, None], other=0.0)
            ddu = tl.load(du + tv, mask=m[:, None], other=0.0)
            St = tl.load(state + vc[:, None])  # S^T [BV, BK]
            dSt = tl.load(dstate + vc[:, None])
            dlast += tl.sum(St.to(tl.float32) * dSt.to(tl.float32))
            ds += tl.dot(ddo, tl.trans(vn))
            dq_ += tl.dot(ddo, St)
            dk_ += tl.dot(vn, dSt)
            dw_ += tl.dot(ddu, St)
        out = (t * HV + hv)[:, None] * D + kc[None, :]
        tl.store(dw + out, (-dw_).to(dw.dtype.element_ty), mask=m[:, None])
        gam = tl.load(gamma + t * HV + hv, mask=m, other=0.0)
        last = tl.load(gamma + (tl.minimum(eos, first + CHUNK) - 1) * HV + hv)
        dlast *= tl.exp2(last)
        dq_ = dq_ * tl.exp2(gam)[:, None] * scale
        dk_ = dk_ * tl.where(m, tl.exp2(last - gam), 0.0)[:, None]
        dlast += tl.sum(dk_ * kk.to(tl.float32))
        ds = tl.where(lower, ds * tl.exp2(gam[:, None] - gam[None, :]), 0.0) * scale
        ds = ds.to(kk.dtype)
        dq_ += tl.dot(ds, kk)
        dk_ += tl.dot(tl.trans(ds), qq)
        dg = tl.sum(dq_ * qq.to(tl.float32), 1) - tl.sum(dk_ * kk.to(tl.float32), 1)
        dg = tl.where(is_last, dg + dlast, dg)
        tl.store(dg_parts + kb * rows * HV + t * HV + hv, dg, mask=m)
        dq_sum += dq_
        dk_sum += dk_
    tl.store(dq + key, dq_sum, mask=m[:, None])
    tl.store(dk + key, dk_sum, mask=m[:, None])


def gated_norm(o, z, norm_weight, config):
    """The gated RMSNorm y = (o r) w silu(z), r = 1/sqrt(mean(o^2) + eps) per value head:
    y [T, Hv V] BF16 and r [T Hv] FP32 (for ``gdn._gated_norm``'s backward) of o [T, Hv, V]
    and z [T, Hv V] (strided rows)."""
    rows, hv = o.shape[:2]
    y = o.new_empty(rows, hv * V)
    rstd = torch.empty(rows * hv, dtype=torch.float32, device=o.device)
    _gated_norm_kernel[(triton.cdiv(rows * hv, config["BT"]),)](
        o, z, z.stride(0), norm_weight, y, rstd, rows, HV=hv, D=V, EPS=EPS, **config
    )
    return y, rstd


def dqkw(q, k, v_new, gamma, h, do, dh, du, table, cu_seqlens, scale, config):
    """(dq, dk [T, Hk, K] FP32 summed over each key head's value heads, dw [T, Hv, K] BF16,
    dg parts [K / BK, T, Hv] FP32)."""
    rows, hk = q.shape[:2]
    hv = do.shape[1]
    dq, dk = (torch.empty(rows, hk, K, dtype=torch.float32, device=do.device) for _ in range(2))
    dw = q.new_empty(rows, hv, K)
    parts = torch.empty(K // config["BK"], rows, hv, dtype=torch.float32, device=do.device)
    # Key blocks of one (chunk, key head) run side by side and share its rows in L2.
    _dqkw_kernel[(K // config["BK"] * hk * table.shape[0],)](
        q, k, v_new, gamma, h, do, dh, du, dq, dk, dw, parts, table, cu_seqlens, scale, rows,
        HK=hk, HV=hv, D=K, CHUNK=CHUNK, **config,
    )  # fmt: skip
    return dq, dk, dw, parts


__all__ = ["dqkw", "gated_norm"]
