# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: fla-core 0.5.2's causal_conv1d_fwd_kernel, l2norm_fwd_kernel and
# fused_beta_sigmoid_fwd_kernel (the arithmetic they define); no kernel code is copied.

"""GDN prefill features in one launch: conv + SiLU, q/k L2 norm, the a copy and beta.

Per token row of ``projected`` [T, P] (rows strided, columns q | k | v | z | b | a):

- q, k [T, Hk, K] and v [T, Hv, V] BF16, contiguous: SiLU of the causal conv over
  [history ++ the sequence's rows], in FP32, taps oldest first (product, then sum); the
  history is read in place from ``conv_state[slot]`` for continued sequences (``slot >= 0``
  and ``has_initial_state``), zeros otherwise. q and k are then L2-normalised per head from
  the BF16-rounded conv output, x / sqrt(sum x^2 + eps) in FP32;
- a [T, Hv] (projected's dtype), a dense copy; beta [T, Hv] FP32 = sigmoid(b).

Each row's sequence is found on the device from ``cu_seqlens``; the grid depends on T only.
Conv, SiLU, a and beta are bitwise fla-core's; the L2 norm's sum may round differently
(tier-1 bounded)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import CONV, EPS, HISTORY, Geometry, K, V

_BT = 32  # token rows per program
_BN = 128  # sequences per search step


@triton.jit
def sequence_of(t, t_in, cu_seqlens, count, BN: tl.constexpr):
    """Each row's sequence: the number of sequences ending at or before it."""
    seq = tl.zeros(t.shape, dtype=tl.int32)
    for start in range(0, count, BN):
        n = start + tl.arange(0, BN)
        ends = tl.load(cu_seqlens + 1 + n, mask=n < count, other=0).to(tl.int64)
        seq += tl.sum(((ends[None, :] <= t[:, None]) & (n < count)[None, :]).to(tl.int32), 1)
    return tl.minimum(seq, count - 1)


@triton.jit(do_not_specialize=["rows", "count"])
def _features_kernel(
    projected, p_stride, conv_weight, conv_state, cu_seqlens, slot_indices, has_initial,
    q, k, v, a, beta, rows, count,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BH: tl.constexpr,
    CONV: tl.constexpr, HISTORY: tl.constexpr, EPS: tl.constexpr,
    BT: tl.constexpr, BN: tl.constexpr,
):  # fmt: skip
    tile = tl.program_id(1)
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    t_in = t < rows
    row = projected + t * p_stride
    if tile == 2 * HK + HV:
        h = tl.arange(0, BH)
        m = t_in[:, None] & (h < HV)[None, :]
        gates = row[:, None] + (2 * HK + 2 * HV) * D + h[None, :]
        b = tl.load(gates, mask=m, other=0.0).to(tl.float32)
        tl.store(a + t[:, None] * HV + h[None, :], tl.load(gates + HV, mask=m, other=0.0), mask=m)
        tl.store(beta + t[:, None] * HV + h[None, :], tl.sigmoid(b), mask=m)
        return
    width: tl.constexpr = (2 * HK + HV) * D
    c = tl.arange(0, D)
    d = tile * D + c
    seq = sequence_of(t, t_in, cu_seqlens, count, BN)
    position = t - tl.load(cu_seqlens + seq, mask=t_in, other=0).to(tl.int64)
    slot = tl.load(slot_indices + seq, mask=t_in, other=-1).to(tl.int64)
    continued = t_in & (slot >= 0) & (tl.load(has_initial + seq, mask=t_in, other=0) != 0)
    history = conv_state + slot * (HISTORY * width)
    pre = tl.zeros((BT, D), dtype=tl.float32)
    for j in tl.static_range(CONV):
        back = HISTORY - j
        w = tl.load(conv_weight + d * CONV + j).to(tl.float32)
        inside = (t_in & (position >= back))[:, None]
        x = tl.load(row[:, None] - back * p_stride + d[None, :], mask=inside, other=0.0)
        before = (continued & (position < back))[:, None]
        past = history[:, None] + (HISTORY - back + position)[:, None] * width + d[None, :]
        x = x.to(tl.float32) + tl.load(past, mask=before, other=0.0).to(tl.float32)
        pre = pre + x * w[None, :]
    y = pre * tl.sigmoid(pre)
    if tile < 2 * HK:
        # L2 norm of the BF16-rounded conv output, one head per tile.
        y = y.to(q.dtype.element_ty).to(tl.float32)
        y = y * (1 / tl.sqrt(tl.sum(y * y, 1) + EPS))[:, None]
    tt = t[:, None]
    if tile < HK:
        tl.store(
            q + (tt * HK + tile) * D + c[None, :], y.to(q.dtype.element_ty), mask=t_in[:, None]
        )
    elif tile < 2 * HK:
        tl.store(k + (tt * HK + tile - HK) * D + c[None, :], y.to(k.dtype.element_ty),
                 mask=t_in[:, None])  # fmt: skip
    else:
        tl.store(v + (tt * HV + tile - 2 * HK) * D + c[None, :], y.to(v.dtype.element_ty),
                 mask=t_in[:, None])  # fmt: skip


def features(
    geometry: Geometry,
    projected: torch.Tensor,
    conv_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    slot_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_state: torch.Tensor,
    launch,
) -> tuple[torch.Tensor, ...]:
    """q, k [T, Hk, K], v [T, Hv, V] BF16, a [T, Hv] and beta [T, Hv] FP32 (see module)."""
    rows, hk, hv = projected.shape[0], geometry.k_heads, geometry.v_heads
    q, k = (projected.new_empty(rows, hk, K) for _ in range(2))
    v = projected.new_empty(rows, hv, V)
    a = projected.new_empty(rows, hv)
    beta = projected.new_empty(rows, hv, dtype=torch.float32)
    if rows:
        launch(
            _features_kernel,
            (triton.cdiv(rows, _BT), 2 * hk + hv + 1, 1),
            projected, projected.stride(0), conv_weight, conv_state, cu_seqlens, slot_indices,
            has_initial_state, q, k, v, a, beta, rows, cu_seqlens.shape[0] - 1,
            HK=hk, HV=hv, D=K, BH=triton.next_power_of_2(hv), CONV=CONV, HISTORY=HISTORY,
            EPS=EPS, BT=_BT, BN=_BN, num_warps=4,
        )  # fmt: skip
    return q, k, v, a, beta


__all__ = ["features", "sequence_of"]
