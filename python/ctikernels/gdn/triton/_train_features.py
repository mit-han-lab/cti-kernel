# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: fla-core 0.5.2's causal_conv1d, l2norm_fwd/bwd, fused_beta_sigmoid and
# layer_norm_gated_fwd kernels (the arithmetic they define); no kernel code is copied.

"""GDN training, the elementwise kernels around the chunked rule.

- ``features``: per token row of ``projected`` (strided rows) and 128-channel tile: SiLU of
  the causal conv within the row's sequence (zero history; FP32, taps oldest first, product
  then sum), rounded to BF16; q and k tiles are then L2-normalised x / sqrt(sum x^2 + eps)
  in FP32 (BF16 out, the FP32 1/norm kept), v stored; beta = sigmoid(b) FP32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import CONV, EPS, HISTORY, Geometry, K, V


@triton.jit(do_not_specialize=["rows"])
def _features_kernel(
    projected, p_stride, conv_weight, sequence, cu_seqlens, q, k, v, q_rstd, k_rstd, beta,
    rows,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BH: tl.constexpr,
    CONV: tl.constexpr, HISTORY: tl.constexpr, EPS: tl.constexpr, BT: tl.constexpr,
):  # fmt: skip
    tile = tl.program_id(1)
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    t_in = t < rows
    row = projected + t * p_stride
    if tile == 2 * HK + HV:
        h = tl.arange(0, BH)
        m = t_in[:, None] & (h < HV)[None, :]
        b = tl.load(row[:, None] + (2 * HK + 2 * HV) * D + h[None, :], mask=m, other=0.0)
        tl.store(beta + t[:, None] * HV + h[None, :], tl.sigmoid(b.to(tl.float32)), mask=m)
        return
    c = tl.arange(0, D)
    d = tile * D + c
    seq = tl.load(sequence + t, mask=t_in, other=0)
    position = t - tl.load(cu_seqlens + seq, mask=t_in, other=0).to(tl.int64)
    pre = tl.zeros((BT, D), dtype=tl.float32)
    for j in tl.static_range(CONV):
        back = HISTORY - j
        w = tl.load(conv_weight + d * CONV + j).to(tl.float32)
        inside = (t_in & (position >= back))[:, None]
        x = tl.load(row[:, None] - back * p_stride + d[None, :], mask=inside, other=0.0)
        pre = pre + x.to(tl.float32) * w[None, :]
    y = (pre * tl.sigmoid(pre)).to(v.dtype.element_ty)
    if tile < 2 * HK:
        y = y.to(tl.float32)
        r = 1 / tl.sqrt(tl.sum(y * y, 1) + EPS)
        y = (y * r[:, None]).to(q.dtype.element_ty)
        if tile < HK:
            x_out, inv, head = q, q_rstd, tile
        else:
            x_out, inv, head = k, k_rstd, tile - HK
        tl.store(x_out + (t[:, None] * HK + head) * D + c[None, :], y, mask=t_in[:, None])
        tl.store(inv + t * HK + head, r, mask=t_in)
    else:
        tl.store(v + (t[:, None] * HV + tile - 2 * HK) * D + c[None, :], y, mask=t_in[:, None])


def features(geometry: Geometry, projected, conv_weight, sequence, cu_seqlens, config):
    """q, k [T, Hk, K] BF16 (normalised) with their 1/norms [T, Hk] FP32, v [T, Hv, V] BF16
    and beta [T, Hv] FP32."""
    rows, hk, hv = projected.shape[0], geometry.k_heads, geometry.v_heads
    q, k = (projected.new_empty(rows, hk, K) for _ in range(2))
    q_rstd, k_rstd = (torch.empty(rows, hk, dtype=torch.float32, device=q.device) for _ in range(2))
    v = projected.new_empty(rows, hv, V)
    beta = torch.empty(rows, hv, dtype=torch.float32, device=q.device)
    _features_kernel[(triton.cdiv(rows, config["BT"]), 2 * hk + hv + 1)](
        projected, projected.stride(0), conv_weight, sequence, cu_seqlens, q, k, v, q_rstd,
        k_rstd, beta, rows, HK=hk, HV=hv, D=K, BH=triton.next_power_of_2(hv), CONV=CONV,
        HISTORY=HISTORY, EPS=EPS, **config,
    )  # fmt: skip
    return q, k, v, q_rstd, k_rstd, beta


__all__ = ["features"]
