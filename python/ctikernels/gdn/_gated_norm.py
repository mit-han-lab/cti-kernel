# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/modules/fused_norm_gate.py`); see THIRD_PARTY_NOTICES.

"""Input gradients of the gated RMSNorm with a frozen weight, y = (o r) w silu(z) with r the
row's 1/rms (from the forward), for rows of D = 128 channels.

``gated_rms_norm_backward(o, z, weight, rstd, dy, do, dz)`` writes, per row, in FP32 and
rounded once to the outputs' dtype:

    x^ = o r,  y' = x^ w,  s = sigmoid(z)
    dz = (dy y') (s + (z s)(1 - s))
    e  = ((dy z) s) w
    do = (e - x^ c) r,  c = sum(x^ e) / D

No weight gradient is formed. The products are fla-core 0.5.2's, in its order and association,
over tiles of its shape ([BT, BD], BF16 operands widened to FP32), so a row sum reduces in the
same order and the results equal that kernel's (run with the weight) bit for bit, tile for
tile."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Every tile of fla-core's autotune space but 16 rows x 16 warps (whose row sum crosses warps)
# sums a row in one order; one tile of that order is pinned.
ROWS = 32
WARPS = 4


@triton.jit(do_not_specialize=["rows"])
def _gated_rms_norm_bwd_kernel(
    o, z, weight, rstd, dy, do, dz,
    o_stride, z_stride, dy_stride, do_stride, dz_stride,
    rows,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    channel = tl.arange(0, BD)
    row_in = row < rows
    channel_in = channel < D
    inside = row_in[:, None] & channel_in[None, :]
    # Row r is head r % H of token r // H; each tensor has its own token stride.
    within = ((row % H) * D)[:, None] + channel[None, :]
    token = (row // H)[:, None]
    w = tl.load(weight + channel, mask=channel_in).to(tl.float32)
    x = tl.load(o + token * o_stride + within, mask=inside, other=0.0).to(tl.float32)
    gate = tl.load(z + token * z_stride + within, mask=inside, other=0.0).to(tl.float32)
    grad = tl.load(dy + token * dy_stride + within, mask=inside, other=0.0).to(tl.float32)
    r = tl.load(rstd + row, mask=row_in, other=0.0)
    x_hat = tl.where(channel_in[None, :], x * r[:, None], 0.0)
    s = tl.sigmoid(gate)
    d_gate = grad * (x_hat * w[None, :]) * (s + gate * s * (1 - s))
    e = grad * gate * s * w[None, :]
    c = tl.sum(x_hat * e, axis=1) / D
    d_o = (e - x_hat * c[:, None]) * r[:, None]
    tl.store(do + token * do_stride + within, d_o.to(do.dtype.element_ty), mask=inside)
    tl.store(dz + token * dz_stride + within, d_gate.to(dz.dtype.element_ty), mask=inside)


def gated_rms_norm_backward(
    o: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    do: torch.Tensor,
    dz: torch.Tensor,
    *,
    rows_per_program: int = ROWS,
    warps: int = WARPS,
) -> None:
    """``do`` and ``dz`` [T, H, D] from o, z, dy [T, H, D], the weight [D] and the forward's
    rstd [T H] (FP32, row t H + h). Each [T, H, D] tensor has dense [H, D] rows and any token
    stride (z and dz may be column slices of a wider matrix). No host synchronisation."""
    t, h, d = o.shape
    for name, x in (("z", z), ("dy", dy), ("do", do), ("dz", dz)):
        if x.shape != (t, h, d):
            raise ValueError(f"{name} is {tuple(x.shape)}, expected {(t, h, d)}")
    for name, x in (("o", o), ("z", z), ("dy", dy), ("do", do), ("dz", dz)):
        if x.stride(2) != 1 or x.stride(1) != d:
            raise ValueError(f"{name} must have dense [H, D] rows")
    if not (weight.is_contiguous() and rstd.is_contiguous()):
        raise ValueError("weight and rstd must be contiguous")
    if weight.shape != (d,) or rstd.shape != (t * h,) or rstd.dtype != torch.float32:
        raise ValueError("weight must be [D] and rstd FP32 [T H]")
    if t * h == 0:
        return
    _gated_rms_norm_bwd_kernel[(triton.cdiv(t * h, rows_per_program),)](
        o, z, weight, rstd, dy, do, dz,
        o.stride(0), z.stride(0), dy.stride(0), do.stride(0), dz.stride(0), t * h,
        H=h, D=d, BT=rows_per_program, BD=triton.next_power_of_2(d), num_warps=warps,
    )  # fmt: skip


__all__ = ["gated_rms_norm_backward"]
