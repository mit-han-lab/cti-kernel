# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton residual add + LayerNorm, the provider of ``inference.add_layer_norm_``.

One program per row. The sum is formed in FP32 and rounded once to the stream dtype, as
torch's add stores it; the norm reads that stored sum. Statistics are FP32: a two-pass mean
and biased variance, and an IEEE ``1 / sqrt(var + eps)``. The output is ATen's
``weight * (rstd * (sum - mean)) + bias`` (layer_norm_kernel.cu) with the last multiply-add
fused, rounded once to the stream dtype. FMA contraction is otherwise off, and the reduction
layout depends on the hidden size only, so a row's bits never depend on the row count.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...triton.rounding import INTERPRET, round_to

# Rows up to this width stay in registers; wider rows loop over chunks of it.
MAX_BLOCK = 8192


@triton.jit
def _stored_sum(x_row, residual_row, columns, mask, DTYPE: tl.constexpr):
    residual = tl.load(residual_row + columns, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_row + columns, mask=mask, other=0.0).to(tl.float32)
    return round_to(residual + x, DTYPE)


@triton.jit
def _normalized(stored, mean, rstd, weight, bias, columns, mask, DTYPE: tl.constexpr):
    gamma = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(bias + columns, mask=mask, other=0.0).to(tl.float32)
    output = tl.math.fma(gamma, rstd * (stored - mean), beta)
    if INTERPRET:  # the interpreter's FP32 -> BF16 store truncates
        output = round_to(output, DTYPE)
    return output.to(DTYPE)


@triton.jit
def _mean(total, hidden: tl.constexpr):
    return tl.math.div_rn(total, tl.full((), hidden, tl.float32))


@triton.jit
def _rstd(squares, eps, hidden: tl.constexpr):
    variance = _mean(squares, hidden)
    return tl.math.div_rn(tl.full((), 1.0, tl.float32), tl.math.sqrt_rn(variance + eps))


@triton.jit(do_not_specialize=["x_stride", "residual_stride", "store_residual"])
def _add_layer_norm_kernel(
    x,
    residual,
    weight,
    bias,
    x_stride,
    residual_stride,
    store_residual,
    eps,
    hidden: tl.constexpr,
    BLOCK: tl.constexpr,
):
    DTYPE: tl.constexpr = x.dtype.element_ty
    row = tl.program_id(0).to(tl.int64)
    x_row = x + row * x_stride
    residual_row = residual + row * residual_stride
    if hidden <= BLOCK:
        columns = tl.arange(0, BLOCK)
        mask = columns < hidden
        stored = _stored_sum(x_row, residual_row, columns, mask, DTYPE)
        if store_residual:
            tl.store(residual_row + columns, stored.to(DTYPE), mask=mask)
        mean = _mean(tl.sum(stored, axis=0), hidden)
        centered = tl.where(mask, stored - mean, 0.0)
        rstd = _rstd(tl.sum(centered * centered, axis=0), eps, hidden)
        output = _normalized(stored, mean, rstd, weight, bias, columns, mask, DTYPE)
        tl.store(x_row + columns, output, mask=mask)
    else:
        # Every pass rebuilds the stored sum bit for bit; only the last one writes.
        lanes = tl.arange(0, BLOCK)
        totals = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, hidden, BLOCK):
            columns = start + lanes
            totals += _stored_sum(x_row, residual_row, columns, columns < hidden, DTYPE)
        mean = _mean(tl.sum(totals, axis=0), hidden)
        squares = tl.zeros((BLOCK,), tl.float32)
        for start in range(0, hidden, BLOCK):
            columns = start + lanes
            mask = columns < hidden
            stored = _stored_sum(x_row, residual_row, columns, mask, DTYPE)
            centered = tl.where(mask, stored - mean, 0.0)
            squares += centered * centered
        rstd = _rstd(tl.sum(squares, axis=0), eps, hidden)
        for start in range(0, hidden, BLOCK):
            columns = start + lanes
            mask = columns < hidden
            stored = _stored_sum(x_row, residual_row, columns, mask, DTYPE)
            if store_residual:
                tl.store(residual_row + columns, stored.to(DTYPE), mask=mask)
            output = _normalized(stored, mean, rstd, weight, bias, columns, mask, DTYPE)
            tl.store(x_row + columns, output, mask=mask)


def add_layer_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    store_residual: bool,
) -> None:
    """Launch on the current device over validated ``[M, hidden]`` rows with unit hidden stride."""
    rows, hidden = x.shape
    block = min(triton.next_power_of_2(hidden), MAX_BLOCK)
    _add_layer_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        bias,
        x.stride(0),
        residual.stride(0),
        # Triton types a bool argument i1 and an int i32; an int keeps both modes in one binary.
        int(store_residual),
        eps,
        hidden,
        block,
        # Warps follow the block alone (unmeasured); never the row count, as for the RMS norms.
        num_warps=max(1, min(16, block // 256)),
        enable_fp_fusion=False,
    )
