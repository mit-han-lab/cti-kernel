# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA bce39a4efe9b2c70f546133ef51bb875e3e00071
# (`fla/modules/activations.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..triton.silu import sigmoid, silu

_BLOCK_SIZE = 1024


@triton.jit(do_not_specialize=["num_output_elements"])
def _forward_kernel(
    x,
    output,
    num_output_elements,
    x_stride,
    output_stride,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_output_elements
    row = offsets // width
    column = offsets % width
    row_start = row * x_stride

    gate = tl.load(x + row_start + column, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + row_start + width + column, mask=mask, other=0.0).to(tl.float32)
    value = silu(gate) * up
    tl.store(output + row * output_stride + column, value.to(output.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["num_output_elements"])
def _backward_kernel(
    x,
    grad_output,
    grad_x,
    num_output_elements,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_output_elements
    row = offsets // width
    column = offsets % width
    row_start = row * (2 * width)

    gate = tl.load(x + row_start + column, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x + row_start + width + column, mask=mask, other=0.0).to(tl.float32)
    upstream = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = sigmoid(gate)
    grad_gate = upstream * sig * (1.0 + gate * (1.0 - sig)) * up
    grad_up = upstream * silu(gate)

    tl.store(grad_x + row_start + column, grad_gate.to(grad_x.dtype.element_ty), mask=mask)
    tl.store(
        grad_x + row_start + width + column,
        grad_up.to(grad_x.dtype.element_ty),
        mask=mask,
    )


def row_stride(tensor: torch.Tensor) -> int | None:
    """The single row stride of ``tensor`` with a unit last stride, else ``None``."""

    columns = tensor.shape[-1]
    try:
        rows = tensor.view(-1, columns)
    except RuntimeError:
        return None
    if rows.stride(1) != 1 and columns > 1:
        return None
    return rows.stride(0) if rows.shape[0] > 1 else columns


def silu_and_mul_forward(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Rows of ``x`` and ``out`` may be strided; each needs a unit last stride."""

    width = x.shape[-1] // 2
    output = (
        torch.empty(x.shape[:-1] + (width,), device=x.device, dtype=x.dtype) if out is None else out
    )
    num_output_elements = output.numel()
    grid = (triton.cdiv(num_output_elements, _BLOCK_SIZE),)
    _forward_kernel[grid](
        x,
        output,
        num_output_elements,
        row_stride(x),
        row_stride(output),
        width,
        _BLOCK_SIZE,
        num_warps=8,
    )
    return output


def silu_and_mul_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    grad_x = torch.empty_like(x)
    num_output_elements = grad_output.numel()
    grid = (triton.cdiv(num_output_elements, _BLOCK_SIZE),)
    _backward_kernel[grid](
        x,
        grad_output,
        grad_x,
        num_output_elements,
        x.shape[-1] // 2,
        _BLOCK_SIZE,
        num_warps=8,
    )
    return grad_x
