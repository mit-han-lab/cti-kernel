# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2023-present Daniel Han-Chen. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-present the Unsloth team. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Unsloth 4c06c1dcc771f3d4f2aecc8d3ab77f2e01aa9107
# (`unsloth/kernels/rms_layernorm.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_MAX_BLOCK_SIZE = 65_536


@triton.jit
def _forward_kernel(
    x,
    weight,
    output,
    inverse_rms,
    x_outer_stride,
    x_group_stride,
    num_groups: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    outer = row // num_groups
    group = row % num_groups
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size

    x_offsets = outer * x_outer_stride + group * x_group_stride + columns
    values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + columns, mask=mask, other=0.0)

    mean_square = tl.sum(values * values, axis=0) / hidden_size
    scale = tl.rsqrt(mean_square + eps)
    normalized = (values * scale).to(weights.dtype)

    tl.store(output + row * hidden_size + columns, normalized * weights, mask=mask)
    tl.store(inverse_rms + row, scale)


@triton.jit
def _backward_kernel(
    x,
    weight,
    inverse_rms,
    grad_output,
    grad_x,
    x_outer_stride,
    x_group_stride,
    grad_outer_stride,
    grad_group_stride,
    grad_hidden_stride,
    num_groups: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    outer = row // num_groups
    group = row % num_groups
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size

    x_offsets = outer * x_outer_stride + group * x_group_stride + columns
    grad_offsets = (
        outer * grad_outer_stride + group * grad_group_stride + columns * grad_hidden_stride
    )
    values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + columns, mask=mask, other=0.0)
    upstream = tl.load(grad_output + grad_offsets, mask=mask, other=0.0)
    scale = tl.load(inverse_rms + row).to(tl.float32)

    normalized = values * scale
    # Match Llama/HF's low-precision multiplication before FP32 reduction.
    weighted_upstream = (upstream * weights).to(tl.float32)
    projection = tl.sum(weighted_upstream * normalized, axis=0)
    grad = scale * (weighted_upstream - normalized * projection / hidden_size)
    tl.store(grad_x + row * hidden_size + columns, grad, mask=mask)


def _grouped_view(
    x: torch.Tensor,
    *,
    name: str,
    require_contiguous_hidden: bool = True,
) -> torch.Tensor:
    if x.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if require_contiguous_hidden and x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous in its hidden dimension")

    hidden_size = x.shape[-1]
    if x.ndim == 1:
        return x.view(1, 1, hidden_size)

    num_groups = x.shape[-2]
    try:
        return x.view(-1, num_groups, hidden_size)
    except RuntimeError as error:
        raise ValueError(
            f"{name} prefix cannot be represented by two strides without a copy"
        ) from error


def _validate(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    if weight.requires_grad:
        raise ValueError("frozen RMSNorm requires a frozen weight")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("frozen RMSNorm requires CUDA tensors")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if x.dtype not in (torch.bfloat16, torch.float32) or weight.dtype != x.dtype:
        raise TypeError("x and weight must use one BF16 or FP32 dtype")
    if x.ndim == 0 or weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape [x.shape[-1]]")
    if x.numel() == 0:
        raise ValueError("frozen RMSNorm does not support empty inputs")
    if weight.stride(0) != 1:
        raise ValueError("weight must be contiguous")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")


def _launch_settings(hidden_size: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > _MAX_BLOCK_SIZE:
        raise ValueError(
            f"hidden size {hidden_size} exceeds the supported maximum {_MAX_BLOCK_SIZE}"
        )
    if block_size >= 32_768:
        return block_size, 32
    if block_size >= 8_192:
        return block_size, 16
    if block_size >= 2_048:
        return block_size, 8
    return block_size, 4


def _forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped_x = _grouped_view(x, name="x")
    num_outer, num_groups, hidden_size = grouped_x.shape
    block_size, num_warps = _launch_settings(hidden_size)
    num_rows = num_outer * num_groups

    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    inverse_rms = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    _forward_kernel[(num_rows,)](
        grouped_x,
        weight,
        output,
        inverse_rms,
        grouped_x.stride(0),
        grouped_x.stride(1),
        num_groups,
        hidden_size,
        eps,
        block_size,
        num_warps=num_warps,
    )
    return output, inverse_rms


def _backward_general_prefix(
    x: torch.Tensor,
    weight: torch.Tensor,
    inverse_rms: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Handle valid upstream strides that cannot collapse to two prefix strides."""

    scale = inverse_rms.view(*x.shape[:-1], 1)
    normalized = x.float() * scale
    weighted_upstream = (grad_output * weight).float()
    projection = (weighted_upstream * normalized).sum(dim=-1, keepdim=True)
    return (scale * (weighted_upstream - normalized * projection / x.shape[-1])).to(x.dtype)


def _backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    inverse_rms: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grouped_x = _grouped_view(x, name="x")
    if grad_output.shape != x.shape:
        raise ValueError("grad_output must have the same shape as x")
    try:
        grouped_grad = _grouped_view(
            grad_output,
            name="grad_output",
            require_contiguous_hidden=False,
        )
    except ValueError:
        return _backward_general_prefix(x, weight, inverse_rms, grad_output)

    num_outer, num_groups, hidden_size = grouped_x.shape
    block_size, num_warps = _launch_settings(hidden_size)
    num_rows = num_outer * num_groups
    grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _backward_kernel[(num_rows,)](
        grouped_x,
        weight,
        inverse_rms,
        grouped_grad,
        grad_x,
        grouped_x.stride(0),
        grouped_x.stride(1),
        grouped_grad.stride(0),
        grouped_grad.stride(1),
        grouped_grad.stride(2),
        num_groups,
        hidden_size,
        block_size,
        num_warps=num_warps,
    )
    return grad_x


class _FrozenRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output, inverse_rms = _forward(x, weight, eps)
        ctx.save_for_backward(x, weight, inverse_rms)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x, weight, inverse_rms = ctx.saved_tensors
        return _backward(x, weight, inverse_rms, grad_output), None, None


def frozen_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    _validate(x, weight, eps)
    if torch.is_grad_enabled() and x.requires_grad:
        return _FrozenRMSNormFunction.apply(x, weight, eps)
    return _forward(x, weight, eps)[0]
