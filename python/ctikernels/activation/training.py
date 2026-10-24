# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ._silu_and_mul import silu_and_mul_backward as _fused_silu_and_mul_backward
from ._silu_and_mul import silu_and_mul_forward as _fused_silu_and_mul_forward


def _can_fuse_silu_and_mul(x: torch.Tensor) -> bool:
    return (
        x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and x.ndim > 0
        and x.numel() > 0
        and x.shape[-1] > 0
        and x.shape[-1] % 2 == 0
    )


def _silu_and_mul_forward(x: torch.Tensor) -> torch.Tensor:
    if _can_fuse_silu_and_mul(x):
        return _fused_silu_and_mul_forward(x)

    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


def _silu_and_mul_backward(x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    if _can_fuse_silu_and_mul(x) and grad_output.is_contiguous():
        return _fused_silu_and_mul_backward(x, grad_output)

    gate, up = x.chunk(2, dim=-1)
    grad_gate = torch.ops.aten.silu_backward.default(grad_output * up, gate)
    grad_up = grad_output * F.silu(gate)
    return torch.cat((grad_gate, grad_up), dim=-1)


class _SiluAndMulFunction(torch.autograd.Function):
    """First-order SwiGLU that recomputes SiLU during backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return _silu_and_mul_forward(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return _silu_and_mul_backward(x, grad_output)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable SwiGLU without mutating the merged input.

    ``x`` is a floating strided tensor shaped ``[..., 2 * width]``. The result
    has independent ``[..., width]`` storage. The first-order backward saves the
    merged input once and recomputes SiLU. Execution follows the caller's current
    device stream; higher-order gradients are not supported.
    """

    if torch.is_grad_enabled() and x.requires_grad:
        return _SiluAndMulFunction.apply(x)
    return _silu_and_mul_forward(x)


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Apply differentiable exact GeGLU without mutating the merged input.

    ``x`` is a floating strided tensor shaped ``[..., 2 * width]``. The result
    has independent ``[..., width]`` storage. Standard PyTorch autograd owns the
    backward, and execution follows the caller's current device stream.
    """

    gate, up = x.chunk(2, dim=-1)
    return F.gelu(gate) * up


__all__ = ["gelu_and_mul", "silu_and_mul"]
