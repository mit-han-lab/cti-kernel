# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ._lora_segmented import (
    LoRAEpilogue,
    lora_segmented_delta_,
    lora_segmented_projection,
    lora_shrink,
)


class LoRAInferencePlan(Protocol):
    """Precomputed, instance-owned inference metadata for one projection."""

    @property
    def provider(self) -> str: ...

    def can_project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        factors: tuple[torch.Tensor, ...],
        *,
        scaling: float,
    ) -> bool: ...

    def project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor: ...

    def warm_up(self, max_m: int, weight: torch.Tensor | None = None) -> list[dict]: ...


def try_prepare_lora_inference_plan(
    factors: tuple[torch.Tensor, ...],
    *,
    input_size: int,
    output_sizes: tuple[int, ...],
    scaling: float,
    deterministic: bool = True,
    row_parallel: bool = False,
) -> LoRAInferencePlan | None:
    """Prepare an available inference specialization or retain the generic path.

    The default plan is bitwise repeatable on every route. ``deterministic=False``
    also admits the frozen Triton provider (FP-atomic split-K) on the rows where
    measurements found it faster. ``row_parallel`` prepares a K-shard projection whose
    output is this rank's partial (see :func:`lora_segmented_delta_`)."""

    if not factors or factors[0].device.type != "cuda":
        return None

    # Keep CUDA implementations lazy: neither Triton nor cutlass is required
    # to import this API in CPU-only CI. The segmented delta is our Triton
    # kernel pair on cc >= 8.9; the frozen Triton provider keeps
    # its original support range.
    from ._lora_inference import try_prepare_small_batch_lora_plan

    return try_prepare_small_batch_lora_plan(
        factors,
        input_size=input_size,
        output_sizes=output_sizes,
        scaling=scaling,
        deterministic=deterministic,
        row_parallel=row_parallel,
    )


def _forward_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scaling: float,
    lora_factors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    x_2d = x.reshape(-1, x.shape[-1])
    output = F.linear(x_2d, weight, bias)
    low_rank_inputs = []
    output_offset = 0

    for index in range(0, len(lora_factors), 2):
        lora_a = lora_factors[index]
        lora_b = lora_factors[index + 1]
        low_rank_input = F.linear(x_2d, lora_a)
        low_rank_inputs.append(low_rank_input)
        output_slice = output.narrow(1, output_offset, lora_b.shape[0])
        output_slice.addmm_(low_rank_input, lora_b.t(), alpha=scaling)
        output_offset += lora_b.shape[0]
    if output_offset != output.shape[1]:
        raise ValueError("LoRA output slices must cover the base projection exactly")

    return (
        output.reshape(*x.shape[:-1], weight.shape[0]),
        x_2d,
        tuple(low_rank_inputs),
    )


def _forward_projection_inference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scaling: float,
    lora_factors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Accumulate each low-rank slice directly into a fresh base output."""

    x_2d = x.reshape(-1, x.shape[-1])
    output = F.linear(x_2d, weight, bias)
    output_offset = 0
    for index in range(0, len(lora_factors), 2):
        lora_a = lora_factors[index]
        lora_b = lora_factors[index + 1]
        low_rank_input = F.linear(x_2d, lora_a)
        output_slice = output.narrow(1, output_offset, lora_b.shape[0])
        output_slice.addmm_(low_rank_input, lora_b.t(), alpha=scaling)
        output_offset += lora_b.shape[0]
    if output_offset != output.shape[1]:
        raise ValueError("LoRA output slices must cover the base projection exactly")
    return output.reshape(*x.shape[:-1], weight.shape[0])


class _LoRAProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        scaling: float,
        *lora_factors: torch.Tensor,
    ) -> torch.Tensor:
        output, x_2d, low_rank_inputs = _forward_projection(
            x,
            weight,
            bias,
            scaling,
            lora_factors,
        )
        ctx.save_for_backward(
            x_2d,
            weight,
            *lora_factors,
            *low_rank_inputs,
        )
        ctx.input_shape = x.shape
        ctx.num_lora_factors = len(lora_factors)
        ctx.scaling = scaling
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        x_2d, weight, *saved = ctx.saved_tensors
        num_lora_factors = ctx.num_lora_factors
        lora_factors = saved[:num_lora_factors]
        low_rank_inputs = saved[num_lora_factors:]
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        needs_grad = ctx.needs_input_grad

        grad_x = grad_output_2d.mm(weight) if needs_grad[0] else None
        grad_weight = grad_output_2d.t().mm(x_2d) if needs_grad[1] else None
        grad_bias = grad_output_2d.sum(dim=0) if needs_grad[2] else None

        grad_lora_factors: list[torch.Tensor | None] = []
        output_offset = 0
        for pair_index, index in enumerate(range(0, num_lora_factors, 2)):
            lora_a = lora_factors[index]
            lora_b = lora_factors[index + 1]
            output_size = lora_b.shape[0]
            grad_slice = grad_output_2d.narrow(1, output_offset, output_size)
            needs_grad_a = needs_grad[4 + index]
            needs_grad_b = needs_grad[5 + index]

            if needs_grad_a or grad_x is not None:
                grad_low_rank = grad_slice.mm(lora_b)
                grad_a = (
                    torch.addmm(
                        torch.empty_like(lora_a),
                        grad_low_rank.t(),
                        x_2d,
                        beta=0,
                        alpha=ctx.scaling,
                    )
                    if needs_grad_a
                    else None
                )
                if grad_x is not None:
                    grad_x.addmm_(grad_low_rank, lora_a, alpha=ctx.scaling)
            else:
                grad_a = None

            grad_b = None
            if needs_grad_b:
                grad_b = torch.addmm(
                    torch.empty_like(lora_b),
                    grad_slice.t(),
                    low_rank_inputs[pair_index],
                    beta=0,
                    alpha=ctx.scaling,
                )

            grad_lora_factors.extend((grad_a, grad_b))
            output_offset += output_size

        if grad_x is not None:
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, grad_weight, grad_bias, None, *grad_lora_factors


def lora_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_factors: tuple[torch.Tensor, ...],
    *,
    scaling: float,
    inference_plan: LoRAInferencePlan | None = None,
) -> torch.Tensor:
    """Apply a dense projection and contiguous per-slice LoRA updates.

    ``x`` is ``(..., input_size)``, ``weight`` is
    ``(output_size, input_size)``, and an optional ``bias`` is ``(output_size,)``.
    ``lora_factors`` is ``(A0, B0, A1, B1, ...)``. Every A has shape
    ``(rank, input_size)``; each B has shape ``(slice_size, rank)``; and the B
    row counts cover ``output_size`` in order. All tensors are real floating-point
    values on one device with one dtype. The linear/model owners validate this
    topology when they are composed, keeping the hot path free of tensor scans.

    The operation returns fresh storage, preserves every input, uses the current
    stream without host synchronization, supports first-order autograd, and is
    safe to capture while input addresses and the factor tuple's topology remain
    stable; factor values may be updated in place between replays. Higher-order
    gradients are not supported. Distributed completion and non-dense weight
    representations remain the owning linear layer's responsibility.
    """
    if not lora_factors or len(lora_factors) % 2 != 0:
        raise ValueError("LoRA factors must contain one or more A/B pairs")
    if (
        not torch.is_grad_enabled()
        and inference_plan is not None
        and inference_plan.can_project(
            x,
            weight,
            bias,
            lora_factors,
            scaling=scaling,
        )
    ):
        return inference_plan.project(x, weight, bias)
    if not torch.is_grad_enabled():
        return _forward_projection_inference(x, weight, bias, scaling, lora_factors)
    return _LoRAProjectionFunction.apply(x, weight, bias, scaling, *lora_factors)


__all__ = [
    "LoRAEpilogue",
    "LoRAInferencePlan",
    "lora_projection",
    "lora_segmented_delta_",
    "lora_segmented_projection",
    "lora_shrink",
    "try_prepare_lora_inference_plan",
]
