# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from ctikernels.linear.training import Exchange, FrozenProduct

from .torch import _swiglu as _reference_swiglu


def _swiglu_provider(device: torch.device):
    """The SwiGLU kernels: Triton on CUDA compute capability 8.9+ (in place), the PyTorch
    reference on the CPU."""

    if device.type != "cuda":
        return _reference_swiglu
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("gated_mlp_lora needs compute capability 8.9 or newer")
    from .triton import _swiglu  # Triton is absent from CPU-only installs

    return _swiglu


class _GatedMLPLoRA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gate_up: FrozenProduct,
        down: FrozenProduct,
        scaling: float,
        exchange: Exchange | None,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        up_a: torch.Tensor,
        up_b: torch.Tensor,
        down_a: torch.Tensor,
        down_b: torch.Tensor,
    ) -> torch.Tensor:
        swiglu = _swiglu_provider(x.device)
        x_2d = x.reshape(-1, x.shape[-1])
        # Factors (FP32 masters too) are saved as given and cast to x's dtype at each use.
        factors = (gate_a, gate_b, up_a, up_b, down_a, down_b)
        gate_a, gate_b, up_a, up_b, down_a, down_b = (f.to(x.dtype) for f in factors)
        rank, width = gate_a.shape[0], gate_b.shape[0]
        rows = x_2d.shape[0]
        # The products need not accept zero rows.
        merged = gate_up.forward(x_2d) if rows else x_2d.new_empty(0, 2 * width)
        x_a = F.linear(x_2d, torch.cat((gate_a, up_a)))
        merged[:, :width].addmm_(x_a[:, :rank], gate_b.t(), alpha=scaling)
        merged[:, width:].addmm_(x_a[:, rank:], up_b.t(), alpha=scaling)
        h = swiglu.swiglu(merged)
        h_a = F.linear(h, down_a)
        y = x_2d.new_empty(rows, down_b.shape[0])
        if rows:  # row-parallel down: base and LoRA partials in one (overlapped) sum

            def add_low_rank(part: slice, y_part: torch.Tensor) -> None:
                y_part.addmm_(h_a[part], down_b.t(), alpha=scaling)

            down.forward(h, out=y, exchange=exchange, epilogue=add_low_rank)
        del h

        ctx.save_for_backward(x_2d, merged, x_a, h_a, *factors)
        ctx.products, ctx.scaling, ctx.input_shape = (gate_up, down), scaling, x.shape
        ctx.exchange = exchange
        ctx.consumed = False
        ctx.in_place = swiglu.IN_PLACE
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        if ctx.consumed and ctx.in_place:
            raise RuntimeError(
                "gated_mlp_lora's backward runs once: it overwrites its saved gate_up"
            )
        ctx.consumed = True
        x_2d, merged, x_a, h_a, *factors = ctx.saved_tensors
        gate_a, gate_b, up_a, up_b, down_a, down_b = (f.to(x_2d.dtype) for f in factors)
        gate_up, down = ctx.products
        scaling = ctx.scaling
        rank, width = gate_a.shape[0], gate_b.shape[0]
        grad_y = grad_y.reshape(-1, grad_y.shape[-1]).contiguous()
        rows = grad_y.shape[0]

        def scaled(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            """``scaling * a @ b`` computed into ``out`` (one allocation, the caller's)."""
            return torch.addmm(out, a, b, beta=0, alpha=scaling, out=out)

        grad_y_b = grad_y.mm(down_b)
        grad_down_b = scaled(grad_y.t(), h_a, torch.empty_like(down_b))
        # Each dgrad accumulates onto the low-rank terms written first.
        grad_h = scaled(grad_y_b, down_a, grad_y.new_empty(rows, width))
        if rows:
            down.grad_input(grad_y, out=grad_h, accumulate=True)
        # One kernel: the gate | up gradient in place, and the three factor gradients that
        # reduce over rows (down A, gate B, up B).
        reduced = down_a.new_empty(3, rank * width)
        grad_merged = _swiglu_provider(grad_y.device).swiglu_backward_(
            grad_h, merged, grad_y_b, x_a, reduced, scaling=scaling
        )
        del grad_h
        grad_down_a = reduced[0].view(rank, width)
        grad_gate_b, grad_up_b = reduced[1].view(width, rank), reduced[2].view(width, rank)

        grad_gate, grad_up = grad_merged[:, :width], grad_merged[:, width:]
        grad_x_gate_a, grad_x_up_a = grad_gate.mm(gate_b), grad_up.mm(up_b)
        grad_gate_a = scaled(grad_x_gate_a.t(), x_2d, torch.empty_like(gate_a))
        grad_up_a = scaled(grad_x_up_a.t(), x_2d, torch.empty_like(up_a))

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = scaled(grad_x_gate_a, gate_a, x_2d.new_empty(x_2d.shape))
            grad_x.addmm_(grad_x_up_a, up_a, alpha=scaling)
            if rows:
                gate_up.grad_input(grad_merged, out=grad_x, accumulate=True, exchange=ctx.exchange)
            grad_x = grad_x.reshape(ctx.input_shape)
        return (
            grad_x,
            None,
            None,
            None,
            None,
            grad_gate_a,
            grad_gate_b,
            grad_up_a,
            grad_up_b,
            grad_down_a,
            grad_down_b,
        )


def gated_mlp_lora(
    x: torch.Tensor,
    gate_up: FrozenProduct,
    down: FrozenProduct,
    gate_up_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    down_factors: tuple[torch.Tensor, torch.Tensor],
    *,
    scaling: float,
    exchange: Exchange | None = None,
) -> torch.Tensor:
    """``down(swiglu(gate_up(x)))`` of frozen bases with LoRA on gate, up and down.

    ``x`` is ``(..., H)``. ``gate_up`` projects ``H -> 2I`` (gate rows, then up rows) and
    ``down`` ``I -> H_out``; see :class:`ctikernels.linear.training.FrozenProduct`. ``gate_up_factors`` is
    ``(A_gate, B_gate, A_up, B_up)`` and ``down_factors`` ``(A_down, B_down)``: every A is
    ``(r, in)`` and every B ``(out, r)`` on ``x``'s device, of ``x``'s dtype or FP32 (masters:
    saved as given, cast to ``x``'s dtype at each use; their gradients come back in ``x``'s
    dtype, which autograd casts to the factor's). Each projection
    adds ``scaling * x A^T B^T``; SwiGLU is ``silu(gate) * up``. Returns fresh
    ``(..., H_out)``.

    The same math in any grad mode (reentrant checkpointing safe); with grad mode enabled,
    gradients flow to ``x`` and the six factors, never the bases. The op saves only
    ``x``, the merged gate | up and the ``3r`` low-rank inputs per token; the backward
    recomputes ``SwiGLU`` inside one kernel that writes the gate | up gradient over the saved
    gate | up and forms the down A and gate/up B gradients without storing ``h``,
    so on CUDA it runs once (``retain_graph`` re-runs raise). CUDA (compute capability 8.9+) runs Triton and cuBLAS,
    the CPU an out-of-place PyTorch reference; both are deterministic and host-sync free. Zero rows return an
    empty output and zero factor gradients.

    Tensor parallelism (``ctikernels.linear.training``'s contract, gate_up column-parallel
    and down row-parallel over ``I``): the output and ``x``'s gradient are partial sums the
    caller all-reduces; so are the gradients of the replicated ``A_gate``, ``A_up`` and
    ``B_down``. The other factors hold this rank's slice of ``I`` and get complete gradients.
    Given the group's ``exchange`` (:class:`ctikernels.linear.training.Exchange`), the op
    returns the summed output and the summed ``x`` gradient itself (base and LoRA terms in one
    reduction each); the down projection's backward needs no communication.
    """

    gate_a, gate_b, up_a, up_b = gate_up_factors
    down_a, down_b = down_factors
    return _GatedMLPLoRA.apply(
        x, gate_up, down, scaling, exchange, gate_a, gate_b, up_a, up_b, down_a, down_b
    )


__all__ = ["gated_mlp_lora"]
