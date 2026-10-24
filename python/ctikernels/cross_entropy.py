# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Copyright 2024 LinkedIn Corporation. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Liger Kernel c4b16d43f9d8f69068e6a15bd879dfc6a63b2449
# (`src/liger_kernel/ops/fused_linear_cross_entropy.py`, `src/liger_kernel/ops/cross_entropy.py`);
# see THIRD_PARTY_NOTICES.

from __future__ import annotations

from typing import Any, Protocol

import torch

from .vocab_parallel import (
    SHARD_CHUNK_SIZE,
    VocabShardPartials,
    _shard_finalize,
    _shard_partials,
    _validate_provider,
    _validate_saved_bytes_budget,
)

_MAX_BLOCK_SIZE = 32_768


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _ChunkedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None,
        cross_entropy_kernel: Any,
        chunk_size: int,
        ignore_index: int = -100,
        reduction: str = "sum",
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        vocab_size = weight.shape[0]
        block_size = min(_MAX_BLOCK_SIZE, _next_power_of_two(vocab_size))
        needs_hidden_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        needs_bias_grad = bias is not None and ctx.needs_input_grad[3]
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        computes_gradients = reduction != "none" and (
            needs_hidden_grad or needs_weight_grad or needs_bias_grad
        )
        if reduction == "none":
            needs_hidden_grad = needs_weight_grad = needs_bias_grad = False
        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if needs_weight_grad else None
        grad_bias = torch.zeros_like(bias, dtype=torch.float32) if needs_bias_grad else None
        # The provider kernel returns early on ignore_index rows without storing
        # a loss, so those rows must start at zero to contribute nothing.
        loss_by_token = torch.zeros(
            num_tokens,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden_chunk = hidden_states[start:end]
            if bias is None:
                logits_chunk = hidden_chunk @ weight.t()
            else:
                logits_chunk = torch.addmm(bias, hidden_chunk, weight.t())
            logits_chunk = logits_chunk.contiguous()
            target_chunk = target_ids[start:end]
            loss_chunk = loss_by_token[start:end]

            cross_entropy_kernel[(end - start,)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                Y_stride=target_chunk.stride(-1),
                weight_ptr=None,
                loss_ptr=loss_chunk,
                z_loss_ptr=None,
                loss_stride=loss_chunk.stride(-1),
                token_accuracy_ptr=None,
                token_accuracy_stride=0,
                predicted_tokens_ptr=None,
                predicted_tokens_stride=0,
                n_cols=vocab_size,
                n_non_ignore=num_tokens,
                sum_non_ignore_weight=num_tokens,
                weight_sum=0.0,
                ignore_index=ignore_index,
                lse_square_scale=0.0,
                label_smoothing=0.0,
                reduction="sum",
                softcap=None,
                RETURN_Z_LOSS=False,
                RETURN_TOKEN_ACCURACY=False,
                RETURN_PREDICTED_TOKENS=False,
                HAS_WEIGHT=False,
                HAS_SOFTCAPPING=False,
                HAS_GRADIENTS=computes_gradients,
                BLOCK_SIZE=block_size,
                num_warps=32,
            )

            if grad_hidden is not None:
                grad_hidden[start:end] = logits_chunk @ weight
            if grad_weight is not None and hidden_states.is_cuda:
                torch.addmm(
                    grad_weight,
                    logits_chunk.t(),
                    hidden_chunk,
                    out=grad_weight,
                    out_dtype=torch.float32,
                )
            elif grad_weight is not None:
                grad_weight.addmm_(logits_chunk.float().t(), hidden_chunk.float())
            if grad_bias is not None:
                grad_bias.add_(logits_chunk.sum(dim=0, dtype=torch.float32))

        if reduction == "none":
            ctx.save_for_backward(hidden_states, weight, target_ids, bias)
            return loss_by_token
        ctx.save_for_backward(grad_hidden, grad_weight, grad_bias)
        if reduction == "mean":
            ctx.count = (target_ids != ignore_index).sum()
            return loss_by_token.sum() / ctx.count
        return loss_by_token.sum()

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple:
        if ctx.reduction == "none":
            hidden, weight, targets, bias = ctx.saved_tensors
            dx = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
            dw = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
            db = torch.zeros_like(bias, dtype=torch.float32) if ctx.needs_input_grad[3] else None
            for begin in range(0, hidden.shape[0], ctx.chunk_size):
                end = min(begin + ctx.chunk_size, hidden.shape[0])
                rows = hidden[begin:end]
                logits = rows @ weight.t() if bias is None else torch.addmm(bias, rows, weight.t())
                dz = logits.float().softmax(dim=1)
                y = targets[begin:end]
                valid = y != ctx.ignore_index
                dz.scatter_add_(
                    1, y.clamp(0, weight.shape[0] - 1)[:, None], -valid.float()[:, None]
                )
                dz.mul_(grad_output[begin:end, None].float())
                dz.masked_fill_(~valid[:, None], 0.0)
                dz = dz.to(weight.dtype)
                if dx is not None:
                    dx[begin:end] = dz @ weight
                if dw is not None:
                    if hidden.is_cuda and hidden.dtype in {torch.bfloat16, torch.float16}:
                        torch.addmm(dw, dz.t(), rows, out=dw, out_dtype=torch.float32)
                    else:
                        dw.addmm_(dz.float().t(), rows.float())
                if db is not None:
                    db.add_(dz.sum(dim=0, dtype=torch.float32))
            return dx, dw, None, db, None, None, None, None
        grad_hidden, grad_weight, grad_bias = ctx.saved_tensors
        if ctx.reduction == "mean":
            grad_output = torch.where(ctx.count > 0, grad_output / ctx.count, 0.0)
            if grad_hidden is not None:
                grad_hidden = (grad_hidden.float() * grad_output).to(grad_hidden.dtype)
        elif grad_hidden is not None:
            grad_hidden = grad_hidden * grad_output.to(grad_hidden.dtype)
        if grad_weight is not None:
            grad_weight = grad_weight * grad_output.to(grad_weight.dtype)
        if grad_bias is not None:
            grad_bias = grad_bias * grad_output.to(grad_bias.dtype)
        return grad_hidden, grad_weight, None, grad_bias, None, None, None, None


class LinearCrossEntropyLoss(Protocol):
    """Cross entropy over hidden states and a linear head.

    ``target_ids`` must be validated against the global vocabulary before CUDA
    dispatch. Implementations return an FP32 scalar or per-row loss and
    support autograd for requested inputs; logits and auxiliary outputs belong to
    separate contracts. Implementations must not add a device-content scan to this
    hot-path contract.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class FusedLinearCrossEntropyLoss:
    """Eager fused-linear cross entropy using Liger's low-level CE kernel.

    ``hidden_states`` is ``[num_tokens, hidden_size]``; ``weight`` is
    ``[vocab_size, hidden_size]``; optional ``bias`` is ``[vocab_size]``; and
    ``target_ids`` is an int64 ``[num_tokens]`` vector prevalidated against the
    global vocabulary; entries equal to ``ignore_index`` (default -100) contribute
    zero loss and gradient. Floating inputs use one common BF16, FP16 or FP32 CUDA dtype.

    The operation runs on the current CUDA stream. ``reduction="sum"`` (default)
    returns the FP32 sum; ``"mean"`` divides by the device count of valid rows;
    ``"none"`` returns FP32 [N]. Zero valid rows give NaN for mean and zero
    gradients, matching PyTorch. Empty batches launch no GEMM or Triton kernel.
    For none, backward recomputes logits and applies per-row upstream in FP32
    before rounding logit gradients to the head dtype, in TP1 and shard modes.
    It is out-of-place and does not mutate or alias caller tensors. Autograd covers
    hidden states, weight, and bias; head-only training is supported when the
    backbone output does not require gradients. The operation chunks token rows of
    the vocabulary projection instead of materializing full logits. Direct CUDA
    graph capture is not part of this operation's contract.

    ``chunk_size`` defaults to 256 for TP1. Independent ``shard_chunk_size``
    defaults to ``SHARD_CHUNK_SIZE`` (1024); ``shard_partials(chunk_size=...)``
    overrides it per call. A shard logits chunk uses at most
    ``shard_chunk_size * V_local * head-dtype bytes``, plus the same for P with
    a frozen head when hidden needs gradients. FP32 logits double the logits
    term for BF16 heads. ``shard_saved_bytes_budget`` selects saved mode when
    the retained matrix fits; None or zero disables automatic saving. Trainable
    heads save logits; frozen heads save head-dtype P. Explicit per-call
    ``save_probabilities`` overrides the budget. CE ``none`` saves logits even
    with a frozen head. Shard capture needs one eager warm-up per (dtype, mode
    flags); new shard start, V, temperature, N or strides do not compile.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 256,
        shard_chunk_size: int = SHARD_CHUNK_SIZE,
        provider: str | None = None,
        fp32_logits: bool = False,
        shard_saved_bytes_budget: int | None = None,
        ignore_index: int = -100,
        reduction: str = "sum",
    ) -> None:
        if type(ignore_index) is not int:
            raise TypeError("ignore_index must be a Python int")
        if not isinstance(reduction, str) or reduction not in {"sum", "mean", "none"}:
            raise ValueError("reduction must be sum, mean or none")
        self._ignore_index = ignore_index
        self._reduction = reduction
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if isinstance(shard_chunk_size, bool) or not isinstance(shard_chunk_size, int):
            raise TypeError("shard_chunk_size must be an integer")
        if shard_chunk_size <= 0:
            raise ValueError("shard_chunk_size must be positive")

        self._shard_saved_bytes_budget = _validate_saved_bytes_budget(shard_saved_bytes_budget)
        _validate_provider("auto" if provider is None else provider)
        self._provider = "auto" if provider is None else provider
        try:
            from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
        except (ImportError, OSError) as error:
            if provider is None:
                raise RuntimeError(
                    "FusedLinearCrossEntropyLoss requires `liger-kernel==0.8.0`; "
                    "use an explicit shard provider for shard-only operation"
                ) from error
            liger_cross_entropy_kernel = None

        self._cross_entropy_kernel = liger_cross_entropy_kernel
        self._chunk_size = chunk_size
        self._shard_chunk_size = shard_chunk_size
        if type(fp32_logits) is not bool:
            raise TypeError("fp32_logits must be a bool")
        self._fp32_logits = fp32_logits

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    @property
    def reduction(self) -> str:
        return self._reduction

    @property
    def chunk_size(self) -> int:
        """Return the immutable token-chunk policy bound at construction."""
        return self._chunk_size

    @property
    def shard_chunk_size(self) -> int:
        """Return the immutable shard row-chunk policy bound at construction."""
        return self._shard_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_states, weight, target_ids, bias=bias)

        if self._cross_entropy_kernel is None and hidden_states.shape[0]:
            raise RuntimeError("TP1 forward requires `liger-kernel==0.8.0`")
        return _ChunkedLinearCrossEntropyFunction.apply(
            hidden_states,
            weight,
            target_ids,
            bias,
            self._cross_entropy_kernel,
            self._chunk_size,
            self._ignore_index,
            self._reduction,
        )

    def shard_partials(
        self,
        hidden: torch.Tensor,
        weight_shard: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        vocab_start: int,
        vocab_size: int,
        bias: torch.Tensor | None = None,
        temperature: float = 1.0,
        provider: str | None = None,
        chunk_size: int | None = None,
        fp32_logits: bool | None = None,
        save_probabilities: bool | None = None,
        vocab_block_size: int | None = None,
    ) -> VocabShardPartials:
        """Project one vocabulary shard; the engine merges the returned FP32 stats."""
        return _shard_partials(
            hidden,
            weight_shard,
            target_ids,
            vocab_start=vocab_start,
            vocab_size=vocab_size,
            bias=bias,
            temperature=temperature,
            chunk_size=self._shard_chunk_size if chunk_size is None else chunk_size,
            fp32_logits=self._fp32_logits if fp32_logits is None else fp32_logits,
            save_probabilities=save_probabilities,
            shard_saved_bytes_budget=self._shard_saved_bytes_budget,
            vocab_block_size=vocab_block_size,
            provider=self._provider if provider is None else provider,
            kind="ce",
            ignore_index=self._ignore_index,
            reduction=self._reduction,
        )

    def shard_finalize(self, part: VocabShardPartials, merged: torch.Tensor) -> torch.Tensor:
        """Consume one shard forward using its reduction, returning local gradients."""
        return _shard_finalize(part, merged, kind="ce")

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None,
    ) -> None:
        # The output head validates target values while they are still on CPU.
        # Re-scanning CUDA contents here would add a synchronization point to
        # every training step; this layer validates metadata only.
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [vocab_size, hidden_size]")
        if target_ids.ndim != 1:
            raise ValueError("target_ids must have shape [num_tokens]")
        if weight.shape[0] < 1:
            raise ValueError("weight must have at least one vocabulary row")
        if hidden_states.shape[0] != target_ids.shape[0]:
            raise ValueError("hidden_states and target_ids must have the same token count")
        if hidden_states.shape[1] != weight.shape[1]:
            raise ValueError("hidden_states and weight must have the same hidden size")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("bias must have shape [vocab_size]")
        if target_ids.dtype != torch.int64:
            raise TypeError("target_ids must use int64")
        if not hidden_states.is_floating_point() or not weight.is_floating_point():
            raise TypeError("hidden_states and weight must use floating-point dtypes")
        if hidden_states.dtype != weight.dtype:
            raise TypeError("hidden_states and weight must use the same dtype")
        if hidden_states.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise TypeError("hidden_states and weight must use bfloat16, float16 or float32")
        if bias is not None and bias.dtype != hidden_states.dtype:
            raise TypeError("bias must use the same dtype as hidden_states and weight")
        if not hidden_states.is_cuda or not weight.is_cuda or not target_ids.is_cuda:
            raise ValueError("hidden_states, weight, and target_ids must reside on CUDA")
        if (
            weight.device != hidden_states.device
            or target_ids.device != hidden_states.device
            or (bias is not None and bias.device != hidden_states.device)
        ):
            raise ValueError("hidden_states, weight, target_ids, and bias must share a CUDA device")


__all__ = ["FusedLinearCrossEntropyLoss", "LinearCrossEntropyLoss"]
