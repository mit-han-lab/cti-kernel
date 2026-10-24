# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from .._reroute import Provider, engine_checks, reroute
from ..vocab_parallel import (
    SHARD_CHUNK_SIZE,
    VocabShardPartials,
    _matrix,
    _shard_finalize,
    _shard_partials,
    _validate_provider,
    _validate_saved_bytes_budget,
    _validate_target_ids,
)


class _TorchProvider:
    name = "torch"

    def __call__(self, logits, targets, output, temperature, ignore_index=-100):
        divisor = temperature[:, None] if isinstance(temperature, torch.Tensor) else temperature
        scaled = logits.float() / divisor
        selected = scaled.gather(1, targets.clamp(0, logits.shape[1] - 1)[:, None]).squeeze(1)
        output.copy_(
            torch.where(targets != ignore_index, selected - torch.logsumexp(scaled, dim=1), 0.0)
        )


class _ChunkedLinearSelectedLogprobsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float | torch.Tensor,
        chunk_size: int,
        provider: Any,
        ignore_index: int,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        selected_logprobs = torch.empty(
            num_tokens,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            hidden_chunk = hidden_states[start:end]
            row_temperature = isinstance(temperature, torch.Tensor)
            temperature_chunk = temperature[start:end] if row_temperature else temperature
            targets = target_ids[start:end]
            output = selected_logprobs[start:end]
            if end - start == 1 and provider.name == "cute":
                # Measured: cuBLAS's M=1 head GEMM is slower than M=2 on RTX 5090,
                # RTX PRO 6000 and GB10 (GB10 BF16 [151936,2048]: 3.5 vs 2.7 ms).
                # Duplicate the row and keep row 0; the kernel takes the row stride.
                logits = F.linear(hidden_chunk.expand(2, -1).contiguous(), weight, bias)
                if row_temperature:
                    temperature_chunk = temperature_chunk.expand(2)
                    targets = targets.expand(2)
                    output = torch.empty(2, device=output.device, dtype=output.dtype)
                else:
                    logits = logits[:1]
            else:
                logits = F.linear(hidden_chunk, weight, bias)
            provider(
                logits,
                targets,
                output,
                temperature_chunk,
                ignore_index,
            )

            if row_temperature and end - start == 1 and provider.name == "cute":
                selected_logprobs[start:end].copy_(output[:1])

        ctx.chunk_size = chunk_size
        ctx.temperature = None if isinstance(temperature, torch.Tensor) else temperature
        ctx.ignore_index = ignore_index
        ctx.save_for_backward(
            hidden_states,
            weight,
            target_ids,
            bias,
            temperature if isinstance(temperature, torch.Tensor) else None,
        )
        return selected_logprobs

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        hidden_states, weight, target_ids, bias, temperature = ctx.saved_tensors
        needs_hidden_grad = ctx.needs_input_grad[0]
        grad_hidden = torch.empty_like(hidden_states) if needs_hidden_grad else None

        for start in range(0, hidden_states.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, hidden_states.shape[0])
            hidden_chunk = hidden_states[start:end]
            divisor = temperature[start:end, None] if temperature is not None else ctx.temperature
            logits = F.linear(hidden_chunk, weight, bias).float() / divisor
            grad_logits = torch.softmax(logits, dim=1).neg_()
            valid = target_ids[start:end] != ctx.ignore_index
            grad_logits.scatter_add_(
                1,
                target_ids[start:end, None].clamp(0, weight.shape[0] - 1),
                valid[:, None].float(),
            )
            grad_logits.mul_(grad_output[start:end, None].float() / divisor)
            grad_logits.masked_fill_(~valid[:, None], 0.0)
            if grad_hidden is not None:
                grad_hidden[start:end] = grad_logits.to(weight.dtype) @ weight
                grad_hidden[start:end].masked_fill_(~valid[:, None], 0.0)

        return (
            grad_hidden,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LinearSelectedTokenLogprobs(Protocol):
    """Selected log-probabilities from a linear vocabulary head.

    Implementations accept hidden rows ``[N, H]``, weight ``[V, H]``, and one
    int64 target per row. They return FP32 ``[N]`` log-probabilities, preserve
    autograd for requested floating inputs, and never return full logits.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
    ) -> torch.Tensor: ...


class ChunkedLinearSelectedTokenLogprobs:
    """Recompute-backed selected log-probabilities with bounded logits storage.

    BF16, FP16 and FP32 heads are accepted; an empty token batch returns an
    empty FP32 tensor. Logits are upcast to FP32 before the log-softmax.
    Temperature is a positive finite scalar or FP32 [N] tensor on the input
    device; tensor temperatures use true FP32 division and may be row-strided.
    Tensor values are checked only in eager CTI_ENGINE_CHECKS calls;
    replayed inputs are the engine's responsibility.

    ``chunk_size`` defaults to 64 for TP1. Independent ``shard_chunk_size``
    defaults to ``SHARD_CHUNK_SIZE`` (1024); ``shard_partials(chunk_size=...)``
    overrides it per call. A shard logits chunk uses at most
    ``shard_chunk_size * V_local * head-dtype bytes``, plus the same for P when
    hidden needs gradients. FP32 logits double the logits term for BF16/FP16
    heads. ``shard_saved_bytes_budget`` selects saved head-dtype P when its
    ``N * V_local * element_size`` bytes fit; None or zero disables automatic
    saving. Explicit per-call ``save_probabilities`` overrides the budget.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 64,
        shard_chunk_size: int = SHARD_CHUNK_SIZE,
        provider: str = "auto",
        fp32_logits: bool = False,
        shard_saved_bytes_budget: int | None = None,
        ignore_index: int = -100,
    ) -> None:
        if type(ignore_index) is not int:
            raise TypeError("ignore_index must be a Python int")
        self._ignore_index = ignore_index
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(shard_chunk_size) is not int or shard_chunk_size <= 0:
            raise ValueError("shard_chunk_size must be a positive integer")
        self._shard_saved_bytes_budget = _validate_saved_bytes_budget(shard_saved_bytes_budget)
        _validate_provider(provider)
        self._provider = provider
        self._chunk_size = chunk_size
        self._shard_chunk_size = shard_chunk_size
        if type(fp32_logits) is not bool:
            raise TypeError("fp32_logits must be a bool")
        self._fp32_logits = fp32_logits
        self._providers = {}
        self.provider = Provider("torch")

    def _get_provider(self, device: torch.device):
        cached = self._providers.get(device)
        if cached is None:
            from .cute import Provider as CuteProvider
            from .cute import unavailable_reason

            reason = unavailable_reason()
            if reason is None:
                capability = torch.cuda.get_device_capability(device)
                if capability[0] < 8:
                    reason = f"CuTe requires capability >= 8.0; got {capability[0]}.{capability[1]}"
            self.provider = (
                reroute("selected_logprobs", "torch", reason) if reason else Provider("cute")
            )
            provider = _TorchProvider() if reason else CuteProvider(device)
            self._providers[device] = (provider, reason)
        else:
            provider, reason = cached
            self.provider = (
                reroute("selected_logprobs", "torch", reason) if reason else Provider("cute")
            )
        return provider

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def shard_chunk_size(self) -> int:
        return self._shard_chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_states, weight, target_ids, bias=bias)
        if isinstance(temperature, torch.Tensor):
            _matrix(temperature, (hidden_states.shape[0],), hidden_states.device, "temperature")
            capturing = False
            if engine_checks() and temperature.is_cuda:
                with torch.cuda.device(temperature.device):
                    capturing = torch.cuda.is_current_stream_capturing()
            if engine_checks() and not capturing:
                values = temperature.detach().cpu()
                if not bool((torch.isfinite(values) & (values > 0)).all()):
                    raise ValueError("temperature must be finite and positive")
        else:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                raise TypeError("temperature must be a real number or an FP32 tensor [N]")
            temperature = float(temperature)
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
        _validate_target_ids(target_ids, self._ignore_index, weight.shape[0])
        return _ChunkedLinearSelectedLogprobsFunction.apply(
            hidden_states,
            weight,
            target_ids,
            bias,
            temperature,
            self._chunk_size,
            self._get_provider(hidden_states.device),
            self._ignore_index,
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
        temperature: float | torch.Tensor = 1.0,
        provider: str | None = None,
        chunk_size: int | None = None,
        fp32_logits: bool | None = None,
        save_probabilities: bool | None = None,
        vocab_block_size: int | None = None,
    ) -> VocabShardPartials:
        """Return FP32 shard stats without invoking a collective.

        Temperature is a positive finite scalar or FP32 CUDA [N]. Scalars use
        the FP32 reciprocal multiply of PyTorch's Python-scalar division;
        tensor temperatures use IEEE FP32 division per row. Tensor values are
        checked only on eager calls under CTI_ENGINE_CHECKS.
        """
        return _shard_partials(
            hidden,
            weight_shard,
            target_ids,
            vocab_start=vocab_start,
            vocab_size=vocab_size,
            ignore_index=self._ignore_index,
            bias=bias,
            temperature=temperature,
            chunk_size=self._shard_chunk_size if chunk_size is None else chunk_size,
            fp32_logits=self._fp32_logits if fp32_logits is None else fp32_logits,
            save_probabilities=save_probabilities,
            shard_saved_bytes_budget=self._shard_saved_bytes_budget,
            vocab_block_size=vocab_block_size,
            provider=self._provider if provider is None else provider,
            kind="logprob",
        )

    def shard_finalize(self, part: VocabShardPartials, merged: torch.Tensor) -> torch.Tensor:
        """Consume a shard forward; backward returns this shard's partial dX."""
        return _shard_finalize(part, merged, kind="logprob")

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        bias: torch.Tensor | None,
    ) -> None:
        if hidden_states.ndim != 2 or weight.ndim != 2:
            raise ValueError("hidden_states and weight must be matrices")
        if target_ids.ndim != 1 or target_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("target_ids must contain one value per hidden row")
        if not 1 <= weight.shape[0] <= 2**31 - 1:
            raise ValueError("vocabulary size must be between 1 and int32 max")
        if hidden_states.shape[1] != weight.shape[1]:
            raise ValueError("hidden_states must match the weight hidden size")
        if target_ids.dtype != torch.int64:
            raise TypeError("target_ids must use int64")
        if hidden_states.dtype != weight.dtype or hidden_states.dtype not in {
            torch.bfloat16,
            torch.float16,
            torch.float32,
        }:
            raise TypeError("hidden_states and weight must share BF16, FP16 or FP32 dtype")
        if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype):
            raise ValueError("bias must match the vocabulary and floating dtype")
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("selected-token log-probabilities require a frozen vocabulary head")
        if not hidden_states.is_cuda or not weight.is_cuda or not target_ids.is_cuda:
            raise ValueError("hidden_states, weight, and target_ids must reside on CUDA")
        if not (
            hidden_states.device == weight.device == target_ids.device
            and (bias is None or bias.device == hidden_states.device)
        ):
            raise ValueError("all selected-logprob inputs must share one CUDA device")


__all__ = ["ChunkedLinearSelectedTokenLogprobs", "LinearSelectedTokenLogprobs"]
