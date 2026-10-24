# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from vLLM 0decac0d96c42b49572498019f0a0e3600f50398
# (`vllm/lora/ops/triton_ops/lora_shrink_op.py`, `vllm/lora/ops/triton_ops/lora_expand_op.py`,
# `vllm/lora/ops/triton_ops/kernel_utils.py`, `vllm/lora/ops/triton_ops/utils.py`); see
# THIRD_PARTY_NOTICES.

"""Based on Chen et al., "Punica: Multi-Tenant LoRA Serving," MLSys 2024.

https://arxiv.org/abs/2310.18547
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ._lora_policy import (
    SmallBatchLoRAPolicy,
    policy_for_hardware,
    prefer_small_batch_plan,
    select_provider,
    supports_projection,
)
from .inference import Linear

_BLOCK_M = 16
_BLOCK_RANK = 16
_SHRINK_BLOCK_K = 64
_EXPAND_BLOCK_N = 64
_EXPAND_BLOCK_K = 32
_SPLIT_K = 16
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@triton.jit
def _shrink_kernel(
    x_ptr,
    a_ptrs,
    low_rank_ptr,
    num_tokens: tl.constexpr,
    input_size: tl.constexpr,
    rank: tl.constexpr,
    scaling: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_x_input: tl.constexpr,
    stride_low_slice: tl.constexpr,
    stride_low_token: tl.constexpr,
    stride_low_rank: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    split_id = program_id % SPLIT_K
    tile_id = program_id // SPLIT_K
    num_rank_tiles = tl.cdiv(rank, BLOCK_RANK)
    token_tile_id = tile_id // num_rank_tiles
    rank_tile_id = tile_id % num_rank_tiles
    slice_id = tl.program_id(1)

    token_offsets = token_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    rank_offsets = rank_tile_id * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    a_ptr = tl.load(a_ptrs + slice_id).to(tl.pointer_type(x_ptr.dtype.element_ty))
    accumulator = tl.zeros((BLOCK_M, BLOCK_RANK), dtype=tl.float32)

    for iteration in range(0, tl.cdiv(input_size, BLOCK_K * SPLIT_K)):
        input_offsets = split_id * BLOCK_K + iteration * BLOCK_K * SPLIT_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr
            + token_offsets[:, None] * stride_x_token
            + input_offsets[None, :] * stride_x_input,
            mask=(token_offsets[:, None] < num_tokens) & (input_offsets[None, :] < input_size),
            other=0.0,
        )
        a = tl.load(
            a_ptr + rank_offsets[None, :] * input_size + input_offsets[:, None],
            mask=(rank_offsets[None, :] < rank) & (input_offsets[:, None] < input_size),
            other=0.0,
        )
        accumulator += tl.dot(x, a)

    output_ptr = (
        low_rank_ptr
        + slice_id * stride_low_slice
        + token_offsets[:, None] * stride_low_token
        + rank_offsets[None, :] * stride_low_rank
    )
    output_mask = (token_offsets[:, None] < num_tokens) & (rank_offsets[None, :] < rank)
    tl.atomic_add(
        output_ptr,
        accumulator * scaling,
        mask=output_mask,
        sem="relaxed",
    )


@triton.jit
def _expand_kernel(
    low_rank_ptr,
    b_ptrs,
    output_ptr,
    output_offsets,
    output_sizes,
    num_tokens: tl.constexpr,
    rank: tl.constexpr,
    max_output_size: tl.constexpr,
    stride_low_slice: tl.constexpr,
    stride_low_token: tl.constexpr,
    stride_low_rank: tl.constexpr,
    stride_output_token: tl.constexpr,
    stride_output_feature: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_output_tiles = tl.cdiv(max_output_size, BLOCK_N)
    token_tile_id = program_id // num_output_tiles
    output_tile_id = program_id % num_output_tiles
    slice_id = tl.program_id(1)

    output_size = tl.load(output_sizes + slice_id)
    output_offset = tl.load(output_offsets + slice_id)
    b_ptr = tl.load(b_ptrs + slice_id).to(tl.pointer_type(output_ptr.dtype.element_ty))
    token_offsets = token_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    output_feature_offsets = output_tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for rank_start in range(0, rank, BLOCK_K):
        rank_offsets = rank_start + tl.arange(0, BLOCK_K)
        low_rank = tl.load(
            low_rank_ptr
            + slice_id * stride_low_slice
            + token_offsets[:, None] * stride_low_token
            + rank_offsets[None, :] * stride_low_rank,
            mask=(token_offsets[:, None] < num_tokens) & (rank_offsets[None, :] < rank),
            other=0.0,
        ).to(output_ptr.dtype.element_ty)
        b = tl.load(
            b_ptr + output_feature_offsets[None, :] * rank + rank_offsets[:, None],
            mask=(output_feature_offsets[None, :] < output_size) & (rank_offsets[:, None] < rank),
            other=0.0,
        )
        accumulator += tl.dot(low_rank, b)

    destination = (
        output_ptr
        + token_offsets[:, None] * stride_output_token
        + (output_offset + output_feature_offsets[None, :]) * stride_output_feature
    )
    output_mask = (token_offsets[:, None] < num_tokens) & (
        output_feature_offsets[None, :] < output_size
    )
    delta = accumulator.to(output_ptr.dtype.element_ty)
    base = tl.load(destination, mask=output_mask)
    tl.store(destination, base + delta, mask=output_mask)


class SmallBatchLoRAInferencePlan:
    """Instance-owned segmented-delta plan with a frozen Triton small-batch option.

    ``deterministic=True`` (the default) uses only the segmented delta (our Triton kernel pair), whose split-K reduction runs
    in a fixed order: repeated calls and graph replays of one M are bitwise
    identical. ``deterministic=False`` additionally admits the frozen Triton
    provider on its measured table rows; its split-K FP atomics are not bitwise
    repeatable. Neither mode promises batch invariance.

    Call :meth:`warm_up` (or each M eagerly) before CUDA graph capture. The owned
    Linear compiles measured CuTe base routes and the delta caches its metadata
    during that call. An unwarmed CuTe route retains Linear's capture-miss error:
    F.linear is not numerically equivalent to the eager CuTe result, so silently
    changing providers would be unsafe.

    The plan binds factor storage addresses. In-place factor updates are observed
    eagerly and by graph replays. Replacing a factor's storage (``param.data =
    new``) is detected on the next eager call, which rebinds to the new storage;
    the same change raises during capture, and graphs captured before it keep
    reading the old storage until recaptured.
    """

    def __init__(
        self,
        factors: tuple[torch.Tensor, ...],
        *,
        input_size: int,
        output_sizes: tuple[int, ...],
        scaling: float,
        policy: SmallBatchLoRAPolicy | None = None,
        deterministic: bool = True,
        row_parallel: bool = False,
    ) -> None:
        self._factors = factors
        self._row_parallel = bool(row_parallel)
        # The frozen Triton provider rounds its low-rank tile; row-parallel never does.
        self._deterministic = bool(deterministic) or self._row_parallel
        self._input_size = input_size
        self._output_sizes_tuple = output_sizes
        self._output_size = sum(output_sizes)
        self._rank = factors[0].shape[0]
        self._scaling = float(scaling)
        self._device = factors[0].device
        self._dtype = factors[0].dtype
        self._policy = policy
        self._linear = Linear()
        self._base_provider = "torch"
        self._provider = "uninitialized"
        self._bind_factors()
        offsets = []
        offset = 0
        for output_size in output_sizes:
            offsets.append(offset)
            offset += output_size
        self._output_offsets = torch.tensor(
            offsets,
            dtype=torch.int32,
            device=self._device,
        )
        self._output_sizes = torch.tensor(
            output_sizes,
            dtype=torch.int32,
            device=self._device,
        )

    def _storage_key(self) -> tuple:
        return tuple((f.data_ptr(), f.shape, f.stride(), f.dtype, f.device) for f in self._factors)

    def _bind_factors(self) -> None:
        # Detached aliases preserve live updates to trainable adapter storage.
        self._delta_factors = tuple(f.detach() if f.requires_grad else f for f in self._factors)
        self._a_ptrs = torch.tensor(
            [factor.data_ptr() for factor in self._factors[0::2]],
            dtype=torch.uint64,
            device=self._device,
        )
        self._b_ptrs = torch.tensor(
            [factor.data_ptr() for factor in self._factors[1::2]],
            dtype=torch.uint64,
            device=self._device,
        )
        self._storage = self._storage_key()

    def _check_storage(self) -> None:
        """Rebind after a factor's storage was replaced; raise if it cannot be."""
        key = self._storage_key()
        if key == self._storage:
            return
        for (_, shape, stride, dtype, device), (_, shape0, stride0, dtype0, device0) in zip(
            key, self._storage, strict=True
        ):
            if (shape, stride, dtype, device) != (shape0, stride0, dtype0, device0):
                raise ValueError(
                    "LoRA factor storage was replaced with a different shape, layout, "
                    "dtype or device; prepare a new inference plan"
                )
        if self._device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "LoRA factor storage was replaced; call the plan eagerly before capture"
            )
        self._bind_factors()

    @property
    def deterministic(self) -> bool:
        """Whether every route is bitwise repeatable (no FP atomics)."""
        return self._deterministic

    @property
    def provider(self) -> str:
        """Delta provider of the last successful project call."""
        return self._provider

    @property
    def base_provider(self) -> str:
        """Provider of the last successful base call."""
        return self._base_provider

    def base(self, x, weight, bias=None):
        """Base projection, also exposed for comparator precision diagnostics."""
        if (
            x.dtype == weight.dtype == torch.bfloat16
            and x.is_contiguous()
            and weight.is_contiguous()
            and bias is None
            and not weight.requires_grad
        ):
            output = self._linear(x, weight)
            self._base_provider = self._linear.provider
            return output
        output = F.linear(x, weight, bias)
        self._base_provider = "torch"
        return output

    def can_project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        factors: tuple[torch.Tensor, ...],
        *,
        scaling: float,
    ) -> bool:
        if factors is not self._factors or scaling != self._scaling:
            return False
        if x.ndim < 2 or x.shape[-1] != self._input_size:
            return False
        if self._policy is None:
            return False
        num_tokens = x.numel() // self._input_size
        if not prefer_small_batch_plan(
            self._policy,
            num_slices=len(self._output_sizes_tuple),
            rank=self._rank,
            num_tokens=num_tokens,
            k=self._input_size,
            deterministic=self._deterministic,
        ):
            return False
        if x.device != self._device or x.dtype != self._dtype:
            return False
        if weight.device != self._device or weight.dtype != self._dtype:
            return False
        if weight.shape != (self._output_size, self._input_size):
            return False
        return bias is None or (
            bias.shape == (self._output_size,)
            and bias.device == self._device
            and bias.dtype == self._dtype
        )

    def _select_provider(self, num_tokens: int) -> str:
        provider = (
            None
            if self._policy is None
            else select_provider(
                self._policy,
                num_slices=len(self._output_sizes_tuple),
                rank=self._rank,
                num_tokens=num_tokens,
                k=self._input_size,
                deterministic=self._deterministic,
            )
        )
        if provider is None:
            raise ValueError("LoRA projection is outside this plan's provider support range")
        return provider

    def project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, self._input_size)
        provider = self._select_provider(x_2d.shape[0])
        self._check_storage()
        # Preserve the original layout contract even if reshape made a copy.
        if x.is_contiguous():
            output = self.base(x_2d, weight, bias)
        else:
            output = F.linear(x_2d, weight, bias)
            self._base_provider = "torch"
        self._delta_(output, x_2d, provider)
        return output.reshape(*x.shape[:-1], self._output_size)

    def warm_up(self, max_m: int, weight: torch.Tensor | None = None) -> list[dict]:
        """Prepare every M = 1..max_m eagerly so later captures neither compile nor
        allocate metadata; returns the routes, like ``Linear.warm_up``.

        Each delta route is ``{"op": "delta", "m": (lo, hi), "provider", "config"}``
        for a run of M sharing provider and config ("torch" where the plan does not
        project). With ``weight``, the owned Linear's base routes follow as
        ``{"op": "base", **route}``. Runs on the current stream with zero inputs."""
        if max_m < 1:
            raise ValueError("max_m must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("LoRA warm_up must run eagerly, outside CUDA graph capture")
        self._check_storage()
        from ._lora_segmented import last_config

        x = torch.zeros(max_m, self._input_size, dtype=self._dtype, device=self._device)
        output = torch.zeros(max_m, self._output_size, dtype=self._dtype, device=self._device)
        routes: list[dict] = []
        previous = self._provider
        for m in range(1, max_m + 1):
            provider = (
                None
                if self._policy is None
                else select_provider(
                    self._policy,
                    num_slices=len(self._output_sizes_tuple),
                    rank=self._rank,
                    num_tokens=m,
                    k=self._input_size,
                    deterministic=self._deterministic,
                )
            )
            config = None
            if provider is not None:
                self._delta_(output[:m], x[:m], provider)
                config = last_config() if provider == "segmented" else None
            name = "torch" if provider is None else self._provider
            if routes and (routes[-1]["provider"], routes[-1]["config"]) == (name, config):
                routes[-1]["m"] = (routes[-1]["m"][0], m)
            else:
                routes.append({"op": "delta", "m": (m, m), "provider": name, "config": config})
        self._provider = previous
        if weight is not None:
            routes.extend({"op": "base", **route} for route in self._linear.warm_up(weight, max_m))
        return routes

    def _delta_(self, output: torch.Tensor, x_2d: torch.Tensor, provider: str) -> None:
        if provider == "segmented":
            # Lazy import: importing the public CPU API must not import Triton.
            from ._lora_segmented import lora_delta_factors_

            lora_delta_factors_(
                output,
                x_2d.detach().contiguous(),
                self._delta_factors,
                scaling=self._scaling,
                output_sizes=self._output_sizes_tuple,
                row_parallel=self._row_parallel,
            )
            self._provider = "triton-lora-v1"
            return
        num_tokens = x_2d.shape[0]
        low_rank = torch.empty(
            len(self._output_sizes_tuple),
            num_tokens,
            self._rank,
            dtype=torch.float32,
            device=self._device,
        )
        low_rank.zero_()

        shrink_grid = (
            triton.cdiv(num_tokens, _BLOCK_M) * triton.cdiv(self._rank, _BLOCK_RANK) * _SPLIT_K,
            len(self._output_sizes_tuple),
        )
        _shrink_kernel[shrink_grid](
            x_2d,
            self._a_ptrs,
            low_rank,
            num_tokens,
            self._input_size,
            self._rank,
            self._scaling,
            x_2d.stride(0),
            x_2d.stride(1),
            low_rank.stride(0),
            low_rank.stride(1),
            low_rank.stride(2),
            _BLOCK_M,
            _BLOCK_RANK,
            _SHRINK_BLOCK_K,
            _SPLIT_K,
            num_warps=4,
            num_stages=2,
        )
        expand_grid = (
            triton.cdiv(num_tokens, _BLOCK_M)
            * triton.cdiv(max(self._output_sizes_tuple), _EXPAND_BLOCK_N),
            len(self._output_sizes_tuple),
        )
        _expand_kernel[expand_grid](
            low_rank,
            self._b_ptrs,
            output,
            self._output_offsets,
            self._output_sizes,
            num_tokens,
            self._rank,
            max(self._output_sizes_tuple),
            low_rank.stride(0),
            low_rank.stride(1),
            low_rank.stride(2),
            output.stride(0),
            output.stride(1),
            _BLOCK_M,
            _EXPAND_BLOCK_N,
            _EXPAND_BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        self._provider = "triton-small-batch-v1"


def try_prepare_small_batch_lora_plan(
    factors: tuple[torch.Tensor, ...],
    *,
    input_size: int,
    output_sizes: tuple[int, ...],
    scaling: float,
    deterministic: bool = True,
    row_parallel: bool = False,
) -> SmallBatchLoRAInferencePlan | None:
    """Prepare the segmented-delta plan on CUDA devices with compute capability >= 8.9."""

    if input_size <= 0 or not output_sizes or len(factors) != 2 * len(output_sizes):
        return None
    first = factors[0]
    if (
        first.device.type != "cuda"
        or first.dtype not in _SUPPORTED_DTYPES
        or first.ndim != 2
        or first.shape[1] != input_size
    ):
        return None
    policy = policy_for_hardware(
        device_name=torch.cuda.get_device_name(first.device),
        capability=torch.cuda.get_device_capability(first.device),
        sms=torch.cuda.get_device_properties(first.device).multi_processor_count,
    )
    if policy is None:
        return None

    rank = first.shape[0]
    if not supports_projection(num_slices=len(output_sizes), rank=rank):
        return None
    for index, output_size in enumerate(output_sizes):
        a = factors[2 * index]
        b = factors[2 * index + 1]
        if (
            output_size <= 0
            or a.device != first.device
            or b.device != first.device
            or a.dtype != first.dtype
            or b.dtype != first.dtype
            or a.shape != (rank, input_size)
            or b.shape != (output_size, rank)
            or not a.is_contiguous()
            or not b.is_contiguous()
        ):
            return None
    return SmallBatchLoRAInferencePlan(
        factors,
        input_size=input_size,
        output_sizes=output_sizes,
        scaling=scaling,
        policy=policy,
        deterministic=deterministic,
        row_parallel=row_parallel,
    )
