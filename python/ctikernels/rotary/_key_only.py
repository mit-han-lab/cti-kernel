# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_key_kernel(
    key,
    cache,
    positions,
    token_stride,
    head_stride,
    dim_stride,
    cache_row_stride,
    cache_dim_stride,
    position_stride,
    num_heads: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // num_heads
    head = row % num_heads
    dim = tl.arange(0, BLOCK)
    mask = dim < half_dim
    position = tl.load(positions + token * position_stride)
    cache_offset = position * cache_row_stride + dim * cache_dim_stride
    cosine = tl.load(cache + cache_offset, mask=mask, other=0.0)
    sine = tl.load(cache + cache_offset + half_dim * cache_dim_stride, mask=mask, other=0.0)
    key_offset = token * token_stride + head * head_stride + dim * dim_stride
    first = tl.load(key + key_offset, mask=mask, other=0.0).to(tl.float32)
    second = tl.load(key + key_offset + half_dim * dim_stride, mask=mask, other=0.0).to(tl.float32)
    tl.store(key + key_offset, first * cosine - second * sine, mask=mask)
    tl.store(key + key_offset + half_dim * dim_stride, second * cosine + first * sine, mask=mask)


def apply_rotary_key(
    key: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
) -> None:
    if not key.is_cuda or key.device != cache.device or key.device != positions.device:
        raise ValueError("key, cache and positions must share one CUDA device")
    if key.dtype not in (torch.float16, torch.bfloat16) or cache.dtype != torch.float32:
        raise TypeError("key-only rotary requires FP16/BF16 key and FP32 cache")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")
    if type(head_dim) is not int or head_dim < 2 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if key.ndim == 2 and key.shape[1] % head_dim == 0:
        view = key.view(key.shape[0], key.shape[1] // head_dim, head_dim)
    elif key.ndim == 3 and key.shape[2] == head_dim:
        view = key
    else:
        raise ValueError(
            "key must have shape [tokens, heads * head_dim] or [tokens, heads, head_dim]"
        )
    if positions.shape != (view.shape[0],) or cache.ndim != 2 or cache.shape[1] != head_dim:
        raise ValueError(
            "positions must match tokens and cache must have shape [max_position, head_dim]"
        )
    if view.numel() == 0:
        return
    if any(stride <= 0 for stride in view.stride()):
        raise ValueError("key must have positive non-overlapping strides")
    _rotate_key_kernel[(view.shape[0] * view.shape[1],)](
        view,
        cache,
        positions,
        *view.stride(),
        *cache.stride(),
        positions.stride(0),
        view.shape[1],
        head_dim // 2,
        triton.next_power_of_2(head_dim // 2),
        num_warps=4,
        enable_fp_fusion=False,
    )
