# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Ported from the Qwen3.5 vision rotary contributed by Waterpine.

"""Private per-token-angle rotary launch; the gradient guard lives in vision.py."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 256
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit(do_not_specialize=["num_elements"])
def _vision_rotary_kernel(
    query,
    key,
    cos,
    sin,
    output_query,
    output_key,
    num_elements,
    query_stride_token,
    query_stride_head,
    key_stride_token,
    key_stride_head,
    cos_stride_token,
    sin_stride_token,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    head_width = NUM_HEADS * HEAD_DIM
    token = offsets // head_width
    head_offset = offsets % head_width
    head = head_offset // HEAD_DIM
    column = head_offset % HEAD_DIM
    half_dim: tl.constexpr = HEAD_DIM // 2
    partner_column = tl.where(column < half_dim, column + half_dim, column - half_dim)
    partner_sign = tl.where(column < half_dim, -1.0, 1.0)

    query_offset = token * query_stride_token + head * query_stride_head + column
    query_partner_offset = token * query_stride_token + head * query_stride_head + partner_column
    key_offset = token * key_stride_token + head * key_stride_head + column
    key_partner_offset = token * key_stride_token + head * key_stride_head + partner_column
    cos_offset = token * cos_stride_token + column
    sin_offset = token * sin_stride_token + column

    cosine = tl.load(cos + cos_offset, mask=mask, other=0.0).to(tl.float32)
    sine = tl.load(sin + sin_offset, mask=mask, other=0.0).to(tl.float32)
    query_value = tl.load(query + query_offset, mask=mask, other=0.0).to(tl.float32)
    query_partner = tl.load(query + query_partner_offset, mask=mask, other=0.0).to(tl.float32)
    key_value = tl.load(key + key_offset, mask=mask, other=0.0).to(tl.float32)
    key_partner = tl.load(key + key_partner_offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_query + offsets,
        query_value * cosine + partner_sign * query_partner * sine,
        mask=mask,
    )
    tl.store(
        output_key + offsets,
        key_value * cosine + partner_sign * key_partner * sine,
        mask=mask,
    )


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.is_cuda:
        raise ValueError("rotary inputs must be CUDA tensors")
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("query and key must share one [tokens, heads, head_dim] shape")
    if query.shape[-1] % 2:
        raise ValueError("rotary head dimension must be even")
    if cos.shape != (query.shape[0], query.shape[2]) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have shape [tokens, head_dim]")
    if any(tensor.device != query.device for tensor in (key, cos, sin)):
        raise ValueError("rotary inputs must reside on one device")
    if query.dtype not in _DTYPES or key.dtype != query.dtype:
        raise TypeError("query and key must share one FP16, BF16 or FP32 dtype")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise TypeError("cos and sin must be FP32")
    if any(tensor.stride(-1) != 1 for tensor in (query, key, cos, sin)):
        raise ValueError("rotary inputs must be contiguous in their last dimension")

    output_query = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_key = torch.empty_like(output_query)
    num_elements = query.numel()
    if num_elements == 0:
        return output_query, output_key
    _vision_rotary_kernel[(triton.cdiv(num_elements, _BLOCK_SIZE),)](
        query,
        key,
        cos,
        sin,
        output_query,
        output_key,
        num_elements,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        cos.stride(0),
        sin.stride(0),
        query.shape[1],
        query.shape[2],
        _BLOCK_SIZE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output_query, output_key
