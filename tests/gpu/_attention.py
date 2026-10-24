# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Shared paged metadata and an independent CPU FP32 attention oracle."""

from itertools import accumulate
from math import sqrt

import torch


def paged_metadata(query_lengths, kv_lengths, page_size):
    page_counts = tuple((length + page_size - 1) // page_size for length in kv_lengths)
    page_indices = tuple(reversed(range(1, 2 * sum(page_counts), 2)))
    values = {
        "cu_seqlens_q": (0, *accumulate(query_lengths)),
        "cu_seqlens_k": (0, *accumulate(page_counts)),
        "indices": page_indices,
        "last_page_len": tuple((length - 1) % page_size + 1 for length in kv_lengths),
        "seq_lens": kv_lengths,
    }
    return {
        name: torch.tensor(value, dtype=torch.int32, device="cpu", pin_memory=True)
        for name, value in values.items()
    }


def reference_attention(
    query,
    cache,
    query_lengths,
    kv_lengths,
    indices,
    *,
    causal=True,
    sm_scale=None,
    window_left=-1,
    logits_soft_cap=None,
    custom_mask=None,
    k_scale=1.0,
    v_scale=1.0,
    compute_dtype=torch.float32,
):
    """CPU reference (optionally FP64), including bottom-right causal alignment."""
    key_cache, value_cache = (value.to(compute_dtype) for value in cache)
    page_size = key_cache.shape[1]
    group_size = query.shape[1] // key_cache.shape[2]
    outputs = []
    query_start = page_start = mask_start = 0
    for query_length, kv_length in zip(query_lengths, kv_lengths, strict=True):
        page_count = (kv_length + page_size - 1) // page_size
        pages = indices[page_start : page_start + page_count].long()
        key = key_cache[pages].flatten(0, 1)[:kv_length] * k_scale
        value = value_cache[pages].flatten(0, 1)[:kv_length] * v_scale
        q = query[query_start : query_start + query_length].to(compute_dtype).transpose(0, 1)
        k = key.transpose(0, 1).repeat_interleave(group_size, dim=0)
        v = value.transpose(0, 1).repeat_interleave(group_size, dim=0)
        scale = 1 / sqrt(query.shape[-1]) if sm_scale is None else sm_scale
        scores = (q @ k.transpose(-1, -2)) * scale
        # FlashInfer DefaultAttention (variants.cuh:47-49,72-74,89 in both
        # 0.6.11.post3 and 0.7.0.post1): cap*tanh(dot*scale/cap), then masking.
        if logits_soft_cap:
            scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
        query_positions = torch.arange(query_length) + kv_length - query_length
        key_positions = torch.arange(kv_length)[None, :]
        keep = torch.ones(query_length, kv_length, dtype=torch.bool)
        if custom_mask is not None:
            count = query_length * kv_length
            keep &= custom_mask[mask_start : mask_start + count].view(query_length, kv_length)
            mask_start += count
        elif causal:
            keep &= key_positions <= query_positions[:, None]
        if window_left >= 0:
            keep &= query_positions[:, None] - key_positions <= window_left
        scores.masked_fill_(~keep, -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append((probabilities @ v).transpose(0, 1))
        query_start += query_length
        page_start += page_count
    return torch.cat(outputs)


def kv_views(cache, layout):
    """Copy identical values into aligned NHD views with independent K/V strides."""
    k, v = cache
    pages, page, heads, dim = k.shape
    if layout == "heads":
        storage = torch.empty(pages, page, 2 * heads, dim, dtype=k.dtype, device=k.device)
        views = (storage[:, :, ::2], storage[:, :, 1::2])
    elif layout == "packed":
        storage = torch.empty(pages, 2, page, heads, dim, dtype=k.dtype, device=k.device)
        views = (storage[:, 0], storage[:, 1])
    else:
        assert layout in ("padded", "mixed")
        views = tuple(
            torch.empty(pages, page, heads, width * dim, dtype=k.dtype, device=k.device)[..., :dim]
            for width in ((2, 2) if layout == "padded" else (2, 3))
        )
    for view, source in zip(views, cache, strict=True):
        view.copy_(source)
    return views
