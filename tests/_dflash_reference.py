# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent dense DRAFT oracle, intentionally outside the kernel package."""

import math

import torch


def mask_predicate(queries, keys, *, causal, window):
    p, k = queries[:, None], keys[None, :]
    if window is None:
        if causal:
            return k <= p
        return torch.ones((queries.numel(), keys.numel()), dtype=torch.bool, device=p.device)
    # All positions are nonnegative: larger-smaller cannot overflow int64.
    distance = torch.maximum(p, k) - torch.minimum(p, k)
    return (distance < window) & ((k <= p) if causal else True)


def draft_attention_reference(
    q,
    k,
    v,
    *,
    slots,
    positions,
    token_indptr,
    token_indices,
    query_length,
    causal,
    window,
    precision=torch.float32,
):
    """Dense FP32 QK/softmax/V over identical quantized input; FP64 audit optional."""
    result = torch.zeros(q.shape, dtype=precision, device=q.device)
    group = q.shape[1] // k.shape[2]
    flat_k, flat_v = k.flatten(0, 1), v.flatten(0, 1)
    for row, slot in enumerate(slots.tolist()):
        if slot == -1:
            continue
        start, end = map(int, token_indptr[row : row + 2])
        indices = token_indices[start:end].long().to(q.device)
        qs = slice(row * query_length, (row + 1) * query_length)
        p = positions[qs].to(q.device)
        base = int(p[0]) - (end - start - query_length)
        key_positions = torch.arange(end - start, device=q.device, dtype=torch.int64) + base
        allowed = mask_predicate(p, key_positions, causal=causal, window=window)
        query = q[qs].to(precision).reshape(query_length, k.shape[2], group, q.shape[2])
        query = query.permute(1, 2, 0, 3)
        key = flat_k[indices].to(precision).permute(1, 2, 0)[:, None]
        value = flat_v[indices].to(precision).permute(1, 0, 2)[:, None]
        scores = (query @ key) / math.sqrt(q.shape[2])
        probs = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
        result[qs] = (probs @ value).permute(2, 0, 1, 3).reshape_as(result[qs])
    return result


def chunked_draft_attention_reference(
    q,
    k,
    v,
    *,
    slots,
    positions,
    token_indptr,
    token_indices,
    query_length,
    causal,
    window,
    chunk_size=16384,
):
    """FP32 online softmax; scratch scales with key chunk size, never capacity.

    Gather only the union of visible keys for windowed attention. Grouped matmuls
    share K/V across query heads without expanding either operand to Hq heads.
    Disable TF32 in the caller when using this as a correctness oracle on CUDA.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    hk, dim = k.shape[2:]
    group = q.shape[1] // hk
    flat_k, flat_v = k.flatten(0, 1), v.flatten(0, 1)
    for row, slot in enumerate(slots.tolist()):
        if slot == -1:
            continue
        start, end = map(int, token_indptr[row : row + 2])
        length = end - start
        qs = slice(row * query_length, (row + 1) * query_length)
        p = positions[qs].to(device=q.device, dtype=torch.int64)
        base = int(positions[row * query_length]) - (length - query_length)
        query = q[qs].float().reshape(query_length, hk, group, dim)
        query = query.permute(1, 2, 0, 3).reshape(hk, group * query_length, dim)
        maximum = torch.full(query.shape[:2] + (1,), -torch.inf, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(query)
        first = 0 if window is None else max(0, length - query_length - window + 1)
        for offset in range(first, length, chunk_size):
            stop = min(offset + chunk_size, length)
            indices = token_indices[start + offset : start + stop].to(
                device=q.device, dtype=torch.int64
            )
            keys = torch.arange(offset, stop, dtype=torch.int64, device=q.device) + base
            allowed = mask_predicate(p, keys, causal=causal, window=window)
            key = flat_k[indices].float().permute(1, 2, 0)
            value = flat_v[indices].float().permute(1, 0, 2)
            scores = (query @ key) / math.sqrt(dim)
            scores.view(hk, group, query_length, -1).masked_fill_(~allowed, -torch.inf)
            updated = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            # Narrow windows can leave a query fully masked in an early chunk.
            safe_maximum = torch.where(torch.isfinite(updated), updated, 0.0)
            rescale = torch.exp(maximum - safe_maximum)
            scores.sub_(safe_maximum).exp_()
            numerator = numerator * rescale + scores @ value
            denominator = denominator * rescale + scores.sum(dim=-1, keepdim=True)
            maximum = updated
        result[qs] = (
            (numerator / denominator)
            .reshape(hk, group, query_length, dim)
            .permute(2, 0, 1, 3)
            .reshape_as(result[qs])
        )
    return result
