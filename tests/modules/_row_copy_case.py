# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Recurrent-pool-like tensors for RowCopy: per layer, one byte buffer holding each state kind's
[rows, *shape] block, as the engine lays out its pool."""

import torch

# (shape, dtype) per state kind: a GDN-like FP32 state, a BF16 conv history, odd-sized extras.
KINDS = (
    ((4, 8, 8), torch.float32),
    ((3, 24), torch.bfloat16),
    ((5,), torch.int64),
    ((7,), torch.float16),
    ((3,), torch.uint8),
)
# A GDN-shaped state (64 chunks of 1024 8-byte units), an odd tail, 4- and 2-byte words.
GDN_KINDS = (
    ((8, 128, 128), torch.float32),
    ((3, 520), torch.bfloat16),
    ((3, 5), torch.float32),
    ((3,), torch.float16),
)


def pool(device, *, layers=3, rows=9, kinds=KINDS, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tensors = []
    for _ in range(layers):
        sizes = [rows * torch.Size(shape).numel() * dtype.itemsize for shape, dtype in kinds]
        buffer = torch.randint(0, 256, (sum(sizes),), dtype=torch.uint8, generator=generator)
        buffer = buffer.to(device)
        offset = 0
        for (shape, dtype), size in zip(kinds, sizes, strict=True):
            tensors.append(buffer[offset : offset + size].view(dtype).view(rows, *shape))
            offset += size
    return tensors


def expected(tensors, sources, targets):
    """index_select + index_copy_ of the valid pairs, on copies."""
    out = [t.clone() for t in tensors]
    for t in out:
        rows = t.shape[0]
        keep = (sources >= 0) & (targets >= 0) & (sources < rows) & (targets < rows)
        t.view(rows, -1).view(torch.uint8).index_copy_(
            0,
            targets[keep].long(),
            t.view(rows, -1).view(torch.uint8).index_select(0, sources[keep].long()),
        )
    return out


def bits(t):
    return t.contiguous().view(torch.uint8)
