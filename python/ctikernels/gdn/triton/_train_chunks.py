# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GDN training, a batch's sequence index in one launch: ``chunk_index`` copies cu_seqlens,
and writes each sequence's first chunk entry, the chunk table (sequence, chunk within it) and
each token's sequence, all int32 and read from the device (no host synchronisation)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 128  # rows (tokens / chunk table entries) per program
SEQUENCES = 32  # sequences per step of the in-program loop


@triton.jit(do_not_specialize=["count", "rows", "capacity"])
def _index_kernel(
    cu_in, cu, offsets, table, sequence, count, rows, capacity,
    CHUNK: tl.constexpr, BLOCK: tl.constexpr, SEQUENCES: tl.constexpr,
):  # fmt: skip
    # One specialization for every sequence count: sequences in SEQUENCES-wide steps with the
    # running chunk count carried, so no tile grows with the batch.
    first_program = tl.program_id(0) == 0
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = tl.zeros((BLOCK,), dtype=tl.int32)  # sequences ending at or before row i
    s = tl.zeros((BLOCK,), dtype=tl.int32)  # sequences whose chunks end at or before entry i
    first = tl.zeros((BLOCK,), dtype=tl.int32)  # offsets[s] (s clamped to count - 1)
    carry = tl.zeros((), dtype=tl.int32)
    for n0 in range(0, count, SEQUENCES):
        n = n0 + tl.arange(0, SEQUENCES)
        live = n < count
        starts = tl.load(cu_in + n, mask=live, other=0)
        ends = tl.load(cu_in + 1 + n, mask=live, other=0)
        chunks = tl.where(live, (ends - starts + CHUNK - 1) // CHUNK, 0)
        after = carry + tl.cumsum(chunks, 0)  # offsets[n + 1]
        if first_program:
            tl.store(cu + n, starts, mask=live)
            tl.store(offsets + 1 + n, after, mask=live)
        token += tl.sum((live[None, :] & (ends[None, :] <= i[:, None])).to(tl.int32), 1)
        done = live[None, :] & (after[None, :] <= i[:, None])
        s += tl.sum(done.to(tl.int32), 1)
        # offsets is increasing: offsets[s] is the largest offsets[n + 1] <= i with n < count - 1.
        first = tl.maximum(
            first, tl.max(tl.where(done & (n < count - 1)[None, :], after[None, :], 0), 1)
        )
        carry += tl.sum(chunks, 0)
    if first_program:
        tl.store(cu + count, tl.load(cu_in + count))
        tl.store(offsets, 0)
    tl.store(sequence + i, token, mask=i < rows)
    s = tl.minimum(s, count - 1)
    m = i < capacity
    tl.store(table + 2 * i, s, mask=m)
    tl.store(table + 2 * i + 1, i - first, mask=m)


def chunk_index(cu_seqlens: torch.Tensor, rows: int, chunk: int):
    """(cu_seqlens copy [N + 1], offsets [N + 1], table [capacity, 2], sequence [rows]) for a
    ``rows``-row batch of N >= 1 sequences, capacity = (rows + (chunk - 1) N) // chunk; entries
    past the batch's chunks name a chunk past the last sequence's end."""
    count = cu_seqlens.shape[0] - 1
    capacity = (rows + (chunk - 1) * count) // chunk
    cu, offsets = torch.empty_like(cu_seqlens), torch.empty_like(cu_seqlens)
    table = cu_seqlens.new_empty(capacity, 2)
    sequence = cu_seqlens.new_empty(rows)
    _index_kernel[(max(1, triton.cdiv(max(rows, capacity), BLOCK)),)](
        cu_seqlens, cu, offsets, table, sequence, count, rows, capacity,
        CHUNK=chunk, BLOCK=BLOCK, SEQUENCES=SEQUENCES,
    )  # fmt: skip
    return cu, offsets, table, sequence


__all__ = ["chunk_index"]
