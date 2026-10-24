# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton DN4 penalties: memset, count, apply; the torch provider's rounding sequence.

Studied vLLM 0.31.0 v1/worker/gpu/sample/penalties.py (_penalties_kernel: per-row
vocabulary tiles, counts held as state) and SGLang 0.5.22 srt/sampling/penaltylib;
no code copied. Counts are rebuilt per call from the engine's token table, one int32
per (request, token): output count in bits 0-30, prompt flag in bit 31. One apply
pass over (request, position, vocabulary tile) adds the preceding drafts. Arithmetic
uses explicit .rn PTX so nothing is contracted into an FMA.
"""

import torch
import triton
import triton.language as tl

COUNT_BLOCK, APPLY_BLOCK, APPLY_WARPS = 1024, 1024, 4


@triton.jit
def _rn(a, b, op: tl.constexpr):
    return tl.inline_asm_elementwise(
        op + ".rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _count(
    table,
    table_stride,
    table_column,
    slots,
    prompt_lengths,
    lengths,
    counts,
    V,
    BLOCK: tl.constexpr,
):
    """Output counts in bits 0-30 (they stay below 2**31); bit 31 flags prompt tokens."""
    n = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    c = tl.load(lengths + n).to(tl.int64)
    if start >= c:
        return
    p = tl.load(prompt_lengths + n).to(tl.int64)
    slot = tl.load(slots + n).to(tl.int64)
    cols = start + tl.arange(0, BLOCK)
    in_context = cols < c
    ids = tl.load(table + slot * table_stride + cols * table_column, mask=in_context, other=-1).to(
        tl.int64
    )
    valid = in_context & (ids >= 0) & (ids < V)
    base = n.to(tl.int64) * V + ids
    tl.atomic_or(counts + base, -2147483648, mask=valid & (cols < p), sem="relaxed")
    tl.atomic_add(counts + base, 1, mask=valid & (cols >= p), sem="relaxed")


@triton.jit
def _apply(
    logits,
    logits_n,
    logits_q,
    out,
    out_n,
    out_q,
    counts,
    drafts,
    drafts_n,
    drafts_i,
    presence,
    frequency,
    repetition,
    Q,
    V,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    n, j = row // Q, row % Q
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < V
    n64, j64 = n.to(tl.int64), j.to(tl.int64)
    x = tl.load(logits + n64 * logits_n + j64 * logits_q + cols, mask=mask).to(tl.float32)
    packed = tl.load(counts + n64 * V + cols, mask=mask, other=0)
    count = packed & 0x7FFFFFFF
    # Position j counts drafts d_0..d_{j-1}; out-of-range drafts match no column.
    for i in range(0, j):
        d = tl.load(drafts + n64 * drafts_n + i * drafts_i).to(tl.int64)
        count += (cols == d).to(tl.int32)
    seen = (packed != 0) | (count > 0)
    pres = tl.load(presence + n)
    freq = tl.load(frequency + n)
    rep = tl.load(repetition + n)
    scale = tl.where(seen, rep, 1.0)
    reciprocal = _rn(tl.full([BLOCK], 1.0, tl.float32), scale, "div")
    value = _rn(x, tl.where(x > 0, reciprocal, scale), "mul")
    value = _rn(value, _rn(tl.broadcast_to(freq, [BLOCK]), count.to(tl.float32), "mul"), "sub")
    used = (count > 0).to(tl.float32)
    value = _rn(value, _rn(tl.broadcast_to(pres, [BLOCK]), used, "mul"), "sub")
    identity = (pres == 0) & (freq == 0) & (rep == 1)
    value = tl.where(identity, x, value)
    tl.store(out + n64 * out_n + j64 * out_q + cols, value, mask=mask)


def apply(table, vocab, logits, rows, drafts, out):
    """logits/out are [N,Q,V] views; drafts is [N,Q-1] or None. Scratch is per call."""
    n, q, _ = logits.shape
    if n == 0:
        return
    device = logits.device
    # Counts with the prompt flag in bit 31: one memset node clears both.
    counts = torch.zeros((n, vocab), dtype=torch.int32, device=device)
    width = table.shape[1]
    if width:
        _count[(n, triton.cdiv(width, COUNT_BLOCK))](
            table,
            table.stride(0),
            table.stride(1),
            rows.rows,
            rows.prompt_lengths,
            rows.lengths,
            counts,
            vocab,
            BLOCK=COUNT_BLOCK,
        )
    if drafts is None or drafts.shape[1] == 0:
        drafts, strides = counts, (0, 0)  # never read: the draft loop is empty at Q=1
    else:
        strides = drafts.stride()
    _apply[(n * q, triton.cdiv(vocab, APPLY_BLOCK))](
        logits,
        logits.stride(0),
        logits.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        counts,
        drafts,
        *strides,
        rows.presence,
        rows.frequency,
        rows.repetition,
        q,
        vocab,
        BLOCK=APPLY_BLOCK,
        num_warps=APPLY_WARPS,
    )
