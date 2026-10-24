# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Semantic reference: DeepSeek-V4-Flash 60d8d707 inference/model.py Indexer; no upstream code
# copied.

"""Triton indexer logits: the query transform (partial RoPE, Hadamard, FP4 round trip) and
the head-weighted ReLU scores of every visible index key in the request's FP4 pages."""

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ._quant import fp4, hadamard128, rotate


@triton.jit
def transform(Q, W, Pos, Ids, Pages, NP, Rope, Codes, Scales, WOut, Lens, Table, TW, MAX_ENTRIES,
              GATHER: tl.constexpr, PDL: tl.constexpr):  # fmt: skip
    """Query row (one head): RoPE, Hadamard, then E2M1 pairs [64] and UE8M0 bytes [4]; the
    head's weight times ``128^-0.5 * 64^-0.5`` in FP32; per query the visible entry count and,
    with ``GATHER``, its page-table row (the FlashInfer provider's block table)."""
    if PDL:
        gdc_launch_dependents()  # the Triton score kernel's key loads overlap this one
    row = tl.program_id(0)
    query, head = row // 64, row % 64
    d = tl.arange(0, 128)
    pos = tl.load(Pos + query)
    x = tl.load(Q + row * 128 + d).to(tl.float32)
    x = rotate(x, pos, Rope, 128).to(tl.bfloat16).to(tl.float32)
    code, exponent = fp4(hadamard128(x))
    i = tl.arange(0, 64)
    payload = tl.gather(code, i * 2, 0) | (tl.gather(code, i * 2 + 1, 0) << 4)
    tl.store(Codes + row * 64 + i, payload.to(tl.uint8))
    tl.store(Scales + row * 4 + tl.arange(0, 4), exponent.to(tl.uint8))
    tl.store(WOut + row, tl.load(W + row).to(tl.float32) * (128**-0.5 * 64**-0.5))
    if head == 0:
        tl.store(Lens + query, tl.minimum((pos + 1) // 4, MAX_ENTRIES).to(tl.int32))
    if GATHER:
        slot = tl.load(Ids + query).to(tl.int64)
        for first in range(head * 64, TW, 64 * 64):
            j = first + tl.arange(0, 64)
            tl.store(Table + query * TW + j, tl.load(Pages + slot * NP + j, j < TW), j < TW)


@triton.jit
def score(Codes, Scales, WOut, Lens, Ids, Pages, Cache, Out, NP, OUT_STRIDE,
          BLOCK: tl.constexpr, PDL: tl.constexpr):  # fmt: skip
    """FP32 scores of ``BLOCK`` consecutive entries for one query. Entries are the MMA rows,
    heads the columns; query and keys stay E2M1 with UE8M0 per 32 (block-scaled MMA), so
    every product is exact. Keys load before the wait on the query transform (PDL)."""
    query, chunk = tl.program_id(0), tl.program_id(1)
    h, b, g = tl.arange(0, 64), tl.arange(0, 64), tl.arange(0, 4)
    j = chunk * BLOCK + tl.arange(0, BLOCK)
    if PDL:
        # Lengths come from the transform too; the page table and keys do not.
        slot = tl.load(Ids + query).to(tl.int64)
        page = tl.load(Pages + slot * NP + j // 64, mask=j // 64 < NP, other=0).to(tl.int64)
        gdc_wait()
        live = j < tl.load(Lens + query)
    else:
        live = j < tl.load(Lens + query)
        slot = tl.load(Ids + query).to(tl.int64)
        page = tl.load(Pages + slot * NP + j // 64, mask=live, other=0).to(tl.int64)
    # Entry j: row j % 64 of page pages[slot, j // 64]; 64 bytes of E2M1 pairs (even channel
    # low) per row, then 4 UE8M0 bytes per row after the page's 4096 data bytes.
    row = Cache + tl.where(live, page, 0) * 4352
    key = tl.load(row[:, None] + (j % 64)[:, None] * 64 + b[None, :], mask=live[:, None], other=0)
    key_scale = tl.load(row[:, None] + 4096 + (j % 64)[:, None] * 4 + g[None, :],
                        mask=live[:, None], other=127)  # fmt: skip
    q = tl.load(Codes + query * 4096 + h[:, None] * 64 + b[None, :])
    q_scale = tl.load(Scales + query * 256 + h[:, None] * 4 + g[None, :])
    weight = tl.load(WOut + query * 64 + h)
    dots = tl.dot_scaled(key, key_scale, "e2m1", tl.trans(q), q_scale, "e2m1")
    scores = tl.sum(tl.maximum(dots, 0) * weight[None, :], 1)
    tl.store(Out + query * OUT_STRIDE + j, scores, live)


def prepare(op, q, weights, positions, ids, pages, rope, gather):
    t = q.shape[0]
    transform[(t * 64,)](q, weights, positions, ids, pages, pages.shape[1], rope, op._codes,
                         op._scales, op._weights, op._lens, op._table, op._table.shape[1],
                         op.max_entries, gather, op._pdl and not gather,
                         enable_fp_fusion=False)  # fmt: skip


def run(op, ids, pages, cache, out):
    t = out.shape[0]
    # GB300 sweep (DSV4 decode, 8k-128k context, B=1-4, 4 warps): 256-entry tiles once they
    # still fill every SM (B=2/4 at 128k: 1.27x/1.17x), else 128. Same bits either way (the
    # tile only partitions entries); warps stay 4, as 8 changes the FP32 summation order.
    block = 256 if t * op.max_entries >= 256 * op._sms else 128
    score[(t, triton.cdiv(op.max_entries, block))](
        op._codes, op._scales, op._weights, op._lens, ids, pages, cache, out, pages.shape[1],
        out.stride(0), block, op._pdl, num_warps=4, enable_fp_fusion=False, launch_pdl=op._pdl,
    )  # fmt: skip
