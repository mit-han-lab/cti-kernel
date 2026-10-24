# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""fixed_query_addresses: one launch, no allocation, atomics or inter-program dependencies."""

import triton
import triton.language as tl

# Runtime scalars and pointers never specialize: a new B, M, guard, capacity,
# stride or tensor alignment reuses the warmed binary (capture never compiles).
_RUNTIME = ["batch", "m", "guard", "n", "stride", "tiles"]
_POINTERS = ["rows", "start", "done", "cache", "writes", "indptr", "indices"]


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME + _POINTERS)
def fixed_query_addresses(
    rows,
    start,
    done,
    cache,
    writes,
    indptr,
    indices,
    batch,
    m,
    guard,
    n,
    stride,
    tiles,
    Q: tl.constexpr,
    BLOCK: tl.constexpr,
    SCAN: tl.constexpr,
    WAIT: tl.constexpr,
):
    """Work item w is (row b, column tile t) = (w // tiles, w % tiles), with
    tiles*BLOCK >= M+Q > any row's CSR length and >= Q. Each program walks its
    items in increasing w, so its running CSR offset (the sum of the lengths of
    rows < b, from the saved round state) only moves forward."""
    if WAIT:
        tl.extra.cuda.gdc_wait()
    program = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0).to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    scan = tl.arange(0, SCAN).to(tl.int64)
    m = m.to(tl.int64)
    n = n.to(tl.int64)
    stride = stride.to(tl.int64)
    total = batch.to(tl.int64) * tiles
    row = program * 0  # base is the CSR offset of row `row`
    base = program * 0
    item = program
    while item < total:
        b = item // tiles
        t = item - b * tiles
        while row < b:
            k = row + scan
            inside = k < b
            f = tl.load(start + k, mask=inside, other=0)
            d = tl.load(done + k, mask=inside, other=0)
            length = tl.where((d != 0) | (f < 0) | (f >= m), Q, f + Q)
            base += tl.sum(tl.where(inside, length, 0))
            row = tl.minimum(row + SCAN, b)
        r = tl.load(rows + b).to(tl.int64)
        f = tl.load(start + b)
        dummy = (tl.load(done + b) != 0) | (f < 0) | (f >= m)
        length = tl.where(dummy, Q, f + Q)
        g = guard + b * Q
        source = cache + r * stride
        j = t * BLOCK + lane
        if t == 0:
            if b == 0:
                tl.store(indptr, 0)
            tl.store(indptr + b + 1, (base + length).to(tl.int32))
        if t * BLOCK < Q:
            # Cache write index of position j: column f+j of the row, or the guard.
            column = f + j
            mapped = (j < Q) & (column < m) & ~dummy
            value = tl.load(source + column, mask=mapped, other=0)
            guarded = tl.where(dummy, g + j, g + tl.minimum(column - m, Q - 1))
            tl.store(writes + b * Q + j, tl.where(mapped, value, guarded.to(tl.int32)), j < Q)
        # Valid CSR prefix: only entries below the kv_indices capacity n.
        bound = tl.minimum(length, n - base)
        if t * BLOCK < bound:
            inside = j < bound
            mapped = inside & (j < m) & ~dummy
            value = tl.load(source + j, mask=mapped, other=0)
            guarded = tl.where(dummy, g + j, g + tl.minimum(j - m, Q - 1))
            tl.store(indices + base + j, tl.where(mapped, value, guarded.to(tl.int32)), inside)
        item += programs
