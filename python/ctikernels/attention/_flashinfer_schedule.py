# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2023 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer fa381e5162d9b6ab5479b77e5b6d287b732d108e
# (`include/flashinfer/attention/scheduler.cuh`); see THIRD_PARTY_NOTICES.

"""Device schedules for FlashInfer's PrefillSplitQOKVIndptr contract.

The original small-Q kernels remain unchanged; large Q uses bounded chunks.
"""

import triton as tr
import triton.language as tl

# This is not a generic Split-KV implementation. Its outputs are private FA2
# workspace views supplied by _flashinfer.py, not engine-owned page metadata.


@tr.jit
def _write_schedule(
    pages,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    SHORT_TOKENS: tl.constexpr,
    MIN_CHUNK_2: tl.constexpr,
    SHORT_TOKENS_2: tl.constexpr,
    MIN_CHUNK_3: tl.constexpr,
    SHORT_TOKENS_3: tl.constexpr,
    WINDOW_LEFT,
):
    # ``pages`` is [REQUEST_BLOCK]; zero entries schedule no work. Native chunk
    # counts use pages, including a partially filled final page. The smallest
    # chunk comes from the first matching short-request tier, otherwise the
    # native planner's 128 tokens. Selection stays on device; the CTA bound still
    # sets larger chunks. Unused tiers have a zero token bound.
    request = tl.arange(0, REQUEST_BLOCK)
    high = tl.maximum(tl.max(pages, 0), 1)
    low = tl.where(
        high * PAGE <= SHORT_TOKENS,
        tl.full((), max(MIN_CHUNK // PAGE, 1), tl.int32),
        tl.where(
            high * PAGE <= SHORT_TOKENS_2,
            tl.full((), max(MIN_CHUNK_2 // PAGE, 1), tl.int32),
            tl.where(
                high * PAGE <= SHORT_TOKENS_3,
                tl.full((), max(MIN_CHUNK_3 // PAGE, 1), tl.int32),
                tl.full((), max(128 // PAGE, 1), tl.int32),
            ),
        ),
    )
    if QUERY > 32:
        if WINDOW_LEFT >= 0:
            # Native prefill.cuh hard-codes W-1+tile in its partial-output
            # stride. One chunk covers W-1+Q without disagreeing on that stride.
            low = tl.maximum(low, high)
    while low < high:
        middle = (low + high) // 2
        exceeds_grid = tl.sum(tl.cdiv(pages, middle), 0) * QUERY_TILE_COUNT > MAX_CTAS
        low = tl.where(exceeds_grid, middle + 1, low)
        high = tl.where(exceeds_grid, high, middle)
    chunks = tl.cdiv(pages, low)
    chunk_ends = tl.cumsum(chunks, 0)
    chunk_starts = chunk_ends - chunks
    total_chunks = tl.sum(chunks, 0)
    tl.store(CHUNK_SIZE, low * PAGE)
    tl.store(OUTPUT_OFFSETS + request, chunk_starts * QUERY, request < BATCH)
    tl.store(OUTPUT_OFFSETS + BATCH, total_chunks * QUERY)

    # Find each tile's request without a tiles-by-requests broadcast matrix.
    tile = tl.arange(0, TILE_BLOCK)
    left = tl.full((TILE_BLOCK,), 0, tl.int32)
    right = tl.full((TILE_BLOCK,), BATCH, tl.int32)
    for _ in range(SEARCH_STEPS):
        middle = (left + right) // 2
        end = tl.gather(chunk_ends, tl.minimum(middle, BATCH - 1), 0) * QUERY_TILE_COUNT
        advance = (middle < BATCH) & (tile >= end)
        left = tl.where(advance, middle + 1, left)
        right = tl.where(advance, right, middle)
    tile_request = tl.minimum(left, BATCH - 1)
    tile_chunks = tl.maximum(tl.gather(chunks, tile_request, 0), 1)
    local_tile = tile - tl.gather(chunk_starts, tile_request, 0) * QUERY_TILE_COUNT
    valid = tile < total_chunks * QUERY_TILE_COUNT
    tl.store(REQUESTS + tile, tl.where(valid, left, 0), tile < PADDED_CTAS)
    tl.store(QUERY_TILES + tile, tl.where(valid, local_tile // tile_chunks, 0), tile < PADDED_CTAS)
    tl.store(KV_TILES + tile, tl.where(valid, local_tile % tile_chunks, 0), tile < PADDED_CTAS)
    tl.store(VALID + tile, valid, tile < PADDED_CTAS)

    # Fixed query rows: a request without chunks owns an empty merge range,
    # which the native merge kernel writes as zeros.
    output = tl.arange(0, MERGE_BLOCK)
    output_request = tl.minimum(output // QUERY, BATCH - 1)
    output_chunks = tl.gather(chunks, output_request, 0)
    start = tl.gather(chunk_starts, output_request, 0) * QUERY
    merge = start + (output % QUERY) * output_chunks
    tl.store(MERGE_OFFSETS + output, merge, output < BATCH * QUERY)
    tl.store(MERGE_OFFSETS + BATCH * QUERY, total_chunks * QUERY)


@tr.jit(do_not_specialize=["WINDOW_LEFT"], do_not_specialize_on_alignment=["WINDOW_LEFT"])
def prepare_split_kv(
    KV_INDPTR,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WINDOW_LEFT=-1,
    QUERY_TILE_SIZE: tl.constexpr = 0,
    MIN_CHUNK: tl.constexpr = 128,
    SHORT_TOKENS: tl.constexpr = 0,
    MIN_CHUNK_2: tl.constexpr = 128,
    SHORT_TOKENS_2: tl.constexpr = 0,
    MIN_CHUNK_3: tl.constexpr = 128,
    SHORT_TOKENS_3: tl.constexpr = 0,
):
    request = tl.arange(0, REQUEST_BLOCK)
    pages = tl.load(KV_INDPTR + request + 1, request < BATCH, other=0) - tl.load(
        KV_INDPTR + request, request < BATCH, other=0
    )
    # Large Q needs the union of all query windows. P-1 covers a partial last
    # page; the frozen small-Q schedule retains its original tile-only span.
    if WINDOW_LEFT >= 0:
        span: tl.constexpr = max(QUERY, QUERY_TILE_SIZE) + (PAGE - 1 if QUERY > 32 else 0)
        pages = tl.minimum(pages, tl.cdiv(WINDOW_LEFT + span, PAGE))
    pages = tl.where(request < BATCH, tl.maximum(pages, 1), 0)
    _write_schedule(
        pages,
        REQUESTS,
        QUERY_TILES,
        KV_TILES,
        OUTPUT_OFFSETS,
        MERGE_OFFSETS,
        CHUNK_SIZE,
        VALID,
        BATCH,
        QUERY,
        PAGE,
        QUERY_TILE_COUNT,
        MAX_CTAS,
        PADDED_CTAS,
        REQUEST_BLOCK,
        TILE_BLOCK,
        MERGE_BLOCK,
        SEARCH_STEPS,
        MIN_CHUNK,
        SHORT_TOKENS,
        MIN_CHUNK_2,
        SHORT_TOKENS_2,
        MIN_CHUNK_3,
        SHORT_TOKENS_3,
        WINDOW_LEFT,
    )


@tr.jit(
    do_not_specialize=["WINDOW_LEFT", "TRIM_WINDOW_LEFT"],
    do_not_specialize_on_alignment=["WINDOW_LEFT", "TRIM_WINDOW_LEFT"],
)
def prepare_draft(
    SLOTS,
    TOKEN_INDPTR,
    TOKEN_INDICES,
    QUERY_INDPTR,
    PAGE_INDPTR,
    PAGE_INDICES,
    LAST_PAGE_LENGTHS,
    FIXED_QUERY_INDPTR,
    TOTAL_ROWS,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_TILES: tl.constexpr,
    GATHER_TILE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WINDOW_LEFT,
    QUERY_TILE_SIZE: tl.constexpr,
    MIN_CHUNK: tl.constexpr = 128,
    SHORT_TOKENS: tl.constexpr = 0,
    MIN_CHUNK_2: tl.constexpr = 128,
    SHORT_TOKENS_2: tl.constexpr = 0,
    MIN_CHUNK_3: tl.constexpr = 128,
    SHORT_TOKENS_3: tl.constexpr = 0,
    TRIM_WINDOW: tl.constexpr = False,
    TRIM_WINDOW_LEFT=-1,
    WINDOWED: tl.constexpr = False,
):
    """DRAFT prepare in one launch: page CSR, page gather and Split-KV schedule.

    Grid is BATCH*PAGE_TILES programs. Every program derives the (tiny) per-row
    lengths from the token CSR; program 0 also writes the compact query/page CSR,
    last-page lengths and the provider schedule. Slots -1 are masked before any
    token-CSR read. Attention queries use the fixed [B*Q] layout: inactive rows
    receive no tiles, and the merge kernel zero-fills their empty output ranges.
    """
    pid = tl.program_id(0)
    request = tl.arange(0, REQUEST_BLOCK)
    in_batch = request < BATCH
    slot = tl.load(SLOTS + request, in_batch, other=-1)
    active = in_batch & (slot != -1)
    begin = tl.load(TOKEN_INDPTR + request, active, other=0)
    length = tl.load(TOKEN_INDPTR + request + 1, active, other=0) - begin
    if TRIM_WINDOW:
        # Custom masks address the retained CSR, not the scheduled per-tile span.
        # Drop only whole pages preceding the union of all query windows.
        active = active & (length > 0)
        if WINDOWED:
            drop = tl.maximum(length.to(tl.int64) - QUERY - TRIM_WINDOW_LEFT, 0) // PAGE * PAGE
            begin = begin.to(tl.int64) + drop
            length = (length.to(tl.int64) - drop).to(tl.int32)
    pages = tl.cdiv(length, PAGE)
    page_ends = tl.cumsum(pages, 0)

    mine = request == pid // PAGE_TILES
    count = tl.sum(tl.where(mine, pages, 0), 0)
    token_start = tl.sum(tl.where(mine, begin, 0), 0).to(tl.int64)
    page_start = tl.sum(tl.where(mine, page_ends - pages, 0), 0)
    page = (pid % PAGE_TILES) * GATHER_TILE + tl.arange(0, GATHER_TILE)
    keep = page < count
    # Admission proves each logical page starts at physical offset zero.
    token = tl.load(TOKEN_INDICES + token_start + page.to(tl.int64) * PAGE, keep, other=0)
    tl.store(PAGE_INDICES + page_start + page, token // PAGE, keep)

    if pid == 0:
        tl.store(QUERY_INDPTR, 0)
        tl.store(QUERY_INDPTR + request + 1, tl.cumsum(tl.where(active, QUERY, 0), 0), in_batch)
        tl.store(PAGE_INDPTR, 0)
        tl.store(PAGE_INDPTR + request + 1, page_ends, in_batch)
        tl.store(
            LAST_PAGE_LENGTHS + request, tl.where(active, (length - 1) % PAGE + 1, 0), in_batch
        )
        tl.store(FIXED_QUERY_INDPTR + request, request * QUERY, in_batch)
        tl.store(FIXED_QUERY_INDPTR + BATCH, BATCH * QUERY)
        tl.store(TOTAL_ROWS, BATCH * QUERY)
        if WINDOW_LEFT >= 0:
            span: tl.constexpr = max(QUERY, QUERY_TILE_SIZE) + (PAGE - 1 if QUERY > 32 else 0)
            pages = tl.minimum(pages, tl.cdiv(WINDOW_LEFT + span, PAGE))
        _write_schedule(
            pages,
            REQUESTS,
            QUERY_TILES,
            KV_TILES,
            OUTPUT_OFFSETS,
            MERGE_OFFSETS,
            CHUNK_SIZE,
            VALID,
            BATCH,
            QUERY,
            PAGE,
            QUERY_TILE_COUNT,
            MAX_CTAS,
            PADDED_CTAS,
            REQUEST_BLOCK,
            TILE_BLOCK,
            MERGE_BLOCK,
            SEARCH_STEPS,
            MIN_CHUNK,
            SHORT_TOKENS,
            MIN_CHUNK_2,
            SHORT_TOKENS_2,
            MIN_CHUNK_3,
            SHORT_TOKENS_3,
            WINDOW_LEFT,
        )


@tr.jit
def _schedule_pages(
    KV_INDPTR,
    request,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW_LEFT,
    QUERY_TILE_SIZE: tl.constexpr,
    DRAFT: tl.constexpr,
):
    pages = tl.load(KV_INDPTR + request + 1, request < BATCH, other=0).to(tl.int64) - tl.load(
        KV_INDPTR + request, request < BATCH, other=0
    ).to(tl.int64)
    if WINDOW_LEFT >= 0:
        span: tl.constexpr = max(QUERY, QUERY_TILE_SIZE) + (PAGE - 1 if QUERY > 32 else 0)
        pages = tl.minimum(pages, tl.cdiv(WINDOW_LEFT.to(tl.int64) + span, PAGE))
    if not DRAFT:
        pages = tl.where(request < BATCH, tl.maximum(pages, 1), 0)
    return pages


@tr.jit
def _write_schedule_chunked(
    KV_INDPTR,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WINDOW_LEFT,
    QUERY_TILE_SIZE: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    SHORT_TOKENS: tl.constexpr,
    MIN_CHUNK_2: tl.constexpr,
    SHORT_TOKENS_2: tl.constexpr,
    MIN_CHUNK_3: tl.constexpr,
    SHORT_TOKENS_3: tl.constexpr,
    DRAFT: tl.constexpr,
):
    # Independent implementation of the native FA2 schedule contract documented
    # in scheduler.cuh PrefillSplitQOKVIndptr / PrefillBinarySearchKVChunkSize.
    # Reuse the native output CSR as scratch. Every vector is <=1024 lanes;
    # int64 intermediates avoid imposing artificial admission bounds on Q/B.
    lane = tl.arange(0, 1024).to(tl.int64)
    high = tl.full((), 1, tl.int64)
    for base in range(tl.cdiv(BATCH, 1024)):
        pages = _schedule_pages(
            KV_INDPTR, base * 1024 + lane, BATCH, QUERY, PAGE, WINDOW_LEFT, QUERY_TILE_SIZE, DRAFT
        )
        high = tl.maximum(high, tl.max(pages, 0))
    low = tl.full((), max(128 // PAGE, 1), tl.int64)
    if high * PAGE <= SHORT_TOKENS_3:
        low = tl.full((), max(MIN_CHUNK_3 // PAGE, 1), tl.int64)
    if high * PAGE <= SHORT_TOKENS_2:
        low = tl.full((), max(MIN_CHUNK_2 // PAGE, 1), tl.int64)
    if high * PAGE <= SHORT_TOKENS:
        low = tl.full((), max(MIN_CHUNK // PAGE, 1), tl.int64)
    if QUERY > 32:
        if WINDOW_LEFT >= 0:
            # Match native prefill's hard-coded W-1+tile output stride: with
            # one chunk, both that stride and our full-query union use one.
            low = tl.maximum(low, high)
    while low < high:
        middle = (low + high) // 2
        total = tl.full((), 0, tl.int64)
        for base in range(tl.cdiv(BATCH, 1024)):
            pages = _schedule_pages(
                KV_INDPTR,
                base * 1024 + lane,
                BATCH,
                QUERY,
                PAGE,
                WINDOW_LEFT,
                QUERY_TILE_SIZE,
                DRAFT,
            )
            total += tl.sum(tl.cdiv(pages, middle), 0)
        exceeds = total * QUERY_TILE_COUNT > MAX_CTAS
        low = tl.where(exceeds, middle + 1, low)
        high = tl.where(exceeds, high, middle)
    tl.store(CHUNK_SIZE, low * PAGE)
    total_chunks = tl.full((), 0, tl.int64)
    for base in range(tl.cdiv(BATCH, 1024)):
        request = base * 1024 + lane
        pages = _schedule_pages(
            KV_INDPTR, request, BATCH, QUERY, PAGE, WINDOW_LEFT, QUERY_TILE_SIZE, DRAFT
        )
        chunks = tl.cdiv(pages, low)
        ends = total_chunks + tl.cumsum(chunks, 0)
        tl.store(OUTPUT_OFFSETS + request, (ends - chunks) * QUERY, request < BATCH)
        total_chunks += tl.sum(chunks, 0)
    tl.store(OUTPUT_OFFSETS + BATCH, total_chunks * QUERY)
    tl.debug_barrier()
    for base in range(tl.cdiv(PADDED_CTAS, 1024)):
        tile = base * 1024 + lane
        left = tl.full((1024,), 0, tl.int64)
        right = tl.full((1024,), BATCH, tl.int64)
        for _ in range(SEARCH_STEPS):
            middle = (left + right) // 2
            end = tl.load(OUTPUT_OFFSETS + tl.minimum(middle + 1, BATCH)).to(tl.int64)
            advance = (middle < BATCH) & (tile >= end // QUERY * QUERY_TILE_COUNT)
            left = tl.where(advance, middle + 1, left)
            right = tl.where(advance, right, middle)
        request = tl.minimum(left, BATCH - 1)
        start = tl.load(OUTPUT_OFFSETS + request).to(tl.int64) // QUERY
        end = tl.load(OUTPUT_OFFSETS + request + 1).to(tl.int64) // QUERY
        chunks = tl.maximum(end - start, 1)
        local = tile - start * QUERY_TILE_COUNT
        valid = tile < total_chunks * QUERY_TILE_COUNT
        tl.store(REQUESTS + tile, tl.where(valid, left, 0), tile < PADDED_CTAS)
        tl.store(QUERY_TILES + tile, tl.where(valid, local // chunks, 0), tile < PADDED_CTAS)
        tl.store(KV_TILES + tile, tl.where(valid, local % chunks, 0), tile < PADDED_CTAS)
        tl.store(VALID + tile, valid, tile < PADDED_CTAS)
    for base in range(tl.cdiv(BATCH * QUERY, 1024)):
        output = base * 1024 + lane
        request = tl.minimum(output // QUERY, BATCH - 1)
        start = tl.load(OUTPUT_OFFSETS + request).to(tl.int64)
        end = tl.load(OUTPUT_OFFSETS + request + 1).to(tl.int64)
        merge = start + output % QUERY * ((end - start) // QUERY)
        tl.store(MERGE_OFFSETS + output, merge, output < BATCH * QUERY)
    tl.store(MERGE_OFFSETS + BATCH * QUERY, total_chunks * QUERY)


@tr.jit(do_not_specialize=["WINDOW_LEFT"], do_not_specialize_on_alignment=["WINDOW_LEFT"])
def prepare_split_kv_chunked(
    KV_INDPTR,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WINDOW_LEFT=-1,
    QUERY_TILE_SIZE: tl.constexpr = 0,
    MIN_CHUNK: tl.constexpr = 128,
    SHORT_TOKENS: tl.constexpr = 0,
    MIN_CHUNK_2: tl.constexpr = 128,
    SHORT_TOKENS_2: tl.constexpr = 0,
    MIN_CHUNK_3: tl.constexpr = 128,
    SHORT_TOKENS_3: tl.constexpr = 0,
):
    _write_schedule_chunked(
        KV_INDPTR,
        REQUESTS,
        QUERY_TILES,
        KV_TILES,
        OUTPUT_OFFSETS,
        MERGE_OFFSETS,
        CHUNK_SIZE,
        VALID,
        BATCH,
        QUERY,
        PAGE,
        QUERY_TILE_COUNT,
        MAX_CTAS,
        PADDED_CTAS,
        SEARCH_STEPS,
        WINDOW_LEFT,
        QUERY_TILE_SIZE,
        MIN_CHUNK,
        SHORT_TOKENS,
        MIN_CHUNK_2,
        SHORT_TOKENS_2,
        MIN_CHUNK_3,
        SHORT_TOKENS_3,
        False,
    )


@tr.jit(
    do_not_specialize=["WINDOW_LEFT", "TRIM_WINDOW_LEFT"],
    do_not_specialize_on_alignment=["WINDOW_LEFT", "TRIM_WINDOW_LEFT"],
)
def prepare_draft_chunked(
    SLOTS,
    TOKEN_INDPTR,
    TOKEN_INDICES,
    QUERY_INDPTR,
    PAGE_INDPTR,
    PAGE_INDICES,
    LAST_PAGE_LENGTHS,
    FIXED_QUERY_INDPTR,
    TOTAL_ROWS,
    REQUESTS,
    QUERY_TILES,
    KV_TILES,
    OUTPUT_OFFSETS,
    MERGE_OFFSETS,
    CHUNK_SIZE,
    VALID,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_TILES: tl.constexpr,
    GATHER_TILE: tl.constexpr,
    QUERY_TILE_COUNT: tl.constexpr,
    MAX_CTAS: tl.constexpr,
    PADDED_CTAS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WINDOW_LEFT,
    QUERY_TILE_SIZE: tl.constexpr,
    MIN_CHUNK: tl.constexpr = 128,
    SHORT_TOKENS: tl.constexpr = 0,
    MIN_CHUNK_2: tl.constexpr = 128,
    SHORT_TOKENS_2: tl.constexpr = 0,
    MIN_CHUNK_3: tl.constexpr = 128,
    SHORT_TOKENS_3: tl.constexpr = 0,
    TRIM_WINDOW: tl.constexpr = False,
    TRIM_WINDOW_LEFT=-1,
    WINDOWED: tl.constexpr = False,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, 1024).to(tl.int64)
    mine = pid // PAGE_TILES
    page_prefix = tl.full((), 0, tl.int64)
    query_prefix = tl.full((), 0, tl.int64)
    # All programs compute the prefix for their gather; only program zero
    # publishes CSR and schedule. No cross-program synchronization is needed.
    for base in range(tl.cdiv(BATCH, 1024)):
        request = base * 1024 + lane
        slot = tl.load(SLOTS + request, request < BATCH, other=-1)
        active = (request < BATCH) & (slot != -1)
        begin = tl.load(TOKEN_INDPTR + request, active, other=0).to(tl.int64)
        length = tl.load(TOKEN_INDPTR + request + 1, active, other=0).to(tl.int64) - begin
        if TRIM_WINDOW:
            # Custom masks address the retained CSR, not the scheduled per-tile span.
            # Drop only whole pages preceding the union of all query windows.
            active = active & (length > 0)
            if WINDOWED:
                drop = tl.maximum(length.to(tl.int64) - QUERY - TRIM_WINDOW_LEFT, 0) // PAGE * PAGE
                begin = begin.to(tl.int64) + drop
                length = length - drop
        pages = tl.cdiv(length, PAGE)
        ends = page_prefix + tl.cumsum(pages, 0)
        queries = query_prefix + tl.cumsum(tl.where(active, QUERY, 0).to(tl.int64), 0)
        if pid == 0:
            tl.store(QUERY_INDPTR + request + 1, queries, request < BATCH)
            tl.store(PAGE_INDPTR + request + 1, ends, request < BATCH)
            tl.store(
                LAST_PAGE_LENGTHS + request,
                tl.where(active, (length - 1) % PAGE + 1, 0),
                request < BATCH,
            )
            tl.store(FIXED_QUERY_INDPTR + request, request * QUERY, request < BATCH)
        if (mine >= base * 1024) & (mine < (base + 1) * 1024):
            count = tl.sum(tl.where(request == mine, pages, 0), 0)
            start = tl.sum(tl.where(request == mine, begin, 0), 0)
            dest = tl.sum(tl.where(request == mine, ends - pages, 0), 0)
            page = (pid % PAGE_TILES).to(tl.int64) * GATHER_TILE + tl.arange(0, GATHER_TILE)
            token = tl.load(TOKEN_INDICES + start + page * PAGE, page < count, other=0)
            tl.store(PAGE_INDICES + dest + page, token // PAGE, page < count)
        page_prefix += tl.sum(pages, 0)
        query_prefix += tl.sum(tl.where(active, QUERY, 0).to(tl.int64), 0)
    if pid == 0:
        tl.store(QUERY_INDPTR, 0)
        tl.store(PAGE_INDPTR, 0)
        tl.store(FIXED_QUERY_INDPTR + BATCH, BATCH * QUERY)
        tl.store(TOTAL_ROWS, BATCH * QUERY)
        tl.debug_barrier()
        _write_schedule_chunked(
            PAGE_INDPTR,
            REQUESTS,
            QUERY_TILES,
            KV_TILES,
            OUTPUT_OFFSETS,
            MERGE_OFFSETS,
            CHUNK_SIZE,
            VALID,
            BATCH,
            QUERY,
            PAGE,
            QUERY_TILE_COUNT,
            MAX_CTAS,
            PADDED_CTAS,
            SEARCH_STEPS,
            WINDOW_LEFT,
            QUERY_TILE_SIZE,
            MIN_CHUNK,
            SHORT_TOKENS,
            MIN_CHUNK_2,
            SHORT_TOKENS_2,
            MIN_CHUNK_3,
            SHORT_TOKENS_3,
            True,
        )
