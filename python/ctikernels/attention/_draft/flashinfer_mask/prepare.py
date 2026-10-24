# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Pack row-major request masks, little-endian within each byte, on device."""

import triton as tr
import triton.language as tl


@tr.jit(
    do_not_specialize=["WINDOW_LEFT", "STRIDE", "BYTE_TILES"],
    do_not_specialize_on_alignment=["WINDOW_LEFT", "STRIDE", "BYTE_TILES"],
)
def pack_mask(
    PAGE_INDPTR,
    LAST,
    MASK,
    MASK_INDPTR,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    STRIDE,
    BYTE_TILES,
    CAUSAL: tl.constexpr,
    WINDOWED: tl.constexpr,
    WINDOW_LEFT,
):
    pid = tl.program_id(0)
    request = pid // BYTE_TILES
    tile = pid % BYTE_TILES
    pages = tl.load(PAGE_INDPTR + request + 1) - tl.load(PAGE_INDPTR + request)
    length = tl.where(pages > 0, (pages.to(tl.int64) - 1) * PAGE + tl.load(LAST + request), 0)
    byte = tile.to(tl.int64) * 256 + tl.arange(0, 256)
    bit = tl.arange(0, 8)
    offset = byte[:, None] * 8 + bit[None, :]
    # variants.cuh addresses qo_idx * retained_kv_len + kv_idx. Query rows
    # have no byte padding; only the entire request segment is padded.
    row = offset // tl.maximum(length, 1)
    key = offset % tl.maximum(length, 1)
    position = length - QUERY + row
    visible = (offset < QUERY * length) & (byte[:, None] < STRIDE)
    if CAUSAL:
        visible = visible & (key <= position)
    if WINDOWED:
        visible = visible & (tl.abs(position - key) <= WINDOW_LEFT)
    packed = tl.sum(visible.to(tl.int32) << bit[None, :], 1).to(tl.uint8)
    tl.store(MASK + request.to(tl.int64) * STRIDE + byte, packed, byte < STRIDE)
    if tile == 0:
        tl.store(MASK_INDPTR + request, request.to(tl.int64) * STRIDE)
        if request == BATCH - 1:
            tl.store(MASK_INDPTR + BATCH, BATCH * STRIDE)
