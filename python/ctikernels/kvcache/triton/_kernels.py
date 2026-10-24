# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Row copy kernel over a device table of (address, row stride, row units, rows, word class,
first chunk) per tensor; units are the class's word (16-byte classes move 8-byte units in
aligned pairs)."""

import triton
import triton.language as tl

FIELDS = tl.constexpr(6)
VEC16, WORD8, WORD4, WORD2, WORD1 = (tl.constexpr(i) for i in range(5))


@triton.jit(
    do_not_specialize=["count"],
    do_not_specialize_on_alignment=["table", "sources", "targets"],
)
def copy_rows(table, sources, targets, count, LOG: tl.constexpr, BLOCK: tl.constexpr):
    """Program (chunk, copy) moves BLOCK units of one row of the tensor owning ``chunk``;
    invalid rows are skipped."""
    chunk = tl.program_id(0).to(tl.int64)
    copy = tl.program_id(1)
    # The tensor whose chunks start at or before this one: first[lo] <= chunk < first[hi].
    lo = 0
    hi = count
    for _ in tl.static_range(LOG):
        mid = (lo + hi) // 2
        below = tl.load(table + mid * FIELDS + 5) <= chunk
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid)
    entry = table + lo * FIELDS
    base, stride = tl.load(entry), tl.load(entry + 1)
    units, rows = tl.load(entry + 2), tl.load(entry + 3)
    word, first = tl.load(entry + 4), tl.load(entry + 5)
    src = tl.load(sources + copy).to(tl.int64)
    dst = tl.load(targets + copy).to(tl.int64)
    start = (chunk - first) * BLOCK
    if (src >= 0) & (dst >= 0) & (src < rows) & (dst < rows):
        offsets = start + tl.arange(0, BLOCK).to(tl.int64)
        if word == VEC16:
            # Every address, stride and row size is a multiple of 16 bytes (2 units).
            vec = tl.multiple_of(base, 16).to(tl.pointer_type(tl.int64))
            pitch = tl.multiple_of(stride, 2)
            width = tl.multiple_of(units, 2)
            inside = offsets < width
            value = tl.load(vec + src * pitch + offsets, mask=inside)
            tl.store(vec + dst * pitch + offsets, value, mask=inside)
        elif word == WORD8:
            words8 = base.to(tl.pointer_type(tl.int64))
            inside = offsets < units
            tl.store(
                words8 + dst * stride + offsets,
                tl.load(words8 + src * stride + offsets, mask=inside),
                mask=inside,
            )
        elif word == WORD4:
            words4 = base.to(tl.pointer_type(tl.int32))
            inside = offsets < units
            tl.store(
                words4 + dst * stride + offsets,
                tl.load(words4 + src * stride + offsets, mask=inside),
                mask=inside,
            )
        elif word == WORD2:
            words2 = base.to(tl.pointer_type(tl.int16))
            inside = offsets < units
            tl.store(
                words2 + dst * stride + offsets,
                tl.load(words2 + src * stride + offsets, mask=inside),
                mask=inside,
            )
        else:
            words1 = base.to(tl.pointer_type(tl.int8))
            inside = offsets < units
            tl.store(
                words1 + dst * stride + offsets,
                tl.load(words1 + src * stride + offsets, mask=inside),
                mask=inside,
            )
