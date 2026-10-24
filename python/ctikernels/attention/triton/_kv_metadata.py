# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paged-KV CSR metadata (indptr, indices, last-page lengths) built on device in one launch."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver

GATHER_TILE = 1024
# Programs per request stride over its entries; enough to spread one long request.
_MAX_SPLIT = 32


_ARGS = (
    "PAGES",
    "ROWS",
    "LENGTHS",
    "KV_INDPTR",
    "KV_INDICES",
    "LAST_PAGE_LEN",
    "page_stride",
    "capacity",
)


# Nothing specializes on argument values, so one compiled handle serves every call of a
# geometry (pointer alignment included) and is launched without Triton's binder.
@triton.jit(do_not_specialize=list(_ARGS))
def _build(
    PAGES,
    ROWS,
    LENGTHS,
    KV_INDPTR,
    KV_INDICES,
    LAST_PAGE_LEN,
    page_stride,
    capacity,
    BATCH: tl.constexpr,
    PAGE: tl.constexpr,
    TOKENS: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    """Program (b, j): entries j, j + SPLIT, ... (in tiles) of request b. Every program
    derives b's offset from all lengths, so one launch writes the whole CSR."""
    b = tl.program_id(0)
    split = tl.program_id(1)
    requests = tl.arange(0, REQUEST_BLOCK)
    lengths = tl.load(LENGTHS + requests, mask=requests < BATCH, other=0)
    counts = lengths if TOKENS else tl.cdiv(lengths, PAGE)
    ends = tl.cumsum(counts, 0)
    count = tl.sum(tl.where(requests == b, counts, 0), 0)
    start = tl.sum(tl.where(requests == b, ends, 0), 0) - count
    if split == 0:
        length = tl.sum(tl.where(requests == b, lengths, 0), 0)
        tl.store(KV_INDPTR + b + 1, start + count)
        if b == 0:
            tl.store(KV_INDPTR, 0)
        if TOKENS:
            tl.store(LAST_PAGE_LEN + b, tl.minimum(length, 1))
        else:
            tl.store(LAST_PAGE_LEN + b, tl.where(length > 0, (length - 1) % PAGE + 1, 0))
    row = tl.load(ROWS + b).to(tl.int64) * page_stride
    for tile in range(split, tl.cdiv(count, TILE), tl.num_programs(1)):
        entry = tile * TILE + tl.arange(0, TILE)
        keep = (entry < count) & (start + entry < capacity)
        if TOKENS:
            page = tl.load(PAGES + row + entry // PAGE, mask=keep, other=0)
            value = page * PAGE + entry % PAGE
        else:
            value = tl.load(PAGES + row + entry, mask=keep, other=0)
        tl.store(KV_INDICES + start + entry, value, mask=keep)


@lru_cache(maxsize=None)
def _compiled(device, batch, split, page_size, tokens):
    """The compiled builder of one geometry, its module loaded (CUDA loads lazily)."""
    with torch.cuda.device(device):
        kernel = _build.warmup(
            *([torch.int32] * 6),
            1,
            1,
            grid=(batch, split),
            BATCH=batch,
            PAGE=page_size,
            TOKENS=tokens,
            REQUEST_BLOCK=triton.next_power_of_2(batch),
            TILE=GATHER_TILE,
            num_warps=4,
        )
        kernel._init_handles()
    return kernel


def build(
    pages, rows, lengths, kv_indptr, kv_indices, last_page_len, *, page_size, tokens, max_kv_len
):
    """Launch the builder for ``rows.numel()`` requests on the current stream."""
    batch = rows.numel()
    if batch == 0:
        kv_indptr.zero_()
        return
    entries = max_kv_len if tokens else triton.cdiv(max_kv_len, page_size)
    split = max(1, min(triton.cdiv(entries, GATHER_TILE), _MAX_SPLIT))
    device = pages.get_device()
    kernel = _compiled(device, batch, split, page_size, tokens)
    args = (
        pages,
        rows,
        lengths,
        kv_indptr,
        kv_indices,
        last_page_len,
        pages.stride(0),
        kv_indices.numel(),
        batch,
        page_size,
        tokens,
        triton.next_power_of_2(batch),
        GATHER_TILE,
    )
    grid = (batch, split, 1)
    stream = driver.active.get_current_stream(device)
    kernel.run(
        *grid, stream, kernel.function, kernel.packed_metadata,
        kernel.launch_metadata(grid, stream, *args),
        knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args,
    )  # fmt: skip
