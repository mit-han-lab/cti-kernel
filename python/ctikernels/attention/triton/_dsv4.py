# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton glue of DeepSeek-V4 attention over the packed 584-byte KV rows.

Row format (``attention.inference.DSV4Pages``): a page of ``R`` rows holds ``R`` data records
of 576 bytes (448 E4M3 non-rotary channels, then 64 BF16 rotary channels), then ``R`` scale
records of 8 bytes (7 UE8M0 exponents, one per 64 non-rotary channels, and a zero pad byte).
Rotary channels rotate in adjacent pairs; tables are FP32 ``[positions, 64]`` rows holding
32 cosines then 32 sines.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NOPE = tl.constexpr(448)
ROPE = tl.constexpr(64)
DIM = tl.constexpr(512)
DATA_BYTES = tl.constexpr(576)
SCALE_BYTES = tl.constexpr(8)
_RECIPROCAL_FP8_MAX = tl.constexpr(1.0 / 448.0)


@triton.jit
def _ue8m0_ceil(scale):
    """Biased exponent of the next power of two at or above ``scale`` (> 0, normal)."""
    bits = scale.to(tl.int32, bitcast=True)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@triton.jit
def _store_row(
    x_ptr,
    w_ptr,
    table_ptr,
    pos,
    row,
    cache_ptr,
    cache16_ptr,
    eps,
    rows_per_page,
    page_bytes,
    HAS_WEIGHT: tl.constexpr,
    NORM: tl.constexpr,
):
    """One 512-channel row: RMS norm (x weight), rotary, E4M3 + UE8M0 pack into ``row``."""
    blocks = tl.arange(0, 8)[:, None]
    lanes = tl.arange(0, 64)[None, :]
    x = tl.load(x_ptr + blocks * 64 + lanes).to(tl.float32)
    half = tl.arange(0, 32)
    even = tl.load(x_ptr + NOPE + 2 * half).to(tl.float32)
    odd = tl.load(x_ptr + NOPE + 2 * half + 1).to(tl.float32)
    if NORM:
        rs = tl.rsqrt(tl.sum(tl.sum(x * x, 1), 0) / DIM + eps)
        x = x * rs
        even = even * rs
        odd = odd * rs
    if HAS_WEIGHT:
        x = x * tl.load(w_ptr + blocks * 64 + lanes).to(tl.float32)
        even = even * tl.load(w_ptr + NOPE + 2 * half).to(tl.float32)
        odd = odd * tl.load(w_ptr + NOPE + 2 * half + 1).to(tl.float32)
    cos = tl.load(table_ptr + pos * ROPE + half)
    sin = tl.load(table_ptr + pos * ROPE + 32 + half)
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos

    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    exponent = _ue8m0_ceil(amax * _RECIPROCAL_FP8_MAX)
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(x * inverse[:, None], -448.0, 448.0).to(tl.float8e4nv)

    page = row // rows_per_page
    slot = row % rows_per_page
    data = page * page_bytes + slot * DATA_BYTES
    tl.store(cache_ptr + data + blocks * 64 + lanes, q.to(tl.uint8, bitcast=True), mask=blocks < 7)
    rope16 = (data + NOPE) // 2
    tl.store(cache16_ptr + rope16 + 2 * half, rot_even.to(tl.bfloat16).to(tl.int16, bitcast=True))
    tl.store(
        cache16_ptr + rope16 + 2 * half + 1, rot_odd.to(tl.bfloat16).to(tl.int16, bitcast=True)
    )
    scale_at = page * page_bytes + rows_per_page * DATA_BYTES + slot * SCALE_BYTES
    exponents = tl.where(tl.arange(0, 8) < 7, exponent, 0).to(tl.uint8)
    tl.store(cache_ptr + scale_at + tl.arange(0, 8), exponents)


@triton.jit
def _store_rows_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    table_ptr,
    pos_ptr,
    row_ptr,
    cache_ptr,
    cache16_ptr,
    eps,
    rows_per_page,
    page_bytes,
    HAS_WEIGHT: tl.constexpr,
    NORM: tl.constexpr,
):
    t = tl.program_id(0)
    row = tl.load(row_ptr + t).to(tl.int64)
    if row >= 0:
        _store_row(
            x_ptr + t * x_stride,
            w_ptr,
            table_ptr,
            tl.load(pos_ptr + t),
            row,
            cache_ptr,
            cache16_ptr,
            eps,
            rows_per_page,
            page_bytes,
            HAS_WEIGHT,
            NORM,
        )


def store_rows(x, weight, table, positions, rows, cache, *, rows_per_page, page_bytes, eps, norm):
    """Write ``x [T, 512]`` (optionally RMS-normalized and weighted) as packed rows."""
    if x.shape[0] == 0:
        return
    _store_rows_kernel[(x.shape[0],)](
        x,
        x.stride(0),
        weight if weight is not None else x,
        table,
        positions,
        rows,
        cache,
        cache.view(torch.int16),
        float(eps),
        rows_per_page,
        page_bytes,
        HAS_WEIGHT=weight is not None,
        NORM=norm,
        num_warps=4,
    )


@triton.jit
def _prologue_kernel(
    q_ptr,
    q_stride_t,
    q_stride_h,
    out_ptr,
    out_stride_t,
    out_stride_h,
    heads,
    kv_ptr,
    kv_stride,
    w_ptr,
    table_ptr,
    pos_ptr,
    row_ptr,
    cache_ptr,
    cache16_ptr,
    eps,
    rows_per_page,
    page_bytes,
    BLOCK_H: tl.constexpr,
):
    """Programs (t, b < blocks): weightless RMS norm + rotary of BLOCK_H query heads of token t;
    program (t, blocks): the token's KV row, normalized, rotated and packed."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    pos = tl.load(pos_ptr + t)
    if b * BLOCK_H < heads:
        head = b * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
        live = head < heads
        channel = tl.arange(0, 512)[None, :]
        src = q_ptr + t * q_stride_t + head * q_stride_h
        dst = out_ptr + t * out_stride_t + head * out_stride_h
        x = tl.load(src + channel, mask=live, other=0.0).to(tl.float32)
        rs = tl.rsqrt(tl.sum(x * x, 1) / DIM + eps)[:, None]
        tl.store(dst + channel, (x * rs).to(tl.bfloat16), mask=live & (channel < NOPE))
        half = tl.arange(0, 32)[None, :]
        even = tl.load(src + NOPE + 2 * half, mask=live, other=0.0).to(tl.float32) * rs
        odd = tl.load(src + NOPE + 2 * half + 1, mask=live, other=0.0).to(tl.float32) * rs
        cos = tl.load(table_ptr + pos * ROPE + half)
        sin = tl.load(table_ptr + pos * ROPE + 32 + half)
        tl.store(dst + NOPE + 2 * half, (even * cos - odd * sin).to(tl.bfloat16), mask=live)
        tl.store(dst + NOPE + 2 * half + 1, (even * sin + odd * cos).to(tl.bfloat16), mask=live)
    else:
        row = tl.load(row_ptr + t).to(tl.int64)
        if row >= 0:
            _store_row(
                kv_ptr + t * kv_stride,
                w_ptr,
                table_ptr,
                pos,
                row,
                cache_ptr,
                cache16_ptr,
                eps,
                rows_per_page,
                page_bytes,
                True,
                True,
            )


def prologue(q, out, kv, weight, table, positions, rows, cache, *, rows_per_page, page_bytes, eps):
    tokens, heads = q.shape[:2]
    if tokens == 0:
        return
    block = 4  # 4 heads a program; 1 warp for decode, 8 for prefill (RTX 5090 + GB10 sweeps)
    warps = 1 if tokens <= 8 else 8
    _prologue_kernel[(tokens, triton.cdiv(heads, block) + 1)](
        q,
        q.stride(0),
        q.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        heads,
        kv,
        kv.stride(0),
        weight,
        table,
        positions,
        rows,
        cache,
        cache.view(torch.int16),
        float(eps),
        rows_per_page,
        page_bytes,
        BLOCK_H=block,
        num_warps=warps,
    )


@triton.jit
def _gather_block(
    cache_ptr,
    cache16_ptr,
    idx_ptr,
    count,
    k0,
    rows_per_page,
    page_bytes,
    slots,
    scratch_ptr,
    ROWS: tl.constexpr,
):
    """Dequantize up to ``ROWS`` listed rows ``idx[k0:]`` (the first ``count`` count) into the
    scratch rows ``slots``; returns which slots got a row."""
    k = k0 + tl.arange(0, ROWS)
    valid = k < count
    row = tl.load(idx_ptr + k, mask=valid, other=-1).to(tl.int64)
    valid = valid & (row >= 0)
    page = row // rows_per_page
    slot = row % rows_per_page
    data = (page * page_bytes + slot * DATA_BYTES)[:, None]
    channel = tl.arange(0, 512)[None, :]
    nope = channel < NOPE
    nope_mask = valid[:, None] & nope
    raw = tl.load(cache_ptr + data + channel, mask=nope_mask, other=0)
    scale_at = (page * page_bytes + rows_per_page * DATA_BYTES + slot * SCALE_BYTES)[:, None]
    exponent = tl.load(cache_ptr + scale_at + channel // 64, mask=nope_mask, other=127)
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale
    rope = tl.load(
        cache16_ptr + (data + NOPE) // 2 + (channel - NOPE),
        mask=valid[:, None] & (channel >= NOPE),
        other=0,
    )
    values = tl.where(nope, values, rope.to(tl.bfloat16, bitcast=True).to(tl.float32))
    tl.store(
        scratch_ptr + slots.to(tl.int64)[:, None] * DIM + channel,
        values.to(tl.bfloat16),
        mask=valid[:, None],
    )
    return valid


@triton.jit
def _gather_kernel(
    swa_ptr,
    swa16_ptr,
    swa_idx_ptr,
    swa_idx_stride,
    swa_len_ptr,
    swa_rows_per_page,
    swa_page_bytes,
    swa_width,
    ext_ptr,
    ext16_ptr,
    ext_idx_ptr,
    ext_idx_stride,
    ext_len_ptr,
    ext_rows_per_page,
    ext_page_bytes,
    ext_width,
    scratch_ptr,
    table_ptr,
    seq_ptr,
    topk_ptr,
    width,
    SWA_SLOTS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    ROWS: tl.constexpr,
):
    t = tl.program_id(0)
    j0 = tl.program_id(1) * ROWS
    swa_count = tl.minimum(tl.load(swa_len_ptr + t), swa_width)
    ext_count = 0
    if HAS_EXTRA:
        ext_count = tl.minimum(tl.load(ext_len_ptr + t), ext_width)
    if j0 == 0:
        tl.store(seq_ptr + t, tl.maximum(swa_count, 1))
        tl.store(topk_ptr + t, SWA_SLOTS + ext_count)
    slots = t * width + j0 + tl.arange(0, ROWS)
    if j0 < SWA_SLOTS:  # ROWS divides SWA_SLOTS: a block is all window or all extra
        valid = _gather_block(
            swa_ptr,
            swa16_ptr,
            swa_idx_ptr + t * swa_idx_stride,
            swa_count,
            j0,
            swa_rows_per_page,
            swa_page_bytes,
            slots,
            scratch_ptr,
            ROWS,
        )
    else:
        valid = _gather_block(
            ext_ptr,
            ext16_ptr,
            ext_idx_ptr + t * ext_idx_stride,
            ext_count,
            j0 - SWA_SLOTS,
            ext_rows_per_page,
            ext_page_bytes,
            slots,
            scratch_ptr,
            ROWS,
        )
    inside = j0 + tl.arange(0, ROWS) < width
    tl.store(table_ptr + slots, tl.where(valid, slots, -1), mask=inside)


def gather_rows(swa, extra, scratch, table, seq_lens, topk_lens, *, swa_slots, sms):
    """Dequantize every row a query reads into ``scratch [T * width, 512]`` BF16.

    ``swa``/``extra`` are ``(cache, indices, lengths, rows_per_page, page_bytes)``; ``table
    [T, width]`` receives each slot's scratch row, or -1 where the query reads nothing;
    ``seq_lens``/``topk_lens [T]`` the window and total lengths the dense decode takes.
    Rows per program grow with the work: one row while every SM still gets several programs
    (short reads are latency-bound), up to 16 at long context.
    """
    tokens, width = table.shape
    if tokens == 0:
        return
    swa_cache, swa_idx, swa_len, swa_rows, swa_bytes = swa
    if extra is None:
        ext_cache, ext_idx, ext_len, ext_rows, ext_bytes = swa_cache, swa_idx, swa_len, 1, 1
        ext_width = 0
    else:
        ext_cache, ext_idx, ext_len, ext_rows, ext_bytes = extra
        ext_width = ext_idx.shape[1]
    rows = min(16, triton.next_power_of_2(max(1, tokens * width // (4 * sms))))
    _gather_kernel[(tokens, triton.cdiv(width, rows))](
        swa_cache,
        swa_cache.view(torch.int16),
        swa_idx,
        swa_idx.stride(0),
        swa_len,
        swa_rows,
        swa_bytes,
        swa_idx.shape[1],
        ext_cache,
        ext_cache.view(torch.int16),
        ext_idx,
        ext_idx.stride(0),
        ext_len,
        ext_rows,
        ext_bytes,
        ext_width,
        scratch,
        table,
        seq_lens,
        topk_lens,
        width,
        SWA_SLOTS=swa_slots,
        HAS_EXTRA=extra is not None,
        ROWS=rows,
        num_warps=4,
    )


@triton.jit
def _dequant_rows_kernel(
    cache_ptr, cache16_ptr, row_ptr, count, rows_per_page, page_bytes, out_ptr, ROWS: tl.constexpr
):
    k0 = tl.program_id(0) * ROWS
    _gather_block(
        cache_ptr,
        cache16_ptr,
        row_ptr,
        count,
        k0,
        rows_per_page,
        page_bytes,
        k0 + tl.arange(0, ROWS),
        out_ptr,
        ROWS,
    )


def dequant_rows(cache, rows, out, *, rows_per_page, page_bytes):
    """BF16 ``out [N, 512]`` of packed rows ``rows [N]`` (all valid)."""
    count = rows.numel()
    if count == 0:
        return
    block = 16
    _dequant_rows_kernel[(triton.cdiv(count, block),)](
        cache,
        cache.view(torch.int16),
        rows,
        count,
        rows_per_page,
        page_bytes,
        out,
        ROWS=block,
        num_warps=4,
    )


# Extend plans and page-table row maps. Engine page tables hold one id per 256-token page for
# every pool: window row = page * 256 + x % 256, c4 row = page * 64 + e % 64 (entry e of page
# e // 64), c128 row = page * 2 + e % 2 (entry e of page e // 2).
PAGE_TOKENS = tl.constexpr(256)


@triton.jit
def _pool_row(table_ptr, table_stride, request, item, per_page, mask):
    # Masked lanes (negative window positions, entries past the table) read no table cell.
    row = table_ptr + request.to(tl.int64) * table_stride
    page = tl.load(row + item // per_page, mask=mask, other=0)
    return page.to(tl.int64) * per_page + item % per_page


@triton.jit
def _query_rows_kernel(
    table_ptr,
    table_stride,
    pos_ptr,
    req_ptr,
    first_ptr,
    base_ptr,
    out_ptr,
    out_stride,
    col0,
    width,
    offset,
    WINDOW: tl.constexpr,
    STAGED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One query's window rows (positions p, p-1, ..., p-127: valid ones first) or its c128
    entries 0..(p+1)//128-1, as pool rows or as staged local rows (base[r] + offset + k)."""
    t = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(pos_ptr + t)
    r = tl.load(req_ptr + t)
    if WINDOW:
        item = p - j
        valid = (item >= 0) & (j < 128)
        per_page = PAGE_TOKENS
    else:
        item = j.to(tl.int64)
        valid = j < (p + 1) // 128
        per_page = 2
    valid &= j < width
    if STAGED:
        local = tl.load(base_ptr + r) + offset
        if WINDOW:
            local = local + item - tl.load(first_ptr + r)
        else:
            local = local + item
        value = local
    else:
        value = _pool_row(table_ptr, table_stride, r, item, per_page, valid)
    tl.store(
        out_ptr + t * out_stride + col0 + j, tl.where(valid, value, -1).to(tl.int32), mask=j < width
    )


@triton.jit
def _staged_rows_kernel(
    table_ptr,
    table_stride,
    base_ptr,
    first_ptr,
    requests,
    count,
    out_ptr,
    PER_PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Slot i of a staged list: request r = #(base[k + 1] <= i), its item first[r] + i - base[r]
    (window) or i - base[r] (entries), as a pool row."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < count
    r = tl.zeros([BLOCK], dtype=tl.int32)
    for k in range(requests):
        r += (tl.load(base_ptr + k + 1) <= i).to(tl.int32)
    r = tl.minimum(r, requests - 1)
    item = (i - tl.load(base_ptr + r)).to(tl.int64)
    if WINDOW:
        item += tl.load(first_ptr + r)
    page = tl.load(
        table_ptr + r.to(tl.int64) * table_stride + item // PER_PAGE, mask=valid, other=0
    )
    tl.store(out_ptr + i, (page.to(tl.int64) * PER_PAGE + item % PER_PAGE).to(tl.int32), mask=valid)


@triton.jit
def _map_raw_kernel(
    raw_ptr,
    raw_stride,
    req_ptr,
    base_ptr,
    out_ptr,
    out_stride,
    col0,
    offset,
    width,
    BLOCK: tl.constexpr,
):
    """Per-request c4 entry ids -> staged local rows (offset + base[r] + e)."""
    t = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = j < width
    raw = tl.load(raw_ptr + t * raw_stride + j, mask=mask, other=-1)
    value = tl.load(base_ptr + tl.load(req_ptr + t)) + raw + offset
    tl.store(out_ptr + t * out_stride + col0 + j, tl.where(raw >= 0, value, -1), mask=mask)


def query_rows(
    table, positions, requests, out, *, col0, width, window, first=None, base=None, offset=0
):
    tokens = positions.shape[0]
    if tokens == 0 or width == 0:
        return
    block = 128
    staged = base is not None
    _query_rows_kernel[(tokens, triton.cdiv(width, block))](
        table,
        table.stride(0),
        positions,
        requests,
        first if first is not None else positions,
        base if staged else requests,
        out,
        out.stride(0),
        col0,
        width,
        offset,
        WINDOW=window,
        STAGED=staged,
        BLOCK=block,
        num_warps=4,
    )


def staged_rows(table, base, first, requests, count, out, *, per_page, window):
    if count == 0:
        return
    block = 256
    _staged_rows_kernel[(triton.cdiv(count, block),)](
        table,
        table.stride(0),
        base,
        first if first is not None else base,
        requests,
        count,
        out,
        PER_PAGE=per_page,
        WINDOW=window,
        BLOCK=block,
        num_warps=4,
    )


def map_raw(raw, requests, base, out, *, col0, offset):
    tokens, width = raw.shape
    if tokens == 0:
        return
    block = 128
    _map_raw_kernel[(tokens, triton.cdiv(width, block))](
        raw,
        raw.stride(0),
        requests,
        base,
        out,
        out.stride(0),
        col0,
        offset,
        width,
        BLOCK=block,
        num_warps=4,
    )
