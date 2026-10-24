# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# The expert-sorted, block-padded layout of SGLang 0.5.22 (sgl-kernel
# csrc/moe/moe_align_kernel.cu) and vLLM 0.31.0 (vllm/model_executor/layers/fused_moe/
# moe_align_block_size.py), written from that description with a stable in-expert order
# (no atomics); no code copied.
"""Triton kernels of the deterministic slot layout (``_slot_layout``)."""

import triton
import triton.language as tl


@triton.jit
def _bucket(
    a_ptr,
    live_ptr,
    map_ptr,
    p,
    P,
    E,
    G,
    NPL,
    ALIGN: tl.constexpr,
    K: tl.constexpr,
    SHARED: tl.constexpr,
    MAPPED: tl.constexpr,
    INVALID: tl.constexpr,
):
    # Each slot's bucket in [0, NPL]; NPL itself is dropped (no row).
    m = p < P
    if ALIGN:
        # Route p of topk_ids: its local expert; -1 / out-of-range / remote routes go to
        # bucket E with INVALID (blocks of expert -1), else are dropped.
        g = tl.load(a_ptr + p, mask=m, other=-1).to(tl.int32)
        ok = (g >= 0) & (g < G)
        b = g
        if MAPPED:
            b = tl.load(map_ptr + g, mask=m & ok, other=-1)
        b = tl.where(ok & (b >= 0) & (b < E), b, -1)
        if INVALID:
            b = tl.where(b >= 0, b, E)
        else:
            b = tl.where(b >= 0, b, NPL)
    else:
        # Training slot p = t * (K + SHARED) + j: local expert, E the shared slot, E + 1 a
        # remote or dead slot.
        KP1: tl.constexpr = K + SHARED
        t = p // KP1
        j = p % KP1
        routed = m & (j < K)
        b = tl.load(a_ptr + t.to(tl.int64) * K + j, mask=routed, other=0).to(tl.int32)
        if MAPPED:
            local = tl.load(map_ptr + b, mask=routed, other=-1)
            b = tl.where(local >= 0, local, E + 1)
        if SHARED:
            b = tl.where(j == K, E, b)
        b = tl.where(tl.load(live_ptr + t, mask=m, other=0) != 0, b, E + 1)
    return tl.where(m, b, NPL)


@triton.jit
def _gdc(GDC: tl.constexpr):
    # Programmatic dependent launch: wait for the previous kernel's writes before the first
    # load, then let the next kernel launch (it waits the same way for ours).
    if GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _rank(b, BLOCK: tl.constexpr):
    # Earlier slots of the same bucket in this block: slot order within an expert.
    i = tl.arange(0, BLOCK)
    same = (b[:, None] == b[None, :]) & (i[None, :] < i[:, None])
    return tl.sum(same.to(tl.int32), axis=1)


@triton.jit(do_not_specialize=["P", "E", "G", "NPL", "NCH"])
def _layout_count_kernel(
    a_ptr,
    live_ptr,
    map_ptr,
    cnt_ptr,
    meta_ptr,
    P,
    E,
    G,
    NPL,
    NCH,
    ALIGN: tl.constexpr,
    K: tl.constexpr,
    SHARED: tl.constexpr,
    MAPPED: tl.constexpr,
    INVALID: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    GDC: tl.constexpr = False,
):
    # Each chunk's bucket counts (cnt[chunk, bucket]); program 0 also clears the scan's
    # arrival counter (meta[3 NB]).
    _gdc(GDC)
    pid = tl.program_id(0)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    b = _bucket(a_ptr, live_ptr, map_ptr, p, P, E, G, NPL, ALIGN, K, SHARED, MAPPED, INVALID)
    tl.store(cnt_ptr + pid * NB + tl.arange(0, NB), tl.histogram(b, NB))
    if pid == 0:
        tl.store(meta_ptr + 3 * NB, 0)


@triton.jit(do_not_specialize=["NCH", "NPL"])
def _layout_scan_kernel(
    cnt_ptr,
    meta_ptr,
    NCH,
    NPL,
    BM: tl.constexpr,
    NB: tl.constexpr,
    EB: tl.constexpr,
    CT: tl.constexpr,
    GDC: tl.constexpr = False,
):
    # EB buckets per program: each chunk's count becomes the count of the same bucket in
    # earlier chunks (an exclusive scan down the bucket's column, CT chunks at a time); the
    # column totals go to meta[0:NB]. The last program to arrive (an atomic counter) turns
    # the totals into padded bucket starts meta[NB:2NB] and ends meta[2NB:3NB]; the values do
    # not depend on which program that is.
    _gdc(GDC)
    rows = tl.program_id(0) * EB + tl.arange(0, EB)
    carry = tl.zeros((EB,), dtype=tl.int32)
    for c0 in range(0, NCH, CT):
        cols = c0 + tl.arange(0, CT)
        q = cnt_ptr + cols[None, :] * NB + rows[:, None]
        m = (cols < NCH)[None, :]
        n = tl.load(q, mask=m, other=0)
        tl.store(q, tl.cumsum(n, axis=1) - n + carry[:, None], mask=m)
        carry += tl.sum(n, axis=1)
    tl.store(meta_ptr + rows, carry)
    if NB <= 256:
        return  # narrow rows: every place program derives the starts itself
    tl.debug_barrier()
    if tl.atomic_add(meta_ptr + 3 * NB, 1) == tl.num_programs(0) - 1:
        bins = tl.arange(0, NB)
        tot = tl.load(meta_ptr + bins, cache_modifier=".cg")
        padded = tl.where(bins < NPL, (tot + BM - 1) // BM * BM, 0)
        end = tl.cumsum(padded, axis=0)
        tl.store(meta_ptr + NB + bins, end - padded)
        tl.store(meta_ptr + 2 * NB + bins, end)


@triton.jit(do_not_specialize=["P", "NPAD", "NBLK", "NCH", "SENT", "E", "G", "NPL"])
def _layout_place_kernel(
    a_ptr,
    live_ptr,
    map_ptr,
    cnt_ptr,
    meta_ptr,
    slots_ptr,
    experts_ptr,
    inv_ptr,
    total_ptr,
    P,
    NPAD,
    NBLK,
    NCH,
    SENT,
    E,
    G,
    NPL,
    ALIGN: tl.constexpr,
    K: tl.constexpr,
    SHARED: tl.constexpr,
    MAPPED: tl.constexpr,
    INVALID: tl.constexpr,
    BM: tl.constexpr,
    BMP: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB: tl.constexpr,
    FB: tl.constexpr,
    LOGNB: tl.constexpr,
    FUSED: tl.constexpr,
    GDC: tl.constexpr = False,
):
    # Programs [0, NCH) place their chunk's slots; the rest own FB blocks each: their
    # buckets (searchsorted over the padded ends) and the sentinel in their unfilled rows.
    # The two write disjoint rows, so one launch needs no ordering.
    # FUSED (a few chunks): every program counts all slots itself (bucket rows in registers)
    # and no count or scan launch precedes this one. Otherwise the scan's per-chunk offsets,
    # totals, starts and ends are read where needed: no bucket-wide work per program.
    _gdc(GDC)
    pid = tl.program_id(0)
    if FUSED:
        bins = tl.arange(0, NB)
        tot = tl.zeros((NB,), dtype=tl.int32)
        before = tl.zeros((NB,), dtype=tl.int32)
        for c in range(0, NCH):
            q = c * BLOCK + tl.arange(0, BLOCK)
            h = tl.histogram(
                _bucket(a_ptr, live_ptr, map_ptr, q, P, E, G, NPL, ALIGN, K, SHARED, MAPPED,
                        INVALID),
                NB,
            )  # fmt: skip
            tot += h
            before += tl.where(c < pid, h, 0)
        padded = tl.where(bins < NPL, (tot + BM - 1) // BM * BM, 0)
        end = tl.cumsum(padded, axis=0)
        start = end - padded
    elif NB <= 256:
        bins = tl.arange(0, NB)
        tot = tl.load(meta_ptr + bins)
        padded = tl.where(bins < NPL, (tot + BM - 1) // BM * BM, 0)
        end = tl.cumsum(padded, axis=0)
        start = end - padded
    ROWS: tl.constexpr = FUSED or NB <= 256  # bucket rows held in registers
    if pid < NCH:
        if ROWS:
            # SUB slots at a time with a running per-bucket row.
            if FUSED:
                running = start + before
            else:
                running = start + tl.load(cnt_ptr + pid * NB + bins)
            for s in tl.static_range(BLOCK // SUB):
                p = pid * BLOCK + s * SUB + tl.arange(0, SUB)
                b = _bucket(
                    a_ptr, live_ptr, map_ptr, p, P, E, G, NPL, ALIGN, K, SHARED, MAPPED, INVALID
                )
                placed = b < NPL
                pos = tl.gather(running, b, 0) + _rank(b, SUB)
                tl.store(slots_ptr + pos, p, mask=placed)
                if not ALIGN:
                    tl.store(inv_ptr + p, tl.where(placed, pos, -1), mask=p < P)
                if s + 1 < BLOCK // SUB:
                    running += tl.histogram(b, NB)
        else:
            p = pid * BLOCK + tl.arange(0, BLOCK)
            b = _bucket(
                a_ptr, live_ptr, map_ptr, p, P, E, G, NPL, ALIGN, K, SHARED, MAPPED, INVALID
            )
            placed = b < NPL
            first = tl.load(meta_ptr + NB + b, mask=placed, other=0)
            first += tl.load(cnt_ptr + pid * NB + b, mask=placed, other=0)
            pos = first + _rank(b, BLOCK)
            tl.store(slots_ptr + pos, p, mask=placed)
            if not ALIGN:
                tl.store(inv_ptr + p, tl.where(placed, pos, -1), mask=p < P)
    else:
        # FB blocks per program; binary search for the first bucket whose padded end passes
        # the block's first row (ends never decrease).
        blk = (pid - NCH) * FB + tl.arange(0, FB)
        r0 = blk * BM
        lo = tl.zeros((FB,), dtype=tl.int32)
        hi = tl.full((FB,), NB, dtype=tl.int32)
        for _ in tl.static_range(LOGNB + 1):
            mid = tl.minimum((lo + hi) // 2, NB - 1)
            if ROWS:
                at = tl.gather(end, mid, 0)
            else:
                at = tl.load(meta_ptr + 2 * NB + mid)
            right = (at <= r0) & (lo < hi)
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right | (lo >= hi), hi, mid)
        e = tl.minimum(lo, NPL)
        # Past the last placed bucket nothing is filled (bin NPL counts dropped slots).
        ec = tl.minimum(e, NB - 1)
        if ROWS:
            filled = tl.gather(start + tot, ec, 0)
            last = tl.max(end, axis=0)
        else:
            filled = tl.load(meta_ptr + NB + ec) + tl.load(meta_ptr + ec)
            last = tl.load(meta_ptr + 3 * NB - 1)
        filled = tl.where(e < NPL, filled, 0)
        expert = e
        if ALIGN:
            expert = tl.where(e < E, e, -1)
            if pid == NCH:
                tl.store(total_ptr, last)
        tl.store(experts_ptr + blk, expert, mask=blk < NBLK)
        i = tl.arange(0, BMP)
        r = r0[:, None] + i[None, :]
        tl.store(
            slots_ptr + r,
            tl.full((FB, BMP), 0, tl.int32) + SENT,
            mask=(blk < NBLK)[:, None] & (i < BM)[None, :] & (r >= filled[:, None]) & (r < NPAD),
        )
