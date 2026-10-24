# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design learned from SGLang 72730d6 (python/sglang/kernels/ops/attention/dsv4/compress.py;
# jit csrc deepseek_v4/{c4_v2,c128_v2,fused_norm_rope_v2}.cuh); rounding semantics from
# DeepSeek-V4-Flash 60d8d707 inference/model.py Compressor. No upstream code copied.

"""Triton compressor: gated pooling of ``R`` projected tokens per entry (CSA ``R = 4`` over two
overlapping windows, HCA ``R = 128``), RMSNorm, partial RoPE at the window start and the page
store, with each request's open windows in a caller-owned ring (row ``p`` at ``p % RING``).

One program owns one request in decode (``T <= max(16, B)``; HCA splits channels over 32
programs and elects the last one for the epilogue). Prefill pools every entry in parallel in a
first launch and copies the ring tail in a second one, so no launch reads state it writes.
"""

import triton
import triton.language as tl

from ._quant import hadamard128, rotate, write_index, write_latent


@triton.jit
def _pool(V, G, OldV, OldG, Ape, slot, start, a, block,
          R: tl.constexpr, RING: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
          WIDTH: tl.constexpr, channel, DECODE: tl.constexpr = True):  # fmt: skip
    d = channel * WIDTH + tl.arange(0, WIDTH)
    i = tl.arange(0, R * C)
    p = block * R + i - (R if C == 2 else 0)
    half = i // R if C == 2 else tl.full((R,), 0, tl.int32)
    fresh = p >= start
    inp = ((a + p - start)[:, None] * C + half[:, None]) * D + d[None, :]
    old = ((slot * RING + p % RING)[:, None] * C + half[:, None]) * D + d[None, :]
    if DECODE:
        g = tl.load(tl.where(fresh[:, None], G + inp, OldG + old), p[:, None] >= 0, 0)
    else:
        g = tl.load(G + inp, fresh[:, None] & (p[:, None] >= 0), 0)
        g += tl.load(OldG + old, (~fresh[:, None]) & (p[:, None] >= 0), 0)
    ape = tl.load(Ape + ((i % R)[:, None] * C + half[:, None]) * D + d[None, :])
    g = tl.where(p[:, None] >= 0, g + ape, -float("inf"))
    mass = tl.exp(g - tl.max(g, 0)[None, :])
    if DECODE:
        v = tl.load(tl.where(fresh[:, None], V + inp, OldV + old), p[:, None] >= 0, 0)
    else:
        v = tl.load(V + inp, fresh[:, None] & (p[:, None] >= 0), 0)
        v += tl.load(OldV + old, (~fresh[:, None]) & (p[:, None] >= 0), 0)
    return tl.sum(mass * v, 0) / tl.sum(mass, 0)


@triton.jit
def _finish(pool, Pages, Cache, Weight, Rope, Slots, slot, block, b, n,
            R: tl.constexpr, D: tl.constexpr, CAP: tl.constexpr, NP: tl.constexpr,
            ROWS: tl.constexpr, PAGE: tl.constexpr):  # fmt: skip
    d = tl.arange(0, D)
    row = tl.load(Pages + slot * NP + block // ROWS).to(tl.int64) * ROWS + block % ROWS
    if D == 128:
        # Index keys: the official BF16 casts (pooled, normed, rotated), as the indexer's query.
        rounded = pool.to(tl.bfloat16).to(tl.float32)
        norm = (
            rounded
            * tl.rsqrt(tl.sum(rounded * rounded, 0) / D + 1e-6)
            * tl.load(Weight + d).to(tl.float32)
        )
        x = rotate(norm.to(tl.bfloat16).to(tl.float32), block * R, Rope, D)
        write_index(hadamard128(x.to(tl.bfloat16).to(tl.float32)), Cache, row)
    else:
        # Latent entries: as ctikernels.attention.inference.dsv4_store_kv (K-d contract): RMS
        # norm (sum of squares over 8 x 64), weight and RoPE in FP32, no intermediate rounding.
        sq = tl.reshape(pool * pool, (8, 64))
        norm = pool * tl.rsqrt(tl.sum(tl.sum(sq, 1), 0) / D + 1e-6)
        write_latent(rotate(norm * tl.load(Weight + d).to(tl.float32), block * R, Rope, D),
                     Cache, row, ROWS, PAGE)  # fmt: skip
    tl.store(Slots + b * CAP + n, row)


@triton.jit
def _copy_rows(V, G, OldV, OldG, slot, start, a, end, first,
               RING: tl.constexpr, D: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr,
               channel, ROWS: tl.constexpr = 8):  # fmt: skip
    # The newest min(T, RING) input rows enter the ring; channel owners never overlap.
    i = first + tl.arange(0, ROWS)
    p = end - 1 - i
    d = channel * WIDTH + tl.arange(0, WIDTH)
    keep = i < tl.minimum(end - start, RING)
    src = (a + p - start)[:, None] * C * D + d[None, :]
    dst = (slot * RING + p % RING)[:, None] * C * D + d[None, :]
    tl.store(OldV + dst, tl.load(V + src, keep[:, None], 0), keep[:, None])
    tl.store(OldG + dst, tl.load(G + src, keep[:, None], 0), keep[:, None])


@triton.jit
def _clear_tail(Slots, b, completed, CAP: tl.constexpr, SC: tl.constexpr):
    BLOCK: tl.constexpr = min(SC, 256)
    for first in range(completed // BLOCK * BLOCK, CAP, BLOCK):
        n = first + tl.arange(0, BLOCK)
        tl.store(Slots + b * CAP + n, -1, (n >= completed) & (n < CAP))


@triton.jit
def _header(Ids, Offsets, Starts, b):
    # Request id -1 is padding: nothing reads its state or pages and nothing is written for it
    # but rows -1. Window rows before ``start`` come from the slot's ring (written by this
    # request's earlier chunks), rows from ``start`` on from this call, negative ones are masked:
    # a request starting at 0 reads no ring row.
    ids = tl.load(Ids + b)
    real = ids >= 0
    slot = tl.where(real, ids, 0).to(tl.int64)
    start = tl.load(Starts + b)
    a, z = tl.load(Offsets + b), tl.load(Offsets + b + 1)
    return slot, start, a, z, start + z - a, real


@triton.jit
def decode_csa(V, G, OldV, OldG, Ids, Offsets, Starts, Pages, Cache, Ape, Weight,
               Rope, Slots, D: tl.constexpr, CAP: tl.constexpr, NP: tl.constexpr,
               SC: tl.constexpr, ROWS: tl.constexpr, PAGE: tl.constexpr):  # fmt: skip
    b = tl.program_id(0)
    slot, start, a, z, end, real = _header(Ids, Offsets, Starts, b)
    completed = tl.where(real, end // 4 - start // 4, 0)
    _clear_tail(Slots, b, completed, CAP, SC)
    for n in range(completed):
        block = start // 4 + n
        pool = _pool(V, G, OldV, OldG, Ape, slot, start, a, block, 4, 8, D, 2, D, 0)
        _finish(pool, Pages, Cache, Weight, Rope, Slots, slot, block, b, n, 4, D, CAP, NP,
                ROWS, PAGE)  # fmt: skip
    if real:
        # A multi-token request can wrap the ring onto rows this program's pools just read.
        tl.debug_barrier()
        for first in range(0, tl.minimum(z - a, 8), 8):
            _copy_rows(V, G, OldV, OldG, slot, start, a, end, first, 8, D, 2, 2 * D, 0)


@triton.jit
def prefill_csa(V, G, OldV, OldG, Ids, Offsets, Starts, Pages, Cache, Ape, Weight,
                Rope, Slots, D: tl.constexpr, CAP: tl.constexpr, NP: tl.constexpr,
                ROWS: tl.constexpr, PAGE: tl.constexpr):  # fmt: skip
    b, n = tl.program_id(0), tl.program_id(1)
    slot, start, a, z, end, real = _header(Ids, Offsets, Starts, b)
    completed = tl.where(real, end // 4 - start // 4, 0)
    if n < completed:
        block = start // 4 + n
        pool = _pool(V, G, OldV, OldG, Ape, slot, start, a, block, 4, 8, D, 2, D, 0, DECODE=False)
        _finish(pool, Pages, Cache, Weight, Rope, Slots, slot, block, b, n, 4, D, CAP, NP,
                ROWS, PAGE)  # fmt: skip


@triton.jit
def prefill_hca_pool(V, G, OldV, OldG, Ids, Offsets, Starts, Ape, Pool,
                     CAP: tl.constexpr, TILE: tl.constexpr = 64):  # fmt: skip
    b, n, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot, start, a, z, end, real = _header(Ids, Offsets, Starts, b)
    block = start // 128 + n
    if real & ((block + 1) * 128 <= end):
        pool = _pool(V, G, OldV, OldG, Ape, slot, start, a, block, 128, 128, 512, 1, TILE,
                     channel, DECODE=False)  # fmt: skip
        tl.store(Pool + (b * CAP + n) * 512 + channel * TILE + tl.arange(0, TILE), pool)


@triton.jit
def _publish(Tickets, b):
    # Every writer publishes before the elected thread increments the ticket: a last-CTA
    # reduction, never a spin or a grid-residency barrier.
    tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; mov.u32 $0, %tid.x;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    return tl.atomic_add(Tickets + b, 1, sem="acq_rel", scope="gpu")


@triton.jit
def decode_hca(V, G, OldV, OldG, Ids, Offsets, Starts, Pages, Cache, Ape, Weight,
               Rope, Pool, Slots, Tickets, CAP: tl.constexpr, NP: tl.constexpr,
               SC: tl.constexpr, ROWS: tl.constexpr, PAGE: tl.constexpr,
               TILE: tl.constexpr = 16):  # fmt: skip
    b, channel = tl.program_id(0), tl.program_id(1)
    slot, start, a, z, end, real = _header(Ids, Offsets, Starts, b)
    completed = tl.where(real, end // 128 - start // 128, 0)
    if completed == 0:
        # Nothing to pool: channel 0 owns the metadata and the whole ring update.
        if channel == 0:
            _clear_tail(Slots, b, 0, CAP, SC)
            if real:
                for first in range(0, tl.minimum(z - a, 128), 8):
                    _copy_rows(V, G, OldV, OldG, slot, start, a, end, first, 128, 512, 1, 512, 0)
    else:
        for n in range(completed):
            pool = _pool(V, G, OldV, OldG, Ape, slot, start, a, start // 128 + n, 128, 128,
                         512, 1, TILE, channel)  # fmt: skip
            tl.store(Pool + (b * CAP + n) * 512 + channel * TILE + tl.arange(0, TILE), pool,
                     cache_modifier=".cg")  # fmt: skip
        tl.debug_barrier()
        for first in range(0, tl.minimum(z - a, 128), 8):
            _copy_rows(V, G, OldV, OldG, slot, start, a, end, first, 128, 512, 1, TILE, channel)
        ticket = _publish(Tickets, b)
        if ticket == 512 // TILE - 1:
            tl.debug_barrier()
            _clear_tail(Slots, b, completed, CAP, SC)
            for n in range(completed):
                pool = tl.load(Pool + (b * CAP + n) * 512 + tl.arange(0, 512), cache_modifier=".cg")
                _finish(pool, Pages, Cache, Weight, Rope, Slots, slot, start // 128 + n, b,
                        n, 128, 512, CAP, NP, ROWS, PAGE)  # fmt: skip
            tl.debug_barrier()
            tl.atomic_xchg(Tickets + b, 0, sem="release", scope="gpu")


@triton.jit
def prefill_finish(V, G, OldV, OldG, Ids, Offsets, Starts, Pages, Cache, Weight,
                   Rope, Pool, Slots, R: tl.constexpr, RING: tl.constexpr, D: tl.constexpr,
                   C: tl.constexpr, CAP: tl.constexpr, NP: tl.constexpr, SC: tl.constexpr,
                   ROWS: tl.constexpr, PAGE: tl.constexpr):  # fmt: skip
    b, work = tl.program_id(0), tl.program_id(1)
    slot, start, a, z, end, real = _header(Ids, Offsets, Starts, b)
    completed = tl.where(real, end // R - start // R, 0)
    if work == 0:
        n = tl.arange(0, SC)
        tl.store(Slots + b * CAP + n, -1, (n >= completed) & (n < CAP))
    if R == 128:
        if work < completed:
            pool = tl.load(Pool + (b * CAP + work) * D + tl.arange(0, D))
            _finish(pool, Pages, Cache, Weight, Rope, Slots, slot, start // R + work, b, work,
                    R, D, CAP, NP, ROWS, PAGE)  # fmt: skip
    if real & (work * 8 < tl.minimum(z - a, RING)):
        _copy_rows(V, G, OldV, OldG, slot, start, a, end, work * 8, RING, D, C, C * D, 0)


def run(op, kv, score, state_kv, state_score, ids, offsets, starts, pages, cache, ape, weight,
        rope):  # fmt: skip
    b = ids.numel()
    rows, page = op.rows_per_page, op.page_bytes
    common = (kv, score, state_kv, state_score, ids, offsets, starts)
    tail = (pages, cache, ape, weight, rope)
    np_, cap = pages.shape[1], op.capacity
    sc = triton.next_power_of_2(cap)
    decode = kv.shape[0] <= max(16, b)
    if decode and op.ratio == 128:
        decode_hca[(b, 32)](*common, *tail, op.pooled, op.slots, op.tickets, cap, np_, sc, rows,
                            page, num_warps=4, enable_fp_fusion=False)  # fmt: skip
        return
    if decode:
        decode_csa[(b,)](*common, *tail, op.slots, op.dim, cap, np_, sc, rows, page,
                         num_warps=1 if op.dim == 128 else 4, enable_fp_fusion=False)  # fmt: skip
        return
    outputs = min(cap, triton.cdiv(kv.shape[0], op.ratio) + 1)
    if op.ratio == 4:
        prefill_csa[(b, outputs)](*common, *tail, op.slots, op.dim, cap, np_, rows, page,
                                  num_warps=1 if op.dim == 128 else 4,
                                  enable_fp_fusion=False)  # fmt: skip
    else:
        prefill_hca_pool[(b, outputs, 8)](*common, ape, op.pooled, cap, num_warps=4,
                                          enable_fp_fusion=False)  # fmt: skip
    ring = op.ring
    work = max(outputs if op.ratio == 128 else 1, triton.cdiv(ring, 8))
    prefill_finish[(b, work)](*common, pages, cache, weight, rope, op.pooled, op.slots,
                              op.ratio, ring, op.dim, op.coff, cap, np_, sc, rows, page,
                              num_warps=4, enable_fp_fusion=False)  # fmt: skip
