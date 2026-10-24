# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton segmented LoRA delta: the portable provider on every CUDA device off sm_120/sm_121.

Same contract and rounding as the CuTe provider (see ``linear._lora_segmented``): shrink
accumulates ``x @ A^T`` in FP32 over sorted chunks of <=CHUNK tokens, applies the adapter
scaling in FP32 and rounds once to the input dtype; expand multiplies that rounded tile by B
in FP32 and read-modify-writes the output in FP32. Split-K is deterministic: every split
writes FP32 partials, and the last split to arrive at a tile (an acq_rel counter) sums them
in split order and resets the counter. One split writes the rounded values directly, with no
partials or counters. Both launches use programmatic dependent launch: shrink waits for its
inputs then releases expand, which loads its first B tile before waiting on the shrink. Weights
and grouping metadata must be ready before the preceding kernel launches (as for CuTe). A
row-parallel call (x and A are this rank's K shard) keeps the
scaled shrink partial in FP32, never rounded to one 16-bit value; expand takes it as
hi + lo 16-bit parts (two tensor-core dots, ~16 significant bits):
the per-rank delta then joins the base partial, and the caller's base all-reduce sums
both. Graph-captured calls take scratch from the active ``GraphWorkspace``
slot, or else allocate it per call in the graph's pool; eager calls allocate per call.

Original code. Grid and segment handling were studied in vLLM 0.31.0
``vllm/lora/ops/triton_ops/lora_shrink_op.py`` / ``lora_expand_op.py`` (split-K by FP
atomics there) and SGLang 0.5.22 ``sglang/kernels/ops/lora/dense/chunked_sgmv_shrink.py`` /
``chunked_sgmv_expand.py`` (one program per chunk, no split-K).
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from ..._runtime.tables import Dim, TunedOp, pow2_edges
from ..._runtime.workspace import GraphWorkspace

_SLOT = "lora_segmented_triton"
_BLOCK_K = 128
_SCAN = 128
_MIN_SPLIT_K = 256
_MAX_SPLITS = 64
_GROUP = tl.constexpr(8)
_UNROLL = tl.constexpr(8)


@lru_cache(maxsize=None)
def supports_pdl(device) -> bool:
    """Programmatic dependent launch (gdc_wait / launch_pdl) needs cc >= 9.0."""
    return torch.cuda.get_device_capability(device) >= (9, 0)


@triton.jit
def _locate(
    p,
    seg_indptr,
    weight_indices,
    chunk_start,
    chunk_end,
    chunk_adapter,
    num_segments,
    HAS_TABLE: tl.constexpr,
    CHUNK: tl.constexpr,
    SCAN: tl.constexpr,
):
    """Sorted rows [start, end) and adapter of chunk ``p``; end <= start when inactive."""
    if HAS_TABLE:
        start = tl.load(chunk_start + p)
        end = tl.load(chunk_end + p)
        adapter = tl.load(chunk_adapter + p)
    else:
        # CSR scan: segment s owns cdiv(len_s, CHUNK) consecutive chunk ids.
        start = 0
        end = 0
        adapter = -1
        before = 0
        for base in range(0, num_segments, SCAN):
            s = base + tl.arange(0, SCAN)
            live = s < num_segments
            lo = tl.load(seg_indptr + s, mask=live, other=0)
            hi = tl.load(seg_indptr + s + 1, mask=live, other=0)
            count = tl.where(live & (hi > lo), (hi - lo + CHUNK - 1) // CHUNK, 0)
            last = before + tl.cumsum(count, 0)
            hit = (last - count <= p) & (p < last)
            first = lo + (p - (last - count)) * CHUNK
            start += tl.sum(tl.where(hit, first, 0))
            end += tl.sum(tl.where(hit, tl.minimum(first + CHUNK, hi), 0))
            owner = tl.load(weight_indices + s, mask=live, other=0)
            adapter += tl.sum(tl.where(hit, owner + 1, 0))
            before += tl.sum(count)
    return start, end, adapter


@triton.jit(do_not_specialize=["num_rows", "num_segments", "splits", "k_per_split", "num_tiles"])
def _shrink_kernel(
    x,
    a_ptrs,
    ranks,
    scalings,
    seg_indptr,
    weight_indices,
    permutation,
    chunk_start,
    chunk_end,
    chunk_adapter,
    partials,
    reduced,
    counters,
    num_rows,
    num_segments,
    K,
    splits,
    k_per_split,
    num_tiles,
    NUM_SLICES: tl.constexpr,
    BUCKET: tl.constexpr,
    RANK_STRIDE: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_TABLE: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCAN: tl.constexpr,
    PDL: tl.constexpr,
):
    p = tl.program_id(0)
    slice_id = tl.program_id(1) % NUM_SLICES
    rank_tile = tl.program_id(1) // NUM_SLICES
    split = tl.program_id(2)
    start, end, adapter = _locate(
        p,
        seg_indptr,
        weight_indices,
        chunk_start,
        chunk_end,
        chunk_adapter,
        num_segments,
        HAS_TABLE,
        CHUNK,
        SCAN,
    )
    if adapter < 0 or end <= start:
        return
    rank = tl.load(ranks + adapter)
    if rank_tile * BLOCK_R >= rank:
        return
    rows = start + tl.arange(0, CHUNK)
    row_ok = rows < end
    if HAS_PERM:
        tokens = tl.load(permutation + rows, mask=row_ok, other=0).to(tl.int64)
    else:
        tokens = rows.to(tl.int64)
    r = rank_tile * BLOCK_R + tl.arange(0, BLOCK_R)
    r_ok = r < rank
    a = tl.load(a_ptrs + adapter * NUM_SLICES + slice_id).to(tl.pointer_type(x.dtype.element_ty))
    if RANK_STRIDE:
        a += slice_id.to(tl.int64) * rank * K
    k_begin = split * k_per_split
    if PDL:  # x may come from the preceding kernel; let expand launch and prefetch B
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros((CHUNK, BLOCK_R), dtype=tl.float32)
    for k0 in range(k_begin, tl.minimum(k_begin + k_per_split, K), BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        k_ok = kk < K
        xt = tl.load(x + tokens[:, None] * K + kk[None, :], mask=row_ok[:, None] & k_ok[None, :])
        at = tl.load(
            a + r[None, :].to(tl.int64) * K + kk[:, None], mask=r_ok[None, :] & k_ok[:, None]
        )
        acc = tl.dot(xt, at, acc)
    scale = tl.load(scalings + adapter)
    out = (rows.to(tl.int64)[:, None] * NUM_SLICES + slice_id) * BUCKET + r[None, :]
    mask = row_ok[:, None] & r_ok[None, :]
    if splits == 1:
        tl.store(reduced + out, (acc * scale).to(reduced.dtype.element_ty), mask=mask)
        return
    plane = num_rows.to(tl.int64) * NUM_SLICES * BUCKET
    tl.store(partials + split * plane + out, acc, mask=mask)
    # Fixed-order fixup by the last split: the barrier orders every thread's partial
    # stores before the release; the acquire orders the reads after it.
    tl.debug_barrier()
    tile = (p * NUM_SLICES + slice_id) * (BUCKET // BLOCK_R) + rank_tile
    counter = counters + tile
    if splits > 2 * _GROUP:
        # Two levels, groups of _GROUP splits: the last of a group sums it (in order) into
        # the group's first slot; the last group sums the group totals (in order).
        groups = tl.cdiv(splits, _GROUP)
        group = split // _GROUP
        members = tl.minimum(_GROUP, splits - group * _GROUP)
        gcounter = counters + num_tiles + tile * groups + group
        if tl.atomic_add(gcounter, 1, sem="acq_rel", scope="gpu") == members - 1:
            total = tl.zeros((CHUNK, BLOCK_R), dtype=tl.float32)
            for j in tl.static_range(_GROUP):
                total += tl.load(
                    partials + (group * _GROUP + j) * plane + out,
                    mask=mask & (j < members),
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(partials + group * _GROUP * plane + out, total, mask=mask)
            tl.atomic_xchg(gcounter, 0, sem="relaxed", scope="gpu")
            tl.debug_barrier()
            if tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu") == groups - 1:
                total = tl.zeros((CHUNK, BLOCK_R), dtype=tl.float32)
                for g0 in range(0, groups, _UNROLL):
                    for j in tl.static_range(_UNROLL):
                        total += tl.load(
                            partials + (g0 + j) * _GROUP * plane + out,
                            mask=mask & (g0 + j < groups),
                            other=0.0,
                            cache_modifier=".cg",
                        )
                tl.store(reduced + out, (total * scale).to(reduced.dtype.element_ty), mask=mask)
                tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")
    elif tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu") == splits - 1:
        # Eight independent loads per step hide L2 latency; adds stay in split order.
        total = tl.zeros((CHUNK, BLOCK_R), dtype=tl.float32)
        for s0 in range(0, splits, _UNROLL):
            for j in tl.static_range(_UNROLL):
                total += tl.load(
                    partials + (s0 + j) * plane + out,
                    mask=mask & (s0 + j < splits),
                    other=0.0,
                    cache_modifier=".cg",
                )
        tl.store(reduced + out, (total * scale).to(reduced.dtype.element_ty), mask=mask)
        tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")


@triton.jit(do_not_specialize=["num_segments"])
def _expand_kernel(
    output,
    b_ptrs,
    ranks,
    slices,
    seg_indptr,
    weight_indices,
    permutation,
    chunk_start,
    chunk_end,
    chunk_adapter,
    reduced,
    num_segments,
    out_stride,
    b_ld,
    NUM_SLICES: tl.constexpr,
    BUCKET: tl.constexpr,
    HAS_PERM: tl.constexpr,
    HAS_TABLE: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    SCAN: tl.constexpr,
    PDL: tl.constexpr,
):
    p = tl.program_id(0)
    # Axis 1 enumerates the column tiles of all slices back to back (narrow slices take
    # one tile, not max-width many).
    tile = tl.program_id(1)
    slice_id = 0
    col_tile = tile
    before = 0
    for s in tl.static_range(NUM_SLICES):
        count = tl.cdiv(tl.load(slices + 2 * s + 1), BLOCK_N)
        inside = (tile >= before) & (tile < before + count)
        slice_id = tl.where(inside, s, slice_id)
        col_tile = tl.where(inside, tile - before, col_tile)
        before += count
    width = tl.load(slices + 2 * slice_id + 1)
    start, end, adapter = _locate(
        p,
        seg_indptr,
        weight_indices,
        chunk_start,
        chunk_end,
        chunk_adapter,
        num_segments,
        HAS_TABLE,
        CHUNK,
        SCAN,
    )
    if adapter < 0 or end <= start:
        return
    rank = tl.load(ranks + adapter)
    if rank == 0:
        return
    rows = start + tl.arange(0, CHUNK)
    row_ok = rows < end
    if HAS_PERM:
        tokens = tl.load(permutation + rows, mask=row_ok, other=0).to(tl.int64)
    else:
        tokens = rows.to(tl.int64)
    n = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = n < width
    b = tl.load(b_ptrs + adapter * NUM_SLICES + slice_id).to(
        tl.pointer_type(output.dtype.element_ty)
    )
    low = reduced + (rows.to(tl.int64)[:, None] * NUM_SLICES + slice_id) * BUCKET
    acc = tl.zeros((CHUNK, BLOCK_N), dtype=tl.float32)
    # The first B tile is weights, loaded before waiting on the shrink (PDL).
    r = tl.arange(0, BLOCK_R)
    bt = tl.load(
        b + n[None, :].to(tl.int64) * b_ld + r[:, None], mask=(r < rank)[:, None] & n_ok[None, :]
    )
    if PDL:
        tl.extra.cuda.gdc_wait()
    for r0 in range(0, rank, BLOCK_R):
        r = r0 + tl.arange(0, BLOCK_R)
        r_ok = r < rank
        lt = tl.load(low + r[None, :], mask=row_ok[:, None] & r_ok[None, :], other=0.0)
        if r0 > 0:
            bt = tl.load(
                b + n[None, :].to(tl.int64) * b_ld + r[:, None], mask=r_ok[:, None] & n_ok[None, :]
            )
        if reduced.dtype.element_ty == tl.float32:  # row-parallel: hi + lo, ~16 bits
            hi = lt.to(bt.dtype)
            acc = tl.dot(hi, bt, acc)
            acc = tl.dot((lt - hi.to(tl.float32)).to(bt.dtype), bt, acc)
        else:
            acc = tl.dot(lt, bt, acc)
    column = tl.load(slices + 2 * slice_id) + n
    dst = output + tokens[:, None] * out_stride + column[None, :]
    mask = row_ok[:, None] & n_ok[None, :]
    base = tl.load(dst, mask=mask).to(tl.float32)
    tl.store(dst, (base + acc).to(output.dtype.element_ty), mask=mask)


def _rank_tile(bucket):
    # GB10, Qwen3.5-27B down r64 M=4: rank tile 32 beat 64 by 12% (more CTAs per split).
    return min(bucket, 32)


def _host_splits(count, k, bucket, num_slices, sm_count):
    # Decode-sized grids need many K splits to stream A at full bandwidth.
    tiles = max(1, count * num_slices * (bucket // _rank_tile(bucket)))
    target = (4 if count <= 4 else 2) * sm_count
    return min(_MAX_SPLITS, max(1, k // _MIN_SPLIT_K), -(-target // tiles))


def split_k(k, splits, block_k=_BLOCK_K):
    """Effective K splits (each a ``block_k`` multiple) and the per-split K length."""
    per_split = triton.cdiv(triton.cdiv(k, splits), block_k) * block_k
    return -(-k // per_split), per_split


def _default(shape, device):
    """Host rules, from the call's shape and the device's SM count."""
    m, sizes, bucket = shape["m"], shape["sizes"], shape["rank"]
    bound = -(-m // 16) + shape["adapters"]
    splits = _host_splits(min(m, bound), shape["k"], bucket, len(sizes), device.sms)
    wide = bound * sum(-(-n // 128) for n in sizes) >= device.sms
    splits = 1 << (splits.bit_length() - 1)
    # Same anchor: 256-wide K steps with 8 warps were 14% faster wherever a split covers them.
    deep = shape["k"] // splits >= 256
    if m <= 64:
        # GB10 cold anchor (27B down r64 M=4, PDL): 64 splits of 16-rank tiles on 4 warps
        # streamed A 7% faster than 32 x 32 on 8 warps; the two-level fixup keeps the tail short.
        # Splits fill ~4 waves of SMs across the call's tiles (more adapters, fewer splits),
        # rounded up to a power of two, at most 64 and at least 256 K per split.
        chunks = min(m, -(-m // 16) + shape["adapters"] - 1)
        tiles = chunks * len(sizes) * (bucket // 16)
        want = 1 << (max(1, -(-4 * device.sms // tiles)) - 1).bit_length()
        splits = max(1, min(64, want, 1 << (max(1, shape["k"] // 256).bit_length() - 1)))
        return {
            "splits": splits,
            "rank_tile": 16,
            # Many chunks (one token per adapter): 128-column tiles, fewer tiny CTAs. 256 lost
            # everywhere measured: 1.7x slower at r128 (5090 27B gate_up/qkv_gate M16 a16),
            # 1.17x at r16 (5090 in_proj M64 a64), 5% (GB10 o M32 a32).
            "col_tile": 128 if chunks * sum(-(-n // 128) for n in sizes) >= 4 * device.sms else 64,
            "shrink_block_k": 256 if shape["k"] // splits >= 256 else _BLOCK_K,
            "shrink_warps": 4,
            "shrink_stages": 3,
            "expand_warps": 4,
            "expand_stages": 2,
        }
    return {
        "splits": splits,
        "rank_tile": _rank_tile(bucket),
        "col_tile": 128 if wide else 64,
        "shrink_block_k": 256 if deep else _BLOCK_K,
        "shrink_warps": 8 if deep else 4,
        "shrink_stages": 3,
        "expand_warps": 4,
        "expand_stages": 2,
    }


TUNED = TunedOp(
    "linear.lora_segmented.triton",
    version=3,
    space={
        "splits": (1, 2, 4, 8, 16, 32, 64),
        "rank_tile": (16, 32, 64),
        "col_tile": (64, 128, 256),
        "shrink_block_k": (64, 128, 256),
        "shrink_warps": (4, 8),
        "shrink_stages": (2, 3, 4),
        "expand_warps": (4, 8),
        "expand_stages": (2, 3),
    },
    dims=(
        Dim("m", pow2_edges(1, 16384)),
        Dim("adapters", (1, 2, 4, 8)),
        Dim("k"),
        Dim("sizes"),
        Dim("rank", (16, 32, 64, 128, 256, 512)),
        Dim("low"),
        Dim("dtype"),
    ),
    default=_default,
    valid=lambda config, shape: config["rank_tile"] <= shape["rank"],
)


def tuned_config(device, m, segments, k, sizes, bucket, dtype, row_parallel, override):
    """The table's config for this call, with the explicit (splits, rank_tile, col_tile)
    override fields applied on top."""
    config = TUNED.config(
        device,
        m=m,
        adapters=max(1, segments),
        k=k,
        sizes=tuple(sizes),
        rank=bucket,
        low="fp32" if row_parallel else "rounded",
        dtype=str(dtype).removeprefix("torch."),
    )
    for name, value in zip(("splits", "rank_tile", "col_tile"), override[:3], strict=True):
        if value is not None:
            config[name] = value
    if config["rank_tile"] > bucket:  # explicit overrides bypass the table's validity rule
        raise ValueError("config rank_tile must fit the rank bucket")
    return config


def _counts(rows, bound, splits, num_slices, bucket, block_r):
    tiles = bound * num_slices * (bucket // block_r)
    groups = -(-splits // _GROUP.value) if splits > 2 * _GROUP.value else 0
    partials = splits * rows * num_slices * bucket if splits > 1 else 0
    counters = tiles * (1 + groups) if splits > 1 else 0
    return partials, rows * num_slices * bucket, counters


def _reserve_counts(
    max_rows, device, segments, k, sizes, bucket, dtype, row_parallel, override, chunk
):
    # Every row count up to max_rows, each with its own config; bound peaks at one
    # segment per row.
    most = (0, 0, 0)
    for m in range(1, max_rows + 1):
        config = tuned_config(device, m, segments, k, sizes, bucket, dtype, row_parallel, override)
        splits, _ = split_k(k, config["splits"], config["shrink_block_k"])
        bound = -(-m // chunk) + m
        counts = _counts(m, bound, splits, len(sizes), bucket, config["rank_tile"])
        most = tuple(max(a, b) for a, b in zip(most, counts, strict=True))
    return most


def _allocate(device, dtype, counts):
    partials, reduced, counters = counts
    return (
        torch.empty(partials, device=device, dtype=torch.float32),
        torch.empty(reduced, device=device, dtype=dtype),
        torch.zeros(counters, device=device, dtype=torch.int32),
    )


def launch(
    output,
    x,
    tables,
    seg_indptr,
    weight_indices,
    ranks,
    scalings,
    sizes,
    max_rank,
    permutation,
    b_ld,
    rank_stride,
    chunks,
    config,
    chunk,
    row_parallel=False,
    *,
    capturing=False,
    compiled=None,
):
    """Launch shrink + expand (shrink only when ``output`` is None); returns the config
    (splits, rank_tile, col_tile, chunk) and the low-rank tensor [M, slices, bucket].

    Tile, split and launch parameters come from ``TUNED`` (offline tables, else the host
    rules); fields set in ``config`` (splits, rank_tile, col_tile) override them. ``compiled``
    records every launched variant; a variant not run eagerly first is refused while
    ``capturing`` (capture never compiles).

    ``row_parallel`` keeps the scaled shrink result in FP32 (never rounded) for expand."""
    m, k = x.shape
    ns = len(sizes)
    bucket = max(16, 1 << (max_rank - 1).bit_length())
    segments = weight_indices.numel()
    call = (x.device, segments, k, tuple(sizes), bucket, x.dtype, row_parallel, config[:3])
    tuned = tuned_config(x.device, m, *call[1:])
    block_r, col_tile, block_k = tuned["rank_tile"], tuned["col_tile"], tuned["shrink_block_k"]
    splits, per_split = split_k(k, tuned["splits"], block_k)
    bound = (m + chunk - 1) // chunk + segments
    counts = _counts(m, bound, splits, ns, bucket, block_r)
    low_dtype = torch.float32 if row_parallel else x.dtype
    group = GraphWorkspace.active()
    if group is not None:
        slot = (_SLOT, x.device, low_dtype)
        group.reserve(
            slot, (*call, chunk), _reserve_counts, lambda c: _allocate(x.device, low_dtype, c)
        )
        work = group.take(slot, m, counts)
    else:
        work = _allocate(x.device, low_dtype, counts)
    partials, reduced, counters = (t[:n] for t, n in zip(work, counts, strict=True))
    low = reduced.view(m, ns, bucket)
    if m == 0:
        return (splits, block_r, col_tile, chunk), low
    a_ptrs, b_ptrs, identity, slices = tables
    has_table = chunks is not None
    if compiled is not None:
        variant = (
            "triton",
            x.device,
            x.dtype,
            low_dtype,
            output is None,
            k,
            tuple(sizes),
            bucket,
            chunk,
            has_table,
            permutation is not None,
            rank_stride,
            b_ld % 16 == 0,
            output is None or output.stride(0) % 16 == 0,
            tuple(sorted(tuned.items())),
        )
        if variant not in compiled:
            if capturing:
                raise RuntimeError(
                    "new segmented LoRA variant during CUDA graph capture; warm it eagerly first"
                )
            compiled[variant] = True
    starts, ends, owners = chunks[:3] if has_table else (identity,) * 3
    perm = identity if permutation is None else permutation
    common = {
        "NUM_SLICES": ns,
        "BUCKET": bucket,
        "HAS_PERM": permutation is not None,
        "HAS_TABLE": has_table,
        "CHUNK": chunk,
        "SCAN": _SCAN,
    }
    pdl = supports_pdl(x.device)
    _shrink_kernel[(bound, ns * (bucket // block_r), splits)](
        x,
        a_ptrs,
        ranks,
        scalings,
        seg_indptr,
        weight_indices,
        perm,
        starts,
        ends,
        owners,
        partials,
        reduced,
        counters,
        m,
        weight_indices.numel(),
        k,
        splits,
        per_split,
        bound * ns * (bucket // block_r),
        RANK_STRIDE=rank_stride,
        BLOCK_R=block_r,
        BLOCK_K=block_k,
        PDL=pdl,
        launch_pdl=pdl,
        num_warps=tuned["shrink_warps"],
        num_stages=tuned["shrink_stages"],
        **common,
    )
    if output is None:
        return (splits, block_r, col_tile, chunk), low
    _expand_kernel[(bound, sum(-(-n // col_tile) for n in sizes))](
        output,
        b_ptrs,
        ranks,
        slices,
        seg_indptr,
        weight_indices,
        perm,
        starts,
        ends,
        owners,
        reduced,
        weight_indices.numel(),
        output.stride(0),
        b_ld,
        BLOCK_N=col_tile,
        BLOCK_R=min(bucket, 64),
        PDL=pdl,
        launch_pdl=pdl,
        num_warps=tuned["expand_warps"],
        num_stages=tuned["expand_stages"],
        **common,
    )
    return (splits, block_r, col_tile, chunk), low
