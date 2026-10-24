# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton DN3 top logprobs: block statistics and sorts, G-way merges, one finalize.

Studied vLLM 0.31.0 v1/worker/gpu/sample/logprob.py (V2 count(value >= selected)
ranks) and SGLang 0.5.22 srt/layers/sampler.py (get_top_logprobs); no code copied.

Raw mode orders unique int64 keys: the high word is the FP32 value's order-preserving
integer (-0 canonicalized to +0), the low word is 2**32-1-ID, so descending keys are
descending value then ascending ID and IDs decode from the key. Each BLOCK of a row
is statistics-reduced (max, sum exp, count >= selected) and sorted in registers; runs
truncated to the output capacity K are then merged G at a time: an element's output
slot is its run index plus, for every other run of its group, a binary-searched count
of keys before it. Every row runs in its own programs (batch invariant) with fixed
reduction trees and a fixed partial-combine order (deterministic).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK, MERGE_TILE, FINAL_TILE = 1024, 256, 1024
MERGE_PROGRAMS = 4  # per merge group; they stride over its tiles
THRESHOLD_BLOCKS = 4096  # above this many blocks per row, every block is sorted
CANDIDATES = 1024  # at most this many keys reach the threshold: rank them directly
_PAD = tl.constexpr(-9223372036854775807)  # below every real key


@triton.jit
def _order_key(x, ids):
    """FP32 values and int64 IDs -> unique int64 keys, descending value then ascending ID."""
    x = tl.where(x == 0, 0.0, x)
    bits = x.to(tl.int32, bitcast=True)
    high = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return (high.to(tl.int64) << 32) | (4294967295 - ids)


@triton.jit
def _key_id(keys):
    return 4294967295 - (keys & 4294967295)


@triton.jit
def _row(ptr, row_stride, storage_rows, row_indices, row):
    index = tl.minimum(tl.maximum(tl.load(row_indices + row), 0), storage_rows - 1)
    return ptr + index.to(tl.int64) * row_stride


@triton.jit
def _active(num_rows, first, selected_ids, row):
    """Rows past the device count and rows selecting -1 are padding."""
    return (first + row < tl.load(num_rows)) & (tl.load(selected_ids + row) != -1)


@triton.jit
def _raw_stats(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    stats_max,
    stats_sum,
    stats_count,
    block_key,
    V,
    NB,
    BLOCK: tl.constexpr,
):
    """Per block: max, sum exp(x - max), count(x >= selected) and the largest order key."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _row(logits, row_stride, storage_rows, row_indices, row)
    selected = tl.minimum(tl.maximum(tl.load(selected_ids + row), 0), V - 1)
    pivot = tl.load(base + selected).to(tl.float32)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    x = tl.load(base + ids, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, 0)
    s = tl.sum(tl.where(mask & (x > -float("inf")), libdevice.exp(x - m), 0.0), 0)
    slot = row.to(tl.int64) * NB + block
    tl.store(stats_max + slot, m)
    tl.store(stats_sum + slot, s)
    tl.store(stats_count + slot, tl.sum((mask & (x >= pivot)).to(tl.int32), 0))
    tl.store(block_key + slot, tl.max(tl.where(mask, _order_key(x, ids), _PAD), 0))


@triton.jit
def _threshold(
    num_rows, first, selected_ids, block_key, threshold, counter, K, NB, NBP: tl.constexpr
):
    """The K-th largest block maximum: at least K keys reach it, so the top K all do."""
    row = tl.program_id(0)
    if not _active(num_rows, first, selected_ids, row):
        return
    t = tl.full([], _PAD, tl.int64)
    if NBP > 0:
        lanes = tl.arange(0, NBP)
        keys = tl.load(block_key + row.to(tl.int64) * NB + lanes, mask=lanes < NB, other=_PAD)
        keys = tl.sort(keys, descending=True)
        t = tl.max(tl.where(lanes == K - 1, keys, _PAD), 0)
    tl.store(threshold + row, t)
    tl.store(counter + row, 0)


@triton.jit
def _append(keys, ids, kept, row, counter, candidate_keys, candidate_ids, CMAX: tl.constexpr):
    """Append kept keys to the row's candidates; the order of appends never matters."""
    base = tl.atomic_add(counter + row, tl.sum(kept.to(tl.int32), 0), sem="relaxed")
    position = base + tl.cumsum(kept.to(tl.int32), 0) - 1
    store = kept & (position < CMAX)
    out = row.to(tl.int64) * CMAX + position
    tl.store(candidate_keys + out, keys, mask=store)
    tl.store(candidate_ids + out, ids.to(tl.int32), mask=store)


@triton.jit
def _store_run(
    keys,
    ids,
    mask,
    t,
    slot,
    runs,
    runs_ids,
    lengths,
    staging,
    CAP: tl.constexpr,
    STABLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sort one block's keys into its run; its length counts keys at or above t."""
    lanes = tl.arange(0, BLOCK)
    if STABLE:
        # Equal keys: greater keys first, then lower IDs (lanes) of the block.
        # Ranked in 128-lane groups against 32-lane chunks of the staged keys: small
        # register tiles keep the frame within the default stack.
        row = staging + slot * BLOCK
        first = tl.min(ids, 0)  # a block's IDs are contiguous
        tl.store(row + lanes, keys)
        tl.debug_barrier()
        for group in range(0, BLOCK, 128):
            mine = group + tl.arange(0, 128)
            my_keys = tl.load(row + mine)
            rank = tl.zeros([128], tl.int32)
            for start in range(0, BLOCK, 32):
                other = start + tl.arange(0, 32)
                theirs = tl.load(row + other)
                before = (theirs[None, :] > my_keys[:, None]) | (
                    (theirs[None, :] == my_keys[:, None]) & (other[None, :] < mine[:, None])
                )
                rank += tl.sum(before.to(tl.int32), 1)
            live = (rank < CAP) & (my_keys != _PAD)  # padding lanes hold _PAD
            tl.store(runs + slot * CAP + rank, my_keys, mask=live)
            tl.store(runs_ids + slot * CAP + rank, (first + mine).to(tl.int32), mask=live)
    else:
        ordered = tl.sort(keys, descending=True)  # unique keys carry their IDs
        tl.store(runs + slot * CAP + lanes, ordered, mask=lanes < CAP)
        tl.store(runs_ids + slot * CAP + lanes, _key_id(ordered).to(tl.int32), mask=lanes < CAP)
    tl.store(lengths + slot, tl.minimum(tl.sum((mask & (keys >= t)).to(tl.int32), 0), CAP))


@triton.jit
def _raw_keys(logits, row_stride, storage_rows, row_indices, row, block, V, BLOCK: tl.constexpr):
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    base = _row(logits, row_stride, storage_rows, row_indices, row)
    x = tl.load(base + ids, mask=mask, other=-float("inf")).to(tl.float32)
    return ids, mask, tl.where(mask, _order_key(x, ids), _PAD)


@triton.jit
def _raw_collect(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    block_key,
    threshold,
    counter,
    candidate_keys,
    candidate_ids,
    V,
    NB,
    CMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append every key at or above the threshold to the row's candidates."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    t = tl.load(threshold + row)
    if tl.load(block_key + row.to(tl.int64) * NB + block) < t:
        return
    ids, mask, keys = _raw_keys(logits, row_stride, storage_rows, row_indices, row, block, V, BLOCK)
    _append(keys, ids, mask & (keys >= t), row, counter, candidate_keys, candidate_ids, CMAX)


@triton.jit
def _raw_sort(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    block_key,
    threshold,
    counter,
    runs,
    runs_ids,
    lengths,
    V,
    NB,
    CMAX: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fallback past CMAX candidates: sort blocks reaching the threshold into runs."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    if tl.load(counter + row) <= CMAX:
        return
    slot = row.to(tl.int64) * NB + block
    t = tl.load(threshold + row)
    if tl.load(block_key + slot) < t:
        tl.store(lengths + slot, 0)
        return
    ids, mask, keys = _raw_keys(logits, row_stride, storage_rows, row_indices, row, block, V, BLOCK)
    _store_run(keys, ids, mask, t, slot, runs, runs_ids, lengths, runs, CAP, False, BLOCK)


@triton.jit
def _rank_candidates(
    num_rows,
    first,
    selected_ids,
    counter,
    candidate_keys,
    candidate_ids,
    out_keys,
    out_ids,
    K,
    CMAX: tl.constexpr,
    TILE: tl.constexpr,
):
    """Order at most CMAX candidates: rank = keys before it (greater, or equal with lower ID)."""
    row = tl.program_id(0)
    if not _active(num_rows, first, selected_ids, row):
        return
    count = tl.load(counter + row)
    if count > CMAX:
        return
    base = row.to(tl.int64) * CMAX
    for start in range(0, count, TILE):
        mine = start + tl.arange(0, TILE)
        live = mine < count
        keys = tl.load(candidate_keys + base + mine, mask=live, other=_PAD)
        ids = tl.load(candidate_ids + base + mine, mask=live, other=0)
        rank = tl.zeros([TILE], tl.int32)
        for other_start in range(0, count, TILE):
            other = other_start + tl.arange(0, TILE)
            present = other < count
            theirs = tl.load(candidate_keys + base + other, mask=present, other=_PAD)
            their_ids = tl.load(candidate_ids + base + other, mask=present, other=0)
            before = present[None, :] & (
                (theirs[None, :] > keys[:, None])
                | ((theirs[None, :] == keys[:, None]) & (their_ids[None, :] < ids[:, None]))
            )
            rank += tl.sum(before.to(tl.int32), 1)
        out = row.to(tl.int64) * K + rank
        tl.store(out_keys + out, keys, mask=live & (rank < K))
        tl.store(out_ids + out, ids, mask=live & (rank < K))


@triton.jit
def _count_before(run, length, keys, inclusive, LOG: tl.constexpr):
    """Per key: how many of a descending run's first `length` keys precede it.

    A key precedes when greater, or equal and `inclusive` (an earlier run of lower IDs).
    """
    lo = tl.zeros(keys.shape, tl.int64)
    hi = tl.zeros(keys.shape, tl.int64) + length
    for _ in tl.static_range(LOG):
        searching = lo < hi
        mid = (lo + hi) // 2
        value = tl.load(run + mid, mask=searching, other=0)
        before = searching & ((value > keys) | (inclusive & (value == keys)))
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(searching & ~before, mid, hi)
    return lo


@triton.jit
def _merge(
    src,
    dst,
    src_ids,
    dst_ids,
    src_lengths,
    dst_lengths,
    counter,
    num_rows,
    first,
    selected_ids,
    NR_IN,
    NR_OUT,
    TILES,
    CAP_IN: tl.constexpr,
    CAP_OUT: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    LOG: tl.constexpr,
    STABLE: tl.constexpr,
    CMAX: tl.constexpr,
    TILE: tl.constexpr,
):
    """Merge G consecutive runs into one, truncated to CAP_OUT.

    STABLE runs carry IDs beside equal keys; runs are in ID order, so equal keys of an
    earlier run go first. Otherwise keys are unique and carry their IDs.
    """
    row, group = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    if CMAX > 0:
        if tl.load(counter + row) <= CMAX:
            return
    first_run = group.to(tl.int64) * G
    members = tl.arange(0, GP)
    sizes = tl.load(
        src_lengths + row.to(tl.int64) * NR_IN + first_run + members,
        mask=(members < G) & (first_run + members < NR_IN),
        other=0,
    )
    sizes = tl.minimum(tl.maximum(sizes, 0), CAP_IN)
    total = tl.sum(sizes, 0)
    if tl.program_id(2) == 0:
        tl.store(dst_lengths + row.to(tl.int64) * NR_OUT + group, tl.minimum(total, CAP_OUT))
    if total == 0:
        return
    offset = row.to(tl.int64) * NR_IN * CAP_IN
    # A few programs per group stride over its tiles: empty groups cost one check.
    for tile in range(tl.program_id(2), TILES, tl.num_programs(2)):
        t = tile * TILE + tl.arange(0, TILE)
        member, index = t // CAP_IN, t % CAP_IN
        run = first_run + member
        length = tl.load(
            src_lengths + row.to(tl.int64) * NR_IN + run,
            mask=(member < G) & (run < NR_IN),
            other=0,
        )
        valid = (member < G) & (run < NR_IN) & (index < tl.minimum(tl.maximum(length, 0), CAP_IN))
        if tl.max(valid.to(tl.int32), 0) != 0:
            keys = tl.load(src + offset + run * CAP_IN + index, mask=valid, other=0)
            position = index.to(tl.int64)
            for other in range(0, G):
                partner = first_run + other
                size = tl.sum(tl.where(members == other, sizes, 0), 0)
                inclusive = STABLE & (other < member)
                before = _count_before(src + offset + partner * CAP_IN, size, keys, inclusive, LOG)
                position += tl.where(member == other, 0, before)
            out = row.to(tl.int64) * NR_OUT * CAP_OUT + group.to(tl.int64) * CAP_OUT + position
            stored = valid & (position < CAP_OUT)
            tl.store(dst + out, keys, mask=stored)
            if STABLE:
                ids = tl.load(src_ids + offset + run * CAP_IN + index, mask=valid, other=0)
                tl.store(dst_ids + out, ids, mask=stored)


@triton.jit
def _raw_finalize(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    top_k,
    stats_max,
    stats_sum,
    stats_count,
    ordered,
    out_selected,
    out_ranks,
    out_ids,
    out_values,
    floor,
    V,
    NB,
    K,
    LOGITS: tl.constexpr,
    FLOOR: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    active = _active(num_rows, first, selected_ids, row)
    out_row = row.to(tl.int64) * K
    if not active:
        tl.store(out_selected + row, 0.0)
        tl.store(out_ranks + row, 0)
        for start in range(0, K, TILE):
            columns = start + lanes
            tl.store(out_ids + out_row + columns, -1, mask=columns < K)
            tl.store(out_values + out_row + columns, 0.0, mask=columns < K)
        return
    stats = row.to(tl.int64) * NB
    m = tl.full([TILE], -float("inf"), tl.float32)
    for start in range(0, NB, TILE):
        b = start + lanes
        m = tl.maximum(m, tl.load(stats_max + stats + b, mask=b < NB, other=-float("inf")))
    m_row = tl.max(m, 0)
    s = tl.zeros([TILE], tl.float32)
    rank = tl.zeros([TILE], tl.int32)
    for start in range(0, NB, TILE):
        b = start + lanes
        part = tl.load(stats_max + stats + b, mask=b < NB, other=-float("inf"))
        total = tl.load(stats_sum + stats + b, mask=b < NB, other=0.0)
        s += tl.where(part > -float("inf"), total * libdevice.exp(part - m_row), 0.0)
        rank += tl.load(stats_count + stats + b, mask=b < NB, other=0)
    empty = m_row == -float("inf")
    log_s = libdevice.log(tl.sum(s, 0))
    source = tl.minimum(tl.maximum(tl.load(row_indices + row), 0), storage_rows - 1)
    base = logits + source.to(tl.int64) * row_stride
    selected = tl.minimum(tl.maximum(tl.load(selected_ids + row), 0), V - 1)
    chosen = tl.load(base + selected).to(tl.float32)
    if not LOGITS:
        chosen = tl.where(empty, -float("inf"), (chosen - m_row) - log_s)
    if FLOOR:
        chosen = tl.where(chosen == -float("inf"), floor, chosen)
    tl.store(out_selected + row, chosen)
    tl.store(out_ranks + row, tl.sum(rank, 0))
    length = tl.load(top_k + row)
    for start in range(0, K, TILE):
        columns = start + lanes
        live = columns < length
        ids = tl.load(ordered + out_row + columns, mask=live, other=0).to(tl.int64)
        x = tl.load(base + ids, mask=live, other=0.0).to(tl.float32)
        if not LOGITS:
            x = tl.where(empty, -float("inf"), (x - m_row) - log_s)
        if FLOOR:
            x = tl.where(x == -float("inf"), floor, x)
        tl.store(
            out_ids + out_row + columns, tl.where(live, ids.to(tl.int32), -1), mask=columns < K
        )
        tl.store(out_values + out_row + columns, tl.where(live, x, 0.0), mask=columns < K)


def _merge_plan(vocab, block, capacity):
    """[(span_in, runs_in, cap_in, runs_out, cap_out, G)] until one run remains."""
    plan, span, runs, cap = [], block, triton.cdiv(vocab, block), min(capacity, block)
    while runs > 1:
        group = min(runs, 64 if cap <= 128 else 16 if cap <= 1024 else 8)
        out_runs, out_cap = triton.cdiv(runs, group), min(capacity, span * group)
        plan.append((span, runs, cap, out_runs, out_cap, group))
        span, runs, cap = span * group, out_runs, out_cap
    return plan


class SortedRuns:
    """Ping-pong run buffers of one row chunk: block sorts write buffer 0, merges alternate."""

    def __init__(self, n, vocab, block, capacity, device, *, stable=False):
        self.vocab, self.block, self.capacity, self.stable = vocab, block, capacity, stable
        self.plan = _merge_plan(vocab, block, capacity)
        self.cap0 = min(capacity, block)
        slots = max([triton.cdiv(vocab, block) * self.cap0] + [p[3] * p[4] for p in self.plan])
        self.keys = torch.empty((2, n, max(slots, 1)), dtype=torch.int64, device=device)
        self.ids = (
            torch.empty((2, n, max(slots, 1)), dtype=torch.int32, device=device)
            if stable
            else self.keys
        )
        runs = triton.cdiv(vocab, block)
        self.lengths = torch.empty((2, n, runs), dtype=torch.int32, device=device)
        self.current = 0

    def merge(self, counter, num_rows, first, selected_ids):
        """counter=None merges every row; else rows with at most CANDIDATES are skipped."""
        n = self.keys.shape[1]
        for _, runs_in, cap_in, runs_out, cap_out, group in self.plan:
            c = self.current
            tiles = triton.cdiv(group * cap_in, MERGE_TILE)
            _merge[(n, runs_out, min(tiles, MERGE_PROGRAMS))](
                self.keys[c],
                self.keys[1 - c],
                self.ids[c],
                self.ids[1 - c],
                self.lengths[c],
                self.lengths[1 - c],
                self.lengths if counter is None else counter,
                num_rows,
                first,
                selected_ids,
                runs_in,
                runs_out,
                tiles,
                CAP_IN=cap_in,
                CAP_OUT=cap_out,
                G=group,
                GP=triton.next_power_of_2(group),
                LOG=cap_in.bit_length(),
                STABLE=self.stable,
                CMAX=0 if counter is None else CANDIDATES,
                TILE=MERGE_TILE,
            )
            self.current = 1 - c
        return self.keys[self.current], self.ids[self.current]


class TopSelection:
    """The first `capacity` keys of each row (descending key, then ascending ID).

    Block maxima (`block_key`, filled by the caller) give a threshold the top `capacity`
    keys all reach. The caller then launches its collect kernel (append keys at or above
    it, `_append`) and its sort kernel (`_store_run`, only past CANDIDATES keys). Up to
    CANDIDATES keys are ranked directly; past that, sorted runs are merged. Both paths are
    launched; each exits on the rows the other serves.
    """

    def __init__(self, n, vocab, capacity, device, *, stable=False):
        self.vocab, self.capacity, self.stable = vocab, capacity, stable
        self.blocks = triton.cdiv(vocab, BLOCK)
        self.runs = SortedRuns(n, vocab, BLOCK, capacity, device, stable=True)
        self.block_key = torch.empty((n, self.blocks), dtype=torch.int64, device=device)
        self.threshold = torch.empty((n,), dtype=torch.int64, device=device)
        self.counter = torch.empty((n,), dtype=torch.int32, device=device)
        self.candidate_keys = torch.empty((n, CANDIDATES), dtype=torch.int64, device=device)
        self.candidate_ids = torch.empty((n, CANDIDATES), dtype=torch.int32, device=device)
        # Stable block sorts stage a block's keys here (fallback rows only).
        self.staging = (
            torch.empty((n, self.blocks * BLOCK), dtype=torch.int64, device=device)
            if stable
            else self.runs.keys
        )

    def prepare(self, gate):
        n, blocks = self.block_key.shape[0], self.blocks
        nbp = triton.next_power_of_2(blocks) if blocks <= THRESHOLD_BLOCKS else 0
        _threshold[(n,)](
            *gate, self.block_key, self.threshold, self.counter, self.capacity, blocks, NBP=nbp
        )
        return (self.block_key, self.threshold, self.counter)

    def finish(self, gate):
        """Ordered [n, capacity] keys and int32 IDs (row stride capacity)."""
        keys, ids = self.runs.merge(self.counter, *gate)
        _rank_candidates[(self.block_key.shape[0],)](
            *gate,
            self.counter,
            self.candidate_keys,
            self.candidate_ids,
            keys,
            ids,
            self.capacity,
            CMAX=CANDIDATES,
            TILE=64,
        )
        return keys, ids


def run_raw(logits, row_indices, num_rows, first, selected_ids, top_k, values, floor, out):
    """Rows first.. of the call; out = (selected, ranks, top_ids, top_values), all written."""
    n, vocab = row_indices.numel(), logits.shape[1]
    capacity = out[2].shape[1]
    if n == 0:
        return
    device = logits.device
    blocks = triton.cdiv(vocab, BLOCK)
    stats_max = torch.empty((n, blocks), dtype=torch.float32, device=device)
    stats_sum = torch.empty_like(stats_max)
    stats_count = torch.empty((n, blocks), dtype=torch.int32, device=device)
    top = TopSelection(n, vocab, max(capacity, 1), device)
    gate = (num_rows, first, selected_ids)
    source = (logits, logits.stride(0), logits.shape[0], row_indices)
    _raw_stats[(n, blocks)](
        *source,
        *gate,
        stats_max,
        stats_sum,
        stats_count,
        top.block_key,
        vocab,
        blocks,
        BLOCK=BLOCK,
    )
    final = top.candidate_ids
    if capacity:
        state = top.prepare(gate)
        _raw_collect[(n, blocks)](
            *source,
            *gate,
            *state,
            top.candidate_keys,
            top.candidate_ids,
            vocab,
            blocks,
            CMAX=CANDIDATES,
            BLOCK=BLOCK,
        )
        runs = top.runs
        _raw_sort[(n, blocks)](
            *source,
            *gate,
            *state,
            runs.keys[0],
            runs.ids[0],
            runs.lengths[0],
            vocab,
            blocks,
            CMAX=CANDIDATES,
            CAP=runs.cap0,
            BLOCK=BLOCK,
        )
        final = top.finish(gate)[1]
    _raw_finalize[(n,)](
        *source,
        *gate,
        top_k,
        stats_max,
        stats_sum,
        stats_count,
        final,
        *out,
        0.0 if floor is None else floor,
        vocab,
        blocks,
        capacity,
        LOGITS=values == "logits",
        FLOOR=floor is not None,
        TILE=FINAL_TILE,
    )
