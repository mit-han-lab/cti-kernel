# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton DN3 processed logprobs (sources A and B) with the torch provider's arithmetic.

Studied vLLM 0.31.0 v1/sample/ops/topk_topp_sampler.py and SGLang 0.5.22
srt/layers/sampler.py (top-k/top-p/min-p renormalization); no code copied. The law is
this package's S1 order (penalty -> temperature -> top-k -> top-p -> min-p).

Nothing vocabulary-wide is stored: every stage recomputes a token's z, masks and
masses from the logits (`_derive`) and a few per-row scalars kept in small state
arrays, written by one program per row between reductions. Stages over (row, block):
block max/argmax (+ raw-order sorts for a top-k filter); S1 4096-token butterfly sums
of kept weights; probability-order sorts and fixed-mass sums for a top-p filter; the
log-sum-exp; order keys (FP32 mass, then processed logprob) whose top list the shared
TopSelection finds; one finalize. Every row runs in its own programs with fixed
reduction trees: batch invariant and deterministic.

Two exact shortcuts: the smallest-ID argmax always survives top-k, top-p and min-p,
so max(z over the support) is z(argmax) = 0 (the torch provider subtracts that zero)
and the min-p pivot max(p) is the argmax's renormalized mass.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ._top_logprobs import (
    BLOCK,
    CANDIDATES,
    FINAL_TILE,
    SortedRuns,
    TopSelection,
    _active,
    _append,
    _key_id,
    _order_key,
    _store_run,
)

CHUNK = 4096  # S1 reduction chunk: 2 tiles x 512 lanes x 4 components
_MASS = tl.constexpr(4611686018427387904.0)  # 2**62
_INV_MASS = tl.constexpr(2.168404344971009e-19)  # 2**-62
_NEG = tl.constexpr(-float("inf"))
_PAD = tl.constexpr(-9223372036854775807)  # below every real key
# Per-row state: FP32 fields, then int64 fields.
F_M, F_TOTAL, F_SCALE, F_PMAX, F_LOGS, F_NORM, F_APMAX, NF = (tl.constexpr(v) for v in range(8))
I_EMPTY, I_ARGMAX, I_TAU, I_CUTOFF, I_EMPTY_KEY, NI = (tl.constexpr(v) for v in range(6))


@triton.jit
def _rn(a, b, op: tl.constexpr):
    return tl.inline_asm_elementwise(
        op + ".rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _source(ptr, row_stride, storage_rows, row_indices, row):
    index = tl.minimum(tl.maximum(tl.load(row_indices + row), 0), storage_rows - 1)
    return ptr + index.to(tl.int64) * row_stride


@triton.jit
def _row_max(partials, row, NB, TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    m = tl.full([TILE], _NEG, tl.float32)
    for start in range(0, NB, TILE):
        b = start + lanes
        m = tl.maximum(m, tl.load(partials + row * NB + b, mask=b < NB, other=_NEG))
    return tl.max(m, 0)


@triton.jit
def _row_sum(partials, row, NB, TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    s = tl.zeros([TILE], partials.dtype.element_ty)
    for start in range(0, NB, TILE):
        b = start + lanes
        s += tl.load(partials + row * NB + b, mask=b < NB, other=0)
    return tl.sum(s, 0)


@triton.jit
def _row_state(block_max, block_arg, row, NB, TILE: tl.constexpr):
    """Sanitized row maximum, empty flag and smallest-ID argmax (torch.argmax)."""
    lanes = tl.arange(0, TILE)
    m = _row_max(block_max, row, NB, TILE)
    arg = tl.full([TILE], 2147483647, tl.int32)
    for start in range(0, NB, TILE):
        b = start + lanes
        part = tl.load(block_max + row * NB + b, mask=b < NB, other=_NEG)
        ids = tl.load(block_arg + row * NB + b, mask=b < NB, other=2147483647)
        arg = tl.minimum(arg, tl.where((b < NB) & (part == m), ids, 2147483647))
    empty = m == _NEG
    return tl.where(empty, 0.0, m), empty, tl.where(empty, 0, tl.min(arg, 0))


@triton.jit
def _temperature(temperatures, row):
    temperature = tl.load(temperatures + row)
    return tl.where(temperature == 0, 1.0, temperature), temperature == 0


@triton.jit
def _z(raw, empty, m, t):
    """S1 shifted temperature: subtract then divide; scale first only where that overflows."""
    x = tl.where(empty, 0.0, raw)
    mb, tb = tl.broadcast_to(m, raw.shape), tl.broadcast_to(t, raw.shape)
    difference = _rn(x, mb, "sub")
    z = _rn(difference, tb, "div")
    overflow = (difference == _NEG) & (x > _NEG) & (x < float("inf"))
    # Scale first only where l - m overflows (spreads above FLT_MAX): rare, so branch.
    if tl.max(overflow.to(tl.int32)) > 0:
        z = tl.where(overflow, _rn(_rn(x, tb, "div"), _rn(mb, tb, "div"), "sub"), z)
    return x, z


@triton.jit
def _derive(
    raw,
    ids,
    mask,
    m,
    empty,
    t,
    filtered,
    tau_key,
    total,
    top_identity,
    cutoff,
    scale,
    mp,
    pmax,
    greedy,
    argmax,
    HAS_MIN_P: tl.constexpr,
):
    """Per token: masked z (zz), final support, renormalized mass p2 and the pre-top-p mass."""
    _, z = _z(raw, empty, m, t)
    keep = mask & (~filtered | (_order_key(raw, ids) >= tau_key))
    w = tl.where(keep, libdevice.exp(z), 0.0)
    p = _rn(w, tl.broadcast_to(total, raw.shape), "div")
    keep = keep & (top_identity | (_order_key(p, ids) >= cutoff))
    p2 = tl.where(top_identity, p, _rn(p, tl.broadcast_to(scale, raw.shape), "div"))
    p2 = tl.where(keep, p2, 0.0)
    if HAS_MIN_P:
        pivot = _rn(tl.broadcast_to(mp, raw.shape), tl.broadcast_to(pmax, raw.shape), "mul")
        keep = keep & (p2 >= pivot)
    keep = tl.where(greedy, mask & (ids == argmax), keep)
    zz = tl.where(keep, tl.where(greedy, 0.0, z), _NEG)
    return zz, keep, p2, p


@triton.jit
def _masked_a(pbase, ids, mask, live, mp, pmax_a, HAS_MIN_P: tl.constexpr):
    """Source A masses: zero on empty rows; min-p keeps p >= RN(min_p * max p)."""
    p = tl.where(live, tl.load(pbase + ids, mask=mask, other=0.0), 0.0)
    if HAS_MIN_P:
        pivot = _rn(tl.broadcast_to(mp, p.shape), tl.broadcast_to(pmax_a, p.shape), "mul")
        p = tl.where(p >= pivot, p, 0.0)
    return p


@triton.jit
def _order(p, tie):
    """Processed order key: FP32 mass bits, then the tie logprob's order (no ID)."""
    tie = tl.where(tie == 0, 0.0, tie)
    bits = tie.to(tl.int32, bitcast=True)
    low = (tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF) ^ -2147483648).to(tl.int64) & 4294967295
    return (tl.where(p > 0, p, 0.0).to(tl.int32, bitcast=True).to(tl.int64) << 32) | low


@triton.jit
def _min_p(min_p, row, HAS_MIN_P: tl.constexpr):
    mp = 0.0
    if HAS_MIN_P:
        mp = tl.load(min_p + row)
    return mp


@triton.jit
def _token(
    base,
    ids,
    mask,
    row,
    fstate,
    istate,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    V,
    HAS_MIN_P: tl.constexpr,
):
    """`_derive` for tokens `ids` of a row, from the row state written so far."""
    f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
    t, greedy = _temperature(temperatures, row)
    raw = tl.load(base + ids, mask=mask, other=0.0).to(tl.float32)
    return _derive(
        raw,
        ids,
        mask,
        tl.load(f + F_M),
        tl.load(i + I_EMPTY) != 0,
        t,
        tl.load(filter_top_k + row) < V,
        tl.load(i + I_TAU),
        tl.load(f + F_TOTAL),
        tl.load(top_p + row) == 1,
        tl.load(i + I_CUTOFF),
        tl.load(f + F_SCALE),
        _min_p(min_p, row, HAS_MIN_P),
        tl.load(f + F_PMAX),
        greedy,
        tl.load(i + I_ARGMAX),
        HAS_MIN_P,
    )


@triton.jit
def _keys_of(
    base,
    pbase,
    ids,
    mask,
    row,
    fstate,
    istate,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    V,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
):
    """Processed logprobs, the masses that order them, and order keys of tokens `ids`."""
    zz, keep, p2, _ = _token(
        base, ids, mask, row, fstate, istate, temperatures, filter_top_k, top_p, min_p, V, HAS_MIN_P
    )
    f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
    empty = tl.load(i + I_EMPTY) != 0
    lp = tl.where(empty, _NEG, _rn(zz, tl.broadcast_to(tl.load(f + F_LOGS), zz.shape), "sub"))
    if SOURCE_A:
        mp = _min_p(min_p, row, HAS_MIN_P)
        p = _masked_a(pbase, ids, mask, mask & ~empty, mp, tl.load(f + F_APMAX), HAS_MIN_P)
    else:
        greedy = tl.load(temperatures + row) == 0
        p = tl.where(keep & (lp > _NEG), tl.where(greedy, 1.0, p2), 0.0)
    tie = tl.where(tl.load(i + I_EMPTY_KEY) != 0, _NEG, lp)
    return lp, p, tl.where(mask, _order(p, tie), _PAD)


@triton.jit
def _blocks(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    filter_top_k,
    probabilities,
    probability_stride,
    probability_rows,
    block_max,
    block_arg,
    block_apmax,
    block_asum,
    runs,
    runs_ids,
    lengths,
    V,
    NB,
    CAP: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Block max and smallest-ID argmax, source A block max (and sum without min-p), and
    the raw-order sort for a top-k filter."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    x = tl.load(base + ids, mask=mask, other=_NEG).to(tl.float32)
    m = tl.max(x, 0)
    slot = row.to(tl.int64) * NB + block
    tl.store(block_max + slot, m)
    tl.store(block_arg + slot, tl.min(tl.where(mask & (x == m), ids, 2147483647), 0).to(tl.int32))
    if SOURCE_A:
        pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
        pa = tl.load(pbase + ids, mask=mask, other=0.0)
        tl.store(block_apmax + slot, tl.max(pa, 0))
        if not HAS_MIN_P:
            # Without min-p the retained mass is all of it (empty rows: see _state_lse).
            tl.store(block_asum + slot, tl.sum(pa, 0))
    if tl.load(filter_top_k + row) < V:
        keys = tl.where(mask, _order_key(x, ids), _PAD)
        _store_run(keys, ids, mask, _PAD, slot, runs, runs_ids, lengths, runs, CAP, False, BLOCK)
    else:
        tl.store(lengths + slot, 0)


@triton.jit
def _pairwise(lanes, N: tl.constexpr, LOG: tl.constexpr):
    """Natural-order pairwise tree over N lanes: S1's warp, then block, butterflies."""
    for level in tl.static_range(1, LOG + 1):
        a, b = tl.split(tl.reshape(lanes, [N >> level, 2]))
        lanes = a + b
    return tl.sum(lanes, 0)


@triton.jit
def _boundary(runs, row, V, fk):
    """The top-k boundary (fk-th of value desc, ID asc) key and ID; ID 0 when unfiltered."""
    key = tl.load(runs + row.to(tl.int64) * V + tl.maximum(fk - 1, 0))
    filtered = fk < V
    return tl.where(filtered, key, _PAD), tl.where(filtered, tl.minimum(_key_id(key), V - 1), 0)


@triton.jit
def _weights(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    block_max,
    block_arg,
    runs_k,
    chunk_above,
    chunk_all,
    chunk_count,
    V,
    NB,
    NC,
    TILE: tl.constexpr,
):
    """S1 temperature, top-k mask and exp weights; per-chunk S1 butterfly sums."""
    row, chunk = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    m, empty, _ = _row_state(block_max, block_arg, row, NB, TILE)
    t, _ = _temperature(temperatures, row)
    fk = tl.load(filter_top_k + row)
    filtered = fk < V
    tau_key, tau_id = _boundary(runs_k, row, V, fk)
    tau = tl.where(empty, 0.0, tl.load(base + tau_id).to(tl.float32))
    lanes = tl.arange(0, 512).to(tl.int64) * 4
    sum_all = tl.zeros([512], tl.float32)
    sum_above = tl.zeros([512], tl.float32)
    count = tl.zeros([512], tl.int32)
    for part in tl.static_range(8):  # S1 lane order: tile-major, then component
        ids = chunk.to(tl.int64) * 4096 + (part // 4) * 2048 + lanes + part % 4
        mask = ids < V
        raw = tl.load(base + ids, mask=mask, other=0.0).to(tl.float32)
        x, z = _z(raw, empty, m, t)
        keep = mask & (~filtered | (_order_key(raw, ids) >= tau_key))
        w = tl.where(keep, libdevice.exp(z), 0.0)
        above = keep & filtered & (x > tau)
        sum_all = sum_all + w
        sum_above = sum_above + tl.where(above, w, 0.0)
        count += above.to(tl.int32)
    slot = row.to(tl.int64) * NC + chunk
    tl.store(chunk_all + slot, _pairwise(sum_all, 512, 9))
    tl.store(chunk_above + slot, _pairwise(sum_above, 512, 9))
    tl.store(chunk_count + slot, tl.sum(count, 0))


@triton.jit
def _state_total(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    block_max,
    block_arg,
    block_apmax,
    runs_k,
    chunk_above,
    chunk_all,
    chunk_count,
    fstate,
    istate,
    V,
    NB,
    NC,
    SOURCE_A: tl.constexpr,
    TILE: tl.constexpr,
):
    """Row state after the weights: max, argmax, top-k boundary and the S1 total."""
    row = tl.program_id(0)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    m, empty, argmax = _row_state(block_max, block_arg, row, NB, TILE)
    t, _ = _temperature(temperatures, row)
    fk = tl.load(filter_top_k + row)
    tau_key, tau_id = _boundary(runs_k, row, V, fk)
    total = tl.zeros([], tl.float32)
    total_above = tl.zeros([], tl.float32)
    above = tl.zeros([], tl.int64)
    for c in range(0, NC):  # S1: sequential chunk accumulation
        slot = row.to(tl.int64) * NC + c
        total = _rn(total, tl.load(chunk_all + slot), "add")
        total_above = _rn(total_above, tl.load(chunk_above + slot), "add")
        above += tl.load(chunk_count + slot)
    one = tl.arange(0, 1)
    _, tau_z = _z(tl.load(base + tau_id + one).to(tl.float32), empty, m, t)
    ties = tl.broadcast_to((fk - above).to(tl.float32), [1])
    boundary = _rn(tl.broadcast_to(total_above, [1]), _rn(ties, libdevice.exp(tau_z), "mul"), "add")
    f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
    tl.store(f + F_M, m)
    tl.store(f + F_TOTAL, tl.where(fk < V, tl.sum(boundary, 0), total))
    tl.store(f + F_SCALE, 1.0)
    tl.store(f + F_PMAX, 0.0)
    tl.store(f + F_APMAX, 0.0)
    if SOURCE_A:
        tl.store(f + F_APMAX, tl.where(empty, 0.0, _row_max(block_apmax, row, NB, TILE)))
    tl.store(i + I_EMPTY, empty.to(tl.int64))
    tl.store(i + I_ARGMAX, argmax.to(tl.int64))
    tl.store(i + I_TAU, tau_key)
    tl.store(i + I_CUTOFF, _PAD)


@triton.jit
def _probabilities(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    fstate,
    istate,
    runs,
    runs_ids,
    lengths,
    V,
    NB,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Probability-order block sort for a top-p filter (pre-top-p masses)."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    slot = row.to(tl.int64) * NB + block
    if tl.load(top_p + row) == 1:
        tl.store(lengths + slot, 0)
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    _, _, _, p = _token(
        base, ids, mask, row, fstate, istate, temperatures, filter_top_k, top_p, min_p, V, False
    )
    keys = tl.where(mask, _order_key(p, ids), _PAD)
    _store_run(keys, ids, mask, _PAD, slot, runs, runs_ids, lengths, runs, CAP, False, BLOCK)


@triton.jit
def _masses(
    base,
    sorted_ids,
    positions,
    mask,
    row,
    fstate,
    istate,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    V,
):
    """IDs, pre-top-p masses and floor(FP32(p) * 2**62) at sorted positions."""
    ids = tl.load(sorted_ids + positions, mask=mask, other=0).to(tl.int64)
    _, _, _, p = _token(
        base, ids, mask, row, fstate, istate, temperatures, filter_top_k, top_p, min_p, V, False
    )
    return ids, p, tl.where(mask, (p.to(tl.float64) * _MASS).to(tl.int64), 0)


@triton.jit
def _mass_blocks(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    fstate,
    istate,
    sorted_ids,
    block_mass,
    V,
    NB,
    BLOCK: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    if tl.load(top_p + row) == 1:
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    positions = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    _, _, mass = _masses(
        base,
        sorted_ids + row.to(tl.int64) * V,
        positions,
        positions < V,
        row,
        fstate,
        istate,
        temperatures,
        filter_top_k,
        top_p,
        min_p,
        V,
    )
    tl.store(block_mass + row.to(tl.int64) * NB + block, tl.sum(mass, 0))


@triton.jit
def _state_top_p(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    fstate,
    istate,
    sorted_ids,
    block_mass,
    V,
    NB,
    BLOCK: tl.constexpr,
):
    """Top-p prefix (preceding int64 fixed mass < threshold), its scale, and the min-p pivot."""
    row = tl.program_id(0)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
    top = tl.load(top_p + row)
    lanes = tl.arange(0, BLOCK)
    cutoff = tl.full([], _PAD, tl.int64)
    scale = tl.full([], 1.0, tl.float32)
    if top != 1:
        threshold = libdevice.ceil(top.to(tl.float64) * _MASS).to(tl.int64)
        carry = tl.zeros([], tl.int64)
        last_block = tl.zeros([], tl.int64)
        last_preceding = tl.zeros([], tl.int64)
        for start in range(0, NB, BLOCK):
            b = start + lanes
            mass = tl.load(block_mass + row.to(tl.int64) * NB + b, mask=b < NB, other=0)
            preceding = carry + tl.cumsum(mass, 0) - mass
            kept = (b < NB) & (preceding < threshold)
            last_block = tl.maximum(last_block, tl.max(tl.where(kept, b, -1), 0).to(tl.int64))
            last_preceding = tl.maximum(last_preceding, tl.max(tl.where(kept, preceding, -1), 0))
            carry += tl.sum(mass, 0)
        positions = last_block * BLOCK + lanes
        mask = positions < V
        ids, p, mass = _masses(
            base,
            sorted_ids + row.to(tl.int64) * V,
            positions,
            mask,
            row,
            fstate,
            istate,
            temperatures,
            filter_top_k,
            top_p,
            min_p,
            V,
        )
        preceding = last_preceding + tl.cumsum(mass, 0) - mass
        kept = mask & (preceding < threshold)
        kept_mass = last_preceding + tl.sum(tl.where(kept, mass, 0), 0)
        cutoff = tl.min(tl.where(kept, _order_key(p, ids), 9223372036854775807), 0)
        scale = kept_mass.to(tl.float32) * _INV_MASS
    # The argmax survives every filter and has the largest mass: it is the min-p pivot.
    argmax = tl.load(i + I_ARGMAX)
    t, _ = _temperature(temperatures, row)
    raw = tl.load(base + argmax + lanes * 0, mask=lanes == 0, other=0.0).to(tl.float32)
    _, _, p2, _ = _derive(
        raw,
        argmax + lanes * 0,
        lanes == 0,
        tl.load(f + F_M),
        tl.load(i + I_EMPTY) != 0,
        t,
        tl.load(filter_top_k + row) < V,
        tl.load(i + I_TAU),
        tl.load(f + F_TOTAL),
        top == 1,
        cutoff,
        scale,
        0.0,
        0.0,
        False,
        argmax,
        False,
    )
    tl.store(i + I_CUTOFF, cutoff)
    tl.store(f + F_SCALE, scale)
    tl.store(f + F_PMAX, tl.max(tl.where(lanes == 0, p2, 0.0), 0))


@triton.jit
def _unfiltered(temperatures, filter_top_k, top_p, min_p, row, V, HAS_MIN_P: tl.constexpr):
    """T > 0 with identity top-k, top-p and min-p: the support is the whole row."""
    identity = (tl.load(temperatures + row) > 0) & (tl.load(filter_top_k + row) >= V)
    identity = identity & (tl.load(top_p + row) == 1)
    if HAS_MIN_P:
        identity = identity & (tl.load(min_p + row) == 0)
    return identity


@triton.jit
def _exp_sums(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    probability_stride,
    probability_rows,
    fstate,
    istate,
    block_esum,
    block_asum,
    V,
    NB,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Block sums of exp(zz) (the maximum of zz is zero) and of source A's retained masses."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    slot = row.to(tl.int64) * NB + block
    # Unfiltered rows reuse the S1 total (exp of every z): _state_lse skips these sums.
    if not _unfiltered(temperatures, filter_top_k, top_p, min_p, row, V, HAS_MIN_P):
        zz, _, _, _ = _token(
            base,
            ids,
            mask,
            row,
            fstate,
            istate,
            temperatures,
            filter_top_k,
            top_p,
            min_p,
            V,
            HAS_MIN_P,
        )
        tl.store(block_esum + slot, tl.sum(tl.where(zz > _NEG, libdevice.exp(zz), 0.0), 0))
    if SOURCE_A and HAS_MIN_P:
        f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
        empty = tl.load(i + I_EMPTY) != 0
        pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
        mp = _min_p(min_p, row, HAS_MIN_P)
        pa = _masked_a(pbase, ids, mask, mask & ~empty, mp, tl.load(f + F_APMAX), HAS_MIN_P)
        tl.store(block_asum + slot, tl.sum(pa, 0))


@triton.jit
def _state_lse(
    num_rows,
    first,
    selected_ids,
    fstate,
    istate,
    block_esum,
    block_asum,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    V,
    NB,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    TILE: tl.constexpr,
):
    """log(sum exp(zz)); source A's normalizer and whether its retained mass is zero.

    Unfiltered rows keep every token, so sum exp(zz) is the S1 total itself (the
    reduction order of this sum is not part of the contract; ranks never use it).
    """
    row = tl.program_id(0)
    if not _active(num_rows, first, selected_ids, row):
        return
    f, i = fstate + row.to(tl.int64) * NF, istate + row.to(tl.int64) * NI
    if _unfiltered(temperatures, filter_top_k, top_p, min_p, row, V, HAS_MIN_P):
        tl.store(f + F_LOGS, libdevice.log(tl.load(f + F_TOTAL)))
    else:
        tl.store(f + F_LOGS, libdevice.log(_row_sum(block_esum, row, NB, TILE)))
    empty = tl.load(i + I_EMPTY) != 0
    normalizer = 0.0
    if SOURCE_A:
        # Empty logits rows retain no supplied mass, whatever the block sums hold.
        total = tl.where(empty, 0.0, _row_sum(block_asum, row, NB, TILE))
        empty = empty | (total == 0)
        normalizer = libdevice.log(tl.where(empty, 1.0, total))
    tl.store(f + F_NORM, normalizer)
    tl.store(i + I_EMPTY_KEY, empty.to(tl.int64))


@triton.jit
def _keys(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    probability_stride,
    probability_rows,
    fstate,
    istate,
    block_rank,
    block_key,
    V,
    NB,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rank counts against the selected token's key, and per-block maximum keys."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
    lanes = tl.arange(0, BLOCK)
    ids = block.to(tl.int64) * BLOCK + lanes
    mask = ids < V
    selected = tl.minimum(tl.maximum(tl.load(selected_ids + row), 0), V - 1).to(tl.int64)
    _, _, sel = _keys_of(
        base,
        pbase,
        selected + lanes * 0,
        lanes == 0,
        row,
        fstate,
        istate,
        temperatures,
        filter_top_k,
        top_p,
        min_p,
        V,
        HAS_MIN_P,
        SOURCE_A,
    )
    sel_key = tl.max(sel, 0)
    _, _, keys = _keys_of(
        base,
        pbase,
        ids,
        mask,
        row,
        fstate,
        istate,
        temperatures,
        filter_top_k,
        top_p,
        min_p,
        V,
        HAS_MIN_P,
        SOURCE_A,
    )
    slot = row.to(tl.int64) * NB + block
    tl.store(block_rank + slot, tl.sum((mask & (keys >= sel_key)).to(tl.int32), 0))
    tl.store(block_key + slot, tl.max(keys, 0))


@triton.jit
def _collect(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    probability_stride,
    probability_rows,
    fstate,
    istate,
    block_key,
    threshold,
    counter,
    candidate_keys,
    candidate_ids,
    V,
    NB,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Append every order key at or above the threshold to the row's candidates."""
    row, block = tl.program_id(0), tl.program_id(1)
    if not _active(num_rows, first, selected_ids, row):
        return
    t = tl.load(threshold + row)
    if tl.load(block_key + row.to(tl.int64) * NB + block) < t:
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    _, _, keys = _keys_of(
        base,
        pbase,
        ids,
        mask,
        row,
        fstate,
        istate,
        temperatures,
        filter_top_k,
        top_p,
        min_p,
        V,
        HAS_MIN_P,
        SOURCE_A,
    )
    _append(keys, ids, mask & (keys >= t), row, counter, candidate_keys, candidate_ids, CMAX)


@triton.jit
def _sort(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    probability_stride,
    probability_rows,
    fstate,
    istate,
    block_key,
    threshold,
    counter,
    runs,
    runs_ids,
    lengths,
    staging,
    V,
    NB,
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    CMAX: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fallback past CMAX candidates: stable sorts of blocks reaching the threshold."""
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
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
    ids = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < V
    _, _, keys = _keys_of(
        base,
        pbase,
        ids,
        mask,
        row,
        fstate,
        istate,
        temperatures,
        filter_top_k,
        top_p,
        min_p,
        V,
        HAS_MIN_P,
        SOURCE_A,
    )
    _store_run(keys, ids, mask, t, slot, runs, runs_ids, lengths, staging, CAP, True, BLOCK)


@triton.jit
def _finalize(
    logits,
    row_stride,
    storage_rows,
    row_indices,
    num_rows,
    first,
    selected_ids,
    top_k,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    probability_stride,
    probability_rows,
    fstate,
    istate,
    block_rank,
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
    HAS_MIN_P: tl.constexpr,
    SOURCE_A: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    out_row = row.to(tl.int64) * K
    if not _active(num_rows, first, selected_ids, row):
        tl.store(out_selected + row, 0.0)
        tl.store(out_ranks + row, 0)
        for start in range(0, K, TILE):
            columns = start + lanes
            tl.store(out_ids + out_row + columns, -1, mask=columns < K)
            tl.store(out_values + out_row + columns, 0.0, mask=columns < K)
        return
    base = _source(logits, row_stride, storage_rows, row_indices, row)
    pbase = _source(probabilities, probability_stride, probability_rows, row_indices, row)
    t = _temperature(temperatures, row)[0]
    normalizer = tl.load(fstate + row.to(tl.int64) * NF + F_NORM)
    tl.store(out_ranks + row, _row_sum(block_rank, row, NB, TILE))
    selected = tl.minimum(tl.maximum(tl.load(selected_ids + row), 0), V - 1).to(tl.int64)
    length = tl.load(top_k + row)
    # Column -1 is the selected token; columns 0.. are the ordered list.
    for start in range(-TILE, K, TILE):
        columns = start + lanes
        live = (columns >= 0) & (columns < length)
        listed = tl.load(ordered + out_row + columns, mask=live, other=0).to(tl.int64)
        ids = tl.where(columns < 0, selected, listed)
        want = live | (columns == -1)
        lp, p, keys = _keys_of(
            base,
            pbase,
            ids,
            want,
            row,
            fstate,
            istate,
            temperatures,
            filter_top_k,
            top_p,
            min_p,
            V,
            HAS_MIN_P,
            SOURCE_A,
        )
        value = lp
        support = lp > _NEG
        if SOURCE_A:
            support = p > 0
            if not LOGITS:
                value = libdevice.log(p) - normalizer
        if LOGITS:
            x = tl.load(base + ids, mask=want, other=0.0).to(tl.float32)
            value = tl.where(support, _rn(x, tl.broadcast_to(t, x.shape), "div"), _NEG)
        if FLOOR:
            value = tl.where(value == _NEG, floor, value)
        if start < 0:
            tl.store(out_selected + row + lanes * 0, value, mask=columns == -1)
        else:
            ids32 = tl.where(live, ids.to(tl.int32), -1)
            tl.store(out_ids + out_row + columns, ids32, mask=columns < K)
            tl.store(out_values + out_row + columns, tl.where(live, value, 0.0), mask=columns < K)


def run_processed(
    logits,
    row_indices,
    num_rows,
    first,
    selected_ids,
    top_k,
    values,
    floor,
    temperatures,
    filter_top_k,
    top_p,
    min_p,
    probabilities,
    out,
):
    """Rows first.. of the call; out = (selected, ranks, top_ids, top_values), all written."""
    n, vocab = row_indices.numel(), logits.shape[1]
    capacity = out[2].shape[1]
    if n == 0:
        return
    device = logits.device
    blocks, chunks = triton.cdiv(vocab, BLOCK), triton.cdiv(vocab, CHUNK)

    def scratch(dtype, width):
        return torch.empty((n, width), dtype=dtype, device=device)

    f32, i32, i64 = torch.float32, torch.int32, torch.int64
    block_max, block_arg = scratch(f32, blocks), scratch(i32, blocks)
    block_apmax, block_esum, block_asum = (scratch(f32, blocks) for _ in range(3))
    block_mass, block_rank = scratch(i64, blocks), scratch(i32, blocks)
    chunk_all, chunk_above, chunk_count = (
        scratch(f32, chunks),
        scratch(f32, chunks),
        scratch(i64, chunks),
    )
    fstate, istate = scratch(f32, NF.value), scratch(i64, NI.value)
    runs_k = SortedRuns(n, vocab, BLOCK, vocab, device, stable=True)
    runs_p = SortedRuns(n, vocab, BLOCK, vocab, device, stable=True)
    top = TopSelection(n, vocab, max(capacity, 1), device, stable=True)
    source_a, has_min_p = probabilities is not None, min_p is not None
    probabilities = logits if probabilities is None else probabilities  # never read when absent
    min_p = temperatures if min_p is None else min_p  # never read when absent
    gate = (num_rows, first, selected_ids)
    rows = (logits, logits.stride(0), logits.shape[0], row_indices, *gate)
    a = (probabilities, probabilities.stride(0), probabilities.shape[0])
    params = (temperatures, filter_top_k, top_p, min_p)
    state = (fstate, istate)
    flags = {"HAS_MIN_P": has_min_p, "SOURCE_A": source_a}
    _blocks[(n, blocks)](
        *rows,
        filter_top_k,
        *a,
        block_max,
        block_arg,
        block_apmax,
        block_asum,
        runs_k.keys[0],
        runs_k.ids[0],
        runs_k.lengths[0],
        vocab,
        blocks,
        CAP=runs_k.cap0,
        **flags,
        BLOCK=BLOCK,
    )
    sorted_k = runs_k.merge(None, *gate)[0]
    _weights[(n, chunks)](
        *rows,
        temperatures,
        filter_top_k,
        block_max,
        block_arg,
        sorted_k,
        chunk_above,
        chunk_all,
        chunk_count,
        vocab,
        blocks,
        chunks,
        TILE=FINAL_TILE,
    )
    _state_total[(n,)](
        *rows,
        temperatures,
        filter_top_k,
        block_max,
        block_arg,
        block_apmax,
        sorted_k,
        chunk_above,
        chunk_all,
        chunk_count,
        *state,
        vocab,
        blocks,
        chunks,
        SOURCE_A=source_a,
        TILE=FINAL_TILE,
    )
    _probabilities[(n, blocks)](
        *rows,
        *params,
        *state,
        runs_p.keys[0],
        runs_p.ids[0],
        runs_p.lengths[0],
        vocab,
        blocks,
        CAP=runs_p.cap0,
        BLOCK=BLOCK,
    )
    sorted_p_ids = runs_p.merge(None, *gate)[1]
    _mass_blocks[(n, blocks)](
        *rows, *params, *state, sorted_p_ids, block_mass, vocab, blocks, BLOCK=BLOCK
    )
    _state_top_p[(n,)](*rows, *params, *state, sorted_p_ids, block_mass, vocab, blocks, BLOCK=BLOCK)
    _exp_sums[(n, blocks)](
        *rows, *params, *a, *state, block_esum, block_asum, vocab, blocks, **flags, BLOCK=BLOCK
    )
    _state_lse[(n,)](
        *gate, *state, block_esum, block_asum, *params, vocab, blocks, **flags, TILE=FINAL_TILE
    )
    _keys[(n, blocks)](
        *rows,
        *params,
        *a,
        *state,
        block_rank,
        top.block_key,
        vocab,
        blocks,
        **flags,
        BLOCK=BLOCK,
    )
    ordered = top.candidate_ids
    if capacity:
        selection = top.prepare(gate)
        _collect[(n, blocks)](
            *rows,
            *params,
            *a,
            *state,
            *selection,
            top.candidate_keys,
            top.candidate_ids,
            vocab,
            blocks,
            **flags,
            CMAX=CANDIDATES,
            BLOCK=BLOCK,
        )
        runs = top.runs
        _sort[(n, blocks)](
            *rows,
            *params,
            *a,
            *state,
            *selection,
            runs.keys[0],
            runs.ids[0],
            runs.lengths[0],
            top.staging,
            vocab,
            blocks,
            **flags,
            CMAX=CANDIDATES,
            CAP=runs.cap0,
            BLOCK=BLOCK,
        )
        ordered = top.finish(gate)[1]
    _finalize[(n,)](
        *rows,
        top_k,
        *params,
        *a,
        *state,
        block_rank,
        ordered,
        *out,
        0.0 if floor is None else floor,
        vocab,
        blocks,
        capacity,
        LOGITS=values == "logits",
        FLOOR=floor is not None,
        **flags,
        TILE=FINAL_TILE,
    )
