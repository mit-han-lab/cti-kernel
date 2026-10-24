# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""All-layer handoff over the bound layer table: ordered strict control, then bit-preserving
payload copies (the per-layer handoff's result, bit for bit)."""

import triton
import triton.language as tl


@triton.jit
def _read(ptr):
    # Coherent with this program's own earlier atomics.
    return tl.atomic_or(ptr, 0)


@triton.jit(do_not_specialize=["batch", "IS", "PS"])
def control(
    TABLE,
    POS,
    IDX,
    REC,
    SEQ,
    POISON,
    batch,
    IS,
    PS,
    SLOTS: tl.constexpr,
    LAYERS: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Row errors once for all layers, then each layer's record in order (strict only)."""
    error = tl.zeros((), tl.int32)
    inplace_error = tl.zeros((), tl.int32)
    if _read(POISON) == 0:
        for start in range(0, batch, BLOCK):
            b = start + tl.arange(0, BLOCK)
            inside = b < batch
            slot = tl.load(IDX + b.to(tl.int64) * IS, inside, -1).to(tl.int64)
            position = tl.load(POS + b.to(tl.int64) * PS, inside, 0).to(tl.int64)
            used = slot != -1
            bad = used & ((slot < 0) | (slot >= SLOTS))
            row = tl.max(tl.where(used & ~bad & (position < 0), 2, 0), 0)
            moved = used & ~bad & (slot != b)
            error |= tl.max(bad.to(tl.int32), 0) | row
            inplace_error |= tl.max((bad | moved).to(tl.int32), 0) | row
    seq = tl.load(SEQ)
    for j in tl.static_range(LAYERS):
        record = REC + j * RS
        old = tl.load(record.to(tl.pointer_type(tl.int64)))
        code = (record + 8).to(tl.pointer_type(tl.int32))
        done = tl.load((record + 12).to(tl.pointer_type(tl.int32)))
        if (seq <= 0) | (old != seq) | (done != 0):
            tl.atomic_or(POISON, 64)
        elif _read(POISON) != 0:
            tl.atomic_or(code, 32)
        if batch > 0:
            if _read(POISON) == 0:
                inplace = (tl.load(TABLE + j * 8 + 7) & 1) != 0
                tl.atomic_or(code, tl.where(inplace, inplace_error, error))
            tl.atomic_or(POISON, _read(code))


@triton.jit(do_not_specialize=["IS", "PS"])
def payload(
    TABLE,
    PITCH,
    POS,
    IDX,
    REC,
    SEQ,
    IS,
    PS,
    SLOTS: tl.constexpr,
    WINDOW: tl.constexpr,
    DC: tl.constexpr,
    HV: tl.constexpr,
    RS: tl.constexpr,
    STRICT: tl.constexpr,
    RINGLESS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (chunk, row b, layer j) copies its share of row b's state into slot row."""
    chunk, b, j = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    chunks = tl.num_programs(0)
    allowed = tl.full((), 1, tl.int1)
    if STRICT:
        # A later layer may fail after this one earned its write: require the whole prefix.
        seq = tl.load(SEQ)
        for i in range(j + 1):
            record = REC + i * RS
            old = tl.load(record.to(tl.pointer_type(tl.int64)))
            code = tl.load((record + 8).to(tl.pointer_type(tl.int32)))
            done = tl.load((record + 12).to(tl.pointer_type(tl.int32)))
            allowed = allowed & (seq > 0) & (old == seq) & (done == 0) & (code == 0)
    slot = tl.load(IDX + b * IS).to(tl.int64)
    position = tl.load(POS + b * PS).to(tl.int64)
    entry = TABLE + j * 8
    flags = tl.load(entry + 7)
    inplace = (flags & 1) != 0
    has_ring = (flags & 2) == 0
    active = allowed & (slot >= 0) & (slot < SLOTS) & (position >= 0)
    active = active & (~inplace | (slot == b))
    if active:
        pitches = PITCH + j * 4
        cp_pitch, h_pitch = tl.load(pitches), tl.load(pitches + 1)
        src_cp_pitch, src_h_pitch = tl.load(pitches + 2), tl.load(pitches + 3)
        # Row addresses in bytes: a row that is its own destination is not copied.
        dst_cp = tl.load(entry) + slot * cp_pitch * 4
        dst_h = tl.load(entry + 1) + slot * h_pitch * 2
        src_cp = tl.load(entry + 5) + b * src_cp_pitch * 4
        src_h = tl.load(entry + 6) + b * src_h_pitch * 2
        ring = tl.load(entry + 2).to(tl.pointer_type(tl.bfloat16))
        metadata = tl.load(entry + 3).to(tl.pointer_type(tl.int64))
        epoch = tl.load(entry + 4).to(tl.pointer_type(tl.int64))
        lane = tl.arange(0, BLOCK).to(tl.int64)
        stride = chunks * BLOCK
        state = HV * 128 * 128
        if src_cp != dst_cp:
            source_state = src_cp.to(tl.pointer_type(tl.float32))
            target_state = dst_cp.to(tl.pointer_type(tl.float32))
            for start in range(chunk * BLOCK, state, stride):
                v = start + lane
                tl.store(target_state + v, tl.load(source_state + v, v < state), v < state)
        copy_history = src_h != dst_h
        src_h = src_h.to(tl.pointer_type(tl.bfloat16))
        dst_h = dst_h.to(tl.pointer_type(tl.bfloat16))
        width = DC + 2 * HV
        seeds = ring + (slot * (WINDOW + 3) + WINDOW) * width
        for start in range(chunk * BLOCK, 3 * DC, stride):
            v = start + lane
            inside = v < 3 * DC
            value = tl.load(src_h + v, inside)
            if copy_history:
                tl.store(dst_h + v, value, inside)
            if not RINGLESS:
                if has_ring:
                    tl.store(seeds + (v // DC) * width + v % DC, value, inside)
        if not RINGLESS:
            if has_ring:
                for start in range(chunk * BLOCK, 3 * 2 * HV, stride):
                    v = start + lane
                    inside = v < 3 * 2 * HV
                    zero = tl.zeros((BLOCK,), tl.bfloat16)
                    tl.store(seeds + (v // (2 * HV)) * width + DC + v % (2 * HV), zero, inside)
        if chunk == 0:
            base = metadata + slot * 8
            current = tl.load(epoch)
            fresh = (tl.load(base + 4) != 1) | (tl.load(base + 7) != current)
            generation = tl.load(base + 5) + fresh.to(tl.int64)
            tl.store(base, position)
            tl.store(base + 1, position)
            tl.store(base + 2, 0)
            tl.store(base + 3, 0)
            tl.store(base + 4, 1)
            tl.store(base + 5, generation)
            tl.store(base + 7, current)
