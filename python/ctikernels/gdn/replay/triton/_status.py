# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordered, allocation-free status ring kernels. No collective schedule changes."""

import triton
import triton.language as tl


# EXPECTED (the invocation number) is a fixed int64 runtime argument: no value
# specialization (1, multiples of 16, i32 vs i64), so every invocation number
# reuses one compiled variant per MODE.
@triton.jit(do_not_specialize=["EXPECTED"])
def lifecycle(
    REC,
    VALID,
    SEQ,
    POISON,
    RS: tl.constexpr,
    VS: tl.constexpr,
    J: tl.constexpr,
    B: tl.constexpr,
    MODE: tl.constexpr,
    EXPECTED: tl.int64 = 0,
):
    # One CTA checks ALL layers before installing any record or clearing validity.
    seq = tl.load(SEQ)
    good = seq > 0
    for j in range(J):
        r = REC + j * RS
        old = tl.load(r.to(tl.pointer_type(tl.int64)))
        code = tl.load((r + 8).to(tl.pointer_type(tl.int32)))
        done = tl.load((r + 12).to(tl.pointer_type(tl.int32)))
        if MODE == "begin":
            good &= (old == 0) & (code == 0) & (done == 0)
        elif MODE == "clear":
            good &= (old == EXPECTED) & (done == 1) & (seq == EXPECTED)
        else:
            good &= old == seq
    if not good:
        tl.atomic_or(POISON, 64)
        return
    for j in range(J):
        r = REC + j * RS
        sp = r.to(tl.pointer_type(tl.int64))
        cp = (r + 8).to(tl.pointer_type(tl.int32))
        dp = (r + 12).to(tl.pointer_type(tl.int32))
        if MODE == "begin":
            tl.store(sp, seq)
            tl.store(cp, tl.where(tl.load(POISON) != 0, 32, 0))
            tl.store(dp, 0)
        elif MODE == "end":
            if tl.load(POISON) != 0:
                tl.atomic_or(cp, 32)
            tl.store(dp, 1)
        elif MODE == "clear":
            tl.store(sp, 0)
            tl.store(cp, 0)
            tl.store(dp, 0)
        if MODE == "begin" or MODE == "clear":
            for b in range(B):
                tl.store(VALID + j * VS + b, False)


@triton.jit(do_not_specialize=["RD", "SQD"])
def guard(
    REC,
    SEQ,
    POISON,
    SLOT=None,
    HAS_SLOT: tl.constexpr = False,
    SLOT_DEPTH: tl.constexpr = 1,
    RD: tl.int64 = 0,
    SQD: tl.int64 = 0,
):
    if HAS_SLOT:
        invocation = tl.load(SLOT).to(tl.int64)
        if (invocation < 0) | (invocation >= SLOT_DEPTH):
            tl.atomic_or(POISON, 64)
            return
        REC += invocation * RD
        SEQ += invocation * SQD
    seq = tl.load(SEQ)
    old = tl.load(REC.to(tl.pointer_type(tl.int64)))
    done = tl.load((REC + 12).to(tl.pointer_type(tl.int32)))
    cp = (REC + 8).to(tl.pointer_type(tl.int32))
    if (seq <= 0) | (old != seq) | (done != 0):
        tl.atomic_or(POISON, 64)
    elif tl.load(POISON) != 0:
        tl.atomic_or(cp, 32)


@triton.jit(do_not_specialize=["SD_STATUS"])
def propagate(
    CODE,
    POISON,
    SLOT=None,
    HAS_SLOT: tl.constexpr = False,
    SLOT_DEPTH: tl.constexpr = 1,
    SD_STATUS: tl.int64 = 0,
):
    if HAS_SLOT:
        invocation = tl.load(SLOT).to(tl.int64)
        if (invocation < 0) | (invocation >= SLOT_DEPTH):
            return
        CODE += invocation * SD_STATUS
    tl.atomic_or(POISON, tl.load(CODE))


@triton.jit
def invalidate(
    M,
    SM,
    PM,
    E,
    R: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    REV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Quiescent whole-pool reset: invalidation precedes poison clear.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    f = i % 8
    gen = tl.load(M + i, (i < R * 8) & (f == 5), 0)
    epoch = tl.load(E) + 1
    value = tl.where(f == 5, gen + 1, tl.where(f == 6, REV, tl.where(f == 7, epoch, 0)))
    tl.store(M + i, value, i < R * 8)
    tl.store(SM + i, 0, i < S * 4)
    tl.store(PM + i, 0, i < N)


@triton.jit
def recovery_finish(E):
    tl.store(E, tl.load(E) + 1)


@triton.jit
def clear_poison(POISON):
    tl.store(POISON, 0)
