# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Original portable kernels for replay contract §§3–8.

Sequential state updates use one CTA per (batch row, value head), with no inter-CTA
scratch, atomics or barriers in arithmetic. Ordered launches validate all rows,
compute state/output, then publish ring/history/metadata. Reference: an earlier delayed-decode experiment informed raw replay and
compensated convolution.
"""

import os

import triton
import triton.language as tl
from triton.language.extra import libdevice

# LT is the constexpr tuple of _layout.Layout fields, in declaration order:
# checkpoint/history/canonical/source_history, indices/pair/positions,
# accepted/snapshots/offsets/valid, token/out_batch/out_token, then alignments.
# These constants change addressing only; the portable arithmetic stays frozen.

# CPU interpreter lacks libdevice; production always uses accurate log1p.
_LIBDEVICE = tl.constexpr(os.environ.get("TRITON_INTERPRET", "0") != "1")


@triton.jit
def _source(INDICES, b, MODE: tl.constexpr, LT: tl.constexpr):
    if MODE == "export_snapshot" or MODE == "import_snapshot":
        source = tl.load(INDICES + b * LT[4]).to(tl.int64)
        target = tl.load(INDICES + b * LT[4] + LT[5]).to(tl.int64)
    else:
        source = tl.load(INDICES + b * LT[4]).to(tl.int64)
        target = source
    slot = target if MODE == "import_snapshot" else source
    return source, target, slot


@triton.jit
def _pending_row(PM, slot, b, PB: tl.constexpr, MODE: tl.constexpr):
    row = b.to(tl.int64)
    if MODE == "export_snapshot":
        row = tl.full((), -1, tl.int64)
        for j in range(PB):
            match = (tl.load(PM + j * 8) == slot) & (
                (tl.load(PM + j * 8 + 5) == 1) | (tl.load(PM + j * 8 + 5) == 2)
            )
            row = tl.where(match, j, row)
    return row


@triton.jit(do_not_specialize=["SD_STATUS", "SD_PR", "SD_PM", "SD_VALID"])
def validate(
    INDICES,
    POS,
    M,
    E,
    SM,
    PM,
    ACCEPT,
    SR,
    OFF,
    STATUS,
    POISON,
    R: tl.constexpr,
    S: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    PB: tl.constexpr,
    MODE: tl.constexpr,
    CAPTURE: tl.constexpr,
    INPLACE: tl.constexpr,
    LT: tl.constexpr,
    SLOT=None,
    HAS_SLOT: tl.constexpr = False,
    SLOT_DEPTH: tl.constexpr = 1,
    SD_STATUS: tl.int64 = 0,
    SD_PR: tl.int64 = 0,
    SD_PM: tl.int64 = 0,
    SD_VALID: tl.int64 = 0,
):
    if HAS_SLOT:
        invocation = tl.load(SLOT).to(tl.int64)
        if (invocation < 0) | (invocation >= SLOT_DEPTH):
            return
        STATUS += invocation * SD_STATUS
        PM += invocation * SD_PM
    if tl.load(POISON) != 0:
        return
    b = tl.program_id(0).to(tl.int64)
    source, target, slot = _source(INDICES, b, MODE, LT)
    if (source == -1) | (target == -1):
        return
    if (slot < 0) | (slot >= R):
        tl.atomic_or(STATUS, 1)
        return
    epoch = tl.load(E)
    base = M + slot * 8
    cp, pos = tl.load(base), tl.load(base + 1)
    cursor, n = tl.load(base + 2), tl.load(base + 3)
    error = tl.full((), 0, tl.int32)
    if MODE == "import_snapshot":
        if (source < R) | (source >= R + S):
            tl.atomic_or(STATUS, 16)
            return
        sm = SM + (source - R) * 4
        good = (
            (tl.load(sm + 3) == 1)
            & (tl.load(sm + 1) == tl.load(base + 6))
            & (tl.load(sm + 2) == epoch)
        )
        error |= tl.where(good, 0, 16)
    elif MODE != "initialize" and MODE != "handoff" and MODE != "release":
        good = (tl.load(base + 4) == 1) & (tl.load(base + 7) == epoch)
        good &= (n >= 0) & (n < L) & (pos - cp == n) & (cursor >= 0) & (cursor < L + 3)
        error |= tl.where(good, 0, 1)
    if MODE == "initialize" or MODE == "handoff":
        error |= tl.where(tl.load(POS + b * LT[6]).to(tl.int64) >= 0, 0, 2)
        if INPLACE:
            error |= tl.where(slot == b, 0, 1)
    if MODE == "step" or MODE == "bridge_step":
        error |= tl.where(
            (tl.load(POS + b * LT[6]).to(tl.int64) == pos) & (pos < 9223372036854775807), 0, 2
        )
    if MODE == "verify":
        for j in range(Q):
            good = (tl.load(POS + b * LT[6] + j).to(tl.int64) == pos + j) & (
                pos <= 9223372036854775807 - Q
            )
            error |= tl.where(good, 0, 2)
    m = tl.full((), 0, tl.int64)
    if MODE == "commit" or MODE == "reject" or (MODE == "export_snapshot" and PB > 0):
        pb = _pending_row(PM, slot, b, PB, MODE)
        if pb < 0:
            tl.atomic_or(STATUS, 8)
            return
        pm = PM + pb * 8
        tx = tl.load(pm + 5)
        good = (tl.load(pm) == slot) & (tl.load(pm + 1) == tl.load(base + 5))
        good &= (tl.load(pm + 2) == cp) & (tl.load(pm + 3) == pos) & (tl.load(pm + 4) == Q)
        good &= ((tx == 1) | (tx == 2)) & (tl.load(pm + 7) == epoch)
        error |= tl.where(good, 0, 8)
        if MODE != "reject":
            m = tl.load(ACCEPT + pb * LT[7]).to(tl.int64)
            error |= tl.where((m >= 0) & (m <= Q), 0, 4)
            error |= tl.where((tx != 2) | (tl.load(pm + 6) == m), 0, 8)
            error |= tl.where(pos <= 9223372036854775807 - tl.maximum(m, 0), 0, 2)
    if MODE == "export_snapshot" or CAPTURE:
        snap = target if MODE == "export_snapshot" else tl.load(SR + b * LT[8]).to(tl.int64)
        if snap != -1:
            error |= tl.where((snap >= R) & (snap < R + S), 0, 16)
            if CAPTURE or PB > 0:
                error |= tl.where(tl.load(OFF + b * LT[9]).to(tl.int64) >= 0, 0, 4)
    tl.atomic_or(STATUS, error)


@triton.jit
def raw(
    RING,
    X,
    PR,
    slot,
    b,
    pb,
    cursor,
    t,
    col,
    mask,
    L: tl.constexpr,
    W: tl.constexpr,
    DC: tl.constexpr,
    HV: tl.constexpr,
    Q: tl.constexpr,
    XS: tl.constexpr,
    MODE: tl.constexpr,
    LT: tl.constexpr,
):
    """t is relative to the old committed frontier; negative rows come from ring."""
    # Verify passes an int32 loop index; widen before any token * width product.
    t = t + tl.zeros((), tl.int64)
    row = ((cursor + t) % (L + 3) + L + 3) % (L + 3)
    value = tl.load(RING + (slot * (L + 3) + row) * W + col, mask=mask & (t < 0), other=0)
    if (MODE == "step" or MODE == "bridge_step") or MODE == "verify":
        projected_col = tl.where(col < DC, col, col + HV * 128)
        current = tl.load(X + b * XS + t * LT[11] + projected_col, mask=mask & (t >= 0), other=0)
        value = tl.where(t >= 0, current, value)
    elif MODE == "commit" or MODE == "export_snapshot":
        if Q > 0:
            current = tl.load(PR + (pb * Q + t) * W + col, mask=mask & (t >= 0), other=0)
            value = tl.where(t >= 0, current, value)
    return value.to(tl.float32)


@triton.jit
def conv(
    RING,
    X,
    PR,
    CW,
    slot,
    b,
    pb,
    cursor,
    t,
    col,
    L: tl.constexpr,
    W: tl.constexpr,
    DC: tl.constexpr,
    HV: tl.constexpr,
    Q: tl.constexpr,
    XS: tl.constexpr,
    MODE: tl.constexpr,
    LT: tl.constexpr,
):
    total = tl.zeros(col.shape, tl.float32)
    correction = tl.zeros(col.shape, tl.float32)
    for tap in tl.static_range(4):
        x = raw(
            RING, X, PR, slot, b, pb, cursor, t - 3 + tap, col, True, L, W, DC, HV, Q, XS, MODE, LT
        )
        product = x * tl.load(CW + col * 4 + tap).to(tl.float32)
        # Error-free TwoSum on exact BF16 products; fma disabled at launch.
        added = total + product
        residual = added - total
        correction += (total - (added - residual)) + (product - residual)
        total = added
    c = total + correction
    return c * tl.sigmoid(c)


@triton.jit(do_not_specialize=["SD_STATUS", "SD_PR", "SD_PM", "SD_VALID"])
def arithmetic(
    C,
    RING,
    M,
    INDICES,
    X,
    PR,
    PM,
    CW,
    AL,
    BIAS,
    NW,
    ACCEPT,
    SR,
    OFF,
    OUT,
    STATUS,
    POISON,
    CANON,
    R: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    PB: tl.constexpr,
    XS: tl.constexpr,
    MODE: tl.constexpr,
    CAPTURE: tl.constexpr,
    LT: tl.constexpr,
    SLOT=None,
    HAS_SLOT: tl.constexpr = False,
    SLOT_DEPTH: tl.constexpr = 1,
    SD_STATUS: tl.int64 = 0,
    SD_PR: tl.int64 = 0,
    SD_PM: tl.int64 = 0,
    SD_VALID: tl.int64 = 0,
    BV: tl.constexpr = 128,
    PRENORM=None,
):
    if HAS_SLOT:
        invocation = tl.load(SLOT).to(tl.int64)
        if (invocation < 0) | (invocation >= SLOT_DEPTH):
            return
        STATUS += invocation * SD_STATUS
        PR += invocation * SD_PR
        PM += invocation * SD_PM
    b, h = tl.program_id(0).to(tl.int64), tl.program_id(1)
    source, target, slot = _source(INDICES, b, MODE, LT)
    idx = tl.arange(0, 128)
    # Value rows of this program: the state's V rows are independent given the key, so
    # output-free modes split them over programs (axis 2); output modes keep all 128.
    rows = tl.program_id(2) * BV + tl.arange(0, BV)
    if (source == -1) | (target == -1) | (tl.load(STATUS) != 0) | (tl.load(POISON) != 0):
        if (MODE == "step" or MODE == "bridge_step") or MODE == "verify":
            for j in range(Q if MODE == "verify" else 1):
                tl.store(OUT + b * LT[12] + j.to(tl.int64) * LT[13] + h * 128 + idx, 0)
        return
    cp_ptr = C + slot * LT[0] + h * 16384 + rows[:, None] * 128 + idx[None, :]
    if MODE == "initialize":
        tl.store(cp_ptr, tl.full((BV, 128), 0, tl.float32))
        return
    if MODE == "handoff" or MODE == "import_snapshot":
        src = b if MODE == "handoff" else source
        src_ptr = CANON if MODE == "handoff" else C
        value = tl.load(
            src_ptr
            + src * (LT[2] if MODE == "handoff" else LT[0])
            + h * 16384
            + rows[:, None] * 128
            + idx[None, :]
        )
        tl.store(cp_ptr, value)
        return
    DC: tl.constexpr = (2 * HK + HV) * 128
    W: tl.constexpr = DC + 2 * HV
    n, cursor = tl.load(M + slot * 8 + 3), tl.load(M + slot * 8 + 2)
    pb = _pending_row(PM, slot, b, PB, MODE) if PB > 0 else b
    m = tl.full((), 0, tl.int64)
    if MODE == "commit" or (MODE == "export_snapshot" and PB > 0):
        m = tl.load(ACCEPT + pb * LT[7]).to(tl.int64)
    end = tl.full((), 0, tl.int64)
    if MODE == "step" or MODE == "bridge_step":
        end = tl.full((), 1, tl.int64)
    elif MODE == "verify":
        end = tl.full((), Q, tl.int64)
    elif MODE == "commit":
        end = m
    snap, k = tl.full((), -1, tl.int64), tl.full((), 0, tl.int64)
    if MODE == "export_snapshot":
        snap = target
        if PB > 0:
            k = tl.load(OFF + b * LT[9]).to(tl.int64)
            end = tl.where(k <= m, k, 0)
    elif CAPTURE:
        snap, k = tl.load(SR + b * LT[8]).to(tl.int64), tl.load(OFF + b * LT[9]).to(tl.int64)
    capture = (snap >= R) & (k <= m)
    matrix = tl.load(cp_ptr).to(tl.float32)
    # At each boundary, storing a snapshot rounds exactly as a commit at that offset.
    # No source checkpoint writes until the full replay finishes.
    for t in range(-n, end + 1):
        if (t == k) & capture:
            tl.store(C + snap * LT[0] + h * 16384 + rows[:, None] * 128 + idx[None, :], matrix)
        if t < end:
            kh = h // (HV // HK)
            key = conv(
                RING,
                X,
                PR,
                CW,
                slot,
                b,
                pb,
                cursor,
                t,
                HK * 128 + kh * 128 + idx,
                L,
                W,
                DC,
                HV,
                Q,
                XS,
                MODE,
                LT,
            )
            val = conv(
                RING,
                X,
                PR,
                CW,
                slot,
                b,
                pb,
                cursor,
                t,
                2 * HK * 128 + h * 128 + rows,
                L,
                W,
                DC,
                HV,
                Q,
                XS,
                MODE,
                LT,
            )
            key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
            a = raw(
                RING,
                X,
                PR,
                slot,
                b,
                pb,
                cursor,
                t,
                DC + HV + h,
                True,
                L,
                W,
                DC,
                HV,
                Q,
                XS,
                MODE,
                LT,
            ) + tl.load(BIAS + h).to(tl.float32)
            beta = tl.sigmoid(
                raw(
                    RING, X, PR, slot, b, pb, cursor, t, DC + h, True, L, W, DC, HV, Q, XS, MODE, LT
                )
            )
            if _LIBDEVICE:
                softplus = tl.maximum(a, 0) + libdevice.log1p(tl.exp(-tl.abs(a)))
            else:
                softplus = tl.maximum(a, 0) + tl.log(1 + tl.exp(-tl.abs(a)))
            alpha = tl.exp(-tl.exp(tl.load(AL + h).to(tl.float32)) * softplus)
            delta = beta * (val - alpha * tl.sum(matrix * key[None, :], 1))
            matrix = alpha * matrix + delta[:, None] * key[None, :]
            if (MODE == "step" or MODE == "bridge_step") or MODE == "verify":
                if t >= 0:
                    query = conv(
                        RING,
                        X,
                        PR,
                        CW,
                        slot,
                        b,
                        pb,
                        cursor,
                        t,
                        kh * 128 + idx,
                        L,
                        W,
                        DC,
                        HV,
                        Q,
                        XS,
                        MODE,
                        LT,
                    )
                    query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6) * 0.08838834764831845
                    if BV < 128:
                        # Split verify: FP32 pre-norm rows; the publish kernel applies the norm.
                        o = tl.sum(matrix * query[None, :], 1)
                        tl.store(PRENORM + ((b * Q + t) * HV + h) * 128 + rows, o)
                    else:
                        o = tl.sum(matrix * query[None, :], 1)
                        z = tl.load(X + b * XS + t * LT[11] + DC + h * 128 + idx).to(tl.float32)
                        y = (
                            tl.load(NW + idx).to(tl.float32)
                            * o
                            * tl.rsqrt(tl.sum(o * o, 0) / 128 + 1e-6)
                            * z
                            * tl.sigmoid(z)
                        )
                        tl.store(OUT + b * LT[12] + t * LT[13] + h * 128 + idx, y)
    if MODE == "commit" or MODE == "flush" or MODE == "bridge_step":
        tl.store(cp_ptr, matrix)
    elif MODE == "step" or MODE == "bridge_step":
        if n + 1 == L:
            tl.store(cp_ptr, matrix)


@triton.jit(do_not_specialize=["SD_STATUS", "SD_PR", "SD_PM", "SD_VALID"])
def publish(
    C,
    H,
    RING,
    M,
    E,
    SM,
    INDICES,
    POS,
    X,
    PR,
    PM,
    ACCEPT,
    SR,
    OFF,
    VALID,
    STATUS,
    POISON,
    HISTORY,
    R: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    L: tl.constexpr,
    Q: tl.constexpr,
    PB: tl.constexpr,
    XS: tl.constexpr,
    MODE: tl.constexpr,
    CAPTURE: tl.constexpr,
    BLOCK: tl.constexpr,
    LT: tl.constexpr,
    RINGLESS: tl.constexpr = False,
    SLOT=None,
    HAS_SLOT: tl.constexpr = False,
    SLOT_DEPTH: tl.constexpr = 1,
    SD_STATUS: tl.int64 = 0,
    SD_PR: tl.int64 = 0,
    SD_PM: tl.int64 = 0,
    SD_VALID: tl.int64 = 0,
    CHUNKS: tl.constexpr = 1,
    PRENORM=None,
    OUT=None,
    NW=None,
    QP: tl.constexpr = 1,
):
    if HAS_SLOT:
        invocation = tl.load(SLOT).to(tl.int64)
        if (invocation < 0) | (invocation >= SLOT_DEPTH):
            return
        STATUS += invocation * SD_STATUS
        PR += invocation * SD_PR
        PM += invocation * SD_PM
        VALID += invocation * SD_VALID
    b = tl.program_id(0).to(tl.int64)
    source, target, slot = _source(INDICES, b, MODE, LT)
    if tl.load(POISON) != 0:
        return
    if MODE == "export_snapshot" or CAPTURE:
        tl.store(VALID + b * LT[10], False)
    if (source == -1) | (target == -1) | (tl.load(STATUS) != 0) | (tl.load(POISON) != 0):
        return
    DC: tl.constexpr = (2 * HK + HV) * 128
    W: tl.constexpr = DC + 2 * HV
    # Verify (which writes no slot metadata) spreads its records over CHUNKS column blocks.
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    meta = M + slot * 8
    pos, n, cursor = tl.load(meta + 1), tl.load(meta + 3), tl.load(meta + 2)
    pb = _pending_row(PM, slot, b, PB, MODE) if PB > 0 else b
    if MODE == "release":
        tl.store(meta + 2, 0)
        tl.store(meta + 3, 0)
        tl.store(meta + 4, 0)
        tl.store(meta + 5, tl.load(meta + 5) + 1)
        return
    if MODE == "reject":
        tl.store(PM + pb * 8 + 5, 4)
        return
    if MODE == "flush":
        tl.store(meta, pos)
        tl.store(meta + 3, 0)
        return
    if MODE == "verify":
        for j in range(Q):
            values = raw(
                RING, X, PR, slot, b, pb, cursor, j, c, c < W, L, W, DC, HV, Q, XS, MODE, LT
            )
            tl.store(PR + (b * Q + j) * W + c, values, c < W)
        if PRENORM is not None:
            # The output gate of a V-split verify, in arithmetic()'s operation order.
            idx = tl.arange(0, 128)[None, :]
            token = tl.arange(0, QP)[:, None]
            inside = token < Q
            weight = tl.load(NW + idx).to(tl.float32)
            for h in range(tl.program_id(1), HV, CHUNKS):
                o = tl.load(PRENORM + ((b * Q + token) * HV + h) * 128 + idx, mask=inside, other=0)
                z = tl.load(X + b * XS + token * LT[11] + DC + h * 128 + idx, mask=inside, other=0)
                z = z.to(tl.float32)
                y = (
                    weight
                    * o
                    * tl.rsqrt(tl.sum(o * o, 1) / 128 + 1e-6)[:, None]
                    * z
                    * tl.sigmoid(z)
                )
                tl.store(OUT + b * LT[12] + token * LT[13] + h * 128 + idx, y, mask=inside)
        if tl.program_id(1) != 0:
            return
        pm = PM + b * 8
        tl.store(pm, slot)
        tl.store(pm + 1, tl.load(meta + 5))
        tl.store(pm + 2, tl.load(meta))
        tl.store(pm + 3, pos)
        tl.store(pm + 4, Q)
        tl.store(pm + 5, 1)
        tl.store(pm + 6, -1)
        tl.store(pm + 7, tl.load(E))
        return
    end = tl.full((), 0, tl.int64)
    m = tl.full((), 0, tl.int64)
    if MODE == "step" or MODE == "bridge_step":
        end = tl.full((), 1, tl.int64)
    if MODE == "commit" or (MODE == "export_snapshot" and PB > 0):
        m = tl.load(ACCEPT + pb * LT[7]).to(tl.int64)
        end = m
    snap, k = tl.full((), -1, tl.int64), tl.full((), 0, tl.int64)
    if MODE == "export_snapshot":
        snap = target
        if PB > 0:
            k = tl.load(OFF + b * LT[9]).to(tl.int64)
    elif CAPTURE:
        snap, k = tl.load(SR + b * LT[8]).to(tl.int64), tl.load(OFF + b * LT[9]).to(tl.int64)
    capture = (snap >= R) & (k <= m)
    if MODE == "initialize":
        h0 = tl.full((BLOCK,), 0, tl.float32)
        h1, h2 = h0, h0
    elif MODE == "handoff" or MODE == "import_snapshot":
        src = b if MODE == "handoff" else source
        hp = HISTORY if MODE == "handoff" else H
        h0 = tl.load(hp + src * (LT[3] if MODE == "handoff" else LT[1]) + c, c < DC, 0)
        h1 = tl.load(hp + src * (LT[3] if MODE == "handoff" else LT[1]) + DC + c, c < DC, 0)
        h2 = tl.load(hp + src * (LT[3] if MODE == "handoff" else LT[1]) + 2 * DC + c, c < DC, 0)
    else:
        h0 = raw(
            RING, X, PR, slot, b, pb, cursor, end - 3, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT
        )
        h1 = raw(
            RING, X, PR, slot, b, pb, cursor, end - 2, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT
        )
        h2 = raw(
            RING, X, PR, slot, b, pb, cursor, end - 1, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT
        )
    # Load every history row before overwriting any rotating ring seed.
    if capture:
        s0 = raw(RING, X, PR, slot, b, pb, cursor, k - 3, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT)
        s1 = raw(RING, X, PR, slot, b, pb, cursor, k - 2, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT)
        s2 = raw(RING, X, PR, slot, b, pb, cursor, k - 1, c, c < DC, L, W, DC, HV, Q, XS, MODE, LT)
        tl.store(H + snap * LT[1] + c, s0, c < DC)
        tl.store(H + snap * LT[1] + DC + c, s1, c < DC)
        tl.store(H + snap * LT[1] + 2 * DC + c, s2, c < DC)
        sm = SM + (snap - R) * 4
        tl.store(sm, pos + k)
        tl.store(sm + 1, tl.load(meta + 6))
        tl.store(sm + 2, tl.load(E))
        tl.store(sm + 3, 1)
        tl.store(VALID + b * LT[10], True)
    if MODE == "export_snapshot":
        if PB > 0:
            # Duplicate read sources are legal; only the first resolves its descriptor.
            first = tl.full((), True, tl.int1)
            for j in range(b):
                first &= (tl.load(INDICES + j * LT[4]).to(tl.int64) != source) | (
                    tl.load(INDICES + j * LT[4] + LT[5]).to(tl.int64) == -1
                )
            if first:
                tl.store(PM + pb * 8 + 5, 2)
                tl.store(PM + pb * 8 + 6, m)
        return
    tl.store(H + slot * LT[1] + c, h0, c < DC)
    tl.store(H + slot * LT[1] + DC + c, h1, c < DC)
    tl.store(H + slot * LT[1] + 2 * DC + c, h2, c < DC)
    if MODE == "step":
        value = raw(RING, X, PR, slot, b, pb, cursor, 0, c, c < W, L, W, DC, HV, Q, XS, MODE, LT)
        tl.store(RING + (slot * (L + 3) + cursor) * W + c, value, c < W)
        tl.store(meta + 1, pos + 1)
        tl.store(meta + 2, (cursor + 1) % (L + 3))
        tl.store(meta + 3, tl.where(n + 1 == L, 0, n + 1))
        if n + 1 == L:
            tl.store(meta, pos + 1)
        return
    if not RINGLESS:
        tl.store(RING + (slot * (L + 3) + L) * W + c, tl.where(c < DC, h0, 0), c < W)
        tl.store(RING + (slot * (L + 3) + L + 1) * W + c, tl.where(c < DC, h1, 0), c < W)
        tl.store(RING + (slot * (L + 3) + L + 2) * W + c, tl.where(c < DC, h2, 0), c < W)
    if MODE == "initialize" or MODE == "handoff":
        next_pos = tl.load(POS + b * LT[6]).to(tl.int64)
    elif MODE == "import_snapshot":
        next_pos = tl.load(SM + (source - R) * 4)
    else:
        next_pos = pos + (1 if MODE == "bridge_step" else m)
    if MODE == "initialize" or MODE == "import_snapshot":
        tl.store(meta + 5, tl.load(meta + 5) + 1)
    elif MODE == "handoff":
        fresh = (tl.load(meta + 4) != 1) | (tl.load(meta + 7) != tl.load(E))
        tl.store(meta + 5, tl.load(meta + 5) + fresh.to(tl.int64))
    tl.store(meta, next_pos)
    tl.store(meta + 1, next_pos)
    tl.store(meta + 2, 0)
    tl.store(meta + 3, 0)
    tl.store(meta + 4, 1)
    tl.store(meta + 7, tl.load(E))
    if MODE == "commit":
        tl.store(PM + pb * 8 + 5, 3)
        tl.store(PM + pb * 8 + 6, m)


@triton.jit
def advance_epoch(E, STATUS, POISON):
    if (tl.load(STATUS) == 0) & (tl.load(POISON) == 0):
        tl.store(E, tl.load(E) + 1)


@triton.jit
def reset_metadata(
    M, E, SM, STATUS, POISON, REV, R: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr
):
    if (tl.load(STATUS) != 0) | (tl.load(POISON) != 0):
        return
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    field = i % 8
    old_gen = tl.load(M + i, (i < R * 8) & (field == 5), 0)
    value = tl.where(
        field == 5, old_gen + 1, tl.where(field == 6, REV, tl.where(field == 7, tl.load(E), 0))
    )
    tl.store(M + i, value, i < R * 8)
    tl.store(SM + i, 0, i < S * 4)


@triton.jit
def reset_pending(PM, STATUS, POISON, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    if (tl.load(STATUS) == 0) & (tl.load(POISON) == 0):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(PM + i, 0, i < COUNT)
