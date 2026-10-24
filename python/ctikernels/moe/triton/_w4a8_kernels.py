# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""W4A8 routed-expert kernels: MXFP4 weights x MXFP8 activations via ``tl.dot_scaled``.

MXFP4 = E2M1 nibbles packed along K (low nibble first) with one UE8M0 scale per 32 K.
MXFP8 = E4M3 with one UE8M0 scale per 32 K. Arithmetic is FP32; no atomics.
Reference: Triton block-scaled matmul tutorial (``tl.dot_scaled`` operand layout).
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ...triton.silu import silu


@triton.jit
def quant_mxfp8(x):
    """BF16 [..., 32] -> (E4M3 bytes, UE8M0 scale) with RNE on the integer significand.

    The scale is the smallest power of two that maps the group's absmax to at most 448
    (saturating E4M3 max); an all-zero group gets scale 2^0. Exact for every BF16 input,
    subnormals included, with no float reciprocal or FP8 cast.
    """
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    mag = bits & 32767
    peak = tl.max(mag, -1)
    sf = tl.minimum(tl.maximum((peak >> 7) - 8 + ((peak & 127) > 96), 0), 254)
    sf = tl.where(peak == 0, 127, sf)
    ef = mag >> 7
    sig = (mag & 127) | tl.where(ef > 0, 128, 0)
    shift = libdevice.clz(sig) - 24
    sig = sig << shift
    exponent = tl.maximum(ef, 1) - tl.expand_dims(sf, -1) - shift
    # Normal E4M3 keeps 4 significant bits; the subnormal quantum is 2^-9.
    discard = tl.minimum(tl.maximum(4, -exponent - 2), 24)
    kept = sig >> discard
    remainder = sig & ((1 << discard) - 1)
    halfway = 1 << (discard - 1)
    rounded = kept + ((remainder > halfway) | ((remainder == halfway) & ((kept & 1) != 0)))
    code = tl.where(exponent >= -6, (exponent + 6) * 8 + rounded, rounded)
    code = tl.minimum(code, 126) | ((bits >> 8) & 128)
    code = tl.where(tl.expand_dims(peak, -1) == 0, 0, code)
    return code.to(tl.uint8), sf.to(tl.uint8)


@triton.jit
def act_quant(X, Q, S, GROUPS, BG: tl.constexpr):
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    k = tl.arange(0, 32)
    x = tl.load(X + g[:, None] * 32 + k[None, :], g[:, None] < GROUPS, 0)
    q, s = quant_mxfp8(x)
    tl.store(Q + g[:, None] * 32 + k[None, :], q, g[:, None] < GROUPS)
    tl.store(S + g, s, g < GROUPS)


@triton.jit
def _e2m1_to_e4m3(packed):
    """[N, K/2] E2M1 pairs (low nibble first) -> [N, K] E4M3; exact (E4M3 holds every E2M1).

    One ``prmt`` byte lookup per four nibbles: the selector is the nibbles' magnitude bits,
    the table the eight E4M3 magnitudes; a second ``prmt`` places the sign bits.
    """
    lo, hi = tl.inline_asm_elementwise(
        """
        {
        .reg .b32 sel, sgn, mag, w;
        mov.b32 w, $2;
        and.b32 sel, w, 0x7777;
        prmt.b32 mag, 0x3C383000, 0x4C484440, sel;
        shr.u32 sgn, w, 3;
        and.b32 sgn, sgn, 0x1111;
        prmt.b32 sgn, 0x8000, 0, sgn;
        or.b32 $0, mag, sgn;
        shr.u32 w, w, 16;
        and.b32 sel, w, 0x7777;
        prmt.b32 mag, 0x3C383000, 0x4C484440, sel;
        shr.u32 sgn, w, 3;
        and.b32 sgn, sgn, 0x1111;
        prmt.b32 sgn, 0x8000, 0, sgn;
        or.b32 $1, mag, sgn;
        }
        """,
        "=r,=r,r",
        [packed],
        dtype=(tl.uint8, tl.uint8),
        is_pure=True,
        pack=4,
    )
    # Each 4-byte word holds K 0..7: ``lo`` carries K 0..3, ``hi`` K 4..7.
    n: tl.constexpr = packed.shape[0]
    k2: tl.constexpr = packed.shape[1]
    x = tl.join(lo.reshape(n, k2 // 4, 4), hi.reshape(n, k2 // 4, 4))
    return tl.permute(x, 0, 1, 3, 2).reshape(n, 2 * k2).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _scaled_dot(
    a, sa, W, SW, base_row, n, k0, acc, K: tl.constexpr, BK: tl.constexpr, CVT: tl.constexpr
):
    """acc += MXFP8 a @ MXFP4 W[rows].T, weight scales SW in the 128x4 interleave. CVT feeds W
    as E4M3: on sm120 Triton lowers E4M3 x E4M3 to the native block-scaled MMA but upcasts an
    E2M1 operand to BF16."""
    kb = k0 // 2 + tl.arange(0, BK // 2)
    ks = k0 // 32 + tl.arange(0, BK // 32)
    rows = (base_row + n).to(tl.int64)
    b = tl.load(W + rows[:, None] * (K // 2) + kb[None, :])
    # Expert row counts are multiples of 128, so global 128-row blocks are per-expert blocks.
    block = (rows // 128 * (K // 128))[:, None] + (ks // 4)[None, :]
    inner = ((rows % 32) * 16 + (rows % 128) // 32 * 4)[:, None] + (ks % 4)[None, :]
    sb = tl.load(SW + block * 512 + inner)
    if CVT:
        return tl.dot_scaled(a, sa, "e4m3", _e2m1_to_e4m3(b).T, sb, "e4m3", acc)
    return tl.dot_scaled(a, sa, "e4m3", b.T, sb, "e2m1", acc)


@triton.jit
def fc1(
    QX,
    SX,
    SORTED,
    BLOCK_E,
    ROUTE,
    W,
    SW,
    QH,
    SH,
    PAIRS,
    LIMIT,
    H: tl.constexpr,
    INTER: tl.constexpr,
    TOP_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CLAMP: tl.constexpr,
    CVT: tl.constexpr,
):
    """Up/gate projection (w13 rows: up, then gate) of one sorted row block, then clamp, SiLU(gate) * up * route,
    BF16 rounding and MXFP8 quantization, stored at the sorted row."""
    block, tile = tl.program_id(0), tl.program_id(1)
    expert = tl.load(BLOCK_E + block)
    if expert >= 0:
        rows = block * BM + tl.arange(0, BM)
        p = tl.load(SORTED + rows)
        valid = p < PAIRS
        token = tl.where(valid, p // TOP_K, 0).to(tl.int64)
        n = tile * BN + tl.arange(0, BN)
        g = tl.zeros((BM, BN), tl.float32)
        u = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            ks = k0 // 32 + tl.arange(0, BK // 32)
            a = tl.load(QX + token[:, None] * H + k[None, :], valid[:, None], 0)
            a = a.to(tl.float8e4nv, bitcast=True)
            sa = tl.load(SX + token[:, None] * (H // 32) + ks[None, :], valid[:, None], 127)
            u = _scaled_dot(a, sa, W, SW, expert * 2 * INTER, n, k0, u, H, BK, CVT)
            g = _scaled_dot(a, sa, W, SW, expert * 2 * INTER + INTER, n, k0, g, H, BK, CVT)
        if CLAMP:
            g = tl.minimum(g, LIMIT)
            u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        route = tl.load(ROUTE + p, valid, 0.0)
        h = (silu(g) * u * route[:, None]).to(tl.bfloat16)
        q, s = quant_mxfp8(h.reshape(BM, BN // 32, 32))
        r64 = rows.to(tl.int64)
        tl.store(QH + r64[:, None] * INTER + n[None, :], q.reshape(BM, BN))
        ns = tile * (BN // 32) + tl.arange(0, BN // 32)
        tl.store(SH + r64[:, None] * (INTER // 32) + ns[None, :], s)


@triton.jit
def fc2(
    QH,
    SH,
    SORTED,
    BLOCK_E,
    W,
    SW,
    PART,
    PAIRS,
    H: tl.constexpr,
    INTER: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CVT: tl.constexpr,
):
    """Down projection of one sorted row block; FP32 rows land at their route index."""
    block, tile = tl.program_id(0), tl.program_id(1)
    expert = tl.load(BLOCK_E + block)
    if expert >= 0:
        rows = (block * BM + tl.arange(0, BM)).to(tl.int64)
        p = tl.load(SORTED + rows)
        n = tile * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, INTER, BK):
            k = k0 + tl.arange(0, BK)
            ks = k0 // 32 + tl.arange(0, BK // 32)
            a = tl.load(QH + rows[:, None] * INTER + k[None, :]).to(tl.float8e4nv, bitcast=True)
            sa = tl.load(SH + rows[:, None] * (INTER // 32) + ks[None, :])
            acc = _scaled_dot(a, sa, W, SW, expert * H, n, k0, acc, INTER, BK, CVT)
        p64 = p.to(tl.int64)
        tl.store(PART + p64[:, None] * H + n[None, :], acc, (p < PAIRS)[:, None])


@triton.jit
def combine(
    PART,
    IDS,
    MAP,
    OUT,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    GLOBAL: tl.constexpr,
    BH: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    """Sum the local routes of one token in slot order and round once to BF16."""
    t, tile = tl.program_id(0), tl.program_id(1)
    n = tile * BH + tl.arange(0, BH)
    acc = tl.zeros((BH,), tl.float32)
    for slot in tl.static_range(TOP_K):
        e = tl.load(IDS + t * TOP_K + slot)
        local = (e >= 0) & (e < GLOBAL)
        if HAS_MAP:
            local = local & (tl.load(MAP + e, local, -1) >= 0)
        pair = (t * TOP_K + slot).to(tl.int64)
        acc += tl.load(PART + pair * H + n, local & (n < H), 0.0)
    tl.store(OUT + t.to(tl.int64) * H + n, acc.to(tl.bfloat16), n < H)


@triton.jit
def local_routes(
    IDS,
    WEIGHTS,
    MAP,
    OUT_IDS,
    OUT_WEIGHTS,
    ROUTES,
    REMOTE,
    GLOBAL: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Global route ids -> local ids; remote and padding routes -> ``REMOTE`` at weight 0."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < ROUTES
    e = tl.load(IDS + i, live, -1)
    ok = (e >= 0) & (e < GLOBAL)
    if HAS_MAP:
        local = tl.load(MAP + e, live & ok, -1)
    else:
        local = tl.where(ok, e, -1)
    keep = local >= 0
    w = tl.load(WEIGHTS + i, live, 0.0)
    tl.store(OUT_IDS + i, tl.where(keep, local, REMOTE), live)
    tl.store(OUT_WEIGHTS + i, tl.where(keep, w, 0.0), live)
