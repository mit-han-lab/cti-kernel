# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own frozen MoE-block dX tree p2c (2026-10-11); expert-parallel masks
# and the shared expert's own intermediate size added here.
"""Triton kernels for ``ctikernels.moe.training``: the forward and the dX-only backward.

The forward rounds where Unsloth's Qwen3.5-MoE training path does (BF16 gate/up, SiLU,
product and down output; BF16 routing weights; its weighted top-k sum order); the router and
shared gate run as torch ops in ``_training_backend``. The CuTe provider shares the combine.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ...triton.rounding import INTERPRET
from ...triton.silu import silu

# Slot p = t * KP1 + j of token t is routed for j < k, with shared (id E) at j = k if present; rows
# are expert-sorted and block-padded (``_layout``), the block -> expert scheme of vLLM's and SGLang's
# fused MoE, written fresh here. No float atomics; every float reduction has a fixed order.

# Triton's CPU interpreter incorrectly multiplies BF16 tiles and truncates FP32 -> BF16 casts; under it the
# MMAs run on exactly widened FP32 operands and casts round in software, so CPU tests check the math.


@triton.jit
def _mma(a, b, acc):
    if INTERPRET:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
    return tl.dot(a, b, acc)


@triton.jit
def _bf16(v):
    # Round to nearest even. The interpreter truncates on casts, so it rounds the bits itself.
    if INTERPRET:
        b = v.to(tl.int32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
        return b.to(tl.float32, bitcast=True).to(tl.bfloat16)
    return v.to(tl.bfloat16)


@triton.jit
def _gate_up_tile(
    x_ptr,
    tok,
    rmask,
    w1_base,
    n0,
    H: tl.constexpr,
    NI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One I-tile of g = W1_gate x and u = W1_up x for BM gathered rows, FP32 accumulation over H in
    # fixed BK steps. Forward and backward recompute both call this routine with the same block
    # sizes, so the recomputed values equal the forward's bitwise. W1 rows: gate [0, NI), up [NI, 2NI).
    ks = tl.arange(0, BK)
    ns = n0 + tl.arange(0, BN)
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        xt = tl.load(x_ptr + tok[:, None] * H + (k0 + ks)[None, :], mask=rmask[:, None], other=0.0)
        wg = tl.load(w1_base + ns[None, :] * H + (k0 + ks)[:, None])
        wu = tl.load(w1_base + (NI + ns)[None, :] * H + (k0 + ks)[:, None])
        acc_g = _mma(xt, wg, acc_g)
        acc_u = _mma(xt, wu, acc_u)
    return acc_g, acc_u


@triton.jit
def _silu(g_b):
    # BF16 SiLU of the BF16 gate exactly as torch's F.silu (FP32, rounded once).
    g = g_b.to(tl.float32)
    if INTERPRET:  # no libdevice or inline PTX in the interpreter; numpy's exp and divide
        return _bf16(g / (1.0 + tl.exp(-g)))
    return _bf16(silu(g))


@triton.jit
def _row_sum(v, BM: tl.constexpr, BN: tl.constexpr):
    # Per-row FP32 sum over BN (a power of two, 16..256) columns as a fixed pairwise tree: each
    # level adds adjacent column pairs. tl.sum's order follows the register layout, which differs
    # between kernel variants (saved g/u come from loads, recomputed from the MMA); this does not.
    if BN >= 256:
        a, b = tl.split(tl.reshape(v, (BM, 128, 2)))
        v = a + b
    if BN >= 128:
        a, b = tl.split(tl.reshape(v, (BM, 64, 2)))
        v = a + b
    if BN >= 64:
        a, b = tl.split(tl.reshape(v, (BM, 32, 2)))
        v = a + b
    if BN >= 32:
        a, b = tl.split(tl.reshape(v, (BM, 16, 2)))
        v = a + b
    a, b = tl.split(tl.reshape(v, (BM, 8, 2)))
    v = a + b
    a, b = tl.split(tl.reshape(v, (BM, 4, 2)))
    v = a + b
    a, b = tl.split(tl.reshape(v, (BM, 2, 2)))
    v = a + b
    a, b = tl.split(tl.reshape(v, (BM, 1, 2)))
    return tl.reshape(a + b, (BM,))


@triton.jit
def _silu_mul(g_b, u_b):
    # a = BF16 SiLU times u in FP32 (the caller rounds it once more).
    return _silu(g_b).to(tl.float32) * u_b.to(tl.float32)


# The shared expert may have its own intermediate size NIS (an I-shard under expert parallelism).
# Each tile routine takes its block's size as a constexpr, so every K loop keeps a static trip
# count; with NIS == NI one instance serves both, as in the single-size kernels.


@triton.jit
def _gate_up_any(
    x_ptr,
    tok,
    rmask,
    w1_base,
    n0,
    is_shared,
    H: tl.constexpr,
    NI: tl.constexpr,
    NIS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    if NIS == NI:
        acc_g, acc_u = _gate_up_tile(x_ptr, tok, rmask, w1_base, n0, H, NI, BM, BN, BK)
    elif is_shared:
        acc_g, acc_u = _gate_up_tile(x_ptr, tok, rmask, w1_base, n0, H, NIS, BM, BN, BK)
    else:
        acc_g, acc_u = _gate_up_tile(x_ptr, tok, rmask, w1_base, n0, H, NI, BM, BN, BK)
    return acc_g, acc_u


@triton.jit
def _down_tile(
    a_ptr,
    rows,
    rmask,
    w2_base,
    hs,
    N: tl.constexpr,
    NMAX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # W2 a over K = N for BM sorted rows; a has row stride NMAX, W2 is [H, N].
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, N, BK):
        at = tl.load(
            a_ptr + rows[:, None].to(tl.int64) * NMAX + (k0 + ks)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        wt = tl.load(w2_base + hs[None, :] * N + (k0 + ks)[:, None])
        acc = _mma(at, wt, acc)
    return acc


@triton.jit
def _q_tile(
    dy_ptr,
    tok,
    rmask,
    w2_base,
    ns,
    H: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # q = W2^T dy for one I-tile, FP32 accumulation over H; W2 is [H, N].
    ks = tl.arange(0, BK)
    acc_q = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        dyt = tl.load(
            dy_ptr + tok[:, None].to(tl.int64) * H + (k0 + ks)[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        w2t = tl.load(w2_base + (k0 + ks)[:, None] * N + ns[None, :])
        acc_q = _mma(dyt, w2t, acc_q)
    return acc_q


@triton.jit
def _dx_pair_tile(
    gg_ptr,
    rows,
    rmask,
    w1_base,
    hs,
    H: tl.constexpr,
    N: tl.constexpr,
    NMAX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # G W1 over K = 2N in BK steps, gate half then up half (the single-size kernel's order):
    # G's gate columns are [0, N) and its up columns [NMAX, NMAX + N); W1 rows gate [0, N),
    # up [N, 2N).
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    goff = rows[:, None].to(tl.int64) * (2 * NMAX)
    for k0 in range(0, N, BK):
        gt = tl.load(gg_ptr + goff + (k0 + ks)[None, :], mask=rmask[:, None], other=0.0)
        wt = tl.load(w1_base + (k0 + ks)[:, None] * H + hs[None, :])
        acc = _mma(gt, wt, acc)
    for k0 in range(0, N, BK):
        gt = tl.load(gg_ptr + goff + (NMAX + k0 + ks)[None, :], mask=rmask[:, None], other=0.0)
        wt = tl.load(w1_base + (N + k0 + ks)[:, None] * H + hs[None, :])
        acc = _mma(gt, wt, acc)
    return acc


@triton.jit
def _fwd_gate_up_kernel(
    x_ptr,
    w1_ptr,
    w1s_ptr,
    sorted_slot_ptr,
    block_expert_ptr,
    a_ptr,
    gsave_ptr,
    usave_ptr,
    H: tl.constexpr,
    NI: tl.constexpr,
    NIS: tl.constexpr,
    NMAX: tl.constexpr,
    E: tl.constexpr,
    KP1: tl.constexpr,
    SAVE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_b)
    if e > E:  # past the padded slot count
        return
    n = tl.where(e == E, NIS, NI)
    rows = pid_b * BM + tl.arange(0, BM)
    slot = tl.load(sorted_slot_ptr + rows)
    rmask = slot >= 0
    tok = tl.where(rmask, slot // KP1, 0)
    if e == E:
        w1_base = w1s_ptr
    else:
        w1_base = w1_ptr + e.to(tl.int64) * (2 * NI * H)
    n0 = pid_n * BN
    if n0 >= n:
        return
    acc_g, acc_u = _gate_up_any(x_ptr, tok, rmask, w1_base, n0, e == E, H, NI, NIS, BM, BN, BK)
    g_b = _bf16(acc_g)
    u_b = _bf16(acc_u)
    a_b = _bf16(_silu_mul(g_b, u_b))
    ns = n0 + tl.arange(0, BN)
    off = rows[:, None].to(tl.int64) * NMAX + ns[None, :]
    tl.store(a_ptr + off, a_b, mask=rmask[:, None])
    if SAVE:
        tl.store(gsave_ptr + off, g_b, mask=rmask[:, None])
        tl.store(usave_ptr + off, u_b, mask=rmask[:, None])


@triton.jit
def _fwd_down_kernel(
    a_ptr,
    w2_ptr,
    w2s_ptr,
    sorted_slot_ptr,
    block_expert_ptr,
    y_ptr,
    H: tl.constexpr,
    NI: tl.constexpr,
    NIS: tl.constexpr,
    NMAX: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # y = BF16(W2_e a_b) per sorted slot, unweighted (the combine applies the weights); W2 is [E, H, NI].
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_b)
    if e > E:
        return
    rows = pid_b * BM + tl.arange(0, BM)
    slot = tl.load(sorted_slot_ptr + rows)
    rmask = slot >= 0
    if e == E:
        w2_base = w2s_ptr
    else:
        w2_base = w2_ptr + e.to(tl.int64) * (H * NI)
    hs = pid_n * BN + tl.arange(0, BN)
    if NIS == NI:
        acc = _down_tile(a_ptr, rows, rmask, w2_base, hs, NI, NMAX, BM, BN, BK)
    elif e == E:
        acc = _down_tile(a_ptr, rows, rmask, w2_base, hs, NIS, NMAX, BM, BN, BK)
    else:
        acc = _down_tile(a_ptr, rows, rmask, w2_base, hs, NI, NMAX, BM, BN, BK)
    y = _bf16(acc)
    tl.store(y_ptr + rows[:, None].to(tl.int64) * H + hs[None, :], y, mask=rmask[:, None])


@triton.jit
def _combine_kernel(
    y_ptr,
    inv_ptr,
    w_ptr,
    out_ptr,
    valid_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BN: tl.constexpr,
):
    # Unsloth's weighted top-k sum: each routed product BF16(w_j y_j) (BF16 weights), routed slot j
    # into FP32 partial j % 4, ((p0 + p1) + p2) + p3 rounded once; then the shared expert's
    # BF16(gate y_s) added in BF16. Masked rows write +0.0.
    KP1: tl.constexpr = K + HAS_SHARED
    t = tl.program_id(0)
    hs = tl.program_id(1) * BN + tl.arange(0, BN)
    nvalid = T
    if HAS_VALID:
        nvalid = tl.minimum(tl.load(valid_ptr), T)
    p0 = tl.zeros((BN,), dtype=tl.float32)
    p1 = tl.zeros((BN,), dtype=tl.float32)
    p2 = tl.zeros((BN,), dtype=tl.float32)
    p3 = tl.zeros((BN,), dtype=tl.float32)
    live = t < nvalid
    for j in tl.static_range(K):
        pos = tl.load(inv_ptr + t * KP1 + j).to(tl.int64)
        w = tl.load(w_ptr + t * KP1 + j)
        y = tl.load(y_ptr + pos * H + hs, mask=live & (pos >= 0), other=0.0).to(tl.float32)
        p = _bf16(w * y).to(tl.float32)
        if j % 4 == 0:
            p0 += p
        elif j % 4 == 1:
            p1 += p
        elif j % 4 == 2:
            p2 += p
        else:
            p3 += p
    out = _bf16(((p0 + p1) + p2) + p3)
    if HAS_SHARED:
        pos = tl.load(inv_ptr + t * KP1 + K).to(tl.int64)
        gate = tl.load(w_ptr + t * KP1 + K)
        y = tl.load(y_ptr + pos * H + hs, mask=live & (pos >= 0), other=0.0).to(tl.float32)
        out = _bf16(out.to(tl.float32) + _bf16(gate * y).to(tl.float32))
    tl.store(out_ptr + t.to(tl.int64) * H + hs, tl.where(live, out, 0.0).to(tl.bfloat16))


@triton.jit
def _bwd_slot_kernel(
    x_ptr,
    dy_ptr,
    w1_ptr,
    w1s_ptr,
    w2_ptr,
    w2s_ptr,
    sorted_slot_ptr,
    block_expert_ptr,
    wslot_ptr,
    gsave_ptr,
    usave_ptr,
    gg_ptr,
    dwpart_ptr,
    H: tl.constexpr,
    NI: tl.constexpr,
    NIS: tl.constexpr,
    NMAX: tl.constexpr,
    E: tl.constexpr,
    KP1: tl.constexpr,
    NT: tl.constexpr,
    SAVED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # B1. Per sorted block and I-tile: g_b, u_b (recomputed with the forward's routine, or loaded),
    # q = W2_e^T dy, the per-tile partial of dw = <q, a_b> (summed later in tile order), and
    # G = BF16([dg; du]) with da = w_slot * q (BF16 weights), du = da s_b at the BF16 SiLU s_b and
    # dg = da u silu'(g), as autograd through the forward's rounding points gives.
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(block_expert_ptr + pid_b)
    if e > E:
        return
    n = tl.where(e == E, NIS, NI)
    rows = pid_b * BM + tl.arange(0, BM)
    slot = tl.load(sorted_slot_ptr + rows)
    rmask = slot >= 0
    tok = tl.where(rmask, slot // KP1, 0)
    if e == E:
        w1_base = w1s_ptr
        w2_base = w2s_ptr
    else:
        w1_base = w1_ptr + e.to(tl.int64) * (2 * NI * H)
        w2_base = w2_ptr + e.to(tl.int64) * (H * NI)
    n0 = pid_n * BN
    if n0 >= n:
        return
    ns = n0 + tl.arange(0, BN)
    off = rows[:, None].to(tl.int64) * NMAX + ns[None, :]
    if SAVED:
        g_b = tl.load(gsave_ptr + off, mask=rmask[:, None], other=0.0)
        u_b = tl.load(usave_ptr + off, mask=rmask[:, None], other=0.0)
    else:
        acc_g, acc_u = _gate_up_any(x_ptr, tok, rmask, w1_base, n0, e == E, H, NI, NIS, BM, BN, BK)
        g_b = _bf16(acc_g)
        u_b = _bf16(acc_u)
    if NIS == NI:
        acc_q = _q_tile(dy_ptr, tok, rmask, w2_base, ns, H, NI, BM, BN, BK)
    elif e == E:
        acc_q = _q_tile(dy_ptr, tok, rmask, w2_base, ns, H, NIS, BM, BN, BK)
    else:
        acc_q = _q_tile(dy_ptr, tok, rmask, w2_base, ns, H, NI, BM, BN, BK)
    s_b = _silu(g_b).to(tl.float32)
    a_b = _bf16(s_b * u_b.to(tl.float32))
    part = _row_sum(acc_q * a_b.to(tl.float32), BM, BN)
    tl.store(dwpart_ptr + rows.to(tl.int64) * NT + pid_n, part, mask=rmask)
    w = tl.load(wslot_ptr + tl.where(rmask, slot, 0), mask=rmask, other=0.0)
    da = acc_q * w[:, None]
    g = g_b.to(tl.float32)
    u = u_b.to(tl.float32)
    sg = tl.sigmoid(g)
    dg = da * u * sg * (1.0 + g * (1.0 - sg))
    du = da * s_b
    goff = rows[:, None].to(tl.int64) * (2 * NMAX) + ns[None, :]
    tl.store(gg_ptr + goff, _bf16(dg), mask=rmask[:, None])
    tl.store(gg_ptr + goff + NMAX, _bf16(du), mask=rmask[:, None])


@triton.jit
def _bwd_dx_pair_kernel(
    gg_ptr,
    w1_ptr,
    w1s_ptr,
    sorted_slot_ptr,
    block_expert_ptr,
    dxp_ptr,
    H: tl.constexpr,
    NI: tl.constexpr,
    NIS: tl.constexpr,
    NMAX: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # B2. dx_pair = G W1_e over K = 2 NI, rounded once to BF16, stored in sorted order.
    pid_b = tl.program_id(0)
    e = tl.load(block_expert_ptr + pid_b)
    if e > E:
        return
    rows = pid_b * BM + tl.arange(0, BM)
    slot = tl.load(sorted_slot_ptr + rows)
    rmask = slot >= 0
    if e == E:
        w1_base = w1s_ptr
    else:
        w1_base = w1_ptr + e.to(tl.int64) * (2 * NI * H)
    hs = tl.program_id(1) * BN + tl.arange(0, BN)
    if NIS == NI:
        acc = _dx_pair_tile(gg_ptr, rows, rmask, w1_base, hs, H, NI, NMAX, BM, BN, BK)
    elif e == E:
        acc = _dx_pair_tile(gg_ptr, rows, rmask, w1_base, hs, H, NIS, NMAX, BM, BN, BK)
    else:
        acc = _dx_pair_tile(gg_ptr, rows, rmask, w1_base, hs, H, NI, NMAX, BM, BN, BK)
    tl.store(
        dxp_ptr + rows[:, None].to(tl.int64) * H + hs[None, :], _bf16(acc), mask=rmask[:, None]
    )


@triton.jit
def _bwd_token_kernel(
    dxp_ptr,
    inv_ptr,
    dwpart_ptr,
    wslot_ptr,
    sel_ptr,
    r_ptr,
    sg_ptr,
    dx_ptr,
    valid_ptr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    KP1: tl.constexpr,
    NT: tl.constexpr,
    KPAD: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BN: tl.constexpr,
    HAS_SHARED: tl.constexpr = True,
):
    # B3. Per token: dw_j from the tile partials (fixed order); the routing-weight gradient through
    # the selected-set softmax, dz_j = w_j (dw_j - sum_i w_i dw_i); the shared-gate gradient
    # ds = g(1 - g) r; dx = sum of dx_pair rows in slot order + sum_j dz_j R[S_j] + ds w_sg,
    # one BF16 rounding. Masked rows write +0.0. ``wslot`` holds the FP32 router weights (the
    # straight-through derivative of their BF16 rounding) and the BF16 shared-gate sigmoid.
    t = tl.program_id(0)
    hs = tl.program_id(1) * BN + tl.arange(0, BN)
    nvalid = T
    if HAS_VALID:
        nvalid = tl.minimum(tl.load(valid_ptr), T)
    acc = tl.zeros((BN,), dtype=tl.float32)
    if t < nvalid:
        js = tl.arange(0, KPAD)
        jmask = js < K
        pos_r = tl.load(inv_ptr + t * KP1 + js, mask=jmask, other=0).to(tl.int64)
        dw = tl.zeros((KPAD,), dtype=tl.float32)
        for n in tl.static_range(NT):
            dw += tl.load(dwpart_ptr + pos_r * NT + n, mask=jmask & (pos_r >= 0), other=0.0)
        w = tl.load(wslot_ptr + t * KP1 + js, mask=jmask, other=0.0)
        # Remote dw is zero, but ALL selected router rows participate in dz.
        # Summing ranks gives sum_local w_j dw_j (R_j - sum_i w_i R_i).
        dz = w * (dw - tl.sum(w * dw, axis=0))
        if HAS_SHARED:
            pos_s = tl.load(inv_ptr + t * KP1 + K).to(tl.int64)
            r = 0.0
            for n in tl.static_range(NT):
                r += tl.load(dwpart_ptr + pos_s * NT + n)
            game = tl.load(wslot_ptr + t * KP1 + K)
            ds = game * (1.0 - game) * r
        for j in tl.static_range(KP1):
            pos = tl.load(inv_ptr + t * KP1 + j).to(tl.int64)
            acc += tl.load(dxp_ptr + pos * H + hs, mask=pos >= 0, other=0.0).to(tl.float32)
        sel = tl.load(sel_ptr + t * K + js, mask=jmask, other=0).to(tl.int64)
        rt = tl.load(r_ptr + sel[:, None] * H + hs[None, :], mask=jmask[:, None], other=0.0)
        acc += tl.sum(dz[:, None] * rt.to(tl.float32), axis=0)
        if HAS_SHARED:
            acc += ds * tl.load(sg_ptr + hs).to(tl.float32)
    tl.store(dx_ptr + t.to(tl.int64) * H + hs, _bf16(acc))
