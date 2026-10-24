# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton per-expert MoE LoRA kernels (forward).

Written fresh after the published designs of vLLM's fused_moe_lora (shrink and expand in one
program, the rank vector kept in registers) and SGLang's moe_lora align/shrink/expand. Layout:
A slice s of a projection lives at ``a_ptr + s * a_gap`` as [E,R,K], B slice s at
``b_ptr + s * b_gap`` as [E,N,R]; pair p = token * top_k + choice reads input row
p // ROW_DIV and owns cache row p ([S*N] columns, slice-major). Ids outside [0, E) add nothing.
No atomics and no split-K: every sum has a fixed order, so graph replays are bitwise eager.
ONE_A (shared-A gate_up, PEFT's ``gate_up_proj``): both slices read the same A, so the shrink
runs once and z is stored as one slice; the expand still writes both B slices. The values equal
the two-slice path with A_gate == A_up, bit for bit.
"""

import triton
import triton.language as tl

from ...triton.rounding import bf16_round, round_to
from ...triton.silu import sigmoid, silu


@triton.jit
def _finish(old, delta):
    # Exact-zero updates keep the cache bits (as the torch reference does).
    return tl.where(delta == 0.0, old, old + delta)


@triton.jit
def _silu_mul(g, u):
    return silu(g) * u  # torch's F.silu, bit for bit


@triton.jit
def _gemv(
    row_ptr, f_ptr, offs_r, rank_mask, K, R: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr
):
    """z[r] = sum_k f[r, k] * row[k] in FP32; f is one expert slice [R, K]."""
    z = tl.zeros((BR,), dtype=tl.float32)
    offs_k = tl.arange(0, BK)
    for k0 in range(0, K, BK):
        k = k0 + offs_k
        k_mask = k < K
        v = tl.load(row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        f = tl.load(
            f_ptr + offs_r[:, None] * K + k[None, :],
            mask=rank_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        z += tl.sum(f * v[None, :], axis=1)
    return z


@triton.jit
def _expand_row(b_ptr, z, offs_n, n_mask, offs_r, rank_mask, R: tl.constexpr):
    """d[n] = sum_r b[n, r] * z[r] in FP32; b is one expert slice [N, R]."""
    b = tl.load(
        b_ptr + offs_n[:, None] * R + offs_r[None, :],
        mask=n_mask[:, None] & rank_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    return tl.sum(b * z[None, :], axis=1)


@triton.jit(
    do_not_specialize=["P", "a_gap", "b_gap", "work_total", "n_chunks", "tiles_per_chunk", "KS"]
)
def pair_kernel(
    x_ptr,
    a_ptr,
    b_ptr,
    out_ptr,
    act_ptr,
    z_ptr,
    ids_ptr,
    w_ptr,
    scale_ptr,
    part_ptr,
    P,
    E,
    K,
    N,
    a_gap,
    b_gap,
    x_stride,
    out_stride,
    act_stride,
    work_total,
    n_chunks,
    tiles_per_chunk,
    KS,
    ROW_DIV: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ACT: tl.constexpr,
    SAVE_Z: tl.constexpr,
    SPLIT: tl.constexpr,
    ONE_A: tl.constexpr = False,
):
    """Persistent per-pair path (no alignment): one work item = (pair, N chunk).

    Shrink is an FP32 GEMV per slice kept in registers and reused over the chunk's N tiles, or
    with SPLIT the fixed-order sum of ``pair_shrink_kernel``'s KS partials [P, S, KS, R].
    ACT writes silu(gate + d0) * (up + d1) to ``act`` instead of updating the cache.
    """
    offs_r = tl.arange(0, BR)
    rank_mask = offs_r < R
    scale = tl.load(scale_ptr)
    for work in range(tl.program_id(0), work_total, tl.num_programs(0)):
        pair = work // n_chunks
        chunk = work % n_chunks
        e = tl.load(ids_ptr + pair).to(tl.int64)
        if (e >= 0) & (e < E):
            row_ptr = x_ptr + (pair // ROW_DIV).to(tl.int64) * x_stride
            a0 = a_ptr + e * R * K
            b0 = b_ptr + e * N * R
            ZS: tl.constexpr = 1 if ONE_A else S
            if SPLIT:
                pp = part_ptr + pair.to(tl.int64) * ZS * KS * R + offs_r
                z0 = tl.zeros((BR,), dtype=tl.float32)
                z1 = tl.zeros((BR,), dtype=tl.float32)
                for ks in range(KS):
                    z0 += tl.load(pp + ks * R, mask=rank_mask, other=0.0)
                    if ZS == 2:
                        z1 += tl.load(pp + (KS + ks) * R, mask=rank_mask, other=0.0)
                if ONE_A:
                    z1 = z0
            else:
                z0 = _gemv(row_ptr, a0, offs_r, rank_mask, K, R, BR, BK)
                z1 = z0
                if ZS == 2:
                    z1 = _gemv(row_ptr, a0 + a_gap, offs_r, rank_mask, K, R, BR, BK)
            if SAVE_Z:
                if chunk == 0:
                    zp = z_ptr + pair.to(tl.int64) * ZS * R
                    tl.store(zp + offs_r, z0.to(z_ptr.dtype.element_ty), mask=rank_mask)
                    if ZS == 2:
                        tl.store(zp + R + offs_r, z1.to(z_ptr.dtype.element_ty), mask=rank_mask)
            mult = scale
            if WEIGHTED:
                mult = mult * tl.load(w_ptr + pair)
            out_row = out_ptr + pair.to(tl.int64) * out_stride
            for t in range(tiles_per_chunk):
                n0 = (chunk * tiles_per_chunk + t) * BN
                if n0 < N:
                    offs_n = n0 + tl.arange(0, BN)
                    n_mask = offs_n < N
                    d0 = _expand_row(b0, z0, offs_n, n_mask, offs_r, rank_mask, R) * mult
                    old0 = tl.load(out_row + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                    if ACT:
                        d1 = (
                            _expand_row(b0 + b_gap, z1, offs_n, n_mask, offs_r, rank_mask, R) * mult
                        )
                        old1 = tl.load(out_row + N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                        y = _silu_mul(_finish(old0, d0), _finish(old1, d1))
                        act_row = act_ptr + pair.to(tl.int64) * act_stride
                        tl.store(act_row + offs_n, y.to(act_ptr.dtype.element_ty), mask=n_mask)
                    else:
                        tl.store(
                            out_row + offs_n,
                            _finish(old0, d0).to(out_ptr.dtype.element_ty),
                            mask=n_mask,
                        )
                        if S == 2:
                            d1 = (
                                _expand_row(b0 + b_gap, z1, offs_n, n_mask, offs_r, rank_mask, R)
                                * mult
                            )
                            old1 = tl.load(out_row + N + offs_n, mask=n_mask, other=0.0).to(
                                tl.float32
                            )
                            tl.store(
                                out_row + N + offs_n,
                                _finish(old1, d1).to(out_ptr.dtype.element_ty),
                                mask=n_mask,
                            )


@triton.jit
def _shrink_tile(
    x_ptrs,
    a_ptrs,
    row_mask,
    offs_k,
    rank_mask,
    K,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    acc = tl.zeros((BM, BR), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_mask = (k0 + offs_k) < K
        xv = tl.load(x_ptrs + k0, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        av = tl.load(a_ptrs + k0, mask=k_mask[:, None] & rank_mask[None, :], other=0.0)
        acc = tl.dot(xv, av, acc)
    return acc


@triton.jit(do_not_specialize=["sorted_len", "P", "a_gap"])
def shrink_kernel(
    x_ptr,
    a_ptr,
    z_ptr,
    sorted_ptr,
    experts_ptr,
    npad_ptr,
    sorted_len,
    P,
    E,
    K,
    a_gap,
    x_stride,
    k_split,
    ROW_DIV: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    KS: tl.constexpr,
):
    """General path, stage 1: z[ks, p, s, r-tile] = x[p] . A[e, s, r-tile]^T over the ks-th
    K split (``k_split`` columns, whole BK tiles), in FP32 (any rank); KS = 1 is one pass."""
    block = tl.program_id(0)
    r0 = tl.program_id(1) * BR
    s = tl.program_id(2) // KS
    ks = tl.program_id(2) % KS
    if block * BM >= tl.load(npad_ptr):
        return
    e = tl.load(experts_ptr + block).to(tl.int64)
    if (e < 0) | (e >= E):
        return
    slot = block * BM + tl.arange(0, BM)
    pairs = tl.load(sorted_ptr + slot, mask=slot < sorted_len, other=P)
    row_mask = pairs < P
    pairs = tl.where(row_mask, pairs, 0).to(tl.int64)
    offs_r = r0 + tl.arange(0, BR)
    rank_mask = offs_r < R
    safe_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BK)
    k0 = ks * k_split
    x_ptrs = x_ptr + (pairs // ROW_DIV)[:, None] * x_stride + k0 + offs_k[None, :]
    a_ptrs = a_ptr + s * a_gap + e * R * K + safe_r[None, :] * K + k0 + offs_k[:, None]
    k_len = tl.minimum(k_split, K - k0)
    z = _shrink_tile(x_ptrs, a_ptrs, row_mask, offs_k, rank_mask, k_len, BM, BR, BK)
    zp = z_ptr + ks.to(tl.int64) * P * S * R + pairs[:, None] * (S * R) + s * R + offs_r[None, :]
    tl.store(zp, z.to(z_ptr.dtype.element_ty), mask=row_mask[:, None] & rank_mask[None, :])


@triton.jit
def _scaled(d, mult, ROUND: tl.constexpr):
    if ROUND:
        return bf16_round(bf16_round(d) * mult[:, None])  # pinned: Triton folds .to round trips
    return d * mult[:, None]


@triton.jit(do_not_specialize=["sorted_len", "P", "b_gap"])
def expand_kernel(
    z_ptr,
    b_ptr,
    out_ptr,
    act_ptr,
    w_ptr,
    scale_ptr,
    sorted_ptr,
    experts_ptr,
    npad_ptr,
    sorted_len,
    P,
    E,
    N,
    b_gap,
    out_stride,
    act_stride,
    S: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    WEIGHTED: tl.constexpr,
    ACT: tl.constexpr,
    KS: tl.constexpr,
    ROUND_DELTA: tl.constexpr = False,
    ONE_A: tl.constexpr = False,
    PRESCALE: tl.constexpr = False,
):
    """General path, stage 2: cache += scale * (w) * z . B^T over rank tiles, z the fixed-order
    sum of the KS shrink partials [KS, P, S, R] (rounded per tile for the MMA). ROUND_DELTA
    rounds as PEFT's BF16 LoRA (training), cache = BF16(cache + BF16(BF16(z . B^T) * scale)),
    and with ACT the gate and up sums too before SiLU * up. PRESCALE scales B instead, B_s =
    BF16(B * scale), as vLLM's punica expand on its scaling-folded B (the shared expert)."""
    block = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    if block * BM >= tl.load(npad_ptr):
        return
    e = tl.load(experts_ptr + block).to(tl.int64)
    if (e < 0) | (e >= E):
        return
    slot = block * BM + tl.arange(0, BM)
    pairs = tl.load(sorted_ptr + slot, mask=slot < sorted_len, other=P)
    row_mask = pairs < P
    pairs = tl.where(row_mask, pairs, 0).to(tl.int64)
    offs_n = n0 + tl.arange(0, BN)
    n_mask = offs_n < N
    m = row_mask[:, None] & n_mask[None, :]
    dtype = out_ptr.dtype.element_ty
    s = tl.load(scale_ptr)
    if PRESCALE:
        mult = tl.full((BM,), 1.0, dtype=tl.float32)
    else:
        mult = s + tl.zeros((BM,), dtype=tl.float32)
    if WEIGHTED:
        mult = mult * tl.load(w_ptr + pairs, mask=row_mask, other=0.0)
    out_rows = out_ptr + pairs[:, None] * out_stride
    d0 = tl.zeros((BM, BN), dtype=tl.float32)
    d1 = tl.zeros((BM, BN), dtype=tl.float32)
    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        rank_mask = offs_r < R
        zm = row_mask[:, None] & rank_mask[None, :]
        bm = rank_mask[:, None] & n_mask[None, :]
        ZS: tl.constexpr = 1 if ONE_A else S
        zp = z_ptr + pairs[:, None] * (ZS * R) + offs_r[None, :]
        bp = b_ptr + e * N * R + offs_n[None, :] * R + offs_r[:, None]
        mma = b_ptr.dtype.element_ty
        z0 = tl.zeros((BM, BR), dtype=tl.float32)
        z1 = tl.zeros((BM, BR), dtype=tl.float32)
        for ks in range(KS):
            zk = zp + ks * P * ZS * R
            z0 += tl.load(zk, mask=zm, other=0.0).to(tl.float32)
            if ZS == 2:
                z1 += tl.load(zk + R, mask=zm, other=0.0).to(tl.float32)
        if ONE_A:
            z1 = z0
        b0 = tl.load(bp, mask=bm, other=0.0)
        if PRESCALE:
            b0 = bf16_round(b0.to(tl.float32) * s).to(mma)
        d0 = tl.dot(z0.to(mma), b0, d0)
        if S == 2:
            b1 = tl.load(bp + b_gap, mask=bm, other=0.0)
            if PRESCALE:
                b1 = bf16_round(b1.to(tl.float32) * s).to(mma)
            d1 = tl.dot(z1.to(mma), b1, d1)
    d0 = _scaled(d0, mult, ROUND_DELTA)
    old0 = tl.load(out_rows + offs_n[None, :], mask=m, other=0.0).to(tl.float32)
    if S == 2:
        d1 = _scaled(d1, mult, ROUND_DELTA)
        old1 = tl.load(out_rows + N + offs_n[None, :], mask=m, other=0.0).to(tl.float32)
        if ACT:
            g, u = _finish(old0, d0), _finish(old1, d1)
            if ROUND_DELTA:
                g, u = bf16_round(g), bf16_round(u)
            y = _silu_mul(g, u)
            act_rows = act_ptr + pairs[:, None] * act_stride
            tl.store(act_rows + offs_n[None, :], y.to(act_ptr.dtype.element_ty), mask=m)
        else:
            tl.store(out_rows + offs_n[None, :], _finish(old0, d0).to(dtype), mask=m)
            tl.store(out_rows + N + offs_n[None, :], _finish(old1, d1).to(dtype), mask=m)
    else:
        tl.store(out_rows + offs_n[None, :], _finish(old0, d0).to(dtype), mask=m)


@triton.jit(do_not_specialize=["P", "a_gap", "KS", "k_split"])
def pair_shrink_kernel(
    x_ptr,
    a_ptr,
    part_ptr,
    ids_ptr,
    P,
    E,
    K,
    a_gap,
    x_stride,
    KS,
    k_split,
    ROW_DIV: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Split-K decode shrink: part[p, s, ks] = A[e, s, :, K-split ks] . x[p // ROW_DIV] in FP32.

    Spreads each pair's factor read over KS programs (one DRAM pass, no recompute per N chunk);
    the pair kernel sums the KS partials in a fixed order (no atomics). PDL: a dependent
    launch may start right away (it waits for this grid before reading the partials).
    """
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    pair = tl.program_id(0) // S
    s = tl.program_id(0) % S
    ks = tl.program_id(1)
    e = tl.load(ids_ptr + pair).to(tl.int64)
    if (e >= 0) & (e < E):
        offs_r = tl.arange(0, BR)
        rank_mask = offs_r < R
        k0 = ks * k_split
        row_ptr = x_ptr + (pair // ROW_DIV).to(tl.int64) * x_stride + k0
        f_ptr = a_ptr + s * a_gap + e * R * K + k0
        offs_k = tl.arange(0, BK)
        z = tl.zeros((BR,), dtype=tl.float32)
        for kk in range(0, k_split, BK):
            k = kk + offs_k
            k_mask = (k < k_split) & (k0 + k < K)
            v = tl.load(row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
            f = tl.load(
                f_ptr + offs_r[:, None] * K + k[None, :],
                mask=rank_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            z += tl.sum(f * v[None, :], axis=1)
        dst = part_ptr + ((pair.to(tl.int64) * S + s) * KS + ks) * R + offs_r
        tl.store(dst, z, mask=rank_mask)


@triton.jit(do_not_specialize=["P"])
def route_prep_kernel(ids_ptr, out_ptr, P, E, SENTINEL: tl.constexpr, BLOCK: tl.constexpr):
    """int32 local ids with every id outside [0, E) replaced by ``SENTINEL`` (one launch)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < P
    e = tl.load(ids_ptr + offs, mask=m, other=-1)
    e = tl.where((e >= 0) & (e < E), e, SENTINEL)
    tl.store(out_ptr + offs, e.to(tl.int32), mask=m)


@triton.jit(do_not_specialize=["T", "KS"])
def reduce_expand_kernel(
    c3_ptr, out_ptr, part_ptr, b_ptr, ids_ptr, w_ptr, scale_ptr, T, E, N, KS,
    shared_ptr, gate_ptr, shared_stride, gate_stride,
    TOP_K: tl.constexpr, R: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr,
    SHARED: tl.constexpr = False,
):  # fmt: skip
    """The top-k combine with the down LoRA expand fused in: each valid route's unweighted
    BF16 row gains scale * B2[e] . sum_ks part[p, ks] (one rounding), then vLLM's finalize,
    acc = fma(w, row, acc) in route order from zero, rounded once. SHARED joins the gated
    shared expert as moe_sum_reduce_triton does: out = r(r(r(sigmoid(gate)) * shared) +
    r(acc)), each step rounded to the output dtype."""
    tok = tl.program_id(0).to(tl.int64)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    n_mask = offs_n < N
    offs_r = tl.arange(0, BR)
    rank_mask = offs_r < R
    acc = tl.zeros((BN,), dtype=tl.float32)
    scale = tl.load(scale_ptr)
    for j in tl.static_range(TOP_K):
        pair = tok * TOP_K + j
        e = tl.load(ids_ptr + pair).to(tl.int64)
        if (e >= 0) & (e < E):
            y = tl.load(c3_ptr + pair * N + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            z = tl.zeros((BR,), dtype=tl.float32)
            for ks in range(KS):
                z += tl.load(part_ptr + (pair * KS + ks) * R + offs_r, mask=rank_mask, other=0.0)
            d = _expand_row(b_ptr + e * N * R, z, offs_n, n_mask, offs_r, rank_mask, R)
            y = bf16_round(y + scale * d)
            acc = tl.fma(tl.load(w_ptr + pair), y, acc)
    values = acc.to(out_ptr.dtype.element_ty)
    if SHARED:
        out_ty: tl.constexpr = out_ptr.dtype.element_ty
        gate = tl.load(gate_ptr + tok * gate_stride).to(tl.float32)
        shared = tl.load(shared_ptr + tok * shared_stride + offs_n, mask=n_mask, other=0.0)
        shared = shared.to(tl.float32)
        factor = round_to(sigmoid(gate), out_ty)
        product = round_to(factor * shared, out_ty)
        values = (product + round_to(acc, out_ty)).to(out_ty)
    tl.store(out_ptr + tok * N + offs_n, values, mask=n_mask)


@triton.jit(do_not_specialize=["H"])
def gate_logits_kernel(x_ptr, w_ptr, out_ptr, H, x_stride, BK: tl.constexpr):
    """out[t] = x[t] . w (one row of gate logits), FP32 accumulation and output."""
    t = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((BK,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        m = k < H
        xv = tl.load(x_ptr + t * x_stride + k, mask=m, other=0.0).to(tl.float32)
        acc += xv * tl.load(w_ptr + k, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + t, tl.sum(acc, 0))


@triton.jit(do_not_specialize=["P", "INTER"])
def swiglu_kernel(c1_ptr, out_ptr, ids_ptr, P, E, INTER, BN: tl.constexpr):
    """out[p] = silu(gate) * up of the FP32 FC1 row of every local route, one rounding."""
    p = tl.program_id(0).to(tl.int64)
    e = tl.load(ids_ptr + p)
    if (e >= 0) & (e < E):
        offs = tl.program_id(1) * BN + tl.arange(0, BN)
        m = offs < INTER
        g = tl.load(c1_ptr + p * 2 * INTER + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(c1_ptr + p * 2 * INTER + INTER + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + p * INTER + offs, _silu_mul(g, u).to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _weights(ptrs, mask, EVF: tl.constexpr):
    """A weight tile in FP32; EVF: read once, so its L2 lines are the first evicted."""
    if EVF:
        w = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_first")
    else:
        w = tl.load(ptrs, mask=mask, other=0.0)
    return w.to(tl.float32)


@triton.jit
def _z_sum(ptr, offs_r, rank_mask, N, R: tl.constexpr):
    """Fixed-order FP32 sum of N partial shrinks stored R apart."""
    z = tl.zeros(offs_r.shape, dtype=tl.float32)
    for i in range(N):
        z += tl.load(ptr + i * R + offs_r, mask=rank_mask, other=0.0)
    return z


@triton.jit
def _fc1_row(
    row, wg, wu, ag, a_gap, bg, b_gap, scale, offs_n, n_mask, offs_r, rank_mask, H, zpart, KS,
    R: tl.constexpr, BK: tl.constexpr, HAS_LORA: tl.constexpr, EVF: tl.constexpr,
    PRE_Z: tl.constexpr, WAIT_Z: tl.constexpr, ONE_A: tl.constexpr = False,
):  # fmt: skip
    """silu(gate) * up of one row tile: gate/up GEMV over H in FP32 (``wg``/``wu`` point at
    the tile's weight rows, ``ag`` at the expert's LoRA A), the LoRA shrink of the same x
    chunks (PRE_Z: the row's split-K shrink partials [2, KS, R] at ``zpart`` instead, so no
    tile redoes it), the expand."""
    offs_k = tl.arange(0, BK)
    g = tl.zeros(offs_n.shape, dtype=tl.float32)
    u = tl.zeros(offs_n.shape, dtype=tl.float32)
    zg = tl.zeros(offs_r.shape, dtype=tl.float32)
    zu = tl.zeros(offs_r.shape, dtype=tl.float32)
    ag = ag + offs_r[:, None] * H
    for k0 in range(0, H, BK):
        k = k0 + offs_k
        k_mask = k < H
        xv = tl.load(row + k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
        wm = n_mask[:, None] & k_mask[None, :]
        g += tl.sum(_weights(wg + k[None, :], wm, EVF) * xv, 1)
        u += tl.sum(_weights(wu + k[None, :], wm, EVF) * xv, 1)
        if HAS_LORA and not PRE_Z:
            am = rank_mask[:, None] & k_mask[None, :]
            zg += tl.sum(tl.load(ag + k[None, :], mask=am, other=0.0).to(tl.float32) * xv, 1)
            if not ONE_A:
                zu += tl.sum(
                    tl.load(ag + a_gap + k[None, :], mask=am, other=0.0).to(tl.float32) * xv, 1
                )
    if HAS_LORA:
        if PRE_Z:
            if WAIT_Z:  # the shrink grid ran beside this one's weight stream
                tl.extra.cuda.gdc_wait()
            zg = _z_sum(zpart, offs_r, rank_mask, KS, R)
            if not ONE_A:
                zu = _z_sum(zpart + KS * R, offs_r, rank_mask, KS, R)
        if ONE_A:
            zu = zg
        g += scale * _expand_row(bg, zg, offs_n, n_mask, offs_r, rank_mask, R)
        u += scale * _expand_row(bg + b_gap, zu, offs_n, n_mask, offs_r, rank_mask, R)
    return _silu_mul(g, u)


@triton.jit
def _down_shrink_part(y, a2, z2, offs_n, n_mask, offs_r, rank_mask, K):
    """This tile's share of the down LoRA shrink, A2[:, tile] . y (y as FC2 reads it), to z2."""
    a = tl.load(a2 + offs_r[:, None] * K + offs_n[None, :], mask=rank_mask[:, None] & n_mask[None, :],
                other=0.0).to(tl.float32)  # fmt: skip
    tl.store(z2 + offs_r, tl.sum(a * y[None, :], 1), mask=rank_mask)


@triton.jit(
    do_not_specialize=["P", "work_total", "n_tiles", "a_gap", "b_gap", "T", "s_tiles", "sa_gap",
                       "sb_gap"]
)  # fmt: skip
def decode_fc1_kernel(
    x_ptr, w13_ptr, ids_ptr, a_ptr, b_ptr, scale_ptr, c2_ptr, P, E, H, INTER, a_gap, b_gap,
    x_stride, work_total, n_tiles, sw_ptr, sa_ptr, sb_ptr, sc2_ptr, T, S_INTER, s_tiles, sa_gap,
    sb_gap, zp_ptr, KS, z2_ptr, a2_ptr, szp_ptr, sKS, sz2_ptr, sa2_ptr,
    TOP_K: tl.constexpr, R: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_LORA: tl.constexpr, HAS_SHARED: tl.constexpr, SHARED_LORA: tl.constexpr,
    PDL: tl.constexpr, EVF: tl.constexpr, PRE_Z: tl.constexpr, WAIT_Z: tl.constexpr,
    ONE_A: tl.constexpr = False,
):  # fmt: skip
    """Decode FC1 per (pair, N tile of I): gate/up GEMV over H in FP32, the LoRA shrink of the
    same x chunks in registers, the expand, SwiGLU, one rounding into c2 [P, INTER].
    Routes outside [0, E) do no work (their c2 rows are not written). With HAS_SHARED, the
    work after the routes is (token, N tile of S_INTER) of the shared expert ``sw``
    [2 * S_INTER, H] (LoRA ``sa`` [2, R, H], ``sb`` [2, S_INTER, R]) into sc2 [T, S_INTER].
    PDL lets the FC2 launch start while this grid runs (it waits for this grid's writes).
    PRE_Z (LoRA): the gate/up shrinks come from split-K partials (``zp`` [P, 2, KS, R], shared
    ``szp`` [T, 2, sKS, R]); each tile also stores its share of the down shrink (A2 . the
    rounded c2 tile) to ``z2`` [P, n_tiles, R] / ``sz2`` [T, s_tiles, R] for FC2. WAIT_Z: this
    is a dependent launch of the shrink, waiting for it only before reading the partials, so
    the weight stream overlaps the shrink."""
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    offs_r = tl.arange(0, BR)
    rank_mask = offs_r < R
    scale = tl.load(scale_ptr) if HAS_LORA or SHARED_LORA else 0.0
    total = work_total + T * s_tiles if HAS_SHARED else work_total
    for work in range(tl.program_id(0), total, tl.num_programs(0)):
        if work < work_total:
            p = (work // n_tiles).to(tl.int64)
            offs_n = (work % n_tiles) * BN + tl.arange(0, BN)
            n_mask = offs_n < INTER
            e = tl.load(ids_ptr + p).to(tl.int64)
            if (e >= 0) & (e < E):
                wg = w13_ptr + e * 2 * INTER * H + offs_n[:, None] * H
                y = _fc1_row(
                    x_ptr + (p // TOP_K) * x_stride, wg, wg + INTER * H, a_ptr + e * R * H,
                    a_gap, b_ptr + e * INTER * R, b_gap, scale, offs_n, n_mask, offs_r,
                    rank_mask, H, zp_ptr + p * (1 if ONE_A else 2) * KS * R, KS, R, BK, HAS_LORA,
                    EVF, PRE_Z, WAIT_Z, ONE_A,
                )  # fmt: skip
                yc = y.to(c2_ptr.dtype.element_ty)
                tl.store(c2_ptr + p * INTER + offs_n, yc, mask=n_mask)
                if HAS_LORA and PRE_Z:
                    _down_shrink_part(
                        yc.to(tl.float32), a2_ptr + e * R * INTER,
                        z2_ptr + (p * n_tiles + work % n_tiles) * R, offs_n, n_mask, offs_r,
                        rank_mask, INTER,
                    )  # fmt: skip
        else:
            t = ((work - work_total) // s_tiles).to(tl.int64)
            offs_n = ((work - work_total) % s_tiles) * BN + tl.arange(0, BN)
            n_mask = offs_n < S_INTER
            wg = sw_ptr + offs_n[:, None] * H
            y = _fc1_row(
                x_ptr + t * x_stride, wg, wg + S_INTER * H, sa_ptr, sa_gap, sb_ptr, sb_gap, scale,
                offs_n, n_mask, offs_r, rank_mask, H, szp_ptr + t * 2 * sKS * R, sKS, R, BK,
                SHARED_LORA, EVF, PRE_Z, WAIT_Z,
            )  # fmt: skip
            yc = y.to(sc2_ptr.dtype.element_ty)
            tl.store(sc2_ptr + t * S_INTER + offs_n, yc, mask=n_mask)
            if SHARED_LORA and PRE_Z:
                _down_shrink_part(
                    yc.to(tl.float32), sa2_ptr,
                    sz2_ptr + (t * s_tiles + (work - work_total) % s_tiles) * R, offs_n, n_mask,
                    offs_r, rank_mask, S_INTER,
                )  # fmt: skip


@triton.jit
def _fc2_row(
    crow, wrow, arow, b_ptr, scale, offs_n, n_mask, offs_r, rank_mask, K, zpart, NZ,
    R: tl.constexpr, BK: tl.constexpr, HAS_LORA: tl.constexpr, EVF: tl.constexpr,
    PRE_Z: tl.constexpr,
):  # fmt: skip
    """One route's FC2 tile: the down GEMV over the K activations (``wrow`` points at the
    tile's weight rows, ``arow`` at the LoRA A rows) plus its LoRA, in FP32 (PRE_Z: the
    shrink is the fixed-order sum of FC1's NZ tile partials at ``zpart``)."""
    offs_k = tl.arange(0, BK)
    y = tl.zeros(offs_n.shape, dtype=tl.float32)
    z = tl.zeros(offs_r.shape, dtype=tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + offs_k
        k_mask = k < K
        cv = tl.load(crow + k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
        wm = n_mask[:, None] & k_mask[None, :]
        y += tl.sum(_weights(wrow + k[None, :], wm, EVF) * cv, 1)
        if HAS_LORA and not PRE_Z:
            am = rank_mask[:, None] & k_mask[None, :]
            z += tl.sum(tl.load(arow + k[None, :], mask=am, other=0.0).to(tl.float32) * cv, 1)
    if HAS_LORA:
        if PRE_Z:
            z = _z_sum(zpart, offs_r, rank_mask, NZ, R)
        y += scale * _expand_row(b_ptr, z, offs_n, n_mask, offs_r, rank_mask, R)
    return y


@triton.jit(do_not_specialize=["T", "work_total", "n_tiles"])
def decode_fc2_kernel(
    c2_ptr, w2_ptr, ids_ptr, w_ptr, a_ptr, b_ptr, scale_ptr, out_ptr, T, E, H, INTER,
    work_total, n_tiles, x_ptr, x_stride, gate_w_ptr, sc2_ptr, sw_ptr, sa_ptr, sb_ptr, S_INTER,
    z2_ptr, NZ, sz2_ptr, sNZ,
    TOP_K: tl.constexpr, R: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_LORA: tl.constexpr, HAS_SHARED: tl.constexpr, SHARED_LORA: tl.constexpr,
    PDL: tl.constexpr, EVF: tl.constexpr, PRE_Z: tl.constexpr,
):  # fmt: skip
    """Decode FC2 per (token, N tile of H): for each local route in slot order, the down GEMV
    over c2 plus its LoRA (shrink of the same c2 chunks in registers, expand), weighted and
    summed in FP32; with HAS_SHARED, + sigmoid(x . gate_w) * the shared expert's down GEMV
    (``sw`` [H, S_INTER], LoRA ``sa`` [R, S_INTER], ``sb`` [H, R]) over sc2. One rounding into
    out [T, H]. No route cache, no separate reduction. With PDL (a dependent launch of FC1) it
    waits for FC1's grid before reading c2."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    offs_r = tl.arange(0, BR)
    rank_mask = offs_r < R
    scale = tl.load(scale_ptr) if HAS_LORA or SHARED_LORA else 0.0
    for work in range(tl.program_id(0), work_total, tl.num_programs(0)):
        tok = (work // n_tiles).to(tl.int64)
        offs_n = (work % n_tiles) * BN + tl.arange(0, BN)
        n_mask = offs_n < H
        acc = tl.zeros((BN,), dtype=tl.float32)
        for j in tl.static_range(TOP_K):
            p = tok * TOP_K + j
            e = tl.load(ids_ptr + p).to(tl.int64)
            if (e >= 0) & (e < E):
                y = _fc2_row(
                    c2_ptr + p * INTER, w2_ptr + e * H * INTER + offs_n[:, None] * INTER,
                    a_ptr + e * R * INTER + offs_r[:, None] * INTER, b_ptr + e * H * R, scale,
                    offs_n, n_mask, offs_r, rank_mask, INTER, z2_ptr + p * NZ * R, NZ, R, BK,
                    HAS_LORA, EVF, PRE_Z,
                )  # fmt: skip
                acc += tl.load(w_ptr + p) * y
        if HAS_SHARED:
            y = _fc2_row(
                sc2_ptr + tok * S_INTER, sw_ptr + offs_n[:, None] * S_INTER,
                sa_ptr + offs_r[:, None] * S_INTER, sb_ptr, scale, offs_n, n_mask, offs_r,
                rank_mask, S_INTER, sz2_ptr + tok * sNZ * R, sNZ, R, BK, SHARED_LORA, EVF, PRE_Z,
            )  # fmt: skip
            g = tl.zeros((BK,), dtype=tl.float32)
            for k0 in range(0, H, BK):
                k = k0 + tl.arange(0, BK)
                m = k < H
                xv = tl.load(x_ptr + tok * x_stride + k, mask=m, other=0.0).to(tl.float32)
                g += xv * tl.load(gate_w_ptr + k, mask=m, other=0.0).to(tl.float32)
            acc += y / (1.0 + tl.exp(-tl.sum(g, 0)))
        tl.store(out_ptr + tok * H + offs_n, acc.to(out_ptr.dtype.element_ty), mask=n_mask)
