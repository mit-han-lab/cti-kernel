# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# The router and top-k stages of cti's own whole-block decode tree (moe/triton_block), run on
# their own so every MoE provider and every row count routes with the same arithmetic.
"""Triton kernels of ``ctikernels.moe.route`` (CuTe rewrite pending)."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..triton.topk_softmax import (
    reciprocal,
    softmax_padded,
    tree_sum,
    vllm_renormalize,
    vllm_softmax,
    vllm_topk,
)
from .triton_block.kernels import _mm


@triton.jit
def _round(v, ROUND: tl.constexpr):
    # The logit's storage precision (round to nearest even), back in FP32: 1 BF16, 2 FP16.
    if ROUND == 1:
        v = v.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    elif ROUND == 2:
        v = v.to(tl.float16, fp_downcast_rounding="rtne").to(tl.float32)
    return v


@triton.jit
def _opaque(v):
    # An identity the compiler cannot see through, applied to every slice's partial: it
    # keeps the partial dots and their ordered sum apart (``p0 + dot(a, b, 0)`` would
    # otherwise fold into ``dot(a, b, p0)``, in either operand order).
    return tl.inline_asm_elementwise(
        "mov.b32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _key32(z):
    # FP32 bits mapped to a signed key in the value's total order: -NaN < -inf < ... < -0
    # < +0 < ... < +inf < +NaN (the 32-bit form of the block tree's 16-bit key order).
    b = z.to(tl.int32, bitcast=True)
    return tl.where(b >= 0, b, b ^ 0x7FFFFFFF)


@triton.jit
def _route_logits_kernel(
    x_ptr,
    rw_ptr,
    z_ptr,
    T,
    H: tl.constexpr,
    E: tl.constexpr,
    KS: tl.constexpr,
    BT: tl.constexpr,
    BE: tl.constexpr,
    BKC: tl.constexpr,
    SPLIT: tl.constexpr,
    F32: tl.constexpr,
    ROUND: tl.constexpr,
    FLAT: tl.constexpr = False,
    NEB: tl.constexpr = 1,
):
    # A [BT, BE] logit tile from KS partial dots over ceil(H / KS)-wide slices. Each partial
    # is one MMA accumulation chain from zero along its slice; streaming the slice in
    # BKC-wide chunks into the same accumulator keeps that chain (masked tails add exact
    # zeros), so tile shapes can change with T without changing any logit. SPLIT: program
    # axis 2 is the slice, and its FP32 partial is stored for the top-k kernel to sum (the
    # block tree's layout); otherwise the program sums the slices itself in the same order
    # and stores the rounded logit.
    BK: tl.constexpr = (H + KS - 1) // KS
    # Expert blocks of one row block are adjacent programs: x streams from DRAM once.
    # One grid axis (no 65,535 cap on row blocks): expert blocks of a row block adjacent.
    pid = tl.program_id(0)
    cols = (pid % NEB) * BE + tl.arange(0, BE)
    rows = (pid // NEB) * BT + tl.arange(0, BT)
    rmask = rows < T
    cmask = cols < E
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    z = tl.zeros((BT, BE), dtype=tl.float32)
    if FLAT:
        # Fused tiles, BKC dividing BK: one pipelined loop over all of H; each slice's
        # chain ends at its boundary and joins the ordered sum (0 + p0 is p0 up to the sign
        # of zero, which ranking and weights fold anyway).
        part = tl.zeros((BT, BE), dtype=tl.float32)
        for kc in range(0, KS * BK, BKC):
            kk = kc + tl.arange(0, BKC)
            kmask = kk < H
            xa = tl.load(
                x_ptr + rows64[:, None] * H + kk[None, :],
                mask=rmask[:, None] & kmask[None, :],
                other=0.0,
            )
            wb = tl.load(
                rw_ptr + cols64[None, :] * H + kk[:, None],
                mask=cmask[None, :] & kmask[:, None],
                other=0.0,
            )
            part = _mm(xa, wb, part, F32)
            if (kc + BKC) % BK == 0:
                z += _opaque(part)
                part = tl.zeros((BT, BE), dtype=tl.float32)
    for ks in tl.static_range(KS):
        if (not FLAT) and ((not SPLIT) or ks == 0):
            sl = ks
            if SPLIT:
                sl = tl.program_id(1)
            part = tl.zeros((BT, BE), dtype=tl.float32)
            for kc in range(0, BK, BKC):
                off = kc + tl.arange(0, BKC)
                kk = sl * BK + off
                kmask = (off < BK) & (kk < H)
                xa = tl.load(
                    x_ptr + rows64[:, None] * H + kk[None, :],
                    mask=rmask[:, None] & kmask[None, :],
                    other=0.0,
                )
                wb = tl.load(
                    rw_ptr + cols64[None, :] * H + kk[:, None],
                    mask=cmask[None, :] & kmask[:, None],
                    other=0.0,
                )
                part = _mm(xa, wb, part, F32)
            if SPLIT:
                p = z_ptr + (sl * T + rows64)[:, None] * E + cols64[None, :]
                tl.store(p, part, mask=rmask[:, None] & cmask[None, :])
            elif ks == 0:
                z = _opaque(part)
            else:
                z += _opaque(part)
    if not SPLIT:
        # Rounded once to the logit precision, as the block tree's top-k does before ranking.
        p = z_ptr + rows64[:, None] * E + cols64[None, :]
        tl.store(
            p, _round(z, ROUND).to(z_ptr.dtype.element_ty), mask=rmask[:, None] & cmask[None, :]
        )


@triton.jit
def _route_topk_kernel(
    z_ptr,
    ids_ptr,
    w_ptr,
    valid_ptr,
    sel_ptr,
    p_ptr,
    T,
    TE,
    E: tl.constexpr,
    EP: tl.constexpr,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    SB: tl.constexpr,
    RENORM: tl.constexpr,
    VALID: tl.constexpr,
    KS: tl.constexpr,
    SPLIT: tl.constexpr,
    ROUND: tl.constexpr,
    SEL: tl.constexpr = False,
    PROBS: tl.constexpr = False,
    VLLM: tl.constexpr = False,
    ELTS: tl.constexpr = 8,
):
    # vLLM's softmax top-k (ctikernels.triton.topk_softmax) for any E and k (EP, KP: their
    # powers of two); VLLM (E a power of two up to 512) also takes its reduction order (ELTS
    # logits per 16-byte load).
    rows = tl.program_id(0) * SB + tl.arange(0, SB)
    rmask = rows < T
    live = rmask
    if VALID:
        valid = tl.minimum(tl.maximum(tl.load(valid_ptr), 0), T)
        live = rows < valid
    ids = tl.arange(0, EP)
    emask = ids < E
    ks = tl.arange(0, KP)
    base = z_ptr + rows.to(tl.int64)[:, None] * E + ids[None, :]
    lmask = rmask[:, None] & emask[None, :]
    if SPLIT:
        # the FP32 partials summed in the fixed order 0..KS-1, then rounded once
        z = tl.load(base, mask=lmask, other=0.0)
        for _ in tl.static_range(1, KS):
            base += TE  # pointer steps: 64-bit addresses for any T * E
            z += tl.load(base, mask=lmask, other=0.0)
        z = _round(z, ROUND)
    else:
        z = tl.load(base, mask=lmask, other=0.0).to(tl.float32)
    # The softmax over every expert: vLLM's topk_softmax for a power-of-two E up to 512 (its
    # warp's reduction order), otherwise the same formula with this tile's row sum; padding
    # experts get probability -1 and never rank.
    if VLLM:
        prob = vllm_softmax(z, SB, E, ELTS)
    else:
        prob = softmax_padded(z, emask, SB, EP)
    if SEL:
        # Given experts (replay): their probabilities, gathered over the row's experts.
        sel = tl.load(
            sel_ptr + rows.to(tl.int64)[:, None] * TOPK + ks[None, :],
            mask=rmask[:, None] & (ks[None, :] < TOPK),
            other=0,
        ).to(tl.int32)
        val = tl.zeros((SB, KP), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            e = tl.sum(tl.where(ks[None, :] == k, sel, 0), axis=1)
            v = tl.sum(tl.where(ids[None, :] == e[:, None], prob, 0.0), axis=1)
            val = tl.where(ks[None, :] == k, v[:, None], val)
    else:
        sel, val = vllm_topk(prob, SB, EP, TOPK, KP)
    if RENORM:
        wts = vllm_renormalize(val, SB, TOPK, KP)
    else:
        wts = tl.where(ks[None, :] < TOPK, val, 0.0)
    if PROBS:
        # every expert's probability, 0 on padding rows
        tl.store(
            p_ptr + rows.to(tl.int64)[:, None] * E + ids[None, :],
            tl.where(live[:, None], prob, 0.0),
            mask=rmask[:, None] & emask[None, :],
        )
    p = rows.to(tl.int64)[:, None] * TOPK + ks[None, :]
    smask = rmask[:, None] & (ks[None, :] < TOPK)
    tl.store(ids_ptr + p, tl.where(live[:, None], sel, -1), mask=smask)
    tl.store(w_ptr + p, tl.where(live[:, None], wts, 0.0), mask=smask)


@triton.jit
def _logit_chunk(z_ptr, row, c0, E, TE, KS: tl.constexpr, SPLIT: tl.constexpr,
                 ROUND: tl.constexpr, CH: tl.constexpr):  # fmt: skip
    """Experts c0..c0+CH of a row: (ids, in range, logits FP32)."""
    ids = c0 + tl.arange(0, CH)
    m = ids < E
    base = z_ptr + row.to(tl.int64) * E + ids
    z = tl.load(base, mask=m, other=0.0).to(tl.float32)
    if SPLIT:
        for _ in tl.static_range(1, KS):
            base += TE  # pointer steps: 64-bit addresses for any T * E
            z += tl.load(base, mask=m, other=0.0)
        z = _round(z, ROUND)
    return ids, m, z


@triton.jit
def _clamped(p):
    return tl.where((p == p) & (tl.abs(p) < float("inf")), p, 0.0)


@triton.jit
def _route_topk_stream_kernel(
    z_ptr,
    ids_ptr,
    w_ptr,
    valid_ptr,
    sel_ptr,
    p_ptr,
    T,
    TE,
    E,
    TOPK,
    RENORM: tl.constexpr,
    VALID: tl.constexpr,
    KS: tl.constexpr,
    SPLIT: tl.constexpr,
    ROUND: tl.constexpr,
    CH: tl.constexpr,
    SEL: tl.constexpr = False,
    PROBS: tl.constexpr = False,
):
    # One row per program, experts streamed in CH-wide chunks, for very many experts or a
    # large k: the tiled kernel's formula (vLLM's softmax top-k) with the sum added chunk by
    # chunk. Each selection pass takes the best probability strictly after the previous pick
    # in (probability descending, id ascending) order: repeated argmax, ties to the lower id.
    row = tl.program_id(0)
    live = True
    if VALID:
        live = row < tl.minimum(tl.maximum(tl.load(valid_ptr), 0), T)
    out = row.to(tl.int64) * TOPK
    if not live:
        for k in range(TOPK):
            tl.store(ids_ptr + out + k, -1)
            tl.store(w_ptr + out + k, 0.0)
        if PROBS:
            for c0 in range(0, E, CH):
                ids = c0 + tl.arange(0, CH)
                tl.store(
                    p_ptr + row.to(tl.int64) * E + ids, tl.zeros((CH,), tl.float32), mask=ids < E
                )
    else:
        top = float("-inf")
        for c0 in range(0, E, CH):
            ids, m, z = _logit_chunk(z_ptr, row, c0, E, TE, KS, SPLIT, ROUND, CH)
            top = tl.maximum(top, tl.max(tl.where(m, z, float("-inf")), axis=0))
        tot = 0.0
        for c0 in range(0, E, CH):
            ids, m, z = _logit_chunk(z_ptr, row, c0, E, TE, KS, SPLIT, ROUND, CH)
            e = tl.where(m, libdevice.exp(z - top), 0.0)
            tot += tl.sum(tree_sum(tl.reshape(e, (1, CH)), 1, CH), axis=0)
        r = reciprocal(tot)
        total = 0.0
        if SEL:
            # Given experts (replay): each one's probability.
            for k in range(TOPK):
                e = tl.load(sel_ptr + out + k).to(tl.int32)
                base = z_ptr + row.to(tl.int64) * E + e
                v = tl.load(base).to(tl.float32)
                if SPLIT:
                    for _ in tl.static_range(1, KS):
                        base += TE
                        v += tl.load(base)
                    v = _round(v, ROUND)
                v = _clamped(libdevice.exp(v - top) * r)
                tl.store(ids_ptr + out + k, e)
                tl.store(w_ptr + out + k, v)
                total += v
        else:
            prev_p = float("inf")
            prev_i = -1
            for k in range(TOPK):
                best_p = -1.0
                best_i = 2147483647
                for c0 in range(0, E, CH):
                    ids, m, z = _logit_chunk(z_ptr, row, c0, E, TE, KS, SPLIT, ROUND, CH)
                    p = _clamped(libdevice.exp(z - top) * r)
                    elig = m & ((p < prev_p) | ((p == prev_p) & (ids > prev_i)))
                    cb = tl.max(tl.where(elig, p, -1.0), axis=0)
                    ci = tl.min(tl.where(elig & (p == cb), ids, 2147483647), axis=0)
                    better = (cb > best_p) | ((cb == best_p) & (ci < best_i))
                    best_p = tl.where(better, cb, best_p)
                    best_i = tl.where(better, ci, best_i)
                tl.store(ids_ptr + out + k, best_i)
                tl.store(w_ptr + out + k, best_p)
                total += best_p
                prev_p = best_p
                prev_i = best_i
        if RENORM:
            scale = reciprocal(tl.where(total > 0, total, 1.0))
            for k in range(TOPK):
                tl.store(w_ptr + out + k, tl.load(w_ptr + out + k) * scale)
        if PROBS:
            for c0 in range(0, E, CH):
                ids, m, z = _logit_chunk(z_ptr, row, c0, E, TE, KS, SPLIT, ROUND, CH)
                p = p_ptr + row.to(tl.int64) * E + ids
                tl.store(p, _clamped(libdevice.exp(z - top) * r), mask=m)


@triton.jit
def _sqrtsoftplus(z):
    # sqrt(softplus(z)) as torch computes it (softplus threshold 20: z itself above it).
    sp = tl.where(z > 20.0, z, libdevice.log1p(libdevice.exp(z)))
    return libdevice.sqrt(sp)


@triton.jit
def _route_score_kernel(
    z_ptr,
    bias_ptr,
    ids_ptr,
    w_ptr,
    valid_ptr,
    sel_ptr,
    T,
    TE,
    scale,
    E: tl.constexpr,
    EP: tl.constexpr,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    SB: tl.constexpr,
    RENORM: tl.constexpr,
    VALID: tl.constexpr,
    KS: tl.constexpr,
    SPLIT: tl.constexpr,
    BIAS: tl.constexpr,
    SEL: tl.constexpr,
):
    # DeepSeek-V4 scoring (checkpoint inference/model.py Gate; SGLang 0.5.22
    # srt/layers/moe/topk.py biased_topk_impl): s = sqrt(softplus(z)) of the FP32 logit;
    # experts by s + bias descending (the bias only selects), ties to the lower id; weights
    # s / (sum of the selected s, slot order, + 1e-20) with renormalize, then * scale.
    rows = tl.program_id(0) * SB + tl.arange(0, SB)
    rmask = rows < T
    live = rmask
    if VALID:
        live = rows < tl.minimum(tl.maximum(tl.load(valid_ptr), 0), T)
    ids = tl.arange(0, EP)
    emask = ids < E
    ks = tl.arange(0, KP)
    base = z_ptr + rows.to(tl.int64)[:, None] * E + ids[None, :]
    lmask = rmask[:, None] & emask[None, :]
    z = tl.load(base, mask=lmask, other=0.0)
    if SPLIT:
        for _ in tl.static_range(1, KS):
            base += TE
            z += tl.load(base, mask=lmask, other=0.0)
    s = _sqrtsoftplus(z)
    if SEL:
        sel = tl.load(
            sel_ptr + rows.to(tl.int64)[:, None] * TOPK + ks[None, :],
            mask=rmask[:, None] & (ks[None, :] < TOPK),
            other=0,
        ).to(tl.int32)
    else:
        key = s
        if BIAS:
            key = key + tl.load(bias_ptr + ids, mask=emask, other=0.0)[None, :]
        code = _key32(key + 0.0)
        sel = tl.zeros((SB, KP), dtype=tl.int32)
        prev_c = tl.full((SB,), 2147483647, dtype=tl.int32)
        prev_i = tl.full((SB,), -1, dtype=tl.int32)
        for k in tl.static_range(TOPK):
            after = (code < prev_c[:, None]) | (
                (code == prev_c[:, None]) & (ids[None, :] > prev_i[:, None])
            )
            elig = emask[None, :] & after
            best = tl.max(tl.where(elig, code, -2147483648), axis=1)
            pick = tl.min(tl.where(elig & (code == best[:, None]), ids[None, :], EP), axis=1)
            sel = tl.where(ks[None, :] == k, pick[:, None], sel)
            prev_c = best
            prev_i = pick
    val = tl.zeros((SB, KP), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        e = tl.sum(tl.where(ks[None, :] == k, sel, 0), axis=1)
        v = tl.sum(tl.where(ids[None, :] == e[:, None], s, 0.0), axis=1)
        val = tl.where(ks[None, :] == k, v[:, None], val)
    if RENORM:
        tot = tl.zeros((SB,), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            tot += tl.sum(tl.where(ks[None, :] == k, val, 0.0), axis=1)
        val = val / (tot + 1e-20)[:, None]
    wts = val * scale
    p = rows.to(tl.int64)[:, None] * TOPK + ks[None, :]
    smask = rmask[:, None] & (ks[None, :] < TOPK)
    tl.store(ids_ptr + p, tl.where(live[:, None], sel, -1), mask=smask)
    tl.store(w_ptr + p, tl.where(live[:, None], wts, 0.0), mask=smask)


@triton.jit
def _hash_route_kernel(
    x_ptr,
    rw_ptr,
    tok_ptr,
    table_ptr,
    ids_ptr,
    w_ptr,
    valid_ptr,
    T,
    scale,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    KP: tl.constexpr,
    BK: tl.constexpr,
    RENORM: tl.constexpr,
    VALID: tl.constexpr,
):
    # Hash-routed layers (DeepSeek-V4's first num_hash_layers): the experts are
    # table[token id]; each weight is sqrt(softplus(x . w_e)) with the FP32 dot summed in
    # BK-wide chunks in order, normalised over the row's experts in slot order (+ 1e-20),
    # then * scale. One row per program.
    row = tl.program_id(0)
    live = row < T
    if VALID:
        live = row < tl.minimum(tl.maximum(tl.load(valid_ptr), 0), T)
    ks = tl.arange(0, KP)
    kmask = ks < TOPK
    out = row.to(tl.int64) * TOPK + ks
    if live:
        tok = tl.load(tok_ptr + row).to(tl.int64)
        e = tl.load(table_ptr + tok * TOPK + ks, mask=kmask, other=0)
        hs = tl.arange(0, BK)
        acc = tl.zeros((KP,), dtype=tl.float32)
        for h0 in range(0, H, BK):
            hm = (h0 + hs) < H
            xv = tl.load(x_ptr + row.to(tl.int64) * H + h0 + hs, mask=hm, other=0.0)
            wv = tl.load(
                rw_ptr + e.to(tl.int64)[:, None] * H + (h0 + hs)[None, :],
                mask=kmask[:, None] & hm[None, :],
                other=0.0,
            )
            acc += tl.sum(xv.to(tl.float32)[None, :] * wv.to(tl.float32), axis=1)
        s = tl.where(kmask, _sqrtsoftplus(acc), 0.0)
        if RENORM:
            tot = 0.0
            for k in tl.static_range(TOPK):
                tot += tl.sum(tl.where(ks == k, s, 0.0), axis=0)
            s = s / (tot + 1e-20)
        tl.store(ids_ptr + out, e, mask=kmask)
        tl.store(w_ptr + out, s * scale, mask=kmask)
    else:
        tl.store(ids_ptr + out, tl.full((KP,), -1, tl.int32), mask=kmask)
        tl.store(w_ptr + out, tl.zeros((KP,), tl.float32), mask=kmask)


@triton.jit
def _local_ids_kernel(ids_ptr, map_ptr, out_ptr, N, G, BLOCK: tl.constexpr):
    # Global expert id -> local id through the map; -1 for ids outside [0, G) or unmapped.
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    g = tl.load(ids_ptr + i, mask=m, other=-1)
    ok = (g >= 0) & (g < G)
    local = tl.load(map_ptr + g, mask=m & ok, other=-1)
    tl.store(out_ptr + i, tl.where(ok, local, -1).to(tl.int32), mask=m)
