# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton MoE LoRA shrink over expert-sorted rows (no block padding).

Rows are grouped by expert: rows [offsets[e], offsets[e+1]) belong to expert e, row i reads
token ``token[i]`` and owns pair ``pair[i]``. Program (t, s) finds its (expert, BM-row tile)
by walking the experts' tile counts, then writes z[pair, s, :] = x[token] . A_s[e]^T in FP32
(or z[row, s, :], sorted-row order).
Every sum has a fixed order: graph replays are bitwise eager.
"""

import triton
import triton.language as tl

from ._lora_kernels import _shrink_tile


@triton.jit(do_not_specialize=["a_gap"])
def sorted_shrink_kernel(
    x_ptr,
    a_ptr,
    z_ptr,
    token_ptr,
    pair_ptr,
    offsets_ptr,
    a_gap,
    x_stride,
    BY_ROW: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    s = tl.program_id(1)
    expert = tile * 0 - 1
    start = tile * 0
    stop = tile * 0
    before = tile * 0
    for e in range(E):
        lo = tl.load(offsets_ptr + e)
        hi = tl.load(offsets_ptr + e + 1)
        tiles = (hi - lo + BM - 1) // BM
        hit = (tile >= before) & (tile < before + tiles)
        expert = tl.where(hit, e, expert)
        start = tl.where(hit, lo + (tile - before) * BM, start)
        stop = tl.where(hit, hi, stop)
        before += tiles
    if expert < 0:
        return
    rows = start + tl.arange(0, BM)
    row_mask = rows < stop
    tokens = tl.load(token_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    if BY_ROW:
        pairs = rows.to(tl.int64)
    else:
        pairs = tl.load(pair_ptr + rows, mask=row_mask, other=0).to(tl.int64)
    offs_r = tl.arange(0, BR)
    rank_mask = offs_r < R
    safe_r = tl.where(rank_mask, offs_r, 0)
    offs_k = tl.arange(0, BK)
    x_ptrs = x_ptr + tokens[:, None] * x_stride + offs_k[None, :]
    a_ptrs = a_ptr + s * a_gap + expert.to(tl.int64) * R * K + safe_r[None, :] * K + offs_k[:, None]
    z = _shrink_tile(x_ptrs, a_ptrs, row_mask, offs_k, rank_mask, K, BM, BR, BK)
    zp = z_ptr + pairs[:, None] * (S * R) + s * R + offs_r[None, :]
    tl.store(zp, z, mask=row_mask[:, None] & rank_mask[None, :])


def sorted_shrink_(z, x, a, token, pair, offsets):
    """z [P, S, R] FP32 = x[token] . A^T over the sorted rows; ``a`` is FactorSlices. Entries
    are pairs, or the rows themselves when ``pair`` is None."""
    rows, experts = token.shape[0], offsets.shape[0] - 1
    _, slices, rank = z.shape
    bm = 64
    grid = (triton.cdiv(rows, bm) + experts, slices)
    sorted_shrink_kernel[grid](
        x, a[0], z, token, token if pair is None else pair, offsets, a.gap, x.stride(0),
        BY_ROW=pair is None, E=experts, K=x.shape[1], S=slices, R=rank, BM=bm,
        BR=max(16, triton.next_power_of_2(rank)), BK=64, num_warps=4, num_stages=3,
    )  # fmt: skip
    return z
