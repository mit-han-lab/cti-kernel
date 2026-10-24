# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""vLLM 0.31's softmax top-k routing (csrc/libtorch_stable/moe/topk_softmax_kernels.cu,
``topkGating``: power-of-two expert counts up to 512, routed_scaling_factor 1).

In its arithmetic order: the softmax over every expert as ``expf(x - max) * (1 / row_sum)``,
``row_sum`` summed as that kernel's warp holds the row (each lane's VPT values in order, then
the xor butterfly over the row's lanes); top-k by repeated argmax of the probabilities (NaN
and +-inf probabilities clamped to 0), ties to the lower expert; renormalised weights
``p * (1 / selected_sum)`` with ``selected_sum`` added in pick order. libdevice's expf
flushes subnormal results (Triton links it with FTZ): only experts more than ~87 below the
row's maximum."""

import triton
import triton.language as tl
from triton.language.extra import libdevice

from .silu import _div_rn


@triton.jit
def _lanes(e, ROWS: tl.constexpr, E: tl.constexpr, ELTS: tl.constexpr):
    """[ROWS, TPR] per-lane sums of e [ROWS, E]: with ELTS logits per 16-byte load, lane t holds
    columns ldg * ELTS * TPR + ELTS * t + i, added in (ldg, i) order."""
    VPT: tl.constexpr = ELTS * max(1, E // (ELTS * 32))
    TPR: tl.constexpr = E // VPT
    LDG: tl.constexpr = VPT // ELTS
    g = tl.reshape(e, (ROWS, LDG, TPR, ELTS))
    ldg = tl.arange(0, LDG)[None, :, None, None]
    ii = tl.arange(0, ELTS)[None, None, None, :]
    s = tl.zeros((ROWS, TPR), dtype=tl.float32)
    for l in tl.static_range(LDG):  # noqa: E741
        for i in tl.static_range(ELTS):
            # one nonzero term per lane: the extraction is exact
            s += tl.sum(tl.sum(tl.where((ldg == l) & (ii == i), g, 0.0), axis=3), axis=1)
    return s


@triton.jit
def _fold(s, ROWS: tl.constexpr, M: tl.constexpr):
    """One butterfly step at width M, as lanes below M hold it: lane t adds lane t + M."""
    a, b = tl.split(tl.permute(tl.reshape(s, (ROWS, 2, M)), (0, 2, 1)))
    return a + b


@triton.jit
def _butterfly(s, ROWS: tl.constexpr, TPR: tl.constexpr):
    """The xor-butterfly sum over TPR <= 32 lanes, lane 0's value."""
    if TPR >= 32:
        s = _fold(s, ROWS, 16)
    if TPR >= 16:
        s = _fold(s, ROWS, 8)
    if TPR >= 8:
        s = _fold(s, ROWS, 4)
    if TPR >= 4:
        s = _fold(s, ROWS, 2)
    if TPR >= 2:
        s = _fold(s, ROWS, 1)
    return tl.sum(s, axis=1)


@triton.jit
def tree_sum(s, ROWS: tl.constexpr, N: tl.constexpr):
    """[ROWS] sums of s [ROWS, N] (N a power of two up to 8192) as a pairwise tree, element j
    adding element j + N/2 at each level: an order fixed by the code, whatever layout the
    compiler gives the tile (tl.sum's order follows the layout)."""
    if N >= 8192:
        s = _fold(s, ROWS, 4096)
    if N >= 4096:
        s = _fold(s, ROWS, 2048)
    if N >= 2048:
        s = _fold(s, ROWS, 1024)
    if N >= 1024:
        s = _fold(s, ROWS, 512)
    if N >= 512:
        s = _fold(s, ROWS, 256)
    if N >= 256:
        s = _fold(s, ROWS, 128)
    if N >= 128:
        s = _fold(s, ROWS, 64)
    if N >= 64:
        s = _fold(s, ROWS, 32)
    return _butterfly(s, ROWS, min(N, 32))


@triton.jit
def vllm_softmax(z, ROWS: tl.constexpr, E: tl.constexpr, ELTS: tl.constexpr):
    """Every expert's probability [ROWS, E] of logits z (FP32 values of the gating output,
    ELTS = 16 bytes / its element size, at most E)."""
    VPT: tl.constexpr = ELTS * max(1, E // (ELTS * 32))
    top = tl.max(z, axis=1)
    e = libdevice.exp(z - top[:, None])
    row_sum = _butterfly(_lanes(e, ROWS, E, ELTS), ROWS, E // VPT)
    p = e * _div_rn(tl.full((ROWS,), 1.0, tl.float32), row_sum)[:, None]
    # vLLM clamps NaN and +-inf probabilities to 0 (nonfinite logits): picks 0, 1, ... by id
    return tl.where((p == p) & (tl.abs(p) < float("inf")), p, 0.0)


@triton.jit
def reciprocal(x):
    """1 / x, an IEEE divide (not FTZ)."""
    return _div_rn(tl.zeros_like(x) + 1.0, x)


@triton.jit
def softmax_padded(z, emask, ROWS: tl.constexpr, N: tl.constexpr):
    """vLLM's softmax formula on a padded tile z [ROWS, N] (``emask``: the real experts) for
    expert counts outside its warp layouts: expf(z - max) * (1 / sum), the sum a pairwise tree;
    NaN and +-inf clamped to 0, padding -1 (never picked)."""
    zm = tl.where(emask[None, :], z, float("-inf"))
    top = tl.max(zm, axis=1)
    e = tl.where(emask[None, :], libdevice.exp(zm - top[:, None]), 0.0)
    p = e * reciprocal(tree_sum(e, ROWS, N))[:, None]
    p = tl.where((p == p) & (tl.abs(p) < float("inf")), p, 0.0)
    return tl.where(emask[None, :], p, -1.0)


@triton.jit
def vllm_topk(p, ROWS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr, KP: tl.constexpr):
    """(experts int32 [ROWS, KP], probabilities FP32 [ROWS, KP]) of the TOPK picks of p in pick
    order (columns from TOPK on: expert 0, -inf): repeated argmax, ties to the lower expert."""
    ids = tl.arange(0, E)[None, :]
    ks = tl.arange(0, KP)[None, :]
    sel = tl.zeros((ROWS, KP), dtype=tl.int32)
    val = tl.full((ROWS, KP), float("-inf"), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(p, axis=1)
        pick = tl.min(tl.where(p == best[:, None], ids, E), axis=1)
        sel = tl.where(ks == k, pick[:, None], sel)
        val = tl.where(ks == k, best[:, None], val)
        p = tl.where(ids == pick[:, None], -1.0, p)  # probabilities are >= 0
    return sel, val


@triton.jit
def vllm_renormalize(val, ROWS: tl.constexpr, TOPK: tl.constexpr, KP: tl.constexpr):
    """val * (1 / selected_sum), the sum added in slot order 0..TOPK-1 (0 past TOPK)."""
    ks = tl.arange(0, KP)[None, :]
    total = tl.zeros((ROWS,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        total += tl.sum(tl.where(ks == k, val, 0.0), axis=1)
    scale = _div_rn(tl.full((ROWS,), 1.0, tl.float32), tl.where(total > 0, total, 1.0))
    return tl.where(ks < TOPK, val * scale[:, None], 0.0)


@triton.jit
def topk_softmax(z, ROWS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                 ELTS: tl.constexpr = 8):  # fmt: skip
    """(experts int32 [ROWS, TOPK] in pick order, renormalised weights FP32) of logits z (FP32
    [ROWS, E]; ELTS 8 for 16-bit gating outputs)."""
    sel, val = vllm_topk(vllm_softmax(z, ROWS, E, ELTS), ROWS, E, TOPK, TOPK)
    return sel, vllm_renormalize(val, ROWS, TOPK, TOPK)
