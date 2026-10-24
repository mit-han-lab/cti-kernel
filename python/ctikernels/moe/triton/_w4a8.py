# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton W4A8 routed experts: launch policy and composition.

Launches: MXFP8 activation quant, local route alignment, FC1 (+ clamp, SiLU * up, route
weight, MXFP8 requant), FC2 into FP32 route rows, slot-ordered combine. Deterministic.
Scratch comes from the active ``GraphWorkspace`` (one slot); outside one, calls own theirs.
"""

import torch
import triton

from ..._runtime.workspace import GraphWorkspace
from .._align import moe_align_block_size_local
from . import _w4a8_kernels as K

# Fixed (per device class once tuned), by routed rows (T * top_k); tuned offline, no runtime autotune.
# (rows_max, BM, FC1 BN, FC1 BK, FC2 BN, FC2 BK, warps, stages)
_CONFIGS = ((64, 16, 32, 128, 64, 128, 4, 3), (1 << 30, 64, 64, 128, 128, 128, 4, 3))


_KEYS = ("bm", "bn1", "bk1", "bn2", "bk2", "warps", "stages")


def config(capability, rows):
    """Launch config; ``cvt`` feeds weights as E4M3 where that lowers to native MX MMA."""
    for entry in _CONFIGS:
        if rows <= entry[0]:
            return dict(zip(_KEYS, entry[1:], strict=True), cvt=capability == (12, 0))
    raise AssertionError("config table must end with an unbounded bucket")


def _sizes(tokens, h, inter, top_k, local, bm):
    pairs = tokens * top_k
    cap = pairs + local * (bm - 1)
    u8 = tokens * h + tokens * (h // 32) + cap * inter + cap * (inter // 32)
    return u8, pairs * h


def _scratch(x, tokens, call):
    size = lambda rows, *c: _sizes(rows, *c)
    counts = size(tokens, *call)
    group = GraphWorkspace.active()

    def allocate(n):
        return (
            torch.empty(max(n[0], 1), dtype=torch.uint8, device=x.device),
            torch.empty(max(n[1], 1), dtype=torch.float32, device=x.device),
        )

    if group is None:
        return allocate(counts)
    slot = ("moe_w4a8_triton", x.device, torch.uint8)
    group.reserve(slot, call, size, allocate)
    return group.take(slot, tokens, counts)


def quantize(x, q, s):
    groups = x.numel() // 32
    bg = 4 if groups <= 2048 else 64
    K.act_quant[(triton.cdiv(groups, bg),)](x, q, s, groups, bg, num_warps=4)


def experts(
    x, w13, s13, w2, s2, topk_weights, topk_ids, mapping, identity, swiglu_limit, out, device
):
    """``mapping`` maps global ids to local experts (-1 remote), ``identity`` when it is arange;
    ``device`` is (device class, capability); s13/s2 in the 128x4 interleave."""
    tokens, h = x.shape
    local, twice_i = w13.shape[0], w13.shape[1]
    inter, top_k = twice_i // 2, topk_ids.shape[1]
    pairs = tokens * top_k
    cfg = config(device[1], pairs)
    bm = cfg["bm"]
    u8, f32 = _scratch(x, tokens, (h, inter, top_k, local, bm))
    cap = pairs + local * (bm - 1)
    views, at = [], 0
    for n in (tokens * h, tokens * (h // 32), cap * inter, cap * (inter // 32)):
        views.append(u8[at : at + n])
        at += n
    qx, sx, qh, sh = views
    part = f32[: pairs * h]

    quantize(x, qx, sx)
    sorted_ids, block_e, _ = moe_align_block_size_local(
        topk_ids, bm, mapping, local, identity=identity
    )
    # Non-empty blocks come first and number at most one per route.
    blocks = min(block_e.numel(), pairs)
    launch = {"num_warps": cfg["warps"], "num_stages": cfg["stages"]}
    clamp = swiglu_limit is not None
    K.fc1[(blocks, inter // cfg["bn1"])](
        qx,
        sx,
        sorted_ids,
        block_e,
        topk_weights,
        w13,
        s13,
        qh,
        sh,
        pairs,
        float(swiglu_limit) if clamp else 0.0,
        h,
        inter,
        top_k,
        bm,
        cfg["bn1"],
        cfg["bk1"],
        clamp,
        cfg["cvt"],
        **launch,
    )
    K.fc2[(blocks, h // cfg["bn2"])](
        qh,
        sh,
        sorted_ids,
        block_e,
        w2,
        s2,
        part,
        pairs,
        h,
        inter,
        bm,
        cfg["bn2"],
        cfg["bk2"],
        cfg["cvt"],
        **launch,
    )
    bh = min(1024, triton.next_power_of_2(h))
    K.combine[(tokens, triton.cdiv(h, bh))](
        part,
        topk_ids,
        mapping,
        out,
        h,
        top_k,
        mapping.numel(),
        bh,
        not identity,
        num_warps=4,
    )
    return out


def local_routes(topk_weights, topk_ids, mapping, identity, remote, weight_dtype):
    """One launch: local expert ids (int32) and weights (``weight_dtype``), with remote and
    padding routes sent to id ``remote`` at weight 0 (FlashInfer's non-local convention)."""
    ids = torch.empty_like(topk_ids)
    weights = torch.empty(topk_weights.shape, dtype=weight_dtype, device=topk_weights.device)
    n = topk_ids.numel()
    block = min(1024, triton.next_power_of_2(max(n, 1)))
    K.local_routes[(triton.cdiv(n, block),)](
        topk_ids,
        topk_weights,
        mapping,
        ids,
        weights,
        n,
        remote,
        mapping.numel(),
        not identity,
        block,
        num_warps=4,
    )
    return ids, weights
