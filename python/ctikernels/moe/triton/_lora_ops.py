# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton MoE LoRA provider: launch policy, the low-level insertion op and the BF16 composition.

Paths (all deterministic, no atomics):
- pair: per routed pair, no alignment; FP32 shrink GEMV in registers (decode batches, any rank);
- block: per aligned expert block, MMA shrink kept in registers then expand (rank <= 128);
- general: shrink to an FP32 z scratch, then expand (rank > 128 on aligned batches).
Scratch comes from the active ``GraphWorkspace`` (one slot); outside one, calls own theirs.
"""

import torch
import triton

from ..._runtime.tables import DeviceClass, Dim, TunedOp, pow2_edges
from ..._runtime.workspace import GraphWorkspace
from ...activation.inference import silu_and_mul
from ...linear.block_fp8 import quantize_per_token_group_fp8
from ...linear.block_fp8_gated import silu_and_mul_quantize_fp8
from ...moe_impl import (
    block_fp8_moe_provider,
    fused_moe_kernel_triton,
    moe_align_block_size,
    moe_sum_reduce_triton,
)
from .._block import expert_tiles
from .._lora_common import device_info, overlap_stream, scale_tensor
from .._lora_weights import FactorSlices
from . import _lora_kernels as K
from ._lora_backward import TritonSteps, layout  # noqa: F401 (TritonSteps: moe.lora's backward)

_PAIR_ROWS = 64  # pairs at or below run the pair path (one program per pair and N chunk)
_DENSE_PAIR_ROWS = 8  # the same for dense shared rows (one expert)
_DENSE_BLOCK_M = 16  # dense rows per block on the split shrink + expand path
_MAX_SHRINK_SPLITS = 8  # K splits of the general path's shrink (FP32 partials summed in order)
_STATIC = {}  # per-shape metadata tensors, built by eager warm-up calls


def _capturing():
    return torch.cuda.is_current_stream_capturing()


def _static(key, build):
    cached = _STATIC.get(key)
    if cached is None:
        if _capturing():
            raise RuntimeError(f"warm up MoE LoRA shape {key[1:]} before CUDA graph capture")
        cached = _STATIC[key] = build()
    return cached


def _pow2(n, floor=16):
    return max(floor, triton.next_power_of_2(n))


def _sms(device):
    return device_info(device)[1]


def _pow2_floor(n):
    return 1 << (max(n, 1).bit_length() - 1)


def _splits(pairs, slices, k, bk, sms, waves=2):
    """K splits of the decode shrink so its programs fill about ``waves`` waves; none once
    the (pair, slice) programs alone cover the SMs. Measured (graphs, full layer, both
    models, ranks 8-64): split 1.03-1.43x faster on the RTX 5090 at T <= 8, 1.02-1.06x on
    GB10 at T <= 2; GB10 T=4 r64 0.98x without this cut-off. A split may be one BK tile:
    Qwen3.5-35B-A3B EP2 T=1 r16 on GB10, the down shrink (K=512) at 4 splits cut the LoRA
    increment from 8.5 to 6.1 us per layer (1 split: 28.8 us; the base alone takes 90 us)."""
    if pairs * slices >= sms:
        return 1
    return max(1, min(triton.cdiv(k, bk), triton.cdiv(waves * sms, pairs * slices)))


def _default_config(shape, dc: DeviceClass):
    """Today's rules: register tiles of <= 8K FP32 values, two waves, split-K decode."""
    br = _pow2(shape["rank"])
    tile = max(16, min(128, 8192 // br))
    splits = _splits(shape["m"], shape["slices"], shape["k"], tile, dc.sms)
    if dc.sms >= 128:
        # RTX 5090 (170 SMs), Qwen3.5-35B-A3B EP1 + shared expert, rotating routes: narrow
        # tiles fill the SMs; T=1 34 us (no LoRA) / 44 us (r16) vs 76 / 93 us at the GB10 tiles.
        fc2_bn = 8 if shape["m"] <= 8 else 16
        decode = {
            "decode_fc1_bn": 16,
            "decode_fc1_bk": _decode_bk(128, 16, br, 16384),
            "decode_fc2_bn": fc2_bn,
            "decode_fc2_bk": _decode_bk(512 if fc2_bn == 8 else 256, fc2_bn, br, 8192),
            "decode_warps": 8,
        }
    else:
        # GB10, Qwen3.5-35B-A3B EP2 r16, rotating routes, LoRA shrinks once per row: T=1
        # 113.1 us, T=4 431 us (vs 113.8 / 436 at (32, 512) / (32, 256) x 4 warps); alone, FC1
        # and FC2 stream at 247 GB/s here vs 249 for plain loads of the same bytes.
        decode = {
            "decode_fc1_bn": 16,
            "decode_fc1_bk": _decode_bk(128, 16, br, 16384),
            "decode_fc2_bn": 16,
            "decode_fc2_bk": _decode_bk(256, 16, br, 8192),
            "decode_warps": 8,
        }
    # Split shrink + expand (RTX 5090, Qwen3.5-35B-A3B, rotating inputs): routed T=64 r16
    # LoRA +48 us; dense shared rows (one expert) T=64 r64 10.6 us vs 43.5 per pair.
    dense = shape["e_local"] == 1
    split = {
        "block_bk": 128,
        "shrink_br": 16 if br <= 16 else 32,
        "expand_bn": 32 if dense and shape["m"] <= 64 else 64,
        "block_warps": 8 if dense and br >= 64 else 4,
    }
    return {
        "pair_bk": tile,
        "pair_bn": tile,
        "pair_waves": 2,
        "splits": min(16, _pow2_floor(splits)),
        "pair_warps": 4,
        "block_stages": 2,
        **decode,
        **split,
    }


def _decode_bk(bk, bn, br, budget):
    """Largest power-of-two chunk <= bk whose [max(bn, br), chunk] FP32 tile fits budget."""
    while bk > 64 and max(bn, br) * bk > budget:
        bk //= 2
    return bk


def _decode_valid(config, shape):
    """Decode GEMV register budget: [max(BN, BR), BK] FP32 tiles <= 16K (FC1) / 8K (FC2).
    Only the decode kind at ranks <= _DECODE_RANK runs those tiles."""
    if shape["kind"] != "decode" or shape["rank"] > _DECODE_RANK:
        return True  # those shapes never run the decode GEMVs
    br = _pow2(shape["rank"])
    fc1 = max(config["decode_fc1_bn"], br) * config["decode_fc1_bk"] <= 16384
    fc2 = max(config["decode_fc2_bn"], br) * config["decode_fc2_bk"] <= 8192
    return fc1 and fc2


TUNED = TunedOp(
    "moe.lora.triton",
    version=4,
    space={
        "pair_bk": (16, 32, 64, 128),
        "pair_bn": (16, 32, 64, 128),
        "pair_waves": (1, 2, 4),
        "splits": (1, 2, 4, 8, 16),
        "pair_warps": (4, 8),
        "block_bk": (32, 64, 128),
        "block_warps": (4, 8),
        "block_stages": (2, 3, 4),
        "shrink_br": (16, 32, 64),
        "expand_bn": (32, 64, 128),
        "decode_fc1_bn": (8, 16, 32, 64),
        "decode_fc1_bk": (64, 128, 256, 512),
        "decode_fc2_bn": (8, 16, 32, 64),
        "decode_fc2_bk": (64, 128, 256, 512),
        "decode_warps": (4, 8),
    },
    dims=(
        Dim("m", pow2_edges(1, 16384)),  # routed pairs of the projection (its rows)
        Dim("tokens_per_expert", pow2_edges(1, 16384)),
        Dim("e_local"),
        Dim("kind"),  # w13_act | w13 | w2 | decode (whole MLP: k = H, n = I, slices = 2)
        Dim("k"),
        Dim("n"),
        Dim("rank", (1, 16, 32, 64, 128, 256)),  # 1: no LoRA (the decode runs no shrink)
        Dim("slices"),
        Dim("base"),  # bf16 | fp8_block
    ),
    default=_default_config,
    valid=_decode_valid,
)


def tuned_config(device, m, tokens_per_expert, e_local, kind, k, n, rank, slices, base):
    """The only source of the launchers' tile parameters (the tuner pins configs here)."""
    return TUNED.config(
        device, m=m, tokens_per_expert=tokens_per_expert, e_local=e_local, kind=kind, k=k, n=n,
        rank=rank, slices=slices, base=base,
    )  # fmt: skip


def _config(x, p, experts, kind, k, n, rank, slices, base):
    return tuned_config(x.device, p, -(-p // experts), experts, kind, k, n, rank, slices, base)


def _chunks(tiles, programs, target):
    """(tiles per chunk, chunks) splitting N so ``programs`` x chunks reach ``target``."""
    chunks = max(1, min(tiles, triton.cdiv(target, max(programs, 1))))
    per = triton.cdiv(tiles, chunks)
    return per, triton.cdiv(tiles, per)


def _composition_size(max_rows, top_k, width, ids, floats):
    """(cache-dtype, int32, FP32) element counts of the composition at ``max_rows`` tokens."""
    return max_rows * top_k * width, max_rows * top_k * ids, max_rows * top_k * floats


def _split_size(max_rows, top_k, device, experts, kind, k, n, rank, slices, base):
    """FP32 split-K partials: the largest over every token count up to ``max_rows`` (the
    split count shrinks as pairs grow, so the product is not monotone)."""
    sizes = (
        p * slices * rank * _split_count(
            tuned_config(device, p, -(-p // experts), experts, kind, k, n, rank, slices, base),
            k,
        )
        for p in range(top_k, min(max_rows * top_k, _PAIR_ROWS) + 1, top_k)
    )  # fmt: skip
    return 0, 0, max(sizes, default=0)


def _split_count(cfg, k):
    """Splits of a config: whole BK tiles per split, at least one tile."""
    bk = cfg["pair_bk"]
    return triton.cdiv(k, triton.cdiv(triton.cdiv(k, cfg["splits"]), bk) * bk)


def _general_size(max_rows, top_k, slices, rank, splits_rows):
    """Split-shrink partials: splits x pairs stays within max(pairs, ``splits_rows``), as the
    split count only grows while the row blocks leave SMs idle (and splits <= the cap)."""
    pairs = max_rows * top_k
    return 0, 0, max(pairs, min(_MAX_SHRINK_SPLITS * pairs, splits_rows)) * slices * rank


def _allocate(x, counts):
    return tuple(
        torch.empty((max(n, 1),), dtype=d, device=x.device)
        for n, d in zip(counts, (x.dtype, torch.int32, torch.float32), strict=True)
    )


def scratch(x, tokens, call, size, role):
    """Flat (cache dtype, int32, FP32) scratch for one call: the active GraphWorkspace's slot
    for ``role`` (one region per role for every graph and layer of the group, sized for its
    max_rows; roles live at once within a call, so each has its own), otherwise call-owned (a
    capture outside a workspace takes it from the graph's private pool)."""
    counts = size(tokens, *call)
    group = GraphWorkspace.active()
    if group is None or not x.is_cuda:
        return _allocate(x, counts)
    _reserve(group, x, role, call, size)
    return group.take(_slot(x, role), tokens, counts)


def _slot(x, role):
    return ("moe_lora_triton", role, x.device, x.dtype)


def _reserve(group, x, role, call, size):
    group.reserve(
        _slot(x, role), (size, *call), lambda rows, f, *rest: f(rows, *rest),
        lambda n: _allocate(x, n),
    )  # fmt: skip


def _reserve_partials(group, x, kind, top_k, experts, k, n, rank, slices, base, role=None):
    call = (top_k, x.device, experts, kind, k, n, rank, slices, base)
    _reserve(group, x, role or f"partials-{kind}", call, _split_size)


def _reserve_shrink(group, x, top_k, slices, rank):
    """The split-shrink slot for any block_m: the split cap bounds it within max_rows."""
    call = (top_k, slices, rank, _MAX_SHRINK_SPLITS * group.max_rows * top_k)
    _reserve(group, x, "shrink", call, _general_size)


def _reserve_projection(group, x, a, b, top_k, experts, kind, dense):
    """``apply_``'s slots at any T <= max_rows: the pair path's partials (from T = 1) and the
    aligned path's shrink (once T x top_k passes the pair rows)."""
    rank, slices = a.shape[-2], len(a)
    pair_rows = _DENSE_PAIR_ROWS if dense else _PAIR_ROWS
    if top_k <= pair_rows or (not dense and experts < 64):
        _reserve_partials(group, x, kind, top_k, experts, a.shape[-1], b.shape[-2], rank, slices,
                          "bf16")  # fmt: skip
    if group.max_rows * top_k > pair_rows:
        _reserve_shrink(group, x, top_k, slices, rank)


def _reserve_reachable(group, x, e, top_k, inter, lora, base, shared_expert):
    """Reserve every path-dependent slot a call of this shape takes at any T <= max_rows: the
    group may warm and capture its sizes one at a time, largest first, so a path first taken
    at a smaller T (the fused decode, the pair path) must find its slot after the first
    capture. The composition and shared slots are taken on every path."""
    h, dev = x.shape[1], x.device  # noqa: F841
    rank = 0 if lora is None else lora.rank
    factors = None if shared_expert is None else shared_expert[3]
    s_inter = 0 if shared_expert is None else shared_expert[1].shape[1]
    s_rank = 0 if factors is None else factors[0].shape[-2]
    m = group.max_rows

    def partials(kind, top, experts, k, n, r, slices, b, role=None):
        _reserve_partials(group, x, kind, top, experts, k, n, r, slices, b, role)

    def shrink(top, slices, r):
        _reserve_shrink(group, x, top, slices, r)

    decode_ok = False  # no base takes the fused decode on this line: BF16 runs fused_experts' path
    kind13 = "w13"  # the gate/up delta goes into the FC1 cache (BF16 and block-FP8)
    # Each path predicate is monotone in T: its edges and both ends cover every path.
    edges = (e // (4 * top_k), _PAIR_ROWS // top_k, _DENSE_PAIR_ROWS)
    for t in sorted({1, m, *edges, *(v + 1 for v in edges)} & set(range(1, m + 1))):
        p = t * top_k
        if decode_ok and p * 4 <= e:
            if lora is not None:
                partials("w13_act", top_k, e, h, inter, rank, 2, "bf16")
            if lora is not None or factors is not None:
                z_call = (top_k, inter, s_inter, max(rank, 1))
                _reserve(group, x, "decode-z", z_call, _decode_z_size)
            if factors is not None:
                partials("w13_act", 1, 1, h, s_inter, s_rank, 2, "bf16", "partials-shared")
            continue
        if factors is not None:
            if t <= _DENSE_PAIR_ROWS:
                partials("w13_act", 1, 1, h, s_inter, s_rank, 2, "bf16")
                partials("w2", 1, 1, s_inter, h, s_rank, 1, "bf16")
            else:
                shrink(1, 2, s_rank)
                shrink(1, 1, s_rank)
        if lora is not None:
            if p <= _PAIR_ROWS:
                partials(kind13, top_k, e, h, inter, rank, 2, base)
                partials("w2", top_k, e, inter, h, rank, 1, base)
            else:
                shrink(top_k, 2, rank)
                shrink(top_k, 1, rank)


def _naive_alignment(pairs, block_m, device):
    """Block i holds pair i alone (padded with ``pairs``)."""

    def build():
        ids = torch.full((pairs, block_m), pairs, dtype=torch.int32, device=device)
        ids[:, 0] = torch.arange(pairs, dtype=torch.int32, device=device)
        return ids.view(-1), torch.tensor([pairs * block_m], dtype=torch.int32, device=device)

    return _static(("naive", pairs, block_m, device), build)


def _kind(activated, slices):
    return "w13_act" if activated is not None else ("w13" if slices == 2 else "w2")


def _one_a(a):
    """Shared-A gate/up (both slices the same A): the shrink runs once on this one slice."""
    return FactorSlices((a[0],)) if len(a) == 2 and a.gap == 0 else None


def _launch_pair(
    out, x, a, b, ids, scale, weights, z_out, activated, row_div, experts, top_k, base="bf16"
):  # fmt: skip
    p, rank, slices = out.shape[0], a.shape[-2], len(a)
    k, n = a.shape[-1], b.shape[-2]
    kind = _kind(activated, slices)
    cfg = _config(x, p, experts, kind, k, n, rank, slices, base)
    splits = _split_count(cfg, k)
    part = out
    if splits > 1:
        # One DRAM pass over A spread across programs; the pair kernel sums the partials.
        shrink = _one_a(a) or a
        part = _pair_partials(x, shrink, ids, row_div, experts, top_k, kind, n, base, cfg, p)
    _pair_expand(
        out, x, a, b, ids, scale, weights, z_out, activated, row_div, experts, cfg, part, splits
    )


def _partials_buffer(x, a, experts, top_k, kind, n, base, p, role=None):
    rank, slices, k = a.shape[-2], len(a), a.shape[-1]
    call = (top_k, x.device, experts, kind, k, n, rank, slices, base)
    return scratch(x, p // top_k, call, _split_size, role or f"partials-{kind}")[2]


def _pair_partials(
    x, a, ids, row_div, experts, top_k, kind, n, base, cfg, p, part=None, role=None, pdl=False
):  # fmt: skip
    """Split-K shrink partials [P, S, KS, R] in FP32 (scratch), KS = ``_split_count``."""
    rank, slices, k = a.shape[-2], len(a), a.shape[-1]
    bk, splits = cfg["pair_bk"], _split_count(cfg, k)
    k_split = triton.cdiv(triton.cdiv(k, splits), bk) * bk
    if part is None:
        part = _partials_buffer(x, a, experts, top_k, kind, n, base, p, role)
    K.pair_shrink_kernel[(p * slices, splits)](
        x, a[0], part, ids, p, experts, k, a.gap, x.stride(0), splits, k_split,
        ROW_DIV=row_div, S=slices, R=rank, BR=_pow2(rank), BK=bk, PDL=pdl,
        num_warps=cfg["pair_warps"],
    )  # fmt: skip
    return part


def _pair_expand(
    out, x, a, b, ids, scale, weights, z_out, activated, row_div, experts, cfg, part, splits
):  # fmt: skip
    p, rank, slices = out.shape[0], a.shape[-2], len(a)
    k, n = a.shape[-1], b.shape[-2]
    br, bk, bn, warps = _pow2(rank), cfg["pair_bk"], cfg["pair_bn"], cfg["pair_warps"]
    target = cfg["pair_waves"] * _sms(x.device)
    per, chunks = _chunks(triton.cdiv(n, bn), p, target)
    work = p * chunks
    K.pair_kernel[(min(work, target),)](
        x,
        a[0],
        b[0],
        out,
        out if activated is None else activated,
        out if z_out is None else z_out,
        ids,
        scale if weights is None else weights,
        scale,
        part,
        p,
        experts,
        k,
        n,
        a.gap,
        b.gap,
        x.stride(0),
        out.stride(0),
        0 if activated is None else activated.stride(0),
        work,
        chunks,
        per,
        splits,
        ROW_DIV=row_div,
        S=slices,
        R=rank,
        BR=br,
        BK=bk,
        BN=bn,
        WEIGHTED=weights is not None,
        ACT=activated is not None,
        SAVE_Z=z_out is not None,
        SPLIT=splits > 1,
        ONE_A=_one_a(a) is not None,
        num_warps=warps,
    )


def _launch_block(
    out, x, a, b, scale, weights, z_out, activated, row_div, align, experts, top_k, base="bf16",
    round_delta=False, prescale=False,
):  # fmt: skip
    sorted_ids, expert_ids, padded, block_m = align
    p, rank = out.shape[0], a.shape[-2]
    k, n = a.shape[-1], b.shape[-2]
    sms = _sms(x.device)
    cfg = _config(x, p, experts, _kind(activated, len(a)), k, n, rank, len(a), base)
    run = {"num_warps": cfg["block_warps"], "num_stages": cfg["block_stages"]}
    blocks = triton.cdiv(sorted_ids.numel(), block_m)
    one_a = _one_a(a)
    shrink = one_a or a  # the A slices the shrink reads (and z holds)
    common = {"ROW_DIV": row_div, "S": len(shrink), "R": rank, "BM": block_m}
    flags = {"WEIGHTED": weights is not None, "ACT": activated is not None}
    w = scale if weights is None else weights
    act = out if activated is None else activated
    act_stride = 0 if activated is None else activated.stride(0)
    # Split shrink (FP32 partials over K) + expand: each row block's shrink runs once; a fused
    # shrink+expand per N chunk redid the full-K shrink per chunk (RTX 5090, T=64 r16: LoRA
    # +48 us vs +94 us).
    br, bk = cfg["shrink_br"], cfg["block_bk"]
    # Split count as for two A slices: shared-A keeps the separate path's K splits (its bits).
    programs = blocks * triton.cdiv(rank, br) * len(a)
    # A requested z_out takes the shrink directly (one split; only local rows are stored).
    # Splits fill 2 waves: splits x row blocks <= 2 SMs, so splits x pairs <= 2 SMs x block_m.
    splits = min(_MAX_SHRINK_SPLITS, triton.cdiv(k, bk), _pow2_floor(max(1, 2 * sms // programs)))
    splits = 1 if z_out is not None else splits
    k_split = triton.cdiv(triton.cdiv(k, splits), bk) * bk
    splits = triton.cdiv(k, k_split)
    z = z_out
    if z is None:
        call = (top_k, len(a), rank, 2 * sms * block_m)
        _, _, flat = scratch(x, p // top_k, call, _general_size, "shrink")
        z = flat[: splits * p * len(shrink) * rank].view(splits, p, len(shrink), rank)
    K.shrink_kernel[(blocks, triton.cdiv(rank, br), len(shrink) * splits)](
        x, a[0], z, sorted_ids, expert_ids, padded, sorted_ids.numel(), p, experts, k, shrink.gap,
        x.stride(0), k_split, **common, BR=br, BK=bk, KS=splits, **run,
    )  # fmt: skip
    bn = cfg["expand_bn"]
    del common["ROW_DIV"]
    common["S"] = len(a)
    K.expand_kernel[(blocks, triton.cdiv(n, bn))](
        z, b[0], out, act, w, scale, sorted_ids, expert_ids, padded, sorted_ids.numel(), p,
        experts, n, b.gap, out.stride(0), act_stride, **common, BR=br, BN=bn, **flags,
        KS=splits, ROUND_DELTA=round_delta, ONE_A=one_a is not None, PRESCALE=prescale, **run,
    )  # fmt: skip


def apply_(
    out,
    x,
    a,
    b,
    ids,
    scale,
    top_k,
    topk_weights,
    z_out,
    *,
    down=False,
    activated=None,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
):
    """Add the LoRA delta of one projection to ``out`` [P, S*N] in place (or, with
    ``activated``, write silu(gate)*up of the updated gate/up cache there). Inputs are
    validated by ``moe.lora``; ``a``/``b`` are FactorSlices, ``ids`` None means dense
    shared rows (expert 0 everywhere)."""
    p = out.shape[0]
    if p == 0:
        return out if activated is None else activated
    row_div = 1 if down else top_k
    experts = a.shape[0]
    weights = topk_weights.reshape(-1) if down and topk_weights is not None else None
    group = GraphWorkspace.active()
    if group is not None and x.is_cuda and z_out is None and not _capturing():
        kind = _kind(activated, len(a))
        _reserve_projection(group, x, a, b, top_k, experts, kind, ids is None)
    if ids is None:
        sorted_token_ids = None
        flat = _dense_ids(p, x.device)
    else:
        flat = ids.reshape(-1)
    # Without the caller's alignment, only >= 64 experts align here: fewer take the
    # alignment kernel's single-block path, which mishandles the skip bucket.
    pair_rows = _PAIR_ROWS if ids is not None else _DENSE_PAIR_ROWS
    if p <= pair_rows or (sorted_token_ids is None and ids is not None and experts < 64):
        _launch_pair(out, x, a, b, flat, scale, weights, z_out, activated, row_div, experts, top_k)
    else:
        if sorted_token_ids is None:
            if ids is None:
                align = _dense_alignment(p, x.device, _DENSE_BLOCK_M)
            else:
                if _capturing():
                    raise RuntimeError(
                        "pass the base alignment (sorted_token_ids, expert_ids, "
                        "num_tokens_post_padded, block_m) to MoE LoRA under CUDA graph capture"
                    )
                block_m = 16 if p <= 1024 else 64
                ids32 = torch.where((flat >= 0) & (flat < experts), flat, experts).to(torch.int32)
                align = (*moe_align_block_size(ids32.view(ids.shape), block_m, experts), block_m)
        else:
            align = (sorted_token_ids, expert_ids, num_tokens_post_padded, block_m)
        _launch_block(
            out, x, a, b, scale, weights, z_out, activated, row_div, align, experts, top_k
        )
    return out if activated is None else activated


def shared_rows(x, w_gate_up, w_down, w_gate, factors, scale, cfg):
    """The shared expert as vLLM 0.31 serves it, its dense MLP with punica LoRA: (shared [T, H]
    in x's dtype, gate logits [T, 1] FP32 or None without ``w_gate``). The gate/up and down
    GEMMs on the grouped GEMM as one dense expert (FP32 accumulation, rounded once); each LoRA
    delta added as the punica kernels add it, out = BF16(out + BF16(BF16(x . A^T) . B_s^T))
    with B_s = BF16(B * scale) (vLLM folds the scaling into B at load; formed in registers);
    SiLU * up in FP32 rounded once (SiluAndMul under Inductor; fused into the gate/up
    expand). ``factors`` is (a13, b13, a2, b2) or None; ``cfg`` the base tile config."""
    t, h = x.shape
    inter = w_down.shape[1]
    cache, _, floats = scratch(x, t, (inter, h), _shared_size, "shared")
    gate_up = cache[: t * 2 * inter].view(t, 2 * inter)
    act = cache[t * 2 * inter : t * 3 * inter].view(t, inter)
    out = cache[t * 3 * inter : t * (3 * inter + h)].view(t, h)
    gate = floats[:t].view(t, 1)
    sorted_ids, experts, padded, _ = _dense_alignment(t, x.device, cfg["BLOCK_SIZE_M"])
    ids = _dense_ids(t, x.device).view(t, 1)
    lora_align = _dense_alignment(t, x.device, _DENSE_BLOCK_M)
    for a, w, c, lora in ((x, w_gate_up, gate_up, 0), (act, w_down, out, 2)):
        fused_moe_kernel_triton(
            a, w[None], c.view(t, 1, -1), gate, ids, sorted_ids, experts, padded, False, 1, cfg,
            x.dtype,
        )  # fmt: skip
        activated = act if c is gate_up else None
        if factors is not None:
            # with ``activated`` the expand writes SiLU * up of the rounded gate/up sums
            _launch_block(
                c, a, factors[lora], factors[lora + 1], scale, None, None, activated, 1,
                lora_align, 1, 1, round_delta=True, prescale=True,
            )  # fmt: skip
        elif activated is not None:
            silu_and_mul(gate_up, out=act)
    if w_gate is None:
        gate = None
    else:
        K.gate_logits_kernel[(t,)](x, w_gate, gate, h, x.stride(0), BK=1024, num_warps=4)
    return out, gate


def _shared_size(max_rows, inter, h):
    return max_rows * (3 * inter + h), 0, max_rows


def _sentinel_map(experts, device):
    """Identity map over local experts with -1 for the sentinel id ``experts``."""

    def build():
        return torch.cat((torch.arange(experts), torch.tensor([-1]))).int().to(device)

    return _static(("sentinel-map", experts, device), build)


def _dense_ids(p, device):
    """Expert 0 for every dense shared row."""
    return _static(
        ("dense", p, device), lambda: torch.zeros((p,), dtype=torch.int32, device=device)
    )


def _dense_alignment(p, device, block_m=64):
    """Dense rows as consecutive ``block_m``-row blocks of expert 0."""

    def build():
        blocks = triton.cdiv(p, block_m)
        sorted_ids = torch.arange(blocks * block_m, dtype=torch.int32, device=device)
        sorted_ids[p:] = p
        experts = torch.zeros((blocks,), dtype=torch.int32, device=device)
        padded = torch.tensor([blocks * block_m], dtype=torch.int32, device=device)
        return sorted_ids, experts, padded, block_m

    return _static(("dense-align", p, block_m, device), build)


@torch.no_grad()
def fused_experts_lora(
    x,
    w13,
    w2,
    topk_weights,
    topk_ids_local,
    lora,
    *,
    scaling,
    base_config,
    local_only=False,
    return_state=False,
    w13_scale=None,
    w2_scale=None,
    shared_expert=None,
    expert_config=None,
    shared=None,
    shared_gate=None,
):
    """Routed MLP with per-expert LoRA (inputs validated by ``moe.lora``).

    BF16 base: exactly ``moe.inference.fused_experts`` (its tiles for these E local experts
    and ``expert_config``, BF16 FC1 cache, SiLU*up, FP32 weighted routes, the top-k sum),
    each LoRA delta added into its cache; zero factors give its bits.
    Block-FP8 base: as below.

    Base grouped GEMMs (per-pair blocks while P * 4 <= E, else aligned), LoRA gate/up fused
    with SwiGLU, the weighted down LoRA into the route cache, then the top-k reduction.
    EP: routes whose local id is outside [0, E) (-1 for another rank's expert) cost no GEMM
    or LoRA work and add exactly nothing: they are sorted into no-work blocks and masked out
    of the reduction. Routes the caller already parked on expert 0 at weight 0 stay correct.
    ``return_state`` keeps c1 (after LoRA) and the unscaled shrinks z13/z2 for a backward.
    ``base_config`` is the base grouped-GEMM tile config for this batch.
    BF16 inference keeps the FC1 rows and the weighted FC2 routes in FP32 (the LoRA deltas are
    added there): one rounding for the SwiGLU product, one for the output, as the MoE block
    tree. ``shared_expert`` = (w_gate_up, w_down, w_gate, factors or None) joins that FP32 sum
    as sigmoid(x . w_gate) * shared(x), computed unrounded (fused into the decode kernels).
    Block-FP8 base (``w13_scale``/``w2_scale`` given, E4M3 weights): x is quantized 1x128 per
    token, the gate/up LoRA is added to the BF16 FC1 cache, then one fused SiLU*up + 1x128
    quantize feeds FC2 and also emits the BF16 product the down LoRA reads (the LoRA branch
    stays BF16). Inference on either base takes the caller's shared expert output ``shared``
    [T, H] and gate logits ``shared_gate`` [T, 1] (x's dtype) into the combine, joined in vLLM's
    order as ``moe_sum_reduce_triton`` joins them (block-FP8 decode: inside the fused
    down-expand combine).
    """
    t, h = x.shape
    e, twice_i, _ = w13.shape
    inter = twice_i // 2
    top_k = topk_ids_local.shape[1]
    p = t * top_k
    out = torch.empty_like(x)
    if t == 0:
        return (out, {"c1": x.new_empty((0, twice_i))}) if return_state else out
    rank = 0 if lora is None else lora.rank
    scale = scale_tensor(scaling, x.device) if lora is not None else None
    pair = p <= _PAIR_ROWS
    # Per-pair base blocks only while experts rarely repeat; aligned blocks read each
    # expert's weights once (Qwen3-30B-A3B T=4 on GB10: per-pair blocks lost ~5%).
    naive = p * 4 <= e
    cfg = base_config
    bm = cfg["BLOCK_SIZE_M"]
    fp8 = w13_scale is not None
    base = "fp8_block" if fp8 else "bf16"
    if not fp8:
        if return_state:
            raise ValueError("BF16-base training runs moe.triton._lora_train")
        return _bf16_experts(
            out, x, w13, w2, topk_weights, topk_ids_local, lora, scale, local_only,
            shared_expert, expert_config, shared, shared_gate,
        )  # fmt: skip
    group = GraphWorkspace.active()
    if group is not None and x.is_cuda and not (return_state or _capturing()):
        _reserve_reachable(group, x, e, top_k, inter, lora, base, shared_expert)
    # BF16: the FC1 / FC2 route rows (c13) in FP32 (the training state too: c1 and the shrinks
    # in FP32). FP8 (bitwise the block-FP8 MoE) keeps them in the cache dtype and adds the
    # E4M3 FC2 input (inter bytes = inter // 2 cache elements) and its scales.
    wide = not fp8
    c13_size = max(twice_i, h)
    width = inter + (0 if wide else c13_size) + (inter // 2 if fp8 else 0)
    groups = -(-inter // 128) if fp8 else 0
    floats_per_route = (c13_size if wide else 0) + groups
    cache, ints, floats = scratch(
        x, t, (top_k, width, 1, floats_per_route), _composition_size, "composition"
    )
    c13 = (floats if wide else cache)[: p * c13_size]
    c1 = c13[: p * twice_i].view(p, twice_i)
    c3 = c13[: p * h].view(t, top_k, h)
    c2_at = 0 if wide else p * c13_size
    c2 = cache[c2_at : c2_at + p * inter].view(p, inter)
    if fp8:
        fp8_args = {"block_fp8_provider": block_fp8_moe_provider(x.device)}
        q2 = cache[p * (c13_size + inter) : p * width].view(torch.float8_e4m3fn)
        q2 = q2.view(p, inter)
        s2 = floats[: p * groups].view(p, groups)
    if return_state:
        rows = torch.float32 if wide else x.dtype
        c1 = torch.empty((p, twice_i), dtype=rows, device=x.device)
        c3 = torch.empty((t, top_k, h), dtype=rows, device=x.device)
        z13 = torch.zeros((p, 1, rank), dtype=rows, device=x.device)  # training: shared A
        z2 = torch.zeros((p, 1, rank), dtype=rows, device=x.device)
    rw = topk_weights.contiguous()
    # Fused decode only while experts rarely repeat (it reads each route's weights) and the
    # inline LoRA shrink is cheap (measured on GB10: r64 and T=8 EP2 rows ran faster aligned).
    # Its kernels skip ids outside [0, E) themselves: no route preparation launch.
    # Block-FP8 bases keep the W8A8 MMA path (a GB10 E4M3-decoding GEMV measured slower).
    if naive and rank <= _DECODE_RANK and not (fp8 or return_state):
        ids = topk_ids_local if topk_ids_local.is_contiguous() else topk_ids_local.contiguous()
        _fused_decode(out, x, w13, w2, rw, ids, lora, scale, c2, e, top_k, shared_expert)
        return out
    if shared_expert is not None:
        sgu, sd, sg, factors = shared_expert
        shared, shared_gate = shared_rows(x, sgu, sd, sg, factors, scale, cfg)
    exact = local_only and topk_ids_local.dtype == torch.int32
    if exact and topk_ids_local.is_contiguous():
        ids = topk_ids_local
    else:
        # Per-pair blocks of expert -1 do no work; the alignment sorts the sentinel E into
        # trailing no-work blocks. Either way the masked reduction skips the route.
        ids = ints[:p].view(t, top_k)
        K.route_prep_kernel[(triton.cdiv(p, 1024),)](
            topk_ids_local, ids, p, e, SENTINEL=-1 if naive else e, BLOCK=1024
        )
    if naive:
        sorted_ids, padded = _naive_alignment(p, bm, x.device)
        experts = ids.view(-1)
    else:
        sorted_ids, experts, padded = moe_align_block_size(ids, bm, e)
    a13, b13 = _routed(lora, down=False)
    a2, b2 = _routed(lora, down=True)
    flat = ids.view(-1)
    if pair and lora is not None and not return_state:
        fp8_base = None
        if fp8:
            fp8_base = (w13_scale, w2_scale, q2, s2, fp8_args)
        _overlapped_decode(
            out, x, w13, w2, rw, ids, flat, sorted_ids, experts, padded, cfg, c1, c2, c3,
            (a13, b13, a2, b2), scale, e, top_k, base, fp8_base, shared, shared_gate,
        )  # fmt: skip
        return out
    if fp8:
        q1, s1 = quantize_per_token_group_fp8(x)
        fused_moe_kernel_triton(
            q1, w13, c1.view(t, top_k, twice_i), rw, ids, sorted_ids, experts, padded, False,
            top_k, cfg, x.dtype, a_scale=s1, b_scale=w13_scale, **fp8_args,
        )  # fmt: skip
    else:
        fused_moe_kernel_triton(
            x, w13, c1.view(t, top_k, twice_i), rw, ids, sorted_ids, experts, padded, False,
            top_k, cfg, x.dtype,
        )  # fmt: skip
    align = None if pair else (sorted_ids, experts, padded, bm)
    if lora is not None and (fp8 or return_state):
        z = z13 if return_state else None
        _project(c1, x, a13, b13, flat, scale, None, z, None, top_k, align, e, top_k, base)
    if fp8:
        silu_and_mul_quantize_fp8(c1, outputs=(q2, s2), act=c2)
        # Inference: unweighted BF16 rows (vLLM); training (the STE state) keeps weighted rows.
        fused_moe_kernel_triton(
            q2, w2, c3, rw, ids, sorted_ids, experts, padded, return_state, 1, cfg, x.dtype,
            a_scale=s2, b_scale=w2_scale, **fp8_args,
        )  # fmt: skip
    else:
        if wide and (lora is None or return_state):
            K.swiglu_kernel[(p, triton.cdiv(inter, 1024))](
                c1, c2, flat, p, e, inter, BN=1024, num_warps=4
            )
        elif lora is None or return_state:
            silu_and_mul(c1, out=c2)
        else:
            _project(c1, x, a13, b13, flat, scale, None, None, c2, top_k, align, e, top_k, base)
        fused_moe_kernel_triton(
            c2, w2, c3, rw, ids, sorted_ids, experts, padded, True, 1, cfg, x.dtype,
        )  # fmt: skip
    if lora is not None:
        _project(
            c3.view(p, h), c2, a2, b2, flat, scale, rw.reshape(-1) if return_state else None,
            z2 if return_state else None, None, 1, align, e, top_k, base,
        )  # fmt: skip
    joined = {"shared": shared, "shared_gate": shared_gate}  # inference: vLLM's shared join
    if return_state:
        if exact:
            moe_sum_reduce_triton(c3, out)
        else:
            moe_sum_reduce_triton(c3, out, ids, _sentinel_map(e, x.device))
    elif exact:
        moe_sum_reduce_triton(c3, out, ids, topk_weights=rw, **joined)
    else:
        moe_sum_reduce_triton(
            c3, out, ids, _sentinel_map(e, x.device), topk_weights=rw, **joined
        )  # fmt: skip
    if return_state:
        state = {
            "c1": c1,
            "z13": z13,
            "z2": z2,
            "ids": ids.clone(),
            "rw": rw,
            "alignment": layout(topk_ids_local, e),
            "base_alignment": (sorted_ids, experts.clone() if naive else experts, padded),
            "config": cfg,
            "base_skipped": True,  # remote rows of the base dX are zeroed, never reduced stale
            "local_only": local_only,
        }
        if fp8:
            state.update(w13_scale=w13_scale, w2_scale=w2_scale)
        return out, state
    return out


def _bf16_experts(
    out, x, w13, w2, topk_weights, topk_ids_local, lora, scale, local_only, shared_expert,
    expert_config, shared=None, shared_gate=None,
):  # fmt: skip
    """BF16 base as ``fused_experts`` runs it (same tiles, alignment, kernels and rounding
    points: BF16 FC1 cache, SiLU*up, unweighted BF16 FC2 routes combined as vLLM's finalize),
    the gate/up LoRA delta added into the FC1 cache before SiLU*up and the down delta into
    each unweighted BF16 route row (one rounding) before the combine."""
    t, h = x.shape
    e, twice_i, _ = w13.shape
    inter = twice_i // 2
    top_k = topk_ids_local.shape[1]
    p = t * top_k
    up, down = expert_tiles(t, e, inter, e, x.device, expert_config)
    group = GraphWorkspace.active()
    if group is not None and x.is_cuda and not _capturing():
        _reserve_reachable(group, x, e, top_k, inter, lora, "bf16", shared_expert)
    width = twice_i + inter + h
    cache, ints, _ = scratch(x, t, (top_k, width, 1, 0), _composition_size, "composition")
    c1 = cache[: p * twice_i].view(p, twice_i)
    c2 = cache[p * twice_i : p * (twice_i + inter)].view(p, inter)
    c3 = cache[p * (twice_i + inter) : p * width].view(t, top_k, h)
    rw = topk_weights.contiguous()
    if shared_expert is not None:  # else the caller's shared/shared_gate, or none
        sgu, sd, sg, factors = shared_expert
        shared, shared_gate = shared_rows(x, sgu, sd, sg, factors, scale, up)
        # fused_experts' shared input: the shared output and gate logits in the output dtype
        shared, shared_gate = shared.to(x.dtype), shared_gate.to(x.dtype)
    exact = local_only and topk_ids_local.dtype == torch.int32
    if exact and topk_ids_local.is_contiguous():
        ids = topk_ids_local
    else:  # remote (-1) routes onto the sentinel E: no-work blocks, masked out of the sum
        ids = ints[:p].view(t, top_k)
        K.route_prep_kernel[(triton.cdiv(p, 1024),)](
            topk_ids_local, ids, p, e, SENTINEL=e, BLOCK=1024
        )
    aligned = moe_align_block_size(ids, up["BLOCK_SIZE_M"], e)
    aligned_down = (
        aligned
        if up["BLOCK_SIZE_M"] == down["BLOCK_SIZE_M"]
        else moe_align_block_size(ids, down["BLOCK_SIZE_M"], e)
    )
    flat = ids.view(-1)
    pair = p <= _PAIR_ROWS
    fused_moe_kernel_triton(
        x, w13, c1.view(t, top_k, twice_i), rw, ids, *aligned, False, top_k, up, x.dtype
    )  # fmt: skip
    if lora is not None:
        a13, b13 = _routed(lora, down=False)
        align = None if pair else (*aligned, up["BLOCK_SIZE_M"])
        _project(c1, x, a13, b13, flat, scale, None, None, None, top_k, align, e, top_k)
    silu_and_mul(c1, out=c2)
    fused_moe_kernel_triton(c2, w2, c3, rw, ids, *aligned_down, False, 1, down, x.dtype)
    if lora is not None:
        a2, b2 = _routed(lora, down=True)
        align = None if pair else (*aligned_down, down["BLOCK_SIZE_M"])
        _project(c3.view(p, h), c2, a2, b2, flat, scale, None, None, None, 1, align, e, top_k)
    combine = {"topk_weights": rw, "shared": shared, "shared_gate": shared_gate}
    if exact:
        moe_sum_reduce_triton(c3, out, ids, **combine)
    else:
        moe_sum_reduce_triton(c3, out, ids, _sentinel_map(e, x.device), **combine)
    return out


def _fused_decode(out, x, w13, w2, rw, ids, lora, scale, c2, e, top_k, shared_expert):
    """Decode (BF16 base): FC1 per (pair, I tile) with SwiGLU and the gate/up LoRA; FC2 per
    (token, H tile) with the down LoRA and the slot sum. The shared expert, if any, is FC1's
    extra work and FC2's gated extra route. With LoRA the shrinks run once per row: a split-K
    shrink of x before FC1 (one launch for routes and shared rows each) and FC1's per-tile
    shares of the down shrink, summed in FC2 (a per-tile shrink would be redone by every N
    tile). FP32 throughout, one rounding per output; deterministic."""
    t, h = x.shape
    p, inter = ids.numel(), w2.shape[-1]
    sms = _sms(x.device)
    has_lora = lora is not None
    if has_lora:
        a13, b13 = _routed(lora, down=False)
        a2, b2 = _routed(lora, down=True)
        rank = lora.rank
        lora13 = (a13[0], b13[0], a13.gap, b13.gap)
        lora2 = (a2[0], b2[0])
    else:
        rank = 1
        lora13 = (x, x, 0, 0)
        lora2 = (x, x)
    sc = scale if has_lora else rw
    flat = ids.reshape(-1)
    has_shared = shared_expert is not None
    shared_lora = False
    if has_shared:
        sgu, sd, sg, factors = shared_expert
        s_inter = sd.shape[1]
        sc2 = scratch(x, t, (s_inter, h), _shared_size, "shared")[0][: t * s_inter]
        shared_lora = factors is not None
        if shared_lora:
            sa13, sb13, sa2, sb2 = factors
            shared13 = (sa13[0], sb13[0], sa13.gap, sb13.gap)
            shared2 = (sa2[0], sb2[0])
        else:
            shared13, shared2 = (x, x, 0, 0), (x, x)
    else:
        sgu = sd = sg = sc2 = x
        s_inter = 0
        shared13, shared2 = (x, x, 0, 0), (x, x)
    pdl = _pdl(x.device)
    meta = {"TOP_K": top_k, "R": rank, "BR": _pow2(rank), "HAS_LORA": has_lora,
            "HAS_SHARED": has_shared, "SHARED_LORA": shared_lora, "PDL": pdl,
            "EVF": _DECODE_EVICT_FIRST}  # fmt: skip
    cfg = _config(x, p, e, "decode", h, inter, max(rank, 1), 2, "bf16")
    bn1, bk1, warps = cfg["decode_fc1_bn"], cfg["decode_fc1_bk"], cfg["decode_warps"]
    tiles1 = triton.cdiv(inter, bn1)
    work1 = p * tiles1
    s_tiles = triton.cdiv(s_inter, bn1)
    pre = has_lora or shared_lora
    meta["PRE_Z"] = pre
    zp = z2 = szp = sz2 = sc
    ks = s_ks = 1
    if pre:
        z = scratch(x, t, (top_k, inter, s_inter, rank), _decode_z_size, "decode-z")[2]
        z2 = z[: p * tiles1 * rank]
        sz2 = z[p * tiles1 * rank : p * tiles1 * rank + t * s_tiles * rank]
    one_a = has_lora and _one_a(a13) is not None
    if has_lora:
        pc = _config(x, p, e, "w13_act", h, inter, rank, 2, "bf16")
        zp = _pair_partials(x, _one_a(a13) or a13, flat, top_k, e, top_k, "w13_act", inter,
                            "bf16", pc, p, pdl=pdl and not shared_lora)  # fmt: skip
        ks = _split_count(pc, h)
    if shared_lora:
        dense = _dense_ids(t, x.device)
        sp = _config(x, t, 1, "w13_act", h, s_inter, rank, 2, "bf16")
        szp = _pair_partials(x, sa13, dense, 1, 1, 1, "w13_act", s_inter, "bf16", sp, t,
                             role="partials-shared", pdl=pdl)  # fmt: skip
        s_ks = _split_count(sp, h)
    K.decode_fc1_kernel[(min(work1 + t * s_tiles, 4 * sms),)](
        x, w13, flat, lora13[0], lora13[1], sc, c2, p, e, h, inter, lora13[2], lora13[3],
        x.stride(0), work1, tiles1, sgu, shared13[0], shared13[1], sc2, t, s_inter, s_tiles,
        shared13[2], shared13[3], zp, ks, z2, lora2[0], szp, s_ks, sz2, shared2[0],
        **meta, WAIT_Z=pre and pdl, ONE_A=one_a, BN=bn1, BK=bk1, num_warps=warps,
        launch_pdl=pre and pdl,
    )  # fmt: skip
    bn2, bk2 = cfg["decode_fc2_bn"], cfg["decode_fc2_bk"]
    tiles2 = triton.cdiv(h, bn2)
    work2 = t * tiles2
    K.decode_fc2_kernel[(min(work2, 4 * sms),)](
        c2, w2, flat, rw, lora2[0], lora2[1], sc, out, t, e, h, inter, work2, tiles2, x,
        x.stride(0), sg, sc2, sd, shared2[0], shared2[1], s_inter, z2, tiles1, sz2, s_tiles,
        **meta, BN=bn2, BK=bk2, num_warps=warps, launch_pdl=pdl,
    )  # fmt: skip


def _decode_z_size(max_rows, top_k, inter, s_inter, rank):
    """FC1's per-tile down-shrink shares (routes, then shared rows), at the narrowest tile."""
    return 0, 0, max_rows * (top_k * triton.cdiv(inter, 8) + triton.cdiv(s_inter, 8)) * rank


_DECODE_RANK = 32  # largest rank whose LoRA shrink runs in the decode kernels
_DECODE_EVICT_FIRST = True  # decode weights are read once per call: evict-first keeps L2 for the rest (PRO, EP2 T=1 r16: 32.2 vs 36.6 us)


def _pdl(device):
    """Programmatic dependent launch between the two decode kernels (compute capability 9+)."""
    return device_info(device)[0] >= (9, 0)


def _fork(main, side):
    event = torch.cuda.Event()
    event.record(main)
    side.wait_event(event)


def _overlapped_decode(
    out, x, w13, w2, rw, ids, flat, sorted_ids, experts, padded, cfg, c1, c2, c3, factors,
    scale, e, top_k, base, fp8_base=None, shared=None, shared_gate=None,
):  # fmt: skip
    """Decode composition with the LoRA shrinks off the critical path: each split-K shrink
    runs on a side stream while its base grouped GEMM runs; the gate/up expand fuses SwiGLU
    (block-FP8 base: adds into the FC1 cache, then the fused SiLU*up + 1x128 quantize) and
    the down expand is fused into the top-k combine (unweighted BF16 rows, vLLM's finalize).
    Only stream order changes, so the
    result is bitwise the same as running the same launches serially."""
    a13, b13, a2, b2 = factors
    t, h = x.shape
    p, inter, twice_i = flat.numel(), c2.shape[1], c1.shape[1]
    main, side = torch.cuda.current_stream(x.device), overlap_stream(x.device)
    kind13 = "w13" if fp8_base else "w13_act"
    cfg13 = _config(x, p, e, kind13, a13.shape[-1], inter, a13.shape[-2], 2, base)
    cfg2 = _config(x, p, e, "w2", a2.shape[-1], h, a2.shape[-2], 1, base)
    # Partials are allocated on the main stream: their lifetime follows its order.
    shrink13 = _one_a(a13) or a13
    part13 = _partials_buffer(x, shrink13, e, top_k, kind13, inter, base, p)
    part2 = _partials_buffer(x, a2, e, top_k, "w2", h, base, p)
    a1, gemm1, gemm2, a2_in = x, {}, {}, c2
    if fp8_base:
        s13, s2w, q2, qs2, fp8_args = fp8_base
        a1, qs1 = quantize_per_token_group_fp8(x)
        gemm1 = {"a_scale": qs1, "b_scale": s13, **fp8_args}
        gemm2 = {"a_scale": qs2, "b_scale": s2w, **fp8_args}
        a2_in = q2
    _fork(main, side)
    with torch.cuda.stream(side):
        _pair_partials(x, shrink13, flat, top_k, e, top_k, kind13, inter, base, cfg13, p, part13)
    fused_moe_kernel_triton(
        a1, w13, c1.view(t, top_k, twice_i), rw, ids, sorted_ids, experts, padded, False,
        top_k, cfg, x.dtype, **gemm1,
    )  # fmt: skip
    _fork(side, main)
    splits13 = _split_count(cfg13, a13.shape[-1])
    act = None if fp8_base else c2
    _pair_expand(c1, x, a13, b13, flat, scale, None, None, act, top_k, e, cfg13, part13, splits13)
    if fp8_base:
        silu_and_mul_quantize_fp8(c1, outputs=(q2, qs2), act=c2)
    _fork(main, side)
    with torch.cuda.stream(side):
        _pair_partials(c2, a2, flat, 1, e, top_k, "w2", h, base, cfg2, p, part2)
    fused_moe_kernel_triton(
        a2_in, w2, c3, rw, ids, sorted_ids, experts, padded, False, 1, cfg, x.dtype, **gemm2,
    )  # fmt: skip
    _fork(side, main)
    rank = a2.shape[-2]
    bn = 128
    has_shared = shared is not None
    shared_ptr, gate_ptr = (shared, shared_gate) if has_shared else (out, out)
    K.reduce_expand_kernel[(t, triton.cdiv(h, bn))](
        c3, out, part2, b2[0], flat, rw, scale, t, e, h, _split_count(cfg2, a2.shape[-1]),
        shared_ptr, gate_ptr, shared_ptr.stride(0), gate_ptr.stride(0),
        TOP_K=top_k, R=rank, BR=_pow2(rank), BN=bn, SHARED=has_shared, num_warps=4,
    )  # fmt: skip


def _routed(lora, down):
    if lora is None:
        return None, None
    f = lora.routed
    if down:
        return FactorSlices(f[4:5]), FactorSlices(f[5:6])
    return FactorSlices(f[0:2]), FactorSlices(f[2:4])


def _project(
    out, x, a, b, flat, scale, weights, z_out, activated, row_div, align, experts, top_k,
    base="bf16",
):  # fmt: skip
    args = (out, x, a, b)
    tail = (z_out, activated, row_div)
    if align is None:
        _launch_pair(*args, flat, scale, weights, *tail, experts, top_k, base)
    else:
        _launch_block(*args, scale, weights, *tail, align, experts, top_k, base)


def project_rounded(out, x, a, b, scale, z_out, row_div, align, experts, top_k):
    """Training: PEFT's BF16 LoRA on the aligned rows, z_out = BF16(x . A^T) [P, S, r] (kept
    for the backward), then out = BF16(out + BF16(BF16(z . B^T) * scale)) in place."""
    _launch_block(
        out, x, a, b, scale, None, z_out, None, row_div, align, experts, top_k, round_delta=True
    )
