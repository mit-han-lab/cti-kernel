# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Metadata dispatch and the compilation cache; CuTe kernel modules bind on first use."""

from functools import lru_cache
from types import SimpleNamespace

import torch

from ...moe_impl import moe_align_block_size
from .._lora_common import LOCK, available, device_info, scale_tensor

_COMPILED = {}
# Optional block-kernel tuning; values participate in the compilation cache key.
_BLOCK_OVERRIDES = {}

SPLIT_MAX_PAIRS = 1024
SPLIT_MAX_PAIRS_LARGE = 4096
SPLIT_MIN_SMS_LARGE = 100
SPLIT_MIN_RANK_LARGE = 32
SPLIT_CHUNKS = 4


def split_chunks(pairs, rank, sms):
    """N chunks for separate routed shrink/expand, or zero for a fused block.

    Concentrated routes leave few row blocks: on RTX 5090, Qwen3.5 rank 64,
    T=19, splitting improved vLLM/ours from 1.026 to 1.244 (narrow), 1.115 to
    1.318 (skewed), and 1.078 to 1.093 (softmax), but lost about 5% at T=2113
    (cold weights, CUDA graphs). Keep the provisional bounds above together
    for tuning; pairs counts expected local routes, including EP adjustment.
    """
    if pairs <= SPLIT_MAX_PAIRS or (
        sms >= SPLIT_MIN_SMS_LARGE
        and rank >= SPLIT_MIN_RANK_LARGE
        and pairs <= SPLIT_MAX_PAIRS_LARGE
    ):
        return SPLIT_CHUNKS
    return 0


def validate_chunks(chunks, *, stage, shared):
    """Explicit N chunking belongs only to routed expand stages."""
    if chunks is None:
        return
    if type(chunks) is not int or chunks <= 0:
        raise ValueError("chunks must be a positive integer")
    if stage != "expand" or shared:
        raise ValueError("chunks requires a routed expand stage")


def block_size(pairs, experts, rank=16):
    """Row tile from metadata only (measured on sm_120): 16-row blocks win almost
    everywhere; 32 for ranks >= 32 once experts average >= 64 rows (rank 128
    since register shrink accumulators: RTX 5090 T=32768 LoRA 8.0 ms vs 9.0 ms)."""
    return 32 if rank >= 32 and pairs >= 64 * experts else 16


def shared_block_size(rows):
    """Dense shared rows: 16-row blocks keep parallelism; 32 once rows are plentiful
    (measured on sm_120; 64-row blocks were slower at every size)."""
    return 16 if rows <= 4096 else 32


def uses_block_path(pairs, rank=128, sms=None, shared=False):
    """Shared threshold for automatic prefill dispatch and layer compositions.

    Measured crossovers: the clustered pair kernel wins up to 512 pairs at rank <= 16
    (RTX 5090, T=64 top-8: LoRA 29 us vs 47 us for aligned blocks), and on small-SM,
    bandwidth-poor parts up to 1024 pairs at rank <= 16 and 128 pairs above (GB10,
    256 pairs at rank 64: blocks 786 us vs pairs 885 us per graph sample);
    aligned blocks win beyond.
    """
    if shared:
        # One factor set serves every row: per-token pair CTAs would each re-read it.
        return pairs > 16
    small = sms is not None and sms < 64
    if rank <= 16:
        return pairs > (1024 if small else 512)
    return pairs > 128


def supports_rank(rank):
    """CuTe ranks: multiples of 8 in [8, 128] (ML6 widens this)."""
    return 8 <= rank <= 128 and rank % 8 == 0


def configuration(
    pairs, slices, sms, k, n, *, aligned=False, block_m=32, shared=False, rank=32, force_block=False
):
    if (force_block or uses_block_path(pairs, rank, sms, shared)) and (aligned or shared):
        # Routed: a CTA owns every slice and the whole N range, so each block's
        # shrink runs once. Splitting N (recomputing the shrink per chunk) measured
        # slower or equal at every batch and rank on sm_120, much slower on sm_121.
        if not shared:
            return 1, block_m, 1
        # Shared rows reuse one L2-resident factor set, so recomputing the shrink per
        # N chunk is cheap while dense row blocks alone leave most SMs idle.
        # Split only while the row blocks fill less than one wave.
        blocks = max(1, (pairs + block_m - 1) // block_m)
        if blocks >= sms:
            return 1, block_m, 1
        bn, _ = block_pipeline(block_m, rank)
        chunks = max(1, min((2 * sms + blocks - 1) // blocks, (n + bn - 1) // bn))
        return 1, block_m, chunks
    cluster = 1
    while cluster < 8 and pairs * slices * cluster < 2 * sms and cluster * 2 <= k // 8:
        cluster *= 2
    return cluster, 0, 1


def block_pipeline(block_m, rank):
    """Conservative defaults keep common small-rank CTAs below 48 KiB.

    Wider tiles amortize metadata for BM16; BM64 retains BN64 to bound the
    live expand accumulator. Explicit compilation can override both choices.
    """
    return (128 if block_m == 16 and rank <= 16 else 64, 4 if block_m <= 32 and rank <= 16 else 3)


def compile_kernel(
    k,
    n,
    rank,
    slices,
    top_k,
    experts,
    capability,
    *,
    down=False,
    shared=False,
    cluster=1,
    block_m=0,
    chunks=1,
    z_dtype=None,
    id_dtype=torch.int32,
    block_n=64,
    stages=3,
    block_k=64,
    shrink_stages=None,
    act=False,
    stage="full",
    weighted=True,
):
    """Compile without a GPU. All routing sizes are symbolic, shape dimensions static."""
    if act and (not block_m or down or slices != 2):
        raise ValueError("act requires a gate/up block kernel")
    if stage not in ("full", "shrink", "expand"):
        raise ValueError("stage must be full, shrink, or expand")
    if stage == "expand" and (not block_m or z_dtype != torch.bfloat16):
        raise ValueError("an expand stage requires a block kernel and BF16 z buffer")
    if stage == "shrink" and (z_dtype is None or block_m and z_dtype != torch.bfloat16):
        raise ValueError("a shrink stage writes z: FP32/BF16 (pair) or BF16 (block)")
    return kernels().pair.compile_kernel(
        k,
        n,
        rank,
        slices,
        top_k,
        experts,
        capability,
        down=down,
        shared=shared,
        cluster=cluster,
        block_m=block_m,
        chunks=chunks,
        z_dtype=z_dtype,
        id_dtype=id_dtype,
        block_n=block_n,
        stages=stages,
        block_k=block_k,
        shrink_stages=shrink_stages,
        act=act,
        stage=stage,
        weighted=weighted,
    )


@lru_cache(maxsize=None)
def kernels():
    """The CuTe kernel modules, bound on first CUDA use: they import cutlass at module level."""
    from . import _lora_block, _lora_dense, _lora_pair

    return SimpleNamespace(block=_lora_block, dense=_lora_dense, pair=_lora_pair)


@lru_cache(maxsize=None)
def _block_k128_fits_cached(args, items):
    kw = dict(items)
    k, n, rank, slices, top_k, experts, _ = args
    try:
        kernels().block.BlockKernel(
            k,
            n,
            rank,
            slices,
            top_k,
            experts,
            down=kw["down"],
            shared=kw["shared"],
            block_m=kw["block_m"],
            chunks=kw["chunks"],
            save_z=kw["z_dtype"] is not None or kw["stage"] != "full",
            block_n=kw["block_n"],
            stages=kw["stages"],
            block_k=128,
            act=kw["act"],
            stage=kw["stage"],
        )
    except ValueError:  # shared-memory guard: keep 64-wide K tiles
        return False
    return True


def _block_k128_fits(args, kwargs):
    keys = ("down", "shared", "block_m", "chunks", "z_dtype", "stage", "block_n", "stages", "act")
    return _block_k128_fits_cached(args, tuple((key, kwargs[key]) for key in keys))


def apply_(
    out,
    x,
    a,
    b,
    ids,
    scale,
    top_k,
    routed_weights=None,
    z_out=None,
    *,
    down=False,
    sorted_token_ids=None,
    expert_ids=None,
    num_tokens_post_padded=None,
    block_m=None,
    activated=None,
    stage="full",
    chunks=None,
):
    validate_chunks(chunks, stage=stage, shared=ids is None)
    if stage not in ("full", "shrink", "expand"):
        raise ValueError("stage must be full, shrink, or expand")
    if stage != "full" and (z_out is None or z_out.dtype != torch.bfloat16):
        raise ValueError("split stages require a BF16 z buffer")
    act = activated is not None
    capability, sms = device_info(x.device)
    if capability not in ((12, 0), (12, 1)) or not available():
        raise RuntimeError("cute requires sm_120/sm_121 and nvidia-cutlass-dsl 4.8.0")
    if out.shape[0] == 0:
        return activated if act else out
    aligned = sorted_token_ids is not None and ids is not None
    # The fused activation and split stages exist only as block kernels.
    force = act or stage != "full"
    c, bm, configured_chunks = configuration(
        out.shape[0],
        a.shape[1],
        sms,
        x.shape[1],
        b.shape[-2],
        aligned=aligned,
        block_m=block_m
        or (
            shared_block_size(out.shape[0])
            if ids is None
            else block_size(out.shape[0], a.shape[0], a.shape[-2])
        ),
        shared=ids is None,
        rank=a.shape[-2],
        force_block=force,
    )
    # Standalone prefill can build alignment; decode never allocates routing storage.
    if (
        (force or uses_block_path(out.shape[0], a.shape[-2], sms))
        and not aligned
        and ids is not None
    ):
        bm = block_m or block_size(out.shape[0], a.shape[0], a.shape[-2])
        safe_ids = torch.where((ids >= 0) & (ids < a.shape[0]), ids, 0).to(torch.int32)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            safe_ids, bm, a.shape[0]
        )
        c, bm, configured_chunks = configuration(
            out.shape[0],
            a.shape[1],
            sms,
            x.shape[1],
            b.shape[-2],
            aligned=True,
            block_m=bm,
            rank=a.shape[-2],
            force_block=force,
        )
    chunks = configured_chunks if chunks is None else chunks
    args = (x.shape[1], b.shape[-2], a.shape[-2], a.shape[1], top_k, a.shape[0], capability)
    if stage != "full" and not bm:
        raise ValueError("split stages require block alignment")
    kwargs = {
        "stage": stage,
        "act": act,
        "down": down,
        "shared": ids is None,
        "cluster": c,
        "block_m": bm,
        "chunks": chunks,
        "z_dtype": None if z_out is None else z_out.dtype,
        "id_dtype": torch.int32 if ids is None else ids.dtype,
        "weighted": routed_weights is not None,
    }
    if bm:
        kwargs["block_n"], kwargs["stages"] = block_pipeline(bm, a.shape[-2])
        # 128-wide K tiles halve the shrink trips; they measured 1-3% faster on the
        # small-SM, bandwidth-poor sm_121 part and neutral-to-slower on sm_120.
        kwargs["block_k"] = _BLOCK_OVERRIDES.get(
            "block_k", 128 if sms < 64 and _block_k128_fits(args, kwargs) else 64
        )
        kwargs["shrink_stages"] = _BLOCK_OVERRIDES.get("shrink_stages", kwargs["stages"])
    key = (x.device, *args, *kwargs.values())
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "warm up this MoE LoRA shape and z_out dtype before CUDA graph capture"
            )
        with LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                compiled = compile_kernel(*args, **kwargs)
                _COMPILED[key] = compiled
    compiled(
        x,
        a[0],
        a.gap,
        b[0],
        b.gap,
        out,
        ids,
        routed_weights,
        scale,
        z_out,
        sorted_token_ids if bm and ids is not None else None,
        expert_ids if bm and ids is not None else None,
        num_tokens_post_padded if bm and ids is not None else None,
        *((activated,) if act else ()),
    )
    return activated if act else out


def shrink_(z, x, a, ids, top_k, *, down=False):
    """z[p, s] = A_s[ids[p]] @ x[row(p)] for every local pair (pair kernel, no expand).

    ``a`` is the factor's FactorSlices; z [P, slices, r] is FP32 or BF16; pairs whose
    id is not a local expert leave z untouched. For callers fusing the expand elsewhere.
    """
    capability, sms = device_info(x.device)
    if capability not in ((12, 0), (12, 1)) or not available():
        raise RuntimeError("cute requires sm_120/sm_121 and nvidia-cutlass-dsl 4.8.0")
    p, slices, rank = z.shape
    if p == 0:
        return z
    c, _, _ = configuration(p, slices, sms, x.shape[1], 8, block_m=16, rank=rank)
    args = (x.shape[1], 8, rank, slices, top_k, a.shape[0], capability)
    kwargs = {"down": down, "cluster": c, "z_dtype": z.dtype, "id_dtype": ids.dtype}
    key = (x.device, "shrink", *args, *kwargs.values())
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up this MoE LoRA shrink before CUDA graph capture")
        with LOCK:
            compiled = _COMPILED.get(key)
            if compiled is None:
                compiled = compile_kernel(*args, stage="shrink", **kwargs)
                _COMPILED[key] = compiled
    dummy = a[0].new_empty((a[0].shape[0], 8, rank))
    compiled(
        x, a[0], a.gap, dummy, 0, None, ids, None, scale_tensor(1.0, x.device), z,
        None, None, None,
    )  # fmt: skip
    return z
