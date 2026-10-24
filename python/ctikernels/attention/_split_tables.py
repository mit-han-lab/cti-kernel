# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Measured device dispatch choices and host-only layout guards."""

from typing import NamedTuple

import torch

# Graph FA2 prefill: (batch, inclusive query/request range, inclusive KV-page
# range) -> target chunks or {"chunks": C, "tile": t}. None is an unbounded
# upper page count. These RTX 5090
# rows were screened with FlashInfer 0.7.0.post1, equal FP16/BF16 Q/KV.
PREFILL_GRAPH_SPLIT = {
    # Paired engine-regime screens (graph replay, consecutive layers, page 16) vs the
    # provider plan: C=4 gives 0.92-0.95x (KV 2048) and 0.81-0.88x (KV 8192) of the
    # provider at Q=256, B=2-4; C=6/8 gain a few percent more but grow the bounded
    # partial-output workspace by half or double. Q<=128 keeps the provider's split.
    ("rtx5090", 32, 8, 128): {(b, (129, 256), (0, None)): 4 for b in (2, 3, 4)},
    # Qwen3.5 hd256: C=4 gives 0.64-0.91x of the provider at Q=256, B=2-4.
    ("rtx5090", 16, 4, 256): {(b, (129, 256), (0, None)): 4 for b in (2, 3, 4)},
    # RTX PRO 6000 (188 SMs), same screens: B2 needs C=6 (0.88-0.95x; C=4 was 1.08-1.11x at
    # KV 2048), B3/B4 C=4 (0.67-0.96x). Qwen3-8B B4 at <=256 pages keeps the provider plan
    # (every C lost to vLLM there; C=4 was worse than the provider).
    ("rtxpro6000", 32, 8, 128): {
        (2, (129, 256), (0, None)): 6,
        (3, (129, 256), (0, None)): 4,
        (4, (129, 256), (257, None)): 4,
    },
    ("rtxpro6000", 16, 4, 256): {
        (2, (129, 256), (0, None)): 6,
        (3, (129, 256), (0, None)): 4,
        (4, (129, 256), (0, None)): 4,
    },
    # GB10 Qwen3-8B: steering the provider to its 64-row tile (more CTAs per SM on the
    # DRAM-bound extend shapes) gives 0.92-0.97x of vLLM at B1 Q=256 and B2 Q=64
    # (KV 2048/8192); the provider's 128-row tile was 1.07-1.13x.
    ("gb10", 32, 8, 128): {
        (1, (129, 256), (0, None)): {"chunks": 1, "tile": 64},
        (2, (33, 128), (0, None)): {"chunks": 1, "tile": 64},
        # Q<=16: the 16-row tile gives 4x the CTAs (as vLLM's per-head blocks), 0.98-0.99x
        # of vLLM at B3/B4 up to 256 pages; longer KV keeps the provider tile.
        (3, (9, 16), (0, 256)): {"chunks": 1, "tile": 16},
        (4, (9, 16), (0, 256)): {"chunks": 1, "tile": 16},
    },
    ("gb10", 16, 4, 256): {
        (2, (9, 16), (0, None)): {"chunks": 1, "tile": 16},
        (3, (9, 16), (0, None)): {"chunks": 1, "tile": 16},
        (4, (9, 16), (0, None)): {"chunks": 2, "tile": 16},
        # Q=256, B2 beyond 256 pages: C=4 gives 0.93x of vLLM (provider 1.05x).
        (2, (129, 256), (257, None)): 4,
    },
}


def prefill_graph_rows(device, hq, hk, d):
    return PREFILL_GRAPH_SPLIT.get((device_key(device), hq, hk, d), {})


def prefill_graph_config(value):
    """Normalize legacy chunk counts and optional graph tile targets."""
    return {"chunks": value} if isinstance(value, int) else value


def prefill_graph_policy(device, hq, hk, d, batch, max_query_per_request, max_kv_pages):
    for (b, (qlo, qhi), (plo, phi)), value in prefill_graph_rows(device, hq, hk, d).items():
        if (
            b == batch
            and qlo <= max_query_per_request <= qhi
            and plo <= max_kv_pages
            and (phi is None or max_kv_pages <= phi)
        ):
            return prefill_graph_config(value)
    return {"chunks": 1}


def prefill_graph_chunks(device, hq, hk, d, batch, max_query_per_request, max_kv_pages) -> int:
    return prefill_graph_policy(device, hq, hk, d, batch, max_query_per_request, max_kv_pages)[
        "chunks"
    ]


def prefill_graph_max_chunks(device, hq, hk, d, batch) -> int:
    return max(
        (
            prefill_graph_config(c)["chunks"]
            for (b, _, _), c in prefill_graph_rows(device, hq, hk, d).items()
            if b == batch
        ),
        default=1,
    )


class PrefillGraphSplit(NamedTuple):
    padding: int
    tile: int
    padded: int


def prefill_graph_split_layout(
    total, batch, group, head_dim, max_chunks, split_capacity, target_tile=None
):
    """First-plan capacity preserving the natural tile or reaching the requested tile."""
    if (max_chunks <= 1 and target_tile is None) or head_dim > 256:
        return None

    def tile_for(requests):
        # FlashInfer 0.7.0.post1 scheduler.cuh:603-605 subtracts in uint32 before
        # promotion to uint64; utils.cuh:429-479 chooses these tiles for D<=256.
        packed = ((total - requests + 1) & 0xFFFFFFFF) * group
        return 16 if packed <= 16 else (128 if packed > 64 and head_dim < 256 else 64)

    if target_tile is not None:
        padding = total + 1 - target_tile // group - batch
        if padding < 0 or tile_for(batch + padding) != target_tile:
            return None
        tiles = (total * group + target_tile - 1) // target_tile + batch - 1
        return PrefillGraphSplit(padding, target_tile, max(split_capacity, tiles + padding))

    tile = tile_for(batch)
    tiles = (total * group + tile - 1) // tile + batch - 1
    padding = (max_chunks - 1) * tiles + 1
    if tile_for(batch + padding) != tile:
        return None
    return PrefillGraphSplit(padding, tile, max(split_capacity, tiles + padding))


# Opt-in CuTe graph prefill configurations, keyed by named device and geometry.
PREFILL_GRAPH_CUTE: dict[tuple[str, int, int, int], dict[tuple[int, tuple[int, int]], dict]] = {
    # GB10 Qwen3-8B extend at B3/B4, Q<=128: the CuTe prefill (64-row GQA tiles) gives
    # 0.95-0.99x of vLLM's FA2 at KV 2048/8192 where FlashInfer was 1.05-1.33x.
    ("gb10", 32, 8, 128): {
        (3, (33, 128)): {"block_m": 64},
        (4, (33, 128)): {"block_m": 64, "block_n": 32},
    },
    # GB10 Qwen3.5 hd256 B3 Q<=128: 0.94-1.00x of vLLM (FlashInfer 1.03-1.16x).
    ("gb10", 16, 4, 256): {(3, (33, 128)): {}},
}


def prefill_graph_cute_rows(device, hq, hk, d):
    return PREFILL_GRAPH_CUTE.get((device_key(device), hq, hk, d), {})


def prefill_graph_cute(device, hq, hk, d, batch, max_query_per_request):
    for (b, (qlo, qhi)), config in prefill_graph_cute_rows(device, hq, hk, d).items():
        if b == batch and qlo <= max_query_per_request <= qhi:
            return config
    return None


def prefill_graph_chunk(queries, pages, group, chunks, layout) -> int | None:
    """Keep the captured split path and its CTA capacity, including empty KV."""
    maximum = max(pages, default=0)
    # scheduler.cuh:639-640,653-654 only enables the fixed split path if some
    # request has >1 partition. Auto planning keeps split_kv=True in graph mode
    # even for tiny KV (scheduler.cuh:129), preserving the captured merge nodes.
    if chunks <= 1 or maximum <= 1:
        return None
    chunk = (maximum + chunks - 1) // chunks
    count = sum(
        ((q * group + layout.tile - 1) // layout.tile) * ((max(p, 1) + chunk - 1) // chunk)
        for q, p in zip(queries, pages, strict=True)
    )
    return chunk if count <= layout.padded else None


# Decode batches routed to FlashInfer's CUDA-core kernel, in eager and graph mode alike
# (FlashInferPagedAttention._native_decode_tensor_cores). GB10 keeps tensor cores
# (SGLang 72730d6's choice): CUDA cores win 3-12% at KV <= 2048 there but lose 1-2% to
# it at KV >= 4096, and a captured graph cannot switch kernels with KV length.
DECODE_CUDA_CORE = {
    ("rtx5090", 32, 8, 128): {2, 3, 4},
    ("rtxpro6000", 32, 8, 128): {2, 3, 4},
}

# FP8 e4m3 KV decode through XQA: sub-sequences per (request, KV head), by device,
# per-rank heads (Qwen3-8B, Qwen3.5-9B/27B and their TP2 ranks) and graph batch.
# Absent batches keep FA2. RTX 5090 32/8/128 uses 170 // (B * 8), matching
# vLLM 0.31's split and bitwise XQA output with the same block-table width.
# Tuned counts elsewhere beat both markets by 2-15% in the paired engine-regime
# receipts, on rows where vLLM's split only ties or loses. On those rows output
# differs from vLLM's XQA only by sub-sequence merge order.
# Receipts: benchmark/kernel_tuning/results/paged_attention_range.md;
# FlashInfer 0.7.0.post1, page 16, context-wide tables, KV 512..32768 (hd128)
# / 512..131072 (hd256). RTX PRO 6000 retains the earlier mirrored tuned counts
# pending its own screen.
XQA_DECODE_SUBSEQ: dict[tuple[str, int, int, int], dict[int, int]] = {
    ("rtx5090", 32, 8, 128): {1: 21, 2: 10, 3: 7, 4: 5},
    ("rtx5090", 16, 4, 128): {1: 16, 2: 16, 3: 8, 4: 8},
    ("rtx5090", 16, 4, 256): {2: 8, 3: 8, 4: 4},
    ("rtx5090", 24, 4, 256): {2: 8, 3: 8, 4: 4},
    ("rtx5090", 8, 2, 256): {4: 8},
    ("rtx5090", 12, 2, 256): {4: 8},
    ("rtxpro6000", 32, 8, 128): {1: 16, 2: 8, 3: 4, 4: 4},
    ("rtxpro6000", 16, 4, 128): {1: 16, 2: 16, 3: 8, 4: 8},
    ("rtxpro6000", 16, 4, 256): {2: 8, 3: 8, 4: 4},
    ("rtxpro6000", 24, 4, 256): {2: 8, 3: 8, 4: 4},
    ("rtxpro6000", 8, 2, 256): {4: 8},
    ("rtxpro6000", 12, 2, 256): {4: 8},
    ("gb10", 32, 8, 128): {1: 4, 2: 1, 3: 1, 4: 1},
    ("gb10", 16, 4, 256): {1: 4, 2: 3, 3: 2, 4: 1},
    ("gb10", 24, 4, 256): {1: 4, 2: 3, 3: 2, 4: 1},
}


# CuTe FP8 decode configuration overrides by device, geometry and batch.
# Unlisted batches retain the existing XQA/FA2 route.
FP8_DECODE_CUTE: dict[tuple[str, int, int, int], dict[int, dict]] = {
    # RTX 5090 paired engine-regime screens (FlashInfer 0.7.0.post1, page 16, KV 512-32768):
    # the runtime split serves every KV length of a captured batch. Qwen3.5 hd256 B1/B3
    # (min_chunk 64), B4 (at or below XQA S4 at every KV) and the TP2 ranks B1-B4 win or tie the faster of vLLM XQA and SGLang FA2
    # where the fixed XQA/FA2 route lost one side of the KV range. Qwen3-8B hd128 keeps
    # XQA/FA2 (the CuTe kernel trails by 4-10% at KV 2048-8192, B>=2).
    ("rtx5090", 16, 4, 256): {1: {"min_chunk": 64}, 3: {"min_chunk": 64}, 4: {}},
    ("rtx5090", 8, 2, 256): {
        1: {"min_chunk": 128},
        2: {"min_chunk": 128},
        3: {"min_chunk": 128},
        4: {},
    },
    ("rtx5090", 12, 2, 256): {
        1: {"min_chunk": 128},
        2: {"min_chunk": 128},
        3: {"min_chunk": 128},
        4: {},
    },
    # RTX PRO 6000 B2 paired screen: FlashInfer 0.7.0.post1, page 16,
    # engine regime, 16 rounds. KV 512/2048/8192/32768: XQA S8 vs CuTe
    # 11.98/10.28, 12.86/12.89, 30.22/30.09, 99.62/96.98 us;
    # CuTe / faster market = 0.820/0.823/0.973/0.977 (XQA S8 1.004
    # at KV32768 was the OPEN row). Other rows mirror RTX 5090;
    # splits follow the SM count.
    ("rtxpro6000", 16, 4, 256): {
        1: {"min_chunk": 64},
        2: {"min_chunk": 64},
        3: {"min_chunk": 64},
        4: {},
    },
    ("rtxpro6000", 8, 2, 256): {b: {"min_chunk": 128} for b in (1, 2, 3)} | {4: {}},
    ("rtxpro6000", 12, 2, 256): {b: {"min_chunk": 128} for b in (1, 2, 3)} | {4: {}},
}


def fp8_decode_cute(device, hq, hk, d, batch) -> dict | None:
    return FP8_DECODE_CUTE.get((device_key(device), hq, hk, d), {}).get(batch)


def xqa_decode_subseq(device, num_qo_heads, num_kv_heads, head_dim, batch) -> int | None:
    """Table split bound for listed batches, independent of KV length."""
    return XQA_DECODE_SUBSEQ.get(
        (device_key(device), num_qo_heads, num_kv_heads, head_dim), {}
    ).get(batch)


# Divisor, minimum chunk, minimum/maximum total queries, maximum batch.
PREFILL_QUERY_SPLIT = {
    "rtx5090": (16, 4, 1, 256, 4),
    "rtxpro6000": (16, 8, 17, 256, 4),
    "gb10": (6, 8, 1, 128, 2),
}


def decode_uses_cuda_cores(device, num_qo_heads, num_kv_heads, head_dim, batch) -> bool:
    return batch in DECODE_CUDA_CORE.get(
        (device_key(device), num_qo_heads, num_kv_heads, head_dim), ()
    )


class PrefillQuerySplit(NamedTuple):
    qo_indptr: list[int]
    kv_indptr: list[int]
    last_page_len: list[int]
    seq_lens: list[int]
    gather_positions: list[int]


def _prefill_chunk(total: int, divisor: int, minimum: int) -> int:
    target = (total + divisor - 1) // divisor
    return min(16, max(minimum, 1 << (target - 1).bit_length()))


def prefill_split_batch_bound(
    key: str | None, max_batch_size: int, max_query_tokens: int
) -> int | None:
    """Maximum pseudo-request count over all admitted fresh-prefill lengths."""
    policy = PREFILL_QUERY_SPLIT.get(key)
    if policy is None:
        return None
    divisor, minimum, min_total, max_total, max_batch = policy
    bound = None
    for total in range(max(1, min_total), min(max_query_tokens, max_total) + 1):
        chunk = _prefill_chunk(total, divisor, minimum)
        for batch in range(1, min(max_batch_size, max_batch, total) + 1):
            # One token per request, then each chunk buys one extra pseudo-request.
            extra = (total - batch) // chunk
            if extra:
                bound = max(bound or 0, batch + extra)
    return bound


def prefill_query_split(
    key: str | None,
    qo_indptr: list[int],
    kv_indptr: list[int],
    seq_lens: list[int],
    *,
    head_dim: int,
    causal: bool,
    page_size: int,
) -> PrefillQuerySplit | None:
    """Build consecutive query chunks sharing each fresh request's KV prefix."""
    policy = PREFILL_QUERY_SPLIT.get(key)
    if policy is None or head_dim != 128 or not causal or page_size != 1:
        return None
    divisor, minimum, min_total, max_total, max_batch = policy
    total = qo_indptr[-1]
    lengths = [end - start for start, end in zip(qo_indptr[:-1], qo_indptr[1:], strict=True)]
    if (
        not 1 <= len(lengths) <= max_batch
        or not min_total <= total <= max_total
        or lengths != seq_lens
        or any(length <= 0 for length in lengths)
    ):
        return None
    chunk = _prefill_chunk(total, divisor, minimum)
    if max(lengths) <= chunk:
        return None
    result = PrefillQuerySplit([0], [0], [], [], [])
    for request, length in enumerate(lengths):
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            result.qo_indptr.append(result.qo_indptr[-1] + end - start)
            result.kv_indptr.append(result.kv_indptr[-1] + end)
            result.last_page_len.append(1)
            result.seq_lens.append(end)
            result.gather_positions.extend(range(kv_indptr[request], kv_indptr[request] + end))
    return result


def device_key(device) -> str | None:
    properties = torch.cuda.get_device_properties(device)
    return {
        (12, 0, 170): "rtx5090",
        (12, 0, 188): "rtxpro6000",
        (12, 1, 48): "gb10",
    }.get((properties.major, properties.minor, properties.multi_processor_count))


# Decode only; FixedQuery keeps its own schedule.
# S is splits per request: chunk = ceil(max KV pages / S); zero keeps the provider
# default. Rows were chosen as >=2% wins under both FlashInfer 0.6.11.post3 and
# 0.7.0.post1 from paired engine-regime sweeps (consecutive layers with distinct KV)
# in benchmark/kernel_tuning/paged_attention_sweep.py. GB10 hd256 rows instead pass
# the GB10 rule against the provider plan (>= 60 interleaved pairs, X925-pinned,
# bootstrap 95% CI of the median ratio entirely below 1) under both versions.
# Unlisted shapes, devices and batches keep the provider default. Rows are B=1..4;
# columns are KV buckets 0..5.
_TABLES = {
    ("rtx5090", "decode", 32, 8, 128): (
        (8, 0, 0, 0, 16, 16),
        (8, 0, 0, 16, 16, 16),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
    ("rtxpro6000", "decode", 32, 8, 128): (
        (8, 16, 0, 0, 0, 0),
        (8, 0, 0, 0, 0, 0),
        (8, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
    ),
    ("rtx5090", "decode", 16, 4, 256): (
        (16, 0, 0, 16, 16, 16),
        (8, 8, 8, 8, 16, 16),
        (0, 0, 8, 8, 8, 8),
        (0, 0, 8, 8, 8, 8),
    ),
    ("rtxpro6000", "decode", 16, 4, 256): (
        (16, 16, 16, 16, 16, 16),
        (8, 8, 0, 16, 16, 16),
        (8, 8, 8, 8, 8, 8),
        (0, 0, 8, 8, 8, 8),
    ),
    ("gb10", "decode", 16, 4, 256): (
        (0, 0, 0, 0, 8, 8),
        (0, 0, 0, 0, 4, 4),
        (0, 0, 0, 0, 2, 2),
        (0, 0, 0, 0, 2, 2),
    ),
}


def kv_bucket(tokens: int) -> int:
    return sum(tokens >= edge for edge in (1024, 2048, 4096, 8192, 16384))


def lookup(
    device, op: str, num_qo_heads: int, num_kv_heads: int, head_dim: int, batch: int
) -> tuple[int, ...] | None:
    if not 1 <= batch <= 4:
        return None
    table = _TABLES.get((device_key(device), op, num_qo_heads, num_kv_heads, head_dim))
    return None if table is None else table[batch - 1]


def decode_chunk(
    row: tuple[int, ...] | None,
    pages: list[int],
    max_tokens: int,
    capacity: int,
) -> int | None:
    """Choose a page chunk only when it preserves Split-KV within ``capacity``.

    ``capacity`` is the mode-independent work bound of decode_split_capacity, so
    eager and graph plans of the same lengths pick the same chunk.
    """
    splits = 0 if row is None else row[kv_bucket(max_tokens)]
    if splits == 0 or not pages:
        return None
    max_pages = max(pages)
    chunk = (max_pages + splits - 1) // splits
    if max_pages <= chunk:
        return None
    if sum((count + chunk - 1) // chunk for count in pages) > capacity:
        return None
    return chunk
