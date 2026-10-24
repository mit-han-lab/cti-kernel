# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired segmented-delta timings: ours versus SGLang 72730d6, or our Triton versus CuTe.

Run as a module with --sglang-python pointing to the external python directory.
``--compare cute`` pairs our two providers on the same card instead (both through
lora_segmented_delta_, the provider forced per side); --sglang-python is then unused.
No vendor kernel code is vendored. Both sides use stable adapter grouping and
16-token chunks, prepared outside capture/timing, with dense packed A slices. Each timing sample resets
output *before* the start event, then replays a one-operation CUDA graph. Two
independent captures per side provide self-pair noise estimates. Data is shared
between captures (warm/resident operands); this is not a DRAM-rotation benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import statistics
import traceback
from itertools import accumulate
from pathlib import Path

import torch

from benchmark.kernel_tuning.lora_sglang import CASES, BatchInfo, _pair_result, load_sglang
from ctikernels.linear import (
    _lora_policy,
    _lora_segmented,
    lora_segment_metadata,
    lora_segmented_delta_,
)

SHAPES = {
    "qkv": (4096, (8192, 1024, 1024)),
    "o": (4096, (4096,)),
    "gate_up": (4096, (12288, 12288)),
    "down": (12288, (4096,)),
}


def cases():
    for m in (1, 2, 4, 8, 16, 32, 64):
        # Counts larger than B cannot all be represented by a decode batch.
        for adapters in sorted({a for a in (1, 2, 4, m) if a <= m}):
            yield "decode", m, m, adapters
    for m in (512, 2048):
        yield "prefill", m, 4, 4


def metadata(m, segments, adapters, rank, *, device="cuda"):
    ptr = [i * m // segments for i in range(segments + 1)]
    ids = [s % adapters for s in range(segments)]
    integer = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
    ours = {
        # CSR, permutation and O(1) chunk table are built once here, outside
        # warm-up/capture/timing, and reused by every ours projection below.
        **lora_segment_metadata(
            ids, [b - a for a, b in zip(ptr, ptr[1:], strict=False)], device=device
        ),
        "ranks": integer([rank] * adapters),
        "scalings": torch.linspace(0.5, 2.0, adapters, device=device),
    }
    # Match backend semantics: stable grouping by adapter, then 16-token
    # chunks in logical order; permutation maps those rows back to requests.
    permutation, chunk_ptr, chunk_ids = [], [0], []
    for adapter in range(adapters):
        tokens = [
            token
            for seg, owner in enumerate(ids)
            if owner == adapter
            for token in range(ptr[seg], ptr[seg + 1])
        ]
        permutation.extend(tokens)
        for lo in range(0, len(tokens), 16):
            chunk_ptr.append(chunk_ptr[-1] + min(16, len(tokens) - lo))
            chunk_ids.append(adapter)
    live = len(chunk_ids)
    # Decode allocates one graph segment slot per request. Prefill allocates
    # ceil(M/chunk_size) (at least 16) plus the adapter-pool capacity.
    slots = m if m == segments else max(16, (m + 15) // 16) + adapters
    chunk_ptr.extend([m] * (slots - live))
    chunk_ids.extend([0] * (slots - live))
    info = BatchInfo(
        bs=segments,
        num_segments=live,
        max_len=16,
        seg_lens=integer([b - a for a, b in zip(chunk_ptr, chunk_ptr[1:], strict=False)]),
        seg_indptr=integer(chunk_ptr),
        weight_indices=integer(chunk_ids),
        lora_ranks=ours["ranks"],
        scalings=ours["scalings"],
        permutation=integer(permutation),
        expected_tokens=m,
        use_cuda_graph=True,
        req_seg_indptr=integer(ptr),
        req_weight_indices=integer(ids),
    )
    return ours, info


@contextlib.contextmanager
def forced(provider):
    """Force the segmented provider while a side warms up and captures (None: device rule)."""
    rule, pick = _lora_policy.segmented_provider, _lora_segmented.PROVIDER.config
    if provider is not None:
        _lora_policy.segmented_provider = lambda capability: provider
        _lora_segmented.PROVIDER.config = lambda device, **shape: {"provider": provider}
    try:
        yield
    finally:
        _lora_policy.segmented_provider, _lora_segmented.PROVIDER.config = rule, pick


def oracle(x, a, b, meta, sizes):
    xc, ac, bc = x.cpu(), a.cpu(), b.cpu()
    ptr, ids, ranks = (
        meta[key].cpu().tolist() for key in ("seg_indptr", "weight_indices", "ranks")
    )
    scales = meta["scalings"].cpu().tolist()
    permutation = meta["permutation"].cpu().long()
    delta = torch.zeros(x.shape[0], b.shape[1], dtype=torch.float64)
    for seg, adapter in enumerate(ids):
        lo, hi = ptr[seg], ptr[seg + 1]
        if adapter < 0 or lo == hi:
            continue
        rank = ranks[adapter]
        rows = permutation[lo:hi]
        off = 0
        for s, size in enumerate(sizes):
            aa = ac[adapter, s * rank : (s + 1) * rank].double()
            bb = bc[adapter, off : off + size, :rank].double()
            delta[rows, off : off + size] = scales[adapter] * (xc[rows].double() @ aa.T) @ bb.T
            off += size
    return delta


def correctness(output, before, delta):
    actual, before = output.cpu().double(), before.cpu().double()
    expected = before + delta
    norm = lambda value: float(torch.linalg.vector_norm(value))
    full = norm(actual - expected) / max(norm(expected), 1e-30)
    change = norm((actual - before) - delta) / max(norm(delta), 1e-30)
    return {
        "ok": math.isfinite(full) and math.isfinite(change) and full <= 4e-3 and change <= 2e-2,
        "full_rel_l2": full if math.isfinite(full) else None,
        "delta_rel_l2": change if math.isfinite(change) else None,
    }


def capture(call, output, before):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output.copy_(before)
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


def sample(graph, output, before):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(10):
        output.copy_(before)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def paired(graphs, outputs, before, pairs, names=("ours", "sglang")):
    first, second = names
    order = ((first, 0), (second, 0), (first, 1), (second, 1))
    samples = {key: [] for key in order}
    for round_index in range(-3, pairs):
        shift = round_index % 4
        rotation = order[shift:] + order[:shift]
        if round_index % 2:
            rotation = rotation[::-1]
        for key in rotation:
            value = sample(graphs[key], outputs[key], before)
            if round_index >= 0:
                samples[key].append(value)
    result = _pair_result(samples, first, second, pairs)
    return {
        f"{first}_us": result["ours_us"],
        f"{second}_us": result["sgl_us"],
        f"speedup_{second}_over_{first}": result["ratio_sgl_ours"],
        "epsilon": result["eps"],
        "verdict": result["verdict"],
        "ratio_iqr": result["ratio_iqr"],
        "self_pairs": result["self_pairs"],
    }


def run(row, sglang, dtype, pairs, sides=(("ours", None), ("sglang", None))):
    m, k, rank, adapters = row["M"], row["K"], row["rank"], row["adapters"]
    sizes = row["output_sizes"]
    meta, info = metadata(m, row["segments"], adapters, rank)
    row.update(
        ours_chunk_size=16,
        ours_grouped_by_adapter=True,
        ours_live_segments=meta["weight_indices"].numel(),
        ours_chunk_slots=meta["chunks"].chunk_adapter.numel(),
        ours_precomputed_chunks=True,
        sglang_chunk_size=16,
        sglang_grouped_by_adapter=True,
        sglang_live_segments=info.num_segments,
        sglang_segment_slots=info.weight_indices.numel(),
    )
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
    a = torch.randn(adapters, len(sizes) * rank, k, device="cuda", dtype=dtype) * 0.02
    b = torch.randn(adapters, sum(sizes), rank, device="cuda", dtype=dtype) * 0.02
    before = torch.randn(m, sum(sizes), device="cuda", dtype=dtype) * 0.005
    expected = oracle(x, a, b, meta, sizes)
    offsets_cpu = torch.tensor([0, *accumulate(sizes)], dtype=torch.int32)
    offsets = offsets_cpu.to("cuda")
    if sglang is not None:
        modules = sglang.modules
        shrink = modules["chunked_sgmv_shrink"].chunked_sgmv_lora_shrink_forward
        expand = modules["chunked_sgmv_expand"].chunked_sgmv_lora_expand_forward
    outputs, graphs, owners = {}, {}, []
    row["correctness"] = {}
    for name, provider in sides:
        for i in range(2):
            output = before.clone()
            if name != "sglang":

                def call(out=output):
                    return lora_segmented_delta_(out, x, a, b, **meta, output_sizes=sizes)
            else:

                def call(out=output):
                    low = shrink(x=x, weights=a, batch_info=info, num_slices=len(sizes))
                    return expand(
                        x=low,
                        weights=b,
                        batch_info=info,
                        slice_offsets=offsets,
                        max_slice_size=max(sizes),
                        base_output=out,
                        slice_offsets_cpu=offsets_cpu,
                    )

            with forced(provider):
                call()
                checked = correctness(output, before, expected)
                row["correctness"][f"{name}_{i}_eager"] = checked
                graph = capture(call, output, before)
            output.copy_(before)
            graph.replay()
            row["correctness"][f"{name}_{i}_graph"] = correctness(output, before, expected)
            outputs[name, i], graphs[name, i] = output, graph
            owners.append(call)
    row.update(paired(graphs, outputs, before, pairs, tuple(name for name, _ in sides)))
    row["correct"] = all(v["ok"] for v in row["correctness"].values())
    if not row["correct"]:
        row["timing_verdict"] = row["verdict"]
        row["verdict"] = "incorrect"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-python", type=Path)
    parser.add_argument("--compare", choices=("sglang", "cute"), default="sglang")
    parser.add_argument(
        "--model", choices=tuple(CASES), help="Qwen3.5 projection shapes instead of --shapes"
    )
    parser.add_argument("--ms", nargs="+", type=int, help="keep only these M")
    parser.add_argument(
        "--rows", nargs="+", help="keep only these rows, each shape,M,rank,adapters"
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--shapes", nargs="+", choices=tuple(SHAPES), default=tuple(SHAPES))
    parser.add_argument("--ranks", nargs="+", type=int, default=(16, 64, 128))
    parser.add_argument("--mode", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument(
        "--provider",
        choices=("auto", "triton"),
        default="auto",
        help="triton forces our Triton provider (what non-sm_120/sm_121 devices select)",
    )
    args = parser.parse_args(argv)
    if args.provider == "triton":
        from ctikernels.linear import _lora_policy

        _lora_policy.segmented_provider = lambda capability: "triton"
    if args.pairs < 2:
        parser.error("--pairs must be at least two")
    # File-path loading and minimal dependency stubs match the existing harness.
    from benchmark.kernel_tuning._prologue_market import quiet_external

    sglang, sides = None, (("triton", "triton"), ("cute", "cute"))
    if args.compare == "sglang":
        if args.sglang_python is None:
            parser.error("--compare sglang needs --sglang-python")
        with quiet_external():
            sglang = load_sglang(args.sglang_python)
        sides = (("ours", None), ("sglang", None))
    shapes = (
        [(c.name, c.input_size, c.output_sizes) for c in CASES[args.model]]
        if args.model
        else [(name, *SHAPES[name]) for name in args.shapes]
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    with torch.inference_mode():
        for name, k, sizes in shapes:
            for rank in args.ranks:
                for mode, m, segments, adapters in cases():
                    if args.mode != "all" and args.mode != mode:
                        continue
                    if args.ms and m not in args.ms:
                        continue
                    if args.rows and f"{name},{m},{rank},{adapters}" not in args.rows:
                        continue
                    row = {
                        "model": args.model,
                        "compare": args.compare,
                        "device": torch.cuda.get_device_name(),
                        "shape": name,
                        "K": k,
                        "output_sizes": sizes,
                        "rank": rank,
                        "M": m,
                        "mode": mode,
                        "segments": segments,
                        "adapters": adapters,
                        "dtype": args.dtype,
                        "pairs": args.pairs,
                        "reset_timed": False,
                        "sglang_source": "72730d6",
                        "warm_rounds": 3,
                        "replays_per_sample": 10,
                        "operand_policy": "shared-resident",
                        "sglang_split_k": None
                        if sglang is None
                        else sglang.environ.envs.SGLANG_ENABLE_LORA_CSGMV_SPLIT_K.get(),
                        "provider": args.provider,
                        "snapshot": os.environ.get("CTI_BENCH_SNAPSHOT"),
                    }
                    try:
                        with quiet_external():
                            run(row, sglang, dtype, args.pairs, sides)
                    except Exception as error:  # noqa: BLE001 -- report other cases independently
                        row.update(
                            error=type(error).__name__,
                            error_message=str(error)[:300],
                            error_where=[
                                f"{f.filename.rsplit('/', 3)[-1]}:{f.lineno} {f.line}"
                                for f in traceback.extract_tb(error.__traceback__)[-3:]
                            ],
                            verdict="unavailable",
                            correct=False,
                        )
                    print("ROW " + json.dumps(row, allow_nan=False), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
