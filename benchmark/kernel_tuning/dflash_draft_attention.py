# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DF2 complete-op parity; run as a module from the repository root.

PYTHONPATH=python python -m benchmark.kernel_tuning.dflash_draft_attention
Only JSON Lines go to stdout. Stock FlashInfer host plan is inside the eager
reference interval. DF2 gates complete-op graph replay against stock FlashInfer
fixed-plan graph replay; run-only graph cost is informational.
All rows are active and immutable, so neither comparator needs a padding adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict

import torch

try:
    from benchmark.sol.sol import Call, Kernel, estimate, load_ceilings
except ImportError:  # the shared SOL helper ships separately; SOL is informational
    Call = Kernel = estimate = load_ceilings = None
from ctikernels.attention.inference import DraftAttention

CONFIGS = {
    "full": (16, False, None),
    "causal_sliding": (16, True, 4096),
    "noncausal_sliding": (8, False, 2048),
}
PAIRS = 20
ATOL = RTOL = 2e-2
GATE_REFERENCE = (
    "stock FlashInfer BatchPrefillWithPagedKVCacheWrapper fa2 fixed-plan CUDA graph "
    "(host plan outside timing; cannot follow changing metadata)"
)
QUERY_LENGTHS = (4, 5, 8, 9, 12, 13, 16, 17)
SOL_DEFINITION = (
    "Informational useful-work empirical reference from benchmark/sol/ceilings/{table}.json; "
    "serial graph kernels, no PDL: sum(max(graph floor, DRAM time, BF16 MMA time)). "
    "One attention node carries BF16 Q reads, union-visible K/V reads per KV head, "
    "one int32 page index per union key and BF16 output writes; FLOPs = "
    "4*B*Hq*128*sum(query visible-key counts). Remaining launches are floor-only nodes. "
    "The benchmark replays identical buffers, so min(1, L2 size/working set) of every "
    "attention byte is served at the measured L2 read rate (warm_l2 when it all fits, "
    "else partial_l2), the rest at the sustained DRAM read rate as an aggregate traffic "
    "proxy; cold_t_ref_us keeps the all-DRAM reference. Excludes padding, "
    "scalar work, scratch traffic and eager host overhead. "
    "Utilization is t_ref/measured latency, with ratios above one flagged."
)


def visible_key_counts(config, context, *, query_length=None):
    """Return per-query counts and union size for bottom-right aligned queries."""
    qlen, causal, window = CONFIGS[config]
    qlen = qlen if query_length is None else query_length
    length = context + qlen
    intervals = []
    for position in range(context, length):
        start = 0 if window is None else max(0, position - window + 1)
        end = position + 1 if causal else length
        if window is not None and not causal:
            end = min(end, position + window)
        intervals.append((start, end))
    return [end - start for start, end in intervals], intervals[-1][1] - intervals[0][0]


def sol_estimate(
    device_name, config, b, context, hq, hk, kernels_per_call, l2_fraction=0.0, *, query_length=None
):
    """Return (table name, useful-work Result), or None for an unknown device.

    l2_fraction of every attention byte is served from L2: the benchmark replays
    identical buffers, so up to the L2 size of the working set can stay resident.
    """
    name = device_name.upper()
    table = next(
        (
            table
            for model, table in (
                ("RTX 5090", "rtx5090"),
                ("RTX PRO 6000", "rtx-pro-6000"),
                ("GB10", "gb10"),
            )
            if model in name
        ),
        None,
    )
    if table is None or estimate is None:
        return None
    if not isinstance(kernels_per_call, int) or kernels_per_call < 1:
        raise ValueError("kernels_per_call must be a positive integer")
    counts, union = visible_key_counts(config, context, query_length=query_length)
    q_bytes = b * len(counts) * hq * 128 * 2
    dram_bytes = {
        "activations": q_bytes,
        "kv_state": b * union * (2 * hk * 128 * 2 + 4),
        "outputs": q_bytes,
    }
    attention = Kernel(
        "attention",
        dram_bytes=dram_bytes,
        flops={"bf16_m16n8k16": 4 * b * hq * 128 * sum(counts)},
        cache_state="warm" if l2_fraction else "cold",
        l2_resident_bytes={key: size * l2_fraction for key, size in dram_bytes.items()},
    )
    kernels = [Kernel(f"launch_{index}") for index in range(kernels_per_call - 1)]
    return table, estimate(Call([*kernels, attention]), table, bw_model="large")


def l2_fraction(table, config, b, context, hq, hk, *, query_length=None):
    """Share of the attention working set that fits in the table's L2."""
    counts, union = visible_key_counts(config, context, query_length=query_length)
    working_set = 2 * b * len(counts) * hq * 128 * 2 + b * union * (2 * hk * 128 * 2 + 4)
    l2_bytes = load_ceilings(table)["l2_bytes"]
    return working_set, l2_bytes, min(1.0, l2_bytes / working_set)


def sol_report(table, cold, warm, config, b, context, hq, hk, timings, *, query_length=None):
    """Report the L2-reuse reference (warm) next to the cold-DRAM one."""
    working_set, l2_bytes, fraction = l2_fraction(
        table, config, b, context, hq, hk, query_length=query_length
    )
    result = warm
    return {
        "table": table,
        "regime": "warm_l2" if fraction == 1 else "partial_l2",
        "l2_fraction": fraction,
        "working_set_bytes": working_set,
        "l2_bytes": l2_bytes,
        "t_ref_us": result.t_ref_us,
        "limiting_resource": result.limiting_resource,
        "cold_t_ref_us": cold.t_ref_us,
        "warm_l2_t_ref_us": warm.t_ref_us,
        "candidate_graph_complete_utilization": asdict(
            result.utilization(timings["complete_vs_fixed_plan"]["candidate_median_us"])
        ),
        "candidate_eager_utilization": asdict(
            result.utilization(timings["eager_gate"]["candidate_median_us"])
        ),
        "ceiling_definition": SOL_DEFINITION.format(table=table),
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def measure(call, repetitions):
    # The same wall-clock protocol captures Python host planning and completed
    # GPU work. Synchronization is measurement scaffolding, outside each op.
    torch.cuda.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(repetitions):
        call()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter_ns() - begin) / 1000 / repetitions
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid timing")
    return elapsed


def graph_capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    begin = time.perf_counter_ns()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    stream.synchronize()
    return graph, (time.perf_counter_ns() - begin) / 1e6


def launch_count(call):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        call()
        torch.cuda.synchronize()
    events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not any(word in event.name.lower() for word in ("memcpy", "memset"))
    ]
    return {"kernels": len(events), "names": [event.name for event in events]}


def paired_timings(candidate, reference, *, calls):
    if calls < 100:
        raise ValueError("require >=100 calls per interval")
    self_pairs = [[measure(reference, calls), measure(reference, calls)] for _ in range(PAIRS)]
    epsilon = sorted(abs(first / second - 1) for first, second in self_pairs)[18]
    pairs = []
    for index in range(PAIRS):
        if index % 2:
            ref, cand = measure(reference, calls), measure(candidate, calls)
        else:
            cand, ref = measure(candidate, calls), measure(reference, calls)
        pairs.append([cand, ref])
    ratios = [cand / ref for cand, ref in pairs]
    median = statistics.median(ratios)
    return {
        "reference_self_pairs_us": self_pairs,
        "epsilon": epsilon,
        "candidate_reference_pairs_us": pairs,
        "ratios": ratios,
        "median_ratio": median,
        "candidate_median_us": statistics.median(pair[0] for pair in pairs),
        "reference_median_us": statistics.median(pair[1] for pair in pairs),
        "speed": "PASS" if median <= 1 + epsilon else "FAIL",
        "pairs": PAIRS,
        "calls_per_interval": calls,
    }


def timing_report(candidate, reference, complete_replay, fixed_replay, run_replay, *, calls):
    eager = paired_timings(candidate, reference, calls=calls)
    graphs = {}
    for name, replay in (("complete", complete_replay), ("run_only", run_replay)):
        comparison = paired_timings(replay, fixed_replay, calls=calls)
        graphs[f"{name}_vs_fixed_plan"] = {"is_gate": name == "complete", **comparison}
    return {
        **eager,
        "speed_eager": eager["speed"],
        "eager_gate": eager,
        "speed_graph": graphs["complete_vs_fixed_plan"]["speed"],
        "speed_graph_run_only": graphs["run_only_vs_fixed_plan"]["speed"],
        "gate_reference": GATE_REFERENCE,
        **graphs,
        "modes": {
            "candidate": "eager prepare+run (speed_eager gate)",
            "reference": "eager plan+run (speed_eager gate)",
            "graph_gate": "DF2 complete-op graph replay vs stock fixed-plan graph replay",
            "complete_vs_fixed_plan": "graph prepare+run vs fixed-plan graph run; speed_graph gate",
            "run_only_vs_fixed_plan": "graph run vs fixed-plan graph run; informational per-layer cost",
        },
    }


def provider_report(op):
    implementation = getattr(op, "_implementation", None)
    return {
        "provider": op.provider,
        "cute_tuning": {
            name: getattr(implementation, name, None)
            for name in (
                "splits",
                "key_tile",
                "stages",
                "key_groups",
                "short_union",
                "inline_merge",
                "min_chunk",
            )
        }
        if op.provider == "cute-splitkv"
        else None,
    }


def run_case(config, b, context, hq, hk, *, calls, warmup, query_length=None, provider=None):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    from tests._dflash_reference import draft_attention_reference

    qlen, causal, window = CONFIGS[config]
    qlen = qlen if query_length is None else query_length
    length = context + qlen
    geometry = {
        "batch_size": b,
        "query_length": qlen,
        "max_kv_tokens": length,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": 128,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
        "device": torch.device("cuda:0"),
    }
    torch.cuda.reset_peak_memory_stats()
    init_begin = time.perf_counter_ns()
    sizes = DraftAttention.workspace_size(**geometry, provider=provider)
    workspaces = [torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes]
    cpu_slots = torch.arange(b, dtype=torch.int32)
    cpu_positions = torch.arange(context, length, dtype=torch.int64).repeat(b)
    cpu_query_ptr = torch.arange(b + 1, dtype=torch.int32) * qlen
    cpu_token_ptr = torch.arange(b + 1, dtype=torch.int32) * length
    cpu_indices = torch.arange(b * length, dtype=torch.int32)
    cpu_last = torch.ones(b, dtype=torch.int32)
    slots, positions, token_ptr, token_idx = [
        t.cuda() for t in (cpu_slots, cpu_positions, cpu_token_ptr, cpu_indices)
    ]
    query_ptr, page_ptr, page_idx, last = [
        torch.empty_like(t, device="cuda")
        for t in (cpu_query_ptr, cpu_token_ptr, cpu_indices, cpu_last)
    ]
    op = DraftAttention(
        *workspaces,
        provider=provider,
        request_indices=slots,
        positions=positions,
        token_indptr=token_ptr,
        token_indices=token_idx,
        query_indptr=query_ptr,
        page_indptr=page_ptr,
        page_indices=page_idx,
        last_page_lengths=last,
        **{
            key: geometry[key]
            for key in (
                "query_length",
                "num_qo_heads",
                "num_kv_heads",
                "head_dim",
                "page_size",
                "dtype",
                "causal",
                "window",
            )
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(20261014)
    q = torch.randn((b * qlen, hq, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(
        (b * length, 1, hk, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    v = torch.randn(k.shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    out, reference_out, graph_out = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)

    def candidate():
        op.prepare()
        op.run(q=q, paged_kv_cache=(k, v), out=out)

    candidate()
    torch.cuda.synchronize()
    candidate_init_ms = (time.perf_counter_ns() - init_begin) / 1e6
    # Stock wrapper allocations stay outside timing; ordinary plan() remains
    # inside every reference call, including its CPU scheduling and uploads.
    reference_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    fixed_workspace = torch.zeros_like(reference_workspace)
    eager = BatchPrefillWithPagedKVCacheWrapper(reference_workspace, kv_layout="NHD", backend="fa2")
    fixed = BatchPrefillWithPagedKVCacheWrapper(
        fixed_workspace,
        kv_layout="NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(query_ptr),
        paged_kv_indptr_buf=torch.empty_like(page_ptr),
        paged_kv_indices_buf=torch.empty_like(page_idx),
        paged_kv_last_page_len_buf=torch.empty_like(last),
    )
    plan_args = {
        "qo_indptr": cpu_query_ptr,
        "paged_kv_indptr": cpu_token_ptr,
        "paged_kv_indices": token_idx,
        "paged_kv_last_page_len": cpu_last,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim_qk": 128,
        "page_size": 1,
        "causal": causal,
        "window_left": -1 if window is None else window - 1,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "disable_split_kv": False,
        # Complete public metadata uploads before reusing the native planner's
        # pinned staging on the next iteration. This stock-plan cost is timed.
        "non_blocking": False,
    }

    def plan():
        eager.plan(**plan_args)

    def reference():
        plan()
        eager.run(q=q, paged_kv_cache=(k, v), out=reference_out)

    def fixed_run():
        fixed.run(q=q, paged_kv_cache=(k, v), out=graph_out)

    reference()
    fixed.plan(**plan_args)
    fixed_run()
    expected = draft_attention_reference(
        q,
        k,
        v,
        slots=cpu_slots,
        positions=cpu_positions,
        token_indptr=cpu_token_ptr,
        token_indices=cpu_indices,
        query_length=qlen,
        causal=causal,
        window=window,
    )
    for actual in (out, reference_out, graph_out):
        torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)
    cand_graph, capture_ms = graph_capture(candidate)
    ref_graph, reference_capture_ms = graph_capture(fixed_run)
    # Per-layer cost when one prepare serves every layer of a round.
    run_graph, _ = graph_capture(lambda: op.run(q=q, paged_kv_cache=(k, v), out=out))
    for _ in range(warmup):
        candidate()
        cand_graph.replay()
        reference()
        ref_graph.replay()
        run_graph.replay()
    timings = timing_report(
        candidate, reference, cand_graph.replay, ref_graph.replay, run_graph.replay, calls=calls
    )
    stage_times = {
        "reference_plan_us": measure(plan, calls),
        "reference_fixed_plan_graph_us": measure(ref_graph.replay, calls),
        "candidate_prepare_eager_us": measure(op.prepare, calls),
        "candidate_run_graph_us": measure(run_graph.replay, calls),
    }
    counts = {
        "candidate_eager_complete": launch_count(candidate),
        "candidate_complete_graph": launch_count(cand_graph.replay),
        "candidate_prepare": launch_count(op.prepare),
        "candidate_run": launch_count(lambda: op.run(q=q, paged_kv_cache=(k, v), out=out)),
        "reference_eager_complete": launch_count(reference),
        "reference_plan": launch_count(plan),
        "reference_fixed_plan_graph": launch_count(ref_graph.replay),
    }
    if any(
        counts[name]["kernels"] < 1
        for name in (
            "candidate_eager_complete",
            "candidate_complete_graph",
            "reference_eager_complete",
            "reference_fixed_plan_graph",
        )
    ):
        raise RuntimeError("profiler did not observe CUDA kernels; launch counts are unqualified")
    sol = sol_estimate(
        torch.cuda.get_device_name(),
        config,
        b,
        context,
        hq,
        hk,
        counts["candidate_complete_graph"]["kernels"],
        query_length=qlen,
    )
    sol_fields = {
        "sol": None,
        "sol_reason": "No measured SOL table for this device, or the shared SOL helper is not installed",
    }
    if sol is not None:
        table, cold = sol
        _, warm = sol_estimate(
            torch.cuda.get_device_name(),
            config,
            b,
            context,
            hq,
            hk,
            counts["candidate_complete_graph"]["kernels"],
            l2_fraction=l2_fraction(table, config, b, context, hq, hk, query_length=qlen)[2],
            query_length=qlen,
        )
        sol_fields = {
            "sol": sol_report(
                table, cold, warm, config, b, context, hq, hk, timings, query_length=qlen
            )
        }
    return {
        **timings,
        **stage_times,
        **sol_fields,
        "correctness": "PASS",
        **provider_report(op),
        "kernels_per_call": counts,
        "workspace": op.workspace_report,
        "reference_float_workspace_bytes_each": reference_workspace.numel(),
        "candidate_first_use_ms": candidate_init_ms,
        "candidate_capture_ms": capture_ms,
        "reference_capture_ms": reference_capture_ms,
        "resident_peak_bytes": torch.cuda.max_memory_allocated(),
        "reference_non_blocking_plan": False,
        "calls_per_interval": calls,
        "warmup_calls": warmup,
        "pairs": PAIRS,
        "timing": "synchronized wall-clock, microseconds/call; alternating pair order",
        "sglang_comparator": {
            "status": "see dflash_draft_market.py",
            "comparators": ["flashinfer", "sglang", "vllm"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    selection.add_argument(
        "--queries",
        nargs="+",
        type=int,
        choices=QUERY_LENGTHS,
        help="run each Q with all three masks (instead of named default CONFIGS)",
    )
    parser.add_argument("--provider", choices=("auto", "cute", "flashinfer"), default="auto")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 512, 4096])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if (
        args.calls < 100
        or args.warmup < 20
        or any(b not in (1, 2, 3, 4) for b in args.batch_sizes)
        or min(args.contexts) < 0
    ):
        parser.error("require >=100 calls, >=20 warmup calls, B=1..4 and nonnegative contexts")
    available = torch.cuda.is_available()
    source = os.environ.get("CTIK_SOURCE_SHA", "")
    source = source if re.fullmatch(r"[0-9a-fA-F]{40}", source) else None
    emit(
        {
            "type": "manifest",
            "torch": torch.__version__,
            "flashinfer": package_version("flashinfer-python"),
            "triton": package_version("triton"),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if available else None,
            "capability": torch.cuda.get_device_capability() if available else None,
            "ctikernels_source_sha": source,
            "provider_requested": args.provider,
            "gate_reference": GATE_REFERENCE,
            "seed": 20261014,
            "dtype": "bfloat16",
            "page_size": 1,
            "atol": ATOL,
            "rtol": RTOL,
        }
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = (
        [(config, q) for q in args.queries for config in CONFIGS]
        if args.queries is not None
        else [(config, CONFIGS[config][0]) for config in args.configs]
    )
    for config, q in cases:
        _, causal, window = CONFIGS[config]
        boundary = window or 4096
        contexts = sorted(set(args.contexts + [boundary - 1, boundary, boundary + 1]))
        for b in args.batch_sizes:
            for context in contexts:
                for hq, hk in ((32, 8), (16, 4)):
                    row = {
                        "type": "measurement",
                        "mask": config,
                        "query_length": q,
                        "causal": causal,
                        "window": window,
                        "batch_size": b,
                        "context": context,
                        "num_qo_heads": hq,
                        "num_kv_heads": hk,
                        "head_dim": 128,
                    }
                    if not available or package_version("flashinfer-python") is None:
                        result = {
                            "speed": "N/A",
                            "correctness": "N/A",
                            "reason": "CUDA and FlashInfer required",
                        }
                    else:
                        try:
                            # Keep provider build chatter out of the JSONL stream.
                            with contextlib.redirect_stdout(sys.stderr):
                                result = run_case(
                                    config,
                                    b,
                                    context,
                                    hq,
                                    hk,
                                    calls=args.calls,
                                    warmup=args.warmup,
                                    query_length=q,
                                    provider=None if args.provider == "auto" else args.provider,
                                )
                        except (NotImplementedError, ImportError) as error:
                            result = {"speed": "N/A", "correctness": "N/A", "reason": str(error)}
                        except (AssertionError, RuntimeError, ValueError) as error:
                            result = {"speed": "FAIL", "correctness": "FAIL", "reason": str(error)}
                    result.setdefault("speed_eager", result["speed"])
                    result.setdefault("eager_gate", None)
                    result.setdefault("speed_graph", "N/A")
                    result.setdefault("speed_graph_run_only", "N/A")
                    result.setdefault("gate_reference", GATE_REFERENCE)
                    if "sol" not in result:
                        result.update(sol=None, sol_reason="Complete-op measurements unavailable")
                    emit(row | result)


if __name__ == "__main__":
    main()
