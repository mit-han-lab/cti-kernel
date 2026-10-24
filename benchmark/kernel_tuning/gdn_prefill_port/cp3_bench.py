#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired graphs: CP/non-CP, --pdl both on/off, or --fixup-only (200 replays)."""

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

from ctikernels.gdn.chunk_ws._cp import cp_plan
from ctikernels.gdn.chunk_ws.prefill import PortPrefill

root = Path(__file__).resolve().parents[3]
if not (root / "tests/gpu/_gdn_reference.py").is_file():
    root = Path.cwd()
sys.path.insert(0, str(root))
from benchmark.kernel_tuning.gdn_prefill_port.cp1_bench import percentile  # noqa: E402
from tests.gpu.test_gdn_port_prefill import make_case  # noqa: E402


def profile_replay(graph, segments):
    """CUDA activity times, sorted in stream launch order; no subprocess/ncu needed."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        graph.replay()
        torch.cuda.synchronize()
    events = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    # The replay contains only our kernels, but exclude profiler/graph driver
    # annotations if a Kineto version exposes them as CUDA events.
    kernels = [e for e in events if "kernel_cutlass" in e.name]
    labels = ["A", "fixup", "C", "commit"] if segments > 2 else ["A", "C", "commit"]
    if segments == 1:
        labels = ["full", "commit"]
    if len(kernels) == len(labels):
        return {
            label: {"us": e.time_range.elapsed_us(), "kernel": e.name}
            for label, e in zip(labels, kernels, strict=True)
        }
    # Some PyTorch versions omit CUDA graph nodes from FunctionEvents.
    # Kineto's Chrome activities retain them (same replay, no extra profiling).
    with tempfile.TemporaryDirectory(prefix=".cp3-profile-", dir=Path.cwd()) as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        activities = json.loads(path.read_text()).get("traceEvents", [])
    kernels = sorted(
        (e for e in activities if e.get("cat", "").lower() == "kernel"),
        key=lambda e: e["ts"],
    )
    if len(kernels) != len(labels):
        raise RuntimeError(
            f"profiler expected {len(labels)} kernels, got {[e['name'] for e in kernels]}"
        )
    return {
        label: {"us": e["dur"], "kernel": e["name"]}
        for label, e in zip(labels, kernels, strict=True)
    }


def benchmark(hv, length, target, pairs, baseline_op, cp_op, flush):
    x, weights, meta, base = make_case(cp_op.hk, hv, (length,))
    comparison = "pdl_off" if baseline_op.port_options.cp else "non_cp"
    pools = {name: tuple(t.clone() for t in base) for name in ("cp", comparison)}
    outputs = {
        name: torch.empty(length, hv * 128, device=x.device, dtype=x.dtype) for name in pools
    }

    def restore(name):
        for dst, src in zip(pools[name], base, strict=True):
            dst.copy_(src)

    graphs = {}
    for name, op in (("cp", cp_op), (comparison, baseline_op)):

        def launch(op=op, name=name):
            op(x, *weights.args(), *meta, *pools[name], out=outputs[name])

        for _ in range(3):
            restore(name)
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graphs[name] = graph
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))

    def sample(name):
        restore(name)
        flush.zero_()
        start.record()
        graphs[name].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000

    self_pairs = {}
    for name in graphs:
        values = [(sample(name), sample(name)) for _ in range(20)]
        self_pairs[name] = {
            "eps": percentile([abs(b / a - 1) for a, b in values], 0.95),
            "samples_us": values,
        }
    # Each iteration is AB; the full sequence is ABAB, with identical restores.
    timings = [(sample("cp"), sample(comparison)) for _ in range(pairs)]
    cp_us, comparison_us = zip(*timings, strict=True)
    ratio = statistics.median([a / b for a, b in timings])
    eps = max(v["eps"] for v in self_pairs.values())
    j, ell = cp_plan(length, hv, cp_op.num_sms, target)
    restore("cp")
    flush.zero_()
    torch.cuda.synchronize()
    breakdown = profile_replay(graphs["cp"], j)
    comparison_breakdown = None
    if comparison == "pdl_off":
        restore(comparison)
        flush.zero_()
        torch.cuda.synchronize()
        comparison_breakdown = profile_replay(graphs[comparison], j)
    return {
        "hk": cp_op.hk,
        "hv": hv,
        "tokens": length,
        "segments_target": target,
        "plan": {"J": j, "L": ell},
        "pdl": cp_op.port_options.pdl,
        "comparison": comparison,
        "cp_median_us": statistics.median(cp_us),
        f"{comparison}_median_us": statistics.median(comparison_us),
        f"ratio_cp_{comparison}": ratio,
        "eps": eps,
        "ratio_noise_band": [ratio * (1 - eps), ratio * (1 + eps)],
        "paired_samples_us": timings,
        "self_pairs": self_pairs,
        "launches": breakdown,
        "comparison_launches": comparison_breakdown,
        "gpu": torch.cuda.get_device_name(),
        "num_sms": cp_op.num_sms,
    }


def benchmark_fixup(hv):
    """Warm-cache fixup only: one kernel per replay, J-2 sequential steps."""
    from ctikernels.gdn.chunk_ws._cp_fixup import compile_fixup

    cc = torch.cuda.get_device_capability()
    fixup = compile_fixup(hv, f"--enable-tvm-ffi --gpu-arch sm_{cc[0]}{cc[1]}a")
    rows = []
    for segments in range(3, 9):
        w = torch.randn(3, segments, hv, 128, 128, device="cuda", dtype=torch.float32) * 0.02
        w[2].add_(torch.eye(128, device="cuda") * 0.8)
        for _ in range(3):
            fixup(w, segments)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fixup(w, segments)
        for _ in range(10):
            graph.replay()
        # S[1], M, N are immutable inputs: repeated replays compute identical
        # results without restores or allocation inside the measured region.
        events = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(2)) for _ in range(200)]
        # Initialize event handles before measurement.
        for start, end in events:
            start.record()
            end.record()
        torch.cuda.synchronize()
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        events[-1][1].synchronize()
        samples = [start.elapsed_time(end) * 1000 for start, end in events]
        elapsed = statistics.median(samples)
        row = {
            "kernel": "fixup",
            "hv": hv,
            "J": segments,
            "steps": segments - 2,
            "replays": len(events),
            "median_us": elapsed,
            "us_per_step": elapsed / (segments - 2),
            "p95_us": percentile(samples, 0.95),
            "gpu": torch.cuda.get_device_name(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--hk", type=int, default=16, help="key heads (8 for TP2)")
    parser.add_argument("--hv", default="32,48", help="comma-separated value head counts")
    parser.add_argument("--segments", default="planner", help="sweep, e.g. planner,2,3,4,5")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument(
        "--pdl",
        choices=("on", "off", "both"),
        default="on",
        help="on/off compare CP with non-CP; both pairs CP PDL on/off with the same plan",
    )
    parser.add_argument(
        "--fixup-only", action="store_true", help="time fixup alone for J=3..8, 200 graph replays"
    )
    args = parser.parse_args()
    try:
        lengths = [int(s) for s in args.lengths.split(",")]
        heads = [int(s) for s in args.hv.split(",")]
        targets = [None if s == "planner" else int(s) for s in args.segments.split(",")]
        if (
            min(lengths) < 1
            or any(h not in (16, 24, 32, 48) or h % args.hk for h in heads)
            or args.pairs < 1
        ):
            raise ValueError
        if any(s is not None and s < 1 for s in targets):
            raise ValueError
    except ValueError:
        parser.error(
            "positive lengths/pairs/segments and hv in {16,24,32,48}, a multiple of hk, required"
        )
    if args.fixup_only:
        rows = [row for hv in heads for row in benchmark_fixup(hv)]
        print("Hv   J steps replays  fixup us   us/step")
        for row in rows:
            print(
                f"{row['hv']:2} {row['J']:3} {row['steps']:5} {row['replays']:7} "
                f"{row['median_us']:9.3f} {row['us_per_step']:9.3f}"
            )
        return
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    rows = []
    for hv in heads:
        baseline = PortPrefill(args.hk, hv, "cuda", cp=False)
        # Keep the expensive compile and workspace warm across all lengths.
        for target in targets:
            op = PortPrefill(args.hk, hv, "cuda", cp_segments=target, pdl=args.pdl != "off")
            comparator = (
                PortPrefill(args.hk, hv, "cuda", cp_segments=target, pdl=False)
                if args.pdl == "both"
                else baseline
            )
            for length in lengths:
                row = benchmark(hv, length, target, args.pairs, comparator, op, flush)
                rows.append(row)
                print(json.dumps(row), flush=True)
    label = "PDL-off" if args.pdl == "both" else "non-CP"
    print(
        f"Hv      T target   J      L    CP us {label:>7} us   ratio     eps   A/fixup/C/commit us"
    )
    for row in rows:
        times = "/".join(
            f"{row['launches'].get(k, {}).get('us', 0):.1f}" for k in ("A", "fixup", "C", "commit")
        )
        print(
            f"{row['hv']:2} {row['tokens']:6} {str(row['segments_target'] or 'auto'):>6} "
            f"{row['plan']['J']:3} {row['plan']['L']:6} {row['cp_median_us']:8.2f} "
            f"{row[row['comparison'] + '_median_us']:9.2f} "
            f"{row['ratio_cp_' + row['comparison']]:7.3f} {row['eps']:7.3f}   {times}"
        )


if __name__ == "__main__":
    main()
