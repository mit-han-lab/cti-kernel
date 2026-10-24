# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Single-device paired timings. Run with python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from .market import NAMES, Provider, failure_reason
from .reference import (
    MhcConfig,
    error_stats,
    exit_fp64,
    exit_torch_bf16,
    make_state,
    make_weights,
    output_dict,
)

ROWS = (1, 2, 3, 4, 8, 128, 256, 1024, 8192)
DEFAULT_ROWS = (1, 2, 3, 4, 128, 1024, 8192)


def arguments(pair=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers", nargs="+", choices=NAMES, default=["torch", "vllm", "sglang", "flashinfer"]
    )
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--sglang-python", type=Path)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4) if pair else ROWS,
        default=[1, 2, 3, 4] if pair else list(DEFAULT_ROWS),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("eager", "graph"), default=["eager", "graph"]
    )
    parser.add_argument("--weight-sets", type=int, default=16)
    parser.add_argument("--hot", action="store_true")
    parser.add_argument("--calls", type=int, default=86)
    if not pair:
        parser.add_argument(
            "--spacer-mib",
            type=float,
            default=0,
            help="unrelated MiB read by a non-PDL add between exits",
        )
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--count-kernels", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def validate(parser, args):
    if min(args.calls, args.pairs, args.weight_sets) <= 0:
        parser.error("calls, pairs, and weight-sets must be positive")
    spacer_mib = getattr(args, "spacer_mib", 0)
    if not math.isfinite(spacer_mib) or spacer_mib < 0:
        parser.error("spacer-mib must be finite and nonnegative")
    if len(set(args.providers)) != len(args.providers):
        parser.error("provider names must be unique")


def rotation_count(cfg, T, requested, l2_bytes, hot=False):
    # Count only fn + R: other inputs/outputs further enlarge the working set.
    per_set = cfg.mult * (cfg.mult + 2) * cfg.mult * cfg.hidden * 4
    per_set += T * cfg.mult * cfg.hidden * 2
    if not hot and l2_bytes <= 0:
        raise ValueError("device L2 size is unavailable; cannot establish cold regime")
    count = 1 if hot else max(requested, (2 * l2_bytes) // per_set + 1)
    return count, count * per_set


def setup_row(cfg, T, args, l2_bytes):
    count, working_set = rotation_count(cfg, T, args.weight_sets, l2_bytes, args.hot)
    if count > args.calls:
        raise ValueError(f"cold rotation needs {count} sets; increase --calls to at least {count}")
    weights = [make_weights(cfg, args.seed + i, "cuda") for i in range(count)]
    states = [make_state(cfg, T, args.seed + 10000 + i, "cuda") for i in range(count)]
    return (
        weights,
        states,
        {
            "weight_sets": count,
            "fn_plus_R_bytes": working_set,
            "l2_bytes": l2_bytes,
            "regime": "hot" if args.hot else "cold",
        },
    )


def inputs(state):
    return tuple(state[key] for key in ("x", "R", "post", "comb"))


def check(cfg, weights, state, result):
    """Chunk the FP64 oracle on CPU so even prefill never needs a huge FP64 GPU buffer."""
    values = output_dict(result)
    # Evaluate the floor on the same device and full batch as the provider.
    # Otherwise CPU/GPU GEMM reduction order can make even torch fail its own floor.
    floor_values = exit_torch_bf16(cfg, weights, *inputs(state))
    w = {key: value.detach().cpu() for key, value in weights.items()}
    combined, floor = {}, {}
    T = state["x"].shape[0]
    for start in range(0, T, 128):
        end = min(T, start + 128)
        args = [value[start:end].detach().cpu() for value in inputs(state)]
        oracle = exit_fp64(cfg, w, *args)
        candidate = {key: value[start:end] for key, value in values.items()}
        for dest, metrics in (
            (combined, error_stats(candidate, oracle)),
            (floor, error_stats({key: floor_values[key][start:end] for key in values}, oracle)),
        ):
            for key, stats in metrics.items():
                accum = dest.setdefault(key, {"max_abs": 0.0, "rms": 0.0})
                for metric in ("max_abs", "rms"):
                    if not math.isfinite(stats[metric]):
                        accum[metric] = math.inf
                    elif metric == "max_abs":
                        accum[metric] = max(accum[metric], stats[metric])
                    else:
                        accum[metric] += stats[metric] ** 2 * (end - start) / T
    for metrics in (combined, floor):
        for stats in metrics.values():
            stats["rms"] = math.sqrt(stats["rms"])
    finite = all(math.isfinite(v) for stats in combined.values() for v in stats.values())
    passes = finite and all(combined[k][m] <= floor[k][m] for k in combined for m in combined[k])
    # Strict JSON: nonfinite error is a failed check, represented by null.
    for metrics in (combined, floor):
        for stats in metrics.values():
            for key, value in stats.items():
                if not math.isfinite(value):
                    stats[key] = None
    return {
        "errors": combined,
        "torch_bf16_floor": floor,
        "finite": finite,
        "provider_le_torch_bf16": passes,
        "unexposed_outputs": sorted({"pre", "layer_input"} - values.keys()),
    }


def prepare_provider(name, cfg, weights, states, args):
    provider = Provider(name, args.vllm_root, args.sglang_python)
    functions = []
    for w, state in zip(weights, states, strict=True):
        fn = provider.prepare(cfg, w, state["x"].shape[0], "cuda")
        if fn is None:
            return provider, None
        functions.append(fn)
    return provider, functions


def make_spacer(mib, device="cuda"):
    """One ordinary ATen pointwise launch; inputs/outputs never alias exit data."""
    if mib == 0:
        return None
    elements = max(1, math.ceil(mib * 1024 * 1024 / 4))
    source = torch.zeros(elements, dtype=torch.float32, device=device)
    target = torch.empty_like(source)

    def spacer():
        torch.add(source, 0, out=target)

    spacer.read_bytes = elements * 4
    return spacer


def spacer_block(spacer, calls):
    def block():
        for _ in range(calls - 1):
            spacer()

    return block


def block_calls(functions, states, calls, spacer=None):
    def block():
        result = None
        for i in range(calls):
            if spacer is not None and i > 0:
                spacer()
            index = i % len(states)
            result = functions[index](*inputs(states[index]))
        return result

    return block


def make_launcher(block, mode):
    # Warm up on the capture stream: imports, compilation, and allocations finish
    # before capture. Holding block also keeps its input buffers alive.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            block()
    stream.synchronize()
    if mode == "eager":
        return block
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = block()
    torch.cuda.current_stream().wait_stream(stream)

    def replay():
        graph.replay()
        return result

    replay.keepalive = (block, graph, stream)
    return replay


def timed_sample(launch, repeats, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * calls)


def bootstrap(ratios, seed, resamples=5000):
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples))
    return [draws[int(0.025 * resamples)], draws[int(0.975 * resamples)]]


def measure(launchers, args, timer=timed_sample):
    if not launchers:
        return {}
    pilots = {name: timer(fn, 1, args.calls) for name, fn in launchers.items()}
    repeats = {
        name: max(1, math.ceil(3000 / (args.calls * max(us, 0.001)))) for name, us in pilots.items()
    }
    rng, samples = random.Random(args.seed), []
    for _ in range(args.pairs):
        order = list(launchers)
        rng.shuffle(order)
        samples.append({name: timer(launchers[name], repeats[name], args.calls) for name in order})
    medians = {name: statistics.median(row[name] for row in samples) for name in launchers}
    fastest = min(medians, key=medians.get)
    results = {}
    for name, median in medians.items():
        ratios = [row[name] / row[fastest] for row in samples]
        results[name] = {
            "median_us": median,
            "median_step_us": median * args.calls,
            "repeats": repeats[name],
            "fastest_provider": fastest,
            "median_ratio": statistics.median(ratios),
            "ratio_ci95": bootstrap(ratios, args.seed),
            "samples_us": [row[name] for row in samples],
        }
    return results


def count_kernels(launch, calls):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        launch()
        torch.cuda.synchronize()
    # CUDA activities include memcpy/memset as well as compute kernels. Disclose
    # both counts; no trace file (which could embed local paths) is emitted.
    events = [
        event for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    compute = [event for event in events if not event.name.lower().startswith(("memcpy", "memset"))]
    return {
        "profiled_kernels_per_exit": len(compute) / calls,
        "profiled_cuda_activities_per_exit": len(events) / calls,
    }


def print_row(T, mode, name, result):
    if result["status"] != "available":
        print(f"T={T:4d} {mode:5s} {name:18s} {result['status']}", flush=True)
        return
    timing, correct = result["timing"], result.get("correctness", {})
    lo, hi = timing["ratio_ci95"]
    print(
        f"T={T:4d} {mode:5s} {name:18s} {timing['median_us']:10.3f} us "
        f"step={timing['median_step_us']:.3f} us "
        f"ratio={timing['median_ratio']:.3f} [{lo:.3f},{hi:.3f}] "
        f"vs={timing['fastest_provider']} floor={correct.get('provider_le_torch_bf16', 'n/a')}",
        flush=True,
    )


def save_report(report, out):
    if out:
        out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = arguments()
    args = parser.parse_args()
    validate(parser, args)
    if not torch.cuda.is_available():
        parser.error("benchmark requires CUDA; reference.py and CPU tests do not")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    cfg = MhcConfig()
    props = torch.cuda.get_device_properties(0)
    l2_bytes = getattr(props, "L2_cache_size", 0)
    report = {
        "config": asdict(cfg),
        "calls": args.calls,
        "pairs": args.pairs,
        "spacer_mib": args.spacer_mib,
        "spacer": {},
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability()),
        "results": [],
    }
    with torch.inference_mode():
        spacer = make_spacer(args.spacer_mib)
        if spacer is not None:
            for mode in args.modes:
                if args.calls > 1:
                    launch = make_launcher(spacer_block(spacer, args.calls), mode)
                    timing = measure({"spacer": launch}, args)["spacer"]
                    step_us = timing["median_step_us"]
                    per_spacer_us = step_us / (args.calls - 1)
                else:
                    launch = make_launcher(spacer, mode)
                    per_spacer_us = statistics.median(
                        timed_sample(launch, 10, 1) for _ in range(args.pairs)
                    )
                    step_us = 0.0
                report["spacer"][mode] = {
                    "read_bytes": spacer.read_bytes,
                    "write_bytes": spacer.read_bytes,
                    "count_per_step": args.calls - 1,
                    "standalone_us": per_spacer_us,
                    "standalone_step_us": step_us,
                    "amortized_us_per_exit": step_us / args.calls,
                }
                print(
                    f"spacer {mode}: {spacer.read_bytes / 2**20:.3f} MiB read + write, "
                    f"{per_spacer_us:.3f} us/launch, "
                    f"{step_us / args.calls:.3f} us/exit amortized",
                    flush=True,
                )
        for T in args.rows:
            weights, states, regime = setup_row(cfg, T, args, l2_bytes)
            prepared = {}
            for name in args.providers:
                provider, functions = prepare_provider(name, cfg, weights, states, args)
                correct = None
                if functions is not None:
                    try:
                        correct = check(
                            cfg, weights[0], states[0], functions[0](*inputs(states[0]))
                        )
                        torch.cuda.synchronize()
                    except Exception as exc:  # noqa: BLE001 - optional providers report unavailable
                        provider.status = "unavailable: " + failure_reason(
                            exc, "warmup/correctness"
                        )
                        functions = None
                prepared[name] = provider, functions, correct
            for mode in args.modes:
                launchers, results = {}, {}
                for name, (provider, functions, correct) in prepared.items():
                    result = {
                        "status": provider.status,
                        "metadata": dict(provider.metadata),
                        "correctness": correct,
                    }
                    results[name] = result
                    if functions is None:
                        continue
                    try:
                        launchers[name] = make_launcher(
                            block_calls(functions, states, args.calls, spacer), mode
                        )
                        # Check captured execution too, including output lifetime/aliasing.
                        output = launchers[name]()
                        index = (args.calls - 1) % len(states)
                        result["correctness"] = check(cfg, weights[index], states[index], output)
                        if args.count_kernels:
                            result["metadata"].update(count_kernels(launchers[name], args.calls))
                    except Exception as exc:  # noqa: BLE001 - optional providers report unavailable
                        result["status"] = "unavailable: " + failure_reason(exc, "capture/warmup")
                        launchers.pop(name, None)
                timings = measure(launchers, args)
                for name, result in results.items():
                    if name in timings:
                        result["timing"] = timings[name]
                        if spacer is not None:
                            cost = report["spacer"][mode]["amortized_us_per_exit"]
                            result["timing"]["spacer_amortized_us"] = cost
                            result["timing"]["spacer_subtracted_us"] = (
                                timings[name]["median_us"] - cost
                            )
                    print_row(T, mode, name, result)
                report["results"].append(
                    {
                        "T": T,
                        "mode": mode,
                        "decode": T <= 4,
                        "gated": False,
                        **regime,
                        "providers": results,
                    }
                )
                save_report(report, args.out)
                del launchers
            del prepared, weights, states
    save_report(report, args.out)


if __name__ == "__main__":
    main()
