# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordinary seeded sampling versus FlashInfer, with paired timing and SOL estimates.

Run with --device {rtx5090,rtx-pro-6000,gb10}; SOL uses the shared
benchmark.sol helper and its measured ceilings, with cache residency. Each
graph contains 20 calls; 20 rotated rounds measure medians of five replays per path. FlashInfer is imported only on GPU;
this standalone harness never loads an inference engine. JSONL goes to stdout.
"""

import argparse
import contextlib
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.sol.sol import DEVICES, Call, Kernel, estimate, load_ceilings
from ctikernels.sampling import RequestLocalSampling

ROUNDS = 20
CALLS = 20
REPEATS = 5
T_CRITICAL = 2.093024054408263  # two-sided 95%, df=19


def paired_summary(ratios, self_pairs):
    """95% Student-t interval of 20 paired log ratios; p95 relative self noise."""
    if len(ratios) != ROUNDS or len(self_pairs) != ROUNDS:
        raise ValueError("requires twenty ratios and self-pairs")
    values = [*ratios, *(v for pair in self_pairs for v in pair)]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("timings and ratios must be positive and finite")
    epsilon = sorted(max(a / b, b / a) - 1 for a, b in self_pairs)[18]
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    margin = T_CRITICAL * statistics.stdev(logs) / math.sqrt(ROUNDS)
    lo, hi = math.exp(mean - margin), math.exp(mean + margin)
    return {"gm": math.exp(mean), "lo": lo, "hi": hi, "epsilon": epsilon, "claim": lo > 1 + epsilon}


def sol_reference(device, b, v, launches):
    """Shared-helper SOL for one sample call as a PDL chain of `launches` kernels.

    Useful traffic: BF16 logits read (2 B), FP32 probabilities written (4 B) and
    read back (4 B) per element. The graph replays the same logits, so when the
    footprint (logits + probabilities, 6 B per element) fits in L2 every byte is
    served at the L2 ceiling; otherwise all bytes go to DRAM. The remaining
    launches cost the PDL graph-node floor.
    """
    table = load_ceilings(device)
    elements = b * v
    resident = elements * (2 + 4) <= table["l2_bytes"]
    cache = "warm" if resident else "cold"

    def kernel(name, dram, mode):
        return Kernel(
            name,
            dram_bytes=dram,
            l2_resident_bytes=dict(dram) if resident else {},
            cache_state=cache,
            dram_mode=mode,
            pdl=True,
        )

    kernels = [
        kernel(
            "probabilities", {"activations": 2 * elements, "intermediates": 4 * elements}, "copy"
        ),
        kernel("draw", {"intermediates": 4 * elements}, "read"),
        *(kernel(f"launch_{i}", {}, "read") for i in range(max(launches - 2, 0))),
    ]
    result = estimate(Call(kernels, mode="pdl_chain"), table, bw_model="large")
    return {
        "sol_us": result.t_ref_us,
        "sol_residency": "l2" if resident else "dram",
        "sol_limiting_resource": result.limiting_resource,
        "ideal_bytes": elements * (2 + 4 + 4),
        "sol_ceilings": table["source"],
    }


def emit(**record):
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def count_launches(call):
    """Actual eager kernel events; missing CUPTI is reported as unavailable."""
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            call()
            torch.cuda.synchronize()
        kernels = [
            e
            for e in profile.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and not e.name.lower().startswith(("memcpy", "memset"))
        ]
        return {
            "kernel_launches": len(kernels) if kernels else None,
            "launch_count_source": "torch.profiler",
        }
    except (RuntimeError, AttributeError):
        return {"kernel_launches": None, "launch_count_source": "profiler unavailable"}


class Timer:
    def __init__(self, call, *, allow_eager=False):
        self.call = call
        self.graph = None
        self.mode = "graph"
        self.capture_error = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(CALLS):
                    call()
            self.graph = graph
        except RuntimeError as error:
            if not allow_eager:
                raise
            self.mode, self.capture_error = "eager", type(error).__name__
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def measure(self):
        times = []
        for _ in range(REPEATS):
            self.start.record()
            if self.graph is not None:
                self.graph.replay()
            else:
                for _ in range(CALLS):
                    self.call()
            self.end.record()
            self.end.synchronize()
            times.append(self.start.elapsed_time(self.end) * 1000 / CALLS)
        return statistics.median(times)


def flashinfer_calls(fi, logits, temperatures, k, p, mode, min_p=None):
    def engine():
        probs = fi.softmax(logits, temperatures)
        if mode in ("min_p", "all"):
            if mode == "all":
                probs = fi.top_k_renorm_probs(probs, k)
                probs = fi.top_p_renorm_probs(probs, p)
            return fi.min_p_sampling_from_probs(probs, min_p)
        if mode == "none":
            return fi.sampling_from_probs(probs)
        if mode == "top_k":
            return fi.top_k_sampling_from_probs(probs, k)
        if mode == "top_p":
            return fi.top_p_sampling_from_probs(probs, p)
        return fi.top_k_top_p_sampling_from_probs(probs, k, p)

    def from_logits():
        if mode in ("min_p", "all"):
            return engine()  # No fused logits path: same SGLang probability pipeline.
        if mode == "top_p":
            return fi.top_p_sampling_from_probs(fi.softmax(logits, temperatures), p)
        scaled = logits.float() / temperatures[:, None]
        if mode == "none":
            return fi.sampling_from_logits(scaled)
        return fi.top_k_top_p_sampling_from_logits(scaled, k, p if mode == "both" else 1.0)

    return engine, from_logits


def benchmark_case(fi, v, b, mode, temperature, k, p, hint, device):
    op = RequestLocalSampling("cuda", v)
    generator = torch.Generator(device="cuda").manual_seed(42)
    logits = (torch.randn(b, v, generator=generator, device="cuda") * 3).bfloat16()
    temperatures = torch.full((b,), temperature, device="cuda")
    top_k = torch.full((b,), k, dtype=torch.int32, device="cuda")
    top_p = torch.full((b,), p, device="cuda")
    min_p = torch.full((b,), 0.05, device="cuda") if mode in ("min_p", "all") else None
    args = {
        "min_p": min_p,
        "request_indices": torch.arange(b, dtype=torch.int32, device="cuda"),
        "seeds": torch.tensor([[42, 0]], dtype=torch.uint32, device="cuda").repeat(b, 1),
        "positions": torch.arange(b, dtype=torch.int64, device="cuda"),
        "workspace": torch.empty(op.workspace_size(b), dtype=torch.uint8, device="cuda"),
        "out_tokens": torch.empty(b, dtype=torch.int32, device="cuda"),
        "filters": hint,
    }

    def ours():
        op.sample(logits, temperatures, top_k, top_p, **args)

    engine, from_logits = flashinfer_calls(fi, logits, temperatures, top_k, top_p, mode, min_p)
    calls = {"ours": ours, "fi_engine": engine, "fi_logits": from_logits}
    with contextlib.redirect_stdout(sys.stderr):
        timers = {name: Timer(call, allow_eager=name != "ours") for name, call in calls.items()}
        launches = {name: count_launches(call) for name, call in calls.items()}
    timers["ours_again"] = timers["ours"]
    names = list(timers)
    rounds = []
    for i in range(ROUNDS):
        order = names[i % 4 :] + names[: i % 4]
        with contextlib.redirect_stdout(sys.stderr):
            times = {name: timers[name].measure() for name in order}
        rounds.append(times)
    best = min(
        ("fi_engine", "fi_logits"), key=lambda name: statistics.median(r[name] for r in rounds)
    )
    ratios = [r[best] / r["ours"] for r in rounds]
    summary = paired_summary(ratios, [(r["ours"], r["ours_again"]) for r in rounds])
    metadata = {
        "vocab": v,
        "batch": b,
        "mode": mode,
        "filters": hint,
        "dtype": "bfloat16",
        "temperature": temperature,
        "top_k": k,
        "top_p": p,
        "min_p": 0.05 if min_p is not None else None,
        "fi_logits_path": "softmax -> [top_k -> top_p renorm] -> min_p_sampling_from_probs"
        if min_p is not None
        else "native logits (top_p uses softmax)",
    }
    for i, (times, ratio) in enumerate(zip(rounds, ratios, strict=True)):
        emit(record="round", **metadata, round=i, us=times, fi_best=best, log_ratio=math.log(ratio))
    emit(
        record="summary",
        **metadata,
        **summary,
        fi_best=best,
        median_us={name: statistics.median(r[name] for r in rounds) for name in names},
        timing_modes={name: timer.mode for name, timer in timers.items()},
        capture_errors={name: timer.capture_error for name, timer in timers.items()},
        launches=launches,
        expected_launches=op.launch_counts[hint],
        workspace_bytes=op.workspace_size(b),
        rounds=ROUNDS,
        calls_per_graph=CALLS,
        samples_per_round=REPEATS,
        ci="95% paired log-ratio t, df=19",
        epsilon_method="nearest-rank p95 relative ours/ours-again noise",
        fi_selection="path with faster median across rounds",
        **sol_reference(
            device, b, v, launches["ours"]["kernel_launches"] or op.launch_counts[hint]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICES, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        emit(status="unavailable", reason="CUDA unavailable")
        return
    with contextlib.redirect_stdout(sys.stderr):
        fi = importlib.import_module("flashinfer.sampling")
    for v in (151936, 248320):
        for b in (1, 2, 3, 4):
            for mode, t, k, p in (
                ("none", 0.7, v, 1.0),
                ("top_k", 0.7, 50, 1.0),
                ("top_p", 0.7, v, 0.9),
                ("both", 0.6, 20, 0.95),
                ("min_p", 0.7, v, 1.0),
                ("all", 0.6, 20, 0.95),
            ):
                matching_hint = {"min_p": "none", "all": "both"}.get(mode, mode)
                for hint in (matching_hint, "auto"):
                    benchmark_case(fi, v, b, mode, t, k, p, hint, args.device)


if __name__ == "__main__":
    main()
