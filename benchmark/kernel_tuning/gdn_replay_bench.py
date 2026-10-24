# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CUDA-graph conv-through-output windows; 20 interleaved pairs and 20 self-pairs.

All measured calls include layout preparation, status begin/end and periodic
flush. Status records are retained until host settlement between graph trials.
No separator subtraction. One graph trial contains at least 256 complete windows
across the cold rotation. Missing comparators and budget truncation are explicit.
"""

import argparse
import gc
import math
import statistics
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmark.kernel_tuning.gdn_replay_common import (
    Budget,
    BudgetExpired,
    Product,
    Replay,
    allocate,
    emit,
    manifest,
)
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_replay_sol import report as sol_report
from ctikernels.gdn.replay.cute import select

PROVIDERS = ("cute", "triton", "product", "sglang_default", "sglang_replay", "official")


def p95(values):
    if len(values) != 20:
        raise ValueError("exactly 20 pairs are required")
    return sorted(values)[18]


def pair_stats(pairs, self_pairs):
    if len(pairs) != 20 or len(self_pairs) != 20:
        raise ValueError("cannot qualify incomplete pairs")
    if any(not math.isfinite(t) or t <= 0 for pair in pairs + self_pairs for t in pair):
        raise ValueError("invalid measured duration")
    epsilon = p95([abs(a / b - 1) for a, b in self_pairs])
    ratios = [a / b for a, b in pairs]
    median = statistics.median(ratios)
    return {
        "pairs_us": pairs,
        "self_pairs_us": self_pairs,
        "ratios": ratios,
        "epsilon": epsilon,
        "median_ratio": median,
        "passed": median >= 1 - epsilon,
    }


def external_pin(root, revision):
    if not root or not Path(root).is_dir():
        raise ImportError("comparator checkout not available")
    result = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    sha = result.stdout.strip()
    if result.returncode:
        # Read-only mirrors without .git record "<sha> <date>" in REVISION.
        for marker in (Path(root) / "REVISION", Path(root).parent / "REVISION"):
            if marker.is_file():
                sha, result = marker.read_text().split()[0], None
                break
    if (result is not None and result.returncode) or not sha.startswith(revision):
        raise ImportError("comparator checkout revision does not match pin")
    return sha


class WindowGraph:
    def __init__(self, provider, hk, hv, b, window, copies, args, budget):
        self.provider, self.calls = provider, window * copies * math.ceil(256 / copies)
        self.window, self.b, self.hk, self.hv = window, b, hk, hv
        self.replay = provider in ("cute", "triton")
        slots = copies * b
        weights, xs, initial, history = case_inputs(
            hk, hv, b, slots, window, "ordinary", 11, "cuda"
        )
        weights = weights.args()
        self.groups = [
            torch.arange(i * b, (i + 1) * b, dtype=torch.int32, device="cuda")
            for i in range(copies)
        ]
        self.positions = torch.zeros(slots, dtype=torch.int64, device="cuda")
        self.xs = xs
        self.statuses = []
        if self.replay:
            config = None
            if provider == "cute":
                config = select(
                    torch.cuda.get_device_name(),
                    b,
                    mode=args.mode,
                    geometry=(hk, hv),
                    window=window,
                )
                if args.value_tile is not None:
                    config = replace(config, value_tile=args.value_tile)
                if args.speculative_ring is not None:
                    config = replace(config, speculative_ring=args.speculative_ring)
                if args.warps is not None:
                    config = replace(config, warps=args.warps)
            runner = Replay(
                provider,
                hk,
                hv,
                b,
                window,
                16,
                weights,
                initial[:b],
                history[:b],
                self.positions[:b],
                num_slots=slots,
                cute_config=config,
            )
            runner.op.handoff(
                canonical_state=initial,
                history=history,
                next_positions=self.positions,
                slot_indices=torch.arange(slots, dtype=torch.int32, device="cuda"),
                state=runner.state,
                status=runner.status,
            )
            # Retain one invocation record per captured call; never overwrite an unsettled one.
            runner.op.end(status=runner.status)
            result = runner.op.read_status_copy(*runner.status.copy_to_host(), expected_sequence=1)
            runner.op.settle(status=runner.status, result=result)
            self.controls = runner.op.bind_graph_buffers(
                buffers=allocate(
                    runner.op.graph_buffer_spec(batch_size=b, depth=self.calls), "cuda"
                )
            )
            self.controls.buffers["sequence"].copy_(
                torch.arange(2, self.calls + 2, dtype=torch.int64, device="cuda")
            )
            self.statuses = [self.controls.status(i) for i in range(self.calls)]
            self.outputs = torch.empty(self.calls, b, hv * 128, dtype=torch.bfloat16, device="cuda")
        elif provider == "product":
            runner = Product(hk, hv, weights, initial, history)
        else:
            from benchmark.kernel_tuning.gdn_replay_external import External

            runner = External(
                provider,
                hk,
                hv,
                window,
                weights,
                initial,
                history,
                args.sglang_root,
                args.replayssm_root,
            )
        self.runner = runner
        calls = []
        for index in range(self.calls):
            phase = (index // copies) % window
            group = index % copies
            slots_t = self.groups[group]
            pos = self.positions[group * b : (group + 1) * b]
            x = xs[phase]
            if self.replay:
                status = self.statuses[index]
                static_slots = self.controls.buffers["slot_indices"][index]
                static_slots.copy_(slots_t)
                static_pos = self.controls.buffers["positions"][index]
                out = self.outputs[index]

                def call(
                    x=x,
                    slots_t=static_slots,
                    pos=pos,
                    static_pos=static_pos,
                    out=out,
                    status=status,
                ):
                    if provider != "cute":
                        static_pos.copy_(pos)
                        runner.op.begin(status=status, layer_weights=(runner.weights,))
                    runner.op.step(
                        x,
                        *runner.weights,
                        slot_indices=slots_t,
                        positions=pos if provider == "cute" else static_pos,
                        state=runner.state,
                        out=out,
                        status=status,
                        workspace=runner.workspace,
                    )
                    if provider != "cute":
                        runner.op.end(status=status)
                    pos.add_(1)
            elif provider == "product":

                def call(x=x, slots_t=slots_t):
                    return runner.op(x, *runner.weights, slots_t, runner.state, runner.history)
            else:
                call = runner.call(x, slots_t, phase)
            calls.append(call)
        budget.check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # All phases and rotation groups, a full canonical window, before capture.
            for call in calls[: window * copies]:
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if self.replay:
            self.settle(window * copies)
        budget.check()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for call in calls:
                call()
        self.start, self.end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self.kernels_per_call = None
        # Profile one actual replay, outside all timing pairs; count GPU kernel events.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            self.graph.replay()
            torch.cuda.synchronize()
        count = sum(
            e.device_type == torch.autograd.DeviceType.CUDA
            and e.name not in ("[memory]",)
            and "memcpy" not in e.name.lower()
            and "memset" not in e.name.lower()
            for e in profile.events()
        )
        self.kernels_per_call = count / self.calls
        self.settle()

    def settle(self, count=None):
        # Trusted CuTe steps write no S1 status; correctness is the GPU tests' job.
        if not self.replay or getattr(self.runner.op, "checks", "strict") == "trusted":
            return
        count = self.calls if count is None else count
        # One batched existing-completion read; no device reads in timed serving work.
        records = self.controls.buffers["records"].cpu()
        validity = self.controls.buffers["validity"].cpu()
        seqs = self.controls.buffers["sequence"].cpu()
        for i in range(count):
            status = self.statuses[i]
            result = self.runner.op.read_status_copy(
                records[:, i], validity[:, i], expected_sequence=int(seqs[i])
            )
            if not result.ok:
                raise RuntimeError("timed replay status failed")
            self.runner.op.settle(status=status, result=result)
        self.controls.buffers["sequence"].add_(count)

    def trial(self):
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        total_us = self.start.elapsed_time(self.end) * 1000
        self.settle()
        return total_us / self.calls


def benchmark(args, hk, hv, b, budget, info):
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
    if not l2:
        if args.l2_bytes is None:
            raise ValueError(
                "device L2 query unavailable; pass --l2-bytes from the ceiling receipt"
            )
        l2 = args.l2_bytes
    read_bytes = 4 * b * hv * 128 * 128
    copies = max(1, math.ceil(3 * l2 / read_bytes)) if args.regime == "cold" else 1
    graphs = {}
    external_sources = {}
    if args.sglang_root:
        from benchmark.kernel_tuning.gdn_replay_common import digest

        for leaf in (
            "sglang/kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py",
            "sglang/kernels/ops/attention/fla/fused_recurrent.py",
            "sglang/kernels/ops/attention/triton_gdn_fused_proj.py",
            "sglang/kernels/ops/attention/fla/layernorm_gated.py",
        ):
            path = Path(args.sglang_root) / leaf
            if path.is_file():
                external_sources[leaf] = digest(path)
    if args.replayssm_root:
        from benchmark.kernel_tuning.gdn_replay_common import digest
        from benchmark.kernel_tuning.gdn_replay_external import CONFIG_FILE, OFFICIAL_FILE

        for leaf in (OFFICIAL_FILE, CONFIG_FILE):
            path = Path(args.replayssm_root) / leaf
            if path.is_file():
                external_sources[leaf] = digest(path)
    for provider in dict.fromkeys(["product", *args.providers]):
        budget.check()
        if provider in ("sglang_default", "sglang_replay", "official"):
            try:
                external_sources["sglang_revision"] = external_pin(args.sglang_root, "72730d6")
                if provider == "official":
                    external_pin(args.replayssm_root, "73a9592e9d8a2268620e8883a82f5cec7442ed36")
            except ImportError as error:
                emit(
                    {
                        "type": "not_available",
                        "provider": provider,
                        "shape": [hk, hv],
                        "B": b,
                        "reason": str(error),
                        "qualified": False,
                    }
                )
                continue
        try:
            graphs[provider] = WindowGraph(provider, hk, hv, b, args.window, copies, args, budget)
        except (ImportError, FileNotFoundError) as error:
            if provider == "product":
                raise
            emit(
                {
                    "type": "not_available",
                    "provider": provider,
                    "shape": [hk, hv],
                    "B": b,
                    "reason": type(error).__name__ + ": optional provider dependencies unavailable",
                    "qualified": False,
                }
            )
            continue
        if torch.cuda.max_memory_allocated() > 30 * 1024**3:
            raise RuntimeError("qualification memory ceiling exceeded (30 GiB)")
    product = graphs["product"]
    self_pairs = []
    for _ in range(20):
        budget.check()
        self_pairs.append([product.trial(), product.trial()])
    for provider, graph in graphs.items():
        pairs = []
        for i in range(20):
            budget.check()
            if i % 2:
                ours, baseline = graph.trial(), product.trial()
            else:
                baseline, ours = product.trial(), graph.trial()
            pairs.append([baseline, ours])
        stats = pair_stats(pairs, self_pairs)
        med = statistics.median([p[1] for p in pairs])
        measured_config = product.runner.launch_config()
        comparisons = {"product": stats}
        for comparator in ("sglang_default", "sglang_replay", "official"):
            if comparator not in graphs or comparator == provider:
                continue
            cp = graphs[comparator]
            calibration = []
            paired = []
            for i in range(20):
                budget.check()
                calibration.append([cp.trial(), cp.trial()])
                if i % 2:
                    ours, baseline = graph.trial(), cp.trial()
                else:
                    baseline, ours = cp.trial(), graph.trial()
                paired.append([baseline, ours])
            comparisons[comparator] = pair_stats(paired, calibration)
        phases = None
        if provider in ("cute", "triton") and not args.skip_phases:
            from benchmark.kernel_tuning.gdn_replay_phases import phase_times

            phases = phase_times(
                provider, hk, hv, b, args.window, budget, cute_config=graph.runner.op.cute_config
            )
        emit(
            {
                "phase_timings": phases,
                "product_launch_config": measured_config,
                "cute_config": asdict(graph.runner.op.cute_config) if provider == "cute" else None,
                "external_sources": external_sources,
                "type": "timing",
                "provider": provider,
                "shape": [hk, hv],
                "B": b,
                "L": args.window,
                "regime": args.regime,
                "rotation_copies": copies,
                "measured_l2_bytes": l2,
                "read_working_set_bytes": copies * read_bytes,
                "calls_per_trial": graph.calls,
                "per_token_us": med,
                "window_total_us": med * graph.calls,
                "kernels_per_call": graph.kernels_per_call,
                "comparisons": comparisons,
                "sol": sol_report(
                    provider,
                    b,
                    hk,
                    hv,
                    args.window,
                    med,
                    math.ceil(graph.kernels_per_call),
                    args.ceilings,
                    args.regime,
                ),
                "startup_peak_bytes": torch.cuda.max_memory_allocated(),
                "available_kv_tokens": None,
                "qualified": False,
                "missing_receipts": [
                    "engine-end-to-end",
                    "KV-residency",
                    "precision/model",
                ],
                "manifest": info,
            }
        )
    del graphs
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("immediate", "delayed", "auto"), default="auto")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--value-tile", type=int, choices=(32, 64, 128))
    parser.add_argument("--warps", type=int, choices=(4, 8))
    parser.add_argument("--speculative-ring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    parser.add_argument(
        "--shapes", nargs="+", choices=["16,32", "16,48"], default=["16,32", "16,48"]
    )
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--regime", choices=["cold", "warm", "both"], default="both")
    parser.add_argument("--sglang-root")
    parser.add_argument("--replayssm-root")
    parser.add_argument(
        "--skip-phases", action="store_true", help="timing subset; no phase receipt"
    )
    parser.add_argument(
        "--ceilings", help="Measured device ceiling JSON; no copied helper required"
    )
    parser.add_argument("--l2-bytes", type=int)
    parser.add_argument("--seconds", type=float, default=1440)
    args = parser.parse_args(argv)
    if "cute" not in args.providers and (
        args.mode != "auto"
        or args.value_tile is not None
        or args.speculative_ring is not None
        or args.warps is not None
    ):
        parser.error("CuTe tunables require cute in --providers")
    if not torch.cuda.is_available():
        emit({"type": "summary", "qualified": False, "reason": "CUDA not available"})
        return 2
    budget, info = Budget(args.seconds), manifest()
    if args.ceilings:
        from benchmark.kernel_tuning.gdn_replay_common import digest

        info["ceiling_hashes"] = {
            device: digest(path)
            for device in ("rtx5090", "rtx-pro-6000", "gb10")
            if (path := Path(args.ceilings).parent / (device + ".json")).is_file()
        }
    truncated = False
    try:
        for shape in args.shapes:
            hk, hv = map(int, shape.split(","))
            for b in args.batches:
                for regime in ("cold", "warm") if args.regime == "both" else (args.regime,):
                    row_args = argparse.Namespace(**vars(args))
                    row_args.regime = regime
                    benchmark(row_args, hk, hv, b, budget, info)
    except BudgetExpired:
        truncated = True
    emit(
        {
            "type": "summary",
            "qualified": False,
            "truncated": truncated,
            "boundary": "merged projected input through BF16 gated output; full windows; no subtraction",
        }
    )
    return 2 if truncated else 0


if __name__ == "__main__":
    raise SystemExit(main())
