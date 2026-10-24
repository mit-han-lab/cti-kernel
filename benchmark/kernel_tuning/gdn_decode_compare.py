# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired T=1 GDN decode, conv through gated norm, with both GEMMs excluded.

Example (external source trees are read-only inputs)::

    python -m benchmark.kernel_tuning.gdn_decode_compare \
        --sglang-root checkouts/sglang/python --vllm-root checkouts/vllm-0.31.0 \
        --jsonl results/decode.jsonl

The selected vLLM boundary is 0.31.0's fused-decode-enabled CUDA default,
non-interleaved Qwen3.5, no speculation, packed recurrent decode enabled.
All projection splitting/materialization and persistent cache layout conversion
occur before capture. vLLM conv mutates its projection: each captured invocation
therefore owns a qkvz replica, restored from the distinct GEMM outputs
OUTSIDE every timed replay. This is recorded, and never subtracted from timing.
Its zeros, b/a contiguous calls, gate reshape, conv, recurrence and in-place norm
remain inside capture. SGLang allocates its normal uninitialized intermediates.

--window sets the replay window and calls per input cycle (default 8).
--shapes takes per-rank Hk,Hv, including TP2 8,16 and 8,24. With --ours unified,
--ours-mode selects auto, immediate or delayed CuTe decode. The optional
sglang_replay provider uses SGLang's standalone ReplaySSM leaf with the same
ring window and the ordinary SGLang conv/norm boundary; it is not a serving default.

The cold rotation and event trial follow gdn_replay_bench.WindowGraph. There is
no separator kernel, subtraction, GEMM, or state reset in a timed graph. State
continues across trials. Importing this module requires only the standard library.
"""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

STEPS = 8
MEMORY_LIMIT = 20 * 1024**3
PROVIDERS = ("ours", "product", "sglang", "sglang_replay", "vllm")


def nearest_rank(values, quantile):
    if not values or not 0 < quantile <= 1:
        raise ValueError("nonempty values and quantile in (0,1] required")
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def _validate_pairs(pairs):
    if not pairs or any(
        len(pair) != 2 or any(not math.isfinite(t) or t <= 0 for t in pair) for pair in pairs
    ):
        raise ValueError("nonempty pairs of positive finite durations required")


def noise_epsilon(self_pairs):
    _validate_pairs(self_pairs)
    return nearest_rank([abs(a / b - 1) for a, b in self_pairs], 0.95)


def verdict(ratio, epsilon):
    if not math.isfinite(ratio) or ratio <= 0 or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid ratio or epsilon")
    if ratio > 1 + epsilon:
        return "win"
    if ratio < 1 - epsilon:
        return "loss"
    return "tie"


def bootstrap_median_ci(values, seed=11, resamples=2000):
    if not values or resamples < 1 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("positive finite values and resample count required")
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(resamples)]
    return [nearest_rank(medians, 0.025), nearest_rank(medians, 0.975)]


def pair_stats(pairs, self_pairs):
    """Pair order in storage is always [comparator_us, ours_us]."""
    _validate_pairs(pairs)
    if len(pairs) != len(self_pairs):
        raise ValueError("incomplete paired trials")
    epsilon = noise_epsilon(self_pairs)
    ratios = [comparator / ours for comparator, ours in pairs]
    median = statistics.median(ratios)
    return {
        "pairs_us": pairs,
        "ratios": ratios,
        "median_ratio": median,
        "epsilon": epsilon,
        "median_ratio_ci95": bootstrap_median_ci(ratios),
        "verdict": verdict(median, epsilon),
        "comparator_us": statistics.median(p[0] for p in pairs),
        "ours_us": statistics.median(p[1] for p in pairs),
    }


def vs_faster(comparisons):
    comparisons = {k: v for k, v in comparisons.items() if k in ("sglang", "sglang_replay", "vllm")}
    if not comparisons:
        return None
    name = min(comparisons, key=lambda key: comparisons[key]["median_ratio"])
    return {
        "comparator": name,
        **{
            key: comparisons[name][key]
            for key in ("median_ratio", "epsilon", "median_ratio_ci95", "verdict")
        },
    }


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _window(value):
    result = int(value)
    if result < 2 or result & (result - 1):
        raise argparse.ArgumentTypeError("must be a power of two >= 2")
    return result


def _shape(value):
    try:
        hk, hv = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be Hk,Hv") from error
    if hk <= 0 or hv <= 0 or hv % hk:
        raise argparse.ArgumentTypeError("positive heads with Hv a multiple of Hk required")
    return hk, hv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=_positive_int, default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--shapes", nargs="+", type=_shape, default=[(16, 32), (16, 48)])
    parser.add_argument("--window", type=_window, default=STEPS)
    parser.add_argument("--ours", choices=("product", "unified"), default="product")
    parser.add_argument("--ours-mode", choices=("auto", "immediate", "delayed"), default="auto")
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=["ours", "sglang", "vllm"]
    )
    parser.add_argument("--sglang-root", type=Path)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--regime", choices=("cold", "warm"), default="cold")
    parser.add_argument("--l2-bytes", type=_positive_int)
    parser.add_argument("--pairs", type=_positive_int, default=20)
    parser.add_argument("--seconds", type=float, default=1500)
    parser.add_argument("--ceilings", type=Path)
    parser.add_argument("--jsonl", type=Path, default=Path("gdn_decode_compare.jsonl"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if args.ours_mode != "auto" and args.ours != "unified":
        parser.error("--ours-mode requires --ours unified")
    args.providers = list(dict.fromkeys(["ours", *args.providers]))
    if "product" in args.providers and args.ours != "unified":
        parser.error("the product comparator requires --ours unified")
    for provider in args.providers[1:]:
        if provider == "product":
            continue
        if getattr(args, ("sglang" if provider == "sglang_replay" else provider) + "_root") is None:
            root = "sglang" if provider == "sglang_replay" else provider
            parser.error(f"--{root}-root is required for {provider}")
    return args


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, start=None):
        self.deadline = (time.monotonic() if start is None else start) + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired("process budget exhausted")


def rotation(b, hv, l2, regime, window=STEPS):
    copies = max(1, math.ceil(3 * l2 / (4 * b * hv * 128 * 128))) if regime == "cold" else 1
    return copies, window * copies * math.ceil(256 / copies)


def _memory_check(torch):
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    assert max(allocated, reserved) < MEMORY_LIMIT, "20 GiB peak CUDA memory ceiling exceeded"
    return {"allocated": allocated, "reserved": reserved}


def _error(got, ref):
    import torch

    got, ref = got.double(), ref.double()
    finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
    if not finite:
        return {"maxabs": None, "relative_rms": None, "finite": False}
    diff = got - ref
    return {
        "maxabs": float(diff.abs().max()),
        "relative_rms": float(
            diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-6)
        ),
        "finite": True,
    }


def check_correctness(runners, xs, initial, history, b, budget):
    import torch

    # Reserve slot zero for vLLM's NULL_BLOCK_ID, and leave a trailing sentinel.
    # Reverse selected slots so row-to-slot mapping is checked too.
    selected = torch.arange(b, 0, -1, dtype=torch.int32, device=xs.device)
    indices = selected.long()
    inactive = torch.ones(initial.shape[0], dtype=torch.bool, device=xs.device)
    inactive[indices] = False
    expected_history = history.clone()
    dc = history.shape[-1]
    expected_history[indices] = torch.cat((history[indices, 1:], xs[0, :, None, :dc]), dim=1)
    checks = {}
    reference_out = reference_state = reference_history = None
    for name, runner in runners.items():
        budget.check()
        if name == "ours":
            out = runner.op(xs[0], *runner.weights, selected, runner.state, runner.history)
            canonical = runner.history
            reference_out = out.clone()
            reference_state = runner.state[indices].clone()
            reference_history = canonical.clone()
        else:
            out = runner.step(runner.prepare(xs[0]), selected)
            canonical = runner.canonical_history()
        output_error = _error(out, reference_out)
        state_error = _error(runner.state[indices], reference_state)
        unchanged = bool(torch.equal(runner.state[inactive], initial[inactive]))
        history_unchanged = bool(torch.equal(canonical[inactive], history[inactive]))
        history_equal = bool(torch.equal(canonical, reference_history))
        history_expected = bool(torch.equal(canonical, expected_history))
        correct = (
            all(e["finite"] and e["relative_rms"] <= 2e-2 for e in (output_error, state_error))
            and unchanged
            and history_unchanged
            and history_equal
            and history_expected
        )
        checks[name] = {
            "correct": correct,
            "output": output_error,
            "selected_state": state_error,
            "unselected_state_unchanged": unchanged,
            "unselected_history_unchanged": history_unchanged,
            "conv_history_exact": history_equal,
            "conv_history_matches_shift": history_expected,
        }
    # Return all providers to identical starting state before warm-up.
    for name, runner in runners.items():
        runner.state.copy_(initial)
        target = runner.history if name == "ours" else runner.canonical_history()
        target.copy_(history)
    return checks


class Unified:
    """Trusted replay-layout decode, with lifecycle controls only outside timing."""

    def __init__(self, hk, hv, b, weights, initial, history, *, mode="auto", window=STEPS):
        import torch

        from benchmark.kernel_tuning.gdn_replay_common import allocate
        from ctikernels.gdn.replay import GatedDeltaNetReplay

        self.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device="cuda",
            window=window,
            verify_length=16,
            provider="cute",
            mode=mode,
            checks="trusted",
        )
        self.weights, self.hv = weights, hv
        slots = initial.shape[0]
        self.state = self.op.create_state(
            buffers=allocate(self.op.state_spec(num_slots=slots, num_snapshot_rows=1), "cuda")
        )
        self.controls = self.op.bind_graph_buffers(
            buffers=allocate(self.op.graph_buffer_spec(batch_size=slots), "cuda")
        )
        self.sequence = 0
        self.positions = torch.zeros(slots, dtype=torch.int64, device="cuda")
        self.slots = torch.arange(slots, dtype=torch.int32, device="cuda")
        self.workspace = {
            k: v
            for k, v in allocate(self.op.workspace_spec("step", batch_size=b), "cuda").items()
            if k != "out"
        }
        self.reset(initial, history)

    def begin(self, batch_size):
        self.sequence += 1
        status = self.controls.status(0, batch_size=batch_size)
        status.sequence.fill_(self.sequence)
        self.op.begin(status=status, layer_weights=(self.weights,))
        return status

    def settle(self, status):
        self.op.end(status=status)
        result = self.op.read_status_copy(*status.copy_to_host(), expected_sequence=self.sequence)
        if not self.op.settle(status=status, result=result):
            raise RuntimeError("unified lifecycle status failed")

    def reset(self, initial, history):
        self.positions.zero_()
        status = self.begin(initial.shape[0])
        self.op.handoff(
            canonical_state=initial,
            history=history,
            next_positions=self.positions,
            slot_indices=self.slots,
            state=self.state,
            status=status,
        )
        self.settle(status)

    def step(self, x, slots, positions, out):
        self.op.step(
            x,
            *self.weights,
            slot_indices=slots,
            positions=positions,
            state=self.state,
            out=out,
            workspace=self.workspace,
            status=None,
        )
        return out

    def flush(self, slots):
        status = self.begin(slots.numel())
        self.op.flush(
            *self.weights,
            slot_indices=slots,
            state=self.state,
            status=status,
            workspace=self.workspace,
        )
        self.settle(status)


def check_unified_correctness(runner, reference, xs, initial, history, b, budget, window=STEPS):
    """Exercise two ring wraps and a partial flush on independent cloned pools."""
    import torch

    selected = torch.arange(b, 0, -1, dtype=torch.int32, device=xs.device)
    indices = selected.long()
    inactive = torch.ones(initial.shape[0], dtype=torch.bool, device=xs.device)
    inactive[indices] = False
    positions = torch.zeros(b, dtype=torch.int64, device=xs.device)
    out = torch.empty(b, initial.shape[1] * 128, dtype=xs.dtype, device=xs.device)
    expected_history = history.clone()
    dc = history.shape[-1]
    outputs = []
    for step in range(2 * window + 3):
        budget.check()
        x = xs[step % window]
        expected = reference.op(x, *reference.weights, selected, reference.state, reference.history)
        outputs.append(_error(runner.step(x, selected, positions, out), expected))
        positions.add_(1)
        expected_history[indices] = torch.cat(
            (expected_history[indices, 1:], x[:, None, :dc]), dim=1
        )
    runner.flush(selected)
    state_error = _error(runner.state.checkpoint[indices], reference.state[indices])
    history_error = _error(runner.state.conv_history[indices], reference.history[indices])
    unchanged = bool(
        torch.equal(runner.state.checkpoint[: initial.shape[0]][inactive], initial[inactive])
    )
    history_unchanged = bool(
        torch.equal(runner.state.conv_history[: history.shape[0]][inactive], history[inactive])
    )
    history_equal = bool(
        torch.equal(runner.state.conv_history[indices], reference.history[indices])
    )
    history_expected = bool(
        torch.equal(runner.state.conv_history[indices], expected_history[indices])
    )
    correct = (
        all(
            e["finite"] and e["relative_rms"] <= 2e-2
            for e in (*outputs, state_error, history_error)
        )
        and unchanged
        and history_unchanged
        and history_equal
        and history_expected
    )
    return {
        "correct": correct,
        "steps": len(outputs),
        "outputs": outputs,
        "selected_state": state_error,
        "selected_history": history_error,
        "unselected_state_unchanged": unchanged,
        "unselected_history_unchanged": history_unchanged,
        "conv_history_exact": history_equal,
        "conv_history_matches_shift": history_expected,
    }


def check_replay_correctness(runner, reference, xs, initial, history, b, budget, window=STEPS):
    """Check two full ReplaySSM windows and a forced partial flush, then reset."""
    import torch

    selected = torch.arange(b, 0, -1, dtype=torch.int32, device=xs.device)
    indices = selected.long()
    inactive = torch.ones(initial.shape[0], dtype=torch.bool, device=xs.device)
    inactive[indices] = False
    outputs, checkpoints = [], []
    steps = 2 * window + 3
    for step in range(steps):
        budget.check()
        x = xs[step % window]
        expected = reference.op(x, *reference.weights, selected, reference.state, reference.history)
        flush = step == steps - 1
        projected = runner.prepare(x, step % window, force_flush=flush)
        outputs.append(_error(runner.step(projected, selected), expected))
        if step % window == window - 1 or flush:
            checkpoints.append(_error(runner.state[indices], reference.state[indices]))
    canonical = runner.canonical_history()
    unchanged = bool(torch.equal(runner.state[inactive], initial[inactive]))
    history_unchanged = bool(torch.equal(canonical[inactive], history[inactive]))
    history_equal = bool(torch.equal(canonical, reference.history))
    result = {
        "correct": (
            all(e["finite"] and e["relative_rms"] <= 2e-2 for e in (*outputs, *checkpoints))
            and unchanged
            and history_unchanged
            and history_equal
        ),
        "steps": steps,
        "outputs": outputs,
        "checkpoints": checkpoints,
        "unselected_state_unchanged": unchanged,
        "unselected_history_unchanged": history_unchanged,
        "conv_history_exact": history_equal,
    }
    runner.reset(initial, history)
    return result


class DecodeGraph:
    def __init__(self, name, runner, xs, groups, calls, budget, window=STEPS):
        import torch

        self.name, self.runner, self.calls = name, runner, calls
        self.groups = groups
        self.xs = xs
        self.bank = self.templates = None
        if name in ("ours", "product"):
            prepared = list(xs.unbind(0))
        elif name == "sglang_replay":
            prepared = [runner.prepare(x, phase) for phase, x in enumerate(xs)]
        else:
            prepared = [runner.prepare(x) for x in xs]
        self.prepared = prepared
        copies = len(groups)
        if name == "vllm":
            # One materialization per distinct projection, then replicas for the
            # destructive conv. No reset/copy kernel belongs to the timed graph.
            self.templates = torch.stack([p[0] for p in prepared])
            repeats = calls // (window * copies)
            self.bank = torch.empty(
                (repeats, window, copies, *self.templates.shape[1:]),
                dtype=xs.dtype,
                device=xs.device,
            )
            self.restore_inputs()
        self.invocations = []
        unified = isinstance(runner, Unified)
        if unified:
            self.outputs = torch.empty(
                calls, xs.shape[1], runner.hv * 128, dtype=xs.dtype, device=xs.device
            )
            # Trusted decode uses the state's own cursor. Keep borrowed position
            # buffers static, as the product route also leaves positions to its caller.
            positions = [
                runner.positions[1 + i * xs.shape[1] : 1 + (i + 1) * xs.shape[1]]
                for i in range(copies)
            ]
        for i in range(calls):
            phase, group = (i // copies) % window, i % copies
            projected = prepared[phase]
            if self.bank is not None:
                projected = (self.bank[i // (window * copies), phase, group], projected[1])
            slots = groups[group]
            if unified:

                def call(x=projected, slots=slots, pos=positions[group], out=self.outputs[i]):
                    return runner.step(x, slots, pos, out)
            elif name in ("ours", "product"):

                def call(x=projected, slots=slots):
                    return runner.op(x, *runner.weights, slots, runner.state, runner.history)
            else:

                def call(x=projected, slots=slots):
                    return runner.step(x, slots)

            self.invocations.append(call)
        budget.check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Every input/group is warmed; destructive vLLM replicas also vary
            # alignment, so warm every captured invocation for that provider.
            warm = self.invocations if name == "vllm" else self.invocations[: window * copies]
            for call in warm:
                budget.check()
                call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.restore_inputs()
        torch.cuda.synchronize()
        budget.check()
        _memory_check(torch)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            for call in self.invocations:
                call()
        self.start, self.end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        _memory_check(torch)
        self.restore_inputs()
        torch.cuda.synchronize()
        budget.check()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            self.graph.replay()
            torch.cuda.synchronize()
        events = [
            e
            for e in profile.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and e.name != "[memory]"
        ]
        names = {}
        for event in events:
            names[event.name] = names.get(event.name, 0) + 1
        memcpy = sum(n for key, n in names.items() if "memcpy" in key.lower())
        memset = sum(n for key, n in names.items() if "memset" in key.lower())
        total = len(events)
        if not total:
            raise RuntimeError("profiler did not record CUDA events")
        self.profile = {
            "kernels_per_call_including_memset": (total - memcpy) / calls,
            "kernels_per_call_excluding_memset": (total - memcpy - memset) / calls,
            "memset_per_call": memset / calls,
            "memcpy_per_call": memcpy / calls,
            "all_gpu_events_per_call": total / calls,
            "events_per_replay": names,
        }

    def restore_inputs(self):
        if self.bank is not None:
            self.bank.copy_(self.templates[None, :, None])

    def trial(self):
        # The restore precedes the start event on the same stream. Thus its work
        # is excluded even though it is asynchronous on the host.
        self.restore_inputs()
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / self.calls


def benchmark_row(args, hk, hv, b, budget, row):
    import torch

    from benchmark.kernel_tuning.gdn_decode_external import (
        ExternalDecode,
        SGLangReplayDecode,
        UnavailableComparator,
    )
    from benchmark.kernel_tuning.gdn_replay_common import Product
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    row["stage"] = "allocation"
    copies, calls = row["copies"], row["calls"]
    window = args.window
    # Conservative storage estimate before allocating pools or graph replicas.
    slots = copies * b + 2
    dc = (2 * hk + hv) * 128
    pool = slots * (4 * hv * 128 * 128 + 6 * dc)
    replicas = calls * b * (dc + hv * 128) * 2 if "vllm" in args.providers else 0
    estimate = (len(args.providers) + 5) * pool + 3 * replicas
    if "sglang_replay" in args.providers:
        estimate += 4 * slots * window * (hv * 128 + hk * 128 + hv)
    if args.ours == "unified":
        estimate += 2 * slots * (window + 3) * (dc + 2 * hv) * 2
        estimate += calls * b * hv * 128 * 2
    assert estimate < MEMORY_LIMIT, "20 GiB conservative memory estimate exceeded"
    weights, xs, initial, history = case_inputs(hk, hv, b, slots, window, "ordinary", 11, "cuda")
    weights = weights.args()
    product = Product(hk, hv, weights, initial, history)
    runners = {"ours": product}
    row["ours_kind"] = args.ours
    row["ours_mode"] = args.ours_mode
    row["product_provider"] = product.op.provider
    if args.ours == "unified":
        from ctikernels.gdn.replay.cute import effective_config

        runners["ours"] = Unified(
            hk, hv, b, weights, initial, history, mode=args.ours_mode, window=window
        )
        row["ours_route"] = runners["ours"].op.decode_route(b)
        row["cute_config"] = asdict(effective_config(runners["ours"].op, b))
    row["external"] = {}
    major, minor = torch.cuda.get_device_capability()
    for name in args.providers[1:]:
        budget.check()
        row["stage"] = "load:" + name
        if name == "product":
            runners[name] = product
            continue
        if name == "sglang_replay":
            try:
                runner = SGLangReplayDecode(
                    args.sglang_root, major * 10 + minor, hk, hv, weights, initial, history, window
                )
            except UnavailableComparator as error:
                row.setdefault("skipped", {})[name] = {
                    "status": "skipped",
                    "reason": str(error),
                    "shape": [hk, hv],
                }
                continue
            runners[name] = runner
            row["external"][name] = runner.leaves.metadata
            continue
        runner = ExternalDecode(
            name,
            getattr(args, name + "_root"),
            major * 10 + minor,
            hk,
            hv,
            weights,
            initial,
            history,
        )
        runners[name] = runner
        row["external"][name] = runner.leaves.metadata
    row["stage"] = "correctness"
    reference_runners = {
        "ours": product,
        **{name: runner for name, runner in runners.items() if name in ("sglang", "vllm")},
    }
    row["correctness"] = check_correctness(reference_runners, xs, initial, history, b, budget)
    if "sglang_replay" in runners:
        row["correctness"]["sglang_replay"] = check_replay_correctness(
            runners["sglang_replay"],
            Product(hk, hv, weights, initial, history),
            xs,
            initial,
            history,
            b,
            budget,
            window,
        )
    if args.ours == "unified":
        if "product" in runners:
            row["correctness"]["product"] = row["correctness"]["ours"]
        check_runner = Unified(
            hk, hv, b, weights, initial.clone(), history.clone(), mode=args.ours_mode, window=window
        )
        check_product = Product(hk, hv, weights, initial, history)
        row["correctness"]["ours"] = check_unified_correctness(
            check_runner, check_product, xs, initial, history, b, budget, window
        )
        del check_runner, check_product
        runners["ours"].reset(initial, history)
    row["peak_memory_bytes"] = _memory_check(torch)
    row["correct"] = all(check["correct"] for check in row["correctness"].values())
    row["product_launch_config"] = product.launch_config()
    if not row["correct"]:
        row["status"] = "incorrect"
        return
    del initial, history
    groups = [
        torch.arange(1 + i * b, 1 + (i + 1) * b, dtype=torch.int32, device="cuda")
        for i in range(copies)
    ]
    graphs = {}
    for name, runner in runners.items():
        budget.check()
        row["stage"] = "warm_capture_profile:" + name
        graphs[name] = DecodeGraph(name, runner, xs, groups, calls, budget, window)
        row["profile"][name] = graphs[name].profile
        _memory_check(torch)
    row["product_launch_config"] = product.launch_config()
    ours = graphs["ours"]
    row["stage"] = "self_pairs"
    self_pairs = row["self_pairs_us"]
    for _ in range(args.pairs):
        budget.check()
        self_pairs.append([ours.trial(), ours.trial()])
    row["epsilon"] = noise_epsilon(self_pairs)
    ours_times = [t for pair in self_pairs for t in pair]
    for name, graph in graphs.items():
        if name == "ours":
            continue
        row["stage"] = "pairs:" + name
        pairs = row["raw_pairs_us"][name] = []
        for i in range(args.pairs):
            budget.check()
            if i % 2 == 0:
                comparator_us, ours_us = graph.trial(), ours.trial()
            else:
                ours_us, comparator_us = ours.trial(), graph.trial()
            pairs.append([comparator_us, ours_us])
            ours_times.append(ours_us)
        row["comparisons"][name] = pair_stats(pairs, self_pairs)
        row["median_us"][name] = row["comparisons"][name]["comparator_us"]
    row["median_us"]["ours"] = statistics.median(ours_times)
    row["vs_faster"] = vs_faster(row["comparisons"])
    if args.ceilings is not None:
        from benchmark.kernel_tuning.gdn_replay_sol import report

        row["stage"] = "sol"
        kernel_count = ours.profile["kernels_per_call_including_memset"]
        if args.ours == "product" and not kernel_count.is_integer():
            raise RuntimeError("product profile kernel count is not integral")
        # The SOL model follows the effective route: only delayed buffers a ring and
        # folds the checkpoint once per window; #25 and immediate write back every step.
        delayed = args.ours == "unified" and row.get("ours_route") == "delayed"
        row["sol_model"] = "delayed" if delayed else "per-step-writeback"
        row["sol"] = report(
            "cute" if delayed else "product",
            b,
            hk,
            hv,
            window if delayed else 1,
            row["median_us"]["ours"],
            math.ceil(kernel_count),
            args.ceilings,
            args.regime,
        )
    row["peak_memory_bytes"] = _memory_check(torch)
    row["status"] = "ok"
    row["stage"] = "complete"


def metadata():
    versions = {}
    for package in ("torch", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
        head = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        head = None
    # Only accept a CPU-list syntax; never emit arbitrary environment contents.
    import re

    cpus = os.environ.get("CTI_X925")
    if cpus is not None and re.fullmatch(r"[0-9,\-\s]+", cpus) is None:
        cpus = None
    return {"versions": versions, "git_head": head, "x925_cpu_list": cpus}


def emit(row, stream):
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()
    comparisons = " ".join(
        f"{name}={s['median_ratio']:.3f}x/{s['verdict']}" for name, s in row["comparisons"].items()
    )
    ours = row["median_us"].get("ours")
    duration = "-" if ours is None else f"{ours:.3f}us"
    print(
        f"B={row['B']} shape={row['shape'][0]},{row['shape'][1]} {row['regime']} "
        f"{row['status']} ours={duration} {comparisons}",
        flush=True,
    )


def main(argv=None):
    process_start = time.monotonic()
    args = parse_args(argv)
    budget = Budget(args.seconds, process_start)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for benchmarking; import/statistics are CPU-only")
    info = metadata()
    props = torch.cuda.get_device_properties(0)
    info["device"] = props.name
    l2 = (
        args.l2_bytes
        or getattr(props, "L2_cache_size", None)
        or getattr(props, "l2_cache_size", None)
    )
    if not l2 and args.regime == "cold":
        raise ValueError("L2 size unavailable; supply --l2-bytes")
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.jsonl.open("a") as stream, torch.inference_mode():
        for hk, hv in args.shapes:
            for b in args.batches:
                copies, calls = rotation(b, hv, l2 or 0, args.regime, args.window)
                row = {
                    **info,
                    "B": b,
                    "shape": [hk, hv],
                    "regime": args.regime,
                    "copies": copies,
                    "calls": calls,
                    "pairs": args.pairs,
                    "steps": args.window,
                    "window": args.window,
                    "l2_bytes": l2,
                    "providers": args.providers,
                    "ours_kind": args.ours,
                    "correct": None,
                    "status": "pending",
                    "stage": "budget_check",
                    "self_pairs_us": [],
                    "raw_pairs_us": {},
                    "comparisons": {},
                    "median_us": {},
                    "profile": {},
                    "sol": None,
                    "vs_faster": None,
                    "expected_gpu_operations_per_call": {
                        name: {
                            "ours": None if args.ours == "unified" else 1,
                            "product": 1,
                            "sglang": 3,
                            "sglang_replay": 3,
                            "vllm": 4 if b == 1 else 7,
                        }[name]
                        for name in args.providers
                    },
                    "projection_policy": (
                        f"{'eight' if args.window == 8 else args.window} distinct inputs; "
                        "vLLM per-call qkvz replicas restored before start event"
                    ),
                    "boundary": "T=1 conv through gated norm; no input/output GEMMs",
                }
                try:
                    budget.check()
                    torch.cuda.reset_peak_memory_stats()
                    benchmark_row(args, hk, hv, b, budget, row)
                except BudgetExpired:
                    row["status"] = "budget_truncated"
                except Exception as error:  # noqa: BLE001 - preserve one safe failure receipt per row
                    row["status"] = "error"
                    # Exception messages from external code can contain paths or
                    # arbitrary environment values. Emit only the exception type.
                    row["error_type"] = type(error).__name__
                finally:
                    # benchmark_row has unwound before collection: destroy graphs
                    # and their pools, including on correctness/budget failures.
                    gc.collect()
                    torch.cuda.empty_cache()
                failed |= row["status"] != "ok"
                row["elapsed_seconds"] = time.monotonic() - process_start
                emit(row, stream)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
