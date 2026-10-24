# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Orchestrator-only cold graph probe, one fresh CUDA process per shape.

--tune selects at most ten device/shape-aware configs at M=1,2,4,8, then tests the winner
at M=1..8 (--m restricts both sweeps).
Explicit --tune '64,64,2,4;64,128,2,8' accepts up to ten configs.
The 780-second wall budget includes children, compilation and reference work. It
is cooperative: no new shape starts after it, and a running child is always
waited for, never terminated. A crashed child cannot poison subsequent shapes. An incomplete
sweep exits nonzero. No CUDA work at import time or in the parent process.
"""

# ruff: noqa: E402 -- start the wall clock before importing Torch
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

START = time.monotonic()

import torch
import torch.nn.functional as F

# The harness runs from the shipped source root; add it for benchmark utilities.
sys.path.insert(0, str(Path.cwd()))
from benchmark.kernel_tuning.linear_smallm import (
    Instance,
    L2Flush,
    accuracy,
    builtin_shapes,
    measure_bandwidth,
    paired,
    reference_and_condition,
    sol_bandwidth,
)
from ctikernels.linear._dispatch import (
    Config,
    config_for,
    device_class,
    parse_config,
    tune_configs,
)
from ctikernels.linear.cute._provider import get_compiled
from ctikernels.linear.inference import Linear


def emit(**row):
    row = {"out_dtype": "bf16", **row}
    print("RESULT " + json.dumps(row, allow_nan=False), flush=True)


def candidates(tune, default, n=0, k=0, kind=None):
    if tune is None:
        return [default]
    if tune == "auto":
        return tune_configs(default, n, k, kind)
    configs = list(dict.fromkeys(parse_config(part) for part in tune.split(";")))
    if not 1 <= len(configs) <= 10:
        raise ValueError("tuning requires 1..10 configs")
    return configs


def pair_count(kind, n, k, requested, phase):
    """Bound expensive GB10 cold replays; explicit --pairs always wins."""
    if requested is not None:
        return requested
    if kind == "gb10":
        return 3 if phase == "tune" and 2 * n * k >= 512 * 1024**2 else 6
    return 8


def child_failure(returncode, finished, reported_crash):
    if reported_crash:
        return None
    # Exit 1 after shape_completion is a reported numerical failure. Do not
    # blame the final progress location (which can be a different config).
    if not finished or returncode not in (0, 1):
        return "child_exit"
    return None


def run_shape(args):
    n, k = args.shape
    context = {"N": n, "K": k, "config": None, "M": None, "phase": "setup", "device_class": None}

    def progress(cfg, m, phase):
        context.update(config={**asdict(cfg), "warps": cfg.warps}, M=m, phase=phase)
        emit(kind="probe_progress", **context)

    try:
        os.environ["CTI_LINEAR_PROVIDER"] = "cute"
        os.environ.pop("CTI_LINEAR_CONFIG", None)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        props = torch.cuda.get_device_properties(0)
        cap = (props.major, props.minor)
        kind = device_class(cap, props.multi_processor_count)
        context["device_class"] = kind
        if cap not in ((12, 0), (12, 1)):
            raise RuntimeError("probe requires sm120 or sm121")
        configs = candidates(
            args.tune, config_for(kind, props.multi_processor_count, n, k), n, k, kind
        )
        smem_limit = props.shared_memory_per_block_optin
        if any(cfg.shared_bytes > smem_limit for cfg in configs):
            raise ValueError("candidate exceeds device opt-in shared memory limit")
        flush = L2Flush(props.L2_cache_size)
        bw, sol_metadata = sol_bandwidth(kind, lambda: measure_bandwidth(flush))
        emit(
            kind="probe_environment",
            N=n,
            K=k,
            device_class=kind,
            capability=cap,
            sms=props.multi_processor_count,
            shared_memory_per_block_optin=smem_limit,
            candidate_count=len(configs),
            tune_pairs=pair_count(kind, n, k, args.pairs, "tune"),
            best_pairs=pair_count(kind, n, k, args.pairs, "best"),
            bandwidth_bytes_per_second=bw,
            **sol_metadata,
            cold_flush_bytes=flush.nbytes,
            flush_method="read_amax",
            max_ulp_error_floor=0.5,
            max_ulp_error_slack=0.5,
            ulp_condition_floor=2 * torch.finfo(torch.float32).eps * max(1.0, math.sqrt(k / 1024)),
            rms_multiplier=1.05,
            normalized_multiplier=1.5,
            normalized_output_slack=2**-9,
            normalized_condition_slack=2 * torch.finfo(torch.float32).eps,
        )
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        x = torch.randn((8, k), device="cuda", dtype=torch.bfloat16, generator=gen)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16, generator=gen)
        ref, condition = reference_and_condition(x, w)
        baseline = {}
        failures = 0
        rows = 0

        def measure(cfg, ms, phase):
            nonlocal failures, rows
            pairs = pair_count(kind, n, k, args.pairs, phase)
            progress(cfg, None, "compile")
            begin = time.monotonic()
            get_compiled(x.device, n, k, cfg, cap, False)
            compile_seconds = time.monotonic() - begin
            op = Linear(config=cfg)
            valid, times, ratios = True, {}, []
            for m in ms:
                xx = x[:m]
                # Synchronize an eager correctness check BEFORE graph capture/timing.
                progress(cfg, m, "eager_correctness")
                begin = time.monotonic()
                output = op(xx, w)
                torch.cuda.synchronize()
                first_launch_seconds = time.monotonic() - begin
                if op.provider != "cute":
                    raise AssertionError("forced CuTe did not run")
                expected = F.linear(xx, w)
                torch.cuda.synchronize()
                metrics = accuracy(output, expected, ref[:m], condition[:m], k)
                if not metrics["correct"]:
                    emit(kind="smallm_probe", status="incorrect", sweep=phase, **context, **metrics)
                    failures += 1
                    valid = False
                    continue
                progress(cfg, m, "baseline_graph")
                if m not in baseline:
                    baseline[m] = Instance("cublas", F.linear, xx, w)
                progress(cfg, m, "graph_correctness")
                ours = Instance("ours", op, xx, w)
                torch.cuda.synchronize()
                metrics = accuracy(ours.output, baseline[m].output, ref[:m], condition[:m], k)
                if not metrics["correct"]:
                    emit(kind="smallm_probe", status="incorrect", sweep=phase, **context, **metrics)
                    failures += 1
                    valid = False
                    continue
                progress(cfg, m, "timing")
                a, b = paired(baseline[m], ours, pairs, 1, flush)
                ours_us, cublas_us = statistics.median(b), statistics.median(a)
                times[m] = ours_us
                ratio = statistics.median(aa / bb for aa, bb in zip(a, b, strict=True))
                ratios.append(ratio)
                compulsory = 2 * (n * k + m * k + m * n)
                emit(
                    kind="smallm_probe",
                    status="ok",
                    sweep=phase,
                    **context,
                    mode="cold",
                    pairs=pairs,
                    grid_ctas=math.ceil(n / cfg.rows) * cfg.splits,
                    weight_slice_bytes=2
                    * cfg.rows
                    * cfg.tile_k
                    * math.ceil(k / (cfg.splits * cfg.tile_k)),
                    shared_bytes=cfg.shared_bytes,
                    **sol_metadata,
                    provider=op.provider,
                    kernels_per_call=op.kernels_per_call,
                    compile_seconds=compile_seconds,
                    first_launch_seconds=first_launch_seconds,
                    ours_us=ours_us,
                    cublas_us=cublas_us,
                    paired_speedup=ratio,
                    time_over_M1=ours_us / times[1] if 1 in times else None,
                    sol_us=compulsory / bw * 1e6,
                    bandwidth_utilization=compulsory / (ours_us * 1e-6) / bw,
                    **metrics,
                )
                rows += 1
                del ours
            return statistics.geometric_mean(ratios) if valid and ratios else None

        if args.tune is not None:
            tune_ms = tuple(m for m in args.ms if m in (1, 2, 4, 8)) or (args.ms[0],)
            scores = [(measure(cfg, tune_ms, "tune"), cfg) for cfg in configs]
            qualified = [(score, cfg) for score, cfg in scores if score is not None]
            if not qualified:
                emit(
                    kind="shape_completion",
                    N=n,
                    K=k,
                    status="incorrect",
                    rows=rows,
                    failures=failures,
                )
                return 1
            score, best = max(qualified, key=lambda item: item[0])
            emit(kind="probe_best", N=n, K=k, config=asdict(best), geometric_speedup=score)
            measure(best, args.ms, "best")
        else:
            measure(configs[0], args.ms, "default")
        emit(
            kind="shape_completion",
            N=n,
            K=k,
            status="ok" if not failures else "incorrect",
            rows=rows,
            failures=failures,
        )
        return int(failures != 0)
    except Exception as exc:  # noqa: BLE001 -- poisoned CUDA context must exit, not continue
        emit(kind="smallm_probe", status="crash", **context, error=type(exc).__name__)
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=[*builtin_shapes(), "all"], default="qwen3.5-9b")
    ap.add_argument("--m", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--tune", nargs="?", const="auto")
    ap.add_argument("--pairs", type=int, help="override pairs (GB10: 6, huge tune: 3; others: 8)")
    ap.add_argument("--budget-seconds", type=float, default=780)
    ap.add_argument("--shape", nargs=2, type=int, help="probe only this N K, with child isolation")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.ms = sorted({int(v) for v in args.m.split(",")})
    if (
        not args.ms
        or min(args.ms) < 1
        or max(args.ms) > 8
        or (args.pairs is not None and args.pairs < 1)
        or (args.shape is not None and (min(args.shape) <= 0 or args.shape[1] % 8))
        or (args.child and args.shape is None)
        or not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
    ):
        ap.error(
            "M must be 1..8, pairs/budget/N/K positive, K divisible by 8; child requires shape"
        )
    if args.tune not in (None, "auto"):
        candidates(args.tune, Config())  # validate before starting children
    if args.child:
        with torch.inference_mode():
            return run_shape(args)
    shapes = set()
    for model, entries in builtin_shapes().items():
        if args.model in (model, "all"):
            shapes.update((r["N"], r["K"]) for r in entries)
    shapes.update([(1024, 4096), (67, 264), (129, 520), (1, 8)])
    if args.shape:
        shapes = {tuple(args.shape)}
    failures, completed = 0, 0
    for n, k in sorted(shapes):
        remaining = args.budget_seconds - (time.monotonic() - START)
        if remaining <= 0:
            break
        child_args = [
            str(Path(__file__).resolve()),
            "--child",
            "--shape",
            str(n),
            str(k),
            "--m",
            args.m,
        ]
        if args.pairs is not None:
            child_args += ["--pairs", str(args.pairs)]
        if args.tune is not None:
            child_args += ["--tune", args.tune]
        # Preserve the orchestrator's pinned DSL paths: site .pth files can
        # otherwise prepend a different installed CuTe version in the child.
        bootstrap = (
            "import sys,json,runpy; "
            "paths,args=json.loads(sys.argv[1]); "
            "sys.path[:]=paths; "
            "[sys.modules.pop(name) for name in list(sys.modules) "
            "if name.split('.')[0] in ('cutlass','nvidia_cutlass_dsl')]; "
            "sys.argv=args; runpy.run_path(args[0],run_name='__main__')"
        )
        cmd = [sys.executable, "-c", bootstrap, json.dumps([sys.path, child_args])]
        # Never terminate the child (or any process): wait for it to finish.
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        ) as child:
            stdout, _ = child.communicate()
        context = {
            "N": n,
            "K": k,
            "config": None,
            "M": None,
            "phase": "setup",
            "device_class": None,
        }
        reported_crash = False
        finished = False
        for line in stdout.splitlines():
            if not line.startswith("RESULT "):
                continue
            row = json.loads(line[7:])
            if row["kind"] == "probe_progress":
                context = {key: row[key] for key in context}
            reported_crash |= row.get("status") == "crash"
            finished |= row["kind"] == "shape_completion"
            emit(**row)
        error = child_failure(child.returncode, finished, reported_crash)
        if error is not None:
            emit(
                kind="smallm_probe",
                status="crash",
                **context,
                error=error,
                exit_code=child.returncode,
            )
        failures += int(bool(child.returncode) or not finished)
        completed += int(finished)
    complete = completed == len(shapes)
    emit(
        kind="probe_completion",
        complete=complete,
        failures=failures,
        shapes=completed,
        expected_shapes=len(shapes),
        elapsed_seconds=time.monotonic() - START,
    )
    return int(failures != 0 or not complete)


if __name__ == "__main__":
    raise SystemExit(main())
