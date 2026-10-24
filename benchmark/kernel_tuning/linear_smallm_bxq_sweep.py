# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Forced CuTe B x Q sweep against F.linear with rotating weights.

Each graph visits R copies with R*weight_bytes >= 3*L2; per-call time is replay/R.
Rotation avoids L2 residency and amortizes event-timer resolution of ~2 us on
some nodes. The companion linear_smallm.py acceptance graphs additionally
target >=1 GiB per graph (>=3x L2, capped at 256 copies and 6 GB), with timing
per call. Each candidate graph is captured before its cuBLAS baseline graph
because subsequent captures clear cuBLAS workspaces. Timing uses torch's default
allow_bf16_reduced_precision_reduction=True; only the accuracy baseline disables it.
RESULT smallm_probe rows feed linear_smallm_table.py. The cooperative budget
includes setup, compilation and checks; running CUDA work is never interrupted.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.kernel_tuning.linear_smallm import (
    FP32_MARKET_PROVENANCE,
    MARKET_PROVENANCE,
    READ_GBPS,
    accuracy,
    builtin_shapes,
    emit,
    parse_ms,
    reference_and_condition,
)
from ctikernels.linear._dispatch import bxq_configs, device_class, parse_config
from ctikernels.linear.cute import mtiles_for
from ctikernels.linear.inference import Linear


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    selection = ap.add_mutually_exclusive_group()
    selection.add_argument("--model", choices=[*builtin_shapes(), "all"], default="qwen3.5-9b")
    selection.add_argument("--shapes", help="N:K,N:K")
    ap.add_argument("--out-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--m", default="bxq")
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--budget", type=float, default=1600)
    ap.add_argument(
        "--configs",
        help="explicit candidates 'rows,tile,stages,splits[,warps,variant,prefetch];...' "
        "replacing the offline B x Q list",
    )
    args = ap.parse_args(argv)
    try:
        args.m = parse_ms(args.m)
        if args.configs is not None:
            args.configs = list(dict.fromkeys(parse_config(c) for c in args.configs.split(";")))
        if args.shapes is not None:
            shapes = [tuple(map(int, spec.split(":"))) for spec in args.shapes.split(",")]
            if any(len(shape) != 2 for shape in shapes):
                raise ValueError("shapes must be N:K pairs")
        else:
            shapes = [
                (r["N"], r["K"])
                for model, rows in builtin_shapes().items()
                if args.model in (model, "all")
                for r in rows
            ]
        if any(n <= 0 or k <= 0 or k % 8 for n, k in shapes):
            raise ValueError("positive N/K and K divisible by 8 required")
        if args.pairs < 1 or not math.isfinite(args.budget) or args.budget <= 0:
            raise ValueError("pairs and finite budget must be positive")
        args.shapes = sorted(set(shapes))
    except ValueError as exc:
        ap.error(str(exc))
    return args


def rotation_count(n, k, l2_bytes):
    if min(n, k, l2_bytes) <= 0:
        raise ValueError("positive shape and L2 size required")
    return max(2, math.ceil(3 * l2_bytes / (2 * n * k)))


def graph_of(fn, weights, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            for w in weights:
                fn(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [fn(x, w) for w in weights]
    torch.cuda.synchronize()
    return graph, outputs


def time_graph(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    values = []
    for _ in range(3):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000)
    return statistics.median(values)


def error_metrics(output, baseline, ref):
    ours = (output.double() - ref).abs().max().item()
    cublas = (baseline.double() - ref).abs().max().item()
    return {
        "correct": math.isfinite(ours) and math.isfinite(cublas) and ours <= 2 * cublas + 1e-3,
        "max_abs": ours if math.isfinite(ours) else None,
        "cublas_max_abs": cublas if math.isfinite(cublas) else None,
        "accuracy_gate": "max_abs <= 2*cublas_max_abs + 1e-3 (float64 reference)",
    }


def probe_row(n, k, m, kind, cfg, metrics, ours=(), baseline=(), *, out_dtype="bf16", **metadata):
    """Use the probe schema consumed by select_configs, including rejected configs."""
    row = dict(
        kind="smallm_probe",
        out_dtype=out_dtype,
        device_class=kind,
        N=n,
        K=k,
        M=m,
        config=asdict(cfg),
        status="ok" if metrics["correct"] else "incorrect",
        sweep="tune",
        mode="cold",
        provider="cute",
        fastest_market="torch.mm(out_dtype=torch.float32)" if out_dtype == "fp32" else "F.linear",
        fastest_market_provenance=FP32_MARKET_PROVENANCE
        if out_dtype == "fp32"
        else MARKET_PROVENANCE,
        shared_bytes=cfg.shared_bytes_for(mtiles_for(m)),
        kernels_per_call=cfg.kernels,
        **metrics,
        **metadata,
    )
    if ours:
        ours_us, cublas_us = statistics.median(ours), statistics.median(baseline)
        sol_us = (2 * (n * k + m * k) + (4 if out_dtype == "fp32" else 2) * m * n) / (
            READ_GBPS[kind] * 1e3
        )
        row.update(
            ours_us=ours_us,
            cublas_us=cublas_us,
            paired_speedup=statistics.median(b / a for a, b in zip(ours, baseline, strict=True)),
            sol_us=sol_us,
            sol_fraction=sol_us / ours_us,
            cublas_sol_fraction=sol_us / cublas_us,
        )
    return row


class BudgetExpired(Exception):
    pass


def run(args, started):
    out_dtype = getattr(args, "out_dtype", "bf16")
    fp32 = out_dtype == "fp32"

    def market_call(x, w):
        return torch.mm(x, w.t(), out_dtype=torch.float32)

    if not fp32:
        market_call = F.linear

    def checkpoint(**context):
        elapsed = time.monotonic() - started
        if elapsed >= args.budget:
            emit(
                dict(
                    kind="budget",
                    out_dtype=out_dtype,
                    elapsed_seconds=elapsed,
                    budget_seconds=args.budget,
                    **context,
                )
            )
            raise BudgetExpired

    checkpoint(phase="setup")
    props = torch.cuda.get_device_properties(0)
    kind = device_class((props.major, props.minor), props.multi_processor_count)
    if kind not in READ_GBPS:
        raise ValueError("sweep requires a measured sm120/sm121 device class")
    smem_limit = min(99 * 1024, props.shared_memory_per_block_optin)
    gen = torch.Generator(device="cuda").manual_seed(0)
    for n, k in args.shapes:
        checkpoint(N=n, K=k, phase="shape")
        copies = rotation_count(n, k, props.L2_cache_size)
        emit(
            {
                "kind": "probe_environment",
                "out_dtype": out_dtype,
                "N": n,
                "K": k,
                "device_class": kind,
                "rotation_copies": copies,
                "rotation_bytes": copies * 2 * n * k,
                "L2_bytes": props.L2_cache_size,
                "timing_method": "rotating_weights",
            }
        )
        weights = []
        for _ in range(copies):
            checkpoint(N=n, K=k, phase="weights")
            weights.append(
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
            )
        xall = torch.randn(max(args.m), k, device="cuda", dtype=torch.bfloat16, generator=gen)
        # Bound temporary FP64 weight storage even for vocabulary heads.
        if fp32:
            checkpoint(N=n, K=k, phase="reference")
            ref, condition = reference_and_condition(xall, weights[0])
        else:
            ref = torch.empty(max(args.m), n, device="cuda", dtype=torch.float64)
            for lo in range(0, n, 4096):
                checkpoint(N=n, K=k, phase="reference")
                ref[:, lo : lo + 4096] = xall.double() @ weights[0][lo : lo + 4096].double().T
        for m in args.m:
            checkpoint(N=n, K=k, M=m, phase="baseline")
            x = xall[:m]
            # Accuracy baseline: full-precision cuBLAS; timing below uses torch defaults.
            flags = torch.backends.cuda.matmul
            try:
                flags.allow_bf16_reduced_precision_reduction = False
                baseline = market_call(x, weights[0])
            finally:
                flags.allow_bf16_reduced_precision_reduction = True

            for cfg in args.configs or bxq_configs(n, k, kind):
                checkpoint(N=n, K=k, M=m, phase="candidate")
                if cfg.shared_bytes_for(mtiles_for(m)) > smem_limit:
                    continue
                op = Linear(config=cfg, provider="cute")
                call = partial(op, out_dtype=torch.float32) if fp32 else op
                eager = call(x, weights[0])
                torch.cuda.synchronize()
                if op.provider != "cute":
                    raise RuntimeError("forced candidate did not run CuTe")
                metrics = (
                    accuracy(eager, baseline, ref[:m], condition[:m], k, out_dtype="fp32")
                    if fp32
                    else error_metrics(eager, baseline, ref[:m])
                )
                if not metrics["correct"]:
                    emit(probe_row(n, k, m, kind, cfg, metrics, out_dtype=out_dtype))
                    continue
                checkpoint(N=n, K=k, M=m, phase="capture")

                graph, outputs = graph_of(call, weights, x)
                graph.replay()
                torch.cuda.synchronize()
                metrics = (
                    accuracy(outputs[0], baseline, ref[:m], condition[:m], k, out_dtype="fp32")
                    if fp32
                    else error_metrics(outputs[0], baseline, ref[:m])
                )
                if not metrics["correct"]:
                    emit(probe_row(n, k, m, kind, cfg, metrics, out_dtype=out_dtype))
                    del graph, outputs
                    continue
                # Capture cuBLAS after the candidate: each capture clears cuBLAS
                # workspaces, so an older baseline graph could replay freed memory.
                baseline_graph, baseline_outputs = graph_of(market_call, weights, x)
                ours, cublas_times = [], []
                for pair in range(args.pairs):
                    checkpoint(N=n, K=k, M=m, phase="timing")
                    order = ((graph, ours), (baseline_graph, cublas_times))
                    for g, times in order if pair % 2 else reversed(order):
                        times.append(time_graph(g) / copies)
                emit(
                    probe_row(
                        n,
                        k,
                        m,
                        kind,
                        cfg,
                        metrics,
                        ours,
                        cublas_times,
                        out_dtype=out_dtype,
                        pairs=args.pairs,
                        cublas_bf16_reduced_precision=True,
                        rotation_copies=copies,
                        timing_method="rotating_weights",
                    )
                )
                del graph, outputs, op, eager, order, g, times
                del baseline_graph, baseline_outputs
            del baseline
        del weights, xall, x, ref
        torch.cuda.empty_cache()
    checkpoint(phase="completion")


def main(argv=None):
    started = time.monotonic()
    args = parse_args(argv)
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    complete = False
    failures = 0
    try:
        # Time cuBLAS with torch's default flag, as the engine, vLLM and SGLang run it.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode():
            run(args, started)
        complete = True
    except BudgetExpired:
        pass
    except Exception as exc:  # noqa: BLE001 -- stop after CUDA failure; bounded diagnostics
        emit({"kind": "error", "out_dtype": args.out_dtype, "error": type(exc).__name__})
        failures = 1
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    emit(
        {
            "kind": "probe_completion",
            "out_dtype": args.out_dtype,
            "complete": complete,
            "failures": failures,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return int(not complete or failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
