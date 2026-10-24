# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GPU microbenchmark of per-call host time: shipped Linear vs F.linear.

Run with PYTHONPATH=python:. python -m benchmark.kernel_tuning.linear_host_overhead
--shapes 64:4096,4096:4096 --m 1,8,16,64. After warm-up, time 20 rounds of
100 back-to-back calls with perf_counter_ns, synchronizing between rounds,
outside timing. Providers interleave each round with alternating order. Report
median and p90 microseconds per call and the difference of provider medians.
Small shapes expose host overhead; large shapes may throttle the GPU queue,
so their host times can include time waiting for the GPU to accept more work.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning.linear_smallm import emit, parse_ms
from ctikernels.linear import Linear

ROUNDS = 20
CALLS = 100


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="64:4096,4096:4096", help="comma-separated N:K pairs")
    parser.add_argument("--m", default="1,8,16,64", help="comma-separated M values in 1..64")
    args = parser.parse_args(argv)
    try:
        args.shapes = list(
            dict.fromkeys(tuple(map(int, s.split(":"))) for s in args.shapes.split(","))
        )
        if any(len(shape) != 2 for shape in args.shapes):
            raise ValueError("shapes must be N:K pairs")
        if any(n <= 0 or k <= 0 or k % 8 for n, k in args.shapes):
            raise ValueError("N/K must be positive and K divisible by eight")
        args.m = parse_ms(args.m)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def measure(op, x, weight):
    calls = {"linear": op, "f_linear": F.linear}
    samples = {name: [] for name in calls}
    for call in calls.values():
        for _ in range(10):
            call(x, weight)
    torch.cuda.synchronize()
    for round_index in range(ROUNDS):
        order = tuple(calls) if round_index % 2 == 0 else tuple(reversed(calls))
        for name in order:
            call = calls[name]
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for _ in range(CALLS):
                call(x, weight)
            elapsed = time.perf_counter_ns() - start
            samples[name].append(elapsed / CALLS / 1000)
    torch.cuda.synchronize()
    result = {}
    for name, times in samples.items():
        result[name + "_median_us"] = statistics.median(times)
        result[name + "_p90_us"] = statistics.quantiles(times, n=10, method="inclusive")[-1]
    result["median_delta_us"] = result["linear_median_us"] - result["f_linear_median_us"]
    return result


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        emit({"kind": "error", "reason": "CUDA_required"})
        return 1
    rows = 0
    device = torch.cuda.get_device_name()
    with torch.inference_mode():
        for n, k in args.shapes:
            x = torch.randn(max(args.m), k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            op = Linear(provider="auto")
            for m in args.m:
                result = measure(op, x[:m], weight)
                emit(
                    dict(
                        kind="host_overhead",
                        device=device,
                        N=n,
                        K=k,
                        M=m,
                        rounds=ROUNDS,
                        calls_per_round=CALLS,
                        provider=op.provider,
                        **result,
                    )
                )
                rows += 1
    emit({"kind": "host_overhead_completion", "complete": True, "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
