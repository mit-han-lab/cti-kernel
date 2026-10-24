# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired B1 versus PyNccl benchmark; one identically configured process per host."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from ctikernels.collective._two_node.discovery import big_cores, proxy_core
from ctikernels.collective._two_node.proxy import barrier
from ctikernels.collective.two_node import TwoNodeAllReduce
from ctikernels.pynccl import init_pynccl

CALLS = 43
DEFAULT_SIZES = (2048, 8192, 16384, 24576, 32768, 65536, 131072, 262144)
GATHER_SIZES = (129280, 258560, 387840, 517120)


def median_ci(ratios, resamples=10000, seed=2026):
    """Paired percentile bootstrap of the median; resample whole ratios."""
    rng = random.Random(seed)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples))
    return [draws[int(resamples * 0.025)], draws[min(resamples - 1, int(resamples * 0.975))]]


class Session:
    def __init__(self, rank, store):
        self.rank, self.store, self.counter = rank, store, 0

    def sync(self):
        key = f"bench/barrier/{self.counter}"
        self.counter += 1
        barrier(self.store, key, timeout_s=120)

    def exchange(self, key, value):
        self.store.set(f"bench/{key}/{self.rank}", json.dumps(value))
        return json.loads(self.store.get(f"bench/{key}/{1 - self.rank}"))

    def agree(self, key, ok):
        peer = self.exchange(key, ok)
        if not (ok and peer):
            raise RuntimeError(f"two-rank correctness check failed: {key}")


def cpu_input(rank, size):
    generator = torch.Generator().manual_seed(1709 + size + rank * 99991)
    return torch.randn(size // 2, generator=generator).to(torch.bfloat16)


def bits(x):
    return bytes(x.contiguous().view(torch.uint8).flatten().cpu().tolist())


def correctness(session, op, nccl, size, operation="all_reduce"):
    a, b = [cpu_input(rank, size) for rank in range(2)]
    x = (a if session.rank == 0 else b).cuda()
    if operation == "all_gather":
        ours = op.all_gather(x)
        market = torch.empty_like(ours)
        nccl.all_gather(market, x)
        reference = torch.stack((a, b))
        session.agree(f"correct/{size}", bits(ours) == bits(reference) == bits(market))
        return {"bitwise": True}
    ours, market = op(x), x.clone()
    nccl.all_reduce(market, "sum")
    reference = (a.float() + b.float()).to(torch.bfloat16)
    session.agree(f"correct/{size}", bits(ours) == bits(reference) == bits(market))
    oracle = a.double() + b.double()
    metrics = {}
    for name, value in (("ours", ours), ("nccl", market)):
        err = value.cpu().double() - oracle
        metrics[name] = {"max": err.abs().max().item(), "rms": err.square().mean().sqrt().item()}
    session.agree(
        f"floor/{size}", all(metrics["ours"][k] <= metrics["nccl"][k] for k in ("max", "rms"))
    )
    return metrics


def timed_sample(session, launch, repeats, calls=CALLS):
    session.sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # The store barrier releases the ranks up to about a millisecond apart. One untimed
    # block of the same calls first: it completes on both GPUs within one call of each
    # other, so the start event does not include the other rank's arrival.
    launch()
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    # Retain the slower rank for each matched sample.
    us = start.elapsed_time(end) * 1000 / (repeats * calls)
    peer = session.exchange(f"timing/{session.counter}", us)
    return max(us, peer)


def launchers(op, nccl, size, mode, operation="all_reduce"):
    # Zero is stable under repeated in-place NCCL sums: no overflow or resets
    # contaminate timed samples. Correctness uses independent random inputs.
    x = torch.zeros(size // 2, dtype=torch.bfloat16, device="cuda")
    if operation == "all_gather":
        y = torch.empty((2, *x.shape), dtype=x.dtype, device=x.device)
        market = torch.empty_like(y)
    else:
        y, market = torch.empty_like(x), x.clone()

    def ours():
        if operation == "all_gather":
            op.all_gather(x, y)
        else:
            for _ in range(CALLS):
                op(x, y)

    def baseline():
        if operation == "all_gather":
            nccl.all_gather(market, x)
        else:
            for _ in range(CALLS):
                nccl.all_reduce(market, "sum")

    # Warm up both before capture, with identical rank ordering.
    ours()
    baseline()
    torch.cuda.synchronize()
    if mode == "eager":
        return {"ours": ours, "nccl": baseline}
    graphs = {}
    for name, fn in (("ours", ours), ("nccl", baseline)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    # Closures keep graph buffers alive for all samples.
    return {
        "ours": lambda: (x, y, graphs["ours"].replay())[-1],
        "nccl": lambda: (x, market, graphs["nccl"].replay())[-1],
    }


def measure(session, op, nccl, size, mode, pairs, operation="all_reduce"):
    calls = 1 if operation == "all_gather" else CALLS
    funcs = launchers(op, nccl, size, mode, operation)
    pilots = {name: timed_sample(session, fn, 1, calls) for name, fn in funcs.items()}
    # Calibrate each implementation to about 3 ms. Repeat counts are agreed
    # from slower-rank pilots, so both ranks issue precisely the same calls.
    repeats = {name: max(1, round(3000 / (calls * us))) for name, us in pilots.items()}
    samples = []
    for pair in range(pairs):
        order = ("nccl", "ours") if pair % 2 == 0 else ("ours", "nccl")
        row = {name: timed_sample(session, funcs[name], repeats[name], calls) for name in order}
        row["ratio"] = row["nccl"] / row["ours"]
        samples.append(row)
    ratios = [row["ratio"] for row in samples]
    ci = median_ci(ratios)
    return {
        "bytes": size,
        "mode": mode,
        "pairs": pairs,
        "repeats": repeats,
        "nccl_us": statistics.median(row["nccl"] for row in samples),
        "ours_us": statistics.median(row["ours"] for row in samples),
        "median_ratio": statistics.median(ratios),
        "ratio_ci95": ci,
        "pass": ci[0] > 1,
        "samples": samples,
    }


def idle_cpu(session, op):
    op.disarm()
    torch.cuda.synchronize()
    session.sync()
    time.sleep(0.05)
    start = op.stats()
    time.sleep(2)
    end = op.stats()
    fraction = (end["cpu_ns"] - start["cpu_ns"]) / (end["wall_ns"] - start["wall_ns"])
    peer = session.exchange("idle", fraction)
    return {"local": fraction, "peer": peer, "max": max(fraction, peer)}


def wake_penalty(session, op):
    """Extra latency when rank 0's proxy has backed off: rank 0 idles for a CPU-side gap
    while rank 1, its proxy held hot, already waits on the GPU; then rank 0 times two
    back-to-back 8 KiB calls on its own GPU (its doorbell goes through its own proxy, and
    the second call needs rank 1's reply, so the time includes rank 0's proxy wake-up
    once). Policies for rank 0's proxy: ``held`` (hot throughout, the reference),
    ``armed`` (arm() right before the launch, as the engine does at a step boundary),
    ``rest`` (its own backoff). Penalty = median minus the ``held`` median.
    """
    x = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    rows = []
    for gap_ms in (2, 20, 100):
        for policy in ("held", "armed", "rest"):
            samples = []
            for repeat in range(15):
                op.disarm()
                torch.cuda.synchronize()
                session.sync()
                if session.rank == 1 or policy == "held":
                    op.arm()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                if session.rank == 0:
                    time.sleep(gap_ms / 1000)
                    if policy == "armed":
                        op.arm(2000)
                start.record()
                op(x, y)
                op(x, y)
                end.record()
                end.synchronize()
                op.disarm()
                us = start.elapsed_time(end) * 1000
                peer = session.exchange(f"wake/{gap_ms}/{policy}/{repeat}", us)
                samples.append(us if session.rank == 0 else peer)
            rows.append(
                {
                    "gap_ms": gap_ms,
                    "policy": policy,
                    "median_us": statistics.median(samples),
                    "samples_us": samples,
                }
            )
    for row in rows:
        held = next(r for r in rows if r["gap_ms"] == row["gap_ms"] and r["policy"] == "held")
        row["penalty_us"] = row["median_us"] - held["median_us"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--store", required=True, help="rank-0 TCPStore address:port")
    parser.add_argument("--op", choices=("all_reduce", "all_gather"), default="all_reduce")
    parser.add_argument("--sizes", type=int, nargs="+", help="per-rank byte counts")
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ctas", type=int, help="override CTA policy for a sweep")
    parser.add_argument("--threads", type=int, default=512, choices=(128, 256, 512))
    args = parser.parse_args()
    if args.sizes is None:
        args.sizes = GATHER_SIZES if args.op == "all_gather" else DEFAULT_SIZES
    if args.pairs < 60:
        parser.error("--pairs must be >=60")
    limit = (1 << 24) - 16 if args.op == "all_gather" else 256 * 1024
    if any(size <= 0 or size % 16 or size > limit for size in args.sizes):
        parser.error(f"sizes must be positive multiples of 16 through {limit} bytes")
    core = proxy_core()
    cores = sorted(set(big_cores()) & os.sched_getaffinity(0) - {core})
    if not cores:
        raise RuntimeError("need an available big core besides the proxy core")
    os.sched_setaffinity(0, cores)
    # Set before NCCL is loaded (an explicit env value wins for config sweeps);
    # its proxy threads inherit this process's mask.
    os.environ.setdefault("NCCL_MAX_NCHANNELS", "2")
    torch.cuda.set_device(0)
    if args.ctas is not None or args.threads != 512:
        import ctikernels.collective.two_node as provider

        sms = torch.cuda.get_device_properties(0).multi_processor_count
        if args.ctas is not None and not 1 <= args.ctas <= sms:
            parser.error("--ctas must be between 1 and the SM count")
        provider.grid_policy = lambda size, sms: (
            args.ctas or min(sms, math.ceil(size / 65536)),
            args.threads,
        )
    address, port = args.store.rsplit(":", 1)
    store = dist.TCPStore(address, int(port), 2, args.rank == 0, timedelta(seconds=120))
    session = Session(args.rank, store)
    config = {
        "op": args.op,
        "sizes": list(args.sizes),
        "pairs": args.pairs,
        "ctas": args.ctas,
        "threads": args.threads,
    }
    if session.exchange("config", config) != config:
        raise ValueError("benchmark rank configurations differ")
    dist.init_process_group(
        "gloo",
        store=dist.PrefixStore("bench/gloo", store),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    nccl = init_pynccl(
        tp_rank=args.rank, tp_size=2, tp_cpu_group=dist.group.WORLD, max_size_bytes=0
    )
    max_bytes = max(256 * 1024, (max(args.sizes) + 15) // 16 * 16)
    op = TwoNodeAllReduce(
        rank=args.rank, store=store, prefix="bench/provider", proxy_cpu=core, max_bytes=max_bytes
    )
    if args.op == "all_gather":
        op.prepare_all_gather(args.sizes)
    else:
        op.prepare(args.sizes)
    report = {"config": config, "correctness": {}, "results": []}
    try:
        for size in args.sizes:
            report["correctness"][size] = correctness(session, op, nccl, size, args.op)
        # Measure the intended step-boundary arm hook. Idle tests disarm below.
        op.arm()
        for size in args.sizes:
            for mode in ("eager", "graph"):
                row = measure(session, op, nccl, size, mode, args.pairs, args.op)
                report["results"].append(row)
                if args.rank == 0:
                    lo, hi = row["ratio_ci95"]
                    print(
                        f"{size:7d} {mode:5s} NCCL {row['nccl_us']:8.3f} us "
                        f"B1 {row['ours_us']:8.3f} us ratio {row['median_ratio']:.3f} "
                        f"95% CI [{lo:.3f}, {hi:.3f}] pass={row['pass']}",
                        flush=True,
                    )
        report["idle_cpu"] = idle_cpu(session, op) if args.op == "all_reduce" else None
        report["wake_penalty"] = wake_penalty(session, op) if args.op == "all_reduce" else []
        op.check()
        if args.rank == 0:
            if report["idle_cpu"] is not None:
                print(f"Idle CPU fraction (worse rank): {report['idle_cpu']['max']:.3%}")
            for row in report["wake_penalty"]:
                print(
                    f"Wake {row['gap_ms']:3d} ms {row['policy']:>5}: {row['median_us']:.3f} us"
                    f" (penalty {row['penalty_us']:+.3f} us)"
                )
            if args.out:
                args.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        torch.cuda.synchronize()
        session.sync()
        op.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
