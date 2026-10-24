# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""W4A8 routed experts at DeepSeek V4 Flash shapes: CUDA-graph latency, cold or warm L2.

Cold: every timed call follows an in-graph write of 2x L2 bytes, so weights come from DRAM as
in a real decode step (each layer has its own experts). Prints one JSON row per (M, mode).
``--sweep`` times launch-config candidates of the Triton provider (cold for M <= 4, warm above),
the input of its fixed per-device table. ``--provider old:<path>`` times another ctikernels tree's ``Dsv4RoutedExperts`` (comparators:
``flashinfer`` CUTLASS MXFP4 or ``cute`` W4A8) with the same weights, routes and timer.

    python benchmark/kernel_tuning/moe_w4a8.py --ms 1 2 3 4 512 2048 8192
"""

import argparse
import json
import statistics
import sys

import torch

H, INTER, EXPERTS, TOP_K = 4096, 2048, 256, 6


def weights(local, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)

    def scales(*shape, center):
        return torch.randint(center - 2, center + 2, shape, generator=g, device="cuda").to(
            torch.uint8
        )

    return (
        nibbles(local, 2 * INTER, H // 2),
        scales(local, 2 * INTER, H // 32, center=121),
        nibbles(local, H, INTER // 2),
        scales(local, H, INTER // 32, center=122),
    )


def routes(m, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g, device="cuda")[:TOP_K] for _ in range(m)]
    ).to(torch.int32)
    w = torch.rand(m, TOP_K, generator=g, device="cuda")
    return (w / w.sum(-1, keepdim=True)).contiguous(), ids.contiguous()


def time_graph(fn, *, cold, calls=20, samples=15):
    """Median and p10/p90 microseconds of ``fn`` replayed inside one graph."""
    fn()
    torch.cuda.synchronize()
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    # Cold L2 by reading 2x L2 of clean lines: a write flush would leave dirty lines whose
    # writeback lands inside the timed window. The row sums write only 1/1024 of that.
    flush = torch.ones((max(1, 2 * l2 // 4096), 1024), dtype=torch.float32, device="cuda")
    sink = torch.empty(flush.shape[0], dtype=torch.float32, device="cuda")
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(calls)
    ]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
        with torch.cuda.graph(graph, stream=stream):
            for start, end in events:
                if cold:
                    torch.sum(flush, dim=1, out=sink)
                start.record()
                fn()
                end.record()
    torch.cuda.current_stream().wait_stream(stream)
    times = []
    for _ in range(samples):
        graph.replay()
        torch.cuda.synchronize()
        times += [s.elapsed_time(e) * 1e3 for s, e in events]
    times.sort()
    q = lambda f: times[int(f * (len(times) - 1))]
    return {"us": statistics.median(times), "p10": q(0.1), "p90": q(0.9)}


def bytes_moved(m, ids, local=EXPERTS):
    touched = ids[ids < local].unique().numel()
    per_expert = 2 * INTER * H // 2 + 2 * INTER * H // 32 + H * INTER // 2 + H * INTER // 32
    return touched * per_expert + m * H * 2 * 2


def build(provider, w, local):
    names = ("triton", "flashinfer", "flashinfer-default", "cute", "cute-atomic")
    if provider in names or provider.startswith("flashinfer:"):
        from ctikernels.moe import MXFP4Experts

        if provider == "flashinfer-default":  # FlashInfer's default tactics: no shipped cache
            from ctikernels.moe.flashinfer import _w4a8 as fi

            fi.tactics = lambda *_: None
        kind = {"flashinfer-default": "flashinfer", "cute-atomic": "cute"}.get(provider, provider)
        op = MXFP4Experts(
            *w, swiglu_limit=10.0, provider=kind.split(":")[0], scales_layout="canonical"
        )
        if provider == "cute-atomic":  # the same kernels with the BF16 atomic combine
            from ctikernels.moe._common import ExpertWeights
            from ctikernels.moe.cute._w4a8 import Config, W4A8Experts

            # The op holds its scales in the 128x4 interleave the kernels read.
            words = (op._weights[1].view(torch.int32), op._weights[3].view(torch.int32))
            op._cute = W4A8Experts(
                ExpertWeights(*op._weights),
                config=Config(up_first=True, combine="atomic"),
                scales=words,
            )

        expert_map = None
        if local < EXPERTS:  # EP rank 0 owns the first ``local`` experts
            expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
            expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")

        def run(x, rw, ids, out):
            return op(x, rw, ids, out=out, expert_map=expert_map)

        if provider.startswith("flashinfer:"):
            # Offline tuning: profile once per M into the cache file, then time from it.
            op._flashinfer.tactics = provider.split(":", 1)[1]

            def prepare(m):
                x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
                op._flashinfer.tune = True
                run(x, *routes(m), torch.empty_like(x))
                op._flashinfer.tune = False

            run.prepare = prepare
        return run
    kind = provider.split(":")[2]
    from ctikernels.moe._common import ExpertWeights, Geometry
    from ctikernels.moe.inference import Dsv4RoutedExperts

    geometry = Geometry(hidden=H, intermediate=INTER, experts=EXPERTS)
    op = Dsv4RoutedExperts(ExpertWeights(*w), provider=kind, geometry=geometry)

    def run(x, rw, ids, out):
        return op(x, ids, rw, out=out, routes_unique=True)

    run.prepare = op.prepare
    return run


def candidates(m):
    """(bm, bn1, bk1, bn2, bk2, warps, stages, cvt) to time at M tokens."""
    import itertools

    if m * TOP_K <= 64:
        grid = itertools.product(
            [16],
            [(32, 128), (32, 256), (64, 128)],
            [(32, 128), (64, 128), (64, 256), (128, 128)],
            [2, 3],
            [True, False],
        )
        return [(bm, *f1, *f2, 4, st, cvt) for bm, f1, f2, st, cvt in grid]
    grid = itertools.product(
        [32, 64, 128],
        [(64, 128), (128, 128), (64, 64)],
        [(128, 128), (64, 128), (256, 64)],
        [True, False],
    )
    return [
        (bm, *f1, *f2, 8 if bm == 128 else 4, 2 if bm == 128 else 3, cvt)
        for bm, f1, f2, cvt in grid
    ]


def sweep(w, ms):
    from ctikernels.moe import MXFP4Experts
    from ctikernels.moe.triton import _w4a8

    op = MXFP4Experts(*w, swiglu_limit=10.0, provider="triton", scales_layout="canonical")
    device = torch.cuda.get_device_name()
    keys = ("bm", "bn1", "bk1", "bn2", "bk2", "warps", "stages", "cvt")
    for m in ms:
        x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
        rw, ids = routes(m)
        out = torch.empty_like(x)
        mode = "cold" if m <= 4 else "warm"
        call = lambda x=x, rw=rw, ids=ids, out=out: op(x, rw, ids, out=out)
        for cand in candidates(m):
            cfg = dict(zip(keys, cand, strict=True))
            _w4a8.config = lambda capability, rows, cfg=cfg: cfg
            row = {"device": device, "provider": "triton", "M": m, "mode": mode, "config": cfg}
            try:
                row.update(time_graph(call, cold=mode == "cold"))
            except Exception as e:  # noqa: BLE001 - a candidate that does not compile is skipped
                row["error"] = str(e)[-200:]
            print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4, 512, 2048, 8192])
    p.add_argument(
        "--provider",
        default="triton",
        help="triton | flashinfer[:<tactic cache to tune>] | flashinfer-default | cute | cute-atomic | old:<python dir>:<flashinfer|cute>",
    )
    p.add_argument("--modes", nargs="+", default=["cold", "warm"])
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--local", type=int, default=EXPERTS, help="local experts (EP rank 0)")
    args = p.parse_args()
    if args.sweep:
        sweep(weights(EXPERTS), args.ms)
        return
    if args.provider.startswith("old:"):
        sys.path.insert(0, args.provider.split(":")[1])
    w = weights(args.local)
    run = build(args.provider, w, args.local)
    device = torch.cuda.get_device_name()
    for m in args.ms:
        x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
        rw, ids = routes(m)
        out = torch.empty_like(x)
        if hasattr(run, "prepare"):
            run.prepare(m)
        call = lambda x=x, rw=rw, ids=ids, out=out: run(x, rw, ids, out)
        for mode in args.modes:
            label = (
                args.provider.rsplit(":", 1)[-1]
                if args.provider.startswith("old:")
                else args.provider
            )
            row = {"device": device, "provider": label, "M": m, "local": args.local}
            row.update(mode=mode, **time_graph(call, cold=mode == "cold"))
            row["GBps"] = bytes_moved(m, ids, args.local) / row["us"] / 1e3
            row["TFLOPs"] = 2 * m * TOP_K * 3 * H * INTER / row["us"] / 1e6
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
