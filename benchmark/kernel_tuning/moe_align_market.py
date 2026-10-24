# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""moe_align_block_size market: our Triton alignment (the path that replaced the sgl_kernel
fallback) vs sgl_kernel's moe_align_block_size (sglang-kernel 0.4.6.post1, called as the old
fallback did: E + 1 buckets, padded). Top-8 ids, block 64; graph-replayed and eager, paired A-B-A,
median us per call; speedup = market / ours. Also: same routes per block as sgl_kernel, and
whether five repeated calls give identical sorted ids (run-to-run determinism)."""

import os
import statistics

import torch

from ctikernels.moe.triton._slot_layout import align_block_size

K, BS = 8, 64


def timed(fn, graph, inner=20, outer=15):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn()
        run = g.replay
    else:

        def run():
            for _ in range(inner):
                fn()

    run()
    torch.cuda.synchronize()
    times = []
    for _ in range(outer):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / inner)
    return statistics.median(times)


def sgl_align(sgl_kernel, ids, e):
    capacity = ids.numel() + (e + 1) * (BS - 1)
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=ids.device)
    experts = torch.empty(-(-capacity // BS), dtype=torch.int32, device=ids.device)
    total = torch.empty(1, dtype=torch.int32, device=ids.device)
    cumsum = torch.empty(e + 2, dtype=torch.int32, device=ids.device)
    sgl_kernel.moe_align_block_size(ids, e + 1, BS, sorted_ids, experts, total, cumsum, True)
    return sorted_ids, experts, total


def same_routes(a, b):
    n = int(a[2].item())
    if n != int(b[2].item()) or not torch.equal(a[1][: n // BS], b[1][: n // BS]):
        return False
    rows = lambda x: x[:n].view(-1, BS).sort(1).values
    return torch.equal(rows(a[0]), rows(b[0]))


def deterministic(fn):
    first = fn()[0]
    return all(torch.equal(first, fn()[0]) for _ in range(5))


def case(sgl_kernel, e, t):
    gen = torch.Generator(device="cuda").manual_seed(t + e)
    ids = torch.rand(t, e, generator=gen, device="cuda").topk(K, 1).indices.int()
    n = ids.numel()

    def ours():
        return align_block_size(ids, BS, e, invalid_bucket=True)

    markets = {"sgl": lambda: sgl_align(sgl_kernel, ids, e)}
    line = f"E={e} T={t} routes={n}"
    for mode in ("graph", "eager"):
        graph = mode == "graph"
        first = timed(ours, graph)
        market = {name: timed(fn, graph) for name, fn in markets.items()}
        mine = (first + timed(ours, graph)) / 2
        line += f" | {mode} triton {mine:.2f}us"
        for name, us in market.items():
            line += f" {name} {us:.2f}us ({us / mine:.2f}x)"
    return line + (
        f" | same-routes-vs-sgl {same_routes(ours(), markets['sgl']())}"
        f" deterministic triton {deterministic(ours)} sgl {deterministic(markets['sgl'])}"
    )


def profile():
    """Host time per call (eager), and where it goes (cProfile, top entries)."""
    import cProfile
    import pstats
    import time

    ids = torch.randint(0, 128, (2048, K), dtype=torch.int32, device="cuda")
    for _ in range(5):
        align_block_size(ids, BS, 128, invalid_bucket=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        align_block_size(ids, BS, 128, invalid_bucket=True)
    print(f"host us/call {(time.perf_counter() - start) / 200 * 1e6:.1f}")
    torch.cuda.synchronize()
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(200):
        align_block_size(ids, BS, 128, invalid_bucket=True)
    prof.disable()
    pstats.Stats(prof).sort_stats("tottime").print_stats(8)
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    e, t = int(os.environ.get("PROF_E", 128)), int(os.environ.get("PROF_T", 4096))
    ids = torch.randint(0, e, (t, K), dtype=torch.int32, device="cuda")
    with torch_profile(activities=[ProfilerActivity.CUDA]) as gpu:
        for _ in range(50):
            align_block_size(ids, BS, e, invalid_bucket=True)
        torch.cuda.synchronize()
    print(gpu.key_averages().table(sort_by="cuda_time_total", row_limit=8))


def main():
    import sys

    if "--profile" in sys.argv:
        profile()
        return
    import sgl_kernel

    if "--sweep" in sys.argv:  # launch configurations at the anchor (graph), checked bitwise
        from ctikernels.moe.triton import _slot_layout as lay

        want = None
        import itertools

        t = int(os.environ.get("SWEEP_T", 4096))
        ids = torch.rand(t, 128, device="cuda").topk(K, 1).indices.int()
        e = int(os.environ.get("SWEEP_E", 128))
        ids = torch.rand(t, e, device="cuda").topk(K, 1).indices.int()
        grid = itertools.product((128, 256), (8, 16, 32), (1, 2))
        for chunk, eb, wp in grid:
            if True:
                if True:
                    lay._CHUNK, lay._EB = chunk, eb
                    lay._place_config = lambda nb, nch, wp=wp: (False, 32, wp)
                    got = align_block_size(ids, BS, e, invalid_bucket=True)
                    if want is None:
                        want = got
                    same = all(torch.equal(x, y) for x, y in zip(got, want, strict=True))
                    us = timed(lambda: align_block_size(ids, BS, e, invalid_bucket=True), True)
                    print(
                        f"E {e} T {t} chunk {chunk} eb {eb} wp {wp}: {us:.2f}us same {same}",
                        flush=True,
                    )
        return
    if "--anchor" in sys.argv:  # the tuning anchor: E=128, T=4096 x top-8
        print(torch.cuda.get_device_name(), flush=True)
        print(case(sgl_kernel, 128, 4096), flush=True)
        profile()
        return

    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    for e in (128, 256):
        for t in (64, 256, 512, 1024, 2048, 4096, 8192):
            print(case(sgl_kernel, e, t), flush=True)


if __name__ == "__main__":
    main()
