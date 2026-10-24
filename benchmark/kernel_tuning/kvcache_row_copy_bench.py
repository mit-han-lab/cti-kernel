# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""RowCopy vs the engine's grouped index_copy restore, A-B-A, eager and CUDA graph.

The pool is the engine's recurrent pool for Qwen3.5-9B: per layer one byte buffer holding the
FP32 state [32, 128, 128] and the BF16 conv history [3, 8192] of every row. The baseline is
RecurrentStatePool.copy_rows: per group of 16 copies, index_select + index_copy_ per view.

    python -m benchmark.kernel_tuning.kvcache_row_copy_bench --copies 1 8 64
"""

import argparse
import json
import statistics

import torch

from ctikernels.kvcache.inference import RowCopy

KINDS = (((32, 128, 128), torch.float32), ((3, 8192), torch.bfloat16))
GROUP = 16


def pool(layers, rows):
    views = []
    for _ in range(layers):
        sizes = [rows * torch.Size(shape).numel() * dtype.itemsize for shape, dtype in KINDS]
        buffer = torch.zeros(sum(sizes), dtype=torch.uint8, device="cuda")
        offset = 0
        for (shape, dtype), size in zip(KINDS, sizes, strict=True):
            views.append(buffer[offset : offset + size].view(dtype).view(rows, *shape))
            offset += size
    return views


def engine(views, sources, targets):
    for start in range(0, sources.numel(), GROUP):
        s, t = sources[start : start + GROUP], targets[start : start + GROUP]
        for view in views:
            view.index_copy_(0, t, view.index_select(0, s))


def timed(call, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    call()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us per call


def graphed(call):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    return graph.replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 8, 64])
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    for n in args.copies:
        rows = 2 * n + 1
        views = pool(args.layers, rows)
        sources = torch.arange(n, dtype=torch.int64, device="cuda")
        targets = sources + n + 1
        op = RowCopy(views)
        op.warm_up()
        calls = {
            "engine": lambda v=views, s=sources, t=targets: engine(v, s, t),
            "rowcopy": lambda o=op, s=sources, t=targets: o(s, t),
        }
        for mode in ("eager", "graph"):
            run = {k: (graphed(c) if mode == "graph" else c) for k, c in calls.items()}
            # A-B-A per repeat: engine, ours, engine; the baseline is the mean of the two A arms.
            ratios, samples = [], {"engine": [], "rowcopy": []}
            for _ in range(args.repeats):
                a1 = timed(run["engine"], args.iters)
                b = timed(run["rowcopy"], args.iters)
                a2 = timed(run["engine"], args.iters)
                samples["engine"] += [a1, a2]
                samples["rowcopy"].append(b)
                ratios.append((a1 + a2) / 2 / b)
            print(json.dumps({
                "device": torch.cuda.get_device_name(), "copies": n, "layers": args.layers,
                "mode": mode, "engine_us": statistics.median(samples["engine"]),
                "rowcopy_us": statistics.median(samples["rowcopy"]),
                "speedup_median": statistics.median(ratios), "speedup_min": min(ratios),
            }), flush=True)  # fmt: skip
        del views, op
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
