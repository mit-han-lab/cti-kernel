# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Measured graph-replay floor for launch-bound DRAFT rows, as JSONL on stdout.

The market gate times each complete op as back-to-back ``CUDAGraph.replay()``
calls between two synchronizations (``dflash_draft_attention.measure``). Short
rows then sit on a host/device replay step that no kernel can go below. This
script measures that step with the same protocol: graphs holding k = 0..4 and 8
one-element kernels (k = 0 is an empty capture), 30 intervals of 100 replays
each. The per-kernel step is the least-squares slope over k >= 1.

The SOL floor of a complete op with k kernels is the measured replay median of
the k-kernel graph; see dflash_draft_sol_floor.py for how it enters t_ref.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import warnings

import torch

from benchmark.kernel_tuning.dflash_draft_attention import emit, measure
from benchmark.kernel_tuning.dflash_draft_market import pin_x925

KERNELS = (0, 1, 2, 3, 4, 8)


def capture(kernels, scratch):
    def body():
        for _ in range(kernels):
            scratch.add_(1)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings():
        # An empty capture warns; replaying it is still a valid host step.
        warnings.simplefilter("ignore")
        with torch.cuda.graph(graph, stream=stream):
            body()
    stream.synchronize()
    return graph


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=int, default=30)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--pin-x925", action="store_true")
    args = parser.parse_args(argv)
    affinity = pin_x925(args.pin_x925 or platform.machine() == "aarch64")
    device = torch.device("cuda")
    scratch = torch.zeros(1, device=device)
    graphs = {k: capture(k, scratch) for k in KERNELS}
    for graph in graphs.values():
        for _ in range(200):
            graph.replay()
    samples = {k: [] for k in KERNELS}
    # Interleave graph sizes so clock drift spreads over every k.
    for _ in range(args.intervals):
        for k in KERNELS:
            samples[k].append(measure(graphs[k].replay, args.calls))
    medians = {k: statistics.median(v) for k, v in samples.items()}
    xs = [k for k in KERNELS if k]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(medians[k] for k in xs)
    slope = sum((k - mean_x) * (medians[k] - mean_y) for k in xs) / sum(
        (k - mean_x) ** 2 for k in xs
    )
    emit(
        {
            "type": "replay_floor",
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "affinity": affinity,
            "intervals": args.intervals,
            "calls_per_interval": args.calls,
            "replay_us": {
                str(k): {
                    "median": medians[k],
                    "min": min(v),
                    "p90": sorted(v)[int(0.9 * (len(v) - 1))],
                }
                for k, v in samples.items()
            },
            "per_kernel_step_us": slope,
            "intercept_us": mean_y - slope * mean_x,
        }
    )


if __name__ == "__main__":
    main()
