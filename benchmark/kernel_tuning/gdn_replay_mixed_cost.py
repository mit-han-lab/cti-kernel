# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Lower bound of the drain cost a mixed policy adds to #25 buckets.

A #25 bucket of a policy that also contains delayed buckets must make sure each
row's ring is empty before #25 reads the checkpoint. That needs at least one
device launch before #25. This measures the cheapest such launch (the CuTe
``ring_empty`` metadata check, B threads) followed by #25, against #25 alone:
CUDA graphs, cold rotation, 20 interleaved pairs, epsilon from 20 self-pairs.
"""

import argparse
import json
import math
import statistics

import torch

from benchmark.kernel_tuning.gdn_replay_common import Product, allocate
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from ctikernels.gdn.replay import GatedDeltaNetReplay


def graph(calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        for call in calls:
            call()
    return g


def trial(g, n):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="16,48")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--pairs", type=int, default=20)
    args = parser.parse_args()
    hk, hv = map(int, args.shape.split(","))
    props = torch.cuda.get_device_properties(0)
    for b in args.batches:
        copies = max(1, math.ceil(3 * props.L2_cache_size / (4 * b * hv * 128 * 128)))
        slots = copies * b
        weights, xs, initial, history = case_inputs(hk, hv, b, slots, 1, "ordinary", 11, "cuda")
        product = Product(hk, hv, weights.args(), initial, history)
        op = GatedDeltaNetReplay(
            num_k_heads=hk, num_v_heads=hv, device="cuda", provider="cute", mode="delayed"
        )
        state = op.create_state(
            buffers=allocate(op.state_spec(num_slots=slots, num_snapshot_rows=1), "cuda")
        )
        flag = torch.zeros(1, dtype=torch.bool, device="cuda")
        groups = [
            torch.arange(i * b, (i + 1) * b, dtype=torch.int32, device="cuda")
            for i in range(copies)
        ]
        n = copies * math.ceil(128 / copies)
        x = xs[0]

        def plain(group, product=product, x=x):
            return lambda: product.op(x, *product.weights, group, product.state, product.history)

        def gated(group, op=op, state=state, flag=flag, product=product, x=x):
            def call():
                op.ring_empty(state, group, flag)
                product.op(x, *product.weights, group, product.state, product.history)

            return call

        base = [plain(groups[i % copies]) for i in range(n)]
        g_a, g_a2 = graph(base), graph(base)
        g_g = graph([gated(groups[i % copies]) for i in range(n)])
        for g in (g_a, g_a2, g_g):
            trial(g, n)
        pairs, self = [], []
        for i in range(args.pairs):
            if i % 2:
                gt, at = trial(g_g, n), trial(g_a, n)
            else:
                at, gt = trial(g_a, n), trial(g_g, n)
            a1, a2 = trial(g_a, n), trial(g_a2, n)
            pairs.append((gt, at))
            self.append((a1, a2))
        eps = sorted(abs(a / c - 1) for a, c in self)[math.ceil(0.95 * len(self)) - 1]
        ratios = [at / gt for gt, at in pairs]
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "shape": [hk, hv],
                    "B": b,
                    "copies": copies,
                    "product_us": statistics.median(at for _, at in pairs),
                    "gated_us": statistics.median(gt for gt, _ in pairs),
                    "median_ratio_product_over_gated": statistics.median(ratios),
                    "epsilon": eps,
                    "regression_beyond_eps": statistics.median(ratios) < 1 - eps,
                }
            ),
            flush=True,
        )
        del g_a, g_a2, g_g, product, state
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
