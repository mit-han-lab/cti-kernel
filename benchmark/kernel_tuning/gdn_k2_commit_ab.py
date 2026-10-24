# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Same-process ABBA eager/delayed K-round graphs, normalized per round."""

import argparse
import json
from itertools import product
from statistics import median

import torch

from benchmark.kernel_tuning.gdn_decode_compare import _shape
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_verify_compare import GraphRound, Ours, fold_period, percentile


def measure(hk, hv, q, batch, window):
    weights, x, initial, history = case_inputs(hk, hv, batch, batch, q + 1, "ordinary", 23, "cuda")
    drafts = x[-q:].transpose(0, 1).contiguous().view(batch, q, -1)
    rounds = fold_period("delayed", q, window)
    providers = [
        Ours(
            weights,
            drafts,
            initial,
            history,
            commit_mode=mode,
            window=window,
            rounds=rounds,
            hk=hk,
            hv=hv,
            ring=True,
        )
        for mode in ("eager", "delayed")
    ]
    eager, delayed = [GraphRound(p) for p in providers]
    # Both sides execute the same K rounds. Check the entire cycle's outputs and
    # canonical endpoint outside timing, before collecting a performance receipt.
    eager.sample()
    delayed.sample()
    for a, b in zip(providers[0].outputs, providers[1].outputs, strict=True):
        if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            raise AssertionError("eager/delayed output mismatch")
    a, b = (p.runner.state for p in providers)
    for name in ("checkpoint", "conv_history"):
        if not torch.equal(getattr(a, name).view(torch.uint8), getattr(b, name).view(torch.uint8)):
            raise AssertionError(f"eager/delayed {name} mismatch")
    pairs, self_pairs = [], []
    # Ten ABBA blocks = 20 interleaved pairs; two eager self-pairs per block.
    for _ in range(10):
        a0, b0, b1, a1 = eager.sample(), delayed.sample(), delayed.sample(), eager.sample()
        pairs.extend(((a0, b0), (a1, b1)))
        self_pairs.extend((eager.sample(), eager.sample()) for _ in range(2))
    return {
        "q": q,
        "b": batch,
        "shape": [hk, hv],
        "window": window,
        "K": rounds,
        "commit_modes": [p.commit_mode for p in providers],
        "pairs": 20,
        "self_pairs": 20,
        "eager_us": median(a for a, _ in pairs),
        "delayed_us": median(b for _, b in pairs),
        "median_ratio_eager_over_delayed": median(a / b for a, b in pairs),
        "eps": percentile([abs(a / b - 1) for a, b in self_pairs], 0.95),
        "effective_launch": providers[0].effective_launch,
        "samples_us": pairs,
        "self_samples_us": self_pairs,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", nargs="+", type=int, default=[4, 5, 6, 8, 9, 12, 16])
    parser.add_argument(
        "--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4]
    )
    parser.add_argument("--shape", type=_shape, default="16,32", metavar="HK,HV")
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    hk, hv = args.shape
    with torch.inference_mode():
        for q, batch in product(args.q, args.batches):
            print(json.dumps(measure(hk, hv, q, batch, args.window), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
