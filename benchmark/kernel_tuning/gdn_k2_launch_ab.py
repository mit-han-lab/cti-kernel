# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Same-process ABBA launch-policy K-round graphs, timed in us per round."""

import argparse
import json
import os
import random
from itertools import product
from statistics import median

import torch

from benchmark.kernel_tuning.gdn_decode_compare import _positive_int, _shape
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_verify_compare import GraphRound, Ours, fold_period, percentile


def capture(inputs, setting, **kwargs):
    import ctikernels.gdn.replay.cute._phase_launch as launch

    names = (
        "PHASE_TILE",
        "PHASE_WARPS",
        "PHASE_MIN_BLOCKS",
        "VECTOR_PUBLISH",
        "PDL",
        "FUSED_PUBLISH",
    )
    saved = {name: getattr(launch, name) for name in names}
    try:
        provider = Ours(*inputs, **kwargs)
        # Ours resets the overrides; apply them AFTER construction and keep them
        # through GraphRound's warmup and capture. Replays need no host overrides.
        launch.PHASE_TILE, launch.PHASE_WARPS, launch.PHASE_MIN_BLOCKS = setting
        op = provider.runner.op
        batch, q, _ = provider.drafts.shape
        props = torch.cuda.get_device_properties(provider.drafts.device)
        expected = (
            launch.phase_launch(batch, kwargs["hv"], props.multi_processor_count, q, op.cute_config)
            if setting == (None, None, None)
            else setting
        )
        graph = GraphRound(provider)
        # Inspect the actual warmed specialization keys (config=7, tile=9).
        for mode in ("verify", "commit"):
            effective = {
                (key[9], key[7].warps, key[7].min_blocks)
                for key in op._cute_launchers
                if key[0] == mode
            }
            assert effective == {expected}, (mode, effective, expected)
        provider.effective_launch = expected
        provider.effective_tile = "/".join(map(str, expected))
        return graph
    finally:
        for name, value in saved.items():
            setattr(launch, name, value)


def measure(hk, hv, q, batch, window, commit_mode, b_setting, pair_count):
    weights, x, initial, history = case_inputs(hk, hv, batch, batch, q + 1, "ordinary", 23, "cuda")
    drafts = x[-q:].transpose(0, 1).contiguous().view(batch, q, -1)
    rounds = fold_period("delayed", q, window)
    a, b = [
        capture(
            (weights, drafts, initial, history),
            setting,
            commit_mode=commit_mode,
            window=window,
            rounds=rounds,
            hk=hk,
            hv=hv,
            ring=True,
        )
        for setting in ((None, None, None), b_setting)
    ]
    effective_a, effective_b = (g.provider.effective_launch for g in (a, b))
    if effective_a != b_setting:
        assert effective_a != effective_b, "launch override did not reach the captured graph"

    # A full fold cycle gives a canonical endpoint even when auto selects delayed.
    a.sample()
    b.sample()
    tensors = list(zip(a.provider.outputs, b.provider.outputs, strict=True))
    for name in ("checkpoint", "conv_history"):
        tensors.append(
            (getattr(a.provider.runner.state, name), getattr(b.provider.runner.state, name))
        )
    bitwise = all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in tensors)

    pairs, self_pairs = [], []
    for _ in range(0, pair_count, 2):
        a0, b0, b1, a1 = a.sample(), b.sample(), b.sample(), a.sample()
        block = [(a0, b0), (a1, b1)][: pair_count - len(pairs)]
        pairs.extend(block)
        self_pairs.extend((a.sample(), a.sample()) for _ in block)
    ratios = [x / y for x, y in pairs]
    rng = random.Random(23)
    boot = [median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
    return {
        "q": q,
        "b": batch,
        "shape": [hk, hv],
        "window": window,
        "K": rounds,
        "commit_mode": a.provider.commit_mode,
        "effective_a": effective_a,
        "effective_b": effective_b,
        "a_us": median(x for x, _ in pairs),
        "b_us": median(y for _, y in pairs),
        "median_ratio_a_over_b": median(ratios),
        "eps": percentile([abs(x / y - 1) for x, y in self_pairs], 0.95),
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "bitwise": bitwise,
        "samples": {"pairs_us": pairs, "self_pairs_us": self_pairs},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_shape, default="8,16", metavar="HK,HV")
    parser.add_argument("--q", nargs="+", type=int, default=[4, 5, 6, 8, 9, 12, 16])
    parser.add_argument(
        "--batches", nargs="+", type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4]
    )
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--commit-mode", choices=("eager", "auto"), default="eager")
    parser.add_argument("--pairs", type=_positive_int, default=20)
    parser.add_argument("--b-tile", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--b-warps", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--b-min-blocks", type=int, choices=(1, 2, 3), default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = {"device": torch.cuda.get_device_name()}
    if hasattr(os, "sched_getaffinity"):
        device["affinity"] = sorted(os.sched_getaffinity(0))
    print(json.dumps(device), flush=True)
    hk, hv = args.shape
    with torch.inference_mode():
        for q, batch in product(args.q, args.batches):
            row = measure(
                hk,
                hv,
                q,
                batch,
                args.window,
                args.commit_mode,
                (args.b_tile, args.b_warps, args.b_min_blocks),
                args.pairs,
            )
            print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
