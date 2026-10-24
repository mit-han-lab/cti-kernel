# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe dense/pitched layout cost: graph step and Q=8 verify + full commit.

Run from the checkout with ``python -m benchmark.kernel_tuning.gdn_replay_layout_cost``.
Uses the RP3 layout fixtures (including their pytest dependency) so the measured
strides, layer views and unaligned bases stay identical to the qualification tests.
Each graph contains 64 calls on the same state (warm-cache cost). State resets
are outside timing. Twenty alternating pairs and twenty dense self-pairs produce
the median pitched/dense ratio and nearest-rank p95 noise floor.
"""

import json
import statistics
from functools import partial

import torch

from benchmark.kernel_tuning.gdn_replay_bench import pair_stats
from benchmark.kernel_tuning.gdn_replay_mixed_cost import graph, trial
from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import Config


def measure_pairs(dense, pitched, dense_self):
    """Callables return microseconds; alternate pair order to limit drift bias."""
    pairs, self_pairs = [], []
    for i in range(20):
        if i % 2:
            p, d = pitched(), dense()
            d2, d1 = dense_self(), dense()
        else:
            d, p = dense(), pitched()
            d1, d2 = dense(), dense_self()
        pairs.append((p, d))
        self_pairs.append((d2, d1))
    stats = pair_stats(pairs, self_pairs)
    return {
        "dense_us": statistics.median(d for _, d in pairs),
        "pitched_us": statistics.median(p for p, _ in pairs),
        "ratio": stats["median_ratio"],
        "eps": stats["epsilon"],
    }


class LayoutGraphs:
    def __init__(self, hk, hv, batch, phase):
        from dataclasses import fields

        from tests.modules._replay_case import clone
        from tests.modules.test_gdn_replay_range import LayoutPair

        self.calls = 64
        self.pair = pair = LayoutPair(
            device="cuda",
            provider="cute",
            hk=hk,
            hv=hv,
            batch=batch,
            q=8,
            mode="immediate",
            checks="trusted",
        )
        # Fix the CuTe route for both layouts, as in test_layout_step.
        pair.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device="cuda",
            provider="cute",
            mode="immediate",
            checks="trusted",
            cute_config=Config(immediate=True),
            commit_mode="eager",
            window=4,
            verify_length=8,
        )
        pair.both("initialize")
        pair.both("handoff")
        pair.update("accepted", torch.full_like(pair.accepted, 8))
        # Save only logical views: copying back preserves the pitched allocations.
        self.saved = []
        for side in range(2):
            copies = []
            for obj in (pair.states[side], pair.pending[side]):
                initial = clone(obj)
                copies.extend((getattr(obj, f.name), getattr(initial, f.name)) for f in fields(obj))
            self.saved.append(copies)

        def call(side):
            if phase == "step":
                pair.call("step", side)
            else:
                pair.call("verify", side)
                pair.op.commit(
                    *pair.weights[side],
                    slot_indices=pair.controls["slots"][side],
                    accepted_lengths=pair.controls["accepted"][side],
                    state=pair.states[side],
                    pending=pair.pending[side],
                    workspace=pair.workspaces["commit"][side],
                )

        # Check both layouts eagerly before timing, including padding guards.
        for side in range(2):
            call(side)
        pair.compare()
        self.graphs = []
        for side in (0, 1, 0):
            self.reset(side)
            self.graphs.append(graph([lambda side=side: call(side)] * self.calls))
        for index in range(3):
            self.time(index)

    def reset(self, side):
        for target, initial in self.saved[side]:
            target.copy_(initial)

    def time(self, index):
        self.reset(1 if index == 1 else 0)
        return trial(self.graphs[index], self.calls)


@torch.inference_mode()
def main():
    for hk, hv in ((16, 32), (16, 48)):
        for batch in (1, 4):
            for phase in ("step", "k2"):
                graphs = LayoutGraphs(hk, hv, batch, phase)
                stats = measure_pairs(
                    partial(graphs.time, 0), partial(graphs.time, 1), partial(graphs.time, 2)
                )
                print(
                    json.dumps({"shape": [hk, hv], "b": batch, "phase": phase, **stats}), flush=True
                )
                del graphs


if __name__ == "__main__":
    main()
