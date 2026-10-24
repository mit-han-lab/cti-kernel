# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU host cost of 36 layers' verify/commit calls; providers do no work."""

import argparse
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay import _api


def pairwise(tensors, *, overlap, exclude=None):
    for i, a in enumerate(tensors):
        for j in range(i + 1, len(tensors)):
            if exclude is not None and exclude(i, j):
                continue
            if overlap(a, tensors[j]):
                yield i, j


def allocate(specs):
    return {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype)
        for name, spec in specs.items()
    }


def calls():
    op = GatedDeltaNetReplay(
        num_k_heads=16, num_v_heads=32, device="cpu", verify_length=16, checks="trusted"
    )
    op._backends["torch_reference"] = SimpleNamespace(run=lambda *a, **k: None)
    g = op.geometry
    graph = op.bind_graph_buffers(
        buffers=allocate(op.graph_buffer_spec(batch_size=1, num_layers=36))
    )
    work = graph.buffers
    invocations = []
    for layer in range(36):
        state = op.create_state(buffers=allocate(op.state_spec(num_slots=1, num_snapshot_rows=1)))
        pending = graph.pending(
            torch.empty(1, 16, g.conv_dim + 2 * g.v_heads, dtype=torch.bfloat16),
            slot=0,
            layer=layer,
        )
        weights = (
            torch.empty(g.conv_dim, 4, dtype=torch.bfloat16),
            torch.empty(g.v_heads),
            torch.empty(g.v_heads),
            torch.empty(128),
        )
        common = {
            "state": state,
            "status": graph.status(0, layer),
            "pending": pending,
            "weights": weights,
            "slot_indices": work["slot_indices"][0],
        }
        invocations.append(
            (
                "verify",
                {
                    **common,
                    "projected": torch.empty(1, 16, g.proj_width, dtype=torch.bfloat16),
                    "positions": work["verify_positions"][0],
                    "out": torch.empty(1, 16, g.value_width, dtype=torch.bfloat16),
                },
            )
        )
        invocations.append(("commit", {**common, "accepted_lengths": work["accepted_lengths"][0]}))
    return op, invocations


def measure(op, invocations, repeats):
    for mode, kw in invocations:
        op._run(mode, **kw)
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for mode, kw in invocations:
            op._run(mode, **kw)
        samples.append(perf_counter_ns() - start)
    elapsed = median(samples)
    return elapsed / len(invocations) / 1000, elapsed / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="also time per-call pairwise checks")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    op, invocations = calls()
    rows = []
    if args.legacy:
        with (
            patch.object(op, "_validation_key", lambda *a, **k: None),
            patch.object(_api, "overlapping_pairs", pairwise),
        ):
            rows.append(("legacy", *measure(op, invocations, args.repeats)))
    rows.append(("trusted memo", *measure(op, invocations, args.repeats)))
    print("| Host checks | Median us/call | Total ms/72 calls |")
    print("|---|---:|---:|")
    for name, per_call, total in rows:
        print(f"| {name} | {per_call:.2f} | {total:.3f} |")


if __name__ == "__main__":
    main()
