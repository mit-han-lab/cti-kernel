# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired static-slot versus device-selected K2 graphs, in microseconds per round.

Each launch starts from identical restored state, then evolves for --rounds
verify/commit rounds. Restoration, selector writes and lifecycle admission are
outside timing. A captures one graph per invocation slot; only slot 0 is timed.
B captures one graph over stacked pending/status/arrivals, selecting slot 0.
No snapshots are requested. Inputs and acceptance match gdn_verify_compare.Ours.
"""

import argparse
import json
from dataclasses import fields
from itertools import product
from pathlib import Path
from statistics import median

import torch

from benchmark.kernel_tuning.gdn_decode_compare import _positive_int, _shape
from benchmark.kernel_tuning.gdn_replay_common import allocate
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from benchmark.kernel_tuning.gdn_verify_compare import GraphRound, Restorable, percentile
from ctikernels.gdn import GatedDeltaNetReplay


class RoundBuffers(Restorable):
    """Engine-owned allocations shared by all of one form's graph bindings."""

    def __init__(self, weights, drafts, initial, history, *, hk, hv, depth, rounds):
        batch, q, _ = drafts.shape
        device = drafts.device
        self.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device=device,
            window=4,
            verify_length=q,
            checks="trusted",
            reference_dtype=torch.float32,
        )
        self.rounds = rounds
        self.weights = weights.args()
        self.drafts = drafts
        self.state = self.op.create_state(
            buffers=allocate(self.op.state_spec(num_slots=batch, num_snapshot_rows=0), device)
        )
        self.controls = self.op.bind_graph_buffers(
            buffers=allocate(self.op.graph_buffer_spec(batch_size=batch, depth=depth), device)
        )
        specs = self.op.graph_data_spec(batch_size=batch, depth=depth)
        self.data = allocate(
            {
                name: spec
                for name, spec in specs.items()
                if name in ("pending.raw", "phase_arrivals")
            },
            device,
        )
        self.raw = self.data["pending.raw"][0]
        self.selector = torch.zeros(1, dtype=torch.int32, device=device)
        self.slots = torch.arange(batch, dtype=torch.int32, device=device)
        self.accepted = torch.full((batch,), q // 2, dtype=torch.int32, device=device)
        # Precomputed controls keep position-update kernels out of the round.
        self.positions = [
            (torch.arange(q, dtype=torch.int64) + r * (q // 2))
            .expand(batch, q)
            .contiguous()
            .to(device)
            for r in range(rounds)
        ]
        self.outputs = [
            torch.zeros(batch, q, hv * 128, dtype=torch.bfloat16, device=device)
            for _ in range(rounds)
        ]
        for d in range(depth):
            status = self.controls.status(d)
            status.sequence.fill_(d + 1)
            self.op.begin(status=status, layer_weights=(self.weights,))
        self.op.handoff(
            canonical_state=initial,
            history=history,
            next_positions=torch.zeros(batch, dtype=torch.int64, device=device),
            slot_indices=self.slots,
            state=self.state,
            status=self.controls.status(0),
        )
        self.freeze(self.comparable().values())

    def comparable(self):
        """All mutable payloads, including unused invocation slots and scratch."""
        return (
            {f"state.{f.name}": getattr(self.state, f.name) for f in fields(self.state)}
            | {f"control.{name}": t for name, t in self.controls.buffers.items()}
            | self.data
            | {f"out.{r}": t for r, t in enumerate(self.outputs)}
        )

    def prepare(self):
        super().prepare()
        self.selector.zero_()

    def check_replay(self):
        # Equality alone could accept two no-ops or matching nonfinite outputs.
        if self.controls.buffers["poison"].any().item():
            raise RuntimeError("Replay poisoned its status")
        expected = self.rounds * (self.op.verify_length // 2)
        if not (self.state.metadata[:, 1] == expected).all().item():
            raise RuntimeError("Replay did not advance every live row by N*m")
        for tensor in (self.state.checkpoint, *self.outputs):
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError("Replay produced nonfinite state or outputs")


class BoundRound:
    def __init__(self, buffers, slot=None):
        self.buffers = buffers
        self.rounds = buffers.rounds
        controls = buffers.controls
        self.extra = {"status_slot": buffers.selector} if slot is None else {}
        self.pending = (
            controls.slotted_pending(buffers.raw)
            if slot is None
            else controls.pending(buffers.raw[slot], slot=slot)
        )
        self.status = controls.slotted_status() if slot is None else controls.status(slot)
        self.workspace = None
        if "phase_arrivals" in buffers.data:
            arrivals = buffers.data["phase_arrivals"]
            self.workspace = {"arrivals": arrivals if slot is None else arrivals[slot]}

    def prepare(self):
        self.buffers.prepare()

    def finish(self):
        pass

    def call(self):
        p = self.buffers
        for positions, output in zip(p.positions, p.outputs, strict=True):
            p.op.verify(
                p.drafts,
                *p.weights,
                slot_indices=p.slots,
                positions=positions,
                state=p.state,
                pending=self.pending,
                out=output,
                status=self.status,
                **self.extra,
            )
            p.op.commit(
                *p.weights,
                slot_indices=p.slots,
                accepted_lengths=p.accepted,
                state=p.state,
                pending=self.pending,
                status=self.status,
                workspace=self.workspace,
                **self.extra,
            )


def paired_samples(a, b, pair_count):
    """ABBA blocks interleaved with the same number of A/A noise pairs."""
    pairs, self_pairs = [], []
    for _ in range(0, pair_count, 2):
        a0, b0, b1, a1 = a.sample(), b.sample(), b.sample(), a.sample()
        block = [(a0, b0), (a1, b1)][: pair_count - len(pairs)]
        pairs.extend(block)
        self_pairs.extend((a.sample(), a.sample()) for _ in block)
    ratio = median(x / y for x, y in pairs)
    epsilon = percentile([abs(x / y - 1) for x, y in self_pairs], 0.95)
    return {
        "a_us": median(x for x, _ in pairs),
        "b_us": median(y for _, y in pairs),
        "median_ratio_a_over_b": ratio,
        "epsilon": epsilon,
        "verdict": "no-regression" if ratio >= 1 - epsilon else "regression",
        "samples": {"pairs_us": pairs, "self_pairs_us": self_pairs},
    }


def measure(hk, hv, batch, q, *, pairs, rounds, depth):
    weights, x, initial, history = case_inputs(hk, hv, batch, batch, q + 1, "ordinary", 23, "cuda")
    drafts = x[-q:].transpose(0, 1).contiguous().view(batch, q, -1)
    a, b = [
        RoundBuffers(weights, drafts, initial, history, hk=hk, hv=hv, depth=depth, rounds=rounds)
        for _ in range(2)
    ]
    # Clone A's exact baseline, including handoff metadata and status records.
    for (_, source), (target, saved) in zip(a.saved, b.saved, strict=True):
        target.copy_(source)
        saved.copy_(source)
    static_graphs = [
        GraphRound(BoundRound(a, d), assert_no_extra_copies=False) for d in range(depth)
    ]
    slotted_graph = GraphRound(BoundRound(b), assert_no_extra_copies=False)
    static_graph = static_graphs[0]
    # One replay each from cloned identical state; each replay contains N rounds.
    static_graph.sample()
    slotted_graph.sample()
    a.check_replay()
    b.check_replay()
    bitwise = all(
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
        for left, right in zip(a.comparable().values(), b.comparable().values(), strict=True)
    )
    return {
        "device": torch.cuda.get_device_name(),
        "shape": [hk, hv],
        "b": batch,
        "q": q,
        "window": 4,
        "accepted": q // 2,
        "depth": depth,
        "rounds": rounds,
        "pairs": pairs,
        "requested_checks": "trusted",
        "effective_checks": a.op.checks,
        "provider": a.op.provider["verify"],
        "commit_mode": a.op.commit_mode_for(batch),
        "bitwise": bitwise,
        **paired_samples(static_graph, slotted_graph, pairs),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", type=_shape, default=[(16, 32), (8, 16)])
    parser.add_argument("--batches", nargs="+", type=_positive_int, default=[1, 2, 4])
    parser.add_argument("--qs", nargs="+", type=_positive_int, default=[8, 16])
    parser.add_argument("--pairs", type=_positive_int, default=20)
    parser.add_argument("--rounds", type=_positive_int, default=20)
    parser.add_argument("--depth", type=_positive_int, default=2)
    parser.add_argument("--out", type=Path, required=True, help="JSON lines output file")
    args = parser.parse_args(argv)
    if min(args.qs) < 2:
        parser.error("--qs must be at least 2 for nonzero Q//2 acceptance")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this benchmark does not run on CPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(
        "| Device | Shape | B | Q | A us | B us | A/B | Epsilon | Verdict | bitwise |", flush=True
    )
    print("|---|---|---|---|---|---|---|---|---|---|", flush=True)
    correct = True
    with args.out.open("w") as output, torch.inference_mode():
        for (hk, hv), batch, q in product(args.shapes, args.batches, args.qs):
            row = measure(hk, hv, batch, q, pairs=args.pairs, rounds=args.rounds, depth=args.depth)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            correct &= row["bitwise"]
            print(
                f"| {row['device']} | {hk},{hv} | {batch} | {q} | {row['a_us']:.3f} | "
                f"{row['b_us']:.3f} | {row['median_ratio_a_over_b']:.4f} | "
                f"{row['epsilon']:.4f} | {row['verdict']} | {row['bitwise']} |",
                flush=True,
            )
    if not correct:
        raise SystemExit(
            "Static and slotted replays are not bitwise equal; timings are unqualified"
        )


if __name__ == "__main__":
    main()
