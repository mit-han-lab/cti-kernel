# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Eager engine-pattern host timing; also runs against v0.1.13rc6."""

import argparse
import json
import statistics
import time
from dataclasses import fields
from pathlib import Path

import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay import _api


def allocate(specs):
    return {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device="cuda")
        for name, spec in specs.items()
    }


class Workload:
    def __init__(self, batch, q, layers, slots, extend, bmax, *, geometry=(16, 32), snapshots=0):
        self.batch, self.q, self.layers = batch, q, layers
        self.op = op = GatedDeltaNetReplay(
            num_k_heads=geometry[0],
            num_v_heads=geometry[1],
            window=4,
            verify_length=q,
            device="cuda",
            checks="trusted",
        )
        if any(op.provider[phase] != "cute" for phase in ("verify", "commit")):
            raise RuntimeError("host benchmark requires CuTe verify and commit")
        self.states = [
            op.create_state(
                buffers=allocate(op.state_spec(num_slots=slots, num_snapshot_rows=snapshots))
            )
            for _ in range(layers)
        ]
        self.controls = op.bind_graph_buffers(
            buffers=allocate(op.graph_buffer_spec(batch_size=batch, depth=2, num_layers=layers))
        )
        g = op.geometry
        self.raw = torch.zeros(
            layers, 2, batch, q, g.conv_dim + 2 * g.v_heads, dtype=torch.bfloat16, device="cuda"
        )
        self.arrivals = torch.zeros(
            op.graph_data_spec(batch_size=batch)["phase_arrivals"].shape,
            dtype=torch.int32,
            device="cuda",
        )
        self.weights = [
            (
                torch.randn(g.conv_dim, 4, dtype=torch.bfloat16, device="cuda") * 0.1,
                torch.full((g.v_heads,), -1.0, device="cuda"),
                torch.zeros(g.v_heads, device="cuda"),
                torch.ones(128, device="cuda"),
            )
            for _ in range(layers)
        ]
        self.joint = (
            torch.randn(extend + batch * q, g.proj_width, dtype=torch.bfloat16, device="cuda") * 0.1
        )
        self.extend = extend
        self.out = torch.zeros(bmax, q, g.value_width, dtype=torch.bfloat16, device="cuda")
        self.slots = torch.arange(batch, dtype=torch.int32, device="cuda")
        self.positions = torch.arange(q, dtype=torch.int64, device="cuda").repeat(batch, 1)
        self.accepted = torch.full((batch,), q // 2, dtype=torch.int32, device="cuda")
        canonical = torch.zeros(batch, g.v_heads, 128, 128, device="cuda")
        history = torch.zeros(batch, 3, g.conv_dim, dtype=torch.bfloat16, device="cuda")
        next_positions = torch.zeros(batch, dtype=torch.int64, device="cuda")
        # begin installs records for all layers of this invocation.
        status = self.controls.status(0, 0)
        status.sequence.fill_(1)
        op.begin(status=status, sequence=status.sequence, layer_weights=self.weights)
        for i in range(layers):
            status = self.controls.status(0, i)
            op.handoff(
                canonical_state=canonical,
                history=history,
                next_positions=next_positions,
                slot_indices=self.slots,
                state=self.states[i],
                status=status,
            )

    def verify(self, i, slot):
        b = self.batch
        self.op.verify(
            self.joint[self.extend :].view(b, self.q, -1),
            *self.weights[i],
            slot_indices=self.slots,
            positions=self.positions,
            state=self.states[i],
            pending=self.controls.pending(self.raw[i, slot, :b], slot=slot, layer=i),
            out=self.out[:b],
            status=None,
        )

    def commit(self, i, slot, **snapshots):
        b = self.batch
        self.op.commit(
            *self.weights[i],
            slot_indices=self.slots,
            accepted_lengths=self.accepted,
            state=self.states[i],
            pending=self.controls.pending(self.raw[i, slot, :b], slot=slot, layer=i),
            status=None,
            workspace={"arrivals": self.arrivals[slot, :b]},
            **snapshots,
        )

    def round(self, slot, *, timed=False):
        samples = ([], [])
        for phase, method in enumerate((self.verify, self.commit)):
            for i in range(self.layers):
                start = time.perf_counter() if timed else 0
                method(i, slot)
                if timed:
                    samples[phase].append((time.perf_counter() - start) * 1e6)
        return samples

    def mutable(self):
        return [
            *(getattr(s, f.name) for s in self.states for f in fields(s)),
            self.raw,
            self.out,
            self.arrivals,
            *self.controls.buffers.values(),
        ]

    def check(self):
        present = hasattr(_api, "HOST_FAST_PATH")
        original = getattr(_api, "HOST_FAST_PATH", None)
        tensors = self.mutable()
        initial = [t.clone() for t in tensors]
        try:
            if present:
                _api.HOST_FAST_PATH = True
            self.round(0)
            expected = [t.clone() for t in tensors]
            for t, value in zip(tensors, initial, strict=True):
                t.copy_(value)
            if present:
                _api.HOST_FAST_PATH = False
            self.round(0)
            torch.cuda.synchronize()
            for t, value in zip(tensors, expected, strict=True):
                assert torch.equal(t, value)
        finally:
            if present:
                _api.HOST_FAST_PATH = original
        print("bitwise ok")


def stats(samples):
    ordered = sorted(samples)

    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (
            position - lo
        )

    return {
        "mean": statistics.mean(samples),
        "p50": percentile(0.5),
        "p10": percentile(0.1),
        "p90": percentile(0.9),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="1,2,3,4")
    parser.add_argument("--q", default="16,8")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--extend", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warm", type=int, default=5)
    parser.add_argument("--accepted", type=int)
    parser.add_argument("--slow", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    batches, qs = [int(b) for b in args.batch.split(",")], [int(q) for q in args.q.split(",")]
    if min(*batches, *qs, args.layers, args.rounds) < 1 or args.warm < 2 or args.extend < 0:
        parser.error("positive batches/Q/layers/rounds, warm >= 2 and extend >= 0 required")
    if max(batches) > args.slots or any(not 0 <= (args.accepted or 0) <= q for q in qs):
        parser.error("slots must cover batch; accepted must be in [0,Q]")
    present = hasattr(_api, "HOST_FAST_PATH")
    if present:
        _api.HOST_FAST_PATH = not args.slow
    rows = []
    for q in qs:
        for batch in batches:
            work = Workload(batch, q, args.layers, args.slots, args.extend, max(batches))
            accepted = q // 2 if args.accepted is None else args.accepted
            work.accepted.fill_(accepted)
            for r in range(args.warm):
                work.round(r % 2)
                work.positions.add_(accepted)
                torch.cuda.synchronize()
            before = len(work.op._cute_launchers)
            samples = ([], [])
            for r in range(args.rounds):
                result = work.round(r % 2, timed=True)
                torch.cuda.synchronize()
                work.positions.add_(accepted)
                for dest, values in zip(samples, result, strict=True):
                    dest.extend(values)
            after = len(work.op._cute_launchers)
            assert before == after, "specializations compiled during timing"
            verify, commit = (stats(s) for s in samples)
            row = {
                "batch": batch,
                "q": q,
                "layers": args.layers,
                "rounds": args.rounds,
                "warm": args.warm,
                "slow": args.slow,
                "host_fast_path_exists": present,
                "verify_us": verify,
                "commit_us": commit,
                "round_ms": args.layers * (verify["mean"] + commit["mean"]) / 1000,
                "launchers_before": before,
                "launchers_after": after,
            }
            rows.append(row)
            print(json.dumps(row))
            if args.check:
                work.check()
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
