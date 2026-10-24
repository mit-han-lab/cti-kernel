# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Kineto kernel durations and graph-replay step time at every replay window phase.

State restoration, sequence staging and settlement are outside the event interval.
This isolates phase cost on a warm working set; use gdn_replay_bench for cold timing.
Each trial replays exactly one captured decode invocation, including its controls.
"""

import argparse
import itertools
import statistics
from dataclasses import asdict, fields, replace

import torch

from benchmark.kernel_tuning.gdn_replay_common import Product, Replay, emit, manifest
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from ctikernels.gdn.replay.cute import select


def kernel_events(profile):
    return [
        e
        for e in profile.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and e.name != "[memory]"
        and "memcpy" not in e.name.lower()
        and "memset" not in e.name.lower()
    ]


SECTION_NAMES = ("controls", "state", "token_prologue", "replay", "epilogue", "output_norm")


def section_medians(samples):
    """Nanosecond CTA timestamps -> microseconds; aggregate differences, not clocks."""
    if not samples:
        return None
    deltas = [[(row[i + 1] - row[i]) / 1000 for i in range(6)] for row in samples]
    if any(value < 0 for row in deltas for value in row):
        raise ValueError("nonmonotonic CTA section timestamps")
    return {
        name: statistics.median(row[i] for row in deltas) for i, name in enumerate(SECTION_NAMES)
    }


def configs(args):
    if args.provider != "cute":
        return [None]
    base = select(
        torch.cuda.get_device_name(),
        args.batch,
        mode=args.mode,
        geometry=tuple(map(int, args.shape.split(","))),
        window=args.window,
    )
    if args.value_tile is not None:
        base = replace(base, value_tile=args.value_tile)
    if args.speculative_ring is not None:
        base = replace(base, speculative_ring=args.speculative_ring)

    def policy(warps, lanes, prefetch, snapshot):
        # Snapshot and the legacy eight-lane experiments retain full heads.
        tile = base.value_tile
        if args.value_tile is None and (snapshot or lanes != 4):
            tile = 128
        return replace(
            base,
            value_tile=tile,
            warps=warps,
            lanes=lanes,
            prefetch=prefetch,
            sections=args.sections,
            snapshot_controls=snapshot,
            trusted=False,
        )

    if args.sweep:
        return [
            policy(w, lanes, p, c)
            for w, lanes, p, c in itertools.product(
                args.sweep_warps,
                args.sweep_lanes,
                (False, True),
                [mode == "snapshot" for mode in args.sweep_controls],
            )
        ]
    return [policy(args.warps, args.lanes, args.prefetch, args.snapshot_controls)]


class PhaseGraph:
    def __init__(self, args, phase, config):
        hk, hv = map(int, args.shape.split(","))
        b, window = args.batch, args.window
        weights, xs, initial, history = case_inputs(hk, hv, b, b, window, "ordinary", 11, "cuda")
        weights = weights.args()
        self.sequence = 1
        self.section_samples = []
        self.epilogue_maxima = []
        self.replay = args.provider != "product"
        if self.replay:
            runner = Replay(
                args.provider,
                hk,
                hv,
                b,
                window,
                16,
                weights,
                initial,
                history,
                torch.zeros(b, dtype=torch.int64, device="cuda"),
                cute_config=config,
                checks="strict",
            )
        else:
            runner = Product(hk, hv, weights, initial, history)
        self.runner = runner
        for t in range(phase):
            runner.step(xs[t])
        if self.replay:
            runner.check()
            runner.op.end(status=runner.status)
            self.settle()
            self.saved = {
                f.name: getattr(runner.state, f.name).clone() for f in fields(runner.state)
            }
            self.pos = runner.pos.clone()
            self.static_pos = runner.pos.clone()
        else:
            self.saved = {"state": runner.state.clone(), "history": runner.history.clone()}
        x = xs[phase]

        def call():
            if not self.replay:
                runner.step(x)
                return
            self.static_pos.copy_(runner.pos)
            runner.op.begin(status=runner.status, layer_weights=(weights,))
            runner.op.step(
                x,
                *weights,
                slot_indices=runner.slots,
                positions=self.static_pos,
                state=runner.state,
                out=runner.out,
                status=runner.status,
                workspace=runner.workspace,
            )
            runner.op.end(status=runner.status)
            runner.pos.add_(1)

        self.restore()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.settle()
        self.restore()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            call()
        self.start, self.end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def settle(self):
        if self.replay:
            op, status = self.runner.op, self.runner.status
            result = op.read_status_copy(*status.copy_to_host(), expected_sequence=self.sequence)
            if not result.ok:
                raise RuntimeError("profile invocation failed validation")
            op.settle(status=status, result=result)
            self.sequence += 1
            status.sequence.fill_(self.sequence)

    def restore(self):
        if self.replay:
            for name, value in self.saved.items():
                getattr(self.runner.state, name).copy_(value)
            self.runner.pos.copy_(self.pos)
        else:
            for name, value in self.saved.items():
                getattr(self.runner, name).copy_(value)

    def profile(self):
        self.restore()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            self.graph.replay()
            torch.cuda.synchronize()
        events = kernel_events(prof)
        result = [{"name": e.name, "duration_us": e.time_range.elapsed_us()} for e in events]
        self.settle()
        return result

    def trial(self):
        self.restore()
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        us = self.start.elapsed_time(self.end) * 1000
        if self.replay and self.runner.op.cute_config is not None:
            if self.runner.op.cute_config.sections:
                # D2H is diagnostic only, after the measured event and completion.
                stamps = self.runner.workspace["sections"][:-1].cpu().tolist()
                self.section_samples.extend(stamps)
                self.epilogue_maxima.append(max((r[5] - r[4]) / 1000 for r in stamps))
        self.settle()
        return us


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("immediate", "delayed", "auto"), default="auto")
    parser.add_argument("--provider", choices=("cute", "triton", "product"), default="cute")
    parser.add_argument("--shape", choices=("16,32", "16,48", "8,16", "8,24"), default="16,32")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--lanes", type=int, choices=(4, 8), default=4)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--value-tile", type=int, choices=(32, 64, 128))
    parser.add_argument("--speculative-ring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--sections", action="store_true", help="instrument CTA boundaries (adds overhead)"
    )
    parser.add_argument(
        "--snapshot-controls",
        action="store_true",
        help="ordered validation snapshot + counter-free compute (two launches)",
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-warps", nargs="+", type=int, choices=(4, 8), default=[4, 8])
    parser.add_argument("--sweep-lanes", nargs="+", type=int, choices=(4, 8), default=[4, 8])
    parser.add_argument(
        "--sweep-controls", nargs="+", choices=("grid", "snapshot"), default=["grid"]
    )
    args = parser.parse_args(argv)
    if args.batch < 1 or args.trials < 1:
        parser.error("batch and trials must be positive")
    if args.provider != "cute" and (
        args.mode != "auto"
        or args.value_tile is not None
        or args.speculative_ring is not None
        or args.sweep
        or args.sections
        or args.snapshot_controls
        or args.prefetch
        or args.warps != 8
        or args.lanes != 4
    ):
        parser.error("CuTe tunables require --provider cute")
    if not torch.cuda.is_available():
        emit({"type": "summary", "qualified": False, "reason": "CUDA not available"})
        return 2
    choices = configs(args)
    emit({"type": "manifest", "manifest": manifest(), "regime": "warm", "qualified": False})
    for config in choices:
        means, measured = [], None
        for phase in range(args.window):
            graph = PhaseGraph(args, phase, config)
            measured = graph.runner.op.cute_config if args.provider == "cute" else None
            kernels = graph.profile()
            times = [graph.trial() for _ in range(args.trials)]
            median = statistics.median(times)
            means.append(median)
            emit(
                {
                    "type": "phase",
                    "provider": args.provider,
                    "shape": args.shape,
                    "B": args.batch,
                    "L": args.window,
                    "n": phase,
                    "config": asdict(measured) if measured else None,
                    "NV": 128 // measured.value_tile if measured else None,
                    "instrumented": args.sections,
                    "sections_median_us": section_medians(graph.section_samples),
                    "epilogue_cta_max_median_us": (
                        statistics.median(graph.epilogue_maxima) if graph.epilogue_maxima else None
                    ),
                    "kernels_per_call": len(kernels),
                    "kernels": kernels,
                    "graph_replay_us": median,
                    "trials_us": times,
                    "qualified": False,
                }
            )
            del graph
        emit(
            {
                "type": "config",
                "provider": args.provider,
                "shape": args.shape,
                "B": args.batch,
                "L": args.window,
                "config": asdict(measured) if measured else None,
                "phase_us": means,
                "average_us": statistics.mean(means),
                "qualified": False,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
