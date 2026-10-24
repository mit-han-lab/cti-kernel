# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Per-round K2 kernel time (verify + commit) by value tile, with vector publication.

Eager calls under torch.profiler; device time per kernel class (control, compute,
publish) averaged over rounds. One JSON line per (tile, Q, B, checks, PDL, fused).
"""

import argparse
import json
from itertools import product

import torch
from torch.profiler import ProfilerActivity, profile

import ctikernels.gdn.replay.cute._phase_launch as launch
from benchmark.kernel_tuning.gdn_decode_compare import _shape
from benchmark.kernel_tuning.gdn_replay_common import Replay
from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs
from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import select


def _device_us(event):
    for name in ("device_time_total", "cuda_time_total"):
        if hasattr(event, name):
            return getattr(event, name)
    return 0.0


def measure(hk, hv, q, b, rounds, checks, commit_mode="eager", window=8):
    weights, x, initial, history = case_inputs(hk, hv, b, b, q + 1, "ordinary", 23, "cuda")
    runner = Replay(
        "cute",
        hk,
        hv,
        b,
        window,
        q,
        weights.args(),
        initial,
        history,
        torch.zeros(b, dtype=torch.int64, device="cuda"),
    )
    drafts = x[-q:].transpose(0, 1)
    accepted = torch.full((b,), max(1, q // 2), dtype=torch.int32, device="cuda")
    runner.op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        window=window,
        verify_length=q,
        provider="cute",
        checks=checks,
        commit_mode=commit_mode,
    )
    op, status = runner.op, runner.status
    if checks == "strict":
        # The fixture leaves sequence 1 begun; close it before the timed rounds.
        op.end(status=status)
        op.settle(
            status=status, result=op.read_status_copy(*status.copy_to_host(), expected_sequence=1)
        )
    sequence = [1]

    def round_():
        if checks == "strict":
            sequence[0] += 1
            status.sequence.fill_(sequence[0])
            op.begin(status=status, layer_weights=(runner.weights,))
        runner.verify(drafts)
        op.commit(
            *runner.weights,
            slot_indices=runner.slots,
            accepted_lengths=accepted,
            state=runner.state,
            pending=runner.pending,
            status=status if checks == "strict" else None,
            workspace=runner.phase_workspace if checks == "trusted" else None,
        )
        if checks == "strict":
            op.end(status=status)
            result = op.read_status_copy(*status.copy_to_host(), expected_sequence=sequence[0])
            if not result.ok:
                raise RuntimeError(f"round failed: {result}")
            op.settle(status=status, result=result)
        runner.pos.add_(accepted)

    for _ in range(3):
        round_()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(rounds):
            round_()
        torch.cuda.synchronize()
    classes = {}
    for event in prof.key_averages():
        us = _device_us(event)
        if us <= 0:
            continue
        name = event.key
        kind = next(
            (k for k in ("publish", "compute", "control") if k in name),
            "other:" + name[:60],
        )
        classes[kind] = classes.get(kind, 0.0) + us / rounds
    gdn = sum(v for k, v in classes.items() if not k.startswith("other"))
    return {
        "per_round_us": classes,
        "gdn_kernels_us": gdn,
        "commit_mode": op.commit_mode_for(b),
        "window": window,
        "rounds": rounds,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-mode", nargs="+", choices=("eager", "delayed"), default=["eager"])
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--shape", type=_shape, default="16,32", metavar="HK,HV")
    parser.add_argument("--q", nargs="+", type=int, default=[4, 5, 8, 9, 12, 16])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--rounds", type=int, default=48)
    parser.add_argument(
        "--tiles",
        nargs="+",
        choices=("128", "64", "32", "auto", "legacy"),
        default=["128", "64", "32", "auto"],
        help="auto: the shipped phase_launch policy; legacy: tile by wave, 8 warps, 1 CTA/SM",
    )
    parser.add_argument(
        "--checks", nargs="+", choices=("trusted", "strict"), default=["trusted", "strict"]
    )
    parser.add_argument("--pdl", nargs="+", choices=("on", "off"), default=["on", "off"])
    parser.add_argument("--fused", nargs="+", choices=("on", "off"), default=["on"])
    parser.add_argument("--warps", nargs="+", choices=("auto", "16", "8", "4"), default=["auto"])
    parser.add_argument(
        "--min-blocks", nargs="+", choices=("auto", "1", "2", "3"), default=["auto"]
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    hk, hv = args.shape
    print(json.dumps({"device": torch.cuda.get_device_name()}), flush=True)
    launch.VECTOR_PUBLISH = True
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    config = select(torch.cuda.get_device_name())

    def effective(tile, warps, min_blocks, q, b):
        """The (tile, warps, min CTAs per SM) that run() launches for these settings."""
        if tile == "auto":
            chosen = launch.phase_launch(b, hv, sms, q, config)
        elif tile == "legacy":
            chosen = (launch.phase_tile(b, hv, sms), 8, 1)
        else:
            chosen = (int(tile), config.warps, config.min_blocks)
        return (
            chosen[0],
            chosen[1] if warps == "auto" else int(warps),
            chosen[2] if min_blocks == "auto" else int(min_blocks),
        )

    for tile, checks, pdl, fused, warps, min_blocks, commit_mode in product(
        args.tiles, args.checks, args.pdl, args.fused, args.warps, args.min_blocks, args.commit_mode
    ):
        if checks == "strict" and commit_mode == "delayed":
            continue
        launch.FUSED_PUBLISH = fused == "on"
        launch.PDL = pdl == "on"
        for q in args.q:
            for b in args.batches:
                # Pin all three launch settings explicitly, so the row records
                # exactly what ran (no override can silently replace the policy).
                selected = effective(tile, warps, min_blocks, q, b)
                launch.PHASE_TILE, launch.PHASE_WARPS, launch.PHASE_MIN_BLOCKS = selected
                row = measure(hk, hv, q, b, args.rounds, checks, commit_mode, args.window)
                row.update(
                    requested_commit_mode=commit_mode,
                    publication="vector",
                    checks=checks,
                    pdl=pdl,
                    fused=fused,
                    tile=tile,
                    warps=warps,
                    min_blocks=min_blocks,
                    selected=selected,
                    q=q,
                    b=b,
                    shape=",".join(map(str, args.shape)),
                )
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
