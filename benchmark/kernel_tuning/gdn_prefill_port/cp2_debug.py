#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Eager 16x48 fused watchdog repro; --check adds CP1 plus FP32 torch glue.

Each --lengths value is a packed batch, e.g. --lengths 64 or --lengths '[16,100]'.
Repeat --lengths to run separate batches. Default: [64], [128], [16], [100].
The first CUDA failure stops the script: a trapped context cannot be reused.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from ctikernels.gdn.chunk_ws.prefill import PortPrefill

root = Path(__file__).resolve().parents[3]
if not (root / "tests/gpu/_gdn_reference.py").is_file():
    root = Path.cwd()
sys.path.insert(0, str(root))
from tests.gpu.test_gdn_port_prefill import cp1_glue, make_case  # noqa: E402


def parse_lengths(value):
    try:
        lengths = json.loads(value) if value.startswith("[") else [int(v) for v in value.split(",")]
        if not isinstance(lengths, list) or not lengths:
            raise ValueError
        if any(type(length) is not int or length <= 0 for length in lengths):
            raise ValueError
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "lengths must be a nonempty list of positive integers"
        ) from error
    return lengths


def compare(name, got, expected, floor):
    delta = got.double() - expected.double()
    rms = delta.square().mean().sqrt().item()
    reference_rms = expected.double().square().mean().sqrt().item()
    limit = 0.02 * reference_rms + floor
    print(
        f"CHECK {name} rms={rms:.8g} max={delta.abs().max().item():.8g} limit={limit:.8g}",
        flush=True,
    )
    if not torch.isfinite(got).all().item() or rms > limit:
        raise AssertionError(f"{name} exceeds CP1 RMS tolerance")


def run_case(op, lengths, check):
    x, weights, meta, base = make_case(16, 48, lengths)
    # Exercise all requested sequences as live state owners.
    meta[1].copy_(torch.arange(1, len(lengths) + 1, device=x.device, dtype=torch.int32))
    pools = tuple(t.clone() for t in base)
    print(f"LAUNCH lengths={lengths} watchdog=True", flush=True)
    out = op(x, *weights.args(), *meta, *pools)
    # Device printf is emitted to stdout by CUDA synchronization. Keep this
    # inside main's try, including for asynchronous launch failures/traps.
    torch.cuda.synchronize()
    sys.stdout.flush()
    print(f"COMPLETE lengths={lengths}", flush=True)
    if check:
        cp1_pools = tuple(t.clone() for t in base)
        expected = cp1_glue(x, weights, 16, meta, cp1_pools)
        torch.cuda.synchronize()
        compare("y", out, expected, 1e-4)
        compare("state", pools[0][1:-1], cp1_pools[0][1:-1], 1e-5)
        torch.testing.assert_close(pools[1], cp1_pools[1], rtol=0, atol=0)
        torch.testing.assert_close(pools[0][[0, -1]], base[0][[0, -1]], rtol=0, atol=0)
        print("CHECK PASS (y, state, exact history, untouched slots)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_lengths, action="append")
    parser.add_argument("--remap-y", action="store_true")
    parser.add_argument("--q-lookahead", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--cp-segments", type=int, default=1, help="force CP target (e.g. 4 with --lengths 1000)"
    )
    args = parser.parse_args(argv)
    try:
        op = PortPrefill(
            16,
            48,
            "cuda",
            watchdog=True,
            cp_segments=args.cp_segments,
            q_lookahead=args.q_lookahead,
            remap_y=args.remap_y,
        )
        for lengths in args.lengths or [[64], [128], [16], [100]]:
            run_case(op, lengths, args.check)
        torch.cuda.synchronize()
    except (RuntimeError, AssertionError) as error:
        print(f"GDN_DEBUG_ERROR {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
