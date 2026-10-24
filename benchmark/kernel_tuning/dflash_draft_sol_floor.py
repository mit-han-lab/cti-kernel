# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT SOL against a floor that includes the measured graph-replay step.

The market gate's SOL t_ref (dflash_draft_attention.sol_estimate, L2-aware) counts
useful work plus independently measured graph-node costs. The gate itself times
back-to-back ``CUDAGraph.replay()`` calls, so a launch-bound row can never go
below the replay step of a k-kernel graph. Consecutive replays pipeline host
launch with device work, so the corrected reference is

    t_ref_corrected = max(t_ref_work, replay_floor(k))

where replay_floor(k) is the median replay of a k-kernel graph measured with the
same protocol by dflash_replay_floor.py on the same device class, and k is the
complete op's kernel count recorded with each row. Utilization is
t_ref_corrected / candidate latency.

Usage:
  python -m benchmark.kernel_tuning.dflash_draft_sol_floor \
    --floor rtx5090 floor-5090.jsonl --gate rtx5090 'gate-5090-*.log' [...]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DEVICES = {"rtx5090": "RTX 5090", "rtx-pro-6000": "RTX PRO 6000", "gb10": "GB10"}
KEY = ("mask", "batch_size", "query_length", "context", "num_qo_heads", "num_kv_heads")
# Candidate latency comes from the same comparator order as the market gate.
ORDER = ("flashinfer", "sglang", "vllm")


def records(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for raw in re.sub(r"USDT:[^\n]*\n", "", text).splitlines():
        line = raw.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_floor(paths):
    floors = [r for p in paths for r in records(p) if r.get("type") == "replay_floor"]
    if not floors:
        raise ValueError(f"no replay_floor record in {paths}")
    # Several hosts: take the lowest median per kernel count (the strictest floor).
    return {
        int(k): min(f["replay_us"][k]["median"] for f in floors if k in f["replay_us"])
        for k in floors[0]["replay_us"]
    }


def load_gate(paths):
    rows = defaultdict(dict)
    for path in paths:
        for r in records(path):
            if r.get("type") != "row" or r.get("comparator") not in ORDER:
                continue
            if not r.get("candidate_median_us") or not r.get("kernels_per_call"):
                continue
            rows[tuple(r[k] for k in KEY)][r["comparator"]] = r
    return {k: next(v[c] for c in ORDER if c in v) for k, v in rows.items()}


def work_t_ref(device, row):
    from benchmark.kernel_tuning.dflash_draft_attention import l2_fraction, sol_estimate

    mask, b, q, context, hq, hk = (row[k] for k in KEY)
    kernels = row["kernels_per_call"]["kernels"]
    _, _, fraction = l2_fraction(device, mask, b, context, hq, hk, query_length=q)
    _, warm = sol_estimate(
        DEVICES[device], mask, b, context, hq, hk, kernels, l2_fraction=fraction, query_length=q
    )
    return warm.t_ref_us


def evaluate(device, floor, rows):
    out = []
    for key, row in sorted(rows.items()):
        kernels = row["kernels_per_call"]["kernels"]
        work = work_t_ref(device, row)
        corrected = max(work, floor[kernels])
        candidate = row["candidate_median_us"]
        out.append(
            {
                "device": device,
                "key": ":".join(str(k) for k in key),
                "kernels": kernels,
                "candidate_us": candidate,
                "t_ref_work_us": work,
                "replay_floor_us": floor[kernels],
                "t_ref_corrected_us": corrected,
                "util_work": work / candidate,
                "util_corrected": corrected / candidate,
                "launch_bound": floor[kernels] >= work,
            }
        )
    return out


def summary(results):
    lines = []
    for device in DEVICES:
        rows = [r for r in results if r["device"] == device]
        if not rows:
            continue
        util = [r["util_corrected"] for r in rows]
        bound = Counter(r["launch_bound"] for r in rows)
        met = sum(u >= 0.9 for u in util)
        lines.append(
            f"{device}: rows={len(rows)} util_corrected {min(util):.1%}-{max(util):.1%}, "
            f">=90%: {met}/{len(rows)}; launch-bound rows {bound[True]} "
            f"(>=90%: {sum(r['util_corrected'] >= 0.9 for r in rows if r['launch_bound'])})"
        )
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--floor", nargs=2, action="append", required=True, metavar=("DEVICE", "GLOB")
    )
    parser.add_argument(
        "--gate", nargs=2, action="append", required=True, metavar=("DEVICE", "GLOB")
    )
    parser.add_argument("--jsonl", help="write per-row results here")
    args = parser.parse_args(argv)
    floors = defaultdict(list)
    for device, pattern in args.floor:
        floors[device] += sorted(glob.glob(pattern))
    gates = defaultdict(list)
    for device, pattern in args.gate:
        gates[device] += sorted(glob.glob(pattern))
    results = []
    for device, paths in gates.items():
        if device not in DEVICES:
            parser.error(f"unknown device {device}")
        results += evaluate(device, load_floor(floors[device]), load_gate(paths))
    if args.jsonl:
        Path(args.jsonl).write_text("".join(json.dumps(r) + "\n" for r in results))
    print("\n".join(summary(results)))


if __name__ == "__main__":
    main()
