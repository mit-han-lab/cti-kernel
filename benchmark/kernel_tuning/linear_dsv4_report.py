# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DSV4 Flash / Qwen3-MoE Linear rows: paired acceptance table from RESULT logs (CPU only).

Reads `linear_smallm.py --accept` logs (cold/engine/hot runs), resolves retries
with the shared evidence resolver (`linear_smallm_table.resolve_retries`: the
latest finished run decides; a correctness FAIL counts from any run) and prints one
Markdown row per device x shape x M: the shipped provider, our and the market's
(cuBLAS) cold time, the paired speedup and epsilon per regime, and cold SOL
against the round-3 measured DRAM read ceilings. Correctness FAIL rows are shown
as FAIL and their timings never enter the per-step totals. ``--model`` selects the
projection names and calls per step (DeepSeek-V4-Flash by default).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Support the documented direct-script invocation as well as module imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.kernel_tuning import linear_smallm_table as evidence
from ctikernels.linear import _dispatch

SHAPES = {
    (129280, 4096, "bf16"): ("lm_head", "F.linear (vLLM) / torch.matmul (SGLang)"),
    (64, 4096, "bf16"): ("indexer.weights_proj", "F.linear (both)"),
    (2048, 4096, "fp32"): ("compressor.wkv_gate, ratio 4", "torch.mm(out_dtype=fp32) (both)"),
    (1024, 4096, "fp32"): ("compressor.wkv_gate, ratio 128", "torch.mm(out_dtype=fp32) (both)"),
    (512, 4096, "fp32"): ("indexer.compressor.wkv_gate", "torch.mm(out_dtype=fp32) (both)"),
}
REGIMES = ("cold", "engine", "hot")
# Calls per decode step of the 43-layer target (MTP excluded): 21 ratio-4 and 20
# ratio-128 compressed-attention layers; the indexer runs in the ratio-4 layers.
CALLS = {(129280, 4096, "bf16"): 1, (64, 4096, "bf16"): 21, (2048, 4096, "fp32"): 21}
CALLS.update({(1024, 4096, "fp32"): 20, (512, 4096, "fp32"): 21})
# Qwen3-30B-A3B (Qwen3-MoE, K-Q6): 48 MoE layers, one router each; vLLM 0.31.0
# GateLinear (specialized tiers need SM90/SM100, out_dtype unset) and SGLang
# 72730d6 ReplicatedLinear both run F.linear on sm120/sm121.
QWEN3_MOE_SHAPES = {
    (128, 2048, "bf16"): ("router.gate", "F.linear (both)"),
    (151936, 2048, "bf16"): ("lm_head", "F.linear (vLLM) / torch.matmul (SGLang)"),
}
QWEN3_MOE_CALLS = {(128, 2048, "bf16"): 48, (151936, 2048, "bf16"): 1}
MODELS = {"dsv4": (SHAPES, CALLS), "qwen3-30b-a3b": (QWEN3_MOE_SHAPES, QWEN3_MOE_CALLS)}
# Round-3 measured DRAM read ceilings, GB/s (benchmark/sol/ceilings/*.json, sol-ceilings).
READ_GBPS = {"rtx5090": 1681.1, "pro6000": 1533.1, "gb10": 244.6}


def sol_fraction(row, us):
    """Weight + activation reads and output writes over the read ceiling, as a fraction."""
    n, k, m = row["N"], row["K"], row["M"]
    out = 4 if row.get("out_dtype") == "fp32" else 2
    return (2 * (n * k + m * k) + out * m * n) / (READ_GBPS[row["device_class"]] * 1e3) / us


def valid(row):
    """A correct measurement: not a correctness FAIL and not invalidated by a later one."""
    return row.get("correct") is True and not row.get("_invalidated")


def records(paths):
    """Resolved record per (device, out dtype, N, K, M, L2) from finished runs or FAILs."""
    grouped = defaultdict(list)
    for row in evidence.read_logs(paths, round_label="current round"):
        kind = evidence.device_kind(row)
        if row.get("kind") == "smallm_accept" and kind is not None:
            grouped[kind].append(row)
    latest = {}
    for kind, entries in grouped.items():
        for row in evidence.resolve_retries(entries):
            if evidence.decided(row):
                key = (kind, row.get("out_dtype", "bf16"), row["N"], row["K"])
                latest[key + (row["M"], row.get("l2", "cold"))] = row
    return latest


def shipped(device, n, k, m, out_dtype):
    table = _dispatch.TUNED_F32 if out_dtype == "fp32" else _dispatch.TUNED
    for (lo, hi), config in table.get(device, {}).get((n, k), {}).items():
        if lo <= m <= hi:
            return f"cute {config.rows}r/{config.tile_k}k/s{config.splits}/{config.variant}"
    return "cuBLAS"


def markdown(latest, shapes=SHAPES):
    lines = [
        "| device | projection | dtype | N | M | shipped | ours µs | market µs | "
        "cold × (ε) | engine × (ε) | hot × (ε) | SOL ours / market |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    keys = sorted({key[:5] for key in latest})
    for device, out_dtype, n, k, m in keys:
        name = shapes.get((n, k, out_dtype), (f"{n}x{k}", ""))[0]
        cold = latest.get((device, out_dtype, n, k, m, "cold"))
        cells = []
        for regime in REGIMES:
            row = latest.get((device, out_dtype, n, k, m, regime))
            if row is None:
                cells.append("pending")
            elif not valid(row):
                cells.append("FAIL")
            else:
                cells.append(f"{evidence.speedup_cell(row, 3)} ({row['epsilon']:.3f})")
        if cold and not valid(cold):
            times, sol = "FAIL | FAIL", "FAIL"
        elif cold:
            times = f"{cold['ours_us']:.2f} | {cold['cublas_us']:.2f}"
            sol = f"{sol_fraction(cold, cold['ours_us']):.0%} / {sol_fraction(cold, cold['cublas_us']):.0%}"
        else:
            times, sol = "pending | pending", "pending"
        lines.append(
            f"| {device} | {name} | {out_dtype} | {n} | {m} | "
            f"{shipped(device, n, k, m, out_dtype)} | {times} | {' | '.join(cells)} | {sol} |"
        )
    return "\n".join(lines)


def step_totals(latest, shapes=SHAPES, calls_per_step=CALLS):
    """Cold per-step sums: shipped (CuTe where dispatched, else cuBLAS) vs cuBLAS.

    A step whose projection has no correct cold measurement has no total; the
    row names the failing projections instead.
    """
    lines = [
        "| device | M | shipped ms | market ms | speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    devices = sorted({key[0] for key in latest})
    for device in devices:
        for m in sorted({key[4] for key in latest if key[0] == device}):
            ours = market = 0.0
            failed, missing = [], False
            for (n, k, out_dtype), calls in calls_per_step.items():
                row = latest.get((device, out_dtype, n, k, m, "cold"))
                if row is None:
                    missing = True
                    break
                if not valid(row):
                    failed.append(shapes[(n, k, out_dtype)][0])
                    continue
                cute = shipped(device, n, k, m, out_dtype) != "cuBLAS"
                ours += calls * (row["ours_us"] if cute else row["cublas_us"])
                market += calls * row["cublas_us"]
            if missing:
                continue
            if failed:
                lines.append(f"| {device} | {m} | FAIL: {', '.join(failed)} | excluded | n/a |")
            else:
                lines.append(
                    f"| {device} | {m} | {ours / 1e3:.3f} | {market / 1e3:.3f} | "
                    f"{market / ours:.3f} |"
                )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--model", choices=sorted(MODELS), default="dsv4")
    args = parser.parse_args(argv)
    shapes, calls = MODELS[args.model]
    latest = records(args.logs)
    latest = {key: row for key, row in latest.items() if (key[2], key[3], key[1]) in shapes}
    print(markdown(latest, shapes))
    print()
    print(step_totals(latest, shapes, calls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
