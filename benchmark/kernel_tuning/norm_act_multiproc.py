# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GB10 C3 acceptance over fresh, sequential, X925-pinned harness processes.

Each candidate is compared with min(vllm_us, sglang_us) in the SAME round.
Pool pairs across processes, requiring >=60 pairs across >=3 processes before
acceptance. PASS requires median >=1 and pooled bootstrap CI_low >=1-tol.
Row tolerance is measured A/A noise: use aa_us/ours_us in each round, then
tol = max(0.005, abs(1 - median_aa) + (CI_high_aa - CI_low_aa) / 2).
All CIs are percentile 95% bootstrap intervals for the median, using 10,000
resamples and a fresh random.Random(20261016). Percentiles interpolate linearly.
The informational hierarchical CI resamples processes with replacement, then
pairs within each selected process (preserving that process's sample count).

--consumer adds a down-projection GEMM to each activation's graph. Its A/B
columns compare each --keep candidate with ours: candidate_us/ours_us, >1 means
slower than ours. Hot/cold still describe the activation input working set.
Example: python -m benchmark.kernel_tuning.norm_act_multiproc --procs 4
--rounds 20 --market-root ../market --sglang-root ../sglang --consumer
--keep flashinfer,cute_t256_v1_e2_p1_te_cstef -- --check

This module imports neither torch nor numpy; only the child harness uses CUDA.
MPSUM contains statistics; MPRAW archives per-process samples in microseconds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261016
MIN_PAIRS = 60
MIN_PROCS = 3


def percentile(values, fraction):
    """Linearly interpolated quantile of sorted, nonempty values."""
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_ci(groups, *, hierarchical=False, resamples=BOOTSTRAP_RESAMPLES):
    """Bootstrap a pooled median from nonempty per-process lists of ratios."""
    if not groups or any(not group for group in groups) or resamples < 1:
        raise ValueError("nonempty process samples and positive resamples required")
    rng = random.Random(BOOTSTRAP_SEED)
    pooled = [value for group in groups for value in group]
    medians = []
    for _ in range(resamples):
        if hierarchical:
            sample = [
                value
                for group in rng.choices(groups, k=len(groups))
                for value in rng.choices(group, k=len(group))
            ]
        else:
            sample = rng.choices(pooled, k=len(pooled))
        medians.append(statistics.median(sample))
    medians.sort()
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def pooled_statistics(process_pairs, *, resamples=BOOTSTRAP_RESAMPLES):
    """Summarize {process_id: paired_ratios} without retaining raw samples."""
    groups = list(process_pairs.values())
    values = [value for group in groups for value in group]
    if not values:
        raise ValueError("empty sample")
    return {
        "n_pairs": len(values),
        "n_procs": len(groups),
        "median": statistics.median(values),
        "ci95": bootstrap_ci(groups, resamples=resamples),
        "process_medians": [
            {"process": process, "n_pairs": len(pairs), "median": statistics.median(pairs)}
            for process, pairs in process_pairs.items()
        ],
        "hierarchical_ci95": bootstrap_ci(groups, hierarchical=True, resamples=resamples),
    }


def aa_tolerance(aa_stats):
    return max(0.005, abs(1 - aa_stats["median"]) + (aa_stats["ci95"][1] - aa_stats["ci95"][0]) / 2)


def verdict(stats, tol):
    if stats["n_pairs"] < MIN_PAIRS or stats["n_procs"] < MIN_PROCS:
        return "INSUFFICIENT"
    return "PASS" if stats["median"] >= 1 and stats["ci95"][0] >= 1 - tol else "FAIL"


def validate_samples(samples):
    """Reject incomplete or unpaired receipts rather than silently truncating zip."""
    if not isinstance(samples, dict) or not {"ours", "aa", "vllm", "sglang"} <= samples.keys():
        raise ValueError("raw samples require ours, aa, vllm and sglang")
    count = len(samples["ours"])
    if not count:
        raise ValueError("empty raw samples")
    for values in samples.values():
        if len(values) != count:
            raise ValueError("unpaired sample counts")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("sample times must be finite and positive")


def summarize(processes, *, keep=(), consumer=False, resamples=BOOTSTRAP_RESAMPLES):
    """Pool archived process rows by (M, regime), retaining each round's pairing."""
    grouped = {}
    for process in processes:
        for row in process["rows"]:
            validate_samples(row["raw"])
            by_process = grouped.setdefault((row["M"], row["regime"]), {})
            if process["process"] in by_process:
                raise ValueError("duplicate row in one process")
            by_process[process["process"]] = row["raw"]
    rows = []
    for (m, regime), samples in sorted(grouped.items()):
        aa = pooled_statistics(
            {
                process: [a / o for a, o in zip(raw["aa"], raw["ours"], strict=True)]
                for process, raw in samples.items()
            },
            resamples=resamples,
        )
        tol = aa_tolerance(aa)
        row = {"M": m, "regime": regime, "aa": aa, "tol": tol, "candidates": {}}
        names = sorted(set().union(*(raw.keys() for raw in samples.values())) - {"vllm", "sglang"})
        for name in names:
            stats = pooled_statistics(
                {
                    process: [
                        min(v, s) / candidate
                        for v, s, candidate in zip(
                            raw["vllm"], raw["sglang"], raw[name], strict=True
                        )
                    ]
                    for process, raw in samples.items()
                    if name in raw
                },
                resamples=resamples,
            )
            stats["verdict"] = verdict(stats, tol)
            row["candidates"][name] = stats
        if consumer:
            row["consumer"] = "down_proj_mm_4096"
            row["consumer_ab"] = {
                name: {
                    "numerator": name,
                    "denominator": "ours",
                    **pooled_statistics(
                        {
                            process: [c / o for c, o in zip(raw[name], raw["ours"], strict=True)]
                            for process, raw in samples.items()
                            if name in raw
                        },
                        resamples=resamples,
                    ),
                }
                for name in sorted(set(keep))
                if any(name in raw for raw in samples.values())
            }
        rows.append(row)
    return rows


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--rows", type=int, nargs="+", default=[256])
    parser.add_argument("--regimes", nargs="+", choices=("hot", "cold"), default=["hot", "cold"])
    parser.add_argument("--consumer", action="store_true")
    parser.add_argument("--keep")
    parser.add_argument("--market-root")
    parser.add_argument("--sglang-root")
    args = parser.parse_args(argv)
    if args.procs < 1 or args.rounds < 6 or min(args.rows) <= 0:
        parser.error("positive procs/rows and at least 6 rounds required")
    args.extra = extra
    return args


def child_command(args):
    command = [
        sys.executable,
        "-m",
        "benchmark.kernel_tuning.norm_act_paired",
        *args.extra,
        "--op",
        "silu_mul",
        "--raw",
        "--rounds",
        str(args.rounds),
        "--rows",
        *map(str, args.rows),
        "--regimes",
        *args.regimes,
    ]
    for flag in ("market_root", "sglang_root", "keep"):
        value = getattr(args, flag)
        if value is not None:
            command.extend(["--" + flag.replace("_", "-"), value])
    if args.consumer:
        command.append("--consumer")
    return command


def collect_row(line, process):
    """Parse only ROW receipts; retain timing data, never provider diagnostics."""
    row = json.loads(line[4:])
    validate_samples(row.get("raw"))
    key = (row["M"], row["regime"])
    if any((prior["M"], prior["regime"]) == key for prior in process["rows"]):
        raise ValueError("duplicate row")
    process["rows"].append({"M": key[0], "regime": key[1], "raw": row["raw"]})


def print_table(rows):
    print(
        "M regime candidate pairs procs median CI95 tol verdict process_medians hierarchical_CI95"
    )
    for row in rows:
        for name, stats in row["candidates"].items():
            medians = ",".join(
                f"P{p['process']}:{p['median']:.4f}" for p in stats["process_medians"]
            )
            print(
                row["M"],
                row["regime"],
                name,
                stats["n_pairs"],
                stats["n_procs"],
                f"{stats['median']:.4f}",
                f"[{stats['ci95'][0]:.4f},{stats['ci95'][1]:.4f}]",
                f"{row['tol']:.4f}",
                stats["verdict"],
                medians,
                f"[{stats['hierarchical_ci95'][0]:.4f},{stats['hierarchical_ci95'][1]:.4f}]",
            )
        for name, stats in row.get("consumer_ab", {}).items():
            print(
                "AB",
                row["M"],
                row["regime"],
                f"{name}/ours",
                f"{stats['median']:.4f}",
                f"[{stats['ci95'][0]:.4f},{stats['ci95'][1]:.4f}] (>1 = slower than ours)",
            )


def main(argv=None):
    args = parse_args(argv)
    processes = []
    command = child_command(args)
    for index in range(1, args.procs + 1):
        process = {"process": index, "rows": [], "errors": []}
        processes.append(process)
        try:
            child = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except OSError as error:
            process["returncode"] = 1
            process["errors"].append(type(error).__name__)
            continue
        for line in child.stdout:
            if line.startswith("ROW "):
                try:
                    collect_row(line, process)
                except (ValueError, KeyError, TypeError) as error:
                    process["errors"].append("invalid ROW: " + type(error).__name__)
            elif len(line.rstrip("\n")) <= 400 and not any(
                word in line.upper() for word in ("SECRET", "TOKEN")
            ):
                print(f"P{index} {line.rstrip()}", flush=True)
        child.stdout.close()
        process["returncode"] = child.wait()
    rows = summarize(processes, keep=(args.keep or "").split(","), consumer=args.consumer)
    print_table(rows)
    summary = {
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "method": "percentile",
        },
        "requirements": {
            "min_pairs": MIN_PAIRS,
            "min_procs": MIN_PROCS,
            "pinning": "child pin_x925",
        },
        "tol_definition": "max(0.005, abs(1 - median(aa/ours)) + (CI_high_aa - CI_low_aa)/2)",
        "rows": rows,
        "processes": [{key: value for key, value in p.items() if key != "rows"} for p in processes],
    }
    print("MPSUM " + json.dumps(summary, separators=(",", ":")))
    archive = [
        {
            **process,
            "rows": [
                {
                    **row,
                    "raw": {
                        name: [round(us, 3) for us in times] for name, times in row["raw"].items()
                    },
                }
                for row in process["rows"]
            ],
        }
        for process in processes
    ]
    print("MPRAW " + json.dumps({"processes": archive}, separators=(",", ":")))
    return int(any(process["returncode"] != 0 for process in processes))


if __name__ == "__main__":
    sys.exit(main())
