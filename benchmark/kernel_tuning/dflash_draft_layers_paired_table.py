# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Summarize paired per-layer DFlash JSONL mixed with runner output (CPU only).

    python -m benchmark.kernel_tuning.dflash_draft_layers_paired_table \
        --log GPU=paired-q4.log --log GPU=paired-q16.log --csv paired.csv

Inputs are processed in command-line order; the last complete receipt for each
device/shape/head_dim/Q/B/ctx/capacity wins, including explicit ERROR receipts.
Ratios are the runner's medians of paired ratios, never ratios of medians.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from benchmark.kernel_tuning.dflash_draft_layers import SHAPES

REFERENCES = {"vllm_fa2": "vllm_flash_attn", "sglang": "SGLang FI 0.7.1"}
ROW_KEY = ("shape", "head_dim", "Q", "B", "ctx", "capacity")
FIELDS = (
    "device",
    *ROW_KEY,
    "op.provider",
    "candidate_us",
    "vllm_flash_attn_us",
    "sglang_fi071_us",
    "faster_market",
    "median_ratio",
    "epsilon",
    "other_reference",
    "other_median_ratio",
    "other_epsilon",
    "vllm_flash_attn_status",
    "sglang_fi071_status",
    "correctness",
    "market_gate",
    "status",
)


def log_argument(value):
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path:
        raise argparse.ArgumentTypeError("expected DEVICE_LABEL=PATH")
    return label, Path(path)


def complete_row(row):
    if not all(key in row for key in ("shape", "Q", "B", "ctx", "capacity")):
        return False
    if row.get("status") == "ERROR":
        return True
    if row.get("status") not in ("PASS", "FAIL", "N/A"):
        return False
    if not all(key in row for key in ("references", "correctness", "market_gate", "faster_market")):
        return False
    for name in REFERENCES:
        ref = row["references"].get(name, {})
        if ref.get("status") == "N/A":
            continue
        if ref.get("status") not in ("PASS", "FAIL") or any(
            ref.get(key) is None
            for key in ("median_ratio", "epsilon", "candidate_median_us", "reference_median_us")
        ):
            return False
    return True


def table_row(device, row):
    refs = row.get("references", {})
    faster = row.get("faster_market")
    other = next((name for name in REFERENCES if name != faster), None) if faster else None
    fast_ref, other_ref = refs.get(faster, {}), refs.get(other, {})
    capacity = row["capacity"]
    correctness = row.get("correctness", {})
    result = {
        "device": device,
        **{key: row[key] for key in ("shape", "Q", "B", "ctx")},
        "head_dim": row.get("d", SHAPES.get(row["shape"], {}).get("d")),
        "capacity": "exact" if capacity == row["ctx"] + row["Q"] else f"C{capacity}",
        "op.provider": row.get("op.provider"),
        "candidate_us": fast_ref.get("candidate_median_us"),
        "vllm_flash_attn_us": refs.get("vllm_fa2", {}).get("reference_median_us"),
        "sglang_fi071_us": refs.get("sglang", {}).get("reference_median_us"),
        "faster_market": REFERENCES.get(faster),
        "median_ratio": fast_ref.get("median_ratio"),
        "epsilon": fast_ref.get("epsilon"),
        "other_reference": REFERENCES.get(other),
        "other_median_ratio": other_ref.get("median_ratio"),
        "other_epsilon": other_ref.get("epsilon"),
        "vllm_flash_attn_status": refs.get("vllm_fa2", {}).get("status", "N/A"),
        "sglang_fi071_status": refs.get("sglang", {}).get("status", "N/A"),
        "correctness": json.dumps(
            {name: correctness.get(name, "N/A") for name in ("candidate", *REFERENCES)},
            sort_keys=True,
        ),
        "market_gate": row.get("market_gate", "ERROR"),
        "status": row["status"],
    }
    if row["status"] == "ERROR":
        result["market_gate"] = "ERROR"
    return result


def read_logs(logs):
    rows, provenance = {}, defaultdict(list)
    duplicates = incomplete = 0
    decoder = json.JSONDecoder()
    for device, path in logs:
        metadata = {}
        provenance[device].append(metadata)
        raw = re.sub(r"USDT:[^\n]*\n", "", path.read_text())
        for line in raw.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            prefix = line[:start].strip()
            try:
                record, _ = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                continue  # Chatter and truncated retries are not complete receipts.
            if not isinstance(record, dict):
                continue
            if prefix == "WHEEL":
                metadata.update(
                    wheel=Path(record["file"]).name if record.get("file") else None,
                    sha256=record.get("sha256"),
                )
            elif prefix == "INSTALLED":
                metadata.update(
                    version=record.get("version"),
                    import_path=record.get("import"),
                    record_check=record.get("init_matches_RECORD"),
                    source_tree_removed=(
                        not record["source_tree_present"]
                        if isinstance(record.get("source_tree_present"), bool)
                        else None
                    ),
                )
            elif record.get("type") == "header":
                # Keep only provenance fields, excluding host/environment chatter.
                metadata.setdefault("workers", []).append(
                    {
                        name: {
                            key: worker.get(key)
                            for key in ("device", "cores", "x925_cores", "versions")
                        }
                        for name, worker in record.get("workers", {}).items()
                    }
                )
            elif record.get("type") == "row":
                if not complete_row(record):
                    incomplete += 1
                    continue
                row = table_row(device, record)
                key = (device, *(row[field] for field in ROW_KEY))
                duplicates += key in rows
                rows[key] = row
    if duplicates:
        print(
            f"warning: {duplicates} duplicate row(s); kept the last complete receipt per key",
            file=sys.stderr,
        )
    if incomplete:
        print(f"warning: ignored {incomplete} incomplete row receipt(s)", file=sys.stderr)
    return list(rows.values()), provenance


def cell(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def grid(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
        "",
    ]


def provenance_row(device, records):
    values = defaultdict(list)

    def add(key, value):
        if value is not None and cell(value) not in values[key]:
            values[key].append(cell(value))

    for record in records:
        for key in (
            "wheel",
            "sha256",
            "version",
            "import_path",
            "record_check",
            "source_tree_removed",
        ):
            add(key, record.get(key))
        for workers in record.get("workers", []):
            for name, worker in workers.items():
                add("device_name", worker.get("device"))
                for key in ("cores", "x925_cores"):
                    if worker.get(key):
                        add(key, f"{name}: {worker[key]}")
                versions = worker.get("versions") or {}
                for package in ("torch", "cti-kernels", "flashinfer"):
                    if versions.get(package) is not None:
                        add("versions", f"{name}: {package}={versions[package]}")
                if record.get("version") is None:
                    add("version", versions.get("cti-kernels"))
                if record.get("import_path") is None:
                    add("import_path", versions.get("ctikernels_import_path"))
    return [
        device,
        *(
            "; ".join(values[key]) or None
            for key in (
                "device_name",
                "wheel",
                "sha256",
                "version",
                "import_path",
                "record_check",
                "source_tree_removed",
                "x925_cores",
                "versions",
                "cores",
            )
        ),
    ]


def summary(rows):
    failed = [
        row for row in rows if row["market_gate"] == "FAIL" and row["median_ratio"] is not None
    ]
    worst = max(failed, key=lambda row: row["median_ratio"], default={})
    # Strict count: correct rows whose paired median is not above the faster market.
    strict = sum(
        row["market_gate"] != "N/A"
        and row["status"] != "ERROR"
        and row["median_ratio"] is not None
        and row["median_ratio"] <= 1
        and "FAIL" not in str(row["correctness"])
        for row in rows
    )
    epsilons = sorted(row["epsilon"] for row in rows if row["epsilon"] is not None)
    return [
        f"{sum(row['market_gate'] == 'PASS' for row in rows)} / {len(rows)}",
        f"{strict} / {len(rows)}",
        epsilons[len(epsilons) // 2] if epsilons else None,
        worst.get("median_ratio"),
        worst.get("epsilon"),
        sum(row["status"] == "ERROR" for row in rows),
        sum(row["market_gate"] == "N/A" for row in rows),
    ]


def markdown(rows, provenance):
    output = ["## Provenance", ""]
    output += grid(
        [
            "Device",
            "Device name",
            "Wheel",
            "SHA256",
            "Version",
            "Import path",
            "RECORD check",
            "Source tree removed",
            "X925 cores",
            "Worker versions",
            "Worker cores",
        ],
        [provenance_row(device, records) for device, records in sorted(provenance.items())],
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["device"], row["shape"], row["ctx"]].append(row)
    columns = [
        "Passing rows / rows",
        "Ratio <= 1 / rows",
        "Median epsilon",
        "Worst failing median ratio",
        "Its epsilon",
        "ERROR",
        "N/A",
    ]
    output += ["## Summary", ""]
    output += grid(
        ["Device", "Shape", "ctx", *columns],
        [[*key, *summary(group)] for key, group in sorted(grouped.items())],
    )
    output += [
        "## DF3 target totals",
        "",
        "Observed rows: both hd128 shapes, ctx 100/400/1000, B 1–4, exact/C32768. "
        "Expected coverage is 48 rows per Q present on the device. ERROR and N/A do not pass.",
        "",
    ]
    totals = []
    for device in sorted(provenance):
        device_rows = [row for row in rows if row["device"] == device]
        queries = sorted({row["Q"] for row in device_rows})
        target = [
            row
            for row in device_rows
            if row["shape"] in ("draft_noncausal", "q3_target")
            and row["head_dim"] == 128
            and row["ctx"] in (100, 400, 1000)
            and row["B"] in (1, 2, 3, 4)
            and row["capacity"] in ("exact", "C32768")
        ]
        totals.append([device, ", ".join(map(str, queries)), *summary(target), 48 * len(queries)])
    output += grid(["Device", "Q present", *columns, "Expected rows"], totals)
    output += ["## Failing rows", ""]
    failures = [row for row in rows if row["market_gate"] in ("FAIL", "ERROR")]
    if failures:
        output += grid(
            [
                "Device",
                "Shape",
                "Q",
                "B",
                "ctx",
                "Capacity",
                "Ratio",
                "Epsilon",
                "Faster market",
                "Status",
            ],
            [
                [
                    row[key]
                    for key in (
                        "device",
                        "shape",
                        "Q",
                        "B",
                        "ctx",
                        "capacity",
                        "median_ratio",
                        "epsilon",
                        "faster_market",
                        "market_gate",
                    )
                ]
                for row in failures
            ],
        )
    else:
        output += ["None.", ""]
    return "\n".join(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", action="append", type=log_argument, required=True, metavar="DEVICE_LABEL=PATH"
    )
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, provenance = read_logs(args.log)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(markdown(rows, provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
