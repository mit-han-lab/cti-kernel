# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Extract the numeric SOL inputs from full SOL ceiling tables.

Usage: python benchmark/sol/compact_ceilings.py FULL_DIRECTORY [--output DIRECTORY]
See PROVENANCE.md for the access-path audit and upstream hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.sol import sol

MEMORY_RATES = ("dram_read_gbs", "dram_write_gbs", "dram_copy_gbs", "l2_read_gbs")
FIXED_COSTS = ("graph_node_us", "graph_node_pdl_us", "eager_launch_us")
CTA_FIELDS = (
    "threads",
    "smem_bytes",
    "regs_per_thread",
    "theoretical_residency_per_sm",
    "measured_max_concurrent",
    "q_saturation",
    "cta_capacity",
)


def select(mapping, keys):
    """Preserve absent rates so missing_measurements retains its meaning."""
    return {key: mapping[key] for key in keys if key in mapping}


def compact(raw: bytes) -> dict:
    table = sol.load_ceilings(json.loads(raw))
    memory = table["memory"]
    result = select(
        table,
        ("SPDX-License-Identifier", "schema_version", "device", "sm_count", "l2_bytes"),
    )
    result["source"] = (
        "Compacted from kernels/sol-ceilings commit e95f6688; full-file SHA-256: "
        + hashlib.sha256(raw).hexdigest()
    )
    result.update(select(table, ("missing_measurements",)))
    result["memory"] = select(memory, MEMORY_RATES)
    # Required legacy list: validated, but never interpolated. Its full points
    # duplicate the read curve. Keep every computational curve point unchanged.
    result["memory"]["small_transfer"] = []
    result["memory"]["small_transfer_by_direction"] = {
        direction: [select(point, ("bytes", "gbs", "time_us")) for point in curve]
        for direction, curve in memory.get("small_transfer_by_direction", {}).items()
        if direction in ("read", "write", "copy")
    }
    result["tensor_core"] = {
        kind: select(table["tensor_core"][kind], ("tflops_gpu", "tflops_per_sm"))
        for kind in sol.MMA_KINDS
        if kind in table.get("tensor_core", {})
    }
    result["fixed_costs"] = select(table["fixed_costs"], FIXED_COSTS)
    # Preserve every shape and its index (GDN measurement_ref uses that index).
    # The helper also echoes **chosen as evidence; omit only raw measurements.
    result["scheduling"] = {
        "cta_capacity": [select(entry, CTA_FIELDS) for entry in table["scheduling"]["cta_capacity"]]
    }
    return sol.load_ceilings(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing the full tables")
    parser.add_argument("--output", type=Path, default=sol.CEILINGS_DIR)
    args = parser.parse_args()
    if args.directory.resolve() == args.output.resolve():
        parser.error("input and output directories must differ; preserve the full tables")
    outputs = {}
    for device in sol.DEVICES:
        raw = (args.directory / f"{device}.json").read_bytes()
        outputs[device] = json.dumps(compact(raw), separators=(",", ":"), allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    for device, data in outputs.items():
        (args.output / f"{device}.json").write_text(data, encoding="utf-8")
        print(f"{device}.json: {len(data.encode('utf-8'))} bytes")


if __name__ == "__main__":
    main()
