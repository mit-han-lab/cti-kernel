#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""IKET workload and decoded-trace summarizer. See CP2.md for run-iket commands."""

import argparse
import collections
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
if not (root / "tests/gpu/_gdn_reference.py").is_file():
    root = Path.cwd()
sys.path.insert(0, str(root))


def summarize(path, chunks):
    """Accept the native decoder's range objects or Chrome complete events.

    Duration units are preserved: native start_ts/end_ts use the decoder's clock;
    Chrome ts/dur use microseconds. Aggregate per warp, then divide by chunks.
    Nested waits are included in enclosing compute ranges and excluded from the
    denominator to avoid double-counting them.
    """
    data = json.loads(path.read_text())
    groups = collections.defaultdict(list)

    def walk(node, strings=()):
        if isinstance(node, dict):
            strings = node.get("stringTable", strings)
            name = node.get("range_name", node.get("rangeName", node.get("name", "")))
            if "rangeNameIdx" in node:
                index = node["rangeNameIdx"]
                if isinstance(strings, dict):
                    name = strings.get(str(index), "")
                elif 0 <= index < len(strings):
                    name = strings[index]
            if isinstance(name, str) and name.startswith(("producer.", "math.")):
                if "startTs" in node and "endTs" in node:
                    groups[name].append(float(node["endTs"]) - float(node["startTs"]))
                    return
                if "start_ts" in node and "end_ts" in node:
                    groups[name].append(float(node["end_ts"]) - float(node["start_ts"]))
                    return
                if node.get("ph") == "X" and "dur" in node:
                    groups[name].append(float(node["dur"]))
                    return
            for value in node.values():
                walk(value, strings)
        elif isinstance(node, list):
            for value in node:
                walk(value, strings)

    walk(data)
    if not groups:
        raise ValueError(
            "no CP2 ranges found; supply decoded JSON, not the raw iket.data.json manifest"
        )
    per_chunk = dict.fromkeys(
        (
            "producer.raw_wait",
            "producer.conv",
            "producer.conv_norm",
            "producer.smem_store",
            "producer.z_load",
            "producer.epilogue_norm",
        ),
        32,
    )
    # Old conv_norm traces loaded one row at a time; the ring has a priming range.
    per_chunk["producer.raw_load"] = 32 if "producer.conv_norm" in groups else 33
    # TMA traces have one transform range per K/Q/V, two y ranges, and
    # three elected-warp issue ranges per chunk. Accept old row-load traces.
    transforms = 1 if "producer.tma_issue" in groups else 2
    per_chunk.update(dict.fromkeys(("producer.k", "producer.q", "producer.v"), transforms))
    per_chunk["producer.y"] = 2
    per_chunk["producer.tma_issue"] = 3
    for name, durations in sorted(groups.items()):
        occurrences = per_chunk.get(name, 1)
        print(
            json.dumps(
                {
                    "range": name,
                    "events": len(durations),
                    "mean_duration": sum(durations) / len(durations),
                    "mean_duration_per_chunk": sum(durations) / len(durations) * occurrences,
                }
            )
        )
    for role in ("producer", "math"):
        wait = sum(
            sum(v)
            for k, v in groups.items()
            if k.startswith(role + ".wait") or (role == "producer" and k == "producer.raw_wait")
        )
        total = sum(sum(v) for k, v in groups.items() if k.startswith(role + "."))
        # wait_v is nested inside SK_NewV. Other wait ranges are disjoint.
        if role == "math":
            total -= sum(groups.get("math.wait_v", []))
        print(
            json.dumps(
                {
                    "role": role,
                    "wait_fraction": wait / total if total else 0,
                    "supplied_chunks": chunks,
                    "units": "decoder timestamp units",
                }
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--lengths", default="1024")
    parser.add_argument("--producer-registers", type=int, default=72)
    parser.add_argument("--remap-y", action="store_true")
    parser.add_argument("--q-lookahead", action="store_true")
    parser.add_argument("--prefetch", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args()
    lengths = [int(v) for v in args.lengths.split(",")]
    chunks = sum((v + 63) // 64 for v in lengths)
    if args.summarize:
        summarize(args.summarize, chunks)
        return
    import torch

    from benchmark.kernel_tuning.gdn_prefill_port.cp2_kernel import build

    providers, restore = build(
        args.hk,
        args.hv,
        lengths,
        producer_registers=args.producer_registers,
        iket=True,
        prefetch=args.prefetch,
        q_lookahead=args.q_lookahead,
        remap_y=args.remap_y,
    )
    # One eager invocation, no graph warmups: run-iket runs this script once to
    # discover cubins and once with its native-dump instrumentation injected.
    restore("fused")
    providers["fused"]()
    torch.cuda.synchronize()
    print(f"CP2_IKET sequence_chunks={chunks}; CTA_chunks={chunks * args.hv}")


if __name__ == "__main__":
    main()
