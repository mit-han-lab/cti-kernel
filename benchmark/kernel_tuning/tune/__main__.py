# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Offline tuner CLI.

  run <adapter> --out DIR [adapter options]   sweep on the current card; writes
      <op>@<device class>.{json,provenance.json,raw.json} into DIR
  merge DIR... --into DEST                     union of per-job tables of one device class
  report DIR...                                which dimensions move the best config

Adapters live next to this file (``lora_segmented``, ``moe_lora``, ...); each exposes
``OP``, ``add_arguments(parser)``, ``buckets(args)`` (shape dicts) and
``case(shape, device)``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from benchmark._harness import paired, timing
from benchmark.kernel_tuning.tune import harness
from ctikernels._runtime.tables import DeviceClass, device_class, dumps, encode


def _device_info(device):
    import torch
    import triton

    props = torch.cuda.get_device_properties(device)
    return {
        "device_class": device_class(device).name,
        "device": props.name,
        "driver": _driver(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }


def _driver():
    try:
        query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        return subprocess.run(query, capture_output=True, text=True, check=True).stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


# Device classes with a measured ceiling table in benchmark/sol/ceilings.
SOL_DEVICES = {"sm120-170": "rtx5090", "sm120-188": "rtx-pro-6000", "sm121-48": "gb10"}


def _annotate(adapter, device_class_name, cases, rows, timer):
    """Time each row's served config; add us, and the SOL fraction where a ceiling exists."""
    from benchmark.sol.sol import estimate

    ceilings = SOL_DEVICES.get(device_class_name)
    bar = getattr(adapter, "SOL_BAR", 0.90)  # Triton 90%, CuTe 95% of the measured SOL
    for row in rows:
        case = cases.get(row["shape"])
        us = statistics.median(timer(case.run(row["config"])))
        row["us"] = round(us, 3)
        if ceilings is not None and case.sol is not None:
            sol_us = estimate(case.sol(), ceilings).t_ref_us
            row["sol_us"] = round(sol_us, 3)
            row["sol_fraction"] = round(sol_us / us, 4)
            row["under_bar"] = sol_us / us < bar


class _Cases:
    """Builds each bucket's inputs on demand and keeps only the latest alive."""

    def __init__(self, adapter, device):
        self.adapter, self.device = adapter, device
        self._key, self._case = None, None

    def get(self, shape):
        key = json.dumps(shape, sort_keys=True)
        if key != self._key:
            self._case = None
            self._case = self.adapter.case(shape, self.device)
            self._key = key
        return self._case


def run(args, adapter):
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    op = adapter.OP
    timer = timing.TIMERS[args.timer]
    screen = lambda fn: timer(fn, samples=args.screen_samples, calls=args.calls)
    full = lambda fn: timer(fn, samples=args.samples, calls=args.calls)
    cases = _Cases(adapter, device)
    hardware = device_class(device)
    info = _device_info(device)
    index, count = map(int, args.shard.split("/"))
    out = Path(args.out) if count == 1 else Path(args.out) / f"shard-{index}"
    stem = out / f"{op.name}@{info['device_class']}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    buckets = []
    with open(f"{stem}.raw.jsonl", "w") as raw:  # one line per bucket, kept if the call dies
        for shape in harness.shard(op, adapter.buckets(args), index, count):
            bucket = harness.search(
                op,
                hardware,
                cases.get(shape),
                screen,
                lambda t: t.clone(),
                lambda a, b: torch.equal(a, b),
            )
            print(
                f"{shape} best {bucket.best} {bucket.default_us:.2f}->{bucket.best_us:.2f} us",
                flush=True,
            )
            if bucket.all_rejected:
                print(f"WARNING {shape}: every candidate rejected: {bucket.errors}", flush=True)
            raw.write(json.dumps(asdict(bucket), default=harness._plain) + "\n")
            raw.flush()
            buckets.append(bucket)

    if args.screen_only:
        print(f"screen only: {len(buckets)} buckets in raw.jsonl", flush=True)
        return

    def aba(bucket, a, b):
        case = cases.get(bucket.shape)
        return 1 / paired.aba(case.run(a), case.run(b), full, rounds=args.rounds).ratio

    rows, provenance = harness.keep_rows(op, hardware, buckets, aba)
    for row in provenance:
        case = cases.get(row["shape"])
        if case.comparator is not None:
            name, fn = case.comparator
            pair = paired.aba(fn, case.run(row["config"]), full, rounds=args.rounds)
            row["comparator"] = {"name": name, "speedup": round(1 / pair.ratio, 4)}
    _annotate(adapter, info["device_class"], cases, provenance, full)
    Path(f"{stem}.json").write_text(encode(op, rows))
    Path(f"{stem}.provenance.json").write_text(harness.provenance(op, info, provenance, {}))
    print(f"table: {len(rows)} rows of {len(buckets)} buckets", flush=True)

    def probe(bucket, configs):
        case = cases.get(bucket.shape)
        times = []
        for config in configs:
            try:
                valid = op.valid(config, bucket.shape)
                times.append(statistics.median(screen(case.run(config))) if valid else None)
            except Exception:  # noqa: BLE001 - a neighbour's config may not build here
                times.append(None)
        return times

    report = harness.sensitivity(op, buckets, probe)
    Path(f"{stem}.provenance.json").write_text(harness.provenance(op, info, provenance, report))
    print(json.dumps(harness.summarize([report]), indent=1))


def annotate(args, adapter):
    """Add us / SOL columns to an existing provenance file, on the card it was measured on."""
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    info = _device_info(device)
    path = Path(args.table) / f"{adapter.OP.name}@{info['device_class']}.provenance.json"
    source = json.loads(path.read_text())
    cases = _Cases(adapter, device)
    timer = timing.TIMERS[args.timer]
    _annotate(adapter, info["device_class"], cases, source["rows"],
              lambda fn: timer(fn, samples=args.samples, calls=args.calls))  # fmt: skip
    source["annotated_on"] = {k: info[k] for k in ("device", "driver", "triton", "torch")}
    path.write_text(json.dumps(source, indent=1) + "\n")
    under = [r for r in source["rows"] if r.get("under_bar")]
    print(f"{path.name}: {len(source['rows'])} rows annotated, {len(under)} under the SOL bar")


def merge(dirs, into):
    """Union the per-job tables (and provenance) of each (op, device class) in ``dirs``."""
    stems = {}
    for directory in dirs:
        for path in sorted(Path(directory).glob("*@*.provenance.json")):
            stems.setdefault(path.name.removesuffix(".provenance.json"), []).append(path)
    for stem, paths in stems.items():
        heads, rows, provenance, reports = {}, {}, [], []
        for path in paths:
            table = json.loads(path.with_name(f"{stem}.json").read_text())
            head = {k: v for k, v in table.items() if k != "rows"}
            heads[json.dumps(head, sort_keys=True)] = head
            for bucket, values in table["rows"]:
                key = json.dumps(bucket)
                if rows.get(key, values) != values:
                    raise SystemExit(f"{stem}: jobs disagree on bucket {bucket}")
                rows[key] = values
            source = json.loads(path.read_text())
            provenance += source["rows"]
            reports.append(source["sensitivity"])
        if len(heads) != 1:
            raise SystemExit(f"{stem}: jobs used different op versions or spaces")
        (head,) = heads.values()
        rows_out = [[json.loads(key), values] for key, values in rows.items()]
        Path(into, f"{stem}.json").write_text(dumps(head, rows_out))
        source.update(rows=provenance, sensitivity=harness.summarize(reports))
        Path(into, f"{stem}.provenance.json").write_text(json.dumps(source, indent=1) + "\n")
        print(f"{stem}: {len(rows)} rows from {len(paths)} jobs")


def import_rows(args):
    """Write a table from external paired measurements (JSON lines, see harness.imported)."""
    module, _, name = args.op.partition(":")
    op = getattr(importlib.import_module(module), name)
    major, minor, sms = map(int, args.device_class.split(","))
    hardware = DeviceClass(major, minor, sms)
    measured = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line]
    rows, provenance = harness.imported(op, hardware, measured)
    stem = Path(args.out) / f"{op.name}@{hardware.name}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.json").write_text(encode(op, rows))
    info = {"device_class": hardware.name, "device": args.device, "source": args.source}
    Path(f"{stem}.provenance.json").write_text(harness.provenance(op, info, provenance, {}))
    print(f"{stem}: {len(rows)} rows of {len(provenance)} buckets")


def report(dirs):
    """Per dimension, across every provenance file in ``dirs``."""
    for directory in dirs:
        for path in sorted(Path(directory).glob("*@*.provenance.json")):
            source = json.loads(path.read_text())
            summary = source["sensitivity"]
            if any("speedups" in entry for entry in summary.values()):
                summary = harness.summarize([summary])
            print(f"## {path.name.removesuffix('.provenance.json')} ({source['device']})")
            fractions = sorted(r["sol_fraction"] for r in source["rows"] if "sol_fraction" in r)
            if fractions:
                under = sum(r.get("under_bar", False) for r in source["rows"])
                print(
                    f"SOL fraction: median {statistics.median(fractions):.3f}, "
                    f"min {fractions[0]:.3f}; {under} of {len(fractions)} rows under the bar"
                )
            print("| dims | adjacent pairs, other best | median speedup | max speedup | >= keep |")
            print("|---|---|---|---|---|")
            for name, e in summary.items():
                print(
                    f"| {name} | {e['pairs']} | {e['median_speedup']:.3f} "
                    f"| {e['max_speedup']:.3f} | {e['over_keep']} |"
                )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m benchmark.kernel_tuning.tune")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("adapter")
    runner.add_argument("--out", required=True)
    runner.add_argument("--timer", choices=tuple(timing.TIMERS), default="graph")
    runner.add_argument("--screen-samples", type=int, default=5)
    runner.add_argument("--samples", type=int, default=10)
    runner.add_argument("--calls", type=int, default=20)
    runner.add_argument("--rounds", type=int, default=7)
    runner.add_argument("--cores", default="auto")
    runner.add_argument("--shard", default="0/1", help="i/n: this process's share of groups")
    runner.add_argument(
        "--screen-only", action="store_true", help="search only (raw.jsonl), no table or A-B-A"
    )
    merger = commands.add_parser("merge")
    merger.add_argument("dirs", nargs="+")
    merger.add_argument("--into", required=True)
    importer = commands.add_parser("import", help="table from external paired measurements")
    importer.add_argument("op", help="module:attribute of the TunedOp")
    importer.add_argument("rows", help="JSON lines: {shape, config, speedup vs default}")
    importer.add_argument("--device-class", required=True, help="major,minor,SMs")
    importer.add_argument("--device", required=True, help="device name for provenance")
    importer.add_argument("--source", required=True, help="how the rows were measured")
    importer.add_argument("--out", required=True)
    reporter = commands.add_parser("report")
    reporter.add_argument("dirs", nargs="+")
    annotator = commands.add_parser("annotate", help="add us / SOL columns to a provenance file")
    annotator.add_argument("adapter")
    annotator.add_argument("--table", required=True, help="directory of the provenance file")
    annotator.add_argument("--timer", choices=tuple(timing.TIMERS), default="graph")
    annotator.add_argument("--samples", type=int, default=10)
    annotator.add_argument("--calls", type=int, default=20)
    known, rest = parser.parse_known_args(argv)
    if known.command == "annotate":
        adapter = importlib.import_module(f"benchmark.kernel_tuning.tune.{known.adapter}")
        annotate(known, adapter)
    elif known.command == "run":
        adapter = importlib.import_module(f"benchmark.kernel_tuning.tune.{known.adapter}")
        adapter.add_arguments(runner)
        args = parser.parse_args(argv)
        paired.pin(paired.parse_cores(args.cores))
        run(args, adapter)
    elif rest:
        parser.error(f"unknown arguments {rest}")
    elif known.command == "merge":
        merge(known.dirs, known.into)
    elif known.command == "import":
        import_rows(known)
    else:
        report(known.dirs)


if __name__ == "__main__":
    sys.exit(main())
