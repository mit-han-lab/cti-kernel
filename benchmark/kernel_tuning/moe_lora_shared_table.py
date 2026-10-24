# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Generate shared dispatch choices from projection probe JSON-lines files."""

import argparse
import json
import math
import re
from pathlib import Path
from pprint import pformat

from ctikernels.moe.cute._lora_shared_table import device_class

BEGIN = "# BEGIN GENERATED SHARED TABLE"
END = "# END GENERATED SHARED TABLE"
POLICY = Path(__file__).resolve().parents[2] / "python/ctikernels/moe/cute/_lora_shared_table.py"


def candidate_choice(name):
    if name in ("cute", "split", "library"):
        return name, None
    if isinstance(name, str) and re.fullmatch(r"d\d+\.\d+\.\d+\.\d+\.\d+", name):
        bm, sk, ebm, bn, stages = map(int, name[1:].split("."))
        if (
            bm in (16, 32, 64)
            and sk > 0
            and ebm in (16, 32, 64)
            and bn in (64, 128)
            and stages in (2, 3, 4)
        ):
            return "dense", (bm, sk, ebm, bn, stages)
    return None


def build_table(lines):
    """Ignore non-probe lines; take the fastest successful measurement at each T."""
    winners = {}
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(record, dict)
            or not {"capability", "sms", "T", "r", "proj", "K", "N", "times"} <= record.keys()
        ):
            continue
        device = device_class(record["capability"], record["sms"])
        if device is None or record["proj"] not in ("w13", "w2"):
            continue
        shape = (2 if record["proj"] == "w13" else 1, record["K"], record["N"], record["r"])
        key = device, shape, record["T"]
        for name, time in record["times"].items():
            choice = candidate_choice(name)
            if (
                choice is None
                or type(time) not in (int, float)
                or not math.isfinite(time)
                or time <= 0
            ):
                continue
            # Names break ties deterministically, independent of file order.
            value = time, name, choice
            if key not in winners or value[:2] < winners[key][:2]:
                winners[key] = value
    table = {}
    for (device, shape, rows), (_, _, (mode, config)) in sorted(winners.items()):
        entries = table.setdefault(device, {}).setdefault(shape, [])
        if entries and entries[-1][1:] == (mode, config):
            entries[-1] = (rows, mode, config)
        else:
            entries.append((rows, mode, config))
    for shapes in table.values():
        for shape, entries in shapes.items():
            entries[-1] = (None, *entries[-1][1:])
            shapes[shape] = tuple(entries)
    return table


def table_literal(table):
    return "SHARED_TABLE = " + pformat(table, width=100, sort_dicts=True) + "\n"


def write_table(path, literal):
    source = path.read_text()
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("policy must contain exactly one pair of shared table markers")
    before, rest = source.split(BEGIN)
    _, after = rest.split(END)
    path.write_text(before + BEGIN + "\n" + literal + END + after)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="replace the generated policy block")
    parser.add_argument("--policy", type=Path, default=POLICY, help="policy file to update")
    args = parser.parse_args(argv)
    lines = (line for path in args.files for line in path.read_text().splitlines())
    literal = table_literal(build_table(lines))
    if args.write:
        write_table(args.policy, literal)
    else:
        print(literal, end="")


if __name__ == "__main__":
    main()
