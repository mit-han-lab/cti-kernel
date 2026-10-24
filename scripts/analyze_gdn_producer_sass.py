#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Count uninstrumented CP2 producer SASS from compile_gdn_chunk_ws.py artifacts.

For the old scalar loops, count one iteration, including its backedge. For
straight-line transforms, count the entire tile body and divide by sixteen.
Includes predicated instruction sites, not measured executed instructions.
The dependency analysis is a register RAW graph (unit edge cost, not cycles),
with region inputs as roots; predicates and 64/128-bit register operands count.
It does not model warp scheduling, scoreboard stalls, or memory latency.
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

INTEGER = {
    "IADD",
    "IADD3",
    "IMAD",
    "IMNMX",
    "LEA",
    "LOP3",
    "SHF",
    "PRMT",
    "BMSK",
    "UIADD3",
    "UIMAD",
    "ULEA",
    "ULOP3",
    "USHF",
    "UBMSK",
}
REGISTER = re.compile(r"\b(?:UR|R|UP|P)\d+\b")


@dataclass
class Instruction:
    address: int
    op: str
    args: str
    predicate: str

    @property
    def kind(self):
        return self.op.split(".")[0]


def read_producer(path):
    instructions = [
        Instruction(int(m[1], 16), m[3], m[4].strip(), m[2].strip())
        for m in re.finditer(
            r"/\*([0-9a-f]+)\*/\s+((?:@!?\w+\s+)?)([A-Z][\w.]*)\s*([^;]*);",
            path.read_text(),
        )
    ]
    start = next(i for i, x in enumerate(instructions) if x.op.startswith("USETMAXREG.DEALLOC"))
    end = next(i for i, x in enumerate(instructions) if x.op.startswith("USETMAXREG.TRY_ALLOC"))
    return instructions[start:end]


def row_body(body):
    loops = [
        (i, int(x.args.split(",")[-1].strip(), 16))
        for i, x in enumerate(body)
        if x.kind == "BRA" and int(x.args.split(",")[-1].strip(), 16) < x.address
    ]
    if loops:
        assert len(loops) == 1, "ambiguous transform loop"
        end, start = loops[0]
        iteration = [x for x in body[: end + 1] if x.address >= start]
        # Scalar legacy rows store once; remapped y handles four row-equivalents.
        return iteration, sum(x.kind == "STS" for x in iteration)
    return body, 16


def transform_regions(instructions):
    bars = [i for i, x in enumerate(instructions) if x.op.startswith("BAR.SYNC")]
    transforms = []
    output = []
    for start in bars:
        end = next(
            (
                j
                for j in range(start + 1, len(instructions))
                if instructions[j].op.startswith(("SYNCS.ARRIVE", "BAR.SYNC"))
            ),
            len(instructions),
        )
        body = instructions[start + 1 : end]
        if sum(x.kind == "FFMA" for x in body) >= 16:
            transforms.append(row_body(body))
    # K prime / [Q prime or Q current] / V / [Q lookahead] / K steady.
    # MN omits Q, and zero-V stores do not contain convolution FFMAs.
    names = {
        3: ("k_prime", "v", "k"),
        4: ("k_prime", "q", "v", "k"),
        5: ("k_prime", "q_prime", "v", "q", "k"),
    }
    assert len(transforms) in names, "could not identify producer transforms"
    for end in bars:
        start = max((i for i in bars if i < end), default=0)
        arrivals = [i for i in range(start, end) if instructions[i].op.startswith("SYNCS.ARRIVE")]
        if not arrivals:
            continue
        waits = [
            i
            for i in range(arrivals[-1] + 1, end)
            if instructions[i].op.startswith("SYNCS.PHASECHK")
        ]
        if not waits:
            continue
        body = instructions[waits[-1] + 2 : end]
        if sum(x.kind == "STS" for x in body) in (4, 16):
            output.append(("y", *row_body(body)))
    assert len(output) == (len(transforms) != 3), "could not identify output norm"
    return [
        (name, *region) for name, region in zip(names[len(transforms)], transforms, strict=True)
    ] + output


def registers(operand, width=1):
    result = set(REGISTER.findall(operand))
    if width > 1 and re.fullmatch(r"(?:UR|R)\d+(?:\.reuse)?", operand):
        prefix, index = re.match(r"(UR|R)(\d+)", operand).groups()
        result.update(f"{prefix}{int(index) + j}" for j in range(width))
    return result


def dataflow(body):
    definitions = {}
    parents = []
    paths = []
    stores = []
    address_nodes = set()
    for i, ins in enumerate(body):
        args = [x.strip() for x in ins.args.split(",")]
        width = 4 if ".128" in ins.op else 2 if ".64" in ins.op else 1
        count = 1
        if ins.kind in {"SHFL", "ISETP", "UISETP", "FSETP", "PLOP3"}:
            count = 2
        elif ins.kind in {"IADD3", "UIADD3"}:
            count = 3
        elif ins.kind in {"STS", "STL", "BRA", "BAR", "MEMBAR", "FENCE", "NOP"}:
            count = 0
        dest = set().union(*(registers(a, width) for a in args[:count]))
        uses = set().union(*(registers(a, width) for a in args[count:]))
        uses.update(registers(ins.predicate))
        # Count RAW edges on the issuing path, not false WAW dependencies
        # from a predicated overwrite to the old destination. Read-modify-write
        # fixups already list their old destination as an explicit source.
        predecessors = {definitions[r] for r in uses if r in definitions}
        parents.append(predecessors)
        longest = max((paths[j] for j in predecessors), key=len, default=[])
        paths.append([*longest, i])
        if ins.kind in {"LDS", "STS"}:
            address_arg = next(a for a in args if "[" in a)
            todo = [definitions[r] for r in registers(address_arg) if r in definitions]
            while todo:
                node = todo.pop()
                if node not in address_nodes:
                    address_nodes.add(node)
                    todo.extend(parents[node])
        if ins.kind == "STS":
            stores.append(paths[-1])
        for r in dest:
            definitions[r] = i
    longest = max(stores, key=len)
    return {
        "address_integer_sites": sum(body[i].kind in INTEGER for i in address_nodes),
        "store_raw_depths": [len(p) for p in stores],
        "longest_store_raw_chain": [f"0x{body[i].address:x} {body[i].op}" for i in longest],
    }


def analyze(path):
    result = {}
    for name, body, rows in transform_regions(read_producer(path)):
        counts = Counter(x.kind for x in body)
        flow = dataflow(body)
        result[name] = {
            "begin": f"0x{body[0].address:x}",
            "end_inclusive": f"0x{body[-1].address:x}",
            "rows": rows,
            "instructions_per_row": len(body) / rows,
            # A row-equivalent is four elements per lane, including remapped y.
            "instructions_per_element": len(body) / (rows * 4),
            "classes_per_element": {k: v / (rows * 4) for k, v in sorted(counts.items())},
            "classes_per_row": {k: v / rows for k, v in sorted(counts.items())},
            "integer_per_row": sum(v for k, v in counts.items() if k in INTEGER) / rows,
            "address_integer_per_row": flow.pop("address_integer_sites") / rows,
            "shared_opcodes": dict(
                sorted(Counter(x.op for x in body if x.kind in {"LDS", "STS"}).items())
            ),
            "spills": [
                f"0x{x.address:x} {x.op} {x.args}" for x in body if x.kind in {"LDL", "STL"}
            ],
            **flow,
        }
        assert counts["STS"] == rows and counts["LDS"] == rows, (name, counts)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.sass), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
