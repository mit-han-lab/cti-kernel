# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline slot-layout audit on sm_120a and sm_121a, without a CUDA device.

Set CUDA_VISIBLE_DEVICES= and put the CUDA toolkit binaries on PATH. Artifacts
are written only to the required --out directory, which must be outside the repo.
"""

import argparse
import itertools
import json
import os
import re
import subprocess
from pathlib import Path

from benchmark.kernel_tuning.moe_cute_compile import artifact
from ctikernels.moe.cute._routing import compile_layout


def cases(cap):
    # Cover all constexpr feature combinations, zero local experts, and both
    # sides of the small/multi-CTA dispatch boundary.
    for e, k, bm in ((256, 8, 16), (64, 8, 64), (7, 3, 32), (0, 8, 128)):
        for shared, mapped, small in itertools.product((False, True), repeat=3):
            yield (
                f"layout-{e}-{k}-{bm}-{shared}-{mapped}-{small}",
                compile_layout,
                (e, k, shared, mapped, bm, small, cap),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= for offline compilation")
    root = Path(__file__).resolve().parents[2]
    if args.out.resolve().is_relative_to(root):
        parser.error("--out must be outside the source tree")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.out.resolve())
    with (args.out / "results.jsonl").open("w") as report:
        for cap in ((12, 0), (12, 1)):
            for label, compile_fn, key in cases(cap):
                name = f"sm{cap[0]}{cap[1]}-{label}"
                fn = compile_fn(*key)
                cubin = args.out / f"{name}.cubin"
                artifact(fn, "cubin", cubin)
                artifact(fn, "ptx", args.out / f"{name}.ptx")
                usage = subprocess.check_output(
                    ["cuobjdump", "--dump-resource-usage", str(cubin)], text=True
                )
                registers = [int(v) for v in re.findall(r"REG:(\d+)", usage)]
                local = [int(v) for v in re.findall(r"LOCAL:(\d+)", usage)]
                stack = [int(v) for v in re.findall(r"STACK:(\d+)", usage)]
                if not registers or not local or not stack:
                    raise RuntimeError(f"Missing resource report: {name}")
                row = {
                    "name": name,
                    "registers": max(registers),
                    "local_bytes": max(local),
                    "stack_bytes": max(stack),
                }
                line = json.dumps(row)
                print(line, flush=True)
                report.write(line + "\n")
                report.flush()
                if max(local) or max(stack):
                    raise RuntimeError(f"Unexpected local memory: {name}")


if __name__ == "__main__":
    main()
