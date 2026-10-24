# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline token backward audit for sm_120a/sm_121a (no CUDA device required).

Checks vector memory instructions and rejects spills/local memory. Artifacts
must stay outside the source tree. Set CUDA_VISIBLE_DEVICES= and put cuobjdump
on PATH, then pass --out out/moe-token-compile.
"""

import argparse
import itertools
import json
import os
import re
import subprocess
from pathlib import Path

from benchmark.kernel_tuning.moe_cute_compile import artifact
from ctikernels.moe.cute._token import compile_token


def cases():
    for k, shared, fused, masked, splits in itertools.product(
        (2, 8), (False, True), (False, True), (False, True), (1, 8)
    ):
        yield 2048, 256, k, 16, shared, fused, masked, not masked, splits
    # Narrow H exercises partial warps; H=320 exercises a partial final strip.
    for h, k in itertools.product((64, 192, 320), range(1, 9)):
        yield h, 256, k, 3, True, False, True, True, 1
    for nt in (1, 8, 24, 64):
        for fused in (False, True):
            yield 2048, 256, 8, nt, True, fused, True, False, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= for offline compilation")
    if args.out.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("--out must be outside the source tree")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.out.resolve())
    with (args.out / "results.jsonl").open("w") as report:
        for cap in ((12, 0), (12, 1)):
            for key in cases():
                name = f"sm{cap[0]}{cap[1]}-token-" + "-".join(str(int(v)) for v in key)
                fn = compile_token(*key, cap)
                cubin, ptx_file = args.out / f"{name}.cubin", args.out / f"{name}.ptx"
                artifact(fn, "cubin", cubin)
                artifact(fn, "ptx", ptx_file)
                ptx = ptx_file.read_text()
                # BF16 vectors are four b32 registers; FP32 dacc uses two
                # four-float vectors for each eight-element output fragment.
                for instruction in ("ld.global.v4", "st.global.v4"):
                    if instruction not in ptx:
                        raise RuntimeError(
                            f"Missing 128-bit memory operation {instruction}: {name}"
                        )
                usage = subprocess.check_output(
                    ["cuobjdump", "--dump-resource-usage", str(cubin)], text=True
                )
                resources = {}
                for field in ("REG", "LOCAL", "STACK"):
                    values = [int(v) for v in re.findall(rf"{field}:(\d+)", usage)]
                    if not values:
                        raise RuntimeError(f"Missing resource report: {name}")
                    resources[field.lower()] = max(values)
                sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
                local_ops = len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))
                row = {"name": name, **resources, "local_ops": local_ops}
                line = json.dumps(row)
                print(line, flush=True)
                report.write(line + "\n")
                report.flush()
                if resources["local"] or resources["stack"] or local_ops:
                    raise RuntimeError(f"Unexpected local memory: {name}")


if __name__ == "__main__":
    main()
