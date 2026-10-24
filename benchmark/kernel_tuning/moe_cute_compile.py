# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline qualification of dispatch and GPU-test configurations on both targets.

Run with CUDA_VISIBLE_DEVICES= and PYTHONPATH=python:. Pass --out out/moe-compile.
Artifacts stay outside the source tree because compiler debug paths are local.
Requires cuobjdump on PATH; reports registers, local memory and shared-memory
eligibility for two CTAs. Any local memory in a dispatched kernel fails the audit.
CTI_MOE_CUTE_PDL=0 audits the baseline without PDL instructions or early prefetch.
"""

import argparse
import ast
import json
import os
import re
import subprocess
from pathlib import Path

from ctikernels.moe.cute._provider import DISPATCH, KINDS, compile_kernel, parse_config

TMA_CANDIDATES = {
    "gate_up": ("16x32x128s3w4g1x4t", "16x64x64s4w4g1x4t", "64x64x64s3w4g2x2nt"),
    "down": ("16x32x64s6w4g1x4t", "16x64x128s3w4g1x4t", "64x128x64s2w4g2x2nt"),
    "dx_pair": ("16x128x64s4w4g1x4t", "64x128x32s4w4g2x2nt"),
    "bwd_slot": ("16x128x64s4w4g1x4t", "64x64x32s4w4g2x2nt"),
}


def test_specs():
    source = Path(__file__).resolve().parents[2] / "tests/gpu/test_moe_cute_gemm.py"
    names = ("DEFAULT_CONFIGS", "EXHAUSTIVE_CONFIGS", "SINGLE_B_CONFIGS")
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }


def cases():
    result = {}

    def add(kind, spec, dispatched=False):
        cfg = parse_config(spec)
        for saved in (False, True) if kind in ("gate_up", "bwd_slot") else (False,):
            key = kind, cfg, saved
            result[key] = result.get(key, False) or dispatched

    # Compile every table entry for BOTH targets, regardless of table ownership. The
    # bwd_slot entry serves saved backward only; recompute runs with gate_up's entry.
    for table in DISPATCH.values():
        for kind, spec in table.items():
            if kind == "bwd_slot":
                result[kind, parse_config(spec), True] = True
                result[kind, parse_config(table["gate_up"]), False] = True
            else:
                add(kind, spec, True)
    specs = test_specs()
    for spec in specs["DEFAULT_CONFIGS"] + specs["EXHAUSTIVE_CONFIGS"]:
        for kind in KINDS:
            add(kind, spec)
    for spec in specs["SINGLE_B_CONFIGS"]:
        for kind in ("down", "dx_pair"):
            add(kind, spec)
    # Extra configurations used by the short-K and independent-saved tests.
    for kind in KINDS:
        add(kind, "32x64x64s4w4")
        add(kind, "32x64x64s4w4t")
    for kind, candidates in TMA_CANDIDATES.items():
        for spec in candidates:
            if kind == "bwd_slot":
                result[kind, parse_config(spec), True] = False
            else:
                add(kind, spec)
    add("bwd_slot", "16x32x32s6w4g1x4n")
    return result


def artifact(fn, ext, path):
    data = getattr(fn, f"__{ext}__")
    if isinstance(data, str) and not data.lstrip().startswith(("//", ".version")):
        data = Path(data.strip()).read_bytes()
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=("120", "121", "both"), default="both")
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--tma-only", action="store_true", help="Audit TMA tests and candidates")
    parser.add_argument(
        "--geometry", choices=("test", "ep1", "ep2", "ep4", "routed"), default="test"
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES= for offline compilation")
    args.out.mkdir(parents=True, exist_ok=True)
    pdl = os.environ.get("CTI_MOE_CUTE_PDL", "1") != "0"
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.out.resolve())
    targets = ((12, 0), (12, 1)) if args.arch == "both" else ((12, int(args.arch[-1])),)
    h, ni, nis, e, kp1 = {
        "test": (256, 128, 128, 8, 9),
        "ep1": (2048, 512, 512, 256, 9),
        "ep2": (2048, 512, 256, 128, 9),
        "ep4": (2048, 512, 128, 64, 9),
        "routed": (2048, 768, 0, 128, 8),
    }[args.geometry]
    with (args.out / "results.jsonl").open("w") as report:
        for cap in targets:
            for (kind, cfg, saved), dispatched in cases().items():
                if args.dispatch_only and not dispatched:
                    continue
                if args.tma_only and not cfg.tma:
                    continue
                name = f"sm{cap[0]}{cap[1]}-{kind}-{cfg}-saved{int(saved)}-pdl{int(pdl)}"
                fn = compile_kernel(
                    kind,
                    cfg,
                    h,
                    ni,
                    nis,
                    e,
                    kp1,
                    saved,
                    bool(nis),
                    cap,
                    pdl=pdl,
                )
                cubin = args.out / f"{name}.cubin"
                artifact(fn, "cubin", cubin)
                artifact(fn, "ptx", args.out / f"{name}.ptx")
                ptx = (args.out / f"{name}.ptx").read_text()
                if ("cp.async.bulk.tensor" in ptx) != cfg.tma:
                    raise RuntimeError(f"Incorrect TMA instruction presence: {name}")
                if cfg.tma:
                    for instruction in ("mbarrier.arrive.expect_tx", "mbarrier.try_wait"):
                        if instruction not in ptx:
                            raise RuntimeError(f"Missing {instruction}: {name}")
                    if pdl and kind in ("down", "dx_pair"):
                        if ptx.index("cp.async.bulk.tensor") > ptx.index("griddepcontrol.wait"):
                            raise RuntimeError(f"Missing TMA prefetch before PDL wait: {name}")
                for instruction in ("griddepcontrol.wait", "griddepcontrol.launch_dependents"):
                    if (instruction in ptx) != pdl:
                        raise RuntimeError(f"Incorrect PDL instruction presence: {name}")
                usage = subprocess.check_output(
                    ["cuobjdump", "--dump-resource-usage", str(cubin)], text=True
                )
                registers = [int(v) for v in re.findall(r"REG:(\d+)", usage)]
                local = [int(v) for v in re.findall(r"LOCAL:(\d+)", usage)]
                stack = [int(v) for v in re.findall(r"STACK:(\d+)", usage)]
                if not registers or not local or not stack:
                    raise RuntimeError(f"Missing resource report for {name}")
                sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
                local_ops = len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", sass))
                row = {
                    "name": name,
                    "geometry": args.geometry,
                    "dispatched": dispatched,
                    "pdl": pdl,
                    "registers": max(registers),
                    "local_bytes": max(local),
                    "stack_bytes": max(stack),
                    "local_ops": local_ops,
                    "shared_bytes": cfg.shared_bytes(kind, saved),
                    "two_ctas_smem": cfg.two_ctas_smem(kind, saved),
                }
                line = json.dumps(row)
                print(line, flush=True)
                report.write(line + "\n")
                report.flush()
                if dispatched and (max(local) or max(stack) or local_ops):
                    raise RuntimeError(f"Dispatched kernel has local memory: {name}")


if __name__ == "__main__":
    main()
