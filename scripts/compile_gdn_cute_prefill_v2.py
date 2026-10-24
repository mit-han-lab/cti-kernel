#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""V2BUDGET compile probe; fake tensors, explicit SM targets, no device queries.

Example: --arch 120,121 --geometry 16x32,16x48,8x16,8x24 --vsplit 1,2,4
         --output out/gdn-v2-budget

One JSON line per variant. Registers, local/stack and static shared bytes come
from the v1 cubin resource extractor. Spill loads/stores count static LDL/STL
instructions in the actual cubin (not dynamic traffic or ptxas spill-byte totals).
No intentionally local arrays exist in this kernel. PTX and cubins are retained.
This script compiles kernels; run it on a machine with the CUDA toolchain.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def local_instructions(disassembly):
    """Count local-memory instructions by function, keeping units explicit."""
    records = {}
    parts = re.split(r"Function\s*:\s*(\S+)", disassembly)
    for name, body in zip(parts[1::2], parts[2::2], strict=True):
        if "_cutlass_fused_v2_" in name:
            records[name] = {
                "spill_stores": len(re.findall(r"\bSTL(?:\.[A-Z0-9]+)*\s", body)),
                "spill_loads": len(re.findall(r"\bLDL(?:\.[A-Z0-9]+)*\s", body)),
                "spill_units": "static local-memory instruction count",
            }
    return records


def spill_info(directory):
    executable = shutil.which("cuobjdump")
    if executable is None and Path("/usr/local/cuda/bin/cuobjdump").is_file():
        executable = "/usr/local/cuda/bin/cuobjdump"
    if executable is None:
        raise RuntimeError("V2BUDGET requires cuobjdump")
    records = {}
    for binary in directory.rglob("*.cubin"):
        result = subprocess.run(
            [executable, "--dump-sass", str(binary)], capture_output=True, text=True, check=True
        )
        records.update(local_instructions(result.stdout))
    if len(records) != 1:
        raise RuntimeError("expected exactly one fused_v2 symbol for spill inspection")
    return next(iter(records.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="120,121")
    parser.add_argument("--geometry", default="16x32,16x48,8x16,8x24")
    parser.add_argument("--vsplit", default="1,2,4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dsl-root", type=Path)
    parser.add_argument("--extra", default="", help="extra tune items, e.g. v2_snorm=1")
    args = parser.parse_args()
    arches = args.arch.split(",")
    try:
        geometries = [tuple(map(int, g.split("x"))) for g in args.geometry.split(",")]
        splits = [int(v) for v in args.vsplit.split(",")]
    except ValueError:
        parser.error("expected comma-separated HkxHv geometries and integer vsplit values")
    if any(a not in ("120", "121") for a in arches):
        parser.error("--arch must select 120 and/or 121")
    if any(g not in ((16, 32), (16, 48), (8, 16), (8, 24)) for g in geometries):
        parser.error("unsupported phase-1 geometry")
    if any(v not in (1, 2, 4) for v in splits):
        parser.error("--vsplit must select 1, 2 and/or 4")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUTE_DSL_CACHE_DIR"] = str(args.output.resolve() / "cache")
    if args.dsl_root is not None:
        sys.path[:] = [p for p in sys.path if "nvidia_cutlass_dsl" not in p]
        sys.path.insert(0, str(args.dsl_root.resolve()))
        import nvidia_cutlass_dsl

        root = Path(nvidia_cutlass_dsl.__path__[0])
        packages = next(
            root / p for p in ("dsl_packages", "python_packages") if (root / p).is_dir()
        )
        sys.path.insert(0, str(packages))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from ctikernels.gdn import _cute_prefill

    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        parser.error("V2BUDGET requires nvidia-cutlass-dsl==4.8.0")
    for arch in arches:
        for hk, hv in geometries:
            for vsplit in splits:
                tune = _cute_prefill.read_tune(
                    ",".join(filter(None, (f"v2=1,vsplit={vsplit}", args.extra))),
                    (12, int(arch[-1])),
                    (hk, hv),
                )
                directory = args.output.resolve() / f"sm{arch}_{hk}x{hv}_vsplit{vsplit}"
                directory.mkdir(parents=True, exist_ok=True)
                launcher = object.__new__(_cute_prefill.Launcher)
                launcher.tune = tune
                launcher.options = (
                    f"--enable-tvm-ffi --gpu-arch sm_{arch}a --keep-ptx --keep-cubin "
                    f"--dump-dir={directory}"
                )
                width = (2 * hk + 2 * hv) * 128 + 2 * hv
                key = (hk, hv, "serial", False, width, tuple(sorted(tune.items())), 0)
                compiled = launcher._compile(key)
                assert compiled.__cubin__.startswith(b"\x7fELF")
                resources = _cute_prefill.compile_info(directory, hk, hv, 64, 1, tune)
                fused = next(r for r in resources if r["kernel"] == "fused_v2")
                fused.update(spill_info(directory))
                fused["static_dynamic_smem_bytes"] = (
                    fused["static_smem_bytes"] + fused["dynamic_smem_bytes"]
                )
                ptx = "\n".join(p.read_text() for p in directory.rglob("*.ptx"))
                assert all(op in ptx for op in ("mma.sync", "ldmatrix.sync", "cp.async"))
                assert ".f64" not in ptx and "griddepcontrol" not in ptx
                record = {
                    "check": "V2BUDGET",
                    "arch": arch,
                    "geometry": f"{hk}x{hv}",
                    "vsplit": vsplit,
                    "resources": resources,
                    "shared_budget_ok": fused["static_dynamic_smem_bytes"] <= 99 * 1024,
                    "register_budget_ok": fused["register_ctas_per_sm"] >= 1,
                    "smem_layout": _cute_prefill.v2_smem_layout(hk, hv, vsplit),
                }
                (directory / "resources.json").write_text(json.dumps(record, indent=2) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
