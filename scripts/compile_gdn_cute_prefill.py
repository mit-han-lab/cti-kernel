#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Compile GDN CuTe kernels without a GPU, using fake tensors and explicit targets.

Example (use a CPU torch Python):
  python scripts/compile_gdn_cute_prefill.py --dsl-root /path/to/site-packages \
      --arch 120,121 --tune ';split_h=1;recurrence_warps=8;form=wy;bv=64'

No CUDA device queries, kernel launches, remote runners, or third-party imports beyond
CuTe itself. PTX/cubins are written under --output for offline resource inspection.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from itertools import product
from pathlib import Path


def output_matrix_tunes(base_tunes):
    """Expand output forms before resolving per-geometry defaults."""
    return [
        ",".join(filter(None, (base, variant, f"overlap={overlap}")))
        for base in base_tunes
        for variant in (
            "o_form=pre,qp_split=0",
            "o_form=pre,qp_split=1",
            "o_form=legacy,aq_qk=0,aq_out=0",
            "o_form=legacy,aq_qk=0,aq_out=1",
            # QK-only is an ablation at Hv=32, whose default is now coarse.
            "o_form=legacy,aq_qk=1,aq_out=0",
            "o_form=legacy,aq_precise=1",
        )
        for overlap in (0, 1)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsl-root", type=Path)
    parser.add_argument("--arch", default="120,121")
    parser.add_argument("--tune", default="", help="semicolon-separated configurations")
    parser.add_argument(
        "--overlap-matrix",
        action="store_true",
        help="compile overlap/prep/recurrence combinations per base tune",
    )
    parser.add_argument(
        "--output-matrix",
        action="store_true",
        help="compile pre qp_split=0/1 and all legacy aq_qk/aq_out pairs, serial and overlap",
    )
    parser.add_argument("--captures", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--geometry", default="16x48")
    parser.add_argument(
        "--sms", type=int, help="host SM count for automatic prefetch depth (120:170, 121:48)"
    )
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--stride-pad", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/gdn-cute-compile"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUTE_DSL_CACHE_DIR"] = str(args.output / "cache")
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
    from ctikernels.gdn import _cute_prefill, _cute_prefill_kernels

    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        parser.error("this compile check requires DSL 4.8.0")
    hk, hv = map(int, args.geometry.split("x"))
    if (
        hk < 1
        or hv % hk
        or args.tokens < args.sequences
        or args.sequences < 1
        or args.stride_pad < 0
    ):
        parser.error("invalid geometry, token count, sequence count, or stride padding")
    width = (2 * hk + 2 * hv) * 128 + 2 * hv
    for arch in args.arch.split(","):
        if arch not in ("120", "121"):
            parser.error("--arch must select 120 and/or 121")
        tune_strings = args.tune.split(";")
        if args.output_matrix:
            # Resolve geometry defaults after selecting precision, just like runtime.
            tune_strings = output_matrix_tunes(tune_strings)
        configs = [
            _cute_prefill.read_tune(config, (12, int(arch[-1])), (hk, hv))
            for config in tune_strings
        ]
        if args.overlap_matrix:
            configs = [
                dict(
                    base,
                    **dict(
                        zip(
                            ("overlap", "prep_mode", "rec_mode"),
                            values,
                            strict=True,
                        )
                    ),
                )
                for base in configs
                for values in product((0, 1), (0, 1), (0, 1, 2))
            ]
        for index, base_tune in enumerate(configs):
            tune = base_tune.copy()
            if args.tokens < tune["overlap_min_tokens"]:
                tune["overlap"] = 0
            sms = args.sms if args.sms is not None else (170 if arch == "120" else 48)
            one_wave = args.sequences * hv * (128 // tune["bv"]) <= sms
            tune = _cute_prefill.resolve_tune(tune, hv, one_wave)
            directory = args.output.resolve() / f"sm{arch}_{index}"
            directory.mkdir(parents=True, exist_ok=True)
            launcher = object.__new__(_cute_prefill.Launcher)
            launcher.kernels = _cute_prefill_kernels
            launcher.tune = tune
            launcher.options = (
                f"--enable-tvm-ffi --gpu-arch sm_{arch}a --keep-ptx --keep-cubin "
                f"--dump-dir={directory}"
            )
            key = (
                hk,
                hv,
                _cute_prefill.pipeline_variant(args.tokens, tune),
                one_wave,
                width + args.stride_pad,
                tuple(sorted(tune.items())),
                args.captures,
            )
            (directory / "tune.json").write_text(json.dumps(tune, indent=2) + "\n")
            compiled = launcher._compile(key)
            if isinstance(compiled, dict):
                assert set(compiled) == {"pipeline", "norm"}
                has_key = any(
                    "_cutlass_prepare_key_" in symbol for symbol in compiled["pipeline"].kernel_info
                )
                assert has_key == (not (tune["merge_prepare"] or tune["prep_mode"]))
            ptx = "\n".join(p.read_text() for p in directory.rglob("*.ptx"))
            instructions = ["cp.async", "st.global.v4"]
            if tune["stop_after"] not in ("conv", "norm"):
                instructions += ["mma.sync", "ldmatrix.sync"]
            if tune["solve_mma"] == 1 and tune["stop_after"] in ("solve", "wy", "all"):
                instructions += ["cvt.rna.tf32.f32", ".f32.tf32.tf32.f32"]
            for instruction in instructions:
                assert instruction in ptx, instruction
            if tune["overlap"] and tune["stop_after"] == "all":
                for instruction in ("st.release.gpu.global", "ld.acquire.gpu.global", "nanosleep"):
                    assert instruction in ptx, instruction
            else:
                assert "ld.acquire.gpu.global" not in ptx, "serial path contains a device wait"
            for part in ptx.split(".visible .entry"):
                name = next(
                    (
                        n
                        for n in ("prepare_key", "prepare_value", "recurrence", "output", "norm")
                        if part.lstrip().startswith(f"kernel_cutlass_{n}_")
                    ),
                    None,
                )
                if name is None:
                    continue
                overlap = bool(tune["overlap"] and tune["stop_after"] == "all")
                trigger = overlap and name.startswith("prepare_")
                wait = overlap and (
                    name == "recurrence"
                    or (
                        name == "prepare_value" and not (tune["merge_prepare"] or tune["prep_mode"])
                    )
                )
                assert ("griddepcontrol.launch_dependents" in part) == trigger, name
                assert ("griddepcontrol.wait" in part) == wait, name
            if (tune["merge_prepare"] or tune["prep_mode"]) and tune["stop_after"] in (
                "solve",
                "wy",
                "all",
            ):
                assert "kernel_cutlass_prepare_key_" not in ptx, (
                    "single-prepare path launched key prepare"
                )
            assert ".f64" not in ptx, "unexpected FP64 instruction"
            assert "shared::cluster" not in ptx and "barrier.cluster" not in ptx
            if (
                tune["form"] == "tk"
                and not (tune["merge_prepare"] or tune["prep_mode"])
                and tune["solve_mma"] != 2
                and tune["stop_after"] in ("solve", "wy", "all")
            ):
                value_ptx = next(
                    part
                    for part in ptx.split(".visible .entry")
                    if part.lstrip().startswith("kernel_cutlass_prepare_value_")
                )
                assert ".f32.bf16.bf16.f32" not in value_ptx, "unexpected WY product in TK prepare"
            if tune["stop_after"] == "all" and tune["rec_mode"] == 2 and not tune["rec_ws"]:
                recurrence_ptx = next(
                    part
                    for part in ptx.split(".visible .entry")
                    if part.lstrip().startswith("kernel_cutlass_recurrence_")
                )
                waits = re.findall(r"cp\.async\.wait_group\s+(\d+)", recurrence_ptx)
                if tune["prefetch_depth"] == 4:
                    assert waits == ["0"], "full pipeline must have one loop-entry wait"
                else:
                    assert waits.count("1") == (1 if tune["o_split"] else 2), (
                        "compact lookahead was drained too soon"
                    )
                assert "st.release.gpu.global" not in recurrence_ptx, (
                    "norm completion flags returned"
                )
            if tune["stop_after"] == "all":
                assert "kernel_cutlass_commit_history_" not in ptx, (
                    "separate pipeline history commit returned"
                )
                rec_ptx = next(
                    part
                    for part in ptx.split(".visible .entry")
                    if part.lstrip().startswith("kernel_cutlass_recurrence_")
                )
                assert "st.release.gpu.global" not in rec_ptx
                if tune["rec_ws"]:
                    assert re.search(r"bar\.sync\s+1,\s*256", rec_ptx)
                    assert re.search(r"bar\.sync\s+3,\s*128", rec_ptx)
                    if tune["update_split"]:
                        assert re.search(r"bar\.sync\s+2,\s*128", rec_ptx)
            resources = _cute_prefill.compile_info(
                directory, hk, hv, args.tokens, args.sequences, tune
            )
            executables = compiled.values() if isinstance(compiled, dict) else [compiled]
            # Offline artifacts retain resource inspection without accessing a CUDA library.
            for executable in executables:
                assert isinstance(executable.__cubin__, bytes)
                assert executable.__cubin__.startswith(b"\x7fELF")
            symbols = [symbol for executable in executables for symbol in executable.kernel_info]
            assert all(
                any(f"_cutlass_{record['kernel']}_" in sym for sym in symbols)
                for record in resources
            ), "compiled kernel symbols unavailable for offline inspection"
            # Prepare never spins on flags; split value prepare waits only on key.
            for part in ptx.split(".visible .entry"):
                if part.lstrip().startswith(
                    ("kernel_cutlass_prepare_key_", "kernel_cutlass_prepare_value_")
                ):
                    assert "ld.acquire.gpu.global" not in part, "prepare kernel must never spin"
            (directory / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")
            print(json.dumps({"resources": resources}), flush=True)
            print(f"compiled sm_{arch}a {tune}: {directory}", flush=True)


if __name__ == "__main__":
    main()
