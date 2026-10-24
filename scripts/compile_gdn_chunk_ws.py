#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Compile CP1, fused CP2, or CP fixup offline; no device queries, launches or timing."""

import argparse
import contextlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".worklog/gdn-chunk-ws"))
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--geoms", default="16x32,16x48", help="comma-separated HKxHV geometries")
    parser.add_argument("--fixup-only", action="store_true", help="compile fixup for Hv=32,48")
    parser.add_argument("--cp-variants", default="full", help="fused variants: full,mn,seg")
    parser.add_argument(
        "--producer-registers",
        default="56,72",
        help="comma-separated WG0 budgets for --fused",
    )
    parser.add_argument("--prefetch", default="2,4", help="comma-separated z-load batch sizes")
    parser.add_argument("--remap-y", action="store_true")
    parser.add_argument("--q-lookahead", action="store_true")
    parser.add_argument("--iket", action="store_true")
    parser.add_argument("--watchdog", action="store_true", help="bounded fused debug waits")
    parser.add_argument("--pdl", choices=("on", "off", "both"), default="on")
    args = parser.parse_args()
    try:
        args.geoms = [tuple(map(int, item.split("x"))) for item in args.geoms.split(",")]
        if not args.geoms or any(
            g not in ((16, 32), (16, 48), (8, 16), (8, 24)) for g in args.geoms
        ):
            raise ValueError("unsupported geometry")
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_CACHE_DIR"] = str(args.output.resolve() / "cache")
    if args.fixup_only or args.fused:
        output = args.output
        modes = (True, False) if args.pdl == "both" else (args.pdl == "on",)
        for pdl in modes:
            args.use_pdl = pdl
            args.output = output / ("pdl-on" if pdl else "pdl-off")
            args.output.mkdir(parents=True, exist_ok=True)
            if args.fixup_only:
                compile_fixups(args.output, pdl)
            else:
                compile_fused(args)
        return
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.runtime import make_fake_stream

    from ctikernels.gdn.chunk_ws._kernel import read_nsmid
    from ctikernels.gdn.chunk_ws._launch import compile_kernel

    reader_dir = args.output.resolve() / "nsmid"
    reader_dir.mkdir(parents=True, exist_ok=True)
    reader = cute.compile(
        read_nsmid,
        cute.runtime.make_fake_tensor(Int32, (1,), (1,), assumed_align=4),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch sm_120a --keep-cubin --dump-dir={reader_dir}",
    )
    assert reader.__cubin__.startswith(b"\x7fELF")
    print("compiled sm_120a nsmid reader", flush=True)

    rows = []

    def cuda_tool(name):
        local = Path(sys.executable).parent / name
        return str(local) if local.is_file() else shutil.which(name)

    cuobjdump = cuda_tool("cuobjdump")
    for hk, hv in args.geoms:
        directory = args.output.resolve() / f"{hk}x{hv}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "ptxas.log").open("w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                compiled = compile_kernel(
                    hk,
                    hv,
                    f"--enable-tvm-ffi --gpu-arch sm_120a --keep-ptx --keep-cubin --dump-dir={directory} remarks{{ptx}}",
                )
        assert compiled.__cubin__.startswith(b"\x7fELF")
        ptx_paths = list(directory.glob("*.ptx"))
        assert len(ptx_paths) == 1, ptx_paths
        ptx = re.sub(r"\s+", " ", ptx_paths[0].read_text())
        for instruction in (
            "mma.sync",
            "ldmatrix.sync",
            "stmatrix.sync",
            "cp.async.bulk.tensor",
            "setmaxnreg.dec.sync.aligned.u32 24",
            "setmaxnreg.inc.sync.aligned.u32 240",
            "bar.sync",
        ):
            assert instruction in ptx, instruction
        assert "tensormap.replace" in ptx
        # Use CuTe's embedded ptxas diagnostics: the external CUDA toolkit may
        # be older than the PTX emitted by the DSL's bundled NVVM compiler.
        log = (directory / "ptxas.log").read_text()
        actual = directory / "actual.cubin"
        actual.write_bytes(compiled.__cubin__)
        resource = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(actual)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (directory / "resources.log").write_text(resource)
        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(actual)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (directory / "kernel.sass").write_text(sass)
        sass_ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][\w.]*)\s*", sass)
        reg = re.search(r"Used (\d+) registers", log)
        spills = re.search(
            r"(\d+) bytes stack frame,\s+(\d+) bytes spill stores,\s+(\d+) bytes spill\s+loads",
            log,
        )
        actual_reg = re.search(r"REG:(\d+)", resource)
        assert reg and spills and actual_reg, "missing assembler resource statistics"
        row = {
            "hk": hk,
            "hv": hv,
            "ptxas_registers": int(reg[1]),
            "cubin_registers": int(actual_reg[1]),
            "stack_bytes": int(spills[1]),
            "spill_store_bytes": int(spills[2]),
            "spill_load_bytes": int(spills[3]),
            # Static instruction sites across first/middle/final chunk bodies.
            "ptx_mma_sites": ptx.count("mma.sync.aligned."),
            "ptx_branch_sites": len(re.findall(r"\bbra\b", ptx)),
            "sass_instructions": len(sass_ops),
            "sass_hmma": sum(op.startswith("HMMA.") for op in sass_ops),
            "sass_ldsm": sum(op.startswith("LDSM.") for op in sass_ops),
        }
        rows.append(row)
        (args.output / "resources.json").write_text(json.dumps(rows, indent=2) + "\n")
        print("COMPILE_ROW " + json.dumps(row), flush=True)


def compile_fused(args):
    from ctikernels.gdn._contract import Geometry
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, compile_prefill

    tool = Path(sys.executable).parent / "cuobjdump"
    cuobjdump = str(tool) if tool.is_file() else shutil.which("cuobjdump")
    rows = []
    for budget, prefetch in itertools.product(
        map(int, args.producer_registers.split(",")), map(int, args.prefetch.split(","))
    ):
        opts = PortOptions(
            budget,
            args.iket,
            prefetch,
            args.watchdog,
            pdl=args.use_pdl,
            q_lookahead=args.q_lookahead,
            remap_y=args.remap_y,
        )
        for (hk, hv), cp_variant in itertools.product(args.geoms, args.cp_variants.split(",")):
            producer_budget, math_budget = opts.registers(cp_variant)
            directory = (
                args.output.resolve() / f"pf{prefetch}-p{budget}" / f"{hk}x{hv}" / cp_variant
            )
            directory.mkdir(parents=True, exist_ok=True)
            flags = (
                "--enable-tvm-ffi --gpu-arch sm_120a --keep-ptx --keep-cubin "
                f"--dump-dir={directory} remarks{{ptx}}"
            )
            with (directory / "ptxas.log").open("w") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    binaries = compile_prefill(
                        hk, hv, Geometry(hk, hv).proj_width, flags, opts, cp_variant
                    )
            ptx = re.sub(
                r"\s+", " ", " ".join(path.read_text() for path in directory.glob("*.ptx"))
            )
            assert ("griddepcontrol.launch_dependents;" in ptx) == (
                args.use_pdl and cp_variant == "mn"
            )
            assert ("griddepcontrol.wait;" in ptx) == (args.use_pdl and cp_variant == "seg")
            for instruction in (
                "mma.sync",
                "ldmatrix.sync",
                "bar.sync",
                f"setmaxnreg.dec.sync.aligned.u32 {producer_budget}",
                f"setmaxnreg.inc.sync.aligned.u32 {math_budget}",
            ):
                assert instruction in ptx, instruction
            # The FP16 inverse and its BF16 operand use STMatrix in every variant.
            assert "stmatrix.sync" in ptx
            assert re.search(r"cp\.async\.bulk\.tensor\.2d\.shared::(?:cta|cluster)\.global", ptx)
            assert "mbarrier.arrive.expect_tx" in ptx
            assert "fence.proxy.async.shared::cta" in ptx
            assert 128 * producer_budget + 256 * math_budget <= 168 * 384
            producer_ptx = ptx[ptx.index("setmaxnreg.dec") : ptx.index("setmaxnreg.inc")]
            for instruction in (
                "bar.sync 11, 128;",
                "ld.global.v2.b32",
                "ld.shared.v2.b32",
                "st.shared.v2.b32",
                "cvt.rn.bf16x2.f32",
                "shfl.sync.bfly.b32",
                "ex2.approx.f32",
            ):
                assert instruction in producer_ptx, instruction
            raw_tma_sites = len(
                re.findall(
                    r"cp\.async\.bulk\.tensor\.2d\.shared::(?:cta|cluster)\.global",
                    producer_ptx,
                )
            )
            # K priming + K lookahead + Q + V, each split into two SW128
            # 64-column boxes by the same CuTe layout used in CP1. Q
            # lookahead splits Q into prime/steady static sites (not extra
            # dynamic transfers). MN omits both Q sites.
            assert raw_tma_sites == (6 if cp_variant == "mn" else 10 if args.q_lookahead else 8), (
                raw_tma_sites
            )
            assert producer_ptx.count("mbarrier.arrive.expect_tx") == (
                3 if cp_variant == "mn" else 5 if args.q_lookahead else 4
            )
            if cp_variant == "mn":
                assert "cp.async.bulk.tensor.2d.global.shared" not in ptx
                assert "tensormap.replace" not in ptx
            if args.watchdog:
                assert "vprintf" in ptx and "trap;" in ptx
                formats = re.findall(r"printfFormat_\d+\[(\d+)\] = \{([^}]+)\}", ptx)
                assert any(
                    bytes(map(int, fmt.split(","))).startswith(b"GDN_WATCHDOG tag=%d")
                    and bytes(map(int, fmt.split(","))).ljust(int(size), b"\0").endswith(b"\n\0")
                    for size, fmt in formats
                )
                assert "%globaltimer" in ptx and "2000000000" in ptx
                assert "LAB_WAIT" not in ptx, "unbounded mbarrier wait remains"
                assert not re.search(r"bar\.sync\s+[45],", ptx), "ordered math wait remains"
                assert "st.volatile.shared.u32" in ptx
                assert "ld.volatile.shared.u32" in ptx
            else:
                assert "vprintf" not in ptx
                if not args.iket:
                    assert "%globaltimer" not in ptx
                assert "st.volatile.shared.u32" not in ptx
            # Each raw barrier arrival must reconverge the issuing warp
            # before CuTe's internal election for the first of two copies.
            raw_issues = producer_ptx.split("mbarrier.arrive.expect_tx")[1:]
            for issue in raw_issues:
                before_copy = issue.split("cp.async.bulk.tensor", 1)[0]
                assert "bar.warp.sync" in before_copy
                assert "elect.sync" in before_copy.split("bar.warp.sync", 1)[1]
            assert not re.search(r"bar\.sync\s+12\b", ptx), "math norm barrier remains"
            for name, binary in zip(("chunk", "history"), binaries, strict=True):
                if name == "chunk":
                    assert binary.cp2_smem_bytes <= 101376
                cubin = directory / f"{name}.cubin"
                cubin.write_bytes(binary.__cubin__)
                resource = subprocess.run(
                    [cuobjdump, "--dump-resource-usage", str(cubin)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                (directory / f"{name}.resources.log").write_text(resource)
                reg = re.search(r"REG:(\d+)", resource)
                local = re.search(r"LOCAL:(\d+)", resource)
                sass = subprocess.run(
                    [cuobjdump, "--dump-sass", str(cubin)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                (directory / f"{name}.sass").write_text(sass)
                # Resource data is per kernel. Do not label it as per-WG spill counts.
                row = {
                    "cp_variant": cp_variant,
                    "hk": hk,
                    "hv": hv,
                    "kernel": name,
                    "producer_budget": producer_budget,
                    "requested_producer_budget": budget,
                    "prefetch": prefetch,
                    "math_budget": math_budget,
                    "registers": int(reg[1]) if reg else None,
                    "local_bytes": int(local[1]) if local else None,
                    "local_load_instructions": len(re.findall(r"\bLDL\b", sass)),
                    "local_store_instructions": len(re.findall(r"\bSTL\b", sass)),
                    "dynamic_smem_bytes": getattr(binary, "cp2_smem_bytes", 0),
                    "iket": args.iket,
                    "watchdog": args.watchdog,
                    "pdl": args.use_pdl,
                    "q_lookahead": args.q_lookahead,
                    "effective_q_lookahead": args.q_lookahead and cp_variant != "mn",
                    "remap_y": args.remap_y,
                }
                log = (directory / "ptxas.log").read_text()
                frames = re.findall(
                    r"(\d+) bytes stack frame,\s+(\d+) bytes spill stores,\s+(\d+) bytes spill\s+loads",
                    log,
                )
                frame = frames[0 if name == "chunk" else -1]
                row.update(
                    stack_bytes=int(frame[0]),
                    spill_store_bytes=int(frame[1]),
                    spill_load_bytes=int(frame[2]),
                )
                if name == "chunk":
                    row["raw_tma_instruction_sites"] = raw_tma_sites
                    # Static SASS layout, not dynamic attribution: the role branch
                    # precedes DEALLOC, math starts at TRY_ALLOC, common tail follows.
                    begin = sass.index("USETMAXREG.DEALLOC")
                    end = sass.index("USETMAXREG.TRY_ALLOC")
                    for region, body in (
                        ("producer_region", sass[begin:end]),
                        ("math_and_tail", sass[end:]),
                        ("prelude", sass[:begin]),
                    ):
                        row[region] = {
                            "LDL": len(re.findall(r"\bLDL\b", body)),
                            "STL": len(re.findall(r"\bSTL\b", body)),
                        }
                if args.iket and name == "chunk":
                    import iket

                    context = iket.Context(binary.__cubin__)
                    assert context.is_instrumented(), "DSL stripped IKET events"
                    row["iket_ranges"] = sorted(r.range_name for r in context.get_all_ranges())
                    expected = {
                        "producer.tma_issue",
                        "producer.wait_raw_k",
                        "producer.wait_raw_q",
                        "producer.wait_raw_v",
                        "producer.k",
                        "producer.q",
                        "producer.v",
                        "producer.gates",
                        "producer.y",
                        "producer.wait_k",
                        "producer.wait_q",
                        "producer.wait_v",
                        "producer.wait_gates",
                        "producer.wait_o",
                        "producer.wait_y",
                        "math.O2_norm",
                    }
                    if cp_variant == "mn":
                        expected -= {
                            "producer.wait_raw_q",
                            "producer.q",
                            "producer.y",
                            "producer.wait_q",
                            "producer.wait_o",
                            "producer.wait_y",
                            "math.O2_norm",
                        }
                    assert expected <= set(row["iket_ranges"])
                rows.append(row)
                print("COMPILE_ROW " + json.dumps(row), flush=True)
            (args.output / "resources.json").write_text(json.dumps(rows, indent=2) + "\n")

    if any(v != "full" for v in args.cp_variants.split(",")):
        compile_fixups(args.output, args.use_pdl)


def compile_fixups(output, pdl=False):
    from ctikernels.gdn.chunk_ws._cp_fixup import (
        PREFETCH,
        ROWS,
        SLAB,
        SMEM_BYTES,
        THREADS,
        compile_fixup,
    )

    tool = Path(sys.executable).parent / "cuobjdump"
    cuobjdump = str(tool) if tool.is_file() else shutil.which("cuobjdump")
    rows = []
    for hv in (32, 48):
        directory = output.resolve() / "fixup" / f"16x{hv}"
        directory.mkdir(parents=True, exist_ok=True)
        flags = (
            "--enable-tvm-ffi --gpu-arch sm_120a --keep-ptx --keep-cubin "
            f"--dump-dir={directory} remarks{{ptx}}"
        )
        with (directory / "ptxas.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                binary = compile_fixup(hv, flags, pdl)
        assert binary.__cubin__.startswith(b"\x7fELF")
        cubin = directory / "fixup.cubin"
        cubin.write_bytes(binary.__cubin__)
        ptx = " ".join(path.read_text() for path in directory.glob("*.ptx"))
        for instruction in (
            "fma.rn.f32",
            "cp.async.cg.shared.global",
            "cp.async.commit_group",
            "cp.async.wait_group",
            "ld.shared.v4.f32",
            "st.shared.v4.f32",
            "st.global.v4.f32",
            "bar.sync",
        ):
            assert instruction in ptx, instruction
        assert "fma.rn.ftz" not in ptx and "mma.sync" not in ptx and "tf32" not in ptx
        assert re.search(rf"cp\.async\.wait_group\s+{PREFETCH - 1};", ptx)
        assert ("griddepcontrol.wait;" in ptx) == pdl
        assert ("griddepcontrol.launch_dependents;" in ptx) == pdl
        resource = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (directory / "resources.log").write_text(resource)
        sass = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (directory / "kernel.sass").write_text(sass)
        log = (directory / "ptxas.log").read_text()
        reg = re.search(r"Used (\d+) registers", log)
        actual_reg = re.search(r"REG:(\d+)", resource)
        static_smem = re.search(r"SHARED:(\d+)", resource)
        spills = re.search(
            r"(\d+) bytes stack frame,\s+(\d+) bytes spill stores,\s+(\d+) bytes spill\s+loads",
            log,
        )
        assert reg and actual_reg and static_smem and spills, (
            "missing fixup assembler resource statistics"
        )
        assert int(static_smem[1]) + SMEM_BYTES <= 50 * 1024
        row = {
            "kernel": "fixup",
            "pdl": pdl,
            "hv": hv,
            "rows": ROWS,
            "threads": THREADS,
            "slab": SLAB,
            "dynamic_smem_bytes": SMEM_BYTES,
            "static_smem_bytes": int(static_smem[1]),
            "ptxas_registers": int(reg[1]),
            "cubin_registers": int(actual_reg[1]),
            "stack_bytes": int(spills[1]),
            "spill_store_bytes": int(spills[2]),
            "spill_load_bytes": int(spills[3]),
            "local_load_instructions": len(re.findall(r"\bLDL\b", sass)),
            "local_store_instructions": len(re.findall(r"\bSTL\b", sass)),
        }
        rows.append(row)
        print("COMPILE_ROW " + json.dumps(row), flush=True)
    (output / "fixup" / "resources.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
