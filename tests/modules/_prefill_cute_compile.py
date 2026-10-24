# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline CuTe 4.8 and external ptxas receipts; CUDA remains uninitialized.

CUDA_VISIBLE_DEVICES='' python -m tests.modules._prefill_cute_compile --output out/prefill
Use --ptx-version 9.0 only for an external CUDA 13.0 assembler. This changes
the version directive in a separate PTX copy, never the CuTe-produced binary.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CASES = {
    f"d{dim}_m{rows}": (dim, rows, 32 if dim == 256 else 64, 2, 1, "bfloat16", 16, 4)
    for dim in (64, 128, 256)
    for rows in (64, 128)
} | {
    "d128_m64_negative": (128, 64, 32, 2, 4, "bfloat16", 7, 3),
    "d128_m64_zero": (128, 64, 32, 2, 4, "bfloat16", 7, 3),
    "d128_m64_split4": (128, 64, 64, 2, 4, "bfloat16", 16, 4),
    "d128_m128_split2": (128, 128, 64, 2, 2, "bfloat16", 16, 4),
    "d256_m64_split8": (256, 64, 32, 2, 8, "bfloat16", 16, 4),
    "d256_m128_split15": (256, 128, 32, 2, 15, "bfloat16", 16, 4),
    "d256_m128_split16": (256, 128, 32, 2, 16, "bfloat16", 16, 4),
    "d256_m128_split4": (256, 128, 32, 2, 4, "bfloat16", 16, 4),
    "d128_m64_s3": (128, 64, 64, 3, 3, "float16", 1, 4),
    "d256_m64_s7": (256, 64, 32, 3, 7, "bfloat16", 7, 9),
    "d64_m128_g33": (64, 128, 32, 3, 1, "float16", 3, 33),
    "d128_m128_n32": (128, 128, 32, 3, 5, "float16", 7, 3),
    "d256_m128_fp16": (256, 128, 32, 2, 7, "float16", 1, 4),
    "d256_m64_n64": (256, 64, 64, 2, 1, "bfloat16", 16, 4),
    "d256_m64_n64_split3": (256, 64, 64, 2, 3, "bfloat16", 16, 4),
    "d256_m64_n64_split11": (256, 64, 64, 2, 11, "bfloat16", 16, 4),
    "d256_m64_n64_fp16": (256, 64, 64, 2, 7, "float16", 16, 4),
    "d256_m64_n64_page7": (256, 64, 64, 2, 4, "bfloat16", 7, 3),
    "d256_m64_n64_page1": (256, 64, 64, 2, 4, "bfloat16", 1, 4),
    "d256_m64_n64_negative": (256, 64, 64, 2, 4, "bfloat16", 7, 3),
    "d256_m64_n64_zero": (256, 64, 64, 2, 4, "bfloat16", 7, 3),
    "d256_gb10_b1q64": (256, 64, 64, 2, 3, "bfloat16", 16, 4),
    "d256_gb10_b3q64": (256, 64, 64, 2, 1, "bfloat16", 16, 4),
    "d256_gb10_b2q256": (256, 64, 64, 2, 1, "bfloat16", 16, 4),
    "d256_gb10_b4q256": (256, 64, 64, 2, 1, "bfloat16", 16, 4),
}
BOUNDS = {
    "d256_gb10_b1q64": (1, 64),
    "d256_gb10_b3q64": (3, 64),
    "d256_gb10_b2q256": (2, 256),
    "d256_gb10_b4q256": (4, 256),
}


def tile_instructions(sass, dim, rows, keys):
    """Static SASS issue counts along the full-tile control-flow paths.

    Collapse constant-trip gather loops, then choose paths with the fewest
    FSELs (the unmasked softmax specialization). Report shortest/longest such
    paths: rescaling and the page-copy warp predicate can vary. Predicated
    instructions count as issued regardless of active lanes. These are static
    paths, not Nsight dynamic instruction counts or a latency model.
    """
    instructions = {}
    for address, text in re.findall(r"/\*([0-9a-f]+)\*/\s+([^;]+);", sass):
        words = text.split()
        predicate = words[0].startswith("@")
        opcode = words[int(predicate)].split(".")[0]
        target = None
        conditional = predicate
        if opcode == "BRA":
            target = int(re.search(r"0x([0-9a-f]+)", text)[1], 16)
            conditional |= bool(re.search(r"!?UP[0-9]+,", text))
        instructions[int(address, 16)] = (opcode, target, conditional)
    loops = [
        (target, address)
        for address, (_, target, _) in instructions.items()
        if target is not None and target < address
    ]
    # The KV backedge is the widest natural loop containing tensor work.
    tensor_loops = [
        (start, end)
        for start, end in loops
        if any(instructions[a][0] == "HMMA" for a in range(start, end + 16, 16))
    ]
    start, end = max(tensor_loops, key=lambda pair: pair[1] - pair[0])
    nested = {a: b for a, b in loops if start <= a and b < end}

    def path(first, last, longest):
        # Nested loops are disjoint in this kernel; compute their cost first.
        collapsed = {}
        for a, b in nested.items():
            if first <= a and b < last:
                body = path(a, b, longest)
                if body["HMMA"]:
                    qk_hmma = (dim // 16) * (keys // 8)
                    assert qk_hmma % body["HMMA"] == 0, body
                    trips = qk_hmma // body["HMMA"]
                else:
                    assert body["LDGSTS"], (a, b, body)
                    copies = keys * dim // (rows * 16)
                    assert copies % body["LDGSTS"] == 0, body
                    trips = copies // body["LDGSTS"]
                collapsed[a] = (b, Counter({k: v * trips for k, v in body.items()}))
        costs = {last + 16: Counter()}
        for address in range(last, first - 16, -16):
            if any(a < address <= b for a, (b, _) in collapsed.items()):
                continue
            if address in collapsed:
                b, cost = collapsed[address]
                costs[address] = cost + costs[b + 16]
                continue
            opcode, target, conditional = instructions[address]
            successors = [address + 16]
            if opcode == "BRA" and address != last:
                assert target > address, (address, target)
                successors = [target] + ([address + 16] if conditional else [])
            candidates = [costs[a] for a in successors]
            chosen = min(candidates, key=lambda c: (c["FSEL"], (-1 if longest else 1) * c.total()))
            costs[address] = Counter({opcode: 1}) + chosen
        return costs[first]

    low, high = path(start, end, False), path(start, end, True)
    expected_hmma = 2 * (dim // 16) * (keys // 8)
    assert low["HMMA"] == high["HMMA"] == expected_hmma, (low, high)
    # Runtime zero-scale handling may select factor=1 once per score row.
    # These two scalar selects are independent of the number of keys; the
    # diagonal mask's per-element selects must still be absent on this path.
    assert low["FSEL"] == high["FSEL"] and low["FSEL"] <= 2, (low, high)
    # Two rescale tests plus a uniform zero comparison, unless it is hoisted
    # out of the loop or uses a uniform-register predicate.
    assert low["FSETP"] <= 3 and high["FSETP"] <= 3, (low, high)
    assert low["LDG"] == high["LDG"] == 0, (low, high)
    return {
        "kv_loop_start": hex(start),
        "kv_loop_backedge": hex(end),
        "full_tile_min": dict(sorted(low.items())),
        "full_tile_max": dict(sorted(high.items())),
        "full_tile_total": [low.total(), high.total()],
        "hmma_per_tile": expected_hmma,
        "hmma_fraction": [expected_hmma / high.total(), expected_hmma / low.total()],
    }


def occupancy(registers, shared, rows):
    # Register allocation rounds each warp to 256 registers. The conservative
    # 100 KiB carveout also covers devices reporting less than 128 KiB/SM.
    warps = rows // 16
    registers_per_cta = ((registers * 32 + 255) // 256 * 256) * warps
    shared = (shared + 255) // 256 * 256
    blocks = min(65536 // registers_per_cta, (100 * 1024) // shared, 48 // warps, 32)
    return {
        "ctas_per_sm": blocks,
        "warps_per_sm": blocks * warps,
        "warp_occupancy": blocks * warps / 48,
        "resident_ctas": {str(sms): blocks * sms for sms in (170, 188, 48)},
        "assumed_shared_carveout": 100 * 1024,
    }


def worker(args):
    import cutlass
    import torch

    from ctikernels.attention._prefill_cute import _compile, _shared_bytes

    assert cutlass.__version__ == "4.8.0"
    dim, rows, keys, stages, splits, dtype, page, group = CASES[args.case]
    batch, qmax = BOUNDS.get(args.case, (3, 256))
    min_chunk = 256 if args.case in BOUNDS else 64
    _compile(
        batch,
        qmax,
        4,
        group,
        dim,
        page,
        getattr(torch, dtype),
        args.arch,
        rows,
        keys,
        stages,
        splits,
        min_chunk,
    )
    assert not torch.cuda.is_initialized()
    (ptx_file,) = args.output.glob("*Prefill*.ptx")
    (cubin,) = args.output.glob("*Prefill*.cubin")
    ptx = ptx_file.read_text()
    for instruction in (
        "cp.async.cg.shared.global",
        "cp.async.ca.shared.global",
        "ex2.approx.ftz.f32",
        "fma.rn.f32",
        "ldmatrix.sync.aligned",
        "mma.sync.aligned.m16n8k16",
        "griddepcontrol.wait",
        "griddepcontrol.launch_dependents",
        "st.global.v4",
    ):
        assert instruction in ptx, instruction
    # Sign handling is confined to Q staging, before the key loop's MMAs.
    assert ptx.count("// PREFILL_Q_SIGN_BEGIN") == ptx.count("// PREFILL_Q_SIGN_END") == 1
    sign_end = ptx.index("// PREFILL_Q_SIGN_END")
    assert sign_end < ptx.index("mma.sync.aligned")
    if splits > 1:
        assert "atom.acq_rel.gpu.global.add" in ptx
        assert "ld.global.cg.v4.f32" in ptx
    resources = subprocess.run(
        [args.cuobjdump, "--dump-resource-usage", str(cubin)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (args.output / "cute-resources.txt").write_text(resources)
    usage = {
        key: int(value)
        for key, value in re.findall(r"(REG|STACK|SHARED|LOCAL):([0-9]+)", resources)
    }
    external = ptx_file
    if args.ptx_version:
        external = args.output / "external.ptx"
        external.write_text(re.sub(r"(?m)^\.version [0-9.]+", f".version {args.ptx_version}", ptx))
    result = subprocess.run(
        [
            args.ptxas,
            f"-arch={args.arch}",
            "-v",
            str(external),
            "-o",
            str(args.output / "external.cubin"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (args.output / "ptxas.txt").write_text(result.stderr)
    stack, stores, loads = map(
        int,
        re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            result.stderr,
        ).groups(),
    )
    sass = subprocess.run(
        [args.cuobjdump, "--dump-sass", str(cubin)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (args.output / "native.sass").write_text(sass)
    receipt = {
        "arch": args.arch,
        "case": args.case,
        "geometry": CASES[args.case],
        "batch": batch,
        "qmax": qmax,
        "min_chunk": min_chunk,
        "cute": usage,
        "external_ptxas": {
            "registers": int(re.search(r"Used (\d+) registers", result.stderr)[1]),
            "stack": stack,
            "spill_stores": stores,
            "spill_loads": loads,
        },
        "shared_per_cta": _shared_bytes(dim, rows, keys, stages) + 1024,
    }
    receipt["occupancy"] = occupancy(usage["REG"], receipt["shared_per_cta"], rows)
    receipt["instructions"] = tile_instructions(sass, dim, rows, keys)
    assert receipt["shared_per_cta"] <= 99 * 1024
    assert usage["REG"] <= 255
    assert usage["STACK"] == usage["LOCAL"] == stack == stores == loads == 0, receipt
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", choices=CASES, nargs="+", help="Subset for the matrix runner")
    parser.add_argument("--arch", choices=("sm_120a", "sm_121a"))
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--ptx-version")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        worker(args)
        return
    receipts = []
    for arch in (args.arch,) if args.arch else ("sm_120a", "sm_121a"):
        for case in args.cases or CASES:
            target = args.output / arch / case
            target.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "tests.modules._prefill_cute_compile",
                "--output",
                str(target),
                "--case",
                case,
                "--arch",
                arch,
                "--ptxas",
                args.ptxas,
                "--cuobjdump",
                args.cuobjdump,
            ]
            if args.ptx_version:
                command += ["--ptx-version", args.ptx_version]
            env = os.environ | {
                "CUTE_DSL_ARCH": arch,
                "CUTE_DSL_KEEP": "ptx,cubin",
                "CUTE_DSL_DUMP_DIR": str(target),
                "CUTE_DSL_DISABLE_FILE_CACHING": "1",
            }
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            (target / "compile.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{arch}/{case} failed; see {target / 'compile.log'}")
            receipt = json.loads((target / "receipt.json").read_text())
            receipts.append(receipt)
            print(arch, case, receipt["cute"], receipt["external_ptxas"], flush=True)
    (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    with (args.output / "resources.csv").open("w") as file:
        file.write("# SPDX-License-Identifier: MIT\n")
        writer = csv.writer(file)
        writer.writerow(
            [
                "arch",
                "case",
                "cute_registers",
                "cute_stack_bytes",
                "cute_local_bytes",
                "ptxas_registers",
                "ptxas_stack_bytes",
                "ptxas_spill_store_bytes",
                "ptxas_spill_load_bytes",
                "shared_per_cta_bytes",
            ]
        )
        for receipt in receipts:
            native, external = receipt["cute"], receipt["external_ptxas"]
            writer.writerow(
                [
                    receipt["arch"],
                    receipt["case"],
                    native["REG"],
                    native["STACK"],
                    native["LOCAL"],
                    external["registers"],
                    external["stack"],
                    external["spill_stores"],
                    external["spill_loads"],
                    receipt["shared_per_cta"],
                ]
            )


if __name__ == "__main__":
    main()
