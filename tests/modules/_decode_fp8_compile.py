# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline CuTe 4.8 + PTX resource receipt, without initializing CUDA.

Run with the DSL installed and CUDA_VISIBLE_DEVICES='':
  python -m tests.modules._decode_fp8_compile --output out/fp8-decode

Every case runs in a fresh interpreter so its dump directory is unambiguous.
Optional --ptx-version=9.0 changes ONLY the PTX version directive in a separate
copy for CUDA 13.0 ptxas, when CuTe's bundled compiler emits PTX 9.4. Both original
CuTe cubin resources and external ptxas resources are recorded, not conflated.
Stage counts are static PTX instructions per warp (all predicated paths once),
including the next-stage gather, math, ring advance and warp synchronization;
they exclude prologue/epilogue and the loop backedge outside the marker pair.
Native SASS counts cover the smallest backward-branch interval containing one
stage's HMMAs, including the backedge and all static full/tail paths. These are
instruction-footprint counts, not dynamic execution counts or GPU timings.
The harness also checks page-bank/ring bounds at chunk and refill boundaries.
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CASES = {
    # B, Hkv, G, D, page, Q dtype, splits, min_chunk, warps, keys, stages, merge
    "qwen3": (1, 8, 4, 128, 16, "bfloat16", 21, 64, 4, 16, 3, "distributed"),
    "qwen35": (1, 4, 4, 256, 16, "bfloat16", 42, 64, 4, 16, 3, "distributed"),
    "engine128": (4, 8, 4, 128, 16, "bfloat16", 5, 256, 4, 16, 3, "distributed"),
    "engine_tp2": (4, 2, 6, 256, 16, "bfloat16", 21, 256, 4, 16, 3, "distributed"),
    "engine256": (4, 4, 4, 256, 16, "bfloat16", 10, 256, 4, 16, 3, "distributed"),
    "page1": (1, 2, 4, 64, 1, "float16", 1, 1, 4, 32, 2, "distributed"),
    "tp2": (1, 2, 4, 256, 16, "bfloat16", 85, 64, 4, 16, 3, "distributed"),
    "last128": (1, 8, 4, 128, 16, "bfloat16", 21, 64, 4, 16, 3, "last"),
    "last256": (1, 4, 4, 256, 16, "bfloat16", 42, 64, 4, 16, 3, "last"),
    "single64": (2, 2, 6, 64, 7, "float16", 1, 64, 4, 16, 3, "distributed"),
    "single128": (1, 8, 4, 128, 16, "float16", 1, 64, 4, 16, 3, "distributed"),
    "g16d256": (3, 4, 16, 256, 64, "bfloat16", 4, 64, 4, 16, 3, "distributed"),
    "g9d128": (2, 2, 9, 128, 7, "float16", 7, 1, 4, 16, 3, "distributed"),
    "tiny_splits": (2, 2, 3, 128, 7, "float16", 7, 1, 4, 16, 3, "distributed"),
    "keys32": (2, 2, 6, 64, 7, "bfloat16", 4, 64, 2, 32, 2, "distributed"),
    "warp1": (2, 2, 6, 64, 7, "bfloat16", 4, 64, 1, 16, 4, "distributed"),
    "warps8": (2, 2, 6, 64, 7, "bfloat16", 4, 64, 8, 16, 2, "distributed"),
}


def instruction_counts(body):
    instructions = re.findall(r"^\s*(?:@!?%?\w+\s+)?([a-z][a-z0-9_.]*)(?=\s|;)", body, re.M)
    counts = dict(sorted(collections.Counter(instructions).items()))
    return {"total": len(instructions), "opcodes": counts}


def stage_counts(ptx):
    assert ptx.count("// FP8_STAGE_BEGIN") == ptx.count("// FP8_STAGE_END")
    variants = []
    for region in ptx.split("// FP8_STAGE_BEGIN")[1:]:
        body = region.split("// FP8_STAGE_END")[0]
        variants.append(stage_body_counts(body))
    assert len(variants) in (1, 2)
    return variants[0] | {"variants": variants}


def stage_body_counts(body):
    counts = instruction_counts(body)
    assert not any(op.startswith(("div.", "rem.", "ld.global")) for op in counts["opcodes"])
    assert "bar.sync" not in counts["opcodes"]
    for kind in ("FULL", "TAIL"):
        regions = body.split(f"// FP8_GATHER_{kind}_BEGIN")
        gather = regions[1].split(f"// FP8_GATHER_{kind}_END")[0] if len(regions) > 1 else ""
        counts[f"gather_{kind.lower()}"] = instruction_counts(gather)
    for region in body.split("// FP8_MASK_BEGIN")[1:]:
        mask = region.split("// FP8_MASK_END")[0]
        assert mask.index("@!tail bra.uni MASK_DONE") < mask.index("setp.ge.u64 invalid")
        assert mask.index("setp.ge.u64 invalid") < mask.index("MASK_DONE:")
    return counts


def sass_stage_counts(sass, expected_mma):
    instructions = []
    for match in re.finditer(
        r"/\*([0-9a-f]+)\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);", sass
    ):
        address, opcode, operands = match.groups()
        instructions.append((int(address, 16), opcode, operands))
    candidates = []
    for address, opcode, operands in instructions:
        target = re.search(r"0x([0-9a-f]+)", operands)
        if opcode.startswith("BRA") and target and int(target[1], 16) < address:
            first = int(target[1], 16)
            body = [item for item in instructions if first <= item[0] <= address]
            if sum(item[1].startswith("HMMA.") for item in body) == expected_mma:
                candidates.append(body)
    assert candidates, "No unambiguous key-stage SASS loop found"
    body = min(candidates, key=len)
    counts = dict(sorted(collections.Counter(item[1] for item in body).items()))
    return {
        "total": len(body),
        "first_pc": hex(body[0][0]),
        "backedge_pc": hex(body[-1][0]),
        "opcodes": counts,
    }


def check_schedule(op):
    from ctikernels.attention._decode_fp8 import _split_geometry

    # Q's byte mapping must fit the final, initially unissued K/V slots.
    size = op.nk * op.pitch
    q_addresses = set()
    for byte in range(op.group * op.dim * 2):
        address = (op.stages - 1) * size + byte % size + byte // size * op.k_bytes
        assert address < op.warp_bytes
        assert address % op.k_bytes >= (op.stages - 1) * size
        assert address not in q_addresses
        q_addresses.add(address)
    intervals = set()
    for begin in (0, 1, op.page - 1, op.page_cap * op.page - 1):
        for length in (
            0,
            1,
            op.kt - 1,
            op.kt,
            op.kt + 1,
            op.piece_keys - 1,
            op.piece_keys,
            op.piece_keys + 1,
            3 * op.piece_keys + 17,
        ):
            intervals.add((begin, begin + length))
    # Include actual aligned split tails, including splits shorter than a CTA
    # stage and starts that are not aligned to a page or a CTA stage.
    for length in (1, 5, 16, 17, 33, 257, 2048, 8192, 8193, 3 * op.piece_keys + 17):
        chunk, count = _split_geometry(length, op.minimum, op.splits)
        for split in range(count):
            intervals.add((split * chunk, min((split + 1) * chunk, length)))
    checked = 0
    for begin, end in sorted(intervals):
        for piece in range(begin, end, op.piece_keys):
            stop = min(piece + op.piece_keys, end)
            base = piece // op.page
            pages = (stop + op.page - 1) // op.page - base
            assert 0 < pages <= op.page_cap
            visited = []
            for warp in range(op.warps):
                ring = [piece + i * op.kt for i in range(op.stages - 1)] + [None]
                slot = 0
                warp_end = stop - min(warp * op.nk, stop - piece)
                for first in range(piece, warp_end, op.kt):
                    assert first + warp * op.nk < stop, "Empty warp must skip its MMA work"
                    assert ring[slot] == first
                    if stop - piece > op.kt:
                        lookahead = (slot + op.stages - 1) % op.stages
                        assert lookahead != slot
                        ring[lookahead] = first + (op.stages - 1) * op.kt
                    for row in range(warp * op.nk, (warp + 1) * op.nk):
                        key = first + row
                        if key < stop:
                            assert 0 <= key // op.page - base < pages
                            visited.append(key)
                        elif first + op.kt <= stop:
                            raise AssertionError("Full-stage path contains an invalid row")
                    slot = (slot + 1) % op.stages
                    checked += 1
            assert sorted(visited) == list(range(piece, stop))
            # Only the last piece may require per-element masking.
            assert stop == end or (stop - piece) % op.kt == 0
    return checked


def worker(args):
    import cutlass
    import torch

    from ctikernels.attention._decode_fp8 import _compile
    from ctikernels.attention._decode_fp8._kernel import Decode

    assert cutlass.__version__ == "4.8.0"
    batch, heads, group, dim, page, dtype, splits, minimum, warps, keys, stages, merge = CASES[
        args.case
    ]
    op = Decode(
        batch,
        heads,
        group,
        dim,
        page,
        cutlass.BFloat16 if dtype == "bfloat16" else cutlass.Float16,
        splits,
        minimum,
        warps,
        keys,
        stages,
        merge,
        args.kv_dense == "dense",
    )
    checked_stages = check_schedule(op)
    _compile(
        batch,
        heads,
        group,
        dim,
        page,
        getattr(torch, dtype),
        args.arch,
        splits,
        minimum,
        warps,
        keys,
        stages,
        merge,
        args.kv_dense == "dense",
    )
    assert not torch.cuda.is_initialized()
    (ptx_path,) = args.output.glob("*Decode*.ptx")
    (cubin_path,) = args.output.glob("*Decode*.cubin")
    ptx = ptx_path.read_text()
    qk_type = "bf16" if dtype == "bfloat16" else "f16"
    assert f"mma.sync.aligned.m16n8k16.row.col.f32.{qk_type}.{qk_type}.f32" in ptx
    assert "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32" in ptx
    if dtype == "bfloat16":
        assert "fma.rn.bf16x2" in ptx
        assert ptx.count("bar.red.and.pred") == 1
        assert ptx.count("// FP8_STAGE_BEGIN") == 2
        # Q is packed before either specialised walk enters its key loop. Neither
        # representability conversion nor the CTA vote belongs in a stage.
        for region in ptx.split("// FP8_STAGE_BEGIN")[1:]:
            body = region.split("// FP8_STAGE_END")[0]
            assert "bar.red" not in body
            assert "cvt.rn.f16.bf16" not in body
        fast, fallback = ptx.split("// FP8_STAGE_BEGIN")[1:]
        fast = fast.split("// FP8_STAGE_END")[0]
        fallback = fallback.split("// FP8_STAGE_END")[0]
        assert "fma.rn.bf16x2" not in fast
        assert "f32.bf16.bf16.f32" not in fast
        assert "fma.rn.bf16x2" in fallback
        assert "f32.bf16.bf16.f32" in fallback
    else:
        assert "bar.red" not in ptx
        assert ptx.count("// FP8_STAGE_BEGIN") == 1
    counts = stage_counts(ptx)
    assert "ld.global.v2.u32" not in ptx
    assert "cp.async.ca.shared.global" in ptx
    assert "ld.shared.v2.u32" in ptx
    prime = ptx.split("// FP8_PRIME_BEGIN")[1].split("// FP8_PRIME_END")[0]
    assert "cp.async.cg.shared.global" in prime and "cp.async.wait_group" not in prime
    after_prime = ptx.split("// FP8_PRIME_END")[1]
    assert after_prime.index("cp.async.wait_group") < after_prime.index("ld.shared.v2.u32")
    assert not any(op.startswith("ld.global") for op in counts["opcodes"])
    for full in ptx.split("// FP8_GATHER_FULL_BEGIN")[1:]:
        body = full.split("// FP8_GATHER_FULL_END")[0]
        assert "ld.shared" in body
        assert "ld.global" not in body
        assert "shfl.sync" not in body
        assert "setp." not in body
        products = re.findall(r"(?:mul|mad).lo.[su]64\s+[^,]+,\s*([^,]+),\s*([^,;]+)[,;]", body)
        runtime_products = [a for a, b in products if a.startswith("%") and b.startswith("%")]
        if args.kv_dense == "dense":
            assert not runtime_products
        else:
            assert len(runtime_products) >= 2

    if merge == "distributed" and splits > 1:
        assert ptx.count("ld.global.cg.v4.f32") == 4
        assert "ld.volatile.global" not in ptx
        assert "ld.acquire.gpu.global.u32" in ptx
        assert "st.release.gpu.global.u32" in ptx
    resources = subprocess.run(
        [args.cuobjdump, "--dump-resource-usage", str(cubin_path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (args.output / "cute-resources.txt").write_text(resources)
    usage = {
        key: int(value)
        for key, value in re.findall(r"(REG|STACK|SHARED|LOCAL):([0-9]+)", resources)
    }
    # Primary performance cases must remain spill-free in the actual CuTe cubin.
    if args.case in ("qwen3", "qwen35", "tp2", "engine128", "engine256", "engine_tp2"):
        assert usage["STACK"] == usage["LOCAL"] == 0, usage
    sass = subprocess.run(
        [args.cuobjdump, "--dump-sass", str(cubin_path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (args.output / "cute-sass.txt").write_text(sass)
    sass_counts = sass_stage_counts(sass, dim // 16 * (keys // 8) * 2)
    assert not any(
        opcode == "LDG" or opcode.startswith("LDG.") for opcode in sass_counts["opcodes"]
    )
    # With a one-stage piece, ptxas folds the key loop into the piece loop;
    # its native backedge necessarily includes the page-bank CTA barrier.
    if op.piece_keys > op.kt:
        assert "BAR.SYNC" not in sass_counts["opcodes"]
    external_ptx = ptx_path
    if args.ptx_version:
        external_ptx = args.output / "external.ptx"
        external_ptx.write_text(
            re.sub(r"(?m)^\.version [0-9.]+", f".version {args.ptx_version}", ptx)
        )
    assembled = subprocess.run(
        [
            args.ptxas,
            f"-arch={args.arch}",
            "-v",
            str(external_ptx),
            "-o",
            str(args.output / "external.cubin"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (args.output / "ptxas.txt").write_text(assembled.stderr)
    regs = int(re.search(r"Used (\d+) registers", assembled.stderr)[1])
    stack, stores, loads = map(
        int,
        re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            assembled.stderr,
        ).groups(),
    )
    receipt = {
        "arch": args.arch,
        "case": args.case,
        "kv_dense": args.kv_dense == "dense",
        "geometry": CASES[args.case],
        "stage": counts,
        "sass_stage": sass_counts,
        "schedule_stages_checked": checked_stages,
        "cute": usage,
        "external_ptxas": {
            "registers": regs,
            "stack": stack,
            "spill_stores": stores,
            "spill_loads": loads,
        },
        "page_cap_per_bank": op.page_cap,
        "piece_keys": op.piece_keys,
        "metadata_bytes": op.metadata_bytes,
        "dynamic_shared": op.arena_bytes + op.metadata_bytes,
        "shared_per_cta": op.arena_bytes + op.metadata_bytes + usage["SHARED"],
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", choices=CASES, nargs="+", help="Subset for the matrix runner")
    parser.add_argument("--kv-dense", choices=("dense", "strided"))
    parser.add_argument("--arch", choices=("sm_120a", "sm_121a"))
    parser.add_argument("--ptxas", default="ptxas")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--ptx-version")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        assert args.kv_dense and args.arch
        worker(args)
        return
    receipts = []
    for arch in (args.arch,) if args.arch else ("sm_120a", "sm_121a"):
        for dense in (args.kv_dense,) if args.kv_dense else ("dense", "strided"):
            for case in args.cases or CASES:
                target = args.output / arch / case / dense
                target.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    "-m",
                    "tests.modules._decode_fp8_compile",
                    "--output",
                    str(target),
                    "--arch",
                    arch,
                    "--case",
                    case,
                    "--kv-dense",
                    dense,
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
                    "CUTE_DSL_COMPILER_OPT": "remarks{ptx}",
                }
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                (target / "compile.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(
                        f"{arch}/{case}/{dense} failed; see {target / 'compile.log'}"
                    )
                receipt = json.loads((target / "receipt.json").read_text())
                receipts.append(receipt)
                print(
                    arch,
                    case,
                    dense,
                    "PTX",
                    receipt["stage"]["total"],
                    "SASS",
                    receipt["sass_stage"]["total"],
                    "CuTe",
                    receipt["cute"],
                    flush=True,
                )
    (args.output / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
