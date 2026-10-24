# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline only: python -m tests.gpu._dflash_row_argmax_compile (no CUDA device)."""

import os
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def main():
    # Configure before importing CuTe; never place compiler artifacts in the repo.
    with TemporaryDirectory(prefix="dflash-row-argmax-") as dump:
        os.environ["CUTE_DSL_DUMP_DIR"] = dump
        os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
        os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
        from ctikernels.sampling._dflash.cute.argmax import compile_variant

        for arch, dtype, out_dtype, splits, proposal_dtype in product(
            ("120a", "121a"),
            (torch.bfloat16, torch.float16, torch.float32),
            (torch.int32, torch.int64),
            (1, 8),
            (None, torch.int32, torch.int64),
        ):
            compiled = compile_variant(
                dtype, out_dtype, splits, arch, proposal_dtype=proposal_dtype
            )
            ptx = compiled.__ptx__
            if "\n" not in ptx:
                ptx = Path(ptx).read_text()
            cubin = compiled.__cubin__
            if isinstance(cubin, str):
                cubin = Path(cubin).read_bytes()
            assert cubin.startswith(b"\x7fELF")
            for instruction in (
                ".reqntid 128, 1, 1",
                "ld.global.v4.",
                "max.u64",
                "atom.relaxed.gpu.global.max.u64",
                "membar.gl;",
                "atom.acq_rel.gpu.global.add.u32",
                "atom.acquire.gpu.global.add.u64",
                "atom.relaxed.gpu.global.exch.b64",
                "atom.release.gpu.global.exch.b32",
            ):
                assert instruction in ptx, instruction
            if dtype == torch.bfloat16:
                assert "max.NaN.bf16x2" in ptx
            if proposal_dtype is not None:
                for instruction in (
                    "atom.relaxed.gpu.global.add.u32",
                    "atom.relaxed.gpu.global.max.u32",
                    "atom.relaxed.gpu.global.exch.b32",
                ):
                    assert instruction in ptx, instruction
            print(
                f"sm_{arch} {dtype} {out_dtype} S={splits} proposals={proposal_dtype}: PTX/cubin passed",
                flush=True,
            )


if __name__ == "__main__":
    main()
