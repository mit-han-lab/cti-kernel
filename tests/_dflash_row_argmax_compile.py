# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline only: PYTHONPATH=python python tests/_dflash_row_argmax_compile.py."""

import os
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    # Set before importing torch/CuTe; never initialize or launch on a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with TemporaryDirectory(prefix="dflash-row-argmax-") as dump:
        os.environ["CUTE_DSL_DUMP_DIR"] = dump
        os.environ["CUTE_DSL_KEEP"] = "ptx,cubin"
        os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
        import torch

        from ctikernels.sampling._dflash.cute.argmax import compile_variant, split_count

        assert split_count(1, 2**31 - 1, 2, 65536, waves=8) == 65535
        variants = [(None, False, False)] + [
            (proposal_dtype, stream, long_rows)
            for proposal_dtype in (torch.int32, torch.int64)
            for stream, long_rows in ((False, False), (True, False), (False, True))
        ]
        count = 0
        for arch, dtype, out_dtype, (proposal_dtype, stream, long_rows) in product(
            ("120", "121"),
            (torch.bfloat16, torch.float16, torch.float32),
            (torch.int32, torch.int64),
            variants,
        ):
            # The sample split value is an Int32 runtime parameter, not a variant.
            compiled = compile_variant(
                dtype,
                out_dtype,
                8,
                arch,
                proposal_dtype=proposal_dtype,
                stream=stream,
                long_rows=long_rows,
            )
            ptx = compiled.__ptx__
            if "\n" not in ptx:
                ptx = Path(ptx).read_text()
            cubin = compiled.__cubin__
            if isinstance(cubin, str):
                cubin = Path(cubin).read_bytes()
            assert cubin.startswith(b"\x7fELF")
            threads = 512 if long_rows else 128
            for instruction in (
                f".reqntid {threads}, 1, 1",
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
            if stream and dtype == torch.bfloat16:
                assert "ld.global.nc.L1::no_allocate.v4.b32" in ptx
            else:
                assert "ld.global.v4." in ptx
            if proposal_dtype is not None:
                for instruction in (
                    "atom.relaxed.gpu.global.add.u32",
                    "atom.relaxed.gpu.global.max.u32",
                    "atom.relaxed.gpu.global.exch.b32",
                ):
                    assert instruction in ptx, instruction
                if not long_rows:
                    assert "%nctaid.y" in ptx
            count += 1
            print(
                f"sm_{arch} {dtype} {out_dtype} proposals={proposal_dtype} "
                f"stream={stream} long_rows={long_rows}: PTX/cubin passed",
                flush=True,
            )
        assert not torch.cuda.is_initialized()
        print(f"All {count} production variants passed without a GPU", flush=True)


if __name__ == "__main__":
    main()
