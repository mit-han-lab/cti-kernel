# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline DF4 compile coverage; CUDA_VISIBLE_DEVICES must be empty."""

import os
from itertools import product

import torch

from ctikernels.attention._draft.cute import compile_kernels
from ctikernels.attention._draft.flashinfer._table_csr import compile_table_csr


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    for arch, dim, mode, metadata_index in product(
        ("sm_120a", "sm_121a"),
        (128, 256),
        ("single", "tree", "distributed"),
        ("request", "row"),
    ):
        for pieces in (False, True):
            prepare, _, merge = compile_kernels(
                9,
                4,
                48,
                4,
                True,
                4096,
                1 if mode == "single" else 7,
                torch.bfloat16,
                1,
                arch,
                min_chunk=64,
                key_tile=32,
                key_groups=2,
                stages=2,
                chunk_cap=128,
                pieces=pieces,
                short_union=64,
                inline_merge=mode != "single",
                distributed_merge=mode == "distributed",
                head_dim=dim,
                table=metadata_index,
            )
            assert prepare is None and merge is None
            print(arch, dim, mode, metadata_index, "pieces", pieces, "OK", flush=True)
    # CSR entry point remains compilable; noncausal windows and cooperative no-PDL.
    for table in (False, "request", "row"):
        compile_kernels(
            5,
            4,
            32,
            8,
            False,
            2048,
            7,
            torch.float16,
            1,
            "sm_120a",
            key_tile=64,
            stages=3,
            pieces=True,
            inline_merge=True,
            distributed_merge=True,
            coop_pdl=False,
            table=table,
        )
    for arch, metadata_index in product(
        ("sm_80", "sm_89", "sm_90", "sm_120", "sm_121"), ("request", "row")
    ):
        compile_table_csr(5, 4100, arch, row_metadata=metadata_index == "row")
        print(arch, metadata_index, "table CSR OK", flush=True)
    assert not torch.cuda.is_initialized()
    print("DF4 offline compilations passed without CUDA initialization", flush=True)


if __name__ == "__main__":
    main()
