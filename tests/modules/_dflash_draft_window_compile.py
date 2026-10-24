# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline runtime-window signature matrix; no CUDA allocation or launch."""

import os
from itertools import product

import torch

from ctikernels.attention._draft.cute import compile_kernels


def extended_shapes(arch, dims=(64, 128, 256)):
    # Fused-QKV column views use static row strides; out stays compact.
    count = 0
    for dim, mode, pages in product(dims, ("legacy", "tree", "distributed"), (False, True)):
        compile_kernels(
            8,
            64,
            32,
            1,
            False,
            True,
            7,
            torch.bfloat16,
            16,
            arch,
            min_chunk=64,
            key_groups=2,
            pieces=True,
            sparse=True,
            short_union=288,
            inline_merge=mode != "legacy",
            distributed_merge=mode == "distributed",
            head_dim=dim,
            write_pages=pages,
            q_row_stride=66 * dim,
        )
        count += 1
        print(arch, dim, "Q8/G64", mode, "write_pages", pages, "strided q OK", flush=True)
    for dim in dims:
        compile_kernels(
            32,
            128,
            32,
            1,
            True,
            True,
            1,
            torch.float16,
            1,
            arch,
            key_groups=2,
            pieces=True,
            head_dim=dim,
            write_pages=False,
            q_row_stride=130 * dim,
        )
        count += 1
        print(arch, dim, "Q32/G128, no pages, strided q OK", flush=True)
    return count


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    arch = os.environ["CUTE_DSL_ARCH"]
    cases = list(
        product(
            (64, 128, 256), (False, True), (False, True), ("distributed", "tree", "legacy"), (True,)
        )
    )
    cases += [(dim, True, False, "tree", False) for dim in (64, 128, 256)]
    for dim, windowed, causal, mode, pieces in cases:
        compile_kernels(
            8,
            4,
            32,
            2,
            causal,
            windowed,
            7,
            torch.bfloat16,
            16,
            arch,
            min_chunk=64,
            key_groups=2,
            chunk_cap=1024,
            pieces=pieces,
            sparse=True,
            short_union=288,
            inline_merge=mode != "legacy",
            distributed_merge=mode == "distributed",
            head_dim=dim,
        )
        print(arch, dim, windowed, causal, mode, pieces, "OK", flush=True)
    count = len(cases) + extended_shapes(arch)
    assert not torch.cuda.is_initialized()
    print(count, "offline configurations passed", flush=True)


if __name__ == "__main__":
    main()
