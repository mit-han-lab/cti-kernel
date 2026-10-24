# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DF3 fake-tensor compile matrix; set CUDA_VISIBLE_DEVICES='' and CUTE_DSL_ARCH."""

import os
from itertools import product

import torch

from ctikernels.attention._draft.cute import compile_kernels


def distributed_shapes(arch):
    count = 0
    for rows, groups, tile in product((16, 48, 64), (1, 2), (32, 64)):
        compile_kernels(
            9,
            4,
            rows,
            8,
            True,
            False,
            43,
            torch.float16,
            16,
            arch,
            min_chunk=64,
            key_tile=tile,
            key_groups=groups,
            stages=3,
            chunk_cap=1024,
            pieces=True,
            short_union=416,
            inline_merge=True,
            distributed_merge=True,
        )
        count += 1
        print(arch, rows, groups, tile, "distributed FP16/padding OK", flush=True)
    return count


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    arch = os.environ["CUTE_DSL_ARCH"]
    count = 0
    for mode, tile, short, query, heads, (causal, windowed) in product(
        ("legacy", "tree", "distributed"),
        (32, 64),
        (0, 416),
        (4, 16),
        (4, 8),
        ((False, False), (True, False), (True, True), (False, True)),
    ):
        _, _, merge = compile_kernels(
            query,
            4,
            32,
            heads,
            causal,
            windowed,
            7,
            torch.bfloat16,
            16,
            arch,
            min_chunk=64,
            key_tile=tile,
            key_groups=2,
            stages=2,
            chunk_cap=1024,
            pieces=True,
            sparse=True,
            short_union=short,
            inline_merge=mode != "legacy",
            distributed_merge=mode == "distributed",
        )
        assert (merge is None) == (mode != "legacy")
        count += 1
        print(arch, mode, tile, short, query, heads, causal, windowed, "OK", flush=True)
    # Exercise all row/group shapes, single-split, no piece loop, and FP16/deeper rings.
    for rows, groups, splits in product((16, 32, 48, 64), (1, 2), (1, 7)):
        compile_kernels(
            16,
            4,
            rows,
            8,
            True,
            True,
            splits,
            torch.float16,
            1,
            arch,
            key_tile=64,
            key_groups=groups,
            stages=3,
            chunk_cap=1024,
            short_union=416,
            inline_merge=True,
        )
        count += 1
        print(arch, rows, groups, splits, "FP16 OK", flush=True)
    # Large split counts exercise G=2/4, all levels, singleton tails and padding.
    for tile, rows, groups, splits in product((32, 64), (16, 32, 48, 64), (1, 2), (23, 43)):
        compile_kernels(
            9,
            4,
            rows,
            8,
            True,
            False,
            splits,
            torch.bfloat16,
            16,
            arch,
            min_chunk=64,
            key_tile=tile,
            key_groups=groups,
            stages=2,
            chunk_cap=1024,
            pieces=True,
            short_union=416,
            inline_merge=True,
        )
        count += 1
        print(arch, tile, rows, groups, splits, "hierarchical OK", flush=True)
    # The cooperative variant must also compile without any PDL instructions.
    compile_kernels(
        4,
        4,
        32,
        8,
        True,
        False,
        7,
        torch.bfloat16,
        16,
        arch,
        key_tile=64,
        inline_merge=True,
        distributed_merge=True,
        coop_pdl=False,
    )
    count += 1
    count += distributed_shapes(arch)
    from ._dflash_draft_window_compile import extended_shapes

    count += extended_shapes(arch, dims=(64, 128))
    assert not torch.cuda.is_initialized()
    print(f"{count} offline DF3 compilations passed for {arch}", flush=True)


if __name__ == "__main__":
    main()
