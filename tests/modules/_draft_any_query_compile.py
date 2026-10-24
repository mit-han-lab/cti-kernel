# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Offline lowering for large-Q prepare and every CuTe merge mode."""

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from ctikernels.attention._draft.cute import compile_kernels
from ctikernels.attention._flashinfer_schedule import (
    prepare_draft_chunked,
    prepare_split_kv_chunked,
)


def main():
    # Constants deliberately exceed Triton's maximum vector size. The compiled
    # kernels must use fixed chunks, including the request scan (B>1024).
    for kernel in (prepare_draft_chunked, prepare_split_kv_chunked):
        values = {
            "BATCH": 1025,
            "QUERY": 4096,
            "PAGE": 16,
            "QUERY_TILE_COUNT": 128,
            "MAX_CTAS": 42,
            "PADDED_CTAS": 132224,
            "REQUEST_BLOCK": 2048,
            "TILE_BLOCK": 262144,
            "MERGE_BLOCK": 8388608,
            "SEARCH_STEPS": 11,
            "QUERY_TILE_SIZE": 128,
            "MIN_CHUNK": 128,
            "SHORT_TOKENS": 0,
            "MIN_CHUNK_2": 128,
            "SHORT_TOKENS_2": 0,
            "MIN_CHUNK_3": 128,
            "SHORT_TOKENS_3": 0,
            "PAGE_TILES": 1,
            "GATHER_TILE": 1024,
            "TRIM_WINDOW": False,
            "WINDOWED": False,
        }
        constants = {p.name: values[p.name] for p in kernel.params if p.is_constexpr}
        signature = {
            p.name: "i32"
            if p.name in ("WINDOW_LEFT", "TRIM_WINDOW_LEFT")
            else "*i1"
            if p.name == "VALID"
            else "*i32"
            for p in kernel.params
            if not p.is_constexpr
        }
        compiled = triton.compile(
            ASTSource(kernel, signature=signature, constexprs=constants),
            target=GPUTarget("cuda", 120, 32),
        )
        assert compiled.asm["ptx"]
    for dim in (64, 128, 256):
        for query, mode in ((1, "distributed"), (129, "legacy"), (4096, "tree")):
            compile_kernels(
                query,
                4,
                32,
                8,
                False,
                True,
                3,
                torch.bfloat16,
                16,
                "sm_120a",
                key_groups=2,
                pieces=True,
                sparse=True,
                short_union=288,
                inline_merge=mode != "legacy",
                distributed_merge=mode == "distributed",
                head_dim=dim,
            )
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
