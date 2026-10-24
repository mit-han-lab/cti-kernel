# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Compile production DRAFT prepare/attention/merge with fake tensors and streams."""

import os

import torch

from ctikernels.attention._draft.cute import compile_kernels


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    arch = os.environ["CUTE_DSL_ARCH"]
    count = 0
    for q, rows, page in [(4, 32, 1), (16, 64, 16)]:
        for causal, windowed in [(False, False), (True, True), (False, True), (True, False)]:
            for splits, pieces in [(1, False), (7, True)]:
                compile_kernels(
                    q,
                    4,
                    rows,
                    4,
                    causal,
                    windowed,
                    splits,
                    torch.bfloat16,
                    page,
                    arch,
                    chunk_cap=128,
                    pieces=pieces,
                )
                count += 1
                print(arch, q, rows, page, causal, windowed, splits, pieces, "OK", flush=True)
    # Non-power-of-two Q, FP16 and sparse merge are independent compile branches.
    compile_kernels(
        9,
        4,
        48,
        8,
        False,
        False,
        7,
        torch.float16,
        16,
        arch,
        sparse=True,
    )
    count += 1
    assert not torch.cuda.is_initialized()
    print(f"{count} offline DRAFT configurations passed for {arch}")


if __name__ == "__main__":
    main()
