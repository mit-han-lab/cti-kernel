# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from SGLang 0.5.22 baa9befa381511cd4d56f83980951e5a45fffc6d
# (`python/sglang/kernels/ops/gemm/gptq_marlin.py`,
# `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`); see THIRD_PARTY_NOTICES.

"""The ``marlin`` provider: SGLang's Marlin W4A16 GEMM pruned to NVFP4
(``csrc/jit/marlin_nvfp4.cu``, built on first use through tvm-ffi; one module per
activation dtype). ``repack`` converts a checkpoint NVFP4 weight once into the kernel's
tile layout; ``gemm`` runs on compute capability 8.9 or newer (the package floor)."""

from __future__ import annotations

import functools

import torch

from ...utils import load_jit

NAME = "marlin"
MAX_TILE_ROWS = 64  # the largest per-block row tile; sizes the FP32 reduction scratch
MAX_TILE_N = 256


@functools.cache
def _module(dtype: torch.dtype):
    scalar = {torch.bfloat16: "nv_bfloat16", torch.float16: "half"}[dtype]
    return load_jit(
        "marlin_nvfp4",
        scalar,
        cuda_files=["marlin_nvfp4.cu"],
        cuda_wrappers=[
            ("gemm", f"MarlinNVFP4<{scalar}>::gemm"),
            ("gemm_rows", f"MarlinNVFP4<{scalar}>::gemm_rows"),
            ("repack", "marlin_nvfp4_repack"),
            ("repack_rows", "marlin_nvfp4_repack_rows"),
        ],
    )


def padded_shape(n: int, k: int) -> tuple[int, int]:
    """The smallest (N, K) at or above the given one that the thread tiles cover: both
    multiples of 64 and one of 128 (SGLang's marlin_padded_nk choice)."""

    def up(value, grain):
        return -(-value // grain) * grain

    return min(
        ((up(n, 64), up(k, 128)), (up(n, 128), up(k, 64))),
        key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]),
    )


def scratch_counts(sms: int) -> tuple[int, int]:
    """Element counts of (FP32 reduction partials, int32 slice locks) for ``sms`` SMs."""
    return sms * MAX_TILE_ROWS * MAX_TILE_N, sms


def repack(data: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Marlin layout of an NVFP4 weight ``[N, K]``: int32 ``[K / 16, 2 N]`` codes and uint8
    ``[K / 16, N]`` scales. Codes of zero-scale blocks become zero."""
    n, k = data.shape[0], data.shape[1] * 2
    weight = torch.empty(k // 16, 2 * n, dtype=torch.int32, device=data.device)
    scales = torch.empty(k // 16, n, dtype=torch.uint8, device=data.device)
    _module(torch.bfloat16).repack(data, scale, weight, scales)
    return weight, scales


def repack_rows(data: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Marlin layout of the same weight ``W [N, K]`` for ``dy . W`` (reduction over N, the
    rows): int32 ``[N / 16, 2 K]`` codes and uint8 ``[N / 16, K]`` row scales."""
    n, k = data.shape[0], data.shape[1] * 2
    weight = torch.empty(n // 16, 2 * k, dtype=torch.int32, device=data.device)
    scales = torch.empty(n // 16, k, dtype=torch.uint8, device=data.device)
    _module(torch.bfloat16).repack_rows(data, scale, weight, scales)
    return weight, scales


def gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    partials: torch.Tensor,
    locks: torch.Tensor,
) -> None:
    """``out = alpha * x . W^T`` rounded once; ``locks`` are zero and left zero."""
    _module(x.dtype).gemm(x, weight, scales, alpha, out, partials, locks)


def gemm_rows(x, weight, scales, alpha, out, partials, locks) -> None:
    """``out = alpha * x . W`` with the ``repack_rows`` layout of ``W``."""
    _module(x.dtype).gemm_rows(x, weight, scales, alpha, out, partials, locks)
