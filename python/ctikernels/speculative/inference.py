# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Borrowed-buffer DFlash round bookkeeping and the DFlash2 block
convolution (``block_conv``)."""

import torch

from ._round.op import (
    ALL_VARIANTS,
    RoundVariant,
    commit_block,
    compile_count,
    fixed_query_addresses,
    prepare_block,
    prepare_round,
    warm_up,
)

__all__ = [
    "block_conv",
    "prepare_block",
    "prepare_round",
    "fixed_query_addresses",
    "commit_block",
    "warm_up",
    "RoundVariant",
    "ALL_VARIANTS",
    "compile_count",
]


def block_conv(
    x: torch.Tensor,
    deltas: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """DFlash2 block convolution, one launch per call. Row ``n`` of a ``Q``-row block, channel
    ``c`` of segment ``s = c // (C / S)``: ``out[n, c] = sum_{t < taps, t <= n % Q}
    (base[t, c] + deltas[n, t, s]) * x[n - t, c]`` (taps before the block start contribute
    nothing). Rounding matches the composed torch form bitwise: FP32 coefficient sum, FP32
    products, FP32 sums in tap order (no FMA), one cast to ``x.dtype``.

    ``x`` [N, C] (N a multiple of ``block_size``, rows in consecutive blocks), ``deltas``
    [N, taps, S] (any row stride; each row's [taps, S] contiguous, e.g. one side of the
    projection's [N, 2, taps, S]), ``base`` [taps, C] contiguous, C a multiple of S. Returns
    ``out`` [N, C] in ``x.dtype``. No host sync; CUDA-graph capturable."""
    if x.ndim != 2 or deltas.ndim != 3 or base.ndim != 2:
        raise ValueError("x [N, C], deltas [N, taps, S], base [taps, C]")
    n, c = x.shape
    taps, segments = deltas.shape[1], deltas.shape[2]
    if deltas.shape[0] != n or base.shape != (taps, c) or c % segments or taps < 1:
        raise ValueError("deltas [N, taps, S] and base [taps, C] with C a multiple of S")
    if block_size < 1 or n % block_size:
        raise ValueError("N must be a multiple of block_size")
    if not x.is_contiguous() or not base.is_contiguous():
        raise ValueError("x and base must be contiguous")
    if deltas.stride(2) != 1 or deltas.stride(1) != segments:
        raise ValueError("each row's deltas [taps, S] must be contiguous")
    if any(t.device != x.device for t in (deltas, base)):
        raise ValueError("x, deltas and base must share a device")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("out must be a contiguous [N, C] tensor of x.dtype")
    if n == 0:
        return out
    from .triton._block_conv import launch

    launch(x, deltas, base, out, block_size)
    return out
