# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Embedding-table inference ops."""

from __future__ import annotations

import math
import struct

import torch

__all__ = ["bilinear_pos_embed_"]

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def bilinear_pos_embed_(
    table: torch.Tensor,
    out: torch.Tensor,
    *,
    offset: int,
    height: int,
    width: int,
    merge_size: int,
) -> None:
    """Write one image's bilinearly resampled position embeddings to rows
    ``out[offset : offset + height * width]``, its patches in merge-window order: the
    ``merge_size x merge_size`` windows of the ``height x width`` patch grid row by row, and
    within a window its patches row by row.

    ``table`` [side * side, C] is a square grid of learned embeddings whose corners align with
    the image's. Patch (i, j) blends the four cells around source point (s(i, height),
    s(j, width)), in FP32 as cti-engine's composed Qwen3.5 path on CUDA computes it:
    ``s(i, n) = fl(i * (side - 1)) * inv`` with ``inv = fp32(1 / max(n - 1, 1))`` (CUDA torch
    divides by a host scalar through its reciprocal), ``lo = floor(s)``, per-axis taps
    ``min(lo + k, side - 1)`` with weights ``max(1 - |(s - lo) - k|, 0)`` for k = 0, 1, the cell
    weight ``row weight * column weight``, and ``sum_t table[cell_t] * weight_t`` accumulated in
    tap order (r0, c0), (r0, c1), (r1, c0), (r1, c1), each product and sum rounded to FP32
    (no FMA), then rounded once to ``out.dtype``.

    ``table`` and ``out`` are FP16, BF16 or FP32 with unit column stride on one device;
    ``out`` [N, C] with non-overlapping rows. Requires ``height`` and ``width`` divisible by
    ``merge_size`` and ``offset + height * width <= N``. CUDA: one Triton launch, no host sync
    or allocation, CUDA-graph capturable. CPU: the torch form of the same law. No autograd.
    """
    side = _check(table, out, offset, height, width, merge_size)
    inv_h, inv_w = _reciprocal(height), _reciprocal(width)
    if out.is_cuda:
        from .triton._bilinear import launch

        launch(table, out, offset, height, width, side, merge_size, inv_h, inv_w)
    else:
        _bilinear_torch(table, out, offset, height, width, side, merge_size, inv_h, inv_w)


def _check(table, out, offset, height, width, merge_size) -> int:
    if torch.is_grad_enabled() and (table.requires_grad or out.requires_grad):
        raise RuntimeError("bilinear_pos_embed_ does not support autograd")
    if table.ndim != 2 or out.ndim != 2 or out.shape[1] != table.shape[1]:
        raise ValueError("table must be [side * side, C] and out [N, C]")
    side = math.isqrt(table.shape[0])
    if side < 1 or side * side != table.shape[0]:
        raise ValueError("table rows must be a square grid, side * side")
    if table.dtype not in _DTYPES or out.dtype not in _DTYPES:
        raise TypeError("table and out must be FP16, BF16 or FP32")
    if table.device != out.device:
        raise ValueError("table and out must share one device")
    if table.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("table and out must have unit column stride")
    if out.shape[0] > 1 and out.stride(0) < out.shape[1]:
        raise ValueError("out rows must not overlap")
    if merge_size < 1 or height < 1 or width < 1 or height % merge_size or width % merge_size:
        raise ValueError("height and width must be positive multiples of merge_size")
    if offset < 0 or offset + height * width > out.shape[0]:
        raise ValueError("rows [offset, offset + height * width) must lie in out")
    return side


def _reciprocal(size: int) -> float:
    """``fp32(1 / max(size - 1, 1))``: the double reciprocal rounded to nearest FP32."""
    return struct.unpack("f", struct.pack("f", 1.0 / max(size - 1, 1)))[0]


def _bilinear_torch(table, out, offset, height, width, side, merge_size, inv_h, inv_w):
    shape = (height // merge_size, width // merge_size, merge_size, merge_size)
    rows = torch.arange(height).view(shape[0], 1, merge_size, 1).expand(shape).flatten()
    columns = torch.arange(width).view(1, shape[1], 1, merge_size).expand(shape).flatten()

    def axis(index, inv):
        source = index.float() * (side - 1) * torch.tensor(inv, dtype=torch.float32)
        lower = source.floor()
        k = torch.arange(2)
        taps = (lower.long().unsqueeze(-1) + k).clamp_(max=side - 1)
        weights = (1 - (source - lower).unsqueeze(-1).sub(k).abs()).clamp_(min=0)
        return taps, weights

    row_taps, row_weights = axis(rows, inv_h)
    column_taps, column_weights = axis(columns, inv_w)
    cells = (row_taps.unsqueeze(-1) * side + column_taps.unsqueeze(-2)).flatten(1)
    weights = (row_weights.unsqueeze(-1) * column_weights.unsqueeze(-2)).flatten(1)
    acc = table[cells[:, 0]].float() * weights[:, :1]
    for tap in range(1, 4):
        acc += table[cells[:, tap]].float() * weights[:, tap : tap + 1]
    out[offset : offset + height * width] = acc.to(out.dtype)
