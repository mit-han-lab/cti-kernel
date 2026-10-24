# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Slot-pool glue for implementations whose kernels take dense per-sequence states.

A kernel without state indices (fla-core's) reads the initial states of a batch from a dense
``[entries, ...]`` tensor and returns the final ones in another. These helpers move exactly the
batch's pool rows between the pool and such tensors, on the device, with no host sync, so a
CUDA graph captured once follows changed slot indices and masks on replay:

- :func:`gather_rows` copies ``pool[slot]`` into the entry's row of a zeroed buffer for the
  active entries; inactive entries' rows stay zero and their pool rows are never read;
- :func:`write_rows` copies kept entries' rows into ``pool[slot]``; no other pool row is
  written, and no pool row is read.

Both run one Triton row copy whose program for an entry with a negative (or out-of-range)
row index returns before computing an address. Written for this contract; the early-return
masking follows SGLang 0.5.19's ``fused_conv_window_scatter_with_mask``
(``sglang/kernels/ops/mamba/mamba_state_scatter_triton.py``), nothing is copied from it."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit(do_not_specialize=["src_count", "dst_count"])
def _masked_row_copy_kernel(
    src,
    dst,
    src_rows,
    dst_rows,
    src_count,
    dst_count,
    width,
    BLOCK: tl.constexpr,
):
    entry = tl.program_id(0)
    source = tl.load(src_rows + entry).to(tl.int64)
    target = tl.load(dst_rows + entry).to(tl.int64)
    # An inactive entry (or an index outside its tensor) touches no memory at all.
    if (source < 0) | (target < 0) | (source >= src_count) | (target >= dst_count):
        return
    offsets = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < width
    values = tl.load(src + source * width + offsets, mask=inside)
    tl.store(dst + target * width + offsets, values, mask=inside)


def _masked_row_copy(
    dst: torch.Tensor, src: torch.Tensor, src_rows: torch.Tensor, dst_rows: torch.Tensor
) -> None:
    """``dst[dst_rows[e]] = src[src_rows[e]]`` bit for bit for every entry whose two indices
    are valid rows. ``dst`` and ``src`` are contiguous, of one dtype and row size."""
    width = dst[0].numel() if dst.shape[0] else src[0].numel() if src.shape[0] else 0
    entries = src_rows.shape[0]
    if entries == 0 or width == 0:
        return
    grid = (entries, triton.cdiv(width, _BLOCK))
    _masked_row_copy_kernel[grid](
        src, dst, src_rows, dst_rows, src.shape[0], dst.shape[0], width, BLOCK=_BLOCK
    )


def gather_rows(
    pool: torch.Tensor, slot_indices: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """``pool[slot_indices[i]]`` where ``active[i]``, zeros elsewhere, as a new
    ``[entries, *pool.shape[1:]]`` tensor; reads only the active entries' pool rows."""
    count = slot_indices.shape[0]
    rows = torch.zeros((count, *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    entries = torch.arange(count, dtype=torch.int32, device=pool.device)
    sources = torch.where(active, slot_indices, -1).to(torch.int32)
    _masked_row_copy(rows, pool, sources, entries)
    return rows


def write_rows(
    pool: torch.Tensor, slot_indices: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
) -> None:
    """``pool[slot_indices[i]] = rows[i]`` where ``keep[i]``, bit for bit; writes only the
    kept entries' pool rows (kept slots are unique) and reads none."""
    if rows.dtype != pool.dtype or rows.shape[1:] != pool.shape[1:]:
        raise ValueError(f"rows {rows.dtype} {tuple(rows.shape)} do not fit the pool")
    entries = torch.arange(slot_indices.shape[0], dtype=torch.int32, device=pool.device)
    targets = torch.where(keep, slot_indices, -1).to(torch.int32)
    _masked_row_copy(pool, rows.contiguous(), entries, targets)


__all__ = ["gather_rows", "write_rows"]
