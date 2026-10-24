# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Slot glue of the fla-core prefill adapter, one Triton launch on each side of fla's kernels.

fla-core's chunk kernels take dense per-sequence states and a varlen chunk table:

- :func:`prepare`: the chunk table [capacity, 2] int32 (sequence, chunk in it; entries past
  the batch's chunks name a chunk past the last sequence's end), the chunk offsets [N + 1]
  int64 and the dense initial states [N, Hv, V, K] FP32: a continued sequence's
  (``slot >= 0`` and ``has_initial_state``) pool rows, zeros otherwise;
- :func:`commit`: for each sequence with ``slot >= 0`` that is non-empty or fresh, its final
  state (zeros if empty) and its new conv history, the last three rows of
  [history ++ its rows of projected], written to its slot. Each program reads its columns'
  old history before writing them.

Bit copies only; device values only (no host sync), so a captured graph follows new lengths,
slots and flags on replay."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._contract import HISTORY, Geometry, K, V

CHUNK = 64
_BLOCK = 4096
_BN = 128
_BE = 128


@triton.jit
def _chunks(cu_seqlens, n, m, CHUNK: tl.constexpr):
    lengths = tl.load(cu_seqlens + n + 1, mask=m, other=0) - tl.load(
        cu_seqlens + n, mask=m, other=0
    )
    return ((lengths.to(tl.int64) + CHUNK - 1) // CHUNK).to(tl.int64)


@triton.jit(do_not_specialize=["count", "capacity"])
def _prepare_kernel(
    cu_seqlens, slot_indices, has_initial, state, table, offsets, initial, count, capacity,
    STATE_WIDTH: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr, BN: tl.constexpr, BE: tl.constexpr,
):  # fmt: skip
    n, block = tl.program_id(0), tl.program_id(1)
    if n == count:
        if block == 0:
            tl.store(offsets, 0)
            carry = tl.zeros((), dtype=tl.int64)
            for start in range(0, count, BN):
                i = start + tl.arange(0, BN)
                chunks = _chunks(cu_seqlens, i, i < count, CHUNK)
                tl.store(offsets + 1 + i, carry + tl.cumsum(chunks, 0), mask=i < count)
                carry += tl.sum(chunks, 0)
            for first in range(0, capacity, BE):
                e = first + tl.arange(0, BE)
                found = tl.zeros((BE,), dtype=tl.int32)
                carry = tl.zeros((), dtype=tl.int64)
                for start in range(0, count, BN):
                    i = start + tl.arange(0, BN)
                    chunks = _chunks(cu_seqlens, i, i < count, CHUNK)
                    ends = carry + tl.cumsum(chunks, 0)
                    done = (ends[None, :] <= e[:, None]) & (i < count)[None, :]
                    found += tl.sum(done.to(tl.int32), 1)
                    carry += tl.sum(chunks, 0)
                seq = tl.minimum(found, count - 1)
                begin = tl.zeros((BE,), dtype=tl.int64)
                for start in range(0, count, BN):
                    i = start + tl.arange(0, BN)
                    chunks = _chunks(cu_seqlens, i, i < count, CHUNK)
                    begin += tl.sum(tl.where(i[None, :] < seq[:, None], chunks[None, :], 0), 1)
                inside = e < capacity
                tl.store(table + e * 2, seq.to(table.dtype.element_ty), mask=inside)
                tl.store(table + e * 2 + 1, (e - begin).to(table.dtype.element_ty), mask=inside)
        return
    slot = tl.load(slot_indices + n).to(tl.int64)
    continued = (slot >= 0) & (tl.load(has_initial + n) != 0)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    m = cols < STATE_WIDTH
    rows = tl.load(state + slot * STATE_WIDTH + cols, mask=m & continued, other=0.0)
    tl.store(initial + n * STATE_WIDTH + cols, rows, mask=m)


def prepare(
    geometry: Geometry,
    rows: int,
    cu_seqlens: torch.Tensor,
    slot_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(table, offsets, initial); see the module."""
    count = cu_seqlens.shape[0] - 1
    capacity = (rows + (CHUNK - 1) * count) // CHUNK
    device = state.device
    table = torch.empty(capacity, 2, dtype=cu_seqlens.dtype, device=device)
    offsets = torch.empty(count + 1, dtype=torch.int64, device=device)
    initial = torch.empty(count, *state.shape[1:], dtype=state.dtype, device=device)
    state_width = geometry.v_heads * V * K
    _prepare_kernel[(count + 1, triton.cdiv(state_width, _BLOCK))](
        cu_seqlens, slot_indices, has_initial_state, state, table, offsets, initial, count,
        capacity, STATE_WIDTH=state_width, CHUNK=CHUNK, BLOCK=_BLOCK, BN=_BN, BE=_BE, num_warps=4,
    )  # fmt: skip
    return table, offsets, initial


@triton.jit
def _commit_kernel(
    projected, p_stride, cu_seqlens, slot_indices, has_initial, final, state, conv_state,
    STATE_WIDTH: tl.constexpr, WIDTH: tl.constexpr, HISTORY: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    n, block = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slot_indices + n).to(tl.int64)
    continued = tl.load(has_initial + n) != 0
    end = tl.load(cu_seqlens + n + 1).to(tl.int64)
    length = end - tl.load(cu_seqlens + n).to(tl.int64)
    if (slot >= 0) & ((length > 0) | (continued == 0)):
        cols = block * BLOCK + tl.arange(0, BLOCK)
        m = cols < STATE_WIDTH
        rows = tl.load(final + n * STATE_WIDTH + cols, mask=m & (length > 0), other=0.0)
        tl.store(state + slot * STATE_WIDTH + cols, rows, mask=m)
        r = tl.arange(0, 4)  # history rows, oldest first (row 3 is padding)
        m = (r < HISTORY)[:, None] & (cols < WIDTH)[None, :]
        old = conv_state + slot * (HISTORY * WIDTH) + cols[None, :]
        back = (HISTORY - r)[:, None]
        from_x = m & (length >= back)
        x = tl.load(projected + (end - back) * p_stride + cols[None, :], mask=from_x, other=0.0)
        kept = m & (length < back) & continued
        rows = tl.load(old + (HISTORY + length - back) * WIDTH, mask=kept, other=0.0)
        tl.store(old + r[:, None] * WIDTH, tl.where(from_x, x, rows), mask=m)


def commit(
    geometry: Geometry,
    projected: torch.Tensor,
    cu_seqlens: torch.Tensor,
    slot_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    final: torch.Tensor,
    state: torch.Tensor,
    conv_state: torch.Tensor,
) -> None:
    """Write final states and conv histories to the pools (see the module)."""
    count = cu_seqlens.shape[0] - 1
    if count == 0:
        return
    state_width = geometry.v_heads * V * K
    _commit_kernel[(count, triton.cdiv(state_width, _BLOCK))](
        projected, projected.stride(0), cu_seqlens, slot_indices, has_initial_state, final,
        state, conv_state, STATE_WIDTH=state_width, WIDTH=geometry.conv_dim, HISTORY=HISTORY,
        BLOCK=_BLOCK, num_warps=4,
    )  # fmt: skip


__all__ = ["CHUNK", "commit", "prepare"]
