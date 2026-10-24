# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""SwiGLU forward, and its backward: the gate | up gradient over the saved gate | up, with the
down A and gate/up B gradients fused (``h`` recomputed in registers)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..._runtime.tables import Dim, TunedOp, pow2_edges
from ...triton.silu import sigmoid, silu

_BLOCK = 1024
# The fused backward's launch config, tuned offline per device class (no runtime autotune).
# ``programs`` is the grid size the row chunks aim for: their FP32 partials are summed in order,
# so a config, a pure function of the shape and the table, fixes the bits.
TUNED = TunedOp(
    "mlp.swiglu_backward.triton",
    version=3,
    space={
        "block_i": (32, 64, 128),
        "block_t": (32, 64, 128),
        "warps": (4, 8),
        "stages": (1, 2, 3),
        "programs": (256, 512, 1024),
    },
    dims=(Dim("t", pow2_edges(1, 16384)), Dim("i"), Dim("rank", (16, 32, 64, 128)), Dim("dtype")),
    # Measured on RTX PRO 6000 (the fused gate/up B gradients; sweep at T 4k-16k, r 16/64).
    default=lambda shape, device_class: {
        "block_i": 32 if shape["dtype"] == "float32" else 64,  # sm120 FP32 shared memory
        "block_t": 32,
        "warps": 4,
        "stages": 1 if shape["dtype"] == "float32" else 2,
        "programs": 1024,
    },
)


@triton.jit
def _gate_terms(gate, up):
    # torch's silu and silu_backward: accurate expf, IEEE division.
    act = silu(gate)
    return sigmoid(gate), act, act * up


@triton.jit(do_not_specialize=["elements"])
def _forward_kernel(gate_up, h, elements, width: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    row_start = (offsets // width) * (2 * width) + offsets % width
    gate = tl.load(gate_up + row_start, mask=mask).to(tl.float32)
    up = tl.load(gate_up + row_start + width, mask=mask).to(tl.float32)
    _, _, value = _gate_terms(gate, up)
    tl.store(h + offsets, value.to(h.dtype.element_ty), mask=mask)


@triton.jit
def _slot(chunk, q, cols, rows_per_chunk, width, rank, BLOCK_I: tl.constexpr, SPARE: tl.constexpr):
    """FP32 offset of partial ``(chunk, q, col)``, ``q = section * rank + r`` (section 0: down A,
    1: gate B, 2: up B). ``SPARE``: inside the chunk's own rows of ``grad_h`` (FP32 view, rows
    of ``width / 2``): each program's ``2 * 3 * rank`` rows by ``BLOCK_I / 2`` FP32 columns are
    read only by that program, so its partials take them once its reads are done. Otherwise a
    separate ``[chunks, 3 * rank, width]`` buffer."""
    if SPARE:
        j = cols % BLOCK_I
        row = chunk.to(tl.int64) * rows_per_chunk + 2 * q + j // (BLOCK_I // 2)
        return row * (width // 2) + (cols - j) // 2 + j % (BLOCK_I // 2)
    return (chunk.to(tl.int64) * 3 * rank + q) * width + cols


@triton.jit(do_not_specialize=["rows", "rows_per_chunk"])
def _backward_kernel(
    grad_h,
    gate_up,
    grad_y_b,
    x_a,
    partial,
    rows,
    rows_per_chunk,
    width: tl.constexpr,
    rank: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_I: tl.constexpr,
    SPARE: tl.constexpr,
):
    """One column block of ``I`` over one chunk of rows: the gate and up gradients over the
    gate | up they are computed from, and this chunk's FP32 partials of the three factor
    gradients that reduce over rows: ``grad_y_b^T . h`` [rank, I] (``h`` recomputed in
    registers and rounded as the forward rounds it; never stored), ``d gate^T . x A_gate^T``
    and ``d up^T . x A_up^T`` [I, rank] (the stored, rounded gradients)."""
    pid_i = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    col_mask = cols < width
    ranks = tl.arange(0, RANK_BLOCK)
    rank_mask = ranks < rank
    first = chunk * rows_per_chunk
    last = tl.minimum(first + rows_per_chunk, rows)
    dtype = gate_up.dtype.element_ty
    down_a = tl.zeros((RANK_BLOCK, BLOCK_I), dtype=tl.float32)
    gate_b = tl.zeros((BLOCK_I, RANK_BLOCK), dtype=tl.float32)
    up_b = tl.zeros((BLOCK_I, RANK_BLOCK), dtype=tl.float32)
    for start in range(first, last, BLOCK_T):
        t = (start + tl.arange(0, BLOCK_T)).to(tl.int64)
        t_mask = t < last
        mask = t_mask[:, None] & col_mask[None, :]
        low_mask = t_mask[:, None] & rank_mask[None, :]
        merged = gate_up + t[:, None] * (2 * width) + cols[None, :]
        gate = tl.load(merged, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(merged + width, mask=mask, other=0.0).to(tl.float32)
        upstream = tl.load(grad_h + t[:, None] * width + cols[None, :], mask=mask, other=0.0)
        upstream = upstream.to(tl.float32)
        sigmoid, silu, value = _gate_terms(gate, up)
        grad_gate = (upstream * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))).to(dtype)
        grad_up = (upstream * silu).to(dtype)
        tl.store(merged, grad_gate, mask=mask)
        tl.store(merged + width, grad_up, mask=mask)
        low = tl.load(grad_y_b + t[:, None] * rank + ranks[None, :], mask=low_mask, other=0.0)
        x_gate = tl.load(x_a + t[:, None] * (2 * rank) + ranks[None, :], mask=low_mask, other=0.0)
        x_up = tl.load(
            x_a + t[:, None] * (2 * rank) + rank + ranks[None, :], mask=low_mask, other=0.0
        )
        if dtype == tl.float32:
            down_a += tl.dot(tl.trans(low), value, input_precision="ieee")
            gate_b += tl.dot(tl.trans(grad_gate), x_gate, input_precision="ieee")
            up_b += tl.dot(tl.trans(grad_up), x_up, input_precision="ieee")
        else:
            down_a += tl.dot(tl.trans(low), value.to(dtype))
            gate_b += tl.dot(tl.trans(grad_gate), x_gate)
            up_b += tl.dot(tl.trans(grad_up), x_up)
    down_slots = _slot(
        chunk, ranks[:, None], cols[None, :], rows_per_chunk, width, rank, BLOCK_I, SPARE
    )
    tl.store(partial + down_slots, down_a, mask=rank_mask[:, None] & col_mask[None, :])
    b_mask = col_mask[:, None] & rank_mask[None, :]
    gate_slots = _slot(
        chunk, rank + ranks[None, :], cols[:, None], rows_per_chunk, width, rank, BLOCK_I, SPARE
    )
    tl.store(partial + gate_slots, gate_b, mask=b_mask)
    up_slots = _slot(
        chunk, 2 * rank + ranks[None, :], cols[:, None], rows_per_chunk, width, rank, BLOCK_I, SPARE
    )
    tl.store(partial + up_slots, up_b, mask=b_mask)


@triton.jit
def _reduce_kernel(
    partial,
    out,
    chunks,
    rows_per_chunk,
    scaling,
    width: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_I: tl.constexpr,
    SPARE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``out`` [3, rank * width] (down A as [rank, width], gate and up B as [width, rank]) =
    ``scaling`` times the sum of the chunks' partials in chunk order (deterministic)."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    section_size = rank * width
    mask = offsets < 3 * section_size
    section = offsets // section_size
    rest = offsets % section_size
    r = tl.where(section == 0, rest // width, rest % rank)
    cols = tl.where(section == 0, rest % width, rest // rank)
    q = section * rank + r
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for chunk in range(chunks):
        slots = _slot(chunk, q, cols, rows_per_chunk, width, rank, BLOCK_I, SPARE)
        total += tl.load(partial + slots, mask=mask, other=0.0)
    tl.store(out + offsets, (total * scaling).to(out.dtype.element_ty), mask=mask)


IN_PLACE = True  # the backward overwrites gate_up


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` of a contiguous ``[T, 2I]`` gate | up, fresh ``[T, I]``."""

    width = gate_up.shape[-1] // 2
    h = gate_up.new_empty(gate_up.shape[0], width)
    elements = h.numel()
    if not elements:
        return h
    _forward_kernel[(triton.cdiv(elements, _BLOCK),)](
        gate_up, h, elements, width, _BLOCK, num_warps=8
    )
    return h


def swiglu_backward_(
    grad_h: torch.Tensor,
    gate_up: torch.Tensor,
    grad_y_b: torch.Tensor,
    x_a: torch.Tensor,
    grads: torch.Tensor,
    *,
    scaling: float,
    config: dict | None = None,
) -> torch.Tensor:
    """The gate | up gradient written over ``gate_up`` [T, 2I] (``grad_h`` is consumed: its
    storage may carry the partials), and ``grads`` [3, r * I] =
    ``scaling`` times ``grad_y_b^T . h`` ([r, I]; ``grad_y_b`` [T, r]), ``d gate^T . x_a[:, :r]``
    and ``d up^T . x_a[:, r:]`` ([I, r] each; ``x_a`` [T, 2r]); every operand contiguous.
    ``config`` (a :data:`TUNED` config) overrides the table, for the tuner."""

    rows, width = grad_h.shape
    rank = grad_y_b.shape[1]
    if not rows:
        grads.zero_()
        return gate_up
    if config is None:
        dtype = str(gate_up.dtype).removeprefix("torch.")
        config = TUNED.config(gate_up.device, t=rows, i=width, rank=rank, dtype=dtype)
    block_t, block_i = config["block_t"], config["block_i"]
    column_blocks = triton.cdiv(width, block_i)
    chunks = max(1, min(triton.cdiv(rows, block_t), config["programs"] // column_blocks))
    rows_per_chunk = triton.cdiv(triton.cdiv(rows, chunks), block_t) * block_t
    chunks = triton.cdiv(rows, rows_per_chunk)
    # The partials go into grad_h's spare rows when every chunk has 2 rows per partial row and
    # the column blocks tile I exactly (16-bit grad_h); else into a separate buffer.
    shortest = rows - (chunks - 1) * rows_per_chunk
    spare = (
        grad_h.element_size() == 2
        and width % block_i == 0
        and min(shortest, rows_per_chunk) >= 6 * rank
    )
    if spare:
        partial = grad_h.view(torch.float32)
    else:
        partial = torch.empty(chunks * grads.numel(), dtype=torch.float32, device=grad_h.device)
    _backward_kernel[(column_blocks, chunks)](
        grad_h,
        gate_up,
        grad_y_b,
        x_a,
        partial,
        rows,
        rows_per_chunk,
        width,
        rank,
        max(16, triton.next_power_of_2(rank)),
        block_t,
        block_i,
        spare,
        num_warps=config["warps"],
        num_stages=config["stages"],
    )
    _reduce_kernel[(triton.cdiv(grads.numel(), _BLOCK),)](
        partial,
        grads,
        chunks,
        rows_per_chunk,
        scaling,
        width,
        rank,
        block_i,
        spare,
        _BLOCK,
        num_warps=4,
    )
    return gate_up
