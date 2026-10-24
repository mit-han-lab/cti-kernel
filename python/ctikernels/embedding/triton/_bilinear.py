# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Bilinear position-embedding kernel and launch for
``ctikernels.embedding.inference.bilinear_pos_embed_``: one program per 16-patch, 128-channel
tile, FP32 math without FMA contraction."""

import triton
import triton.language as tl

_BLOCK_P = 16
_BLOCK_C = 128


@triton.jit
def _axis(index, inv, SIDE: tl.constexpr):
    """Both taps and weights of grid coordinate ``index`` along one axis."""
    source = (index.to(tl.float32) * (SIDE - 1)) * inv
    lower = tl.floor(source)
    fraction = source - lower
    # max(1 - |fraction - k|, 0) for k = 0, 1; fraction - 1 can round.
    weight0 = tl.maximum(1.0 - tl.abs(fraction), 0.0)
    weight1 = tl.maximum(1.0 - tl.abs(fraction - 1.0), 0.0)
    tap0 = tl.minimum(lower.to(tl.int32), SIDE - 1)
    tap1 = tl.minimum(lower.to(tl.int32) + 1, SIDE - 1)
    return tap0, tap1, weight0, weight1


@triton.jit
def _tap(Table, table_row, cell, column, mask, weight):
    row = tl.load(Table + cell.to(tl.int64)[:, None] * table_row + column[None, :], mask, 0.0)
    return row.to(tl.float32) * weight[:, None]


@triton.jit(do_not_specialize=["offset", "height", "width", "inv_h", "inv_w"])
def bilinear_kernel(Table, Out, table_row, out_row, columns, offset, height, width, inv_h,
                    inv_w, SIDE: tl.constexpr, MERGE: tl.constexpr, BLOCK_P: tl.constexpr,
                    BLOCK_C: tl.constexpr):  # fmt: skip
    patch = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    column = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (patch < height * width)[:, None] & (column < columns)[None, :]
    # Merge-window order: MERGE x MERGE windows row by row, each window's patches row by row.
    window, inner = patch // (MERGE * MERGE), patch % (MERGE * MERGE)
    windows = width // MERGE
    row0, row1, row_w0, row_w1 = _axis((window // windows) * MERGE + inner // MERGE, inv_h, SIDE)
    col0, col1, col_w0, col_w1 = _axis((window % windows) * MERGE + inner % MERGE, inv_w, SIDE)
    # Taps (r0, c0), (r0, c1), (r1, c0), (r1, c1): FP32 products, summed in that order.
    acc = _tap(Table, table_row, row0 * SIDE + col0, column, mask, row_w0 * col_w0)
    acc = acc + _tap(Table, table_row, row0 * SIDE + col1, column, mask, row_w0 * col_w1)
    acc = acc + _tap(Table, table_row, row1 * SIDE + col0, column, mask, row_w1 * col_w0)
    acc = acc + _tap(Table, table_row, row1 * SIDE + col1, column, mask, row_w1 * col_w1)
    out = Out + (offset + patch).to(tl.int64)[:, None] * out_row + column[None, :]
    tl.store(out, acc.to(Out.dtype.element_ty), mask)


def launch(table, out, offset, height, width, side, merge_size, inv_h, inv_w):
    """Launch over validated tensors (see ``ctikernels.embedding.inference.bilinear_pos_embed_``);
    ``inv_h``, ``inv_w`` are FP32 values."""
    columns = table.shape[1]
    block_c = min(_BLOCK_C, triton.next_power_of_2(columns))
    grid = (triton.cdiv(height * width, _BLOCK_P), triton.cdiv(columns, block_c))
    bilinear_kernel[grid](
        table, out, table.stride(0), out.stride(0), columns, offset, height, width, inv_h,
        inv_w, SIDE=side, MERGE=merge_size, BLOCK_P=_BLOCK_P, BLOCK_C=block_c, num_warps=4,
        enable_fp_fusion=False,
    )  # fmt: skip
