# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from vLLM 0.31.0 db9527a46873454610df6dbedf79a36d6bf1a7f6
# (`vllm/model_executor/models/qwen3_dflash2.py`); see THIRD_PARTY_NOTICES.

"""DFlash2 block convolution kernel and launch for ``ctikernels.speculative.block_conv``:
one program per row and 512-channel tile, FP32 math without FMA contraction."""

import triton
import triton.language as tl


@triton.jit
def block_conv_kernel(X, D, B, O, d_row, C: tl.constexpr, S: tl.constexpr,  # noqa: E741
                      Q: tl.constexpr, TAPS: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    n, tile = tl.program_id(0), tl.program_id(1)
    c = tile * BLOCK + tl.arange(0, BLOCK)
    valid = c < C
    segment = c // (C // S)
    position = n % Q
    row = n.to(tl.int64) * C
    coefficient = tl.load(B + c, valid, 0).to(tl.float32) + tl.load(
        D + n.to(tl.int64) * d_row + segment, valid, 0
    ).to(tl.float32)
    acc = coefficient * tl.load(X + row + c, valid, 0).to(tl.float32)
    for tap in tl.static_range(1, TAPS):
        coefficient = tl.load(B + tap * C + c, valid, 0).to(tl.float32) + tl.load(
            D + n.to(tl.int64) * d_row + tap * S + segment, valid, 0
        ).to(tl.float32)
        earlier = tl.load(X + row - tap * C + c, valid & (position >= tap), 0)
        # Skip (not +0) before the block start, so -0 stays -0 as in the composed form.
        acc = tl.where(position >= tap, acc + coefficient * earlier.to(tl.float32), acc)
    tl.store(O + row + c, acc.to(O.dtype.element_ty), valid)


_BLOCK_C = 512


def launch(x, deltas, base, out, block_size):
    """Launch over validated tensors (see ``ctikernels.speculative.block_conv``)."""
    n, c = x.shape
    taps, segments = deltas.shape[1], deltas.shape[2]
    block_conv_kernel[(n, triton.cdiv(c, _BLOCK_C))](
        x, deltas, base, out, deltas.stride(0), C=c, S=segments, Q=block_size, TAPS=taps,
        BLOCK=_BLOCK_C, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
