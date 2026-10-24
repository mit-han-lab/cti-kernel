# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FP32 values rounded to BF16 or FP16 (nearest even) and back, kept in registers. Triton folds
an in-register ``x.to(dtype).to(tl.float32)`` round trip and contracts the arithmetic around it
(one FMA, no rounding), so a kernel that rounds and keeps computing in FP32, or stores the
rounded value to FP32 memory, takes the round trip as inline PTX instead."""

import os

import triton
import triton.language as tl

# Triton's CPU interpreter: no inline PTX (nor the fold); its FP32 -> BF16 cast truncates, so
# BF16 rounds in software there. Kernels with interpreter fallbacks read this one flag.
INTERPRET = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")


@triton.jit
def _cvt(x, ASM: tl.constexpr):
    return tl.inline_asm_elementwise(ASM, "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def round_to(x, DTYPE: tl.constexpr):
    """``x`` (FP32) rounded to ``DTYPE`` (BF16, FP16, or FP32: unchanged), as FP32."""
    if DTYPE == tl.float32:
        return x
    if INTERPRET:
        if DTYPE == tl.bfloat16:  # nearest even on the bits
            b = x.to(tl.int32, bitcast=True)
            return ((b + 0x7FFF + ((b >> 16) & 1)) & -65536).to(tl.float32, bitcast=True)
        return x.to(DTYPE).to(tl.float32)
    if DTYPE == tl.bfloat16:
        return _cvt(x, "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }")
    return _cvt(x, "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }")


@triton.jit
def bf16_round(x):
    return round_to(x, tl.bfloat16)
