# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Recipe follows FlashInfer 0.7.0.post1 (csrc/nv_internal/tensorrt_llm/kernels/
# quantization_utils.cuh, cvt_warp_fp16_to_fp4), design reference only (Apache-2.0).

"""Public PTX primitives for the market NVFP4 recipe; no provider imports.

The plain-input max is exact after widening BF16/FP16 to FP32: no rounding or
FTZ is allowed in that reduction. Starting with an input (not zero) preserves
all-NaN blocks. Gated values stay FP32 and use the same NaN-ignoring max.
"""

DIV = "div.rn.f32 $0, 0f3f800000, $1;"
RCP = "rcp.approx.ftz.f32 $0, $1;"
MUL = "mul.ftz.f32 $0, $1, $2;"
MAX = "max.f32 $0, $1, $2;"
# H1: the FlashInfer fast-math comparison flushes subnormal amax. Flip only
# this instruction to setp.ne.f32 if GPU qualification disproves H1.
NONZERO = "{ .reg .pred p; setp.ne.ftz.f32 p, $1, 0f00000000; selp.u32 $0, 1, 0, p; }"
E4M3 = "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair, $1, $1; cvt.u32.u16 $0, pair; }"
E2M1 = (
    "{ .reg .b8 byte; cvt.rn.satfinite.e2m1x2.f32 byte, $2, $1; "
    "mov.b32 $0, {byte, byte, byte, byte}; }"
)


def native_e2m1(capability):
    """Whether the device has the hardware E2M1 cvt (SM100+). Older devices (8.9 <= cc < 10)
    take the software rounding of ``_nvfp4_triton._e2m1_soft``, bitwise equal to it."""
    return tuple(capability) >= (10, 0)


def cute_asm(asm, inputs):
    """Translate operand placeholders, preserving literal PTX register braces."""
    asm = asm.replace("$0", "{$w0}")
    for i in range(inputs):
        asm = asm.replace(f"${i + 1}", "{$r" + str(i) + "}")
    return asm
