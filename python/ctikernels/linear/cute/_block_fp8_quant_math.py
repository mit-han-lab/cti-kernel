# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 SGLang Team
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Extracted from our _block_fp8_act_quant.py, retaining its arithmetic lineage.
# Adapted from SGLang 2428d0f531602b882892f9f29521cd07ddd3c959 and vLLM
# f4b8d43a681a908ce22b63ece3a86ab5b198e760; see THIRD_PARTY_NOTICES.

"""Shared device arithmetic for activation quantization and fused FC1.

Imported only while compiling. Keep the explicit PTX rounding/FTZ boundaries.
Reference design: package CuTe activation quantizer (SGLang/vLLM semantics).
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint16


def fp32_op(instruction, a, b):
    return cute.arch.inline_ptx(
        instruction + " {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


@cute.jit
def silu_product(g, u, limit: cutlass.Constexpr):
    if cutlass.const_expr(limit is not None):
        bound = Float32(Float32(limit).to(BFloat16))
        g = fp32_op("min.f32", Float32(g.to(BFloat16)), bound)
        u = fp32_op("max.f32", Float32(u.to(BFloat16)), -bound)
        u = fp32_op("min.f32", u, bound)
    # SGLang's activation: x / (1 + expf(-x)) * up (accurate exp, IEEE divide).
    return fp32_op("div.rn.f32", g, 1.0 + cute.math.exp(-g, fastmath=False)) * u


@cute.jit
def quant_scale(amax, ue8m0: cutlass.Constexpr):
    amax = fp32_op("max.f32" if ue8m0 else "max.NaN.f32", amax, Float32(1e-10))
    if amax != amax:
        amax = Float32(float("nan"))
    stored = fp32_op("mul.rn.f32", amax, Float32(1.0 / 448.0))
    multiplier = fp32_op("div.rn.f32", Float32(448.0), amax)
    if cutlass.const_expr(ue8m0):
        stored, multiplier = cute.arch.inline_ptx(
            "{ .reg .b32 b, e, m, inv; .reg .pred p;\n"
            "mov.b32 b, {$r0}; shr.u32 e, b, 23;\n"
            "and.b32 m, b, 0x7fffff; setp.ne.u32 p, m, 0;\n"
            "@p add.u32 e, e, 1; sub.u32 inv, 254, e;\n"
            "shl.b32 e, e, 23; shl.b32 inv, inv, 23;\n"
            "mov.b32 {$w0}, e; mov.b32 {$w1}, inv; }",
            write_only_types=[Float32, Float32],
            read_only_args=[stored],
        )
    return stored, multiplier


@cute.jit
def quant_value(value, multiplier, ue8m0: cutlass.Constexpr):
    value = fp32_op("mul.rn.f32", value, multiplier)
    value = fp32_op("max.f32" if ue8m0 else "max.NaN.f32", value, Float32(-448.0))
    return fp32_op("min.f32" if ue8m0 else "min.NaN.f32", value, Float32(448.0))


@cute.jit
def pack_pair(lo, hi):
    return cute.arch.inline_ptx(
        "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Uint16],
        read_only_args=[hi, lo],
    )
