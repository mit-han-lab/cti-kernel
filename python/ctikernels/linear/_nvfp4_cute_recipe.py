# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Recipe follows FlashInfer 0.7.0.post1 (csrc/nv_internal/tensorrt_llm/kernels/
# quantization_utils.cuh, cvt_warp_fp16_to_fp4), design reference only (Apache-2.0).

"""Shared CuTe scale/pack arithmetic, with explicit public PTX rounding/FTZ."""

import cutlass
from cutlass import Float8E4M3FN, Float32, Uint8, Uint32, cute

from . import _nvfp4_recipe as ptx

_DIV = ptx.cute_asm(ptx.DIV, 1)
_RCP = ptx.cute_asm(ptx.RCP, 1)
_MUL = ptx.cute_asm(ptx.MUL, 2)
_MAX = ptx.cute_asm(ptx.MAX, 2)
_NONZERO = ptx.cute_asm(ptx.NONZERO, 1)
_E4M3 = ptx.cute_asm(ptx.E4M3, 1)
_E2M1 = ptx.cute_asm(ptx.E2M1, 2)


@cute.jit
def reciprocal(value: Float32):
    """Only global descales use IEEE division (matching torch 1/input_scale)."""
    return cute.arch.inline_ptx(_DIV, write_only_types=[Float32], read_only_args=[value])


@cute.jit
def rcp(value: Float32):
    return cute.arch.inline_ptx(_RCP, write_only_types=[Float32], read_only_args=[value])


@cute.jit
def mul(a: Float32, b: Float32):
    return cute.arch.inline_ptx(_MUL, write_only_types=[Float32], read_only_args=[a, b])


@cute.jit
def maximum(a: Float32, b: Float32):
    return cute.arch.inline_ptx(_MAX, write_only_types=[Float32], read_only_args=[a, b])


@cute.jit
def amax_nonzero(value: Float32):
    return cute.arch.inline_ptx(_NONZERO, write_only_types=[Uint32], read_only_args=[value])


@cute.jit
def scale(amax: Float32, inverse: Float32):
    sf_f32 = mul(inverse, mul(amax, rcp(Float32(6.0))))
    bits = cute.arch.inline_ptx(_E4M3, write_only_types=[Uint32], read_only_args=[sf_f32])
    sf = Uint8(bits).bitcast(Float8E4M3FN)
    factor = rcp(mul(sf.to(Float32), rcp(inverse)))
    factor = Float32(cutlass.select_(amax_nonzero(amax) != Uint32(0), factor, Float32(0.0)))
    return sf, factor


@cute.jit
def pack(low: Float32, high: Float32):
    """Convert the products directly: hardware handles NaNs and signed zeros."""
    return Uint8(
        cute.arch.inline_ptx(
            _E2M1,
            write_only_types=[Uint32],
            read_only_args=[low, high],
        )
    )
