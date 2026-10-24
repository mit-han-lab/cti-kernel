# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Semantic reference: FlashInfer 0.7.0.post1 quantization_utils.cuh cvt_warp_fp16_to_mxfp8 and
# get_sf_out_offset_128x4 (the recipe and layout to match bit for bit); no upstream code copied.

"""Triton MXFP8 activation quantizer, bitwise equal to FlashInfer's CUDA quantizer: the same
approximate reciprocals, the UE8M0 round-up with its subnormal and saturation cases, and the
satfinite E4M3 conversion; row-major or 128 x 4 swizzled scales."""

import triton
import triton.language as tl


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_ftz(a, b):
    # FlashInfer builds its quantizer with -use_fast_math: products flush to zero.
    return tl.inline_asm_elementwise(
        "mul.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _ue8m0_up(value):
    """UE8M0 byte of the power of two at or above positive ``value`` (FlashInfer/CUDA
    ``cvt.rp.satfinite``): byte 0 is 2^-127; 254 saturates; NaN gives 255."""
    bits = value.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    up = (mantissa > 0) & (exponent != 254) & ~((exponent == 0) & (mantissa <= 0x400000))
    byte = exponent + up.to(tl.int32)
    return tl.where(value != value, 255, tl.minimum(byte, 255))


@triton.jit
def quantize_kernel(X, Q, S, M, K, X_STRIDE, K_TILES, SWIZZLED: tl.constexpr,
                    ROWS: tl.constexpr, GROUPS: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = (rows[:, None, None] < M) & (cols[None, :, :] < K)
    x = tl.load(X + rows[:, None, None].to(tl.int64) * X_STRIDE + cols[None, :, :], mask, 0)
    amax = tl.max(tl.abs(x), 2).to(tl.float32)  # exact in the input dtype
    byte = _ue8m0_up(_mul_ftz(amax, _rcp_approx_ftz(tl.full(amax.shape, 448.0, tl.float32))))
    scale = tl.where(byte == 0, 0x00400000, byte << 23).to(tl.float32, bitcast=True)
    # FlashInfer's "scale != 0" compares with FTZ: byte 0 (the denormal 2^-127) counts as zero.
    factor = tl.where(byte != 0, _rcp_approx_ftz(scale), 0.0)
    y = _mul_ftz(x.to(tl.float32), tl.broadcast_to(factor[:, :, None], x.shape))
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    tl.store(Q + rows[:, None, None].to(tl.int64) * K + cols[None, :, :], y.to(tl.float8e4nv), mask)
    gmask = (rows[:, None] < M) & (groups[None, :] * 32 < K)
    if SWIZZLED:
        # The padding rows (to 128) and groups (to 4) hold zero bytes.
        byte = tl.where(gmask, byte, 0)
        gmask = groups[None, :] < K_TILES * 4
        # [M tiles, K tiles, 32 (row % 32), 4 (row % 128 // 32), 4 (group % 4)]
        offset = (
            (rows[:, None] // 128) * K_TILES * 512
            + (groups[None, :] // 4) * 512
            + (rows[:, None] % 32) * 16
            + (rows[:, None] % 128 // 32) * 4
            + groups[None, :] % 4
        )
        tl.store(S + offset.to(tl.int64), byte.to(tl.uint8), gmask)
    else:
        tl.store(
            S + rows[:, None].to(tl.int64) * (K // 32) + groups[None, :], byte.to(tl.uint8), gmask
        )


def quantize(x, q, scale, *, swizzled: bool) -> None:
    """Fill ``q`` [M, K] E4M3 and ``scale`` (uint8 view: [M, K/32] or flat swizzled) from
    ``x`` [M, K] (unit column stride, K % 32 == 0)."""
    m, k = x.shape
    rows, groups = 8, 32
    k_tiles = triton.cdiv(k // 32, 4)
    padded_m = triton.cdiv(m, 128) * 128 if swizzled else m
    grid = (triton.cdiv(padded_m, rows), triton.cdiv(k_tiles * 4, groups))
    quantize_kernel[grid](x, q, scale, m, k, x.stride(0), k_tiles, swizzled, rows, groups,
                          num_warps=4)  # fmt: skip
