# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Semantic reference: DeepSeek-V4-Flash 60d8d707 inference/kernel.py (act_quant, fp4_act_quant);
# no upstream code copied.

"""Device helpers shared by the compressor and the indexer: partial RoPE, the 128-point
Hadamard transform, FP4/FP8 row quantizers with UE8M0 scales and the page writers."""

import triton
import triton.language as tl

# Latent rows (the DSV4 attention page ABI): 448 E4M3 + 64 BF16 = 576 data bytes per row, then
# 8 footer bytes per row (7 UE8M0 exponents + 1 zero), the page padded to a multiple of 576.
LATENT_DATA = 576
LATENT_FOOTER = 8
# Index rows: 64 bytes of E2M1 pairs per row, then 4 UE8M0 bytes per row; 64 rows per page.
INDEX_ROWS = 64
INDEX_PAGE = 4352


@triton.jit
def rotate(x, pos, Rope, D: tl.constexpr):
    """Rotate the trailing 64 channels of ``x [D]`` in adjacent pairs at ``pos``; ``Rope`` is
    FP32 ``[positions, 64]``: 32 cosines, then 32 sines."""
    d = tl.arange(0, D)
    partner = tl.gather(x, d ^ 1, 0)
    c = tl.load(Rope + pos * 64 + (d - (D - 64)) // 2, d >= D - 64, 1)
    s = tl.load(Rope + pos * 64 + 32 + (d - (D - 64)) // 2, d >= D - 64, 0)
    rotated = x * c + tl.where(d % 2 == 0, -partner, partner) * s
    return tl.where(d < D - 64, x, rotated)


@triton.jit
def hadamard128(x):
    """Orthonormal Sylvester-order Walsh-Hadamard transform of ``x [128]``, rounded to BF16."""
    d = tl.arange(0, 128)
    for stage in tl.static_range(7):
        step = 1 << stage
        other = tl.gather(x, d ^ step, 0)
        x = tl.where(d & step == 0, x + other, other - x)
    return (x * (128**-0.5)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def ue8m0(exponent):
    """Exact ``2^(e - 127)`` of UE8M0 bytes, without FTZ: byte 0 is the subnormal 2^-127."""
    e = exponent.to(tl.int32)
    bits = tl.where(e == 0, 0x00400000, e << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def ceil_exponent(raw):
    """Biased-free exponent of the smallest power of two >= ``raw`` (positive, finite)."""
    bits = raw.to(tl.int32, bitcast=True)
    return ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@triton.jit
def fp4(x):
    """E2M1 codes ``[128]`` and UE8M0 bytes ``[4]`` of ``x [128]``, one scale per 32."""
    v = tl.reshape(x, (4, 32))
    exp = ceil_exponent(tl.maximum(tl.max(tl.abs(v), 1), 6 * 2.0**-126) * (1 / 6))
    a = tl.abs(v / ue8m0(exp + 127)[:, None])
    # Round to nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even.
    code = (a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
    code += (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32)
    code += (a >= 3.5).to(tl.int32) + (a > 5).to(tl.int32)
    code |= (v.to(tl.int32, bitcast=True) >> 28) & 8
    return tl.reshape(code, (128,)), exp + 127


@triton.jit
def decode4(code):
    m = code & 7
    value = tl.where(
        m < 4, m * 0.5, tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0)))
    )
    return tl.where(code & 8 != 0, -value, value)


@triton.jit
def write_latent(x, Cache, slot, ROWS: tl.constexpr, PAGE: tl.constexpr):
    """Store latent ``x [512]`` (post norm and RoPE) as row ``slot`` of ROWS-row pages."""
    page, row = (slot // ROWS).to(tl.int64), slot % ROWS
    base = Cache + page * PAGE
    d = tl.arange(0, 512)
    groups = tl.reshape(tl.where(d < 448, x, 0), (8, 64))
    exponent = ceil_exponent(tl.maximum(tl.max(tl.abs(groups), 1), 1e-4) * (1 / 448))
    quant = tl.minimum(tl.maximum(groups / ue8m0(exponent + 127)[:, None], -448), 448)
    data = tl.reshape(quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True), (512,))
    tl.store(base + row * 576 + d, data, d < 448)
    rope = x.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    tl.store(base + row * 576 + 448 + (d - 448) * 2, (rope & 255).to(tl.uint8), d >= 448)
    tl.store(base + row * 576 + 449 + (d - 448) * 2, (rope >> 8).to(tl.uint8), d >= 448)
    g = tl.arange(0, 8)
    tl.store(base + ROWS * 576 + row * 8 + g, tl.where(g < 7, exponent + 127, 0).to(tl.uint8))


@triton.jit
def write_index(x, Cache, slot):
    """Store index key ``x [128]`` (post Hadamard) as row ``slot`` of 64-row FP4 pages."""
    page, row = (slot // 64).to(tl.int64), slot % 64
    code, exponent = fp4(x)
    i = tl.arange(0, 64)
    payload = tl.gather(code, i * 2, 0) | (tl.gather(code, i * 2 + 1, 0) << 4)
    tl.store(Cache + page * 4352 + row * 64 + i, payload.to(tl.uint8))
    tl.store(Cache + page * 4352 + 4096 + row * 4 + tl.arange(0, 4), exponent.to(tl.uint8))
