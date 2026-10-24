# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""NVFP4 W4A16 input gradient ``dx = alpha * dy . W`` for a weight ``W [N, K]`` stored
as NVFP4 (``data [N, K / 2]``, E4M3 ``scale [N, K / 16]``), read in place.

The reduction runs over the weight's rows, across its scale blocks, so no repacked
copy can keep the checkpoint's blocks. Every program instead decodes ``[BLOCK_N, BLOCK_K]``
tiles of ``W`` (even and odd columns apart, so no interleaving shuffle) to BF16, which is
exact (an E2M1 value times an E4M3 scale has at most 6 significant bits). FP32 ``dy``
enters as ``terms`` BF16 terms: 1 rounds it once, 2 leave a 2^-16 relative remainder,
3 sum to it exactly (normal range). BF16/FP16 ``dy`` multiplies the weight decoded to its
own dtype. Products accumulate in FP32; ``alpha`` (FP32 ``[1]``, read on the device)
multiplies the sum, which is rounded once to the output dtype.

With few output tiles the rows are split over ``splits`` programs that store FP32
partials; a second kernel adds them in split order, so results do not depend on
timing. Configurations depend only on the shape: equal inputs give equal outputs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

MAX_SPLITS = 16
# The largest program tile (rows x output columns); bounds the FP32 split partials.
MAX_TILE = 256 * 128


class Config(NamedTuple):
    """Program tile (rows, weight rows per step, output columns), warps, stages, splits."""

    block_m: int
    block_n: int
    block_k: int
    warps: int
    stages: int
    splits: int


def config_for(m: int, k: int, n: int, sms: int) -> Config:
    """The tile for ``m`` rows (RTX PRO 6000 sweeps at the 27B head, 248320 x 5120): above 128
    rows one 256-row tile decodes each weight tile once per 256 rows. Rows are split so the
    programs fill one wave of 256-row tiles or two of smaller ones (``splits * tiles *
    block_m * block_k <= sms * MAX_TILE``, hence ``partial_count``)."""
    if m > 128:
        bm, bn, bk, warps, stages, waves = 256, 32, 128, 8, 3, 1
    else:
        bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
        bn, bk, warps, stages, waves = 64, 128, 4 if bm <= 32 else 8, 3, 2
    tiles = triton.cdiv(m, bm) * triton.cdiv(k, bk)
    splits = max(1, min(MAX_SPLITS, waves * sms // tiles, triton.cdiv(n, bn)))
    return Config(bm, bn, bk, warps, stages, splits)


def partial_count(sms: int) -> int:
    """FP32 partial elements any split call needs on a device with ``sms`` SMs."""
    return sms * MAX_TILE


@triton.jit
def _decode_e2m1(code):
    # E2M1 to FP32 bits: 0.5 * m when the exponent field is zero, else 2^(e-1) * (1 + m/2).
    sign = (code >> 3).to(tl.uint32) << 31
    e = ((code >> 1) & 3).to(tl.uint32)
    m = (code & 1).to(tl.uint32)
    bits = tl.where(e == 0, m * (126 << 23), ((e + 126) << 23) | (m << 22))
    return (bits | sign).to(tl.float32, bitcast=True)


@triton.jit
def _decode_e4m3(byte):
    # Positive E4M3 to FP32: subnormals m * 2^-9, normals 2^(e-7) * (1 + m/8).
    e = ((byte >> 3) & 15).to(tl.uint32)
    m = (byte & 7).to(tl.uint32)
    normal = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    return tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)


@triton.jit
def input_grad_kernel(
    dy,
    data,
    scale,
    alpha,
    out,
    partials,
    m,
    n,
    k,
    stride_dy,
    SPLITS: tl.constexpr,
    TERMS: tl.constexpr,
    DOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Byte columns: each holds output columns 2j (low nibble) and 2j + 1 (high nibble),
    # decoded into separate tiles with separate accumulators (no interleaving shuffle).
    pairs = pid_k * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    row_mask = rows < m
    pair_mask = pairs < k // 2
    steps = tl.cdiv(n, BLOCK_N)
    per_split = tl.cdiv(steps, SPLITS)
    first = pid_s * per_split
    last = tl.minimum(first + per_split, steps)
    even = tl.zeros((BLOCK_M, BLOCK_K // 2), dtype=tl.float32)
    odd = tl.zeros((BLOCK_M, BLOCK_K // 2), dtype=tl.float32)
    for step in range(first, last):
        wrows = step * BLOCK_N + tl.arange(0, BLOCK_N)
        wmask = wrows < n
        a = tl.load(
            dy + rows[:, None].to(tl.int64) * stride_dy + wrows[None, :],
            mask=row_mask[:, None] & wmask[None, :],
            other=0.0,
        )
        tile_mask = wmask[:, None] & pair_mask[None, :]
        packed = tl.load(
            data + wrows[:, None].to(tl.int64) * (k // 2) + pairs[None, :], mask=tile_mask, other=0
        )
        # The scale of byte column j is block j // 8 (16 output columns per block).
        block_scale = _decode_e4m3(
            tl.load(
                scale + wrows[:, None].to(tl.int64) * (k // 16) + (pairs // 8)[None, :],
                mask=tile_mask,
                other=0,
            )
        )
        w_even = (_decode_e2m1(packed & 15) * block_scale).to(DOT)
        w_odd = (_decode_e2m1(packed >> 4) * block_scale).to(DOT)
        if TERMS == 1:
            term = a.to(DOT)
            even = tl.dot(term, w_even, even)
            odd = tl.dot(term, w_odd, odd)
        else:
            term = a.to(tl.bfloat16)
            rest = a - term.to(tl.float32)
            even = tl.dot(term, w_even, even)
            odd = tl.dot(term, w_odd, odd)
            if TERMS == 3:
                term = rest.to(tl.bfloat16)
                rest = rest - term.to(tl.float32)
                even = tl.dot(term, w_even, even)
                odd = tl.dot(term, w_odd, odd)
            term = rest.to(tl.bfloat16)
            even = tl.dot(term, w_even, even)
            odd = tl.dot(term, w_odd, odd)
    out_mask = row_mask[:, None] & pair_mask[None, :]
    if SPLITS == 1:
        scale_out = tl.load(alpha)
        base = out + rows[:, None].to(tl.int64) * k + 2 * pairs[None, :]
        tl.store(base, (even * scale_out).to(out.dtype.element_ty), out_mask)
        tl.store(base + 1, (odd * scale_out).to(out.dtype.element_ty), out_mask)
    else:
        base = partials + (pid_s * m + rows[:, None]).to(tl.int64) * k + 2 * pairs[None, :]
        tl.store(base, even, out_mask)
        tl.store(base + 1, odd, out_mask)


@triton.jit
def _dequant_kernel(data, scale, out, pairs, k, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < pairs
    packed = tl.load(data + i, mask=mask, other=0)
    row_scale = scale + (i // (k // 2)) * (k // 16) + i % (k // 2) // 8
    s = _decode_e4m3(tl.load(row_scale, mask=mask, other=0))
    tl.store(out + 2 * i, (_decode_e2m1(packed & 15) * s).to(out.dtype.element_ty), mask)
    tl.store(out + 2 * i + 1, (_decode_e2m1(packed >> 4) * s).to(out.dtype.element_ty), mask)


def dequant(data, scale, out) -> None:
    """``out [R, K]`` = the exact BF16/FP16 values of NVFP4 rows ``data [R, K / 2]``, ``scale [R,
    K / 16]`` (no global)."""
    pairs = data.numel()
    block = 1024
    _dequant_kernel[(triton.cdiv(pairs, block),)](
        data, scale.view(dtype=data.dtype), out, pairs, out.shape[1], BLOCK=block
    )


@triton.jit
def _scale_kernel(src, alpha, dst, cols, src_stride, dst_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(src + row * src_stride + offsets, mask=mask) * tl.load(alpha)
    tl.store(dst + row * dst_stride + offsets, values.to(dst.dtype.element_ty), mask)


def scale_into(src, alpha, dst) -> None:
    """``dst [M, C] = round(alpha * src)`` for FP32 ``src``; both with unit inner stride."""
    m, cols = src.shape
    block = 1024
    _scale_kernel[(m, triton.cdiv(cols, block))](
        src, alpha, dst, cols, src.stride(0), dst.stride(0), BLOCK=block
    )


@triton.jit
def _cast_kernel(src, dst, n, src_stride, dst_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n
    values = tl.load(src + row * src_stride + cols, mask=mask)
    tl.store(dst + row * dst_stride + cols, values.to(tl.bfloat16), mask)


def cast_bf16(src, dst) -> None:
    """``dst [M, N] = BF16(src)``, rounded to nearest even like the kernel's one-term load;
    both with unit inner stride."""
    m, n = src.shape
    block = 2048
    _cast_kernel[(m, triton.cdiv(n, block))](src, dst, n, src.stride(0), dst.stride(0), BLOCK=block)


@triton.jit
def reduce_kernel(partials, alpha, out, total, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    acc = tl.load(partials + offsets, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLITS):
        acc += tl.load(partials + s * total + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, (acc * tl.load(alpha)).to(out.dtype.element_ty), mask)


def input_grad(dy, data, scale, alpha, out, partials, terms: int, config: Config) -> None:
    """``out [M, K] = alpha * dy [M, N] . W``; ``partials`` holds ``splits * M * K`` FP32
    elements when ``config.splits > 1``."""
    m, n = dy.shape
    k = out.shape[1]
    bm, bn, bk, warps, stages, splits = config
    grid = (triton.cdiv(m, bm), triton.cdiv(k, bk), splits)
    input_grad_kernel[grid](
        dy,
        data,
        scale.view(dtype=data.dtype),
        alpha,
        out,
        partials,
        m,
        n,
        k,
        dy.stride(0),
        SPLITS=splits,
        TERMS=terms,
        DOT=tl.float16 if dy.dtype == torch.float16 else tl.bfloat16,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    if splits > 1:
        total = m * k
        block = 1024
        reduce_kernel[(triton.cdiv(total, block),)](
            partials, alpha, out, total, SPLITS=splits, BLOCK=block
        )
