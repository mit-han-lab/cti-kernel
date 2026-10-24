# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from SGLang 023288645b80fb41b3eed55fd413dd69a7904593
# (`python/sglang/srt/layers/quantization/fp8_kernel.py`) and the sgl-kernel group quantizer at
# ae848116662ece923501131ce773a4602223237e; see THIRD_PARTY_NOTICES.

"""Triton kernels of the 128x128 / 1x128 block-FP8 contract (see ``block_fp8``).

Every kernel uses E4M3 conversions and E4M3 tensor-core dots: compute capability 8.9 and
newer (the package floor). Products are formed one 128-column scale group at a time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..._runtime.launch import FastLaunch
from ..._runtime.tables import device_class
from ...collective.triton.exchange import publish
from ...triton.silu import silu
from .. import _block_fp8_dispatch
from ._block_fp8_tuning import tuned_config

GROUP = 128  # activation group width and weight block edge
# The quantizer's recipe: amax = max(|x| over the group, 1e-10), the stored (multiplying) scale
# is amax * fp32(1 / 448) and the quantized value is E4M3_RN(clamp(x * fp32(448 / amax))).
# This is sgl-kernel's per-token-group quantizer bit for bit on finite inputs.
_GROUP = tl.constexpr(GROUP)
_E4M3_MAX = tl.constexpr(448.0)
_AMAX_FLOOR = tl.constexpr(1e-10)
_INV_E4M3_MAX = tl.constexpr(1.0 / 448.0)


@triton.jit
def _group_scales(x, valid):
    """Per-row FP32 (stored scale, multiplier) of a [rows, 128] tile. A row holding NaN or
    Inf gets a non-finite scale and NaN multiplier, so it never turns into finite values."""
    magnitude = tl.where(valid, tl.abs(x), 0.0)
    amax = tl.maximum(tl.max(magnitude, axis=1), _AMAX_FLOOR)
    bad = tl.max(tl.where(valid & (x != x), 1, 0), axis=1) > 0
    amax = tl.where(bad, float("nan"), amax)
    return amax * _INV_E4M3_MAX, libdevice.div_rn(tl.zeros_like(amax) + _E4M3_MAX, amax)


@triton.jit
def _max_propagating_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _loaded_group_scales(x):
    """``_group_scales`` of a tile whose masked elements were loaded as 0: no masking, and
    NaN carried by the amax reduction itself (max is exact, so finite rows match bit for bit)."""
    amax = tl.reduce(tl.abs(x), 1, _max_propagating_nan)
    amax = tl.maximum(amax, _AMAX_FLOOR, propagate_nan=tl.PropagateNan.ALL)
    amax = tl.where(amax != amax, float("nan"), amax)
    return amax * _INV_E4M3_MAX, libdevice.div_rn(tl.zeros_like(amax) + _E4M3_MAX, amax)


@triton.jit
def _saturate(v):
    v = tl.maximum(v, -_E4M3_MAX, propagate_nan=tl.PropagateNan.ALL)
    return tl.minimum(v, _E4M3_MAX, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _ue8m0_scales(x):
    """(stored, multiplier) of each row: the ceil-power-of-two of amax / 448 and its inverse."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), _AMAX_FLOOR)
    bits = (amax * _INV_E4M3_MAX).to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    stored = (exponent << 23).to(tl.float32, bitcast=True)
    return stored, ((254 - exponent) << 23).to(tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["M"])  # M only bounds masks: one compile per tile
def _quantize_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    M,
    K,
    stride_xm,
    stride_qm,
    stride_sm,
    BLOCK_M: tl.constexpr,
    UE8M0: tl.constexpr = False,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = group * _GROUP + tl.arange(0, _GROUP)
    row_mask = rows < M
    valid = row_mask[:, None] & (cols[None, :] < K)
    rows64 = rows.to(tl.int64)
    x = tl.load(x_ptr + rows64[:, None] * stride_xm + cols[None, :], mask=valid, other=0.0)
    x = x.to(tl.float32)
    if UE8M0:  # the activation kernel's power-of-two scales (NaN-ignoring maxima)
        stored, multiplier = _ue8m0_scales(x)
        v = tl.minimum(tl.maximum(x * multiplier[:, None], -_E4M3_MAX), _E4M3_MAX)
    else:
        stored, multiplier = _loaded_group_scales(x)
        v = _saturate(x * multiplier[:, None])
    q_ptrs = q_ptr + rows64[:, None] * stride_qm + cols[None, :]
    tl.store(q_ptrs, v.to(q_ptr.dtype.element_ty), mask=valid)
    tl.store(scale_ptr + rows64 * stride_sm + group, stored, mask=row_mask)


@triton.jit(do_not_specialize=["M", "stride_sm", "stride_sg"])
def _silu_quantize_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    act_ptr,
    M,
    width,
    stride_xm,
    stride_qm,
    stride_sm,
    stride_sg,
    stride_am,
    limit,
    BLOCK_M: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    ACT_OUT: tl.constexpr,
    UE8M0: tl.constexpr,
):
    """q, scale = the 1x128 quantization of SiLU(gate) * up, gate-first ``x`` [M, 2 width].

    The product is rounded to ``x``'s dtype first (the materialized two-kernel result). With
    ``HAS_LIMIT`` gate and up are rounded to BF16 and clamped to <= limit / [-limit, limit]
    (DSV4). ``ACT_OUT`` also stores the rounded product to ``act``. ``UE8M0`` takes NaN-ignoring maxima and power-of-two scales (ceil of amax / 448)."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = group * _GROUP + tl.arange(0, _GROUP)
    row_mask = rows < M
    valid = row_mask[:, None] & (cols[None, :] < width)
    rows64 = rows.to(tl.int64)
    x_ptrs = x_ptr + rows64[:, None] * stride_xm + cols[None, :]
    gate = tl.load(x_ptrs, mask=valid, other=0.0)
    up = tl.load(x_ptrs + width, mask=valid, other=0.0)
    g, u = gate.to(tl.float32), up.to(tl.float32)
    if HAS_LIMIT:
        g = tl.minimum(g.to(tl.bfloat16).to(tl.float32), limit)
        u = tl.minimum(tl.maximum(u.to(tl.bfloat16).to(tl.float32), -limit), limit)
    # SGLang's activation, as the CuTe quantizer computes it: g / (1 + expf(-g)) * u
    # (accurate exp, IEEE divide), so the two providers agree bit for bit.
    v = silu(g) * u
    v = v.to(gate.dtype)
    if ACT_OUT:  # the rounded product, e.g. for a LoRA down-projection
        tl.store(act_ptr + rows64[:, None] * stride_am + cols[None, :], v, mask=valid)
    v = v.to(tl.float32)
    if UE8M0:
        stored, multiplier = _ue8m0_scales(v)
        value = tl.minimum(tl.maximum(v * multiplier[:, None], -_E4M3_MAX), _E4M3_MAX)
    else:
        stored, multiplier = _loaded_group_scales(v)
        value = _saturate(v * multiplier[:, None])
    q_ptrs = q_ptr + rows64[:, None] * stride_qm + cols[None, :]
    tl.store(q_ptrs, value.to(q_ptr.dtype.element_ty), mask=valid)
    tl.store(scale_ptr + rows64 * stride_sm + group * stride_sg, stored, mask=row_mask)


@triton.jit
def _lora_epilogue(
    accumulator,
    out_ptr,
    rows64,
    cols,
    row_mask,
    col_mask,
    low_ptr,
    b_ptrs,
    slices_ptr,
    num_slices,
    rank,
    b_ld,
    stride_lm,
    stride_ls,
    LORA_R: tl.constexpr,
):
    """accumulator += low[rows, s, :rank] . B_s[cols - offset_s, :rank]^T for every slice s
    that the tile's columns touch, in FP32. B has the output's dtype. A BF16/FP16 ``low`` is the scaled, rounded
    low-rank tile; an FP32 one (a row-parallel partial) is not rounded to one 16-bit value:
    it enters as hi + lo, ~16 significant bits, far inside the TP accuracy bound."""
    first, last = tl.min(tl.where(col_mask, cols, 2**31 - 1)), tl.max(tl.where(col_mask, cols, -1))
    for s in range(0, num_slices):
        offset = tl.load(slices_ptr + 2 * s)
        width = tl.load(slices_ptr + 2 * s + 1)
        if (offset <= last) & (first < offset + width):
            n = cols - offset
            n_mask = col_mask & (n >= 0) & (n < width)
            b_ptr = tl.load(b_ptrs + s).to(tl.pointer_type(out_ptr.dtype.element_ty))
            for r0 in range(0, rank, LORA_R):
                r = r0 + tl.arange(0, LORA_R)
                r_mask = r < rank
                low = tl.load(
                    low_ptr + rows64[:, None] * stride_lm + s * stride_ls + r[None, :],
                    mask=row_mask[:, None] & r_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + n.to(tl.int64)[None, :] * b_ld + r[:, None],
                    mask=n_mask[None, :] & r_mask[:, None],
                    other=0.0,
                )
                if low_ptr.dtype.element_ty == tl.float32:
                    # FP32 partial as hi + lo in B's dtype: two tensor-core dots, ~16 bits.
                    hi = low.to(b.dtype)
                    accumulator += tl.dot(hi, b)
                    accumulator += tl.dot((low - hi.to(tl.float32)).to(b.dtype), b)
                else:
                    accumulator += tl.dot(low, b)
    return accumulator


@triton.jit
def _gemm_kernel(
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wsn,
    stride_wsk,
    stride_os,
    stride_om,
    lora_low,
    lora_b,
    lora_slices,
    lora_num_slices,
    lora_rank,
    lora_b_ld,
    stride_lm,
    stride_ls,
    xch,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    QUANTIZE_X: tl.constexpr,
    GROUP_M: tl.constexpr = 1,
    SCALED: tl.constexpr = True,
    LORA_R: tl.constexpr = 0,
    EXCHANGE: tl.constexpr = False,
    XCH_RANK: tl.constexpr = 0,
    XCH_WORLD: tl.constexpr = 1,
):
    """out[split] = sum over this split's K groups of (x_group . w_group^T) * sx * sw.

    ``SCALED`` False is the BF16/FP16 mode of the same kernel: ``x`` and ``w`` in one 16-bit
    dtype, no scales, FP32 accumulation over 128-column steps (the epilogue hooks' host).

    ``x`` is BF16/FP16 (``QUANTIZE_X``: quantized here per row and group, as the standalone
    quantizer does) or E4M3 with ``x_scale`` [M, groups]. ``w`` is E4M3 [N, K] with
    ``w_scale`` [ceil(N / 128), groups]. With one split the result is rounded to ``out``'s
    dtype; otherwise FP32 partials go to ``out[split]`` for ``_split_sum_kernel``.
    ``GROUP_M`` > 1 walks the tiles in bands of ``GROUP_M`` row tiles, column by column, so
    that concurrent programs share weight tiles in L2 (the Triton matmul tutorial's order).
    ``EXCHANGE`` (one split) publishes each rounded output tile to the group's exchange
    (``collective.triton.exchange.publish``, tiles ``BLOCK_M x BLOCK_N``).
    """
    if EXCHANGE:
        # The exchange's reduce (launched with PDL) starts now and polls the peers meanwhile.
        tl.extra.cuda.gdc_launch_dependents()
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if GROUP_M > 1:
        n_tiles = tl.num_programs(0)
        linear = pid_m * n_tiles + pid_n
        band = GROUP_M * n_tiles
        first_m = (linear // band) * GROUP_M
        band_m = tl.minimum(tl.num_programs(1) - first_m, GROUP_M)
        pid_m = first_m + (linear % band) % band_m
        pid_n = (linear % band) // band_m
    split = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    rows64 = rows.to(tl.int64)
    cols64 = cols.to(tl.int64)
    offs_k = tl.arange(0, _GROUP)
    groups = tl.cdiv(K, _GROUP)
    per_split = tl.cdiv(groups, SPLIT_K)
    first = split * per_split
    last = tl.minimum(first + per_split, groups)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(first, last):
        k = g * _GROUP + offs_k
        k_mask = k < K
        x_mask = row_mask[:, None] & k_mask[None, :]
        x_ptrs = x_ptr + rows64[:, None] * stride_xm + k[None, :]
        w_ptrs = w_ptr + cols64[None, :] * stride_wn + k[:, None]
        w_mask = col_mask[None, :] & k_mask[:, None]
        if not SCALED:
            a = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            accumulator = tl.dot(a, w, accumulator)
        elif QUANTIZE_X:
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            x_scale, multiplier = _group_scales(x, x_mask)
            v = _saturate(x * multiplier[:, None])
            a = v.to(w_ptr.dtype.element_ty)
        else:
            a = tl.load(x_ptrs, mask=x_mask, other=0.0)
            x_scale = tl.load(
                x_scale_ptr + rows64 * stride_xsm + g * stride_xsk, mask=row_mask, other=0.0
            ).to(tl.float32)
        if SCALED:
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            w_scale = tl.load(
                w_scale_ptr + (cols64 // _GROUP) * stride_wsn + g * stride_wsk,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(a, w) * (x_scale[:, None] * w_scale[None, :])

    if LORA_R > 0 and SPLIT_K == 1:
        accumulator = _lora_epilogue(
            accumulator,
            out_ptr,
            rows64,
            cols,
            row_mask,
            col_mask,
            lora_low,
            lora_b,
            lora_slices,
            lora_num_slices,
            lora_rank,
            lora_b_ld,
            stride_lm,
            stride_ls,
            LORA_R,
        )
    out_ptrs = out_ptr + split * stride_os + rows64[:, None] * stride_om + cols64[None, :]
    out_mask = row_mask[:, None] & col_mask[None, :]
    values = accumulator.to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, values, mask=out_mask)
    if EXCHANGE and SPLIT_K == 1:
        tile = pid_m * tl.cdiv(N, BLOCK_N) + pid_n
        publish(xch, values, pid_m * BLOCK_M, pid_n * BLOCK_N, M, N, tile, XCH_RANK, XCH_WORLD)


@triton.jit
def _split_sum_kernel(
    partial_ptr,
    out_ptr,
    numel,
    N,
    stride_om,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out = sum of the FP32 partials in split order (deterministic), rounded once."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    total = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLIT_K):
        total += tl.load(partial_ptr + split * numel + offs, mask=mask, other=0.0)
    out_offs = (offs // N) * stride_om + offs % N
    tl.store(out_ptr + out_offs, total.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["M", "N"])
def _split_sum_lora_kernel(
    partial_ptr,
    out_ptr,
    M,
    N,
    stride_om,
    lora_low,
    lora_b,
    lora_slices,
    lora_num_slices,
    lora_rank,
    lora_b_ld,
    stride_lm,
    stride_ls,
    xch,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LORA_R: tl.constexpr,
    EXCHANGE: tl.constexpr = False,
    XCH_RANK: tl.constexpr = 0,
    XCH_WORLD: tl.constexpr = 1,
):
    """``_split_sum_kernel`` over [BLOCK_M, BLOCK_N] tiles, plus the LoRA epilogue (LORA_R > 0)
    before the single rounding: the same split order, so the base sum is the LORA_R=0 result.
    ``EXCHANGE`` publishes each rounded tile, as ``_gemm_kernel`` does with one split."""
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    rows64 = rows.to(tl.int64)
    offs = rows64[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    numel = M.to(tl.int64) * N
    total = tl.load(partial_ptr + offs, mask=mask, other=0.0)
    for split in tl.static_range(1, SPLIT_K):
        total += tl.load(partial_ptr + split * numel + offs, mask=mask, other=0.0)
    if LORA_R > 0:
        total = _lora_epilogue(
            total,
            out_ptr,
            rows64,
            cols,
            row_mask,
            col_mask,
            lora_low,
            lora_b,
            lora_slices,
            lora_num_slices,
            lora_rank,
            lora_b_ld,
            stride_lm,
            stride_ls,
            LORA_R,
        )
    out_ptrs = out_ptr + rows64[:, None] * stride_om + cols[None, :]
    values = total.to(out_ptr.dtype.element_ty)
    tl.store(out_ptrs, values, mask=mask)
    if EXCHANGE:
        pid_m, pid_n = tl.program_id(1), tl.program_id(0)
        tile = pid_m * tl.cdiv(N, BLOCK_N) + pid_n
        publish(xch, values, pid_m * BLOCK_M, pid_n * BLOCK_N, M, N, tile, XCH_RANK, XCH_WORLD)


@triton.jit(do_not_specialize=["M"])  # M only bounds masks: one compile per tile
def _dgrad_kernel(
    dy_ptr,
    w_ptr,
    w_scale_ptr,
    dx_ptr,
    M,
    N,
    K,
    stride_dym,
    stride_wn,
    stride_wsn,
    stride_wsk,
    stride_dxm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALED: tl.constexpr = True,
    ACCUMULATE: tl.constexpr = False,
):
    """dx = dy . W_hat for a frozen block-FP8 ``W`` [N, K]: per 128-row group g of N, the
    exact BF16/FP16 dot of dy's columns with the decoded weight rows accumulates in FP32 and
    takes its scale ``w_scale[g, k // 128]``; groups are summed in order (deterministic, no
    split), and the result is rounded once to ``dx``'s dtype. ``SCALED`` False: a 16-bit ``W``
    in dy's dtype, no scales. ``ACCUMULATE`` adds dx's existing values in FP32 first (beta 1)."""
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    rows64 = rows.to(tl.int64)
    offs_n = tl.arange(0, _GROUP)
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for g in range(0, tl.cdiv(N, _GROUP)):
        n = g * _GROUP + offs_n
        n_mask = n < N
        dy = tl.load(
            dy_ptr + rows64[:, None] * stride_dym + n[None, :],
            mask=row_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        w_ptrs = w_ptr + n.to(tl.int64)[:, None] * stride_wn + cols[None, :]
        w_mask = n_mask[:, None] & col_mask[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        if SCALED:
            w_scale = tl.load(
                w_scale_ptr + g * stride_wsn + (cols // _GROUP) * stride_wsk,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.dot(dy, w.to(dy.dtype)) * w_scale[None, :]
        else:
            accumulator = tl.dot(dy, w, accumulator)
    dx_ptrs = dx_ptr + rows64[:, None] * stride_dxm + cols[None, :]
    dx_mask = row_mask[:, None] & col_mask[None, :]
    if ACCUMULATE:
        accumulator += tl.load(dx_ptrs, mask=dx_mask, other=0.0).to(tl.float32)
    tl.store(dx_ptrs, accumulator.to(dx_ptr.dtype.element_ty), mask=dx_mask)


@triton.jit
def _dequantize_kernel(
    w_ptr,
    scale_ptr,
    out_ptr,
    N,
    K,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_sk,
    stride_oe,
    stride_on,
):
    """out[e, n, k] = round(fp32(w[e, n, k]) * scale[e, n // 128, k // 128]), one block each."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.program_id(2).to(tl.int64)
    rows = pid_n * _GROUP + tl.arange(0, _GROUP)
    cols = pid_k * _GROUP + tl.arange(0, _GROUP)
    mask = (rows < N)[:, None] & (cols < K)[None, :]
    rows64 = rows.to(tl.int64)
    w_ptrs = w_ptr + expert * stride_we + rows64[:, None] * stride_wn + cols[None, :]
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + expert * stride_se + pid_n * stride_sn + pid_k * stride_sk)
    out_ptrs = out_ptr + expert * stride_oe + rows64[:, None] * stride_on + cols[None, :]
    tl.store(out_ptrs, (w * scale.to(tl.float32)).to(out_ptr.dtype.element_ty), mask=mask)


_DEVICE_KEYS: dict[torch.device, tuple[tuple[int, int], int]] = {}


def _device_key(device: torch.device) -> tuple[tuple[int, int], int]:
    key = _DEVICE_KEYS.get(device)
    if key is None:
        properties = torch.cuda.get_device_properties(device)
        key = (properties.major, properties.minor), properties.multi_processor_count
        _DEVICE_KEYS[device] = key
    return key


# Each eager launch reuses its compiled kernel without JITFunction.run's binding (FastLaunch).
_LAUNCHERS = {
    kernel: FastLaunch(kernel)
    for kernel in (
        _quantize_kernel,
        _gemm_kernel,
        _split_sum_kernel,
        _split_sum_lora_kernel,
        _dgrad_kernel,
        _silu_quantize_kernel,
        _dequantize_kernel,
    )
}


def _launch(kernel, grid, *args, **kwargs):
    """``kernel[grid](*args, **kwargs)`` through its compiled-kernel cache (a kernel object
    without one, e.g. a wrapper, launches itself)."""
    launcher = _LAUNCHERS.get(kernel)
    if launcher is None:
        kernel[grid](*args, **kwargs)
    else:
        launcher(grid, *args, **kwargs)


def _direct(kernel, grid, *args, **kwargs):
    """``kernel[grid](*args, **kwargs)`` through ``JITFunction.run``."""
    kernel[grid](*args, **kwargs)


def _bind(kernel, grid, *args, **kwargs):
    """Launch ``kernel[grid](*args, **kwargs)`` and return ``relaunch(*args)``: the same grid,
    keyword and non-tensor arguments with new tensors (its compiled kernel, bound)."""
    launcher = _LAUNCHERS.get(kernel)
    if launcher is None:
        kernel[grid](*args, **kwargs)
        return lambda *values: kernel[grid](*values, **kwargs)
    return launcher.bind(grid, *args, **kwargs)


def quantize(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    tile: tuple[int, int] | None = None,
    scale_ue8m0: bool = False,
) -> None:
    """Fill ``q`` [M, K] E4M3 and ``scale`` [M, groups] FP32 from ``x`` [M, K] (checked);
    ``scale_ue8m0`` rounds each scale up to a power of two (bitwise the CuTe quantizer's).

    ``tile`` (rows per program, warps) defaults to the device's measured one; any tile gives
    the same bits. Contiguous operands with whole groups run as ``[M * groups, 128]`` (one
    group per kernel row), so that each program reads and writes one contiguous span."""
    M, K = x.shape
    groups = -(-K // GROUP)
    if tile is None:
        tile = _block_fp8_dispatch.quantizer_tile(*_device_key(x.device), M, groups)
    block_m, num_warps = tile
    if K % GROUP == 0 and x.is_contiguous() and q.is_contiguous() and scale.is_contiguous():
        x, q, scale = x.view(-1, GROUP), q.view(-1, GROUP), scale.view(-1, 1)
        M, K = x.shape
    grid = (-(-M // block_m), -(-K // GROUP))
    _launch(
        _quantize_kernel,
        grid,
        x,
        q,
        scale,
        M,
        K,
        x.stride(0),
        q.stride(0),
        scale.stride(0),
        BLOCK_M=block_m,
        UE8M0=scale_ue8m0,
        num_warps=num_warps,
    )


def gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    out: torch.Tensor,
    *,
    config: tuple[int, ...] | None = None,
    partial: torch.Tensor | None = None,
    lora=None,
    exchange=None,
) -> tuple[int, int]:
    """``out`` [M, N] = the block-FP8 product; ``x_scale`` None quantizes BF16/FP16 ``x``.
    A 16-bit ``weight`` (``weight_scale`` None) selects the BF16/FP16 mode: ``x`` in the same
    dtype, ``out = x . weight^T`` with FP32 accumulation, split-K summed in order.
    ``config`` is (BLOCK_M, BLOCK_N, SPLIT_K, warps, stages[, GROUP_M]), by default
    ``_block_fp8_tuning.tuned_config`` (the device class's table, else today's rule). ``lora`` (a
    ``LoRAEpilogue``) is added before the single output rounding: in the GEMM with one
    split, else in the split sum. ``exchange`` (a ``collective.inference.Epilogue``) publishes
    the rounded output tiles from the same kernel. Returns the output tiling (rows, columns)
    that ``Exchange.reduce`` takes."""
    M, K = x.shape
    N = weight.shape[0]
    if config is None:
        mode = "fp8" if weight_scale is not None else "bf16"
        config = tuned_config(device_class(x.device), M, N, K, mode)
    block_m, block_n, split_k, num_warps, num_stages, *group_m = config
    scaled = weight_scale is not None
    quantize_x = scaled and x_scale is None
    grid = (-(-N // block_n), -(-M // block_m), split_k)
    target = (
        out
        if split_k == 1
        else partial
        if partial is not None
        else out.new_empty((split_k, M, N), dtype=torch.float32)
    )
    scales = weight_scale if scaled else weight  # placeholders, never read when not scaled
    lora_args = (
        (out, out, out, 0, 0, 0, 0, 0)
        if lora is None
        else (
            lora.low,
            lora.b_ptrs,
            lora.slices,
            lora.b_ptrs.numel(),
            lora.rank,
            lora.b_ld,
            lora.low.stride(0),
            lora.low.stride(1),
        )
    )
    lora_r = 0 if lora is None else lora.rank_tile
    xch_args = (
        {}
        if exchange is None
        else {"EXCHANGE": True, "XCH_RANK": exchange.rank, "XCH_WORLD": exchange.world}
    )
    desc = out if exchange is None else exchange.desc
    # The exchange's descriptor goes through JITFunction.run's binding; the rest reuse launches.
    launch = _launch if exchange is None else _direct
    launch(
        _gemm_kernel,
        grid,
        x,
        x_scale if x_scale is not None else scales,
        weight,
        scales,
        target,
        M,
        N,
        K,
        x.stride(0),
        0 if x_scale is None else x_scale.stride(0),
        0 if x_scale is None else x_scale.stride(1),
        weight.stride(0),
        scales.stride(0) if scaled else 0,
        scales.stride(1) if scaled else 0,
        target.stride(0) if split_k > 1 else 0,
        target.stride(-2),
        *lora_args,
        desc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        QUANTIZE_X=quantize_x,
        GROUP_M=group_m[0] if group_m else 1,
        SCALED=scaled,
        LORA_R=lora_r,
        **(xch_args if split_k == 1 else {}),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if split_k == 1:
        return block_m, block_n
    if lora is not None or exchange is not None:
        sum_m = max(16, block_m)
        launch(
            _split_sum_lora_kernel,
            (-(-N // block_n), -(-M // sum_m)),
            target,
            out,
            M,
            N,
            out.stride(0),
            *lora_args,
            desc,
            SPLIT_K=split_k,
            BLOCK_M=sum_m,
            BLOCK_N=block_n,
            LORA_R=lora_r,
            **xch_args,
            num_warps=4,
        )
        return sum_m, block_n
    block = 1024
    _launch(
        _split_sum_kernel,
        (-(-(M * N) // block),),
        target,
        out,
        M * N,
        N,
        out.stride(0),
        SPLIT_K=split_k,
        BLOCK=block,
        num_warps=4,
    )
    return block_m, block_n


def dequantize(weight: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """``out`` [E, N, K] = the decoded ``weight`` [E, N, K] with ``scale`` [E, bn, bk]."""
    E, N, K = weight.shape
    grid = (-(-K // GROUP), -(-N // GROUP), E)
    _launch(
        _dequantize_kernel,
        grid,
        weight,
        scale,
        out,
        N,
        K,
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        scale.stride(2),
        out.stride(0),
        out.stride(1),
        num_warps=4,
    )


def dgrad(
    dy: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    dx: torch.Tensor,
    *,
    accumulate: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """``dx`` [M, K] = ``dy`` [M, N] . W_hat (both with unit column stride; checked by the
    caller); ``weight_scale`` None takes a 16-bit ``weight`` in dy's dtype as is (BF16 mode).
    ``accumulate`` adds dx's existing values (beta 1, in FP32, one rounding). Tiles by rows
    only (the weight streams once per row tile); deterministic. Returns ``relaunch(dy, dx)``
    for operands of the same shapes, strides and dtypes."""
    M, N = dy.shape
    K = weight.shape[1]
    scaled = weight_scale is not None
    stages = 3
    if M <= 16:
        block_m, block_k, warps = 16, 64, 4
    elif M <= 64:
        block_m, block_k, warps = triton.next_power_of_2(M), 64, 4
    else:  # BF16 mode (BlockFP8Linear decodes + cuBLAS past 64 rows); fits sm12x's 99 KiB
        block_m, block_k, warps, stages = 128, 64, 8, 2
    grid = (-(-K // block_k), -(-M // block_m))
    scales = weight_scale if scaled else weight
    sizes = (
        M,
        N,
        K,
        dy.stride(0),
        weight.stride(0),
        weight_scale.stride(0) if scaled else 0,
        weight_scale.stride(1) if scaled else 0,
        dx.stride(0),
    )
    bound = _bind(
        _dgrad_kernel,
        grid,
        dy,
        weight,
        scales,
        dx,
        *sizes,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        SCALED=scaled,
        ACCUMULATE=accumulate,
        num_warps=warps,
        num_stages=stages,
    )
    return lambda dy, dx: bound(dy, weight, scales, dx, *sizes)


def silu_quantize(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, *,
    swiglu_limit: float | None = None, scale_ue8m0: bool = False, act: torch.Tensor | None = None,
) -> Callable[..., None] | None:  # fmt: skip
    """Fill ``q`` [M, I] E4M3 and ``scale`` [M, ceil(I / 128)] FP32 (any strides) with the
    1x128 quantization of SiLU(gate) * up of gate-first ``x`` [M, 2I] (checked by the caller);
    ``act`` [M, I] in ``x``'s dtype, if given, receives the rounded product.
    One launch; zero rows launch nothing. Nothing in the compiled signature depends on M
    (row tiles of four, M and scale strides unspecialized): one eager call per dtype, width
    and option set warms every M for graph capture. Returns ``relaunch(x, q, scale, act)`` for
    operands of the same shapes, strides and dtypes (None when nothing launched)."""
    M, width = q.shape
    if not M or not width:
        return None
    limit = 0.0
    if swiglu_limit is not None:
        limit = torch.tensor(swiglu_limit, dtype=torch.bfloat16).item()
    block_m = 4
    grid = (-(-M // block_m), -(-width // GROUP))
    sizes = (
        M,
        width,
        x.stride(0),
        q.stride(0),
        scale.stride(0),
        scale.stride(1),
        0 if act is None else act.stride(0),
        limit,
    )
    bound = _bind(
        _silu_quantize_kernel,
        grid,
        x,
        q,
        scale,
        q if act is None else act,
        *sizes,
        BLOCK_M=block_m,
        HAS_LIMIT=swiglu_limit is not None,
        ACT_OUT=act is not None,
        UE8M0=scale_ue8m0,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return lambda x, q, scale, act: bound(x, q, scale, q if act is None else act, *sizes)
