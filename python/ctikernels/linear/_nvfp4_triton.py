# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Recipe follows FlashInfer 0.7.0.post1 (csrc/nv_internal/tensorrt_llm/kernels/
# quantization_utils.cuh, cvt_warp_fp16_to_fp4), design reference only (Apache-2.0).

"""NVFP4 Triton kernels of ``ctikernels.linear.nvfp4``: activation quantization and the
small-batch W4A4 GEMM.

**Quantization.** The FlashInfer market recipe uses NaN-ignoring amax, explicit
FTZ products, approximate FTZ reciprocals, saturating E4M3 and direct hardware
E2M1 conversion (SM100+; compute capability 8.9 up to 10 uses a bitwise-equal software
rounding). Only S = 1/input_scale uses div.rn. See _nvfp4_recipe for PTX
and the centralized H1 zero comparison. ``quantize``
reads rows of BF16/FP16 values, or rows interleaving gate and up columns (``g0, u0, g1, u1,
...``) whose SiLU(gate) * up it quantizes. Scales are stored in the blocked layout (scale ``(r,
b)`` at ``[r // 128, b // 4, r % 32, r // 32 % 4, b % 4]``); rows and blocks past the tensor up
to the layout's padding get zero scales.

**Small-batch GEMM** (``gemm``, up to 64 rows): every program decodes a ``[BLOCK_N, BLOCK_K]``
weight tile and the matching activation tile to BF16 (exact: an E2M1 value times an E4M3 scale
has at most 6 significant bits) and accumulates their tensor-core products in FP32. A BF16/FP16
activation is quantized in the kernel by the recipe above (the values equal ``quantize``'s). With
one split, each program writes its tile's result: BF16/FP16 ``acc * column_scale * input_scale``,
or for an interleaved gate/up weight the NVFP4 quantization of SiLU(gate) * up. With K split over several
programs, each stores its FP32 partial and the last to arrive at its column tile's counter sums
them in split order, resets the counter and writes the result through the same epilogue, so one
launch serves every split and the result does not depend on arrival timing. Counters belong to
the operation object: its calls and graph replays must run in order on one stream (one object,
one stream). Everything runs on compute capability 8.9 or newer.

The fused gated epilogue applies SiLU * up to FP32 accumulators, without intermediate
BF16 gate/up rounding. SGLang rounds gate_up GEMM output to BF16 before SiLU*up+quant;
our fused path does not; gate 2 checks that it is no worse vs FP64. The unfused Triton
quantizer instead consumes an already rounded BF16 projection.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..triton.silu import silu
from . import _nvfp4_recipe as ptx

try:
    from triton.backends.nvidia.compiler import CUDAOptions
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

    _HAS_PDL = "launch_pdl" in getattr(CUDAOptions, "__dataclass_fields__", {})
except ImportError:
    _HAS_PDL = False


def _pdl_options(device):
    enabled = _HAS_PDL and torch.cuda.get_device_capability(device) >= (9, 0)
    return dict(PDL=enabled, **({"launch_pdl": True} if enabled else {}))


ROW_TILE = 128  # scale-layout rows per atom
BLOCK = 16  # columns per scale
_PTX_DIV = tl.constexpr(ptx.DIV)
_PTX_RCP = tl.constexpr(ptx.RCP)
_PTX_MUL = tl.constexpr(ptx.MUL)
_PTX_MAX = tl.constexpr(ptx.MAX)
_PTX_NONZERO = tl.constexpr(ptx.NONZERO)
_PTX_E4M3 = tl.constexpr(ptx.E4M3)
_PTX_E2M1 = tl.constexpr(ptx.E2M1)


@triton.jit
def _inverse_scale(value):
    return tl.inline_asm_elementwise(
        _PTX_DIV, "=f,f", [value], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rcp(value):
    return tl.inline_asm_elementwise(
        _PTX_RCP, "=f,f", [value], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise(
        _PTX_MUL, "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _maximum(a, b):
    return tl.inline_asm_elementwise(
        _PTX_MAX, "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _amax_nonzero(value):
    return tl.inline_asm_elementwise(
        _PTX_NONZERO, "=r,f", [value], dtype=tl.int32, is_pure=True, pack=1
    )


@triton.jit
def _e2m1_soft(v):
    """``cvt.rn.satfinite.e2m1`` of FP32 ``v`` in software (uint8 code): round to nearest
    (ties to even), saturate at +-6 (infinities too), keep the sign bit (also of zero
    results); NaN of either sign gives +6, as the hardware does."""
    a = tl.abs(v)
    code = (
        (a > 0.25).to(tl.uint8)
        + (a >= 0.75).to(tl.uint8)
        + (a > 1.25).to(tl.uint8)
        + (a >= 1.75).to(tl.uint8)
        + (a > 2.5).to(tl.uint8)
        + (a >= 3.5).to(tl.uint8)
        + (a > 5.0).to(tl.uint8)
    )
    sign = (v.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8) << 3
    return tl.where(a != a, 7, code | sign).to(tl.uint8)


@triton.jit
def _pack(low, high, NATIVE: tl.constexpr):
    if NATIVE:
        return tl.inline_asm_elementwise(
            _PTX_E2M1, "=r,f,f", [low, high], dtype=tl.uint32, is_pure=True, pack=1
        ).to(tl.uint8)
    else:
        return _e2m1_soft(low) | (_e2m1_soft(high) << 4)


@triton.jit
def _e2m1_codes(v, NATIVE: tl.constexpr):
    """Hardware conversion (or its software equal), including NaN and signed-zero semantics."""
    return _pack(v, v, NATIVE) & 15


@triton.jit
def _decode(packed):
    """FP32 values ``[R, 2C]`` of packed E2M1 bytes ``[R, C]`` (the low nibble first).

    A code ``s e1 e0 m`` placed as FP16 bits ``s 0000 e1 e0 m 0...`` is its E2M1 value times
    2^-14, for subnormal ``e = 0`` codes too."""
    bits = packed.to(tl.int16)
    low = ((bits & 7) << 9) | ((bits & 8) << 12)
    high = ((bits & 0x70) << 5) | ((bits & 0x80) << 8)
    low = low.to(tl.float16, bitcast=True).to(tl.float32)
    high = high.to(tl.float16, bitcast=True).to(tl.float32)
    rows: tl.constexpr = packed.shape[0]
    cols: tl.constexpr = packed.shape[1]
    return tl.reshape(tl.join(low, high), (rows, 2 * cols)) * 16384.0


@triton.jit
def _block_scales(blocks, inverse_global):
    """(Decoded E4M3 scale, rescale factor, raw SF byte) per ``blocks [R, B, 16]``."""
    # Widened BF16/FP16 max is exact; max.f32 deliberately has no .ftz/.NaN.
    amax = tl.reduce(tl.abs(blocks), 2, _maximum)
    sf_f32 = _mul(inverse_global, _mul(amax, _rcp(tl.full((), 6.0, tl.float32))))
    bits = tl.inline_asm_elementwise(
        _PTX_E4M3, "=r,f", [sf_f32], dtype=tl.uint32, is_pure=True, pack=1
    ).to(tl.uint8)
    block_scale = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    rescale = _rcp(_mul(block_scale, _rcp(inverse_global)))
    return block_scale, tl.where(_amax_nonzero(amax) != 0, rescale, 0.0), bits


@triton.jit
def _scale_offsets(row, block, atoms_k):
    """Blocked-layout byte offsets ``[R, B]`` of scales (row, block)."""
    return (
        ((row // 128).to(tl.int64) * atoms_k)[:, None] * 512
        + ((block // 4) * 512)[None, :]
        + ((row % 32) * 16 + (row // 32 % 4) * 4)[:, None]
        + (block % 4)[None, :]
    )


@triton.jit
def _store_quantized(
    value, row, rows, cols, block0, data, stride_data, scale, atoms_k, inverse_global,
    NATIVE: tl.constexpr,
):  # fmt: skip
    """Quantize ``value [R, C]`` (rows ``row``, columns from ``block0 * 16``) and store it."""
    R: tl.constexpr = value.shape[0]
    C: tl.constexpr = value.shape[1]
    blocks = tl.reshape(value, (R, C // 16, 16))
    _, rescale, bits = _block_scales(blocks, inverse_global)
    products = tl.reshape(_mul(blocks, rescale[:, :, None]), (R, C // 2, 2))
    low, high = tl.split(products)
    byte = block0 * 8 + tl.arange(0, C // 2)
    tl.store(
        data + row.to(tl.int64)[:, None] * stride_data + byte[None, :],
        _pack(low, high, NATIVE),
        mask=(row < rows)[:, None] & (byte < cols // 2)[None, :],
    )
    block = block0 + tl.arange(0, C // 16)
    bits = tl.where((row < rows)[:, None] & (block < cols // 16)[None, :], bits, 0)
    tl.store(scale + _scale_offsets(row, block, atoms_k), bits, mask=(block < atoms_k * 4)[None, :])


@triton.jit
def _call_scales(
    input_scale,
    weight_scale,
    up_scale,
    alpha,
    output_scale,
    reciprocal,
    column_scale,
    COLUMNS: tl.constexpr,
    ALPHA: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    GATED: tl.constexpr,
    pid,
    programs,
):
    if pid == 0:
        if ALPHA:
            tl.store(alpha, tl.load(input_scale) * tl.load(weight_scale))
        if RECIPROCAL:
            tl.store(reciprocal, _inverse_scale(tl.load(output_scale)))
    if COLUMNS:
        for start in range(pid * 128, COLUMNS, programs * 128):
            col = start + tl.arange(0, 128)
            factor = tl.load(weight_scale)
            if GATED:
                factor = tl.where(col % 2 == 0, factor, tl.load(up_scale))
            tl.store(column_scale + col.to(tl.int64), factor, col < COLUMNS)


@triton.jit
def _refresh_scales_kernel(
    input_scale,
    weight_scale,
    up_scale,
    alpha,
    output_scale,
    reciprocal,
    column_scale,
    COLUMNS: tl.constexpr,
    ALPHA: tl.constexpr,
    RECIPROCAL: tl.constexpr,
    GATED: tl.constexpr,
    PDL: tl.constexpr,
):
    if PDL:
        # No explicit trigger: dependents launch at exit, after the stores. Some GEMM epilogues
        # read these scalars before their own grid-dependency wait (FlashInfer's sm120 CUTLASS
        # FP4 GEMM, always launched with PDL, reads alpha so).
        gdc_wait()
    _call_scales(
        input_scale,
        weight_scale,
        up_scale,
        alpha,
        output_scale,
        reciprocal,
        column_scale,
        COLUMNS,
        ALPHA,
        RECIPROCAL,
        GATED,
        tl.program_id(0),
        tl.num_programs(0),
    )


def refresh_scales(
    input_scale,
    weight_scale,
    *,
    up_scale=None,
    alpha=None,
    output_scale=None,
    reciprocal=None,
    column_scale=None,
):
    """The quantized entry points have no input kernel to carry scalar preparation."""
    for value in (weight_scale, up_scale):
        if value is not None and value.requires_grad:
            raise ValueError("NVFP4 quantize: weights are frozen and must not require grad")
    columns = 0 if column_scale is None else column_scale.numel()
    _refresh_scales_kernel[(max(1, triton.cdiv(columns, 128)),)](
        input_scale,
        weight_scale,
        up_scale,
        alpha,
        output_scale,
        reciprocal,
        column_scale,
        columns,
        alpha is not None,
        reciprocal is not None,
        up_scale is not None,
        **_pdl_options(input_scale.device),
    )


@triton.jit(do_not_specialize=["rows"])
def _quantize_kernel(
    x, stride_x, data, stride_data, scale, global_scale, rows,
    weight_scale, up_scale, alpha, output_scale, reciprocal, column_scale,
    gate_factor, up_factor, clear_scale,
    cols: tl.constexpr, scale_atoms_k: tl.constexpr, COLUMNS: tl.constexpr,
    ALPHA: tl.constexpr, RECIPROCAL: tl.constexpr, WEIGHT_GATED: tl.constexpr,
    SCALE_GATED: tl.constexpr, CLEAR: tl.constexpr,
    GATED: tl.constexpr, GATED_ROW_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    NATIVE: tl.constexpr, PDL: tl.constexpr,
):  # fmt: skip
    if PDL:
        gdc_wait()
        if not (ALPHA or RECIPROCAL or COLUMNS):
            # Consumers wait for completion before reading the data; their prologue may
            # overlap us. Scalar side outputs trigger at exit (see _refresh_scales_kernel).
            gdc_launch_dependents()
    pid = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    programs = tl.num_programs(0) * tl.num_programs(1)
    _call_scales(
        global_scale,
        weight_scale,
        up_scale,
        alpha,
        output_scale,
        reciprocal,
        column_scale,
        COLUMNS,
        ALPHA,
        RECIPROCAL,
        WEIGHT_GATED,
        pid,
        programs,
    )
    if CLEAR:
        for start in range(pid * 128, CLEAR, programs * 128):
            offset = start + tl.arange(0, 128)
            tl.store(clear_scale + offset.to(tl.int64), 0, offset < CLEAR)
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    width: tl.constexpr = 2 * BLOCK_K if GATED else BLOCK_K
    col = tl.program_id(1) * width + tl.arange(0, width)
    mask = (row < rows)[:, None] & (col < (2 * cols if GATED else cols))[None, :]
    source = x + row.to(tl.int64)[:, None] * stride_x + col[None, :]
    if GATED_ROW_BLOCKS:
        source = x + (row.to(tl.int64)[:, None] + (col % 2)[None, :] * rows) * cols
        source += (col // 2)[None, :]
    value = tl.load(source, mask=mask, other=0.0).to(tl.float32)
    if GATED:
        gate, up = tl.split(tl.reshape(value, (BLOCK_M, BLOCK_K, 2)))
        if SCALE_GATED:
            gate = (gate * tl.load(gate_factor)).to(tl.bfloat16).to(tl.float32)
            up = (up * tl.load(up_factor)).to(tl.bfloat16).to(tl.float32)
        value = silu(gate) * up
    inverse_global = _inverse_scale(tl.load(global_scale).to(tl.float32))
    _store_quantized(
        value, row, rows, cols, tl.program_id(1) * (BLOCK_K // 16), data, stride_data, scale,
        scale_atoms_k, inverse_global, NATIVE,
    )  # fmt: skip


def scale_shape(rows: int, cols: int) -> tuple[int, int, int, int, int]:
    """The blocked scale layout's shape for a ``[rows, cols]`` tensor."""
    return (-(-rows // ROW_TILE), -(-cols // (4 * BLOCK)), 32, 4, 4)


def quantize(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    gated: bool,
    data: torch.Tensor,
    scale: torch.Tensor,
    weight_scale=None,
    up_scale=None,
    alpha=None,
    output_scale=None,
    reciprocal=None,
    column_scale=None,
    gate_factor=None,
    up_factor=None,
    clear_scale=None,
    gated_row_blocks=False,
) -> None:
    """Quantize ``x`` ``[rows, cols]`` (or SwiGLU of interleaved ``x`` ``[rows, 2 * cols]``)
    into ``data`` ``[rows, cols // 2]`` uint8 and ``scale`` (``scale_shape``, uint8 view).
    ``gated_row_blocks`` instead reads two contiguous [rows, cols] blocks in ``x``.
    Shapes, dtypes and alignment are the caller's to check."""
    for value in (weight_scale, up_scale, gate_factor, up_factor):
        if value is not None and value.requires_grad:
            raise ValueError("NVFP4 quantize: weights are frozen and must not require grad")
    rows, cols = data.shape[0], data.shape[1] * 2
    atoms_m, atoms_k = scale.shape[0], scale.shape[1]
    block_m = 16 if rows > 512 else 8
    block_k = 256 if cols >= 256 else 64
    if gated:
        block_k //= 2
    grid = (atoms_m * ROW_TILE // block_m, triton.cdiv(atoms_k * 4 * BLOCK, block_k))
    _quantize_kernel[grid](
        x, x.stride(0), data, data.stride(0), scale, global_scale, rows,
        weight_scale, up_scale, alpha, output_scale, reciprocal, column_scale,
        gate_factor, up_factor, clear_scale, cols, atoms_k,
        COLUMNS=0 if column_scale is None else column_scale.numel(),
        ALPHA=alpha is not None, RECIPROCAL=reciprocal is not None,
        WEIGHT_GATED=up_scale is not None, SCALE_GATED=gate_factor is not None,
        CLEAR=0 if clear_scale is None else clear_scale.numel(),
        GATED=gated, GATED_ROW_BLOCKS=gated_row_blocks,
        BLOCK_M=block_m, BLOCK_K=block_k, num_warps=4,
        NATIVE=ptx.native_e2m1(torch.cuda.get_device_capability(x.device)),
        **_pdl_options(x.device),
    )  # fmt: skip


# --- Small-batch GEMM -----------------------------------------------------------------------


@triton.jit
def _activation_tile(
    x, stride_x, x_scale, x_atoms_k, inverse_input, rows, k0,
    QUANTIZED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, NATIVE: tl.constexpr,
):  # fmt: skip
    """BF16 ``[BLOCK_M, BLOCK_K]`` dequantized activation values (without the global scale)."""
    row = tl.arange(0, BLOCK_M)
    if QUANTIZED:
        byte = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        packed = tl.load(
            x + row[:, None] * stride_x + byte[None, :], mask=(row < rows)[:, None], other=0
        )
        block = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        bits = tl.load(x_scale + _scale_offsets(row, block, x_atoms_k))
        block_scale = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        values = tl.reshape(_decode(packed), (BLOCK_M, BLOCK_K // 16, 16))
    else:
        col = k0 + tl.arange(0, BLOCK_K)
        source = tl.load(
            x + row[:, None] * stride_x + col[None, :], mask=(row < rows)[:, None], other=0.0
        ).to(tl.float32)
        blocks = tl.reshape(source, (BLOCK_M, BLOCK_K // 16, 16))
        block_scale, rescale, _ = _block_scales(blocks, inverse_input)
        values = _decode_codes(_e2m1_codes(_mul(blocks, rescale[:, :, None]), NATIVE))
    return tl.reshape(values * block_scale[:, :, None], (BLOCK_M, BLOCK_K)).to(tl.bfloat16)


@triton.jit
def _decode_codes(codes):
    """FP32 values of unpacked E2M1 codes (same FP16 placement as ``_decode``)."""
    bits = codes.to(tl.int16)
    half = ((bits & 7) << 9) | ((bits & 8) << 12)
    return half.to(tl.float16, bitcast=True).to(tl.float32) * 16384.0


@triton.jit
def _fence(value):
    """A GPU-scope acquire-release fence executed by every thread holding ``value``."""
    return tl.inline_asm_elementwise(
        "fence.acq_rel.gpu;", "=r,r", [value], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _epilogue(
    acc, n, rows, column_scale, weight_global, up_global, input_scale, out, stride_out, out_scale, output_scale,
    N: tl.constexpr, OUT_ATOMS_K: tl.constexpr, GATED: tl.constexpr, NATIVE: tl.constexpr,
):  # fmt: skip
    BLOCK_M: tl.constexpr = acc.shape[0]
    BLOCK_N: tl.constexpr = acc.shape[1]
    row = tl.arange(0, BLOCK_M)
    n_ok = n < N
    factor = tl.where(n_ok, tl.load(weight_global), 0.0)
    if GATED:
        factor = tl.where(n % 2 == 0, factor, tl.load(up_global))
    value = acc * factor[None, :] * tl.load(input_scale)
    if GATED:
        gate, up = tl.split(tl.reshape(value, (BLOCK_M, BLOCK_N // 2, 2)))
        inverse_output = _inverse_scale(tl.load(output_scale))
        block0 = tl.program_id(0) * (BLOCK_N // 32)
        _store_quantized(
            silu(gate) * up, row, rows, N // 2, block0, out, stride_out, out_scale,
            OUT_ATOMS_K, inverse_output, NATIVE,
        )  # fmt: skip
        # The layout's padding rows past this tile.
        pad = tl.arange(0, 128)
        block = block0 + tl.arange(0, BLOCK_N // 32)
        tl.store(
            out_scale + _scale_offsets(pad, block, OUT_ATOMS_K),
            tl.zeros((128, BLOCK_N // 32), dtype=tl.uint8),
            mask=(pad >= BLOCK_M)[:, None] & (block < OUT_ATOMS_K * 4)[None, :],
        )
    else:
        tl.store(
            out + row[:, None] * stride_out + n[None, :],
            value.to(out.dtype.element_ty),
            mask=(row < rows)[:, None] & n_ok[None, :],
        )


@triton.jit(do_not_specialize=["rows"])
def _gemm_kernel(
    x, stride_x, x_scale, input_scale,
    w, w_scale, column_scale, weight_global, up_global,
    out, stride_out, out_scale, output_scale, partial, counters,
    rows,
    N: tl.constexpr, K: tl.constexpr,
    X_ATOMS_K: tl.constexpr, W_ATOMS_K: tl.constexpr, OUT_ATOMS_K: tl.constexpr,
    QUANTIZED: tl.constexpr, GATED: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NATIVE: tl.constexpr,
):  # fmt: skip
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = n < N
    inverse_input = _inverse_scale(tl.load(input_scale))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    span: tl.constexpr = K // SPLITS
    for k0 in range(split * span, split * span + span, BLOCK_K):
        a = _activation_tile(
            x,
            stride_x,
            x_scale,
            X_ATOMS_K,
            inverse_input,
            rows,
            k0,
            QUANTIZED,
            BLOCK_M,
            BLOCK_K,
            NATIVE,
        )
        byte = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        packed = tl.load(
            w + n.to(tl.int64)[:, None] * (K // 2) + byte[None, :], mask=n_ok[:, None], other=0
        )
        block = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        bits = tl.load(w_scale + _scale_offsets(n, block, W_ATOMS_K), mask=n_ok[:, None], other=0)
        block_scale = bits.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        values = tl.reshape(_decode(packed), (BLOCK_N, BLOCK_K // 16, 16)) * block_scale[:, :, None]
        b = tl.reshape(values, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(b), acc)

    if SPLITS == 1:
        _epilogue(
            acc, n, rows, column_scale, weight_global, up_global, input_scale, out, stride_out, out_scale, output_scale,
            N, OUT_ATOMS_K, GATED, NATIVE,
        )  # fmt: skip
    else:
        # Every split stores its FP32 partial; the last to arrive at this column tile's counter
        # sums them in split order (deterministic), resets the counter and applies the epilogue.
        row = tl.arange(0, BLOCK_M)
        mask = (row < rows)[:, None] & n_ok[None, :]
        offsets = row.to(tl.int64)[:, None] * N + n[None, :]
        tl.store(partial + split * rows * N + offsets, acc, mask=mask)
        _fence(pid_n)
        tl.debug_barrier()
        arrived = tl.atomic_add(counters + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLITS - 1:
            _fence(pid_n)
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for other in tl.static_range(SPLITS):
                total += tl.load(
                    partial + other * rows * N + offsets, mask=mask, other=0.0, cache_modifier=".cg"
                )
            tl.atomic_xchg(counters + pid_n, 0, sem="relaxed", scope="gpu")
            _epilogue(
                total, n, rows, column_scale, weight_global, up_global, input_scale, out, stride_out, out_scale,
                output_scale, N, OUT_ATOMS_K, GATED, NATIVE,
            )  # fmt: skip


SMALL_ROWS = 64  # the most rows ``gemm`` serves


def gemm(
    a: tuple[torch.Tensor, torch.Tensor | None],
    input_scale: torch.Tensor,
    weight: tuple[torch.Tensor, torch.Tensor],
    column_scale: torch.Tensor,
    *,
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    counters: torch.Tensor,
    weight_globals: tuple[torch.Tensor, torch.Tensor],
    splits: int,
    block_n: int,
    block_k: int,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    """Up to 64 rows of ``x . W^T`` (one row tile of 16, 32 or 64), in one launch.

    ``a``: (BF16/FP16 ``x [M, K]``, None) or a quantized activation (uint8 ``data [M, K /
    2]``, blocked uint8 ``scale``) with ``input_scale`` its global scale; ``weight``: (uint8
    ``[N, K / 2]``, blocked uint8 scale), ``column_scale`` FP32 ``[N]``. ``out``: BF16/FP16
    ``[M, N]``, or for an interleaved gate/up weight (uint8 data ``[M, N / 4]``, blocked uint8
    scale, FP32 output global scale). ``counters``: zeroed int32, one per 32 columns of ``N`` (of
    ``128 * ceil(N / 128)`` when gated), left zeroed. With multiple splits, the last program to
    arrive at each column tile's counter sums its FP32 partials in split order, resets the
    counter and applies the epilogue. Launches and graph replays sharing counters must run in
    order on one stream (one operation object, one stream). ``K`` must be a multiple of
    ``splits * block_k``; the caller checks shapes."""
    x, x_scale = a
    data, w_scale = weight
    rows = x.shape[0]
    n, k = data.shape[0], data.shape[1] * 2
    gated = isinstance(out, tuple)
    partial = x
    if splits > 1:
        partial = torch.empty(splits, rows, n, dtype=torch.float32, device=x.device)
    if gated:
        out_data, out_scale, output_scale = out
        target, stride, atoms = out_data, out_data.stride(0), out_scale.shape[1]
    else:
        target, stride, out_scale, output_scale, atoms = out, out.stride(0), out, input_scale, 1
    programs = triton.cdiv(n, block_n)
    if gated:  # programs past N zero the output scales' padding blocks
        programs = max(programs, triton.cdiv(atoms * 4 * 32, block_n))
    _gemm_kernel[(programs, splits)](
        x, x.stride(0), x if x_scale is None else x_scale, input_scale,
        data, w_scale, column_scale, *weight_globals,
        target, stride, out_scale, output_scale, partial, counters,
        rows, n, k,
        1 if x_scale is None else x_scale.shape[1], w_scale.shape[1], atoms,
        QUANTIZED=x_scale is not None, GATED=gated, SPLITS=splits,
        BLOCK_M=max(16, triton.next_power_of_2(rows)), BLOCK_N=block_n, BLOCK_K=block_k,
        NATIVE=ptx.native_e2m1(torch.cuda.get_device_capability(x.device)),
        num_warps=num_warps, num_stages=num_stages,
    )  # fmt: skip
