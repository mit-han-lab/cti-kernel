# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Portable static E4M3 activation quantization."""

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

_ready: set[tuple] = set()
_INTERPRET = tl.constexpr(triton.knobs.runtime.interpret)


@triton.jit
def _interpreter_e4m3(v):
    """Encode RTNE directly; the interpreter's FP8 cast mishandles ties and NaNs."""
    bits = v.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    exponent = (magnitude >> 23).to(tl.int32)
    normal = ((magnitude + 0x7FFFF + ((magnitude >> 20) & 1)) >> 20) - 960
    shift = tl.minimum(tl.maximum(141 - exponent, 21), 24)
    significand = (magnitude & 0x7FFFFF) | 0x800000
    subnormal = (significand + (1 << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift
    encoded = tl.where(exponent >= 121, normal, tl.where(exponent >= 117, subnormal, 0))
    encoded = tl.where(v != v, 127, encoded) | ((bits >> 24) & 128)
    return encoded.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit(
    do_not_specialize=["M", "K", "STRIDE"],
    do_not_specialize_on_alignment=["X", "S", "Q", "SCALE", "M", "K", "STRIDE"],
)
def _quantize(X, S, Q, SCALE, M: tl.int64, K: tl.int64, STRIDE: tl.int64, MODE: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * 256 + tl.arange(0, 256).to(tl.int64)
    k = K.to(tl.int64)
    row, col = pos // k, pos % k
    valid = row < M
    s = tl.load(S)
    x = tl.load(X + row * STRIDE.to(tl.int64) + col, valid, other=0).to(tl.float32)
    v = tl.math.div_rn(x, s)
    v = tl.where(v != v, v, tl.minimum(tl.maximum(v, -448.0), 448.0))
    if _INTERPRET:
        q = _interpreter_e4m3(v)
    else:
        q = v.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Q + pos, q, valid)
    if MODE == "block":
        groups = (k + 127) // 128
        tl.store(SCALE + row * groups + col // 128, s, valid & (col % 128 == 0))
    elif MODE == "row":
        tl.store(SCALE + row, s, valid & (col == 0))


def quantize(x, input_scale, *, block_scale=False, row_scale=False, pdl=False, outputs=None):
    """Validated 2-D input; pdl changes scheduling only and is ignored."""
    if block_scale and row_scale:
        raise ValueError("block_scale and row_scale are mutually exclusive")
    mode = "block" if block_scale else "row" if row_scale else "plain"
    m, k = x.shape
    if outputs is None:
        q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
        scale = (
            torch.empty(
                (m, 1 if row_scale else (k + 127) // 128), dtype=torch.float32, device=x.device
            )
            if block_scale or row_scale
            else None
        )
    else:
        q, scale = outputs
    if not m:
        return (q, scale) if block_scale or row_scale else q
    key = (x.device, x.dtype, mode)
    with torch.cuda.device(x.device) if x.is_cuda else nullcontext():
        if x.is_cuda and torch.cuda.is_current_stream_capturing() and key not in _ready:
            raise RuntimeError("static FP8 quantizer: warm up this dtype/mode before capture")
        _quantize[(triton.cdiv(m * k, 256),)](
            x, input_scale, q, scale, m, k, x.stride(0), mode, num_warps=4, enable_fp_fusion=False
        )
        _ready.add(key)
    return (q, scale) if block_scale or row_scale else q
