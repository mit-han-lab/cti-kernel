# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""MXFP8 activation quantization (OCP MX: E4M3 values, one UE8M0 scale per 32 columns).

``quantize_mxfp8(x)`` maps BF16/FP16 ``x`` ``[M, K]`` (``K`` a multiple of 32) to ``q``
``[M, K]`` ``float8_e4m3fn`` and ``scale`` with ``x ~ fp32(q[m, k]) * 2^(scale[m, k // 32] -
127)``. Recipe (FlashInfer 0.7.0.post1's CUDA quantizer): ``s = amax * rcp.approx(448)`` over
each group of 32, rounded *up* to a power of two (UE8M0, saturating), and ``q =
E4M3_RN(clamp(x * rcp.approx(2^s), -448, 448))``; an all-zero group gets byte 0 and zeros.

``scale`` is ``float8_e8m0fnu`` ``[M, K / 32]`` (row major) by default; ``swizzled=True``
returns the 128 x 4 tensor-core layout FlashInfer's CUTLASS MoE reads, as flat ``uint8``
padded to whole 128-row x 4-group tiles (padding bytes zero). Providers: FlashInfer's CUDA
quantizer on compute capability 10.x; our Triton kernel, bitwise equal to it, on 8.9 and the
12.x family (the library kernel compiles to nothing there); older devices raise
``RuntimeError``. One launch,
no host synchronization, outputs from the caching allocator: capturable after one eager
call per process (the first call loads, or JIT-builds, the library module; later calls go
straight to its compiled entry). Inference only.
"""

from __future__ import annotations

import functools

import torch

from .flashinfer import _mxfp8

GROUP = 32


@functools.cache
def _capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def quantize_mxfp8(x: torch.Tensor, *, swizzled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` ``[M, K]``; returns new ``(q, scale)`` (see the module docstring)."""
    operation = "quantize_mxfp8"
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{operation}: x must be a tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"{operation}: x must be BF16 or FP16, not {x.dtype}")
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % GROUP:
        raise ValueError(f"{operation}: x must be [M, K] with K a positive multiple of 32")
    if not x.is_cuda:
        raise ValueError(f"{operation}: x must be on a CUDA device")
    if _capability(x.device) < (8, 9):
        raise RuntimeError(f"{operation}: needs compute capability 8.9 or newer")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError(f"{operation} is inference-only and does not support autograd")
    rows, cols = x.shape
    if rows == 0:
        scale = torch.empty((0, cols // GROUP), dtype=torch.uint8, device=x.device)
        empty = torch.empty((0, cols), dtype=torch.float8_e4m3fn, device=x.device)
        return empty, (scale.view(-1) if swizzled else scale.view(torch.float8_e8m0fnu))
    x = x if x.is_contiguous() else x.contiguous()
    if _capability(x.device)[0] == 10:
        return _mxfp8.quantize(x, swizzled=swizzled)
    from .triton import _mxfp8 as _mxfp8_triton

    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=x.device)
    if swizzled:
        scale = torch.empty(
            -(-rows // 128) * 128 * (-(-cols // 128) * 4), dtype=torch.uint8, device=x.device
        )
    else:
        scale = torch.empty((rows, cols // GROUP), dtype=torch.float8_e8m0fnu, device=x.device)
    _mxfp8_triton.quantize(x, q, scale.view(torch.uint8), swizzled=swizzled)
    return q, scale


def dequantize_mxfp8(
    q: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``fp32(q) * 2^(scale - 127)`` for a row-major ``scale`` ``[M, K / 32]``; for references."""
    if q.ndim != 2 or tuple(scale.shape) != (q.shape[0], q.shape[1] // GROUP):
        raise ValueError("dequantize_mxfp8: expected q [M, K] and scale [M, K / 32]")
    factor = scale.view(torch.uint8).to(torch.float32).sub(127).exp2()
    return (q.to(torch.float32) * factor.repeat_interleave(GROUP, dim=1)).to(dtype)


__all__ = ["dequantize_mxfp8", "quantize_mxfp8"]
