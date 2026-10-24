# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent oracles; importing this module never imports a GPU compiler."""

import pytest
import torch

from ctikernels.linear.nvfp4 import blocked_scale, scale_shape


def flashinfer_quantize(x, input_scale):
    """FlashInfer 0.7.0.post1's default CUDA fast-math, E4M3, 128x4 recipe."""
    flashinfer = pytest.importorskip("flashinfer")
    data, scale = flashinfer.fp4_quantize(
        x.contiguous(),
        torch.ones_like(input_scale) / input_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    return data.view(torch.uint8), scale.view(torch.uint8).reshape(scale_shape(*x.shape))


def cute_quantize(x, input_scale):
    """Scalar CuTe PTX oracle, including unrounded FP32 values FlashInfer cannot take.

    Checked against FlashInfer on BF16/FP16 inputs and their FP32 widenings.
    Shares public PTX primitives, but no production quantizer or scale routine.
    """
    from .nvfp4_oracle_cute import quantize

    x = x.contiguous()
    rows, cols = x.shape
    data = torch.empty((rows, cols // 2), device=x.device, dtype=torch.uint8)
    scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.uint8)
    quantize(x, input_scale, data, scales)
    return data, blocked_scale(scales.view(torch.float8_e4m3fn)).view(torch.uint8)


def tolerance_quantize(v, input_scale):
    """Torch approximation for CPU/tolerance checks only, NEVER a bit-exact oracle.

    Torch division cannot reproduce rcp.approx.ftz. Explicitly model FTZ/order,
    NaN-ignoring amax and saturating encodings to avoid the old recipe's clamps.
    """
    rows, cols = v.shape
    blocks = v.float().reshape(rows, cols // 16, 16)

    def ftz(x):
        return torch.where(
            x.abs() < torch.finfo(torch.float32).tiny, torch.copysign(torch.zeros_like(x), x), x
        )

    def mul(a, b):
        return ftz(ftz(a) * ftz(b))

    inverse = torch.ones_like(input_scale) / input_scale
    amax = blocks.abs()[..., 0]
    for i in range(1, 16):
        amax = torch.fmax(amax, blocks.abs()[..., i])
    sf = mul(inverse, mul(amax, torch.full_like(amax, 1 / 6)))
    # torch's E4M3 cast is not satfinite: emulate saturation, preserving NaN.
    sf = sf.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    factor = ftz(1 / mul(sf, ftz(1 / ftz(inverse))))
    factor = torch.where((ftz(amax) != 0) & ~amax.isnan(), factor, 0)
    value = mul(blocks, factor[..., None])
    magnitude = value.abs()
    code = sum((magnitude > t).to(torch.uint8) for t in (0.25, 1.25, 2.5, 5))
    code += sum((magnitude >= t).to(torch.uint8) for t in (0.75, 1.75, 3.5))
    code = torch.where(value.isnan(), 7, code).to(torch.uint8)
    code |= torch.signbit(value).to(torch.uint8) << 3
    code = code.reshape(rows, cols)
    return code[:, ::2] | (code[:, 1::2] << 4), sf.to(torch.float8_e4m3fn)
