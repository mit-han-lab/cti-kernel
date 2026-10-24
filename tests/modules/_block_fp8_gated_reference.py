# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference design: SGLang 72730d6 silu_and_mul_clamp / calculate_fp8_scales.

"""CPU torch oracle: materialized activation followed by per-128 quantization.

CPU exp/div and CUDA approximate exp/div can differ near a BF16/FP16 midpoint;
the GPU suite also checks the old CUDA two-kernel reference exhaustively.
"""

import torch
import torch.nn.functional as F


def activation(x, swiglu_limit=None):
    dtype = x.dtype
    if swiglu_limit is not None:
        x = x.bfloat16()
    g, u = x.float().chunk(2, dim=-1)
    if swiglu_limit is not None:
        limit = torch.tensor(swiglu_limit, dtype=torch.bfloat16).float()
        g = torch.fmin(g, limit)
        u = torch.fmin(torch.fmax(u, -limit), limit)
    return (g / (1 + torch.exp(-g)) * u).to(dtype)


def quantize(x, scale_ue8m0=False):
    m, k = x.shape
    groups = -(-k // 128)
    values = F.pad(x.float(), (0, groups * 128 - k)).reshape(m, groups, 128)
    magnitudes = values.abs()
    if scale_ue8m0:
        magnitudes = torch.nan_to_num(magnitudes, nan=0.0, posinf=float("inf"))
    amax = magnitudes.amax(dim=-1).clamp_min(1e-10)
    scale = amax * (1.0 / 448.0)
    if scale_ue8m0:
        # An integer exponent avoids log2 rounding just above exact powers of two.
        bits = scale.view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0)
        scale = (exponent << 23).view(torch.float32)
        multiplier = ((254 - exponent) << 23).view(torch.float32)
    else:
        multiplier = torch.div(torch.full_like(amax, 448.0), amax)
    scaled = values * multiplier[..., None]
    if scale_ue8m0:
        scaled = torch.fmin(torch.fmax(scaled, torch.tensor(-448.0)), torch.tensor(448.0))
    else:
        scaled = scaled.clamp(-448, 448)
    q = scaled.to(torch.float8_e4m3fn).reshape(m, groups * 128)[:, :k].contiguous()
    return q, scale


def silu_quantize(x, *, swiglu_limit=None, scale_ue8m0=False):
    return quantize(activation(x, swiglu_limit), scale_ue8m0)
