# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Portable linear provider."""

import torch
import torch.nn.functional as F

__all__ = ["linear"]


def linear(x, weight, out=None):
    if out is None:
        return F.linear(x, weight)
    return torch.mm(x, weight.t(), out=out)


def _mm_fp32_legacy_torch(x, weight, out=None):
    """Compatibility for torch builds without the mm out_dtype keyword."""
    return torch.mm(x.float(), weight.float().t(), out=out)


def _out_dtype_mm(x):
    """mm out_dtype is CUDA-only; other devices take the FP32-cast fallback."""
    return x.device.type == "cuda"


def linear_fp32(x, weight, out=None):
    """Market BF16-input, FP32-output cuBLAS call, without output rounding."""
    if not _out_dtype_mm(x):
        return _mm_fp32_legacy_torch(x, weight, out)
    try:
        return torch.mm(
            x, weight.t(), out_dtype=torch.float32, **({"out": out} if out is not None else {})
        )
    except TypeError as exc:
        if "out_dtype" not in str(exc):
            raise
        return _mm_fp32_legacy_torch(x, weight, out)
