# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original FlashInfer inference provider; imports remain lazy."""

import torch

from ctikernels._flashinfer_version import require_allowed


def silu_and_mul(x, out=None):
    require_allowed()
    _validate(x, out)
    from flashinfer import silu_and_mul as operation

    # FlashInfer activation accepts [..., 2 * width] and preserves all prefix axes.
    return operation(x, out=out)


def gelu_and_mul(x, out=None):
    require_allowed()
    _validate(x, out)
    from flashinfer import gelu_and_mul as operation

    return operation(x, out=out)


def gelu_tanh_and_mul(x, out=None):
    require_allowed()
    _validate(x, out)
    from flashinfer import gelu_tanh_and_mul as operation

    return operation(x, out=out)


def _validate(x, out):
    # FlashInfer 0.7.0.post1 activation.py's vector ABI, without silent packing.
    if (
        not x.is_cuda
        or x.dtype not in (torch.float16, torch.bfloat16)
        or not x.is_contiguous()
        or x.shape[-1] * x.element_size() % 16
        or (out is not None and not out.is_contiguous())
    ):
        raise ValueError(
            "FlashInfer activation requires contiguous CUDA FP16/BF16 "
            "and an input row size divisible by 16 bytes"
        )
