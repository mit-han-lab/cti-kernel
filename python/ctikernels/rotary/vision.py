# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply inference NeoX rotary embedding with per-token angles, out of place.

    ``query`` and ``key`` are CUDA FP16/BF16/FP32 tensors of one dtype and shape
    ``[tokens, heads, head_dim]`` with an even ``head_dim`` and a contiguous last
    dimension; strided views of a merged QKV projection are read without a copy.
    ``cos`` and ``sin`` are FP32 ``[tokens, head_dim]`` rows, for example the 2-D
    (height, width) patch angles of a vision encoder repeated across both halves.
    Each output is ``x * cos + rotate_half(x) * sin`` evaluated in FP32 without
    fused multiply-add, then rounded once to the input dtype.

    Inputs are borrowed without mutation and fresh contiguous ``(query, key)`` are
    returned. The operation follows the current CUDA stream and has no backward.
    """

    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, cos, sin)):
        raise RuntimeError("apply_rotary does not support autograd")
    from ._vision import apply_rotary as apply_per_token_rotary

    return apply_per_token_rotary(query, key, cos, sin)


__all__ = ["apply_rotary"]
