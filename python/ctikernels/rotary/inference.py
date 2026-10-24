# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import functools

import torch

try:
    from flashinfer import apply_rope_with_cos_sin_cache_inplace as _apply_rotary
except (ImportError, OSError) as error:  # pragma: no cover - environment validation
    raise RuntimeError("inference rotary requires the formal `flashinfer-python` dependency") from (
        error
    )


def apply_rotary_(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
    position_axes: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply inference NeoX rotary embedding by overwriting Q and K.

    The caller owns ``query`` and ``key`` and must prove their previous values are
    dead. Positions are int64; another dtype raises TypeError before anything runs.
    They are ``[tokens]``, or with ``position_axes`` (the axis of each of the
    ``rotary_dim // 2`` frequencies, multi-axis rotary such as M-RoPE) contiguous
    ``[3, tokens]``: frequency ``f`` rotates by table row ``positions[position_axes[f], t]``.
    The operation follows the current CUDA stream, returns the same tensors, and
    has no backward.
    """

    if query.requires_grad or key.requires_grad:
        raise RuntimeError("apply_rotary_ cannot modify tensors that require gradients")
    if torch.is_grad_enabled() and cos_sin_cache.requires_grad:
        raise RuntimeError("apply_rotary_ does not support autograd")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")
    if positions.dim() == 2:
        if position_axes is None or positions.shape[0] != 3:
            raise ValueError("[3, tokens] positions need position_axes")
        # One table row per token, each column read at its frequency's axis position.
        axes, columns = _axis_columns(tuple(position_axes), positions.device)
        cos_sin_cache = cos_sin_cache[positions[axes].T, columns]
        positions = torch.arange(positions.shape[1], device=positions.device)
    _apply_rotary(
        positions=positions,
        query=query,
        key=key,
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
    )
    return query, key


@functools.cache
def _axis_columns(position_axes: tuple[int, ...], device: torch.device):
    """Each table column's position axis and the column indices, built once per device (an
    eager call, before capture, creates them)."""
    axes = torch.tensor(position_axes * 2, device=device)
    return axes, torch.arange(len(axes), device=device)


def apply_rotary_key_(
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    head_dim: int,
) -> torch.Tensor:
    """Overwrite only K with full-head inference NeoX rotary embedding.

    K is CUDA FP16/BF16, shaped ``[tokens, kv_heads * head_dim]`` or
    ``[tokens, kv_heads, head_dim]``. Non-overlapping strided views, including
    merged-QKV slices, are supported without copying. The FP32 cache has shape
    ``[max_position, head_dim]`` with half cosine then half sine; positions are
    CUDA int64 ``[tokens]`` on the same device and within the cache bounds.

    The caller owns K, proves its old values dead and its storage non-overlapping
    with itself/cache/positions, and retains all inputs through stream completion.
    Only K is written; its identical object is returned. No Q tensor is borrowed
    or aliased. This operation has no backward and follows the current stream;
    stable input storage supports CUDA graph capture after initial warmup.
    """
    if key.requires_grad:
        raise RuntimeError("apply_rotary_key_ cannot modify tensors that require gradients")
    if torch.is_grad_enabled() and cos_sin_cache.requires_grad:
        raise RuntimeError("apply_rotary_key_ does not support autograd")

    from ._key_only import apply_rotary_key

    apply_rotary_key(key, cos_sin_cache, positions, head_dim=head_dim)
    return key


__all__ = ["apply_rotary_", "apply_rotary_key_"]
