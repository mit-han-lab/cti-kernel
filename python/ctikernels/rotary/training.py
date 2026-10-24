# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch

try:
    from flash_attn.layers.rotary import apply_rotary_emb as _apply_rotary_emb
except (ImportError, OSError) as error:  # pragma: no cover - exercised by environment checks
    raise RuntimeError("training rotary requires the formal `flash-attn` dependency") from error


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    max_sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply differentiable NeoX rotary embedding without mutating Q or K.

    Padded Q/K use ``[batch, sequence, heads, head_dim]`` and one shared,
    consecutive int64 ``positions`` vector. Packed Q/K use
    ``[tokens, heads, head_dim]``; ``cu_seqlens`` divides ``positions`` into
    consecutive ranges and each range may start at a different offset.

    ``cos`` and ``sin`` are contiguous FP32 caches of one shape
    ``[max_position, rotary_dim / 2]``, where ``rotary_dim`` is at most the
    ``head_dim`` that Q and K share. The leading ``rotary_dim`` channels of each
    head rotate, the first half of that span against the second; the remaining
    ``head_dim - rotary_dim`` channels and their gradients pass through unchanged
    (partial rotary). ``rotary_dim == head_dim`` rotates the full head.

    The operation follows the current CUDA stream and returns fresh storage.
    Callers own semantic validation of the position ranges; this hot-path leaf
    performs no device-content scan.

    BF16/FP16 execution uses the optimized provider. FP32 preserves the eager
    PyTorch reference used by deterministic correctness trajectories.
    """

    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")
    if cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("cos and sin must share shape [max_position, rotary_dim / 2]")
    if cu_seqlens is None:
        if max_sequence_length is not None:
            raise ValueError("max_sequence_length is only valid for packed rotary")
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("padded query and key must have shape [batch, sequence, heads, dim]")
        if query.shape[:2] != key.shape[:2] or positions.shape != (query.shape[1],):
            raise ValueError("padded query, key, and positions must share batch sequence geometry")
        cache_shape = (1, query.shape[1], 1, cos.shape[1])
    else:
        if query.ndim != 3 or key.ndim != 3:
            raise ValueError("packed query and key must have shape [tokens, heads, dim]")
        if query.shape[0] != key.shape[0] or positions.shape != (query.shape[0],):
            raise ValueError("packed query, key, and positions must share the token count")
        if max_sequence_length is None:
            raise ValueError("packed rotary requires max_sequence_length")
        cache_shape = (query.shape[0], 1, cos.shape[1])

    head_dim = query.shape[-1]
    rotary_dim = 2 * cos.shape[1]
    if key.shape[-1] != head_dim:
        raise ValueError("query and key must share head_dim")
    if not 0 < rotary_dim <= head_dim:
        raise ValueError(
            f"rotary_dim {rotary_dim} must be positive and at most head_dim {head_dim}"
        )

    if query.dtype == key.dtype == torch.float32:
        selected_cos = cos.index_select(0, positions).view(cache_shape)
        selected_sin = sin.index_select(0, positions).view(cache_shape)

        def apply_reference(x: torch.Tensor) -> torch.Tensor:
            rotated, passed = x.float().split((rotary_dim, head_dim - rotary_dim), dim=-1)
            first, second = rotated.chunk(2, dim=-1)
            return torch.cat(
                (
                    first * selected_cos - second * selected_sin,
                    second * selected_cos + first * selected_sin,
                    passed,
                ),
                dim=-1,
            ).to(x.dtype)

        return apply_reference(query), apply_reference(key)

    position_offsets = (
        positions[:1].expand(query.shape[0])
        if cu_seqlens is None
        else positions.index_select(0, cu_seqlens[:-1])
    )

    common = {
        "interleaved": False,
        "inplace": False,
        "seqlen_offsets": position_offsets,
        "cu_seqlens": cu_seqlens,
        "max_seqlen": max_sequence_length,
    }
    return (
        _apply_rotary_emb(query, cos, sin, **common),
        _apply_rotary_emb(key, cos, sin, **common),
    )


__all__ = ["apply_rotary"]
