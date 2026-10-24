# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Differentiable attention within each sequence of a packed token batch.

Both operations wrap FlashAttention's varlen function and differ only in the mask.
``q`` is ``[num_tokens, num_heads, head_dim]``; ``k`` and ``v`` share its head
dimension, with a head count that divides ``num_heads``. Inputs are matching
FP16 or BF16 CUDA tensors with a contiguous head dimension, so merged-QKV slices
need no copy. Head dimensions follow the provider (at most 256), and scores are scaled by
``head_dim ** -0.5`` unless ``softmax_scale`` is supplied. ``cu_seqlens`` is a
contiguous int32 CUDA vector of ``num_sequences + 1`` boundaries that starts at 0, never decreases and ends at
``num_tokens``; there are at most ``num_tokens`` sequences, and empty ones are
allowed. ``max_sequence_length`` is a host integer no smaller than any sequence
length. Supply both ``cu_seqlens_k`` and ``max_sequence_length_k`` for a different
packed KV extent with the same sequence count. Otherwise Q/KV share boundaries.
Causal masking aligns bottom-right: query i sees keys j <= Lkv - Lq + i.
With max query length 1, supply one query token per sequence (the provider's GQA
path maps query row i to sequence i).

A forward returns a new ``[num_tokens, num_heads, head_dim]`` tensor, follows
PyTorch's current CUDA stream and differentiates through the provider's fast,
nondeterministic backward. It checks tensor metadata only: callers validate
cumulative-length values before copying them to CUDA, and no D2H copy or host
synchronization occurs.

Without autograd, a forward may be captured in a CUDA graph. Capture fixes the
input addresses, both token counts, the sequence count and both maximum lengths.
Replays may rewrite q, k, v and both cumulative-length vectors in place while
every sequence stays within its captured maximum; each replay overwrites
the output tensor returned during capture.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _bind_flash_attn_varlen(operation: str) -> Callable[..., torch.Tensor]:
    """Resolve the provider entry point once, so a missing build fails at construction."""
    unavailable = (
        f"{operation} requires `flash-attn`; install a compatible build providing "
        "`flash_attn.flash_attn_interface.flash_attn_varlen_func`"
    )
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        raise RuntimeError(unavailable) from error

    if not callable(flash_attn_varlen_func):
        raise RuntimeError(unavailable)
    return flash_attn_varlen_func


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_sequence_length: int,
    cu_seqlens_k: torch.Tensor | None = None,
    max_sequence_length_k: int | None = None,
) -> None:
    """Check tensor metadata only; cumulative-length values are never read on the host."""
    if (cu_seqlens_k is None) != (max_sequence_length_k is None):
        raise ValueError("cu_seqlens_k and max_sequence_length_k must be supplied together")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [num_tokens, num_heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if (cu_seqlens_k is None and q.shape[0] != k.shape[0]) or q.shape[2] != k.shape[2]:
        raise ValueError("q, k, and v must share the token count and head dimension")
    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError("the query-head count must be a multiple of the key/value-head count")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must use matching FP16 or BF16")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a vector describing at least one sequence")
    # With max_sequence_length 1 this leaves exactly one token per sequence, which
    # FlashAttention's single-query GQA path assumes when it maps row i to sequence i.
    if cu_seqlens.numel() > q.shape[0] + 1:
        raise ValueError("cu_seqlens cannot describe more sequences than packed tokens")
    if cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must use int32")
    if cu_seqlens.stride(0) != 1:
        raise ValueError("cu_seqlens must be contiguous")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must be contiguous in the head dimension")
    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise TypeError("max_sequence_length must be an integer")
    if max_sequence_length < 1 or max_sequence_length > q.shape[0]:
        raise ValueError("max_sequence_length must be within the packed token extent")
    if cu_seqlens_k is not None:
        if cu_seqlens_k.ndim != 1 or cu_seqlens_k.numel() != cu_seqlens.numel():
            raise ValueError("cu_seqlens_k must describe the same sequence count as cu_seqlens")
        if cu_seqlens_k.dtype != torch.int32:
            raise TypeError("cu_seqlens_k must use int32")
        if cu_seqlens_k.stride(0) != 1:
            raise ValueError("cu_seqlens_k must be contiguous")
        if isinstance(max_sequence_length_k, bool) or not isinstance(max_sequence_length_k, int):
            raise TypeError("max_sequence_length_k must be an integer")
        if not 1 <= max_sequence_length_k <= k.shape[0]:
            raise ValueError("max_sequence_length_k must be within the packed KV token extent")
        if not cu_seqlens_k.is_cuda or cu_seqlens_k.device != q.device:
            raise ValueError("cu_seqlens_k must reside on the same CUDA device as q")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not cu_seqlens.is_cuda:
        raise ValueError("q, k, v, and cu_seqlens must reside on CUDA")
    if k.device != q.device or v.device != q.device or cu_seqlens.device != q.device:
        raise ValueError("q, k, v, and cu_seqlens must reside on the same CUDA device")


class CausalVarlenAttention:
    """Causal attention with bottom-right alignment for unequal query/KV lengths."""

    def __init__(self) -> None:
        self._flash_attn_varlen = _bind_flash_attn_varlen("CausalVarlenAttention")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_sequence_length_k: int | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        _validate_inputs(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
            cu_seqlens_k=cu_seqlens_k,
            max_sequence_length_k=max_sequence_length_k,
        )
        return self._flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens if cu_seqlens_k is None else cu_seqlens_k,
            max_seqlen_q=max_sequence_length,
            max_seqlen_k=max_sequence_length
            if max_sequence_length_k is None
            else max_sequence_length_k,
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            deterministic=False,
        )


class NoncausalVarlenAttention:
    """Each token attends to every token of its own sequence, as vision-encoder patches do."""

    def __init__(self) -> None:
        self._flash_attn_varlen = _bind_flash_attn_varlen("NoncausalVarlenAttention")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_sequence_length: int,
        cu_seqlens_k: torch.Tensor | None = None,
        max_sequence_length_k: int | None = None,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        _validate_inputs(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_sequence_length=max_sequence_length,
            cu_seqlens_k=cu_seqlens_k,
            max_sequence_length_k=max_sequence_length_k,
        )
        return self._flash_attn_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens if cu_seqlens_k is None else cu_seqlens_k,
            max_seqlen_q=max_sequence_length,
            max_seqlen_k=max_sequence_length
            if max_sequence_length_k is None
            else max_sequence_length_k,
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=False,
            deterministic=False,
        )


__all__ = ["CausalVarlenAttention", "NoncausalVarlenAttention"]
