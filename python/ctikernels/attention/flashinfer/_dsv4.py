# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FlashInfer 0.7.0.post1 entries behind DeepSeek-V4 sparse attention and row selection.

sm100/sm103: TRTLLM-GEN ``trtllm_batch_decode_sparse_mla_dsv4`` reads dense 512-wide BF16
rows, so the caller dequantizes each query's rows into a scratch first. sm120/sm121: the
packed paged attention reads the 584-byte pages directly (64-row primary pages; 2- or
64-row extra pages).
"""

from __future__ import annotations

import torch
from flashinfer import top_k_page_table_transform
from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

_TIE_LOWER_INDEX = 1


def dense_decode(q, scratch, table, seq_lens, topk_lens, sinks, scale, out, workspace):
    """Each query ``q [T, H, 512]`` reads scratch rows ``table [T, W]`` (-1: none).

    Every query is its own one-token request. The first 128 columns are the window part.
    """
    tokens, heads, dim = q.shape
    cache = scratch.view(1, 1, -1, dim)
    trtllm_batch_decode_sparse_mla_dsv4(
        q.view(tokens, 1, heads, dim),
        cache,
        workspace,
        sparse_indices=table,
        compressed_kv_cache=cache,
        sparse_topk_lens=topk_lens,
        seq_lens=seq_lens,
        out=out.view(tokens, 1, heads, dim),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        sinks=sinks,
        backend="trtllm-gen",
    )


def packed_view(cache, rows, page_bytes):
    """The ``[pages, rows, 1, 584]`` view FlashInfer's sparse backend reads (strided pages)."""
    pages = cache.numel() // page_bytes
    return cache.as_strided((pages, rows, 1, 584), (page_bytes, 584, 584, 1))


def packed_attention(q, swa, extra, sinks, scale, out, lse, mid_out, mid_lse):
    """sm12x: FlashInfer's packed paged attention, reading the 584-byte pages directly (the
    function SGLang's sm120 backend calls; its public DSV4 entry adds per-call checks and
    allocations). ``swa``/``extra`` are (view, indices, lengths); ``lse`` FP32 ``[T, H]``;
    ``mid_out``/``mid_lse`` the split-K scratch FlashInfer's decode kernels use for ``T <= 64``
    (``[>= T, Hs, >= splits, 512]`` BF16 and ``[>= T, Hs, >= splits]``, ``Hs`` = 8 for 8 heads, else H rounded up to 16;
    FP32, ``splits`` = index columns / 64 rounded up, summed over both caches)."""
    # FlashInfer loads its sm120 runtime lazily; so do we (sm10x never needs it).
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    swa_view, swa_idx, swa_len = swa
    ext_view, ext_idx, ext_len = extra if extra is not None else (None, None, None)
    _sparse_mla_sm120_paged_attention(
        q,
        swa_view,
        swa_idx,
        out,
        lse,
        scale,
        d_v=512,
        topk_length=swa_len,
        attn_sink=sinks,
        extra_kv_cache=ext_view,
        extra_indices=ext_idx,
        extra_topk_length=ext_len,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )


def topk_rows(scores, page_table, lengths, k, page_rows, out, row_to_batch, entries):
    """Top-``k`` scores per row (ties to the lower index) as physical cache rows into
    ``out``, and the selected entry ids into ``entries`` when given."""
    shape = (scores.shape[0], k)
    if out is None:
        out = torch.empty(shape, dtype=torch.int32, device=scores.device)
    top_k_page_table_transform(
        scores,
        page_table,
        lengths,
        k,
        row_to_batch=row_to_batch,
        deterministic=True,
        tie_break=_TIE_LOWER_INDEX,
        dsa_graph_safe=True,
        page_size=page_rows,
        out=out,
        out_raw_indices=entries,
    )
    return out


def cutlass_includes() -> tuple[str, ...]:
    """CUTLASS header folders shipped with the pinned FlashInfer (for local CUDA builds)."""
    return tuple(str(path) for path in CUTLASS_INCLUDE_DIRS)
