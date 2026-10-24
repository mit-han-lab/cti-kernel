# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Portable Triton DRAFT provider: the paged token-CSR kernel on fixed [B*Q] query rows."""

import torch
import triton
import triton.language as tl

from . import _paged


def check_support(device, **kwargs) -> None:
    _paged.device_traits(device)


def _launch(*, batch_size, query_length, num_qo_heads, num_kv_heads, head_dim, device, **kwargs):
    """``(segments, PDL)`` of a construction."""
    sms, pdl = _paged.device_traits(device)
    segments = _paged.split_count(
        sms=sms,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_seqs=batch_size,
        rows=batch_size * query_length,
        fixed_q=query_length,
    )
    return segments, pdl


def workspace_size(**configuration):
    """FP32 split partials (none for one segment); no integer workspace."""
    segments = _launch(**configuration)[0]
    if segments == 1:
        return 0, 0
    rows = configuration["batch_size"] * configuration["query_length"]
    size = (
        rows
        * segments
        * _paged.partial_bytes(configuration["num_qo_heads"], configuration["head_dim"])
    )
    return -(-size // 256) * 256, 0


@triton.jit
def _prepare(
    SLOTS,
    TOKEN_INDPTR,
    TOKEN_INDICES,
    QUERY_INDPTR,
    PAGE_INDPTR,
    PAGE_INDICES,
    LAST_PAGE_LENGTHS,
    BATCH: tl.constexpr,
    QUERY: tl.constexpr,
    PAGE: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    GATHER_TILE: tl.constexpr,
    WRITE_PAGES: tl.constexpr,
):
    """Query/page CSR of the token CSR; program b also gathers row b's page ids. Slots -1
    are masked before any token-CSR read."""
    row = tl.program_id(0)
    request = tl.arange(0, REQUEST_BLOCK)
    in_batch = request < BATCH
    active = in_batch & (tl.load(SLOTS + request, in_batch, other=-1) != -1)
    begin = tl.load(TOKEN_INDPTR + request, active, other=0)
    length = tl.load(TOKEN_INDPTR + request + 1, active, other=0) - begin
    pages = tl.cdiv(length, PAGE)
    page_ends = tl.cumsum(pages, 0)
    if row == 0:
        tl.store(QUERY_INDPTR, 0)
        tl.store(QUERY_INDPTR + request + 1, tl.cumsum(tl.where(active, QUERY, 0), 0), in_batch)
        tl.store(PAGE_INDPTR, 0)
        tl.store(PAGE_INDPTR + request + 1, page_ends, in_batch)
        tl.store(
            LAST_PAGE_LENGTHS + request, tl.where(active, (length - 1) % PAGE + 1, 0), in_batch
        )
    if WRITE_PAGES:
        mine = request == row
        count = tl.sum(tl.where(mine, pages, 0), 0)
        token_start = tl.sum(tl.where(mine, begin, 0), 0).to(tl.int64)
        page_start = tl.sum(tl.where(mine, page_ends - pages, 0), 0)
        for tile in range(0, tl.cdiv(count, GATHER_TILE)):
            page = tile * GATHER_TILE + tl.arange(0, GATHER_TILE)
            keep = page < count
            token = tl.load(TOKEN_INDICES + token_start + page.to(tl.int64) * PAGE, keep, other=0)
            tl.store(PAGE_INDICES + page_start + page, token // PAGE, keep)


class Provider:
    """One attention launch (plus an in-order reduction when keys are split) reading the
    token CSR directly; prepare only refreshes the caller's query/page CSR."""

    merge = "reduce"

    @property
    def pdl(self) -> bool:
        return self._pdl

    def __init__(
        self,
        float_workspace,
        int_workspace,
        *,
        prepare_inputs,
        query_indptr,
        kv_indptr,
        kv_indices,
        last_page_lengths,
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        window_left,
        query_row_stride=None,
    ):
        slots, token_indptr, token_indices, _, _ = prepare_inputs
        self._prepare_args = (
            slots,
            token_indptr,
            token_indices,
            query_indptr,
            kv_indptr,
            kv_indices if kv_indices is not None else query_indptr,
            last_page_lengths,
        )
        self._write_pages = kv_indices is not None
        self._token_csr = slots, token_indptr, token_indices
        self._batch, self._query, self._page = slots.numel(), query_length, page_size
        self._hk, self._d = num_kv_heads, head_dim
        self._segments, self._pdl = _launch(
            batch_size=slots.numel(),
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=slots.device,
        )
        self._scratch = float_workspace if self._segments > 1 else None
        # DRAFT windows are symmetric (|i - j| < W), intersected with causality.
        self._window = (window_left, 0 if causal else window_left)
        self._scale = head_dim**-0.5

    def prepare(self) -> None:
        _prepare[(self._batch,)](
            *self._prepare_args,
            BATCH=self._batch,
            QUERY=self._query,
            PAGE=self._page,
            REQUEST_BLOCK=triton.next_power_of_2(self._batch),
            GATHER_TILE=1024,
            WRITE_PAGES=self._write_pages,
            num_warps=4,
        )

    def prepare_and_run(self, *, q, paged_kv_cache, out):
        # Attention reads the token CSR, not prepare's outputs: no ordering between them.
        self.prepare()
        return self.run(q=q, paged_kv_cache=paged_kv_cache, out=out)

    def run(self, *, q, paged_kv_cache, out):
        slots, token_indptr, token_indices = self._token_csr
        k, v = (cache.view(-1, self._hk, self._d) for cache in paged_kv_cache)
        return _paged.attend(
            q,
            k,
            v,
            out,
            q_indptr=token_indptr,
            kv_indptr=token_indptr,
            kv_indices=token_indices,
            num_seqs=self._batch,
            fixed_q=self._query,
            segments=self._segments,
            window_left=self._window[0],
            window_right=self._window[1],
            scale=self._scale,
            slots=slots,
            scratch=self._scratch,
            pdl=self._pdl,
        )


class TableProvider:
    """One attention launch (plus the in-order reduction when keys are split) reading the
    engine tables directly: no CSR expansion."""

    merge = "reduce"

    @property
    def pdl(self) -> bool:
        return self._pdl

    def __init__(
        self,
        float_workspace,
        int_workspace,
        *,
        request_indices,
        frontier_table,
        done_table,
        cache_index_table,
        guard_start,
        max_seq_len,
        metadata_index="request",
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal,
        window_left,
        **kwargs,
    ):
        self._requests, self._indices = request_indices, cache_index_table
        self._table = (
            frontier_table,
            done_table.view(torch.uint8),
            cache_index_table.stride(0),
            guard_start,
            max_seq_len,
            metadata_index == "row",
        )
        self._batch, self._query = request_indices.numel(), query_length
        self._hk, self._d = num_kv_heads, head_dim
        self._segments, self._pdl = _launch(
            batch_size=self._batch,
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=request_indices.device,
        )
        self._scratch = float_workspace if self._segments > 1 else None
        self._window = (window_left, 0 if causal else window_left)
        self._scale = head_dim**-0.5

    def run(self, *, q, paged_kv_cache, out):
        k, v = (cache.view(-1, self._hk, self._d) for cache in paged_kv_cache)
        return _paged.attend(
            q,
            k,
            v,
            out,
            q_indptr=self._requests,
            kv_indptr=None,
            kv_indices=self._indices,
            num_seqs=self._batch,
            fixed_q=self._query,
            segments=self._segments,
            window_left=self._window[0],
            window_right=self._window[1],
            scale=self._scale,
            slots=self._requests,
            table=self._table,
            scratch=self._scratch,
            pdl=self._pdl,
        )
