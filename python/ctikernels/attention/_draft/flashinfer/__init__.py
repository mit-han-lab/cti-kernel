# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FlashInfer FA2 DRAFT adapter; native plans stay inside this package."""

import os
from importlib.metadata import PackageNotFoundError, version

import torch

from ...._storage import reject_grad
from ..._flashinfer import (
    FIXED_QUERY_MIN_CHUNK,
    FlashInferFixedQueryAttention,
    chunked_schedule,
    fixed_query_pdl,
    fixed_query_tile,
)


def check_capacity(
    *,
    batch_size,
    max_kv_tokens,
    page_size,
    query_length,
    window,
    causal=False,
    num_qo_heads=1,
    num_kv_heads=1,
    query_tile_size=None,
    query_tile_count=None,
    padded_ctas=None,
    **kwargs,
) -> None:
    """Host-only int32 admission for the DRAFT schedule, before allocation.

    Small Q retains its frozen bounds over all tiles and chunk tiers. Large Q
    uses the native graph tile and occupancy bound, with int64 intermediates.
    Prepare repeats the check with the actual plan's tile geometry.
    """
    b, c, p, q = batch_size, max_kv_tokens, page_size, query_length
    if not causal and window is not None and window < q:
        raise NotImplementedError("FA2 window_left has no right bound; noncausal W<Q requires CuTe")
    window = min(window, c) if window is not None else None
    pages = (c + p - 1) // p
    if q > 32:
        # Chunked prepare widens arithmetic; only stored native int32 values
        # constrain admission. Binary search fits split CTAs to the occupancy
        # budget, or uses one chunk per active request when Q tiles fill it.
        tile = query_tile_size or fixed_query_tile(
            b, q, num_qo_heads // num_kv_heads, kwargs.get("head_dim", 128)
        )
        count = query_tile_count or -(-q * (num_qo_heads // num_kv_heads) // tile)
        max_ctas = kwargs.get("max_ctas")
        if max_ctas is None:
            max_ctas = (
                2
                * torch.cuda.get_device_properties(kwargs["device"]).multi_processor_count
                // num_kv_heads
            )
        span = 0 if window is None else window - 1 + max(q, tile) + p - 1
        scheduled = pages if window is None else min(pages, -(-span // p))
        chunks = max(b, max_ctas // count)
        padded = padded_ctas or max(
            max_ctas, -(-b * q * (num_qo_heads // num_kv_heads) // tile) + b - 1
        )
        bounds = (
            b * c,
            b * pages,
            b * q,
            chunks * q,
            b * count,
            padded,
            max(scheduled, 128 // p, 1) * p,
        )
        if max(bounds) > 2**31 - 1:
            raise NotImplementedError(
                "FlashInfer DRAFT capacity exceeds native int32 schedule fields"
            )
        return
    requests = 1 << (b - 1).bit_length()
    gather = (pages + 1023) // 1024
    bounds = [b * c, c + p - 1, b * pages, b * q, (requests - 1) * q, 2 * b]
    bounds.extend((b * gather, gather * 1024 - 1))
    tiles = (16, 64, 128) if query_tile_size is None else (query_tile_size,)
    for tile in tiles:
        count = query_tile_count
        if count is None:
            count = (q * (num_qo_heads // num_kv_heads) + tile - 1) // tile
        scheduled = pages
        if window is not None:
            span = window - 1 + max(q, tile)
            bounds.append(span + p - 1)
            scheduled = min(pages, (span + p - 1) // p)
        minimum = max(128 // p, 1)
        chunks = (scheduled + minimum - 1) // minimum
        for tiers in FIXED_QUERY_MIN_CHUNK.values():
            for tokens, short in tiers:
                size = max(tokens // p, 1)
                high = min(scheduled, short // p)
                chunks = max(chunks, (high + size - 1) // size)
        largest_chunk = max(scheduled, minimum)
        # M=2**31-1; admit iff every bound below (and above) is <=M.
        # N=ceil(C/P) bounds page cumsum by B*N; C+P-1 is tl.cdiv's numerator.
        # S=min(N,ceil((W-1+tile)/P)) bounds scheduled pages (S=N without W).
        # low<=D=max(S,128//P,1): cdiv numerator<=S+D-1, low*P<=D*P;
        # while low<high<=S, low+high<=2*S-1. high*P<=D*P as well.
        # K bounds chunks/request over every short tier and the 128-token default;
        # binary search only increases the minimum. Thus sums/cumsums<=B*K,
        # query/tile offsets and merge additions<=B*K*max(Q,query_tile_count).
        # Masked lanes still compute: (pow2ceil(B)-1)*Q, gather*1024-1,
        # and tile indices through pow2ceil(padded_ctas)-1 must fit too.
        # Request search adds at most 2*B; merge lanes clamp to request B-1.
        # Remaining differences are >=-M, last-page arithmetic is in [-1,P],
        # and token gather multiplies/adds only after widening to int64.
        bounds.extend((scheduled + largest_chunk - 1, 2 * scheduled - 1, largest_chunk * p))
        bounds.append(b * chunks * max(q, count))
        # Native padding also covers packed query tiles across the whole batch.
        padded = (b * q * (num_qo_heads // num_kv_heads) + tile - 1) // tile + b - 1
        padded = max(padded, padded_ctas or 1)
        bounds.append((1 << (padded - 1).bit_length()) - 1)
    if max(bounds) > 2**31 - 1:
        raise NotImplementedError(
            "FlashInfer DRAFT capacity exceeds the frozen schedule's int32 intermediates "
            f"(B={b}, C={c}, P={p}, Q={q}, window={window}); use CuTe if supported"
        )


# The FA2 plan ABI this adapter drives (a different ABI from the fixed-query Split-KV
# layout in ``attention/_flashinfer.py``) was checked on each listed release with the draft,
# DFlash integration and capture-trace GPU suites on the RTX 5090, RTX PRO 6000 and GB10.
# Explicit versions only: a new release needs requalifying.
QUALIFIED_VERSIONS = ("0.6.11.post3", "0.7.0.post1", "0.7.1")


def check_support(device: torch.device) -> None:
    accepted = " or ".join(QUALIFIED_VERSIONS)
    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError as error:
        raise NotImplementedError(
            f"DraftAttention requires flashinfer-python {accepted}"
        ) from error
    if installed not in QUALIFIED_VERSIONS:
        raise NotImplementedError(
            f"DraftAttention FA2 plan ABI requires {accepted}; found {installed}"
        )
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)):
        raise NotImplementedError(f"DraftAttention FA2 is not supported on capability {capability}")
    # FA2 uses ordinary tensor-core MMA on SM12x, not the SM90 FA3 or SM100
    # TRTLLM paths. The latter's noncausal-window restriction does not apply.
    minimum = (12, 9) if capability == (12, 1) else (12, 8)
    if capability[0] >= 10 and tuple(map(int, torch.version.cuda.split(".")[:2])) < minimum:
        raise NotImplementedError(f"DraftAttention on {capability} needs CUDA >= {minimum}")


def fixed_query_bytes(batch_size: int) -> int:
    """Aligned int32 [B+1] fixed query CSR appended to the provider workspace."""
    return ((batch_size + 1) * 4 + 255) // 256 * 256


class Provider(FlashInferFixedQueryAttention):
    """FA2 on fixed [B*Q] query rows; inactive rows get no tiles and zero output."""

    _custom_mask = None
    _mask_indptr = None
    _trim_window = False
    _trim_window_left = -1
    _check_capacity = staticmethod(check_capacity)

    @property
    def pdl(self) -> bool:
        # Steady state: only the first call after each prepare() runs without PDL
        # (fixed_query_pdl); CTI_FIXED_QUERY_PDL=0 disables it on every call.
        return os.environ.get("CTI_FIXED_QUERY_PDL", "1") != "0"

    @property
    def merge(self) -> str:
        return "flashinfer"

    def __init__(
        self, float_workspace, int_workspace, *, prepare_inputs, query_row_stride=None, **kwargs
    ):
        batch_size = kwargs["last_page_lengths"].numel()
        native_int = int_workspace.numel() - fixed_query_bytes(batch_size)
        super().__init__(float_workspace, int_workspace.narrow(0, 0, native_int), **kwargs)
        from flashinfer.utils import MaskMode, TensorLayout

        self._fixed_query = int_workspace.narrow(0, native_int, (batch_size + 1) * 4).view(
            torch.int32
        )
        offset = self._layout.query_count_offset
        if offset < 0 or offset % 4 or offset + 4 > native_int:
            raise RuntimeError("FlashInfer device query count exceeds integer workspace")
        self._total_rows = int_workspace.narrow(0, offset, 4).view(torch.int32)
        self._prepare_inputs = prepare_inputs
        self._mask_mode = MaskMode.CAUSAL.value if kwargs["causal"] else MaskMode.NON_CAUSAL.value
        self._causal = kwargs["causal"]
        self._layout_code = TensorLayout.NHD.value
        self._scale = kwargs["head_dim"] ** -0.5
        # The public native run() obtains an unused global ALiBi allocation even
        # for PosEncodingMode.NONE. Invoke the pinned package entry point with
        # absent optional features instead. No provider-owned runtime GPU scratch.
        self._execute = self._native._cached_module.paged_run
        self._native._kv_lens_buffer = None  # only used during host setup / other backends

    def prepare_and_run(self, *, q, paged_kv_cache, out):
        # FA2 reads the device schedule that prepare writes: keep them ordered.
        self.prepare()
        return self.run(q=q, paged_kv_cache=paged_kv_cache, out=out)

    def run(self, *, q, paged_kv_cache, out):
        native = self._native
        k, v = paged_kv_cache
        # The native kernel reads its device schedule before griddepcontrol.wait:
        # the first call after prepare() runs without PDL (fixed_query_pdl).
        pdl = fixed_query_pdl(self)
        self._execute(
            float_workspace_buffer=native._float_workspace_buffer,
            int_workspace_buffer=native._int_workspace_buffer,
            plan_info_vec=native._plan_info,
            q=q,
            paged_k_cache=k,
            paged_v_cache=v,
            qo_indptr=self._fixed_query,
            paged_kv_indptr=self._kv_indptr,
            paged_kv_indices=native._paged_kv_indices_buf,
            paged_kv_last_page_len=native._paged_kv_last_page_len_buf,
            o=out,
            maybe_lse=None,
            mask_mode=self._mask_mode,
            layout=self._layout_code,
            window_left=self._window_left,
            enable_pdl=pdl,
            maybe_custom_mask=self._custom_mask,
            maybe_mask_indptr=self._mask_indptr,
            maybe_alibi_slopes=None,
            maybe_prefix_len_ptr=None,
            maybe_token_pos_in_items_ptr=None,
            maybe_max_item_len_ptr=None,
            logits_soft_cap=0.0,
            sm_scale=self._scale,
            scale_q=None,
            scale_k=None,
            scale_v=None,
            rope_scale=1.0,
            rope_theta=1e4,
            token_pos_in_items_len=0,
            workspace_size=native._float_workspace_buffer.numel(),
        )
        return out

    def prepare(self) -> None:
        slots, token_indptr, token_indices, query_indptr, max_kv_tokens = self._prepare_inputs
        b, page = self._batch_size, self._page_size
        self._check_capacity(
            batch_size=b,
            max_kv_tokens=max_kv_tokens,
            page_size=page,
            query_length=self._query_length,
            window=None if self._window_left < 0 else self._window_left + 1,
            causal=self._causal,
            query_tile_size=self._layout.query_tile_size,
            query_tile_count=self._query_tiles,
            padded_ctas=self._layout.padded_ctas,
            max_ctas=self._max_ctas if self._query_length > 32 else None,
        )
        tile = 1024
        page_tiles = ((max_kv_tokens + page - 1) // page + tile - 1) // tile
        reject_grad(
            slots,
            token_indptr,
            token_indices,
            query_indptr,
            self._kv_indptr,
            self._native._paged_kv_indices_buf,
            self._native._paged_kv_last_page_len_buf,
            self._fixed_query,
            self._total_rows,
            *self._schedule,
        )
        from ..._flashinfer_schedule import prepare_draft, prepare_draft_chunked

        if chunked_schedule(b, self._query_length, self._layout.padded_ctas):
            prepare_draft = prepare_draft_chunked

        self._serialize_next = True
        prepare_draft[(b * page_tiles,)](
            slots,
            token_indptr,
            token_indices,
            query_indptr,
            self._kv_indptr,
            self._native._paged_kv_indices_buf,
            self._native._paged_kv_last_page_len_buf,
            self._fixed_query,
            self._total_rows,
            *self._schedule,
            BATCH=b,
            QUERY=self._query_length,
            PAGE=page,
            PAGE_TILES=page_tiles,
            GATHER_TILE=tile,
            QUERY_TILE_COUNT=self._query_tiles,
            MAX_CTAS=self._max_ctas,
            PADDED_CTAS=self._layout.padded_ctas,
            REQUEST_BLOCK=1 << (b - 1).bit_length(),
            TILE_BLOCK=1 << (self._layout.padded_ctas - 1).bit_length(),
            MERGE_BLOCK=1 << (b * self._query_length - 1).bit_length(),
            SEARCH_STEPS=b.bit_length(),
            WINDOW_LEFT=self._window_left,
            QUERY_TILE_SIZE=self._layout.query_tile_size,
            MIN_CHUNK=self._chunk_tiers[0][0],
            SHORT_TOKENS=self._chunk_tiers[0][1],
            MIN_CHUNK_2=self._chunk_tiers[1][0],
            SHORT_TOKENS_2=self._chunk_tiers[1][1],
            MIN_CHUNK_3=self._chunk_tiers[2][0],
            SHORT_TOKENS_3=self._chunk_tiers[2][1],
            TRIM_WINDOW=self._trim_window,
            TRIM_WINDOW_LEFT=self._trim_window_left,
            WINDOWED=self._trim_window_left >= 0,
            num_warps=4,
        )


def table_bytes(batch_size, max_kv_tokens):
    # Slots, token/query/page pointers, token/page indices and last-page lengths.
    words = 2 * batch_size * max_kv_tokens + 5 * batch_size + 3
    return (words * 4 + 255) // 256 * 256


class TableProvider:
    """Private CSR expansion followed by the existing FA2 prepare and run."""

    @property
    def pdl(self) -> bool:
        return self._provider.pdl

    @property
    def merge(self) -> str:
        return self._provider.merge

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
        **kwargs,
    ):
        from ._table_csr import compile_table_csr

        b, c = request_indices.numel(), max_seq_len + kwargs["query_length"]
        native_bytes = int_workspace.numel() - table_bytes(b, c)
        private = int_workspace[native_bytes:].view(torch.int32)
        offset = 0

        def take(count):
            nonlocal offset
            result = private.narrow(0, offset, count)
            offset += count
            return result

        slots, ptr, indices = take(b), take(b + 1), take(b * c)
        query, pages, page_indices, last = take(b + 1), take(b + 1), take(b * c), take(b)
        torch.arange(b, dtype=torch.int32, device=slots.device, out=slots)
        self._provider = Provider(
            float_workspace,
            int_workspace[:native_bytes],
            prepare_inputs=(slots, ptr, indices, query, c),
            query_indptr=query,
            kv_indptr=pages,
            kv_indices=page_indices,
            last_page_lengths=last,
            **kwargs,
        )
        flat = cache_index_table.as_strided(
            (
                (cache_index_table.shape[0] - 1) * cache_index_table.stride(0)
                + cache_index_table.shape[1],
            ),
            (1,),
        )
        self._inputs = (
            request_indices,
            frontier_table,
            done_table.view(torch.uint8),
            flat,
            ptr,
            indices,
            cache_index_table.stride(0),
            guard_start,
            max_seq_len,
        )
        major, minor = torch.cuda.get_device_capability(request_indices.device)
        self._expand = compile_table_csr(
            kwargs["query_length"], c, f"sm_{major}{minor}", metadata_index == "row"
        )

    def run(self, *, q, paged_kv_cache, out):
        self._expand(*self._inputs)
        return self._provider.prepare_and_run(q=q, paged_kv_cache=paged_kv_cache, out=out)
