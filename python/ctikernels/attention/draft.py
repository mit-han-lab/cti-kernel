# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Padded, bottom-right-aligned DRAFT attention (K0.17–21)."""

from __future__ import annotations

import os

import torch

from ctikernels._storage import _byte_span, disjoint, reject_grad

from ._flashinfer import FlashInferFixedQueryAttention


def _geometry(
    *,
    batch_size,
    query_length,
    max_kv_tokens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    causal,
    window,
    device,
):
    values = (
        batch_size,
        query_length,
        max_kv_tokens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )
    if any(type(x) is not int or x < 1 for x in values):
        raise ValueError("DRAFT dimensions must be positive integers")
    if max_kv_tokens < query_length:
        raise ValueError("DRAFT requires Q >= 1 and C >= Q")
    if batch_size * max_kv_tokens >= 2**31 or batch_size * query_length >= 2**31:
        raise ValueError("DRAFT CSR capacities must fit signed int32")
    if num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("DRAFT requires FP16 or BF16")
    if type(causal) is not bool or (window is not None and (type(window) is not int or window < 1)):
        raise ValueError("causal must be bool and window must be a positive int or None")
    if head_dim not in (64, 128, 256):
        raise NotImplementedError("DRAFT FA2 supports head dimensions 64/128/256")
    if torch.device(device).type != "cuda":
        raise ValueError("DraftAttention requires a CUDA device")


def _provider(provider, configuration):
    from ._draft import cute, dispatch, flashinfer, flashinfer_mask

    selected = provider if provider is not None else os.environ.get("CTI_DRAFT_PROVIDER")
    if selected not in (None, "cute", "flashinfer", "flashinfer-mask", "triton"):
        raise ValueError(
            "DRAFT provider must be 'cute', 'flashinfer', 'flashinfer-mask', 'triton', or None"
        )
    if selected == "triton":
        from .triton import _draft as triton

        triton.check_support(**configuration)
        return triton
    if selected == "flashinfer-mask":
        flashinfer_mask.check_capacity(**configuration)
        return flashinfer_mask
    if selected == "cute":
        cute.check_support(**configuration)
        return cute
    if selected is None:
        try:
            cute.check_support(**configuration)
        except NotImplementedError:
            pass
        else:
            if not dispatch.prefers_flashinfer(**configuration):
                return cute
            try:
                flashinfer.check_capacity(**configuration)
            except NotImplementedError:
                return cute
            return flashinfer
    if (
        selected is None
        and not configuration["causal"]
        and configuration["window"] is not None
        and configuration["window"] < configuration["query_length"]
    ):
        flashinfer_mask.check_capacity(**configuration)
        return flashinfer_mask
    flashinfer.check_capacity(**configuration)
    return flashinfer


def _workspace(implementation, configuration):
    from ._draft import flashinfer

    if implementation is not flashinfer:
        return implementation.workspace_size(**configuration)
    floating, integer = FlashInferFixedQueryAttention.workspace_size(
        **{
            key: configuration[key]
            for key in (
                "batch_size",
                "query_length",
                "num_qo_heads",
                "num_kv_heads",
                "head_dim",
                "device",
            )
        }
    )
    return floating, integer + flashinfer.fixed_query_bytes(configuration["batch_size"])


def _tensor(tensor, shape, dtype, device, name):
    reject_grad(tensor)
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {shape} on the operation device")


def _query(tensor, shape, dtype, device, row_stride):
    reject_grad(tensor)
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or tensor.stride() != (row_stride, shape[2], 1)
    ):
        raise ValueError(
            f"q requires {dtype} {shape} on the operation device with strides "
            f"({row_stride}, {shape[2]}, 1) matching construction-time query_row_stride"
        )


def _operand_alignment(q, k, v, out):
    # Both CuTe and FlashInfer FA2 use 16-byte accesses; data_ptr is host metadata.
    if any(tensor.data_ptr() % 16 for tensor in (q, k, v, out)):
        raise ValueError("DRAFT requires 16-byte aligned q/k/v/out base pointers")


class DraftAttention:
    """Borrow operands/scratch; construct/warm up outside capture, prepare each replay.

    Masks intersect K0.17's symmetric window with optional causality. The engine proves
    K0.18's contiguous logical layout, int64 positions, Q<=L<=C, no post-block
    keys, and page-aligned physical token segments. Position values never pass through a provider int32 cast: attention
    uses translation-invariant relative positions L-Q+i. Full attention requires
    base zero; sliding suffixes must retain every needed key.

    Any construction-time Q >= 1 is accepted, subject to provider grid/int32 bounds.
    Q=1 is aligned to the final key (L-1). For Q>32 auto prefers FA2;
    noncausal W<Q uses CuTe when supported, otherwise FlashInfer FA2 with a
    custom mask (including SM80..SM103). Explicit provider="flashinfer-mask"
    supports every mask and reports "flashinfer-fa2-custom-mask". It borrows
    packed mask/byte-indptr storage from the integer workspace, sized using
    construction-time W, and retains only the page-aligned window union.
    Its prepare uses two device launches; it requires page_indices.
    Explicit provider="triton" (the portable floor, CUDA 8.9+) supports every
    mask, reads the token CSR directly and reports "triton-splitkv".

    Any integer W >= 1 is accepted. W >= capacity C is normalized to None
    before provider selection: every legal distance is at most C-1.
    query_row_stride=None requires contiguous Q (Hq*D,D,1). A fused-QKV column
    view must declare its row stride once at construction because the CuTe kernel
    folds it into its address arithmetic. Every run requires exactly (s,D,1),
    s >= Hq*D, with 16-byte aligned rows/base;
    K/V and output remain contiguous. page_indices may be None for CuTe;
    requires_page_indices() reports the selected provider's requirement.

    Slots -1 cause no reads of that row's Q/K/V/token CSR; query/page segments
    are empty and output is zero. Other slots are caller-admitted identities.
    Buffers must remain stable and ordered through prepare -> run -> reuse.

    The CuTe provider owns N bytes of zeroed device memory for merge counters:
    N = 4 * B * Hkv * row_tiles * sum(level_group_counts), where each level
    groups the previous count by compile-time fan-in G, from maximum splits to
    one root. N is zero for one split or legacy Merge and is excluded from
    workspace_size(). Every complete call leaves the counters zero. One
    DraftAttention instance must not run two calls concurrently on different
    streams; calls ordered on one stream or graph are fine. Two instances never
    share counters, even when they borrow the same scratch workspace.
    """

    @staticmethod
    def workspace_size(
        *,
        batch_size,
        query_length,
        max_kv_tokens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        window,
        device,
        provider: str | None = None,
    ):
        """Return uint8 capacities; both bases require 256-byte alignment.

        Provider capacities are aligned payload bounds. FlashInfer also reserves
        a private fixed query CSR. Allocator backing is caller-owned and may be larger.
        Provider selection and split tuning must match construction.
        """
        configuration = locals().copy()
        configuration.pop("provider")
        _geometry(**configuration)
        configuration["window"] = None if window is not None and window >= max_kv_tokens else window
        configuration["device"] = torch.device(device)
        return _workspace(_provider(provider, configuration), configuration)

    @staticmethod
    def requires_page_indices(
        *,
        batch_size,
        query_length,
        max_kv_tokens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        window,
        device,
        provider: str | None = None,
    ) -> bool:
        """Whether the selected provider needs a caller-owned page-index buffer."""
        configuration = locals().copy()
        configuration.pop("provider")
        from ._draft import flashinfer, flashinfer_mask

        _geometry(**configuration)
        configuration["window"] = None if window is not None and window >= max_kv_tokens else window
        configuration["device"] = torch.device(device)
        return _provider(provider, configuration) in (flashinfer, flashinfer_mask)

    @property
    def provider(self) -> str:
        """Selected attention provider, fixed at construction."""
        return self._provider_name

    @property
    def pdl(self) -> bool:
        """Effective attention PDL; may change once after a warned eager fallback."""
        return self._implementation.pdl

    @property
    def merge(self) -> str:
        """Effective merge mode; may change once after a warned eager fallback."""
        return self._implementation.merge

    def __init__(
        self,
        float_workspace,
        int_workspace,
        *,
        request_indices,
        positions,
        token_indptr,
        token_indices,
        query_indptr,
        page_indptr,
        page_indices,
        last_page_lengths,
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        causal,
        window,
        provider: str | None = None,
        query_row_stride: int | None = None,
    ):
        if request_indices.ndim != 1 or request_indices.numel() < 1:
            raise ValueError("request_indices must have shape [B], B>=1")
        b = request_indices.numel()
        if token_indices.ndim != 1 or token_indices.numel() % b:
            raise ValueError("token_indices must have shape [B*C]")
        c = token_indices.numel() // b
        device = request_indices.device
        configuration = {
            "batch_size": b,
            "query_length": query_length,
            "max_kv_tokens": c,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "page_size": page_size,
            "dtype": dtype,
            "causal": causal,
            "window": window,
            "device": device,
        }
        _geometry(**configuration)
        row_stride = num_qo_heads * head_dim if query_row_stride is None else query_row_stride
        if (
            type(row_stride) is not int
            or row_stride < num_qo_heads * head_dim
            or row_stride * dtype.itemsize % 16
        ):
            raise ValueError("query_row_stride must be an int >= Hq*D with 16-byte aligned rows")
        self._query_row_stride = row_stride
        window = None if window is not None and window >= c else window
        configuration["window"] = window
        implementation = _provider(provider, configuration)
        if page_indices is None and implementation.__name__.endswith(
            (".flashinfer", ".flashinfer_mask")
        ):
            raise ValueError("page_indices is required by the FlashInfer DRAFT provider")
        self._metadata = (
            (request_indices, (b,), torch.int32, "request_indices"),
            (positions, (b * query_length,), torch.int64, "positions"),
            (token_indptr, (b + 1,), torch.int32, "token_indptr"),
            (token_indices, (b * c,), torch.int32, "token_indices"),
            (query_indptr, (b + 1,), torch.int32, "query_indptr"),
            (page_indptr, (b + 1,), torch.int32, "page_indptr"),
            (page_indices, (b * ((c + page_size - 1) // page_size),), torch.int32, "page_indices"),
            (last_page_lengths, (b,), torch.int32, "last_page_lengths"),
        )
        self._metadata = tuple(
            entry for entry in self._metadata if entry[3] != "page_indices" or entry[0] is not None
        )
        self._device = device
        self._validate_metadata()
        if implementation.__name__.endswith((".flashinfer", ".flashinfer_mask")):
            implementation.check_support(device)
        required = _workspace(implementation, configuration)
        for workspace, size in zip((float_workspace, int_workspace), required, strict=True):
            _tensor(workspace, (workspace.numel(),), torch.uint8, device, "workspace")
            if workspace.numel() < size or workspace.data_ptr() % 256:
                raise ValueError(f"workspace requires {size} bytes and 256-byte alignment")
        self._readonly = (request_indices, positions, token_indptr, token_indices)
        self._writable = (
            query_indptr,
            page_indptr,
            page_indices,
            last_page_lengths,
            float_workspace,
            int_workspace,
        )
        self._writable = tuple(value for value in self._writable if value is not None)
        disjoint(self._writable, self._readonly)
        self._provider_name = {
            "_draft.cute": "cute-splitkv",
            "_draft.flashinfer_mask": "flashinfer-fa2-custom-mask",
            "_draft.flashinfer": "flashinfer-fa2-device-schedule",
            "triton._draft": "triton-splitkv",
        }[implementation.__name__.split(".", 2)[2]]
        self._b, self._q, self._c, self._p = b, query_length, c, page_size
        self._hq, self._hk, self._d, self._dtype = num_qo_heads, num_kv_heads, head_dim, dtype
        self.workspace_report = {
            "alignment": 256,
            "float_payload_bound_bytes": required[0],
            "int_payload_bound_bytes": required[1],
            "float_aligned_bytes": required[0],
            "int_aligned_bytes": required[1],
            "float_allocator_backing_bytes": float_workspace.untyped_storage().nbytes(),
            "int_allocator_backing_bytes": int_workspace.untyped_storage().nbytes(),
        }
        self._implementation = implementation.Provider(
            float_workspace.narrow(0, 0, required[0]),
            int_workspace.narrow(0, 0, required[1]),
            prepare_inputs=(request_indices, token_indptr, token_indices, query_indptr, c),
            query_indptr=query_indptr,
            kv_indptr=page_indptr,
            kv_indices=page_indices,
            last_page_lengths=last_page_lengths,
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            causal=causal,
            window_left=-1 if window is None else window - 1,
            query_row_stride=query_row_stride,
        )
        self._bind()

    def _bind(self):
        """Fix the per-call operand contract once, so run() checks it in a few compares.

        Shapes, strides, dtype and device of q/k/v/out follow from construction-time
        values; only the operands' base pointers and the cache's page count vary.
        """
        rows, heads, dim, size = self._b * self._q, self._hq, self._d, self._dtype.itemsize
        self._call_shape = torch.Size((rows, heads, dim))
        self._query_strides = (self._query_row_stride, dim, 1)
        self._out_strides = (heads * dim, dim, 1)
        self._cache_tail = torch.Size((self._p, self._hk, dim))
        self._cache_strides = (self._p * self._hk * dim, self._hk * dim, dim, 1)
        self._page_bytes = self._p * self._hk * dim * size
        self._query_bytes = ((rows - 1) * self._query_row_stride + heads * dim) * size
        self._out_bytes = rows * heads * dim * size
        # Tensor.get_device(): -1 on CPU, the ordinal on CUDA.
        self._device_index = -1 if self._device.type == "cpu" else self._device.index
        self._bind_buffers()

    def _bind_buffers(self):
        """Check bound buffers are pairwise disjoint and record their byte intervals."""
        writable = tuple(
            (t.data_ptr(), t.data_ptr() + _byte_span(t)) for t in self._writable if t.numel()
        )
        readonly = tuple(
            (t.data_ptr(), t.data_ptr() + _byte_span(t)) for t in self._readonly if t.numel()
        )
        for i, (begin, end) in enumerate(writable):
            if any(begin < e and b < end for b, e in writable[i + 1 :] + readonly):
                raise ValueError("writable buffers must not overlap each other or read-only inputs")
        self._writable_spans, self._bound_spans = writable, writable + readonly

    def _validate_metadata(self):
        for value, shape, dtype, name in self._metadata:
            _tensor(value, shape, dtype, self._device, name)
            if dtype == torch.int32 and value.data_ptr() % 4:
                raise ValueError(f"{name} requires a 4-byte aligned base pointer")

    def prepare(self) -> None:
        """Refresh query/page CSR and provider schedule/mask on device.

        Also revalidates the bound metadata and workspaces (layout, alignment,
        disjointness); run() then checks only the per-layer operands.
        """
        self._validate_metadata()
        self._bind_buffers()
        self._implementation.prepare()

    def prepare_and_run(self, *, q, paged_kv_cache, out):
        """prepare() then run() for the first layer sharing this prepare.

        Once this call completes in stream order, output and metadata outputs equal
        those of the two calls in sequence. Providers may overlap the two: the CuTe
        provider's attention CTAs write prepare's outputs, saving one launch.
        """
        self._validate_metadata()
        self._bind_buffers()
        self._operands(q, paged_kv_cache, out)
        return self._implementation.prepare_and_run(q=q, paged_kv_cache=paged_kv_cache, out=out)

    def run(self, *, q, paged_kv_cache, out):
        """Write caller-owned output with no allocation or host metadata planning.

        Bound metadata and workspaces are validated at construction and by each
        prepare(); run() checks q/k/v/out against the bound contract.
        """
        self._operands(q, paged_kv_cache, out)
        # Fixed [B*Q] rows: inactive rows have no tiles (Q unread) and the
        # provider's merge writes their output as zeros.
        return self._implementation.run(q=q, paged_kv_cache=paged_kv_cache, out=out)

    def _operands(self, q, paged_kv_cache, out):
        """Accept exactly what _check_operands accepts, through precomputed compares."""
        if type(paged_kv_cache) is tuple and len(paged_kv_cache) == 2:
            k, v = paged_kv_cache
            dtype, index, shape = self._dtype, self._device_index, self._call_shape
            if (
                q.dtype is dtype
                and k.dtype is dtype
                and v.dtype is dtype
                and out.dtype is dtype
                and not (q.requires_grad or k.requires_grad or v.requires_grad or out.requires_grad)
                and q.get_device() == index
                and k.get_device() == index
                and v.get_device() == index
                and out.get_device() == index
                and q.shape == shape
                and out.shape == shape
                and q.stride() == self._query_strides
                and out.stride() == self._out_strides
                and k.stride() == self._cache_strides
                and v.stride() == self._cache_strides
                and k.shape[1:] == self._cache_tail
                and v.shape == k.shape
                and k.shape[0] >= 1
            ):
                qp, kp, vp, op = q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr()
                if not (qp | kp | vp | op) & 15:
                    cache_bytes = k.shape[0] * self._page_bytes
                    reads = (
                        (qp, qp + self._query_bytes),
                        (kp, kp + cache_bytes),
                        (vp, vp + cache_bytes),
                    )
                    oe = op + self._out_bytes
                    if not any(b < oe and op < e for b, e in reads + self._bound_spans) and not any(
                        b < read_end and read < e
                        for b, e in self._writable_spans
                        for read, read_end in reads
                    ):
                        return
        # Anything else (including valid but unusual strides of size-1 dimensions)
        # takes the complete check, which raises the specific error.
        self._check_operands(q, paged_kv_cache, out)

    def _check_operands(self, q, paged_kv_cache, out):
        self._validate_metadata()
        shape = (self._b * self._q, self._hq, self._d)
        _query(q, shape, self._dtype, self._device, self._query_row_stride)
        _tensor(out, shape, self._dtype, self._device, "out")
        if not isinstance(paged_kv_cache, tuple) or len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache must be a (k,v) tuple")
        k, v = paged_kv_cache
        if k.ndim != 4 or k.shape[0] < 1:
            raise ValueError("K/V must have shape [pages,P,Hkv,d]")
        cache_shape = (k.shape[0], self._p, self._hk, self._d)
        _tensor(k, cache_shape, self._dtype, self._device, "k")
        _tensor(v, cache_shape, self._dtype, self._device, "v")
        _operand_alignment(q, k, v, out)
        disjoint((out,) + self._writable, (q, k, v) + self._readonly)


def _table_configuration(*, max_seq_len, **configuration):
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("max_seq_len must be a positive integer")
    query = configuration["query_length"]
    if type(query) is not int or not 2 <= query <= 32:
        raise ValueError("DRAFT requires 2 <= Q <= 32")
    if max_seq_len + query >= 2**31:
        raise ValueError("M + Q must fit signed int32")
    configuration.update(max_kv_tokens=max_seq_len + query, page_size=1)
    _geometry(**configuration)
    configuration["device"] = torch.device(configuration["device"])
    return configuration


def _table_workspace(implementation, configuration):
    from ._draft import cute, flashinfer

    if implementation is cute:
        if cute.configuration(**configuration)[-1] > 1 and not cute.merge_tuning()[0]:
            raise ValueError("table attention requires an inline merge")
    floating, integer = _workspace(implementation, configuration)
    if implementation is flashinfer:
        integer += flashinfer.table_bytes(
            configuration["batch_size"], configuration["max_kv_tokens"]
        )
    return floating, integer


class CacheTableDraftAttention(DraftAttention):
    """Bottom-right DRAFT/VERIFY attention directly on mutable engine tables.

    Requests are distinct, in-range row identities (never -1). Out-of-range
    int64 frontiers behave as done rows. Construct outside capture; run reads
    the current device metadata without allocation or host synchronization.
    Workspaces, operands and tables are borrowed. CuTe owns merge counters as
    described by DraftAttention; calls on an instance must be stream-ordered.
    """

    @staticmethod
    def workspace_size(
        *,
        batch_size,
        query_length,
        max_seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        causal,
        window,
        device,
        provider=None,
    ):
        configuration = locals().copy()
        configuration.pop("provider")
        configuration = _table_configuration(**configuration)
        return _table_workspace(_provider(provider, configuration), configuration)

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
        query_length,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        causal,
        window,
        provider=None,
        metadata_index="request",
    ):
        """metadata_index="request" reads frontier_table[r]/done_table[r] (R5 tables);
        "row" reads frontier_table[b]/done_table[b] of shape [B] (R5 P's
        start_frontiers/round_done), one dependent load fewer per call."""
        reject_grad(
            float_workspace,
            int_workspace,
            request_indices,
            frontier_table,
            done_table,
            cache_index_table,
        )
        if request_indices.ndim != 1 or request_indices.numel() < 1:
            raise ValueError("request_indices must have shape [B], B>=1")
        b, device = request_indices.numel(), request_indices.device
        configuration = _table_configuration(
            batch_size=b,
            query_length=query_length,
            max_seq_len=max_seq_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            causal=causal,
            window=window,
            device=device,
        )
        if (
            type(guard_start) is not int
            or guard_start < 0
            or guard_start + b * query_length >= 2**31
        ):
            raise ValueError("guard_start + B*Q must fit signed int32")
        if frontier_table.ndim != 1 or frontier_table.numel() < 1:
            raise ValueError("frontier_table must be a nonempty 1-D tensor")
        if metadata_index not in ("request", "row"):
            raise ValueError("metadata_index must be 'request' or 'row'")
        if metadata_index == "row" and frontier_table.numel() != b:
            raise ValueError("row-indexed frontier/done tensors must have shape [B]")
        self._device = device
        self._metadata = (
            (request_indices, (b,), torch.int32, "request_indices"),
            (frontier_table, frontier_table.shape, torch.int64, "frontier_table"),
            (done_table, frontier_table.shape, torch.bool, "done_table"),
        )
        self._table, self._maximum = cache_index_table, max_seq_len
        self._validate_metadata()
        # Include row padding in alias checks: disjoint() expects dense byte intervals.
        self._flat_table = cache_index_table.as_strided(
            (
                (cache_index_table.shape[0] - 1) * cache_index_table.stride(0)
                + cache_index_table.shape[1],
            ),
            (1,),
        )
        self._table_layout = (cache_index_table.shape, cache_index_table.stride())
        implementation = _provider(provider, configuration)
        if implementation.__name__.endswith(".flashinfer"):
            implementation.check_support(device)
        required = _table_workspace(implementation, configuration)
        for workspace, size in zip((float_workspace, int_workspace), required, strict=True):
            _tensor(workspace, (workspace.numel(),), torch.uint8, device, "workspace")
            if workspace.numel() < size or workspace.data_ptr() % 256:
                raise ValueError(f"workspace requires {size} bytes and 256-byte alignment")
        self._readonly = (request_indices, frontier_table, done_table, self._flat_table)
        self._writable = (float_workspace, int_workspace)
        disjoint(self._writable, self._readonly)
        self._b, self._q, self._p = b, query_length, 1
        # Queries are contiguous [B*Q,Hq,D] on this interface.
        self._query_row_stride = num_qo_heads * head_dim
        self._hq, self._hk, self._d, self._dtype = num_qo_heads, num_kv_heads, head_dim, dtype
        self._guard_end = guard_start + b * query_length
        is_cute = implementation.__name__.endswith(".cute")
        is_triton = implementation.__name__.endswith(".triton._draft")
        self._provider_name = (
            "cute-table"
            if is_cute
            else ("triton-table" if is_triton else "flashinfer-fa2-table-csr")
        )
        # FI paged_run launches attention and its merge, in addition to both prepares;
        # Triton launches attention and, when it splits keys, the reduction.
        if is_triton:
            self.kernels_per_call = 1 + (implementation.workspace_size(**configuration)[0] > 0)
        else:
            self.kernels_per_call = 1 if is_cute else 4
        self._implementation = implementation.TableProvider(
            float_workspace[: required[0]],
            int_workspace[: required[1]],
            request_indices=request_indices,
            frontier_table=frontier_table,
            done_table=done_table,
            cache_index_table=cache_index_table,
            guard_start=guard_start,
            max_seq_len=max_seq_len,
            metadata_index=metadata_index,
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=1,
            dtype=dtype,
            causal=causal,
            window_left=-1 if window is None else window - 1,
        )
        self._bind()

    def _validate_metadata(self):
        super()._validate_metadata()
        table = self._table
        reject_grad(table)
        if (
            table.ndim != 2
            or table.shape[0] < 1
            or table.dtype != torch.int32
            or table.device != self._device
            or table.stride(1) != 1
            or table.stride(0) < table.shape[1]
            or table.shape[1] < self._maximum
            or table.data_ptr() % 4
        ):
            raise ValueError(
                "cache_index_table must be int32 [R,W], W>=M, with unit column stride, nonoverlapping rows and a 4-byte aligned base"
            )
        if hasattr(self, "_table_layout") and (
            (table.shape, table.stride()) != self._table_layout
            or table.data_ptr() != self._flat_table.data_ptr()
        ):
            raise ValueError("cache_index_table layout must remain stable")

    def run(self, *, q, paged_kv_cache, out):
        # No prepare() here: the tables are revalidated on every call.
        self._validate_metadata()
        self._bind_buffers()
        self._operands(q, paged_kv_cache, out)
        if paged_kv_cache[0].shape[0] < self._guard_end:
            raise ValueError("K/V rows must cover guard_start + B*Q")
        return self._implementation.run(q=q, paged_kv_cache=paged_kv_cache, out=out)

    def prepare(self):
        raise NotImplementedError("CacheTableDraftAttention reads metadata directly in run()")

    def prepare_and_run(self, *, q, paged_kv_cache, out):
        return self.run(q=q, paged_kv_cache=paged_kv_cache, out=out)


__all__ = ["DraftAttention", "CacheTableDraftAttention"]
