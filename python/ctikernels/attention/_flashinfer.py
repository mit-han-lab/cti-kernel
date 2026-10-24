# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2023 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer fa381e5162d9b6ab5479b77e5b6d287b732d108e
# (`include/flashinfer/attention/scheduler.cuh`, `include/flashinfer/utils.cuh`,
# `include/flashinfer/allocator.h`); see THIRD_PARTY_NOTICES.

"""FlashInfer 0.6.11.post3 / 0.7.0.post1 / 0.7.1 private plan layout and device workspace sizing."""

from __future__ import annotations

import functools
import os
from importlib.metadata import PackageNotFoundError, version
from typing import NamedTuple

import torch

from ._split_tables import prefill_graph_config, prefill_graph_rows, prefill_graph_split_layout

# The FA2 Split-KV plan layout was checked on each listed release (GPU suites on the
# RTX 5090, RTX PRO 6000 and GB10). Explicit versions only: a new release needs requalifying.
QUALIFIED_VERSIONS = ("0.6.11.post3", "0.7.0.post1", "0.7.1")


@functools.cache
def qualified_flashinfer_version() -> bool:
    """Check both the distribution and the imported module's qualification (once per process)."""
    import flashinfer

    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError:
        return False
    return installed in QUALIFIED_VERSIONS and getattr(flashinfer, "__version__", None) == installed


STAGED_OUTPUT_VERSIONS = ("0.6.11.post3", "0.7.0.post1")


def staged_output_flashinfer_version() -> bool:
    """Output-module swapping has a narrower qualification than plan layout reads."""
    return qualified_flashinfer_version() and version("flashinfer-python") in STAGED_OUTPUT_VERSIONS


class BF16OutputModule:
    """Run an FP16 plan with BF16 output on the two qualified releases.

    Both decode.plan (0.6.11.post3) / decode._plan_impl (0.7.0.post1)
    and paged prefill.plan use the getter arguments below. FA2 adds the
    fp16-QK-reduction flag; CUDA-core decode does not. Runtime options (scale,
    causal mask, RoPE, PDL) remain on the native wrapper and its existing plan.

    In both scheduler.cuh versions, DecodePlan and PrefillPlan reserve tmp_v
    and tmp_s using sizeof(float). The run kernels interpret tmp_v as DTypeO;
    FP16/BF16 have equal size, so offsets and the existing schedule remain valid.
    CUDA-core planning queries occupancy, but run consumes the supplied schedule
    without replanning. Both Python runners read _cached_module and
    _cached_o_data_type on each call, then multiply BF16 out by v_scale in place.
    FlashInfer has no FP32 output. No provider implementation is copied here.
    """

    def __init__(self, native, *, head_dim, idx_dtype, decode):
        if not staged_output_flashinfer_version():
            raise RuntimeError("unqualified FlashInfer output-module swap")
        from flashinfer.decode import get_batch_decode_module
        from flashinfer.prefill import get_batch_prefill_module
        from flashinfer.utils import PosEncodingMode

        if native._jit_module is not None or native._cached_o_data_type != torch.float16:
            raise ValueError("BF16 fallback requires a native FP16 output plan")
        prefill = not decode or native.use_tensor_cores
        if prefill and native._backend != "fa2":
            raise ValueError("BF16 fallback requires FA2 or CUDA-core decode")
        args = (
            native._cached_q_data_type,
            native._cached_kv_data_type,
            torch.bfloat16,
            idx_dtype,
            head_dim,
            head_dim,
            PosEncodingMode[native._pos_encoding_mode].value,
            native._window_left != -1 if decode else native._window_left >= 0,
            native._logits_soft_cap > 0,
        )
        self.module = (
            get_batch_prefill_module(
                native._backend, *args, False if decode else native._use_fp16_qk_reduction
            )
            if prefill
            else get_batch_decode_module(*args)
        )

    def run(self, native, **kwargs):
        module, dtype = native._cached_module, native._cached_o_data_type
        try:
            native._cached_module = self.module
            native._cached_o_data_type = torch.bfloat16
            return native.run(**kwargs)
        finally:
            native._cached_module, native._cached_o_data_type = module, dtype


def custom_mask_metadata(custom_mask, *metadata):
    """Prefill metadata the provider can pack a boolean ``custom_mask`` with.

    0.6.11.post3 packs on the mask's device with an indptr derived on the query
    indptr's device (prefill.py:1876-1888), so host metadata fails for a CUDA mask;
    0.7.0.post1 moves that indptr itself (prefill.py:2495-2511). Older providers get
    device copies, and their planner then reads them back synchronously.
    """
    if not custom_mask.is_cuda or version("flashinfer-python").startswith("0.7."):
        return metadata
    return tuple(value.to(custom_mask.device) for value in metadata)


def decode_plan_capacity(native) -> tuple[int, tuple[int, ...]] | None:
    """Read the qualified FA2 tensor-core decode capacity and opaque layout."""
    if (
        not qualified_flashinfer_version()
        or not native.use_tensor_cores
        or native._backend != "fa2"
    ):
        return None
    values = getattr(native, "_plan_info", None)
    if values is None or len(values) != len(_PlanLayout._fields):
        return None
    # Tensor-core decode uses PrefillPlanInfo::ToVector()[0] = padded_batch_size:
    # attention/scheduler.cuh:651 (0.6.11.post3), :721 (0.7.0.post1), :720 (0.7.1).
    layout = tuple(values)
    return int(layout[0]), layout


def decode_split_capacity(device, num_kv_heads: int) -> int:
    """Work items per KV head that every FA2 decode plan admits, eager or graph.

    PrefillPlan allots 2 CTAs per SM, max_batch_size_if_split = 2*SMs // Hkv
    (scheduler.cuh:717-720 / :788-791, no colocated CTAs). Graph plans pad to
    max(that, query tiles), eager plans to the work they need (:602-603 /
    :672-673), so a split within this bound fits both modes alike.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return 2 * sms // num_kv_heads


class _PlanLayout(NamedTuple):
    """The provider's serialized FA2 layout, confined to this compatibility module."""

    padded_ctas: int
    query_count: int
    query_count_offset: int
    query_tile_size: int
    requests_offset: int
    query_tiles_offset: int
    kv_tiles_offset: int
    merge_offsets_offset: int
    output_offsets_offset: int
    chunk_size_offset: int
    partial_output_offset: int
    partial_lse_offset: int
    valid_offset: int
    cuda_graph: int
    split_kv: int

    @classmethod
    def from_native(cls, values, *, batch_size: int, query_length: int) -> _PlanLayout:
        if len(values) != len(cls._fields):
            raise RuntimeError("unsupported FlashInfer fixed-query plan layout")
        plan = cls(*values)
        if (
            not plan.cuda_graph
            or not plan.split_kv
            or plan.query_count != batch_size * query_length
            or plan.query_tile_size not in (16, 64, 128)
            or plan.padded_ctas < batch_size
        ):
            raise RuntimeError("FlashInfer fixed-query plan does not match the requested geometry")
        return plan

    def views(self, workspace: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, ...]:
        fields = (
            (self.requests_offset, self.padded_ctas, torch.int32),
            (self.query_tiles_offset, self.padded_ctas, torch.int32),
            (self.kv_tiles_offset, self.padded_ctas, torch.int32),
            (self.output_offsets_offset, batch_size + 1, torch.int32),
            (self.merge_offsets_offset, self.query_count + 1, torch.int32),
            (self.chunk_size_offset, 1, torch.int32),
            (self.valid_offset, self.padded_ctas, torch.bool),
        )
        result = []
        for offset, count, dtype in fields:
            width = 1 if dtype == torch.bool else 4
            if offset < 0 or offset % width or offset + count * width > workspace.numel():
                raise RuntimeError("FlashInfer schedule exceeds its borrowed integer workspace")
            result.append(workspace.narrow(0, offset, count * width).view(dtype))
        return tuple(result)


def fixed_query_tile(batch: int, query: int, group: int, head_dim: int) -> int:
    """FA2 graph tile rule on SM80+: scheduler.cuh PrefillSplitQOKVIndptr,
    utils.cuh FA2DetermineCtaTileQ (qualified FlashInfer versions above).
    This is a description of the planner contract, not copied implementation.
    """
    longest_packed = (batch * query - batch + 1) * group
    if longest_packed <= 16:
        return 16
    return 128 if longest_packed > 64 and head_dim < 256 else 64


def chunked_schedule(batch: int, query: int, padded_ctas: int) -> bool:
    """Keep the small-Q kernel unless a vector would exceed Triton's block limit."""
    return query > 32 or max(batch * query, padded_ctas) > 2**20


class FlashInferFixedQueryAttention:
    """Fixed B/Q paged attention with exact, device-resident changing KV lengths.

    Setup initializes borrowed CUDA int32 CSR buffers for uniform ``query_length``
    queries and a dummy KV prefix. Callers replace KV indptr/indices/last-page
    lengths before execution; each request must have KV length >= query length.
    Query/output are FP16/BF16 NHD; KV may independently use provider-supported
    dtypes including FP8 e4m3/e5m2, with run-time k_scale/v_scale. Head dimensions
    are equal. Causal/noncausal attention supports a left window, sm_scale and
    logits_soft_cap, but no custom mask. Tuned minimum chunks apply only to equal
    FP16/BF16 dtypes and default window/softcap; other cases use native 128 tokens.

    Any fixed query_length >= 1 is supported within native grid/int32 bounds.
    Fixed query describes B/Q geometry, not a bound on the changing KV length.
    The caller owns logical KV validity; this operation only builds the provider's
    schedule from that already-valid metadata. It does not admit or commit tokens.

    Call ``prepare()`` once after each device KV metadata update, before all layers
    sharing this operation. Preparation is capture-safe and allocation-free;
    neither preparation nor execution introduces host synchronization or H2D/D2H.
    Run does not mutate query/K/V; supply ``out`` to avoid output allocation.
    Workspace and metadata addresses
    must remain stable; callers serialize preparation, attention and shared scratch
    on one stream. The caller owns capture and all borrowed-buffer lifetimes.

    Construction may synchronize and must remain outside capture. The float
    workspace is caller-zeroed on first allocation, not cleared on reuse. Private
    FlashInfer plan compatibility is intentionally confined here; adding a provider
    version requires kernel qualification, not changes to engine scheduling.

    BF16 Q with FP8 KV owns an FP16 output stage of B * Q * Hq * D * 2 bytes,
    allocated at construction, separate from borrowed workspace_size() scratch.
    For |v_scale| <= 1 (None means 1), one CuTe launch scales in FP32 and rounds
    to BF16, with absolute floor ~2^-24 * |v_scale|. Otherwise, including inf/NaN,
    native BF16 output uses the same plan. FlashInfer has no FP32 output.
    Capture bakes the host scale and output route.
    """

    @staticmethod
    def workspace_size(
        *,
        batch_size: int,
        query_length: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[int, int]:
        """Borrowed byte capacities, independent of actual or maximum KV length.

        Excludes owned BF16-Q/FP8-KV staging: B * Q * Hq * D * 2 bytes.
        FP16 staging applies for |v_scale| <= 1 (None means 1), with absolute
        floor ~2^-24 * |v_scale|; otherwise native BF16 output uses the same plan.
        FlashInfer has no FP32 output.
        """
        if min(batch_size, query_length, num_qo_heads, num_kv_heads, head_dim) < 1:
            raise ValueError("fixed-query attention dimensions must be positive")
        if num_qo_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if head_dim not in (64, 128, 256):
            raise ValueError("fixed-query attention supports Q>=1 and head dimensions 64/128/256")
        if batch_size * query_length >= 2**31:
            raise ValueError("fixed-query query CSR must fit signed int32")
        properties = torch.cuda.get_device_properties(device)
        if properties.major < 8:
            raise ValueError("fixed-query attention requires an SM80 or newer CUDA device")
        max_ctas = 2 * properties.multi_processor_count // num_kv_heads
        packed_queries = batch_size * query_length * (num_qo_heads // num_kv_heads)
        # FA2 graph planning may use query tiles of 16, 64 or 128. Bound the
        # allocated tile rows and CTA count without duplicating its tile heuristic.
        tile_rows = max(max_ctas * 128, ((packed_queries + 127) // 128 + batch_size - 1) * 128)
        padded_ctas = max(max_ctas, (packed_queries + 15) // 16 + batch_size - 1)
        if query_length > 32:
            tile = fixed_query_tile(
                batch_size, query_length, num_qo_heads // num_kv_heads, head_dim
            )
            padded_ctas = max(max_ctas, (packed_queries + tile - 1) // tile + batch_size - 1)
            tile_rows = padded_ctas * tile
        float_bytes = num_qo_heads * tile_rows * (head_dim + 1) * 4 + 32
        int_bytes = (
            (3 * padded_ctas + batch_size + batch_size * query_length + 5) * 4 + padded_ctas + 128
        )
        return ((float_bytes + 255) // 256 * 256, (int_bytes + 255) // 256 * 256)

    def __init__(
        self,
        float_workspace: torch.Tensor,
        int_workspace: torch.Tensor,
        *,
        query_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        last_page_lengths: torch.Tensor,
        query_length: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        causal: bool,
        kv_dtype: torch.dtype | None = None,
        sm_scale: float | None = None,
        logits_soft_cap: float | None = None,
        window_left: int = -1,
        min_chunk_tokens: int | None = None,
    ) -> None:
        # Paged workspace bounds share this qualification.
        installed = version("flashinfer-python")
        if installed not in QUALIFIED_VERSIONS:
            raise RuntimeError(
                "fixed-query Split-KV is qualified with flashinfer-python=="
                f"{' or =='.join(QUALIFIED_VERSIONS)}; found {installed}"
            )
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        device, batch_size = kv_indptr.device, last_page_lengths.numel()
        required = self.workspace_size(
            batch_size=batch_size,
            query_length=query_length,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=device,
        )
        if dtype not in (torch.float16, torch.bfloat16) or page_size < 1:
            raise ValueError("fixed-query attention requires FP16/BF16 and a positive page size")
        buffers = (query_indptr, kv_indptr, kv_indices, last_page_lengths)
        if device.type != "cuda" or any(
            value.device != device
            or value.dtype != torch.int32
            or value.ndim != 1
            or not value.is_contiguous()
            for value in buffers
        ):
            raise ValueError(
                "fixed-query metadata must be contiguous CUDA int32 vectors on one device"
            )
        if query_indptr.numel() != batch_size + 1 or kv_indptr.numel() != batch_size + 1:
            raise ValueError("fixed-query indptr dimensions do not match the batch")
        for workspace, capacity in zip((float_workspace, int_workspace), required, strict=True):
            if (
                workspace.device != device
                or workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
                or workspace.numel() < capacity
            ):
                raise ValueError(
                    f"fixed-query attention needs at least {capacity} borrowed workspace bytes"
                )
        self._batch_size, self._query_length, self._page_size = batch_size, query_length, page_size
        self._window_left = window_left
        kv_dtype = dtype if kv_dtype is None else kv_dtype
        if min_chunk_tokens is not None and not 1 <= min_chunk_tokens <= 128:
            raise ValueError("min_chunk_tokens must be in [1, 128]")
        if kv_dtype != dtype or window_left != -1 or logits_soft_cap not in (None, 0.0):
            tiers = ((128, 0),)
        elif min_chunk_tokens is None:
            tiers = fixed_query_min_chunk(device, head_dim=head_dim)
        else:
            # An explicit override covers every admitted int32 token length.
            tiers = ((min_chunk_tokens, (1 << 31) - 1),)
        self._chunk_tiers = tiers + ((128, 0),) * (3 - len(tiers))
        self._kv_indptr = kv_indptr
        self._max_ctas = (
            2 * torch.cuda.get_device_properties(device).multi_processor_count // num_kv_heads
        )
        self._native = BatchPrefillWithPagedKVCacheWrapper(
            float_workspace,
            kv_layout="NHD",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buf=query_indptr,
            paged_kv_indptr_buf=kv_indptr,
            paged_kv_indices_buf=kv_indices,
            paged_kv_last_page_len_buf=last_page_lengths,
        )
        self._native.reset_workspace_buffer(float_workspace, int_workspace)
        pages = (query_length + page_size - 1) // page_size
        if kv_indices.numel() < batch_size * pages:
            raise ValueError("fixed-query indices cannot hold the initial query blocks")
        # All initial values are known from geometry: no GPU length readback,
        # even when constructing the first plan for an uncaptured eager shape.
        cpu_q = torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * query_length
        cpu_kv = torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * pages
        cpu_last = torch.full(
            (batch_size,), (query_length - 1) % page_size + 1, dtype=torch.int32, device="cpu"
        )
        self._fp8_output = None
        output_options = {}
        if (
            dtype == torch.bfloat16
            and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and staged_output_flashinfer_version()
        ):
            from ._scale_cast import FP16StagedOutput

            self._fp8_output = FP16StagedOutput(
                (batch_size * query_length, num_qo_heads, head_dim), device
            )
            output_options["o_data_type"] = torch.float16
        self._native.plan(
            qo_indptr=cpu_q,
            paged_kv_indptr=cpu_kv,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=cpu_last,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=causal,
            window_left=window_left,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=kv_dtype,
            sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap,
            disable_split_kv=False,
            **output_options,
        )
        if self._fp8_output is not None:
            self._fp8_output.prepare(
                self._native, head_dim=head_dim, idx_dtype=cpu_kv.dtype, decode=False
            )
        # The native planner's pinned staging is asynchronous. Complete it before
        # schedule warmup or releasing constructor-local metadata.
        torch.cuda.current_stream(device).synchronize()
        self._layout = _PlanLayout.from_native(
            self._native._plan_info, batch_size=batch_size, query_length=query_length
        )
        self._schedule = self._layout.views(int_workspace, batch_size)
        self._query_tiles = (
            query_length * (num_qo_heads // num_kv_heads) + self._layout.query_tile_size - 1
        ) // self._layout.query_tile_size

    def prepare(self) -> None:
        """Update private FA2 scheduling from device metadata on the current stream.

        Mutates only borrowed provider scratch. Does not allocate, copy metadata
        to/from the host, or update caller-owned query/KV indices and lengths.
        Prepare once per metadata update, not once per model layer.
        """
        from .._storage import reject_grad

        reject_grad(self._kv_indptr, *self._schedule)
        from ._flashinfer_schedule import prepare_split_kv, prepare_split_kv_chunked

        if chunked_schedule(self._batch_size, self._query_length, self._layout.padded_ctas):
            prepare_split_kv = prepare_split_kv_chunked

        self._serialize_next = True

        prepare_split_kv[(1,)](
            self._kv_indptr,
            *self._schedule,
            BATCH=self._batch_size,
            QUERY=self._query_length,
            PAGE=self._page_size,
            QUERY_TILE_COUNT=self._query_tiles,
            MAX_CTAS=self._max_ctas,
            PADDED_CTAS=self._layout.padded_ctas,
            REQUEST_BLOCK=1 << (self._batch_size - 1).bit_length(),
            TILE_BLOCK=1 << (self._layout.padded_ctas - 1).bit_length(),
            MERGE_BLOCK=1 << (self._batch_size * self._query_length - 1).bit_length(),
            SEARCH_STEPS=self._batch_size.bit_length(),
            WINDOW_LEFT=self._window_left,
            QUERY_TILE_SIZE=self._layout.query_tile_size,
            MIN_CHUNK=self._chunk_tiers[0][0],
            SHORT_TOKENS=self._chunk_tiers[0][1],
            MIN_CHUNK_2=self._chunk_tiers[1][0],
            SHORT_TOKENS_2=self._chunk_tiers[1][1],
            MIN_CHUNK_3=self._chunk_tiers[2][0],
            SHORT_TOKENS_3=self._chunk_tiers[2][1],
            num_warps=4,
        )

    def run(
        self,
        *,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        """Execute with the latest prepared schedule; ``out`` avoids output allocation.

        Stream order: the first run after each prepare() launches without PDL; later
        runs launch with PDL (host call order, also as captured into a graph).
        """
        options = {
            name: value
            for name, value in (("k_scale", k_scale), ("v_scale", v_scale))
            if value is not None
        }
        check_dense_output(out)
        if out is not None:
            options["out"] = out
        pdl = fixed_query_pdl(self)
        if self._fp8_output is not None:
            return self._fp8_output.run(
                self._native,
                q=q,
                paged_kv_cache=paged_kv_cache,
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
                pdl=pdl,
            )
        return self._native.run(q=q, paged_kv_cache=paged_kv_cache, enable_pdl=pdl, **options)


# Per (major, minor, head_dim): ordered (minimum chunk, inclusive longest-request
# token bound) tiers. The first matching tier applies; above the last bound use
# the native 128-token minimum. Per-layer graph receipt:
# benchmark/kernel_tuning/results/dflash_hd256_chunks.csv (Qwen3.5 hd256, 16/4
# heads, B1-4, Q4/8/16, ctx100-3000, RTX 5090/RTX PRO 6000/GB10). On sm_120,
# at kvlen~700-1000, chunks 32/64 give 0.68-0.90x default (geomean over Q,B;
# worst row 1.10x); at ~500, chunk 32 gives 0.82-0.86x. Above ~1500 no chunk
# wins consistently; GB10 gains nothing past 256. hd128 keeps the native
# minimum: its R4 chunk-32 tier flipped a near-tie greedy token in Qwen3-8B
# DFlash parity on RTX 5090. Explicit min_chunk_tokens applies at all lengths.
FIXED_QUERY_MIN_CHUNK: dict[tuple[int, int, int], tuple[tuple[int, int], ...]] = {
    (12, 0, 256): ((16, 512), (32, 768), (64, 1280)),
    (12, 1, 256): ((32, 256),),
}


def check_dense_output(out: torch.Tensor | None) -> None:
    """FA2 and XQA write [tokens, heads, dim] rows densely, ignoring output strides.

    Both providers read query strides but take the output as a bare data pointer, so
    a strided output view would be written out of place. Host-only stride check.
    """
    if out is not None and not out.is_contiguous():
        raise ValueError("paged attention writes a dense output; pass a contiguous out")


def fixed_query_pdl(provider) -> bool:
    """PDL for the next FA2 call: off for the first call after each prepare().

    FA2 reads its device schedule (request/tile indices, chunk size, validity, query
    and KV CSR) before griddepcontrol.wait. The first call after prepare() is a full
    stream dependency, so prepare's writes are complete before that call starts.
    Every later call starts after that call started, so its pre-wait reads see the
    same completed schedule. The first call after construction also keeps PDL off.
    CTI_FIXED_QUERY_PDL=0 keeps PDL off on every call.
    """
    serialize = getattr(provider, "_serialize_next", True)
    provider._serialize_next = False
    if os.environ.get("CTI_FIXED_QUERY_PDL", "1") == "0":
        return False
    return not serialize


def fixed_query_min_chunk(device, *, head_dim) -> tuple[tuple[int, int], ...]:
    major, minor = torch.cuda.get_device_capability(device)
    return FIXED_QUERY_MIN_CHUNK.get((major, minor, head_dim), ((128, 0),))


def _aligned(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment


# XQA decode is qualified from 0.7.0.post1: 0.6.11.post3 returns wrong FP8-KV
# outputs for head_dim 256 with 16-token pages on sm_120 and sm_121
# (tests/gpu/test_xqa_decode.py, ~41% of elements outside tolerance).
XQA_VERSIONS = ("0.7.0.post1", "0.7.1")
# Head groups checked on GPU: powers of two and Qwen3.5-27B's 6 (24/4 and 12/2).
XQA_GROUPS = (1, 2, 4, 6, 8, 16, 32)


@functools.cache
def xqa_flashinfer_version() -> bool:
    """A qualified provider whose XQA decode passed the GPU suites (once per process)."""
    return qualified_flashinfer_version() and version("flashinfer-python") in XQA_VERSIONS


def xqa_decode_qualified(
    device,
    *,
    dtype,
    kv_dtype,
    page_size,
    head_dim,
    num_qo_heads,
    num_kv_heads,
    pos_encoding_mode,
    logits_soft_cap,
    window_left,
    rope_scale,
    rope_theta,
    custom_mask=None,
    packed_custom_mask=None,
) -> bool:
    """Common, deliberately bounded XQA domain of the qualified providers."""
    # Source references below are FlashInfer 0.6.11.post3 / 0.7.0.post1.
    # jit/xqa.py:43-74 / :66-106 selects FP16/BF16 Q, e4m3 KV, pages and
    # HEAD_GRP_SIZE at compilation. Keep the GPU-checked XQA_GROUPS (<=32, one
    # warp's row statistics) and the older release's head_dim<=256 domain.
    # xqa.py:319-320 / :402-403 admits SM120/121; this route uses NHD, Q=1,
    # matching output dtype, no sinks, RoPE, softcap or custom mask.
    if (
        dtype not in (torch.float16, torch.bfloat16)
        or kv_dtype != torch.float8_e4m3fn
        or page_size not in (16, 32, 64, 128)
        or head_dim not in range(16, 257, 16)
        or num_kv_heads < 1
        or num_qo_heads % num_kv_heads
        or num_qo_heads // num_kv_heads not in XQA_GROUPS
        or pos_encoding_mode != "NONE"
        or logits_soft_cap not in (None, 0.0)
        or window_left < -1
        or rope_scale is not None
        or rope_theta is not None
        or custom_mask is not None
        or packed_custom_mask is not None
    ):
        return False
    return xqa_flashinfer_version() and torch.cuda.get_device_capability(device) in (
        (12, 0),
        (12, 1),
    )


def xqa_decode_scratch_size(
    batch: int, num_kv_heads: int, subseq: int, head_dim: int, group: int
) -> int:
    """Scratch bytes for Q=1, independently of KV length; semaphores excluded."""
    if (
        min(batch, num_kv_heads, subseq) < 1
        or head_dim not in range(16, 257, 16)
        or group not in XQA_GROUPS
    ):
        raise ValueError("unsupported XQA scratch geometry")
    # Original derivation from data/csrc/xqa/ in 0.6.11.post3 / 0.7.0.post1:
    # mha.h:33-34 / :33-36 pads D to 64/128/256. mha.cu:87-89 / :87-89
    # rounds group rows to 16; mha_components.cuh:38-39 rounds row stats to
    # a full warp. SMemWarpRowMax (mha.cu:302 / :313) is 4*round_up(G,32).
    # mha.cu:2580-2588 / :2712-2721 reserves two stats arrays followed by
    # N output slices of 2*G*padded_D bytes (input/output are FP16/BF16).
    # utils.h:226-230,252-256 aligns each segment to sizeof(element), including
    # the entire output slice. Bound N by B*Hkv*S, then round to 256 bytes.
    # The existing qualified 0.7.1 also keeps these segments and the split cap:
    # https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/csrc/xqa/mha.cu
    # :2579-2588 (segments), :2623-2626 (reset), :2977-2982 (launch bound).
    count = batch * num_kv_heads * subseq
    padded_dim = max(64, 1 << (head_dim - 1).bit_length())
    stats = 4 * _aligned(group, 32)
    output = 2 * group * padded_dim
    return _aligned(_aligned(2 * count * stats, output) + count * output, 256)


class _FlashInferXQADecode:
    """Own XQA metadata; snapshot CUDA CSR indices during host planning."""

    def __init__(
        self,
        *,
        device,
        batch,
        width,
        num_kv_heads,
        head_dim,
        group,
        page_size,
        subseq,
        graph,
        dtype=torch.bfloat16,
    ):
        self.batch, self.num_kv_heads = batch, num_kv_heads
        self.page_size, self.subseq, self.graph = page_size, subseq, graph
        self.scratch_bytes = xqa_decode_scratch_size(batch, num_kv_heads, subseq, head_dim, group)
        # mha.cu:2575,2615-2618 / :2707,2759-2762: one uint32 per (B,Hkv).
        # atomicInc with limit splits-1 wraps the last arrival to zero. Owned
        # storage stays zero between serialized runs; shared scratch need not.
        self.semaphores = torch.zeros(batch * num_kv_heads, dtype=torch.uint32, device=device)
        self.page_table = torch.empty((batch, width), dtype=torch.int32, device=device)
        self.seq_lens = torch.empty((batch, 1), dtype=torch.int32, device=device)
        self._lengths_host = torch.empty(
            (batch, 1), dtype=torch.int32, device="cpu", pin_memory=True
        )
        # XQA indexes Q as dense [B, 1, H, D] rows (a fused-QKV slice
        # [2, 48, 128][:, :32] read query 1 at element 4096), while FA2 honours
        # query strides. Strided queries go through this owned dense buffer,
        # allocated here and never inside a capture.
        self._q_stage = torch.empty(
            (batch, num_kv_heads * group, head_dim), dtype=dtype, device=device
        )

    def plan(self, cu_seqlens_k, indices, last_page_len, seq_lens):
        if any(value.device.type != "cpu" for value in (cu_seqlens_k, last_page_len, seq_lens)):
            raise ValueError("XQA planning requires CPU CSR and sequence lengths")
        if (
            cu_seqlens_k.numel() != self.batch + 1
            or last_page_len.numel() != self.batch
            or seq_lens.numel() != self.batch
        ):
            raise ValueError("XQA metadata does not match the fixed batch")
        offsets = cu_seqlens_k.tolist()
        pages = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        width = max(1, max(pages))
        if width > self.page_table.shape[1]:
            if self.graph:
                raise ValueError("XQA pages exceed the graph indices buffer capacity")
            self.page_table = torch.empty(
                (self.batch, width), dtype=torch.int32, device=indices.device
            )
        # Host offsets select CUDA slices; all page-index movement stays on device.
        # No D2H or synchronization. Unused columns contain the valid index zero.
        self.page_table.zero_()
        for request, (start, count) in enumerate(zip(offsets[:-1], pages, strict=True)):
            self.page_table[request, :count].copy_(
                indices[start : start + count], non_blocking=True
            )
        # As with provider metadata, the caller completes the previous staging
        # copy before replanning. The pinned allocation lives with this operation.
        self._lengths_host.copy_(seq_lens.reshape(self.batch, 1))
        self.seq_lens.copy_(self._lengths_host, non_blocking=True)

    def run(self, q, k_cache, v_cache, out, scratch, k_scale, v_scale, sm_scale, window_left):
        from flashinfer.xqa import xqa

        k_scale = 1.0 if k_scale is None else k_scale
        v_scale = 1.0 if v_scale is None else v_scale
        sm_scale = q.shape[-1] ** -0.5 if sm_scale is None else sm_scale
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        dense_q = q
        if not q.is_contiguous():
            dense_q = self._q_stage[: q.shape[0]]
            dense_q.copy_(q)
        # FA2 scales the output by v_scale, so zero yields zeros. XQA scales the
        # logits by q_scale*kv_scale, so kv_scale=0 gives uniform weights times
        # zero; keep q_scale finite by not dividing by the zero scale.
        bmm2 = v_scale if v_scale != 0 else 1.0
        # decode.py:2795-2796 / :3922-3923, with output scale=1:
        # bmm1=sm_scale*k_scale, bmm2=v_scale; q_scale=bmm1/bmm2*sqrt(D).
        # xqa.py:278-279 / :361-363 derives max_seq_len from table width.
        # mha.cu:2950-2955 / :3123-3128 caps splits at sm_count/(B*Hkv),
        # further bounded by ceil(W*page_size/256), all host geometry. Pass
        # S*B*Hkv to select the table split covered by the scratch bound.
        xqa(
            q=dense_q.unsqueeze(1),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=self.page_table,
            seq_lens=self.seq_lens,
            output=out.unsqueeze(1),
            workspace_buffer=scratch,
            semaphores=self.semaphores,
            num_kv_heads=self.num_kv_heads,
            page_size=self.page_size,
            q_scale=sm_scale * k_scale / bmm2 * q.shape[-1] ** 0.5,
            kv_scale=v_scale,
            sliding_win_size=window_left + 1,
            kv_layout="NHD",
            sm_count=self.subseq * self.batch * self.num_kv_heads,
            # Provider default (PDL on sm_120/121, as vLLM runs it): the kernel waits
            # (griddepcontrol.wait) before reading any global input, mha.cu:1664-1670
            # in 0.7.0.post1; this route writes no metadata inside the graph.
            enable_pdl=None,
        )
        return out


def prefill_graph_split_workspace(total, batch, num_qo_heads, head_dim, layout):
    """Graph-prefill allocator bound for one fixed padded layout (bytes)."""
    count = layout.padded
    rows = num_qo_heads * count * layout.tile
    int_bytes = (
        12 * _aligned(count, 4)
        + 4 * _aligned(batch + layout.padding + 2, 4)
        + 16
        + 4 * _aligned(total + 1, 4)
        + _aligned(count, 16)
    )
    return _aligned(rows * (head_dim + 1) * 4, 256), _aligned(int_bytes, 256)


def prefill_graph_split_workspace_size(
    device, max_batch_size, max_query_tokens, num_qo_heads, num_kv_heads, head_dim
):
    """Cover every activating first-plan batch/row total, including empty requests."""
    table = prefill_graph_rows(device, num_qo_heads, num_kv_heads, head_dim)
    sizes = (0, 0)
    if not table:
        return sizes
    capacity = 2 * torch.cuda.get_device_properties(device).multi_processor_count // num_kv_heads
    for batch in {b for b, _, _ in table if b <= max_batch_size}:
        max_chunks = max(
            prefill_graph_config(c)["chunks"] for (b, _, _), c in table.items() if b == batch
        )
        totals = set()
        for (b, (qlo, qhi), _), value in table.items():
            config = prefill_graph_config(value)
            if b == batch and (config["chunks"] > 1 or "tile" in config):
                # A maximum request in [qlo, qhi] admits totals [qlo, B*qhi].
                totals.update(
                    (total, config.get("tile"))
                    for total in range(max(1, qlo), min(max_query_tokens, batch * qhi) + 1)
                )
        for total, tile in totals:
            layout = prefill_graph_split_layout(
                total, batch, num_qo_heads // num_kv_heads, head_dim, max_chunks, capacity, tile
            )
            if layout is not None:
                required = prefill_graph_split_workspace(
                    total, batch, num_qo_heads, head_dim, layout
                )
                sizes = tuple(max(a, b) for a, b in zip(sizes, required, strict=True))
    return sizes


def paged_attention_workspace_size(
    *,
    mode: str,
    backend: str,
    use_tensor_cores: bool,
    use_cuda_graph: bool,
    max_batch_size: int,
    max_query_tokens: int | None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[int, int]:
    """Closed-form capacities for any KV length/page size within the declared limits.

    Qualified against FlashInfer 0.6.11.post3, 0.7.0.post1 and 0.7.1; these bounds use scheduler capacities,
    not serialized offsets. Sizing does not require an exact version gate.
    """
    if mode not in ("prefill", "decode"):
        raise ValueError(f"unknown paged attention mode {mode!r}")
    if backend not in ("fa2", "fa3"):
        raise ValueError(f"paged attention workspace is declared for fa2 and fa3, not {backend!r}")
    batch = max_batch_size
    # Decode plans take one query token per request.
    tokens = max_query_tokens if mode == "prefill" else batch
    if tokens is None:
        raise ValueError("prefill workspace sizes need max_query_tokens")
    if min(batch, tokens, num_qo_heads, num_kv_heads, head_dim) < 1:
        raise ValueError("paged attention dimensions must be positive")
    if num_qo_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    properties = torch.cuda.get_device_properties(device)
    if properties.major < 8:
        raise ValueError("paged attention workspace sizing requires an SM80 or newer CUDA device")
    sms = properties.multi_processor_count
    group = num_qo_heads // num_kv_heads
    if mode == "decode" and not use_tensor_cores:
        # scheduler.cuh DecodePlan: >=128 threads/CTA; eager empty-KV requests
        # add at most one CTA each. A graph filling the occupancy grid cannot split.
        grid = sms * (properties.max_threads_per_multi_processor // 128)
        share = grid // num_kv_heads
        split = share + min(batch, share)
        if use_cuda_graph:
            split = share if batch * num_kv_heads < grid else 0
        rows = num_qo_heads * split
        int_bytes = 13 * max(batch, split) + 64  # three int32 arrays + mask + alignment
    elif backend == "fa3":
        # scheduler.cuh PrefillSM90Plan: no partial outputs; seven int32 arrays
        # schedule heads separately up to 4096 tiles, plus one work offset per SM.
        tile = 192 if head_dim == 64 else 128
        tiles = (tokens + tile - 1) // tile + (batch if use_cuda_graph else min(batch, tokens)) - 1
        works = max(num_qo_heads * min(tiles, 4096), tiles)
        rows = 0
        int_bytes = 28 * _aligned(works, 4) + 4 * _aligned(sms + 1, 4)
    else:
        # utils.cuh FA2DetermineCtaTileQ (0.6.11.post3 :384, 0.7.0.post1 :408): with
        # equal Q/KV head dimensions <= 256 its 0.7 shared-memory probe always admits
        # the 16-row tile, so KV dtype does not change the choice.
        def cta_tile(packed):
            return 16 if packed <= 16 else (128 if packed > 64 and head_dim < 256 else 64)

        split = 2 * sms // num_kv_heads
        if use_cuda_graph:
            # PrefillSplitQOKVIndptr graph branch (scheduler.cuh:534 / :603) picks the
            # tile from max_seq_len = total_rows - batch + 1 (unsigned: fewer rows than
            # requests wrap to the largest tile) and pads CTAs to
            # max(split, ceil(rows * group / tile) + batch - 1). Later plans may carry
            # fewer rows, hence any tile up to the declared one. Plans may also admit
            # fewer requests than max_batch_size; fewer requests mean a longer longest
            # request and a larger tile, so take the maximum over every admitted count.
            # CTAs grow with the row total, so per count two totals bound every plan:
            # the declared one, and the largest below the count (zero-length requests,
            # e.g. qo_indptr [0, 1, 1, ..., 1]), whose longest request wraps.
            rows = count = 0
            for requests in range(1, batch + 1):
                plans = [(tokens, tokens - requests + 1 if tokens >= requests else 1 << 62)]
                if requests > 1:
                    plans.append((min(requests - 1, tokens), 1 << 62))
                for total, longest in plans:
                    for t in (16, 64, 128):
                        if t <= cta_tile(longest * group):
                            n = max(split, (total * group + t - 1) // t + requests - 1)
                            rows = max(rows, num_qo_heads * n * t)
                            count = max(count, n)
            # Three CTA arrays, output indptr + chunk scalar, total_num_rows,
            # merge indptr and validity mask (PrefillPlan:737-780 / :810-868).
            int_bytes = (
                12 * _aligned(count, 4)
                + 4 * _aligned(batch + 2, 4)
                + 16
                + 4 * _aligned(tokens + 1, 4)
                + _aligned(count, 16)
            )
        else:
            # scheduler.cuh PrefillPlan: <=2 CTAs/SM per KV head; the eager tile comes
            # from the average packed query length, bounded by the declared total.
            tile = cta_tile((tokens if mode == "prefill" else 1) * group)
            rows = num_qo_heads * split * tile
            # Short tiles imply <17 packed rows/request (or <65 for tile=64).
            # Their count is <=ceil(17B/16); long tiles add <one tile/nonempty request.
            tiles = max(
                (tokens * group + tile - 1) // tile,
                min((tokens * group + 15) // 16, (17 * batch + 15) // 16),
            )
            tiles += min(batch, tokens) - 1
            merge = min(tokens, tile * split // group)
            # scheduler.cuh / allocator.h: three tile arrays + validity mask, merge
            # offsets, output offsets + scalars. Round each aggregate up to 16 bytes.
            split_ints = 12 * _aligned(split, 4) + _aligned(split, 16) + 4 * _aligned(merge + 1, 4)
            int_bytes = max(split_ints, 12 * _aligned(tiles, 4)) + 4 * _aligned(batch + 2, 4)
    # FP32-sized partial-V reservation (FA2 writes output dtype) and FP32 log-sum-exp.
    float_bytes = _aligned(rows * head_dim * 4, 16) + rows * 4
    return _aligned(float_bytes, 256), _aligned(int_bytes, 256)
