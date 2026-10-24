# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/attention/fi.py`, `python/minisgl/attention/fa.py`, and
# `python/minisgl/attention/trtllm.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import inspect
import os
import threading
from contextlib import contextmanager
from typing import Literal, NamedTuple

import torch

from .._runtime.workspace import GraphWorkspace
from .._storage import reject_grad
from . import _decode_fp8, _prefill_cute
from ._flashinfer import FlashInferFixedQueryAttention as FlashInferFixedQueryAttention
from ._flashinfer import (
    _FlashInferXQADecode,
    check_dense_output,
    custom_mask_metadata,
    decode_plan_capacity,
    decode_split_capacity,
    paged_attention_workspace_size,
    prefill_graph_split_workspace,
    prefill_graph_split_workspace_size,
    qualified_flashinfer_version,
    staged_output_flashinfer_version,
    xqa_decode_qualified,
    xqa_decode_scratch_size,
)
from ._prologue import Prologue as Prologue
from ._split_tables import (
    decode_chunk,
    decode_uses_cuda_cores,
    device_key,
    fp8_decode_cute,
    lookup,
    prefill_graph_chunk,
    prefill_graph_chunks,
    prefill_graph_cute,
    prefill_graph_cute_rows,
    prefill_graph_max_chunks,
    prefill_graph_policy,
    prefill_graph_split_layout,
    prefill_query_split,
    prefill_split_batch_bound,
    xqa_decode_subseq,
)
from .draft import CacheTableDraftAttention as CacheTableDraftAttention
from .draft import DraftAttention as DraftAttention

_prefill_config_lock = threading.RLock()


@contextmanager
def _cute_prefill_config(config):
    """Bridge env-only CuTe releases; serialize and restore overrides on failure too."""
    parameters = inspect.signature(_prefill_cute.CutePagedPrefill).parameters
    if all(name in parameters for name in config) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        with _prefill_config_lock:
            yield config
        return
    allowed = {"block_m", "block_n", "stages", "splits", "min_chunk"}
    if config.keys() - allowed:
        raise ValueError("unsupported CuTe prefill configuration keys")
    overrides = {"CTI_PREFILL_" + key.upper(): str(value) for key, value in config.items()}
    with _prefill_config_lock:
        previous = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            yield {}
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _cute_prefill_workspace_size(device, max_batch, max_total, hq, hk, dim):
    required = 0
    for (batch, (lo, hi)), config in prefill_graph_cute_rows(device, hq, hk, dim).items():
        if batch > max_batch:
            continue
        # Default split counts vary with query tiles: evaluate every admitted total.
        with _cute_prefill_config(config) as kwargs:
            for total in range(max(1, lo), min(max_total, batch * hi) + 1):
                required = max(
                    required,
                    _prefill_cute.workspace_size(
                        device,
                        batch_size=batch,
                        max_query_per_request=total,
                        num_qo_heads=hq,
                        num_kv_heads=hk,
                        head_dim=dim,
                        **kwargs,
                    ),
                )
    return required


def check_flashinfer_available() -> None:
    from flashinfer import (  # noqa: F401
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )
    from flashinfer.utils import PosEncodingMode, determine_attention_backend  # noqa: F401


def check_flash_attention_available() -> None:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401


def check_trtllm_available() -> None:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache  # noqa: F401


def select_flashinfer_prefill_implementation(device: torch.device, dtype: torch.dtype) -> str:
    """Ask the provider for its non-quantized prefill implementation at setup time."""
    from flashinfer.utils import PosEncodingMode, determine_attention_backend

    return determine_attention_backend(
        device,
        PosEncodingMode.NONE.value,
        False,  # use_fp16_qk_reductions
        False,  # use_custom_mask
        dtype,
        dtype,
    )


def fp8_decode_cute_qualified(
    device,
    *,
    dtype,
    kv_dtype,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    window_left,
    logits_soft_cap,
    rope_scale,
    rope_theta,
    pos_encoding_mode,
):
    if (
        window_left != -1
        or logits_soft_cap not in (None, 0.0)
        or rope_scale is not None
        or rope_theta is not None
        or pos_encoding_mode != "NONE"
    ):
        return False
    try:
        _decode_fp8.check_support(
            device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_dtype=dtype,
            kv_dtype=kv_dtype,
        )
    except NotImplementedError:
        return False
    return True


class FlashInferPagedAttention:
    """One native paged-attention operation, with caller-owned planning storage.

    Ordinary planning consumes CPU int32 cumulative lengths, last-page lengths
    and sequence lengths, plus CUDA int32 indices. It stages provider metadata;
    the caller must finish the previous staging copy before planning again and
    retain the input/storage lifetimes. Workspaces are borrowed, never replaced
    here. The caller owns capture/replay and planning cadence.

    Native graph prefill borrows query/KV CSR buffers and optional packed-mask storage.
    The first host plan declares the maximum query rows; subsequent plans keep
    the batch fixed and may use fewer rows. Replan outside capture before replay,
    after previous work and staging copies complete. Capture only run(), keeping
    input/output addresses and geometry/options fixed. For changing device KV
    lengths without host replanning, use ``FlashInferFixedQueryAttention``.

    The caller sizes both workspaces with ``workspace_size()`` and must
    zero-initialize each newly allocated float workspace before its first native
    use. Keep both workspaces alive with stable storage through
    all asynchronous planning, execution and graph replays. Reuse does not require
    per-plan, per-run or per-replay clearing; callers must serialize uses of shared
    workspaces. This wrapper neither initializes nor clears borrowed storage.

    Execution consumes query [tokens, qo_heads, head_dim] and NHD paged K/V caches
    [pages, page_size, kv_heads, head_dim] on the current CUDA stream. It has no
    backward, does not write query/K/V, and may mutate provider workspace. With
    ``out``, execution writes to that caller-owned non-gradient buffer;
    otherwise output is allocated. No host synchronization or host/device copies
    are introduced by the execution wrapper.

    ``dtype`` is the query/default output dtype; ``kv_dtype`` defaults to it.
    Prefill may override ``o_dtype``. Plans forward window, mask, softcap, scale
    and RoPE options; runs accept K/V scales without casting query/KV tensors.
    ``dispatch="device"`` tunes FA2 decode and short fresh prefill for equal
    FP16/BF16 Q/KV and default options (custom sm_scale is allowed). Qualified
    SM120/121 FP8 e4m3 KV decode with FP16/BF16 Q can use table-selected CuTe
    or XQA (including sliding windows for XQA). Table-selected causal graph prefill
    uses CuTe first, then padded requests with optional query-tile steering and KV
    chunks. CuTe borrows all caller CSR buffers; padded prefill owns its padded
    CSR buffers and shares the caller's indices buffer. Other cases keep
    native choices; ``"provider"`` always keeps native choices. Device decode and
    graph prefill construct their provider at the first plan, before capture.
    XQA owns semaphores, a page table and staged lengths; plan snapshots indices,
    so write them before plan.
    Graph routes and metadata addresses stay fixed after the first plan. Eager
    planning may grow owned metadata; borrowed workspaces are never resized.

    Qualified native decode with BF16 Q and FP8 KV owns an FP16 output stage
    (B * Hq * D * 2 bytes, eager capacity grows at plan). One CuTe launch scales
    in FP32 and rounds to BF16 for |v_scale| <= 1 (None means 1), with absolute
    floor ~2^-24 * |v_scale|. Larger scales and inf/NaN use native BF16 output
    with the same plan. FlashInfer has no FP32 output. This storage is separate
    from workspace_size(); capture bakes the host scale and output route.
    """

    @staticmethod
    def workspace_size(
        *,
        mode: Literal["prefill", "decode"],
        backend: str,
        use_tensor_cores: bool = False,
        use_cuda_graph: bool = False,
        max_batch_size: int,
        max_query_tokens: int | None = None,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dispatch: Literal["device", "provider"] = "device",
    ) -> tuple[int, int]:
        """Float and integer workspace bytes that every plan of this operation fits.

        Default sizes cover every route device dispatch may pick;
        ``dispatch="provider"`` sizes the provider route alone.
        These borrowed capacities exclude the owned BF16-Q/FP8-KV native decode
        output stage: max_batch_size * num_qo_heads * head_dim * 2 bytes.
        FP16 staging applies for |v_scale| <= 1 (None means 1), with absolute
        floor ~2^-24 * |v_scale|; otherwise native BF16 output uses the same plan.
        FlashInfer has no FP32 output.

        Pass the construction options, with ``backend`` ``"fa2"`` or ``"fa3"``, the
        per-rank head counts and bounds on every plan: at most ``max_batch_size``
        requests and, for prefill, ``max_query_tokens`` query tokens. For a decode
        graph, ``max_batch_size`` is its fixed batch size and the sizes hold for that
        batch alone; a caller capturing several batches takes the maximum over them.
        Prefill graph sizes hold for every fixed batch up to ``max_batch_size``.
        Device FA2 graph prefill also covers every activating first-plan row total
        T: with natural tile t, tiles=ceil(T*group/t)+B-1 and table maximum C,
        P=(C-1)*tiles+1 empty requests reserve n=max(2*SMs//kv_heads, tiles+P)
        CTAs. Partial V/LSE cost qo_heads*n*t*(head_dim+1)*4 bytes, plus graph
        integer arrays for B+P requests, n CTAs and T rows (aligned upward).
        Tile steering instead uses P=T+1-floor(t/group)-B and the same CTA formula.
        At the largest shipped table rows the padded routes grow the float
        workspace to about 0.2-0.3 GB (RTX PRO 6000, 32/8 heads, D128, B4,
        T=1024, C=4: n=141 CTAs, about 0.30 GB of partial V/LSE).
        Table-selected CuTe scratch covers every activating batch/total using
        the package's workspace_size with max_query_per_request=T.
        Decode plans one query token per request, so decode sizes depend on
        ``max_batch_size`` alone and ignore ``max_query_tokens``. KV lengths, page
        size, dtype and causality do not change the sizes.

        The sizes hold for workspaces that start 16-byte aligned (256-byte aligned
        for CuTe), as CUDA allocations do. Operations that share a workspace pair need the
        elementwise maximum of their sizes. Sizing requires an SM80 or newer device
        and supports eager/graph prefill and decode. It does not require an
        installed provider. Query this at setup, outside capture: it reads the
        device's properties.
        """
        if dispatch not in ("device", "provider"):
            raise ValueError("dispatch must be 'device' or 'provider'")
        options = {
            "mode": mode,
            "backend": backend,
            "use_tensor_cores": use_tensor_cores,
            "use_cuda_graph": use_cuda_graph,
            "max_batch_size": max_batch_size,
            "max_query_tokens": max_query_tokens,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "device": device,
        }
        sizes = paged_attention_workspace_size(**options)
        if dispatch == "provider":
            return sizes
        if mode == "prefill" and use_cuda_graph:
            scratch = _cute_prefill_workspace_size(
                device, max_batch_size, max_query_tokens, num_qo_heads, num_kv_heads, head_dim
            )
            sizes = (max(sizes[0], scratch), sizes[1])
            if backend == "fa2":
                routed = prefill_graph_split_workspace_size(
                    device, max_batch_size, max_query_tokens, num_qo_heads, num_kv_heads, head_dim
                )
                sizes = tuple(max(a, b) for a, b in zip(sizes, routed, strict=True))
            return sizes
        if mode == "decode":
            batches = (max_batch_size,) if use_cuda_graph else range(1, max_batch_size + 1)
            for batch in batches:
                config = fp8_decode_cute(device, num_qo_heads, num_kv_heads, head_dim, batch)
                if config is not None:
                    scratch = _decode_fp8.workspace_size(
                        device,
                        batch_size=batch,
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        **config,
                    )
                    sizes = (max(sizes[0], scratch), sizes[1])
                subseq = xqa_decode_subseq(device, num_qo_heads, num_kv_heads, head_dim, batch)
                if subseq is not None:
                    scratch = xqa_decode_scratch_size(
                        batch, num_kv_heads, subseq, head_dim, num_qo_heads // num_kv_heads
                    )
                    sizes = (max(sizes[0], scratch), sizes[1])
        if mode == "decode":
            # Eager and graph plans share the native kernel choice; CUDA-core
            # batches are sized as graph plans in both modes (_workspace_fits).
            batches = (max_batch_size,) if use_cuda_graph else range(1, max_batch_size + 1)
            for batch in batches:
                if use_tensor_cores and decode_uses_cuda_cores(
                    device, num_qo_heads, num_kv_heads, head_dim, batch
                ):
                    routed = paged_attention_workspace_size(
                        **options
                        | {
                            "use_tensor_cores": False,
                            "use_cuda_graph": True,
                            "max_batch_size": batch,
                        }
                    )
                    sizes = tuple(max(a, b) for a, b in zip(sizes, routed, strict=True))
            return sizes
        if mode == "prefill" and head_dim == 128:
            bound = prefill_split_batch_bound(device_key(device), max_batch_size, max_query_tokens)
            if bound is None:
                return sizes
            options["max_batch_size"] = max(max_batch_size, bound)
        else:
            return sizes
        routed = paged_attention_workspace_size(**options)
        return tuple(max(a, b) for a, b in zip(sizes, routed, strict=True))

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        *,
        mode: Literal["prefill", "decode"],
        backend: str,
        use_tensor_cores: bool = False,
        use_cuda_graph: bool = False,
        qo_indptr_buffer: torch.Tensor | None = None,
        custom_mask_buffer: torch.Tensor | None = None,
        mask_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indices_buffer: torch.Tensor | None = None,
        paged_kv_last_page_len_buffer: torch.Tensor | None = None,
        dispatch: Literal["device", "provider"] = "device",
    ) -> None:
        if dispatch not in ("device", "provider"):
            raise ValueError("dispatch must be 'device' or 'provider'")
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self._is_decode = mode == "decode"
        self._dispatch = dispatch
        self._requested_tensor_cores = use_tensor_cores
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = use_cuda_graph
        self._capture_plan_info: tuple[int, ...] | None = None
        self._native_decoders = {}
        self._backend = backend
        self._workspaces = (float_workspace_buffer, int_workspace_buffer)
        self._planned = False
        self._gather_host = None
        self._gather_device = None
        self._split_indices = None
        self._prefill_metadata = None
        self._prefill_route = "native"
        self._prefill_graph_layout = None
        self._prefill_graph_geometry = None
        self._prefill_constructor = BatchPrefillWithPagedKVCacheWrapper
        self._decode_constructor = BatchDecodeWithPagedKVCacheWrapper
        self._decode_options = None
        self._decode_route = None
        self._fp8_output = None
        self._use_fp8_output = False
        self._xqa = None
        self._xqa_geometry = None
        self._cute = None
        self._cute_geometry = None
        self._cute_metadata = None
        if self._is_decode:
            self._decode_options = {
                "kv_layout": "NHD",
                "use_tensor_cores": use_tensor_cores,
                "backend": backend,
            }
            if use_cuda_graph:
                self._decode_options.update(
                    {
                        "use_cuda_graph": True,
                        "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                        "paged_kv_indices_buffer": paged_kv_indices_buffer,
                        "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
                    }
                )
            self._wrapper = (
                None
                if dispatch == "device"
                else BatchDecodeWithPagedKVCacheWrapper(
                    float_workspace_buffer, **self._decode_options
                )
            )
        elif mode == "prefill":
            graph_options = {}
            if use_cuda_graph:
                required = {
                    "qo_indptr_buffer": qo_indptr_buffer,
                    "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
                    "paged_kv_indices_buffer": paged_kv_indices_buffer,
                    "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
                }
                missing = [name for name, value in required.items() if value is None]
                if missing:
                    raise ValueError("graph prefill requires " + ", ".join(missing))
                graph_options = {
                    name.removesuffix("buffer") + "buf": value for name, value in required.items()
                }
                graph_options.update(
                    use_cuda_graph=True,
                    custom_mask_buf=custom_mask_buffer,
                    mask_indptr_buf=mask_indptr_buffer,
                )
            self._prefill_options = dict(kv_layout="NHD", backend=backend, **graph_options)
            self._wrapper = (
                None
                if use_cuda_graph and dispatch == "device"
                else BatchPrefillWithPagedKVCacheWrapper(
                    float_workspace_buffer, **self._prefill_options
                )
            )
        else:
            raise ValueError("mode must be prefill or decode")
        if self._wrapper is not None:
            self._wrapper.reset_workspace_buffer(*self._workspaces)

    @property
    def decode_route(self) -> Literal["fa2", "cuda_cores", "xqa", "cute"]:
        """Selected decode implementation; native choice until the first plan."""
        if self._decode_route is not None and self._decode_route[0] in ("xqa", "cute"):
            return self._decode_route[0]
        return "fa2" if self._use_tensor_cores else "cuda_cores"

    @property
    def prefill_route(self) -> Literal["native", "padded_split", "cute"]:
        """Selected at the first graph plan; native until a route is selected."""
        return self._prefill_route

    def _plan_cute_decode(self, kv, indices, last, *, geometry, config, **options):
        device = indices.device
        batch = options["batch_size"]
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CuTe decode plan must run outside capture")
        for value, count in ((kv, batch + 1), (last, batch)):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.int32
                or value.ndim != 1
                or value.numel() != count
            ):
                raise ValueError("CuTe decode requires CPU CSR int32 vectors of the fixed batch")
        if indices.dtype != torch.int32 or indices.ndim != 1 or not indices.is_contiguous():
            raise ValueError("CuTe decode requires contiguous device int32 indices")
        required = _decode_fp8.workspace_size(
            device,
            **{
                name: options[name]
                for name in ("batch_size", "num_qo_heads", "num_kv_heads", "head_dim")
            },
            **config,
        )
        scratch = self._workspaces[0]
        if scratch.dtype != torch.uint8 or scratch.numel() < required:
            raise ValueError(f"CuTe needs at least {required} borrowed float workspace bytes")
        if self._use_cuda_graph:
            buffers = tuple(
                self._decode_options[name]
                for name in (
                    "paged_kv_indptr_buffer",
                    "paged_kv_indices_buffer",
                    "paged_kv_last_page_len_buffer",
                )
            )
            for value, count in zip(buffers, (batch + 1, None, batch), strict=True):
                if (
                    value is None
                    or value.device != device
                    or value.dtype != torch.int32
                    or value.ndim != 1
                    or not value.is_contiguous()
                    or (count is not None and value.numel() != count)
                ):
                    raise ValueError(
                        "graph CuTe decode requires device int32 CSR buffers of the fixed batch"
                    )
            if buffers[1].numel() < indices.numel():
                raise ValueError("graph CuTe indices exceed the caller's buffer capacity")
            self._cute_metadata = (buffers[0], buffers[2])
            self._cute_indices = buffers[1]
        else:
            if self._cute_metadata is None or self._cute_metadata[1].numel() < batch:
                self._cute_metadata = (
                    torch.empty(batch + 1, dtype=torch.int32, device=device),
                    torch.empty(batch, dtype=torch.int32, device=device),
                )
            self._cute_indices = indices
        if self._cute_geometry != geometry:
            self._cute = _decode_fp8.FP8PagedDecode(scratch, device=device, **options, **config)
            self._cute_q_stage = torch.empty(
                (batch, options["num_qo_heads"], options["head_dim"]),
                dtype=options["q_dtype"],
                device=device,
            )
            self._cute_geometry = geometry
        self._cute_metadata[0][: batch + 1].copy_(kv, non_blocking=True)
        self._cute_metadata[1][:batch].copy_(last, non_blocking=True)
        if self._use_cuda_graph:
            self._cute_indices[: indices.numel()].copy_(indices, non_blocking=True)
        self._cute_csr = (
            self._cute_metadata[0][: batch + 1],
            self._cute_indices,
            self._cute_metadata[1][:batch],
        )

    def _plan_cute_prefill(self, qo, kv, indices, last, *, heads, page_size, dtype, options, tuned):
        """Choose CuTe once and stage into the caller's fixed graph metadata buffers."""
        active = self._prefill_route == "cute"
        if not active and self._wrapper is not None:
            return False
        host = all(value.device.type == "cpu" for value in (qo, kv, last))
        geometry = (*heads, page_size, dtype, *options.values())
        device = self._workspaces[0].device
        if active:
            if not host or not tuned or geometry != self._prefill_graph_geometry:
                raise ValueError(
                    "CuTe graph prefill requires host metadata and first-plan geometry/options"
                )
            batch, total = self._prefill_graph_limits
        else:
            if not tuned or not host or heads[2] not in (64, 128, 256) or qo.numel() < 2:
                return False
            batch, total = qo.numel() - 1, int(qo[-1])
            config = prefill_graph_cute(device, *heads, batch, int(qo.diff().max()))
            if config is None:
                return False
            try:
                _prefill_cute.check_support(
                    device,
                    num_qo_heads=heads[0],
                    num_kv_heads=heads[1],
                    head_dim=heads[2],
                    page_size=page_size,
                    dtype=dtype,
                )
            except NotImplementedError:
                return False
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CuTe prefill plan must run outside capture")
        if qo.numel() != batch + 1 or kv.numel() != batch + 1 or last.numel() != batch:
            raise ValueError("graph prefill plans must keep the first plan's batch")
        if int(qo[-1]) > total:
            raise ValueError("graph prefill rows cannot exceed the first plan")
        if any(value.dtype != torch.int32 or value.ndim != 1 for value in (qo, kv, last)):
            raise ValueError("CuTe prefill requires CPU CSR int32 vectors")
        if (
            indices.device != device
            or indices.dtype != torch.int32
            or indices.ndim != 1
            or not indices.is_contiguous()
        ):
            raise ValueError("CuTe prefill requires contiguous device int32 indices")
        buffers = tuple(
            self._prefill_options[name]
            for name in (
                "qo_indptr_buf",
                "paged_kv_indptr_buf",
                "paged_kv_indices_buf",
                "paged_kv_last_page_len_buf",
            )
        )
        for value, count in zip(buffers, (batch + 1, batch + 1, None, batch), strict=True):
            if (
                value.device != device
                or value.dtype != torch.int32
                or value.ndim != 1
                or not value.is_contiguous()
                or (count is not None and value.numel() != count)
            ):
                raise ValueError(
                    "graph CuTe prefill requires device int32 CSR buffers of the batch"
                )
        if buffers[2].numel() < indices.numel():
            raise ValueError("graph CuTe indices exceed the caller's buffer capacity")
        if not active:
            bounds = {
                "batch_size": batch,
                "max_query_per_request": total,
                "num_qo_heads": heads[0],
                "num_kv_heads": heads[1],
                "head_dim": heads[2],
            }
            with _cute_prefill_config(config) as kwargs:
                required = _prefill_cute.workspace_size(device, **bounds, **kwargs)
                scratch = self._workspaces[0]
                if scratch.dtype != torch.uint8 or scratch.numel() < required:
                    raise ValueError(
                        f"CuTe needs at least {required} borrowed float workspace bytes"
                    )
                self._cute_prefill = _prefill_cute.CutePagedPrefill(
                    scratch,
                    device=device,
                    **bounds,
                    page_size=page_size,
                    dtype=dtype,
                    sm_scale=options["sm_scale"],
                    **kwargs,
                )
            self._cute_q_stage = torch.empty(
                (total, heads[0], heads[2]), dtype=dtype, device=device
            )
            self._prefill_graph_geometry = geometry
            self._prefill_graph_limits = (batch, total)
            self._prefill_route = "cute"
            self._cute_prefill_csr = buffers
        # Retain host sources until the next stream-ordered plan has staged new values.
        self._prefill_metadata = (qo, kv, last)
        for target, source in zip(buffers, (qo, kv, indices, last), strict=True):
            target[: source.numel()].copy_(source, non_blocking=True)
        return True

    def _plan_padded_prefill(self, qo, kv, last, seq, *, heads, page_size, dtype, options, tuned):
        """Select once, then stage empty trailing requests without replacing device storage."""
        # FlashInfer 0.7.0.post1 prefill.py:2555-2574,2815 freezes the first row
        # ceiling and batch; the scheduler therefore keeps this tile/grid on replan.
        host = all(value.device.type == "cpu" for value in (qo, kv, last, seq))
        geometry = (*heads, page_size, dtype, *options.values())
        hq, hk, dim = heads
        if self._wrapper is None:
            if tuned and host and self._backend == "fa2" and qualified_flashinfer_version():
                batch, total = qo.numel() - 1, int(qo[-1])
                device = self._workspaces[0].device
                policy = prefill_graph_policy(
                    device, *heads, batch, int(qo.diff().max()), int(kv.diff().max())
                )
                if policy["chunks"] > 1 or "tile" in policy:
                    capacity = (
                        2 * torch.cuda.get_device_properties(device).multi_processor_count // hk
                    )
                    layout = prefill_graph_split_layout(
                        total,
                        batch,
                        hq // hk,
                        dim,
                        prefill_graph_max_chunks(device, *heads, batch),
                        capacity,
                        policy.get("tile"),
                    )
                    if layout is not None:
                        required = prefill_graph_split_workspace(total, batch, hq, dim, layout)
                        if all(
                            w.numel() * w.element_size() >= n
                            for w, n in zip(self._workspaces, required, strict=True)
                        ):
                            self._prefill_graph_layout = layout
                            self._prefill_graph_geometry = geometry
                            self._prefill_graph_limits = (batch, total)
                            self._prefill_route = "padded_split"
                            for name, count in (
                                ("qo_indptr_buf", batch + layout.padding + 1),
                                ("paged_kv_indptr_buf", batch + layout.padding + 1),
                                ("paged_kv_last_page_len_buf", batch + layout.padding),
                            ):
                                self._prefill_options[name] = torch.empty(
                                    count, dtype=torch.int32, device=device
                                )
            self._wrapper = self._prefill_constructor(self._workspaces[0], **self._prefill_options)
            self._wrapper.reset_workspace_buffer(*self._workspaces)
        split_options = {}
        if self._prefill_route == "padded_split":
            if not host or not tuned or geometry != self._prefill_graph_geometry:
                raise ValueError(
                    "padded graph prefill requires host metadata and first-plan geometry/options"
                )
            batch, total = self._prefill_graph_limits
            if (
                qo.numel() != batch + 1
                or kv.numel() != batch + 1
                or last.numel() != batch
                or seq.numel() != batch
            ):
                raise ValueError("graph prefill plans must keep the first plan's batch")
            if int(qo[-1]) > total:
                raise ValueError("graph prefill rows cannot exceed the first plan")
            queries, pages = qo.diff().tolist(), kv.diff().tolist()
            chunks = prefill_graph_chunks(
                self._workspaces[0].device, *heads, batch, max(queries), max(pages)
            )
            layout = self._prefill_graph_layout
            chunk = prefill_graph_chunk(queries, pages, hq // hk, chunks, layout)
            if chunk is not None:
                split_options["fixed_split_size"] = chunk
            self._prefill_metadata = tuple(
                torch.cat((value, value.new_full((layout.padding,), tail)))
                for value, tail in ((qo, int(qo[-1])), (kv, int(kv[-1])), (last, 0), (seq, 0))
            )
            qo, kv, last, seq = self._prefill_metadata
        return qo, kv, last, seq, split_options

    def _native_decode_tensor_cores(
        self, *, tuned, batch, num_qo_heads, num_kv_heads, head_dim
    ) -> bool:
        """Native decode kernel for a batch, shared by eager and graph plans.

        A graph replays one kernel for every KV length of its fixed batch, so an
        eager plan of the same batch takes the same kernel. CUDA-core rows are
        sized as graph plans in both modes, so the workspace check agrees too.
        """
        if (
            self._requested_tensor_cores
            and tuned
            and decode_uses_cuda_cores(
                self._workspaces[0].device, num_qo_heads, num_kv_heads, head_dim, batch
            )
            and self._workspace_fits(
                batch=batch,
                total=None,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                use_cuda_graph=True,
            )
        ):
            return False
        return self._requested_tensor_cores

    def _workspace_fits(
        self, *, batch, total, num_qo_heads, num_kv_heads, head_dim, use_cuda_graph=None
    ):
        required = paged_attention_workspace_size(
            mode="decode" if self._is_decode else "prefill",
            backend=self._backend,
            use_tensor_cores=False,
            use_cuda_graph=self._use_cuda_graph if use_cuda_graph is None else use_cuda_graph,
            max_batch_size=batch,
            max_query_tokens=total,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=self._workspaces[0].device,
        )
        return all(
            workspace.numel() * workspace.element_size() >= size
            for workspace, size in zip(self._workspaces, required, strict=True)
        )

    def _gather_prefill_indices(self, indices, positions):
        count = len(positions)
        if self._gather_host is None or self._gather_host.numel() < count:
            self._gather_host = torch.empty(count, dtype=torch.int32, device="cpu", pin_memory=True)
            self._gather_device = torch.empty(count, dtype=torch.int32, device=indices.device)
            self._split_indices = torch.empty_like(self._gather_device)
        self._gather_host[:count].copy_(torch.tensor(positions, dtype=torch.int32, device="cpu"))
        self._gather_device[:count].copy_(self._gather_host[:count], non_blocking=True)
        torch.index_select(indices, 0, self._gather_device[:count], out=self._split_indices[:count])
        return self._split_indices[:count]

    def plan(
        self,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str,
        seq_lens: torch.Tensor,
        dtype: torch.dtype,
        kv_dtype: torch.dtype | None = None,
        o_dtype: torch.dtype | None = None,
        sm_scale: float | None = None,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        custom_mask: torch.Tensor | None = None,
        packed_custom_mask: torch.Tensor | None = None,
        causal: bool = True,
        disable_split_kv: bool = False,
    ) -> None:
        """Plan public metadata; ``causal`` and ``disable_split_kv`` affect prefill only.

        A plan needs at least one request; an empty batch raises ValueError.
        """
        if cu_seqlens_k.numel() < 2:
            raise ValueError("paged attention plan requires at least one request")
        kv_dtype = dtype if kv_dtype is None else kv_dtype
        options = {
            "sm_scale": sm_scale,
            "window_left": window_left,
            "logits_soft_cap": logits_soft_cap,
            "rope_scale": rope_scale,
            "rope_theta": rope_theta,
        }
        tuned = (
            self._dispatch == "device"
            and kv_dtype == dtype
            and dtype in (torch.float16, torch.bfloat16)
            and o_dtype in (None, dtype)
            and window_left == -1
            and logits_soft_cap in (None, 0.0)
            and custom_mask is None
            and packed_custom_mask is None
            and rope_scale is None
            and rope_theta is None
            and pos_encoding_mode == "NONE"
            and not disable_split_kv
        )
        if self._is_decode:
            if custom_mask is not None or packed_custom_mask is not None:
                raise ValueError("custom masks are supported for prefill only")
            if o_dtype is not None:
                raise ValueError("o_dtype is supported for prefill only")
            # Graph decode fixes its route at the first plan; captured replays keep
            # that native wrapper, so a later plan may not switch it.
            batch = cu_seqlens_k.numel() - 1
            if self._use_cuda_graph:
                last_buffer = self._decode_options["paged_kv_last_page_len_buffer"]
                if last_buffer is None:
                    raise ValueError("graph decode requires paged_kv_last_page_len_buffer")
                batch = last_buffer.numel()
            use_tensor_cores = self._native_decode_tensor_cores(
                tuned=tuned,
                batch=batch,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            subseq = None
            if (
                self._dispatch == "device"
                and kv_dtype == torch.float8_e4m3fn
                and xqa_decode_qualified(
                    indices.device,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    page_size=page_size,
                    head_dim=head_dim,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    pos_encoding_mode=pos_encoding_mode,
                    logits_soft_cap=logits_soft_cap,
                    window_left=window_left,
                    rope_scale=rope_scale,
                    rope_theta=rope_theta,
                )
            ):
                subseq = xqa_decode_subseq(
                    indices.device, num_qo_heads, num_kv_heads, head_dim, batch
                )
            geometry = (batch, num_qo_heads, num_kv_heads, head_dim, page_size, dtype, subseq)
            route = (
                ("native", use_tensor_cores)
                if subseq is None
                else ("xqa", geometry, sm_scale, window_left)
            )
            cute_config = None
            if (
                self._dispatch == "device"
                and kv_dtype == torch.float8_e4m3fn
                and fp8_decode_cute_qualified(
                    indices.device,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                    pos_encoding_mode=pos_encoding_mode,
                    **{name: value for name, value in options.items() if name != "sm_scale"},
                )
            ):
                cute_config = fp8_decode_cute(
                    indices.device, num_qo_heads, num_kv_heads, head_dim, batch
                )
            if cute_config is not None:
                cute_options = {
                    "batch_size": batch,
                    "num_qo_heads": num_qo_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                }
                # Resolve environment overrides now so graph replans cannot silently
                # change the captured configuration. Construction can clamp splits.
                cute_config = _decode_fp8.configuration(
                    indices.device, **cute_options, **cute_config
                )
                cute_options.update(
                    page_size=page_size,
                    q_dtype=dtype,
                    sm_scale=_decode_fp8._scale(
                        head_dim**-0.5 if sm_scale is None else sm_scale, "sm_scale"
                    ),
                )
                geometry = (*cute_options.values(), tuple(sorted(cute_config.items())))
                route = ("cute", geometry)
            if (
                self._use_cuda_graph
                and self._decode_route is not None
                and route != self._decode_route
            ):
                raise RuntimeError(
                    "graph decode plans must keep the dtypes and options of the first plan"
                )
            if cute_config is not None:
                self._plan_cute_decode(
                    cu_seqlens_k,
                    indices,
                    last_page_len,
                    geometry=geometry,
                    config=cute_config,
                    **cute_options,
                )
                self._decode_route = route
                self._planned = True
                return
            if subseq is not None:
                required = xqa_decode_scratch_size(
                    batch, num_kv_heads, subseq, head_dim, num_qo_heads // num_kv_heads
                )
                scratch = self._workspaces[0]
                if scratch.dtype != torch.uint8 or scratch.numel() < required:
                    raise ValueError(
                        f"XQA needs at least {required} borrowed float workspace bytes"
                    )
                if self._xqa_geometry != geometry:
                    width = 1
                    if self._use_cuda_graph:
                        buffer = self._decode_options["paged_kv_indices_buffer"]
                        if buffer is None:
                            raise ValueError("graph XQA decode requires paged_kv_indices_buffer")
                        width = buffer.numel()
                    self._xqa = _FlashInferXQADecode(
                        device=indices.device,
                        batch=batch,
                        width=width,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        group=num_qo_heads // num_kv_heads,
                        page_size=page_size,
                        subseq=subseq,
                        graph=self._use_cuda_graph,
                        dtype=dtype,
                    )
                    self._xqa_geometry = geometry
                self._xqa.plan(cu_seqlens_k, indices, last_page_len, seq_lens)
                self._xqa_options = (sm_scale, window_left)
                self._decode_route = route
                self._planned = True
                return
            if self._dispatch == "device":
                # Eager plans of different batches may take different kernels; each
                # kernel keeps one native wrapper over the shared workspaces.
                wrapper = self._native_decoders.get(use_tensor_cores)
                if wrapper is None:
                    wrapper = self._decode_constructor(
                        self._workspaces[0],
                        **self._decode_options | {"use_tensor_cores": use_tensor_cores},
                    )
                    wrapper.reset_workspace_buffer(*self._workspaces)
                    self._native_decoders[use_tensor_cores] = wrapper
                self._wrapper = wrapper
                self._use_tensor_cores = use_tensor_cores
                self._decode_options["use_tensor_cores"] = use_tensor_cores
            self._decode_route = route
            split_options = {}
            use_table = (
                self._use_tensor_cores
                and tuned
                and self._backend == "fa2"
                and qualified_flashinfer_version()
            )
            # The public planning contract supplies CPU lengths; device lengths keep the
            # provider default rather than synchronizing to read them.
            if use_table and cu_seqlens_k.device.type == "cpu" and seq_lens.device.type == "cpu":
                pages = cu_seqlens_k.diff().tolist()
                row = lookup(
                    indices.device, "decode", num_qo_heads, num_kv_heads, head_dim, len(pages)
                )
                # One mode-independent capacity: eager and graph plans of the same
                # lengths take the same chunk, hence the same partials and merge.
                chunk = decode_chunk(
                    row,
                    pages,
                    int(seq_lens.max()) if pages else 0,
                    decode_split_capacity(indices.device, num_kv_heads),
                )
                if chunk is not None:
                    split_options["fixed_split_size"] = chunk
            use_fp8_output = (
                dtype == torch.bfloat16
                and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and (not self._use_tensor_cores or self._backend == "fa2")
                and staged_output_flashinfer_version()
            )
            shape = (batch, num_qo_heads, head_dim)
            if (
                self._use_cuda_graph
                and self._planned
                and (
                    use_fp8_output != self._use_fp8_output
                    or (use_fp8_output and self._fp8_output.stage.shape != shape)
                )
            ):
                raise RuntimeError("graph decode plans must keep output staging of the first plan")
            self._use_fp8_output = use_fp8_output
            if self._use_fp8_output:
                from ._scale_cast import FP16StagedOutput

                if self._fp8_output is None or self._fp8_output.device != indices.device:
                    self._fp8_output = FP16StagedOutput(shape, indices.device)
                else:
                    self._fp8_output.reserve(shape)
                split_options["o_data_type"] = torch.float16
            self._wrapper.plan(
                indptr=cu_seqlens_k,
                indices=indices,
                last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                pos_encoding_mode=pos_encoding_mode,
                seq_lens=seq_lens,
                data_type=kv_dtype,
                q_data_type=dtype,
                kv_data_type=kv_dtype,
                **options,
                non_blocking=True,
                **split_options,
            )
            if self._use_fp8_output:
                self._fp8_output.prepare(
                    self._wrapper, head_dim=head_dim, idx_dtype=cu_seqlens_k.dtype, decode=True
                )
            if use_table and self._use_cuda_graph:
                capacity = decode_plan_capacity(self._wrapper)
                if capacity is not None:
                    _, layout = capacity
                    if self._capture_plan_info is None:
                        self._capture_plan_info = layout
                    elif split_options and layout != self._capture_plan_info:
                        raise RuntimeError(
                            "FlashInfer table split changed the captured plan layout"
                        )
        else:
            split_options = {}
            if self._use_cuda_graph and self._dispatch == "device":
                if self._plan_cute_prefill(
                    cu_seqlens_q,
                    cu_seqlens_k,
                    indices,
                    last_page_len,
                    heads=(num_qo_heads, num_kv_heads, head_dim),
                    page_size=page_size,
                    dtype=dtype,
                    options=options,
                    tuned=tuned and causal,
                ):
                    self._planned = True
                    return
                cu_seqlens_q, cu_seqlens_k, last_page_len, seq_lens, split_options = (
                    self._plan_padded_prefill(
                        cu_seqlens_q,
                        cu_seqlens_k,
                        last_page_len,
                        seq_lens,
                        heads=(num_qo_heads, num_kv_heads, head_dim),
                        page_size=page_size,
                        dtype=dtype,
                        options=options,
                        tuned=tuned and causal,
                    )
                )
            if (
                tuned
                and not self._use_cuda_graph
                and head_dim == 128
                and causal
                and page_size == 1
                and all(
                    value.device.type == "cpu" for value in (cu_seqlens_q, cu_seqlens_k, seq_lens)
                )
            ):
                split = prefill_query_split(
                    device_key(self._workspaces[0].device),
                    cu_seqlens_q.tolist(),
                    cu_seqlens_k.tolist(),
                    seq_lens.tolist(),
                    head_dim=head_dim,
                    causal=causal,
                    page_size=page_size,
                )
                if split is not None and self._workspace_fits(
                    batch=len(split.seq_lens),
                    total=split.qo_indptr[-1],
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                ):
                    indices = self._gather_prefill_indices(indices, split.gather_positions)
                    self._prefill_metadata = tuple(
                        torch.tensor(values, dtype=torch.int32, device="cpu")
                        for values in split[:4]
                    )
                    cu_seqlens_q, cu_seqlens_k, last_page_len, seq_lens = self._prefill_metadata
            if custom_mask is not None and packed_custom_mask is None:
                cu_seqlens_q, cu_seqlens_k, last_page_len = custom_mask_metadata(
                    custom_mask, cu_seqlens_q, cu_seqlens_k, last_page_len
                )
            self._wrapper.plan(
                qo_indptr=cu_seqlens_q,
                paged_kv_indptr=cu_seqlens_k,
                paged_kv_indices=indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                pos_encoding_mode=pos_encoding_mode,
                seq_lens=seq_lens,
                q_data_type=dtype,
                kv_data_type=kv_dtype,
                **options,
                non_blocking=True,
                causal=causal,
                disable_split_kv=disable_split_kv,
                o_data_type=o_dtype,
                custom_mask=custom_mask,
                packed_custom_mask=packed_custom_mask,
                **split_options,
            )
        self._planned = True

    def run(
        self,
        *,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        if not self._planned:
            raise RuntimeError("FlashInfer paged attention requires plan() before run()")
        check_dense_output(out)
        if self.prefill_route == "cute" or self.decode_route == "cute":
            if q.ndim == 3 and (
                q.data_ptr() % 16
                or q.stride(0) % 8
                or q.stride(1) % 8
                or (self.decode_route == "cute" and q.stride(1) != q.shape[2])
            ):
                reject_grad(q)
                if (
                    q.stride(2) != 1
                    or q.dtype != self._cute_q_stage.dtype
                    or q.device != self._cute_q_stage.device
                ):
                    raise ValueError(
                        "q must match the plan dtype/device and have contiguous last dimension"
                    )
                dense_q = self._cute_q_stage[: q.shape[0]]
                if dense_q.shape != q.shape:
                    raise ValueError("q must match the plan shape and capacity")
                dense_q.copy_(q)
                q = dense_q
        if self.prefill_route == "cute":
            if out is None:
                out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            return self._cute_prefill.run(
                q,
                *paged_kv_cache,
                *self._cute_prefill_csr,
                out,
                k_scale=1.0 if k_scale is None else k_scale,
                v_scale=1.0 if v_scale is None else v_scale,
            )
        if self.decode_route == "cute":
            if out is None:
                out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            return self._cute.run(
                q,
                *paged_kv_cache,
                *self._cute_csr,
                out,
                k_scale=1.0 if k_scale is None else k_scale,
                v_scale=1.0 if v_scale is None else v_scale,
            )
        if self._decode_route is not None and self._decode_route[0] == "xqa":
            return self._xqa.run(
                q,
                *paged_kv_cache,
                out,
                self._workspaces[0][: self._xqa.scratch_bytes],
                k_scale,
                v_scale,
                *self._xqa_options,
            )
        if self._is_decode and self._use_fp8_output:
            return self._fp8_output.run(
                self._wrapper,
                q=q,
                paged_kv_cache=paged_kv_cache,
                out=out,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        options = {
            name: value
            for name, value in (("k_scale", k_scale), ("v_scale", v_scale))
            if value is not None
        }
        if out is not None:
            options["out"] = out
        return self._wrapper.run(q=q, paged_kv_cache=paged_kv_cache, **options)


def flash_attention_with_kv_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: bool | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Read NHD paged K/V using CUDA int32 page/length metadata for inference.

    Query is [tokens, qo_heads, head_dim]; caches are [pages, page_size,
    kv_heads, head_dim]. The provider allocates output on the current CUDA stream
    without modifying inputs. There is no backward. Callers own metadata validity
    and stable storage during capture/replay; this wrapper does not scan values.
    """
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as error:
        raise ImportError(
            "The `sglang-kernel` distribution (imported as `sgl_kernel`) is unavailable.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from error

    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,
    )


def trtllm_context_attention(
    q: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    *,
    workspace_buffer: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Causal NHD paged context attention, without backward or input mutation.

    Query is [tokens, qo_heads, head_dim]; K/V are [pages, page_size, kv_heads,
    head_dim]. Page tables, sequence lengths and cumulative lengths are CUDA
    int32 tensors. The provider writes the borrowed workspace and allocates output
    on the current CUDA stream. Callers own metadata validity and storage lifetime.
    """
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache

    return trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=page_table,
        seq_lens=cache_seqlens,
        max_q_len=max_seqlen_q,
        max_kv_len=max_seqlen_k,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        cum_seq_lens_q=cu_seqlens_q,
        cum_seq_lens_kv=cu_seqlens_k,
        kv_layout="NHD",
        batch_size=batch_size,
        out_dtype=q.dtype,
    )


def trtllm_decode_attention(
    q: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    *,
    workspace_buffer: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """NHD paged decode attention with one query token per sequence.

    Uses the context operation's layouts, dtype/metadata ownership and mutation
    contract. This inference-only operation allocates output on the current CUDA
    stream. For graph replay, callers retain workspace/cache/metadata addresses.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    return trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=page_table,
        seq_lens=cache_seqlens,
        max_seq_len=max_seqlen_k,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        kv_layout="NHD",
        out_dtype=q.dtype,
    )


def paged_kv_metadata(
    page_table: torch.Tensor,
    rows: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    page_size: int,
    granularity: Literal["page", "token"],
    max_kv_len: int,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    """Build FlashInfer's paged-KV CSR on device in one Triton launch (CUDA 8.9+).

    Request b reads ``page_table[rows[b]]`` (int32 page IDs, ``[R, W]`` with unit column
    stride) over ``kv_lens[b]`` tokens. ``granularity="page"`` writes its page IDs
    (FlashInfer page size ``page_size``); ``"token"`` writes token slots
    ``page * page_size + offset`` (FlashInfer page size 1). Outputs, int32 and caller-owned:
    ``kv_indptr [B + 1]``, ``kv_indices`` (entries past its length are dropped, never
    written out of bounds) and ``kv_last_page_len [B]`` (``(len - 1) % page + 1``, 1 per
    token entry, 0 for an empty request). ``max_kv_len`` bounds every ``kv_lens[b]`` and
    only sizes the grid. No host reads of device values: graph capturable over borrowed
    buffers; B and the output addresses are fixed per capture.
    """
    from .triton import _kv_metadata

    if granularity not in ("page", "token"):
        raise ValueError("granularity must be 'page' or 'token'")
    if type(page_size) is not int or page_size < 1 or type(max_kv_len) is not int or max_kv_len < 0:
        raise ValueError("page_size must be a positive int and max_kv_len a non-negative int")
    # Cheap per-call checks (this runs every eager decode step).
    batch = rows.numel()
    device = page_table.get_device()
    vectors = (rows, kv_lens, kv_indptr, kv_indices, kv_last_page_len)
    if (
        device < 0
        or page_table.dtype is not torch.int32
        or page_table.ndim != 2
        or page_table.stride(1) != 1
        or page_table.requires_grad
        or any(
            t.dtype is not torch.int32
            or t.ndim != 1
            or t.stride(0) != 1
            or t.get_device() != device
            or t.requires_grad
            for t in vectors
        )
        or kv_lens.numel() != batch
        or kv_last_page_len.numel() != batch
        or kv_indptr.numel() != batch + 1
    ):
        raise ValueError(
            "paged_kv_metadata takes int32 CUDA tensors without grad: page_table [R, W] with "
            "unit column stride; contiguous rows, kv_lens, kv_last_page_len [B], "
            "kv_indptr [B + 1] and kv_indices [capacity], all on one device"
        )
    _kv_metadata.build(
        page_table,
        rows,
        kv_lens,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        page_size=page_size,
        tokens=granularity == "token",
        max_kv_len=max_kv_len,
    )


class TritonPagedAttention:
    """Portable Triton paged attention (CUDA compute capability 8.9+): decode, extend with a
    cached prefix, and mixed batches, over a page-size-1 token CSR.

    ``plan`` borrows int32 device metadata: sequence s owns query rows
    ``q_indptr[s]:q_indptr[s + 1]`` and the cache slots ``kv_indices[kv_indptr[s]:
    kv_indptr[s + 1]]``, its own queries' keys last (queries are bottom-right aligned).
    It launches nothing and reads no device value, so a CUDA graph may capture ``run`` over
    planned buffers that the caller rewrites before each replay. ``run`` allocates only its
    FP32 partials (in the capturing graph's pool) and writes every row of ``out``; a
    sequence with no keys gets zeros, zero rows are a no-op. No atomics: deterministic.

    ``kv_dtype`` (default ``dtype``) may be FP8 e4m3/e5m2 with per-tensor ``k_scale`` and
    ``v_scale`` at run; a captured graph keeps the scales it was captured with.

    Each run splits every row's keys into S segments reduced in order, S about 2.5
    programs per SM over the run's query blocks (at most 64; long prefills use one); a
    row's bits depend on the rest of the batch only through S. With
    ``batch_invariant=True`` decode plans keep the S of one sequence (lowered until
    ``max_rows`` rows of FP32 partials fit 64 MiB) and other plans one segment, so a
    row's bits never depend on the rest of its plan.
    """

    def __init__(
        self,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_dtype: torch.dtype | None = None,
        sm_scale: float | None = None,
        batch_invariant: bool = False,
        max_rows: int | None = None,
    ) -> None:
        from .triton import _paged

        device = torch.device(device)
        if device.type != "cuda":
            raise NotImplementedError("Triton paged attention needs a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        sms, self._pdl = _paged.device_traits(device)
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Triton paged attention requires FP16 or BF16 queries")
        kv_dtype = dtype if kv_dtype is None else kv_dtype
        if kv_dtype not in (dtype, torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("the KV cache must have the query dtype or FP8 e4m3/e5m2")
        if num_kv_heads < 1 or num_qo_heads % num_kv_heads:
            raise ValueError("query heads must be a positive multiple of KV heads")
        if head_dim % 16 or not 16 <= head_dim <= 256:
            raise ValueError("head_dim must be a multiple of 16 in [16, 256]")
        if batch_invariant != (max_rows is not None):
            raise ValueError("max_rows is given exactly when batch_invariant is set")
        if max_rows is not None and (type(max_rows) is not int or max_rows < 1):
            raise ValueError("max_rows must be a positive int")
        self._provider = _paged
        self._heads = (num_qo_heads, num_kv_heads, head_dim)
        self._dtype, self._kv_dtype, self._device = dtype, kv_dtype, device
        self._scale = head_dim**-0.5 if sm_scale is None else sm_scale
        self._sms = sms
        # A batch-invariant op keeps one split count; otherwise each decode plan picks it.
        self._segments = None
        if batch_invariant:
            self._segments = _paged.invariant_split_count(
                sms=sms,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_rows=max_rows,
            )
        self._plan = None

    def plan(
        self,
        *,
        q_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        max_q_len: int,
        causal: bool = True,
        window_left: int = -1,
    ) -> None:
        """Borrow the metadata of the following runs; ``max_q_len`` bounds every
        sequence's queries (1 selects decode). ``window_left`` W >= 0 keeps keys at most
        W positions before their query, as in FlashInfer."""
        for name, value in (("q_indptr", q_indptr), ("kv_indptr", kv_indptr)):
            if value.ndim != 1 or value.numel() < 1:
                raise ValueError(f"{name} must have shape [B + 1]")
        if q_indptr.numel() != kv_indptr.numel():
            raise ValueError("q_indptr and kv_indptr must describe the same sequences")
        for value in (q_indptr, kv_indptr, kv_indices):
            reject_grad(value)
            if value.dtype != torch.int32 or value.device != self._device:
                raise ValueError("metadata must be int32 on the operation device")
        if type(max_q_len) is not int or max_q_len < 1:
            raise ValueError("max_q_len must be a positive int")
        if type(window_left) is not int or window_left < -1:
            raise ValueError("window_left must be an int >= -1")
        self._plan = (
            q_indptr,
            kv_indptr,
            kv_indices,
            max_q_len,
            0 if causal else -1,
            window_left,
        )

    def run(
        self,
        *,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> torch.Tensor:
        """Attend ``q`` [rows, Hq, D] (any row stride) over ``(k, v)`` [pages, page, Hkv, D]
        or [slots, Hkv, D] caches into ``out`` [rows, Hq, D]."""
        if self._plan is None:
            raise RuntimeError("plan() must precede run()")
        q_indptr, kv_indptr, kv_indices, max_q_len, window_right, window_left = self._plan
        hq, hk, d = self._heads
        k, v = (cache.view(-1, hk, d) for cache in paged_kv_cache)
        for name, value, dtype in (
            ("q", q, self._dtype),
            ("out", out, self._dtype),
            ("k", k, self._kv_dtype),
            ("v", v, self._kv_dtype),
        ):
            reject_grad(value)
            if value.dtype != dtype or value.device != self._device or value.stride(-1) != 1:
                raise ValueError(
                    f"{name} must be {dtype} on the operation device with unit last stride"
                )
        if q.shape != out.shape or q.shape[1:] != (hq, d):
            raise ValueError(f"q and out must have shape [rows, {hq}, {d}]")
        num_seqs = q_indptr.numel() - 1
        decode = max_q_len == 1
        if decode and q.shape[0] != num_seqs:
            raise ValueError("a decode plan takes one query row per sequence")
        if self._segments is not None:
            segments = self._segments if decode else 1
        else:
            segments = self._provider.split_count(
                sms=self._sms,
                num_qo_heads=hq,
                num_kv_heads=hk,
                head_dim=d,
                num_seqs=num_seqs,
                rows=q.shape[0],
                fixed_q=1 if decode else 0,
            )
        return self._provider.attend(
            q,
            k,
            v,
            out,
            q_indptr=q_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            num_seqs=num_seqs,
            fixed_q=1 if decode else 0,
            segments=segments,
            window_left=window_left,
            window_right=window_right,
            scale=self._scale,
            k_scale=1.0 if k_scale is None else k_scale,
            v_scale=1.0 if v_scale is None else v_scale,
            pdl=self._pdl,
        )


# DeepSeek-V4 sparse attention over packed 584-byte KV rows (one ABI on every device).


class DSV4Pages(NamedTuple):
    """Geometry of one packed DSV4 KV pool: ``rows`` rows per page (64: window and c4, 2: c128).

    A page holds ``rows`` data records of 576 bytes (448 E4M3 non-rotary channels, then 64
    BF16 rotary channels), then ``rows`` scale records of 8 bytes (7 UE8M0 exponents, one per
    64 non-rotary channels, and a zero pad byte), padded to a multiple of 576 bytes. A pool is
    a contiguous uint8 tensor of ``pages * page_bytes`` bytes; row ``r`` lives in page
    ``r // rows`` at slot ``r % rows``. Run :meth:`prepare` once on every new pool: row 0 must
    hold finite bytes (the sm12x kernels read it for masked index slots).
    """

    rows: int

    ROW_BYTES = 584
    DATA_BYTES = 576
    SCALE_BYTES = 8
    NOPE = 448
    ROPE = 64

    @property
    def page_bytes(self) -> int:
        return -(-self.rows * self.ROW_BYTES // self.DATA_BYTES) * self.DATA_BYTES

    def check(self, cache: torch.Tensor) -> None:
        if cache.dtype != torch.uint8 or not cache.is_contiguous():
            raise TypeError("a DSV4 KV pool is a contiguous uint8 tensor")
        if cache.numel() % self.page_bytes:
            raise ValueError(
                f"pool bytes {cache.numel()} are not whole {self.page_bytes}-byte pages"
            )

    def prepare(self, cache: torch.Tensor) -> None:
        """Zero row 0 (its 576 data bytes and 8 scale bytes), once after allocating the pool.
        Later KV writes to row 0 keep it finite, so page 0 may still hold data."""
        self.check(cache)
        flat, scale = cache.view(-1), self.rows * self.DATA_BYTES
        flat[: self.DATA_BYTES].zero_()
        flat[scale : scale + self.SCALE_BYTES].zero_()


class DSV4Rows(NamedTuple):
    """The rows each query reads from one pool: ``indices [T, K]`` int32 row ids, of which the
    first ``lengths [T]`` int32 count (-1 entries are skipped)."""

    cache: torch.Tensor
    pages: DSV4Pages
    indices: torch.Tensor
    lengths: torch.Tensor


_DSV4_WINDOW = 128


_FI_SPLIT_TOKENS = 64  # FlashInfer sm12x: calls of <= 64 queries may run its split-K decode


def _split_heads(heads: int) -> int:
    """Head extent of FlashInfer's sm12x split-K scratch: 8 for 8 heads, else a multiple of 16."""
    return 8 if heads == 8 else -(-heads // 16) * 16


def _dsv4_kernels():
    """The Triton, FlashInfer and FlashMLA DSV4 modules (imported on first use: CPU builds
    lack Triton)."""
    from .cuda import _dsv4_flashmla as flashmla
    from .flashinfer import _dsv4 as flashinfer
    from .triton import _dsv4 as triton

    return triton, flashinfer, flashmla


def _dsv4_table(table: torch.Tensor, positions: torch.Tensor) -> None:
    if table.dtype != torch.float32 or table.ndim != 2 or table.shape[1] != DSV4Pages.ROPE:
        raise TypeError("the rotary table is float32 [positions, 64] (32 cosines, 32 sines)")
    if not table.is_contiguous():
        raise ValueError("the rotary table must be contiguous")
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")


class DSV4ExtendPlan(NamedTuple):
    """One forward's extend metadata, built by :meth:`DSV4SparseAttention.plan_extend` and
    reused by every layer's :meth:`DSV4SparseAttention.extend` (valid while the page table
    and the request boundaries it was built from hold)."""

    tokens: int
    counts: tuple[int, int, int]  # staged window rows, c4 entries, c128 entries
    positions: torch.Tensor  # int64 [T]
    query_request: torch.Tensor  # int32 [T]
    window: torch.Tensor  # int32 [T, 128] (sm10x: staged local rows; sm12x: pool rows)
    window_lengths: torch.Tensor  # int32 [T]
    c4_lengths: torch.Tensor  # int32 [T]
    c128: torch.Tensor  # int32 [T, C] (same convention as window)
    c128_lengths: torch.Tensor  # int32 [T]
    window_rows: torch.Tensor | None  # sm10x: int32 [Nw] pool rows to stage
    c4_rows: torch.Tensor | None  # sm10x: int32 [N4] pool rows of every request's c4 entries
    c128_rows: torch.Tensor | None  # sm10x: int32 [N128]
    c4_base: torch.Tensor  # int32 [R + 1] request -> first c4 entry slot
    c128_base: torch.Tensor  # int32 [R + 1]


class DSV4SparseAttention:
    """DeepSeek-V4 sparse MLA: each query attends to its window rows and, on compressed
    layers, its extra (c4 top-k or c128) rows, key = value, with one FP32 sink per head.

    ``q [T, H, 512]`` BF16; outputs ``[T, H, 512]`` BF16 (the rotary tail still rotated).

    Decode (``__call__``, per-query :class:`DSV4Rows`, build them from the page table with
    :func:`dsv4_page_rows`): sm100/sm103 dequantize each query's rows into a BF16 scratch
    (Triton) for FlashInfer's TRTLLM-GEN DSV4 decode; sm120/sm121 run FlashInfer's packed
    sparse backend. Calls of up to ``max_tokens`` queries with up to ``max_extra`` extra
    rows are graph capturable after one eager call.

    Extend (:meth:`plan_extend` once per forward, :meth:`extend` per layer; no host syncs):
    sm100/sm103 stage each request's window rows and compressed entries once per layer into
    a BF16 row set and run FlashMLA's sm100 sparse prefill (heads padded to 64); sm120/sm121
    map to pool rows for FlashInfer's packed sparse backend. Sized by ``max_extend_tokens``,
    ``max_seq_len`` (the longest request, in tokens) and ``max_requests``.

    Every pool must be prepared once (:meth:`DSV4Pages.prepare`): FlashInfer's sm12x kernels
    gather row 0 for masked index slots, and a non-finite row 0 turns the output NaN.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        num_heads: int,
        max_tokens: int,
        max_extra: int,
        scale: float,
        max_extend_tokens: int = 0,
        max_seq_len: int = 0,
        max_requests: int = 1,
    ):
        device = torch.device(device)
        properties = torch.cuda.get_device_properties(device)
        self.capability = (properties.major, properties.minor)
        self.sms = properties.multi_processor_count
        self.provider = {
            (10, 0): "flashinfer-trtllm",
            (10, 3): "flashinfer-trtllm",
            (12, 0): "flashinfer-sparse",
            (12, 1): "flashinfer-sparse",
        }.get(self.capability)
        if self.provider is None:
            raise NotImplementedError(
                f"DSV4 sparse attention has no provider on sm{self.capability}"
            )
        self.device, self.num_heads, self.scale = device, num_heads, float(scale)
        self.max_tokens, self.max_extra = max_tokens, max_extra
        self._module = None
        if self.provider != "flashinfer-sparse":
            self.workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=device)
            self.width = _DSV4_WINDOW + -(-max_extra // 4) * 4
            self.scratch = torch.empty(
                max_tokens * self.width, 512, dtype=torch.bfloat16, device=device
            )
            self.table = torch.empty(max_tokens * self.width, dtype=torch.int32, device=device)
            self.lengths = torch.empty(2, max_tokens, dtype=torch.int32, device=device)
        self._extend = None
        if max_extend_tokens:
            self._extend = self._extend_buffers(max_extend_tokens, max_seq_len, max_requests)

    @staticmethod
    def _split_bytes(tokens: int, heads: int, splits: int) -> int:
        """sm12x: FlashInfer's output LSE for ``tokens`` queries plus its split-K scratch, used
        by calls of <= 64 queries."""
        split_tokens = min(tokens, _FI_SPLIT_TOKENS)
        return tokens * heads * 4 + split_tokens * _split_heads(heads) * splits * (512 * 2 + 4)

    @staticmethod
    def _split_counts(rows: int, heads: int, splits: int) -> tuple[int, int, int]:
        """Elements of (split-K partial outputs BF16, split-K LSE FP32, output LSE FP32)."""
        split_rows, padded = min(rows, _FI_SPLIT_TOKENS), _split_heads(heads)
        return split_rows * padded * splits * 512, split_rows * padded * splits, rows * heads

    def _decode_scratch(self, tokens: int, splits: int) -> dict[str, torch.Tensor]:
        """sm12x decode scratch: the active GraphWorkspace's slot (shared by every graph and
        layer of the group, sized at its warm-ups for ``max_rows``), else call-owned (a
        capture outside a workspace takes it from the graph's pool)."""
        heads, call = self.num_heads, (self.num_heads, splits)
        counts = self._split_counts(tokens, *call)

        def allocate(n):
            return tuple(
                torch.empty((max(k, 1),), dtype=d, device=self.device)
                for k, d in zip(n, (torch.bfloat16, torch.float32, torch.float32), strict=True)
            )

        group = GraphWorkspace.active()
        if group is None:
            work = allocate(counts)
        else:
            slot = ("dsv4_sparse_attention", self.device)
            group.reserve(slot, call, self._split_counts, allocate)
            work = group.take(slot, tokens, counts)
        split_rows, padded = min(tokens, _FI_SPLIT_TOKENS), _split_heads(heads)
        shape = (split_rows, padded, splits)
        return {
            "mid_out": work[0][: counts[0]].view(*shape, 512),
            "mid_lse": work[1][: counts[1]].view(shape),
            "lse": work[2][: counts[2]],
        }

    def _split_buffers(self, tokens: int, splits: int) -> dict[str, torch.Tensor]:
        heads, padded = self.num_heads, _split_heads(self.num_heads)
        split_tokens = min(tokens, _FI_SPLIT_TOKENS)
        f32 = {"dtype": torch.float32, "device": self.device}
        return {
            "lse": torch.empty(tokens * heads, **f32),
            "mid_out": torch.empty(
                split_tokens, padded, splits, 512, dtype=torch.bfloat16, device=self.device
            ),
            "mid_lse": torch.empty(split_tokens, padded, splits, **f32),
        }

    @staticmethod
    def _extend_splits(c128: int) -> int:
        return (_DSV4_WINDOW + max(512, c128)) // 64

    @staticmethod
    def _extend_sizes(staged: bool, max_tokens: int, max_seq_len: int, max_requests: int):
        c128 = -(-(max_seq_len // 128) // 64) * 64  # one query's c128 entries, 64-padded
        rows = {
            "window": max_tokens * _DSV4_WINDOW,
            "c128": max_tokens * c128,
            "lengths": 3 * max_tokens,
            "bases": 2 * (max_requests + 1),
        }
        if staged:
            rows["c4_rows"] = max_requests * max_seq_len // 4 + max_requests
            rows["extra"] = max_tokens * (_DSV4_WINDOW + max(512, c128))
            rows["window_rows"] = max_tokens + 127 * max_requests
            rows["c128_rows"] = max_requests * max_seq_len // 128 + max_requests
        staged_rows = rows.get("window_rows", 0) + max(
            rows.get("c4_rows", 0), rows.get("c128_rows", 0)
        )
        bf16 = staged_rows * 512 * 2 if staged else 0
        pad = 2 * max_tokens * 64 * 512 * 2 + 2 * max_tokens * 64 * 4 if staged else 0
        return rows, c128, bf16 + pad + 4 * sum(rows.values())

    @staticmethod
    def extend_workspace_bytes(
        device: torch.device,
        max_tokens: int,
        max_seq_len: int,
        max_requests: int = 1,
        *,
        num_heads: int = 128,
    ) -> int:
        """Bytes the extend buffers of an instance built with these limits hold for its
        lifetime (on top of the decode buffers). ``num_heads`` (default: the upper bound 128)
        sizes sm12x's FlashInfer scratch."""
        staged = torch.cuda.get_device_capability(device)[0] == 10
        cls = DSV4SparseAttention
        rows, c128, total = cls._extend_sizes(staged, max_tokens, max_seq_len, max_requests)
        if not staged:
            total += cls._split_bytes(max_tokens, num_heads, cls._extend_splits(c128))
        return total

    def _extend_buffers(self, max_tokens, max_seq_len, max_requests):
        staged = self.provider == "flashinfer-trtllm"
        rows, c128, _ = self._extend_sizes(staged, max_tokens, max_seq_len, max_requests)
        i32 = {"dtype": torch.int32, "device": self.device}
        buffers = {name: torch.empty(n, **i32) for name, n in rows.items()}
        buffers["window"] = buffers["window"].view(max_tokens, _DSV4_WINDOW)
        buffers["c128"] = buffers["c128"].view(max_tokens, c128)
        if staged:
            buffers["extra"] = buffers["extra"].view(max_tokens, -1)
        else:
            buffers.update(self._split_buffers(max_tokens, self._extend_splits(c128)))
        buffers["lengths"] = buffers["lengths"].view(3, max_tokens)
        buffers["bases"] = buffers["bases"].view(2, max_requests + 1)
        if staged:
            n = rows["window_rows"] + max(rows["c4_rows"], rows["c128_rows"])
            buffers["kv"] = torch.empty(n, 512, dtype=torch.bfloat16, device=self.device)
            buffers["q_pad"] = torch.zeros(
                max_tokens, 64, 512, dtype=torch.bfloat16, device=self.device
            )
            buffers["out_pad"] = torch.empty_like(buffers["q_pad"])
            buffers["stats"] = torch.empty(
                2, max_tokens, 64, dtype=torch.float32, device=self.device
            )
            buffers["sinks_pad"] = torch.full((64,), float("-inf"), device=self.device)
        buffers["limits"] = (max_tokens, max_seq_len, max_requests)
        return buffers

    def plan_extend(
        self,
        page_table: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        positions: torch.Tensor,
        query_request: torch.Tensor,
        *,
        counts: tuple[int, int, int],
    ) -> DSV4ExtendPlan:
        """Metadata of one extend forward, without host syncs.

        ``page_table`` int32 ``[R, W]`` holds one id per 256-token page for every pool;
        request ``r`` extends tokens ``[starts[r], ends[r])`` (int64 ``[R]``); query ``t`` is
        position ``positions[t]`` (int64) of request ``query_request[t]`` (int32).
        ``counts = (Nw, N4, N128)`` as Python ints: ``Nw = Σ ends − max(starts − 127, 0)``,
        ``N4 = Σ ends // 4``, ``N128 = Σ ends // 128``.
        """
        if self._extend is None:
            raise RuntimeError("build DSV4SparseAttention with max_extend_tokens for extend")
        triton = _dsv4_kernels()[0]
        b = self._extend
        max_tokens, max_seq_len, max_requests = b["limits"]
        tokens, requests = positions.shape[0], page_table.shape[0]
        nw, n4, n128 = counts
        staged = self.provider == "flashinfer-trtllm"
        if tokens > max_tokens or requests > max_requests or (staged and n4 > b["c4_rows"].numel()):
            raise ValueError("extend batch exceeds the instance's extend limits")
        if positions.dtype != torch.int64 or starts.dtype != torch.int64:
            raise TypeError("positions and starts are int64")
        first = (starts - 127).clamp_min(0)
        c4_base, c128_base = b["bases"][0, : requests + 1], b["bases"][1, : requests + 1]
        c4_base[0], c128_base[0] = 0, 0
        c4_base[1:].copy_(torch.cumsum(ends // 4, 0))
        c128_base[1:].copy_(torch.cumsum(ends // 128, 0))
        lengths = b["lengths"][:, :tokens]
        lengths[0].copy_((positions + 1).clamp_(max=_DSV4_WINDOW))
        lengths[1].copy_(((positions + 1) // 4).clamp_(max=512))
        lengths[2].copy_((positions + 1) // 128)
        window, c128 = b["window"][:tokens], b["c128"][:tokens]
        window_rows = c4_rows = c128_rows = None
        if staged:
            c4_rows = b["c4_rows"][:n4]
            triton.staged_rows(
                page_table, c4_base, None, requests, n4, c4_rows, per_page=64, window=False
            )
            w_base = torch.cat((first.new_zeros(1), torch.cumsum(ends - first, 0))).int()
            window_rows = b["window_rows"][:nw]
            triton.staged_rows(
                page_table, w_base, first, requests, nw, window_rows, per_page=256, window=True
            )
            c128_rows = b["c128_rows"][:n128]
            triton.staged_rows(
                page_table, c128_base, None, requests, n128, c128_rows, per_page=2, window=False
            )
            triton.query_rows(
                page_table,
                positions,
                query_request,
                window,
                col0=0,
                width=_DSV4_WINDOW,
                window=True,
                first=first,
                base=w_base,
            )
            triton.query_rows(
                page_table,
                positions,
                query_request,
                c128,
                col0=0,
                width=c128.shape[1],
                window=False,
                base=c128_base,
                offset=nw,
            )
        else:
            triton.query_rows(
                page_table,
                positions,
                query_request,
                window,
                col0=0,
                width=_DSV4_WINDOW,
                window=True,
            )
            triton.query_rows(
                page_table,
                positions,
                query_request,
                c128,
                col0=0,
                width=c128.shape[1],
                window=False,
            )
        return DSV4ExtendPlan(
            tokens,
            (nw, n4, n128),
            positions,
            query_request,
            window,
            lengths[0],
            lengths[1],
            c128,
            lengths[2],
            window_rows,
            c4_rows,
            c128_rows,
            c4_base,
            c128_base,
        )

    def extend(
        self,
        q: torch.Tensor,
        sinks: torch.Tensor,
        plan: DSV4ExtendPlan,
        window: torch.Tensor,
        extra: tuple[torch.Tensor, DSV4Pages] | None = None,
        *,
        c4_rows: torch.Tensor | None = None,
        c4_entries: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer's extend: ``window`` is the layer's window pool (64-row pages),
        ``extra`` its ``(pool, DSV4Pages(64))`` c4 or ``(pool, DSV4Pages(2))`` c128 pool;
        c4 layers pass the indexer's selection as ``c4_rows`` (pool rows) and ``c4_entries``
        (per-request entry ids), int32 [T, 512], both from one :func:`dsv4_topk_rows` call."""
        tokens = plan.tokens
        if q.dtype != torch.bfloat16 or q.shape != (tokens, self.num_heads, 512):
            raise ValueError(f"q must be BF16 [{tokens}, {self.num_heads}, 512]")
        if sinks.dtype != torch.float32 or sinks.shape != (self.num_heads,):
            raise ValueError("sinks must be float32 [num_heads]")
        kind = None if extra is None else ("c4" if extra[1].rows == 64 else "c128")
        if (kind == "c4") != (c4_rows is not None and c4_entries is not None):
            raise ValueError("c4 layers (and only they) take c4_rows and c4_entries")
        if out is None:
            out = torch.empty_like(q)
        triton = _dsv4_kernels()[0]
        b = self._extend
        nw, n4, n128 = plan.counts
        if self.provider == "flashinfer-sparse":
            extra_rows = None
            if kind == "c4":
                extra_rows = DSV4Rows(extra[0], extra[1], c4_rows, plan.c4_lengths)
            elif kind == "c128":
                extra_rows = DSV4Rows(extra[0], extra[1], plan.c128, plan.c128_lengths)
            window_rows = DSV4Rows(window, DSV4Pages(64), plan.window, plan.window_lengths)
            self._packed(q, sinks, window_rows, extra_rows, out, b)
            return out
        if kind == "c128":
            width, lengths = _DSV4_WINDOW + plan.c128.shape[1], plan.c128_lengths + _DSV4_WINDOW
        elif kind == "c4":
            width, lengths = _DSV4_WINDOW + 512, plan.c4_lengths + _DSV4_WINDOW
        else:
            width, lengths = _DSV4_WINDOW, plan.window_lengths
        index = b["extra"][:tokens, :width]
        index[:, :_DSV4_WINDOW].copy_(plan.window)
        kv = b["kv"]
        triton.dequant_rows(
            window, plan.window_rows, kv[:nw], rows_per_page=64, page_bytes=DSV4Pages(64).page_bytes
        )
        if kind == "c4":
            triton.dequant_rows(
                extra[0],
                plan.c4_rows,
                kv[nw : nw + n4],
                rows_per_page=64,
                page_bytes=extra[1].page_bytes,
            )
            triton.map_raw(
                c4_entries, plan.query_request, plan.c4_base, index, col0=_DSV4_WINDOW, offset=nw
            )
        elif kind == "c128":
            triton.dequant_rows(
                extra[0],
                plan.c128_rows,
                kv[nw : nw + n128],
                rows_per_page=2,
                page_bytes=extra[1].page_bytes,
            )
            index[:, _DSV4_WINDOW:].copy_(plan.c128)
        flashmla, module = self._flashmla()
        target, heads = out, q.shape[1]
        if heads < flashmla.HEADS:  # tensor-parallel ranks: zero heads with -inf sinks
            b["q_pad"][:tokens, :heads].copy_(q)
            b["sinks_pad"][:heads].copy_(sinks)
            q, sinks, target = b["q_pad"][:tokens], b["sinks_pad"], b["out_pad"][:tokens]
        flashmla.prefill(
            module,
            q,
            kv,
            index,
            lengths,
            sinks,
            self.scale,
            target,
            self.sms,
            stats=b["stats"][:, :tokens],
        )
        if target is not out:
            out.copy_(target[:, :heads])
        return out

    def __call__(
        self,
        q: torch.Tensor,
        sinks: torch.Tensor,
        window: DSV4Rows,
        extra: DSV4Rows | None = None,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode: up to ``max_tokens`` queries, each with its own rows."""
        tokens = q.shape[0]
        if q.dtype != torch.bfloat16 or q.shape[1:] != (self.num_heads, 512):
            raise ValueError(
                f"q must be BF16 [T, {self.num_heads}, 512], got {q.dtype} {tuple(q.shape)}"
            )
        if sinks.dtype != torch.float32 or sinks.shape != (self.num_heads,):
            raise ValueError("sinks must be float32 [num_heads]")
        if tokens > self.max_tokens:
            raise ValueError(f"{tokens} decode queries exceed max_tokens={self.max_tokens}")
        for rows in (window, extra):
            if rows is None:
                continue
            rows.pages.check(rows.cache)
            if rows.indices.dtype != torch.int32 or rows.lengths.dtype != torch.int32:
                raise TypeError("row indices and lengths must be int32")
            if rows.indices.shape[0] != tokens or rows.lengths.shape != (tokens,):
                raise ValueError("row indices/lengths must have one row per query")
        if window.indices.shape[1] > _DSV4_WINDOW:
            raise ValueError(f"at most {_DSV4_WINDOW} window rows per query")
        if out is None:
            out = torch.empty_like(q)
        if extra is not None and extra.indices.shape[1] > self.max_extra:
            raise ValueError(
                f"{extra.indices.shape[1]} extra rows exceed max_extra={self.max_extra}"
            )
        if self.provider == "flashinfer-sparse":
            width = window.indices.shape[1] + (0 if extra is None else extra.indices.shape[1])
            self._packed(
                q, sinks, window, extra, out, self._decode_scratch(tokens, -(-width // 64))
            )
            return out
        self._dense(q, sinks, window, extra, out)
        return out

    def _flashmla(self):
        _, flashinfer, flashmla = _dsv4_kernels()
        if self._module is None:
            self._module = flashmla.load(self.capability, flashinfer.cutlass_includes())
        return flashmla, self._module

    def _dense(self, q, sinks, window, extra, out):
        triton, flashinfer, _ = _dsv4_kernels()
        tokens = q.shape[0]
        table = self.table[: tokens * self.width].view(tokens, self.width)
        seq_lens, topk_lens = self.lengths[0, :tokens], self.lengths[1, :tokens]
        pool = lambda r: (r.cache, r.indices, r.lengths, r.pages.rows, r.pages.page_bytes)
        triton.gather_rows(
            pool(window),
            None if extra is None else pool(extra),
            self.scratch,
            table,
            seq_lens,
            topk_lens,
            swa_slots=_DSV4_WINDOW,
            sms=self.sms,
        )
        flashinfer.dense_decode(
            q, self.scratch, table, seq_lens, topk_lens, sinks, self.scale, out, self.workspace
        )

    def _packed(self, q, sinks, window, extra, out, buffers):
        """sm12x: FlashInfer's packed paged attention over the pages in place, with the
        instance's LSE and split-K scratch (``buffers``: decode's or extend's)."""
        flashinfer = _dsv4_kernels()[1]

        def pool(r):
            view = flashinfer.packed_view(r.cache, r.pages.rows, r.pages.page_bytes)
            return view, r.indices, r.lengths

        tokens, heads = q.shape[:2]
        flashinfer.packed_attention(
            q,
            pool(window),
            None if extra is None else pool(extra),
            sinks,
            self.scale,
            out,
            buffers["lse"][: tokens * heads].view(tokens, heads),
            buffers["mid_out"],
            buffers["mid_lse"],
        )


def dsv4_store_kv(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    table: torch.Tensor,
    positions: torch.Tensor,
    rows: torch.Tensor,
    cache: torch.Tensor,
    pages: DSV4Pages,
) -> None:
    """Write ``kv [T, 512]`` (BF16 or FP32, row stride free) as packed rows: RMS norm with
    ``weight [512]``, rotary at ``positions [T]`` int64, E4M3 + UE8M0 per 64 non-rotary
    channels. ``rows [T]`` int64 are row ids in ``cache``; negative rows are skipped. Also
    stores compressed entries (pass the pooled entries and their window starts)."""
    _dsv4_table(table, positions)
    pages.check(cache)
    if kv.shape[-1] != 512 or kv.stride(-1) != 1 or rows.dtype != torch.int64:
        raise ValueError("kv must be [T, 512] with unit channel stride and rows int64")
    _dsv4_kernels()[0].store_rows(
        kv,
        weight,
        table,
        positions,
        rows,
        cache,
        rows_per_page=pages.rows,
        page_bytes=pages.page_bytes,
        eps=eps,
        norm=True,
    )


def dsv4_prologue(
    q: torch.Tensor,
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    table: torch.Tensor,
    positions: torch.Tensor,
    rows: torch.Tensor,
    cache: torch.Tensor,
    pages: DSV4Pages,
    *,
    q_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """The attention prologue in one launch: each query head of ``q [T, H, 512]`` gets a
    weightless RMS norm and rotary on its last 64 channels (BF16 ``q_out``, default in place;
    head stride free), and each token's ``kv [T, 512]`` row is stored as by
    :func:`dsv4_store_kv` (``rows`` int64, negative skipped)."""
    _dsv4_table(table, positions)
    pages.check(cache)
    if q.shape[-1] != 512 or q.stride(-1) != 1 or kv.shape[-1] != 512 or kv.stride(-1) != 1:
        raise ValueError("q [T, H, 512] and kv [T, 512] need unit channel strides")
    if rows.dtype != torch.int64:
        raise TypeError("rows must be int64")
    q_out = q if q_out is None else q_out
    if q_out.dtype != torch.bfloat16:
        raise TypeError("the query output is BF16")
    _dsv4_kernels()[0].prologue(
        q,
        q_out,
        kv,
        weight,
        table,
        positions,
        rows,
        cache,
        rows_per_page=pages.rows,
        page_bytes=pages.page_bytes,
        eps=eps,
    )
    return q_out


def dsv4_inverse_rope_(
    o: torch.Tensor, table: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Rotate the last 64 channels of the BF16 attention output ``o [T, H, 512]`` back, in
    place (token/head strides free, unit channel stride)."""
    _dsv4_table(table, positions)
    if o.dtype != torch.bfloat16 or o.shape[-1] != 512 or o.stride(-1) != 1:
        raise ValueError("o must be BF16 [T, H, 512] with unit channel stride")
    from .cuda._dsv4_glue import inverse_rope

    return inverse_rope(o, table, positions)


def dsv4_topk_rows(
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    k: int,
    pages: DSV4Pages,
    row_to_batch: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    entries: torch.Tensor | None = None,
) -> torch.Tensor:
    """The top-``k`` of ``scores [T, N]`` (first ``lengths [T]`` int32 valid; ties to the lower
    entry) as row ids through ``page_table [B, pages]`` int32 (row ``i`` uses request
    ``row_to_batch[i]``, default ``i``): ``[T, k]`` int32, -1 past the valid count. Given
    ``entries [T, k]`` int32, the same launch also writes the selected entry ids there (an
    extend's c4 layers take both). The selection and its order are deterministic.

    sm12x: an own CUDA radix select (FlashInfer 0.7.0.post1's radix top-k raises
    cudaErrorNotSupported there). Elsewhere FlashInfer's deterministic top-k."""
    if k != 512:
        raise ValueError("DSV4 selects 512 rows")
    if torch.cuda.get_device_capability(scores.device)[0] == 12:
        from .cuda._dsv4_glue import topk_rows

        if out is None:
            out = torch.empty(scores.shape[0], k, dtype=torch.int32, device=scores.device)
        return topk_rows(
            scores, lengths, page_table, row_to_batch, out, entries, page_rows=pages.rows
        )
    return _dsv4_kernels()[1].topk_rows(
        scores, page_table, lengths, k, pages.rows, out, row_to_batch, entries
    )


def dsv4_page_rows(
    page_table: torch.Tensor,
    positions: torch.Tensor,
    query_request: torch.Tensor,
    *,
    window: tuple[torch.Tensor, torch.Tensor],
    c128: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Decode rows from the page table (one id per 256-token page for every pool), no syncs:
    ``window = (rows [T, 128], lengths [T])`` receives each query's window pool rows
    (positions p, p-1, ..., valid ones first) and ``c128 = (rows [T, C], lengths [T])`` its
    visible c128 entries' pool rows (C columns, -1 past (p+1)//128); int32 outputs."""
    if positions.dtype != torch.int64:
        raise TypeError(f"positions must be int64, not {positions.dtype}")
    triton = _dsv4_kernels()[0]
    rows, lengths = window
    triton.query_rows(
        page_table, positions, query_request, rows, col0=0, width=_DSV4_WINDOW, window=True
    )
    lengths.copy_((positions + 1).clamp_(max=_DSV4_WINDOW))
    if c128 is not None:
        rows, lengths = c128
        triton.query_rows(
            page_table, positions, query_request, rows, col0=0, width=rows.shape[1], window=False
        )
        lengths.copy_((positions + 1) // 128)
