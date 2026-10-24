# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Fixed-bound, graph-safe causal paged prefill for sm_120/sm_121.

CSR contents are caller-owned: nonnegative monotone pointers, valid page IDs,
last-page lengths, and per-request query lengths within the construction bound.
Rows with negative bottom-right positions (Q > KV) produce zero. Metadata can
change on replay. An instance/workspace requires stream-ordered, nonconcurrent
calls, including construction's counter initialization.
"""

import math
import os
import struct
import threading
from functools import lru_cache

import torch

from ..._storage import reject_grad
from .._decode_fp8 import _scale

_compiling = threading.Lock()


def _geometry(batch_size, max_query_per_request, num_qo_heads, num_kv_heads, head_dim):
    values = (batch_size, max_query_per_request, num_qo_heads, num_kv_heads, head_dim)
    if (
        any(type(v) is not int or v < 1 for v in values)
        or num_qo_heads % num_kv_heads
        or head_dim not in (64, 128, 256)
        or batch_size > 65535
        or num_kv_heads > 65535
        or batch_size * max_query_per_request * num_qo_heads > 2**31 - 1
    ):
        raise NotImplementedError(
            "prefill requires positive integer bounds, dividing GQA, D=64/128/256, "
            "B/Hkv <= 65535 and packed row capacity fitting int32"
        )


def check_support(device, *, num_qo_heads, num_kv_heads, head_dim, page_size, dtype):
    _geometry(1, 1, num_qo_heads, num_kv_heads, head_dim)
    if type(page_size) is not int or not 1 <= page_size <= 2**31 - 1:
        raise NotImplementedError("prefill requires a positive int32 page size")
    if dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError("prefill requires BF16 or FP16 Q/K/V")
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise NotImplementedError("prefill requires sm_120 or sm_121")
    try:
        import cutlass.cute  # noqa: F401
        from cutlass.runtime import make_fake_stream  # noqa: F401
    except ImportError as error:
        raise NotImplementedError("prefill requires nvidia-cutlass-dsl 4.8") from error


def _positive(name, default, choices=None):
    key = "CTI_PREFILL_" + name
    try:
        value = int(os.environ.get(key, default))
    except ValueError as error:
        raise ValueError(f"{key} must be a positive int32") from error
    if not 1 <= value <= 2**31 - 1 or (choices is not None and value not in choices):
        raise ValueError(f"{key} must be a positive int32" + (f" in {choices}" if choices else ""))
    return value


def _shared_bytes(dim, block_m, block_n, stages):
    # Q, ring and vector epilogue share an arena. Reserve N page IDs per stage
    # (worst case page=1), plus one last-arrival flag. CuTe reserves another KiB.
    arena = max(block_m * dim * 2, (stages + 1) * block_n * dim * 2)
    if dim == 256:
        # Persistent Q, stages-1 K tiles, and an independent V tile.
        arena = (block_m + stages * block_n) * dim * 2
    return arena + stages * block_n * 4 + 16


def configuration(
    device, *, batch_size, max_query_per_request, num_qo_heads, num_kv_heads, head_dim
):
    _geometry(batch_size, max_query_per_request, num_qo_heads, num_kv_heads, head_dim)
    capability = torch.cuda.get_device_capability(device)
    multiprocessors = torch.cuda.get_device_properties(device).multi_processor_count
    block_m = _positive(
        "BLOCK_M", 64 if head_dim == 256 else 128 if capability == (12, 1) else 64, (64, 128)
    )
    stages = _positive("STAGES", 2, (2, 3))
    config = {
        "block_m": block_m,
        "block_n": _positive(
            "BLOCK_N", 32 if head_dim == 256 and (block_m == 128 or stages == 3) else 64, (32, 64)
        ),
        "stages": stages,
        "min_chunk": _positive("MIN_CHUNK", 256 if head_dim == 256 else 64),
    }
    tiles = (
        max_query_per_request * (num_qo_heads // num_kv_heads) + config["block_m"] - 1
    ) // config["block_m"]
    grid = batch_size * num_kv_heads * tiles
    # A D=256 CTA consumes essentially the entire shared carveout. Target one
    # resident wave; extra splits duplicate Q/output and lengthen the merge.
    target = multiprocessors * (1 if head_dim == 256 else 2)
    config["splits"] = _positive("SPLITS", max(1, (target + grid - 1) // grid))
    if tiles * config["splits"] > 2**31 - 1:
        raise ValueError("prefill tile/split grid exceeds int32")
    if (
        _shared_bytes(head_dim, config["block_m"], config["block_n"], config["stages"]) + 1024
        > 99 * 1024
    ):
        raise ValueError("prefill configuration exceeds 99 KiB shared-memory budget")
    return config


def _workspace_layout(batch, qmax, hq, hk, dim, block_m, splits):
    tiles = (qmax * (hq // hk) + block_m - 1) // block_m
    count = batch * hk * tiles
    # Normalized FP32 partials, log2 LSE, then int32 arrival counters.
    partial_bytes = count * splits * block_m * (dim + 1) * 4 if splits > 1 else 0
    counter_offset = (partial_bytes + 255) // 256 * 256
    size = counter_offset + (count * 4 if splits > 1 else 0)
    return max(256, (size + 255) // 256 * 256), partial_bytes, counter_offset, count


def workspace_size(device, **geometry):
    config = configuration(device, **geometry)
    return _workspace_layout(
        geometry["batch_size"],
        geometry["max_query_per_request"],
        geometry["num_qo_heads"],
        geometry["num_kv_heads"],
        geometry["head_dim"],
        config["block_m"],
        config["splits"],
    )[0]


@lru_cache(maxsize=None)
def _compile(
    batch,
    qmax,
    heads,
    group,
    dim,
    page,
    dtype,
    arch,
    block_m,
    block_n,
    stages,
    splits,
    min_chunk,
    kv_dense=True,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._kernel import Prefill

    element = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    fake = cute.runtime.make_fake_tensor
    q = fake(
        element,
        (cute.sym_int(), heads * group, dim),
        (cute.sym_int(64, divisibility=8), cute.sym_int(64, divisibility=8), 1),
        assumed_align=16,
    )
    out = fake(
        element,
        (cute.sym_int(), heads * group, dim),
        (heads * group * dim, dim, 1),
        assumed_align=16,
    )
    # K and V take separate fakes: one fake would tie their runtime KV strides.
    cache = lambda: fake(
        element,
        (cute.sym_int(), page, heads, dim),
        (
            cute.sym_int(64, divisibility=8),
            heads * dim if kv_dense else cute.sym_int(64, divisibility=8),
            dim if kv_dense else cute.sym_int(64, divisibility=8),
            1,
        ),
        assumed_align=16,
    )
    vector = lambda dt: fake(dt, (cute.sym_int(),), (1,), assumed_align=4)
    op = Prefill(
        batch,
        qmax,
        heads,
        group,
        dim,
        page,
        element,
        block_m,
        block_n,
        stages,
        splits,
        min_chunk,
        kv_dense,
    )
    return cute.compile(
        op,
        q,
        cache(),
        cache(),
        vector(cutlass.Int32),
        vector(cutlass.Int32),
        vector(cutlass.Int32),
        vector(cutlass.Int32),
        out,
        vector(cutlass.Float32),
        vector(cutlass.Int32),
        cutlass.Float32(1),
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


class CutePagedPrefill:
    """Construction compiles and initializes workspace; run validates and launches.

    KV accepts NHD views with unit inner stride and 16-byte aligned base and
    page/token/head strides, including interleaved heads and packed K/V pages.
    K and V may have different strides.
    """

    def __init__(
        self,
        workspace,
        *,
        batch_size,
        max_query_per_request,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        device,
        sm_scale=None,
        q_row_stride=None,
    ):
        check_support(
            device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
        )
        self.config = configuration(
            device,
            batch_size=batch_size,
            max_query_per_request=max_query_per_request,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch, self.qmax = batch_size, max_query_per_request
        self.hq, self.hk, self.dim = num_qo_heads, num_kv_heads, head_dim
        self.page, self.dtype = page_size, dtype
        self.q_row_stride = num_qo_heads * head_dim if q_row_stride is None else q_row_stride
        if (
            type(self.q_row_stride) is not int
            or self.q_row_stride < num_qo_heads * head_dim
            or self.q_row_stride % 8
        ):
            raise ValueError("q_row_stride must fit all heads and be a multiple of 8 elements")
        scale = head_dim**-0.5 if sm_scale is None else sm_scale
        if (
            type(scale) not in (int, float)
            or not math.isfinite(scale)
            or abs(scale) > 3.402823466e38
        ):
            raise ValueError("sm_scale must be a finite host float representable in FP32")
        self.sm_scale = struct.unpack("f", struct.pack("f", float(scale)))[0]
        needed, payload, counters, count = _workspace_layout(
            batch_size,
            max_query_per_request,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            self.config["block_m"],
            self.config["splits"],
        )
        if (
            workspace.device != self.device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < needed
            or workspace.data_ptr() % 256
        ):
            raise ValueError(
                f"workspace must be a contiguous 256-byte aligned uint8 vector of >= {needed} bytes"
            )
        self._workspace = workspace
        self._work = workspace[: max(4, payload)].view(torch.float32)
        self._counters = workspace[
            counters : counters + (count * 4 if self.config["splits"] > 1 else 4)
        ].view(torch.int32)
        self._counters.zero_()
        major, minor = torch.cuda.get_device_capability(self.device)
        with _compiling, torch.cuda.device(self.device):
            self._kernel, self._strided_kernel = (
                _compile(
                    batch_size,
                    max_query_per_request,
                    num_kv_heads,
                    num_qo_heads // num_kv_heads,
                    head_dim,
                    page_size,
                    dtype,
                    f"sm_{major}{minor}a",
                    **self.config,
                    kv_dense=dense,
                )
                for dense in (True, False)
            )

    def run(
        self,
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        *,
        k_scale=1.0,
        v_scale=1.0,
    ):
        tensors = (q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, out)
        reject_grad(*tensors)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all prefill tensors must be on the construction device")
        if (
            q.ndim != 3
            or q.shape[1:] != (self.hq, self.dim)
            or out.shape != q.shape
            or q.shape[0] > self.batch * self.qmax
            or q.dtype != self.dtype
            or out.dtype != self.dtype
        ):
            raise ValueError("q/out must match [T,Hq,D], dtype and construction capacity")
        if (
            q.stride(2) != 1
            or q.stride(0) % 8
            or q.stride(1) % 8
            or not out.is_contiguous()
            or q.data_ptr() % 16
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "q requires 8-element row/head strides; out contiguous; both 16-byte aligned"
            )
        for cache in (k_cache, v_cache):
            if (
                cache.ndim != 4
                or cache.shape[1:] != (self.page, self.hk, self.dim)
                or cache.dtype != self.dtype
                or cache.stride(3) != 1
                or any(stride % 8 for stride in cache.stride()[:3])
                or cache.data_ptr() % 16
            ):
                raise ValueError(
                    "KV requires construction dtype, unit inner stride, 16-byte aligned base and page/token/head strides"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("K/V cache shapes must match")
        for tensor, size in (
            (qo_indptr, self.batch + 1),
            (kv_indptr, self.batch + 1),
            (kv_indices, None),
            (kv_last_page_len, self.batch),
        ):
            if (
                tensor.ndim != 1
                or tensor.dtype != torch.int32
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 4
                or (size is not None and tensor.numel() != size)
            ):
                raise ValueError(
                    "CSR metadata requires contiguous aligned int32 vectors of fixed batch size"
                )
        k_scale, v_scale = _scale(k_scale, "k_scale"), _scale(v_scale, "v_scale")
        scale = _scale(self.sm_scale * k_scale, "sm_scale*k_scale")
        dense_strides = (self.hk * self.dim, self.dim, 1)
        kernel = (
            self._kernel
            if k_cache.stride()[1:] == v_cache.stride()[1:] == dense_strides
            else self._strided_kernel
        )
        with torch.cuda.device(self.device):
            kernel(*tensors, self._work, self._counters, scale, v_scale)
        return out
