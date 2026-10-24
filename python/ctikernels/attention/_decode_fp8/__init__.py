# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FP8 NHD paged decode with device-selected splits and construction-only compilation.

CSR contents must describe valid pages (int32, nonnegative lengths, in-bounds
indices). Like FlashInfer, checking those contents is the caller's responsibility.
An instance and its workspace must not be used concurrently on different streams.
Construction and the first use must also be ordered on the same stream, or by an
explicit stream dependency, because construction initializes the arrival counters.
"""

import math
import os
import threading
from functools import lru_cache

import torch

from ..._storage import reject_grad
from .._draft.cute import _cooperative

_compiling = threading.Lock()


def _geometry(batch_size, num_qo_heads, num_kv_heads, head_dim):
    if (
        any(type(v) is not int or v < 1 for v in (batch_size, num_qo_heads, num_kv_heads))
        or num_qo_heads % num_kv_heads
        or num_qo_heads // num_kv_heads > 16
        or type(head_dim) is not int
        or head_dim not in (64, 128, 256)
    ):
        raise NotImplementedError(
            "FP8 decode requires B,Hkv >= 1, dividing GQA <= 16, D=64/128/256"
        )
    if batch_size > 65535 or num_kv_heads > 65535:
        raise NotImplementedError("FP8 decode grid-y/grid-z exceeds 65535")


def check_support(device, *, num_qo_heads, num_kv_heads, head_dim, page_size, q_dtype, kv_dtype):
    _geometry(1, num_qo_heads, num_kv_heads, head_dim)
    if type(page_size) is not int or page_size < 1:
        raise NotImplementedError("FP8 decode requires a positive integer page size")
    if q_dtype not in (torch.bfloat16, torch.float16) or kv_dtype != torch.float8_e4m3fn:
        raise NotImplementedError("FP8 decode requires BF16/FP16 Q and e4m3fn KV")
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise NotImplementedError("FP8 decode requires sm_120 or sm_121")
    try:
        import cutlass.cute  # noqa: F401
        from cutlass.runtime import make_fake_stream  # noqa: F401
    except ImportError as error:
        raise NotImplementedError("FP8 decode requires nvidia-cutlass-dsl 4.8") from error


def _positive(name, default, choices=None):
    name = f"CTI_FP8DEC_{name}"
    try:
        raw = os.environ.get(name, default)
        if type(raw) not in (str, int):
            raise ValueError
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1 or (choices is not None and value not in choices):
        raise ValueError(f"{name} must be positive" + (f" and one of {choices}" if choices else ""))
    return value


def _split_geometry(length, minimum, splits):
    """CPU mirror of the kernel's splits; an empty sequence has zero active splits."""
    count = min(max((length + minimum - 1) // minimum, 1), splits)
    # Align to one 16-key MMA block, not the whole CTA stage. A nonzero
    # chunk also makes the empty-sequence calculation safe.
    chunk = (max((length + count - 1) // count, 1) + 15) // 16 * 16
    count = (length + chunk - 1) // chunk
    return chunk, count


def _warp_bytes(head_dim, keys_per_warp, stages):
    # Every warp owns its ring AND reduction storage. No cross-warp barrier
    # is needed when reusing the ring after its last ldmatrix.
    ring = 2 * stages * keys_per_warp * max(128, head_dim)
    combine = 16 * (head_dim + 1) * 4
    return (max(ring, combine) + 127) // 128 * 128


def _metadata_bytes(warps, keys_per_warp, page_size=None):
    # Two int32 banks; retain the conservative bound when page size is unknown.
    keys = warps * keys_per_warp
    page_cap = keys + 1 if page_size is None else (keys + page_size - 2) // page_size + 1
    return 8 * max(128, page_cap)


def _shared_bytes(head_dim, warps, keys_per_warp, stages, splits=1, group=1, page_size=None):
    pages = _metadata_bytes(warps, keys_per_warp, page_size)
    return max(warps * _warp_bytes(head_dim, keys_per_warp, stages), splits * group * 4) + pages


def configuration(
    device,
    *,
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size=None,
    splits=None,
    min_chunk=None,
    warps=None,
    keys_per_warp=None,
    stages=None,
    merge=None,
):
    """Resolve defaults, caller overrides, then CTI_FP8DEC_* screen overrides."""
    _geometry(batch_size, num_qo_heads, num_kv_heads, head_dim)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    config = {
        "splits": _positive(
            "SPLITS", max(1, sms // (batch_size * num_kv_heads)) if splits is None else splits
        ),
        "min_chunk": _positive("MIN_CHUNK", 256 if min_chunk is None else min_chunk),
        "warps": _positive("WARPS", 4 if warps is None else warps, (1, 2, 4, 8)),
        "keys_per_warp": _positive(
            "KEYS", 16 if keys_per_warp is None else keys_per_warp, (16, 32)
        ),
        "stages": _positive("STAGES", 3 if stages is None else stages, (2, 3, 4)),
        "merge": os.environ.get("CTI_FP8DEC_MERGE", "distributed" if merge is None else merge),
    }
    if config["merge"] not in ("distributed", "last"):
        raise ValueError("CTI_FP8DEC_MERGE must be distributed or last")
    if config["splits"] > 2**31 - 1 or config["min_chunk"] > 2**31 - 1:
        raise ValueError("FP8 decode splits and min_chunk must fit positive int32")
    if (
        _shared_bytes(
            head_dim,
            config["warps"],
            config["keys_per_warp"],
            config["stages"],
            page_size=page_size,
        )
        + 1024
        > 99 * 1024
    ):
        raise ValueError("FP8 decode configuration exceeds 99 KiB shared-memory budget")
    return config


def _workspace_bytes(batch_size, num_qo_heads, head_dim, splits):
    # Normalized FP32 O followed by log2 LSE; counters are separately owned.
    size = batch_size * num_qo_heads * splits * (head_dim + 1) * 4 if splits > 1 else 0
    return max(256, (size + 255) // 256 * 256)


def workspace_size(device, *, batch_size, num_qo_heads, num_kv_heads, head_dim, **overrides):
    config = configuration(
        device,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        **overrides,
    )
    return _workspace_bytes(batch_size, num_qo_heads, head_dim, config["splits"])


@lru_cache(maxsize=None)
def _compile(
    batch,
    heads,
    group,
    dim,
    page,
    dtype,
    arch,
    splits,
    min_chunk,
    warps,
    keys_per_warp,
    stages,
    merge="distributed",
    kv_dense=True,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._kernel import Decode

    fake = cute.runtime.make_fake_tensor
    element = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    q = fake(
        element,
        (batch, heads * group, dim),
        (cute.sym_int(64, divisibility=8), dim, 1),
        assumed_align=16,
    )
    out = fake(
        element, (batch, heads * group, dim), (heads * group * dim, dim, 1), assumed_align=16
    )
    # K and V take separate fakes: one fake would tie their runtime KV strides.
    cache = lambda: fake(
        cutlass.Float8E4M3FN,
        (cute.sym_int(), page, heads, dim),
        (
            page * heads * dim if kv_dense else cute.sym_int(64, divisibility=16),
            heads * dim if kv_dense else cute.sym_int(64, divisibility=16),
            dim if kv_dense else cute.sym_int(64, divisibility=16),
            1,
        ),
        assumed_align=16,
    )
    vector = lambda dtype: fake(dtype, (cute.sym_int(),), (1,), assumed_align=4)
    op = Decode(
        batch,
        heads,
        group,
        dim,
        page,
        element,
        splits,
        min_chunk,
        warps,
        keys_per_warp,
        stages,
        merge,
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
        out,
        vector(cutlass.Float32),
        vector(cutlass.Int32),
        cutlass.Float32(1),
        cutlass.Float32(1),
        cutlass.Float32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )


def _compile_resident(device, batch, heads, group, dim, page, dtype, arch, config):
    """Clamp before compiling, then verify the actual binary's occupancy.

    The allocator reserves 1 KiB. A smaller recompiled grid is queried again:
    register allocation can change with the static split bound. Single-split
    kernels never wait for another CTA and need no cooperative launch.
    """
    distributed = config["merge"] == "distributed"
    props = torch.cuda.get_device_properties(device)
    sms = props.multi_processor_count
    threads = config["warps"] * 32

    def shared():
        return (
            _shared_bytes(
                dim,
                config["warps"],
                config["keys_per_warp"],
                config["stages"],
                config["splits"] if distributed else 1,
                group,
                page_size=page,
            )
            + 1024
        )

    if distributed:
        # Resource-only upper bound with a 2 KiB reserve per SM; the driver
        # query below verifies registers, allocation rounding, and reservations.
        per_sm = min(
            (getattr(props, "shared_memory_per_multiprocessor", 100 * 1024) - 2048)
            // _shared_bytes(
                dim, config["warps"], config["keys_per_warp"], config["stages"], page_size=page
            ),
            getattr(props, "max_threads_per_multi_processor", 1536) // threads,
        )
        config["splits"] = min(
            config["splits"],
            max(1, sms * per_sm // (batch * heads)),
            (99 * 1024 - 1024 - _metadata_bytes(config["warps"], config["keys_per_warp"], page))
            // (group * 4),
        )
        if config["splits"] > 1 and not _cooperative.supported(device):
            raise NotImplementedError(
                "FP8 distributed merge requires cooperative launch; use CTI_FP8DEC_MERGE=last"
            )
    while True:
        # Both ABIs are capture-ready and share one safe split/workspace layout.
        kernels = tuple(
            _compile(batch, heads, group, dim, page, dtype, arch, **config, kv_dense=dense)
            for dense in (True, False)
        )
        if not distributed or config["splits"] == 1:
            return kernels, None
        resident = min(
            _cooperative.blocks_per_sm(kernel, device, threads, shared()) for kernel in kernels
        )
        if resident < 1:
            raise ValueError("FP8 decode kernel has no resident CTA on this device")
        limit = max(1, sms * resident // (batch * heads))
        if config["splits"] <= limit:
            return kernels, resident
        config["splits"] = limit


def _scale(value, name):
    # Do not permit Tensor.__float__: it could synchronize a CUDA tensor in run.
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3.402823466e38:
        raise ValueError(f"{name} must be a finite host float representable in FP32")
    return float(value)


class FP8PagedDecode:
    """Fixed geometry, runtime CSR lengths; no device allocation or sync in run.

    KV accepts NHD views with unit inner stride and 16-byte aligned base and
    page/token/head strides, including interleaved heads and packed K/V pages.
    K and V may have different strides.

    Scales are host floats passed as runtime Float32 arguments; graph capture
    bakes their values. Metadata contents may change between graph replays.
    One instance/workspace must have stream-ordered, nonconcurrent calls.
    Optional configuration keywords override defaults; CTI_FP8DEC_* environment
    values take precedence. Size the workspace with the same overrides.
    """

    def __init__(
        self,
        workspace,
        *,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_dtype,
        device,
        sm_scale=None,
        splits=None,
        min_chunk=None,
        warps=None,
        keys_per_warp=None,
        stages=None,
        merge=None,
    ):
        check_support(
            device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_dtype=q_dtype,
            kv_dtype=torch.float8_e4m3fn,
        )
        self.config = configuration(
            device,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            splits=splits,
            min_chunk=min_chunk,
            warps=warps,
            keys_per_warp=keys_per_warp,
            stages=stages,
            merge=merge,
        )
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.batch, self.hq, self.hk, self.dim = batch_size, num_qo_heads, num_kv_heads, head_dim
        self.page, self.dtype = page_size, q_dtype
        self.sm_scale = _scale(head_dim**-0.5 if sm_scale is None else sm_scale, "sm_scale")
        needed = _workspace_bytes(batch_size, num_qo_heads, head_dim, self.config["splits"])
        if (
            workspace.device != self.device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < needed
            or workspace.data_ptr() % 256
        ):
            raise ValueError(
                "workspace must be contiguous CUDA uint8, large enough, 256-byte aligned"
            )
        self._work = workspace[:needed].view(torch.float32)
        major, minor = torch.cuda.get_device_capability(self.device)
        with _compiling:
            kernels, self.cooperative_blocks_per_sm = _compile_resident(
                self.device,
                batch_size,
                num_kv_heads,
                num_qo_heads // num_kv_heads,
                head_dim,
                page_size,
                q_dtype,
                f"sm_{major}{minor}a",
                self.config,
            )
        self._kernel, self._strided_kernel = kernels
        self._counters = torch.zeros(
            batch_size * num_kv_heads * (2 if self.config["merge"] == "distributed" else 1),
            dtype=torch.int32,
            device=self.device,
        )
        # Generation flags advance; only arrival counters return to zero.
        self._arrivals = (
            self._counters[::2] if self.config["merge"] == "distributed" else self._counters
        )

    def run(
        self,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        *,
        k_scale=1.0,
        v_scale=1.0,
    ):
        tensors = (q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, out)
        reject_grad(*tensors)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all decode tensors must be on the construction device")
        if q.shape != (self.batch, self.hq, self.dim) or out.shape != q.shape:
            raise ValueError("q/out shapes must match fixed construction [B,Hq,D]")
        if q.dtype != self.dtype or out.dtype != self.dtype:
            raise ValueError("q/out dtypes must match construction dtype")
        if (
            q.stride(2) != 1
            or q.stride(1) != self.dim
            or q.stride(0) % 8
            or not out.is_contiguous()
            or q.data_ptr() % 16
            or out.data_ptr() % 16
        ):
            raise ValueError(
                "q requires dense heads, 8-element row strides and 16-byte alignment; out contiguous"
            )
        for cache in (k_cache, v_cache):
            if (
                cache.ndim != 4
                or cache.shape[1:] != (self.page, self.hk, self.dim)
                or cache.dtype != torch.float8_e4m3fn
                or cache.stride(3) != 1
                or any(stride % 16 for stride in cache.stride()[:3])
                or cache.data_ptr() % 16
            ):
                raise ValueError(
                    "KV requires e4m3fn, unit inner stride, 16-byte aligned base and page/token/head strides"
                )
        if k_cache.shape != v_cache.shape:
            raise ValueError("K/V cache shapes must match")
        for tensor, size in (
            (kv_indptr, self.batch + 1),
            (kv_last_page_len, self.batch),
            (kv_indices, None),
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
        # TVM-FFI borrows Torch/DLPack storage and the current device stream,
        # matching the DRAFT provider; no from_dlpack wrappers allocate in run.
        dense_strides = (self.page * self.hk * self.dim, self.hk * self.dim, self.dim, 1)
        kernel = (
            self._kernel
            if k_cache.stride() == v_cache.stride() == dense_strides
            else self._strided_kernel
        )
        with torch.cuda.device(self.device):
            kernel(*tensors[:6], out, self._work, self._counters, self.sm_scale, k_scale, v_scale)
        return out
