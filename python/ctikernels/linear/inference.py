# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen BF16 inference: Linear()(x, weight, out=None, *, out_dtype=None, gate=None, bias=None, workspace=None).

x may be [..., K] (leading dims flatten into M) and bias [N], as F.linear accepts;
calls with bias run the cuBLAS bias epilogue (the market call).

FP32 output uses out_dtype=torch.float32 or an FP32 out tensor. Its separate
TUNED_F32 table falls back to torch.mm(out_dtype=torch.float32), without autotuning.
The default output remains x.dtype. Gate is unsupported for FP32 output.

Optional gate is a frozen BF16 [M,K] view with unit last stride. It applies
x * sigmoid(gate) in FP32, rounding once to BF16 before the matmul. Fused CuTe
requires row stride >= K divisible by eight and a 16-byte aligned gate pointer;
otherwise one Triton gating launch (the fused prologue's instructions, so both routes give the
same bits; within one BF16 ULP of PyTorch's FP32 ops, which still run on
the CPU and under activation autograd) precedes normal plain dispatch. ``gate_fused``
queries shape/device admission without compiling; callers still own layout checks.

Any M is supported. This repository owns dispatch: measured (hardware tuple, M
bucket, N, K) rows select CuTe; other rows run cuBLAS. Inside an explicit
``ctikernels.linear.autotune`` context (or with CTI_LINEAR_AUTOTUNE=cache) table
holes may use cached or tuned decisions. Every M in one bucket
(1, 2, 3-4, 5-8, then eight-row buckets to 64) runs the same provider and
configuration, so CuTe rows are bitwise identical across M in a bucket,
including M padded to the bucket edge.

CuTe accepts contiguous BF16 [M,K] and [N,K], M=1..64, K divisible by 8,
16-byte aligned pointers on allowlisted Blackwell devices. Other contracts use
cuBLAS. Eight-row M buckets must fit 99 KiB shared memory; auto falls back
to cuBLAS on overflow, while forced CuTe raises. Trainable weights always raise before device inspection. Activation
training uses cuBLAS. out must have matching shape/dtype/device and not overlap
either input; it may be a row-strided [M,N] view (unit last stride, row stride
>= N), e.g. a column slice of a wider buffer that several calls fill. CuTe
writes it in place when the row stride is a multiple of 16 bytes and the pointer
is 16-byte aligned (each stride compiles its own kernel during eager warm-up);
other strides run cuBLAS, which also writes in place. Buffers are borrowed;
callers serialize writes.

Split-K scratch: ``workspace_bytes(n, k, max_m)`` bounds the FP32 scratch one
call needs on this device; nonzero ``gate_stride`` also covers the gated plans,
whose unfused fallback runs the plain plan. Rows whose Config has ``cluster``
reduce their splits inside one launch (thread-block cluster, fixed split order);
DSMEM rows need no scratch, L2 rows use the same scratch as two-launch rows.
Callers may pass ``workspace`` (contiguous CUDA
tensor of at least that many bytes on the input device, 16-byte aligned, not
sharing storage with x, weight, gate or out) and then own its lifetime and stream ordering. Otherwise eager calls share one
scratch buffer per (device, stream) that only grows to the largest call seen
(``scratch_bytes()``), so memory is flat after warm-up; captured calls without
a workspace allocate from the graph's own pool. ``release_scratch()`` frees that eager
cache, e.g. once the caller owns workspaces; it never affects captured graphs.

provider, kernels_per_call and launch_config describe the last successful call
(initially "cublas", 1, None); launch_config names every code-selecting CuTe
parameter (Config fields, M bucket, CTA threads) and is None for cuBLAS. Construct one operation per engine call site. Warm eagerly before
capture: a CuTe plan not compiled before capture raises instead of switching
provider. An optional Config or CTI_LINEAR_CONFIG=rows,tile_k,stages,splits[,warps,variant,prefetch,cluster] supports forced tuning.
An explicit constructor provider=auto|cute|cublas takes precedence over the
provider environment setting, without mutating process-wide policy.
CTI_LINEAR_PROVIDER=cute forces shape admission but never bypasses layout,
hardware or dependency checks; missing/incompatible DSL raises for a forced call.
CTI_LINEAR_AUTOTUNE=0|context|cache (or autotune=; default context) never reads the
autotune cache, reads it only inside autotune(), or always reads it; forced providers
ignore it. Environment settings are read once, at construction.
``decision(n, k, m)`` reports provider, config and source without compiling.
``warm_up(weight, max_m)`` compiles the table's CuTe routes for M = 1..max_m without
timing or tuning (call it before capture; it returns the routes compiled).
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from copy import deepcopy
from dataclasses import asdict

import torch

from .._runtime.launch import prime
from .._runtime.workspace import GraphWorkspace
from . import _autotune, _dispatch
from ._dispatch import (
    M_BUCKETS,
    Config,
    m_bucket,
    parse_config,
    tuned_config,
    tuned_gated_config,
)
from .cublas import linear, linear_fp32

VARIABLE = "CTI_LINEAR_PROVIDER"
MAX_M = 64  # Largest M with CuTe routes; larger M always runs cuBLAS.
# Opt-in dynamic shared memory on sm120/sm121 (99 KiB).
_SMEM_LIMIT = 99 * 1024


# Largest thread-block cluster per capability: sm_120 opts into the non-portable 16,
# sm_121 rejects more than the portable 8 at launch (cudaErrorInvalidClusterSize).
_MAX_CLUSTER = {(12, 0): 16, (12, 1): 8}


def _cluster_fits(config, capability):
    return not config.cluster or config.splits <= _MAX_CLUSTER.get(tuple(capability), 8)


def _admits(config, mtiles, gate_stride, capability):
    """A route can launch ``config``: shared memory fits, the cluster size is legal on
    this capability, and a fused gate needs a variant with the gated prologue (``wide``
    has none; such calls run unfused)."""
    if gate_stride and config.variant == "wide":
        return False
    if not _cluster_fits(config, capability):
        return False
    return config.shared_bytes_for(mtiles, gated=bool(gate_stride)) <= _SMEM_LIMIT


# (device index, stream handle) -> flat FP32 eager split-K scratch.
_SCRATCH: dict[tuple[int, int], torch.Tensor] = {}
# Same key -> (that slot's buffer, {(splits, M, N): view of it}).
_VIEWS: dict[tuple[int, int], tuple] = {}
# Guards _SCRATCH/_VIEWS against concurrent release; uncontended cost is ~0.1 us.
_SCRATCH_LOCK = threading.Lock()
_AUTOTUNE_COMPILE_FAILURES = set()
_FAILED = "compile failed in this process"


def _override():
    value = os.environ.get(VARIABLE, "auto")
    if value not in ("auto", "cute", "cublas"):
        raise ValueError(f"{VARIABLE} must be auto, cute or cublas")
    return value


def _overlaps(a, b):
    # Conservative storage-level check also covers strided fallback inputs.
    # Independent empty storages both have null pointers, but do not overlap.
    pointer = a.untyped_storage().data_ptr()
    return pointer != 0 and pointer == b.untyped_storage().data_ptr()


def _gate_routes(k, gate_stride):
    """Gate strides of the plans a call can launch (0 = the plain plan).

    A fusable gate layout tries the fused plan first; every gated call can fall back
    to the unfused path, which runs the plain plan.
    """
    if gate_stride and gate_stride >= k and gate_stride % 8 == 0:
        return (gate_stride, 0)
    return (0,)


def _output_dtype(out_dtype, gate=False):
    if out_dtype not in (None, torch.bfloat16, torch.float32):
        raise ValueError("out_dtype must be None, torch.bfloat16 or torch.float32")
    if gate and out_dtype == torch.float32:
        raise ValueError("gate is unsupported with FP32 output")
    return out_dtype


def _capture_miss(n, k, edge, gate_stride=0, out_stride=0):
    gate = f", gate stride {gate_stride}" if gate_stride else ""
    gate += f", out row stride {out_stride}" if out_stride else ""
    return RuntimeError(
        f"Linear CuTe plan for N={n}, K={k}, M bucket edge {edge}{gate} is not compiled; "
        "call the op eagerly with an M in that bucket before CUDA graph capture"
    )


def _dtype_key(key, out_dtype):
    return (*key, torch.float32) if out_dtype == torch.float32 else key


def _scratch_index(device):
    if device is None or type(device) is int:
        return device
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("device must be a CUDA device")
    return torch.cuda.current_device() if device.index is None else device.index


def scratch_bytes(device=None) -> int:
    """Bytes held by the shared eager split-K scratch on one device (default: all)."""
    index = _scratch_index(device)
    with _SCRATCH_LOCK:
        return sum(b.numel() * 4 for (i, _), b in _SCRATCH.items() if index in (None, i))


def release_scratch(device=None) -> None:
    """Drop the shared eager split-K scratch on one device (default: all).

    Safe between and after captures: captured calls allocate from the graph's pool,
    never this cache, so no captured graph references it. Pending eager launches stay
    valid through the allocator's stream order; torch.cuda.empty_cache() returns the memory.
    """
    index = _scratch_index(device)
    with _SCRATCH_LOCK:
        for slot in [s for s in _SCRATCH if index in (None, s[0])]:
            del _SCRATCH[slot]
            _VIEWS.pop(slot, None)


def release_cublas_workspaces() -> None:
    """Free every stream's cuBLAS workspace (warm-up and tuning streams included).

    Call only at a quiescent point: no other thread or stream is running cuBLAS work
    (e.g. after warm-up/capture, before serving). Later calls recreate them lazily.
    """
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is None:
        raise RuntimeError(f"torch {torch.__version__} cannot clear cuBLAS workspaces")
    clear()


class _Plan:
    """Resolved CuTe launch for one (device, N, K, M bucket); cuBLAS plans are None."""

    __slots__ = ("compiled", "splits", "kernels", "launch", "_build", "_strided")

    def __init__(self, compiled, config, mtiles, build=None):
        self.compiled = compiled
        self.splits = config.scratch_splits
        self.kernels = config.kernels
        self.launch = {"config": asdict(config), "mtiles": mtiles, "cta_threads": 32 * config.warps}
        # build(out_stride, capturing) compiles the same launch for a row-strided output.
        self._build = build
        self._strided = {}

    def kernel(self, out_stride):
        """Compiled launch for output row stride ``out_stride`` (0: contiguous), or None
        when that stride was not compiled before capture."""
        if not out_stride:
            return self.compiled
        compiled = self._strided.get(out_stride)
        if compiled is None:
            compiled = self._build(out_stride, torch.cuda.is_current_stream_capturing())
            if compiled is not None:
                self._strided[out_stride] = compiled
        return compiled


class _BucketRun:
    """One warm-up call of ``op`` at ``m`` rows on zeroed scratch, allocated at the first."""

    def __init__(self, op, weight, rows, gate_stride, out_dtype):
        self._op, self._weight, self._rows = op, weight, rows
        self._gate_stride, self._out_dtype = gate_stride, out_dtype
        self._scratch = None

    def __call__(self, m, stride):
        weight, (n, k) = self._weight, self._weight.shape
        if self._scratch is None:
            device, dtype, gated = weight.device, weight.dtype, min(self._rows, MAX_M)
            words = self._op.workspace_bytes(
                n, k, gated, device, self._gate_stride, out_dtype=self._out_dtype
            )
            self._scratch = (
                torch.zeros(self._rows, k, dtype=dtype, device=device),
                torch.zeros(gated * max(self._gate_stride, k), dtype=dtype, device=device),
                torch.empty(words // 4 + 4, dtype=torch.float32, device=device),
            )
        x, gates, workspace = self._scratch
        gate = gates[: m * stride].view(m, stride)[:, :k] if stride else None
        self._op(x[:m], weight, out_dtype=self._out_dtype, gate=gate, workspace=workspace)


class Linear:
    def __init__(
        self,
        *,
        config: Config | None = None,
        provider: str | None = None,
        autotune: str | None = None,
    ):
        if config is not None and not isinstance(config, Config):
            raise TypeError("config must be a Config")
        if provider not in (None, "auto", "cute", "cublas"):
            raise ValueError("provider must be auto, cute or cublas")
        self._autotune_mode = _autotune.mode(autotune)
        self.config = config
        self._override = provider if provider is not None else _override()
        self._forced = config
        if self._override == "cute" and config is None:
            setting = os.environ.get("CTI_LINEAR_CONFIG")
            # Forced testing only; auto uses table/cache decisions.
            self._forced = parse_config(setting) if setting else Config()
        self._provider = "cublas"
        self._kernels_per_call = 1
        self._launch = None
        # (device index, N, K, bucket edge[, gate stride]) -> _Plan or fallback None.
        self._plans = {}
        # (device index, N, K, bucket edge, gate stride) -> (config or None, source,
        # record) of each retained plan; queries reuse it even after a custom-cache
        # context exits, so they always describe the plan that launches.
        self._pinned = {}
        # Last caller-owned workspace view (storage, key, view).
        self._views = {}

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def kernels_per_call(self) -> int:
        return self._kernels_per_call

    @property
    def launch_config(self) -> dict | None:
        return (
            None
            if self._launch is None
            else dict(self._launch, config=dict(self._launch["config"]))
        )

    def workspace_bytes(
        self,
        n: int,
        k: int,
        max_m: int = MAX_M,
        device=None,
        gate_stride: int = 0,
        *,
        out_dtype=None,
    ) -> int:
        """Split-K bound over every plan a call can launch (fused gated, unfused fallback to
        the plain plan for any gate layout, cached autotuned rows); queries never tune or
        compile."""
        _output_dtype(out_dtype, gate_stride)
        if self._override == "cublas" or n <= 0 or k <= 0 or k % 8:
            return 0
        if not 1 <= max_m:
            raise ValueError("max_m must be positive")
        if not torch.cuda.is_available():
            return 0
        device = torch.device("cuda") if device is None else torch.device(device)
        if device.type != "cuda":
            return 0
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        kind, capability, _sms = _dispatch._hardware(device)
        if capability not in ((12, 0), (12, 1)):
            return 0
        need = 0
        for lo, hi in M_BUCKETS:
            if lo > max_m:
                break
            # The same routes the call takes: fused (if the layout admits it) and plain.
            for stride in _gate_routes(k, gate_stride):
                config = self._config(kind, n, k, hi, stride, device, out_dtype=out_dtype)
                if config is not None and config.scratch_splits:
                    need = max(need, config.scratch_splits * min(hi, max_m) * n * 4)
        return need

    def warm_up(self, weight, max_m: int = MAX_M, *, gate_stride: int = 0, out_dtype=None):
        """Compile, without timing or tuning, every CuTe route the shipped table selects
        for this weight at M = 1..max_m (once per M bucket), including the plain plan a
        gated call falls back to, and run each bucket once on zeroed scratch, cuBLAS holes
        and other devices included, plus cuBLAS at a sqrt(2) ladder of M up to ``max_m``, so
        serving calls find their kernels loaded (``_runtime.launch``); returns the CuTe routes
        compiled. Call eagerly before capture; row-strided outputs still compile per stride
        on first use."""
        _output_dtype(out_dtype, gate_stride)
        if weight.requires_grad:
            raise ValueError("Linear requires frozen weight (requires_grad=False)")
        if max_m < 1:
            raise ValueError("max_m must be positive")
        n, k = weight.shape
        if not weight.is_cuda:
            return []
        device = weight.device
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Linear.warm_up must run eagerly, outside CUDA graph capture")
        kind, capability, _sms = _dispatch._hardware(device)
        cute = (
            self._override != "cublas"
            and weight.dtype == torch.bfloat16
            and not k % 8
            and capability in ((12, 0), (12, 1))
        )
        routes = []
        run = _BucketRun(self, weight, max_m, gate_stride, out_dtype)
        for lo, edge in M_BUCKETS:
            if lo > max_m:
                break
            for stride in _gate_routes(k, gate_stride) if out_dtype != torch.float32 else (0,):
                config, source, _ = (
                    self._selection(kind, n, k, edge, stride, device, pinned=False,
                                    out_dtype=out_dtype) if cute else (None, None, None)
                )  # fmt: skip
                if config is None or source not in ("table", "forced"):
                    # cuBLAS loads its kernels lazily too: run the bucket once.
                    prime(("cublas", device, n, k, edge, stride, out_dtype), run, lo, stride)
                    continue
                key = _dtype_key((device.index, n, k, edge), out_dtype)
                if stride:
                    key = (device.index, n, k, edge, stride)
                plan = self._plans.get(key, False)
                if plan is False:
                    plan = self._resolve(device, n, k, edge, stride, out_dtype=out_dtype)
                    self._plans[key] = plan
                if plan is not None:
                    routes.append(dict(plan.launch, bucket=(lo, edge)))
                # Keyed by the compiled kernel, which call sites of one shape share.
                key = plan.compiled if plan else ("cublas", device, n, k, edge, stride, out_dtype)
                prime(key, run, lo, stride)
        # Past the buckets cuBLAS still picks kernels by M: a sqrt(2) ladder, the powers of
        # two its heuristics switch at, and max_m.
        ladder, m = {max_m} if max_m > MAX_M else set(), MAX_M + 1
        while m <= max_m:
            ladder.add(m)
            m = m * 1414 // 1000 + 1
        m = 2 * MAX_M
        while m <= max_m:
            ladder.add(m)
            m *= 2
        for m in sorted(ladder):
            prime(("cublas", device, n, k, m, out_dtype), run, m, 0)
        return routes

    def gate_fused(self, n, k, gate_stride, m, device=None, *, out_dtype=None) -> bool:
        """Fused admission for aligned BF16 inference operands, without compilation.

        Uses cached device properties; does not inspect streams or launch CUDA work.
        Tensor layout, alignment and activation autograd are checked by the call.
        """
        _output_dtype(out_dtype, True)
        if (
            self._override == "cublas"
            or not 1 <= m <= MAX_M
            or n <= 0
            or k <= 0
            or k % 8
            or not _gate_routes(k, gate_stride)[0]
        ):
            return False
        from ctikernels.linear.cute import available, mtiles_for

        if not available():
            return False
        device = torch.device("cuda") if device is None else torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return False
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        kind, capability, _sms = _dispatch._hardware(device)
        if capability not in ((12, 0), (12, 1)):
            return False
        edge = m_bucket(m)[1]
        config = self._config(kind, n, k, edge, gate_stride, device)
        return config is not None and _admits(config, mtiles_for(edge), gate_stride, capability)

    def _selection(self, kind, n, k, edge, gate_stride, device, pinned=True, *, out_dtype=None):
        if self._override == "cute":
            return self._forced, "forced", None
        if self._override == "cublas":
            return None, "forced", None
        if pinned and device is not None:
            # Outcome this op retained for its launches (table, cache or fallback).
            retained = self._pinned.get(
                _dtype_key((device.index, n, k, edge, gate_stride), out_dtype)
            )
            if retained is not None:
                config, source, record = retained
                return config, source, deepcopy(record)
        config = (
            tuned_gated_config(kind, n, k, gate_stride, edge)
            if gate_stride
            else tuned_config(
                kind, n, k, edge, **({"fp32": True} if out_dtype == torch.float32 else {})
            )
        )
        if config is not None:
            return config, "table", None
        if out_dtype == torch.float32:
            return None, "default", None
        if self._autotune_mode == "0":
            return None, "off", None
        context = self._autotune_mode == "context"
        if context and not _autotune.ACTIVE:  # Unlocked fast path; lookup rechecks.
            return None, "default", None
        bucket = m_bucket(edge)
        if device is None or bucket is None:
            return None, "default", None
        record = _autotune.lookup(
            _autotune.identity(device), n, k, gate_stride, bucket, in_context=context
        )
        config = _autotune.record_config(record, bucket, kind)
        if (device, n, k, edge, gate_stride, config) in _AUTOTUNE_COMPILE_FAILURES:
            # This process could not compile the cached winner and runs cuBLAS.
            config = None
            record = dict(record, provider="cublas", reason=_FAILED)
        valid = record and (record.get("provider") == "cublas" or config is not None)
        return config, "autotune" if valid else "default", record

    def _config(self, kind, n, k, edge, gate_stride, device=None, *, out_dtype=None):
        return self._selection(kind, n, k, edge, gate_stride, device, out_dtype=out_dtype)[0]

    def decision(self, n, k, m, *, gate_stride=0, device=None, out_dtype=None) -> dict:
        """Inspect shape policy without compiling or tuning (operand layout is not checked)."""
        _output_dtype(out_dtype, gate_stride)
        bucket = m_bucket(m)
        source = (
            "forced"
            if self._override != "auto"
            else ("off" if self._autotune_mode == "0" else "default")
        )
        result = {
            "provider": "cublas",
            "config": None,
            "source": source,
            "bucket": bucket,
            "record": None,
        }
        if (
            self._override == "cublas"
            or bucket is None
            or n <= 0
            or k <= 0
            or k % 8
            or (gate_stride and (gate_stride < k or gate_stride % 8))
        ):
            return result
        device = torch.device("cuda") if device is None else torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return result
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        kind, capability, _sms = _dispatch._hardware(device)
        if capability not in ((12, 0), (12, 1)):
            return result
        config, source, record = self._selection(
            kind, n, k, bucket[1], gate_stride, device, out_dtype=out_dtype
        )
        if config and not _admits(config, (bucket[1] + 7) // 8, gate_stride, capability):
            config = None
        return {
            "provider": "cute" if config else "cublas",
            "config": asdict(config) if config else None,
            "source": source,
            "bucket": bucket,
            "record": record,
        }

    def __call__(
        self, x, weight, out=None, *, out_dtype=None, gate=None, bias=None, workspace=None
    ):
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("Linear requires frozen weight (requires_grad=False)")
        if weight.ndim != 2 or x.ndim < 1 or x.shape[-1] != weight.shape[1]:
            raise ValueError("expected x [..., K] and weight [N,K]")
        if x.ndim != 2 or bias is not None:
            # F.linear's range (SGLang's dense call): leading dims and bias.
            return self._general(x, weight, out, out_dtype, gate, bias, workspace)
        if x.device != weight.device or x.dtype != weight.dtype:
            raise ValueError("x and weight must have matching device and dtype")
        _output_dtype(out_dtype)
        requested = out_dtype
        if out_dtype is None:
            out_dtype = out.dtype if out is not None else x.dtype
        fp32 = out_dtype == torch.float32 and (requested is not None or x.dtype == torch.bfloat16)
        if fp32 and x.dtype != torch.bfloat16:
            raise ValueError("FP32 output requires BF16 inputs and weight")
        if requested is not None and not fp32 and out_dtype != x.dtype:
            raise ValueError("out_dtype must match inputs or select FP32 output")
        if gate is not None and fp32:
            raise ValueError("gate is unsupported with FP32 output")
        m, k = x.shape
        n = weight.shape[0]
        if out is not None:
            if (
                out.shape != (m, n)
                or out.dtype != out_dtype
                or (not fp32 and out.dtype != x.dtype)
                or out.device != x.device
            ):
                raise ValueError("out must match output shape, dtype and device")
            if (n > 1 and out.stride(1) != 1) or (m > 1 and out.stride(0) < n):
                raise ValueError(
                    "out must be a row-strided view: unit last stride, row stride >= N"
                )
            if out.requires_grad:
                raise ValueError("out must not require gradients")
            if _overlaps(out, x) or _overlaps(out, weight):
                raise ValueError("out must not share input storage")
            if torch.is_grad_enabled() and x.requires_grad:
                raise ValueError("out is unsupported with activation autograd")
        if workspace is not None and (
            _overlaps(workspace, x)
            or _overlaps(workspace, weight)
            or (out is not None and _overlaps(workspace, out))
        ):
            raise ValueError("workspace must not share storage with x, weight or out")
        # Row stride of a strided out (0: contiguous rows); one row has no stride.
        ldy = out.stride(0) if out is not None and m > 1 and out.stride(0) != n else 0
        if workspace is None and (group := GraphWorkspace.active()) is not None:
            gate_stride = gate.stride(0) if gate is not None and gate.ndim == 2 else 0
            self._reserve_split_k(group, x, n, k, gate_stride, torch.float32 if fp32 else None)
        if gate is not None:
            return self._gated(x, weight, out, workspace, gate, m, n, k, ldy)
        if (
            self._override != "cublas"
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and 1 <= m <= MAX_M
            and n > 0
            and k % 8 == 0
            and k > 0
            and x.is_contiguous()
            and weight.is_contiguous()
            and not (x.requires_grad and torch.is_grad_enabled())
            and x.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0
            and (out is None or out.data_ptr() % 16 == 0)
            and ldy * (4 if fp32 else 2) % 16 == 0
        ):
            result = self._cute(
                x,
                weight,
                out,
                workspace,
                m,
                n,
                k,
                **({"out_dtype": torch.float32} if fp32 else {}),
                **({"ldy": ldy} if ldy else {}),
            )
            if result is not None:
                return result
        result = linear_fp32(x, weight, out) if fp32 else linear(x, weight, out)
        self._provider, self._kernels_per_call, self._launch = "cublas", 1, None
        return result

    def _general(self, x, weight, out, out_dtype, gate, bias, workspace):
        """[..., K] inputs flatten to [M, K]; bias calls run the market F.linear epilogue."""
        _output_dtype(out_dtype, gate is not None)
        lead, n, k = x.shape[:-1], weight.shape[0], x.shape[-1]
        m = math.prod(lead)
        operands = [t for t in (x, weight, bias, gate) if t is not None]
        out2 = None
        if out is not None:
            if out.shape != (*lead, n) or out.device != x.device:
                raise ValueError("out must match output shape, dtype and device")
            # Layout and alias rules of 2D calls, checked on the original operands:
            # a flattening copy of x or gate would hide shared storage.
            try:
                out2 = out.view(m, n)
            except RuntimeError:
                out2 = None
            if out2 is None or (n > 1 and out2.stride(1) != 1) or (m > 1 and out2.stride(0) < n):
                raise ValueError(
                    "out must be a row-strided view: unit last stride, row stride >= N"
                )
            if out.requires_grad:
                raise ValueError("out must not require gradients")
            if any(_overlaps(out, t) for t in operands):
                raise ValueError("out must not share input storage")
        if workspace is not None and any(
            _overlaps(workspace, t) for t in (*operands, *([out] if out is not None else []))
        ):
            raise ValueError("workspace must not share storage with x, weight or out")
        x2 = x.reshape(m, k)
        gate2 = None if gate is None else gate.reshape(m, gate.shape[-1])
        if bias is None:
            result = self(x2, weight, out2, out_dtype=out_dtype, gate=gate2, workspace=workspace)
            return out if out is not None else result.view(*lead, n)
        if bias.shape != (n,) or bias.device != x.device:
            raise ValueError("bias must be [N] on the input device")
        # Same dtype contract as bias-free calls: FP32 output only when requested
        # (out_dtype or an FP32 out) with BF16 inputs; otherwise the input dtype.
        dtype = out_dtype or (out.dtype if out is not None else x.dtype)
        fp32 = dtype == torch.float32 and (out_dtype is not None or x.dtype == torch.bfloat16)
        if fp32 and x.dtype != torch.bfloat16:
            raise ValueError("FP32 output requires BF16 inputs and weight")
        if (not fp32 and dtype != x.dtype) or (out is not None and out.dtype != dtype):
            raise ValueError("out must match output shape, dtype and device")
        a, gating = (x2, 0) if gate2 is None else _gated_input(x2, gate2)
        if fp32:
            result = linear_fp32(a, weight, out2).add_(bias)
        else:
            result = torch.addmm(bias, a, weight.t(), **({"out": out2} if out2 is not None else {}))
        self._provider, self._kernels_per_call, self._launch = "cublas", 1 + gating, None
        return out if out is not None else result.view(*lead, n)

    def _gated(self, x, weight, out, workspace, gate, m, n, k, ldy=0):
        if gate.ndim != 2 or gate.shape != x.shape:
            raise ValueError("gate must match x shape [M,K]")
        if gate.dtype != x.dtype or gate.device != x.device:
            raise ValueError("gate and x must have matching device and dtype")
        if gate.dtype != torch.bfloat16:
            raise ValueError("gate requires BF16 inputs")
        if gate.stride(1) != 1:
            raise ValueError("gate must have unit last stride")
        if gate.requires_grad:
            raise ValueError("gate must not require gradients")
        if out is not None and _overlaps(out, gate):
            raise ValueError("out must not share gate storage")
        if workspace is not None and _overlaps(workspace, gate):
            raise ValueError("workspace must not share gate storage")
        gate_stride = gate.stride(0)
        if (
            self._override != "cublas"
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and 1 <= m <= MAX_M
            and n > 0
            and k > 0
            and k % 8 == 0
            and x.is_contiguous()
            and weight.is_contiguous()
            and not (x.requires_grad and torch.is_grad_enabled())
            and x.data_ptr() % 16 == 0
            and weight.data_ptr() % 16 == 0
            and (out is None or out.data_ptr() % 16 == 0)
            and ldy % 8 == 0
            and _gate_routes(k, gate_stride)[0]
            and gate.data_ptr() % 16 == 0
        ):
            index = x.get_device()
            edge = m_bucket(m)[1]
            key = (index, n, k, edge, gate_stride)
            plan = self._plans.get(key, False)
            if plan is False or (plan is None and self._undecided(key)):
                plan = self._resolve(x.device, n, k, edge, gate_stride, weight)
                if plan is False:
                    raise _capture_miss(n, k, edge, gate_stride)
                self._plans[key] = plan
            if plan is not None:
                if index != torch.cuda.current_device():
                    with torch.cuda.device(index):
                        return self._launch_gated(
                            plan, x, weight, out, workspace, gate, m, n, ldy, key
                        )
                return self._launch_gated(plan, x, weight, out, workspace, gate, m, n, ldy, key)
        a, gating = _gated_input(x, gate)
        result = self(a, weight, out, workspace=workspace)
        self._provider = "unfused+" + self._provider
        self._kernels_per_call += gating
        return result

    def _cute(self, x, weight, out, workspace, m, n, k, *, out_dtype=None, ldy=0):
        index = x.get_device()
        edge = m_bucket(m)[1]
        key = _dtype_key((index, n, k, edge), out_dtype)
        plan = self._plans.get(key, False)
        if plan is False or (
            plan is None and out_dtype != torch.float32 and self._undecided((*key, 0))
        ):
            plan = self._resolve(x.device, n, k, edge, weight=weight, out_dtype=out_dtype)
            if plan is False:
                raise _capture_miss(n, k, edge)
            self._plans[key] = plan
        if plan is None:
            return None
        if index != torch.cuda.current_device():
            with torch.cuda.device(index):
                return self._launch_plan(plan, x, weight, out, workspace, m, n, out_dtype, ldy, key)
        return self._launch_plan(plan, x, weight, out, workspace, m, n, out_dtype, ldy, key)

    def _undecided(self, key):
        """Whether a cuBLAS key may still be tuned: only eagerly, in an autotune context,
        and only if warm-up retained no table, cache or tuning decision for it. A decided
        route is frozen, so capture and replay never consult autotune for it."""
        if not _autotune.ACTIVE:
            return False
        retained = self._pinned.get(key)
        if retained is not None and retained[1] != "default":
            return False
        with torch.cuda.device(key[0]):  # Capture state of the call's device stream.
            return not torch.cuda.is_current_stream_capturing()

    def _resolve(self, device, n, k, edge, gate_stride=0, weight=None, *, out_dtype=None):
        """_Plan, None (cuBLAS for this bucket), or False (capture miss, do not cache).

        Each retained outcome is pinned, so later queries describe the plan that
        launches even after an autotune context with another cache has exited.
        """
        plan, selection = self._resolve_fresh(
            device, n, k, edge, gate_stride, weight, out_dtype=out_dtype
        )
        if plan is not False and self._override == "auto":
            self._pinned[_dtype_key((device.index, n, k, edge, gate_stride), out_dtype)] = selection
        return plan

    def _resolve_fresh(self, device, n, k, edge, gate_stride, weight, *, out_dtype=None):
        from ctikernels.linear.cute import available, get_compiled, mtiles_for

        kind, capability, sms = _dispatch._hardware(device)
        if capability not in ((12, 0), (12, 1)):
            return None, (None, "default", None)
        forced = self._override == "cute"
        config, source, record = self._selection(
            kind, n, k, edge, gate_stride, device, pinned=False, out_dtype=out_dtype
        )
        if (
            out_dtype != torch.float32
            and source == "default"
            and record is None
            and _autotune.ACTIVE
            and weight is not None
        ):
            with torch.cuda.device(device):
                if not torch.cuda.is_current_stream_capturing():
                    record = _autotune.tune(
                        weight, _autotune.identity(device), kind, n, k, gate_stride, m_bucket(edge)
                    )
                    config = _autotune.record_config(record, m_bucket(edge), kind)
                    source = "autotune" if record is not None else "default"
        record = deepcopy(record)
        fallback = None, source, record
        if config is None:
            return None, fallback
        failure_key = (device, n, k, edge, gate_stride, config)
        if source == "autotune" and failure_key in _AUTOTUNE_COMPILE_FAILURES:
            return None, (None, source, dict(record, provider="cublas", reason=_FAILED))
        if not available():
            if forced:
                raise RuntimeError("forced cute requires nvidia-cutlass-dsl==4.8.0")
            return None, fallback
        mtiles = mtiles_for(edge)
        if not _admits(config, mtiles, gate_stride, capability):
            if forced and not _cluster_fits(config, capability):
                raise ValueError(
                    f"config cluster of {config.splits} CTAs exceeds this device's limit"
                )
            if forced and not (gate_stride and config.variant == "wide"):
                raise ValueError("config exceeds shared memory at this M bucket")
            return None, fallback
        with torch.cuda.device(device):
            # Capture state of the target device's current stream.
            capturing = torch.cuda.is_current_stream_capturing()
            # Ungated plans keep the original compile signature.
            gated = (gate_stride,) if gate_stride else ()
            try:
                compiled = get_compiled(
                    device,
                    n,
                    k,
                    config,
                    capability,
                    capturing,
                    mtiles,
                    *gated,
                    **({"fp32": True} if out_dtype == torch.float32 else {}),
                )
            except Exception:  # Cached autotune decisions are optional.
                if source != "autotune":
                    raise
                if failure_key not in _AUTOTUNE_COMPILE_FAILURES:
                    _AUTOTUNE_COMPILE_FAILURES.add(failure_key)
                    warnings.warn(
                        "Linear autotune cached config compilation failed; using cuBLAS",
                        stacklevel=2,
                    )
                return None, (None, source, dict(record, provider="cublas", reason=_FAILED))
        if compiled is None:
            return False, None

        def build(out_stride, capturing):
            with torch.cuda.device(device):
                return get_compiled(
                    device,
                    n,
                    k,
                    config,
                    capability,
                    capturing,
                    mtiles,
                    *gated,
                    **({"fp32": True} if out_dtype == torch.float32 else {}),
                    out_stride=out_stride,
                )

        plan = _Plan(compiled, config, mtiles, build)
        if not forced:
            plan.launch["source"] = source
        if out_dtype == torch.float32:
            plan.launch["out_dtype"] = "fp32"
        if gate_stride:
            plan.launch["gate_stride"] = gate_stride
        return plan, (config, source, record)

    def _scratch(self, plan, x, workspace, m, n):
        need = plan.splits * m * n
        if workspace is not None:
            if (
                not workspace.is_cuda
                or workspace.device != x.device
                or not workspace.is_contiguous()
                or workspace.data_ptr() % 16
                or workspace.numel() * workspace.element_size() < 4 * need
            ):
                raise ValueError(
                    "workspace must be a contiguous, aligned CUDA buffer of workspace_bytes"
                )
            storage = workspace
        elif torch.cuda.is_current_stream_capturing():
            # The graph's own pool tracks the lifetime and keeps the address on replay.
            return torch.empty((plan.splits, m, n), device=x.device, dtype=torch.float32)
        elif (group := GraphWorkspace.active()) is not None:
            # Eager calls inside a graph workspace share its slot (reserved on entry to the
            # call) instead of the stream cache.
            storage = group.take(_split_k_slot(x), m, (need,))[0]
        else:
            index = x.get_device()
            slot = (index, torch._C._cuda_getCurrentRawStream(index))
            with _SCRATCH_LOCK:
                storage = _SCRATCH.get(slot)
                if storage is None or storage.numel() < need:
                    # Grows only to the largest (splits, M, N) seen; flat after warm-up.
                    # Views live with their buffer, so growth releases every old view.
                    storage = torch.empty(need, device=x.device, dtype=torch.float32)
                    _SCRATCH[slot] = storage
                owner, views = _VIEWS.get(slot, (None, None))
                if owner is not storage:
                    views = {}
                    _VIEWS[slot] = (storage, views)
                view = views.get((plan.splits, m, n))
                if view is None:
                    view = storage[:need].view(plan.splits, m, n)
                    views[plan.splits, m, n] = view
                return view
        # Caller-owned workspace: keep at most one cached view, never an old buffer.
        cached = self._views
        if cached.get("storage") is storage and cached.get("key") == (plan.splits, m, n):
            return cached["view"]
        flat = storage.view(-1)
        if flat.dtype != torch.float32:
            flat = flat.view(torch.uint8)[: 4 * need].view(torch.float32)
        view = flat[:need].view(plan.splits, m, n)
        self._views = {"storage": storage, "key": (plan.splits, m, n), "view": view}
        return view

    def _reserve_split_k(self, group, x, n, k, gate_stride, out_dtype):
        """Every eager call inside a graph workspace reserves the split-K slot for each M up to
        ``max_rows``, whatever its own route: a later, smaller bucket of this op may split
        after the first capture has frozen the workspace."""
        if (
            self._override == "cublas"
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or n <= 0
            or k <= 0
            or k % 8
        ):
            return
        group.reserve(
            _split_k_slot(x),
            (self, n, k, x.device, gate_stride, out_dtype),
            _split_k_counts,
            lambda counts: (torch.empty(counts[0], device=x.device, dtype=torch.float32),),
        )

    def _launch_plan(self, plan, x, weight, out, workspace, m, n, out_dtype=None, ldy=0, key=()):
        compiled, view = _output(plan, out, m, n, ldy, key)
        if out is None:
            out = view = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
        partial = self._scratch(plan, x, workspace, m, n) if plan.splits else _dummy(x.device)
        compiled(x, weight, view, partial)
        self._provider = "cute"
        self._kernels_per_call = plan.kernels
        self._launch = dict(plan.launch, out_stride=ldy) if ldy else plan.launch
        return out

    def _launch_gated(self, plan, x, weight, out, workspace, gate, m, n, ldy=0, key=()):
        compiled, view = _output(plan, out, m, n, ldy, key)
        if out is None:
            out = view = torch.empty((m, n), device=x.device, dtype=x.dtype)
        partial = self._scratch(plan, x, workspace, m, n) if plan.splits else _dummy(x.device)
        compiled(x, weight, view, partial, gate)
        self._provider = "cute-gated"
        self._kernels_per_call = plan.kernels
        self._launch = dict(plan.launch, out_stride=ldy) if ldy else plan.launch
        return out


def _split_k_slot(x):
    return ("linear_split_k", x.device, torch.float32)


def _split_k_counts(max_rows, op, n, k, device, gate_stride, out_dtype):
    return (op.workspace_bytes(n, k, max_rows, device, gate_stride, out_dtype=out_dtype) // 4,)


def gated_input(x, gate):
    """``x * sigmoid(gate)`` in FP32, rounded once to ``x.dtype``: the input the unfused gated
    Linear multiplies, for callers that need it themselves. On CUDA its bits equal the fused gated
    Linear's own gating (one shared instruction sequence, so fused and unfused calls agree), within
    one ULP of ``torch.sigmoid(gate.float()).mul_(x).to(x.dtype)``. ``gate`` has ``x``'s shape and
    device, any strides; the result has ``x``'s shape, contiguous. One Triton launch on CUDA for
    BF16/FP16/FP32; the PyTorch ops on the CPU and under activation autograd (differentiable
    in ``x``). Graph-capturable after one eager call per dtype pair; no host sync."""
    if not isinstance(gate, torch.Tensor) or gate.shape != x.shape:
        raise ValueError("gate must be a tensor of x's shape")
    if gate.device != x.device:
        raise ValueError("gate and x must be on one device")
    if not (x.is_floating_point() and gate.is_floating_point()):
        raise ValueError("x and gate must be floating point")
    if x.ndim == 2:
        return _gated_input(x, gate)[0]
    k = x.shape[-1] if x.ndim else 1
    return _gated_input(x.reshape(-1, k), gate.reshape(-1, k))[0].view(x.shape)


def _gated_input(x, gate):
    """(x * sigmoid(gate) in FP32 rounded once to x.dtype, its launches): one Triton launch on
    CUDA; PyTorch's four ops (FP32 conversion, sigmoid, multiply, conversion) on the CPU, under
    activation autograd (differentiable in x) and for dtypes beyond BF16/FP16/FP32."""
    if (
        x.device.type == "cuda"
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and not (x.requires_grad and torch.is_grad_enabled())
    ):
        from .triton._gate import gate_multiply

        return gate_multiply(x, gate), 1
    return torch.sigmoid(gate.float()).mul_(x).to(x.dtype), 4


def _output(plan, out, m, n, ldy, key):
    """Compiled launch for ``out``'s row stride and the view it writes. A single row
    or column has an arbitrary unused stride; normalize it to the contiguous launch."""
    compiled = plan.kernel(ldy)
    if compiled is None:
        gate_stride = key[4] if len(key) > 4 and type(key[4]) is int else 0
        raise _capture_miss(*key[1:4], gate_stride, ldy)
    if out is not None and not ldy and out.stride() != (n, 1):
        out = out.as_strided((m, n), (n, 1))
    return compiled, out


_DUMMIES: dict[int, torch.Tensor] = {}


def _dummy(device):
    """Unused [1,1,1] partial argument of single-split kernels (one per device)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    tensor = _DUMMIES.get(index)
    if tensor is None:
        tensor = torch.empty((1, 1, 1), device=device, dtype=torch.float32)
        _DUMMIES[index] = tensor
    return tensor


__all__ = ["Linear"]
