# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Inference norms; last_standalone_provider maps op names to their last successful provider.

Rerouted providers are strings with a ``reason`` attribute, also retained by cached plans.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from math import gcd
from typing import Literal

import torch
import torch.nn.functional as F

from ctikernels import _flashinfer_version
from ctikernels._norm_act_reroute import engine_checks, reroute
from ctikernels._runtime.launch import prime

from . import flashinfer
from ._dispatch import Config, device_class, entry_for, parse_config

VARIABLE = "CTI_NORM_PROVIDER"
_PROVIDERS = ("auto", "cute", "flashinfer", "triton")
last_standalone_provider = {}


@lru_cache(maxsize=None)
def _hardware(device):
    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    return device_class(capability, props.multi_processor_count), capability


def _overlap(a, b):
    # Admission calls this only for contiguous buffers. Disjoint views are safe.
    return (
        a.numel()
        and b.numel()
        and (
            a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
            and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
        )
    )


def _admitted(x, residual, weight, zero_centered, store_residual, normalize):
    return (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim > 0
        and x.shape[-1] > 0
        and x.shape[-1] % 8 == 0
        and residual.shape == x.shape
        and residual.dtype in (x.dtype, torch.float32)
        and weight.shape == (x.shape[-1],)
        and weight.dtype in (x.dtype, torch.float32)
        and all(
            t.device == x.device and t.is_contiguous() and t.data_ptr() % 16 == 0
            for t in (x, residual, weight)
        )
        and not any(_overlap(a, b) for a, b in ((x, residual), (x, weight), (residual, weight)))
        and normalize == "fp32_sum"
    )


def _span(tensor):
    if not tensor.numel():
        return tensor.data_ptr(), tensor.data_ptr()
    size = 1 + sum(
        (n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


def _sum_rows(x, residual, weight, sum_out):
    if sum_out is None:
        return None
    _require_mutable("add_rms_norm_ sum_out", sum_out)
    if x.ndim == 0 or not x.shape[-1] or sum_out.shape != x.shape:
        raise ValueError("sum_out must match x's shape with a nonempty hidden dimension")
    if sum_out.dtype != residual.dtype:
        raise TypeError("sum_out must use residual's dtype")
    if sum_out.device != x.device:
        raise ValueError("sum_out must be on the input device")
    hidden = x.shape[-1]
    try:
        rows = sum_out.view(x.numel() // hidden, hidden)
    except RuntimeError as error:
        raise ValueError("sum_out leading dimensions must collapse into one row stride") from error
    if rows.stride(1) != 1 or (rows.shape[0] > 1 and rows.stride(0) < hidden):
        raise ValueError("sum_out needs unit hidden stride and row stride >= hidden")
    lo, hi = _span(rows)
    for tensor in (x, residual, weight):
        other_lo, other_hi = _span(tensor)
        if lo < hi and other_lo < other_hi and lo < other_hi and other_lo < hi:
            raise ValueError("sum_out must not overlap x, residual or weight")
    return rows


def _signature(tensor):
    # No pointers/tensors retained: the same plan can launch on new storage.
    return None if tensor is None else (tensor.device, tensor.shape, tensor.stride(), tensor.dtype)


class AddRMSNorm:
    """Call-site provider/config override; ``provider`` reports the successful launch.

    ``dispatched_config`` reports the last successful CuTe layout (including split),
    or None for other providers. Empty calls return without provider resolution,
    but still check the version when FlashInfer is forced.
    Forced CuTe rejects inadmissible buffers or missing compilation. Triton supports
    both weight conventions. ``sum_out`` and ``store_residual`` follow :func:`add_rms_norm_`.

    ``warm_up(max_m, hidden=..., dtype=..., device=...)`` resolves M=1..max_m
    and returns consecutive provider/config routes. CuTe routes compile without
    launching; FlashInfer/Triton routes launch once per M on zeroed scratch to
    trigger their lazy compilation. No timing is performed.
    Plans for these contiguous aligned shapes are pinned for the instance lifetime,
    independently of the bounded ordinary cache. Warm before capture with warm_up:
    an uncompiled selected CuTe config raises during capture; a recorded eager
    compile failure retains its fallback.

    ``static_weight=True`` is valid only when weight is never written while the op
    can run: no writes between graph replays, between capture and replay, or by a
    preceding kernel in the same stream. Any writer requires the default False.
    The opt-in reads weight before griddepcontrol.wait; config/environment alone
    cannot enable early prefetch. CuTe owns no workspace or output allocations.
    """

    def __init__(self, *, provider=None, config=None, static_weight=False):
        if provider is not None and provider not in _PROVIDERS:
            raise ValueError("provider must be auto, cute, flashinfer or triton")
        if config is not None and not isinstance(config, Config):
            raise TypeError("config must be Config")
        if not isinstance(static_weight, bool):
            raise TypeError("static_weight must be bool")
        self._override, self.config = provider, config
        self.static_weight = static_weight
        self.provider = None
        self.dispatched_config = None
        self._plans = {}
        self._pinned = {}
        # Benchmarks can forbid resolution before it performs any CUDA setup.
        self._require_cached_plan = False

    def warm_up(
        self,
        max_m,
        *,
        hidden,
        dtype,
        device,
        residual_dtype=None,
        weight_dtype=None,
        zero_centered=False,
        store_residual=True,
        sum_out=False,
    ):
        """Compile and pin contiguous aligned plans for M=1..max_m.

        CuTe routes compile and launch each compiled kernel once (``_runtime.launch``);
        FlashInfer/Triton routes launch once per M on zeroed scratch to trigger their lazy
        compilation; no timing.

        Returns maximal runs ``{"m": (lo, hi), "provider": ..., "config": name_or_None}``.
        Uses this instance's overrides/static_weight and the current environment.
        Call eagerly before CUDA graph capture. ``sum_out=True`` warms a contiguous
        residual-dtype output; other layouts need their own eager call.
        """
        if max_m < 1:
            raise ValueError("max_m must be positive")
        if hidden < 1:
            raise ValueError("hidden must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "AddRMSNorm.warm_up must run eagerly, outside CUDA graph capture"
                )
            x = torch.zeros((max_m, hidden), dtype=dtype, device=device)
            residual = torch.zeros_like(x, dtype=residual_dtype or dtype)
            weight = torch.zeros(hidden, dtype=weight_dtype or dtype, device=device)
            summed = torch.zeros_like(residual) if sum_out else None
            if residual.dtype not in (dtype, torch.float32):
                raise TypeError("residual must use the input dtype or FP32")
            routes = []
            for m in range(1, max_m + 1):
                xx, rr = x[:m], residual[:m]
                ss = None if summed is None else summed[:m]
                rows = _sum_rows(xx, rr, weight, ss)
                provider, launch = self._plan(
                    xx, rr, weight, zero_centered, store_residual, "fp32_sum", ss, rows, pin=True
                )
                if provider == "cute":
                    prime(launch.kernel, launch, xx, rr, weight, 1e-6, rows)
                else:
                    launch(xx, rr, weight, 1e-6, rows)
                config = getattr(launch, "config", None)
                name = None if config is None else config.name
                if routes and (routes[-1]["provider"], routes[-1]["config"]) == (provider, name):
                    routes[-1]["m"] = (routes[-1]["m"][0], m)
                else:
                    routes.append({"m": (m, m), "provider": provider, "config": name})
            return routes

    def __call__(
        self,
        x,
        residual,
        weight,
        eps,
        *,
        zero_centered=False,
        store_residual=True,
        normalize="fp32_sum",
        sum_out=None,
    ):
        _require_frozen_weight("add_rms_norm_", weight)
        _require_mutable("add_rms_norm_", x, residual)
        if normalize not in ("fp32_sum", "stored_sum"):
            raise ValueError(f"normalize must be 'fp32_sum' or 'stored_sum', not {normalize!r}")
        sum_rows = _sum_rows(x, residual, weight, sum_out)
        if residual.dtype not in (x.dtype, torch.float32):
            raise TypeError("residual must use the input dtype or FP32")
        if not x.numel():
            if (self._override or os.environ.get(VARIABLE, "auto")) == "flashinfer":
                _flashinfer_version.require_allowed()
            self.provider, self.dispatched_config = "empty", None
            return x, residual
        provider, launch = self._plan(
            x, residual, weight, zero_centered, store_residual, normalize, sum_out, sum_rows
        )
        launch(x, residual, weight, eps, sum_rows)
        self.provider = provider
        self.dispatched_config = getattr(launch, "config", None)
        return x, residual

    def _plan(
        self,
        x,
        residual,
        weight,
        zero_centered,
        store_residual,
        normalize,
        sum_out,
        sum_rows,
        *,
        pin=False,
    ):
        # Admission includes pointer alignment/overlap, and MUST run even on cache hits.
        admitted = _admitted(x, residual, weight, zero_centered, store_residual, normalize)
        admitted = admitted and (
            sum_rows is None or (sum_rows.data_ptr() % 16 == 0 and sum_rows.stride(0) % 8 == 0)
        )
        env = (
            os.environ.get(VARIABLE, "auto"),
            os.environ.get("CTI_NORM_CONFIG"),
            _flashinfer_version.allowed(),
            engine_checks(),
        )
        key = (
            _signature(x),
            _signature(residual),
            _signature(weight),
            _signature(sum_out),
            zero_centered,
            store_residual,
            self.static_weight,
            normalize,
            env,
            self._override,
            self.config,
            admitted,
            # FlashInfer admission depends on pointer alignment, which the
            # shape/stride signature does not carry: key on it per call.
            tuple(None if t is None else t.data_ptr() % 16 for t in (x, residual, weight)),
        )
        plan = self._pinned.get(key) or self._plans.get(key)
        if plan is None:
            if self._require_cached_plan and not pin:
                raise RuntimeError("resolution cache miss: warm the exact capture views first")
            plan = self._resolve(
                x,
                residual,
                weight,
                zero_centered,
                store_residual,
                sum_rows,
                normalize,
                admitted,
                env,
            )
            if not pin and (
                x.device.type != "cuda" or not torch.cuda.is_current_stream_capturing()
            ):
                if len(self._plans) >= 128:
                    self._plans.clear()
                self._plans[key] = plan
        if pin:
            self._pinned[key] = plan
        return plan

    def _resolve(self, x, residual, weight, zero, store, sum_rows, normalize, admitted, env):
        override = self._override or env[0]
        if override not in _PROVIDERS:
            raise ValueError(f"{VARIABLE} must be auto, cute, flashinfer or triton")
        extra = sum_rows is not None or not store
        fi_compatible = (
            not extra
            and normalize == "fp32_sum"
            and x.dtype in (torch.float16, torch.bfloat16)
            and residual.dtype == weight.dtype == x.dtype
            and weight.is_contiguous()
            and all(
                _one_stride(t) and _ordinary_rms_direct(t.view(-1, t.shape[-1]))
                for t in (x, residual)
            )
            and weight.data_ptr() % 16 == 0
        )
        if override == "flashinfer":
            if extra:
                raise RuntimeError(
                    "forced flashinfer does not support sum_out or store_residual=False"
                )
            if not fi_compatible:
                raise RuntimeError(
                    "forced flashinfer requires same-dtype fp32_sum/store_residual buffers with direct alignment"
                )
            _flashinfer_version.require_allowed()
        reason = None
        entry, capability, kind = None, None, None
        selected = override if override != "auto" else "triton"
        if override == "cute" and not admitted and not x.is_cuda:
            raise RuntimeError("forced cute requires admitted aligned CUDA buffers and device")
        if override == "cute" or (override == "auto" and admitted):
            kind, capability = _hardware(x.device)
            if override == "cute" and capability not in ((12, 0), (12, 1)):
                raise RuntimeError("CuTe norm is unqualified outside capability 12.x")
            if capability in ((12, 0), (12, 1)) and admitted:
                from .cute import available

                entry = entry_for(
                    kind,
                    x.shape[-1],
                    str(x.dtype).removeprefix("torch."),
                    x.numel() // x.shape[-1],
                    zero,
                    self.static_weight,
                    residual_dtype=str(residual.dtype).removeprefix("torch."),
                )
                if override == "auto":
                    if (
                        entry is None
                        and self.config is None
                        and not Config(threads=1024).admits(x.shape[-1])
                    ):
                        selected = "triton"
                    elif not available():
                        reason = "CuTe norm requires nvidia-cutlass-dsl==4.8.0"
                    else:
                        selected = entry.provider if entry else "cute"
            elif override == "cute":
                raise RuntimeError("forced cute requires admitted aligned CUDA buffers and device")
        if selected == "flashinfer" and override == "auto":
            if not fi_compatible:
                # The extra local semantics intentionally bypass the FlashInfer table band.
                selected = "cute"
            elif not env[2]:
                reason = _flashinfer_version.unusable_reason() or "FlashInfer is unusable"
                selected = "cute"
        if reason and engine_checks():
            reroute("add_rms_norm_", selected, reason)
        if selected == "cute":
            setting = env[1] if override == "cute" else None
            config = (
                self.config
                or (parse_config(setting) if setting else None)
                or (entry.config if entry else Config())
            )
            explicit_config = self.config is not None or setting is not None
            if not config.admits(x.shape[-1]) and not explicit_config and entry is None:
                config = Config(threads=1024)
                if not config.admits(x.shape[-1]):
                    if override == "cute":
                        raise RuntimeError("CuTe norm configuration exceeds 16 chunks per thread")
                    selected = "triton"
            config = replace(config, static_weight=self.static_weight)
            if selected == "cute":
                launch = self._cute(
                    x,
                    residual,
                    weight,
                    zero,
                    store,
                    sum_rows,
                    config,
                    capability,
                    override == "cute",
                )
                if launch is not None:
                    launch.config = config
                    return (reroute("add_rms_norm_", "cute", reason) if reason else "cute"), launch
                if override == "cute":
                    raise RuntimeError(
                        "forced cute requires admitted aligned CUDA buffers and device"
                    )
                from .cute import available

                failure = (
                    "CuTe norm configuration does not admit hidden size"
                    if not config.admits(x.shape[-1])
                    else "CuTe norm requires nvidia-cutlass-dsl==4.8.0"
                    if not available()
                    else "CuTe norm compilation unavailable (recorded eager failure)"
                )
                reason = f"{reason}; {failure}" if reason else failure
        if selected == "flashinfer":

            def launch_fi(x, residual, weight, eps, summed):
                if x.numel():
                    flashinfer.add_rms_norm_(x, residual, weight, eps, zero_centered=zero)

            return "flashinfer", launch_fi
        from ._zero_centered_rms_norm import add_rms_norm_ as legacy

        pack = not (_one_stride(x) and _one_stride(residual))
        if pack:
            reason = "pack"

        def launch_triton(x, residual, weight, eps, summed):
            if pack and x.data_ptr() == residual.data_ptr():
                raise ValueError("residual must be distinct from the input")
            xx = x.clone(memory_format=torch.contiguous_format) if pack else x
            rr = residual.clone(memory_format=torch.contiguous_format) if pack else residual
            kwargs = {} if summed is None else {"sum_out": summed}
            legacy(
                xx,
                rr,
                weight,
                eps,
                store_residual=store,
                normalize_stored=normalize == "stored_sum",
                zero_centered=zero,
                **kwargs,
            )
            if pack:
                x.copy_(xx)
                if store:
                    residual.copy_(rr)

        return (reroute("add_rms_norm_", "triton", reason) if reason else "triton"), launch_triton

    @staticmethod
    def _cute(x, residual, weight, zero, store, sum_rows, config, capability, strict):
        from .cute import available, get_compiled

        if not config.admits(x.shape[-1]):
            if strict:
                raise RuntimeError("CuTe norm configuration exceeds 16 chunks per thread")
            return None
        if not available():
            if strict:
                raise RuntimeError("forced cute requires nvidia-cutlass-dsl==4.8.0")
            return None
        m, hidden = x.numel() // x.shape[-1], x.shape[-1]
        if m == 0:
            return lambda x, residual, weight, eps, summed: None
        with torch.cuda.device(x.device):
            compiled = get_compiled(
                x.device,
                str(x.dtype).removeprefix("torch."),
                str(weight.dtype).removeprefix("torch."),
                hidden,
                zero,
                config,
                capability,
                residual_dtype=str(residual.dtype).removeprefix("torch."),
                store_residual=store,
                has_sum=sum_rows is not None,
                capturing=torch.cuda.is_current_stream_capturing(),
                strict=strict,
                capture_miss="raise",
            )
        if compiled is None:
            return None

        def launch(x, residual, weight, eps, summed):
            with torch.cuda.device(x.device):
                xx = x if x.ndim == 2 else x.view(m, hidden)
                rr = residual if residual.ndim == 2 else residual.view(m, hidden)
                summed = rr if summed is None else summed
                compiled(xx, rr, weight, summed, summed.stride(0), float(eps))

        launch.kernel = compiled
        return launch


_DEFAULT_ADD = AddRMSNorm()


def warm_up_add_rms_norm(max_m, **kw):
    """Compile and pin shared add_rms_norm_ plans; see :meth:`AddRMSNorm.warm_up`."""
    return _DEFAULT_ADD.warm_up(max_m, **kw)


def _require_no_autograd(operation: str, *tensors: torch.Tensor) -> None:
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(f"{operation} does not support autograd")


def _require_mutable(operation: str, *tensors: torch.Tensor) -> None:
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(f"{operation} cannot modify a tensor that requires gradients")


def _require_frozen_weight(operation: str, weight: torch.Tensor) -> None:
    if weight.requires_grad:
        raise ValueError(f"{operation} requires a frozen weight")


@lru_cache(maxsize=128)
def _resolve_standalone(
    operation, x_signature, weight_signature, zero, out_dtype, env, direct=True
):
    override = env[0]
    provider = "triton"
    if not zero and operation != "residual_rms_norm":
        if override not in _PROVIDERS:
            raise ValueError(f"{VARIABLE} must be auto, cute, flashinfer or triton")
        dtype = x_signature[-1]
        fi_compatible = (
            dtype in (torch.float16, torch.bfloat16)
            and weight_signature[-1] == dtype
            and out_dtype in (None, dtype)
        )
        if override == "flashinfer":
            _flashinfer_version.require_allowed()
            if not fi_compatible or not direct:
                raise TypeError(
                    "forced flashinfer requires FP16/BF16 x, weight and output of the same dtype"
                )
        if override == "flashinfer" or (
            override != "triton" and fi_compatible and env[2] and direct
        ):
            from flashinfer import rmsnorm

            if operation == "rms_norm_":
                return "flashinfer", lambda x, weight, eps: rmsnorm(x, weight, eps, out=x)
            return "flashinfer", rmsnorm
        if override != "triton" and fi_compatible:
            provider = reroute(
                operation,
                "triton",
                (_flashinfer_version.unusable_reason() or "FlashInfer is unusable")
                if not env[2]
                else "FlashInfer requires directly aligned rank-2/3 rows",
            )
    from . import _zero_centered_rms_norm as kernels

    fn = getattr(kernels, operation)
    if operation == "rms_norm_":
        return provider, lambda x, weight, eps: fn(x, weight, eps, zero_centered=zero)
    return provider, lambda x, weight, eps: fn(
        x, weight, eps, out_dtype=out_dtype, zero_centered=zero
    )


def _standalone(operation, x, weight, eps, zero, out_dtype=None):
    env = (
        os.environ.get(VARIABLE, "auto"),
        os.environ.get("CTI_NORM_CONFIG"),
        _flashinfer_version.allowed(),
        engine_checks(),
    )
    provider, launch = _resolve_standalone(
        operation,
        _signature(x),
        _signature(weight),
        zero,
        out_dtype,
        env,
        _ordinary_rms_direct(x) and weight.data_ptr() % 16 == 0,
    )
    native = _one_stride(x) if operation == "residual_rms_norm" else _two_strides(x)
    if not native:
        provider = reroute(operation, provider, "pack")
        packed = x.clone(memory_format=torch.contiguous_format)
        result = launch(packed, weight, eps)
        if operation == "rms_norm_":
            x.copy_(packed)
            result = x
    else:
        result = launch(x, weight, eps)
    last_standalone_provider[operation] = provider
    return result


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply inference RMS normalization without mutating ``x``.

    ``zero_centered`` scales by ``1 + weight`` instead of ``weight`` (Qwen3.5 and
    Gemma store the weight zero-centered; it is never pre-incremented). Both paths
    accept CUDA FP16/BF16/FP32 ``x`` shaped ``[..., hidden]`` whose prefix collapses
    to two strides with a contiguous hidden dimension, such as per-head views of a
    merged projection, and a contiguous ``[hidden]`` weight of any of those dtypes.
    It normalizes and scales in FP32 and rounds once to ``out_dtype``, any of those
    dtypes and ``x``'s by default; for FP32 ``x`` the result thus equals
    ``rms_norm(x, ...).to(out_dtype)`` bitwise on Triton. Standard weights use
    FlashInfer only for matching FP16/BF16 input, weight and output dtypes when qualified.

    Rank-2/3 unit-hidden-stride views run natively on Triton at any row/head stride.
    Higher ranks run natively when their prefix collapses to two strides. Other
    layouts use a contiguous temporary, reported as a reroute with reason "pack".
    FlashInfer requires its aligned rank-2/3 ABI directly; otherwise auto visibly
    reroutes to Triton without packing an already native view.
    Empty inputs of every rank return without a provider launch.

    The operation follows the current CUDA stream and returns independent output
    storage. It has no backward and must not participate in an autograd graph.
    Weight must be frozen in both grad modes; otherwise it raises ValueError.
    """

    _require_frozen_weight("rms_norm", weight)
    _require_no_autograd("rms_norm", x)
    if not x.numel():
        last_standalone_provider["rms_norm"] = "empty"
        return torch.empty_like(x, dtype=out_dtype or x.dtype)
    return _standalone("rms_norm", x, weight, eps, zero_centered, out_dtype)


def rms_norm_(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
) -> torch.Tensor:
    """Apply inference RMS normalization by overwriting ``x``.

    The caller must own ``x``, prove its previous value is dead, and pass storage
    that does not require gradients. ``zero_centered`` has the contract of
    :func:`rms_norm` and writes through strided views, so it normalizes Q/K heads
    inside a merged projection without a copy. The operation follows the current
    CUDA stream, returns ``x``, and has no backward. Ordinary layouts are adapted
    as in :func:`rms_norm`, with copy-back when needed; empty inputs never launch.
    """

    _require_frozen_weight("rms_norm_", weight)
    _require_mutable("rms_norm_", x)
    if not x.numel():
        last_standalone_provider["rms_norm_"] = "empty"
        return x
    return _standalone("rms_norm_", x, weight, eps, zero_centered)


def _ordinary_rms_direct(x):
    """Preserve FlashInfer's rank-2/3 ABI, including strided Q/K head views."""
    if x.ndim not in (2, 3) or not x.shape[-1] or x.stride(-1) != 1:
        return False
    lanes = gcd(16 // x.element_size(), x.shape[-1])
    return x.data_ptr() % (lanes * x.element_size()) == 0 and all(
        stride % lanes == 0 for stride in x.stride()[:-1]
    )


def _one_stride(x):
    if x.ndim == 0 or not x.shape[-1] or x.stride(-1) != 1:
        return False
    try:
        x.view(-1, x.shape[-1])
    except RuntimeError:
        return False
    return True


def _two_strides(x):
    if x.ndim == 0 or not x.shape[-1] or x.stride(-1) != 1:
        return False
    groups = 1
    for size in (*reversed(x.shape[:-1]), 1):
        groups *= size
        try:
            x.view(-1, groups, x.shape[-1])
        except RuntimeError:
            continue
        return True
    return False


def add_rms_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    store_residual: bool = True,
    normalize: Literal["fp32_sum", "stored_sum"] = "fp32_sum",
    sum_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add and normalize in place for inference, returning ``(x, residual)``.

    ``x`` receives the normalization of the FP32 sum, scaled by ``weight`` or,
    with ``zero_centered=True``, by ``1 + weight``. ``residual`` receives the sum
    rounded to its dtype unless ``store_residual=False`` leaves it unchanged.
    ``normalize="stored_sum"`` instead normalizes that rounded sum (Triton).
    With an FP32 residual the sum has no storage rounding: both modes coincide.
    Forced Triton equals the training forward bitwise; CuTe uses a different
    reduction and inverse root and meets the FP64 accuracy gate instead.

    ``sum_out`` also receives that same rounded sum, even when residual is not
    written. It must match x's shape, residual's dtype and device, have unit last
    stride and a row stride >= hidden (a single row may have any row stride).
    Leading dimensions must collapse to [M, hidden] without copying. Column slices
    of wider auxiliary buffers are supported. Its address span, including row
    gaps, must not overlap x, residual or weight; validation precedes any launch.
    The output must not require gradients.

    Provider overrides are checked first. Auto on capability 12.0/12.1 uses the
    existing speed table for CuTe-admitted inputs, or generic Config() off-table;
    all other inputs use Triton.
    Triton accepts any H >= 1, FP16/BF16/FP32 x, residual and weight in x's dtype
    or FP32, and arbitrary row strides with unit hidden stride. N-D prefixes that
    cannot collapse to one row stride, or strided hidden axes, use a visible "pack"
    reroute with contiguous temporaries and copy-back of the modified inputs.
    All weight, residual, normalization and store modes are independent. Alignment
    only selects the CuTe fast path. sum_out never overlaps the input spans.

    Empty inputs return the original buffers without resolving or launching a provider.
    Inputs to be modified must not require gradients; weight must be frozen
    (ValueError in either grad mode). Execution follows the current CUDA stream.
    Warm before capture with
    :func:`warm_up_add_rms_norm`, which pins contiguous aligned plans for M=1..max_m.
    An uncompiled selected CuTe config raises during capture instead of changing
    provider; recorded eager compile failures retain their fallback.
    """

    _require_frozen_weight("add_rms_norm_", weight)
    return _DEFAULT_ADD(
        x,
        residual,
        weight,
        eps,
        zero_centered=zero_centered,
        store_residual=store_residual,
        normalize=normalize,
        sum_out=sum_out,
    )


def residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize a residual without a fused add into new ``out_dtype`` storage for inference.

    This is the ``residual=None`` branch of vLLM's and SGLang's ``RMSNorm.forward``: the
    norm of a residual as it stands, such as the first layer's input, where the residual
    stream starts. Where an add comes right before the norm, :func:`add_rms_norm_` fuses
    the two. Scaling is by ``weight``, or ``1 + weight`` with ``zero_centered=True``,
    in FP32. It runs the Triton residual-stream kernel shared with training and
    ``AddRMSNorm(provider="triton")(..., normalize="stored_sum")``. With the
    same weight convention, those stored-sum normalizations agree bitwise.
    ``residual`` is a CUDA ``[..., hidden]`` tensor with arbitrary row stride and
    unit hidden stride, in ``out_dtype`` or FP32. Other layouts visibly reroute
    through a contiguous temporary (reason "pack"); ``out_dtype`` is FP16, BF16 or FP32, and the
    weight is as for :func:`add_rms_norm_`. ``residual`` is not modified. The operation
    follows the current CUDA stream and has no backward. Empty inputs return empty
    output storage without launching a provider.
    """

    _require_frozen_weight("residual_rms_norm", weight)
    _require_no_autograd("residual_rms_norm", residual)
    if not residual.numel():
        last_standalone_provider["residual_rms_norm"] = "empty"
        return torch.empty_like(residual, dtype=out_dtype)
    return _standalone("residual_rms_norm", residual, weight, eps, zero_centered, out_dtype)


def _layer_norm_rows(name, tensor):
    hidden = tensor.shape[-1]
    try:
        rows = tensor.view(-1, hidden)
    except RuntimeError as error:
        raise ValueError(
            f"{name}'s leading dimensions must collapse into one row stride"
        ) from error
    if hidden > 1 and rows.stride(1) != 1:
        raise ValueError(f"{name} needs a unit hidden stride")
    if rows.shape[0] > 1 and rows.stride(0) < hidden:
        raise ValueError(f"{name}'s rows must not overlap")
    return rows


def add_layer_norm_(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    store_residual: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add and LayerNorm in place for inference, returning ``(x, residual)``.

    ``residual`` receives ``residual + x`` rounded once to its dtype, bitwise torch's add,
    unless ``store_residual=False`` leaves it unchanged (a stream's last norm). ``x`` receives
    the LayerNorm of that stored sum over the last dimension with FP32 statistics (biased
    variance), ``weight * (sum - mean) / sqrt(var + eps) + bias`` rounded once to x's dtype:
    nn.LayerNorm after a torch add, as Hugging Face, vLLM and SGLang compose it, to within
    rounding of the FP32 statistics.

    ``x`` and ``residual`` share one FP16 or BF16 dtype and a ``[..., hidden]`` shape whose
    leading dimensions collapse to one row stride over a unit-stride hidden dimension (rows
    may be strided). ``weight`` and ``bias`` are frozen contiguous ``[hidden]`` tensors in x's
    dtype or FP32. The address spans of ``x`` and ``residual``, row gaps included, must not
    overlap each other, ``weight`` or ``bias``. Validation precedes any launch; empty inputs
    return without one.

    CUDA runs one Triton kernel (CTI_NORM_PROVIDER does not apply); other devices run the
    reference composition, ``residual + x`` then F.layer_norm in FP32. Inputs to be modified
    must not require gradients; there is no backward. Execution follows the current CUDA
    stream; compile eagerly (one call per hidden size and dtypes) before graph capture.
    """

    if weight.requires_grad or bias.requires_grad:
        raise ValueError("add_layer_norm_ requires a frozen weight and bias")
    _require_mutable("add_layer_norm_", x, residual)
    if x.dtype not in (torch.float16, torch.bfloat16) or residual.dtype != x.dtype:
        raise TypeError("x and residual must share one FP16 or BF16 dtype")
    if weight.dtype not in (x.dtype, torch.float32) or bias.dtype not in (x.dtype, torch.float32):
        raise TypeError("weight and bias must use x's dtype or FP32")
    if x.ndim == 0 or not x.shape[-1] or residual.shape != x.shape:
        raise ValueError("residual must match x's shape, with a nonempty hidden dimension")
    hidden = x.shape[-1]
    if any(t.shape != (hidden,) or not t.is_contiguous() for t in (weight, bias)):
        raise ValueError("weight and bias must be contiguous [hidden] tensors")
    if any(t.device != x.device for t in (residual, weight, bias)):
        raise ValueError("x, residual, weight and bias must share one device")
    x_rows, residual_rows = _layer_norm_rows("x", x), _layer_norm_rows("residual", residual)
    spans = [_span(t) for t in (x_rows, residual_rows, weight, bias)]
    for written in range(2):
        for other in range(written + 1, 4):
            (lo, hi), (other_lo, other_hi) = spans[written], spans[other]
            if lo < hi and other_lo < other_hi and lo < other_hi and other_lo < hi:
                raise ValueError("x and residual must not overlap each other, weight or bias")
    if not x.numel():
        last_standalone_provider["add_layer_norm_"] = "empty"
        return x, residual
    if x.is_cuda:
        from .triton import _layer_norm

        with torch.cuda.device(x.device):
            _layer_norm.add_layer_norm_(
                x_rows, residual_rows, weight, bias, float(eps), store_residual
            )
        last_standalone_provider["add_layer_norm_"] = "triton"
    else:
        summed = residual + x
        x.copy_(F.layer_norm(summed.float(), (hidden,), weight.float(), bias.float(), eps))
        if store_residual:
            residual.copy_(summed)
        last_standalone_provider["add_layer_norm_"] = "torch"
    return x, residual


__all__ = [
    "warm_up_add_rms_norm",
    "AddRMSNorm",
    "add_layer_norm_",
    "add_rms_norm_",
    "residual_rms_norm",
    "rms_norm",
    "rms_norm_",
]
