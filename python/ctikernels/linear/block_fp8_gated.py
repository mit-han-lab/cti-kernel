# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen block-FP8 gated FC1, with gate-first [2I,K] weights.

The package BlockFP8Linear dispatch computes BF16/FP16 gate_up; the cute-fp8
activation variant fuses SiLU x up and 1x128 quantization. Returns E4M3 [M,I]
and FP32 [M,ceil(I/128)] scales, directly consumed by BlockFP8Linear.quantized.
Small M uses one CuTe fused FC1 (BF16 gate/up rounding, including FP16 input).
Optional FlashInfer full fusion is selected only in measured cells and requires
FlashInfer >= 0.7.1. DSV4 options always use local CuTe kernels.
Warm up every dtype/layout/batch on the capture stream before CUDA graph capture.
Calls/graphs sharing an instance must be stream-ordered; use separate instances
for concurrent streams. The quantized-input entry retains the existing GEMM path.

``workspace_bytes(max_m: int) -> int`` takes the maximum over M in 1..max_m
and both entries, including non-monotonic split-K scratch in the unfused GEMM.
``op(x, *, workspace=None)`` and ``op.quantized(q, scale, *,
dtype=torch.bfloat16, workspace=None)`` borrow contiguous, 16-byte aligned FP32
CUDA storage on the weight device. Validation matches BlockFP8Linear, before
any launch. Only internal gate/up, activation staging/scales, fused scratch/counters
and FlashInfer offsets use workspace views. Returned (q, scale), including padded
MN-major scales, are freshly allocated and never borrowed from workspace. Callers
own workspace lifetime and stream ordering, including graph replay. Stream-ordered
reuse leaves earlier outputs valid; capture still allocates the returned outputs.
Warm before capture. None preserves the existing allocations. No host sync.
"""

from __future__ import annotations

import contextlib

import torch

from ctikernels._runtime.launch import CallPlans, call_class, eager_on

from . import _block_fp8_dispatch, _block_fp8_gated_dispatch
from ._fp8_workspace import Layout, maximum
from .block_fp8 import (
    _ACTIVATION_DTYPES,
    BlockFP8Linear,
    _activation_rows,
    _check_activation,
    _check_tensor,
    _check_triton_capture_layout,
    _check_warmed_input,
    _check_weight,
    _cuda_capability,
    _groups,
    _input_layout_representatives,
    _layout_tensor,
    _require_frozen,
    _require_inference,
    _selected_input_layouts,
    _triton,
    _warm_up_sizes,
)
from .cute import _block_fp8_act_quant as act_quant
from .cute._block_fp8_fused_provider import FusedConfig, config_for
from .cute._block_fp8_provider import available

CUTE_ACT_QUANT = "cute-act-quant"
FLASHINFER_CUTE_DSL = "flashinfer-cute-dsl"
CUTE_FP8_FUSED = "cute-fp8-fused"
PROVIDERS = (CUTE_ACT_QUANT, FLASHINFER_CUTE_DSL, CUTE_FP8_FUSED)
HYBRID = f"{CUTE_ACT_QUANT}+{FLASHINFER_CUTE_DSL}"
_E4M3 = torch.float8_e4m3fn


def _check_gate_up(operation: str, gate_up: object) -> torch.Tensor:
    if not isinstance(gate_up, torch.Tensor):
        raise TypeError(f"{operation}: gate_up must be a tensor")
    if gate_up.dtype not in _ACTIVATION_DTYPES:
        raise TypeError(f"{operation}: gate_up must be BF16 or FP16, not {gate_up.dtype}")
    if not gate_up.is_cuda:
        raise ValueError(f"{operation}: gate_up must be on a CUDA device")
    if gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError(f"{operation}: gate_up must be [M, 2I], not {tuple(gate_up.shape)}")
    if gate_up.stride(1) != 1 and gate_up.numel() > 0:
        raise ValueError(f"{operation}: gate_up must have unit column stride")
    return gate_up


def silu_and_mul_quantize_fp8(
    gate_up: torch.Tensor,
    *,
    scale_layout: str = "k",
    swiglu_limit: float | None = None,
    scale_ue8m0: bool = False,
    outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    act: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch, preserving the materialized BF16/FP16 product rounding: CuTe PDL on
    sm120/sm121 with DSL 4.8.0, else our Triton kernel (compute capability 8.9+); both
    give the same bits.

    With swiglu_limit=L, clamp BF16 gate to <=L and BF16 up to [-L,L]
    before SiLU. UE8M0 uses ceil-to-power-of-two scales in FP32 storage.
    'mn' scales have a row capacity padded to four, but logical shape [M,groups].
    ``outputs`` (q E4M3 [M, I], scale FP32 in ``scale_layout``) are written in place when
    given (callers own their shapes); ``act`` [M, I] in ``gate_up``'s dtype, if given,
    also receives the rounded product (the Triton kernel serves this on every device).
    Warm each dtype/width/option set with one eager call before graph capture.
    """
    operation = "silu_and_mul_quantize_fp8"
    key = None
    if isinstance(gate_up, torch.Tensor) and isinstance(act, torch.Tensor | None):
        given = outputs if isinstance(outputs, tuple) else (None, None)
        if all(t is None or isinstance(t, torch.Tensor) for t in given):
            key = (
                call_class(gate_up),
                call_class(act),
                *map(call_class, given),
                outputs is None,
                scale_layout,
                swiglu_limit,
                scale_ue8m0,
            )
            plan = _PLANS.get(key)
            if plan is not None and _steady(gate_up):
                return plan(gate_up, outputs, act)
    gate_up = _check_gate_up(operation, gate_up)
    _require_inference(operation, gate_up)
    if act is None and available(_cuda_capability(gate_up.device)):
        return act_quant.silu_quantize(
            gate_up, scale_layout=scale_layout, swiglu_limit=swiglu_limit,
            scale_ue8m0=scale_ue8m0, outputs=outputs,
        )  # fmt: skip
    act_quant.check_options(swiglu_limit, scale_ue8m0, scale_layout)
    m, width = gate_up.shape[0], gate_up.shape[1] // 2
    q, scale = (
        act_quant.allocate_outputs(m, width, gate_up.device, scale_layout)
        if outputs is None
        else outputs
    )
    current = gate_up.get_device() == torch.cuda.current_device()
    with contextlib.nullcontext() if current else torch.cuda.device(gate_up.device):
        relaunch = _triton().silu_quantize(
            gate_up, q, scale, swiglu_limit=swiglu_limit, scale_ue8m0=scale_ue8m0, act=act,
        )  # fmt: skip
    if key is not None and relaunch is not None and _steady(gate_up):
        _PLANS.put(key, _plan(relaunch, m, width, gate_up.device, scale_layout))
    return q, scale


# Triton launches of validated eager calls, per call class (see ``_steady``).
_PLANS = CallPlans()


def _steady(gate_up) -> bool:
    """Whether a call may reuse its class's launch: eager on the current device, no autograd."""
    return eager_on(gate_up.get_device()) and not (
        torch.is_grad_enabled() and gate_up.requires_grad
    )


def _plan(relaunch, m, width, device, scale_layout):
    def plan(gate_up, outputs, act):
        q, scale = (
            act_quant.allocate_outputs(m, width, device, scale_layout)
            if outputs is None
            else outputs
        )
        relaunch(gate_up, q, scale, act)
        return q, scale

    return plan


class BlockFP8GateUpSiLU:
    """The quantized ``silu(x . W_gate^T) * (x . W_up^T)`` of one frozen block-FP8 merged
    gate/up weight (see the module docstring).

    ``BlockFP8GateUpSiLU(weight, weight_scale_inv, *, provider=None)``; ``op(x)`` and
    ``op.quantized(q, scale, *, dtype=torch.bfloat16)`` return ``(q, scale)``. ``dtype`` is
    the gate_up GEMM's output dtype in the unfused path (``op(x)`` uses ``x``'s).
    Fused FC1 always rounds gate/up to BF16. Explicit ``cute-fp8-fused`` extends
    its M ceiling to 16; larger M and quantized inputs use the existing path.
    SMALL_M selects measured provider/config runs; unlisted shapes fuse only M=1.
    The package default fused engine is FMA at M=1 and MMA at M=2..16. Pass
    ``fused_config=FusedConfig(engine="mma", warps_n=2, warps_k=4)`` (or
    ``engine="fma", r=8, warps=8``) to override the per-device/M table.
    MMA uses eight matched columns per row warp; CTA K splitting is unsupported.
    Off sm120/121 (or without DSL 4.8.0) the unfused path runs: ``BlockFP8Linear``'s default
    GEMM, then our Triton activation (``swiglu_limit`` and ``scale_ue8m0`` included).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        *,
        provider: str | None = None,
        scale_layout: str = "k",
        swiglu_limit: float | None = None,
        scale_ue8m0: bool = False,
        fused_config: FusedConfig | None = None,
    ) -> None:
        name = type(self).__name__
        weight, scale = _check_weight(name, weight, weight_scale_inv, ndim=2)
        if weight.shape[0] % 2:
            raise ValueError(f"{name}: weight must be [2I, K] (gate rows, then up rows)")
        if provider is not None and provider not in PROVIDERS:
            raise ValueError(f"{name}: provider must be one of {PROVIDERS}, not {provider!r}")
        act_quant.check_options(swiglu_limit, scale_ue8m0, scale_layout)
        if fused_config is not None and not isinstance(fused_config, FusedConfig):
            raise TypeError("fused_config must be FusedConfig")
        if provider == FLASHINFER_CUTE_DSL and (swiglu_limit is not None or scale_ue8m0):
            raise ValueError("flashinfer-cute-dsl does not implement the DSV4 options")
        self._warmed_inputs = set()
        self._scale_layout = scale_layout
        self._swiglu_limit = swiglu_limit
        self._scale_ue8m0 = scale_ue8m0
        self._weight, self._scale = weight, scale
        self._device = weight.device
        capability = _cuda_capability(self._device)
        sms = torch.cuda.get_device_properties(self._device).multi_processor_count
        self._measured_gemm = _block_fp8_dispatch.TABLE.get((*capability, sms), {}).get(
            tuple(weight.shape)
        )
        self._flashinfer = None
        self._fused = None
        self._fused_config = fused_config
        self._capability = capability
        self._sms = sms
        self._small_m = ()
        if provider is None:
            self._small_m = _block_fp8_gated_dispatch.small_m_buckets(
                capability, sms, self.intermediate_size, self.in_features
            )
        elif provider == CUTE_FP8_FUSED:
            if not _block_fp8_gated_dispatch.fused_supported(
                capability, self.intermediate_size, self.in_features
            ):
                raise ValueError("cute-fp8-fused: needs sm120/sm121 and I/H multiples of 128")
            self._small_m = ((16, CUTE_FP8_FUSED, None),)
        self._fused_configs = {}
        previous = 0
        for ceiling, choice, config in self._small_m:
            if choice == CUTE_FP8_FUSED:
                for m in range(previous + 1, ceiling + 1):
                    self._fused_configs[m] = (
                        fused_config
                        or config
                        or config_for(capability, self.intermediate_size, m, sms=sms)
                    )
            previous = ceiling
        self._fused_specs = {
            m: _block_fp8_gated_dispatch.config_spec(config)
            for m, config in self._fused_configs.items()
        }
        self._gate_up = None
        self._buckets = ((0, provider or CUTE_ACT_QUANT),)
        if provider == FLASHINFER_CUTE_DSL:
            self._flashinfer = self._flashinfer_provider(weight, scale, capability)
        else:
            # The activation runs in CuTe on sm120/121 with DSL 4.8.0, our Triton elsewhere.
            self._gate_up = BlockFP8Linear(weight, scale, scale_ue8m0=scale_ue8m0)
            buckets = _block_fp8_gated_dispatch.buckets(
                capability, sms, self.intermediate_size, self.in_features
            )
            small_flashinfer = any(name == FLASHINFER_CUTE_DSL for _, name, _ in self._small_m)
            if (
                provider is None
                and swiglu_limit is None
                and not scale_ue8m0
                and (buckets or small_flashinfer)
            ):
                try:
                    self._flashinfer = self._flashinfer_provider(weight, scale, capability)
                except (RuntimeError, ValueError):
                    pass
                else:
                    self._buckets = buckets or self._buckets
                    provider = HYBRID
        self._provider = provider or CUTE_ACT_QUANT
        if self._fused_configs:
            self._provider = CUTE_FP8_FUSED

    def _flashinfer_provider(self, weight, scale, capability):
        from .flashinfer._block_fp8_gated import FlashInferFc1ActQ1

        return FlashInferFc1ActQ1(
            weight, scale, capability=capability, operation=type(self).__name__
        )

    @property
    def provider(self) -> str:
        """Provider family; provider_for resolves dense-input M-dependent fallbacks."""
        return self._provider

    def provider_for(self, rows: int) -> str:
        """Dense-input implementation; quantized inputs retain the GEMM path."""
        if rows > 0:
            for ceiling, name, _ in self._small_m:
                if rows <= ceiling:
                    if name == CUTE_FP8_FUSED:
                        return name
                    if name == FLASHINFER_CUTE_DSL and self._flashinfer is not None:
                        return name
                    return f"{self._gate_up.provider_for(rows)}+{CUTE_ACT_QUANT}"
        return self._unfused_provider_for(rows)

    def resolve(self, m: int) -> tuple[str, str | None]:
        """Dense-input (provider, canonical fused spec); M is a nonnegative host int.

        Unfused calls report ``cute-act-quant``; ``provider_for`` also names their GEMM.
        Package defaults and explicit overrides are expanded to the actual fused spec.
        """
        if type(m) is not int or m < 0:
            raise ValueError("M must be a nonnegative integer")
        name = self.provider_for(m)
        if name == CUTE_FP8_FUSED:
            return name, self._fused_specs[m]
        return (FLASHINFER_CUTE_DSL if name == FLASHINFER_CUTE_DSL else CUTE_ACT_QUANT), None

    def _unfused_provider_for(self, rows):
        if self._gate_up is None:
            return FLASHINFER_CUTE_DSL
        return _block_fp8_gated_dispatch.resolved_provider(
            self._buckets, rows, self._gate_up.provider_for(rows), self._measured_gemm
        )

    @property
    def gate_up(self) -> BlockFP8Linear | None:
        """The gate_up GEMM; None for the FlashInfer FC1 provider."""
        return self._gate_up

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def in_features(self) -> int:
        return self._weight.shape[1]

    @property
    def intermediate_size(self) -> int:
        return self._weight.shape[0] // 2

    def _check_frozen(self) -> None:
        _require_frozen(
            type(self).__name__, {"weight": self._weight, "weight_scale_inv": self._scale}
        )

    def _workspace_layout(self, m, *, quantized=False):
        from .cute._block_fp8_fused_provider import workspace_shapes
        from .flashinfer._block_fp8_gated import padded_rows

        provider = self._unfused_provider_for(m) if quantized else self.provider_for(m)
        i, k = self.intermediate_size, self.in_features
        layout = Layout()
        if not quantized:
            layout.add("x", (m, k), torch.bfloat16)
        if not m:
            return layout
        if provider == CUTE_FP8_FUSED:
            hshape, cshape = workspace_shapes(i)
            layout.add("hidden", hshape, torch.bfloat16)
            layout.add("counters", cshape, torch.int32)
        elif provider == FLASHINFER_CUTE_DSL:
            layout.add("q", (m, k), _E4M3)
            layout.add("scale", (_groups(k), padded_rows(m)))
            layout.add("indptr", (2,), torch.int32)
        else:
            layout.add("gate_up", (m, 2 * i), torch.bfloat16)
            layout.include("gemm", self._gate_up._workspace_layout(m, quantized=quantized))
        return layout

    def _workspace_bytes(self, m):
        return max(self._workspace_layout(m, quantized=q).nbytes for q in (False, True))

    def workspace_bytes(self, max_m: int) -> int:
        """Maximum scratch bytes over M in 1..max_m and both entries; excludes outputs."""
        return maximum(max_m, self._workspace_bytes)

    def _run_workspace(self, x, scale, dtype, workspace):
        quantized = scale is not None
        m = x.numel() // self.in_features
        operands = [self._weight, self._scale]
        if self._gate_up is not None:
            operands.extend(self._gate_up._workspace_operands())
        for name in ("_fused", "_flashinfer"):
            provider = getattr(self, name, None)
            operands.extend(getattr(provider, field, None) for field in ("_weight", "_scale"))
        views = self._workspace_layout(m, quantized=quantized).bind(
            workspace, self.device, x, scale, *operands
        )
        x = x if quantized else views.flatten(x)
        provider = self._unfused_provider_for(m) if quantized else self.provider_for(m)
        layout = "mn" if provider == FLASHINFER_CUTE_DSL else self._scale_layout
        outputs = act_quant.allocate_outputs(m, self.intermediate_size, self.device, layout)
        q, sf = outputs
        if not m:
            return outputs
        if provider == CUTE_FP8_FUSED:
            return self._fused_provider().run(x, buffers=views, outputs=outputs)
        if provider == FLASHINFER_CUTE_DSL:
            kernel = _triton()

            xq, xs = views.get("q"), views.get("scale")
            if scale is None:
                _check_triton_capture_layout(x)
                kernel.quantize(x, xq, xs.t())
            else:
                xq.copy_(x)
                xs[:, :m].copy_(scale.t())
            offsets = views.get("indptr")
            offsets[:1].zero_()
            offsets[1:].fill_(m)
            # FlashInfer takes the full padded MN-major scale allocation.
            scale_mn = sf.t().as_strided(
                (_groups(self.intermediate_size), xs.shape[1]), (xs.shape[1], 1)
            )
            self._flashinfer.run(xq, xs, q, scale_mn, indptr=offsets)
            return outputs
        op = self._gate_up
        op._check_frozen()
        gemm = views.child("gemm", op._workspace_layout(m, quantized=quantized))
        gate_up = views.get("gate_up", dtype)
        op._run_workspace(x, scale, gate_up, gemm)
        return silu_and_mul_quantize_fp8(
            gate_up,
            scale_layout=self._scale_layout,
            swiglu_limit=self._swiglu_limit,
            scale_ue8m0=self._scale_ue8m0,
            outputs=outputs,
        )

    def __call__(
        self, x: torch.Tensor, *, workspace: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = type(self).__name__
        self._check_frozen()
        x = _check_activation(name, x, self._device, self.in_features)
        _require_inference(name, x)
        if workspace is not None:
            return self._run_workspace(x, None, x.dtype, workspace)
        rows = _activation_rows(self, x)
        return self._run(rows, None, x.dtype)

    def quantized(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        *,
        dtype: torch.dtype = torch.bfloat16,
        workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = type(self).__name__
        self._check_frozen()
        q = _check_tensor(name, "q", q, (_E4M3,), 2)
        scale = _check_tensor(name, "scale", scale, (torch.float32,), 2)
        k = self.in_features
        if q.device != self._device or scale.device != self._device:
            raise ValueError(f"{name}: q and scale must be on {self._device}")
        if q.shape[1] != k or tuple(scale.shape) != (q.shape[0], _groups(k)):
            raise ValueError(
                f"{name}: q must be [M, {k}] and scale [M, {_groups(k)}], not "
                f"{tuple(q.shape)} and {tuple(scale.shape)}"
            )
        if q.stride(1) != 1 and q.numel() > 0:
            raise ValueError(f"{name}: q must have unit column stride")
        if dtype not in _ACTIVATION_DTYPES:
            raise TypeError(f"{name}: dtype must be BF16 or FP16")
        _require_inference(name, q, scale)
        if workspace is not None:
            return self._run_workspace(q, scale, dtype, workspace)
        return self._run(q, scale, dtype)

    def _run(self, x, x_scale, dtype):
        m = x.shape[0]
        if m:
            _check_warmed_input(self, x, x_scale, dtype=dtype)
        i = self.intermediate_size
        provider = self.provider_for(m) if x_scale is None else self._unfused_provider_for(m)
        if provider == CUTE_FP8_FUSED:
            return self._fused_provider().run(x)
        if provider == FLASHINFER_CUTE_DSL:
            return self._run_flashinfer(x, x_scale)
        if not m:
            return act_quant.allocate_outputs(m, i, self._device, self._scale_layout)
        gate_up = (
            self._gate_up(x)
            if x_scale is None
            else self._gate_up.quantized(x, x_scale, out_dtype=dtype)
        )
        return silu_and_mul_quantize_fp8(
            gate_up,
            scale_layout=self._scale_layout,
            swiglu_limit=self._swiglu_limit,
            scale_ue8m0=self._scale_ue8m0,
        )

    def _fused_provider(self):
        self._check_frozen()
        if self._fused is None:
            from .cute._block_fp8_fused_provider import FusedFC1

            self._fused = FusedFC1(
                self._weight,
                self._scale,
                capability=self._capability,
                sms=self._sms,
                config=self._fused_config,
                configs=self._fused_configs,
                swiglu_limit=self._swiglu_limit,
                scale_ue8m0=self._scale_ue8m0,
                scale_layout=self._scale_layout,
            )
        return self._fused

    def _dispatch_key(self, m, *, dtype=torch.bfloat16, quantized=False):
        return dtype, quantized, self._provider_dispatch_key(m, dtype=dtype, quantized=quantized)

    def _provider_dispatch_key(self, m, *, dtype=torch.bfloat16, quantized=False):
        from .cute._block_fp8_fused_provider import compile_key

        provider = self._unfused_provider_for(m) if quantized else self.provider_for(m)
        if provider == CUTE_FP8_FUSED:
            return provider, compile_key(
                self._device,
                self._capability,
                self.intermediate_size,
                self.in_features,
                dtype,
                m,
                self._fused_configs[m],
                self._swiglu_limit,
                self._scale_ue8m0,
                self._scale_layout,
            )
        if provider == FLASHINFER_CUTE_DSL:
            # indptr is cached by exact M, not the padded scale-row count.
            return provider, m
        # The SiLU quantizer's signature is independent of M for packed gate_up.
        return provider, self._gate_up._dispatch_key(
            m, dtype=dtype, out_dtype=dtype, quantized=quantized
        )

    def _input_layout_key(self, m, layout, scale_layout=None, *, quantized=False):
        from .triton._block_fp8_provider import _int_key

        provider = self._unfused_provider_for(m) if quantized else self.provider_for(m)
        if provider == CUTE_FP8_FUSED:
            # Fused FC1 takes symbolic strides and scalar input loads.
            return ("cute-fused",)
        if provider == FLASHINFER_CUTE_DSL:
            if not quantized and m == 1:
                layout = self.in_features, layout[1]
            return ("staged-q",) if quantized else (_int_key(layout[0]), layout[1])
        return self._gate_up._input_layout_key(m, layout, scale_layout, quantized=quantized)

    def _input_layouts(self, m, *, quantized=False, input_layouts="packed"):
        # Share normalization and deduplication with the dense linear entry.
        return _input_layout_representatives(
            self, m, quantized=quantized, input_layouts=input_layouts
        )

    def warm_up(
        self,
        max_m=None,
        *,
        batch_sizes=None,
        dtype=torch.bfloat16,
        input_layouts="packed",
        quantized=False,
    ) -> list[dict]:
        """Compile selected routes before capture, returning config and input layout keys.

        Exactly one of max_m and batch_sizes is required. input_layouts defaults to
        packed; 'all' or explicit examples opt into other classes (see BlockFP8Linear).
        quantized=True warms the quantized entry; dtype is the gate/up output dtype.
        Unwarmed dispatch/layout pairs raise before launching during capture.
        """
        sizes = _warm_up_sizes(
            "BlockFP8GateUpSiLU",
            max_m,
            batch_sizes,
            lambda m: (
                self._dispatch_key(m, dtype=dtype, quantized=quantized),
                tuple(
                    key
                    for _, _, key in self._input_layouts(
                        m, quantized=quantized, input_layouts=input_layouts
                    )
                ),
            ),
        )
        return self.warmup(sizes, dtype=dtype, input_layouts=input_layouts, quantized=quantized)

    def warmup(self, ms, *, dtype=torch.bfloat16, input_layouts="packed", quantized=False):
        """Eagerly prepare the GEMM, input quantizer and fused act-quant for capture."""
        self._check_frozen()
        if dtype not in _ACTIVATION_DTYPES:
            raise TypeError("BlockFP8GateUpSiLU: dtype must be BF16 or FP16")
        ms = tuple(ms)
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BlockFP8GateUpSiLU: warmup must run before capture")
            _selected_input_layouts(self.in_features, input_layouts, m=1, quantized=quantized)
            routes = []
            provider_for = self._unfused_provider_for if quantized else self.provider_for
            dense_sizes = [
                m for m in ms if m and provider_for(m) not in (CUTE_FP8_FUSED, FLASHINFER_CUTE_DSL)
            ]
            if dense_sizes:
                self._gate_up.warm_up(
                    batch_sizes=dense_sizes,
                    dtype=dtype,
                    out_dtype=dtype,
                    input_layouts=input_layouts,
                    quantized=quantized,
                )
            for m in ms:
                if not m:
                    continue
                for layout, scale_layout, key in self._input_layouts(
                    m, quantized=quantized, input_layouts=input_layouts
                ):
                    x = _layout_tensor(
                        m,
                        self.in_features,
                        layout,
                        dtype=_E4M3 if quantized else dtype,
                        device=self._device,
                    )
                    if quantized:
                        scale = _layout_tensor(
                            m,
                            _groups(self.in_features),
                            scale_layout,
                            dtype=torch.float32,
                            device=self._device,
                            scale=True,
                        )
                        self.quantized(x, scale, dtype=dtype)
                    else:
                        self(x)
                    dispatch = self._dispatch_key(m, dtype=dtype, quantized=quantized)
                    self._warmed_inputs.add((dispatch, key))
                    route = {
                        "provider": provider_for(m),
                        "m": m,
                        "bucket": (m, m),
                        "config": dispatch,
                        "input_layout": key,
                        "dtype": dtype,
                        "quantized": quantized,
                    }
                    if m in dense_sizes:
                        route["gemm_provider"] = self._gate_up.provider_for(m)
                    routes.append(route)
            return routes

    def _run_flashinfer(self, x, x_scale):
        from .flashinfer._block_fp8_gated import padded_rows

        _block_fp8_triton = _triton()

        m, k = x.shape
        i = self.intermediate_size
        q = torch.empty((m, i), dtype=_E4M3, device=self._device)
        scale_mn = torch.empty(
            (_groups(i), padded_rows(m)), dtype=torch.float32, device=self._device
        )
        if not m:
            return q, scale_mn.t()[:m]
        x_scale_mn = torch.empty(
            (_groups(k), padded_rows(m)), dtype=torch.float32, device=self._device
        )
        if x_scale is None:
            _check_triton_capture_layout(x)
            x_q = torch.empty((m, k), dtype=_E4M3, device=self._device)
            _block_fp8_triton.quantize(x, x_q, x_scale_mn.t())
        else:
            x_q = x.contiguous()
            x_scale_mn[:, :m].copy_(x_scale.t())
        self._flashinfer.run(x_q, x_scale_mn, q, scale_mn)
        return q, scale_mn.t()[:m]


__all__ = [
    "BlockFP8GateUpSiLU",
    "CUTE_ACT_QUANT",
    "CUTE_FP8_FUSED",
    "FusedConfig",
    "FLASHINFER_CUTE_DSL",
    "HYBRID",
    "PROVIDERS",
    "silu_and_mul_quantize_fp8",
]
