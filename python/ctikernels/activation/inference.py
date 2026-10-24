# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import os
from functools import lru_cache

import torch

from ctikernels import _flashinfer_version
from ctikernels._norm_act_reroute import Provider, engine_checks, reroute
from ctikernels._runtime.launch import prime

from . import flashinfer
from ._dispatch import Config, GeneralConfig, config_for, device_class, parse_config

VARIABLE = "CTI_ACTIVATION_PROVIDER"


@lru_cache(maxsize=None)
def _hardware(device):
    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    return device_class(capability, props.multi_processor_count), capability


def _admitted(x, out):
    return (
        x.device.type == "cuda"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim > 0
        and x.shape[-1] > 0
        and x.shape[-1] % 16 == 0
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and (out is None or (out.is_contiguous() and out.data_ptr() % 16 == 0))
    )


def _flashinfer_runs(x, out):
    return (
        x.dtype in (torch.float16, torch.bfloat16)
        and x.is_contiguous()
        and x.shape[-1] * x.element_size() % 16 == 0
        and (out is None or out.is_contiguous())
    )


def _signature(tensor):
    return None if tensor is None else (tensor.device, tensor.shape, tensor.stride(), tensor.dtype)


class SiluAndMul:
    """Reusable inference gate with provider reporting and pinned capture plans.

    ``activation`` selects silu, gelu or gelu_tanh. Auto, in order: accurate CuTe
    on capability 12.x; else FlashInfer for FP16/BF16 inputs it runs when usable;
    else Triton SiLU or the torch pointwise GELU form as a visible reroute.
    Outside 12.x overrides are honored (triton, flashinfer) and forcing CuTe
    raises; FP32 never selects CuTe there. The SiLU table only
    chooses a faster configuration. General CuTe supports any width and
    collapsible row-strided layout. Other layouts are visibly packed
    (``provider.reason``); CTI_ENGINE_CHECKS rejects those copies before launch.
    On 12.x without the CuTe DSL, SiLU reroutes to Triton and GELU to FlashInfer.
    FP32 on 12.x always uses CuTe. Explicit tuned configs require aligned contiguous
    FP16/BF16 SiLU on capability major 12. Empty inputs never launch.

    ``warm_up`` compiles without launching CuTe and pins contiguous plans. Other
    layouts can reuse compiled vector lanes, or need an eager call before capture.
    Compilation failures raise; capture never compiles or switches providers.
    """

    def __init__(self, *, provider=None, config=None, activation="silu"):
        if provider not in (None, "auto", "cute", "flashinfer", "triton"):
            raise ValueError("provider must be auto, cute, flashinfer or triton")
        if config is not None and not isinstance(config, Config):
            raise TypeError("config must be Config")
        if activation not in ("silu", "gelu", "gelu_tanh"):
            raise ValueError("activation must be silu, gelu or gelu_tanh")
        if activation != "silu" and provider == "triton":
            raise ValueError("triton supports only silu")
        self.activation = activation
        self._override, self.config = provider, config
        self.provider = None
        self.dispatched_config = None
        self._plans = {}
        self._pinned = {}
        # Benchmarks can forbid resolution before it performs any CUDA setup.
        self._require_cached_plan = False

    def warm_up(self, max_m, *, width, dtype, device):
        """Compile and pin contiguous aligned plans for M=1..max_m.

        CuTe routes compile and launch each compiled kernel once (``_runtime.launch``);
        FlashInfer/Triton routes launch once per M and output variant on zeroed scratch to
        trigger their lazy compilation; no timing.

        ``width`` is the output width (input width is twice this). Returns maximal
        runs ``{"m": (lo, hi), "provider": ..., "config": name_or_None}`` using
        the same overrides/environment as calls, with and without an output buffer.
        General routes report a frozen ``cute_general_nN`` lane descriptor.
        Call eagerly before CUDA graph capture; other layouts need an eager call.
        """
        if max_m < 1:
            raise ValueError("max_m must be positive")
        if width < 1:
            raise ValueError("width must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "SiluAndMul.warm_up must run eagerly, outside CUDA graph capture"
                )
            x = torch.zeros((max_m, 2 * width), dtype=dtype, device=device)
            out = torch.zeros((max_m, width), dtype=dtype, device=device)
            routes = []
            for m in range(1, max_m + 1):
                xx, yy = x[:m], out[:m]
                provider, launch = self._plan(xx, None, width, yy.shape, pin=True)
                if provider == "cute":
                    prime(launch.kernel, launch, xx, yy)
                else:
                    launch(xx, None)
                out_provider, out_launch = self._plan(xx, yy, width, yy.shape, pin=True)
                if out_provider == "cute":
                    prime(out_launch.kernel, out_launch, xx, yy)
                else:
                    out_launch(xx, yy)
                config = getattr(launch, "config", None)
                name = None if config is None else config.name
                if routes and (routes[-1]["provider"], routes[-1]["config"]) == (provider, name):
                    routes[-1]["m"] = (routes[-1]["m"][0], m)
                else:
                    routes.append({"m": (m, m), "provider": provider, "config": name})
            return routes

    def __call__(self, x, out=None):
        _require_inference_only(self.activation + "_and_mul", x, out)
        if x.ndim == 0 or x.shape[-1] % 2:
            raise ValueError("x must have rank >= 1 and shape [..., 2 * width]")
        width = x.shape[-1] // 2
        shape = x.shape[:-1] + (width,)
        if out is not None:
            if out.shape != shape or out.dtype != x.dtype or out.device != x.device:
                raise ValueError("out must match output shape, dtype and device")
            # Disjoint contiguous views may share an allocation. For strided fallback
            # tensors conservatively reject shared storage, since gaps can interleave.
            if (
                x.numel()
                and out.numel()
                and x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
            ):
                if not (x.is_contiguous() and out.is_contiguous()) or (
                    x.data_ptr() < out.data_ptr() + out.numel() * out.element_size()
                    and out.data_ptr() < x.data_ptr() + x.numel() * x.element_size()
                ):
                    raise ValueError("out must not overlap input storage")
        if x.numel() == 0:
            self.provider, self.dispatched_config = "empty", None
            return torch.empty(shape, dtype=x.dtype, device=x.device) if out is None else out
        provider, launch = self._plan(x, out, width, shape)
        result = launch(x, out)
        self.provider = provider
        self.dispatched_config = getattr(launch, "config", None)
        return result

    def _plan(self, x, out, width, shape, *, pin=False):
        admitted = _admitted(x, out)  # Alignment must be rechecked for new storage.
        from .cute import available

        env = (
            os.environ.get("CTI_ACT_PROVIDER", os.environ.get(VARIABLE, "auto")),
            os.environ.get("CTI_ACTIVATION_CONFIG"),
            _flashinfer_version.allowed(),
            engine_checks(),
            available(),
        )
        key = (
            _signature(x),
            _signature(out),
            env,
            self._override,
            self.config,
            self.activation,
            admitted,
            _vector_lanes(x, out, width),
        )
        plan = self._pinned.get(key) or self._plans.get(key)
        if plan is None:
            if self._require_cached_plan and not pin:
                raise RuntimeError("resolution cache miss: warm the exact capture views first")
            plan = self._resolve(x, out, width, shape, admitted, env)
            if not pin and (
                x.device.type != "cuda" or not torch.cuda.is_current_stream_capturing()
            ):
                if len(self._plans) >= 128:
                    self._plans.clear()
                self._plans[key] = plan
        if pin:
            self._pinned[key] = plan
        return plan

    def _resolve(self, x, out, width, shape, admitted, env):
        # One ordered predicate, identical in eager and graph (auto only):
        # 1. capability 12.x (sm_120/121) -> CuTe, below;
        # 2. else FP16/BF16 that FlashInfer runs, when FlashInfer is usable;
        # 3. else Triton (SiLU) or the torch pointwise form (GELU kinds) as a
        #    visible reroute. CuTe is unqualified there, also when forced.
        override = self._override or env[0]
        if override not in ("auto", "cute", "flashinfer", "triton"):
            raise ValueError(f"{VARIABLE} must be auto, cute, flashinfer or triton")
        if x.is_cuda and _hardware(x.device)[1][0] != 12:
            if override == "cute":
                raise RuntimeError("CuTe activation is unqualified outside capability 12.x")
            if override == "triton":
                if self.activation != "silu":
                    raise ValueError("triton supports only silu")
                return self._triton(x, out, width)
            if override == "flashinfer" or (env[2] and _flashinfer_runs(x, out)):
                if override == "flashinfer":
                    _flashinfer_version.require_allowed()
                    if not _flashinfer_runs(x, out):
                        raise RuntimeError(
                            "FlashInfer activation requires contiguous FP16/BF16 inputs"
                            " with 16-byte rows"
                        )
                return "flashinfer", getattr(flashinfer, self.activation + "_and_mul")
            reason = "outside capability 12.x and " + (
                "FlashInfer cannot run this input" if env[2] else "FlashInfer is unusable"
            )
            if self.activation != "silu":
                return (
                    reroute(self.activation + "_and_mul", "torch", reason),
                    _torch_gated(self.activation),
                )
            actual, launch = self._triton(x, out, width)
            if actual.reason:
                reason += "; " + actual.reason
            return reroute("silu_and_mul", "triton", reason), launch
        # On 12.x FP32 ignores provider/config overrides, including FlashInfer's dtype limits.
        if x.dtype == torch.float32 and x.is_cuda:
            return _general_plan(x, out, self.activation, shape)
        if override == "triton":
            if self.activation != "silu":
                raise ValueError("triton supports only silu")
            return self._triton(x, out, width)
        if override == "flashinfer":
            _flashinfer_version.require_allowed()
            return "flashinfer", getattr(flashinfer, self.activation + "_and_mul")
        forced = (
            (self.config or (parse_config(env[1]) if env[1] else None))
            if override == "cute"
            else None
        )
        if forced is not None:
            if self.activation != "silu":
                raise ValueError("tuned configs support only silu")
            if width % forced.elems:
                raise ValueError(
                    f"CuTe SiLU elems={forced.elems} requires width divisible by {forced.elems}"
                )
            if not admitted or _hardware(x.device)[1][0] != 12:
                raise RuntimeError(
                    "forced cute config requires aligned contiguous FP16/BF16 on sm12x"
                )
        if not env[4]:
            reason = "CuTe activation requires nvidia-cutlass-dsl==4.8.0"
            if override == "cute" or x.dtype == torch.float32:
                raise RuntimeError(reason)
            if self.activation == "silu":
                provider = reroute("silu_and_mul", "triton", reason)
                actual, launch = self._triton(x, out, width)
                if actual.reason:
                    provider = reroute("silu_and_mul", "triton", reason + "; " + actual.reason)
                return provider, launch
            if not env[2]:
                raise RuntimeError(reason + "; no usable FlashInfer fallback")
            return (
                reroute(self.activation + "_and_mul", "flashinfer", reason),
                getattr(flashinfer, self.activation + "_and_mul"),
            )
        if self.activation == "silu" and admitted:
            kind, capability = _hardware(x.device)
            config = forced or config_for(
                kind, width, str(x.dtype).removeprefix("torch."), x.numel() // (2 * width)
            )
            if capability[0] == 12 and config is not None:
                return "cute", self._cute(x, width, shape, config, capability)
        return _general_plan(x, out, self.activation, shape)

    @staticmethod
    def _triton(x, out, width):
        # Keep the existing inference fallback without routing through training. The
        # kernel takes one row stride for x and for out; other layouts are packed first.
        from ._silu_and_mul import row_stride, silu_and_mul_forward

        if (
            not x.is_cuda
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or width == 0
        ):

            def pointwise(x, out):
                gate, up = x.chunk(2, dim=-1)
                result = torch.nn.functional.silu(gate) * up
                return result if out is None else out.copy_(result)

            return Provider("triton"), pointwise
        packed_x = row_stride(x) is None
        packed_out = out is not None and row_stride(out) is None
        if not (packed_x or packed_out):
            return Provider("triton"), silu_and_mul_forward
        reason = "layout without one row stride and a unit last stride is packed"
        provider = reroute("silu_and_mul", "triton", reason)

        def launch(x, out):
            x = x.contiguous() if packed_x else x
            if out is None or not packed_out:
                return silu_and_mul_forward(x, out)
            return out.copy_(silu_and_mul_forward(x))

        return provider, launch

    def _cute(self, x, width, shape, config, capability):
        from .cute import get_compiled

        m = x.numel() // (2 * width)
        dtype = str(x.dtype).removeprefix("torch.")
        with torch.cuda.device(x.device):
            compiled = get_compiled(
                x.device,
                dtype,
                width,
                config,
                capability,
                capturing=torch.cuda.is_current_stream_capturing(),
                strict=True,
                capture_miss="raise",
            )

        def launch(x, out):
            if out is None:
                out = torch.empty(shape, dtype=x.dtype, device=x.device)
            with torch.cuda.device(x.device):
                compiled(
                    x if x.ndim == 2 else x.view(m, 2 * width),
                    out if out.ndim == 2 else out.view(m, width),
                )
            return out

        launch.config, launch.kernel = config, compiled
        return launch


def _torch_gated(kind):
    # Pointwise reroute outside capability 12.x when neither CuTe nor FlashInfer
    # applies: FP32 math (erf or cubic tanh form), one rounding to the output dtype.
    approximate = "tanh" if kind == "gelu_tanh" else "none"

    def launch(x, out):
        gate, up = x.float().chunk(2, dim=-1)
        result = (torch.nn.functional.gelu(gate, approximate=approximate) * up).to(x.dtype)
        return result if out is None else out.copy_(result)

    return launch


_DEFAULT_SILU = SiluAndMul()
_DEFAULT_GELU = SiluAndMul(activation="gelu")
_DEFAULT_GELU_TANH = SiluAndMul(activation="gelu_tanh")


def warm_up_silu_and_mul(max_m, **kw):
    """Compile and pin shared silu_and_mul plans; see :meth:`SiluAndMul.warm_up`."""
    return _DEFAULT_SILU.warm_up(max_m, **kw)


def warm_up_gelu_and_mul(max_m, **kw):
    """Compile and pin shared gelu_and_mul plans; see :meth:`SiluAndMul.warm_up`."""
    return _DEFAULT_GELU.warm_up(max_m, **kw)


def warm_up_gelu_tanh_and_mul(max_m, **kw):
    """Compile and pin shared tanh GELU plans; see :meth:`SiluAndMul.warm_up`."""
    return _DEFAULT_GELU_TANH.warm_up(max_m, **kw)


def _require_inference_only(
    operation: str,
    x: torch.Tensor,
    out: torch.Tensor | None,
) -> None:
    if out is not None and out.requires_grad:
        raise RuntimeError(f"{operation} cannot write to output that requires gradients")
    if x.requires_grad and torch.is_grad_enabled():
        raise RuntimeError(f"{operation} does not support autograd")


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Inference SwiGLU on CUDA FP16/BF16/FP32 ``[..., 2 * width]``.

    FP32 math rounds once on store. Native CuTe accepts any width and collapsible
    row-strided x/out with unit last stride. Other layouts visibly pack/copy and
    raise under CTI_ENGINE_CHECKS. Output must match shape/dtype/device and cannot
    overlap input. Autograd is rejected; empty inputs never launch.
    ``CTI_ACTIVATION_PROVIDER`` (or higher-priority ``CTI_ACT_PROVIDER``) selects
    auto, cute, flashinfer or triton. See SiluAndMul for the auto selection,
    for fallback and forced-config rules. Warm before CUDA graph capture.
    """
    return _DEFAULT_SILU(x, out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Inference erf GeGLU; layout/output/capture contracts match silu_and_mul.

    Auto uses accurate CuTe FP32 math for FP16/BF16/FP32, with one output rounding.
    Use warm_up_gelu_and_mul before capture. Triton is not supported.
    """
    return _DEFAULT_GELU(x, out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Inference cubic tanh GeGLU; contracts and providers match gelu_and_mul.

    Use warm_up_gelu_tanh_and_mul before capture.
    """
    return _DEFAULT_GELU_TANH(x, out)


def relu2(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Square positive elements, preserving shape; CUDA FP16/BF16/FP32 CuTe.

    SGLang's strict positive comparison maps NaN and negative zero to +0.
    Square in FP32, round once on output. Any width and rank >= 1 is accepted.
    Output, autograd and capture contracts match :func:`gelu_and_mul`.
    An output view of exactly the same elements as ``x`` is also supported.
    """
    return _activation(x, out, "relu2")


def _activation(x, out, kind):
    operation = kind if kind == "relu2" else kind + "_and_mul"
    _require_inference_only(operation, x, out)
    if x.ndim == 0 or (kind != "relu2" and x.shape[-1] % 2):
        raise ValueError("x must have rank >= 1 and gated inputs need shape [..., 2 * width]")
    shape = x.shape if kind == "relu2" else x.shape[:-1] + (x.shape[-1] // 2,)
    if out is not None:
        if out.shape != shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError("out must match output shape, dtype and device")
        same_elements = (
            kind == "relu2" and out.data_ptr() == x.data_ptr() and out.stride() == x.stride()
        )
        if (
            x.numel()
            and out.numel()
            and not same_elements
            and x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
        ):
            if not (x.is_contiguous() and out.is_contiguous()) or (
                x.data_ptr() < out.data_ptr() + out.numel() * out.element_size()
                and out.data_ptr() < x.data_ptr() + x.numel() * x.element_size()
            ):
                raise ValueError("out must not overlap input storage")
    if x.numel() == 0:
        return torch.empty(shape, dtype=x.dtype, device=x.device) if out is None else out
    return _cute_activation(x, out, kind, shape)


def _general_compiled(x, width, kind, lanes):
    from .cute._provider import get_general_compiled

    with torch.cuda.device(x.device):
        return get_general_compiled(
            x.device,
            str(x.dtype).removeprefix("torch."),
            width,
            kind,
            lanes,
            _hardware(x.device)[1],
            capturing=torch.cuda.is_current_stream_capturing(),
        )


def _row_stride(tensor):
    """Return the stride of a no-copy [-1, columns] view, or None."""
    if tensor.stride(-1) != 1:
        return None
    try:
        return tensor.view(-1, tensor.shape[-1]).stride(0)
    except RuntimeError:
        return None


def _vector_lanes(x, out, width):
    # Packed temporaries and newly allocated outputs have at least 16-byte alignment.
    xs = _row_stride(x)
    ys = None if out is None else _row_stride(out)
    xptr = 0 if xs is None else x.data_ptr()
    yptr = 0 if out is None or ys is None else out.data_ptr()
    xs = x.shape[-1] if xs is None else xs
    ys = width if ys is None else ys
    itemsize = x.element_size()
    lanes = 16 // itemsize
    while lanes > 1:
        if (
            width % lanes == xs % lanes == ys % lanes == 0
            and xptr % (lanes * itemsize) == yptr % (lanes * itemsize) == 0
        ):
            break
        lanes //= 2
    return lanes


def _general_plan(x, out, kind, shape):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("CuTe activation requires CUDA FP16/BF16/FP32")
    packed_x = _row_stride(x) is None
    packed_out = out is not None and _row_stride(out) is None
    provider = Provider("cute")
    if packed_x or packed_out:
        provider = reroute(
            kind if kind == "relu2" else kind + "_and_mul",
            "cute",
            "layout without one row stride and a unit last stride is packed",
        )
    width = shape[-1]
    lanes = _vector_lanes(x, out, width)
    compiled = _general_compiled(x, width, kind, lanes)

    def launch(x, out):
        # contiguous() can preserve non-unit strides on singleton dimensions.
        source = x.clone(memory_format=torch.contiguous_format) if packed_x else x
        target = (
            out
            if out is not None and not packed_out
            else torch.empty(shape, dtype=x.dtype, device=x.device)
        )
        m = target.numel() // width
        with torch.cuda.device(x.device):
            compiled(source.view(m, source.shape[-1]), target.view(m, width))
        return target if out is None or target is out else out.copy_(target)

    launch.config, launch.kernel = GeneralConfig(lanes), compiled
    return provider, launch


def _cute_activation(x, out, kind, shape):
    _, launch = _general_plan(x, out, kind, shape)
    return launch(x, out)


__all__ = [
    "warm_up_silu_and_mul",
    "warm_up_gelu_and_mul",
    "warm_up_gelu_tanh_and_mul",
    "SiluAndMul",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "relu2",
    "silu_and_mul",
]
