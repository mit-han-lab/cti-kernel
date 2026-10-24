# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Static per-tensor FP8 inference with scales in engine slot order.

Weights are E4M3 [N,K], with FP32 weight_scale[S] and one FP32 input_scale.
slot_rows partitions N into logical siblings, preserving unequal scales exactly:
q = E4M3_RN(clamp(fp32(x) / input_scale, -448, 448)); each slot's output is
(q @ weight_slot.T) * input_scale * weight_scale[slot], rounded to BF16/FP16.
NaN stays NaN and infinities saturate before conversion. There is no backward.

FP8PT owns preparation, static activation quantization and dispatch only.
GEMMs use cuBLASLt or FlashInfer; block-fp8 reuses existing block-FP8 kernels.
No FP8PT-specific GEMM/GEMV kernels are maintained.
Default/table routes use one GEMM: unaligned slot blocks take rowwise when supported
(BF16 only, K and N multiples of 16); otherwise block-fp8 runs one GEMM per slot.
Aligned multi-GEMM library routes take block-fp8. Explicit providers
are unchanged. warm_up prepares instance-owned row/block scales, retaining older
buffers for graph replay so covered quantized calls need no scale-copy launch.
Unavailable table choices reroute visibly to block-fp8 (or raise under
CTI_ENGINE_CHECKS); failed rowwise construction also reports its reason.
Explicit unavailable choices raise.

out= accepts BF16/FP16 [M,N], unit column stride and row stride >= N, returned
by identity. Libraries use direct aligned contiguous output where supported,
otherwise temporary output plus copy. Calls never read scales on the host.
Construction and warm_up precede capture. Tensor mutation is undefined;
requires_grad is checked on every entry, including retained scales.
CUDA compute capability 8.9 or newer is required. Quantization uses CuTe on
sm12x with CuTe DSL 4.8.0, otherwise the bitwise-equivalent Triton quantizer.
Optional frozen BF16/FP16 bias[N] adds one elementwise launch after the GEMM:
round_out(round_out(GEMM) + bias), broadcasting across rows.

``workspace_bytes(max_m: int) -> int`` bounds both entries and output dtypes
over every M in 1..max_m. It uses the actual static/inner dispatch: split-K
partials are non-monotonic, so sizing only at max_m is insufficient.
``op(x, *, out_dtype=None, out=None, workspace=None)`` and
``op.quantized(q, *, out_dtype=None, out=None, workspace=None)`` accept contiguous,
16-byte aligned FP32 CUDA storage on the weight device. Size, dtype, device,
contiguity, alignment, no gradient and no storage overlap with operands are
checked before launches, without host synchronization. The bound includes q,
block/row scales, inner scratch, library workspace and layout/output staging.
Outputs are never borrowed from workspace; without out= they are freshly
allocated with torch.empty. Callers own workspace lifetime and stream ordering,
including graph replay. Stream-ordered reuse leaves earlier outputs valid. Pass
out= for allocation-free capture. Warm on the capture stream before capture. None retains the
existing allocation behavior. Torch library routes require _scaled_mm out support.

Design reference: SGLang 72730d6 (python/sglang/srt/layers/quantization/modelopt_quant.py)
Design reference: vLLM 0.31.0 (vllm/_custom_ops.py)
SGLang uses ModelOptFp8LinearMethod and vLLM uses static_scaled_fp8_quant.
Their market ModelOpt paths requantize siblings to the maximum weight scale;
this implementation instead retains each slot's original scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ctikernels._reroute import reroute

from . import _fp8_static_dispatch
from ._fp8_workspace import Layout, maximum
from .block_fp8 import (
    BlockFP8Linear,
    _check_activation,
    _check_tensor,
    _cuda_capability,
    _groups,
    _LinearOutput,
    _require_frozen,
    _require_inference,
    _warm_up_sizes,
)
from .cute._block_fp8_provider import CONFIGS
from .cute._block_fp8_provider import available as cute_available
from .flashinfer._fp8_static import FlashInferBMM
from .torch._fp8_static import OUTPUT_DTYPES, TorchScaledMM

PROVIDERS = (
    "block-fp8",
    "torch-scaled-mm",
    "torch-scaled-mm:rowwise",
    "torch-scaled-mm:slots",
    "flashinfer-bmm:cublas",
    "flashinfer-bmm:cutlass",
    "flashinfer-bmm:auto",
)
_INNER_ATTRS = {
    "triton-fp8": "_triton",
    "flashinfer-cutlass": "_flashinfer",
    "cuda-sm12x": "_sm12x",
    "cute-fp8": "_cute",
}


def _check_device(name, device):
    if device.type != "cuda" or _cuda_capability(device) < (8, 9):
        raise RuntimeError(f"{name} needs CUDA compute capability 8.9 or newer")


def _resolve_quantizer(quantizer, device):
    if quantizer is not None and (
        not isinstance(quantizer, str) or quantizer not in ("cute", "triton")
    ):
        raise ValueError(f"invalid static FP8 quantizer: {quantizer!r}")
    available = cute_available(_cuda_capability(device))
    if quantizer == "cute" and not available:
        raise ValueError("static FP8 quantizer 'cute' needs sm12x and CuTe DSL 4.8.0")
    return quantizer or ("cute" if available else "triton")


def _quantizer(choice):
    if choice == "cute":
        from .cute._fp8_static_quant import quantize
    else:
        from .triton._fp8_static_quant import quantize
    return quantize


def _check_input_scale(name, scale, device):
    ndim = scale.ndim if isinstance(scale, torch.Tensor) else 0
    scale = _check_tensor(name, "input_scale", scale, (torch.float32,), ndim)
    if scale.numel() != 1:
        raise ValueError(f"{name}: input_scale must have exactly one element")
    if scale.device != device:
        raise ValueError(f"{name}: input_scale must be on {device}")
    return scale


def _check_weight(name, weight, weight_scale, input_scale, slot_rows):
    _require_frozen(
        name, {"weight": weight, "weight_scale": weight_scale, "input_scale": input_scale}
    )
    weight = _check_tensor(name, "weight", weight, (torch.float8_e4m3fn,), 2)
    ndim = weight_scale.ndim if isinstance(weight_scale, torch.Tensor) else 1
    weight_scale = _check_tensor(name, "weight_scale", weight_scale, (torch.float32,), ndim)
    if ndim not in (0, 1) or not weight_scale.numel():
        raise ValueError(f"{name}: weight_scale must be scalar or nonempty [S]")
    n, k = weight.shape
    if not n or not k:
        raise ValueError(f"{name}: weight must not be empty")
    if weight.stride(1) != 1:
        raise ValueError(f"{name}: weight must have unit stride along K")
    if weight_scale.device != weight.device:
        raise ValueError(f"{name}: weight_scale must be on {weight.device}")
    _check_input_scale(name, input_scale, weight.device)
    if slot_rows is None:
        if weight_scale.numel() != 1:
            raise ValueError(f"{name}: slot_rows is required for multiple scales")
        slot_rows = (n,)
    if not isinstance(slot_rows, Sequence):
        raise TypeError(f"{name}: slot_rows must be a sequence of positive Python ints")
    slot_rows = tuple(slot_rows)
    if any(type(rows) is not int or rows <= 0 for rows in slot_rows):
        raise ValueError(f"{name}: slot_rows must contain positive Python ints")
    if len(slot_rows) != weight_scale.numel() or sum(slot_rows) != n:
        raise ValueError(f"{name}: slot_rows must have S entries summing to N={n}")
    return slot_rows


def _block_grids(weight_scale, slot_rows, k):
    """Pure tensor construction on CPU or CUDA: (weight row slice, packed scale grid)."""
    scales = weight_scale.reshape(-1)
    grids = tuple(
        scales[i : i + 1].expand(_groups(rows), _groups(k)) for i, rows in enumerate(slot_rows)
    )
    if all(rows % 128 == 0 for rows in slot_rows[:-1]):
        return ((slice(0, sum(slot_rows)), torch.cat(grids, dim=0).contiguous()),)
    result = []
    start = 0
    for rows, grid in zip(slot_rows, grids, strict=True):
        result.append((slice(start, start + rows), grid.contiguous()))
        start += rows
    return tuple(result)


def quantize_static_fp8(
    x: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    block_scale: bool = False,
    row_scale: bool = False,
    pdl: bool = False,
    quantizer: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantize E4M3 [M,K], optionally with block [M,ceil(K/128)] or row [M,1] scales.

    pdl enables programmatic launch dependencies for a following PDL CuTe GEMM.
    It changes scheduling only; quantized bytes and scales are identical.
    """
    name = "quantize_static_fp8"
    _require_frozen(name, {"input_scale": input_scale})
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name}: x must be a tensor")
    x = _check_activation(name, x, x.device, x.shape[-1] if x.ndim else -1)
    _require_inference(name, x)
    if x.shape[-1] == 0:
        raise ValueError(f"{name}: K must be positive")
    _check_input_scale(name, input_scale, x.device)
    if not isinstance(block_scale, bool):
        raise TypeError(f"{name}: block_scale must be bool")
    if not isinstance(row_scale, bool):
        raise TypeError(f"{name}: row_scale must be bool")
    if block_scale and row_scale:
        raise ValueError(f"{name}: block_scale and row_scale are mutually exclusive")
    if not isinstance(pdl, bool):
        raise TypeError(f"{name}: pdl must be bool")
    _check_device(name, x.device)
    quantize = _quantizer(_resolve_quantizer(quantizer, x.device))

    return quantize(
        x.reshape(-1, x.shape[-1]),
        input_scale,
        block_scale=block_scale,
        row_scale=row_scale,
        pdl=pdl,
    )


class StaticFP8Linear(_LinearOutput):
    """Frozen static FP8 weight with one scale per logical slot; see module contract."""

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        *,
        slot_rows: Sequence[int] | None = None,
        provider: str | None = None,
        quantizer: str | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        name = type(self).__name__
        _require_frozen(name, {"bias": bias})
        self._slot_rows = _check_weight(name, weight, weight_scale, input_scale, slot_rows)
        if provider is not None:
            _fp8_static_dispatch.parse_provider(provider)
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or bias.dtype not in OUTPUT_DTYPES:
                raise ValueError(f"{name}: bias must be BF16 or FP16 [N]")
            if bias.shape != (weight.shape[0],) or bias.stride(0) != 1:
                raise ValueError(f"{name}: bias must be [N] with unit stride")
            if bias.device != weight.device:
                raise ValueError(f"{name}: bias must be on the weight device")
        self._bias = bias
        _check_device(name, weight.device)
        self._quantizer = _resolve_quantizer(quantizer, weight.device)
        with torch.cuda.device(weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{name}: construct and warm up before capture")
        self._weight, self._weight_scale, self._input_scale = weight, weight_scale, input_scale
        self._device = weight.device
        capability = _cuda_capability(self._device)
        sms = torch.cuda.get_device_properties(self._device).multi_processor_count
        runs = (
            ((0, provider),)
            if provider is not None
            else _fp8_static_dispatch.buckets(capability, sms, *weight.shape)
        )
        self._libraries = {}
        self._buckets_by_dtype = {}
        errors = {}
        for requested in {p for _, p in runs}:
            outer, inner, _ = _fp8_static_dispatch.parse_provider(requested)
            if outer == "block-fp8":
                continue
            try:
                if outer == "torch-scaled-mm":
                    op = TorchScaledMM(
                        weight, weight_scale, input_scale, self._slot_rows, mode=inner or "auto"
                    )
                else:
                    op = FlashInferBMM(
                        weight,
                        weight_scale,
                        input_scale,
                        self._slot_rows,
                        capability=capability,
                        backend=inner,
                    )
                self._libraries[requested] = op
            except (RuntimeError, ValueError, ImportError) as error:
                errors[requested] = str(error)
        for dtype in OUTPUT_DTYPES:
            resolved = []
            for last, requested in runs:
                op = self._libraries.get(requested)
                error = errors.get(requested)
                if isinstance(op, TorchScaledMM):
                    error = op.errors.get(dtype)
                if error is not None:
                    if provider is not None:
                        raise ValueError(
                            f"{name}: provider {provider!r} unavailable for {dtype}: {error}"
                        )
                    chosen = reroute(
                        name, "block-fp8", f"table provider {requested!r} unavailable: {error}"
                    )
                else:
                    chosen = op.providers[dtype] if isinstance(op, TorchScaledMM) else requested
                resolved.append((last, chosen))
            self._buckets_by_dtype[dtype] = tuple(resolved)
        # Keep each resolved name's prepared implementation. An alias can resolve
        # differently by output dtype, with both decisions made before capture.
        self._resolved_libraries = {}
        for op in self._libraries.values():
            if isinstance(op, TorchScaledMM):
                for dtype, chosen in op.providers.items():
                    self._resolved_libraries[chosen, dtype] = op
        for requested, op in self._libraries.items():
            if not isinstance(op, TorchScaledMM):
                for dtype in OUTPUT_DTYPES:
                    self._resolved_libraries[requested, dtype] = op
        if provider is None:
            self._single_launch(weight, weight_scale, input_scale)
        # Static row/block activation scales for the prequantized entry, prepared by
        # warm_up so that quantized(q) launches only the GEMM (older buffers are kept
        # alive for graphs captured against them).
        self._static_scales = {}
        self._retired_scales = []
        wanted = {p for buckets in self._buckets_by_dtype.values() for _, p in buckets}
        parsed = {p: _fp8_static_dispatch.parse_provider(p) for p in wanted}
        extras = tuple(
            sorted(
                {
                    inner
                    for outer, inner, _ in parsed.values()
                    if outer == "block-fp8" and inner is not None
                }
            )
        )
        self._blocks = ()
        if any(outer == "block-fp8" for outer, _, _ in parsed.values()):
            self._blocks = tuple(
                (columns, BlockFP8Linear(weight[columns], grid, _extra_providers=extras))
                for columns, grid in _block_grids(weight_scale, self._slot_rows, self.in_features)
            )
        unavailable = {
            p
            for p, (outer, inner, _) in parsed.items()
            if outer == "block-fp8"
            and inner is not None
            and any(getattr(op, _INNER_ATTRS[inner]) is None for _, op in self._blocks)
        }
        if provider in unavailable:
            raise ValueError(f"{name}: provider {provider!r} is unavailable on this device/layout")
        self._buckets_by_dtype = {
            dtype: tuple(
                (
                    last,
                    reroute(
                        name,
                        "block-fp8",
                        f"table provider {p!r} unavailable on this device/layout",
                    )
                    if p in unavailable
                    else p,
                )
                for last, p in buckets
            )
            for dtype, buckets in self._buckets_by_dtype.items()
        }
        self._buckets = self._buckets_by_dtype[torch.bfloat16]
        self._provider = self.provider_for(1)

    def _single_launch(self, weight, weight_scale, input_scale):
        """Resolve table/default routes that would need one GEMM per slot to one GEMM.

        One predicate on every device: when a slot boundary is not 128-aligned, every
        route that would issue more than one GEMM (block routes, and library routes with
        unequal slot scales) takes cuBLASLt rowwise (per-channel weight scales) where it
        is available for the output dtype; no table row overrides this. With aligned
        slots, library routes that need one GEMM per slot take block-fp8 (one GEMM).
        Otherwise the per-slot form stays.
        """
        split = any(rows % 128 for rows in self._slot_rows[:-1])
        rowwise = None
        rowwise_error = None
        if split and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0:
            try:
                rowwise = TorchScaledMM(
                    weight, weight_scale, input_scale, self._slot_rows, mode="rowwise"
                )
            except (RuntimeError, ValueError, ImportError) as error:
                rowwise_error = str(error)
        for dtype, buckets in self._buckets_by_dtype.items():
            resolved = []
            for last, original in buckets:
                chosen = original
                outer = _fp8_static_dispatch.parse_provider(chosen)[0]
                library = self._resolved_libraries.get((chosen, dtype))
                multi = outer == "block-fp8" or (
                    library is not None
                    and not chosen.endswith(":rowwise")
                    and len(library._slots) > 1
                )
                if (
                    split
                    and multi
                    and rowwise is not None
                    and rowwise.errors.get(dtype) is None
                    and rowwise.providers[dtype] == "torch-scaled-mm:rowwise"
                ):
                    chosen = "torch-scaled-mm:rowwise"
                    self._resolved_libraries[chosen, dtype] = rowwise
                elif split and multi and rowwise_error is not None:
                    chosen = reroute(
                        type(self).__name__,
                        chosen,
                        f"rowwise promotion unavailable: {rowwise_error}",
                    )
                elif not split and library is not None and multi:
                    chosen = "block-fp8"
                if chosen != original and getattr(original, "reason", None):
                    chosen = reroute(type(self).__name__, chosen, original.reason)
                resolved.append((last, chosen))
            self._buckets_by_dtype[dtype] = tuple(resolved)

    def _prepare_static_scales(self, m):
        """Before capture: packed [m, ceil(K/128)] and [m, 1] FP32 copies of input_scale."""
        for kind, columns in (("block", _groups(self.in_features)), ("row", 1)):
            current = self._static_scales.get(kind)
            if current is not None and current.shape[0] >= m:
                continue
            if current is not None:
                self._retired_scales.append(current)
            self._static_scales[kind] = (
                self._input_scale.reshape(1, 1).expand(m, columns).contiguous()
            )

    def _static_scale(self, kind, m):
        scale = self._static_scales.get(kind)
        return scale[:m] if scale is not None and scale.shape[0] >= m else None

    @property
    def quantizer(self) -> str:
        return self._quantizer

    @property
    def bias(self) -> torch.Tensor | None:
        return self._bias

    @property
    def provider(self) -> str:
        """Resolved BF16 provider at M=1; use provider_for for other sizes/dtypes."""
        return self._provider

    def provider_for(self, m: int, out_dtype: torch.dtype = torch.bfloat16) -> str:
        """Resolved provider for M and output dtype (BF16 by default)."""
        return _fp8_static_dispatch.provider(self._buckets_by_dtype[out_dtype], m)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def in_features(self) -> int:
        return self._weight.shape[1]

    @property
    def out_features(self) -> int:
        return self._weight.shape[0]

    @property
    def slot_rows(self) -> tuple[int, ...]:
        return self._slot_rows

    @property
    def input_scale(self) -> torch.Tensor:
        return self._input_scale

    @property
    def weight_scale(self) -> torch.Tensor:
        return self._weight_scale

    def _check_frozen(self):
        _require_frozen(
            type(self).__name__,
            {
                "weight": self._weight,
                "weight_scale": self._weight_scale,
                "input_scale": self._input_scale,
                "bias": self._bias,
            },
        )

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_frozen()
        _check_activation(type(self).__name__, x, self.device, self.in_features)
        return quantize_static_fp8(x, self.input_scale, quantizer=self.quantizer)

    def _workspace_layout(self, m, out_dtype=torch.bfloat16, *, quantized=False):
        chosen = self.provider_for(m, out_dtype)
        outer, inner, config = _fp8_static_dispatch.parse_provider(chosen)
        layout = Layout()
        if not quantized:
            layout.add("x", (m, self.in_features), torch.bfloat16)
        layout.add("q", (m, self.in_features), torch.float8_e4m3fn)
        if not m:
            return layout
        if outer == "block-fp8":
            layout.add("scale", (m, _groups(self.in_features)))
            size = max(
                op._workspace_layout(m, quantized=True, provider=inner, config=config).nbytes
                for _, op in self._blocks
            )
            layout.add("gemm", (size // 4,))
        else:
            library = self._resolved_libraries[chosen, out_dtype]
            if chosen.endswith(":rowwise"):
                layout.add("scale", (m, 1))
            columns = (
                self.out_features
                if chosen.endswith(":rowwise")
                else max(s.stop - s.start for s, _, _ in library._slots)
            )
            layout.add("staging", (m, columns), out_dtype)
            if outer == "flashinfer-bmm":
                library.add_workspace(layout, m)
        return layout

    def _workspace_bytes(self, m):
        return max(
            self._workspace_layout(m, dtype, quantized=q).nbytes
            for dtype in OUTPUT_DTYPES
            for q in (False, True)
        )

    def workspace_bytes(self, max_m: int) -> int:
        """Maximum scratch bytes over M in 1..max_m and both entries/dtypes; excludes outputs."""
        return maximum(max_m, self._workspace_bytes)

    def _workspace(self, m, dtype, workspace, *operands, quantized=False):
        chosen = self.provider_for(m, dtype)
        library = self._resolved_libraries.get((chosen, dtype))
        if isinstance(library, TorchScaledMM) and not library._has_out:
            raise RuntimeError("workspace needs torch._scaled_mm with out support")
        weights = [self._weight, self._weight_scale, self._input_scale, self._bias]
        for _, block in self._blocks:
            weights.extend(block._workspace_operands())
        if library is not None:
            weights.extend(
                getattr(library, field, None) for field in ("_weight", "_columns", "_input_scale")
            )
            for _, weight, scale in library._slots:
                weights.extend((weight, scale))
        return self._workspace_layout(m, dtype, quantized=quantized).bind(
            workspace, self.device, *operands, *weights
        )

    def _run_workspace(self, q, scale, out, views):
        m = q.shape[0]
        if not m:
            return
        chosen = self.provider_for(m, out.dtype)
        outer, inner, config = _fp8_static_dispatch.parse_provider(chosen)
        if outer == "torch-scaled-mm":
            if scale is None and chosen.endswith(":rowwise"):
                scale = self._static_scale("row", m)
            self._resolved_libraries[chosen, out.dtype].run(q, out, scale, buffers=views)
        elif outer == "flashinfer-bmm":
            self._resolved_libraries[chosen, out.dtype].run(q, out, buffers=views)
        else:
            if not q.is_contiguous() or q.data_ptr() % 16:
                q = views.get("q").copy_(q)
            if scale is None:
                scale = self._block_scale(q, inner, target=views.get("scale"))
            for columns, op in self._blocks:
                layout = op._workspace_layout(m, quantized=True, provider=inner, config=config)
                op._check_frozen()
                op._run_workspace(
                    q,
                    scale,
                    out[:, columns],
                    views.child("gemm", layout),
                    provider=inner,
                    config=config,
                )

    def _block_scale(self, q, inner, *, target=None):
        prepared = self._static_scale("block", q.shape[0])
        if prepared is not None:
            return prepared
        scale = self._input_scale.reshape(1, 1).expand(q.shape[0], _groups(self.in_features))
        if target is not None:
            return target.copy_(scale)
        # Triton and CuTe honor stride-0 scales; the CUDA JIT needs a packed grid.
        if inner == "cuda-sm12x" or (
            inner is None
            and any(op.provider_for(q.shape[0]) == "cuda-sm12x" for _, op in self._blocks)
        ):
            scale = scale.contiguous()
        return scale

    def _run(self, q, scale, out):
        chosen = self.provider_for(q.shape[0], out.dtype)
        outer, inner, config = _fp8_static_dispatch.parse_provider(chosen)
        if outer == "torch-scaled-mm":
            if scale is None and chosen.endswith(":rowwise"):
                scale = self._static_scale("row", q.shape[0])
            self._resolved_libraries[chosen, out.dtype].run(q, out, scale)
        elif outer == "flashinfer-bmm":
            self._resolved_libraries[chosen, out.dtype].run(q, out)
        else:
            if scale is None:
                scale = self._block_scale(q, inner)
            for columns, op in self._blocks:
                target = out[:, columns]
                if inner is None:
                    op.quantized(q, scale, out=target)
                elif inner == "cute-fp8":
                    # Same TMA alignment repair as BlockFP8Linear.quantized.
                    if q.stride(0) < self.in_features or q.stride(0) % 16 or q.data_ptr() % 16:
                        q = q.clone(memory_format=torch.contiguous_format)
                    op._cute.run(q, scale, target, config)
                else:
                    getattr(op, _INNER_ATTRS[inner]).run(q, scale, target)

    def _uses_pdl(self, m, out_dtype=torch.bfloat16):
        outer, inner, config = _fp8_static_dispatch.parse_provider(self.provider_for(m, out_dtype))
        if inner == "cute-fp8":
            return CONFIGS[config].pdl
        if outer == "block-fp8" and inner is None:
            return all(
                op.provider_for(m) == "cute-fp8" and CONFIGS[op._cute_config(m)].pdl
                for _, op in self._blocks
            )
        return False

    def __call__(
        self,
        x: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        name = type(self).__name__
        self._check_frozen()
        x = _check_activation(name, x, self.device, self.in_features)
        _require_inference(name, x)
        supplied = out is not None
        m = x.numel() // self.in_features
        if workspace is not None:
            if out is not None:
                out = self._output(m, out_dtype, x.dtype, out)
            dtype = self._out_dtype(out_dtype, x.dtype if out is None else out.dtype)
            views = self._workspace(m, dtype, workspace, x, out)
            out = self._output(m, out_dtype, x.dtype, out)
            if m:
                quantize = _quantizer(self.quantizer)

                chosen = self.provider_for(m, dtype)
                block, row = chosen.startswith("block-fp8"), chosen.endswith(":rowwise")
                result = quantize(
                    views.flatten(x),
                    self._input_scale,
                    block_scale=block,
                    row_scale=row,
                    pdl=self._uses_pdl(m, dtype),
                    outputs=(views.get("q"), views.get("scale") if block or row else None),
                )
                q, scale = result if block or row else (result, None)
                self._run_workspace(q, scale, out, views)
            if self._bias is not None:
                out.add_(self._bias)
            return out if supplied else out.reshape(*x.shape[:-1], self.out_features)
        out = self._output(m, out_dtype, x.dtype, out)
        if m:
            chosen = self.provider_for(m, out.dtype)
            block = chosen.startswith("block-fp8")
            row = chosen == "torch-scaled-mm:rowwise"
            result = quantize_static_fp8(
                x,
                self._input_scale,
                block_scale=block,
                row_scale=row,
                pdl=self._uses_pdl(m, out.dtype),
                quantizer=self.quantizer,
            )
            q, scale = result if block or row else (result, None)
            self._run(q, scale, out)
        if self._bias is not None:
            out.add_(self._bias)
        return out if supplied else out.reshape(*x.shape[:-1], self.out_features)

    def quantized(
        self,
        q: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        name = type(self).__name__
        self._check_frozen()
        q = _check_tensor(name, "q", q, (torch.float8_e4m3fn,), 2)
        if q.device != self.device or q.shape[1] != self.in_features:
            raise ValueError(f"{name}: q must be [M,{self.in_features}] on {self.device}")
        if q.numel() and q.stride(1) != 1:
            raise ValueError(f"{name}: q must have unit column stride")
        _require_inference(name, q)
        if workspace is not None:
            if out is not None:
                out = self._output(q.shape[0], out_dtype, torch.bfloat16, out)
            dtype = self._out_dtype(out_dtype, torch.bfloat16 if out is None else out.dtype)
            views = self._workspace(q.shape[0], dtype, workspace, q, out, quantized=True)
            out = self._output(
                q.shape[0],
                out_dtype,
                torch.bfloat16,
                out,
            )
            self._run_workspace(q, None, out, views)
            if self._bias is not None:
                out.add_(self._bias)
            return out
        out = self._output(q.shape[0], out_dtype, torch.bfloat16, out)
        if q.shape[0]:
            self._run(q, None, out)
        if self._bias is not None:
            out.add_(self._bias)
        return out

    def _dispatch_key(self, m, out_dtype=torch.bfloat16):
        chosen = self.provider_for(m, out_dtype)
        outer, inner, config = _fp8_static_dispatch.parse_provider(chosen)
        if outer != "block-fp8":
            return chosen, ()
        if inner is None:
            return chosen, tuple(op._dispatch_key(m) for _, op in self._blocks)
        # Forced inner kernels launch directly (see _run), so key on their own
        # specialisation: every Triton M class, the CuTe config, FlashInfer's per-M state.
        keys = []
        for _, op in self._blocks:
            if inner == "triton-fp8":
                keys.append(op._triton.dispatch_key(m, quantized=True))
            elif inner == "flashinfer-cutlass":
                keys.append(m)
            elif inner == "cute-fp8":
                keys.append(op._cute._key(config, out_dtype))
            else:
                keys.append(op._sm12x._config)
        return chosen, tuple(keys)

    def warm_up(
        self,
        max_m: int | None = None,
        *,
        batch_sizes: Sequence[int] | int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        out_dtype: torch.dtype | None = None,
        quantized: bool = False,
        out_row_stride: int | None = None,
    ) -> None:
        """Prepare every dispatch bucket in 1..max_m, or explicit sizes, on the capture stream."""
        self._check_frozen()
        name = type(self).__name__
        dtype = self._out_dtype(dtype, torch.bfloat16)
        out_dtype = self._out_dtype(out_dtype, torch.bfloat16 if quantized else dtype)
        sizes = _warm_up_sizes(name, max_m, batch_sizes, lambda m: self._dispatch_key(m, out_dtype))
        if out_row_stride is not None and (
            not isinstance(out_row_stride, int) or out_row_stride < self.out_features
        ):
            raise ValueError(f"{name}: out_row_stride must be an integer >= N")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"{name}: warm_up must run before capture")
            if sizes:
                self._prepare_static_scales(max(sizes))
            for m in sizes:
                outputs = [None]
                if out_row_stride is not None:
                    # Packed output plus both base-alignment classes of the strided
                    # one: a misaligned row-strided view selects the packed kernel.
                    parent = torch.empty(
                        (m * out_row_stride + 1,), dtype=out_dtype, device=self.device
                    )
                    outputs.extend(
                        parent[offset:].as_strided((m, self.out_features), (out_row_stride, 1))
                        for offset in (0, 1)
                    )
                x = torch.zeros((m, self.in_features), dtype=dtype, device=self.device)
                block_inputs = None
                pdl_variants = (False, True) if self._uses_pdl(m, out_dtype) else (False,)
                # Warm both quantizer layouts and all three scale-writing modes, even for
                # the prequantized entry. No input-layout argument is needed for capture.
                for source in (
                    x,
                    torch.zeros((m, self.in_features + 1), dtype=dtype, device=self.device)[:, 1:],
                ):
                    for block, row in ((False, False), (True, False), (False, True)):
                        for pdl in pdl_variants:
                            result = quantize_static_fp8(
                                source,
                                self._input_scale,
                                block_scale=block,
                                row_scale=row,
                                pdl=pdl,
                                quantizer=self.quantizer,
                            )
                            if block:
                                block_inputs = result
                if m and self.provider_for(m, out_dtype) == "block-fp8":
                    q, scale = block_inputs
                    scales = [self._block_scale(q, None) if quantized else scale, scale]
                    # The quantizer writes packed scales, as does the workspace
                    # prequantized entry. Keep both alongside the broadcast view.
                    examples = [
                        (
                            (q.stride(0), bool(q.data_ptr() % 16)),
                            (*s.stride(), bool(s.data_ptr() % 16)),
                        )
                        for s in scales
                    ]
                    for _, op in self._blocks:
                        op.warm_up(
                            batch_sizes=[m],
                            dtype=dtype,
                            out_dtype=out_dtype,
                            quantized=True,
                            out_row_stride=out_row_stride or self.out_features,
                            input_layouts=examples,
                        )
                for out in outputs:
                    if quantized:
                        self.quantized(self.quantize(x), out_dtype=out_dtype, out=out)
                    else:
                        self(x, out_dtype=out_dtype, out=out)
                if m and self.provider_for(m, out_dtype).startswith("block-fp8"):
                    self._warm_up_workspace(x, outputs, out_dtype, quantized)

    def _warm_up_workspace(self, x, outputs, out_dtype, quantized):
        # The workspace entries hand the block kernels packed scales (and their own
        # scratch); forced inner routes launch directly, so warm that path as well.
        workspace = torch.empty(
            self.workspace_bytes(x.shape[0]) // 4, dtype=torch.float32, device=self.device
        )
        for out in outputs:
            if quantized:
                self.quantized(self.quantize(x), out_dtype=out_dtype, out=out, workspace=workspace)
            else:
                self(x, out_dtype=out_dtype, out=out, workspace=workspace)


__all__ = ["PROVIDERS", "StaticFP8Linear", "quantize_static_fp8"]
