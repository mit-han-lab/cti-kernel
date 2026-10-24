# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""NVFP4 (W4A4) inference: activation quantization, the dense GEMM ``y = x . W^T``
(``NVFP4Linear``) and the gated-MLP input projection ``silu(x . G^T) * (x . U^T)`` quantized
to NVFP4 for the down projection (``NVFP4GateUpSiLU``).

**Format.** An NVFP4 tensor ``[rows, cols]`` (``cols`` a multiple of 32) is packed E2M1
values, two per byte along ``cols`` (the even column in the low nibble), one positive E4M3
scale per 16 columns and one FP32 global scale: ``v[r, c] = e2m1(data[r, c]) * fp32(scale[r,
c // 16]) * global_scale``. Weights (``NVFP4Weight``) carry their scales as the checkpoint's
``[rows, cols / 16]`` grid; activations (``NVFP4Activation``) carry them in the blocked layout
the tensor cores read: ``[ceil(rows / 128), ceil(cols / 64), 32, 4, 4]``, where scale ``(r,
b)`` sits at ``[r // 128, b // 4, r % 32, r // 32 % 4, b % 4]``; the padding blocks past
``cols / 16`` hold zeros and the padding rows past ``rows`` are not read (``blocked_scale`` /
``unblocked_scale`` convert).

**Activation quantization** uses a calibrated global scale supplied by the caller (a ModelOpt
``input_scale``: amax / (448 * 6)), so no call reduces over the whole tensor. Per row and 16
columns, the local quantizers follow FlashInfer 0.7.0.post1's market recipe:
``S = div.rn(1, input_scale)``, ``sf = E4M3_satfinite(S * (amax * rcp.approx(6)))``,
``factor = amax != 0 ? rcp.approx(float(sf) * rcp.approx(S)) : 0``,
``q = E2M1_satfinite(v * factor)``. Products, approximate reciprocals and the
zero comparison explicitly use FTZ PTX; amax is NaN-ignoring without FTZ.
No NaN/signed-zero preprocessing precedes hardware conversion.

**GEMMs** take exact products of the quantized operands, accumulate in FP32, scale by both
global scales in FP32 and round once: to BF16 in ``NVFP4Linear``; in fused ``NVFP4GateUpSiLU``,
SiLU(gate) * up is formed from the FP32 values and quantized as above with the down
projection's input scale as global scale. Unfused gated paths first round gate/up to BF16.

**Frozen weights, inference only.** A weight, scale or global scale that requires grad raises
ValueError at construction, before anything is launched or converted; an input that requires
grad under grad mode raises RuntimeError. Global scales are read on the device by every call
(never on the host), so construction and calls never synchronize; derived per-column scales
are refreshed by input quantization (or a gather for Linear's independent slots).
Shapes, dtypes, devices and strides are checked against this
contract and violations raise TypeError or ValueError before any launch.

**Implementations** (``provider``), resolved once at construction:

- ``"quack"``: on the RTX 5090, RTX PRO 6000 and GB10 with ``quack-kernels`` 0.6.5 and
  ``nvidia-cutlass-dsl`` 4.8.0 (the ``cute`` extra). Linear uses the default QuACK tile
  when explicitly requested or as fallback. Gated batches run their measured plans
  (``plan_for``; ``_nvfp4_dispatch``): QuACK's W4A4
  tensor-core GEMM after the local CuTe/Triton quantizer or, up to 64 rows, the local Triton
  small-batch GEMM, which decodes weight and activation tiles to BF16 (exact) for FP32
  tensor-core accumulation. ``NVFP4GateUpSiLU`` applies SiLU * up and the NVFP4 quantization
  on the FP32 accumulator in either GEMM's epilogue ("quack-fused", "triton"), or writes the
  BF16 projection and quantizes SiLU * up in one CuTe/Triton kernel ("quack-unfused"). Gate and up
  rows (and their scales) are interleaved once at construction.
- ``"flashinfer"``: optional version-qualified FlashInfer CUTLASS FP4 GEMM for dense
  linear and the unfused gated projection, using the same local quantizer. Selected by
  default for uniform Linear at every shape/M except retained CuTe wins; gated selects it by
  table entry. Linear shapes or installations without FlashInfer support use QuACK,
  then dequant.
- ``"cute"``: local CuTe DSL 4.8.0 small-M GEMM on RTX 5090 / RTX PRO 6000 / GB10.
  ``Cute`` table entries select it for 1..32 rows, with input quantization and the
  gated epilogue in one launch. Weights are prepared at construction; scratch is cached
  per M/config/input dtype on the first eager call. Warm up each input stride as well
  before capture. Linear output is BF16; FP16 output and ``quantized`` use a fallback.
  ``provider="cute"`` uses a table CuTe plan or the default CuTe Config when eligible.
- ``"dequant"``: any other device of compute capability 8.9 or newer. Weights are decoded
  once to BF16 (exact: an E2M1 value times an E4M3 scale has at most 6 significant bits), the
  quantized activation likewise per call, and cuBLAS multiplies them with FP32 output; the
  global scales and rounding follow as above.

Older devices raise RuntimeError. QuACK and Triton compile each configuration on its first call
(``prewarm`` compiles them all); afterwards calls are CUDA-graph capturable. The Triton GEMM's
split-K reduction uses per-operation arrival counters: the last program at each column tile
sums the FP32 partials in split order, resets the counter and applies the epilogue in the same
launch. One object's calls and graph replays must run in order on one stream. No backward.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch

from .._runtime.workspace import GraphWorkspace
from . import _nvfp4_cute_quant, _nvfp4_dispatch, _nvfp4_flashinfer, _nvfp4_quack

BLOCK = 16
PROVIDERS = ("quack", "flashinfer", "cute", "dequant")
LINEAR_PROVIDERS = ("quack", "flashinfer", "dequant")
# One batch size per dispatch bucket (powers of two) up to the last one plans distinguish.
PREWARM_ROWS = tuple(1 << i for i in range(13))
_INPUT_DTYPES = (torch.bfloat16, torch.float16)


class NVFP4Weight(NamedTuple):
    """A frozen NVFP4 weight ``[N, K]``: ``data`` uint8 ``[N, K / 2]`` (contiguous),
    ``scale`` ``float8_e4m3fn`` ``[N, K / 16]`` (contiguous) and ``global_scale`` FP32 with one
    element (a ModelOpt ``weight_scale_2``), all on one device. ``NVFP4Linear`` also accepts
    a 1-D tensor of globals with ``slot_rows``; gated weights each keep one element."""

    data: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor


class NVFP4Activation(NamedTuple):
    """A quantized activation ``[M, K]``: ``data`` uint8 ``[M, K / 2]`` (unit inner stride),
    ``scale`` ``float8_e4m3fn`` in the blocked layout (contiguous) and ``global_scale`` FP32
    ``[1]``."""

    data: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1] * 2


def scale_shape(rows: int, cols: int) -> tuple[int, int, int, int, int]:
    """The blocked scale layout's shape for an NVFP4 tensor ``[rows, cols]``."""
    return (-(-rows // 128), -(-cols // (4 * BLOCK)), 32, 4, 4)


def blocked_scale(scale: torch.Tensor) -> torch.Tensor:
    """Blocked layout of a ``[rows, cols / 16]`` scale grid (zero padding); any device."""
    rows, blocks = scale.shape
    atoms_m, atoms_k = -(-rows // 128), -(-blocks // 4)
    padded = scale.new_zeros(atoms_m * 128, atoms_k * 4)
    padded[:rows, :blocks] = scale
    return padded.view(atoms_m, 4, 32, atoms_k, 4).permute(0, 3, 2, 1, 4).contiguous()


def unblocked_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The ``[rows, cols / 16]`` grid of a blocked scale tensor; any device."""
    atoms_m, atoms_k = scale.shape[:2]
    grid = scale.permute(0, 3, 2, 1, 4).reshape(atoms_m * 128, atoms_k * 4)
    return grid[:rows, : cols // BLOCK]


def _decode_e2m1(data: torch.Tensor) -> torch.Tensor:
    # Arithmetic decoding (no host-to-device table copy, so it is graph capturable).
    codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
    exponent, mantissa = ((codes >> 1) & 3).float(), (codes & 1).float()
    magnitude = torch.where(
        exponent == 0, mantissa * 0.5, torch.exp2(exponent - 1) * (1 + mantissa * 0.5)
    )
    return torch.where(codes >= 8, -magnitude, magnitude)


def dequantize_nvfp4(
    data: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor | None = None
) -> torch.Tensor:
    """FP32 values ``[rows, cols]`` of uint8 ``data`` ``[rows, cols / 2]`` with a ``[rows,
    cols / 16]`` scale grid (``unblocked_scale`` converts an activation's) times
    ``global_scale``, if given; any device."""
    values = _decode_e2m1(data) * scale.float().repeat_interleave(BLOCK, dim=-1)
    return values if global_scale is None else values * global_scale.float().reshape(())


def _cuda_device(operation: str, tensor: torch.Tensor) -> torch.device:
    if tensor.device.type != "cuda":
        raise ValueError(f"{operation} runs on a CUDA device, not {tensor.device}")
    return tensor.device


def _require_fp8_capable(operation: str, device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 9):
        raise RuntimeError(
            f"{operation} needs compute capability 8.9 or newer; {device} has {capability}"
        )


def _require_inference(operation: str, *inputs: torch.Tensor) -> None:
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        raise RuntimeError(f"{operation} is inference-only and does not support autograd")


def _check_tensor(
    operation: str, name: str, value: object, dtypes: Sequence[torch.dtype], ndim: int
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{operation}: {name} must be a tensor")
    if value.dtype not in dtypes:
        expected = " or ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{operation}: {name} must be {expected}, not {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{operation}: {name} must have {ndim} dimensions, not {value.ndim}")
    return value


def _check_global_scale(operation: str, name: str, value: object, device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{operation}: {name} must be a tensor")
    if value.dtype != torch.float32 or value.numel() != 1:
        raise TypeError(f"{operation}: {name} must be one float32 element")
    if device is not None and value.device != device:
        raise ValueError(f"{operation}: {name} must be on {device}, not {value.device}")
    return value.reshape(1)


def _require_frozen(operation: str, tensors: dict[str, object]) -> None:
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            raise ValueError(f"{operation}: {name} is frozen and must not require grad")


def _require_frozen_weight(operation: str, name: str, weight: object) -> None:
    if isinstance(weight, tuple):
        fields = ("data", "scale", "global_scale")
        _require_frozen(
            operation, {f"{name}.{field}": t for field, t in zip(fields, weight, strict=False)}
        )


def _check_weight(operation: str, name: str, weight: object) -> NVFP4Weight:
    _require_frozen_weight(operation, name, weight)
    if not isinstance(weight, tuple) or len(weight) != 3:
        raise TypeError(f"{operation}: {name} must be an NVFP4Weight (data, scale, global_scale)")
    weight = NVFP4Weight(*weight)
    data = _check_tensor(operation, f"{name}.data", weight.data, (torch.uint8,), 2)
    scale = _check_tensor(operation, f"{name}.scale", weight.scale, (torch.float8_e4m3fn,), 2)
    rows, half = data.shape
    if rows == 0 or half == 0 or half % 16:
        raise ValueError(
            f"{operation}: {name}.data must be [N, K / 2] with N > 0 and K a positive "
            f"multiple of 32, not {tuple(data.shape)}"
        )
    if tuple(scale.shape) != (rows, half // 8):
        raise ValueError(
            f"{operation}: {name}.scale must be {(rows, half // 8)}, not {tuple(scale.shape)}"
        )
    if not data.is_contiguous() or not scale.is_contiguous():
        raise ValueError(f"{operation}: {name}.data and {name}.scale must be contiguous")
    device = _cuda_device(operation, data)
    if scale.device != device:
        raise ValueError(f"{operation}: {name}'s tensors must share one device")
    global_scale = _check_global_scale(
        operation, f"{name}.global_scale", weight.global_scale, device
    )
    return NVFP4Weight(data, scale, global_scale)


def _check_linear_global_scale(value: object) -> torch.Tensor:
    name = "NVFP4Linear: weight.global_scale"
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be float32")
    if value.numel() == 0 or (value.numel() != 1 and value.ndim != 1):
        raise ValueError(f"{name} must be one element or a nonempty 1-D tensor")
    return value.reshape(1) if value.numel() == 1 else value


def _check_linear_weight(weight, slot_rows):
    operation = "NVFP4Linear"
    _require_frozen_weight(operation, "weight", weight)
    if not isinstance(weight, tuple) or len(weight) != 3:
        raise TypeError(f"{operation}: weight must be an NVFP4Weight (data, scale, global_scale)")
    globals_ = _check_linear_global_scale(weight[2])
    # Reuse the weight layout/device checks without relaxing the gated scalar contract.
    checked = _check_weight(operation, "weight", NVFP4Weight(*weight[:2], globals_[:1]))
    slots, rows = globals_.numel(), checked.data.shape[0]
    if slot_rows is None:
        if slots != 1:
            raise ValueError(f"{operation}: slot_rows is required for multiple globals")
        slot_rows = (rows,)
    if not isinstance(slot_rows, Sequence):
        raise TypeError(f"{operation}: slot_rows must be a sequence of positive ints")
    slot_rows = tuple(slot_rows)
    if len(slot_rows) != slots:
        raise ValueError(f"{operation}: slot_rows must have {slots} entries")
    if any(not isinstance(count, int) or isinstance(count, bool) for count in slot_rows):
        raise TypeError(f"{operation}: slot_rows must contain positive ints")
    if any(count <= 0 for count in slot_rows):
        raise ValueError(f"{operation}: slot_rows must contain positive ints")
    if sum(slot_rows) != rows:
        raise ValueError(f"{operation}: slot_rows must sum to N={rows}")
    uniform = slots == 1 or globals_.stride(0) == 0
    return checked._replace(global_scale=globals_[:1] if uniform else globals_), slot_rows, uniform


def _check_input(operation: str, x: object, cols: int, device: torch.device) -> torch.Tensor:
    x = _check_tensor(operation, "x", x, _INPUT_DTYPES, 2)
    if x.shape[1] != cols:
        raise ValueError(f"{operation}: x must be [M, {cols}], not {tuple(x.shape)}")
    if x.device != device:
        raise ValueError(f"{operation}: x must be on {device}, not {x.device}")
    if x.shape[0] and x.stride(1) != 1:
        raise ValueError(f"{operation}: x must have unit inner stride")
    _require_inference(operation, x)
    return x


def _check_activation(
    operation: str, name: str, value: object, cols: int, device: torch.device
) -> NVFP4Activation:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{operation}: {name} must be an NVFP4Activation")
    data = _check_tensor(operation, f"{name}.data", value[0], (torch.uint8,), 2)
    scale = _check_tensor(operation, f"{name}.scale", value[1], (torch.float8_e4m3fn,), 5)
    rows = data.shape[0]
    if data.shape[1] * 2 != cols:
        raise ValueError(f"{operation}: {name} must be [M, {cols}] (data [M, {cols // 2}])")
    if tuple(scale.shape) != scale_shape(rows, cols):
        raise ValueError(
            f"{operation}: {name}.scale must be {scale_shape(rows, cols)}, not {tuple(scale.shape)}"
        )
    if rows and (data.stride(1) != 1 or (data.stride(0) % 16 and rows > 1) or data.data_ptr() % 16):
        raise ValueError(
            f"{operation}: {name}.data needs unit inner stride and 16-byte aligned rows"
        )
    if not scale.is_contiguous():
        raise ValueError(f"{operation}: {name}.scale must be contiguous")
    if data.device != device or scale.device != device:
        raise ValueError(f"{operation}: {name} must be on {device}")
    global_scale = _check_global_scale(operation, f"{name}.global_scale", value[2], device)
    return NVFP4Activation(data, scale, global_scale)


def _unaligned(*tensors: torch.Tensor) -> bool:
    """Whether a caller-owned output misses the 16-byte base alignment providers store with."""
    return any(t.data_ptr() % 16 for t in tensors)


def _copy_activation(out: NVFP4Activation, value: NVFP4Activation) -> NVFP4Activation:
    out.data.copy_(value.data)
    out.scale.copy_(value.scale)
    return out


def _empty_activation(rows: int, cols: int, global_scale, device) -> NVFP4Activation:
    data = torch.empty(rows, cols // 2, dtype=torch.uint8, device=device)
    scale = torch.empty(scale_shape(rows, cols), dtype=torch.float8_e4m3fn, device=device)
    return NVFP4Activation(data, scale, global_scale)


def _check_out_activation(
    operation: str, out: object, rows: int, cols: int, device: torch.device
) -> None:
    out = _check_activation(operation, "out", out, cols, device)
    if out.data.shape[0] != rows:
        raise ValueError(f"{operation}: out must have {rows} rows")
    if not out.data.is_contiguous():
        raise ValueError(f"{operation}: out.data must be contiguous")


def _quantize_into(
    x: torch.Tensor, global_scale: torch.Tensor, out: NVFP4Activation, *, gated: bool, **kwargs
) -> str | None:
    if x.shape[0]:
        launcher = _nvfp4_cute_quant.load(x.device)
        if launcher is None:
            from . import _nvfp4_triton

            launcher = _nvfp4_triton
            name = "triton"
        else:
            name = "cute"
        launcher.quantize(
            x, global_scale, gated=gated, data=out.data, scale=out.scale.view(torch.uint8), **kwargs
        )
        return name
    return None


def quantize_nvfp4(
    x: torch.Tensor, global_scale: torch.Tensor, *, out: NVFP4Activation | None = None
) -> NVFP4Activation:
    """Quantize BF16/FP16 ``x`` ``[M, K]`` (unit inner stride, ``K`` a multiple of 32) with
    the FP32 one-element ``global_scale`` on the current stream; compute capability 8.9 or
    newer (below SM100 with a software E2M1 rounding equal to the hardware's). ``out``, if given, receives the result (its ``global_scale`` is
    ignored); the returned activation carries ``global_scale``. Warm each CuTe
    shape/dtype/alignment/geometry before CUDA graph capture; an unwarmed specialization raises
    ``RuntimeError``."""
    operation = "quantize_nvfp4"
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{operation}: x must be a tensor")
    device = _cuda_device(operation, x)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] % 32:
        raise ValueError(f"{operation}: x must be [M, K] with K a positive multiple of 32")
    x = _check_input(operation, x, x.shape[1], device)
    global_scale = _check_global_scale(operation, "global_scale", global_scale, device)
    _require_fp8_capable(operation, device)
    rows, cols = x.shape
    if out is None:
        out = _empty_activation(rows, cols, global_scale, device)
    else:
        _check_out_activation(operation, out, rows, cols, device)
        out = NVFP4Activation(out[0], out[1], global_scale)
    _quantize_into(x, global_scale, out, gated=False)
    return out


def _resolve(operation: str, device: torch.device, provider: str | None):
    """(provider, QuACK launcher or None)."""
    if provider is not None and provider not in PROVIDERS:
        raise ValueError(f"{operation}: provider must be one of {PROVIDERS}, not {provider!r}")
    _require_fp8_capable(operation, device)
    launcher = _nvfp4_quack.load(device) if provider != "dequant" else None
    if provider == "quack" and launcher is None:
        raise RuntimeError(
            f"{operation}: provider 'quack' needs one of the qualified devices and "
            f"quack-kernels {_nvfp4_quack.DISTRIBUTIONS['quack-kernels']} with "
            f"nvidia-cutlass-dsl {_nvfp4_quack.DISTRIBUTIONS['nvidia-cutlass-dsl']}"
        )
    return ("quack" if launcher is not None else "dequant"), launcher


def _resolve_providers(operation, device, provider, shape, *, gated, shared_globals=True):
    """Resolve Linear FlashInfer by default; gated remains table-driven."""
    if provider == "flashinfer":
        _require_fp8_capable(operation, device)
        launcher = _nvfp4_flashinfer.load(device)
        if launcher is None:
            raise RuntimeError(
                f"{operation}: provider 'flashinfer' needs FlashInfer "
                f"one of {tuple(_nvfp4_flashinfer.QUALIFIED)} on a qualified capability "
                "with CUTLASS FP4 support"
            )
        return "flashinfer", None, launcher
    selected, quack = _resolve(operation, device, provider)
    flashinfer = None
    if (
        provider is None
        and (not gated or quack is not None)
        and _nvfp4_dispatch.uses_flashinfer(
            device, shape, gated=gated, shared_globals=shared_globals
        )
    ):
        flashinfer = _nvfp4_flashinfer.load(device)
    return ("flashinfer" if flashinfer is not None else selected), quack, flashinfer


def _prewarm_rows(op, rows, max_rows=None):
    if rows is not None:
        return tuple(rows)
    if max_rows is not None and not isinstance(max_rows, int):
        raise TypeError("prewarm: max_rows is an int; pass explicit row counts as rows=")
    limit = max(PREWARM_ROWS) if max_rows is None else max_rows
    if limit < 1:
        raise ValueError(f"prewarm: max_rows must be at least 1, not {max_rows}")
    extra = range(1, 33) if getattr(op, "_cute", None) is not None else ()
    intermediate = getattr(op, "intermediate_size", 0)
    cols = (op.hidden_size, intermediate) if intermediate else (op.in_features,)
    wide = torch.cuda.get_device_capability(op.device) == (12, 0)
    name = torch.cuda.get_device_name(op.device)
    last_bucket = _nvfp4_dispatch.LAST_BUCKET

    def specialization(count):
        # Dispatch/Triton BLOCK_M change at powers of two up to the last bucket any
        # table distinguishes (later rows reuse its plan); CuTe small-M compiles exact
        # rows (covered by extra). Quantizer geometry can change *inside* a bucket,
        # independently for the input and gated output widths. For prewarm's contiguous
        # BF16 tensors, alignment flags/strides are fixed by width; other quantizer
        # flags depend on the plan, not the exact M. The gated padding clear and QuACK
        # and FlashInfer take M at run time.
        return (
            1 << (min(count, last_bucket) - 1).bit_length(),
            tuple(_nvfp4_cute_quant._geometry(width, count, wide) for width in cols),
            tuple(_nvfp4_cute_quant._early_trigger(width, count) for width in cols),
            tuple(_nvfp4_cute_quant._pdl(width, count, name) for width in cols),
        )

    scan = min(limit, max(PREWARM_ROWS))
    representatives = dict.fromkeys(m for m in (*PREWARM_ROWS, *extra) if m <= limit)
    seen = {specialization(count) for count in representatives}
    # Only 4096 cheap metadata evaluations; prefer the largest M for each key.
    for count in range(scan, 0, -1):
        key = specialization(count)
        if key not in seen:
            seen.add(key)
            representatives[count] = None
    # Above the scan, keys change only where a bucket or a quantizer threshold does;
    # visit the first row of each interval (the smallest allocation).
    edges = {scan + 1, *((1 << b) + 1 for b in range(scan.bit_length(), limit.bit_length()))}
    for width in cols:
        edges.update(_nvfp4_cute_quant._row_edges(width, name))
    for count in sorted(m for m in edges if scan < m <= limit):
        key = specialization(count)
        if key not in seen:
            seen.add(key)
            representatives[count] = None
    return tuple(representatives)


def _route(plan) -> str:
    if plan is None:
        return "dequant"
    if isinstance(plan, _nvfp4_dispatch.FlashInfer):
        return "flashinfer"
    if isinstance(plan, _nvfp4_dispatch.Cute):
        return "cute"
    if isinstance(plan, _nvfp4_dispatch.Small):
        return "triton"
    return "quack"


def _decoded_bf16(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return dequantize_nvfp4(data, scale).to(torch.bfloat16)


def _prepare_cute(op, provider, shape):
    """NVFP4GateUpSiLU's CuTe small-M kernel and its interleaved weight, when a plan uses it."""
    op._cute = None
    op._cute_weight = None
    op._cute_workspaces = {}
    op._force_cute = provider == "cute"
    if provider not in (None, "cute"):
        return
    if not op._force_cute and not _nvfp4_dispatch.uses_cute(op.device, shape):
        return
    from . import _nvfp4_cute_small as cute

    if not cute.available(op.device):
        if op._force_cute:
            raise RuntimeError("provider 'cute' requires DSL 4.8.0 and sm120/sm121")
        return
    with torch.cuda.device(op.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct CuTe NVFP4 operations outside CUDA graph capture")
        op._cute_weight = cute.prepare_interleaved(*op._interleaved, *op._weight_globals)
    op._cute = cute
    op._provider = "cute"


def _reserve_cute(op, x):
    """Every eager call inside a graph workspace reserves the CuTe slot for each row count up
    to ``max_rows``, whatever its own plan: a later, smaller bucket may run CuTe after the
    first capture has frozen the workspace."""
    if op._cute is not None and (group := GraphWorkspace.active()) is not None:
        slot = ("nvfp4_cute", x.device, torch.float32)
        group.reserve(slot, (op,), _cute_scratch_counts, _allocate_cute_scratch(x.device))


def _cute_workspace(op, x, plan):
    rows = x.shape[0]
    group = GraphWorkspace.active()
    scratch = None
    if group is not None:
        # Split-K scratch is shared by the graph group (reserved on entry to the call);
        # each call binds its own output.
        shapes = op._cute.Workspace.scratch_shapes(
            op._cute_weight.shape, rows, plan.config(), gated=True
        )
        scratch = group.take(
            ("nvfp4_cute", x.device, torch.float32),
            rows,
            tuple(math.prod(shape) for shape in shapes),
        )
    key = (rows, plan, x.dtype, None if scratch is None else scratch[1].data_ptr())
    if key not in op._cute_workspaces:
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up each CuTe M/config/dtype outside CUDA graph capture")
            if scratch is not None:
                # Forget workspaces bound to an earlier group's scratch.
                stale = [k for k in op._cute_workspaces if k[3] not in (None, key[3])]
                for k in stale:
                    del op._cute_workspaces[k]
            op._cute_workspaces[key] = op._cute.Workspace(
                op._cute_weight, rows, plan.config(), gated=True, outputs=False, scratch=scratch
            )
    return op._cute_workspaces[key]


def _cute_scratch_counts(max_rows, op):
    """Largest (counters, partials) any CuTe plan of ``op`` takes at up to ``max_rows`` rows."""
    counts = [0, 0]
    for rows in range(1, min(max_rows, 32) + 1):  # CuTe small-M plans cover 1..32 rows
        _, plan = op.plan_for(rows)
        if isinstance(plan, _nvfp4_dispatch.Cute):
            shapes = op._cute.Workspace.scratch_shapes(
                op._cute_weight.shape, rows, plan.config(), gated=True
            )
            counts = [max(c, math.prod(shape)) for c, shape in zip(counts, shapes, strict=True)]
    return tuple(counts)


def _allocate_cute_scratch(device):
    return lambda counts: (
        torch.zeros(counts[0], device=device, dtype=torch.int32),
        torch.empty(counts[1], device=device, dtype=torch.float32),
    )


def _activation_bf16(a: NVFP4Activation) -> torch.Tensor:
    rows, cols = a.shape
    return _decoded_bf16(a.data, unblocked_scale(a.scale, rows, cols))


class NVFP4Linear:
    """``y = x . W^T`` with an NVFP4 weight ``W`` ``[N, K]`` (``K`` a multiple of 32).

    ``op = NVFP4Linear(weight, input_scale, *, out_dtype=torch.bfloat16, provider=None,
    slot_rows=None)``:
    ``weight`` an ``NVFP4Weight``; ``input_scale`` FP32 one-element tensor on its device, the
    global scale ``op(x)`` quantizes ``x`` with.

    ``weight.global_scale`` is FP32: one element of any shape, or a 1-D tensor of S globals
    for consecutive row slots. ``slot_rows`` gives S positive row counts summing to N;
    it is required for S > 1 (e.g. gate|up: ``(I, I)``). One global, or a stride-0 expansion
    of one global, preserves scalar dispatch and launches exactly. Independent slots use
    QuACK (one GEMM launch) or dequant; FlashInfer cannot be forced for these slots.
    One gather launch refreshes their per-column factors before activation quantization
    (or before multiplication for ``quantized``), preserving quantizer-to-GEMM PDL adjacency.
    Each slot's global is applied exactly in FP32 before the single output rounding.
    SGLang 0.5.22 and vLLM 0.31 instead take the maximum slot global for a merged weight
    and requantize it.

    - ``op(x, out=None)``: ``x`` BF16/FP16 ``[M, K]`` with unit inner stride; quantized, then
      multiplied.
    - ``op.quantized(a, out=None)``: ``a`` an ``NVFP4Activation`` ``[M, K]`` (e.g. the output
      of ``NVFP4GateUpSiLU``) multiplied as is, with its own global scale.

    For uniform slots, automatic dispatch uses qualified FlashInfer ``mm_fp4`` for every M, with
    the backend ``_nvfp4_flashinfer.linear_backend`` names for the device's compute capability.
    Unsupported FlashInfer shapes and installations fall back to QuACK, then dequant;
    ``provider="flashinfer"`` refuses such shapes, and ``provider="quack"`` forces the default
    QuACK tile.

    Both return ``[M, N]`` of ``out_dtype`` (BF16 or FP16), new storage or the contiguous
    ``out``. Read-only ``provider``: ``"quack"``, ``"flashinfer"`` or ``"dequant"``;
    ``plan_for(M)`` names the kernel configuration a batch runs. Input and weight global scales are
    read on device on every call and graph replay, including after in-place updates.
    """

    def __init__(
        self,
        weight: NVFP4Weight,
        input_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
        provider: str | None = None,
        slot_rows: Sequence[int] | None = None,
    ) -> None:
        operation = "NVFP4Linear"
        _require_frozen(operation, {"input_scale": input_scale})
        weight, slot_rows, self._uniform = _check_linear_weight(weight, slot_rows)
        device = weight.data.device
        self.input_scale = _check_global_scale(operation, "input_scale", input_scale, device)
        if out_dtype not in _INPUT_DTYPES:
            raise TypeError(f"{operation}: out_dtype must be torch.bfloat16 or torch.float16")
        self.out_features, self.in_features = weight.data.shape[0], weight.data.shape[1] * 2
        self.out_dtype, self.device = out_dtype, device
        if provider == "cute":
            raise ValueError(f"{operation}: provider must be one of {LINEAR_PROVIDERS}, not 'cute'")
        if not self._uniform and provider == "flashinfer":
            raise ValueError(
                f"{operation}: non-uniform slot globals are unsupported: "
                "FlashInfer mm_fp4 takes one scalar alpha"
            )
        if provider == "flashinfer" and not isinstance(
            _nvfp4_dispatch.linear_plan(device, self.out_features, self.in_features),
            _nvfp4_dispatch.FlashInfer,
        ):
            raise ValueError(
                f"{operation}: FlashInfer mm_fp4 does not support N={self.out_features}, "
                f"K={self.in_features}"
            )
        if self._uniform:
            self._provider, self._launcher, self._flashinfer = _resolve_providers(
                operation, device, provider, (self.out_features, self.in_features), gated=False
            )
        else:
            self._provider, self._launcher = _resolve(operation, device, provider)
            self._flashinfer = None
        self._force_quack = provider == "quack"
        self._force_flashinfer = provider == "flashinfer"
        self._flash_weight = (
            self._flashinfer.prepare(*weight) if self._flashinfer is not None else None
        )
        self._weights = (weight,)
        self._weight_globals = (weight.global_scale, weight.global_scale)
        self._alpha = torch.empty_like(self.input_scale)
        self._slot_index = None
        if self._uniform:
            self._column_scale = weight.global_scale.expand(self.out_features).contiguous()
        else:
            self._slot_index = torch.tensor(
                [slot for slot, count in enumerate(slot_rows) for _ in range(count)],
                dtype=torch.int64,
                device=device,
            )
            self._column_scale = torch.empty(self.out_features, dtype=torch.float32, device=device)
        self._weight = (
            None
            if self._flashinfer is not None and self._launcher is None
            else _prepare(self._launcher, weight.data, weight.scale)
        )
        self._quantizer = None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def quantizer(self) -> str | None:
        """Quantizer used by the last call; None when that call did not quantize."""
        return self._quantizer

    def plan_for(self, rows: int):
        """The configuration a batch of ``rows`` runs: a ``FlashInfer`` marker, a QuACK
        ``Config``, or None (``"dequant"``)."""
        if self._force_quack:
            return _nvfp4_quack.DEFAULT
        if self._force_flashinfer:
            return _nvfp4_dispatch.FlashInfer()
        if self._uniform and self._flashinfer is not None:
            plan = _nvfp4_dispatch.linear_plan(self.device, self.out_features, self.in_features)
            if isinstance(plan, _nvfp4_dispatch.FlashInfer):
                return plan
        return _nvfp4_quack.DEFAULT if self._launcher is not None else None

    def prewarm(
        self, max_rows: int | None = None, *, rows: Sequence[int] | None = None
    ) -> list[tuple[int, str]]:
        """Compile and load what every batch of 1..``max_rows`` rows runs (default 4096): one
        row count per power-of-two bucket (FlashInfer b12x picks its tile per bucket: M=1, up
        to 64, 128, 256, beyond) and in-bucket quantizer geometry/trigger/PDL changes, so the
        cost does not grow with ``max_rows``. ``rows`` instead runs exactly those counts. Uses
        contiguous BF16 inputs; warm other dtypes/strides separately. Call it outside graph
        capture. Returns the ``(rows, provider)`` route of each count it ran."""
        routes = []
        for count in _prewarm_rows(self, rows, max_rows):
            x = torch.zeros(count, self.in_features, dtype=torch.bfloat16, device=self.device)
            self(x)
            self.quantized(quantize_nvfp4(x, self.input_scale))
            routes.append((count, _route(self.plan_for(count))))
        return routes

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        _require_frozen_weight("NVFP4Linear", "weight", self._weights[0])
        x = _check_input("NVFP4Linear", x, self.in_features, self.device)
        rows = x.shape[0]
        out = self._output(rows, out)
        self._quantizer = None
        if rows == 0:
            return out
        if _unaligned(out):
            # Providers store 16-byte vectors; stage a valid but unaligned caller view.
            return out.copy_(self(x))
        plan = self.plan_for(rows)
        a = _empty_activation(rows, self.in_features, self.input_scale, self.device)
        self._refresh_slot_scales()
        self._quantizer = _quantize_into(
            x, self.input_scale, a, gated=False, **self._scale_args(plan)
        )
        return self._multiply(a, out, plan, prepared=True)

    def quantized(self, a: NVFP4Activation, out: torch.Tensor | None = None) -> torch.Tensor:
        _require_frozen_weight("NVFP4Linear", "weight", self._weights[0])
        a = _check_activation("NVFP4Linear", "a", a, self.in_features, self.device)
        rows = a.data.shape[0]
        out = self._output(rows, out)
        self._quantizer = None
        if rows and _unaligned(out):
            return out.copy_(self.quantized(a))
        if rows:
            self._refresh_slot_scales()
        return self._multiply(a, out, self.plan_for(rows)) if rows else out

    def _refresh_slot_scales(self):
        if not self._uniform:
            torch.index_select(
                self._weights[0].global_scale, 0, self._slot_index, out=self._column_scale
            )

    def _output(self, rows: int, out: torch.Tensor | None) -> torch.Tensor:
        if out is None:
            return torch.empty(rows, self.out_features, dtype=self.out_dtype, device=self.device)
        out = _check_tensor("NVFP4Linear", "out", out, (self.out_dtype,), 2)
        if tuple(out.shape) != (rows, self.out_features) or not out.is_contiguous():
            raise ValueError(f"NVFP4Linear: out must be contiguous {(rows, self.out_features)}")
        if out.device != self.device:
            raise ValueError(f"NVFP4Linear: out must be on {self.device}")
        return out

    def _scale_args(self, plan):
        if not self._uniform:
            return {"weight_scale": None, "alpha": None, "column_scale": None}
        return {
            "weight_scale": self._weight_globals[0],
            "alpha": self._alpha if isinstance(plan, _nvfp4_dispatch.FlashInfer) else None,
            "column_scale": None
            if isinstance(plan, _nvfp4_dispatch.FlashInfer)
            else self._column_scale,
        }

    def _multiply(
        self, a: NVFP4Activation, out: torch.Tensor, plan, *, prepared=False
    ) -> torch.Tensor:
        if self._uniform and not prepared:
            from ._nvfp4_triton import refresh_scales

            refresh_scales(a.global_scale, **self._scale_args(plan))
        if isinstance(plan, _nvfp4_dispatch.FlashInfer):
            self._flashinfer.linear(
                a, self._flash_weight, out, alpha=self._alpha, backend=self._flashinfer.backend
            )
        elif plan is not None:
            self._launcher.linear(
                a.data.view(torch.float4_e2m1fn_x2),
                a.scale,
                self._weight[0].view(torch.float4_e2m1fn_x2),
                self._weight[1].view(torch.float8_e4m3fn),
                self._column_scale,
                a.global_scale,
                out,
                plan,
            )
        else:
            product = torch.mm(_activation_bf16(a), self._weight.t(), out_dtype=torch.float32)
            out.copy_(product * self._column_scale * a.global_scale)
        return out


def _prepare(launcher, data: torch.Tensor, scale: torch.Tensor):
    """The weight as the provider reads it: (uint8 data, blocked uint8 scale) for QuACK and
    Triton, the decoded BF16 matrix for ``"dequant"``."""
    if launcher is None:
        return _decoded_bf16(data, scale)
    return data, blocked_scale(scale).view(torch.uint8)


def _small_gemm(
    plan, a, input_scale, weight, column_scale, counters, out, *, weight_globals
) -> None:
    from . import _nvfp4_triton

    _nvfp4_triton.gemm(
        a,
        input_scale,
        weight,
        column_scale,
        out=out,
        counters=counters,
        weight_globals=weight_globals,
        splits=plan.splits,
        block_n=plan.block_n,
        block_k=plan.block_k,
        num_warps=plan.warps,
        num_stages=plan.stages,
    )


def _counters(launcher, rows: int, device: torch.device) -> torch.Tensor | None:
    """Arrival counters of the Triton small-batch GEMM's split-K reduction (zeros; each launch
    leaves them zeroed)."""
    return (
        None if launcher is None else torch.zeros(-(-rows // 32), dtype=torch.int32, device=device)
    )


class NVFP4GateUpSiLU:
    """``silu(x . G^T) * (x . U^T)`` quantized to NVFP4, for NVFP4 gate and up weights ``G``,
    ``U`` ``[I, H]`` (``H`` and ``I`` multiples of 32).

    ``op = NVFP4GateUpSiLU(gate, up, input_scale=..., output_scale=..., provider=None)``:
    ``gate`` and ``up`` ``NVFP4Weight``s of one shape and device (their global scales may
    differ); ``input_scale`` the global scale ``op(x)`` quantizes ``x`` with and
    ``output_scale`` the result's (the down projection's ``input_scale``), FP32 one-element
    tensors on that device.

    - ``op(x, out=None)``: ``x`` BF16/FP16 ``[M, H]`` with unit inner stride.
    - ``op.quantized(a, out=None)``: ``a`` an ``NVFP4Activation`` ``[M, H]``.

    Both return an ``NVFP4Activation`` ``[M, I]`` (``data`` contiguous) with global scale
    ``output_scale``, new storage or written into ``out``'s data and scale. Read-only
    ``provider``: ``"quack"``, ``"flashinfer"``, ``"cute"`` or ``"dequant"``; ``plan_for(M)`` gives the path a batch takes
    (``"quack-fused"``, ``"quack-unfused"``, ``"triton"``, ``"cute"`` or ``"dequant"``) and its kernel
    configuration, ``provider_for(M)`` the path alone.

    The fused QuACK and Triton epilogues feed FP32 gate/up accumulators into SiLU * up.
    SGLang rounds gate_up GEMM output to BF16 before SiLU*up+quant; our fused path does
    not; gate 2 checks that it is no worse vs FP64 (also versus our unfused BF16 path).
    Construction keeps one weight copy, gate and up rows interleaved (g0, u0, g1, u1, ...)
    with blocked scales, which every provider reads; ``gate`` and ``up`` are not retained.
    ``from_interleaved`` runs over caller-held interleaved storage without a weight copy.

    Input and output global scales are read on device on every call and graph replay;
    in-place updates are supported. Weight global scales are also re-read on each call.
    """

    def __init__(
        self,
        gate: NVFP4Weight,
        up: NVFP4Weight,
        *,
        input_scale: torch.Tensor,
        output_scale: torch.Tensor,
        provider: str | None = None,
    ) -> None:
        operation = "NVFP4GateUpSiLU"
        _require_frozen(operation, {"input_scale": input_scale, "output_scale": output_scale})
        _require_frozen_weight(operation, "gate", gate)
        _require_frozen_weight(operation, "up", up)
        # The caller's global-scale tensors are kept as given (the checks reshape them).
        globals_ = (gate, up)
        gate = _check_weight(operation, "gate", gate)
        up = _check_weight(operation, "up", up)
        globals_ = tuple(weight[2] for weight in globals_)
        if gate.data.shape != up.data.shape or gate.data.device != up.data.device:
            raise ValueError(f"{operation}: gate and up must share one shape and device")
        width = gate.data.shape[0]
        if width % 32:
            raise ValueError(f"{operation}: the intermediate size must be a multiple of 32")
        for name, value in (("input_scale", input_scale), ("output_scale", output_scale)):
            _check_global_scale(operation, name, value, gate.data.device)
        # The one weight copy every provider reads: rows interleave gate and up (g0, u0,
        # g1, u1, ...) with blocked scales. ``gate`` and ``up`` are not retained.
        data = torch.stack((gate.data, up.data), dim=1).view(2 * width, -1)
        scale = blocked_scale(torch.stack((gate.scale, up.scale), dim=1).view(2 * width, -1))
        self._setup(data, scale, *globals_, input_scale, output_scale, provider)

    @classmethod
    def from_interleaved(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        gate_scale: torch.Tensor,
        up_scale: torch.Tensor,
        *,
        input_scale: torch.Tensor,
        output_scale: torch.Tensor,
        provider: str | None = None,
    ) -> NVFP4GateUpSiLU:
        """The operation over caller-held interleaved storage, read in place (no weight copy).

        ``data`` uint8 ``[2I, H / 2]`` (contiguous) whose rows interleave gate
        and up (g0, u0, g1, u1, ...); ``scale`` E4M3 either the ``[2I, H / 16]`` grid in the
        same row order (re-laid out once into the blocked layout) or already blocked
        (``scale_shape(2I, H)``, contiguous, zero padding; then read in place);
        ``gate_scale`` and ``up_scale`` the two slots' FP32 one-element global scales (one
        tensor when shared). The caller keeps the storage alive and unchanged. Only CuTe
        small-M re-lays the scales out (sm120 tables), and ``provider="flashinfer"`` with
        independent globals copies the two slots out (two CUTLASS GEMMs). Storage that is not
        16-byte aligned is copied once into aligned storage."""
        operation = "NVFP4GateUpSiLU"
        _require_frozen(
            operation,
            {
                "data": data,
                "scale": scale,
                "gate_scale": gate_scale,
                "up_scale": up_scale,
                "input_scale": input_scale,
                "output_scale": output_scale,
            },
        )
        data = _check_tensor(operation, "data", data, (torch.uint8,), 2)
        rows, half = data.shape
        if rows == 0 or rows % 64 or half == 0 or half % 16:
            raise ValueError(
                f"{operation}: data must be [2I, H / 2] with I and H positive multiples of "
                f"32, not {tuple(data.shape)}"
            )
        if not data.is_contiguous():
            raise ValueError(f"{operation}: data must be contiguous")
        if data.data_ptr() % 16:
            data = data.clone()  # Kernels (CUTLASS TMA included) need 16-byte-aligned rows.
        device = _cuda_device(operation, data)
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{operation}: scale must be a float8_e4m3fn tensor")
        if scale.device != device or not scale.is_contiguous():
            raise ValueError(f"{operation}: scale must be contiguous on {device}")
        if tuple(scale.shape) == (rows, half // 8):
            scale = blocked_scale(scale)
        elif tuple(scale.shape) != scale_shape(rows, half * 2):
            raise ValueError(
                f"{operation}: scale must be {(rows, half // 8)} or blocked "
                f"{scale_shape(rows, half * 2)}, not {tuple(scale.shape)}"
            )
        elif scale.data_ptr() % 16:
            scale = scale.clone()  # An aligned copy; CUTLASS TMA reads the blocked scales.
        op = cls.__new__(cls)
        op._setup(data, scale, gate_scale, up_scale, input_scale, output_scale, provider)
        return op

    def _setup(self, data, scale, gate_scale, up_scale, input_scale, output_scale, provider):
        operation = "NVFP4GateUpSiLU"
        device = data.device
        # The caller's tensors (before reshaping), for the per-call frozen check.
        self._global_sources = (gate_scale, up_scale)
        self.input_scale = _check_global_scale(operation, "input_scale", input_scale, device)
        self.output_scale = _check_global_scale(operation, "output_scale", output_scale, device)
        gate_scale = _check_global_scale(operation, "gate.global_scale", gate_scale, device)
        up_scale = _check_global_scale(operation, "up.global_scale", up_scale, device)
        width = data.shape[0] // 2
        self.intermediate_size, self.hidden_size = width, data.shape[1] * 2
        self.device = device
        # Shared scalar storage stays shared after in-place updates; equal values in
        # independent tensors do not (either can change before a replay).
        self._shared_globals = gate_scale.data_ptr() == up_scale.data_ptr()
        self._provider, self._launcher, self._flashinfer = _resolve_providers(
            operation,
            device,
            provider,
            (self.hidden_size, self.intermediate_size),
            gated=True,
            shared_globals=self._shared_globals,
        )
        self._force_flashinfer = provider == "flashinfer"
        self._weight_globals = (gate_scale, up_scale)
        self._interleaved = (data, scale)
        self._flash_weight = (
            self._flashinfer.prepare_interleaved(data, scale, gate_scale, up_scale)
            if self._flashinfer is not None
            else None
        )
        _prepare_cute(self, provider, (self.hidden_size, self.intermediate_size))
        self._alpha = torch.empty_like(self.input_scale)
        self._reciprocal = torch.empty_like(self.output_scale)
        self._column_scale = torch.cat((gate_scale, up_scale)).repeat(width)
        if self._force_flashinfer:
            self._weight = None
            if not self._flash_weight[1]:
                self._interleaved = ()  # The two per-slot copies are all it reads.
        elif self._launcher is None:
            self._weight = _decoded_bf16(data, unblocked_scale(scale, 2 * width, self.hidden_size))
        else:
            self._weight = data, scale.view(torch.uint8)
        # The gated GEMM's programs also cover the output scales' padding (64-column atoms).
        self._counters = _counters(self._launcher, 128 * -(-width // 64), device)
        self._quantizer = None

    def _check_frozen(self) -> None:
        tensors = dict(zip(("data", "scale"), self._interleaved, strict=False))
        tensors.update(
            zip(("gate.global_scale", "up.global_scale"), self._global_sources, strict=True)
        )
        _require_frozen("NVFP4GateUpSiLU", tensors)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def quantizer(self) -> str | None:
        """Last activation quantizer, including the unfused SiLU output quantizer."""
        return self._quantizer

    def plan_for(self, rows: int, *, allow_cute: bool = True) -> tuple[str, object]:
        """(path, kernel configuration) of a batch of ``rows``; ``quantized`` skips CuTe.
        ``allow_cute=False`` selects the fallback used for unaligned inputs."""
        if self._force_flashinfer:
            return _nvfp4_dispatch.UNFUSED, _nvfp4_dispatch.FlashInfer()
        path, plan = _nvfp4_dispatch.gated_plan(
            self.device,
            self.hidden_size,
            self.intermediate_size,
            rows,
            allow_cute=allow_cute and self._cute is not None,
            shared_globals=self._shared_globals,
        )
        if self._force_cute and allow_cute and not isinstance(plan, _nvfp4_dispatch.Cute):
            default = _nvfp4_dispatch.Cute()
            if _nvfp4_dispatch.cute_fits(
                default, self.device, rows, 2 * self.intermediate_size, self.hidden_size, gated=True
            ):
                path, plan = _nvfp4_dispatch.CUTE, default
        if isinstance(plan, _nvfp4_dispatch.Cute):
            return path, plan
        if self._launcher is None:
            return "dequant", None
        if isinstance(plan, _nvfp4_dispatch.FlashInfer) and self._flashinfer is None:
            return _nvfp4_dispatch.UNFUSED, _nvfp4_quack.DEFAULT
        return path, plan

    def provider_for(self, rows: int) -> str:
        return self.plan_for(rows)[0]

    def prewarm(
        self, max_rows: int | None = None, *, rows: Sequence[int] | None = None
    ) -> list[tuple[int, str]]:
        """Compile and load what every batch of 1..``max_rows`` rows runs (default 4096):
        each dispatch bucket, input/output quantizer geometry/trigger/PDL changes and every
        row count 1..32 for CuTe. The padding clear takes its length at run time, so
        128-row padding buckets need no compile. One row count per specialization runs.
        ``rows`` instead runs exactly those counts. Uses contiguous BF16 inputs; warm other
        dtypes/strides separately. Call it outside graph capture. Returns the
        ``(rows, path)`` route of each count it ran."""
        routes = []
        for count in _prewarm_rows(self, rows, max_rows):
            x = torch.zeros(count, self.hidden_size, dtype=torch.bfloat16, device=self.device)
            self(x)
            self.quantized(quantize_nvfp4(x, self.input_scale))
            routes.append((count, self.provider_for(count)))
        return routes

    def __call__(self, x: torch.Tensor, out: NVFP4Activation | None = None) -> NVFP4Activation:
        self._check_frozen()
        x = _check_input("NVFP4GateUpSiLU", x, self.hidden_size, self.device)
        rows = x.shape[0]
        out = self._output(rows, out)
        self._quantizer = None
        _reserve_cute(self, x)
        if rows == 0:
            return out
        if _unaligned(out.data, out.scale):
            return _copy_activation(out, self(x))
        path, plan = self.plan_for(rows)
        if isinstance(plan, _nvfp4_dispatch.Cute) and not self._cute.input_aligned(x):
            path, plan = self.plan_for(rows, allow_cute=False)
        if isinstance(plan, _nvfp4_dispatch.Cute):
            workspace = _cute_workspace(self, x, plan)
            self._quantizer = "cute"
            return self._cute.gated(
                x,
                self._cute_weight,
                self.input_scale,
                self.output_scale,
                plan.config(),
                workspace=workspace,
                out=out,
            )
        if path == "triton" and not plan.prequantize:
            self._small(plan, (x, None), self.input_scale, out)
            self._quantizer = "triton"
            return out
        a = _empty_activation(rows, self.hidden_size, self.input_scale, self.device)
        clear = (
            out.scale.view(torch.uint8)
            if path == "quack-fused" and self.intermediate_size % 64
            else None
        )
        self._quantizer = _quantize_into(
            x, self.input_scale, a, gated=False, clear_scale=clear, **self._scale_args(path, plan)
        )
        return self._project(a, out, path, plan, prepared=True)

    def quantized(self, a: NVFP4Activation, out: NVFP4Activation | None = None) -> NVFP4Activation:
        self._check_frozen()
        a = _check_activation("NVFP4GateUpSiLU", "a", a, self.hidden_size, self.device)
        rows = a.data.shape[0]
        out = self._output(rows, out)
        self._quantizer = None
        if rows and _unaligned(out.data, out.scale):
            return _copy_activation(out, self.quantized(a))
        return self._project(a, out, *self.plan_for(rows)) if rows else out

    def _output(self, rows: int, out: NVFP4Activation | None) -> NVFP4Activation:
        width = self.intermediate_size
        if out is None:
            return _empty_activation(rows, width, self.output_scale, self.device)
        _check_out_activation("NVFP4GateUpSiLU", out, rows, width, self.device)
        return NVFP4Activation(out[0], out[1], self.output_scale)

    def _small(self, plan, a, input_scale, out: NVFP4Activation) -> None:
        target = (out.data, out.scale.view(torch.uint8), self.output_scale)
        _small_gemm(
            plan,
            a,
            input_scale,
            self._weight,
            self._column_scale,
            self._counters,
            target,
            weight_globals=self._weight_globals,
        )

    def _scale_args(self, path, plan):
        flash = isinstance(plan, _nvfp4_dispatch.FlashInfer)
        return {
            "weight_scale": self._weight_globals[0],
            "up_scale": self._weight_globals[1],
            "alpha": self._alpha if flash else None,
            "output_scale": self.output_scale,
            "reciprocal": self._reciprocal if path == "quack-fused" else None,
            "column_scale": None if flash or path == "triton" else self._column_scale,
        }

    def _project(
        self, a: NVFP4Activation, out: NVFP4Activation, path: str, plan, *, prepared=False
    ):
        if isinstance(plan, _nvfp4_dispatch.Cute):
            path, plan = (
                _nvfp4_dispatch.gated_plan(
                    self.device,
                    self.hidden_size,
                    self.intermediate_size,
                    a.shape[0],
                    allow_cute=False,
                    shared_globals=self._shared_globals,
                )
                if self._launcher is not None
                else ("dequant", None)
            )
            if isinstance(plan, _nvfp4_dispatch.FlashInfer) and self._flashinfer is None:
                plan = _nvfp4_quack.DEFAULT
        if not prepared and path != "triton":
            from ._nvfp4_triton import refresh_scales

            refresh_scales(a.global_scale, **self._scale_args(path, plan))
        if path == "triton":
            self._small(plan, (a.data, a.scale.view(torch.uint8)), a.global_scale, out)
            return out
        if path == "quack-fused":
            if (
                not prepared and self.intermediate_size % 64
            ):  # the epilogue leaves the scales' padding blocks
                out.scale.view(torch.uint8).zero_()
            self._launcher.gated(
                a.data.view(torch.float4_e2m1fn_x2),
                a.scale,
                self._weight[0].view(torch.float4_e2m1fn_x2),
                self._weight[1].view(torch.float8_e4m3fn),
                self._column_scale,
                a.global_scale,
                self._reciprocal,  # written by input quantization (or quantized scalar preparation)
                out.data.view(torch.float4_e2m1fn_x2),
                out.scale,
                plan,
            )
            return out
        projection = torch.empty(
            a.data.shape[0], 2 * self.intermediate_size, dtype=torch.bfloat16, device=self.device
        )
        if path == "quack-unfused" and isinstance(plan, _nvfp4_dispatch.FlashInfer):
            self._flashinfer.gated(a, self._flash_weight, projection, alpha=self._alpha)
        elif path == "quack-unfused":
            self._launcher.linear(
                a.data.view(torch.float4_e2m1fn_x2),
                a.scale,
                self._weight[0].view(torch.float4_e2m1fn_x2),
                self._weight[1].view(torch.float8_e4m3fn),
                self._column_scale,
                a.global_scale,
                projection,
                plan,
            )
        else:
            product = torch.mm(_activation_bf16(a), self._weight.t(), out_dtype=torch.float32)
            projection.copy_(product * self._column_scale * a.global_scale)
        row_blocks = isinstance(plan, _nvfp4_dispatch.FlashInfer) and not self._flash_weight[1]
        self._quantizer = _quantize_into(
            projection, self.output_scale, out, gated=True, gated_row_blocks=row_blocks
        )
        return out


__all__ = [
    "NVFP4Activation",
    "NVFP4GateUpSiLU",
    "NVFP4Linear",
    "NVFP4Weight",
    "blocked_scale",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "scale_shape",
    "unblocked_scale",
]
