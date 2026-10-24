# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-scaled FP8 (W8A8) inference: per-token-group activation quantization, weight decoding
and the dense GEMM ``y = x . W^T``. ``slice_block_fp8`` gives block-aligned TP/EP weight and
scale views for loaders, including on CPU. ``BlockFP8Linear`` dispatches to independent
``TritonFP8Gemm``, ``FlashInferCutlassGemm``,
``CudaSm12xGemm`` and ``CuteFP8Gemm`` providers.

**Contract.** A weight is ``weight`` ``[N, K]`` ``float8_e4m3fn`` with ``weight_scale_inv``
``[ceil(N / 128), ceil(K / 128)]`` FP32, BF16 or UE8M0 (``torch.float8_e8m0fnu``, as in
DeepSeek V4 Flash checkpoints). ``BlockFP8Linear`` converts UE8M0 exactly to FP32 once at
construction: finite values are 2^-127 through 2^127, and byte zero is 2^-127, not zero.
Byte 0xFF remains NaN; validation checks metadata, not finiteness, without host sync.
Despite the checkpoint name, scales *multiply*: the weight's value is
``W[n, k] = fp32(weight[n, k]) * weight_scale_inv[n // 128, k // 128]``. Activations are
quantized per token and per group of 128 columns, ``x[m, k] ~ fp32(q[m, k]) * scale[m, k //
128]``, with the recipe of sgl-kernel's quantizer (bit for bit on finite inputs):
``amax = max(|x| over the group, 1e-10)``, ``scale = amax * fp32(1 / 448)`` and ``q =
E4M3_RN(clamp(x * fp32(448 / amax), -448, 448))``. An all-zero group gives zeros and a positive
scale; columns past ``K`` in the last group are not read and do not count toward its amax. A
group holding NaN or Inf gets a non-finite scale and NaN values, so non-finite inputs never
become finite outputs. Layouts are never guessed from shapes: every tensor is checked against
the shape, dtype, device and strides stated here, and violations raise TypeError or
ValueError before any launch.

The GEMM forms ``y[m, n] = sum_g (q_g . w_g) * scale[m, g] * weight_scale_inv[n // 128, g]``
over the 128-column groups g: each group's dot accumulates in FP32, scaling and the sum over
groups are FP32, and the result is rounded once to BF16 or FP16.

Both linear entries accept ``out=``: a BF16/FP16 ``[M, N]`` view on the weight's device,
with unit column stride and row stride >= N, where M flattens the input's leading axes.
They return that same object, leaving columns outside the view untouched. Its dtype is
used unless an explicit ``out_dtype`` disagrees (an error). Output requiring grad under
grad mode raises like inputs. Overlap with inputs is undefined and is not checked.
Triton writes strided outputs directly. CuTe does too with 16-byte aligned bases and rows,
through its own compiled variant when the row stride differs from N (packed outputs keep the
static-stride kernel); CUDA sm12x needs 4-byte alignment. Other alignments use a contiguous temporary and
``out.copy_(tmp)`` (one extra kernel), as does FlashInfer for non-contiguous output or
a base not aligned to 16 bytes.

**Frozen weights, inference only.** A weight or scale that requires grad raises ValueError
before anything else is checked or launched; an input that requires grad under grad mode
raises RuntimeError. There is no backward.

**Implementations** (``BlockFP8Linear.provider``), resolved once at construction from the
device and weight shape; ``op.provider_for(rows)`` names the one a batch runs:

- ``"triton-fp8"``: local Triton, E4M3 tensor-core dots; compute capability 8.9 and newer.
  Up to 16 rows the GEMM quantizes ``x`` itself; larger batches run the quantizer first.
  Small batches split K and sum the FP32 partials in a second launch, in a fixed order
  (results do not depend on timing).
- ``"flashinfer-cutlass"``: FlashInfer's groupwise CUTLASS GEMM after the local quantizer;
  compute capability 10.x and 12.x, ``N`` and ``K`` multiples of 128, and a packed
  (contiguous) weight: the pinned GEMM derives the weight's strides from its shape, so a
  strided shard view raises ValueError (the Triton providers read views in place). It needs
  ``flashinfer-python`` 0.7.0.post1 (the core pin; 0.6.11.post3 and 0.7.1 are also
  qualified) and compiles on its first call.
- ``"cuda-sm12x"``: the local CUDA GEMM for compute capability 12.x
  (``csrc/jit/block_fp8_gemm_sm12x.cu``, built through tvm-ffi on first use): E4M3
  ``mma.sync`` tensor-core MMAs in persistent, warp-specialized CTAs fed by TMA. It quantizes
  BF16/FP16 ``x`` inside the GEMM with the quantizer's recipe, bit for bit (no quantized
  activation is written), and reads caller-quantized ``q`` by TMA; ``N`` and ``K`` multiples
  of 128 and 16-byte aligned weight rows (shard views are read in place). On GB10 the
  measured dispatch runs it where the activation outgrows the L2 cache.
- ``"cute-fp8"``: the dense CuTe DSL 4.8.0 GEMM for sm_120/sm_121,
  after the existing activation quantizer (two launches). Packed N and K must be multiples
  of 128. Grid, persistent and tail-split configs support normal/swapped operands and
  MX/plain MMA; device and M-range dispatch selects a config. Warm up each config and
  output dtype on the capture stream before capture to compile.
- ``"triton-fp8+flashinfer-cutlass"`` (``HYBRID``), the default on the RTX 5090, RTX PRO
  6000 and GB10 for a packed weight with ``N`` and ``K`` multiples of 128 when the pinned
  FlashInfer imports: each batch size runs the provider measured fastest for this device and
  weight shape (``_block_fp8_dispatch.TABLE``, which on GB10 also holds ``cuda-sm12x`` runs;
  shapes it lacks switch to FlashInfer from a per-device GEMM size on, on any 12.0/12.1
  device). On GB300 the measured DeepSeek V4 Flash shapes run FlashInfer from M = 33 on
  (other shapes stay on ``triton-fp8``). A provider that cannot be constructed hands its runs to
  ``triton-fp8``. Otherwise compute capability 8.9 and newer
  defaults to ``triton-fp8``. Calls with a LoRA epilogue always run on the Triton GEMM.

Devices below compute capability 8.9 (no E4M3 tensor cores) raise RuntimeError at
construction.

**CUDA graphs.** No call synchronizes with the host; outputs and split-K partials come from
the caching allocator. Triton compiles a kernel on its first call per configuration.
Table-dispatched ``cute-fp8``, like ``cuda-sm12x``, compiles per configuration on first
eager use; call ``op.warm_up(max_m)`` (or ``batch_sizes=``) before capture, with the
intended dtypes,
``quantized=True`` for that entry and ``out_row_stride`` for a strided output.
``input_layouts="packed"`` is the default: aligned contiguous x/q rows, packed scales
and native CuTe MN-major scales. Use ``input_layouts="all"`` for every supported
specialization class, or examples such as ``[(K + 16, False), (K + 1, True)]``
(row stride, misaligned base). Quantized examples can also specify scales:
``[((K + 1, True), (scale_row_stride, scale_column_stride, True))]``.
Warm-up records dispatch/layout pairs per instance. During capture, a missing pair
raises RuntimeError naming the layout and ``warm_up(..., input_layouts=...)`` before
quantization, staging or provider launches. Eager calls remain unrestricted; they do
not register warm-up coverage. Layout keys describe compiled classes, so equivalent
strides share coverage. Match the batch/dispatch and input/output dtypes too.
No instance keeps scratch. Without ``workspace=`` every call's scratch (quantized input,
FlashInfer staging, split-K partials, CuTe tail-split slots) comes from the caching allocator
(the graph's private pool under capture); inside an active ``ctikernels.GraphWorkspace`` it
is one slot shared by every BlockFP8Linear on the device, sized by the largest op's
``workspace_bytes(max_rows)``, for warm-up and calls alike.

**Engine workspace.** ``workspace_bytes(max_m: int) -> int`` takes the maximum
over every M in 1..max_m, using the same provider/config selectors as calls:
split-K partials are non-monotonic in M. Both entries accept ``workspace=None``.
A supplied workspace must be a contiguous, 16-byte aligned FP32 CUDA tensor on
the weight's device, with at least the per-call byte requirement and no gradient
or shared storage with operands. All validation precedes launches; no host sync.
The bound covers dense and quantized entries, activation/scales, split-K scratch,
FlashInfer workspace, layout repair and internal output staging. Outputs are never
borrowed from workspace: without out=, torch.empty allocates a fresh result. The
caller owns workspace lifetime and stream ordering, including graph replay;
stream-ordered calls may reuse it while earlier outputs remain live. Pass out=
for allocation-free capture. Warm the intended routes before capture.
``workspace=None`` preserves the existing allocations and ownership.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Sequence
from itertools import product

import torch

from ctikernels._runtime.launch import CallPlans, WarmRegistry, call_class, eager_on
from ctikernels._runtime.workspace import GraphWorkspace

from . import _block_fp8_dispatch
from ._fp8_workspace import Layout, maximum
from .cuda._block_fp8_sm12x import CudaSm12xGemm
from .cute._block_fp8_provider import CONFIGS as CUTE_CONFIGS
from .cute._block_fp8_provider import CuteFP8Gemm
from .flashinfer._block_fp8 import FLASHINFER_VERSION as FLASHINFER_VERSION
from .flashinfer._block_fp8 import FlashInferCutlassGemm, FlashInferGroupedGemm
from .flashinfer._block_fp8 import _flashinfer_gemm as _flashinfer_gemm
from .lora import LoRAEpilogue
from .triton._block_fp8_provider import TritonFP8Gemm
from .triton._block_fp8_provider import _config as _config

GROUP = 128
_MIB = 1 << 20


def _dgrad_budget(m: int) -> int:
    """Decoded-weight scratch bytes for ``m`` rows of ``BlockFP8Linear.dgrad``, so the training
    step's peak stays at or below Unsloth's FP8 LoRA MLP: 96 MiB up to 4096 rows, 64 MiB past."""
    return 96 * _MIB if m <= 4096 else 64 * _MIB


PROVIDERS = (
    "triton-fp8",
    "flashinfer-cutlass",
    "cuda-sm12x",
    "cute-fp8",
)
HYBRID = "triton-fp8+flashinfer-cutlass"
_ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)
_SCALE_DTYPES = (torch.float32, torch.bfloat16, torch.float8_e8m0fnu)


def _groups(size: int) -> int:
    return -(-size // GROUP)


def _cuda_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _triton():
    """Our Triton block-FP8 kernels, imported on first use (Triton is optional on CPU)."""
    from .triton import _block_fp8

    return _block_fp8


def default_provider(device: torch.device | str | int) -> str:
    """The provider of small batches on ``device`` without ``provider=`` (a weight shape
    and device FlashInfer serves add it for large batches: ``HYBRID``)."""
    device = _cuda_device("BlockFP8Linear", device)
    capability = _cuda_capability(device)
    if capability >= (8, 9):
        return "triton-fp8"
    raise RuntimeError(
        f"BlockFP8Linear needs E4M3 tensor cores (compute capability 8.9 or newer); "
        f"{device} has {capability}"
    )


def _cuda_device(operation: str, device: torch.device | str | int) -> torch.device:
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{operation} runs on a CUDA device, not {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _require_frozen(operation: str, tensors: dict[str, torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            raise ValueError(f"{operation}: {name} is frozen and must not require grad")


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


def _check_weight(
    operation: str, weight: object, scale: object, *, ndim: int, require_cuda: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check a block-FP8 weight ``[..., N, K]`` (unit K stride) and its scale grid."""
    _require_frozen(operation, {"weight": weight, "weight_scale_inv": scale})
    weight = _check_tensor(operation, "weight", weight, (torch.float8_e4m3fn,), ndim)
    scale = _check_tensor(operation, "weight_scale_inv", scale, _SCALE_DTYPES, ndim)
    *lead, n, k = weight.shape
    if n == 0 or k == 0:
        raise ValueError(f"{operation}: weight must not be empty, got {tuple(weight.shape)}")
    expected = (*lead, _groups(n), _groups(k))
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"{operation}: weight_scale_inv must be {expected} (128 x 128 blocks of weight "
            f"{tuple(weight.shape)}), not {tuple(scale.shape)}"
        )
    if weight.stride(-1) != 1:
        raise ValueError(f"{operation}: weight must have unit stride along K")
    if (require_cuda and not weight.is_cuda) or scale.device != weight.device:
        device = "CUDA device" if require_cuda else "device"
        raise ValueError(f"{operation}: weight and weight_scale_inv must share one {device}")
    return weight, scale


def slice_block_fp8(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    dim: int,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weight and scale views of ``[start, stop)`` on CPU or CUDA, without launches.

    ``weight`` is frozen E4M3 ``[N, K]`` or ``[E, N, K]`` with unit K stride and matching
    FP32/BF16/UE8M0 128 x 128 scales. ``dim=-2`` splits N (column-parallel TP), ``dim=-1`` splits
    K (row-parallel TP), and ``dim=0`` splits experts of a stack (EP). TP starts must be
    multiples of 128; stops must be multiples of 128 or the full dimension's size.
    """
    operation = "slice_block_fp8"
    _require_frozen(operation, {"weight": weight, "weight_scale_inv": weight_scale_inv})
    ndim = weight.ndim if isinstance(weight, torch.Tensor) else 2
    if ndim not in (2, 3):
        raise ValueError(f"{operation}: weight must be [N, K] or [E, N, K]")
    weight, scale = _check_weight(
        operation, weight, weight_scale_inv, ndim=ndim, require_cuda=False
    )
    if dim not in (-2, -1) and not (dim == 0 and ndim == 3):
        raise ValueError(f"{operation}: dim must be -2, -1 or 0 for an expert stack")
    size = weight.shape[dim]
    if not 0 <= start <= stop <= size:
        raise ValueError(f"{operation}: need 0 <= start <= stop <= {size}")
    if dim == 0:
        return weight.narrow(dim, start, stop - start), scale.narrow(dim, start, stop - start)
    if start % GROUP or (stop % GROUP and stop != size):
        raise ValueError(
            f"{operation}: 128-block rule: start must be a multiple of 128; "
            "stop must be a multiple of 128 or the full dimension size"
        )
    first, last = start // GROUP, _groups(stop)
    return weight.narrow(dim, start, stop - start), scale.narrow(dim, first, last - first)


def _check_activation(operation: str, x: object, device: torch.device, k: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{operation}: x must be a tensor")
    if x.dtype not in _ACTIVATION_DTYPES:
        raise TypeError(f"{operation}: x must be BF16 or FP16, not {x.dtype}")
    if x.device != device:
        raise ValueError(f"{operation}: x is on {x.device}, the operation on {device}")
    if x.ndim < 1 or x.shape[-1] != k:
        raise ValueError(f"{operation}: x must be [..., {k}], not {tuple(x.shape)}")
    if x.stride(-1) != 1 and x.numel() > 0:
        raise ValueError(f"{operation}: x must have unit stride along its last dimension")
    return x


def quantize_per_token_group_fp8(
    x: torch.Tensor, *, q: torch.Tensor | None = None, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 CUDA ``x`` ``[M, K]`` (unit column stride) per token and per group
    of 128 columns: returns ``q`` ``[M, K]`` E4M3 and ``scale`` ``[M, ceil(K / 128)]`` FP32
    with ``x ~ q * scale`` per group (see the module docstring); new contiguous tensors, or
    the given ``q`` and ``scale`` (both or neither; on ``x``'s device, unit column stride,
    any row stride, e.g. workspace slices). One launch; capturable."""
    operation = "quantize_per_token_group_fp8"
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{operation}: x must be a tensor")
    if not x.is_cuda:
        raise ValueError(f"{operation}: x must be on a CUDA device")
    x = _check_activation(operation, x, x.device, x.shape[-1] if x.ndim else -1)
    if x.ndim != 2:
        raise ValueError(f"{operation}: x must be [M, K], not {tuple(x.shape)}")
    _require_inference(operation, x)
    M, K = x.shape
    if (q is None) != (scale is None):
        raise ValueError(f"{operation}: pass both q and scale, or neither")
    if q is None:
        q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((M, _groups(K)), dtype=torch.float32, device=x.device)
    else:
        q = _check_tensor(operation, "q", q, (torch.float8_e4m3fn,), 2)
        scale = _check_tensor(operation, "scale", scale, (torch.float32,), 2)
        if tuple(q.shape) != (M, K) or tuple(scale.shape) != (M, _groups(K)):
            raise ValueError(f"{operation}: q must be [{M}, {K}] and scale [{M}, {_groups(K)}]")
        if q.device != x.device or scale.device != x.device:
            raise ValueError(f"{operation}: q and scale must be on {x.device}")
        if M and K and (q.stride(1) != 1 or scale.stride(1) != 1):
            raise ValueError(f"{operation}: q and scale must have unit column stride")
        _require_inference(operation, q, scale)
    if M and K:
        _triton().quantize(x, q, scale)
    return q, scale


def dgrad_bf16(
    dy: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """``dx = dy . W`` for a frozen 16-bit weight ``W`` [N, K] (training backward): the BF16
    mode of the block-FP8 dgrad kernel, for callers that fuse into it (cuBLAS stays the plain
    path). ``dy`` [M, N] and ``W`` share one BF16/FP16 dtype, unit column stride (``dy`` any
    row stride); returns new [M, K] or fills ``out`` ([M, K], that dtype, unit column stride).
    ``accumulate=True`` (needs ``out``) adds ``out``'s values first in FP32 (beta 1). FP32
    accumulation, one rounding, no split: deterministic. Zero rows launch nothing."""
    name = "dgrad_bf16"
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"{name}: weight must be a 2-D tensor")
    if weight.dtype not in _ACTIVATION_DTYPES or weight.stride(1) != 1:
        raise ValueError(f"{name}: weight must be BF16/FP16 with unit column stride")
    if weight.requires_grad:
        raise ValueError(f"{name}: weight must be frozen")
    n, k = weight.shape
    dy = _check_activation(name, dy, weight.device, n)
    if dy.ndim != 2 or dy.dtype != weight.dtype:
        raise ValueError(f"{name}: dy must be [M, {n}] in the weight's dtype")
    m = dy.shape[0]
    if accumulate and out is None:
        raise ValueError(f"{name}: accumulate=True needs out")
    if out is None:
        out = torch.empty((m, k), dtype=dy.dtype, device=dy.device)
    else:
        out = _check_tensor(name, "out", out, (dy.dtype,), 2)
        if tuple(out.shape) != (m, k) or out.device != dy.device:
            raise ValueError(f"{name}: out must be [{m}, {k}] on {dy.device}")
        if out.stride(1) != 1 and out.numel() > 0:
            raise ValueError(f"{name}: out must have unit column stride")
    if m and k:
        with torch.cuda.device(dy.device):
            _triton().dgrad(dy, weight, None, out, accumulate=accumulate)
    return out


def dequantize_block_fp8(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode a block-FP8 weight ``[N, K]`` or a stack of expert weights ``[E, N, K]`` with
    its FP32/BF16/UE8M0 scales ``[(E,) ceil(N / 128), ceil(K / 128)]`` (UE8M0 converted to FP32 first): returns new contiguous
    ``round(fp32(weight) * scale)`` in ``dtype`` (BF16, FP16 or FP32). One launch over
    128 x 128 tiles; the output is the only allocation, or ``out`` (the weight's shape, ``dtype``,
    device, unit last stride; e.g. a scratch slice) is filled and returned. For references and
    loaders: a caller decoding a whole model this way holds all of it."""
    operation = "dequantize_block_fp8"
    if dtype not in (*_ACTIVATION_DTYPES, torch.float32):
        raise TypeError(f"{operation}: dtype must be BF16, FP16 or FP32, not {dtype}")
    ndim = weight.ndim if isinstance(weight, torch.Tensor) else 2
    if ndim not in (2, 3):
        raise ValueError(f"{operation}: weight must be [N, K] or [E, N, K]")
    weight, scale = _check_weight(operation, weight, weight_scale_inv, ndim=ndim)
    if scale.dtype == torch.float8_e8m0fnu:
        scale = scale.to(torch.float32)  # exact; the decode kernel reads FP32/BF16
    if out is None:
        out = torch.empty(weight.shape, dtype=dtype, device=weight.device)
    elif (
        not isinstance(out, torch.Tensor)
        or out.shape != weight.shape
        or out.dtype != dtype
        or out.device != weight.device
        or (out.numel() and out.stride(-1) != 1)
    ):
        raise ValueError(
            f"{operation}: out must be {dtype} {tuple(weight.shape)} with unit last stride"
        )
    if weight.ndim == 2:
        weight, scale, stacked = weight[None], scale[None], out[None]
    else:
        stacked = out
    if weight.shape[0]:
        _block_fp8_triton = _triton()

        _block_fp8_triton.dequantize(weight, scale, stacked)
    return out


def _warm_up_sizes(
    name: str,
    max_m: int | None,
    batch_sizes: Sequence[int] | int | None,
    key: Callable[[int], object],
) -> tuple[int, ...]:
    """Sizes for ``warm_up``: exactly one of ``max_m`` / ``batch_sizes``.

    ``max_m`` picks the largest representative of each distinct specialization /
    cached-state key (scratch then sized once), plus the first size for safety. The key
    owns scalar specialization classes as well as provider/configuration changes.
    """
    if (max_m is None) == (batch_sizes is None):
        raise ValueError(f"{name}: give exactly one of max_m and batch_sizes")
    if batch_sizes is not None:
        sizes = (batch_sizes,) if isinstance(batch_sizes, int) else tuple(batch_sizes)
        if any(isinstance(m, bool) or not isinstance(m, int) or m < 0 for m in sizes):
            raise ValueError(f"{name}: batch_sizes must contain nonnegative integers")
        return sizes
    if isinstance(max_m, bool) or not isinstance(max_m, int) or max_m < 1:
        raise ValueError(f"{name}: max_m must be a positive integer")
    largest = {key(m): m for m in range(1, max_m + 1)}
    return tuple(sorted({1, *largest.values()}))


def _layout_candidates(columns, *, scale=False):
    # Triton specializes integers equal to 1 / divisible by 16 and pointer
    # alignment to 16 bytes. CuTe additionally distinguishes rows divisible by 8.
    strides = [(columns, 1), (columns + 1, 1), (columns + 16 - columns % 16, 1), (1, 1), (8, 1)]
    if scale:
        strides.extend(product((1, 2, 16), repeat=2))
    return tuple((row, col, offset) for row, col in strides for offset in (0, 1))


def _selected_input_layouts(columns, input_layouts, *, m, quantized=False):
    """Normalize public examples to allocation descriptors (row, column, offset)."""
    packed = (columns, 1, 0)
    scales = (_groups(columns), 1, 0)
    if isinstance(input_layouts, str):
        if input_layouts == "packed":
            # CuTe/UE8M0 quantize returns packed scales; gated CuTe also emits MN-major.
            return (
                ((packed, scales), (packed, (1, (m + 3) // 4 * 4, 0)))
                if quantized
                else ((packed, None),)
            )
        if input_layouts == "all":
            return product(
                _layout_candidates(columns),
                _layout_candidates(_groups(columns), scale=True) if quantized else (None,),
            )
        raise ValueError("input_layouts must be 'packed', 'all', or a sequence of layouts")
    if not isinstance(input_layouts, Sequence):
        raise ValueError("input_layouts must be 'packed', 'all', or a sequence of layouts")

    def descriptor(example, *, scale=False):
        if not isinstance(example, (tuple, list)) or len(example) != (3 if scale else 2):
            raise ValueError(
                "input_layouts: use (row_stride, misaligned), or "
                "((row_stride, misaligned), (scale_row, scale_col, misaligned))"
            )
        *strides, misaligned = example
        if not isinstance(misaligned, bool) or any(
            type(stride) is not int or not 0 <= stride <= 2**31 - 1 for stride in strides
        ):
            raise ValueError("input_layouts: strides must be nonnegative int32 and misaligned bool")
        return (strides[0], strides[1] if scale else 1, int(misaligned))

    selected = []
    for example in input_layouts:
        input_example = example
        scale = scales if quantized else None
        if (
            quantized
            and isinstance(example, (tuple, list))
            and len(example) == 2
            and isinstance(example[0], (tuple, list))
        ):
            input_example, scale_example = example
            scale = descriptor(scale_example, scale=True)
        selected.append((descriptor(input_example), scale))
    return selected


def _tensor_layout(tensor):
    return tensor.stride(0), tensor.data_ptr() % 16 == 0


def _tensor_scale_layout(tensor):
    return *tensor.stride(), tensor.data_ptr() % 16 == 0


def _check_warmed_input(op, x, scale, *, dtype, out_dtype=None, packed=False, lora=None):
    # Host metadata only. Check before quantization, staging or any provider launch.
    current = torch.cuda.is_available() and op.device.index == torch.cuda.current_device()
    # The common case (op on the current device) enters no device context.
    with contextlib.nullcontext() if current else torch.cuda.device(op.device):
        if not torch.cuda.is_current_stream_capturing():
            return
        _check_triton_capture_layout(x, *(() if scale is None else (scale,)))
        if lora is not None and op._triton is not None:
            m = x.numel() // op.in_features
            # UE8M0 inputs reach the Triton GEMM quantized, as the eager call that warmed it.
            quantized_gemm = scale is not None or op._scale_ue8m0
            if op._triton.lora_key(m, quantized_gemm, lora) not in op._triton.warmed_lora:
                raise RuntimeError(
                    f"{type(op).__name__}: LoRA epilogue variant was not run eagerly before capture"
                )
        quantized = scale is not None
        kwargs = {"dtype": dtype, "quantized": quantized}
        if out_dtype is not None:
            kwargs["out_dtype"] = out_dtype
        m = x.numel() // op.in_features
        row, aligned = (op.in_features, True) if packed else _tensor_layout(x)
        dispatch = op._dispatch_key(m, **kwargs)
        layout = op._input_layout_key(
            m,
            (row, aligned),
            _tensor_scale_layout(scale) if quantized else None,
            quantized=quantized,
        )
        if (dispatch, layout) not in op._warmed_inputs:
            description = f"row stride={row}, aligned={aligned}"
            if quantized:
                description += (
                    f", scale strides={scale.stride()}, scale aligned={scale.data_ptr() % 16 == 0}"
                )
            raise RuntimeError(
                f"{type(op).__name__}: input layout class {layout} ({description}) was not warmed; "
                "warm_up(..., input_layouts=...) must cover it with the matching batch and dtypes before capture"
            )


def _activation_rows(op, x, *, out_dtype=None):
    try:
        return x.view(-1, op.in_features)
    except RuntimeError:
        # Non-flattenable leading axes need a packed copy. Check its eventual
        # launch class before reshape can enqueue that copy during capture.
        if x.numel():
            _check_warmed_input(op, x, None, dtype=x.dtype, out_dtype=out_dtype, packed=True)
        return x.reshape(-1, op.in_features)


def _input_layout_representatives(self, m, *, quantized=False, input_layouts="packed"):
    seen = set()
    for layout, scale_layout in _selected_input_layouts(
        self.in_features, input_layouts, m=m, quantized=quantized
    ):
        key = self._input_layout_key(
            m,
            (layout[0], layout[2] == 0),
            (*scale_layout[:2], scale_layout[2] == 0) if quantized else None,
            quantized=quantized,
        )
        if key not in seen:
            seen.add(key)
            yield layout, scale_layout, key


def _layout_tensor(rows, columns, layout, *, dtype, device, scale=False):
    row, col, offset = layout
    size = max(rows - 1, 0) * row + (columns - 1) * col + offset + 1
    parent = (
        torch.ones(size, dtype=dtype, device=device)
        if scale
        else torch.zeros(size, dtype=dtype, device=device)
    )
    return parent[offset:].as_strided((rows, columns), (row, col))


def _check_triton_capture_layout(*tensors):
    # Integer argument types also specialize Triton. Representatives with int64
    # strides would require multi-GiB backing storage for every M bucket.
    if any(stride > 2**31 - 1 for t in tensors for stride in t.stride()):
        with torch.cuda.device(tensors[0].device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "BlockFP8: warm_up covers int32 input strides; use smaller strides "
                    "for CUDA graph capture"
                )


# Warm-up classes already launched in this process (see BlockFP8Linear.warm_up).
_WARMED = WarmRegistry()


def _tensors(*values) -> bool:
    """Whether every operand is a tensor or None (other values take the validating path)."""
    return all(v is None or isinstance(v, torch.Tensor) for v in values)


def _lora_kwargs(lora):
    # Forward the LoRA epilogue only when present: LoRA-free calls keep their signature.
    return {} if lora is None else {"lora": lora}


class _LinearOutput:
    """Shared BF16/FP16 output allocation and validation for inference linears."""

    def _out_dtype(self, out_dtype: torch.dtype | None, default: torch.dtype) -> torch.dtype:
        out_dtype = default if out_dtype is None else out_dtype
        if out_dtype not in _ACTIVATION_DTYPES:
            raise TypeError(f"{type(self).__name__}: out_dtype must be BF16 or FP16")
        return out_dtype

    def _output(
        self,
        rows: int,
        out_dtype: torch.dtype | None,
        default: torch.dtype,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        name = type(self).__name__
        if out is not None:
            out = _check_tensor(name, "out", out, _ACTIVATION_DTYPES, 2)
            _require_inference(name, out)
            if out.device != self._device:
                raise ValueError(f"{name}: out must be on {self._device}")
            if out.shape != (rows, self.out_features):
                raise ValueError(f"{name}: out must be [{rows}, {self.out_features}]")
            if out.stride(1) != 1 or out.stride(0) < self.out_features:
                raise ValueError(f"{name}: out needs unit column stride and row stride >= N")
            dtype = self._out_dtype(out_dtype, out.dtype)
            if dtype != out.dtype:
                raise ValueError(f"{name}: out_dtype must equal out.dtype")
            return out
        dtype = self._out_dtype(out_dtype, default)
        return torch.empty((rows, self.out_features), dtype=dtype, device=self._device)


class BlockFP8Linear(_LinearOutput):
    """``y = x . W^T`` for one frozen block-FP8 weight (see the module docstring).

    ``BlockFP8Linear(weight, weight_scale_inv, *, provider=None)`` keeps references to the
    weight ``[N, K]`` E4M3 (unit K stride) and its scale grid ``[ceil(N / 128), ceil(K /
    128)]`` FP32, BF16 or UE8M0 on one CUDA device. UE8M0 is converted exactly to FP32 once
    at construction and retained for every provider; byte 0xFF stays NaN without a finiteness
    check, following the metadata-only validation policy.
    ``op(x, *, out_dtype=None, out=None, workspace=None)`` takes BF16/FP16 ``x`` ``[..., K]`` with unit
    last stride and returns new ``[..., N]`` in ``out_dtype`` (default:
    ``x``'s dtype); ``op.quantized(q, scale, *, out_dtype=None, out=None, workspace=None)`` takes an
    activation already quantized by ``quantize_per_token_group_fp8`` (``q`` ``[M, K]`` E4M3
    with unit column stride, ``scale`` ``[M, ceil(K / 128)]`` FP32), for one quantization
    shared by several projections (default output dtype: BF16).
    ``out=`` must be BF16/FP16 ``[M, N]`` on the same device, with unit column stride and
    row stride >= N, even for higher-rank inputs. It determines the dtype unless an
    explicit ``out_dtype`` disagrees, and is returned unchanged in identity. Only its
    columns are written; overlap with inputs is undefined. Output requiring grad under
    grad mode raises. ``M = 0`` returns without launching.
    Triton writes views directly; CuTe and CUDA sm12x do so with bases and rows aligned
    to 16 and 4 bytes respectively. Other alignments, and non-contiguous FlashInfer
    outputs or bases not aligned to 16 bytes, use a contiguous temporary followed by
    one ``copy_`` kernel (per-call staging; see the module docstring for GraphWorkspace).
    ``warm_up(max_m=None, *, batch_sizes=None, dtype=torch.bfloat16, out_dtype=None,
    quantized=False, out_row_stride=None, input_layouts="packed")`` takes exactly one of
    ``max_m`` (every dispatch bucket in ``1..max_m``) and ``batch_sizes``, and runs zero
    inputs for the selected input layout classes through the selected provider, returning a list of route dictionaries
    on the current stream, compiling and allocating scratch before graph capture without
    synchronization. Match the later input/output dtypes and output row stride; use
    ``quantized=True`` for that entry. Supplying ``out_row_stride`` also warms packed
    output and both aligned and unaligned output bases. Calling warm_up during
    capture raises RuntimeError, as does a call with an unwarmed dispatch/input layout
    pair during capture. See the module docstring for explicit input layout examples.
    ``scale_ue8m0=True`` quantizes BF16/FP16 inputs with power-of-two activation
    scales (K divisible by 128): the CuTe quantizer on sm120/sm121 with DSL 4.8.0, our Triton
    one elsewhere, with the same bits.
    Prequantized inputs are always consumed directly, regardless of this flag.
    With FlashInfer in the provider, a BF16 or strided scale grid is copied to contiguous
    FP32 at construction (a few KiB per projection): construct after loading the weights.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        *,
        provider: str | None = None,
        scale_ue8m0: bool = False,
        _extra_providers: tuple[str, ...] = (),
    ) -> None:
        name = type(self).__name__
        weight, scale = _check_weight(name, weight, weight_scale_inv, ndim=2)
        if not isinstance(scale_ue8m0, bool):
            raise TypeError("scale_ue8m0 must be bool")
        if scale_ue8m0 and weight.shape[1] % 128:
            raise ValueError("scale_ue8m0 needs K divisible by 128")
        self._warmed_inputs = set()
        self._plans = CallPlans()  # steady-state call class -> launch (see _plan)
        self._scale_ue8m0 = scale_ue8m0
        self._scale_source = scale
        if scale.dtype == torch.float8_e8m0fnu:
            scale = scale.to(torch.float32)
        self._weight = weight
        self._scale = scale
        self._device = weight.device
        capability = _cuda_capability(self._device)
        sms = torch.cuda.get_device_properties(self._device).multi_processor_count
        self._capability, self._sms = capability, sms
        default = provider is None
        if default:
            provider = default_provider(self._device)
        elif provider not in PROVIDERS:
            raise ValueError(f"{name}: provider must be one of {PROVIDERS}, not {provider!r}")
        self._triton = None
        if provider == "triton-fp8":
            self._triton = TritonFP8Gemm(
                weight,
                scale,
                capability=capability,
                sms=sms,
                operation=name,
            )
        self._flashinfer = None
        self._sm12x = None
        self._cute = None
        self._buckets: _block_fp8_dispatch.Buckets = ((0, provider),)
        if provider == "flashinfer-cutlass":
            self._flashinfer = FlashInferCutlassGemm(
                weight, scale, capability=capability, operation=name
            )
        elif provider == "cuda-sm12x":
            self._sm12x = CudaSm12xGemm(
                weight, scale, capability=capability, sms=sms, operation=name
            )
        elif provider == "cute-fp8":
            self._cute = CuteFP8Gemm(weight, scale, capability=capability, sms=sms, operation=name)
        elif provider == "triton-fp8" and default:
            buckets = _block_fp8_dispatch.buckets(capability, sms, *weight.shape)
            wanted = {p for _, p in buckets or ()}
            if FlashInferCutlassGemm.name in wanted:
                try:
                    self._flashinfer = FlashInferCutlassGemm(
                        weight, scale, capability=capability, operation=name
                    )
                except (RuntimeError, ValueError):
                    pass  # unaligned or strided weight, or no qualified FlashInfer
            if CudaSm12xGemm.name in wanted:
                try:
                    self._sm12x = CudaSm12xGemm(
                        weight, scale, capability=capability, sms=sms, operation=name
                    )
                except (RuntimeError, ValueError):
                    pass  # unaligned weight or no JIT toolchain
            # Instantiate only providers named by an explicitly installed dispatch table.
            for cute_type, attr in ((CuteFP8Gemm, "_cute"),):
                if cute_type.name in wanted:
                    try:
                        setattr(
                            self,
                            attr,
                            cute_type(
                                weight, scale, capability=capability, sms=sms, operation=name
                            ),
                        )
                    except (RuntimeError, ValueError):
                        pass  # Optional DSL or layout unavailable; restrict to Triton below.
            available = {TritonFP8Gemm.name}
            available |= {
                g.name for g in (self._flashinfer, self._sm12x, self._cute) if g is not None
            }
            buckets = _block_fp8_dispatch.restrict(buckets, available) if buckets else buckets
            if buckets is not None:
                self._buckets = buckets
                provider = HYBRID if len(buckets) > 1 else buckets[0][1]
        self._provider = provider

        # StaticFP8Linear can request direct access to additional GEMMs. Resolve
        # our own buckets FIRST: these optional objects must not change dispatch.
        for extra in _extra_providers:
            attr, factory = {
                "triton-fp8": ("_triton", TritonFP8Gemm),
                "flashinfer-cutlass": ("_flashinfer", FlashInferCutlassGemm),
                "cuda-sm12x": ("_sm12x", CudaSm12xGemm),
                "cute-fp8": ("_cute", CuteFP8Gemm),
            }[extra]
            if getattr(self, attr) is None:
                kwargs = {} if extra == "flashinfer-cutlass" else {"sms": sms}
                try:
                    gemm = factory(weight, scale, capability=capability, operation=name, **kwargs)
                except (RuntimeError, ValueError, ImportError):
                    continue  # Caller resolves an unavailable override to its fallback.
                setattr(self, attr, gemm)

    @property
    def provider(self) -> str:
        """The resolved implementation: one of ``PROVIDERS``, or ``HYBRID``."""
        return self._provider

    def provider_for(self, rows: int) -> str:
        """The implementation a batch of ``rows`` rows runs."""
        return _block_fp8_dispatch.provider(self._buckets, rows)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def weight(self) -> torch.Tensor:
        """The E4M3 ``[N, K]`` weight, by reference."""
        return self._weight

    @property
    def weight_scale(self) -> torch.Tensor:
        """The scale grid the GEMMs read: the caller's FP32/BF16 tensor by reference, or for a
        UE8M0 grid its exact FP32 conversion (a copy made at construction)."""
        return self._scale

    @property
    def in_features(self) -> int:
        return self._weight.shape[1]

    @property
    def out_features(self) -> int:
        return self._weight.shape[0]

    def _check_frozen(self) -> None:
        _require_frozen(
            type(self).__name__, {"weight": self._weight, "weight_scale_inv": self._scale}
        )
        _require_frozen(type(self).__name__, {"weight_scale_inv": self._scale_source})

    def _cute_config(self, rows: int) -> str:
        return _block_fp8_dispatch.cute_config(
            self._capability, self._sms, self.out_features, self.in_features, rows
        )

    def _workspace_operands(self):
        operands = [self._weight, self._scale, self._scale_source]
        for name in ("_triton", "_cute", "_sm12x", "_flashinfer"):
            provider = getattr(self, name, None)
            operands.extend(getattr(provider, field, None) for field in ("_weight", "_scale"))
        return operands

    def _workspace_layout(self, m, *, quantized=False, provider=None, config=None):
        from .cute._block_fp8_provider import CONFIGS, tail_workspace

        chosen = provider or self.provider_for(m)
        n, k = self.out_features, self.in_features
        layout = Layout()
        if not quantized:
            layout.add("x", (m, k), torch.bfloat16)
        if not m:
            return layout
        # FlashInfer's sm100 GEMM runs on M padded to 4 rows (FlashInferCutlassGemm.rows).
        rows = self._flashinfer.rows(m) if chosen == "flashinfer-cutlass" else m
        # Internal staging for unsupported out= alignment/strides, never returned.
        if chosen in ("cute-fp8", "flashinfer-cutlass", "cuda-sm12x"):
            layout.add("staging", (rows, n), torch.bfloat16)
        prequantized = quantized or self._scale_ue8m0
        needs_quant = not quantized and (
            self._scale_ue8m0
            or chosen in ("cute-fp8", "flashinfer-cutlass")
            or (chosen.startswith("triton-") and m > self._triton.fused_quantization_rows)
        )
        if needs_quant or (
            prequantized and chosen in ("cute-fp8", "flashinfer-cutlass", "cuda-sm12x")
        ):
            layout.add("q", (rows, k), torch.float8_e4m3fn)
        if needs_quant or (prequantized and chosen in ("flashinfer-cutlass", "cuda-sm12x")):
            layout.add("scale", (rows, _groups(k)))
        if chosen.startswith("triton-"):
            splits = self._triton.config_for(m)[2]
            if splits > 1:
                layout.add("partial", (splits, m, n))
        elif chosen == "cute-fp8":
            cfg = CONFIGS[config or self._cute_config(m)]
            if cfg.scheduler == "tailsplit":
                layout.add("partial", (sum(tail_workspace(cfg, self._sms)),))
        elif chosen == "flashinfer-cutlass":
            layout.add("library", (self._flashinfer.workspace_bytes,), torch.uint8)
        return layout

    def _workspace_bytes(self, m):
        # LoRA calls run on Triton whatever the table picks (``_lora_route``).
        providers = (None, "triton-fp8") if self._triton is not None else (None,)
        return max(
            self._workspace_layout(m, quantized=q, provider=p).nbytes
            for q in (False, True)
            for p in providers
        )

    def _graph_workspace(self, m):
        """FP32 scratch from the active GraphWorkspace's block-FP8 slot, shared by every
        BlockFP8Linear on the device (sized by the largest op's ``workspace_bytes``), or None."""
        group = GraphWorkspace.active()
        if group is None or not m:
            return None
        slot = ("block_fp8_linear", self._device)
        shape = (self.out_features, self.in_features, self._provider, self._buckets)
        shape += (self._scale_ue8m0, self._triton is not None)
        group.reserve(
            slot,
            shape,
            lambda max_rows, *_: (-(-self.workspace_bytes(max_rows) // 4),),
            lambda counts: (torch.empty(counts, dtype=torch.float32, device=self._device),),
        )
        return group.take(slot, m, (-(-self._workspace_bytes(m) // 4),))[0]

    def workspace_bytes(self, max_m: int) -> int:
        """Maximum aligned scratch bytes over M in 1..max_m; excludes returned outputs."""
        return maximum(max_m, self._workspace_bytes)

    def _quantize_ue8m0(self, x, outputs=None):
        """Power-of-two-scale quantization of ``x`` [M, K]: the CuTe quantizer on sm120/121
        with DSL 4.8.0, our Triton one elsewhere (the same bits)."""
        from .cute._block_fp8_provider import available

        if available(self._capability):
            from .cute._block_fp8_act_quant import quantize

            return quantize(
                x, scale_ue8m0=True, **({} if outputs is None else {"outputs": outputs})
            )
        m, k = x.shape
        q, scale = outputs or (
            torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((m, k // GROUP), dtype=torch.float32, device=x.device),
        )
        if m:
            _check_triton_capture_layout(x)
            _triton().quantize(x, q, scale, scale_ue8m0=True)
        return q, scale

    def _lora_route(self, m, lora):
        """The provider of an ``m``-row call: the table's, except that a LoRA epilogue runs
        on this instance's Triton GEMM whenever the table picks another provider."""
        chosen = self.provider_for(m)
        if lora is not None and not chosen.startswith("triton-") and self._triton is not None:
            return TritonFP8Gemm.name
        return chosen

    def _lora_override(self, m, lora):
        """``{"provider": "triton-fp8"}`` when ``_lora_route`` leaves the table, else ``{}``."""
        route = self._lora_route(m, lora)
        return {} if route == self.provider_for(m) else {"provider": route}

    def _lora_provider(self, chosen, lora):
        if lora is not None and not chosen.startswith("triton-"):
            raise ValueError(
                f"{type(self).__name__}: the LoRA epilogue runs in the Triton provider, "
                f"not {chosen!r}"
            )

    def _run_workspace(self, x, scale, out, views, *, provider=None, config=None, lora=None):
        m, k = x.shape
        if not m:
            return
        chosen = provider or self.provider_for(m)
        self._lora_provider(chosen, lora)
        config = config or (self._cute_config(m) if chosen == "cute-fp8" else None)
        staged_x, staged_scale = x, scale
        if scale is not None:
            if (
                chosen == "cute-fp8"
                and (x.stride(0) < k or x.stride(0) % 16 or x.data_ptr() % 16)
                or chosen == "flashinfer-cutlass"
                and (not x.is_contiguous() or x.data_ptr() % 16)
                or chosen == "cuda-sm12x"
                and (x.stride(0) % 16 or x.data_ptr() % 16)
            ):
                staged_x = views.get("q")[:m]
            if (
                chosen == "flashinfer-cutlass"
                and (not scale.is_contiguous() or scale.data_ptr() % 16)
                or chosen == "cuda-sm12x"
                and not scale.is_contiguous()
            ):
                staged_scale = views.get("scale")[:m]
        elif chosen == "cuda-sm12x" and (x.stride(0) * x.element_size() % 16 or x.data_ptr() % 16):
            staged_x = views.get("x", x.dtype)
        if provider is None or lora is not None:
            # Key on the caller's layout, as the workspace-free path and warm_up do;
            # the staging repair below is part of that layout class. Check before
            # any repair copies or quantization. Explicit StaticFP8Linear
            # overrides warm their providers directly; LoRA calls rerouted to Triton
            # (``_lora_override``) are checked like table-routed ones.
            _check_warmed_input(self, x, scale, dtype=x.dtype, out_dtype=out.dtype, lora=lora)
        if staged_x is not x:
            x = staged_x.copy_(x)
        if staged_scale is not scale:
            scale = staged_scale.copy_(scale)
        if scale is None and (self._scale_ue8m0 or chosen in ("cute-fp8", "flashinfer-cutlass")):
            from .cute._block_fp8_provider import CONFIGS

            if self._scale_ue8m0:
                x, scale = self._quantize_ue8m0(
                    x, outputs=(views.get("q")[:m], views.get("scale")[:m])
                )
            elif chosen == "cute-fp8" and CONFIGS[config].pdl:
                from .cute._block_fp8_act_quant import quantize

                x, scale = quantize(x, outputs=(views.get("q")[:m], views.get("scale")[:m]))
            else:
                kernel = _triton()

                _check_triton_capture_layout(x)
                q, scale = views.get("q")[:m], views.get("scale")[:m]
                kernel.quantize(x, q, scale)
                x = q
        if chosen == "cute-fp8":
            self._cute.run(x, scale, out, config, buffers=views)
        elif chosen == "flashinfer-cutlass":
            self._flashinfer.run(x, scale, out, buffers=views)
        elif chosen == "cuda-sm12x":
            self._sm12x.run(x, scale, out, buffers=views)
        else:
            _check_triton_capture_layout(x, *(() if scale is None else (scale,)))
            self._triton.run(x, scale, out, buffers=views, **_lora_kwargs(lora))

    def __call__(
        self,
        x: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        lora: LoRAEpilogue | None = None,
    ) -> torch.Tensor:
        """``lora`` adds a LoRA delta before the output rounding. It runs in the Triton GEMM:
        default-dispatched batches whose table provider is another one run on this instance's
        Triton GEMM; an explicit non-Triton ``provider`` raises ValueError."""
        plain = lora is None and workspace is None and _tensors(x, out)
        if plain:
            key = ("x", call_class(x), call_class(out), out_dtype)
            plan = self._plans.get(key)
            if plan is not None and self._steady(x):
                rows = x if x.ndim == 2 else x.view(-1, self.in_features)
                result = out
                if result is None:
                    result = torch.empty(
                        (rows.shape[0], self.out_features), dtype=plan[1], device=self._device
                    )
                plan[0](rows, None, result)
                return out if out is not None else result.reshape(*x.shape[:-1], self.out_features)
        name = type(self).__name__
        self._check_frozen()
        x = _check_activation(name, x, self._device, self.in_features)
        _require_inference(name, x)
        # B is read in the output dtype; lora_shrink stores it in x's dtype.
        if lora is not None and (out.dtype if out is not None else out_dtype or x.dtype) != x.dtype:
            raise ValueError(f"{name}: LoRA needs the output dtype to equal x's dtype")
        supplied = out is not None
        if workspace is None:
            workspace = self._graph_workspace(x.numel() // self.in_features)
        if workspace is not None:
            m = x.numel() // self.in_features
            route = self._lora_override(m, lora)
            views = self._workspace_layout(m, **route).bind(
                workspace, self.device, x, out, *self._workspace_operands()
            )
            out = self._output(m, out_dtype, x.dtype, out)
            try:
                rows = x.view(-1, self.in_features)
            except RuntimeError:
                rows = views.get("x", x.dtype)
                if m:
                    _check_warmed_input(self, rows, None, dtype=x.dtype, out_dtype=out.dtype)
                rows.view(x.shape).copy_(x)
            self._run_workspace(rows, None, out, views, **route, **_lora_kwargs(lora))
            return out if supplied else out.reshape(*x.shape[:-1], self.out_features)
        out = self._output(x.numel() // self.in_features, out_dtype, x.dtype, out)
        rows = _activation_rows(self, x, out_dtype=out.dtype)
        if rows.shape[0]:
            _check_warmed_input(self, rows, None, dtype=x.dtype, out_dtype=out.dtype, lora=lora)
        if self._scale_ue8m0:
            if rows.shape[0]:
                q, scale = self._quantize_ue8m0(rows)
                self._run_quantized(q, scale, out, **_lora_kwargs(lora))
            return out if supplied else out.reshape(*x.shape[:-1], self.out_features)
        m = rows.shape[0]
        if m:
            chosen = self._lora_route(m, lora)
            self._lora_provider(chosen, lora)
            if chosen == "flashinfer-cutlass":
                _check_triton_capture_layout(rows)
                q, scale = quantize_per_token_group_fp8(rows)
                self._flashinfer.run(q, scale, out)
            elif chosen == "cuda-sm12x":
                self._sm12x.run(rows, None, out)
            elif chosen == "cute-fp8":
                from .cute._block_fp8_provider import CONFIGS

                if not CONFIGS[self._cute_config(m)].pdl:
                    _check_triton_capture_layout(rows)
                self._cute.run_x(rows, out, self._cute_config(m))
            else:
                _check_triton_capture_layout(rows)
                self._triton.run(rows, None, out, **_lora_kwargs(lora))
            if plain and rows.data_ptr() == x.data_ptr() and self._steady(x):
                self._remember(key, rows, None, out)
        return out if supplied else out.reshape(*x.shape[:-1], self.out_features)

    def dgrad(
        self, dy: torch.Tensor, *, out: torch.Tensor | None = None, accumulate: bool = False
    ) -> torch.Tensor:
        """The frozen weight's input gradient ``dx = dy . W_hat`` (training backward).

        ``dy`` is BF16/FP16 ``[..., N]`` with unit last stride; a 2-D ``dy`` may have any row
        stride (e.g. a column slice of a merged gradient). Returns new ``[..., K]`` in
        ``dy``'s dtype, or fills ``out`` (``[M, K]``, ``dy``'s dtype, unit column stride) and
        returns it. Up to 64 rows (memory-bound): our Triton kernel decodes each 128-row block
        of W exactly to BF16/FP16, its product with ``dy`` accumulates in FP32 and takes the
        block's FP32 scale; blocks are summed in a fixed order (no split, no weight copy).
        More rows: W is decoded into a scratch in ``dy``'s dtype in the fewest column chunks
        within ``_dgrad_budget`` and each chunk is one cuBLAS GEMM with FP32 accumulation.
        Either way deterministic, rounded once.
        ``accumulate=True`` (needs ``out``) adds ``out``'s existing values first (beta 1), e.g.
        a LoRA dX term written before. Compute capability 8.9+; the Triton kernels compile on
        first use, no warm-up needed eagerly; zero rows launch nothing."""
        plain = _tensors(dy, out)
        key = ("dy", call_class(dy), call_class(out), accumulate) if plain else None
        relaunch = self._plans.get(key) if plain else None
        if relaunch is not None and self._steady():
            rows = dy if dy.ndim == 2 else dy.view(-1, self.out_features)
            dx = out
            if dx is None:
                dx = torch.empty(
                    (rows.shape[0], self.in_features), dtype=dy.dtype, device=self._device
                )
            relaunch(rows, dx)
            return dx if out is not None else dx.reshape(*dy.shape[:-1], self.in_features)
        name = f"{type(self).__name__}.dgrad"
        self._check_frozen()
        dy = _check_activation(name, dy, self._device, self.out_features)
        n, k = self.out_features, self.in_features
        if dy.ndim == 2:
            rows = dy
        else:
            try:
                rows = dy.view(-1, n)
            except RuntimeError as error:
                raise ValueError(f"{name}: dy's leading dimensions must flatten") from error
        m = rows.shape[0]
        if accumulate and out is None:
            raise ValueError(f"{name}: accumulate=True needs out")
        if out is None:
            dx = torch.empty((m, k), dtype=dy.dtype, device=self._device)
        else:
            dx = _check_tensor(name, "out", out, (dy.dtype,), 2)
            if tuple(dx.shape) != (m, k) or dx.device != self._device:
                raise ValueError(f"{name}: out must be [{m}, {k}] on {self._device}")
            if dx.stride(1) != 1 and dx.numel() > 0:
                raise ValueError(f"{name}: out must have unit column stride")
        if m:
            kernel = _triton()

            with torch.cuda.device(self._device):
                if m <= 64:
                    relaunch = kernel.dgrad(
                        rows, self._weight, self._scale, dx, accumulate=accumulate
                    )
                else:  # decode and cuBLAS: nothing to remember
                    relaunch = None
                    self._decoded_dgrad(kernel, rows, dx, accumulate=accumulate)
            if plain and relaunch is not None and self._steady():
                self._plans.put(key, relaunch)
        return dx if out is not None else dx.reshape(*dy.shape[:-1], k)

    def _decoded_dgrad(
        self, kernel, dy: torch.Tensor, dx: torch.Tensor, *, accumulate: bool
    ) -> None:
        """``dx = dy . W_hat`` by column chunks of W: decode a chunk (exact, in dy's dtype), then
        one cuBLAS GEMM into ``dx``'s columns; the fewest chunks whose scratch fits
        ``_dgrad_budget``. cuBLAS picks its kernel by layout, so an ``out`` laid out unlike a new
        [M, K] tensor goes through one (same bits)."""
        n, k = self.out_features, self.in_features
        if dx.stride(0) != k or dx.data_ptr() % 16:
            if accumulate:
                fresh = dx.clone(memory_format=torch.contiguous_format)
            else:
                fresh = torch.empty(dx.shape, dtype=dx.dtype, device=dx.device)
            self._decoded_dgrad(kernel, dy, fresh, accumulate=accumulate)
            dx.copy_(fresh)
            return
        budget = _dgrad_budget(dy.shape[0]) // (n * dy.element_size()) // GROUP * GROUP
        width = min(k, max(GROUP, budget))
        decoded = torch.empty((1, n, width), dtype=dy.dtype, device=self._device)
        for start in range(0, k, width):
            stop = min(start + width, k)
            chunk = decoded[:, :, : stop - start]
            kernel.dequantize(
                self._weight[None, :, start:stop],
                self._scale[None, :, start // GROUP : -(-stop // GROUP)],
                chunk,
            )
            if accumulate:
                dx[:, start:stop].addmm_(dy, chunk[0])
            else:
                torch.mm(dy, chunk[0], out=dx[:, start:stop])

    def quantized(
        self,
        q: torch.Tensor,
        scale: torch.Tensor,
        *,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        lora: LoRAEpilogue | None = None,
    ) -> torch.Tensor:
        plain = lora is None and workspace is None and _tensors(q, scale, out)
        if plain:
            key = ("q", call_class(q), call_class(scale), call_class(out), out_dtype)
            plan = self._plans.get(key)
            if plan is not None and self._steady(q, scale):
                result = out
                if result is None:
                    result = torch.empty(
                        (q.shape[0], self.out_features), dtype=plan[1], device=self._device
                    )
                plan[0](q, scale, result)
                return result
        name = type(self).__name__
        self._check_frozen()
        q = _check_tensor(name, "q", q, (torch.float8_e4m3fn,), 2)
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
        _require_inference(name, q, scale)
        if workspace is None:
            workspace = self._graph_workspace(q.shape[0])
        if workspace is not None:
            route = self._lora_override(q.shape[0], lora)
            views = self._workspace_layout(q.shape[0], quantized=True, **route).bind(
                workspace, self.device, q, scale, out, *self._workspace_operands()
            )
            out = self._output(q.shape[0], out_dtype, torch.bfloat16, out)
            self._run_workspace(q, scale, out, views, **route, **_lora_kwargs(lora))
            return out
        out = self._output(q.shape[0], out_dtype, torch.bfloat16, out)
        if q.shape[0]:
            _check_warmed_input(
                self, q, scale, dtype=torch.bfloat16, out_dtype=out.dtype, lora=lora
            )
            self._run_quantized(q, scale, out, **_lora_kwargs(lora))
            if plain and self._steady(q, scale):
                self._remember(key, q, scale, out)
        return out

    def _run_quantized(self, q, scale, out, lora=None):
        k = self.in_features
        if q.shape[0]:
            chosen = self._lora_route(q.shape[0], lora)
            self._lora_provider(chosen, lora)
            if chosen == "flashinfer-cutlass":
                self._flashinfer.run(q, scale, out)
            elif chosen == "cuda-sm12x":
                self._sm12x.run(q, scale, out)
            elif chosen == "cute-fp8":
                if q.stride(0) < k or q.stride(0) % 16 or q.data_ptr() % 16:
                    # TMA needs 16-byte aligned rows; copy the other unit-stride views.
                    q = q.clone(memory_format=torch.contiguous_format)
                self._cute.run(q, scale, out, self._cute_config(q.shape[0]))
            else:
                _check_triton_capture_layout(q, scale)
                self._triton.run(q, scale, out, **_lora_kwargs(lora))
        return out

    def _steady(self, *inputs) -> bool:
        """Whether a call may take its remembered plan: eager, on the op's device, no active
        GraphWorkspace, frozen weight, and no autograd input (the checks a plan skips)."""
        return (
            eager_on(self._device.index)
            and GraphWorkspace.active() is None
            and not self._weight.requires_grad
            and not self._scale_source.requires_grad
            and not (torch.is_grad_enabled() and any(t.requires_grad for t in inputs))
        )

    def _remember(self, key, a, a_scale, out) -> None:
        """After a validated eager call, keep its call class's launch: later calls of the class
        skip validation, routing and capture checks (the same launches, the same bits)."""
        launch = self._plan(a, a_scale, out)
        if launch is not None:
            self._plans.put(key, (launch, out.dtype))

    def _plan(self, a, a_scale, out):
        """``launch(a, a_scale, out)`` of an LoRA-free, workspace-free call class, or None to
        keep validating (UE8M0 inputs and provider layouts that stage or copy)."""
        m = a.shape[0]
        chosen = self.provider_for(m)
        if self._scale_ue8m0:
            return None
        if chosen == "cute-fp8":
            config = self._cute_config(m)
            if out.data_ptr() % 16 or out.stride(0) * out.element_size() % 16:
                return None
            strided = out.stride(0) != self.out_features
            if a_scale is not None:
                if a.stride(0) % 16 or a.data_ptr() % 16:
                    return None
                return self._cute.bind(config, out.dtype, strided=strided)
            if not CUTE_CONFIGS[config].pdl:
                run_x = self._cute.run_x
                return lambda a, _, out: run_x(a, out, config)
            vectorized = a.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0
            return self._cute.bind(
                config, out.dtype, strided=strided, quantize=(a.dtype, vectorized)
            )
        if chosen == "flashinfer-cutlass":
            run = self._flashinfer.run
            if a_scale is not None:
                return run
            return lambda a, _, out: run(*quantize_per_token_group_fp8(a), out)
        if chosen == "cuda-sm12x":
            return self._sm12x.run
        return self._triton.run

    def _dispatch_key(
        self, rows: int, *, dtype=torch.bfloat16, out_dtype=torch.bfloat16, quantized=False
    ) -> tuple:
        return (
            torch.float8_e4m3fn if quantized else dtype,
            out_dtype,
            quantized,
            self._provider_dispatch_key(
                rows, dtype=dtype, out_dtype=out_dtype, quantized=quantized
            ),
        )

    def _provider_dispatch_key(
        self, rows: int, *, dtype=torch.bfloat16, out_dtype=torch.bfloat16, quantized=False
    ) -> tuple:
        """M-dependent compiled signatures / cached state.

        Weight/device are fixed per instance. Input/output layouts and pointer
        alignment classes are exercised separately for every representative M.
        """
        from .cute._block_fp8_provider import CONFIGS
        from .triton._block_fp8_provider import quantizer_key

        chosen = self.provider_for(rows)
        prequantized = quantized or self._scale_ue8m0
        quantizer = None
        if not quantized and self._scale_ue8m0:
            quantizer = ("cute-act-quant", self._device, self.in_features, dtype, True, True)
        if chosen == "cute-fp8":
            config = CONFIGS[self._cute_config(rows)]
            if not prequantized:
                quantizer = (
                    ("cute-act-quant", self._device, self.in_features, dtype, True, False)
                    if config.pdl
                    else (
                        "triton",
                        quantizer_key(rows, self.in_features, self._capability, self._sms),
                    )
                )
            return chosen, self._cute._key(config, out_dtype), quantizer
        if chosen == "triton-fp8":
            return chosen, self._triton.dispatch_key(rows, quantized=prequantized), quantizer
        if chosen == "flashinfer-cutlass":
            # The library's own M classes; the Triton quantizer keeps its specializations.
            if not prequantized:
                quantizer = (
                    "triton",
                    quantizer_key(rows, self.in_features, self._capability, self._sms),
                )
            return chosen, self._flashinfer.row_class(rows), out_dtype, quantizer
        # CUDA's module is keyed by config/capability; M and LDO are runtime
        # arguments, with no per-M cached state. Quantizer mode selects run/fused.
        return chosen, self._sm12x._config, self._capability, prequantized, quantizer

    def _input_layout_key(self, m, layout, scale_layout=None, *, quantized=False):
        from .cute._block_fp8_provider import CONFIGS
        from .triton._block_fp8_provider import _int_key

        row, aligned = layout
        k = self.in_features
        chosen = self.provider_for(m)
        packed = m == 1 or row == k
        if not quantized:
            # Flattening singleton rows normalizes their unused row stride.
            row = k if m == 1 else row
            if self._scale_ue8m0 or (chosen == "cute-fp8" and CONFIGS[self._cute_config(m)].pdl):
                return "cute-quantizer", row % 8 == 0 and aligned
            if chosen == "cuda-sm12x":
                return "cuda-input", row % 8 == 0 and aligned
            if chosen in ("cute-fp8", "flashinfer-cutlass") or m > 16:
                # quantize() flattens contiguous whole groups before launching.
                if packed and k % GROUP == 0:
                    return "triton-quantizer-flat", aligned
                return "triton-quantizer", _int_key(row), aligned
            return "triton-fused", _int_key(row), aligned
        sr, sc, scale_aligned = scale_layout
        if chosen == "triton-fp8":
            return _int_key(row), aligned, _int_key(sr), _int_key(sc), scale_aligned
        if chosen == "flashinfer-cutlass":
            scale_packed = (m == 1 or sr == _groups(k)) and (sc == 1 or _groups(k) == 1)
            return "flashinfer-packed", packed and aligned, scale_packed and scale_aligned
        # CuTe and CUDA accept dynamic scale strides; only q staging changes.
        return "aligned-q", row >= k and row % 16 == 0 and aligned

    def _input_layouts(self, m, *, quantized=False, input_layouts="packed"):
        return _input_layout_representatives(
            self, m, quantized=quantized, input_layouts=input_layouts
        )

    def warm_up(
        self,
        max_m: int | None = None,
        *,
        batch_sizes: Sequence[int] | int | None = None,
        dtype: torch.dtype = torch.bfloat16,
        out_dtype: torch.dtype | None = None,
        quantized: bool = False,
        out_row_stride: int | None = None,
        input_layouts: str | Sequence = "packed",
    ) -> list[dict]:
        """Compile, without timing or tuning, every selected route; return the routes compiled.

        Return a list of route dictionaries with ``provider``, representative ``m``,
        ``bucket`` (the representative M interval), ``config`` (the dispatch key),
        ``input_layout`` and ``output_layout``. Call eagerly before capture on
        the capture stream; never synchronize.
        Exactly one of ``max_m`` and ``batch_sizes`` is required. ``input_layouts``
        defaults to "packed" (aligned contiguous x/q, packed or native MN-major scales).
        "all" opts into every deduplicated class. Explicit examples are (row_stride,
        misaligned) or, for quantized inputs, ((row_stride, misaligned),
        (scale_row_stride, scale_column_stride, misaligned)). Missing dispatch/layout
        pairs raise during capture; eager calls remain unrestricted.
        """
        self._check_frozen()
        with torch.cuda.device(self._device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("BlockFP8Linear: warm_up must run before capture")
            dtype = self._out_dtype(dtype, torch.bfloat16)
            out_dtype = self._out_dtype(out_dtype, torch.bfloat16 if quantized else dtype)
            sizes = _warm_up_sizes(
                "BlockFP8Linear",
                max_m,
                batch_sizes,
                lambda m: (
                    self._dispatch_key(m, dtype=dtype, out_dtype=out_dtype, quantized=quantized),
                    tuple(
                        key
                        for _, _, key in self._input_layouts(
                            m, quantized=quantized, input_layouts=input_layouts
                        )
                    ),
                ),
            )
            if out_row_stride is not None and (
                not isinstance(out_row_stride, int) or out_row_stride < self.out_features
            ):
                raise ValueError("BlockFP8Linear: out_row_stride must be an integer >= N")
            # Validate examples even for an empty batch list, before preparing providers.
            _selected_input_layouts(self.in_features, input_layouts, m=1, quantized=quantized)
            routes = []
            owner = self._warm_class()
            # Largest first: each smaller call's buffers then split the blocks the largest one
            # freed, so the allocator's reserve stays at one call's peak.
            for m in sorted(sizes, reverse=True):
                if not m:
                    continue
                outputs = [None]
                if out_row_stride is not None:
                    # Include the packed kernel and both base-alignment classes.
                    # +1 permits the offset view even when row_stride == N.
                    parent = torch.empty(
                        (m * out_row_stride + 1,), dtype=out_dtype, device=self._device
                    )
                    outputs.extend(
                        parent[offset:].as_strided((m, self.out_features), (out_row_stride, 1))
                        for offset in (0, 1)
                    )
                for layout, scale_layout, layout_key in self._input_layouts(
                    m, quantized=quantized, input_layouts=input_layouts
                ):
                    dispatch = self._dispatch_key(
                        m, dtype=dtype, out_dtype=out_dtype, quantized=quantized
                    )
                    classes = [
                        (owner, m, dispatch, layout_key, dtype, out_dtype, quantized, out_class)
                        for out_class in (
                            [None]
                            if out_row_stride is None
                            else [
                                None,
                                (out_row_stride, 0),
                                (out_row_stride, torch.finfo(out_dtype).bits // 8 % 16),
                            ]
                        )
                    ]
                    if all(key in _WARMED for key in classes):
                        # Another instance of this class already compiled and loaded these
                        # routes: keep only the per-instance work (workspace reservation).
                        if GraphWorkspace.active() is not None:
                            self._graph_workspace(m)
                        self._warmed_inputs.add((dispatch, layout_key))
                        routes.extend(
                            self._warm_route(
                                m, dispatch, layout_key, dtype, out_dtype, quantized, o
                            )
                            for o in classes
                        )
                        continue
                    x = _layout_tensor(
                        m,
                        self.in_features,
                        layout,
                        dtype=torch.float8_e4m3fn if quantized else dtype,
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
                    for out, key in zip(outputs, classes, strict=True):
                        if quantized:
                            self.quantized(x, scale, out_dtype=out_dtype, out=out)
                        else:
                            self(x, out_dtype=out_dtype, out=out)
                        _WARMED.add(key)
                        self._warmed_inputs.add((dispatch, layout_key))
                        routes.append(
                            self._warm_route(
                                m, dispatch, layout_key, dtype, out_dtype, quantized, key
                            )
                        )
            routes.sort(key=lambda route: route["m"])
            return routes

    def _warm_class(self) -> tuple:
        """Everything a warm-up launch depends on besides M and layouts: the weight's and scale
        grid's dtype, shape, strides and 16-byte alignment, the routing table and its providers."""
        weight, scale = self._weight, self._scale
        return (
            type(self).__name__,
            self._device,
            weight.dtype,
            tuple(weight.shape),
            weight.stride(),
            weight.data_ptr() % 16,
            scale.dtype,
            tuple(scale.shape),
            scale.stride(),
            scale.data_ptr() % 16,
            self._scale_ue8m0,
            self._provider,
            self._buckets,
            self._capability,
            self._sms,
            tuple(p is not None for p in (self._triton, self._flashinfer, self._sm12x, self._cute)),
        )

    def _warm_route(self, m, dispatch, layout_key, dtype, out_dtype, quantized, key) -> dict:
        out_class = key[-1]
        return {
            "provider": self.provider_for(m),
            "m": m,
            "bucket": (m, m),
            "config": dispatch,
            "input_layout": layout_key,
            "dtype": dtype,
            "out_dtype": out_dtype,
            "quantized": quantized,
            "output_layout": (self.out_features, 0) if out_class is None else out_class,
        }


# GB300 sweep (DSV4 wo_a, T=1-4, cold weights): thin N tiles give 8 * N / 16 programs streaming
# K; 1.6x the MoE default (BLOCK_N 64), bitwise the same.
_GROUPED_CONFIG = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128,
                   "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 6}  # fmt: skip
# The fixed routing (row t * groups + g to expert g) of every GroupedBlockFP8Linear on a device
# with ``groups`` groups: one table per (device, groups) whose prefixes serve every T. Growth
# doubles it; replaced tables stay alive (_RETIRED) because captured graphs may read them.
_ROUTES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ROUTES_ALIGNED: dict[tuple, int] = {}  # largest T whose alignment ran eagerly (compiled)
_RETIRED: list[tuple[torch.Tensor, torch.Tensor]] = []
_ROUTES_LOCK = threading.Lock()


def _grouped_routes(device: torch.device, groups: int, t: int) -> tuple:
    """(ids ``[t * groups, 1]`` int32, weights ``[t * groups, 1]`` FP32 ones): prefixes of the
    shared table, grown eagerly when needed."""
    rows, key = t * groups, (device, groups)
    table = _ROUTES.get(key)
    if table is None or table[0].shape[0] < rows:
        with _ROUTES_LOCK:
            table = _ROUTES.get(key)
            if table is None or table[0].shape[0] < rows:
                size = max(rows, 2 * table[0].shape[0]) if table is not None else rows
                size = -(-size // groups) * groups
                ids = torch.arange(groups, dtype=torch.int32, device=device)
                ids = ids.repeat(size // groups).view(size, 1)
                weights = torch.ones((size, 1), dtype=torch.float32, device=device)
                if table is not None:
                    _RETIRED.append(table)
                table = _ROUTES[key] = (ids, weights)
    return table[0][:rows], table[1][:rows]


class GroupedBlockFP8Linear:
    """``groups`` independent block-FP8 linears over equal slices of the input: the
    block-diagonal projection of DeepSeek-V4's attention output (``wo_a``).

    ``weight`` ``[groups * N, K]`` E4M3 (group ``g`` owns rows ``[g * N, (g + 1) * N)``) with
    the module's ``weight_scale_inv`` grid ``[groups * N / 128, K / 128]``; ``N`` and ``K``
    multiples of 128 and the weight packed. ``op(x)`` maps BF16 ``x`` ``[T, groups * K]`` (or
    ``[T, groups, K]``, unit inner stride) to new BF16 ``[T, groups * N]`` with ``y[t, g * N +
    n] = x[t, g * K:(g + 1) * K] . W_g[n]``, activations quantized per token and per 128
    columns with ``quantize_per_token_group_fp8``'s recipe and the GEMM's FP32 accumulation and
    single rounding as in ``BlockFP8Linear``.

    Providers (``provider=``): ``"triton-fp8"`` (default, compute capability 8.9 and newer):
    the block-FP8 grouped expert GEMM of ``moe_impl.fused_moe_kernel_triton`` with row
    ``t * groups + g`` of ``x`` viewed ``[T * groups, K]`` routed to expert ``g``, writing
    ``out`` viewed ``[T * groups, N]`` in place: quantize + GEMM, two launches, no staging, no
    split-K (bitwise run to run). ``"flashinfer-cutlass"`` (compute capability 10.x only):
    FlashInfer's grouped groupwise GEMM after staging ``x`` group-major with rows padded to a
    multiple of 4 (four launches). ``warm_up(max_t)`` compiles every call with ``T`` in
    ``1..max_t`` (and builds the segment tables); capturing a larger ``T`` raises
    ``RuntimeError``. The Triton routing is one table per (device, groups), shared by every
    instance, of ``O(max_t * groups)`` bytes; each call aligns it (one small launch). Scratch
    comes from the caching allocator. Frozen weights, inference only.
    """

    PROVIDERS = ("triton-fp8", "flashinfer-cutlass")

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        *,
        groups: int,
        provider: str | None = None,
    ) -> None:
        operation = "GroupedBlockFP8Linear"
        weight, scale = _check_weight(
            operation, weight, weight_scale_inv, ndim=2, require_cuda=False
        )
        rows, k = weight.shape
        if type(groups) is not int or groups < 1 or rows % groups:
            raise ValueError(f"{operation}: groups must be a positive divisor of {rows}")
        if (rows // groups) % GROUP or k % GROUP:
            raise ValueError(f"{operation}: N per group and K must be multiples of 128")
        if not weight.is_contiguous():
            raise ValueError(f"{operation}: the weight must be packed (contiguous)")
        if not weight.is_cuda:
            raise ValueError(f"{operation}: weight must be on a CUDA device")
        provider = provider or "triton-fp8"
        if provider not in self.PROVIDERS:
            raise ValueError(f"{operation}: provider must be one of {self.PROVIDERS}")
        if _cuda_capability(weight.device) < (8, 9):
            raise RuntimeError(f"{operation}: needs compute capability 8.9 or newer")
        self.groups, self.out_per_group, self.in_per_group = groups, rows // groups, k
        self.device, self.provider = weight.device, provider
        if provider == "flashinfer-cutlass":
            self._gemm = FlashInferGroupedGemm(
                weight, scale, groups=groups, capability=_cuda_capability(weight.device),
                operation=operation,
            )  # fmt: skip
        else:
            self._weight = weight.view(groups, rows // groups, k)
            # The grouped kernel reads FP32/BF16 scales: UE8M0 converts exactly, once.
            self._scale = scale.view(groups, -1, scale.shape[-1])
            if self._scale.dtype == torch.float8_e8m0fnu:
                self._scale = self._scale.float()
            self._max_t = 0

    @property
    def in_features(self) -> int:
        return self.groups * self.in_per_group

    @property
    def out_features(self) -> int:
        return self.groups * self.out_per_group

    def _route(self, t: int):
        """Fixed routing of ``T`` tokens: row ``t * groups + g`` to expert ``g``."""
        if t > self._max_t and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"GroupedBlockFP8Linear: warm_up T={t} before graph capture")
        from ..moe_impl import moe_align_block_size

        g = self.groups
        ids, weights = _grouped_routes(self.device, g, t)
        aligned = moe_align_block_size(ids, _GROUPED_CONFIG["BLOCK_SIZE_M"], g)
        return (_GROUPED_CONFIG, ids, weights, *aligned)

    def warm_up(self, max_t: int) -> None:
        """Compile every call with ``T`` in ``1..max_t``; eager."""
        x = torch.zeros((1, self.in_features), dtype=torch.bfloat16, device=self.device)
        if self.provider == "flashinfer-cutlass":
            for rows in range(4, -(-max_t // 4) * 4 + 1, 4):
                self._gemm.indptr(rows, self.device)
        else:
            from ..moe_impl import moe_align_block_size

            g, key = self.groups, (self.device, self.groups)
            ids, _ = _grouped_routes(self.device, g, max_t)
            # Each T's alignment compiles once per process (its launcher's specialization).
            for t in range(_ROUTES_ALIGNED.get(key, 0) + 1, max_t + 1):
                moe_align_block_size(ids[: t * g], _GROUPED_CONFIG["BLOCK_SIZE_M"], g)
            _ROUTES_ALIGNED[key] = max(_ROUTES_ALIGNED.get(key, 0), max_t)
            self._max_t = max(self._max_t, max_t)
        self(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        operation = "GroupedBlockFP8Linear"
        g, k, n = self.groups, self.in_per_group, self.out_per_group
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            raise TypeError(f"{operation}: x must be a BF16 tensor")
        if x.device != self.device:
            raise ValueError(f"{operation}: x is on {x.device}, the operation on {self.device}")
        if x.ndim == 3 and x.shape[1:] == (g, k):
            x = x.flatten(1)
        if x.ndim != 2 or x.shape[1] != g * k or (x.numel() and x.stride(-1) != 1):
            raise ValueError(f"{operation}: x must be [T, {g * k}] or [T, {g}, {k}]")
        _require_inference(operation, x)
        t = x.shape[0]
        out = torch.empty((t, g * n), dtype=torch.bfloat16, device=x.device)
        if t == 0:
            return out
        if self.provider == "flashinfer-cutlass":
            rows = -(-t // 4) * 4
            staged = torch.zeros((g, rows, k), dtype=torch.bfloat16, device=x.device)
            staged[:, :t].copy_(x.view(t, g, k).transpose(0, 1))
            q, scale = quantize_per_token_group_fp8(staged.view(g * rows, k))
            result = torch.empty((g * rows, n), dtype=torch.bfloat16, device=x.device)
            self._gemm.run(q, scale, result, rows)
            out.view(t, g, n).copy_(result.view(g, rows, n)[:, :t].transpose(0, 1))
            return out
        from ..moe_impl import fused_moe_kernel_triton

        config, ids, weights, sorted_ids, experts, total = self._route(t)
        q, scale = quantize_per_token_group_fp8(x.contiguous().view(t * g, k))
        fused_moe_kernel_triton(
            q, self._weight, out.view(t * g, 1, n), weights, ids, sorted_ids, experts, total,
            False, 1, config, torch.bfloat16, a_scale=scale, b_scale=self._scale,
            block_fp8_provider="triton-fp8",
        )  # fmt: skip
        return out


__all__ = [
    "BlockFP8Linear",
    "GroupedBlockFP8Linear",
    "HYBRID",
    "PROVIDERS",
    "default_provider",
    "dequantize_block_fp8",
    "dgrad_bf16",
    "quantize_per_token_group_fp8",
    "slice_block_fp8",
]
