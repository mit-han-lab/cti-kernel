# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Training-side linear ops: frozen-base projections with trainable LoRA factors, and the
frozen NVFP4 W4A16 linear (:class:`NVFP4W4A16Linear`).

Tensor parallelism: without an ``exchange`` every op computes this rank's values and leaves the
collectives to the caller; given the group's :class:`Exchange`, the op performs the one
reduction its sharding needs (base and LoRA terms in the same sum).
- Column-parallel (``W`` holds this rank's output rows, ``x`` replicated): the output is this
  rank's slice; ``x``'s gradient is a partial sum (reduced by the exchange in the backward). A
  replicated A's gradient is likewise partial (reduced by the caller); B holds this rank's rows
  and its gradient is complete.
- Row-parallel (``W`` holds this rank's input columns, ``x`` is this rank's slice): the output is
  a partial sum, base and LoRA delta together (reduced by the exchange in the forward). The
  backward needs no communication: the output gradient is that of the reduced output, already
  replicated, so ``x``'s gradient is complete for this slice and A (this rank's columns) gets a
  complete gradient; a replicated B's gradient is partial (reduced by the caller).

NVFP4 W4A16
-----------

Frozen NVFP4 linear for training with BF16/FP16 activations (W4A16), first used by the
NVFP4 LM head: forward ``y = alpha * x . W^T`` and input gradient ``dx = alpha * dy . W``.

**Scales.** The forward's ``alpha`` is the input scale times the weight global scale (the
activation is fake-quantized without its global); ``input_grad``'s ``alpha`` is the weight
global scale alone (straight-through quantizer: ``dx = dy . W_hat * weight_global``).

**Weight.** An :class:`~ctikernels.linear.nvfp4.NVFP4Weight` ``[N, K]`` (checkpoint
layout, frozen; its ``global_scale`` is not read: the caller's ``alpha`` carries every
global). Construction repacks it once into the forward kernel's layout on the weight's
device (``prepared_bytes`` extra, about 0.56 bytes per element); the input gradient
reads the original weight in place. Blocks whose scale is zero contribute zero. ``K``
is a multiple of 16; the prepared copy is zero-padded to the forward kernel's tile
grain (N, K to multiples of 64, one of them of 128, as SGLang does), and calls on a
padded shape copy ``x`` into a padded buffer and the output out of one.

**Math.** Dequantized weights (E2M1 value times E4M3 scale) are exact in BF16 and FP16,
so every tensor-core product is exact; sums accumulate in FP32, the FP32 ``alpha``
(``[1]``, read on the device, e.g. input scale times weight global scale) multiplies the
sum, and the result is rounded once. Only the FP32 summation order differs from an FP32
reference. ``input_grad`` takes FP32 ``dy`` as ``dy_terms`` BF16 terms: 1 (default)
rounds it to BF16 once, like a dense head's ``dy.to(bf16) @ W``; 2 leave a 2^-16 relative
remainder; 3 are exact. BF16/FP16 ``dy`` enters as is (one term). The result is FP32,
BF16 or FP16. Equal inputs give equal outputs.

**Implementations**, chosen per call by direction and row count from the per-device table
``linear.nvfp4_w4a16.provider`` (default: forward ``marlin``; input gradient ``triton``
below 512 rows, ``dequant_cublas`` from 512):

- ``marlin``: SGLang 0.5.22's Marlin pruned to NVFP4 on a repacked copy (FP32 split-K
  reduction in slice order).
- ``triton``: the input gradient decoding weight tiles in place; with ``input_grad_copy=True``
  (off by default, another ~0.56 bytes per element) Marlin on a second copy repacked for row
  scales replaces it for one dy term and a BF16/FP16 output (FP32 dy rounded to that dtype
  first).
- ``dequant_cublas``: weight rows dequantized in chunks (Triton) into a scratch of
  ``chunk_bytes`` (default 256 MiB, with the matching activation chunk), cuBLAS products
  with FP32 output, FP32 accumulation across chunks for the input gradient, one rounding
  with ``alpha``. A larger budget makes fewer, faster chunks.

All run on compute capability 8.9 or newer. Zero rows return an empty result without
launching. Calls are CUDA-graph capturable once each kernel is built (one eager call per
input dtype and row count). Scratch comes from the active :class:`~ctikernels.GraphWorkspace`
when one is entered (run those warm-ups inside it; its slots grow to the largest object), else
from the object. One object's calls must run in order on one stream. No autograd: call ``input_grad`` from the caller's backward; an input that requires
grad under grad mode raises RuntimeError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .._runtime.tables import DeviceClass, Dim, TunedOp, pow2_edges
from .._runtime.workspace import GraphWorkspace
from .cuda import _marlin_nvfp4 as _marlin

if TYPE_CHECKING:
    from .block_fp8 import BlockFP8Linear
    from .nvfp4 import NVFP4Weight


class Exchange(Protocol):
    """A tensor-parallel group's reduction over ``world`` ranks. ``all_reduce(x)`` sums ``x``
    [M, N] over the group in place, in a fixed rank order so every rank holds the same bits, and
    returns it; ``all_reduce_async(x)`` enqueues the same and returns a handle whose ``wait()``
    makes the current stream wait for it. Every rank issues the same calls with the same
    shapes in the same order. :class:`ProcessGroupExchange` wraps a torch process group."""

    world: int

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    def all_reduce_async(self, x: torch.Tensor): ...


class ProcessGroupExchange:
    """:class:`Exchange` over a ``torch.distributed`` process group (NCCL on CUDA)."""

    def __init__(self, group=None) -> None:
        import torch.distributed as dist

        self._dist, self.group = dist, group
        self.world = dist.get_world_size(group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self._dist.all_reduce(x, group=self.group)
        return x

    def all_reduce_async(self, x: torch.Tensor):
        return self._dist.all_reduce(x, group=self.group, async_op=True)


class _Done:
    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self._event)


class CommStreamExchange:
    """:class:`Exchange` from a synchronous one (``world`` and an in-place ``all_reduce`` only,
    serialized per instance): every call runs on one dedicated CUDA
    stream, ordered against the caller's stream by events, so ``all_reduce_async`` overlaps the
    caller's next work. Eager and graph-capturable; one instance per exchange."""

    def __init__(self, exchange, stream: torch.cuda.Stream | None = None) -> None:
        self.exchange, self.world = exchange, exchange.world
        self.stream = torch.cuda.Stream() if stream is None else stream

    def all_reduce_async(self, x: torch.Tensor) -> _Done:
        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            self.exchange.all_reduce(x)
            x.record_stream(self.stream)
            done = torch.cuda.Event()
            done.record()
        return _Done(done)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.all_reduce_async(x).wait()
        return x


# How many row chunks a reduced product splits into, so that each chunk's reduce overlaps the
# next chunk's GEMM (1: one GEMM, then one reduce). Tuned offline per device class; the split
# is fixed by the shape and the table, so results are deterministic and identical on every rank
# (though not bitwise equal across chunk counts: the GEMM library may choose by row count).
REDUCE_CHUNKS = TunedOp(
    "linear.training.reduce_chunks",
    version=1,
    space={"chunks": (1, 2, 4, 8)},
    dims=(Dim("m", pow2_edges(1, 65536)), Dim("n"), Dim("world")),
    # Measured on an RTX PRO 6000 pair (TP2, NCCL, whole gated-MLP step, n=4096): 4 chunks win
    # from 1k rows (1.06x), 8 from 8k rows (1.15x); 2 never pays.
    default=lambda shape, device_class: {
        "chunks": 8 if shape["m"] >= 8192 else 4 if shape["m"] >= 1024 else 1
    },
)


def _reduce_chunks(rows: int, columns: int, exchange: Exchange, device: torch.device) -> int:
    shape = {"m": rows, "n": columns, "world": exchange.world}
    if device.type == "cuda":
        return REDUCE_CHUNKS.config(device, **shape)["chunks"]
    return REDUCE_CHUNKS.default(DeviceClass(0, 0, 0), **shape)["chunks"]


def _reduced_by_rows(out: torch.Tensor, exchange: Exchange, compute) -> torch.Tensor:
    """``compute(rows)`` fills ``out[rows]`` chunk by chunk; each chunk's all-reduce overlaps
    the next chunk's compute. Returns ``out``, summed over the group."""
    chunks = _reduce_chunks(out.shape[0], out.shape[1], exchange, out.device)
    if chunks == 1:
        compute(slice(None))
        return exchange.all_reduce(out)
    step = -(-out.shape[0] // chunks)
    handles = []
    for start in range(0, out.shape[0], step):
        rows = slice(start, start + step)
        compute(rows)
        handles.append(exchange.all_reduce_async(out[rows]))
    for handle in handles:
        handle.wait()
    return out


class FrozenProduct(Protocol):
    """One frozen base projection ``W`` (``[out, in]``) of a training op.

    - ``forward(x, *, out=None, exchange=None, epilogue=None)``: ``x`` is a ``[T, in]``
      activation of the op's dtype; returns ``x @ W^T`` plus the frozen bias if the base has
      one, as fresh, contiguous ``[T, out]`` storage of the same dtype that the caller owns, or
      written into ``out`` (contiguous ``[T, out]``) and returned. A quantized base returns what
      serving computes (its activation and weight quantization included). ``epilogue(rows,
      y)`` adds the caller's terms to the rows it is handed; with ``exchange`` the result,
      epilogue included, is summed over the group (row-parallel outputs) and the bias, if any,
      is added once after the sum.
    - ``grad_input(dy, *, out=None, accumulate=False, exchange=None)``: ``dy`` is a contiguous
      ``[T, out]`` gradient of the same dtype; returns ``dy @ W`` (``W`` the dequantized weight
      for a quantized base, with the activation quantizer passed straight through) as fresh
      ``[T, in]`` storage of that dtype, or written into ``out`` (``[T, in]``, that dtype, unit
      column stride) and returned; ``accumulate`` adds ``out``'s values first (one rounding
      where the kernel fuses it). With ``exchange`` the result (the accumulated values
      included) is summed over the group: column-parallel ``x`` gradients. The bias
      contributes nothing.

    With an exchange both may split the rows into chunks whose reduces overlap the next
    chunk's GEMM (see ``REDUCE_CHUNKS``). Neither call is differentiable nor mutates its input;
    both run on the current stream and are called with ``T >= 1``. The base receives no
    gradient. ``out_features`` and ``in_features`` give ``W``'s shape.
    """

    out_features: int
    in_features: int

    def forward(
        self,
        x: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        exchange: Exchange | None = None,
        epilogue=None,
    ) -> torch.Tensor: ...

    def grad_input(
        self,
        dy: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        exchange: Exchange | None = None,
    ) -> torch.Tensor: ...


class _Product:
    """The exchange and epilogue plumbing shared by the products; subclasses give the GEMMs
    ``_forward(x, out, bias)`` and ``_grad_input(dy, out, accumulate)`` on row blocks."""

    bias: torch.Tensor | None = None

    def forward(self, x, *, out=None, exchange=None, epilogue=None):
        if out is None:
            out = x.new_empty(x.shape[0], self.out_features)
        # A reduced (row-parallel) output takes the bias once, after the sum.
        bias = self.bias if exchange is None else None

        def compute(rows):
            self._forward(x[rows], out[rows], bias)
            if epilogue is not None:
                epilogue(rows, out[rows])

        if exchange is None:
            compute(slice(None))
            return out
        out = _reduced_by_rows(out, exchange, compute)
        return out if self.bias is None else out.add_(self.bias)

    def grad_input(self, dy, *, out=None, accumulate=False, exchange=None):
        if out is None:
            out, accumulate = dy.new_empty(dy.shape[0], self.in_features), False

        def compute(rows):
            self._grad_input(dy[rows], out[rows], accumulate)

        if exchange is None:
            compute(slice(None))
            return out
        return _reduced_by_rows(out, exchange, compute)


class DenseProduct(_Product):
    """A frozen unquantized ``weight`` (``[out, in]``) and optional ``bias`` (``[out]``) of the
    op's dtype, multiplied by cuBLAS."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.weight, self.bias = weight, bias
        self.out_features, self.in_features = weight.shape

    def _forward(self, x, out, bias):
        if bias is None:
            torch.mm(x, self.weight.t(), out=out)
        else:
            torch.addmm(bias, x, self.weight.t(), out=out)

    def _grad_input(self, dy, out, accumulate):
        if accumulate:
            out.addmm_(dy, self.weight)
        else:
            torch.mm(dy, self.weight, out=out)


class BlockFP8Product(_Product):
    """A frozen block-FP8 (W8A8, 128x128 weight blocks) base held by ``linear``: ``forward``
    is what serving computes (1x128 activation quantization included), ``grad_input`` the
    dequantized weight's ``dy . W_hat`` in-kernel (the activation quantizer passed straight
    through). BF16/FP16 activations; no bias."""

    def __init__(self, linear: BlockFP8Linear) -> None:
        self.linear = linear
        self.out_features, self.in_features = linear.out_features, linear.in_features

    def _forward(self, x, out, bias):
        self.linear(x, out=out)

    def _grad_input(self, dy, out, accumulate):
        self.linear.dgrad(dy, out=out, accumulate=accumulate)


def _scaled_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, scaling: float) -> torch.Tensor:
    """``scaling * a @ b`` computed into ``out`` (one allocation, the caller's)."""
    return torch.addmm(out, a, b, beta=0, alpha=scaling, out=out)


class _LoRAProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        base: FrozenProduct,
        scaling: float,
        exchange: Exchange | None,
        reduce: str,
        *factors: torch.Tensor,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        rows, width = x_2d.shape[0], sum(b.shape[0] for b in factors[1::2])
        if sum(b.shape[0] for b in factors[1::2]) != base.out_features:
            raise ValueError("LoRA output slices must cover the base projection exactly")
        # Factors (FP32 masters too) are saved as given and cast to x's dtype at each use.
        used = [f.to(x.dtype) for f in factors]
        low_rank = [F.linear(x_2d, a) for a in used[::2]]

        def add_low_rank(rows: slice, y: torch.Tensor) -> None:
            offset = 0
            for b, a_x in zip(used[1::2], low_rank, strict=True):
                y[:, offset : offset + b.shape[0]].addmm_(a_x[rows], b.t(), alpha=scaling)
                offset += b.shape[0]

        output = x_2d.new_empty(rows, width)
        if rows:  # the products need not accept zero rows
            row_exchange = exchange if reduce == "output" else None
            base.forward(x_2d, out=output, exchange=row_exchange, epilogue=add_low_rank)
        ctx.save_for_backward(x_2d, *factors, *low_rank)
        ctx.base, ctx.scaling, ctx.input_shape = base, scaling, x.shape
        ctx.exchange = exchange if reduce == "input_grad" else None
        return output.reshape(*x.shape[:-1], width)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x_2d, *saved = ctx.saved_tensors
        factors, low_rank = saved[: len(saved) * 2 // 3], saved[len(saved) * 2 // 3 :]
        factors = [f.to(x_2d.dtype) for f in factors]
        scaling = ctx.scaling
        grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        # The low-rank terms land in grad_x first; the base's dgrad accumulates onto them.
        grad_x = x_2d.new_empty(x_2d.shape) if ctx.needs_input_grad[0] else None
        grads, offset = [], 0
        for a, b, a_x in zip(factors[::2], factors[1::2], low_rank, strict=True):
            grad_slice = grad[:, offset : offset + b.shape[0]]
            grad_a_x = grad_slice.mm(b)
            grads.append(_scaled_mm(grad_a_x.t(), x_2d, torch.empty_like(a), scaling))
            grads.append(_scaled_mm(grad_slice.t(), a_x, torch.empty_like(b), scaling))
            if grad_x is not None:
                torch.addmm(grad_x, grad_a_x, a, beta=1 if offset else 0, alpha=scaling, out=grad_x)
            offset += b.shape[0]
        if grad_x is not None:
            if grad.shape[0]:
                ctx.base.grad_input(grad, out=grad_x, accumulate=True, exchange=ctx.exchange)
            grad_x = grad_x.reshape(ctx.input_shape)
        return grad_x, None, None, None, None, *grads


def lora_projection_training(
    x: torch.Tensor,
    base: FrozenProduct,
    lora_factors: tuple[torch.Tensor, ...],
    *,
    scaling: float,
    exchange: Exchange | None = None,
    reduce: Literal["input_grad", "output"] = "input_grad",
) -> torch.Tensor:
    """``base(x)`` plus each output slice's ``scaling * x A^T B^T``: the same math in any grad
    mode (reentrant checkpointing safe), with first-order gradients for ``x`` and the factors
    when grad mode is enabled.

    ``x`` is ``(..., in)``; ``lora_factors`` is ``(A0, B0, A1, B1, ...)``, every A ``(r, in)``,
    each B ``(slice, r)``, the B rows covering the base's output in order, on ``x``'s device,
    of ``x``'s dtype or FP32 (masters: saved as given, cast to ``x``'s dtype at each use; their
    gradients come back in ``x``'s dtype, which autograd casts to the factor's); see
    :class:`FrozenProduct` for ``base``. Returns fresh ``(..., out)``; inputs
    are not mutated. Saves ``x`` and the low-rank inputs only. GEMMs run in cuBLAS (and the
    base's own kernels): deterministic, host-sync free, first-order only. Zero rows return an
    empty output and zero factor gradients.

    Tensor parallelism (see the module docstring): with ``exchange``, ``reduce="input_grad"``
    (column-parallel) returns ``x``'s gradient summed over the group, and ``reduce="output"``
    (row-parallel) returns the summed output.
    """

    if not lora_factors or len(lora_factors) % 2:
        raise ValueError("LoRA factors must contain one or more A/B pairs")
    if reduce not in ("input_grad", "output"):
        raise ValueError(f"reduce must be 'input_grad' or 'output', not {reduce!r}")
    return _LoRAProjection.apply(x, base, scaling, exchange, reduce, *lora_factors)


def _triton():
    # Triton is imported on first use: the LoRA projections above load without it.
    from .triton import _nvfp4_w4a16

    return _nvfp4_w4a16


_ACTIVATIONS = (torch.bfloat16, torch.float16)
_GRADS = (torch.float32, torch.bfloat16, torch.float16)
# One-term FP32 dy up to this size is rounded to BF16 once into scratch rather than in
# every program that reads it (each output column block re-reads dy).
_STAGING_BYTES = 256 << 20
# Default weight-chunk scratch of the dequant + cuBLAS provider (BF16 weight rows plus the
# matching activation/output chunk); the dX accumulator comes on top.
CHUNK_BYTES = 256 << 20


_PROVIDERS = {"forward": ("marlin", "dequant_cublas"), "dx": ("triton", "dequant_cublas")}


def _default_provider(shape: dict, device_class) -> dict:
    if shape["direction"] == "forward":
        return {"provider": "marlin"}
    return {"provider": "triton" if shape["m"] < 512 else "dequant_cublas"}


# The provider per direction and row bucket, tuned per device class (offline A-B-A
# against the other provider; benchmark/kernel_tuning/nvfp4_w4a16.py). The input-grad
# Marlin copy (``input_grad_copy``) replaces "triton" where it is selectable.
PROVIDER = TunedOp(
    "linear.nvfp4_w4a16.provider",
    version=1,
    space={"provider": ("marlin", "triton", "dequant_cublas")},
    dims=(
        Dim("m", pow2_edges(1, 16384)),
        Dim("n", pow2_edges(1024, 1 << 20)),
        Dim("k", pow2_edges(256, 1 << 16)),
        Dim("direction"),
    ),
    default=_default_provider,
    valid=lambda config, shape: config["provider"] in _PROVIDERS[shape["direction"]],
)


def _check(name: str, value: object, dtypes, shape, device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"NVFP4W4A16Linear: {name} must be a tensor")
    if value.dtype not in dtypes:
        raise TypeError(f"NVFP4W4A16Linear: {name} must be one of {dtypes}, not {value.dtype}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"NVFP4W4A16Linear: {name} must have shape {shape}, not {value.shape}")
    if device is not None and value.device != device:
        raise ValueError(f"NVFP4W4A16Linear: {name} must be on {device}, not {value.device}")
    return value


def _take(group: GraphWorkspace, slot, call: tuple, rows: int, counts, size, allocate):
    """``slot``'s tensors from ``group``. Eager calls reserve ``size(max_rows, *call)``: each
    distinct ``call`` (an object's sizes) is covered, so the slot grows to the largest."""
    if not torch.cuda.is_current_stream_capturing():
        group.reserve(slot, call, size, allocate)
    return group.take(slot, rows, counts)


def _aligned(t: torch.Tensor) -> bool:
    return t.stride(-1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0


class NVFP4W4A16Linear:
    """A frozen NVFP4 weight ``[N, K]`` multiplied by BF16/FP16 activations."""

    def __init__(
        self,
        weight: NVFP4Weight,
        *,
        input_grad_copy: bool = False,
        chunk_bytes: int = CHUNK_BYTES,
    ) -> None:
        data, scale = weight.data, weight.scale
        if not isinstance(data, torch.Tensor) or data.device.type != "cuda":
            raise ValueError("NVFP4W4A16Linear: the weight must be on a CUDA device")
        n, k = data.shape[0], data.shape[1] * 2
        device = data.device
        _check("weight.data", data, (torch.uint8,), (n, k // 2), device)
        _check("weight.scale", scale, (torch.float8_e4m3fn,), (n, k // 16), device)
        if not (data.is_contiguous() and scale.is_contiguous()):
            raise ValueError("NVFP4W4A16Linear: weight data and scale must be contiguous")
        if k % 16 or n == 0:
            raise ValueError(f"NVFP4W4A16Linear: K={k} must be a multiple of 16 and N > 0")
        if torch.cuda.get_device_capability(device) < (8, 9):
            raise RuntimeError("NVFP4W4A16Linear needs compute capability 8.9 or newer")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("NVFP4W4A16Linear: chunk_bytes must be a positive int")
        self.weight = weight
        self.shape = (n, k)
        self.chunk_bytes = chunk_bytes
        self.device = device
        self._padded = _marlin.padded_shape(n, k)
        pad_n, pad_k = self._padded[0] - n, self._padded[1] - k
        if pad_n or pad_k:
            data = torch.nn.functional.pad(data, (0, pad_k // 2, 0, pad_n))
            scale = torch.nn.functional.pad(scale.view(torch.uint8), (0, pad_k // 16, 0, pad_n))
            scale = scale.view(torch.float8_e4m3fn)
        self._prepared = _marlin.repack(data, scale)
        self._prepared_rows = None
        if input_grad_copy:
            data, scale = weight.data, weight.scale.view(torch.uint8)
            self._padded_rows = _marlin.padded_shape(k, n)  # (K, N) of dy . W
            pad_k, pad_n = self._padded_rows[0] - k, self._padded_rows[1] - n
            if pad_n or pad_k:
                data = torch.nn.functional.pad(data, (0, pad_k // 2, 0, pad_n))
                scale = torch.nn.functional.pad(scale, (0, pad_k // 16, 0, pad_n))
            self._prepared_rows = _marlin.repack_rows(data, scale.view(torch.float8_e4m3fn))
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._scratch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._dy_staging: torch.Tensor | None = None
        self._chunk_scratch: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def provider(self) -> str:
        return _marlin.NAME

    @property
    def prepared_bytes(self) -> int:
        copies = (self._prepared, self._prepared_rows or ())
        return sum(t.numel() * t.element_size() for c in copies for t in c)

    def _counts(self) -> tuple[int, int]:
        partials, locks = _marlin.scratch_counts(self._sms)
        return max(partials, _triton().partial_count(self._sms)), locks

    def _allocate(self, counts) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(counts[0], dtype=torch.float32, device=self.device),
            torch.zeros(counts[1], dtype=torch.int32, device=self.device),
        )

    def _work(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        counts = self._counts()
        group = GraphWorkspace.active()
        if group is None:
            if self._scratch is None:
                self._scratch = self._allocate(counts)
            return self._scratch
        return _take(
            group, ("nvfp4_w4a16", self.device), (), rows, counts, lambda _: counts, self._allocate
        )

    def _staging(self, rows: int) -> torch.Tensor:
        n = self.shape[0]
        count = rows * n
        group = GraphWorkspace.active()
        if group is None:
            if self._dy_staging is None or self._dy_staging.numel() < count:
                self._dy_staging = torch.empty(count, dtype=torch.bfloat16, device=self.device)
            return self._dy_staging[:count].view(rows, n)
        cap = _STAGING_BYTES // (2 * n)
        work = _take(
            group,
            ("nvfp4_w4a16_dy", self.device),
            (cap, n),
            rows,
            (count,),
            lambda max_rows, cap, n: (min(max_rows, cap) * n,),
            lambda counts: (torch.empty(counts[0], dtype=torch.bfloat16, device=self.device),),
        )
        return work[0][:count].view(rows, n)

    def _provider(self, direction: str, m: int) -> str:
        n, k = self.shape
        config = PROVIDER.config(self.device, m=m, n=n, k=k, direction=direction)
        return config["provider"]

    def _require_no_autograd(self, x: torch.Tensor) -> None:
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError(
                "NVFP4W4A16Linear does not record autograd; call input_grad from your backward"
            )

    def __call__(
        self, x: torch.Tensor, alpha: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``alpha * x . W^T`` as ``x``'s dtype ``[M, N]``; ``x`` is ``[M, K]``."""
        n, k = self.shape
        _check("x", x, _ACTIVATIONS, None, self.device)
        if x.ndim != 2 or x.shape[1] != k:
            raise ValueError(f"NVFP4W4A16Linear: x must have shape [M, {k}], not {x.shape}")
        _check("alpha", alpha, (torch.float32,), None, self.device)
        if alpha.numel() != 1:
            raise ValueError("NVFP4W4A16Linear: alpha must have one element")
        self._require_no_autograd(x)
        m = x.shape[0]
        if out is None:
            out = torch.empty(m, n, dtype=x.dtype, device=self.device)
        _check("out", out, (x.dtype,), (m, n), self.device)
        if not (out.is_contiguous() and out.data_ptr() % 16 == 0):
            raise ValueError("NVFP4W4A16Linear: out must be contiguous and 16-byte aligned")
        if m == 0:
            return out
        if self._provider("forward", m) == "dequant_cublas":
            self._dequant_forward(x, alpha.reshape(1), out)
            return out
        padded_n, padded_k = self._padded
        if padded_k != k:
            x = torch.nn.functional.pad(x, (0, padded_k - k))
        elif not _aligned(x):
            x = x.clone(memory_format=torch.contiguous_format)
        target = out if padded_n == n else torch.empty(m, padded_n, dtype=x.dtype, device=x.device)
        partials, locks = self._work(m)
        _marlin.gemm(x, *self._prepared, alpha.reshape(1), target, partials, locks)
        if target is not out:
            out.copy_(target[:, :n])
        return out

    def input_grad(
        self,
        dy: torch.Tensor,
        alpha: torch.Tensor,
        *,
        out_dtype: torch.dtype = torch.float32,
        dy_terms: int = 1,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``alpha * dy . W`` as ``out_dtype`` ``[M, K]``; ``dy`` is ``[M, N]``."""
        n, k = self.shape
        _check("dy", dy, _GRADS, None, self.device)
        if dy.ndim != 2 or dy.shape[1] != n or dy.stride(1) != 1:
            raise ValueError(
                f"NVFP4W4A16Linear: dy must be [M, {n}] with unit inner stride, not {dy.shape}"
            )
        _check("alpha", alpha, (torch.float32,), None, self.device)
        if alpha.numel() != 1:
            raise ValueError("NVFP4W4A16Linear: alpha must have one element")
        if out_dtype not in _GRADS:
            raise TypeError(f"NVFP4W4A16Linear: out_dtype must be one of {_GRADS}")
        if dy_terms not in (1, 2, 3) or (dy_terms > 1 and dy.dtype != torch.float32):
            raise ValueError("NVFP4W4A16Linear: dy_terms is 1, or 2 or 3 for FP32 dy")
        self._require_no_autograd(dy)
        m = dy.shape[0]
        if out is None:
            out = torch.empty(m, k, dtype=out_dtype, device=self.device)
        _check("out", out, (out_dtype,), (m, k), self.device)
        if not out.is_contiguous():
            raise ValueError("NVFP4W4A16Linear: out must be contiguous")
        if m == 0:
            return out
        if dy_terms == 1 and self._provider("dx", m) == "dequant_cublas":
            self._dequant_input_grad(dy, alpha.reshape(1), out)
            return out
        if (
            self._prepared_rows is not None
            and dy_terms == 1
            and out_dtype in _ACTIVATIONS
            and dy.dtype in (torch.float32, out_dtype)
        ):
            self._marlin_input_grad(dy.to(out_dtype), alpha.reshape(1), out)
            return out
        if dy.dtype == torch.float32 and dy_terms == 1 and m * n * 2 <= _STAGING_BYTES:
            staged = self._staging(m)
            _triton().cast_bf16(dy, staged)
            dy = staged
        partials, _ = self._work(m)
        config = _triton().config_for(m, k, n, self._sms)
        _triton().input_grad(
            dy,
            self.weight.data,
            self.weight.scale,
            alpha.reshape(1),
            out,
            partials,
            dy_terms,
            config,
        )
        return out

    def _marlin_input_grad(self, dy, alpha, out) -> None:
        n, k = self.shape
        padded_k, padded_n = self._padded_rows
        if padded_n != n or not _aligned(dy):
            dy = torch.nn.functional.pad(dy, (0, padded_n - n))
        target = (
            out
            if padded_k == k
            else torch.empty(dy.shape[0], padded_k, dtype=out.dtype, device=out.device)
        )
        partials, locks = self._work(dy.shape[0])
        _marlin.gemm_rows(dy, *self._prepared_rows, alpha, target, partials, locks)
        if target is not out:
            out.copy_(target[:, :k])

    def _chunks(self, m: int, row_bytes: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Weight rows per chunk, the chunk scratch (bytes) and an FP32 ``[m, K]``
        accumulator; ``row_bytes`` per chunk row besides the weight's (activation chunk)."""
        n, k = self.shape
        per_row = 2 * k + row_bytes * m
        if per_row > self.chunk_bytes:
            raise ValueError(f"NVFP4W4A16Linear: chunk_bytes holds no weight row at M={m}")
        rows = self.chunk_bytes // per_row
        rows = rows // 64 * 64 or rows
        counts = (self.chunk_bytes, m * k)
        group = GraphWorkspace.active()
        if group is None:
            work = self._chunk_scratch
            if work is None or work[1].numel() < counts[1]:
                work = self._chunk_scratch = (
                    torch.empty(counts[0], dtype=torch.uint8, device=self.device),
                    torch.empty(counts[1], dtype=torch.float32, device=self.device),
                )
        else:
            work = _take(
                group,
                ("nvfp4_w4a16_chunks", self.device),
                (self.chunk_bytes, k),
                m,
                counts,
                lambda max_rows, budget, k: (budget, max_rows * k),
                lambda c: (
                    torch.empty(c[0], dtype=torch.uint8, device=self.device),
                    torch.empty(c[1], dtype=torch.float32, device=self.device),
                ),
            )
        return min(rows, n), work[0], work[1][: m * k].view(m, k)

    def _dequant_forward(self, x, alpha, out) -> None:
        # Per chunk of weight rows: dequantize, FP32 GEMM, one rounding with alpha.
        n, k = self.shape
        m = x.shape[0]
        rows, scratch, _ = self._chunks(m, 4)
        for start in range(0, n, rows):
            r = min(rows, n - start)
            w = scratch[: r * k * 2].view(x.dtype).view(r, k)
            tmp = scratch[rows * k * 2 : rows * k * 2 + m * r * 4].view(torch.float32).view(m, r)
            _triton().dequant(
                self.weight.data[start : start + r], self.weight.scale[start : start + r], w
            )
            torch.mm(x, w.T, out_dtype=torch.float32, out=tmp)
            _triton().scale_into(tmp, alpha, out[:, start : start + r])

    def _dequant_input_grad(self, dy, alpha, out) -> None:
        # Per chunk of weight rows: dequantize, round dy once, accumulate in FP32; then one
        # rounding with alpha.
        n, k = self.shape
        m = dy.shape[0]
        dtype = torch.float16 if dy.dtype == torch.float16 else torch.bfloat16
        rows, scratch, acc = self._chunks(m, 2)
        for start in range(0, n, rows):
            r = min(rows, n - start)
            w = scratch[: r * k * 2].view(dtype).view(r, k)
            _triton().dequant(
                self.weight.data[start : start + r], self.weight.scale[start : start + r], w
            )
            part = dy[:, start : start + r]
            if dy.dtype == torch.float32:
                rounded = scratch[rows * k * 2 : rows * k * 2 + m * r * 2].view(dtype).view(m, r)
                _triton().cast_bf16(part, rounded)
                part = rounded
            if start == 0:
                torch.mm(part, w, out_dtype=torch.float32, out=acc)
            else:
                torch.addmm(acc, part, w, out_dtype=torch.float32, out=acc)
        _triton().scale_into(acc, alpha, out)


__all__ = [
    "BlockFP8Product",
    "CommStreamExchange",
    "DenseProduct",
    "Exchange",
    "FrozenProduct",
    "NVFP4W4A16Linear",
    "ProcessGroupExchange",
    "lora_projection_training",
]
