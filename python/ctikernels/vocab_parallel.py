# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Row-wise reduction layout inspired by Liger Kernel's cross-entropy kernel;
# vocab-shard statistics follow Megatron-Core's vocab_parallel_cross_entropy.
# Independently implemented; no third-party source copied.
"""Collective-free vocabulary shards. The engine owns merging and dX reduction."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from functools import cache
from numbers import Real
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from ._reroute import Provider, engine_checks, reroute

SHARD_CHUNK_SIZE = 1024


@cache
def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


@cache
def _triton_kernels() -> tuple[Any, Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=[
            "Z",
            "Y",
            "S",
            "P",
            "T",
            "IGNORE_INDEX",
            "V",
            "START",
            "Z_ROW",
            "P_ROW",
            "Y_STRIDE",
            "T_STRIDE",
            "INV_T",
        ]
    )
    def _stats_kernel(
        Z,
        Y,
        S,
        P,
        T,
        IGNORE_INDEX,
        V,
        START,
        Z_ROW,
        P_ROW,
        Y_STRIDE,
        T_STRIDE,
        INV_T,
        WRITE_P: tl.constexpr,
        PER_ROW_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        target = tl.load(Y + row * Y_STRIDE)
        valid = target != IGNORE_INDEX
        local = target - START
        if PER_ROW_T:
            temperature = tl.load(T + row * T_STRIDE)
        maximum = tl.full((), -float("inf"), tl.float32)
        total = tl.full((), 0, tl.float32)
        for offset in range(tl.cdiv(V, BLOCK)):
            indices = offset * BLOCK + cols
            values = tl.load(Z + row * Z_ROW + indices, indices < V, -float("inf")).to(tl.float32)
            if PER_ROW_T:
                values = tl.div_rn(values, temperature)
            else:
                values = values * INV_T
            next_max = tl.maximum(maximum, tl.max(values, 0))
            total = total * tl.exp(maximum - next_max) + tl.sum(tl.exp(values - next_max), 0)
            maximum = next_max
        selected = tl.load(
            Z + row * Z_ROW + local,
            valid & (local >= 0) & (local < V),
            0,
        ).to(tl.float32)
        if PER_ROW_T:
            selected = tl.div_rn(selected, temperature)
        else:
            selected = selected * INV_T
        tl.store(S + row * 3, tl.where(valid, maximum, 0.0))
        tl.store(S + row * 3 + 1, tl.where(valid, total, 1.0))
        tl.store(S + row * 3 + 2, tl.where(valid, selected, 0.0))
        if WRITE_P:
            for offset in range(tl.cdiv(V, BLOCK)):
                indices = offset * BLOCK + cols
                values = tl.load(
                    Z + row * Z_ROW + indices,
                    indices < V,
                    -float("inf"),
                ).to(tl.float32)
                if PER_ROW_T:
                    values = tl.div_rn(values, temperature)
                else:
                    values = values * INV_T
                probabilities = tl.where(valid, tl.exp(values - maximum), 0.0)
                tl.store(P + row * P_ROW + indices, probabilities, indices < V)

    @triton.jit(
        do_not_specialize=[
            "Z",
            "Y",
            "LSE",
            "G",
            "D",
            "S",
            "T",
            "IGNORE_INDEX",
            "V",
            "Z_ROW",
            "D_ROW",
            "START",
            "Y_STRIDE",
            "G_STRIDE",
            "SCALE",
            "INV_T",
            "T_STRIDE",
        ]
    )
    def _grad_kernel(
        Z,
        Y,
        LSE,
        G,
        D,
        S,
        T,
        IGNORE_INDEX,
        V,
        Z_ROW,
        D_ROW,
        START,
        Y_STRIDE,
        G_STRIDE,
        SCALE,
        INV_T,
        T_STRIDE,
        SAVED: tl.constexpr,
        APPLY_UPSTREAM: tl.constexpr,
        PER_ROW_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        target = tl.load(Y + row * Y_STRIDE)
        lse = tl.load(LSE + row * 2)
        upstream = SCALE
        if PER_ROW_T:
            temperature = tl.load(T + row * T_STRIDE)
            upstream = SCALE * tl.div_rn(1.0, temperature)
        if APPLY_UPSTREAM:
            upstream = tl.load(G + row * G_STRIDE) * upstream
        values = tl.load(Z + row * Z_ROW + cols, cols < V, 0).to(tl.float32)
        if SAVED:
            probability = values * tl.exp(tl.load(S + row * 3) - lse)
        else:
            if PER_ROW_T:
                values = tl.div_rn(values, temperature)
            else:
                values = values * INV_T
            probability = tl.exp(values - lse)
        grad = (probability - (cols == target - START)) * upstream
        grad = tl.where(target != IGNORE_INDEX, grad, 0.0)
        tl.store(D + row * D_ROW + cols, grad, cols < V)

    @triton.jit(
        do_not_specialize=[
            "A",
            "W",
            "Y",
            "S",
            "LSE",
            "G",
            "DX",
            "T",
            "H",
            "IGNORE_INDEX",
            "V",
            "START",
            "W_ROW",
            "W_COL",
            "Y_STRIDE",
            "G_STRIDE",
            "SCALE",
            "A_ROW",
            "DX_ROW",
            "T_STRIDE",
        ]
    )
    def _frozen_dx_kernel(
        A,
        W,
        Y,
        S,
        LSE,
        G,
        DX,
        T,
        H,
        IGNORE_INDEX,
        V,
        START,
        W_ROW,
        W_COL,
        Y_STRIDE,
        G_STRIDE,
        SCALE,
        A_ROW,
        DX_ROW,
        T_STRIDE,
        PER_ROW_T: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        target = tl.load(Y + row * Y_STRIDE)
        local = target - START
        valid = target != IGNORE_INDEX
        owned = valid & (local >= 0) & (local < V)
        average = tl.load(A + row * A_ROW + cols, cols < H, 0)
        selected = tl.load(W + local * W_ROW + cols * W_COL, owned & (cols < H), 0).to(tl.float32)
        factor = tl.exp(tl.load(S + row * 3) - tl.load(LSE + row * 2))
        scale = SCALE
        if PER_ROW_T:
            scale = SCALE * tl.div_rn(1.0, tl.load(T + row * T_STRIDE))
        upstream = tl.load(G + row * G_STRIDE) * scale
        value = tl.where(valid, (factor * average - selected) * upstream, 0.0)
        tl.store(DX + row * DX_ROW + cols, value, cols < H)

    return _stats_kernel, _grad_kernel, _frozen_dx_kernel


def _temperature_args(temperature: float | torch.Tensor) -> tuple:
    if isinstance(temperature, torch.Tensor):
        return temperature, 1.0, temperature.stride(0), True
    # Import lazily to avoid the logprob package's import of this module.
    # The provider helper itself imports neither Triton nor Cutlass.
    from .logprob.cute._provider import inverse_temperature

    return None, inverse_temperature(temperature), 0, False


def _scaled_logits(z: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    return z.float() / (
        temperature[:, None] if isinstance(temperature, torch.Tensor) else temperature
    )


def _resolve_provider(provider: str, cuda: bool) -> str:
    """auto -> Triton on CUDA; a CUDA run without Triton is a visible reroute."""
    if provider != "auto":
        return Provider(provider)
    if not cuda:
        return Provider("torch")  # private CPU reference path
    if _triton_available():
        return Provider("triton")
    return reroute("vocab_parallel", "torch", "Triton is not installed")


def _validate_provider(provider: str) -> None:
    if provider not in {"auto", "torch", "triton"}:
        raise ValueError("provider must be auto, torch or triton")
    if provider == "triton" and not _triton_available():
        raise RuntimeError("provider='triton' requires Triton")


def _matrix(value: torch.Tensor, shape: tuple[int, ...], device: torch.device, name: str) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use float32")
    if value.device != device:
        raise ValueError(f"{name} must share the input device")


def _output(out: torch.Tensor | None, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if out is None:
        return torch.empty(shape, device=device, dtype=torch.float32)
    _matrix(out, shape, device, "out")
    if out.requires_grad:
        raise ValueError("out must not require gradients")
    return out


@torch.no_grad()
def vocab_shard_sum_terms(
    stats: torch.Tensor, global_max: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Return rescaled sumexp and target for the engine's SUM all-reduce."""
    if stats.ndim != 2 or stats.shape[1] != 3:
        raise ValueError("stats must have shape [N, 3]")
    _matrix(stats, tuple(stats.shape), stats.device, "stats")
    _matrix(global_max, (stats.shape[0],), stats.device, "global_max")
    result = _output(out, (stats.shape[0], 2), stats.device)
    scaled = stats[:, 1] * torch.exp(stats[:, 0] - global_max)
    target = stats[:, 2].clone()
    result[:, 0].copy_(scaled)
    result[:, 1].copy_(target)
    return result


@torch.no_grad()
def vocab_shard_merged_from_sums(
    global_max: torch.Tensor, sums: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Finalize MAX/SUM route into [global lse, global target logit]."""
    if sums.ndim != 2 or sums.shape[1] != 2:
        raise ValueError("sums must have shape [N, 2]")
    _matrix(sums, tuple(sums.shape), sums.device, "sums")
    _matrix(global_max, (sums.shape[0],), sums.device, "global_max")
    result = _output(out, tuple(sums.shape), sums.device)
    lse = global_max + sums[:, 0].log()
    target = sums[:, 1].clone()
    result[:, 0].copy_(lse)
    result[:, 1].copy_(target)
    return result


@torch.no_grad()
def merge_vocab_shard_stats(
    gathered: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Merge [tp, N, 3] in explicit rank order, independent of local rank."""
    if gathered.ndim != 3 or gathered.shape[0] < 1 or gathered.shape[2] != 3:
        raise ValueError("gathered must have shape [tp >= 1, N, 3]")
    _matrix(gathered, tuple(gathered.shape), gathered.device, "gathered")
    maximum = gathered[0, :, 0].clone()
    for rank in range(1, gathered.shape[0]):
        torch.maximum(maximum, gathered[rank, :, 0], out=maximum)
    sums = vocab_shard_sum_terms(gathered[0], maximum)
    for rank in range(1, gathered.shape[0]):
        sums.add_(vocab_shard_sum_terms(gathered[rank], maximum))
    return vocab_shard_merged_from_sums(maximum, sums, out=out)


def _mm(
    left: torch.Tensor, right: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if left.is_cuda and left.dtype in {torch.bfloat16, torch.float16}:
        return torch.mm(left, right, out_dtype=torch.float32, out=out)
    return torch.mm(left.float(), right.float(), out=out)


def _logits(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    fp32_logits: bool,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not fp32_logits:
        return torch.mm(x, w.t(), out=out) if bias is None else torch.addmm(bias, x, w.t(), out=out)
    if bias is None:
        return _mm(x, w.t(), out=out)
    if x.is_cuda and x.dtype in {torch.bfloat16, torch.float16}:
        return torch.addmm(bias.float(), x, w.t(), out_dtype=torch.float32, out=out)
    return torch.addmm(bias.float(), x.float(), w.float().t(), out=out)


def _torch_stats(
    z: torch.Tensor,
    targets: torch.Tensor,
    start: int,
    stats: torch.Tensor,
    probabilities: torch.Tensor | None,
    temperature: float | torch.Tensor,
    ignore_index: int,
) -> None:
    z = _scaled_logits(z, temperature)
    maximum = z.max(dim=1).values
    shifted = (z - maximum[:, None]).exp_()
    valid = targets != ignore_index
    local = targets - start
    owned = valid & (local >= 0) & (local < z.shape[1])
    selected = z.gather(1, local.clamp(0, z.shape[1] - 1)[:, None])[:, 0]
    stats[:, 0] = torch.where(valid, maximum, 0.0)
    stats[:, 1] = torch.where(valid, shifted.sum(dim=1), 1.0)
    stats[:, 2] = torch.where(owned, selected, 0.0)
    if probabilities is not None:
        probabilities.copy_(torch.where(valid[:, None], shifted, 0.0))


@dataclass
class VocabShardPartials:
    """One forward's stats and private autograd state; finalize exactly once.

    Treat stats and saved inputs as read-only until backward finishes. The merge
    tensors are communication buffers, deliberately outside the autograd graph.
    Saved projection state holds logits in logits dtype for trainable heads
    and CE none, or unnormalized head-dtype probabilities for other frozen heads.
    """

    stats: torch.Tensor
    _hidden: torch.Tensor = field(repr=False)
    _weight: torch.Tensor = field(repr=False)
    _targets: torch.Tensor = field(repr=False)
    _bias: torch.Tensor | None = field(repr=False)
    _average: torch.Tensor | None = field(repr=False)
    _saved_projection: torch.Tensor | None = field(repr=False)
    _vocab_start: int
    _ignore_index: int
    _temperature: float | torch.Tensor
    _reduction: str
    _chunk_size: int
    _provider: str
    _kind: str
    _fp32_logits: bool
    _vocab_block_size: int | None
    _finalized: bool = False

    @property
    def provider(self) -> str:
        """Resolved kernel provider and saved-buffer policy for this forward."""
        mode = "saved" if self._saved_projection is not None else "recompute"
        return Provider(f"{self._provider}:{mode}", getattr(self._provider, "reason", None))


def _validate_saved_bytes_budget(value: int | None) -> int:
    if value is None:
        return 0
    if type(value) is not int:
        raise TypeError("shard_saved_bytes_budget must be a non-negative integer or None")
    if value < 0:
        raise ValueError("shard_saved_bytes_budget must be non-negative")
    return value


def _validate_target_ids(target_ids: torch.Tensor, ignore_index: int, vocab_size: int) -> None:
    if not engine_checks():
        return
    if target_ids.is_cuda:
        with torch.cuda.device(target_ids.device):
            if torch.cuda.is_current_stream_capturing():
                # CTI_ENGINE_CHECKS validates eager calls; target ids replayed in a
                # captured CUDA graph are the engine's responsibility (no device checks).
                return
    targets = target_ids.detach().cpu()
    invalid = (targets < 0) | (targets >= vocab_size)
    rows = ((targets != ignore_index) & invalid).nonzero()
    if rows.numel():
        raise ValueError(f"invalid target id at row {rows[0, 0].item()}")


def _shard_partials(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_start: int,
    vocab_size: int,
    ignore_index: int = -100,
    bias: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
    reduction: str = "sum",
    chunk_size: int = SHARD_CHUNK_SIZE,
    provider: str = "auto",
    kind: str = "ce",
    fp32_logits: bool = False,
    save_probabilities: bool | None = None,
    shard_saved_bytes_budget: int | None = None,
    vocab_block_size: int | None = None,
    _allow_cpu: bool = False,
) -> VocabShardPartials:
    """Private CPU reference entry; public operators always require CUDA.

    ``save_probabilities`` retains projected logits when weight or bias trains
    or CE reduction is none, and unnormalized head-dtype probabilities otherwise. Saved state supports
    one backward per forward, with no projection recomputation. None resolves
    from the retained matrix size and ``shard_saved_bytes_budget``; None/zero
    budgets never save automatically.
    """
    _validate_provider(provider)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if type(fp32_logits) is not bool:
        raise TypeError("fp32_logits must be a bool")
    if save_probabilities is not None and type(save_probabilities) is not bool:
        raise TypeError("save_probabilities must be a bool or None")
    budget = _validate_saved_bytes_budget(shard_saved_bytes_budget)
    if vocab_block_size is not None and (
        type(vocab_block_size) is not int or vocab_block_size <= 0
    ):
        raise ValueError("vocab_block_size must be a positive integer or None")
    if type(vocab_start) is not int:
        raise TypeError("vocab_start must be a non-negative Python int")
    if vocab_start < 0:
        raise ValueError("vocab_start must be non-negative")
    if kind not in {"ce", "logprob"}:
        raise ValueError("unknown shard operation")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("hidden and weight_shard must be matrices")
    if weight.shape[0] < 1 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("weight_shard needs at least one row and matching hidden size")
    if target_ids.shape != (hidden.shape[0],):
        raise ValueError("target_ids must have shape [N]")
    if target_ids.dtype != torch.int64:
        raise TypeError("target_ids must use int64")
    dtypes = {torch.float32, torch.bfloat16, torch.float16}
    if hidden.dtype not in dtypes or weight.dtype != hidden.dtype:
        raise TypeError("hidden and weight_shard must share a supported floating dtype")
    if bias is not None and (bias.shape != (weight.shape[0],) or bias.dtype != weight.dtype):
        raise ValueError("bias must match the shard vocabulary and floating dtype")
    if kind == "logprob" and (weight.requires_grad or (bias is not None and bias.requires_grad)):
        raise ValueError("selected-token log-probabilities require a frozen vocabulary head")
    if any(
        t.device != hidden.device for t in (weight, target_ids) + (() if bias is None else (bias,))
    ):
        raise ValueError("all inputs must share one device")
    if not _allow_cpu and not hidden.is_cuda:
        raise ValueError("shard inputs must reside on CUDA")
    if provider == "triton" and not hidden.is_cuda:
        raise ValueError("Triton requires CUDA inputs")
    if type(ignore_index) is not int:
        raise TypeError("ignore_index must be a Python int")
    if not isinstance(reduction, str) or reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction must be sum, mean or none")
    if isinstance(temperature, torch.Tensor):
        if kind == "ce":
            raise TypeError("CE temperature must be a scalar real number")
        _matrix(temperature, (hidden.shape[0],), hidden.device, "temperature")
        capturing = False
        if engine_checks() and temperature.is_cuda:
            with torch.cuda.device(temperature.device):
                capturing = torch.cuda.is_current_stream_capturing()
        if engine_checks() and not capturing:
            values = temperature.detach().cpu()
            if not bool((torch.isfinite(values) & (values > 0)).all()):
                raise ValueError("temperature must be finite and positive")
    else:
        if isinstance(temperature, bool) or not isinstance(temperature, Real):
            raise TypeError("temperature must be a real number or an FP32 tensor [N]")
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
    if type(vocab_size) is not int:
        raise TypeError("vocab_size must be a Python int")
    if vocab_size < vocab_start + weight.shape[0]:
        raise ValueError("vocab_size must be >= vocab_start + V_local")
    _validate_target_ids(target_ids, ignore_index, vocab_size)
    resolved = _resolve_provider(provider, hidden.is_cuda)
    with torch.no_grad():
        stats = torch.empty((hidden.shape[0], 3), dtype=torch.float32, device=hidden.device)
        frozen = not weight.requires_grad and (bias is None or not bias.requires_grad)
        use_average = frozen and not (kind == "ce" and reduction == "none")
        if save_probabilities is None:
            saved_bytes = (
                hidden.shape[0]
                * weight.shape[0]
                * (4 if fp32_logits and not use_average else weight.element_size())
            )
            save_probabilities = budget > 0 and saved_bytes <= budget
        average = (
            torch.empty(hidden.shape, dtype=torch.float32, device=hidden.device)
            if hidden.requires_grad and use_average and not save_probabilities
            else None
        )
        # Saved modes write row chunks directly into the retained buffer: logits
        # for trainable heads, unnormalized P (over the logits in place when they
        # share the head dtype) for frozen heads. Every mode projects the same row
        # chunks, so the forward does not depend on the save policy.
        save_logits = save_probabilities and not use_average
        rows = chunk_size
        probabilities = (
            torch.empty(
                (min(rows, hidden.shape[0]), weight.shape[0]),
                dtype=weight.dtype,
                device=weight.device,
            )
            if average is not None and not save_probabilities
            else None
        )
        saved = None
        if save_probabilities:
            saved = torch.empty(
                (hidden.shape[0], weight.shape[0]),
                dtype=torch.float32 if save_logits and fp32_logits else weight.dtype,
                device=weight.device,
            )
        direct = saved is not None and (save_logits or not fp32_logits)
        for begin in range(0, hidden.shape[0], rows):
            end = min(begin + rows, hidden.shape[0])
            z = _logits(
                hidden[begin:end],
                weight,
                bias,
                fp32_logits,
                out=saved[begin:end] if direct else None,
            )
            p = probabilities[: end - begin] if probabilities is not None else None
            if save_probabilities and use_average:
                p = saved[begin:end]
            targets = target_ids[begin:end]
            temperature_chunk = (
                temperature[begin:end] if isinstance(temperature, torch.Tensor) else temperature
            )
            if resolved == "triton":
                stats_kernel, _, _ = _triton_kernels()
                t, inv_t, t_stride, per_row_t = _temperature_args(temperature_chunk)
                stats_kernel[(end - begin,)](
                    z,
                    targets,
                    stats[begin:end],
                    p,
                    t,
                    ignore_index,
                    weight.shape[0],
                    vocab_start,
                    z.stride(0),
                    0 if p is None else p.stride(0),
                    targets.stride(0),
                    t_stride,
                    inv_t,
                    p is not None,
                    per_row_t,
                    8192,
                    num_warps=16,
                    enable_fp_fusion=False,
                )
            else:
                _torch_stats(
                    z,
                    targets,
                    vocab_start,
                    stats[begin:end],
                    p,
                    temperature_chunk,
                    ignore_index,
                )
            if average is not None:
                _mm(p, weight, out=average[begin:end])
            del z
    return VocabShardPartials(
        stats,
        hidden,
        weight,
        target_ids,
        bias,
        average,
        saved,
        vocab_start,
        ignore_index,
        temperature,
        reduction,
        chunk_size,
        resolved,
        kind,
        fp32_logits,
        vocab_block_size,
    )


def _gradient(
    z: torch.Tensor,
    dz: torch.Tensor,
    targets: torch.Tensor,
    merged: torch.Tensor,
    stats: torch.Tensor,
    upstream: torch.Tensor,
    start: int,
    scale: float,
    ctx: Any,
    *,
    saved: bool = False,
    apply_upstream: bool = True,
) -> None:
    n, v = z.shape
    if ctx.provider == "triton":
        if n:
            _, grad_kernel, _ = _triton_kernels()
            t, inv_t, t_stride, per_row_t = _temperature_args(ctx.temperature)
            grad_kernel[(n, (v + 1023) // 1024)](
                z,
                targets,
                merged,
                upstream,
                dz,
                stats,
                t,
                ctx.ignore_index,
                v,
                z.stride(0),
                dz.stride(0),
                start,
                targets.stride(0),
                0 if upstream.ndim == 0 else upstream.stride(0),
                scale,
                inv_t,
                t_stride,
                saved,
                apply_upstream,
                per_row_t,
                1024,
                num_warps=8,
                enable_fp_fusion=False,
            )
        return
    if saved:
        value = z.float() * (stats[:, 0] - merged[:, 0]).exp()[:, None]
    else:
        value = (_scaled_logits(z, ctx.temperature) - merged[:, 0, None]).exp_()
    local = targets - start
    owned = (targets != ctx.ignore_index) & (local >= 0) & (local < v)
    value.scatter_add_(1, local.clamp(0, v - 1)[:, None], -owned.float()[:, None])
    if isinstance(ctx.temperature, torch.Tensor):
        scale = scale * (1.0 / ctx.temperature)
    factor = upstream * scale if apply_upstream else scale
    value *= factor.reshape(-1, 1) if isinstance(factor, torch.Tensor) else factor
    dz.copy_(torch.where((targets != ctx.ignore_index)[:, None], value, 0.0))


class _Finalize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        merged: torch.Tensor,
        part: VocabShardPartials,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            hidden,
            weight,
            bias,
            part._targets,
            part.stats,
            part._average,
            merged,
            part._saved_projection,
        )
        ctx.start = part._vocab_start
        ctx.ignore_index = part._ignore_index
        ctx.temperature = part._temperature
        ctx.reduction = part._reduction
        ctx.chunk_size = part._chunk_size
        ctx.provider = part._provider
        ctx.kind = part._kind
        ctx.fp32_logits = part._fp32_logits
        ctx.vocab_block_size = part._vocab_block_size
        ctx.consumed = False
        values = torch.where(part._targets != ctx.ignore_index, merged[:, 0] - merged[:, 1], 0.0)
        if part._kind == "logprob":
            return -values
        if ctx.reduction == "none":
            return values
        if ctx.reduction == "mean":
            ctx.count = (part._targets != ctx.ignore_index).sum()
            return values.sum() / ctx.count
        return values.sum()

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        if ctx.consumed:
            raise RuntimeError("saved projections support only one backward per forward")
        hidden, weight, bias, targets, stats, average, merged, saved = ctx.saved_tensors
        dx = (
            torch.empty_like(hidden, memory_format=torch.contiguous_format)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            torch.empty_like(weight, memory_format=torch.contiguous_format)
            if ctx.needs_input_grad[1]
            else None
        )
        db = torch.empty_like(bias) if bias is not None and ctx.needs_input_grad[2] else None
        if hidden.shape[0] == 0:
            ctx.consumed = saved is not None
            return (
                dx,
                None if dw is None else dw.zero_(),
                None if db is None else db.zero_(),
                None,
                None,
            )
        sign = 1.0 if ctx.kind == "ce" else -1.0
        per_row_t = isinstance(ctx.temperature, torch.Tensor)
        scale = sign if per_row_t else sign / ctx.temperature
        g = grad_output.float()
        if ctx.kind == "ce" and ctx.reduction == "mean":
            g = torch.where(ctx.count > 0, g / ctx.count, 0.0)
        per_row_g = ctx.kind == "logprob" or ctx.reduction == "none"
        if (
            saved is not None
            and dw is None
            and db is None
            and not (ctx.kind == "ce" and ctx.reduction == "none")
        ):
            # For frozen heads, normalize and subtract the target after the
            # GEMM to avoid rounding the dominant onehot term into head-dtype dz.
            ctx.consumed = True
            average = _mm(saved, weight)
        if average is not None and ctx.provider == "triton":
            if hidden.shape[0] and hidden.shape[1]:
                _, _, frozen_dx_kernel = _triton_kernels()
                t, _, t_stride, per_row_t = _temperature_args(ctx.temperature)
                frozen_dx_kernel[(hidden.shape[0], (hidden.shape[1] + 255) // 256)](
                    average,
                    weight,
                    targets,
                    stats,
                    merged,
                    g,
                    dx,
                    t,
                    hidden.shape[1],
                    ctx.ignore_index,
                    weight.shape[0],
                    ctx.start,
                    weight.stride(0),
                    weight.stride(1),
                    targets.stride(0),
                    g.stride(0) if per_row_g else 0,
                    scale,
                    average.stride(0),
                    dx.stride(0),
                    t_stride,
                    per_row_t,
                    256,
                    enable_fp_fusion=False,
                )
            return dx, None, None, None, None
        if average is not None:
            for begin in range(0, hidden.shape[0], ctx.chunk_size):
                end = min(begin + ctx.chunk_size, hidden.shape[0])
                y = targets[begin:end]
                valid = y != ctx.ignore_index
                local = y - ctx.start
                owned = valid & (local >= 0) & (local < weight.shape[0])
                upstream = g[begin:end] if per_row_g else g
                selected = weight[local.clamp(0, weight.shape[0] - 1)].float()
                value = (stats[begin:end, 0] - merged[begin:end, 0]).exp()[:, None] * average[
                    begin:end
                ]
                value -= torch.where(owned[:, None], selected, 0.0)
                row_scale = scale * (1.0 / ctx.temperature[begin:end]) if per_row_t else scale
                value *= (upstream * row_scale).reshape(-1, 1)
                dx[begin:end] = torch.where(valid[:, None], value, 0.0)
            return dx, None, None, None, None
        if saved is not None:
            ctx.consumed = True

        if ctx.kind != "ce":
            raise RuntimeError("trainable-head backward requires CE")
        from ._alpha_gemm import WORKSPACE_BYTES, alpha_gemm

        workspace = (
            torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=hidden.device)
            if dw is not None and hidden.is_cuda
            else None
        )
        # The binding supports row strides; pack only non-unit column strides.
        gemm_hidden = hidden.contiguous() if hidden.stride(1) != 1 else hidden

        # Each weight block is written once with K=N. Round down to keep the
        # vocabulary scratch near the forward row-chunk budget (minimum 128).
        n, v = hidden.shape[0], weight.shape[0]
        block = ctx.vocab_block_size
        if block is None and saved is not None and saved.dtype == weight.dtype:
            # Head-dtype saved logits become dz in place: no scratch to bound.
            block = v
        elif block is None:
            block = v if n <= ctx.chunk_size else max(128, (ctx.chunk_size * v // n) // 128 * 128)
        block = min(v, block)
        accumulator = torch.zeros_like(hidden, dtype=torch.float32) if dx is not None else None
        y = targets
        alpha = torch.ones((), dtype=torch.float32, device=hidden.device) if per_row_g else g
        for begin in range(0, v, block):
            end = min(begin + block, v)
            wb = weight[begin:end]
            z = (
                saved[:, begin:end]
                if saved is not None
                else _logits(hidden, wb, None if bias is None else bias[begin:end], ctx.fp32_logits)
            )
            dz = (
                z
                if z.dtype == weight.dtype
                else torch.empty_like(z, dtype=weight.dtype, memory_format=torch.contiguous_format)
            )
            _gradient(
                z,
                dz,
                y,
                merged,
                stats,
                g,
                ctx.start + begin,
                scale,
                ctx,
                apply_upstream=per_row_g,
            )
            if accumulator is not None:
                if dz.is_cuda and dz.dtype in {torch.bfloat16, torch.float16}:
                    torch.addmm(accumulator, dz, wb, out=accumulator, out_dtype=torch.float32)
                else:
                    accumulator.addmm_(dz.float(), wb.float())
            if dw is not None:
                alpha_gemm(dz, gemm_hidden, alpha, out=dw[begin:end], workspace=workspace)
            if db is not None:
                torch.mul(dz.sum(dim=0, dtype=torch.float32), alpha, out=db[begin:end])
            del z, dz
        if dx is not None:
            if per_row_g:
                dx.copy_(accumulator)
            else:
                torch.mul(accumulator, g, out=dx)
        return dx, dw, db, None, None


def _shard_finalize(part: VocabShardPartials, merged: torch.Tensor, *, kind: str) -> torch.Tensor:
    if not isinstance(part, VocabShardPartials) or part._kind != kind:
        raise TypeError("part must come from the matching shard operation")
    if part._finalized:
        raise RuntimeError("shard partials have already been finalized")
    _matrix(merged, (part.stats.shape[0], 2), part.stats.device, "merged")
    # Triton backward uses the documented contiguous two-column ABI.
    if not merged.is_contiguous():
        merged = merged.contiguous()
    result = _Finalize.apply(part._hidden, part._weight, part._bias, merged, part)
    part._finalized = True
    return result


__all__ = [
    "SHARD_CHUNK_SIZE",
    "VocabShardPartials",
    "merge_vocab_shard_stats",
    "vocab_shard_sum_terms",
    "vocab_shard_merged_from_sums",
]
