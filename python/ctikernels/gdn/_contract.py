# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The Gated DeltaNet contract shared by every implementation of ``ctikernels.gdn``: per-rank
geometry, the frozen-weight rule, the checks of each call and the protocols implementations
satisfy.

Implementations (built-in kernels and explicit external providers) receive
inputs that already passed these checks, so they never repeat them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

K = V = 128  # key and value head dimensions
CONV = 4  # causal conv width
HISTORY = CONV - 1  # pre-conv rows a slot keeps
EPS = 1e-6  # q/k L2 norm and gated RMSNorm epsilon


@dataclass(frozen=True, slots=True)
class Geometry:
    """Per-rank head counts; value head h reads key head h // (v_heads // k_heads)."""

    k_heads: int
    v_heads: int

    def __post_init__(self) -> None:
        for name in ("k_heads", "v_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, not {value!r}")
        if self.v_heads % self.k_heads:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")

    @property
    def key_width(self) -> int:
        return self.k_heads * K

    @property
    def value_width(self) -> int:
        return self.v_heads * V

    @property
    def conv_dim(self) -> int:
        """Channels of the causal conv: the q | k | v columns."""
        return 2 * self.key_width + self.value_width

    @property
    def proj_width(self) -> int:
        """Columns of the merged in_proj row: q | k | v | z | b | a."""
        return self.conv_dim + self.value_width + 2 * self.v_heads


class DecodeKernel(Protocol):
    """An implementation of decode, called with checked inputs (see ``inference``)."""

    def __call__(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor: ...


class PrefillKernel(Protocol):
    """An implementation of prefill, called with checked inputs (see ``inference``)."""

    def __call__(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor: ...


class TrainKernel(Protocol):
    """An implementation of training, called with checked inputs (see ``training``).

    ``train_forward`` returns y and ``saved``, the flat tuple its ``train_backward`` reads,
    kept through autograd's saved tensors (so checkpointing can drop and recompute it).
    ``train_backward`` returns the gradient of the merged projection rows only.
    ``training_bytes`` bounds a call's memory from its own save list and allocations.
    """

    def train_forward(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...

    def train_backward(
        self, geometry: Geometry, saved: Sequence[torch.Tensor], dy: torch.Tensor
    ) -> torch.Tensor: ...

    def training_bytes(self, geometry: Geometry, tokens: int, sequences: int) -> tuple[int, int]:
        """(saved, backward) payload bytes of a call on ``tokens`` rows of at most
        ``sequences`` sequences (``GatedDeltaNetTraining.training_bytes``)."""
        ...


def cuda_device(operation: str, device: torch.device | str | int) -> torch.device:
    """``device`` as an indexed CUDA device; any other device type raises ValueError."""
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{operation} runs on a CUDA device, not {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def require_frozen(operation: str, weights: dict[str, torch.Tensor]) -> None:
    """Base weights are frozen: a weight that requires grad raises ValueError, in or out of
    grad mode, before anything else is checked or launched. No weight gradient exists."""
    for name, weight in weights.items():
        if isinstance(weight, torch.Tensor) and weight.requires_grad:
            raise ValueError(
                f"{operation}: {name} is a frozen base weight and must not require grad"
            )


def require_no_autograd(operation: str, projected: torch.Tensor, pools: Sequence) -> None:
    """Inference only: no autograd graph through ``projected``, and pools updated in place
    must not require grad."""
    if torch.is_grad_enabled() and projected.requires_grad:
        raise RuntimeError(f"{operation} does not support autograd")
    if any(pool.requires_grad for pool in pools):
        raise RuntimeError(f"{operation} cannot update a pool that requires gradients")


class HashedTuple(tuple):
    """A tuple whose hash is computed once: host-path keys (compile keys, capture-guard
    entries) are looked up on every call, and tuples rehash all their items each time. It
    compares and hashes equal to the plain tuple."""

    def __new__(cls, items):
        self = super().__new__(cls, items)
        self._hash = tuple.__hash__(self)
        return self

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self):
        return tuple, (tuple(self),)


def overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the bounding byte spans of two tensors intersect (no device reads)."""
    if not a.numel() or not b.numel():
        return False

    def end(x: torch.Tensor) -> int:
        span = 1 + sum((n - 1) * s for n, s in zip(x.shape, x.stride(), strict=True))
        return x.data_ptr() + span * x.element_size()

    return a.data_ptr() < end(b) and b.data_ptr() < end(a)


def check_tensor(
    name: str,
    value: object,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
    *,
    contiguous: bool = True,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, not {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must be {shape}, not {tuple(value.shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, not {value.device}")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def check_weights(
    geometry: Geometry,
    device: torch.device,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
) -> None:
    check_tensor("conv_weight", conv_weight, torch.bfloat16, (geometry.conv_dim, CONV), device)
    check_tensor("A_log", A_log, torch.float32, (geometry.v_heads,), device)
    check_tensor("dt_bias", dt_bias, torch.float32, (geometry.v_heads,), device)
    check_tensor("norm_weight", norm_weight, torch.float32, (V,), device)


def check_projected(geometry: Geometry, device: torch.device, projected: object) -> int:
    """``projected`` [rows, P] BF16 with unit column stride (rows may be strided); its rows."""
    if not isinstance(projected, torch.Tensor):
        raise TypeError("projected must be a tensor")
    rows = projected.shape[0] if projected.dim() == 2 else -1
    check_tensor(
        "projected",
        projected,
        torch.bfloat16,
        (rows, geometry.proj_width),
        device,
        contiguous=False,
    )
    if projected.stride(1) != 1 or (rows > 1 and projected.stride(0) < geometry.proj_width):
        raise ValueError("projected rows must be dense with non-overlapping rows")
    return rows


def check_cu_seqlens(
    device: torch.device, cu_seqlens: object, rows: int, *, allow_empty: bool = False
) -> int:
    """Int32 [N+1]; training requires 1 <= N <= T. Inference allows empty
    sequences and N=0 only at T=0. Values (nondecreasing, 0..T) stay on device, except under
    CTI_ENGINE_CHECKS, which reads them back and raises ValueError unless they are.
    Provider grid limits are checked separately from these shared tensor checks.
    """
    if not isinstance(cu_seqlens, torch.Tensor):
        raise TypeError("cu_seqlens must be a tensor")
    if cu_seqlens.dim() != 1 or cu_seqlens.shape[0] < (1 if allow_empty else 2):
        raise ValueError("cu_seqlens must be [num_sequences + 1] with at least one sequence")
    count = cu_seqlens.shape[0] - 1
    if count == 0 and rows != 0:
        raise ValueError("nonempty projected requires at least one sequence")
    if not allow_empty and count > rows:
        raise ValueError("cu_seqlens describes more sequences than projected has rows")
    check_tensor("cu_seqlens", cu_seqlens, torch.int32, (count + 1,), device)
    from ctikernels._reroute import engine_checks

    if engine_checks():
        values = cu_seqlens.tolist()
        if (
            values[0] != 0
            or values[-1] != rows
            or any(b < a for a, b in zip(values, values[1:], strict=False))
        ):
            raise ValueError(
                f"cu_seqlens must rise from 0 to {rows} (projected rows), not {values}"
            )
    return count


def check_max_sequence_length(projected: torch.Tensor, rows: int, value: object) -> None:
    """``max_sequence_length``: a host integer from 1 to ``rows``, no smaller than any
    sequence (``cu_seqlens`` is not read to check that). Its product with ``projected``'s
    row stride must stay below 2^31: a training-only bound kept until the provider audit.
    Training's own kernels use int64 row offsets."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_sequence_length must be an integer")
    if not 1 <= value <= rows:
        raise ValueError("max_sequence_length must be within the packed token extent")
    if value * projected.stride(0) >= 2**31:
        raise ValueError(
            f"a sequence of {value} rows of projected (row stride {projected.stride(0)}) "
            "reaches 2^31 elements, a training-only bound kept until the provider audit"
        )


def check_pools(
    geometry: Geometry, device: torch.device, state: object, conv_state: object
) -> None:
    """The FP32 state pool [S, Hv, V, K] and the BF16 conv pool [S, 3, conv_dim]."""
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a tensor")
    slots = state.shape[0] if state.dim() == 4 else -1
    check_tensor("state", state, torch.float32, (slots, geometry.v_heads, V, K), device)
    check_tensor(
        "conv_state", conv_state, torch.bfloat16, (slots, HISTORY, geometry.conv_dim), device
    )


def check_decode(
    geometry: Geometry,
    device: torch.device,
    projected,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    slot_indices,
    state,
    conv_state,
) -> int:
    """The decode contract after the frozen rule; returns the batch size."""
    batch = check_projected(geometry, device, projected)
    check_weights(geometry, device, conv_weight, A_log, dt_bias, norm_weight)
    check_tensor("slot_indices", slot_indices, torch.int32, (batch,), device)
    check_pools(geometry, device, state, conv_state)
    return batch


def check_prefill(
    geometry: Geometry,
    device: torch.device,
    projected,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    cu_seqlens,
    slot_indices,
    has_initial_state,
    state,
    conv_state,
) -> tuple[int, int]:
    """The prefill contract after the frozen rule; returns (tokens, sequences)."""
    rows = check_projected(geometry, device, projected)
    check_weights(geometry, device, conv_weight, A_log, dt_bias, norm_weight)
    count = check_cu_seqlens(device, cu_seqlens, rows, allow_empty=True)
    if rows > 2**31 - 1:
        raise ValueError("prefill T must fit int32 cu_seqlens (T <= 2^31-1)")
    check_tensor("slot_indices", slot_indices, torch.int32, (count,), device)
    check_tensor("has_initial_state", has_initial_state, torch.bool, (count,), device)
    check_pools(geometry, device, state, conv_state)
    return rows, count


def check_training(
    geometry: Geometry,
    device: torch.device,
    projected,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    cu_seqlens,
    max_sequence_length,
) -> tuple[int, int]:
    """The training contract after the frozen rule; returns (tokens, sequences)."""
    rows = check_projected(geometry, device, projected)
    check_weights(geometry, device, conv_weight, A_log, dt_bias, norm_weight)
    count = check_cu_seqlens(device, cu_seqlens, rows)
    check_max_sequence_length(projected, rows, max_sequence_length)
    return rows, count


__all__ = [
    "CONV",
    "EPS",
    "HISTORY",
    "K",
    "V",
    "DecodeKernel",
    "Geometry",
    "HashedTuple",
    "PrefillKernel",
    "TrainKernel",
    "check_decode",
    "check_max_sequence_length",
    "check_prefill",
    "check_tensor",
    "check_training",
    "cuda_device",
    "require_frozen",
    "require_no_autograd",
]
