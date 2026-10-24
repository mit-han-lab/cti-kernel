# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Gated DeltaNet training: the varlen forward of packed sequences and its activation backward.

The operation covers what ``inference``'s prefill covers, everything between the frozen
``in_proj`` and ``out_proj`` GEMMs of a Qwen3.5 linear-attention layer (the causal conv with
SiLU, the q/k L2 norm, the gates, the delta-rule recurrence and the gated RMSNorm; formulas in
``inference``), for training: every sequence starts from a zero state and a zero conv history
and keeps no final state, and the result is differentiable in ``projected`` only.

**Geometry and tensors.** As ``inference``: per rank ``Hk`` key heads and ``Hv`` value heads
(a multiple of ``Hk``), K = V = 128, ``conv_dim = 2 Hk K + Hv V``, ``P = conv_dim + Hv V +
2 Hv``; Qwen3.5-9B and 35B-A3B are (16, 32), Qwen3.5-27B is (16, 48).

- ``projected`` ``[T, P]`` BF16, unit column stride: the merged ``in_proj`` rows of N packed
  sequences.
- Frozen weights: ``conv_weight`` ``[conv_dim, 4]`` BF16, ``A_log`` and ``dt_bias`` ``[Hv]``
  FP32, ``norm_weight`` ``[V]`` FP32, all contiguous. A weight that requires grad raises
  ValueError, in or out of grad mode, before anything else is checked and before any
  implementation is called; no weight gradient is ever formed.
- ``cu_seqlens`` int32 ``[N + 1]``, from 0 to T, every length at least 1.
- ``max_sequence_length``, keyword-only: a host integer from 1 to T, no smaller than any
  sequence. ``max_sequence_length`` times ``projected``'s row stride must stay below 2^31:
  a training-only bound kept until the provider audit. With dense rows (stride P) that is
  173,857 tokens per sequence at (16, 32) and 130,308 at (16, 48). A longer one raises
  ValueError before any implementation is called. Training's own kernels use int64 row
  offsets.
- Returns ``y`` ``[T, Hv V]`` BF16, new storage. Under grad mode with ``projected`` requiring
  grad, ``y`` carries an autograd node whose backward returns the gradient of ``projected``
  (``[T, P]`` BF16, sections ``d x_pre | dz | db | da``) and nothing for the weights; ``y``
  is not a view, so in-place ops on it are recorded as on any autograd result. Any
  other call (no grad mode, inference mode, or ``projected`` without grad) runs the forward
  alone and keeps nothing for a backward.

What the backward needs is kept through autograd's saved tensors, so activation checkpointing
drops it and recomputes it: the forward is deterministic, and a recompute equals the first
run bit for bit. Shape, dtype, stride and device violations raise TypeError or ValueError.
``cu_seqlens`` is not read on the host, and no call synchronises with the host.

**Implementations.** As ``inference``: resolved once at construction for the device and
geometry (``_training_kernel``, implementing ``_contract.TrainKernel``), never switched by a
call. The CuTe provider (``_cute``, ``provider == "cute"``) runs on sm_120 and sm_121
with the ``cti-kernels[cute]`` extra at the geometries in ``CUTE_GEOMETRIES`` ((16, 32),
(16, 48), (8, 16): it misses a per-case precision bound in the slow-decay backward elsewhere),
except where our Triton provider measured faster on the device
(``TRITON_FASTER``, by device name). Otherwise, or with ``CTI_GDN_CUTE=0``, our Triton provider (``triton.training``,
``provider == "triton"``) runs on compute capabilities 10.0 to 12.1, and the fla-core
provider (``_fla``, ``provider == "fla-core"``, ``cti-kernels[gdn]``) on the rest, refused
where it computes wrongly (``_fla.WRONG_ON``). All three use the shared conv input gradient:
Gluon only on devices named ``NVIDIA GeForce RTX 5090``, ``NVIDIA RTX PRO 6000 Blackwell
Server Edition`` and ``NVIDIA GB10``; Triton on all other devices (``_fla_training``).
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from ._contract import Geometry, TrainKernel, check_training, cuda_device, require_frozen

# Per-rank geometries where the CuTe provider passes every per-case precision bound of
# tests/gpu/test_gdn_cute.py on RTX 5090, RTX PRO 6000 and GB10 (2026-10-17).
CUTE_GEOMETRIES = frozenset({Geometry(16, 32), Geometry(16, 48), Geometry(8, 16)})
# By device name, the CuTe geometries where our Triton provider measured faster on that card
# (benchmark/kernel_tuning/gdn_training_triton_bench.py, A-B-A); Triton runs there.
TRITON_FASTER = {
    "NVIDIA GB10": CUTE_GEOMETRIES,
    "NVIDIA GeForce RTX 5090": CUTE_GEOMETRIES,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": CUTE_GEOMETRIES,
}
# The compute capabilities our Triton provider covers (one interval); fla-core outside it.
TRITON_CAPABILITIES = ((10, 0), (12, 1))


def _triton(device: torch.device) -> TrainKernel | None:
    """Our Triton provider on ``device`` within ``TRITON_CAPABILITIES``, else None (and None
    in a process without CUDA)."""
    if not torch.cuda.is_available():
        return None
    low, high = TRITON_CAPABILITIES
    capability = torch.cuda.get_device_capability(device)
    if not low <= capability <= high:
        return None
    from .triton.training import Training

    return Training(capability)


def _cute_geometry(geometry: Geometry, device: torch.device) -> bool:
    """Whether CuTe is the pick for ``geometry`` on ``device`` (where it resolves; without
    CUDA it never does, and the card is not asked)."""
    if geometry not in CUTE_GEOMETRIES or not torch.cuda.is_available():
        return geometry in CUTE_GEOMETRIES
    return geometry not in TRITON_FASTER.get(torch.cuda.get_device_name(device), frozenset())


def _training_kernel(geometry: Geometry, device: torch.device) -> TrainKernel:
    """Training's implementation for ``geometry`` on ``device``: CuTe where it resolves and
    ``_cute_geometry`` picks it, else Triton within ``TRITON_CAPABILITIES``, else fla-core."""
    from . import _cute

    kernel = _cute.training(device) if _cute_geometry(geometry, device) else None
    if kernel is None:
        kernel = _triton(device)
    if kernel is not None:
        return kernel
    from . import _fla

    return _fla.load(GatedDeltaNetTraining.__name__, device)


class _GatedDeltaNetFunction(torch.autograd.Function):
    """The autograd node of a training call: the gradient of ``projected`` only, from the
    kernel's backward over the tensors its forward saved."""

    @staticmethod
    def forward(ctx, kernel: TrainKernel, geometry: Geometry, projected, *rest):
        y, saved = kernel.train_forward(geometry, projected, *rest)
        ctx.kernel, ctx.geometry = kernel, geometry
        ctx.save_for_backward(*saved)
        # A view returned from here (the provider's y is its norm output reshaped) could not be
        # modified in place under autograd; the same storage without the view is.
        return y.detach()

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        grad = ctx.kernel.train_backward(ctx.geometry, ctx.saved_tensors, dy.contiguous())
        # kernel, geometry, projected, the four weights, cu_seqlens
        return None, None, grad, None, None, None, None, None


class GatedDeltaNetTraining:
    """Varlen training forward of N sequences packed in ``projected`` ``[T, P]``,
    differentiable in ``projected`` only, each from a zero state and history.

    ``op(projected, conv_weight, A_log, dt_bias, norm_weight, cu_seqlens, *,
    max_sequence_length) -> y``; the module docstring is the full contract.
    """

    def __init__(
        self, *, num_k_heads: int, num_v_heads: int, device: torch.device | str | int
    ) -> None:
        name = type(self).__name__
        self._geometry = Geometry(num_k_heads, num_v_heads)
        self._device = cuda_device(name, device)
        self._kernel = _training_kernel(self._geometry, self._device)

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def provider(self) -> str:
        """The implementation resolved at construction: ``"cute"``, ``"triton"`` or
        ``"fla-core"``."""
        return getattr(self._kernel, "provider", "fla-core")

    def training_bytes(self, tokens: int, sequences: int) -> tuple[int, int]:
        """``(saved, backward)`` bytes of a training call on ``tokens`` rows of at most
        ``sequences`` sequences, for the provider resolved at construction: ``saved`` is what
        the forward keeps for the backward (``projected`` included, the weights not) and
        ``backward`` the backward's peak above that at its start (dy in place, d projected
        included). Requested payload bytes before the allocator's rounding, an upper bound
        from the provider's own save list and allocations; no allocation, no synchronisation."""
        if tokens < 0 or sequences < 1:
            raise ValueError(f"training_bytes needs tokens >= 0 and sequences >= 1, not "
                             f"{tokens} and {sequences}")  # fmt: skip
        return self._kernel.training_bytes(self._geometry, tokens, sequences)

    def __call__(
        self,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        max_sequence_length: int,
    ) -> torch.Tensor:
        weights = {
            "conv_weight": conv_weight,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "norm_weight": norm_weight,
        }
        require_frozen(type(self).__name__, weights)
        check_training(
            self._geometry,
            self._device,
            projected,
            *weights.values(),
            cu_seqlens,
            max_sequence_length,
        )
        if torch.is_grad_enabled() and projected.requires_grad:
            return _GatedDeltaNetFunction.apply(
                self._kernel, self._geometry, projected, *weights.values(), cu_seqlens
            )
        y, _ = self._kernel.train_forward(self._geometry, projected, *weights.values(), cu_seqlens)
        return y


__all__ = ["GatedDeltaNetTraining"]
