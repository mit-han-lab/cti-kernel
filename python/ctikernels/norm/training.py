# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from typing import Literal

import torch
from liger_kernel.transformers import functional as _liger

from ._frozen_rms_norm import frozen_rms_norm as _frozen_rms_norm
from ._zero_centered_rms_norm import training_add_rms_norm as _zero_centered_add_rms_norm
from ._zero_centered_rms_norm import training_residual_rms_norm as _zero_centered_residual_rms_norm
from ._zero_centered_rms_norm import training_rms_norm as _zero_centered_rms_norm


def _require_frozen(operation: str, weight: torch.Tensor) -> None:
    if weight.requires_grad:
        raise ValueError(f"{operation} requires a frozen weight")


def _require_input_dtype(operation: str, name: str, dtype: torch.dtype, x: torch.Tensor) -> None:
    # Liger and the standalone no-dW kernel retain their same-dtype contract.
    if dtype != x.dtype:
        raise TypeError(f"{operation} needs zero_centered=True for {name} {dtype} with x {x.dtype}")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply differentiable RMS normalization on the current CUDA stream.

    ``x`` has shape ``[..., hidden_size]`` and ``weight`` has shape
    ``[hidden_size]``. Both tensors use one CUDA BF16 or FP32 dtype. Frozen
    weights use the local no-dW kernel; trainable weights retain Liger's dW
    path. The result is out-of-place, and backward does not reuse its incoming
    gradient storage.

    ``zero_centered`` scales by ``1 + weight`` instead of ``weight`` and takes the
    inputs of :func:`ctikernels.norm.inference.rms_norm` with that flag: FP16, BF16
    or FP32 ``x`` whose prefix collapses to two strides with a contiguous hidden
    dimension, such as Q/K heads of a merged projection, and a contiguous weight of
    any of those dtypes. Its weight must be frozen; one that requires gradients
    raises ``ValueError``, and no weight gradient is ever formed. The forward runs
    the inference kernel with the same ``out_dtype``, so the output equals the
    inference ``rms_norm`` bitwise. Backward saves only the input, recomputes the
    inverse RMS from it, and returns the input gradient of the FP32 graph
    (normalize, multiply by ``1 + weight``, round once), rounded once to ``x``'s
    dtype in fresh contiguous storage; an FP32 ``x`` thus gets an FP32 gradient
    whatever ``out_dtype`` is. Without the flag the output keeps ``x``'s dtype, and
    another ``out_dtype`` raises ``TypeError`` before launching.
    """

    if zero_centered:
        _require_frozen("rms_norm", weight)
        return _zero_centered_rms_norm(x, weight, eps, out_dtype=out_dtype)
    if out_dtype is not None:
        _require_input_dtype("rms_norm", "out_dtype", out_dtype, x)
    if not weight.requires_grad:
        return _frozen_rms_norm(x, weight, eps)

    return _liger.liger_rms_norm(
        x,
        weight,
        eps,
        casting_mode="llama",
        in_place=False,
    )


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    save: Literal["sum", "inputs"] = "sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a residual and apply differentiable RMS normalization.

    ``x`` and ``residual`` share shape ``[..., hidden_size]``; ``weight`` has
    shape ``[hidden_size]``. The returned pair is ``(normalized, summed)``.
    Inputs are not mutated or aliased, and backward does not reuse its incoming
    gradient storage.

    ``zero_centered`` scales by ``1 + weight`` instead of ``weight``. Zero-centered
    calls require frozen weights and use the residual-stream kernel. Standard
    weights use it when frozen and either the residual is FP32 with FP16/BF16 x,
    or ``save="inputs"``. It requires contiguous, 16-byte aligned rows, with the
    residual in x's dtype or FP32. Otherwise the existing Liger path is retained:
    same-dtype residual, ``save="sum"``, including frozen standard weights.
    Trainable standard weights with an FP32 residual and lower-precision x raise
    ``TypeError``; trainable weights with ``save="inputs"`` raise ``ValueError``.

    The residual-stream kernel normalizes the stored sum. For an FP32 residual
    this is the exact FP32 sum, so inference ``fp32_sum`` and ``stored_sum`` agree,
    and forced Triton inference equals this forward bitwise. CuTe decode uses a
    different reduction and inverse root: it satisfies the FP64 accuracy gate,
    without a bitwise training/inference requirement.
    For a narrow residual, inference's default skips the stored-sum rounding.

    Backward adds the norm's FP32 gradient and summed's gradient, then rounds once
    per input dtype. ``save="sum"`` keeps summed; ``save="inputs"`` keeps the two
    summands and rebuilds it bit for bit. Both give identical gradients. No weight
    gradient is formed by the residual-stream kernel.
    """

    if save not in ("sum", "inputs"):
        raise ValueError(f"save must be 'sum' or 'inputs', not {save!r}")
    if zero_centered or (
        not weight.requires_grad
        and (save == "inputs" or (residual.dtype == torch.float32 and x.dtype != torch.float32))
    ):
        _require_frozen("add_rms_norm", weight)
        return _zero_centered_add_rms_norm(
            x, residual, weight, eps, save_inputs=save == "inputs", zero_centered=zero_centered
        )
    if save != "sum":
        raise ValueError("add_rms_norm needs zero_centered=True to save the inputs")
    _require_input_dtype("add_rms_norm", "residual", residual.dtype, x)

    return _liger.liger_fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
        casting_mode="llama",
        in_place=False,
    )


def residual_rms_norm(
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    zero_centered: bool = False,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a residual without a fused add, with a frozen weight of either convention.

    Returns ``(normalized, residual)``. It normalizes a residual as it stands, such as the
    first layer's input or a sum a torch add produced: a checkpoint segment may end in that
    add, and the next segment or the final norm then starts here. The norm is that of
    :func:`ctikernels.norm.inference.residual_rms_norm`, on the same inputs and equal to it
    bitwise. Scaling is by ``weight``, or ``1 + weight`` with ``zero_centered=True``.
    ``residual`` comes back unchanged, under autograd as a view carrying this
    operation, so the gradient of its later uses joins the norm's in one backward launch,
    rounded once to the residual's dtype; callers carry the returned residual onward. The
    weight must be frozen; one that requires gradients raises ``ValueError``, and no weight
    gradient is ever formed.
    """

    _require_frozen("residual_rms_norm", weight)
    return _zero_centered_residual_rms_norm(
        residual, weight, eps, out_dtype=out_dtype, zero_centered=zero_centered
    )


__all__ = ["add_rms_norm", "residual_rms_norm", "rms_norm"]
