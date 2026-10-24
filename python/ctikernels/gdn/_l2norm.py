# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/modules/l2norm.py` wrappers); see THIRD_PARTY_NOTICES.

"""FLA's D=128 L2 norm kernels with one fixed launch, bypassing runtime autotuning.

Each launch imports FLA after ``pin_fla()``; importing this module does not import FLA. Its JIT functions keep T runtime; NB is fixed so row counts do not recompile.
"""

from __future__ import annotations

import torch

from ._fla_fixed import CONFIGS, pin_fla

BT = 16
NUM_WARPS = 4
BD = 128
NB = 1


def _rows(x: torch.Tensor) -> int:
    if x.ndim == 0 or x.shape[-1] != BD:
        raise ValueError("l2norm requires D == 128")
    if not x.is_contiguous():
        raise ValueError("l2norm requires contiguous rows")
    return x.numel() // BD


def _launch_forward(x, y, rstd, eps=1e-6):
    """Launch the bare FLA JIT function; return its compiled kernel for spill inspection."""
    rows = x.numel() // BD
    if rows == 0:
        return None
    pin_fla()
    from fla.modules.l2norm import l2norm_fwd_kernel

    return l2norm_fwd_kernel.fn[((rows + BT - 1) // BT,)](
        x=x, y=y, rstd=rstd, eps=eps, T=rows,
        D=BD, BD=BD, NB=NB, **CONFIGS["l2norm_fwd_kernel"],
    )  # fmt: skip


def _launch_backward(y, rstd, dy, dx, eps=1e-6):
    """Launch the bare FLA JIT function; return its compiled kernel for spill inspection."""
    rows = y.numel() // BD
    if rows == 0:
        return None
    pin_fla()
    from fla.modules.l2norm import l2norm_bwd_kernel

    return l2norm_bwd_kernel.fn[((rows + BT - 1) // BT,)](
        y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=rows,
        D=BD, BD=BD, NB=NB, **CONFIGS["l2norm_bwd_kernel"],
    )  # fmt: skip


def l2norm_forward(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return y like x and FP32 inverse norms with x's leading shape, without host sync."""
    rows = _rows(x)
    flat = x.view(rows, BD)
    y = torch.empty_like(flat)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _launch_forward(flat, y, rstd, eps)
    return y.view(x.shape), rstd.view(x.shape[:-1])


def l2norm_backward(
    y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Return dx like y, using the rounded forward y and FP32 inverse norms FLA saves."""
    rows = _rows(y)
    _rows(dy)
    if dy.shape != y.shape:
        raise ValueError("dy must have y's shape")
    if rstd.shape != y.shape[:-1] or rstd.dtype != torch.float32 or not rstd.is_contiguous():
        raise ValueError("rstd must be contiguous FP32 with y's leading shape")
    flat = y.view(rows, BD)
    dx = torch.empty_like(flat)
    _launch_backward(flat, rstd, dy.view(rows, BD), dx, eps)
    return dx.view(y.shape)


__all__ = ["l2norm_forward", "l2norm_backward"]
