# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Design reference: fla-core 0.5.2's layer_norm_gated_fwd_kernel (RMS, SiLU gate); no kernel
# code is copied.

"""GDN's gated RMSNorm with z read in place from ``projected`` (no copy).

y[t, h] = o[t, h] / sqrt(mean(o[t, h]^2) + eps) * w * z * sigmoid(z) in FP32, rounded to BF16,
per (token, value head) row of V values; fla-core's operation order, the mean's sum may round
differently (tier-1 bounded)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import EPS, Geometry, V

_BR = 16  # (token, head) rows per program


@triton.jit(do_not_specialize=["rows"])
def _gated_norm_kernel(
    o, projected, p_stride, z_column, weight, y, rows,
    HV: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, D)
    m = (r < rows)[:, None]
    x = tl.load(o + r[:, None] * D + c[None, :], mask=m, other=0.0).to(tl.float32)
    rstd = 1 / tl.sqrt(tl.sum(x * x, 1) / D + EPS)
    out = x * rstd[:, None] * tl.load(weight + c).to(tl.float32)[None, :]
    z = projected + (r // HV)[:, None] * p_stride + z_column + (r % HV)[:, None] * D + c[None, :]
    z = tl.load(z, mask=m, other=0.0).to(tl.float32)
    out = out * z * tl.sigmoid(z)
    tl.store(y + r[:, None] * D + c[None, :], out.to(y.dtype.element_ty), mask=m)


def gated_norm(
    geometry: Geometry,
    o: torch.Tensor,
    projected: torch.Tensor,
    norm_weight: torch.Tensor,
    launch,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """y [T, Hv V] in projected's dtype (into ``out`` when given, contiguous) from o
    [T, Hv, V] (contiguous) and z in projected."""
    rows, hv = projected.shape[0], geometry.v_heads
    y = projected.new_empty(rows, hv * V) if out is None else out
    if rows:
        launch(
            _gated_norm_kernel,
            (triton.cdiv(rows * hv, _BR), 1, 1),
            o, projected, projected.stride(0), geometry.conv_dim, norm_weight, y, rows * hv,
            HV=hv, D=V, EPS=EPS, BR=_BR, num_warps=4,
        )  # fmt: skip
    return y


__all__ = ["gated_norm"]
