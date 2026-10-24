# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Algorithm reference: fla-core 0.5.2's chunk_fwd_kernel_o and layer_norm_gated_fwd_kernel (the
# arithmetic they define); no kernel code is copied.

"""GDN prefill output with the gated RMSNorm: one program per (64-token chunk, value head),
all chunks in parallel.

    o = scale (2^gamma q h + P v'),  P = lower(q k^T 2^(gamma_i - gamma_j))   (h, v' from the
                                                                           state kernel)
    y = o / sqrt(mean(o^2) + eps) * w * z * sigmoid(z)

o stays in FP32 registers (rounded once, as y in BF16); z is read in place from projected."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._contract import EPS, Geometry, K, V
from ._intra import CHUNK, chunk_of

_BN = 128  # sequences per scan step
_WARPS = 8
_STAGES = 1


@triton.jit(do_not_specialize=["count"])
def _output_kernel(
    q, k, hbuf, vnew, gamma, projected, p_stride, weight, y, cu_seqlens, count, scale,
    HK: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
    BN: tl.constexpr, Z: tl.constexpr, EPS: tl.constexpr,
):  # fmt: skip
    entry, hv = tl.program_id(0).to(tl.int64), tl.program_id(1)
    seq, chunk = chunk_of(cu_seqlens, count, entry, CHUNK, BN)
    if seq >= count:
        return
    t = tl.load(cu_seqlens + seq).to(tl.int64) + chunk * CHUNK + tl.arange(0, CHUNK)
    inside = t < tl.load(cu_seqlens + seq + 1).to(tl.int64)
    m = inside[:, None]
    hk = hv // (HV // HK)
    i = tl.arange(0, CHUNK)
    c = tl.arange(0, D)
    qq = tl.load(q + (t * HK + hk)[:, None] * D + c[None, :], mask=m, other=0.0)
    h = tl.load(hbuf + ((entry * HV + hv) * D + c[:, None]) * D + c[None, :])
    rows = (t * HV + hv)[:, None]
    vv = tl.load(vnew + rows * D + c[None, :], mask=m, other=0.0)
    gam = tl.load(gamma + t * HV + hv, mask=inside, other=0.0)
    kT = tl.load(k + (t * HK + hk)[None, :] * D + c[:, None], mask=inside[None, :], other=0.0)
    P = tl.dot(qq, kT) * tl.exp2(gam[:, None] - gam[None, :])
    P = tl.where(i[:, None] >= i[None, :], P, 0.0).to(qq.dtype)
    o = tl.dot(qq, h) * (tl.exp2(gam) * scale)[:, None] + tl.dot(P, vv) * scale
    rstd = 1 / tl.sqrt(tl.sum(o * o, 1) / D + EPS)
    out = o * rstd[:, None] * tl.load(weight + c).to(tl.float32)[None, :]
    z = tl.load(projected + t[:, None] * p_stride + Z + hv * D + c[None, :], mask=m, other=0.0)
    z = z.to(tl.float32)
    out = out * z * tl.sigmoid(z)
    tl.store(y + t[:, None] * (HV * D) + hv * D + c[None, :], out.to(y.dtype.element_ty), mask=m)


def output(
    geometry: Geometry,
    q: torch.Tensor,
    k: torch.Tensor,
    hbuf: torch.Tensor,
    vnew: torch.Tensor,
    gamma: torch.Tensor,
    projected: torch.Tensor,
    norm_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """y [T, Hv V] in projected's dtype (into ``out`` when given, contiguous)."""
    rows, hv = projected.shape[0], geometry.v_heads
    y = projected.new_empty(rows, hv * V) if out is None else out
    if hbuf.shape[0]:
        launch(
            _output_kernel,
            (hbuf.shape[0], hv, 1),
            q, k, hbuf, vnew, gamma, projected, projected.stride(0), norm_weight, y,
            cu_seqlens, cu_seqlens.shape[0] - 1, K**-0.5,
            HK=geometry.k_heads, HV=hv, D=K, CHUNK=CHUNK, BN=_BN, Z=geometry.conv_dim, EPS=EPS,
            num_warps=_WARPS, num_stages=_STAGES,
        )  # fmt: skip
    return y


__all__ = ["output"]
