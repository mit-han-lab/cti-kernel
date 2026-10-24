# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e (fla-core 0.5.2:
# `fla/modules/conv/triton/kernels.py`, `fla/modules/l2norm.py`); see THIRD_PARTY_NOTICES.

"""GDN training, the conv's input gradient with the q/k L2 norms' backward in one launch: one
program per (BT token rows, 128-channel tile = one q, k or v head) writes the rows' gradient of
the pre-conv input x,

    dx_t = sum_j w_j g_s silu'(c_s),  s = t + 3 - j within t's sequence,

where c_s is the pre-activation recomputed as the forward's (FP32, taps oldest first, rounded
to BF16) and g_s the gradient of the conv's output at row s: dv for a v head; for a q or k head
the L2 norm's input gradient r (dy - y (y . dy)) of dy (dq of the key head, or dk plus each of
its value heads' WY terms in head order), y the normalised BF16 rows and r their 1/norms. In
Gluon: each row's term is formed once into shared memory (the program's rows and the HISTORY
after them) and gathered per tap; the row's whole head is in the tile, so the norm's dot
product is local."""

from __future__ import annotations

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .._contract import CONV, HISTORY, Geometry, K


@gluon.jit
def _term(
    x, x_stride, weight, sequence, cu_seqlens, y, y_rstd, dy, dy_extra, dv, s, s_in, d, c, tile,
    L: gl.constexpr, HK: gl.constexpr, HV: gl.constexpr, D: gl.constexpr, CONV: gl.constexpr,
    HISTORY: gl.constexpr,
):  # fmt: skip
    """g silu'(c) at rows s (zero outside ``s_in``) for channels d (c = d within the tile)."""
    seq = gl.load(sequence + s, mask=s_in, other=0)
    position = s - gl.load(cu_seqlens + seq, mask=s_in, other=0).to(gl.int64)
    col = gl.expand_dims(d, 0)
    pre = gl.zeros([s.shape[0], D], gl.float32, layout=L)
    for j in gl.static_range(CONV):
        w = gl.load(weight + d * CONV + j).to(gl.float32)
        inside = gl.expand_dims(s_in & (position >= HISTORY - j), 1)
        value = gl.load(x + gl.expand_dims(s - (HISTORY - j), 1) * x_stride + col, mask=inside,
                        other=0.0)  # fmt: skip
        pre = pre + value.to(gl.float32) * gl.expand_dims(w, 0)
    pre = pre.to(gl.bfloat16).to(gl.float32)  # the forward's BF16 conv output, recomputed
    m = gl.expand_dims(s_in, 1)
    cc = gl.expand_dims(c, 0)
    if tile < 2 * HK:
        head = tile % HK
        rows_of = (gl.expand_dims(s, 1) * HK + head) * D + cc
        g = gl.load(dy + rows_of, mask=m, other=0.0)
        if tile >= HK:
            for v in gl.static_range(HV // HK):
                value_head = head * (HV // HK) + v
                extra = gl.load(dy_extra + (gl.expand_dims(s, 1) * HV + value_head) * D + cc,
                                mask=m, other=0.0)  # fmt: skip
                g += extra.to(gl.float32)
        yy = gl.load(y + rows_of, mask=m, other=0.0).to(gl.float32)
        r = gl.load(y_rstd + s * HK + head, mask=s_in, other=0.0)
        g = (g - yy * gl.expand_dims(gl.sum(g * yy, axis=1), 1)) * gl.expand_dims(r, 1)
    else:
        hv = tile - 2 * HK
        g = gl.load(dv + (gl.expand_dims(s, 1) * HV + hv) * D + cc, mask=m, other=0.0)
        g = g.to(gl.float32)
    sig = 1 / (1 + gl.exp(-pre))
    return gl.where(m, g * sig * (1.0 + pre * (1.0 - sig)), 0.0)


@gluon.jit(do_not_specialize=["rows"])
def _conv_norm_bwd_kernel(
    x, x_stride, weight, sequence, cu_seqlens, q, k, q_rstd, k_rstd, dq, dk, dk_terms, dv,
    out, out_stride, rows,
    HK: gl.constexpr, HV: gl.constexpr, D: gl.constexpr, CONV: gl.constexpr,
    HISTORY: gl.constexpr, BT: gl.constexpr, HALO: gl.constexpr,
):  # fmt: skip
    L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [gl.num_warps(), 1], [1, 0])
    rows_of: gl.constexpr = gl.SliceLayout(1, L)
    tile = gl.program_id(1)
    t0 = gl.program_id(0).to(gl.int64) * BT
    t = t0 + gl.arange(0, BT, layout=rows_of)
    c = gl.arange(0, D, layout=gl.SliceLayout(0, L))
    d = tile * D + c
    if tile < HK:
        y, y_rstd, dy = q, q_rstd, dq
    else:
        y, y_rstd, dy = k, k_rstd, dk
    # Each row's term once into shared memory: rows [t0, t0 + BT) and the HISTORY after them.
    terms = gl.allocate_shared_memory(
        gl.float32, [2 * BT, D], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    for half in gl.static_range(2):
        if half == 0:
            r = t
        else:
            r = t0 + BT + gl.arange(0, HALO, layout=rows_of)
        term = _term(
            x, x_stride, weight, sequence, cu_seqlens, y, y_rstd, dy, dk_terms, dv, r, r < rows,
            d, c, tile, L, HK, HV, D, CONV, HISTORY,
        )  # fmt: skip
        terms.slice(half * BT, r.shape[0], dim=0).store(term)
    gl.barrier()
    t_in = t < rows
    seq = gl.load(sequence + t, mask=t_in, other=-1)
    dx = gl.zeros([BT, D], gl.float32, layout=L)
    local = gl.expand_dims(gl.arange(0, BT, layout=rows_of), 1) + gl.zeros([BT, D], gl.int32,
                                                                           layout=L)  # fmt: skip
    for j in gl.static_range(CONV - 1, -1, -1):  # the tap of s itself first
        s = t + HISTORY - j  # w_j multiplies the term at s
        seq_s = gl.load(sequence + s, mask=s < rows, other=-2)
        same = gl.expand_dims((s < rows) & (seq_s == seq), 1)
        term = terms.gather(local + (HISTORY - j), 0)
        w = gl.load(weight + d * CONV + j).to(gl.float32)
        dx = gl.fma(gl.expand_dims(w, 0), gl.where(same, term, 0.0), dx)
    store = out + gl.expand_dims(t, 1) * out_stride + gl.expand_dims(d, 0)
    gl.store(store, dx.to(out.dtype.element_ty), mask=gl.expand_dims(t_in, 1))


def conv_norm_backward(geometry: Geometry, x, conv_weight, sequence, cu_seqlens, q, k, q_rstd,
                       k_rstd, dq, dk, dk_terms, dv, out, config):  # fmt: skip
    """The conv's input gradient into ``out`` [T, conv_dim] (strided rows) from the conv input
    x [T, conv_dim] (strided rows), the normalised q, k [T, Hk, K] with their 1/norms, dq, dk
    [T, Hk, K] FP32 (summed over each key head's value heads), dk_terms [T, Hv, K] and dv
    [T, Hv, V]."""
    rows = x.shape[0]
    _conv_norm_bwd_kernel[(triton.cdiv(rows, config["BT"]), geometry.conv_dim // K)](
        x, x.stride(0), conv_weight, sequence, cu_seqlens, q, k, q_rstd, k_rstd, dq, dk,
        dk_terms, dv, out, out.stride(0), rows,
        HK=geometry.k_heads, HV=geometry.v_heads, D=K, CONV=CONV, HISTORY=HISTORY,
        HALO=triton.next_power_of_2(HISTORY), **config,
    )  # fmt: skip


__all__ = ["conv_norm_backward"]
