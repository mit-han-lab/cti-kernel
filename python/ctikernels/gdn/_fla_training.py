# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Songlin Yang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Yu Zhang
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FLA 9c8e42e762fce087c27b673af4922795d9edb85e
# (`fla/modules/conv/triton/ops.py`, `fla/modules/conv/triton/kernels.py`,
# `fla/ops/gated_delta_rule/gate.py`); see THIRD_PARTY_NOTICES.

"""Conv kernels for fla-core 0.5.2 training: the conv forward written straight into contiguous
q, k and v (Triton), and the conv's input gradient with the pre-activation recomputed in the
kernel. Gluon computes each row's term once in shared memory only on devices named
``NVIDIA GeForce RTX 5090``, ``NVIDIA RTX PRO 6000 Blackwell Server Edition`` and
``NVIDIA GB10``; other sm12x boards and all other devices take the Triton path. All keep
fla-core's FP32 arithmetic (bitwise its results). Scratch: int32 [T] sequence indices.

``gate_gradient`` computes only the gate input gradient da, without weight gradients.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ._contract import CONV, HISTORY

# Token tile, channel tile and warps of the Gluon conv backward by device key (sm12x only).
_GLUON_TILES = {
    "NVIDIA GeForce RTX 5090": (32, 64, 4),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": (32, 64, 4),
    "NVIDIA GB10": (32, 128, 4),
}


def _device_key(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


@triton.jit
def _pre(x, x_stride, weight, t, position, valid, d, CONV: tl.constexpr, HISTORY: tl.constexpr):
    """fla-core's FP32 conv pre-activation of rows t (oldest tap first, product then sum)."""
    pre = tl.zeros((t.shape[0], d.shape[0]), dtype=tl.float32)
    for j in tl.static_range(CONV):
        back = HISTORY - j
        w = tl.load(weight + d * CONV + j).to(tl.float32)
        inside = (valid & (position >= back))[:, None]
        value = tl.load(x + (t - back)[:, None] * x_stride + d[None, :], mask=inside, other=0.0)
        pre = pre + value.to(tl.float32) * w[None, :]
    return pre


@triton.jit(do_not_specialize=["rows"])
def _conv_fwd_kernel(
    x, x_stride, weight, sequence, cu_seqlens, q, k, v, rows, KW, VW,
    CONV: tl.constexpr, HISTORY: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    first = tl.program_id(1) * BD  # BD divides KW: a tile lies in one of q, k, v
    d = first + tl.arange(0, BD)
    t_in = t < rows
    seq = tl.load(sequence + t, mask=t_in, other=0)
    position = t - tl.load(cu_seqlens + seq, mask=t_in, other=0).to(tl.int64)
    pre = _pre(x, x_stride, weight, t, position, t_in, d, CONV, HISTORY)
    y = pre * tl.sigmoid(pre)
    if first < KW:
        out, col, width = q, d, KW
    elif first < 2 * KW:
        out, col, width = k, d - KW, KW
    else:
        out, col, width = v, d - 2 * KW, VW
    tl.store(
        out + t[:, None] * width + col[None, :], y.to(out.dtype.element_ty), mask=t_in[:, None]
    )


def conv_forward(
    x: torch.Tensor, weight: torch.Tensor, sequence: torch.Tensor, cu_seqlens: torch.Tensor,
    key_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """fla-core's SiLU(causal conv(x)) of x [T, Dc] (strided rows, BF16) as contiguous q, k
    [T, KW] and v [T, Dc - 2 KW]; ``sequence`` is each row's sequence (int32 [T])."""
    rows, d = x.shape
    bt, bd, warps = 16, 128, 4
    assert key_width % bd == 0 and d % bd == 0
    q, k = (x.new_empty(rows, key_width) for _ in range(2))
    v = x.new_empty(rows, d - 2 * key_width)
    if rows:
        _conv_fwd_kernel[(triton.cdiv(rows, bt), d // bd)](
            x, x.stride(0), weight, sequence, cu_seqlens, q, k, v, rows, key_width, v.shape[1],
            CONV=CONV, HISTORY=HISTORY, BT=bt, BD=bd, num_warps=warps,
        )  # fmt: skip
    return q, k, v


@triton.jit(do_not_specialize=["rows"])
def _conv_bwd_kernel(
    x, x_stride, weight, sequence, cu_seqlens, dq, dk, dv, dq_stride, dk_stride, dv_stride,
    out, out_stride, rows, KW,
    CONV: tl.constexpr, HISTORY: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    t_in = t < rows
    # The gradient of the conv's output, by column: dq, dk (KW wide each), dv (the rest).
    is_q, is_k, is_v = d < KW, (d >= KW) & (d < 2 * KW), d >= 2 * KW
    seq = tl.load(sequence + t, mask=t_in, other=-1)
    dx = tl.zeros((BT, BD), dtype=tl.float32)
    for j in tl.static_range(CONV - 1, -1, -1):  # the tap of s itself first
        s = t + HISTORY - j  # w_j multiplies dc_s
        seq_s = tl.load(sequence + s, mask=s < rows, other=-2)
        same = (s < rows) & (seq_s == seq)
        m = same[:, None]
        # The BF16 pre-activation fla-core's conv backward recomputes (its forward, rounded).
        position = s - tl.load(cu_seqlens + seq_s, mask=same, other=0).to(tl.int64)
        c = _pre(x, x_stride, weight, s, position, same, d, CONV, HISTORY)
        c = c.to(tl.bfloat16).to(tl.float32)
        g = tl.load(dq + s[:, None] * dq_stride + d[None, :], mask=m & is_q[None, :], other=0.0)
        g = g.to(tl.float32) + tl.load(
            dk + s[:, None] * dk_stride + (d - KW)[None, :], mask=m & is_k[None, :], other=0.0
        ).to(tl.float32)
        g += tl.load(
            dv + s[:, None] * dv_stride + (d - 2 * KW)[None, :], mask=m & is_v[None, :], other=0.0
        ).to(tl.float32)
        sig = tl.sigmoid(c)  # fla-core's SiLU backward expression (its swish_bwd)
        term = tl.where(m, g * sig * (1.0 + c * (1.0 - sig)), 0.0)
        w = tl.load(weight + d * CONV + j).to(tl.float32)
        dx = tl.fma(w[None, :], term, dx)
    tl.store(
        out + t[:, None] * out_stride + d[None, :], dx.to(out.dtype.element_ty), mask=t_in[:, None]
    )


def conv_backward(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    sequence: torch.Tensor,
    cu_seqlens: torch.Tensor,
    d_mixed: Sequence[torch.Tensor],
    out: torch.Tensor,
) -> None:
    """The conv's input gradient with SiLU's slope into ``out`` [T, Dc] (BF16) from the conv
    input x [T, Dc] and the gradient of its SiLU output as (dq, dk) [T, KW] and dv
    [T, Dc - 2 KW]. Every [T, *] operand may have strided rows (dense columns)."""
    rows, d = x.shape
    if not rows:
        return
    dq, dk, dv = d_mixed
    args = (x, x.stride(0), conv_weight, sequence, cu_seqlens, dq, dk, dv,
            dq.stride(0), dk.stride(0), dv.stride(0), out, out.stride(0), rows, dq.shape[1])  # fmt: skip
    gluon_tiles = _GLUON_TILES.get(_device_key(x.device))
    if gluon_tiles is not None and dq.shape[1] % gluon_tiles[1] == 0 and d % gluon_tiles[1] == 0:
        bt, bd, warps = gluon_tiles
        _conv_bwd_gluon[(triton.cdiv(rows, bt), d // bd)](
            *args,
            CONV=CONV,
            HISTORY=HISTORY,
            BT=bt,
            BD=bd,
            HALO=triton.next_power_of_2(HISTORY),
            num_warps=warps,
        )
        return
    bt, bd, warps = 16, 128, 4
    assert dq.shape[1] % bd == 0 and d % bd == 0
    _conv_bwd_kernel[(triton.cdiv(rows, bt), d // bd)](
        *args, CONV=CONV, HISTORY=HISTORY, BT=bt, BD=bd, num_warps=warps
    )


@gluon.jit
def _term(
    x, x_stride, weight, sequence, cu_seqlens, dq, dk, dv, dq_stride, dk_stride, dv_stride,
    s, s_in, d, KW, L: gl.constexpr, CONV: gl.constexpr, HISTORY: gl.constexpr,
):  # fmt: skip
    """g silu'(c) at rows s, with c fla-core's BF16-rounded FP32 pre-activation and the
    Triton kernel's operation order (so its bits)."""
    seq = gl.load(sequence + s, mask=s_in, other=0)
    position = s - gl.load(cu_seqlens + seq, mask=s_in, other=0).to(gl.int64)
    col = gl.expand_dims(d, 0)
    pre = gl.zeros([s.shape[0], d.shape[0]], gl.float32, layout=L)
    for j in gl.static_range(CONV):
        w = gl.load(weight + d * CONV + j).to(gl.float32)
        inside = gl.expand_dims(s_in & (position >= HISTORY - j), 1)
        row = gl.expand_dims(s - (HISTORY - j), 1)
        value = gl.load(x + row * x_stride + col, mask=inside, other=0.0)
        pre = pre + value.to(gl.float32) * gl.expand_dims(w, 0)
    c = pre.to(gl.bfloat16).to(gl.float32)
    m, row = gl.expand_dims(s_in, 1), gl.expand_dims(s, 1)
    g = gl.load(dq + row * dq_stride + col, mask=m & (col < KW), other=0.0).to(gl.float32)
    in_k = m & (col >= KW) & (col < 2 * KW)
    g += gl.load(dk + row * dk_stride + (col - KW), mask=in_k, other=0.0).to(gl.float32)
    in_v = m & (col >= 2 * KW)
    g += gl.load(dv + row * dv_stride + (col - 2 * KW), mask=in_v, other=0.0).to(gl.float32)
    sig = 1 / (1 + gl.exp(-c))  # tl.sigmoid's expression
    return g * sig * (1.0 + c * (1.0 - sig))


@gluon.jit(do_not_specialize=["rows"])
def _conv_bwd_gluon(
    x, x_stride, weight, sequence, cu_seqlens, dq, dk, dv, dq_stride, dk_stride, dv_stride,
    out, out_stride, rows, KW,
    CONV: gl.constexpr, HISTORY: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr,
    HALO: gl.constexpr,
):  # fmt: skip
    L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [gl.num_warps(), 1], [1, 0])
    rows_of: gl.constexpr = gl.SliceLayout(1, L)
    t0 = gl.program_id(0).to(gl.int64) * BT
    t = t0 + gl.arange(0, BT, layout=rows_of)
    d = gl.program_id(1) * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, L))
    # Each row's term once: rows [t0, t0 + BT) and the HISTORY rows after them.
    terms = gl.allocate_shared_memory(
        gl.float32, [2 * BT, BD], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    for half in gl.static_range(2):
        r = t if half == 0 else t0 + BT + gl.arange(0, HALO, layout=rows_of)
        term = _term(
            x, x_stride, weight, sequence, cu_seqlens, dq, dk, dv, dq_stride, dk_stride,
            dv_stride, r, r < rows, d, KW, L, CONV, HISTORY,
        )  # fmt: skip
        terms.slice(half * BT, r.shape[0], dim=0).store(term)
    gl.barrier()
    t_in = t < rows
    seq = gl.load(sequence + t, mask=t_in, other=-1)
    dx = gl.zeros([BT, BD], gl.float32, layout=L)
    local = gl.expand_dims(gl.arange(0, BT, layout=rows_of), 1) + gl.zeros(
        [BT, BD], gl.int32, layout=L
    )
    for j in gl.static_range(CONV - 1, -1, -1):  # the tap of s itself first
        s = t + HISTORY - j  # w_j multiplies dc_s
        seq_s = gl.load(sequence + s, mask=s < rows, other=-2)
        same = gl.expand_dims((s < rows) & (seq_s == seq), 1)
        term = terms.gather(local + (HISTORY - j), 0)
        w = gl.load(weight + d * CONV + j).to(gl.float32)
        dx = gl.fma(gl.expand_dims(w, 0), gl.where(same, term, 0.0), dx)
    store = out + gl.expand_dims(t, 1) * out_stride + gl.expand_dims(d, 0)
    gl.store(store, dx.to(out.dtype.element_ty), mask=gl.expand_dims(t_in, 1))


@triton.jit(do_not_specialize=["rows"])
def _gate_kernel(
    a, a_stride, A_log, dt_bias, dyg, out, out_stride, rows,
    H: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    h = tl.arange(0, BH)
    m = (t < rows)[:, None] & (h < H)[None, :]
    g = tl.load(a + t[:, None] * a_stride + h[None, :], mask=m, other=0.0).to(tl.float32)
    g = g + tl.load(dt_bias + h, mask=h < H, other=0.0).to(tl.float32)[None, :]
    neg_exp_a = -tl.exp(tl.load(A_log + h, mask=h < H, other=0.0).to(tl.float32))
    d = tl.load(dyg + t[:, None] * H + h[None, :], mask=m, other=0.0).to(tl.float32)
    da = neg_exp_a[None, :] * (d * tl.sigmoid(g))  # fla-core gdn_gate_bwd's dg expression
    tl.store(out + t[:, None] * out_stride + h[None, :], da.to(out.dtype.element_ty), mask=m)


def gate_gradient(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    dyg: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """da = dyg d(-exp(A_log) softplus(a + dt_bias))/da into ``out`` [T, Hv] (BF16, strided rows)
    from a [T, Hv] (strided rows) and dyg [T, Hv] (FP32, dense): fla-core's gate backward without
    its A_log and dt_bias gradients."""
    rows, heads = a.shape
    if rows:
        bt = 32
        _gate_kernel[(triton.cdiv(rows, bt),)](
            a, a.stride(0), A_log, dt_bias, dyg, out, out.stride(0), rows,
            H=heads, BT=bt, BH=triton.next_power_of_2(heads),
        )  # fmt: skip


def sequence_index(cu_seqlens: torch.Tensor, tokens: int) -> torch.Tensor:
    """Each token's sequence (int32 [T]), from device ops only."""
    token = torch.arange(tokens, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    return torch.searchsorted(cu_seqlens[1:], token, right=True).to(torch.int32)


__all__ = ["conv_backward", "conv_forward", "gate_gradient", "sequence_index"]
