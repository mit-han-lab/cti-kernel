# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Chunk-zero WY intermediates, in FP64 on the inputs' CPU or GPU device.

This is the mathematical reference, not a BF16/MMA emulator. Names Q_h/K_h/A_h
identify the corresponding dump slots; reference values are *not* rounded there.
Expected small differences include the precision boundaries in out-digest.md.
State pools use [slot, Hv, V, K]; the algebra below uses H [K, Vs].
"""

from __future__ import annotations

import torch

from ._gdn_reference import CONV, EPS, F64, K, V, _silu, _softplus


def chunk0_intermediates(
    projected,
    conv_weight,
    A_log,
    dt_bias,
    cu_seqlens,
    slots,
    has_initial_state,
    state,
    history,
    *,
    k_heads: int,
    sequence: int,
    value_head: int,
    vsplit: int = 1,
    part: int = 0,
):
    """Return Q_h, K_h, g, beta, L, A_h, S, W, U, r, o, H1 for one CTA.

    Token tiles have 64 rows, with zero Q/K/V/beta and carried g on padding.
    H1 is [Vs, K], U/r/o are [64, Vs], and o includes 1/sqrt(K).
    Only chunk zero is evaluated even if the sequence is longer than 64.
    This diagnostic may synchronize metadata; it is not a capture-safe op.
    """
    hv = A_log.numel()
    if vsplit not in (1, 2, 4) or not 0 <= part < vsplit:
        raise ValueError("vsplit must be 1/2/4 and part must select one slice")
    if hv % k_heads or not 0 <= value_head < hv or not 0 <= sequence < slots.numel():
        raise ValueError("invalid head geometry or sequence/value-head index")
    dc = (2 * k_heads + hv) * K
    if projected.shape[1] != dc + hv * V + 2 * hv or conv_weight.shape != (dc, CONV):
        raise ValueError("expected q|k|v|z|b|a projection and channel-major [Dc, 4] taps")
    begin, end = (int(cu_seqlens[sequence + i]) for i in (0, 1))
    length = min(64, end - begin)
    if length <= 0:
        raise ValueError("an empty sequence has no chunk zero")
    slot = int(slots[sequence])
    initialized = slot >= 0 and bool(has_initial_state[sequence])
    vs = V // vsplit
    vbase = part * vs
    kh = value_head // (hv // k_heads)
    device = projected.device

    def zeros(*shape):
        return torch.zeros(*shape, dtype=F64, device=device)

    def convolve(channel, columns):
        # cw[channel, 0] multiplies x[t-3], cw[channel, 3] multiplies x[t].
        halo = (
            history[slot, :, channel : channel + columns].to(F64)
            if initialized
            else zeros(CONV - 1, columns)
        )
        rows = projected[begin : begin + length, channel : channel + columns].to(F64)
        window = torch.cat((halo, rows))
        taps = conv_weight[channel : channel + columns].to(F64)
        result = zeros(64, columns)
        result[:length] = _silu(sum(window[t : t + length] * taps[:, t] for t in range(CONV)))
        return result

    q = convolve(kh * K, K)
    k = convolve((k_heads + kh) * K, K)
    v = convolve(2 * k_heads * K + value_head * V + vbase, vs)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + EPS)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + EPS)
    gate_base = dc + hv * V
    b = projected[begin : begin + length, gate_base + value_head].to(F64)
    a = projected[begin : begin + length, gate_base + hv + value_head].to(F64)
    delta, beta = zeros(64), zeros(64)
    # The product caps each log-decay at -512; natural-log prefix, not log2.
    delta[:length] = (
        -A_log[value_head].to(F64).exp() * _softplus(a + dt_bias[value_head].to(F64))
    ).clamp_min(-512)
    beta[:length] = b.sigmoid()
    g = delta.cumsum(0)
    causal = torch.ones(64, 64, dtype=torch.bool, device=device).tril()
    # Mask BEFORE exp: the unused upper triangle can overflow for fast decay.
    decay = torch.where(causal, g[:, None] - g[None, :], 0.0).exp()
    lower = torch.where(causal.tril(-1), (k @ k.T) * beta[:, None] * decay, 0.0)
    a_tile = torch.where(causal, (q @ k.T) * decay, 0.0)
    eye = torch.eye(64, dtype=F64, device=device)
    s = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
    w = s @ (k * (beta * g.exp())[:, None])
    b_tile = beta[:, None] * v
    u = s @ b_tile
    h0 = state[slot, value_head, vbase : vbase + vs].to(F64).T if initialized else zeros(K, vs)
    r = u - w @ h0
    o = (g.exp()[:, None] * (q @ h0) + a_tile @ r) * K**-0.5
    h1 = g[-1].exp() * h0 + k.T @ (r * (g[-1] - g).exp()[:, None])
    return {
        "Q_h": q,
        "K_h": k,
        "g": g,
        "beta": beta,
        "L": lower,
        "A_h": a_tile,
        "S": s,
        "W": w,
        "B": b_tile,
        "U": u,
        "r": r,
        "o": o,
        "H1": h1.T,
    }
