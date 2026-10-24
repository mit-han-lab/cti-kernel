# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Algorithm reference: the chunked WY form of the gated delta rule and its backward (the
# flash-linear-attention papers; fla-core 0.5.2's chunk_gated_delta_rule_{fwd,bwd}).

"""Torch reference of GDN training: the chunked gated delta rule with its explicit backward,
the algorithm ``gdn.triton.training`` runs, in the inputs' dtype (FP64 for gradcheck) on any
device. Host loops over sequences and chunks; small shapes only.

Per 64-token chunk and value head, with gamma the chunk-local cumulative sum of the log decay
g, the state S [K, V] entering the chunk and q, k, v, beta its rows:

    L = strict_lower(beta_i k_i.k_j e^(gamma_i - gamma_j)),  T = (I + L)^-1
    w = T (beta e^gamma k),  u = T (beta v),  v' = u - w S
    o = scale (e^gamma q S + lower(q k^T e^(gamma_i - gamma_j)) v')
    S <- e^gamma_last S + k^T (e^(gamma_last - gamma) v')

``chunk_backward`` returns the gradients of q, k, v, g, beta and the initial state from those
of o and the final state, chunk by chunk in reverse (the pieces fla-core's backward kernels
compute). ``ChunkRule`` is the autograd function over both; ``layer`` the whole training
operation (conv + SiLU, q/k L2 norm, gates, rule, gated RMSNorm) on it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .._contract import CONV, EPS, Geometry, K, V

CHUNK = 64


def _bounds(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    cu = cu_seqlens.tolist()
    return list(zip(cu[:-1], cu[1:], strict=True))


def _per_value_head(x: torch.Tensor, v_heads: int) -> torch.Tensor:
    """[C, Hk, D] -> [Hv, C, D]: value head h reads key head h // (Hv // Hk)."""
    return x.repeat_interleave(v_heads // x.shape[1], 1).transpose(0, 1)


def _chunk_terms(q, k, v, g, beta):
    """Per chunk (heads first): gamma, the decay ratio e^(gamma_i - gamma_j) masked lower,
    T = (I + L)^-1, w and u."""
    gamma = g.t().cumsum(-1)  # [H, C]
    ratio = torch.exp(gamma[:, :, None] - gamma[:, None, :])
    c = gamma.shape[1]
    strict = torch.ones(c, c, dtype=torch.bool, device=g.device).tril(-1)
    b = beta.t()[:, :, None]
    L = torch.where(strict, b * (k @ k.transpose(1, 2)) * ratio, 0.0)
    eye = torch.eye(c, dtype=q.dtype, device=q.device)
    T = torch.linalg.solve_triangular(eye + L, eye.expand_as(L), upper=False)
    w = T @ (b * torch.exp(gamma)[:, :, None] * k)
    u = T @ (b * v)
    return gamma, ratio, T, w, u


def _chunks(start: int, end: int):
    for first in range(start, end, CHUNK):
        yield first, min(first + CHUNK, end)


def chunk_forward(q, k, v, g, beta, scale, cu_seqlens, initial_state=None):
    """o [T, Hv, V] and the final states [N, Hv, V, K] (the pool layout, S transposed) of q, k
    [T, Hk, K], v [T, Hv, V], g and beta [T, Hv]; ``initial_state`` like the final states."""
    hv = v.shape[1]
    o = torch.empty_like(v)
    finals = []
    for n, (start, end) in enumerate(_bounds(cu_seqlens)):
        S = (
            initial_state[n].transpose(1, 2).to(q.dtype)
            if initial_state is not None
            else q.new_zeros(hv, K, V)
        )
        for a, b in _chunks(start, end):
            qc, kc = _per_value_head(q[a:b], hv), _per_value_head(k[a:b], hv)
            vc = v[a:b].transpose(0, 1)
            gamma, ratio, _, w, u = _chunk_terms(qc, kc, vc, g[a:b], beta[a:b])
            vn = u - w @ S
            P = (qc @ kc.transpose(1, 2) * ratio).tril()
            out = torch.exp(gamma)[:, :, None] * (qc @ S) + P @ vn
            o[a:b] = (scale * out).transpose(0, 1)
            last = gamma[:, -1:]
            S = torch.exp(last)[:, :, None] * S + kc.transpose(1, 2) @ (
                torch.exp(last - gamma)[:, :, None] * vn
            )
        finals.append(S.transpose(1, 2))
    return o, torch.stack(finals)


def chunk_backward(q, k, v, g, beta, scale, cu_seqlens, initial_state, do, dfinal):
    """(dq, dk, dv, dg, dbeta, dh0) of ``chunk_forward`` from do [T, Hv, V] and the final
    states' gradient (None: zero). dh0 is None without an initial state."""
    hk, hv = q.shape[1], v.shape[1]
    dq, dk = (torch.zeros(q.shape[0], hv, K, dtype=q.dtype, device=q.device) for _ in range(2))
    dv, dg, dbeta = torch.zeros_like(v), torch.zeros_like(g), torch.zeros_like(beta)
    dh0 = []
    for n, (start, end) in enumerate(_bounds(cu_seqlens)):
        S = (
            initial_state[n].transpose(1, 2).to(q.dtype)
            if initial_state is not None
            else q.new_zeros(hv, K, V)
        )
        states = []  # the state entering each chunk
        for a, b in _chunks(start, end):
            qc, kc = _per_value_head(q[a:b], hv), _per_value_head(k[a:b], hv)
            gamma, _, _, w, u = _chunk_terms(qc, kc, v[a:b].transpose(0, 1), g[a:b], beta[a:b])
            states.append(S)
            vn = u - w @ S
            last = gamma[:, -1:]
            S = torch.exp(last)[:, :, None] * S + kc.transpose(1, 2) @ (
                torch.exp(last - gamma)[:, :, None] * vn
            )
        dS = q.new_zeros(hv, K, V) if dfinal is None else dfinal[n].transpose(1, 2).to(q.dtype)
        for (a, b), S in reversed(list(zip(_chunks(start, end), states, strict=True))):
            qc, kc = _per_value_head(q[a:b], hv), _per_value_head(k[a:b], hv)
            vc, bc = v[a:b].transpose(0, 1), beta[a:b].t()[:, :, None]
            gamma, ratio, T, w, u = _chunk_terms(qc, kc, vc, g[a:b], beta[a:b])
            vn = u - w @ S
            doc = do[a:b].transpose(0, 1)
            eg = torch.exp(gamma)[:, :, None]
            last = gamma[:, -1:]
            tail = torch.exp(last - gamma)[:, :, None]
            # dv': the chunk's own outputs, then the state it leaves.
            P = (scale * qc @ kc.transpose(1, 2) * ratio).tril()
            dvn = P.transpose(1, 2) @ doc + tail * (kc @ dS)
            # dq, dk, dw and the decay gradient by gamma (fla-core's dqkwg).
            dqc = scale * eg * (doc @ S.transpose(1, 2))
            dkc = tail * (vn @ dS.transpose(1, 2))
            dlast = torch.exp(last) * (S * dS).sum((1, 2))[:, None] + (dkc * kc).sum(-1).sum(
                -1, keepdim=True
            )
            ds = scale * (doc @ vn.transpose(1, 2) * ratio).tril()
            dqc = dqc + ds @ kc
            dkc = dkc + ds.transpose(1, 2) @ qc
            dgam = (dqc * qc).sum(-1) - (dkc * kc).sum(-1)
            dgam[:, -1:] += dlast
            dw = -dvn @ S.transpose(1, 2)
            # The state entering the chunk.
            dS = (
                torch.exp(last)[:, :, None] * dS
                + scale * (eg * qc).transpose(1, 2) @ doc
                - w.transpose(1, 2) @ dvn
            )
            # The WY representation (fla-core's prepare_wy_repr_bwd): du = dv'.
            kbg = bc * eg * kc
            dkbg = T.transpose(1, 2) @ dw
            dkc = dkc + dkbg * bc * eg
            db = (dkbg * kc * eg).sum(-1)
            dgam = dgam + (dkbg * kbg).sum(-1)
            dvb = T.transpose(1, 2) @ dvn
            dvc = dvb * bc
            db = db + (dvb * vc).sum(-1)
            c = gamma.shape[1]
            strict = torch.ones(c, c, dtype=torch.bool, device=q.device).tril(-1)
            dA = torch.where(strict, dw @ kbg.transpose(1, 2) + dvn @ (bc * vc).transpose(1, 2), 0)
            dA = T.transpose(1, 2) @ dA @ T.transpose(1, 2)
            dA = torch.where(strict, -dA * ratio, 0)
            dkb = dA @ kc
            db = db + (dkb * kc).sum(-1)
            dkc = dkc + dkb * bc + dA.transpose(1, 2) @ (bc * kc)
            Ab = (kc @ kc.transpose(1, 2)) * bc
            dgam = dgam + (dA * Ab).sum(-1) - (dA * Ab).sum(-2)
            # gamma is a cumulative sum of g: its gradient sums in reverse.
            dg[a:b] = dgam.flip(-1).cumsum(-1).flip(-1).t()
            dbeta[a:b] = db.t()
            dq[a:b], dk[a:b], dv[a:b] = (
                dqc.transpose(0, 1),
                dkc.transpose(0, 1),
                dvc.transpose(0, 1),
            )
        dh0.append(dS.transpose(1, 2))
    group = (q.shape[0], hk, hv // hk, K)
    dq, dk = dq.view(group).sum(2), dk.view(group).sum(2)
    return dq, dk, dv, dg, dbeta, torch.stack(dh0) if initial_state is not None else None


class ChunkRule(torch.autograd.Function):
    """``chunk_forward`` under autograd with ``chunk_backward`` as its backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, scale, cu_seqlens):
        o, final = chunk_forward(q, k, v, g, beta, scale, cu_seqlens, initial_state)
        ctx.save_for_backward(q, k, v, g, beta, initial_state, cu_seqlens)
        ctx.scale = scale
        return o, final

    @staticmethod
    def backward(ctx, do, dfinal):
        q, k, v, g, beta, initial_state, cu_seqlens = ctx.saved_tensors
        grads = chunk_backward(q, k, v, g, beta, ctx.scale, cu_seqlens, initial_state, do, dfinal)
        return (*grads, None, None)


def layer(
    geometry: Geometry,
    projected: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """y [T, Hv V] of the training contract (``gdn.training``) in ``projected``'s dtype,
    differentiable in ``projected``; the weights are constants."""
    hk, hv, dc = geometry.k_heads, geometry.v_heads, geometry.conv_dim
    dtype, rows = projected.dtype, projected.shape[0]
    taps = conv_weight.to(dtype)
    pieces = []
    for start, end in _bounds(cu_seqlens):
        x = F.pad(projected[start:end, :dc], (0, 0, CONV - 1, 0))
        pieces.append(sum(taps[:, j] * x[j : j + end - start] for j in range(CONV)))
    mixed = F.silu(torch.cat(pieces))
    q, k, v = mixed.split([geometry.key_width, geometry.key_width, geometry.value_width], 1)
    q, k = (x.view(rows, hk, K) for x in (q, k))
    q, k = (x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + EPS) for x in (q, k))
    z = projected[:, dc : dc + geometry.value_width].view(rows, hv, V)
    b, a = projected[:, dc + geometry.value_width :].split([hv, hv], 1)
    g = -torch.exp(A_log.to(dtype)) * F.softplus(a + dt_bias.to(dtype))
    o, _ = ChunkRule.apply(
        q, k, v.view(rows, hv, V), g, torch.sigmoid(b), None, K**-0.5, cu_seqlens
    )
    y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * norm_weight.to(dtype) * F.silu(z)
    return y.reshape(rows, hv * V)


__all__ = ["CHUNK", "ChunkRule", "chunk_backward", "chunk_forward", "layer"]
