# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FP64 reference of the ctikernels.gdn contract, token by token, independent of any provider.

Written from the contract in ``ctikernels/gdn/inference.py``: every intermediate in FP64,
the exact softplus, no BF16 rounding anywhere. Loops run on the host over sequences and
tokens; meant for small test shapes only. :func:`training` is differentiable: autograd through
it is the reference of the training backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

K = V = 128
CONV = 4
EPS = 1e-6
F64 = torch.float64


@dataclass(frozen=True)
class Weights:
    conv_weight: torch.Tensor  # [Dc, 4] BF16
    A_log: torch.Tensor  # [Hv] FP32
    dt_bias: torch.Tensor  # [Hv] FP32
    norm_weight: torch.Tensor  # [V] FP32

    def args(self) -> tuple[torch.Tensor, ...]:
        return (self.conv_weight, self.A_log, self.dt_bias, self.norm_weight)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(x, torch.zeros_like(x))


class _Sequence:
    """One sequence's recurrence: the FP64 state S [Hv, K, V] and the pre-conv history."""

    def __init__(self, weights: Weights, k_heads: int, state: torch.Tensor, history: torch.Tensor):
        self.taps = weights.conv_weight.to(F64)  # [Dc, 4]
        self.A = torch.exp(weights.A_log.to(F64))
        self.dt_bias = weights.dt_bias.to(F64)
        self.norm_weight = weights.norm_weight.to(F64)
        self.k_heads, self.v_heads = k_heads, weights.A_log.shape[0]
        self.S = state.to(F64).transpose(-1, -2).clone()  # pool [Hv, V, K] -> S [Hv, K, V]
        self.history = history.clone()  # [3, Dc], the pool's dtype (bit copies)

    def step(self, row: torch.Tensor) -> torch.Tensor:
        """One token of ``row`` [P] (BF16); returns y [Hv * V] in FP64."""
        hk, hv = self.k_heads, self.v_heads
        dc = self.taps.shape[0]
        x = row[:dc]
        window = torch.cat((self.history, x[None])).to(F64)  # x_{t-3} .. x_t
        mixed = _silu((window.t() * self.taps).sum(-1))
        q, k, v = mixed.split([hk * K, hk * K, hv * V])
        q = q.view(hk, K)
        k = k.view(hk, K)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + EPS) * K**-0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
        ratio = hv // hk
        q, k = q.repeat_interleave(ratio, 0), k.repeat_interleave(ratio, 0)  # per value head
        v = v.view(hv, V)
        z = row[dc : dc + hv * V].to(F64).view(hv, V)
        b, a = row[dc + hv * V :].to(F64).split([hv, hv])
        decay = torch.exp(-self.A * _softplus(a + self.dt_bias))[:, None, None]
        beta = torch.sigmoid(b)[:, None]
        S = decay * self.S
        delta = beta * (v - torch.einsum("hkv,hk->hv", S, k))
        self.S = S + k[:, :, None] * delta[:, None, :]
        o = torch.einsum("hkv,hk->hv", self.S, q)
        y = self.norm_weight * o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * _silu(z)
        self.history = torch.cat((self.history[1:], x[None]))
        return y.reshape(-1)

    def pool_state(self) -> torch.Tensor:
        return self.S.transpose(-1, -2)  # [Hv, V, K]


def decode(projected, weights: Weights, k_heads: int, slot_indices, state, conv_state):
    """(y FP64 [B, Hv V], state FP64 copy of the pool with updated rows, conv_state copy)."""
    hv = weights.A_log.shape[0]
    y = torch.zeros(projected.shape[0], hv * V, dtype=F64, device=projected.device)
    new_state, new_conv = state.to(F64).clone(), conv_state.clone()
    for row, slot in enumerate(slot_indices.tolist()):
        if slot < 0:
            continue
        sequence = _Sequence(weights, k_heads, state[slot], conv_state[slot])
        y[row] = sequence.step(projected[row])
        new_state[slot] = sequence.pool_state()
        new_conv[slot] = sequence.history
    return y, new_state, new_conv


def prefill(
    projected,
    weights: Weights,
    k_heads: int,
    cu_seqlens,
    slot_indices,
    has_initial_state,
    state,
    conv_state,
):
    """(y FP64 [T, Hv V], state FP64 copy of the pool with updated rows, conv_state copy)."""
    hv = weights.A_log.shape[0]
    dc = weights.conv_weight.shape[0]
    device = projected.device
    y = torch.zeros(projected.shape[0], hv * V, dtype=F64, device=device)
    new_state, new_conv = state.to(F64).clone(), conv_state.clone()
    bounds = cu_seqlens.tolist()
    for n, (slot, initial) in enumerate(
        zip(slot_indices.tolist(), has_initial_state.tolist(), strict=True)
    ):
        if slot >= 0 and initial:
            start_state, history = state[slot], conv_state[slot]
        else:
            start_state = torch.zeros(hv, V, K, dtype=torch.float32, device=device)
            history = torch.zeros(CONV - 1, dc, dtype=conv_state.dtype, device=device)
        sequence = _Sequence(weights, k_heads, start_state, history)
        for t in range(bounds[n], bounds[n + 1]):
            y[t] = sequence.step(projected[t])
        if slot >= 0:
            new_state[slot] = sequence.pool_state()
            new_conv[slot] = sequence.history
    return y, new_state, new_conv


def training(projected: torch.Tensor, weights: Weights, k_heads: int, cu_seqlens) -> torch.Tensor:
    """y (FP64 [T, Hv V]) of the training contract, each sequence from a zero state and history,
    differentiable in ``projected`` (FP64 [T, P]; the weights stay constants)."""
    hv, dc = weights.A_log.shape[0], weights.conv_weight.shape[0]
    taps = weights.conv_weight.to(F64)
    A, dt_bias = torch.exp(weights.A_log.to(F64)), weights.dt_bias.to(F64)
    norm_weight = weights.norm_weight.to(F64)
    ratio = hv // k_heads
    bounds = cu_seqlens.tolist()
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        rows, length = projected[start:end], end - start
        window = torch.cat((rows.new_zeros(CONV - 1, dc), rows[:, :dc]))  # x_{t-3} .. x_t
        mixed = _silu(sum(taps[:, j] * window[j : j + length] for j in range(CONV)))
        q, k, v = mixed.split([k_heads * K, k_heads * K, hv * V], -1)
        q, k = q.reshape(length, k_heads, K), k.reshape(length, k_heads, K)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + EPS) * K**-0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
        q, k = q.repeat_interleave(ratio, 1), k.repeat_interleave(ratio, 1)  # per value head
        v = v.reshape(length, hv, V)
        z = rows[:, dc : dc + hv * V].reshape(length, hv, V)
        b, a = rows[:, dc + hv * V :].split([hv, hv], -1)
        decay = torch.exp(-A * _softplus(a + dt_bias))[:, :, None, None]
        beta = torch.sigmoid(b)[:, :, None]
        S = rows.new_zeros(hv, K, V)
        o = []
        for t in range(length):
            S = decay[t] * S
            delta = beta[t] * (v[t] - torch.einsum("hkv,hk->hv", S, k[t]))
            S = S + k[t][:, :, None] * delta[:, None, :]
            o.append(torch.einsum("hkv,hk->hv", S, q[t]))
        o = torch.stack(o)
        y = norm_weight * o * torch.rsqrt((o * o).mean(-1, keepdim=True) + EPS) * _silu(z)
        outputs.append(y.reshape(length, hv * V))
    return torch.cat(outputs)


__all__ = ["Weights", "decode", "prefill", "training"]


@torch.no_grad()
def prefill_chunked(
    projected,
    weights,
    k_heads,
    cu_seqlens,
    slot_indices,
    has_initial_state,
    state,
    conv_state,
    *,
    value_heads=(0, -1),
    boundaries=(),
    chunk_size=64,
):
    """Independent FP64 triangular-solve oracle for selected value heads.

    Returns y [T,H,128], selected pool states [S,H,V,K], raw history, and a
    dictionary of (sequence, local token boundary) -> [H,V,K] states. Boundaries
    must be chunk-aligned (or sequence ends). Precompute 32 chunks at a time to
    bound memory; only the state recurrence is serial. No BF16 intermediates.
    """
    hv, dc = weights.A_log.numel(), weights.conv_weight.shape[0]
    device = projected.device
    heads = torch.tensor([h % hv for h in value_heads], device=device)
    kh = heads // (hv // k_heads)
    columns = torch.cat(
        [
            (kh[:, None] * K + torch.arange(K, device=device)).flatten(),
            (k_heads * K + kh[:, None] * K + torch.arange(K, device=device)).flatten(),
            (2 * k_heads * K + heads[:, None] * V + torch.arange(V, device=device)).flatten(),
        ]
    )
    hcount = len(value_heads)
    taps = weights.conv_weight[columns].double()
    A = weights.A_log[heads].double().exp()
    bias = weights.dt_bias[heads].double()
    norm = weights.norm_weight.double()
    y = torch.empty(projected.shape[0], hcount, V, dtype=F64, device=device)
    states = state[:, heads].double().clone()
    history = conv_state.clone()
    snapshots = {}
    bounds = cu_seqlens.tolist()
    for n, (slot, initial) in enumerate(
        zip(slot_indices.tolist(), has_initial_state.tolist(), strict=True)
    ):
        begin, end = bounds[n : n + 2]
        length = end - begin
        S = (
            state[slot, heads].double().transpose(-1, -2).clone()
            if slot >= 0 and initial
            else torch.zeros(hcount, K, V, dtype=F64, device=device)
        )
        hist = conv_state[slot] if slot >= 0 and initial else projected.new_zeros(3, dc)
        snapshots[n, 0] = S.transpose(-1, -2).clone()
        for batch_start in range(0, length, 32 * chunk_size):
            count = min(32 * chunk_size, length - batch_start)
            start = begin + batch_start
            raw = projected[start : start + count, columns]
            prefix = hist[:, columns] if batch_start == 0 else projected[start - 3 : start, columns]
            window = torch.cat((prefix, raw)).double()
            mixed = _silu(sum(window[j : j + count] * taps[:, j] for j in range(4)))
            q, k, v = [
                part.reshape(count, hcount, K).transpose(0, 1)
                for part in mixed.split(hcount * K, dim=-1)
            ]
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + EPS) * K**-0.5
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + EPS)
            rows = projected[start : start + count]
            beta = rows[:, dc + hv * V + heads].double().sigmoid().t()
            g = -A[:, None] * _softplus(
                rows[:, dc + hv * V + hv + heads].double().t() + bias[:, None]
            )
            # Pad with identity recurrence steps, never fake observations.
            pad = (-count) % chunk_size
            if pad:
                q, k, v = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (q, k, v)]
                beta, g = [torch.nn.functional.pad(t, (0, pad)) for t in (beta, g)]
            chunks = (count + pad) // chunk_size
            q, k, v = [t.reshape(hcount, chunks, chunk_size, K).transpose(0, 1) for t in (q, k, v)]
            beta, g = [t.reshape(hcount, chunks, chunk_size).transpose(0, 1) for t in (beta, g)]
            cumulative = g.cumsum(-1)
            decay = cumulative.exp()
            # Clamp the unused upper triangle before exp: heavy decay can overflow it.
            relative = (
                (cumulative[..., :, None] - cumulative[..., None, :]).clamp(max=0).exp().tril()
            )
            kk = k @ k.transpose(-1, -2)
            system = (beta[..., :, None] * kk * relative).tril(-1)
            system += torch.eye(chunk_size, dtype=F64, device=device)
            rhs = torch.cat((beta[..., None] * v, (beta * decay)[..., None] * k), dim=-1)
            solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
            U, W = solved.split(V, dim=-1)
            attention = (q @ k.transpose(-1, -2)) * relative
            end_k = k * (cumulative[..., -1:] - cumulative).exp()[..., None]
            for c in range(chunks):
                delta = U[c] - W[c] @ S
                o = decay[c, ..., None] * (q[c] @ S) + attention[c] @ delta
                S = decay[c, :, -1, None, None] * S + end_k[c].transpose(-1, -2) @ delta
                local = batch_start + c * chunk_size
                valid = min(chunk_size, length - local)
                z = (
                    projected[begin + local : begin + local + valid, dc : dc + hv * V]
                    .reshape(valid, hv, V)[:, heads]
                    .double()
                    .transpose(0, 1)
                )
                o = o[:, :valid]
                result = norm * o * torch.rsqrt(o.square().mean(-1, keepdim=True) + EPS) * _silu(z)
                y[begin + local : begin + local + valid] = result.transpose(0, 1)
                boundary = local + valid
                if boundary in boundaries or boundary == length:
                    snapshots[n, boundary] = S.transpose(-1, -2).clone()
        if slot >= 0:
            states[slot] = S.transpose(-1, -2)
            history[slot] = (
                projected[end - 3 : end, :dc]
                if length >= 3
                else torch.cat((hist, projected[begin:end, :dc]))[-3:]
            )
    return y, states, history, snapshots
