# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import torch
import torch.nn.functional as F

from ctikernels.linear.training import DenseProduct
from ctikernels.mlp.training import gated_mlp_lora

H, W, R, T, SCALING = 6, 5, 2, 7, 0.75


def _case(rows: int = T, dtype: torch.dtype = torch.float64, bias: bool = True):
    torch.manual_seed(0)

    def leaf(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype).requires_grad_()

    x = leaf(rows, H)
    w_gu, w_d = torch.randn(2 * W, H, dtype=dtype), torch.randn(H, W, dtype=dtype)
    b_gu, b_d = (
        (torch.randn(2 * W, dtype=dtype), torch.randn(H, dtype=dtype)) if bias else (None, None)
    )
    gate_up = (leaf(R, H), leaf(W, R), leaf(R, H), leaf(W, R))
    down = (leaf(R, W), leaf(H, R))
    return x, (w_gu, b_gu, w_d, b_d), gate_up, down


def _op(x, weights, gate_up, down):
    w_gu, b_gu, w_d, b_d = weights
    return gated_mlp_lora(
        x, DenseProduct(w_gu, b_gu), DenseProduct(w_d, b_d), gate_up, down, scaling=SCALING
    )


def _three_nodes(x, weights, gate_up, down):
    w_gu, b_gu, w_d, b_d = weights
    a_g, b_g, a_u, b_u = gate_up
    merged = F.linear(x, w_gu, b_gu) + SCALING * torch.cat(
        (x @ a_g.t() @ b_g.t(), x @ a_u.t() @ b_u.t()), -1
    )
    gate, up = merged.chunk(2, -1)
    h = F.silu(gate) * up
    return F.linear(h, w_d, b_d) + SCALING * (h @ down[0].t() @ down[1].t())


def test_fp64_gradcheck() -> None:
    x, weights, gate_up, down = _case()
    assert torch.autograd.gradcheck(
        lambda *t: _op(t[0], weights, t[1:5], t[5:]), (x, *gate_up, *down)
    )


def test_matches_three_nodes_and_is_deterministic() -> None:
    def run(fn):
        x, weights, gate_up, down = _case()
        y = fn(x.reshape(1, T, H), weights, gate_up, down)
        y.backward(torch.linspace(-1, 1, y.numel(), dtype=y.dtype).view_as(y))
        return [y.detach(), x.grad, *(t.grad for t in (*gate_up, *down))]

    ours = run(_op)
    for actual, expected in zip(ours, run(_three_nodes), strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, again in zip(ours, run(_op), strict=True):
        assert torch.equal(actual, again)


def test_saves_x_gate_up_and_low_rank_inputs_only() -> None:
    x, weights, gate_up, down = _case(dtype=torch.float32)
    factors = {t.data_ptr() for t in (*gate_up, *down)}
    saved: list[torch.Tensor] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _op(x, weights, gate_up, down)
    activations = sum(t.nbytes for t in saved if t.data_ptr() not in factors)
    assert activations == T * (H + 2 * W + 3 * R) * 4


def test_grad_mode_independent() -> None:
    x, weights, gate_up, down = _case()
    with torch.no_grad():
        detached = _op(x, weights, gate_up, down)
    assert torch.equal(detached, _op(x, weights, gate_up, down))


def test_zero_rows() -> None:
    x, weights, gate_up, down = _case(rows=0)
    y = _op(x, weights, gate_up, down)
    assert y.shape == (0, H)
    y.sum().backward()
    assert x.grad.shape == (0, H)
    assert all(not t.grad.any() for t in (*gate_up, *down))
