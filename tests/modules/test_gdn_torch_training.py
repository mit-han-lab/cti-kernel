# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""gdn.torch.training, the CPU reference of the Triton training provider: its explicit chunked
backward against autograd and FP64 gradcheck, and its layer against the token recurrence."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.torch import training

from ..gpu import _gdn_reference as reference

F64 = torch.float64
LENGTHS = [1, 63, 64, 65, 130, 7]  # around the 64-token chunks


def _cu(lengths) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _rule_inputs(k_heads, v_heads, lengths, seed, decay=0.5):
    generator = torch.Generator().manual_seed(seed)
    tokens = sum(lengths)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=F64)

    return (
        F.normalize(rand(tokens, k_heads, 128), dim=-1),
        F.normalize(rand(tokens, k_heads, 128), dim=-1),
        rand(tokens, v_heads, 128),
        -decay * torch.rand(tokens, v_heads, generator=generator, dtype=F64),
        torch.rand(tokens, v_heads, generator=generator, dtype=F64),
        0.1 * rand(len(lengths), v_heads, 128, 128),
    )


@pytest.mark.parametrize(("k_heads", "v_heads"), [(1, 1), (2, 6)])
@pytest.mark.parametrize("initial", [False, True])
def test_the_explicit_backward_is_autograd_of_the_forward(k_heads, v_heads, initial):
    q, k, v, g, beta, h0 = _rule_inputs(k_heads, v_heads, LENGTHS, seed=k_heads)
    h0 = h0 if initial else None
    cu = _cu(LENGTHS)
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta) + ((h0,) if initial else ())]
    o, final = training.chunk_forward(*leaves[:5], 128**-0.5, cu, leaves[5] if initial else None)
    do, dfinal = torch.randn_like(o), torch.randn_like(final)
    expected = torch.autograd.grad((o * do).sum() + (final * dfinal).sum(), leaves)
    found = training.chunk_backward(q, k, v, g, beta, 128**-0.5, cu, h0, do, dfinal)
    assert (found[5] is None) == (not initial)
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), found, expected, strict=False):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11, msg=name)


def test_gradcheck_in_fp64():
    """Directional FP64 gradcheck of the autograd function (fast mode: random projections)."""
    lengths = [3, 66]
    q, k, v, g, beta, h0 = (x.requires_grad_() for x in _rule_inputs(1, 2, lengths, seed=3))

    def rule(*args):
        return training.ChunkRule.apply(*args, 128**-0.5, _cu(lengths))

    assert torch.autograd.gradcheck(rule, (q, k, v, g, beta, h0), fast_mode=True)


@pytest.mark.parametrize(("k_heads", "v_heads"), [(1, 2), (2, 2)])
def test_the_layer_is_the_token_recurrence(k_heads, v_heads):
    """y and d projected of ``layer`` equal the FP64 token recurrence's (tests.gpu)."""
    geometry = Geometry(k_heads, v_heads)
    generator = torch.Generator().manual_seed(5)
    weights = reference.Weights(
        conv_weight=(0.5 * torch.randn(geometry.conv_dim, 4, generator=generator)).bfloat16(),
        A_log=torch.rand(v_heads, generator=generator) * 2.8 - 0.7,
        dt_bias=torch.rand(v_heads, generator=generator) * 4 - 2,
        norm_weight=1 + 0.1 * torch.randn(128, generator=generator),
    )
    tokens = sum(LENGTHS)
    projected = torch.randn(tokens, geometry.proj_width, generator=generator).bfloat16()
    cu = _cu(LENGTHS)
    ours, theirs = (projected.to(F64).requires_grad_() for _ in range(2))
    y = training.layer(geometry, ours, *weights.args(), cu)
    expected = reference.training(theirs, weights, k_heads, cu)
    torch.testing.assert_close(y, expected, rtol=1e-11, atol=1e-11)
    dy = torch.randn(y.shape, generator=generator, dtype=F64)
    (grad,) = torch.autograd.grad(y, ours, dy)
    (expected_grad,) = torch.autograd.grad(expected, theirs, dy)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-11, atol=1e-11)
