# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear.training import DenseProduct, lora_projection_training
from ctikernels.mlp.training import gated_mlp_lora

pytestmark = pytest.mark.gpu

# The LoRA projection training bounds (tests/gpu/test_lora_projection.py): per-tensor RMS error
# within atol + rtol * reference RMS, and a collection relative L2 bound. BF16 rounds gate | up
# and h before the down GEMM, so a BF16 tensor may instead stay within 1.25x the error of the
# three-node path the engine ran before (two LoRA projections around SwiGLU).
_BOUNDS = {torch.float32: (2e-6, 2e-6, 2e-5), torch.bfloat16: (1e-4, 4e-3, 4e-3)}
SCALING = 2.0


def _case(rows: int, hidden: int, width: int, rank: int, dtype: torch.dtype, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def tensor(*shape, scale):
        return torch.randn(*shape, device="cuda", generator=generator) * scale

    x = tensor(rows, hidden, scale=1.0)
    weights = (
        tensor(2 * width, hidden, scale=hidden**-0.5),
        tensor(hidden, width, scale=width**-0.5),
    )
    factors = tuple(
        tensor(*shape, scale=0.05)
        for shape in (
            (rank, hidden),
            (width, rank),
            (rank, hidden),
            (width, rank),
            (rank, width),
            (hidden, rank),
        )
    )
    grad = tensor(rows, hidden, scale=1.0)
    cast = lambda t: t.to(dtype)
    return cast(x), tuple(map(cast, weights)), tuple(map(cast, factors)), cast(grad)


def _run(x, weights, factors, grad, *, reference: bool = False, three_nodes: bool = False):
    x = x.detach().clone().requires_grad_()
    factors = tuple(f.detach().clone().requires_grad_() for f in factors)
    if three_nodes:
        from ctikernels.activation import training as activation  # imports Triton

        merged = lora_projection_training(x, DenseProduct(weights[0]), factors[:4], scaling=SCALING)
        h = activation.silu_and_mul(merged)
        y = lora_projection_training(h, DenseProduct(weights[1]), factors[4:], scaling=SCALING)
    elif reference:
        a_g, b_g, a_u, b_u, a_d, b_d = factors
        merged = F.linear(x, weights[0]) + SCALING * torch.cat(
            (x @ a_g.t() @ b_g.t(), x @ a_u.t() @ b_u.t()), -1
        )
        gate, up = merged.chunk(2, -1)
        h = F.silu(gate) * up
        y = F.linear(h, weights[1]) + SCALING * (h @ a_d.t() @ b_d.t())
    else:
        products = DenseProduct(weights[0]), DenseProduct(weights[1])
        y = gated_mlp_lora(x, *products, factors[:4], factors[4:], scaling=SCALING)
    y.backward(grad)
    return [y.detach(), x.grad, *(f.grad for f in factors)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
# The last shape keeps the backward's FP32 partials in grad_h's spare rows (5 chunks of 224 rows).
@pytest.mark.parametrize(
    "shape",
    [(1, 64, 96, 8), (257, 256, 384, 16), (1024, 1024, 3072, 64), (1024, 512, 12288, 16)],
)
def test_matches_fp32_autograd(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    x, weights, factors, grad = _case(*shape, dtype)
    ours = _run(x, weights, factors, grad)
    expected = _run(
        x.float(),
        [w.float() for w in weights],
        [f.float() for f in factors],
        grad.float(),
        reference=True,
    )
    three = _run(x, weights, factors, grad, three_nodes=True)
    atol, rtol, collection = _BOUNDS[dtype]
    slack = 1.25 if dtype == torch.bfloat16 else 0.0
    names = ["y", "x", "A_gate", "B_gate", "A_up", "B_up", "A_down", "B_down"]
    sums = [0.0, 0.0, 0.0]  # ours, three-node, reference
    for name, actual, baseline, oracle in zip(names, ours, three, expected, strict=True):
        assert actual.dtype == dtype
        reference = oracle.double()
        ours_sq = (actual.double() - reference).square().sum().item()
        three_sq = (baseline.double() - reference).square().sum().item()
        magnitude = reference.square().sum().item()
        rms, three_rms, reference_rms = (
            math.sqrt(value / reference.numel()) for value in (ours_sq, three_sq, magnitude)
        )
        bound = max(atol + rtol * reference_rms, slack * three_rms)
        assert rms <= bound, (name, rms, three_rms, reference_rms)
        sums = [a + b for a, b in zip(sums, (ours_sq, three_sq, magnitude), strict=True)]
    ours_l2, three_l2 = math.sqrt(sums[0] / sums[2]), math.sqrt(sums[1] / sums[2])
    assert ours_l2 <= max(collection, slack * three_l2), (ours_l2, three_l2)


def test_bitwise_deterministic_and_inputs_preserved() -> None:
    x, weights, factors, grad = _case(4096, 1024, 3072, 16, torch.bfloat16)
    before = [t.clone() for t in (x, *weights, *factors, grad)]
    first, second = _run(x, weights, factors, grad), _run(x, weights, factors, grad)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(
        torch.equal(a, b) for a, b in zip(before, (x, *weights, *factors, grad), strict=True)
    )

    x.requires_grad_()
    y = gated_mlp_lora(x, *map(DenseProduct, weights), factors[:4], factors[4:], scaling=SCALING)
    y.backward(grad, retain_graph=True)
    with pytest.raises(RuntimeError, match="runs once"):
        y.backward(grad)


def test_saved_bytes_are_x_gate_up_and_low_rank() -> None:
    rows, hidden, width, rank = 512, 256, 768, 16
    x, weights, factors, _ = _case(rows, hidden, width, rank, torch.bfloat16)
    x.requires_grad_()
    factors = tuple(f.requires_grad_() for f in factors)
    excluded = {t.data_ptr() for t in (*weights, *factors)}
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        gated_mlp_lora(x, *map(DenseProduct, weights), factors[:4], factors[4:], scaling=SCALING)
    activations = sum(t.nbytes for t in saved if t.data_ptr() not in excluded)
    assert activations == rows * (hidden + 2 * width + 3 * rank) * 2


class _StraightThroughRounding(DenseProduct):
    """A quantized-base stand-in: forward rounds the activation, grad_input passes it through."""

    def _forward(self, x: torch.Tensor, out: torch.Tensor, bias: torch.Tensor | None) -> None:
        torch.mm(x.to(torch.float8_e4m3fn).to(x.dtype), self.weight.t(), out=out)


def test_frozen_product_adapter_and_zero_rows() -> None:
    x, weights, factors, grad = _case(64, 128, 192, 8, torch.bfloat16)
    products = tuple(map(_StraightThroughRounding, weights))
    x.requires_grad_()
    y = gated_mlp_lora(x, *products, factors[:4], factors[4:], scaling=SCALING)
    y.backward(grad)
    assert torch.isfinite(x.grad).all()

    empty = x[:0].detach().requires_grad_()
    leaves = tuple(f.detach().requires_grad_() for f in factors)
    y = gated_mlp_lora(empty, *map(DenseProduct, weights), leaves[:4], leaves[4:], scaling=SCALING)
    y.sum().backward()
    assert y.shape == (0, 128) and empty.grad.shape == (0, 128)
    assert all(not f.grad.any() for f in leaves)
