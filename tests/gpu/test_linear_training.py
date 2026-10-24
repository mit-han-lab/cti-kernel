# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear.block_fp8 import (
    BlockFP8Linear,
    dequantize_block_fp8,
    quantize_per_token_group_fp8,
)
from ctikernels.linear.training import BlockFP8Product, DenseProduct, lora_projection_training
from ctikernels.mlp.training import gated_mlp_lora

pytestmark = pytest.mark.gpu
SCALING = 2.0


def _fp8_weight(n: int, k: int, generator: torch.Generator):
    weight = torch.randn(n, k, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(-(-n // 128), -(-k // 128), device="cuda", generator=generator) * 0.01 + 0.002
    )
    return weight, scale


class _Reference:
    """FP32 contract of a block-FP8 base: 1x128-quantized activation times the decoded weight;
    the gradient straight through the quantizer."""

    def __init__(self, weight, scale):
        self.weight = dequantize_block_fp8(weight, scale, dtype=torch.float32)

    def forward(self, x):
        q, s = quantize_per_token_group_fp8(x.detach().to(torch.bfloat16))
        decoded = q.float() * s.repeat_interleave(128, dim=1)[:, : q.shape[1]]
        x_hat = decoded + (x - x.detach())  # straight-through
        return F.linear(x_hat, self.weight)


def _lora(x, base, factors, reference: bool):
    if not reference:
        return lora_projection_training(x, base, factors, scaling=SCALING)
    deltas = [x @ a.t() @ b.t() for a, b in zip(factors[::2], factors[1::2], strict=True)]
    return base.forward(x) + SCALING * torch.cat(deltas, -1)


def _relative_rms(actual, expected):
    error = (actual.double() - expected.double()).square().mean().sqrt()
    return float(error / expected.double().square().mean().sqrt())


@pytest.mark.parametrize("rows", [1, 300])
@pytest.mark.parametrize("n,k", [(384, 256), (200, 320)])
def test_block_fp8_base_lora_projection(rows: int, n: int, k: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows + n)
    weight, scale = _fp8_weight(n, k, generator)
    x = torch.randn(rows, k, device="cuda", generator=generator)
    factors = [torch.randn(*s, device="cuda", generator=generator) * 0.05 for s in ((8, k), (n, 8))]
    grad = torch.randn(rows, n, device="cuda", generator=generator)

    def run(dtype, reference):
        leaves = [t.to(dtype).requires_grad_() for t in (x, *factors)]
        base = (
            _Reference(weight, scale)
            if reference
            else BlockFP8Product(BlockFP8Linear(weight, scale))
        )
        y = _lora(leaves[0], base, leaves[1:], reference)
        y.backward(grad.to(dtype))
        return [y.detach(), *(t.grad for t in leaves)]

    ours, again = run(torch.bfloat16, False), run(torch.bfloat16, False)
    assert all(torch.equal(a, b) for a, b in zip(ours, again, strict=True))
    for name, actual, expected in zip(
        ("y", "x", "A", "B"), ours, run(torch.float32, True), strict=True
    ):
        assert _relative_rms(actual, expected) <= 8e-3, name


def test_block_fp8_base_gated_mlp_matches_dense_decoded() -> None:
    """FP8 bases through gated_mlp_lora against the same op on the decoded weights with the
    activations rounded as the FP8 forward rounds them: only the quantizer differs."""
    generator = torch.Generator(device="cuda").manual_seed(1)
    rows, hidden, width, rank = 256, 512, 768, 16
    (w_gu, s_gu), (w_d, s_d) = (
        _fp8_weight(2 * width, hidden, generator),
        _fp8_weight(hidden, width, generator),
    )
    x = torch.randn(rows, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    factors = [
        (torch.randn(*s, device="cuda", generator=generator) * 0.05).to(torch.bfloat16)
        for s in (
            (rank, hidden),
            (width, rank),
            (rank, hidden),
            (width, rank),
            (rank, width),
            (hidden, rank),
        )
    ]
    grad = torch.randn(rows, hidden, device="cuda", generator=generator).to(torch.bfloat16)

    def run(products):
        leaves = [t.clone().requires_grad_() for t in (x, *factors)]
        y = gated_mlp_lora(leaves[0], *products, leaves[1:5], leaves[5:], scaling=SCALING)
        y.backward(grad)
        return [y.detach(), *(t.grad for t in leaves)]

    fp8 = run(
        (BlockFP8Product(BlockFP8Linear(w_gu, s_gu)), BlockFP8Product(BlockFP8Linear(w_d, s_d)))
    )
    dense = run(
        (
            DenseProduct(dequantize_block_fp8(w_gu, s_gu)),
            DenseProduct(dequantize_block_fp8(w_d, s_d)),
        )
    )
    for actual, expected in zip(fp8, dense, strict=True):
        assert math.isfinite(_relative_rms(actual, expected))
        assert _relative_rms(actual, expected) <= 0.06  # E4M3 activation quantization (~2^-4 rel)
