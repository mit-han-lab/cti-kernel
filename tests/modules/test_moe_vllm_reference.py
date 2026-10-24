# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The vLLM MoE inference reference's FP32 FMA is one rounding of the exact a * b + c."""

import struct
from fractions import Fraction

import torch

from . import _moe_reference as ref


def _round_f32(q: Fraction) -> float:
    near = torch.tensor([float(q)], dtype=torch.float64).float()
    best = None
    for cand in (near, torch.nextafter(near, near + 1), torch.nextafter(near, near - 1)):
        v = cand.item()
        d = abs(Fraction(v) - q)
        even = struct.unpack("I", struct.pack("f", v))[0] % 2 == 0
        if best is None or d < best[0] or (d == best[0] and even):
            best = (d, v)
    return best[1]


def test_fma32_rounds_once():
    torch.manual_seed(0)
    n = 400
    a, b = torch.randn(n), torch.randn(n)
    c = torch.randn(n) * torch.exp2(torch.randint(-30, 30, (n,)).float())
    c[:100] = -(a[:100].double() * b[:100].double()).float()  # cancellation
    got = ref.fma32(a, b, c)
    for j in range(n):
        want = _round_f32(Fraction(a[j].item()) * Fraction(b[j].item()) + Fraction(c[j].item()))
        assert got[j].item() == want, j


def test_combine_skips_invalid_routes_in_order():
    torch.manual_seed(1)
    y = torch.randn(3, 4, 8).bfloat16()
    w = torch.rand(3, 4)
    valid = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    acc = torch.zeros(3, 8)
    for j in range(4):
        acc = torch.where(
            valid[:, j : j + 1], ref.fma32(w[:, j : j + 1].expand(3, 8), y[:, j].float(), acc), acc
        )
    assert torch.equal(ref.combine_reference(y, w, valid), acc.bfloat16())
    assert torch.equal(ref.combine_reference(y, w, valid)[2], torch.zeros(8, dtype=torch.bfloat16))
