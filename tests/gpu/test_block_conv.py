# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DFlash2 block convolution on the device: bitwise against the composed torch form."""

import pytest
import torch

from ctikernels.speculative import block_conv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def composed(x, deltas, base, q, dtype=None):
    """The engine's composed form: FP32 coefficients, FP32 products, tap-order FP32 sums,
    rounded to ``dtype`` (default ``x.dtype``)."""
    taps, segments = deltas.shape[1], deltas.shape[2]
    kernel = base.float().view(1, 1, taps, segments, -1)
    kernel = kernel + deltas.float().reshape(-1, q, taps, segments, 1)
    rows = x.float().view(-1, q, segments, kernel.shape[-1])
    out = kernel[:, :, 0] * rows
    for tap in range(1, taps):
        out[:, tap:] += kernel[:, tap:, tap] * rows[:, :-tap]
    return out.view_as(x).to(dtype or x.dtype)


CASES = [
    # (blocks, Q, taps, C, S, x dtype, delta dtype)
    (4, 8, 2, 5120, 320, torch.bfloat16, torch.bfloat16),
    (3, 7, 2, 1024, 64, torch.bfloat16, torch.bfloat16),
    (2, 2, 2, 768, 48, torch.float32, torch.bfloat16),
    (2, 16, 3, 640, 40, torch.bfloat16, torch.float32),
    (5, 1, 2, 256, 16, torch.bfloat16, torch.bfloat16),
    (1, 8, 1, 520, 13, torch.float16, torch.bfloat16),
]


def _case(blocks, q, taps, c, s, dtype, ddtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = blocks * q
    x = (torch.randn(n, c, generator=g) * 3).to(dtype)
    x[0, :8] = -0.0  # signed zeros survive the first-row skip
    if n > 1:
        x[1] = 0.0
    both = (torch.randn(n, 2, taps, s, generator=g) * 0.5).to(ddtype)
    base = (torch.randn(2, taps, c, generator=g) * 0.5).to(torch.bfloat16)
    return x.to(device), both.to(device), base.to(device)


def same_bits(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("index", range(len(CASES)))
def test_bitwise_vs_composed(index):
    blocks, q, *_ = CASES[index]
    x, both, base = _case(*CASES[index], "cuda", index)
    for side in (0, 1):
        deltas = both[:, side]
        expected = composed(x, deltas, base[side], q)
        assert same_bits(block_conv(x, deltas, base[side].contiguous(), q), expected)


def test_graph_replay_and_out():
    # DFlash2 draft geometry: B = 4 blocks of Q = 8, hidden 5120, 320 groups of 16.
    x, both, base = _case(4, 8, 2, 5120, 320, torch.bfloat16, torch.bfloat16, "cuda", 7)
    deltas, weight = both[:, 1], base[1].contiguous()
    out = torch.empty_like(x)
    block_conv(x, deltas, weight, 8, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        block_conv(x, deltas, weight, 8, out=out)
    fresh = _case(4, 8, 2, 5120, 320, torch.bfloat16, torch.bfloat16, "cuda", 8)
    x.copy_(fresh[0])
    both.copy_(fresh[1])
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(out, composed(x, both[:, 1], weight, 8))
