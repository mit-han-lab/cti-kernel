# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""topk_by_id: bitwise against packed (value, ID) keys through torch.topk, the composed form."""

import pytest
import torch

from ctikernels.sampling import topk_by_id

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
MASK = 0xFFFFFFFF


def composed(logits, k):
    bits = (logits.float() + 0.0).view(torch.int32)
    ids = torch.arange(logits.shape[-1], device=logits.device)
    words = ((bits ^ ((bits >> 31) & 0x7FFFFFFF)).long() << 32) | (MASK - ids)
    words = words.topk(k, sorted=False).values
    chosen = MASK - (words & MASK)
    order = chosen.argsort(-1)
    key = (words.gather(-1, order) >> 32).int()
    return (key ^ ((key >> 31) & 0x7FFFFFFF)).view(torch.float32), chosen.gather(-1, order)


def rows(kind, r, v, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":
        x = torch.randn(r, v, generator=g, device="cuda") * 4
    elif kind == "ties":
        x = torch.randint(-3, 2, (r, v), generator=g, device="cuda").float()
        x[:, ::977] = 1.0
    else:  # signed zeros at the boundary, NaN and infinities
        x = torch.full((r, v), float("-inf"), device="cuda")
        x[:, 5:11] = -0.0
        x[:, 100:111] = 0.0
        x[1, 50:60] = 2.0
        x[2, 7] = float("nan")
        x[2, v - 1] = float("inf")
        x[3] = -0.0
        x[3, 3:8] = 0.0
    return x.to(dtype).contiguous()


def same_bits(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("width", [248320, 124160, 4099])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["random", "ties", "zeros"])
def test_bitwise_vs_composed(width, dtype, kind):
    x = rows(kind, 28, width, dtype, 11)
    for k in (16, 5):
        values, ids = topk_by_id(x, k)
        expected_values, expected_ids = composed(x, k)
        assert torch.equal(ids, expected_ids)
        assert same_bits(values, expected_values)


def test_graph_replay():
    x = rows("random", 28, 248320, torch.bfloat16, 3)
    topk_by_id(x, 16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, ids = topk_by_id(x, 16)
    x.copy_(rows("zeros", 28, 248320, torch.bfloat16, 4))
    graph.replay()
    torch.cuda.synchronize()
    expected_values, expected_ids = composed(x, 16)
    assert torch.equal(ids, expected_ids) and same_bits(values, expected_values)
