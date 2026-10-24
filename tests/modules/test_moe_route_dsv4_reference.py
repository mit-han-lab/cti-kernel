# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The DeepSeek-V4 routing reference on the CPU: the bias selects only, weights scale."""

import torch

from ._moe_route_reference import dsv4_hash_route, dsv4_route, scores


def _inputs(t=9, e=16, h=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, h, generator=g).bfloat16(), torch.randn(e, h, generator=g).bfloat16()


def test_bias_selects_but_does_not_weigh():
    x, w = _inputs()
    bias = torch.zeros(16)
    bias[3] = 100.0
    weights, ids = dsv4_route(x, w, 4, bias, scale=1.5)
    assert (ids == 3).any(dim=1).all()
    s = scores(x, w).gather(1, ids.long())
    torch.testing.assert_close(weights, 1.5 * s / s.sum(1, keepdim=True))
    torch.testing.assert_close(weights.sum(1), torch.full((9,), 1.5))


def test_hash_route_reads_the_table_and_masks_padding():
    x, w = _inputs()
    table = torch.randint(16, (100, 6), dtype=torch.int32)
    tokens = torch.randint(100, (9,))
    weights, ids = dsv4_hash_route(x, w, tokens, table, scale=1.5, valid=7)
    assert torch.equal(ids[:7], table[tokens[:7]]) and (ids[7:] == -1).all()
    assert (weights[7:] == 0).all()
    torch.testing.assert_close(weights[:7].sum(1), torch.full((7,), 1.5))
