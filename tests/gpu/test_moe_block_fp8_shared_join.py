# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""BlockFP8MoE's gated shared-expert join, bit for bit the engine's BF16 torch join
(``F.sigmoid(gate) * shared + routed``, vLLM's Qwen2-MoE order) on every provider path, at
Qwen3.5-35B-A3B-FP8 widths (E 256, H 2048, I 512, top 8)."""

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE

from ..modules._moe_reference import shared_join_reference
from .test_moe_block_fp8_cute import random_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, H, INTER, K = 256, 2048, 512, 8
TOKENS = (1, 2, 8, 16, 64, 512, 4096)


@pytest.fixture(scope="module")
def weights():
    pytest.importorskip("triton")
    return random_weights(E, H, INTER)


def _inputs(t, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, H, device="cuda", generator=gen).bfloat16()
    w, ids = torch.randn(t, E, device="cuda", generator=gen).softmax(-1).topk(K, -1)
    shared = torch.randn(t, H, device="cuda", generator=gen).bfloat16()
    gate = torch.randn(t, 1, device="cuda", generator=gen).mul_(4)
    extremes = torch.tensor([0.0, -0.0, 90.0, -90.0, 1e4, -1e4], device="cuda")
    gate[: min(t, len(extremes)), 0] = extremes[: min(t, len(extremes))]
    weights = (w / w.sum(-1, keepdim=True)).contiguous()
    return x, weights, ids.int().contiguous(), shared, gate.bfloat16()


def _bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _op(weights, provider):
    if provider == "cute-fp8" and torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("cute-fp8 needs sm120/sm121")
    return BlockFP8MoE(*weights, top_k=K, provider=provider)


@pytest.mark.parametrize("provider", ["triton-fp8", "cute-fp8"])
@torch.inference_mode()
def test_shared_join_is_the_engine_join(weights, provider):
    op = _op(weights, provider)
    for t in TOKENS:
        x, w, ids, shared, gate = _inputs(t, seed=t)
        routed = op(x, w, ids)
        out = op(x, w, ids, shared=shared, shared_gate=gate)
        assert _bits(out, shared_join_reference(routed, shared, gate)), (provider, t)
        assert _bits(op(x, w, ids), routed)  # None/None unchanged


@pytest.mark.parametrize("delta", [False, True])
@torch.inference_mode()
def test_staged_down_hook_shared_join(weights, delta):
    op = _op(weights, "triton-fp8")
    if not op._available:
        pytest.skip("down_hook needs sm120/sm121 and the CuTe DSL")
    for t in (1, 16, 512):
        x, w, ids, shared, gate = _inputs(t, seed=100 + t)
        extra = torch.randn(t * K, H, device="cuda").mul_(0.01).bfloat16() if delta else None
        hook = lambda act, extra=extra: extra
        routed = op.forward_staged(x, w, ids, down_hook=hook)
        out = op.forward_staged(x, w, ids, down_hook=hook, shared=shared, shared_gate=gate)
        assert _bits(out, shared_join_reference(routed, shared, gate)), (delta, t)


@pytest.mark.parametrize("provider", ["triton-fp8", "cute-fp8"])
@pytest.mark.parametrize("t", [1, 16])
@torch.inference_mode()
def test_shared_join_graph_replay(weights, provider, t):
    op = _op(weights, provider)
    op.prepare([t])
    x, w, ids, shared, gate = _inputs(t, seed=200 + t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(x, w, ids, shared=shared, shared_gate=gate)
    for seed in (300, 301):
        x2, w2, ids2, shared2, gate2 = _inputs(t, seed=seed + t)
        for dst, src in ((x, x2), (w, w2), (ids, ids2), (shared, shared2), (gate, gate2)):
            dst.copy_(src)
        graph.replay()
        expected = shared_join_reference(op(x, w, ids), shared, gate)
        assert _bits(out, expected), (provider, t, seed)


def test_shared_join_contract(weights):
    op = _op(weights, "triton-fp8")
    x, w, ids, shared, gate = _inputs(2, seed=7)
    with pytest.raises(ValueError, match="both shared and shared_gate"):
        op(x, w, ids, shared=shared)
    with pytest.raises(ValueError, match="shared must be"):
        op(x, w, ids, shared=shared.float(), shared_gate=gate)
    with pytest.raises(ValueError, match="shared must be"):
        op(x, w, ids, shared=shared, shared_gate=gate.view(1, 2))
