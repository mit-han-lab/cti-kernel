# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""MoE inference combine as vLLM 0.31 computes it (fused_experts; block-FP8 MoE is gated in
test_moe_block_fp8_vllm_combine.py).

Each provider's own unweighted BF16 FC2 rows, captured at the reduction, must combine bit for
bit as ``combine_reference`` (FlashInfer's finalize: an FP32 FMA chain in route order, one
rounding) and join the gated shared expert as ``shared_join_reference`` (vLLM's BF16 ops). The
whole BF16 output is also held to ``vllm_infer_reference`` within GEMM-order tolerance.
"""

import pytest
import torch

from ctikernels import moe_impl
from ctikernels.moe import Experts

from ..modules._moe_reference import combine_reference, shared_join_reference, vllm_infer_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Qwen3.5-35B-A3B (with its gated shared expert) and Qwen3-Coder-30B-A3B routed experts.
SHAPES = {"35b": (256, 2048, 512, 8, True), "coder30b": (128, 2048, 768, 8, False)}
TOKENS = (1, 4, 64, 300, 2048, 8192)


def _routing(t, e, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(t, e, device="cuda", generator=gen)
    weights, ids = logits.softmax(-1).topk(k, -1)
    ids = ids.int()
    if t > 2:
        ids[1, -1] = -1  # graph padding: a skipped route keeps its slot
    return (weights / weights.sum(-1, keepdim=True)).contiguous(), ids.contiguous()


@pytest.fixture(scope="module", params=sorted(SHAPES))
def bf16_model(request):
    e, h, i, k, has_shared = SHAPES[request.param]
    gen = torch.Generator(device="cuda").manual_seed(5)
    w13 = torch.randn(e, 2 * i, h, device="cuda", generator=gen).div_(h**0.5).bfloat16()
    w2 = torch.randn(e, h, i, device="cuda", generator=gen).div_(i**0.5).bfloat16()
    return request.param, w13, w2, k, has_shared


def _capture(monkeypatch):
    rows = []
    real = moe_impl.moe_sum_reduce_triton

    def reduce(input, output, *args, **kwargs):
        rows.append(input.clone())
        return real(input, output, *args, **kwargs)

    monkeypatch.setattr(moe_impl, "moe_sum_reduce_triton", reduce)
    return rows


@pytest.mark.parametrize("t", TOKENS)
@torch.inference_mode()
def test_fused_experts_combine_is_vllm(monkeypatch, bf16_model, t):
    name, w13, w2, k, has_shared = bf16_model
    e, h = w13.shape[0], w13.shape[2]
    x = torch.randn(t, h, device="cuda").bfloat16()
    weights, ids = _routing(t, e, k, seed=t)
    shared = gate = None
    if has_shared:
        shared = torch.randn(t, h, device="cuda").bfloat16()
        gate = torch.randn(t, 1, device="cuda").mul_(4).bfloat16()
    rows = _capture(monkeypatch)
    out = Experts()(x, w13, w2, weights, ids, shared=shared, shared_gate=gate)
    assert len(rows) == 1 and rows[0].dtype == torch.bfloat16
    expected = combine_reference(rows[0].view(t, k, h), weights, ids >= 0)
    if has_shared:
        expected = shared_join_reference(expected, shared, gate)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16)), name
    if not has_shared and t <= 300:
        reference = vllm_infer_reference(x, w13, w2, weights, ids)
        scale = reference.float().abs().amax().clamp_min(1e-3)
        error = (out.float() - reference.float()).abs().amax() / scale
        assert error < 2e-2, (name, t, float(error))


@pytest.mark.parametrize("t", (1, 64, 2048))
@torch.inference_mode()
def test_fused_experts_ep2_combine_is_vllm(monkeypatch, bf16_model, t):
    _, w13, w2, k, _ = bf16_model
    e, h = w13.shape[0], w13.shape[2]
    mapping = torch.full((e,), -1, dtype=torch.int32, device="cuda")
    mapping[e // 2 :] = torch.arange(e // 2, dtype=torch.int32, device="cuda")
    x = torch.randn(t, h, device="cuda").bfloat16()
    weights, ids = _routing(t, e, k, seed=100 + t)
    rows = _capture(monkeypatch)
    out = Experts()(x, w13[e // 2 :], w2[e // 2 :], weights, ids, expert_map=mapping)
    live = (ids >= 0) & (mapping[ids.clamp_min(0).long()] >= 0)
    expected = combine_reference(rows[0].view(t, k, h), weights, live)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
