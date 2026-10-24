# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""FP16 inference qualification, using small experts to target <2 min per device.

The FP64 oracle sees the original FP16 operands; the BF16 comparison casts only
x, so both errors are measured against the same mathematical MoE. No GPU timing
or numerical qualification is implied by successful CPU/offline tests.
"""

import pytest
import torch

from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE, _dsl_available
from ctikernels.moe_impl import fused_moe_kernel_triton, moe_align_block_size, moe_sum_reduce_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or not _dsl_available(),
    reason="requires SM120/121 and CuTe DSL 4.8.0",
)

E, H, INTERMEDIATE, TOP_K = 8, 256, 256, 8


@pytest.fixture(autouse=True)
def enable_cute_activation(monkeypatch):
    monkeypatch.setenv("CTI_MOE_FP8_CUTE", "1")


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(218)
    tensors = []
    for n, k in ((2 * INTERMEDIATE, H), (H, INTERMEDIATE)):
        w = (torch.randn(E, n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
        s = torch.exp2(torch.rand(E, n // 128, k // 128, device="cuda") * 2 - 11)
        tensors.extend((w, s.bfloat16()))
    return tuple(tensors)


def case(m, skewed=False):
    torch.manual_seed(128 + m)
    x = torch.randn(m, H, device="cuda", dtype=torch.float16)
    ids = (torch.arange(m * TOP_K, device="cuda") % E).to(torch.int32).view(m, TOP_K)
    if skewed:
        ids[:, :6] = 0
        # Keep both EP ranks represented, even at M=1 in the skewed case.
        ids[:, -2:] = torch.tensor([0, E - 1], device="cuda", dtype=torch.int32)
    ids.flatten()[::11] = -1
    weights = torch.rand(m, TOP_K, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    return x, weights, ids


def dequantize(w, s):
    return w.double() * s.double().repeat_interleave(128, -2).repeat_interleave(128, -1)


def fp64_reference(x, weights, ids, params, start, end):
    w13, s13, w2, s2 = params
    out = torch.zeros_like(x, dtype=torch.float64)
    # Sum slots in FP64, without atomics; only the oracle inspects route values.
    down = torch.zeros(ids.numel(), H, device=x.device, dtype=torch.float64)
    for expert in range(start, end):
        routes = (ids.flatten() == expert).nonzero().flatten()
        if not routes.numel():
            continue
        gu = x[routes // TOP_K].double() @ dequantize(w13[expert], s13[expert]).T
        gate, up = gu.chunk(2, -1)
        act = torch.nn.functional.silu(gate) * up
        down[routes] = (act @ dequantize(w2[expert], s2[expert]).T) * weights.flatten()[
            routes, None
        ].double()
    for route in down.view(x.shape[0], TOP_K, H).unbind(1):
        out += route
    return out


def original_chain(op, x, weights, ids):
    """The public split FP16 chain, independently composed with global alignment."""
    configs = tuple(dict(c) for c in op.config_for(x.shape[0]))
    for config in configs:
        config.pop("route_blocks")
        config.pop("act", None)
    layout = moe_align_block_size(ids, configs[0]["BLOCK_SIZE_M"], E)
    gu = torch.empty(x.shape[0], TOP_K, 2 * INTERMEDIATE, device=x.device, dtype=x.dtype)
    down = torch.empty(x.shape[0], TOP_K, H, device=x.device, dtype=x.dtype)
    q, scales = quantize_per_token_group_fp8(x)
    fused_moe_kernel_triton(
        q,
        op.w13,
        gu,
        weights,
        ids,
        *layout,
        False,
        TOP_K,
        configs[0],
        x.dtype,
        a_scale=scales,
        b_scale=op.s13,
        block_fp8_provider="triton-fp8",
        expert_map=op.expert_map,
        num_global_experts=E,
    )
    act = torch.empty(ids.numel(), INTERMEDIATE, device=x.device, dtype=x.dtype)
    silu_and_mul(gu.view(-1, 2 * INTERMEDIATE), act)
    q, scales = quantize_per_token_group_fp8(act)
    fused_moe_kernel_triton(
        q,
        op.w2,
        down,
        weights,
        ids,
        *layout,
        False,
        1,
        configs[1],
        x.dtype,
        a_scale=scales,
        b_scale=op.s2,
        block_fp8_provider="triton-fp8",
        expert_map=op.expert_map,
        num_global_experts=E,
    )
    out = torch.empty_like(x)
    moe_sum_reduce_triton(down, out, topk_ids=ids, expert_map=op.expert_map, topk_weights=weights)
    return out, q, scales


@pytest.mark.parametrize("m", [1, 2, 4, 8, 64, 300])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("skewed", [False, True])
@torch.inference_mode()
def test_fp16_matches_original_chain(params, m, ep, skewed):
    """FP16 runs the public split chain (the CuTe activation is BF16-only): bitwise equal.

    The FP64 oracle is a sanity bound only; FP16 intermediates can be less accurate than
    BF16 for small magnitudes (subnormals), exactly as in SGLang's FP16 composition.
    """
    x, weights, ids = case(m, skewed)
    for rank in range(ep):
        start, end = rank * E // ep, (rank + 1) * E // ep
        mapping = None
        if ep == 2:
            mapping = torch.full((E,), -1, device="cuda", dtype=torch.int32)
            mapping[start:end] = torch.arange(end - start, device="cuda", dtype=torch.int32)
        op = BlockFP8MoE(*(t[start:end] for t in params), expert_map=mapping, top_k=TOP_K)
        actual = op(x, weights, ids)
        assert actual.dtype == torch.float16
        original, _, _ = original_chain(op, x, weights, ids)
        assert torch.equal(actual, original), (m, ep, rank, skewed)
        reference = fp64_reference(x, weights, ids, params, start, end)
        error = (actual.double() - reference).norm() / reference.norm().clamp_min(1e-30)
        assert torch.isfinite(actual).all() and error < 0.1, (m, ep, rank, skewed, error)


@torch.inference_mode()
def test_fp16_prepare_capture_and_rejections(params):
    op = BlockFP8MoE(*params, top_k=TOP_K)
    x, weights, ids = case(2)
    op.prepare([2])  # BF16 preparation must leave FP16 cold.
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        with pytest.raises(RuntimeError, match="prepare.*dtype"):
            op(x, weights, ids)
    op.prepare([1, 2, 4, 8, 64, 300], dtype=torch.float16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weights, ids)
    graph.replay()
    torch.testing.assert_close(captured, op(x, weights, ids), rtol=0, atol=0)
    assert captured.dtype == torch.float16
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(captured) == 0
    with pytest.raises(TypeError, match="FP32 hidden dtype gap"):
        op(x.float(), weights, ids)
    cute = BlockFP8MoE(*params, top_k=TOP_K, provider="cute-fp8")
    with pytest.raises(TypeError, match="cute-fp8.*BF16 only"):
        cute(x, weights, ids)
    for kwargs in (
        {"return_act": True},
        {"gate_up_hook": lambda gu: None},
        {"down_hook": lambda act: None},
    ):
        with pytest.raises(TypeError, match="BF16 only; FP16 inference"):
            op.forward_staged(x, weights, ids, **kwargs)
    with pytest.raises(TypeError, match="BF16 only; FP16 inference"):
        op.dx(x, weights, ids, torch.ones_like(x))
