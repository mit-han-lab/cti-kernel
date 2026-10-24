# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton block-FP8 expert parallelism and checkpoint ABI qualification."""

import pytest
import torch

from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE, pack_block_fp8_experts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires compute capability 8.0 or newer",
)


@pytest.fixture(scope="module", autouse=True)
def exact_fp32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_params(experts, intermediate):
    torch.manual_seed(218)
    tensors = []
    for n, k in ((2 * intermediate, 2048), (2048, intermediate)):
        w = torch.empty(experts, n, k, dtype=torch.float8_e4m3fn, device="cuda")
        for e in range(experts):
            w[e].copy_((torch.randn(n, k, device="cuda") * 16).to(w.dtype))
        s = torch.exp2(torch.rand(experts, n // 128, k // 128, device="cuda") * 2 - 11)
        tensors.extend((w, s.bfloat16()))
    return tuple(tensors)


@pytest.fixture(scope="module")
def params():
    return make_params(256, 512)


@pytest.fixture(scope="module")
def qwen3_params():
    return make_params(128, 768)


def dequantize(q, s):
    return q.float() * s.float().repeat_interleave(128, -2).repeat_interleave(128, -1)


def quantized_input(x):
    q, s = quantize_per_token_group_fp8(x)
    return q.float() * s.repeat_interleave(128, -1)


def reference(x, weights, ids, params, mapping):
    w13, s13, w2, s2 = params
    # Host inspection is confined to the oracle, never the operation.
    local = torch.where(ids >= 0, mapping[ids.clamp_min(0).long()], -1).flatten()
    a = quantized_input(x)
    act = torch.zeros(ids.numel(), w2.shape[-1], dtype=torch.bfloat16, device=x.device)
    rows = [(e, (local == e).nonzero().flatten()) for e in local.unique().tolist() if e >= 0]
    for e, routes in rows:
        gu = (a[routes // ids.shape[1]] @ dequantize(w13[e], s13[e]).T).bfloat16().float()
        gate, up = gu.chunk(2, -1)
        act[routes] = (torch.nn.functional.silu(gate) * up).bfloat16()
    h = quantized_input(act)
    down = torch.zeros(ids.numel(), x.shape[1], device=x.device, dtype=torch.bfloat16)
    for e, routes in rows:
        down[routes] = (
            (h[routes] @ dequantize(w2[e], s2[e]).T) * weights.flatten()[routes, None]
        ).bfloat16()
    total = torch.zeros_like(x, dtype=torch.float32)
    for route in down.view(x.shape[0], ids.shape[1], -1).unbind(1):
        total += route.float()
    return total.bfloat16()


def close(actual, expected):
    # Global relative RMS avoids over-penalizing near-zero cancellation. This
    # includes FP32 dot order and BF16 intermediate/route rounding differences.
    error = (actual.float() - expected.float()).norm()
    assert error <= 0.02 * expected.float().norm().clamp_min(1e-20)


def case(m, skewed, global_experts=256):
    x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(global_experts, (m, 8), device="cuda", dtype=torch.int32)
    if skewed:
        ids[:, :6] %= 4
    ids.flatten()[::11] = -1
    weights = torch.rand(m, 8, device="cuda")
    weights /= weights.sum(-1, keepdim=True)
    return x, weights, ids


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("skewed", [False, True])
@torch.inference_mode()
def test_ep(params, m, skewed):
    x, weights, ids = case(m, skewed)
    baseline = BlockFP8MoE(*params)(x, weights, ids)
    for ep in (1, 2):
        outputs = []
        for rank in range(ep):
            start, end = rank * 256 // ep, (rank + 1) * 256 // ep
            mapping = torch.full((256,), -1, dtype=torch.int32, device="cuda")
            mapping[start:end] = torch.arange(end - start, dtype=torch.int32, device="cuda")
            local_params = tuple(t[start:end] for t in params)
            op = BlockFP8MoE(*local_params, expert_map=mapping, num_global_experts=256)
            assert op.provider == op.provider_for(m) == "triton-fp8"
            actual = op(x, weights, ids)
            close(actual, reference(x, weights, ids, local_params, mapping))
            assert torch.equal(actual, op(x, weights, ids))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op(x, weights, ids)
            graph.replay()
            assert torch.equal(actual, captured)
            assert op._cute_scales is None
            outputs.append(actual.float())
        close(sum(outputs), baseline)


@torch.inference_mode()
def test_holes_scales_packing_graph(params):
    # Shuffled local ids exercise translation independently of contiguous EP windows.
    chosen = torch.randperm(256, device="cuda")[:128]
    mapping = torch.full((256,), -1, dtype=torch.int32, device="cuda")
    mapping[chosen] = torch.arange(128, dtype=torch.int32, device="cuda")
    local = tuple(t[chosen] for t in params)
    w13, s13, w2, s2 = local
    packed = pack_block_fp8_experts(
        w13[:, :512].contiguous(),
        s13[:, :4].contiguous(),
        w13[:, 512:].contiguous(),
        s13[:, 4:].contiguous(),
        w2,
        s2,
    )
    op = BlockFP8MoE(*packed, expert_map=mapping)
    fp32 = BlockFP8MoE(w13, s13.float(), w2, s2.float(), expert_map=mapping)
    x, weights, ids = case(4, False)
    remote = (mapping == -1).nonzero().flatten()[:8].to(torch.int32)
    ids[0] = remote  # All eight routes remote, without any padding ids.
    ids[1] = -1
    actual = op(x, weights, ids)
    assert torch.count_nonzero(actual[:2]).item() == 0
    close(actual, reference(x, weights, ids, local, mapping))
    assert torch.equal(actual, fp32(x, weights, ids))
    op.prepare([4])
    graph = torch.cuda.CUDAGraph()
    saved_ids = ids.clone()
    with torch.cuda.graph(graph):
        captured = op(x, weights, ids)
    graph.replay()
    assert torch.equal(actual, captured)
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    ids.copy_(saved_ids)
    graph.replay()
    assert torch.equal(actual, captured)
    assert op(x[:0], weights[:0], ids[:0]).shape == (0, 2048)


@torch.inference_mode()
def test_bf16_kernel_map():
    from ctikernels.moe_impl import fused_moe_kernel_triton, moe_align_block_size

    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[3, 0, -1, 1], [0, 2, -1, 0]], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([-1, 1, -1, 0], device="cuda", dtype=torch.int32)
    weights = torch.full((2, 4), 0.25, device="cuda")
    sorted_ids, experts, padded = moe_align_block_size(ids, 16, 4)
    out = torch.empty(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    fused_moe_kernel_triton(
        x,
        b,
        out,
        weights,
        ids,
        sorted_ids,
        experts,
        padded,
        True,
        4,
        config,
        torch.bfloat16,
        expert_map=mapping,
    )
    expected = torch.zeros_like(out)
    expected[0, 0] = ((x[0].float() @ b[0].float().T) * 0.25).bfloat16()
    expected[0, 3] = ((x[0].float() @ b[1].float().T) * 0.25).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
    assert torch.count_nonzero(out[1]).item() == 0


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("fused_act", [False, True])
@torch.inference_mode()
def test_route_blocks_match_alignment(params, m, ep, fused_act, monkeypatch):
    from ctikernels.moe_block_fp8 import _dsl_available

    if fused_act and (
        torch.cuda.get_device_capability() not in ((12, 0), (12, 1)) or not _dsl_available()
    ):
        pytest.skip("fused activation requires SM120/121 and DSL 4.8.0")
    monkeypatch.setenv("CTI_MOE_FP8_CUTE", "1" if fused_act else "0")
    x, weights, ids = case(m, True)
    for rank in range(ep):
        start, end = rank * 256 // ep, (rank + 1) * 256 // ep
        mapping = torch.full((256,), -1, dtype=torch.int32, device="cuda")
        mapping[start:end] = torch.arange(end - start, dtype=torch.int32, device="cuda")
        op = BlockFP8MoE(*(t[start:end] for t in params), expert_map=mapping, provider="triton-fp8")
        config = op.config_for(m)
        assert config[1]["act"] == ("cute-silu-quant" if fused_act else "triton-silu-quant")
        monkeypatch.setattr(op, "config_for", lambda _, config=config: config)
        for c in config:
            c["route_blocks"] = False
        op.prepare([m])
        aligned = op(x, weights, ids)
        for c in config:
            c["route_blocks"] = True
        # Changing policy invalidates readiness for the new configuration.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            with pytest.raises(RuntimeError, match="prepare"):
                op(x, weights, ids)
        op.prepare([m])
        actual = op(x, weights, ids)
        assert torch.equal(actual, aligned)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x, weights, ids)
        graph.replay()
        assert torch.equal(actual, captured)
        saved_ids = ids.clone()
        ids.fill_(-1)
        graph.replay()
        assert torch.count_nonzero(captured).item() == 0
        ids.copy_(saved_ids)
        graph.replay()
        assert torch.equal(actual, captured)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("ep", [1, 2])
@torch.inference_mode()
def test_qwen3_ep(qwen3_params, m, ep):
    x, weights, ids = case(m, False, global_experts=128)
    ids[:, 0], ids[:, 1] = 0, 127
    baseline = BlockFP8MoE(*qwen3_params)(x, weights, ids)
    outputs = []
    for rank in range(ep):
        start, end = rank * 128 // ep, (rank + 1) * 128 // ep
        mapping = torch.full((128,), -1, dtype=torch.int32, device="cuda")
        mapping[start:end] = torch.arange(end - start, dtype=torch.int32, device="cuda")
        local = tuple(t[start:end] for t in qwen3_params)
        op = BlockFP8MoE(*local, expert_map=mapping, num_global_experts=128)
        op.prepare([m])
        actual = op(x, weights, ids)
        close(actual, reference(x, weights, ids, local, mapping))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op(x, weights, ids)
        graph.replay()
        assert torch.equal(actual, captured)
        outputs.append(actual.float())
        saved_ids = ids.clone()
        ids.fill_(-1)
        graph.replay()
        assert torch.count_nonzero(captured).item() == 0
        ids.copy_(saved_ids)
        graph.replay()
        assert torch.equal(actual, captured)
    close(sum(outputs), baseline)
