# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Local-only MoE alignment, numerics, and capture qualification."""

import pytest
import torch

from ctikernels.moe import Experts, fused_experts, moe_align_block_size_local
from ctikernels.moe_impl import moe_align_block_size, moe_sum_reduce_triton

from ._moe_helpers import _check_align

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _mapping(kind, e=256):
    mapping = torch.full((e,), -1, dtype=torch.int32)
    if kind == "identity":
        owned = torch.arange(e)
    elif kind == "halves":
        owned = torch.arange(e // 2, e)
    elif kind == "random":
        owned = torch.randperm(e, generator=torch.Generator().manual_seed(10))[: e // 4]
    else:
        owned = torch.empty(0, dtype=torch.int64)
    mapping[owned] = torch.arange(len(owned), dtype=torch.int32)
    return mapping, max(len(owned), 1)


@pytest.mark.parametrize("t", [0, 1, 2, 3, 4, 8, 64, 512, 513, 4096, 8192])
@pytest.mark.parametrize("kind", ["identity", "halves", "random", "remote"])
def test_align(t, kind):
    mapping, local = _mapping(kind)
    ids = torch.randint(256, (t, 8), device="cuda", dtype=torch.int32)
    if t:
        ids[-1] = -1
    if t > 1:
        ids[0, 0] = -1
    _check_align(ids, mapping.cuda(), local)


@pytest.mark.parametrize("block", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("t", [1, 512, 513])
def test_align_block_sizes_and_skew(block, t):
    ids = torch.full((t, 8), 3, device="cuda", dtype=torch.int32)
    mapping = torch.tensor([-1, 2, -1, 0], device="cuda", dtype=torch.int32)
    _check_align(ids, mapping, 3, block)


def test_align_million_routes_and_runtime_shapes():
    mapping = torch.arange(512, device="cuda", dtype=torch.int32)
    for shape in ((1, 8), (9, 8), (8, 8), (2, 8), (512, 8), (1280, 8), (2**17, 8)):
        _check_align(torch.randint(512, shape, device="cuda", dtype=torch.int32), mapping, 512)


@pytest.mark.parametrize("t", [1, 2, 3, 4])
@pytest.mark.parametrize("block", [8, 16, 64, 256])
def test_ep1_small_align_matches_provider(t, block):
    provider = pytest.importorskip("sgl_kernel")
    e = 128
    ids = torch.randint(e, (t, 8), device="cuda", dtype=torch.int32)
    n = ids.numel()
    actual = moe_align_block_size(ids, block, e, expert_map=None)
    capacity = n + (e + 1) * (block - 1)
    options = {"device": ids.device, "dtype": torch.int32}
    old_ids = torch.empty(capacity, **options)
    old_experts = torch.empty((capacity + block - 1) // block, **options)
    old_total = torch.empty(1, **options)
    provider.moe_align_block_size(
        ids, e + 1, block, old_ids, old_experts, old_total, torch.empty(e + 2, **options), True
    )
    assert actual[0].shape == old_ids.shape and actual[1].shape == old_experts.shape
    expected_routes = ids.cpu().flatten()
    for aligned in (actual, (old_ids, old_experts, old_total)):
        sorted_ids, experts = aligned[0].cpu(), aligned[1].cpu()
        total = aligned[2].item()
        assigned = experts[: total // block].repeat_interleave(block)
        for expert in range(e):
            segment = sorted_ids[:total][assigned == expert]
            routes = torch.where(expected_routes == expert)[0]
            assert len(segment) == (len(routes) + block - 1) // block * block
            assert torch.equal(segment[segment < n].sort().values, routes)
            assert (segment[segment >= n] == n).all()
    total = actual[2].item()
    assert total == old_total.item()
    assert (actual[0][total:] == n).all()
    assert (actual[1][total // block :] == -1).all()


@pytest.fixture(scope="module", params=[(8, 64, 32), (256, 2048, 512)], ids=["small", "qwen"])
def weights(request):
    for provider in ("cutlass", "triton", "flashinfer", "sgl_kernel"):
        pytest.importorskip(provider)
    e, h, i = request.param
    torch.manual_seed(12)
    w1 = (torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) / h**0.5).contiguous()
    w2 = (torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) / i**0.5).contiguous()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield w1, w2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _routing(t, e, h):
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    probs, ids = torch.rand(t, e, device="cuda").topk(8, dim=-1)
    probs /= probs.sum(-1, keepdim=True)
    return x, probs, ids.to(torch.int32)


def _reference(x, w1, w2, probs, ids, mapping):
    # FP32 projections and activation; avoid making an FP32 copy of all experts.
    result = torch.zeros_like(x, dtype=torch.float32)
    valid = ids >= 0
    local_ids = torch.full_like(ids, -1)
    local_ids[valid] = mapping[ids[valid].long()]
    for e in range(w1.shape[0]):
        rows, choices = torch.where(local_ids == e)
        if rows.numel():
            up = x[rows].float() @ w1[e].float().T
            gate, value = up.chunk(2, -1)
            down = (torch.nn.functional.silu(gate) * value) @ w2[e].float().T
            result.index_add_(0, rows, down * probs[rows, choices, None])
    return result


@pytest.mark.parametrize("ep", [2, 4])
def test_fused_experts_and_ep_sum(weights, ep):
    w1, w2 = weights
    e, _, h = w1.shape
    x, probs, ids = _routing(8, e, h)
    full_map = torch.arange(e, device="cuda", dtype=torch.int32)
    baseline = fused_experts(x, w1, w2, probs, ids)
    ref_full = _reference(x, w1, w2, probs, ids, full_map)
    baseline_error = (baseline.float() - ref_full).square().mean().sqrt()
    partials = []
    partition = torch.randperm(e, device="cuda").reshape(ep, -1)
    for owned in partition:
        mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
        mapping[owned] = torch.arange(len(owned), device="cuda", dtype=torch.int32)
        local_w1, local_w2 = w1[owned].contiguous(), w2[owned].contiguous()
        actual = fused_experts(x, local_w1, local_w2, probs, ids, expert_map=mapping)
        expected = _reference(x, local_w1, local_w2, probs, ids, mapping)
        rms = expected.square().mean().sqrt().clamp_min(1e-6)
        relative = (actual.float() - expected).square().mean().sqrt() / rms
        gate = max(0.012, float(3 * baseline_error / rms))
        assert float(relative) <= gate
        partials.append(actual.float())
    torch.testing.assert_close(sum(partials), baseline.float(), atol=0.012, rtol=0.025)
    # Identity maps also check full BF16 composition against the old align path.
    identity = fused_experts(x, w1, w2, probs, ids, expert_map=full_map)
    torch.testing.assert_close(identity, baseline, atol=0.004, rtol=0.015)


def test_remote_padding_out_and_empty(weights):
    w1, w2 = weights
    e, _, h = w1.shape
    x, probs, ids = _routing(3, e, h)
    mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    mapping[0] = 0
    ids[0] = 1
    ids[1] = -1
    ids[2] = 0
    expected = fused_experts(x, w1[:1], w2[:1], probs, ids, expert_map=mapping)
    assert torch.equal(expected[:2], torch.zeros_like(expected[:2]))
    assert not torch.signbit(expected[:2]).any()
    clone = x.clone()
    actual = fused_experts(clone, w1[:1], w2[:1], probs, ids, expert_map=mapping, out=clone)
    assert actual is clone
    torch.testing.assert_close(actual, expected)
    config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 3}
    overridden = fused_experts(x, w1[:1], w2[:1], probs, ids, expert_map=mapping, config=config)
    torch.testing.assert_close(overridden, expected, atol=0.004, rtol=0.015)
    assert config == {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 3}
    assert fused_experts(x[:0], w1[:1], w2[:1], probs[:0], ids[:0], expert_map=mapping).shape == (
        0,
        h,
    )
    empty = x[:0]
    assert (
        fused_experts(empty, w1[:1], w2[:1], probs[:0], ids[:0], expert_map=mapping, out=empty)
        is empty
    )
    mapping.fill_(-1)
    all_remote = fused_experts(x, w1[:1], w2[:1], probs, ids, expert_map=mapping)
    assert torch.equal(all_remote, torch.zeros_like(all_remote))
    assert not torch.signbit(all_remote).any()


@pytest.mark.parametrize("t", [4, 512])
def test_capture_replays_changed_routes(weights, t):
    w1, w2 = weights
    e, _, h = w1.shape
    x, probs, ids = _routing(t, e, h)
    owned = torch.randperm(e, device="cuda")[: e // 2]
    mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    mapping[owned] = torch.arange(len(owned), device="cuda", dtype=torch.int32)
    local_w1, local_w2 = w1[owned].contiguous(), w2[owned].contiguous()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_experts(x, local_w1, local_w2, probs, ids, expert_map=mapping)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fused_experts(x, local_w1, local_w2, probs, ids, expert_map=mapping)
    for _ in range(2):
        _, new_probs, new_ids = _routing(t, e, h)
        new_ids[-1] = -1
        ids.copy_(new_ids)
        probs.copy_(new_probs)
        graph.replay()
        eager = fused_experts(x, local_w1, local_w2, probs, ids, expert_map=mapping)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)


@pytest.mark.parametrize("t", [4, 512])
@pytest.mark.parametrize("provider", ["triton-fp8"])
@torch.inference_mode()
def test_block_fp8_capture_replays_changed_routes(t, provider):
    for dependency in ("cutlass", "triton", "flashinfer"):
        pytest.importorskip(dependency)
    if provider == "triton-fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 MMA requires compute capability 8.9+")
    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.moe_impl import (
        block_fp8_moe_config,
        fused_moe_kernel_triton,
        moe_align_block_size,
    )

    torch.manual_seed(31 + t)
    e, local, h, inter, top_k = 32, 8, 512, 256, 8
    mapping = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    mapping[8:16] = torch.arange(local, device="cuda", dtype=torch.int32)
    w1 = (torch.randn(local, 2 * inter, h, device="cuda") * 16).to(torch.float8_e4m3fn)
    w2 = (torch.randn(local, h, inter, device="cuda") * 16).to(torch.float8_e4m3fn)
    s1 = torch.full((local, 2 * inter // 128, h // 128), 0.002, device="cuda")
    s2 = torch.full((local, h // 128, inter // 128), 0.002, device="cuda")
    x, probs, ids = _routing(t, e, h)
    config = block_fp8_moe_config(
        t, local, 2 * inter, h, top_k, capability=torch.cuda.get_device_capability()
    )

    def forward(mapped=True):
        route_ids = ids if mapped else torch.where(ids >= 0, mapping[ids.clamp(min=0).long()], -1)
        aligned = moe_align_block_size(
            route_ids, config["BLOCK_SIZE_M"], local, mapping if mapped else None
        )
        # K2MAP leaves remote rows unwritten; poison them on every replay.
        gate_up = torch.full((t, top_k, 2 * inter), torch.nan, device="cuda", dtype=x.dtype)
        down = torch.full((t, top_k, h), torch.nan, device="cuda", dtype=x.dtype)
        q, scale = quantize_per_token_group_fp8(x)
        fused_moe_kernel_triton(
            q, w1, gate_up, probs, route_ids, *aligned, False, top_k, config,
            x.dtype, a_scale=scale, b_scale=s1, block_fp8_provider=provider,
        )  # fmt: skip
        act = silu_and_mul(gate_up.view(-1, 2 * inter))
        q2, scale2 = quantize_per_token_group_fp8(act)
        fused_moe_kernel_triton(
            q2, w2, down, probs, route_ids, *aligned, False, 1, config,
            x.dtype, a_scale=scale2, b_scale=s2, block_fp8_provider=provider,
        )  # fmt: skip
        out = torch.empty_like(x)
        if mapped:
            moe_sum_reduce_triton(down, out, topk_ids=ids, expert_map=mapping, topk_weights=probs)
        else:
            moe_sum_reduce_triton(down, out, topk_ids=route_ids, topk_weights=probs)
        return out, gate_up, down

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, gate_up, down = forward()
    for all_remote in (False, False, True):
        _, new_probs, new_ids = _routing(t, e, h)
        new_ids[0] = torch.arange(top_k, device="cuda")  # all remote
        new_ids[1] = torch.arange(8, 16, device="cuda")  # all local
        new_ids[-1] = -1  # graph padding
        if all_remote:
            new_ids.fill_(0)
        ids.copy_(new_ids)
        probs.copy_(new_probs)
        graph.replay()
        eager, _, _ = forward()
        premapped, _, _ = forward(mapped=False)
        assert torch.equal(captured, eager) and torch.equal(captured, premapped)
        remote = ids.lt(0) | mapping[ids.clamp(min=0).long()].lt(0)
        assert gate_up[remote].isnan().all() and down[remote].isnan().all()
        assert torch.equal(captured[remote.all(1)], torch.zeros_like(captured[remote.all(1)]))
        assert not torch.signbit(captured[remote.all(1)]).any()


@pytest.mark.parametrize("masked", [False, True])
def test_reduce_unwritten_remote_rows(masked):
    pytest.importorskip("triton")
    x = torch.randn(3, 8, 2053, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor(
        [[0, 1, -1, 0, 1, 0, 1, 1], [-1] * 8, [1] * 8], device="cuda", dtype=torch.int32
    )
    mapping = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    args = {}
    if masked:
        keep = ids == 0
        x[~keep] = float("nan")
        expected = torch.where(keep[..., None], x, 0).float().sum(1).bfloat16()
        args = {"topk_ids": ids, "expert_map": mapping}
    else:
        expected = x.float().sum(1).bfloat16()
    out = torch.empty_like(expected)
    moe_sum_reduce_triton(x, out, **args)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    if masked:
        assert not torch.signbit(out[1:]).any()


def test_prefill_align_latency(record_property):
    if "5090" not in torch.cuda.get_device_name():
        pytest.skip("latency budget targets RTX 5090")
    ids = torch.randint(256, (8192, 8), device="cuda", dtype=torch.int32)
    mapping = torch.arange(256, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        moe_align_block_size_local(ids, 64, mapping, 256)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        aligned = moe_align_block_size_local(ids, 64, mapping, 256)
    for _ in range(5):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    microseconds = start.elapsed_time(end) * 1000 / 100
    record_property("align_65536_routes_us", microseconds)
    assert aligned[2].item() > 0
    assert microseconds < 30


@pytest.mark.parametrize("t", [1, 4, 64, 512])
@pytest.mark.parametrize("block", [16, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ep1_small_align_keeps_negative_routes(t, block, dtype):
    """Without a map, -1 routes fill expert -1 blocks as in SGLang's layout (int64 IDs too)."""
    provider = pytest.importorskip("sgl_kernel")
    e = 128
    gen = torch.Generator(device="cuda").manual_seed(7 + t)
    ids = torch.randint(e, (t, 8), device="cuda", dtype=dtype, generator=gen)
    ids[torch.rand(ids.shape, device="cuda", generator=gen) < 0.3] = -1
    ids[0] = -1
    n = ids.numel()
    aligned = moe_align_block_size(ids, block, e)
    narrow = moe_align_block_size(ids.to(torch.int32), block, e)
    assert all(torch.equal(a, b) for a, b in zip(aligned, narrow, strict=True))
    options = {"device": ids.device, "dtype": torch.int32}
    capacity = n + (e + 1) * (block - 1)
    old_total = torch.empty(1, **options)
    provider.moe_align_block_size(
        ids.to(torch.int32), e + 1, block, torch.empty(capacity, **options),
        torch.empty((capacity + block - 1) // block, **options), old_total,
        torch.empty(e + 2, **options), True,
    )  # fmt: skip
    routes = ids.cpu().flatten()
    for sorted_ids, experts, padded in (aligned, narrow):
        total = int(padded.item())
        assert total == old_total.item()
        assigned = experts.cpu()[: total // block].repeat_interleave(block)
        used = sorted_ids.cpu()[:total]
        assert (sorted_ids.cpu()[total:] == n).all()
        for expert in [-1, *range(e)]:
            segment = used[assigned == expert]
            expected = torch.where(routes == expert)[0].to(torch.int32)
            assert len(segment) == (len(expected) + block - 1) // block * block
            assert torch.equal(segment[segment < n].sort().values, expected)
            assert (segment[segment >= n] == n).all()


@pytest.mark.parametrize("t", [1, 4, 64, 512])
def test_reused_experts_return_zero_for_negative_routes(weights, t):
    """A reused op must not sum the previous call's rows for routes that are now -1."""
    w1, w2 = weights
    e = w1.shape[0]
    x, probs, ids = _routing(t, e, w1.shape[2])
    op = Experts()
    assert op(x, w1, w2, probs, ids).abs().amax() > 0
    ids.fill_(-1)
    assert torch.count_nonzero(op(x, w1, w2, probs, ids).view(torch.int16)) == 0
    x, probs, ids = _routing(t, e, w1.shape[2])
    op(x, w1, w2, probs, ids)
    ids[::2] = -1
    ids[:, 0] = -1
    actual = op(x, w1, w2, probs, ids)
    assert torch.count_nonzero(actual[::2].view(torch.int16)) == 0
    # The masked expert_map path never reads -1 routes' rows.
    full_map = torch.arange(e, device="cuda", dtype=torch.int32)
    masked = Experts()(x, w1, w2, probs, ids, expert_map=full_map)
    torch.testing.assert_close(actual, masked, atol=0.004, rtol=0.015)
