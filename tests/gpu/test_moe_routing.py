# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import pytest
import torch

from ctikernels.moe_impl import (
    fused_moe_kernel_triton,
    moe_align_block_size,
    moe_sum_reduce_triton,
)


def _stable_align(ids, block, experts, mapping=None, invalid=True):
    """moe_align_block_size's layout with routes in route order inside each bucket."""
    flat = ids.flatten().long().cpu()
    n = flat.numel()
    g = mapping.numel() if mapping is not None else experts
    ok = (flat >= 0) & (flat < g)
    local = flat.clamp(0, g - 1)
    if mapping is not None:
        local = mapping.cpu().long()[local]
    local = torch.where(ok & (local >= 0) & (local < experts), local, -1)
    buckets = experts + invalid
    capacity = n + max(buckets, experts + 1 if invalid else 0) * (block - 1)
    sorted_ids = torch.full((capacity,), n, dtype=torch.int32)
    owner = torch.full((-(-capacity // block),), -1, dtype=torch.int32)
    offset = 0
    for b in range(buckets):
        routes = (local == (b if b < experts else -1)).nonzero().flatten()
        sorted_ids[offset : offset + routes.numel()] = routes.to(torch.int32)
        padded = -(-routes.numel() // block) * block
        if b < experts:
            owner[offset // block : (offset + padded) // block] = b
        offset += padded
    return sorted_ids, owner, torch.tensor([offset], dtype=torch.int32)


@pytest.mark.parametrize(
    ("tokens", "top_k", "experts", "block"),
    (
        (1, 8, 128, 16),
        (37, 8, 128, 24),
        (513, 8, 128, 64),
        (600, 8, 256, 48),
        (5000, 8, 128, 16),
        (4097, 1, 513, 512),
        (2048, 9, 8, 128),
    ),
)
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_align_matches_stable_reference(tokens, top_k, experts, block, dtype):
    """Every row count and block size, -1 padding and out-of-range ids, bitwise."""
    from ctikernels.moe.triton._slot_layout import align_block_size

    gen = torch.Generator(device="cuda").manual_seed(tokens)
    ids = torch.randint(experts, (tokens, top_k), generator=gen, device="cuda", dtype=dtype)
    ids.view(-1)[::7] = -1
    ids.view(-1)[3::11] = experts
    got = align_block_size(ids, block, experts, invalid_bucket=True)
    want = _stable_align(ids, block, experts)
    for a, b in zip(got, want, strict=True):
        assert a.dtype == torch.int32 and torch.equal(a.cpu(), b)
    cute = ids.numel() <= 4096 and experts <= 512 and 16 <= block <= 256 and not block & (block - 1)
    if not cute:
        # The public entry takes this path where CuTe does not run.
        public = moe_align_block_size(ids, block, experts)
        for a, b in zip(public, want, strict=True):
            assert torch.equal(a.cpu(), b)


@pytest.mark.parametrize("tokens", (3, 700))
def test_align_expert_map_and_graph_replay(tokens):
    from ctikernels.moe.triton._slot_layout import align_block_size

    experts, block = 16, 32
    ids = torch.zeros((tokens, 8), dtype=torch.int32, device="cuda")
    mapping = torch.zeros(64, dtype=torch.int32, device="cuda")

    def run():
        return align_block_size(ids, block, experts, mapping, capacity_experts=experts)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        got = run()
    for seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        ids.copy_(torch.randint(-1, 65, ids.shape, generator=gen, device="cuda"))
        mapping.copy_(torch.randperm(64, generator=gen, device="cuda") % 17 - 1)
        graph.replay()
        want = _stable_align(ids, block, experts, mapping, invalid=False)
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a.cpu(), b)


@pytest.mark.parametrize(
    "tokens, ep", ((1, 1), (4, 1), (8, 1), (9, 1), (512, 1), (4, 2), (512, 2), (1024, 2))
)
def test_align_cute_and_triton_agree(tokens, ep):
    """Where both run, CuTe and Triton give the same layout bitwise (route order per expert):
    the decode kernel (<= 64 routes) and the chunked one, EP1 and with an EP map."""
    pytest.importorskip("cutlass")
    from ctikernels.moe import moe_align_block_size_local
    from ctikernels.moe.triton._slot_layout import align_block_size

    experts = 128
    local = experts // ep
    ids = torch.randint(-1, experts, (tokens, 8), dtype=torch.int32, device="cuda")
    mapping = torch.full((experts,), -1, dtype=torch.int32, device="cuda")
    mapping[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
    if ep == 1:
        cute = moe_align_block_size_local(
            ids, 16, mapping, local, capacity_experts=local + 1, identity=True, invalid_bucket=True
        )
        ours = align_block_size(ids, 16, local, invalid_bucket=True)
    else:
        cute = moe_align_block_size_local(ids, 16, mapping, local)
        ours = align_block_size(ids, 16, local, mapping)
    for a, b in zip(cute, ours, strict=True):
        assert torch.equal(a, b)


def _expert_projection_reference(
    inputs: torch.Tensor,
    experts: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    expanded_input: bool,
    weighted: bool,
) -> torch.Tensor:
    """CPU FP32 math, independent of the routed layout and both Triton kernels."""
    inputs, experts = inputs.cpu().float(), experts.cpu().float()
    expert_ids, routing_weights = expert_ids.cpu(), routing_weights.cpu()
    num_tokens, top_k = expert_ids.shape
    output = torch.stack(
        [
            torch.stack(
                [
                    inputs[token * top_k + choice if expanded_input else token]
                    @ experts[expert_ids[token, choice]].T
                    for choice in range(top_k)
                ]
            )
            for token in range(num_tokens)
        ]
    )
    return output * routing_weights[..., None] if weighted else output


@pytest.mark.parametrize(
    ("dtype", "num_tokens", "input_size", "expanded_input", "weighted"),
    [
        (torch.bfloat16, 1, 64, False, False),
        (torch.bfloat16, 19, 70, False, True),
        (torch.float16, 17, 64, True, True),
        (torch.float16, 19, 70, True, False),
    ],
)
@torch.inference_mode()
def test_expert_projection_matches_independent_math_and_replays_live_inputs(
    dtype: torch.dtype,
    num_tokens: int,
    input_size: int,
    expanded_input: bool,
    weighted: bool,
) -> None:
    torch.manual_seed(52)
    top_k, num_experts, output_size, block_size = 3, 5, 65, 16
    inputs = torch.randn(
        num_tokens * (top_k if expanded_input else 1), input_size, device="cuda", dtype=dtype
    )
    experts = torch.randn(num_experts, output_size, input_size, device="cuda", dtype=dtype) * 0.1
    # Experts 1 and 4 are empty. Every token uses a different permutation of 0, 2, 3.
    ids = torch.tensor(
        [
            [(0, 2, 3)[(token + choice) % top_k] for choice in range(top_k)]
            for token in range(num_tokens)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    weights = torch.randn(num_tokens, top_k, device="cuda").softmax(dim=-1)
    sorted_ids, block_experts, padded_count = moe_align_block_size(ids, block_size, num_experts)
    storage = torch.full(
        (num_tokens + 2, top_k, output_size), torch.nan, device="cuda", dtype=dtype
    )
    output = storage[1:-1]
    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 2,
    }
    originals = tuple(tensor.clone() for tensor in (inputs, experts, ids, weights))

    def project() -> None:
        fused_moe_kernel_triton(
            inputs,
            experts,
            output,
            weights,
            ids,
            sorted_ids,
            block_experts,
            padded_count,
            weighted,
            1 if expanded_input else top_k,
            config,
            dtype,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        project()
    stream.synchronize()
    expected = _expert_projection_reference(
        inputs, experts, ids, weights, expanded_input=expanded_input, weighted=weighted
    )
    tolerance = 0.005 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(output.cpu().float(), expected, rtol=tolerance, atol=tolerance)
    for actual, original in zip((inputs, experts, ids, weights), originals, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        project()
    first_output = output.clone()
    inputs.add_(0.125)
    experts.mul_(0.75)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output.fill_(torch.nan)
        graph.replay()
    stream.synchronize()
    expected = _expert_projection_reference(
        inputs, experts, ids, weights, expanded_input=expanded_input, weighted=weighted
    )
    torch.testing.assert_close(output.cpu().float(), expected, rtol=tolerance, atol=tolerance)
    assert not torch.equal(output, first_output)
    assert torch.isnan(storage[[0, -1]]).all().item()
    for actual, original in zip((ids, weights), originals[2:], strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(inputs, originals[0] + 0.125, rtol=0, atol=0)
    torch.testing.assert_close(experts, originals[1] * 0.75, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "top_k", "hidden_size"),
    [
        (torch.bfloat16, 1, 17),
        (torch.bfloat16, 3, 2049),
        (torch.float16, 8, 4096),
    ],
)
@torch.inference_mode()
def test_expert_reduction_matches_fp32_sum_and_preserves_inputs(
    dtype: torch.dtype, top_k: int, hidden_size: int
) -> None:
    torch.manual_seed(53)
    inputs = torch.randn(7, top_k, hidden_size, device="cuda", dtype=dtype)
    original = inputs.clone()
    storage = torch.full((9, hidden_size), torch.nan, device="cuda", dtype=dtype)
    output = storage[1:-1]
    moe_sum_reduce_triton(inputs, output)
    torch.testing.assert_close(output, inputs.float().sum(dim=1).to(dtype), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe_sum_reduce_triton(inputs, output)
    inputs.add_(0.25)
    output.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(output, inputs.float().sum(dim=1).to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(inputs, original + 0.25, rtol=0, atol=0)
    assert torch.isnan(storage[[0, -1]]).all().item()


@pytest.mark.parametrize("t", (1, 7, 64, 1000, 8192))
@pytest.mark.parametrize("shared,mapped,masked", ((True, False, False), (False, True, True)))
def test_training_layout_is_the_stable_layout(t, shared, mapped, masked):
    """The training block's layout equals the stable torch specification, bitwise."""
    from ctikernels.moe.triton._slot_layout import build_layout

    from ..modules._moe_layout_reference import build_slot_layout, slot_expert

    e, k, local = 256, 8, 64 if mapped else 256
    gen = torch.Generator(device="cuda").manual_seed(t)
    sel = torch.rand(t, e, generator=gen, device="cuda").topk(k, dim=1).indices.int()
    live = torch.arange(t, device="cuda") < (t // 2 if masked else t)
    mapping = None
    if mapped:
        mapping = torch.arange(e, dtype=torch.int32, device="cuda") % local
        mapping[::3] = -1
    for bm in (16, 64):
        got = build_layout(sel, live, local, bm, mapping, has_shared=shared)
        want = build_slot_layout(slot_expert(sel.long(), live, local, mapping, shared), local, bm)
        for name in ("sorted_slot", "block_expert", "inv"):
            assert torch.equal(getattr(got, name), getattr(want, name)), name
