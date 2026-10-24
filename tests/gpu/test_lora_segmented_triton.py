# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Triton segmented LoRA provider, through the public ops, on any CUDA cc >= 10.0.

sm_120/sm_121 hosts force the provider that every other device selects.
"""

import pytest
import torch
import torch.nn.functional as F

from ctikernels import GraphWorkspace
from ctikernels.linear import (
    _lora_policy,
    lora_delta_factors_,
    lora_segment_metadata,
    lora_segmented_delta_,
    lora_segmented_projection,
)
from ctikernels.linear.lora import lora_projection, try_prepare_lora_inference_plan

from .test_lora_segmented import SIZES, check, make_case, oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="CUDA cc >= 10.0 required",
)
DTYPES = [torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def triton_provider(monkeypatch):
    monkeypatch.setattr(_lora_policy, "segmented_provider", lambda capability: "triton")


def factor_case(m, k, rank, sizes, dtype, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
    factors = tuple(
        t
        for n in sizes
        for t in (
            torch.randn(rank, k, device="cuda", dtype=dtype) * 0.1,
            torch.randn(n, rank, device="cuda", dtype=dtype) * 0.1,
        )
    )
    return x, factors


def factor_oracle(x, factors, scale):
    return torch.cat(
        [
            scale * (x.cpu().double() @ a.cpu().double().T) @ b.cpu().double().T
            for a, b in zip(factors[::2], factors[1::2], strict=True)
        ],
        dim=1,
    )


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    return graph


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 33, 64, 200, 2048])
def test_ragged_mixed_ranks_csr_scan(m, dtype):
    output, x, a, b, meta, storage = make_case(m, dtype, every_token=m == 64)
    before, sentinel = output.clone(), storage.clone()
    expected = oracle(x, a, b, meta)
    assert lora_segmented_delta_(output, x, a, b, **meta) is output
    check(output, before, expected)
    torch.testing.assert_close(storage[:, :3], sentinel[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, -10:], sentinel[:, -10:], rtol=0, atol=0)
    ptr, ids = meta["seg_indptr"].cpu().tolist(), meta["weight_indices"].cpu().tolist()
    for s, adapter in enumerate(ids):
        if adapter < 0:
            torch.testing.assert_close(
                output[ptr[s] : ptr[s + 1]], before[ptr[s] : ptr[s + 1]], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "sizes,k", [((128,), 256), ((2048, 2048, 4096, 4096, 32, 32), 64), ((17, 9), 37)]
)
def test_slice_topologies_and_tails(dtype, sizes, k):
    pool_rank = 19 if k == 37 else 512
    output, x, a, b, meta, _ = make_case(4, dtype, sizes, k, pool_rank)
    if k == 37:
        meta["ranks"].copy_(torch.tensor([1, 7, 13, 19], device="cuda", dtype=torch.int32))
    before = output.clone()
    expected = oracle(x, a, b, meta)
    lora_segmented_delta_(output, x, a, b, **meta)
    check(output, before, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stride", ["rank", "max_rank"])
@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize("device_metadata", [False, True])
def test_grouped_permutation_graph_replay(dtype, stride, chunk, device_metadata):
    output, x, a, b, meta, _ = make_case(133, dtype)
    ids = ([2, 0, -1, 3] * 34)[:133]
    if device_metadata:
        grouped = lora_segment_metadata(
            torch.tensor(ids, device="cuda"),
            torch.ones(133, device="cuda", dtype=torch.int32),
            total_tokens=133,
            chunk_tokens=chunk,
        )
    else:
        grouped = lora_segment_metadata(ids, [1] * 133, device="cuda", chunk_tokens=chunk)
    meta.update(grouped, a_slice_stride=stride)
    before = output.clone()
    lora_segmented_delta_(output, x, a, b, **meta)
    check(output, before, oracle(x, a, b, meta))
    skipped = torch.tensor([i for i, adapter in enumerate(ids) if adapter < 0], device="cuda")
    torch.testing.assert_close(output[skipped], before[skipped], rtol=0, atol=0)
    eager = output.clone()
    graph = capture(lambda: lora_segmented_delta_(output, x, a, b, **meta))
    output.copy_(before)
    graph.replay()
    torch.testing.assert_close(output, eager, rtol=0, atol=0)  # graph == eager
    meta["permutation"].copy_(meta["permutation"].flip(0))
    output.copy_(before)
    graph.replay()
    check(output, before, oracle(x, a, b, meta))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("splits", [None, 1, 3, 64])
@pytest.mark.parametrize(
    "m,k,rank,sizes", [(1, 4096, 16, (4096,)), (4, 37, 19, (17, 9)), (33, 2048, 64, SIZES)]
)
def test_factors_deterministic_graph_equals_eager(dtype, splits, m, k, rank, sizes):
    x, factors = factor_case(m, k, rank, sizes, dtype)
    scaling = torch.tensor(1.5, device="cuda")
    base = torch.randn(m, sum(sizes), device="cuda", dtype=dtype) * 0.005
    output = base.clone()

    def call():
        return lora_delta_factors_(
            output,
            x,
            factors,
            scaling=scaling,
            output_sizes=sizes,
            config=(splits, None, None, None),
        )

    call()
    check(output, base, factor_oracle(x, factors, 1.5))
    eager = output.clone()
    for _ in range(2):  # repeatable eager calls
        output.copy_(base)
        call()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
    graph = capture(call)
    for _ in range(3):  # replays reuse the self-resetting split counters
        output.copy_(base)
        graph.replay()
        torch.testing.assert_close(output, eager, rtol=0, atol=0)
    for factor in factors:
        factor.mul_(0.9)
    scaling.fill_(-0.75)
    output.copy_(base)
    graph.replay()
    check(output, base, factor_oracle(x, factors, -0.75))


@pytest.mark.parametrize("k,x_value", [(256, 128.0), (4096, 8.0)])  # direct and split-K paths
def test_fp16_shrink_scales_before_rounding(k, x_value):
    x = torch.full((4, k), x_value, device="cuda", dtype=torch.float16)
    a = torch.full((16, k), 2.0, device="cuda", dtype=torch.float16)
    b = torch.full((64, 16), 1 / 16, device="cuda", dtype=torch.float16)
    output = torch.zeros(4, 64, device="cuda", dtype=torch.float16)
    lora_delta_factors_(output, x, (a, b), scaling=1 / 32, output_sizes=(64,))
    torch.testing.assert_close(output, torch.full_like(output, 2048.0))


def test_zero_rank_and_zero_rows_are_noops():
    output, x, a, b, meta, _ = make_case(8, torch.bfloat16)
    before = output.clone()
    meta["ranks"].zero_()
    lora_segmented_delta_(output, x, a, b, **meta)
    torch.testing.assert_close(output, before, rtol=0, atol=0)
    empty = output[:0]
    assert lora_segmented_delta_(empty, x[:0], a, b, **meta) is empty
    x1, factors = factor_case(0, 64, 16, (32,), torch.bfloat16)
    out = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16)
    assert lora_delta_factors_(out, x1, factors, scaling=1.0, output_sizes=(32,)) is out


@pytest.mark.parametrize("rows", [(64, 33, 16, 4), (4, 16, 33, 64)])
def test_graph_workspace_shared_replay_bitwise(rows):
    dtype, k, rank, sizes = torch.bfloat16, 2048, 64, (512, 128)
    cases = []
    for m in rows:
        x, factors = factor_case(m, k, rank, sizes, dtype, seed=m)
        base = torch.randn(m, sum(sizes), device="cuda", dtype=dtype) * 0.005
        cases.append((x, factors, base, base.clone()))
    calls = [
        (lambda c=c: lora_delta_factors_(c[3], c[0], c[1], scaling=0.5, output_sizes=sizes))
        for c in cases
    ]
    eager = []
    for call, c in zip(calls, cases, strict=True):
        call()
        eager.append(c[3].clone())
    with GraphWorkspace(max(rows)) as workspace:
        for call in calls:  # warm every call shape inside the workspace, then capture
            call()
        graphs = [capture(call) for call in calls]
        assert workspace.nbytes > 0
        for _ in range(2):
            for graph, c, expected in zip(graphs, cases, eager, strict=True):
                c[3].copy_(c[2])
                graph.replay()
                torch.testing.assert_close(c[3], expected, rtol=0, atol=0)
    del graphs
    workspace.close()


@pytest.mark.parametrize("dtype,bias_on", [(torch.bfloat16, False), (torch.float16, True)])
def test_segmented_projection(dtype, bias_on):
    _, x, a, b, meta, _ = make_case(4, dtype)
    meta.update(lora_segment_metadata([3, 2, 0, -1], [1] * 4, device="cuda"))
    weight = torch.randn(sum(SIZES), x.shape[1], device="cuda", dtype=dtype) * 0.01
    bias = torch.randn(sum(SIZES), device="cuda", dtype=dtype) * 0.01 if bias_on else None
    if dtype == torch.bfloat16:
        from ctikernels.linear import Linear

        before = Linear()(x, weight)
    else:
        before = F.linear(x, weight, bias)
    output = lora_segmented_projection(x, weight, bias, a, b, **meta)
    check(output, before, oracle(x, a, b, meta))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 4, 37])
def test_lora_projection_plan_uses_triton(dtype, m):
    k, rank, sizes = 1024, 16, (1024, 256, 256)
    x, factors = factor_case(m, k, rank, sizes, dtype)
    weight = torch.randn(sum(sizes), k, device="cuda", dtype=dtype) * 0.02
    plan = try_prepare_lora_inference_plan(factors, input_size=k, output_sizes=sizes, scaling=0.5)
    assert plan is not None
    with torch.no_grad():
        out = lora_projection(x, weight, None, factors, scaling=0.5, inference_plan=plan)
    assert plan.provider == "triton-lora-v1"
    base = plan.base(x, weight)
    check(out, base, factor_oracle(x, factors, 0.5))
    reference = F.linear(x.float(), weight.float()).double().cpu() + factor_oracle(x, factors, 0.5)
    error = (out.double().cpu() - reference).norm() / reference.norm()
    assert error < 1e-2


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [1, 4, 33])
def test_tensor_parallel_shards(dtype, m):
    """Two simulated ranks. Column-parallel (A replicated, B and output split on N) equals
    TP=1 bitwise; row-parallel K shards summed like the base all-reduce match the oracle."""
    k, rank, sizes, tp = 1024, 16, (512, 256), 2
    x, factors = factor_case(m, k, rank, sizes, dtype)
    full = torch.zeros(m, sum(sizes), device="cuda", dtype=dtype)
    lora_delta_factors_(full, x, factors, scaling=0.5, output_sizes=sizes)
    for r in range(tp):
        shard = tuple(n // tp for n in sizes)
        rows = [slice(r * n, (r + 1) * n) for n in shard]
        local = tuple(
            f if i % 2 == 0 else f[rows[i // 2]].contiguous() for i, f in enumerate(factors)
        )
        out = torch.zeros(m, sum(shard), device="cuda", dtype=dtype)
        lora_delta_factors_(out, x, local, scaling=0.5, output_sizes=shard)
        offsets = [sum(sizes[:s]) for s in range(len(sizes))]
        columns = torch.cat(
            [torch.arange(o + w.start, o + w.stop) for o, w in zip(offsets, rows, strict=True)]
        )
        torch.testing.assert_close(out, full[:, columns.cuda()], rtol=0, atol=0)
    total = torch.zeros(m, sum(sizes), device="cuda", dtype=torch.float32)
    for r in range(tp):
        ks = slice(r * k // tp, (r + 1) * k // tp)
        local = tuple(f[:, ks].contiguous() if i % 2 == 0 else f for i, f in enumerate(factors))
        partial = torch.zeros(m, sum(sizes), device="cuda", dtype=dtype)
        lora_delta_factors_(
            partial,
            x[:, ks].contiguous(),
            local,
            scaling=0.5,
            output_sizes=sizes,
            row_parallel=True,
        )
        total += partial.float()  # the caller's base all-reduce
    zero = torch.zeros_like(full)
    check(total.to(dtype), zero, factor_oracle(x, factors, 0.5))
