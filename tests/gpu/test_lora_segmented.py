# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Segmented LoRA qualification (cc >= 8.9). Reuse shapes/buckets to bound compilation cost.

Run: pytest -q tests/gpu/test_lora_segmented.py
Use pytest --durations to audit cold compilation.
"""

import pytest
import torch
import torch.nn.functional as F

from ctikernels import GraphWorkspace
from ctikernels.linear import _lora_segmented as impl
from ctikernels.linear import (
    lora_delta_factors_,
    lora_segment_metadata,
    lora_segmented_delta_,
    lora_segmented_projection,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SIZES = (96, 32, 16)


@pytest.fixture(scope="module", autouse=True)
def supported_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires compute capability 8.9 or newer")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    yield
    torch.set_num_threads(previous_threads)


def make_case(m, dtype, sizes=SIZES, k=256, pool_rank=512, every_token=False):
    torch.manual_seed(123 + m)
    adapters = m if every_token else 4
    indptr = list(range(m + 1)) if every_token else [0, m // 4, m // 3, m // 2, m]
    # Tiny batches have empty segments; the final live segment still uses an adapter.
    ids = list(range(m)) if every_token else [0, -1, 2, 3]
    available_ranks = [r for r in (8, 16, 64, 128, 256, 512) if r <= pool_rank]
    rank_values = [available_ranks[(a + m) % len(available_ranks)] for a in range(adapters)]
    dtype_tensor = lambda shape: torch.randn(shape, device="cuda", dtype=dtype)
    x = dtype_tensor((m, k)) * 0.25
    a = dtype_tensor((adapters, len(sizes) * pool_rank, k)) * 0.1
    b = dtype_tensor((adapters, sum(sizes), pool_rank)) * 0.1
    storage = dtype_tensor((m, sum(sizes) + 13)) * 0.01
    output = storage[:, 3 : 3 + sum(sizes)]  # unaligned start and odd row stride
    meta = {
        "seg_indptr": torch.tensor(indptr, device="cuda", dtype=torch.int32),
        "weight_indices": torch.tensor(ids, device="cuda", dtype=torch.int32),
        "ranks": torch.tensor(rank_values, device="cuda", dtype=torch.int32),
        "scalings": torch.linspace(-1.0, 2.0, adapters, device="cuda", dtype=torch.float32),
        "output_sizes": sizes,
    }
    return output, x, a, b, meta, storage


def oracle(x, a, b, meta):
    # The oracle evaluates every token's two matmuls in FP64. Segment batching
    # only amortizes CPU dispatch; it does not change the per-token reference.
    xc, ac, bc = x.cpu(), a.cpu(), b.cpu()
    ptr, ids, ranks = (
        meta[key].cpu().tolist() for key in ("seg_indptr", "weight_indices", "ranks")
    )
    scales = meta["scalings"].cpu().tolist()
    permutation = meta.get("permutation")
    permutation = torch.arange(x.shape[0]) if permutation is None else permutation.cpu().long()
    result = torch.zeros(x.shape[0], b.shape[1], dtype=torch.float64)
    for seg, adapter in enumerate(ids):
        if adapter < 0 or ptr[seg] == ptr[seg + 1]:
            continue
        lo, hi, rank = ptr[seg], ptr[seg + 1], ranks[adapter]
        rows = permutation[lo:hi]
        stride = rank if meta.get("a_slice_stride", "rank") == "rank" else b.shape[2]
        offset = 0
        for s, n in enumerate(meta["output_sizes"]):
            aa = ac[adapter, s * stride : s * stride + rank].double()
            bb = bc[adapter, offset : offset + n, :rank].double()
            result[rows, offset : offset + n] = scales[adapter] * (xc[rows].double() @ aa.T) @ bb.T
            offset += n
    return result


def check(output, before, delta):
    actual, before = output.cpu().double(), before.cpu().double()
    expected = before + delta
    norm = lambda v: torch.linalg.vector_norm(v).item()
    delta_error = norm((actual - before) - delta) / max(norm(delta), 1e-30)
    full_error = norm(actual - expected) / max(norm(expected), 1e-30)
    assert delta_error <= 2e-2, (delta_error, full_error)
    assert full_error <= 4e-3, (delta_error, full_error)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 33, 64, 200, 2048])
def test_ragged_mixed_ranks(m, dtype):
    output, x, a, b, meta, storage = make_case(m, dtype, every_token=m == 64)
    before, sentinel = output.clone(), storage.clone()
    expected = oracle(x, a, b, meta)
    assert lora_segmented_delta_(output, x, a, b, **meta) is output
    check(output, before, expected)
    torch.testing.assert_close(storage[:, :3], sentinel[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, -10:], sentinel[:, -10:], rtol=0, atol=0)
    # An unassigned token must remain bit-identical, independent of stale scratch.
    ptr, ids = meta["seg_indptr"].cpu().tolist(), meta["weight_indices"].cpu().tolist()
    for s, adapter in enumerate(ids):
        if adapter < 0:
            torch.testing.assert_close(
                output[ptr[s] : ptr[s + 1]], before[ptr[s] : ptr[s + 1]], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "sizes,k",
    [
        ((128,), 256),
        ((128, 128), 256),
        ((2048, 2048, 4096, 4096, 32, 32), 64),
        ((17, 9), 37),
    ],
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_graph_metadata_and_weight_mutation(dtype):
    output, x, a, b, meta, _ = make_case(33, dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lora_segmented_delta_(output, x, a, b, **meta)
    torch.cuda.current_stream().wait_stream(stream)
    before = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lora_segmented_delta_(output, x, a, b, **meta)
    compiled_count = len(impl._COMPILED)
    for ptr, ids, ranks, scales in [
        ([0, 1, 17, 19, 33], [3, 0, -1, 1], [8, 512, 16, 256], [2.0, -1.0, 0.5, 1.0]),
        ([0, 0, 0, 33, 33], [-1, 3, 2, -1], [0, 16, 128, 512], [0.0, 1.0, -0.75, 3.0]),
        ([0, 8, 16, 24, 33], [-1, -1, -1, -1], [512, 512, 512, 512], [1.0, 1.0, 1.0, 1.0]),
    ]:
        for name, values in zip(
            ("seg_indptr", "weight_indices", "ranks", "scalings"),
            (ptr, ids, ranks, scales),
            strict=True,
        ):
            meta[name].copy_(torch.tensor(values, device="cuda", dtype=meta[name].dtype))
        a.mul_(0.9)
        b.add_(0.001)
        output.copy_(before)
        expected = oracle(x, a, b, meta)
        graph.replay()
        check(output, before, expected)
        assert len(impl._COMPILED) == compiled_count

    # Capture executes Python once. A warmed variant never adds a compiled variant.
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
        lora_segmented_delta_(output, x, a, b, **meta)
    assert len(impl._COMPILED) == compiled_count


def test_current_stream_and_independent_graph_scratch():
    cases = [make_case(16, torch.bfloat16), make_case(33, torch.bfloat16)]
    for case in cases:
        case[4]["config"] = (64, 16, 128)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, before, expected = [], [], []
    for (output, x, a, b, meta, _), stream in zip(cases, streams, strict=True):
        before.append(output.clone())
        expected.append(oracle(x, a, b, meta))
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            lora_segmented_delta_(output, x, a, b, **meta)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            lora_segmented_delta_(output, x, a, b, **meta)
        graphs.append(graph)
    # Producer, graph replay and snapshot are all ordered on nondefault streams.
    snapshots = []
    for i, ((output, x, _a, _b, _meta, _), stream) in enumerate(zip(cases, streams, strict=True)):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            output.copy_(before[i])
            x.mul_(2)
            graphs[i].replay()
            snapshots.append(output.clone())
    for i, stream in enumerate(streams):
        torch.cuda.current_stream().wait_stream(stream)
        check(snapshots[i], before[i], expected[i] * 2)


@pytest.mark.parametrize("dtype,bias_on", [(torch.bfloat16, False), (torch.float16, True)])
def test_projection(dtype, bias_on):
    _, x, a, b, meta, _ = make_case(4, dtype)
    meta.update(lora_segment_metadata([3, 2, 0, -1], [1] * 4, device="cuda"))
    meta["a_slice_stride"] = "rank" if dtype == torch.bfloat16 else "max_rank"
    weight = torch.randn(sum(SIZES), x.shape[1], device="cuda", dtype=dtype) * 0.01
    bias = torch.randn(sum(SIZES), device="cuda", dtype=dtype) * 0.01 if bias_on else None
    if dtype == torch.bfloat16:
        from ctikernels.linear import Linear

        before = Linear()(x, weight)
    else:
        before = F.linear(x, weight, bias)
    expected = oracle(x, a, b, meta)
    output = lora_segmented_projection(x, weight, bias, a, b, **meta)
    assert output.data_ptr() != before.data_ptr()
    check(output, before, expected)


def test_zero_rank_empty_inputs_and_metadata():
    output, x, a, b, meta, _ = make_case(8, torch.bfloat16)
    before = output.clone()
    meta["ranks"].zero_()
    lora_segmented_delta_(output, x, a, b, **meta)
    torch.testing.assert_close(output, before, rtol=0, atol=0)
    meta["weight_indices"] = torch.empty(0, device="cuda", dtype=torch.int32)
    meta["seg_indptr"] = torch.zeros(1, device="cuda", dtype=torch.int32)
    assert lora_segmented_delta_(output, x, a, b, **meta) is output
    empty = output[:0]
    assert lora_segmented_delta_(empty, x[:0], a, b, **meta) is empty


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("stride", ["rank", "max_rank"])
@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_grouped_permutation_and_pool_layout(dtype, stride, chunk):
    output, x, a, b, meta, _ = make_case(133, dtype)
    ids = ([2, 0, -1, 3] * 34)[:133]
    meta.update(
        lora_segment_metadata(ids, [1] * 133, device="cuda", chunk_tokens=chunk),
        a_slice_stride=stride,
    )
    before = output.clone()
    expected = oracle(x, a, b, meta)
    lora_segmented_delta_(output, x, a, b, **meta)
    check(output, before, expected)
    skipped = torch.tensor([i for i, adapter in enumerate(ids) if adapter < 0], device="cuda")
    torch.testing.assert_close(output[skipped], before[skipped], rtol=0, atol=0)
    # The permutation is device data, including on graph replay.
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lora_segmented_delta_(output, x, a, b, **meta)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream):
        lora_segmented_delta_(output, x, a, b, **meta)
    meta["permutation"].copy_(meta["permutation"].flip(0))
    expected = oracle(x, a, b, meta)
    output.copy_(before)
    graph.replay()
    check(output, before, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("chunk", [None, 16, 32, 64])
def test_separate_factors_eager_and_graph_updates(dtype, tail, chunk):
    # Reuse pool compile variants, including odd B stride with separate factors.
    m, k, rank, sizes = (4, 37, 19, (17, 9)) if tail else (33, 256, 512, SIZES)
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
    factors = tuple(
        t
        for n in sizes
        for t in (
            torch.randn(rank, k, device="cuda", dtype=dtype) * 0.1,
            torch.randn(n, rank, device="cuda", dtype=dtype) * 0.1,
        )
    )
    scaling = torch.tensor(1.5, device="cuda")
    output = torch.randn(m, sum(sizes), device="cuda", dtype=dtype) * 0.005
    before = output.clone()

    def expected():
        return torch.cat(
            [
                scaling.cpu().double()
                * (x.cpu().double() @ a.cpu().double().T)
                @ b.cpu().double().T
                for a, b in zip(factors[::2], factors[1::2], strict=True)
            ],
            dim=1,
        )

    def call():
        return lora_delta_factors_(
            output,
            x,
            factors,
            scaling=scaling,
            output_sizes=sizes,
            config=(None, None, None, chunk),
        )

    call()
    check(output, before, expected())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    count = len(impl._COMPILED)
    addresses = [factor.data_ptr() for factor in factors]
    for scale in (-0.75, 2.0):
        for factor in factors:
            factor.mul_(0.9)
        scaling.fill_(scale)
        output.copy_(before)
        graph.replay()
        check(output, before, expected())
    assert addresses == [factor.data_ptr() for factor in factors]
    assert len(impl._COMPILED) == count


@pytest.mark.parametrize("lengths", [[1] * 64, [0, 2, 17, 3, 0, 19], []])
@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_device_metadata_matches_host(lengths, chunk):
    ids = [i % 3 - 1 for i in range(len(lengths))]
    host = lora_segment_metadata(ids, lengths, chunk_tokens=chunk)
    device = lora_segment_metadata(
        torch.tensor(ids, device="cuda", dtype=torch.int32),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        total_tokens=sum(lengths),
        chunk_tokens=chunk,
    )
    live = host["weight_indices"].numel()
    torch.testing.assert_close(device["permutation"].cpu(), host["permutation"])
    torch.testing.assert_close(device["seg_indptr"][: live + 1].cpu(), host["seg_indptr"])
    torch.testing.assert_close(device["weight_indices"][:live].cpu(), host["weight_indices"])
    assert device["seg_indptr"][live:].eq(sum(lengths)).all()
    for actual, expected in zip(device["chunks"][:3], host["chunks"][:3], strict=True):
        torch.testing.assert_close(actual[:live].cpu(), expected[:live])
    assert device["chunks"].chunk_adapter[live:].eq(-1).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("device_metadata", [False, True])
def test_chunk_table_many_adapters_and_padding(dtype, device_metadata):
    output, x, a, b, meta, storage = make_case(64, dtype, pool_rank=16, every_token=True)
    ids, lengths = list(reversed(range(64))), [1] * 64
    if device_metadata:
        ids = torch.tensor(ids, device="cuda", dtype=torch.int32)
        lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    meta.update(lora_segment_metadata(ids, lengths, device="cuda", total_tokens=64))
    table = meta["chunks"]
    # Poison inactive intervals: adapter=-1 alone must prevent all dereferences.
    table.chunk_start[64:] = 2**30
    table.chunk_end[64:] = 2**30 + 16
    assert table.chunk_adapter[64:].eq(-1).all()
    before, sentinel = output.clone(), storage.clone()
    expected = oracle(x, a, b, meta)
    lora_segmented_delta_(output, x, a, b, **meta)
    check(output, before, expected)
    table_result = output.clone()
    output.copy_(before)
    scan_meta = {k: v for k, v in meta.items() if k != "chunks"}
    lora_segmented_delta_(output, x, a, b, **scan_meta)
    torch.testing.assert_close(output, table_result, rtol=0, atol=0)
    torch.testing.assert_close(storage[:, :3], sentinel[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, -10:], sentinel[:, -10:], rtol=0, atol=0)


@pytest.mark.parametrize("k,x_value", [(256, 128.0), (4096, 8.0)])  # direct and split-K paths
def test_fp16_shrink_scales_before_rounding(k, x_value):
    # x @ A^T is 65536 (FP16 inf); scaling 1/32 brings it to 2048 before rounding.
    x = torch.full((4, k), x_value, device="cuda", dtype=torch.float16)
    a = torch.full((16, k), 2.0, device="cuda", dtype=torch.float16)
    b = torch.full((64, 16), 1 / 16, device="cuda", dtype=torch.float16)
    output = torch.zeros(4, 64, device="cuda", dtype=torch.float16)
    lora_delta_factors_(output, x, (a, b), scaling=1 / 32, output_sizes=(64,))
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.full_like(output, 2048.0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,q", [(64, 16), (128, 8), (256, 4), (512, 16)])
def test_verify_shaped_capture_changed_groups_bitwise(dtype, m, q):
    # Speculative verify: B requests of Q rows each, mixed adapters and -1 (no
    # adapter) requests, captured at bucket M and replayed with new groups.
    torch.manual_seed(m + q)
    k, adapters, pool_rank = 256, 4, 128
    requests = m // q
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
    a = torch.randn(adapters, len(SIZES) * pool_rank, k, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(adapters, sum(SIZES), pool_rank, device="cuda", dtype=dtype) * 0.1
    base = torch.randn(m, sum(SIZES), device="cuda", dtype=dtype) * 0.01
    ranks = torch.tensor([16, 64, 128, 8], device="cuda", dtype=torch.int32)
    scalings = torch.tensor([0.5, -1.0, 2.0, 0.25], device="cuda", dtype=torch.float32)
    lengths = torch.full((requests,), q, device="cuda", dtype=torch.int32)

    def groups(seed):
        generator = torch.Generator().manual_seed(seed)
        ids = torch.randint(-1, adapters, (requests,), generator=generator, dtype=torch.int32)
        ids[0] = -1  # always include padding requests
        return lora_segment_metadata(ids.cuda(), lengths, total_tokens=m)

    meta = groups(0)
    tensors = [meta["seg_indptr"], meta["weight_indices"], meta["permutation"], *meta["chunks"][:3]]

    def call(out):
        return lora_segmented_delta_(
            out,
            x,
            a,
            b,
            seg_indptr=meta["seg_indptr"],
            weight_indices=meta["weight_indices"],
            ranks=ranks,
            scalings=scalings,
            output_sizes=SIZES,
            permutation=meta["permutation"],
            chunks=meta["chunks"],
        )

    output = base.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(output)
    for seed in (1, 2, 3):
        new = groups(seed)
        assert new["chunks"].count == meta["chunks"].count
        for target, source in zip(
            tensors,
            [new["seg_indptr"], new["weight_indices"], new["permutation"], *new["chunks"][:3]],
            strict=True,
        ):
            target.copy_(source)
        output.copy_(base)
        graph.replay()
        replayed = output.clone()
        eager = call(base.clone())
        assert torch.equal(replayed, eager)
        host = {key: meta[key] for key in ("seg_indptr", "weight_indices", "permutation")}
        expected = oracle(
            x, a, b, {**host, "ranks": ranks, "scalings": scalings, "output_sizes": SIZES}
        )
        check(replayed, base, expected)


def test_graph_workspace_rejects_rows_over_max():
    output, x, a, b, meta, _ = make_case(64, torch.bfloat16, k=4096)
    with GraphWorkspace(max_rows=16), pytest.raises(RuntimeError, match="max_rows=16"):
        lora_segmented_delta_(output, x, a, b, **meta)
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            lora_segmented_delta_(output, x, a, b, **meta)
