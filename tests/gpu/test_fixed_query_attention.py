# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Exact changing KV state inside fixed-query graphs, against CPU FP32 math."""

from dataclasses import dataclass

import pytest
import torch

from ctikernels.attention._flashinfer import FIXED_QUERY_MIN_CHUNK
from ctikernels.attention.inference import FlashInferFixedQueryAttention

from ._attention import paged_metadata, reference_attention
from ._exhaustive import ci_cases
from ._fp8_precision import staged_precision_case


@pytest.mark.parametrize("query_length", [1, 4, 16])
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 2, 256), (8, 1, 256)])
@pytest.mark.parametrize("page", [16, 40])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fixed_query_fp8_staged_precision(query_length, heads, page, kv_dtype):
    staged_precision_case(heads=heads, page=page, kv_dtype=kv_dtype, query_length=query_length)


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    query: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page: int
    max_kv: int
    dtype: torch.dtype
    kv_lengths: tuple[int, ...] = ()


SHAPES = (
    AttentionShape(1, 2, 8, 2, 64, 1, 2048, torch.float16),
    AttentionShape(2, 16, 32, 8, 128, 1, 8192, torch.bfloat16),
    AttentionShape(4, 16, 32, 8, 128, 16, 8192, torch.float16),
    AttentionShape(8, 32, 8, 2, 256, 16, 1024, torch.bfloat16),
    AttentionShape(32, 16, 8, 2, 64, 1, 512, torch.bfloat16),
    AttentionShape(64, 32, 32, 8, 128, 1, 128, torch.bfloat16),
    AttentionShape(3, 7, 8, 2, 64, 1, 513, torch.float16),
    AttentionShape(1, 32, 32, 8, 128, 1, 16384, torch.bfloat16),
    AttentionShape(
        2,
        16,
        8,
        8,
        128,
        1,
        4097,
        torch.bfloat16,
        (2047, 2048, 2049, 4095, 4096, 4097),
    ),
    AttentionShape(
        3,
        16,
        16,
        8,
        128,
        16,
        4097,
        torch.bfloat16,
        (16, 17, 4080, 4081, 4096, 4097),
    ),
    AttentionShape(
        4,
        32,
        64,
        8,
        64,
        1,
        8193,
        torch.float16,
        (32, 33, 4095, 4096, 4097, 8191, 8192, 8193),
    ),
    AttentionShape(
        1,
        16,
        32,
        8,
        128,
        1,
        40960,
        torch.bfloat16,
        (16, 20481, 40959, 40960),
    ),
    AttentionShape(
        4,
        16,
        32,
        8,
        128,
        1,
        32768,
        torch.bfloat16,
        (32768, 16, 16384, 32767),
    ),
    AttentionShape(1, 4, 32, 8, 128, 16, 1024, torch.bfloat16),
)


class FixedQueryCase:
    """Own GPU buffers and graph; keep at most two pinned input sets in flight."""

    def __init__(self, shape: AttentionShape, causal: bool, min_chunk_tokens: int | None = None):
        self.shape, self.causal = shape, causal
        self.generator = torch.Generator().manual_seed(7163 + shape.batch)
        self.page_capacity = shape.batch * ((shape.max_kv + shape.page - 1) // shape.page)
        self.base_cache = tuple(
            torch.randn(
                self.page_capacity + 1,
                shape.page,
                shape.kv_heads,
                shape.head_dim,
                dtype=shape.dtype,
                generator=self.generator,
            )
            for _ in range(2)
        )
        self.cache = tuple(value.cuda() for value in self.base_cache)
        self.query = torch.empty(
            shape.batch * shape.query,
            shape.query_heads,
            shape.head_dim,
            dtype=shape.dtype,
            device="cuda",
        )
        self.storage = torch.full(
            (shape.batch * shape.query + 2, shape.query_heads, shape.head_dim),
            torch.nan,
            dtype=shape.dtype,
            device="cuda",
        )
        self.output = self.storage[1:-1]
        self.metadata = {
            "cu_seqlens_q": torch.empty(shape.batch + 1, dtype=torch.int32, device="cuda"),
            "cu_seqlens_k": torch.empty(shape.batch + 1, dtype=torch.int32, device="cuda"),
            "indices": torch.zeros(self.page_capacity, dtype=torch.int32, device="cuda"),
            "last_page_len": torch.empty(shape.batch, dtype=torch.int32, device="cuda"),
        }
        self.workspace = tuple(
            torch.zeros(size, dtype=torch.uint8, device="cuda")
            for size in FlashInferFixedQueryAttention.workspace_size(
                batch_size=shape.batch,
                query_length=shape.query,
                num_qo_heads=shape.query_heads,
                num_kv_heads=shape.kv_heads,
                head_dim=shape.head_dim,
                device=self.query.device,
            )
        )
        self.operation = FlashInferFixedQueryAttention(
            *self.workspace,
            query_indptr=self.metadata["cu_seqlens_q"],
            kv_indptr=self.metadata["cu_seqlens_k"],
            kv_indices=self.metadata["indices"],
            last_page_lengths=self.metadata["last_page_len"],
            query_length=shape.query,
            num_qo_heads=shape.query_heads,
            num_kv_heads=shape.kv_heads,
            head_dim=shape.head_dim,
            page_size=shape.page,
            dtype=shape.dtype,
            causal=causal,
            **({} if min_chunk_tokens is None else {"min_chunk_tokens": min_chunk_tokens}),
        )

    def inputs(self, round_index):
        shape = self.shape
        cycle = shape.kv_lengths or (
            shape.query,
            min(127, shape.max_kv),
            min(129, shape.max_kv),
            shape.max_kv,
        )
        lengths = tuple(cycle[(round_index + 3 * row) % len(cycle)] for row in range(shape.batch))
        metadata = paged_metadata((shape.query,) * shape.batch, lengths, shape.page)
        active = int(metadata["cu_seqlens_k"][-1])
        selected = torch.randperm(self.page_capacity, generator=self.generator)[:active]
        metadata["indices"] = torch.full(
            (self.page_capacity,), self.page_capacity, dtype=torch.int32, pin_memory=True
        )
        metadata["indices"][:active] = selected
        query = torch.randn(
            self.query.shape, dtype=shape.dtype, generator=self.generator
        ).pin_memory()
        cache = tuple(torch.full_like(value, torch.nan).pin_memory() for value in self.base_cache)
        for dst, src in zip(cache, self.base_cache, strict=True):
            dst[selected] = src[selected] + round_index * 0.015625
            for row in range(shape.batch):
                last_page = selected[int(metadata["cu_seqlens_k"][row + 1]) - 1]
                dst[last_page, int(metadata["last_page_len"][row]) :] = torch.nan
        expected = reference_attention(
            query,
            cache,
            (shape.query,) * shape.batch,
            lengths,
            metadata["indices"],
            causal=self.causal,
        )
        return query, cache, metadata, expected

    def load(self, inputs):
        query, cache, metadata, _ = inputs
        self.query.copy_(query, non_blocking=True)
        for dst, src in zip(self.cache, cache, strict=True):
            dst.copy_(src, non_blocking=True)
        for name, dst in self.metadata.items():
            dst.copy_(metadata[name], non_blocking=True)
        self.output.fill_(torch.nan)

    def forward(self):
        self.operation.prepare()
        assert (
            self.operation.run(q=self.query, paged_kv_cache=self.cache, out=self.output)
            is self.output
        )


@pytest.mark.parametrize(
    "shape",
    # Core: B4/Q16 FP16 and B1/Q4 BF16, both masks and changing lengths.
    ci_cases(SHAPES, core=[(SHAPES[2],), (SHAPES[-1],)]),
    ids=lambda s: (
        f"b{s.batch}-q{s.query}-g{s.query_heads // s.kv_heads}-d{s.head_dim}-p{s.page}-kv{s.max_kv}-{s.dtype}"
    ),
)
@pytest.mark.parametrize("causal", [False, True])
def test_fixed_query_split_kv_graph_with_changing_lengths(shape, causal):
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        case = FixedQueryCase(shape, causal)
        warmup = case.inputs(0)
        case.load(warmup)
        case.forward()
        stream.synchronize()
        torch.testing.assert_close(case.output.cpu().float(), warmup[-1], atol=0.01, rtol=0.01)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            case.forward()
        buffers = (case.query, *case.cache, *case.metadata.values(), *case.workspace, case.output)
        pointers = tuple(value.data_ptr() for value in buffers)
        rounds = len(shape.kv_lengths) or 4
        for start in (*range(0, rounds, 2), 0):
            # Submit two changing-metadata replays without an intervening host fence.
            retained = []
            for round_index in (start, start + 1):
                inputs = case.inputs(round_index)
                case.load(inputs)
                graph.replay()
                retained.append((inputs, case.output.to("cpu", non_blocking=True)))
            stream.synchronize()
            for inputs, actual in retained:
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual.float(), inputs[-1], atol=0.01, rtol=0.01)
            query, cache, metadata, _ = retained[-1][0]
            for actual, expected in zip((case.query, *case.cache), (query, *cache), strict=True):
                assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
            for name, actual in case.metadata.items():
                torch.testing.assert_close(actual.cpu(), metadata[name], rtol=0, atol=0)
            assert torch.isnan(case.storage[[0, -1]]).all()
            assert tuple(value.data_ptr() for value in buffers) == pointers

            # Same shape, inputs and schedule: repeatability, NOT batch invariance.
            previous = case.output.clone()
            repeats = []
            for _ in range(2):
                case.output.fill_(torch.nan)
                graph.replay()
                repeats.append(case.output.clone())
            stream.synchronize()
            for repeated in repeats:
                assert torch.equal(previous.view(torch.int16), repeated.view(torch.int16))


@pytest.mark.parametrize("page_size", [1, 16])
def test_constructor_initializes_buffers_then_caller_supplies_actual_kv(page_size):
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    shape = AttentionShape(2, 7, 8, 2, 64, page_size, 128, torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        case = FixedQueryCase(shape, causal=True)
        pages = (shape.query + page_size - 1) // page_size
        expected = {
            "cu_seqlens_q": torch.arange(3, dtype=torch.int32) * shape.query,
            "cu_seqlens_k": torch.arange(3, dtype=torch.int32) * pages,
            "last_page_len": torch.full((2,), (shape.query - 1) % page_size + 1, dtype=torch.int32),
        }
        for name, values in expected.items():
            torch.testing.assert_close(case.metadata[name].cpu(), values, rtol=0, atol=0)
        pointers = tuple(value.data_ptr() for value in case.metadata.values())
        inputs = case.inputs(3)  # Both requests have KV128, much longer than dummy Q7.
        case.load(inputs)
        case.forward()
        stream.synchronize()
        torch.testing.assert_close(case.output.cpu().float(), inputs[-1], atol=0.01, rtol=0.01)
        for name, value in case.metadata.items():
            torch.testing.assert_close(value.cpu(), inputs[2][name], rtol=0, atol=0)
        assert tuple(value.data_ptr() for value in case.metadata.values()) == pointers


@pytest.mark.parametrize(
    "min_chunk_tokens,head_dim,query_heads,kv_heads,causal,batch,query",
    ci_cases(
        [16, 32, 64],
        [(128, 32, 8), (256, 16, 4)],
        [False, True],
        [1, 4],
        [4, 16],
        core=[(32, 128, 32, 8, True, 1, 4), (16, 256, 16, 4, False, 4, 16)],
    ),
)
def test_min_chunk_graph_short_long_short(
    min_chunk_tokens, head_dim, query_heads, kv_heads, causal, batch, query
):
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    # Three lengths make the fixture's stride of three uniform across the batch.
    # Every request crosses from the minimum-chunk regime to the CTA bound and back.
    max_ctas = 2 * torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    max_ctas //= kv_heads
    long_kv = max(4100, min_chunk_tokens * max_ctas // batch + 1)
    shape = AttentionShape(
        batch,
        query,
        query_heads,
        kv_heads,
        head_dim,
        1,
        long_kv,
        torch.bfloat16,
        (query + 1, 116, long_kv),
    )
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        case = FixedQueryCase(shape, causal, min_chunk_tokens=min_chunk_tokens)
        warmup = case.inputs(0)
        case.load(warmup)
        case.forward()
        stream.synchronize()
        torch.testing.assert_close(case.output.cpu().float(), warmup[-1], atol=0.01, rtol=0.01)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            case.forward()  # Includes prepare() reading the mutable device metadata.
        buffers = (case.query, *case.cache, *case.metadata.values(), *case.workspace, case.output)
        pointers = tuple(value.data_ptr() for value in buffers)
        for round_index in (0, 1, 2, 0):
            inputs = case.inputs(round_index)
            case.load(inputs)
            graph.replay()
            stream.synchronize()
            actual = case.output.cpu().float()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, inputs[-1], atol=0.01, rtol=0.01)
            assert torch.isnan(case.storage[[0, -1]]).all()
            assert tuple(value.data_ptr() for value in buffers) == pointers


@pytest.mark.parametrize("head_dim", [128, 256])
def test_automatic_short_request_chunk_switch(head_dim):
    capability = torch.cuda.get_device_capability()
    tiers = FIXED_QUERY_MIN_CHUNK.get((*capability, head_dim), ((128, 256),))
    lengths = tuple(length for _, bound in tiers for length in (bound, bound + 1))
    expected = tuple(
        next((minimum for minimum, bound in tiers if length <= bound), 128) for length in lengths
    )
    # One KV head keeps the CTA bound below the tuned minimum on qualified devices.
    shape = AttentionShape(1, 4, 4, 1, head_dim, 1, lengths[-1], torch.bfloat16, lengths)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        case = FixedQueryCase(shape, causal=True)
        offset = case.operation._layout.chunk_size_offset
        chunk = case.workspace[1][offset : offset + 4].view(torch.int32)
        case.load(case.inputs(0))
        case.forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            case.forward()
        # The same graph must select each tier from runtime lengths, then return
        # to the first tier. Read the actual device schedule outside capture.
        for turn in (*range(len(lengths)), 0):
            inputs = case.inputs(turn)
            case.load(inputs)
            graph.replay()
            stream.synchronize()
            assert chunk.item() == expected[turn]
            torch.testing.assert_close(case.output.cpu().float(), inputs[-1], atol=0.01, rtol=0.01)


@pytest.mark.parametrize(
    "qh,kh,dim,page,kv_dtype",
    ci_cases(
        [(32, 8, 128), (16, 4, 256)],
        [1, 16],
        [torch.float8_e4m3fn, torch.float8_e5m2],
        core=[(32, 8, 128, 1, torch.float8_e4m3fn), (16, 4, 256, 16, torch.float8_e4m3fn)],
    ),
)
def test_fixed_query_fp8_scales_eager_and_graph(qh, kh, dim, page, kv_dtype):
    generator = torch.Generator().manual_seed(20261016)
    cpu_query = torch.randn(32, qh, dim, dtype=torch.bfloat16, generator=generator) * 8
    cpu_cache = tuple(
        torch.randn(2 * (2048 // page) + 1, page, kh, dim, generator=generator).to(kv_dtype)
        for _ in range(2)
    )
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        query = cpu_query.cuda()
        cache = tuple(value.cuda() for value in cpu_cache)
        output = torch.empty_like(query)
        workspace = tuple(
            torch.zeros(n, dtype=torch.uint8, device="cuda")
            for n in FlashInferFixedQueryAttention.workspace_size(
                batch_size=2,
                query_length=16,
                num_qo_heads=qh,
                num_kv_heads=kh,
                head_dim=dim,
                device=query.device,
            )
        )
        buffers = {
            "query_indptr": torch.empty(3, dtype=torch.int32, device="cuda"),
            "kv_indptr": torch.empty(3, dtype=torch.int32, device="cuda"),
            "kv_indices": torch.empty(2048 // page, dtype=torch.int32, device="cuda"),
            "last_page_lengths": torch.empty(2, dtype=torch.int32, device="cuda"),
        }
        operation = FlashInferFixedQueryAttention(
            *workspace,
            **buffers,
            query_length=16,
            num_qo_heads=qh,
            num_kv_heads=kh,
            head_dim=dim,
            page_size=page,
            dtype=query.dtype,
            kv_dtype=kv_dtype,
            causal=True,
            sm_scale=0.07,
            logits_soft_cap=30.0,
        )
        graph = None
        for lengths in ((512, 1024), (1007, 33), (16, 17)):
            metadata = paged_metadata((16, 16), lengths, page)
            buffers["kv_indptr"].copy_(metadata["cu_seqlens_k"])
            buffers["kv_indices"][: metadata["indices"].numel()].copy_(metadata["indices"])
            buffers["last_page_lengths"].copy_(metadata["last_page_len"])
            operation.prepare()
            operation.run(q=query, paged_kv_cache=cache, out=output, k_scale=0.75, v_scale=1.25)
            stream.synchronize()
            expected = reference_attention(
                cpu_query,
                cpu_cache,
                (16, 16),
                lengths,
                metadata["indices"],
                k_scale=0.75,
                v_scale=1.25,
                sm_scale=0.07,
                logits_soft_cap=30.0,
            )
            torch.testing.assert_close(output.cpu().float(), expected, rtol=0.01, atol=0.01)
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    operation.prepare()
                    operation.run(
                        q=query, paged_kv_cache=cache, out=output, k_scale=0.75, v_scale=1.25
                    )
            output.fill_(torch.nan)
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output.cpu().float(), expected, rtol=0.01, atol=0.01)
