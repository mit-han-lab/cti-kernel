# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from math import sqrt
from unittest.mock import patch

import pytest
import torch

from ctikernels.attention import _decode_fp8
from ctikernels.attention import _split_tables as split_tables
from ctikernels.attention.inference import FlashInferPagedAttention, trtllm_decode_attention

from ._attention import paged_metadata as _metadata
from ._attention import reference_attention as _reference
from ._exhaustive import ci_cases
from ._fp8_precision import staged_precision_case
from ._paged_support import paged_support_case


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 2, 256), (8, 1, 256)])
@pytest.mark.parametrize("page", [16, 40])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_native_fp8_decode_staged_precision(graph, heads, page, kv_dtype):
    staged_precision_case(heads=heads, page=page, kv_dtype=kv_dtype, decode_graph=graph)


@pytest.mark.parametrize("graph", [False, True])
def test_cuda_cores_fp8_decode_staged_precision(graph):
    staged_precision_case(
        heads=(32, 8, 128),
        page=16,
        kv_dtype=torch.float8_e4m3fn,
        decode_graph=graph,
        use_tensor_cores=False,
    )


@pytest.mark.parametrize(
    "route,graph",
    [
        ("cuda_cores", False),
        ("cuda_cores", True),
        ("fa2", False),
        ("fa2", True),
        ("fixed_query", None),
    ],
)
@pytest.mark.parametrize("range_case", ["underflow", "overflow"])
def test_fp8_staged_range(route, graph, range_case):
    staged_precision_case(
        heads=(1, 1, 64),
        page=16,
        kv_dtype=torch.float8_e4m3fn,
        query_length=2 if route == "fixed_query" else 1,
        decode_graph=graph,
        use_tensor_cores=route != "cuda_cores",
        range_case=range_case,
    )


@pytest.mark.parametrize(
    "route,graph",
    [
        ("cuda_cores", False),
        ("cuda_cores", True),
        ("fa2", False),
        ("fa2", True),
        ("fixed_query", None),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 0.5, 2.0])
def test_fp8_staged_scale_threshold(route, graph, scale):
    staged_precision_case(
        heads=(1, 1, 64),
        page=16,
        kv_dtype=torch.float8_e4m3fn,
        query_length=2 if route == "fixed_query" else 1,
        decode_graph=graph,
        use_tensor_cores=route != "cuda_cores",
        range_case="threshold",
        range_scale=scale,
    )


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("count", [8, 1032])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("scale", [0.0, -0.31, 1.0, 1e30, float("inf"), float("nan")])
def test_scale_cast_ieee_tail_alignment_and_capture(offset, count, pdl, scale):
    from ctikernels.attention._scale_cast import FP16StagedOutput

    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        stage = FP16StagedOutput((1, 1, count), torch.device("cuda"))
        values = torch.tensor(
            [0, -0.0, 1.5, -2, 65504, float("inf"), -float("inf"), float("nan")],
            dtype=torch.float16,
            device="cuda",
        ).repeat(count // 8)
        stage.stage.view(-1).copy_(values)
        storage = torch.full((count + 16,), 17, dtype=torch.bfloat16, device="cuda")
        begin = 8 + offset
        output = storage[begin : begin + count]
        expected = (values.float() * scale).bfloat16()
        enabled = pdl and stage.supports_pdl
        stage._kernel(stage.stage.view(-1), output, scale, enabled)
        torch.testing.assert_close(output, expected, atol=0, rtol=0, equal_nan=True)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            stage._kernel(stage.stage.view(-1), output, scale, enabled)
        output.fill_(42)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0, equal_nan=True)
        assert (storage[:begin] == 17).all() and (storage[begin + count :] == 17).all()


def _inputs(
    query_lengths: tuple[int, ...],
    metadata: dict[str, torch.Tensor],
    page_size: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(sum(query_lengths), 8, head_dim, generator=generator, dtype=torch.bfloat16)
    cache_shape = (2 * metadata["indices"].numel() + 1, page_size, 2, head_dim)
    cache = tuple(
        torch.randn(cache_shape, generator=generator, dtype=torch.bfloat16) for _ in range(2)
    )
    return query, cache


def _plan(
    operation: FlashInferPagedAttention,
    metadata: dict[str, torch.Tensor],
    q: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor],
    *,
    causal: bool = True,
    disable_split_kv: bool = False,
) -> None:
    operation.plan(
        **metadata,
        num_qo_heads=q.shape[1],
        num_kv_heads=cache[0].shape[2],
        head_dim=q.shape[-1],
        page_size=cache[0].shape[1],
        pos_encoding_mode="NONE",
        dtype=q.dtype,
        causal=causal,
        disable_split_kv=disable_split_kv,
    )


def _fixed_query_round(base_cache, page_size: int, round_index: int):
    # Retain rounds from the qualified probe: grow/shrink and cross page boundaries.
    length_cycle = (16, 1024, 17, 33, 31, 512, 32, 65, 1023, 63, 257, 16)
    lengths = (length_cycle[round_index], length_cycle[(round_index + 3) % 12])
    metadata = _metadata((16, 16), lengths, page_size)
    active_pages = int(metadata["cu_seqlens_k"][-1])
    poison_page = base_cache[0].shape[0] - 1
    generator = torch.Generator().manual_seed(20260909 + 200 + page_size + 10000 + round_index)
    selected = torch.randperm(poison_page, generator=generator)[:active_pages]
    metadata["indices"] = torch.full(
        (2 * (1024 // page_size),), poison_page, dtype=torch.int32, pin_memory=True
    )
    metadata["indices"][:active_pages] = selected
    cache = tuple(torch.full_like(value, torch.nan).pin_memory() for value in base_cache)
    for destination, source in zip(cache, base_cache, strict=True):
        destination[selected] = source[selected]
        for row in range(2):
            final_page = selected[int(metadata["cu_seqlens_k"][row + 1]) - 1]
            destination[final_page, int(metadata["last_page_len"][row]) :] = torch.nan
    return lengths, metadata, cache


def _check_fixed_query_inputs(query, cache, cpu_query, cpu_cache, snapshot, *metadata):
    for actual, expected in zip((query, *cache), (cpu_query, *cpu_cache), strict=True):
        # Compare the BF16 bits, including poisoned NaNs, without weakening immutability.
        assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
    for values in metadata:
        for name, expected in snapshot.items():
            torch.testing.assert_close(values[name].cpu(), expected, rtol=0, atol=0)


def _fixed_query_graph_case(page_size: int, causal: bool) -> None:
    generator = torch.Generator().manual_seed(20260909 + 200 + page_size)
    cpu_query = torch.randn(32, 32, 128, generator=generator, dtype=torch.bfloat16)
    base_cache = tuple(
        torch.randn(
            2 * (1024 // page_size) + 17,
            page_size,
            8,
            128,
            generator=generator,
            dtype=torch.bfloat16,
        )
        for _ in range(2)
    )
    _, cpu_metadata, cpu_cache = _fixed_query_round(base_cache, page_size, 0)
    snapshot = {name: value.clone() for name, value in cpu_metadata.items()}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        query = cpu_query.cuda()
        cache = tuple(value.cuda() for value in cpu_cache)
        metadata = {name: value.cuda() for name, value in cpu_metadata.items()}
        staging = {name: value.clone() for name, value in metadata.items()}
        float_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        eager_float_workspace = torch.zeros_like(float_workspace)
        eager_int_workspace = torch.empty_like(int_workspace)
        storage = torch.full((34, 32, 128), torch.nan, dtype=query.dtype, device="cuda")
        eager_storage = torch.full_like(storage, torch.nan)
        output, eager_output = storage[1:-1], eager_storage[1:-1]
        operation = FlashInferPagedAttention(
            float_workspace, int_workspace, mode="prefill", backend="fa2", use_cuda_graph=False
        )
    borrowed = (
        query,
        *cache,
        *metadata.values(),
        *staging.values(),
        output,
        eager_output,
        float_workspace,
        int_workspace,
        eager_float_workspace,
        eager_int_workspace,
    )
    pointers = tuple(value.data_ptr() for value in borrowed)
    # Spy only on our public wrapper; no provider code or private state is patched.
    with patch.object(operation, "plan", wraps=operation.plan) as graph_plan:
        with torch.cuda.stream(stream), torch.inference_mode():
            _plan(operation, metadata, query, cache, causal=causal, disable_split_kv=True)
            assert operation.run(q=query, paged_kv_cache=cache, out=output) is output
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
            assert operation.run(q=query, paged_kv_cache=cache, out=output) is output
        stream.synchronize()
        _check_fixed_query_inputs(
            query, cache, cpu_query, cpu_cache, snapshot, cpu_metadata, metadata, staging
        )

        for round_index in (0, 1, 2, 11):
            lengths, cpu_metadata, cpu_cache = _fixed_query_round(
                base_cache, page_size, round_index
            )
            snapshot = {name: value.clone() for name, value in cpu_metadata.items()}
            expected = _reference(
                cpu_query, cpu_cache, (16, 16), lengths, cpu_metadata["indices"], causal=causal
            )
            assert torch.isfinite(expected).all()
            with torch.cuda.stream(stream), torch.inference_mode():
                for destination, source in zip(cache, cpu_cache, strict=True):
                    destination.copy_(source, non_blocking=True)
                for name, value in staging.items():
                    value.copy_(cpu_metadata[name], non_blocking=True)
                eager = FlashInferPagedAttention(
                    eager_float_workspace,
                    eager_int_workspace,
                    mode="prefill",
                    backend="fa2",
                    use_cuda_graph=False,
                )
                eager_metadata = {
                    **cpu_metadata,
                    "indices": staging["indices"][: int(cpu_metadata["cu_seqlens_k"][-1])],
                }
                _plan(eager, eager_metadata, query, cache, causal=causal, disable_split_kv=True)
                eager_output.fill_(torch.nan)
                assert eager.run(q=query, paged_kv_cache=cache, out=eager_output) is eager_output
            stream.synchronize()
            eager_cpu = eager_output.cpu()
            assert torch.isfinite(eager_cpu).all()
            torch.testing.assert_close(eager_cpu.float(), expected, rtol=0.01, atol=0.01)
            assert torch.isnan(eager_storage[[0, -1]]).all().item()
            _check_fixed_query_inputs(
                query, cache, cpu_query, cpu_cache, snapshot, cpu_metadata, staging
            )
            for _ in range(2):
                with torch.cuda.stream(stream), torch.inference_mode():
                    for name, value in metadata.items():
                        value.copy_(staging[name])
                    output.fill_(torch.nan)
                    graph.replay()
                stream.synchronize()
                actual = output.cpu()
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.01)
                torch.testing.assert_close(actual, eager_cpu, rtol=0, atol=0)
                assert torch.isnan(storage[[0, -1]]).all().item()
                assert tuple(value.data_ptr() for value in borrowed) == pointers
                _check_fixed_query_inputs(
                    query, cache, cpu_query, cpu_cache, snapshot, cpu_metadata, metadata, staging
                )
            del eager  # Previous native plan/run completed before its owner expires.
        assert graph_plan.call_count == 1


@pytest.mark.parametrize("page_size", [1, 16])
def test_fixed_query_prefill_graph_reuses_one_public_plan_with_changed_kv(page_size: int) -> None:
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    # Keep both modes sequential in one process: isolated cases missed the original
    # provider native-graph-mode padding/lifetime failure. That mode is not used here.
    for causal in (False, True):
        _fixed_query_graph_case(page_size, causal)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize(("page_size", "head_dim"), [(1, 64), (16, 128)])
def test_paged_attention_matches_causal_gqa_reference_and_caller_output(
    mode: str, page_size: int, head_dim: int
) -> None:
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    query_lengths = (1, 7, 19) if mode == "prefill" else (1, 1, 1)
    kv_lengths = (1, 32, 49)
    cpu_metadata = _metadata(query_lengths, kv_lengths, page_size)
    original_metadata = {name: tensor.clone() for name, tensor in cpu_metadata.items()}
    cpu_query, cpu_cache = _inputs(query_lengths, cpu_metadata, page_size, head_dim, 42)
    expected = _reference(cpu_query, cpu_cache, query_lengths, kv_lengths, cpu_metadata["indices"])
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        query = cpu_query.cuda()
        cache = tuple(tensor.cuda() for tensor in cpu_cache)
        metadata = {**cpu_metadata, "indices": cpu_metadata["indices"].cuda()}
        # Native FlashInfer requires zero-initialized float scratch on first use.
        float_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        # Explicit FA2 is supported on the qualification GPU; this does not qualify
        # the separate native FlashAttention or TRTLLM public operations.
        operation = FlashInferPagedAttention(
            float_workspace, int_workspace, mode=mode, backend="fa2", use_tensor_cores=True
        )
        _plan(operation, metadata, query, cache)
        allocated_output = operation.run(q=query, paged_kv_cache=cache)
        # An offset view checks the caller's actual storage, including its guards.
        output_storage = torch.full(
            (query.shape[0] + 2, *query.shape[1:]),
            torch.nan,
            dtype=query.dtype,
            device=query.device,
        )
        output = output_storage[1:-1]
        result = operation.run(q=query, paged_kv_cache=cache, out=output)
        completion = torch.cuda.Event()
        completion.record(stream)

    # Retain CPU staging inputs, CUDA metadata and both workspaces until the
    # asynchronous plan and all runs finish; there is no synchronization in run().
    completion.synchronize()
    assert result is output
    assert result.data_ptr() == output_storage[1].data_ptr()
    for actual in (allocated_output, output):
        assert actual.shape == query.shape and actual.dtype == torch.bfloat16
        # BF16 output rounding and tiled native reductions versus CPU FP32 math.
        torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.01, atol=0.01)
    assert torch.isnan(output_storage[[0, -1]]).all().item()
    for actual, original in zip((query, *cache), (cpu_query, *cpu_cache), strict=True):
        torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)
    for name, tensor in metadata.items():
        torch.testing.assert_close(tensor.cpu(), original_metadata[name], rtol=0, atol=0)


def test_paged_decode_graph_replays_changed_inputs_and_replans_borrowed_storage() -> None:
    assert torch.cuda.is_available(), "GPU suite requires CUDA"
    query_lengths, kv_lengths = (1, 1, 1), (1, 32, 49)
    page_size, head_dim = 16, 128
    cpu_metadata = _metadata(query_lengths, kv_lengths, page_size)
    cpu_query, cpu_cache = _inputs(query_lengths, cpu_metadata, page_size, head_dim, 81)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        query = cpu_query.cuda()
        cache = tuple(tensor.cuda() for tensor in cpu_cache)
        output = torch.empty_like(query)
        metadata = {**cpu_metadata, "indices": cpu_metadata["indices"].cuda()}
        float_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        int_workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        graph_buffers = {
            "paged_kv_indptr_buffer": torch.empty_like(metadata["cu_seqlens_k"], device="cuda"),
            "paged_kv_indices_buffer": torch.empty_like(metadata["indices"]),
            "paged_kv_last_page_len_buffer": torch.empty_like(
                metadata["last_page_len"], device="cuda"
            ),
        }
        operation = FlashInferPagedAttention(
            float_workspace,
            int_workspace,
            mode="decode",
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=True,
            **graph_buffers,
        )
        _plan(operation, metadata, query, cache)
        for _ in range(3):
            operation.run(q=query, paged_kv_cache=cache, out=output)

    # JIT, lazy allocations and asynchronous planning must finish before capture.
    stream.synchronize()
    borrowed = (
        query,
        *cache,
        output,
        float_workspace,
        int_workspace,
        *metadata.values(),
        *graph_buffers.values(),
    )
    pointers = tuple(tensor.data_ptr() for tensor in borrowed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
        captured_output = operation.run(q=query, paged_kv_cache=cache, out=output)
    assert captured_output is output

    first_output = None
    for replay in range(2):
        if replay:
            # The previous completion has been synchronized before touching pinned
            # staging metadata or replanning into the same borrowed workspaces.
            kv_lengths = (1, 31, 50)  # Same page counts and fixed graph batch size.
            changed_metadata = _metadata(query_lengths, kv_lengths, page_size)
            changed_metadata["indices"].copy_(cpu_metadata["indices"].roll(1))
            for name, tensor in cpu_metadata.items():
                tensor.copy_(changed_metadata[name])
            cpu_query, cpu_cache = _inputs(query_lengths, cpu_metadata, page_size, head_dim, 82)
        expected = _reference(
            cpu_query, cpu_cache, query_lengths, kv_lengths, cpu_metadata["indices"]
        )
        with torch.cuda.stream(stream), torch.inference_mode():
            query.copy_(cpu_query)
            for destination, source in zip(cache, cpu_cache, strict=True):
                destination.copy_(source)
            if replay:
                metadata["indices"].copy_(cpu_metadata["indices"])
                _plan(operation, metadata, query, cache)
            output.fill_(torch.nan)
            graph.replay()
            completion = torch.cuda.Event()
            completion.record(stream)
        completion.synchronize()

        assert tuple(tensor.data_ptr() for tensor in borrowed) == pointers
        actual = output.cpu()
        torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.01)
        for actual_input, original in zip((query, *cache), (cpu_query, *cpu_cache), strict=True):
            torch.testing.assert_close(actual_input.cpu(), original, rtol=0, atol=0)
        for buffer_name, metadata_name in (
            ("paged_kv_indptr_buffer", "cu_seqlens_k"),
            ("paged_kv_indices_buffer", "indices"),
            ("paged_kv_last_page_len_buffer", "last_page_len"),
        ):
            torch.testing.assert_close(
                graph_buffers[buffer_name].cpu(), cpu_metadata[metadata_name], rtol=0, atol=0
            )
        if first_output is None:
            first_output = actual
        else:
            assert not torch.equal(actual, first_output)


@pytest.mark.parametrize("page_size", [16, 64])
@torch.inference_mode()
def test_trtllm_decode_matches_reference_and_graph_replays_changed_cache(page_size: int) -> None:
    """The public TRTLLM decode entry uses native XQA on SM120, not TRTLLM-gen.

    This does not qualify TRTLLM context or native FlashAttention inference: their
    installed providers do not support SM120. Neither is replaced with FA2 here.
    """
    query_lengths, kv_lengths = (1, 1, 1), (1, 32, 65)
    cpu_metadata = _metadata(query_lengths, kv_lengths, page_size)
    cpu_query, cpu_cache = _inputs(query_lengths, cpu_metadata, page_size, 128, 93)

    def page_table() -> torch.Tensor:
        page_counts = tuple((length + page_size - 1) // page_size for length in kv_lengths)
        table = torch.zeros((len(kv_lengths), max(page_counts)), dtype=torch.int32)
        start = 0
        for index, count in enumerate(page_counts):
            table[index, :count] = cpu_metadata["indices"][start : start + count]
            start += count
        return table

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query = cpu_query.cuda()
        cache = tuple(tensor.cuda() for tensor in cpu_cache)
        table = page_table().cuda()
        sequence_lengths = cpu_metadata["seq_lens"].cuda()
        # XQA reserves the first 8 MiB for semaphores; zero them before first use.
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        def decode() -> torch.Tensor:
            return trtllm_decode_attention(
                query,
                cache,
                workspace_buffer=workspace,
                page_table=table,
                cache_seqlens=sequence_lengths,
                max_seqlen_k=128,
                softmax_scale=1 / sqrt(query.shape[-1]),
            )

        eager = decode()
    stream.synchronize()
    expected = _reference(cpu_query, cpu_cache, query_lengths, kv_lengths, cpu_metadata["indices"])
    torch.testing.assert_close(eager.cpu().float(), expected, rtol=0.01, atol=0.01)
    for actual, original in zip((query, *cache), (cpu_query, *cpu_cache), strict=True):
        torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = decode()
    borrowed = (query, *cache, table, sequence_lengths, workspace, output)
    pointers = tuple(tensor.data_ptr() for tensor in borrowed)
    assert output.data_ptr() not in pointers[:-1]
    first_output = None
    for replay in range(2):
        if replay:
            kv_lengths = (1, 31, 66)
            changed = _metadata(query_lengths, kv_lengths, page_size)
            changed["indices"].copy_(cpu_metadata["indices"].roll(1))
            cpu_metadata = changed
            cpu_query, cpu_cache = _inputs(query_lengths, cpu_metadata, page_size, 128, 94)
        expected = _reference(
            cpu_query, cpu_cache, query_lengths, kv_lengths, cpu_metadata["indices"]
        )
        expected_table = page_table()
        with torch.cuda.stream(stream):
            query.copy_(cpu_query)
            for destination, source in zip(cache, cpu_cache, strict=True):
                destination.copy_(source)
            table.copy_(expected_table)
            sequence_lengths.copy_(cpu_metadata["seq_lens"])
            output.fill_(torch.nan)
            graph.replay()
        stream.synchronize()
        actual = output.cpu()
        torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(table.cpu(), expected_table, rtol=0, atol=0)
        torch.testing.assert_close(sequence_lengths.cpu(), cpu_metadata["seq_lens"], rtol=0, atol=0)
        for tensor, original in zip((query, *cache), (cpu_query, *cpu_cache), strict=True):
            torch.testing.assert_close(tensor.cpu(), original, rtol=0, atol=0)
        assert tuple(tensor.data_ptr() for tensor in borrowed) == pointers
        if first_output is None:
            first_output = actual
        else:
            assert not torch.equal(actual, first_output)


@pytest.mark.parametrize(
    "qh,kh,dim,batch,page,layout",
    ci_cases(
        [(32, 8, 128), (16, 4, 256)],
        [1, 2, 3, 4],
        [1, 16],
        ["ragged", "long"],
        core=[(32, 8, 128, 4, 1, "ragged"), (16, 4, 256, 1, 16, "long")],
    ),
)
def test_native_graph_prefill_replans_host_lengths(qh, kh, dim, batch, page, layout):
    paged_support_case(
        mode="prefill", graph=True, heads=(qh, kh, dim), batch=batch, page=page, layout=layout
    )


@pytest.mark.parametrize("page", [1, 16])
def test_native_graph_prefill_custom_mask(page):
    paged_support_case(
        mode="prefill", graph=True, heads=(32, 8, 128), batch=2, page=page, custom=True
    )


@pytest.mark.parametrize(
    "dtype,dim,page,batch",
    ci_cases(
        [torch.bfloat16, torch.float16],
        [128, 256],
        [1, 16],
        [2, 3, 4],
        core=[
            (torch.bfloat16, 128, 16, 3),
            (torch.float16, 128, 1, 2),
            (torch.bfloat16, 256, 1, 4),
            (torch.float16, 256, 16, 3),
        ],
    ),
)
@torch.inference_mode()
def test_padded_split_graph_prefill_replans(dtype, dim, page, batch, monkeypatch):
    """Exact advertised storage, one capture, grow/shrink KV and return from C=1."""
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("padded-split qualification requires SM120/121")
    device = torch.device("cuda")
    # Force both geometries on every supported device without adding unmeasured rows.
    low, pivot = (384 + page - 1) // page, (512 + page - 1) // page
    monkeypatch.setitem(
        split_tables.PREFILL_GRAPH_SPLIT,
        (split_tables.device_key(device), 32, 8, dim),
        {(batch, (129, 256), (low, pivot)): 4, (batch, (129, 256), (pivot + 1, None)): 6},
    )
    sizes = FlashInferPagedAttention.workspace_size(
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        max_batch_size=batch,
        max_query_tokens=batch * 256,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=dim,
        device=device,
    )
    generator = torch.Generator().manual_seed(20261016)
    cpu_query = torch.randn(batch * 256, 32, dim, generator=generator, dtype=dtype)
    capacity = batch * ((1024 + page - 1) // page)
    cpu_cache = tuple(
        torch.randn(2 * capacity + 1, page, 8, dim, generator=generator, dtype=dtype)
        for _ in range(2)
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query = cpu_query.cuda()
        cache = tuple(value.cuda() for value in cpu_cache)
        output = torch.empty_like(query)
        workspaces = (
            torch.zeros(sizes[0], dtype=torch.uint8, device=device),
            torch.empty(sizes[1], dtype=torch.uint8, device=device),
        )
        indices = torch.empty(capacity, dtype=torch.int32, device=device)
        operation = FlashInferPagedAttention(
            *workspaces,
            mode="prefill",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32, device=device),
            paged_kv_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32, device=device),
            paged_kv_last_page_len_buffer=torch.empty(batch, dtype=torch.int32, device=device),
            paged_kv_indices_buffer=indices,
        )

    def prepare(queries, lengths):
        host = _metadata(queries, lengths, page)
        active = host["indices"].numel()
        indices[:active].copy_(host["indices"], non_blocking=True)
        metadata = host | {"indices": indices[:active]}
        _plan(operation, metadata, query, cache)
        return host

    queries = (256,) * batch
    lengths = tuple(512 - i * 17 for i in range(batch))
    with torch.cuda.stream(stream):
        host = prepare(queries, lengths)
        assert operation.prefill_route == "padded_split"
        operation.run(q=query, paged_kv_cache=cache, out=output)
    stream.synchronize()
    owned = tuple(
        operation._prefill_options[name]
        for name in (
            "qo_indptr_buf",
            "paged_kv_indptr_buf",
            "paged_kv_last_page_len_buf",
        )
    )
    stable = (query, *cache, output, indices, *workspaces, *owned)
    pointers = tuple(value.data_ptr() for value in stable)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation.run(q=query, paged_kv_cache=cache, out=output)
    stream.synchronize()
    rounds = [
        (queries, lengths),
        (queries, tuple(1024 - i * 31 for i in range(batch))),
        (queries, tuple(384 - i * 13 for i in range(batch))),
        (queries, (256,) * batch),  # No table row: C=1 and provider auto split.
        (tuple(129 + i for i in range(batch)), (768,) * batch),
    ]
    with patch.object(operation._wrapper, "plan", wraps=operation._wrapper.plan) as native_plan:
        for queries, lengths in rounds:
            with torch.cuda.stream(stream):
                host = prepare(queries, lengths)
                if max(lengths) == 256:
                    assert "fixed_split_size" not in native_plan.call_args.kwargs
                else:
                    assert native_plan.call_args.kwargs["fixed_split_size"] > 0
                output.fill_(torch.nan)
                graph.replay()
            stream.synchronize()
            expected = _reference(
                cpu_query[: sum(queries)], cpu_cache, queries, lengths, host["indices"]
            )
            torch.testing.assert_close(
                output[: sum(queries)].cpu().float(), expected, rtol=0.01, atol=0.01
            )
            assert tuple(value.data_ptr() for value in stable) == pointers
            assert operation.prefill_route == "padded_split"
            assert operation._prefill_options["paged_kv_indices_buf"] is indices


@pytest.mark.parametrize(
    "mode,graph,qh,kh,dim,page",
    ci_cases(
        ["decode", "prefill"],
        [False, True],
        [(32, 8, 128), (16, 4, 256)],
        [1, 16],
        core=[
            ("decode", False, 32, 8, 128, 1),
            ("decode", True, 16, 4, 256, 16),
            ("prefill", False, 16, 4, 256, 1),
            ("prefill", True, 32, 8, 128, 16),
        ],
    ),
)
def test_fp8_kv_scales_with_bf16_query(mode, graph, qh, kh, dim, page):
    paged_support_case(mode=mode, graph=graph, heads=(qh, kh, dim), batch=2, page=page, fp8=True)


@pytest.mark.parametrize(
    "mode,option,graph",
    ci_cases(
        ["decode", "prefill"],
        ["window_left", "logits_soft_cap", "sm_scale"],
        [False, True],
        core=[
            (mode, option, False)
            for mode in ("decode", "prefill")
            for option in ("window_left", "logits_soft_cap", "sm_scale")
        ],
    ),
)
def test_paged_attention_options_against_fp32(mode, option, graph):
    values = {"window_left": 128, "logits_soft_cap": 30.0, "sm_scale": 0.07}
    paged_support_case(
        mode=mode,
        graph=graph,
        heads=(32, 8, 128),
        batch=2,
        page=16,
        options={option: values[option]},
    )


@pytest.mark.parametrize(
    "mode,fp8,graph",
    ci_cases(
        ["decode", "prefill"],
        [False, True],
        [False, True],
        core=[(mode, fp8, True) for mode in ("decode", "prefill") for fp8 in (False, True)],
    ),
)
def test_fused_qkv_views_on_every_route(mode, fp8, graph):
    # Device dispatch picks XQA for FP8 decode where qualified, FA2 otherwise.
    paged_support_case(
        mode=mode, graph=graph, heads=(32, 8, 128), batch=2, page=16, fp8=fp8, fused=True
    )


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("graph", [False, True])
def test_zero_v_scale_matches_provider(mode, graph):
    paged_support_case(
        mode=mode,
        graph=graph,
        heads=(32, 8, 128),
        batch=2,
        page=16,
        fp8=True,
        scales={"k_scale": 0.75, "v_scale": 0.0},
    )


@pytest.mark.parametrize(
    "graph,dtype,dim,page,v_scale",
    ci_cases(
        [False, True],
        [torch.bfloat16, torch.float16],
        [128, 256],
        [1, 16],
        [None, 1.25, 0.0],
        core=[
            (False, torch.bfloat16, 128, 1, None),
            (False, torch.float16, 256, 16, 1.25),
            (False, torch.bfloat16, 256, 1, 0.0),
            (False, torch.float16, 128, 16, 0.0),
            (True, torch.bfloat16, 128, 16, 1.25),
            (True, torch.float16, 256, 1, None),
            (True, torch.bfloat16, 256, 16, 0.0),
            (True, torch.float16, 128, 1, 0.0),
        ],
    ),
)
@torch.inference_mode()
def test_cute_fp8_decode_replans_and_replays(monkeypatch, graph, dtype, dim, page, v_scale):
    """Exact advertised scratch, fused-QKV query strides and runtime KV splits."""
    device = torch.device("cuda")
    try:
        _decode_fp8.check_support(
            device,
            num_qo_heads=16,
            num_kv_heads=4,
            head_dim=dim,
            page_size=page,
            q_dtype=dtype,
            kv_dtype=torch.float8_e4m3fn,
        )
    except NotImplementedError as error:
        pytest.skip(str(error))
    for key in ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE"):
        monkeypatch.delenv(f"CTI_FP8DEC_{key}", raising=False)
    batch = 2
    # Force the route on every supported device without adding measured table rows.
    monkeypatch.setitem(
        split_tables.FP8_DECODE_CUTE,
        (split_tables.device_key(device), 16, 4, dim),
        {batch: {"splits": 4, "min_chunk": 64}},
    )
    sizes = FlashInferPagedAttention.workspace_size(
        mode="decode",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=graph,
        max_batch_size=batch,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=dim,
        device=device,
    )
    generator = torch.Generator().manual_seed(20261017)
    cpu_qkv = torch.randn(batch, 24, dim, dtype=dtype, generator=generator)
    cpu_query = cpu_qkv[:, :16]
    capacity = batch * ((1024 + page - 1) // page)
    cpu_cache = tuple(
        torch.randn(2 * capacity + 1, page, 4, dim, generator=generator).to(torch.float8_e4m3fn)
        for _ in range(2)
    )
    scales = {} if v_scale is None else {"k_scale": 0.75, "v_scale": v_scale}
    sm_scale = None if v_scale is None else 0.07
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        qkv = cpu_qkv.cuda()
        query = qkv[:, :16]
        assert not query.is_contiguous()
        cache = tuple(value.cuda() for value in cpu_cache)
        output = torch.empty(query.shape, dtype=dtype, device=device)
        workspaces = (
            torch.zeros(sizes[0], dtype=torch.uint8, device=device),
            torch.empty(sizes[1], dtype=torch.uint8, device=device),
        )
        buffers = (
            {
                "paged_kv_indptr_buffer": torch.empty(batch + 1, dtype=torch.int32, device=device),
                "paged_kv_indices_buffer": torch.empty(capacity, dtype=torch.int32, device=device),
                "paged_kv_last_page_len_buffer": torch.empty(
                    batch, dtype=torch.int32, device=device
                ),
            }
            if graph
            else {}
        )
        operation = FlashInferPagedAttention(
            *workspaces,
            mode="decode",
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=graph,
            **buffers,
        )
    stable = (query, *cache, output, *workspaces, *buffers.values())
    pointers = tuple(value.data_ptr() for value in stable)
    captured = None
    for lengths in ((33, 17), (1024, 777), (1, 31), (513, 129)):
        # Finish prior execution/staging before replacing the host plan inputs.
        stream.synchronize()
        host = _metadata((1, 1), lengths, page)
        with torch.cuda.stream(stream):
            indices = host["indices"].to(device, non_blocking=True)
            operation.plan(
                **(host | {"indices": indices}),
                num_qo_heads=16,
                num_kv_heads=4,
                head_dim=dim,
                page_size=page,
                pos_encoding_mode="NONE",
                dtype=dtype,
                kv_dtype=torch.float8_e4m3fn,
                sm_scale=sm_scale,
            )
            assert operation.decode_route == "cute"
            if captured is None:
                # Warm up before capture and check the optional output allocation.
                allocated = operation.run(q=query, paged_kv_cache=cache, **scales)
            if not graph:
                assert operation.run(q=query, paged_kv_cache=cache, out=output, **scales) is output
        if graph and captured is None:
            stream.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                operation.run(q=query, paged_kv_cache=cache, out=output, **scales)
        if graph:
            with torch.cuda.stream(stream):
                output.fill_(torch.nan)
                captured.replay()
        stream.synchronize()
        expected = _reference(
            cpu_query, cpu_cache, (1, 1), lengths, host["indices"], sm_scale=sm_scale, **scales
        )
        torch.testing.assert_close(output.cpu().float(), expected, rtol=0.01, atol=0.01)
        if lengths == (33, 17):
            torch.testing.assert_close(allocated.cpu().float(), expected, rtol=0.01, atol=0.01)
        assert tuple(value.data_ptr() for value in stable) == pointers
        assert operation.decode_route == "cute"
        if graph:
            ptr, idx, last = operation._cute_csr
            assert ptr.data_ptr() == buffers["paged_kv_indptr_buffer"].data_ptr()
            assert idx is buffers["paged_kv_indices_buffer"]
            assert last.data_ptr() == buffers["paged_kv_last_page_len_buffer"].data_ptr()
            torch.testing.assert_close(ptr.cpu(), host["cu_seqlens_k"], rtol=0, atol=0)
            torch.testing.assert_close(last.cpu(), host["last_page_len"], rtol=0, atol=0)


@pytest.mark.parametrize("route", ["steered", "steered_split", "cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("page", [1, 16])
@torch.inference_mode()
def test_graph_prefill_table_routes_capture_and_replan(route, dtype, dim, page, monkeypatch):
    """Advertised workspaces, borrowed CSR, and changed host lengths on one graph."""
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("graph prefill table routes require SM120/121")
    device = torch.device("cuda")
    batch, hq, hk, total = 2, 32, 8, 128
    table_key = (split_tables.device_key(device), hq, hk, dim)
    monkeypatch.setitem(
        split_tables.PREFILL_GRAPH_SPLIT,
        table_key,
        {
            (batch, (1, 128), (0, None)): {
                "chunks": 4 if route == "steered_split" else 1,
                "tile": 64,
            }
        },
    )
    monkeypatch.setitem(
        split_tables.PREFILL_GRAPH_CUTE,
        table_key,
        {(batch, (64, 64)): {"block_m": 64, "block_n": 32, "stages": 2, "splits": 2}}
        if route == "cute"
        else {},
    )
    sizes = FlashInferPagedAttention.workspace_size(
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        max_batch_size=batch,
        max_query_tokens=total,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=dim,
        device=device,
    )
    generator = torch.Generator().manual_seed(20261017)
    cpu_q = torch.randn(total, hq, dim, dtype=dtype, generator=generator)
    capacity = batch * ((512 + page - 1) // page)
    cpu_cache = tuple(
        torch.randn(2 * capacity + 1, page, hk, dim, dtype=dtype, generator=generator)
        for _ in range(2)
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query = cpu_q.cuda()
        cache = tuple(value.cuda() for value in cpu_cache)
        output = torch.empty_like(query)
        workspaces = (
            torch.zeros(sizes[0], dtype=torch.uint8, device=device),
            torch.empty(sizes[1], dtype=torch.uint8, device=device),
        )
        buffers = tuple(
            torch.empty(size, dtype=torch.int32, device=device)
            for size in (batch + 1, batch + 1, capacity, batch)
        )
        source_indices = torch.empty_like(buffers[2])
        operation = FlashInferPagedAttention(
            *workspaces,
            mode="prefill",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buffer=buffers[0],
            paged_kv_indptr_buffer=buffers[1],
            paged_kv_indices_buffer=buffers[2],
            paged_kv_last_page_len_buffer=buffers[3],
        )

    def prepare(queries, lengths):
        host = _metadata(queries, lengths, page)
        active = host["indices"].numel()
        source_indices[:active].copy_(host["indices"], non_blocking=True)
        operation.plan(
            **(host | {"indices": source_indices[:active]}),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=dim,
            page_size=page,
            pos_encoding_mode="NONE",
            dtype=dtype,
            sm_scale=0.125,
        )
        return host

    expected_route = "cute" if route == "cute" else "padded_split"
    with torch.cuda.stream(stream):
        prepare((64, 64), (257, 193))
        assert operation.prefill_route == expected_route
        if route != "cute":
            assert operation._prefill_graph_layout.tile == 64
        operation.run(q=query, paged_kv_cache=cache, out=output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        operation.run(q=query, paged_kv_cache=cache, out=output)
    stream.synchronize()
    stable = (query, output, *cache, *buffers, *workspaces)
    pointers = tuple(value.data_ptr() for value in stable)
    for queries, lengths in (
        ((64, 64), (257, 193)),
        ((64, 64), (511, 385)),
        ((80, 7), (193, 129)),  # Fewer rows; a request exceeds its first-plan length.
        ((19, 13), (65, 49)),
    ):
        with torch.cuda.stream(stream):
            host = prepare(queries, lengths)
            output.fill_(torch.nan)
            graph.replay()
        stream.synchronize()
        expected = _reference(
            cpu_q[: sum(queries)], cpu_cache, queries, lengths, host["indices"], sm_scale=0.125
        )
        torch.testing.assert_close(
            output[: sum(queries)].cpu().float(), expected, rtol=0.01, atol=0.01
        )
        assert operation.prefill_route == expected_route
        assert tuple(value.data_ptr() for value in stable) == pointers


def _query_view(rows, heads, dim, layout, device):
    if layout == "misaligned":
        storage = torch.empty(rows * heads * dim + 1, dtype=torch.bfloat16, device=device)
        return storage[1:].view(rows, heads, dim)
    if layout == "odd":
        return torch.empty(rows, heads, dim + 1, dtype=torch.bfloat16, device=device)[:, :, :dim]
    width = heads * 3 // 2 if layout == "fused" else heads * 2
    storage = torch.empty(rows, width, dim, dtype=torch.bfloat16, device=device)
    return storage[:, :heads] if layout == "fused" else storage[:, ::2]


@pytest.mark.parametrize(
    "route,layout",
    [
        ("cute", "fused"),
        ("cute", "heads"),
        ("cute", "misaligned"),
        ("cute", "odd"),
        ("native", "fused"),
    ],
)
@torch.inference_mode()
def test_graph_prefill_query_views_packed_kv_and_scales(monkeypatch, route, layout):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("CuTe prefill requires SM120/121")
    for key in ("BLOCK_M", "BLOCK_N", "STAGES", "SPLITS", "MIN_CHUNK"):
        monkeypatch.delenv("CTI_PREFILL_" + key, raising=False)
    device = torch.device("cuda")
    batch, rows, hq, hk, dim, page = 3, 192, 32, 8, 128, 16
    key = (split_tables.device_key(device), hq, hk, dim)
    monkeypatch.setitem(
        split_tables.PREFILL_GRAPH_CUTE,
        key,
        {
            (batch, (64, 64)): {
                "block_m": 64,
                "block_n": 32,
                "stages": 2,
                "splits": 3,
                "min_chunk": 64,
            }
        }
        if route == "cute"
        else {},
    )
    monkeypatch.setitem(split_tables.PREFILL_GRAPH_SPLIT, key, {})
    sizes = FlashInferPagedAttention.workspace_size(
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        max_batch_size=batch,
        max_query_tokens=rows,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=dim,
        device=device,
    )
    lengths, queries = (257, 193, 129), (64, 64, 64)
    host = _metadata(queries, lengths, page)
    pages = 2 * host["indices"].numel() + 1
    torch.manual_seed(61017)
    cpu_q = torch.randn(rows, hq, dim, dtype=torch.bfloat16)
    cpu_packed = torch.randn(pages, 2, page, hk, dim, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query = _query_view(rows, hq, dim, layout, device)
        query.copy_(cpu_q)
        packed = cpu_packed.cuda()
        cache = (packed[:, 0], packed[:, 1])
        out = torch.empty(rows, hq, dim, dtype=query.dtype, device=device)
        buffers = [
            torch.empty(n, dtype=torch.int32, device=device)
            for n in (batch + 1, batch + 1, host["indices"].numel(), batch)
        ]
        op = FlashInferPagedAttention(
            *(torch.empty(n, dtype=torch.uint8, device=device) for n in sizes),
            mode="prefill",
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr_buffer=buffers[0],
            paged_kv_indptr_buffer=buffers[1],
            paged_kv_indices_buffer=buffers[2],
            paged_kv_last_page_len_buffer=buffers[3],
        )
        op.plan(
            **(host | {"indices": host["indices"].cuda()}),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=dim,
            page_size=page,
            dtype=query.dtype,
            pos_encoding_mode="NONE",
        )
        assert op.prefill_route == route
    scales = ((0.37, 1.9), (1.0, 0.0), (-0.5, 1.0), (0.0, 2.0))
    if route == "native":
        # FlashInfer 0.7.0.post1 FA2 folds k_scale into its logit scale and returns
        # NaN for every row when that scale is <= 0; the CuTe route accepts it.
        scales = tuple(pair for pair in scales if pair[0] > 0)
    for k_scale, v_scale in scales:
        with torch.cuda.stream(stream):
            op.run(q=query, paged_kv_cache=cache, out=out, k_scale=k_scale, v_scale=v_scale)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op.run(q=query, paged_kv_cache=cache, out=out, k_scale=k_scale, v_scale=v_scale)
        for factor in (1.0, 0.5):
            with torch.cuda.stream(stream):
                query.copy_(cpu_q * factor)
                out.fill_(torch.nan)
                graph.replay()
            stream.synchronize()
            expected = _reference(
                cpu_q * factor,
                (cpu_packed[:, 0], cpu_packed[:, 1]),
                queries,
                lengths,
                host["indices"],
                k_scale=k_scale,
                v_scale=v_scale,
            )
            torch.testing.assert_close(out.cpu().float(), expected, atol=0.02, rtol=0.01)
            if v_scale == 0:
                assert torch.count_nonzero(out).item() == 0
        assert op.prefill_route == route


@pytest.mark.parametrize(
    "route,layout",
    [
        ("cute", "fused"),
        ("cute", "heads"),
        ("cute", "misaligned"),
        ("cute", "odd"),
        ("fa2", "fused"),
        ("xqa", "fused"),
    ],
)
@torch.inference_mode()
def test_decode_query_views_packed_kv_and_bf16_range(monkeypatch, route, layout):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("CuTe/XQA decode requires SM120/121")
    from ctikernels.attention._flashinfer import xqa_decode_qualified

    device = torch.device("cuda")
    batch, hq, hk, dim, page = 2, 16, 4, 128, 16
    for key in ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE"):
        monkeypatch.delenv("CTI_FP8DEC_" + key, raising=False)
    key = (split_tables.device_key(device), hq, hk, dim)
    monkeypatch.setitem(
        split_tables.FP8_DECODE_CUTE,
        key,
        {batch: {"splits": 3, "min_chunk": 64}} if route == "cute" else {},
    )
    monkeypatch.setitem(split_tables.XQA_DECODE_SUBSEQ, key, {batch: 8})
    if route == "xqa" and not xqa_decode_qualified(
        device,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=dim,
        page_size=page,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
        logits_soft_cap=None,
        window_left=-1,
        rope_scale=None,
        rope_theta=None,
    ):
        pytest.skip("installed FlashInfer does not qualify for XQA")
    sizes = FlashInferPagedAttention.workspace_size(
        mode="decode",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        max_batch_size=batch,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=dim,
        device=device,
    )
    lengths = (17, 193)
    host = _metadata((1, 1), lengths, page)
    pages = 2 * host["indices"].numel() + 1
    torch.manual_seed(61018)
    cpu_q = torch.randn(batch, hq, dim, dtype=torch.bfloat16)
    cpu_packed = torch.randn(pages, 2, page, hk, dim).to(torch.float8_e4m3fn)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        query = _query_view(batch, hq, dim, layout, device)
        query.copy_(cpu_q)
        packed = cpu_packed.cuda()
        cache = (packed[:, 0], packed[:, 1])
        out = torch.empty(batch, hq, dim, dtype=query.dtype, device=device)
        op = FlashInferPagedAttention(
            *(torch.empty(n, dtype=torch.uint8, device=device) for n in sizes),
            mode="decode",
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=True,
            dispatch="provider" if route == "fa2" else "device",
            paged_kv_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32, device=device),
            paged_kv_indices_buffer=torch.empty(
                host["indices"].numel(), dtype=torch.int32, device=device
            ),
            paged_kv_last_page_len_buffer=torch.empty(batch, dtype=torch.int32, device=device),
        )
        op.plan(
            **(host | {"indices": host["indices"].cuda()}),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=dim,
            page_size=page,
            dtype=query.dtype,
            kv_dtype=torch.float8_e4m3fn,
            pos_encoding_mode="NONE",
        )
        assert op.decode_route == route
        if route == "cute":
            stage_pointer = op._cute_q_stage.data_ptr()
            dense_cache = tuple(t.contiguous() for t in cache)
            dense_out = torch.empty_like(out)
            op.run(q=query.contiguous(), paged_kv_cache=dense_cache, out=dense_out)
        op.run(q=query, paged_kv_cache=cache, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.run(q=query, paged_kv_cache=cache, out=out)
    with torch.cuda.stream(stream):
        out.fill_(torch.nan)
        graph.replay()
    stream.synchronize()
    expected = _reference(
        cpu_q, (cpu_packed[:, 0], cpu_packed[:, 1]), (1, 1), lengths, host["indices"]
    )
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.01, rtol=0.03)
    if route == "cute":
        assert torch.equal(out.view(torch.int16), dense_out.view(torch.int16))
        assert op._cute_q_stage.data_ptr() == stage_pointer
        if layout in ("heads", "misaligned", "odd"):
            assert torch.equal(op._cute_q_stage.view(torch.int16), query.view(torch.int16))
        for magnitude in (65536, 1e30):
            with torch.cuda.stream(stream):
                query.fill_(magnitude)
                query[:, :, ::2].neg_()
                cache[0].zero_()
                cache[1].fill_(2)
                out.fill_(torch.nan)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(out, torch.full_like(out, 2), atol=0, rtol=0)
    assert op.decode_route == route


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("layout", ["heads", "padded", "mixed", "packed"])
@torch.inference_mode()
def test_cute_aligned_kv_views(monkeypatch, mode, layout):
    from ._attention import kv_views

    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("CuTe paged attention requires SM120/121")
    for prefix, keys in (
        ("CTI_FP8DEC_", ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE")),
        ("CTI_PREFILL_", ("BLOCK_M", "BLOCK_N", "STAGES", "SPLITS", "MIN_CHUNK")),
    ):
        for key in keys:
            monkeypatch.delenv(prefix + key, raising=False)
    device = torch.device("cuda")
    batch, hq, hk, dim, page = 2, 8, 2, 128, 16
    queries = (1, 1) if mode == "decode" else (17, 17)
    lengths = (193, 129)
    kv_dtype = torch.float8_e4m3fn if mode == "decode" else torch.bfloat16
    key = (split_tables.device_key(device), hq, hk, dim)
    monkeypatch.setitem(split_tables.FP8_DECODE_CUTE, key, {batch: {"splits": 3, "min_chunk": 64}})
    monkeypatch.setitem(
        split_tables.PREFILL_GRAPH_CUTE,
        key,
        {
            (batch, (17, 17)): {
                "block_m": 64,
                "block_n": 32,
                "stages": 2,
                "splits": 3,
                "min_chunk": 64,
            }
        },
    )
    sizes = FlashInferPagedAttention.workspace_size(
        mode=mode,
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        max_batch_size=batch,
        max_query_tokens=sum(queries),
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=dim,
        device=device,
    )
    host = _metadata(queries, lengths, page)
    torch.manual_seed(42)
    cpu_q = torch.randn(sum(queries), hq, dim, dtype=torch.bfloat16)
    cpu_cache = tuple(
        torch.randn(2 * host["indices"].numel() + 1, page, hk, dim).to(kv_dtype) for _ in range(2)
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = cpu_q.cuda()
        dense = tuple(x.cuda() for x in cpu_cache)
        cache = kv_views(dense, layout)
        out, dense_out = torch.empty_like(q), torch.empty_like(q)
        buffers = {
            "paged_kv_indptr_buffer": torch.empty(batch + 1, dtype=torch.int32, device=device),
            "paged_kv_indices_buffer": torch.empty(
                host["indices"].numel(), dtype=torch.int32, device=device
            ),
            "paged_kv_last_page_len_buffer": torch.empty(batch, dtype=torch.int32, device=device),
        }
        if mode == "prefill":
            buffers["qo_indptr_buffer"] = torch.empty(batch + 1, dtype=torch.int32, device=device)
        op = FlashInferPagedAttention(
            *(torch.zeros(n, dtype=torch.uint8, device=device) for n in sizes),
            mode=mode,
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=True,
            **buffers,
        )
        op.plan(
            **(host | {"indices": host["indices"].cuda()}),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim=dim,
            page_size=page,
            pos_encoding_mode="NONE",
            dtype=q.dtype,
            kv_dtype=kv_dtype,
        )
        assert getattr(op, f"{mode}_route") == "cute"
        op.run(q=q, paged_kv_cache=dense, out=dense_out)
        op.run(q=q, paged_kv_cache=cache, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.run(q=q, paged_kv_cache=cache, out=out)
    for _ in range(2):
        with torch.cuda.stream(stream):
            out.fill_(torch.nan)
            graph.replay()
        stream.synchronize()
        assert torch.equal(out.view(torch.int16), dense_out.view(torch.int16))
    expected = _reference(cpu_q, cpu_cache, queries, lengths, host["indices"])
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.01, rtol=0.03)
    assert getattr(op, f"{mode}_route") == "cute"
