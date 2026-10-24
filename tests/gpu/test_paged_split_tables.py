# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Device Split-KV tables through capture, real-length planning and replay."""

import pytest
import torch

from ctikernels.attention._split_tables import (
    decode_uses_cuda_cores,
    device_key,
    prefill_query_split,
)
from ctikernels.attention.inference import FlashInferPagedAttention

from ._attention import paged_metadata, reference_attention
from ._exhaustive import ci_cases
from .test_inference_attention import _plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
CASES = ci_cases(
    (1, 2, 3, 4),
    (512, 1024, 2048, 4096, 8192, 16384, 1000, 3000),
    core={(1, 512), (2, 8192), (4, 1024)},
)


@pytest.fixture(autouse=True)
def provider_available():
    pytest.importorskip("flashinfer")


def inputs(batch, length, query_length, heads):
    query_heads, kv_heads, dim = heads
    generator = torch.Generator().manual_seed(7163 + batch + length)
    metadata = paged_metadata((query_length,) * batch, (length,) * batch, 1)
    query = torch.randn(
        batch * query_length, query_heads, dim, generator=generator, dtype=torch.bfloat16
    )
    cache = tuple(
        torch.randn(
            2 * batch * length + 1, 1, kv_heads, dim, generator=generator, dtype=torch.bfloat16
        )
        for _ in range(2)
    )
    expected = reference_attention(
        query, cache, (query_length,) * batch, (length,) * batch, metadata["indices"]
    )
    return query.cuda(), tuple(value.cuda() for value in cache), metadata, expected


def workspaces(operation, **geometry):
    sizes = operation.workspace_size(device=torch.device("cuda"), **geometry)
    return tuple(torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes)


@pytest.mark.parametrize("batch,length", CASES)
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 4, 256)], ids=["qwen3-8b", "qwen35"])
def test_graph_decode_device_splits_match_reference_and_provider(batch, length, heads):
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        q, cache, cpu_metadata, expected = inputs(batch, length, 1, heads)
        actual_metadata = {**cpu_metadata, "indices": cpu_metadata["indices"].cuda()}
        dummy = paged_metadata((1,) * batch, (1,) * batch, 1)
        dummy_metadata = {**dummy, "indices": dummy["indices"].cuda()}
        geometry = {"num_qo_heads": heads[0], "num_kv_heads": heads[1], "head_dim": heads[2]}
        options = {
            "mode": "decode",
            "backend": "fa2",
            "use_tensor_cores": True,
            "use_cuda_graph": True,
        }
        outputs = []
        for policy in ("device", "provider"):
            workspace = workspaces(
                FlashInferPagedAttention,
                **options,
                **geometry,
                max_batch_size=batch,
                dispatch=policy,
            )
            indptr = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
            indices = torch.empty(batch * length, dtype=torch.int32, device="cuda")
            last = torch.empty(batch, dtype=torch.int32, device="cuda")
            operation = FlashInferPagedAttention(
                *workspace,
                **options,
                dispatch=policy,
                paged_kv_indptr_buffer=indptr,
                paged_kv_indices_buffer=indices,
                paged_kv_last_page_len_buffer=last,
            )
            # Engine capture starts with one dummy token per request, always using
            # the provider plan. Actual-length host planning follows capture.
            _plan(operation, dummy_metadata, q, cache)
            routed = policy == "device" and decode_uses_cuda_cores(
                torch.device("cuda"), *heads, batch
            )
            assert operation._use_tensor_cores is not routed
            assert operation._wrapper.use_tensor_cores is not routed
            out = torch.empty_like(q)
            operation.run(q=q, paged_kv_cache=cache, out=out)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                operation.run(q=q, paged_kv_cache=cache, out=out)
            stream.synchronize()
            _plan(operation, actual_metadata, q, cache)
            graph.replay()
            stream.synchronize()
            actual = out.cpu().float()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            outputs.append(actual)
            # Finish all use before releasing borrowed metadata/workspace storage.
            del graph, operation
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_exact_workspace_routes_cuda_core_graph_decode(batch):
    heads = (32, 8, 128)
    if not decode_uses_cuda_cores(torch.device("cuda"), *heads, batch):
        pytest.skip("no CUDA-core dispatch policy for this device and batch")
    test_graph_decode_device_splits_match_reference_and_provider(batch, 512, heads)


PREFILL_CASES = ci_cases(
    (1, 2, 3, 4),
    (8, 16, 22, 32, 48, 64, 128, 256),
    core={(1, 22), (2, 64), (4, 8), (2, 22), (1, 64)},
)


def check_prefill(query_lengths, kv_lengths):
    generator = torch.Generator().manual_seed(7163)
    metadata = paged_metadata(query_lengths, kv_lengths, 1)
    q = torch.randn(sum(query_lengths), 32, 128, generator=generator, dtype=torch.bfloat16)
    cache = tuple(
        torch.randn(2 * sum(kv_lengths) + 1, 1, 8, 128, generator=generator, dtype=torch.bfloat16)
        for _ in range(2)
    )
    expected = reference_attention(q, cache, query_lengths, kv_lengths, metadata["indices"])
    split = prefill_query_split(
        device_key(torch.device("cuda")),
        metadata["cu_seqlens_q"].tolist(),
        metadata["cu_seqlens_k"].tolist(),
        list(kv_lengths),
        head_dim=128,
        causal=True,
        page_size=1,
    )
    stream = torch.cuda.Stream()
    outputs = []
    with torch.inference_mode(), torch.cuda.stream(stream):
        q = q.cuda()
        cache = tuple(value.cuda() for value in cache)
        metadata["indices"] = metadata["indices"].cuda()
        for policy in ("device", "provider"):
            workspace = workspaces(
                FlashInferPagedAttention,
                mode="prefill",
                backend="fa2",
                max_batch_size=len(query_lengths),
                max_query_tokens=sum(query_lengths),
                num_qo_heads=32,
                num_kv_heads=8,
                head_dim=128,
                dispatch=policy,
            )
            operation = FlashInferPagedAttention(
                *workspace, mode="prefill", backend="fa2", dispatch=policy
            )
            _plan(operation, metadata, q, cache)
            assert (operation._split_indices is not None) == (
                policy == "device" and split is not None
            )
            assert (operation._prefill_metadata is not None) == (
                policy == "device" and split is not None
            )
            out = torch.empty_like(q)
            operation.run(q=q, paged_kv_cache=cache, out=out)
            stream.synchronize()
            torch.testing.assert_close(out.cpu().float(), expected, atol=2e-2, rtol=2e-2)
            # Replanning reuses owned storage after the previous staging copy completes.
            owned = operation._split_indices
            _plan(operation, metadata, q, cache)
            assert operation._split_indices is owned
            operation.run(q=q, paged_kv_cache=cache, out=out)
            stream.synchronize()
            outputs.append(out.cpu().float())
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch,query", PREFILL_CASES)
def test_prefill_device_matches_reference_and_provider(batch, query):
    check_prefill((query,) * batch, (query,) * batch)


@pytest.mark.parametrize("kv_lengths", [(5, 22, 40), (5, 29, 40)])
def test_prefill_mixed_and_cached_prefix(kv_lengths):
    check_prefill((5, 22, 40), kv_lengths)
