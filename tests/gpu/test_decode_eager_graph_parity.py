# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Eager and CUDA-graph decode of the same batch give bitwise equal outputs."""

import pytest
import torch

from ctikernels.attention.inference import FlashInferPagedAttention

from ._attention import paged_metadata
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PAGE = 16
HEADS = {
    "qwen3-8b": (32, 8, 128),
    "qwen35": (16, 4, 256),
    "qwen3-8b-tp2": (16, 4, 128),
    "qwen35-27b": (24, 4, 256),
    "qwen35-tp2": (8, 2, 256),
}
# Two length sets per context replan the same graph; long sets cross every KV
# bucket of the Split-KV tables (16384+) and ragged page tails.
CONTEXTS = {
    "short": ((37, 300, 511, 129), (1, 17, 64, 1000)),
    "long": ((32773, 20000, 9001, 16461), (16384, 4097, 30001, 2049)),
}
CASES = ci_cases(
    (1, 2, 3, 4),
    tuple(HEADS),
    ("bf16", "fp8"),
    tuple(CONTEXTS),
    core={
        (batch, heads, kv, context)
        for batch in (1, 2, 3, 4)
        for heads in ("qwen3-8b", "qwen35")
        for kv in ("bf16", "fp8")
        for context in ("short", "long")
    },
)


@pytest.fixture(autouse=True)
def provider_available():
    pytest.importorskip("flashinfer")


def workspaces(**options):
    sizes = FlashInferPagedAttention.workspace_size(device=torch.device("cuda"), **options)
    return tuple(torch.zeros(size, dtype=torch.uint8, device="cuda") for size in sizes)


def plan(operation, metadata, heads, kv_dtype):
    operation.plan(
        **metadata,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim=heads[2],
        page_size=PAGE,
        pos_encoding_mode="NONE",
        dtype=torch.bfloat16,
        kv_dtype=kv_dtype,
    )


def device_metadata(lengths):
    metadata = paged_metadata((1,) * len(lengths), lengths, PAGE)
    return {**metadata, "indices": metadata["indices"].cuda()}


@pytest.mark.parametrize("batch,heads,kv,context", CASES)
def test_decode_eager_and_graph_outputs_are_bitwise_equal(batch, heads, kv, context):
    heads = HEADS[heads]
    kv_dtype = torch.float8_e4m3fn if kv == "fp8" else torch.bfloat16
    scales = {"k_scale": 0.75, "v_scale": 0.5} if kv == "fp8" else {}
    rounds = [lengths[:batch] for lengths in CONTEXTS[context]]
    other = (2048,) * (5 - batch)
    pages = [sum(-(-length // PAGE) for length in lengths) for lengths in (*rounds, other)]
    # Engine-sized graph indices buffer (40k tokens per request): the XQA table
    # width follows it in graph mode and the plan's pages in eager mode.
    capacity = max(max(pages[:-1]), batch * 40960 // PAGE)
    generator = torch.Generator(device="cuda").manual_seed(4091 + batch)
    cache = tuple(
        torch.randn(
            2 * max(pages) + 1,  # paged_metadata uses odd page indices.
            PAGE,
            heads[1],
            heads[2],
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(kv_dtype)
        for _ in range(2)
    )
    q = torch.randn(batch, heads[0], heads[2], generator=generator, device="cuda")
    q = q.to(torch.bfloat16)
    geometry = {"num_qo_heads": heads[0], "num_kv_heads": heads[1], "head_dim": heads[2]}
    options = {"mode": "decode", "backend": "fa2", "use_tensor_cores": True}
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        # One eager operation for every batch, as an engine keeps it; a plan of
        # another batch first, so the target plan may switch native kernels.
        eager = FlashInferPagedAttention(
            *workspaces(**options, **geometry, max_batch_size=4), **options
        )
        plan(eager, device_metadata(other), heads, kv_dtype)
        eager.run(
            q=torch.zeros(len(other), *q.shape[1:], dtype=q.dtype, device="cuda"),
            paged_kv_cache=cache,
            **scales,
        )
        graph_options = {**options, "use_cuda_graph": True}
        operation = FlashInferPagedAttention(
            *workspaces(**graph_options, **geometry, max_batch_size=batch),
            **graph_options,
            paged_kv_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indices_buffer=torch.empty(capacity, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.empty(batch, dtype=torch.int32, device="cuda"),
        )
        # Engine capture: one dummy token per request, then actual-length replans.
        plan(operation, device_metadata((1,) * batch), heads, kv_dtype)
        graph_out = torch.empty_like(q)
        operation.run(q=q, paged_kv_cache=cache, out=graph_out, **scales)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            operation.run(q=q, paged_kv_cache=cache, out=graph_out, **scales)
        stream.synchronize()
        for lengths in rounds:
            metadata = device_metadata(lengths)
            plan(operation, metadata, heads, kv_dtype)
            graph.replay()
            plan(eager, metadata, heads, kv_dtype)
            eager_out = eager.run(q=q, paged_kv_cache=cache, **scales)
            stream.synchronize()
            assert eager.decode_route == operation.decode_route
            assert torch.isfinite(graph_out.float()).all()
            assert torch.equal(eager_out, graph_out), (
                lengths,
                operation.decode_route,
                (eager_out.float() - graph_out.float()).abs().max().item(),
            )
        del graph, operation
