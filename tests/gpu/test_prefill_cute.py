# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paged prefill versus FP32, including mutable CSR under CUDA graph replay."""

import pytest
import torch

from ctikernels.attention._prefill_cute import CutePagedPrefill, workspace_size

from ._exhaustive import pairwise_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def device_and_config(monkeypatch):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("CuTe prefill requires sm_120/sm_121")
    pytest.importorskip("cutlass.cute")
    for key in ("BLOCK_M", "BLOCK_N", "STAGES", "SPLITS", "MIN_CHUNK"):
        monkeypatch.delenv("CTI_PREFILL_" + key, raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def metadata(queries, lengths, page, cache_pages, rotation=0):
    counts = [(length + page - 1) // page for length in lengths]
    qp = torch.tensor([0, *queries], dtype=torch.int32).cumsum(0).int()
    kp = torch.tensor([0, *counts], dtype=torch.int32).cumsum(0).int()
    # A permutation plus repeated physical pages exercises indirection/sharing.
    indices = (torch.arange(sum(counts), dtype=torch.int32) * 7 + rotation) % cache_pages
    last = torch.tensor(
        [(length - 1) % page + 1 if length else 0 for length in lengths], dtype=torch.int32
    )
    return qp, kp, indices, last


def reference(q, k, v, host, queries, lengths, scale):
    qp, kp, ids, _ = host
    output = torch.zeros_like(q, dtype=torch.float32)
    group = q.shape[1] // k.shape[2]
    mapping = torch.arange(q.shape[1], device=q.device) // group
    for b, (query, length) in enumerate(zip(queries, lengths, strict=True)):
        if not query or not length:
            continue
        pages = ids[int(kp[b]) : int(kp[b + 1])].long().to(q.device)
        keys = k[pages].flatten(0, 1)[:length, mapping].float()
        values = v[pages].flatten(0, 1)[:length, mapping].float()
        first = int(qp[b])
        scores = torch.einsum("ihd,jhd->hij", q[first : first + query].float(), keys) * scale
        positions = length - query + torch.arange(query, device=q.device)
        visible = torch.arange(length, device=q.device)[None, :] <= positions[:, None]
        scores.masked_fill_(~visible[None, :, :], -float("inf"))
        weights = scores.softmax(-1).nan_to_num(0)
        output[first : first + query] = torch.einsum("hij,jhd->ihd", weights, values)
    return output


def exercise(
    queries, lengths, hq, hk, dim, page, dtype, *, qmax=None, strided=False, replay=None, scale=None
):
    batch = len(queries)
    qmax = max(1, max(queries)) if qmax is None else qmax
    capacity = batch * qmax if replay else sum(queries)
    max_length = max([*lengths, *(replay[1] if replay else [])], default=1)
    cache_pages = max(1, batch * ((max_length + page - 1) // page))
    torch.manual_seed(73)
    storage = torch.randn(capacity, hq * (3 if strided else 1), dim, dtype=dtype, device="cuda")
    q = storage[:, :hq]
    k, v = [torch.randn(cache_pages, page, hk, dim, dtype=dtype, device="cuda") for _ in range(2)]
    out = torch.full((capacity, hq, dim), 7.0, dtype=dtype, device="cuda")
    host = metadata(queries, lengths, page, cache_pages)
    gpu = [value.to("cuda") for value in host]
    # Replay can change page counts in either direction without replacing buffers.
    ids = torch.zeros(cache_pages, dtype=torch.int32, device="cuda")
    ids[: gpu[2].numel()].copy_(gpu[2])
    gpu[2] = ids
    geometry = {
        "batch_size": batch,
        "max_query_per_request": qmax,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
    }
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    scale = dim**-0.5 if scale is None else scale
    op = CutePagedPrefill(
        workspace,
        **geometry,
        page_size=page,
        dtype=dtype,
        device="cuda",
        sm_scale=scale,
        q_row_stride=q.stride(0),
    )

    def call():
        op.run(q, k, v, *gpu, out)

    call()
    expected = reference(q, k, v, host, queries, lengths, scale)
    torch.testing.assert_close(
        out[: sum(queries)].float(), expected[: sum(queries)], rtol=0.01, atol=0.01
    )
    torch.testing.assert_close(
        out[sum(queries) :], torch.full_like(out[sum(queries) :], 7), rtol=0, atol=0
    )
    if replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        # Fewer rows, new page IDs and lengths, split-count transitions and zero
        # requests. Captured pointers/shapes/grid stay unchanged.
        for iteration in range(3):
            next_queries, next_lengths = replay if iteration != 1 else (queries, lengths)
            next_host = metadata(
                next_queries, next_lengths, page, cache_pages, rotation=iteration + 3
            )
            for target, source in zip(gpu, next_host, strict=True):
                target[: source.numel()].copy_(source)
            out.fill_(7)
            q.mul_(0.9)
            graph.replay()
            expected = reference(q, k, v, next_host, next_queries, next_lengths, scale)
            live = sum(next_queries)
            torch.testing.assert_close(out[:live].float(), expected[:live], rtol=0.01, atol=0.01)
            torch.testing.assert_close(out[live:], torch.full_like(out[live:], 7), rtol=0, atol=0)
        if op.config["splits"] > 1:
            assert torch.count_nonzero(op._counters).item() == 0


@pytest.mark.parametrize(
    "batch,query,length_mode,hq,hk,dim,page,dtype,ragged",
    pairwise_cases(
        [1, 2, 3, 4],
        [1, 16, 64, 100, 256],
        ["self", "2048", "8192", "mixed"],
        [(32, 8, 128), (16, 4, 256), (16, 4, 128), (8, 2, 256), (4, 4, 64)],
        [1, 16],
        [torch.bfloat16, torch.float16],
        [False, True],
        seed=73,
        compile_axes=(0, 1, 3, 4, 5),  # batch, qmax, heads/dim, page, dtype
        core=[
            (3, 64, "2048", 32, 8, 128, 16, torch.bfloat16, False),
            (1, 256, "2048", 32, 8, 128, 16, torch.float16, False),
            (3, 64, "2048", 16, 4, 256, 16, torch.bfloat16, False),
            (2, 16, "self", 16, 4, 128, 1, torch.float16, True),
            (4, 100, "mixed", 8, 2, 256, 16, torch.float16, True),
            (2, 1, "8192", 4, 4, 64, 1, torch.bfloat16, True),
            (1, 1, "self", 4, 4, 64, 16, torch.float16, False),
        ],
    ),
)
def test_prefill(batch, query, length_mode, hq, hk, dim, page, dtype, ragged):
    queries = [query] * batch
    if ragged:
        queries = [[0, 1, query, max(1, query // 2)][b % 4] for b in range(batch)]
    if length_mode == "self":
        lengths = queries
    elif length_mode == "mixed":
        lengths = [max(q, [0, 17, 2048, 8192][b % 4]) for b, q in enumerate(queries)]
    else:
        lengths = [int(length_mode)] * batch
    exercise(queries, lengths, hq, hk, dim, page, dtype, qmax=query)


@pytest.mark.parametrize(
    "dim,rows,stages,splits,dtype",
    [
        (64, 128, 3, 7, torch.float16),
        (128, 64, 2, 1, torch.bfloat16),
        (128, 128, 3, 4, torch.float16),
        (256, 64, 3, 4, torch.bfloat16),
        (256, 128, 2, 7, torch.float16),
    ],
)
def test_graph_replay_and_strided_q(monkeypatch, dim, rows, stages, splits, dtype):
    for key, value in {
        "BLOCK_M": rows,
        "STAGES": stages,
        "SPLITS": splits,
        "MIN_CHUNK": 17,
    }.items():
        monkeypatch.setenv("CTI_PREFILL_" + key, str(value))
    exercise(
        [64, 1, 37],
        [2048, 13, 257],
        12,
        3,
        dim,
        16,
        dtype,
        qmax=100,
        strided=True,
        replay=([1, 0, 16], [1, 0, 31]),
        scale=0.17,
    )


@pytest.mark.parametrize("group,page", [(1, 1), (3, 7), (9, 3), (17, 16), (33, 31)])
@pytest.mark.parametrize("splits", [1, 7])
def test_arbitrary_gqa_pages_and_no_visible_keys(monkeypatch, group, page, splits):
    monkeypatch.setenv("CTI_PREFILL_SPLITS", str(splits))
    monkeypatch.setenv("CTI_PREFILL_MIN_CHUNK", "1")
    exercise(
        [17, 0, 1, 9],
        [3, 0, 0, 71],
        group * 2,
        2,
        64,
        page,
        torch.bfloat16,
        replay=([0, 0, 0, 0], [0, 0, 0, 0]),
    )


@pytest.mark.parametrize("rows,keys,stages", [(64, 32, 2), (64, 32, 3), (128, 64, 2)])
@pytest.mark.parametrize("scale", [-0.17, 0.0, 0.17])
def test_full_diagonal_tail_tiles_and_scale(monkeypatch, rows, keys, stages, scale):
    # Non-page-aligned tiles, padded query tiles, Q > KV, and full tiles all
    # share the pipelined page-ID ring, including its final zero-filled slots.
    for key, value in {
        "BLOCK_M": rows,
        "BLOCK_N": keys,
        "STAGES": stages,
        "SPLITS": 4,
        "MIN_CHUNK": 17,
    }.items():
        monkeypatch.setenv("CTI_PREFILL_" + key, str(value))
    exercise(
        [35, 1, 17, 64],
        [67, 1, 3, 1025],
        12,
        4,
        128,
        7,
        torch.bfloat16,
        scale=scale,
        replay=([1, 0, 17, 33], [1, 0, 32, 65]),
    )


@pytest.mark.parametrize(
    "rows,keys,stages,splits,page,dtype",
    [
        (64, 64, 2, 1, 16, torch.bfloat16),
        (64, 64, 2, 3, 16, torch.bfloat16),
        (64, 64, 2, 7, 7, torch.float16),
        (64, 64, 2, 4, 1, torch.bfloat16),
        (64, 32, 3, 4, 16, torch.bfloat16),
        (128, 32, 2, 7, 16, torch.float16),
    ],
)
@pytest.mark.parametrize("scale", [-0.17, 0.0, 0.17])
def test_d256_pipeline_and_merge_replay(
    monkeypatch, rows, keys, stages, splits, page, dtype, scale
):
    # Exercise independent K/V storage across many wraps, full/masked tiles,
    # all-empty rows, and reuse of the winner's normalized-weight workspace.
    for key, value in {
        "BLOCK_M": rows,
        "BLOCK_N": keys,
        "STAGES": stages,
        "SPLITS": splits,
        "MIN_CHUNK": 64,
    }.items():
        monkeypatch.setenv("CTI_PREFILL_" + key, str(value))
    exercise(
        [64, 1, 37, 17],
        [2048, 0, 257, 3],
        12,
        4,
        256,
        page,
        dtype,
        qmax=100,
        strided=True,
        scale=scale,
        replay=([1, 0, 17, 33], [1, 0, 32, 65]),
    )


@pytest.mark.parametrize("dim,splits", [(128, 1), (128, 3), (256, 1), (256, 3)])
@torch.inference_mode()
def test_runtime_strides_and_scales(monkeypatch, dim, splits):
    monkeypatch.setenv("CTI_PREFILL_SPLITS", str(splits))
    monkeypatch.setenv("CTI_PREFILL_MIN_CHUNK", "64")
    queries, lengths, page, hq, hk = (17, 3), (193, 1), 16, 8, 2
    host = metadata(queries, lengths, page, 19)
    gpu = [x.cuda() for x in host]
    geometry = {
        "batch_size": 2,
        "max_query_per_request": 17,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
    }
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    op = CutePagedPrefill(
        workspace, **geometry, page_size=page, dtype=torch.bfloat16, device="cuda"
    )
    torch.manual_seed(42)
    packed = torch.randn(19, 2, page, hk, dim, device="cuda", dtype=torch.bfloat16)
    other = torch.randn(19, 3, page, hk, dim, device="cuda", dtype=torch.bfloat16)
    storage = torch.randn(sum(queries), 3 * hq, dim, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(sum(queries), hq, dim, device="cuda", dtype=torch.bfloat16)
    # Reuse one compiled instance across both strides and every scale sign.
    for q in (storage[:, :hq], storage[:, ::3]):
        for k, v in ((packed[:, 0], packed[:, 1]), (packed[:, 0], other[:, 1])):
            for k_scale, v_scale in ((0.37, 1.9), (1.0, 0.0), (-0.5, 1.0), (0.0, 2.0), (1.0, -1.3)):
                op.run(q, k, v, *gpu, out, k_scale=k_scale, v_scale=v_scale)
                expected = reference(q, k, v, host, queries, lengths, dim**-0.5 * k_scale) * v_scale
                torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.01)
                if v_scale == 0:
                    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("layout", ["heads", "padded", "mixed", "packed"])
@pytest.mark.parametrize("dim,page", [(128, 7), (256, 7), (256, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_aligned_kv_views(monkeypatch, layout, dim, page, dtype):
    from ._attention import kv_views

    monkeypatch.setenv("CTI_PREFILL_SPLITS", "3")
    monkeypatch.setenv("CTI_PREFILL_MIN_CHUNK", "64")
    queries, lengths = (17, 3), (193, 1)
    geometry = {
        "batch_size": 2,
        "max_query_per_request": 17,
        "num_qo_heads": 8,
        "num_kv_heads": 2,
        "head_dim": dim,
    }
    workspace = torch.empty(workspace_size("cuda", **geometry), dtype=torch.uint8, device="cuda")
    op = CutePagedPrefill(workspace, **geometry, page_size=page, dtype=dtype, device="cuda")
    torch.manual_seed(42)
    q = torch.randn(sum(queries), 8, dim, dtype=dtype, device="cuda")
    dense = tuple(torch.randn(19, page, 2, dim, dtype=dtype, device="cuda") for _ in range(2))
    cache = kv_views(dense, layout)
    host = metadata(queries, lengths, page, 19)
    gpu = [x.cuda() for x in host]
    out, dense_out = torch.empty_like(q), torch.empty_like(q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(q, *cache, *gpu, out)
        op.run(q, *dense, *gpu, dense_out)
    graph.replay()
    assert torch.equal(out.view(torch.int16), dense_out.view(torch.int16))
    expected = reference(q, *dense, host, queries, lengths, dim**-0.5)
    torch.testing.assert_close(out.float(), expected, atol=0.01, rtol=0.01)
