# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DF4 direct tables: CSR bitwise parity, poison isolation and mutable graph inputs."""

from itertools import product

import pytest
import torch

from ctikernels.attention.inference import CacheTableDraftAttention

from .._dflash_table_reference import table_csr
from ._dflash import Fixture
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
MASKS = [(False, None), (True, None), (True, 4096), (False, 2048)]


@pytest.mark.parametrize(
    "b,q,hq,hk,dim,causal,window,metadata_index",
    ci_cases(
        [
            (b, q, hq, hk, dim, causal, window)
            for b, q, (hq, hk, dim), (causal, window) in product(
                [1, 2, 3, 4],
                [4, 5, 8, 9, 16],
                [(32, 8, 128), (16, 4, 256), (16, 2, 256), (8, 1, 256)],
                MASKS,
            )
        ],
        ["request", "row"],
        core=[
            (1, 4, 32, 8, 128, False, None, "request"),
            (4, 5, 32, 8, 128, True, 4096, "row"),
            (3, 9, 16, 4, 256, False, 2048, "request"),
            (2, 16, 16, 2, 256, True, None, "row"),
            (1, 4, 8, 1, 256, True, None, "request"),
        ],
    ),
)
def test_table_csr_parity_and_graph(
    b, q, hq, hk, dim, causal, window, metadata_index, draft_provider, monkeypatch
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # Table attention always merges inline; pin the CSR reference to the same merge so
    # rows where CSR defaults to the legacy merge (DW1 draft pairs) stay bitwise.
    monkeypatch.setenv("CTI_DRAFT_MERGE", "distributed")
    # Table attention keeps the per-request split grid; so does its CSR reference.
    monkeypatch.setenv("CTI_DRAFT_SPREAD", "0")
    maximum = (window + 17) if window else 257
    f = Fixture(
        b, q, 1, hq, hk, causal, window, c=maximum + q, provider=draft_provider, head_dim=dim
    )
    # All physical rows, including guard rows, contain finite random values.
    f.k.normal_()
    f.v.normal_()
    guard = b * f.c
    rows = 2 * b + 3
    requests = torch.empty(b, dtype=torch.int32, device="cuda")
    frontier = torch.empty(rows, dtype=torch.int64, device="cuda")
    done = torch.empty(rows, dtype=torch.bool, device="cuda")
    frontier_row = torch.empty(b, dtype=torch.int64, device="cuda")
    done_row = torch.empty(b, dtype=torch.bool, device="cuda")
    backing = torch.empty((rows, maximum + 15), dtype=torch.int32, device="cuda")
    table = backing[:, : maximum + 7]
    geometry = {
        "batch_size": b,
        "query_length": q,
        "max_seq_len": maximum,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
        "device": "cuda",
        "provider": draft_provider,
    }
    arguments = {k: v for k, v in geometry.items() if k not in ("batch_size", "device")}

    def construct(mode):
        work = [
            torch.empty(n, dtype=torch.uint8, device="cuda")
            for n in CacheTableDraftAttention.workspace_size(**geometry)
        ]
        return CacheTableDraftAttention(
            *work,
            request_indices=requests,
            frontier_table=frontier_row if mode == "row" else frontier,
            done_table=done_row if mode == "row" else done,
            cache_index_table=table,
            guard_start=guard,
            metadata_index=mode,
            **arguments,
        )

    op = construct(metadata_index)
    request_op = construct("request") if metadata_index == "row" else None
    out = torch.empty_like(f.out)
    if draft_provider == "triton":
        assert op.kernels_per_call == 1 + (f.op._implementation._segments > 1)
    else:
        assert op.kernels_per_call == (1 if draft_provider == "cute" else 4)
    if draft_provider == "cute":
        for attribute in (
            "splits",
            "key_tile",
            "key_groups",
            "stages",
            "chunk_cap",
            "pieces",
            "merge_mode",
        ):
            assert getattr(op._implementation, attribute) == getattr(
                f.op._implementation, attribute
            )

    contexts = [0, 1, maximum - 1, -1, maximum, maximum + 5, 2**62, -(2**62), -(2**63), 2**63 - 1]
    if window:
        contexts += [window - q - 1, window - q, window - q + 1, window - 1, window, window + 1]

    def stage(turn):
        cpu_requests = ((torch.arange(b, dtype=torch.int32) * 2 + 1 + turn) % rows).roll(turn)
        cpu_frontier = torch.full((rows,), -(2**63), dtype=torch.int64)
        cpu_done = torch.ones(rows, dtype=torch.bool)
        mapping = torch.randint(0, guard, (rows, maximum + 15), dtype=torch.int32)
        allowed = torch.zeros_like(mapping, dtype=torch.bool)
        for row, r in enumerate(cpu_requests.tolist()):
            value = contexts[(turn + row) % len(contexts)]
            cpu_frontier[r] = value
            cpu_done[r] = (turn + row) % 7 == 3
            if not cpu_done[r] and 0 <= value < maximum:
                allowed[r, : min(value + q, maximum)] = True
        ptr, indices, positions = table_csr(
            cpu_requests, cpu_frontier, cpu_done, mapping, guard, maximum, q
        )
        requests.copy_(cpu_requests)
        frontier.copy_(cpu_frontier)
        done.copy_(cpu_done)
        frontier_row.copy_(cpu_frontier[cpu_requests.long()])
        done_row.copy_(cpu_done[cpu_requests.long()])
        backing.copy_(mapping)
        f.cpu_slots = torch.arange(b, dtype=torch.int32)
        f.cpu_positions, f.cpu_token_ptr = positions, ptr
        f.cpu_token_idx = torch.full((b * f.c,), 2**31 - 1, dtype=torch.int32)
        f.cpu_token_idx[: indices.numel()] = indices
        f.slots.copy_(f.cpu_slots)
        f.positions.copy_(positions)
        f.token_ptr.copy_(ptr)
        f.token_idx.copy_(f.cpu_token_idx)
        return mapping.masked_fill(~allowed, 2**31 - 1), indices.numel()

    stage(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for turn in range(len(contexts)):
        poisoned, count = stage(turn)
        op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=out)
        eager = out.clone()
        f.run()
        assert torch.equal(eager, f.out)
        torch.testing.assert_close(eager.float(), f.oracle(), atol=2e-2, rtol=2e-2)
        if request_op is not None:
            request_op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
            assert torch.equal(eager, f.out)
        backing.copy_(poisoned)
        if draft_provider == "flashinfer":
            private_indices = op._implementation._inputs[5]
            private_indices.fill_(2**31 - 1)
        graph.replay()
        assert torch.equal(out, eager)
        op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
        assert torch.equal(out, f.out)
        if draft_provider == "flashinfer":
            assert torch.all(private_indices[count:] == 2**31 - 1)


@pytest.mark.exhaustive
@pytest.mark.parametrize("mode", ["tree", "distributed"])
@pytest.mark.parametrize("metadata_index", ["request", "row"])
def test_table_piece_loop(mode, metadata_index, monkeypatch, draft_provider):
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "2")
    monkeypatch.setenv("CTI_DRAFT_SHORT", "0")
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", "64")
    monkeypatch.setenv("CTI_DRAFT_PIECES", "1")
    test_table_csr_parity_and_graph(
        4, 8, 32, 8, 128, True, 4096, metadata_index, draft_provider, monkeypatch
    )
