# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe split tails, window boundaries and changing-metadata graph replay."""

from itertools import product

import pytest
import torch

from ctikernels.attention._draft import cute

from ._dflash import Fixture
from ._exhaustive import ci_cases

ATOL = RTOL = 2e-2
QUERIES = [4, 8, 16, 2, 9, 32]
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("draft_provider", ["cute"], indirect=True),
]


@pytest.fixture(autouse=True)
def per_request_grid(monkeypatch):
    # These tests pin the per-request split grid (splits, counters, merge levels);
    # the length-aware spread grid has its own suite in test_draft_spread.py.
    monkeypatch.setenv("CTI_DRAFT_SPREAD", "0")


def _arrivals(implementation):
    # Distributed merge stores (arrival, generation flag) pairs; flags change by design.
    counters = implementation._counters
    if getattr(implementation, "merge_mode", None) == "distributed":
        return counters.view(-1, 2)[:, 0]
    return counters


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("hq,hk", [(4, 1), (16, 4)])
@pytest.mark.parametrize("mode", ["legacy", "tree", "distributed"])
def test_uniform_attention_preserves_value_columns(
    head_dim, hq, hk, mode, draft_provider, monkeypatch
):
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "7")
    f = Fixture(2, 4, 16, hq, hk, False, None, provider=draft_provider, head_dim=head_dim)
    columns = torch.arange(head_dim, device="cuda", dtype=torch.float32) / head_dim
    columns = columns.to(torch.bfloat16)
    f.k[1::2].zero_()
    f.v[1::2].copy_(columns.expand_as(f.v[1::2]))
    # Direct short output and split/merge output must preserve every column.
    for context in (28, 4092):
        f.stage([context, context], [0, 1])
        f.run()
        torch.testing.assert_close(f.out, columns.expand_as(f.out), atol=0, rtol=0)
        f.check_metadata()


@pytest.mark.parametrize(
    "b,q,hq,hk,p,causal,window,splits,head_dim",
    ci_cases(
        [
            (b, q, hq, hk, p, causal, window, splits, 128)
            for b, q, (hq, hk), p, (causal, window), splits in product(
                [1, 2, 3, 4],
                QUERIES,
                [(32, 8), (16, 4)],
                [1, 16],
                [(False, None), (True, 4096), (False, 2048)],
                [1, 2, 7, None],
            )
        ]
        + [
            (3, q, 32, 8, p, True, None, splits, 128)
            for q, p, splits in product([4, 16], [1, 16], [1, 2, 7, None])
        ]
        + [
            (b, q, hq, hk, p, causal, None, splits, 256)
            for b, q, (hq, hk), p, causal, splits in product(
                [1, 2, 3, 4],
                [4, 8, 16],
                [(16, 4), (8, 2)],
                [1, 16],
                [False, True],
                [1, 2, 7, None],
            )
        ],
        core=[
            (1, 4, 16, 4, 1, False, None, None, 128),
            (4, 16, 32, 8, 16, True, 4096, None, 128),
            (4, 4, 16, 4, 16, False, 2048, 7, 128),
            (3, 16, 32, 8, 16, True, None, None, 128),
            (4, 16, 16, 4, 16, True, None, None, 256),
            (2, 8, 8, 2, 1, False, None, 7, 256),
        ],
    ),
)
def test_split_boundaries_and_replay(
    b, q, hq, hk, p, causal, window, splits, head_dim, draft_provider, monkeypatch
):
    if splits is None:
        monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    else:
        monkeypatch.setenv("CTI_DRAFT_SPLITS", str(splits))
    f = Fixture(b, q, p, hq, hk, causal, window, provider=draft_provider, head_dim=head_dim)
    assert f.op.provider == "cute-splitkv"
    if splits is not None:
        # Inline preserves explicit splits; only legacy applies the capacity floor.
        expected_splits = splits
        if not f.op._implementation.inline_merge:
            # Legacy keeps the tuned split floor of the selected capacity tier.
            expected_splits = cute.configuration(
                batch_size=b,
                query_length=q,
                max_kv_tokens=f.c,
                num_qo_heads=hq,
                num_kv_heads=hk,
                window=window,
                device=torch.device("cuda"),
                head_dim=head_dim,
                causal=causal,
            )[3]
            assert expected_splits >= splits
        assert f.op._implementation.splits == expected_splits
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        f.stage([128] * b, list(range(b)))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            f.run()
        stream.synchronize()
        # Capture run alone: page CSR is deliberately not refreshed on replay.
        run_graph, complete_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(run_graph, stream=stream):
            f.op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
        with torch.cuda.graph(complete_graph, stream=stream):
            f.run()
        torch.cuda.current_stream().wait_stream(stream)
        # Short lengths leave whole splits empty; boundary contexts exercise partial
        # visibility within a split. stage() permutes physical pages on every turn.
        contexts = [
            0,
            1,
            415 - q,
            416 - q,
            417 - q,
            2047,
            2048,
            2049,
            4095,
            4096,
            4097,
            f.c - q - 1,
            f.c - q,
        ]
        saved_k, saved_v = f.k.flatten(0, 1).clone(), f.v.flatten(0, 1).clone()
        cases = [(context, list(range(b))) for context in contexts]
        cases += [
            (128, list(range(b - 1)) + [-1]),
            (4096, [-1] * b),
            (2048, list(reversed(range(b)))),
        ]
        for turn, (context, slots) in enumerate(cases):
            lengths = [context] + [contexts[(turn + row) % len(contexts)] for row in range(1, b)]
            f.stage(lengths, slots, turn)
            physical = f.cpu_token_idx[: int(f.cpu_token_ptr[-1])].long().to("cuda")
            f.k.fill_(float("nan"))
            f.v.fill_(float("nan"))
            f.k.flatten(0, 1).index_copy_(0, physical, saved_k.index_select(0, physical))
            f.v.flatten(0, 1).index_copy_(0, physical, saved_v.index_select(0, physical))
            expected = f.oracle()
            for graph in (run_graph, complete_graph):
                f.out.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
                assert torch.count_nonzero(_arrivals(f.op._implementation)) == 0
                for row, slot in enumerate(slots):
                    if slot == -1:
                        assert torch.count_nonzero(f.out[row * q : (row + 1) * q]) == 0
            f.check_metadata()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "b,q,causal,window",
    ci_cases(
        [1, 3],
        [5, 16],
        [(False, None), (True, 4096), (False, 2048), (True, None)],
        core=[(1, 5, False, None), (3, 16, True, 4096), (3, 5, False, 2048), (1, 16, True, None)],
    ),
)
def test_multiple_pieces_and_replay(b, q, causal, window, draft_provider, monkeypatch):
    # Keep every compile argument except chunk_cap fixed across the env change.
    # This split count satisfies the floor for both caps; the minimum forces
    # multiple pieces even for sliding masks whose unions fit within that floor.
    monkeypatch.setenv("CTI_DRAFT_SHORT", "0")
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "65")
    monkeypatch.setenv("CTI_DRAFT_MIN_CHUNK", "256")
    monkeypatch.setenv("CTI_DRAFT_KEY_TILE", "64")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    # Use a cold production cache so both cap values must trigger compilation.
    compile_kernels = cute.compile_kernels
    compile_kernels.cache_clear()
    previous_attention, previous_misses = None, 0
    for cap in (64, 128):
        monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", str(cap))
        f = Fixture(b, q, 16, 32, 8, causal, window, c=8192, provider=draft_provider)
        assert f.op.provider == "cute-splitkv"
        implementation = f.op._implementation
        assert implementation.splits == 65
        assert implementation.key_tile == 64
        assert implementation.min_chunk == 256
        assert implementation.chunk_cap == cap
        # Distributed construction may also compile its tree fallback: count new misses.
        assert compile_kernels.cache_info().misses > previous_misses
        previous_misses = compile_kernels.cache_info().misses
        assert implementation._attention is not previous_attention
        previous_attention = implementation._attention

        slots = [0] if b == 1 else [0, -1, 2]
        f.stage([100 - q] * b, slots)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            f.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            f.run()
        torch.cuda.current_stream().wait_stream(stream)

        # stage() takes context lengths, so subtract Q to get these exact KV lengths.
        # Replays grow and shrink across piece boundaries and permute physical pages.
        lengths = [100, 700, 1500, 5000, 2 * cap + 1, 2 * cap, 2 * cap - 1, cap + 1, cap, cap - 1]
        for turn in range(len(lengths)):
            contexts = [lengths[(turn + row) % len(lengths)] - q for row in range(b)]
            f.stage(contexts, slots, turn)
            expected = f.oracle()
            f.run()
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
            f.out.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
            if b == 3:
                assert torch.count_nonzero(f.out[q : 2 * q]) == 0
            f.check_metadata()


@pytest.mark.parametrize(
    "minimum,q", ci_cases([128, 256, 512], [2, 8, 9, 16, 32], core=[(128, 16)])
)
def test_runtime_single_split_and_merge_tail(minimum, q, draft_provider, monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "43")
    monkeypatch.setenv("CTI_DRAFT_MIN_CHUNK", str(minimum))
    f = Fixture(4, q, 1, 32, 8, False, None, provider=draft_provider)
    # One graph transitions between direct output, multi-split merge and inactive rows.
    f.stage([0] * 4, list(range(4)))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        f.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        f.run()
    torch.cuda.current_stream().wait_stream(stream)
    for turn, lengths in enumerate(
        ([minimum - 1, minimum, minimum + 1, 4096 + q], [4096 + q, q, 4096 + q, q])
    ):
        slots = [0, 1, 2, -1] if turn == 0 else [-1, 1, 2, 3]
        f.stage([length - q for length in lengths], slots, turn)
        # Inactive CSR entries are intentionally not valid addresses.
        inactive = slots.index(-1)
        if inactive == 0:
            f.token_ptr[0] = -(2**30)
        else:
            f.token_ptr[-1] = -(2**30)
        expected = f.oracle()
        graph.replay()
        torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
        assert torch.count_nonzero(f.out[inactive * q : (inactive + 1) * q]) == 0
