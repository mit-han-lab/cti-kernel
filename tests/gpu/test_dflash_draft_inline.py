# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""One captured call crosses split boundaries and reuses op-owned zeroed counters."""

from itertools import product

import pytest
import torch

from benchmark.kernel_tuning.dflash_draft_capture_trace import graph_node_counts
from ctikernels.attention._draft import cute

from ..support.child import run_child
from ._dflash import Fixture
from ._exhaustive import ci_cases

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.parametrize("draft_provider", ["cute"], indirect=True),
]


@pytest.fixture(autouse=True)
def per_request_grid(monkeypatch):
    # These tests pin the per-request split grid (splits, counters, merge levels);
    # the length-aware spread grid has its own suite in test_draft_spread.py.
    monkeypatch.setenv("CTI_DRAFT_SPREAD", "0")


@pytest.mark.parametrize(
    "causal,window,capacity,b,q,inline,short,fused,hq,hk,head_dim",
    ci_cases(
        [
            (causal, window, capacity, b, q, inline, short, fused, 32, 8, 128)
            for (causal, window), capacity, b, q, inline, short, fused in product(
                [(False, None), (True, None), (True, 4096), (False, 2048)],
                [4128, 32768],
                [1, 4],
                [4, 16],
                [0, 1],
                [0, 416],
                [False, True],
            )
        ]
        + [
            (causal, None, capacity, b, q, inline, short, fused, hq, hk, 256)
            for causal, capacity, b, q, inline, short, fused, (hq, hk) in product(
                [False, True],
                [4128, 32768],
                [1, 2, 3, 4],
                [4, 8, 16],
                [0, 1],
                [0, 416],
                [False, True],
                [(16, 4), (8, 2)],
            )
        ],
        core=[
            (False, None, 32768, 1, 4, 1, 416, False, 32, 8, 128),
            (True, None, 32768, 1, 16, 1, 416, True, 32, 8, 128),
            (True, None, 4128, 4, 16, 1, 416, True, 32, 8, 128),
            (True, 4096, 32768, 4, 4, 1, 0, False, 32, 8, 128),
            (False, 2048, 4128, 1, 16, 0, 416, True, 32, 8, 128),
            (True, 4096, 4128, 4, 16, 0, 0, False, 32, 8, 128),
            (True, None, 32768, 4, 16, 1, 416, True, 16, 4, 256),
            (True, None, 4128, 1, 4, 0, 416, False, 8, 2, 256),
            (False, None, 4128, 3, 8, 1, 0, True, 8, 2, 256),
        ],
    ),
)
def test_inline_replay_transitions(
    causal,
    window,
    capacity,
    b,
    q,
    inline,
    short,
    fused,
    hq,
    hk,
    head_dim,
    draft_provider,
    monkeypatch,
):
    monkeypatch.delenv("CTI_DRAFT_MERGE", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", str(inline))
    monkeypatch.setenv("CTI_DRAFT_SHORT", str(short))
    monkeypatch.setenv("CTI_DRAFT_KEY_TILE", "32" if head_dim == 256 else "64")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    f = Fixture(
        b, q, 16, hq, hk, causal, window, c=capacity, provider=draft_provider, head_dim=head_dim
    )
    impl = f.op._implementation
    _, _, rt, splits = cute.configuration(
        batch_size=b,
        query_length=q,
        max_kv_tokens=capacity,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim=head_dim,
        window=window,
        device=f.slots.device,
    )
    assert impl.splits == splits
    assert impl.inline_merge == bool(inline)
    assert (impl._merge is None) == (impl.inline_merge or splits == 1)
    entries = 2 if impl.merge_mode == "distributed" and splits > 1 else sum(impl.merge_levels)
    assert impl._counters.numel() == b * hk * rt * entries
    assert impl.counter_bytes == impl._counters.numel() * 4
    if b == 1:
        assert splits > 1
    assert torch.count_nonzero(impl._counters) == 0
    slots = list(range(b))
    if b > 1:
        slots[-2] = -1
    f.stage([100 - q] * b, slots)
    call = f.op.prepare_and_run if fused else f.op.run

    def run():
        call(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.stream(stream):
        graph.capture_begin()
        run()
        graph.capture_end()
    nodes = graph_node_counts(graph.raw_cuda_graph())
    assert nodes.get("KERNEL", 0) == (1 + int(impl._merge is not None))
    assert nodes.get("MEMSET", 0) == 0
    graph.instantiate()
    torch.cuda.current_stream().wait_stream(stream)
    counters = impl._counters
    if counters.numel():
        assert counters.untyped_storage().data_ptr() != impl._work.untyped_storage().data_ptr()
    lengths = [q, 191, 192, 193, 200, 255, 256, 257, 415, 416, 417, 1000, 4100, 4096 + q, 100]
    if capacity == 32768:
        # Reach the maximum runtime count, including more splits than valid rows.
        lengths[-1:-1] = [16000, capacity]
        if window is None:
            chunk = max(
                ((capacity + impl.splits - 1) // impl.splits + impl.key_tile - 1)
                // impl.key_tile
                * impl.key_tile,
                impl.min_chunk,
            )
            # Tile rounding can leave compiled splits unused at capacity
            # (e.g. 85 splits, 416-key chunks: 79 used).
            assert -(-capacity // chunk) <= impl.splits
    if impl.merge_mode == "distributed" and counters.numel():
        counters[1::2] = 2**31 - 1
    for turn, length in enumerate(lengths):
        # Mixed active lengths make each request's arrival count independent.
        contexts = [length - q] + [lengths[(turn + row) % len(lengths)] - q for row in range(1, b)]
        f.stage(contexts, slots, turn)
        if impl.merge_mode == "distributed" and counters.numel() and length == 1000:
            counters[1::2] = -1  # Exercise the full 32-bit wrap on this replay.
        expected = f.oracle()
        graph.replay()
        torch.testing.assert_close(f.out.float(), expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(impl._arrivals) == 0
        if (
            impl.merge_mode == "distributed"
            and counters.numel()
            and slots[0] != -1
            and length == 1000
        ):
            assert counters[1] == 0
        if b > 1:
            assert torch.count_nonzero(f.out[(b - 2) * q : (b - 1) * q]) == 0
        if fused:
            f.check_metadata()


@pytest.mark.parametrize(
    "b,causal,window,rows,tile,fan_in,head_dim",
    ci_cases(
        [
            (b, causal, window, rows, tile, fan_in, 128)
            for b, (causal, window), (rows, tile, fan_in) in product(
                [1, 4],
                [(False, None), (True, None), (True, 4096), (False, 2048)],
                [(32, 64, 4), (64, 32, 2)],
            )
        ]
        + [
            (b, causal, None, rows, 32, 2, 256)
            for b, causal, rows in product([1, 2, 3, 4], [False, True], [32, 64])
        ],
        core=[
            (b, causal, window, rows, tile, fan_in, 128)
            for b, (causal, window), (rows, tile, fan_in) in product(
                [1, 4],
                [(False, None), (True, None), (True, 4096), (False, 2048)],
                [(32, 64, 4), (64, 32, 2)],
            )
        ]
        + [(4, True, None, 64, 32, 2, 256), (1, False, None, 32, 32, 2, 256)],
    ),
)
def test_hierarchy_boundaries(
    b, causal, window, rows, tile, fan_in, head_dim, draft_provider, monkeypatch
):
    monkeypatch.setenv("CTI_DRAFT_MERGE", "tree")
    # Keep all counts reachable even for W=2048; default SHORT has separate coverage above.
    for key, value in {
        "INLINE": 1,
        "SHORT": 64,
        "MIN_CHUNK": 64,
        "SPLITS": 23,
        "ROWS": rows,
        "KEY_TILE": tile,
        "KEY_GROUPS": 2,
        "STAGES": 2,
    }.items():
        monkeypatch.setenv(f"CTI_DRAFT_{key}", str(value))
    f = Fixture(
        b, 9, 16, 16, 4, causal, window, c=23 * 64, provider=draft_provider, head_dim=head_dim
    )
    impl = f.op._implementation
    assert impl.merge_fan_in == fan_in
    slots = list(range(b))
    if b > 1:
        slots[1] = -1
    counts = sorted({1, 2, fan_in, fan_in + 1, fan_in * fan_in, fan_in * fan_in + 1, 23})
    lengths = [63, 64, 65] + [(count - 1) * 64 + 1 if count > 1 else 9 for count in counts]
    f.stage([0] * b, slots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        f.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        f.op.prepare_and_run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
    torch.cuda.current_stream().wait_stream(stream)
    # Descending transitions exercise counters that higher levels no longer use.
    for turn, length in enumerate(lengths + lengths[::-1]):
        contexts = [length - f.q] * b
        f.stage(contexts, slots, turn)
        expected = f.oracle()
        graph.replay()
        torch.testing.assert_close(f.out.float(), expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(impl._counters) == 0
    f.stage([0] * b, [-1] * b)
    graph.replay()
    assert torch.count_nonzero(f.out) == 0
    assert torch.count_nonzero(impl._counters) == 0


@pytest.mark.parametrize("mode", ["distributed", "tree"])
def test_concurrent_stream_forward_progress(draft_provider, monkeypatch, mode):
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    import subprocess
    import sys

    # The independent load is bounded; a deadlock is reported by the parent without a CUDA
    # sync, and a deadlocked child is killed (or, if unkillable, ends the run).
    try:
        child = run_child([sys.executable, "-m", "tests.gpu._dflash_draft_progress"], timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail("concurrent DRAFT calls exceeded the 120 s forward-progress timeout")
    assert child.returncode == 0, child.stderr[-3000:]
