# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Length-aware CuTe split grid: one row of B * splits CTAs follows the runtime lengths."""

import pytest
import torch

from ctikernels.attention._draft import cute

from ._dflash import Fixture
from .test_dflash_draft_window import capture, compare_replay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def spread_environment(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    # Force the spread grid: by default it is off where S = B (no room to spread).
    monkeypatch.setenv("CTI_DRAFT_SPREAD", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def inner(op):
    assert isinstance(op._implementation, cute.Provider)
    return op._implementation


def contexts(b, c, q, pattern):
    if pattern == "uniform":
        return [c - q] * b
    if pattern == "one_long":
        return [c - q] + [5 + 7 * row for row in range(b - 1)]
    # Mixed: long, medium, short and empty requests in a fixed order.
    lengths = [c - q, c // 3, 300, 0, c // 2, 64, 1000, c - 2 * q] * ((b + 7) // 8)
    return lengths[:b]


@pytest.mark.parametrize("merge", ["distributed", "tree"])
@pytest.mark.parametrize(
    "b,hq,hk,head_dim,causal,window",
    [
        (2, 8, 2, 128, False, None),
        (3, 32, 8, 128, False, None),
        (4, 16, 4, 256, True, None),
        (5, 8, 2, 64, True, 2048),
        (8, 16, 4, 128, True, None),
        (3, 64, 1, 256, True, 2048),
        (16, 16, 4, 128, True, None),
        (32, 8, 2, 128, False, None),
    ],
    ids=lambda v: str(v),
)
def test_spread_ragged(monkeypatch, merge, b, hq, hk, head_dim, causal, window):
    monkeypatch.setenv("CTI_DRAFT_MERGE", merge)
    q = 8 if hq // hk < 64 else 2
    c = 16384
    f = Fixture(b, q, 16, hq, hk, causal, window, c=c, provider="cute", head_dim=head_dim)
    provider = inner(f.op)
    assert provider.spread == b
    slots = list(range(b))
    f.stage(contexts(b, c, q, "uniform"), slots)
    graph = capture([f])
    for turn, pattern in enumerate(("one_long", "mixed", "uniform"), start=1):
        lengths = contexts(b, c, q, pattern)
        active = [-1 if row == 1 and pattern == "mixed" else row for row in slots]
        f.stage(lengths, active, turn)
        compare_replay([f], graph)


@pytest.mark.parametrize("b", [4, 40])
@pytest.mark.parametrize("c", [512, 4160, 16384])
def test_spread_uniform_matches_per_request_grid(monkeypatch, b, c):
    # Uniform lengths give the same chunks and counts under both grids: same bits.
    outputs = []
    for spread in ("0", "1"):
        monkeypatch.setenv("CTI_DRAFT_SPREAD", spread)
        f = Fixture(b, 16, 16, 32, 8, False, None, c=c, provider="cute")
        assert inner(f.op).spread == (b if spread == "1" else 0)
        torch.manual_seed(0)
        f.stage([c - 16] * b, list(range(b)))
        f.run()
        outputs.append(f.out.clone())
        del f
    assert torch.equal(outputs[0], outputs[1])


@pytest.mark.parametrize(
    "b,hq,hk,head_dim,causal,window",
    [
        (33, 32, 8, 128, True, None),
        (64, 32, 8, 128, False, None),
        (84, 32, 8, 128, False, None),
        (40, 16, 4, 256, True, None),
        (48, 8, 2, 64, True, 2048),
        (36, 64, 1, 256, True, 2048),
        # More than WIDE_CACHED_GROUPS groups: the runtime-loop schedule.
        (300, 8, 1, 64, False, None),
    ],
)
def test_spread_wide_ragged(monkeypatch, b, hq, hk, head_dim, causal, window):
    monkeypatch.delenv("CTI_DRAFT_MERGE", raising=False)
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    q = 8 if hq // hk < 64 else 2
    # Fixture keeps fragmentation gaps in K/V; bound the largest case's allocation.
    c = 8192
    f = Fixture(b, q, 16, hq, hk, causal, window, c=c, provider="cute", head_dim=head_dim)
    provider = inner(f.op)
    assert provider.spread == b
    assert provider.merge == "tree"
    assert provider.fallbacks == ()
    slots = list(range(b))
    f.stage(contexts(b, c, q, "uniform"), slots)
    graph = capture([f])
    for turn, pattern in enumerate(("uniform", "one_long", "mixed"), start=1):
        lengths = contexts(b, c, q, pattern)
        # The last group owns the single long request; mixed crosses both boundaries.
        if pattern == "one_long":
            lengths[0], lengths[-1] = lengths[-1], lengths[0]
        active = [-1 if row % 7 == 1 and pattern == "mixed" else row for row in slots]
        f.stage(lengths, active, turn)
        compare_replay([f], graph)


def test_spread_wide_long_context(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_MERGE", raising=False)
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    b, q, c = 40, 8, 131072
    f = Fixture(b, q, 16, 8, 2, False, None, c=c, provider="cute", head_dim=64)
    assert inner(f.op).spread == b
    assert inner(f.op).merge == "tree"
    lengths = [c - q, 200] + [0] * (b - 2)
    slots = [0, 1] + [-1] * (b - 2)
    f.stage(lengths, slots)
    graph = capture([f])
    compare_replay([f], graph)
    lengths[0], lengths[-1] = 0, c - q
    slots[0], slots[-1] = -1, b - 1
    f.stage(lengths, slots, 1)
    compare_replay([f], graph)


@pytest.mark.parametrize("c", [131072])
def test_spread_long_context(c):
    f = Fixture(3, 16, 16, 32, 8, False, None, c=c, provider="cute")
    assert inner(f.op).spread == 3
    f.stage([c - 16, 200, 30], [0, 1, 2])
    graph = capture([f])
    f.stage([90000, c - 16, 0], [2, 0, 1], 1)
    compare_replay([f], graph)


def test_spread_fused_publish():
    # prepare_and_run: each request's CTAs publish its page CSR under the spread grid.
    f = Fixture(4, 8, 16, 8, 2, True, None, c=8256, provider="cute")
    assert inner(f.op).spread == 4

    def fused():
        return f.op.prepare_and_run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

    f.run = fused
    f.stage([8248, 0, 92, 4000], [0, 1, 2, 3])
    graph = capture([f])
    for turn, (lengths, slots) in enumerate(
        (([8248, 5, 0, 31], [3, -1, 0, 1]), ([100, 8248, 8248, 0], [0, 1, 2, 3])), start=1
    ):
        f.stage(lengths, slots, turn)
        compare_replay([f], graph)


def test_spread_overflow_without_pieces(monkeypatch):
    # S = 6 (B=2, s=3), C=3072 fits one 1024-key index buffer per split (no piece loop).
    # Lengths 3072 and 2200 round to 4 + 3 > 6 chunks, and the refit chunk (1056) exceeds
    # the buffer, so each request keeps its own split; then it fits again.
    monkeypatch.setenv("CTI_DRAFT_SPLITS", "3")
    monkeypatch.setenv("CTI_DRAFT_CHUNK_CAP", "1024")
    monkeypatch.setenv("CTI_DRAFT_SHORT", "0")
    f = Fixture(2, 8, 16, 8, 2, False, None, c=3072, provider="cute")
    provider = inner(f.op)
    assert (provider.spread, provider.splits, provider.pieces) == (2, 6, False)
    f.stage([3064, 2192], [0, 1])
    graph = capture([f])
    for turn, lengths in enumerate(([3064, 2192], [2192, 3064], [3064, 100]), start=1):
        f.stage(lengths, [0, 1], turn)
        compare_replay([f], graph)
