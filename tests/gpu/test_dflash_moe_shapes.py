# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Qwen3.5-35B-A3B VERIFY and TP2: GQA8, D256, mutable full-attention graphs.

Contexts exclude the Q appended keys; exact capacity is context + Q. Keep only
three DRAFT and two fixed-query cases in CI; qualify the full sweep per device.
"""

import os
from itertools import product

import pytest
import torch

from ctikernels.attention._draft import cute, dispatch

from . import test_fixed_query_attention as fixed_query
from ._dflash import Fixture
from ._exhaustive import ci_cases
from .test_dflash_draft_attention import ATOL, RTOL, capture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
HEADS = [(16, 2), (8, 1)]
CASES = [
    (b, q, hq, hk, context, context + q if exact else 32768, True)
    for b, q, (hq, hk), context, exact in product(
        [1, 2, 4], [4, 8, 16], HEADS, [100, 1000], [True, False]
    )
] + [(1, 4, hq, hk, 100, 32768, False) for hq, hk in HEADS]


@pytest.fixture(autouse=True)
def default_tuning_and_precision(monkeypatch):
    for name in list(os.environ):
        if name.startswith("CTI_DRAFT_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


@pytest.mark.parametrize(
    "b,q,hq,hk,context,capacity,causal,provider",
    ci_cases(
        CASES,
        ["cute", "flashinfer", None],
        core=[
            (1, 4, 16, 2, 100, 104, True, "cute"),
            (2, 8, 8, 1, 100, 32768, True, "flashinfer"),
            (4, 16, 16, 2, 1000, 32768, True, None),
        ],
    ),
)
def test_draft_moe_graph(b, q, hq, hk, context, capacity, causal, provider):
    geometry = {
        "batch_size": b,
        "query_length": q,
        "max_kv_tokens": capacity,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": 256,
        "causal": causal,
        "window": None,
        "device": "cuda",
    }
    supported = True
    try:
        cute.check_support(**geometry)
    except NotImplementedError as error:
        supported = False
        if provider == "cute":
            pytest.skip(str(error))
    expected_provider = provider or (
        "cute" if supported and not dispatch.prefers_flashinfer(**geometry) else "flashinfer"
    )
    if expected_provider == "flashinfer":
        pytest.importorskip("flashinfer")
    f = Fixture(b, q, 1, hq, hk, causal, None, c=capacity, provider=provider, head_dim=256)
    assert (
        f.op.provider
        == {
            "cute": "cute-splitkv",
            "flashinfer": "flashinfer-fa2-device-schedule",
        }[expected_provider]
    )
    f.stage([context] * b, list(range(b)))
    graph = capture(f)
    torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
    # Short -> inactive -> original length, with new Q, positions and physical
    # indices on every replay; the last turn reactivates every exact-B row.
    for turn, length in enumerate((min(100, context // 2), 0, context), 1):
        slots = list(reversed(range(b))) if turn != 2 else [-1] * b
        lengths = [max(0, length - row) for row in range(b)]
        f.stage(lengths, slots, turn)
        graph.replay()
        torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
        f.check_metadata()
        if turn == 2:
            assert torch.count_nonzero(f.out) == 0


@pytest.mark.parametrize(
    "b,q,hq,hk,context,capacity,causal",
    ci_cases(
        CASES,
        core=[
            (1, 8, 8, 1, 100, 108, True),
            (1, 4, 16, 2, 100, 32768, False),
        ],
    ),
)
def test_fixed_query_moe_graph(b, q, hq, hk, context, capacity, causal):
    pytest.importorskip("flashinfer")
    shape = fixed_query.AttentionShape(
        b, q, hq, hk, 256, 1, capacity, torch.bfloat16, (q, min(100, context) + q, context + q)
    )
    fixed_query.test_fixed_query_split_kv_graph_with_changing_lengths(shape, causal)
