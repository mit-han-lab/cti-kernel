# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Any fixed Q: provider parity, bottom-right masks and changing-length graphs."""

import pytest
import torch

from ctikernels.attention._draft import cute
from ctikernels.attention.inference import FlashInferFixedQueryAttention

from .._dflash_reference import chunked_draft_attention_reference
from ._dflash import Fixture
from .test_dflash_draft_window import capture, compare_replay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
QUERIES = (1, 2, 33, 64, 129, 512)
MASKS = ((False, None), (True, None), (True, 64), (False, 7), (False, 2048))


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


class QueryCase(Fixture):
    def oracle(self):
        # Bound FP32 scratch even for Q4096, retaining the shared independent oracle.
        return chunked_draft_attention_reference(
            self.query,
            self.k,
            self.v,
            slots=self.cpu_slots,
            positions=self.cpu_positions,
            token_indptr=self.cpu_token_ptr,
            token_indices=self.cpu_token_idx,
            query_length=self.q,
            causal=self.causal,
            window=self.window,
            chunk_size=256,
        )


def exercise(f):
    f.stage([0] * f.b, list(range(f.b)))
    graph = capture([f])
    misses = cute.compile_kernels.cache_info().misses
    for turn in (1, 2):
        # Include C, Q, partial final pages, and inactivity without reallocating.
        contexts = [f.c - f.q if row == 0 else 17 * row for row in range(f.b)]
        if turn == 2:
            contexts = [f.c - f.q - 1] * f.b
        slots = list(range(f.b))
        if f.b > 1 and turn == 1:
            slots[-1] = -1
        f.stage(contexts, slots, turn, base=2**33)
        compare_replay([f], graph)
    assert cute.compile_kernels.cache_info().misses == misses


def core_rows():
    # One unmarked row for every (Q, provider, D), rotating masks/B/P/GQA.
    for qi, query in enumerate(QUERIES):
        for di, dim in enumerate((64, 128, 256)):
            causal, window = MASKS[(qi + di) % len(MASKS)]
            for provider in (None, "cute", "flashinfer", "triton"):
                chosen = window
                if (
                    provider == "flashinfer"
                    and not causal
                    and window is not None
                    and window < query
                ):
                    chosen = 2048
                yield pytest.param(
                    query,
                    dim,
                    provider,
                    causal,
                    chosen,
                    1 if (qi + di) % 2 == 0 else 3,
                    1 if qi % 2 == 0 else 16,
                    (64, 1) if (qi + di) % 3 == 0 else (32, 8),
                    id=f"q{query}-hd{dim}-{provider or 'auto'}-w{chosen}-causal{causal}",
                )


@pytest.mark.parametrize("query,dim,provider,causal,window,batch,page,heads", list(core_rows()))
def test_any_query(query, dim, provider, causal, window, batch, page, heads):
    f = QueryCase(
        batch,
        query,
        page,
        *heads,
        causal,
        window,
        c=max(1024, query + 64),
        provider=provider,
        head_dim=dim,
    )
    narrow = not causal and window is not None and window < query
    if provider == "triton":
        assert f.op.provider == "triton-splitkv"
    elif narrow:
        assert f.op.provider == "cute-splitkv"
    elif provider is None and query > 32:
        from ctikernels.attention._draft import dispatch

        c = max(1024, query + 64)
        fa2 = dispatch.prefers_flashinfer(
            batch_size=batch,
            query_length=query,
            max_kv_tokens=c,
            num_qo_heads=heads[0],
            num_kv_heads=heads[1],
            causal=causal,
            window=None if window is not None and window >= c else window,
            device=f.op._device,
            head_dim=dim,
        )
        assert f.op.provider == ("flashinfer-fa2-device-schedule" if fa2 else "cute-splitkv")
    exercise(f)


@pytest.mark.exhaustive
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("causal,window", MASKS)
def test_extended_masks(query, causal, window):
    # Cover every Q/mask while rotating provider and D instead of a cold-compile
    # Cartesian product. The core matrix already covers every Q/provider/D.
    qi, mi = QUERIES.index(query), MASKS.index((causal, window))
    dim = (64, 128, 256)[(qi + mi) % 3]
    provider = (None, "cute", "flashinfer")[(qi + 2 * mi) % 3]
    narrow = not causal and window is not None and window < query
    if narrow and provider == "flashinfer":
        provider = None
    f = QueryCase(1, query, 16, 32, 8, causal, window, c=4096, provider=provider, head_dim=dim)
    if narrow:
        assert f.op.provider == "cute-splitkv"
    exercise(f)


@pytest.mark.parametrize("query", (129, 512))
@pytest.mark.parametrize("causal,window", ((False, 64), (False, 2048), (True, 64), (True, 2048)))
@pytest.mark.parametrize("page,dim", ((1, 128), (1, 256), (16, 128), (16, 256)))
def test_sliding_query_exceeds_tile(query, causal, window, page, dim):
    # Full and partial final pages, eager and graph replay against FP32.
    # W<Q needs the custom mask's right bound; the other rows exercise FA2.
    provider = "flashinfer-mask" if not causal and window < query else "flashinfer"
    f = QueryCase(1, query, page, 32, 8, causal, window, c=4096, provider=provider, head_dim=dim)
    f.stage([4096 - query], [0])
    graph = capture([f])
    if provider == "flashinfer-mask":
        from .test_draft_custom_mask import compare

        check = lambda: compare(f, graph, page)
    else:
        assert f.op.provider == "flashinfer-fa2-device-schedule"
        check = lambda: compare_replay([f], graph)
    for turn, length in enumerate((4096, 4095, 4081), 1):
        f.stage([length - query], [0], turn, base=2**33)
        check()


@pytest.mark.parametrize("provider", (None, "cute", "flashinfer", "triton"))
@pytest.mark.parametrize("dim", (64, 128, 256))
def test_large_query(provider, dim):
    # Multiple query tiles, merge chunks and padded CTA chunks on every provider.
    exercise(QueryCase(2, 4096, 16, 32, 8, True, 64, c=8192, provider=provider, head_dim=dim))


@pytest.mark.parametrize("query", (1, 33, 64, 129, 512))
@pytest.mark.parametrize(
    "causal,window", ((False, None), (True, None), (False, 2048), (True, 64), (True, 2048))
)
def test_fixed_query(query, causal, window):
    capacity = 4096
    f = QueryCase(3, query, 16, 32, 8, causal, window, c=capacity, provider="flashinfer")
    f.op = FlashInferFixedQueryAttention(
        *f.workspaces,
        query_indptr=f.query_ptr,
        kv_indptr=f.page_ptr,
        kv_indices=f.page_idx,
        last_page_lengths=f.last,
        query_length=query,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        dtype=torch.bfloat16,
        causal=causal,
        window_left=-1 if window is None else window - 1,
    )

    def stage(contexts, turn):
        f.stage(contexts, [0, 1, 2], turn)
        for dst, src in (
            (f.query_ptr, f.expected_query_ptr),
            (f.page_ptr, f.expected_page_ptr),
            (f.page_idx, f.expected_pages),
            (f.last, f.expected_last),
        ):
            dst.copy_(src)

    stage([0, 1, 17], 0)
    graph = capture([f])
    contexts_by_turn = (
        [capacity - query, 0, 33],
        [0, capacity - 1 - query, 17],
        [0, capacity - 15 - query, 17],
    )
    for turn, contexts in enumerate(contexts_by_turn, 1):
        stage(contexts, turn)
        compare_replay([f], graph)
