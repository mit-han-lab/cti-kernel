# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FA2 custom masks: FP32 oracle, fragmented pages and restaged CUDA graphs.

The default covering grid avoids the full Cartesian product; extra combinations
are exhaustive. Explicit selection exercises this fallback on SM120/SM121 too.
"""

import os

import pytest
import torch

from ctikernels.attention._draft import cute

from ._dflash import SENTINEL, Fixture
from .test_dflash_draft_window import ATOL, RTOL, capture
from .test_dflash_long_context import PAGE, LongContextCase

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def check_trimmed_metadata(f, page):
    counts, physical, last, queries = [], [], [], []
    for row, slot in enumerate(f.cpu_slots.tolist()):
        begin, end = f.cpu_token_ptr[row : row + 2].tolist()
        length = end - begin if slot != -1 else 0
        drop = 0 if f.window is None else max(length - f.q - f.window + 1, 0) // page * page
        count = (length - drop + page - 1) // page
        counts.append(count)
        physical.extend((f.cpu_token_idx[begin + drop : begin + length : page] // page).tolist())
        last.append((length - 1) % page + 1 if length else 0)
        queries.append(f.q if length else 0)
    for tensor, expected in (
        (f.page_ptr, torch.tensor([0, *counts], dtype=torch.int32).cumsum(0, dtype=torch.int32)),
        (f.query_ptr, torch.tensor([0, *queries], dtype=torch.int32).cumsum(0, dtype=torch.int32)),
        (f.last, torch.tensor(last, dtype=torch.int32)),
        (f.token_ptr, f.cpu_token_ptr),
        (f.token_idx, f.cpu_token_idx),
        (f.slots, f.cpu_slots),
        (f.positions, f.cpu_positions),
    ):
        torch.testing.assert_close(tensor.cpu(), expected, atol=0, rtol=0)
    expected = torch.full_like(f.page_idx.cpu(), SENTINEL)
    expected[: len(physical)] = torch.tensor(physical, dtype=torch.int32)
    torch.testing.assert_close(f.page_idx.cpu(), expected, atol=0, rtol=0)
    assert (f.out_backing[:128] == 123).all() and (f.out_backing[-128:] == 123).all()


def compare(f, graph, page):
    expected = f.oracle()
    f.run()
    torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
    eager = f.out.clone()
    f.out.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(f.out, eager)
    torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
    check_trimmed_metadata(f, page)
    for row, slot in enumerate(f.cpu_slots.tolist()):
        if slot == -1 or f.cpu_token_ptr[row] == f.cpu_token_ptr[row + 1]:
            assert torch.count_nonzero(f.out[row * f.q : (row + 1) * f.q]) == 0


# All requested Q/W pairs; rotate other axes for <4 min steady-state coverage.
CASES = [
    (
        q,
        w,
        False,
        (64, 128, 256)[(i + j) % 3],
        ((32, 8), (64, 1))[j % 2],
        (1, 3)[(i + j) % 2],
        (1, 16)[i % 2],
        (257, 8192)[j % 2],
    )
    for i, q in enumerate((2, 8, 16, 33, 64))
    for j, w in enumerate(dict.fromkeys((1, 2, 7, q - 1)))
] + [
    (33, w, causal, dim, heads, 3, 16, 1024)
    for dim, heads in ((64, (32, 8)), (128, (64, 1)), (256, (32, 8)))
    for causal, w in ((False, 33), (False, 64), (False, None), (True, 7), (True, None))
]


def exercise(query, window, causal, dim, heads, batch, page, capacity, provider="flashinfer-mask"):
    f = Fixture(
        batch, query, page, *heads, causal, window, c=capacity, provider=provider, head_dim=dim
    )
    assert f.op.provider == "flashinfer-fa2-custom-mask"
    # Steady-state FA2 PDL (DF4): the first run after each prepare() is serialized,
    # so the mask prepare writes are visible; CTI_FIXED_QUERY_PDL=0 disables PDL.
    assert f.op.pdl == (os.environ.get("CTI_FIXED_QUERY_PDL", "1") != "0")
    assert f.op.merge == "flashinfer"
    f.stage([capacity - query] * batch, list(range(batch)))
    graph = capture([f])
    # Cross page boundaries, shrink to Q, swap slots, then reactivate all rows.
    for turn in range(3):
        lengths = [query if (r + turn) % 3 == 0 else capacity - turn for r in range(batch)]
        slots = [(-1 if batch > 1 and r == turn else r) for r in range(batch)]
        if turn == 2:
            slots = list(reversed(range(batch)))
        f.stage([n - query for n in lengths], slots, turn + 1)
        compare(f, graph, page)


@pytest.mark.parametrize("query,window,causal,dim,heads,batch,page,capacity", CASES)
def test_custom_mask(query, window, causal, dim, heads, batch, page, capacity):
    exercise(query, window, causal, dim, heads, batch, page, capacity)


@pytest.mark.exhaustive
@pytest.mark.parametrize("query", [2, 8, 16, 33, 64])
@pytest.mark.parametrize("window_kind", [1, 2, 7, "q-1"])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("heads", [(32, 8), (64, 1)])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("page", [1, 16])
def test_custom_mask_exhaustive(query, window_kind, dim, heads, batch, page):
    window = query - 1 if window_kind == "q-1" else window_kind
    exercise(query, window, False, dim, heads, batch, page, 8192)


def test_auto_without_cute(monkeypatch):
    def unavailable(**kwargs):
        raise NotImplementedError("CuTe unavailable")

    monkeypatch.setattr(cute, "check_support", unavailable)
    exercise(33, 7, False, 128, (32, 8), 3, 16, 1024, provider=None)


@pytest.mark.parametrize(
    "provider,causal", [("flashinfer-mask", False), ("flashinfer-mask", True), ("flashinfer", True)]
)
@pytest.mark.parametrize("query", [8, 33])
def test_prepare_cache_reuses_windows(provider, causal, query):
    from ctikernels.attention._draft.flashinfer_mask.prepare import pack_mask
    from ctikernels.attention._flashinfer_schedule import prepare_draft, prepare_draft_chunked

    prepare = prepare_draft if query <= 32 else prepare_draft_chunked
    kernels = (prepare, pack_mask) if provider == "flashinfer-mask" else (prepare,)
    for kernel in kernels:
        kernel.device_caches.clear()

    def counts():
        # Triton 3.x: per-device tuple starts with the compiled kernel cache.
        return tuple(len(k.device_caches[torch.cuda.current_device()][0]) for k in kernels)

    baseline = None
    for window in (1, 2, 3, 5, 7, 16, 17, 31, 64, 2048):
        f = Fixture(2, query, 16, 8, 2, causal, window, c=4096, provider=provider)
        f.stage([4096 - query, 4095 - query], [0, 1])
        f.run()
        torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
        if baseline is None:
            baseline = counts()
            assert all(n > 0 for n in baseline)
        else:
            assert counts() == baseline, f"new compilation for W={window}"


def test_long_context():
    f = LongContextCase(2, 131072, 64, 32, 8, 128, False, 7, "flashinfer-mask")
    f.stage([131072, 131071], 0)
    graph = capture([f])
    for turn, lengths in enumerate(([64, 131071], [131072, 193]), 1):
        f.stage(lengths, turn)
        compare(f, graph, PAGE)


def test_zero_length_and_inactive():
    f = Fixture(3, 8, 16, 32, 8, False, 2, c=128, provider="flashinfer-mask")
    f.stage([-8, 120, 0], [0, 1, -1])
    f.query[:8].fill_(torch.nan)
    graph = capture([f])
    compare(f, graph, 16)
    f.stage([120, -8, 0], [0, 1, -1], 1)
    f.query[8:16].fill_(torch.nan)
    compare(f, graph, 16)


@pytest.mark.exhaustive
@pytest.mark.parametrize("query,window", [(129, 128), (1024, 511)])
def test_union_exceeds_query_tile(query, window):
    # G=1, Q>tile: a per-tile W+tile bound cannot size the request mask/CSR.
    exercise(query, window, False, 128, (1, 1), 1, 16, 8192)
