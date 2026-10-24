# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Runtime symmetric windows, cache reuse and changing-metadata graph replay."""

import os

import pytest
import torch

from ctikernels.attention._draft import cute

from ._dflash import Fixture

ATOL = RTOL = 2e-2
WINDOWS = [1, 2, 7, 64, 2048, 4096, 8191, None]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def window_environment(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def capture(fixtures):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for f in fixtures:
            f.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for f in fixtures:
            f.run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def compare_replay(fixtures, graph):
    expected, eager = [], []
    for f in fixtures:
        expected.append(f.oracle())
        f.run()
        torch.testing.assert_close(f.out.float(), expected[-1], atol=ATOL, rtol=RTOL)
        eager.append(f.out.clone())
        f.out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for f, result, reference in zip(fixtures, eager, expected, strict=True):
        assert torch.equal(f.out, result)
        torch.testing.assert_close(f.out.float(), reference, atol=ATOL, rtol=RTOL)
        f.check_metadata()
        for row, slot in enumerate(f.cpu_slots.tolist()):
            if slot == -1:
                assert torch.count_nonzero(f.out[row * f.q : (row + 1) * f.q]) == 0


# Auto covers the full grid; explicit arms repeat GQA plus the targeted MQA tests below.
@pytest.mark.parametrize(
    "provider,hq,hk",
    [("auto", 2, 2), ("auto", 8, 2), ("auto", 64, 1), ("cute", 8, 2), ("flashinfer", 8, 2)],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=lambda d: f"hd{d}")
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w}")
def test_runtime_window(provider, head_dim, hq, hk, causal, window):
    q = 2 if (hq, hk) == (64, 1) else 8
    narrow = not causal and window is not None and window < q

    def construct():
        return Fixture(
            4,
            q,
            16,
            hq,
            hk,
            causal,
            window,
            c=8256,
            provider=None if provider == "auto" else provider,
            head_dim=head_dim,
        )

    if narrow and provider == "flashinfer":
        with pytest.raises(NotImplementedError, match="FA2 window_left has no right bound"):
            construct()
        return
    f = construct()
    assert isinstance(f.op.pdl, bool)
    assert f.op.merge in ("none", "legacy", "tree", "distributed", "flashinfer")
    if narrow:
        assert f.op.provider == "cute-splitkv"
    f.stage([0, 100 - q, 3000 - q, 8256 - q], [0, 1, 2, -1])
    graph = capture([f])
    # Construction/capture keep storage fixed; replay changes lengths, slots and pages.
    cases = [
        ([q, 100, 8256, q], [0, 1, 2, -1]),
        ([3000, q, 100, 8255], [3, -1, 1, 0]),
    ]
    for turn, (lengths, slots) in enumerate(cases, start=1):
        f.stage([length - q for length in lengths], slots, turn)
        compare_replay([f], graph)


def test_window_cache_and_multilayer_graph():
    # Earlier grid points must not hide a per-window cache miss.
    cute.compile_kernels.cache_clear()

    def layer(window):
        return Fixture(4, 8, 16, 8, 2, False, window, c=8256, provider="cute")

    fixtures = [layer(64)]
    misses = cute.compile_kernels.cache_info().misses
    for window in (2048, 3000, 4096, 8191):
        fixtures.append(layer(window))
        assert cute.compile_kernels.cache_info().misses == misses
    for f in fixtures:
        f.stage([0, 92, 2992, 8248], [0, 1, 2, -1])
    graph = capture(fixtures)
    for turn, lengths in enumerate(([8, 100, 3000, 8256], [8255, 3000, 8, 100]), start=1):
        for f in fixtures:
            f.stage([length - 8 for length in lengths], [0, -1, 2, 3], turn)
        compare_replay(fixtures, graph)


@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [128, 129, 2**31 - 1])
def test_window_at_least_capacity(provider, causal, window):
    f = Fixture(2, 8, 16, 8, 2, causal, window, c=128, provider=provider)
    f.stage([120, 0], [0, -1])
    graph = capture([f])
    for turn, contexts in enumerate(([120, 24], [0, 120])):
        f.stage(contexts, [0, 1], turn)
        compare_replay([f], graph)


@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
@pytest.mark.parametrize("query", [8, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal,window", [(True, 64), (False, 3), (False, None)])
def test_mqa_many_query_tiles(provider, query, head_dim, causal, window):
    # Q*G = 512/1024 packed rows per KV head; noncausal W<Q has only the CuTe provider.
    def construct():
        return Fixture(
            2, query, 16, 64, 1, causal, window, c=256, provider=provider, head_dim=head_dim
        )

    if provider == "flashinfer" and not causal and window is not None:
        with pytest.raises(NotImplementedError, match="FA2 window_left has no right bound"):
            construct()
        return
    f = construct()
    f.stage([0, 256 - query], [0, 1])
    graph = capture([f])
    f.stage([256 - query, 32], [1, 0], 1)
    compare_replay([f], graph)


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
def test_dsv4_draft_geometry(provider, batch):
    f = Fixture(batch, 8, 16, 64, 1, True, 2048, c=4160, provider=provider, head_dim=256)
    f.stage([128] * batch, list(range(batch)))
    graph = capture([f])
    f.stage([4152] * batch, list(range(batch)), 1)
    compare_replay([f], graph)


@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fused_qkv_query_stride(provider, head_dim):
    compact = Fixture(2, 8, 16, 8, 2, True, 64, c=256, provider=provider, head_dim=head_dim)
    compact.stage([248, 24], [0, 1])
    compact.run()
    f = Fixture(
        2,
        8,
        16,
        8,
        2,
        True,
        64,
        c=256,
        provider=provider,
        head_dim=head_dim,
        query_row_stride=12 * head_dim,
    )
    f.stage([248, 24], [0, 1])
    for op, query in (
        (f.op, compact.query),
        (compact.op, torch.empty((16, 16, head_dim), dtype=f.query.dtype, device="cuda")[:, :8]),
    ):
        for name in ("run", "prepare_and_run"):
            with pytest.raises(ValueError, match="query_row_stride"):
                getattr(op, name)(q=query, paged_kv_cache=(f.k, f.v), out=f.out)
    fused = torch.empty((16, 12, head_dim), dtype=f.query.dtype, device="cuda")
    fused[:, :8].copy_(f.query)
    f.query = fused[:, :8]
    f.run()
    assert torch.equal(f.out, compact.out)
    graph = capture([f])
    f.stage([0, 248], [0, 1], 1)
    compare_replay([f], graph)


def test_cute_without_page_indices():
    f = Fixture(2, 8, 16, 8, 2, False, 7, c=128, provider="cute", page_indices=False)
    assert f.page_idx is None
    f.stage([120, 0], [0, -1])
    expected = f.oracle()
    f.op.prepare()
    f.op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
    torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
    eager = f.out.clone()
    f.out.fill_(float("nan"))
    f.op.prepare_and_run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)
    assert torch.equal(f.out, eager)
    graph = capture([f])
    f.stage([0, 120], [0, 1], 1)
    compare_replay([f], graph)
    with pytest.raises(ValueError, match="page_indices.*required.*FlashInfer"):
        Fixture(2, 8, 16, 8, 2, True, 7, c=128, provider="flashinfer", page_indices=False)


def test_fa2_prepare_cache_reuses_all_windows():
    from ctikernels.attention._flashinfer_schedule import prepare_draft

    prepare_draft.device_caches.clear()

    def kernels():
        # Triton 3.x: per-device (kernel cache, key cache, target, backend, binder).
        return len(prepare_draft.device_caches[torch.cuda.current_device()][0])

    def layer(window):
        f = Fixture(2, 8, 16, 8, 2, True, window, c=8256, provider="flashinfer")
        f.stage([92, 2992], [0, 1])
        f.op.prepare()
        return f

    first = layer(64)
    count = kernels()
    assert count > 0
    # W-1 includes zero, one, divisible-by-16 and ordinary scalars.
    for window in (1, 2, 17, 2048, 3000, 4096, 8191):
        f = layer(window)
        assert kernels() == count
        f.run()
        torch.testing.assert_close(f.out.float(), f.oracle(), atol=ATOL, rtol=RTOL)
    assert first.op.merge == "flashinfer"
    assert first.op.pdl is (os.environ.get("CTI_FIXED_QUERY_PDL", "1") != "0")


@pytest.mark.parametrize(
    "query,causal,window", [(16, False, None), (16, True, 4096), (8, False, 2048)]
)
def test_effective_legacy_settings(monkeypatch, query, causal, window):
    from ctikernels.attention._draft import dispatch

    for name in ("MERGE", "INLINE", "PDL"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    f = Fixture(1, query, 16, 32, 8, causal, window, c=8256, provider="cute")
    f.stage([512 - query], [0])
    f.run()
    assert isinstance(f.op.pdl, bool)
    assert f.op.pdl == f.op._implementation.pdl
    assert f.op.merge == f.op._implementation.merge
    if dispatch.hardware(torch.device("cuda")) in dispatch.LEGACY_MERGE_ROWS:
        assert f.op.merge == ("legacy" if window is not None else "distributed")
    for name in ("provider", "pdl", "merge"):
        with pytest.raises(AttributeError):
            setattr(f.op, name, None)


@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
@pytest.mark.parametrize(
    "query,causal,window", [(16, False, None), (16, True, None), (8, True, None), (8, True, 2048)]
)
def test_qwen3_moe_kq7(provider, query, causal, window):
    f = Fixture(2, query, 16, 32, 4, causal, window, c=4160, provider=provider)
    f.stage([128, 512], [0, 1])
    graph = capture([f])
    f.stage([4160 - query, 0], [1, 0], 1)
    compare_replay([f], graph)
