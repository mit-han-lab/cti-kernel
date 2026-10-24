# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Cooperative mode selection, query fallback and PDL retry without a device."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute
from ctikernels.attention._draft.cute import _cooperative as coop


def construct(
    monkeypatch, mode="distributed", support=True, occupancy=1, splits=4, fail=None, head_dim=128
):
    for key in ("INLINE", "ROWS", "TARGET_CTAS", "SHORT", "CHUNK_CAP", "COOP_PDL"):
        monkeypatch.delenv("CTI_DRAFT_" + key, raising=False)
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setenv("CTI_DRAFT_SPLITS", str(splits))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=32),
    )
    monkeypatch.setattr(coop, "supported", lambda _: support)
    limits = iter(occupancy) if isinstance(occupancy, tuple) else None
    monkeypatch.setattr(coop, "blocks_per_sm", lambda *args: next(limits) if limits else occupancy)
    compiled, launched = [], []

    def compile_stub(*args, head_dim, write_pages=True, **options):
        distributed = len(args) > 19 and args[19]
        pdl = len(args) <= 20 or args[20]
        compiled.append((distributed, pdl))
        if fail is not None:
            fail(distributed, pdl)

        def attention(*args):
            launched.append((distributed, pdl))

        return lambda *args: None, attention, None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(1, dtype=torch.int32)
    provider = cute.Provider(
        torch.zeros(256, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, 32768),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=4,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
        page_size=16,
        dtype=torch.bfloat16,
        causal=False,
        window_left=-1,
    )
    return provider, compiled, launched, integer


@pytest.mark.parametrize(
    "mode,support,occupancy,splits,chosen",
    [
        ("distributed", True, 1, 4, "distributed"),
        ("distributed", True, 1, 5, "tree"),
        ("distributed", True, 2, 5, "distributed"),
        ("distributed", True, (2, 1), 5, "tree"),
        ("distributed", True, 0, 1, "tree"),
        ("distributed", False, 1, 4, "tree"),
        ("tree", True, 2, 4, "tree"),
        ("legacy", True, 2, 4, "legacy"),
        ("distributed", True, 1, 1, "distributed"),
    ],
)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_selection(monkeypatch, mode, support, occupancy, splits, chosen, head_dim):
    p, compiled, _, _ = construct(monkeypatch, mode, support, occupancy, splits, head_dim=head_dim)
    assert p.merge_mode == chosen
    assert p.merge == ("none" if splits == 1 else chosen)
    assert p.pdl
    assert bool(p.fallbacks) == (mode == "distributed" and chosen == "tree")
    assert p.inline_merge == (chosen != "legacy")
    if chosen == "distributed":
        assert p.counter_bytes == (64 if splits > 1 else 0)
        assert p._arrivals.numel() * 2 == p._counters.numel()
        p._counters[1::2] = -1
        assert torch.count_nonzero(p._arrivals) == 0
        assert compiled == [(True, True), (True, False)]
    elif chosen == "tree":
        assert p.counter_bytes == 32 * sum(p.merge_levels)
        assert compiled[-1][0] is False
    else:
        assert p.counter_bytes == 0


def test_compile_retries_without_pdl(monkeypatch):
    def fail(distributed, pdl):
        if distributed and pdl:
            raise RuntimeError("cooperative programmatic launch rejected")

    p, compiled, _, _ = construct(monkeypatch, fail=fail)
    assert p.merge_mode == "distributed" and not p.coop_pdl
    assert not p.pdl and "compile rejected PDL" in p.fallbacks[-1]
    assert compiled == [(True, True), (True, False)]


def test_compile_unavailable_uses_tree(monkeypatch):
    def fail(distributed, pdl):
        if distributed:
            raise TypeError("cooperative launch unavailable")

    p, compiled, _, _ = construct(monkeypatch, fail=fail)
    assert p.merge_mode == "tree"
    assert compiled == [(True, True), (True, False), (False, True)]


def test_unrelated_compile_error_is_not_suppressed(monkeypatch):
    def fail(*args):
        raise RuntimeError("bad kernel arithmetic")

    with pytest.raises(RuntimeError, match="arithmetic"):
        construct(monkeypatch, fail=fail)


@pytest.mark.parametrize("capturing", [False, True])
def test_launch_rejection_keeps_cooperative(monkeypatch, capturing):
    p, _, launched, integer = construct(monkeypatch)

    def reject(*args):
        raise RuntimeError("CUDA_ERROR_NOT_SUPPORTED")

    p._attention = reject
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    if capturing:
        with pytest.raises(RuntimeError, match="CTI_DRAFT_COOP_PDL=0"):
            p.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
        assert not launched
    else:
        import warnings

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            p.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
            p.run(q=integer, paged_kv_cache=(integer, integer), out=integer)
        assert len(recorded) == 1
        assert recorded[0].category is RuntimeWarning
        assert "PDL disabled" in str(recorded[0].message)
        assert launched == [(True, False), (True, False)]
        assert p.merge_mode == "distributed" and not p.coop_pdl
        assert not p.pdl and "eager" in p.fallbacks[-1]


def test_occupancy_uses_compiled_function(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        CUlibrary=int,
        CUfunction_attribute=SimpleNamespace(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES=8),
        cuLibraryGetKernelCount=lambda lib: (0, 1),
        cuLibraryEnumerateKernels=lambda count, lib: (0, [23]),
        cuKernelGetFunction=lambda kernel: (0, kernel + 1),
        cuFuncGetAttribute=lambda attr, function: (0, 1024),
    )

    def query(*args):
        calls.append(args)
        return 0, 2

    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor = query
    monkeypatch.setattr(coop, "driver", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    assert coop.blocks_per_sm(SimpleNamespace(library=42), "cuda", 128, 34000) == 2
    assert calls == [(24, 128, 32976)]
    assert coop.blocks_per_sm(object(), "cuda", 128, 34000) == 1

    class UnavailableLibrary:
        @property
        def library(self):
            raise Exception("CuTe library is unavailable before execution")

    assert coop.blocks_per_sm(UnavailableLibrary(), "cuda", 128, 34000) == 1
    cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor = lambda *args: (1, 0)
    assert coop.blocks_per_sm(SimpleNamespace(library=42), "cuda", 128, 34000) == 1


@pytest.mark.parametrize("mode", ["distributed", "tree", "legacy"])
def test_merge_knob_precedes_inline(monkeypatch, mode):
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setenv("CTI_DRAFT_INLINE", "0")
    assert cute.merge_tuning() == (mode != "legacy", 0 if mode == "legacy" else 288)


@pytest.mark.parametrize("key,value", [("MERGE", "unknown"), ("COOP_PDL", "2")])
def test_invalid_knobs(monkeypatch, key, value):
    monkeypatch.setenv("CTI_DRAFT_" + key, value)
    with pytest.raises(ValueError, match=key):
        cute.requested_merge_mode() if key == "MERGE" else cute.cooperative_pdl()


def test_default_distributed(monkeypatch):
    for key in ("MERGE", "INLINE", "SHORT", "COOP_PDL"):
        monkeypatch.delenv("CTI_DRAFT_" + key, raising=False)
    assert cute.requested_merge_mode() == "distributed"
    assert cute.merge_tuning() == (True, 288)
    assert cute.cooperative_pdl()
