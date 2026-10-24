# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Capacity-only compile sets and frozen DF2/DF3 draft-pair launch geometry."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute

# Captured before DW1 completion, B=1..4. Each entry is (rows, splits, chunk_cap).
PAIR_KEYS = {
    (0, 170): (
        ((32, 10, 1024), (32, 10, 1024), (32, 7, 1024), (32, 5, 1024)),
        ((32, 10, 416), (64, 10, 416), (64, 14, 320), (64, 10, 416)),
        ((32, 21, 128), (32, 10, 224), (32, 14, 160), (32, 10, 224)),
    ),
    (0, 188): (
        ((32, 11, 1024), (32, 11, 1024), (32, 7, 1024), (32, 5, 1024)),
        ((32, 11, 384), (64, 11, 384), (64, 15, 288), (64, 11, 384)),
        ((32, 23, 96), (32, 11, 192), (32, 15, 160), (32, 11, 192)),
    ),
    (1, 48): (
        ((32, 3, 4096), (32, 2, 4096), (64, 2, 4096), (64, 2, 4096)),
        ((64, 6, 704), (64, 3, 1408), (64, 2, 2112), (64, 2, 2112)),
        ((32, 6, 384), (32, 3, 704), (32, 2, 1088), (32, 1, 2112)),
    ),
}
PAIRS = ((16, False, None), (16, True, 4096), (8, False, 2048))
# Only the W2048 selected-tier chunk_cap changes; every other tuple field is frozen.
WINDOW_CAPS = {
    (0, 170): (224, 416, 320, 416),
    (0, 188): (192, 384, 288, 384),
    (1, 48): (704, 1408, 2112, 4160),
}


def device(monkeypatch, minor=0, sms=170):
    for name in (
        "ROWS",
        "SPLITS",
        "MIN_CHUNK",
        "TARGET_CTAS",
        "CHUNK_CAP",
        "KEY_TILE",
        "KEY_GROUPS",
        "STAGES",
        "SHORT",
        "INLINE",
        "MERGE",
        "COOP_PDL",
        "PIECES",
    ):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12,
            minor=minor,
            multi_processor_count=sms,
        ),
    )
    monkeypatch.setattr(cute._cooperative, "supported", lambda _: False)


def construct(
    monkeypatch,
    window,
    *,
    batch=1,
    query=8,
    causal=False,
    dim=128,
    heads=8,
    group=4,
    capacity=32768,
    pages=True,
    query_row_stride=None,
):
    calls = []

    def compile_stub(*args, **kwargs):
        calls.append((args, kwargs))
        return None, lambda *args: None, None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(batch, dtype=torch.int32)
    provider = cute.Provider(
        torch.empty(256, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, capacity),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer if pages else None,
        last_page_lengths=integer,
        query_length=query,
        num_qo_heads=heads * group,
        num_kv_heads=heads,
        head_dim=dim,
        page_size=1,
        dtype=torch.bfloat16,
        causal=causal,
        window_left=-1 if window is None else window - 1,
        query_row_stride=query_row_stride,
    )
    return provider, calls


@pytest.mark.parametrize("minor,sms", PAIR_KEYS)
@pytest.mark.parametrize("pair", range(3))
@pytest.mark.parametrize("batch", range(1, 5))
def test_frozen_pairs(monkeypatch, minor, sms, pair, batch):
    device(monkeypatch, minor, sms)
    query, causal, window = PAIRS[pair]
    p, calls = construct(monkeypatch, window, batch=batch, query=query, causal=causal)
    if pair == 0:
        # The Q16 full pair left the frozen 0.1.12 legacy merge for the distributed
        # merge (market receipts in draft_window_dwr.md); the windowed pairs stay frozen.
        assert p.merge != "legacy"
        return
    rows, splits, cap = PAIR_KEYS[minor, sms][pair][batch - 1]
    tile = 64 if minor else 32
    if pair == 2:
        cap = WINDOW_CAPS[minor, sms][batch - 1]
    expected = (
        query,
        4,
        rows,
        8,
        causal,
        window is not None,
        splits,
        torch.bfloat16,
        1,
        f"sm_12{minor}a",
        64,
        tile,
        2 if rows <= 48 or minor else 1,
        2,
        cap,
        pair == 0,
        pair == 0,
        0,
        False,
    )
    assert calls[0][0] == expected
    assert p.merge == ("none" if splits == 1 else "legacy")
    assert cute.configuration(
        batch_size=batch,
        query_length=query,
        max_kv_tokens=32768,
        num_qo_heads=32,
        num_kv_heads=8,
        window=window,
        device="cuda",
        causal=causal,
    ) == (4, rows, -(-query * 4 // rows), splits)


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("dim", (64, 128, 256))
@pytest.mark.parametrize("capacity", (32, 4128, 32768))
@pytest.mark.parametrize("mode", ("legacy", "tree", "distributed"))
@pytest.mark.parametrize("pages", (False, True))
def test_compile_set_independent_of_window(monkeypatch, causal, dim, capacity, mode, pages):
    device(monkeypatch)
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setattr(cute._cooperative, "supported", lambda _: True)
    monkeypatch.setattr(cute._cooperative, "blocks_per_sm", lambda *args: 2)
    previous = None
    sizes = set()
    for window in (1, 7, 64, 513, 2048, 3000, 4096, 4097, 8191, 2**31 - 1):
        p, calls = construct(
            monkeypatch, window, causal=causal, dim=dim, capacity=capacity, pages=pages
        )
        if previous is not None:
            assert calls == previous
        previous = calls
        assert all(kw["write_pages"] is pages for _, kw in calls)
        assert p.tier == next(
            t for t in sorted({min(capacity, 4128), capacity}) if t >= min(capacity, window + 7)
        )
        sizes.add(
            cute.workspace_size(
                batch_size=1,
                query_length=8,
                max_kv_tokens=capacity,
                num_qo_heads=32,
                num_kv_heads=8,
                window=window,
                device="cuda",
                head_dim=dim,
                causal=causal,
            )
        )
    assert len(sizes) == 1
    _, full = construct(monkeypatch, None, causal=causal, dim=dim, capacity=capacity, pages=pages)
    assert all(not args[5] for args, _ in full)
    assert all(args[5] for args, _ in previous)


@pytest.mark.parametrize("dim", (64, 128, 256))
@pytest.mark.parametrize("group", (1, 3, 64, 128))
@pytest.mark.parametrize("query", (2, 8, 32))
def test_large_gqa_and_optional_pages(monkeypatch, dim, group, query):
    device(monkeypatch)
    p, calls = construct(monkeypatch, 7, dim=dim, heads=1, group=group, query=query, pages=False)
    assert p.rt == -(-query * group // p.rows)
    assert all(kw["write_pages"] is False for _, kw in calls)
    assert p._prepare_inputs[5] is p._page_dummy
    assert p._page_dummy.shape == (1,)
    assert p._page_dummy.dtype == torch.int32
    assert p.pdl is True
    for attr in ("pdl", "merge", "tier"):
        with pytest.raises(AttributeError):
            setattr(p, attr, None)
    pytest.importorskip("cutlass")
    cute.check_support("cuda", query_length=query, num_qo_heads=group, num_kv_heads=1, head_dim=dim)


def test_grid_limits(monkeypatch):
    device(monkeypatch)
    args = {
        "batch_size": 1,
        "query_length": 32,
        "num_qo_heads": 65536 * 4,
        "num_kv_heads": 65536,
        "head_dim": 128,
        "window": None,
        "device": "cuda",
        "max_kv_tokens": 32768,
    }
    with pytest.raises(NotImplementedError, match="grid-y"):
        cute.check_support(**args)
    with pytest.raises(NotImplementedError, match="grid-y"):
        cute.configuration(**args)


@pytest.mark.parametrize("row_stride", (None, 4096, 6144))
def test_fake_q_has_static_stride_and_out_is_compact(monkeypatch, row_stride):
    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as dsl

    calls = []
    monkeypatch.setattr(dsl, "compile", lambda kernel, *args, **kw: calls.append((kernel, args)))
    cute.compile_kernels.__wrapped__(
        8,
        64,
        32,
        1,
        False,
        True,
        7,
        torch.bfloat16,
        16,
        "sm_120a",
        head_dim=64,
        write_pages=False,
        q_row_stride=row_stride,
    )
    prepare, attention, merge = calls
    assert not prepare[0].write_pages and not attention[0].write_pages
    q, out = attention[1][3], attention[1][6]
    assert q is not out and out is merge[1][3]
    assert q.stride == (row_stride or 4096, 64, 1)
    assert attention[0].q_row_stride == (row_stride or 4096)
    assert out.stride == (4096, 64, 1)
    assert q.element_type is cutlass.BFloat16


@pytest.mark.parametrize("row_stride", (None, 6144))
@pytest.mark.parametrize("mode", ("legacy", "tree", "distributed"))
def test_compile_stride_for_every_tier_and_pdl_variant(monkeypatch, row_stride, mode):
    device(monkeypatch)
    monkeypatch.setenv("CTI_DRAFT_MERGE", mode)
    monkeypatch.setattr(cute._cooperative, "supported", lambda _: True)
    monkeypatch.setattr(cute._cooperative, "blocks_per_sm", lambda *args: 2)
    _, calls = construct(monkeypatch, 7, query_row_stride=row_stride)
    assert len(calls) == (4 if mode == "distributed" else 2)
    assert all(kwargs.get("q_row_stride") == row_stride for _, kwargs in calls)
    if row_stride is None:
        assert all("q_row_stride" not in kwargs for _, kwargs in calls)


@pytest.mark.parametrize("dim", (64, 128, 256))
@pytest.mark.parametrize("rows", (16, 32, 48, 64))
@pytest.mark.parametrize("groups", (1, 2))
def test_merge_staging_fits_owned_rows(dim, rows, groups):
    cutlass = pytest.importorskip("cutlass")
    from ctikernels.attention._draft.cute._kernel import Attention

    kernel = Attention(
        32,
        128,
        rows,
        1,
        False,
        True,
        7,
        cutlass.BFloat16,
        key_tile=32,
        key_groups=groups,
        stages=2,
        head_dim=dim,
        inline_merge=True,
        distributed_merge=True,
    )
    assert kernel.merge_capacity >= (rows + 1) // 2
    assert kernel.index_parts * kernel.threads >= kernel.kt * (dim // 8)
    assert kernel.merge_cols >= kernel.strip
    assert kernel.epilogue_cols >= 32


@pytest.mark.parametrize("arch", ("sm_120a", "sm_121a"))
@pytest.mark.parametrize("rows", (16, 32, 48, 64))
@pytest.mark.parametrize("groups", (1, 2))
@pytest.mark.parametrize("operand", ("Q", "K", "V"))
def test_hd64_ldmatrix_has_no_aliased_blocks(arch, rows, groups, operand, tmp_path):
    pytest.importorskip("cutlass")
    from .test_dflash_draft_layout import test_operand_copy_covers_all_blocks

    # Check lowered ldmatrix instructions, not just composed-layout dimensions.
    test_operand_copy_covers_all_blocks(arch, 64, rows, groups, operand, tmp_path)
