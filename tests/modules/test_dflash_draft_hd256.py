# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""VERIFY provider routing, geometry and resource bounds without CUDA allocations."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute, dispatch, flashinfer
from ctikernels.attention.draft import _provider


@pytest.fixture(autouse=True)
def tuning(monkeypatch):
    for name in (
        "PROVIDER",
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


@pytest.mark.parametrize("hq,hk,default_cute", [(16, 4, True), (16, 2, True), (8, 1, False)])
def test_hd256_provider_overrides_and_support(monkeypatch, hq, hk, default_cute):
    monkeypatch.setattr(dispatch, "hardware", lambda _: (12, 0, 170))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170),
    )
    pytest.importorskip("cutlass")
    args = {
        "device": "cuda",
        "query_length": 16,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": 256,
        "batch_size": 1,
        "max_kv_tokens": 32768,
        "page_size": 16,
        "causal": True,
        "window": None,
    }
    cute.check_support(**args)
    # GQA-8 rows follow HD256_GQA8_CUTE_ROWS: 8/1 at B=1 measured faster on FlashInfer.
    assert _provider(None, args) is (cute if default_cute else flashinfer)
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "flashinfer")
    assert _provider(None, args) is flashinfer
    assert _provider("cute", args) is cute
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "cute")
    assert _provider(None, args) is cute
    assert _provider("flashinfer", args) is flashinfer
    monkeypatch.delenv("CTI_DRAFT_PROVIDER")
    # The GQA-8 table covers group 8 only; groups 16 and 64 follow HD256_CUTE_ROWS.
    assert _provider(None, args | {"num_qo_heads": 64}) is cute
    cute.check_support(**(args | {"query_length": 33}))
    with pytest.raises(ValueError, match=r"Q >= 1"):
        cute.check_support(**(args | {"query_length": 0}))


@pytest.mark.parametrize("minor", [0, 1])
@pytest.mark.parametrize("inline", [0, 1])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("query", [4, 8, 16])
@pytest.mark.parametrize("hq,heads", [(8, 2), (16, 4), (16, 2), (8, 1)])
@pytest.mark.parametrize("capacity", [416, 417, 4128, 32768])
def test_hd256_provider_defaults_fit(monkeypatch, minor, inline, batch, query, hq, heads, capacity):
    cutlass = pytest.importorskip("cutlass")
    from ctikernels.attention._draft.cute._kernel import Attention

    monkeypatch.setenv("CTI_DRAFT_INLINE", str(inline))
    # Per-request grid sizing; spread grids: test_dflash_cute_dispatch.py.
    monkeypatch.setenv("CTI_DRAFT_SPREAD", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12, minor=minor, multi_processor_count=170 if minor == 0 else 48
        ),
    )
    args = {
        "batch_size": batch,
        "query_length": query,
        "max_kv_tokens": capacity,
        "num_qo_heads": hq,
        "num_kv_heads": heads,
        "head_dim": 256,
        "window": None,
        "device": "cuda",
    }
    group, rows, rt, splits = cute.configuration(**args)
    assert (group, rows, rt) == (hq // heads, 32, (query * (hq // heads) + 31) // 32)
    floating, integer_bytes = cute.workspace_size(**args)
    payload = (batch * heads * rt * splits * rows * 257 + batch) * 4
    assert floating == (max(256, (payload + 255) // 256 * 256) if splits > 1 else 256)
    assert integer_bytes == 256
    kernels = []

    def compile_stub(*constants, head_dim, write_pages=True, **options):
        (
            q,
            g,
            r,
            h,
            causal,
            window,
            s,
            _,
            page,
            _arch,
            minimum,
            tile,
            groups,
            stages,
            cap,
            pieces,
            sparse,
            short,
            inline_merge,
        ) = constants
        kernel = Attention(
            q,
            g,
            r,
            h,
            causal,
            window,
            s,
            cutlass.BFloat16,
            minimum,
            tile,
            groups,
            stages,
            cap,
            page=page,
            pieces=pieces,
            sparse=sparse,
            short_union=short,
            inline_merge=inline_merge,
            head_dim=head_dim,
        )
        assert kernel.dim == 256
        assert kernel.kt == 32 and kernel.stages == 2 and kernel.groups == 2
        assert kernel.smem_bytes + 1024 <= 99 * 1024
        assert kernel.scale == 256**-0.5 * 1.4426950408889634
        kernels.append(kernel)
        return None, None, None

    monkeypatch.setattr(cute, "compile_kernels", compile_stub)
    integer = torch.zeros(batch, dtype=torch.int32)
    provider = cute.Provider(
        torch.empty(floating, dtype=torch.uint8),
        integer,
        prepare_inputs=(integer, integer, integer, integer, capacity),
        query_indptr=integer,
        kv_indptr=integer,
        kv_indices=integer,
        last_page_lengths=integer,
        query_length=query,
        num_qo_heads=hq,
        num_kv_heads=heads,
        head_dim=256,
        page_size=16,
        dtype=torch.bfloat16,
        causal=True,
        window_left=-1,
    )
    assert len(kernels) == 1
    assert provider.merge_fan_in == kernels[0].merge_fan_in
    assert provider._counters.numel() == batch * heads * rt * sum(provider.merge_levels)


@pytest.mark.parametrize("rows", [16, 32, 48, 64])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("cap", [1024, 4096])
def test_hd256_shared_arena(rows, groups, inline, cap):
    cutlass = pytest.importorskip("cutlass")
    from ctikernels.attention._draft.cute._kernel import Attention

    kernel = Attention(
        16,
        4,
        rows,
        4,
        True,
        False,
        23,
        cutlass.BFloat16,
        key_tile=32,
        key_groups=groups,
        stages=2,
        chunk_cap=cap,
        inline_merge=inline,
        head_dim=256,
    )
    arena = kernel.smem_bytes - cap * 4
    assert arena >= 2 * 2 * 32 * 256 * 2  # K/V, stages, keys, D, BF16.
    assert arena >= rows * 256 * 2  # Q, reused before K/V.
    assert arena >= rows * kernel.epilogue_cols * 4
    assert kernel.epilogue_cols in (128, 256)
    if groups == 2:
        assert arena >= rows * 256 * 4 + rows * 8
    if inline:
        assert arena >= kernel.merge_fan_in * rows * (kernel.merge_cols + 1) * 4 + 16
        assert kernel.merge_cols == (256 if rows <= 32 else 128)
    assert kernel.smem_bytes + 1024 <= 99 * 1024


@pytest.mark.parametrize("tile,stages,cap", [(64, 2, 1024), (32, 3, 1024), (32, 2, 16384)])
def test_hd256_rejects_oversized_shared_memory(tile, stages, cap):
    cutlass = pytest.importorskip("cutlass")
    from ctikernels.attention._draft.cute._kernel import Attention

    with pytest.raises(ValueError, match="shared memory exceeds 99 KiB"):
        Attention(
            16,
            4,
            32,
            4,
            True,
            False,
            7,
            cutlass.BFloat16,
            key_tile=tile,
            stages=stages,
            chunk_cap=cap,
            inline_merge=True,
            head_dim=256,
        )
