# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Setup and provider compatibility, without importing or executing native CUDA."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ctikernels.attention._flashinfer as implementation
from ctikernels.attention._flashinfer import _PlanLayout
from ctikernels.attention.inference import FlashInferFixedQueryAttention


@pytest.mark.parametrize("sms", [1, 80, 132, 170])
@pytest.mark.parametrize(
    "batch,query,heads,kv_heads,head_dim",
    [
        (1, 2, 8, 2, 64),
        (4, 16, 32, 8, 128),
        (64, 32, 32, 8, 128),
        (8, 32, 8, 2, 256),
        (2, 16, 8, 8, 128),
        (3, 16, 16, 8, 128),
        (4, 32, 64, 8, 64),
    ],
)
def test_workspace_capacity_bounds_every_native_query_tile(
    monkeypatch, sms, batch, query, heads, kv_heads, head_dim
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=8, multi_processor_count=sms),
    )
    floating, integer = FlashInferFixedQueryAttention.workspace_size(
        batch_size=batch,
        query_length=query,
        num_qo_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device=torch.device("cuda"),
    )
    assert floating % 256 == integer % 256 == 0
    for tile_size in (16, 64, 128):
        packed = batch * query * (heads // kv_heads)
        padded_ctas = max(2 * sms // kv_heads, (packed + tile_size - 1) // tile_size + batch - 1)
        assert floating >= heads * padded_ctas * tile_size * (head_dim + 1) * 4
        offset = 0
        for size, alignment in ((padded_ctas * 4, 16),) * 3 + (
            ((batch + 1) * 4, 16),
            (4, 1),
            (4, 16),
            ((batch * query + 1) * 4, 16),
            (padded_ctas, 16),
        ):
            offset = (offset + alignment - 1) // alignment * alignment + size
        assert integer >= offset


@pytest.mark.parametrize(
    "overrides", [{"batch_size": 0}, {"query_length": 0}, {"head_dim": 96}, {"num_qo_heads": 3}]
)
def test_unsupported_geometry_fails_before_cuda_queries(monkeypatch, overrides):
    properties = Mock(side_effect=AssertionError("must validate geometry first"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    values = {
        "batch_size": 1,
        "query_length": 16,
        "num_qo_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 128,
        "device": torch.device("cuda"),
    }
    with pytest.raises(ValueError):
        FlashInferFixedQueryAttention.workspace_size(**(values | overrides))
    properties.assert_not_called()


def test_private_layout_rejects_unqualified_provider_before_cuda(monkeypatch):
    monkeypatch.setattr(implementation, "version", lambda _name: "0.6.99")
    with pytest.raises(RuntimeError, match="qualified with.*0.6.11.post3"):
        FlashInferFixedQueryAttention(
            None,
            None,
            query_indptr=None,
            kv_indptr=None,
            kv_indices=None,
            last_page_lengths=None,
            query_length=16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
        )


def _layout():
    return _PlanLayout(4, 32, 64, 64, 0, 16, 32, 80, 48, 60, 0, 1024, 212, 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_count", 31),
        ("query_tile_size", 48),
        ("cuda_graph", 0),
        ("split_kv", 0),
        ("padded_ctas", 1),
    ],
)
def test_native_layout_must_match_fixed_geometry(field, value):
    with pytest.raises(RuntimeError, match="geometry"):
        _PlanLayout.from_native(_layout()._replace(**{field: value}), batch_size=2, query_length=16)
    with pytest.raises(RuntimeError, match="layout"):
        _PlanLayout.from_native((1, 2), batch_size=2, query_length=16)


def test_schedule_views_borrow_storage_and_reject_invalid_offsets():
    workspace = torch.zeros(256, dtype=torch.uint8)
    views = _layout().views(workspace, 2)
    assert [view.numel() for view in views] == [4, 4, 4, 3, 33, 1, 4]
    assert all(view.untyped_storage().data_ptr() == workspace.data_ptr() for view in views)
    for offset in (-4, 3, 256):
        with pytest.raises(RuntimeError, match="workspace"):
            _layout()._replace(requests_offset=offset).views(workspace, 2)


def test_schedule_views_write_only_their_borrowed_ranges():
    workspace = torch.full((256,), 255, dtype=torch.uint8)
    layout = _layout()
    requests, *other_views = layout.views(workspace, 2)
    before = [view.clone() for view in other_views]
    requests.zero_()
    assert torch.equal(workspace[:16], torch.zeros(16, dtype=torch.uint8))
    assert torch.equal(workspace[16:], torch.full((240,), 255, dtype=torch.uint8))
    for view, original in zip(other_views, before, strict=True):
        assert torch.equal(view, original)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_unknown_device_uses_native_min_chunk(monkeypatch, head_dim):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (99, 0))
    # No short-request override: the native 128-token bound at every length.
    assert implementation.fixed_query_min_chunk(torch.device("cuda"), head_dim=head_dim) == (
        (128, 0),
    )


@pytest.mark.parametrize(
    "capability,head_dim,expected",
    [
        ((12, 0), 128, ((128, 0),)),
        ((12, 0), 256, ((16, 512), (32, 768), (64, 1280))),
        ((12, 1), 256, ((32, 256),)),
        ((12, 1), 128, ((128, 0),)),
        ((12, 0), 64, ((128, 0),)),
        ((12, 1), 64, ((128, 0),)),
    ],
)
def test_measured_min_chunk_table(monkeypatch, capability, head_dim, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert implementation.fixed_query_min_chunk(torch.device("cuda"), head_dim=head_dim) == expected


@pytest.mark.parametrize("min_chunk_tokens", [-1, 0, 129, 256])
def test_invalid_min_chunk_rejected_without_native_cuda(monkeypatch, min_chunk_tokens):
    monkeypatch.setattr(implementation, "version", lambda _name: "0.6.11.post3")
    native = Mock(side_effect=AssertionError("invalid chunk must fail before native setup"))
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=native)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(major=8, multi_processor_count=80),
    )

    def buffer(size, dtype):
        return SimpleNamespace(
            device=torch.device("cuda"),
            dtype=dtype,
            ndim=1,
            is_contiguous=lambda: True,
            numel=lambda: size,
        )

    sizes = FlashInferFixedQueryAttention.workspace_size(
        batch_size=1,
        query_length=16,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        device=torch.device("cuda"),
    )
    with pytest.raises(ValueError, match=r"min_chunk_tokens must be in \[1, 128\]"):
        FlashInferFixedQueryAttention(
            *(buffer(size, torch.uint8) for size in sizes),
            query_indptr=buffer(2, torch.int32),
            kv_indptr=buffer(2, torch.int32),
            kv_indices=buffer(16, torch.int32),
            last_page_lengths=buffer(1, torch.int32),
            query_length=16,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
            min_chunk_tokens=min_chunk_tokens,
        )
    native.assert_not_called()


@pytest.mark.parametrize("provided_out", [False, True])
def test_run_disables_pdl_before_consuming_device_schedule(provided_out):
    operation = FlashInferFixedQueryAttention.__new__(FlashInferFixedQueryAttention)
    operation._fp8_output = None
    q = torch.zeros(1, dtype=torch.bfloat16)
    cache = (torch.zeros_like(q), torch.zeros_like(q))
    expected = torch.zeros_like(q)
    operation._native = SimpleNamespace(run=Mock(return_value=expected))
    kwargs = {"q": q, "paged_kv_cache": cache}
    if provided_out:
        kwargs["out"] = expected
    assert operation.run(**kwargs) is expected
    operation._native.run.assert_called_once_with(**kwargs, enable_pdl=False)


def test_only_the_first_run_after_prepare_disables_pdl(monkeypatch):
    operation = FlashInferFixedQueryAttention.__new__(FlashInferFixedQueryAttention)
    operation._fp8_output = None
    q = torch.zeros(1, dtype=torch.bfloat16)
    operation._native = SimpleNamespace(run=Mock(return_value=q))
    kwargs = {"q": q, "paged_kv_cache": (q, q), "out": q}

    def flags():
        return [call.kwargs["enable_pdl"] for call in operation._native.run.call_args_list]

    for _ in range(3):
        operation.run(**kwargs)
    # prepare() sets the flag before its launch; model its effect without a GPU.
    operation._serialize_next = True
    for _ in range(2):
        operation.run(**kwargs)
    assert flags() == [False, True, True, False, True]
    monkeypatch.setenv("CTI_FIXED_QUERY_PDL", "0")
    operation.run(**kwargs)
    assert flags()[-1] is False


@pytest.mark.parametrize("tile", [16, 64, 128])
def test_fp8_native_tiles_are_accepted(tile):
    layout = _layout()._replace(query_tile_size=tile)
    assert _PlanLayout.from_native(layout, batch_size=2, query_length=16) == layout


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("options", [{}, {"window_left": 128}, {"logits_soft_cap": 30.0}])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("qualified", [False, True])
def test_fixed_query_dtype_options_and_native_minimum(
    monkeypatch, kv_dtype, options, dtype, qualified
):
    monkeypatch.setattr(implementation, "qualified_flashinfer_version", lambda: qualified)
    monkeypatch.setattr(implementation, "staged_output_flashinfer_version", lambda: qualified)
    staged = Mock()
    factory = Mock(return_value=staged)
    monkeypatch.setattr("ctikernels.attention._scale_cast.FP16StagedOutput", factory)
    monkeypatch.setattr(implementation, "version", lambda _: "0.7.0.post1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, multi_processor_count=170),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: Mock())
    layout = _layout()._replace(query_count=16)
    native = Mock(_plan_info=layout)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=Mock(return_value=native)),
    )
    monkeypatch.setattr(_PlanLayout, "views", lambda *a: ())

    def buffer(size, dtype):
        return SimpleNamespace(
            device=torch.device("cuda"),
            dtype=dtype,
            ndim=1,
            is_contiguous=lambda: True,
            numel=lambda: size,
        )

    operation = FlashInferFixedQueryAttention(
        buffer(1 << 30, torch.uint8),
        buffer(1 << 30, torch.uint8),
        query_indptr=buffer(2, torch.int32),
        kv_indptr=buffer(2, torch.int32),
        kv_indices=buffer(16, torch.int32),
        last_page_lengths=buffer(1, torch.int32),
        query_length=16,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=256,
        page_size=1,
        dtype=dtype,
        kv_dtype=kv_dtype,
        causal=True,
        sm_scale=0.07,
        **options,
    )
    assert native.plan.call_args.kwargs["kv_data_type"] == kv_dtype
    assert native.plan.call_args.kwargs["q_data_type"] == dtype
    assert native.plan.call_args.kwargs["sm_scale"] == 0.07
    for name, value in options.items():
        assert native.plan.call_args.kwargs[name] == value
    expected = (
        ((16, 512), (32, 768), (64, 1280)) if kv_dtype == dtype and not options else ((128, 0),) * 3
    )
    assert operation._chunk_tiers == expected
    operation.run(q=None, paged_kv_cache=None, k_scale=0.5, v_scale=1.5)
    if not (qualified and dtype == torch.bfloat16 and kv_dtype != dtype):
        factory.assert_not_called()
        assert "o_data_type" not in native.plan.call_args.kwargs
        native.run.assert_called_once_with(
            q=None, paged_kv_cache=None, k_scale=0.5, v_scale=1.5, enable_pdl=False
        )
    else:
        factory.assert_called_once_with((16, 16, 256), torch.device("cuda"))
        assert native.plan.call_args.kwargs["o_data_type"] == torch.float16
        staged.prepare.assert_called_once_with(
            native, head_dim=256, idx_dtype=torch.int32, decode=False
        )
        staged.run.assert_called_once_with(
            native, q=None, paged_kv_cache=None, out=None, k_scale=0.5, v_scale=1.5, pdl=False
        )
        native.run.assert_not_called()


def test_staged_fixed_query_preserves_pdl_sequence(monkeypatch):
    monkeypatch.delenv("CTI_FIXED_QUERY_PDL", raising=False)
    operation = FlashInferFixedQueryAttention.__new__(FlashInferFixedQueryAttention)
    operation._native = Mock()
    operation._fp8_output = Mock()
    q = torch.zeros(1, dtype=torch.bfloat16)
    for _ in range(2):
        operation.run(q=q, paged_kv_cache=(q, q), out=q)
    operation._serialize_next = True
    operation.run(q=q, paged_kv_cache=(q, q), out=q)
    monkeypatch.setenv("CTI_FIXED_QUERY_PDL", "0")
    operation.run(q=q, paged_kv_cache=(q, q), out=q)
    assert [call.kwargs["pdl"] for call in operation._fp8_output.run.call_args_list] == [
        False,
        True,
        False,
        False,
    ]
    operation._native.run.assert_not_called()
