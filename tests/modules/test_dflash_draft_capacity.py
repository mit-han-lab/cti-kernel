# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FlashInfer DRAFT admission without allocations or a CUDA installation."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute, dispatch, flashinfer, flashinfer_mask
from ctikernels.attention.draft import DraftAttention, _provider


def geometry(**overrides):
    result = {
        "batch_size": 1,
        "query_length": 8,
        "max_kv_tokens": 4096,
        "page_size": 16,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "dtype": torch.bfloat16,
        "causal": False,
        "window": None,
        "device": torch.device("cuda"),
    }
    return result | overrides


@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
@pytest.mark.parametrize("batch,capacity", [(4, 2**29), (3, (2**31 + 2) // 3)])
def test_draft_int32_csr_bound_before_cuda(monkeypatch, provider, batch, capacity):
    from ctikernels.attention import draft

    def unexpected(*args, **kwargs):
        pytest.fail("CSR rejection must precede provider/device/workspace access")

    monkeypatch.setattr(draft, "_provider", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    args = geometry(batch_size=batch, max_kv_tokens=capacity)
    with pytest.raises(ValueError, match="CSR capacities must fit signed int32"):
        DraftAttention.workspace_size(**args, provider=provider)
    for name in ("batch_size", "max_kv_tokens", "device"):
        args.pop(name)
    with pytest.raises(ValueError, match="CSR capacities must fit signed int32"):
        DraftAttention(
            None,
            None,
            request_indices=SimpleNamespace(
                ndim=1, numel=lambda: batch, device=torch.device("cuda")
            ),
            token_indices=SimpleNamespace(ndim=1, numel=lambda: batch * capacity),
            positions=None,
            token_indptr=None,
            query_indptr=None,
            page_indptr=None,
            page_indices=None,
            last_page_lengths=None,
            provider=provider,
            **args,
        )


@pytest.mark.parametrize("page,capacity", [(1, 2**30), (16, 2**31 - 16)])
def test_largest_full_capacity(page, capacity):
    flashinfer.check_capacity(**geometry(page_size=page, max_kv_tokens=capacity))
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        flashinfer.check_capacity(**geometry(page_size=page, max_kv_tokens=capacity + 1))


@pytest.mark.parametrize("window", [None, 2048, 4096])
def test_original_overflow_is_rejected_even_with_window(window):
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        flashinfer.check_capacity(**geometry(max_kv_tokens=2**31 - 1, window=window))


@pytest.mark.parametrize("tile", [16, 64, 128])
@pytest.mark.parametrize("window", [128, 129, 2**31 - 1, 2**63])
def test_large_window_is_clamped_to_capacity(tile, window):
    flashinfer.check_capacity(**geometry(max_kv_tokens=128, window=window), query_tile_size=tile)


@pytest.mark.parametrize("window", [2048, 4096])
def test_window_bounds_schedule_but_not_page_csr(window):
    flashinfer.check_capacity(**geometry(page_size=1, max_kv_tokens=2**31 - 1, window=window))


@pytest.mark.parametrize(
    "inside,outside",
    [
        # Token/page CSR totals, including the B*Q total-row count.
        (
            {"batch_size": (2**31 - 1) // 16, "max_kv_tokens": 16, "query_length": 16},
            {"batch_size": (2**31 - 1) // 16 + 1, "max_kv_tokens": 16, "query_length": 16},
        ),
        # Masked request lanes also multiply the padded request index by Q.
        (
            {"batch_size": 2**26, "query_length": 17, "max_kv_tokens": 17, "page_size": 32},
            {"batch_size": 2**26 + 1, "query_length": 17, "max_kv_tokens": 17, "page_size": 32},
        ),
        # The shortest qualified tier permits 32 chunks/request at C=512, P=1.
        (
            {
                "batch_size": (2**31 - 1) // 1024,
                "query_length": 32,
                "max_kv_tokens": 512,
                "page_size": 1,
            },
            {
                "batch_size": (2**31 - 1) // 1024 + 1,
                "query_length": 32,
                "max_kv_tokens": 512,
                "page_size": 1,
            },
        ),
    ],
)
def test_batch_and_merge_boundaries(inside, outside):
    flashinfer.check_capacity(**geometry(**inside))
    with pytest.raises(NotImplementedError, match="int32"):
        flashinfer.check_capacity(**geometry(**outside))


def test_query_tile_count_and_padded_tile_lanes():
    args = geometry(max_kv_tokens=128)
    # Eight pages/chunks in the shortest tier; query tile count can exceed Q.
    count = (2**31 - 1) // 8
    flashinfer.check_capacity(**args, query_tile_size=64, query_tile_count=count)
    with pytest.raises(NotImplementedError, match="int32"):
        flashinfer.check_capacity(**args, query_tile_size=64, query_tile_count=count + 1)
    flashinfer.check_capacity(**args, padded_ctas=2**31)
    with pytest.raises(NotImplementedError, match="int32"):
        flashinfer.check_capacity(**args, padded_ctas=2**31 + 1)


@pytest.mark.parametrize("provider", ["flashinfer", None])
def test_workspace_rejects_before_provider_sizing(monkeypatch, provider):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)

    def unsupported(**kwargs):
        raise NotImplementedError("CuTe unavailable")

    def unexpected(*args, **kwargs):
        pytest.fail("capacity rejection must precede device access and sizing")

    monkeypatch.setattr(cute, "check_support", unsupported)
    monkeypatch.setattr(torch.cuda, "get_device_properties", unexpected)
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        DraftAttention.workspace_size(**geometry(max_kv_tokens=2**31 - 1), provider=provider)


def test_auto_overrides_preference_only_when_capacity_requires_it(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    monkeypatch.setattr(dispatch, "prefers_flashinfer", lambda **kwargs: True)
    assert _provider(None, geometry()) is flashinfer
    oversized = geometry(max_kv_tokens=2**31 - 1)
    assert _provider(None, oversized) is cute
    for provider in ("flashinfer", None):
        monkeypatch.setenv("CTI_DRAFT_PROVIDER", "flashinfer")
        with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
            _provider(provider, oversized)


def test_prepare_rechecks_host_capacity_before_touching_buffers():
    provider = object.__new__(flashinfer.Provider)
    provider._prepare_inputs = (None, None, None, None, 2**31 - 1)
    provider._batch_size, provider._page_size, provider._query_length = 1, 16, 8
    provider._window_left, provider._query_tiles = -1, 1
    provider._causal = False
    provider._layout = SimpleNamespace(query_tile_size=64, padded_ctas=128)
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        provider.prepare()


def test_construction_rejects_before_metadata_or_workspace_access():
    args = geometry()
    for name in ("batch_size", "max_kv_tokens", "device"):
        args.pop(name)
    slots = SimpleNamespace(ndim=1, numel=lambda: 1, device=torch.device("cuda"))
    tokens = SimpleNamespace(ndim=1, numel=lambda: 2**31 - 1)
    with pytest.raises(NotImplementedError, match="FlashInfer DRAFT.*int32"):
        DraftAttention(
            None,
            None,
            request_indices=slots,
            token_indices=tokens,
            positions=None,
            token_indptr=None,
            query_indptr=None,
            page_indptr=None,
            page_indices=None,
            last_page_lengths=None,
            provider="flashinfer",
            **args,
        )


@pytest.mark.parametrize("window", [1, 2, 7])
@pytest.mark.parametrize("causal", [False, True])
def test_fa2_window_admission(monkeypatch, window, causal):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    args = geometry(window=window, causal=causal)
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    monkeypatch.setattr(dispatch, "prefers_flashinfer", lambda **kwargs: True)
    if causal:
        flashinfer.check_capacity(**args)
        assert _provider("flashinfer", args) is flashinfer
        assert _provider(None, args) is flashinfer
    else:
        with pytest.raises(NotImplementedError, match="FA2 window_left has no right bound"):
            flashinfer.check_capacity(**args)
        assert _provider(None, args) is cute
        with pytest.raises(NotImplementedError, match="FA2 window_left has no right bound"):
            DraftAttention.workspace_size(**args, provider="flashinfer")

        def unsupported(**kwargs):
            raise NotImplementedError("CuTe unavailable")

        monkeypatch.setattr(cute, "check_support", unsupported)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170),
        )
        assert _provider(None, args) is flashinfer_mask
        assert DraftAttention.workspace_size(**args) == DraftAttention.workspace_size(
            **args, provider="flashinfer-mask"
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [8, 64, 8191, None])
def test_fa2_window_at_least_query(window, causal):
    flashinfer.check_capacity(**geometry(window=window, causal=causal))


def test_narrow_noncausal_construction_rejects_before_metadata():
    args = geometry(window=1)
    for name in ("batch_size", "max_kv_tokens", "device"):
        args.pop(name)
    slots = SimpleNamespace(ndim=1, numel=lambda: 1, device=torch.device("cuda"))
    tokens = SimpleNamespace(ndim=1, numel=lambda: 4096)
    with pytest.raises(NotImplementedError, match="FA2 window_left has no right bound"):
        DraftAttention(
            None,
            None,
            request_indices=slots,
            token_indices=tokens,
            positions=None,
            token_indptr=None,
            query_indptr=None,
            page_indptr=None,
            page_indices=None,
            last_page_lengths=None,
            provider="flashinfer",
            **args,
        )


@pytest.mark.parametrize("window", [128, 129, 2**31 - 1, 2**63])
@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
def test_public_sizing_normalizes_window_before_selection(monkeypatch, window, provider):
    from ctikernels.attention import draft

    seen = []

    def select(selected, configuration):
        assert selected == provider
        seen.append(configuration["window"])
        return flashinfer

    monkeypatch.setattr(draft, "_provider", select)
    monkeypatch.setattr(draft, "_workspace", lambda implementation, configuration: (256, 256))
    args = geometry(max_kv_tokens=128, window=window)
    assert DraftAttention.workspace_size(**args, provider=provider) == (256, 256)
    assert DraftAttention.requires_page_indices(**args, provider=provider)
    assert seen == [None, None]


@pytest.mark.parametrize("provider", ["cute", "flashinfer"])
@pytest.mark.parametrize("window", [64, 128, 129, 2**31 - 1])
@pytest.mark.parametrize("row_stride", [None, 6144])
def test_construction_normalizes_and_optional_page_indices(
    monkeypatch, provider, window, row_stride
):
    from ctikernels.attention import draft

    args = geometry(max_kv_tokens=128, window=window, device="cpu")
    args.pop("batch_size")
    args.pop("max_kv_tokens")
    args.pop("device")
    args["query_row_stride"] = row_stride
    selected = cute if provider == "cute" else flashinfer
    monkeypatch.setattr(draft, "_geometry", lambda **kwargs: None)
    expected = 64 if window == 64 else None

    def select(name, configuration):
        assert configuration["window"] == expected
        return selected

    monkeypatch.setattr(draft, "_provider", select)
    monkeypatch.setattr(draft, "_workspace", lambda *args: (0, 0))
    received = {}

    def build(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(pdl=False, merge="legacy")

    monkeypatch.setattr(selected, "Provider", build)
    inputs = {
        "request_indices": torch.empty(1, dtype=torch.int32),
        "positions": torch.empty(8, dtype=torch.int64),
        "token_indptr": torch.empty(2, dtype=torch.int32),
        "token_indices": torch.empty(128, dtype=torch.int32),
        "query_indptr": torch.empty(2, dtype=torch.int32),
        "page_indptr": torch.empty(2, dtype=torch.int32),
        "page_indices": None,
        "last_page_lengths": torch.empty(1, dtype=torch.int32),
    }
    workspaces = [torch.empty(0, dtype=torch.uint8) for _ in range(2)]
    if provider == "flashinfer":
        with pytest.raises(ValueError, match="page_indices.*required.*FlashInfer"):
            DraftAttention(*workspaces, **inputs, **args, provider=provider)
        return
    op = DraftAttention(*workspaces, **inputs, **args, provider=provider)
    assert received["kv_indices"] is None
    assert received["window_left"] == (-1 if expected is None else expected - 1)
    assert received["query_row_stride"] == row_stride
    assert op._query_row_stride == (4096 if row_stride is None else row_stride)
    assert all(entry[0] is not None for entry in op._metadata)
    assert not op.pdl and op.merge == "legacy"
    for name in ("provider", "pdl", "merge"):
        with pytest.raises(AttributeError):
            setattr(op, name, None)
    op._implementation.pdl = True
    op._implementation.merge = "tree"
    assert op.pdl and op.merge == "tree"


@pytest.mark.parametrize("row_stride", [True, 4096.0, "4096", 0, -8, 4088, 4097])
def test_query_row_stride_rejected_before_provider_access(monkeypatch, row_stride):
    from ctikernels.attention import draft

    def unexpected(*args, **kwargs):
        pytest.fail("stride validation must precede provider access")

    monkeypatch.setattr(draft, "_provider", unexpected)
    args = geometry()
    for name in ("batch_size", "max_kv_tokens", "device"):
        args.pop(name)
    slots = SimpleNamespace(ndim=1, numel=lambda: 1, device=torch.device("cuda"))
    tokens = SimpleNamespace(ndim=1, numel=lambda: 4096)
    with pytest.raises(ValueError, match="query_row_stride.*int.*Hq\\*D.*16-byte"):
        DraftAttention(
            None,
            None,
            request_indices=slots,
            token_indices=tokens,
            positions=None,
            token_indptr=None,
            query_indptr=None,
            page_indptr=None,
            page_indices=None,
            last_page_lengths=None,
            query_row_stride=row_stride,
            **args,
        )


@pytest.mark.parametrize("name", ["prepare_draft", "prepare_split_kv"])
def test_runtime_window_has_one_triton_specialization(name):
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime.jit import create_function_from_signature

    from ctikernels.attention import _flashinfer_schedule

    kernel = getattr(_flashinfer_schedule, name)
    backend = CUDABackend(GPUTarget("cuda", 120, 32))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    # No launch/device access: exercise Triton's actual argument specialization.
    args = {param.name: 16 for param in kernel.params}
    keys = [
        binder(**(args | {"WINDOW_LEFT": value}))[1]
        for value in (-1, 0, 1, 16, 63, 2047, 2999, 4095, 8190)
    ]
    assert all(key == keys[0] for key in keys)
    window = next(param for param in kernel.params if param.name == "WINDOW_LEFT")
    assert not window.is_constexpr


def test_flashinfer_effective_settings_are_read_only(monkeypatch):
    provider = object.__new__(flashinfer.Provider)
    monkeypatch.delenv("CTI_FIXED_QUERY_PDL", raising=False)
    assert provider.pdl is True
    monkeypatch.setenv("CTI_FIXED_QUERY_PDL", "0")
    assert provider.pdl is False
    assert provider.merge == "flashinfer"
    for name in ("pdl", "merge"):
        with pytest.raises(AttributeError):
            setattr(provider, name, None)
