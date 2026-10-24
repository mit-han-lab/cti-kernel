# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU staging contracts; kernel compilation is exercised by _scale_cast_compile."""

import contextlib
import math
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _flashinfer as compatibility
from ctikernels.attention import _scale_cast as implementation


@pytest.fixture
def stage(monkeypatch):
    kernel = Mock()
    monkeypatch.setattr(implementation, "_compile", Mock(return_value=kernel))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    value = implementation.FP16StagedOutput((2, 4, 128), "cpu")
    implementation._compile.assert_called_once_with(torch.float16, torch.bfloat16, "sm_120")
    return value


def test_plan_reserves_capacity_and_reuses_for_smaller_shapes(stage):
    first = stage.stage.data_ptr()
    stage.reserve((1, 4, 128))
    assert stage.stage.data_ptr() == first
    stage.reserve((3, 4, 128))
    assert stage.stage.data_ptr() != first
    assert stage._storage.numel() == 3 * 4 * 128
    larger = stage.stage.data_ptr()
    stage.reserve((2, 4, 128))
    assert stage.stage.data_ptr() == larger


@pytest.mark.parametrize("shape", [(0, 4, 128), (1, 1, 7), (1, 128), (1, True, 128)])
def test_invalid_shape(stage, shape):
    with pytest.raises(ValueError):
        stage.reserve(shape)


@pytest.mark.parametrize("scale", [None, 0.0, -0.31, -1.0, 0.5, 1.0])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("provided", [False, True])
def test_stage_bypasses_provider_scale_and_launches_once(stage, scale, pdl, provided):
    q = torch.empty(stage.stage.shape, dtype=torch.bfloat16)
    cache = (torch.empty(8), torch.empty(8))
    out = torch.empty_like(q) if provided else None
    native = Mock()
    pointer = stage.stage.data_ptr()
    actual = stage.run(
        native, q=q, paged_kv_cache=cache, out=out, k_scale=0.59, v_scale=scale, pdl=pdl
    )
    if provided:
        assert actual is out
    assert actual.shape == q.shape and actual.dtype == q.dtype
    assert stage.last_output_dtype == torch.float16
    native.run.assert_called_once_with(
        q=q, paged_kv_cache=cache, out=stage.stage, k_scale=0.59, v_scale=None, enable_pdl=pdl
    )
    stage._kernel.assert_called_once()
    source, destination, passed_scale, flag = stage._kernel.call_args.args
    assert source.data_ptr() == pointer and destination.data_ptr() == actual.data_ptr()
    assert source.ndim == destination.ndim == 1
    assert passed_scale == (1.0 if scale is None else scale) or math.isnan(passed_scale)
    assert flag == pdl


@pytest.mark.parametrize("value", [torch.tensor(1.0), "1", True, object()])
def test_scale_does_not_coerce_tensors(value):
    with pytest.raises(ValueError, match="host float"):
        implementation._scale(value)


@pytest.mark.parametrize("bad", ["shape", "dtype", "strides", "grad"])
def test_output_validation_precedes_native(stage, bad):
    q = torch.empty(stage.stage.shape, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    if bad == "shape":
        out = out[:1]
    elif bad == "dtype":
        out = out.float()
    elif bad == "strides":
        out = torch.empty(2, 4, 256, dtype=q.dtype)[..., ::2]
    else:
        out.requires_grad_(True)
    native = Mock()
    with pytest.raises(ValueError):
        stage.run(native, q=q, paged_kv_cache=(q, q), out=out, k_scale=None, v_scale=None)
    native.run.assert_not_called()
    stage._kernel.assert_not_called()


def test_unsupported_compile_pair_fails_before_dsl_import():
    with pytest.raises(ValueError, match="FP16 to BF16"):
        implementation._compile(torch.bfloat16, torch.bfloat16, "sm_120")


def test_pdl_is_disabled_on_ampere(stage):
    stage.supports_pdl = False
    q = torch.empty(stage.stage.shape, dtype=torch.bfloat16)
    native = Mock()
    stage.run(native, q=q, paged_kv_cache=(q, q), out=q, k_scale=None, v_scale=1, pdl=True)
    assert native.run.call_args.kwargs["enable_pdl"] is False
    assert stage._kernel.call_args.args[-1] is False


@pytest.mark.parametrize(
    "scale", [2.0, -2.0, 1e8, 1e3, math.nextafter(1.0, 2.0), math.inf, -math.inf, math.nan]
)
@pytest.mark.parametrize("provided", [False, True])
def test_large_or_nonfinite_scale_uses_bf16_without_stage_allocation(
    stage, monkeypatch, scale, provided
):
    q = torch.empty(stage.stage.shape, dtype=torch.bfloat16)
    out = torch.empty_like(q) if provided else None
    result = out if provided else torch.empty_like(q)
    original, fallback, plan = object(), object(), object()
    native = SimpleNamespace(
        _cached_module=original, _cached_o_data_type=torch.float16, _plan_info=plan
    )
    stage._bf16_output = compatibility.BF16OutputModule.__new__(compatibility.BF16OutputModule)
    stage._bf16_output.module = fallback

    def run(**kwargs):
        assert native._cached_module is fallback
        assert native._cached_o_data_type == torch.bfloat16
        assert native._plan_info is plan
        assert kwargs["q"] is q and kwargs["out"] is out
        assert kwargs["paged_kv_cache"] == (q, q)
        assert kwargs["v_scale"] is scale and kwargs["k_scale"] == 0.59
        assert kwargs["enable_pdl"] is False
        return result

    native.run = Mock(side_effect=run)
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("unexpected allocation")))
    assert (
        stage.run(
            native, q=q, paged_kv_cache=(q, q), out=out, k_scale=0.59, v_scale=scale, pdl=False
        )
        is result
    )
    native.run.assert_called_once()
    assert stage.last_output_dtype == torch.bfloat16
    assert native._cached_module is original and native._cached_o_data_type == torch.float16
    stage._kernel.assert_not_called()


def test_bf16_module_restores_native_after_failure(stage):
    original = object()
    native = SimpleNamespace(
        _cached_module=original,
        _cached_o_data_type=torch.float16,
        run=Mock(side_effect=RuntimeError("native failure")),
    )
    fallback = compatibility.BF16OutputModule.__new__(compatibility.BF16OutputModule)
    fallback.module = object()
    with pytest.raises(RuntimeError, match="native failure"):
        fallback.run(native, out=None)
    assert native._cached_module is original and native._cached_o_data_type == torch.float16


@pytest.mark.parametrize("release", ["0.6.11.post3", "0.7.0.post1"])
@pytest.mark.parametrize("route", ["cuda_cores", "fa2", "fixed_query"])
@pytest.mark.parametrize("options", [False, True])
def test_qualified_bf16_getter_matches_native_plan(monkeypatch, release, route, options):
    monkeypatch.setattr(compatibility, "qualified_flashinfer_version", lambda: True)
    monkeypatch.setattr(compatibility, "version", lambda _: release)
    decode_getter, prefill_getter = Mock(), Mock()
    for name, attributes in (
        ("flashinfer.decode", {"get_batch_decode_module": decode_getter}),
        ("flashinfer.prefill", {"get_batch_prefill_module": prefill_getter}),
        (
            "flashinfer.utils",
            {"PosEncodingMode": Enum("PosEncodingMode", {"NONE": 0, "ROPE_LLAMA": 1})},
        ),
    ):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    native = SimpleNamespace(
        _cached_q_data_type=torch.bfloat16,
        _cached_kv_data_type=torch.float8_e5m2,
        _cached_o_data_type=torch.float16,
        _cached_module=object(),
        _jit_module=None,
        _backend="fa2",
        use_tensor_cores=route == "fa2",
        _pos_encoding_mode="ROPE_LLAMA" if options else "NONE",
        _window_left=128 if options else -1,
        _logits_soft_cap=30.0 if options else 0.0,
        _use_fp16_qk_reduction=options,
    )
    original = native._cached_module
    fallback = compatibility.BF16OutputModule(
        native, head_dim=128, idx_dtype=torch.int32, decode=route != "fixed_query"
    )
    args = (
        torch.bfloat16,
        torch.float8_e5m2,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        int(options),
        options,
        options,
    )
    if route == "cuda_cores":
        decode_getter.assert_called_once_with(*args)
        prefill_getter.assert_not_called()
        assert fallback.module is decode_getter.return_value
    else:
        prefill_getter.assert_called_once_with(
            "fa2", *args, options if route == "fixed_query" else False
        )
        decode_getter.assert_not_called()
        assert fallback.module is prefill_getter.return_value
    assert native._cached_module is original and native._cached_o_data_type == torch.float16


@pytest.mark.parametrize(
    "release,qualified,expected",
    [
        ("0.6.11.post3", True, True),
        ("0.7.0.post1", True, True),
        ("0.7.0.post1", False, False),
        ("0.7.1", True, False),
        ("0.8.0", True, False),
    ],
)
def test_output_swap_version_gate(monkeypatch, release, qualified, expected):
    monkeypatch.setattr(compatibility, "qualified_flashinfer_version", lambda: qualified)
    monkeypatch.setattr(compatibility, "version", lambda _: release)
    assert compatibility.staged_output_flashinfer_version() is expected
    if not expected:
        with pytest.raises(RuntimeError, match="unqualified"):
            compatibility.BF16OutputModule(None, head_dim=128, idx_dtype=torch.int32, decode=True)
