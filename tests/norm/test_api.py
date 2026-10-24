# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU tests of the norm op: interface, selection truth table, error types."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import pytest
import torch

import ctikernels.norm
from benchmark.kernel_tuning.norm_act_reference import norm_reference, vllm_norm_native
from ctikernels import _flashinfer_version
from ctikernels._norm_act_reroute import Provider, RerouteError
from ctikernels.norm import cute
from ctikernels.norm import inference as norm
from ctikernels.norm._dispatch import (
    _FP32_BANDS,
    _LARGE,
    TABLE,
    Config,
    Entry,
    entry_for,
    parse_config,
    tune_configs,
)
from ctikernels.norm.cute import _provider

# interface and argument errors


_LOCAL_RMS_NORM = "ctikernels.norm._zero_centered_rms_norm"


def _recorder(calls: list[tuple[str, tuple[object, ...]]], name: str, result):
    def call(*args, **kwargs):
        calls.append((name, args + tuple(kwargs.items())))
        return result(*args, **kwargs)

    return call


def _load_inference_norm(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, tuple[object, ...]]],
) -> ModuleType:
    """Import the leaf with recording stand-ins for FlashInfer and the local kernels."""

    monkeypatch.setattr(_flashinfer_version, "installed_version", lambda: "0.7.0.post1")
    monkeypatch.setattr(_flashinfer_version, "importable", lambda: True)
    flashinfer = ModuleType("flashinfer")
    flashinfer.rmsnorm = _recorder(
        calls, "flashinfer.rmsnorm", lambda x, *_, out=None: x.clone() if out is None else out
    )
    flashinfer.fused_add_rmsnorm = _recorder(calls, "flashinfer.fused_add_rmsnorm", lambda *_: None)
    local_rms_norm = ModuleType(_LOCAL_RMS_NORM)
    local_rms_norm.rms_norm = _recorder(calls, "local.rms_norm", lambda x, *_, **__: x.clone())
    local_rms_norm.rms_norm_ = _recorder(calls, "local.rms_norm_", lambda x, *_, **__: x)
    local_rms_norm.add_rms_norm_ = _recorder(
        calls, "local.add_rms_norm_", lambda x, r, *_, **__: (x, r)
    )
    local_rms_norm.residual_rms_norm = _recorder(
        calls, "local.residual_rms_norm", lambda x, *_, **__: x.clone()
    )

    monkeypatch.setattr(ctikernels.norm, "_zero_centered_rms_norm", local_rms_norm, raising=False)
    for module in (flashinfer, local_rms_norm):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "ctikernels.norm.inference", raising=False)
    return importlib.import_module("ctikernels.norm.inference")


def test_inference_norm_leaf_imports_local_kernels_lazily(monkeypatch) -> None:
    for name in ("ctikernels.norm.inference", _LOCAL_RMS_NORM):
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("ctikernels.norm.inference")

    assert _LOCAL_RMS_NORM not in sys.modules


@pytest.mark.parametrize("zero_centered", [False, True])
def test_rms_norms_route_by_weight_convention(monkeypatch, zero_centered: bool) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    residual = torch.randn(3, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.bfloat16)

    normalized = operation.rms_norm(x, weight, 1e-6, zero_centered=zero_centered)
    assert operation.rms_norm_(x, weight, 1e-6, zero_centered=zero_centered) is x
    returned = operation.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero_centered)

    assert normalized is not x
    assert returned[0] is x and returned[1] is residual
    if zero_centered:
        assert calls == [
            ("local.rms_norm", (x, weight, 1e-6, ("out_dtype", None), ("zero_centered", True))),
            ("local.rms_norm_", (x, weight, 1e-6, ("zero_centered", True))),
            (
                "local.add_rms_norm_",
                (
                    x,
                    residual,
                    weight,
                    1e-6,
                    ("store_residual", True),
                    ("normalize_stored", False),
                    ("zero_centered", True),
                ),
            ),
        ]
    else:
        assert calls == [
            ("flashinfer.rmsnorm", (x, weight, 1e-6)),
            ("flashinfer.rmsnorm", (x, weight, 1e-6, ("out", x))),
            (
                "local.add_rms_norm_",
                (
                    x,
                    residual,
                    weight,
                    1e-6,
                    ("store_residual", True),
                    ("normalize_stored", False),
                    ("zero_centered", False),
                ),
            ),
        ]


@pytest.mark.parametrize(
    "name,gradient_input,message",
    [
        ("rms_norm", "weight", "frozen weight"),
        ("rms_norm_", "x", "cannot modify"),
        ("rms_norm_", "weight", "frozen weight"),
        ("add_rms_norm_", "residual", "cannot modify"),
        ("add_rms_norm_", "weight", "frozen weight"),
        ("residual_rms_norm", "residual", "does not support autograd"),
        ("residual_rms_norm", "weight", "frozen weight"),
    ],
)
def test_zero_centered_guards_run_before_any_kernel(
    monkeypatch, name: str, gradient_input: str, message: str
) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    tensors = {
        "x": torch.randn(3, 8, requires_grad=gradient_input == "x"),
        "residual": torch.randn(3, 8, requires_grad=gradient_input == "residual"),
        "weight": torch.randn(8, requires_grad=gradient_input == "weight"),
    }
    calls_by_name = {
        "rms_norm": lambda t: operation.rms_norm(t["x"], t["weight"], 1e-6, zero_centered=True),
        "rms_norm_": lambda t: operation.rms_norm_(t["x"], t["weight"], 1e-6, zero_centered=True),
        "add_rms_norm_": lambda t: operation.add_rms_norm_(
            t["x"], t["residual"], t["weight"], 1e-6, zero_centered=True
        ),
        "residual_rms_norm": lambda t: operation.residual_rms_norm(
            t["residual"], t["weight"], 1e-6, zero_centered=True, out_dtype=torch.float32
        ),
    }

    with pytest.raises(ValueError if gradient_input == "weight" else RuntimeError, match=message):
        calls_by_name[name](tensors)

    assert calls == []


def test_zero_centered_passes_out_dtype_and_an_fp32_residual_through(monkeypatch) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    residual = torch.randn(3, 8)
    weight = torch.randn(8, dtype=torch.bfloat16)

    operation.rms_norm(residual, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16)
    operation.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True)

    assert calls == [
        (
            "local.rms_norm",
            (residual, weight, 1e-6, ("out_dtype", torch.bfloat16), ("zero_centered", True)),
        ),
        (
            "local.add_rms_norm_",
            (
                x,
                residual,
                weight,
                1e-6,
                ("store_residual", True),
                ("normalize_stored", False),
                ("zero_centered", True),
            ),
        ),
    ]


def test_residual_norm_entries_reach_the_local_kernels(monkeypatch) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    residual = torch.randn(3, 8)
    weight = torch.randn(8, dtype=torch.bfloat16)

    normalized = operation.residual_rms_norm(
        residual, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16
    )
    returned = operation.add_rms_norm_(
        x, residual, weight, 1e-6, zero_centered=True, store_residual=False
    )
    operation.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=True, normalize="stored_sum")

    assert normalized is not residual
    assert returned[0] is x and returned[1] is residual
    assert calls == [
        (
            "local.residual_rms_norm",
            (residual, weight, 1e-6, ("out_dtype", torch.bfloat16), ("zero_centered", True)),
        ),
        (
            "local.add_rms_norm_",
            (
                x,
                residual,
                weight,
                1e-6,
                ("store_residual", False),
                ("normalize_stored", False),
                ("zero_centered", True),
            ),
        ),
        (
            "local.add_rms_norm_",
            (
                x,
                residual,
                weight,
                1e-6,
                ("store_residual", True),
                ("normalize_stored", True),
                ("zero_centered", True),
            ),
        ),
    ]


def test_add_rms_norm_names_the_value_its_norm_reads(monkeypatch) -> None:
    """Callers tell from the signature alone, before launching anything, that the add+norm
    names the value its norm reads and by default reads the FP32 sum, as FlashInfer's does."""

    operation = _load_inference_norm(monkeypatch, [])
    parameter = inspect.signature(operation.add_rms_norm_).parameters["normalize"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "fp32_sum"


@pytest.mark.parametrize("zero", [False, True])
def test_both_weight_modes_reach_residual_kernels(monkeypatch, zero):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)
    operation.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero, store_residual=False)
    operation.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero, normalize="stored_sum")
    operation.residual_rms_norm(residual, weight, 1e-6, zero_centered=zero, out_dtype=torch.float32)
    assert [name for name, _ in calls] == [
        "local.add_rms_norm_",
        "local.add_rms_norm_",
        "local.residual_rms_norm",
    ]
    assert all(
        dict(args[4:] if name == "local.add_rms_norm_" else args[3:])["zero_centered"] == zero
        for name, args in calls
    )


@pytest.mark.parametrize("zero_centered", [False, True])
def test_add_rms_norm_rejects_an_unknown_normalize_before_any_kernel(
    monkeypatch, zero_centered: bool
) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)

    with pytest.raises(ValueError, match="'fp32_sum' or 'stored_sum'"):
        operation.add_rms_norm_(
            x, residual, weight, 1e-6, zero_centered=zero_centered, normalize="sum"
        )

    assert calls == []


@pytest.mark.parametrize("name", ["rms_norm", "add_rms_norm_"])
def test_flashinfer_rejects_mixed_dtypes_before_any_kernel(monkeypatch, name: str) -> None:
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    residual = torch.randn(3, 8)
    weight = torch.randn(8, dtype=torch.bfloat16)
    original_x, original_residual = x.clone(), residual.clone()

    if name == "rms_norm":
        monkeypatch.setenv(operation.VARIABLE, "flashinfer")
        with pytest.raises(TypeError, match="same dtype"):
            operation.rms_norm(x, weight, 1e-6, out_dtype=torch.float32)
    else:
        with pytest.raises(RuntimeError, match="forced flashinfer requires same-dtype"):
            operation.AddRMSNorm(provider="flashinfer")(x, residual, weight, 1e-6)

    assert calls == []
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


@pytest.mark.parametrize("alias", ("x", "residual", "weight", "gap"))
def test_sum_out_rejects_overlapping_spans_before_launch(monkeypatch, alias):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    if alias == "x":
        summed = x
    elif alias == "residual":
        summed = residual
    elif alias == "weight":
        summed = torch.empty_like(x)
        weight = summed[1]
    else:
        buffer = torch.empty(3, 48)
        summed = buffer[:, 16:32]
        # Inside a row gap of sum_out: spans must conservatively reject this.
        weight = buffer[1, :16]
    with pytest.raises(ValueError, match="must not overlap"):
        operation.add_rms_norm_(x, residual, weight, 1e-6, sum_out=summed)
    assert not calls


@pytest.mark.parametrize(
    "case,error,message",
    [
        ("shape", ValueError, "shape"),
        ("dtype", TypeError, "residual's dtype"),
        ("device", ValueError, "device"),
        ("column_stride", ValueError, "unit hidden stride"),
        ("short_stride", ValueError, "row stride"),
        ("prefix", ValueError, "collapse"),
        ("grad", RuntimeError, "requires gradients"),
    ],
)
def test_sum_out_metadata_errors_precede_launch(monkeypatch, case, error, message):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    outputs = {
        "shape": torch.empty(2, 16),
        "dtype": torch.empty(3, 16, dtype=torch.float16),
        "device": torch.empty(3, 16, device="meta"),
        "column_stride": torch.empty(3, 32)[:, ::2],
        "row_stride": torch.empty(3, 17)[:, :16],
        "short_stride": torch.empty(16).expand(3, 16),
        "alignment": torch.empty(49)[1:].view(3, 16),
        "grad": torch.empty(3, 16, requires_grad=True),
        "prefix": torch.empty(3, 2, 16).transpose(0, 1),
    }
    if case == "prefix":
        x, residual = torch.ones(2, 3, 16), torch.ones(2, 3, 16)
    with pytest.raises(error, match=message):
        operation.add_rms_norm_(x, residual, weight, 1e-6, sum_out=outputs[case])
    assert not calls


@pytest.mark.parametrize("shape", ((3, 16), (2, 3, 16), (0, 16)))
def test_sum_out_collapsible_column_slices_reach_triton(monkeypatch, shape):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.ones(shape), torch.ones(shape), torch.ones(16)
    aux = torch.empty((*shape[:-1], 48))
    summed = aux[..., 16:32]
    operation.AddRMSNorm(provider="triton")(
        x, residual, weight, 1e-6, zero_centered=True, sum_out=summed, store_residual=False
    )
    if not x.numel():
        assert not calls
        return
    kwargs = dict(calls[0][1][4:])
    assert not kwargs["store_residual"]
    assert kwargs["sum_out"].shape == (x.numel() // 16, 16)
    assert kwargs["sum_out"].stride() == (48, 1)


@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("store,has_sum", ((True, True), (False, False), (False, True)))
def test_forced_flashinfer_rejects_sum_and_no_store(monkeypatch, zero, store, has_sum):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x, residual, weight = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    with pytest.raises(RuntimeError, match="forced flashinfer does not support"):
        operation.AddRMSNorm(provider="flashinfer")(
            x,
            residual,
            weight,
            1e-6,
            zero_centered=zero,
            store_residual=store,
            sum_out=torch.empty_like(x) if has_sum else None,
        )
    assert not calls


@pytest.mark.parametrize("shape", [(8,), (2, 3, 4, 8), (2, 1, 3, 4, 8)])
@pytest.mark.parametrize("inplace", [False, True])
def test_ordinary_rmsnorm_higher_rank_is_native(monkeypatch, shape, inplace):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.ones(8, dtype=x.dtype)
    result = (operation.rms_norm_ if inplace else operation.rms_norm)(x, weight, 1e-6)
    rows = calls[0][1][0]
    assert rows.shape == x.shape
    assert rows.data_ptr() == x.data_ptr()
    assert result.shape == x.shape
    assert (result is x) == inplace


@pytest.mark.parametrize(
    "layout", ["transpose_prefix", "row_gaps", "odd_row_pitch", "hidden_stride", "pointer"]
)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_ordinary_rmsnorm_layout_copy_and_writeback(monkeypatch, layout, inplace, rank):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    base = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16)
    if layout == "transpose_prefix":
        x = base.transpose(0, 1)
    elif layout == "row_gaps":
        x = base[..., :8]
    elif layout == "odd_row_pitch":
        x = torch.randn(2, 3, 4, 17, dtype=base.dtype)[..., :8]
    elif layout == "pointer":
        x = torch.randn(2 * 3 * 4 * 8 + 1, dtype=base.dtype)[1:].view(2, 3, 4, 8)
    else:
        x = base[..., ::2]
    if rank < 4:
        x = x[0] if rank == 3 else x[0, 0]
    saved = x.clone()
    weight = torch.ones(x.shape[-1], dtype=x.dtype)
    seen = []

    def rmsnorm(rows, weight, eps, out=None):
        seen.append(rows)
        value = rows + 1  # Recording CPU stand-in makes missing copy-back observable.
        return value if out is None else out.copy_(value)

    monkeypatch.setattr(sys.modules["flashinfer"], "rmsnorm", rmsnorm)
    local = sys.modules[_LOCAL_RMS_NORM]
    monkeypatch.setattr(local, "rms_norm", lambda x, w, e, **kw: rmsnorm(x, w, e))
    monkeypatch.setattr(local, "rms_norm_", lambda x, w, e, **kw: rmsnorm(x, w, e, out=x))
    result = (operation.rms_norm_ if inplace else operation.rms_norm)(x, weight, 1e-6)
    assert torch.equal(result, saved + 1)
    assert result.shape == x.shape
    assert seen[0].shape == x.shape
    pack = layout == "hidden_stride" or (layout == "transpose_prefix" and rank == 4)
    assert (seen[0].data_ptr() == x.data_ptr()) == (not pack)
    provider = operation.last_standalone_provider["rms_norm_" if inplace else "rms_norm"]
    if pack:
        assert provider.reason == "pack"
    if inplace:
        assert result is x
    else:
        assert torch.equal(x, saved)


@pytest.mark.parametrize("shape", [(3, 8), (2, 3, 8)])
@pytest.mark.parametrize("inplace", [False, True])
def test_ordinary_rmsnorm_existing_rank_abi_unchanged(monkeypatch, shape, inplace):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    x = torch.randn(shape, dtype=torch.bfloat16)
    (operation.rms_norm_ if inplace else operation.rms_norm)(x, torch.ones(8), 1e-6)
    assert calls[0][1][0] is x


@pytest.mark.parametrize("shape", [(0, 4096), (0, 3, 4096), (0, 2, 3, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("zero", [False, True])
def test_empty_norm_never_resolves_provider(monkeypatch, shape, dtype, zero):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setattr(operation, "_standalone", lambda *a: pytest.fail("empty launch"))
    monkeypatch.setattr(
        operation.AddRMSNorm, "_resolve", lambda *a: pytest.fail("empty resolution")
    )
    x = torch.empty(shape, dtype=dtype)
    weight = torch.ones(shape[-1], dtype=dtype)
    assert operation.rms_norm(x, weight, 1e-6, zero_centered=zero).shape == x.shape
    assert operation.rms_norm_(x, weight, 1e-6, zero_centered=zero) is x
    result = operation.residual_rms_norm(
        x, weight, 1e-6, zero_centered=zero, out_dtype=torch.float32
    )
    assert result.shape == x.shape and result.dtype == torch.float32
    for provider in ("auto", "cute", "triton", "flashinfer"):
        monkeypatch.setenv(operation.VARIABLE, provider)
        residual = torch.empty_like(x)
        assert operation.add_rms_norm_(x, residual, weight, 1e-6)[0] is x
        assert operation.AddRMSNorm(provider=provider)(x, residual, weight, 1e-6)[0] is x
    assert not calls


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize(
    "name",
    ["rms_norm", "rms_norm_", "residual_rms_norm", "add_rms_norm_", "AddRMSNorm", "flashinfer"],
)
def test_frozen_weight_rejected_before_all_work(monkeypatch, name, zero, grad_enabled):
    calls = []
    operation = _load_inference_norm(monkeypatch, calls)

    x = torch.empty((2, 8), device="meta", requires_grad=True)
    residual = torch.empty_like(x)
    weight = torch.empty(8, device="meta", requires_grad=True)

    def unexpected(*args, **kwargs):
        calls.append("unexpected")
        raise AssertionError("work before frozen weight rejection")

    local = sys.modules[_LOCAL_RMS_NORM]
    for name_ in ("_launch", "_launch_residual_norm", "add_rms_norm_"):
        monkeypatch.setattr(local, name_, unexpected, raising=False)
    monkeypatch.setattr(cute, "get_compiled", unexpected)
    monkeypatch.setattr(operation, "_hardware", unexpected)
    monkeypatch.setattr(operation._flashinfer_version, "allowed", unexpected)
    monkeypatch.setattr(torch, "empty", unexpected)
    monkeypatch.setattr(torch, "empty_like", unexpected)
    monkeypatch.setattr(torch, "zeros", unexpected)
    call = (
        operation.AddRMSNorm()
        if name == "AddRMSNorm"
        else (
            operation.flashinfer.add_rms_norm_ if name == "flashinfer" else getattr(operation, name)
        )
    )
    args = (
        (x, residual, weight, 1e-6)
        if name in ("add_rms_norm_", "AddRMSNorm", "flashinfer")
        else (x, weight, 1e-6)
    )
    kwargs = {"zero_centered": zero}
    if name == "residual_rms_norm":
        kwargs["out_dtype"] = x.dtype
    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="frozen weight"):
        call(*args, **kwargs)
    assert calls == []


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("override", ["auto", "cute", "triton", "flashinfer"])
@pytest.mark.parametrize("usable", [False, True])
def test_standalone_provider_policy_and_cached_reason(monkeypatch, inplace, override, usable):

    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setenv(op.VARIABLE, override)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(op._flashinfer_version, "allowed", lambda: usable)
    monkeypatch.setattr(
        op._flashinfer_version, "unusable_reason", lambda: None if usable else "not importable"
    )
    x, weight = torch.ones(2, 8, dtype=torch.bfloat16), torch.ones(8, dtype=torch.bfloat16)
    name = "rms_norm_" if inplace else "rms_norm"
    call = getattr(op, name)
    if override == "flashinfer" and not usable:
        with pytest.raises(RuntimeError, match="not importable"):
            call(x, weight, 1e-6)
        assert calls == []
        return
    for _ in range(2):
        call(x, weight, 1e-6)
        provider = op.last_standalone_provider[name]
        assert provider == ("flashinfer" if usable and override != "triton" else "triton")
        if not usable and override != "triton":
            assert isinstance(provider, Provider) and provider.reason == "not importable"
    assert op._resolve_standalone.cache_info().hits == 1
    if not usable and override != "triton":
        calls.clear()
        size = op._resolve_standalone.cache_info().currsize
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
        with pytest.raises(RerouteError):
            call(x, weight, 1e-6)
        assert calls == [] and op._resolve_standalone.cache_info().currsize == size


@pytest.mark.parametrize(
    "dtype,weight_dtype,out_dtype",
    [
        (torch.float32, torch.float32, None),
        (torch.bfloat16, torch.float32, None),
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.float16, torch.float16, torch.bfloat16),
    ],
)
def test_standard_generic_dtype_routes_are_not_reroutes(
    monkeypatch, dtype, weight_dtype, out_dtype
):
    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setattr(op._flashinfer_version, "allowed", lambda: False)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    x, weight = torch.ones(2, 8, dtype=dtype), torch.ones(8, dtype=weight_dtype)
    op.rms_norm(x, weight, 1e-6, out_dtype=out_dtype)
    assert calls[0][0] == "local.rms_norm"
    assert calls[0][1][-2:] == (("out_dtype", out_dtype), ("zero_centered", False))
    assert op.last_standalone_provider["rms_norm"] == "triton"
    assert getattr(op.last_standalone_provider["rms_norm"], "reason", None) is None


@pytest.mark.parametrize("inplace", [False, True])
def test_engine_checks_reject_before_layout_copy(monkeypatch, inplace):

    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setattr(op._flashinfer_version, "allowed", lambda: False)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    x = torch.ones(2, 16, dtype=torch.bfloat16)[:, ::2]
    weight = torch.ones(8, dtype=x.dtype)
    monkeypatch.setattr(
        op, "_two_strides", lambda *a: pytest.fail("layout check before resolution")
    )
    with pytest.raises(RerouteError):
        (op.rms_norm_ if inplace else op.rms_norm)(x, weight, 1e-6)
    assert calls == []


@pytest.mark.parametrize("layout", ["hidden", "prefix"])
@pytest.mark.parametrize("store", [False, True])
def test_add_pack_is_visible_and_copies_only_modified_inputs(monkeypatch, layout, store):

    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    x = torch.ones(2, 3, 4, 16)
    x = x[..., ::2] if layout == "hidden" else x.transpose(0, 1)
    residual = x.clone(memory_format=torch.preserve_format)
    saved_x, saved_r = x.clone(), residual.clone()
    seen = []

    def launch(x, r, w, eps, **kw):
        seen.append((x.is_contiguous(), r.is_contiguous()))
        x.add_(2)
        if kw["store_residual"]:
            r.add_(3)

    monkeypatch.setattr(sys.modules[_LOCAL_RMS_NORM], "add_rms_norm_", launch)
    operation = op.AddRMSNorm(provider="triton")
    weight = torch.ones(x.shape[-1])
    operation(x, residual, weight, 1e-6, store_residual=store)
    assert operation.provider == "triton" and operation.provider.reason == "pack"
    assert torch.equal(x, saved_x + 2)
    assert torch.equal(residual, saved_r + 3 if store else saved_r)
    assert seen == [(True, True)]
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError, match="pack"):
        operation(x, residual, weight, 1e-6, store_residual=store)
    assert len(seen) == 1


@pytest.mark.parametrize("name", ["rms_norm", "rms_norm_", "residual_rms_norm"])
def test_standalone_pack_and_checks(monkeypatch, name):

    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setenv(op.VARIABLE, "triton")
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    x = torch.ones(3, 16)[:, ::2]
    kwargs = {"out_dtype": x.dtype} if name == "residual_rms_norm" else {}
    getattr(op, name)(x, torch.ones(8), 1e-6, **kwargs)
    assert calls[0][1][0].is_contiguous()
    assert op.last_standalone_provider[name].reason == "pack"
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError, match="pack"):
        getattr(op, name)(x, torch.ones(8), 1e-6, **kwargs)
    assert len(calls) == 1


def test_standalone_prefix_can_collapse_inner_groups(monkeypatch):
    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    monkeypatch.setenv(op.VARIABLE, "triton")
    x = torch.ones(2, 13, 8)[:, :12].view(2, 3, 4, 8)
    assert op._two_strides(x)
    op.rms_norm(x, torch.ones(8), 1e-6)
    assert calls[0][1][0] is x
    assert not getattr(op.last_standalone_provider["rms_norm"], "reason", None)


@pytest.mark.parametrize("layout", ["row_stride", "alignment"])
def test_sum_out_alignment_is_only_a_fast_path_gate(monkeypatch, layout):
    calls = []
    op = _load_inference_norm(monkeypatch, calls)
    x = torch.ones(3, 16)
    summed = (
        torch.empty(3, 17)[:, :16] if layout == "row_stride" else torch.empty(49)[1:].view(3, 16)
    )
    operation = op.AddRMSNorm(provider="triton")
    operation(x, x.clone(), torch.ones(16), 1e-6, sum_out=summed)
    assert operation.provider == "triton"
    assert dict(calls[-1][1][4:])["sum_out"].data_ptr() == summed.data_ptr()


# selection truth table (dispatch tables / provider resolution)


@pytest.fixture
def qualified_flashinfer(monkeypatch):
    monkeypatch.setattr(norm._flashinfer_version, "installed_version", lambda: "0.7.0.post1")
    monkeypatch.setattr(norm._flashinfer_version, "importable", lambda: True)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("kind", tuple(TABLE))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("static", (False, True))
def test_policy_bands_cover_every_m(kind, dtype, zero, static):
    for m in (0, 1, 2, 3, 4, 5, 16, 17, 64, 65, 255, 256, 300, 512, 1024, 1025, 2048, 1 << 20):
        entry = entry_for(kind, 4096, dtype, m, zero, static)
        if entry.provider == "cute":
            assert entry.config.static_weight == static
        if m < 256:
            # Decode/DFlash rows: CuTe with an early PDL trigger for both weight kinds.
            assert entry.provider == "cute" and entry.config.trigger == "early"
        elif not zero:
            assert entry.provider == "flashinfer"
        else:
            assert entry.provider in ("cute", "flashinfer", "triton")
    for key in (
        (kind, 136, dtype, 3, zero),
        (None, 4096, dtype, 1, zero),
        (kind, 4096, "float32", 1, zero),
    ):
        assert entry_for(*key) is None


def tensor(pointer=4096, **kwargs):
    attrs = {
        "device": torch.device("cuda"),
        "is_cuda": True,
        "dtype": torch.bfloat16,
        "shape": (3, 16),
        "ndim": 2,
        "is_contiguous": lambda: True,
        "data_ptr": lambda: pointer,
        "element_size": lambda: 2,
    }
    attrs.update(kwargs)
    attrs["numel"] = lambda: math.prod(attrs["shape"])
    return SimpleNamespace(**attrs)


@pytest.mark.usefixtures("qualified_flashinfer")
def test_admission_covers_each_operand_and_mode():
    x, residual, weight = tensor(), tensor(8192), tensor(12288, shape=(16,), ndim=1)
    assert norm._admitted(x, residual, weight, False, True, "fp32_sum")
    for index in range(3):
        for change in (
            {"device": torch.device("cpu")},
            {"is_contiguous": lambda: False},
            {"data_ptr": lambda: 3},
        ):
            operands = [x, residual, weight]
            operands[index] = SimpleNamespace(**(vars(operands[index]) | change))
            assert not norm._admitted(*operands, False, True, "fp32_sum")
    assert not norm._admitted(x, x, weight, False, True, "fp32_sum")
    assert not norm._admitted(x, tensor(dtype=torch.float32), weight, True, True, "fp32_sum")
    assert norm._admitted(x, residual, weight, True, False, "fp32_sum")
    assert not norm._admitted(x, residual, weight, True, True, "stored_sum")
    fp32_weight = tensor(12288, shape=(16,), ndim=1, dtype=torch.float32, element_size=lambda: 4)
    assert norm._admitted(x, residual, fp32_weight, True, True, "fp32_sum")
    assert norm._admitted(x, residual, fp32_weight, False, True, "fp32_sum")
    assert not norm._admitted(tensor(shape=(3, 15)), residual, weight, True, True, "fp32_sum")
    assert norm._admitted(
        tensor(shape=(0, 16)), tensor(shape=(0, 16)), weight, True, True, "fp32_sum"
    )


@pytest.mark.usefixtures("qualified_flashinfer")
def test_overrides_and_empty_no_launch(monkeypatch):
    calls = []
    monkeypatch.setattr(norm.flashinfer, "add_rms_norm_", lambda *a, **k: calls.append(a))
    x, residual, weight = (
        torch.ones(1, 16, dtype=torch.bfloat16),
        torch.ones(1, 16, dtype=torch.bfloat16),
        torch.ones(16, dtype=torch.bfloat16),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **k: None),
    )
    op = norm.AddRMSNorm()
    op(x, residual, weight, 1e-6)
    assert op.provider == "triton" and len(calls) == 0
    monkeypatch.setenv(norm.VARIABLE, "cute")
    with pytest.raises(RuntimeError, match="forced cute"):
        op(x, residual, weight, 1e-6)
    norm.AddRMSNorm(provider="flashinfer")(x, residual, weight, 1e-6)
    monkeypatch.setenv(norm.VARIABLE, "bad")
    with pytest.raises(ValueError, match=norm.VARIABLE):
        op(x, residual, weight, 1e-6)

    legacy = ModuleType("ctikernels.norm._zero_centered_rms_norm")
    legacy.add_rms_norm_ = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    triton = norm.AddRMSNorm(provider="triton")
    triton(x, residual, weight, 1e-6)
    assert triton.provider == "triton"
    norm.AddRMSNorm(provider="flashinfer")(x[:0], residual[:0], weight, 1e-6)
    assert len(calls) == 1


@pytest.mark.usefixtures("qualified_flashinfer")
def test_table_selected_cute_and_compile_fallback(monkeypatch):
    x, residual = torch.ones(3, 16, dtype=torch.bfloat16), torch.ones(3, 16, dtype=torch.bfloat16)
    weight = torch.ones(16, dtype=torch.bfloat16)
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    monkeypatch.setattr(norm, "entry_for", lambda *a, **k: Entry("cute", Config(256, 2)))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **k: calls.append("fallback")),
    )
    op = norm.AddRMSNorm(provider="auto")
    monkeypatch.setattr(op, "_cute", lambda *a: calls.append(a[-3]) or (lambda *args: None))
    op(x, residual, weight, 1e-6)
    assert op.provider == "cute" and calls == [Config(256, 2)]
    op._plans.clear()  # Simulate a cold handle with an unavailable compiler.
    monkeypatch.setattr(op, "_cute", lambda *a: None)
    op(x, residual, weight, 1e-6)
    assert op.provider == "triton" and calls[-1] == "fallback"


@pytest.mark.usefixtures("qualified_flashinfer")
def test_cache_failure_and_capture_boundary(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls, compiled = [], object()
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a: calls.append(a) or compiled)
    args = (0, "bfloat16", "float32", 4096, True, Config(), (12, 0))
    assert _provider.get_compiled(*args, capturing=True) is None
    with pytest.raises(RuntimeError, match="warm"):
        _provider.get_compiled(*args, capturing=True, strict=True)
    assert _provider.get_compiled(*args) is compiled
    assert _provider.get_compiled(*args, capturing=True) is compiled
    assert len(calls) == 1
    _provider._COMPILED.clear()

    def fail(*a):
        calls.append(a)
        raise ImportError("missing compiler")

    monkeypatch.setattr(_provider, "compile_kernel", fail)
    assert _provider.get_compiled(*args) is None
    assert _provider.get_compiled(*args) is None
    with pytest.raises(RuntimeError, match="unavailable"):
        _provider.get_compiled(*args, strict=True)
    assert len(calls) == 2


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("zero", (False, True))
def test_reference_uses_unrounded_sum_and_vllm_weight_rounding(dtype, zero):
    x = torch.tensor([[1, 0.125, -2, 3]], dtype=dtype)
    residual = torch.full_like(x, torch.finfo(dtype).eps / 2)
    weight = torch.tensor([0.37, -0.62, 1.125, 2.75], dtype=dtype)
    exact, stored = norm_reference(x, residual, weight, zero_centered=zero)
    summed = (x.float() + residual.float()).double()
    scale = (1 + weight.float() if zero else weight.float()).double()
    torch.testing.assert_close(
        exact, summed / (summed.square().mean(-1, True) + 1e-6).sqrt() * scale
    )
    assert torch.equal(stored, summed.to(dtype))
    assert not torch.equal(stored.double(), summed)
    actual, actual_residual = vllm_norm_native(x, residual, weight, zero_centered=zero)
    assert torch.equal(actual_residual, stored)
    normed = summed.float() * torch.rsqrt(summed.float().square().mean(-1, True) + 1e-6)
    wanted = (normed * (1 + weight.float())).to(dtype) if zero else normed.to(dtype) * weight
    assert torch.equal(actual, wanted)


@pytest.mark.usefixtures("qualified_flashinfer")
def test_config_sweep():
    assert len(set(tune_configs())) == 72  # 12 layouts x 2 weight modes x (no PDL, end, early).
    assert len({c.name for c in tune_configs()}) == 72
    assert parse_config("512,4,1,1") == Config(512, 4, True, True)
    assert parse_config("512,4,1") == Config(512, 4, True)
    assert {c.threads for c in tune_configs()} == {128, 256, 512, 1024}
    assert parse_config("1024,1,1") == Config(1024, 1, True)
    for setting in ("64,1,0", "128,4,0,0,1", "128,4,1,0,2", "256,3,0", "256,1,2", "512,1,1,2"):
        with pytest.raises(ValueError):
            parse_config(setting)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("store,has_sum", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("zero", (False, True))
def test_sum_modes_bypass_flashinfer_bands(monkeypatch, store, has_sum, zero):
    x, residual, weight = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    summed = torch.empty(3, 48)[:, 16:32] if has_sum else None
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    monkeypatch.setattr(norm, "entry_for", lambda *a, **k: Entry("flashinfer"))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    op = norm.AddRMSNorm(provider="auto")
    calls = []
    monkeypatch.setattr(op, "_cute", lambda *a: calls.append(a) or (lambda *args: None))
    op(x, residual, weight, 1e-6, zero_centered=zero, store_residual=store, sum_out=summed)
    assert op.provider == "cute"
    assert calls[0][4] == store
    assert (calls[0][5] is not None) == has_sum
    # A missing compiler must never silently lose a requested store/no-store mode.
    op._plans.clear()  # Simulate a cold handle with an unavailable compiler.
    monkeypatch.setattr(op, "_cute", lambda *a: None)

    legacy = ModuleType("ctikernels.norm._zero_centered_rms_norm")
    launches = []
    legacy.add_rms_norm_ = lambda *a, **k: launches.append(k)
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    op(x, residual, weight, 1e-6, zero_centered=zero, store_residual=store, sum_out=summed)
    assert op.provider == "triton"
    assert launches[0]["zero_centered"] == zero
    assert launches[0]["store_residual"] == store


@pytest.mark.usefixtures("qualified_flashinfer")
def test_sum_store_flags_have_distinct_compilation_cache_entries(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls = []
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a: calls.append(a) or object())
    args = (0, "bfloat16", "bfloat16", 4096, False, Config(), (12, 0))
    results = []
    for store in (False, True):
        for has_sum in (False, True):
            result = _provider.get_compiled(*args, store_residual=store, has_sum=has_sum)
            results.append(result)
            assert (
                _provider.get_compiled(*args, store_residual=store, has_sum=has_sum, capturing=True)
                is result
            )
    assert len(set(results)) == len(calls) == 4


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("threads,rows", ((128, 2), (128, 4), (256, 4)))
def test_small_row_trigger_config(threads, rows):
    early = Config(threads, rows, True, False, "early")
    assert early.name.endswith("_te")
    assert Config(threads, rows, True).name.endswith("_tn")
    assert parse_config(f"{threads},{rows},1,0,1") == early
    assert early in tune_configs(4096)
    assert early.admits(4096)
    assert early.admits(8192) == (threads // rows >= 64)
    assert (early in tune_configs(8192)) == (threads // rows >= 64)
    assert not early.admits(threads // rows * 8 * 16 + 8)
    assert early not in tune_configs(threads // rows * 8 * 16 + 8)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("kwargs", ({"trigger": "early"}, {"trigger": "bad", "pdl": True}))
def test_norm_rejects_invalid_trigger(kwargs):
    with pytest.raises(ValueError, match="trigger"):
        Config(**kwargs)


@pytest.mark.usefixtures("qualified_flashinfer")
def test_norm_chunk_admission_fails_before_compilation(monkeypatch):

    monkeypatch.setattr(cute, "available", lambda: pytest.fail("must check staging bound first"))
    x, weight = torch.empty(1, 4104), torch.empty(4104)
    args = (x, x, weight, False, True, None, Config(128, 4), (12, 1))
    assert norm.AddRMSNorm._cute(*args, strict=False) is None
    with pytest.raises(RuntimeError, match="16 chunks"):
        norm.AddRMSNorm._cute(*args, strict=True)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("kind", tuple(TABLE))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
@pytest.mark.parametrize("hidden", (2048, 4096))
@pytest.mark.parametrize("zero", (False, True))
@pytest.mark.parametrize("static", (False, True))
def test_fp32_residual_measured_table(kind, dtype, hidden, zero, static):

    bands = sorted(_FP32_BANDS[kind, hidden, static])
    # Contiguous inclusive bands from M=0 to the open end, one entry per M.
    assert bands[0][0] == 0 and bands[-1][1] == _LARGE
    assert all(prev[1] + 1 == cur[0] for prev, cur in zip(bands, bands[1:], strict=False))
    for m in (0, 1, 2, 3, 8, 9, 40, 41, 255, 256, 511, 512, 1024, 1 << 20):
        entry = entry_for(kind, hidden, dtype, m, zero, static, residual_dtype="float32")
        assert entry.provider in ("cute", "triton")
        if entry.provider == "cute":
            assert entry.config.static_weight == static
            assert entry.config.admits(hidden)
        # Both weight kinds and both x dtypes share the measured bands.
        assert entry == entry_for(kind, hidden, "bfloat16", m, False, static, "float32")
        assert entry_for(kind, hidden, dtype, m, zero, static) == entry_for(
            kind, hidden, dtype, m, zero, static, residual_dtype=dtype
        )
        if hidden == 2048:
            assert entry_for(kind, hidden, dtype, m, zero, static) is None
    assert entry_for(kind, 4104, dtype, 3, zero, static, residual_dtype="float32") is None


@pytest.mark.usefixtures("qualified_flashinfer")
def test_fp32_residual_table_spot_checks():

    def get(kind, hidden, m, static=False):
        return entry_for(kind, hidden, "bfloat16", m, False, static, residual_dtype="float32")

    assert get("rtx5090", 4096, 1) == Entry("cute", Config(512, 1, True, False, "early"))
    assert get("rtx5090", 4096, 16) == Entry("triton")
    assert get("gb10", 2048, 1024) == Entry("triton")
    assert get("gb10", 4096, 300) == Entry("triton")
    assert get("pro6000", 4096, 4, True) == Entry("cute", Config(512, 1, True, True, "early"))


@pytest.mark.usefixtures("qualified_flashinfer")
def test_fp32_admission_uses_byte_spans():
    x = tensor(4096)
    weight = tensor(16384, shape=(16,), ndim=1)
    residual = tensor(8192, dtype=torch.float32, element_size=lambda: 4)
    for zero in (False, True):
        assert norm._admitted(x, residual, weight, zero, True, "fp32_sum")
        # x starts after 48 two-byte elements but inside 48 four-byte elements.
        assert not norm._admitted(tensor(8192 + 128), residual, weight, zero, True, "fp32_sum")
        assert norm._admitted(tensor(8192 + 192), residual, weight, zero, True, "fp32_sum")


@pytest.mark.usefixtures("qualified_flashinfer")
def test_compilation_cache_separates_residual_dtype(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls = []
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a: calls.append(a) or object())
    args = (0, "bfloat16", "bfloat16", 2048, False, Config(), (12, 0))
    narrow = _provider.get_compiled(*args)
    wide = _provider.get_compiled(*args, residual_dtype="float32")
    assert wide is not narrow and len(calls) == 2
    assert _provider.get_compiled(*args, residual_dtype="bfloat16") is narrow
    assert _provider.get_compiled(*args, residual_dtype="float32", capturing=True) is wide


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("m,provider", [(3, "cute"), (64, "cute"), (1024, "triton")])
@pytest.mark.parametrize("zero", [False, True])
def test_auto_resolves_fp32_residual_table(monkeypatch, m, provider, zero):

    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    legacy = ModuleType("ctikernels.norm._zero_centered_rms_norm")
    launches = []
    legacy.add_rms_norm_ = lambda *a, **k: launches.append(k)
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    op = norm.AddRMSNorm()
    monkeypatch.setattr(op, "_cute", lambda *a: lambda *args: None)
    x = torch.ones(m, 2048, dtype=torch.bfloat16)
    residual, weight = torch.ones(m, 2048), torch.ones(2048, dtype=x.dtype)
    op(x, residual, weight, 1e-6, zero_centered=zero)
    assert op.provider == provider
    if provider == "cute":
        assert (
            op.dispatched_config
            == entry_for("gb10", 2048, "bfloat16", m, zero, residual_dtype="float32").config
        )
    monkeypatch.setenv(norm.VARIABLE, "triton")
    op(x, residual, weight, 1e-6, zero_centered=zero)
    assert op.provider == "triton" and launches[-1]["zero_centered"] == zero


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("split", range(2, 9))
@pytest.mark.parametrize("threads", (64, 128))
def test_split_config_contract(split, threads):
    config = Config(threads, split=split)
    assert config.name == f"cute_t{threads}_r1_p0_s0_tn_c{split}"
    assert parse_config(f"{threads},1,0,0,0,{split}") == config
    assert config.admits(threads * split * 8 * 16)
    assert not config.admits(threads * split * 8 * 16 + 8)
    assert not config.admits((split - 1) * 8)
    assert config.admits(split * 8)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"split": 0},
        {"split": 9},
        {"split": 2.5},
        {"split": True},
        {"split": 2, "rows": 2},
        {"threads": 64},
    ],
)
def test_split_config_invalid(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.usefixtures("qualified_flashinfer")
def test_compile_cache_separates_split(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a: object())
    args = (0, "bfloat16", "bfloat16", 2048, False)
    plain = _provider.get_compiled(*args, Config(), (12, 1))
    split = _provider.get_compiled(*args, Config(split=3), (12, 1))
    assert split is not plain
    assert _provider.get_compiled(*args, Config(split=3), (12, 1), capturing=True) is split


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("cute_available", [False, True])
@pytest.mark.parametrize("pin", [False, True])
def test_unusable_fi_reroute_survives_plan_cache_and_checks(monkeypatch, zero, cute_available, pin):

    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(norm._flashinfer_version, "allowed", lambda: False)
    monkeypatch.setattr(norm._flashinfer_version, "unusable_reason", lambda: "not importable")
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    monkeypatch.setattr(norm, "entry_for", lambda *a, **k: Entry("flashinfer"))
    launches = []
    launch = lambda *a, **k: launches.append("launch")
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=launch),
    )
    op = norm.AddRMSNorm()
    monkeypatch.setattr(op, "_cute", lambda *a: launch if cute_available else None)
    x, r, w = (
        torch.ones(2, 16, dtype=torch.bfloat16),
        torch.ones(2, 16, dtype=torch.bfloat16),
        torch.ones(16, dtype=torch.bfloat16),
    )
    if pin:
        op._plan(x, r, w, zero, True, "fp32_sum", None, None, pin=True)
    for _ in range(2):
        op(x, r, w, 1e-6, zero_centered=zero)
        assert isinstance(op.provider, Provider) and "not importable" in op.provider.reason
        assert op.provider == ("cute" if cute_available else "triton")
    assert len(launches) == 2
    size = len(op._plans), len(op._pinned)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError):
        op(x, r, w, 1e-6, zero_centered=zero)
    assert len(launches) == 2 and (len(op._plans), len(op._pinned)) == size


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("zero", [False, True])
def test_unknown_device_generic_path_has_no_reroute(monkeypatch, zero):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(norm._flashinfer_version, "allowed", lambda: False)
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: (None, (12, 0)))
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **k: None),
    )
    x = torch.ones(2, 16, dtype=torch.bfloat16)
    op = norm.AddRMSNorm()
    monkeypatch.setattr(op, "_cute", lambda *a: lambda *a: None)
    op(x, x.clone(), torch.ones(16, dtype=x.dtype), 1e-6, zero_centered=zero)
    assert op.provider == "cute" and getattr(op.provider, "reason", None) is None


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (9, 0), (8, 9)])
@pytest.mark.parametrize("dsl", [False, True])
@pytest.mark.parametrize("hidden", [4096, 5120, 1000, 1001])
@pytest.mark.parametrize("layout", ["bf16", "fp32", "stored_sum", "strided", "unaligned_sum"])
def test_add_norm_ordered_predicate(monkeypatch, capability, dsl, hidden, layout):

    monkeypatch.delenv(norm.VARIABLE, raising=False)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", capability))
    monkeypatch.setattr(cute, "available", lambda: dsl)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launches = []
    monkeypatch.setattr(cute, "get_compiled", lambda *a, **kw: lambda *a: launches.append("cute"))
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **kw: launches.append("triton")),
    )
    dtype = torch.float32 if layout == "fp32" else torch.bfloat16
    x = torch.ones(3, hidden + (1 if layout == "strided" else 0), dtype=dtype)[:, :hidden]
    residual, weight = torch.ones_like(x), torch.ones(hidden, dtype=dtype)
    summed = (
        torch.empty(3 * hidden + 1, dtype=dtype)[1:].view(3, hidden)
        if layout == "unaligned_sum"
        else None
    )
    normalize = "stored_sum" if layout == "stored_sum" else "fp32_sum"
    admitted = layout == "bf16" and hidden % 8 == 0 and capability[0] == 12
    expected = "cute" if admitted and dsl else "triton"
    op = norm.AddRMSNorm()
    for _ in range(2):
        op(x, residual, weight, 1e-6, normalize=normalize, sum_out=summed)
        assert op.provider == expected
        assert bool(getattr(op.provider, "reason", None)) == (admitted and not dsl)
    assert launches == [expected, expected]
    if expected == "cute":
        entry = entry_for("gb10", hidden, "bfloat16", 3, False)
        assert op.dispatched_config == (entry.config if entry else Config())
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    if admitted and not dsl:
        with pytest.raises(RerouteError, match="nvidia-cutlass-dsl"):
            op(x, residual, weight, 1e-6, normalize=normalize, sum_out=summed)
    else:
        op(x, residual, weight, 1e-6, normalize=normalize, sum_out=summed)


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (9, 0), (8, 9)])
@pytest.mark.parametrize("override", ["constructor", "environment"])
def test_add_norm_overrides_precede_auto(monkeypatch, capability, override):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(norm, "_hardware", lambda *a: (None, capability))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(norm.AddRMSNorm, "_cute", staticmethod(lambda *a: lambda *a: None))
    calls = []
    monkeypatch.setattr(
        norm.flashinfer, "add_rms_norm_", lambda *a, **kw: calls.append("flashinfer")
    )
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **kw: calls.append("triton")),
    )
    x = torch.ones(3, 1000, dtype=torch.bfloat16)
    r, w = x.clone(), torch.ones(1000, dtype=x.dtype)
    for provider in ("triton", "flashinfer", "cute"):
        monkeypatch.setenv(norm.VARIABLE, provider if override == "environment" else "auto")
        op = norm.AddRMSNorm(provider=provider if override == "constructor" else None)
        if provider == "cute" and capability[0] != 12:
            with pytest.raises(
                RuntimeError, match="CuTe norm is unqualified outside capability 12.x"
            ):
                op(x, r, w, 1e-6)
            with pytest.raises(RuntimeError, match="unqualified outside capability 12.x"):
                op(x.float(), r.float(), w.float(), 1e-6, normalize="stored_sum")
        else:
            op(x, r, w, 1e-6)
            assert op.provider == provider
            if provider != "cute":
                assert calls[-1] == provider


@pytest.mark.usefixtures("qualified_flashinfer")
@pytest.mark.parametrize("hidden,threads", [(5120, 512), (65544, 1024), (131080, None)])
@pytest.mark.parametrize("dsl", [False, True])
def test_off_table_generic_config_size(monkeypatch, hidden, threads, dsl):
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: (None, (12, 0)))
    monkeypatch.setattr(cute, "available", lambda: dsl)
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **kw: None),
    )
    op = norm.AddRMSNorm()
    monkeypatch.setattr(op, "_cute", lambda *a: lambda *a: None)
    x = torch.ones(1, hidden, dtype=torch.bfloat16)
    op(x, x.clone(), torch.ones(hidden, dtype=x.dtype), 1e-6)
    assert op.provider == ("cute" if threads and dsl else "triton")
    assert bool(getattr(op.provider, "reason", None)) == (bool(threads) and not dsl)
    if threads and dsl:
        assert op.dispatched_config == Config(threads=threads)


@pytest.mark.usefixtures("qualified_flashinfer")
def test_bad_table_config_is_visible(monkeypatch):

    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: (None, (12, 0)))
    monkeypatch.setattr(norm, "entry_for", lambda *a, **kw: Entry("cute", Config(128, 4)))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda *a, **kw: None),
    )
    x = torch.ones(1, 5120, dtype=torch.bfloat16)
    op = norm.AddRMSNorm()
    op(x, x.clone(), torch.ones(5120, dtype=x.dtype), 1e-6)
    assert op.provider == "triton" and "does not admit" in op.provider.reason
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError, match="does not admit"):
        op(x, x.clone(), torch.ones(5120, dtype=x.dtype), 1e-6)


# residual norm


_CONSTEXPRS = ("hidden_size", "eps", "BLOCK_SIZE", "HAS_SUM", "ZERO_CENTERED")


_RTX_PRO_6000 = ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0))


_UNLISTED = ("CPU stand-in", (0, 0))


EPS = 1e-6


def _load_local_kernels(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the private kernels against a Triton stand-in; CPU CI installs no Triton."""

    triton = ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda function=None, **_: function or (lambda decorated: decorated)
    triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", ModuleType("triton.language"))
    # Record the current entry so the module built on the stand-in is dropped afterwards.
    monkeypatch.setitem(sys.modules, _LOCAL_RMS_NORM, None)
    del sys.modules[_LOCAL_RMS_NORM]
    # Importing a submodule also binds it on its package, which ``from . import`` reads first:
    # restore that binding too, or later tests run the stand-in's kernels.
    package, _, name = _LOCAL_RMS_NORM.rpartition(".")
    package = importlib.import_module(package)
    monkeypatch.setattr(package, name, getattr(package, name, None), raising=False)
    return importlib.import_module(_LOCAL_RMS_NORM)


def _norm(values: torch.Tensor, weight: torch.Tensor, zero_centered=True) -> torch.Tensor:
    return (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS)
        * (1 + weight.float() if zero_centered else weight.float())
    )


class _Launch(NamedTuple):
    grid: tuple[int, ...]
    args: dict[str, object]
    options: dict[str, object]


class _Kernel:
    """Stands in for a residual-stream kernel launched as ``kernel[grid](*args, **options)``:
    records each launch's arguments under the kernel's parameter names and runs ``formula``
    on them."""

    def __init__(self, kernel, formula) -> None:
        self.signature = inspect.signature(kernel)
        self.formula = formula
        self.launches: list[_Launch] = []

    def __getitem__(self, grid):
        def launch(*args, **options):
            arguments = self.signature.bind(*args).arguments
            self.launches.append(_Launch(grid, arguments, options))
            self.formula(**arguments)

        return launch


def _flags(launch: _Launch) -> dict[str, object]:
    """The launch's runtime mode flags: its arguments that are neither tensors nor constexprs."""

    return {
        name: value
        for name, value in launch.args.items()
        if name not in (*_CONSTEXPRS, "sum_stride", "x_stride", "residual_stride")
        and not isinstance(value, torch.Tensor)
    }


def _forward(
    x,
    residual,
    residual_out,
    weight,
    output,
    has_x,
    store_residual,
    normalize_stored,
    sum_out,
    HAS_SUM,
    ZERO_CENTERED,
    **_,
):
    values = residual.float()
    if has_x:
        values = values + x.float()
        stored = values.to(residual_out.dtype)
        if store_residual:
            residual_out.copy_(stored)
        if HAS_SUM:
            sum_out.copy_(stored)
        if normalize_stored:
            values = stored.float()
    output.copy_(_norm(values, weight, ZERO_CENTERED))


def _backward(
    x,
    residual,
    weight,
    grad_output,
    grad_residual_out,
    grad_x,
    grad_residual,
    saved_inputs,
    has_grad_residual_out,
    has_grad_x,
    ZERO_CENTERED,
    **_,
):
    values = residual.float()
    if saved_inputs:
        values = (values + x.float()).to(grad_residual.dtype).float()
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS)
    normalized = values * scale
    scaled_upstream = grad_output.float() * (
        1 + weight.float() if ZERO_CENTERED else weight.float()
    )
    grad = scale * (scaled_upstream - normalized * (scaled_upstream * normalized).mean(-1, True))
    if has_grad_residual_out:
        grad = grad + grad_residual_out.float()
    grad_residual.copy_(grad)
    if has_grad_x:
        grad_x.copy_(grad)


@pytest.fixture
def kernels(monkeypatch) -> ModuleType:
    """The private module with torch stand-ins for its residual-stream kernels, on CPU tensors
    that pass its CUDA check, on a device no launch policy qualifies."""

    module = _load_local_kernels(monkeypatch)
    for name, formula in (
        ("_residual_norm_kernel", _forward),
        ("_residual_norm_backward_kernel", _backward),
    ):
        monkeypatch.setattr(module, name, _Kernel(getattr(module, name), formula))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    monkeypatch.setattr(module, "_hardware", lambda device_index: _UNLISTED)
    return module


def _signature(launch: _Launch) -> tuple:
    """What Triton keys a binary on, given contiguous aligned operands: each runtime argument's
    dtype (bool and int scalars differ), the constexprs' values and the launch options."""

    keys = []
    for name, value in launch.args.items():
        if name in _CONSTEXPRS:
            keys.append(value)
        elif isinstance(value, torch.Tensor):
            keys.append(value.dtype)
        else:
            keys.append(type(value))
    return tuple(keys), tuple(sorted(launch.options.items()))


@pytest.mark.parametrize("residual_dtype", [torch.bfloat16, torch.float32])
def test_every_entry_point_and_mode_launches_one_signature(kernels, residual_dtype) -> None:
    hidden_size = 64
    weight = torch.randn(hidden_size)

    def operand(dtype, rows, transposed=False):
        if transposed:  # [1, hidden] with row stride 1
            return torch.randn(hidden_size, 1, dtype=dtype).t()
        return torch.randn(rows, hidden_size, dtype=dtype)

    for rows, transposed in ((1, False), (1, True), (7, False)):
        x, residual = operand(torch.bfloat16, rows, transposed), operand(residual_dtype, rows)
        with torch.inference_mode():
            for store_residual in (True, False):
                for normalize_stored in (False, True):
                    kernels.add_rms_norm_(
                        x,
                        residual,
                        weight,
                        EPS,
                        store_residual=store_residual,
                        normalize_stored=normalize_stored,
                    )
            kernels.residual_rms_norm(residual, weight, EPS, out_dtype=torch.bfloat16)
        for save_inputs in (False, True):
            for gradients in ("both", "normalized"):
                x = operand(torch.bfloat16, rows, transposed).requires_grad_()
                residual = operand(residual_dtype, rows).requires_grad_()
                outputs = kernels.training_add_rms_norm(
                    x, residual, weight, EPS, save_inputs=save_inputs
                )
                outputs = outputs if gradients == "both" else outputs[:1]
                torch.autograd.backward(outputs, [torch.randn_like(t) for t in outputs])
        for gradients in ("both", "normalized"):
            residual = operand(residual_dtype, rows).requires_grad_()
            outputs = kernels.training_residual_rms_norm(
                residual, weight, EPS, out_dtype=torch.bfloat16
            )
            outputs = outputs if gradients == "both" else outputs[:1]
            torch.autograd.backward(outputs, [torch.randn_like(t) for t in outputs])

    for kernel, launches in (
        (kernels._residual_norm_kernel, 33),
        (kernels._residual_norm_backward_kernel, 18),
    ):
        assert len(kernel.launches) == launches
        assert len({_signature(launch) for launch in kernel.launches}) == 1
        assert {launch.grid for launch in kernel.launches} == {(1,), (7,)}
        # Triton types a bool argument i1 and an int i32, so a bool flag would split binaries.
        assert {type(flag) for flag in _flags(kernel.launches[0]).values()} == {int}
        assert kernel.launches[0].options == {"num_warps": 4, "enable_fp_fusion": False}


def _unaligned(shape, dtype) -> torch.Tensor:
    return torch.zeros(shape[0] * shape[1] + 1, dtype=dtype)[1:].view(shape)


def test_residual_norms_check_their_operands_before_any_launch(kernels) -> None:
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    residual = torch.randn(4, 64)
    weight = torch.randn(64)
    strided = torch.randn(4, 128)[:, :64]
    # Inference takes row-strided operands; a unit hidden stride is still required.
    transposed = torch.randn(64, 4).t()
    unaligned_x = _unaligned((4, 64), torch.bfloat16)
    unaligned_weight = _unaligned((1, 64), torch.float32)[0]
    cases = {
        "strided residual": (x, transposed, weight, ValueError, "contiguous"),
        "unaligned x": (unaligned_x, residual, weight, ValueError, "aligned"),
        "unaligned weight": (x, residual, unaligned_weight, ValueError, "aligned"),
        "narrow residual": (x, residual.half(), weight, TypeError, "input dtype or FP32"),
        "shape": (x, residual[:, :32], weight, ValueError, "shape and device"),
    }
    for name, (case_x, case_residual, case_weight, error, message) in cases.items():
        if "unaligned" not in name:
            with pytest.raises(error, match=message), torch.inference_mode():
                kernels.add_rms_norm_(
                    case_x,
                    case_residual,
                    case_weight,
                    EPS,
                    store_residual=True,
                    normalize_stored=False,
                )
        with pytest.raises(error, match=message):
            kernels.training_add_rms_norm(
                case_x, case_residual, case_weight, EPS, save_inputs=False
            )
    with pytest.raises(ValueError, match="distinct"), torch.inference_mode():
        kernels.add_rms_norm_(x, x, weight, EPS, store_residual=True, normalize_stored=False)
    residual_cases = {
        "strided": (strided, torch.bfloat16, ValueError, "contiguous"),
        "narrow residual to FP32": (x, torch.float32, TypeError, "out_dtype or FP32"),
        "FP16 residual to BF16": (x.half(), torch.bfloat16, TypeError, "out_dtype or FP32"),
    }
    for case_residual, out_dtype, error, message in residual_cases.values():
        for operation in (kernels.residual_rms_norm, kernels.training_residual_rms_norm):
            if case_residual is strided and operation == kernels.residual_rms_norm:
                continue
            with pytest.raises(error, match=message):
                operation(case_residual, weight, EPS, out_dtype=out_dtype)

    assert kernels._residual_norm_kernel.launches == []


def test_zero_centered_residual_takes_the_input_dtype_or_fp32(monkeypatch) -> None:
    validate = _load_local_kernels(monkeypatch)._validate_residual
    x = torch.randn(4, 2, 128, dtype=torch.bfloat16)
    shared = torch.randn(4, 2, 128)

    validate(x, torch.randn(4, 2, 128, dtype=torch.bfloat16))
    validate(x, shared)
    # Empty tensors report the same null data pointer whatever their dtype.
    validate(torch.empty(0, 2, 128, dtype=torch.bfloat16), torch.empty(0, 2, 128))
    with pytest.raises(TypeError, match="input dtype or FP32"):
        validate(x, x.half())
    with pytest.raises(TypeError, match="input dtype or FP32"):
        validate(shared, x)
    with pytest.raises(ValueError, match="shape and device"):
        validate(x, shared[:, :1])
    with pytest.raises(ValueError, match="overlap"):
        validate(shared.view(torch.bfloat16)[..., :128], shared)


def test_inference_stores_in_place_or_leaves_the_residual(kernels) -> None:
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    residual = torch.randn(3, 64)
    weight = torch.randn(64)
    summed = residual + x.float()
    expected = _norm(summed, weight).bfloat16()
    kept_x, kept_residual = x.clone(), residual.clone()

    with torch.inference_mode():
        kernels.add_rms_norm_(
            kept_x, kept_residual, weight, EPS, store_residual=False, normalize_stored=False
        )
        returned = kernels.add_rms_norm_(
            x, residual, weight, EPS, store_residual=True, normalize_stored=False
        )

    assert returned[0] is x and returned[1] is residual
    assert torch.equal(x, expected) and torch.equal(residual, summed)
    assert torch.equal(kept_x, expected) and not torch.equal(kept_residual, summed)
    kept, stored = kernels._residual_norm_kernel.launches
    assert _flags(kept) == {"has_x": 1, "store_residual": 0, "normalize_stored": 0}
    assert _flags(stored) == {"has_x": 1, "store_residual": 1, "normalize_stored": 0}
    # In place: x is read and overwritten, and the residual is its own output.
    assert stored.args["x"] is x and stored.args["output"] is x
    assert stored.args["residual"] is residual and stored.args["residual_out"] is residual


def test_training_normalizes_the_stored_sum_and_inference_the_one_it_names(kernels) -> None:
    """With a residual in x's dtype, the FP32 sum and the stored one differ by a rounding."""

    x = torch.randn(3, 64, dtype=torch.bfloat16)
    residual = torch.randn(3, 64, dtype=torch.bfloat16)
    weight = torch.randn(64)
    fp32_sum = residual.float() + x.float()
    outputs = {}
    with torch.inference_mode():
        for normalize_stored in (False, True):
            outputs[normalize_stored], _ = kernels.add_rms_norm_(
                x.clone(),
                residual.clone(),
                weight,
                EPS,
                store_residual=True,
                normalize_stored=normalize_stored,
            )
    no_grad, _ = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    leaf = x.clone().requires_grad_()
    with_grad, _ = kernels.training_add_rms_norm(leaf, residual, weight, EPS, save_inputs=False)

    assert torch.equal(outputs[False], _norm(fp32_sum, weight).bfloat16())
    assert torch.equal(outputs[True], _norm(fp32_sum.bfloat16().float(), weight).bfloat16())
    assert torch.equal(no_grad, outputs[True]) and torch.equal(with_grad, outputs[True])
    launches = kernels._residual_norm_kernel.launches
    assert [launch.args["normalize_stored"] for launch in launches] == [0, 1, 1, 1]


def _saved(function, *args, **kwargs) -> tuple[object, list[int]]:
    """``function``'s result and the storage of every tensor autograd keeps meanwhile."""

    pointers: list[int] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        pointers.append(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = function(*args, **kwargs)
    return result, pointers


@pytest.mark.parametrize("residual_dtype", [torch.bfloat16, torch.float32])
def test_backward_keeps_the_sum_or_the_inputs(kernels, residual_dtype) -> None:
    x_data = torch.randn(5, 64, dtype=torch.bfloat16)
    residual_data = torch.randn(5, 64, dtype=residual_dtype)
    weight = torch.randn(64)
    upstream = (torch.randn(5, 64, dtype=torch.bfloat16), torch.randn(5, 64, dtype=residual_dtype))
    gradients = {}
    for save_inputs in (False, True):
        x = x_data.clone().requires_grad_()
        residual = residual_data.clone().requires_grad_()
        (normalized, summed), pointers = _saved(
            kernels.training_add_rms_norm, x, residual, weight, EPS, save_inputs=save_inputs
        )
        kept = (x, residual) if save_inputs else (summed,)
        assert pointers == [tensor.untyped_storage().data_ptr() for tensor in (*kept, weight)]
        torch.autograd.backward((normalized, summed), upstream)
        gradients[save_inputs] = (x.grad, residual.grad)
        launch = kernels._residual_norm_backward_kernel.launches[-1]
        assert launch.args["saved_inputs"] == int(save_inputs)

    for kept_sum, kept_inputs in zip(gradients[False], gradients[True], strict=True):
        assert torch.equal(kept_sum, kept_inputs)


@pytest.mark.parametrize("other_uses", [False, True])
def test_a_residual_in_xs_dtype_gives_both_inputs_one_gradient(kernels, other_uses) -> None:
    """Backward returns one tensor for x and residual. Whether autograd hands it to both leaves
    or first adds other uses' gradients, each leaf's gradient is its own, so a second backward
    accumulates into one without touching the other."""

    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64)
    upstream = [torch.randn(4, 64, dtype=torch.bfloat16) for _ in range(4)]
    if not other_uses:
        upstream[2:] = [torch.zeros_like(upstream[0])] * 2

    normalized, summed = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    (norm_gradient,) = torch.autograd.grad(
        (normalized, summed), (x,), upstream[:2], retain_graph=True
    )
    outputs = (normalized, summed, x, residual) if other_uses else (normalized, summed)
    for _ in range(2):
        torch.autograd.backward(outputs, upstream[: len(outputs)], retain_graph=True)

    for leaf, other in ((x, upstream[2]), (residual, upstream[3])):
        once = norm_gradient + other
        assert torch.equal(leaf.grad, once + once)
    assert x.grad.untyped_storage().data_ptr() != residual.grad.untyped_storage().data_ptr()
    launches = kernels._residual_norm_backward_kernel.launches
    assert {launch.args["has_grad_x"] for launch in launches} == {0}


def test_an_fp32_residual_rounds_xs_gradient_from_the_same_value(kernels) -> None:
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(4, 64, requires_grad=True)
    weight = torch.randn(64)

    normalized, summed = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=True)
    torch.autograd.backward(
        (normalized, summed), (torch.randn_like(normalized), torch.randn_like(summed))
    )

    assert x.grad.dtype == torch.bfloat16 and residual.grad.dtype == torch.float32
    assert torch.equal(x.grad, residual.grad.bfloat16())
    [launch] = kernels._residual_norm_backward_kernel.launches
    assert _flags(launch) == {"saved_inputs": 1, "has_grad_residual_out": 1, "has_grad_x": 1}


def test_unused_outputs_send_backward_no_gradient(kernels) -> None:
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(4, 64, requires_grad=True)
    weight = torch.randn(64)

    normalized, summed = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    normalized.backward(torch.randn_like(normalized))
    [launch] = kernels._residual_norm_backward_kernel.launches
    # The absent gradient of the sum is stood in for by the kept sum, of the residual's dtype.
    assert launch.args["has_grad_residual_out"] == 0
    assert launch.args["grad_residual_out"] is launch.args["residual"]

    upstream = torch.randn_like(residual)
    x.grad = residual.grad = None
    normalized, summed = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    summed.backward(upstream)
    # Only the sum is used: both gradients are its gradient, the norm adding nothing.
    assert torch.equal(residual.grad, upstream) and torch.equal(x.grad, upstream.bfloat16())


def test_the_residual_passes_through_and_its_gradient_joins_the_norms(kernels) -> None:
    residual = torch.randn(4, 64, requires_grad=True)
    weight = torch.randn(64)
    upstream = torch.randn(4, 64, dtype=torch.bfloat16)
    residual_upstream = torch.randn(4, 64)

    normalized, passed = kernels.training_residual_rms_norm(
        residual, weight, EPS, out_dtype=torch.bfloat16
    )
    assert passed.data_ptr() == residual.data_ptr() and passed.grad_fn is normalized.grad_fn
    (norm_only,) = torch.autograd.grad(normalized, residual, upstream, retain_graph=True)
    torch.autograd.backward((normalized, passed), (upstream, residual_upstream))

    assert torch.equal(residual.grad, norm_only + residual_upstream)
    assert [_flags(launch) for launch in kernels._residual_norm_backward_kernel.launches] == [
        {"saved_inputs": 0, "has_grad_residual_out": 0, "has_grad_x": 0},
        {"saved_inputs": 0, "has_grad_residual_out": 1, "has_grad_x": 0},
    ]
    with torch.no_grad():
        _, passed = kernels.training_residual_rms_norm(
            residual, weight, EPS, out_dtype=torch.bfloat16
        )
    assert passed is residual


def test_backward_copies_only_a_gradient_it_cannot_read_as_rows(kernels) -> None:
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64)
    upstream = torch.randn(4, 64, dtype=torch.bfloat16)

    normalized, _ = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    normalized.backward(upstream)
    # sum() hands backward an expanded, stride-0 gradient.
    normalized, _ = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    normalized.sum().backward()

    read, copied = kernels._residual_norm_backward_kernel.launches
    assert read.args["grad_output"] is upstream
    assert copied.args["grad_output"].stride() == (64, 1)


def test_empty_rows_launch_nothing(kernels) -> None:
    x = torch.empty(0, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.empty(0, 64, requires_grad=True)
    weight = torch.randn(64)

    normalized, summed = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    (normalized.float().sum() + summed.sum()).backward()
    output, passed = kernels.training_residual_rms_norm(
        residual, weight, EPS, out_dtype=torch.bfloat16
    )
    output.float().sum().backward()

    assert x.grad.shape == x.shape and residual.grad.shape == residual.shape
    assert kernels._residual_norm_kernel.launches == []
    assert kernels._residual_norm_backward_kernel.launches == []


def _num_warps(
    module: ModuleType,
    hidden_size: int,
    backward: bool,
    hardware: tuple[str, tuple[int, int]] = _RTX_PRO_6000,
) -> int:
    device_name, capability = hardware
    return module._num_warps(
        hidden_size, backward=backward, device_name=device_name, capability=capability
    )


@pytest.mark.parametrize(
    ("hidden_size", "forward_warps", "backward_warps"),
    [(2048, 16, 8), (4096, 16, 8), (5120, 16, 8), (2047, 4, 4), (128, 4, 4)],
)
def test_qualified_hardware_takes_the_measured_warps(
    monkeypatch, hidden_size, forward_warps, backward_warps
) -> None:
    module = _load_local_kernels(monkeypatch)

    assert [_num_warps(module, hidden_size, backward) for backward in (False, True)] == [
        forward_warps,
        backward_warps,
    ]


@pytest.mark.parametrize(
    "hardware",
    [
        ("NVIDIA GeForce RTX 5090", (12, 0)),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0)),
        (_RTX_PRO_6000[0], (12, 1)),
        ("NVIDIA H100 80GB HBM3", (9, 0)),
    ],
)
def test_other_hardware_keeps_the_portable_launch(monkeypatch, hardware) -> None:
    """Qualification is the exact device name and capability pair."""

    module = _load_local_kernels(monkeypatch)

    for hidden_size, num_warps in ((2047, 4), (2048, 4), (4095, 4), (4096, 32), (5120, 32)):
        for backward in (False, True):
            assert _num_warps(module, hidden_size, backward, hardware) == num_warps


@pytest.mark.parametrize("hardware", [_RTX_PRO_6000, _UNLISTED])
def test_every_launch_takes_the_warps_of_its_direction(kernels, monkeypatch, hardware) -> None:
    """The grouped kernels launch as the residual-stream kernels do, whatever the row count,
    so the two reduce a row in one layout."""

    monkeypatch.setattr(kernels, "_hardware", lambda device_index: hardware)
    for name in ("_rms_norm_kernel", "_rms_norm_backward_kernel"):
        monkeypatch.setattr(kernels, name, _Kernel(getattr(kernels, name), lambda **_: None))
    hidden_size = 2048
    weight = torch.randn(hidden_size)
    for rows in (1, 300):
        x = torch.randn(rows, hidden_size, dtype=torch.bfloat16, requires_grad=True)
        residual = torch.randn(rows, hidden_size, requires_grad=True)
        outputs = kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
        torch.autograd.backward(outputs, [torch.randn_like(output) for output in outputs])
        heads = torch.randn(1, rows, hidden_size, dtype=torch.bfloat16, requires_grad=True)
        normalized = kernels.training_rms_norm(heads, weight, EPS)
        normalized.backward(torch.randn_like(normalized))

    forward_warps, backward_warps = (16, 8) if hardware == _RTX_PRO_6000 else (4, 4)
    for kernel, num_warps in (
        (kernels._residual_norm_kernel, forward_warps),
        (kernels._rms_norm_kernel, forward_warps),
        (kernels._residual_norm_backward_kernel, backward_warps),
        (kernels._rms_norm_backward_kernel, backward_warps),
    ):
        assert [launch.grid for launch in kernel.launches] == [(1,), (300,)]
        assert [launch.options["num_warps"] for launch in kernel.launches] == [num_warps] * 2


def test_the_device_is_asked_once_per_index(monkeypatch) -> None:
    module = _load_local_kernels(monkeypatch)
    asked = []

    def device_name(index: int) -> str:
        asked.append(index)
        return _RTX_PRO_6000[0]

    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: _RTX_PRO_6000[1])
    for index in (0, 1, 0, 1, 0):
        assert module._hardware(index) == _RTX_PRO_6000

    assert asked == [0, 1]


@pytest.mark.parametrize("store", (False, True))
@pytest.mark.parametrize("residual_dtype", (torch.bfloat16, torch.float32))
def test_sum_output_launcher_preserves_stride_flags_and_launch_options(
    kernels, store, residual_dtype
):
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    residual = torch.randn(4, 64, dtype=residual_dtype)
    weight = torch.randn(64)
    expected = x.float() + residual.float()
    before = residual.clone()
    aux = torch.full((4, 192), 17.0, dtype=residual_dtype)
    summed = aux[:, 64:128]
    kernels.add_rms_norm_(
        x, residual, weight, EPS, store_residual=store, normalize_stored=True, sum_out=summed
    )
    launch = kernels._residual_norm_kernel.launches[0]
    assert launch.args["sum_out"] is summed
    assert launch.args["sum_stride"] == 192 and launch.args["HAS_SUM"] is True
    assert launch.options == {"num_warps": 4, "enable_fp_fusion": False}
    assert torch.equal(summed, expected.to(residual_dtype))
    assert torch.equal(residual, expected.to(residual_dtype) if store else before)
    assert (aux[:, :64] == 17).all() and (aux[:, 128:] == 17).all()


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("save_inputs", [False, True])
def test_weight_mode_forward_and_backward(kernels, zero, save_inputs):
    torch.manual_seed(17)
    x = torch.randn(3, 64, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(3, 64, requires_grad=True)
    weight = torch.randn(64, dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    grad_sum = torch.randn_like(residual)
    result, summed = kernels.training_add_rms_norm(
        x, residual, weight, EPS, save_inputs=save_inputs, zero_centered=zero
    )
    torch.autograd.backward((result, summed), (upstream, grad_sum))
    reference = residual.detach().clone().requires_grad_()
    values = reference + x.detach().float()
    expected = _norm(values, weight, zero)
    torch.autograd.backward((expected, values), (upstream.float(), grad_sum))
    assert torch.equal(result, expected.bfloat16())
    assert torch.equal(summed, values)
    torch.testing.assert_close(residual.grad, reference.grad)
    torch.testing.assert_close(x.grad, reference.grad.bfloat16())
    with torch.no_grad():
        inferred, stored = kernels.add_rms_norm_(
            x.detach().clone(),
            residual.detach().clone(),
            weight,
            EPS,
            store_residual=True,
            normalize_stored=False,
            zero_centered=zero,
        )
        alone = kernels.residual_rms_norm(
            stored, weight, EPS, out_dtype=x.dtype, zero_centered=zero
        )
    assert torch.equal(inferred, result) and torch.equal(alone, result)
    trained, _ = kernels.training_residual_rms_norm(
        residual, weight, EPS, out_dtype=x.dtype, zero_centered=zero
    )
    trained.backward(upstream)
    assert all(
        launch.args["ZERO_CENTERED"] == zero for launch in kernels._residual_norm_kernel.launches
    )
    assert all(
        launch.args["ZERO_CENTERED"] == zero
        for launch in kernels._residual_norm_backward_kernel.launches
    )


def test_inference_residual_accepts_unaligned_rows_and_weight(kernels):
    x = _unaligned((3, 65), torch.bfloat16)[:, :64]
    residual = _unaligned((3, 67), torch.float32)[:, :64]
    weight = _unaligned((1, 64), torch.float32)[0]
    kernels.add_rms_norm_(x, residual, weight, EPS, store_residual=True, normalize_stored=True)
    launch = kernels._residual_norm_kernel.launches[-1]
    assert launch.args["x_stride"] == 65 and launch.args["residual_stride"] == 67
    kernels.residual_rms_norm(residual, weight, EPS, out_dtype=x.dtype)
    assert kernels._residual_norm_kernel.launches[-1].args["residual_stride"] == 67
    with pytest.raises(ValueError, match="aligned"):
        kernels.training_add_rms_norm(x, residual, weight, EPS, save_inputs=False)
    with pytest.raises(ValueError, match="aligned"):
        kernels.training_residual_rms_norm(residual, weight, EPS, out_dtype=x.dtype)


def test_forward_epsilon_is_runtime_and_large_width_is_chunked(kernels):
    # The CPU launch stand-in retains the real kernel signature, including annotations.
    assert kernels._residual_norm_kernel.signature.parameters["eps"].annotation != "tl.constexpr"
    x = torch.zeros(1, 131080, dtype=torch.bfloat16)
    kernels.add_rms_norm_(
        x, x.float(), torch.ones(131080), 2e-5, store_residual=True, normalize_stored=False
    )
    args = kernels._residual_norm_kernel.launches[-1].args
    assert args["eps"] == 2e-5 and args["BLOCK_SIZE"] == 65536


def test_native_two_stride_groups(kernels):
    x = torch.ones(2, 13, 8)[:, :12].view(2, 3, 4, 8)
    rows = kernels._rows(x, "x")
    assert rows.shape == (2, 12, 8) and rows.stride() == (104, 8, 1)
    assert rows.data_ptr() == x.data_ptr()


def test_launch_preserves_strides_over_int32(kernels, monkeypatch):
    signature = kernels._residual_norm_kernel.signature
    # Meta storage carries the large strides without allocating the gaps.
    launches = []

    class Recorder:
        def __getitem__(self, grid):
            return lambda *args, **kw: launches.append(args)

    monkeypatch.setattr(kernels, "_residual_norm_kernel", Recorder())
    stride = 2**31 + 9
    x = torch.empty_strided((2, 3), (stride, 1), device="meta")
    kernels._launch_residual_norm(
        x,
        x,
        x,
        torch.empty(3, device="meta"),
        x,
        EPS,
        has_x=True,
        store_residual=True,
        normalize_stored=False,
        sum_out=x,
    )
    args = signature.bind(*launches[0]).arguments
    assert args["x_stride"] == args["residual_stride"] == args["sum_stride"] == stride


# training wrappers


_FROZEN_RMS_NORM = "ctikernels.norm._frozen_rms_norm"


_ZERO_CENTERED_RMS_NORM = "ctikernels.norm._zero_centered_rms_norm"


def _load_training_norm(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, tuple[object, ...]]],
) -> ModuleType:
    """Import the leaf with recording stand-ins for Liger and the local kernels."""

    liger = ModuleType("liger_kernel")
    liger_transformers = ModuleType("liger_kernel.transformers")
    liger_functional = ModuleType("liger_kernel.transformers.functional")
    liger_functional.liger_rms_norm = _recorder(
        calls, "liger.rms_norm", lambda x, *_, **__: x.clone()
    )
    liger_functional.liger_fused_add_rms_norm = _recorder(
        calls, "liger.fused_add_rms_norm", lambda x, residual, *_, **__: (x.clone(), x + residual)
    )
    liger_transformers.functional = liger_functional
    frozen = ModuleType(_FROZEN_RMS_NORM)
    frozen.frozen_rms_norm = _recorder(calls, "frozen.rms_norm", lambda x, *_: x.clone())
    zero_centered = ModuleType(_ZERO_CENTERED_RMS_NORM)
    zero_centered.training_rms_norm = _recorder(
        calls, "zero_centered.rms_norm", lambda x, *_, **__: x.clone()
    )
    zero_centered.training_add_rms_norm = _recorder(
        calls, "zero_centered.add_rms_norm", lambda x, residual, *_, **__: (x.clone(), x + residual)
    )
    zero_centered.training_residual_rms_norm = _recorder(
        calls,
        "zero_centered.residual_rms_norm",
        lambda residual, *_, **__: (residual.clone(), residual),
    )
    for module in (liger, liger_transformers, liger_functional, frozen, zero_centered):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "ctikernels.norm.training", raising=False)
    return importlib.import_module("ctikernels.norm.training")


@pytest.mark.parametrize("train_weight", [False, True])
def test_direct_weighting_keeps_its_routes(monkeypatch, train_weight: bool) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8)
    residual = torch.randn(3, 8)
    weight = torch.randn(8, requires_grad=train_weight)

    operation.rms_norm(x, weight, 1e-6)
    operation.add_rms_norm(x, residual, weight, 1e-6)

    liger_options = (("casting_mode", "llama"), ("in_place", False))
    expected_norm = (
        ("liger.rms_norm", (x, weight, 1e-6, *liger_options))
        if train_weight
        else ("frozen.rms_norm", (x, weight, 1e-6))
    )
    assert calls == [
        expected_norm,
        ("liger.fused_add_rms_norm", (x, residual, weight, 1e-6, *liger_options)),
    ]


def test_zero_centered_routes_to_the_local_kernels(monkeypatch) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8, requires_grad=True)
    residual = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8)

    operation.rms_norm(x, weight, 1e-6, zero_centered=True)
    operation.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)
    operation.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True, save="inputs")
    operation.residual_rms_norm(residual, weight, 1e-6, zero_centered=True, out_dtype=torch.float32)

    assert calls == [
        ("zero_centered.rms_norm", (x, weight, 1e-6, ("out_dtype", None))),
        (
            "zero_centered.add_rms_norm",
            (x, residual, weight, 1e-6, ("save_inputs", False), ("zero_centered", True)),
        ),
        (
            "zero_centered.add_rms_norm",
            (x, residual, weight, 1e-6, ("save_inputs", True), ("zero_centered", True)),
        ),
        (
            "zero_centered.residual_rms_norm",
            (residual, weight, 1e-6, ("out_dtype", torch.float32), ("zero_centered", True)),
        ),
    ]


def test_zero_centered_passes_out_dtype_and_an_fp32_residual_through_training(monkeypatch) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, dtype=torch.bfloat16)

    operation.rms_norm(residual, weight, 1e-6, zero_centered=True, out_dtype=torch.bfloat16)
    operation.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)

    assert calls == [
        ("zero_centered.rms_norm", (residual, weight, 1e-6, ("out_dtype", torch.bfloat16))),
        (
            "zero_centered.add_rms_norm",
            (x, residual, weight, 1e-6, ("save_inputs", False), ("zero_centered", True)),
        ),
    ]


@pytest.mark.parametrize("train_weight", [False, True])
def test_direct_weighting_rejects_mixed_dtypes_before_any_kernel(
    monkeypatch, train_weight: bool
) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8, dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, dtype=torch.bfloat16, requires_grad=train_weight)

    with pytest.raises(TypeError, match="zero_centered=True"):
        operation.rms_norm(x, weight, 1e-6, out_dtype=torch.float32)
    if train_weight:
        with pytest.raises(TypeError, match="zero_centered=True"):
            operation.add_rms_norm(x, residual, weight, 1e-6)
        assert calls == []
    else:
        operation.add_rms_norm(x, residual, weight, 1e-6)
        assert calls[0][0] == "zero_centered.add_rms_norm"
        assert dict(calls[0][1][4:])["zero_centered"] is False


@pytest.mark.parametrize("train_weight", [False, True])
def test_direct_weighting_residual_norm_modes_require_frozen_weight(
    monkeypatch, train_weight: bool
) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8, requires_grad=True)
    residual = torch.randn(3, 8, requires_grad=True)
    weight = torch.randn(8, requires_grad=train_weight)

    if train_weight:
        with pytest.raises(ValueError, match="zero_centered=True"):
            operation.add_rms_norm(x, residual, weight, 1e-6, save="inputs")
        with pytest.raises(ValueError, match="frozen weight"):
            operation.residual_rms_norm(residual, weight, 1e-6, out_dtype=torch.float32)
        assert calls == []
    else:
        operation.add_rms_norm(x, residual, weight, 1e-6, save="inputs")
        operation.residual_rms_norm(residual, weight, 1e-6, out_dtype=torch.float32)
        assert [name for name, _ in calls] == [
            "zero_centered.add_rms_norm",
            "zero_centered.residual_rms_norm",
        ]
        assert dict(calls[0][1][4:]) == {"save_inputs": True, "zero_centered": False}
        assert dict(calls[1][1][3:]) == {"out_dtype": torch.float32, "zero_centered": False}


@pytest.mark.parametrize("zero_centered", [False, True])
def test_add_rms_norm_rejects_an_unknown_save_before_any_kernel(monkeypatch, zero_centered) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x, residual, weight = torch.randn(3, 8), torch.randn(3, 8), torch.randn(8)

    with pytest.raises(ValueError, match="'sum' or 'inputs'"):
        operation.add_rms_norm(x, residual, weight, 1e-6, zero_centered=zero_centered, save="x")

    assert calls == []


@pytest.mark.parametrize("grad_enabled", [True, False])
@pytest.mark.parametrize("name", ["rms_norm", "add_rms_norm", "residual_rms_norm"])
def test_zero_centered_rejects_a_trainable_weight_before_any_kernel(
    monkeypatch, name: str, grad_enabled: bool
) -> None:
    calls = []
    operation = _load_training_norm(monkeypatch, calls)
    x = torch.randn(3, 8, requires_grad=True)
    residual = torch.randn(3, 8)
    weight = torch.randn(8, requires_grad=True)

    with torch.set_grad_enabled(grad_enabled), pytest.raises(ValueError, match="frozen weight"):
        if name == "rms_norm":
            operation.rms_norm(x, weight, 1e-6, zero_centered=True)
        elif name == "add_rms_norm":
            operation.add_rms_norm(x, residual, weight, 1e-6, zero_centered=True)
        else:
            operation.residual_rms_norm(
                residual, weight, 1e-6, zero_centered=True, out_dtype=torch.float32
            )

    assert calls == []
