# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import contextlib
import importlib.metadata
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import torch

from ctikernels import _flashinfer_version as guard
from ctikernels.activation import inference as act
from ctikernels.norm import cute as norm_cute
from ctikernels.norm import inference as norm
from ctikernels.norm._dispatch import Config, Entry


@pytest.fixture(autouse=True)
def clean_version_cache(monkeypatch):
    guard.installed_version.cache_clear()
    guard.importable.cache_clear()
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace())
    monkeypatch.delenv("CTI_FLASHINFER_ALLOW_ANY", raising=False)
    yield
    guard.installed_version.cache_clear()
    guard.importable.cache_clear()


@pytest.mark.parametrize(
    "version", [*guard.ALLOWED, "0.7.1", "0.7.2", "0.6.11.post3", "0.6.11", "missing"]
)
def test_runtime_version_is_cached_and_override_is_explicit(monkeypatch, version):
    calls = []

    def lookup(package):
        calls.append(package)
        if version == "missing":
            raise importlib.metadata.PackageNotFoundError(package)
        return version

    monkeypatch.setattr(importlib.metadata, "version", lookup)
    assert guard.allowed() == (version in guard.ALLOWED)
    assert guard.allowed() == (version in guard.ALLOWED)
    assert calls == ["flashinfer-python"]
    if version not in guard.ALLOWED:
        with pytest.raises(RuntimeError, match="allowed versions") as error:
            guard.require_allowed()
        assert (version if version != "missing" else "not installed") in str(error.value)
        assert all(allowed in str(error.value) for allowed in guard.ALLOWED)
    monkeypatch.setenv("CTI_FLASHINFER_ALLOW_ANY", "true")
    assert guard.allowed() == (version in guard.ALLOWED)
    monkeypatch.setenv("CTI_FLASHINFER_ALLOW_ANY", "1")
    guard.require_allowed()
    assert calls == ["flashinfer-python"]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("source", ["argument", "environment"])
def test_forced_providers_reject_before_launch(monkeypatch, empty, source):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")

    def unexpected(*a, **k):
        pytest.fail("unqualified provider launched")

    monkeypatch.setattr(norm.flashinfer, "add_rms_norm_", unexpected)
    monkeypatch.setattr(act.flashinfer, "silu_and_mul", unexpected)
    x = torch.ones((0 if empty else 1, 16), dtype=torch.bfloat16)
    monkeypatch.setenv(norm.VARIABLE, "flashinfer" if source == "environment" else "auto")
    with pytest.raises(RuntimeError, match="9.9.9"):
        norm.AddRMSNorm(provider="flashinfer" if source == "argument" else None)(
            x, x.clone(), torch.ones(16, dtype=x.dtype), 1e-6
        )
    if empty:
        # MB-R2: an empty activation never resolves or launches any provider.
        assert act.SiluAndMul(provider="flashinfer")(x).shape == (0, 8)
    else:
        with pytest.raises(RuntimeError, match="9.9.9"):
            act.SiluAndMul(provider="flashinfer")(x)


def test_direct_providers_and_no_alternative_reject_unqualified_version(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    monkeypatch.setenv(norm.VARIABLE, "flashinfer")
    # No vendor methods are present: rejection must precede even their import.
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace())
    x, weight = torch.ones(1, 16, dtype=torch.bfloat16), torch.ones(16, dtype=torch.bfloat16)
    for call in (
        lambda: norm.flashinfer.add_rms_norm_(x, x.clone(), weight, 1e-6),
        lambda: norm.rms_norm(x, weight, 1e-6),
        lambda: norm.rms_norm_(x, weight, 1e-6),
        lambda: act.flashinfer.silu_and_mul(x),
        lambda: act.flashinfer.gelu_and_mul(x),
        lambda: act.flashinfer.gelu_tanh_and_mul(x),
    ):
        with pytest.raises(RuntimeError, match="9.9.9"):
            call()


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("cute_available", [False, True])
def test_auto_norm_avoids_unqualified_flashinfer(monkeypatch, zero, cute_available):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    # The resolver checks DSL presence first; _cute below models the compile outcome.
    monkeypatch.setattr(norm_cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("rtx5090", (12, 0)))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    # Real 5090 FlashInfer band, including higher rank: flattened M is 512.
    x = torch.ones((2, 256, 4096), dtype=torch.bfloat16)
    residual, weight = x.clone(), torch.ones(4096, dtype=x.dtype)
    op = norm.AddRMSNorm(provider="auto")
    monkeypatch.setattr(op, "_cute", lambda *a: (lambda *args: None) if cute_available else None)
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.norm._zero_centered_rms_norm",
        SimpleNamespace(add_rms_norm_=lambda x, r, *a, **k: (x, r)),
    )
    op(x, residual, weight, 1e-6, zero_centered=zero)
    assert op.provider == ("cute" if cute_available else "triton")
    assert "9.9.9" in op.provider.reason


def test_auto_activation_reroutes_only_for_missing_dsl(monkeypatch):
    from ctikernels._norm_act_reroute import Provider
    from ctikernels.activation import cute

    monkeypatch.setattr(importlib.metadata, "version", lambda _: "9.9.9")
    monkeypatch.setattr(cute, "available", lambda: False)
    x = torch.ones((2, 3, 256), dtype=torch.bfloat16)
    op = act.SiluAndMul(provider="auto")
    monkeypatch.setattr(op, "_triton", lambda *a: (Provider("triton"), lambda x, out: x))
    assert op(x) is x
    assert op.provider == "triton" and "nvidia-cutlass-dsl" in op.provider.reason


@pytest.mark.parametrize("zero", [False, True])
def test_flashinfer_norm_flattens_views_without_replacing_outputs(monkeypatch, zero):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.7.0.post1")
    x, residual, weight = torch.ones(2, 3, 16), torch.ones(2, 3, 16), torch.ones(16)

    def launch(a, b, w, eps):
        assert a.shape == b.shape == (6, 16)
        assert a.data_ptr() == x.data_ptr() and b.data_ptr() == residual.data_ptr()
        assert w is weight and eps == 1e-6
        a.fill_(2)
        b.fill_(3)

    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(fused_add_rmsnorm=launch, gemma_fused_add_rmsnorm=launch),
    )
    result = norm.flashinfer.add_rms_norm_(x, residual, weight, 1e-6, zero_centered=zero)
    assert result[0] is x and result[1] is residual
    assert torch.all(x == 2) and torch.all(residual == 3)


@pytest.mark.parametrize("source", ["table", "config", "env"])
@pytest.mark.parametrize("static_weight", [False, True])
def test_weight_prefetch_requires_public_opt_in(monkeypatch, source, static_weight):
    monkeypatch.setattr(norm_cute, "available", lambda: True)
    monkeypatch.setattr(norm, "_admitted", lambda *a: True)
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    monkeypatch.setenv("CTI_NORM_CONFIG", "512,1,1,1")
    requested = Config(512, 1, True, True)
    seen = []

    def entry(kind, hidden, dtype, m, zero, static, residual_dtype=None):
        assert static == static_weight
        assert m == 6  # Rank is never used as the row count.
        return Entry("cute", requested)

    monkeypatch.setattr(norm, "entry_for", entry)
    op = norm.AddRMSNorm(
        provider="auto" if source == "table" else "cute",
        config=requested if source == "config" else None,
        static_weight=static_weight,
    )
    monkeypatch.setattr(op, "_cute", lambda *a: seen.append(a[-3]) or (lambda *args: None))
    x = torch.ones((2, 3, 4096), dtype=torch.bfloat16)
    op(x, x.clone(), torch.ones(4096, dtype=x.dtype), 1e-6)
    assert seen == [Config(512, 1, True, static_weight)]


def test_norm_activation_allow_list_is_exact():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    pin = next(
        dep.split("==")[1]
        for dep in pyproject["project"]["dependencies"]
        if dep.startswith("flashinfer-python==")
    )
    assert guard.ALLOWED == (pin,) == ("0.7.0.post1",)


@pytest.mark.parametrize("bypass", [False, True])
@pytest.mark.parametrize("failure", [ImportError, ModuleNotFoundError])
def test_failed_import_is_cached_and_cannot_be_bypassed(monkeypatch, bypass, failure):
    calls = []

    def broken(name):
        calls.append(name)
        raise failure("blocked package")

    monkeypatch.setattr(importlib, "import_module", broken)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: guard.ALLOWED[0])
    if bypass:
        monkeypatch.setenv("CTI_FLASHINFER_ALLOW_ANY", "1")
    assert not guard.allowed()
    assert not guard.allowed()
    assert guard.unusable_reason() == "FlashInfer is not importable"
    with pytest.raises(RuntimeError, match="not importable"):
        guard.require_allowed()
    assert calls == ["flashinfer"]


def test_shadow_package_import_failure(monkeypatch, tmp_path):
    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "__init__.py").write_text('raise ImportError("unavailable dependency")\n')
    monkeypatch.delitem(sys.modules, "flashinfer")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: guard.ALLOWED[0])
    monkeypatch.setenv("CTI_FLASHINFER_ALLOW_ANY", "1")
    assert not guard.allowed()
    assert guard.unusable_reason() == "FlashInfer is not importable"
