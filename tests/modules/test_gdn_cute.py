# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn._cute's resolution on the CPU: None (the fla-core fallback) wherever the CuTe
provider cannot run, nvidia-cutlass-dsl imported only by a resolution that can use it, and one
resolution per device."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ctikernels.gdn
from ctikernels.gdn import _cute

from ..support.child import run_child

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "python"
# nvidia-cutlass-dsl's modules; its .pth imports only the small nvidia_cutlass_dsl wrapper at
# interpreter start, so "cutlass" (the DSL) is what must stay unimported.
CUTLASS = ("cutlass", "nvidia_cutlass_dsl")
DEVICE = torch.device("cuda", 0)


class NoCutlass(importlib.abc.MetaPathFinder):
    """nvidia-cutlass-dsl as absent, recording each import of it."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in CUTLASS:
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def _installed(version: str | None):
    def lookup(name: str) -> str:
        assert name == _cute.DISTRIBUTION, name
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    return lookup


def _never(*args, **kwargs):
    raise AssertionError("looked up")


@pytest.fixture
def blocked(monkeypatch: pytest.MonkeyPatch) -> NoCutlass:
    """An sm_120 device stand-in resolved afresh, without nvidia-cutlass-dsl."""
    finder = NoCutlass()
    # Nothing imported earlier stands in for an import.
    for name in [name for name in sys.modules if name.split(".")[0] in CUTLASS]:
        monkeypatch.delitem(sys.modules, name)
    for leaf in ("_cute_bwd", "_cute_fwd"):
        monkeypatch.delitem(sys.modules, f"ctikernels.gdn.{leaf}", raising=False)
        monkeypatch.delattr(ctikernels.gdn, leaf, raising=False)
    monkeypatch.setattr(_cute, "_forwards", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.delenv(_cute.VARIABLE, raising=False)
    return finder


def test_off_cuda_it_is_none():
    for device in (torch.device("cpu"), torch.device("meta")):
        assert _cute.load(device) is None


def test_without_cuda_it_never_queries_an_accelerator(blocked, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _never)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _never)
    monkeypatch.setattr(importlib.metadata, "version", _never)
    assert _cute.training(DEVICE) is None
    assert blocked.attempts == []


@pytest.mark.parametrize("version", [None, "4.6.0.dev0", "4.8.1"])
def test_without_the_pinned_version_it_is_none_and_imports_nothing(blocked, monkeypatch, version):
    monkeypatch.setattr(importlib.metadata, "version", _installed(version))
    assert _cute.load(DEVICE) is None
    assert blocked.attempts == []


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (10, 3), (12, 2)])
def test_on_other_devices_it_is_none_and_looks_up_nothing(blocked, monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(importlib.metadata, "version", _never)
    assert _cute.load(DEVICE) is None
    assert blocked.attempts == []


def test_the_variable_turns_it_off_before_any_lookup(blocked, monkeypatch):
    monkeypatch.setenv(_cute.VARIABLE, "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", _never)
    monkeypatch.setattr(importlib.metadata, "version", _never)
    assert _cute.load(DEVICE) is None


def test_a_pinned_install_that_does_not_import_warns_once_and_is_none(blocked, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _installed(_cute.VERSION))
    with pytest.warns(RuntimeWarning, match="does not import"):
        assert _cute.load(DEVICE) is None
    assert "cutlass" not in sys.modules
    # Resolved once per device: no second lookup, import or warning.
    monkeypatch.setattr(importlib.metadata, "version", _never)
    blocked.attempts.clear()
    assert _cute.load(DEVICE) is None
    assert blocked.attempts == []


def test_importing_the_provider_never_imports_cutlass(tmp_path: Path):
    code = """
import sys

import torch

from ctikernels.gdn import _cute, _fla  # noqa: F401

assert _cute.training(torch.device("cpu")) is None
if not torch.cuda.is_available():
    assert _cute.training(torch.device("cuda", 0)) is None
for prefix in ("cutlass", "tvm_ffi", "fla"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    run_child((sys.executable, "-c", code), env=environment, cwd=tmp_path, timeout=600, check=True)


@pytest.mark.parametrize(
    ("stage", "kernel", "shape"),
    [
        ("_prologue", "IntraP", ()),
        ("_recurrence", "RecurS", (32,)),
        ("_wu", "WuB", ()),
        ("_local_dv", "LocalDV", ()),
        ("_states", "StatesR", (32, 4)),
        ("_state_grads", "GradStatesD", (32,)),
        ("_qk_grads", "GradQK", ()),
        ("_wy_grads", "GradWY", ()),
    ],
)
def test_warm_stages_skip_compile_setup_and_keep_geometry_and_device_separate(
    monkeypatch, stage, kernel, shape
):
    """Exercise stage lookup without CUDA; a hit must bypass argument/factory setup entirely."""
    monkeypatch.setattr(_cute, "_compile", lambda *args: object())

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(multi_processor_count=1)
    )

    def provider():
        kernels = SimpleNamespace(**{kernel: lambda *args: object()})
        return _cute.ChunkKernels(torch.device("cpu"), kernels, kernels, gate=None)

    first, second = provider(), provider()
    lookup = getattr(first, stage)
    compiled = lookup(16, 32, *shape)
    assert lookup(16, 48, *shape) is not compiled
    assert getattr(second, stage)(16, 32, *shape) is not compiled
    monkeypatch.setattr(first, "_compile", _never)
    assert lookup(16, 32, *shape) is compiled
