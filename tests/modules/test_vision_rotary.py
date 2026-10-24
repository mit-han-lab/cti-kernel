# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch

_LOCAL_ROTARY = "ctikernels.rotary._vision"


def _load_vision_rotary(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[torch.Tensor, ...]],
) -> ModuleType:
    """Import the leaf with a recording stand-in for the local kernel."""

    def apply_rotary(*args):
        calls.append(args)
        return args[0].clone(), args[1].clone()

    local_rotary = ModuleType(_LOCAL_ROTARY)
    local_rotary.apply_rotary = apply_rotary
    monkeypatch.setitem(sys.modules, _LOCAL_ROTARY, local_rotary)
    monkeypatch.delitem(sys.modules, "ctikernels.rotary.vision", raising=False)
    return importlib.import_module("ctikernels.rotary.vision")


def _inputs(gradient_input: str | None = None) -> dict[str, torch.Tensor]:
    shapes = {"query": (3, 2, 8), "key": (3, 2, 8), "cos": (3, 8), "sin": (3, 8)}
    return {
        name: torch.randn(shape, requires_grad=name == gradient_input)
        for name, shape in shapes.items()
    }


def test_vision_rotary_leaf_imports_local_kernel_lazily(monkeypatch) -> None:
    for name in ("ctikernels.rotary.vision", _LOCAL_ROTARY):
        monkeypatch.delitem(sys.modules, name, raising=False)

    importlib.import_module("ctikernels.rotary.vision")

    assert _LOCAL_ROTARY not in sys.modules


def test_vision_rotary_routes_borrowed_inputs_to_local_kernel(monkeypatch) -> None:
    calls = []
    operation = _load_vision_rotary(monkeypatch, calls)
    inputs = _inputs()

    query, key = operation.apply_rotary(**inputs)

    assert len(calls) == 1
    assert all(received is sent for received, sent in zip(calls[0], inputs.values(), strict=True))
    assert query is not inputs["query"] and key is not inputs["key"]


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("gradient_input", ["query", "key", "cos", "sin"])
def test_vision_rotary_gradient_guard_runs_before_the_kernel(
    monkeypatch, grad_enabled: bool, gradient_input: str
) -> None:
    calls = []
    operation = _load_vision_rotary(monkeypatch, calls)
    inputs = _inputs(gradient_input)

    with torch.set_grad_enabled(grad_enabled):
        if grad_enabled:
            with pytest.raises(RuntimeError, match="does not support autograd"):
                operation.apply_rotary(**inputs)
            assert calls == []
        else:
            operation.apply_rotary(**inputs)
            assert len(calls) == 1
