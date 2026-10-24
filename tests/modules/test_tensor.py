# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from ctikernels.tensor import weak_tensor_alias


def test_weak_alias_rejects_missing_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sgl_kernel", None)

    with pytest.raises(RuntimeError, match=r"require sgl_kernel\.weak_ref_tensor") as caught:
        weak_tensor_alias()
    assert isinstance(caught.value.__cause__, ImportError)


def test_weak_alias_rejects_missing_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sgl_kernel", ModuleType("sgl_kernel"))

    with pytest.raises(RuntimeError, match=r"require sgl_kernel\.weak_ref_tensor") as caught:
        weak_tensor_alias()
    assert isinstance(caught.value.__cause__, ImportError)


@pytest.mark.parametrize("entry_point", [None, object()])
def test_weak_alias_rejects_non_callable_entry_point(
    monkeypatch: pytest.MonkeyPatch, entry_point: object
) -> None:
    provider = ModuleType("sgl_kernel")
    provider.weak_ref_tensor = entry_point
    monkeypatch.setitem(sys.modules, "sgl_kernel", provider)

    with pytest.raises(RuntimeError, match=r"require callable sgl_kernel\.weak_ref_tensor"):
        weak_tensor_alias()


def test_weak_alias_resolves_the_native_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    def native_alias(tensor):
        return tensor

    provider = ModuleType("sgl_kernel")
    provider.weak_ref_tensor = native_alias
    monkeypatch.setitem(sys.modules, "sgl_kernel", provider)

    assert weak_tensor_alias() is native_alias
