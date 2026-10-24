# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest
import torch


def _load_inference_rotary(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, object]],
) -> ModuleType:
    def apply_rotary(**kwargs: object) -> None:
        calls.append(kwargs)

    flashinfer = ModuleType("flashinfer")
    flashinfer.apply_rope_with_cos_sin_cache_inplace = apply_rotary
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.delitem(sys.modules, "ctikernels.rotary.inference", raising=False)
    return importlib.import_module("ctikernels.rotary.inference")


def test_inference_rotary_wraps_provider_with_explicit_mutation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    inference_rotary = _load_inference_rotary(monkeypatch, calls)
    query = torch.randn(3, 2, 8)
    key = torch.randn(3, 1, 8)
    cache = torch.randn(16, 8)
    positions = torch.arange(3)

    rotated_query, rotated_key = inference_rotary.apply_rotary_(
        query,
        key,
        cache,
        positions,
        head_dim=8,
    )

    assert rotated_query is query
    assert rotated_key is key
    assert len(calls) == 1
    assert calls[0]["positions"] is positions
    assert calls[0]["query"] is query
    assert calls[0]["key"] is key
    assert calls[0]["head_size"] == 8
    assert calls[0]["cos_sin_cache"] is cache


def test_inference_rotary_rejects_gradient_storage_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    inference_rotary = _load_inference_rotary(monkeypatch, calls)

    with pytest.raises(RuntimeError, match="cannot modify"):
        inference_rotary.apply_rotary_(
            torch.randn(3, 2, 8, requires_grad=True),
            torch.randn(3, 1, 8),
            torch.randn(16, 8),
            torch.arange(3),
            head_dim=8,
        )

    assert calls == []


@pytest.mark.parametrize("rank", [2, 3])
def test_inference_key_rotary_routes_only_owned_key_storage(monkeypatch, rank: int) -> None:
    provider_calls: list[dict[str, object]] = []
    operation = _load_inference_rotary(monkeypatch, provider_calls)
    received = []

    def rotate_key(key, cache, positions, *, head_dim):
        received.append((key, cache, positions, head_dim))
        key.add_(1)

    local = ModuleType("ctikernels.rotary._key_only")
    local.apply_rotary_key = rotate_key
    monkeypatch.setitem(sys.modules, local.__name__, local)
    storage = torch.randn(5, 48)
    expected = storage.clone()
    expected[:, 8:24].add_(1)
    key = storage[:, 8:24]
    if rank == 3:
        key = key.view(5, 2, 8)
    cache = torch.randn(16, 8)
    positions = torch.arange(5)
    original_cache, original_positions = cache.clone(), positions.clone()

    assert operation.apply_rotary_key_(key, cache, positions, head_dim=8) is key

    assert len(received) == 1
    assert received[0][0] is key and received[0][1] is cache and received[0][2] is positions
    assert received[0][3] == 8
    assert provider_calls == []
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)
    torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("gradient_input", ["key", "cache"])
def test_inference_key_rotary_gradient_guards(monkeypatch, grad_enabled, gradient_input) -> None:
    operation = _load_inference_rotary(monkeypatch, [])
    calls = []
    local = ModuleType("ctikernels.rotary._key_only")
    local.apply_rotary_key = lambda *args, **kwargs: calls.append(args)
    monkeypatch.setitem(sys.modules, local.__name__, local)
    key = torch.randn(3, 16, requires_grad=gradient_input == "key")
    cache = torch.randn(16, 8, requires_grad=gradient_input == "cache")
    positions = torch.arange(3)

    with torch.set_grad_enabled(grad_enabled):
        if gradient_input == "key" or grad_enabled:
            with pytest.raises(RuntimeError, match="cannot modify|autograd"):
                operation.apply_rotary_key_(key, cache, positions, head_dim=8)
            assert calls == []
        else:
            assert operation.apply_rotary_key_(key, cache, positions, head_dim=8) is key
            assert len(calls) == 1


def test_inference_rotary_takes_int64_positions_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    inference_rotary = _load_inference_rotary(monkeypatch, calls)
    positions = torch.arange(3, dtype=torch.int32)
    with pytest.raises(TypeError, match="int64"):
        inference_rotary.apply_rotary_(
            torch.randn(3, 2, 8), torch.randn(3, 1, 8), torch.randn(16, 8), positions, head_dim=8
        )
    assert calls == []
