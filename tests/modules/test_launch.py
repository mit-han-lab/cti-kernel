# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest
import torch

from ctikernels._runtime import launch as module


def test_prime_launches_once_per_kernel(monkeypatch):
    monkeypatch.setattr(module, "_primed", set())
    calls = []
    kernel = object()
    assert module.prime(kernel, calls.append, 1)
    assert not module.prime(kernel, calls.append, 2)
    assert module.prime(("cublas", 7), lambda *a, **k: calls.append((a, k)), 3, x=4)
    assert calls == [1, ((3,), {"x": 4})]


def test_prime_refuses_capture_and_retries_after_a_failed_launch(monkeypatch):
    monkeypatch.setattr(module, "_primed", set())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        module.prime("k", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def fail():
        raise ValueError("launch failed")

    with pytest.raises(ValueError):
        module.prime("k", fail)
    assert module.prime("k", lambda: None)
