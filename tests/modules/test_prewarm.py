# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FlashInfer prewarm table and guards (CPU)."""

import pytest

from ctikernels._runtime import prewarm


def test_table_names_known_classes_and_generators():
    for what, classes, target, kwargs in prewarm.MODULES:
        assert what and set(classes) <= set(prewarm.DEVICE_CLASSES) and isinstance(kwargs, dict)
        module, generator = target.split(":")
        assert module.startswith("flashinfer.jit") and generator.startswith("gen_")
    covered = {c for _, classes, *_ in prewarm.MODULES for c in classes}
    assert covered == set(prewarm.DEVICE_CLASSES)
    assert prewarm.default_jobs() >= 1


def test_refuses_another_arch(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0")
    with pytest.raises(ValueError, match="12.0"):
        prewarm.prewarm("sm120")


def test_startup_entry_rejects_unknown_providers(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    with pytest.raises(ValueError, match="unknown"):
        prewarm.prewarm_flashinfer(modules=["no such provider"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    assert prewarm.prewarm_flashinfer() == []
