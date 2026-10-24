# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FlashInfer prewarm on this device's class: a second run builds nothing."""

import json
import os
import sys

import pytest
import torch

from ..support.child import run_child

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _device_class():
    major, minor = torch.cuda.get_device_capability()
    return {(10, 0): "sm100", (10, 3): "sm103", (12, 0): "sm120", (12, 1): "sm121"}.get(
        (major, minor)
    )


def _run(device_class):
    env = {k: v for k, v in os.environ.items() if k != "FLASHINFER_CUDA_ARCH_LIST"}
    out = run_child(
        [sys.executable, "-m", "ctikernels._runtime.prewarm", "--device-class", device_class],
        env=env,
        timeout=600,
        check=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_second_run_builds_nothing():
    device_class = _device_class()
    if device_class is None:
        pytest.skip("not a FlashInfer prewarm device class")
    pytest.importorskip("flashinfer")
    first = _run(device_class)
    assert first and {row["for"] for row in first}
    assert not any(row["built"] for row in _run(device_class))


def test_startup_entry_is_a_noop_when_warm():
    from ctikernels import prewarm_flashinfer

    if _device_class() is None:
        pytest.skip("not a FlashInfer prewarm device class")
    pytest.importorskip("flashinfer")
    prewarm_flashinfer()
    rows = prewarm_flashinfer()
    assert rows and not any(row["built"] for row in rows)
