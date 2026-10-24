# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Offline compilation only: fake tensors and streams, no CUDA allocations or launches."""

import pytest

from ctikernels.linear._dispatch import Config
from ctikernels.linear.cute._provider import available, compile_kernel

pytestmark = pytest.mark.skipif(not available(), reason="nvidia-cutlass-dsl 4.8.0 absent")


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("mtiles", [1, 8])
@pytest.mark.parametrize("variant", ["smem", "reg"])
def test_gated_compile_smoke(capability, mtiles, variant):
    config = Config(32, 128, 2, 3, 1, variant)
    assert compile_kernel(4096, 4096, config, capability, mtiles, gate_stride=10240) is not None


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("splits", [1, 3])
@pytest.mark.parametrize("variant", ["smem", "reg"])
@pytest.mark.parametrize("n", [67, 256])
def test_fp32_compile_smoke(capability, splits, variant, n):
    config = Config(32, 128, 2, splits, 1, variant)
    assert compile_kernel(n, 4096, config, capability, mtiles=8, fp32=True) is not None
