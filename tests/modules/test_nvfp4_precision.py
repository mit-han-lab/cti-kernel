# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU regressions for the GPU precision gate's provider and table coverage."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ctikernels.linear import _nvfp4_dispatch as dispatch


@pytest.fixture
def precision():
    path = Path(__file__).parents[1] / "gpu" / "test_nvfp4.py"
    spec = importlib.util.spec_from_file_location("nvfp4_precision_gpu", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "tests.gpu"  # its oracle helpers are relative imports
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("quack", [False, True])
def test_precision_variants_include_flashinfer_independently_of_quack(
    precision, monkeypatch, available, quack
):
    monkeypatch.setattr(precision, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(precision, "_quack_runs", lambda: quack)
    monkeypatch.setattr(
        precision._nvfp4_flashinfer, "load", lambda _: object() if available else None
    )
    linear = precision._linear_variants(3, 2048, 512)
    gated = precision._gated_variants(3, 2048, 512)
    assert (("flashinfer", dispatch.FlashInfer()) in linear) == available
    assert (("flashinfer", dispatch.UNFUSED, dispatch.FlashInfer()) in gated) == available
    assert any(provider == "quack" for provider, _ in linear) == quack
    assert any(provider == "quack" for provider, _, _ in gated) == quack


def test_flashinfer_precision_cases_union_devices_and_cover_bucket_boundaries(
    precision, monkeypatch
):
    shape = (64, 32)
    monkeypatch.setattr(precision, "FI_LINEAR_SHAPES", (shape,))
    monkeypatch.setattr(
        dispatch, "GATED", {"third GPU": {shape: ((4, dispatch.UNFUSED, dispatch.FlashInfer()),)}}
    )
    cases = precision._flashinfer_dispatch_cases()
    # Linear: FlashInfer at every M, so both ends of every bucket up to 4096, then the next.
    linear_rows = {1, 2} | {(1 << b) + 1 for b in range(1, 13)} | {1 << b for b in range(2, 14)}
    assert cases == sorted(
        [("linear", shape, m, "linear") for m in linear_rows]
        # The final entry also serves beyond M=4: test the next bucket's endpoints.
        + [("gated", shape, m, dispatch.UNFUSED) for m in (1, 2, 3, 4, 5, 8)]
    )


def test_fp64_gated_reference_detects_review_double_rounding(precision):
    import torch.nn.functional as F

    from ctikernels.linear.nvfp4 import NVFP4Activation, NVFP4Weight, blocked_scale

    k = 5120
    data = torch.full((32, k // 2), 0x22, dtype=torch.uint8)
    blocks = torch.ones(32, k // 16).to(torch.float8_e4m3fn)
    gate = NVFP4Weight(data, blocks, torch.tensor([0.7]))
    up = NVFP4Weight(data, blocks, torch.tensor([0.9]))
    a = NVFP4Activation(data[:1], blocked_scale(blocks[:1]), torch.tensor([1.004 / k]))
    output_scale = torch.tensor([0.066942364])
    g64, u64 = precision._gated_fp64_projections(a, gate, up)
    correct = F.silu(g64.bfloat16().float()) * u64.bfloat16().float()
    _, scale = precision.reference_quantize(correct, output_scale)
    assert torch.all(scale.view(torch.uint8) == 0x38)
    # This intentionally reproduces the rejected order, to prove oracle sensitivity.
    unscaled = (a.global_scale * k).bfloat16().float()
    bad_gate = (unscaled * gate.global_scale).bfloat16().float()
    bad_up = (unscaled * up.global_scale).bfloat16().float()
    _, wrong_scale = precision.reference_quantize(
        (F.silu(bad_gate) * bad_up).expand(1, 32), output_scale
    )
    assert torch.all(wrong_scale.view(torch.uint8) == 0x39)
