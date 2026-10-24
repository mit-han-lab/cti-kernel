# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Accuracy gates must tolerate cancellation without hiding worse BF16 results."""

import pytest
import torch

from benchmark.kernel_tuning.linear_smallm import accuracy, reference_and_condition


def test_cancellation_uses_condition_not_output_magnitude():
    ref = torch.tensor([[100.1, 1e-8]], dtype=torch.float64)
    condition = torch.tensor([[100.1, 10000.0]], dtype=torch.float64)
    baseline = ref.bfloat16()
    y = baseline.clone()
    y[0, 1] = 0
    assert accuracy(y, baseline, ref, condition, k=4096)["correct"]
    assert ((y.double() - ref).abs() / ref.abs()).max() == 1


@pytest.mark.parametrize(
    "y,baseline,condition,gate",
    [
        ([1.02, 0], [1, 0], [100, 100], "max_abs"),
        ([1, 0.3, 0.3, 0.3], [1, 0, 0, 0], [100] * 4, "rms"),
        ([0.1, 0.01], [0.1, 0], [1e6, 1], "max_normalized"),
        ([1e-8, 0], [0, 0], [100, 100], "max_abs"),
    ],
)
def test_each_error_gate_rejects_regressions(y, baseline, condition, gate):
    y, baseline, condition = (
        torch.tensor(a, dtype=torch.float64) for a in (y, baseline, condition)
    )
    metrics = accuracy(y, baseline, torch.zeros_like(y), condition, k=4096)
    assert not metrics["correct"]
    assert metrics[gate] > metrics["cublas_" + gate]


def test_reference_and_condition_keep_cancellation_scale():
    x = torch.tensor([[1.0, -1.0]], dtype=torch.bfloat16)
    w = torch.ones(1025, 2, dtype=torch.bfloat16)
    ref, condition = reference_and_condition(x, w)
    assert ref.dtype == condition.dtype == torch.float64
    assert torch.equal(ref, torch.zeros_like(ref))
    assert torch.equal(condition, torch.full_like(condition, 2))
    assert accuracy(ref.bfloat16(), ref.bfloat16(), ref, condition, k=x.shape[1])["correct"]


@pytest.mark.parametrize("scale", [2**-40, 1, 2**40])
@pytest.mark.parametrize("sign", [-1, 1])
def test_single_bf16_ulp_rounding_flip_passes(scale, sign):
    ref = torch.full((1, 4096), 100.1, dtype=torch.float64)
    ref[0, 0] = 128.49
    condition = torch.full_like(ref, 4096)
    baseline = ref.bfloat16()
    y = baseline.clone()
    y[0, 0] = 129  # Adjacent BF16 value across the rounding boundary.
    y, baseline, ref = (a * (scale * sign) for a in (y, baseline, ref))
    condition *= scale
    metrics = accuracy(y, baseline, ref, condition, k=4096)
    assert metrics["max_abs"] > metrics["cublas_max_abs"] * 1.01
    assert metrics["correct"]
    assert metrics["max_ulp_error"] == pytest.approx(0.51)
    assert metrics["cublas_max_ulp_error"] == pytest.approx(0.49)
    assert metrics["max_ulp_error_limit"] == 1

    # Another ULP is outside the rounding-flip budget, even though the other
    # two aggregate error gates pass when this element has more cancellation.
    y[0, 0] = 130 * scale * sign
    condition[0, 0] *= 10
    metrics = accuracy(y, baseline, ref, condition, k=4096)
    assert not metrics["correct"]
    assert metrics["max_ulp_error"] > metrics["max_ulp_error_limit"]
    assert metrics["max_normalized"] <= metrics["max_normalized_limit"]
    assert metrics["rms"] <= metrics["rms_limit"]


@pytest.mark.parametrize("corruption", ["dropped_k_chunk", "swapped_fragment"])
@pytest.mark.parametrize("k", [256, 4096, 17408])
def test_structural_errors_fail(corruption, k):
    x = torch.ones((2, k), dtype=torch.bfloat16)
    w = torch.ones((32, k), dtype=torch.bfloat16)
    w[16:] = 2
    w[:, 0] = 0.1
    ref, condition = reference_and_condition(x, w)
    baseline = ref.bfloat16()
    if corruption == "dropped_k_chunk":
        y = (x[:, : k // 2].double() @ w[:, : k // 2].double().t()).bfloat16()
    else:
        y = baseline.roll(16, dims=1)
    metrics = accuracy(y, baseline, ref, condition, k=k)
    assert not metrics["correct"]
    assert metrics["max_normalized"] > 0.4
    assert metrics["max_normalized"] > metrics["max_normalized_limit"]
    assert metrics["max_ulp_error"] > metrics["max_ulp_error_limit"]


@pytest.mark.parametrize("k", [4096, 12288, 17408])
def test_long_reduction_cancellation_floor(k):
    # An ordinary rounding error dominates RMS/normalized error; a cancellation
    # element differs by 1.14 old floor units, reproducing the isolated ULP fail.
    ref = torch.tensor([[100.1, 0.0]], dtype=torch.float64)
    condition = torch.tensor([[1120.0, float(k)]], dtype=torch.float64)
    baseline = ref.bfloat16()
    y = baseline.clone()
    y[0, 1] = 1.14 * 2 * torch.finfo(torch.float32).eps * k
    old = accuracy(y, baseline, ref, condition, k=1024)
    assert old["max_ulp_error"] > old["max_ulp_error_limit"]
    assert old["rms"] <= old["rms_limit"]
    assert old["max_normalized"] <= old["max_normalized_limit"]
    fixed = accuracy(y, baseline, ref, condition, k=k)
    assert fixed["correct"]
    assert fixed["max_normalized_limit"] == old["max_normalized_limit"]
    assert fixed["rms_limit"] == old["rms_limit"]


@pytest.mark.parametrize("k", [4096, 17408])
def test_one_missing_128_element_chunk_fails_at_large_k(k):
    x = torch.ones((1, k), dtype=torch.bfloat16)
    w = torch.ones((16, k), dtype=torch.bfloat16)
    w[:, 0] = 0.1
    ref, condition = reference_and_condition(x, w)
    y = (x[:, :-128].double() @ w[:, :-128].double().t()).bfloat16()
    metrics = accuracy(y, ref.bfloat16(), ref, condition, k)
    assert not metrics["correct"]
    # At K=17408, 128 happens to be one BF16 ULP. The independent normalized
    # gate must still reject that missing chunk even if the ULP gate allows it.
    assert metrics["max_normalized"] > metrics["max_normalized_limit"]


def test_fp32_accuracy_has_no_bf16_rounding_allowance():
    ref = torch.tensor([[100.1, 1.01]], dtype=torch.float64)
    condition = torch.full_like(ref, 4096)
    baseline = ref.float()
    assert accuracy(baseline, baseline, ref, condition, 4096, out_dtype="fp32")["correct"]
    assert not accuracy(ref.bfloat16().float(), baseline, ref, condition, 4096, out_dtype="fp32")[
        "correct"
    ]


def _fp32_check(y):
    """Exact FP32 baseline, zero reference, unit condition: errors are in eps32."""
    ref = torch.zeros_like(y, dtype=torch.float64)
    return accuracy(y, ref.clone(), ref, torch.ones_like(ref), 4096, out_dtype="fp32")


@pytest.mark.parametrize("scale,correct", [(0.99, True), (1.01, False)])
def test_fp32_accuracy_max_floor_is_four_eps32(scale, correct):
    # One element near 4 eps32; 63 exact elements keep RMS below eps32.
    eps32 = torch.finfo(torch.float32).eps
    y = torch.zeros((1, 64), dtype=torch.float64)
    y[0, 0] = scale * 4 * eps32
    metrics = _fp32_check(y)
    assert metrics["max_normalized_limit"] == 4 * eps32
    assert metrics["rms_normalized"] < eps32
    assert metrics["correct"] is correct


@pytest.mark.parametrize("scale,correct", [(0.99, True), (1.01, False)])
def test_fp32_accuracy_rms_floor_is_eps32(scale, correct):
    # Every element at the same error: max = RMS, well inside 4 eps32.
    eps32 = torch.finfo(torch.float32).eps
    metrics = _fp32_check(torch.full((1, 64), scale * eps32, dtype=torch.float64))
    assert metrics["rms_normalized_limit"] == eps32
    assert metrics["max_normalized"] < 4 * eps32
    assert metrics["correct"] is correct


def test_fp32_accuracy_rejects_six_eps32():
    eps32 = torch.finfo(torch.float32).eps
    y = torch.zeros((1, 64), dtype=torch.float64)
    y[0, 0] = 6 * eps32
    assert not _fp32_check(y)["correct"]
