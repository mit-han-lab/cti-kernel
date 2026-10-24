# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The shared accuracy gate (tests/support/accuracy.py) fails on undefined bounds; ulps count
exactly."""

from __future__ import annotations

import math

import pytest
import torch

from ..support.accuracy import assert_accurate, assert_within_ulps, ulp_distance, ulps

NAN = float("nan")


def _gate(actual, reference, baseline, **kwargs) -> None:
    assert_accurate(
        torch.tensor(actual, dtype=torch.float32),
        torch.tensor(reference, dtype=torch.float64),
        torch.tensor(baseline, dtype=torch.float32),
        family="norm",
        **kwargs,
    )


def test_accepts_output_within_bound() -> None:
    _gate([1.0, 2.0], [1.0, 2.0], [1.0, 2.0])


def test_rejects_wrong_output() -> None:
    with pytest.raises(AssertionError, match="error"):
        _gate([100.0], [0.0], [0.0])


def test_rejects_nan_baseline() -> None:
    with pytest.raises(AssertionError, match="non-finite torch"):
        _gate([100.0], [0.0], [NAN])


def test_forbid_rejects_nan_reference() -> None:
    with pytest.raises(AssertionError, match="non-finite reference"):
        _gate([100.0, 1.0], [NAN, 1.0], [1.0, 1.0], nonfinite="forbid")


def test_match_accepts_matching_nan() -> None:
    _gate([NAN, 1.0], [NAN, 1.0], [NAN, 1.0])


@pytest.mark.parametrize(
    "dtype,actual,expected",
    [
        (torch.bfloat16, 0.1240234375, 0.12353515625),  # 2**-11 in [1/16, 1/8)
        (torch.float16, 0.0897216796875, 0.08978271484375),  # 2**-14 in [1/16, 1/8)
    ],
)
def test_one_output_ulp_counts_exactly_one(dtype, actual, expected) -> None:
    a, e = torch.tensor([actual], dtype=dtype), torch.tensor([expected], dtype=dtype)
    assert ulp_distance(a, e).item() == 1.0
    assert_within_ulps(a, e)
    assert_within_ulps(a, e, torch.tensor([0.1156], dtype=torch.float64))  # same-binade scale


def test_spacing_comes_from_the_larger_value() -> None:
    def distance(actual, expected):
        a = torch.tensor([actual], dtype=torch.bfloat16)
        return ulp_distance(a, torch.tensor([expected], dtype=torch.bfloat16)).item()

    assert distance(0.125, 0.12451171875) == 0.5  # 2**-11 at the spacing of [1/8, 1/4)
    assert distance(0.12255859375, 0.12353515625) == 2.0
    with pytest.raises(AssertionError, match="1 elements beyond 1 ulp"):
        assert_within_ulps(
            torch.tensor([0.12255859375], dtype=torch.bfloat16),
            torch.tensor([0.12353515625], dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_spacing_is_exact_in_every_binade(dtype) -> None:
    info = torch.finfo(dtype)
    mantissa, low = round(-math.log2(info.eps)), round(math.log2(info.tiny))
    binades = range(low, math.floor(math.log2(info.max)) + 1)
    values = torch.tensor([1.5 * 2.0**k for k in binades], dtype=torch.float64)
    spacing = torch.tensor([2.0 ** (k - mantissa) for k in binades], dtype=torch.float64)
    assert torch.equal(ulps(values, dtype), spacing)
    assert ulps(torch.zeros(1), dtype).item() == 2.0 ** (low - mantissa)  # subnormals
