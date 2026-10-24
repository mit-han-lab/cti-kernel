# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent checks for the dependency-free GPU qualification intervals."""

import math

import pytest

from ._binomial import clopper_pearson


@pytest.mark.parametrize(
    "k,n,expected",
    [
        (0, 10, (0.0, 0.3084971078187608)),
        (1, 10, (0.002528578544461785, 0.4450161170281954)),
        (5, 10, (0.1870860284473985, 0.8129139715526015)),
        (10, 10, (0.6915028921812392, 1.0)),
        (0, 1, (0.0, 0.975)),
        (1, 1, (0.025, 1.0)),
    ],
)
def test_known_95_percent_intervals(k, n, expected):
    assert clopper_pearson(k, n, 0.05) == pytest.approx(expected, abs=2e-14, rel=0)


@pytest.mark.parametrize("k", [1, 2, 15, 30, 59])
def test_frozen_alpha_against_direct_binomial_sum(k):
    n, alpha = 60, 0.01 / 3948
    lower, upper = clopper_pearson(k, n, alpha)

    # Small-n oracle uses exact integer combinations and direct PMFs, with
    # neither lgamma, recurrence, early termination nor log-space summation.
    def mass(p, counts):
        return math.fsum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in counts)

    assert mass(lower, range(k, n + 1)) == pytest.approx(alpha / 2, rel=2e-12)
    assert mass(upper, range(k + 1)) == pytest.approx(alpha / 2, rel=2e-12)


def test_frozen_sample_size_edges_and_symmetry():
    n, alpha = 200_000, 0.01 / 3948
    lower, upper = clopper_pearson(1, n, alpha)
    # P[X>=1] = 1-(1-p)^n, evaluated without catastrophic cancellation.
    expected_lower = -math.expm1(math.log1p(-alpha / 2) / n)
    assert lower == pytest.approx(expected_lower, rel=2e-9, abs=1e-18)
    log_cdf = n * math.log1p(-upper) + math.log1p(n * upper / (1 - upper))
    assert math.exp(log_cdf) == pytest.approx(alpha / 2, rel=2e-9)
    for k in (0, 1, 12_500, 100_000):
        left = clopper_pearson(k, n, alpha)
        right = clopper_pearson(n - k, n, alpha)
        assert left == pytest.approx((1 - right[1], 1 - right[0]), abs=2e-12, rel=0)
