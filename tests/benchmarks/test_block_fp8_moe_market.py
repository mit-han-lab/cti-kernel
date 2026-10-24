# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only checks of the market benchmark's paired speedup bootstrap."""

import random
import statistics

import pytest

from benchmark.kernel_tuning.block_fp8_moe_market import speedup_statistics


def stats(oa, ob, sa, sb, **kwargs):
    return speedup_statistics(
        oa, ob, sa, sb, **({"seed": 11, "m": 4, "ep": 2, "side": "sglang"} | kwargs)
    )


@pytest.mark.parametrize("speedup,rule", [(2.0, "pass"), (0.5, "fail"), (1.0, "inconclusive")])
def test_bootstrap_known_verdict(speedup, rule):
    ours = list(range(1, 61))
    side = [t * speedup for t in ours]
    result = stats(ours, ours, side, side)
    assert result == {
        "speedup_vs_sglang": speedup,
        "speedup_ci95_vs_sglang": [speedup, speedup],
        "gb10_rule_vs_sglang": rule,
        "rule_pairs": 60,
    }


def test_bootstrap_is_deterministic_and_does_not_use_global_rng():
    ours = [1, 2, 3, 5, 8, 13]
    side = [2, 3, 5, 7, 11, 17]
    state = random.getstate()
    first = stats(ours, side, side, ours, bootstrap=137)
    assert first == stats(ours, side, side, ours, bootstrap=137)
    assert random.getstate() == state
    lo, hi = first["speedup_ci95_vs_sglang"]
    assert lo < 1 < hi
    assert first["gb10_rule_vs_sglang"] == "inconclusive"


def test_speedups_pair_rounds_and_include_b_graphs():
    # Ratios of medians, crossed rounds, or A-only samples give different results.
    oa, ob, sa, sb = [1, 100], [2, 200], [2, 300], [8, 1000]
    result = stats(oa, ob, sa, sb, bootstrap=1000)
    assert result["speedup_vs_sglang"] == statistics.median([2, 3, 4, 5])
    assert result["speedup_ci95_vs_sglang"] == [2, 5]
    assert result["rule_pairs"] == 2  # Count rounds, not the four graph-pair samples.


@pytest.mark.parametrize("speedups", [[1, 2], [0.5, 1]])
def test_ci_touching_one_is_inconclusive(speedups):
    result = stats([1, 1], [1, 1], speedups, speedups, bootstrap=1000)
    assert 1 in result["speedup_ci95_vs_sglang"]
    assert result["gb10_rule_vs_sglang"] == "inconclusive"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_durations(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        stats([1], [1], [bad], [1])


def test_invalid_bootstrap_or_pairs():
    with pytest.raises(ValueError, match="bootstrap must be positive"):
        stats([1], [1], [1], [1], bootstrap=0)
    with pytest.raises(ValueError, match="at least one"):
        stats([], [], [], [])
    with pytest.raises(ValueError, match="zip"):
        stats([1, 2], [1], [1], [1])
