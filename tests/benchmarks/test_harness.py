# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU tests of benchmark/_harness: pairing, statistics, tables and market loading."""

from __future__ import annotations

import itertools
import os

import pytest

from benchmark._harness import markets, paired, stats, table


def test_aba_compares_each_b_with_its_own_a_flanks():
    # A drifts upward every call; B is always 20% slower than the A around it.
    clock = itertools.count()

    def timer(fn):
        base = 10 + next(clock)
        return [base * fn() for _ in range(5)]

    result = paired.aba(lambda: 1.0, lambda: 1.2, timer, rounds=5)
    assert result.ratio == pytest.approx(1.2)  # drift cancels: B sits midway between its flanks
    assert result.ci[0] <= result.ratio <= result.ci[1]


def test_bootstrap_is_seeded_and_brackets_the_median():
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8]
    assert stats.bootstrap_ci(values) == stats.bootstrap_ci(values)
    lo, hi = stats.bootstrap_ci(values)
    assert lo <= 1.0 <= hi
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])


def test_parse_cores():
    assert paired.parse_cores("16-19,24") == [16, 17, 18, 19, 24]
    assert paired.parse_cores(None) is None
    assert set(paired.parse_cores("auto")) <= os.sched_getaffinity(0)


def test_table_aligns_columns():
    text = table.render("op", ["shape", "us"], [["1x4096", 1.5], ["1024x4096", 12.25]])
    lines = text.splitlines()
    assert lines[0] == "## op"
    assert len({len(line) for line in lines[2:]}) == 1
    assert table.ratio_cell(1.234, (1.1, 1.3)) == "1.23 [1.10, 1.30]"


def test_missing_market_reports_a_reason(monkeypatch):
    monkeypatch.setitem(markets.MODULES, "absent", "no_such_module_for_cti_tests")
    market = markets.load("absent")
    assert market.module is None and "no_such_module_for_cti_tests" in market.reason
    assert market.version == "n/a"
