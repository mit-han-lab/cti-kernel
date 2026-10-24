# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Qualification rule of the paired MoE market benchmark: both baselines are required."""

import pytest

from benchmark.kernel_tuning.moe_market_paired import qualify

PATHS = ["ours", "torch_gmm", "unsloth"]


def _timing(median, low):
    return {"median_us": median, "ci95": [low, low + 0.2], "median_ratio": low + 0.1}


def _row(torch_gmm=None, unsloth=None):
    row = {"sample_duration_ok": True, "ours": {"timings": {"graph": {"median_us": 50.0}}}}
    row["torch_gmm"] = (
        torch_gmm if torch_gmm is not None else {"timings": {"eager": _timing(300.0, 5.0)}}
    )
    row["unsloth"] = (
        unsloth
        if unsloth is not None
        else {"timings": {"eager": _timing(80.0, 1.3), "graph": _timing(70.0, 1.2)}}
    )
    return row


def test_both_baselines_pass():
    row = _row()
    qualify(row, ("ours", "graph"), PATHS)
    assert row["passed"] and row["market_best"] == "unsloth"
    assert row["unsloth"]["best_mode"] == "graph" and "baseline_missing" not in row


def test_slower_than_fastest_market_fails():
    row = _row(unsloth={"timings": {"graph": _timing(45.0, 0.9)}})
    qualify(row, ("ours", "graph"), PATHS)
    assert not row["passed"] and row["market_best"] == "unsloth"


@pytest.mark.parametrize(
    "broken,reason",
    (
        ({"error": "ImportError: no unsloth"}, "ImportError: no unsloth"),
        ({"correct": False}, "failed correctness"),
        ({"correct": True, "eager_timing_error": "RuntimeError: boom"}, "eager_timing_error"),
        ({"correct": True, "timings": {"eager": {"median_us": 90.0}}}, "no complete paired"),
    ),
)
@pytest.mark.parametrize("which", ("torch_gmm", "unsloth"))
def test_missing_baseline_fails(broken, reason, which):
    row = _row(**{which: broken})
    qualify(row, ("ours", "graph"), PATHS)
    assert not row["passed"] and "market_best" not in row
    (missing,) = row["baseline_missing"]
    assert missing.startswith(which) and reason in missing


def test_unrequested_baseline_fails():
    row = _row()
    del row["torch_gmm"]
    qualify(row, ("ours", "graph"), ["ours", "unsloth"])
    assert not row["passed"] and row["baseline_missing"] == ["torch_gmm: not requested"]
