# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU check of layout benchmark pairing and ratio direction."""

import pytest

from benchmark.kernel_tuning.gdn_replay_layout_cost import measure_pairs


def test_layout_cost_interleaves_pairs_and_reports_pitched_over_dense():
    order = []

    def sample(name, value):
        def run():
            order.append(name)
            return value

        return run

    result = measure_pairs(sample("dense", 10), sample("pitched", 12), sample("self", 11))
    assert order == ["dense", "pitched", "dense", "self", "pitched", "dense", "self", "dense"] * 10
    assert result == pytest.approx({"dense_us": 10, "pitched_us": 12, "ratio": 1.2, "eps": 0.1})
