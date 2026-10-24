# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import math

import pytest
import torch

from benchmark.kernel_tuning.norm_act_reference import TAU, no_worse
from benchmark.kernel_tuning.norm_act_tau import derive_taus, measure, measured_maxima, pair_excess


def test_gate_uses_best_comparator_independently_for_every_metric():
    markets = [
        {"max_ulp": 0.5, "rms": 0.6, "rms_ulp": 0.5},
        {"max_ulp": 0.6, "rms": 0.5, "rms_ulp": 0.6},
    ]
    assert not no_worse(dict.fromkeys(TAU, 0.6), markets)
    assert no_worse(dict.fromkeys(TAU, 0.5), markets)
    assert not no_worse(dict.fromkeys(TAU, 0.6), list(reversed(markets)))


@pytest.mark.parametrize("key", ("max_ulp", "rms", "rms_ulp"))
def test_gate_accepts_measured_noise_but_rejects_just_beyond_it(key):
    best = dict.fromkeys(TAU, 0.5)
    markets = [best, dict.fromkeys(TAU, 0.6)]
    ours = best | {key: best[key] * (1 + TAU[key] / 2)}
    assert no_worse(ours, markets)
    ours[key] = best[key] * (1 + TAU[key])
    assert no_worse(ours, markets)
    ours[key] = math.nextafter(ours[key], math.inf)
    assert not no_worse(ours, markets)


@pytest.mark.parametrize("key", ("max_ulp", "rms", "rms_ulp"))
def test_gate_absolute_floor_applies_only_to_exact_zero(key):
    zeros = dict.fromkeys(TAU, 0.0)
    assert no_worse(zeros, [zeros])
    assert no_worse(zeros | {key: 1e-12}, [zeros])
    assert not no_worse(zeros | {key: math.nextafter(1e-12, math.inf)}, [zeros])
    # Even a tiny nonzero comparator retains the relative gate.
    tiny = zeros | {key: 1e-15}
    assert not no_worse(zeros | {key: 1e-13}, [tiny])


@pytest.mark.parametrize("bad", (math.nan, math.inf))
def test_gate_nonfinite_errors_fail(bad):
    assert not no_worse(dict.fromkeys(TAU, bad), [dict.fromkeys(TAU, bad)])


def test_hardcoded_taus_cover_small_cpu_measurement():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        measured = measure(seeds=range(2), rows=(1, 3))
    finally:
        torch.set_num_threads(threads)
    # All three widths per operation, both dtypes, and six finite-encoding
    # layouts per dtype; includes the GPU test seed as well as seeds 0 and 1.
    assert len(measured) == 3 * 2 * 2 * 3 + 2 * 6
    random = [row for row in measured if row["layout"] == "random"]
    assert all(row["seed_count"] == 3 for row in random)
    assert {row["width"] for row in random} == {8, 128, 136, 4096, 12288}
    maxima = measured_maxima(measured)
    assert any(value > 0 for value in maxima.values())
    assert all(TAU[key] >= value for key, value in maxima.items()), maxima


def test_gate_covers_observed_small_norm_gpu_rounding_noise():
    # M=300, hidden=8, FP16, standard weights: equal-quality GPU paths.
    ours = {"max_ulp": 0.5, "rms": 1.666628297909275e-4, "rms_ulp": 0.2858184500575709}
    market = {"max_ulp": 0.5, "rms": 1.666625937844082e-4, "rms_ulp": 0.2858181632222676}
    assert no_worse(ours, [market])


def test_tau_is_symmetric_excess_rounded_up():
    assert pair_excess(0.5, 0.6) == pair_excess(0.6, 0.5) == pytest.approx(0.2)
    assert pair_excess(0.0, 0.0) == 0
    assert pair_excess(0.0, 2e-12) == 1
    assert derive_taus({"a": 0.003, "b": 2**-10, "c": 0.0}) == {
        "a": 2**-8,
        "b": 2**-10,
        "c": 0.0,
    }
