# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""No worse than unmodified fla-core against FP64.

Per case, the relative RMS is within 5 % of fla-core's (da 15 %) and the largest error within
25 % (da and the chunked backward's dk, dv, dbeta, dg 50 %). Over a test's cases, the geometric
means of both ratios are within 5 %. The bounds are fixed: fla-core's own error moves with its
timing-autotuned launch configs, so a bound derived from the measured spread is not reproducible.
"""

from __future__ import annotations

import math
import statistics

import torch

RMS_VERSUS_FLA = 1.05
DA_RMS_VERSUS_FLA = 1.15
RMS_FLOOR = 1e-6
PEAK_VERSUS_FLA = 1.25
GRAD_PEAK_VERSUS_FLA = 1.5
MEAN_VERSUS_FLA = 1.05
PEAK_FLOOR = 2.0**-24


def errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Relative RMS and largest error, each relative to the reference's (absolute where the
    reference is all zero)."""
    actual, expected = actual.double(), expected.double()
    error = actual - expected
    rms, peak = expected.pow(2).mean().sqrt().item(), expected.abs().max().item()
    return (
        error.pow(2).mean().sqrt().item() / (rms or 1.0),
        error.abs().max().item() / (peak or 1.0),
    )


def output_ulp(expected: torch.Tensor) -> float:
    """One BF16 ulp at the reference's largest magnitude, relative to it (the largest error's
    unit): a BF16 output can differ from fla-core's by one ulp with neither kernel at fault."""
    peak = expected.double().abs().max().item()
    return 2.0 ** (math.floor(math.log2(peak)) - 7) / peak if peak else PEAK_FLOOR


def no_worse(
    ours,
    flas,
    expected,
    label,
    *,
    rms_versus_fla=RMS_VERSUS_FLA,
    peak_versus_fla=PEAK_VERSUS_FLA,
    peak_floor=PEAK_FLOOR,
) -> tuple[float, float]:
    """Asserts one case's bounds, the largest error within ``peak_versus_fla`` times
    fla-core's plus ``peak_floor``; returns the RMS and largest-error ratios (ours / fla-core's)."""
    assert torch.isfinite(ours.double()).all(), label
    (rms, peak), (flas_rms, flas_peak) = errors(ours, expected), errors(flas, expected)
    found = {"rms": rms, "peak": peak, "fla_rms": flas_rms, "fla_peak": flas_peak}
    if expected.any():  # an all-zero reference is compared by its largest error only
        assert rms <= rms_versus_fla * flas_rms + RMS_FLOOR, (label, found)
    assert peak <= peak_versus_fla * flas_peak + peak_floor, (label, found)
    return (rms + RMS_FLOOR) / (flas_rms + RMS_FLOOR), (peak + PEAK_FLOOR) / (
        flas_peak + PEAK_FLOOR
    )


def no_worse_section(name, ours, flas, expected, label) -> tuple[float, float]:
    """``no_worse`` for a section of d projected: da takes the gate gradient's bounds."""
    if name != "da":
        return no_worse(ours, flas, expected, label)
    return no_worse(
        ours,
        flas,
        expected,
        label,
        rms_versus_fla=DA_RMS_VERSUS_FLA,
        peak_versus_fla=GRAD_PEAK_VERSUS_FLA,
    )


def assert_means(ratios: dict[str, list[tuple[float, float]]]) -> None:
    for name, values in ratios.items():
        for k, norm in enumerate(("rms", "peak")):
            mean = statistics.geometric_mean([v[k] for v in values])
            assert mean <= MEAN_VERSUS_FLA, (name, norm, values)
