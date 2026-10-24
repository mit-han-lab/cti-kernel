# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The shared accuracy gate (docs/testing.md), after flash-attention's tests.

An op's output is compared with an upcast reference (FP64, or FP32 where FP64 is unavailable).
The yardstick is a plain PyTorch implementation at the op's own precision, written the way
frameworks write it: its error against the same reference says how much error that precision
costs. The op passes when, for both the maximum and the mean absolute error over the finite
elements,

    error(ours) <= k * error(torch at the same precision) + atol

with ``k`` and ``atol`` stated once per op family and output dtype in ``TOLERANCES``.
Non-finite values must sit where the reference has them, with the same value (NaN, +inf,
-inf); ``nonfinite="forbid"`` instead requires every output and reference element to be finite.
The baseline must be finite wherever the reference is, and a bound that is not a finite number
fails the gate: an undefined yardstick never passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Tolerance:
    k: float
    atol: float


# family -> output dtype -> tolerance. Raise a value only with a measured reason in the change.
TOLERANCES: dict[str, dict[torch.dtype, Tolerance]] = {
    # RMS norms round once from FP32; the yardstick rounds as frameworks do (FP32 statistics,
    # then the output dtype). k=2 is flash-attention's factor; atol only covers exact zeros.
    "norm": {
        torch.float16: Tolerance(k=2.0, atol=1e-5),
        torch.bfloat16: Tolerance(k=2.0, atol=1e-5),
        # FP32 outputs differ from torch only by reduction order (a few FP32 ulps).
        torch.float32: Tolerance(k=4.0, atol=1e-6),
    },
}


@dataclass(frozen=True)
class Errors:
    max: float
    mean: float
    worst: tuple[int, ...]  # index of the largest error, () when nothing is finite


def errors(actual: torch.Tensor, reference: torch.Tensor) -> Errors:
    """Absolute errors over the elements where the reference is finite."""
    if actual.shape != reference.shape:
        raise ValueError(f"shape {tuple(actual.shape)} != reference {tuple(reference.shape)}")
    finite = torch.isfinite(reference)
    difference = (actual.double() - reference.double()).abs()
    difference = torch.where(finite, difference, torch.zeros_like(difference))
    if not finite.any():
        return Errors(0.0, 0.0, ())
    flat = int(difference.argmax())
    worst = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), difference.shape))
    count = int(finite.sum())
    return Errors(float(difference.max()), float(difference.sum()) / count, worst)


def _nonfinite_problems(actual, reference, policy) -> list[str]:
    a, r = actual.double(), reference.double()
    if policy == "forbid":
        bad = int((~torch.isfinite(a)).sum())
        return [f"{bad} non-finite output elements"] if bad else []
    problems = []
    for name, mask_a, mask_r in (
        ("NaN", torch.isnan(a), torch.isnan(r)),
        ("+inf", a == float("inf"), r == float("inf")),
        ("-inf", a == float("-inf"), r == float("-inf")),
    ):
        mismatch = int((mask_a != mask_r).sum())
        if mismatch:
            problems.append(f"{mismatch} elements disagree with the reference on {name}")
    return problems


def assert_accurate(
    actual: torch.Tensor,
    reference: torch.Tensor,
    baseline: torch.Tensor,
    *,
    family: str,
    name: str = "output",
    nonfinite: Literal["match", "forbid"] = "match",
    tolerance: Tolerance | None = None,
) -> None:
    """Gate ``actual`` against ``reference`` using ``baseline`` (torch at the op's precision).

    ``tolerance`` overrides the family entry for one call; give the reason at the call site.
    """
    if actual.dtype != baseline.dtype:
        raise ValueError(f"{name}: dtype {actual.dtype} != baseline dtype {baseline.dtype}")
    if not actual.numel():
        assert reference.numel() == 0 and baseline.numel() == 0, f"{name}: empty vs non-empty"
        return
    tol = tolerance or TOLERANCES[family][actual.dtype]
    problems = _nonfinite_problems(actual, reference, nonfinite)
    finite = torch.isfinite(reference)
    if nonfinite == "forbid" and not finite.all():
        problems.append(f"{int((~finite).sum())} non-finite reference elements")
    undefined = int((finite & ~torch.isfinite(baseline)).sum())
    if undefined:
        problems.append(f"{undefined} non-finite torch elements where the reference is finite")
    ours, torch_error = errors(actual, reference), errors(baseline, reference)
    for metric in ("max", "mean"):
        error, bound = getattr(ours, metric), tol.k * getattr(torch_error, metric) + tol.atol
        if not math.isfinite(bound):
            problems.append(f"{metric} bound {bound} is undefined")
        elif not error <= bound:  # a NaN error fails too
            problems.append(f"{metric} error {error:.3e} > bound {bound:.3e}")
    if problems:
        at = ours.worst
        detail = (
            f" worst at {at}: ours {float(actual[at]):.8g}, reference {float(reference[at]):.8g},"
            f" torch {float(baseline[at]):.8g}"
            if at
            else ""
        )
        raise AssertionError(
            f"{name} ({tuple(actual.shape)}, {actual.dtype}, family {family}, k={tol.k},"
            f" atol={tol.atol:g}): " + "; ".join(problems) + f". Errors: ours max"
            f" {ours.max:.3e} mean {ours.mean:.3e}; torch max {torch_error.max:.3e}"
            f" mean {torch_error.mean:.3e}." + detail
        )


def assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, *, name: str = "output") -> None:
    """Exact equality of bits (NaN payloads included), with the first differing index."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, (
        f"{name}: {actual.dtype}{tuple(actual.shape)} vs {expected.dtype}{tuple(expected.shape)}"
    )
    integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    a = actual.contiguous().view(integer[actual.element_size()])
    e = expected.contiguous().view(integer[expected.element_size()])
    if not torch.equal(a, e):
        flat = int((a != e).flatten().nonzero()[0])
        index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), a.shape))
        raise AssertionError(
            f"{name} differs bitwise at {index}: {float(actual[index])!r} vs"
            f" {float(expected[index])!r} ({int((a != e).sum())} elements differ)"
        )


def ulps(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of ``dtype`` at each magnitude in ``values``, exactly, in FP64 on values'
    device (subnormals included). It is built from exponent bits: pow/ldexp are not exact on
    CUDA (2**-4 came out below it, so 1-ulp differences in [1/16, 1/8) read as more)."""
    info = torch.finfo(dtype)
    magnitude = values.double().abs().clamp_min(info.tiny)
    exponent = ((magnitude.view(torch.int64) >> 52) & 0x7FF) - 1023  # floor(log2), normal FP64
    spacing = exponent - round(-math.log2(info.eps))
    return ((spacing + 1023) << 52).view(torch.float64)


def ulp_distance(
    actual: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor | None = None
) -> torch.Tensor:
    """``|actual - expected|`` in ulps of actual's dtype, exactly (FP64): the spacing at the
    larger of the two values' magnitudes, or at ``scale`` where that is larger."""
    a, e = actual.double(), expected.double()
    magnitude = torch.maximum(a.abs(), e.abs())
    if scale is not None:
        magnitude = torch.maximum(magnitude, scale.double().abs())
    return (a - e).abs() / ulps(magnitude, actual.dtype)


def assert_within_ulps(
    actual: torch.Tensor,
    expected: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    count: int = 1,
    name: str = "output",
) -> None:
    """:func:`ulp_distance` at most ``count`` on every finite element.

    ``scale`` bounds the magnitude of a result's terms where they can cancel: an affine norm's
    ``max(|weight * normalized|, |bias|)``, since an FP32 ulp of either moves a small sum by
    many of its own ulps. Non-finite elements must match exactly.
    """
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
    a, e = actual.double(), expected.double()
    finite = torch.isfinite(e)
    assert torch.equal(torch.isfinite(a), finite), f"{name}: non-finite elements differ"
    assert torch.equal(a[~finite].nan_to_num(), e[~finite].nan_to_num()), name
    distance = torch.where(finite, ulp_distance(actual, expected, scale), 0.0)
    if (distance > count).any():
        at = tuple(int(i) for i in torch.unravel_index(distance.argmax(), distance.shape))
        raise AssertionError(
            f"{name}: {int((distance > count).sum())} elements beyond {count} ulp; worst at"
            f" {at}: {float(a[at])!r} vs {float(e[at])!r}, {float(distance[at])} ulp"
        )
