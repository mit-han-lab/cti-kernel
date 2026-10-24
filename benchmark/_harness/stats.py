# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Medians and seeded bootstrap confidence intervals (no NumPy)."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = statistics.median,
    *,
    level: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``statistic``; deterministic for a given seed."""
    if not values:
        raise ValueError("bootstrap needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    tail = (1 - level) / 2
    return estimates[int(tail * (resamples - 1))], estimates[int((1 - tail) * (resamples - 1))]
