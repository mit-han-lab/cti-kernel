# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired A-B-A comparison on fixed CPU cores.

Each round times A, then B, then A again; B is compared with the mean of its two A flanks, so
slow drift (clocks, temperature, neighbours) cancels within the round. The reported ratio is
the median over rounds with a bootstrap interval; ratio < 1 means B is faster than A.
"""

from __future__ import annotations

import os
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .stats import bootstrap_ci


def pin(cores: Sequence[int] | None) -> None:
    """Pin this process (the launching thread's host work) to ``cores``; None keeps the mask."""
    if cores:
        os.sched_setaffinity(0, set(cores))


def parse_cores(text: str | None) -> list[int] | None:
    """``"16-19,24"`` -> [16, 17, 18, 19, 24]; ``"auto"`` -> the first 4 cores of the current mask."""
    if not text:
        return None
    if text == "auto":
        return sorted(os.sched_getaffinity(0))[:4]
    cores = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cores += range(int(lo), int(hi or lo) + 1)
    return cores


@dataclass(frozen=True)
class Pair:
    a_us: float  # median per-call time of A over all its samples
    b_us: float
    ratio: float  # median over rounds of B / mean(A flanks)
    ci: tuple[float, float]


def aba(
    a: Callable[[], object],
    b: Callable[[], object],
    timer: Callable[[Callable[[], object]], list[float]],
    *,
    rounds: int = 7,
) -> Pair:
    """Paired A-B-A; ``timer(fn)`` returns per-call microsecond samples (see timing.py)."""
    a_samples: list[float] = []
    b_samples: list[float] = []
    ratios = []
    for _ in range(rounds):
        first, middle, last = timer(a), timer(b), timer(a)
        a_samples += first + last
        b_samples += middle
        flank = (statistics.median(first) + statistics.median(last)) / 2
        ratios.append(statistics.median(middle) / flank)
    return Pair(
        statistics.median(a_samples),
        statistics.median(b_samples),
        statistics.median(ratios),
        bootstrap_ci(ratios),
    )
