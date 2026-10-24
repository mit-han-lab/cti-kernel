# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Float64 equal-tailed Clopper–Pearson intervals, using only the standard library."""

import math
from functools import lru_cache


def _bound(k, n, log_alpha, upper):
    # Upper solves P_p[X <= k] = alpha/2 above k/n; lower solves
    # P_p[X >= k] = alpha/2 below k/n. Sum the small tail directly,
    # avoiding cancellation from subtracting a CDF close to one.
    lo, hi = (k / n, 1.0) if upper else (0.0, k / n)
    log_choose = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    for _ in range(64):
        p = (lo + hi) / 2
        if p == lo or p == hi:
            break
        # The boundary PMF is the largest term in this tail. Scale by it
        # before summing (log-sum-exp), then restore its log probability.
        total = term = 1.0
        j = k
        while (j > 0) if upper else (j < n):
            ratio = (
                (j / (n - j + 1)) * ((1 - p) / p)
                if upper
                else (((n - j) / (j + 1)) * (p / (1 - p)))
            )
            # Ratios decrease as we move away from k. A geometric series
            # bounds the entire omitted tail, not just its next term.
            if ratio < 1 and term * ratio <= 1e-16 * total * (1 - ratio):
                break
            term *= ratio
            total += term
            j += -1 if upper else 1
        log_tail = log_choose + k * math.log(p) + (n - k) * math.log1p(-p) + math.log(total)
        if (log_tail > log_alpha) == upper:
            lo = p
        else:
            hi = p
    return hi if upper else lo


@lru_cache(maxsize=4096)
def clopper_pearson(k: int, n: int, alpha: float) -> tuple[float, float]:
    """Return the exact-binomial interval at confidence 1-alpha (up to FP64 error).

    No normal approximation, external statistics package, or changed test alpha.
    Closed forms cover zero/all successes, including n=1.
    """
    if n <= 0 or not 0 <= k <= n or not 0 < alpha < 1:
        raise ValueError("require n>0, 0<=k<=n and 0<alpha<1")
    log_alpha = math.log(alpha / 2)
    lower = (
        0.0 if k == 0 else (math.exp(log_alpha / n) if k == n else _bound(k, n, log_alpha, False))
    )
    upper = (
        1.0 if k == n else (-math.expm1(log_alpha / n) if k == 0 else _bound(k, n, log_alpha, True))
    )
    return lower, upper
