# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent top-k check for a provider's routing record (tests and benchmarks).

The reference ranks experts by FP64 router logits of the BF16 inputs (softmax is
monotone, so logit order is score order). A provider computes BF16 logits: each one
rounds to BF16 (relative error <= 2**-8) after an FP32 accumulation of K products
(error <= K * 2**-24 * sum|x*w|, any split or order). So a provider logit lies within

    delta_e = 2**-8 * |l_e| + (1 + 2**-8) * K * 2**-24 * a_e + 2**-20

of the reference ``l_e`` (``a_e = |x| @ |w_e|``; 2**-20 covers the FP32 softmax).
A row whose selected set differs from the reference top-k is a genuine boundary tie
only if every swapped pair (e chosen but outside the reference top-k, f in the
reference top-k but not chosen) satisfies ``l_f - l_e <= delta_e + delta_f``. That
implies the reference k-th and (k+1)-th scores are within the same epsilon. Any
other difference is a true mismatch.
"""

import torch


def classify_topk(x, router, ids, valid=None):
    """Return ``(ties, mismatches)`` over the first ``valid`` rows of ``ids`` (global IDs)."""
    count = x.shape[0] if valid is None else max(0, min(int(valid), x.shape[0]))
    if count == 0:
        return 0, 0
    k = ids.shape[-1]
    xd, wd = x[:count].double(), router.double()
    logits = xd @ wd.T
    bound = xd.abs() @ wd.abs().T
    delta = 2.0**-8 * logits.abs() + (1 + 2.0**-8) * x.shape[-1] * 2.0**-24 * bound + 2.0**-20
    chosen = ids[:count].long()
    if chosen.min() < 0 or chosen.max() >= router.shape[0]:
        raise AssertionError("routing record holds an out-of-range expert ID on a valid row")
    reference = logits.topk(k, dim=-1).indices
    picked = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, chosen, True)
    expected = torch.zeros_like(picked).scatter_(1, reference, True)
    if (picked.sum(-1) != k).any():
        raise AssertionError("routing record repeats an expert within a row")
    extra = picked & ~expected  # chosen, outside the reference top-k
    missing = expected & ~picked  # in the reference top-k, not chosen
    differs = extra.any(-1)
    inf = torch.tensor(float("inf"), dtype=logits.dtype, device=logits.device)
    # Worst swapped pair per row: the largest dropped f against the smallest added e.
    hi = torch.where(missing, logits - delta, -inf).amax(-1)
    lo = torch.where(extra, logits + delta, inf).amin(-1)
    tie = differs & (hi <= lo)
    return int(tie.sum()), int((differs & ~tie).sum())
