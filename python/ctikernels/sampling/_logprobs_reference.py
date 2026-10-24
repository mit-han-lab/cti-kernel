# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Eager CPU FP64 oracle for DN3 qualification; never used by the provider."""

import math

import torch

from ._dflash.reference.ops import fixed_mass, probability_row

# Qualified for |input logits| <= 1e4, V <= 248320 and T in {0, .3, 1, 2.5}.
ERROR_ATOL = 8e-6
ERROR_RTOL = 5e-7


def reference_row(
    logits,
    *,
    mode="raw",
    values="logprobs",
    floor=None,
    probabilities=None,
    temperature=1.0,
    filter_top_k=None,
    top_p=1.0,
    min_p=0.0,
):
    """Return FP64 output values, exact ordered IDs, and values used for ranks.

    Both processed sources order by FP32 mass, then source-B processed logprob,
    then lower ID. Logprobs are computed independently in FP64; ties are defined
    after rounding them to FP32, so sub-FP32 differences do not break ties.
    Provider-specific FP32 intermediate rounding can differ at rounding edges.
    Rank values are dense codes for (mass, rounded logprob): equal pairs share
    a code, and count(values >= values[selected]) includes all such ties.

    Support and masses use the independent S1 CPU reference. Retained finite
    log-space tails survive FP32 exp underflow without gaining sampling mass.
    """
    if values not in ("logprobs", "logits"):
        raise ValueError("values must be logprobs or logits")
    if floor is not None and (not isinstance(floor, float) or not math.isfinite(floor)):
        raise ValueError("floor must be a finite Python float or None")
    x = logits.detach().cpu().float()
    vocab = x.numel()

    def output(lp):
        if values == "logits":
            t = float(torch.tensor(temperature, dtype=torch.float32))
            scaled = x.double() / (t if mode == "processed" and t != 0 else 1)
            lp = scaled if mode == "raw" else torch.where(torch.isfinite(lp), scaled, -torch.inf)
        return lp if floor is None else torch.where(torch.isneginf(lp), floor, lp)

    supplied = None if probabilities is None else probabilities.detach().cpu().float()
    if torch.isneginf(x).all() or (
        mode == "processed" and supplied is not None and (supplied == 0).all()
    ):
        # Empty support overrides even greedy processing and all tie keys.
        result = torch.full((vocab,), -torch.inf if floor is None else floor, dtype=torch.float64)
        return result, torch.arange(vocab), torch.zeros(vocab, dtype=torch.int64)
    if mode == "raw":
        return output(x.double().log_softmax(-1)), torch.argsort(x, descending=True, stable=True), x
    k = vocab if filter_top_k is None else filter_top_k
    raw_order = torch.argsort(x, descending=True, stable=True)
    p = probability_row(x, temperature, k, top_p)
    if temperature == 0:
        keep = torch.arange(vocab) == raw_order[0]
        z = torch.where(keep, 0.0, -torch.inf).double()
    else:
        keep = torch.zeros(vocab, dtype=torch.bool)
        keep[raw_order[:k]] = True
        if top_p != 1:
            before_top_p = probability_row(x, temperature, k, 1.0)
            order = torch.argsort(before_top_p, descending=True, stable=True)
            threshold, preceding = fixed_mass(top_p, ceiling=True), 0
            for token in order.tolist():
                if preceding >= threshold:
                    keep[token] = False
                preceding += fixed_mass(before_top_p[token])
        keep &= p >= torch.tensor(min_p, dtype=torch.float32) * p.max()
        t = float(torch.tensor(temperature, dtype=torch.float32))
        z = torch.where(keep, (x.double() - x.max().double()) / t, -torch.inf)
    lp = z.log_softmax(-1)
    tie_lp = lp.float()
    if probabilities is not None:
        p = supplied
    p = torch.where(p >= torch.tensor(min_p, dtype=torch.float32) * p.max(), p, 0)
    if probabilities is not None:
        lp = p.double().log() - p.double().sum().log()
    order = torch.argsort(tie_lp, descending=True, stable=True)
    order = order[torch.argsort(p[order], descending=True, stable=True)]
    starts = torch.ones(vocab, dtype=torch.bool)
    starts[1:] = (p[order[1:]] != p[order[:-1]]) | (tie_lp[order[1:]] != tie_lp[order[:-1]])
    scores = torch.empty(vocab, dtype=torch.int64)
    scores[order] = -starts.cumsum(0)
    return output(lp), order, scores
