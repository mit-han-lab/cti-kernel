# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent FP64 mathematical oracles for sampling tests."""

import torch


def acceptance_residual_fp64(p, q):
    """Independent mathematical oracle; not used for bitwise execution matching."""
    p, q = p.to(torch.float64), q.to(torch.float64)
    acceptance = torch.minimum(p, q).sum(-1)
    r = (p - q).clamp_min(0)
    mass = r.sum(-1, keepdim=True)
    return acceptance, torch.where(mass > 0, r / mass.clamp_min(torch.finfo(r.dtype).tiny), p)


def point_mass_acceptance_residual_fp64(p, drafts):
    """Independent point-mass law on supplied FP32 p; normalized FP64 residual.

    Acceptance probability is p(d). Correction conditions p on t!=d,
    falling back to p when that event has zero mass. No execution emulation
    or dense-q arithmetic is shared with this oracle.
    """
    p = p.to(torch.float64)
    index = drafts.to(torch.int64).unsqueeze(-1)
    acceptance = p.gather(-1, index).squeeze(-1)
    r = p.clone().scatter_(-1, index, 0)
    mass = r.sum(-1, keepdim=True)
    return acceptance, torch.where(mass > 0, r / mass.clamp_min(torch.finfo(r.dtype).tiny), p)


def probabilities_fp64(logits, temperatures, top_k, top_p, *, request_indices):
    """Independent torch FP64 sorted/cumsum oracle (no fixed-point emulation)."""
    out = torch.zeros_like(logits, dtype=torch.float64)
    for b in range(len(request_indices)):
        if request_indices[b] == -1:
            continue
        for i in range(logits.shape[1]):
            row = logits[b, i].double()
            if temperatures[b] == 0:
                out[b, i, torch.argmax(row)] = 1
                continue
            z = row / temperatures[b].double()
            order = torch.argsort(z, descending=True, stable=True)
            z[order[int(top_k[b]) :]] = -torch.inf
            probs = torch.softmax(z, -1)
            if top_p[b] < 1:
                order = torch.argsort(probs, descending=True, stable=True)
                sorted_p = probs[order]
                preceding = torch.cat((sorted_p.new_zeros(1), sorted_p.cumsum(0)[:-1]))
                probs[order[preceding >= top_p[b].double()]] = 0
                probs /= probs.sum()
            out[b, i] = probs
    return out
