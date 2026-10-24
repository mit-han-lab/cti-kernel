# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Independent CPU DN4 oracles: semantics follow vLLM 0.31.0 V2; no code copied.

Int64 bincount over actual context slices, ignoring IDs outside [0,V).
This test-only oracle intentionally reads values on the host; it is not capturable.
"""

import torch

UNIT_ROUNDOFF = 2.0**-24
# ((1+u)**4 - 1)/u, expanded to avoid cancellation.
ERROR_COEFFICIENT = 4 + 6 * UNIT_ROUNDOFF + 4 * UNIT_ROUNDOFF**2 + UNIT_ROUNDOFF**3


def counts_for_request(token_table, rows, n, vocab, drafts=None):
    """Return independent int64 prompt/output counts on CPU."""
    table = token_table.cpu()
    slot, p, c = int(rows.rows[n]), int(rows.prompt_lengths[n]), int(rows.lengths[n])
    prompt = table[slot, :p].long()
    output = table[slot, p:c].long()
    if drafts is not None:
        output = torch.cat((output, drafts.cpu().long()))
    return tuple(
        torch.bincount(ids[(ids >= 0) & (ids < vocab)], minlength=vocab) for ids in (prompt, output)
    )


def reference_penalties(logits, token_table, rows, lookahead=None, *, fp64=False):
    """FP32 separate operations, or FP64 true division on identical counts.

    FP64 mode also returns the magnitude sum |x_rep|+|frequency*count|+|presence|
    used by the documented normal-arithmetic error bound.
    """
    source = logits.cpu().float()
    values = source[:, None] if source.ndim == 2 else source
    dtype = torch.float64 if fp64 else torch.float32
    result = torch.empty_like(values, dtype=dtype)
    magnitudes = torch.empty_like(result) if fp64 else None
    for n in range(values.shape[0]):
        presence, frequency, repetition = (
            getattr(rows, name)[n].cpu().to(dtype)
            for name in ("presence", "frequency", "repetition")
        )
        for j in range(values.shape[1]):
            prompt, output = counts_for_request(
                token_table,
                rows,
                n,
                values.shape[-1],
                None if lookahead is None else lookahead[n, :j],
            )
            x = values[n, j].to(dtype)
            scale = torch.where((prompt > 0) | (output > 0), repetition, 1.0)
            if fp64:
                repeated = torch.where(x > 0, x / scale, x * scale)
            else:
                reciprocal = torch.ones_like(scale) / scale
                repeated = x * torch.where(x > 0, reciprocal, scale)
            product = frequency * output.to(dtype)
            value = repeated - product
            adjustment = presence * (output > 0).to(dtype)
            value = value - adjustment
            if presence == 0 and frequency == 0 and repetition == 1:
                value = x.clone()
            result[n, j] = value
            if fp64:
                magnitudes[n, j] = repeated.abs() + product.abs() + presence.abs()
    if source.ndim == 2:
        result = result[:, 0]
        if fp64:
            magnitudes = magnitudes[:, 0]
    return (result, magnitudes) if fp64 else result
