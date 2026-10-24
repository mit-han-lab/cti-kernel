# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen CPU generator of the delayed-replay case inputs (r2.1).

Always generate on CPU before transfer. Generation is independent of provider.
"""

import torch

SEEDS = (11, 23, 47, 89, 131)
CLASSES = ("ordinary", "heavy", "zero_keys")
SHAPES = ((16, 32), (16, 48))


def case_inputs(hk, hv, batch, slots, steps, kind, seed, device):
    from tests.gpu import _gdn_reference as reference

    gen = torch.Generator().manual_seed(seed)
    dc = (2 * hk + hv) * 128
    width = dc + hv * 128 + 2 * hv
    low, high = (-6.0, 6.0) if kind == "heavy" else (-2.0, 2.0)
    weights = reference.Weights(
        conv_weight=(0.5 * torch.randn(dc, 4, generator=gen)).to(torch.bfloat16).to(device),
        A_log=(torch.rand(hv, generator=gen) * 2.8 - 0.7).to(device),
        dt_bias=(low + (high - low) * torch.rand(hv, generator=gen)).to(device),
        norm_weight=(1 + 0.1 * torch.randn(128, generator=gen)).to(device),
    )
    projected = torch.randn(steps, batch, width, generator=gen)
    state = 0.2 * torch.randn(slots, hv, 128, 128, generator=gen)
    conv = torch.randn(slots, 3, dc, generator=gen)
    if kind == "heavy":
        projected *= torch.exp(1.5 * torch.randn(steps, batch, 1, generator=gen))
        state *= torch.exp(2.0 * torch.randn(slots, hv, 1, 1, generator=gen))
        conv *= torch.exp(torch.randn(slots, 3, 1, generator=gen))
    if kind == "zero_keys":
        keys = slice(hk * 128, 2 * hk * 128)
        projected[:, :, keys] = 0
        conv[:, :, keys] = 0
    return (
        weights,
        projected.to(torch.bfloat16).to(device),
        state.to(device),
        conv.to(torch.bfloat16).to(device),
    )
