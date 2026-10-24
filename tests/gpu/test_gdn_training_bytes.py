# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GatedDeltaNetTraining.training_bytes against each provider's measured memory: the saved
payload and the backward's requested-bytes peak, at the engine's (T, N) points."""

from __future__ import annotations

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetTraining, _cute, training

# The engine's planning points, (tokens, sequences).
POINTS = [(1024, 1), (8192, 1), (8192, 8)]
GEOMETRIES = [(16, 32), (4, 12)]


def _op(provider: str, k_heads: int, v_heads: int, monkeypatch) -> GatedDeltaNetTraining:
    with monkeypatch.context() as patch:
        if provider != "triton":
            patch.setattr(training, "_triton", lambda device: None)
        if provider == "cute":
            patch.setattr(training, "_cute_geometry", lambda geometry, device: True)
        else:
            patch.setattr(_cute, "training", lambda device: None)
        op = GatedDeltaNetTraining(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    if op.provider != provider:
        pytest.skip(f"the {provider} provider does not resolve here ({op.provider} does)")
    return op


def _measure(op: GatedDeltaNetTraining, tokens: int, sequences: int) -> tuple[int, int]:
    """(saved, backward): the forward's saved storages without the weights, and the
    backward's requested-bytes peak above its start."""
    geometry = op.geometry
    g = torch.Generator(device="cuda").manual_seed(0)
    projected = torch.randn(tokens, geometry.proj_width, device="cuda", generator=g).bfloat16()
    weights = (
        (0.5 * torch.randn(geometry.conv_dim, 4, device="cuda", generator=g)).bfloat16(),
        torch.rand(geometry.v_heads, device="cuda", generator=g) * 2.8 - 0.7,
        torch.rand(geometry.v_heads, device="cuda", generator=g) * 4 - 2,
        1 + 0.1 * torch.randn(128, device="cuda", generator=g),
    )
    lengths = [tokens // sequences] * sequences
    lengths[-1] += tokens - sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    y, saved = op._kernel.train_forward(geometry, projected, *weights, cu.cuda())
    skip = {w.untyped_storage().data_ptr() for w in weights}
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in saved}
    payload = sum(n for ptr, n in storages.items() if ptr not in skip)
    dy = torch.randn_like(y)
    torch.cuda.synchronize()
    start = torch.cuda.memory_stats()["requested_bytes.all.current"]
    torch.cuda.reset_peak_memory_stats()
    op._kernel.train_backward(geometry, saved, dy)
    torch.cuda.synchronize()
    return payload, torch.cuda.memory_stats()["requested_bytes.all.peak"] - start


@pytest.mark.parametrize("provider", ["triton", "cute", "fla-core"])
@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_training_bytes_bounds_the_measured_memory_within_five_percent(
    provider, k_heads, v_heads, monkeypatch
):
    op = _op(provider, k_heads, v_heads, monkeypatch)
    for tokens, sequences in POINTS:
        reported = op.training_bytes(tokens, sequences)
        measured = _measure(op, tokens, sequences)
        for name, r, m in zip(("saved", "backward"), reported, measured, strict=True):
            assert m <= r <= 1.05 * m, (provider, tokens, sequences, name, r, m)
