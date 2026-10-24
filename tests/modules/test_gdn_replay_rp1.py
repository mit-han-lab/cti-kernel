# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host-only regressions for the RP1 replay review."""

from dataclasses import replace

import pytest
import torch

from ctikernels.gdn.replay.cute import Config
from ctikernels.gdn.replay.cute._index import wide_pool_offsets

from ._replay_case import case, clone, projected, same


def alias_input(state, inputs, name, region):
    """Preserve shapes/values while sharing an allocation across the API boundary."""
    target, source = getattr(state, region), inputs[name]
    size = max(target.numel() * target.element_size(), source.numel() * source.element_size())
    backing = torch.empty(size, dtype=torch.uint8, device=source.device)
    target_alias = backing[: target.numel() * target.element_size()].view(target.dtype)
    target_alias = target_alias.reshape(target.shape)
    target_alias.copy_(target)
    source_alias = backing[: source.numel() * source.element_size()].view(source.dtype)
    inputs[name] = source_alias.reshape(source.shape)
    return replace(state, **{region: target_alias})


@pytest.mark.parametrize(
    "name",
    [
        "projected",
        "conv_weight",
        "A_log",
        "dt_bias",
        "norm_weight",
        "out",
        "positions",
        "slot_indices",
    ],
)
@pytest.mark.parametrize(
    "region", ["checkpoint", "conv_history", "metadata", "pool_epoch", "snapshot_metadata"]
)
def test_product_rejects_input_state_alias_before_launch(monkeypatch, name, region):
    op, state, _, weights, slots, pos, _ = case(batch=1)
    op.cute_config = Config(provider="product", immediate=True)
    op._step_config = op.cute_config
    op._ringless = True
    op.checks = "trusted"
    state = replace(state, ring=state.ring[:, :0].contiguous())
    inputs = dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True))
    inputs.update(
        projected=projected(op, 1),
        out=torch.empty(1, op.geometry.value_width, dtype=torch.bfloat16),
        positions=pos,
        slot_indices=slots,
    )
    state = alias_input(state, inputs, name, region)
    before = clone(state)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("reached CUDA before alias guard"),
    )
    with pytest.raises(ValueError, match="overlapping buffers"):
        op.step(**inputs, state=state)
    same(state, before)


@pytest.mark.parametrize("hk,hv", [(1, 1), (16, 32), (16, 48)])
@pytest.mark.parametrize("window", [4, 8, 16])
@pytest.mark.parametrize("pool", ["checkpoint", "ring"])
def test_pool_index_promotion_at_large_offsets(hk, hv, window, pool):
    width = (2 * hk + hv) * 128 + 2 * hv
    strides = {"checkpoint": hv * 16384, "ring": (window + 3) * width}
    stride = strides[pool]
    high = (2**31 + stride - 1) // stride
    assert high * stride >= 2**31
    assert wide_pool_offsets(high + 1, hv, window, width)
    slot = torch.tensor(high, dtype=torch.int32)
    if wide_pool_offsets(high + 1, hv, window, width):
        slot = slot.to(torch.int64)
    checkpoint_last = (slot * hv + hv - 1) * 16384 + 16383
    ring_last = (slot * (window + 3) + window + 2) * width + width - 1
    assert checkpoint_last.item() == (high + 1) * strides["checkpoint"] - 1
    assert ring_last.item() == (high + 1) * strides["ring"] - 1
    bound = 2**31 // max(strides.values())
    assert not wide_pool_offsets(bound - 1, hv, window, width)
    assert wide_pool_offsets(bound + 1, hv, window, width)
    # Step casts pool_slot to Int64 BEFORE (slot * hv + head) * 16384 and
    # (slot * (L+3) + row) * W. K2's source() casts both slots to Int64 before
    # raw_value/raw_vector/publication; import's Triton wrapper does likewise.
    # Even the smallest legal full pool crossing the ring's 2**31 boundary
    # exceeds 12 GB: at Hk=Hv=1,L=16 it needs >19 GB of checkpoint alone.
    # Therefore ring-boundary device allocation cannot meet the test budget.
    if pool == "ring":
        assert (high + 1) * hv * 16384 * 4 > 12_000_000_000
