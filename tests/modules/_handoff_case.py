# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shared deterministic handoff fixtures without GPU-only imports."""

import torch

from ctikernels.gdn import GatedDeltaNetReplay

from ._replay_case import clone, graph_buffers, same


def setup(
    *, device="cpu", hk=1, hv=4, window=4, layers=4, inplace=True, batch=4, snapshots=2,
    provider=None,
):  # fmt: skip
    """``provider`` defaults to the device's portable one (Triton, or the CPU reference)."""
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        window=window,
        device=device,
        provider=provider or ("triton" if device == "cuda" else "torch_reference"),
        checks="strict",
    )
    generator = torch.Generator().manual_seed(983)
    states = []
    for _ in range(layers):
        buffers = {}
        for name, spec in op.state_spec(num_slots=batch, num_snapshot_rows=snapshots).items():
            if spec.dtype in (torch.float32, torch.bfloat16):
                value = torch.randn(spec.shape, generator=generator).to(spec.dtype)
            else:
                value = torch.randint(1, 17, spec.shape, generator=generator, dtype=spec.dtype)
            buffers[name] = value.to(device)
        state = op.create_state(buffers=buffers)
        state.metadata[:, 4] = torch.arange(batch, device=device) % 2
        state.metadata[::2, 7] = state.pool_epoch[0]
        states.append(state)
    copies = [clone(state) for state in states]
    canonical = [s.checkpoint[:batch] if inplace else s.checkpoint[:batch].clone() for s in states]
    histories = [
        s.conv_history[:batch] if inplace else s.conv_history[:batch].clone() for s in states
    ]
    other_canonical = (
        [s.checkpoint[:batch] for s in copies] if inplace else [x.clone() for x in canonical]
    )
    other_histories = (
        [s.conv_history[:batch] for s in copies] if inplace else [x.clone() for x in histories]
    )
    table = torch.empty((layers, 8), dtype=torch.int64, device=device)
    binding = op.bind_layers(
        states=copies, table=table, canonical_states=other_canonical, histories=other_histories
    )
    statuses = [graph_buffers(op, batch=batch, layers=layers).status(1) for _ in range(2)]
    for status in statuses:
        status.sequence.fill_(17)
        op.begin(status=status)
        status.validity.fill_(True)  # Handoff must preserve even preexisting validity.
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    if not inplace:
        slots = slots.flip(0)
    slots[-1] = -1
    positions = torch.arange(batch, dtype=torch.int64, device=device) + 2**34
    return op, states, binding, canonical, histories, statuses, slots, positions


def per_layer(op, states, canonical, histories, status, slots, positions):
    for j, state in enumerate(states):
        op.handoff(
            canonical_state=canonical[j],
            history=histories[j],
            next_positions=positions,
            slot_indices=slots,
            state=state,
            status=status.for_layer(j),
        )


def compare(states, binding, statuses=None):
    for left, right in zip(states, binding.states, strict=True):
        same(left, right)
    if statuses is not None:
        for name in ("records", "validity", "sequence", "poison"):
            assert torch.equal(getattr(statuses[0], name), getattr(statuses[1], name)), name


def error_case(kind, statuses, slots, positions, layer=1):
    if kind == "position":
        positions[0] = -1
    elif kind == "slot":
        slots[0] = slots.numel()
    elif kind == "negative_slot":
        slots[0] = -2
    elif kind == "mapping":
        # Row 0 takes row 1's slot (slot != row, still unique destinations).
        slots[0] = slots[1].clone()
        slots[1] = -1
    elif kind == "padding":
        positions[-1] = -1
    for status in statuses:
        if kind == "poison":
            status.poison.fill_(8)
        elif kind == "sequence":
            status.records[layer, :8].view(torch.int64).fill_(19)
        elif kind == "done":
            status.records[layer, 12:].view(torch.int32).fill_(1)
        elif kind == "code":
            status.records[layer, 8:12].view(torch.int32).fill_(4)
        elif kind == "zero_sequence":
            status.sequence.zero_()


def snapshot(states):
    return [clone(state) for state in states]


def assert_layer_prefix(before, binding, failed):
    """Layers before ``failed`` were handed off; ``failed`` and later are untouched."""
    for j, (old, state) in enumerate(zip(before, binding.states, strict=True)):
        if j < failed:
            assert not torch.equal(old.metadata, state.metadata), j
        else:
            same(old, state)


def assert_layer_status(statuses, error, failed):
    """Exact codes and poison after a failure injected at record ``failed``.

    A guard failure (sequence/done) poisons with 64 and leaves its own code; a
    preset code 4 propagates itself. Every later layer records 32 (dependency)."""
    for status in statuses:
        codes = status.records[:, 8:12].contiguous().view(torch.int32).flatten().tolist()
        own = 4 if error == "code" else 0
        later = len(codes) - failed - 1
        assert codes == [0] * failed + [own] + [32] * later, codes
        expected = (4 if error == "code" else 64) | (32 if later else 0)
        assert int(status.poison[0]) == expected


def mixed_setup(*, device="cpu", inplace=True):
    """Alternate full/ringless allocations under an immediate op that accepts both."""
    from dataclasses import replace

    from ctikernels.gdn.replay.cute import Config

    _, states, old, canonical, histories, statuses, slots, positions = setup(
        device=device, hk=16, hv=32, inplace=inplace
    )
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        window=4,
        device=device,
        provider="cute" if device == "cuda" else "torch_reference",
        checks="strict",
        mode="immediate" if device == "cuda" else "auto",
        cute_config=Config(immediate=True) if device == "cuda" else None,
    )
    # CPU provider has no immediate policy; exercise the same binding shape contract.
    if device == "cpu":
        op._ringless = True
    copies = list(old.states)
    for j in (0, 2):
        states[j] = replace(states[j], ring=states[j].ring[:, :0].contiguous())
        copies[j] = replace(copies[j], ring=copies[j].ring[:, :0].contiguous())
    binding = op.bind_layers(
        states=copies,
        table=old.table,
        canonical_states=old.canonical_states,
        histories=old.histories,
    )
    return op, states, binding, canonical, histories, statuses, slots, positions
