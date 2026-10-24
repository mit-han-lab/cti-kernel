# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Binding validation and portable all-layer handoff semantics."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay, ReplayLayers

from ._handoff_case import (
    assert_layer_prefix,
    assert_layer_status,
    compare,
    error_case,
    per_layer,
    setup,
    snapshot,
)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        None,
        "position",
        "slot",
        "negative_slot",
        "mapping",
        "padding",
        "poison",
        "sequence",
        "done",
        "code",
        "zero_sequence",
    ],
)
def test_cpu_handoff_layers_equivalence(inplace, error):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(inplace=inplace)
    assert isinstance(binding, ReplayLayers)
    assert op.capabilities["phase_providers"]["handoff_layers"] == "torch_reference"
    error_case(error, statuses, slots, positions)
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    compare(states, binding, statuses)


def test_layer_table_spec_and_defaults():
    op, _, binding, _, _, _, _, _ = setup(snapshots=0)
    spec = op.layer_table_spec(num_layers=4)["layer_table"]
    assert (spec.shape, spec.dtype, spec.payload_bytes, spec.aligned_bytes) == (
        (4, 8),
        torch.int64,
        256,
        256,
    )
    default = op.bind_layers(states=binding.states, table=torch.empty_like(binding.table))
    assert default.batch_size == default.num_slots == 4
    for j, state in enumerate(default.states):
        assert default.canonical_states[j] is state.checkpoint
        assert default.histories[j] is state.conv_history
        assert default.table[j].tolist() == [
            state.checkpoint.data_ptr(),
            state.conv_history.data_ptr(),
            state.ring.data_ptr(),
            state.metadata.data_ptr(),
            state.pool_epoch.data_ptr(),
            state.checkpoint.data_ptr(),
            state.conv_history.data_ptr(),
            1,
        ]
    with pytest.raises(FrozenInstanceError):
        default.batch_size = 2
    for n in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            op.layer_table_spec(num_layers=n)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_state",
        "cross_layer",
        "partial",
        "table",
        "input_overlap",
        "shape",
        "batch",
        "history_shape",
        "slots",
        "snapshots",
        "alignment",
        "noncontiguous",
    ],
)
def test_bind_rejects_bad_regions(kind):
    op, _, binding, _, _, _, _, _ = setup(inplace=False)
    states, canonical, histories = (
        list(binding.states),
        list(binding.canonical_states),
        list(binding.histories),
    )
    table = torch.empty_like(binding.table)
    if kind == "duplicate_state":
        states[1] = states[0]
    elif kind == "cross_layer":
        canonical[1] = states[0].checkpoint[:4]
    elif kind == "partial":
        canonical[0] = states[0].checkpoint[1:5]
    elif kind == "table":
        table = states[0].metadata
    elif kind == "input_overlap":
        canonical[1] = canonical[0]
    elif kind == "shape":
        canonical[0] = canonical[0][:, :, :, :127]
    elif kind == "batch":
        canonical[1] = canonical[1][:3]
    elif kind == "history_shape":
        histories[1] = histories[1][:, :2]
    elif kind == "slots":
        states[1] = replace(states[1], metadata=states[1].metadata[:3])
    elif kind == "snapshots":
        states[1] = replace(states[1], snapshot_metadata=states[1].snapshot_metadata[:1])
    elif kind == "alignment":
        canonical[0] = torch.empty(canonical[0].numel() + 1)[1:].view_as(canonical[0])
    else:
        canonical[0] = canonical[0].transpose(2, 3)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        op.bind_layers(states=states, table=table, canonical_states=canonical, histories=histories)


def test_foreign_binding_and_dynamic_controls():
    op, _, binding, _, _, statuses, slots, positions = setup()
    foreign = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, window=4, device="cpu")
    with pytest.raises(ValueError, match="foreign"):
        foreign.handoff_layers(
            binding, next_positions=positions, slot_indices=slots, status=statuses[0]
        )
    with pytest.raises(ValueError, match="status"):
        op.handoff_layers(binding, next_positions=positions, slot_indices=slots)
    for bad_slots, bad_positions in (
        (slots.short(), positions),
        (slots[:2], positions),
        (slots, positions.float()),
    ):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            op.handoff_layers(
                binding, next_positions=bad_positions, slot_indices=bad_slots, status=statuses[0]
            )
    status = replace(
        statuses[0], records=statuses[0].records[:1], validity=statuses[0].validity[:1]
    )
    with pytest.raises(ValueError, match="record"):
        op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=status)


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("provider", ["triton", "cute"])
def test_handoff_layers_follows_the_op_provider(monkeypatch, available, provider):
    """CuTe copies layers only for a CuTe op (its qualified interval); Triton ops stay Triton."""
    from ctikernels.gdn.replay import _api
    from ctikernels.gdn.replay.cute import Config

    monkeypatch.setattr(_api, "_cute_handoff_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA H100 80GB HBM3")
    config = Config() if provider == "cute" else None
    op = GatedDeltaNetReplay(
        num_k_heads=1, num_v_heads=4, device="cuda:0", provider=provider, cute_config=config
    )
    assert op.provider["handoff_layers"] == (
        "cute" if provider == "cute" and available else "triton"
    )


def test_fused_host_path_does_not_revalidate_layers(monkeypatch):
    from ctikernels.gdn.replay.cute import _handoff_launch

    op, _, binding, _, _, statuses, slots, positions = setup()
    op.provider = {**op.provider, "handoff_layers": "cute"}
    calls = []
    monkeypatch.setattr(_handoff_launch, "run", lambda *args: calls.append(args))

    def forbidden(*args, **kw):
        pytest.fail("fused invocation revisited per-layer checks")

    monkeypatch.setattr(op, "_check_state", forbidden)
    monkeypatch.setattr(op, "handoff", forbidden)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    assert len(calls) == 1
    assert calls[0][1] is binding


@pytest.mark.parametrize("strict", [True, False])
def test_launcher_preserves_strided_records_and_reuses_compile(monkeypatch, strict):
    from ctikernels.gdn.replay.cute import _handoff_launch

    op, _, binding, _, _, statuses, slots, positions = setup()
    status = statuses[1] if strict else None
    compiles, launches = [], []

    def compile_fake(*args, **kwargs):
        compiles.append(args)
        return lambda *inputs: launches.append(inputs)

    monkeypatch.setattr(_handoff_launch, "compile_handoff", compile_fake)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    _handoff_launch.run(op, binding, positions, slots, status)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _handoff_launch.run(op, binding, positions, slots, status)
    assert len(compiles) == 1 and len(launches) == 2
    if strict:
        records = launches[0][3]
        assert records.data_ptr() == status.records.data_ptr()
        assert records.numel() == (len(binding.states) - 1) * status.records.stride(0) + 16
        for j in range(len(binding.states)):
            offset = j * status.records.stride(0)
            assert torch.equal(records[offset : offset + 16], status.records[j])
    else:
        assert launches[0][4].data_ptr() == positions.data_ptr()
        assert launches[0][5].data_ptr() == slots.data_ptr()


@pytest.mark.parametrize("error", ["sequence", "done", "code"])
@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("inplace", [True, False])
def test_cpu_handoff_layers_failure_prefix(error, layer, inplace):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(inplace=inplace)
    before = snapshot(binding.states)
    error_case(error, statuses, slots, positions, layer=layer)
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    compare(states, binding, statuses)
    assert_layer_prefix(before, binding, layer)
    assert_layer_status(statuses, error, layer)


@pytest.mark.parametrize("inplace", [False, True])
def test_layer_table_mixed_ring_capacity_flags(inplace):
    from ._handoff_case import mixed_setup

    _, _, binding, _, _, _, _, _ = mixed_setup(inplace=inplace)
    assert binding.table[:, 7].tolist() == [
        int(inplace) | 2,
        int(inplace),
        int(inplace) | 2,
        int(inplace),
    ]
    assert [s.ring.shape[1] for s in binding.states] == [0, 7, 0, 7]
