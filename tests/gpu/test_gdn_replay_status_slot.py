# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""One captured K2 graph selects invocation storage on device (DN1)."""

from dataclasses import fields

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay._api import _cute_qualified

from ..modules._replay_case import case, clone
from ..modules.test_gdn_replay_status_slot import all_equal, bytes_equal, stacked, verify_args

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def setup(provider, checks, hk, hv, batch, q, pad=0):
    if provider == "cute" and not _cute_qualified(torch.device("cuda")):
        pytest.skip("CuTe transaction arithmetic is not qualified on this device")
    _, state, _, weights, slots, pos, _ = case(
        device="cuda", hk=hk, hv=hv, batch=batch, q=q, window=8
    )
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        window=8,
        verify_length=q,
        provider=provider,
        checks=checks,
    )
    state.checkpoint.fill_(0.01)
    state.conv_history.fill_(0.125)
    state.ring[:, -3:, : op.geometry.conv_dim].fill_(0.125)
    controls, pending = stacked(op, batch, 2, pad=pad)
    reference_controls, reference_pending = stacked(op, batch, 2)
    reference = clone(state)
    status = controls.slotted_status(1, batch_size=batch) if op.checks == "strict" else None
    kwargs = verify_args(op, state, pending, weights, slots, pos, status)
    return (
        op,
        state,
        reference,
        weights,
        controls,
        reference_controls,
        pending,
        reference_pending,
        kwargs,
    )


def copy_state(target, source):
    for f in fields(source):
        getattr(target, f.name).copy_(getattr(source, f.name))


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("checks", ["strict", "trusted"])
@pytest.mark.parametrize(
    "geometry", [(16, 32), (8, 16), pytest.param((8, 24), marks=pytest.mark.exhaustive)]
)
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("q", [8, 16])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_status_slot_rounds(provider, checks, geometry, batch, q, capture):
    exercise(provider, checks, geometry, batch, q, capture=capture, snapshot=False)


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("checks", ["strict", "trusted"])
@pytest.mark.parametrize("pad", [1, 7])
@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
def test_status_slot_odd_pitch(provider, checks, pad, capture):
    # Slot 1 of the stacked pending raw is 2/14 bytes off 16: scalar publication.
    exercise(provider, checks, (16, 32), 2, 8, capture=capture, snapshot=False, pad=pad)


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("checks", ["strict", "trusted"])
def test_status_slot_snapshot_graph(provider, checks):
    exercise(provider, checks, (8, 16), 4, 8, capture=True, snapshot=True)


def exercise(provider, checks, geometry, batch, q, *, capture, snapshot, pad=0):
    op, state, reference, weights, controls, ref_controls, pending, ref_pending, kwargs = setup(
        provider, checks, *geometry, batch, q, pad
    )
    selector = torch.zeros(1, dtype=torch.int32, device="cuda")
    slot_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    arrivals = torch.zeros(2, batch, dtype=torch.int32, device="cuda")
    ref_arrivals = arrivals.clone()
    valid = torch.ones(2, batch, dtype=torch.bool, device="cuda")
    ref_valid = valid.clone()
    rows = torch.arange(batch, dtype=torch.int32, device="cuda") + state.metadata.shape[0]
    offsets = torch.arange(batch, dtype=torch.int64, device="cuda")

    def launch(d=None):
        dynamic = d is None
        args = (
            kwargs
            if dynamic
            else dict(
                kwargs,
                state=reference,
                out=expected_out,
                pending=ref_controls.pending(ref_pending.raw[d], slot=d, layer=1),
                status=ref_controls.status(d, 1, batch_size=batch)
                if op.checks == "strict"
                else None,
            )
        )
        extra = {"status_slot": selector} if dynamic else {}
        op.verify(**args, **extra)
        workspace = (
            {"arrivals": arrivals if dynamic else ref_arrivals[d]}
            if op._phase_arrivals("commit")
            else None
        )
        snapshots = (
            {
                "snapshot_rows": rows,
                "snapshot_offsets": offsets,
                "snapshot_valid": valid if dynamic else ref_valid[d],
            }
            if snapshot
            else {}
        )
        op.commit(
            *weights,
            slot_indices=kwargs["slot_indices"],
            accepted_lengths=accepted,
            state=args["state"],
            pending=args["pending"],
            status=args["status"],
            workspace=workspace,
            **snapshots,
            **extra,
        )

    expected_out = kwargs["out"].clone()
    # Eager warm-up on a side stream, then restore every mutable warm-up buffer.
    initial_pending = clone(pending)
    initial_controls = {name: t.clone() for name, t in controls.buffers.items()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    copy_state(state, reference)
    copy_state(pending, initial_pending)
    for name, t in controls.buffers.items():
        t.copy_(initial_controls[name])
    valid.copy_(ref_valid)
    kwargs["out"].copy_(expected_out)
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()

    for round_index, d in enumerate((0, 1, 0)):
        # Update shared controls and the device selector before replay. The host
        # source remains alive until the stream has consumed the asynchronous copy.
        slot_host.fill_(d)
        selector.copy_(slot_host, non_blocking=True)
        # Rotate within the initialized live slots (case() initializes 0..B-1).
        indices = (torch.arange(batch, device="cuda") + round_index) % batch
        kwargs["slot_indices"].copy_(indices)
        kwargs["positions"].copy_(
            reference.metadata[indices, 1, None] + torch.arange(q, device="cuda")
        )
        accepted.fill_((0, q, q // 2)[round_index])
        untouched_raw, untouched_pm = pending.raw[1 - d].clone(), pending.metadata[1 - d].clone()
        if graph is not None:
            graph.replay()
        else:
            launch()
        launch(d)
        bytes_equal(kwargs["out"], expected_out)
        all_equal(state, reference)
        all_equal(pending, ref_pending)
        bytes_equal(pending.raw[1 - d], untouched_raw)
        bytes_equal(pending.metadata[1 - d], untouched_pm)
        bytes_equal(valid, ref_valid)
        bytes_equal(arrivals, ref_arrivals)
        assert not arrivals.any().item()
        for name in ("records", "validity", "sequence", "poison", "pending_metadata"):
            bytes_equal(controls.buffers[name], ref_controls.buffers[name])
        assert not controls.buffers["poison"].any().item()

    # The same stacked descriptors also support rejection of a fresh draft.
    kwargs["positions"].copy_(
        state.metadata[kwargs["slot_indices"].long(), 1, None] + torch.arange(q, device="cuda")
    )
    op.verify(**kwargs, status_slot=selector)
    ref_args = dict(
        kwargs,
        state=reference,
        pending=ref_controls.pending(ref_pending.raw[0], slot=0, layer=1),
        status=ref_controls.status(0, 1, batch_size=batch) if op.checks == "strict" else None,
        out=expected_out,
    )
    op.verify(**ref_args)
    op.reject(
        slot_indices=kwargs["slot_indices"],
        state=state,
        pending=pending,
        status=kwargs["status"],
        status_slot=selector,
    )
    op.reject(
        slot_indices=kwargs["slot_indices"],
        state=reference,
        pending=ref_args["pending"],
        status=ref_args["status"],
    )
    all_equal(state, reference)
    all_equal(pending, ref_pending)


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("checks", ["strict", "trusted"])
@pytest.mark.parametrize("value", [-1, 2])
@pytest.mark.parametrize("mode", ["verify", "commit", "reject"])
def test_invalid_status_slot(provider, checks, value, mode):
    op, state, reference, weights, controls, _, pending, _, kwargs = setup(
        provider, checks, 8, 16, 1, 8
    )
    before = clone(pending)
    output = kwargs["out"].clone()
    records = controls.buffers["records"].clone()
    valid = torch.ones(2, 1, dtype=torch.bool, device="cuda")
    arrivals = torch.zeros(2, 1, dtype=torch.int32, device="cuda")
    extra = {"status_slot": torch.tensor([value], dtype=torch.int32, device="cuda")}
    if mode == "verify":
        op.verify(**kwargs, **extra)
    else:
        args = {
            "slot_indices": kwargs["slot_indices"],
            "state": state,
            "pending": pending,
            "status": kwargs["status"],
        }
        if mode == "reject":
            op.reject(**args, **extra)
        else:
            op.commit(
                *weights,
                **args,
                **extra,
                accepted_lengths=torch.zeros(1, dtype=torch.int32, device="cuda"),
                snapshot_rows=torch.full(
                    (1,), state.metadata.shape[0], dtype=torch.int32, device="cuda"
                ),
                snapshot_offsets=torch.zeros(1, dtype=torch.int64, device="cuda"),
                snapshot_valid=valid,
                workspace={"arrivals": arrivals} if op._phase_arrivals("commit") else None,
            )
    all_equal(state, reference)
    all_equal(pending, before)
    bytes_equal(kwargs["out"], output)
    bytes_equal(controls.buffers["records"], records)
    assert valid.all().item() and not arrivals.any().item()
    assert bool(controls.buffers["poison"].item()) == (op.checks == "strict")
