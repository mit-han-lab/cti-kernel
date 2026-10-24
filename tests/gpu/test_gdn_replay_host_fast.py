# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""HF-R eager/captured launch equivalence and warm binding guards."""

from dataclasses import fields
from unittest.mock import Mock

import pytest
import torch

from benchmark.kernel_tuning.gdn_replay_host_time import Workload
from ctikernels.gdn.replay import _api
from ctikernels.gdn.replay.cute import _phase_launch

from ..modules._replay_case import clone
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CASES = ci_cases(
    ((16, 32), (8, 16), (8, 24)),
    (1, 2, 3, 4),
    (8, 16),
    (False, True),
    core=[
        (*g, b, q, snapshot)
        for g in ((16, 32), (8, 16))
        for b, q, snapshot in ((1, 8, False), (2, 16, True), (3, 8, True), (4, 16, False))
    ]
    + [(8, 24, 1, 8, False), (8, 24, 4, 8, True)],
)


def setup(batch=1, q=8, *, geometry=(8, 16), layers=1, snapshots=0):
    if not _api._cute_qualified(torch.device("cuda")):
        pytest.skip("CuTe transaction arithmetic is not qualified on this device")
    work = Workload(
        batch, q, layers, max(8, batch), 3, batch + 1, geometry=geometry, snapshots=snapshots
    )
    if any(work.op.provider[p] != "cute" for p in ("verify", "commit")):
        pytest.skip("CuTe verify/commit required")
    return work


def copy_work(target, source):
    for a, b in zip(target.mutable(), source.mutable(), strict=True):
        a.copy_(b)
    target.joint.copy_(source.joint)
    for a, b in zip(target.weights, source.weights, strict=True):
        for x, y in zip(a, b, strict=True):
            x.copy_(y)


def equal(left, right):
    for a, b in zip(left.mutable(), right.mutable(), strict=True):
        assert torch.equal(a, b)


def controls(work, index, accepted):
    b, q = work.batch, work.q
    slots = (torch.arange(b, device="cuda") + index) % b
    work.positions.copy_(work.states[0].metadata[slots, 1, None] + torch.arange(q, device="cuda"))
    if b > 1:
        slots[-1] = -1
    work.slots.copy_(slots)
    work.accepted.fill_(accepted)


@pytest.mark.parametrize("hk,hv,batch,q,snapshot", CASES)
@torch.no_grad()
def test_rounds_bitwise(monkeypatch, hk, hv, batch, q, snapshot):
    geometry = hk, hv
    fast = setup(batch, q, geometry=geometry, snapshots=batch)
    slow = setup(batch, q, geometry=geometry, snapshots=batch)
    copy_work(slow, fast)
    valid = [torch.zeros(batch, dtype=torch.bool, device="cuda") for _ in range(2)]
    rows = torch.arange(batch, dtype=torch.int32, device="cuda") + fast.states[0].metadata.shape[0]
    offsets = torch.zeros(batch, dtype=torch.int64, device="cuda")
    for r, accepted in enumerate((0, q // 2, q, q // 2, 0, q)):
        offsets.fill_(accepted)
        for i, work in enumerate((fast, slow)):
            monkeypatch.setattr(_api, "HOST_FAST_PATH", i == 0)
            controls(work, r, accepted)
            work.verify(0, r % 2)
            work.commit(
                0,
                r % 2,
                **(
                    {"snapshot_rows": rows, "snapshot_offsets": offsets, "snapshot_valid": valid[i]}
                    if snapshot
                    else {}
                ),
            )
        equal(fast, slow)
        assert torch.equal(valid[0], valid[1])
    assert fast.op._plans and not slow.op._plans


@torch.no_grad()
def test_hits_skip_validation(monkeypatch):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.round(0)
    mocks = []
    for obj, name in (
        (work.op, "_run"),
        (work.op, "_check_state"),
        (work.op, "_check_pending"),
        (work.op, "_validate_overlaps"),
        (_phase_launch, "compile_phases"),
        (_api, "_identity"),
    ):
        mock = Mock(wraps=getattr(obj, name))
        monkeypatch.setattr(obj, name, mock)
        mocks.append(mock)
    for _ in range(3):
        work.positions.add_(work.q // 2)
        work.round(0)
    assert all(mock.call_count == 0 for mock in mocks)


@torch.no_grad()
def test_new_dynamic_addresses_reuse_plan(monkeypatch):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.round(0)
    work.positions.add_(work.q // 2)
    work.joint = work.joint.clone()
    work.out = work.out.clone()
    work.raw = work.raw.clone()
    work.arrivals = work.arrivals.clone()
    before = [t.clone() for t in work.mutable()]
    run = Mock(wraps=work.op._run)
    monkeypatch.setattr(work.op, "_run", run)
    work.round(0)
    assert run.call_count == 0
    expected = [t.clone() for t in work.mutable()]
    for t, value in zip(work.mutable(), before, strict=True):
        t.copy_(value)
    monkeypatch.setattr(_api, "HOST_FAST_PATH", False)
    work.round(0)
    for t, value in zip(work.mutable(), expected, strict=True):
        assert torch.equal(t, value)


@torch.no_grad()
def test_graph_two_layers(monkeypatch):
    works = [setup(batch=2, layers=2) for _ in range(3)]
    for work in works[1:]:
        copy_work(work, works[0])
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    stream = torch.cuda.Stream()
    initial = [[t.clone() for t in work.mutable()] for work in works]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for work in works:
            work.round(0)
    torch.cuda.current_stream().wait_stream(stream)
    for work, values in zip(works, initial, strict=True):
        for t, value in zip(work.mutable(), values, strict=True):
            t.copy_(value)
    counts = [len(work.op._cute_launchers) for work in works]
    misses = _phase_launch.compile_phases.cache_info().misses
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        works[0].round(0)
    for r, accepted in enumerate((0, 4, 8, 4)):
        for work in works:
            controls(work, r, accepted)
        graph.replay()
        monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
        works[1].round(0)
        monkeypatch.setattr(_api, "HOST_FAST_PATH", False)
        works[2].round(0)
        equal(works[0], works[1])
        equal(works[0], works[2])
    assert counts == [len(work.op._cute_launchers) for work in works]
    assert misses == _phase_launch.compile_phases.cache_info().misses


@pytest.mark.parametrize("mode", ["verify", "commit"])
@torch.no_grad()
def test_frozen_before_launch(monkeypatch, mode):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.round(0)
    before = [t.clone() for t in work.mutable()]
    work.weights[0][1].requires_grad_(True)
    launch = Mock(side_effect=AssertionError("launched"))
    monkeypatch.setattr(_phase_launch.PhasePlan, "launch", launch)
    with pytest.raises(ValueError, match="A_log is a frozen base weight"):
        getattr(work, mode)(0, 0)
    assert not launch.called
    for t, value in zip(work.mutable(), before, strict=True):
        assert torch.equal(t, value)


@pytest.mark.parametrize("source", ["checkpoint", "pending"])
@torch.no_grad()
def test_overlap_matches_slow_error(monkeypatch, source):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.verify(0, 0)
    state = clone(work.states[0])
    original = work.out
    base = work.states[0].checkpoint.view(torch.bfloat16) if source == "checkpoint" else work.raw
    work.out = base.flatten()[: work.batch * work.q * work.op.geometry.value_width].view(
        work.batch, work.q, work.op.geometry.value_width
    )
    assert work.out.data_ptr() % 256 == original.data_ptr() % 256
    errors = []
    for fast in (False, True):
        monkeypatch.setattr(_api, "HOST_FAST_PATH", fast)
        with pytest.raises(ValueError, match="overlapping buffers") as error:
            work.verify(0, 0)
        errors.append(str(error.value))
    assert errors[0] == errors[1]
    for f in fields(state):
        assert torch.equal(getattr(state, f.name), getattr(work.states[0], f.name))


@torch.no_grad()
def test_alignment_new_state_and_repoint(monkeypatch):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.verify(0, 0)
    count = len(work.op._plans)
    # A new BF16 view with identical pitches but two-byte base alignment.
    shifted = torch.empty(work.joint.numel() + 1, dtype=torch.bfloat16, device="cuda")[1:]
    shifted.copy_(work.joint.flatten())
    work.joint = shifted.view_as(work.joint)
    work.verify(0, 0)
    assert len(work.op._plans) == count + 1
    out, pending = work.out.clone(), work.raw.clone()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", False)
    work.verify(0, 0)
    assert torch.equal(work.out, out) and torch.equal(work.raw, pending)
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    run = Mock(wraps=work.op._run)
    monkeypatch.setattr(work.op, "_run", run)
    work.states[0] = clone(work.states[0])
    work.verify(0, 0)
    assert run.call_count == 1
    count = len(work.op._plans)
    work.states[0].checkpoint.set_(work.states[0].checkpoint.clone())
    work.verify(0, 0)
    assert run.call_count == 2 and len(work.op._plans) == count - 1
    assert torch.equal(work.out, out)


@torch.no_grad()
def test_mutated_layout_revalidated(monkeypatch):
    work = setup()
    monkeypatch.setattr(_api, "HOST_FAST_PATH", True)
    work.verify(0, 0)
    work.states[0].checkpoint.transpose_(0, 1)
    with pytest.raises((ValueError, TypeError)):
        work.verify(0, 0)
