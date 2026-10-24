# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only host binding, interval and launch-recipe checks."""

import random
import weakref
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay, ReplayPending, ReplayState
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.replay import _api
from ctikernels.gdn.replay._host import HostPlan, intervals, possible_overlap
from ctikernels.gdn.replay._layout import physical
from ctikernels.gdn.replay.cute import _phase_launch


def brute(persistent, dynamic):
    return any(
        a < b and c < d and a < d and c < b and (write or other_write)
        for i, (a, b, write) in enumerate(persistent + dynamic)
        for j, (c, d, other_write) in enumerate(persistent + dynamic)
        if i < j and j >= len(persistent)
    )


def test_random_spans():
    rng = random.Random(412)
    for _ in range(300):
        spans = []
        for _ in range(20):
            start = rng.randrange(200)
            spans.append((start, start + rng.randrange(20), bool(rng.randrange(2))))
        persistent, dynamic = spans[:8], spans[8:]
        assert possible_overlap(
            intervals((a, b) for a, b, _ in persistent),
            intervals((a, b) for a, b, w in persistent if w),
            dynamic,
        ) == brute(persistent, dynamic)


@pytest.mark.parametrize("left,right", [(False, False), (False, True), (True, False), (True, True)])
def test_readonly_and_empty(left, right):
    assert possible_overlap(((), ()), ((), ()), [(1, 9, left), (2, 4, right)]) == (left or right)
    assert not possible_overlap(intervals([(1, 9)]), intervals([(1, 9)]), [(4, 4, True)])


def test_cpu_never_eligible():
    op = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=1, device="cpu")
    weights = tuple(torch.zeros(1) for _ in range(4))
    with torch.no_grad():
        assert not op._fast_phase("verify", weights, {})
    weights[0].requires_grad_(True)
    with pytest.raises(ValueError, match="conv_weight is a frozen base weight"):
        op._fast_phase("verify", weights, {})


def fixture(monkeypatch):
    g = Geometry(1, 1)
    op = SimpleNamespace(
        geometry=g,
        verify_length=2,
        window=4,
        device=torch.device("cpu"),
        checks="trusted",
        cute_config=None,
        _cute_launchers={},
        commit_mode_for=lambda b: "eager",
    )
    state = ReplayState(
        torch.zeros(2, 1, 128, 128),
        torch.zeros(2, 3, g.conv_dim, dtype=torch.bfloat16),
        torch.zeros(1, 7, g.conv_dim + 2, dtype=torch.bfloat16),
        torch.zeros(1, 8, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 4, dtype=torch.int64),
    )
    weights = (
        torch.zeros(g.conv_dim, 4, dtype=torch.bfloat16),
        *(torch.zeros(1) for _ in range(3)),
    )
    kw = {
        "slot_indices": torch.zeros(1, dtype=torch.int32),
        "positions": torch.zeros(1, 2, dtype=torch.int64),
        "projected": torch.zeros(1, 2, g.proj_width, dtype=torch.bfloat16),
        "out": torch.zeros(1, 2, g.value_width, dtype=torch.bfloat16),
        "pending": ReplayPending(
            torch.zeros(1, 2, g.conv_dim + 2, dtype=torch.bfloat16),
            torch.zeros(1, 8, dtype=torch.int64),
        ),
    }
    calls = []
    monkeypatch.setattr(_phase_launch, "device_sms", lambda device: 128)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        _phase_launch, "compile_phases", lambda *a, **k: lambda *args: calls.append(args)
    )
    return op, state, weights, kw, calls


def abi(args):
    return [(t.data_ptr(), t.shape, t.stride(), t.dtype) for t in args]


def expected_args(op, mode, state, weights, kw):
    """Original phase ABI, independent of the recipe builder."""
    flat = lambda t: t.view(-1) if t.is_contiguous() else physical(t)
    pending = kw["pending"]
    indices = kw["pairs"] if mode == "export_snapshot" else kw["slot_indices"]
    x = kw.get("projected")
    xf = state.ring.view(-1)
    if x is not None:
        width = op.geometry.proj_width * (op.verify_length if mode == "verify" else 1)
        xf = x.as_strided((max(0, (indices.shape[0] - 1) * x.stride(0) + width),), (1,))
    capture = kw.get("snapshot_rows") is not None
    return (
        xf,
        *(flat(w) for w in weights),
        flat(indices),
        flat(kw["positions"]),
        flat(state.checkpoint),
        flat(state.conv_history),
        state.ring.view(-1),
        state.metadata.view(-1),
        state.pool_epoch,
        state.snapshot_metadata.view(-1),
        flat(pending.raw),
        flat(pending.metadata),
        flat(kw.get("accepted_lengths", indices)),
        flat(kw["snapshot_rows"] if capture else indices),
        flat(
            kw["offsets"]
            if mode == "export_snapshot"
            else kw["snapshot_offsets"]
            if capture
            else state.pool_epoch
        ),
        flat(kw.get("snapshot_valid", state.ring.view(torch.bool))),
        flat(kw.get("out", state.ring)),
        state.ring.view(torch.uint8).view(-1),
        state.pool_epoch,
        flat(indices),
        flat(kw.get("arrivals", indices)),
    )


@pytest.mark.parametrize(
    "mode", ["verify", "commit", "bridge_step", "delayed_step", "flush", "export_snapshot"]
)
@pytest.mark.parametrize("snapshot", [False, True])
def test_recipe_rebinds_exact_views(monkeypatch, mode, snapshot):
    op, state, weights, kw, calls = fixture(monkeypatch)
    if mode in ("bridge_step", "delayed_step"):
        kw["projected"] = kw["projected"][:, 0]
        kw["out"] = kw["out"][:, 0]
    elif mode != "verify":
        kw.pop("projected")
        kw.pop("out")
    if mode == "export_snapshot":
        kw["pairs"] = torch.zeros(1, 2, dtype=torch.int32)
        kw["offsets"] = torch.zeros(1, dtype=torch.int64)
    if snapshot:
        kw.update(
            snapshot_rows=torch.ones(1, dtype=torch.int32),
            snapshot_offsets=torch.ones(1, dtype=torch.int64),
            snapshot_valid=torch.ones(1, dtype=torch.bool),
            accepted_lengths=torch.ones(1, dtype=torch.int32),
            arrivals=torch.zeros(1, dtype=torch.int32),
        )
    prepared = _phase_launch.run(op, mode, state=state, status=None, weights=weights, **kw)
    assert len(calls) == 1
    assert abi(calls[0]) == abi(expected_args(op, mode, state, weights, kw))
    per_call = _phase_launch.per_call(kw)
    rebound = {name: value.clone() for name, value in per_call.items()}
    prepared.launch(rebound)
    fresh = {
        **kw,
        **{name: value for name, value in rebound.items() if not name.startswith("pending.")},
    }
    fresh["pending"] = ReplayPending(rebound["pending.raw"], rebound["pending.metadata"])
    _phase_launch.run(op, mode, state=state, status=None, weights=weights, **fresh)
    assert abi(calls[-1]) == abi(calls[-2])
    for index, _, _ in prepared.recipe:
        assert prepared.args[index] is None
    # All placeholder occurrences of indices are rebound too.
    index_name = "pairs" if mode == "export_snapshot" else "slot_indices"
    assert calls[-1][22].data_ptr() == rebound[index_name].data_ptr()


def test_pitched_recipe_and_no_borrowed_references(monkeypatch):
    op, state, weights, kw, calls = fixture(monkeypatch)
    kw["projected"] = torch.zeros(1, 2, op.geometry.proj_width + 8, dtype=torch.bfloat16)[..., :-8]
    kw["out"] = torch.zeros(1, 2, op.geometry.value_width + 8, dtype=torch.bfloat16)[..., :-8]
    prepared = _phase_launch.plan(op, "verify", state=state, status=None, weights=weights, **kw)
    per_call = _phase_launch.per_call(kw)
    pointers = tuple(t.data_ptr() for t in per_call.values())
    entry = HostPlan(state, weights, prepared, per_call, pointers)
    refs = [weakref.ref(t) for t in per_call.values()]
    prepared.launch(per_call)
    assert abi(calls[0][:1]) == abi((physical(kw["projected"]),))
    assert abi(calls[0][19:20]) == abi((physical(kw["out"]),))
    calls.clear()
    del kw, per_call
    assert all(ref() is None for ref in refs)
    assert entry.unchanged()
    state.checkpoint.set_(state.checkpoint.clone())
    assert not entry.unchanged()


def test_layout_change_without_repointing(monkeypatch):
    op, state, weights, kw, _ = fixture(monkeypatch)
    prepared = _phase_launch.plan(op, "verify", state=state, status=None, weights=weights, **kw)
    per_call = _phase_launch.per_call(kw)
    entry = HostPlan(
        state, weights, prepared, per_call, tuple(t.data_ptr() for t in per_call.values())
    )
    pointer = state.checkpoint.data_ptr()
    state.checkpoint.transpose_(0, 1)
    assert state.checkpoint.data_ptr() == pointer
    assert not entry.unchanged()


def test_capture_cannot_create_launcher(monkeypatch):
    op, state, weights, kw, calls = fixture(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm CuTe specialization"):
        _phase_launch.plan(op, "verify", state=state, status=None, weights=weights, **kw)
    assert not calls


def test_switch_key(monkeypatch):
    before = _phase_launch.switches()
    monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", not _phase_launch.FUSED_PUBLISH)
    assert _phase_launch.switches() != before
    assert _api.HOST_FAST_PATH


def test_slotted_recipe_selector_is_borrowed(monkeypatch):
    op, state, weights, kw, calls = fixture(monkeypatch)
    pending = kw["pending"]
    kw["pending"] = ReplayPending(pending.raw.repeat(2, 1, 1, 1), pending.metadata.repeat(2, 1, 1))
    kw["status_slot"] = torch.zeros(1, dtype=torch.int32)
    prepared = _phase_launch.run(op, "verify", state=state, status=None, weights=weights, **kw)
    values = _phase_launch.per_call(kw)
    values["status_slot"] = torch.ones(1, dtype=torch.int32)
    prepared.launch(values)
    assert calls[-1][-1] is values["status_slot"]
    assert prepared.args[-1] is None


def test_addresses_recheck_readers_against_state(monkeypatch):
    op, state, weights, kw, _ = fixture(monkeypatch)
    prepared = _phase_launch.plan(op, "verify", state=state, status=None, weights=weights, **kw)
    values = _phase_launch.per_call(kw)
    pointers = tuple(t.data_ptr() for t in values.values())
    entry = HostPlan(state, weights, prepared, values, pointers)
    assert entry.addresses_valid(pointers)
    changed = list(pointers)
    changed[list(values).index("projected")] = state.checkpoint.data_ptr()
    assert not entry.addresses_valid(tuple(changed))
    assert entry.last_pointers == pointers
