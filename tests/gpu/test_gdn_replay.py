# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Portable K2 acceptance suite. Run: pytest -q tests/gpu/test_gdn_replay.py.

Small deterministic schedules keep execution practical on all three target devices.
This is arithmetic/ABI validation, not the contract §10 model promotion matrix.
CPU-generated inputs are identical on every device. No external provider is needed.
"""

from dataclasses import fields

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay import ReplayStatus

from ..modules._replay_case import (
    case,
    clone,
    finish,
    graph_buffers,
    live_ring,
    projected,
    same,
    zero_rows,
)
from . import _gdn_reference as pinned
from ._exhaustive import ci_cases
from .test_gdn_triton import HARD_FLOOR_ULP, ORDINARY_ULP, _max_ulp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Pair:
    def __init__(self, *, hk=8, hv=16, window=4, q=8, batch=2):
        self.op, self.state, self.pending, self.weights, self.slots, self.pos, self.status = case(
            device="cuda", hk=hk, hv=hv, window=window, q=q, batch=batch
        )
        self.ref = GatedDeltaNetReplay(
            num_k_heads=hk, num_v_heads=hv, window=window, verify_length=q, device="cpu"
        )
        self.cpu = clone(self.state, "cpu")
        self.cp = clone(self.pending, "cpu")
        self.cw = tuple(x.cpu() for x in self.weights)
        self.cs = ReplayStatus(
            self.status.records.cpu().clone(),
            self.status.validity.cpu().clone(),
            self.status.sequence.cpu().clone(),
            self.status.poison.cpu().clone(),
        )
        self.workspace = {
            name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device="cuda")
            for name, spec in self.op.workspace_spec("step", batch_size=batch).items()
            if name != "out"
        }
        self.batch = batch
        self.limit = ORDINARY_ULP

    def call(self, phase, **kw):
        kw = dict(kw, state=self.state, status=self.status)
        ck = {
            name: value.cpu().clone() if isinstance(value, torch.Tensor) else value
            for name, value in kw.items()
        }
        ck.update(state=self.cpu, status=self.cs)
        if "pending" in kw:
            ck["pending"] = self.cp
        weights = (
            self.weights
            if phase in ("step", "bridge_step", "verify", "flush", "commit", "export_snapshot")
            else ()
        )
        cw = self.cw if weights else ()
        if phase in ("step", "bridge_step", "verify"):
            x, cx = kw.pop("projected"), ck.pop("projected")
            if phase == "step":
                kw["workspace"] = self.workspace
            getattr(self.op, phase)(x, *weights, **kw)
            getattr(self.ref, phase)(cx, *cw, **ck)
            got = kw["out"].cpu().reshape(-1, self.op.geometry.v_heads, 128)
            truth = ck["out"].reshape_as(got)
            assert _max_ulp(got, truth) <= self.limit
        else:
            getattr(self.op, phase)(*weights, **kw)
            getattr(self.ref, phase)(*cw, **ck)
        assert torch.equal(self.status.code.cpu(), self.cs.code)
        for name in ("snapshot_valid",):
            if name in kw:
                assert torch.equal(kw[name].cpu(), ck[name])
        self.check()

    def check(self):
        for f in fields(self.state):
            got, truth = getattr(self.state, f.name).cpu(), getattr(self.cpu, f.name)
            if f.name == "checkpoint":
                assert _max_ulp(got, truth) <= self.limit
            elif f.name == "ring":
                assert torch.equal(live_ring(self.state).cpu(), live_ring(self.cpu)), f.name
            else:
                assert torch.equal(got, truth), f.name
        same(clone(self.pending, "cpu"), self.cp)

    def step(self, seed):
        out = torch.empty(
            self.batch, self.op.geometry.value_width, dtype=torch.bfloat16, device="cuda"
        )
        self.call(
            "step",
            projected=projected(self.op, self.batch, seed=seed),
            slot_indices=self.slots,
            positions=self.pos,
            out=out,
        )
        self.pos.add_(1)

    def verify(self, seed):
        q = self.op.verify_length
        x = projected(self.op, self.batch, length=q, seed=seed)
        out = torch.empty(
            self.batch, q, self.op.geometry.value_width, dtype=torch.bfloat16, device="cuda"
        )
        before = clone(self.state)
        self.call(
            "verify",
            projected=x,
            slot_indices=self.slots,
            positions=self.pos[:, None] + torch.arange(q, device="cuda"),
            pending=self.pending,
            out=out,
        )
        same(before, self.state)

    def commit(self, m, k=None):
        lengths = torch.tensor(m, dtype=torch.int32, device="cuda")
        controls = {}
        if k is not None:
            controls = {
                "snapshot_rows": torch.arange(self.batch, dtype=torch.int32, device="cuda")
                + self.state.metadata.shape[0],
                "snapshot_offsets": torch.tensor(k, dtype=torch.int64, device="cuda"),
                "snapshot_valid": torch.empty(self.batch, dtype=torch.bool, device="cuda"),
            }
        self.call(
            "commit",
            slot_indices=self.slots,
            accepted_lengths=lengths,
            pending=self.pending,
            **controls,
        )
        self.pos.add_(lengths.to(torch.int64))


# Boundary CI keeps 8/16 heads with window/Q/B rows 4/2/1 and 8/4/2, plus the TP2
# engine shape (8,24) window 4, Q=8, B=1.
@pytest.mark.parametrize(
    "hk,hv,window,q,batch",
    ci_cases(
        [(8, 16, 4, 2, 1), (8, 24, 8, 8, 2), (16, 32, 8, 16, 4), (16, 48, 16, 32, 1)]
        + [(8, 16, 8, q, 2) for q in (4, 5, 9, 12, 13, 17)]
        + [(8, 24, 4, 8, b) for b in (1, 2, 3, 4)],
        core=[(8, 16, 4, 2, 1), (8, 16, 8, 4, 2), (8, 24, 4, 8, 1)],
    ),
)
def test_boundaries_all_acceptances_offsets_and_prefix_cache(hk, hv, window, q, batch):
    p = Pair(hk=hk, hv=hv, window=window, q=q, batch=batch)
    assert set(p.op.provider.values()) <= {"triton", "cute"}
    for i in range(3 * window + 1):
        p.step(100 + i)
    p.verify(200)
    base, tentative = clone(p.state), clone(p.pending)
    base_cpu, tentative_cpu = clone(p.cpu), clone(p.cp)
    old_pos = p.pos.clone()
    r = base.metadata.shape[0]
    for m in range(q + 1):
        p.state, p.pending, p.cpu, p.cp = (
            clone(base),
            clone(tentative),
            clone(base_cpu),
            clone(tentative_cpu),
        )
        p.pos.copy_(old_pos)
        p.commit([m] * batch, [m] * batch)
        # Fused capture at k=m equals canonical committed state bitwise.
        assert torch.equal(p.state.checkpoint[r : r + batch], p.state.checkpoint[:batch])
        assert torch.equal(p.state.conv_history[r : r + batch], p.state.conv_history[:batch])
        oracle = p.state.checkpoint[:batch].clone()
        oracle_h = p.state.conv_history[:batch].clone()
        # Eventual larger acceptance must not change the captured prefix bits.
        p.state, p.pending, p.cpu, p.cp = (
            clone(base),
            clone(tentative),
            clone(base_cpu),
            clone(tentative_cpu),
        )
        p.pos.copy_(old_pos)
        export_pairs = torch.stack(
            (p.slots, torch.arange(batch, dtype=torch.int32, device="cuda") + r), 1
        )
        p.call(
            "export_snapshot",
            pairs=export_pairs,
            pending=p.pending,
            accepted_lengths=torch.full((batch,), q, dtype=torch.int32, device="cuda"),
            offsets=torch.full((batch,), m, dtype=torch.int64, device="cuda"),
            snapshot_valid=torch.empty(batch, dtype=torch.bool, device="cuda"),
        )
        assert torch.equal(p.state.checkpoint[r : r + batch], oracle)
        assert torch.equal(p.state.conv_history[r : r + batch], oracle_h)
        p.commit([q] * batch, [m] * batch)
        assert torch.equal(p.state.checkpoint[r : r + batch], oracle)
        assert torch.equal(p.state.conv_history[r : r + batch], oracle_h)
    p.verify(300)
    before = clone(p.state)
    p.call("reject", slot_indices=p.slots, pending=p.pending)
    same(before, p.state)
    p.step(301)  # export an unflushed tail
    # Non-mutating export, duplicated read sources, inactive pairs, fork/import.
    pair = torch.tensor([[0, r], [0, r + 1], [-1, r + 2]], dtype=torch.int32, device="cuda")
    before = clone(p.state)
    p.call(
        "export_snapshot",
        pairs=pair,
        snapshot_valid=torch.empty(3, dtype=torch.bool, device="cuda"),
    )
    for name in ("checkpoint", "conv_history", "ring", "metadata"):
        assert torch.equal(getattr(before, name)[:r], getattr(p.state, name)[:r])
    p.call("import_snapshot", pairs=torch.tensor([[r, r - 1]], dtype=torch.int32, device="cuda"))
    p.call("flush", slot_indices=p.slots)
    if p.op.provider["flush"] == "cute":
        # CuTe materializes with the product recipe of each call's batch
        # specialization (here export B=3, flush B=1): equal within FP32 rounding.
        torch.testing.assert_close(
            p.state.checkpoint[0], p.state.checkpoint[r - 1], rtol=1e-5, atol=1e-6
        )
    else:
        assert torch.equal(p.state.checkpoint[0], p.state.checkpoint[r - 1])
    p.call("release", slot_indices=p.slots)
    p.call("initialize", slot_indices=p.slots, next_positions=p.pos)


@pytest.mark.parametrize("input_class", ["ordinary", "heavy", "zero_keys"])
def test_prefill_handoff_and_immediate_fp64_oracle(input_class):
    p = Pair(hk=1, hv=4, batch=1)
    p.limit = ORDINARY_ULP if input_class == "ordinary" else HARD_FLOOR_ULP
    g = p.op.geometry
    # The pinned independent reference supplies FP32 prefill state and exact history.
    w = pinned.Weights(*p.cw)
    seq = pinned._Sequence(
        w, g.k_heads, torch.zeros_like(p.cpu.checkpoint[0]), p.cpu.conv_history[0]
    )
    for j in range(5):
        seq.step(projected(p.ref, 1, seed=j)[0])
    canonical = seq.pool_state().float()[None].contiguous().cuda()
    history = seq.history[None].contiguous().cuda()
    p.call(
        "handoff",
        canonical_state=canonical,
        history=history,
        next_positions=p.pos,
        slot_indices=p.slots,
    )
    seq = pinned._Sequence(w, g.k_heads, canonical[0].cpu(), history[0].cpu())
    out = torch.empty(1, g.value_width, dtype=torch.bfloat16, device="cuda")
    for j in range(13):
        x = projected(p.ref, 1, seed=600 + j)
        if input_class == "heavy":
            x *= 8
        elif input_class == "zero_keys":
            x[:, g.key_width : 2 * g.key_width] = 0
        exact = seq.step(x[0]).view(g.v_heads, 128)
        p.call("step", projected=x.cuda(), slot_indices=p.slots, positions=p.pos, out=out)
        assert _max_ulp(out.cpu().view(g.v_heads, 128), exact) <= p.limit
        p.pos.add_(1)
    p.call("flush", slot_indices=p.slots)
    assert _max_ulp(p.state.checkpoint[0].cpu(), seq.pool_state()) <= p.limit


def test_multiphase_flushes_rollbacks_reset_and_invalid_calls():
    p = Pair(hk=1, hv=4, batch=2)
    p.slots[1] = -1
    for cycle in range(12):
        for j in range(5):
            p.step(1000 + cycle * 5 + j)
        p.verify(2000 + cycle)
        if cycle % 3 == 0:
            p.call("reject", slot_indices=p.slots, pending=p.pending)
        else:
            m = cycle % 9
            p.commit([m, -900], [m + 1, 0])  # unavailable capture, inactive invalid m
        if cycle % 4 == 0:
            p.call("flush", slot_indices=p.slots)
    p.verify(3000)
    saved = clone(p.state)
    p.call(
        "commit",
        slot_indices=p.slots,
        pending=p.pending,
        accepted_lengths=torch.tensor([-1, -999], dtype=torch.int32, device="cuda"),
    )
    assert p.status.code.item() & 4
    same(saved, p.state)
    finish(p.op, p.status)
    finish(p.ref, p.cs)
    p.op.recover_reset(
        states=(p.state,), status=p.status, pending_blocks=(p.pending,), drained=True
    )
    p.ref.recover_reset(states=(p.cpu,), status=p.cs, pending_blocks=(p.cp,), drained=True)
    p.status.sequence.add_(1)
    p.cs.sequence.add_(1)
    p.op.begin(status=p.status)
    p.ref.begin(status=p.cs)
    # Reset payload must stay bitwise intact, across ten discard/resume cycles.
    for _ in range(10):
        before = clone(p.state)
        p.op.reset(state=p.state, weight_revision=9, status=p.status, pending_blocks=(p.pending,))
        p.ref.reset(state=p.cpu, weight_revision=9, status=p.cs, pending_blocks=(p.cp,))
        assert torch.equal(p.state.checkpoint, before.checkpoint)
        assert torch.equal(p.state.ring, before.ring)
        p.check()
        p.call("initialize", slot_indices=p.slots, next_positions=p.pos)
    p.verify(4000)
    p.state.metadata[0, 5].add_(1)
    p.cpu.metadata[0, 5].add_(1)
    p.call("reject", slot_indices=p.slots, pending=p.pending)
    assert p.status.code.item() & 8


@pytest.mark.parametrize("fault,code", [("position", 2), ("uninitialized", 1), ("snapshot", 16)])
def test_validation_suppresses_other_rows_and_later_phases(fault, code):
    p = Pair(hk=1, hv=4, batch=2)
    if fault == "position":
        p.pos[1] += 1
    elif fault == "uninitialized":
        p.state.metadata[1, 4] = 0
        p.cpu.metadata[1, 4] = 0
    before = clone(p.state)
    if fault == "snapshot":
        p.call("import_snapshot", pairs=torch.tensor([[4, 1]], dtype=torch.int32, device="cuda"))
    else:
        p.step(88)
    assert p.status.code.item() & code
    same(before, p.state)
    p.call("initialize", slot_indices=p.slots, next_positions=p.pos)
    same(before, p.state)


@pytest.mark.parametrize(
    "batch,q",
    ci_cases([1, 2, 3, 4], [5, 8, 16, 17], core=[(1, 5)]),
)
def test_graph_ring_async_completion_device_controls_and_mixed_phases(batch, q):
    # TP-local geometry is exercised by B3; small heads bound compilation/runtime.
    hk, hv = (8, 16) if batch == 3 else (1, 4)
    op, _, _, weights, _, _, _ = case(device="cuda", hk=hk, hv=hv, batch=batch, q=q)
    ref = GatedDeltaNetReplay(
        num_k_heads=hk, num_v_heads=hv, window=4, verify_length=q, device="cpu"
    )
    g = op.geometry
    workspace = {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device="cuda")
        for name, spec in op.workspace_spec("step", batch_size=batch).items()
        if name != "out"
    }
    state = op.create_state(
        buffers={
            name: torch.zeros(spec.shape, dtype=spec.dtype, device="cuda")
            for name, spec in op.state_spec(
                num_slots=2 * batch, num_snapshot_rows=2 * batch
            ).items()
        }
    )
    controls = graph_buffers(op, batch=batch)
    buf = controls.buffers
    handles = [controls.status(i) for i in range(2)]
    raw = torch.zeros(2, batch, q, g.conv_dim + 2 * hv, dtype=torch.bfloat16, device="cuda")
    pending = [controls.pending(raw[i], slot=i) for i in range(2)]
    block = projected(op, 2 * batch, length=q).reshape(2, batch, q, g.proj_width)
    sx = projected(op, 2 * batch).reshape(2, batch, g.proj_width)
    out = torch.empty(2, batch, q, g.value_width, dtype=torch.bfloat16, device="cuda")
    step_out = torch.empty(2, batch, g.value_width, dtype=torch.bfloat16, device="cuda")
    rows = torch.arange(batch, dtype=torch.int32, device="cuda")
    qrows = torch.arange(q, dtype=torch.int64, device="cuda")
    mirrors = [
        (
            torch.empty(1, 16, dtype=torch.uint8, pin_memory=True),
            torch.empty(1, batch, dtype=torch.bool, pin_memory=True),
        )
        for _ in range(2)
    ]
    events = [torch.cuda.Event() for _ in range(2)]
    graphs = {}
    sequence = [1, 2]

    def stage(slot, cycle):
        h = handles[slot]
        h.sequence.fill_(sequence[slot])
        # Slot permutations, padding, logical positions and m are device controls.
        mapping = (rows + cycle) % batch + slot * batch
        mapping = torch.where((rows == batch - 1) & (cycle % 4 == 3), -1, mapping)
        buf["slot_indices"][slot].copy_(mapping)
        positions = state.metadata[:, 1].index_select(0, mapping.clamp_min(0).long())
        buf["positions"][slot].copy_(positions)
        buf["verify_positions"][slot].copy_(positions[:, None] + qrows)
        buf["snapshot_rows"][slot].copy_(rows + 2 * batch + slot * batch)
        buf["snapshot_offsets"][slot].fill_(cycle % (q + 2))

    def body(slot, mode):
        h = handles[slot]
        op.begin(status=h, layer_weights=(weights,))
        if mode == "verify":
            op.verify(
                block[slot],
                *weights,
                slot_indices=buf["slot_indices"][slot],
                positions=buf["verify_positions"][slot],
                state=state,
                pending=pending[slot],
                out=out[slot],
                status=h,
            )
            # A graph-resident acceptance surrogate, no host acceptance readback.
            torch.add(rows, h.sequence[0], out=buf["accepted_lengths"][slot])
            buf["accepted_lengths"][slot].remainder_(q + 1)
            op.commit(
                *weights,
                slot_indices=buf["slot_indices"][slot],
                accepted_lengths=buf["accepted_lengths"][slot],
                state=state,
                pending=pending[slot],
                status=h,
                snapshot_rows=buf["snapshot_rows"][slot],
                snapshot_offsets=buf["snapshot_offsets"][slot],
                snapshot_valid=h.valid,
            )
        else:
            getattr(op, mode)(
                sx[slot],
                *weights,
                slot_indices=buf["slot_indices"][slot],
                positions=buf["positions"][slot],
                state=state,
                out=step_out[slot],
                workspace=workspace if mode == "step" else None,
                status=h,
            )
        op.end(status=h)

    # Initialize and stagger cursors without depending on Python during replay.
    for slot in range(2):
        stage(slot, 0)
        op.begin(status=handles[slot])
        op.initialize(
            slot_indices=buf["slot_indices"][slot],
            next_positions=buf["positions"][slot],
            state=state,
            status=handles[slot],
        )
        finish(op, handles[slot])
        sequence[slot] += 2
    # Each fixed slot gets one graph of each phase, captured once after eager warmup.
    for mode in ("step", "verify", "bridge_step"):
        for slot in range(2):
            stage(slot, 0)
            body(slot, mode)
            finish(op, handles[slot])
            sequence[slot] += 2
            stage(slot, 0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body(slot, mode)
            # Capture records launches but does not execute them.
            graphs[slot, mode] = graph
    cw = tuple(w.cpu() for w in weights)
    cpu = clone(state, "cpu")
    cpu_pending = [clone(p, "cpu") for p in pending]
    cpu_controls = graph_buffers(ref, batch=batch)
    cpu_handles = [cpu_controls.status(i) for i in range(2)]
    # Warmed canonical bridge must also match the FP64 token reference through wraps.
    schedule = ["step"] * 5 + ["verify", "step", "bridge_step", "step", "verify"]
    for cycle, mode in enumerate(schedule):
        expected_outputs = []
        for slot in range(2):
            stage(slot, cycle)
            # Snapshot test inputs before either graph; CPU arithmetic is independent.
            cs = cpu_handles[slot]
            cs.sequence.fill_(sequence[slot])
            ref.begin(status=cs)
            slots = buf["slot_indices"][slot].cpu()
            pos = buf["positions"][slot].cpu()
            if mode == "verify":
                co = torch.empty_like(out[slot], device="cpu")
                ref.verify(
                    block[slot].cpu(),
                    *cw,
                    slot_indices=slots,
                    positions=pos[:, None] + torch.arange(q),
                    state=cpu,
                    pending=cpu_pending[slot],
                    out=co,
                    status=cs,
                )
                ref.commit(
                    *cw,
                    slot_indices=slots,
                    accepted_lengths=(torch.arange(batch, dtype=torch.int32) + sequence[slot])
                    % (q + 1),
                    state=cpu,
                    pending=cpu_pending[slot],
                    status=cs,
                    snapshot_rows=buf["snapshot_rows"][slot].cpu(),
                    snapshot_offsets=buf["snapshot_offsets"][slot].cpu(),
                    snapshot_valid=cs.valid,
                )
            else:
                co = torch.empty_like(step_out[slot], device="cpu")
                getattr(ref, mode)(
                    sx[slot].cpu(),
                    *cw,
                    slot_indices=slots,
                    positions=pos,
                    state=cpu,
                    out=co,
                    status=cs,
                )
            expected_outputs.append(co)
        # Submit BOTH graphs and their existing D2H completion before settling either.
        for slot in range(2):
            graphs[slot, mode].replay()
            mirrors[slot][0].copy_(handles[slot].records, non_blocking=True)
            mirrors[slot][1].copy_(handles[slot].validity, non_blocking=True)
            events[slot].record()
        for slot in range(2):
            events[slot].synchronize()  # engine's existing settle-side completion wait
            result = op.read_status_copy(*mirrors[slot], expected_sequence=sequence[slot])
            assert result.ok
            assert op.settle(status=handles[slot], result=result)
            assert finish(ref, cpu_handles[slot]).ok
            got = out[slot] if mode == "verify" else step_out[slot]
            assert (
                _max_ulp(
                    got.cpu().reshape(-1, hv, 128), expected_outputs[slot].reshape(-1, hv, 128)
                )
                <= ORDINARY_ULP
            )
            sequence[slot] += 2
        for f in fields(state):
            got, truth = getattr(state, f.name).cpu(), getattr(cpu, f.name)
            if f.name == "checkpoint":
                assert _max_ulp(got, truth) <= ORDINARY_ULP
            elif f.name == "ring":
                assert torch.equal(live_ring(state).cpu(), live_ring(cpu)), f.name
            else:
                assert torch.equal(got, truth), f.name
        for p, cp in zip(pending, cpu_pending, strict=True):
            same(clone(p, "cpu"), cp)


def test_graph_partial_failure_poison_collision_and_recovery():
    op, state, pending, weights, slots, pos, _ = case(device="cuda", batch=1)
    states = [state, clone(state)]
    control = graph_buffers(op, layers=2)
    first, later = control.status(0), control.status(1)
    x = projected(op, 1)
    out = torch.empty(1, op.geometry.value_width, dtype=torch.bfloat16, device="cuda")
    bad_pos = pos + 1
    # Providers with borrowed step scratch (CuTe arrivals) get it zero-initialized.
    workspace = {
        n: torch.full(s.shape, s.initial_value or 0, dtype=s.dtype, device="cuda")
        for n, s in op.workspace_spec("step", batch_size=1).items()
    }

    def body(h):
        op.begin(status=h, layer_weights=(weights, weights))
        for j in range(2):
            op.step(
                x,
                *weights,
                slot_indices=slots,
                positions=pos if j == 0 else bad_pos,
                state=states[j],
                out=out,
                status=h.for_layer(j),
                workspace=workspace,
            )
        op.end(status=h)

    first.sequence.fill_(1)
    body(first)  # warms both layer shapes and produces a partial write
    assert not finish(op, first).ok
    op.recover_reset(states=states, status=first, pending_blocks=(pending,), drained=True)
    first.sequence.fill_(2)
    op.begin(status=first)
    for j in range(2):
        op.initialize(
            slot_indices=slots, next_positions=pos, state=states[j], status=first.for_layer(j)
        )
    assert finish(op, first).ok
    first.sequence.fill_(3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body(first)
    before = [clone(s) for s in states]
    graph.replay()
    later.sequence.fill_(4)
    body(later)
    host = [
        (
            torch.empty(2, 16, dtype=torch.uint8, pin_memory=True),
            torch.empty(2, 1, dtype=torch.bool, pin_memory=True),
        )
        for _ in range(2)
    ]
    event = torch.cuda.Event()
    for h, (rec, val) in zip((first, later), host, strict=True):
        rec.copy_(h.records, non_blocking=True)
        val.copy_(h.validity, non_blocking=True)
    event.record()
    event.synchronize()
    assert states[0].metadata[0, 1].item() == pos[0].item() + 1
    same(states[1], before[1])
    saved = first.records.clone()
    first.sequence.fill_(5)
    graph.replay()  # same modulo address still owns seq3: must preserve its record
    assert torch.equal(saved, first.records)
    assert first.poison.item() & 64
    first.sequence.fill_(3)
    for h, mirror, seq in zip((first, later), host, (3, 4), strict=True):
        result = op.read_status_copy(*mirror, expected_sequence=seq)
        assert not result.ok
        op.settle(status=h, result=result)
    op.recover_reset(states=states, status=first, pending_blocks=(pending,), drained=True)
    assert not first.poison.item()
    assert all(not s.metadata[:, 4].any() for s in states)


@pytest.mark.parametrize("provider", ["triton", "cute"])
def test_zero_rows_are_a_no_op_eager_and_captured(provider):
    """B=0 on every per-row phase changes no byte of state or pending and sets no status, eager
    and replayed from a CUDA graph."""
    op, state, pending, weights, slots, pos, status = case(
        device="cuda", hk=8, hv=16, batch=2, provider=provider
    )
    before, pending_before = clone(state), clone(pending)
    zero_rows(op, state, pending, weights, slots, pos, status)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        zero_rows(op, state, pending, weights, slots, pos, status, capturable=True)
    graph.replay()
    torch.cuda.synchronize()
    same(state, before)
    same(pending, pending_before)
    assert finish(op, status).codes == (0,)


def test_portable_replay_requires_cc_8_9(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    with pytest.raises(RuntimeError, match="cc >= 8.9"):
        case(device="cuda", hk=8, hv=16, provider="triton")
