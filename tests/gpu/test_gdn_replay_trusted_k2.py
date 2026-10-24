# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Fused/unfused trusted K2 versus strict, byte for byte. Select with -k fused_k2."""

import importlib.util
from dataclasses import fields

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import _phase_launch

from ..modules._replay_case import case, clone, projected, same
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and cutlass DSL 4.8.0 required",
)


GEOMETRIES = ((16, 32), (16, 48), (8, 24))


def _policy_tile(batch, hv, q):
    """The value tile this device's phase launch policy selects (128 off-GPU)."""
    if not torch.cuda.is_available():
        return 128
    from ctikernels.gdn.replay.cute import select

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return _phase_launch.phase_launch(
        batch, hv, props.multi_processor_count, q, select(props.name)
    )[0]


# Graph phase CI core: four rows per TP1 geometry cover B/Q pairs and PDL settings
# at the policy tile; the TP2 (8,24) rows keep the engine's Q=8 at B=1 and B=4.
# The bitwise phase test below has its own smaller core.
PHASE_CASES = ci_cases(
    GEOMETRIES,
    (1, 3, 4),
    (4, 5, 8, 9, 16),
    (False, True),
    (32, 64, 128),
    core=[
        (hk, hv, b, q, pdl, _policy_tile(b, hv, q))
        for hk, hv in GEOMETRIES[:2]
        for b, q, pdl in ((1, 4, False), (1, 16, True), (4, 4, True), (4, 16, False))
    ]
    + [(8, 24, b, 8, pdl, _policy_tile(b, 24, 8)) for b, pdl in ((1, True), (4, False))],
)


def _variant_cases(batches, qs, variants):
    """Keep the Hv=32, B=4, Q=16 auto launch in CI."""
    return ci_cases(
        GEOMETRIES,
        batches,
        qs,
        variants,
        core=[(16, 32, 4, 16, 128, "auto")],
    )


def _setup(hk, hv, batch, q, *, triple=False):
    _, state, pending, weights, slots, pos, status = case(
        device="cuda", hk=hk, hv=hv, batch=batch, q=q, window=8
    )
    ops = [
        GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device="cuda",
            window=8,
            verify_length=q,
            provider="cute",
            checks=checks,
        )
        for checks in (("strict", "trusted", "trusted") if triple else ("strict", "trusted"))
    ]
    # Nonzero checkpoint plus a rotated, unflushed three-token tail exercises
    # materialization before the draft and both negative/positive raw indexing.
    state.checkpoint[:batch].copy_(
        torch.randn(batch, hv, 128, 128, generator=torch.Generator().manual_seed(812))
        .mul_(0.05)
        .to("cuda")
    )
    workspace = {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device="cuda")
        for name, spec in ops[0].workspace_spec("step", batch_size=batch).items()
        if name != "out"
    }
    out = torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda")
    for seed in range(3):
        ops[0].step(
            projected(ops[0], batch, seed=91 + seed),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            status=status,
            workspace=workspace,
        )
        pos.add_(1)
    if batch > 1:
        slots[-1] = -1
    return (
        ops,
        [state] + [clone(state) for _ in ops[1:]],
        [pending] + [clone(pending) for _ in ops[1:]],
        weights,
        slots,
        pos,
        status,
    )


def _workspace(batch):
    return {"arrivals": torch.zeros(max(1, batch), dtype=torch.int32, device="cuda")}


def _same_all(objects, compare=same):
    for obj in objects[1:]:
        compare(objects[0], obj)


def _bytes(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _copy(dst, src):
    for field in fields(dst):
        getattr(dst, field.name).copy_(getattr(src, field.name))


def _launch(monkeypatch, i, tile, variant):
    """Op 0 keeps the 8-warp, one-CTA-per-SM baseline at ``tile``; the other ops
    use ``variant``: (warps, min CTAs per SM) at ``tile``, or "auto" (policy)."""
    if i and variant == "auto":
        values = (None, None, None)
    elif i and variant is not None:
        values = (tile, *variant)
    else:
        values = (tile, None, None)
    for name, value in zip(("PHASE_TILE", "PHASE_WARPS", "PHASE_MIN_BLOCKS"), values, strict=True):
        monkeypatch.setattr(_phase_launch, name, value)


# Bitwise phase CI keeps Hv=32, B=1, tile=32 with Q=4/16 and both PDL settings, plus
# (8,24) Q=8 at B=1 tile=32 and B=3 tile=64 (the finisher path).
@pytest.mark.parametrize(
    "hk,hv,batch,q,pdl,tile",
    ci_cases(
        GEOMETRIES,
        (1, 3, 4),
        (4, 5, 8, 9, 16),
        (False, True),
        (32, 64, 128),
        core=[
            (16, 32, 1, 16, True, 32),
            (16, 32, 1, 4, False, 32),
            (8, 24, 1, 8, True, 32),
            (8, 24, 3, 8, False, 64),
        ],
    ),
)
def test_fused_k2_all_phases_bitwise(hk, hv, batch, q, pdl, tile, monkeypatch):
    _all_phases_bitwise(hk, hv, batch, q, pdl, tile, None, monkeypatch)


@pytest.mark.parametrize(
    "hk,hv,batch,q,tile,variant",
    _variant_cases(
        (1, 4),
        (4, 9, 16),
        (
            (64, (16, 1)),
            (128, (16, 1)),
            (64, (8, 2)),
            (128, (8, 2)),
            (32, (16, 1)),
            (128, "auto"),
        ),
    ),
)
def test_fused_k2_launch_variants_bitwise(hk, hv, batch, q, tile, variant, monkeypatch):
    """16-warp CTAs and 2-CTA/SM launch bounds (R3b policy) match 8 warps bitwise."""
    _all_phases_bitwise(hk, hv, batch, q, True, tile, variant, monkeypatch)


def _all_phases_bitwise(hk, hv, batch, q, pdl, tile, variant, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    _launch(monkeypatch, 0, tile, None)
    ops, states, pending, weights, slots, pos, status = _setup(hk, hv, batch, q, triple=True)
    workspaces = [_workspace(batch) for _ in ops]
    snapshots = torch.arange(batch, dtype=torch.int32, device="cuda") + states[0].metadata.shape[0]
    pairs = torch.stack((slots, snapshots), 1)
    accepted = torch.empty(batch, dtype=torch.int32, device="cuda")
    offsets = torch.empty(batch, dtype=torch.int64, device="cuda")

    def call(mode, *, transaction=False, output=False, capture=False, **kw):
        outputs, validities = [], []
        for i, op in enumerate(ops):
            monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
            monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", i == 2)
            _launch(monkeypatch, i, tile, variant)
            args = dict(kw, state=states[i], status=status if i == 0 else None)
            if i and mode in ("commit", "bridge_step", "export_snapshot", "flush"):
                args["workspace"] = workspaces[i]
            if transaction:
                args["pending"] = pending[i]
            if output:
                shape = (batch, q, hv * 128) if mode == "verify" else (batch, hv * 128)
                outputs.append(torch.full(shape, 7, dtype=torch.bfloat16, device="cuda"))
                args["out"] = outputs[-1]
            if capture:
                validities.append(torch.ones(batch, dtype=torch.bool, device="cuda"))
                args["snapshot_valid"] = validities[-1]
            if mode in ("verify", "bridge_step"):
                op_x = args.pop("projected")
                getattr(op, mode)(op_x, *weights, **args)
            elif mode == "reject":
                op.reject(**args)
            else:
                getattr(op, mode)(*weights, **args)
        _same_all(states)
        _same_all(pending)
        for tensors in (outputs, validities):
            if tensors:
                _same_all(tensors, _bytes)
        assert all(torch.count_nonzero(w["arrivals"]).item() == 0 for w in workspaces)
        assert status.poison.item() == 0
        assert status.record.view(torch.int32)[2].item() == 0

    for cycle, m in enumerate((0, q // 2, q)):
        # Each active row visits zero, partial and full acceptance; rows differ.
        accepted.copy_(
            torch.tensor(
                [(0, q // 2, q)[(cycle + b) % 3] for b in range(batch)],
                dtype=torch.int32,
                device="cuda",
            )
        )
        offsets.fill_(m)
        call(
            "verify",
            transaction=True,
            output=True,
            projected=projected(ops[0], batch, length=q, seed=712 + cycle),
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(q, device="cuda"),
        )
        call(
            "export_snapshot",
            transaction=True,
            capture=True,
            pairs=pairs,
            accepted_lengths=accepted,
            offsets=offsets,
        )
        call(
            "commit",
            transaction=True,
            capture=True,
            slot_indices=slots,
            accepted_lengths=accepted,
            snapshot_rows=snapshots,
            snapshot_offsets=offsets,
        )
        pos.add_(accepted)
        call(
            "bridge_step",
            output=True,
            slot_indices=slots,
            positions=pos,
            projected=projected(ops[0], batch, seed=803 + cycle),
        )
        pos.add_(1)
        call("flush", slot_indices=slots)
        call("export_snapshot", capture=True, pairs=pairs)
        call(
            "verify",
            transaction=True,
            output=True,
            projected=projected(ops[0], batch, length=q, seed=900 + cycle),
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(q, device="cuda"),
        )
        call("reject", transaction=True, slot_indices=slots)
    # A one-row batch also exercises the fully masked invocation.
    slots.fill_(-1)
    call(
        "verify",
        transaction=True,
        output=True,
        projected=projected(ops[0], batch, length=q),
        slot_indices=slots,
        positions=pos[:, None] + torch.arange(q, device="cuda"),
    )
    call(
        "commit",
        transaction=True,
        capture=True,
        slot_indices=slots,
        accepted_lengths=accepted,
        snapshot_rows=snapshots,
        snapshot_offsets=offsets,
    )


@pytest.mark.parametrize("hk,hv,batch,q,pdl,tile", PHASE_CASES)
def test_fused_k2_graph_device_acceptance(hk, hv, batch, q, pdl, tile, monkeypatch):
    _graph_device_acceptance(hk, hv, batch, q, pdl, tile, None, monkeypatch)


@pytest.mark.parametrize(
    "hk,hv,batch,q,tile,variant",
    _variant_cases((3, 4), (4, 16), ((128, (16, 1)), (64, (8, 2)), (128, "auto"))),
)
def test_fused_k2_graph_launch_variants(hk, hv, batch, q, tile, variant, monkeypatch):
    _graph_device_acceptance(hk, hv, batch, q, True, tile, variant, monkeypatch)


def _graph_device_acceptance(hk, hv, batch, q, pdl, tile, variant, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    _launch(monkeypatch, 0, tile, None)
    ops, states, pending, weights, slots, pos, status = _setup(hk, hv, batch, q, triple=True)
    workspaces = [_workspace(batch) for _ in ops]
    initial, initial_pending = clone(states[0]), clone(pending[0])
    drafts = projected(ops[0], batch, length=q)
    positions = [(pos[:, None] + torch.arange(q, device="cuda")).clone() for _ in ops]
    initial_positions = positions[0].clone()
    outputs = [torch.empty(batch, q, hv * 128, dtype=torch.bfloat16, device="cuda") for _ in ops]
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    snapshots = torch.arange(batch, dtype=torch.int32, device="cuda") + states[0].metadata.shape[0]
    offsets = torch.full((batch,), q // 2, dtype=torch.int64, device="cuda")
    valid = [torch.zeros(batch, dtype=torch.bool, device="cuda") for _ in ops]

    def round_(i):
        monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
        monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", i == 2)
        _launch(monkeypatch, i, tile, variant)
        ops[i].verify(
            drafts,
            *weights,
            slot_indices=slots,
            positions=positions[i],
            state=states[i],
            pending=pending[i],
            out=outputs[i],
            status=status if i == 0 else None,
        )
        ops[i].commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=accepted,
            state=states[i],
            pending=pending[i],
            status=status if i == 0 else None,
            snapshot_rows=snapshots,
            snapshot_offsets=offsets,
            snapshot_valid=valid[i],
            workspace=workspaces[i] if i else None,
        )
        positions[i].add_(accepted[:, None])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            round_(i)
    torch.cuda.current_stream().wait_stream(stream)
    graphs = [torch.cuda.CUDAGraph() for _ in ops[1:]]
    for i, graph in enumerate(graphs, 1):
        with torch.cuda.graph(graph, stream=stream):
            round_(i)
    for i in range(3):
        _copy(states[i], initial)
        _copy(pending[i], initial_pending)
        positions[i].copy_(initial_positions)
    for m in (0, q, q // 2, 1, q, 0):
        # Fill and row updates are device operations on the graph's input buffer.
        accepted.fill_(m)
        if batch > 1:
            accepted[1].fill_(q - m)
        round_(0)
        for graph in graphs:
            graph.replay()
        _same_all(outputs, _bytes)
        _same_all(valid, _bytes)
        _same_all(positions, _bytes)
        assert all(torch.count_nonzero(w["arrivals"]).item() == 0 for w in workspaces)
        _same_all(states)
        _same_all(pending)
    assert status.poison.item() == 0


@pytest.mark.parametrize("pdl", [pytest.param(False, marks=pytest.mark.exhaustive), True])
def test_fused_k2_bounds_and_pending_search(pdl, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    ops, states, pending, weights, slots, pos, status = _setup(16, 32, 3, 5)
    workspace = _workspace(5)
    outputs = [torch.empty(3, 5, 4096, dtype=torch.bfloat16, device="cuda") for _ in ops]
    drafts = projected(ops[0], 3, length=5)
    for i, op in enumerate(ops):
        monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
        op.verify(
            drafts,
            *weights,
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(5, device="cuda"),
            state=states[i],
            pending=pending[i],
            out=outputs[i],
            status=status if i == 0 else None,
        )
    accepted = [
        torch.tensor(v, dtype=torch.int32, device="cuda") for v in ([0, 5, 0], [-9, 99, -1])
    ]
    # Repeated, reordered sources require per-pair pending lookup. Invalid pair
    # targets in trusted mode act like the strict reference's -1 target.
    r = states[0].metadata.shape[0]
    pairs = [
        torch.tensor(v, dtype=torch.int32, device="cuda")
        for v in (
            [[1, -1], [1, r], [0, r + 1], [1, r + 2], [2, -1]],
            [[1, r + 99], [1, r], [0, r + 1], [1, r + 2], [2, r + 3]],
        )
    ]
    offsets = torch.tensor([0, 3, 0, 5, 0], dtype=torch.int64, device="cuda")
    valid = [torch.ones(5, dtype=torch.bool, device="cuda") for _ in ops]
    for i, op in enumerate(ops):
        monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
        op.export_snapshot(
            *weights,
            pairs=pairs[i],
            state=states[i],
            pending=pending[i],
            accepted_lengths=accepted[i],
            offsets=offsets,
            snapshot_valid=valid[i],
            workspace=workspace if i else None,
            status=status if i == 0 else None,
        )
    same(*states)
    same(*pending)
    _bytes(*valid)
    # Strict sees the canonicalized accepted length recorded by export.
    for i, op in enumerate(ops):
        monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=accepted[i],
            state=states[i],
            pending=pending[i],
            status=status if i == 0 else None,
            workspace={"arrivals": workspace["arrivals"][:3]} if i else None,
        )
    same(*states)
    same(*pending)
    # Out-of-range source rows are masked, including values other than -1.
    for i, op in enumerate(ops):
        monkeypatch.setattr(_phase_launch, "PDL", pdl if i else False)
        masked = torch.tensor(
            [-1, -1, -1] if i == 0 else [-2, r, r + 99], dtype=torch.int32, device="cuda"
        )
        op.verify(
            drafts,
            *weights,
            slot_indices=masked,
            positions=pos[:, None] + torch.arange(5, device="cuda"),
            state=states[i],
            pending=pending[i],
            out=outputs[i],
            status=status if i == 0 else None,
        )
    _bytes(*outputs)
    same(*states)
    same(*pending)


@pytest.mark.parametrize(
    "hk,hv,q,pdl",
    ci_cases(
        GEOMETRIES,
        (4, 16),
        (False, True),
        core=[(16, 32, 4, True)],
    ),
)
def test_fused_k2_shared_counter_two_graphs_many_rounds(hk, hv, q, pdl, monkeypatch):
    """One counter buffer shared by a B=3 and a B=1 graph on disjoint slots.

    Rounds are issued back to back on one stream with no host synchronization
    between them (two rounds in flight), replayed many times with device-side
    acceptance changes, and compared byte for byte with strict eager rounds.
    """
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    ops, states, pending, weights, slots, pos, status = _setup(hk, hv, 4, q)
    strict, fused = ops
    groups = (slice(0, 3), slice(3, 4))
    slot_sets = [torch.arange(4, dtype=torch.int32, device="cuda")[g].contiguous() for g in groups]
    # Row 0 is shared by both graphs; eager warm-up uses row 1 (a 16-byte offset view).
    counters = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    shared = counters[0]
    row = [shared]
    drafts = projected(fused, 4, length=q)
    accepted = torch.zeros(4, dtype=torch.int32, device="cuda")
    base = pos.clone()
    positions = [
        [(base[g][:, None] + torch.arange(q, device="cuda")).contiguous() for g in groups]
        for _ in ops
    ]
    outputs = [
        [
            torch.empty(g.stop - g.start, q, hv * 128, dtype=torch.bfloat16, device="cuda")
            for g in groups
        ]
        for _ in ops
    ]

    def views(i, k):
        g = groups[k]
        p = pending[i]
        return type(p)(p.raw[g], p.metadata[g])

    def round_(i, k):
        g = groups[k]
        monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", i == 1)
        ops[i].verify(
            drafts[g],
            *weights,
            slot_indices=slot_sets[k],
            positions=positions[i][k],
            state=states[i],
            pending=views(i, k),
            out=outputs[i][k],
            status=status if i == 0 else None,
        )
        ops[i].commit(
            *weights,
            slot_indices=slot_sets[k],
            accepted_lengths=accepted[g],
            state=states[i],
            pending=views(i, k),
            status=status if i == 0 else None,
            workspace={"arrivals": row[0][: g.stop - g.start]} if i else None,
        )
        positions[i][k].add_(accepted[g][:, None])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    initial, initial_pending = clone(states[1]), clone(pending[1])
    initial_positions = [p.clone() for p in positions[1]]
    row[0] = counters[1]
    with torch.cuda.stream(stream):
        round_(1, 0)
        round_(1, 1)
    torch.cuda.current_stream().wait_stream(stream)
    row[0] = shared
    graphs = []
    for k in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            round_(1, k)
        graphs.append(graph)
    _copy(states[1], initial)
    _copy(pending[1], initial_pending)
    for p, p0 in zip(positions[1], initial_positions, strict=True):
        p.copy_(p0)
    schedule = [(m, k) for m in (0, q, q // 2, 1, q - 1, 2, q, 0) * 3 for k in (0, 1)]
    for m, k in schedule:
        # Device-side acceptance update, then strict eager and the fused graph,
        # with no host synchronization between successive rounds.
        accepted.fill_(m)
        accepted[0].fill_(q - m)
        round_(0, k)
        graphs[k].replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(counters).item() == 0
    for k in range(2):
        _bytes(outputs[0][k], outputs[1][k])
        _bytes(positions[0][k], positions[1][k])
    same(states[0], states[1])
    same(pending[0], pending[1])
    assert status.poison.item() == 0


@pytest.mark.parametrize(
    "hk,hv,q,pdl",
    ci_cases(
        GEOMETRIES,
        (4, 16),
        (False, True),
        core=[(hk, hv, q, pdl) for hk, hv in GEOMETRIES for q, pdl in ((4, True), (16, False))],
    ),
)
def test_fused_k2_concurrent_rounds_two_streams(hk, hv, q, pdl, monkeypatch):
    """Two rounds in flight on two streams: a B=3 and a B=1 graph on disjoint slots
    of one pool, each with its own row of a [2, 3] counter buffer (row 1 starts at
    a 12-byte offset and is first used at capture; eager warm-up used row 0 only).
    Streams order only through device events; no host synchronization until the
    end. Outputs, positions, pool and pending match strict eager rounds bitwise."""
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    ops, states, pending, weights, slots, pos, status = _setup(hk, hv, 4, q)
    groups = (slice(0, 3), slice(3, 4))
    slot_sets = [torch.arange(4, dtype=torch.int32, device="cuda")[g].contiguous() for g in groups]
    counters = torch.zeros(2, 3, dtype=torch.int32, device="cuda")
    assert counters[1].data_ptr() % 16 == 12
    rows = [counters[0], counters[1]]
    drafts = projected(ops[1], 4, length=q)
    accepted = [torch.zeros(g.stop - g.start, dtype=torch.int32, device="cuda") for g in groups]
    base = pos.clone()
    positions = [
        [(base[g][:, None] + torch.arange(q, device="cuda")).contiguous() for g in groups]
        for _ in ops
    ]
    outputs = [
        [
            torch.empty(g.stop - g.start, q, hv * 128, dtype=torch.bfloat16, device="cuda")
            for g in groups
        ]
        for _ in ops
    ]

    def views(i, k):
        p = pending[i]
        return type(p)(p.raw[groups[k]], p.metadata[groups[k]])

    def round_(i, k, row):
        g = groups[k]
        monkeypatch.setattr(_phase_launch, "FUSED_PUBLISH", i == 1)
        ops[i].verify(
            drafts[g],
            *weights,
            slot_indices=slot_sets[k],
            positions=positions[i][k],
            state=states[i],
            pending=views(i, k),
            out=outputs[i][k],
            status=status if i == 0 else None,
        )
        ops[i].commit(
            *weights,
            slot_indices=slot_sets[k],
            accepted_lengths=accepted[k],
            state=states[i],
            pending=views(i, k),
            status=status if i == 0 else None,
            workspace={"arrivals": row[: g.stop - g.start]} if i else None,
        )
        positions[i][k].add_(accepted[k][:, None])

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    main = torch.cuda.current_stream()
    initial, initial_pending = clone(states[1]), clone(pending[1])
    initial_positions = [p.clone() for p in positions[1]]
    streams[0].wait_stream(main)
    with torch.cuda.stream(streams[0]):
        round_(1, 0, rows[0])
        round_(1, 1, rows[0])
    main.wait_stream(streams[0])
    graphs = []
    for k in range(2):
        graph = torch.cuda.CUDAGraph()
        streams[k].wait_stream(main)
        with torch.cuda.graph(graph, stream=streams[k]):
            round_(1, k, rows[k])
        graphs.append(graph)
        main.wait_stream(streams[k])
    _copy(states[1], initial)
    _copy(pending[1], initial_pending)
    for p, p0 in zip(positions[1], initial_positions, strict=True):
        p.copy_(p0)
    for streak, m in enumerate((0, q, q // 2, 1, q - 1, 2, q, 0) * 2):
        for k in range(2):
            # The acceptance buffer of group k is rewritten only after stream k's
            # previous round consumed it (device event, not a host fence).
            main.wait_stream(streams[k])
            accepted[k].fill_(m if (streak + k) % 2 else q - m)
            round_(0, k, None)
            streams[k].wait_stream(main)
            with torch.cuda.stream(streams[k]):
                graphs[k].replay()
    for stream in streams:
        main.wait_stream(stream)
    torch.cuda.synchronize()
    assert torch.count_nonzero(counters).item() == 0
    for k in range(2):
        _bytes(outputs[0][k], outputs[1][k])
        _bytes(positions[0][k], positions[1][k])
    same(states[0], states[1])
    same(pending[0], pending[1])
    assert status.poison.item() == 0


def test_fused_k2_graph_buffer_spec_gives_each_depth_its_counter():
    op = GatedDeltaNetReplay(
        num_k_heads=16, num_v_heads=32, device="cuda", window=8, verify_length=8, provider="cute"
    )
    spec = op.graph_data_spec(batch_size=3, depth=2)
    assert tuple(spec["phase_arrivals"].shape) == (2, 3)
