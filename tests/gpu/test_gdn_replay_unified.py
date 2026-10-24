# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""GDNDEC2 routing, deterministic decode and snapshot exactness and reset on undefined memory."""

import importlib.util
from dataclasses import fields, replace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import Config

from ..modules._replay_case import case, clone, finish, projected, restart, same, verify

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and cutlass DSL 4.8.0 required",
)

# TP2 per-rank (8,24) is exhaustive here except the verify/commit transaction test.
GEOMETRIES = [(16, 32), (16, 48), pytest.param(8, 24, marks=pytest.mark.exhaustive)]


def allocate(specs):
    return {
        k: torch.full(s.shape, s.initial_value or 0, dtype=s.dtype, device="cuda")
        for k, s in specs.items()
    }


def restore(state, saved):
    for f in fields(state):
        getattr(state, f.name).copy_(getattr(saved, f.name))


def bitwise(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def setup(hk, hv, batch, mode="auto", *, checks="trusted", cute_config=None):
    _, state, _, weights, slots, pos, status = case(
        device="cuda", hk=hk, hv=hv, batch=batch, window=4
    )
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        provider="cute",
        window=4,
        verify_length=8,
        mode=mode,
        checks=checks,
        cute_config=cute_config,
    )
    if op._ringless:
        state = replace(state, ring=state.ring[:, :0].contiguous())
    return op, state, weights, slots, pos, status


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_product_step_matches_direct_eager_and_graph(hk, hv, batch, padded):
    op, state, weights, slots, pos, _ = setup(
        hk, hv, batch, cute_config=Config(provider="product", immediate=True)
    )
    assert op.decode_route(batch) == "product"
    if padded:
        slots[-1] = -1
    direct = GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda")
    expected = clone(state)
    before = clone(state)
    original_pos = pos.clone()
    x = projected(op, batch)
    out = torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda")
    reference = torch.empty_like(out)
    assert set(op.workspace_spec("step", batch_size=batch)) == {"out"}
    spec = op.state_spec(num_slots=state.metadata.shape[0], num_snapshot_rows=0)
    assert op.bytes_per_slot()["ring"] == spec["ring"].payload_bytes // state.metadata.shape[0] == 0

    def step():
        op.step(x, *weights, slot_indices=slots, positions=pos, state=state, out=out)

    def baseline():
        assert (
            direct(x, *weights, slots, expected.checkpoint, expected.conv_history, out=reference)
            is reference
        )
        active = slots[slots >= 0].long()
        expected.metadata[active, 1] += 1
        expected.metadata[active, 0] = expected.metadata[active, 1]

    def compare():
        bitwise(out, reference)
        same(state, expected)
        bitwise(pos, original_pos)
        bitwise(state.ring, before.ring)
        if padded:
            assert not out[-1].any()

    for k in range(1, 4):
        step()
        baseline()
        compare()
        active = slots[slots >= 0].long()
        for field in (0, 1):
            bitwise(state.metadata[active, field], before.metadata[active, field] + k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for t in range(4):
        slots.copy_(
            (torch.arange(batch, device="cuda", dtype=torch.int32) + t) % state.metadata.shape[0]
        )
        if padded:
            slots[-1] = -1
        x.copy_(projected(op, batch, seed=101 + t))
        graph.replay()
        baseline()
        torch.cuda.synchronize()
        compare()
    flag = torch.empty(1, dtype=torch.bool, device="cuda")
    op.ring_empty(state, slots, flag)
    assert flag.item()
    saved = clone(state)
    op.flush(
        *weights,
        slot_indices=slots,
        state=state,
        status=None,
        workspace=allocate(op.workspace_spec("flush", batch_size=(slots).shape[0])),
    )
    same(state, saved)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("mode", ["auto", "immediate", "delayed"])
def test_run_to_run_bitwise(hk, hv, batch, mode):
    a, state_a, weights, slots, pos, _ = setup(hk, hv, batch, mode)
    b, state_b, _, _, _, _ = setup(hk, hv, batch, mode)
    wa, wb = (allocate(op.workspace_spec("step", batch_size=batch)) for op in (a, b))
    for t in range(2 * a.window + 3):
        x = projected(a, batch, seed=500 + t)
        for op, state, workspace in ((a, state_a, wa), (b, state_b, wb)):
            op.step(
                x,
                *weights,
                slot_indices=slots,
                positions=pos,
                state=state,
                out=workspace["out"],
                workspace=workspace,
            )
        bitwise(wa["out"], wb["out"])
        same(state_a, state_b)
        pos.add_(1)


def handoff(op, state, slots, pos, status):
    generator = torch.Generator().manual_seed(912)
    canonical = (
        torch.randn(slots.numel(), op.geometry.v_heads, 128, 128, generator=generator) * 0.01
    ).to("cuda")
    history = (torch.randn(slots.numel(), 3, op.geometry.conv_dim, generator=generator) * 0.1).to(
        device="cuda", dtype=torch.bfloat16
    )
    op.handoff(
        canonical_state=canonical,
        history=history,
        next_positions=pos,
        slot_indices=slots,
        state=state,
        status=status,
    )


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("forced", [False, True])
def test_strict_route_honours_earlier_layer_poison(hk, hv, batch, forced):
    config = Config(provider="product", immediate=True) if forced else None
    op, state, weights, slots, pos, status = setup(
        hk, hv, batch, checks="strict", cute_config=config
    )
    assert op.decode_route(batch) != "product"
    if forced:
        assert op.decode_route(batch) == "immediate"
    workspace = allocate(op.workspace_spec("step", batch_size=batch))
    handoff(op, state, slots, pos, status)
    saved = clone(state)
    # Another layer failed after begin, before this layer's step.
    status.poison.fill_(1)
    op.step(
        projected(op, batch),
        *weights,
        slot_indices=slots,
        positions=pos,
        state=state,
        out=workspace["out"],
        workspace=workspace,
        status=status,
    )
    same(state, saved)
    assert status.poison.item() != 0


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_strict_forced_product_matches_trusted_product(hk, hv, batch):
    config = Config(provider="product", immediate=True)
    strict, state, weights, slots, pos, status = setup(
        hk, hv, batch, checks="strict", cute_config=config
    )
    trusted, _, _, _, _, _ = setup(hk, hv, batch, cute_config=config)
    assert strict.decode_route(batch) == "immediate"
    assert trusted.decode_route(batch) == "product"
    handoff(strict, state, slots, pos, status)
    expected = clone(state)
    workspace = allocate(strict.workspace_spec("step", batch_size=batch))
    out = torch.empty_like(workspace["out"])
    for t in range(3):
        x = projected(strict, batch, seed=600 + t)
        strict.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=workspace["out"],
            workspace=workspace,
            status=status,
        )
        trusted.step(x, *weights, slot_indices=slots, positions=pos, state=expected, out=out)
        bitwise(workspace["out"], out)
        same(state, expected)
        pos.add_(1)
    assert finish(strict, status).ok


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48), (8, 24)])
@pytest.mark.parametrize("batch", [1, 4])
def test_product_snapshot_import_verify_and_commit(hk, hv, batch):
    op, state, weights, slots, pos, status = setup(
        hk, hv, batch, cute_config=Config(provider="product", immediate=True)
    )
    handoff(op, state, slots, pos, status)
    out = allocate(op.workspace_spec("step", batch_size=batch))["out"]
    for t in range(3):
        op.step(
            projected(op, batch, seed=700 + t),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
        )
        pos.add_(1)
    for field in (0, 1):
        bitwise(state.metadata[slots.long(), field], pos)
    saved = clone(state)
    rows = state.metadata.shape[0]
    valid = torch.zeros(1, dtype=torch.bool, device="cuda")
    op.export_snapshot(
        *weights,
        pairs=torch.tensor([[0, rows]], dtype=torch.int32, device="cuda"),
        state=state,
        status=status,
        snapshot_valid=valid,
        arithmetic_batch=batch,
        workspace=allocate(
            op.workspace_spec(
                "export_snapshot",
                batch_size=(torch.tensor([[0, rows]], dtype=torch.int32, device="cuda")).shape[0],
            )
        ),
    )
    assert valid.item()
    live_unchanged(state, saved)
    assert restart(op, status).ok
    imported = clone(state)
    imported_slots = slots.clone()
    imported_slots[0] = batch
    op.import_snapshot(
        pairs=torch.tensor([[rows, batch]], dtype=torch.int32, device="cuda"),
        state=imported,
        status=status,
    )
    for field in (0, 1):
        bitwise(imported.metadata[imported_slots.long(), field], pos)
    assert restart(op, status).ok
    outputs = []
    for target, selected in ((state, slots), (imported, imported_slots)):
        workspace = allocate(op.workspace_spec("verify", batch_size=batch))
        pending = op.create_pending(
            raw=workspace["pending.raw"], metadata=workspace["pending.metadata"]
        )
        _, result = verify(op, target, pending, weights, selected, pos, status)
        outputs.append(result)
        assert status.code.item() == status.poison.item() == 0
        op.commit(
            *weights,
            slot_indices=selected,
            accepted_lengths=torch.full((batch,), 8, dtype=torch.int32, device="cuda"),
            state=target,
            pending=pending,
            status=status,
            workspace=allocate(op.workspace_spec("commit", batch_size=(selected).shape[0])),
        )
        assert restart(op, status).ok
        for field in (0, 1):
            bitwise(target.metadata[selected.long(), field], pos + 8)
    bitwise(*outputs)
    for name in ("checkpoint", "conv_history"):
        bitwise(getattr(state, name)[slots.long()], getattr(imported, name)[imported_slots.long()])
    assert finish(op, status).ok


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("alias", ["projected", "checkpoint"])
def test_product_rejects_borrowed_output_alias(hk, hv, alias, monkeypatch):
    op, state, weights, slots, pos, _ = setup(
        hk, hv, 1, cute_config=Config(provider="product", immediate=True)
    )
    x = projected(op, 1)
    out = (
        x[:, : hv * 128]
        if alias == "projected"
        else state.checkpoint.view(torch.bfloat16).reshape(-1)[: hv * 128].view(1, -1)
    )
    direct = GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda")
    before = state.metadata.clone()

    def launch(*args, **kwargs):
        pytest.fail("decode launched before output alias rejection")

    monkeypatch.setattr(direct, "_kernel", launch)
    with pytest.raises(ValueError, match="out must not overlap"):
        direct(x, *weights, slots, state.checkpoint, state.conv_history, out=out)
    op._product_decode = direct
    with pytest.raises(ValueError, match="overlapping buffers"):
        op.step(x, *weights, slot_indices=slots, positions=pos, state=state, out=out)
    # Rejection precedes the metadata branch: no position advanced.
    torch.cuda.synchronize()
    bitwise(state.metadata, before)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
def test_product_rejects_output_positions_alias(hk, hv):
    op, state, weights, slots, pos, _ = setup(
        hk, hv, 1, cute_config=Config(provider="product", immediate=True)
    )
    x = projected(op, 1)
    out = torch.zeros(1, hv * 128, dtype=torch.bfloat16, device="cuda")
    positions = out.view(torch.int64).reshape(-1)[:1]
    positions.copy_(pos)
    before, kept = state.metadata.clone(), positions.clone()
    with pytest.raises(ValueError, match="overlapping buffers: out, positions"):
        op.step(x, *weights, slot_indices=slots, positions=positions, state=state, out=out)
    torch.cuda.synchronize()
    bitwise(state.metadata, before)
    bitwise(positions, kept)


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
def test_product_rejects_positions_state_alias(hk, hv):
    op, state, weights, slots, pos, _ = setup(
        hk, hv, 1, cute_config=Config(provider="product", immediate=True)
    )
    x = projected(op, 1)
    out = torch.zeros(1, hv * 128, dtype=torch.bfloat16, device="cuda")
    positions = state.metadata[0, :1]
    before, out_before = state.metadata.clone(), out.clone()
    with pytest.raises(ValueError, match="overlapping buffers: metadata, positions"):
        op.step(x, *weights, slot_indices=slots, positions=positions, state=state, out=out)
    torch.cuda.synchronize()
    bitwise(state.metadata, before)
    bitwise(out, out_before)


def step(op, state, weights, slots, pos, workspace, x):
    op.step(
        x,
        *weights,
        slot_indices=slots,
        positions=pos,
        state=state,
        out=workspace["out"],
        workspace=workspace,
    )

    pos.add_((slots >= 0).to(torch.int64))


def live_unchanged(state, saved):
    rows = state.metadata.shape[0]
    bitwise(state.checkpoint[:rows], saved.checkpoint[:rows])
    bitwise(state.conv_history[:rows], saved.conv_history[:rows])
    for name in ("ring", "metadata", "pool_epoch"):
        bitwise(getattr(state, name), getattr(saved, name))


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("steps", [1, 3, 4, 5, 11])
def test_delayed_export_exactness_and_import_continuation(hk, hv, batch, steps):
    op, state, weights, slots, pos, status = setup(hk, hv, batch, "delayed")
    immediate, written, _, _, ipos, istatus = setup(hk, hv, batch, "immediate")
    handoff(op, state, slots, pos, status)
    handoff(immediate, written, slots, ipos, istatus)
    workspace = allocate(op.workspace_spec("step", batch_size=batch))
    iw = allocate(immediate.workspace_spec("step", batch_size=batch))
    flag = torch.empty(1, dtype=torch.bool, device="cuda")
    op.ring_empty(state, slots, flag)
    assert flag.item()
    initial, initial_pos = clone(state), pos.clone()
    x = projected(op, batch)
    # Warm and capture the decode call; export stays eager after graph replay.
    step(op, state, weights, slots, pos, workspace, x)
    restore(state, initial)
    pos.copy_(initial_pos)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step(op, state, weights, slots, pos, workspace, x)
    restore(state, initial)
    pos.copy_(initial_pos)
    for t in range(steps):
        x.copy_(projected(op, batch, seed=800 + t))
        graph.replay()
        step(immediate, written, weights, slots, ipos, iw, x)
        bitwise(workspace["out"], iw["out"])
        if t == 0:
            op.ring_empty(state, slots, flag)
            assert not flag.item()
            op.ring_empty(state, torch.full_like(slots, -1), flag)
            assert flag.item()
    saved, flushed = clone(state), clone(state)
    rows = state.metadata.shape[0]
    assert op.bytes_per_slot()["ring"] == state.ring[0].numel() * state.ring.element_size()
    op.flush(
        *weights,
        slot_indices=slots,
        state=flushed,
        status=status,
        workspace=allocate(op.workspace_spec("flush", batch_size=(slots).shape[0])),
    )
    op.ring_empty(flushed, slots, flag)
    assert flag.item()
    for selected, destinations in (
        (slots, slots + rows),
        (slots[:1], slots[:1] + rows + batch),
    ):
        pairs = torch.stack((selected, destinations), 1)
        valid = torch.zeros(selected.numel(), dtype=torch.bool, device="cuda")
        op.export_snapshot(
            *weights,
            pairs=pairs,
            state=state,
            status=status,
            snapshot_valid=valid,
            arithmetic_batch=batch,
            workspace=allocate(op.workspace_spec("export_snapshot", batch_size=(pairs).shape[0])),
        )
        assert valid.all().item()
        live_unchanged(state, saved)
        for source, destination in pairs.tolist():
            for name in ("checkpoint", "conv_history"):
                exported = getattr(state, name)[destination]
                bitwise(exported, getattr(written, name)[source])
                bitwise(exported, getattr(flushed, name)[source])
    assert restart(op, status).ok
    assert finish(immediate, istatus).ok

    # Restore row zero into an unused live slot in an independent pool, preserving
    # the original B for both continuations even when only one row was exported.
    imported = clone(state)
    imported_slots = slots.clone()
    imported_slots[0] = batch
    pairs = torch.tensor([[rows + batch, batch]], dtype=torch.int32, device="cuda")
    op.import_snapshot(pairs=pairs, state=imported, status=status)
    op.ring_empty(imported, imported_slots[:1], flag)
    assert flag.item()
    imported_pos = pos.clone()
    imported_workspace = allocate(op.workspace_spec("step", batch_size=batch))
    for t in range(op.window + 2):
        x.copy_(projected(op, batch, seed=1000 + t))
        step(op, state, weights, slots, pos, workspace, x)
        step(op, imported, weights, imported_slots, imported_pos, imported_workspace, x)
        bitwise(workspace["out"], imported_workspace["out"])
    for target, selected in ((state, slots), (imported, imported_slots)):
        op.flush(
            *weights,
            slot_indices=selected,
            state=target,
            status=status,
            workspace=allocate(op.workspace_spec("flush", batch_size=(selected).shape[0])),
        )
    for name in ("checkpoint", "conv_history"):
        bitwise(getattr(state, name)[slots.long()], getattr(imported, name)[imported_slots.long()])
    # Import reseeds cursor rotation and generation; canonical state/positions match.
    for field in (0, 1, 3, 4, 6, 7):
        bitwise(
            state.metadata[slots.long(), field], imported.metadata[imported_slots.long(), field]
        )
    assert finish(op, status).ok


def random_bytes(tensor):
    raw = tensor.view(torch.uint8)
    raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8, device=raw.device))


@pytest.mark.parametrize("hk,hv", GEOMETRIES)
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_reset_on_garbage(hk, hv, batch):
    op, state, weights, slots, pos, status = setup(hk, hv, batch, "delayed")
    fresh_op, fresh, _, _, fresh_pos, fresh_status = setup(hk, hv, batch, "delayed")
    workspace = allocate(op.workspace_spec("step", batch_size=batch))
    fresh_workspace = allocate(fresh_op.workspace_spec("step", batch_size=batch))
    pending_workspace = allocate(op.workspace_spec("verify", batch_size=batch))
    pending = op.create_pending(
        raw=pending_workspace["pending.raw"], metadata=pending_workspace["pending.metadata"]
    )
    for name in op.state_spec(
        num_slots=state.metadata.shape[0], num_snapshot_rows=state.snapshot_metadata.shape[0]
    ):
        random_bytes(getattr(state, name))
    random_bytes(pending.metadata)
    random_bytes(workspace["arrivals"])
    garbage = clone(state)
    old_generation, old_epoch = state.metadata[:, 5].clone(), state.pool_epoch.clone()
    op.reset(
        state=state,
        weight_revision=7,
        status=status,
        pending_blocks=(pending,),
        workspaces=(workspace,),
    )
    fresh_op.reset(
        state=fresh, weight_revision=7, status=fresh_status, workspaces=(fresh_workspace,)
    )
    assert finish(op, status).ok
    assert finish(fresh_op, fresh_status).ok
    assert not workspace["arrivals"].any() and not pending.metadata.any()
    assert not state.snapshot_metadata.any() and not state.metadata[:, :5].any()
    for name in ("checkpoint", "conv_history", "ring"):
        bitwise(getattr(state, name), getattr(garbage, name))
    bitwise(state.pool_epoch, old_epoch + 1)
    bitwise(state.metadata[:, 5], old_generation + 1)
    assert (state.metadata[:, 6] == 7).all()
    assert (state.metadata[:, 7] == state.pool_epoch[0]).all()
    for current, pool, positions, handle in (
        (op, state, pos, status),
        (fresh_op, fresh, fresh_pos, fresh_status),
    ):
        handle.sequence.add_(1)
        current.begin(status=handle)
        handoff(current, pool, slots, positions, handle)
        flag = torch.empty(1, dtype=torch.bool, device="cuda")
        current.ring_empty(pool, slots, flag)
        assert flag.item()
        assert (pool.metadata[:, 6] == 7).all()
        assert (pool.metadata[:, 7] == pool.pool_epoch[0]).all()
    for t in range(2 * op.window + 3):
        x = projected(op, batch, seed=1200 + t)
        step(op, state, weights, slots, pos, workspace, x)
        step(fresh_op, fresh, weights, slots, fresh_pos, fresh_workspace, x)
        bitwise(workspace["out"], fresh_workspace["out"])
    op.flush(
        *weights,
        slot_indices=slots,
        state=state,
        status=status,
        workspace=allocate(op.workspace_spec("flush", batch_size=(slots).shape[0])),
    )
    fresh_op.flush(
        *weights,
        slot_indices=slots,
        state=fresh,
        status=fresh_status,
        workspace=allocate(fresh_op.workspace_spec("flush", batch_size=(slots).shape[0])),
    )
    for name in ("checkpoint", "conv_history"):
        bitwise(getattr(state, name)[slots.long()], getattr(fresh, name)[slots.long()])
    bitwise(state.metadata[slots.long(), :5], fresh.metadata[slots.long(), :5])
    assert finish(op, status).ok and finish(fresh_op, fresh_status).ok


@pytest.mark.bitwise
@pytest.mark.parametrize("route", ["product", "immediate", "delayed"])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("vector", [False, True])
def test_delayed_commit_then_step_matches_flushed_reference(route, batch, vector, monkeypatch):
    """R3c and GDNDEC2 share cursor/count, including flush -> lazy append seeds."""
    from ctikernels.gdn.replay.cute import _phase_launch

    monkeypatch.setattr(_phase_launch, "VECTOR_PUBLISH", vector)
    _, state, pending, weights, slots, pos, _ = case(device="cuda", batch=batch, window=8, q=4)
    config = Config(
        immediate=route != "delayed", provider="product" if route == "product" else "cute"
    )
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda",
        provider="cute",
        cute_config=config,
        commit_mode="delayed",
        window=8,
        verify_length=4,
    )
    assert op.decode_route(batch) == ("delayed" if route == "delayed" else "immediate")
    assert op.bytes_per_slot()["ring"] > 0
    step_ws = allocate(op.workspace_spec("step", batch_size=batch))
    phase_ws = allocate(op.workspace_spec("flush", batch_size=batch))
    accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
    verify_out = torch.empty(batch, 4, 512, dtype=torch.bfloat16, device="cuda")
    # Repeated cycles exercise nonzero cursors and new lazy appends after flush.
    for cycle in range(3):
        op.verify(
            projected(op, batch, length=4, seed=800 + cycle),
            *weights,
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(4, device="cuda"),
            state=state,
            pending=pending,
            out=verify_out,
        )
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=accepted,
            state=state,
            pending=pending,
            workspace=phase_ws,
        )
        assert (state.metadata[slots.long(), 3] > 0).all()
        assert (
            state.metadata[slots.long(), 1] - state.metadata[slots.long(), 0]
            == state.metadata[slots.long(), 3]
        ).all()
        pos.add_(1)
        reference = clone(state)
        op.flush(*weights, slot_indices=slots, state=reference, workspace=phase_ws)
        x = projected(op, batch, seed=900 + cycle)
        reference_ws = allocate(op.workspace_spec("step", batch_size=batch))
        for target, workspace in ((state, step_ws), (reference, reference_ws)):
            op.step(
                x,
                *weights,
                slot_indices=slots,
                positions=pos,
                state=target,
                out=workspace["out"],
                workspace=workspace,
            )
        bitwise(step_ws["out"], reference_ws["out"])
        bitwise(state.conv_history, reference.conv_history)
        for target in (state, reference):
            op.flush(*weights, slot_indices=slots, state=target, workspace=phase_ws)
        bitwise(state.checkpoint, reference.checkpoint)
        bitwise(state.metadata[:, :2], reference.metadata[:, :2])
        assert not state.metadata[:, 3].any()
        assert op._product_decode is None
        pos.add_(1)
