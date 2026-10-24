# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contract tests for the replay companion; no CUDA provider is loaded."""

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay, replay_sizes
from ctikernels.gdn.replay.torch_reference import compose, raw_row, token

from ._replay_case import (
    case,
    clone,
    decode_steps,
    finish,
    graph_buffers,
    projected,
    same,
    verify,
    zero_rows,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize(
    "hv,ring_kib,pending_kib", [(32, 177.4375, 258.0625), (48, 222.125, 323.0625)]
)
def test_contract_sizing(hv, ring_kib, pending_kib):
    spec = replay_sizes(num_k_heads=16, num_v_heads=hv)
    assert (spec["ring"].payload_bytes + spec["metadata"].payload_bytes) / 1024 == ring_kib
    assert (
        spec["pending.raw"].payload_bytes + spec["pending.metadata"].payload_bytes
    ) / 1024 == pending_kib
    assert spec["checkpoint"].payload_bytes == 2 * hv * 128 * 128 * 4
    assert spec["conv_history"].payload_bytes == 2 * 6 * (32 * 128 + hv * 128)
    for region in spec.values():
        assert region.offset % 256 == 0
        assert region.aligned_bytes == (region.payload_bytes + 255) // 256 * 256
    many = replay_sizes(
        num_k_heads=16, num_v_heads=hv, num_slots=3, num_snapshot_rows=2, batch_size=3
    )
    assert many["metadata"].aligned_bytes == 256  # NOT 3*256
    assert many["snapshot_metadata"].payload_bytes == 64
    tp = replay_sizes(num_k_heads=8, num_v_heads=hv // 2)
    assert tp["ring"].payload_bytes * 2 == spec["ring"].payload_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key_dim": 64},
        {"conv_width": 3},
        {"dtype": torch.float32},
        {"window": 3},
        {"verify_length": 0},
        {"verify_length": 8.0},
        {"num_slots": -1},
    ],
)
def test_sizing_refusals(kwargs):
    with pytest.raises(ValueError):
        replay_sizes(num_k_heads=16, num_v_heads=32, **kwargs)


@pytest.mark.parametrize("q", [2, 4, 5, 8, 9, 12, 13, 16, 17, 32])
def test_any_verify_length_sizes_pending_by_q(q):
    op = GatedDeltaNetReplay(num_k_heads=8, num_v_heads=16, device="cpu", verify_length=q)
    assert op.capabilities["verify_lengths"] == (1, None)
    raw = op.workspace_spec("verify", batch_size=3)["pending.raw"]
    assert raw.shape == (3, q, op.geometry.conv_dim + 2 * op.geometry.v_heads)


@pytest.mark.parametrize("window", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_delayed_equals_immediate_at_boundaries_and_wraparound(window, dtype):
    op, state, _, weights, slots, pos, status = case(window=window, batch=1, oracle=True)
    op.reference_dtype = dtype
    matrix = state.checkpoint[0].to(dtype).clone()
    hist = state.conv_history[0].clone()
    out = torch.empty(1, op.geometry.value_width, dtype=torch.bfloat16)
    for i in range(3 * window + 3):
        x = projected(op, 1, seed=100 + i)
        matrix, hist, y = token(
            op.geometry,
            weights,
            matrix,
            hist,
            raw_row(op.geometry, x[0]),
            x[0, op.geometry.conv_dim : op.geometry.conv_dim + op.geometry.value_width],
        )
        op.step(x, *weights, slot_indices=slots, positions=pos, state=state, out=out, status=status)
        pos.add_(1)
        logical, history = compose(op, state, 0, weights)
        assert torch.equal(logical, matrix)
        assert torch.equal(history, hist)
        assert torch.equal(out[0], y.to(torch.bfloat16))
        assert int(state.metadata[0, 3]) == (i + 1) % window
    assert status.code.item() == 0


@pytest.mark.parametrize("window,q", [(4, 2), (4, 8), (8, 16), (16, 32)])
def test_every_acceptance_and_offset_bitwise(window, q):
    op, base, pending, weights, slots, pos, status = case(window=window, q=q, batch=1)
    decode_steps(op, base, weights, slots, pos, status, 2 * window - 1)
    before = clone(base)
    x, output = verify(op, base, pending, weights, slots, pos, status)
    same(before, base)
    logical, hist = compose(op, base, 0, weights)
    for j in range(q):
        logical, hist, y = token(
            op.geometry,
            weights,
            logical,
            hist,
            raw_row(op.geometry, x[0, j]),
            x[0, j, op.geometry.conv_dim : op.geometry.conv_dim + op.geometry.value_width],
        )
        assert torch.equal(output[0, j], y.to(torch.bfloat16))
    r = base.metadata.shape[0]
    # Every m and every k, including k>m; snapshot bits must equal independent commit(k).
    oracles = []
    for m in range(q + 1):
        dest, p = clone(base), clone(pending)
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=torch.tensor([m], dtype=torch.int32),
            state=dest,
            pending=p,
            status=status,
        )
        oracles.append(dest)
        assert dest.metadata[0, :4].tolist() == [int(pos[0]) + m, int(pos[0]) + m, 0, 0]
    for m in range(q + 1):
        for k in range(m + 2):
            dest, p = clone(base), clone(pending)
            valid = torch.ones(1, dtype=torch.bool)
            op.commit(
                *weights,
                slot_indices=slots,
                accepted_lengths=torch.tensor([m], dtype=torch.int32),
                state=dest,
                pending=p,
                status=status,
                snapshot_rows=torch.tensor([r], dtype=torch.int32),
                snapshot_offsets=torch.tensor([k]),
                snapshot_valid=valid,
            )
            assert bool(valid[0]) == (k <= m)
            if k <= m:
                assert torch.equal(dest.checkpoint[r], oracles[k].checkpoint[0])
                assert torch.equal(dest.conv_history[r], oracles[k].conv_history[0])
            else:
                assert torch.equal(dest.checkpoint[r], base.checkpoint[r])
            assert torch.equal(dest.checkpoint[0], oracles[m].checkpoint[0])
    assert status.code.item() == 0


def test_export_import_reject_reuse_reset_and_handoff():
    op, state, pending, weights, slots, pos, status = case(batch=1)
    decode_steps(op, state, weights, slots, pos, status, 7)
    before = clone(state)
    _, _ = verify(op, state, pending, weights, slots, pos, status)
    op.reject(slot_indices=slots, state=state, pending=pending, status=status)
    same(before, state)
    r = state.metadata.shape[0]
    valid = torch.empty(2, dtype=torch.bool)
    op.export_snapshot(
        *weights,
        pairs=torch.tensor([[0, r], [0, r + 1]], dtype=torch.int32),
        state=state,
        status=status,
        snapshot_valid=valid,
    )
    for f in ("checkpoint", "conv_history", "ring", "metadata"):
        assert torch.equal(getattr(before, f)[:r], getattr(state, f)[:r])
    assert valid.all()
    op.import_snapshot(pairs=torch.tensor([[r, 1]], dtype=torch.int32), state=state, status=status)
    out = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    x = projected(op, 1).expand(2, -1).clone()
    op.step(
        x,
        *weights,
        slot_indices=torch.tensor([0, 1], dtype=torch.int32),
        positions=pos.repeat(2),
        state=state,
        out=out,
        status=status,
    )
    assert torch.equal(out[0], out[1])
    # Handoff accepts external FP32 canonical prefill results, including large positions.
    op.handoff(
        canonical_state=state.checkpoint[r : r + 1].clone(),
        history=state.conv_history[r : r + 1].clone(),
        next_positions=pos,
        slot_indices=slots,
        state=state,
        status=status,
    )
    assert state.metadata[0, :4].tolist() == [pos.item(), pos.item(), 0, 0]
    gen = state.metadata[0, 5].item()
    op.release(slot_indices=slots, state=state, status=status)
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    assert state.metadata[0, 5] == gen + 2
    assert state.checkpoint[0].count_nonzero() == 0
    payload = [state.checkpoint.clone(), state.conv_history.clone(), state.ring.clone()]
    op.reset(state=state, weight_revision=3, pending_blocks=(pending,), status=status)
    for old, new in zip(payload, (state.checkpoint, state.conv_history, state.ring), strict=True):
        assert torch.equal(old, new)
    assert state.pool_epoch.item() == 1
    assert state.metadata[:, 2:5].count_nonzero() == 0
    assert (state.metadata[:, 6] == 3).all()
    assert pending.metadata.count_nonzero() == 0
    assert state.snapshot_metadata.count_nonzero() == 0
    assert status.code.item() == 0


@pytest.mark.parametrize("m,k", [(0, 0), (5, 3), (8, 8), (2, 3)])
def test_resolved_export_and_changed_acceptance(m, k):
    op, state, pending, weights, slots, pos, status = case(batch=1)
    decode_steps(op, state, weights, slots, pos, status, 7)
    verify(op, state, pending, weights, slots, pos, status)
    original, original_pending = clone(state), clone(pending)
    r = state.metadata.shape[0]
    valid = torch.empty(2, dtype=torch.bool)
    op.export_snapshot(
        *weights,
        pairs=torch.tensor([[0, r], [0, r + 1]], dtype=torch.int32),
        state=state,
        status=status,
        snapshot_valid=valid,
        pending=pending,
        accepted_lengths=torch.tensor([m], dtype=torch.int32),
        offsets=torch.tensor([k, k]),
    )
    assert valid.tolist() == [k <= m] * 2
    assert pending.metadata[0, 5:7].tolist() == [2, m]
    if k <= m:
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=torch.tensor([k], dtype=torch.int32),
            state=original,
            pending=original_pending,
            status=status,
        )
        assert torch.equal(state.checkpoint[r], original.checkpoint[0])
        assert torch.equal(state.conv_history[r], original.conv_history[0])
    saved = clone(state)
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([(m + 1) % 9], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    assert status.code.item() == 8
    same(state, saved)


def test_padding_and_sticky_failure_suppress_all_rows():
    op, state, pending, weights, slots, pos, status = case()
    slots[1] = -1
    saved = clone(state)
    p_saved = clone(pending)
    _, out = verify(op, state, pending, weights, slots, pos, status)
    same(saved, state)
    assert torch.equal(pending.raw[1], p_saved.raw[1])
    assert torch.equal(pending.metadata[1], p_saved.metadata[1])
    assert not out[1].count_nonzero()
    saved_p = clone(pending)
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([9, -999], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    assert status.code.item() == 4
    same(saved, state)
    same(saved_p, pending)
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    same(saved, state)
    assert not finish(op, status).ok
    op.recover_reset(states=(state,), pending_blocks=(pending,), status=status, drained=True)
    status.sequence.add_(1)
    op.begin(status=status)
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    verify(op, state, pending, weights, slots, pos, status)
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([0, -999], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    assert status.code.item() == 0
    assert torch.equal(state.checkpoint[1], saved.checkpoint[1])
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([0, -999], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    assert status.code.item() == 8


@pytest.mark.parametrize("weight_index", range(4))
def test_frozen_weights_are_first_prelaunch_check(weight_index):
    op, state, pending, weights, slots, pos, status = case()
    weights[weight_index].requires_grad_(True)
    op._backends = {}
    with pytest.raises(ValueError, match="frozen"):
        op.step(
            None, *weights, slot_indices=None, positions=None, state=None, out=None, status=None
        )
    with pytest.raises(ValueError, match="frozen"):
        op.commit(
            *weights,
            slot_indices=None,
            accepted_lengths=None,
            state=None,
            pending=None,
            status=None,
        )
    with pytest.raises(ValueError, match="frozen"):
        op.export_snapshot(*weights, pairs=None, state=None, status=None, snapshot_valid=None)


def test_alias_grad_and_capability_checks():
    op, state, pending, weights, slots, pos, status = case(batch=1)
    with pytest.raises(ValueError, match="strict"):
        GatedDeltaNetReplay(num_k_heads=1, num_v_heads=4, device="cpu", precision="strict")
    assert set(op.provider.values()) == {"torch_reference"}
    assert op.capabilities["scratch_bytes"] == 0
    x = projected(op, 1)
    with pytest.raises(ValueError, match="overlapping"):
        op.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=x[:, : op.geometry.value_width],
            status=status,
        )
    state.checkpoint.requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        op.release(slot_indices=slots, state=state, status=status)
    # No tensor allocations in host sizing.
    assert all(s.offset % 256 == 0 for s in op.workspace_spec("verify", batch_size=2).values())


@pytest.mark.parametrize(
    "fault,code", [("position", 2), ("epoch", 1), ("generation", 8), ("offset", 4)]
)
def test_invalid_metadata_is_transactional(fault, code):
    op, state, pending, weights, slots, pos, status = case()
    verify(op, state, pending, weights, slots, pos, status)
    if fault == "epoch":
        state.pool_epoch.add_(1)
    elif fault == "generation":
        state.metadata[1, 5] += 1
    saved, ps = clone(state), clone(pending)
    if fault == "position":
        bad = pos.clone()
        bad[1] += 1
        out = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
        op.step(
            projected(op, 2),
            *weights,
            slot_indices=slots,
            positions=bad,
            state=state,
            out=out,
            status=status,
        )
        assert out.count_nonzero() == 0
    else:
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=torch.zeros(2, dtype=torch.int32),
            state=state,
            pending=pending,
            status=status,
            snapshot_rows=torch.tensor([state.metadata.shape[0], -1], dtype=torch.int32),
            snapshot_offsets=torch.tensor([-1 if fault == "offset" else 0, 0]),
            snapshot_valid=torch.empty(2, dtype=torch.bool),
        )
    assert status.code.item() & code
    same(saved, state)
    same(ps, pending)


def test_all_padding_never_reads_poisoned_payload_or_stale_pending():
    op, state, pending, weights, slots, pos, status = case()
    slots.fill_(-1)
    state.checkpoint.fill_(float("nan"))
    state.conv_history.fill_(float("nan"))
    state.ring.fill_(float("nan"))
    pending.raw.fill_(float("nan"))
    pending.metadata.fill_(-700)
    saved, ps = clone(state), clone(pending)
    out = decode_steps(op, state, weights, slots, pos, status, 1)
    assert not out.count_nonzero()
    _, out = verify(op, state, pending, weights, slots, pos, status)
    assert not out.count_nonzero()
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.full((2,), -999, dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    op.reject(slot_indices=slots, state=state, pending=pending, status=status)
    op.flush(*weights, slot_indices=slots, state=state, status=status)
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    op.release(slot_indices=slots, state=state, status=status)
    op.handoff(
        canonical_state=state.checkpoint[:2].clone(),
        history=state.conv_history[:2].clone(),
        next_positions=pos,
        slot_indices=slots,
        state=state,
        status=status,
    )
    pairs = torch.tensor([[-1, 4], [1, -1]], dtype=torch.int32)
    valid = torch.ones(2, dtype=torch.bool)
    op.export_snapshot(*weights, pairs=pairs, state=state, status=status, snapshot_valid=valid)
    op.import_snapshot(pairs=pairs, state=state, status=status)
    same(state, saved)
    same(pending, ps)
    assert not valid.any()
    assert not status.code.item()


def test_snapshot_revision_and_epoch_validation_and_inplace_handoff():
    op, state, pending, weights, slots, pos, status = case(batch=1)
    # Exact prefix views are legal for an identity handoff; no allocation/copy required.
    decode_steps(op, state, weights, slots, pos, status, 4)
    before = state.checkpoint[0].clone()
    op.handoff(
        canonical_state=state.checkpoint[:1],
        history=state.conv_history[:1],
        next_positions=pos,
        slot_indices=slots,
        state=state,
        status=status,
    )
    assert torch.equal(before, state.checkpoint[0])
    r = state.metadata.shape[0]
    op.export_snapshot(
        *weights,
        pairs=torch.tensor([[0, r]], dtype=torch.int32),
        state=state,
        status=status,
        snapshot_valid=torch.empty(1, dtype=torch.bool),
    )
    for field in (1, 2):
        state.snapshot_metadata[0, field] += 1
        saved = clone(state)
        op.import_snapshot(
            pairs=torch.tensor([[r, 1]], dtype=torch.int32), state=state, status=status
        )
        assert status.code.item() == 16
        same(saved, state)
        state.snapshot_metadata[0, field] -= 1
        finish(op, status)
        op.recover_reset(states=(state,), status=status, pending_blocks=(pending,), drained=True)
        status.sequence.add_(1)
        op.begin(status=status)
        op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
        op.export_snapshot(
            *weights,
            pairs=torch.tensor([[0, r]], dtype=torch.int32),
            state=state,
            status=status,
            snapshot_valid=torch.empty(1, dtype=torch.bool),
        )
    # Invalid fresh payload is never read when a slot is reused.
    state.checkpoint[0].fill_(float("nan"))
    state.conv_history[0].fill_(float("nan"))
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    assert state.checkpoint[0].count_nonzero() == 0
    assert state.conv_history[0].count_nonzero() == 0


def test_reference_matches_independent_pinned_token_recurrence():
    from ..gpu._gdn_reference import Weights, _Sequence

    op, state, _, weights, slots, pos, status = case(batch=1, oracle=True)
    seq = _Sequence(
        Weights(*weights), op.geometry.k_heads, state.checkpoint[0], state.conv_history[0]
    )
    out = torch.empty(1, op.geometry.value_width, dtype=torch.bfloat16)
    for j in range(13):
        x = projected(op, 1, seed=800 + j)
        expected = seq.step(x[0])
        op.step(x, *weights, slot_indices=slots, positions=pos, state=state, out=out, status=status)
        pos.add_(1)
        matrix, hist = compose(op, state, 0, weights)
        torch.testing.assert_close(matrix, seq.pool_state(), rtol=1e-12, atol=1e-14)
        assert torch.equal(hist, seq.history)
        assert torch.equal(out[0], expected.to(torch.bfloat16))


@pytest.mark.parametrize("q,totals", [(16, [494, 936, 1378, 1820]), (8, [366, 680, 994, 1308])])
def test_static_control_payload_and_binding(q, totals):
    for b, total in enumerate(totals, 1):
        op = GatedDeltaNetReplay(num_k_heads=8, num_v_heads=16, device="cpu", verify_length=q)
        spec = op.graph_buffer_spec(batch_size=b)
        assert sum(v.payload_bytes for v in spec.values()) == total
        control = graph_buffers(op, batch=b)
        h0, h1 = control.status(0), control.status(1)
        assert h0.record.data_ptr() + 16 == h1.record.data_ptr()
        assert h0.poison.data_ptr() == h1.poison.data_ptr()
        assert control.buffers["positions"].dtype == torch.int64
        assert control.buffers["verify_positions"].dtype == torch.int64
        assert (
            control.buffers["positions"].data_ptr()
            != control.buffers["verify_positions"].data_ptr()
        )
        multi = op.graph_buffer_spec(batch_size=b, depth=3, num_layers=4)
        assert (
            sum(v.payload_bytes for v in multi.values())
            == 3 * b * (8 * q + 28 + 65 * 4) + 3 * (8 + 16 * 4) + 4
        )


def test_submit_before_settle_record_survival_and_modulo_collision():
    op, state, pending, weights, slots, pos, _ = case(batch=1)
    controls = graph_buffers(op)
    first, second = controls.status(0), controls.status(1)
    # Free zero records are never completed successes.
    assert not op.read_status_copy(*first.copy_to_host(), expected_sequence=1).ok
    first.sequence.fill_(1)
    op.begin(status=first, layer_weights=(weights,))
    decode_steps(op, state, weights, slots, pos, first, 1)
    op.end(status=first)
    original = first.records.clone()
    second.sequence.fill_(2)
    op.begin(status=second, layer_weights=(weights,))
    decode_steps(op, state, weights, slots, pos, second, 1)
    op.end(status=second)
    assert torch.equal(first.records, original)
    result1 = op.read_status_copy(*first.copy_to_host(), expected_sequence=1)
    result2 = op.read_status_copy(*second.copy_to_host(), expected_sequence=2)
    assert result1.ok and result2.ok
    op.settle(status=first, result=result1)
    assert not first.records.count_nonzero()
    assert second.record[:8].view(torch.int64)[0] == 2
    # Slot zero may be reused while slot one remains unsettled.
    first.sequence.fill_(3)
    op.begin(status=first)
    op.end(status=first)
    saved = first.records.clone()
    first.valid.fill_(True)
    first.sequence.fill_(5)  # illegal modulo reuse before settle(3)
    op.begin(status=first)
    op.end(status=first)
    assert torch.equal(first.records, saved)
    assert first.valid.all()  # collision must not clear an older candidate
    assert first.poison.item() & 64
    assert not op.read_status_copy(*first.copy_to_host(), expected_sequence=5).ok
    before = clone(state)
    decode_steps(op, state, weights, slots, pos, first, 1)
    same(state, before)
    with pytest.raises(ValueError, match="stale"):
        op.settle(
            status=first, result=op.read_status_copy(*first.copy_to_host(), expected_sequence=5)
        )
    # Drain original records using their original sequence inputs, then recover.
    first.sequence.fill_(3)
    op.settle(status=first, result=op.read_status_copy(*first.copy_to_host(), expected_sequence=3))
    op.settle(status=second, result=result2)
    assert first.poison.item() & 64
    with pytest.raises(ValueError, match="drain"):
        op.recover_reset(states=(state,), status=first)
    op.recover_reset(states=(state,), status=first, pending_blocks=(pending,), drained=True)
    assert not first.poison.item()
    assert not state.metadata[:, 4].any()
    first.sequence.fill_(6)
    op.begin(status=first)
    op.initialize(slot_indices=slots, next_positions=pos, state=state, status=first)
    decode_steps(op, state, weights, slots, pos, first, 1)
    assert finish(op, first).ok


@pytest.mark.parametrize("depth", [2, 3])
def test_partial_layer_failure_poison_and_later_invocations(depth):
    op, state, pending, weights, slots, pos, _ = case(batch=1)
    layers = [clone(state) for _ in range(3)]
    controls = graph_buffers(op, depth=depth, layers=3)
    h = controls.status(0)
    h.sequence.fill_(1)
    op.begin(status=h, layer_weights=(weights, weights, weights))
    out = torch.empty(1, op.geometry.value_width, dtype=torch.bfloat16)
    x = projected(op, 1)
    saved = [clone(s) for s in layers]
    for j in range(3):
        op.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos + (j == 1),
            state=layers[j],
            out=out,
            status=h.for_layer(j),
        )
    assert layers[0].metadata[0, 1] == pos[0] + 1  # already written: no rollback
    same(layers[1], saved[1])
    same(layers[2], saved[2])
    op.end(status=h)
    result = op.read_status_copy(*h.copy_to_host(), expected_sequence=1)
    assert not result.ok and result.codes[1] & 2 and result.codes[2] & 32
    later = controls.status(1)
    later.sequence.fill_(2)
    op.begin(status=later)
    after = clone(layers[0])
    op.initialize(slot_indices=slots, next_positions=pos, state=layers[0], status=later)
    same(after, layers[0])
    op.end(status=later)
    op.settle(status=h, result=result)
    assert not finish(op, later).ok
    assert h.poison.item()
    op.recover_reset(states=layers, status=h, pending_blocks=(pending,), drained=True)
    for s in layers:
        assert not s.metadata[:, 4].any()
    # Multiple complete rotations prove that only settled slots are reusable.
    for n in range(3, 3 + depth * 3):
        handle = controls.status((n - 1) % depth)
        handle.sequence.fill_(n)
        op.begin(status=handle)
        assert finish(op, handle).ok


def test_zero_sequence_and_all_layer_grad_preflight():
    op, *_ = case(batch=1)
    controls = graph_buffers(op)
    h = controls.status(0)
    op.begin(status=h)
    assert h.poison.item() == 64
    assert not h.records.count_nonzero()
    _, _, _, weights, *_ = case(batch=1)
    weights[-1].requires_grad_(True)
    saved = h.records.clone()
    with pytest.raises(ValueError, match="frozen"):
        op.begin(status=h, layer_weights=(weights,))
    assert torch.equal(saved, h.records)
    with pytest.raises(ValueError, match="bound"):
        op.begin(status=h, sequence=torch.tensor([1]))


@pytest.mark.parametrize("q", [8, 16])
@pytest.mark.parametrize("tail", [3, 4, 5])
@pytest.mark.parametrize("fraction", [0, 0.5, 1])
def test_chain_acceptance_flush_boundaries(q, tail, fraction):
    op, state, pending, weights, slots, pos, status = case(q=q, batch=2)
    decode_steps(op, state, weights, slots, pos, status, 8 + tail)
    base = clone(state)
    x, _ = verify(op, state, pending, weights, slots, pos, status)
    m = int(q * fraction)
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([m, q], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    for row, count in enumerate((m, q)):
        matrix, hist = compose(op, base, row, weights)
        for j in range(count):
            matrix, hist, _ = token(
                op.geometry, weights, matrix, hist, raw_row(op.geometry, x[row, j])
            )
        assert torch.equal(state.checkpoint[row], matrix.float())
        assert torch.equal(state.conv_history[row], hist)
    assert finish(op, status).ok


@pytest.mark.parametrize("q", [8, 16])
def test_bridge_order_and_frozen_weight_preflight(q):
    op, state, _, weights, slots, pos, status = case(q=q, batch=2)
    decode_steps(op, state, weights, slots, pos, status, 7)
    before = clone(state)
    x = projected(op, 2)
    out = torch.empty(2, op.geometry.value_width, dtype=torch.bfloat16)
    op.bridge_step(
        x, *weights, slot_indices=slots, positions=pos, state=state, out=out, status=status
    )
    for row in range(2):
        matrix, hist = compose(op, before, row, weights)
        matrix, hist, y = token(
            op.geometry,
            weights,
            matrix.float().double(),
            hist,
            raw_row(op.geometry, x[row]),
            x[row, op.geometry.conv_dim : op.geometry.conv_dim + op.geometry.value_width],
        )
        assert torch.equal(state.checkpoint[row], matrix.float())
        assert torch.equal(state.conv_history[row], hist)
        assert torch.equal(out[row], y.bfloat16())
    assert not state.metadata[:2, 2:4].any()
    assert torch.equal(state.metadata[:2, 1], pos + 1)
    # A validation failure must precede materialization of even a healthy row.
    before = clone(state)
    op.bridge_step(
        x, *weights, slot_indices=slots, positions=pos, state=state, out=out, status=status
    )
    same(before, state)
    weights[0].requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        op.bridge_step(
            None, *weights, slot_indices=None, positions=None, state=None, out=None, status=None
        )


def test_tp_local_heads_and_mixed_q_blocks_share_pool():
    op, state, _, weights, slots, pos, status = case(hk=8, hv=16, batch=2, q=8)
    other = GatedDeltaNetReplay(
        num_k_heads=8, num_v_heads=16, window=4, verify_length=16, device="cpu"
    )
    blocks = [
        provider.create_pending(
            raw=torch.zeros(
                1, provider.verify_length, op.geometry.conv_dim + 32, dtype=torch.bfloat16
            ),
            metadata=torch.zeros(1, 8, dtype=torch.int64),
        )
        for provider in (op, other)
    ]
    # Both Q-specialized transactions coexist on different rows of one pool.
    before = clone(state)
    verify(op, state, blocks[0], weights, slots[:1], pos[:1], status)
    verify(other, state, blocks[1], weights, slots[1:], pos[1:], status)
    same(before, state)
    for row, (provider, count) in enumerate(((op, 4), (other, 16))):
        provider.commit(
            *weights,
            slot_indices=slots[row : row + 1],
            accepted_lengths=torch.tensor([count], dtype=torch.int32),
            state=state,
            pending=blocks[row],
            status=status,
        )
    assert torch.equal(state.metadata[:2, 1], pos + torch.tensor([4, 16]))
    assert finish(op, status).ok


def test_smaller_graph_variant_clears_full_validity_capacity_and_stale_settle():
    op, *_ = case(batch=1)
    controls = graph_buffers(op, batch=4)
    h = controls.status(0, batch_size=1)
    h.sequence.fill_(1)
    h.validity.fill_(True)
    op.begin(status=h)
    assert h.valid.shape == (1,)
    assert h.validity.shape == (1, 4)
    assert not h.validity.any()
    old = finish(op, h)
    h.sequence.fill_(3)
    op.begin(status=h)
    saved = h.records.clone()
    op.settle(status=h, result=old)  # stale host result cannot clear a new invocation
    assert torch.equal(saved, h.records)
    assert h.poison.item() & 64


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_dense_pending_slot_views_have_natural_alignment(batch):
    op, *_ = case(batch=batch, q=8)
    control = graph_buffers(op, batch=batch)
    raw = torch.empty(
        2, batch, 8, op.geometry.conv_dim + 2 * op.geometry.v_heads, dtype=torch.bfloat16
    )
    for slot in range(2):
        p = control.pending(raw[slot], slot=slot)
        op._check_pending(p)
        assert (
            p.metadata.data_ptr() - control.buffers["pending_metadata"].data_ptr()
            == slot * batch * 64
        )
        assert (
            p.raw.data_ptr() - raw.data_ptr()
            == slot * batch * 8 * (op.geometry.conv_dim + 2 * op.geometry.v_heads) * 2
        )


def test_zero_rows_are_a_no_op():
    op, state, pending, weights, slots, pos, status = case(batch=2)
    before, pending_before = clone(state), clone(pending)
    zero_rows(op, state, pending, weights, slots, pos, status)
    same(state, before)
    same(pending, pending_before)
    assert finish(op, status).codes == (0,)
