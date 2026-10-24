# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""RP1 regressions: empty-ring transitions, aliases and wide pool addresses."""

import importlib.util

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import Config

from ..modules._replay_case import case, clone, projected, same
from ..modules.test_gdn_replay_rp1 import alias_input
from .test_gdn_replay_unified import allocate, bitwise, setup

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and cutlass DSL 4.8.0 required",
)


def step(op, state, weights, slots, pos, status, workspace, x):
    op.step(
        x,
        *weights,
        slot_indices=slots,
        positions=pos,
        state=state,
        out=workspace["out"],
        status=status,
        workspace=workspace,
    )


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("extra_flush", [False, True])
@pytest.mark.parametrize("controls", ["trusted", "strict", "snapshot"])
def test_empty_ring_strict_immediate_round_trip(hk, hv, batch, speculative, extra_flush, controls):
    config = Config(speculative_ring=speculative, snapshot_controls=controls == "snapshot")
    delayed, state, weights, slots, pos, status = setup(
        hk,
        hv,
        batch,
        "delayed",
        checks="trusted" if controls == "trusted" else "strict",
        cute_config=config,
    )
    immediate, reference, _, _, _, _ = setup(
        hk, hv, batch, "immediate", checks="strict", cute_config=Config(immediate=True)
    )
    # A nonzero initial history makes stale seeds observable on the first token.
    state.conv_history.normal_()
    reference.checkpoint.copy_(state.checkpoint)
    reference.conv_history.copy_(state.conv_history)
    ws = allocate(delayed.workspace_spec("step", batch_size=batch))
    iw = allocate(immediate.workspace_spec("step", batch_size=batch))
    fw = allocate(delayed.workspace_spec("flush", batch_size=batch))
    for t in range(9):
        if t == 2:
            delayed.flush(*weights, slot_indices=slots, state=state, status=status, workspace=fw)
        if t == 4 and extra_flush:
            delayed.flush(*weights, slot_indices=slots, state=state, status=status, workspace=fw)
        selected, workspace = (immediate, iw) if 2 <= t < 4 else (delayed, ws)
        x = projected(delayed, batch, seed=700 + t)
        step(selected, state, weights, slots, pos, status, workspace, x)
        actual = workspace["out"].clone()
        step(immediate, reference, weights, slots, pos, status, iw, x)
        bitwise(actual, iw["out"])
        bitwise(state.conv_history, reference.conv_history)
        pos.add_(1)
    delayed.flush(*weights, slot_indices=slots, state=state, status=status, workspace=fw)
    bitwise(state.checkpoint, reference.checkpoint)
    assert not status.poison.any()


@pytest.mark.bitwise
@pytest.mark.parametrize(
    "transition", ["strict_immediate", "product", "eager_commit", "handoff", "import", "reset"]
)
@pytest.mark.parametrize("consumer", ["step", "k2_eager", "k2_delayed"])
@pytest.mark.parametrize("speculative", [False, True])
def test_empty_ring_after_publication(transition, consumer, speculative):
    op, state, weights, slots, pos, status = setup(
        16, 32, 1, "delayed", cute_config=Config(speculative_ring=speculative)
    )
    ws = allocate(op.workspace_spec("step", batch_size=1))
    _, _, pending, _, _, _, _ = case(device="cuda", hk=16, hv=32, batch=1)
    accepted = torch.ones(1, dtype=torch.int32, device="cuda")
    verify_out = torch.empty(1, 8, 4096, dtype=torch.bfloat16, device="cuda")

    def transaction(owner, pool, transaction, x):
        owner.verify(
            x,
            *weights,
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(8, device="cuda"),
            state=pool,
            pending=transaction,
            out=verify_out,
            status=status,
        )
        owner.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=accepted,
            state=pool,
            pending=transaction,
            status=status,
            workspace=allocate(owner.workspace_spec("commit", batch_size=1)),
        )

    for t in range(2):
        step(op, state, weights, slots, pos, status, ws, projected(op, 1, seed=900 + t))
        pos.add_(1)
    op.flush(
        *weights,
        slot_indices=slots,
        state=state,
        status=status,
        workspace=allocate(op.workspace_spec("flush", batch_size=1)),
    )
    if transition in ("strict_immediate", "product"):
        other = GatedDeltaNetReplay(
            num_k_heads=16,
            num_v_heads=32,
            device="cuda",
            provider="cute",
            window=4,
            verify_length=8,
            checks="strict" if transition == "strict_immediate" else "trusted",
            cute_config=Config(
                immediate=True, provider="product" if transition == "product" else "cute"
            ),
        )
        other_ws = allocate(other.workspace_spec("step", batch_size=1))
        other_ws.setdefault("arrivals", torch.zeros(1, dtype=torch.int32, device="cuda"))
        step(other, state, weights, slots, pos, status, other_ws, projected(op, 1, seed=909))
        pos.add_(1)
    elif transition == "eager_commit":
        transaction(op, state, pending, projected(op, 1, length=8, seed=910))
        pos.add_(1)
    elif transition == "handoff":
        op.handoff(
            canonical_state=state.checkpoint[:1].clone(),
            history=torch.randn_like(state.conv_history[:1]),
            next_positions=pos,
            slot_indices=slots,
            state=state,
            status=status,
        )
    elif transition == "import":
        rows = state.metadata.shape[0]
        valid = torch.zeros(1, dtype=torch.bool, device="cuda")
        op.export_snapshot(
            *weights,
            pairs=torch.tensor([[0, rows]], dtype=torch.int32, device="cuda"),
            state=state,
            status=status,
            snapshot_valid=valid,
            workspace=allocate(op.workspace_spec("export_snapshot", batch_size=1)),
        )
        assert valid.item()
        op.import_snapshot(
            pairs=torch.tensor([[rows, 0]], dtype=torch.int32, device="cuda"),
            state=state,
            status=status,
        )
    else:
        op.reset(
            state=state,
            weight_revision=1,
            status=status,
            pending_blocks=(pending,),
            workspaces=(ws,),
        )
        op.initialize(slot_indices=slots, next_positions=pos, state=state, status=status)
    assert state.metadata[0, 3] == 0
    reference = clone(state)
    # Poison every empty-ring physical row in one pool; neither consumer may use it.
    state.ring.fill_(17)
    if consumer == "step":
        x = projected(op, 1, seed=920)
        for _ in range(3):
            step(op, state, weights, slots, pos, status, ws, x)
            actual = ws["out"].clone()
            step(op, reference, weights, slots, pos, status, ws, x)
            bitwise(actual, ws["out"])
            pos.add_(1)
    else:
        if consumer == "k2_delayed":
            op = GatedDeltaNetReplay(
                num_k_heads=16,
                num_v_heads=32,
                device="cuda",
                provider="cute",
                window=4,
                verify_length=8,
                mode="delayed",
                commit_mode="delayed",
                cute_config=Config(),
            )
        other_pending = clone(pending)
        x = projected(op, 1, length=8, seed=921)
        transaction(op, state, pending, x)
        actual = verify_out.clone()
        transaction(op, reference, other_pending, x)
        bitwise(actual, verify_out)
        pos.add_(1)
        # A second consumer must use the newly seeded ring if commit stayed lazy.
        step(op, state, weights, slots, pos, status, ws, projected(op, 1, seed=922))
        actual = ws["out"].clone()
        step(op, reference, weights, slots, pos, status, ws, projected(op, 1, seed=922))
        bitwise(actual, ws["out"])
    for pool in (state, reference):
        op.flush(
            *weights,
            slot_indices=slots,
            state=pool,
            status=status,
            workspace=allocate(op.workspace_spec("flush", batch_size=1)),
        )
    bitwise(state.checkpoint, reference.checkpoint)
    bitwise(state.conv_history, reference.conv_history)
    assert not status.poison.any()


@pytest.mark.parametrize(
    "name,region", [("projected", "metadata"), ("conv_weight", "checkpoint"), ("A_log", "metadata")]
)
def test_product_aliases_rejected_without_mutation(name, region):
    op, state, weights, slots, pos, _ = setup(
        16, 32, 1, cute_config=Config(provider="product", immediate=True)
    )
    inputs = dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True))
    inputs.update(
        projected=projected(op, 1),
        out=torch.zeros(1, 4096, dtype=torch.bfloat16, device="cuda"),
        positions=pos,
        slot_indices=slots,
    )
    state = alias_input(state, inputs, name, region)
    before = clone(state)
    with pytest.raises(ValueError, match="overlapping buffers"):
        op.step(**inputs, state=state)
    same(state, before)


@pytest.mark.bitwise
@pytest.mark.parametrize("mode", ["delayed", "immediate"])
def test_checkpoint_offsets_above_int32_step_phases_snapshots(mode):
    op, _, weights, _, _, status = setup(16, 32, 1, mode, checks="strict")
    stride = 32 * 16384
    high = (2**31 + stride - 1) // stride
    rows = high + 1
    specs = op.state_spec(num_slots=rows, num_snapshot_rows=2)
    required = sum(s.payload_bytes for s in specs.values())
    assert high * stride >= 2**31 and required < 12_000_000_000
    free, _ = torch.cuda.mem_get_info()
    if free < required + 1024**3:
        pytest.skip(f"requires {required + 1024**3} free bytes for sparse pool and scratch")
    state = op.create_state(
        buffers={
            name: torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            for name, spec in specs.items()
        }
    )
    # Never zero the full allocation: only two live rows and two snapshots are touched.
    state.pool_epoch.zero_()
    state.snapshot_metadata.zero_()
    for slot in (0, high):
        state.metadata[slot].zero_()
        state.checkpoint[slot].zero_()
        state.conv_history[slot].zero_()
        state.ring[slot].zero_()
    slots = [torch.tensor([i], dtype=torch.int32, device="cuda") for i in (0, high)]
    pos = torch.zeros(1, dtype=torch.int64, device="cuda")
    for selected in slots:
        op.initialize(slot_indices=selected, next_positions=pos, state=state, status=status)
    ws = allocate(op.workspace_spec("step", batch_size=1))

    def compare_live():
        for name in ("checkpoint", "conv_history", "ring", "metadata"):
            bitwise(getattr(state, name)[0], getattr(state, name)[high])

    def pair_steps(seed):
        outputs = []
        for selected in slots:
            step(op, state, weights, selected, pos, status, ws, projected(op, 1, seed=seed))
            outputs.append(ws["out"].clone())
        bitwise(*outputs)
        compare_live()
        pos.add_(1)

    pair_steps(950)
    # Verify reads the high checkpoint/ring; eager commit writes the high checkpoint.
    x = projected(op, 1, length=8, seed=951)
    outputs = []
    for selected in slots:
        pending = op.create_pending(
            raw=torch.empty(1, 8, op.geometry.conv_dim + 64, dtype=torch.bfloat16, device="cuda"),
            metadata=torch.zeros(1, 8, dtype=torch.int64, device="cuda"),
        )
        out = torch.empty(1, 8, 4096, dtype=torch.bfloat16, device="cuda")
        op.verify(
            x,
            *weights,
            slot_indices=selected,
            positions=pos[:, None] + torch.arange(8, device="cuda"),
            state=state,
            pending=pending,
            out=out,
            status=status,
        )
        outputs.append(out)
        op.commit(
            *weights,
            slot_indices=selected,
            accepted_lengths=torch.full((1,), 3, dtype=torch.int32, device="cuda"),
            state=state,
            pending=pending,
            status=status,
        )
    bitwise(*outputs)
    compare_live()
    pos.add_(3)
    pair_steps(952)
    # Both snapshot rows themselves also lie beyond the signed-int32 boundary.
    for index, slot in enumerate((0, high)):
        valid = torch.zeros(1, dtype=torch.bool, device="cuda")
        op.export_snapshot(
            *weights,
            pairs=torch.tensor([[slot, rows + index]], dtype=torch.int32, device="cuda"),
            state=state,
            status=status,
            snapshot_valid=valid,
        )
        assert valid.item()
    bitwise(state.checkpoint[rows], state.checkpoint[rows + 1])
    bitwise(state.conv_history[rows], state.conv_history[rows + 1])
    for index, slot in enumerate((0, high)):
        state.checkpoint[slot].fill_(19)
        state.conv_history[slot].fill_(19)
        op.import_snapshot(
            pairs=torch.tensor([[rows + index, slot]], dtype=torch.int32, device="cuda"),
            state=state,
            status=status,
        )
    compare_live()
    pair_steps(953)
    assert not status.poison.any()
