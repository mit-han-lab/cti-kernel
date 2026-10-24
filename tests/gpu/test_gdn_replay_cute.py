# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe arithmetic with the portable S1/G1 graph and lifecycle acceptance suite."""

import importlib.util

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import Config

from ..modules._replay_case import clone, live_ring, same
from . import test_gdn_replay as portable
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and cutlass DSL 4.8.0 required",
)


@pytest.fixture(autouse=True)
def cute_cases(monkeypatch):
    original = portable.case

    def case(**kw):
        values = list(original(**kw))
        old = values[0]
        values[0] = GatedDeltaNetReplay(
            num_k_heads=old.geometry.k_heads,
            num_v_heads=old.geometry.v_heads,
            window=old.window,
            verify_length=old.verify_length,
            device=old.device,
            provider="cute",
            mode="delayed",
            checks="strict",
        )
        return tuple(values)

    monkeypatch.setattr(portable, "case", case)


# Boundary CI keeps 8/16 heads with an 8-token window; other rows are exhaustive.
@pytest.mark.parametrize(
    "hk,hv,window",
    ci_cases(
        [(16, 32), (16, 48), (8, 16), (8, 24)],
        [4, 8, 16],
        core=[(8, 16, 8)],
    ),
)
def test_multiphase_boundaries_and_slot_reuse(hk, hv, window):
    p = portable.Pair(hk=hk, hv=hv, window=window, q=8, batch=2)
    assert p.op.provider["step"] == "cute"
    assert p.op.provider["commit"] == "cute"
    for cycle, m in enumerate((0, 3, 8)):
        for t in range(2 * window + 1):
            p.step(1000 + cycle * 100 + t)
        p.verify(2000 + cycle)
        p.commit([m, 8 - m], [0, 8 - m])
        p.verify(3000 + cycle)
        saved = clone(p.state)
        p.call("reject", slot_indices=p.slots, pending=p.pending)
        same(p.state, saved)
        p.call("release", slot_indices=p.slots)
        p.call("initialize", slot_indices=p.slots, next_positions=p.pos)
        p.step(4000 + cycle)


@pytest.mark.parametrize(
    "batch,q",
    ci_cases([1, 2, 3, 4], [5, 8, 16, 17], core=[(1, 5)]),
)
def test_graph_changing_controls(batch, q):
    portable.test_graph_ring_async_completion_device_controls_and_mixed_phases(batch, q)


def test_poison_and_recovery():
    portable.test_multiphase_flushes_rollbacks_reset_and_invalid_calls()


@pytest.mark.parametrize("fault,code", [("position", 2), ("uninitialized", 1), ("snapshot", 16)])
def test_whole_batch_validation(fault, code):
    portable.test_validation_suppresses_other_rows_and_later_phases(fault, code)


@pytest.mark.parametrize(
    "warps,lanes,prefetch",
    [
        pytest.param(4, 4, False, marks=pytest.mark.exhaustive),
        pytest.param(4, 8, False, marks=pytest.mark.exhaustive),
        (8, 4, False),
        (8, 8, False),
        pytest.param(4, 4, True, marks=pytest.mark.exhaustive),
        pytest.param(8, 8, True, marks=pytest.mark.exhaustive),
    ],
)
def test_head_workspace(warps, lanes, prefetch):
    from ..modules._replay_case import case, projected

    op, state, _, weights, slots, pos, status = case(device="cuda", batch=2)
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        window=4,
        verify_length=8,
        device="cuda",
        provider="cute",
        checks="strict",
        cute_config=Config(warps=warps, lanes=lanes, prefetch=prefetch),
    )
    ref = clone(state)
    full = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        window=4,
        verify_length=8,
        device="cuda",
        provider="cute",
        checks="strict",
        cute_config=Config(lanes=lanes),
    )
    workspace = {
        n: torch.full(s.shape, s.initial_value or 0, dtype=s.dtype, device="cuda")
        for n, s in op.workspace_spec("step", batch_size=2).items()
        if n != "out"
    }
    full_workspace = {
        n: torch.full(s.shape, s.initial_value or 0, dtype=s.dtype, device="cuda")
        for n, s in full.workspace_spec("step", batch_size=2).items()
        if n != "out"
    }
    out = torch.empty(2, 512, dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(out)
    for t in range(13):
        x = projected(op, 2, seed=50 + t)
        op.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            status=status,
            workspace=workspace,
        )
        full.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=ref,
            out=expected,
            status=status,
            workspace=full_workspace,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        same(state, ref)
        pos.add_(1)


@pytest.mark.parametrize("weight", range(4))
def test_frozen_weights_before_any_launch(weight, monkeypatch):
    p = portable.Pair(hk=1, hv=4, batch=1)

    def launch(*args, **kwargs):
        pytest.fail("provider launched before frozen-weight rejection")

    from ctikernels.gdn.replay import cute, triton

    for backend in (cute, triton):
        monkeypatch.setattr(backend, "run", launch)
    p.weights[weight].requires_grad_(True)
    with pytest.raises(ValueError, match="grad"):
        p.step(47)


@pytest.mark.parametrize(
    "batch,hk,hv", [(1, 1, 4), pytest.param(3, 1, 4, marks=pytest.mark.exhaustive), (4, 16, 48)]
)
@pytest.mark.parametrize("snapshot", [False, pytest.param(True, marks=pytest.mark.exhaustive)])
def test_explicit_completion_graph_self_cleaning_and_validation(batch, hk, hv, snapshot):
    # The production B4 geometry exceeds one resident CTA wave on sm_120.
    # Exercise early reader election with late CTAs, flushes and status reuse.
    from benchmark.kernel_tuning.gdn_replay_common import allocate

    from ..modules._replay_case import graph_buffers, projected

    p = portable.Pair(hk=hk, hv=hv, batch=batch)
    op, state = p.op, p.state
    if snapshot:
        op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            window=op.window,
            verify_length=op.verify_length,
            device="cuda",
            provider="cute",
            mode="delayed",
            checks="strict",
            cute_config=Config(snapshot_controls=True, sections=True, trusted=False),
        )
        p.op = op
        p.workspace = {
            n: t
            for n, t in allocate(op.workspace_spec("step", batch_size=batch), "cuda").items()
            if n != "out"
        }
    reference = clone(state)
    controls = graph_buffers(op, batch=batch)
    status = controls.status(0)
    status.sequence.fill_(2)
    slots, pos = p.slots.clone(), p.pos.clone()
    x = projected(op, batch)
    out, expected = (
        torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda"),
        torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda"),
    )
    workspace = {
        n: t
        for n, t in allocate(op.workspace_spec("step", batch_size=batch), "cuda").items()
        if n != "out"
    }

    def body():
        op.begin(status=status, layer_weights=(p.weights,))
        op.step(
            x,
            *p.weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            status=status,
            workspace=workspace,
        )

        op.end(status=status)
        pos.add_((slots >= 0).to(torch.int64))

    def settle(sequence):
        result = op.read_status_copy(*status.copy_to_host(), expected_sequence=sequence)
        assert result.ok
        op.settle(status=status, result=result)

    # Eager warmup and comparison, then reuse the same graph across mixed phases.
    body()
    settle(2)
    op.step(
        x,
        *p.weights,
        slot_indices=slots,
        positions=p.pos,
        state=reference,
        out=expected,
        status=p.status,
        workspace=p.workspace,
    )
    p.pos.add_(1)
    same(state, reference)
    status.sequence.fill_(3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for t in range(3, 3 + 3 * op.window):
        status.sequence.fill_(t)
        if batch > 1:
            slots[-1] = -1 if t % 3 else batch - 1
        # Position controls stay attached to their rows, including inactive rows.
        graph.replay()
        op.step(
            x,
            *p.weights,
            slot_indices=slots,
            positions=p.pos,
            state=reference,
            out=expected,
            status=p.status,
            workspace=p.workspace,
        )
        p.pos.add_((slots >= 0).to(torch.int64))
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        same(state, reference)
        assert torch.equal(pos, p.pos)
        assert "arrivals" not in workspace or not workspace["arrivals"].count_nonzero().item()
        settle(t)
    # A fault in any active row suppresses ALL payload mutations.
    status.sequence.fill_(100)
    pos[0].add_(1)
    before = clone(state)
    graph.replay()
    same(state, before)
    assert not out.count_nonzero().item()
    assert "arrivals" not in workspace or not workspace["arrivals"].count_nonzero().item()
    result = op.read_status_copy(*status.copy_to_host(), expected_sequence=100)
    assert result.complete and result.codes[0] & 2
    # The completed record cannot be overwritten without settlement.
    saved = status.record.clone()
    status.validity.fill_(True)
    status.sequence.fill_(101)
    graph.replay()
    assert torch.equal(saved, status.record)
    assert status.validity.all().item()
    assert status.poison.item() & 64
    assert "arrivals" not in workspace or not workspace["arrivals"].count_nonzero().item()


def test_counter_free_mixed_phase_graph_across_many_ctas():
    """384 head CTAs exercise late readers, rotating cursors, int64 p and flushes."""
    from benchmark.kernel_tuning.gdn_replay_common import allocate

    from ..modules._replay_case import case, projected

    base, state, _, weights, slots, pos, status = case(
        device="cuda", hk=16, hv=48, batch=8, window=8
    )
    ops = [
        GatedDeltaNetReplay(
            num_k_heads=16,
            num_v_heads=48,
            window=8,
            verify_length=base.verify_length,
            device="cuda",
            provider="cute",
            mode="delayed",
            checks="strict",
            cute_config=Config(snapshot_controls=policy, trusted=not policy),
        )
        for policy in (False, True)
    ]
    states = [state, clone(state)]
    workspaces = [
        {
            n: t
            for n, t in allocate(op.workspace_spec("step", batch_size=8), "cuda").items()
            if n != "out"
        }
        for op in ops
    ]
    outputs = [torch.empty(8, 48 * 128, dtype=torch.bfloat16, device="cuda") for _ in ops]
    x = projected(base, 8)
    rows = torch.arange(8, device="cuda", dtype=torch.int32)

    def body():
        for op, live, ws, out in zip(ops, states, workspaces, outputs, strict=True):
            op.step(
                x,
                *weights,
                slot_indices=slots,
                positions=pos,
                state=live,
                out=out,
                status=status,
                workspace=ws,
            )
        pos.add_((slots >= 0).to(torch.int64))

    body()  # Warm both specializations before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for t in range(25):
        slots.copy_(torch.where((rows + t) % 5 == 0, -1, rows))
        x.copy_(projected(base, 8, seed=100 + t))
        graph.replay()
        same(*states)
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        assert not outputs[1][slots < 0].count_nonzero().item()
        assert not status.code.item() and not status.poison.item()
    # All rows must be suppressed even if only the last batch row is invalid.
    slots.copy_(rows)
    pos[-1].add_(1)
    before = [clone(live) for live in states]
    graph.replay()
    for live, saved, out in zip(states, before, outputs, strict=True):
        same(live, saved)
        assert not out.count_nonzero().item()
    assert status.code.item() & 2


def _assert_bitwise(actual, expected, *, phase, step):
    """Report the first differing element, with an integer IEEE ULP distance."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if torch.equal(actual, expected):
        return
    a, b = actual.detach().cpu().contiguous(), expected.detach().cpu().contiguous()
    flat = int((a != b).flatten().nonzero()[0, 0])
    index, remainder = [], flat
    for size in reversed(a.shape):
        index.append(remainder % size)
        remainder //= size
    index = tuple(reversed(index))
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    sign = 1 << (15 if dtype == torch.int16 else 31)

    def ordered(value):
        bits = int(value.reshape(1).view(dtype)[0])
        return -(bits & (sign - 1)) if bits < 0 else bits

    distance = abs(ordered(a[index]) - ordered(b[index]))
    pytest.fail(
        f"{phase}: first mismatch step={step}, element={index}, "
        f"actual={float(a[index]):.17g}, product={float(b[index]):.17g}, ulp={distance}"
    )


@pytest.mark.bitwise
@pytest.mark.parametrize(
    "hk,hv,batch,kind",
    ci_cases(
        [(16, 32), (16, 48)],
        [1, 2, 3, 4, 5, 6, 7, 8],
        ["ordinary", "heavy"],
        # (16, 32, 8) keeps the RTX 5090 BV=64, warps=8 specialization in CI.
        core=[(16, 32, 1, "ordinary"), (16, 48, 4, "heavy"), (16, 32, 8, "ordinary")],
    ),
)
def test_product_bitwise_multiwindow(hk, hv, batch, kind):
    """Decode outputs and automatic checkpoints match product at the same B.

    Fixed dispatch throughout the raw ring's lifetime, including materialization
    and transaction phases.
    """
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, state, history = case_inputs(hk, hv, batch, batch, 32, kind, 11, "cuda")
    pos = torch.full((batch,), 2**33 + 7, dtype=torch.int64, device="cuda")
    base = Product(hk, hv, weights.args(), state, history)
    replay = Replay("cute", hk, hv, batch, 8, 8, weights.args(), state, history, pos)
    for t, x in enumerate(xs):
        _assert_bitwise(replay.step(x), base.step(x), phase="decode/output", step=t)
        if (t + 1) % 8 == 0:
            _assert_bitwise(
                replay.state.checkpoint[:batch], base.state, phase="automatic-flush/state", step=t
            )
        _assert_bitwise(
            replay.state.conv_history[:batch], base.history, phase="decode/history", step=t
        )
    replay.check()


DFLASH_Q = [4, 5, 8, 9, 12, 13, 16, 17]


@pytest.mark.bitwise
@pytest.mark.parametrize(
    "q",
    ci_cases(DFLASH_Q, core=[(5,)]),
)
def test_product_bitwise_any_q_per_row_acceptance(q):
    """Any Q in [2, 32]: verify outputs and per-row commits m=0..Q are product bits."""
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    hk, hv, batch, window = 16, 32, 4, 8
    weights, xs, initial, history = case_inputs(
        hk, hv, batch, batch, window + 1 + q, "ordinary", 29, "cuda"
    )
    positions = torch.full((batch,), 2**33 + 7, dtype=torch.int64, device="cuda")
    product = Product(hk, hv, weights.args(), initial, history)
    replay = Replay("cute", hk, hv, batch, window, q, weights.args(), initial, history, positions)
    # The prefix crosses the automatic flush, leaving one buffered token.
    for t in range(window + 1):
        replay.step(xs[t])
        product.step(xs[t])
    live = clone(replay.state)
    states, histories, outputs = [product.state.clone()], [product.history.clone()], []
    for j in range(q):
        outputs.append(product.step(xs[window + 1 + j]).clone())
        states.append(product.state.clone())
        histories.append(product.history.clone())
    drafts = xs[window + 1 : window + 1 + q].transpose(0, 1)
    _assert_bitwise(replay.verify(drafts), torch.stack(outputs, 1), phase="verify", step=q)
    same(replay.state, live)
    tentative = clone(replay.pending)
    op = replay.op
    rounds = -(-(q + 1) // batch)
    for j in range(rounds):
        # Each row takes a different accepted length; across rounds rows cover 0..Q.
        lengths = [(j * batch + r) % (q + 1) for r in range(batch)]
        accepted = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        state, pending = clone(live), clone(tentative)
        op.commit(
            *replay.weights,
            slot_indices=replay.slots,
            accepted_lengths=accepted,
            state=state,
            pending=pending,
            status=replay.status,
            workspace=replay.phase_workspace,
        )
        replay.check()
        for r, m in enumerate(lengths):
            _assert_bitwise(state.checkpoint[r], states[m][r], phase="commit/state", step=(q, m))
            _assert_bitwise(
                state.conv_history[r], histories[m][r], phase="commit/history", step=(q, m)
            )
            meta = state.metadata[r]
            assert meta[0].item() == meta[1].item() == live.metadata[r, 1].item() + m
            assert meta[3].item() == 0
    # Reject leaves the live state untouched.
    state, pending = clone(live), clone(tentative)
    op.reject(slot_indices=replay.slots, state=state, pending=pending, status=replay.status)
    replay.check()
    same(state, live)


@pytest.mark.parametrize(
    "batch,kind",
    ci_cases([1, 4], ["ordinary", "heavy"], core=[(1, "ordinary"), (4, "heavy")]),
)
def test_fp64_graph_oracle_matches_eager(batch, kind):
    from benchmark.kernel_tuning.gdn_replay_common import Oracle
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, state, history = case_inputs(1, 2, batch, batch, 8, kind, 11, "cuda")
    eager = Oracle(1, weights.args(), state, history)
    graph = Oracle(1, weights.args(), state, history, graph=True)
    retained = []
    for x in xs[:3]:
        actual, expected = graph.step(x), eager.step(x)
        assert torch.equal(actual, expected)
        assert torch.equal(graph.canonical(), eager.canonical())
        assert torch.equal(graph.history, eager.history)
        retained.append((actual, expected.clone()))
    # Replacing state/history is used by stagger, chunked handoff and verify.
    for ref in (graph, eager):
        ref.state = ref.state.float().double() * 0.5
        ref.history = history.clone()
    drafts = xs[3:].transpose(0, 1).contiguous()
    for lengths in ([0] * batch, [5] * batch, [0, 1, 3, 5][:batch]):
        accepted = torch.tensor(lengths, device="cuda")
        assert torch.equal(graph.verify(drafts, accepted), eager.verify(drafts, accepted))
        assert torch.equal(graph.canonical(), eager.canonical())
        assert torch.equal(graph.history, eager.history)
        assert torch.equal(graph.step(xs[0]), eager.step(xs[0]))
    assert all(torch.equal(actual, expected) for actual, expected in retained)


@pytest.mark.bitwise
@pytest.mark.parametrize(
    "hk,hv,batch,kind,window,q",
    ci_cases(
        [(16, 32), (16, 48)],
        [1, 2, 4],
        ["ordinary", "heavy"],
        [(8, 8), (8, 16), (16, 32), (4, 2)],
        core=[(16, 32, 4, "heavy", 8, 16)],
    ),
)
def test_product_bitwise_transaction_phases(hk, hv, batch, kind, window, q):
    """Every output/prefix uses product B=N, across automatic flush and ring wrap."""
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, initial, history = case_inputs(
        hk, hv, batch, batch, 3 * window + 1 + q, kind, 19, "cuda"
    )
    positions = torch.full((batch,), 2**33 + 7, dtype=torch.int64, device="cuda")
    for prefix in (window - 1, window, window + 1, 3 * window + 1):
        product = Product(hk, hv, weights.args(), initial, history)
        replay = Replay(
            "cute", hk, hv, batch, window, q, weights.args(), initial, history, positions
        )
        for t in range(prefix):
            replay.step(xs[t])
            product.step(xs[t])
        live = clone(replay.state)
        states, histories = [product.state.clone()], [product.history.clone()]
        outputs = []
        for j in range(q):
            outputs.append(product.step(xs[prefix + j]).clone())
            states.append(product.state.clone())
            histories.append(product.history.clone())
        drafts = xs[prefix : prefix + q].transpose(0, 1)
        _assert_bitwise(replay.verify(drafts), torch.stack(outputs, 1), phase="verify", step=prefix)
        same(replay.state, live)
        tentative = clone(replay.pending)
        op, status = replay.op, replay.status
        for m in (0, min(3, q - 1), q):
            accepted = torch.full((batch,), m, dtype=torch.int32, device="cuda")
            for k in sorted({0, m // 2, m}):
                state, pending = clone(live), clone(tentative)
                offset = torch.full((batch,), k, dtype=torch.int64, device="cuda")
                op.export_snapshot(
                    *replay.weights,
                    pairs=replay.pairs,
                    state=state,
                    status=status,
                    snapshot_valid=replay.valid,
                    pending=pending,
                    accepted_lengths=accepted,
                    offsets=offset,
                    workspace=replay.phase_workspace,
                )
                _assert_bitwise(
                    state.checkpoint[batch:], states[k], phase="export/state", step=(prefix, m, k)
                )
                _assert_bitwise(
                    state.conv_history[batch:],
                    histories[k],
                    phase="export/history",
                    step=(prefix, m, k),
                )
                assert replay.valid.all()
                for name in ("checkpoint", "conv_history", "ring", "metadata"):
                    assert torch.equal(getattr(state, name)[:batch], getattr(live, name)[:batch]), (
                        name
                    )
                assert torch.equal(pending.metadata[:, 5], torch.full_like(accepted, 2))
                assert torch.equal(pending.metadata[:, 6], accepted)
                # Commit accepts the resolved descriptor and independently captures
                # the same prefix before canonicalizing the full accepted frontier.
                op.commit(
                    *replay.weights,
                    slot_indices=replay.slots,
                    accepted_lengths=accepted,
                    state=state,
                    pending=pending,
                    status=status,
                    snapshot_rows=replay.pairs[:, 1].contiguous(),
                    snapshot_offsets=offset,
                    snapshot_valid=replay.valid,
                    workspace=replay.phase_workspace,
                )
                _assert_bitwise(
                    state.checkpoint[:batch], states[m], phase="commit/state", step=(prefix, m, k)
                )
                _assert_bitwise(
                    state.conv_history[:batch],
                    histories[m],
                    phase="commit/history",
                    step=(prefix, m, k),
                )
                _assert_bitwise(
                    state.checkpoint[batch:], states[k], phase="capture/state", step=(prefix, m, k)
                )
                _assert_bitwise(
                    state.conv_history[batch:],
                    histories[k],
                    phase="capture/history",
                    step=(prefix, m, k),
                )
                assert replay.valid.all()
        flushed = clone(live)
        op.export_snapshot(
            *replay.weights,
            pairs=replay.pairs,
            state=flushed,
            status=status,
            snapshot_valid=replay.valid,
            workspace=replay.phase_workspace,
        )
        _assert_bitwise(flushed.checkpoint[batch:], states[0], phase="export/frontier", step=prefix)
        _assert_bitwise(
            flushed.conv_history[batch:], histories[0], phase="export/frontier-history", step=prefix
        )
        assert torch.equal(flushed.checkpoint[:batch], live.checkpoint[:batch])
        assert torch.equal(flushed.metadata, live.metadata)
        assert torch.equal(flushed.ring, live.ring)
        op.flush(
            *replay.weights,
            slot_indices=replay.slots,
            state=flushed,
            status=status,
            workspace=replay.phase_workspace,
        )
        _assert_bitwise(flushed.checkpoint[:batch], states[0], phase="flush/state", step=prefix)
        _assert_bitwise(
            flushed.conv_history[:batch], histories[0], phase="flush/history", step=prefix
        )
        bridged = clone(live)
        op.bridge_step(
            xs[prefix],
            *replay.weights,
            slot_indices=replay.slots,
            positions=replay.pos,
            state=bridged,
            out=replay.out,
            status=status,
            workspace=replay.phase_workspace,
        )
        _assert_bitwise(replay.out, outputs[0], phase="bridge/output", step=prefix)
        _assert_bitwise(bridged.checkpoint[:batch], states[1], phase="bridge/state", step=prefix)
        _assert_bitwise(
            bridged.conv_history[:batch], histories[1], phase="bridge/history", step=prefix
        )
        replay.check()


def _portable_phase_pair(q=8):
    from benchmark.kernel_tuning.gdn_replay_common import Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, initial, history = case_inputs(1, 4, 3, 3, 24, "ordinary", 23, "cuda")
    pos = torch.full((3,), 2**33 + 7, dtype=torch.int64, device="cuda")
    runs = [
        # Error codes and status records are strict-mode semantics; the portable
        # provider is delayed, so CuTe takes the delayed step on every device.
        Replay(
            p,
            1,
            4,
            3,
            4,
            q,
            weights.args(),
            initial,
            history,
            pos,
            checks="strict",
            mode="delayed" if p == "cute" else "auto",
        )
        for p in ("cute", "triton")
    ]
    return runs, xs


def _same_phase_controls(a, b):
    names = ["conv_history", "metadata", "snapshot_metadata", "pool_epoch"]
    # A ringless CuTe allocation (every bucket #25/immediate) has no ring to compare.
    if a.state.ring.shape[1] and b.state.ring.shape[1]:
        names.append("ring")
    for name in names:
        x, y = getattr(a.state, name), getattr(b.state, name)
        if name == "ring":
            x, y = live_ring(a.state), live_ring(b.state)
        assert torch.equal(x, y), name
    same(a.pending, b.pending)
    for name in ("records", "validity", "sequence", "poison"):
        assert torch.equal(getattr(a.status, name), getattr(b.status, name)), name
    assert torch.equal(a.valid, b.valid)


@pytest.mark.parametrize(
    "q",
    ci_cases([8, 16], core=[(8,)]),
)
def test_phase_storage_matches_portable(q):
    """Exact raw/descriptor semantics, independent of the provider's FP recipe."""
    runs, xs = _portable_phase_pair(q)
    for replay in runs:
        replay.slots[1] = -1
    for cycle, m in enumerate((0, 3, q)):
        for t in range(9):
            for replay in runs:
                replay.step(xs[t])
            _same_phase_controls(*runs)
        for replay in runs:
            before = clone(replay.state)
            replay.verify(xs[:q].transpose(0, 1))
            same(before, replay.state)
            assert not replay.verify_out[1].count_nonzero()
            # Repeated source lookup differs from pending row order; the middle
            # inactive pair must not resolve its descriptor or touch its payload.
            replay.pairs.copy_(
                torch.tensor([[2, 3], [-1, 4], [2, 5]], dtype=torch.int32, device="cuda")
            )
            replay.op.export_snapshot(
                *replay.weights,
                pairs=replay.pairs,
                state=replay.state,
                status=replay.status,
                snapshot_valid=replay.valid,
                pending=replay.pending,
                accepted_lengths=torch.tensor([m, -900, m], dtype=torch.int32, device="cuda"),
                offsets=torch.tensor([m, 0, m + 1], dtype=torch.int64, device="cuda"),
                workspace=replay.phase_workspace,
            )
        _same_phase_controls(*runs)
        for replay in runs:
            replay.resolve(torch.tensor([m, -900, m], dtype=torch.int32, device="cuda"))
        _same_phase_controls(*runs)
        for replay in runs:
            replay.verify(xs[:q].transpose(0, 1))
            before = clone(replay.state)
            replay.resolve(None, reject=True)
            same(before, replay.state)
            replay.flush()
            replay.op.bridge_step(
                xs[cycle],
                *replay.weights,
                slot_indices=replay.slots,
                positions=replay.pos,
                state=replay.state,
                out=replay.out,
                status=replay.status,
                workspace=replay.phase_workspace,
            )
            replay.pos.add_(1)
        _same_phase_controls(*runs)
        for replay in runs:
            replay.check()


@pytest.mark.parametrize(
    "fault,phase,code",
    [
        ("slot", "verify", 1),
        ("position", "verify", 2),
        ("combined", "verify", 3),
        ("overflow", "verify", 2),
        ("metadata", "flush", 1),
        ("accepted", "commit", 4),
        ("descriptor", "reject", 8),
        ("snapshot", "commit", 16),
        ("offset", "export_snapshot", 4),
        ("missing", "export_snapshot", 8),
        ("resolved", "commit", 8),
        ("poison", "verify", 32),
        ("collision", "verify", 0),
        ("old_code", "verify", 4),
    ],
)
def test_phase_errors_match_portable(fault, phase, code):
    runs, xs = _portable_phase_pair()
    for replay in runs:
        replay.verify(xs[:8].transpose(0, 1))
        accepted = torch.full((3,), 3, dtype=torch.int32, device="cuda")
        offsets = torch.zeros(3, dtype=torch.int64, device="cuda")
        snaps = replay.pairs[:, 1].contiguous()
        if fault == "slot":
            replay.slots[-1] = 99
        elif fault == "position":
            replay.pos[-1] += 1
        elif fault == "combined":
            replay.slots[-1] = 99
            replay.pos[0] += 1
        elif fault == "overflow":
            replay.pos[-1] = 2**63 - 4
            replay.state.metadata[-1, :2] = 2**63 - 4
        elif fault == "metadata":
            replay.state.metadata[-1, 3] = 4
        elif fault == "accepted":
            accepted[-1] = 9
        elif fault == "descriptor":
            replay.pending.metadata[-1, 1] += 1
        elif fault == "snapshot":
            snaps[-1] = 99
        elif fault == "offset":
            offsets[-1] = -1
        elif fault == "missing":
            replay.pending.metadata[-1, 5] = 4
        elif fault == "resolved":
            replay.pending.metadata[-1, 5:7] = torch.tensor([2, 2], device="cuda")
        elif fault == "poison":
            replay.status.poison.fill_(1)
        elif fault == "collision":
            replay.status.sequence.add_(1)
        elif fault == "old_code":
            replay.status.code.fill_(4)
        before, pending_before = clone(replay.state), clone(replay.pending)
        replay.valid.fill_(True)
        if phase == "verify":
            replay.verify(xs[:8].transpose(0, 1))
            assert not replay.verify_out.count_nonzero()
        elif phase == "commit":
            replay.op.commit(
                *replay.weights,
                slot_indices=replay.slots,
                state=replay.state,
                pending=replay.pending,
                accepted_lengths=accepted,
                status=replay.status,
                snapshot_rows=snaps,
                snapshot_offsets=offsets,
                snapshot_valid=replay.valid,
                workspace=replay.phase_workspace,
            )
        elif phase == "export_snapshot":
            replay.op.export_snapshot(
                *replay.weights,
                pairs=replay.pairs,
                state=replay.state,
                status=replay.status,
                pending=replay.pending,
                accepted_lengths=accepted,
                offsets=offsets,
                snapshot_valid=replay.valid,
                workspace=replay.phase_workspace,
            )
        elif phase == "flush":
            replay.flush()
        else:
            replay.resolve(None, reject=True)
        same(replay.state, before)
        same(replay.pending, pending_before)
        assert replay.status.code.item() == code
        assert replay.status.poison.item() != 0
    _same_phase_controls(*runs)


@pytest.mark.parametrize(
    "hk,hv,window,q,batch",
    ci_cases(
        [(8, 16, 4, 2, 1), (8, 24, 8, 8, 2), (16, 32, 8, 16, 4), (16, 48, 16, 32, 1)]
        + [(8, 16, 8, q, 2) for q in (4, 5, 9, 12, 13, 17)]
        + [(8, 24, 4, 8, b) for b in (1, 2, 3, 4)],
        core=[(8, 16, 8, 4, 2)],
    ),
)
def test_portable_contract_all_acceptances(hk, hv, window, q, batch):
    portable.test_boundaries_all_acceptances_offsets_and_prefix_cache(hk, hv, window, q, batch)


@pytest.mark.parametrize("input_class", ["ordinary", "heavy", "zero_keys"])
def test_portable_contract_prefill_handoff(input_class):
    portable.test_prefill_handoff_and_immediate_fp64_oracle(input_class)


def test_portable_contract_graph_failure():
    portable.test_graph_partial_failure_poison_collision_and_recovery()


@pytest.mark.parametrize(
    "tile,batch,q,hk,hv",
    ci_cases(
        [32, 64, 128],
        [1, 3],
        [4, 5, 8, 9, 16],
        [(16, 32), (16, 48)],
        core=[(128, 1, 16, 16, 32)],
    ),
)
def test_tilesplit_all_phases_bitwise(tile, batch, q, hk, hv, monkeypatch):
    from ctikernels.gdn.replay import ReplayStatus
    from ctikernels.gdn.replay.cute import _phase_launch

    from ..modules._replay_case import projected

    op, state, pending, weights, slots, pos, status = portable.case(
        hk=hk, hv=hv, window=8, q=q, batch=batch, device="cuda"
    )
    workspace = {
        name: torch.full(spec.shape, spec.initial_value or 0, dtype=spec.dtype, device="cuda")
        for name, spec in op.workspace_spec("step", batch_size=batch).items()
        if name != "out"
    }
    # Give verify a nonzero checkpoint and a three-token unflushed ring tail.
    state.checkpoint[:batch].copy_(
        torch.randn(batch, hv, 128, 128, generator=torch.Generator().manual_seed(812))
        .mul_(0.05)
        .to("cuda")
    )
    for seed in range(3):
        out = torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda")
        op.step(
            projected(op, batch, seed=91 + seed),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=out,
            status=status,
            workspace=workspace,
        )
        pos.add_(1)
    states, pending_states = [clone(state), state], [clone(pending), pending]
    statuses = [
        ReplayStatus(
            status.records.clone(),
            status.validity.clone(),
            status.sequence.clone(),
            status.poison.clone(),
        ),
        status,
    ]
    if batch > 1:
        slots[-1] = -1
    snapshots = torch.arange(batch, dtype=torch.int32, device="cuda") + state.metadata.shape[0]
    pairs = torch.stack((slots, snapshots), 1)
    accepted = torch.full((batch,), q - 1, dtype=torch.int32, device="cuda")
    offsets = torch.full((batch,), q // 2, dtype=torch.int64, device="cuda")

    def equal_bytes(a, b):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def call(mode, *, use_pending=False, output=False, snapshot=False, **kw):
        outputs, validities = [], []
        for i, selected in enumerate((128, tile)):
            monkeypatch.setattr(_phase_launch, "PHASE_TILE", selected)
            args = dict(kw, state=states[i], status=statuses[i])
            if use_pending:
                args["pending"] = pending_states[i]
            if output:
                shape = (batch, q, hv * 128) if mode == "verify" else (batch, hv * 128)
                outputs.append(torch.full(shape, 7, dtype=torch.bfloat16, device="cuda"))
                args["out"] = outputs[-1]
            if snapshot:
                validities.append(torch.ones(batch, dtype=torch.bool, device="cuda"))
                args["snapshot_valid"] = validities[-1]
            if mode == "step":
                args["workspace"] = workspace
            if mode in ("verify", "bridge_step", "step"):
                x = args.pop("projected")
                getattr(op, mode)(x, *weights, **args)
            else:
                getattr(op, mode)(*weights, **args)
        same(states[0], states[1])
        same(pending_states[0], pending_states[1])
        for field in ("records", "validity", "sequence", "poison"):
            equal_bytes(getattr(statuses[0], field), getattr(statuses[1], field))
        for values in (outputs, validities):
            if values:
                equal_bytes(*values)
        return outputs

    drafts = projected(op, batch, length=q, seed=712)
    call(
        "verify",
        use_pending=True,
        output=True,
        projected=drafts,
        slot_indices=slots,
        positions=pos[:, None] + torch.arange(q, device="cuda"),
    )
    call(
        "export_snapshot",
        use_pending=True,
        snapshot=True,
        pairs=pairs,
        accepted_lengths=accepted,
        offsets=offsets,
    )
    # Cover capture at end for B=1, and before/interior to replay for B=3.
    capture_offsets = offsets.clone()
    capture_offsets[0] = q - 1 if batch == 1 else 0
    call(
        "commit",
        use_pending=True,
        snapshot=True,
        slot_indices=slots,
        accepted_lengths=accepted,
        snapshot_rows=snapshots,
        snapshot_offsets=capture_offsets,
    )
    pos.add_(accepted)
    call(
        "bridge_step",
        output=True,
        projected=projected(op, batch, seed=713),
        slot_indices=slots,
        positions=pos,
    )
    pos.add_(1)
    for seed in (714, 715):
        call(
            "step",
            output=True,
            projected=projected(op, batch, seed=seed),
            slot_indices=slots,
            positions=pos,
        )
        pos.add_(1)
    call("export_snapshot", snapshot=True, pairs=pairs)
    call("flush", slot_indices=slots)
    assert status.code.item() == 0 and status.poison.item() == 0

    # Even a whole masked cluster must reach every barrier and zero its outputs.
    masked = torch.full_like(slots, -1)
    for mode in ("verify", "bridge_step"):
        outputs = call(
            mode,
            use_pending=mode == "verify",
            output=True,
            projected=drafts if mode == "verify" else drafts[:, 0].contiguous(),
            slot_indices=masked,
            positions=pos[:, None] + torch.arange(q, device="cuda") if mode == "verify" else pos,
        )
        assert torch.count_nonzero(outputs[1]).item() == 0

    # Invalid controls poison the whole batch; subsequent phases stay suppressed.
    before = clone(states[1])
    outputs = call(
        "verify",
        use_pending=True,
        output=True,
        projected=drafts,
        slot_indices=slots,
        positions=pos[:, None] + torch.arange(q, device="cuda") + 1,
    )
    assert status.code.item() & 2 and torch.count_nonzero(outputs[1]).item() == 0
    call(
        "bridge_step",
        output=True,
        projected=drafts[:, 0].contiguous(),
        slot_indices=slots,
        positions=pos,
    )
    call(
        "commit",
        use_pending=True,
        snapshot=True,
        slot_indices=slots,
        accepted_lengths=accepted,
        snapshot_rows=snapshots,
        snapshot_offsets=capture_offsets,
    )
    call("export_snapshot", snapshot=True, pairs=pairs)
    call("flush", slot_indices=slots)
    same(before, states[1])


@pytest.mark.bitwise
@pytest.mark.parametrize("tile", [128, 64, 32])
@pytest.mark.parametrize("speculative", [False, True])
def test_step_unconsumed_physical_rows_do_not_influence_output(tile, speculative):
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    hk, hv, batch, window = 16, 32, 1, 8
    weights, xs, state, history = case_inputs(hk, hv, batch, batch, 24, "ordinary", 11, "cuda")
    base = Product(hk, hv, weights.args(), state, history)
    replay = Replay(
        "cute",
        hk,
        hv,
        batch,
        window,
        8,
        weights.args(),
        state,
        history,
        torch.zeros(batch, dtype=torch.int64, device="cuda"),
        cute_config=Config(value_tile=tile, speculative_ring=speculative),
    )
    for t, x in enumerate(xs):
        cursor, n = (int(v) for v in replay.state.metadata[0, 2:4])
        consumed = {(cursor + i - n - 3) % (window + 3) for i in range(n + 3)}
        for rr in set(range(window + 3)) - consumed:
            replay.state.ring[0, rr].fill_(float("nan"))
        _assert_bitwise(replay.step(x), base.step(x), phase="unused-ring/output", step=t)
        _assert_bitwise(
            replay.state.conv_history[:batch], base.history, phase="unused-ring/history", step=t
        )
        if (t + 1) % window == 0:
            _assert_bitwise(
                replay.state.checkpoint[:batch], base.state, phase="unused-ring/state", step=t
            )
    replay.check()


@pytest.mark.bitwise
@pytest.mark.parametrize("batch", range(1, 9))
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("kind", ["ordinary", "heavy"])
@pytest.mark.parametrize("tile", [128, 64, 32])
def test_immediate_every_step_product_bits(batch, hk, hv, kind, tile):
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, initial, history = case_inputs(hk, hv, batch, batch, 17, kind, 37, "cuda")
    pos = torch.full((batch,), 2**33 + 7, dtype=torch.int64, device="cuda")
    product = Product(hk, hv, weights.args(), initial, history)
    replay = Replay(
        "cute",
        hk,
        hv,
        batch,
        8,
        8,
        weights.args(),
        initial,
        history,
        pos,
        cute_config=Config(immediate=True, value_tile=tile),
    )
    assert replay.state.ring.numel() == 0
    cursor = replay.state.metadata[:, 2].clone()
    for t, x in enumerate(xs):
        _assert_bitwise(replay.step(x), product.step(x), phase="immediate/output", step=t)
        _assert_bitwise(
            replay.state.checkpoint[:batch], product.state, phase="immediate/state", step=t
        )
        _assert_bitwise(
            replay.state.conv_history[:batch], product.history, phase="immediate/history", step=t
        )
        assert torch.equal(replay.state.metadata[:, 0], replay.state.metadata[:, 1])
        assert torch.equal(replay.state.metadata[:, 2], cursor)
        assert not replay.state.metadata[:, 3].any()
    replay.check()


@pytest.mark.parametrize("tile", [32, 64, 128])
def test_switch_after_flush_and_reject_buffered_slot(tile):
    from benchmark.kernel_tuning.gdn_replay_common import Product, Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, initial, history = case_inputs(16, 32, 2, 2, 5, "ordinary", 37, "cuda")
    pos = torch.zeros(2, dtype=torch.int64, device="cuda")
    product = Product(16, 32, weights.args(), initial, history)
    replay = Replay(
        "cute", 16, 32, 2, 8, 8, weights.args(), initial, history, pos, cute_config=Config()
    )
    flag = torch.empty(1, dtype=torch.bool, device="cuda")
    replay.op.ring_empty(replay.state, replay.slots, flag)
    assert flag.item()
    for x in xs[:3]:
        _assert_bitwise(replay.step(x), product.step(x), phase="switch/delayed", step=0)
    replay.op.ring_empty(replay.state, replay.slots, flag)
    assert not flag.item()
    replay.flush()
    replay.op.ring_empty(replay.state, replay.slots, flag)
    assert flag.item()
    replay.op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda",
        window=8,
        verify_length=8,
        provider="cute",
        checks="strict",
        cute_config=Config(immediate=True, value_tile=tile),
    )
    replay.workspace["arrivals"] = replay.workspace["arrivals"][:1]
    saved_ring = replay.state.ring.clone()
    cursor = replay.state.metadata[:, 2].clone()
    _assert_bitwise(replay.step(xs[3]), product.step(xs[3]), phase="switch/immediate", step=3)
    _assert_bitwise(replay.state.checkpoint[:2], product.state, phase="switch/state", step=3)
    _assert_bitwise(replay.state.conv_history[:2], product.history, phase="switch/history", step=3)
    assert torch.equal(replay.state.ring, saved_ring)
    assert torch.equal(replay.state.metadata[:, 2], cursor)
    # A valid delayed descriptor with one buffered token must fail the whole batch.
    replay.state.metadata[1, 3] = 1
    replay.state.metadata[1, 0] -= 1
    before = clone(replay.state)
    replay.step(xs[4])
    same(replay.state, before)
    assert replay.status.code.item() == 1
    assert replay.status.poison.item() == 1
    assert not replay.out.any()


@pytest.mark.parametrize("batch", [0, 1, 8, 65])
def test_ring_empty_device_graph(batch):
    from ..modules._replay_case import case

    _, state, _, _, _, _, _ = case(device="cuda", batch=max(1, batch))
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        device="cuda",
        window=4,
        verify_length=8,
        provider="cute",
        checks="strict",
        cute_config=Config(),
    )
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    flag = torch.empty(1, dtype=torch.bool, device="cuda")
    op.ring_empty(state, slots, flag)
    assert flag.item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.ring_empty(state, slots, flag)
    graph.replay()
    assert flag.item()
    if batch:
        state.metadata[batch - 1, 3] = 1
        graph.replay()
        assert not flag.item()
        slots[-1] = -1
        graph.replay()
        assert flag.item()
        slots[-1] = state.metadata.shape[0]
        graph.replay()
        assert not flag.item()
        slots[-1] = -2
        graph.replay()
        assert not flag.item()
        slots[-1] = batch - 1
        state.metadata[batch - 1, 3] = 0
        state.metadata[batch - 1, 0] -= 1
        graph.replay()
        assert not flag.item()
        state.metadata[batch - 1, 0] += 1
        graph.replay()
        assert flag.item()


def _trusted_runner(hk, hv, batch, config, checks, *, seed=91):
    from benchmark.kernel_tuning.gdn_replay_common import Replay
    from benchmark.kernel_tuning.gdn_replay_inputs import case_inputs

    weights, xs, initial, history = case_inputs(
        hk, hv, batch, batch, 3 * 8 + 3, "ordinary", seed, "cuda"
    )
    runner = Replay(
        "cute",
        hk,
        hv,
        batch,
        8,
        8,
        weights.args(),
        initial,
        history,
        torch.full((batch,), 2**33 + 7, dtype=torch.int64, device="cuda"),
        num_slots=batch + 1,
        cute_config=config,
        checks=checks,
    )
    return runner, xs


def _trusted_call(runner, x, *, status=None):
    runner.op.step(
        x,
        *runner.weights,
        slot_indices=runner.slots,
        positions=runner.pos,
        state=runner.state,
        out=runner.out,
        workspace=runner.workspace,
        status=status,
    )


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("batch", [1, 3, 4])
@pytest.mark.parametrize("kind", ["delayed", "immediate", "flush"])
@pytest.mark.parametrize("graph", [False, True])
def test_trusted_strict_multiwindow_device_controls(hk, hv, batch, kind, graph):
    config = Config(
        value_tile=32 if batch == 1 else 64,
        immediate=kind == "immediate",
        flush_always=kind == "flush",
    )
    trusted, xs = _trusted_runner(hk, hv, batch, config, "trusted")
    strict, _ = _trusted_runner(hk, hv, batch, config, "strict")
    # Include untouched physical rows, inactive slots and snapshot payload in comparison.
    trusted.state = clone(strict.state)
    baseline = clone(strict.state)
    trusted.status.records.fill_(123)
    trusted.status.poison.fill_(31)
    trusted.status.sequence.zero_()
    trusted.status.validity.fill_(True)
    status_before = [
        t.clone()
        for t in (
            trusted.status.records,
            trusted.status.poison,
            trusted.status.sequence,
            trusted.status.validity,
        )
    ]
    x = xs[0].clone()

    def body():
        _trusted_call(trusted, x, status=trusted.status)

    body()  # Warm launcher without begin/status, then restore all state.
    for name in baseline.__dataclass_fields__:
        getattr(trusted.state, name).copy_(getattr(baseline, name))
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
    for t, source in enumerate(xs):
        # All controls change at fixed addresses using device operations only.
        assignment = (torch.arange(batch, device="cuda", dtype=torch.int32) + t) % batch
        if t % 5 == 0:
            assignment[-1] = -1
        strict.slots.copy_(assignment)
        trusted.slots.copy_(assignment)
        strict.pos.copy_(strict.state.metadata[assignment.clamp_min(0).long(), 1])
        trusted.pos.fill_(-99)  # Trusted derives position from metadata, not this input.
        x.copy_(source)
        _trusted_call(strict, x, status=strict.status)
        if graph:
            g.replay()
        else:
            body()
        _assert_bitwise(trusted.out, strict.out, phase="trusted/output", step=t)
        same(trusted.state, strict.state)
        active = assignment >= 0
        assert (trusted.pos == -99).all()
        assert not trusted.out[~active].any()
        assert not trusted.workspace["arrivals"].any()
        for name in ("checkpoint", "conv_history", "ring", "metadata"):
            _assert_bitwise(
                getattr(trusted.state, name)[batch],
                getattr(baseline, name)[batch],
                phase="trusted/padded-slot",
                step=t,
            )
    for actual, before in zip(
        (
            trusted.status.records,
            trusted.status.poison,
            trusted.status.sequence,
            trusted.status.validity,
        ),
        status_before,
        strict=True,
    ):
        assert torch.equal(actual, before)


@pytest.mark.parametrize("slot", [-2, -1, 2, 999])
def test_trusted_unreadable_row_only_zeros_output(slot):
    runner, xs = _trusted_runner(16, 32, 1, Config(value_tile=32), "trusted")
    before = clone(runner.state)
    runner.slots.fill_(slot)
    _trusted_call(runner, xs[0])
    same(runner.state, before)
    assert not runner.out.any()
    assert not runner.workspace["arrivals"].any()


@pytest.mark.bitwise
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("first", ["delayed", "immediate", "product"])
@pytest.mark.parametrize("second", ["delayed", "immediate", "product"])
@pytest.mark.parametrize("k", [0, 1, 7])
def test_trusted_switch_mixed_buckets_graph(monkeypatch, hk, hv, first, second, k):
    from benchmark.kernel_tuning.gdn_replay_common import Product, allocate
    from ctikernels.gdn.replay.cute import effective_config
    from ctikernels.gdn.replay.cute._config import POLICY

    # Product policy entries in this mixed op must resolve to the flush bridge.
    device_name = torch.cuda.get_device_name()
    name = next((prefix for prefix, _, _ in POLICY if device_name.startswith(prefix)), device_name)
    batches = {"delayed": 1, "immediate": 2, "product": 4}
    monkeypatch.setitem(POLICY, (name, None, 1), {"immediate": False, "value_tile": 32})
    monkeypatch.setitem(POLICY, (name, None, 2), {"immediate": True, "value_tile": 32})
    monkeypatch.setitem(
        POLICY, (name, None, 4), {"immediate": True, "provider": "product", "value_tile": 64}
    )
    runner, xs = _trusted_runner(hk, hv, 1, Config(), "trusted")
    runner.op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        provider="cute",
        window=8,
        verify_length=8,
    )
    initial = clone(runner.state)
    reference, _ = _trusted_runner(hk, hv, 1, Config(immediate=True), "trusted")
    product = Product(hk, hv, runner.weights, initial.checkpoint[:1], initial.conv_history[:1])
    graphs, buffers, reference_ops = {}, {}, {}
    for mode, batch in batches.items():
        config = effective_config(runner.op, batch)
        assert not config.immediate
        assert config.flush_always == (mode != "delayed")
        assert runner.op.decode_route(batch) == ("delayed" if mode == "delayed" else "immediate")
        ws = allocate(runner.op.workspace_spec("step", batch_size=batch), "cuda")
        slots = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
        slots[0] = 0
        pos = torch.zeros(batch, dtype=torch.int64, device="cuda")
        x = torch.zeros(batch, xs.shape[-1], dtype=torch.bfloat16, device="cuda")

        def body(x=x, slots=slots, pos=pos, ws=ws):
            runner.op.step(
                x,
                *runner.weights,
                slot_indices=slots,
                positions=pos,
                state=runner.state,
                out=ws["out"],
                workspace=ws,
            )

        body()
        for field in initial.__dataclass_fields__:
            getattr(runner.state, field).copy_(getattr(initial, field))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        graphs[mode], buffers[mode] = graph, (x, ws["out"], slots, pos)
        reference_ops[mode] = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device="cuda",
            provider="cute",
            cute_config=Config(immediate=True, value_tile=config.value_tile),
        )
    schedule = [first] * k + [second] * (3 * 8)
    for t, mode in enumerate(schedule):
        source = xs[t % len(xs)]
        buffers[mode][0][0].copy_(source[0])
        graphs[mode].replay()
        x, actual, slots, pos = buffers[mode]
        reference.op = reference_ops[mode]
        reference.slots, reference.pos = slots, pos.clone()
        # Match each bucket's product arithmetic plan in the all-immediate oracle.
        reference.out = torch.empty_like(actual)
        reference.workspace = {"arrivals": torch.zeros_like(slots)}
        _trusted_call(reference, x)
        product.slots = slots
        expected = product.step(x)
        # A delayed->bridge switch replays tokens buffered under one bucket with the
        # next bucket's product plan, which may differ by B (default tier, not
        # bitwise across buckets). Check that step within FP32 tolerance, then resync
        # both oracles so every later step stays bitwise.
        if first == "delayed" and second != "delayed" and k > 0 and t == k:
            torch.testing.assert_close(
                actual.float(),
                (expected if second == "product" else reference.out).float(),
                rtol=8e-3,
                atol=1e-4,
            )
            torch.testing.assert_close(
                runner.state.checkpoint[:1], reference.state.checkpoint[:1], rtol=1e-5, atol=1e-7
            )
            reference.state.checkpoint[:1].copy_(runner.state.checkpoint[:1])
            product.state.copy_(runner.state.checkpoint[:1])
            continue
        _assert_bitwise(
            actual,
            expected if "product" in (first, second) else reference.out,
            phase="switch/output",
            step=t,
        )
        assert not actual[1:].any()
        _assert_bitwise(
            runner.state.conv_history[:1],
            reference.state.conv_history[:1],
            phase="switch/history",
            step=t,
        )
        # Only compare checkpoint when this step has materialized it.
        if not runner.state.metadata[0, 3].item():
            _assert_bitwise(
                runner.state.checkpoint[:1],
                reference.state.checkpoint[:1],
                phase="switch/checkpoint",
                step=t,
            )
            if "product" in (first, second):
                _assert_bitwise(
                    runner.state.checkpoint[:1],
                    product.state,
                    phase="switch/product-checkpoint",
                    step=t,
                )
        if t == k - 1:
            assert runner.state.metadata[0, 3].item() == (k if first == "delayed" else 0)


@pytest.mark.bitwise
@pytest.mark.parametrize(
    "first,second", [("product", "immediate"), ("immediate", "product"), ("product", "product")]
)
@pytest.mark.parametrize("k", [0, 1, 7])
@pytest.mark.parametrize("batch", [1, 3, 4])
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
def test_trusted_switch_ringless_product(hk, hv, batch, first, second, k):
    from benchmark.kernel_tuning.gdn_replay_common import Product

    runner, xs = _trusted_runner(hk, hv, batch, Config(immediate=True), "trusted")
    product = Product(
        hk, hv, runner.weights, runner.state.checkpoint[:batch], runner.state.conv_history[:batch]
    )
    reference = Product(
        hk, hv, runner.weights, runner.state.checkpoint[:batch], runner.state.conv_history[:batch]
    )
    # #25 and CuTe borrow the same ringless pool. The engine maintains replay
    # positions around #25; no ring seeds or materialization are needed here.
    product.state = runner.state.checkpoint[:batch]
    product.history = runner.state.conv_history[:batch]
    for t, mode in enumerate([first] * k + [second] * 24):
        x = xs[t % len(xs)]
        if mode == "product":
            actual = product.step(x)
            runner.state.metadata[:batch, :2].add_(1)
        else:
            _trusted_call(runner, x)
            actual = runner.out
        _assert_bitwise(actual, reference.step(x), phase="switch/ringless-output", step=t)
        _assert_bitwise(product.state, reference.state, phase="switch/ringless-state", step=t)
        _assert_bitwise(product.history, reference.history, phase="switch/ringless-history", step=t)
        assert not runner.state.metadata[:batch, 3].any()
        assert torch.equal(runner.state.metadata[:batch, 0], runner.state.metadata[:batch, 1])
