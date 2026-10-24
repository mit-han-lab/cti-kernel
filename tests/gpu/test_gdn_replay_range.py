# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""RP2 bounded range qualification: short cases plus one int32 token-offset boundary."""

import contextlib
from dataclasses import fields, replace

import pytest
import torch

from ctikernels._runtime.workspace import GraphWorkspace
from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import _phase_launch

from ..modules._replay_case import case, clone, projected, same
from ._exhaustive import ci_cases
from .test_gdn_replay_delayed_commit import Pair as DelayedPair
from .test_gdn_replay_trusted_k2 import _bytes, _copy, _workspace
from .test_gdn_replay_trusted_k2 import pytestmark as pytestmark
from .test_gdn_replay_unified import allocate
from .test_gdn_triton import ORDINARY_ULP, _max_ulp

# CuTe vs portable recipes differ by at most one BF16 ulp; the metric divides by a
# non-power-of-two spacing, so an exact one-ulp error can read 1 + 2**-52.
ULP_SLACK = 1e-9


class RangeRound:
    def __init__(self, hk, hv, batch, q, window):
        _, state, pending, self.weights, self.slots, self.pos, self.status = case(
            device="cuda", hk=hk, hv=hv, batch=batch, q=q, window=window
        )
        self.ops = [
            GatedDeltaNetReplay(
                num_k_heads=hk,
                num_v_heads=hv,
                device="cuda",
                provider=provider,
                mode="delayed" if provider == "cute" else "auto",
                checks=checks,
                verify_length=q,
                window=window,
            )
            for provider, checks in (("triton", "strict"), ("cute", "strict"), ("cute", "trusted"))
        ]
        gen = torch.Generator().manual_seed(812)
        state.checkpoint[:batch].copy_(torch.randn(batch, hv, 128, 128, generator=gen).mul_(0.05))
        state.conv_history[:batch].copy_(
            torch.randn(batch, 3, self.ops[0].geometry.conv_dim, generator=gen).to(torch.bfloat16)
        )
        state.ring[:batch, window:, : self.ops[0].geometry.conv_dim].copy_(
            state.conv_history[:batch]
        )
        # An unaligned committed prefix makes the QR boundary straddle a staging chunk.
        out = torch.empty(batch, hv * 128, dtype=torch.bfloat16, device="cuda")
        for t in range(min(3, window - 1)):
            self.ops[0].step(
                projected(self.ops[0], batch, seed=91 + t),
                *self.weights,
                slot_indices=self.slots,
                positions=self.pos,
                state=state,
                out=out,
                status=self.status,
            )
            self.pos.add_(1)
        self.states = [state, clone(state), clone(state)]
        self.pending = [pending, clone(pending), clone(pending)]
        self.outputs = [
            torch.full((batch, q, hv * 128), 7, dtype=torch.bfloat16, device="cuda")
            for _ in self.ops
        ]
        self.workspaces = [_workspace(batch) for _ in self.ops]
        self.q, self.batch = q, batch
        self.x = projected(self.ops[0], batch, length=q)
        self.positions = self.pos[:, None] + torch.arange(q, device="cuda")
        self.accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
        self.rows = torch.arange(batch, device="cuda", dtype=torch.int32) + state.metadata.shape[0]
        self.offsets = torch.zeros(batch, device="cuda", dtype=torch.int64)
        self.valid = [torch.zeros(batch, device="cuda", dtype=torch.bool) for _ in self.ops]

    def call(self, phase, capture=False):
        for i, op in enumerate(self.ops):
            kw = {
                "state": self.states[i],
                "pending": self.pending[i],
                "slot_indices": self.slots,
                "status": self.status if i < 2 else None,
            }
            if phase == "verify":
                op.verify(
                    self.x, *self.weights, positions=self.positions, out=self.outputs[i], **kw
                )
            elif phase == "reject":
                op.reject(**kw)
            else:
                if i == 2:
                    kw["workspace"] = self.workspaces[i]
                if capture:
                    kw.update(
                        snapshot_rows=self.rows,
                        snapshot_offsets=self.offsets,
                        snapshot_valid=self.valid[i],
                    )
                op.commit(*self.weights, accepted_lengths=self.accepted, **kw)

    def compare(self):
        same(self.states[1], self.states[2])
        same(self.pending[1], self.pending[2])
        _bytes(self.outputs[1], self.outputs[2])
        for f in fields(self.states[0]):
            portable, cute = getattr(self.states[0], f.name), getattr(self.states[1], f.name)
            if f.name == "checkpoint":
                assert _max_ulp(cute, portable) <= ORDINARY_ULP + ULP_SLACK
            else:
                _bytes(cute, portable)
        same(self.pending[0], self.pending[1])
        assert (
            _max_ulp(self.outputs[1].reshape(-1, 128), self.outputs[0].reshape(-1, 128))
            <= ORDINARY_ULP + ULP_SLACK
        )
        _bytes(self.valid[0], self.valid[1])
        _bytes(self.valid[1], self.valid[2])
        assert self.status.code.item() == self.status.poison.item() == 0


@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("q", [1, 33, 64])
def test_range_rounds(hk, hv, batch, q):
    p = RangeRound(hk, hv, batch, q, 2 if q == 1 else 32)
    if batch > 1:
        p.slots[-1] = -1
    for cycle in range(3):
        p.x.copy_(projected(p.ops[0], batch, length=q, seed=300 + cycle))
        p.accepted.copy_(
            torch.tensor(
                [(0, q // 2, q)[(cycle + b) % 3] for b in range(batch)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        p.offsets.copy_(p.accepted.to(torch.int64) // 2)
        before = [clone(s) for s in p.states]
        p.call("verify")
        for a, b in zip(p.states, before, strict=True):
            same(a, b)
        p.compare()
        p.call("commit", capture=True)
        p.compare()
        p.positions.add_(p.accepted[:, None])
        p.call("verify")
        before = [clone(s) for s in p.states]
        p.call("reject")
        for a, b in zip(p.states, before, strict=True):
            same(a, b)
        p.compare()
    # Entirely inactive clusters must still reach every large-Q exchange and zero all tokens.
    p.slots.fill_(-1)
    p.call("verify")
    p.compare()
    assert not any(out.any().item() for out in p.outputs)


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("tile", [32, 64, 128])
def test_range_output_prefix(hv, tile, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", tile)
    monkeypatch.setattr(_phase_launch, "VECTOR_PUBLISH", not (tile == 128 and hv == 48))
    p = RangeRound(16, hv, 3, 64, 32)
    p.slots[-1] = -1
    p.call("verify")
    p.compare()
    prefix = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=hv,
        device="cuda",
        provider="cute",
        checks="trusted",
        mode="delayed",
        verify_length=32,
        window=32,
    )
    ws = allocate(prefix.workspace_spec("verify", batch_size=3))
    pending = prefix.create_pending(raw=ws["pending.raw"], metadata=ws["pending.metadata"])
    prefix.verify(
        p.x[:, :32],
        *p.weights,
        positions=p.positions[:, :32].contiguous(),
        slot_indices=p.slots,
        state=p.states[2],
        pending=pending,
        out=ws["out"],
    )
    # Verify is read-only and causal; splitting into committed Q32 rounds is NOT the identity.
    _bytes(p.outputs[2][:, :32], ws["out"])


@pytest.mark.parametrize("q", [4, 8, 64])
def test_range_delayed_commit(q):
    p = DelayedPair(q=q, batch=3, window=32, shape=(16, 32), padding=True)
    # Reach n=L-1 by lazy appends, then fold exactly at L (and above L for Q64).
    left = 31
    while left:
        m = min(q, left)
        p.accepted.fill_(m)
        p.verify()
        p.commit()
        left -= m
    p.accepted.copy_(torch.tensor([1, q, 0], device="cuda", dtype=torch.int32))
    p.verify()
    p.commit()
    p.flush()
    # Flush preserves the rotated cursor. Subsequent appends wrap physical L+3 rows.
    for m in (0, 1) + (min(q, 8),) * (35 // min(q, 8) + 1):
        p.accepted.fill_(min(q, m))
        p.verify()
        p.commit()
        p.flush()


@pytest.mark.parametrize("hv", [32, 48])
def test_range_graph_round(hv):
    p = RangeRound(16, hv, 3, 64, 32)
    initial, pending = [clone(s) for s in p.states], [clone(t) for t in p.pending]
    p.call("verify")
    p.call("commit")
    graphs = []
    for i, op in enumerate(p.ops):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op.verify(
                p.x,
                *p.weights,
                positions=p.positions,
                slot_indices=p.slots,
                state=p.states[i],
                pending=p.pending[i],
                out=p.outputs[i],
                status=p.status if i < 2 else None,
            )
            op.commit(
                *p.weights,
                slot_indices=p.slots,
                accepted_lengths=p.accepted,
                state=p.states[i],
                pending=p.pending[i],
                status=p.status if i < 2 else None,
                **({"workspace": p.workspaces[i]} if i == 2 else {}),
            )
        graphs.append(graph)
    for i in range(3):
        _copy(p.states[i], initial[i])
        _copy(p.pending[i], pending[i])
    for r, lengths in enumerate(((0, 1, 64), (33, 0, 8), (64, 32, 1), (1, 64, 0))):
        p.slots.copy_(torch.roll(torch.arange(3, device="cuda", dtype=torch.int32), r))
        p.positions.copy_(
            p.states[0].metadata[p.slots.long(), 1, None] + torch.arange(64, device="cuda")
        )
        p.accepted.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        if r == 2:
            p.slots[-1] = -1
        p.x.copy_(projected(p.ops[0], 3, length=64, seed=600 + r))
        for graph in graphs:
            graph.replay()
        p.compare()


@pytest.mark.parametrize(
    "hk,hv,window,mode,checks",
    ci_cases(
        [(16, 32), (16, 48)],
        [2, 32, 64],
        [("auto", "trusted"), ("delayed", "trusted"), ("delayed", "strict")],
        core=[
            (hk, hv, w, mode, checks)
            for hk, hv in ((16, 32), (16, 48))
            for w in (2, 32, 64)
            for mode, checks in (("auto", "trusted"), ("delayed", "trusted"))
        ]
        + [(16, 32, 64, "delayed", "strict")],
    ),
)
def test_range_step_graph(hk, hv, window, mode, checks):
    _, state, _, weights, slots, pos, status = case(
        device="cuda", hk=hk, hv=hv, batch=1, window=window
    )
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        provider="cute",
        mode=mode,
        checks=checks,
        window=window,
    )
    eager = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        provider="cute",
        mode="immediate",
        window=window,
    )
    state.conv_history[0].copy_(projected(op, 3)[:, : op.geometry.conv_dim])
    state.checkpoint[0].fill_(0.001)
    # Deliberately do not seed ring: n=0 must read authoritative convolution history.
    expected = replace(clone(state), ring=state.ring[:, :0].clone())
    if op._ringless:
        state = replace(state, ring=state.ring[:, :0].clone())
    ws, ew = (allocate(o.workspace_spec("step", batch_size=1)) for o in (op, eager))
    x = projected(op, 1)
    initial = clone(state)

    def step():
        op.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=ws["out"],
            workspace=ws,
            status=status if checks == "strict" else None,
        )

    step()
    _copy(state, initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    _copy(state, initial)
    for t in range(window + 2):
        x.copy_(projected(op, 1, seed=900 + t))
        graph.replay()
        eager.step(
            x,
            *weights,
            slot_indices=slots,
            positions=pos,
            state=expected,
            out=ew["out"],
            workspace=ew,
        )
        _bytes(ws["out"], ew["out"])
        _bytes(
            state.conv_history[: state.metadata.shape[0]],
            expected.conv_history[: state.metadata.shape[0]],
        )
        if mode == "delayed" or op.decode_route(1) == "delayed":
            assert state.metadata[0, 3].item() == (t + 1) % window
            assert state.metadata[0, 2].item() == (t + 1) % (window + 3)
        if t + 1 in (window - 1, window, window + 1):
            rows = state.metadata.shape[0]
            valid = torch.zeros(1, dtype=torch.bool, device="cuda")
            op.export_snapshot(
                *weights,
                pairs=torch.tensor([[0, rows]], dtype=torch.int32, device="cuda"),
                state=state,
                snapshot_valid=valid,
                status=status if checks == "strict" else None,
                workspace=allocate(op.workspace_spec("export_snapshot", batch_size=1)),
            )
            assert valid.item()
            _bytes(state.checkpoint[rows], expected.checkpoint[0])
            _bytes(state.conv_history[rows], expected.conv_history[0])
        pos.add_(1)
    assert status.code.item() == status.poison.item() == 0


@pytest.mark.parametrize("provider", ["triton", "cute"])
def test_range_token_offsets_cross_int32(provider):
    # (16,48) projected rows are 16480 wide: token 130309 is the first whose x offset
    # needs more than int32, so Q=131072 crosses it in verify staging and commit.
    hk, hv, q = 16, 48, 131072
    if torch.cuda.mem_get_info()[0] < 14 * 2**30:
        pytest.skip("needs about 9 GiB free")
    op = GatedDeltaNetReplay(
        num_k_heads=hk,
        num_v_heads=hv,
        device="cuda",
        provider=provider,
        checks="strict",
        verify_length=q,
        window=8,
    )
    g = op.geometry
    assert (q - 1) * g.proj_width >= 2**31 > (130308 * g.proj_width)
    _, state, pending, weights, slots, pos, status = case(
        device="cuda", hk=hk, hv=hv, batch=1, q=q, window=8
    )
    x = torch.randn(
        1,
        q,
        g.proj_width,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator("cuda").manual_seed(5),
    )
    out = torch.empty(1, q, g.value_width, dtype=torch.bfloat16, device="cuda")
    op.verify(
        x,
        *weights,
        slot_indices=slots,
        positions=pos[:, None] + torch.arange(q, device="cuda"),
        state=state,
        pending=pending,
        out=out,
        status=status,
    )
    dc = g.conv_dim
    gates = x[0, :, dc + g.value_width :]
    assert torch.equal(pending.raw[0, :, :dc], x[0, :, :dc])
    assert torch.equal(pending.raw[0, :, dc:], gates)
    del out
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([q], dtype=torch.int32, device="cuda"),
        state=state,
        pending=pending,
        status=status,
    )
    assert torch.equal(state.conv_history[0], x[0, -3:, :dc])
    assert torch.isfinite(state.checkpoint[0]).all()
    assert status.code.item() == status.poison.item() == 0


def test_lifecycle_expected_one_variant(monkeypatch):
    from ctikernels.gdn.replay.triton._status import lifecycle

    from ..modules._replay_case import graph_buffers

    op = GatedDeltaNetReplay(num_k_heads=16, num_v_heads=32, device="cuda")
    status = graph_buffers(op, batch=3).status(0)
    device = torch.cuda.current_device()
    # Isolate this test from earlier begin/settle calls without discarding their cache.
    # Triton 3.7.1 stores (kernel cache, key cache, target, backend, binder) per device.
    caches = lifecycle.device_caches[device]
    monkeypatch.setitem(lifecycle.device_caches, device, ({}, {}, *caches[2:]))
    cache = lifecycle.device_caches[device][0]
    before = len(cache)
    for expected in (1, 16, 17, 2**31):
        op._lifecycle("clear", status, expected)
        assert len(cache) - before == 1
    assert status.poison.item() == 64

    # A completed matching invocation still clears records and validity at int64 range.
    status.poison.zero_()
    status.sequence.fill_(2**31)
    op.begin(status=status)
    op.end(status=status)
    status.validity.fill_(True)
    before = len(cache)
    op._lifecycle("clear", status, 2**31)
    assert len(cache) == before
    assert not status.records.count_nonzero().item()
    assert not status.validity.any().item()
    assert status.poison.item() == 0


def test_strided_controls_cross_int32():
    from ..modules.test_gdn_replay_range import SupportPair

    stride, batch, q = 2**30, 3, 8
    assert (torch.tensor(2, dtype=torch.int32) * stride).item() == -(2**31)
    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        pytest.skip("needs at least 12 GiB free for an 8 GiB control buffer")
    pair = SupportPair("controls", device="cuda", provider="cute", hk=16, hv=32, batch=batch, q=q)
    for side in range(2):
        pair.call("initialize", side)
        pair.call("handoff", side)
    positions = (pair.pos[:, None] + torch.arange(q, device="cuda")).int()
    # Disjoint views share one large allocation. Only the few used elements are written.
    backing = torch.empty(2 * stride + q + 1, dtype=torch.int32, device="cuda")
    slots = backing.as_strided((batch,), (stride,))
    strided_positions = backing.as_strided((batch, q), (stride, 1), 1)
    slots.copy_(pair.slots)
    strided_positions.copy_(positions)
    assert pair.op.checks == "strict"  # Exercise the per-batch validation loop too.
    for side, indices, pos in ((0, pair.slots, positions), (1, slots, strided_positions)):
        pair.op.verify(
            pair.vx,
            *pair.weights[side],
            slot_indices=indices,
            positions=pos,
            state=pair.states[side],
            pending=pair.pending[side],
            out=pair.voutputs[side],
            status=pair.status,
        )
    pair.compare()
    for side, indices in enumerate((pair.slots, slots)):
        pair.op.commit(
            *pair.weights[side],
            slot_indices=indices,
            accepted_lengths=pair.accepted,
            state=pair.states[side],
            pending=pair.pending[side],
            workspace=pair.workspaces["commit"][side],
            status=pair.status,
        )
    pair.compare()
    assert torch.equal(pair.states[0].metadata[:batch, 1], pair.pos + pair.accepted)


def test_snapshot_controls_cross_int32():
    from ..modules.test_gdn_replay_range import SupportPair

    stride, batch, q = 2**30, 3, 8
    assert (torch.tensor(2, dtype=torch.int32) * stride).item() == -(2**31)
    elements = (batch - 1) * stride + 2
    # Backing allocation, 64 MiB for the pair, and 4 GiB headroom.
    if torch.cuda.mem_get_info()[0] < elements * 4 + 64 * 2**20 + 4 * 2**30:
        pytest.skip("needs about 12.1 GiB free for an 8 GiB control buffer")
    pair = SupportPair("controls", device="cuda", provider="cute", hk=16, hv=32, batch=batch, q=q)
    for side in range(2):
        pair.call("initialize", side)
        pair.call("handoff", side)
        pair.call("verify", side)
    pair.compare()
    offsets = pair.offsets.int()
    backing = torch.empty(elements, dtype=torch.int32, device="cuda")
    rows = backing.as_strided((batch,), (stride,))
    strided_offsets = backing.as_strided((batch,), (stride,), 1)
    rows.copy_(pair.rows)
    strided_offsets.copy_(offsets)
    assert pair.op.checks == "strict"
    # The capture arguments enable snapshot validation independently of snapshot_controls.
    for side, snapshot_rows, snapshot_offsets in (
        (0, pair.rows, offsets),
        (1, rows, strided_offsets),
    ):
        pair.op.commit(
            *pair.weights[side],
            slot_indices=pair.slots,
            accepted_lengths=pair.accepted,
            state=pair.states[side],
            pending=pair.pending[side],
            snapshot_rows=snapshot_rows,
            snapshot_offsets=snapshot_offsets,
            snapshot_valid=pair.valid[side],
            workspace=pair.workspaces["commit"][side],
            status=pair.status,
        )
        assert pair.valid[side].all().item()
        assert pair.status.code.item() == pair.status.poison.item() == 0
    pair.compare()
    assert torch.equal(pair.states[0].metadata[:batch, 1], pair.pos + pair.accepted)


def test_strict_delayed_history_projected_stride_cross_int32():
    from ctikernels.gdn.replay.cute import effective_config

    from ..modules.test_gdn_replay_range import SupportPair

    stride, batch = 2**30, 3
    assert (torch.tensor(2, dtype=torch.int32) * stride).item() == -(2**31)
    # The projected rows fit in the 64 MiB pair allowance beyond the 4 GiB backing span.
    if torch.cuda.mem_get_info()[0] < (batch - 1) * stride * 2 + 64 * 2**20 + 4 * 2**30:
        pytest.skip("needs about 8.1 GiB free for a 4 GiB projected buffer")
    pair = SupportPair(
        "controls",
        device="cuda",
        provider="cute",
        hk=16,
        hv=32,
        batch=batch,
        mode="delayed",
        checks="strict",
    )
    config = effective_config(pair.op, batch)
    assert not config.immediate and not config.trusted and not config.snapshot_controls
    for side in range(2):
        pair.call("initialize", side)
        pair.call("handoff", side)
        assert not pair.states[side].metadata[:batch, 3].any().item()
    pair.compare()
    width = pair.op.geometry.proj_width
    backing = torch.empty((batch - 1) * stride + width, dtype=torch.bfloat16, device="cuda")
    x = backing.as_strided((batch, width), (stride, 1))
    x.copy_(pair.x)
    # With n=0, strict delayed publication calls roll_row_history with an int32 row.
    # The ring is allocated but empty; no ringless configuration is needed.
    for side, projected_x in enumerate((pair.x, x)):
        pair.op.step(
            projected_x,
            *pair.weights[side],
            slot_indices=pair.slots,
            positions=pair.pos,
            state=pair.states[side],
            out=pair.outputs[side],
            workspace=pair.workspaces["step"][side],
            status=pair.status,
        )
        _bytes(pair.states[side].conv_history[:batch, -1], pair.x[:, : pair.op.geometry.conv_dim])
        assert pair.status.code.item() == pair.status.poison.item() == 0
    pair.compare()
    assert torch.equal(pair.states[0].metadata[:batch, 1], pair.pos + 1)
    assert (pair.states[0].metadata[:batch, 3] == 1).all().item()


_SUPPORT_VARIANTS = ("bf16", "fp16", "mixed", "controls", "grad_storage")
_SUPPORT_CORE = [
    (
        provider,
        16 if n % 2 else 8,
        32 if n % 2 else 16,
        3 if n % 2 else 1,
        8 if n % 2 else 4,
        mode,
        checks,
        variant,
    )
    for provider, mode, checks in (
        ("triton", "delayed", "strict"),
        ("cute", "delayed", "strict"),
        ("cute", "immediate", "trusted"),
    )
    for n, variant in enumerate(_SUPPORT_VARIANTS)
]


@pytest.mark.parametrize(
    "provider,hk,hv,batch,q,mode,checks,variant",
    ci_cases(
        ("triton", "cute"),
        ((16, 32), (8, 16)),
        (1, 3),
        (4, 8),
        (("delayed", "strict"), ("delayed", "trusted"), ("immediate", "trusted")),
        _SUPPORT_VARIANTS,
        core=_SUPPORT_CORE,
    ),
)
def test_support_range_bitwise(provider, hk, hv, batch, q, mode, checks, variant):
    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair(
        variant,
        device="cuda",
        provider=provider,
        hk=hk,
        hv=hv,
        batch=batch,
        q=q,
        mode=mode,
        checks=checks,
    )
    with torch.no_grad():
        pair.round()


@pytest.mark.parametrize(
    ("provider", "checks", "workspace"),
    [
        ("triton", "strict", False),
        ("cute", "strict", False),
        ("triton", "trusted", False),
        ("triton", "strict", True),
        ("triton", "trusted", True),
    ],
)
def test_support_range_graph(provider, checks, workspace):
    """Graph replay == eager; a trusted op's verify and commit run without status; the split
    verify's scratch comes from a GraphWorkspace when one is active (else call-owned)."""
    group = GraphWorkspace(max_rows=3 * 4) if workspace else contextlib.nullcontext()
    with group:
        _support_range_graph(provider, checks)
    if workspace:
        assert group.nbytes > 0  # the verify's pre-norm slot


def _support_range_graph(provider, checks):
    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair(
        "bf16",
        device="cuda",
        provider=provider,
        hk=8,
        hv=16,
        batch=3,
        q=4,
        mode="delayed",
        checks=checks,
    )
    status = None if checks == "trusted" else pair.status
    for i in range(2):
        pair.call("initialize", i)
        pair.call("handoff", i)
    op = pair.op
    slots = pair.slots.long()
    positions = pair.pos[:, None] + torch.arange(pair.q, device="cuda")
    accepted, rows, offsets = pair.accepted.long(), pair.rows.long(), pair.offsets.long()

    def round(i):
        op.verify(
            pair.vx,
            *pair.weights[i],
            slot_indices=slots if i else pair.slots,
            positions=positions,
            state=pair.states[i],
            pending=pair.pending[i],
            out=pair.voutputs[i],
            status=status,
        )
        op.commit(
            *pair.weights[i],
            slot_indices=slots if i else pair.slots,
            accepted_lengths=accepted if i else pair.accepted,
            snapshot_rows=rows if i else pair.rows,
            snapshot_offsets=offsets,
            snapshot_valid=pair.valid[i],
            state=pair.states[i],
            pending=pair.pending[i],
            status=status,
            workspace=pair.workspaces["commit"][i],
        )

    initial, pending = clone(pair.states[1]), clone(pair.pending[1])
    round(1)
    _copy(pair.states[1], initial)
    _copy(pair.pending[1], pending)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        round(1)
    _copy(pair.states[1], initial)
    _copy(pair.pending[1], pending)
    for iteration in range(3):
        pair.vx.copy_(projected(op, 3, length=4, seed=410 + iteration))
        positions.copy_(pair.states[0].metadata[:3, 1, None] + torch.arange(4, device="cuda"))
        if iteration == 1:
            slots[-1] = pair.slots[-1] = -1
        round(0)
        graph.replay()
        pair.compare()


@pytest.mark.parametrize("variant", ["bf16", "fp16", "mixed", "controls", "grad_storage"])
def test_support_range_product_dispatch(variant):
    from ctikernels.gdn.replay.cute import Config

    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair(
        variant,
        device="cuda",
        provider="cute",
        hk=16,
        hv=32,
        batch=3,
        q=8,
        mode="immediate",
        checks="trusted",
    )
    pair.op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda",
        provider="cute",
        cute_config=Config(provider="product", immediate=True),
        checks="trusted",
        commit_mode="eager",
        window=4,
        verify_length=8,
    )
    assert pair.op.decode_route(3) == "product"
    # Exercise the old workspace-free #25 call form, including all-int64 indices.
    pair.workspaces["step"] = [{}, {}]
    if variant == "mixed":
        pair.integer_variant = True
    with torch.inference_mode():
        pair.round()
        assert pair.op._product_decode is not None
        assert pair.op._product_decode.provider == "triton"
        assert not any("step" in key[0] for key in pair.op._cute_launchers)
        # A fresh pool under capture exercises the warmed slot-dtype zeroing kernel.
        initial = clone(pair.states[1])
        pair.states[1] = clone(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pair.call("step", 1)
        _copy(pair.states[1], initial)
        for _ in range(3):
            pair.call("step", 0)
            graph.replay()
            pair.compare()
            pair.pos.add_(1)
        assert not any("step" in key[0] for key in pair.op._cute_launchers)


def test_decode_direct_bf16_weights_bitwise():
    from ctikernels.gdn import GatedDeltaNetDecode

    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair("bf16", device="cuda", provider="cute", hk=16, hv=32, batch=1)
    op = GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda")
    assert op.provider == "triton"
    with torch.inference_mode():
        for _ in range(3):
            for side in range(2):
                op(
                    pair.x,
                    *pair.weights[side],
                    pair.slots,
                    pair.states[side].checkpoint,
                    pair.states[side].conv_history,
                    out=pair.outputs[side],
                )
            pair.compare()


def test_support_range_handoff_layers():
    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair(
        "controls",
        device="cuda",
        provider="cute",
        hk=8,
        hv=16,
        batch=3,
        q=4,
        mode="delayed",
        checks="strict",
    )
    op = pair.op
    canonical = torch.full_like(pair.states[0].checkpoint[:3], 0.002)
    history = pair.vx[:, :3, : op.geometry.conv_dim].contiguous()
    for i in range(2):
        layers = op.bind_layers(
            states=(pair.states[i],),
            table=torch.empty(1, 8, dtype=torch.int64, device="cuda"),
            canonical_states=(canonical,),
            histories=(history,),
        )
        op.handoff_layers(
            layers,
            next_positions=pair.control(pair.pos, i),
            slot_indices=pair.control(pair.slots, i),
            status=pair.status,
        )
    pair.compare()
    flags = [torch.empty(1, dtype=torch.bool, device="cuda") for _ in range(2)]
    for i in range(2):
        op.ring_empty(pair.states[i], pair.control(pair.slots, i), flags[i])
    assert flags[0].item() and flags[1].item()


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("hk,hv", [(16, 32), (8, 16)])
@pytest.mark.parametrize("batch,q", [(1, 4), (3, 8)])
def test_layout_round_graph(provider, hk, hv, batch, q):
    from ..modules.test_gdn_replay_range import LayoutPair

    pair = LayoutPair(device="cuda", provider=provider, hk=hk, hv=hv, batch=batch, q=q)
    pair.round()

    # Resume from imported snapshots after rejecting the last speculative block.
    def transaction(side):
        pair.call("verify", side)
        pair.call("commit", side)

    initial, pending = clone(pair.states[1]), clone(pair.pending[1])
    transaction(1)
    _copy(pair.states[1], initial)
    _copy(pair.pending[1], pending)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        transaction(1)
    _copy(pair.states[1], initial)
    _copy(pair.pending[1], pending)
    for iteration in range(3):
        value = projected(pair.op, batch, length=q, seed=870 + iteration)
        for x in pair.vxs:
            x.copy_(value)
        pair.update(
            "positions", pair.states[0].metadata[:batch, 1, None] + torch.arange(q, device="cuda")
        )
        pair.update("accepted", torch.arange(batch, device="cuda", dtype=torch.int32) % q + 1)
        if batch > 1 and iteration == 1:
            slots = torch.arange(batch, device="cuda", dtype=torch.int32)
            slots[-1] = -1
            pair.update("slots", slots)
        transaction(0)
        graph.replay()
        pair.compare()


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("mode", ["immediate", "delayed"])
@pytest.mark.parametrize("hk,hv,batch,q", [(16, 32, 1, 4), (8, 16, 3, 8)])
def test_layout_step(provider, mode, hk, hv, batch, q):
    from ..modules.test_gdn_replay_range import LayoutPair

    pair = LayoutPair(
        device="cuda",
        provider=provider,
        hk=hk,
        hv=hv,
        batch=batch,
        q=q,
        mode=mode,
        checks="trusted" if provider == "cute" else "strict",
    )
    if provider == "cute":
        from ctikernels.gdn.replay.cute import Config

        pair.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device="cuda",
            provider="cute",
            mode=mode,
            checks="trusted",
            cute_config=Config(immediate=mode == "immediate"),
            commit_mode="eager",
            window=4,
            verify_length=q,
        )
        pair.workspaces["step"] = [
            allocate(pair.op.workspace_spec("step", batch_size=batch)) for _ in range(2)
        ]
    pair.both("initialize")
    pair.both("handoff")

    # Portable's immediate recurrence is its canonical bridge. CuTe uses Step.
    def step(side):
        if provider == "triton" and mode == "immediate":
            pair.op.bridge_step(
                pair.xs[side],
                *pair.weights[side],
                slot_indices=pair.controls["slots"][side],
                positions=pair.controls["pos"][side],
                state=pair.states[side],
                out=pair.outputs[side],
                status=pair.status,
            )
        else:
            pair.call("step", side)

    for iteration in range(6):
        value = projected(pair.op, batch, seed=920 + iteration)
        for x in pair.xs:
            x.copy_(value)
        step(0)
        step(1)
        pair.compare()
        pair.update("pos", pair.controls["pos"][0] + 1)
    if provider == "cute":
        flags = [torch.empty(1, dtype=torch.bool, device="cuda") for _ in range(2)]
        for i in range(2):
            pair.op.ring_empty(pair.states[i], pair.controls["slots"][i], flags[i])
        assert torch.equal(*flags)


@pytest.mark.parametrize("provider", ["triton", "cute"])
@pytest.mark.parametrize("hk,hv", [(16, 32), (8, 16)])
def test_layout_handoff_layers(provider, hk, hv):
    from ..modules._replay_case import graph_buffers
    from ..modules.test_gdn_replay_range import LayoutPair

    pair = LayoutPair(device="cuda", provider=provider, hk=hk, hv=hv, batch=3, q=4)
    op = pair.op
    # Both logical layers share an envelope; layer 2 must remain sentinel-filled.
    template = pair.states[0]
    cp = torch.full(
        (template.checkpoint.shape[0], 3, hv, 128, 128), -17.0, device="cuda", dtype=torch.float32
    )
    history = torch.full(
        (template.conv_history.shape[0], 3, 3, op.geometry.conv_dim),
        -17,
        device="cuda",
        dtype=torch.bfloat16,
    )
    pitched = [
        replace(clone(template), checkpoint=cp[:, j], conv_history=history[:, j]) for j in range(2)
    ]
    dense = [clone(template) for _ in range(2)]
    for a, b in zip(dense, pitched, strict=True):
        b.checkpoint.copy_(a.checkpoint)
        b.conv_history.copy_(a.conv_history)
    sources = [[x.clone() + j for j in range(2)] for x in (pair.canonical[0], pair.histories[0])]
    # Independent source pitches differ from destination pitches and from each other.
    pitched_sources = [[pair.pool(x, j + 2) for j, x in enumerate(values)] for values in sources]
    bindings, statuses = [], []
    for side, states in enumerate((dense, pitched)):
        src = sources if side == 0 else pitched_sources
        bindings.append(
            op.bind_layers(
                states=states,
                table=torch.empty(2, 8, dtype=torch.int64, device="cuda"),
                canonical_states=src[0],
                histories=src[1],
            )
        )
        status = graph_buffers(op, batch=3, layers=2).status(0)
        status.sequence.fill_(1)
        op.begin(status=status)
        statuses.append(status)

    def handoff(side):
        op.handoff_layers(
            bindings[side],
            next_positions=pair.controls["pos"][side],
            slot_indices=pair.controls["slots"][side],
            status=statuses[side],
        )

    handoff(1)  # Warm the pitched table, dtypes and strides before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        handoff(1)
    for old, new in zip(dense, pitched, strict=True):
        _copy(new, old)
    handoff(0)
    graph.replay()
    for a, b in zip(dense, pitched, strict=True):
        for f in fields(a):
            _bytes(getattr(a, f.name), getattr(b, f.name))
    assert torch.all(cp[:, 2] == -17) and torch.all(history[:, 2] == -17)
    for status in statuses:
        assert status.code.item() == status.poison.item() == 0
    pair.compare()


@pytest.mark.parametrize("batch", [1, 3])
def test_layout_product_dispatch(batch):
    from ctikernels.gdn.replay.cute import Config

    from ..modules.test_gdn_replay_range import LayoutPair

    pair = LayoutPair(
        device="cuda",
        provider="cute",
        hk=16,
        hv=32,
        batch=batch,
        q=8,
        mode="immediate",
        checks="trusted",
    )
    pair.op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda",
        provider="cute",
        cute_config=Config(provider="product", immediate=True),
        checks="trusted",
        commit_mode="eager",
        window=4,
        verify_length=8,
    )
    pair.workspaces["step"] = [{}, {}]
    pair.both("initialize")
    pair.both("handoff")
    pair.both("step")
    assert pair.op._product_decode is not None  # Dense side ran #25.
    assert any(key[0] == "bridge_step" for key in pair.op._cute_launchers)
    pair.update("pos", pair.controls["pos"][0] + 1)
    initial = clone(pair.states[1])
    pair.call("step", 1)
    _copy(pair.states[1], initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pair.call("step", 1)
    _copy(pair.states[1], initial)
    for _ in range(3):
        pair.call("step", 0)
        graph.replay()
        pair.compare()
        pair.update("pos", pair.controls["pos"][0] + 1)


@pytest.mark.parametrize(("hk", "hv"), [(16, 32), (8, 16)])
@pytest.mark.parametrize("batch", [1, 3])
def test_trusted_statusless_triton_equals_strict(hk, hv, batch):
    """Every phase of a trusted Triton op without status is the strict op's, bit for bit."""
    from ..modules.test_gdn_replay_range import SupportPair

    pair = SupportPair(
        "bf16", device="cuda", provider="triton", hk=hk, hv=hv, batch=batch, q=4,
        trusted_side=True,
    )  # fmt: skip
    with torch.no_grad():
        pair.round()
