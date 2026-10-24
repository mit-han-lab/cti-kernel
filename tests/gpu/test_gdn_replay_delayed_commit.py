# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""R3c bitwise qualification. GDN_DELAYED_EXHAUSTIVE=1 selects the full matrix."""

import os
from itertools import product

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay.cute import _phase_launch

from ..modules._replay_case import case, clone, projected, same
from .test_gdn_replay_trusted_k2 import _bytes, _copy, _workspace
from .test_gdn_replay_trusted_k2 import pytestmark as pytestmark

SHAPES = ((16, 32), (16, 48), (8, 16))
QUERIES = (4, 5, 6, 8, 9, 12, 16)
BATCHES = (1, 3, 4)
WINDOWS = (4, 8, 16)
if os.environ.get("GDN_DELAYED_EXHAUSTIVE") == "1":
    CASES = [
        pytest.param(q, b, w, shape, marks=pytest.mark.exhaustive)
        for q, b, w, shape in product(QUERIES, BATCHES, WINDOWS, SHAPES)
    ]
else:
    CASES = [
        pytest.param(
            q,
            b,
            WINDOWS[(i + j) % 3],
            SHAPES[(i + 2 * j) % 3],
            marks=() if (q, b) in {(6, 3), (6, 4), (9, 3)} else pytest.mark.exhaustive,
        )
        for i, q in enumerate(QUERIES)
        for j, b in enumerate(BATCHES)
    ]


class Pair:
    def __init__(self, q=8, batch=3, window=8, shape=(8, 16), padding=False):
        hk, hv = shape
        _, state, pending, self.weights, self.slots, pos, self.status = case(
            device="cuda",
            hk=hk,
            hv=hv,
            batch=batch,
            q=q,
            window=window,
        )
        self.ops = [
            GatedDeltaNetReplay(
                num_k_heads=hk,
                num_v_heads=hv,
                device="cuda",
                provider="cute",
                verify_length=q,
                window=window,
                commit_mode=mode,
            )
            for mode in ("eager", "delayed")
        ]
        state.checkpoint[:batch].copy_(
            torch.randn(
                batch,
                hv,
                128,
                128,
                generator=torch.Generator().manual_seed(812),
            )
            .mul_(0.05)
            .to("cuda")
        )
        # Nonzero checkpoint history seeds, including before the first append.
        history = (
            torch.randn(
                batch,
                3,
                self.ops[0].geometry.conv_dim,
                generator=torch.Generator().manual_seed(813),
            )
            .to(torch.bfloat16)
            .to("cuda")
        )
        state.conv_history[:batch].copy_(history)
        state.ring[:batch, window:, : history.shape[-1]].copy_(history)
        self.states, self.pending = [state, clone(state)], [pending, clone(pending)]
        self.q, self.batch, self.window = q, batch, window
        self.positions = pos[:, None] + torch.arange(q, device="cuda")
        self.outputs = [
            torch.empty(batch, q, hv * 128, dtype=torch.bfloat16, device="cuda") for _ in self.ops
        ]
        self.workspaces = [_workspace(batch) for _ in self.ops]
        self.accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
        self.drafts = projected(self.ops[0], batch, length=q)
        if padding and batch > 1:
            self.slots[-1] = -1

    def verify(self):
        for i, op in enumerate(self.ops):
            op.verify(
                self.drafts,
                *self.weights,
                slot_indices=self.slots,
                positions=self.positions,
                state=self.states[i],
                pending=self.pending[i],
                out=self.outputs[i],
            )
        _bytes(*self.outputs)
        _bytes(self.pending[0].raw, self.pending[1].raw)
        # The base checkpoint is intentionally older in delayed mode.
        for state, pending in zip(self.states, self.pending, strict=True):
            active = self.slots >= 0
            _bytes(pending.metadata[active, 2], state.metadata[self.slots[active].long(), 0])

    def commit(self, *, switch=False, captures=None):
        old = clone(self.states[1])
        for i in range(2):
            # Both op instances can bind the delayed-side state.
            op = self.ops[0] if i == 1 and switch else self.ops[i]
            op.commit(
                *self.weights,
                slot_indices=self.slots,
                accepted_lengths=self.accepted,
                state=self.states[i],
                pending=self.pending[i],
                workspace=self.workspaces[i],
                **(captures[i] if captures else {}),
            )
        for b, slot in enumerate(self.slots.cpu().tolist()):
            if slot < 0:
                continue
            m = min(max(int(self.accepted[b]), 0), self.q)
            capture = captures is not None and bool(captures[1]["snapshot_valid"][b])
            n, cursor = (int(old.metadata[slot, j]) for j in (3, 2))
            lazy = not switch and n + m < self.window and not capture
            if lazy:
                _bytes(old.checkpoint[slot], self.states[1].checkpoint[slot])
                assert int(self.states[1].metadata[slot, 3]) == n + m
                assert int(self.states[1].metadata[slot, 2]) == (cursor + m) % (self.window + 3)
                for t in range(m):
                    _bytes(
                        self.states[1].ring[slot, (cursor + t) % (self.window + 3)],
                        self.pending[1].raw[b, t],
                    )
                if m == 0:
                    _bytes(old.metadata[slot], self.states[1].metadata[slot])
                    _bytes(old.ring[slot], self.states[1].ring[slot])
                    _bytes(old.conv_history[slot], self.states[1].conv_history[slot])
            else:
                assert int(self.states[1].metadata[slot, 3]) == 0
                _bytes(self.states[0].checkpoint[slot], self.states[1].checkpoint[slot])
        _bytes(self.states[0].conv_history, self.states[1].conv_history)
        _bytes(
            self.pending[0].metadata[:, [0, 1, 3, 4, 5, 6, 7]],
            self.pending[1].metadata[:, [0, 1, 3, 4, 5, 6, 7]],
        )
        for ws in self.workspaces:
            assert torch.count_nonzero(ws["arrivals"]).item() == 0
        self.positions.add_(self.accepted.clamp(0, self.q)[:, None])

    def flush(self):
        self.ops[1].flush(
            *self.weights,
            slot_indices=self.slots,
            state=self.states[1],
            workspace=self.workspaces[1],
        )
        self.canonical()

    def canonical(self):
        for name in ("checkpoint", "conv_history", "snapshot_metadata", "pool_epoch"):
            _bytes(getattr(self.states[0], name), getattr(self.states[1], name))
        _bytes(
            self.states[0].metadata[:, [0, 1, 4, 5, 6, 7]],
            self.states[1].metadata[:, [0, 1, 4, 5, 6, 7]],
        )
        assert torch.count_nonzero(self.states[1].metadata[:, 3]).item() == 0


@pytest.mark.parametrize("q,batch,window,shape", CASES)
def test_delayed_multiround(q, batch, window, shape):
    p = Pair(q, batch, window, shape, padding=batch == 4)
    gen = torch.Generator().manual_seed(190)
    for r in range(12):
        p.drafts.copy_(projected(p.ops[0], batch, length=q, seed=300 + r))
        lengths = torch.randint(q + 1, (batch,), generator=gen)
        if r < 4:
            lengths.fill_((1, 0, 1, q)[r])
        lengths[r % batch] = (0, 1, q)[r % 3]
        p.accepted.copy_(lengths)
        p.verify()
        p.commit()
    p.flush()
    # Full descriptors become equal once both verifies start from canonical state.
    p.verify()
    same(*p.pending)
    p.accepted.zero_()
    p.commit()
    same(*p.pending)


@pytest.mark.parametrize(
    "vec,tile,pdl",
    [
        pytest.param(True, 32, True, marks=pytest.mark.exhaustive),
        pytest.param(True, 64, False, marks=pytest.mark.exhaustive),
        pytest.param(True, 128, True, marks=pytest.mark.exhaustive),
        (False, 64, True),
    ],
)
def test_delayed_switch_and_rotated_append(vec, tile, pdl, monkeypatch):
    monkeypatch.setattr(_phase_launch, "VECTOR_PUBLISH", vec)
    monkeypatch.setattr(_phase_launch, "PHASE_TILE", tile)
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    p = Pair(window=4, padding=True)
    for r in range(12):
        p.verify()
        p.commit(switch=bool(r % 2))
    p.flush()
    for _ in range(12):
        p.verify()
        p.commit()
        # Flush preserves rotation: repeated appends cross the physical ring end.
        p.flush()
    p.verify()
    same(*p.pending)


@pytest.mark.parametrize("offsets", [(0, 1, 3), (2, -1, 1)])
def test_delayed_capture_folds_only_capture_rows(offsets):
    p = Pair()
    for _ in range(2):
        p.verify()
        p.commit()
    p.accepted.fill_(2)
    p.verify()
    rows = torch.arange(3, dtype=torch.int32, device="cuda") + p.states[0].metadata.shape[0]
    captures = [
        {
            "snapshot_rows": rows,
            "snapshot_offsets": torch.tensor(offsets, dtype=torch.int64, device="cuda"),
            "snapshot_valid": torch.ones(3, dtype=torch.bool, device="cuda"),
        }
        for _ in p.ops
    ]
    p.commit(captures=captures)
    _bytes(captures[0]["snapshot_valid"], captures[1]["snapshot_valid"])
    p.flush()


@pytest.mark.parametrize("with_pending", [False, True])
def test_delayed_export_and_reject(with_pending):
    p = Pair()
    for _ in range(2):
        p.verify()
        p.commit()
    p.verify()
    rows = torch.arange(3, dtype=torch.int32, device="cuda") + p.states[0].metadata.shape[0]
    pairs = torch.stack((p.slots, rows), dim=1)
    validity = [torch.zeros(3, dtype=torch.bool, device="cuda") for _ in p.ops]
    for i, op in enumerate(p.ops):
        before = clone(p.states[i])
        op.export_snapshot(
            *p.weights,
            pairs=pairs,
            state=p.states[i],
            workspace=p.workspaces[i],
            snapshot_valid=validity[i],
            **(
                {
                    "pending": p.pending[i],
                    "accepted_lengths": p.accepted,
                    "offsets": torch.tensor([0, 1, 0], dtype=torch.int64, device="cuda"),
                }
                if with_pending
                else {}
            ),
        )
        for name in ("metadata", "ring"):
            _bytes(getattr(before, name), getattr(p.states[i], name))
        _bytes(
            before.checkpoint[: before.metadata.shape[0]],
            p.states[i].checkpoint[: before.metadata.shape[0]],
        )
        op.reject(slot_indices=p.slots, state=p.states[i], pending=p.pending[i])
        _bytes(before.metadata, p.states[i].metadata)
        _bytes(before.ring, p.states[i].ring)
        assert torch.all(p.pending[i].metadata[:, 5] == 4)
    _bytes(*validity)
    _bytes(p.states[0].checkpoint[rows.long()], p.states[1].checkpoint[rows.long()])
    _bytes(p.states[0].conv_history[rows.long()], p.states[1].conv_history[rows.long()])
    _bytes(p.states[0].snapshot_metadata, p.states[1].snapshot_metadata)
    for _ in range(4):
        p.verify()
        p.commit()
    p.flush()


@pytest.mark.parametrize("pdl", [pytest.param(False, marks=pytest.mark.exhaustive), True])
def test_delayed_d2_graph_and_flush(pdl, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    p = Pair(q=5, padding=True)
    initial = [clone(s) for s in p.states]
    pending = [[clone(p.pending[i]) for _ in range(2)] for i in range(2)]
    counters = [torch.zeros(2, p.batch, dtype=torch.int32, device="cuda") for _ in p.ops]
    positions = [p.positions.clone() for _ in range(2)]
    outputs = [[torch.empty_like(p.outputs[i]) for _ in range(2)] for i in range(2)]

    def round_(i, d):
        p.ops[i].verify(
            p.drafts,
            *p.weights,
            slot_indices=p.slots,
            positions=positions[d],
            state=p.states[i],
            pending=pending[i][d],
            out=outputs[i][d],
        )
        p.ops[i].commit(
            *p.weights,
            slot_indices=p.slots,
            accepted_lengths=p.accepted,
            state=p.states[i],
            pending=pending[i][d],
            workspace={"arrivals": counters[i][d]},
        )

    def flush():
        p.ops[1].flush(
            *p.weights,
            slot_indices=p.slots,
            state=p.states[1],
            workspace={"arrivals": counters[1][0]},
        )

    stream = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        for i in range(2):
            round_(i, 0)
        flush()
    main.wait_stream(stream)
    graphs = [[torch.cuda.CUDAGraph() for _ in range(2)] for _ in range(2)]
    for i, d in product(range(2), repeat=2):
        with torch.cuda.graph(graphs[i][d], stream=stream):
            round_(i, d)
    flush_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(flush_graph, stream=stream):
        flush()
    for i in range(2):
        _copy(p.states[i], initial[i])
    for r, m in enumerate((1, 0, 1, 5, 2, 1, 0, 3) * 3):
        d = r % 2
        positions[d].copy_(p.positions)
        p.accepted.fill_(m)
        p.accepted[1].fill_(5 - m)
        for i in range(2):
            graphs[i][d].replay()
        _bytes(outputs[0][d], outputs[1][d])
        p.positions.add_(p.accepted[:, None])
        if r % 7 == 6:
            flush_graph.replay()
            p.canonical()
    flush_graph.replay()
    p.canonical()
    assert all(torch.count_nonzero(c).item() == 0 for c in counters)


@pytest.mark.parametrize("pdl", [pytest.param(False, marks=pytest.mark.exhaustive), True])
def test_delayed_two_streams_disjoint_slots(pdl, monkeypatch):
    monkeypatch.setattr(_phase_launch, "PDL", pdl)
    p = Pair(q=5, batch=4)
    groups = (slice(0, 3), slice(3, 4))
    positions = [p.positions[g].clone() for g in groups]
    counters = torch.zeros(2, 3, dtype=torch.int32, device="cuda")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    main = torch.cuda.current_stream()
    initial = clone(p.states[1])
    accepted = [p.accepted[g].clone() for g in groups]

    def round_(i, d, row):
        g = groups[d]
        pending = type(p.pending[i])(p.pending[i].raw[g], p.pending[i].metadata[g])
        p.ops[i].verify(
            p.drafts[g],
            *p.weights,
            slot_indices=p.slots[g],
            positions=positions[d],
            state=p.states[i],
            pending=pending,
            out=p.outputs[i][g],
        )
        p.ops[i].commit(
            *p.weights,
            slot_indices=p.slots[g],
            accepted_lengths=accepted[d],
            state=p.states[i],
            pending=pending,
            workspace={"arrivals": row[: g.stop - g.start]},
        )

    streams[0].wait_stream(main)
    with torch.cuda.stream(streams[0]):
        for d in range(2):
            round_(1, d, counters[0])
    main.wait_stream(streams[0])
    graphs = []
    for d in range(2):
        graph = torch.cuda.CUDAGraph()
        streams[d].wait_stream(main)
        with torch.cuda.graph(graph, stream=streams[d]):
            round_(1, d, counters[d])
        main.wait_stream(streams[d])
        graphs.append(graph)
    _copy(p.states[1], initial)
    for r, m in enumerate((1, 0, 1, 5, 2, 1) * 4):
        for d in range(2):
            main.wait_stream(streams[d])
            accepted[d].fill_(m if (r + d) % 2 else 5 - m)
            round_(0, d, p.workspaces[0]["arrivals"])
            streams[d].wait_stream(main)
            with torch.cuda.stream(streams[d]):
                graphs[d].replay()
                positions[d].add_(accepted[d][:, None])
    for stream in streams:
        main.wait_stream(stream)
    _bytes(*p.outputs)
    assert torch.count_nonzero(counters).item() == 0
    for d, g in enumerate(groups):
        p.ops[1].flush(
            *p.weights,
            slot_indices=p.slots[g],
            state=p.states[1],
            workspace={"arrivals": counters[d][: g.stop - g.start]},
        )
    p.canonical()


@pytest.mark.parametrize(
    "window",
    [
        pytest.param(4, marks=pytest.mark.exhaustive),
        8,
        pytest.param(16, marks=pytest.mark.exhaustive),
    ],
)
def test_delayed_capacity_boundary_and_clamped_acceptance(window):
    p = Pair(q=16, window=window)
    p.verify()
    p.commit()  # n=1 on the delayed side.
    p.accepted.copy_(torch.tensor([window - 2, window - 1, 16], device="cuda"))
    p.verify()
    p.commit()  # n+m == L-1, L, and >L in the same launch.
    p.accepted.fill_(1)
    p.verify()
    p.commit()  # The first row now hits the fold boundary exactly.
    p.accepted.copy_(torch.tensor([-1, 17, 0], device="cuda"))
    p.verify()
    p.commit()  # Trusted indexing clamps to 0, Q, 0.
    p.flush()
