# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GraphWorkspace slot sizing, freezing and lifetime, without CUDA."""

import weakref

import pytest
import torch

from ctikernels import GraphWorkspace


def _size(max_rows, width):
    return (max_rows * width, width)  # e.g. rows x width scratch plus per-column counters


def _allocate(counts):
    return torch.empty(counts[0]), torch.zeros(counts[1], dtype=torch.int32)


def _capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


@pytest.mark.parametrize("ascending", [False, True])
def test_two_consumers_share_one_frozen_scope(ascending, monkeypatch):
    rows = (1, 8, 64) if ascending else (64, 8, 1)
    a, b = ("a", "cpu", torch.float32), ("b", "cpu", torch.float32)
    ws = GraphWorkspace(max_rows=64)
    with ws:
        assert GraphWorkspace.active() is ws
        for width in (16, 32):  # call shapes, any order; reservation grows elementwise
            ws.reserve(a, (width,), _size, _allocate)
        ws.reserve(b, (8,), _size, _allocate)
        _capturing(monkeypatch)
        with ws:  # nesting is a no-op
            taken = [ws.take(a, m, (m * 32, 32)) for m in rows]
        assert all(t is taken[0] for t in taken) and taken[0][0].numel() == 64 * 32
        assert ws.take(b, 64, (64 * 8, 8))[0].data_ptr() != taken[0][0].data_ptr()
        assert ws.nbytes == (64 * 32 + 64 * 8) * 4 + (32 + 8) * 4
        ws.reserve(a, (128,), _size, _allocate)  # frozen: no growth in or after a capture
        with pytest.raises(RuntimeError, match=r"slot \('a'.* too small at 64 rows"):
            ws.take(a, 64, (64 * 128, 128))
        with pytest.raises(RuntimeError, match="65 rows exceed max_rows=64"):
            ws.take(a, 65, (0, 0))
        with pytest.raises(RuntimeError, match="was not reserved"):
            ws.take(("c", "cpu", torch.float32), 1, (1, 1))
        with pytest.raises(RuntimeError, match="another GraphWorkspace"):
            GraphWorkspace(max_rows=1).__enter__()
    assert GraphWorkspace.active() is None


def test_eager_take_sees_growth_until_a_capture(monkeypatch):
    slot = ("a", "cpu", torch.float32)
    with GraphWorkspace(max_rows=4) as ws:
        ws.reserve(slot, (2,), _size, _allocate)
        small = ws.take(slot, 4, (8, 2))
        ws.reserve(slot, (4,), _size, _allocate)  # not frozen by an eager take
        grown = ws.take(slot, 4, (16, 4))
        assert grown[0].numel() == 16 and grown[0].data_ptr() != small[0].data_ptr()
        _capturing(monkeypatch)
        assert ws.take(slot, 4, (16, 4)) is grown
        ws.reserve(slot, (8,), _size, _allocate)
        assert ws.take(slot, 4, (16, 4)) is grown


def test_close_releases_and_empty_scope_allocates_nothing():
    with GraphWorkspace(max_rows=4) as empty:
        pass
    assert empty.nbytes == 0
    ws = GraphWorkspace(max_rows=4)
    with ws:
        ws.reserve(("a", "cpu", torch.float32), (2,), _size, _allocate)
        alive = weakref.ref(ws.take(("a", "cpu", torch.float32), 4, (8, 2))[0])
    assert alive() is not None
    ws.close()
    assert alive() is None and ws.nbytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        ws.__enter__()
    with pytest.raises(ValueError, match="max_rows"):
        GraphWorkspace(max_rows=0)


def test_slots_reserved_in_inference_mode_are_normal_tensors():
    group = GraphWorkspace(max_rows=4)
    with torch.inference_mode():
        group.reserve(("slot",), (), lambda rows: (rows,), lambda counts: (torch.zeros(counts[0]),))
    (work,) = group.take(("slot",), 4, (4,))
    assert not work.is_inference()
    work.add_(1)  # in place outside inference mode
