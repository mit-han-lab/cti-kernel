# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU regressions for fixed-buffer replay host validation."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay
from ctikernels.gdn.replay import _api, _layout

from ._replay_case import case, clone, graph_buffers, projected


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def invocation(mode, checks, monkeypatch=None):
    op, state, pending, weights, slots, pos, status = case(batch=1, hv=1, q=2)
    op._host_trusted = checks == "trusted"
    kw = {"state": state, "status": status, "weights": weights, "slot_indices": slots}
    if mode in ("step", "verify"):
        length = op.verify_length if mode == "verify" else None
        kw.update(
            projected=projected(op, 1, length=length),
            positions=pos[:, None] + torch.arange(length) if length else pos,
            out=torch.empty(
                (1, length, op.geometry.value_width) if length else (1, op.geometry.value_width),
                dtype=torch.bfloat16,
            ),
        )
    if mode in ("verify", "commit"):
        kw["pending"] = pending
    if mode == "commit":
        kw.update(
            accepted_lengths=torch.zeros(1, dtype=torch.int32),
            snapshot_rows=torch.tensor([state.metadata.shape[0]], dtype=torch.int32),
            snapshot_offsets=torch.zeros(1, dtype=torch.int64),
            snapshot_valid=status.valid,
        )
    if monkeypatch is not None:
        monkeypatch.setitem(
            op._backends, "torch_reference", SimpleNamespace(run=lambda *a, **k: None)
        )
    return op, kw


def test_trusted_cpu_constructor():
    op = GatedDeltaNetReplay(num_k_heads=1, num_v_heads=1, device="cpu", checks="trusted")
    # Portable providers keep strict status semantics; the request only enables the memo.
    assert op.checks == "strict" and op._host_trusted


@pytest.mark.parametrize("mode", ["step", "verify", "commit"])
@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_repeated_checks(monkeypatch, mode, checks):
    op, kw = invocation(mode, checks, monkeypatch)
    counts = {"overlap": 0, "validation": 0}
    overlap, validate = _api._overlap, op._check_overlaps

    def counted_overlap(a, b):
        counts["overlap"] += 1
        return overlap(a, b)

    def counted_validation(*args, **kwargs):
        counts["validation"] += 1
        return validate(*args, **kwargs)

    monkeypatch.setattr(_api, "_overlap", counted_overlap)
    monkeypatch.setattr(op, "_check_overlaps", counted_validation)
    op._run(mode, **kw)
    first = counts.copy()
    for _ in range(4):
        op._run(mode, **kw)
    if checks == "trusted":
        assert counts == first
    else:
        assert counts["validation"] == 5 * first["validation"] > 0
        if mode == "commit":
            assert counts["overlap"] == 5 * first["overlap"] > 0


@pytest.mark.parametrize("checks", ["trusted", "strict"])
def test_new_alias_rejected_every_time(monkeypatch, checks):
    op, kw = invocation("step", checks, monkeypatch)
    op._run("step", **kw)
    kw["projected"] = (
        kw["state"].checkpoint.view(torch.bfloat16).flatten()[: op.geometry.proj_width].view(1, -1)
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="overlapping buffers: checkpoint, projected"):
            op._run("step", **kw)


@pytest.mark.parametrize("mode", ["step", "verify"])
def test_repeated_results_bitwise(mode):
    op, kw = invocation(mode, "trusted")
    before = clone(kw["state"])
    op._run(mode, **kw)
    output, state = kw["out"].clone(), clone(kw["state"])
    pending = clone(kw["pending"]) if mode == "verify" else None
    for checks in ("trusted", "strict"):
        op._host_trusted = checks == "trusted"
        for field in fields(before):
            getattr(kw["state"], field.name).copy_(getattr(before, field.name))
        op._run(mode, **kw)
        assert kw["status"].code.item() == 0
        assert output.count_nonzero() > 0
        assert torch.equal(output, kw["out"])
        expected = [(kw["state"], state)]
        if pending is not None:
            expected.append((kw["pending"], pending))
        for obj, reference in expected:
            for field in fields(obj):
                assert torch.equal(getattr(obj, field.name), getattr(reference, field.name))


def test_cached_state_and_status_still_validate_new_identity(monkeypatch):
    op, kw = invocation("step", "trusted", monkeypatch)
    op._run("step", **kw)
    with pytest.raises(ValueError, match="invalid status layer"):
        op._run("step", **{**kw, "status": replace(kw["status"], layer=2)})
    with pytest.raises((TypeError, ValueError)):
        op._run(
            "step",
            **{**kw, "state": replace(kw["state"], metadata=kw["state"].metadata.float())},
        )
    kw["weights"][1].requires_grad_(True)
    with torch.no_grad(), pytest.raises(ValueError, match="frozen base weight"):
        op._run("step", **kw)


def test_state_and_status_field_validation_is_cached(monkeypatch):
    op, kw = invocation("verify", "trusted", monkeypatch)
    names = []
    check = op._tensor

    def counted(name, *args, **kwargs):
        names.append(name)
        return check(name, *args, **kwargs)

    monkeypatch.setattr(op, "_tensor", counted)
    op._run("verify", **kw)
    assert "checkpoint" in names and "status.records" in names and "pending.raw" in names
    names.clear()
    op._run("verify", **kw)
    assert (
        "checkpoint" not in names and "status.records" not in names and "pending.raw" not in names
    )
    kw["state"].checkpoint.requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        op._run("verify", **kw)
    with torch.no_grad():
        op._run("verify", **kw)
    with pytest.raises(RuntimeError, match="autograd"):
        op._run("verify", **kw)


def test_status_slice_exemption_is_part_of_cache_key(monkeypatch):
    op, kw = invocation("commit", "trusted", monkeypatch)
    status = graph_buffers(op, batch=2).status(0, batch_size=1)
    kw.update(status=status, snapshot_valid=status.valid)
    op._run("commit", **kw)
    kw["status"] = replace(status, batch_size=2)
    with pytest.raises(ValueError, match="overlapping buffers: status.validity, snapshot_valid"):
        op._run("commit", **kw)


def test_advance_positions_changes_readonly_exemption(monkeypatch):
    op, kw = invocation("step", "trusted", monkeypatch)
    kw["slot_indices"] = kw["positions"]
    op._run("step", **kw)
    with pytest.raises(ValueError, match="overlapping buffers: slot_indices, positions"):
        op._run("step", **kw, advance_positions=True)


def test_validation_cache_is_bounded_and_rechecks_evicted_buffers(monkeypatch):
    op, kw = invocation("step", "trusted", monkeypatch)
    op._run("step", **kw)
    state_key = op._validation_key("state", _api._fields(kw["state"]))
    for i in range(4096):
        op._remember(("eviction", i))
    assert len(op._validated) == 4096
    assert state_key not in op._validated
    names = []
    check = op._tensor

    def counted(name, *args, **kwargs):
        names.append(name)
        return check(name, *args, **kwargs)

    monkeypatch.setattr(op, "_tensor", counted)
    op._run("step", **kw)
    assert "checkpoint" in names and "status.records" in names
    assert len(op._validated) == 4096


def test_overlapping_pairs_random_views():
    gen = torch.Generator().manual_seed(19)
    for _ in range(30):
        base = torch.empty(512, dtype=torch.uint8)
        views = [
            base[:0],
            base[:16],
            base[16:32],
            base[::2],
            base[1::2],
            base.view(torch.int32)[::3],
            base.view(torch.float64)[1::2],
            torch.empty(4),
        ]
        for _ in range(15):
            rows, cols = torch.randint(0, 9, (2,), generator=gen).tolist()
            strides = tuple(torch.randint(0, 16, (2,), generator=gen).tolist())
            offset = int(torch.randint(0, 128, (), generator=gen))
            views.append(base.as_strided((rows, cols), strides, offset))
        expected = {
            (i, j)
            for i, a in enumerate(views)
            for j, b in enumerate(views)
            if i < j and _layout.overlaps(a, b)
        }
        actual = list(_layout.overlapping_pairs(views))
        assert len(actual) == len(set(actual))
        assert set(actual) == expected


def test_overlapping_pairs_prunes_disjoint_extents(monkeypatch):
    base = torch.empty(128)
    views = [base[i : i + 8] for i in range(0, 128, 8)] + [base[:0]]
    monkeypatch.setattr(_layout, "overlaps", lambda *a: pytest.fail("disjoint extents"))
    assert list(_layout.overlapping_pairs(views)) == []
