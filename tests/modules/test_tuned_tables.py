# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import json

import pytest

from ctikernels._runtime import tables
from ctikernels._runtime.tables import DeviceClass, Dim, TunedOp, encode, pow2_edges

PRO = DeviceClass(12, 0, 188)
GB10 = DeviceClass(12, 1, 48)


def _op(version=1):
    return TunedOp(
        "test.op",
        version=version,
        space={"block": (16, 32, 64), "warps": (4, 8)},
        dims=(Dim("m", pow2_edges(1, 64)), Dim("k"), Dim("sizes")),
        default=lambda shape, device: {"block": 16, "warps": 8 if device.sms > 100 else 4},
    )


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "DATA", tmp_path)
    return tmp_path


def _write(op, rows, device=PRO.name):
    tables.table_path(op.name, device).write_text(encode(op, rows))


def test_bucket_edges_and_exact_dims():
    op = _op()
    assert op.bucket(m=3, k=2048, sizes=(4096, 512)) == (4, 2048, (4096, 512))
    assert op.bucket(m=0, k=1, sizes=(1,))[0] == 1
    assert op.bucket(m=10_000, k=1, sizes=(1,))[0] == 64
    with pytest.raises(ValueError):
        op.bucket(m=1, k=1)


@pytest.mark.parametrize(
    "m, k, expected",
    [
        (4, 2048, {"block": 32, "warps": 8}),  # exact
        (8, 2048, {"block": 32, "warps": 8}),  # inherits from m=4
        (16, 2048, {"block": 64, "warps": 4}),  # exact
        (64, 2048, {"block": 64, "warps": 4}),  # inherits from m=16
        (2, 2048, {"block": 16, "warps": 8}),  # nothing below: default
        (16, 4096, {"block": 16, "warps": 8}),  # other dims differ: default
    ],
)
def test_lookup_exact_then_smaller_bucket_then_default(data, m, k, expected):
    op = _op()
    _write(
        op,
        {
            (4, 2048, (512,)): {"block": 32, "warps": 8},
            (16, 2048, (512,)): {"block": 64, "warps": 4},
        },
    )
    assert op.lookup(PRO, m=m, k=k, sizes=(512,)) == expected
    assert op.lookup(GB10, m=m, k=k, sizes=(512,)) == {"block": 16, "warps": 4}


def test_encode_round_trips_and_is_sorted(data):
    op = _op()
    rows = {(16, 1, (2, 3)): {"block": 64, "warps": 8}, (1, 1, (2, 3)): {"block": 32, "warps": 4}}
    _write(op, rows)
    text = tables.table_path(op.name, PRO.name).read_text()
    assert json.loads(text)["rows"][0][0][0] == 1
    assert op.table(PRO) == rows


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(version=2),
        lambda d: d.update(params=["block"]),
        lambda d: d["rows"][0][1].__setitem__(0, 128),
        lambda d: d["rows"].append(d["rows"][0]),
    ],
)
def test_stale_or_invalid_tables_raise(data, edit):
    op = _op()
    _write(op, {(1, 1, (2,)): {"block": 32, "warps": 4}})
    path = tables.table_path(op.name, PRO.name)
    content = json.loads(path.read_text())
    edit(content)
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        op.lookup(PRO, m=1, k=1, sizes=(2,))


def test_default_must_lie_in_space():
    op = TunedOp(
        "test.bad",
        version=1,
        space={"block": (16,)},
        dims=(Dim("m", (1,)),),
        default=lambda shape, device: {"block": 32},
    )
    with pytest.raises(ValueError):
        op.default(PRO, m=1)


def _tiled():
    return TunedOp(
        "test.tiled",
        version=1,
        space={"tile": (16, 64)},
        dims=(Dim("m", pow2_edges(1, 64)), Dim("rank", (16, 64))),
        default=lambda shape, device: {"tile": 16},
        valid=lambda config, shape: config["tile"] <= shape["rank"],
    )


def test_valid_hook_guards_lookup_and_load(data):
    op = _tiled()
    assert op.valid({"tile": 64}, {"m": 3, "rank": 40})  # rank 40 buckets to 64
    assert not op.valid({"tile": 64}, {"m": 3, "rank": 8})
    _write(op, {(1, 64): {"tile": 64}})
    assert op.lookup(PRO, m=8, rank=64) == {"tile": 64}  # inherited from m=1
    _write(op, {(1, 16): {"tile": 64}})
    op = _tiled()
    with pytest.raises(ValueError):
        op.lookup(PRO, m=1, rank=16)


@pytest.mark.parametrize("path", sorted(tables.DATA.glob("*@*.json")), ids=lambda p: p.name)
def test_shipped_files_are_well_formed(path):
    data = json.loads(path.read_text())
    if path.name.endswith(".provenance.json"):
        assert data["device_class"] in path.name and data["op"] in path.name
        return
    assert (
        data["format"] == tables.FORMAT and path.name == f"{data['op']}@{path.name.split('@')[1]}"
    )
    buckets = [json.dumps(bucket) for bucket, _ in data["rows"]]
    assert len(buckets) == len(set(buckets))
    assert all(
        len(b) == len(data["dims"]) and len(v) == len(data["params"]) for b, v in data["rows"]
    )
