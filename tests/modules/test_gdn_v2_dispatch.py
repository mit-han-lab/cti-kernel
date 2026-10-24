# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only GB10 row dispatch: host shapes, the capturing executable and symbolic keys."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import _cute_prefill as host
from ctikernels.gdn._contract import Geometry
from ctikernels.gdn.inference import GatedDeltaNetPrefill


@pytest.fixture
def launcher_factory(monkeypatch):
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=48),
    )

    def make(hv, cc=(12, 1)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: cc)
        return host.Launcher(torch.device("cpu"), object(), (8 if hv in (16, 24) else 16, hv))

    return make


def geometry_for(hv):
    return Geometry(8 if hv in (16, 24) else 16, hv)


# vsplit selected on GB10; 0 means the row stays on v1.
@pytest.mark.parametrize(
    "hv,tokens,count,v2",
    [
        (32, 255, 1, 0),
        (32, 255, 8, 0),
        (32, 256, 1, 1),
        (32, 256, 8, 1),
        (32, 1024, 1, 1),
        (48, 255, 1, 0),
        (48, 255, 8, 0),
        (48, 256, 1, 1),
        (48, 256, 8, 1),
        (24, 2, 1, 0),
        (24, 2, 2, 0),
        (24, 511, 2, 0),
        (24, 512, 1, 0),
        (24, 1023, 1, 0),
        (24, 1024, 1, 2),
        (24, 512, 2, 1),
        (24, 8192, 1, 2),
        (24, 8192, 2, 1),
        (16, 1024, 2, 0),
        (16, 4095, 4, 0),
        (16, 4095, 5, 0),
        (16, 4096, 1, 0),
        (16, 4096, 2, 1),
        (16, 4096, 3, 1),
        (16, 4096, 4, 2),
        (16, 4096, 5, 1),
        (16, 8192, 8, 1),
    ],
)
@pytest.mark.parametrize("captures", [0, 1])
def test_gb10_boundaries_with_and_without_captures(
    launcher_factory, hv, tokens, count, v2, captures
):
    launcher = launcher_factory(hv)
    geometry = geometry_for(hv)
    x = torch.empty(tokens, geometry.proj_width, device="meta")
    tune, tune_key = launcher._row_tune(tokens, count)
    selected = bool(v2)
    assert bool(tune.get("v2", 0)) == selected
    host.validate_captures(tune, captures)
    key = launcher.compile_key(geometry, x, count, captures)
    assert key[5:] == (tune_key, captures)
    # The guard's variant is the non-capturing key; its capturing twin compiles beside it.
    assert launcher.capture_variant(geometry, x, count) == (*key[:-1], 0)
    assert launcher._twin(key) == (*key[:-1], 1 - captures)
    effective = launcher._effective_tune(hv, key[3], key)
    if selected:
        assert tune["vsplit"] == v2 and tune["overlap"] == 0
        assert key[2:4] == ("serial", False)
        assert effective["overlap"] == 0
        assert host.pipeline_variant(tokens, effective) == "serial"
        assert [
            r["kernel"] for r in host.launch_info(geometry.k_heads, hv, tokens, count, effective)
        ] == ["fused_v2", "commit"]
    elif hv == 16:
        # GB10 8x16 v1 rows use two-plane Q K^T (the fla-core max-error gate).
        assert tune == launcher.tune | {"aq_qk": 1}
    else:
        assert (tune, tune_key) == (launcher.tune, launcher.tune_key)


def test_first_matching_v2_row(launcher_factory, monkeypatch):
    monkeypatch.setitem(
        host.ROW_TUNES,
        ((12, 1), (8, 24)),
        (
            (100, 200, 2, 4, {"v2": 1}),
            (100, 200, 4, {"o_split": 1}),
        ),
    )
    launcher = launcher_factory(24)
    for tokens in (100, 200):
        for count in (2, 4):
            assert launcher._row_tune(tokens, count)[0]["v2"] == 1
        assert launcher._row_tune(tokens, 1)[0]["o_split"] == 1
    for tokens, count in ((99, 2), (201, 2), (150, 5)):
        assert launcher._row_tune(tokens, count) == (launcher.tune, launcher.tune_key)


@pytest.mark.parametrize("hv", [16, 24, 32, 48])
def test_sm120_stays_v1(launcher_factory, hv):
    launcher = launcher_factory(hv, (12, 0))
    for tokens in (1, 255, 256, 512, 1024, 4096, 8192, 16384):
        for count in (1, 2, 4, 5, 8):
            assert "v2" not in launcher._row_tune(tokens, count)[0]
    # Every sm120 row table (all geometries) stays v1.
    assert all(
        row[-1].get("v2", 0) == 0
        for (cc, _), rows in host.ROW_TUNES.items()
        if cc == (12, 0)
        for row in rows
    )


@pytest.mark.parametrize("hv", [16, 24, 32, 48])
@pytest.mark.parametrize("value", ["v2=0", "bvo=32", "v2=1"])
def test_explicit_tune_bypasses_gb10_rows(launcher_factory, monkeypatch, hv, value):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", value)
    launcher = launcher_factory(hv)
    assert launcher.row_tunes == ()
    assert launcher._row_tune(8192, 8) == (launcher.tune, launcher.tune_key)
    assert launcher.tune == host.read_tune(value, (12, 1), (8 if hv in (16, 24) else 16, hv))
    if value == "v2=1":
        geometry = geometry_for(hv)
        x = torch.empty(8192, geometry.proj_width, device="meta")
        key = launcher.compile_key(geometry, x, 8, 1)
        assert launcher.capture_variant(geometry, x, 8) == launcher._twin(key)
        assert dict(key[5])["v2"] == 1 and key[-1] == 1


@pytest.mark.parametrize(
    "cc,hv,tokens,count,overrides",
    [
        ((12, 0), 16, 256, 1, {"bv": 16, "overlap": 0, "o_split": 1, "bvo": 128}),
        ((12, 0), 24, 512, 1, {"o_split": 1, "bvo": 64}),
        ((12, 0), 24, 4096, 1, {"o_split": 1, "bvo": 64}),
        ((12, 0), 24, 4097, 1, {"o_split": 1, "bvo": 128}),
        ((12, 0), 32, 512, 1, {"o_split": 1, "bvo": 64}),
        ((12, 0), 32, 1024, 1, {"o_split": 1, "bvo": 64}),
        ((12, 0), 32, 2048, 2, {}),
        ((12, 0), 48, 8192, 8, {}),
        ((12, 1), 16, 4095, 5, {"aq_qk": 1}),
        ((12, 1), 24, 1023, 1, {}),
        ((12, 1), 32, 255, 1, {}),
        ((12, 1), 48, 255, 1, {}),
    ],
)
def test_existing_v1_compile_keys_unchanged(launcher_factory, cc, hv, tokens, count, overrides):
    launcher = launcher_factory(hv, cc)
    geometry = geometry_for(hv)
    x = torch.empty(tokens, geometry.proj_width, device="meta")
    tune = (
        host.DEFAULT_TUNE | host.GEOMETRY_TUNES[(cc, (8 if hv in (16, 24) else 16, hv))] | overrides
    )
    expected = (
        geometry.k_heads,
        hv,
        "serial" if tokens < tune["overlap_min_tokens"] or not tune["overlap"] else 0,
        count * hv * (128 // tune["bv"]) <= 48,
        geometry.proj_width,
        tuple(sorted(tune.items())),
        0,
    )
    assert launcher.compile_key(geometry, x, count) == expected
    assert launcher.capture_variant(geometry, x, count) == expected


@pytest.mark.parametrize("hv", [16, 24, 32, 48])
@pytest.mark.parametrize("captures", [0, 1])
def test_count_bands_keep_finite_symbolic_keys(launcher_factory, hv, captures):
    launcher = launcher_factory(hv)
    geometry = geometry_for(hv)
    keys = {
        launcher.compile_key(
            geometry, torch.empty(t, geometry.proj_width, device="meta"), n, captures
        )
        for t in (8192, 16384)
        for n in range(1, 258)
    }
    # Hv16: v1 at N=1, v2 vsplit 2 at N=4, v2 vsplit 1 otherwise. Hv24: v2 vsplit 2
    # at N=1, vsplit 1 from N=2. Hv32/48: one v2 key.
    expected = {16: 3, 24: 2}.get(hv, 1)
    assert len(keys) == expected


@pytest.mark.parametrize("hv", [16, 24, 32, 48])
@pytest.mark.parametrize("captures", [0, 1])
def test_call_scratch_ignores_captures(launcher_factory, hv, captures):
    launcher = launcher_factory(hv)
    geometry = geometry_for(hv)
    tokens, count = 8192, 8
    x = torch.empty(tokens, geometry.proj_width, device="meta", dtype=torch.bfloat16)
    key = launcher.compile_key(geometry, x, count, captures)
    calls = []
    launcher.compiled[key] = lambda *args: calls.append(args)

    def tensor(*shape, dtype=torch.float32):
        return torch.empty(*shape, device="meta", dtype=dtype)

    launcher(
        geometry,
        x,
        tensor(geometry.conv_dim, 4, dtype=torch.bfloat16),
        tensor(hv),
        tensor(hv),
        tensor(128),
        tensor(count + 1, dtype=torch.int32),
        tensor(count, dtype=torch.int32),
        tensor(count, dtype=torch.bool),
        tensor(count, hv, 128, 128),
        tensor(count, 3, geometry.conv_dim, dtype=torch.bfloat16),
        capture_rows=tensor(count, dtype=torch.int32) if captures else None,
    )
    (args,) = calls
    assert args[10].numel() == 0  # Captures stay on v2, without Q or ready scratch.
    assert args[23].numel() == 0
    # One device-computed offset per sequence (int64), or the cu_seqlens placeholder.
    assert args[26].numel() == (count if captures else count + 1)
    assert args[26].dtype == (torch.int64 if captures else torch.int32)
    assert args[27].numel() == count


def test_cute_provider_still_supports_state_capture():
    operation = object.__new__(GatedDeltaNetPrefill)
    operation._kernel = SimpleNamespace(provider="cute")
    assert operation.supports_state_capture
