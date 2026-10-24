# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Device-independent CuTe configuration and allocation contracts."""

from types import SimpleNamespace

import pytest
import torch

# The MoE training modules import Triton; the CPU CI env has none (the GPU suites run these).
pytest.importorskip("triton")

from benchmark.kernel_tuning.moe_cute_gemm_bench import config_sweep
from ctikernels.moe.cute._provider import (
    BUCKET_TOKENS,
    DISPATCH,
    KINDS,
    CuteBackend,
    configs,
    overrides,
    parse_config,
)


@pytest.mark.parametrize(
    "spec,grid",
    [
        ("32x64x64s2w4", (2, 2)),
        ("128x64x32s4w8", (2, 4)),
        ("16x32x64s6w4n", (1, 4)),
        ("64x64x128s2w8g4x2n", (4, 2)),
        ("16x32x128s3w4g1x4t", (1, 4)),
        ("64x128x32s4w4g2x2nt", (2, 2)),
    ],
)
def test_config_roundtrip(spec, grid):
    cfg = parse_config(spec)
    assert (cfg.wm, cfg.wn) == grid
    assert parse_config(str(cfg)) == cfg


@pytest.mark.parametrize(
    "spec",
    [
        "16x64x64s4w4g2x2",
        "32x64x64s7w4",
        "64x64x64s4w8g2x2",
        "16x32x64s4w8g1x8",
    ],
)
def test_invalid_geometry(spec):
    with pytest.raises(ValueError):
        parse_config(spec)


@pytest.mark.parametrize("key", DISPATCH)
def test_dispatch_constraints(key):
    cap, bound = key
    for ni, nis, h in ((512, 256, 2048), (768, 0, 2048), (64, 64, 64)):
        table = configs(cap, bound if bound != float("inf") else 8192, ni, nis, h, {})
        assert len({cfg.bm for cfg in table.values()}) == 1
        for kind, cfg in table.items():
            # The bwd_slot entry serves the saved backward (one B operand).
            saved = kind == "bwd_slot"
            cfg.validate(kind, h, ni, nis, saved=saved)
            assert cfg.shared_bytes(kind, saved=saved) <= 99 * 1024


def test_limits_and_single_b():
    cfg = parse_config("32x256x32s3w8g2x4n")
    for kind in ("down", "dx_pair"):
        cfg.validate(kind, 256, 128, 64)
    with pytest.raises(ValueError):
        cfg.validate("gate_up", 256, 256, 256)
    with pytest.raises(ValueError):
        parse_config("16x32x128s5w4").validate("gate_up", 256, 128, 128)
    with pytest.raises(ValueError):
        parse_config("16x32x128s4w4").validate("down", 256, 128, 64)
    assert parse_config("32x64x64s2w4").two_ctas_smem("gate_up")


def test_saved_and_recomputed_allocation():
    be = object.__new__(CuteBackend)
    be.capability, be.h, be.ni, be.nis = (12, 0), 256, 128, 64
    be.override = overrides("gate_up:16x64x64s5w4,bwd_slot:16x32x32s6w4n")
    lay = SimpleNamespace(block_m=16)
    assert be._config("bwd_slot", lay, False) == be._config("gate_up", lay)
    assert be._config("bwd_slot", lay, True).bn == 32
    assert be.dw_tiles(None, 7, 128, 64) == 2
    assert be.dw_tiles(None, 7, 128, 64, saved=True) == 4
    for kind in KINDS:
        assert be._config(kind, lay).bm == 16


def test_sweep_and_fixed_override_bucket():
    sweep = config_sweep("down:16x64x64s6w4g1x4|32x256x32s3w8g2x4n,bwd_slot:16x32x64s6w4")
    assert len(sweep["down"]) == 2
    assert sweep["bwd_slot"][0]["gate_up"] == sweep["bwd_slot"][0]["bwd_slot"]
    for override in sweep["down"]:
        assert configs((12, 0), 7, 128, 64, 256, override) == configs(
            (12, 0), 2048, 128, 64, 256, override
        )


def test_tma_sweep_and_shared_memory():
    sweep = config_sweep("down:16x32x64s6w4g1x4|16x32x64s6w4g1x4t")
    baseline, tma = (entry["down"] for entry in sweep["down"])
    assert not baseline.tma and tma.tma
    assert baseline.workspace_bytes("down") == tma.workspace_bytes("down")
    assert tma.shared_bytes("down") == 1024 + tma.workspace_bytes("down") + 6 * 8
    # Alignment and barrier storage must participate in the launch limit.
    cfg = parse_config("16x32x128s5w4t")
    with pytest.raises(ValueError, match="shared-memory"):
        cfg.validate("gate_up", 256, 128, 128)
    for table in DISPATCH.values():
        assert all(not parse_config(spec).tma for spec in table.values())


def test_saved_tma_sweep_does_not_enlarge_recompute():
    sweep = config_sweep("bwd_slot:16x128x64s4w4g1x4t", saved="yes")
    override = sweep["bwd_slot"][0]
    assert set(override) == {"bwd_slot"}
    table = configs((12, 1), 8, 512, 512, 2048, override)
    assert table["bwd_slot"].tma
    assert table["gate_up"].bn == 32


def test_large_bm_single_kind_sweep():
    table = configs((12, 0), 7, 128, 64, 256, overrides("down:128x128x32s3w8g4x2"))
    assert {c.bm for c in table.values()} == {128}
    assert table["down"] == parse_config("128x128x32s3w8g4x2")
    for kind, cfg in table.items():
        cfg.validate(kind, 256, 128, 64)


@pytest.mark.parametrize("kind", ("down", "dx_pair"))
def test_slot_dispatch_and_cache(monkeypatch, kind):
    from ctikernels.moe.cute import _provider as provider

    be = object.__new__(CuteBackend)
    be.capability, be.h, be.ni, be.nis = (12, 0), 64, 64, 64
    be.override, be.pdl = {}, True
    lay = SimpleNamespace(
        block_m=16,
        inv=torch.empty(27, dtype=torch.int32),
        sorted_slot=torch.empty(64, dtype=torch.int32),
        block_expert=torch.empty(4, dtype=torch.int32),
    )
    a = torch.empty(64, 64 * (2 if kind == "dx_pair" else 1), dtype=torch.bfloat16)
    w = torch.empty(2, 128 if kind == "dx_pair" else 64, 64, dtype=torch.bfloat16)
    y = torch.empty(64, 64, dtype=torch.bfloat16)
    compiled, launched = [], []

    def compile_fake(*key):
        compiled.append(key)
        return lambda *args: launched.append(args)

    def launch():
        getattr(be, kind)(a, w, None, lay, y, ni=64, nis=64)

    monkeypatch.setattr(provider, "_COMPILED", {})
    monkeypatch.setattr(provider, "compile_kernel", compile_fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launch()
    launch()
    assert launched[-1][6] is y and len(compiled) == 1
    assert compiled[0][6] == 1 and compiled[0][-1] is True

    # Warm executables can be reused during capture without compilation.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    launch()
    assert len(compiled) == 1

    # PDL is part of the executable identity: toggling must not reuse a kernel
    # whose wait/trigger instructions and launch attribute were compiled out.
    be.pdl = False
    with pytest.raises(RuntimeError, match="Warm up"):
        launch()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launch()
    assert len(compiled) == 2 and compiled[-1][-1] is False


@pytest.mark.parametrize("capability", ((12, 0), (12, 1)))
@pytest.mark.parametrize("setting,enabled", ((None, True), ("0", False), ("1", True)))
def test_pdl_environment(monkeypatch, capability, setting, enabled):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr("ctikernels.moe.cute._provider.supports", lambda *_: True)
    monkeypatch.delenv("CTI_MOE_CUTE_CONFIG", raising=False)
    if setting is None:
        monkeypatch.delenv("CTI_MOE_CUTE_PDL", raising=False)
    else:
        monkeypatch.setenv("CTI_MOE_CUTE_PDL", setting)
    assert CuteBackend("cuda", 128, 64, 256).pdl is enabled


@pytest.mark.parametrize("cap", ((12, 0), (12, 1)))
def test_layout_bm_maps_back_to_its_bucket(cap):
    be = object.__new__(CuteBackend)
    be.capability, be.h, be.ni, be.nis, be.override = cap, 2048, 512, 512, {}
    for tokens in (8, 512, 4096, 8192):
        bm = be.block_m(None, tokens, 512, 512)
        assert configs(cap, BUCKET_TOKENS[bm], 512, 512, 2048, {})["gate_up"].bm == bm
        for kind in KINDS:
            cfg = be._config(kind, SimpleNamespace(block_m=bm), kind == "bwd_slot")
            assert cfg.bm == bm
            cfg.validate(kind, 2048, 512, 512, kind == "bwd_slot")


def test_gb10_large_chunks_use_128_rows():
    assert configs((12, 1), 8192, 768, 0, 2048, {})["gate_up"].bm == 128
    assert configs((12, 1), 4096, 768, 0, 2048, {})["gate_up"].bm == 64
    be = object.__new__(CuteBackend)
    be.capability, be.h, be.ni, be.nis, be.override = (12, 1), 2048, 768, 0, {}
    lay = SimpleNamespace(block_m=128)
    assert be._config("down", lay).bn == 256
