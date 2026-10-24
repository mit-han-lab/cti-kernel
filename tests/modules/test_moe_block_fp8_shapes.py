# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Exact-shape launch policy and Qwen3 benchmark metadata, without a GPU."""

import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.block_fp8_moe_market import sglang_config_source, sol
from benchmark.kernel_tuning.block_fp8_moe_sol import CHAINS, DEVICES, SHAPES, estimate
from ctikernels import _moe_block_fp8_triton_table as table
from ctikernels.moe_impl import block_fp8_moe_config


def test_accepted_qwen35_rows_unchanged():
    # Fingerprint of the accepted MEASURED values (fcf2fd6, plus the 2026-10-19 PRO E=256
    # row whose T=4096 FC1 takes the fused activation epilogue), with the old integer keys.
    old = {
        device: {e: rows for (e, h, i, k), rows in shapes.items() if (h, i, k) == (2048, 512, 8)}
        for device, shapes in table.MEASURED.items()
    }
    assert hashlib.sha256(repr(old).encode()).hexdigest() == (
        "0ee04c498a28f875e278559c83c4bc70b3fcd945285fe43f6ce20290e1b5fb7c"
    )


def test_exact_shape_rows_and_copies():
    for device, shapes in table.TABLE.items():
        for shape, rows in shapes.items():
            e, h, i, k = shape
            for limit, fc1, fc2 in rows:
                actual = table.selection(device, limit, e, h, i, k)
                expected = tuple(
                    dict(
                        c,
                        route_blocks=c.get(
                            "route_blocks", limit * k <= 64 and c["BLOCK_SIZE_M"] == 16
                        ),
                    )
                    for c in (fc1, fc2)
                )
                assert actual == expected
                actual[0]["BLOCK_SIZE_N"] = -1
                assert table.selection(device, limit, e, h, i, k) == expected


@pytest.mark.parametrize("device", [(12, 0, 170), (12, 0, 188), (12, 1, 48), (99, 0, 1)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 2048, 768, 8),
        (256, 2048, 768, 8),
        (128, 4096, 512, 8),
        (128, 2048, 512, 4),
    ],
)
@pytest.mark.parametrize("m", [0, 1, 2, 3, 4, 8, 64, 128, 4096, 4097])
def test_unmeasured_shape_uses_heuristic(device, shape, m):
    e, h, i, k = shape
    assert shape not in table.MEASURED.get(device, {})
    expected = block_fp8_moe_config(m, e, 2 * i, h, k, capability=device[:2])
    expected["route_blocks"] = m * k <= 64 and expected["BLOCK_SIZE_M"] == 16
    assert table.selection(device, m, e, h, i, k) == (expected, expected)


def test_new_shape_can_receive_measured_rows(monkeypatch):
    shape = (128, 2048, 768, 8)
    fc1 = block_fp8_moe_config(1, 128, 1536, 2048, 8, capability=(99, 0)) | {"BLOCK_SIZE_N": 32}
    fc2 = dict(fc1, BLOCK_SIZE_N=128)
    monkeypatch.setitem(table.TABLE, (99, 0, 1), {shape: [(4, fc1, fc2)]})
    assert table.selection((99, 0, 1), 4, *shape) == (
        dict(fc1, route_blocks=True),
        dict(fc2, route_blocks=True),
    )
    assert table.selection((99, 0, 1), 4, 128, 2048, 512, 8)[0]["BLOCK_SIZE_N"] == 64


@pytest.mark.parametrize("ep", [1, 2])
def test_qwen3_sol(ep):
    e, h, i, k = SHAPES["qwen3"]
    unique, ws, act, flops, _, _, launch, _ = estimate("RTX 5090", 1, "decode", "qwen3", ep)
    assert unique == pytest.approx(k / ep)
    assert ws == pytest.approx(unique * (3 * h * i + 4 * (12 * 16 + 16 * 6)))
    assert act == 4 * h + 4 * (k / ep) * i
    assert flops == 6 * (k / ep) * h * i
    assert launch == CHAINS["decode"] * DEVICES["RTX 5090"][2]
    assert estimate("RTX 5090", 4096, "grouped", "qwen3", ep)[0] == pytest.approx(e / ep)
    ids = torch.arange(k, dtype=torch.int32).view(1, k) * (e // k)
    cases = [(torch.empty(1, h), torch.ones(1, k) / k, ids)]
    floor = sol(cases, ep, 4, None, shape="qwen3")
    assert floor["weight_scale_read_bytes"] == ws
    assert floor["activation_traffic_bytes"] == act


def test_sglang_config_provenance(tmp_path, monkeypatch):
    triton = pytest.importorskip("triton")
    root = tmp_path / "configs"
    current = root / ("triton_" + triton.__version__.replace(".", "_"))
    older = root / "triton_0_0_1"
    current.mkdir(parents=True)
    older.mkdir()
    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    module = SimpleNamespace(
        __file__=str(tmp_path / "config.py"),
        get_config_file_name=lambda e, i, dtype, block, down_moe: f"{e}-{i}-{down_moe}.json",
        get_moe_configs=lambda *args, **kwargs: None,
    )
    assert sglang_config_source(module, 64, 768) == {"source": "default"}
    module.get_moe_configs = lambda *args, **kwargs: {1: {"BLOCK_SIZE_M": 16}}
    (older / "64-768-False.json").write_text("{}")
    assert sglang_config_source(module, 64, 768, down=True) == {
        "source": "tuned_json",
        "file": "triton_0_0_1/64-768-False.json",
    }
    (current / "64-768-False.json").write_text("{}")
    assert sglang_config_source(module, 64, 768)["file"].startswith(current.name)
    (older / "64-768-True.json").write_text("{}")
    assert sglang_config_source(module, 64, 768, down=True)["file"].endswith("64-768-True.json")


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
def test_qwen3_offline_activation(arch, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", arch)
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass.cute")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        pytest.skip("requires DSL 4.8.0")
    from ctikernels._moe_block_fp8_cute_act import compile_kernel

    fn = compile_kernel(768, arch=arch, options=f"--keep-ptx --dump-dir={tmp_path}")
    ptx = fn.__ptx__
    if "\n" not in ptx:
        ptx = Path(ptx).read_text()
    assert f".target {arch}" in ptx
    assert "e4m3" in ptx


def test_gb10_qwen3_decode_rows_use_alignment():
    for e in (128, 64):
        for m in (1, 2, 3, 4, 8):
            fc1, fc2 = table.selection((12, 1, 48), m, e, 2048, 768, 8)
            assert fc1["route_blocks"] == fc2["route_blocks"] == (m == 1)
