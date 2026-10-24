# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU policy and offline compiler checks; compiled callables are NEVER invoked."""

from dataclasses import replace
from pathlib import Path

import pytest

from ctikernels import _moe_block_fp8_cute_table as policy
from ctikernels.moe_block_fp8 import BlockFP8MoE

PRESETS = {name: c for name, c in policy.PRESETS.items() if c.path == "decode"}


def test_decode_policy_boundaries(monkeypatch):
    for m in (1, 2, 3, 4, 8):
        assert set(PRESETS) <= set(policy.valid_presets(m))
    assert not set(PRESETS).intersection(policy.valid_presets(9))
    # Exercise a future measured table entry without changing production policy.
    monkeypatch.setitem(policy.TABLE, (12, 0, 170), ((8, "cute-fp8", "decode-n16"),))
    op = object.__new__(BlockFP8MoE)
    op.device_class = (12, 0, 170)
    op.provider, op._available, op._enabled = "cute-fp8", True, True
    op._preset, op.top_k = None, 8
    assert op.config_for(8).path == "decode"
    op._preset = "decode-n16"
    with pytest.raises(ValueError, match="at most 8"):
        op.config_for(9)
    op.top_k = 9
    with pytest.raises(ValueError, match="top_k"):
        op.config_for(1)


@pytest.mark.parametrize(
    "change",
    [
        {"bm": 16},
        {"bni": 64},
        {"bn": 128},
        {"stages": 1},
        {"ctas_per_sm": 1},
        {"a_source": "fp8"},
        {"swap_ab": False},
    ],
)
def test_invalid_decode_config(change):
    with pytest.raises(ValueError):
        replace(PRESETS["decode-n16"], **change)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("fc1,external", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("shape", [(256, 2048, 512, 8), (3, 256, 128, 2)])
def test_offline_decode_compilation(preset, fc1, external, shape, tmp_path, monkeypatch):
    import importlib.metadata

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120a")
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass.cute")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        pytest.skip("requires DSL 4.8.0")
    from ctikernels._moe_block_fp8_cute_decode import compile_kernel

    fn = compile_kernel(
        PRESETS[preset],
        *shape,
        fc1=fc1,
        external_quant=external,
        options=f"--keep-ptx --dump-dir={tmp_path}",
    )
    ptx = fn.__ptx__
    # DSL cache versions expose either text or an artifact pathname.
    if "\n" not in ptx:
        ptx = Path(ptx).read_text()
    assert ".target sm_120a" in ptx
    assert "mma.sync.aligned" in ptx and ".m16n8k32." in ptx
    assert ".e4m3.e4m3.f32.ue8m0" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x2.shared.b16" in ptx
    assert "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in ptx
    assert "cp.async.bulk.tensor" in ptx
    assert "atom." not in ptx and "red.global" not in ptx
    if PRESETS[preset].use_pdl:
        assert ("griddepcontrol.launch_dependents" if fc1 else "griddepcontrol.wait") in ptx
    else:
        assert "griddepcontrol" not in ptx
