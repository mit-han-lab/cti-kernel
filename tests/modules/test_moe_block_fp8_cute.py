# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU metadata and dispatch tests; no CuTe import or CUDA initialization needed."""

import sys
from dataclasses import replace

import pytest
import torch

from ctikernels import _moe_block_fp8_cute_table as table
from ctikernels.moe_block_fp8 import (
    MOE_BLOCK_FP8_PROVIDERS,
    BlockFP8MoE,
    _check_inputs,
    _check_weights,
    _select_provider,
)

from ..support.child import run_child


def weights():
    return (
        torch.empty((2, 256, 256), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2, 2)),
        torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn),
        torch.ones((2, 2, 1)),
    )


def test_lazy_import():
    result = run_child(
        [
            sys.executable,
            "-c",
            "import sys; import ctikernels.moe_block_fp8; "
            "assert not any(n in sys.modules for n in ('cutlass', 'triton', 'sgl_kernel')); "
            "import torch; assert not torch.cuda.is_initialized()",
        ],
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_table_and_presets():
    for device, entries in table.TABLE.items():
        assert device[:2] in ((12, 0), (12, 1)) and device[2] > 0
        assert [row[0] for row in entries] == sorted({row[0] for row in entries})
        for limit, provider, name in entries:
            assert provider in MOE_BLOCK_FP8_PROVIDERS and name in table.PRESETS
            assert table.selection(device, limit) == (provider, name)
    for config in table.PRESETS.values():
        assert config.path in ("grouped", "decode")
        assert max(config.smem_bytes(True), config.smem_bytes(False)) <= 99 * 1024
        if config.path == "grouped":
            assert config.a_source == "fp8"
            if config.ctas_per_sm == 2:
                assert 2 * max(config.smem_bytes(True), config.smem_bytes(False)) <= 99 * 1024
    assert table.selection((12, 0, 1), 4) == ("triton-fp8", None)
    assert table.selection((12, 1, 48), 4097) == ("triton-fp8", None)


@pytest.mark.parametrize(
    "change",
    [
        {"bm": 8},
        {"bni": 16},
        {"bn": 32},
        {"stages": 1},
        {"warps": 3},
        {"a_source": "bf16"},
        {"gather_warps": 3},
        {"path": "decode"},
        {"ctas_per_sm": 0},
        {"bm": 128},
        {"bm": 128, "bni": 128, "bn": 256, "swap_ab": False, "stages": 4},
    ],
)
def test_bad_presets(change):
    with pytest.raises(ValueError):
        replace(table.Config(), **change)


@pytest.mark.parametrize("device", [(12, 0, 170), (12, 1, 48), (9, 0, 132), (12, 0, 1)])
@pytest.mark.parametrize("available,enabled", [(True, True), (False, True), (True, False)])
def test_provider_selection(device, available, enabled):
    expected = "cute-fp8" if available and enabled and device in table.TABLE else "triton-fp8"
    assert _select_provider(None, device, 4, available, enabled) == expected
    assert _select_provider("triton-fp8", device, 4, available, enabled) == "triton-fp8"
    if available and enabled and device[:2] in ((12, 0), (12, 1)):
        assert _select_provider("cute-fp8", device, 4, available, enabled) == "cute-fp8"
    else:
        with pytest.raises(RuntimeError, match="sm120/121"):
            _select_provider("cute-fp8", device, 4, available, enabled)


def test_cpu_weights_fail_after_metadata_validation():
    assert _check_weights(*weights(), 8) == (2, 256, 128)
    with pytest.raises(ValueError, match="CUDA weights"):
        BlockFP8MoE(*weights())
    with pytest.raises(ValueError, match="provider must"):
        BlockFP8MoE(*weights(), provider="other")


@pytest.mark.parametrize(
    "slot,change,error,match",
    [
        (0, lambda t: t.float(), TypeError, "float8"),
        (1, lambda t: t.half(), TypeError, "float32"),
        (1, lambda t: t[:, :, :1], ValueError, "shape"),
        (2, lambda t: t[0], ValueError, "rank 3"),
        (2, lambda t: t[:, :128], ValueError, "contiguous|shape"),
        (3, lambda t: t.requires_grad_(), ValueError, "inference only"),
        (0, lambda t: t[:, :254].contiguous(), ValueError, "multiples of 128"),
        (0, lambda t: t.transpose(1, 2), ValueError, "contiguous"),
    ],
)
def test_weight_validation(slot, change, error, match):
    args = list(weights())
    args[slot] = change(args[slot])
    with pytest.raises(error, match=match):
        _check_weights(*args, 8)


@pytest.mark.parametrize(
    "slot,change,error,match",
    [
        (0, lambda t: t.float(), TypeError, "FP32 hidden dtype gap"),
        (0, lambda t: t[:, :128].contiguous(), ValueError, "x must"),
        (0, lambda t: t.requires_grad_(), ValueError, "inference only"),
        (1, lambda t: t[:, :4].contiguous(), ValueError, "shape"),
        (1, lambda t: t.double(), TypeError, "float32"),
        (2, lambda t: t.long(), TypeError, "int32"),
        (2, lambda t: t.T, ValueError, "shape"),
    ],
)
def test_input_validation(slot, change, error, match):
    args = [
        torch.empty(2, 256, dtype=torch.bfloat16),
        torch.ones(2, 8),
        torch.zeros(2, 8, dtype=torch.int32),
    ]
    args[slot] = change(args[slot])
    with pytest.raises(error, match=match):
        _check_inputs(*args, device=torch.device("cpu"), hidden=256, top_k=8)


@pytest.mark.parametrize("m", [-1, 1.5, "1"])
def test_invalid_batch_size(m):
    with pytest.raises(ValueError, match="nonnegative integer"):
        table.selection((12, 0, 170), m)


@pytest.mark.parametrize(
    "preset", [name for name, c in table.PRESETS.items() if c.path == "grouped"]
)
@pytest.mark.parametrize("fc1", [True, False])
def test_async_pipeline_ptx(preset, fc1, tmp_path, monkeypatch):
    """Compile only: catch scalar tiles accidentally repeated by a 128-bit CopyAtom.

    The original (1,16) tile emitted cp.async copies at byte offsets +0..+15,
    faulting on GPU. Exercise the real kernel and inspect public generated PTX;
    never load a CUDA module or invoke the compiled callable.
    """
    import importlib.metadata
    import re

    pytest.importorskip("cutlass.cute")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        pytest.skip("compile regression requires CuTe DSL 4.8.0")
    import cutlass.cute as cute

    from ctikernels._moe_block_fp8_cute import compile_kernel

    original = cute.compile

    def with_ptx(*args, **kwargs):
        kwargs["options"] += f" --keep-ptx --dump-dir={tmp_path}"
        return original(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", with_ptx)
    fn = compile_kernel(table.PRESETS[preset], 188, 256, 2048, 512, 8, fc1=fc1)
    ptx = fn.__ptx__
    assert "cp.async.wait_group" not in ptx
    assert len(re.findall(r"bar.sync\s+0;", ptx)) == 1  # initialization only
    assert "st.global.v2.b16" in ptx  # BF16x2 epilogue, including swap-AB
    if fc1:
        copies = [line.strip() for line in ptx.splitlines() if "cp.async.ca.shared.global" in line]
        assert len(copies) == 1, copies
        assert re.search(r", 16, %r[0-9]+;", copies[0]), copies
        assert re.search(r"selp.b32\s+%r[0-9]+, 16, 0,", ptx)  # padding zero-fill
        assert all(int(offset) % 16 == 0 for offset in re.findall(r"\+(\d+)\]", copies[0])), copies
        assert "cp.async.mbarrier.arrive.noinc.shared::cta.b64" in ptx
    else:
        assert "ld.global.v4.b32" in ptx  # eight BF16 h values per lane


def test_group128_sliced_b_copy_coordinates():
    """Check the real copy builder's coordinates at compile time, with no GPU.

    The faulty full-width copy had N repetitions hidden inside its value mode.
    Indexing its extent-one repetition mode and retiling a single-N fragment
    silently addressed the wrong columns and exceeded that fragment's extent.
    """
    import importlib.metadata

    pytest.importorskip("cutlass.cute")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        pytest.skip("layout regression requires CuTe DSL 4.8.0")
    import cutlass.cute as cute
    from cutlass import Float8E4M3FN, Float8E8M0FNU, Float32
    from cutlass.cute.nvgpu import warp

    from ctikernels._moe_block_fp8_cute import _make_b_copy

    def check_static_coordinates():
        mma = cute.make_tiled_mma(
            warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU),
            cute.make_layout((4, 2, 1)),
            permutation_mnk=(128, 128, 32),
        )
        whole = _make_b_copy(mma, 128, 8, False)
        sliced = _make_b_copy(mma, 128, 8, True)
        identity = cute.make_identity_tensor((128, 128))
        # B is replicated across the four M warps. Check every lane of both
        # distinct N warps, all four N pairs and all K slices. Both weight
        # halves use this same copy layout with different base addresses.
        for tid in (*range(32), *range(128, 160)):
            full_source = whole.get_slice(tid).partition_S(identity)
            copy_thread = sliced.get_slice(tid)
            source = copy_thread.partition_S(identity)
            fragment = mma.get_slice(tid).partition_B(identity)
            full_target = copy_thread.retile(fragment)
            assert cute.size(source, mode=[0]) == 16
            assert cute.size(source, mode=[1]) == 4
            for n in range(4):
                pair = cute.composition(
                    cute.domain_offset((0, 2 * n, 0), fragment), (None, cute.make_layout(2), None)
                )
                target = copy_thread.retile(pair)
                assert cute.size(target, mode=[1]) == 1
                assert cute.size(target) == cute.size(pair)
                for k in range(4):
                    for v in range(16):
                        assert source[v, n, k] == full_source[v + 16 * n, 0, k]
                        assert target[v, 0, k] == full_target[v, n, k]

    @cute.jit
    def check():
        check_static_coordinates()

    cute.compile(check, options="--gpu-arch sm_120a")
