# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""NVFP4 layout helpers, the frozen-weight and device contract, provider resolution and the
dispatch tables, on CPU."""

from __future__ import annotations

import pytest
import torch

from ctikernels.linear import _nvfp4_cute_quant, _nvfp4_dispatch, _nvfp4_quack
from ctikernels.linear.nvfp4 import (
    NVFP4GateUpSiLU,
    NVFP4Linear,
    NVFP4Weight,
    blocked_scale,
    dequantize_nvfp4,
    quantize_nvfp4,
    scale_shape,
    unblocked_scale,
)

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _weight(rows: int = 64, cols: int = 128, device: str = "cpu") -> NVFP4Weight:
    data = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device=device)
    scale = torch.rand(rows, cols // 16, device=device).to(torch.float8_e4m3fn)
    return NVFP4Weight(data, scale, torch.ones(1, device=device))


def test_every_byte_decodes_to_its_two_e2m1_values() -> None:
    data = torch.arange(256, dtype=torch.uint8).view(1, 256)
    values = dequantize_nvfp4(data, torch.ones(1, 32, dtype=torch.float8_e4m3fn))
    table = E2M1 + tuple(-v for v in E2M1)
    expected = [table[code] for byte in range(256) for code in (byte & 15, byte >> 4)]
    assert values.view(-1).tolist() == expected
    scaled = dequantize_nvfp4(
        data, torch.full((1, 32), 2.0).to(torch.float8_e4m3fn), torch.tensor([0.25])
    )
    torch.testing.assert_close(scaled, values * 0.5, rtol=0, atol=0)


@pytest.mark.parametrize("rows, cols", [(1, 32), (127, 64), (128, 96), (300, 2048)])
def test_blocked_scale_places_each_scale_and_pads_with_zeros(rows: int, cols: int) -> None:
    grid = torch.randint(1, 127, (rows, cols // 16), dtype=torch.uint8)
    blocked = blocked_scale(grid)
    assert tuple(blocked.shape) == scale_shape(rows, cols)
    for r, b in [(0, 0), (rows - 1, cols // 16 - 1), (rows // 2, cols // 32)]:
        assert blocked[r // 128, b // 4, r % 32, r // 32 % 4, b % 4] == grid[r, b]
    assert torch.equal(unblocked_scale(blocked, rows, cols), grid)
    assert int(blocked.sum()) == int(grid.sum())  # every padding byte is zero


def test_frozen_weights_raise_before_device_checks() -> None:
    w = _weight()
    scale = torch.ones(1)
    frozen = torch.ones(1, requires_grad=True)
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(NVFP4Weight(w.data, w.scale, frozen), scale)
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(w, frozen)
    float_scale = w.scale.float().requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(NVFP4Weight(w.data, float_scale, w.global_scale), scale)
    with pytest.raises(ValueError, match="frozen"):
        NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=frozen)
    with pytest.raises(ValueError, match="frozen"):
        NVFP4GateUpSiLU(
            w, NVFP4Weight(w.data, w.scale, frozen), input_scale=scale, output_scale=scale
        )


def test_operations_need_cuda_and_valid_shapes() -> None:
    w = _weight()
    scale = torch.ones(1)
    with pytest.raises(ValueError, match="CUDA"):
        NVFP4Linear(w, scale)
    with pytest.raises(ValueError, match="CUDA"):
        quantize_nvfp4(torch.zeros(2, 64, dtype=torch.bfloat16), scale)
    with pytest.raises(ValueError, match="multiple of 32"):
        NVFP4Linear(NVFP4Weight(w.data[:, :8], w.scale[:, :1], scale), scale)
    with pytest.raises(ValueError):
        NVFP4Linear(NVFP4Weight(w.data, w.scale[:, :4], scale), scale)
    with pytest.raises(TypeError):
        NVFP4Linear(NVFP4Weight(w.data.view(torch.int8), w.scale, scale), scale)
    with pytest.raises(TypeError):
        NVFP4Linear((w.data, w.scale), scale)


@pytest.fixture
def linear_on_cpu(monkeypatch):
    from ctikernels.linear import nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Test GPU", (12, 0)))
    return nvfp4


@pytest.mark.parametrize(
    "globals_,slot_rows,error,match",
    [
        (torch.ones(2), None, ValueError, "slot_rows is required"),
        (torch.ones(2), (64,), ValueError, "2 entries"),
        (torch.ones(2), (0, 64), ValueError, "positive ints"),
        (torch.ones(2), (-1, 65), ValueError, "positive ints"),
        (torch.ones(2), (32, 31), ValueError, "sum to N"),
        (torch.ones(1), (63,), ValueError, "sum to N"),
        (torch.ones(2), (32.0, 32), TypeError, "positive ints"),
        (torch.ones(2), (True, 63), TypeError, "positive ints"),
        (torch.ones(2), 64, TypeError, "sequence"),
        (torch.ones(2, dtype=torch.float16), (32, 32), TypeError, "float32"),
        (torch.ones(2, device="meta"), (32, 32), ValueError, "must be on"),
        (torch.ones(2, 1), (32, 32), ValueError, "1-D"),
        (torch.empty(0), (), ValueError, "nonempty"),
        ([0.7, 0.9], (32, 32), TypeError, "tensor"),
        (torch.ones(2, requires_grad=True), (32, 32), ValueError, "frozen"),
    ],
)
def test_linear_slot_validation_before_provider_load(
    monkeypatch, linear_on_cpu, globals_, slot_rows, error, match
):
    fail = lambda *args, **kwargs: pytest.fail("provider load before validation")
    monkeypatch.setattr(linear_on_cpu, "_resolve", fail)
    monkeypatch.setattr(linear_on_cpu, "_resolve_providers", fail)
    with pytest.raises(error, match=f"NVFP4Linear:.*{match}"):
        NVFP4Linear(_weight()._replace(global_scale=globals_), torch.ones(1), slot_rows=slot_rows)


@pytest.mark.parametrize(
    "provider,reason", [("flashinfer", "one scalar alpha"), ("cute", "must be one of")]
)
def test_linear_slot_forced_scalar_provider_rejected(monkeypatch, linear_on_cpu, provider, reason):
    fail = lambda *args, **kwargs: pytest.fail("provider load before rejection")
    monkeypatch.setattr(linear_on_cpu, "_resolve", fail)
    monkeypatch.setattr(linear_on_cpu, "_resolve_providers", fail)
    w = _weight()._replace(global_scale=torch.tensor([0.7, 0.9]))
    with pytest.raises(ValueError, match=f"NVFP4Linear:.*{reason}"):
        NVFP4Linear(w, torch.ones(1), slot_rows=(32, 32), provider=provider)


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("provider", [None, "quack", "dequant"])
def test_linear_slot_plan_mapping(monkeypatch, linear_on_cpu, available, provider):
    from ctikernels.linear import _nvfp4_cute_small, _nvfp4_flashinfer

    fail = lambda *args, **kwargs: pytest.fail("scalar provider loaded for independent slots")
    monkeypatch.setattr(_nvfp4_flashinfer, "load", fail)
    monkeypatch.setattr(_nvfp4_cute_small, "available", fail)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object() if available else None)
    w = _weight()._replace(global_scale=torch.tensor([0.7, 0.9]))
    if provider == "quack" and not available:
        with pytest.raises(RuntimeError, match="provider 'quack'"):
            NVFP4Linear(w, torch.ones(1), slot_rows=(32, 32), provider=provider)
        return
    op = NVFP4Linear(w, torch.ones(1), slot_rows=(32, 32), provider=provider)
    quack = available and provider != "dequant"
    assert op.provider == ("quack" if quack else "dequant")
    assert op._flash_weight is None
    assert op.plan_for(16) == (_nvfp4_quack.DEFAULT if quack else None)


def test_linear_uniform_slots_keep_scalar_plans(monkeypatch, linear_on_cpu):
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_flashinfer

    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object())
    monkeypatch.setattr(
        _nvfp4_flashinfer, "load", lambda _: SimpleNamespace(prepare=lambda *args: args)
    )
    g = torch.tensor([0.7])
    w = _weight()
    ops = [
        NVFP4Linear(w._replace(global_scale=global_), torch.ones(1), slot_rows=slots)
        for global_, slots in ((g.reshape(()), None), (g, (64,)), (g.expand(3), (16, 16, 32)))
    ]
    for op in ops:
        assert op._weights[0].global_scale.numel() == 1
        assert op._weights[0].global_scale.data_ptr() == g.data_ptr()
        assert op._slot_index is None
        assert op.provider == ops[0].provider
        assert op.plan_for(16) == ops[0].plan_for(16)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_linear_slot_gather_precedes_quantizer_and_reads_updates(
    monkeypatch, linear_on_cpu, quantized, strided
):
    from types import SimpleNamespace

    from ctikernels.linear.nvfp4 import NVFP4Activation

    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object())
    g = torch.tensor([0.7, -1.0, 0.9, -1.0])[::2] if strided else torch.tensor([0.7, 0.9])
    op = NVFP4Linear(_weight()._replace(global_scale=g), torch.ones(1), slot_rows=(17, 47))
    assert op._slot_index.tolist() == [0] * 17 + [1] * 47
    events = []
    gather = torch.index_select

    def index_select(*args, **kwargs):
        events.append("gather")
        return gather(*args, **kwargs)

    def quantize(*args, **kwargs):
        events.append("quantize")
        assert kwargs["column_scale"] is None and kwargs["alpha"] is None
        return "mock"

    def multiply(*args):
        events.append("gemm")
        args[6].copy_(args[4])

    monkeypatch.setattr(torch, "index_select", index_select)
    monkeypatch.setattr(linear_on_cpu, "_quantize_into", quantize)
    monkeypatch.setattr(op, "_launcher", SimpleNamespace(linear=multiply))
    a = NVFP4Activation(
        torch.zeros(2, 64, dtype=torch.uint8),
        torch.zeros(scale_shape(2, 128), dtype=torch.float8_e4m3fn),
        torch.ones(1),
    )
    for _ in range(2):
        events.clear()
        out = op.quantized(a) if quantized else op(torch.ones(2, 128, dtype=torch.bfloat16))
        assert events == (["gather", "gemm"] if quantized else ["gather", "quantize", "gemm"])
        expected = torch.cat((g[:1].expand(17), g[1:].expand(47))).bfloat16()
        torch.testing.assert_close(out, expected.expand(2, -1), rtol=0, atol=0)
        g.mul_(torch.tensor([2.0, 0.5]))


def test_quack_resolves_only_for_cuda_devices_it_is_enabled_on(monkeypatch) -> None:
    assert _nvfp4_quack._resolve(torch.device("cpu")) is None
    monkeypatch.setenv(_nvfp4_quack.VARIABLE, "0")
    assert _nvfp4_quack._resolve(torch.device("cuda", 0)) is None


def _valid(config: object) -> bool:
    if isinstance(config, _nvfp4_dispatch.Cute):
        config.config()
        return True
    if isinstance(config, _nvfp4_dispatch.FlashInfer):
        return True
    if isinstance(config, _nvfp4_dispatch.Small):
        return config.splits >= 1 and config.block_n % 32 == 0 and config.block_k % 32 == 0
    return (
        isinstance(config, _nvfp4_quack.Config)
        and config.tile_m in (128, 256)
        and config.tile_n in (128, 256)
    )


def test_gated_table_is_ordered_and_names_valid_plans() -> None:
    for device, shapes in _nvfp4_dispatch.GATED.items():
        assert device in _nvfp4_quack.DEVICES, device
        for shape, plan in shapes.items():
            bounds = [entry[0] for entry in plan]
            assert bounds == sorted(set(bounds)), (device, shape)
            assert all(bound & (bound - 1) == 0 for bound in bounds), (device, shape)
            for entry in plan:
                assert len(entry) == 3 and _valid(entry[-1]), (device, shape, entry)
                small = isinstance(entry[-1], _nvfp4_dispatch.Small)
                cute = isinstance(entry[-1], _nvfp4_dispatch.Cute)
                assert not small or entry[0] <= _nvfp4_dispatch.SMALL_ROWS
                assert not cute or entry[0] <= 32
                if cute:
                    entry[-1].config().validate(shape[0])
                if isinstance(entry[-1], _nvfp4_dispatch.FlashInfer):
                    assert entry[1] == _nvfp4_dispatch.UNFUSED
                paths = (_nvfp4_dispatch.FUSED, _nvfp4_dispatch.UNFUSED)
                if cute:
                    assert entry[1] == _nvfp4_dispatch.CUTE
                else:
                    assert entry[1] == _nvfp4_dispatch.TRITON if small else entry[1] in paths


def test_dispatch_buckets_take_the_next_power_of_two(monkeypatch) -> None:
    device = ("Test GPU", (12, 0))
    a = _nvfp4_quack.Config(128, 128, True, True)
    b = _nvfp4_quack.Config(128, 256, False, True)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    u = _nvfp4_dispatch.UNFUSED
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device, {(32, 64): ((4, u, a), (64, u, b))})
    gated = [_nvfp4_dispatch.gated_plan(None, 32, 64, rows) for rows in (1, 4, 5, 64, 999)]
    assert gated == [(u, a), (u, a), (u, b), (u, b), (u, b)]
    assert _nvfp4_dispatch.gated_plan(None, 64, 64, 3) == (
        _nvfp4_dispatch.FUSED,
        _nvfp4_quack.DEFAULT,
    )


def test_independent_gate_up_globals_skip_flashinfer_entries(monkeypatch) -> None:
    device = ("Test GPU", (12, 0))
    fused = _nvfp4_quack.Config(128, 128, True, True)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    entries = (
        (32, _nvfp4_dispatch.UNFUSED, _nvfp4_dispatch.FlashInfer()),
        (4096, _nvfp4_dispatch.FUSED, fused),
    )
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device, {(64, 32): entries})
    for rows in (1, 32):
        assert _nvfp4_dispatch.gated_plan(None, 64, 32, rows) == entries[0][1:]
        assert _nvfp4_dispatch.gated_plan(None, 64, 32, rows, shared_globals=False) == (
            _nvfp4_dispatch.FUSED,
            fused,
        )
    assert _nvfp4_dispatch.uses_flashinfer(None, (64, 32), gated=True)
    assert not _nvfp4_dispatch.uses_flashinfer(None, (64, 32), gated=True, shared_globals=False)


def test_small_configurations_fall_back_where_they_do_not_fit(monkeypatch) -> None:
    device = ("Test GPU", (12, 0))
    small = _nvfp4_dispatch.Small(2, 64, 128, 4, 3, False)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    monkeypatch.setitem(
        _nvfp4_dispatch.GATED, device, {(512, 32): ((64, _nvfp4_dispatch.TRITON, small),)}
    )
    assert _nvfp4_dispatch.gated_plan(None, 512, 32, 3) == (_nvfp4_dispatch.TRITON, small)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_flashinfer_dispatch_prepares_once_and_falls_back(monkeypatch, gated, available):
    from ctikernels.linear import _nvfp4_flashinfer, nvfp4

    device_type = ("Test GPU", (12, 0))
    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device_type)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object())
    loads, repacks = [], []
    launcher = _nvfp4_flashinfer.Launcher()
    original = launcher.prepare

    def prepare(*args):
        repacks.append(args)
        return original(*args)

    def load(device):
        loads.append(device)
        return launcher if available else None

    interleaved = launcher.prepare_interleaved

    def prepare_interleaved(*args):
        repacks.append(args)
        return interleaved(*args)

    monkeypatch.setattr(launcher, "prepare", prepare)
    monkeypatch.setattr(launcher, "prepare_interleaved", prepare_interleaved)
    monkeypatch.setattr(_nvfp4_flashinfer, "load", load)
    f = _nvfp4_dispatch.FlashInfer()
    q = _nvfp4_quack.DEFAULT
    assert _valid(f)
    w = _weight()
    table, shape = _nvfp4_dispatch.GATED, (128, 64)
    if gated:
        entries = ((1, _nvfp4_dispatch.UNFUSED, f), (4, _nvfp4_dispatch.FUSED, q))
        monkeypatch.setitem(table, device_type, {shape: entries})
    op = (
        NVFP4GateUpSiLU(w, w, input_scale=torch.ones(1), output_scale=torch.ones(1))
        if gated
        else NVFP4Linear(w, torch.ones(1))
    )
    assert op.provider == ("flashinfer" if available else "quack")
    expected = f if available else q
    assert op.plan_for(1) == ((_nvfp4_dispatch.UNFUSED, expected) if gated else expected)
    assert op.plan_for(3) == ((_nvfp4_dispatch.FUSED, q) if gated else expected)
    assert op.plan_for(1) == ((_nvfp4_dispatch.UNFUSED, expected) if gated else expected)
    assert len(loads) == 1
    assert len(repacks) == int(available)
    if gated and available:
        # Shared globals: FlashInfer reads views of the one interleaved copy.
        assert len(repacks[0]) == 4
        assert op._flash_weight[0][0].data_ptr() == op._weight[0].data_ptr()
        assert op._flash_weight[0][1].data_ptr() == op._weight[1].data_ptr()
    # Gated needs a table entry; Linear has no table.
    monkeypatch.setitem(table, device_type, {})
    other = (
        NVFP4GateUpSiLU(w, w, input_scale=torch.ones(1), output_scale=torch.ones(1))
        if gated
        else NVFP4Linear(w, torch.ones(1))
    )
    assert other.provider == ("flashinfer" if available and not gated else "quack")
    count = 1 if gated else 2
    assert len(loads) == count and len(repacks) == count * int(available)


def test_flashinfer_wrapper_uses_layout_views_and_fresh_device_alpha(monkeypatch):
    import sys
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_flashinfer
    from ctikernels.linear.nvfp4 import NVFP4Activation

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(mm_fp4=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    w = _weight(128, 256)
    a = NVFP4Activation(w.data, blocked_scale(w.scale), torch.tensor([0.25]))
    launcher = _nvfp4_flashinfer.Launcher()
    weight = launcher.prepare(*w)
    out = torch.empty(128, 128, dtype=torch.bfloat16)
    launcher.linear(a, weight, out)
    w.global_scale.mul_(2)
    a.global_scale.mul_(3)
    launcher.linear(a, weight, out)
    for args, kwargs in calls:
        assert args[0].data_ptr() == a.data.data_ptr()
        assert args[1].stride() == (1, 128)
        assert args[2].data_ptr() == a.scale.data_ptr()
        assert args[3].data_ptr() == weight[1].data_ptr()
        assert kwargs["backend"] == "cutlass" and kwargs["out"] is out
        assert kwargs["use_8x4_sf_layout"] is False
        assert kwargs["enable_pdl"] is True
    torch.testing.assert_close(calls[0][1]["alpha"], torch.tensor([0.25]))
    torch.testing.assert_close(calls[1][1]["alpha"], torch.tensor([1.5]))
    # Per-device PDL policy: launches without PDL from the configured row count.
    calls.clear()
    launcher = _nvfp4_flashinfer.Launcher(pdl_off_rows=128)
    launcher.linear(a, weight, out)
    small = NVFP4Activation(w.data[:64], blocked_scale(w.scale[:64]), a.global_scale)
    launcher.linear(small, weight, out[:64])
    assert [kwargs["enable_pdl"] for _, kwargs in calls] == [False, True]


def test_flashinfer_availability_is_cached_and_missing_import_falls_back(monkeypatch):
    import sys
    from contextlib import nullcontext

    from ctikernels.linear import _nvfp4_flashinfer as fi

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # The pinned version, whatever FlashInfer is installed: it does not qualify sm121.
    monkeypatch.setattr(fi.importlib.metadata, "version", lambda _: fi.VERSION)
    assert fi._resolve(torch.device("cpu")) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    assert fi._resolve(torch.device("cuda", 0)) is None
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    assert fi._resolve(torch.device("cuda", 0)) is None  # sm121 first-replay defect
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    monkeypatch.setattr(fi, "_launchers", {})
    with pytest.warns(RuntimeWarning, match="unavailable"):
        assert fi.load(torch.device("cuda", 0)) is None
    # A failed resolution is retained, even if import state changes afterwards.
    monkeypatch.setattr(fi, "_resolve", lambda _: pytest.fail("resolved twice"))
    assert fi.load(torch.device("cuda", 0)) is None


def test_flashinfer_frozen_weights_raise_before_resolving_or_repacking(monkeypatch):
    from ctikernels.linear import _nvfp4_flashinfer

    monkeypatch.setattr(_nvfp4_flashinfer, "load", lambda _: pytest.fail("launched too early"))
    w = _weight()
    bad = w._replace(global_scale=w.global_scale.requires_grad_())
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(bad, torch.ones(1), provider="flashinfer")
    with pytest.raises(ValueError, match="frozen"):
        NVFP4GateUpSiLU(
            w, bad, input_scale=torch.ones(1), output_scale=torch.ones(1), provider="flashinfer"
        )


@pytest.mark.parametrize(
    "version,capability,expected",
    [
        ("0.6.11.post3", (12, 0), True),
        ("0.6.11.post3", (12, 1), False),
        ("0.7.1", (12, 0), True),
        ("0.7.1", (12, 1), True),
        ("0.7.0", (12, 0), False),
        ("0.7.2", (12, 0), False),
        ("0.7.1", (9, 0), False),
    ],
)
def test_flashinfer_version_capability_gate(monkeypatch, version, capability, expected):
    import contextlib
    import sys
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_flashinfer as fi

    builds = []
    monkeypatch.setattr(fi.importlib.metadata, "version", lambda _: version)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_fp4=lambda: None))
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.gemm.gemm_base",
        SimpleNamespace(get_cutlass_fp4_gemm_module=lambda *cc: builds.append(cc)),
    )
    assert (fi._resolve(torch.device("cuda", 0)) is not None) == expected
    assert builds == ([capability] if expected else [])


@pytest.mark.parametrize("gated", [False, True])
def test_unqualified_flashinfer_dispatch_falls_back_without_build(monkeypatch, gated):
    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear import nvfp4

    device_type = ("Test GPU", (12, 0))
    monkeypatch.setattr(nvfp4, "_cuda_device", lambda op, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object())
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device_type)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(fi.importlib.metadata, "version", lambda _: "0.7.2")
    monkeypatch.setattr(fi, "load", lambda _: fi._resolve(torch.device("cuda", 0)))
    entry = (1, _nvfp4_dispatch.UNFUSED, _nvfp4_dispatch.FlashInfer())
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device_type, {(128, 64): (entry,)})
    w, scale = _weight(), torch.ones(1)
    op = (
        NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale)
        if gated
        else NVFP4Linear(w, scale)
    )
    assert op.plan_for(1) == (
        (_nvfp4_dispatch.UNFUSED, _nvfp4_quack.DEFAULT) if gated else _nvfp4_quack.DEFAULT
    )
    with pytest.raises(RuntimeError, match="qualified"):
        if gated:
            NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale, provider="flashinfer")
        else:
            NVFP4Linear(w, scale, provider="flashinfer")


def _parity_module():
    # The required CPU invocation sets PYTHONPATH=python, not the repository root.
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmark.kernel_tuning import nvfp4_parity

    return nvfp4_parity


def test_parity_statistics_use_paired_ratios_and_interpolated_noise():
    parity = _parity_module()
    result = parity.parity_statistics([1, 10, 100], [2, 30, 100], [100] * 20, [100] * 20)
    assert result["speedup"] == 2  # ratio of medians would be 3
    assert result["eps"] == 0 and result["verdict"] == "pass"
    result = parity.parity_statistics([100] * 20, [98] * 20, [100] * 19 + [102], [100] * 20)
    assert result["eps"] == pytest.approx(0.001)
    assert result["verdict"] == "fail"
    assert parity.parity_statistics([100], [99], [101], [100])["verdict"] == "pass"
    assert parity.parity_statistics([100], [200], [103], [100])["verdict"] == "inconclusive"
    for args in (([], [], [1], [1]), ([1], [1, 2], [1], [1]), ([0], [1], [1], [1])):
        with pytest.raises(ValueError):
            parity.parity_statistics(*args)
    assert len(parity.WORKLOAD) == 98
    assert parity.select_rows("gated/27b-lm/1") == []
    assert parity.select_rows("linear/9b/1,8") == [("linear", "9b", 1), ("linear", "9b", 8)]


def test_parity_samples_median_individually_timed_replays(monkeypatch):
    parity = _parity_module()
    elapsed = iter([7.65, 8.48, 7.65, 7.65, 8.48])
    calls = []

    def timing(graph, replays):
        calls.append((graph, replays))
        return next(elapsed)

    monkeypatch.setattr(parity, "_elapsed_ms", timing)
    assert parity._sample("graph", 1000, 5) == pytest.approx(7.65)
    assert calls == [("graph", 1)] * 5
    assert parity.graph_cycles(0.02) == 50
    assert parity.graph_cycles(43) == 1


def test_parity_pairs_alternate_and_candidates_rotate(monkeypatch):
    parity = _parity_module()
    order = []
    monkeypatch.setattr(parity, "_sample", lambda graph, *args: order.append(graph) or 1)
    assert parity._pairs([("ours", 2), ("sglang", 4)], 4, 5) == ([1] * 4, [1] * 4)
    assert order == ["ours", "sglang", "sglang", "ours"] * 2
    order.clear()
    assert parity._round_robin([("a", 2), ("b", 4), ("sglang", 2)], 3, 5) == [[1] * 3] * 3
    assert order == ["a", "b", "sglang", "b", "sglang", "a", "sglang", "a", "b"]


def test_parity_candidate_plans_are_owned_qualified_and_bounded():
    parity = _parity_module()
    device = ("NVIDIA GB10", (12, 1))
    current = _nvfp4_dispatch.Small(4, 64, 128, 4, 2, False)
    plans = parity.candidate_plans("linear", "35b-shared", 8, current, device, "0.7.1")
    assert sum(p[3] for p in plans) == 1
    assert plans[0] == ("triton", None, current, True)
    assert sum(p[0] == "triton" for p in plans) <= 8
    assert any(p[0] == "flashinfer" for p in plans)
    assert all(p[0] != "sglang" for p in plans)
    plans = parity.candidate_plans(
        "gated", "9b", 128, (_nvfp4_dispatch.FUSED, _nvfp4_quack.DEFAULT), device, "unqualified"
    )
    assert all(p[0] == "quack" for p in plans)
    assert {p[1] for p in plans} == {_nvfp4_dispatch.FUSED, _nvfp4_dispatch.UNFUSED}


def test_candidates_remeasure_all_plans_when_reference_is_noisy(monkeypatch):
    parity = _parity_module()
    counts = []

    def sample(graphs, pairs, replays, deadline):
        counts.append((pairs, replays))
        return [[7] * pairs, [10] * pairs, [11 if pairs == 20 else 10] * pairs]

    monkeypatch.setattr(parity, "_round_robin", sample)
    times, noise = parity._measure_candidates(["ours", "sglang", "control"], 20, 5)
    assert counts == [(20, 5), (60, 5)]
    assert all(len(samples) == 60 for samples in times)
    assert noise["remeasured"] and noise["initial_eps"] > 0.02 and noise["eps"] == 0


def test_candidate_tables_keep_current_within_two_percent(tmp_path, capsys):
    import json

    _parity_module()
    from benchmark.kernel_tuning.nvfp4_dispatch import emit_candidates

    records = []
    for m in (1, 2, 4):
        for provider, config, us, current in (
            ("quack", [128, 128, True, True], 10.19 if m < 4 else 10.21, True),
            ("flashinfer", [], 10, False),
            ("sglang", [], 1, False),
        ):
            records.append(
                {
                    "device": "NVIDIA GB10",
                    "capability": [12, 1],
                    "op": "gated",
                    "H": 4096,
                    "I": 12288,
                    "M": m,
                    "provider": provider,
                    "path": "quack-fused" if provider == "quack" else "quack-unfused",
                    "config": config,
                    "candidate_id": provider,
                    "current": current,
                    "median_us": us,
                    "samples_us": [us],
                    "sglang_us": 1,
                }
            )
    records.append(
        {
            "device": "NVIDIA GB10",
            "capability": [12, 1],
            "op": "gated",
            "H": 5120,
            "I": 17408,
            "M": 1,
            "provider": "cute",
            "path": "cute",
            "config": {"n_tile": 64, "split_k": 8},
            "candidate_id": "cute",
            "current": True,
            "median_us": 5,
            "samples_us": [5],
            "sglang_us": 1,
        }
    )
    log = tmp_path / "candidates.log"
    log.write_text("\n".join("CAND " + json.dumps(r) for r in records))
    emit_candidates([str(log)])
    output = capsys.readouterr().out
    assert "(4096, 12288): ((2, FUSED, _Q(128, 128, True, True)), (4, UNFUSED, _F()))" in output
    assert "(5120, 17408): ((1, 'cute', Cute(n_tile=64," in output
    assert "reduce_mode='global'" in output
    assert "sglang" not in output


@pytest.mark.parametrize(
    "gated,provider",
    [(False, "flashinfer"), (True, "flashinfer"), (False, "quack"), (True, "quack")],
)
def test_bf16_calls_fold_scalar_preparation_into_quantization(monkeypatch, gated, provider):
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear import nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Test GPU", (12, 0)))
    events = []
    launcher = SimpleNamespace(
        linear=lambda *a: events.append("gemm"), gated=lambda *a: events.append("gemm")
    )
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: launcher)
    flash = fi.Launcher()
    monkeypatch.setattr(fi, "load", lambda _: flash)
    monkeypatch.setattr(flash, "linear", lambda *a, **kw: events.append("gemm"))
    w, scale = _weight(), torch.ones(1)
    op = (
        NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale, provider=provider)
        if gated
        else NVFP4Linear(w, scale, provider=provider)
    )
    if provider == "quack":
        op.plan_for = (
            (lambda _: (_nvfp4_dispatch.FUSED, _nvfp4_quack.DEFAULT))
            if gated
            else (lambda _: _nvfp4_quack.DEFAULT)
        )

    def quantize(x, global_scale, out, *, gated, **kw):
        events.append("silu-quant" if gated else "input-quant")
        if not gated:
            assert kw["alpha"] is (op._alpha if provider == "flashinfer" else None)
            if provider == "quack":
                assert kw["column_scale"] is op._column_scale
                if hasattr(op, "_reciprocal"):
                    assert kw["reciprocal"] is op._reciprocal
        return "cute"

    monkeypatch.setattr(nvfp4, "_quantize_into", quantize)
    op(torch.zeros(2, 128, dtype=torch.bfloat16))
    assert op.quantizer == "cute"
    assert events == ["input-quant", "gemm"] + (
        ["silu-quant"] if gated and provider == "flashinfer" else []
    )


@pytest.mark.parametrize("gated", [False, True])
def test_weight_becoming_trainable_raises_before_call_launch(monkeypatch, gated):
    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear import nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda op, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Test GPU", (12, 0)))
    monkeypatch.setattr(fi, "load", lambda _: fi.Launcher())
    w, scale = _weight(), torch.ones(1)
    op = (
        NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale, provider="flashinfer")
        if gated
        else NVFP4Linear(w, scale, provider="flashinfer")
    )
    monkeypatch.setattr(
        nvfp4, "_quantize_into", lambda *a, **kw: pytest.fail("launched before validation")
    )
    w.global_scale.requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        op(torch.zeros(1, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="frozen"):
        op.quantized(None)


@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (8, 9), (9, 0)])
@pytest.mark.parametrize("version", ["4.8.0", "4.6.3", "4.8.1", None])
def test_cute_quantizer_version_and_capability(monkeypatch, capability, version):
    from ctikernels.linear import _nvfp4_cute_quant as quant

    def installed(_):
        if version is None:
            raise quant.importlib.metadata.PackageNotFoundError
        return version

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(quant.importlib.metadata, "version", installed)
    assert not quant.available(torch.device("cpu"))
    assert quant.available(torch.device("cuda", 0)) == (
        capability in ((12, 0), (12, 1)) and version == "4.8.0"
    )


@pytest.mark.parametrize("available", [False, True])
def test_quantizer_override_and_resolution_cache(monkeypatch, available):
    from ctikernels.linear import _nvfp4_cute_quant as quant

    device = torch.device("cuda", 0)
    resolved, launcher = [], object()
    monkeypatch.setattr(quant, "_launchers", {})
    monkeypatch.setattr(
        quant, "_resolve", lambda d: resolved.append(d) or (launcher if available else None)
    )
    monkeypatch.setenv(quant.VARIABLE, "triton")
    assert quant.load(device) is None and not resolved
    monkeypatch.delenv(quant.VARIABLE)
    assert quant.load(device) is (launcher if available else None)
    assert resolved == [device]
    monkeypatch.setenv(quant.VARIABLE, "cute")
    for _ in range(2):
        if available:
            assert quant.load(device) is launcher
        else:
            with pytest.raises(RuntimeError, match="DSL 4.8.0"):
                quant.load(device)
    assert resolved == [device]
    monkeypatch.setenv(quant.VARIABLE, "triton")
    assert quant.load(device) is None
    monkeypatch.delenv(quant.VARIABLE)
    assert quant.load(device) is (launcher if available else None)
    monkeypatch.setenv(quant.VARIABLE, "")
    assert quant.load(device) is (launcher if available else None)
    assert resolved == [device]
    monkeypatch.setenv(quant.VARIABLE, "unknown")
    with pytest.raises(ValueError, match="must be"):
        quant.load(device)


@pytest.mark.parametrize("entry", ["public", "linear", "gated"])
@pytest.mark.parametrize("provider", ["flashinfer", "quack"])
@pytest.mark.parametrize("available", [False, True])
def test_quantizer_selection_is_live_at_every_entry(monkeypatch, entry, provider, available):
    pytest.importorskip("triton")
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_quant as quant
    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear import _nvfp4_triton, nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Test GPU", (12, 0)))
    events = []
    launcher = SimpleNamespace(quantize=lambda *a, **kw: events.append(("cute", kw["gated"])))
    monkeypatch.setattr(quant, "_launchers", {})
    monkeypatch.setattr(quant, "_resolve", lambda _: launcher if available else None)
    monkeypatch.setattr(
        _nvfp4_triton, "quantize", lambda *a, **kw: events.append(("triton", kw["gated"]))
    )
    gemm = SimpleNamespace(linear=lambda *a, **kw: None, gated=lambda *a, **kw: None)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: gemm)
    flash = fi.Launcher()
    monkeypatch.setattr(fi, "load", lambda _: flash)
    monkeypatch.setattr(flash, "linear", lambda *a, **kw: None)
    w, scale = _weight(), torch.ones(1)
    if entry == "public":
        call = lambda x: quantize_nvfp4(x, scale)
    else:
        call = (
            NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale, provider=provider)
            if entry == "gated"
            else NVFP4Linear(w, scale, provider=provider)
        )
        if provider == "quack":
            call.plan_for = (
                (lambda _: (_nvfp4_dispatch.UNFUSED, _nvfp4_quack.DEFAULT))
                if entry == "gated"
                else (lambda _: _nvfp4_quack.DEFAULT)
            )
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    for choice in (None, "cute", "triton", None, "", "cute"):
        if choice is None:
            monkeypatch.delenv(quant.VARIABLE, raising=False)
        else:
            monkeypatch.setenv(quant.VARIABLE, choice)
        events.clear()
        if choice == "cute" and not available:
            with pytest.raises(RuntimeError, match="DSL 4.8.0"):
                call(x)
            assert not events
            continue
        call(x)
        expected = "cute" if available and choice != "triton" else "triton"
        assert events == [(expected, False)] + ([(expected, True)] if entry == "gated" else [])
        if entry != "public":
            assert call.quantizer == expected


@pytest.mark.parametrize("cute", [False, True])
def test_quantize_into_routes_all_scalar_arguments(monkeypatch, cute):
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_quant, nvfp4

    events = []
    launcher = SimpleNamespace(quantize=lambda *a, **kw: events.append(("cute", a, kw)))
    monkeypatch.setattr(_nvfp4_cute_quant, "load", lambda _: launcher if cute else None)
    if not cute:
        pytest.importorskip("triton")
        from ctikernels.linear import _nvfp4_triton

        monkeypatch.setattr(
            _nvfp4_triton, "quantize", lambda *a, **kw: events.append(("triton", a, kw))
        )
    x, global_scale = torch.zeros(3, 64, dtype=torch.bfloat16), torch.ones(1)
    out = nvfp4._empty_activation(3, 32, global_scale, x.device)
    names = (
        "weight_scale",
        "up_scale",
        "alpha",
        "output_scale",
        "reciprocal",
        "column_scale",
        "gate_factor",
        "up_factor",
        "clear_scale",
    )
    args = {name: torch.empty(1) for name in names}
    name = nvfp4._quantize_into(x, global_scale, out, gated=True, **args)
    assert name == ("cute" if cute else "triton")
    assert len(events) == 1 and events[0][0] == name
    assert events[0][2]["gated"] is True
    assert events[0][2]["data"] is out.data
    assert events[0][2]["scale"].data_ptr() == out.scale.data_ptr()
    assert all(events[0][2][key] is value for key, value in args.items())
    assert nvfp4._quantize_into(x[:0], global_scale, out, gated=False) is None
    assert len(events) == 1


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (12, 0), (12, 1)])
def test_triton_pdl_is_feature_and_capability_gated(monkeypatch, supported, capability):
    pytest.importorskip("triton")
    from ctikernels.linear import _nvfp4_triton

    monkeypatch.setattr(_nvfp4_triton, "_HAS_PDL", supported)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    expected = (
        {"PDL": True, "launch_pdl": True} if supported and capability >= (9, 0) else {"PDL": False}
    )
    assert _nvfp4_triton._pdl_options(torch.device("cuda", 0)) == expected


def test_quack_pdl_is_explicit(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    calls = []
    epilogue = SimpleNamespace(gemm=lambda *a, **kw: calls.append(kw))
    data = torch.empty(2, 32, dtype=torch.uint8)
    _nvfp4_quack.Launcher._run(epilogue, data, data, data, data, None, {}, _nvfp4_quack.DEFAULT)
    assert calls[0]["post_init_attrs"] == (("use_pdl", True),)


@pytest.mark.parametrize("refresh", [False, True])
def test_triton_scalar_weights_must_be_frozen_before_launch(monkeypatch, refresh):
    pytest.importorskip("triton")
    from ctikernels.linear import _nvfp4_triton

    monkeypatch.setattr(_nvfp4_triton, "_pdl_options", lambda _: pytest.fail("launch prepared"))
    scale = torch.ones(1)
    weight = torch.ones(1, requires_grad=True)
    with torch.no_grad(), pytest.raises(ValueError, match="frozen"):
        if refresh:
            _nvfp4_triton.refresh_scales(scale, weight)
        else:
            _nvfp4_triton.quantize(
                torch.zeros(1, 32), scale, gated=False, data=None, scale=None, weight_scale=weight
            )


@pytest.mark.parametrize("available", [False, True])
def test_cute_table_validation_and_fallback(monkeypatch, available):
    from ctikernels.linear import _nvfp4_cute_small as cute

    device = ("NVIDIA GB10", (12, 1))
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    monkeypatch.setattr(cute, "available", lambda _: available)
    valid = _nvfp4_dispatch.Cute(n_tile=32, k_tile=128, split_k=2, reduce_mode="cluster")
    fallback = _nvfp4_quack.Config(128, 128, False, True)
    table, shape = _nvfp4_dispatch.GATED, (128, 64)
    for plan in (
        valid,
        valid._replace(stages=9),
        valid._replace(reduce_mode="bad"),
        valid._replace(split_k=3),
        valid._replace(n_tile=16),
        valid._replace(n_tile="bad"),
        valid._replace(n_tile=128, k_tile=1024, stages=8),
    ):
        entries = ((32, "cute", plan), (64, _nvfp4_dispatch.UNFUSED, fallback))
        monkeypatch.setitem(table, device, {shape: entries})
        for rows in (0, 1, 3, 32, 33, 65):
            fits = available and 1 <= rows <= 32 and plan == valid
            expected = plan if fits else fallback
            assert _nvfp4_dispatch.gated_plan(None, *shape, rows) == (
                "cute" if fits else _nvfp4_dispatch.UNFUSED,
                expected,
            )
        assert _nvfp4_dispatch.gated_plan(None, *shape, 1, allow_cute=False) == (
            _nvfp4_dispatch.UNFUSED,
            fallback,
        )
    monkeypatch.setitem(table, device, {shape: entries[:1]})
    assert _nvfp4_dispatch.gated_plan(None, *shape, 64) == (
        _nvfp4_dispatch.FUSED,
        _nvfp4_quack.DEFAULT,
    )


@pytest.mark.parametrize("quack", [False, True])
def test_cute_dispatch_prepares_once_caches_and_returns_fresh_outputs(monkeypatch, quack):
    if quack:
        # The prequantized QuACK fallback below refreshes scales through Triton.
        pytest.importorskip("triton")
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_small as cute
    from ctikernels.linear import nvfp4

    device = ("NVIDIA GB10", (12, 1))
    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object() if quack else None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    monkeypatch.setattr(cute, "available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    repacks, workspaces, calls = [], [], []
    monkeypatch.setattr(cute, "prepare_interleaved", lambda *w: repacks.append(w) or w)

    def workspace(weight, rows, config, **kw):
        workspaces.append((rows, config, kw))
        return SimpleNamespace()

    def launch(*args, workspace, out):
        calls.append((args, workspace, out))
        return out

    monkeypatch.setattr(cute, "Workspace", workspace)
    monkeypatch.setattr(cute, "gated", launch)
    plan = _nvfp4_dispatch.Cute(n_tile=32, k_tile=128, split_k=2)
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device, {(128, 64): ((32, "cute", plan),)})
    w, scale = _weight(), torch.ones(1)
    op = NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale)
    assert len(repacks) == 1 and not workspaces
    assert op.provider == "cute"
    assert op.plan_for(3) == ("cute", plan)
    x = torch.zeros(3, 128, dtype=torch.bfloat16)
    first, second = op(x), op(x)
    assert first.data.data_ptr() != second.data.data_ptr()
    reused = op(x, out=first)
    assert reused.data.data_ptr() == first.data.data_ptr()
    assert len(workspaces) == 1 and len(repacks) == 1 and len(calls) == 3
    assert calls[0][0][2].data_ptr() == scale.data_ptr()
    assert op.quantizer == "cute"
    op(x.to(torch.float16))
    assert len(workspaces) == 2
    if quack:
        from ctikernels.linear import _nvfp4_triton

        fallback_calls = []
        fallback = lambda *args: fallback_calls.append(args[-1])
        op._launcher = SimpleNamespace(linear=fallback, gated=fallback)
        monkeypatch.setattr(_nvfp4_triton, "refresh_scales", lambda *args, **kw: None)
        a = nvfp4.NVFP4Activation(
            torch.zeros(3, 64, dtype=torch.uint8),
            torch.zeros(scale_shape(3, 128), dtype=torch.float8_e4m3fn),
            scale,
        )
        op.quantized(a)
        assert fallback_calls == [_nvfp4_quack.DEFAULT]
        assert len(calls) == 4
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op(x)
    with pytest.raises(RuntimeError, match="warm up"):
        op(x[:1])
    w.global_scale.requires_grad_()
    with pytest.raises(ValueError, match="require grad"):
        op(x)


def test_unavailable_cute_does_not_prepare_weights(monkeypatch):
    from ctikernels.linear import _nvfp4_cute_small as cute
    from ctikernels.linear import nvfp4

    device = ("NVIDIA GB10", (12, 1))
    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object())
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: device)
    monkeypatch.setattr(cute, "available", lambda _: False)
    fail = lambda *args: pytest.fail("prepared unavailable CuTe weights")
    monkeypatch.setattr(cute, "prepare_interleaved", fail)
    entries = ((32, "cute", _nvfp4_dispatch.Cute()),)
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device, {(128, 64): entries})
    w, scale = _weight(), torch.ones(1)
    op = NVFP4GateUpSiLU(w, w, input_scale=scale, output_scale=scale)
    assert op.provider == "quack" and op._cute_weight is None
    assert op.plan_for(3) == (_nvfp4_dispatch.FUSED, _nvfp4_quack.DEFAULT)


def test_cute_candidate_roundtrip_and_shape_sweep(monkeypatch):
    from ctikernels.linear import _nvfp4_cute_small as cute

    parity = _parity_module()
    from benchmark.kernel_tuning.nvfp4_dispatch import _candidate_literal

    monkeypatch.delenv("CTI_PARITY_NO_CUTE", raising=False)
    monkeypatch.setattr(cute, "available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    for gated in (False, True):
        for name in ("9b", "27b", "35b-shared"):
            _, plans = parity._cute_plans("cuda", gated, 32, name)
            assert plans
            for plan in plans:
                record = parity._candidate_record("cute", "cute" if gated else None, plan, False)
                literal = _candidate_literal(record)
                assert eval(literal, {"Cute": _nvfp4_dispatch.Cute}) == plan
    assert parity._cute_plans("cuda", False, 33, "9b") == (None, ())
    monkeypatch.setenv("CTI_PARITY_NO_CUTE", "1")
    assert parity._cute_plans("cuda", False, 1, "9b") == (None, ())


def test_cute_out_binding_preserves_workspace_storage():
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_small as cute

    workspace = SimpleNamespace(
        out=torch.empty(3, 64, dtype=torch.bfloat16),
        out_scale=torch.empty(1, dtype=torch.float8_e4m3fn),
        out_shapes=((3, 64), (1,)),
        out_dtypes=(torch.bfloat16, torch.float8_e4m3fn),
        signature=(torch.device("cpu"),),
    )
    out = torch.empty_like(workspace.out)
    bound = cute._output_workspace(workspace, out)
    assert bound.out is out and workspace.out is not out
    with pytest.raises(ValueError, match="out must match"):
        cute._output_workspace(workspace, out.to(torch.float16))


def test_prewarm_takes_rows_only_by_keyword():
    from ctikernels.linear import nvfp4

    assert nvfp4._prewarm_rows(object(), (3, 7)) == (3, 7)
    with pytest.raises(TypeError, match="rows="):
        nvfp4._prewarm_rows(object(), None, [1, 2])


def test_default_prewarm_covers_every_cute_row_count(monkeypatch):
    from types import SimpleNamespace

    from ctikernels.linear import nvfp4

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    op = SimpleNamespace(_cute=object(), in_features=64, device=torch.device("cuda:0"))
    with_cute = nvfp4._prewarm_rows(op, None)
    assert set(range(1, 33)) <= set(with_cute) and set(nvfp4.PREWARM_ROWS) <= set(with_cute)
    assert len(with_cute) == len(set(with_cute))
    op._cute = None
    assert nvfp4._prewarm_rows(op, None) == tuple(dict.fromkeys(nvfp4.PREWARM_ROWS))
    assert nvfp4._prewarm_rows(SimpleNamespace(_cute=object()), [3]) == (3,)


@pytest.fixture
def quantizer_launcher_class():
    """Load the real cache/dispatch code without importing CuTe or initializing CUDA."""
    import ast
    import threading
    from pathlib import Path

    from ctikernels.linear import _nvfp4_cute_quant

    tree = ast.parse(Path(_nvfp4_cute_quant.__file__).with_name("launcher.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {
        "torch": torch,
        "threading": threading,
        "_geometry": _nvfp4_cute_quant._geometry,
        "_early_trigger": _nvfp4_cute_quant._early_trigger,
        "_pdl": _nvfp4_cute_quant._pdl,
    }
    exec(compile(tree, "<quantizer launcher>", "exec"), namespace)
    return namespace["Launcher"]


@pytest.mark.parametrize("warm", [False, True])
def test_quantizer_capture_miss_never_compiles(monkeypatch, quantizer_launcher_class, warm):
    from contextlib import nullcontext

    launcher = quantizer_launcher_class.__new__(quantizer_launcher_class)
    launcher.device, launcher.wide, launcher.sm_count = torch.device("cpu"), True, 1
    launcher.name = "Test GPU"
    launcher.compiled = {}
    calls = []
    monkeypatch.setattr(launcher, "_compile", lambda key: calls.append(key) or (lambda *a: None))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    args = {
        "gated": False,
        "data": torch.empty(2, 32, dtype=torch.uint8),
        "scale": torch.empty(scale_shape(2, 64), dtype=torch.uint8),
    }
    if warm:
        launcher.quantize(x, torch.ones(1), **args)
        x = torch.zeros(129, dtype=torch.bfloat16)[1:].view(2, 64)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prewarm.*outside CUDA graph capture"):
        launcher.quantize(x, torch.ones(1), **args)
    assert len(calls) == len(launcher.compiled) == int(warm)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launcher.quantize(x, torch.ones(1), **args)
    assert len(calls) == int(warm) + 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    launcher.quantize(x, torch.ones(1), **args)
    assert len(calls) == int(warm) + 1


def test_flashinfer_cold_capture_never_builds(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.linear import _nvfp4_flashinfer as fi

    monkeypatch.setattr(fi, "_launchers", {})
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(fi, "_resolve", lambda _: pytest.fail("build during capture"))
    with pytest.raises(RuntimeError, match="prewarm.*outside CUDA graph capture"):
        fi.load(torch.device("cuda"))
    assert not fi._launchers


@pytest.mark.parametrize("shared", [False, True])
def test_flashinfer_gated_alpha_precedes_bf16_rounding(monkeypatch, shared):
    import sys
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear.nvfp4 import NVFP4Activation

    # Independent but initially equal values must still use two GEMMs: replay
    # can update either slot. Each mm receives a contiguous view, never a copy.
    gate, up = _weight(32, 64), _weight(32, 64)
    gate.global_scale.fill_(0.7)
    up = up._replace(global_scale=gate.global_scale if shared else gate.global_scale.clone())
    a = NVFP4Activation(gate.data[:3], blocked_scale(gate.scale[:3]), torch.tensor([0.2]))
    out = torch.empty(3, 64, dtype=torch.bfloat16)
    calls = []

    def mm(*args, alpha, out, **kwargs):
        assert out.is_contiguous()
        calls.append(alpha.clone())
        out.fill_(float(alpha * 5120))

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(mm_fp4=mm))
    launcher = fi.Launcher()
    prepared = launcher.prepare_gated(gate, up)
    assert prepared[1] is shared
    for _ in range(2):
        calls.clear()
        launcher.gated(a, prepared, out, alpha=a.global_scale * gate.global_scale)
        scales = (gate.global_scale,) if shared else (gate.global_scale, up.global_scale)
        assert len(calls) == len(scales)
        for actual, weight_scale in zip(calls, scales, strict=True):
            torch.testing.assert_close(actual, a.global_scale * weight_scale)
        if shared:
            torch.testing.assert_close(
                out, (a.global_scale * gate.global_scale * 5120).bfloat16().expand_as(out)
            )
        else:
            for actual, weight_scale in zip(out.view(2, 3, 32), scales, strict=True):
                torch.testing.assert_close(
                    actual, (a.global_scale * weight_scale * 5120).bfloat16().expand_as(actual)
                )
        a.global_scale.mul_(1.5)
        up.global_scale.mul_(2)


def test_default_prewarm_skips_gated_padding_buckets(monkeypatch, quantizer_launcher_class):
    """The padding clear takes its length at run time: one key for every 128-row bucket."""
    import math
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ctikernels.linear import nvfp4

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    op = SimpleNamespace(
        _cute=None, hidden_size=256, intermediate_size=32, device=torch.device("cuda:0")
    )
    counts = nvfp4._prewarm_rows(op, None, 16384)
    assert len(counts) < 30 and max(counts) <= 16384
    assert nvfp4._prewarm_rows(op, [3]) == (3,)

    launcher = quantizer_launcher_class.__new__(quantizer_launcher_class)
    launcher.device, launcher.wide, launcher.sm_count = op.device, False, 1
    launcher.name, launcher.compiled = "Test GPU", {}
    observed = []
    monkeypatch.setattr(launcher, "_compile", lambda key: lambda *args: observed.append(key))
    x = torch.empty(1, 256, dtype=torch.bfloat16)
    data = torch.empty(1, 128, dtype=torch.uint8)
    scale = torch.empty(scale_shape(1, 256), dtype=torch.uint8)
    for rows in (1, 129, 4097, 16384):
        clear = torch.empty(math.prod(scale_shape(rows, 32)), dtype=torch.uint8)
        launcher.quantize(x, torch.ones(1), gated=False, data=data, scale=scale, clear_scale=clear)
    assert len(set(observed)) == 1 and len(launcher.compiled) == 1


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize(
    "shape", [(2048, 512), (4096, 12288), (5120, 17408), (72, 128), (1001, 128), (1004, 128)]
)
def test_linear_plan_is_flashinfer_wherever_it_fits(monkeypatch, capability, shape):
    from ctikernels.linear import _nvfp4_flashinfer

    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Any GPU", capability))
    fits = shape[0] % 8 == 0  # b12x silently miscomputes N % 8 != 0: refused, not run
    assert _nvfp4_flashinfer.supports_shape(*shape, capability) == fits
    assert _nvfp4_dispatch.linear_plan(None, *shape) == (
        _nvfp4_dispatch.FlashInfer() if fits else _nvfp4_quack.DEFAULT
    )
    assert _nvfp4_flashinfer.linear_backend(capability) == (
        "b12x" if capability == (12, 0) else "cutlass"
    )


def test_forced_flashinfer_refuses_unsupported_n(monkeypatch):
    from ctikernels.linear import _nvfp4_flashinfer, nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Test GPU", (12, 0)))
    monkeypatch.setattr(_nvfp4_flashinfer, "load", lambda _: _nvfp4_flashinfer.Launcher())
    with pytest.raises(ValueError, match="does not support N=1004"):
        NVFP4Linear(_weight(1004), torch.ones(1), provider="flashinfer")


@pytest.mark.parametrize("quack", [False, True])
@pytest.mark.parametrize("flashinfer", [False, True])
@pytest.mark.parametrize("n", [7, 8, 63, 64, 72])
def test_linear_default_and_overrides_without_table(monkeypatch, quack, flashinfer, n):
    from ctikernels.linear import _nvfp4_flashinfer, nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: ("Unlisted GPU", (12, 1)))
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object() if quack else None)
    loads = []
    launcher = _nvfp4_flashinfer.Launcher()
    monkeypatch.setattr(
        _nvfp4_flashinfer, "load", lambda _: loads.append(1) or (launcher if flashinfer else None)
    )
    weight, scale = _weight(n), torch.ones(1)
    op = NVFP4Linear(weight, scale)
    use_flash = flashinfer and n % 8 == 0
    expected = "flashinfer" if use_flash else "quack" if quack else "dequant"
    assert op.provider == expected
    assert loads == ([1] if n % 8 == 0 else [])
    plan = _nvfp4_dispatch.FlashInfer() if use_flash else _nvfp4_quack.DEFAULT if quack else None
    for rows in (1, 16, 17, 4097):
        assert op.plan_for(rows) == plan
    loads.clear()
    if quack:
        forced = NVFP4Linear(weight, scale, provider="quack")
        assert forced.provider == "quack" and forced.plan_for(1) == _nvfp4_quack.DEFAULT
    forced = NVFP4Linear(weight, scale, provider="dequant")
    assert forced.provider == "dequant" and forced.plan_for(1) is None
    assert not loads


@pytest.mark.parametrize(
    "capability,n,k,expected",
    [
        ((12, 0), 8, 32, True),
        ((12, 1), 72, 96, True),
        ((12, 0), 7, 128, False),
        ((12, 1), 64, 16, False),
        ((10, 0), 8, 32, False),
        ((10, 0), 32, 32, True),
        ((12, 0), 0, 32, False),
        ((12, 1), 8, 0, False),
    ],
)
def test_flashinfer_shape_constraints(capability, n, k, expected):
    from ctikernels.linear._nvfp4_flashinfer import supports_shape

    assert supports_shape(n, k, capability) is expected


@pytest.fixture
def target_device_context(monkeypatch):
    """Model capture on cuda:1 while cuda:0 is initially current."""
    from contextlib import contextmanager

    current = [torch.device("cuda:0")]
    target = torch.device("cuda:1")
    checks = []

    @contextmanager
    def device(selected):
        previous = current[0]
        current[0] = selected
        try:
            yield
        finally:
            current[0] = previous

    def capturing():
        checks.append(current[0])
        return current[0] == target

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    yield target, current, checks
    assert current[0] == torch.device("cuda:0")
    assert checks == [target]


@pytest.mark.parametrize("site", ["quantizer", "flashinfer", "cute-workspace", "cute-prepare"])
def test_capture_guard_checks_target_device(
    monkeypatch, quantizer_launcher_class, target_device_context, site
):
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_small as cute
    from ctikernels.linear import _nvfp4_flashinfer as fi
    from ctikernels.linear import nvfp4

    target, _, _ = target_device_context
    fail = lambda *args, **kwargs: pytest.fail("cold build/allocation during target capture")
    if site == "quantizer":
        launcher = quantizer_launcher_class.__new__(quantizer_launcher_class)
        launcher.device, launcher.wide, launcher.sm_count = target, True, 1
        launcher.name = "Test GPU"
        launcher.compiled = {}
        monkeypatch.setattr(launcher, "_compile", fail)
        call = lambda: launcher.quantize(
            torch.empty(3, 64, dtype=torch.bfloat16),
            torch.ones(1),
            gated=False,
            data=torch.empty(3, 32, dtype=torch.uint8),
            scale=torch.empty(512, dtype=torch.uint8),
        )
    elif site == "flashinfer":
        monkeypatch.setattr(fi, "_launchers", {})
        monkeypatch.setattr(fi, "_resolve", fail)
        call = lambda: fi.load(target)
    elif site == "cute-workspace":
        op = SimpleNamespace(
            device=target,
            _cute_workspaces={},
            _cute=SimpleNamespace(Workspace=fail),
            _cute_weight=None,
        )
        x = SimpleNamespace(shape=(3, 64), dtype=torch.bfloat16, device=target)
        call = lambda: nvfp4._cute_workspace(op, x, _nvfp4_dispatch.Cute())
    else:
        monkeypatch.setattr(cute, "available", lambda _: True)
        monkeypatch.setattr(cute, "prepare_interleaved", fail)
        op = SimpleNamespace(device=target, _interleaved=(None, None), _weight_globals=(None, None))
        call = lambda: nvfp4._prepare_cute(op, "cute", (64, 64))
    with pytest.raises(RuntimeError, match="outside.*capture"):
        call()


def test_quack_frontend_runs_under_target_device(target_device_context):
    from types import SimpleNamespace

    target, current, _ = target_device_context

    def gemm(*args, **kwargs):
        assert current[0] == target
        assert torch.cuda.is_current_stream_capturing()

    operand = SimpleNamespace(device=target, unsqueeze=lambda _: None)
    _nvfp4_quack.Launcher._run(
        SimpleNamespace(gemm=gemm),
        operand,
        operand,
        operand,
        operand,
        None,
        {},
        _nvfp4_quack.DEFAULT,
    )


def test_cute_small_compile_guard_checks_target_device(monkeypatch, target_device_context):
    """Exercise the real small-M launch function without importing the DSL."""
    import ast
    import threading
    from pathlib import Path
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_small

    target, _, _ = target_device_context
    tree = ast.parse(Path(_nvfp4_cute_small.__file__).with_name("kernel.py").read_text())
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "launch"
    ]
    tensor = SimpleNamespace(
        device=target,
        dtype=torch.bfloat16,
        stride=lambda _: 64,
        data_ptr=lambda: 16,
        element_size=lambda: 2,
    )
    names = (
        "out",
        "out_scale",
        "partial",
        "counters",
        "debug_data",
        "debug_scale",
        "projection",
        "postact",
        "errors",
        "stamps",
        "probe_counts",
    )
    workspace = SimpleNamespace(
        signature=(3, 64, 64, None, False, False, "global", None),
        debug_timeout=False,
        **dict.fromkeys(names, tensor),
    )
    weight = SimpleNamespace(**dict.fromkeys(("data", "scale", "global0", "global1"), tensor))
    namespace = {
        "torch": torch,
        "available": lambda _: True,
        "_cache": {},
        "_lock": threading.Lock(),
        "cutlass": SimpleNamespace(BFloat16=0, Float16=0),
        "cute": SimpleNamespace(AddressSpace=SimpleNamespace(gmem=0)),
        "make_ptr": lambda *args, **kwargs: None,
        "cuda": SimpleNamespace(CUstream=lambda _: None),
        **dict.fromkeys(("Uint8", "Float32", "Int32", "Int64"), 0),
    }
    # The stream lookup is device-specific even while cuda:0 is initially current.
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=0))
    exec(compile(tree, "<small-M launch>", "exec"), namespace)
    with pytest.raises(RuntimeError, match="prewarm.*outside capture"):
        namespace["launch"](tensor, weight, tensor, tensor, None, workspace)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize(
    "hidden,intermediate", [(5120, None), (5120, 32), (32, 5120), (512, None), (512, 96)]
)
def test_default_prewarm_covers_quantizer_specializations(
    monkeypatch, quantizer_launcher_class, wide, hidden, intermediate
):
    """Compare actual launcher keys for all M with keys reached by default prewarm."""
    import math
    from contextlib import nullcontext
    from types import SimpleNamespace

    from ctikernels.linear import nvfp4

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0 if wide else 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    # A no-PDL band inside one dispatch bucket: prewarm must visit both launch modes.
    monkeypatch.setitem(_nvfp4_cute_quant.PDL_OFF, "Test GPU", ((hidden, 33, 200),))
    op = SimpleNamespace(device=torch.device("cuda:0"), _cute=None)
    if intermediate is None:
        op.in_features = hidden
    else:
        op.hidden_size, op.intermediate_size = hidden, intermediate
    # K or I = 512 moves the load-group and early-trigger thresholds above 4096 rows.
    limit = 24_000 if hidden == 512 else 4096
    counts = nvfp4._prewarm_rows(op, None, limit)
    assert len(counts) == len(set(counts)) < 100 and max(counts) <= limit
    assert nvfp4._prewarm_rows(op, [780]) == (780,)

    launcher = quantizer_launcher_class.__new__(quantizer_launcher_class)
    launcher.device, launcher.wide, launcher.sm_count = op.device, wide, 1
    launcher.name = "Test GPU"
    launcher.compiled = {}
    observed = []
    monkeypatch.setattr(launcher, "_compile", lambda key: lambda *args: observed.append(key))
    scalar = torch.ones(1)
    for gated, cols in [(False, hidden)] + ([(True, intermediate)] if intermediate else []):
        x = torch.empty(limit, cols * (2 if gated else 1), dtype=torch.bfloat16)
        data = torch.empty(limit, cols // 2, dtype=torch.uint8)
        scale = torch.empty(scale_shape(limit, cols), dtype=torch.uint8)
        clear = torch.empty(scale_shape(limit, intermediate or 64), dtype=torch.uint8)
        keys = {}
        for rows in range(1, limit + 1):
            kwargs = {}
            if not gated and intermediate and intermediate % 64:
                size = math.prod(scale_shape(rows, intermediate))
                kwargs["clear_scale"] = clear.view(-1)[:size]
            launcher.quantize(
                x[:rows], scalar, gated=gated, data=data[:rows], scale=scale, **kwargs
            )
            bucket = min(rows, _nvfp4_dispatch.LAST_BUCKET)
            keys[rows] = (1 << (bucket - 1).bit_length(), observed.pop())
        assert {keys[m] for m in counts} == set(keys.values())
        if wide and cols == 5120:
            assert any(769 <= m <= 781 for m in counts)
