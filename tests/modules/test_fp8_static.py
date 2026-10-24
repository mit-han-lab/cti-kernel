# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Static FP8 host contracts, with real CPU tensors and substituted CUDA metadata."""

import logging
import os
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels import _reroute
from ctikernels.linear import _fp8_static_dispatch as dispatch
from ctikernels.linear import block_fp8
from ctikernels.linear import fp8_static as static
from ctikernels.linear.cute import _fp8_static_quant as quantizer
from ctikernels.linear.torch import _fp8_static as market

from ..support.child import run_child

E4M3 = torch.float8_e4m3fn
REAL_CHECK_DEVICE = static._check_device
REAL_ROWWISE_PROBE = market.rowwise_probe
REAL_BLOCK_QUANTIZED = block_fp8.BlockFP8Linear.quantized
REAL_BLOCK_WARM_UP = block_fp8.BlockFP8Linear.warm_up


def tensors(n=256, k=128, scales=1):
    return torch.zeros(n, k, dtype=E4M3), torch.arange(1, scales + 1).float(), torch.tensor(0.03)


@pytest.fixture
def cpu(monkeypatch):
    state = SimpleNamespace(capturing=False, calls=[], workspace_warms=[])
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: "rowwise unsupported")
    monkeypatch.setattr(static, "_check_device", lambda *a: None)
    monkeypatch.setattr(static, "cute_available", lambda _: True)
    monkeypatch.setattr(static, "_cuda_capability", lambda _: (12, 1))
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: (12, 1))
    monkeypatch.setattr(block_fp8, "_cuda_device", lambda name, device: device)
    check = block_fp8._check_weight
    monkeypatch.setattr(
        block_fp8, "_check_weight", lambda *a, **kw: check(*a, **kw, require_cuda=False)
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=48, major=12, minor=1),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)

    def run(self, q, scale, *, out_dtype=None, out=None):
        state.calls.append((q, scale, out))
        out.zero_()
        return out

    monkeypatch.setattr(block_fp8.BlockFP8Linear, "quantized", run)
    monkeypatch.setattr(block_fp8.BlockFP8Linear, "warm_up", lambda *a, **kw: None)
    # The workspace entries need a CUDA buffer; the GPU cold-capture tests cover them.
    monkeypatch.setattr(
        static.StaticFP8Linear,
        "_warm_up_workspace",
        lambda self, x, outputs, out_dtype, quantized: state.workspace_warms.append(
            (x.shape[0], len(outputs), quantized)
        ),
    )
    return state


@pytest.mark.parametrize("index", [0, 1, 2])
def test_frozen_check_is_first(index):
    args = list(tensors())
    args[index] = args[index].float().requires_grad_()
    with pytest.raises(ValueError, match="is frozen"):
        static.StaticFP8Linear(*args, slot_rows=(-1,))


@pytest.mark.parametrize("index", [0, 1, 2])
def test_dtype_validation(index):
    args = list(tensors())
    args[index] = args[index].to(torch.float16)
    with pytest.raises(TypeError, match="must be"):
        static.StaticFP8Linear(*args)


@pytest.mark.parametrize(
    "index,value,error",
    [
        (0, None, TypeError),
        (0, torch.zeros(128, dtype=E4M3), ValueError),
        (0, torch.zeros(0, 128, dtype=E4M3), ValueError),
        (0, torch.zeros(256, 256, dtype=E4M3)[:, ::2], ValueError),
        (1, torch.ones(1, 1), ValueError),
        (1, torch.empty(0), ValueError),
        (2, torch.ones(2), ValueError),
        (2, None, TypeError),
        (1, torch.ones(1, device="meta"), ValueError),
        (2, torch.ones(1, device="meta"), ValueError),
    ],
)
def test_bad_shapes_and_devices(index, value, error):
    args = list(tensors())
    args[index] = value
    with pytest.raises(error):
        static.StaticFP8Linear(*args)


@pytest.mark.parametrize(
    "slots", [None, (), (256,), (100, 100), (0, 256), (-1, 257), (128.0, 128), (True, 255), "ab"]
)
def test_invalid_slots(slots):
    with pytest.raises(ValueError, match="slot_rows"):
        static.StaticFP8Linear(*tensors(scales=2), slot_rows=slots)


def test_slots_require_sequence():
    with pytest.raises(TypeError, match="sequence"):
        static.StaticFP8Linear(*tensors(), slot_rows=256)


def test_unsupported_device():
    with pytest.raises(RuntimeError, match="compute capability 8.9 or newer"):
        static.StaticFP8Linear(*tensors())
    with pytest.raises(RuntimeError, match="compute capability 8.9 or newer"):
        static.quantize_static_fp8(torch.zeros(0, 128, dtype=torch.bfloat16), torch.ones(1))


@pytest.mark.parametrize(
    "slots,k,expected",
    [
        ((128, 256, 17), 200, [[1, 2, 2, 3]]),
        ((100, 36, 120), 200, [[1], [2], [3]]),
        ((128, 128), 128, [[1, 2]]),
    ],
)
def test_block_scale_grids(slots, k, expected):
    scales = torch.arange(1, len(slots) + 1).float()
    grids = static._block_grids(scales, slots, k)
    assert len(grids) == len(expected)
    start = 0
    for (columns, grid), values in zip(grids, expected, strict=True):
        assert columns.start == start
        start = columns.stop
        assert grid.is_contiguous()
        torch.testing.assert_close(
            grid, torch.tensor(values).float()[:, None].expand(-1, (k + 127) // 128)
        )
    assert start == sum(slots)


@pytest.mark.parametrize("slots", [(128, 128), (100, 36, 120)])
def test_construction_references_and_blocks(cpu, slots):
    w, ws, s = tensors(scales=len(slots))
    op = static.StaticFP8Linear(w, ws, s, slot_rows=slots)
    assert op._weight is w and op.weight_scale is ws and op.input_scale is s
    assert op.slot_rows == slots and op.device == w.device
    assert (op.in_features, op.out_features) == (128, 256)
    assert op.provider == "block-fp8"
    assert op.provider_for(10000) == "block-fp8"
    assert len(op._blocks) == (1 if slots == (128, 128) else 3)
    for columns, block in op._blocks:
        assert block._weight.data_ptr() == w[columns].data_ptr()


def test_scalar_scale(cpu):
    w, ws, s = tensors()
    op = static.StaticFP8Linear(w, ws.squeeze(), s.reshape(1, 1))
    assert op.slot_rows == (256,)


@pytest.mark.parametrize(
    "k,slots,allowed", [(128, (128, 128), True), (200, (128, 128), False), (128, (100, 156), False)]
)
def test_market_availability_and_auto_fallback(cpu, monkeypatch, k, slots, allowed):
    assert market.available(k, slots) is allowed
    monkeypatch.setitem(
        dispatch.TABLE, (12, 1, 48), {(256, k): ((8, "torch-scaled-mm"), (16, "block-fp8"))}
    )
    args = tensors(k=k, scales=2)
    op = static.StaticFP8Linear(*args, slot_rows=slots)
    assert op.provider_for(8) == "block-fp8"
    assert op.provider_for(9) == op.provider_for(1000) == "block-fp8"
    forced = static.StaticFP8Linear(*args, slot_rows=slots, provider="block-fp8")
    assert forced.provider_for(1) == "block-fp8"
    if allowed:
        forced = static.StaticFP8Linear(*args, slot_rows=slots, provider="torch-scaled-mm")
        assert forced.provider_for(1000) == "torch-scaled-mm:slots"
    else:
        with pytest.raises(ValueError, match="divisible by 16"):
            static.StaticFP8Linear(*args, slot_rows=slots, provider="torch-scaled-mm")


def test_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        static.StaticFP8Linear(*tensors(), provider="other")


@pytest.mark.parametrize(
    "values,equal",
    [([1.0], True), ([0.2, 0.2], True), ([0.2, 0.3], False), ([float("nan"), float("nan")], False)],
)
def test_equal_scales(values, equal):
    assert market._equal_scales(torch.tensor(values)) is equal


@pytest.mark.parametrize("equal", [False, True])
def test_market_calls_and_strided_copy(cpu, monkeypatch, equal):
    w, ws, s = tensors(scales=2)
    if equal:
        ws.fill_(0.25)
    op = static.StaticFP8Linear(w, ws, s, slot_rows=(128, 128), provider="torch-scaled-mm")
    calls = []

    def mm(q, w, **kw):
        calls.append((w, kw))
        return torch.full((q.shape[0], w.shape[1]), 7, dtype=kw["out_dtype"])

    monkeypatch.setattr(torch, "_scaled_mm", mm)
    buf = torch.full((3, 270), -1, dtype=torch.bfloat16)
    out = buf[:, 3:259]
    assert op.quantized(torch.zeros(3, 128, dtype=E4M3), out=out) is out
    assert len(calls) == (1 if equal else 2)
    assert torch.all(out == 7) and torch.all(buf[:, :3] == -1) and torch.all(buf[:, 259:] == -1)
    for i, (_, kw) in enumerate(calls):
        assert kw["scale_a"].data_ptr() == s.data_ptr()
        assert kw["scale_b"].data_ptr() == ws[i : i + 1].data_ptr()
        assert kw["use_fast_accum"] is False


@pytest.mark.parametrize("entry", ["__call__", "quantized", "quantize", "warm_up"])
@pytest.mark.parametrize("field", ["_weight", "_weight_scale", "_input_scale", "_bias"])
def test_late_grad_check(cpu, entry, field):
    op = static.StaticFP8Linear(*tensors(), bias=torch.zeros(256, dtype=torch.bfloat16))
    getattr(op, field).requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        getattr(op, entry)(None)


@pytest.mark.parametrize(
    "entry,dtype", [("__call__", torch.bfloat16), ("quantized", E4M3), ("quantize", torch.float16)]
)
def test_input_grad(cpu, entry, dtype):
    op = static.StaticFP8Linear(*tensors())
    with pytest.raises(RuntimeError, match="inference-only"):
        getattr(op, entry)(torch.zeros(1, 128, dtype=dtype, requires_grad=True))


@pytest.mark.parametrize("quantized", [False, True])
def test_output_validation_and_empty(cpu, quantized):
    op = static.StaticFP8Linear(*tensors())
    call = op.quantized if quantized else op
    x = torch.empty(0, 128, dtype=E4M3 if quantized else torch.float16)
    assert call(x).dtype == (torch.bfloat16 if quantized else torch.float16)
    out = torch.empty_strided((0, 256), (300, 1), dtype=torch.float16)
    assert call(x, out=out) is out
    for bad in (
        torch.empty(0, 255, dtype=torch.float16),
        torch.empty(0, 256),
        torch.empty(0, 512, dtype=torch.float16)[:, ::2],
    ):
        with pytest.raises((ValueError, TypeError)):
            call(x, out=bad)
    with pytest.raises(ValueError, match="out_dtype"):
        call(x, out=out, out_dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="inference-only"):
        call(x, out=out.requires_grad_())
    assert not cpu.calls


def test_quantized_expands_scales_without_copy(cpu):
    op = static.StaticFP8Linear(*tensors())
    op.quantized(torch.zeros(3, 128, dtype=E4M3))
    scale = cpu.calls[0][1]
    assert scale.data_ptr() == op.input_scale.data_ptr()
    assert scale.stride(0) == 0
    assert torch.equal(scale, op.input_scale.expand(3, 1))


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("k", [128, 384])
@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
def test_warm_up_passes_actual_block_layouts(cpu, monkeypatch, quantized, k, dtype):
    op = static.StaticFP8Linear(*tensors(k=k), provider="block-fp8")
    warmed = []

    def warm(self, **kwargs):
        warmed.append((self, kwargs))

    def quant(x, s, *, block_scale=False, row_scale=False, **kwargs):
        q = torch.zeros(x.shape, dtype=E4M3)
        scale = torch.empty(x.shape[0], (k + 127) // 128 if block_scale else 1).copy_(s)
        return (q, scale) if block_scale or row_scale else q

    monkeypatch.setattr(block_fp8.BlockFP8Linear, "warm_up", warm)
    monkeypatch.setattr(quantizer, "quantize", quant)
    op.warm_up(batch_sizes=[1, 3, 17], dtype=dtype, out_dtype=dtype, quantized=quantized)
    assert len(warmed) == len(cpu.calls) == 3
    for (inner, kwargs), (q, scale, _) in zip(warmed, cpu.calls, strict=True):
        assert kwargs["batch_sizes"] == [q.shape[0]]
        assert kwargs["quantized"] is True and kwargs["out_dtype"] == dtype
        examples = kwargs["input_layouts"]
        assert ((q.stride(0), False), (*scale.stride(), False)) in examples
        # Both warmed prequantized and workspace scale grids are packed.
        assert ((k, False), ((k + 127) // 128, 1, False)) in examples
        assert inner is op._blocks[0][1]


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
def test_static_block_capture_after_warm_up(cpu, monkeypatch, quantized, workspace, dtype):
    monkeypatch.setattr(block_fp8.BlockFP8Linear, "quantized", REAL_BLOCK_QUANTIZED)
    monkeypatch.setattr(block_fp8.BlockFP8Linear, "warm_up", REAL_BLOCK_WARM_UP)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    op = static.StaticFP8Linear(*tensors(), provider="block-fp8")
    launches = []
    for _, block in op._blocks:
        monkeypatch.setattr(block._triton, "run", lambda *a, **kw: launches.append(a))

    def quant(x, s, *, block_scale=False, row_scale=False, outputs=None, **kwargs):
        if outputs is None:
            q = torch.empty(x.shape, dtype=E4M3)
            scale = torch.empty(x.shape[0], 1) if block_scale or row_scale else None
        else:
            q, scale = outputs
        q.zero_()
        if scale is not None:
            scale.copy_(s)
        return (q, scale) if block_scale or row_scale else q

    monkeypatch.setattr(quantizer, "quantize", quant)
    op.warm_up(max_m=17, dtype=dtype, out_dtype=dtype, quantized=quantized)
    scratch = torch.empty(op.workspace_bytes(17) // 4) if workspace else None
    cpu.capturing = True
    for m in (1, 3, 16, 17):
        x = torch.zeros(m, 128, dtype=dtype)
        q = torch.zeros(m, 129 if workspace else 128, dtype=E4M3)
        if workspace:
            q = q[:, 1:]
        out = torch.empty(m, 256, dtype=dtype)
        if quantized:
            op.quantized(q, out=out, workspace=scratch)
        else:
            op(x, out=out, workspace=scratch)
        launched_q, launched_scale, _ = launches[-1]
        assert launched_q.data_ptr() % 16 == 0
        assert launched_scale.stride() == (1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_m": 1, "batch_sizes": [1]},
        {"max_m": 0},
        {"max_m": True},
        {"batch_sizes": [-1]},
        {"batch_sizes": [1.0]},
        {"max_m": 1, "out_row_stride": 255},
        {"max_m": 1, "dtype": torch.float32},
    ],
)
def test_warm_up_arguments(cpu, kwargs):
    op = static.StaticFP8Linear(*tensors())
    with pytest.raises((ValueError, TypeError)):
        op.warm_up(**kwargs)
    assert not cpu.calls


@pytest.mark.parametrize("quantized", [False, True])
def test_warm_up_dispatch_and_capture(cpu, monkeypatch, quantized):
    op = static.StaticFP8Linear(*tensors())
    calls = []

    def quant(x, s, *, block_scale=False, row_scale=False, pdl=False):
        calls.append((x.shape, x.stride(), block_scale))
        q = torch.zeros(x.shape, dtype=E4M3)
        return (q, s.expand(x.shape[0], 1).contiguous()) if block_scale or row_scale else q

    monkeypatch.setattr(quantizer, "quantize", quant)
    op.warm_up(max_m=17, quantized=quantized, out_row_stride=300)
    sizes = block_fp8._warm_up_sizes("op", 17, None, op._dispatch_key)
    assert {op._dispatch_key(m) for m in sizes} == {op._dispatch_key(m) for m in range(1, 18)}
    # Each dispatch representative warms packed output and both base-alignment
    # classes of the requested row-strided output.
    assert [(q.shape[0], out.stride(0), out.data_ptr() % 16) for q, _, out in cpu.calls] == [
        (m, stride, offset) for m in sizes for stride, offset in ((256, 0), (300, 0), (300, 2))
    ]
    assert {b for _, _, b in calls} == {False, True}
    assert {stride[0] for _, stride, _ in calls} == {128, 129}
    cpu.capturing = True
    with pytest.raises(RuntimeError, match="before capture"):
        op.warm_up(1)


def test_warm_key_includes_inner_dispatch(cpu, monkeypatch):
    op = static.StaticFP8Linear(*tensors())
    monkeypatch.setattr(
        op._blocks[0][1], "_dispatch_key", lambda m: ("cute-fp8", "a" if m < 3 else "b")
    )
    assert op._dispatch_key(2) != op._dispatch_key(3)


def test_quantizer_metadata(cpu):
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    for bad, scale, error in [
        (x.float(), torch.ones(1), TypeError),
        (x[:, ::2], torch.ones(1), ValueError),
        (x[:, :0], torch.ones(1), ValueError),
        (x, torch.ones(2), ValueError),
        (x, torch.ones(1, requires_grad=True), ValueError),
    ]:
        with pytest.raises(error):
            static.quantize_static_fp8(bad, scale)


def test_quantizer_cache_capture_and_singleton_stride(cpu, monkeypatch):
    from ctikernels.linear.cute import _block_fp8_provider

    compiles, launches = [], []
    monkeypatch.setattr(quantizer, "_compiled", {})
    monkeypatch.setattr(quantizer, "_ready", set())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(_block_fp8_provider, "available", lambda _: True)

    def compile_fn(*args):
        compiles.append(args)
        return lambda *args: launches.append(args)

    monkeypatch.setattr(quantizer, "_compile", compile_fn)
    s = torch.ones(17)[::17]
    assert s.stride() == (17,)
    for k in (128, 200, 6144):
        quantizer.quantize(torch.zeros(3, k, dtype=torch.bfloat16), s)
    assert len(compiles) == 1  # K is symbolic, including K % 128 tails.
    assert all(
        scale.stride() == (1,) and scale.data_ptr() == s.data_ptr() for _, scale, _, _ in launches
    )
    quantizer.quantize(torch.zeros(0, 1, dtype=torch.float16), s, block_scale=True)
    assert len(compiles) == 1 and len(launches) == 3
    cpu.capturing = True
    quantizer.quantize(torch.zeros(1, 5120, dtype=torch.bfloat16), s)
    with pytest.raises(RuntimeError, match="warm up"):
        quantizer.quantize(torch.zeros(1, 1, dtype=torch.bfloat16), s)
    cpu.capturing = False
    quantizer.quantize(torch.zeros(1, 1, dtype=torch.bfloat16), s, block_scale=True)
    assert compiles[-1][1:3] == (False, True)


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 127, dtype=E4M3),
        torch.zeros(128, dtype=E4M3),
        torch.zeros(1, 128),
        torch.zeros(1, 256, dtype=E4M3)[:, ::2],
        torch.zeros(1, 128, device="meta", dtype=E4M3),
    ],
)
def test_quantized_validation(cpu, bad):
    op = static.StaticFP8Linear(*tensors())
    with pytest.raises((ValueError, TypeError)):
        op.quantized(bad)
    assert not cpu.calls


@pytest.mark.parametrize("layout", ["strided", "overlap", "unaligned"])
def test_market_packs_weight_once(cpu, layout):
    base = torch.zeros(256 * 144 + 1, dtype=E4M3)
    stride = {"strided": 144, "overlap": 0, "unaligned": 128}[layout]
    w = base.as_strided((256, 128), (stride, 1), storage_offset=int(layout == "unaligned"))
    op = static.StaticFP8Linear(w, torch.ones(1), torch.ones(1), provider="torch-scaled-mm")
    assert op._weight is w
    packed = op._libraries["torch-scaled-mm"]._slots[0][1].t()
    assert packed.is_contiguous() and packed.data_ptr() % 16 == 0
    assert torch.equal(packed.view(torch.uint8), w.contiguous().view(torch.uint8))


INNER_CONFIG = "swap128x32s3-mx"
PDL_CONFIG = "swap128x32s3-mx-pdl"
INNER_PROVIDERS = (
    "block-fp8:triton-fp8",
    "block-fp8:flashinfer-cutlass",
    "block-fp8:cuda-sm12x",
    f"block-fp8:cute-fp8:{INNER_CONFIG}",
    f"block-fp8:cute-fp8:{PDL_CONFIG}",
)


@pytest.mark.parametrize("provider", (*static.PROVIDERS, *INNER_PROVIDERS))
def test_parse_provider(provider):
    outer, inner, config = dispatch.parse_provider(provider)
    assert outer in ("block-fp8", "torch-scaled-mm", "flashinfer-bmm")
    if inner:
        assert provider == f"{outer}:{inner}" + (f":{config}" if config else "")
    if config:
        assert config in static.CONFIGS


@pytest.mark.parametrize(
    "provider",
    [
        "",
        "cute-fp8",
        "block-fp8:",
        "block-fp8:triton-dequant-bf16",
        "block-fp8:cute-fp8",
        "block-fp8:cute-fp8:missing",
        "block-fp8:triton-fp8:extra",
    ],
)
def test_reject_invalid_inner(provider):
    with pytest.raises(ValueError, match="provider"):
        static.StaticFP8Linear(*tensors(), provider=provider)


def test_dispatch_table_names(monkeypatch):
    dispatch.validate_table()
    for shapes in dispatch.TABLE.values():
        for runs in shapes.values():
            for _, provider in runs:
                _, _, config = dispatch.parse_provider(provider)
                assert config is None or config in static.CONFIGS
    monkeypatch.setitem(dispatch.TABLE, (12, 1, 48), {(256, 128): ((4, "bad"),)})
    with pytest.raises(ValueError, match="provider"):
        dispatch.validate_table()


def fake_inner(monkeypatch, inner, *, fail_n=None):
    calls = []

    class Gemm:
        name = inner

        def __init__(self, weight, scale, **kwargs):
            if weight.shape[0] == fail_n:
                raise RuntimeError("optional dependency/layout unavailable")
            self.shape = tuple(weight.shape)
            self._config = ("fake-sm12x", self.shape)

        def dispatch_key(self, m, *, quantized=False):
            return self.shape, m, quantized

        def _key(self, config, dtype):
            return self.shape, config, dtype

        def run(self, q, scale, out, *config):
            calls.append((q, scale, out, config))
            out.fill_(7)

    factory = {
        "cute-fp8": "CuteFP8Gemm",
        "triton-fp8": "TritonFP8Gemm",
        "flashinfer-cutlass": "FlashInferCutlassGemm",
        "cuda-sm12x": "CudaSm12xGemm",
    }[inner]
    monkeypatch.setattr(block_fp8, factory, Gemm)
    return calls


@pytest.mark.parametrize("route", [None, "block-fp8", *INNER_PROVIDERS])
@pytest.mark.parametrize(
    "slots,k,probe_error,expected",
    [
        ((112, 144), 128, None, "torch-scaled-mm:rowwise"),
        ((100, 156), 128, None, "torch-scaled-mm:rowwise"),
        ((112, 144), 128, "rowwise unsupported", "block-fp8"),
        ((111, 144), 128, None, "block-fp8"),
        ((112, 144), 120, None, "block-fp8"),
    ],
)
def test_unaligned_default_single_launch(cpu, monkeypatch, route, slots, k, probe_error, expected):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: probe_error)
    n = sum(slots)
    table = {} if route is None else {(n, k): ((4, route), (64, route))}
    monkeypatch.setitem(dispatch.TABLE, (12, 1, 48), table)
    if route is not None:
        _, inner, _ = dispatch.parse_provider(route)
        if inner is not None:
            fake_inner(monkeypatch, inner)
    args = tensors(n=n, k=k, scales=2)
    op = static.StaticFP8Linear(*args, slot_rows=slots)
    block_route = route or "block-fp8"
    for m in (1, 4, 5, 64):
        assert op.provider_for(m, torch.bfloat16) == (
            expected if expected.endswith(":rowwise") else block_route
        )
        assert op.provider_for(m, torch.float16) == block_route
    forced = static.StaticFP8Linear(*args, slot_rows=slots, provider=block_route)
    for dtype in market.OUTPUT_DTYPES:
        assert forced.provider_for(3, dtype) == block_route


@pytest.mark.parametrize("route", ["torch-scaled-mm:slots", "flashinfer-bmm:cublas"])
@pytest.mark.parametrize("equal", [False, True])
def test_aligned_default_single_launch(cpu, monkeypatch, route, equal):
    from ctikernels.linear.flashinfer import _fp8_static as fi

    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(fi, "_flashinfer_bmm", lambda _: object())
    monkeypatch.setitem(dispatch.TABLE, (12, 1, 48), {(256, 128): ((4, route), (64, route))})
    w, ws, s = tensors(scales=2)
    if equal:
        ws.fill_(0.25)
    op = static.StaticFP8Linear(w, ws, s, slot_rows=(128, 128))
    forced = static.StaticFP8Linear(w, ws, s, slot_rows=(128, 128), provider=route)
    for dtype in market.OUTPUT_DTYPES:
        for m in (1, 4, 5, 64):
            assert op.provider_for(m, dtype) == (route if equal else "block-fp8")
            assert forced.provider_for(m, dtype) == route


@pytest.mark.parametrize("use_workspace", [False, True])
@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("quantized_warm", [False, True])
def test_prepared_scales_reused_and_retained(
    cpu, monkeypatch, use_workspace, rowwise, quantized_warm
):
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: None)
    monkeypatch.setattr(market, "_has_out", lambda: True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    slots = (112, 144) if rowwise else (128, 128)
    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((4, "block-fp8"),))
    op = static.StaticFP8Linear(*tensors(k=256, scales=2), slot_rows=slots)
    kind = "row" if rowwise else "block"
    columns = 1 if rowwise else 2
    passed, used = [], []

    def quant(x, s, *, block_scale=False, row_scale=False, **kwargs):
        q = torch.zeros(x.shape, dtype=E4M3)
        scale = s.expand(x.shape[0], 2 if block_scale else 1).contiguous()
        return (q, scale) if block_scale or row_scale else q

    def mm(q, weight, **kwargs):
        used.append(kwargs["scale_a"])
        return kwargs["out"].zero_()

    def block_run(q, scale, out, *args, **kwargs):
        passed.append(scale)
        used.append(scale)
        return out.zero_()

    monkeypatch.setattr(quantizer, "quantize", quant)
    if rowwise:
        library = op._resolved_libraries["torch-scaled-mm:rowwise", torch.bfloat16]
        original_run = library.run

        def run(q, out, scale=None, **kwargs):
            passed.append(scale)
            return original_run(q, out, scale, **kwargs)

        monkeypatch.setattr(library, "run", run)
        monkeypatch.setattr(torch, "_scaled_mm", mm)
    else:
        block = op._blocks[0][1]
        monkeypatch.setattr(block, "quantized", block_run)
        monkeypatch.setattr(block, "_run_workspace", block_run)

    op.warm_up(max_m=8, quantized=quantized_warm)
    for name, width in (("block", 2), ("row", 1)):
        buffer = op._static_scales[name]
        assert buffer.shape == (8, width) and buffer.dtype == torch.float32
        assert buffer.is_contiguous() and torch.all(buffer == op.input_scale)
        assert buffer.untyped_storage().data_ptr() != op.input_scale.untyped_storage().data_ptr()
    prepared = op._static_scales[kind]
    workspace = torch.empty(op.workspace_bytes(16) // 4) if use_workspace else None
    for m in (1, 3, 8):
        q = torch.zeros(m, 256, dtype=E4M3)
        out = torch.empty(m, 256, dtype=torch.bfloat16)
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", lambda *a, **kw: pytest.fail("scale copy"))
            assert op.quantized(q, out=out, workspace=workspace) is out
        assert passed[-1].shape == (m, columns)
        assert passed[-1].untyped_storage().data_ptr() == prepared.untyped_storage().data_ptr()
        assert passed[-1].data_ptr() == prepared.data_ptr()
        assert used[-1] is passed[-1]

    # Sizes not covered by warm_up still use the original broadcast/materialization path.
    q = torch.zeros(9, 256, dtype=E4M3)
    out = torch.empty(9, 256, dtype=torch.bfloat16)
    op.quantized(q, out=out, workspace=workspace)
    assert used[-1].shape == (9, columns) and torch.all(used[-1] == op.input_scale)
    assert used[-1].untyped_storage().data_ptr() != prepared.untyped_storage().data_ptr()
    if rowwise:
        assert passed[-1] is None and used[-1].is_contiguous()
    elif use_workspace:
        assert used[-1].untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    else:
        assert used[-1].data_ptr() == op.input_scale.data_ptr() and used[-1].stride(0) == 0

    earlier = {name: weakref.ref(scale) for name, scale in op._static_scales.items()}
    del buffer, prepared
    passed.clear()
    used.clear()
    op.warm_up(max_m=16, quantized=quantized_warm)
    for name, ref in earlier.items():
        assert ref() is not None and any(old is ref() for old in op._retired_scales)
        assert torch.all(ref() == op.input_scale)
        assert op._static_scales[name].shape[0] == 16
        assert (
            op._static_scales[name].untyped_storage().data_ptr()
            != ref().untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("provider", INNER_PROVIDERS)
@pytest.mark.parametrize("slots", [(128, 128), (100, 156)])
def test_direct_inner_invocation(cpu, monkeypatch, provider, slots):
    _, inner, config = dispatch.parse_provider(provider)
    calls = fake_inner(monkeypatch, inner)
    op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=slots, provider=provider)
    assert op.provider == op.provider_for(1000) == provider
    keys = []
    for columns, _ in op._blocks:
        shape = (columns.stop - columns.start, 128)
        if inner == "triton-fp8":
            keys.append((shape, 3, True))
        elif inner == "flashinfer-cutlass":
            keys.append(3)
        elif inner == "cuda-sm12x":
            keys.append(("fake-sm12x", shape))
        else:
            keys.append((shape, config, torch.bfloat16))
    assert op._dispatch_key(3) == (provider, tuple(keys))
    assert op._uses_pdl(3) == bool(config and static.CONFIGS[config].pdl)
    q = torch.zeros(3, 129, dtype=E4M3)[:, 1:]
    buf = torch.full((3, 270), -1, dtype=torch.bfloat16)
    out = buf[:, 3:259]
    assert op.quantized(q, out=out) is out
    assert not cpu.calls  # Named overrides bypass BlockFP8Linear.quantized.
    assert len(calls) == len(op._blocks)
    assert torch.all(out == 7) and torch.all(buf[:, :3] == -1) and torch.all(buf[:, 259:] == -1)
    for (got_q, scale, target, args), (columns, _) in zip(calls, op._blocks, strict=True):
        assert target.data_ptr() == out[:, columns].data_ptr() and target.stride(0) == 270
        if provider.endswith("cuda-sm12x"):  # its JIT GEMM needs a materialized grid
            assert scale.is_contiguous() and torch.all(scale == op.input_scale)
        else:
            assert scale.data_ptr() == op.input_scale.data_ptr() and scale.stride(0) == 0
        assert args == ((config,) if config else ())
        if config:
            assert got_q.data_ptr() % 16 == 0 and got_q.stride(0) == 128
        else:
            assert got_q is q


@pytest.mark.parametrize("inner", ["flashinfer-cutlass", "cuda-sm12x", "cute-fp8"])
@pytest.mark.parametrize("slots,fail_n", [((128, 128), 256), ((100, 156), 156)])
def test_inner_unavailable_fallback(cpu, monkeypatch, inner, slots, fail_n):
    fake_inner(monkeypatch, inner, fail_n=fail_n)
    provider = f"block-fp8:{inner}" + (f":{PDL_CONFIG}" if inner == "cute-fp8" else "")
    monkeypatch.setitem(
        dispatch.TABLE,
        (12, 1, 48),
        {(256, 128): ((4, provider), (8, "torch-scaled-mm"), (16, "block-fp8"))},
    )
    op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=slots)
    assert op.provider_for(1) == op.provider_for(4) == "block-fp8"
    assert op.provider_for(5) == "block-fp8"
    assert not op._uses_pdl(1)
    with pytest.raises(ValueError, match="unavailable"):
        static.StaticFP8Linear(*tensors(scales=2), slot_rows=slots, provider=provider)


def test_sm12x_override_uses_inner_admission(cpu, monkeypatch):
    fake_inner(monkeypatch, "cuda-sm12x")
    monkeypatch.setattr(static, "_cuda_capability", lambda _: (12, 0))
    monkeypatch.setitem(dispatch.TABLE, (12, 0, 48), {(256, 128): ((4, "block-fp8:cuda-sm12x"),)})
    assert static.StaticFP8Linear(*tensors()).provider == "block-fp8:cuda-sm12x"
    assert (
        static.StaticFP8Linear(*tensors(), provider="block-fp8:cuda-sm12x").provider
        == "block-fp8:cuda-sm12x"
    )


@pytest.fixture(params=["library", "dtype", "inner", "device"])
def unavailable_table_choice(cpu, monkeypatch, request):
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(_reroute, "_WARNED", set())
    capability = (12, 1)
    if request.param == "library":
        provider = "flashinfer-bmm:auto"

        def unavailable(*args, **kwargs):
            raise ImportError("test library unavailable")

        monkeypatch.setattr(static, "FlashInferBMM", unavailable)
    elif request.param == "dtype":
        provider = "torch-scaled-mm:rowwise"
        monkeypatch.setattr(market, "rowwise_probe", lambda *a: "test dtype unavailable")
    else:
        provider = "block-fp8:cuda-sm12x"
        fake_inner(monkeypatch, "cuda-sm12x", fail_n=256)
        if request.param == "device":
            capability = (12, 0)
            monkeypatch.setattr(static, "_cuda_capability", lambda _: capability)
    monkeypatch.setitem(
        dispatch.TABLE,
        (*capability, 48),
        {(256, 128): ((4, provider), (8, "block-fp8"))},
    )
    return provider


def test_table_reroute_reason_and_one_log(unavailable_table_choice, caplog):
    caplog.set_level(logging.WARNING, logger=_reroute.__name__)
    for _ in range(2):
        op = static.StaticFP8Linear(*tensors())
        assert op.provider == "block-fp8"
        assert unavailable_table_choice in op.provider.reason
        assert "unavailable" in op.provider.reason
        for dtype in market.OUTPUT_DTYPES:
            if unavailable_table_choice == "torch-scaled-mm:rowwise" and dtype == torch.float16:
                assert op.provider_for(4, dtype) == "torch-scaled-mm:slots"
                assert getattr(op.provider_for(4, dtype), "reason", None) is None
            else:
                assert op.provider_for(4, dtype).reason == op.provider.reason
            assert op.provider_for(5, dtype) == "block-fp8"
            assert getattr(op.provider_for(5, dtype), "reason", None) is None
    messages = [r.getMessage() for r in caplog.records if r.name == _reroute.__name__]
    assert messages == [f"StaticFP8Linear rerouted to block-fp8: {op.provider.reason}"]


def test_table_reroute_engine_checks(unavailable_table_choice, monkeypatch, caplog):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(_reroute.RerouteError, match="StaticFP8Linear: CTI_ENGINE_CHECKS"):
        static.StaticFP8Linear(*tensors())
    assert not caplog.records
    with pytest.raises(ValueError, match="unavailable"):
        static.StaticFP8Linear(*tensors(), provider=unavailable_table_choice)


def test_extra_provider_does_not_change_block_dispatch(cpu, monkeypatch):
    fake_inner(monkeypatch, "cute-fp8")
    w, _, _ = tensors()
    scale = torch.ones(2, 1)
    plain = block_fp8.BlockFP8Linear(w, scale)
    extra = block_fp8.BlockFP8Linear(w, scale, _extra_providers=("cute-fp8",))
    assert plain._cute is None and extra._cute is not None
    assert plain.provider == extra.provider and plain._buckets == extra._buckets
    assert all(plain._dispatch_key(m) == extra._dispatch_key(m) for m in (1, 4, 64, 4096))


@pytest.mark.parametrize("quantized", [False, True])
def test_inner_warm_up_sizes_and_pdl(cpu, monkeypatch, quantized):
    calls = fake_inner(monkeypatch, "cute-fp8")
    plain, pdl = (f"block-fp8:cute-fp8:{cfg}" for cfg in (INNER_CONFIG, PDL_CONFIG))
    monkeypatch.setitem(
        dispatch.TABLE, (12, 1, 48), {(256, 128): ((2, plain), (4, pdl), (20, plain))}
    )
    op = static.StaticFP8Linear(*tensors())
    launches = []

    def quant(x, s, *, block_scale=False, row_scale=False, pdl=False):
        launches.append((x.shape[0], x.stride(0), block_scale, pdl))
        q = torch.zeros(x.shape, dtype=E4M3)
        return (q, s.expand(x.shape[0], 1).contiguous()) if block_scale or row_scale else q

    monkeypatch.setattr(quantizer, "quantize", quant)
    op.warm_up(max_m=17, quantized=quantized, out_row_stride=272)
    sizes = block_fp8._warm_up_sizes("op", 17, None, op._dispatch_key)
    assert {op._dispatch_key(m) for m in sizes} == {op._dispatch_key(m) for m in range(1, 18)}
    assert [(q.shape[0], out.stride(0), out.data_ptr() % 16) for q, _, out, _ in calls] == [
        (m, stride, offset) for m in sizes for stride, offset in ((256, 0), (272, 0), (272, 2))
    ]
    assert {m for m, _, _, pdl in launches if pdl} == {m for m in sizes if 3 <= m <= 4} != set()
    assert {(stride, block) for _, stride, block, pdl in launches if pdl} == {
        (128, False),
        (128, True),
        (129, False),
        (129, True),
    }
    assert op._dispatch_key(2) != op._dispatch_key(3)
    # Forced inner routes also warm the workspace entries (packed scales).
    assert cpu.workspace_warms == [(m, 3, quantized) for m in sizes]


def test_quantizer_pdl_cache_is_separate(cpu, monkeypatch):
    from ctikernels.linear.cute import _block_fp8_provider

    compiles = []
    monkeypatch.setattr(quantizer, "_compiled", {})
    monkeypatch.setattr(quantizer, "_ready", set())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(_block_fp8_provider, "available", lambda _: True)

    def compile_fn(*args):
        compiles.append(args)
        return lambda *args: None

    monkeypatch.setattr(quantizer, "_compile", compile_fn)
    x, s = torch.zeros(3, 128, dtype=torch.bfloat16), torch.ones(1)
    quantizer.quantize(x, s)
    cpu.capturing = True
    with pytest.raises(RuntimeError, match="warm up"):
        quantizer.quantize(x, s, pdl=True)
    cpu.capturing = False
    quantizer.quantize(x, s, pdl=True)
    cpu.capturing = True
    quantizer.quantize(x, s, pdl=True)
    assert [args[-1] for args in compiles] == [False, True]


def test_plain_block_pdl_follows_resolved_inner(cpu, monkeypatch):
    op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=(100, 156))
    for _, block in op._blocks:
        monkeypatch.setattr(block, "provider_for", lambda m: "cute-fp8" if m <= 4 else "triton-fp8")
        monkeypatch.setattr(block, "_cute_config", lambda m: PDL_CONFIG if m <= 2 else INNER_CONFIG)
    assert op._uses_pdl(1) and op._uses_pdl(2)
    assert not op._uses_pdl(3) and not op._uses_pdl(5)
    # A single non-PDL segment keeps the producer on the ordinary launch path.
    monkeypatch.setattr(op._blocks[-1][1], "_cute_config", lambda m: INNER_CONFIG)
    assert not op._uses_pdl(1)


def test_static_quantizer_without_cute_uses_triton(monkeypatch):
    monkeypatch.setattr(static, "cute_available", lambda _: False)
    monkeypatch.setattr(static, "_cuda_capability", lambda _: (9, 0))
    monkeypatch.setattr(quantizer, "quantize", lambda *a, **kw: pytest.fail("CuTe unavailable"))
    static._check_device("op", torch.device("cuda", 0))
    assert static._resolve_quantizer(None, torch.device("cuda", 0)) == "triton"
    with pytest.raises(ValueError, match="cute"):
        static._resolve_quantizer("cute", torch.device("cuda", 0))


@pytest.mark.parametrize("slots", [(1,), (3, 5, 2), (100, 36, 120)])
def test_column_scale_and_alpha_preparation(slots):
    scales = torch.arange(1, len(slots) + 1, dtype=torch.float32) * 0.0137
    columns = market.column_scales(scales, slots)
    expected = torch.tensor([float(scales[i]) for i, rows in enumerate(slots) for _ in range(rows)])
    assert columns.shape == (1, sum(slots)) and columns.is_contiguous()
    assert columns.dtype == torch.float32
    torch.testing.assert_close(columns.flatten(), expected, rtol=0, atol=0)
    # The rowwise factorization gives the same per-column FP32 alpha.
    input_scale = torch.tensor(0.0217)
    torch.testing.assert_close(columns * input_scale, expected[None] * input_scale, rtol=0, atol=0)


@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_alias_resolution_and_no_probe_for_equal(cpu, monkeypatch, equal, supported):
    probes = []

    def probe(device, dtype):
        probes.append((device, dtype))
        return None if supported else "probe rejected rowwise scales"

    monkeypatch.setattr(market, "rowwise_probe", probe)
    w, ws, s = tensors(scales=2)
    if equal:
        ws.fill_(0.3)
    op = static.StaticFP8Linear(w, ws, s, slot_rows=(128, 128), provider="torch-scaled-mm")
    chosen = "slots" if equal or not supported else "rowwise"
    assert op.provider == f"torch-scaled-mm:{chosen}"
    assert len(probes) == (0 if equal else 2)
    assert not op._blocks


def test_dtype_specific_table_probe_fallback(cpu, monkeypatch):
    monkeypatch.setattr(
        market,
        "rowwise_probe",
        lambda device, dtype: "FP16 probe failed" if dtype == torch.float16 else None,
    )
    monkeypatch.setitem(
        dispatch.TABLE, (12, 1, 48), {(256, 128): ((0, "torch-scaled-mm:rowwise"),)}
    )
    op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=(100, 156))
    assert op.provider_for(3) == "torch-scaled-mm:rowwise"
    assert op.provider_for(3, torch.float16) == "block-fp8"
    assert op._blocks
    # FP16 never uses rowwise (precision gate); its per-slot form needs 16-row slots.
    with pytest.raises(ValueError, match="divisible by 16"):
        static.StaticFP8Linear(
            *tensors(scales=2), slot_rows=(100, 156), provider="torch-scaled-mm:rowwise"
        )


@pytest.mark.parametrize("failure", [False, True])
def test_actual_probe_cache(cpu, monkeypatch, failure):
    # Restore the real function replaced by the general CPU fixture.
    source_probe = REAL_ROWWISE_PROBE
    monkeypatch.setattr(market, "_ROWWISE_PROBES", {})
    calls = []

    def mm(a, b, **kwargs):
        calls.append(kwargs)
        assert a.shape == (16, 32) and b.shape == (32, 16)
        assert kwargs["scale_a"].shape == (16, 1)
        assert kwargs["scale_b"].shape == (1, 16)
        if failure:
            raise RuntimeError("rowwise probe reason")

    monkeypatch.setattr(torch, "_scaled_mm", mm)
    for dtype in market.OUTPUT_DTYPES:
        for _ in range(2):
            assert source_probe("cpu", dtype) == ("rowwise probe reason" if failure else None)
    assert len(calls) == 2
    cpu.capturing = True
    assert source_probe("cpu", torch.bfloat16) == ("rowwise probe reason" if failure else None)
    monkeypatch.setattr(market, "_ROWWISE_PROBES", {})
    with pytest.raises(RuntimeError, match="before capture"):
        source_probe("cpu", torch.bfloat16)


@pytest.mark.parametrize("has_out", [False, True])
@pytest.mark.parametrize("layout", ["packed", "strided", "unaligned", "singleton_strided"])
@pytest.mark.parametrize("mode", ["rowwise", "slots"])
def test_library_out_and_rowwise_scales(cpu, monkeypatch, has_out, layout, mode):
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: None)
    monkeypatch.setattr(market, "_has_out", lambda: has_out)
    w, ws, s = tensors(scales=2)
    if mode == "slots":
        ws.fill_(0.25)
    op = static.StaticFP8Linear(w, ws, s, slot_rows=(96, 160), provider=f"torch-scaled-mm:{mode}")
    m = 1 if layout == "singleton_strided" else 3
    stride = 272 if layout in ("strided", "singleton_strided") else 256
    buf = torch.full((m * stride + 2,), -9, dtype=torch.bfloat16)
    out = buf.as_strided((m, 256), (stride, 1), storage_offset=int(layout == "unaligned"))
    calls = []

    def mm(q, weight, **kwargs):
        calls.append(kwargs)
        assert q.data_ptr() % 16 == 0 and q.stride(0) % 16 == 0
        target = kwargs.get("out", torch.empty((m, 256), dtype=out.dtype))
        return target.fill_(7)

    monkeypatch.setattr(torch, "_scaled_mm", mm)
    q = torch.zeros((m, 129), dtype=E4M3)[:, 1:]
    assert op.quantized(q, out=out) is out
    assert len(calls) == 1 and torch.all(out == 7)
    assert (calls[0].get("out") is out) == (has_out and layout == "packed")
    if mode == "rowwise":
        assert calls[0]["scale_a"].shape == (m, 1)
        assert calls[0]["scale_a"].is_contiguous()
        torch.testing.assert_close(calls[0]["scale_b"], market.column_scales(ws, (96, 160)))
    assert buf[-1] == -9


def test_rowwise_call_uses_fused_scale_and_warmup(cpu, monkeypatch):
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: None)
    op = static.StaticFP8Linear(
        *tensors(scales=2), slot_rows=(128, 128), provider="torch-scaled-mm:rowwise"
    )
    outputs, launches = [], []

    def quant(x, s, *, block_scale=False, row_scale=False, pdl=False):
        launches.append((block_scale, row_scale))
        q = torch.zeros(x.shape, dtype=E4M3)
        scale = s.expand(x.shape[0], 1).clone()
        if row_scale:
            outputs.append(scale)
        return (q, scale) if block_scale or row_scale else q

    def mm(q, w, **kwargs):
        assert kwargs["scale_a"] is outputs[-1]
        return kwargs["out"].zero_()

    monkeypatch.setattr(quantizer, "quantize", quant)
    monkeypatch.setattr(torch, "_scaled_mm", mm)
    op.warm_up(batch_sizes=[3])
    assert set(launches) == {(False, False), (True, False), (False, True)}


@pytest.mark.parametrize("backend", ["cublas", "cutlass", "auto"])
@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_flashinfer_calls(cpu, monkeypatch, backend, equal, strided):
    from ctikernels.linear.flashinfer import _fp8_static as fi

    calls = []

    def bmm(a, b, sa, sb, dtype, *, out, backend):
        calls.append((a, b, sa, sb, out, backend))
        assert a.stride(-1) == 1 and b.stride(-2) == 1
        assert sa.numel() == sb.numel() == 1
        assert out.is_contiguous() and out.dtype == dtype
        out.fill_(7)
        return out

    monkeypatch.setattr(fi, "_flashinfer_bmm", lambda _: bmm)
    w, ws, s = tensors(scales=2)
    if equal:
        ws.fill_(0.2)
    op = static.StaticFP8Linear(
        w, ws, s, slot_rows=(128, 128), provider=f"flashinfer-bmm:{backend}"
    )
    buf = torch.full((3, 270 if strided else 256), -9, dtype=torch.bfloat16)
    out = buf[:, 3:259] if strided else buf
    assert op.quantized(torch.zeros(3, 129, dtype=E4M3)[:, 1:], out=out) is out
    assert len(calls) == (1 if equal else 2) and torch.all(out == 7)
    assert (calls[0][4].data_ptr() == out.data_ptr()) == (equal and not strided)
    assert all(call[-1] == backend for call in calls)
    if strided:
        assert torch.all(buf[:, :3] == -9) and torch.all(buf[:, 259:] == -9)


def test_flashinfer_unavailable_and_version_gate(cpu, monkeypatch):
    import sys

    from ctikernels.linear.flashinfer import _fp8_static as fi

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(bmm_fp8=lambda *a: None))
    monkeypatch.setattr(fi.importlib.metadata, "version", lambda _: "unqualified")
    monkeypatch.setitem(dispatch.TABLE, (12, 1, 48), {(256, 128): ((0, "flashinfer-bmm:auto"),)})
    assert static.StaticFP8Linear(*tensors()).provider == "block-fp8"
    with pytest.raises(ValueError, match="qualified"):
        static.StaticFP8Linear(*tensors(), provider="flashinfer-bmm:auto")
    for version in fi.QUALIFIED_VERSIONS:
        monkeypatch.setattr(fi.importlib.metadata, "version", lambda _, v=version: v)
        assert callable(fi._flashinfer_bmm((12, 1)))


def test_quantizer_row_mode_metadata_and_cache(cpu, monkeypatch):
    from ctikernels.linear.cute import _block_fp8_provider

    monkeypatch.setattr(_block_fp8_provider, "available", lambda _: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(quantizer, "_compiled", {})
    monkeypatch.setattr(quantizer, "_ready", set())
    compiles = []

    def compile_fn(*args):
        compiles.append(args)
        return lambda *a: None

    monkeypatch.setattr(quantizer, "_compile", compile_fn)
    x, s = torch.zeros(3, 200, dtype=torch.bfloat16), torch.ones(1)
    for kwargs in ({}, {"block_scale": True}, {"row_scale": True}):
        result = static.quantize_static_fp8(x, s, **kwargs)
        if kwargs.get("row_scale"):
            assert result[1].shape == (3, 1) and result[1].is_contiguous()
    assert len(compiles) == 3
    with pytest.raises(ValueError, match="mutually exclusive"):
        static.quantize_static_fp8(x, s, block_scale=True, row_scale=True)
    with pytest.raises(TypeError, match="row_scale"):
        static.quantize_static_fp8(x, s, row_scale=1)


@pytest.mark.parametrize("hardware", [(12, 0, 170), (12, 0, 188), (12, 1, 48)])
@pytest.mark.parametrize("shape", [(5120, 3072), (14336, 5120)])
def test_workspace_bytes_covers_static_dispatch(hardware, shape):
    from ctikernels.linear import _block_fp8_dispatch
    from ctikernels.linear.triton._block_fp8_provider import TritonFP8Gemm

    n, k = shape
    block = block_fp8.BlockFP8Linear.__new__(block_fp8.BlockFP8Linear)
    block._weight = torch.empty(n, k, dtype=E4M3, device="meta")
    block._capability, block._sms = hardware[:2], hardware[2]
    block._scale_ue8m0 = False
    block._buckets = _block_fp8_dispatch.buckets(hardware[:2], hardware[2], n, k)
    block._triton = TritonFP8Gemm(block._weight, None, capability=hardware[:2], sms=hardware[2])
    library = SimpleNamespace(workspace_bytes=40 * 1024 * 1024, _slots=[(slice(0, n), None, None)])
    library.add_workspace = lambda layout, m: layout.add(
        "library", (library.workspace_bytes,), torch.uint8
    )
    block._flashinfer = library
    op = static.StaticFP8Linear.__new__(static.StaticFP8Linear)
    op._weight = block._weight
    runs = dispatch.buckets(hardware[:2], hardware[2], n, k)
    op._buckets_by_dtype = dict.fromkeys(market.OUTPUT_DTYPES, runs)
    op._blocks = ((slice(0, n), block),)
    op._resolved_libraries = {
        (p, dtype): library for _, p in runs for dtype in market.OUTPUT_DTYPES
    }
    requirements = [op._workspace_bytes(m) for m in range(1, 514)]
    for max_m in (1, 32, 256, 320, 512, 513):
        assert op.workspace_bytes(max_m) == max(requirements[:max_m])
    for m in (1, 33, 320, 513):
        for dtype in market.OUTPUT_DTYPES:
            layout = op._workspace_layout(m, dtype)
            assert layout.slots["q"][1] == m * k
            outer, inner, config = dispatch.parse_provider(op.provider_for(m, dtype))
            if outer == "block-fp8":
                assert layout.slots["scale"][1] == m * ((k + 127) // 128) * 4
                assert (
                    layout.slots["gemm"][1]
                    == block._workspace_layout(
                        m, quantized=True, provider=inner, config=config
                    ).nbytes
                )
            elif outer == "flashinfer-bmm":
                assert layout.slots["library"][1] == library.workspace_bytes


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "dtype",
        "cpu",
        "device",
        "strides",
        "size",
        "alignment",
        "grad",
        "overlap",
        "weight",
        "weight_scale",
        "input_scale",
        "out",
        "block_scale",
    ],
)
def test_static_workspace_validation_before_quantization(cpu, monkeypatch, quantized, bad):
    op = static.StaticFP8Linear(*tensors(), provider="block-fp8")
    workspace = torch.empty(max(op.workspace_bytes(2) // 4, op._weight.numel() // 4) + 4)
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    q = torch.zeros(2, 128, dtype=E4M3)
    if bad != "cpu":
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    if bad == "dtype":
        workspace = workspace.to(torch.float16)
    elif bad == "device":
        op._device = torch.device("meta")
        x, q = x.to("meta"), q.to("meta")
    elif bad == "strides":
        workspace = workspace.expand(2, -1)
    elif bad == "size":
        workspace = workspace[:1]
    elif bad == "alignment":
        workspace = workspace[1:]
    elif bad == "grad":
        workspace.requires_grad_()
    elif bad == "overlap":
        x = workspace.view(torch.bfloat16)[:256].view(2, 128)
        q = workspace.view(E4M3)[:256].view(2, 128)
    out = None
    if bad == "weight":
        op._weight = workspace.view(E4M3)[: op._weight.numel()].view_as(op._weight)
    elif bad in ("weight_scale", "input_scale"):
        value = getattr(op, "_" + bad)
        setattr(op, "_" + bad, workspace[: value.numel()].view_as(value))
    elif bad == "block_scale":
        block = op._blocks[0][1]
        block._scale = workspace[: block._scale.numel()].view_as(block._scale)
    elif bad == "out":
        out = workspace.view(torch.bfloat16)[: 2 * op.out_features].view(2, op.out_features)
    monkeypatch.setattr(
        quantizer, "quantize", lambda *a, **kw: pytest.fail("quantized before validation")
    )
    monkeypatch.setattr(op, "_run_workspace", lambda *a: pytest.fail("launch before validation"))
    with pytest.raises(ValueError, match="workspace"):
        op.quantized(q, out=out, workspace=workspace) if quantized else op(
            x, out=out, workspace=workspace
        )


@pytest.mark.parametrize("max_m", [0, -1, True, 1.5, None])
def test_static_workspace_invalid_bound(cpu, max_m):
    op = static.StaticFP8Linear(*tensors(), provider="block-fp8")
    with pytest.raises(ValueError, match="max_m must be positive"):
        op.workspace_bytes(max_m)


@pytest.mark.parametrize("mode", ["rowwise", "slots"])
@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
@pytest.mark.parametrize("quantized", [False, True])
def test_static_workspace_uses_library_out_views(cpu, monkeypatch, mode, dtype, quantized):
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: None)
    monkeypatch.setattr(market, "_has_out", lambda: True)
    op = static.StaticFP8Linear(
        *tensors(scales=2), slot_rows=(96, 160), provider=f"torch-scaled-mm:{mode}"
    )
    workspace = torch.empty(op.workspace_bytes(6) // 4)
    parent = torch.full((6, 259), -9, dtype=dtype)
    out = parent[:, 1:257]
    x = torch.zeros(3, 2, 128, dtype=dtype).transpose(0, 1)
    q = torch.zeros(6, 129, dtype=E4M3)[:, 1:]
    launches = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))

    def quant(x, s, *, block_scale, row_scale, pdl, outputs):
        q, scale = outputs
        assert x.is_contiguous()
        assert q.untyped_storage().data_ptr() == workspace.data_ptr()
        q.zero_()
        if scale is not None:
            scale.copy_(s)
        return (q, scale) if row_scale or block_scale else q

    def mm(q, w, **kwargs):
        target = kwargs["out"]
        assert target.untyped_storage().data_ptr() == workspace.data_ptr()
        assert q.untyped_storage().data_ptr() == workspace.data_ptr()
        assert target.is_contiguous() and target.data_ptr() % 16 == 0
        launches.append(target.shape)
        return target.fill_(7)

    monkeypatch.setattr(quantizer, "quantize", quant)
    monkeypatch.setattr(torch, "_scaled_mm", mm)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace allocation"))
    monkeypatch.setattr(torch.Tensor, "clone", lambda *a, **kw: pytest.fail("workspace clone"))
    if quantized:
        assert op.quantized(q, out=out, workspace=workspace) is out
    else:
        assert op(x, out=out, workspace=workspace) is out
    assert len(launches) == (1 if op.provider_for(6, dtype).endswith(":rowwise") else 2)
    assert torch.all(out == 7)
    assert torch.all(parent[:, 0] == -9) and torch.all(parent[:, 257:] == -9)


@pytest.mark.parametrize("n,k", [(256, 128), (96, 144), (160, 128), (128, 144)])
@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
def test_flashinfer_workspace_padded_operands(cpu, monkeypatch, n, k, dtype):
    import sys

    from ctikernels.linear._fp8_workspace import Layout, Views
    from ctikernels.linear.flashinfer import _fp8_static as fi

    monkeypatch.setattr(fi, "_flashinfer_bmm", lambda _: object())
    monkeypatch.setattr(fi.FlashInferBMM, "workspace_bytes", property(lambda _: 256))
    weight, ws, scale = tensors(n, k)
    op = fi.FlashInferBMM(weight, ws, scale, (n,), capability=(12, 0), backend="cutlass")
    layout = Layout()
    op.add_workspace(layout, 3)
    workspace = torch.empty(layout.nbytes // 4)
    buffers = Views(layout, workspace)
    q = torch.ones(1, 3, k).to(E4M3)
    out = torch.empty(1, 3, n, dtype=dtype)
    np, kp = (n + 127) // 128 * 128, (k + 127) // 128 * 128
    calls = []

    def gemm(scratch, a, b, sa, sb, target, *policy):
        calls.append(policy)
        assert a.shape == (1, 3, kp) and b.shape == (1, kp, np)
        assert target.shape == (1, 3, np) and target.dtype == dtype
        assert sa.shape == (3, kp // 128) and sb.shape == (np // 128, kp // 128)
        assert torch.all(sa == scale) and torch.all(sb == ws)
        assert torch.equal(a[:, :, :k].view(torch.uint8), q.view(torch.uint8))
        if kp != k:
            assert torch.count_nonzero(a[:, :, k:].view(torch.uint8)) == 0
        if np != n or kp != k:
            assert b.untyped_storage().data_ptr() == workspace.data_ptr()
            assert torch.count_nonzero(b[:, k:, :].view(torch.uint8)) == 0
            assert torch.count_nonzero(b[:, :, n:].view(torch.uint8)) == 0
        for tensor in (scratch, sa, sb):
            assert tensor.untyped_storage().data_ptr() == workspace.data_ptr()
            assert tensor.data_ptr() % 16 == 0
        target.fill_(9)

    base = SimpleNamespace(
        get_gemm_sm120_module=lambda: SimpleNamespace(gemm_fp8_nt_groupwise=gemm)
    )
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", SimpleNamespace(gemm_base=base))
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace allocation"))
    op._cutlass_workspace(q, op._slots[0][1], ws, out, buffers)
    assert calls == [(1, 128, 128, "MN")]
    assert torch.all(out == 9)


@pytest.mark.parametrize("offset", range(0, 64, 4))
@pytest.mark.parametrize("quantized", [False, True])
def test_flashinfer_workspace_cublas_alignment(cpu, monkeypatch, offset, quantized):
    from ctikernels.linear._fp8_workspace import Views
    from ctikernels.linear.flashinfer import _fp8_static as fi

    monkeypatch.setattr(fi, "_flashinfer_bmm", lambda _: object())
    monkeypatch.setattr(fi.FlashInferBMM, "workspace_bytes", property(lambda _: 256))
    op = static.StaticFP8Linear(
        *tensors(256, 144, scales=2), slot_rows=(96, 160), provider="flashinfer-bmm:cublas"
    )
    layout = op._workspace_layout(17, quantized=quantized)
    parent = torch.empty(op.workspace_bytes(17) // 4 + 64)
    workspace = parent[offset : offset + layout.nbytes // 4]
    buffers = Views(layout, workspace)
    scratch = buffers.get("library")
    assert scratch.numel() == 256 and scratch.data_ptr() % 256 == 0
    assert scratch.data_ptr() + scratch.numel() <= workspace.data_ptr() + workspace.numel() * 4
    for name in ("q", "staging"):
        tensor = buffers.get(name)
        assert tensor.data_ptr() % 16 == 0
        assert tensor.stride(0) % 16 == 0


@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
@pytest.mark.parametrize("m", [1, 17])
def test_flashinfer_workspace_matches_bmm_layout(cpu, monkeypatch, dtype, m):
    import sys

    from ctikernels.linear.flashinfer import _fp8_static as fi

    calls = []

    def bmm(q, weight, sa, sb, dtype, *, out, backend):
        calls.append(tuple((t.shape, t.stride()) for t in (q, weight, out)))
        out.fill_(7)

    monkeypatch.setattr(fi, "_flashinfer_bmm", lambda _: bmm)
    monkeypatch.setattr(fi.FlashInferBMM, "workspace_bytes", property(lambda _: 256))
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", SimpleNamespace(gemm_base=object()))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    op = static.StaticFP8Linear(
        *tensors(256, 144, scales=2), slot_rows=(96, 160), provider="flashinfer-bmm:cublas"
    )
    library = op._resolved_libraries["flashinfer-bmm:cublas", dtype]

    def run(q, weight, scale, out, buffers, backends):
        assert backends == ["cublas"]
        assert buffers.get("library").data_ptr() % 256 == 0
        for t in (q, weight, out):
            assert t.data_ptr() % 16 == 0
        assert q.stride(-2) % 16 == weight.stride(-1) % 16 == out.stride(-2) % 16 == 0
        bmm(q, weight, library._input_scale, scale, dtype, out=out, backend="cublas")

    monkeypatch.setattr(library, "_run_workspace", run)
    q = torch.zeros(m, 145, dtype=E4M3)[:, 1:]
    out = torch.empty(m, 259, dtype=dtype)[:, 1:257]
    workspace = torch.empty(op.workspace_bytes(m) // 4)
    op.quantized(q, out=out)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("workspace allocation"))
    op.quantized(q, out=out, workspace=workspace)
    assert calls[:2] == calls[2:]
    assert torch.all(out == 7)


@pytest.mark.parametrize("quantized", [False, True])
def test_static_workspace_outputs_survive_reuse(cpu, monkeypatch, quantized):
    op = static.StaticFP8Linear(*tensors(), provider="block-fp8")
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    workspace = torch.empty(op.workspace_bytes(2) // 4)
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    q = x.to(E4M3)
    calls = []

    def quant(x, s, *, outputs, **kwargs):
        return outputs

    def run(q, scale, out, views):
        views.storage.fill_(len(calls) + 1)
        out.fill_(len(calls) + 1)
        calls.append(out)

    monkeypatch.setattr(quantizer, "quantize", quant)
    monkeypatch.setattr(op, "_run_workspace", run)

    def call():
        return op.quantized(q, workspace=workspace) if quantized else op(x, workspace=workspace)

    first = call()
    saved = first.clone()
    second = call()
    assert torch.equal(first, saved)
    for output in (first, second):
        assert output.untyped_storage().data_ptr() != workspace.untyped_storage().data_ptr()
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (12, 0)])
def test_capability_admission(cpu, monkeypatch, capability):
    monkeypatch.setattr(static, "_cuda_capability", lambda _: capability)
    monkeypatch.setattr(block_fp8, "_cuda_capability", lambda _: capability)
    monkeypatch.setattr(static, "cute_available", lambda _: False)
    REAL_CHECK_DEVICE("test", torch.device("cuda"))
    op = static.StaticFP8Linear(*tensors())
    assert op.provider == "block-fp8" and op.quantizer == "triton"
    with pytest.raises(AttributeError):
        op.quantizer = "cute"
    with pytest.raises(ValueError, match="quantizer"):
        static.StaticFP8Linear(*tensors(), quantizer="other")
    with pytest.raises(ValueError, match="quantizer"):
        static.quantize_static_fp8(
            torch.empty(0, 128, dtype=torch.float16), torch.ones(1), quantizer="other"
        )
    with pytest.raises(ValueError, match="cute"):
        static.StaticFP8Linear(*tensors(), quantizer="cute")


def test_capability_too_old(monkeypatch):
    monkeypatch.setattr(static, "_cuda_capability", lambda _: (8, 6))
    with pytest.raises(RuntimeError, match="compute capability 8.9 or newer"):
        REAL_CHECK_DEVICE("test", torch.device("cuda"))


@pytest.mark.parametrize(
    "bias",
    [
        torch.zeros(255, dtype=torch.bfloat16),
        torch.zeros(1, 256, dtype=torch.bfloat16),
        torch.zeros(256),
        torch.zeros(256, dtype=torch.float16, device="meta"),
        torch.zeros(256, dtype=torch.float16, requires_grad=True),
        torch.zeros(512, dtype=torch.float16)[::2],
    ],
)
def test_bias_validation(cpu, bias):
    with pytest.raises(ValueError, match="bias"):
        static.StaticFP8Linear(*tensors(), bias=bias)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("dtype", market.OUTPUT_DTYPES)
@pytest.mark.parametrize("bias_dtype", market.OUTPUT_DTYPES)
@pytest.mark.parametrize("strided", [False, True])
def test_bias_after_gemm(cpu, monkeypatch, quantized, workspace, dtype, bias_dtype, strided):
    bias = torch.linspace(-0.3, 0.3, 256).to(bias_dtype)
    op = static.StaticFP8Linear(*tensors(), bias=bias, quantizer="triton")
    assert op.bias is bias
    with pytest.raises(AttributeError):
        op.bias = None
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    calls = []

    def quant(x, s, *, outputs=None, block_scale=False, row_scale=False, **kwargs):
        calls.append((block_scale, row_scale, kwargs))
        return outputs if outputs else (x.to(E4M3), s.expand(x.shape[0], 1).contiguous())

    monkeypatch.setattr(static, "_quantizer", lambda choice: quant if choice == "triton" else None)
    monkeypatch.setattr(op, "_run", lambda q, scale, out: out.fill_(0.123))
    monkeypatch.setattr(op, "_run_workspace", lambda q, scale, out, views: out.fill_(0.123))
    x = torch.zeros(3, 128, dtype=E4M3 if quantized else dtype)
    parent = torch.full((3, 259 if strided else 256), -19, dtype=dtype)
    out = parent[:, 1:257] if strided else parent
    scratch = torch.empty(op.workspace_bytes(3) // 4) if workspace else None
    call = op.quantized if quantized else op
    assert call(x, out=out, workspace=scratch) is out
    expected = torch.full((3, 256), 0.123, dtype=dtype).add_(bias)
    assert torch.equal(out, expected)
    assert len(calls) == int(not quantized)
    if strided:
        assert torch.all(parent[:, 0] == -19) and torch.all(parent[:, 257:] == -19)


def test_triton_static_quantizer_interpreter():
    pytest.importorskip("triton")
    # By module name from the repository root: the file's package-relative imports need it.
    result = run_child(
        [sys.executable, "-m", __spec__.name],
        env=os.environ | {"TRITON_INTERPRET": "1"},
        cwd=Path(__file__).resolve().parents[2],
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _interpreter_quantizer_cases():
    from itertools import product

    from ctikernels.linear.triton import _fp8_static_quant as portable

    for dtype, k, m, layout in product(
        market.OUTPUT_DTYPES,
        (1, 7, 129, 300),
        (0, 1, 3),
        ("packed", "padded", "broadcast", "overlap"),
    ):
        values = torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                -0.0,
                0.0,
                1.0625,
                1.0625 - 2**-7,
                1.0625 + 2**-7,
                2**-10,
                -(2**-10),
                500.0,
                -500.0,
            ],
            dtype=dtype,
        )
        storage = values.repeat((max(1, m) * (k + 17) + 11) // 12 + 1)
        stride = {"packed": k, "padded": k + 17, "broadcast": 0, "overlap": k // 2}[layout]
        x = storage.as_strided((m, k), (stride, 1))
        for s in (torch.tensor(1.0), torch.full((17,), 0.0137)[::17]):
            ref = (x.float() / s).clamp(-448, 448).to(E4M3)
            nan = torch.isnan(ref.float())
            for block, row in ((False, False), (True, False), (False, True)):
                result = portable.quantize(x, s, block_scale=block, row_scale=row, pdl=True)
                q, scale = result if block or row else (result, None)
                assert q.shape == (m, k) and q.is_contiguous()
                assert torch.equal(torch.isnan(q.float()), nan)
                assert torch.all((q.view(torch.uint8)[nan] & 127) == 127)
                assert torch.equal(q.view(torch.uint8)[~nan], ref.view(torch.uint8)[~nan])
                if block or row:
                    assert scale.shape == (m, 1 if row else (k + 127) // 128)
                    assert torch.equal(scale, s.expand_as(scale))
                target = torch.empty_like(q), torch.empty_like(scale) if scale is not None else None
                written = portable.quantize(x, s, block_scale=block, row_scale=row, outputs=target)
                assert (written[0] if block or row else written) is target[0]
                assert torch.equal(target[0].view(torch.uint8), q.view(torch.uint8))
                if block or row:
                    assert torch.equal(target[1], scale)

    for dtype in market.OUTPUT_DTYPES:
        x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).reshape(256, 256)
        s = torch.tensor(0.0137)
        q = portable.quantize(x, s)
        ref = (x.float() / s).clamp(-448, 448).to(E4M3)
        nan = torch.isnan(ref.float())
        assert torch.equal(torch.isnan(q.float()), nan)
        assert torch.equal(q.view(torch.uint8)[~nan], ref.view(torch.uint8)[~nan])


@pytest.mark.parametrize("choice", [None, "cute", "triton"])
def test_quantizer_choice_and_warm_up(cpu, monkeypatch, choice):
    calls = []

    def quant(x, s, *, block_scale=False, row_scale=False, **kwargs):
        calls.append((x.stride(0), block_scale, row_scale))
        q = x.to(E4M3)
        scale = s.expand(x.shape[0], 1).contiguous()
        return (q, scale) if block_scale or row_scale else q

    resolved = choice or "cute"

    def lookup(name):
        assert name == resolved
        return quant

    monkeypatch.setattr(static, "_quantizer", lookup)
    op = static.StaticFP8Linear(*tensors(), quantizer=choice)
    assert op.quantizer == resolved
    op.warm_up(max_m=3)
    op.quantize(torch.zeros(3, 128, dtype=torch.bfloat16))
    assert {(block, row) for _, block, row in calls} == {
        (False, False),
        (True, False),
        (False, True),
    }
    assert {stride for stride, _, _ in calls} == {128, 129}


def test_triton_capture_readiness(cpu, monkeypatch):
    pytest.importorskip("triton")
    from ctikernels.linear.triton import _fp8_static_quant as portable

    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **kw: calls.append((grid, a, kw))

    monkeypatch.setattr(portable, "_ready", set())
    monkeypatch.setattr(portable, "_quantize", Kernel())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    x = torch.zeros(3, 128, dtype=torch.bfloat16)
    s = torch.ones(17)[::17]
    cpu.capturing = True
    portable.quantize(x[:0], s)
    assert not calls
    with pytest.raises(RuntimeError, match="warm up.*before capture"):
        portable.quantize(x, s)
    cpu.capturing = False
    portable.quantize(x, s)
    cpu.capturing = True
    portable.quantize(torch.zeros(5, 131, dtype=x.dtype)[:, 1:], s, pdl=True)
    portable.quantize(x[:1].expand(7, -1), s)
    assert len(calls) == 3
    for source, kwargs in ((x.half(), {}), (x, {"block_scale": True}), (x, {"row_scale": True})):
        with pytest.raises(RuntimeError, match="warm up.*before capture"):
            portable.quantize(source, s, **kwargs)
    with pytest.raises(ValueError, match="mutually exclusive"):
        portable.quantize(x, s, block_scale=True, row_scale=True)


def test_triton_specialization_classes():
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime.jit import create_function_from_signature

    from ctikernels.linear.triton import _fp8_static_quant as portable

    kernel = portable._quantize
    binder = create_function_from_signature(
        kernel.signature, kernel.params, CUDABackend(GPUTarget("cuda", 89, 32))
    )
    x = torch.zeros(512, dtype=torch.bfloat16)
    s = torch.ones(17)
    q = torch.zeros(512, dtype=E4M3)
    scale = torch.zeros(512)
    for mode in ("plain", "row", "block"):
        keys = [
            binder(
                x[offset:],
                s[offset:],
                q[offset:],
                scale[offset:] if mode != "plain" else None,
                m,
                k,
                stride,
                mode,
            )[1]
            for offset, m, k, stride in ((0, 1, 128, 128), (1, 3, 7, 9), (0, 16, 129, 0))
        ]
        assert keys[0] == keys[1] == keys[2]
        large = binder(x, s, q, scale if mode != "plain" else None, 2**31, 128, 128, mode)[1]
        assert large == keys[0]


@pytest.mark.parametrize("n,k", [(1000, 128), (256, 1000), (256, 129), (256, 128)])
def test_generic_shapes_do_not_probe_libraries(cpu, monkeypatch, n, k):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((0, "block-fp8"),))
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: pytest.fail("generic library probe"))
    op = static.StaticFP8Linear(*tensors(n, k, scales=2), slot_rows=(n // 2, n - n // 2))
    assert op.provider == "block-fp8" and getattr(op.provider, "reason", None) is None


@pytest.mark.parametrize("checks", [False, True])
def test_rowwise_constructor_reroute(cpu, monkeypatch, checks):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", str(int(checks)))
    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((4, "block-fp8"),))

    def unavailable(*args, **kwargs):
        raise RuntimeError("test rowwise constructor unavailable")

    monkeypatch.setattr(market.TorchScaledMM, "__init__", unavailable)
    if checks:
        with pytest.raises(_reroute.RerouteError, match="test rowwise constructor unavailable"):
            static.StaticFP8Linear(*tensors(scales=2), slot_rows=(112, 144))
    else:
        op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=(112, 144))
        assert "test rowwise constructor unavailable" in op.provider_for(3).reason


def test_promoted_route_retains_failure_reason(cpu, monkeypatch):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "0")
    monkeypatch.setattr(dispatch, "buckets", lambda *a: ((4, "flashinfer-bmm:auto"),))
    monkeypatch.setattr(market, "rowwise_probe", lambda *a: None)

    def unavailable(*args, **kwargs):
        raise ImportError("test missing library")

    monkeypatch.setattr(static, "FlashInferBMM", unavailable)
    op = static.StaticFP8Linear(*tensors(scales=2), slot_rows=(112, 144))
    assert op.provider_for(3) == "torch-scaled-mm:rowwise"
    assert "test missing library" in op.provider_for(3).reason


@pytest.mark.parametrize("choice", ["other", "", 1, [], {}, torch.ones(2)])
def test_invalid_quantizer(cpu, choice):
    with pytest.raises(ValueError, match="quantizer"):
        static.StaticFP8Linear(*tensors(), quantizer=choice)
    with pytest.raises(ValueError, match="quantizer"):
        static.quantize_static_fp8(
            torch.empty(0, 128, dtype=torch.float16), torch.ones(1), quantizer=choice
        )


if __name__ == "__main__":
    _interpreter_quantizer_cases()
