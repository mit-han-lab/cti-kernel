# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only product port policy, layout routing and symbolic capture admission."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels._runtime.tables import DeviceClass
from ctikernels.gdn import _cute_prefill, inference
from ctikernels.gdn import _triton_prefill as dispatch
from ctikernels.gdn._contract import Geometry, HashedTuple


class FakePort:
    def __init__(self, hk, hv, device):
        self.geometry = Geometry(hk, hv)
        self.compiled = set()
        self.calls = []
        self.rows = []
        self.warms = []

    def plan(self, tokens, sequences):
        return (2, (tokens + 1) // 2) if sequences == 1 and tokens >= 1024 else (1, tokens)

    def variants(self, tokens, sequences):
        return ("mn", "seg") if self.plan(tokens, sequences)[0] >= 2 else ("full",)

    def supports(self, projected, out=None):
        return (
            projected.data_ptr() % 16 == 0
            and projected.stride(0) % 8 == 0
            and projected.stride(1) == 1
        )

    def warm_up(self, row_stride, *, variants):
        self.warms.append((row_stride, variants))
        keys = {(row_stride, v) for v in variants}
        count = len(keys - self.compiled)
        self.compiled |= keys
        return count

    def __call__(self, projected, *args, out=None, capture_rows=None):
        self.calls.append((projected, args, out))
        self.rows.append(capture_rows)
        self.compiled.update(
            (projected.stride(0), v) for v in self.variants(projected.shape[0], args[5].numel())
        )
        return (
            out
            if out is not None
            else projected.new_empty(projected.shape[0], self.geometry.value_width)
        )


class FakeCute(_cute_prefill.Launcher):
    def __call__(self, geometry, projected, *args, **kwargs):
        self.calls.append((projected, args, kwargs))
        return (
            kwargs.get("out")
            if kwargs.get("out") is not None
            else projected.new_empty(projected.shape[0], geometry.value_width)
        )


class FakeTriton:
    def __init__(self, device, geometry):
        self.geometry = geometry
        self.calls = []

    def plan(self, projected, count):
        return ("triton", HashedTuple(((), ("triton", projected.stride(0)))), None)

    def variant(self, projected, *tensors):
        return ("triton", projected.stride(0))

    def run(self, plan, projected, *args):
        return self(None, projected, out=args[9])

    def __call__(self, geometry, projected, *args, out=None, capture_rows=None):
        self.calls.append(projected)
        width = self.geometry.value_width
        return projected.new_empty(projected.shape[0], width) if out is None else out


SMS = {dispatch.RTX_5090: 170, dispatch.RTX_PRO_6000: 188, dispatch.GB10: 48}
# Lower token bound of the shipped N = 1 port band per Hv (RTX 5090 table); Hv 32 and 48 run
# the port at every T and N on both devices.
PORT_FROM = {16: 6144, 24: 3072}


def install(monkeypatch, kind, capability, rows):
    """Make ``rows`` ({(tokens, sequences, hk, hv): provider}) the device's provider table."""
    bucket = dispatch.PROVIDER.bucket
    table = {
        bucket(tokens=t, sequences=n, k_heads=hk, v_heads=hv): {"provider": choice}
        for (t, n, hk, hv), choice in rows.items()
    }
    name = DeviceClass(*capability, SMS[kind]).name
    monkeypatch.setitem(dispatch.PROVIDER._tables, name, table)


def all_port():
    """The port on every bucket of the port geometries."""
    edges = dispatch.PROVIDER.dims[1].edges
    return {(16, n, hk, hv): "port" for hk, hv in dispatch.PORT_GEOMETRIES for n in edges}


def shipped(monkeypatch):
    """Read the shipped table files again."""
    monkeypatch.setattr(dispatch.PROVIDER, "_tables", {})


@pytest.fixture
def factory(monkeypatch):
    # Mechanism tests run with an all-port table; shipped() restores the measured ones.
    for kind in (dispatch.RTX_5090, dispatch.RTX_PRO_6000):
        install(monkeypatch, kind, (12, 0), all_port())
    capturing = [False]
    built = []
    monkeypatch.delenv("CTI_GDN_PREFILL_CUTE", raising=False)
    monkeypatch.delenv("CTI_GDN_PREFILL_PORT", raising=False)
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "0")
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(multi_processor_count=170)
    )
    monkeypatch.setattr(inference, "cuda_device", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(inference, "check_prefill", lambda g, d, x, *a: (x.shape[0], a[5].numel()))

    def port(hk, hv, device):
        result = FakePort(hk, hv, device)
        built.append(result)
        return result

    def cute(device, geometry):
        result = FakeCute(device, object(), geometry)
        result.calls = []
        result._compile = lambda key: object()
        return result

    monkeypatch.setattr(dispatch, "_load_port", port)
    monkeypatch.setattr(_cute_prefill, "load", cute)
    monkeypatch.setattr(dispatch, "Prefill", FakeTriton)

    def make(hv=32, kind=dispatch.RTX_5090, capability=(12, 0), hk=None):
        hk = (8 if hv in (16, 24) else 16) if hk is None else hk
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
        props = SimpleNamespace(multi_processor_count=SMS[kind])
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: props)
        kernel = dispatch.TritonPrefill(kind, torch.device("cpu"), hv, hk=hk)
        monkeypatch.setattr(inference, "_prefill_kernel", lambda *a, **k: kernel)
        return inference.GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cpu")

    make.built, make.capturing = built, capturing
    return make


def arguments(op, tokens=16, count=1, *, offset=0, padding=0):
    width = op.geometry.proj_width
    x = torch.empty(tokens * (width + padding) + offset, dtype=torch.bfloat16)[offset:].view(
        tokens, width + padding
    )[:, :width]
    return (
        x,
        *(torch.empty(1) for _ in range(4)),
        torch.empty(count + 1, dtype=torch.int32),
        torch.empty(count, dtype=torch.int32),
        torch.empty(count, dtype=torch.bool),
        torch.empty(1),
        torch.empty(1),
    )


@pytest.mark.parametrize("kind", [dispatch.RTX_5090, dispatch.RTX_PRO_6000])
@pytest.mark.parametrize("hv", [16, 24, 32, 48])
def test_default_and_tp2_bind_both_paths(factory, monkeypatch, kind, hv):
    op = factory(hv, kind)
    assert op.provider == "cute" and op.supports_state_capture
    assert op._kernel._port.geometry == op.geometry
    assert op.selected_variant(16, 1) == "port"
    args = arguments(op)
    out = torch.empty(16 * op.geometry.value_width + 1, dtype=torch.bfloat16)[1:].view(16, -1)
    assert op(*args, out=out) is out  # Misaligned out is the port's responsibility.
    assert len(op._kernel._port.calls) == 1
    assert op._guard._eager == {((), ("port", args[0].stride(0), ("full",)))}


@pytest.mark.parametrize("forced", [None, "1"])
@pytest.mark.parametrize("hk, hv", [(16, 16), (4, 16), (8, 32), (8, 48), (2, 6)])
def test_port_never_binds_outside_its_geometries(factory, monkeypatch, forced, hk, hv):
    # GP1 built the port with Hk inferred from Hv; only its four (Hk, Hv) pairs bind it.
    if forced is not None:
        monkeypatch.setenv("CTI_GDN_PREFILL_PORT", forced)
    op = factory(hv, hk=hk)
    assert op.provider == "cute" and op._kernel._port is None and not factory.built
    assert op.selected_variant(16384, 1) == "cute"


SHAPES = [
    (16, 1), (64, 1), (128, 1), (512, 1), (1024, 1), (2048, 1), (4096, 1), (8192, 1),
    (16384, 1), (262144, 1), (128, 2), (512, 4), (8192, 2), (8192, 4), (8192, 8),
]  # fmt: skip


@pytest.mark.parametrize("kind", [dispatch.RTX_5090, dispatch.RTX_PRO_6000])
@pytest.mark.parametrize("hv", [16, 24, 32, 48])
def test_shipped_table(factory, monkeypatch, kind, hv):
    # Hv 32 and 48: the port at every T and N on both devices. RTX 5090 Hv 16 and 24: the port
    # on their N = 1 bands. Everything else runs v1.
    shipped(monkeypatch)
    op = factory(hv, kind)
    for tokens, count in SHAPES:
        if hv in (32, 48):
            expected = "port"
        elif kind == dispatch.RTX_5090:
            expected = "port" if count == 1 and tokens >= PORT_FROM[hv] else "cute"
        else:
            expected = "cute"
        assert op.selected_variant(tokens, count) == expected, (tokens, count)


def test_triton_rows_bind_and_route_triton(factory, monkeypatch):
    install(monkeypatch, dispatch.RTX_PRO_6000, (12, 0), {(1024, 1, 16, 32): "triton"})
    op = factory(32, dispatch.RTX_PRO_6000)
    assert op.selected_variant(1024, 1) == "triton" and op.selected_variant(512, 1) == "cute"
    op(*arguments(op, 1024))
    op(*arguments(op, 1024), capture_rows=torch.ones(1, dtype=torch.int32))
    assert len(op._kernel._triton.calls) == 2 and not op._kernel._cute.calls
    assert op._guard._eager == {((), ("triton", op.geometry.proj_width))}
    # A geometry without triton rows never binds it; nor does disabled CuTe (no dispatch).
    assert factory(48, dispatch.RTX_PRO_6000)._kernel._triton is None


def test_bucket_inheritance_never_crosses_sequences_or_geometry(factory, monkeypatch):
    rows = {(1535, 1, 16, 32): "port", (4096, 1, 16, 32): "cute", (64, 2, 16, 32): "port"}
    install(monkeypatch, dispatch.RTX_5090, (12, 0), rows)
    op = factory()
    expected = {
        (1024, 1): "cute", (1025, 1): "port", (3071, 1): "port", (3072, 1): "cute",
        (65536, 1): "cute", (32, 2): "cute", (33, 2): "port", (262144, 2): "port",
        (1535, 3): "cute", (1535, 0): "cute",
    }  # fmt: skip
    for (tokens, count), choice in expected.items():
        assert op.selected_variant(tokens, count) == choice, (tokens, count)
    assert factory(48).selected_variant(1535, 1) == "cute"


@pytest.mark.parametrize("kind", [dispatch.RTX_5090, dispatch.RTX_PRO_6000])
@pytest.mark.parametrize("hv", [16, 24])
def test_packed_batches_never_inherit_a_port_choice(factory, monkeypatch, kind, hv):
    # Hv 16 and 24 (Hv 32 and 48 run the port on every packed batch): [6000,1500,500,192] shares
    # (T, N) = (8192, 4) with [2048]x4 and the host never sees the lengths, so no N >= 2 batch
    # inherits an N = 1 port row.
    shipped(monkeypatch)
    op = factory(hv, kind)
    for tokens, count in (
        (512, 4),
        (1025, 5),
        (8192, 4),
        (0, 2),
        (64, 2),
        (262144, 2),
        (262144, 64),
    ):
        assert op.selected_variant(tokens, count) != "port", (tokens, count)


@pytest.mark.parametrize("offset,padding", [(1, 0), (0, 1)])
def test_call_and_guard_route_layouts_to_v1(factory, offset, padding):
    op = factory()
    rows = torch.ones(1, dtype=torch.int32)
    op(*arguments(op, offset=offset, padding=padding), capture_rows=rows)
    assert not op._kernel._port.calls
    ((_, _, keywords),) = op._kernel._cute.calls
    assert keywords["capture_rows"] is rows
    assert all(shape == () and variant[0] == "cute" for shape, variant in op._guard._eager)


@pytest.mark.parametrize("capture", [False, True])
def test_captures_never_change_dispatch(factory, capture):
    op = factory()
    rows = torch.ones(1, dtype=torch.int32) if capture else None
    op(*arguments(op), capture_rows=rows)
    assert not op._kernel._cute.calls and op._kernel._port.rows == [rows]
    assert op._guard._eager == {((), ("port", op.geometry.proj_width, ("full",)))}


@pytest.mark.parametrize("mode,expected", [("0", "cute"), ("1", "port")])
def test_environment_is_construction_time_policy(factory, monkeypatch, mode, expected):
    install(monkeypatch, dispatch.RTX_5090, (12, 0), {})
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", mode)
    op = factory()
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "1" if mode == "0" else "0")
    assert op.selected_variant(16, 1) == expected
    op(*arguments(op, offset=1))
    assert len(op._kernel._cute.calls) == 1
    assert len(factory.built) == (mode == "1")


@pytest.mark.parametrize("kind,cc", [(dispatch.GB10, (12, 1)), (dispatch.RTX_5090, (12, 1))])
def test_exclusions_never_construct_port(factory, kind, cc):
    op = factory(kind=kind, capability=cc)
    assert factory.built == []
    assert op.selected_variant(16, 1) == "cute"


def test_unavailable_port_retains_v1(factory, monkeypatch):
    monkeypatch.setattr(dispatch, "_load_port", lambda *a: None)
    op = factory()
    assert op.provider == "cute" and op.selected_variant(16, 1) == "cute"
    op(*arguments(op))
    assert len(op._kernel._cute.calls) == 1


@pytest.mark.parametrize("failure", [ImportError, RuntimeError])
def test_optional_import_and_construction_fail_silently(monkeypatch, failure):
    import sys

    def fail(*args):
        raise failure("unavailable")

    monkeypatch.setitem(
        sys.modules, "ctikernels.gdn.chunk_ws.prefill", SimpleNamespace(PortPrefill=fail)
    )
    assert dispatch._load_port(16, 32, torch.device("cpu")) is None


@pytest.mark.parametrize("tokens,cp", [(512, False), (2048, True)])
def test_warmup_registers_full_cp_and_v1_for_misalignment(factory, tokens, cp):
    op = factory()
    stride = op.geometry.proj_width
    assert op.warm_up(tokens, max_sequences=4) > 0
    assert op._kernel._port.warms == [(stride, ("full", "mn", "seg") if cp else ("full",))]
    assert ((), ("port", stride, ("full",))) in op._guard._eager
    assert (((), ("port", stride, ("mn", "seg"))) in op._guard._eager) == cp
    # v1 compiles each variant with its capturing twin; guard entries carry no capture count.
    assert {key[6] for key in op._kernel._cute.compiled} == {0, 1}
    assert all(v[-1][-1] == 0 for _, v in op._guard._eager if v[0] == "cute")
    assert op.warm_up(tokens, max_sequences=4) == 0


def test_warmup_respects_narrow_table_and_cp_breakpoints(factory, monkeypatch):
    # At max_tokens the policy is v1; the reachable CP band still needs warming.
    rows = {(1535, 1, 16, 32): "port", (2048, 1, 16, 32): "cute"}
    install(monkeypatch, dispatch.RTX_5090, (12, 0), rows)
    op = factory()
    op._kernel._port.plan = lambda t, n: (2, 64) if n == 1 and t > 1024 else (1, t)
    op.warm_up(4096, max_sequences=2)
    # Only N = 1 at 1025-1535 tokens reaches the port, and that plans CP: no full variant.
    stride = op.geometry.proj_width
    assert op._kernel._port.warms == [(stride, ("mn", "seg"))]
    assert ((), ("port", stride, ("full",))) not in op._guard._eager
    op = factory()
    op.warm_up(1024, max_sequences=2)
    assert op._kernel._port.warms == []
    op.warm_up(4096, row_stride=op.geometry.proj_width + 1)
    assert op._kernel._port.warms == []


def test_guard_rejects_unwarmed_port_then_shares_symbolic_shapes(factory, monkeypatch):
    op = factory()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    factory.capturing[0] = True
    with pytest.raises(RuntimeError, match="once eagerly"):
        op(*arguments(op))
    assert not op._kernel._port.calls
    factory.capturing[0] = False
    op(*arguments(op))
    factory.capturing[0] = True
    op(*arguments(op, 128, 3))
    with pytest.raises(RuntimeError, match="once eagerly"):
        op(*arguments(op, 2048))
    with pytest.raises(RuntimeError, match="capture"):
        op.warm_up(2048)
    factory.capturing[0] = False
    op.warm_up(2048, max_sequences=4)
    factory.capturing[0] = True
    op(*arguments(op, 2048))
    op(*arguments(op, 3072))  # CP segment length and T are symbolic too.


@pytest.mark.parametrize("tokens,count", [(0, 1), (0, 3), (1, 3)])
def test_empty_rows_admitted_by_both_shape_policies(factory, monkeypatch, tokens, count):
    op = factory()
    assert op.selected_variant(tokens, count) == "port"
    assert _cute_prefill.chunk_capacity(tokens, count) == min(tokens, count)
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")
    assert factory().selected_variant(tokens, count) == "cute"


@pytest.mark.parametrize("tokens,count", [(0, 1), (0, 5), (1, 5)])
def test_product_validation_accepts_empty_rows_but_training_keeps_its_bound(tokens, count):
    from ctikernels.gdn import _contract

    from .test_gdn import _prefill_arguments

    args = _prefill_arguments(8, 16, tokens=tokens, sequences=count, slots=3)
    assert _contract.check_prefill(Geometry(8, 16), torch.device("meta"), *args.values()) == (
        tokens,
        count,
    )
    with pytest.raises(ValueError, match="more sequences"):
        _contract.check_cu_seqlens(torch.device("meta"), args["cu_seqlens"], tokens)


@pytest.mark.parametrize("tokens,count", [(1, 5), (3, 5), (65, 2)])
@pytest.mark.parametrize("overlap", [0, 1])
def test_empty_v1_launch_grids_are_nonzero(tokens, count, overlap):
    # T=0 launches no token kernel (tests/modules/test_gdn_general.py); with T>0, empty
    # sequences still leave every launched grid nonzero.
    from .test_gdn_cute_prefill import test_prefill_pdl_launch_chain_and_kernel_boundaries

    test_prefill_pdl_launch_chain_and_kernel_boundaries(
        overlap, 0, 0, 0, 2, count, o_split=1, tokens=tokens
    )
    tune = _cute_prefill.read_tune(f"overlap={overlap},o_split=1")
    assert all(
        min(row["grid"]) > 0 for row in _cute_prefill.launch_info(16, 48, tokens, count, tune)
    )


def test_nonmonotone_cp_plan_is_not_inferred_only_from_max_tokens(factory):
    op = factory()
    op._kernel._port.plan = lambda t, n: (2, 64) if n == 1 and 1024 <= t <= 1280 else (1, t)
    op.warm_up(4096, max_sequences=1)
    assert op._kernel._port.warms == [(op.geometry.proj_width, ("full", "mn", "seg"))]


def _engine_warm_up(op, max_tokens=8192, max_sequences=64, **options):
    """The engine's warm-up: doubling token bounds up to max_tokens (models/base.py)."""
    bounds = [1 << p for p in range((max_tokens - 1).bit_length())] + [max_tokens]
    return [op.warm_up(t, max_sequences=min(t, max_sequences), **options) for t in bounds]


def test_shipped_band_warms_only_reachable_port_variants(factory, monkeypatch):
    # RTX 5090, 8x24, the shipped N = 1 band from 3072 tokens and the real CP planner on
    # 170 SMs: every band T plans CP, so the full variant is never compiled.
    from ctikernels.gdn.chunk_ws._cp import cp_plan

    shipped(monkeypatch)
    op = factory(24)
    port = op._kernel._port
    port.plan = lambda t, n: cp_plan(t, 24, 170) if n == 1 and t else (1, t)
    assert all(cp_plan(t, 24, 170)[0] >= 2 for t in range(3072, 8193))
    assert op._kernel.port_variants(8192, 64) == {("mn", "seg")}
    _engine_warm_up(op)
    stride = op.geometry.proj_width
    # From the 4096-token step on (8192 compiles nothing new).
    assert port.warms == [(stride, ("mn", "seg"))] * 2
    assert port.compiled == {(stride, "mn"), (stride, "seg")}
    assert ((), ("port", stride, ("mn", "seg"))) in op._guard._eager
    assert not any(v[0] == "port" and v[2] == ("full",) for _, v in op._guard._eager)


def test_warm_up_port_opt_out(factory, monkeypatch):
    shipped(monkeypatch)
    op = factory(32)
    _engine_warm_up(op, port=False)
    assert op._kernel._port.warms == []
    assert not any(v[0] == "port" for _, v in op._guard._eager)
    # Dispatch is unchanged: a port-band call is still the port, and unwarmed in capture.
    assert op.selected_variant(4096, 1) == "port"
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    factory.capturing[0] = True
    with pytest.raises(RuntimeError, match="once eagerly"):
        op(*arguments(op, 4096))
    factory.capturing[0] = False
    with pytest.raises(TypeError, match="port must be a bool"):
        op.warm_up(16, port=1)


@pytest.mark.parametrize("hv", [16, 24])
def test_load_binds_tp2_port_without_frozen_triton(factory, monkeypatch, hv):
    factory(hv)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dispatch, "device_type", lambda d: dispatch.RTX_5090)
    kernel = dispatch.load(Geometry(8, hv), torch.device("cpu"))
    assert kernel._port.geometry == Geometry(8, hv)
    assert kernel.selected_variant(16, 1) == "port"
    monkeypatch.setattr(dispatch, "device_type", lambda d: None)
    assert dispatch.load(Geometry(8, hv), torch.device("cpu")) is None
