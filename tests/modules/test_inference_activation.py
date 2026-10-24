# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import contextlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.norm_act_reference import error_metrics, market_formulas, reference
from ctikernels.activation import inference as act
from ctikernels.activation._dispatch import TABLE, Config, config_for, device_class, tune_configs
from ctikernels.activation.cute import _provider

try:
    import triton
except ImportError:  # the CPU CI env has no Triton
    triton = None

# These tests patch the Triton SiLU module, which imports Triton.
requires_triton = pytest.mark.skipif(triton is None, reason="requires Triton")


@pytest.fixture(autouse=True)
def qualified_flashinfer(monkeypatch):
    monkeypatch.setattr(act._flashinfer_version, "installed_version", lambda: "0.7.0.post1")
    monkeypatch.setattr(act._flashinfer_version, "importable", lambda: True)


@pytest.mark.parametrize(
    "cap,sms,kind",
    [
        ((12, 0), 170, "rtx5090"),
        ((12, 0), 188, "pro6000"),
        ((12, 1), 48, "gb10"),
        ((12, 0), 148, None),
    ],
)
def test_hardware_classes(cap, sms, kind):
    assert device_class(cap, sms) == kind


@pytest.mark.parametrize("kind", tuple(TABLE))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
@pytest.mark.parametrize("m", (0, 1, 3, 4, 5, 16, 17, 64, 65, 256, 1024))
def test_dispatch_rows_cover_every_m(kind, dtype, m):
    config = config_for(kind, 12288, dtype, m)
    assert config is not None
    assert config == config_for(kind, 12288, dtype, 1 << 20 if m > 1024 else m)


@pytest.mark.parametrize(
    "kind,width,dtype,m",
    [
        (None, 12288, "bfloat16", 1),
        ("gb10", 128, "bfloat16", 1),
        ("rtx5090", 12288, "float32", 1),
        ("pro6000", 12290, "float16", 1025),
        ("gb10", 12288, "float16", -1),
    ],
)
def test_policy_holes(kind, width, dtype, m):
    assert config_for(kind, width, dtype, m) is None


def test_admission_metadata():
    def tensor(**kwargs):
        attrs = {
            "device": SimpleNamespace(type="cuda"),
            "dtype": torch.bfloat16,
            "ndim": 2,
            "shape": (3, 24576),
            "is_contiguous": lambda: True,
            "data_ptr": lambda: 16,
        }
        attrs.update(kwargs)
        return SimpleNamespace(**attrs)

    assert act._admitted(tensor(), None)
    assert act._admitted(tensor(shape=(0, 24576)), None)
    for bad in (
        tensor(dtype=torch.float32),
        tensor(shape=(3, 14)),
        tensor(data_ptr=lambda: 2),
        tensor(is_contiguous=lambda: False),
        tensor(device=SimpleNamespace(type="cpu")),
    ):
        assert not act._admitted(bad, None)
    assert not act._admitted(tensor(), tensor(data_ptr=lambda: 2))
    assert not act._admitted(tensor(), tensor(is_contiguous=lambda: False))


def test_fallback_and_override(monkeypatch):
    calls = []
    monkeypatch.setattr(act.flashinfer, "silu_and_mul", lambda x, out=None: calls.append(x) or x)
    x = torch.ones((1, 16))
    operation = act.SiluAndMul(provider="flashinfer")
    assert operation(x) is x and operation.provider == "flashinfer"
    monkeypatch.setenv(act.VARIABLE, "invalid")
    with pytest.raises(ValueError, match=act.VARIABLE):
        act.SiluAndMul()(x)
    assert len(calls) == 1


def test_empty_and_output_validation():
    x = torch.empty(0, 24576, dtype=torch.bfloat16)
    out = torch.empty(0, 12288, dtype=x.dtype)
    assert act.silu_and_mul(x, out) is out
    assert act.silu_and_mul(x).shape == out.shape
    x = torch.ones(1, 16)
    with pytest.raises(ValueError, match="overlap input"):
        act.silu_and_mul(x, x[:, :8])
    with pytest.raises(ValueError, match="shape, dtype"):
        act.silu_and_mul(x, torch.empty(1, 9))
    with pytest.raises(RuntimeError, match="autograd"):
        act.silu_and_mul(x.requires_grad_())


def test_compilation_cache_ignores_m_and_never_compiles_in_capture(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls = []
    compiled = object()
    monkeypatch.setattr(_provider, "compile_kernel", lambda *args: calls.append(args) or compiled)
    args = (0, "bfloat16", 12288, Config(), (12, 0))
    assert _provider.get_compiled(*args, capturing=True) is None
    assert calls == []
    assert _provider.get_compiled(*args) is compiled
    assert _provider.get_compiled(*args, capturing=True) is compiled
    assert len(calls) == 1


def test_compilation_failure_falls_back_but_forced_is_strict(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise ImportError("compiler missing")

    monkeypatch.setattr(_provider, "compile_kernel", unavailable)
    args = (0, "float16", 8, Config(), (12, 1))
    assert _provider.get_compiled(*args) is None
    assert _provider.get_compiled(*args) is None
    with pytest.raises(RuntimeError, match="unavailable"):
        _provider.get_compiled(*args, strict=True)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_torch_reference_and_metrics(dtype):
    x = torch.tensor([[0, -1, 1, -90, 1, 2, 3, 4]], dtype=dtype)
    exact = reference(x)
    assert exact.dtype == torch.float64
    assert exact[0, 0] == 0 and exact[0, -1] < 0
    rounded = exact.to(dtype)
    assert error_metrics(rounded, exact)["max_ulp"] <= 0.5
    assert set(market_formulas(x)) == {"sglang_formula", "inductor_formula"}
    assert error_metrics(torch.empty(0, dtype=dtype), torch.empty(0))["rms"] == 0
    assert error_metrics(torch.full_like(rounded, float("nan")), exact)["rms"] == float("inf")


def test_sweep_space():
    configs = tune_configs()
    assert len(set(configs)) == len(configs) == 90
    assert len({config.name for config in configs}) == len(configs)
    assert configs[:54] == [
        Config(t, v, p, e, trigger)
        for t in (64, 128, 256)
        for v in (1, 2)
        for e in (2, 4, 8, 16)
        for p in (False, True)
        for trigger in (("end", "early") if p else ("end",))
        if not (v == 2 and e in (2, 16))
    ]
    assert set(configs[54:]) == {
        Config(t, 1, True, e, "early", cache)
        for t in (64, 128, 256, 512)
        for e in (2, 4, 8)
        for cache in ("st_ef", "ld_el", "both")
    }
    assert {config for config in configs if config.elems == 16} == {
        Config(threads, 1, pdl, 16, trigger)
        for threads in (64, 128, 256)
        for pdl in (False, True)
        for trigger in (("end", "early") if pdl else ("end",))
    }
    with pytest.raises(ValueError):
        Config(32)


@pytest.mark.parametrize("elems", (2, 4, 8, 16))
def test_activation_vector_config(elems):
    from ctikernels.activation._dispatch import parse_config

    config = Config(128, 1, True, elems, "early")
    assert parse_config(f"128,1,1,{elems},1") == config
    assert parse_config(f"128,1,0,{elems}") == Config(elems=elems)
    assert config.name == f"cute_t128_v1_e{elems}_p1_te"
    assert config in tune_configs()


@pytest.mark.parametrize("elems", (0, 1, 3, 32))
def test_activation_invalid_vector_config(elems):
    from ctikernels.activation._dispatch import parse_config

    with pytest.raises(ValueError, match="elems must be"):
        Config(elems=elems)
    with pytest.raises(ValueError, match="elems must be"):
        parse_config(f"128,1,0,{elems}")


@pytest.mark.parametrize("width", (8, 10, 24, 12296))
@pytest.mark.parametrize("admitted", (False, True))
def test_activation_e16_width_validation(monkeypatch, width, admitted):
    config = Config(elems=16)
    with pytest.raises(ValueError, match="elems=16 requires positive width divisible by 16"):
        _provider.compile_kernel("float16", width, config, (12, 0))
    with pytest.raises(ValueError, match="elems=16 requires positive width divisible by 16"):
        _provider.get_compiled(0, "float16", width, config, (12, 0), strict=True)
    monkeypatch.setattr(act, "_admitted", lambda *a: admitted)
    monkeypatch.setattr(act, "_hardware", lambda *a: ("rtx5090", (12, 0)))
    x = torch.ones(1, 2 * width, dtype=torch.float16)
    with pytest.raises(ValueError, match="elems=16 requires width divisible by 16"):
        act.SiluAndMul(provider="cute", config=config)(x)
    monkeypatch.setenv("CTI_ACTIVATION_CONFIG", "128,1,0,16")
    with pytest.raises(ValueError, match="elems=16 requires width divisible by 16"):
        act.SiluAndMul(provider="cute")(x)
    monkeypatch.setitem(TABLE["rtx5090"], (width, "float16"), {(0, 16): config})
    assert config_for("rtx5090", width, "float16", 1) is None
    monkeypatch.setattr(act.flashinfer, "silu_and_mul", lambda x, out=None: x)
    operation = act.SiluAndMul(provider="flashinfer", config=config)
    assert operation(x) is x and operation.provider == "flashinfer"


def test_disjoint_output_view_can_share_an_allocation(monkeypatch):
    storage = torch.empty(24)
    x, out = storage[:16].view(1, 16), storage[16:].view(1, 8)
    monkeypatch.setattr(act.flashinfer, "silu_and_mul", lambda x, out=None: out)
    assert act.SiluAndMul(provider="flashinfer")(x, out) is out


def test_activation_trigger_config():
    from ctikernels.activation._dispatch import parse_config

    early = Config(128, 1, True, 8, "early")
    assert early.name.endswith("_te")
    assert Config(128, 1, True).name.endswith("_tn")
    assert parse_config("128,1,1,8,1") == early
    assert parse_config("128,1,1") == parse_config("128,1,1,8,0")
    assert early in tune_configs()
    for setting in ("128,1,0,8,1", "128,1,1,8,2", "128,1,1,8,0,0,0"):
        with pytest.raises(ValueError):
            parse_config(setting)
    for kwargs in ({"trigger": "early"}, {"pdl": True, "trigger": "bad"}):
        with pytest.raises(ValueError, match="trigger"):
            Config(**kwargs)


@pytest.mark.parametrize(
    "code,cache,suffix",
    [(0, "none", ""), (1, "st_ef", "_cstef"), (2, "ld_el", "_cldel"), (3, "both", "_cboth")],
)
def test_activation_cache_config(code, cache, suffix):
    from ctikernels.activation._dispatch import parse_config

    config = Config(128, 1, True, 8, "early", cache)
    assert config.name == f"cute_t128_v1_e8_p1_te{suffix}"
    assert parse_config(f"128,1,1,8,1,{code}") == config
    assert Config().cache == "none"
    assert parse_config("128,1,0") == parse_config("128,1,0,8,0,0") == Config()
    assert config in tune_configs()
    # Cache hints are opt-in only; no shipped band uses one.
    assert not {
        (kind, band)
        for kind, rows in TABLE.items()
        for bands in rows.values()
        for band, c in bands.items()
        if c.cache != "none"
    }


@pytest.mark.parametrize("cache", ("", "stef", "ldel", "invalid", None, 1))
def test_activation_invalid_cache_config(cache):
    with pytest.raises(ValueError, match="cache must be"):
        Config(cache=cache)


@pytest.mark.parametrize("setting", ("128,1,0,8,0,-1", "128,1,0,8,0,4", "128,1,0,8,0,1,0"))
def test_activation_invalid_cache_setting(setting):
    from ctikernels.activation._dispatch import parse_config

    with pytest.raises(ValueError):
        parse_config(setting)


def test_activation_compilation_cache_separates_policies(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "_FAILURES", {})
    calls = []

    def compile_kernel(dtype, width, config, capability):
        calls.append(config)
        return object()

    monkeypatch.setattr(_provider, "compile_kernel", compile_kernel)
    configs = [replace(Config(), cache=c) for c in ("none", "st_ef", "ld_el", "both")]
    compiled = [_provider.get_compiled(0, "bfloat16", 12288, c, (12, 0)) for c in configs]
    assert len(set(compiled)) == 4 and calls == configs
    for config, kernel in zip(configs, compiled, strict=True):
        assert (
            _provider.get_compiled(0, "bfloat16", 12288, config, (12, 0), capturing=True) is kernel
        )
    assert calls == configs


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "relu2"])
def test_activation_public_surface(name):
    import inspect

    assert name in act.__all__
    signature = inspect.signature(getattr(act, name))
    assert list(signature.parameters) == ["x", "out"]
    assert signature.parameters["out"].default is None


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "relu2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_activation_native_routing(monkeypatch, name, dtype):
    calls = []
    _fake_cuda(monkeypatch)
    monkeypatch.setattr(act, "_admitted", lambda *a: False)
    monkeypatch.setattr(
        act, "_general_compiled", lambda x, w, kind, n: lambda x, y: calls.append((kind, n))
    )
    x = torch.ones(3, 16, dtype=dtype)
    out = torch.empty(3, 16 if name == "relu2" else 8, dtype=dtype)
    operation = (
        act.SiluAndMul(activation=name.removesuffix("_and_mul")) if name != "relu2" else act.relu2
    )
    assert operation(x, out) is out
    assert calls == [(name.removesuffix("_and_mul"), 16 // x.element_size())]


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "relu2"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 26), (2, 0, 26), (3, 0)])
def test_all_activation_empty_no_provider(monkeypatch, name, dtype, shape):
    def unexpected(*args, **kwargs):
        pytest.fail("empty activation must not resolve or launch any provider")

    monkeypatch.setattr(act, "_cute_activation", unexpected)
    monkeypatch.setattr(act, "_hardware", unexpected)
    monkeypatch.setattr(act._flashinfer_version, "allowed", unexpected)
    if name != "relu2":
        monkeypatch.setattr(act.flashinfer, name, unexpected)
    x = torch.empty(shape, dtype=dtype)
    output_shape = shape if name == "relu2" else shape[:-1] + (shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=dtype)
    operation = getattr(act, name)
    assert operation(x).shape == out.shape
    assert operation(x, out) is out
    if name == "silu_and_mul":
        for provider in ("auto", "cute", "flashinfer", "triton"):
            assert act.SiluAndMul(provider=provider)(x, out) is out


@pytest.mark.parametrize("name", ["gelu_and_mul", "gelu_tanh_and_mul", "relu2"])
def test_new_activation_validation(monkeypatch, name):
    operation = getattr(act, name)
    x = torch.ones(3, 16)
    with pytest.raises(RuntimeError, match="autograd"):
        operation(x.requires_grad_())
    x = x.detach()
    shape = (3, 16 if name == "relu2" else 8)
    with pytest.raises(RuntimeError, match="requires gradients"):
        operation(x, torch.empty(shape, requires_grad=True))
    with pytest.raises(ValueError, match="shape, dtype and device"):
        operation(x, torch.empty(shape, dtype=torch.float16))
    storage = torch.ones(49)
    with pytest.raises(ValueError, match="overlap"):
        if name == "relu2":
            operation(storage[:48].view(3, 16), storage[1:].view(3, 16))
        else:
            operation(x, x[:, :8])
    with pytest.raises(ValueError, match="rank"):
        operation(torch.tensor(1.0))
    if name != "relu2":
        with pytest.raises(ValueError, match=r"2 \* width"):
            operation(torch.ones(3, 15))


def test_general_compile_cache_separates_kind_alignment_dtype_and_capture(monkeypatch):
    monkeypatch.setattr(_provider, "_GENERAL_COMPILED", {})
    monkeypatch.setattr(_provider, "available", lambda: True)
    calls = []
    monkeypatch.setattr(_provider, "compile_general", lambda *args: calls.append(args) or object())
    args = (0, "float32", 12, "gelu", 1, (12, 0))
    with pytest.raises(RuntimeError, match="warm"):
        _provider.get_general_compiled(*args, capturing=True)
    compiled = _provider.get_general_compiled(*args)
    assert _provider.get_general_compiled(*args, capturing=True) is compiled
    for dtype, kind, lanes in (
        ("float32", "gelu_tanh", 1),
        ("float32", "gelu", 4),
        ("bfloat16", "relu2", 2),
    ):
        assert _provider.get_general_compiled(0, dtype, 12, kind, lanes, (12, 0)) is not compiled
    assert len(calls) == 4


def test_relu2_allows_identical_output_view(monkeypatch):
    monkeypatch.setattr(act, "_cute_activation", lambda x, out, kind, shape: out)
    x = torch.ones(3, 13)
    out = x.view_as(x)
    assert act.relu2(x, out) is out


@pytest.mark.parametrize("pin", [False, True])
def test_silu_reroute_cached_reason_and_engine_check_toggle(monkeypatch, pin):
    from ctikernels._norm_act_reroute import Provider, RerouteError

    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(act._flashinfer_version, "allowed", lambda: False)
    from ctikernels.activation import cute

    monkeypatch.setattr(cute, "available", lambda: False)
    op = act.SiluAndMul()
    launches = []
    monkeypatch.setattr(
        op, "_triton", lambda *a: (Provider("triton"), lambda x, out: launches.append(x) or x)
    )
    x = torch.ones(2, 16, dtype=torch.bfloat16)
    if pin:
        op._plan(x, None, 8, (2, 8), pin=True)
    for _ in range(2):
        op(x)
        assert isinstance(op.provider, Provider) and op.provider == "triton"
        assert "nvidia-cutlass-dsl" in op.provider.reason
    assert len(launches) == 2
    size = len(op._plans), len(op._pinned)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError):
        op(x)
    assert len(launches) == 2 and (len(op._plans), len(op._pinned)) == size


def _fake_cuda(monkeypatch, hardware=("rtx5090", (12, 0))):
    from ctikernels.activation import cute

    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(act, "_hardware", lambda *a: hardware)
    monkeypatch.setattr(cute, "available", lambda: True)


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("case", ["table", "width", "unknown", "strided", "offset", "fp32"])
def test_auto_cute_capability_and_table(monkeypatch, kind, case):
    from ctikernels.activation import cute

    hardware = (None, (12, 0)) if case == "unknown" else ("rtx5090", (12, 0))
    _fake_cuda(monkeypatch, hardware)
    monkeypatch.setattr(act._flashinfer_version, "allowed", lambda: False)
    calls = []
    monkeypatch.setattr(
        cute, "get_compiled", lambda *a, **k: calls.append("tuned") or (lambda *a: None)
    )
    monkeypatch.setattr(
        act, "_general_compiled", lambda *a: calls.append("general") or (lambda *a: None)
    )
    width = 24 if case == "width" else 12288
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    if case == "strided":
        x = torch.ones(3, 2 * width + 4, dtype=dtype)[:, : 2 * width]
    elif case == "offset":
        x = torch.ones(3 * 2 * width + 1, dtype=dtype)[1:].view(3, 2 * width)
    else:
        x = torch.ones(3, 2 * width, dtype=dtype)
    monkeypatch.setattr(
        act,
        "_admitted",
        lambda x, out: x.is_contiguous() and x.data_ptr() % 16 == 0 and dtype != torch.float32,
    )
    op = act.SiluAndMul(activation=kind)
    op(x)
    assert op.provider == "cute"
    tuned = kind == "silu" and case == "table"
    assert calls == ["tuned" if tuned else "general"]
    assert isinstance(op.dispatched_config, Config if tuned else act.GeneralConfig)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "width,pad,offset", [(24, 0, 0), (24, 4, 0), (24, 2, 0), (24, 1, 0), (24, 0, 1), (3, 0, 0)]
)
def test_general_vector_divisibility(dtype, width, pad, offset):
    x = torch.empty(3 * (2 * width + pad) + offset, dtype=dtype)[offset:].view(3, -1)[
        :, : 2 * width
    ]
    out = torch.empty(3 * (width + pad) + offset, dtype=dtype)[offset:].view(3, -1)[:, :width]
    expected = 16 // x.element_size()
    while any(v % expected for v in (width, x.stride(0), out.stride(0), offset)):
        expected //= 2
    assert act._vector_lanes(x, out, width) == expected


@pytest.mark.parametrize("packed", ["x", "out", "leading"])
def test_pack_reroute_and_engine_checks(monkeypatch, packed, caplog):
    from ctikernels._norm_act_reroute import RerouteError

    _fake_cuda(monkeypatch)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    from ctikernels import _norm_act_reroute

    monkeypatch.setattr(_norm_act_reroute, "_WARNED", set())
    calls = []
    monkeypatch.setattr(
        act, "_general_compiled", lambda *a: lambda x, y: calls.append((x.stride(), y.stride()))
    )
    x = torch.ones(3, 32)[:, ::2] if packed == "x" else torch.ones(3, 16)
    if packed == "leading":
        x = torch.ones(2, 3, 16).transpose(0, 1)
    out = torch.empty(3, 16)[:, ::2] if packed == "out" else None
    op = act.SiluAndMul(activation="gelu")
    op(x, out)
    op(x, out)
    assert op.provider == "cute" and "packed" in op.provider.reason
    assert len(caplog.records) == 1
    assert all(xs[-1] == ys[-1] == 1 for xs, ys in calls)
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(RerouteError, match="packed"):
        op(x, out)
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
def test_missing_dsl_and_forced_providers(monkeypatch, kind):
    from ctikernels._norm_act_reroute import Provider
    from ctikernels.activation import cute

    _fake_cuda(monkeypatch)
    monkeypatch.setattr(cute, "available", lambda: False)
    monkeypatch.setattr(
        act.SiluAndMul, "_triton", staticmethod(lambda *a: (Provider("triton"), lambda x, out: out))
    )
    monkeypatch.setattr(act.flashinfer, kind + "_and_mul", lambda x, out=None: out)
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    op = act.SiluAndMul(activation=kind)
    op(x)
    assert op.provider == ("triton" if kind == "silu" else "flashinfer")
    assert "nvidia-cutlass-dsl" in op.provider.reason
    monkeypatch.setattr(act._flashinfer_version, "allowed", lambda: False)
    if kind != "silu":
        with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
            op(x)
        with pytest.raises(ValueError, match="only silu"):
            act.SiluAndMul(activation=kind, provider="triton")
    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
        act.SiluAndMul(activation=kind, provider="cute")(x)
    monkeypatch.setattr(_provider, "available", lambda: False)
    monkeypatch.setattr(_provider, "_GENERAL_COMPILED", {})
    with pytest.raises(RuntimeError, match="nvidia-cutlass-dsl"):
        op(x.float())


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
def test_forced_flashinfer_only_and_alias(monkeypatch, kind):
    _fake_cuda(monkeypatch)
    calls = []
    monkeypatch.setattr(
        act.flashinfer, kind + "_and_mul", lambda x, out=None: calls.append("flashinfer") or out
    )
    monkeypatch.setattr(act, "_general_compiled", lambda *a: lambda *a: calls.append("cute"))
    monkeypatch.setenv(act.VARIABLE, "cute")
    monkeypatch.setenv("CTI_ACT_PROVIDER", "flashinfer")
    op = act.SiluAndMul(activation=kind)
    op(torch.ones(3, 16, dtype=torch.bfloat16))
    assert op.provider == "flashinfer"
    op(torch.ones(3, 16))
    assert op.provider == "cute"
    assert calls == ["flashinfer", "cute"]


def test_collapsible_and_preserves_native_strides(monkeypatch):
    _fake_cuda(monkeypatch)
    seen = []
    monkeypatch.setattr(
        act, "_general_compiled", lambda *a: lambda x, y: seen.append((x.stride(), y.stride()))
    )
    x = torch.ones(2, 3, 40)[:, :, :32]
    out = torch.empty(2, 3, 24)[:, :, :16]
    op = act.SiluAndMul()
    assert op(x, out) is out
    assert seen == [((40, 1), (24, 1))]
    assert op.provider.reason is None


@pytest.mark.parametrize("name", ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"])
@pytest.mark.parametrize("bad", ["width", "input_stride", "output_stride", "dtype"])
def test_forced_flashinfer_layout_errors(monkeypatch, name, bad):
    _fake_cuda(monkeypatch)
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    out = torch.empty(3, 8, dtype=x.dtype)
    if bad == "width":
        x = x[:, :14].contiguous()
        out = out[:, :7].contiguous()
    elif bad == "input_stride":
        x = torch.ones(3, 32, dtype=x.dtype)[:, ::2]
    elif bad == "output_stride":
        out = torch.empty(3, 16, dtype=x.dtype)[:, ::2]
    else:
        x, out = x.double(), out.double()
    with pytest.raises(ValueError, match="contiguous CUDA FP16/BF16"):
        getattr(act.flashinfer, name)(x, out)


@pytest.mark.parametrize("shape,strides", [((1, 1), (99, 5)), ((3, 1), (1, 2))])
def test_relu2_packing_canonicalizes_singleton_stride(monkeypatch, shape, strides):
    _fake_cuda(monkeypatch)
    calls = []
    monkeypatch.setattr(
        act, "_general_compiled", lambda *a: lambda x, y: calls.append((x.stride(), y.stride()))
    )
    x = torch.ones(100).as_strided(shape, strides)
    assert x.is_contiguous() and x.stride(-1) != 1
    assert act.relu2(x, x) is x
    assert calls == [((1, 1), (1, 1))]


@requires_triton
@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("layout", ["bf16", "fp16", "fp32", "strided"])
@pytest.mark.parametrize("fi", [True, False])
@pytest.mark.parametrize("dsl", [True, False])
@pytest.mark.parametrize("capability", [(12, 0), (12, 1), (9, 0), (8, 9)])
def test_auto_selection_predicate(monkeypatch, kind, layout, fi, dsl, capability):
    from ctikernels.activation import cute

    _fake_cuda(monkeypatch, (None, capability))
    monkeypatch.setattr(cute, "available", lambda: dsl)
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.delenv("CTI_ACT_PROVIDER", raising=False)
    monkeypatch.delenv(act.VARIABLE, raising=False)
    monkeypatch.setattr(act._flashinfer_version, "allowed", lambda: fi)
    calls = []
    monkeypatch.setattr(
        act.flashinfer, kind + "_and_mul", lambda x, out=None: calls.append("flashinfer")
    )

    def general(*a):
        if not dsl:  # Mirrors the failing DSL import.
            raise RuntimeError("no CuTe DSL")
        calls.append("cute")
        return lambda *a: None

    monkeypatch.setattr(act, "_general_compiled", general)
    import ctikernels.activation._silu_and_mul as triton_module

    monkeypatch.setattr(
        triton_module, "silu_and_mul_forward", lambda x, out=None: calls.append("triton")
    )
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(layout, torch.bfloat16)
    x = torch.ones(3, 72, dtype=dtype)
    if layout == "strided":
        x = x[:, :64]
    op = act.SiluAndMul(activation=kind)
    if capability[0] == 12:
        if not dsl and (dtype == torch.float32 or (kind != "silu" and not fi)):
            with pytest.raises(RuntimeError):
                op(x)
            return
        op(x)
        expected = "cute" if dsl else ("triton" if kind == "silu" else "flashinfer")
        assert calls == [expected] and op.provider == expected
        assert bool(getattr(op.provider, "reason", None)) == (not dsl)
        return
    if fi and layout in ("bf16", "fp16"):
        op(x)
        assert calls == ["flashinfer"] and op.provider == "flashinfer"
        assert not getattr(op.provider, "reason", None)
        return
    if kind != "silu":
        out = torch.empty(3, x.shape[-1] // 2, dtype=dtype)
        assert op(x, out) is out
        gate, up = x.double().chunk(2, dim=-1)
        approximate = "tanh" if kind == "gelu_tanh" else "none"
        expected = (torch.nn.functional.gelu(gate, approximate=approximate) * up).to(dtype)
        assert calls == [] and op.provider == "torch"
        torch.testing.assert_close(out, expected)
        assert "capability 12.x" in op.provider.reason
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
        with pytest.raises(Exception, match="capability 12.x"):
            act.SiluAndMul(activation=kind)(x)
        return
    op(x)
    assert calls == ["triton"] and op.provider == "triton"
    assert "capability 12.x" in op.provider.reason
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    with pytest.raises(Exception, match="capability 12.x"):
        act.SiluAndMul(activation=kind)(x)


@requires_triton
@pytest.mark.parametrize("layout", ["bf16", "fp32", "strided"])
@pytest.mark.parametrize("capability", [(9, 0), (8, 9)])
def test_overrides_outside_capability_12(monkeypatch, layout, capability):
    _fake_cuda(monkeypatch, (None, capability))
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.delenv("CTI_ACT_PROVIDER", raising=False)
    monkeypatch.delenv(act.VARIABLE, raising=False)
    monkeypatch.setattr(act._flashinfer_version, "allowed", lambda: True)
    monkeypatch.setattr(act._flashinfer_version, "require_allowed", lambda: None)
    calls = []
    monkeypatch.setattr(act, "_general_compiled", lambda *a: calls.append("cute"))
    monkeypatch.setattr(act.flashinfer, "silu_and_mul", lambda x, out=None: calls.append("fi"))
    import ctikernels.activation._silu_and_mul as triton_module

    monkeypatch.setattr(
        triton_module, "silu_and_mul_forward", lambda x, out=None: calls.append("triton")
    )
    dtype = torch.float32 if layout == "fp32" else torch.bfloat16
    x = torch.ones(3, 72, dtype=dtype)
    if layout == "strided":
        x = x[:, :64]
    op = act.SiluAndMul(provider="triton")
    op(x)
    assert calls == ["triton"] and op.provider == "triton"
    with pytest.raises(RuntimeError, match="unqualified outside capability 12.x"):
        act.SiluAndMul(provider="cute")(x)
    with pytest.raises(RuntimeError, match="unqualified outside capability 12.x"):
        act.SiluAndMul(provider="cute", activation="gelu")(x)
    monkeypatch.setenv(act.VARIABLE, "cute")
    with pytest.raises(RuntimeError, match="unqualified outside capability 12.x"):
        act.SiluAndMul()(x)
    monkeypatch.delenv(act.VARIABLE)
    fi = act.SiluAndMul(provider="flashinfer")
    if layout == "bf16":
        fi(x)
        assert fi.provider == "flashinfer"
    else:
        with pytest.raises(RuntimeError, match="FlashInfer activation requires"):
            fi(x)
    assert "cute" not in calls
