# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Exercise plan reuse/invalidation with real CPU storage and fake device launches."""

import contextlib

import pytest
import torch

from ctikernels.activation import inference as act
from ctikernels.norm import inference as norm

try:
    import triton
except ImportError:  # the CPU CI env has no Triton
    triton = None

# These paths import the Triton kernels (norm fallbacks, the Triton SiLU provider).
requires_triton = pytest.mark.skipif(triton is None, reason="requires Triton")


@pytest.mark.parametrize("module", [norm, act], ids=["norm", "activation"])
def test_default_function_caches_compiled_launch_without_retaining_buffers(monkeypatch, module):
    is_norm = module is norm
    provider = __import__(
        f"ctikernels.{'norm' if is_norm else 'activation'}.cute", fromlist=["cute"]
    )
    monkeypatch.setattr(module, "_admitted", lambda *a: True)
    monkeypatch.setattr(module, "_hardware", lambda *a: ("gb10", (12, 1)))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(provider, "available", lambda: True)
    if not is_norm:
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(module._flashinfer_version, "allowed", lambda: False)
    resolutions, compilations, launches = [], [], []

    def compile(*args, **kwargs):
        compilations.append((args, kwargs))
        return lambda *args: launches.append(args[0].data_ptr())

    monkeypatch.setattr(provider, "get_compiled", compile)
    op = (
        norm.AddRMSNorm(provider="cute")
        if is_norm
        else act.SiluAndMul(provider="cute", config=act.Config())
    )
    resolve = op._resolve

    def record(*args):
        resolutions.append(1)
        return resolve(*args)

    monkeypatch.setattr(op, "_resolve", record)
    monkeypatch.setattr(module, "_DEFAULT_ADD" if is_norm else "_DEFAULT_SILU", op)
    x = torch.ones(3, 16, dtype=torch.bfloat16)
    r, w = x.clone(), torch.ones(16, dtype=x.dtype)
    out = torch.empty(3, 8, dtype=x.dtype)

    def call(x, eps=1e-6):
        if is_norm:
            return norm.add_rms_norm_(x, r, w, eps)
        return act.silu_and_mul(x, out)

    call(x)
    call(x, 1e-4)  # eps is a runtime argument, not a compilation signature.
    fresh = x.clone()
    call(fresh)
    assert len(resolutions) == len(compilations) == 1
    assert launches == [x.data_ptr(), x.data_ptr(), fresh.data_ptr()]
    # Environment is read on hits, but configuration parsing only happens on a miss.
    monkeypatch.setenv("CTI_NORM_CONFIG" if is_norm else "CTI_ACTIVATION_CONFIG", "256,1,0")
    call(x)
    assert len(resolutions) == len(compilations) == 2
    monkeypatch.setattr(module._flashinfer_version, "allowed", lambda: True)
    call(x)
    assert len(resolutions) == len(compilations) == 3


@pytest.mark.parametrize(
    "change",
    [
        "shape",
        "stride",
        "dtype",
        "weight_dtype",
        "zero",
        "sum",
        "sum_stride",
        "store",
        "static",
        "normalize",
        "provider",
        "config",
        "allow",
    ],
)
def test_norm_plan_signature_invalidates(monkeypatch, change):
    monkeypatch.setattr(norm._flashinfer_version, "allowed", lambda: True)
    op = norm.AddRMSNorm()
    calls = []
    monkeypatch.setattr(op, "_resolve", lambda *a: calls.append(1) or ("fake", lambda *a: None))
    x, r, w = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    kwargs = {"zero_centered": True}
    if change == "sum_stride":
        kwargs["sum_out"] = torch.empty_like(x)
    op(x, r, w, 1e-6, **kwargs)
    op(x, r, w, 1e-6, **kwargs)
    assert len(calls) == 1
    if change == "shape":
        x, r = torch.ones(4, 16), torch.ones(4, 16)
    elif change == "stride":
        x = torch.ones(3, 32)[:, :16]
    elif change == "dtype":
        x, r = x.half(), r.half()
    elif change == "weight_dtype":
        w = w.half()
    elif change == "zero":
        kwargs["zero_centered"] = False
    elif change in ("sum", "sum_stride"):
        kwargs["sum_out"] = torch.empty(3, 32)[:, :16]
    elif change == "store":
        kwargs["store_residual"] = False
    elif change == "static":
        op.static_weight = True
    elif change == "normalize":
        kwargs["normalize"] = "stored_sum"
    elif change == "provider":
        monkeypatch.setenv(norm.VARIABLE, "triton")
    elif change == "config":
        monkeypatch.setenv("CTI_NORM_CONFIG", "256,1,0")
    elif change == "allow":
        monkeypatch.setattr(norm._flashinfer_version, "allowed", lambda: False)
    op(x, r, w, 1e-6, **kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "change", ["shape", "stride", "dtype", "out", "out_stride", "provider", "alias"]
)
def test_activation_plan_signature_invalidates(monkeypatch, change):
    op = act.SiluAndMul()
    calls = []
    monkeypatch.setattr(op, "_resolve", lambda *a: calls.append(1) or ("fake", lambda x, out: out))
    x, out = torch.ones(3, 16), torch.empty(3, 8)
    op(x, out)
    op(x, out)
    assert len(calls) == 1
    if change == "shape":
        x, out = torch.ones(4, 16), torch.empty(4, 8)
    elif change == "stride":
        x = torch.ones(3, 32)[:, :16]
    elif change == "dtype":
        x, out = x.half(), out.half()
    elif change == "out":
        out = None
    elif change == "out_stride":
        out = torch.empty(3, 16)[:, :8]
    elif change == "provider":
        monkeypatch.setenv(act.VARIABLE, "triton")
    elif change == "alias":
        monkeypatch.setenv("CTI_ACT_PROVIDER", "triton")
    op(x, out)
    assert len(calls) == 2


def test_pointer_checks_are_not_cached(monkeypatch):
    op = norm.AddRMSNorm(provider="cute")
    x, r, w = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    summed = torch.empty_like(x)
    calls = []
    monkeypatch.setattr(op, "_resolve", lambda *a: ("fake", lambda *a: calls.append(1)))
    op(x, r, w, 1e-6, sum_out=summed)
    with pytest.raises(ValueError, match="overlap"):
        op(x, r, w, 1e-6, sum_out=x)
    unaligned = torch.empty(49)[1:].view(3, 16)
    op(x, r, w, 1e-6, sum_out=unaligned)
    assert len(calls) == 2
    activation = act.SiluAndMul()
    monkeypatch.setattr(activation, "_resolve", lambda *a: ("fake", lambda x, out: out))
    storage = torch.empty(72)
    input_x = storage[:48].view(3, 16)
    activation(input_x, storage[48:].view(3, 8))
    with pytest.raises(ValueError, match="overlap"):
        activation(input_x, storage[:24].view(3, 8))


def test_norm_cache_rechecks_operand_alignment(monkeypatch):
    # CPU tensors stand in for CUDA operands; preserve the actual pointer checks.
    monkeypatch.setattr(
        norm,
        "_admitted",
        lambda x, r, w, *a: (
            all(t.data_ptr() % 16 == 0 for t in (x, r, w)) and not norm._overlap(x, r)
        ),
    )
    monkeypatch.setattr(norm, "_hardware", lambda *a: ("gb10", (12, 1)))
    op = norm.AddRMSNorm(provider="cute")
    launches = []
    monkeypatch.setattr(op, "_cute", lambda *a: lambda *a: launches.append(1))
    x, r, w = torch.ones(3, 16), torch.ones(3, 16), torch.ones(16)
    op(x, r, w, 1e-6)
    for bad_x, bad_r in ((torch.empty(49)[1:].view(3, 16), r), (x, x)):
        with pytest.raises(RuntimeError, match="forced cute"):
            op(bad_x, bad_r, w, 1e-6)
    assert launches == [1]


@requires_triton
@pytest.mark.parametrize("shape", [(3, 16), (0, 16), (3, 0)])
@pytest.mark.parametrize("strided", [False, True])
def test_triton_activation_keeps_pointwise_fallback_without_training(shape, strided):
    x = torch.randn(shape)
    if strided:
        x = torch.randn(shape[0], shape[1] * 2)[:, ::2]
    gate, up = x.chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    op = act.SiluAndMul(provider="triton")
    out = torch.empty(shape[0], shape[1])[:, ::2]
    assert torch.equal(op(x), expected)
    assert op(x, out) is out and torch.equal(out, expected)
    assert op.provider == ("empty" if x.numel() == 0 else "triton")


@pytest.mark.parametrize(
    "operation,zero",
    [
        ("rms_norm", False),
        ("rms_norm_", False),
        ("rms_norm", True),
        ("rms_norm_", True),
        ("residual_rms_norm", True),
    ],
)
def test_standalone_norm_reuses_resolved_provider(monkeypatch, operation, zero):
    import sys
    from types import SimpleNamespace

    launches = []

    def launch(x, *args, **kwargs):
        launches.append(x.data_ptr())
        return x

    monkeypatch.setattr(norm._flashinfer_version, "allowed", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(rmsnorm=launch))
    # Patch the package attribute as well: standalone resolution imports the module.
    from ctikernels import norm as package

    kernels = SimpleNamespace(rms_norm=launch, rms_norm_=launch, residual_rms_norm=launch)
    monkeypatch.setattr(package, "_zero_centered_rms_norm", kernels, raising=False)
    norm._resolve_standalone.cache_clear()
    try:
        x, weight = torch.ones(3, 16), torch.ones(16)
        kwargs = {"zero_centered": zero}
        if operation == "residual_rms_norm":
            kwargs["out_dtype"] = x.dtype
        fn = getattr(norm, operation)
        fn(x, weight, 1e-6, **kwargs)
        fn(x.clone(), weight, 1e-4, **kwargs)
        info = norm._resolve_standalone.cache_info()
        assert info.misses == 1 and info.hits == 1 and len(launches) == 2
        fn(torch.ones(4, 16), weight, 1e-6, **kwargs)
        assert norm._resolve_standalone.cache_info().misses == 2
    finally:
        norm._resolve_standalone.cache_clear()


def _warm_environment(monkeypatch, module, kind="gb10"):
    from importlib import import_module
    from types import SimpleNamespace

    is_norm = module is norm
    package = import_module(f"ctikernels.{'norm' if is_norm else 'activation'}.cute")
    compiler = import_module(package.__name__ + "._provider")
    state = SimpleNamespace(capturing=False, compiles=[], launches=[], fail=False)
    monkeypatch.setattr(module, "_admitted", lambda *a: True)
    monkeypatch.setattr(module, "_hardware", lambda *a: (kind, (12, 1)))
    monkeypatch.setattr(module._flashinfer_version, "allowed", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(package, "available", lambda: True)
    if not is_norm:
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(compiler, "available", lambda: True)
    monkeypatch.setattr(compiler, "_COMPILED", {})
    monkeypatch.setattr(compiler, "_FAILURES", {})
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **dict(kw, device="cpu")))
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **dict(kw, device="cpu")))
    for variable in (
        module.VARIABLE,
        "CTI_ACT_PROVIDER",
        "CTI_NORM_CONFIG",
        "CTI_ACTIVATION_CONFIG",
    ):
        monkeypatch.delenv(variable, raising=False)

    def compile(*args):
        state.compiles.append(args)
        if state.fail:
            raise RuntimeError("fake compiler failure")
        return lambda *a: state.launches.append("cute")

    def fallback(*args, **kw):
        state.launches.append("fallback")
        return args[1] if len(args) == 2 else None

    monkeypatch.setattr(compiler, "compile_kernel", compile)
    if not is_norm and triton is not None:
        from ctikernels.activation import _silu_and_mul

        monkeypatch.setattr(_silu_and_mul, "silu_and_mul_forward", fallback)
    monkeypatch.setattr(module.flashinfer, "add_rms_norm_" if is_norm else "silu_and_mul", fallback)
    if is_norm:
        kernels = import_module("ctikernels.norm._zero_centered_rms_norm")

        monkeypatch.setattr(kernels, "add_rms_norm_", fallback)
    return state, compiler


def _warm_args(module):
    return dict(
        **({"hidden": 4096} if module is norm else {"width": 12288}),
        dtype=torch.bfloat16,
        device="cuda",
    )


@requires_triton
@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("summed", [False, True])
def test_norm_warm_up_launches_each_non_cute_m_and_each_cute_kernel_once(
    monkeypatch, store, summed
):
    _warm_environment(monkeypatch, norm)
    op = norm.AddRMSNorm()
    launches, kernel = [], object()

    def resolve(x, residual, weight, zero, actual_store, sum_rows, *args):
        m = x.shape[0]
        provider = ("cute", "flashinfer", "triton")[m % 3]
        assert zero and actual_store == store
        assert (sum_rows is not None) == summed

        def launch(x, residual, weight, eps, sum_rows):
            assert eps == 1e-6
            for tensor in (x, residual, weight, sum_rows):
                if tensor is not None:
                    assert torch.count_nonzero(tensor) == 0
            launches.append((m, provider))

        launch.kernel = kernel
        return provider, launch

    monkeypatch.setattr(op, "_resolve", resolve)
    op.warm_up(33, **_warm_args(norm), zero_centered=True, store_residual=store, sum_out=summed)
    # One CuTe kernel serves every cute M here: warm-up launches it once (at M=3).
    assert launches == [
        (m, ("cute", "flashinfer", "triton")[m % 3]) for m in range(1, 34) if m % 3 or m == 3
    ]


def test_silu_warm_up_launches_each_non_cute_m_per_output_variant_and_each_cute_kernel_once(
    monkeypatch,
):
    _warm_environment(monkeypatch, act)
    op = act.SiluAndMul()
    launches, kernel = [], object()

    def resolve(x, out, *args):
        m = x.shape[0]
        provider = ("cute", "flashinfer", "triton")[m % 3]
        has_out = out is not None

        def launch(x, out):
            assert (out is not None) == has_out or provider == "cute"
            assert torch.count_nonzero(x) == 0
            if out is not None:
                assert torch.count_nonzero(out) == 0
            launches.append((m, provider, has_out))
            return out

        launch.kernel = kernel
        return provider, launch

    monkeypatch.setattr(op, "_resolve", resolve)
    op.warm_up(33, **_warm_args(act))
    assert launches == [
        (m, ("cute", "flashinfer", "triton")[m % 3], has_out)
        for m in range(1, 34)
        for has_out in (False, True)
        if m % 3 or (m, has_out) == (3, False)
    ]


def _call_buffers(module, max_m, **kw):
    size = 4096 if module is norm else 24576
    x = torch.empty(max_m, size, dtype=torch.bfloat16)
    residual = torch.empty_like(x, dtype=kw.get("residual_dtype", x.dtype))
    weight = torch.empty(size, dtype=kw.get("weight_dtype", x.dtype))
    out = (
        torch.empty_like(residual)
        if module is norm
        else torch.empty(max_m, size // 2, dtype=x.dtype)
    )

    def call(op, m, *, output=False):
        if module is norm:
            return op(
                x[:m],
                residual[:m],
                weight,
                1e-6,
                zero_centered=kw.get("zero_centered", False),
                store_residual=kw.get("store_residual", True),
                sum_out=out[:m] if kw.get("sum_out", False) else None,
            )
        return op(x[:m], out[:m] if output else None)

    return call


@requires_triton
@pytest.mark.parametrize("module", [norm, act], ids=["norm", "activation"])
@pytest.mark.parametrize("kind", ["gb10", "rtx5090"])
def test_warm_up_routes_match_every_table_row_and_pin(monkeypatch, module, kind):
    state, _ = _warm_environment(monkeypatch, module, kind)
    op = norm.AddRMSNorm() if module is norm else act.SiluAndMul()
    # Includes activation's boundary at 65 without a large prefill loop.
    routes = op.warm_up(65, **_warm_args(module))
    assert state.launches == ["cute"] * len(state.compiles)  # each kernel primed once
    assert op._plans == {}
    assert len(op._pinned) == 65 * (1 if module is norm else 2)
    expanded = []
    previous = None
    for route in routes:
        assert set(route) == {"m", "provider", "config"}
        lo, hi = route["m"]
        assert lo == len(expanded) + 1 and lo <= hi
        identity = route["provider"], route["config"]
        assert identity != previous
        previous = identity
        expanded.extend([identity] * (hi - lo + 1))
    assert len(expanded) == 65
    for m, identity in enumerate(expanded, 1):
        if module is norm:
            entry = norm.entry_for(kind, 4096, "bfloat16", m, False, False)
            expected = entry.provider, entry.config.name if entry.provider == "cute" else None
        else:
            config = act.config_for(kind, 12288, "bfloat16", m)
            expected = ("cute", config.name) if config else ("flashinfer", None)
        assert identity == expected
    assert len(state.compiles) == len(set(expanded))
    pinned = op._pinned.copy()
    call = _call_buffers(module, 199)
    for m in range(66, 200):
        call(op, m)
    assert op._pinned == pinned
    assert len(op._plans) < 128

    def unexpected(*a):
        pytest.fail("warmed shape was resolved again")

    monkeypatch.setattr(op, "_resolve", unexpected)
    state.capturing = True
    for m in (1, 4, 16, 64, 65):
        call(op, m)
        assert (op.provider, op.dispatched_config.name) == expanded[m - 1]
        if module is act:
            call(op, m, output=True)
    assert op._pinned == pinned


@requires_triton
@pytest.mark.parametrize("kind", ["gb10", "rtx5090"])
@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_norm_warm_up_weight_conventions_and_fallback_routes(monkeypatch, kind, zero, static):
    state, _ = _warm_environment(monkeypatch, norm, kind)
    op = norm.AddRMSNorm(static_weight=static)
    routes = op.warm_up(257, **_warm_args(norm), zero_centered=zero)
    fallbacks = sum(r["m"][1] - r["m"][0] + 1 for r in routes if r["provider"] != "cute")
    assert sorted(state.launches) == sorted(
        ["fallback"] * fallbacks + ["cute"] * len(state.compiles)
    )
    for m in range(1, 258):
        route = next(r for r in routes if r["m"][0] <= m <= r["m"][1])
        entry = norm.entry_for(kind, 4096, "bfloat16", m, zero, static)
        assert route["provider"] == entry.provider
        assert route["config"] == (entry.config.name if entry.provider == "cute" else None)


@requires_triton
@pytest.mark.parametrize("module", [norm, act], ids=["norm", "activation"])
@pytest.mark.parametrize("failure", [False, True])
def test_auto_capture_miss_raises_but_recorded_failure_keeps_fallback(monkeypatch, module, failure):
    state, _ = _warm_environment(monkeypatch, module)
    op = norm.AddRMSNorm() if module is norm else act.SiluAndMul()
    call = _call_buffers(module, 5)
    if failure:
        state.fail = True
        if module is act:
            with pytest.raises(RuntimeError, match="compilation unavailable"):
                call(op, 5)
            state.capturing = True
            with pytest.raises(RuntimeError, match="compilation unavailable"):
                call(op, 5)
            assert len(state.compiles) == 1 and not state.launches
            return
        call(op, 5)
        eager_provider = op.provider
        assert eager_provider == ("triton" if module is norm else "flashinfer")
        op._plans.clear()  # Exercise compiler failure cache, not a retained fallback plan.
    else:
        op.warm_up(4, **_warm_args(module))  # M=5 selects an uncompiled config.
    state.capturing = True
    if failure:
        call(op, 5)
        assert op.provider == eager_provider
        assert state.launches == ["fallback", "fallback"]
        assert len(state.compiles) == 1
    else:
        launches = list(state.launches)  # warm-up primed each compiled kernel once
        with pytest.raises(RuntimeError, match=r"warm.*(capture|CUDA graph)"):
            call(op, 5)
        assert state.launches == launches == ["cute"] * len(state.compiles)
        state.capturing = False
        call(op, 5)
        assert op.provider == "cute"


@requires_triton
@pytest.mark.parametrize("module", [norm, act], ids=["norm", "activation"])
def test_warm_up_validation_and_singleton(monkeypatch, module):
    state, _ = _warm_environment(monkeypatch, module)
    op = norm.AddRMSNorm() if module is norm else act.SiluAndMul()
    with pytest.raises(ValueError, match="max_m"):
        op.warm_up(0, **_warm_args(module))
    state.capturing = True
    with pytest.raises(RuntimeError, match="must run eagerly"):
        op.warm_up(1, **_warm_args(module))
    assert not state.compiles and not op._pinned
    state.capturing = False
    name = "warm_up_add_rms_norm" if module is norm else "warm_up_silu_and_mul"
    monkeypatch.setattr(module, "_DEFAULT_ADD" if module is norm else "_DEFAULT_SILU", op)
    assert name in module.__all__
    assert getattr(module, name)(5, **_warm_args(module)) == op.warm_up(5, **_warm_args(module))
    assert op._pinned and state.launches == ["cute"] * len(state.compiles)


@requires_triton
@pytest.mark.parametrize("module", [norm, act], ids=["norm", "activation"])
@pytest.mark.parametrize("override", ["constructor", "environment", "fallback"])
def test_warm_up_respects_overrides_and_environment(monkeypatch, module, override):
    state, _ = _warm_environment(monkeypatch, module)
    cls = norm.AddRMSNorm if module is norm else act.SiluAndMul
    config = norm.Config(threads=128) if module is norm else act.Config(threads=256)
    if override == "constructor":
        monkeypatch.setenv(module.VARIABLE, "flashinfer")
        op = cls(provider="cute", config=config)
    elif override == "environment":
        monkeypatch.setenv(module.VARIABLE, "cute")
        monkeypatch.setenv(
            "CTI_NORM_CONFIG" if module is norm else "CTI_ACTIVATION_CONFIG",
            "128,1,0" if module is norm else "256,1,0",
        )
        op = cls()
    else:
        monkeypatch.setenv(module.VARIABLE, "flashinfer")
        op = cls()
    routes = op.warm_up(5, **_warm_args(module))
    assert routes == [
        {
            "m": (1, 5),
            "provider": "flashinfer" if override == "fallback" else "cute",
            "config": None if override == "fallback" else config.name,
        }
    ]
    assert state.launches == (
        ["fallback"] * (5 if module is norm else 10) if override == "fallback" else ["cute"]
    )
    _call_buffers(module, 5)(op, 5)
    assert op.provider == routes[0]["provider"]
    if override != "constructor":
        monkeypatch.setenv(module.VARIABLE, "triton")
        _call_buffers(module, 5)(op, 5)
        assert op.provider == "triton"


@requires_triton
def test_norm_warm_up_mixed_sum_signature(monkeypatch):
    state, _ = _warm_environment(monkeypatch, norm)
    op = norm.AddRMSNorm(static_weight=True)
    kw = {
        "residual_dtype": torch.float32,
        "weight_dtype": torch.float32,
        "store_residual": False,
        "sum_out": True,
        "zero_centered": True,
    }
    routes = op.warm_up(5, **_warm_args(norm), **kw)
    assert state.launches == ["cute"] * len(state.compiles)
    assert all(r["provider"] == "cute" for r in routes)
    call = _call_buffers(norm, 5, **kw)
    op._require_cached_plan = True
    for m in range(1, 6):
        call(op, m)
        assert op.dispatched_config.static_weight
    # Compiled ABI includes store_residual=False, has_sum=True, residual_dtype=FP32.
    assert all(args[-3:] == (False, True, "float32") for args in state.compiles)
    assert all(args[1] == "float32" for args in state.compiles)


@pytest.mark.parametrize("kind", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_activation_warm_up_general_and_capture(monkeypatch, kind, dtype):
    state, compiler = _warm_environment(monkeypatch, act)
    monkeypatch.setattr(compiler, "_GENERAL_COMPILED", {})

    def compile(*args):
        state.compiles.append(args)
        return lambda *a: state.launches.append("cute")

    monkeypatch.setattr(compiler, "compile_general", compile)
    op = act.SiluAndMul(activation=kind)
    n = 16 // torch.empty((), dtype=dtype).element_size()
    routes = op.warm_up(5, width=24, dtype=dtype, device="cuda")
    assert routes == [{"m": (1, 5), "provider": "cute", "config": f"cute_general_n{n}"}]
    assert state.launches == ["cute"] and len(state.compiles) == 1
    state.capturing = True
    op._require_cached_plan = True
    op(torch.empty(3, 48, dtype=dtype))
    assert op.provider == "cute" and op.dispatched_config.lanes == n
    op._require_cached_plan = False
    # Runtime row strides reuse the warmed kernel, even on a new layout.
    op(torch.empty(3, 56, dtype=dtype)[:, :48], torch.empty(3, 32, dtype=dtype)[:, :24])
    assert len(state.compiles) == 1 and state.launches == ["cute"] * 3
    with pytest.raises(RuntimeError, match="warm"):
        op(torch.empty(145, dtype=dtype)[1:].view(3, 48))


@pytest.mark.parametrize("kind", ["gelu", "gelu_tanh"])
def test_gelu_warm_up_shared_instance(monkeypatch, kind):
    name = "warm_up_" + kind + "_and_mul"
    assert name in act.__all__
    instance = getattr(act, "_DEFAULT_" + kind.upper())
    calls = []
    monkeypatch.setattr(instance, "warm_up", lambda *a, **kw: calls.append((a, kw)) or [])
    assert getattr(act, name)(7, width=24) == []
    assert calls == [((7,), {"width": 24})]


@pytest.mark.parametrize("change", ["alignment", "out_alignment", "dsl", "checks", "activation"])
def test_activation_plan_extra_signature(monkeypatch, change):
    from ctikernels.activation import cute

    op = act.SiluAndMul()
    calls = []
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(op, "_resolve", lambda *a: calls.append(1) or ("fake", lambda x, y: y))
    x, out = torch.empty(3, 48), torch.empty(3, 24)
    op(x, out)
    op(x.clone(), out)
    assert len(calls) == 1
    if change == "alignment":
        x = torch.empty(145)[1:].view(3, 48)
    elif change == "out_alignment":
        out = torch.empty(73)[1:].view(3, 24)
    elif change == "dsl":
        monkeypatch.setattr(cute, "available", lambda: False)
    elif change == "checks":
        monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    else:
        op.activation = "gelu"
    op(x, out)
    assert len(calls) == 2


@requires_triton
@pytest.mark.parametrize("hidden", [1000, 5120])
def test_norm_warm_up_off_table_pins_generic_config(monkeypatch, hidden):
    state, _ = _warm_environment(monkeypatch, norm)
    op = norm.AddRMSNorm()
    routes = op.warm_up(5, **dict(_warm_args(norm), hidden=hidden))
    assert routes == [{"m": (1, 5), "provider": "cute", "config": norm.Config().name}]
    assert len(state.compiles) == 1 and state.launches == ["cute"]
    state.capturing = True
    op._require_cached_plan = True
    x = torch.ones(5, hidden, dtype=torch.bfloat16)
    op(x, x.clone(), torch.ones(hidden, dtype=x.dtype), 1e-6)
    assert len(state.compiles) == 1 and state.launches == ["cute", "cute"]
