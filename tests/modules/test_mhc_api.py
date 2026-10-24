# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contracts and reference numerics; no GPU or CuTe installation required."""

import ctypes
import os
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning.dsv4_mhc_exit import reference as benchmark_ref
from ctikernels.mhc import MhcExit, exact_reference, inference, measured_floor, reference, support
from ctikernels.mhc._reference import MhcConfig, make_state, make_weights


def data(hidden=16, tokens=3):
    cfg = MhcConfig(hidden=hidden)
    state, weights = make_state(cfg, tokens), make_weights(cfg)
    return dict(
        x=state["x"], residual=state["R"], post=state["post"], comb=state["comb"], **weights
    )


def fake_op(monkeypatch, capture=False):
    monkeypatch.setattr(inference, "support", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    # Routing tests cover the cluster path; the production default is grid.
    if "CTI_MHC_DECODE" not in os.environ:
        monkeypatch.setenv("CTI_MHC_DECODE", "cluster")
    op = MhcExit("cpu", hidden=16, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)
    op._plan = (64, 1, 32)
    op._counter = torch.zeros(12, dtype=torch.int64)
    op._dummy_words = op._counter[:4]
    op._ysum = torch.empty(256, 16, dtype=torch.bfloat16)
    op._partials = torch.empty(32, 35, 4, dtype=torch.float64)
    op._grow_stage(256)
    op._plans = {cap: (64, 1, cap) for cap in (4, 16, 32, 64, 128, 256)}
    op._plans.update({1000 + cap: inference._cluster_plan(16, cap) for cap in (1, 2, 4, 8)})
    calls = []
    op._compiled = {
        (mode, decode, False, op._compile_key(mode, 1)[3]): lambda *a: calls.append(a)
        for mode in ("exit", "pre", "post")
        for decode in (0, 4, 16, 32, 64, 128, 256, 1001, 1002, 1004, 1008, -1, -2, -4, -8, -16, -32)
    }
    return op, calls


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"hc_mult": 8}, "hc_mult"),
        ({"hidden": 7}, "divisible by 8"),
        ({"hidden": 0}, "positive"),
        ({"hidden": 8.0}, "integer"),
        ({"dtype": torch.float16}, "bfloat16"),
        ({}, "CUDA"),
    ],
)
def test_support_cpu_no_probe(monkeypatch, kwargs, reason):
    def forbidden(*args, **kw):
        pytest.fail("CPU support must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert reason in support("cpu", **({"hidden": 16} | kwargs))


def test_support_cuda_reasons(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 9))
    assert "sm_90" in support("cuda", hidden=4096)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (12, 1))
    monkeypatch.setattr(inference, "_cute_available", lambda: False)
    assert "4.8.0" in support("cuda", hidden=4096)
    monkeypatch.setattr(inference, "_cute_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(
            multi_processor_count=48, shared_memory_per_block_optin=99 * 1024
        ),
    )
    assert support("cuda", hidden=4104) is None
    assert "more decode CTAs" in support("cuda", hidden=8192)


def test_launch_policy():
    assert inference._policy(4096, 48, 99 * 1024) == (128, 32, 32)
    assert inference._policy(4104, 48, 99 * 1024) == (128, 33, 32)
    assert inference._policy(1024, 48, 99 * 1024) == (64, 16, 32)
    assert inference._policy(4096, 48, 99 * 1024, 4) == (64, 64, 4)
    assert inference._policy(4096, 48, 99 * 1024, 16) == (64, 64, 16)
    assert inference._policy(4104, 48, 99 * 1024, 16) == (64, 65, 16)
    # One resident block per SM cannot support the smaller slice at this H.
    with pytest.raises(ValueError, match="co-resident"):
        inference._policy(4096, 48, 99 * 1024, 32, slice_size=64)
    with pytest.raises(ValueError, match="co-resident"):
        inference._policy(4096, 48, 99 * 1024, 4, regs_per_sm=32768, slice_size=64)
    with pytest.raises(ValueError, match="shared memory"):
        inference._policy(4096, 48, 32 * 1024)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rms_eps", 0),
        ("hc_eps", -1),
        ("norm_eps", float("nan")),
        ("post_mult", float("inf")),
        ("sinkhorn_iters", 0),
        ("sinkhorn_iters", 1.5),
    ],
)
def test_invalid_scalars(name, value):
    kwargs = {"hidden": 16, "rms_eps": 1e-6, "hc_eps": 1e-6, "norm_eps": 1e-6}
    with pytest.raises(ValueError):
        MhcExit("cpu", **(kwargs | {name: value}))


def test_constructor_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        MhcExit("cpu", hidden=16, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6)


@pytest.mark.parametrize(
    "name,transform,reason",
    [
        ("x", lambda t: t.float(), "dtype"),
        ("residual", lambda t: t[:, :3].contiguous(), "shape"),
        ("post", lambda t: t[:, :3].contiguous(), "shape"),
        ("comb", lambda t: t.transpose(1, 2), "contiguous"),
        ("fn", lambda t: t.T, "shape"),
        ("scale", lambda t: t.repeat(2), "shape"),
        ("base", lambda t: torch.empty(25)[1:], "aligned"),
        ("norm_weight", lambda t: t.to("meta"), "device"),
        ("fn", lambda t: t.requires_grad_(), "inference-only"),
        ("residual", lambda t: t[:0], "T > 0"),
    ],
)
def test_input_validation(monkeypatch, name, transform, reason):
    op, calls = fake_op(monkeypatch)
    inputs = data()
    inputs[name] = transform(inputs[name])
    with pytest.raises(ValueError, match=reason):
        op(**inputs)
    assert not calls


def test_output_validation_and_aliases(monkeypatch):
    op, calls = fake_op(monkeypatch)
    inputs = data()
    for outputs, reason in [
        ({"out": torch.empty(3, 16)}, "dtype"),
        ({"out": inputs["x"]}, "overlap"),
        ({"post_out": inputs["post"]}, "overlap"),
        ({"residual_out": inputs["residual"]}, "overlap"),
    ]:
        with pytest.raises(ValueError, match=reason):
            op(**inputs, **outputs)
    assert not calls


def test_all_modes_and_post_trailing_dimension(monkeypatch):
    op, calls = fake_op(monkeypatch)
    inputs = data()
    inputs["post"] = inputs["post"].unsqueeze(-1)
    outputs = op(**inputs)
    assert [t.shape for t in outputs] == [(3, 4, 16), (3, 4), (3, 4, 4), (3, 16)]
    assert len(calls) == 1 and calls[0][-6] == 3
    assert calls[0][:4] == tuple(inputs[k].data_ptr() for k in ("x", "residual", "post", "comb"))
    pre = op.pre(**{k: inputs[k] for k in ("residual", "fn", "scale", "base", "norm_weight")})
    post = op.post(**{k: inputs[k] for k in ("x", "residual", "post", "comb")})
    assert len(pre) == 3 and post.shape == outputs[0].shape
    assert len(calls) == 3


def test_capture_requires_outputs_and_preparation(monkeypatch):
    op, calls = fake_op(monkeypatch, capture=True)
    inputs = data()
    with pytest.raises(ValueError, match="output buffers"):
        op(**inputs)
    outputs = {
        "residual_out": torch.empty_like(inputs["residual"]),
        "post_out": torch.empty_like(inputs["post"]),
        "comb_out": torch.empty_like(inputs["comb"]),
        "out": torch.empty_like(inputs["x"]),
    }
    result = op(**inputs, **outputs)
    assert all(a is b for a, b in zip(result, outputs.values(), strict=True))
    assert len(calls) == 1
    op._compiled.clear()
    with pytest.raises(ValueError, match="prepare"):
        op(**inputs, **outputs)
    with pytest.raises(ValueError, match="outside CUDA graph"):
        op.prepare([1])


def test_prepare_counts_and_close(monkeypatch):
    op, calls = fake_op(monkeypatch)
    for counts in ([0], [-1], [2.0], [True], [2**30]):
        with pytest.raises(ValueError, match="token counts"):
            op.prepare(counts)
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: syncs.append(device))
    op.close()
    op.close()
    assert len(syncs) == 1 and op._partials is None
    with pytest.raises(RuntimeError, match="after close"):
        op(**data())
    with pytest.raises(RuntimeError, match="after close"):
        op.prepare([1])
    assert not calls


def test_prepare_compiles_buckets_once(monkeypatch):
    op, _ = fake_op(monkeypatch)
    op._compiled.clear()
    compiled, warmed = [], []
    kernels = ModuleType("ctikernels.mhc._cute.kernels")

    def compile_kernel(cfg, plan, mode, decode, **kwargs):
        compiled.append(
            (
                mode,
                (1000 if kwargs.get("cluster") else 0) + plan[2]
                if decode
                else -kwargs["token_tile"]
                if mode != "post"
                else 0,
            )
        )
        return lambda *args: None

    kernels.compile_kernel = compile_kernel
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(op, "_warm", lambda mode, t: warmed.append((mode, t)))
    workspace = (op._counter.data_ptr(), op._partials.data_ptr())
    op.prepare([1, 2, 31, 32, 33, 1000])
    op.prepare([4, 7, 128])
    assert set(compiled) == (
        {("exit", cap) for cap in (1001, 1002, 1004, 1008, 32, 64, 128, -8)}
        | {("pre", cap) for cap in (4, 16, 32, -1, -8)}
        | {("post", 0)}
    )
    assert len(compiled) == len(warmed) == 14
    assert workspace == (op._counter.data_ptr(), op._partials.data_ptr())


@pytest.mark.parametrize("mode", ["exit", "pre", "post"])
@pytest.mark.parametrize("iters", [1, 20])
def test_reference_and_exact(mode, iters):
    inputs = data()
    market = reference(**inputs, mode=mode, sinkhorn_iters=iters, post_mult=1.7)
    exact = exact_reference(**inputs, mode=mode, sinkhorn_iters=iters, post_mult=1.7)
    assert len(market) == {"exit": 4, "pre": 3, "post": 1}[mode]
    for a, b in zip(market, exact, strict=True):
        assert b.dtype == torch.float64
        torch.testing.assert_close(a.double(), b, rtol=0, atol=0.04)
    assert not measured_floor(market, market, exact)[0]
    if mode == "pre":
        assert torch.equal(
            market[-1],
            reference(
                None,
                inputs["residual"],
                **{k: inputs[k] for k in ("fn", "scale", "base", "norm_weight")},
                mode="pre",
                sinkhorn_iters=iters,
                post_mult=1.7,
            )[-1],
        )


def test_benchmark_reuses_package_reference():
    inputs = data()
    cfg = MhcConfig(hidden=16)
    weights = {k: inputs[k] for k in ("fn", "scale", "base", "norm_weight")}
    args = [inputs[k] for k in ("x", "residual", "post", "comb")]
    for package_fn, bench_fn in (
        (reference, benchmark_ref.exit_torch_bf16),
        (exact_reference, benchmark_ref.exit_fp64),
    ):
        actual = package_fn(**inputs)
        bench = bench_fn(cfg, weights, *args)
        assert all(
            torch.equal(a, bench[k])
            for a, k in zip(actual, ("residual", "post", "comb", "out"), strict=True)
        )


def test_comb_orientation_and_plain_norm_weight():
    inputs = data(tokens=1)
    inputs["post"].zero_()
    inputs["comb"].zero_()
    inputs["comb"][0, 1, 3] = 2
    exact = exact_reference(**inputs, mode="post")[0]
    assert torch.equal(exact[:, 3], 2 * inputs["residual"][:, 1].double())
    assert torch.count_nonzero(exact[:, :3]) == 0
    inputs["norm_weight"].zero_()
    assert torch.count_nonzero(exact_reference(**inputs)[-1]) == 0


@pytest.mark.parametrize("count", [1, 3, 4])
def test_floor_is_strict_per_output(count):
    exact = tuple(torch.zeros(2) for _ in range(count))
    market = tuple(torch.ones(2) for _ in range(count))
    ours = tuple(torch.ones(2) * 0.5 for _ in range(count))
    failures, stats = measured_floor(ours, market, exact)
    assert not failures and len(stats) == count * 4
    for index in range(count):
        bad = list(ours)
        bad[index] = torch.tensor([2.0, 0])
        failures, _ = measured_floor(bad, market, exact)
        assert len(failures) == 2
    for value in (float("nan"), float("inf")):
        assert measured_floor((torch.tensor([value]),), (torch.ones(1),), (torch.zeros(1),))[0]
    # A per-output slack scales the market bound and floors it at a number of FP32 ULPs.
    twice = ((torch.tensor([1.5]),), (torch.ones(1),), (torch.zeros(1),))
    assert measured_floor(*twice)[0]
    assert not measured_floor(*twice, slack={"residual_out": (2.0, 0)})[0]
    tiny = ((torch.tensor([1.0 + 2**-22]),), (torch.ones(1),), (torch.ones(1),))
    assert (
        measured_floor(*tiny)[0] and not measured_floor(*tiny, slack={"residual_out": (1.0, 2)})[0]
    )
    with pytest.raises(ValueError, match="shapes"):
        measured_floor((torch.zeros(1),), (torch.zeros(2),), (torch.zeros(1),))
    with pytest.raises(ValueError, match="matching"):
        measured_floor((), (), ())


def test_decode_capacities_use_distinct_counter_epochs(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, calls = fake_op(monkeypatch)
    op._counter = torch.zeros(12, dtype=torch.int64)
    op._counter_views = {
        cap: op._counter.narrow(0, 2 * index, 2)
        for index, cap in enumerate((4, 16, 32, 64, 128, 256))
    }
    for tokens in (17, 1, 7, 31, 4, 33, 65, 129, 256, 64):
        op(**data(tokens=tokens))
    # Exit/pre share a capacity's counter and flag, with the same G/epoch.
    inputs = data(tokens=1)
    op.pre(**{k: inputs[k] for k in ("residual", "fn", "scale", "base", "norm_weight")})
    assert calls[-1][-7].data_ptr() == calls[1][-7].data_ptr()
    assert all(args[-7].numel() == 2 for args in calls)
    counters = [args[-7].data_ptr() for args in calls[:-1]]
    assert counters[0] == counters[3]
    assert counters[1] == counters[4]
    assert len(set(counters[:3])) == 3
    assert counters[5] == counters[9]
    assert counters[7] == counters[8]
    assert len(set(counters)) == 6
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    op.close()
    assert not op._counter_views


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("exponent", [-133, -126, -1, 0, 80, 127])
@pytest.mark.parametrize("low", [0, 1, 126])
def test_direct_fp64_bf16_rounding(sign, exponent, low):
    from ctikernels.mhc._reference import round_bf16

    # Both parity directions, normals/subnormals, and a perturbation lost in FP32.
    unit = 2.0 ** max(exponent - 7, -133)
    base = 0.0 if exponent == -133 else 2.0**exponent
    lower, upper = base + low * unit, base + (low + 1) * unit
    midpoint = (lower + upper) / 2
    values = (
        torch.tensor(
            [midpoint - unit * 2**-24, midpoint, midpoint + unit * 2**-24], dtype=torch.float64
        )
        * sign
    )
    even = lower if low % 2 == 0 else upper
    expected = torch.tensor([lower, even, upper], dtype=torch.float64) * sign
    assert torch.equal(round_bf16(values).double(), expected)


def test_direct_fp64_bf16_special_values():
    from ctikernels.mhc._reference import round_bf16

    values = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")]).double()
    result = round_bf16(values)
    assert torch.equal(result[:4].view(torch.int16), values[:4].bfloat16().view(torch.int16))
    assert torch.isnan(result[-1])


@pytest.mark.parametrize("tokens,tile", [(33, 1), (128, 1), (256, 2), (1024, 8), (8192, 16)])
def test_adaptive_prefill_tile(tokens, tile):
    assert inference._prefill_tile(tokens, 48) == tile


def test_prefill_override_and_cache_keys(monkeypatch):
    op, _ = fake_op(monkeypatch)
    assert op._key("exit", 128) == ("exit", 128)
    assert op._key("exit", 1024) == ("exit", -8)
    monkeypatch.setenv("CTI_MHC_PREFILL_TILE", "4")
    forced, _ = fake_op(monkeypatch)
    assert forced._key("pre", 128) == ("pre", -4)
    assert forced._key("exit", 1024) == ("exit", -4)
    explicit = MhcExit("cpu", hidden=16, rms_eps=1e-6, hc_eps=1e-6, norm_eps=1e-6, prefill_tile=8)
    assert explicit._key("exit", 128) == ("exit", -8)
    assert explicit._key("exit", 4) == ("exit", 1004)
    assert explicit._key("post", 128) == ("post", 0)


@pytest.mark.parametrize(
    "tokens,cap",
    [
        (1, 1001),
        (2, 1002),
        (3, 1004),
        (4, 1004),
        (5, 1008),
        (6, 1008),
        (7, 1008),
        (8, 1008),
        (9, 16),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 128),
        (128, 128),
        (129, 256),
        (256, 256),
    ],
)
def test_cluster_and_small_prefill_dispatch(monkeypatch, tokens, cap):
    op, _ = fake_op(monkeypatch)
    assert op._key("exit", tokens) == ("exit", cap)
    assert op._key("pre", tokens)[1] < 1000
    assert op._key("post", tokens) == ("post", 0)
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    grid, _ = fake_op(monkeypatch)
    if tokens <= 8:
        assert grid._key("exit", tokens) == ("exit", 4 if tokens <= 4 else 16)
    else:
        assert grid._key("exit", tokens) == ("exit", cap)


def test_decode_switch_validation(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "invalid")
    with pytest.raises(ValueError, match="CTI_MHC_DECODE"):
        fake_op(monkeypatch)


@pytest.mark.parametrize("hidden,size", [(1024, 128), (4096, 512), (4104, 576)])
def test_cluster_slice_coverage(hidden, size):
    assert inference._cluster_plan(hidden, 8) == (size, 8, 8)
    assert size * 8 >= hidden and size % 64 == 0


@pytest.mark.parametrize("capacity", [64, 128, 256])
def test_small_prefill_policy(capacity):
    size, grid, cap = inference._policy(4096, 48, 99 * 1024, capacity)
    assert (size, grid, cap) == ((128, 32, 256) if capacity == 256 else (64, 64, capacity))


@pytest.mark.parametrize(
    "name,value",
    [
        ("CTI_MHC_EXCHANGE", "all"),
        ("CTI_MHC_PUBLISH", "late"),
        ("CTI_MHC_FENCE", "all"),
        ("CTI_MHC_PROJ", "token"),
        ("CTI_MHC_IKET", "1"),
        ("CTI_MHC_EARLY_TRIGGER", "0"),
        ("CTI_MHC_FASTDIV", "1"),
        ("CTI_MHC_CHAIN", "serial"),
    ],
)
def test_compile_cache_separates_environment_variants(monkeypatch, name, value):
    from ctikernels.mhc._config import kernel_options

    for knob in (
        "CTI_MHC_EXCHANGE",
        "CTI_MHC_PUBLISH",
        "CTI_MHC_FENCE",
        "CTI_MHC_PROJ",
        "CTI_MHC_IKET",
        "CTI_MHC_EARLY_TRIGGER",
        "CTI_MHC_FASTDIV",
        "CTI_MHC_CHAIN",
        "CTI_MHC_AR_PREFETCH",
    ):
        monkeypatch.delenv(knob, raising=False)
    op, _ = fake_op(monkeypatch)
    old = op._compile_key("exit", 1)
    assert old in op._compiled and old[-1][-5:] == ("last", "early", "single", "split", True)
    monkeypatch.setenv(name, value)
    new = op._compile_key("exit", 1)
    assert new[:2] == old[:2] and new[-1] == kernel_options()
    assert new != old and new not in op._compiled
    compiled = []
    kernels = ModuleType("ctikernels.mhc._cute.kernels")
    kernels.compile_kernel = lambda *args, **kwargs: (
        compiled.append(kernel_options()) or (lambda *a: None)
    )
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(op, "_warm", lambda *args: None)
    op.prepare([1])
    assert new in op._compiled and compiled == [kernel_options()] * 3
    op.prepare([1])
    assert len(compiled) == 3
    monkeypatch.delenv(name)
    assert op._compile_key("exit", 1) == old


@pytest.mark.parametrize(
    "name", ["CTI_MHC_EXCHANGE", "CTI_MHC_PUBLISH", "CTI_MHC_FENCE", "CTI_MHC_PROJ"]
)
def test_exchange_switch_validation(monkeypatch, name):
    from ctikernels.mhc._config import kernel_options

    monkeypatch.setenv(name, "invalid")
    with pytest.raises(ValueError, match=name):
        kernel_options()


def test_prepare_reserves_counter_flags_and_aligned_gains(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, _ = fake_op(monkeypatch)
    op._plan = None
    op._compiled.clear()
    props = SimpleNamespace(multi_processor_count=48, shared_memory_per_block_optin=99 * 1024)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: props)
    kernels = ModuleType("ctikernels.mhc._cute.kernels")
    kernels.compile_kernel = lambda *args, **kwargs: lambda *args: None
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(op, "_warm", lambda *args: None)
    op.prepare([1, 17, 33, 129])
    assert op._counter.numel() == 12 and torch.count_nonzero(op._counter) == 0
    for index, (cap, view) in enumerate(op._counter_views.items()):
        assert view.numel() == 2 and view.data_ptr() == op._counter.data_ptr() + 16 * index
        for mode in ("exit", "pre"):
            if mode == "pre" and cap > 32:
                continue
            _, grid, _ = inference._policy(op.hidden, 48, 99 * 1024, cap)
            pitch, size = (grid + 3) // 4 * 4, 8 if mode == "pre" else 4
            gains_offset = cap * 35 * pitch * size
            assert gains_offset % 16 == 0
            assert gains_offset + cap * 5 * size <= op._partials.numel() * 4


class FakeAllReduce:
    def __init__(self, *, active=True, max_bytes=256 * 1024):
        self.active, self.max_bytes = active, max_bytes
        self.device = torch.device("cpu")
        self.words = torch.zeros(4, dtype=torch.int64)
        self.base, self.layout, self.rank = 4096, (8192,), 1
        self.calls, self.prepared = [], []

    def prepare(self, sizes):
        self.prepared.extend(sizes)

    def __call__(self, x, out=None):
        self.calls.append(x)
        self.result = x.clone() if out is None else out.copy_(x)
        return self.result


@pytest.mark.parametrize(
    "active,decode,tokens,limit,fused",
    [
        (True, "grid", 3, 96, True),
        (True, "grid", 32, 1024, True),
        (True, "grid", 64, 2048, True),
        (True, "grid", 128, 4096, True),
        (True, "grid", 256, 8192, True),
        (False, "grid", 3, 96, False),
        (True, "cluster", 3, 96, False),
        (True, "grid", 257, 16384, False),
        (True, "grid", 3, 80, False),
    ],
)
def test_two_node_dispatch(monkeypatch, active, decode, tokens, limit, fused):
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch)
    ar = FakeAllReduce(active=active, max_bytes=limit)
    inputs = data(tokens=tokens)
    op._compiled[op._compile_key("exit", tokens, True)] = lambda *a: launches.append(a)
    op(**inputs, all_reduce=ar)
    assert len(launches) == 1 and len(ar.calls) == (0 if fused else 1)
    args = launches[0]
    assert args[0] == (inputs["x"] if fused else ar.result).data_ptr()
    if fused:
        assert args[-5] is ar.words
        assert args[-4:] == (ar.base, ar.layout[-1], ar.rank, op._ysum.data_ptr())
    else:
        assert args[-5] is op._dummy_words and args[-4:] == (0, 0, 0, 0)
    assert not op._fused(tokens, torch.float32, ar)


def misaligned(value, offset=1):
    """Same values in a view whose storage offset breaks 16-byte alignment."""
    flat = torch.empty(value.numel() + offset, dtype=value.dtype)[offset:]
    return flat.view(value.shape).copy_(value)


def in_stage(op, pointer):
    lo = op._stage_buf.data_ptr()
    return lo <= pointer < lo + op._stage_buf.numel() and pointer % 16 == 0


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("offset", [0, 8, 1])
def test_two_node_local_layout_is_staged_not_rejected(monkeypatch, offset, capture):
    # Alignment is a local fact: a misaligned rank must still join the fused
    # collective, so it is staged into prepared aligned buffers, never raised.
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce()
    inputs = data()
    expected = {k: v.clone() for k, v in inputs.items()}
    inputs["x"] = misaligned(inputs["x"], offset)
    inputs["residual"] = misaligned(inputs["residual"], offset)
    outputs = {
        "residual_out": misaligned(torch.zeros_like(inputs["residual"]), offset),
        "post_out": torch.zeros(3, 4),
        "comb_out": torch.zeros(3, 4, 4),
        "out": torch.zeros(3, 16, dtype=torch.bfloat16),
    }

    def kernel(*args):
        launches.append(args)
        # Write a recognizable value through a staged residual_out argument.
        stage = op._stage_view(3)("residual_out", outputs["residual_out"])
        if args[8] == stage.data_ptr():
            stage.fill_(7)

    op._compiled[op._compile_key("exit", 3, True)] = kernel
    assert op._fused(3, torch.bfloat16, ar)
    result = op(**inputs, **outputs, all_reduce=ar)
    assert len(launches) == 1 and not ar.calls
    args = launches[0]
    assert all(args[i] % 16 == 0 for i in range(12))
    staged = offset % 8 != 0
    assert in_stage(op, args[0]) == staged and in_stage(op, args[1]) == staged
    assert in_stage(op, args[8]) == staged
    if staged:
        view = op._stage_view(3)
        assert torch.equal(view("x", inputs["x"]), expected["x"])
        assert torch.equal(view("residual", inputs["residual"]), expected["residual"])
        assert bool((outputs["residual_out"] == 7).all())
    assert result[0] is outputs["residual_out"]


def test_two_node_output_overlapping_input_is_staged(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, launches = fake_op(monkeypatch)
    ar = FakeAllReduce()
    inputs = data()
    op._compiled[op._compile_key("exit", 3, True)] = lambda *a: launches.append(a)
    op(**inputs, residual_out=inputs["residual"], all_reduce=ar)
    assert len(launches) == 1 and in_stage(op, launches[0][8])


def overlapping_outputs():
    """post_out and comb_out share bytes 16..47 of one buffer."""
    shared = torch.zeros(3 * 16 + 3 * 4 * 4, dtype=torch.float32)
    return shared, {"post_out": shared[:12].view(3, 4), "comb_out": shared[4:52].view(3, 4, 4)}


@pytest.mark.parametrize("case", ["grad", "outputs"])
def test_two_node_local_errors_raise_after_the_collective(monkeypatch, case):
    # Eager: the collective runs first on both ranks, then the local error.
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, launches = fake_op(monkeypatch)
    ar = FakeAllReduce()
    inputs = data()
    outputs = {}
    if case == "grad":
        inputs["x"] = inputs["x"].requires_grad_()
        match = "inference-only"
    else:
        outputs = overlapping_outputs()[1]
        match = "overlap"
    op._compiled[op._compile_key("exit", 3, True)] = lambda *a: launches.append(a)
    with pytest.raises(ValueError, match=match):
        op(**inputs, **outputs, all_reduce=ar)
    # The fused collective launched before the local error surfaced.
    assert len(launches) == 1


@pytest.mark.parametrize("decode,limit", [("grid", 256 * 1024), ("cluster", 80)])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("name", ["fn", "scale", "base", "norm_weight"])
def test_two_node_trainable_weight_rejected_before_collective(
    monkeypatch, name, capture, decode, limit
):
    # Weights are per module and identical across ranks, so both ranks reject
    # a trainable weight before any collective is issued or recorded.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce(max_bytes=limit)
    inputs = data()
    inputs[name] = inputs[name].requires_grad_()
    outputs = {
        "residual_out": torch.empty(3, 4, 16, dtype=torch.bfloat16),
        "post_out": torch.empty(3, 4),
        "comb_out": torch.empty(3, 4, 4),
        "out": torch.empty(3, 16, dtype=torch.bfloat16),
    }
    op._compiled[op._compile_key("exit", 3, True)] = lambda *a: launches.append(a)
    with pytest.raises(ValueError, match="weight requires grad"):
        op(**inputs, **outputs, all_reduce=ar)
    assert not launches and not ar.calls


def test_prepare_rejects_trainable_weights(monkeypatch):
    op, launches = fake_op(monkeypatch)
    weights = {k: v for k, v in data().items() if k in ("fn", "scale", "base", "norm_weight")}
    op.prepare([], weights=weights)
    weights["base"] = weights["base"].requires_grad_()
    with pytest.raises(ValueError, match="weight requires grad"):
        op.prepare([3], all_reduce=FakeAllReduce(), weights=weights)
    assert not launches


@pytest.mark.parametrize("decode,limit", [("grid", 256 * 1024), ("cluster", 80)])
def test_two_node_capture_absorbs_local_errors(monkeypatch, decode, limit):
    # Under capture nothing runs, so a local raise would strand the peer at
    # replay: grad-tracking activations/outputs are used as raw memory and an
    # output overlapping an earlier one is staged and copied back after it.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=True)
    ar = FakeAllReduce(max_bytes=limit)
    fused = op._fused(3, torch.bfloat16, ar)
    inputs = data()
    inputs["x"] = inputs["x"].requires_grad_()
    shared, outputs = overlapping_outputs()
    outputs["residual_out"] = torch.empty(3, 4, 16, dtype=torch.bfloat16).requires_grad_()
    outputs["out"] = torch.empty(3, 16, dtype=torch.bfloat16)

    ones, twos = torch.full((12,), 1.0), torch.full((48,), 2.0)

    def kernel(*args):
        launches.append(args)
        ctypes.memmove(args[9], ones.data_ptr(), 3 * 4 * 4)
        ctypes.memmove(args[10], twos.data_ptr(), 3 * 16 * 4)

    op._compiled[op._compile_key("exit", 3, fused)] = kernel
    result = op(**inputs, **outputs, all_reduce=ar)
    assert len(launches) == 1 and len(ar.calls) == (0 if fused else 1)
    args = launches[0]
    assert args[9] == outputs["post_out"].data_ptr()
    assert in_stage(op, args[10])
    # comb_out is copied back after post_out, so it owns the shared bytes.
    assert torch.equal(shared[:4], torch.ones(4))
    assert torch.equal(shared[4:52], torch.full((48,), 2.0))
    assert result[1] is outputs["post_out"] and result[2] is outputs["comb_out"]


def expanded(name):
    """An output view whose elements share memory (expanded along T)."""
    shapes = {
        "residual_out": ((1, 4, 16), torch.bfloat16),
        "post_out": ((1, 4), torch.float32),
        "comb_out": ((1, 4, 4), torch.float32),
        "out": ((1, 16), torch.bfloat16),
    }
    shape, dtype = shapes[name]
    return torch.zeros(shape, dtype=dtype).expand(3, *shape[1:])


@pytest.mark.parametrize("decode,limit", [("grid", 256 * 1024), ("cluster", 80)])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("name", ["residual_out", "post_out", "comb_out", "out"])
def test_two_node_internal_overlap_output_rejected_before_collective(
    monkeypatch, name, capture, decode, limit
):
    # Layout is identical on every rank under the same-calls contract, so a
    # self-overlapping output is rejected on both ranks before any collective
    # is issued or recorded; its staging copy-back could otherwise raise after.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce(max_bytes=limit)
    fused = op._fused(3, torch.bfloat16, ar)
    outputs = {
        "residual_out": torch.empty(3, 4, 16, dtype=torch.bfloat16),
        "post_out": torch.empty(3, 4),
        "comb_out": torch.empty(3, 4, 4),
        "out": torch.empty(3, 16, dtype=torch.bfloat16),
    }
    outputs[name] = expanded(name)
    op._compiled[op._compile_key("exit", 3, fused)] = lambda *a: launches.append(a)
    with pytest.raises(ValueError, match="internal overlap"):
        op(**data(), **outputs, all_reduce=ar)
    assert not launches and not ar.calls


def test_internal_overlap_output_rejected_without_collective(monkeypatch):
    op, launches = fake_op(monkeypatch)
    with pytest.raises(ValueError, match="internal overlap|contiguous"):
        op(**data(), residual_out=expanded("residual_out"))
    assert not launches


@pytest.mark.parametrize("decode,limit", [("grid", 256 * 1024), ("cluster", 80)])
@pytest.mark.parametrize("capture", [False, True])
def test_two_node_strided_interleaved_outputs_staged(monkeypatch, capture, decode, limit):
    # post_out (stride 2) and comb_out share bytes only past post_out's first
    # contiguous-numel bytes; the byte span from strides must still see it.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce(max_bytes=limit)
    fused = op._fused(3, torch.bfloat16, ar)
    shared = torch.zeros(64)
    outputs = {
        "post_out": shared[:24:2].view(3, 4),
        "comb_out": shared[16:64].view(3, 4, 4),
        "residual_out": torch.empty(3, 4, 16, dtype=torch.bfloat16),
        "out": torch.empty(3, 16, dtype=torch.bfloat16),
    }
    ones, twos = torch.full((12,), 1.0), torch.full((48,), 2.0)

    def kernel(*args):
        launches.append(args)
        ctypes.memmove(args[9], ones.data_ptr(), 12 * 4)
        ctypes.memmove(args[10], twos.data_ptr(), 48 * 4)

    op._compiled[op._compile_key("exit", 3, fused)] = kernel
    if not capture:
        with pytest.raises(ValueError, match="overlap"):
            op(**data(), **outputs, all_reduce=ar)
        # Eagerly the collective (fused launch or provider) ran before the raise.
        assert len(launches if fused else ar.calls) == 1
        return
    op(**data(), **outputs, all_reduce=ar)
    assert len(launches) == 1 and len(ar.calls) == (0 if fused else 1)
    assert in_stage(op, launches[0][9]) and in_stage(op, launches[0][10])
    # comb_out is copied back after post_out, so it owns the shared bytes.
    assert torch.equal(shared[0:16:2], torch.ones(8))
    assert torch.equal(shared[1:16:2], torch.zeros(8))
    assert torch.equal(shared[16:64], twos)


@pytest.mark.parametrize("decode", ["grid", "cluster"])
@pytest.mark.parametrize("capture", [False, True])
def test_two_node_unfused_stages_after_the_provider(monkeypatch, capture, decode):
    # Unfused buckets (cluster decode, sizes above the provider's fused limit,
    # an inactive provider routed to NCCL) stage a rank-local misaligned
    # residual/output instead of raising: under capture into the staging
    # prepared for T, eagerly into fresh buffers.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce(max_bytes=80)
    inputs = data()
    inputs["residual"] = misaligned(inputs["residual"])
    outputs = {
        "residual_out": misaligned(torch.zeros(3, 4, 16, dtype=torch.bfloat16)),
        "post_out": torch.empty(3, 4),
        "comb_out": torch.empty(3, 4, 4),
        "out": torch.empty(3, 16, dtype=torch.bfloat16),
    }
    assert not op._fused(3, torch.bfloat16, ar)
    staged = {}

    def kernel(*args):
        launches.append(args)
        # The staged residual_out buffer must be copied back to the caller's.
        staged["residual_out"] = args[8]
        ctypes.memset(args[8], 0x3F, 3 * 4 * 16 * 2)

    op._compiled[op._compile_key("exit", 3)] = kernel
    result = op(**inputs, **outputs, all_reduce=ar)
    assert len(ar.calls) == 1 and len(launches) == 1
    assert launches[0][0] == ar.result.data_ptr()
    assert launches[0][1] % 16 == 0 and launches[0][8] % 16 == 0
    assert in_stage(op, launches[0][1]) == capture
    assert in_stage(op, launches[0][8]) == capture
    assert result[0] is outputs["residual_out"]
    assert torch.all(outputs["residual_out"].view(torch.int16) == 0x3F3F)


def test_two_node_unfused_capture_needs_prepared_staging(monkeypatch):
    # Prepare state is rank-symmetric, so this raises on both ranks before
    # the provider's collective.
    monkeypatch.setenv("CTI_MHC_DECODE", "cluster")
    op, launches = fake_op(monkeypatch, capture=True)
    op._stage_buf, op._stage_cap = None, 0
    ar = FakeAllReduce(max_bytes=80)
    inputs = data()
    outputs = {
        "residual_out": torch.empty(3, 4, 16, dtype=torch.bfloat16),
        "post_out": torch.empty(3, 4),
        "comb_out": torch.empty(3, 4, 4),
        "out": torch.empty(3, 16, dtype=torch.bfloat16),
    }
    with pytest.raises(ValueError, match="prepare"):
        op(**inputs, **outputs, all_reduce=ar)
    assert not ar.calls and not launches


def test_two_node_prepare_sizes_unfused_staging(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "cluster")
    op, launches = fake_op(monkeypatch)
    op._stage_buf, op._stage_cap = None, 0
    kernels = ModuleType("ctikernels.mhc._cute.kernels")
    kernels.compile_kernel = lambda *args, **kwargs: lambda *a: launches.append(a)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(op, "_warm", lambda *args: None)
    ar = FakeAllReduce(max_bytes=80)
    # An eager first call prepares kernels but no unfused staging.
    op(**data(), all_reduce=ar)
    assert op._stage_cap == 0
    op.prepare([3, 5], all_reduce=ar)
    assert op._stage_cap == 5
    first = op._stage_buf
    op.prepare([7], all_reduce=ar)
    # Superseded staging stays alive for graphs captured against it.
    assert op._stage_cap == 7 and op._stage_old[-1] is first


def test_two_node_prepare_compile_only_and_capture(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, launches = fake_op(monkeypatch)
    ar = FakeAllReduce(max_bytes=2048)
    compiled, warmed = [], []
    kernels = ModuleType("ctikernels.mhc._cute.kernels")

    def compile_kernel(*args, **kwargs):
        compiled.append(kwargs)
        return lambda *args: launches.append(args)

    kernels.compile_kernel = compile_kernel
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(op, "_warm", lambda *args: warmed.append(args))
    workspace = op._ysum
    op.prepare([1, 3, 64, 257], all_reduce=ar)
    assert ar.prepared == [32, 96, 2048]
    assert compiled == [{"allreduce": True}] * 2
    assert not warmed and not launches and not ar.calls
    assert op._ysum is workspace
    assert op._compile_key("exit", 3) != op._compile_key("exit", 3, True)
    inputs = data()
    outputs = dict(
        zip(
            ("residual_out", "post_out", "comb_out", "out"),
            (torch.empty_like(inputs[k]) for k in ("residual", "post", "comb", "x")),
            strict=True,
        )
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op(**inputs, **outputs, all_reduce=ar)
    assert len(launches) == 1 and not ar.calls
    del op._compiled[op._compile_key("exit", 3, True)]
    with pytest.raises(ValueError, match="prepare"):
        op(**inputs, **outputs, all_reduce=ar)


def test_two_node_prepare_allocates_workspace_once(monkeypatch):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, _ = fake_op(monkeypatch)
    op._plan = None
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(
            multi_processor_count=48, shared_memory_per_block_optin=99 * 1024
        ),
    )
    kernels = ModuleType("ctikernels.mhc._cute.kernels")
    kernels.compile_kernel = lambda *args, **kwargs: lambda *args: None
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    op.prepare([3], all_reduce=FakeAllReduce())
    workspace = op._ysum
    assert workspace.shape == (256, 16) and workspace.dtype == torch.bfloat16
    op.prepare([64], all_reduce=FakeAllReduce())
    assert op._ysum is workspace


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False)])
def test_ar_prefetch_option(monkeypatch, value, expected):
    from ctikernels.mhc._config import kernel_options

    monkeypatch.delenv("CTI_MHC_AR_PREFETCH", raising=False)
    op, _ = fake_op(monkeypatch)
    default = op._compile_key("exit", 3, True)
    if value is not None:
        monkeypatch.setenv("CTI_MHC_AR_PREFETCH", value)
    assert kernel_options()[-1] is expected
    assert (op._compile_key("exit", 3, True) == default) is expected
    monkeypatch.setenv("CTI_MHC_AR_PREFETCH", "yes")
    with pytest.raises(ValueError, match="CTI_MHC_AR_PREFETCH"):
        kernel_options()


def kernel_ast():
    import ast
    from pathlib import Path

    return ast.parse((Path(inference.__file__).parent / "_cute" / "kernels.py").read_text())


@pytest.mark.parametrize(
    "mode,decode,cluster,cap",
    [
        ("pre", True, False, 4),
        ("post", True, False, 4),
        ("exit", False, False, 4),
        ("exit", True, True, 4),
        ("exit", True, False, 8),
    ],
)
def test_compile_rejects_invalid_allreduce(mode, decode, cluster, cap):
    import ast

    # Execute the real entry validation without importing CuTe or probing CUDA.
    node = next(
        n
        for n in kernel_ast().body
        if isinstance(n, ast.FunctionDef) and n.name == "compile_kernel"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<compile_kernel>", "exec"), namespace)
    with pytest.raises(ValueError, match="allreduce requires"):
        namespace["compile_kernel"](
            None, (64, 8, cap), mode, decode, cluster=cluster, allreduce=True
        )


def test_allreduce_protocol_is_constexpr_decode_only():
    import ast

    tree = kernel_ast()
    kernel = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Kernel")
    decode = next(
        n for n in kernel.body if isinstance(n, ast.FunctionDef) and n.name == "decode_kernel"
    )
    guarded = next(
        n
        for n in decode.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "const_expr(self.allreduce)"
    )
    protocol = ast.unparse(guarded)
    for text in (
        "word_base + 8",
        "word_base + 16",
        "word_base + 24",
        "seq << 24",
        "seq + 1",
        "fence_acq_rel_sys",
        "cop='wt'",
        "cop='cv'",
        "_add_rn",
        "x = ysum",
        "const_expr(self.ar_prefetch)",
        "prefetch.global.L2",
        "mhc.ar_send",
        "mhc.ar_wait",
        "mhc.ar_sum",
    ):
        assert text in protocol
    decode.body.remove(guarded)
    remaining = ast.unparse(tree)
    assert "doorbell" not in remaining and "word_base + 16" not in remaining
    assert "cop='nc'" not in remaining


@pytest.mark.parametrize("closed", [False, True])
def test_two_node_provider_validation(monkeypatch, closed):
    monkeypatch.setenv("CTI_MHC_DECODE", "grid")
    op, launches = fake_op(monkeypatch)
    ar = FakeAllReduce()
    if closed:
        ar._closed = True
    else:
        ar.device = torch.device("cuda:1")
    with pytest.raises(
        RuntimeError if closed else ValueError, match="closed" if closed else "configured device"
    ):
        op(**data(), all_reduce=ar)
    assert not ar.calls and not launches


GRAD_MODES = {
    "inference": torch.inference_mode,
    "no_grad": torch.no_grad,
    "grad": torch.enable_grad,
}


@pytest.mark.parametrize("mode", list(GRAD_MODES))
@pytest.mark.parametrize("decode,limit", [("grid", 256 * 1024), ("cluster", 80)])
@pytest.mark.parametrize("capture", [False, True])
def test_two_node_copy_back_ignores_tensor_flags(monkeypatch, capture, decode, limit, mode):
    # Inference mode, grad mode and version counters are not layout facts and
    # may differ between ranks: staging and copy-back are raw device copies, so
    # outputs and staging created under any grad mode never raise after the
    # collective (fused launch or provider) was issued or recorded.
    monkeypatch.setenv("CTI_MHC_DECODE", decode)
    op, launches = fake_op(monkeypatch, capture=capture)
    ar = FakeAllReduce(max_bytes=limit)
    fused = op._fused(3, torch.bfloat16, ar)
    inputs = data()
    with GRAD_MODES[mode]():
        op._stage_buf = torch.zeros_like(op._stage_buf)
        inputs["residual"] = misaligned(inputs["residual"])
        outputs = {
            "residual_out": misaligned(torch.zeros(3, 4, 16, dtype=torch.bfloat16)),
            "post_out": torch.zeros(3, 4),
            "comb_out": torch.zeros(3, 4, 4),
            "out": torch.zeros(3, 16, dtype=torch.bfloat16),
        }
    assert outputs["out"].is_inference() == (mode == "inference")

    def kernel(*args):
        launches.append(args)
        ctypes.memset(args[8], 0x3F, 3 * 4 * 16 * 2)

    op._compiled[op._compile_key("exit", 3, fused)] = kernel
    op(**inputs, **outputs, all_reduce=ar)
    assert len(launches) == 1 and len(ar.calls) == (0 if fused else 1)
    assert launches[0][1] % 16 == 0 and launches[0][8] % 16 == 0
    assert torch.all(outputs["residual_out"].view(torch.int16) == 0x3F3F)
