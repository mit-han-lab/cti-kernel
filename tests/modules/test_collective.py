# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU dispatch, scheduling and numerical contract tests."""

import pytest
import torch

from ctikernels.collective import inference
from ctikernels.collective._host_exchange.cute import exchange


def make_op(**kwargs):
    args = {
        "rank": 0,
        "world_size": 1,
        "cpu_group": None,
        "device": "cpu",
        "hidden": 8,
        "max_tokens": 64,
        "eps": 1e-6,
    }
    return inference.AllReduceAddRMSNorm(**(args | kwargs))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("name", "RTX 5090", "RTX PRO 6000"),
        ("capability", (12, 1), "sm_120"),
        ("world_size", 1, "world_size"),
        ("hidden", 1024, "hidden"),
        ("dtype", torch.float16, "bfloat16"),
        ("cute_available", False, "cutlass"),
    ],
)
def test_support_reasons(field, value, reason):
    args = {
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "capability": (12, 0),
        "world_size": 2,
        "hidden": 2048,
        "dtype": torch.bfloat16,
        "zero_centered": True,
        "cute_available": True,
    }
    assert inference._support_reason(**args) is None
    args[field] = value
    assert reason in inference._support_reason(**args)


def test_support_both_norm_forms():
    args = {
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "capability": (12, 0),
        "world_size": 2,
        "hidden": 4096,
        "dtype": torch.bfloat16,
        "cute_available": True,
    }
    for zero_centered in (True, False):
        assert inference._support_reason(**args, zero_centered=zero_centered) is None


def test_compile_key_separates_norm_forms(monkeypatch):
    import sys
    import types

    built = []

    class FakeKernel:
        def __init__(self, *args, **kwargs):
            built.append(kwargs["zero_centered"])

    fake_kernels = types.SimpleNamespace(FlagKernel=FakeKernel, PCKernel=FakeKernel)
    cute = pytest.importorskip("cutlass.cute")
    package = sys.modules["ctikernels.collective._host_exchange.cute"]
    monkeypatch.setitem(sys.modules, f"{package.__name__}.kernels", fake_kernels)
    monkeypatch.setattr(package, "kernels", fake_kernels, raising=False)
    monkeypatch.setattr(cute, "compile", lambda op, *args, **kwargs: op)
    monkeypatch.setattr(cute.runtime, "make_fake_stream", lambda **kwargs: None)
    monkeypatch.setattr(cute.runtime, "make_fake_tensor", lambda *args, **kwargs: None)
    keys = set()
    for zero_centered in (True, False):
        compiler = exchange._KernelCompiler(2048, 188, 2, 0, 188, 35, zero_centered)
        compiler.prepare([1])
        keys |= set(compiler.compiled)
    assert built == [True, False]
    assert {key[-1] for key in keys} == {True, False}


def test_support_lazy(monkeypatch):
    assert "CUDA" in inference.support(
        "cpu", world_size=2, hidden=2048, dtype=torch.bfloat16, zero_centered=True
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "RTX PRO 6000")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(inference, "_cute_available", lambda: False)
    assert "cutlass" in inference.support(
        "cuda:0", world_size=2, hidden=2048, dtype=torch.bfloat16, zero_centered=True
    )


@pytest.mark.parametrize("hidden", [2048, 4096, 5120])
def test_schedule_boundaries(hidden):
    for rows in [1, 188, 189, 188 * 44]:
        cfg = exchange.schedule(rows, hidden, slots=188, link_rate=20)
        assert cfg[0] == exchange.FLAG
        assert cfg[3] == -(-rows // 188)
    assert exchange.schedule(188 * 44 + 1, hidden, slots=188, link_rate=20)[0] == exchange.PC
    for rows in range(1, 9):
        assert exchange.schedule(rows, 2048, slots=188, link_rate=35) == ("flag", 16, 0, 1)


def test_schedule_pacing_and_padding():
    for rows in range(1, 189):
        padded = next((b for b in exchange.GRAPH_BUCKETS if b >= rows), rows)
        cfg = exchange.schedule(rows, 2048, slots=188, link_rate=20)
        assert (cfg[2] > 0) == (padded * 4096 > 20 * exchange.PACE_FROM_NS)
        assert cfg == exchange.schedule(padded, 2048, slots=188, link_rate=20)
    # Measured independent-link exceptions intentionally pace below the threshold.
    assert exchange.schedule(5, 5120, slots=188, link_rate=35)[2] > 0
    assert exchange.schedule(6, 5120, slots=188, link_rate=35)[2] == 0
    assert exchange.schedule(1, 4096, slots=188, link_rate=35)[2] == 0


def test_prepare_slices_without_launch(monkeypatch):
    seen = []
    monkeypatch.setattr(
        exchange._KernelCompiler, "_kernel", lambda self, cfg, h: seen.append((cfg, h))
    )
    exchange.compile_kernels(2048, [0, 200], 188, 2, 0, rows_cap=188, link_rate=35)
    assert seen == [(exchange.schedule(t, 2048, slots=188, link_rate=35), 2048) for t in (188, 12)]
    compiler = exchange._KernelCompiler(4096, 188, 2, 0, 188, 35)
    compiler.prepare([1], hidden=2048)
    assert seen[-1][1] == 2048
    with pytest.raises(ValueError):
        compiler.prepare([-1])


@pytest.mark.parametrize("zero_centered", [True, False])
def test_reference_fp64(zero_centered):
    g = torch.Generator().manual_seed(23)
    p = torch.randn(2, 16, 256, generator=g).bfloat16()
    r = torch.randn(16, 256, generator=g).bfloat16()
    w = torch.randn(256, generator=g).bfloat16()
    x, rout = inference.reference(p, r, w, 1e-6, zero_centered=zero_centered)
    h = r.double() + p[0].double() + p[1].double()
    gain = w.double() + int(zero_centered)
    oracle = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * gain
    # Half an output ULP plus the much smaller FP32 normalization error.
    spacing = (torch.nextafter(x.abs(), torch.full_like(x, float("inf"))) - x.abs()).double()
    assert torch.all((x.double() - oracle).abs() <= 0.501 * spacing + 1e-7)
    assert torch.equal(rout, h.bfloat16())


def test_reference_preserves_rank_order():
    r = torch.ones(1, 8)
    p = torch.stack([torch.full_like(r, 2**25), torch.full_like(r, -(2**25))])
    _, result = inference.reference(p, r, torch.zeros(8), 1e-6)
    h = r.float()
    for partial in p:
        h = h + partial.float()
    assert torch.equal(result, h)
    assert not torch.equal(result, r + p.sum(0))


def test_measured_floor_rounds_once_vs_market():
    g = torch.Generator().manual_seed(29)
    p = torch.randn(2, 32, 512, generator=g).bfloat16()
    r = torch.randn(32, 512, generator=g).bfloat16()
    w = torch.randn(512, generator=g).bfloat16()
    exact = inference.exact_reference(p, r, w, 1e-6)
    summed = (p[0].float() + p[1].float()).bfloat16()
    market = inference.market_composition(summed, r, w, 1e-6)
    ours = inference.reference(p, r, w, 1e-6)
    failures, stats = inference.measured_floor(ours, market, exact)
    assert not failures
    assert stats["residual_rms"] < stats["residual_rms_market"]
    assert stats["x_rms"] < stats["x_rms_market"]
    failures, _ = inference.measured_floor(market, ours, exact)
    assert failures
    # Inputs are untouched and the market residual is the BF16 add.
    assert torch.equal(market[1], summed + r)


def test_dispatch_lifecycle_and_validation(monkeypatch):
    op = make_op()
    assert op.provider == "local"
    op.prepare([1, 200])
    x = torch.zeros(200, 8, dtype=torch.bfloat16)
    r, w = x.clone(), torch.zeros(8, dtype=torch.bfloat16)
    called = []
    monkeypatch.setattr(
        "ctikernels.norm.inference.add_rms_norm_", lambda *args, **kw: called.append(kw) or args[:2]
    )
    assert op(x, r, w) == (x, r)
    assert called == [{"zero_centered": True}]
    op(x[:0], r[:0], w)
    assert len(called) == 1
    for tensors in [(x[:, :7], r, w), (x, r, w[:7])]:
        with pytest.raises(ValueError):
            op(*tensors)
    for index, error in ((0, RuntimeError), (1, RuntimeError), (2, ValueError)):
        tensors = [x, r, w]
        tensors[index] = tensors[index].clone().requires_grad_()
        with torch.no_grad(), pytest.raises(error, match="gradients"):
            op(*tensors)
    op.close()
    op.close()
    with pytest.raises(RuntimeError, match="after close"):
        op(x, r, w)
    with pytest.raises(ValueError, match="communicator"):
        make_op(world_size=2)
    with pytest.raises(ValueError, match="world_size"):
        make_op(world_size=2, provider="local")
    with pytest.raises(NotImplementedError):
        make_op(world_size=2, provider="cute")


def test_nccl_calls_reduce_before_norm(monkeypatch):
    calls = []

    class Communicator:
        def all_reduce(self, x, op):
            calls.append(op)

    monkeypatch.setattr(
        "ctikernels.norm.inference.add_rms_norm_", lambda *a, **k: calls.append("norm") or a[:2]
    )
    with make_op(world_size=2, communicator=Communicator()) as op:
        assert op.provider == "nccl"
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        op(x, x.clone(), x[0])
    assert calls == ["sum", "norm"]
