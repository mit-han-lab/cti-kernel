# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU contracts: dispatch, fallback, dependency/capture guards and frozen weights."""

import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ctikernels.linear import Linear, _autotune, cublas, cute, inference
from ctikernels.linear import _dispatch as dispatch
from ctikernels.linear.cute import _provider

from ..support.child import run_child


@pytest.fixture(autouse=True)
def device_name(monkeypatch):
    # Mocked hardware tuples have no CUDA device; the autotune key needs a name.
    monkeypatch.setattr(_autotune, "_device_name", lambda _: "NVIDIA test device")


def test_public_export():
    import ctikernels.linear as linear

    assert linear.__all__ == [
        "MAX_M",
        "Linear",
        "LoRAChunkTable",
        "autotune",
        "autotune_entries",
        "gated_input",
        "last_config",
        "lora_delta_factors_",
        "lora_segment_metadata",
        "lora_segmented_delta_",
        "lora_segmented_projection",
        "release_cublas_workspaces",
        "release_scratch",
        "scratch_bytes",
    ]
    assert Linear is inference.Linear
    assert linear.release_scratch is inference.release_scratch
    assert linear.MAX_M == inference.MAX_M == 64
    assert inference.__all__ == ["Linear"]


@pytest.mark.parametrize(
    "kind,cap,sms",
    [
        ("rtx5090", (12, 0), 170),
        ("pro6000", (12, 0), 188),
        ("gb10", (12, 1), 48),
    ],
)
def test_allowlist(kind, cap, sms):
    assert dispatch.device_class(cap, sms) == kind
    assert dispatch.device_class(cap, sms + 1) is None
    assert dispatch.device_class((cap[0], cap[1] + 1), sms) is None
    for n, k in dispatch.QWEN_SHAPES:
        for m in range(1, 65):
            assert dispatch.accepts(kind, n, k, m) == (
                dispatch.tuned_config(kind, n, k, m) is not None
            )
        assert not dispatch.accepts(kind, n, k, 65)
        assert not dispatch.accepts(kind, n, k, 0)
    assert not dispatch.accepts(kind, 111, 888, 1)
    assert not dispatch.accepts(None, 4096, 4096, 1)


def test_pruned_rows(monkeypatch):
    monkeypatch.setitem(
        dispatch.TUNED["rtx5090"],
        (4096, 4096),
        {(1, 1): dispatch.Config(), (3, 4): dispatch.Config(splits=2)},
    )
    assert dispatch.accepts("rtx5090", 4096, 4096, 3)
    assert not dispatch.accepts("rtx5090", 4096, 4096, 2)


@pytest.mark.parametrize("override", ["auto", "cublas", "cute"])
@pytest.mark.parametrize("m", [0, 1, 3, 8, 9, 128])
def test_cpu_fallback_and_reporting(monkeypatch, override, m):
    monkeypatch.setenv(inference.VARIABLE, override)
    op = Linear()
    assert (op.provider, op.kernels_per_call) == ("cublas", 1)
    x, w = torch.randn(m, 16).bfloat16(), torch.randn(17, 16).bfloat16()
    expected = torch.nn.functional.linear(x, w)
    out = torch.empty_like(expected)
    assert op(x, w, out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(op(x, w), expected, rtol=0, atol=0)
    assert (op.provider, op.kernels_per_call) == ("cublas", 1)
    with pytest.raises(AttributeError):
        op.provider = "cute"
    with pytest.raises(AttributeError):
        op.kernels_per_call = 2


@pytest.mark.parametrize("provider", ["auto", "cute", "cublas"])
def test_constructor_provider_overrides_environment(monkeypatch, provider):
    monkeypatch.setenv(inference.VARIABLE, "invalid")
    x, w = torch.randn(9, 8).bfloat16(), torch.randn(4, 8).bfloat16()
    op = Linear(provider=provider)
    torch.testing.assert_close(op(x, w), torch.nn.functional.linear(x, w))
    assert op.provider == "cublas"
    with pytest.raises(ValueError, match="provider must be auto, cute or cublas"):
        Linear(provider="invalid")


def test_frozen_validation_before_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA inspected before weight validation")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
    with pytest.raises(ValueError, match="frozen"):
        Linear()(torch.zeros(2, 8), torch.zeros(4, 8, requires_grad=True))


def test_arguments_and_autograd(monkeypatch):
    op = Linear()
    x, w = torch.randn(2, 8), torch.randn(4, 8)
    for bad in [torch.empty(3, 8), torch.empty(2, 4, dtype=torch.float64), torch.empty(4, 2).t()]:
        with pytest.raises(ValueError, match="out"):
            op(x, w, out=bad)
    with pytest.raises(ValueError, match="share"):
        op(x, w, out=x.view(-1)[:8].view(2, 4))
    # F.linear's range: bias and leading dims.
    bias = torch.randn(4)
    torch.testing.assert_close(op(x, w, bias=bias), torch.nn.functional.linear(x, w, bias))
    x3 = torch.randn(3, 2, 8)
    torch.testing.assert_close(op(x3, w), torch.nn.functional.linear(x3, w))
    torch.testing.assert_close(op(x3, w, bias=bias), torch.nn.functional.linear(x3, w, bias))
    torch.testing.assert_close(op(x[0], w), torch.nn.functional.linear(x[0], w))
    out3 = torch.empty(3, 2, 4)
    assert op(x3, w, out3) is out3
    torch.testing.assert_close(out3, torch.nn.functional.linear(x3, w))
    with pytest.raises(ValueError, match="frozen"):
        op(x, w, bias=torch.zeros(4, requires_grad=True))
    with pytest.raises(ValueError, match="bias"):
        op(x, w, bias=torch.zeros(5))
    # Bias calls keep the bias-free contracts: no aliasing, explicit dtypes, empty dims.
    xb, wb = torch.tensor([[1.0, 2.0]]).bfloat16(), torch.eye(2).bfloat16()
    with pytest.raises(ValueError, match="share"):
        op(xb, wb, xb, bias=torch.tensor([3.0, 4.0]).bfloat16())
    with pytest.raises(ValueError, match="dtype"):
        op(xb, wb, torch.empty(1, 2), out_dtype=torch.bfloat16, bias=torch.zeros(2).bfloat16())
    torch.testing.assert_close(
        op(xb, wb, out_dtype=torch.float32, bias=torch.ones(2).bfloat16()),
        torch.tensor([[2.0, 3.0]]),
    )
    empty_k = torch.empty(2, 0)
    torch.testing.assert_close(
        op(empty_k, torch.empty(3, 0), bias=torch.arange(3.0)),
        torch.nn.functional.linear(empty_k, torch.empty(3, 0), torch.arange(3.0)),
    )
    assert op(torch.empty(2, 3, 8), torch.empty(0, 8)).shape == (2, 3, 0)
    # Bias calls keep the output-layout rule: overlapping rows are rejected, not misrun.
    xa, eye = torch.arange(4.0).view(2, 2), torch.eye(2)
    for b in (None, torch.tensor([10.0, 11.0])):
        with pytest.raises(ValueError, match="row-strided"):
            op(xa, eye, torch.empty(3).as_strided((2, 2), (1, 1)), bias=b)
        # Reshaping copies of x do not hide storage shared with out.
        base = torch.arange(12.0).view(2, 3, 2)
        with pytest.raises(ValueError, match="share"):
            op(base.transpose(0, 1), eye, base.view(3, 2, 2), bias=b)
        with pytest.raises(ValueError, match="workspace"):
            op(base.transpose(0, 1), eye, bias=b, workspace=base)
    with pytest.raises(ValueError, match="expected"):
        op(x, w[:, :4])
    with pytest.raises(ValueError, match="dtype"):
        op(x, w.double())
    monkeypatch.setenv(inference.VARIABLE, "bad")
    torch.testing.assert_close(op(x, w), torch.nn.functional.linear(x, w))
    with pytest.raises(ValueError, match="must be"):
        Linear()
    monkeypatch.delenv(inference.VARIABLE)
    x.requires_grad_()
    op(x, w).sum().backward()
    torch.testing.assert_close(x.grad, w.sum(0).expand_as(x))


def test_capture_miss_does_not_compile(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(_provider, "_COMPILED", {})
    monkeypatch.setattr(_provider, "compile_kernel", lambda *args: calls.append(args) or sentinel)
    args = (torch.device("cuda:0"), 64, 4096, dispatch.Config(), (12, 0))
    assert _provider.get_compiled(*args, capturing=True) is None
    assert calls == []
    assert _provider.get_compiled(*args, capturing=False) is sentinel
    assert _provider.get_compiled(*args, capturing=True) is sentinel
    assert len(calls) == 1
    assert _provider.get_compiled(torch.device("cuda:1"), *args[1:], capturing=True) is None


def test_selection_and_force(monkeypatch):
    # Test host dispatch independently of CUDA allocation and DSL imports.
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (4096, 4096), {(3, 4): dispatch.Config()})
    x = SimpleNamespace(shape=(3, 4096), device=torch.device("cuda:0"))
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    calls = []
    monkeypatch.setattr(cute, "get_compiled", lambda *args: calls.append(args) or None)
    assert Linear(provider="auto")._resolve(x.device, 4096, 4096, 4) is False
    assert len(calls) == 1  # allowlisted, but a capture miss
    Linear(provider="auto")._resolve(x.device, 17, 4096, 4)
    assert len(calls) == 1
    Linear(provider="cute")._resolve(x.device, 17, 4096, 4)
    assert len(calls) == 2
    monkeypatch.setattr(cute, "available", lambda: False)
    with pytest.raises(RuntimeError, match="4.8.0"):
        Linear(provider="cute")._resolve(x.device, 17, 4096, 4)


def test_configs():
    for kind, sms in [("rtx5090", 170), ("pro6000", 188), ("gb10", 48)]:
        for n, k in dispatch.QWEN_SHAPES:
            cfg = dispatch.config_for(kind, sms, n, k)
            assert cfg.shared_bytes <= 96 * 1024
            assert cfg.splits & (cfg.splits - 1) == 0
        assert dispatch.config_for(kind, sms, 64, 4096).splits > 1
        assert dispatch.config_for(kind, sms, 248320, 4096).splits == 1
    with pytest.raises(ValueError):
        dispatch.Config(rows=48)
    with pytest.raises(ValueError):
        dispatch.Config(rows=128, tile_k=256, stages=3)


def test_config_override(monkeypatch):
    x = SimpleNamespace(shape=(2, 4096), device=torch.device("cuda:0"))
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    configs = []
    monkeypatch.setattr(cute, "get_compiled", lambda *args: configs.append(args[3]) or None)
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "16,64,3,8")
    Linear(provider="cute")._resolve(x.device, 64, 4096, 2)
    assert configs == [dispatch.Config(16, 64, 3, 8)]
    Linear(provider="cute", config=dispatch.Config())._resolve(x.device, 64, 4096, 2)
    assert configs[-1] == dispatch.Config()  # explicit configuration wins
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "64,128")
    with pytest.raises(ValueError, match="needs"):
        Linear(provider="cute")._resolve(x.device, 64, 4096, 2)


def test_unmeasured_devices_shapes_and_m():
    assert not dispatch.accepts("unknown", 24576, 4096, 1)
    assert not dispatch.accepts("rtx5090", 111, 888, 1)
    for shapes in dispatch.TUNED.values():
        for ranges in shapes.values():
            seen = set()
            for (lo, hi), cfg in ranges.items():
                assert 1 <= lo <= hi <= 64
                assert not seen.intersection(range(lo, hi + 1))
                seen.update(range(lo, hi + 1))
                assert isinstance(cfg, dispatch.Config)


def test_auto_uses_table_even_with_tuning_overrides(monkeypatch):
    x = SimpleNamespace(shape=(3, 4096), device=torch.device("cuda:0"))
    selected = dispatch.Config(16, 64, 3, 8)
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (4096, 4096), {(3, 4): selected})
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    calls = []
    monkeypatch.setattr(cute, "get_compiled", lambda *args: calls.append(args) or None)
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "128,128,2,1,8")
    Linear(provider="auto", config=dispatch.Config())._resolve(
        x.device, 4096, 4096, dispatch.m_bucket(x.shape[0])[1]
    )
    assert calls[0][3] == selected
    x.shape = (2, 4096)
    Linear(provider="auto", config=dispatch.Config())._resolve(
        x.device, 4096, 4096, dispatch.m_bucket(x.shape[0])[1]
    )
    assert len(calls) == 1


def test_big_n_candidates_are_bounded():
    for n in (24576, 34816, 248320):
        candidates = dispatch.tune_configs(dispatch.config_for("rtx5090", 170, n, 4096), n)
        assert len(candidates) <= 10
        assert any(c.rows == 128 and c.warps == 8 for c in candidates)
        assert any(c.stages == 4 for c in candidates)
        assert all(c.shared_bytes <= 96 * 1024 for c in candidates)
    with pytest.raises(ValueError, match="warps"):
        dispatch.Config(64, 128, 2, 1, 8)


@pytest.mark.parametrize("n,k", dispatch.QWEN_SHAPES + ((1, 8), (67, 264), (129, 520)))
def test_gb10_streaming_grid(n, k):
    configs = dispatch.tune_configs(dispatch.config_for("gb10", 48, n, k), n, k, "gb10")
    assert len(configs) == len(set(configs)) == 10
    assert configs[0] == dispatch.Config(64, 256, 2, 1, 4)
    assert sum(c.variant == "reg" for c in configs) == 6
    for cfg in configs:
        assert cfg.shared_bytes <= 96 * 1024 < 99 * 1024
        chunks = (k + cfg.splits * cfg.tile_k - 1) // (cfg.splits * cfg.tile_k)
        assert (cfg.splits - 1) * chunks * cfg.tile_k < k
        if cfg.variant == "reg":
            assert cfg.rows // cfg.warps in (16, 32)
            if n >= 4096 and k >= 2048:
                assert (n + cfg.rows - 1) // cfg.rows * cfg.splits * cfg.warps >= 24 * 48


def test_extended_config_limits():
    assert dispatch.Config(16, 512, 3).shared_bytes == 74880
    assert dispatch.Config(16, 128, 6).shared_bytes == 39168
    for args in [(16, 512, 4), (32, 512, 3), (16, 128, 7)]:
        with pytest.raises(ValueError):
            dispatch.Config(*args)


def test_gb10_missing_m_falls_back_without_compilation(monkeypatch):
    x = SimpleNamespace(shape=(2, 4096), device=torch.device("cuda:0"))
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("gb10", (12, 1), 48))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setitem(dispatch.TUNED, "gb10", {(24576, 4096): {(1, 1): dispatch.Config()}})

    def forbidden(*args, **kwargs):
        raise AssertionError("unmeasured GB10 row must not compile")

    monkeypatch.setattr(cute, "get_compiled", forbidden)
    assert Linear(provider="auto")._resolve(x.device, 24576, 4096, 2) is None


@pytest.mark.parametrize("kind,sms", [("rtx5090", 170), ("pro6000", 188), ("gb10", 48)])
@pytest.mark.parametrize("n,k", dispatch.QWEN_SHAPES)
def test_register_grid_on_all_devices(kind, sms, n, k):
    configs = dispatch.tune_configs(dispatch.config_for(kind, sms, n, k), n, k, kind)
    assert len(configs) <= 10
    regs = [cfg for cfg in configs if cfg.variant == "reg"]
    assert len(regs) == 6
    for cfg in regs:
        tiles = (k + cfg.tile_k - 1) // cfg.tile_k
        chunks = (tiles + cfg.splits - 1) // cfg.splits
        assert (cfg.splits - 1) * chunks < tiles
        if n >= 4096 and k >= 2048:
            assert (n + cfg.rows - 1) // cfg.rows * cfg.splits * cfg.warps >= 4 * sms


def test_register_config_validation_and_csv():
    cfg = dispatch.Config(32, 128, 2, 4, 1, "reg", 8)
    assert dispatch.parse_config("32,128,2,4,1,reg,8") == cfg
    assert dispatch.parse_config("64,128,2,1") == dispatch.Config()
    for update in [
        {"variant": "bad"},
        {"prefetch": 1},
        {"prefetch": 9},
        {"warps": 9},
        {"rows": 48, "warps": 1},
    ]:
        with pytest.raises(ValueError):
            dispatch.Config(**{"variant": "reg", **update})


@pytest.mark.parametrize("variant", ["reg", "smem"])
@pytest.mark.parametrize("mtiles", range(1, 9))
def test_bucket_shared_bytes(variant, mtiles):
    cfg = dispatch.Config(64, 128, stages=3, variant=variant)
    expected = (8 * mtiles if variant == "reg" else 3 * (64 + 8 * mtiles)) * 136 * 2
    assert cfg.shared_bytes_for(mtiles) == expected
    assert cfg.shared_bytes == cfg.shared_bytes_for(1)
    for m in range((mtiles - 1) * 8 + 1, mtiles * 8 + 1):
        assert cute.mtiles_for(m) == mtiles
    # Validation remains based on the first bucket, not the largest bucket.
    large = dispatch.Config(64, 256, 2)
    assert large.shared_bytes_for(1) <= 96 * 1024 < large.shared_bytes_for(8)
    with pytest.raises(ValueError, match="96 KiB"):
        dispatch.Config(128, 256, 3)


@pytest.mark.parametrize("kind", ["rtx5090", "pro6000", "gb10"])
@pytest.mark.parametrize("n,k", [(1, 8), (67, 264), (4096, 4096), (5120, 17408), (151936, 4096)])
def test_bxq_candidate_expansion(kind, n, k):
    from dataclasses import replace

    base = dispatch.tune_configs(dispatch.Config(), n, k, kind)
    configs = dispatch.bxq_configs(n, k, kind)
    expected = set(base)
    for cfg in base:
        tiles = (k + cfg.tile_k - 1) // cfg.tile_k
        for multiplier in (2, 4):
            splits = cfg.splits * multiplier
            chunks = (tiles + splits - 1) // splits
            if splits <= 64 and (splits - 1) * chunks < tiles:
                expected.add(replace(cfg, splits=splits))
    assert len(configs) == len(set(configs))
    assert set(configs) == expected
    assert {c.variant for c in configs} == {"reg", "smem"}
    for cfg in set(configs) - set(base):
        chunks = (k + cfg.splits * cfg.tile_k - 1) // (cfg.splits * cfg.tile_k)
        assert (cfg.splits - 1) * chunks * cfg.tile_k < k


def fake_cuda_tensor(shape):
    return SimpleNamespace(
        shape=shape,
        ndim=2,
        is_cuda=True,
        get_device=lambda: 0,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16,
        requires_grad=False,
        is_contiguous=lambda: True,
        data_ptr=lambda: 16,
    )


@pytest.mark.parametrize("m", range(9, 66))
def test_bxq_inference_eligibility_and_bucket_forwarding(monkeypatch, m):
    x, w = fake_cuda_tensor((m, 128)), fake_cuda_tensor((67, 128))
    cfg = dispatch.Config()
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (67, 128), {(9, 64): cfg})
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    calls = []
    monkeypatch.setattr(cute, "get_compiled", lambda *args: calls.append(args) or None)
    sentinel = object()
    monkeypatch.setattr(inference, "linear", lambda *args: sentinel)
    op = Linear(provider="auto")
    if m <= 64:  # Eligible, but nothing was compiled before this capture.
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            op(x, w)
    else:
        assert op(x, w) is sentinel
    assert op.launch_config is None
    assert len(calls) == int(m <= 64)
    if calls:
        assert calls[0][3] == cfg and calls[0][-1] == (m + 7) // 8


def test_bucket_overflow_falls_back_and_forced_raises(monkeypatch):
    cfg = dispatch.Config(64, 256, 2)
    x, w = fake_cuda_tensor((64, 128)), fake_cuda_tensor((67, 128))
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (67, 128), {(1, 64): cfg})
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("overflow must not compile"))
    sentinel = object()
    monkeypatch.setattr(inference, "linear", lambda *args: sentinel)
    op = Linear(provider="auto")
    assert op(x, w) is sentinel and op.provider == "cublas"
    with pytest.raises(ValueError, match="shared memory"):
        Linear(provider="cute", config=cfg)(x, w)


def test_compilation_cache_separates_m_buckets(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    compiled = []
    monkeypatch.setattr(
        _provider, "compile_kernel", lambda *args: compiled.append(args) or object()
    )
    args = (torch.device("cuda:0"), 67, 264, dispatch.Config(), (12, 0))
    buckets = [_provider.get_compiled(*args, False, mt) for mt in range(1, 9)]
    assert len(compiled) == len({id(c) for c in buckets}) == 8
    for mt, result in enumerate(buckets, 1):
        assert _provider.get_compiled(*args, True, mt) is result


@pytest.mark.parametrize("m", range(0, 66))
def test_m_bucket_edges(m):
    expected = [(1, 1), (2, 2), (3, 4), (5, 8)] + [(lo, lo + 7) for lo in range(9, 58, 8)]
    assert dispatch.M_BUCKETS == tuple(expected)
    assert dispatch.m_bucket(m) == next((b for b in expected if b[0] <= m <= b[1]), None)


def test_tuned_ranges_cover_whole_buckets():
    starts, ends = map(set, zip(*dispatch.M_BUCKETS, strict=True))
    for kind, shapes in dispatch.TUNED.items():
        for (n, k), ranges in shapes.items():
            for lo, hi in ranges:
                assert lo in starts and hi in ends
            for lo, hi in dispatch.M_BUCKETS:
                expected = next((cfg for (a, b), cfg in ranges.items() if a <= hi <= b), None)
                assert all(
                    dispatch.tuned_config(kind, n, k, m) == expected for m in range(lo, hi + 1)
                )


@pytest.fixture
def host_dispatch(monkeypatch):
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_plans_reuse_bucket_without_host_resolution(monkeypatch, host_dispatch):
    calls = []
    config = dispatch.Config()
    monkeypatch.setattr(inference, "tuned_config", lambda *a: calls.append(("tuned", a)) or config)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: calls.append(("compiled", a)) or object())
    monkeypatch.setattr(Linear, "_launch_plan", lambda self, plan, *a: plan)
    op = Linear(provider="auto")
    w = fake_cuda_tensor((67, 128))
    first = op(fake_cuda_tensor((5, 128)), w)
    assert len(calls) == 2 and len(op._plans) == 1
    for lo, hi in ((5, 8), (9, 16)):
        if lo == 9:
            second = op(fake_cuda_tensor((lo, 128)), w)
            assert second is not first and len(calls) == 4
        with monkeypatch.context() as hot:

            def forbidden(*args, **kwargs):
                raise AssertionError("cached call repeated host resolution")

            hot.setattr(inference, "tuned_config", forbidden)
            hot.setattr(cute, "get_compiled", forbidden)
            hot.setattr(inference.os.environ, "get", forbidden)
            for m in range(lo, hi + 1):
                assert op(fake_cuda_tensor((m, 128)), w) is op._plans[0, 67, 128, hi]
    assert len(op._plans) == 2


def test_cublas_plan_is_cached(monkeypatch, host_dispatch):
    monkeypatch.setattr(inference, "tuned_config", lambda *a: None)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("fallback compiled"))
    op = Linear(provider="auto")
    x, w = fake_cuda_tensor((5, 128)), fake_cuda_tensor((67, 128))
    assert op._cute(x, w, None, None, 5, 67, 128) is None
    assert op._plans == {(0, 67, 128, 8): None}
    monkeypatch.setattr(
        inference, "tuned_config", lambda *a: pytest.fail("fallback resolved again")
    )
    assert op._cute(x, w, None, None, 8, 67, 128) is None


def test_capture_miss_raises_then_eager_compiles(monkeypatch, host_dispatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        cute, "get_compiled", lambda *a: calls.append(a) or (None if a[5] else object())
    )
    monkeypatch.setattr(Linear, "_launch_plan", lambda self, plan, *a: plan)
    op = Linear(provider="cute")
    x, w = fake_cuda_tensor((5, 128)), fake_cuda_tensor((67, 128))
    # Never a silent cuBLAS fallback inside a graph whose eager twin ran CuTe.
    with pytest.raises(RuntimeError, match="N=67, K=128, M bucket edge 8"):
        op._cute(x, w, None, None, 5, 67, 128)
    assert op._plans == {}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    plan = op._cute(x, w, None, None, 8, 67, 128)
    assert plan is op._plans[0, 67, 128, 8] and len(calls) == 2


def test_environment_is_frozen_at_construction(monkeypatch, host_dispatch):
    monkeypatch.setenv(inference.VARIABLE, "cute")
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "16,64,3,8")
    op = Linear()
    monkeypatch.setenv(inference.VARIABLE, "invalid")
    monkeypatch.setenv("CTI_LINEAR_CONFIG", "invalid")
    monkeypatch.setattr(cute, "get_compiled", lambda *a: object())
    assert op._resolve(torch.device("cuda:0"), 64, 4096, 8).splits == 8
    with pytest.raises(ValueError, match="CTI_LINEAR_PROVIDER"):
        Linear()
    with pytest.raises(ValueError, match="needs"):
        Linear(provider="cute")


@pytest.mark.parametrize("max_m", [1, 3, 5, 13, 57, 64, 65])
def test_workspace_bound_for_every_table_row(monkeypatch, max_m):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for kind, shapes in dispatch.TUNED.items():
        monkeypatch.setattr(dispatch, "_hardware", lambda _, kind=kind: (kind, (12, 0), 170))
        op = Linear(provider="auto")
        for (n, k), ranges in shapes.items():
            expected = max(
                (
                    cfg.scratch_splits * min(hi, max_m) * n * 4
                    for lo, hi in dispatch.M_BUCKETS
                    if lo <= max_m
                    for (a, b), cfg in ranges.items()
                    if a <= hi <= b and cfg.scratch_splits
                ),
                default=0,
            )
            assert op.workspace_bytes(n, k, max_m, device="cuda:0") == expected
            assert Linear(provider="cublas").workspace_bytes(n, k, max_m, device="cuda:0") == 0


@pytest.mark.parametrize("splits", [1, 3])
def test_forced_workspace_bound(monkeypatch, host_dispatch, splits):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    op = Linear(provider="cute", config=dispatch.Config(splits=splits))
    for m in (1, 5, 13, 64, 65):
        expected = splits * min(m, 64) * 67 * 4 if splits > 1 else 0
        assert op.workspace_bytes(67, 128, m, device="cuda:0") == expected
    with pytest.raises(ValueError, match="positive"):
        op.workspace_bytes(67, 128, 0)
    for n, k in ((0, 128), (67, 0), (67, 7)):
        assert op.workspace_bytes(n, k, device="cuda:0") == 0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert op.workspace_bytes(67, 128) == 0


def test_scratch_grows_and_shares_per_stream(monkeypatch):
    monkeypatch.setattr(inference, "_SCRATCH", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = [11]
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda _: stream[0], raising=False)
    x, plan = torch.empty(1), SimpleNamespace(splits=3)
    op, other = Linear(provider="auto"), Linear(provider="auto")
    first = op._scratch(plan, x, None, 5, 16)
    assert first.shape == (3, 5, 16) and first.dtype == torch.float32
    assert op._scratch(plan, x, None, 5, 16) is first
    assert other._scratch(plan, x, None, 3, 16).data_ptr() == first.data_ptr()
    assert inference.scratch_bytes() == 3 * 5 * 16 * 4
    grown = op._scratch(plan, x, None, 8, 16)
    assert grown.data_ptr() != first.data_ptr()
    assert other._scratch(plan, x, None, 5, 16).data_ptr() == grown.data_ptr()
    assert inference.scratch_bytes() == 3 * 8 * 16 * 4
    stream[0] = 12
    assert op._scratch(plan, x, None, 1, 16).data_ptr() != grown.data_ptr()
    assert inference.scratch_bytes() == 3 * 9 * 16 * 4
    before = inference.scratch_bytes()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    captured = op._scratch(plan, x, None, 64, 16)
    assert captured.shape == (3, 64, 16) and inference.scratch_bytes() == before
    assert captured.data_ptr() not in {v.data_ptr() for v in inference._SCRATCH.values()}


def test_release_cublas_workspaces(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append(1), raising=False
    )
    inference.release_cublas_workspaces()
    assert calls == [1]
    monkeypatch.delattr(torch._C, "_cuda_clearCublasWorkspaces")
    with pytest.raises(RuntimeError, match="cuBLAS workspaces"):
        inference.release_cublas_workspaces()


def test_release_scratch_per_device(monkeypatch):
    monkeypatch.setattr(inference, "_SCRATCH", {})
    monkeypatch.setattr(inference, "_VIEWS", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda _: 7, raising=False)
    plan, size = SimpleNamespace(splits=2), 2 * 4 * 16 * 4
    for index in (0, 1):
        x = SimpleNamespace(get_device=lambda i=index: i, device=torch.device("cpu"))
        Linear(provider="auto")._scratch(plan, x, None, 4, 16)
    assert inference.scratch_bytes("cuda:1") == inference.scratch_bytes(0) == size
    inference.release_scratch(1)
    assert inference.scratch_bytes(1) == 0 and inference.scratch_bytes() == size
    inference.release_scratch()
    assert inference.scratch_bytes() == 0 and inference._VIEWS == {}
    with pytest.raises(ValueError, match="CUDA"):
        inference.release_scratch("cpu")


def test_release_scratch_races_allocation(monkeypatch):
    monkeypatch.setattr(inference, "_SCRATCH", {})
    monkeypatch.setattr(inference, "_VIEWS", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda _: 7, raising=False)
    x, errors = SimpleNamespace(get_device=lambda: 0, device=torch.device("cpu")), []
    start = threading.Barrier(4)

    def run(work):
        start.wait()
        try:
            for i in range(10000):
                work(i)
        except (KeyError, RuntimeError) as error:  # Lost-race symptoms surface here.
            errors.append(error)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # Preempt often so a lost race is likely.
    op = Linear(provider="auto")
    workers = [
        threading.Thread(
            target=run,
            args=(lambda i: op._scratch(SimpleNamespace(splits=2), x, None, 1 + i % 8, 16),),
        ),
        threading.Thread(target=run, args=(lambda i: inference.release_scratch(),)),
        threading.Thread(target=run, args=(lambda i: inference.release_scratch(0),)),
        threading.Thread(target=run, args=(lambda i: inference.scratch_bytes(0),)),
    ]
    for worker in workers:
        worker.start()
    try:
        for worker in workers:
            worker.join()
    finally:
        sys.setswitchinterval(previous)
    assert errors == []


def test_workspace_size_alignment_layout_and_views(monkeypatch):
    op, x, plan = Linear(provider="auto"), torch.empty(1), SimpleNamespace(splits=3)
    workspace = torch.empty(3 * 8 * 16)
    with pytest.raises(ValueError, match="workspace"):
        op._scratch(plan, x, workspace, 8, 16)
    # Exercise real CPU storage/view operations, faking only CUDA admission.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("external workspace inspected capture"),
    )
    for dtype in (torch.float32, torch.uint8, torch.bfloat16):
        buffer = torch.empty(
            3 * 8 * 16 * 4 // torch.empty((), dtype=dtype).element_size(), dtype=dtype
        )
        view = op._scratch(plan, x, buffer, 8, 16)
        assert view.shape == (3, 8, 16) and view.dtype == torch.float32
        assert view.data_ptr() == buffer.data_ptr()
        view.fill_(2)
        assert torch.equal(buffer.view(torch.float32), torch.full((384,), 2.0))
    for bad in (workspace[:-1], torch.empty(384 * 2)[1:385], torch.empty(16, 24).t()):
        with pytest.raises(ValueError, match="workspace"):
            op._scratch(plan, x, bad, 8, 16)
    with pytest.raises(ValueError, match="workspace"):
        op._scratch(plan, SimpleNamespace(device=torch.device("meta")), workspace, 8, 16)


def test_workspace_overlapping_inputs_or_out_is_rejected_before_launch(monkeypatch):
    op = Linear(provider="auto")
    monkeypatch.setattr(op, "_cute", lambda *a: pytest.fail("launched with overlapping workspace"))
    monkeypatch.setattr(inference, "linear", lambda *a: pytest.fail("launched cuBLAS"))
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    weight = torch.zeros(32, 64, dtype=torch.bfloat16)
    out = torch.empty(2, 32, dtype=torch.bfloat16)
    for workspace in (weight, weight.view(-1)[8:], x, out.view(torch.uint8)):
        with pytest.raises(ValueError, match="workspace must not share storage"):
            op(x, weight, out, workspace=workspace)
    # Overlap is rejected even when out is allocated by the op.
    with pytest.raises(ValueError, match="workspace must not share storage"):
        op(x, weight, workspace=weight)


def test_oversized_byte_workspace_and_growth_releases_other_ops_views(monkeypatch):
    op, x, plan = Linear(provider="auto"), torch.empty(1), SimpleNamespace(splits=3)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    # Any aligned buffer with at least the required bytes is valid, even if its
    # byte length is not a multiple of four.
    buffer = torch.empty(3 * 8 * 16 * 4 + 1, dtype=torch.uint8)
    view = op._scratch(plan, x, buffer, 8, 16)
    assert view.shape == (3, 8, 16) and view.data_ptr() == buffer.data_ptr()
    # Shared scratch: when one op grows the buffer, no op keeps the old one alive.
    monkeypatch.setattr(inference, "_SCRATCH", {})
    monkeypatch.setattr(inference, "_VIEWS", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda _: 7, raising=False)
    other = Linear(provider="auto")
    small = op._scratch(plan, x, None, 2, 16)
    old = small.untyped_storage().data_ptr()
    other._scratch(plan, x, None, 8, 16)
    assert inference.scratch_bytes() == 3 * 8 * 16 * 4
    del small
    live = {
        v.untyped_storage().data_ptr()
        for _, views in inference._VIEWS.values()
        for v in views.values()
    }
    assert old not in live
    assert (
        op._scratch(plan, x, None, 2, 16).data_ptr()
        == next(iter(inference._SCRATCH.values())).data_ptr()
    )


def test_workspace_bound_resolves_the_current_device(monkeypatch):
    seen = []

    def hardware(device):
        seen.append(device)
        return ("rtx5090", (12, 0), 170) if device.index == 0 else ("pro6000", (12, 0), 188)

    monkeypatch.setattr(dispatch, "_hardware", hardware)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(
        dispatch.TUNED["pro6000"], (4096, 4096), {(5, 8): dispatch.Config(splits=4)}
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    op = Linear(provider="auto")
    assert op.workspace_bytes(4096, 4096, 8, device="cuda") == 4 * 8 * 4096 * 4
    assert op.workspace_bytes(4096, 4096, 8) == 4 * 8 * 4096 * 4
    assert all(d == torch.device("cuda", 1) for d in seen)


def test_resolve_reads_capture_state_on_the_target_device(monkeypatch):
    current = [0]
    capturing = {0: False, 1: True}

    class Guard:
        def __init__(self, device):
            self.device = torch.device(device)

        def __enter__(self):
            self.previous, current[0] = current[0], self.device.index

        def __exit__(self, *exc):
            current[0] = self.previous

    monkeypatch.setattr(torch.cuda, "device", Guard)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[current[0]])
    monkeypatch.setattr(dispatch, "_hardware", lambda _: ("rtx5090", (12, 0), 170))
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (4096, 4096), {(1, 1): dispatch.Config()})
    calls = []
    monkeypatch.setattr(cute, "get_compiled", lambda *args: calls.append(args[5]) or None)
    assert Linear(provider="auto")._resolve(torch.device("cuda", 1), 4096, 4096, 1) is False
    assert calls == [True]  # capture of cuda:1, not of the current cuda:0


@pytest.mark.parametrize("provider", ["auto", "cute", "cublas"])
def test_gate_cpu_fallback(provider):
    x, w = torch.randn(3, 16).bfloat16(), torch.randn(17, 16).bfloat16()
    gate = (torch.randn(3, 24) * 3).bfloat16()[:, :16]
    gate[:, :2] = torch.tensor([-100, 100], dtype=torch.bfloat16)
    a = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    op = Linear(provider=provider)
    expected = torch.nn.functional.linear(a, w)
    out = torch.empty_like(expected)
    assert op(x, w, out, gate=gate) is out
    assert torch.equal(out, expected)
    assert op.provider == "unfused+cublas" and op.launch_config is None
    assert op.kernels_per_call == 5
    op(x, w)
    assert op.provider == "cublas" and op.kernels_per_call == 1


def test_gate_validation_before_dispatch(monkeypatch):
    monkeypatch.setattr(dispatch, "_hardware", lambda *a: pytest.fail("inspected CUDA"))
    monkeypatch.setattr(inference, "linear", lambda *a: pytest.fail("dispatched"))
    op = Linear(provider="auto")
    x, w = torch.zeros(2, 16).bfloat16(), torch.zeros(8, 16).bfloat16()
    for gate, error in [
        (torch.empty(2, 16, 1, dtype=x.dtype), "shape"),
        (torch.empty(3, 16, dtype=x.dtype), "shape"),
        (torch.empty(2, 16), "dtype"),
        (torch.empty(2, 16, dtype=x.dtype, device="meta"), "device"),
        (torch.empty(2, 32, dtype=x.dtype)[:, ::2], "stride"),
        (torch.empty_like(x, requires_grad=True), "gradients"),
    ]:
        with pytest.raises(ValueError, match=error):
            op(x, w, gate=gate)
    gate = torch.empty(2, 24, dtype=x.dtype)[:, :16]
    with pytest.raises(ValueError, match="out must not share gate"):
        op(x, w, gate=gate, out=gate._base.view(-1)[:16].view(2, 8))
    with pytest.raises(ValueError, match="workspace must not share gate"):
        op(x, w, gate=gate, workspace=gate._base.view(-1)[32:])


def test_gate_activation_autograd():
    x = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(8, 16, dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    expected_x = x.detach().clone().requires_grad_()
    a = (expected_x.float() * torch.sigmoid(gate.float())).bfloat16()
    torch.nn.functional.linear(a, w).sum().backward()
    op = Linear(provider="auto")
    op(x, w, gate=gate).sum().backward()
    assert torch.equal(x.grad, expected_x.grad)
    assert op.provider == "unfused+cublas"


def test_gated_table_schema():
    assert dispatch.GATED_SHAPES == ((4096, 4096, 10240), (5120, 6144, 14336), (5120, 3072, 7168))
    assert set(dispatch.TUNED_GATED) == {"rtx5090", "pro6000", "gb10"}
    starts, ends = map(set, zip(*dispatch.M_BUCKETS, strict=True))
    for kind, shapes in dispatch.TUNED_GATED.items():
        for (n, k, stride), ranges in shapes.items():
            assert n > 0 and k > 0 and k % 8 == 0 and stride >= k and stride % 8 == 0
            seen = set()
            for (lo, hi), config in ranges.items():
                assert lo in starts and hi in ends and lo <= hi
                assert not seen.intersection(range(lo, hi + 1))
                seen.update(range(lo, hi + 1))
                assert isinstance(config, dispatch.Config)
                assert config.shared_bytes_for(cute.mtiles_for(hi), True) <= inference._SMEM_LIMIT
            for lo, hi in dispatch.M_BUCKETS:
                expected = next((cfg for (a, b), cfg in ranges.items() if a <= hi <= b), None)
                assert all(
                    dispatch.tuned_gated_config(kind, n, k, stride, m) == expected
                    for m in range(lo, hi + 1)
                )


def test_gated_bucket_lookup(monkeypatch):
    cfg = dispatch.Config(splits=3)
    monkeypatch.setitem(dispatch.TUNED_GATED["rtx5090"], (96, 256, 264), {(4, 4): cfg})
    for m in (3, 4):
        assert dispatch.tuned_gated_config("rtx5090", 96, 256, 264, m) is cfg
    for m in (0, 1, 2, 5, 65):
        assert dispatch.tuned_gated_config("rtx5090", 96, 256, 264, m) is None
    assert dispatch.tuned_gated_config("rtx5090", 96, 256, 272, 4) is None
    assert dispatch.tuned_gated_config(None, 96, 256, 264, 4) is None


@pytest.mark.parametrize("variant", ["smem", "reg"])
@pytest.mark.parametrize("mtiles", [1, 8])
def test_gated_shared_bytes(variant, mtiles):
    cfg = dispatch.Config(32, 128, 3, variant=variant)
    extra = 8 * mtiles * 136 * 2 * (3 if variant == "smem" else 1)
    assert cfg.shared_bytes_for(mtiles, True) == cfg.shared_bytes_for(mtiles) + extra


def test_gate_query_and_workspace(monkeypatch, host_dispatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("query compiled"))
    monkeypatch.setattr(torch.cuda, "device", lambda *a: pytest.fail("query entered CUDA"))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: pytest.fail("query inspected stream")
    )
    cfg = dispatch.Config(rows=32, splits=3)
    monkeypatch.setitem(dispatch.TUNED_GATED["rtx5090"], (96, 256, 264), {(3, 8): cfg})
    op = Linear(provider="auto")
    assert op.gate_fused(96, 256, 264, 3, device="cuda:0")
    assert op.gate_fused(96, 256, 264, 8)
    assert not op.gate_fused(96, 256, 264, 2)
    assert not op.gate_fused(96, 256, 272, 3)
    assert not op.gate_fused(96, 256, 264, 3, device="cpu")
    assert not Linear(provider="cublas").gate_fused(96, 256, 264, 3)
    assert op.workspace_bytes(96, 256, 5, gate_stride=264) == 3 * 5 * 96 * 4
    assert op.workspace_bytes(96, 256, 64, gate_stride=264) == 3 * 8 * 96 * 4
    assert op.workspace_bytes(96, 256, gate_stride=272) == 0
    assert op.workspace_bytes(96, 256, gate_stride=255) == 0
    assert op.workspace_bytes(96, 256) == 0
    forced = Linear(provider="cute", config=cfg)
    assert forced.gate_fused(96, 256, 272, 64)
    assert forced.workspace_bytes(96, 256, gate_stride=272) == 3 * 64 * 96 * 4
    for n, k, stride, m in [
        (0, 256, 264, 3),
        (96, 0, 264, 3),
        (96, 255, 264, 3),
        (96, 256, 248, 3),
        (96, 256, 265, 3),
        (96, 256, 264, 65),
    ]:
        assert not forced.gate_fused(n, k, stride, m)
    large = dispatch.Config(64, 128, 2)
    assert large.shared_bytes_for(8) < inference._SMEM_LIMIT < large.shared_bytes_for(8, True)
    assert not Linear(provider="cute", config=large).gate_fused(96, 256, 264, 64)
    monkeypatch.setattr(cute, "available", lambda: False)
    assert not forced.gate_fused(96, 256, 264, 3)
    monkeypatch.setattr(cute, "available", lambda: True)
    monkeypatch.setattr(dispatch, "_hardware", lambda _: (None, (9, 0), 132))
    assert not forced.gate_fused(96, 256, 264, 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not forced.gate_fused(96, 256, 264, 3)


def test_wide_config_never_takes_the_fused_gate(monkeypatch, host_dispatch):
    # The wide variant has no gated prologue: a gated call runs the unfused route.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("query compiled"))
    wide = dispatch.Config(64, 256, 2, 16, 4, "wide", 4)
    forced = Linear(provider="cute", config=wide)
    assert not forced.gate_fused(512, 4096, 4096, 2)
    assert forced.decision(512, 4096, 2, gate_stride=4096)["provider"] == "cublas"
    assert forced.decision(512, 4096, 2)["provider"] == "cute"
    # The bound still covers the plain (unfused) plan that the call falls back to.
    assert forced.workspace_bytes(512, 4096, 2, gate_stride=4096) == 16 * 2 * 512 * 4
    assert forced._resolve(torch.device("cuda", 0), 512, 4096, 2, 4096, None) is None


class _Accepted(Exception):
    """The workspace passed validation and _scratch went on to view it."""


@pytest.mark.parametrize(
    "hardware",
    [(kind, capability, sms) for (capability, sms), kind in dispatch._DEVICE_CLASSES.items()],
)
def test_advertised_workspace_covers_every_reachable_plan(monkeypatch, host_dispatch, hardware):
    # A gated call may launch its fused plan or, unfused, the plain plan of the same bucket.
    kind = hardware[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dispatch, "_hardware", lambda _: hardware)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("query compiled"))
    gated = dispatch.TUNED_GATED.get(kind, {})
    keys = {(n, k, 0) for n, k in dispatch.TUNED.get(kind, {})}
    keys |= {(n, k, k) for n, k, _ in keys}  # Gate stride with no gated row: unfused only.
    keys |= set(gated) | {(n, k, 0) for n, k, _ in gated}
    x = fake_cuda_tensor((1, 1))
    op = Linear(provider="auto")

    def accepted(*args):
        raise _Accepted

    strides = set()
    for n, k, stride in keys:
        # Aligned, unaligned and too-short gate strides (M=1 rows admit any stride).
        strides |= {(n, k, s) for s in {stride, k + 1, k + 4, k + 8, k - 8, 4097} if s >= 0}
    for n, k, stride in sorted(strides):
        fusable = stride >= k and stride % 8 == 0
        for lo, hi in dispatch.M_BUCKETS:
            # Independent of the route function: plain always, fused only when admitted.
            reachable = [dispatch.tuned_config(kind, n, k, hi)]
            if stride and fusable:
                reachable.append(dispatch.tuned_gated_config(kind, n, k, stride, hi))
            for m in {lo, hi}:
                advertised = op.workspace_bytes(n, k, m, device="cuda:0", gate_stride=stride)
                workspace = SimpleNamespace(
                    is_cuda=True,
                    device=x.device,
                    is_contiguous=lambda: True,
                    data_ptr=lambda: 16,
                    numel=lambda advertised=advertised: advertised,
                    element_size=lambda: 1,
                    view=accepted,
                )
                for config in reachable:
                    if config is not None and config.scratch_splits:
                        # _scratch raises ValueError on a short workspace, before any view.
                        assert 4 * config.splits * m * n <= advertised, (kind, n, k, stride, m)
                        with pytest.raises(_Accepted):
                            op._scratch(SimpleNamespace(splits=config.splits), x, workspace, m, n)
    # Reported reproductions: GB10, N=K=4096, M=1, gate stride 4096 and 4097.
    plain = dispatch.tuned_config(kind, 4096, 4096, 1)
    if plain is not None and plain.scratch_splits:
        for stride in (4096, 4097):
            bound = op.workspace_bytes(4096, 4096, 1, gate_stride=stride)
            assert bound >= 4 * plain.scratch_splits * 4096


def test_gated_workspace_includes_unfused_fallback(monkeypatch, host_dispatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(dispatch.TUNED_GATED, "rtx5090", {})
    monkeypatch.setitem(
        dispatch.TUNED, "rtx5090", {(96, 256): {(1, 1): dispatch.Config(rows=32, splits=4)}}
    )
    op = Linear(provider="auto")
    assert not op.gate_fused(96, 256, 256, 1)
    assert op.workspace_bytes(96, 256, 1, gate_stride=256) == 4 * 1 * 96 * 4
    fused = dispatch.Config(rows=32, splits=2)
    monkeypatch.setitem(dispatch.TUNED_GATED, "rtx5090", {(96, 256, 256): {(1, 1): fused}})
    assert op.gate_fused(96, 256, 256, 1)
    assert op.workspace_bytes(96, 256, 1, gate_stride=256) == 4 * 1 * 96 * 4
    # Unaligned or too-short strides never fuse but still run the plain plan unfused.
    for stride in (257, 260, 248, 0):
        assert not op.gate_fused(96, 256, stride, 1)
        assert op.workspace_bytes(96, 256, 1, gate_stride=stride) == 4 * 1 * 96 * 4


def test_gated_compile_cache_separates_strides(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    calls = []
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a: calls.append(a) or object())
    args = (torch.device("cuda:0"), 96, 256, dispatch.Config(), (12, 0))
    results = [_provider.get_compiled(*args, False, 1, stride) for stride in (0, 264, 272)]
    assert len({id(result) for result in results}) == 3
    for stride, result in zip((0, 264, 272), results, strict=True):
        assert _provider.get_compiled(*args, True, 1, stride) is result
    assert [call[-1] for call in calls] == [0, 264, 272]
    assert _provider.get_compiled(*args, True, 8, 264) is None


def test_gated_plan_cache_and_reporting(monkeypatch, host_dispatch):
    calls, launches = [], []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "get_device", lambda self: 0)
    monkeypatch.setattr(inference, "_dummy", lambda _: None)
    monkeypatch.setattr(
        cute, "get_compiled", lambda *a: calls.append(a) or (lambda *args: launches.append(args))
    )
    op = Linear(provider="cute")
    x, w = torch.empty(8, 256).bfloat16(), torch.empty(96, 256).bfloat16()
    gate = torch.empty(8, 264).bfloat16()[:, :256]
    for m in (5, 8):
        op(x[:m], w, gate=gate[:m])
        assert op.provider == "cute-gated" and op.kernels_per_call == 1
        assert op.launch_config["gate_stride"] == 264
    assert len(calls) == 1 and calls[0][-2:] == (1, 264)
    assert len(launches[-1]) == 5 and launches[-1][-1].stride() == (264, 1)
    op(x, w)
    assert len(calls) == 2 and len(launches[-1]) == 4
    assert op.provider == "cute" and "gate_stride" not in op.launch_config
    gate = torch.empty(8, 272).bfloat16()[:, :256]
    op(x, w, gate=gate)
    assert len(calls) == 3 and len(op._plans) == 3


@pytest.mark.parametrize("bad_layout", ["stride", "alignment", "broadcast"])
def test_gated_layout_falls_back(monkeypatch, host_dispatch, bad_layout):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    x, w = torch.randn(3, 16).bfloat16(), torch.randn(17, 16).bfloat16()
    if bad_layout == "stride":
        gate = torch.randn(3, 17).bfloat16()[:, :16]
    elif bad_layout == "alignment":
        gate = torch.randn(3, 24).bfloat16()[:, 1:17]
    else:
        gate = torch.randn(1, 16).bfloat16().expand(3, 16)
    op = Linear(provider="cute")
    monkeypatch.setattr(op, "_resolve", lambda *a: pytest.fail("invalid gate compiled"))
    monkeypatch.setattr(op, "_cute", lambda *a: None)
    a = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    assert torch.equal(op(x, w, gate=gate), torch.nn.functional.linear(a, w))
    assert op.provider == "unfused+cublas"


def test_gated_auto_selection_and_capture_retry(monkeypatch, host_dispatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "get_device", lambda self: 0)
    cfg = dispatch.Config(32, 64, 3, 3)
    monkeypatch.setitem(dispatch.TUNED_GATED["rtx5090"], (96, 256, 264), {(5, 8): cfg})
    calls = []
    capturing = [True]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    monkeypatch.setattr(
        cute, "get_compiled", lambda *a: calls.append(a) or (None if a[5] else object())
    )
    op = Linear(provider="auto", config=dispatch.Config())
    # Plain fallback needs no CUDA execution in this host-only test.
    monkeypatch.setattr(op, "_cute", lambda *a: None)
    monkeypatch.setattr(op, "_launch_gated", lambda plan, *a: plan)
    x, w = torch.randn(8, 256).bfloat16(), torch.randn(96, 256).bfloat16()
    gate = torch.randn(8, 264).bfloat16()[:, :256]
    with pytest.raises(RuntimeError, match="gate stride 264"):
        op(x[:5], w, gate=gate[:5])
    assert op._plans == {}
    capturing[0] = False
    plan = op(x[:5], w, gate=gate[:5])
    assert plan.splits == 3 and plan.launch["config"]["stages"] == 3
    assert plan is op(x, w, gate=gate)
    assert len(calls) == 2 and all(call[3] is cfg for call in calls)
    assert set(op._plans) == {(0, 96, 256, 8, 264)}
    # A missing gated row never borrows the plain table's config.
    assert op._resolve(x.device, 96, 256, 8, 272) is None
    assert len(calls) == 2


@pytest.mark.parametrize("provider", ["auto", "cute"])
def test_gated_overflow_uses_unfused_fallback(monkeypatch, host_dispatch, provider):
    # Auto falls back to the unfused path; forced CuTe raises like the plain kernel.
    cfg = dispatch.Config(64, 128, 2)
    monkeypatch.setitem(dispatch.TUNED_GATED["rtx5090"], (96, 256, 264), {(57, 64): cfg})
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "get_device", lambda self: 0)
    monkeypatch.setattr(cute, "get_compiled", lambda *a: pytest.fail("overflow compiled"))
    op = Linear(provider=provider, config=cfg)
    monkeypatch.setattr(op, "_cute", lambda *a: None)
    x, w = torch.randn(64, 256).bfloat16(), torch.randn(96, 256).bfloat16()
    gate = torch.randn(64, 264).bfloat16()[:, :256]
    if provider == "cute":
        with pytest.raises(ValueError, match="shared memory"):
            op(x, w, gate=gate)
        return
    a = (x.float() * torch.sigmoid(gate.float())).bfloat16()
    assert torch.equal(op(x, w, gate=gate), torch.nn.functional.linear(a, w))
    assert op.provider == "unfused+cublas"
    assert op._plans == {(0, 96, 256, 64, 264): None}


@pytest.mark.parametrize("explicit", [False, True])
def test_fp32_market_call_and_out(monkeypatch, explicit):
    monkeypatch.setattr(cublas, "_out_dtype_mm", lambda x: True)
    x = torch.tensor([[1.0, 0.1]], dtype=torch.bfloat16)
    w = torch.ones(3, 2, dtype=torch.bfloat16)
    expected = x.float() @ w.float().t()
    calls = []
    mm = torch.mm

    def market(a, b, *, out_dtype, out=None):
        assert a.dtype == b.dtype == torch.bfloat16
        assert out_dtype == torch.float32
        calls.append(out)
        return mm(a.float(), b.float(), out=out)

    monkeypatch.setattr(torch, "mm", market)
    op = Linear()
    out = torch.empty(1, 3)
    assert op(x, w, out, out_dtype=torch.float32 if explicit else None) is out
    assert torch.equal(out, expected)
    assert not torch.equal(out, expected.bfloat16().float())
    assert torch.equal(op(x, w, out_dtype=torch.float32), expected)
    assert calls == [out, None]
    assert op.provider == "cublas" and op.launch_config is None


def test_fp32_old_torch_fallback_and_unrelated_errors(monkeypatch):
    monkeypatch.setattr(cublas, "_out_dtype_mm", lambda x: True)
    from ctikernels.linear.cublas import linear_fp32

    mm = torch.mm
    x, w = torch.randn(2, 8).bfloat16(), torch.randn(3, 8).bfloat16()

    def old(a, b, **kw):
        if "out_dtype" in kw:
            raise TypeError("unexpected keyword argument 'out_dtype'")
        return mm(a, b, **kw)

    monkeypatch.setattr(torch, "mm", old)
    expected = mm(x.float(), w.float().t())
    out = torch.empty_like(expected)
    assert linear_fp32(x, w, out) is out
    assert torch.equal(out, expected)

    def broken(*a, **kw):
        raise TypeError("unrelated failure")

    monkeypatch.setattr(torch, "mm", broken)
    with pytest.raises(TypeError, match="unrelated"):
        linear_fp32(x, w)


@pytest.mark.parametrize("supplied", [False, True])
def test_fp32_cpu_fallback_unmocked(supplied):
    from ctikernels.linear.cublas import linear_fp32

    x, w = torch.randn(3, 16).bfloat16(), torch.randn(5, 16).bfloat16()
    expected = torch.mm(x.float(), w.float().t())
    out = torch.empty(3, 5) if supplied else None
    result = linear_fp32(x, w, out)
    assert result.dtype == torch.float32 and torch.equal(result, expected)
    if supplied:
        assert result is out
    out = torch.empty(3, 5) if supplied else None
    result = Linear()(x, w, out, out_dtype=torch.float32)
    assert result.dtype == torch.float32 and torch.equal(result, expected)
    if supplied:
        assert result is out


def test_fp32_validation(monkeypatch):
    monkeypatch.setattr(dispatch, "_hardware", lambda *a: pytest.fail("inspected device"))
    x, w = torch.ones(2, 8).bfloat16(), torch.ones(3, 8).bfloat16()
    op = Linear()
    for kwargs in (
        {"out_dtype": torch.float64},
        {"out_dtype": torch.float32, "gate": x},
        {"out": torch.empty(2, 3), "gate": x},
        {"out_dtype": torch.float32, "out": torch.empty(2, 3).bfloat16()},
        {"out_dtype": torch.bfloat16, "out": torch.empty(2, 3)},
        {"out_dtype": torch.float32, "out": torch.empty(2, 4)},
    ):
        with pytest.raises(ValueError):
            op(x, w, **kwargs)
    with pytest.raises(ValueError, match="BF16"):
        op(x.float(), w.float(), out_dtype=torch.float32)
    with pytest.raises(ValueError, match="frozen"):
        op(x, w.requires_grad_(), out_dtype=torch.float32)
    for query in (op.decision, op.workspace_bytes):
        with pytest.raises(ValueError, match="out_dtype"):
            query(3, 8, 2, out_dtype=torch.float64)
        with pytest.raises(ValueError, match="gate"):
            query(3, 8, 2, gate_stride=8, out_dtype=torch.float32)
    with pytest.raises(ValueError, match="gate"):
        op.gate_fused(3, 8, 8, 2, out_dtype=torch.float32)


def test_fp32_table_queries_and_plans_skip_autotune(monkeypatch, host_dispatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_autotune, "ACTIVE", [object()])
    for name in ("lookup", "identity", "tune"):
        monkeypatch.setattr(_autotune, name, lambda *a: pytest.fail("FP32 consulted autotune"))
    cfg = dispatch.Config(splits=4)
    monkeypatch.setitem(dispatch.TUNED, "rtx5090", {(256, 4096): {(1, 64): cfg}})
    monkeypatch.setitem(dispatch.TUNED_F32, "rtx5090", {})
    device = torch.device("cuda:0")
    op = Linear(provider="auto")
    assert op.decision(256, 4096, 3, out_dtype=torch.float32)["provider"] == "cublas"
    assert op.workspace_bytes(256, 4096, 64, out_dtype=torch.float32) == 0
    assert op._resolve(device, 256, 4096, 4, out_dtype=torch.float32) is None
    monkeypatch.setitem(dispatch.TUNED_F32, "rtx5090", {(256, 4096): {(1, 64): cfg}})
    # Retained fallback still describes what this operation launches.
    assert op.decision(256, 4096, 3, out_dtype=torch.float32)["provider"] == "cublas"
    op = Linear(provider="auto")
    calls = []
    monkeypatch.setattr(cute, "get_compiled", lambda *a, **kw: calls.append((a, kw)) or object())
    monkeypatch.setattr(Linear, "_launch_plan", lambda self, plan, *a: plan)
    x, w = fake_cuda_tensor((3, 4096)), fake_cuda_tensor((256, 4096))
    plan = op(x, w, out_dtype=torch.float32)
    assert plan.splits == 4 and calls[0][1] == {"fp32": True}
    assert op.workspace_bytes(256, 4096, 64, out_dtype=torch.float32) == 4 * 64 * 256 * 4
    assert op.decision(256, 4096, 3, out_dtype=torch.float32)["config"]["splits"] == 4
    assert op(x, w, out_dtype=torch.float32) is plan and len(calls) == 1
    bf16_plan = op(x, w)
    assert bf16_plan is not plan and calls[-1][1] == {}
    forced = Linear(provider="cute", config=cfg)
    assert forced.workspace_bytes(256, 4096, 16, out_dtype=torch.float32) == 4 * 16 * 256 * 4
    assert forced(x, w, out_dtype=torch.float32).splits == 4


def test_fp32_compile_cache_and_capture_miss(monkeypatch):
    monkeypatch.setattr(_provider, "_COMPILED", {})
    calls = []
    monkeypatch.setattr(_provider, "compile_kernel", lambda *a, **kw: calls.append(kw) or object())
    args = (torch.device("cuda:0"), 256, 4096, dispatch.Config(), (12, 0))
    bf16 = _provider.get_compiled(*args, False)
    assert _provider.get_compiled(*args, True, fp32=True) is None
    fp32 = _provider.get_compiled(*args, False, fp32=True)
    assert fp32 is not bf16
    assert _provider.get_compiled(*args, True, fp32=True) is fp32
    assert calls == [{}, {"fp32": True}]


def test_inference_import_does_not_load_the_cute_provider():
    import sys

    code = (
        "import sys, ctikernels.linear.inference;"
        "loaded = [m for m in sys.modules if m.startswith('ctikernels.linear.cute')];"
        "assert not loaded, loaded"
    )
    run_child([sys.executable, "-c", code], timeout=600, check=True)


def test_fp32_table_ships_only_gpu_floor_tested_splits():
    # tests/gpu/test_linear.py gates these splits at the 4 eps32 floor; split 1
    # (one 4096-long FP32 chain) exceeds it, so no FP32 row may use it.
    for device, shapes in dispatch.TUNED_F32.items():
        for shape, rows in shapes.items():
            for bucket, cfg in rows.items():
                assert cfg.splits in (4, 8, 16, 32), (device, shape, bucket, cfg)


def test_warm_up_compiles_each_table_route_once_without_tuning(monkeypatch, host_dispatch):
    # KW4: compile-only warm-up from the shipped table; holes compile nothing.
    compiled = []
    monkeypatch.setattr(cute, "get_compiled", lambda *a, **kw: compiled.append(a[1:4]) or object())
    monkeypatch.setattr(inference._autotune, "tune", lambda *a: pytest.fail("warm_up tuned"))
    rows = {(1, 1): dispatch.Config(splits=4, cluster=1), (5, 16): dispatch.Config()}
    monkeypatch.setitem(dispatch.TUNED["rtx5090"], (64, 128), rows)
    primed = []
    monkeypatch.setattr(inference, "prime", lambda kernel, run, m, stride: primed.append(m))
    op = Linear(provider="auto")
    w = fake_cuda_tensor((64, 128))
    routes = op.warm_up(w, 24)
    assert [r["bucket"] for r in routes] == [(1, 1), (5, 8), (9, 16)]
    # Every bucket runs once, cuBLAS holes included (CUDA loads kernels lazily).
    assert primed == [lo for lo, _ in dispatch.M_BUCKETS if lo <= 24]
    assert routes[0]["config"]["cluster"] == 1
    assert [c[2] for c in compiled] == [
        dispatch.Config(splits=4, cluster=1),
        *[dispatch.Config()] * 2,
    ]
    assert op.warm_up(w, 24) == routes and len(compiled) == 3  # Plans are cached.
    # A capture after warm-up finds every plan: no compile, no raise.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(Linear, "_launch_plan", lambda self, plan, *a: plan)
    for m in (1, 5, 8, 16):
        assert op(fake_cuda_tensor((m, 128)), w) is not None
    assert len(compiled) == 3
    with pytest.raises(RuntimeError, match="eagerly"):
        op.warm_up(w, 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    primed.clear()
    assert Linear(provider="cublas").warm_up(w, 64) == []  # cuBLAS only: primed, nothing compiled
    assert primed == [lo for lo, _ in dispatch.M_BUCKETS]
    with pytest.raises(ValueError, match="frozen"):
        op.warm_up(SimpleNamespace(requires_grad=True, shape=(64, 128)), 4)


@pytest.mark.parametrize("shape", [(3, 16), (2, 3, 16), (16,)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gated_input_is_the_fp32_gating_rounded_once(shape, dtype):
    from ctikernels.linear import gated_input

    x = torch.randn(shape).to(dtype)
    gate = (torch.randn(*shape[:-1], shape[-1] + 8) * 6).to(dtype)[..., : shape[-1]]
    gate[..., :2] = torch.tensor([-100.0, 100.0], dtype=dtype)
    expected = torch.sigmoid(gate.float()).mul_(x).to(dtype)
    got = gated_input(x, gate)
    assert got.shape == x.shape and got.dtype == dtype and got.is_contiguous()
    assert torch.equal(
        got.view(-1).view(torch.int16 if dtype != torch.float32 else torch.int32),
        expected.reshape(-1).view(torch.int16 if dtype != torch.float32 else torch.int32),
    )


def test_gated_input_checks_and_autograd():
    from ctikernels.linear import gated_input

    x = torch.randn(2, 8, requires_grad=True)
    for gate, error in [
        (torch.randn(2, 9), "shape"),
        (torch.randn(2, 8, device="meta"), "device"),
        (torch.zeros(2, 8, dtype=torch.int32), "floating"),
    ]:
        with pytest.raises(ValueError, match=error):
            gated_input(x, gate)
    gate = torch.randn(2, 8)
    gated_input(x, gate).sum().backward()
    assert torch.equal(x.grad, torch.sigmoid(gate))
