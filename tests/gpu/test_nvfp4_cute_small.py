# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Native FP4 small-M provider: quantization, FP64 math, tails and graph replay.

Run on RTX 5090 / RTX PRO 6000 / GB10 with DSL 4.8.0. Debug mode is the same
kernel with extra stores, allowing exact quantization checks independent of MMA order.
"""

import ast
import contextlib
import functools
import inspect
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.linear import _nvfp4_cute_small as provider
from ctikernels.linear.nvfp4 import (
    NVFP4Weight,
    blocked_scale,
    dequantize_nvfp4,
    quantize_nvfp4,
    unblocked_scale,
)

from ..support.child import run_child


def gpu_timeout(test):
    """Each GPU case gets a fresh process and a 90-second compile/launch budget."""

    @functools.wraps(test)
    def guarded(*args, **kwargs):
        case = kwargs.get("gpu")
        if isinstance(case, str):
            env = dict(os.environ, CTI_CUTE_SMALL_TEST_CHILD="1", CTI_CUTE_SMALL_DEBUG_TIMEOUT="1")
            try:
                result = run_child(
                    [sys.executable, "-m", "pytest", "-q", case], env=env, timeout=90
                )
            except subprocess.TimeoutExpired:
                pytest.fail(
                    "cute GPU case timed out after 90s (including compilation)", pytrace=False
                )
            if result.returncode:
                detail = result.stdout + result.stderr
                if "SECRET" in detail.upper() or "TOKEN" in detail.upper():
                    detail = "[child diagnostics redacted]"
                pytest.fail(f"cute GPU child exited {result.returncode}:\n{detail}", pytrace=False)
            return
        return test(*args, **kwargs)

    return guarded


@pytest.fixture
def gpu(request, monkeypatch):
    monkeypatch.setenv("CTI_CUTE_SMALL_DEBUG_TIMEOUT", "1")
    if not provider.available():
        pytest.skip("requires DSL 4.8.0 and RTX 5090 / RTX PRO 6000 / GB10")
    if os.environ.get("CTI_CUTE_SMALL_TEST_CHILD") != "1":
        return request.node.nodeid
    torch.manual_seed(912)
    return torch.device("cuda", torch.cuda.current_device())


def weight(n, k, device):
    data = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)
    scale = (torch.rand(n, k // 16, device=device) + 0.25).to(torch.float8_e4m3fn)
    return NVFP4Weight(data, scale, torch.tensor([0.03], device=device))


def oracle(x, w, global_scale):
    a = quantize_nvfp4(x, global_scale)
    av = dequantize_nvfp4(a.data, unblocked_scale(a.scale, *x.shape)).double()
    wv = dequantize_nvfp4(w.data, w.scale).double()
    return (av @ wv.T) * w.global_scale.double() * global_scale.double()


def quant_reference(value, global_scale):
    """FlashInfer byte oracle; independent GPU PTX for unrounded FP32 epilogues."""
    from .nvfp4_oracle import cute_quantize, flashinfer_quantize

    oracle = (
        flashinfer_quantize if value.dtype in (torch.bfloat16, torch.float16) else cute_quantize
    )
    return oracle(value, global_scale)


def assert_quant(x, scale, workspace):
    data, scales = quant_reference(x, scale)
    assert torch.equal(workspace.debug_data, data)
    assert torch.equal(workspace.debug_scale.view(torch.uint8), scales.view(torch.uint8))


def test_cpu_lazy_import_and_frozen_contract():
    if not torch.cuda.is_available():
        assert "ctikernels.linear._nvfp4_cute_small.kernel" not in sys.modules
    data = torch.zeros(32, 16, dtype=torch.uint8)
    sf = torch.ones(32, 2).to(torch.float8_e4m3fn)
    grad = torch.ones(1, requires_grad=True)
    with pytest.raises(ValueError, match="require grad"):
        provider.prepare(NVFP4Weight(data, sf, grad))
    with pytest.raises(ValueError, match="require grad"):
        provider.prepare_gated(NVFP4Weight(data, sf, torch.ones(1)), NVFP4Weight(data, sf, grad))


@pytest.mark.parametrize("n,k", [(1, 32), (41, 96), (128, 512), (264, 1056)])
def test_cpu_compact_tma_scales_preserve_bytes_and_zero_tails(n, k):
    # Include every FP8 bit pattern, including NaNs: preparation must not convert it.
    bits = torch.arange(n * (k // 16), dtype=torch.int64).to(torch.uint8).reshape(n, k // 16)
    scales = bits.view(torch.float8_e4m3fn)
    compact = provider._tma_scales(scales, n, k)
    blocked = provider._tma_scales(blocked_scale(scales), n, k)
    assert torch.equal(compact, blocked)
    assert compact.is_contiguous() and compact.data_ptr() % 16 == 0
    assert compact.numel() == ((n + 7) // 8) * ((k + 63) // 64) * 32
    for row in range((n + 7) // 8 * 8):
        for kb in range((k + 63) // 64 * 4):
            expected = bits[row, kb] if row < n and kb < k // 16 else 0
            assert compact[row // 8, kb // 4, row % 8, kb % 4] == expected


@pytest.mark.parametrize(
    "capability", [(12, 0), (12, 0), (12, 1)], ids=["RTX 5090", "RTX PRO 6000", "GB10"]
)
@pytest.mark.parametrize("shape", provider.SHAPES)
def test_cpu_presets_and_smem_accounting(monkeypatch, capability, shape):
    _, k, gated = provider.SHAPES[shape]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    sweep = provider.sweep(gated=gated, shape=shape)
    assert 1 <= len(sweep) <= 24 and len(set(sweep)) == len(sweep)
    small = capability == (12, 1)
    if shape.startswith("35b-shared"):
        assert {cfg.split_k for cfg in sweep} == {2, 4, 8}
        assert {cfg.n_tile for cfg in sweep} == ({32, 64} if gated else {16, 32})
        assert {cfg.qwarps for cfg in sweep} == {1, 2}
        assert all(cfg.a_stages == 4 for cfg in sweep)
        assert all(cfg.k_tile == 128 and cfg.stages == 2 and cfg.warps == 3 for cfg in sweep)
        assert all(cfg.ctas_per_sm == (2 if small else 3) for cfg in sweep)
        assert all(cfg.reduce_mode == "cluster" for cfg in sweep)
    elif gated:
        assert {cfg.n_tile for cfg in sweep} == {64, 128}
        assert {cfg.stages for cfg in sweep} == {3, 4, 5}
        assert {cfg.ctas_per_sm for cfg in sweep} == {1, 2}
        assert all(cfg.n_tile // 2 % 16 == 0 and cfg.split_k == 1 for cfg in sweep)
        assert all(cfg.reduce_mode == "global" for cfg in sweep)
    else:
        assert {cfg.n_tile for cfg in sweep} == {32, 64}
        assert {cfg.k_tile for cfg in sweep} == ({256} if small else {256, 512})
        assert {cfg.split_k for cfg in sweep} == {1, 2, 4}
        assert all(
            cfg.reduce_mode == ("global" if cfg.split_k == 1 else "cluster") for cfg in sweep
        )
    if not shape.startswith("35b-shared"):
        assert {(cfg.qwarps, cfg.a_stages) for cfg in sweep} == {
            (qw, ast) for qw in (1, 2, 4) for ast in (4, 6)
        }
    assert all((cfg.warps + cfg.qwarps) * 32 <= 256 and not cfg.pdl for cfg in sweep)
    indices = (0, 5, 10) if shape.startswith("35b-shared") else (0, 3, 4)
    assert provider.presets(gated=gated, shape=shape) == tuple(sweep[i] for i in indices)
    for cfg in provider.presets(gated=gated, shape=shape) + sweep:
        assert cfg.probe is None
        cfg.validate(k)
        assert cfg.slice_k(k) == cfg.a_stages * cfg.k_tile
        # Independently account for each allocation and its alignment.
        allocations = [
            (max(cfg.a_stages * 9 * cfg.k_tile, 64 * cfg.n_tile), 128),
            (cfg.stages * cfg.n_tile * cfg.k_tile // 2, 128),
            (cfg.stages * cfg.n_tile * cfg.k_tile // 16, 128),
            (8 * cfg.stages, 8),
            (8 * cfg.stages, 8),
            (8 * cfg.a_stages, 8),
            (8 * cfg.a_stages, 8),
            (4, 4),
        ]
        total = 0
        for size, alignment in allocations:
            total = (total + alignment - 1) // alignment * alignment + size
        assert total <= cfg.smem_bytes(k) <= cfg.smem_limit <= 99 * 1024
        assert cfg.smem_bytes(k) - total < 16


def test_cpu_config_shape_checks_precede_allocation(monkeypatch):
    cfg = provider.Config(128, 512, 2, 4, 5)
    cfg.validate(12288)
    assert cfg.smem_bytes(12288) == 92272
    with pytest.raises(ValueError, match="shared-memory"):
        provider.Config(128, 1024, 2, 8, 5)
    with pytest.raises(ValueError, match="integers"):
        provider.Config(stages=2.0)
    with pytest.raises(ValueError, match="ctas_per_sm"):
        provider.Config(ctas_per_sm=5)
    fake = torch.empty(1)
    w = provider.Weight(torch.empty(128, 6145, dtype=torch.uint8), fake, fake, fake, 0, ())
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **kw: pytest.fail("allocation before validation")
    )
    with pytest.raises(ValueError, match="divisible by 32"):
        provider.Workspace(w, 8, cfg)


@pytest.mark.parametrize("limit", [-1, 0, 7, 3072, 1 << 30])
def test_cpu_config_a_ring_ignores_split_length_and_legacy_limit(limit):
    cfg = provider.Config(16, 256, 4, 1, 3, limit, 4)
    for split in (1, 24):
        variant = replace(cfg, split_k=split)
        for k in (96, 12288, 1 << 30):
            variant.validate(k)
            assert variant.slice_k(k) == 1024
            assert variant.smem_bytes(k) == 18576
        with pytest.raises(
            ValueError, match="reduce stages, k_tile, n_tile or requested ctas_per_sm"
        ):
            replace(variant, k_tile=512)


@pytest.mark.parametrize("split", [1, 24])
def test_cpu_kernel_a_ring_budget_without_dsl_import(split):
    # Execute only the ordinary Python constructor. Importing the decorated
    # module would require the GPU-only DSL and violate the lazy-import contract.
    path = Path(provider.__file__).with_name("kernel.py")
    module = ast.parse(path.read_text())
    kernel = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Kernel"
    )
    init = next(
        node
        for node in kernel.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    scope = {}
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(path), "exec"), scope)

    def forbidden_slice(k):
        pytest.fail("the kernel must not request a full-slice A allocation")

    cfg = SimpleNamespace(
        **vars(provider.Config(16, 256, 4, split, 3, 3072, 4)),
        smem_limit=24 * 1024,
        slice_k=forbidden_slice,
    )
    for limit in (-1, 0, 7, 3072, 1 << 30):
        cfg.max_slice_k = limit
        state = SimpleNamespace()
        scope["__init__"](state, 32, 192, 12288, 12304, cfg, False, False, 0)
        assert state.per_split == (48 + split - 1) // split
    # Increasing split count cannot rescue this A-ring allocation at four CTAs.
    cfg.k_tile = 512
    with pytest.raises(ValueError, match="shared-memory budget"):
        scope["__init__"](SimpleNamespace(), 32, 192, 12288, 12304, cfg, False, False, 0)


def test_cpu_probe_and_narrow_config_contract(monkeypatch):
    cfg = provider.Config(16, 256, 4, 24, 3, 3072, 4)
    cfg.validate(12288)
    assert cfg.smem_limit == 24 * 1024
    for probe in provider.PROBES:
        variant = replace(cfg, probe=probe)
        variant.validate(12288)
        assert variant.smem_bytes(12288) == cfg.smem_bytes(12288)
        assert variant != cfg
    with pytest.raises(ValueError, match="probe"):
        replace(cfg, probe="unknown")
    with pytest.raises(ValueError, match="consumer warps"):
        replace(cfg, warps=5)
    fake = torch.empty(1)
    w = provider.Weight(torch.empty(128, 6144, dtype=torch.uint8), fake, fake, fake, 0, ())
    monkeypatch.setattr(
        torch, "zeros", lambda *a, **kw: pytest.fail("allocation before validation")
    )
    with pytest.raises(ValueError, match="n_tile >= 32"):
        provider.Workspace(w, 8, cfg, gated=True)


@pytest.mark.parametrize("kind,name", [(1, "full"), (2, "empty"), (3, "A full"), (4, "A empty")])
def test_cpu_debug_timeout_decoding_and_capture_guard(monkeypatch, kind, name):
    work = provider.Workspace.__new__(provider.Workspace)
    work.debug_timeout = True
    work.errors = torch.tensor([1, kind, 3, 1, 7, 2, 1], dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for _ in range(2):
        with pytest.raises(RuntimeError, match=rf"{name}.*stage=3, phase=1, block=\(7, 2, 1\)"):
            work.check_debug()
    work.errors.zero_()
    work.check_debug()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        work.check_debug()
    work.debug_timeout = False
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: pytest.fail("production host query")
    )
    work.check_debug()


def test_cpu_gpu_timeout_guard(monkeypatch):
    def stalled(*args, **kwargs):
        assert kwargs["timeout"] == 90
        assert kwargs["env"]["CTI_CUTE_SMALL_DEBUG_TIMEOUT"] == "1"
        assert kwargs["env"]["CTI_CUTE_SMALL_TEST_CHILD"] == "1"
        raise subprocess.TimeoutExpired("pytest", 90)

    monkeypatch.setitem(globals(), "run_child", stalled)
    guarded = gpu_timeout(lambda gpu: pytest.fail("parent must not launch GPU work"))
    with pytest.raises(pytest.fail.Exception, match="timed out after 90s"):
        guarded(gpu="tests/gpu/test_nvfp4_cute_small.py::example")


def test_cpu_timeout_kernel_local_annotations(monkeypatch):
    # Model DSL annotation resolution without importing the DSL or compiling a kernel.
    pointer, stream = object(), object()
    annotations = []

    class Decorated(Exception):
        pass

    def kernel(fn):
        annotations.append(inspect.get_annotations(fn, eval_str=True))
        return fn

    def jit(fn):
        annotations.append(inspect.get_annotations(fn, eval_str=True))
        raise Decorated

    cute = SimpleNamespace(Pointer=pointer, kernel=kernel, jit=jit)
    driver = SimpleNamespace(CUstream=stream)
    bindings = SimpleNamespace(driver=driver)
    modules = {
        "cuda": SimpleNamespace(bindings=bindings),
        "cuda.bindings": bindings,
        "cuda.bindings.driver": driver,
        "cutlass": SimpleNamespace(Int32=object(), Int64=object(), cute=cute),
        "cutlass.cute.runtime": SimpleNamespace(make_ptr=None),
        "cutlass.utils": SimpleNamespace(SmemAllocator=None),
        "ctikernels.linear._nvfp4_cute_small.kernel": SimpleNamespace(debug_wait=None),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(Decorated):
        test_debug_wait_timeout_exits_warps_and_reports.__wrapped__(None, 1, 0)
    assert annotations == [
        {"errors": pointer, "reached": pointer},
        {"errors": pointer, "reached": pointer, "stream": stream},
    ]


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 17, 32])
@pytest.mark.parametrize("n,k", [(41, 96), (128, 512), (264, 1056)])
@pytest.mark.parametrize("splits", [1, 3])
@gpu_timeout
def test_linear_quantized_fp64_and_tails(gpu, m, n, k, splits):
    w = weight(n, k, gpu)
    scale = torch.tensor([0.01], device=gpu)
    # Noncontiguous rows with unit inner stride are part of the public contract.
    x = (torch.randn(m, k + 16, device=gpu, dtype=torch.bfloat16) * 0.1)[:, :k]
    cfg = provider.Config(64, 256, 2, splits, 3)
    packed = provider.prepare(w)
    work = provider.Workspace(packed, m, cfg, debug=True)
    got = provider.linear(x, packed, scale, cfg, workspace=work)
    assert_quant(x, scale, work)
    expected = oracle(x, w, scale)
    torch.testing.assert_close(work.projection.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(got.double(), expected, rtol=0.004, atol=2e-6)
    assert torch.count_nonzero(work.counters) == 0


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 8, 17, 32])
@pytest.mark.parametrize("i,k", [(32, 96), (160, 288)])
@gpu_timeout
def test_gated_fp32_epilogue_and_exact_output_quantization(gpu, m, i, k):
    g, u = weight(i, k, gpu), weight(i, k, gpu)
    u.global_scale.mul_(0.75)
    ig, og = torch.tensor([0.01], device=gpu), torch.tensor([0.0001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.float16) * 0.1
    cfg = provider.Config(32, 128, 3, 3, 2)
    packed = provider.prepare_gated(g, u)
    work = provider.Workspace(packed, m, cfg, gated=True, debug=True)
    got = provider.gated(x, packed, ig, og, cfg, workspace=work)
    assert_quant(x, ig, work)
    gate, up = oracle(x, g, ig), oracle(x, u, ig)
    expected = gate * torch.sigmoid(gate) * up
    torch.testing.assert_close(work.projection[:, ::2].double(), gate, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(work.projection[:, 1::2].double(), up, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(work.postact.double(), expected, rtol=4e-5, atol=2e-6)
    # The epilogue is SGLang's SiLU (torch's accurate expf and IEEE division), bit for bit.
    pg, pu = work.projection[:, ::2], work.projection[:, 1::2]
    assert torch.equal(work.postact, pg / (1 + torch.exp(-pg)) * pu)
    data, sf = quant_reference(work.postact, og)
    assert torch.equal(got.data, data)
    assert torch.equal(got.scale.view(torch.uint8), sf.view(torch.uint8))
    # The output bytes are the exact quantization of the checked FP32 epilogue; the
    # five-seed FP64 gate with the measured-noise margin is in test_nvfp4.py
    # (test_cute_dispatch_buckets_* and the generic-Config precision variants).


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@gpu_timeout
def test_quantization_ties_zero_subnormal_saturation_and_nonfinite(gpu, dtype):
    w = weight(32, 256, gpu)
    x = torch.zeros(2, 256, dtype=dtype, device=gpu)
    ties = torch.tensor(
        [0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -6, -5, -2.5, -0.75, -0.25, 6], device=gpu
    )
    x[0, :16] = ties
    x[0, 16:32] = ties * (2.0**-10)
    x[0, 32:48] = ties * 512
    x[1, :16] = -ties
    x[0, 55] = float("nan")
    x[0, 71] = float("inf")
    x[0, 95] = -float("inf")
    scale = torch.ones(1, device=gpu)
    packed = provider.prepare(w)
    cfg = provider.Config(32, 128, 2, 1, 2)
    work = provider.Workspace(packed, 2, cfg, debug=True)
    provider.linear(x, packed, scale, cfg, workspace=work)
    assert_quant(x, scale, work)
    data, sf = quant_reference(x, scale)
    assert torch.equal(work.debug_data, data)
    assert torch.equal(work.debug_scale.view(torch.uint8), sf.view(torch.uint8))


@pytest.mark.parametrize("gated", [False, True])
@gpu_timeout
def test_mutable_globals_and_deterministic_graph_replays(gpu, gated):
    g, u = weight(96, 1056, gpu), weight(96, 1056, gpu)
    packed = provider.prepare_gated(g, u) if gated else provider.prepare(g)
    ig, og = torch.tensor([0.01], device=gpu), torch.tensor([0.0001], device=gpu)
    x = torch.randn(5, 1056, device=gpu, dtype=torch.bfloat16) * 0.1
    cfg = provider.Config(64, 256, 2, 3, 3)
    work = provider.Workspace(packed, 5, cfg, gated=gated, debug=True)

    def run():
        if gated:
            return provider.gated(x, packed, ig, og, cfg, workspace=work)
        return provider.linear(x, packed, ig, cfg, workspace=work)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        ig.mul_(1.125)
        og.mul_(0.875)
        g.global_scale.mul_(0.75)
        u.global_scale.mul_(1.25)
        graph.replay()
        work.check_debug()
        expected_projection = oracle(x, g, ig)
        actual_projection = work.projection[:, ::2] if gated else work.projection
        torch.testing.assert_close(
            actual_projection.double(), expected_projection, rtol=2e-5, atol=2e-6
        )
        if gated:
            torch.testing.assert_close(
                work.projection[:, 1::2].double(), oracle(x, u, ig), rtol=2e-5, atol=2e-6
            )
            q, sf = quant_reference(work.postact, og)
            assert torch.equal(work.out, q)
            assert torch.equal(work.out_scale.view(torch.uint8), sf.view(torch.uint8))
        assert_quant(x, ig, work)
        baseline, projection = work.out.clone(), work.projection.clone()
        for _ in range(10):
            graph.replay()
            work.check_debug()
            assert torch.equal(work.out, baseline)
            assert torch.equal(work.projection, projection)
            assert torch.count_nonzero(work.counters) == 0
    g.global_scale.requires_grad_()
    # Mutation after preparation must still be rejected by the Python launch wrapper.
    with pytest.raises(ValueError, match="require grad"):
        run()


@pytest.mark.parametrize("field", ["data", "scale", "global_scale"])
@gpu_timeout
def test_every_weight_field_rejects_grad_before_launch(gpu, field, monkeypatch):
    w = weight(32, 128, gpu)
    values = dict(w._asdict())
    values[field] = values[field].float().requires_grad_()
    monkeypatch.setattr(torch, "stack", lambda *a, **kw: pytest.fail("conversion launched"))
    with pytest.raises(ValueError, match="require grad"):
        provider.prepare(NVFP4Weight(**values))
    with pytest.raises(ValueError, match="require grad"):
        provider.prepare_gated(w, NVFP4Weight(**values))


@gpu_timeout
def test_per_column_factors_are_read_on_graph_replay(gpu):
    w = weight(192, 256, gpu)
    factors = torch.full((192,), 0.03, device=gpu)
    packed = provider.prepare(NVFP4Weight(w.data, blocked_scale(w.scale), factors))
    ig = torch.tensor([0.01], device=gpu)
    x = torch.randn(3, 256, device=gpu, dtype=torch.bfloat16) * 0.1
    cfg = provider.Config(128, 256, 2, 2, 5)
    work = provider.Workspace(packed, 3, cfg, debug=True)
    provider.linear(x, packed, ig, cfg, workspace=work)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider.linear(x, packed, ig, cfg, workspace=work)
    factors[::2].mul_(0.5)
    factors[1::2].mul_(1.5)
    graph.replay()
    work.check_debug()
    a = quantize_nvfp4(x, ig)
    av = dequantize_nvfp4(a.data, unblocked_scale(a.scale, *x.shape)).double()
    expected = (av @ dequantize_nvfp4(w.data, w.scale).double().T) * factors.double() * ig.double()
    torch.testing.assert_close(work.projection.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("m", [1, 8])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "cfg,k",
    [
        (provider.Config(128, 512, 2, 8, 5), 12288),
        (provider.Config(32, 256, 8, 1, 3), 2304),
        (provider.Config(32, 1024, 2, 1, 3), 3072),
        (provider.Config(128, 1024, 1, 3, 5, a_stages=2), 6144),
        (provider.Config(64, 512, 2, 12, 3, 3072, 2, a_stages=2), 12288),
        (provider.Config(32, 128, 1, 1, 2, 3072, 2), 288),
        (provider.Config(32, 128, 2, 1, 3, 3072, 2), 640),
        (provider.Config(32, 128, 3, 4, 3, 3072, 2), 288),
    ],
)
@gpu_timeout
def test_tma_and_a_ring_wrap(gpu, m, gated, cfg, k):
    w, up = weight(160, k, gpu), weight(160, k, gpu)
    packed = provider.prepare_gated(w, up) if gated else provider.prepare(w)
    ig = torch.tensor([0.01], device=gpu)
    og = torch.tensor([0.001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.bfloat16) * 0.1
    work = provider.Workspace(packed, m, cfg, gated=gated, debug=True)

    def run():
        if gated:
            return provider.gated(x, packed, ig, og, cfg, workspace=work)
        return provider.linear(x, packed, ig, cfg, workspace=work)

    run()
    assert_quant(x, ig, work)
    expected = oracle(x, w, ig)
    if gated:
        expected = torch.stack((expected, oracle(x, up, ig)), -1).flatten(1)
    torch.testing.assert_close(work.projection.double(), expected, rtol=5e-5, atol=2e-5)
    baseline = work.out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(5):
        graph.replay()
        work.check_debug()
        assert torch.equal(work.out, baseline)
        assert torch.count_nonzero(work.counters) == 0


@gpu_timeout
def test_tensor_maps_follow_weight_addresses_in_captured_rotation(gpu):
    raw = [weight(41, 1056, gpu) for _ in range(3)]
    packed = [provider.prepare(w) for w in raw]
    cfg = provider.Config(32, 512, 2, 1, 3)
    work = [provider.Workspace(w, 8, cfg, debug=True) for w in packed]
    ig = torch.tensor([0.01], device=gpu)
    x = torch.randn(8, 1056, device=gpu, dtype=torch.bfloat16) * 0.1
    for w, ws in zip(packed, work, strict=False):
        provider.linear(x, w, ig, cfg, workspace=ws)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w, ws in zip(packed, work, strict=False):
            provider.linear(x, w, ig, cfg, workspace=ws)
    for _ in range(3):
        ig.mul_(1.125)
        graph.replay()
        for w, ws in zip(raw, work, strict=False):
            ws.check_debug()
            assert_quant(x, ig, ws)
            torch.testing.assert_close(
                ws.projection.double(), oracle(x, w, ig), rtol=2e-5, atol=2e-6
            )


@pytest.mark.parametrize("kind,phase", [(1, 0), (2, 1)])
@gpu_timeout
def test_debug_wait_timeout_exits_warps_and_reports(gpu, kind, phase):
    import cuda.bindings.driver as cuda
    from cutlass import Int32, Int64, cute
    from cutlass.cute.runtime import make_ptr
    from cutlass.utils import SmemAllocator

    from ctikernels.linear._nvfp4_cute_small.kernel import debug_wait

    @cute.kernel
    def stalled(errors: cute.Pointer, reached: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        alloc = SmemAllocator()
        barrier = alloc.allocate_tensor(Int64, 1, byte_alignment=8)
        abort = alloc.allocate_tensor(Int32, 1, byte_alignment=4)
        if tid == 64:
            abort[0] = Int32(0)
            cute.arch.mbarrier_init(barrier.iterator, 1)
            if phase == 1:
                cute.arch.mbarrier_arrive(barrier.iterator)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if tid < 64:
            debug_wait(barrier.iterator, errors, abort.iterator, kind, 2, phase, tries=8)
        cute.arch.sync_threads()
        if tid == 64:
            target = cute.make_tensor(reached, cute.make_layout(1))
            target[0] = abort[0]

    @cute.jit
    def launch(errors: cute.Pointer, reached: cute.Pointer, stream: cuda.CUstream):
        stalled(errors, reached).launch(grid=(1, 1, 1), block=(96, 1, 1), stream=stream)

    work = provider.Workspace.__new__(provider.Workspace)
    work.debug_timeout = True
    work.errors = torch.zeros(7, dtype=torch.int32, device=gpu)
    reached = torch.zeros(1, dtype=torch.int32, device=gpu)
    pointers = [
        make_ptr(Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        for t in (work.errors, reached)
    ]
    stream = cuda.CUstream(torch.cuda.current_stream(gpu).cuda_stream)
    arch = "sm_121a" if torch.cuda.get_device_capability(gpu) == (12, 1) else "sm_120a"
    compiled = cute.compile(launch, *pointers, stream, options=f"--gpu-arch {arch}")
    compiled(*pointers, stream)
    with pytest.raises(RuntimeError, match=rf"stage=2, phase={phase}, block=\(0, 0, 0\)"):
        work.check_debug()
    assert work.errors.cpu().tolist() == [1, kind, 2, phase, 0, 0, 0]
    assert reached.item() == 1


@pytest.mark.parametrize("probe", provider.PROBES)
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("splits,mode", [(1, "global"), (2, "global"), (4, "cluster")])
@gpu_timeout
def test_diagnostic_probes_replay(gpu, probe, gated, splits, mode):
    k, m = 1056, 8
    raw = weight(64, k, gpu)
    packed = provider.prepare_gated(raw, raw) if gated else provider.prepare(raw)
    cfg = provider.Config(32, 128, 2, splits, 3, 512, 4, reduce_mode=mode)
    variant = replace(cfg, probe=probe)
    full = provider.Workspace(packed, m, cfg, gated=gated)
    work = provider.Workspace(packed, m, variant, gated=gated)
    ig = torch.tensor([0.01], device=gpu)
    og = torch.tensor([0.001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.bfloat16) * 0.1

    def run(config, workspace):
        if gated:
            return provider.gated(x, packed, ig, og, config, workspace=workspace)
        return provider.linear(x, packed, ig, config, workspace=workspace)

    run(cfg, full)
    work.out.fill_(7)
    work.out_scale.view(torch.uint8).fill_(7)
    work.partial.fill_(float("nan"))
    run(variant, work)
    baseline = work.out.clone()
    baseline_partial = work.partial.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(variant, work)
    for _ in range(3):
        graph.replay()
        work.check_debug()
        assert torch.count_nonzero(work.counters) == 0
        assert torch.equal(work.out, baseline)
        if probe == "load_only":
            per_split = (9 + splits - 1) // splits
            count = torch.tensor(
                [max(min(per_split, 9 - rank * per_split), 0) for rank in range(splits)],
                device=gpu,
                dtype=torch.int32,
            )
            assert torch.equal(work.probe_counts, count.expand_as(work.probe_counts))
            assert torch.all(work.out == 7)
            assert torch.all(work.out_scale.view(torch.uint8) == 7)
        elif probe == "no_reduce":
            assert torch.equal(work.partial, full.partial)
            assert torch.all(work.out == 7)
            assert torch.all(work.out_scale.view(torch.uint8) == 7)
        elif probe == "no_prologue":
            assert torch.count_nonzero(work.out) == 0
        else:
            assert torch.equal(work.partial, baseline_partial)
            assert torch.isfinite(work.partial).all()


@pytest.mark.parametrize("m", [1, 8])
@gpu_timeout
def test_n16_four_ctas_and_mutable_scales(gpu, m):
    raw = weight(41, 1056, gpu)
    packed = provider.prepare(raw)
    cfg = provider.Config(16, 256, 4, 3, 3, 3072, 4)
    work = provider.Workspace(packed, m, cfg, debug=True)
    ig = torch.tensor([0.01], device=gpu)
    x = torch.randn(m, 1056, device=gpu, dtype=torch.bfloat16) * 0.1
    provider.linear(x, packed, ig, cfg, workspace=work)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        provider.linear(x, packed, ig, cfg, workspace=work)
    for _ in range(3):
        ig.mul_(1.125)
        raw.global_scale.mul_(0.875)
        graph.replay()
        work.check_debug()
        assert_quant(x, ig, work)
        torch.testing.assert_close(
            work.projection.double(), oracle(x, raw, ig), rtol=5e-5, atol=2e-5
        )


@pytest.mark.parametrize(
    "mode,splits", [("global", 1), ("global", 4), ("cluster", 2), ("cluster", 8)]
)
@pytest.mark.parametrize("m", [1, 3, 8, 17, 32])
def test_cpu_reduction_workspace_storage(mode, splits, m):
    fake = torch.empty(1)
    packed = provider.Weight(torch.empty(64, 256, dtype=torch.uint8), fake, fake, fake, 0, ())
    cfg = provider.Config(32, 128, 2, splits, 3, reduce_mode=mode)
    work = provider.Workspace(packed, m, cfg)
    if mode == "global" and splits > 1:
        assert work.partial.shape == (2, splits, m, 32)
        assert work.counters.numel() == ((m + 15) // 16) * 2
    else:
        assert work.partial.numel() == work.counters.numel() == 0


def test_cpu_cluster_validation():
    for splits in (1, 3, 12, 16):
        with pytest.raises(ValueError, match="portable cluster"):
            provider.Config(split_k=splits, reduce_mode="cluster")
    with pytest.raises(ValueError, match="reduce_mode"):
        provider.Config(reduce_mode="unknown")


@pytest.mark.parametrize("m", [1, 3, 8, 17, 32])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "cfg,k",
    [
        (provider.Config(32, 512, 2, 1, 3, 4096), 4096),
        (provider.Config(32, 512, 2, 1, 3, 3072), 12288),
        (provider.Config(32, 128, 2, 1, 3, 512), 1056),
        (provider.Config(32, 128, 2, 2, 3, reduce_mode="cluster"), 1056),
        (provider.Config(64, 128, 2, 4, 3, reduce_mode="cluster"), 1056),
        (provider.Config(32, 128, 2, 8, 3, reduce_mode="cluster"), 288),
        (provider.Config(32, 512, 2, 4, 3), 12288),
    ],
)
@gpu_timeout
def test_reduction_modes_oracle_and_mutable_graph(gpu, m, gated, cfg, k):
    raw, up = weight(96, k, gpu), weight(96, k, gpu)
    packed = provider.prepare_gated(raw, up) if gated else provider.prepare(raw)
    ig = torch.tensor([0.01], device=gpu)
    og = torch.tensor([0.001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.bfloat16) * 0.1
    work = provider.Workspace(packed, m, cfg, gated=gated, debug=True)

    def run():
        if gated:
            return provider.gated(x, packed, ig, og, cfg, workspace=work)
        return provider.linear(x, packed, ig, cfg, workspace=work)

    def check():
        work.check_debug()
        assert_quant(x, ig, work)
        gate = oracle(x, raw, ig)
        expected = torch.stack((gate, oracle(x, up, ig)), -1).flatten(1) if gated else gate
        torch.testing.assert_close(work.projection.double(), expected, rtol=8e-5, atol=3e-5)
        if gated:
            post = gate * torch.sigmoid(gate) * oracle(x, up, ig)
            torch.testing.assert_close(work.postact.double(), post, rtol=1e-4, atol=3e-5)
            data, scale = quant_reference(work.postact, og)
            assert torch.equal(work.out, data)
            assert torch.equal(work.out_scale.view(torch.uint8), scale.view(torch.uint8))
        else:
            torch.testing.assert_close(work.out.double(), gate, rtol=0.004, atol=3e-5)
        assert torch.count_nonzero(work.counters) == 0

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        ig.mul_(1.125)
        raw.global_scale.mul_(0.875)
        up.global_scale.mul_(1.0625)
        og.mul_(0.9375)
        graph.replay()
        check()
        baseline = work.out.clone()
        projection = work.projection.clone()
        scale = work.out_scale.view(torch.uint8).clone() if gated else None
        for _ in range(3):
            graph.replay()
            work.check_debug()
            assert torch.equal(work.out, baseline)
            assert torch.equal(work.projection, projection)
            if gated:
                assert torch.equal(work.out_scale.view(torch.uint8), scale)


@pytest.mark.parametrize("m", [1, 8, 16, 17, 32])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "stages,splits,mode", [(1, 1, "global"), (3, 4, "global"), (6, 4, "cluster")]
)
@gpu_timeout
def test_a_ring_ignores_legacy_slice_and_replays(gpu, m, gated, stages, splits, mode):
    # Odd split lengths wrap A independently of B; the final K32 tail must
    # overwrite live rows in a reused slot, while padded M rows stay zero.
    k = 4128
    raw, up = weight(96, k, gpu), weight(96, k, gpu)
    packed = provider.prepare_gated(raw, up) if gated else provider.prepare(raw)
    ig, og = torch.tensor([0.01], device=gpu), torch.tensor([0.001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.bfloat16) * 0.1
    configs = [
        provider.Config(64, 256, stages, splits, 3, limit, reduce_mode=mode)
        for limit in (256, 3072)
    ]
    works = [provider.Workspace(packed, m, cfg, gated=gated, debug=True) for cfg in configs]

    def run():
        for cfg, work in zip(configs, works, strict=False):
            if gated:
                provider.gated(x, packed, ig, og, cfg, workspace=work)
            else:
                provider.linear(x, packed, ig, cfg, workspace=work)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        x.mul_(0.9375)
        ig.mul_(1.125)
        og.mul_(0.875)
        graph.replay()
        for work in works:
            work.check_debug()
            assert_quant(x, ig, work)
            expected = oracle(x, raw, ig)
            if gated:
                expected = torch.stack((expected, oracle(x, up, ig)), -1).flatten(1)
                data, sf = quant_reference(work.postact, og)
                assert torch.equal(work.out, data)
                assert torch.equal(work.out_scale.view(torch.uint8), sf.view(torch.uint8))
            torch.testing.assert_close(work.projection.double(), expected, rtol=8e-5, atol=3e-5)
            assert torch.count_nonzero(work.counters) == 0
        assert torch.equal(works[0].out, works[1].out)
        assert torch.equal(works[0].projection, works[1].projection)


@pytest.mark.parametrize(
    "field,value",
    [
        ("qwarps", 0),
        ("qwarps", 5),
        ("qwarps", True),
        ("qwarps", 1.0),
        ("a_stages", 1),
        ("a_stages", 9),
        ("a_stages", False),
        ("a_stages", 4.0),
        ("pdl", 1),
        ("pdl", "true"),
    ],
)
def test_cpu_a_ring_config_validation(field, value):
    with pytest.raises(ValueError):
        provider.Config(**{field: value})


@pytest.mark.parametrize("qwarps", [1, 2, 3, 4])
@pytest.mark.parametrize("a_stages", [2, 4, 8])
@pytest.mark.parametrize("n_tile,k_tile", [(16, 128), (128, 128), (32, 256)])
def test_cpu_a_ring_budget_and_legacy_warp_count(qwarps, a_stages, n_tile, k_tile):
    if a_stages < qwarps:
        # Tile-interleaved quantizer warps need one slot per warp in flight.
        with pytest.raises(ValueError, match="a_stages >= qwarps"):
            provider.Config(n_tile, k_tile, 3, 1, 3, qwarps=qwarps, a_stages=a_stages)
        return
    cfg = provider.Config(n_tile, k_tile, 3, 1, 3, qwarps=qwarps, a_stages=a_stages, pdl=True)
    ring = a_stages * (16 * k_tile // 2 + 16 * k_tile // 16)
    result = 16 * n_tile * 4
    expected = max(ring, result) + 3 * n_tile * k_tile * 9 // 16 + 16 * (3 + a_stages) + 16
    assert cfg.smem_bytes(32) == cfg.smem_bytes(12288) == expected
    assert cfg.warps == 3
    from ctikernels.linear._nvfp4_dispatch import Cute

    plan = Cute(**{key: value for key, value in vars(cfg).items() if key != "probe"})
    assert plan.config() == cfg


@pytest.mark.parametrize(
    "value,expected",
    [
        ("64:512:4:1:3", provider.Config(64, 512, 4, 1, 3)),
        ("64:512:4:1:3:6144", provider.Config(64, 512, 4, 1, 3, 6144)),
        (
            "32:256:3:2:3:6144:1:cluster",
            provider.Config(32, 256, 3, 2, 3, 6144, 1, reduce_mode="cluster"),
        ),
        (
            "64:512:4:1:3:6144:1:q2:a4:pdl",
            provider.Config(64, 512, 4, 1, 3, 6144, 1, qwarps=2, a_stages=4, pdl=True),
        ),
        (
            "32:256:3:4:3:q4:a8:pdl:cluster",
            provider.Config(
                32, 256, 3, 4, 3, reduce_mode="cluster", qwarps=4, a_stages=8, pdl=True
            ),
        ),
    ],
)
def test_cpu_config_string_parser(value, expected):
    assert provider.parse_config(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "64:512:4:1",
        "64:512:4:1:3:1:1:1",
        "64:512:4:1:3:q0",
        "64:512:4:1:3:a9",
        "64:512:4:1:3:pdl:pdl",
        "64:512:4:1:3:q1:q2",
        "64:512:4:1:3:a2:a4",
        "64:512:4:1:3:global:cluster",
        "64:512:4:1:3:q2:6144",
        "64:512:4:1:3:unknown",
    ],
)
def test_cpu_config_string_parser_rejects_invalid(value):
    with pytest.raises(ValueError):
        provider.parse_config(value)


@pytest.mark.parametrize("m", [1, 2, 3, 5, 8, 16, 17, 32])
@pytest.mark.parametrize("qwarps,a_stages", [(1, 2), (1, 4), (2, 2), (2, 4)])
@pytest.mark.parametrize(
    "k,splits,mode",
    [
        (96, 1, "global"),
        (1184, 1, "global"),
        (4128, 3, "global"),
        (2336, 2, "cluster"),
        (4128, 4, "cluster"),
        (96, 4, "cluster"),
    ],
)
@pytest.mark.parametrize("gated", [False, True])
@gpu_timeout
def test_quantizer_ring_pdl_and_mutable_graph(gpu, m, qwarps, a_stages, k, splits, mode, gated):
    # Count 1, multiple ring wraps, K tails, and entirely empty cluster splits.
    raw, up = weight(64, k, gpu), weight(64, k, gpu)
    packed = provider.prepare_gated(raw, up) if gated else provider.prepare(raw)
    ig, og = torch.tensor([0.01], device=gpu), torch.tensor([0.001], device=gpu)
    x = torch.randn(m, k, device=gpu, dtype=torch.bfloat16) * 0.1
    configs = [
        provider.Config(
            32, 128, 3, splits, 3, reduce_mode=mode, qwarps=qwarps, a_stages=a_stages, pdl=pdl
        )
        for pdl in (False, True)
    ]
    works = [provider.Workspace(packed, m, cfg, gated=gated, debug=True) for cfg in configs]

    def run(cfg, work):
        if gated:
            return provider.gated(x, packed, ig, og, cfg, workspace=work)
        return provider.linear(x, packed, ig, cfg, workspace=work)

    for cfg, work in zip(configs, works, strict=False):
        run(cfg, work)
    graph = torch.cuda.CUDAGraph()
    observed = [torch.empty_like(work.out) for work in works]
    with torch.cuda.graph(graph):
        for cfg, work, snapshot in zip(configs, works, observed, strict=False):
            # Reuse workspace after an output reader; the next launch's writes
            # must wait even though its independent B producer can start early.
            run(cfg, work)
            snapshot.copy_(work.out)
            run(cfg, work)
    for _ in range(3):
        x.mul_(0.9375)
        ig.mul_(1.125)
        og.mul_(0.875)
        raw.global_scale.mul_(0.75)
        up.global_scale.mul_(1.0625)
        graph.replay()
        expected = oracle(x, raw, ig)
        other = oracle(x, up, ig) if gated else None
        projection = torch.stack((expected, other), -1).flatten(1) if gated else expected
        for work, snapshot in zip(works, observed, strict=False):
            work.check_debug()
            assert_quant(x, ig, work)
            assert torch.equal(work.out, snapshot)
            torch.testing.assert_close(work.projection.double(), projection, rtol=8e-5, atol=3e-5)
            if gated:
                torch.testing.assert_close(
                    work.postact.double(),
                    expected * torch.sigmoid(expected) * other,
                    rtol=1e-4,
                    atol=3e-5,
                )
                data, sf = quant_reference(work.postact, og)
                assert torch.equal(work.out, data)
                assert torch.equal(work.out_scale.view(torch.uint8), sf.view(torch.uint8))
            else:
                torch.testing.assert_close(work.out.double(), expected, rtol=0.004, atol=3e-5)
            assert torch.count_nonzero(work.counters) == 0
        assert torch.equal(works[0].out, works[1].out)
        assert torch.equal(works[0].projection, works[1].projection)
        if gated:
            assert torch.equal(
                works[0].out_scale.view(torch.uint8), works[1].out_scale.view(torch.uint8)
            )


def test_cpu_rejects_single_scale_atom_k():
    for k in (32, 64):
        with pytest.raises(ValueError, match="exceed 64"):
            provider.Config(32, 128, 3, 1, 3).validate(k)
    provider.Config(32, 128, 3, 1, 3).validate(96)


def test_cpu_total_thread_limit():
    provider.Config(64, 128, 2, 1, 5, qwarps=3)
    with pytest.raises(ValueError, match="256 threads"):
        provider.Config(64, 128, 2, 1, 5, qwarps=4)


def test_cpu_expanded_input_is_not_eligible():
    x = torch.zeros(1, 128, dtype=torch.bfloat16).expand(3, 128)
    assert x.stride(0) == 0 and not provider.input_aligned(x)
    assert provider.input_aligned(torch.zeros(3, 128, dtype=torch.bfloat16))


@pytest.mark.parametrize("offset,padding", [(1, 8), (0, 1)])
def test_cpu_unaligned_input_rejected_before_launch(offset, padding, monkeypatch):
    m, n, k = 3, 32, 128
    x = torch.empty(m, k + padding, dtype=torch.bfloat16)[:, offset : offset + k]
    scale = torch.ones(1)
    packed = provider.Weight(
        torch.empty(n, k // 2, dtype=torch.uint8),
        torch.empty(1, dtype=torch.uint8),
        scale,
        scale,
        0,
        (),
    )
    monkeypatch.setattr(
        provider, "available", lambda _: pytest.fail("checked availability before alignment")
    )
    with pytest.raises(ValueError, match="16-byte alignment"):
        provider.linear(x, packed, scale, workspace=SimpleNamespace())


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("quack", [False, True])
@pytest.mark.parametrize("offset,padding", [(1, 8), (0, 1), (None, 0)])
def test_cpu_unaligned_operator_fallback(monkeypatch, force, quack, offset, padding):
    from ctikernels.linear import _nvfp4_dispatch, _nvfp4_quack, nvfp4

    monkeypatch.setattr(nvfp4, "_cuda_device", lambda operation, tensor: tensor.device)
    monkeypatch.setattr(nvfp4, "_require_fp8_capable", lambda *args: None)
    monkeypatch.setattr(_nvfp4_quack, "load", lambda _: object() if quack else None)
    monkeypatch.setattr(_nvfp4_dispatch, "_device_type", lambda _: "test-device")
    monkeypatch.setattr(provider, "available", lambda _: True)
    monkeypatch.setattr(provider, "prepare_gated", lambda *args: args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Capture checks enter the target-device context; these operands live on the CPU.
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    plan = _nvfp4_dispatch.Cute(n_tile=32, k_tile=128, split_k=1)
    fallback = _nvfp4_quack.DEFAULT
    entries = ((32, "cute", plan), (64, _nvfp4_dispatch.UNFUSED, fallback))
    monkeypatch.setitem(_nvfp4_dispatch.GATED, "test-device", {(128, 64): entries})
    raw, scale = weight(64, 128, "cpu"), torch.ones(1)
    kwargs = {"provider": "cute"} if force else {}
    op = nvfp4.NVFP4GateUpSiLU(raw, raw, input_scale=scale, output_scale=scale, **kwargs)
    assert op.plan_for(3) == ("cute", plan)
    events = []

    def quantize(x, scale, out, **kwargs):
        events.append("quantize")
        return "triton"

    def project(a, out, *args, **kwargs):
        events.append(args)
        return out

    monkeypatch.setattr(nvfp4, "_quantize_into", quantize)
    monkeypatch.setattr(op, "_project", project)
    monkeypatch.setattr(
        provider, "Workspace", lambda *a, **kw: pytest.fail("CuTe workspace on fallback")
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("fallback must only inspect input metadata"),
    )
    if offset is None:  # expanded rows (stride 0)
        x = torch.zeros(1, 128, dtype=torch.bfloat16).expand(3, 128)
    else:
        x = torch.zeros(3, 128 + padding, dtype=torch.bfloat16)[:, offset : offset + 128]
    op(x)
    expected = (_nvfp4_dispatch.UNFUSED, fallback) if quack else ("dequant", None)
    assert events == ["quantize", expected]
    assert op.quantizer == "triton" and not op._cute_workspaces
    assert op.plan_for(3) == ("cute", plan)


@pytest.mark.parametrize("m", [1, 3, 8, 16, 17, 32])
@pytest.mark.parametrize("qwarps", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@gpu_timeout
def test_batched_block_quantizer_tail_oracle(gpu, m, qwarps, dtype):
    k, n = 1056, 64
    raw, scale = weight(n, k, gpu), torch.tensor([0.01], device=gpu)
    x = (torch.randn(m, k + 8, dtype=dtype, device=gpu) * 0.1)[:, :k]
    cfg = provider.Config(32, 256, 3, 1, 3, qwarps=qwarps, a_stages=6 if qwarps == 2 else 4)
    packed = provider.prepare(raw)
    work = provider.Workspace(packed, m, cfg, debug=True)
    got = provider.linear(x, packed, scale, cfg, workspace=work)
    assert_quant(x, scale, work)
    expected = oracle(x, raw, scale)
    torch.testing.assert_close(work.projection.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(got.double(), expected, rtol=0.004, atol=2e-6)


@gpu_timeout
def test_quantizer_multiple_load_batches(gpu):
    m, k = 17, 1056
    raw, scale = weight(32, k, gpu), torch.tensor([0.01], device=gpu)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=gpu) * 0.1
    cfg = provider.Config(32, 1024, 2, 1, 3, qwarps=1, a_stages=2)
    packed = provider.prepare(raw)
    work = provider.Workspace(packed, m, cfg, debug=True)
    provider.linear(x, packed, scale, cfg, workspace=work)
    assert_quant(x, scale, work)
    torch.testing.assert_close(
        work.projection.double(), oracle(x, raw, scale), rtol=2e-5, atol=2e-6
    )
