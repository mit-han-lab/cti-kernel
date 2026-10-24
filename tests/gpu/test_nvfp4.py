# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""NVFP4 linear and gated projection on the GPU: the quantizer bit for bit against its recipe,
every provider against an FP64 oracle on the same quantized operands (no worse than an FP32
reference, the fused projection no worse than the unfused one), CUDA-graph replay, odd shapes
and the contract checks. On the RTX 5090, RTX PRO 6000 and GB10 with the pinned QuACK the
"quack" provider must be selected; elsewhere "dequant" is the expected outcome, not a skip."""

from __future__ import annotations

import importlib.metadata
import math
import os
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear import _nvfp4_dispatch, _nvfp4_flashinfer, _nvfp4_quack
from ctikernels.linear._nvfp4_dispatch import Small
from ctikernels.linear.nvfp4 import (
    NVFP4Activation,
    NVFP4GateUpSiLU,
    NVFP4Linear,
    NVFP4Weight,
    dequantize_nvfp4,
    quantize_nvfp4,
    unblocked_scale,
)

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Qwen3.5-35B-A3B shared expert, a tile-straddling shape and odd sizes.
GATED_SHAPES = [(2048, 512), (256, 384), (96, 32)]
LINEAR_SHAPES = [(1024, 2048), (2048, 512), (40, 96), (264, 160)]
ROWS = [1, 3, 16, 40, 128, 129, 300]
PRECISION_SEEDS = tuple(range(5))
# Calibration: RTX 5090, RTX PRO 6000, GB10; all LINEAR_SHAPES (N,K),
# GATED_SHAPES (H,I) and ROWS above; 8 seeds (0..7), 1023 shape/M/path/baseline cells.
# 8184 seed records (dev seed sweep); per-baseline maxima below:
# baseline     sample ratio_std     8-seed geomean       log(err_ref) std
# fp32         0.013765778711779142  1.0032030634502809   0.19795510338830785
# fp32-bf16    0.01390673603819425   1.003573075053134    0.16505492466237365
# unfused      0.01390673603819425   1.003573075053134    0.16505492466237365
# Rule: MARGIN = max sample ratio_std across ALL measured cells (one std; no
# tuning to mean excess). All 1023 seeds-0..4 geomeans pass; max 1.0053388740247202.
# MARGIN is unchanged; GPU tests including fp32-bf16 passed on all three devices
# (89 passed each).
MARGIN = 0.01390673603819425


def _assert_precision(ratios, label):
    assert len(ratios) >= 5
    geomean = math.exp(sum(math.log(max(r, 1e-300)) for r in ratios) / len(ratios))
    assert geomean <= 1.0 + MARGIN, (label, geomean, ratios, MARGIN)


# Triton small-batch configurations: (splits, block N, block K, warps, stages, prequantize).
SMALLS = [
    Small(1, 32, 32, 4, 2, False),
    Small(1, 64, 64, 4, 3, True),
    Small(2, 64, 32, 4, 2, False),
    Small(3, 32, 32, 4, 2, True),
    Small(4, 128, 128, 8, 3, False),
]


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _quack_runs() -> bool:
    """Whether QuACK runs here, after asserting why not where it does not."""
    device = _device()
    if _nvfp4_quack.load(device) is not None:
        return True
    hardware = torch.cuda.get_device_name(device), torch.cuda.get_device_capability(device)
    installed = {}
    for name in _nvfp4_quack.DISTRIBUTIONS:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    assert (
        os.environ.get(_nvfp4_quack.VARIABLE) == "0"
        or hardware not in _nvfp4_quack.DEVICES
        or any(installed[name] != version for name, version in _nvfp4_quack.DISTRIBUTIONS.items())
    ), f"QuACK should run on {hardware} with {installed}"
    return False


def _providers() -> list[str]:
    return ["quack", "dequant"] if _quack_runs() else ["dequant"]


# --- FlashInfer oracle, plus an independent GPU helper for FP32 gated values -----------


def reference_quantize(v: torch.Tensor, global_scale: torch.Tensor):
    """GPU bit oracle; CPU uses a tolerance-only approximation for precision tests."""
    from .nvfp4_oracle import cute_quantize, flashinfer_quantize, tolerance_quantize

    if v.dtype == torch.float64:
        v = v.float()  # FP64 references quantize their FP32 rounding, as the kernels would.
    if v.device.type == "cpu":
        return tolerance_quantize(v, global_scale)
    oracle = flashinfer_quantize if v.dtype in (torch.bfloat16, torch.float16) else cute_quantize
    data, scale = oracle(v, global_scale)
    # Weights built from the oracle must satisfy the operators' contiguity contract.
    return data.contiguous(), unblocked_scale(
        scale.view(torch.float8_e4m3fn), *v.shape
    ).contiguous()


def random_weight(rows: int, cols: int, *, std: float = 0.05) -> NVFP4Weight:
    w = torch.randn(rows, cols, device=_device()) * std
    global_scale = (w.abs().amax() / (448.0 * 6.0)).reshape(1)
    data, scale = reference_quantize(w, global_scale)
    return NVFP4Weight(data, scale, global_scale)


def recipe_weight(rows: int, cols: int, recipe: int = 0, *, std: float = 0.05) -> NVFP4Weight:
    """Recipe 0: a quantized Gaussian (``random_weight``). Recipe 1: every E2M1 code, E4M3
    block scales over the full range 2^-9..448 (subnormals included) with 2% zero blocks, and
    a global scale 2^-14..2^-6, as the linear-nvfp4 precision cases draw their recipe-1 weights."""
    if recipe == 0:
        return random_weight(rows, cols, std=std)
    device = _device()
    data = torch.randint(0, 256, (rows, cols // 2), device=device, dtype=torch.int32)
    shape = (rows, cols // 16)
    scale = torch.exp2(torch.rand(shape, device=device) * 17.8 - 9.0)
    scale = torch.where(torch.rand(shape, device=device) < 0.02, torch.zeros_like(scale), scale)
    global_scale = torch.exp2(torch.empty(1, device=device).uniform_(-14.0, -6.0))
    return NVFP4Weight(
        data.to(torch.uint8), scale.clamp(max=448.0).to(torch.float8_e4m3fn), global_scale
    )


RECIPES = [0, 1]


def _activation_values(a: NVFP4Activation) -> torch.Tensor:
    rows, cols = a.shape
    return dequantize_nvfp4(a.data, unblocked_scale(a.scale, rows, cols), a.global_scale)


def _weight_values(w: NVFP4Weight) -> torch.Tensor:
    return dequantize_nvfp4(w.data, w.scale, w.global_scale)


def _activation_fp64(a: NVFP4Activation) -> torch.Tensor:
    # FP64 oracle operand: decode E2M1 * E4M3 (exact), then apply the global in FP64.
    rows, cols = a.shape
    values = dequantize_nvfp4(a.data, unblocked_scale(a.scale, rows, cols)).double()
    return values * a.global_scale.double().reshape(())


def _weight_fp64(w: NVFP4Weight) -> torch.Tensor:
    return dequantize_nvfp4(w.data, w.scale).double() * w.global_scale.double().reshape(())


def _rms(t: torch.Tensor) -> float:
    return float(t.double().square().mean().sqrt())


# --- Quantizer --------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 77, 128, 700, 2050])  # 2050 x 2048: sm120 two-group tiles
@pytest.mark.parametrize("cols", [32, 96, 2048])
def test_quantizer_matches_its_recipe_bit_for_bit(rows: int, cols: int) -> None:
    torch.manual_seed(rows * 31 + cols)
    x = (torch.randn(rows, cols, device=_device()) * 3).to(torch.bfloat16)
    x[0, :16] = 0.0  # an all-zero block
    global_scale = (x.float().abs().amax() / 2688.0 * 0.5).reshape(1)  # half the range clips
    a = quantize_nvfp4(x, global_scale)
    data, scale = reference_quantize(x, global_scale)
    assert a.global_scale is global_scale or torch.equal(a.global_scale, global_scale)
    assert torch.equal(
        unblocked_scale(a.scale, rows, cols).view(torch.uint8), scale.view(torch.uint8)
    )
    # FlashInfer is the byte oracle, including NaNs and signed zeros.
    assert torch.equal(a.data, data)
    padding = a.scale.view(torch.uint8).permute(0, 3, 2, 1, 4).reshape(a.scale.shape[0] * 128, -1)
    padding[:rows, : cols // 16] = 0
    assert not padding.any(), "scale padding must be zero"


def test_quantizer_nonfinite_and_saturation_follow_flashinfer() -> None:
    x = torch.ones(2, 64, device=_device(), dtype=torch.bfloat16)
    x[0, 3] = float("nan")
    x[0, 20] = float("inf")
    x[1, 40] = 1e6
    x[1, :16] = float("nan")
    global_scale = torch.ones(1, device=_device()) / 2688.0
    a = quantize_nvfp4(x, global_scale)
    data, scale = reference_quantize(x, global_scale)
    assert torch.equal(a.data, data)
    assert torch.equal(unblocked_scale(a.scale, 2, 64).view(torch.uint8), scale.view(torch.uint8))


# --- Linear -----------------------------------------------------------------------------


def _linear_errors(op: NVFP4Linear, a: NVFP4Activation, w: NVFP4Weight):
    oracle = _activation_fp64(a) @ _weight_fp64(w).T
    ours = op.quantized(a).double()
    reference = (_activation_values(a) @ _weight_values(w).T).to(torch.bfloat16).double()
    return ours - oracle, reference - oracle, oracle


def _linear_variants(rows: int, n: int, k: int) -> list[tuple[str, object]]:
    """Available providers, including FlashInfer, shared by the FP64 gate and seed sweep."""
    variants = [("dequant", None)]
    if _nvfp4_flashinfer.load(_device()) is not None:
        variants.append(("flashinfer", _nvfp4_dispatch.FlashInfer()))
    if _quack_runs():
        variants.append(("quack", None))
        variants.append(("quack", _nvfp4_quack.DEFAULT))
    return variants


def _cute_variants(rows: int, n: int, k: int) -> list[tuple]:
    """The generic gated CuTe small-M Config wherever it admits the shape (untuned shapes)."""
    plan = _nvfp4_dispatch.Cute()
    if not _nvfp4_dispatch.cute_fits(plan, _device(), rows, n, k, gated=True):
        return []
    return [("cute", _nvfp4_dispatch.CUTE, plan)]


def _run_cute(op, x, plan):
    """CuTe plans quantize floating-point input inside the GEMM, so ``op.quantized`` cannot
    reach them; run ``op(x)``. Its in-kernel quantizer is byte-exact to ``quantize_nvfp4``
    (FlashInfer's recipe; test_nvfp4_cute_small.py asserts the bytes), so the FP64 oracle
    sees the same quantized operands. Asserts dispatch really took the CuTe plan."""
    out = op(x)
    assert op._quantizer == "cute" and isinstance(plan, _nvfp4_dispatch.Cute)
    return out


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: f"recipe{r}")
@pytest.mark.parametrize("shape", LINEAR_SHAPES)
@pytest.mark.parametrize("rows", ROWS)
def test_linear_is_no_worse_than_fp32_against_the_fp64_oracle(shape, rows, recipe) -> None:
    ratios = defaultdict(list)
    for seed in PRECISION_SEEDS:
        for record in precision_linear(shape, rows, seed, check=True, recipe=recipe):
            ratios[record["path"]].append(record["ratio"])
    for path, values in ratios.items():
        _assert_precision(values, (shape, rows, path))


SLOT_GLOBALS = [((256, 256), (0.7, 0.9)), ((256, 128, 128), (0.7, 0.9, 22.4))]


def _slot_weight(slot_rows, globals_):
    w = random_weight(sum(slot_rows), 512)
    return w._replace(
        scale=(w.scale.float() / 8192).to(torch.float8_e4m3fn),
        global_scale=torch.tensor(globals_, dtype=torch.float32, device=_device()),
    )


@pytest.mark.parametrize("slot_rows,globals_", SLOT_GLOBALS)
@pytest.mark.parametrize("rows", [1, 17, 300])
@pytest.mark.parametrize("provider", [None, "dequant"])
def test_slot_globals_merged_gate_up_against_fp64(slot_rows, globals_, rows, provider):
    torch.backends.cuda.matmul.allow_tf32 = False
    quack = _quack_runs()
    ratios = []
    for seed in PRECISION_SEEDS:
        torch.manual_seed(seed)
        w = _slot_weight(slot_rows, globals_)
        x = (torch.randn(rows, 512, device=_device()) * 0.1).bfloat16()
        si = torch.tensor([0.5 / 2688], device=_device())
        a = quantize_nvfp4(x, si)
        av = dequantize_nvfp4(a.data, unblocked_scale(a.scale, *a.shape))
        wv = dequantize_nvfp4(w.data, w.scale)
        columns = torch.cat(
            [g.expand(count) for g, count in zip(w.global_scale, slot_rows, strict=True)]
        )
        oracle = (av.double() @ wv.double().T) * columns.double() * si.double()
        baseline = ((av @ wv.T) * columns * si).bfloat16().double()
        op = NVFP4Linear(w, si, slot_rows=slot_rows, provider=provider)
        assert op.provider == ("quack" if provider is None and quack else "dequant")
        actual = op(x)
        torch.testing.assert_close(actual, op.quantized(a), rtol=0, atol=0)
        error = actual.double() - oracle
        bound = oracle.abs() * 2.0**-8 + oracle.abs().max() * 2.0**-18 + 1e-30
        assert (error.abs() <= bound).all(), float((error.abs() / bound).max())
        ratios.append(_rms(error) / max(_rms(baseline - oracle), 1e-300))
    _assert_precision(ratios, (slot_rows, globals_, rows, provider))


@pytest.mark.parametrize("slot_rows,globals_", SLOT_GLOBALS)
@pytest.mark.parametrize("rows", [1, 64, 300])
def test_slot_globals_match_per_slot_linears_bit_for_bit(slot_rows, globals_, rows):
    if not _quack_runs():
        pytest.skip("QuACK unavailable")
    torch.manual_seed(72)
    w = _slot_weight(slot_rows, globals_)
    si = torch.tensor([0.5 / 2688], device=_device())
    x = (torch.randn(rows, 512, device=_device()) * 0.1).bfloat16()
    a = quantize_nvfp4(x, si)
    merged = NVFP4Linear(w, si, slot_rows=slot_rows, provider="quack")
    outputs, start = [], 0
    for slot, count in enumerate(slot_rows):
        part = NVFP4Weight(
            w.data[start : start + count],
            w.scale[start : start + count],
            w.global_scale[slot : slot + 1],
        )
        op = NVFP4Linear(part, si, provider="quack")
        assert op.plan_for(rows) == merged.plan_for(rows) == _nvfp4_quack.DEFAULT
        outputs.append(op.quantized(a))
        start += count
    torch.testing.assert_close(merged.quantized(a), torch.cat(outputs, dim=1), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(512, 512), (5120, 17408)])
def test_uniform_slots_are_bitwise_unchanged(shape):
    torch.manual_seed(73)
    n, k = shape
    w = random_weight(n, k)
    g = w.global_scale
    si = torch.tensor([4 / 2688], device=_device())
    ops = [
        NVFP4Linear(w._replace(global_scale=global_), si, slot_rows=slots)
        for global_, slots in (
            (g.reshape(()), None),
            (g, (n,)),
            (g.expand(3), (n // 2, n // 4, n // 4)),
        )
    ]
    for op in ops[1:]:
        assert op.provider == ops[0].provider
        for rows in (1, 16, 128, 4096):
            assert op.plan_for(rows) == ops[0].plan_for(rows)
    for rows in (1, 16, 128):
        x = torch.randn(rows, k, device=_device(), dtype=torch.bfloat16)
        a = quantize_nvfp4(x, si)
        eager, quantized = ops[0](x), ops[0].quantized(a)
        for op in ops[1:]:
            torch.testing.assert_close(op(x), eager, rtol=0, atol=0)
            torch.testing.assert_close(op.quantized(a), quantized, rtol=0, atol=0)


def test_slot_globals_one_gemm_launch():
    if not _quack_runs():
        pytest.skip("QuACK unavailable")
    from benchmark.kernel_tuning.nvfp4_parity import kernel_names, kernels_per_call

    w = _slot_weight((256, 256), (0.7, 0.9))
    si = torch.tensor([4 / 2688], device=_device())
    op = NVFP4Linear(w, si, slot_rows=(256, 256))
    assert op.provider == "quack"
    x = torch.randn(128, 512, device=_device(), dtype=torch.bfloat16)
    assert kernels_per_call(op, x) <= 3
    names = kernel_names(op, x)
    assert len(names) == 3, names
    assert sum("gemm" in name.lower() for name in names) == 1, names


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("provider", [None, "dequant"])
def test_slot_globals_graph_replay_reads_in_place_updates(quantized, provider):
    if provider is None and not _quack_runs():
        pytest.skip("QuACK unavailable")
    torch.manual_seed(74)
    w = _slot_weight((256, 256), (0.7, 0.9))
    si = torch.tensor([0.5 / 2688], device=_device())
    x = (torch.randn(17, 512, device=_device()) * 0.1).bfloat16()
    a = quantize_nvfp4(x, si)
    op = NVFP4Linear(w, si, slot_rows=(256, 256), provider=provider)
    call = lambda op: op.quantized(a) if quantized else op(x)
    for _ in range(3):
        before = call(op)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call(op)
    w.global_scale.mul_(torch.tensor([2.0, 0.5], device=_device()))
    graph.replay()
    eager = call(op)
    assert not torch.equal(before, eager)
    fresh = NVFP4Linear(
        w._replace(global_scale=w.global_scale.clone()), si, slot_rows=(256, 256), provider=provider
    )
    torch.testing.assert_close(result, eager, rtol=0, atol=0)
    torch.testing.assert_close(result, call(fresh), rtol=0, atol=0)


def precision_linear(shape, rows, seed, *, check=False, variants=None, recipe=0):
    """Gate 2 and seed sweep, including FlashInfer; RMS versus identical FP64 operands."""
    torch.backends.cuda.matmul.allow_tf32 = False
    n, k = shape
    torch.manual_seed(seed)
    w = recipe_weight(n, k, recipe)
    x = torch.randn(rows, k, device=_device(), dtype=torch.bfloat16)
    input_scale = (x.float().abs().amax() / 2688.0).reshape(1)
    a = quantize_nvfp4(x, input_scale)
    # Compute the expensive FP64 oracle once per seed, not once per provider.
    av, wv = _activation_values(a), _weight_values(w)
    oracle = _activation_fp64(a) @ _weight_fp64(w).T
    reference_error = _rms((av @ wv.T).to(torch.bfloat16).double() - oracle)
    for provider, plan in _linear_variants(rows, n, k) if variants is None else variants:
        op = NVFP4Linear(w, input_scale, provider=provider)
        if plan is not None:
            op.plan_for = lambda count, plan=plan, **_: plan
        ours = op.quantized(a).double() - oracle
        error = _rms(ours)
        label = f"{provider}:{plan}"
        if check:
            bound = oracle.abs() * 2.0**-8 + oracle.abs().max() * 2.0**-18 + 1e-30
            assert (ours.abs() <= bound).all(), (label, float((ours.abs() / bound).max()))
            torch.testing.assert_close(op(x), op.quantized(a), rtol=0, atol=0)
        yield {
            "path": label,
            "baseline": "fp32",
            "err_path": error,
            "err_ref": reference_error,
            "ratio": error / max(reference_error, 1e-300),
        }


def test_linear_out_and_fp16_and_empty_batches() -> None:
    w = random_weight(64, 128)
    scale = torch.full((1,), 1.0 / 2688.0, device=_device())
    for out_dtype in (torch.bfloat16, torch.float16):
        op = NVFP4Linear(w, scale, out_dtype=out_dtype)
        x = torch.randn(5, 128, device=_device(), dtype=torch.float16)
        out = torch.empty(5, 64, device=_device(), dtype=out_dtype)
        assert op(x, out=out) is out
        torch.testing.assert_close(out, op(x), rtol=0, atol=0)
        assert op(x[:0]).shape == (0, 64)
    # A strided view of wider rows.
    wide = torch.randn(7, 256, device=_device(), dtype=torch.bfloat16)
    torch.testing.assert_close(op(wide[:, :128]), op(wide[:, :128].contiguous()), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 8, 16])
def test_unaligned_contiguous_outputs_at_27b(rows) -> None:
    """Contiguous output views with storage offset 1 (2 / 1 bytes) are valid outputs; the
    dispatched provider (CuTe small-M on sm120 at M<=16) must not reject them."""
    torch.manual_seed(rows)
    w = random_weight(5120, 17408)
    scale = torch.full((1,), 1.0 / 2688.0, device=_device())
    op = NVFP4Linear(w, scale)
    x = torch.randn(rows, 17408, device=_device(), dtype=torch.bfloat16)
    out = torch.empty(rows * 5120 + 1, device=_device(), dtype=torch.bfloat16)[1:].view(rows, 5120)
    assert out.is_contiguous() and out.data_ptr() % 16
    assert op(x, out=out) is out
    torch.testing.assert_close(out, op(x), rtol=0, atol=0)
    a = quantize_nvfp4(x, scale)
    assert op.quantized(a, out=out) is out
    torch.testing.assert_close(out, op.quantized(a), rtol=0, atol=0)

    gate, up = random_weight(17408, 5120), random_weight(17408, 5120)
    gated = NVFP4GateUpSiLU(gate, up, input_scale=scale, output_scale=scale * 4)
    x = torch.randn(rows, 5120, device=_device(), dtype=torch.bfloat16)
    expected = gated(x)
    shape = tuple(expected.scale.shape)
    storage = torch.empty(math.prod(shape) + 1, dtype=torch.float8_e4m3fn, device=_device())
    out = NVFP4Activation(
        torch.empty_like(expected.data), storage[1:].view(shape), torch.zeros(1, device=_device())
    )
    result = gated(x, out=out)
    assert result.data is out.data and result.scale is out.scale
    torch.testing.assert_close(result.data, expected.data, rtol=0, atol=0)
    torch.testing.assert_close(
        _activation_values(result), _activation_values(expected), rtol=0, atol=0
    )


def test_expanded_zero_row_stride_inputs_at_27b(monkeypatch) -> None:
    """Expanded inputs (row stride 0) are valid; CuTe small-M rejects them in eligibility,
    so dispatch routes them to a provider that accepts them, with no raise. The expected
    output runs the contiguous copy through that same non-CuTe route."""
    from ctikernels.linear import _nvfp4_cute_small

    torch.manual_seed(0)
    scale = torch.full((1,), 1.0 / 2688.0, device=_device())
    op = NVFP4Linear(random_weight(5120, 17408), scale)
    gated = NVFP4GateUpSiLU(
        random_weight(17408, 5120), random_weight(17408, 5120),
        input_scale=scale, output_scale=scale * 4,
    )  # fmt: skip
    x = torch.randn(1, 17408, device=_device(), dtype=torch.bfloat16).expand(2, 17408)
    h = torch.randn(1, 5120, device=_device(), dtype=torch.bfloat16).expand(2, 5120)
    assert x.stride(0) == h.stride(0) == 0
    linear, result = op(x), gated(h)
    torch.testing.assert_close(linear[0], linear[1], rtol=0, atol=0)
    monkeypatch.setattr(_nvfp4_cute_small, "input_aligned", lambda _: False)
    torch.testing.assert_close(linear, op(x.contiguous()), rtol=0, atol=0)
    expected = gated(h.contiguous())
    torch.testing.assert_close(result.data, expected.data, rtol=0, atol=0)
    torch.testing.assert_close(
        _activation_values(result), _activation_values(expected), rtol=0, atol=0
    )


def test_linear_rejects_bad_inputs_before_launch() -> None:
    w = random_weight(64, 128)
    scale = torch.full((1,), 1.0 / 2688.0, device=_device())
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(NVFP4Weight(w.data, w.scale, w.global_scale.clone().requires_grad_()), scale)
    with pytest.raises(ValueError, match="frozen"):
        NVFP4Linear(w, scale.clone().requires_grad_())
    with pytest.raises(ValueError):
        NVFP4Linear(NVFP4Weight(w.data[:, :24], w.scale[:, :6], w.global_scale), scale)
    with pytest.raises(TypeError):
        NVFP4Linear(NVFP4Weight(w.data, w.scale.float(), w.global_scale), scale)
    op = NVFP4Linear(w, scale)
    with pytest.raises(ValueError):
        op(torch.randn(3, 64, device=_device(), dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        op(torch.randn(3, 128, device=_device()))
    with pytest.raises(RuntimeError, match="autograd"):
        op(torch.randn(3, 128, device=_device(), dtype=torch.bfloat16, requires_grad=True))


# --- Gated projection -------------------------------------------------------------------


def _gated_case(
    h: int, i: int, rows: int, *, equal_scales: bool = False, seed: int | None = None, recipe=0
):
    torch.manual_seed(h * 7 + i + rows if seed is None else seed)
    gate, up = recipe_weight(i, h, recipe), recipe_weight(i, h, recipe, std=0.08)
    if equal_scales:
        up = random_weight(i, h, std=0.05)
        up = NVFP4Weight(up.data, up.scale, gate.global_scale)
    x = torch.randn(rows, h, device=_device(), dtype=torch.bfloat16)
    input_scale = (x.float().abs().amax() / 2688.0).reshape(1)
    a = quantize_nvfp4(x, input_scale)
    xd = _activation_fp64(a)
    g = xd @ _weight_fp64(gate).T
    u = xd @ _weight_fp64(up).T
    oracle = F.silu(g) * u
    output_scale = (oracle.abs().amax().float() / 2688.0 * 0.9).reshape(1)
    # Two independent harness references: the fused FP32 epilogue and the dense
    # SGLang-style BF16 projection boundary, both followed by the FP32 quantizer.
    g32 = _activation_values(a) @ _weight_values(gate).T
    u32 = _activation_values(a) @ _weight_values(up).T
    references = _gated_references(g32, u32, output_scale)
    return gate, up, x, a, input_scale, output_scale, oracle, references


def _gated_references(g32, u32, output_scale):
    references = {}
    for name, g, u in (
        ("fp32", g32, u32),
        ("fp32-bf16", g32.to(torch.bfloat16).float(), u32.to(torch.bfloat16).float()),
    ):
        data, scale = reference_quantize(F.silu(g) * u, output_scale)
        references[name] = dequantize_nvfp4(data, scale, output_scale).double()
    return references


def test_gated_precision_reference_preserves_the_bf16_projection_boundary() -> None:
    generator = torch.Generator().manual_seed(0)
    gate = torch.randn(8, 32, generator=generator)
    up = torch.randn(8, 32, generator=generator)
    scale = torch.tensor([0.001])
    references = _gated_references(gate, up, scale)
    rounded = _gated_references(gate.bfloat16().float(), up.bfloat16().float(), scale)
    # Rounding before the activation must affect the quantized result, and the
    # BF16-boundary reference must agree with FP32 applied to already-rounded inputs.
    assert not torch.equal(references["fp32"], references["fp32-bf16"])
    torch.testing.assert_close(references["fp32-bf16"], rounded["fp32"], rtol=0, atol=0)
    torch.testing.assert_close(rounded["fp32-bf16"], rounded["fp32"], rtol=0, atol=0)


def _gated_variants(rows: int, h: int, i: int) -> list[tuple[str, str, object]]:
    """Available paths for gate 2 and the sweep, including FlashInfer's unfused GEMM."""
    variants = [("dequant", "dequant", None)]
    if _nvfp4_flashinfer.load(_device()) is not None:
        variants.append(("flashinfer", _nvfp4_dispatch.UNFUSED, _nvfp4_dispatch.FlashInfer()))
    if _quack_runs():
        variants += [
            ("quack", path, _nvfp4_quack.DEFAULT) for path in ("quack-fused", "quack-unfused")
        ]
        variants += [
            ("quack", "triton", small)
            for small in SMALLS
            if _nvfp4_dispatch.small_fits(small, rows, 2 * i, h, gated=True)
        ]
    return variants + _cute_variants(rows, 2 * i, h)


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: f"recipe{r}")
@pytest.mark.parametrize("shape", GATED_SHAPES)
@pytest.mark.parametrize("rows", ROWS)
def test_gated_paths_are_no_worse_than_fp32_against_the_fp64_oracle(shape, rows, recipe) -> None:
    ratios = defaultdict(list)
    for seed in PRECISION_SEEDS:
        for record in precision_gated(shape, rows, seed, check=True, recipe=recipe):
            ratios[(record["path"], record["baseline"])].append(record["ratio"])
    for path, values in ratios.items():
        _assert_precision(values, (shape, rows, path))


def precision_gated(shape, rows, seed, *, check=False, variants=None, recipe=0):
    """All available providers share the FP64 oracle and unchanged seed-noise margin."""
    torch.backends.cuda.matmul.allow_tf32 = False
    h, i = shape
    gate, up, x, a, input_scale, output_scale, oracle, references = _gated_case(
        h, i, rows, seed=seed, recipe=recipe
    )
    reference_errors = {name: _rms(value - oracle) for name, value in references.items()}
    errors = {}
    for provider, path, plan in _gated_variants(rows, h, i) if variants is None else variants:
        op = NVFP4GateUpSiLU(
            gate, up, input_scale=input_scale, output_scale=output_scale, provider=provider
        )
        cute = isinstance(plan, _nvfp4_dispatch.Cute)
        if plan is not None:
            op.plan_for = lambda count, path=path, plan=plan, **_: (path, plan)
        out = _run_cute(op, x, plan) if cute else op.quantized(a)
        errors[(path, plan)] = _rms(_activation_values(out).double() - oracle)
        if check:
            assert out.global_scale is op.output_scale
            _assert_scale_padding_is_zero(out)
            if not cute:
                torch.testing.assert_close(
                    _activation_values(op(x)), _activation_values(out), rtol=0, atol=0
                )
    unfused = errors.get(("quack-unfused", _nvfp4_quack.DEFAULT))
    for (path, plan), error in errors.items():
        baselines = {"fp32": reference_errors["fp32"]}
        if isinstance(plan, _nvfp4_dispatch.FlashInfer):
            baselines["fp32-bf16"] = reference_errors["fp32-bf16"]
        if path in ("quack-fused", "triton", _nvfp4_dispatch.CUTE):
            # Fused paths feed FP32 accumulators to SiLU*up without the BF16 rounding;
            # fp32 alone checks the same mathematical formula as the fused kernel;
            # this separate rounded formula enforces the reference-path owner gate.
            baselines["fp32-bf16"] = reference_errors["fp32-bf16"]
            if unfused is not None:
                baselines["unfused"] = unfused
        for baseline, ref_error in baselines.items():
            yield {
                "path": f"{path}:{plan}",
                "baseline": baseline,
                "err_path": error,
                "err_ref": ref_error,
                "ratio": error / max(ref_error, 1e-300),
            }


# Linear has no table (FlashInfer for every M): the model shapes its buckets are checked at.
FI_LINEAR_SHAPES = ((2048, 512), (4096, 12288), (5120, 17408))


def _flashinfer_dispatch_cases():
    """Union of FlashInfer bucket boundaries across ALL devices, independent of this GPU.

    Dispatch rounds M up to a power of two. Test the first and last row of each
    selected bucket, plus one bucket beyond the table for its last-entry fallback.
    Deduplicate identical cases across devices to keep the five-seed oracle affordable.
    """
    cases = set()
    linear = {"any": {shape: ((4096, _nvfp4_dispatch.FlashInfer()),) for shape in FI_LINEAR_SHAPES}}
    for operation, table in (("linear", linear), ("gated", _nvfp4_dispatch.GATED)):
        for shapes in table.values():
            for shape, entries in shapes.items():
                last_bucket = 1 << max(entries[-1][0] - 1, 0).bit_length()
                lower, upper = 1, 1
                while upper <= 2 * last_bucket:
                    entry = _nvfp4_dispatch._entry(entries, upper)
                    if isinstance(entry[-1], _nvfp4_dispatch.FlashInfer):
                        path = entry[1] if operation == "gated" else "linear"
                        for rows in {lower, upper}:
                            cases.add((operation, shape, rows, path))
                    lower, upper = upper + 1, upper * 2
    return sorted(cases)


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: f"recipe{r}")
@pytest.mark.parametrize("operation,shape,rows,path", _flashinfer_dispatch_cases())
def test_flashinfer_dispatch_buckets_are_no_worse_than_fp32_against_the_fp64_oracle(
    operation, shape, rows, path, recipe
) -> None:
    """Table-selected shapes/M use the same five seeds, RMS gate and check=True bounds."""
    _require_flashinfer()
    plan = _nvfp4_dispatch.FlashInfer()
    if operation == "linear":
        precision = precision_linear
        variants = [("flashinfer", plan)]
    else:
        assert path == _nvfp4_dispatch.UNFUSED
        precision = precision_gated
        variants = [("flashinfer", path, plan)]
    ratios = defaultdict(list)
    for seed in PRECISION_SEEDS:
        for record in precision(shape, rows, seed, check=True, variants=variants, recipe=recipe):
            ratios[(record["path"], record["baseline"])].append(record["ratio"])
    assert ratios
    for label, values in ratios.items():
        _assert_precision(values, (operation, shape, rows, label))


def _cute_dispatch_cases():
    """Every CuTe plan the gated table selects (union across devices), at the first and last
    row of its bucket, so the dispatched small-M path meets the five-seed gate."""
    cases = set()
    for shapes in _nvfp4_dispatch.GATED.values():
        for shape, entries in shapes.items():
            lower = 1
            for entry in entries:
                if isinstance(entry[-1], _nvfp4_dispatch.Cute):
                    for rows in {lower, entry[0]}:
                        cases.add((shape, rows, entry[-1]))
                lower = entry[0] + 1
    return sorted(cases, key=repr)


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: f"recipe{r}")
@pytest.mark.parametrize("shape,rows,plan", _cute_dispatch_cases(), ids=repr)
def test_cute_dispatch_buckets_are_no_worse_than_fp32_against_the_fp64_oracle(
    shape, rows, plan, recipe
) -> None:
    """Five seeds, the measured-noise MARGIN and check=True bounds, like the other providers;
    also compared with the unfused path where QuACK runs."""
    from ctikernels.linear import _nvfp4_cute_small

    if not _nvfp4_cute_small.available(_device()):
        pytest.skip("requires DSL 4.8.0 and RTX 5090 / RTX PRO 6000 / GB10")
    assert _nvfp4_dispatch.cute_fits(plan, _device(), rows, 2 * shape[1], shape[0], gated=True)
    variants = [("cute", _nvfp4_dispatch.CUTE, plan)]
    if _quack_runs():
        variants.append(("quack", "quack-unfused", _nvfp4_quack.DEFAULT))
    ratios = defaultdict(list)
    for seed in PRECISION_SEEDS:
        for record in precision_gated(
            shape, rows, seed, check=True, variants=variants, recipe=recipe
        ):
            ratios[(record["path"], record["baseline"])].append(record["ratio"])
    assert any(path.startswith("cute:") for path, _ in ratios)
    if _quack_runs():
        assert any(baseline == "unfused" for _, baseline in ratios)
    for label, values in ratios.items():
        _assert_precision(values, ("gated", shape, rows, label))


def _assert_scale_padding_is_zero(a: NVFP4Activation) -> None:
    """Scales of column blocks past the tensor are zero (rows past it are unspecified)."""
    rows, cols = a.shape
    grid = a.scale.view(torch.uint8).permute(0, 3, 2, 1, 4).reshape(a.scale.shape[0] * 128, -1)
    assert not grid[:rows, cols // 16 :].any()


@pytest.mark.parametrize("rows", [1, 300])
def test_gated_scales_follow_the_recipe(rows: int) -> None:
    """Fused output scales equal the recipe's applied to the FP64 oracle, but for blocks whose
    amax sits within an accumulation error of an E4M3 rounding boundary (one code apart)."""
    if not _quack_runs():
        pytest.skip("QuACK does not run here (asserted by _quack_runs)")
    gate, up, x, a, input_scale, output_scale, oracle, _ = _gated_case(2048, 512, rows)
    op = NVFP4GateUpSiLU(gate, up, input_scale=input_scale, output_scale=output_scale)
    out = op.quantized(a)
    _, expected = reference_quantize(oracle.float(), output_scale)
    got = unblocked_scale(out.scale, rows, 512).view(torch.uint8).int()
    difference = (got - expected.view(torch.uint8).int()).abs()
    assert difference.max() <= 1
    assert (difference == 0).float().mean() >= 0.995


def test_gated_with_equal_global_scales_and_out_buffers() -> None:
    for provider in _providers():
        gate, up, x, a, input_scale, output_scale, oracle, reference = _gated_case(
            256, 384, 33, equal_scales=True
        )
        op = NVFP4GateUpSiLU(
            gate, up, input_scale=input_scale, output_scale=output_scale, provider=provider
        )
        out = NVFP4Activation(
            torch.empty(33, 192, dtype=torch.uint8, device=_device()),
            torch.empty(1, 6, 32, 4, 4, dtype=torch.float8_e4m3fn, device=_device()),
            torch.zeros(1, device=_device()),
        )
        result = op(x, out=out)
        assert result.data is out.data and result.scale is out.scale
        torch.testing.assert_close(
            _activation_values(result), _activation_values(op(x)), rtol=0, atol=0
        )
        empty = op(x[:0])
        assert empty.data.shape == (0, 192)


def test_down_projection_consumes_the_gated_output() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    gate, up, x, a, input_scale, output_scale, oracle, _ = _gated_case(2048, 512, 9)
    fc1 = NVFP4GateUpSiLU(gate, up, input_scale=input_scale, output_scale=output_scale)
    weight = random_weight(2048, 512)
    down = NVFP4Linear(weight, output_scale)
    h = fc1(x)
    expected = _activation_fp64(h) @ _weight_fp64(weight).T
    error = down.quantized(h).double() - expected
    assert (error.abs() <= expected.abs() * 2.0**-8 + expected.abs().max() * 2.0**-18).all()


# --- CUDA graphs ------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 200])
def test_graph_replay_follows_new_inputs(rows: int) -> None:
    for provider in _providers():
        gate, up, x, a, input_scale, output_scale, _, _ = _gated_case(256, 384, rows)
        fc1 = NVFP4GateUpSiLU(
            gate, up, input_scale=input_scale, output_scale=output_scale, provider=provider
        )
        down = NVFP4Linear(random_weight(256, 384), output_scale, provider=provider)
        static = x.clone()
        expected = down.quantized(fc1(static))  # compiles eagerly
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = down.quantized(fc1(static))
        graph.replay()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        fresh = torch.randn_like(x)
        static.copy_(fresh)
        graph.replay()
        torch.testing.assert_close(result, down.quantized(fc1(fresh)), rtol=0, atol=0)


# --- One shared interleaved gate|up copy ------------------------------------------------


def _interleaved(gate: NVFP4Weight, up: NVFP4Weight):
    i = gate.data.shape[0]
    data = torch.stack((gate.data, up.data), dim=1).view(2 * i, -1)
    return data, torch.stack((gate.scale, up.scale), dim=1).view(2 * i, -1)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("blocked", [False, True], ids=["grid", "blocked"])
def test_from_interleaved_matches_gate_up_bit_for_bit(shared, blocked) -> None:
    from ctikernels.linear.nvfp4 import blocked_scale

    h, i, rows = 256, 384, 3
    gate, up, x, a, si, so, _, _ = _gated_case(h, i, rows, equal_scales=shared)
    data, grid = _interleaved(gate, up)
    scale = blocked_scale(grid) if blocked else grid
    for provider, path, plan in _gated_variants(rows, h, i):
        ops = [
            NVFP4GateUpSiLU(gate, up, input_scale=si, output_scale=so, provider=provider),
            NVFP4GateUpSiLU.from_interleaved(
                data,
                scale,
                gate.global_scale,
                up.global_scale,
                input_scale=si,
                output_scale=so,
                provider=provider,
            ),
        ]
        if plan is not None:
            for op in ops:
                op.plan_for = lambda count, path=path, plan=plan, **_: (path, plan)
        results = [op(x) for op in ops]
        if not isinstance(plan, _nvfp4_dispatch.Cute):
            results += [op.quantized(a) for op in ops]
        for result, expected in zip(results[1::2], results[::2], strict=True):
            assert torch.equal(result.data, expected.data), (provider, path, plan)
            assert torch.equal(result.scale.view(torch.uint8), expected.scale.view(torch.uint8))
        held = ops[1]
        if held._weight is not None and not isinstance(held._weight, torch.Tensor):
            assert held._weight[0].data_ptr() == data.data_ptr()  # read in place, no copy


@pytest.mark.parametrize("which", ["scale", "data"])
def test_from_interleaved_copies_unaligned_storage(which) -> None:
    """Storage-offset-1 views (not 16-byte aligned) get one aligned copy at construction;
    the FlashInfer preparation would otherwise hand the pointer to CUTLASS TMA."""
    from ctikernels.linear.nvfp4 import blocked_scale

    h, i, rows = 256, 384, 3
    gate, up, x, _, si, so, _, _ = _gated_case(h, i, rows)
    data, grid = _interleaved(gate, up)
    scale = blocked_scale(grid)
    if which == "scale":
        storage = torch.empty(scale.numel() + 1, dtype=scale.dtype, device=scale.device)
        storage[1:].copy_(scale.flatten())
        scale = storage[1:].view(scale.shape)
    else:
        storage = torch.empty(data.numel() + 1, dtype=data.dtype, device=data.device)
        storage[1:].copy_(data.flatten())
        data = storage[1:].view(data.shape)
    assert (scale if which == "scale" else data).data_ptr() % 16
    for provider, path, plan in _gated_variants(rows, h, i):
        ops = [
            NVFP4GateUpSiLU(gate, up, input_scale=si, output_scale=so, provider=provider),
            NVFP4GateUpSiLU.from_interleaved(
                data,
                scale,
                gate.global_scale,
                up.global_scale,
                input_scale=si,
                output_scale=so,
                provider=provider,
            ),
        ]
        if plan is not None:
            for op in ops:
                op.plan_for = lambda count, path=path, plan=plan, **_: (path, plan)
        for tensor in ops[1]._interleaved:
            assert tensor.data_ptr() % 16 == 0, (provider, path, plan)
        expected, result = (op(x) for op in ops)
        assert torch.equal(result.data, expected.data), (provider, path, plan)
        assert torch.equal(result.scale.view(torch.uint8), expected.scale.view(torch.uint8))


def test_from_interleaved_graph_replay_reads_updated_global_scales() -> None:
    h, i, rows = 256, 384, 3
    gate, up, x, _, si, so, _, _ = _gated_case(h, i, rows)
    data, grid = _interleaved(gate, up)
    for provider, path, plan in _gated_variants(rows, h, i):
        op = NVFP4GateUpSiLU.from_interleaved(
            data,
            grid,
            gate.global_scale,
            up.global_scale,
            input_scale=si,
            output_scale=so,
            provider=provider,
        )
        if plan is not None:
            op.plan_for = lambda count, path=path, plan=plan, **_: (path, plan)
        out = op(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, out=out)
        for scale in (gate.global_scale, up.global_scale, si, so):
            scale.mul_(1.37)
            graph.replay()
            expected = op(x)
            assert torch.equal(out.data, expected.data), (provider, path, plan)
            assert torch.equal(out.scale.view(torch.uint8), expected.scale.view(torch.uint8))
            scale.div_(1.37)


@pytest.mark.parametrize("shared", [False, True])
def test_gated_weight_storage_is_one_interleaved_copy_at_27b(shared) -> None:
    """Qwen3.8-27B TP1 gate_up (H=5120, I=17408) through the device's dispatched providers:
    after the checkpoint tensors are dropped, the op keeps one interleaved gate|up copy (plus
    CuTe small-M's scale re-layout where its tables use it); engine-held storage adds no
    weight copy. FlashInfer with independent globals is never dispatched (two GEMMs)."""
    from ctikernels.linear.nvfp4 import blocked_scale

    h, i = 5120, 17408
    device = _device()
    data_bytes, scale_bytes = 2 * i * h // 2, 2 * i * h // 16
    copy = data_bytes + scale_bytes
    # CuTe sectors ([2I/8, H/64, 8, 4] bytes) plus per-op state under 2 MiB (per-column FP32
    # global factors, split-K counters, scalars).
    extra = scale_bytes + (2 << 20)
    one = torch.ones(1, device=device)

    def make():
        torch.manual_seed(7)
        weights = []
        for _ in range(2):
            weights.append(
                (
                    torch.randint(0, 256, (i, h // 2), dtype=torch.uint8, device=device),
                    torch.rand(i, h // 16, device=device).to(torch.float8_e4m3fn),
                )
            )
        return weights

    gate_global = one * 0.7
    up_global = gate_global if shared else one * 0.9
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    (gd, gs), (ud, us) = make()
    op = NVFP4GateUpSiLU(
        NVFP4Weight(gd, gs, gate_global),
        NVFP4Weight(ud, us, up_global),
        input_scale=one,
        output_scale=one,
    )
    del gd, gs, ud, us
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base
    assert retained <= copy + extra, (op.provider, retained / copy)
    data, scale = op._interleaved
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    held = NVFP4GateUpSiLU.from_interleaved(
        data, scale, gate_global, up_global, input_scale=one, output_scale=one
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before <= extra, held.provider
    del held
    grid = unblocked_scale(scale, 2 * i, h).contiguous()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    held = NVFP4GateUpSiLU.from_interleaved(
        data, grid, gate_global, up_global, input_scale=one, output_scale=one
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before <= extra + blocked_scale(grid).numel()
    assert held.provider == op.provider


def test_quack_is_selected_on_qualified_devices() -> None:
    runs = _quack_runs()
    w = random_weight(64, 128)
    scale = torch.full((1,), 1.0 / 2688.0, device=_device())
    if runs:
        assert NVFP4Linear(w, scale, provider="quack").provider == "quack"
    if not runs:
        with pytest.raises(RuntimeError):
            NVFP4Linear(w, scale, provider="quack")


@pytest.mark.parametrize("width", [32, 384])
def test_prewarm_makes_unseen_batch_sizes_capturable(monkeypatch, width) -> None:
    from ctikernels.linear import _nvfp4_cute_quant

    # Isolate the cache so earlier tests cannot hide missing prewarm specializations.
    monkeypatch.setattr(_nvfp4_cute_quant, "_launchers", {})
    gate, up, x, a, input_scale, output_scale, _, _ = _gated_case(256, width, 1)
    fc1 = NVFP4GateUpSiLU(gate, up, input_scale=input_scale, output_scale=output_scale)
    down = NVFP4Linear(random_weight(256, width), output_scale)
    fc1.prewarm()
    down.prewarm()
    for rows in (5, 37, 700):
        static = torch.randn(rows, 256, device=_device(), dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = down.quantized(fc1(static))
        graph.replay()
        torch.testing.assert_close(result, down.quantized(fc1(static)), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gated-input", "gated-output", "linear"])
def test_prewarm_captures_in_bucket_quantizer_geometry(monkeypatch, kind):
    from ctikernels.linear import _nvfp4_cute_quant

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("the in-bucket load-group transition requires sm120")
    if not _quack_runs():
        pytest.skip("requires the qualified QuACK provider")
    monkeypatch.setattr(_nvfp4_cute_quant, "_launchers", {})
    quantizer = _cute_quantizer(monkeypatch)
    hidden, width = (32, 5120) if kind == "gated-output" else (5120, 32)
    input_scale = torch.tensor([0.01], device=_device())
    output_scale = torch.tensor([0.1], device=_device())
    if kind == "linear":
        op = NVFP4Linear(random_weight(width, hidden), input_scale, provider="quack")
    else:
        op = NVFP4GateUpSiLU(
            random_weight(width, hidden),
            random_weight(width, hidden),
            input_scale=input_scale,
            output_scale=output_scale,
            provider="quack",
        )
        if kind == "gated-output":
            # Exercise the separate gated quantizer after the BF16 projection.
            monkeypatch.setattr(
                op, "plan_for", lambda rows, **_: ("quack-unfused", _nvfp4_quack.DEFAULT)
            )
        else:
            assert op.provider_for(780) == "quack-fused"
    op.prewarm()
    warmed = set(quantizer.compiled)
    # 769..781 share padding bucket 7 with M=896 but use one load group, not two.
    # Capture before any eager call at these rows can populate the missing key.
    for rows in (780, 782):
        x = torch.randn(rows, hidden, device=_device(), dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = op(x)
        graph.replay()
        expected = op(x)
        if kind == "linear":
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        else:
            assert torch.equal(result.data, expected.data)
            # Padding rows are unspecified; compare only live scale entries.
            torch.testing.assert_close(
                unblocked_scale(result.scale, rows, width).float(),
                unblocked_scale(expected.scale, rows, width).float(),
                rtol=0,
                atol=0,
            )
        assert set(quantizer.compiled) == warmed


# Q4: Linear/gated shapes with quantizer thresholds above 4096 rows (K or I = 512: two
# load groups from 7813 rows on sm120, the early PDL trigger from 11719), a gated
# intermediate size that is not a multiple of 64 (the padding clear grows per 128 rows)
# and the Qwen3.8-27B TP1 down / gate_up shapes.
_PREWARM_MAX = 16384


@pytest.mark.parametrize(
    "kind,hidden,width",
    [
        ("linear", 512, 1024),
        ("gated", 512, 96),
        ("gated", 2048, 512),
        ("linear", 17408, 5120),
        ("gated", 5120, 17408),
    ],
)
def test_prewarm_max_rows_capture_compiles_and_loads_nothing(monkeypatch, kind, hidden, width):
    import cutlass.cute as cute

    from ctikernels.linear import _nvfp4_cute_quant

    if not _quack_runs():
        pytest.skip("requires the qualified QuACK provider")
    monkeypatch.setattr(_nvfp4_cute_quant, "_launchers", {})
    torch.manual_seed(hidden + width)
    input_scale = torch.tensor([0.01], device=_device())
    if kind == "linear":
        op = NVFP4Linear(random_weight(width, hidden), input_scale)
    else:
        op = NVFP4GateUpSiLU(
            random_weight(width, hidden),
            random_weight(width, hidden, std=0.08),
            input_scale=input_scale,
            output_scale=torch.tensor([0.1], device=_device()),
        )
    routes = op.prewarm(_PREWARM_MAX)
    counts = [count for count, _ in routes]
    assert len(counts) == len(set(counts)) and max(counts) <= _PREWARM_MAX
    assert all(isinstance(route, str) for _, route in routes)
    # Every row where a specialization may change (first and last row of each
    # interval), each power-of-two bucket edge and several 128-row padding buckets.
    edges = {4097, 4224, 4225, 8192, 8193, 12345, _PREWARM_MAX}
    edges.update(c for c in counts if c > 4096)
    edges.update(c - 1 for c in counts if c > 4097)
    if hidden * width > 5120 * 5120:
        edges = {4097, 8192, 8193, _PREWARM_MAX}  # bound the 27B graph-capture cost

    def refuse(*_args, **_kwargs):
        raise AssertionError("compile or provider load after prewarm")

    monkeypatch.setattr(cute, "compile", refuse)
    for module in (_nvfp4_cute_quant, _nvfp4_quack, _nvfp4_flashinfer):
        monkeypatch.setattr(module, "_resolve", refuse)
    for rows in sorted(edges):
        x = torch.randn(rows, hidden, device=_device(), dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = op(x)
        graph.replay()
        expected = op(x)
        if kind == "linear":
            assert torch.equal(result, expected), rows
        else:
            assert torch.equal(result.data, expected.data), rows
            assert torch.equal(
                unblocked_scale(result.scale, rows, width).view(torch.uint8),
                unblocked_scale(expected.scale, rows, width).view(torch.uint8),
            ), rows
        del graph, result, expected, x


def test_split_k_reduction_is_deterministic_and_leaves_counters_zeroed() -> None:
    if not _quack_runs():
        pytest.skip("the Triton GEMM serves qualified devices (asserted by _quack_runs)")
    gate, up, x, a, input_scale, output_scale, _, _ = _gated_case(2048, 512, 7)
    fc1 = NVFP4GateUpSiLU(gate, up, input_scale=input_scale, output_scale=output_scale)
    fc1.plan_for = lambda rows, **_: ("triton", Small(8, 32, 128, 4, 2, False))
    first = fc1(x)
    for _ in range(20):
        again = fc1(x)
        assert torch.equal(again.data, first.data) and torch.equal(again.scale, first.scale)
    assert not fc1._counters.any()


@pytest.mark.parametrize("capture", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16-input", "quantized-input"])
def test_inplace_global_scales_are_read_on_every_call(capture, quantized):
    """Fresh operators catch constructor caches; replay catches host-side/cache refreshes.

    Update each scale independently, using both copy_ and mul_. For prequantized calls,
    the activation's own global scale changes without regenerating its packed operands.
    """
    h, i, rows = 256, 384, 3
    gate, up, x, a, si, so, _, _ = _gated_case(h, i, rows)

    def exercise(make, invoke, scales):
        op = make()
        invoke(op)  # compile before capture
        graph = torch.cuda.CUDAGraph() if capture else None
        if capture:
            with torch.cuda.graph(graph):
                result = invoke(op)
        for scale in scales:
            for factor, copy in ((0.37, True), (1.73, False)):
                if copy:
                    scale.copy_(scale * factor)
                else:
                    scale.mul_(factor)
                if capture:
                    graph.replay()
                else:
                    result = invoke(op)
                expected = invoke(make())
                if isinstance(result, NVFP4Activation):
                    torch.testing.assert_close(
                        _activation_values(result), _activation_values(expected), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        unblocked_scale(result.scale, rows, i).float(),
                        unblocked_scale(expected.scale, rows, i).float(),
                        rtol=0,
                        atol=0,
                    )
                else:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # CuTe small-M quantizes floating-point input inside the GEMM, so it has no
    # prequantized entry point.
    for provider, path, plan in _gated_variants(rows, h, i):
        if quantized and isinstance(plan, _nvfp4_dispatch.Cute):
            continue
        si.copy_((x.float().abs().amax() / 2688).reshape(1))
        so.fill_(0.01)

        def make_gated(provider=provider, path=path, plan=plan):
            op = NVFP4GateUpSiLU(gate, up, input_scale=si, output_scale=so, provider=provider)
            if plan is not None:
                op.plan_for = lambda count, **_: (path, plan)
            return op

        exercise(make_gated, lambda op: op.quantized(a) if quantized else op(x), (si, so))
    for provider, plan in _linear_variants(rows, i, h):
        si.copy_((x.float().abs().amax() / 2688).reshape(1))

        def make_linear(provider=provider, plan=plan):
            op = NVFP4Linear(gate, si, provider=provider)
            if plan is not None:
                op.plan_for = lambda count, **_: plan
            return op

        exercise(make_linear, lambda op: op.quantized(a) if quantized else op(x), (si,))


# FlashInfer consumes exactly the same quantized operands as QuACK. Multiplying
# acc * (weight_scale * input_scale), rather than (acc * weight_scale) * input_scale,
# and a different FP32 reduction order can move a result across a rounding boundary.
# Permit one output epsilon plus an FP32 cancellation floor; do not requantize weights.
def _assert_flashinfer_projection(actual, expected):
    eps = torch.finfo(actual.dtype).eps
    torch.testing.assert_close(
        actual, expected, rtol=eps, atol=expected.abs().max().item() * 2**-20
    )


def _require_flashinfer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    launcher = _nvfp4_flashinfer.load(_device())
    if launcher is None:
        pytest.skip("Qualified FlashInfer CUTLASS FP4 unavailable on this device")
    return launcher


@pytest.mark.parametrize("rows", [1, 3, 129])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flashinfer_linear_matches_quack_quantized_operands(rows, dtype):
    _require_flashinfer()
    if _nvfp4_quack.load(_device()) is None:
        pytest.skip("QuACK reference unavailable")
    torch.manual_seed(42)
    w = random_weight(256, 512)
    scale = torch.full((1,), 4 / 2688, device=_device())
    x = torch.randn(rows, 512, device=_device(), dtype=torch.bfloat16)
    a = quantize_nvfp4(x, scale)
    op = NVFP4Linear(w, scale, provider="flashinfer", out_dtype=dtype)
    ref = NVFP4Linear(w, scale, provider="quack", out_dtype=dtype)
    ref.plan_for = lambda _rows, **_: _nvfp4_quack.DEFAULT
    assert op.provider == "flashinfer"
    _assert_flashinfer_projection(op.quantized(a), ref.quantized(a))
    torch.testing.assert_close(op(x), op.quantized(a), rtol=0, atol=0)
    out = torch.empty(rows, 256, dtype=dtype, device=_device())
    assert op.quantized(a, out=out) is out
    torch.testing.assert_close(out, op(x), rtol=0, atol=0)


def _gated_fp64_projections(a, gate, up):
    # Decode exact E2M1 * E4M3 operands first; globals and accumulation stay FP64.
    # In particular, never BF16-round the unscaled GEMM before applying a slot alpha.
    av = dequantize_nvfp4(a.data, unblocked_scale(a.scale, *a.shape)).double()
    return tuple(
        (av @ dequantize_nvfp4(w.data, w.scale).double().T)
        * (a.global_scale.double() * w.global_scale.double())
        for w in (gate, up)
    )


@pytest.mark.parametrize("shared", [False, True])
def test_flashinfer_gated_gemm_matches_fp64(shared):
    launcher = _require_flashinfer()
    torch.manual_seed(71)
    gate, up = random_weight(256, 512), random_weight(256, 512)
    if shared:
        up = up._replace(global_scale=gate.global_scale)
    scale = torch.full((1,), 4 / 2688, device=_device())
    x = torch.randn(129, 512, device=_device(), dtype=torch.bfloat16)
    a = quantize_nvfp4(x, scale)
    op = NVFP4GateUpSiLU(gate, up, input_scale=scale, output_scale=scale, provider="flashinfer")
    projection = torch.empty(129, 512, device=_device(), dtype=torch.bfloat16)
    launcher.gated(a, op._flash_weight, projection)
    g64, u64 = _gated_fp64_projections(a, gate, up)
    refs = [value.bfloat16() for value in (g64, u64)]
    expected_projection = (
        torch.stack(refs, dim=-1).flatten(1) if shared else torch.stack(refs).view_as(projection)
    )
    _assert_flashinfer_projection(projection, expected_projection)
    # Independent oracle includes exactly one BF16 boundary, then SiLU/quantization.
    expected_data, expected_scale = reference_quantize(
        F.silu(refs[0].float()) * refs[1].float(), scale
    )
    expected = dequantize_nvfp4(expected_data, expected_scale, scale).double()
    oracle = F.silu(g64) * u64
    actual = _activation_values(op.quantized(a)).double()
    # Fixed seed; the measured seed-noise MARGIN, no extra slack (five seeds: gate 2).
    assert _rms(actual - oracle) <= (1.0 + MARGIN) * _rms(expected - oracle)


@pytest.mark.parametrize("quantizer", ["cute", "triton"])
def test_flashinfer_gated_single_rounding_scale_byte(monkeypatch, quantizer):
    if quantizer == "cute":
        _cute_quantizer(monkeypatch)
    monkeypatch.setenv("CTI_NVFP4_QUANT", quantizer)
    _require_flashinfer()
    from ctikernels.linear.nvfp4 import blocked_scale

    # Review repro: the old BF16 -> slot scale -> BF16 order produced 0x39 (1.125).
    rows, width, k = 3, 128, 5120
    data = torch.full((width, k // 2), 0x22, dtype=torch.uint8, device=_device())
    blocks = torch.ones(width, k // 16, device=_device()).to(torch.float8_e4m3fn)
    gate = NVFP4Weight(data, blocks, torch.tensor([0.7], device=_device()))
    up = NVFP4Weight(data, blocks, torch.tensor([0.9], device=_device()))
    input_scale = torch.tensor([1.004 / k], device=_device())
    output_scale = torch.tensor([0.066942364], device=_device())
    a = NVFP4Activation(data[:rows], blocked_scale(blocks[:rows]), input_scale)
    g64, u64 = _gated_fp64_projections(a, gate, up)
    rounded = F.silu(g64.bfloat16().float()) * u64.bfloat16().float()
    ref_data, ref_scale = reference_quantize(rounded, output_scale)
    assert torch.all(ref_scale.view(torch.uint8) == 0x38)
    op = NVFP4GateUpSiLU(
        gate, up, input_scale=input_scale, output_scale=output_scale, provider="flashinfer"
    )
    actual = op.quantized(a)
    assert torch.all(unblocked_scale(actual.scale, rows, width).view(torch.uint8) == 0x38)
    expected = dequantize_nvfp4(ref_data, ref_scale, output_scale)
    torch.testing.assert_close(_activation_values(actual), expected, rtol=0, atol=0)
    # At these values the neighboring E2M1 codes are 4 and 6: half a step is sf*global.
    torch.testing.assert_close(
        _activation_values(actual).double(), F.silu(g64) * u64, rtol=0, atol=float(output_scale)
    )


@pytest.mark.parametrize("provider", ["flashinfer", "quack-unfused", "quack-fused", "cute"])
@pytest.mark.parametrize("globals_", [(0.7, 0.9), (0.7, 22.4)])
def test_gated_unequal_scales_against_fp64(provider, globals_):
    if provider == "flashinfer":
        _require_flashinfer()
    elif not _quack_runs():
        pytest.skip("QuACK/CuTe unavailable")
    ratios = []
    for seed in PRECISION_SEEDS:
        torch.manual_seed(seed)
        gate, up = random_weight(256, 512), random_weight(256, 512)
        gate, up = tuple(
            w._replace(
                scale=(w.scale.float() / 8192).to(torch.float8_e4m3fn),
                global_scale=torch.tensor([value], device=_device()),
            )
            for w, value in zip((gate, up), globals_, strict=True)
        )
        x = (torch.randn(17, 512, device=_device()) * 0.1).bfloat16()
        input_scale = torch.tensor([0.5 / 2688], device=_device())
        a = quantize_nvfp4(x, input_scale)
        g64, u64 = _gated_fp64_projections(a, gate, up)
        oracle = F.silu(g64) * u64
        output_scale = (oracle.abs().amax().float() / 2688).reshape(1)
        refs = _gated_references(g64, u64, output_scale)
        op = NVFP4GateUpSiLU(
            gate,
            up,
            input_scale=input_scale,
            output_scale=output_scale,
            provider="quack" if provider.startswith("quack") else provider,
        )
        if provider.startswith("quack"):
            op.plan_for = lambda _rows, **_: (provider, _nvfp4_quack.DEFAULT)
        if provider == "cute":
            assert isinstance(op.plan_for(17)[1], _nvfp4_dispatch.Cute)
            actual = op(x)  # quantized() deliberately selects the non-CuTe fallback
        else:
            actual = op.quantized(a)
        error = _rms(_activation_values(actual).double() - oracle)
        baseline = _rms(refs["fp32-bf16"] - oracle)
        ratios.append(error / max(baseline, 1e-300))
    _assert_precision(ratios, (provider, globals_))


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("quantizer", ["cute", "triton"])
def test_flashinfer_graph_replay_reads_updated_global_scales(monkeypatch, gated, shared, quantizer):
    if quantizer == "cute":
        _cute_quantizer(monkeypatch)
    monkeypatch.setenv("CTI_NVFP4_QUANT", quantizer)
    _require_flashinfer()
    w, up = random_weight(256, 512), random_weight(256, 512)
    if shared:
        up = up._replace(global_scale=w.global_scale)
    scale = torch.full((1,), 4 / 2688, device=_device())
    output_scale = torch.full_like(scale, 0.5 / 2688)
    x = torch.randn(3, 512, device=_device(), dtype=torch.bfloat16)

    def build():
        return (
            NVFP4GateUpSiLU(
                w, up, input_scale=scale, output_scale=output_scale, provider="flashinfer"
            )
            if gated
            else NVFP4Linear(w, scale, provider="flashinfer")
        )

    op = build()
    op(x)  # compile and allocate FlashInfer's workspace before capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(x)
    assert op.quantizer == quantizer
    graph.replay()
    before = (actual.data if gated else actual).clone()
    scale.mul_(1.5)
    w.global_scale.mul_(2)
    if not shared:
        up.global_scale.mul_(0.5)
    output_scale.mul_(1.25)
    graph.replay()
    expected = build()(x)  # a fresh operation also detects construction-time scalar caching
    assert not torch.equal(before, actual.data if gated else actual)
    if gated:
        assert torch.equal(actual.data, expected.data)
        torch.testing.assert_close(
            unblocked_scale(actual.scale, 3, 256).float(),
            unblocked_scale(expected.scale, 3, 256).float(),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gated", [False, True])
def test_flashinfer_requires_grad_raises_before_provider_load(monkeypatch, gated):
    from ctikernels.linear import _nvfp4_flashinfer

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    w = random_weight(128, 128)
    bad = w._replace(global_scale=w.global_scale.clone().requires_grad_())
    scale = torch.ones(1, device=_device())
    monkeypatch.setattr(
        _nvfp4_flashinfer, "load", lambda _: pytest.fail("loaded before validation")
    )
    with pytest.raises(ValueError, match="frozen"):
        if gated:
            NVFP4GateUpSiLU(w, bad, input_scale=scale, output_scale=scale, provider="flashinfer")
        else:
            NVFP4Linear(bad, scale, provider="flashinfer")


@pytest.mark.parametrize("rows", [3, 64, 512])
@pytest.mark.parametrize("quantized", [False, True])
def test_flashinfer_first_graph_replay_matches_eager(rows, quantized):
    """The first replay after capture must already be correct (it was not on sm121)."""
    _require_flashinfer()
    torch.manual_seed(rows)
    w = random_weight(384, 256)
    scale = torch.full((1,), 1 / 2688, device=_device())
    x = torch.randn(rows, 256, device=_device(), dtype=torch.bfloat16)
    a = quantize_nvfp4(x, scale)
    op = NVFP4Linear(w, scale, provider="flashinfer")
    op.plan_for = lambda count, **_: _nvfp4_dispatch.FlashInfer()
    call = (lambda: op.quantized(a)) if quantized else (lambda: op(x))
    expected = call().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    graph.replay()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("provider", ["flashinfer", "quack"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("quantizer", ["cute", "triton"])
def test_fused_scalar_preparation_reads_updates_on_calls_and_replays(
    monkeypatch, provider, quantized, shared, quantizer
):
    if quantizer == "cute":
        _cute_quantizer(monkeypatch)
    monkeypatch.setenv("CTI_NVFP4_QUANT", quantizer)
    if provider == "flashinfer":
        _require_flashinfer()
    elif not _quack_runs():
        pytest.skip("QuACK unavailable")
    gate, up = random_weight(256, 512), random_weight(256, 512)
    if shared:
        up = up._replace(global_scale=gate.global_scale)
    scale = torch.full((1,), 4 / 2688, device=_device())
    output_scale = torch.full_like(scale, 0.5 / 2688)
    x = torch.randn(17, 512, device=_device(), dtype=torch.bfloat16)
    a = quantize_nvfp4(x, scale)

    def build():
        op = NVFP4GateUpSiLU(
            gate, up, input_scale=scale, output_scale=output_scale, provider=provider
        )
        if provider == "quack":
            op.plan_for = lambda _rows, **_: (_nvfp4_dispatch.FUSED, _nvfp4_quack.DEFAULT)
        return op

    op = build()
    call = (lambda: op.quantized(a)) if quantized else (lambda: op(x))
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call()
    graph.replay()
    for _ in range(2):
        scale.mul_(1.25)
        output_scale.mul_(1.125)
        gate.global_scale.mul_(1.5)
        if not shared:
            up.global_scale.mul_(0.75)
        graph.replay()
        fresh = build()
        expected = fresh.quantized(a) if quantized else fresh(x)
        eager = call()
        for result in (actual, eager):
            assert torch.equal(result.data, expected.data)
            torch.testing.assert_close(result.scale.float(), expected.scale.float(), rtol=0, atol=0)
        if not quantized:
            reference = op.quantized(quantize_nvfp4(x, scale))
            assert torch.equal(actual.data, reference.data)
            torch.testing.assert_close(
                actual.scale.float(), reference.scale.float(), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "provider,gated,expected",
    [
        (None, False, 2),  # automatic Linear: quantization + FlashInfer GEMM
        ("flashinfer", False, 2),
        ("flashinfer", True, 3),
        ("quack", False, 2),
        ("quack", True, 2),
    ],
)
@pytest.mark.parametrize("shared", [False, True])
def test_nvfp4_captured_kernel_count(provider, gated, expected, shared):
    if provider in (None, "flashinfer"):
        _require_flashinfer()
    elif not _quack_runs():
        pytest.skip("QuACK unavailable")
    from benchmark.kernel_tuning.nvfp4_parity import kernels_per_call

    w = random_weight(256, 512)
    up = w if shared else w._replace(global_scale=w.global_scale.clone())
    if provider == "flashinfer" and gated and not shared:
        # Two GEMMs plus a device multiply for up alpha: 5 kernels instead of 3.
        # Gate alpha remains fused into input quantization; there is no interleave copy.
        expected += 2
    scale = torch.full((1,), 4 / 2688, device=_device())
    op = (
        NVFP4GateUpSiLU(w, up, input_scale=scale, output_scale=scale, provider=provider)
        if gated
        else NVFP4Linear(w, scale, provider=provider)
    )
    if provider is None:
        assert op.provider == "flashinfer"
        assert op.plan_for(128) == _nvfp4_dispatch.FlashInfer()
    if provider == "quack":
        op.plan_for = (
            (lambda _: (_nvfp4_dispatch.FUSED, _nvfp4_quack.DEFAULT))
            if gated
            else (lambda _: _nvfp4_quack.DEFAULT)
        )
    x = torch.randn(128, 512, device=_device(), dtype=torch.bfloat16)
    assert kernels_per_call(op, x) <= expected


@pytest.mark.parametrize("hidden,intermediate", [(2048, 512), (5120, 17408)])
def test_independent_gate_up_globals_never_dispatch_flashinfer(hidden, intermediate):
    """Independent globals need two FlashInfer GEMMs; dispatch takes the next table entry."""
    gate = random_weight(intermediate, hidden)
    up = gate._replace(global_scale=gate.global_scale * 1.25)
    scale = torch.full((1,), 4 / 2688, device=_device())
    op = NVFP4GateUpSiLU(gate, up, input_scale=scale, output_scale=scale)
    shared = NVFP4GateUpSiLU(gate, gate, input_scale=scale, output_scale=scale)
    for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 4096):
        assert not isinstance(op.plan_for(rows)[1], _nvfp4_dispatch.FlashInfer), rows
        entry = _nvfp4_dispatch.gated_plan(
            op.device, hidden, intermediate, rows, allow_cute=op._cute is not None
        )
        if not isinstance(entry[1], _nvfp4_dispatch.FlashInfer):  # table rows not using it
            assert op.plan_for(rows) == shared.plan_for(rows), rows
    x = torch.randn(32, hidden, device=_device(), dtype=torch.bfloat16)
    reference = op.quantized(quantize_nvfp4(x, scale))
    actual = op(x)
    assert torch.equal(actual.data, reference.data)


def _cute_quantizer(monkeypatch):
    from ctikernels.linear import _nvfp4_cute_quant

    if os.environ.get(_nvfp4_cute_quant.VARIABLE) == "triton":
        pytest.skip("CuTe qualification disabled by CTI_NVFP4_QUANT=triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not _nvfp4_cute_quant.available(_device()):
        pytest.skip("CuTe quantization needs DSL 4.8.0 and sm120/sm121")
    monkeypatch.setenv(_nvfp4_cute_quant.VARIABLE, "cute")
    return _nvfp4_cute_quant.load(_device())


@pytest.mark.parametrize("quantizer", ["cute", "triton"])
@pytest.mark.parametrize("rows,cols", [(1, 32), (3, 256), (129, 512), (1024, 5120)])
def test_gated_quantizer_row_blocks_match_interleaved(monkeypatch, quantizer, rows, cols):
    if quantizer == "cute":
        launcher = _cute_quantizer(monkeypatch)
    else:
        from ctikernels.linear import _nvfp4_triton

        launcher = _nvfp4_triton
    torch.manual_seed(rows)
    slots = torch.randn(2, rows, cols, device=_device(), dtype=torch.bfloat16)
    interleaved = slots.permute(1, 2, 0).contiguous().view(rows, 2 * cols)
    row_blocks = slots.view(rows, 2 * cols)
    scale = torch.tensor([0.01], device=_device())
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    launcher.quantize(interleaved, scale, gated=True, **expected)
    launcher.quantize(row_blocks, scale, gated=True, gated_row_blocks=True, **actual)
    _assert_quant_bytes(actual, expected)


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "unseen-alignment"])
def test_cute_quantizer_capture_miss_is_recoverable(monkeypatch, warm):
    original = _cute_quantizer(monkeypatch)
    cute = type(original)(_device())  # fresh cache, independent of test order
    x = torch.randn(3, 512, device=_device(), dtype=torch.bfloat16)
    scale = torch.ones(1, device=_device())
    buffers = _quant_buffers(3, 512)
    if warm:
        cute.quantize(x, scale, gated=False, **buffers)
        x = torch.randn(3 * 512 + 1, device=_device(), dtype=torch.bfloat16)[1:].view(3, 512)
    count = len(cute.compiled)
    graph = torch.cuda.CUDAGraph()
    # Catch inside the context: capture_end must succeed, not mask the intended error.
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="prewarm.*outside CUDA graph capture"):
            cute.quantize(x, scale, gated=False, **buffers)
    assert not torch.cuda.is_current_stream_capturing()
    assert len(cute.compiled) == count
    cute.quantize(x, scale, gated=False, **buffers)
    expected = {name: value.clone() for name, value in buffers.items()}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute.quantize(x, scale, gated=False, **buffers)
    graph.replay()
    _assert_quant_bytes(buffers, expected)


@pytest.mark.parametrize("site", ["quantizer", "flashinfer"])
def test_capture_guard_with_other_device_current(monkeypatch, site):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    other = torch.cuda.current_device()
    target = torch.device("cuda", (other + 1) % torch.cuda.device_count())
    with torch.cuda.device(target):
        if site == "quantizer":
            original = _cute_quantizer(monkeypatch)
            launcher = type(original)(target)
            x = torch.empty(3, 64, dtype=torch.bfloat16, device=target)
            scale = torch.ones(1, device=target)
            buffers = _quant_buffers(3, 64)
            monkeypatch.setattr(
                launcher, "_compile", lambda _: pytest.fail("compile during capture")
            )
            call = lambda: launcher.quantize(x, scale, gated=False, **buffers)
        else:
            monkeypatch.setattr(_nvfp4_flashinfer, "_launchers", {})
            monkeypatch.setattr(
                _nvfp4_flashinfer, "_resolve", lambda _: pytest.fail("build during capture")
            )
            call = lambda: _nvfp4_flashinfer.load(target)
        stream = torch.cuda.Stream(device=target)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with torch.cuda.device(other):
                assert torch.cuda.current_device() == other
                assert not torch.cuda.is_current_stream_capturing()
                with pytest.raises(RuntimeError, match="prewarm.*outside CUDA graph capture"):
                    call()
                assert torch.cuda.current_device() == other
        assert not torch.cuda.is_current_stream_capturing()
    assert torch.cuda.current_device() == other


def _quant_buffers(rows, cols):
    from ctikernels.linear.nvfp4 import scale_shape

    return {
        "data": torch.full((rows, cols // 2), 0xA5, device=_device(), dtype=torch.uint8),
        "scale": torch.full(scale_shape(rows, cols), 0xA5, device=_device(), dtype=torch.uint8),
    }


def _assert_quant_bytes(actual, expected):
    for name in ("data", "scale"):
        assert torch.equal(actual[name], expected[name]), name


@pytest.mark.parametrize(
    "cols",
    [
        16,
        48,
        80,
        256,
        272,
        512,
        768,
        784,
        2048,
        4080,
        4096,
        4112,
        5120,
        12288,
        17408,
    ],
)
def test_cute_quantizer_row_geometry(cols):
    """CPU contract: tile coverage and scale pairs at every M; sm120 doubles loads per lane."""
    from ctikernels.linear._nvfp4_cute_quant import _TWO_GROUPS_MIN, _geometry

    halves = cols // 8
    for rows in (1, 16, 64, 65, 127, 128, 129, 512, 2048, 8192):
        single = (128, halves, 1, max(1, 128 // halves), 0)
        assert _geometry(cols, rows) == single
        wide = _geometry(cols, rows, True)
        assert wide == (
            single if rows * cols < _TWO_GROUPS_MIN else (128, halves, 2, max(1, 128 // halves), 0)
        )
        threads, _, groups, rows_per_cta, _ = wide
        if rows_per_cta > 1:
            continue
        tiles = (halves + threads * groups - 1) // (threads * groups)
        owned = []
        for tile in range(tiles):
            for tid in range(threads):
                for group in range(groups):
                    half = tile * threads * groups + tid + group * threads
                    if half < halves:
                        owned.append(half)
                        # Scale pairs: the partner lane holds the other half of the block.
                        assert (half ^ 1) < halves and (half ^ 1) // 2 == half // 2
                        assert (tid ^ 1) // 32 == tid // 32
        assert sorted(owned) == list(range(halves))


@pytest.fixture
def cute_quantizer_schedule():
    """Execute scheduling Python without importing the DSL or compiling kernels."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from ctikernels.linear import _nvfp4_cute_quant

    tree = ast.parse(Path(_nvfp4_cute_quant.__file__).with_name("kernel.py").read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    kernel.body = [
        node
        for node in kernel.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("__init__", "__call__", "kernel", "_quantize_row")
    ]
    for node in kernel.body:
        node.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    turns = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_turns"
    )
    turns.decorator_list = []
    module.body.append(turns)
    module.body.append(kernel)
    namespace = {
        "Int32": int,
        "Int64": int,
        "Float32": float,
        "const_expr": bool,
        "_reciprocal": lambda value: 1.0 / value,
        "_launch_bounds": _nvfp4_cute_quant._launch_bounds,
        "cutlass": SimpleNamespace(
            min=min,
            max=max,
            range=range,
            range_constexpr=range,
            select_=lambda condition, yes, no: yes if condition else no,
        ),
    }
    exec(compile(module, "<quantizer scheduling>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("cols", [16, 512, 768, 784, 4096, 5120, 17408])
@pytest.mark.parametrize(
    "rows,sm_count,side_outputs",
    [
        (65, 1, False),
        (128, 24, False),
        (128, 24, True),
        (129, 128, True),
        (512, 24, False),
        (2049, 128, False),
        (2**31 + 129, 24, True),
    ],
)
def test_cute_quantizer_row_grid(cute_quantizer_schedule, cols, rows, sm_count, side_outputs):
    from types import SimpleNamespace

    from ctikernels.linear._nvfp4_cute_quant import _row_loop_geometry

    kernel = cute_quantizer_schedule["Kernel"](
        cols,
        False,
        (False,) * 9,
        259 if side_outputs else 0,
        0,
        True,
        True,
        _row_loop_geometry(cols),
    )
    launched = {}

    def entry(*args):
        launched["args"] = args
        return SimpleNamespace(launch=lambda **kwargs: launched.update(kwargs))

    run_kernel = kernel.kernel
    kernel.kernel = entry
    kernel(SimpleNamespace(shape=(rows, cols)), *([None] * 12), 16, sm_count, None)
    threads = kernel.threads
    quant_ctas = min(rows, sm_count * max(1, 2048 // threads))
    assert launched["args"][-5:-3] == (rows, quant_ctas)
    remainder = rows % 128
    chunks = len({row % 32 for row in range(remainder, 128)}) if remainder else 0
    padding_tasks = chunks * ((cols + 63) // 64)
    tail_tasks = rows if cols % 64 else 0
    pad_ctas = (padding_tasks + tail_tasks + threads - 1) // threads
    assert launched["args"][-3] == padding_tasks
    assert launched["grid"] == [quant_ctas + max(pad_ctas, int(side_outputs)), 1, 1]
    assert launched["block"] == [threads, 1, 1]
    assert launched["use_pdl"] is True
    assert launched["min_blocks_per_mp"] == min(2048 // threads, 24, 1536 // threads)
    # Check the unit-step row trip count, including offsets past signed i32.
    for bid in {0, quant_ctas - 1}:
        trips = cute_quantizer_schedule["_turns"](bid, rows, quant_ctas)
        last = bid + (trips - 1) * quant_ctas
        assert last < rows <= last + quant_ctas
    if rows < 4096:
        owned = []
        kernel._quantize_row = lambda x, data, scale, inverse, gf, uf, row, tid: owned.append(row)
        arch = SimpleNamespace(
            thread_idx=lambda: (0, 0, 0),
            block_idx=lambda: (bid, 0, 0),
            griddepcontrol_wait=lambda: None,
            griddepcontrol_launch_dependents=lambda: None,
        )
        cute_quantizer_schedule["cute"] = SimpleNamespace(arch=arch)
        for bid in range(quant_ctas):  # noqa: B007 (read by the block_idx closure)
            args = list(launched["args"])
            args[3] = [1.0]
            run_kernel(*args)
        assert sorted(owned) == list(range(rows))


@pytest.mark.parametrize("cols", [16, 784, 4096, 4112, 5120, 12288, 17408])
@pytest.mark.parametrize("gated", [False, True])
def test_cute_quantizer_row_load_order(cute_quantizer_schedule, cols, gated):
    from types import SimpleNamespace

    from ctikernels.linear._nvfp4_cute_quant import _row_loop_geometry

    namespace = cute_quantizer_schedule
    source = {}

    def allocate(shape, dtype):
        assert shape == (16 if gated else 8, kernel.groups)
        assert dtype == "native-16-bit"
        return source

    namespace["cute"] = SimpleNamespace(make_rmem_tensor=allocate)
    kernel = namespace["Kernel"](
        cols,
        gated,
        (False,) * 9,
        0,
        0,
        True,
        True,
        _row_loop_geometry(cols),
    )
    for group in range(kernel.groups):
        source[None, group] = group
    events = []
    kernel._load_lane = lambda x, source, row, half: events.append(("load", row, half))
    kernel._quantize_lane = lambda source, data, scale, inverse, gf, uf, row, half, tid: (
        events.append(("convert", row, half))
    )
    for tid in range(kernel.threads):
        events.clear()
        kernel._quantize_row(
            SimpleNamespace(element_type="native-16-bit"), None, None, 1.0, 1.0, 1.0, 2**31 + 1, tid
        )
        halves = list(range(tid, cols // 8, kernel.threads))
        assert events == [
            (action, 2**31 + 1, half) for action in ("load", "convert") for half in halves
        ]


@pytest.mark.parametrize("rows", [65, 129, 2048])
@pytest.mark.parametrize("cols", [512, 768, 784, 2048, 4096, 4112, 5120, 12288, 17408])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["linear", "gated", "scaled-gated"])
def test_cute_quantizer_strided_rows(monkeypatch, rows, cols, dtype, mode):
    """Cover column tails, non-power-of-two CTAs and multiple row-loop turns."""
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    gated = mode != "linear"
    torch.manual_seed(cols)
    x = torch.randn(rows, cols * (2 if gated else 1), device=_device(), dtype=dtype)
    global_scale = torch.full((1,), 0.0017, device=_device())
    factors = {}
    if mode == "scaled-gated":
        factors = {
            "gate_factor": torch.full_like(global_scale, 0.75),
            "up_factor": torch.full_like(global_scale, 1.375),
        }
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected, **factors)
    cute.quantize(x, global_scale, gated=gated, **actual, **factors)
    _assert_quant_bytes(actual, expected)
    # Reuse compiled geometry for another M while reading the updated scale.
    compiled_count = len(cute.compiled)
    x = x[:-1] if rows > 65 else x[:1].expand(67, -1).contiguous()
    global_scale.mul_(1.125)
    expected, actual = _quant_buffers(x.shape[0], cols), _quant_buffers(x.shape[0], cols)
    _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected, **factors)
    cute.quantize(x, global_scale, gated=gated, **actual, **factors)
    assert len(cute.compiled) == compiled_count
    _assert_quant_bytes(actual, expected)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 31, 64, 127, 128, 129, 255, 300])
@pytest.mark.parametrize("cols", [16, 48, 80, 512, 2048, 4096, 12288, 17408])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["linear", "gated", "scaled-gated"])
def test_cute_quantizer_bit_exact(monkeypatch, rows, cols, dtype, mode):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    gated = mode != "linear"
    torch.manual_seed(rows * cols)
    x = torch.randn(rows, cols * (2 if gated else 1), device=_device(), dtype=dtype)
    global_scale = torch.full((1,), 0.0017, device=_device())
    factors = {}
    if mode == "scaled-gated":
        factors = {
            "gate_factor": torch.full_like(global_scale, 0.75),
            "up_factor": torch.full_like(global_scale, 1.375),
        }
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected, **factors)
    cute.quantize(x, global_scale, gated=gated, **actual, **factors)
    _assert_quant_bytes(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("case", ["zero", "nonfinite", "ties", "extremes"])
@pytest.mark.parametrize("rows", [3, 129])
def test_cute_quantizer_special_values(monkeypatch, dtype, gated, case, rows):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    cols = 512
    x = torch.zeros(rows, cols * (2 if gated else 1), device=_device(), dtype=dtype)
    global_scale = torch.ones(1, device=_device())
    if case == "nonfinite":
        x.fill_(0.25)
        x[0, 0], x[1, 17], x[2, -1] = float("nan"), float("inf"), -float("inf")
    elif case == "ties":
        values = [
            0,
            -0.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5,
            6,
            -0.25,
            -0.75,
            -1.25,
            -2.5,
            -3.5,
            -5,
        ]
        x.copy_(torch.tensor(values, device=_device(), dtype=dtype).repeat(rows, x.shape[1] // 16))
    elif case == "extremes":
        limits = torch.finfo(dtype)
        values = [
            limits.max,
            -limits.max,
            limits.tiny,
            -limits.tiny,
            0,
            -0.0,
            2**-20,
            -(2**-20),
            -88,
            88,
            -90,
            90,
            1,
            -1,
            2,
            -2,
        ]
        x.copy_(torch.tensor(values, device=_device(), dtype=dtype).repeat(rows, x.shape[1] // 16))
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected)
    cute.quantize(x, global_scale, gated=gated, **actual)
    _assert_quant_bytes(actual, expected)
    if not gated:
        from .nvfp4_oracle import flashinfer_quantize

        data, scale = flashinfer_quantize(x, global_scale)
        _assert_quant_bytes(actual, {"data": data, "scale": scale})


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("rows", [7, 129])
def test_cute_quantizer_unaligned_views(monkeypatch, gated, rows):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    cols = 80
    width = cols * (2 if gated else 1)
    x = torch.randn(rows, width + 3, device=_device(), dtype=torch.bfloat16)[:, 1 : width + 1]
    global_scale = torch.full((1,), 0.002, device=_device())
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    storage = torch.full((rows, cols // 2 + 3), 0xA5, device=_device(), dtype=torch.uint8)
    actual["data"] = storage[:, 1 : cols // 2 + 1]
    _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected)
    cute.quantize(x, global_scale, gated=gated, **actual)
    _assert_quant_bytes(actual, expected)
    if not gated:
        from .nvfp4_oracle import flashinfer_quantize

        data, scale = flashinfer_quantize(x, global_scale)
        _assert_quant_bytes(actual, {"data": data, "scale": scale})
    assert (storage[:, 0] == 0xA5).all() and (storage[:, -2:] == 0xA5).all()


@pytest.mark.parametrize("offset", [0, 4, 1])
@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 16),
        (1, 32),
        (1, 48),
        (1, 64),
        (32, 80),
        (33, 80),
        (64, 80),
        (65, 80),
        (96, 80),
        (97, 80),
        (127, 80),
        (128, 80),
        (129, 80),
    ],
)
def test_cute_quantizer_padding_store_boundaries(monkeypatch, offset, rows, cols):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    x = torch.randn(rows, cols, device=_device(), dtype=torch.bfloat16)
    global_scale = torch.full((1,), 0.002, device=_device())
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    size, shape = actual["scale"].numel(), actual["scale"].shape
    storage = torch.full((size + offset + 16,), 0xA5, device=_device(), dtype=torch.uint8)
    actual["scale"] = storage[offset : offset + size].view(shape)
    _nvfp4_triton.quantize(x, global_scale, gated=False, **expected)
    cute.quantize(x, global_scale, gated=False, **actual)
    _assert_quant_bytes(actual, expected)
    assert (storage[:offset] == 0xA5).all() and (storage[offset + size :] == 0xA5).all()


@pytest.mark.parametrize("factor_value", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("rows", [3, 129])
def test_cute_quantizer_padding_live_factors(monkeypatch, factor_value, rows):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    cols = 80
    x = torch.randn(rows, cols * 2, device=_device(), dtype=torch.bfloat16)
    global_scale = torch.full((1,), 0.002, device=_device())
    factors = {
        "gate_factor": torch.ones_like(global_scale),
        "up_factor": torch.ones_like(global_scale),
    }
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    cute.quantize(x, global_scale, gated=True, **actual, **factors)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute.quantize(x, global_scale, gated=True, **actual, **factors)
    for value in (factor_value, 1.0):
        global_scale.mul_(1.25)
        for factor in factors.values():
            factor.fill_(value)
        actual["scale"].fill_(0xA5)
        graph.replay()
        _nvfp4_triton.quantize(x, global_scale, gated=True, **expected, **factors)
        _assert_quant_bytes(actual, expected)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("rows,cols", [(17, 80), (128, 64), (129, 5120)])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_cute_quantizer_pdl_graph_live_scalars(monkeypatch, gated, rows, cols, offset):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    source = torch.randn(rows, cols * (2 if gated else 1), device=_device(), dtype=torch.bfloat16)
    x = torch.empty_like(source)
    global_scale = torch.full((1,), 0.001, device=_device())
    inputs = {
        "weight_scale": torch.full_like(global_scale, 0.75),
        "up_scale": torch.full_like(global_scale, 1.25),
        "output_scale": torch.full_like(global_scale, 0.002),
    }
    if gated:
        inputs.update(gate_factor=inputs["weight_scale"], up_factor=inputs["up_scale"])

    def buffers():
        return dict(
            **_quant_buffers(rows, cols),
            alpha=torch.empty_like(global_scale),
            reciprocal=torch.empty_like(global_scale),
            column_scale=torch.empty(259, device=_device()),
            clear_scale=torch.full((1, 1, 2051), 0xA5, device=_device(), dtype=torch.uint8),
        )

    actual, expected = buffers(), buffers()
    columns = torch.full((259 + offset + 16,), -123.0, device=_device())
    cleared = torch.full((2051 + offset + 16,), 0xA5, device=_device(), dtype=torch.uint8)
    actual["column_scale"] = columns[offset : offset + 259]
    actual["clear_scale"] = cleared[offset : offset + 2051].view(1, 1, 2051)

    def run():
        torch.mul(source, 1.125, out=x)  # a prerequisite kernel inside the capture
        cute.quantize(x, global_scale, gated=gated, **actual, **inputs)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for replay in range(3):
        if replay:
            source.mul_(-1.125)
            global_scale.mul_(1.25)
            inputs["weight_scale"].mul_(0.75)
            inputs["up_scale"].mul_(1.5)
            inputs["output_scale"].mul_(1.125)
        actual["clear_scale"].fill_(0xA5)
        actual["scale"].fill_(0xA5)
        graph.replay()
        _nvfp4_triton.quantize(x, global_scale, gated=gated, **expected, **inputs)
        _assert_quant_bytes(actual, expected)
        for name in ("alpha", "reciprocal", "column_scale", "clear_scale"):
            assert torch.equal(actual[name], expected[name]), name
        assert (columns[:offset] == -123).all() and (columns[offset + 259 :] == -123).all()
        assert (cleared[:offset] == 0xA5).all() and (cleared[offset + 2051 :] == 0xA5).all()


def test_cute_quantizer_large_gated(monkeypatch):
    from ctikernels.linear import _nvfp4_triton

    cute = _cute_quantizer(monkeypatch)
    rows, cols = 2048, 12288
    x = torch.randn(rows, 2 * cols, device=_device(), dtype=torch.bfloat16)
    global_scale = torch.full((1,), 0.001, device=_device())
    expected, actual = _quant_buffers(rows, cols), _quant_buffers(rows, cols)
    _nvfp4_triton.quantize(x, global_scale, gated=True, **expected)
    cute.quantize(x, global_scale, gated=True, **actual)
    _assert_quant_bytes(actual, expected)


@pytest.mark.parametrize("provider", ["flashinfer", "quack"])
def test_cute_quantizer_unfused_pdl_chain(monkeypatch, provider):
    _cute_quantizer(monkeypatch)
    if provider == "flashinfer":
        _require_flashinfer()
    elif not _quack_runs():
        pytest.skip("QuACK unavailable")
    gate, up = random_weight(256, 512), random_weight(256, 512)
    global_scale = torch.full((1,), 0.0017, device=_device())
    output_scale = torch.full_like(global_scale, 0.0023)
    op = NVFP4GateUpSiLU(
        gate, up, input_scale=global_scale, output_scale=output_scale, provider=provider
    )
    if provider == "quack":
        op.plan_for = lambda _rows, **_: (_nvfp4_dispatch.UNFUSED, _nvfp4_quack.DEFAULT)
    x = torch.randn(17, 512, device=_device(), dtype=torch.bfloat16)
    op(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(x)
    assert op.quantizer == "cute"
    monkeypatch.setenv("CTI_NVFP4_QUANT", "triton")
    for replay in range(3):
        if replay:
            x.mul_(-1.125)
            global_scale.mul_(1.25)
            gate.global_scale.mul_(0.75)
            up.global_scale.mul_(1.5)
            output_scale.mul_(1.125)
        graph.replay()  # still CuTe: capture freezes the selected launchers
        expected = op(x)
        assert op.quantizer == "triton"
        assert torch.equal(actual.data, expected.data)
        assert torch.equal(actual.scale.view(torch.uint8), expected.scale.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 3, 17, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("splits,reduce_mode", [(1, "global"), (3, "global"), (2, "cluster")])
def test_cute_dispatched_math_graph_and_single_launch(
    monkeypatch, rows, dtype, splits, reduce_mode
):
    from benchmark.kernel_tuning.nvfp4 import _dispatch_metadata
    from benchmark.kernel_tuning.nvfp4_parity import kernels_per_call
    from ctikernels.linear import _nvfp4_cute_small as cute

    if not cute.available():
        pytest.skip("requires DSL 4.8.0 and RTX 5090 / RTX PRO 6000 / GB10")
    monkeypatch.delenv("CTI_CUTE_SMALL_DEBUG_TIMEOUT", raising=False)
    torch.manual_seed(123)
    n, k = 96, 256
    w, up = random_weight(n, k), random_weight(n, k, std=0.08)
    input_scale = torch.tensor([0.002], device=_device())
    output_scale = torch.tensor([0.001], device=_device())
    x = torch.randn(rows, k, device=_device(), dtype=dtype)
    plan = _nvfp4_dispatch.Cute(n_tile=64, k_tile=128, split_k=splits, reduce_mode=reduce_mode)
    device_type = _nvfp4_dispatch._device_type(x.device)
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device_type, {(k, n): ((32, "cute", plan),)})
    op = NVFP4GateUpSiLU(w, up, input_scale=input_scale, output_scale=output_scale)
    assert op.plan_for(rows) == ("cute", plan)
    metadata = _dispatch_metadata(op, rows, gated=True)
    assert metadata["dispatched_provider"] == metadata["implementation_language"] == "cute"
    packed = cute.prepare_gated(w, up)
    workspace = cute.Workspace(packed, rows, plan.config(), gated=True)

    def direct():
        return cute.gated(x, packed, input_scale, output_scale, plan.config(), workspace=workspace)

    def compare(actual, expected):
        assert torch.equal(actual.data, expected.data)
        assert torch.equal(
            unblocked_scale(actual.scale, rows, n).view(torch.uint8),
            unblocked_scale(expected.scale, rows, n).view(torch.uint8),
        )
        _assert_scale_padding_is_zero(actual)

    def check_math(actual):
        # Precision is gated by the five-seed FP64 test (measured-noise MARGIN); here, live
        # scales: a fresh op on copies of the current scales must agree bitwise with the
        # captured / replayed output.
        fresh_up = NVFP4Weight(up.data, up.scale, up.global_scale.clone())
        fresh_w = NVFP4Weight(w.data, w.scale, w.global_scale.clone())
        fresh = NVFP4GateUpSiLU(
            fresh_w, fresh_up, input_scale=input_scale.clone(), output_scale=output_scale.clone()
        )
        compare(actual, fresh(x))

    first = op(x)
    saved = first.data.clone()
    second = op(x)
    assert first.data.data_ptr() != second.data.data_ptr()
    compare(first, direct())
    check_math(first)
    supplied = op(x, out=second)
    assert supplied.data.data_ptr() == second.data.data_ptr()
    assert len(op._cute_workspaces) == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x)
    for factor in (1.0, 0.75, 1.25):
        input_scale.mul_(factor)
        w.global_scale.mul_(factor)
        up.global_scale.mul_(2.0 - factor)
        output_scale.mul_(2.0 - factor)
        graph.replay()
        compare(captured, direct())
        check_math(captured)
        compare(op(x), captured)
    assert torch.equal(first.data, saved)
    assert kernels_per_call(op, x) == 1
    monkeypatch.setattr(
        cute, "gated", lambda *args, **kw: pytest.fail("launched before grad validation")
    )
    # The gated op keeps only the caller's global-scale tensors of the checkpoint weights.
    up.global_scale.requires_grad_()
    with pytest.raises(ValueError, match="require grad"):
        op(x)


@pytest.mark.parametrize("rows", [(32, 3), (3, 32)])
def test_cute_graph_workspace_shares_split_k_scratch(monkeypatch, rows):
    from ctikernels import GraphWorkspace
    from ctikernels.linear import _nvfp4_cute_small as cute

    if not cute.available():
        pytest.skip("requires DSL 4.8.0 and RTX 5090 / RTX PRO 6000 / GB10")
    torch.manual_seed(7)
    n, k = 96, 256
    plan = _nvfp4_dispatch.Cute(n_tile=64, k_tile=128, split_k=3, reduce_mode="global")
    device_type = _nvfp4_dispatch._device_type(_device())
    monkeypatch.setitem(_nvfp4_dispatch.GATED, device_type, {(k, n): ((32, "cute", plan),)})
    input_scale = torch.tensor([0.002], device=_device())
    output_scale = torch.tensor([0.001], device=_device())
    ops = [
        NVFP4GateUpSiLU(
            random_weight(n, k), random_weight(n, k), input_scale=input_scale,
            output_scale=output_scale,
        )
        for _ in range(2)
    ]  # fmt: skip
    xs = {m: torch.randn(m, k, device=_device(), dtype=torch.bfloat16) for m in rows}
    bits = lambda y: y.data.view(torch.uint8)
    with GraphWorkspace(max_rows=32) as group:
        eager = {(i, m): bits(op(xs[m])).clone() for i, op in enumerate(ops) for m in rows}
        graphs, outputs = {}, {}
        for i, op in enumerate(ops):
            for m in rows:
                graphs[i, m] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[i, m]):
                    outputs[i, m] = op(xs[m])
    workspaces = [w for op in ops for w in op._cute_workspaces.values()]
    assert len({w.partial.untyped_storage().data_ptr() for w in workspaces}) == 1
    assert all(w.out is None for w in workspaces) and group.nbytes > 0
    for _ in range(2):
        for key, graph in graphs.items():
            graph.replay()
            assert torch.equal(bits(outputs[key]), eager[key]), key
    graphs.clear()
    group.close()


def test_cute_graph_workspace_largest_first_ladder(monkeypatch):
    """The engine's ladder in one GraphWorkspace: the largest bucket runs Triton first, then a
    smaller CuTe bucket's eager call comes after another consumer's take froze the workspace
    at the first capture; its split-K slot was reserved by the earlier calls."""
    from ctikernels import GraphWorkspace
    from ctikernels.linear import _nvfp4_cute_small as cute

    if not cute.available():
        pytest.skip("requires DSL 4.8.0 and RTX 5090 / RTX PRO 6000 / GB10")
    torch.manual_seed(7)
    n, k = 96, 256
    plan = _nvfp4_dispatch.Cute(n_tile=64, k_tile=128, split_k=3, reduce_mode="global")
    small = _nvfp4_dispatch.Small(1, 32, 128, 4, 2, False)
    entries = ((16, "cute", plan), (64, "triton", small))
    monkeypatch.setitem(
        _nvfp4_dispatch.GATED, _nvfp4_dispatch._device_type(_device()), {(k, n): entries}
    )
    op = NVFP4GateUpSiLU(
        random_weight(n, k), random_weight(n, k),
        input_scale=torch.tensor([0.002], device=_device()),
        output_scale=torch.tensor([0.001], device=_device()),
    )  # fmt: skip
    rows = (64, 16)
    x = torch.randn(max(rows), k, device=_device(), dtype=torch.bfloat16)
    bits = lambda y: y.data.view(torch.uint8)
    other = ("other", x.device, torch.float32)
    graphs, outputs, eager = [], [], []
    with GraphWorkspace(max_rows=max(rows)) as group:
        group.reserve(
            other, (), lambda _: (1,), lambda counts: (torch.empty(counts[0], device=x.device),)
        )
        for m in rows:
            eager.append(bits(op(x[:m])).clone())
            assert (op._quantizer == "cute") == (m <= 16)
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1]):
                group.take(other, m, (1,))
                outputs.append(op(x[:m]))
    for graph, out, expected in zip(graphs, outputs, eager, strict=True):
        graph.replay()
        assert torch.equal(bits(out), expected)
    graphs.clear()
    group.close()


def _require_b12x():
    launcher = _require_flashinfer()
    if launcher.backend != "b12x":
        pytest.skip("NVFP4Linear runs FlashInfer b12x on compute capability 12.0 only")
    return launcher


def test_flashinfer_refuses_n_not_divisible_by_8():
    _require_flashinfer()
    weight, scale = random_weight(1004, 256), torch.tensor([0.002], device=_device())
    with pytest.raises(ValueError, match="does not support N=1004"):
        NVFP4Linear(weight, scale, provider="flashinfer")
    op = NVFP4Linear(weight, scale)
    assert op.provider != "flashinfer" and not isinstance(
        op.plan_for(1), _nvfp4_dispatch.FlashInfer
    )


def _empty_capture_allocations() -> int:
    """Allocations an empty CUDA graph capture counts (torch 2.13 counts +2 per capture, 2.14 none)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        pass
    return torch.cuda.memory_stats()["allocation.all.allocated"] - before


@pytest.mark.parametrize("n,k", [(12288, 5120), (1024, 5120)])  # wide and narrow-N tiles
def test_b12x_capture_at_unwarmed_m_compiles_and_allocates_nothing(n, k):
    from flashinfer.gemm import gemm_base

    _require_b12x()
    torch.manual_seed(0)
    input_scale = torch.tensor([0.002], device=_device())
    op = NVFP4Linear(random_weight(n, k), input_scale)
    assert op.provider == "flashinfer"
    op.prewarm(max_rows=4096)
    compiled = len(gemm_base._B12X_MM_FP4_KERNEL_CACHE)
    baseline = _empty_capture_allocations()
    for m in (3, 300, 3000):  # none of them prewarmed
        a = quantize_nvfp4(torch.randn(m, k, device=_device(), dtype=torch.bfloat16), input_scale)
        out = torch.empty(m, n, device=_device(), dtype=torch.bfloat16)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        with torch.cuda.graph(graph):
            op.quantized(a, out=out)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before + baseline, m
        assert len(gemm_base._B12X_MM_FP4_KERNEL_CACHE) == compiled, m
        graph.replay()
        assert torch.equal(out, op.quantized(a)), m


def test_b12x_keeps_the_cuda_stack_limit():
    """Fresh process: the first b12x GEMM leaves the per-thread stack limit at 1024 B."""
    import subprocess
    import sys

    _require_b12x()
    script = """
import torch
from ctikernels.linear.nvfp4 import NVFP4Linear
from tests.gpu.test_gdn_l2norm import _stack_limit
from tests.gpu.test_nvfp4 import random_weight

op = NVFP4Linear(random_weight(1024, 5120), torch.tensor([0.002], device="cuda"))
assert op.provider == "flashinfer"
before = _stack_limit()
op(torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()
print(before, _stack_limit())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.split()[-2:] == ["1024", "1024"], result.stdout
