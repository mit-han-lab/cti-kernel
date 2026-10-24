# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-FP8 gated-MLP input on the GPU: ``silu_and_mul_quantize_fp8`` and
``BlockFP8GateUpSiLU``.

``cute-act-quant`` is held to the unfused path bit for bit (the gate_up GEMM in the input
dtype, the package's ``silu_and_mul`` (SGLang's SiLU), ``quantize_per_token_group_fp8``): q
bytes and scales are equal, so the down projection's result is too. ``flashinfer-cute-dsl``
(where a qualified FlashInfer is installed) computes differently and is judged against an FP64
oracle on the same quantized operands, next to the unfused path: SiLU x up of the FP64 GEMMs,
compared with the dequantized output (q * scale) by RMS and maximum absolute error across eight
seeds.
"""

from __future__ import annotations

import math
import statistics

import pytest
import torch

from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import BlockFP8Linear, quantize_per_token_group_fp8
from ctikernels.linear.block_fp8_gated import (
    CUTE_ACT_QUANT,
    CUTE_FP8_FUSED,
    FLASHINFER_CUTE_DSL,
    HYBRID,
    BlockFP8GateUpSiLU,
    FusedConfig,
    silu_and_mul_quantize_fp8,
)

E4M3 = torch.float8_e4m3fn
# Merged gate_up shapes (I, K): Qwen3.5-35B-A3B-FP8's shared expert at TP 1, 2 and 4,
# Qwen3.5-27B-FP8's dense MLP at TP 4, and a tail that fills no 128 block.
SHAPES = ((2048, 2048), (512, 2048), (256, 2048), (128, 2048), (4352, 5120), (200, 328))
BATCHES = (1, 2, 4, 7, 16, 17, 31, 64, 129, 300, 1024, 2048)


def _groups(size: int) -> int:
    return -(-size // 128)


def _random_weight(n: int, k: int, scale_dtype=torch.float32):
    """E4M3 weights over the full range and log-uniform block scales over 2^-14..2^-4."""
    weight = (torch.randn(n, k, device="cuda") * 96).clamp(-448, 448).to(E4M3)
    exponent = torch.rand(_groups(n), _groups(k), device="cuda") * 10 - 14
    scale = torch.exp2(exponent) * (1 + torch.rand_like(exponent))
    return weight, scale.to(scale_dtype)


def _random_activation(m: int, k: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Rows over four decades, an outlier column and an all-zero group (finite in FP16)."""
    x = torch.randn(m, k, device="cuda") * torch.exp(torch.rand(m, 1, device="cuda") * 10 - 5)
    if k > 3:
        x[:, 3] *= 30
    if m > 1 and k >= 128:
        x[1, :128] = 0
    return x.to(dtype)


def _unfused(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_per_token_group_fp8(silu_and_mul(gate_up))


def _assert_same(actual, expected) -> None:
    (q, scale), (expected_q, expected_scale) = actual, expected
    assert q.shape == expected_q.shape and scale.shape == expected_scale.shape
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


def _cute_available() -> bool:
    from ctikernels.linear.cute._block_fp8_provider import available

    return torch.cuda.is_available() and available(torch.cuda.get_device_capability())


# The fused FC1 and the CuTe activation entry points exist only on sm120/sm121 (DSL 4.8.0).
requires_cute = pytest.mark.skipif(not _cute_available(), reason="CuTe FC1 needs sm120/sm121")


def _flashinfer_fc1_available(i: int = 512, k: int = 2048) -> bool:
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    from ctikernels.linear.flashinfer._block_fp8_gated import _flashinfer_fc1

    try:
        _flashinfer_fc1("probe", torch.cuda.get_device_capability(), i, k)
    except (RuntimeError, ValueError):
        return False
    return True


# ----------------------------------------------------------------------------- activation


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_activation_is_bitwise_unfused_on_every_gate_value(dtype) -> None:
    """Every BF16/FP16 bit pattern as a gate (NaN, Inf, subnormals included), 16 up values
    each over 20 binades: q and scale equal the two-launch path bit for bit, from both kernels
    (``act`` selects the Triton one, which also returns the rounded product)."""
    torch.manual_seed(1)
    gates = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).cuda()
    gates = gates.repeat(16).view(-1, 512)
    up = torch.randn(gates.shape, device="cuda") * torch.exp2(
        torch.randint(-10, 10, gates.shape, device="cuda").float()
    )
    gate_up = torch.cat([gates, up.to(dtype)], dim=1)
    expected = _unfused(gate_up)
    _assert_same(silu_and_mul_quantize_fp8(gate_up), expected)
    act = torch.empty_like(gates)
    _assert_same(silu_and_mul_quantize_fp8(gate_up, act=act), expected)
    assert torch.equal(act.view(torch.int16), silu_and_mul(gate_up).view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 256), (3, 1024), (33, 400), (300, 8704), (2048, 1024)])
@torch.inference_mode()
def test_activation_matches_the_unfused_path_bitwise(shape, dtype) -> None:
    torch.manual_seed(2)
    m, i = shape
    gate_up = (_random_activation(m, 2 * i).float() * 8).to(dtype)
    if m > 2:
        gate_up[2, i : i + 128] = 0  # a zero up group: zero values, a positive scale
    _assert_same(silu_and_mul_quantize_fp8(gate_up), _unfused(gate_up))
    # A strided view (gate_up as a slice of a wider buffer) reads only its own columns.
    wide = torch.cat([gate_up, torch.full_like(gate_up[:, :8], 1e4)], dim=1)
    _assert_same(silu_and_mul_quantize_fp8(wide[:, : 2 * i]), _unfused(gate_up))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@torch.inference_mode()
def test_nonfinite_groups_stay_nonfinite(bad) -> None:
    gate_up = torch.randn(3, 768, device="cuda", dtype=torch.bfloat16)
    gate_up[1, 200] = bad  # gate of group 1
    gate_up[2, 384 + 10] = bad  # up of group 0
    q, scale = silu_and_mul_quantize_fp8(gate_up)
    _assert_same((q, scale), _unfused(gate_up))
    assert not scale[1, 1].isfinite().item() and not scale[2, 0].isfinite().item()
    assert scale[0].isfinite().all().item()


def test_activation_checks_its_input() -> None:
    gate_up = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="\\[M, 2I\\]"):
        silu_and_mul_quantize_fp8(gate_up[:, :255])
    with pytest.raises(TypeError, match="BF16 or FP16"):
        silu_and_mul_quantize_fp8(gate_up.float())
    with pytest.raises(ValueError, match="unit column stride"):
        silu_and_mul_quantize_fp8(gate_up.t())
    with pytest.raises(RuntimeError, match="inference-only"):
        silu_and_mul_quantize_fp8(gate_up.float().requires_grad_().to(torch.bfloat16))
    q, scale = silu_and_mul_quantize_fp8(gate_up[:0])
    assert q.shape == (0, 128) and scale.shape == (0, 1)


# ----------------------------------------------------------------------------- gated FC1


def _unfused_fc1(op_weight, op_scale, x=None, q=None, scale=None, dtype=torch.bfloat16):
    gemm = BlockFP8Linear(op_weight, op_scale)
    gate_up = gemm(x) if x is not None else gemm.quantized(q, scale, out_dtype=dtype)
    return _unfused(gate_up)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("m", BATCHES)
@torch.inference_mode()
def test_gate_up_silu_matches_the_unfused_path_bitwise(m, shape) -> None:
    """Package GEMM dispatch materializes gate_up before the CuTe activation."""
    torch.manual_seed(4 + m)
    i, k = shape
    weight, scale = _random_weight(2 * i, k, scale_dtype=torch.bfloat16)
    op = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT)
    assert op.provider == CUTE_ACT_QUANT
    assert op.provider_for(m) == op.gate_up.provider_for(m) + "+" + CUTE_ACT_QUANT
    x = _random_activation(m, k)
    _assert_same(op(x), _unfused_fc1(weight, scale, x=x))
    q, s = quantize_per_token_group_fp8(x)
    _assert_same(op.quantized(q, s), _unfused_fc1(weight, scale, q=q, scale=s))


@pytest.mark.parametrize("m", [1, 16, 300])
@torch.inference_mode()
def test_fp16_inputs_and_lead_dimensions(m) -> None:
    torch.manual_seed(5)
    weight, scale = _random_weight(1024, 2048)
    x = _random_activation(m, 2048, dtype=torch.float16)
    op = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT)
    _assert_same(op(x.view(m, 1, 2048)), _unfused_fc1(weight, scale, x=x))
    q, s = quantize_per_token_group_fp8(x)
    expected = _unfused_fc1(weight, scale, q=q, scale=s, dtype=torch.float16)
    _assert_same(op.quantized(q, s, dtype=torch.float16), expected)


@torch.inference_mode()
def test_mlp_output_equals_the_unfused_mlp() -> None:
    torch.manual_seed(6)
    w13, s13 = _random_weight(1024, 2048, scale_dtype=torch.bfloat16)
    w2, s2 = _random_weight(2048, 512)
    fc1 = BlockFP8GateUpSiLU(w13, s13, provider=CUTE_ACT_QUANT)
    down = BlockFP8Linear(w2, s2)
    for m in (1, 4, 64, 1024):
        x = _random_activation(m, 2048)
        expected = down.quantized(*_unfused_fc1(w13, s13, x=x))
        assert torch.equal(down.quantized(*fc1(x)), expected)


@pytest.mark.parametrize("provider", [CUTE_ACT_QUANT, FLASHINFER_CUTE_DSL])
def test_gate_up_silu_replays_in_a_cuda_graph(provider) -> None:
    if provider == FLASHINFER_CUTE_DSL and not _flashinfer_fc1_available():
        pytest.skip("no qualified FlashInfer FC1 (FlashInfer 0.7.1, compute capability 12.x)")
    torch.manual_seed(7)
    weight, scale = _random_weight(1024, 2048)
    op = BlockFP8GateUpSiLU(weight, scale, provider=provider)
    for m in (4, 300, 2, 64):
        x = _random_activation(m, 2048)
        with torch.inference_mode():
            op.warm_up(batch_sizes=[m])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                q, s = op(x)
            x.copy_(_random_activation(m, 2048))
            graph.replay()
            torch.cuda.synchronize()
            _assert_same((q, s), op(x))


@torch.inference_mode()
def test_empty_batches_launch_nothing() -> None:
    weight, scale = _random_weight(512, 384)
    op = BlockFP8GateUpSiLU(weight, scale)
    q, s = op(torch.empty(2, 0, 384, device="cuda", dtype=torch.bfloat16))
    assert q.shape == (0, 256) and s.shape == (0, 2) and q.dtype == E4M3
    q, s = op.quantized(torch.empty(0, 384, device="cuda", dtype=E4M3), torch.empty(0, 3).cuda())
    assert q.shape == (0, 256) and s.shape == (0, 2)


def test_a_scale_set_to_require_grad_after_construction_raises_before_launch() -> None:
    weight, scale = _random_weight(512, 384)
    op = BlockFP8GateUpSiLU(weight, scale)
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    scale.requires_grad_()
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op(x)
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        op.quantized(*quantize_per_token_group_fp8(x))


def test_inputs_requiring_grad_raise_under_grad_mode() -> None:
    weight, scale = _random_weight(512, 384)
    op = BlockFP8GateUpSiLU(weight, scale)
    x = torch.randn(3, 384, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        op(x.to(torch.bfloat16))


# ----------------------------------------------------------------------------- FlashInfer FC1


def _oracle(q, s, weight, scale, i, swiglu_limit=None):
    k = q.shape[1]
    x64 = q.double() * s.double().repeat_interleave(128, -1)[:, :k]
    w64 = (
        weight.double()
        * scale.double().repeat_interleave(128, 0)[: 2 * i].repeat_interleave(128, 1)[:, :k]
    )
    gate_up = x64 @ w64.T
    gate, up = gate_up[:, :i], gate_up[:, i:]
    if swiglu_limit is not None:
        gate, up = gate.clamp(max=swiglu_limit), up.clamp(-swiglu_limit, swiglu_limit)
    return torch.nn.functional.silu(gate) * up


def _dequantize(q, s):
    i = q.shape[1]
    return q.double() * s.double().repeat_interleave(128, -1)[:, :i]


# Calibration: an out-of-tree seed-noise sweep (16 seeds, three devices), FlashInfer 0.7.1 /
# CUTLASS DSL 4.8.0; RTX 5090, RTX PRO 6000 and GB10, seeds 0..15, BF16 inputs/scales.
# (I,K)=(512,2048)/(256,2048)/(128,2048)/(4352,5120)/(200,328),
# M=1/2/4/7/16/64/300/1024. The tail has no FlashInfer provider: 32 eligible cells
# per device, 96 total. Ratios pair provider/unfused errors vs the same FP64 oracle.
# sigma is the LARGEST within-cell sample stddev (ddof=1) across all three devices:
#   RMS 0.050995984033: RTX PRO 6000, (128,2048), M=4;
#   max-abs 0.113318197049: GB10, (128,2048), M=16.
# No cell's mean - 1 exceeds 3*its_stddev/sqrt(16) for either metric. Devices do
# differ, so use the pooled worst-cell spread, not the older eight-seed GB10 data.
# Gate BOTH seed means at 1 + 3*sigma/sqrt(n_test_seeds); never gate per-seed error
# ratios by a finite sample's maximum deviation. This is an empirical three-SE
# noise budget, not a distribution-free or simultaneous confidence guarantee.
# Replay: seeds 0..7 and all eight other contiguous windows pass on every device
# (864/864 cell-windows). Windows overlap and reuse calibration data, not holdouts.
# (17408,5120), which the dispatch selects for FlashInfer at M=64 on RTX 5090 / RTX PRO 6000,
# was measured the same way afterwards (16 seeds, same M grid, three devices): largest
# within-cell stddev RMS 0.0047 / max-abs 0.0917, under the pooled sigma, and no cell's
# mean - 1 exceeds 3*its_stddev/sqrt(16); the pooled sigma applies unchanged.
# Keep eight seeds: 40 cases / 320 seed evaluations. The recorded full 16-seed,
# 40-cell harness took 11.8/17.1/19.9 s respectively (not pytest runtime).
_PRECISION_SEEDS = tuple(range(8))
_RMS_STDDEV = 0.050995984033
_MAX_ABS_STDDEV = 0.113318197049


@pytest.mark.parametrize(
    "shape", [(2048, 4096), (512, 2048), (256, 2048), (128, 2048), (4352, 5120), (17408, 5120)]
)
@pytest.mark.parametrize("m", [1, 2, 4, 7, 16, 64, 300, 1024])
@pytest.mark.parametrize(
    "provider,engine",
    [
        (CUTE_ACT_QUANT, None),
        (FLASHINFER_CUTE_DSL, None),
        (CUTE_FP8_FUSED, "fma"),
        (CUTE_FP8_FUSED, "mma"),
    ],
)
@torch.inference_mode()
def test_fc1_is_no_worse_than_the_unfused_path(m, shape, provider, engine) -> None:
    """No worse than unfused vs FP64 within the measured seed noise documented above."""
    i, k = shape
    if provider == CUTE_FP8_FUSED and m > 16:
        pytest.skip("fused-fc1 covers M <= 16")
    if (provider == CUTE_FP8_FUSED or engine is not None) and not _cute_available():
        pytest.skip("CuTe FC1 needs sm120/sm121")
    if provider == FLASHINFER_CUTE_DSL and not _flashinfer_fc1_available(i, k):
        pytest.skip("no qualified FlashInfer FC1 (FlashInfer 0.7.1, compute capability 12.x)")
    rms_ratios, max_abs_ratios = [], []
    for seed in _PRECISION_SEEDS:
        torch.manual_seed(seed)
        weight, scale = _random_weight(2 * i, k, scale_dtype=torch.bfloat16)
        op = BlockFP8GateUpSiLU(
            weight, scale, provider=provider, fused_config=_fused_config(engine, i)
        )
        assert op.provider == provider
        x = _random_activation(m, k)
        q, s = quantize_per_token_group_fp8(x)
        oracle = _oracle(q, s, weight, scale, i)
        fused_q, fused_s = op(x)
        if provider != CUTE_FP8_FUSED:
            _assert_same((fused_q, fused_s), op.quantized(q, s))
        unfused_q, unfused_s = _unfused_fc1(weight, scale, q=q, scale=s)
        fused_error = (_dequantize(fused_q, fused_s) - oracle).abs()
        unfused_error = (_dequantize(unfused_q, unfused_s) - oracle).abs()
        rms_ratio = (fused_error.square().mean() / unfused_error.square().mean()).sqrt().item()
        max_abs_ratio = (fused_error.max() / unfused_error.max()).item()
        assert math.isfinite(rms_ratio) and math.isfinite(max_abs_ratio), (seed, shape, m)
        rms_ratios.append(rms_ratio)
        max_abs_ratios.append(max_abs_ratio)
    # Eight-seed bounds: RMS 1.054089409184, max-abs 1.120192098348.
    assert statistics.mean(rms_ratios) <= 1 + 3 * _RMS_STDDEV / math.sqrt(len(_PRECISION_SEEDS)), (
        shape,
        m,
        "mean RMS ratio",
        statistics.mean(rms_ratios),
        rms_ratios,
    )
    assert statistics.mean(max_abs_ratios) <= 1 + 3 * _MAX_ABS_STDDEV / math.sqrt(
        len(_PRECISION_SEEDS)
    ), (shape, m, "mean max-abs ratio", statistics.mean(max_abs_ratios), max_abs_ratios)


# Every merged (I, H) the dispatch tables name, at every M they route below the package
# GEMM (small-M runs and the measured FlashInfer cells), under each option set: the DSV4
# clamped SwiGLU, UE8M0 (power-of-two weight and activation scales), and both. Dispatch does
# not depend on the options, so each fused config the tables can select runs under all four.
# Calibration (16 seeds, every case below; out of tree, 2026-10-16): on RTX 5090, RTX PRO 6000
# and GB10 every cell's mean ratio is within 1e-4 of 1, no cell's mean - 1 exceeds
# 3*its_stddev/4, and the largest within-cell stddev (RMS 1.8e-5, max-abs 0) is far inside
# the pooled sigma above, which therefore applies unchanged.
# With the pinned FlashInfer (< 0.7.1) the M=64 FlashInfer cell dispatches the unfused path;
# with FlashInfer >= 0.7.1 it is the (17408,5120) FlashInfer case calibrated above.
DISPATCH_OPTIONS = (
    {},
    {"swiglu_limit": 10.0},
    {"scale_ue8m0": True},
    {"swiglu_limit": 10.0, "scale_ue8m0": True},
)


def _dispatched_rows():
    from ctikernels.linear._block_fp8_gated_table import SMALL_M, TABLE

    rows = {(i, h): {1, 2, 3, 4} for _, i, h in SMALL_M}
    for (_, n, k), runs in TABLE.items():
        previous = 0
        for ceiling, name in runs:
            if name == FLASHINFER_CUTE_DSL:
                rows.setdefault((n // 2, k), {1, 2, 3, 4}).update(range(previous + 1, ceiling + 1))
            previous = ceiling
    return sorted((shape, tuple(sorted(ms))) for shape, ms in rows.items())


def _dispatched_error_ratios(shape, ms, options, seeds):
    """{M: ([RMS ratio], [max-abs ratio], {resolved provider})}: the default-dispatch op's
    error over the unfused path's error, both vs the FP64 oracle on the same quantized x."""
    from ctikernels.linear.triton import _block_fp8 as triton_kernels

    def quantize(x, scale_ue8m0):  # the UE8M0 input quantizer, bitwise CuTe's, on any device
        q = torch.empty(x.shape, dtype=E4M3, device=x.device)
        s = torch.empty(x.shape[0], _groups(x.shape[1]), device=x.device)
        triton_kernels.quantize(x, q, s, scale_ue8m0=scale_ue8m0)
        return q, s

    i, k = shape
    ue8m0 = options.get("scale_ue8m0", False)
    out = {m: ([], [], set()) for m in ms}
    for seed in seeds:
        torch.manual_seed(seed)
        weight, scale = _random_weight(2 * i, k, scale_dtype=torch.bfloat16)
        if ue8m0:
            scale = scale.float().to(torch.float8_e8m0fnu)
        op = BlockFP8GateUpSiLU(weight, scale, **options)
        unfused = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT, **options)
        for m in ms:
            x = _random_activation(m, k)
            q, s = quantize(x, scale_ue8m0=True) if ue8m0 else quantize_per_token_group_fp8(x)
            oracle = _oracle(q, s, weight, scale.float(), i, options.get("swiglu_limit"))
            fused_error = (_dequantize(*op(x)) - oracle).abs()
            unfused_error = (_dequantize(*unfused.quantized(q, s)) - oracle).abs()
            rms = (fused_error.square().mean() / unfused_error.square().mean()).sqrt().item()
            max_abs = (fused_error.max() / unfused_error.max()).item()
            assert math.isfinite(rms) and math.isfinite(max_abs), (seed, shape, m, options)
            out[m][0].append(rms)
            out[m][1].append(max_abs)
            out[m][2].add(op.provider_for(m))
    return out


@pytest.mark.parametrize("options", DISPATCH_OPTIONS, ids=["plain", "clamp", "ue8m0", "dsv4"])
@pytest.mark.parametrize("shape,ms", _dispatched_rows(), ids=lambda v: str(v))
@torch.inference_mode()
def test_dispatched_fc1_is_no_worse_than_the_unfused_path(shape, ms, options) -> None:
    """Whatever this device's dispatch selects (fused engine/config, FlashInfer, or the
    unfused GEMM + activation) meets the seed-noise gate above on RMS and max error."""
    i, k = shape
    if torch.cuda.mem_get_info()[0] < 2 * i * k * 30 + (1 << 30):
        pytest.skip("insufficient free memory for the FP64 oracle at this shape")
    bound_rms = 1 + 3 * _RMS_STDDEV / math.sqrt(len(_PRECISION_SEEDS))
    bound_max = 1 + 3 * _MAX_ABS_STDDEV / math.sqrt(len(_PRECISION_SEEDS))
    ratios = _dispatched_error_ratios(shape, ms, options, _PRECISION_SEEDS)
    for m, (rms, max_abs, providers) in ratios.items():
        assert statistics.mean(rms) <= bound_rms, (shape, m, options, providers, rms)
        assert statistics.mean(max_abs) <= bound_max, (shape, m, options, providers, max_abs)


@torch.inference_mode()
def test_default_dispatch_flashinfer_replays_in_a_cuda_graph() -> None:
    capability = torch.cuda.get_device_capability()
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if capability != (12, 0) or sms not in (170, 188):
        pytest.skip("measured FC1 dispatch cell is on RTX 5090 / RTX PRO 6000")
    i, k, m = 17408, 5120, 64
    if not _flashinfer_fc1_available(i, k):
        pytest.skip("no qualified FlashInfer FC1 (FlashInfer 0.7.1)")
    torch.manual_seed(7)
    weight, scale = _random_weight(2 * i, k, scale_dtype=torch.bfloat16)
    op = BlockFP8GateUpSiLU(weight, scale)
    if op.provider_for(m) == FLASHINFER_CUTE_DSL:
        assert op.provider in (HYBRID, CUTE_FP8_FUSED)
        explicit = BlockFP8GateUpSiLU(weight, scale, provider=FLASHINFER_CUTE_DSL)
        expected_fc1 = explicit
    else:
        assert op.provider_for(m) == "cute-fp8+cute-act-quant"
        expected_fc1 = lambda x: _unfused_fc1(weight, scale, x=x)
    down = BlockFP8Linear(*_random_weight(k, i))
    op.warm_up(batch_sizes=[m])
    x = _random_activation(m, k)
    _assert_same(op(x), expected_fc1(x))
    down.quantized(*op(x))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(batch_sizes=[m])
        q, scale = op(x)
        down.warm_up(
            batch_sizes=[m],
            quantized=True,
            input_layouts=[
                (
                    (q.stride(0), bool(q.data_ptr() % 16)),
                    (*scale.stride(), bool(scale.data_ptr() % 16)),
                )
            ],
        )
    with torch.cuda.graph(graph, stream=stream):
        q, s = op(x)
        y = down.quantized(q, s)
    x.copy_(_random_activation(m, k))
    graph.replay()
    torch.cuda.synchronize()
    expected = expected_fc1(x)
    _assert_same((q, s), expected)
    assert torch.equal(y, down.quantized(*expected))


def test_flashinfer_fc1_down_projection_reads_its_scale_view() -> None:
    if not _flashinfer_fc1_available():
        pytest.skip("no qualified FlashInfer FC1 (FlashInfer 0.7.1, compute capability 12.x)")
    torch.manual_seed(9)
    w13, s13 = _random_weight(1024, 2048)
    w2, s2 = _random_weight(2048, 512)
    op = BlockFP8GateUpSiLU(w13, s13, provider=FLASHINFER_CUTE_DSL)
    down = BlockFP8Linear(w2, s2, provider="triton-fp8")
    with torch.inference_mode():
        q, s = op(_random_activation(5, 2048))
        assert s.shape == (5, 4) and s.stride() == (1, 8)
        assert op._flashinfer.indptr(5, q.device).tolist() == [0, 5]
        assert torch.equal(down.quantized(q, s), down.quantized(q, s.contiguous()))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 16, 64, 256, 1000])
@pytest.mark.parametrize("i", [128, 200, 256, 512, 2048, 4352, 8704])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"swiglu_limit": 10.0},
        {"scale_ue8m0": True},
        {"swiglu_limit": 10.0, "scale_ue8m0": True},
    ],
)
@pytest.mark.parametrize("layout", ["k", "mn"])
@torch.inference_mode()
def test_cute_activation_matches_cpu_reference(m, i, options, layout):
    from ..modules._block_fp8_gated_reference import silu_quantize

    generator = torch.Generator().manual_seed(1234 + m)
    x = (torch.randn(m, 2 * i, generator=generator) * 20).bfloat16()
    expected = silu_quantize(x, **options)
    actual = silu_and_mul_quantize_fp8(x.cuda(), scale_layout=layout, **options)
    _assert_same(tuple(t.cpu() for t in actual), expected)
    if layout == "mn":
        assert actual[1].stride() == (1, -(-m // 4) * 4)


@pytest.mark.parametrize("layout", ["k", "mn"])
@torch.inference_mode()
def test_dsv4_ue8m0_weights_mlp_graph_replay(layout):
    from ..modules._block_fp8_gated_reference import silu_quantize

    w13, s13 = _random_weight(4096, 2048)
    w2, s2 = _random_weight(2048, 2048)
    s13 = s13.to(torch.float8_e8m0fnu)
    s2 = s2.to(torch.float8_e8m0fnu)
    fc1 = BlockFP8GateUpSiLU(
        w13, s13, provider=CUTE_ACT_QUANT, swiglu_limit=10, scale_ue8m0=True, scale_layout=layout
    )
    down = BlockFP8Linear(w2, s2, scale_ue8m0=True)
    x = _random_activation(3, 2048)
    fc1.warm_up(batch_sizes=[3])
    q, scale = fc1(x)
    down.warm_up(
        batch_sizes=[3],
        quantized=True,
        input_layouts=[
            ((q.stride(0), bool(q.data_ptr() % 16)), (*scale.stride(), bool(scale.data_ptr() % 16)))
        ],
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = fc1(x)
        y = down.quantized(q, s)
    x.copy_(_random_activation(3, 2048))
    graph.replay()
    expected = silu_quantize(fc1.gate_up(x).cpu(), swiglu_limit=10, scale_ue8m0=True)
    _assert_same((q.cpu(), s.cpu()), expected)
    assert torch.equal(y, down.quantized(*(t.cuda() for t in expected)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 3, 64, 1000])
@torch.inference_mode()
def test_dsv4_matches_sglang_materialized_clamp_and_ue8m0(dtype, m):
    """Optional external comparator; the audited revision is SGLang 72730d6."""
    dsv4 = pytest.importorskip("sglang.kernels.ops.moe.dsv4")
    quant = pytest.importorskip("sglang.kernels.ops.quantization.fp8_kernel")
    torch.manual_seed(91 + m)
    x = (torch.randn(m, 4096, device="cuda") * 20).to(dtype)
    product = torch.empty(m, 2048, dtype=dtype, device="cuda")
    dsv4.silu_and_mul_clamp(x, product, 10.0)
    expected = quant.sglang_per_token_group_quant_fp8(product, 128, scale_ue8m0=True)
    _assert_same(silu_and_mul_quantize_fp8(x, swiglu_limit=10, scale_ue8m0=True), expected)


# ----------------------------------------------------------------------------- fused FC1

FUSED_SHAPES = ((512, 2048), (256, 2048), (2048, 4096), (17408, 5120))
FUSED_OPTIONS = ({}, {"swiglu_limit": 10.0}, {"scale_ue8m0": True})


def _fused_config(engine, intermediate):
    if engine is None:
        return None
    from ctikernels.linear.cute._block_fp8_fused_provider import config_for

    return config_for(torch.cuda.get_device_capability(), intermediate, 1 if engine == "fma" else 2)


def _assert_one_fp8_step(actual, expected):
    """Compare dequantized values using adjacent E4M3 spacings at either endpoint.

    Scales can differ after BF16 rounding; the tolerance is in real output units,
    not a byte-distance between values quantized with different scales.
    """
    spacings = []
    for q, scale in (actual, expected):
        bits = q.view(torch.uint8).to(torch.int16) & 127
        assert (bits <= 126).all()
        lower = (bits - 1).clamp(0, 126).to(torch.uint8).view(E4M3).double()
        upper = (bits + 1).clamp(0, 126).to(torch.uint8).view(E4M3).double()
        value = q.double().abs()
        gap = torch.maximum(value - lower, upper - value)
        spacings.append(gap * scale.double().repeat_interleave(128, -1))
    reference = _dequantize(*expected)
    error = (_dequantize(*actual) - reference).abs()
    # FP32 summation order (GEMV vs GEMM, K up to 5120) moves gate/up by ~1e-5 of
    # the group maximum; near zero, SiLU x up scales that by |u| <= 10 with the
    # DSV4 clamp, far more than an E4M3 subnormal step (2^-9 of the scale,
    # ~2^-17.8 of the group maximum). Measured worst: 7e-5 of the maximum (RTX
    # 5090, I=17408, H=5120, M=11). 2^-12 of the 128-group maximum bounds it.
    group_max = reference.reshape(*reference.shape[:-1], -1, 128).abs().amax(-1)
    floor = group_max.repeat_interleave(128, -1) * 2.0**-12
    tolerance = torch.maximum(torch.maximum(*spacings), floor)
    worst = int((error / tolerance.clamp_min(1e-30)).argmax())
    assert torch.all(error <= tolerance + 1e-12), (
        (error / tolerance.clamp_min(1e-30)).max().item(),
        (error > tolerance + 1e-12).sum().item(),
        reference.flatten()[worst].item(),
        _dequantize(*actual).flatten()[worst].item(),
        group_max.repeat_interleave(128, -1).flatten()[worst].item(),
        worst,
    )


@requires_cute
@pytest.mark.parametrize("m", range(1, 17))
@pytest.mark.parametrize("shape", FUSED_SHAPES)
@pytest.mark.parametrize("options", FUSED_OPTIONS)
@pytest.mark.parametrize("engine", ["fma", "mma"])
@torch.inference_mode()
def test_fused_fc1_quantization_and_epilogue(m, shape, options, engine):
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    i, k = shape
    # Avoid OOM on smaller qualification devices; no shape is silently substituted.
    if torch.cuda.mem_get_info()[0] < 2 * i * k * 12 + (256 << 20):
        pytest.skip("insufficient free memory for this fused/unfused shape")
    torch.manual_seed(310 + m)
    weight, scale = _random_weight(2 * i, k)
    op = BlockFP8GateUpSiLU(
        weight,
        scale,
        provider=CUTE_FP8_FUSED,
        fused_config=_fused_config(engine, weight.shape[0] // 2),
        **options,
    )
    baseline = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT, **options)
    x = _random_activation(m, k)
    actual = op(x)
    q, s, qx, sx, gate_up = op._fused_provider().run(x, debug=True)
    _assert_same(actual, (q, s))
    _assert_same((qx, sx), quantize(x, scale_ue8m0=options.get("scale_ue8m0", False)))
    if not options.get("scale_ue8m0", False):
        _assert_same((qx, sx), quantize_per_token_group_fp8(x))
    # Exact scale bits as well as q bytes, given the same BF16 gate/up.
    _assert_same(actual, silu_and_mul_quantize_fp8(gate_up, **options))
    _assert_one_fp8_step(actual, baseline(x))
    assert torch.count_nonzero(op._fused._scratch[1]) == 0
    # Consecutive calls exercise reset without a host read between launches.
    outputs = [op(x) for _ in range(3)]
    for output in outputs:
        _assert_same(output, actual)


@requires_cute
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["k", "mn"])
@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("engine", ["fma", "mma"])
@torch.inference_mode()
def test_fused_fc1_strided_input_and_scale_layout(dtype, layout, ue8m0, engine):
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    weight, scale = _random_weight(1024, 2048)
    # Nonzero pointer offset and odd row stride; M=3 shares the M_max=4 bucket.
    x = torch.randn(3, 2051, device="cuda", dtype=dtype)[:, 1:2049]
    options = {"scale_layout": layout, "scale_ue8m0": ue8m0, "swiglu_limit": 10.0}
    op = BlockFP8GateUpSiLU(
        weight,
        scale,
        provider=CUTE_FP8_FUSED,
        fused_config=_fused_config(engine, weight.shape[0] // 2),
        **options,
    )
    q, s, qx, sx, gu = op._fused_provider().run(x, debug=True)
    _assert_same((qx, sx), quantize(x, scale_ue8m0=ue8m0))
    _assert_same((q, s), silu_and_mul_quantize_fp8(gu, **options))
    assert q.stride() == (512, 1)
    assert s.stride() == ((1, 4) if layout == "mn" else (4, 1))
    # Explicit BF16 GEMM boundary, even when x is FP16.
    reference_gu = op.gate_up.quantized(qx, sx, out_dtype=torch.bfloat16)
    _assert_one_fp8_step((q, s), silu_and_mul_quantize_fp8(reference_gu, **options))
    down = BlockFP8Linear(*_random_weight(2048, 512))
    assert torch.equal(down.quantized(q, s), down.quantized(q, s.contiguous()))


@requires_cute
@pytest.mark.parametrize(
    "config",
    [FusedConfig(r=r, warps=w, engine="fma") for r in (2, 4, 8, 16) for w in (4, 8)]
    + [
        FusedConfig(engine="mma", warps_n=wn, warps_k=wk)
        for wn in (1, 2, 4)
        for wk in (1, 2, 4, 8)
        if wn * wk <= 16
    ],
)
@pytest.mark.parametrize("k", [128, 640])
@torch.inference_mode()
def test_fused_fc1_all_configs_and_short_k(config, k):
    # H=128 masks the first step; H=640 exercises an incomplete final step.
    weight, scale = _random_weight(256, k)
    x = _random_activation(16, k)
    op = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_FP8_FUSED, fused_config=config)
    q, s, _, _, gu = op._fused_provider().run(x, debug=True)
    _assert_same((q, s), silu_and_mul_quantize_fp8(gu))
    _assert_one_fp8_step((q, s), _unfused_fc1(weight, scale, x=x))


@requires_cute
@pytest.mark.parametrize("layout", ["k", "mn"])
@pytest.mark.parametrize("options", [{}, {"swiglu_limit": 10.0, "scale_ue8m0": True}])
@pytest.mark.parametrize("engine", [None, "fma", "mma"])
@torch.inference_mode()
def test_fused_fc1_graphs_replay_changing_inputs_and_m(layout, options, engine):
    weight, scale = _random_weight(1024, 2048)
    if options.get("scale_ue8m0", False):
        scale = scale.to(torch.float8_e8m0fnu)
    op = BlockFP8GateUpSiLU(
        weight,
        scale,
        provider=CUTE_FP8_FUSED,
        scale_layout=layout,
        fused_config=_fused_config(engine, 512),
        **options,
    )
    down = BlockFP8Linear(*_random_weight(2048, 512))
    captures = []
    for m in (1, 3, 4, 7, 16, 2):
        x = _random_activation(m, 2048)
        op.warm_up(batch_sizes=[m])
        q, scale = op(x)
        down.warm_up(
            batch_sizes=[m],
            quantized=True,
            input_layouts=[
                (
                    (q.stride(0), bool(q.data_ptr() % 16)),
                    (*scale.stride(), bool(scale.data_ptr() % 16)),
                )
            ],
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, s = op(x)
            y = down.quantized(q, s)
        captures.append((graph, x, q, s, y))
    scratch = op._fused._scratch
    # Alternating separately captured graphs exercises one shared workspace with
    # different live M values, including M=3/4 using the same compiled signature.
    for _ in range(3):
        for graph, x, q, s, y in reversed(captures):
            x.copy_(_random_activation(x.shape[0], 2048))
            graph.replay()
            expected = op(x)
            _assert_same((q, s), expected)
            assert torch.equal(y, down.quantized(*expected))
    assert op._fused._scratch[0] is scratch[0]
    assert op._fused._scratch[1] is scratch[1]
    assert torch.count_nonzero(scratch[1]) == 0


@requires_cute
@pytest.mark.parametrize("frozen", ["weight", "scale"])
@pytest.mark.parametrize("engine", ["fma", "mma"])
def test_fused_fc1_grad_rejected_before_provider_launch(monkeypatch, frozen, engine):
    weight, scale = _random_weight(512, 2048)
    op = BlockFP8GateUpSiLU(
        weight, scale, provider=CUTE_FP8_FUSED, fused_config=_fused_config(engine, 256)
    )
    x = torch.randn(3, 2048, device="cuda", dtype=torch.bfloat16)
    op(x)

    # This is the last point before compile/allocation/launch, so no launch can
    # hide behind the expected exception (including under inference_mode).
    def no_launch(*args, **kwargs):
        pytest.fail("frozen source check must precede provider launch")

    monkeypatch.setattr(op._fused, "run", no_launch)
    (weight if frozen == "weight" else scale).requires_grad_()
    with torch.inference_mode(), pytest.raises(ValueError, match="is frozen"):
        op(x)
    with pytest.raises(ValueError, match="is frozen"):
        op.quantized(None, None)


@torch.inference_mode()
def test_small_m_resolved_provider_and_config_are_launched(monkeypatch):
    from ctikernels.linear import _block_fp8_gated_dispatch as dispatch
    from ctikernels.linear import block_fp8_gated as gated
    from ctikernels.linear.cute._block_fp8_fused_provider import FusedFC1

    capability = torch.cuda.get_device_capability()
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    device = dispatch.DEVICES.get((*capability, sms))
    if device is None:
        pytest.skip("small-M table targets RTX 5090, RTX PRO 6000 and GB10")
    i, h = 512, 2048  # One listed shape per device run.
    runs = dispatch.SMALL_M[(device, i, h)]
    op = BlockFP8GateUpSiLU(*_random_weight(2 * i, h))
    launched = []
    fused_run = FusedFC1.run
    activation = gated.silu_and_mul_quantize_fp8
    flash_run = op._run_flashinfer

    def record_fused(self, x, **kwargs):
        result = fused_run(self, x, **kwargs)
        launched.append((CUTE_FP8_FUSED, dispatch.config_spec(self.config_for(x.shape[0]))))
        return result

    def record_activation(*args, **kwargs):
        result = activation(*args, **kwargs)
        launched.append((CUTE_ACT_QUANT, None))
        return result

    def record_flash(*args):
        result = flash_run(*args)
        launched.append((FLASHINFER_CUTE_DSL, None))
        return result

    monkeypatch.setattr(FusedFC1, "run", record_fused)
    monkeypatch.setattr(gated, "silu_and_mul_quantize_fp8", record_activation)
    monkeypatch.setattr(op, "_run_flashinfer", record_flash)
    for m in range(1, runs[-1][0] + 2):
        expected = op.resolve(m)
        q, scale = op(_random_activation(m, h))
        torch.cuda.synchronize()
        assert launched == [expected]
        assert q.shape == (m, i) and scale.shape == (m, i // 128)
        launched.clear()


@pytest.mark.parametrize(
    "provider,i,k",
    [
        (CUTE_FP8_FUSED, 512, 2048),
        (FLASHINFER_CUTE_DSL, 512, 2048),
        (CUTE_ACT_QUANT, 512, 2048),
        (None, 512, 2048),
        (None, 17408, 5120),  # measured large FC1 cells, including M=64
    ],
)
@torch.inference_mode()
def test_gated_warm_up_captures_every_intermediate_size(provider, i, k, monkeypatch):
    """Fresh instance, only warm_up before all captures, then a fresh reference."""
    from ctikernels.linear.cute import _block_fp8_fused_provider as fused
    from ctikernels.linear.cute._block_fp8_provider import available

    if not torch.cuda.is_available() or not available(torch.cuda.get_device_capability()):
        pytest.skip("requires sm120/sm121 and CuTe DSL 4.8.0")
    if provider == FLASHINFER_CUTE_DSL and not _flashinfer_fc1_available():
        pytest.skip("no qualified FlashInfer FC1")
    # Covers fused 1/2/4/8/16; default dispatch also crosses into the GEMM path.
    max_m = 16 if provider == CUTE_FP8_FUSED else 64
    weight, scale = _random_weight(2 * i, k)
    op = BlockFP8GateUpSiLU(weight, scale, provider=provider)
    monkeypatch.setattr(fused, "_ready", set())  # prior tests cannot hide a missing bucket
    x = _random_activation(max_m, k)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(max_m=max_m)
    torch.cuda.current_stream().wait_stream(stream)
    if provider == FLASHINFER_CUTE_DSL:
        assert set(op._flashinfer._indptr) == set(range(1, max_m + 1))
    sizes = range(1, max_m + 1) if i == 512 else (1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64)
    snapshots = []
    for m in sizes:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            q, s = op(x[:m])
        graph.replay()
        graph.replay()
        snapshots.append((q.cpu(), s.cpu()))
    reference = BlockFP8GateUpSiLU(weight, scale, provider=provider)
    for m, (q, s) in zip(sizes, snapshots, strict=True):
        expected_q, expected_s = reference(x[:m])
        assert torch.equal(q.view(torch.uint8), expected_q.cpu().view(torch.uint8)), m
        assert torch.equal(s.view(torch.int32), expected_s.cpu().view(torch.int32)), m


@pytest.mark.parametrize("provider", [CUTE_ACT_QUANT, CUTE_FP8_FUSED, FLASHINFER_CUTE_DSL])
@torch.inference_mode()
def test_engine_workspace_outputs_survive_reuse_and_scratch_capture(provider, monkeypatch):
    if provider == FLASHINFER_CUTE_DSL and not _flashinfer_fc1_available():
        pytest.skip("no qualified FlashInfer FC1")
    if provider == CUTE_FP8_FUSED and not _cute_available():
        pytest.skip("CuTe FC1 needs sm120/sm121")
    op = BlockFP8GateUpSiLU(*_random_weight(1024, 2048), provider=provider, scale_layout="mn")
    workspace = torch.empty(op.workspace_bytes(320) // 4, dtype=torch.float32, device="cuda")
    for dtype in (torch.bfloat16, torch.float16):
        for m in (1, 7, 17, 320):
            x = _random_activation(m, 2048, dtype)
            q, scale = quantize_per_token_group_fp8(x)
            for quantized in (False, True):

                def call(buffer=None, *, quantized=quantized, q=q, scale=scale, x=x, dtype=dtype):
                    return (
                        op.quantized(q, scale, dtype=dtype, workspace=buffer)
                        if quantized
                        else op(x, workspace=buffer)
                    )

                expected = tuple(t.clone() for t in call())
                first = call(workspace)
                saved = tuple(t.clone() for t in first)
                _assert_same(first, expected)
                for tensor in first:
                    assert (
                        tensor.untyped_storage().data_ptr()
                        != workspace.untyped_storage().data_ptr()
                    )
                if quantized:
                    second = op.quantized(
                        torch.zeros_like(q), scale, dtype=dtype, workspace=workspace
                    )
                else:
                    second = op(torch.zeros_like(x), workspace=workspace)
                _assert_same(first, saved)
                for a, b in zip(first, second, strict=True):
                    assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                graph = torch.cuda.CUDAGraph()
                from ctikernels.linear._fp8_workspace import Views

                get = Views.get
                scratch = []

                def observe(views, name, dtype=None, *, get=get, scratch=scratch):
                    tensor = get(views, name, dtype)
                    scratch.append(tensor.untyped_storage().data_ptr())
                    return tensor

                # Gated has no out= entry: its two fresh outputs allocate by design.
                # Observe scratch ownership separately from output allocations.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    op.warm_up(batch_sizes=[m], dtype=dtype, quantized=quantized)
                    call(workspace)
                    with monkeypatch.context() as patch:
                        patch.setattr(Views, "get", observe)
                        with torch.cuda.graph(graph, stream=stream):
                            result = call(workspace)
                torch.cuda.current_stream().wait_stream(stream)
                assert scratch and all(pointer == workspace.data_ptr() for pointer in scratch)
                for tensor in result:
                    assert tensor.untyped_storage().data_ptr() != workspace.data_ptr()
                workspace.fill_(float("nan"))
                graph.replay()
                _assert_same(result, expected)


@pytest.mark.parametrize("layout", ["packed", "row-padded", "misaligned", "ue8m0-scalar"])
@torch.inference_mode()
def test_gated_warm_up_cold_capture_input_layouts(monkeypatch, layout):
    from ctikernels.linear.cute._block_fp8_provider import available

    from .test_block_fp8 import _track_cold_triton_launches

    if not torch.cuda.is_available() or not available(torch.cuda.get_device_capability()):
        pytest.skip("requires sm120/sm121 and CuTe DSL 4.8.0")
    ue8m0 = layout == "ue8m0-scalar"
    weight, scale = _random_weight(512, 384)
    op = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT, scale_ue8m0=ue8m0)
    _track_cold_triton_launches(monkeypatch)
    x = _random_activation(17, 384 if layout == "packed" else 385)
    x = x[:, int(layout in ("misaligned", "ue8m0-scalar")) :][:, :384]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(batch_sizes=[17])
    if layout != "packed":
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(RuntimeError, match="input layout class.*warm_up.*input_layouts"):
                op(x)
        with torch.cuda.stream(stream):
            op.warm_up(batch_sizes=[17], input_layouts=[(x.stride(0), bool(x.data_ptr() % 16))])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = op(x)
    graph.replay()
    torch.cuda.synchronize()
    reference = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT, scale_ue8m0=ue8m0)
    _assert_same(actual, reference(x))


@pytest.mark.parametrize("layout", ["packed", "strided"])
@torch.inference_mode()
def test_gated_warm_up_cold_capture_quantized_layouts(monkeypatch, layout):
    from ctikernels.linear.cute._block_fp8_provider import available

    from .test_block_fp8 import _track_cold_triton_launches

    if not torch.cuda.is_available() or not available(torch.cuda.get_device_capability()):
        pytest.skip("requires sm120/sm121 and CuTe DSL 4.8.0")
    weight, scale = _random_weight(512, 384)
    op = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT)
    _track_cold_triton_launches(monkeypatch)
    q = _random_activation(17, 384 if layout == "packed" else 385).to(E4M3)[:, -384:]
    scales = (
        torch.ones(17, 3, device="cuda")
        if layout == "packed"
        else torch.ones(17, 7, device="cuda")[:, 1::2]
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(batch_sizes=[17], quantized=True)
    if layout != "packed":
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(
                RuntimeError, match="input layout class.*scale strides=.*warm_up.*input_layouts"
            ):
                op.quantized(q, scales)
        with torch.cuda.stream(stream):
            op.warm_up(
                batch_sizes=[17], quantized=True, input_layouts=[((385, True), (7, 2, True))]
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = op.quantized(q, scales)
    graph.replay()
    torch.cuda.synchronize()
    reference = BlockFP8GateUpSiLU(weight, scale, provider=CUTE_ACT_QUANT)
    _assert_same(actual, reference.quantized(q, scales))


# ----------------------------------------------------------------------------- Triton act-quant


def _triton_silu_quantize(x, layout="k", **options):
    from ctikernels.linear.cute._block_fp8_act_quant import allocate_outputs
    from ctikernels.linear.triton import _block_fp8 as kernel

    q, scale = allocate_outputs(x.shape[0], x.shape[1] // 2, x.device, layout)
    kernel.silu_quantize(x, q, scale, **options)
    return q, scale


@pytest.mark.parametrize("m", [0, 1, 3, 300])
@pytest.mark.parametrize("i", [128, 200, 2048, 8704])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"swiglu_limit": 10.0},
        {"scale_ue8m0": True},
        {"swiglu_limit": 10.0, "scale_ue8m0": True},
    ],
)
@pytest.mark.parametrize("layout", ["k", "mn"])
@torch.inference_mode()
def test_triton_activation_matches_cpu_reference(m, i, options, layout):
    """Our Triton SiLU*up + 1x128 quantizer (every compute capability 8.0+) equals the CPU
    oracle bit for bit, in both scale layouts and all DSV4 options."""
    from ..modules._block_fp8_gated_reference import silu_quantize

    generator = torch.Generator().manual_seed(4321 + m + i)
    x = (torch.randn(m, 2 * i, generator=generator) * 20).bfloat16()
    expected = silu_quantize(x, **options)
    actual = _triton_silu_quantize(x.cuda(), layout, **options)
    _assert_same(tuple(t.cpu() for t in actual), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("options", [{}, {"swiglu_limit": 7.0, "scale_ue8m0": True}])
@torch.inference_mode()
def test_triton_activation_equals_the_package_path_on_every_gate_value(dtype, options) -> None:
    """Every BF16/FP16 bit pattern as a gate (NaN, Inf, subnormals): Triton gives the bits of
    ``silu_and_mul_quantize_fp8`` (CuTe on sm12x, the same Triton kernel elsewhere) and, for
    the plain options, of the unfused two-launch path."""
    torch.manual_seed(5)
    gates = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).cuda()
    gates = gates.repeat(4).view(-1, 512)
    up = torch.randn(gates.shape, device="cuda") * torch.exp2(
        torch.randint(-10, 10, gates.shape, device="cuda").float()
    )
    gate_up = torch.cat([gates, up.to(dtype)], dim=1)
    actual = _triton_silu_quantize(gate_up, **options)
    _assert_same(actual, silu_and_mul_quantize_fp8(gate_up, **options))
    if not options:
        _assert_same(actual, _unfused(gate_up))
