# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Block-FP8 (W8A8) inference on the GPU: the per-token-group quantizer, weight decoding, the
dense ``BlockFP8Linear`` and the grouped MoE GEMM.

Each GEMM is judged on the same quantized operands against an FP64 oracle with real scale
decoding, next to three comparators on those operands: the FP32 floor (FP32 GEMM of the
decoded operands, TF32 off, one rounding), the dequant-to-BF16 baseline (operands rounded to
BF16, BF16 GEMM with FP32 accumulation) and, where it runs, the market path (FlashInfer's
groupwise CUTLASS GEMM). A path passes when its RMS error is at most 1.05 times the best
comparator's (plus 1e-30) and no element is further from the oracle than that comparator's
element plus one BF16 ULP of the oracle value plus 4 FP32 units of roundoff of the sum of
absolute products (FP32 summation orders differ, and cancellation can make that sum much
larger than the result).
"""

from __future__ import annotations

import math

import pytest
import torch

from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import (
    HYBRID,
    BlockFP8Linear,
    default_provider,
    dequantize_block_fp8,
    quantize_per_token_group_fp8,
    slice_block_fp8,
)
from ctikernels.linear.cute._block_fp8_provider import available as cute_available
from ctikernels.linear.triton import _block_fp8
from ctikernels.moe_impl import (
    block_fp8_moe_config,
    fused_moe_kernel_triton,
    moe_align_block_size,
    moe_sum_reduce_triton,
)

from ..modules._moe_reference import combine_reference

E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
PROVIDERS = ("triton-fp8",)
# The local CUDA GEMM of compute capability 12.x (packed N and K multiples of 128).
SM12X = (
    ("cuda-sm12x",)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
    else ()
)
# None: the default, the measured dispatch between the GEMMs on the qualified devices.
DENSE_PROVIDERS = (*PROVIDERS, *SM12X, "cute-fp8", None)
# Projection shapes (N, K) of Qwen3.5-35B-A3B-FP8: merged q/k/v/gate, attention/GDN output,
# merged GDN qkvz, shared-expert gate_up and down; and tails that do not fill a 128 block.
DENSE_SHAPES = ((9216, 2048), (2048, 4096), (12288, 2048), (1024, 2048), (2048, 512))
TAIL_SHAPES = ((200, 328), (1000, 1000), (136, 72))


@pytest.fixture(autouse=True)
def _skip_unavailable_cute(request):
    if getattr(request.node, "callspec", None) is not None:
        if request.node.callspec.params.get("provider") == "cute-fp8":
            _require_cute()


def _require_cute():
    if not torch.cuda.is_available() or not cute_available(torch.cuda.get_device_capability()):
        pytest.skip("cute-fp8 requires sm_120/sm_121 and CuTe DSL 4.8.0")


@pytest.fixture(autouse=True)
def _exact_fp32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _groups(size: int) -> int:
    return -(-size // 128)


def _reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The quantizer recipe in torch FP32 (independent of the Triton kernels)."""
    m, k = x.shape
    padded = torch.zeros(m, _groups(k) * 128, device=x.device, dtype=torch.float32)
    padded[:, :k] = x.float()
    grouped = padded.view(m, -1, 128)
    amax = grouped.abs().amax(-1).clamp_min(1e-10)
    scale = amax * torch.tensor(1 / 448, dtype=torch.float32)
    multiplier = torch.full_like(amax, 448.0) / amax
    q = (grouped * multiplier[..., None]).clamp(-448, 448).to(E4M3)
    return q.view(m, -1)[:, :k].contiguous(), scale


def _activation64(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    k = q.shape[-1]
    return q.double() * scale.double().repeat_interleave(128, -1)[..., :k]


def _weight64(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape[-2:]
    expanded = scale.double().repeat_interleave(128, -2)[..., :n, :]
    return weight.double() * expanded.repeat_interleave(128, -1)[..., :k]


def _random_weight(n: int, k: int, *, experts: int | None = None, scale_dtype=torch.float32):
    """E4M3 weights over the full range and log-uniform block scales over 2^-14..2^-4."""
    lead = () if experts is None else (experts,)
    weight = (torch.randn(*lead, n, k, device="cuda") * 96).clamp(-448, 448).to(E4M3)
    exponent = torch.rand(*lead, _groups(n), _groups(k), device="cuda") * 10 - 14
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


def _ulp_bf16(value: torch.Tensor) -> torch.Tensor:
    magnitude = value.abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


def _errors(out: torch.Tensor, oracle: torch.Tensor) -> torch.Tensor:
    return (out.double() - oracle).abs()


def _rms(error: torch.Tensor) -> float:
    return error.square().mean().sqrt().item() if error.numel() else 0.0


def _assert_no_worse(
    out: torch.Tensor, oracle: torch.Tensor, comparators: dict, magnitude: torch.Tensor
) -> None:
    """``out`` against the oracle versus the best of ``comparators`` (name -> output);
    ``magnitude`` is the oracle's sum of absolute products."""
    if out.numel() == 0:
        return
    errors = _errors(out, oracle)
    scored = {name: _errors(value, oracle) for name, value in comparators.items()}
    best = min(scored, key=lambda name: _rms(scored[name]))
    ratio = _rms(errors) / max(_rms(scored[best]), 1e-30)
    rms = {name: _rms(error) for name, error in scored.items()}
    assert _rms(errors) <= 1.05 * _rms(scored[best]) + 1e-30, (_rms(errors), rms, ratio)
    excess = errors - scored[best] - _ulp_bf16(oracle) - 2.0**-22 * magnitude
    assert excess.max().item() <= 0, (excess.max().item(), best, rms)


def _bf16_baseline(x64: torch.Tensor, w64: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x64.to(torch.bfloat16), w64.to(torch.bfloat16))


def _flashinfer(q, scale, weight, weight_scale):
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    return gemm_fp8_nt_groupwise(
        q,
        weight,
        # FlashInfer requires packed K-major scales; CuTe tests deliberately
        # pass strided scales to the kernel under test, not to this comparator.
        scale.contiguous(),
        weight_scale.float().contiguous(),
        scale_major_mode="K",
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )


def _market_available(n: int, k: int) -> bool:
    return torch.cuda.get_device_capability()[0] in (10, 12) and n % 128 == 0 and k % 128 == 0


# ----------------------------------------------------------------------------- quantizer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 2048), (7, 4096), (33, 328), (129, 512), (300, 2048)])
@torch.inference_mode()
def test_quantizer_matches_its_recipe_and_sgl_kernel_bitwise(shape, dtype) -> None:
    torch.manual_seed(1)
    x = _random_activation(*shape, dtype=dtype)
    q, scale = quantize_per_token_group_fp8(x)
    expected_q, expected_scale = _reference_quantize(x)
    assert q.is_contiguous() and scale.is_contiguous()
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale, expected_scale)
    if shape[0] > 1:
        assert torch.equal(q[1, :128].float(), torch.zeros(128, device="cuda"))
        assert (scale[1, 0] > 0).item()
    if shape[1] % 128 == 0:
        from sgl_kernel import sgl_per_token_group_quant_fp8

        provider_q, provider_scale = torch.empty_like(q), torch.empty_like(scale)
        sgl_per_token_group_quant_fp8(x, provider_q, provider_scale, 128, 1e-10, -448.0, 448.0)
        assert torch.equal(q.view(torch.uint8), provider_q.view(torch.uint8))
        assert torch.equal(scale, provider_scale)
    # A strided view (a merged projection's slice) reads only its own columns.
    wide = torch.cat([x, torch.full_like(x[:, :5], 1e4)], dim=1)
    q_view, scale_view = quantize_per_token_group_fp8(wide[:, : shape[1]])
    assert torch.equal(q_view.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(scale_view, scale)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@torch.inference_mode()
def test_nonfinite_groups_stay_nonfinite(bad: float) -> None:
    x = torch.randn(3, 384, device="cuda", dtype=torch.bfloat16)
    x[1, 200] = bad
    q, scale = torch.empty(3, 384, device="cuda", dtype=E4M3), torch.empty(3, 3, device="cuda")
    _block_fp8.quantize(x, q, scale)
    expected_q, expected_scale = _reference_quantize(x.masked_fill(~x.isfinite(), 0))
    good = torch.ones(3, 3, dtype=torch.bool, device="cuda")
    good[1, 1] = False
    assert torch.equal(scale[good], expected_scale[good])
    assert not _activation64(q, scale)[1, 128:256].isfinite().all().item()
    weight, weight_scale = _random_weight(256, 384)
    for provider in (*PROVIDERS, *SM12X):
        y = BlockFP8Linear(weight, weight_scale, provider=provider)(x)
        assert not y[1].isfinite().any().item()
        assert y[[0, 2]].isfinite().all().item()


def _quantizer_tiles() -> list[tuple[int, int]]:
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    tiles = {(4, 4), (8, 4), (16, 4), (1, 1), (64, 16)}
    tiles.update(tile for runs in dispatch.QUANTIZER.values() for _, tile in runs)
    return sorted(tiles)


@pytest.mark.parametrize("shape", [(1, 2048), (33, 328), (300, 2048), (2051, 512)])
@torch.inference_mode()
def test_every_quantizer_tile_gives_the_same_bits(shape) -> None:
    """Any tile (rows per program, warps), measured or not, writes the default's bits,
    non-finite groups and partial groups and row blocks included, whether the operands run
    as one group per kernel row (contiguous, whole groups) or as [M, K] (a strided view)."""
    torch.manual_seed(3)
    x = _random_activation(*shape, dtype=torch.bfloat16)
    m, k = shape
    x[m // 2, k - 1] = math.nan
    x[m - 1, 0] = -math.inf
    expected_q = torch.empty(m, k, device="cuda", dtype=E4M3)
    expected_scale = torch.empty(m, _groups(k), device="cuda")
    _block_fp8.quantize(x, expected_q, expected_scale)
    finite_q, finite_scale = _reference_quantize(x)
    finite = x.isfinite().all(-1)
    assert torch.equal(expected_q[finite].view(torch.uint8), finite_q[finite].view(torch.uint8))
    assert torch.equal(expected_scale[finite], finite_scale[finite])
    strided = torch.cat([x, x[:, :128]], dim=1)[:, :k]
    for tile in _quantizer_tiles():
        for source in (x, strided):
            q = torch.full_like(expected_q, 1.0)
            scale = torch.full_like(expected_scale, -1.0)
            _block_fp8.quantize(source, q, scale, tile=tile)
            assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8)), tile
            assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32)), tile


@torch.inference_mode()
def test_empty_batches_launch_nothing() -> None:
    weight, scale = _random_weight(256, 384)
    q, s = quantize_per_token_group_fp8(torch.empty(0, 384, device="cuda", dtype=torch.bfloat16))
    assert q.shape == (0, 384) and s.shape == (0, 3)
    op = BlockFP8Linear(weight, scale)
    assert op(torch.empty(2, 0, 384, device="cuda", dtype=torch.bfloat16)).shape == (2, 0, 256)
    assert op.quantized(q, s).shape == (0, 256)


# ----------------------------------------------------------------------------- decoding


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16, E8M0])
@torch.inference_mode()
def test_dequantize_rounds_the_fp32_product_once(dtype, scale_dtype) -> None:
    torch.manual_seed(3)
    for experts in (None, 3):
        weight, scale = _random_weight(300, 200, experts=experts, scale_dtype=scale_dtype)
        decoded = dequantize_block_fp8(weight, scale, dtype=dtype)
        expected = _weight64(weight, scale.float()).float().to(dtype)
        assert decoded.shape == weight.shape and torch.equal(decoded, expected)
        stacked = weight if experts else weight[None]
        stacked_scale = scale if experts else scale[None]
        if scale_dtype == E8M0:  # the decode kernel reads FP32/BF16; the public API converts
            stacked_scale = stacked_scale.float()
        portable = torch.empty(stacked.shape, device="cuda", dtype=dtype)
        _block_fp8.dequantize(stacked, stacked_scale, portable)
        assert torch.equal(portable.view(decoded.shape), decoded)


# ----------------------------------------------------------------------------- dense GEMM


def _dense_case(m: int, n: int, k: int, provider: str, scale_dtype=torch.float32) -> None:
    weight, weight_scale = _random_weight(n, k, scale_dtype=scale_dtype)
    if (provider == "cuda-sm12x" and (n % 128 or k % 128)) or (
        provider == "cute-fp8" and (n % 128 or k % 128)
    ):
        with pytest.raises(ValueError, match="multiple"):
            BlockFP8Linear(weight, weight_scale, provider=provider)
        return
    x = _random_activation(m, k)
    op = BlockFP8Linear(weight, weight_scale, provider=provider)
    assert op.provider == provider or (provider is None and op.provider in (HYBRID, "triton-fp8"))
    y = op(x)
    q, scale = quantize_per_token_group_fp8(x)
    assert torch.equal(y, op.quantized(q, scale)), "fused and separate quantization differ"
    x64, w64 = _activation64(q, scale), _weight64(weight, weight_scale)
    oracle = x64 @ w64.T
    comparators = {
        "fp32-floor": (x64.float() @ w64.float().T).to(torch.bfloat16),
        "dequant-bf16": _bf16_baseline(x64, w64),
    }
    if _market_available(n, k):
        comparators["flashinfer"] = _flashinfer(q, scale, weight, weight_scale)
    _assert_no_worse(y, oracle, comparators, x64.abs() @ w64.abs().T)


@pytest.mark.parametrize("provider", DENSE_PROVIDERS)
@pytest.mark.parametrize("shape", DENSE_SHAPES)
@pytest.mark.parametrize("m", [1, 2, 4, 7, 16, 17, 31, 64, 129, 300, 1024])
@torch.inference_mode()
def test_linear_matches_the_fp64_oracle_on_model_shapes(m, shape, provider) -> None:
    torch.manual_seed(4 + m)
    _dense_case(m, *shape, provider)


@pytest.mark.parametrize("shape", [(1536, 4096), (4096, 2048)])
@pytest.mark.parametrize("m", [1, 4, 256])
@torch.inference_mode()
def test_linear_ue8m0_matches_fp32_scales_bitwise_on_dsv4_shapes(m, shape) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("BlockFP8Linear requires CUDA compute capability 8.0 or newer")
    torch.manual_seed(6 + m)
    n, k = shape
    weight, scale = _random_weight(n, k, scale_dtype=E8M0)
    # Include the minimum finite encoding and an exponent outside E4M3's range.
    scale.view(torch.uint8)[0, :2] = torch.tensor([0, 147], dtype=torch.uint8, device="cuda")
    x = _random_activation(m, k)
    op = BlockFP8Linear(weight, scale)
    reference = BlockFP8Linear(weight, scale.to(torch.float32))
    assert op.provider_for(m) == reference.provider_for(m)
    assert torch.equal(op(x).view(torch.uint8), reference(x).view(torch.uint8))


@pytest.mark.parametrize("provider", DENSE_PROVIDERS)
@pytest.mark.parametrize("shape", (*TAIL_SHAPES, (200, 512)))
@pytest.mark.parametrize("m", [1, 5, 33, 257])
@torch.inference_mode()
def test_linear_handles_partial_blocks_and_bf16_scales(m, shape, provider) -> None:
    torch.manual_seed(5 + m)
    _dense_case(m, *shape, provider, scale_dtype=torch.bfloat16)


def _out_op(n, k, provider):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        return BlockFP8Linear(*_random_weight(n, k), provider=provider)
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("provider", (*DENSE_PROVIDERS, "flashinfer-cutlass"))
@pytest.mark.parametrize("m", [0, 1, 4, 64, 512])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset,padding", [(0, 0), (1, 7), (0, 1)])
@torch.inference_mode()
def test_linear_strided_out_columns_and_graph(provider, m, quantized, dtype, offset, padding):
    torch.manual_seed(100 + m)
    n1, n2, k = (1024, 2048, 2048) if provider is None else (256, 128, 2048)
    ops = [_out_op(n, k, provider) for n in (n1, n2)]
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_per_token_group_fp8(x)
    parent = torch.full((m, offset + n1 + n2 + padding), -19, device="cuda", dtype=dtype)
    outputs = [parent[:, offset : offset + n1], parent[:, offset + n1 : offset + n1 + n2]]

    def call(op, out=None):
        if quantized:
            return op.quantized(q, scale, out=out, **({"out_dtype": dtype} if out is None else {}))
        return op(x.view(m, 1, k), out=out, **({"out_dtype": dtype} if out is None else {}))

    expected = [call(op).reshape(m, n) for op, n in zip(ops, (n1, n2), strict=True)]
    assert call(ops[0], outputs[0]) is outputs[0]
    assert torch.all(parent[:, :offset] == -19)
    assert torch.all(parent[:, offset + n1 :] == -19)
    assert call(ops[1], outputs[1]) is outputs[1]
    for out, reference in zip(outputs, expected, strict=True):
        assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
    assert torch.all(parent[:, offset + n1 + n2 :] == -19)
    if not m:
        return
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for op in ops:
            op.warm_up(
                batch_sizes=m, out_dtype=dtype, quantized=quantized, out_row_stride=parent.stride(0)
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for op, out in zip(ops, outputs, strict=True):
            assert call(op, out) is out
    parent.fill_(-19)
    graph.replay()
    for out, reference in zip(outputs, expected, strict=True):
        assert torch.equal(out.view(torch.uint8), reference.view(torch.uint8))
    assert torch.all(parent[:, :offset] == -19)
    assert torch.all(parent[:, offset + n1 + n2 :] == -19)


@pytest.mark.parametrize("quantized", [False, True])
def test_linear_out_errors_before_launch(monkeypatch, quantized):
    op = _out_op(256, 384, None)
    x = torch.zeros(2, 384, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_per_token_group_fp8(x)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid out launched a provider or quantizer")

    from ctikernels.linear import block_fp8

    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", unexpected)
    for provider in (op._triton, op._cute, op._sm12x, op._flashinfer):
        if provider is not None:
            monkeypatch.setattr(provider, "run", unexpected)
            if hasattr(provider, "run_x"):
                monkeypatch.setattr(provider, "run_x", unexpected)
    valid = torch.empty(2, 256, device="cuda", dtype=torch.bfloat16)
    bad = [
        (valid.float(), {}, TypeError),
        (valid, {"out_dtype": torch.float16}, ValueError),
        (valid[:1], {}, ValueError),
        (valid[:, ::2], {}, ValueError),
        (torch.empty(2, 512, device="cuda", dtype=valid.dtype)[:, ::2], {}, ValueError),
        (valid.as_strided((2, 256), (255, 1)), {}, ValueError),
        (valid.cpu(), {}, ValueError),
    ]
    for out, kwargs, error in bad:
        with pytest.raises(error):
            if quantized:
                op.quantized(q, scale, out=out, **kwargs)
            else:
                op(x, out=out, **kwargs)


@pytest.mark.parametrize("provider", (*DENSE_PROVIDERS, "flashinfer-cutlass"))
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("stride", [None, 2056, 2057])
@torch.inference_mode()
def test_linear_warm_up_then_capture(provider, quantized, stride):
    op = _out_op(2048, 512, provider)
    sizes = (1, 4, 64, 512)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(
            batch_sizes=sizes,
            dtype=torch.float16,
            out_dtype=torch.float16,
            quantized=quantized,
            out_row_stride=stride,
        )
    torch.cuda.current_stream().wait_stream(stream)
    for m in sizes:
        x = torch.randn(m, 512, device="cuda", dtype=torch.float16)
        q, scale = quantize_per_token_group_fp8(x)
        out = (
            None
            if stride is None
            else torch.empty_strided((m, 2048), (stride, 1), device="cuda", dtype=torch.float16)
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = (
                op.quantized(q, scale, out_dtype=torch.float16, out=out)
                if quantized
                else op(x, out_dtype=torch.float16, out=out)
            )
        graph.replay()
        eager = (
            op.quantized(q, scale, out_dtype=torch.float16)
            if quantized
            else op(x, out_dtype=torch.float16)
        )
        assert torch.equal(captured.view(torch.uint8), eager.view(torch.uint8))
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
        with pytest.raises(RuntimeError, match="warm_up must run before capture"):
            op.warm_up(batch_sizes=sizes)
        x.zero_()


@pytest.mark.parametrize("m", [1, 64, 2048])
@torch.inference_mode()
def test_large_batches_and_fp16_activations(m) -> None:
    torch.manual_seed(6)
    weight, weight_scale = _random_weight(1024, 2048)
    x = _random_activation(m, 2048, dtype=torch.float16)
    op = BlockFP8Linear(weight, weight_scale)
    y = op(x.view(m, 1, 2048))
    assert y.shape == (m, 1, 1024) and y.dtype == torch.float16
    q, scale = quantize_per_token_group_fp8(x)
    oracle = _activation64(q, scale) @ _weight64(weight, weight_scale).T
    floor = (oracle.float()).to(torch.float16)
    assert _rms(_errors(y.view(m, 1024), oracle)) <= 1.05 * _rms(_errors(floor, oracle)) + 1e-30
    assert torch.equal(op(x, out_dtype=torch.bfloat16), op.quantized(q, scale))


@pytest.mark.parametrize("provider", (*DENSE_PROVIDERS, "flashinfer-cutlass"))
@pytest.mark.parametrize("dim", [-2, -1])
@pytest.mark.parametrize("m", [3, 300, 2048])
@torch.inference_mode()
def test_linear_runs_tp_shard_views(m, dim, provider) -> None:
    """Column- (N) and row-parallel (K) shards from ``slice_block_fp8``: a K shard is a view
    whose row stride exceeds its K. The Triton providers read it in place; FlashInfer, whose
    GEMM derives strides from the shape, refuses it, and the default keeps it on Triton."""
    if provider == "flashinfer-cutlass" and not _market_available(1024, 2048):
        pytest.skip("FlashInfer's CUTLASS GEMM runs on compute capability 10.x and 12.x")
    torch.manual_seed(12 + m)
    weight, weight_scale = _random_weight(1024, 2048, scale_dtype=torch.bfloat16)
    stop = 1024 if dim == -2 else 1536
    shard, shard_scale = slice_block_fp8(weight, weight_scale, dim, 512, stop)
    assert shard.is_contiguous() == (dim == -2)
    if provider in ("flashinfer-cutlass", "cute-fp8") and dim == -1:
        with pytest.raises(ValueError, match="packed"):
            BlockFP8Linear(shard, shard_scale, provider=provider)
        return
    op = BlockFP8Linear(shard, shard_scale, provider=provider)
    if provider is None and dim == -1:
        assert op.provider_for(1 << 20) != "flashinfer-cutlass"
    x = _random_activation(m, shard.shape[1])
    y = op(x)
    packed, packed_scale = shard.contiguous(), shard_scale.contiguous()
    if provider is not None:
        assert torch.equal(y, BlockFP8Linear(packed, packed_scale, provider=provider)(x))
    q, scale = quantize_per_token_group_fp8(x)
    x64, w64 = _activation64(q, scale), _weight64(packed, packed_scale)
    comparators = {
        "fp32-floor": (x64.float() @ w64.float().T).to(torch.bfloat16),
        "dequant-bf16": _bf16_baseline(x64, w64),
    }
    _assert_no_worse(y, x64 @ w64.T, comparators, x64.abs() @ w64.abs().T)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("n", "k"), [(1024, 2048), (1152, 512), (2048, 4096)])
@pytest.mark.parametrize("m", [1, 63, 64, 65, 300, 4100])
@torch.inference_mode()
def test_sm12x_quantizes_inside_the_gemm_bit_for_bit(m, n, k, dtype) -> None:
    """The local CUDA GEMM quantizes BF16/FP16 activations with the standalone recipe: the
    in-GEMM path, the caller-quantized path and unaligned activation rows agree bitwise, in
    BF16 and FP16 output, for both tile shapes (N a multiple of 256, or only of 128)."""
    if not SM12X:
        with pytest.raises(RuntimeError, match="compute capability 12"):
            BlockFP8Linear(*_random_weight(256, 256), provider="cuda-sm12x")
        return
    torch.manual_seed(m + n)
    weight, weight_scale = _random_weight(n, k)
    op = BlockFP8Linear(weight, weight_scale, provider="cuda-sm12x")
    x = _random_activation(m, k, dtype=dtype)
    if m > 4:
        x[2, 7] = math.nan
        x[3, k - 1] = -math.inf
    q, scale = quantize_per_token_group_fp8(x)
    for out_dtype in (torch.bfloat16, torch.float16):
        y = op(x, out_dtype=out_dtype)
        expected = op.quantized(q, scale, out_dtype=out_dtype)
        assert torch.equal(y.isnan(), expected.isnan())
        assert torch.equal(y.nan_to_num(), expected.nan_to_num())
    padded = torch.zeros(m, k + 4, device="cuda", dtype=dtype)
    padded[:, :k] = x
    assert torch.equal(op(padded[:, :k]).nan_to_num(), op(x).nan_to_num())
    finite = x.isfinite().all(1)
    x64, w64 = _activation64(q[finite], scale[finite]), _weight64(weight, weight_scale)
    comparators = {
        "fp32-floor": (x64.float() @ w64.float().T).to(torch.bfloat16),
        "flashinfer": _flashinfer(q[finite], scale[finite], weight, weight_scale),
    }
    # BF16 output, as the comparators' (these activations can exceed FP16's range in FP16).
    y = op(x, out_dtype=torch.bfloat16)
    _assert_no_worse(y[finite], x64 @ w64.T, comparators, x64.abs() @ w64.abs().T)
    assert not y[~finite].isfinite().any().item()


@torch.inference_mode()
def test_flashinfer_provider_on_request() -> None:
    if torch.cuda.get_device_capability()[0] not in (10, 12):
        with pytest.raises(RuntimeError, match="compute capability"):
            BlockFP8Linear(*_random_weight(256, 256), provider="flashinfer-cutlass")
        return
    torch.manual_seed(7)
    weight, weight_scale = _random_weight(1024, 2048, scale_dtype=torch.bfloat16)
    op = BlockFP8Linear(weight, weight_scale, provider="flashinfer-cutlass")
    assert op.provider == "flashinfer-cutlass"
    for m in (1, 33):
        x = _random_activation(m, 2048)
        q, scale = quantize_per_token_group_fp8(x)
        assert torch.equal(op(x), _flashinfer(q, scale, weight, weight_scale))
        assert torch.equal(op.quantized(q, scale), op(x))
        # A one-row view at an unaligned base reports itself contiguous.
        padded = torch.empty((m, 2049), dtype=q.dtype, device=q.device)[:, 1:]
        padded.copy_(q)
        assert torch.equal(op.quantized(padded, scale), op(x))
    with pytest.raises(ValueError, match="multiples of 128"):
        BlockFP8Linear(*_random_weight(200, 256), provider="flashinfer-cutlass")


def _empty_capture_allocations(stream=None) -> int:
    """Allocations an empty CUDA graph capture counts (torch 2.13 counts +2 per capture, 2.14 none)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
        pass
    return torch.cuda.memory_stats()["allocation.all.allocated"] - before


@pytest.mark.parametrize("quantized", [False, True])
def test_flashinfer_captures_unwarmed_batches_without_allocating(quantized) -> None:
    """Warm-up runs one batch per class: capturing batches it never ran allocates nothing
    and replays the eager bits."""
    op = _out_op(2048, 4096, "flashinfer-cutlass")
    torch.manual_seed(8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmed = {route["m"] for route in op.warm_up(8192, quantized=quantized)}
    torch.cuda.current_stream().wait_stream(stream)
    workspace = torch.empty(op.workspace_bytes(8192) // 4, device="cuda")
    baseline = _empty_capture_allocations(stream)
    for m in (515, 1337, 4097, 8190):  # 8191 is its class's largest, so warmed
        assert m not in warmed
        x = _random_activation(m, 4096)
        q, scale = quantize_per_token_group_fp8(x)
        out = torch.empty(m, 2048, dtype=torch.bfloat16, device="cuda")
        eager = op.quantized(q, scale) if quantized else op(x)
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if quantized:
                op.quantized(q, scale, out=out, workspace=workspace)
            else:
                op(x, out=out, workspace=workspace)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before + baseline
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), eager.view(torch.uint8))


def test_default_provider_follows_the_device() -> None:
    capability = torch.cuda.get_device_capability()
    expected = "triton-fp8"
    assert default_provider("cuda") == expected
    hybrid = capability in ((12, 0), (12, 1))
    op = BlockFP8Linear(*_random_weight(1024, 2048))
    assert op.provider == (HYBRID if hybrid else expected)
    assert op.provider_for(1) == expected
    # Large batches run a measured GEMM provider: FlashInfer, the CuTe GEMM (the RTX 5090's
    # table) or, on GB10, also the local CUDA GEMM.
    large = ("flashinfer-cutlass", "cute-fp8", "cuda-sm12x") if hybrid else (expected,)
    assert op.provider_for(1 << 20) in large
    # Shapes FlashInfer does not take stay on the Triton GEMM for every batch.
    op = BlockFP8Linear(*_random_weight(200, 2048))
    assert op.provider == expected and op.provider_for(1 << 20) == expected
    with pytest.raises(ValueError, match="provider must be one of"):
        BlockFP8Linear(*_random_weight(128, 128), provider="cublas")


def test_frozen_weights_raise_before_launch() -> None:
    weight, scale = _random_weight(256, 256)
    with pytest.raises(ValueError, match="weight_scale_inv is frozen"):
        BlockFP8Linear(weight, scale.clone().requires_grad_())
    op = BlockFP8Linear(weight, scale)
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    scale.requires_grad_()
    with pytest.raises(ValueError, match="frozen"), torch.no_grad():
        op(x)
    scale.requires_grad_(False)
    with pytest.raises(RuntimeError, match="inference-only"):
        op(x.clone().requires_grad_())
    with torch.no_grad():
        op(x.clone().requires_grad_())
    q, x_scale = quantize_per_token_group_fp8(x)
    with pytest.raises(RuntimeError, match="inference-only"):
        op.quantized(q, x_scale.clone().requires_grad_())
    with pytest.raises(ValueError, match="frozen"):
        dequantize_block_fp8(weight, scale.clone().requires_grad_())


def test_layouts_are_checked_not_guessed() -> None:
    weight, scale = _random_weight(256, 384)
    with pytest.raises(ValueError, match="must be \\(2, 3\\)"):
        BlockFP8Linear(weight, scale.T.contiguous())
    with pytest.raises(ValueError, match="unit stride along K"):
        BlockFP8Linear(weight.T.contiguous().T, scale)
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        BlockFP8Linear(weight.to(torch.bfloat16), scale)
    with pytest.raises(TypeError, match="weight_scale_inv must be"):
        BlockFP8Linear(weight, scale.half())
    op = BlockFP8Linear(weight, scale)
    with pytest.raises(ValueError, match="x must be"):
        op(torch.randn(2, 256, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="BF16 or FP16"):
        op(torch.randn(2, 384, device="cuda"))
    with pytest.raises(ValueError, match="unit stride"):
        op(torch.randn(384, 2, device="cuda", dtype=torch.bfloat16).T)
    q, s = quantize_per_token_group_fp8(torch.randn(2, 384, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="scale \\[M, 3\\]"):
        op.quantized(q, s[:, :2])


@pytest.mark.parametrize("provider", DENSE_PROVIDERS)
@pytest.mark.parametrize("m", [1, 4, 64, 1024])
def test_linear_replays_in_a_cuda_graph(m, provider) -> None:
    torch.manual_seed(8)
    op = BlockFP8Linear(*_random_weight(2048, 4096), provider=provider)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.warm_up(batch_sizes=[m])
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = op(x)
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y, op(x))


# ----------------------------------------------------------------------------- grouped MoE


def _routes(m: int, experts: int, top_k: int, routing: str) -> torch.Tensor:
    scores = torch.rand(m, experts, device="cuda")
    if routing == "skewed":
        scores[:, 3] += 2
        scores[:, 5] += 1
    ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    if routing == "remote":  # EP: routes to other ranks' experts are -1
        ids[torch.rand(m, top_k, device="cuda") < 0.4] = -1
        if m > 1:
            ids[1] = -1
    return ids


def _grouped_oracle(a64, w64, ids, weights, *, top_k_rows: int, weighted: bool):
    m, top_k = ids.shape
    out = torch.zeros(m, top_k, w64.shape[1], device="cuda", dtype=torch.float64)
    for expert in ids.unique().tolist():
        if expert < 0:
            continue
        token, choice = (ids == expert).nonzero(as_tuple=True)
        rows = a64[token * top_k + choice if top_k_rows == 1 else token]
        out[token, choice] = rows @ w64[expert].T
    return out * weights.double()[..., None] if weighted else out


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("routing", ["uniform", "skewed", "remote"])
@pytest.mark.parametrize(("m", "config"), [(1, "small"), (5, "engine"), (64, "large")])
@pytest.mark.parametrize(("n", "k"), [(384, 640), (200, 328)])
@torch.inference_mode()
def test_grouped_gemm_matches_the_fp64_oracle(m, config, routing, provider, n, k) -> None:
    torch.manual_seed(9 + m)
    experts, top_k = 16, 4
    weight, weight_scale = _random_weight(n, k, experts=experts)
    ids = _routes(m, experts, top_k, routing)
    weights = torch.rand(m, top_k, device="cuda").softmax(-1)
    if config == "engine":  # the engine's small-batch BF16 configuration: K steps of 64
        config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    else:
        config = block_fp8_moe_config(
            1 if config == "small" else 4096,
            experts,
            n,
            k,
            top_k,
            capability=torch.cuda.get_device_capability(),
        )
    sorted_ids, block_experts, padded = moe_align_block_size(ids, config["BLOCK_SIZE_M"], experts)
    for expanded, weighted in ((False, False), (True, True)):
        x = _random_activation(m * top_k if expanded else m, k)
        q, scale = quantize_per_token_group_fp8(x)
        out = torch.full((m, top_k, n), torch.nan, device="cuda", dtype=torch.bfloat16)
        fused_moe_kernel_triton(
            q, weight, out, weights, ids, sorted_ids, block_experts, padded, weighted,
            1 if expanded else top_k, config, torch.bfloat16,
            a_scale=scale, b_scale=weight_scale, block_fp8_provider=provider,
        )  # fmt: skip
        a64, w64 = _activation64(q, scale), _weight64(weight, weight_scale)
        oracle = _grouped_oracle(
            a64, w64, ids, weights, top_k_rows=1 if expanded else top_k, weighted=weighted
        )
        remote = ids < 0
        assert torch.equal(out[remote], torch.zeros_like(out[remote]))
        local = ~remote
        floor = oracle.float().to(torch.bfloat16)
        baseline = torch.full_like(out, torch.nan)
        fused_moe_kernel_triton(
            a64.to(torch.bfloat16), w64.to(torch.bfloat16), baseline, weights, ids, sorted_ids,
            block_experts, padded, weighted, 1 if expanded else top_k,
            {**config, "BLOCK_SIZE_K": 32}, torch.bfloat16,
        )  # fmt: skip
        assert torch.equal(baseline[remote], torch.zeros_like(baseline[remote]))
        magnitude = _grouped_oracle(
            a64.abs(), w64.abs(), ids, weights, top_k_rows=1 if expanded else top_k,
            weighted=weighted,
        )  # fmt: skip
        comparators = {"fp32-floor": floor[local], "dequant-bf16": baseline[local]}
        _assert_no_worse(out[local], oracle[local], comparators, magnitude[local])


def _expert_forward(x, w13, s13, w2, s2, weights, ids, config, provider, stages=None):
    """quantize -> GEMM1 -> SiLU x up -> quantize -> GEMM2 -> weighted combine; ``stages``
    (a dict) receives each stage's operands and output."""
    m, top_k = ids.shape
    experts, two_i, hidden = w13.shape
    sorted_ids, block_experts, padded = moe_align_block_size(ids, config["BLOCK_SIZE_M"], experts)
    gate_up = torch.empty(m, top_k, two_i, device="cuda", dtype=torch.bfloat16)
    act = torch.empty(m * top_k, two_i // 2, device="cuda", dtype=torch.bfloat16)
    down = torch.empty(m, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_per_token_group_fp8(x)
    fused_moe_kernel_triton(
        q, w13, gate_up, weights, ids, sorted_ids, block_experts, padded, False, top_k, config,
        torch.bfloat16, a_scale=scale, b_scale=s13, block_fp8_provider=provider,
    )  # fmt: skip
    if m:  # FlashInfer's SiLU-and-mul rejects an empty batch
        silu_and_mul(gate_up.view(-1, two_i), act)
    q2, scale2 = quantize_per_token_group_fp8(act)
    fused_moe_kernel_triton(
        q2, w2, down, weights, ids, sorted_ids, block_experts, padded, False, 1, config,
        torch.bfloat16, a_scale=scale2, b_scale=s2, block_fp8_provider=provider,
    )  # fmt: skip
    out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
    moe_sum_reduce_triton(down, out, topk_ids=ids, topk_weights=weights)
    if stages is not None:
        stages.update(q=q, scale=scale, gate_up=gate_up, act=act, q2=q2, scale2=scale2, down=down)
    return out


def _assert_grouped_stage(out, q, scale, weight, weight_scale, ids, weights, *, expanded, weighted):
    """One grouped GEMM of the forward against the FP64 oracle on its own quantized operands,
    with the FP32 floor (the oracle rounded through FP32) as the comparator; -1 routes are 0."""
    a64, w64 = _activation64(q, scale), _weight64(weight, weight_scale)
    top_k_rows = 1 if expanded else ids.shape[1]
    oracle = _grouped_oracle(a64, w64, ids, weights, top_k_rows=top_k_rows, weighted=weighted)
    magnitude = _grouped_oracle(
        a64.abs(), w64.abs(), ids, weights, top_k_rows=top_k_rows, weighted=weighted
    )
    remote = ids < 0
    assert torch.equal(out[remote], torch.zeros_like(out[remote]))
    local = ~remote
    floor = oracle.float().to(torch.bfloat16)
    _assert_no_worse(out[local], oracle[local], {"fp32-floor": floor[local]}, magnitude[local])


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("m", [1, 4, 37])
@torch.inference_mode()
def test_expert_parallel_forward_end_to_end(m, provider) -> None:
    """Rank 1 of 4 holds global experts 8..15; routes elsewhere map to -1 and add nothing.

    Every stage is held to the FP32 floor on its own operands, as the single GEMMs are: both
    quantizations equal the recipe bit for bit, each grouped GEMM meets the FP64 oracle of
    the quantized operands it was given, and the combine is vLLM's finalize bit for bit.
    (A composed tolerance would mostly measure BF16 rounding flips of the intermediates that
    the next quantization amplifies, which no FP32 floor bounds.)"""
    torch.manual_seed(10 + m)
    global_experts, local, top_k, hidden, inter = 32, 8, 8, 512, 256
    w13, s13 = _random_weight(2 * inter, hidden, experts=local, scale_dtype=torch.bfloat16)
    w2, s2 = _random_weight(hidden, inter, experts=local)
    expert_map = torch.full((global_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[8:16] = torch.arange(local, device="cuda", dtype=torch.int32)
    global_ids = _routes(m, global_experts, top_k, "skewed")
    if m > 2:
        global_ids[2] = torch.arange(16, 16 + top_k, device="cuda")  # no local route
    ids = expert_map[global_ids.long()]
    weights = torch.rand(m, top_k, device="cuda").softmax(-1)
    x = (torch.randn(m, hidden, device="cuda") * 4).to(torch.bfloat16)
    config = block_fp8_moe_config(
        m, local, 2 * inter, hidden, top_k, capability=torch.cuda.get_device_capability()
    )
    stages = {}
    out = _expert_forward(x, w13, s13, w2, s2, weights, ids, config, provider, stages)
    for source, q, scale in ((x, "q", "scale"), (stages["act"], "q2", "scale2")):
        expected_q, expected_scale = _reference_quantize(source)
        assert torch.equal(stages[q].view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(stages[scale], expected_scale)
    _assert_grouped_stage(
        stages["gate_up"], stages["q"], stages["scale"], w13, s13, ids, weights,
        expanded=False, weighted=False,
    )  # fmt: skip
    _assert_grouped_stage(
        stages["down"], stages["q2"], stages["scale2"], w2, s2, ids, weights,
        expanded=True, weighted=False,
    )  # fmt: skip
    expected = combine_reference(stages["down"], weights, ids >= 0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    if m > 2:
        assert torch.equal(out[2], torch.zeros_like(out[2]))


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("m", [1, 4, 37])
@torch.inference_mode()
def test_expert_parallel_forward_with_expert_map_alignment(m, provider) -> None:
    """K2MAP: ``moe_align_block_size(..., expert_map=...)`` gives remote routes no block, so
    the block-FP8 GEMMs never write their rows (NaN sentinels stay) and the masked reduce
    never reads them. Local routes and the sum equal the pre-mapped -1 path bit for bit."""
    torch.manual_seed(20 + m)
    global_experts, local, top_k, hidden, inter = 32, 8, 8, 512, 256
    w13, s13 = _random_weight(2 * inter, hidden, experts=local, scale_dtype=torch.bfloat16)
    w2, s2 = _random_weight(hidden, inter, experts=local)
    expert_map = torch.full((global_experts,), -1, device="cuda", dtype=torch.int32)
    expert_map[8:16] = torch.arange(local, device="cuda", dtype=torch.int32)
    global_ids = _routes(m, global_experts, top_k, "skewed")
    if m > 2:
        global_ids[2] = torch.arange(16, 16 + top_k, device="cuda")  # no local route
    global_ids[0, -1] = -1  # graph padding
    weights = torch.rand(m, top_k, device="cuda").softmax(-1)
    x = (torch.randn(m, hidden, device="cuda") * 4).to(torch.bfloat16)
    config = block_fp8_moe_config(
        m, local, 2 * inter, hidden, top_k, capability=torch.cuda.get_device_capability()
    )
    stages = {}
    expected = _expert_forward(
        x, w13, s13, w2, s2, weights, expert_map[global_ids.long()], config, provider, stages
    )
    aligned = moe_align_block_size(global_ids, config["BLOCK_SIZE_M"], local, expert_map)
    nan = float("nan")
    gate_up = torch.full((m, top_k, 2 * inter), nan, device="cuda", dtype=torch.bfloat16)
    down = torch.full((m, top_k, hidden), nan, device="cuda", dtype=torch.bfloat16)
    q, scale = quantize_per_token_group_fp8(x)
    fused_moe_kernel_triton(
        q, w13, gate_up, weights, global_ids, *aligned, False, top_k, config,
        torch.bfloat16, a_scale=scale, b_scale=s13, block_fp8_provider=provider,
    )  # fmt: skip
    act = silu_and_mul(gate_up.view(-1, 2 * inter))
    q2, scale2 = quantize_per_token_group_fp8(act)
    fused_moe_kernel_triton(
        q2, w2, down, weights, global_ids, *aligned, False, 1, config,
        torch.bfloat16, a_scale=scale2, b_scale=s2, block_fp8_provider=provider,
    )  # fmt: skip
    out = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
    moe_sum_reduce_triton(
        down, out, topk_ids=global_ids, expert_map=expert_map, topk_weights=weights
    )
    remote = expert_map[global_ids.clamp(min=0).long()].lt(0) | global_ids.lt(0)
    assert gate_up[remote].isnan().all() and down[remote].isnan().all()
    assert torch.equal(gate_up[~remote], stages["gate_up"][~remote])
    assert torch.equal(down[~remote], stages["down"][~remote])
    assert torch.equal(out, expected)


@torch.inference_mode()
def test_expert_forward_without_tokens_or_local_routes() -> None:
    w13, s13 = _random_weight(256, 256, experts=4)
    w2, s2 = _random_weight(256, 128, experts=4)
    config = block_fp8_moe_config(1, 4, 256, 256, 2, capability=torch.cuda.get_device_capability())
    empty = torch.empty(0, 2, device="cuda", dtype=torch.int32)
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    out = _expert_forward(x, w13, s13, w2, s2, empty.float(), empty, config, None)
    assert out.shape == (0, 256)
    ids = torch.full((3, 2), -1, device="cuda", dtype=torch.int32)
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    out = _expert_forward(x, w13, s13, w2, s2, torch.ones(3, 2, device="cuda"), ids, config, None)
    assert torch.equal(out, torch.zeros_like(out))


def test_expert_forward_replays_in_a_cuda_graph() -> None:
    torch.manual_seed(11)
    w13, s13 = _random_weight(512, 512, experts=8)
    w2, s2 = _random_weight(512, 256, experts=8)
    ids = _routes(4, 8, 2, "uniform")
    weights = torch.rand(4, 2, device="cuda")
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    config = block_fp8_moe_config(4, 8, 512, 512, 2, capability=torch.cuda.get_device_capability())
    with torch.inference_mode():
        _expert_forward(x, w13, s13, w2, s2, weights, ids, config, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _expert_forward(x, w13, s13, w2, s2, weights, ids, config, None)
        x.copy_(torch.randn_like(x))
        ids.copy_(_routes(4, 8, 2, "remote"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _expert_forward(x, w13, s13, w2, s2, weights, ids, config, None))


def test_grouped_gemm_checks_its_block_fp8_operands() -> None:
    w, s = _random_weight(256, 256, experts=2)
    q, scale = quantize_per_token_group_fp8(
        torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    )
    ids = torch.zeros(2, 1, device="cuda", dtype=torch.int32)
    out = torch.empty(2, 1, 256, device="cuda", dtype=torch.bfloat16)
    config = block_fp8_moe_config(2, 2, 256, 256, 1, capability=torch.cuda.get_device_capability())
    aligned = moe_align_block_size(ids, config["BLOCK_SIZE_M"], 2)

    def launch(**kwargs):
        arguments = {"a_scale": scale, "b_scale": s}
        arguments.update(kwargs)
        fused_moe_kernel_triton(
            arguments.pop("A", q), w, out, torch.ones(2, 1, device="cuda"), ids, *aligned,
            False, 1, arguments.pop("config", config), torch.bfloat16, **arguments,
        )  # fmt: skip

    with pytest.raises(ValueError, match="both a_scale and b_scale"):
        launch(b_scale=None)
    with pytest.raises(ValueError, match="b_scale must be"):
        launch(b_scale=s[:, :1])
    with pytest.raises(ValueError, match="BLOCK_SIZE_K"):
        launch(config={**config, "BLOCK_SIZE_K": 256})
    with pytest.raises(TypeError, match="E4M3"):
        launch(A=q.to(torch.bfloat16))
    with pytest.raises(ValueError, match="must not require grad"):
        launch(b_scale=s.clone().requires_grad_())
    with pytest.raises(ValueError, match="block_fp8_provider"):
        launch(block_fp8_provider="cutlass")


CUTE_CONFIGS = (
    "swap128x64s3-mx",
    "normal128x64s3-plain",
    "normal64x128s3-persistent-mx",
    "swap128x32s2c2-mx",
    "swap64x64s2c2-persistent-plain",
    "normal128x128s2-tailsplit-mx",
)

CUTE_OVERLAP_CONFIGS = (
    ("swap128x64s2-persistent-ovl-mx", "swap128x64s2-mx"),
    ("normal64x128s2w4-persistent-ovl-mx", "normal64x128s2w4-mx"),
    ("swap128x32s4-persistent-ovl-pdl-mx", "swap128x32s4-mx"),
)


@pytest.mark.parametrize("config,alias", CUTE_OVERLAP_CONFIGS)
@pytest.mark.parametrize("m", [1, 100, 256, 777])
@pytest.mark.parametrize("shape", [(4096, 2048), (1536, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cute_private_epilogue_bitwise(config, alias, m, shape, dtype):
    _require_cute()
    torch.manual_seed(510 + m)
    weight, ws = _random_weight(*shape)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")._cute
    q, scale = quantize_per_token_group_fp8(_random_activation(m, shape[1]))
    expected = torch.empty(m, shape[0], device="cuda", dtype=dtype)
    out = torch.empty_like(expected)
    op.run(q, scale, expected, alias)
    op.run(q, scale, out, config)
    # Compare representations too: signed zeros must agree.
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_cute_private_epilogue_graph_replay():
    _require_cute()
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    config, alias = CUTE_OVERLAP_CONFIGS[-1]
    m, n, k = 777, 4096, 2048
    weight, ws = _random_weight(n, k)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")._cute
    x = _random_activation(m, k)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run_x(x, out, config)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # Capture the PDL quantizer/GEMM pair, including the weight prefetch.
        q, scale = quantize(x)
        op.run(q, scale, out, config)
    for seed in (71, 88, 99):
        torch.manual_seed(seed)
        x.copy_(_random_activation(m, k))
        graph.replay()
        # Read the exact quantized operands produced by the captured quantizer.
        op.run(q, scale, expected, alias)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    x.zero_()
    graph.replay()
    op.run(q, scale, expected, alias)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(out) == 0


def _assert_cute_precision(out, q, scale, weight, weight_scale):
    """Use the same FP64 gate and comparators as FlashInfer and the other dense providers."""
    x64, w64 = _activation64(q, scale), _weight64(weight, weight_scale)
    comparators = {
        "fp32-floor": (x64.float() @ w64.float().T).to(torch.bfloat16),
        "dequant-bf16": _bf16_baseline(x64, w64),
    }
    if _market_available(*weight.shape):
        comparators["flashinfer"] = _flashinfer(q, scale, weight, weight_scale)
    _assert_no_worse(out, x64 @ w64.T, comparators, x64.abs() @ w64.abs().T)


def _select_cute_configs(monkeypatch, shape, runs):
    from ctikernels.linear import _block_fp8_dispatch as dispatch

    hardware = (
        *torch.cuda.get_device_capability(),
        torch.cuda.get_device_properties(0).multi_processor_count,
    )
    monkeypatch.setitem(dispatch.CUTE_CONFIG, hardware, {shape: runs})


@pytest.mark.parametrize("config", CUTE_CONFIGS)
@pytest.mark.parametrize("m", [1, 7, 128, 200, 256])
@pytest.mark.parametrize("shape", [(1536, 4096), (4096, 2048)])
@torch.inference_mode()
def test_cute_dense_configs_fp64_gate(monkeypatch, config, m, shape):
    _require_cute()
    _select_cute_configs(monkeypatch, shape, ((256, config),))
    torch.manual_seed(410 + m)
    weight, ws = _random_weight(*shape, scale_dtype=torch.bfloat16)
    x = _random_activation(m, shape[1])
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")
    q, scale = quantize_per_token_group_fp8(x)
    # Each entry is judged numerically, including public quantization and caller-quantized input.
    _assert_cute_precision(op(x), q, scale, weight, ws)
    _assert_cute_precision(op.quantized(q, scale), q, scale, weight, ws)


@pytest.mark.parametrize("m", [1, 4, 33])
@torch.inference_mode()
def test_cute_quantized_unaligned_row_views(m):
    """quantized() admits unit-column-stride views whose rows are not 16-byte aligned."""
    _require_cute()
    torch.manual_seed(520 + m)
    shape = (1536, 2048)
    weight, ws = _random_weight(*shape)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")
    q, scale = quantize_per_token_group_fp8(_random_activation(m, shape[1]))
    padded = torch.zeros((m, shape[1] + 1), device=q.device, dtype=q.dtype)
    padded[:, : shape[1]] = q
    shifted = torch.zeros((m * shape[1] + 1,), device=q.device, dtype=q.dtype)
    shifted[1:] = q.reshape(-1)
    expected = op.quantized(q, scale)
    for view in (padded[:, : shape[1]], shifted[1:].view(m, shape[1])):
        assert view.stride(1) == 1
        out = op.quantized(view, scale)
        assert torch.equal(out, expected)
        _assert_cute_precision(out, q, scale, weight, ws)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cute_overlapping_activation_rows(monkeypatch, dtype):
    """Broadcast and overlapping read-only activation rows take the CuTe quantizer path."""
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    _require_cute()
    torch.manual_seed(611)
    shape, m = (1536, 2048), 4
    base = _random_activation(1, shape[1], dtype)
    storage = _random_activation(1, shape[1] + (m - 1) * 128, dtype).reshape(-1)
    views = (base.expand(m, shape[1]), storage.as_strided((m, shape[1]), (128, 1)))
    for x in views:
        assert x.stride(0) < shape[1] and x.stride(1) == 1
        q, scale = quantize(x)
        expected_q, expected_scale = quantize_per_token_group_fp8(x.contiguous())
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))
    _select_cute_configs(monkeypatch, shape, ((256, "swap128x32s2c2-mx-pdl"),))
    weight, ws = _random_weight(*shape)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")
    for x in views:
        out = op(x)
        assert torch.equal(out, op(x.contiguous()))
        q, scale = quantize_per_token_group_fp8(x.contiguous())
        _assert_cute_precision(out, q, scale, weight, ws)


@pytest.mark.parametrize("config", (*CUTE_CONFIGS, *(name + "-pdl" for name in CUTE_CONFIGS)))
@torch.inference_mode()
def test_cute_graph_replay_changed_inputs(monkeypatch, config):
    """Port the dense source's changed-input replay test, exercising the public quantizer too."""
    _require_cute()
    shape, m = (1536, 4096), 7
    _select_cute_configs(monkeypatch, shape, ((256, config),))
    weight, ws = _random_weight(*shape)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")
    x = _random_activation(m, shape[1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(batch_sizes=[m])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = op(x)
    for seed in (71, 88, 99):
        torch.manual_seed(seed)
        x.copy_(_random_activation(m, shape[1]))
        graph.replay()
        q, scale = quantize_per_token_group_fp8(x)
        _assert_cute_precision(out, q, scale, weight, ws)
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0


@torch.inference_mode()
def test_cute_dynamic_m_and_strides_reuse_compilation():
    from ctikernels.linear.cute import _block_fp8_provider as provider

    _require_cute()
    weight, ws = _random_weight(256, 384)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")._cute
    config = "swap128x32s4-mx"
    op.warmup(1, config)
    count = len(provider._compiled)
    q, scale = quantize_per_token_group_fp8(_random_activation(1, 384))
    op.run(q, scale, torch.empty((1, 272), device="cuda", dtype=torch.bfloat16)[:, :256], config)
    # A row-strided output compiles one dynamic-stride variant; packed outputs keep theirs.
    assert len(provider._compiled) == count + 1
    count += 1
    for m in (3, 17, 129):
        q, scale = quantize_per_token_group_fp8(_random_activation(m, 384))
        scale = scale.T.contiguous().T
        assert not scale.is_contiguous()
        for stride, offset in ((256, 0), (272, 8), (273, 1)):
            parent = torch.full((m, stride), -19, device="cuda", dtype=torch.bfloat16)
            out = parent[:, offset : offset + 256]
            op.run(q, scale, out, config)
            _assert_cute_precision(out, q, scale, weight, ws)
            assert torch.all(parent[:, :offset] == -19)
            assert torch.all(parent[:, offset + 256 :] == -19)
    assert len(provider._compiled) == count


@torch.inference_mode()
def test_cute_concurrent_caller_workspaces():
    """Tail flags and partials belong to each concurrent stream/graph."""
    _require_cute()
    # Two tiles * seven K blocks / MIN_SK_ITERS=4 gives three SK CTAs,
    # with a helper on each tile (five K blocks would produce unsplit tiles).
    weight, ws = _random_weight(256, 896)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")._cute
    config = "normal128x128s2-tailsplit-mx"
    operands = [quantize_per_token_group_fp8(_random_activation(65, 896)) for _ in range(2)]
    workspaces = [op.make_workspace(config) for _ in range(2)]
    outputs = [torch.empty(65, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    graphs = [torch.cuda.CUDAGraph() for _ in range(2)]
    for i, stream in enumerate(streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.run(*operands[i], outputs[i], config, workspace=workspaces[i])
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graphs[i], stream=stream):
            op.run(*operands[i], outputs[i], config, workspace=workspaces[i])
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            graphs[i].replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for out, (q, scale) in zip(outputs, operands, strict=True):
        _assert_cute_precision(out, q, scale, weight, ws)


@pytest.mark.parametrize("m", [1, 3, 128, 257])
@pytest.mark.parametrize("k", [128, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("padding,offset", [(0, 0), (16, 0), (1, 1)])
@torch.inference_mode()
def test_cute_activation_quantizer_bitwise(m, k, dtype, padding, offset):
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    _require_cute()
    torch.manual_seed(913 + m)
    storage = torch.empty((m, k + padding + offset), device="cuda", dtype=dtype)
    x = storage[:, offset : offset + k]
    magnitudes = torch.logspace(-12, 4, k, device="cuda")
    x.copy_(torch.randn(m, k, device="cuda") * magnitudes)
    x[0, 0] = torch.finfo(dtype).max
    x[0, 1] = -torch.finfo(dtype).max
    for zero in (False, True):
        if zero:
            x[::2].zero_()
        q, scale = quantize(x)
        expected_q, expected_scale = quantize_per_token_group_fp8(x)
        assert q.stride() == expected_q.stride()
        assert scale.stride() == expected_scale.stride()
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_cute_activation_quantizer_nonfinite_and_rounding(dtype):
    from ctikernels.linear.cute._block_fp8_act_quant import quantize

    _require_cute()
    x = torch.randn(8, 128, device="cuda", dtype=dtype)
    x[0, 17] = float("nan")
    x[1, 31] = float("inf")
    x[2, 70] = -float("inf")
    x[3].fill_(float("nan"))
    x[4].fill_(-0.0)
    # Scale=1: E4M3 midpoint ties, subnormals and signed zeros.
    x[5].zero_()
    x[5, :8] = torch.tensor(
        [448, 1.0625, 1.1875, -1.0625, 2**-10, 3 * 2**-10, -0.0, -(2**-10)],
        device="cuda",
        dtype=dtype,
    )
    x[6].fill_(torch.finfo(dtype).tiny)
    x[7].fill_(-torch.finfo(dtype).max)
    q, scale = quantize(x)
    expected_q, expected_scale = quantize_per_token_group_fp8(x)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), expected_scale.view(torch.int32))


@pytest.mark.parametrize("base", CUTE_CONFIGS)
@pytest.mark.parametrize("m,k", [(3, 128), (128, 2048), (256, 4096)])
@torch.inference_mode()
def test_cute_pdl_same_operands_bitwise_and_fp64(base, m, k):
    _require_cute()
    torch.manual_seed(812 + m)
    weight, ws = _random_weight(256, k)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")
    x = _random_activation(m, k)
    q, scale = quantize_per_token_group_fp8(x)
    ordinary = torch.empty(m, 256, device="cuda", dtype=torch.bfloat16)
    pdl = torch.empty_like(ordinary)
    op._cute.run(q, scale, ordinary, base)
    op._cute.run(q, scale, pdl, base + "-pdl")
    assert torch.equal(pdl, ordinary)
    _assert_cute_precision(pdl, q, scale, weight, ws)
    op._cute_config = lambda rows: base + "-pdl"
    assert torch.equal(op(x), ordinary)


@pytest.mark.parametrize(
    "base,m,n,k,epi,pdl,atom,policy_suffix",
    [
        (base, m, n, k, epi, pdl, atom, "")
        for base, m, n, k in (
            ("normal128x128s2", 2048, 2048, 4096),
            ("normal128x128s2", 256, 12288, 2048),
            ("normal128x128s2", 512, 2048, 4096),
            ("normal128x128s2", 64, 1024, 2048),
            ("normal128x128s2", 33, 4096, 512),
            ("normal128x128s2", 17, 256, 4096),  # T < S on all qualification GPUs.
            ("normal128x128s2", 1024, 4096, 8192),  # DP + SK on the 48-SM GB10.
            ("swap128x64s2", 129, 1024, 2048),  # Swapped accumulator coordinates.
            ("swap128x32s2c2", 65, 1024, 512),  # Two resident CTAs per SM.
        )
        for epi in ("alias", "private")
        for pdl in (False, True)
        for atom in ("mx", "plain")
    ]
    + [
        (base, m, n, k, "alias", False, atom, suffix)
        for base in ("normal128x128s2", "swap128x64s2")
        for m, n, k in (
            (512, 2048, 4096),
            (2048, 2048, 4096),
            (256, 12288, 2048),
            (64, 1024, 2048),
        )
        for atom in ("mx", "plain")
        for suffix in ("-t100-dpf", "-t100-w1", "-t100-dpf-w1")
    ],
)
@torch.inference_mode()
def test_cute_tailsplit_register_fixup_and_replay(base, m, n, k, epi, pdl, atom, policy_suffix):
    """Hybrid numeric gate, deterministic arrivals, reset/reuse, and PDL ordering."""
    _require_cute()
    from ctikernels.linear.cute._block_fp8_provider import CONFIGS, schedule_for, tail_workspace

    config = f"{base}-tailsplit{'-ovl' if epi == 'private' else ''}{'-pdl' if pdl else ''}-{atom}"
    config += policy_suffix
    grid_config = f"{base}-{atom}"
    torch.manual_seed(2751 + m)
    weight, ws = _random_weight(n, k)
    op = BlockFP8Linear(weight, ws, provider="cute-fp8")._cute
    policy = CONFIGS[config]
    workspace = op.make_workspace(config)
    partial_words, _ = tail_workspace(policy, op._sms)
    if m == 17:
        tiles = math.ceil(m / policy.token_tile) * math.ceil(n / policy.weight_tile)
        assert tiles < op._sms
        assert schedule_for(tiles, op._sms, k // 128, policy).sk_tiles == tiles
    q, scale = quantize_per_token_group_fp8(_random_activation(m, k))
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    grid = torch.empty_like(out)
    op.run(q, scale, grid, grid_config)
    # Poison only partials: every read must refer to a fresh helper publication.
    workspace[:partial_words].fill_(float("nan"))
    op.run(q, scale, out, config, workspace=workspace)
    eager = out.clone()
    x64, w64 = _activation64(q, scale), _weight64(weight, ws)
    oracle = x64 @ w64.T
    magnitude = x64.abs() @ w64.abs().T
    comparators = {
        "grid": grid,
        "fp32-floor": (x64.float() @ w64.float().T).to(torch.bfloat16),
        "dequant-bf16": _bf16_baseline(x64, w64),
    }
    if _market_available(n, k):
        comparators["flashinfer"] = _flashinfer(q, scale, weight, ws)
    _assert_no_worse(grid, oracle, comparators, magnitude)
    _assert_no_worse(out, oracle, comparators, magnitude)
    for _ in range(2):  # Three eager launches must agree bit for bit.
        workspace[:partial_words].fill_(float("nan"))
        op.run(q, scale, out, config, workspace=workspace)
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        assert torch.count_nonzero(workspace[partial_words:].view(torch.int32)) == 0

    # Alternate to a different partition using the SAME compiled function and
    # scratch. No synchronization is inserted between these three launches.
    other_m = 2 * m + policy.token_tile + 1
    other_q, other_scale = quantize_per_token_group_fp8(_random_activation(other_m, k))
    other_out = torch.empty(other_m, n, device="cuda", dtype=out.dtype)
    op.run(other_q, other_scale, other_out, config, workspace=workspace)
    other_eager = other_out.clone()

    def launch_pair():
        op.run(q, scale, out, config, workspace=workspace)
        op.run(other_q, other_scale, other_out, config, workspace=workspace)
        op.run(q, scale, out, config, workspace=workspace)

    launch_pair()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
    assert torch.equal(other_out.view(torch.int16), other_eager.view(torch.int16))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch_pair()  # Warm module/stream before capture; scratch is already allocated.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch_pair()
    for _ in range(3):
        workspace[:partial_words].fill_(float("nan"))
        graph.replay()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        assert torch.equal(other_out.view(torch.int16), other_eager.view(torch.int16))
        assert torch.count_nonzero(workspace[partial_words:].view(torch.int32)) == 0


@pytest.mark.parametrize(
    "case,provider",
    [
        ("review-triton", "triton-fp8"),
        ("review-cute", "cute-fp8"),
        ("cute-transition", None),
        ("flashinfer-transition", None),
        ("cuda", "cuda-sm12x"),
    ],
)
@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_warm_up_captures_every_intermediate_size(case, provider, quantized):
    """No output calls before capture; references cannot warm later M signatures."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    capability = torch.cuda.get_device_capability()
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    n, k, max_m = {
        "review-triton": (256, 2048, 512),
        "review-cute": (256, 384, 64),
        "cute-transition": (9216, 2048, 64),
        "cuda": (256, 384, 64),
        # First measured CUTLASS transition on each qualified device.
        "flashinfer-transition": {
            (12, 0, 170): (9216, 512, 256),
            (12, 0, 188): (1536, 4096, 160),
            (12, 1, 48): (2304, 2048, 80),
        }.get((*capability, sms), (256, 2048, 64)),
    }[case]
    op = _out_op(n, k, provider)
    if case == "flashinfer-transition" and op._flashinfer is None:
        pytest.skip("no qualified measured FlashInfer transition on this device")
    if case == "cute-transition" and op._cute is None:
        pytest.skip("no qualified measured CuTe transition on this device")
    x = _random_activation(max_m, k)
    if quantized:
        # Build caller-supplied operands without warming any package kernels.
        x = x.clamp(-448, 448).to(E4M3)
    scales = torch.ones(max_m, _groups(k), device="cuda")
    parents = [
        torch.full((max_m, n + 8), -19, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    packed = torch.empty(max_m, n, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(max_m=max_m, quantized=quantized, out_row_stride=n + 8)
    torch.cuda.current_stream().wait_stream(stream)
    # Cover bucket boundaries and interior M classes without hundreds of graphs per provider.
    from ctikernels.linear.block_fp8 import _warm_up_sizes

    representatives = _warm_up_sizes(
        "test", max_m, None, lambda m: op._dispatch_key(m, quantized=quantized)
    )
    sizes = sorted(
        {m + delta for m in representatives for delta in (-1, 0, 1) if 1 <= m + delta <= max_m}
    )
    snapshots = []
    for m in sizes:
        outputs = (packed[:m], parents[0][:m, :n], parents[1][:m, 1 : n + 1])
        for out in outputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if quantized:
                    op.quantized(x[:m], scales[:m], out=out)
                else:
                    op(x[:m], out=out)
            graph.replay()
            graph.replay()
            # Keep snapshots on CPU so hundreds of graphs need not stay resident.
            snapshots.append((m, out.cpu()))
    assert torch.all(parents[0][:, n:] == -19)
    assert torch.all(parents[1][:, :1] == -19) and torch.all(parents[1][:, n + 1 :] == -19)
    reference = BlockFP8Linear(op._weight, op._scale, provider=provider)
    for index, m in enumerate(sizes):
        expected = (
            (reference.quantized(x[:m], scales[:m]) if quantized else reference(x[:m]))
            .cpu()
            .view(torch.uint8)
        )
        for captured_m, captured in snapshots[3 * index : 3 * (index + 1)]:
            assert captured_m == m
            assert torch.equal(captured.view(torch.uint8), expected), (case, m)


@torch.inference_mode()
def test_cute_warm_up_strided_then_capture_unaligned_base():
    """Exact review repro: no eager output call can prepare the packed fallback."""
    from ctikernels.linear.cute import _block_fp8_provider as cute_provider

    _require_cute()
    op = _out_op(256, 384, "cute-fp8")
    # Other tests may have launched this global signature; require warm_up to do it.
    config = cute_provider.CONFIGS[op._cute_config(4)]
    for strided in (False, True):
        cute_provider._launched.discard(op._cute._key(config, torch.bfloat16, strided))
    x = _random_activation(4, 384)
    parent = torch.full((4, 264), -19, device="cuda", dtype=torch.bfloat16)
    out = parent[:, 1:257]
    op.warm_up(batch_sizes=[4], out_row_stride=264)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(x, out=out)
    graph.replay()
    reference = BlockFP8Linear(op._weight, op._scale, provider="cute-fp8")
    assert torch.equal(out.view(torch.uint8), reference(x).view(torch.uint8))
    assert torch.all(parent[:, :1] == -19) and torch.all(parent[:, 257:] == -19)


@pytest.mark.parametrize("max_m", [1, 8])
@pytest.mark.parametrize("scale_stride", [1, 2])
@torch.inference_mode()
def test_flashinfer_copy_capture_is_exact(max_m, scale_stride):
    """Strided operands and outputs stage through per-call buffers (the graph's pool)."""
    op = _out_op(256, 384, "flashinfer-cutlass")
    q = _random_activation(8, 385).to(E4M3)[:, 1:]
    scale = (torch.rand(8, 3 * scale_stride + 1, device="cuda") + 0.25)[:, 1::scale_stride]
    parent = torch.full((8, 264), -19, device="cuda", dtype=torch.bfloat16)
    op.warm_up(
        max_m=max_m,
        quantized=True,
        out_row_stride=264,
        input_layouts=[((385, True), (*scale.stride(), True))],
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    snapshots = []
    for m in range(1, max_m + 1):
        out = parent[:m, 1:257]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op.quantized(q[:m], scale[:m], out=out)
        graph.replay()
        snapshots.append(out.cpu())
    reference = BlockFP8Linear(op._weight, op._scale, provider="flashinfer-cutlass")
    for m, captured in enumerate(snapshots, 1):
        assert torch.equal(captured, reference.quantized(q[:m], scale[:m]).cpu())


def _track_cold_triton_launches(monkeypatch):
    from ctikernels._runtime.launch import WarmRegistry
    from ctikernels.linear import block_fp8
    from ctikernels.linear.triton import _block_fp8 as kernels

    # Earlier tests warm the same classes; a fresh registry makes this warm_up launch again.
    monkeypatch.setattr(block_fp8, "_WARMED", WarmRegistry())

    class Kernel:
        def __init__(self, kernel):
            self.kernel, self.warmed = kernel, set()

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                key = (
                    tuple(
                        (a.dtype, a.data_ptr() % 16 == 0)
                        if isinstance(a, torch.Tensor)
                        else (a == 1, a % 16 == 0)
                        for a in args
                    ),
                    tuple(kwargs.items()),
                )
                if torch.cuda.is_current_stream_capturing():
                    assert key in self.warmed, "warm_up missed a Triton signature"
                else:
                    self.warmed.add(key)
                return self.kernel[grid](*args, **kwargs)

            return launch

    for name in ("_quantize_kernel", "_gemm_kernel", "_split_sum_kernel"):
        monkeypatch.setattr(kernels, name, Kernel(getattr(kernels, name)))


@pytest.mark.parametrize(
    "layout", ["packed", "row-padded", "row-padded-aligned", "misaligned", "ue8m0-scalar"]
)
@torch.inference_mode()
def test_warm_up_cold_capture_input_layouts(monkeypatch, layout):
    from ctikernels.linear.cute import _block_fp8_act_quant as quantizer

    ue8m0 = layout == "ue8m0-scalar"
    if ue8m0:
        _require_cute()
        monkeypatch.setattr(quantizer, "_ready", set())
    n, k, m = 256, 384 if ue8m0 else 2048, 32
    base = _out_op(n, k, "triton-fp8")
    op = BlockFP8Linear(base._weight, base._scale, provider="triton-fp8", scale_ue8m0=ue8m0)
    _track_cold_triton_launches(monkeypatch)
    pad = 0 if layout == "packed" else 16 if layout == "row-padded-aligned" else 1
    offset = int(layout in ("misaligned", "ue8m0-scalar"))
    parent = _random_activation(m, k + pad)
    x = parent[:, offset : offset + k]
    if layout == "misaligned":
        parent = _random_activation(1, m * k + 1)
        x = parent[:, 1:].view(m, k)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        routes = op.warm_up(batch_sizes=[m])
    if layout != "packed":
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(
                RuntimeError, match="input layout class.*row stride=.*warm_up.*input_layouts"
            ):
                op(x, out=out)
        with torch.cuda.stream(stream):
            routes += op.warm_up(
                batch_sizes=[m], input_layouts=[(x.stride(0), bool(x.data_ptr() % 16))]
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(x, out=out)
    graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    captured = out.clone()
    parent.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    # References are constructed only after both cold capture and changed-input replay.
    reference = BlockFP8Linear(base._weight, base._scale, provider="triton-fp8", scale_ue8m0=ue8m0)
    assert torch.equal(replayed.view(torch.uint8), reference(x).view(torch.uint8))
    assert not torch.equal(captured, replayed)
    assert {r["provider"] for r in routes} == {op.provider_for(i) for i in range(1, m + 1)}


# FlashInfer takes MN-major scales natively, so it has no strided-scales class.
@pytest.mark.parametrize(
    ("provider", "layout"),
    [
        *(
            ("triton-fp8", layout)
            for layout in ("packed", "row-padded", "misaligned", "strided-scales")
        ),
        *(("flashinfer-cutlass", layout) for layout in ("packed", "row-padded", "misaligned")),
    ],
)
@torch.inference_mode()
def test_warm_up_cold_capture_quantized_layouts(monkeypatch, provider, layout):
    op = _out_op(256, 384, provider)
    _track_cold_triton_launches(monkeypatch)
    q = _random_activation(17, 385 if layout == "row-padded" else 384).to(E4M3)[:, :384]
    if layout == "misaligned":
        q = _random_activation(1, 17 * 384 + 1).to(E4M3)[:, 1:].view(17, 384)
    scale = (
        torch.ones(17, 7, device="cuda")[:, 1::2]
        if layout == "strided-scales"
        else torch.ones(17, 3, device="cuda")
    )
    out = torch.empty(17, 256, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        routes = op.warm_up(batch_sizes=[17], quantized=True)
    if layout != "packed":
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
            with pytest.raises(
                RuntimeError, match="input layout class.*scale strides=.*warm_up.*input_layouts"
            ):
                op.quantized(q, scale, out=out)
        with torch.cuda.stream(stream):
            routes += op.warm_up(
                batch_sizes=[17],
                quantized=True,
                input_layouts=[
                    (
                        (q.stride(0), bool(q.data_ptr() % 16)),
                        (*scale.stride(), bool(scale.data_ptr() % 16)),
                    )
                ],
            )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.quantized(q, scale, out=out)
    graph.replay()
    torch.cuda.synchronize()
    reference = BlockFP8Linear(op._weight, op._scale, provider=provider)
    assert torch.equal(out.view(torch.uint8), reference.quantized(q, scale).view(torch.uint8))
    assert {r["provider"] for r in routes} == {provider}


@torch.inference_mode()
def test_workspace_capture_uses_warmed_caller_layout():
    # warm_up keys on the caller's layout; the workspace path stages misaligned q
    # into its packed view and must accept the same warmed layout.
    op = _out_op(256, 256, "cute-fp8")
    q = _random_activation(2, 257).to(E4M3)[:, 1:]
    scale = torch.ones(2, 2, device="cuda")
    out = torch.empty(2, 256, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(op.workspace_bytes(2) // 4, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.warm_up(batch_sizes=[2], quantized=True, input_layouts=[(257, True)])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.quantized(q, scale, out=out, workspace=workspace)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), op.quantized(q, scale).view(torch.uint8))


@torch.inference_mode()
def test_warm_up_all_cold_capture_quantized_layouts(monkeypatch):
    # The only GPU exhaustive-layout warm-up: one provider, small shape and one M.
    from ctikernels.linear.block_fp8 import _layout_tensor

    op = _out_op(128, 128, "triton-fp8")
    _track_cold_triton_launches(monkeypatch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        routes = op.warm_up(batch_sizes=[2], quantized=True, input_layouts="all")
    assert len({r["input_layout"] for r in routes}) == len(routes)
    for row, offset, sr, sc, so in ((129, 1, 2, 2, 1), (0, 0, 0, 0, 0), (1, 1, 1, 16, 0)):
        q = _layout_tensor(2, 128, (row, 1, offset), dtype=E4M3, device="cuda")
        scale = _layout_tensor(2, 1, (sr, sc, so), dtype=torch.float32, device="cuda", scale=True)
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op.quantized(q, scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op.quantized(q, scale))


@torch.inference_mode()
def test_flashinfer_warm_up_retains_no_scratch():
    op = _out_op(9216, 2048, "flashinfer-cutlass")
    # Prime library-global workspaces on another instance, never the measured one.
    reference = BlockFP8Linear(op._weight, op._scale, provider="flashinfer-cutlass")
    reference.warm_up(64, quantized=True, out_row_stride=9217)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    op.warm_up(64, quantized=True, out_row_stride=9217)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before <= 1024 * 1024  # staging is per call
    q = torch.zeros(64, 2048, dtype=E4M3, device="cuda")
    scale = torch.ones(64, 16, device="cuda")
    out = torch.empty(64, 9217, dtype=torch.bfloat16, device="cuda")[:, 1:]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.quantized(q, scale, out=out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize(
    "case,provider",
    [
        ("triton-fp8", "triton-fp8"),
        ("cuda-sm12x", "cuda-sm12x"),
        ("cute-fp8", "cute-fp8"),
        ("flashinfer", "flashinfer-cutlass"),
        ("auto", None),
        ("cute-tailsplit", "cute-fp8"),
        ("ue8m0", "triton-fp8"),
    ],
    ids=[
        "triton-fp8",
        "cuda-sm12x",
        "cute-fp8",
        "flashinfer",
        "auto",
        "cute-tailsplit",
        "ue8m0",
    ],
)
@torch.inference_mode()
def test_engine_workspace_bitwise_and_allocation_free_capture(case, provider, monkeypatch):
    op = _out_op(2048, 2048, provider)
    if case == "cute-tailsplit":
        monkeypatch.setattr(op, "_cute_config", lambda _: "swap128x32s2c2-tailsplit-mx")
    if case == "ue8m0":
        from ctikernels.linear.cute._block_fp8_provider import available

        if not available(torch.cuda.get_device_capability()):
            pytest.skip("UE8M0 quantization needs CuTe")
        op = BlockFP8Linear(op._weight, op._scale, provider=provider, scale_ue8m0=True)
    workspace = torch.empty(op.workspace_bytes(512) // 4, device="cuda", dtype=torch.float32)
    for dtype in (torch.bfloat16, torch.float16):
        for m in (1, 17, 256, 320, 512):
            x = _random_activation(m, 2049).to(dtype)[:, 1:]
            q, scale = quantize_per_token_group_fp8(x)
            q_parent = torch.empty((m, 2049), dtype=E4M3, device="cuda")
            q_parent[:, 1:].copy_(q)
            q = q_parent[:, 1:]
            scale_parent = torch.empty((m, 33), device="cuda")
            scale_parent[:, 1::2].copy_(scale)
            scale = scale_parent[:, 1::2]
            parent = torch.full((m, 2051), -19, device="cuda", dtype=dtype)
            out = parent[:, 1:2049]
            for quantized in (False, True):

                def call(
                    buffer=None,
                    target=None,
                    *,
                    quantized=quantized,
                    q=q,
                    scale=scale,
                    x=x,
                    dtype=dtype,
                ):
                    if quantized:
                        return op.quantized(q, scale, out_dtype=dtype, out=target, workspace=buffer)
                    return op(x, out_dtype=dtype, out=target, workspace=buffer)

                expected = call().clone()
                assert call(workspace, out) is out
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                first = call(workspace)
                saved = first.clone()
                assert first.untyped_storage().data_ptr() != workspace.untyped_storage().data_ptr()
                assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
                # A different input must not overwrite the preceding layer's result.
                if quantized:
                    second = op.quantized(
                        torch.zeros_like(q), scale, out_dtype=dtype, workspace=workspace
                    )
                else:
                    second = op(torch.zeros_like(x), out_dtype=dtype, workspace=workspace)
                assert second.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
                assert torch.equal(first.view(torch.uint8), saved.view(torch.uint8))
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    # Workspace repairs some providers' views; Triton keeps the
                    # original strides. Register both launch classes explicitly.
                    example = (
                        (
                            (q.stride(0), bool(q.data_ptr() % 16)),
                            (*scale.stride(), bool(scale.data_ptr() % 16)),
                        )
                        if quantized
                        else (x.stride(0), bool(x.data_ptr() % 16))
                    )
                    op.warm_up(
                        batch_sizes=[m],
                        dtype=dtype,
                        out_dtype=dtype,
                        quantized=quantized,
                        out_row_stride=out.stride(0),
                        input_layouts=[(op.in_features, False), example],
                    )
                    call(workspace, out)
                    with torch.cuda.graph(graph, stream=stream):
                        before = torch.cuda.memory_stats()["allocation.all.allocated"]
                        result = call(workspace, out)
                        after = torch.cuda.memory_stats()["allocation.all.allocated"]
                torch.cuda.current_stream().wait_stream(stream)
                assert after == before
                workspace.fill_(float("nan"))
                graph.replay()
                assert result is out
                assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
                assert torch.all(parent[:, 0] == -19) and torch.all(parent[:, 2049:] == -19)


@torch.inference_mode()
def test_cute_tailsplit_graph_workspace_shared_by_operations():
    """In a GraphWorkspace, operations share one tailsplit buffer per config, replay serially."""
    from ctikernels import GraphWorkspace

    _require_cute()
    config = "normal128x128s2-tailsplit-mx"
    weights = [_random_weight(256, 896) for _ in range(2)]
    ops = [BlockFP8Linear(w, s, provider="cute-fp8")._cute for w, s in weights]
    operands = [quantize_per_token_group_fp8(_random_activation(65, 896)) for _ in range(2)]
    outputs = [torch.empty(65, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = [torch.cuda.CUDAGraph() for _ in ops]
    with GraphWorkspace(max_rows=65) as group, torch.cuda.stream(stream):
        for op in ops:
            op.prepare(config)
        assert len({op._workspace(config).data_ptr() for op in ops}) == 1
        assert group.nbytes > 0
        eager = []
        for op, (q, scale), out in zip(ops, operands, outputs, strict=True):
            op.run(q, scale, out, config)
            eager.append(out.clone())
        for graph, op, (q, scale), out in zip(graphs, ops, operands, outputs, strict=True):
            with torch.cuda.graph(graph, stream=stream):
                op.run(q, scale, out, config)
    with torch.cuda.stream(stream):
        for _ in range(2):
            for graph, out, expected in zip(graphs, outputs, eager, strict=True):
                out.zero_()
                graph.replay()
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    torch.cuda.current_stream().wait_stream(stream)
    group.close()


# ----------------------------------------------------------------------------- dgrad


def _dgrad_bound(dy, weight, scale, expected) -> float:
    """Relative max error allowed vs FP64: the fused kernel's 2^-7 up to 64 rows; past them the
    market's own composition (dequant to dy's dtype + torch.mm with cuBLAS defaults) sets it."""
    if dy.shape[0] <= 64:
        return 2**-7
    market = dy @ dequantize_block_fp8(weight, scale, dtype=dy.dtype)
    return 2 * ((market.double() - expected).abs().max() / expected.abs().max()).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2048, 512), (4352, 5120), (200, 328)])
@pytest.mark.parametrize("m", [0, 1, 5, 64, 300])
def test_dgrad_matches_fp64_and_is_deterministic(dtype, shape, m) -> None:
    """dx = dy . W_hat against FP64 (decoded weight times its block scales), from a column
    slice of a wider gradient; two calls agree bit for bit; UE8M0 scales match FP32 ones."""
    torch.manual_seed(m + shape[0])
    n, k = shape
    weight, scale = _random_weight(n, k)
    op = BlockFP8Linear(weight, scale)
    wide = (torch.randn(m, n + 24, device="cuda") * 0.1).to(dtype)
    dy = wide[:, 8 : 8 + n]
    dx = op.dgrad(dy)
    assert dx.shape == (m, k) and dx.dtype == dtype
    assert torch.equal(op.dgrad(dy), dx)
    if not m:
        return
    blocks = scale.double().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    expected = dy.double() @ (weight.double() * blocks)
    error = (dx.double() - expected).abs().max() / expected.abs().max()
    assert error < _dgrad_bound(dy, weight, scale, expected), error
    out = torch.full((m, k + 3), 7.0, dtype=dtype, device="cuda")[:, :k]
    assert op.dgrad(dy, out=out) is out and torch.equal(out, dx)
    ue8m0 = torch.exp2(torch.round(torch.log2(scale)))
    op2 = BlockFP8Linear(weight, ue8m0.to(torch.float8_e8m0fnu))
    assert torch.equal(op2.dgrad(dy), BlockFP8Linear(weight, ue8m0).dgrad(dy))


@pytest.mark.parametrize("k", [640, 600])
@torch.inference_mode()
def test_dgrad_column_chunks(monkeypatch, k) -> None:
    """Past 64 rows W is decoded in column chunks (forced to 128 columns here, with a tail
    chunk): FP64 agreement, run-to-run bits, accumulate, and a strided out."""
    from ctikernels.linear import block_fp8

    torch.manual_seed(k)
    n, m = 1024, 200
    weight, scale = _random_weight(n, k)
    op = BlockFP8Linear(weight, scale)
    monkeypatch.setattr(block_fp8, "_dgrad_budget", lambda rows: n * 128 * 2)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    dx = op.dgrad(dy)
    assert torch.equal(op.dgrad(dy), dx)
    blocks = scale.double().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    expected = dy.double() @ (weight.double() * blocks)
    bound = _dgrad_bound(dy, weight, scale, expected)
    assert ((dx.double() - expected).abs().max() / expected.abs().max()).item() < bound
    prior = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = prior.clone()
    op.dgrad(dy, out=out, accumulate=True)
    total = expected + prior.double()
    assert ((out.double() - total).abs().max() / total.abs().max()).item() < bound
    strided = torch.empty(m, k + 8, device="cuda", dtype=torch.bfloat16)[:, 8:]
    assert torch.equal(op.dgrad(dy, out=strided), dx)


@pytest.mark.parametrize("m", [0, 1, 37])
def test_quantizer_writes_given_row_strided_outputs(m) -> None:
    x = _random_activation(m, 640)
    expected = quantize_per_token_group_fp8(x)
    q = torch.empty(m, 700, dtype=E4M3, device="cuda")[:, :640]
    scale = torch.empty(m, 9, device="cuda")[:, :5]
    assert quantize_per_token_group_fp8(x, q=q, scale=scale) == (q, scale)
    assert torch.equal(q.view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(scale, expected[1])


# ----------------------------------------------------------------------------- BF16 mode


@pytest.mark.parametrize("m", [1, 4, 37, 300])
@pytest.mark.parametrize("split_k", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_bf16_mode_gemm(m, split_k, dtype) -> None:
    """The 16-bit mode of the block-FP8 GEMM kernel: FP32 accumulation of x . W^T, rounded
    once, split-K partials summed in order; deterministic."""
    torch.manual_seed(m + split_k)
    n, k = 384, 1024 + 64
    x = torch.randn(m, k + 8, device="cuda", dtype=dtype)[:, :k]
    w = torch.randn(n, k, device="cuda", dtype=dtype) * 0.05
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    config = (16 if m <= 16 else 64, 64, split_k, 4, 3)
    _block_fp8.gemm(x, None, w, None, out, config=config)
    expected = x.double() @ w.double().T
    assert ((out.double() - expected).abs().max() / expected.abs().max()).item() < 2**-7
    again = torch.empty_like(out)
    _block_fp8.gemm(x, None, w, None, again, config=config)
    assert torch.equal(out, again)


@pytest.mark.parametrize("m", [1, 37, 300])
@pytest.mark.parametrize("accumulate", [False, True])
@torch.inference_mode()
def test_bf16_mode_dgrad_and_accumulate(m, accumulate) -> None:
    torch.manual_seed(m)
    n, k = 640, 384
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
    prior = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    dx = prior.clone()
    _block_fp8.dgrad(dy, w, None, dx, accumulate=accumulate)
    expected = dy.double() @ w.double() + (prior.double() if accumulate else 0)
    assert ((dx.double() - expected).abs().max() / expected.abs().max()).item() < 2**-7


@pytest.mark.parametrize("m", [9, 300])
@torch.inference_mode()
def test_block_fp8_dgrad_accumulates_into_out(m) -> None:
    weight, scale = _random_weight(512, 384)
    op = BlockFP8Linear(weight, scale)
    dy = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
    prior = torch.randn(m, 384, device="cuda", dtype=torch.bfloat16)
    out = prior.clone()
    assert op.dgrad(dy, out=out, accumulate=True) is out
    # out rounds prior + dy . W_hat once: judge it against FP64, like the plain dgrad.
    product = dy.double() @ _weight64(weight, scale)
    expected = prior.double() + product
    error = ((out.double() - expected).abs().max() / expected.abs().max()).item()
    assert error < max(2**-7, _dgrad_bound(dy, weight, scale, product))
    with pytest.raises(ValueError, match="accumulate"):
        op.dgrad(dy, accumulate=True)


@pytest.mark.parametrize("m", [0, 5, 130])
@torch.inference_mode()
def test_public_dgrad_bf16(m) -> None:
    from ctikernels.linear.block_fp8 import dgrad_bf16

    torch.manual_seed(m)
    w = (torch.randn(256, 384, device="cuda") * 0.05).bfloat16()
    dy = torch.randn(m, 256 + 8, device="cuda", dtype=torch.bfloat16)[:, :256]
    dx = dgrad_bf16(dy, w)
    expected = dy.double() @ w.double()
    assert dx.shape == (m, 384)
    if m:
        assert ((dx.double() - expected).abs().max() / expected.abs().max()).item() < 2**-7
        out = torch.ones(m, 384, device="cuda", dtype=torch.bfloat16)
        assert dgrad_bf16(dy, w, out=out, accumulate=True) is out
        torch.testing.assert_close(out.double(), expected + 1, rtol=2**-7, atol=2e-2)


@pytest.mark.parametrize("m", [1, 7, 300])
@pytest.mark.parametrize("k", [128, 2048, 7168])
@torch.inference_mode()
def test_ue8m0_quantizer_matches_the_reference_on_every_device(m, k) -> None:
    """``BlockFP8Linear(scale_ue8m0=True)`` input quantization: CuTe on sm120/121, our Triton
    elsewhere; both reproduce the CPU reference's power-of-two scales bit for bit."""
    from ..modules._block_fp8_gated_reference import quantize

    weight, scale = _random_weight(256, k)
    op = BlockFP8Linear(weight, scale, scale_ue8m0=True)
    x = _random_activation(m, k) * 37
    x[0, :128] = 0  # the amax floor
    q, s = op._quantize_ue8m0(x)
    expected_q, expected_s = quantize(x.cpu(), scale_ue8m0=True)
    assert torch.equal(q.cpu().view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(s.cpu(), expected_s)
    q2, s2 = torch.empty_like(q), torch.empty_like(s)
    _block_fp8.quantize(x, q2, s2, scale_ue8m0=True)  # the Triton kernel on sm12x too
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8)) and torch.equal(s2, s)
    torch.testing.assert_close(op(x), op.quantized(q, s), rtol=0, atol=0)


@torch.inference_mode()
def test_graph_workspace_shares_one_scratch_slot_across_layers() -> None:
    """64 same-shape layers in one GraphWorkspace: their staging/quantizer/split-K/tail-split
    scratch is one slot of one op's workspace_bytes(max_rows), not 64 instance buffers, and
    eager and replayed outputs are the bits of the workspace-free calls."""
    from ctikernels import GraphWorkspace

    torch.manual_seed(0)
    ops = [BlockFP8Linear(*_random_weight(1024, 2048)) for _ in range(64)]
    xs = {m: _random_activation(m, 2048) for m in (1, 17, 64, 512)}
    expected = {m: [op(x) for op in ops] for m, x in xs.items()}
    group = GraphWorkspace(max_rows=512)
    outs = [torch.empty(512, 1024, device="cuda", dtype=torch.bfloat16) for _ in ops]
    graph = torch.cuda.CUDAGraph()
    with group:
        for op in ops:
            op.warm_up(batch_sizes=list(xs))
        eager = {m: [op(x) for op in ops] for m, x in xs.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for op, out in zip(ops, outs, strict=True):
                op(xs[512], out=out)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            for op, out in zip(ops, outs, strict=True):
                op(xs[512], out=out)
    for out in outs:
        out.zero_()
    graph.replay()
    assert group.nbytes == -(-ops[0].workspace_bytes(512) // 4) * 4
    for op in ops:
        assert not hasattr(op._flashinfer, "_temporaries") and not hasattr(op._cute, "_workspaces")
    for m in xs:
        for a, b in zip(eager[m], expected[m], strict=True):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), m
    for a, b in zip(outs, expected[512], strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    del graph
    group.close()


@pytest.mark.parametrize(("checks", "n"), [("", 1152), ("1", 1408)])
@torch.inference_mode()
def test_warm_up_launches_once_per_class_and_still_captures_largest_first(
    monkeypatch, checks, n
) -> None:
    """A second instance of an already-warmed class launches nothing in warm_up (compiles and
    module loads are process-wide), yet records its coverage and reserves its GraphWorkspace
    slot: a largest-first capture through it alone replays the eager bits. Another shape still
    warms. Shapes are unique to this test (the registry is process-wide)."""
    from ctikernels import GraphWorkspace

    monkeypatch.setenv("CTI_ENGINE_CHECKS", checks)
    original = BlockFP8Linear.__call__
    calls = []

    def counting(self, *args, **kwargs):
        calls.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BlockFP8Linear, "__call__", counting)
    torch.manual_seed(n)
    k, sizes = 2304, [512, 64, 17, 1]
    first, second = (BlockFP8Linear(*_random_weight(n, k)) for _ in range(2))
    other = BlockFP8Linear(*_random_weight(n + 128, k))
    first.warm_up(batch_sizes=sizes)
    second.warm_up(batch_sizes=sizes)
    other.warm_up(batch_sizes=sizes)
    assert calls.count(id(first)) > 0 and calls.count(id(other)) > 0
    assert calls.count(id(second)) == 0
    xs = {m: _random_activation(m, k) for m in sizes}
    expected = {m: original(second, x) for m, x in xs.items()}
    group = GraphWorkspace(max_rows=512)
    outs = {m: torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for m in sizes}
    graphs = {}
    with group:
        second.warm_up(batch_sizes=sizes)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for m in sizes:  # largest first, no eager call through `second` inside the group
            graphs[m] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[m], stream=stream):
                second(xs[m], out=outs[m])
        torch.cuda.current_stream().wait_stream(stream)
    for m in sizes:
        outs[m].zero_()
        graphs[m].replay()
        assert torch.equal(outs[m].view(torch.uint8), expected[m].view(torch.uint8)), m
    del graphs
    group.close()


@torch.inference_mode()
def test_dequantize_fills_a_caller_buffer_and_ops_expose_their_weight() -> None:
    """``dequantize_block_fp8(out=)`` writes a strided slice of caller scratch with the bits of
    a fresh decode; ``weight`` / ``weight_scale`` are the tensors the GEMMs read."""
    weight, scale = _random_weight(512, 1024)
    expected = dequantize_block_fp8(weight, scale)
    scratch = torch.full((512, 1536), -7.0, device="cuda", dtype=torch.bfloat16)
    out = scratch[:, 256:1280]
    assert dequantize_block_fp8(weight, scale, out=out) is out
    assert torch.equal(out, expected) and torch.all(scratch[:, :256] == -7)
    op = BlockFP8Linear(weight, scale)
    assert op.weight is weight and op.weight_scale is scale
    ue8m0 = BlockFP8Linear(weight, scale.to(torch.float8_e8m0fnu))
    assert ue8m0.weight_scale.dtype == torch.float32
    assert torch.equal(ue8m0.weight_scale, scale.to(torch.float8_e8m0fnu).float())
    with pytest.raises(ValueError, match="out must be"):
        dequantize_block_fp8(weight, scale, out=torch.empty(512, 1024, device="cuda"))
