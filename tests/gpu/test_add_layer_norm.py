# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""add_layer_norm_ on the device against torch's add and F.layer_norm (the Qwen3.5 vision tower).

The residual is bitwise torch's add. The norm is within an ulp of torch's F.layer_norm on the
stored sum, the spacing taken at the larger of the two values and max(|weight * normalized|,
|bias|) (the affine terms can cancel), and passes the shared FP64 gate. On large row chunks its
mean FP64 error is no worse than torch's, up to a tie tolerance.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from ctikernels.norm import inference

from ..support.accuracy import assert_accurate, assert_bitwise, assert_within_ulps, ulps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

EPS = 1e-6
# Rows per FP64 check chunk: bounds the oracle's scratch to about 2**22 elements.
CHUNK_ELEMENTS = 1 << 22
# Chunks this large make enough rounding decisions for their mean FP64 error to be stable.
MEAN_CHECK_ELEMENTS = 1 << 20
DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.float16, torch.float16),
]


def make(rows, hidden, dtype, weight_dtype, seed, pad=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (rows, hidden + pad)
    x = (torch.randn(shape, generator=g, device="cuda") * 2).to(dtype)
    residual = (torch.randn(shape, generator=g, device="cuda") * 4 + 1).to(dtype)
    weight = (torch.randn(hidden, generator=g, device="cuda") * 0.5 + 1).to(weight_dtype)
    bias = torch.randn(hidden, generator=g, device="cuda").to(weight_dtype)
    return x, residual, weight, bias


def torch_layer_norm(stored, weight, bias):
    """nn.LayerNorm on the stored sum: in its own dtype with matching parameters, else FP32."""
    hidden = (stored.shape[-1],)
    if weight.dtype == stored.dtype:
        return F.layer_norm(stored, hidden, weight, bias, EPS)
    return F.layer_norm(stored.float(), hidden, weight, bias, EPS).to(stored.dtype)


def oracle(stored, weight, bias):
    """FP64 ``(output, max(|weight * normalized|, |bias|))``."""
    values = stored.double()
    centered = values - values.mean(-1, keepdim=True)
    normalized = centered / (centered.square().mean(-1, keepdim=True) + EPS).sqrt()
    scaled = weight.double() * normalized
    return scaled + bias.double(), torch.maximum(scaled.abs(), bias.double().abs())


def check_norm(actual, stored, weight, bias):
    """Per row chunk: the ulp rule, the FP64 gate and, on large chunks, a mean FP64 error no
    worse than torch's."""
    baseline = torch_layer_norm(stored, weight, bias)
    rows = max(1, CHUNK_ELEMENTS // stored.shape[-1])
    for start in range(0, stored.shape[0], rows):
        chunk = slice(start, start + rows)
        reference, scale = oracle(stored[chunk], weight, bias)
        assert_within_ulps(actual[chunk], baseline[chunk], scale, name=f"rows {start}+")
        assert_accurate(actual[chunk], reference, baseline[chunk], family="norm")
        if actual[chunk].numel() >= MEAN_CHECK_ELEMENTS:
            ours = float((actual[chunk].double() - reference).abs().sum())
            theirs = float((baseline[chunk].double() - reference).abs().sum())
            # 1e-6: a tie tolerance at the reduction-order noise floor (ties seen at 1e-8-1e-11).
            assert ours <= theirs * (1 + 1e-6), f"rows {start}+: FP64 error {ours!r} > {theirs!r}"


@pytest.mark.parametrize("dtype,weight_dtype", DTYPES)
@pytest.mark.parametrize("hidden", [1152, 4096, 1155])
@pytest.mark.parametrize("rows", [1, 255, 256, 4097, 65536])
def test_matches_torch_add_and_layer_norm(rows, hidden, dtype, weight_dtype):
    x, residual, weight, bias = make(rows, hidden, dtype, weight_dtype, rows + hidden)
    summed = residual + x
    inference.add_layer_norm_(x, residual, weight, bias, EPS)
    assert inference.last_standalone_provider["add_layer_norm_"] == "triton"
    assert_bitwise(residual, summed, name="residual")
    check_norm(x, summed, weight, bias)


@pytest.mark.parametrize("store_residual", [True, False])
@pytest.mark.parametrize("hidden", [1152, 1155, 10000])  # 10000: the chunked path
def test_strided_rows_and_store_residual(hidden, store_residual):
    pad = 24
    x_buffer, residual_buffer, weight, bias = make(
        300, hidden, torch.bfloat16, torch.bfloat16, 7, pad
    )
    x_before, residual_before = x_buffer.clone(), residual_buffer.clone()
    x, residual = x_buffer[:, pad:], residual_buffer[:, :hidden]
    summed = residual + x
    inference.add_layer_norm_(x, residual, weight, bias, EPS, store_residual=store_residual)
    assert_bitwise(residual, summed if store_residual else residual_before[:, :hidden])
    check_norm(x, summed, weight, bias)
    assert torch.equal(x_buffer[:, :pad], x_before[:, :pad])
    assert torch.equal(residual_buffer[:, hidden:], residual_before[:, hidden:])


def test_row_bits_do_not_depend_on_row_count():
    x, residual, weight, bias = make(4097, 1152, torch.bfloat16, torch.bfloat16, 3)
    one_x, one_residual = x[:1].clone(), residual[:1].clone()
    inference.add_layer_norm_(x, residual, weight, bias, EPS)
    inference.add_layer_norm_(one_x, one_residual, weight, bias, EPS)
    assert_bitwise(one_x, x[:1])
    assert_bitwise(one_residual, residual[:1])


def test_subnormal_summands_add_like_torch():
    x, residual, weight, bias = make(8, 1152, torch.bfloat16, torch.bfloat16, 5)
    x[:, :64] = torch.tensor(-2.5e-40, dtype=x.dtype)
    residual[:, :32] = torch.tensor(1e-40, dtype=x.dtype)
    summed = residual + x
    inference.add_layer_norm_(x, residual, weight, bias, EPS)
    assert_bitwise(residual, summed)


def test_graph_capture_replays_eager_bits():
    x0, residual0, weight, bias = make(1024, 1152, torch.bfloat16, torch.bfloat16, 11)
    x, residual = x0.clone(), residual0.clone()
    inference.add_layer_norm_(x, residual, weight, bias, EPS)  # compiles eagerly
    eager = x.clone(), residual.clone()
    x.copy_(x0)
    residual.copy_(residual0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        inference.add_layer_norm_(x, residual, weight, bias, EPS)
    x.copy_(x0)
    residual.copy_(residual0)
    graph.replay()
    torch.cuda.synchronize()
    assert_bitwise(x, eager[0])
    assert_bitwise(residual, eager[1])


def test_empty_does_not_launch():
    x = torch.empty(0, 1152, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(1152, device="cuda")
    inference.add_layer_norm_(x, torch.empty_like(x), weight, torch.zeros_like(weight), EPS)
    assert inference.last_standalone_provider["add_layer_norm_"] == "empty"


def test_ulp_spacing_on_the_device_is_exact():
    for dtype in (torch.bfloat16, torch.float16):
        info = torch.finfo(dtype)
        low, high = math.log2(info.tiny) - 2, math.log2(info.max)
        values = torch.logspace(low, high, 4099, base=2, dtype=torch.float64)
        assert torch.equal(ulps(values.cuda(), dtype).cpu(), ulps(values, dtype))
