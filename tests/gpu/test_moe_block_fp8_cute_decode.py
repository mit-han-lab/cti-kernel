# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Decode GPU qualification. Run explicitly on SM120/121; never during offline checks."""

from functools import lru_cache

import pytest
import torch

from ctikernels._moe_block_fp8_cute_table import PRESETS
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE, _dsl_available

from .test_moe_block_fp8_cute import exact_fp32, random_weights, reference, rms  # noqa: F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or not _dsl_available(),
    reason="requires sm120/121 and nvidia-cutlass-dsl==4.8.0",
)
DECODE_PRESETS = [name for name, config in PRESETS.items() if config.path == "decode"]


@pytest.fixture(scope="module")
def decode_model():
    torch.manual_seed(817)
    params = random_weights(256, 2048, 512)
    op = BlockFP8MoE(*params, provider="cute-fp8")
    triton = BlockFP8MoE(*params, provider="triton-fp8")

    @lru_cache(maxsize=None)
    def case(m, skewed):
        generator = torch.Generator(device="cuda").manual_seed(107 + m)
        x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
        x[:, :128] = 0
        x[:, 128:256] *= 1e-12  # Exercise the amax floor with nonzero inputs.
        if skewed:
            # Duplicates within a token AND across all tokens; only two experts.
            ids = (torch.arange(8, device="cuda", dtype=torch.int32) % 2).repeat(m, 1)
        else:
            ids = torch.randperm(256, device="cuda", generator=generator)[: m * 8].view(m, 8).int()
        ids.view(-1)[::11] = -1
        if m > 1:
            ids[-1] = -1
        weights = torch.rand(m, 8, device="cuda", generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        oracle = reference(x, weights, ids, params)
        baseline = triton(x, weights, ids)
        return x, weights, ids, oracle, baseline

    return op, case


@pytest.mark.parametrize("preset", DECODE_PRESETS)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("skewed", [False, True])
@torch.inference_mode()
def test_decode_accuracy_determinism_and_graph(decode_model, preset, m, skewed):
    op, case = decode_model
    x, weights, source_ids, oracle, baseline = case(m, skewed)
    ids = source_ids.clone()
    op.set_config(preset)
    actual = op(x, weights, ids)
    assert torch.equal(actual, op(x, weights, ids))
    error, floor = rms(actual.float() - oracle), rms(baseline.float() - oracle)
    assert error <= 1.05 * floor + 1e-30, (preset, m, skewed, error, floor)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weights, ids)
    graph.replay()
    assert torch.equal(actual, captured)
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    # Change owner slots, not just the expert value, to detect stale h/routing.
    ids.copy_(source_ids.flip(-1))
    graph.replay()
    assert torch.equal(captured, op(x, weights, ids))
    ids.copy_(source_ids)
    graph.replay()
    assert torch.equal(actual, captured)


@pytest.mark.parametrize("preset", DECODE_PRESETS)
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@torch.inference_mode()
def test_decode_fc1_quantization_bitwise(decode_model, preset, m):
    from ctikernels._moe_block_fp8_cute_decode import forward

    op, case = decode_model
    x, weights, ids, _, _ = case(m, True)
    internal, h = forward(op, x, weights, ids, PRESETS[preset], return_intermediates=True)
    external, hq = forward(
        op,
        x,
        weights,
        ids,
        PRESETS[preset],
        return_intermediates=True,
        quantized_x=quantize_per_token_group_fp8(x),
    )
    # The two variants differ only in quantization. Compare exact BF16 bits
    # after identical FC1 MMA/scaling/activation, as well as the final output.
    active = ids.flatten() >= 0
    assert torch.equal(h[active].view(torch.int16), hq[active].view(torch.int16))
    assert torch.equal(internal, external)


@pytest.mark.parametrize("preset", DECODE_PRESETS)
@torch.inference_mode()
def test_decode_ones_and_prepare(preset):
    # Small aligned shape exercises K=1 with a three-stage pipeline, including
    # FC2 transitions between experts before all ring stages have been used.
    params = (
        torch.ones(3, 256, 256, device="cuda").to(torch.float8_e4m3fn),
        torch.full((3, 2, 2), 1 / 128, device="cuda"),
        torch.ones(3, 256, 128, device="cuda").to(torch.float8_e4m3fn),
        torch.full((3, 2, 1), 1 / 128, device="cuda"),
    )
    op = BlockFP8MoE(*params, top_k=2, provider="cute-fp8").set_config(preset)
    op.prepare([3])
    x = torch.ones(3, 256, device="cuda", dtype=torch.bfloat16)
    weights = torch.full((3, 2), 0.5, device="cuda")
    ids = torch.tensor([[0, 0], [0, 2], [-1, -1]], dtype=torch.int32, device="cuda")
    actual = op(x, weights, ids)
    oracle = reference(x, weights, ids, params)
    baseline = BlockFP8MoE(*params, top_k=2, provider="triton-fp8")(x, weights, ids)
    assert rms(actual.float() - oracle) <= 1.05 * rms(baseline.float() - oracle) + 1e-30
    assert torch.equal(actual[0], actual[1])
    assert torch.count_nonzero(actual[2]).item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weights, ids)
    graph.replay()
    assert torch.equal(captured, actual)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@torch.inference_mode()
def test_decode_nonfinite_quantization(decode_model, value):
    from ctikernels._moe_block_fp8_cute_decode import forward

    op, case = decode_model
    x, weights, ids, _, _ = case(2, True)
    x = x.clone()
    x[0, 17] = value
    cfg = PRESETS["decode-n16"]
    _, actual = forward(op, x, weights, ids, cfg, return_intermediates=True)
    _, expected = forward(
        op,
        x,
        weights,
        ids,
        cfg,
        return_intermediates=True,
        quantized_x=quantize_per_token_group_fp8(x),
    )
    active = ids.flatten() >= 0
    assert torch.isnan(actual[active]).all()
    assert torch.equal(torch.isnan(actual[active]), torch.isnan(expected[active]))
