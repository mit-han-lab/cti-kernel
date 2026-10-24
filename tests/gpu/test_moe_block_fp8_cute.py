# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Grouped FP8 qualification on SM120/121, including replay with changed routing.

Session-scoped weights/cases and the symbolic kernel cache amortize compilation.
The oracle uses the same quantizer and decodes one expert at a time, never a
persistent full BF16/FP32 copy of all weights. Odd H/I are rejected by the API;
there is also a small aligned shape (E=3, top_k=2, H=256, I=128).
"""

from functools import lru_cache

import pytest
import torch

from ctikernels._moe_block_fp8_cute_table import PRESETS
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import BlockFP8MoE, _dsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or not _dsl_available(),
    reason="requires sm120/121 and nvidia-cutlass-dsl==4.8.0",
)


@pytest.fixture(scope="module", autouse=True)
def exact_fp32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def random_weights(e, h, i):
    result = []
    for n, k in ((2 * i, h), (h, i)):
        # Build per expert to keep the fixture's peak memory close to checkpoint size.
        w = torch.empty((e, n, k), dtype=torch.float8_e4m3fn, device="cuda")
        for expert in range(e):
            w[expert].copy_((torch.randn(n, k, device="cuda") * 32).clamp(-448, 448).to(w.dtype))
        s = torch.exp2(torch.rand(e, n // 128, k // 128, device="cuda") * 5 - 13)
        result.extend((w, s))
    return tuple(result)


def decoded(q, s):
    return q.float() * s.repeat_interleave(128, -1)


def decoded_weight(q, s):
    return q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def reference(x, weights, ids, params):
    """FP32 oracle on quantized operands, using the requested FP32 FC1 epilogue.

    The Triton chain is the comparator, including its extra BF16 gate/up round.
    Only the contract's intermediate h is rounded before the second quantizer;
    the final reference retains FP32 weighted routes and their sum.
    """
    w13, s13, w2, s2 = params
    e, h, i = w2.shape
    top_k = ids.shape[1]
    flat_ids = ids.flatten()
    q, scale = quantize_per_token_group_fp8(x)
    a = decoded(q, scale)
    intermediate = torch.zeros((ids.numel(), i), dtype=torch.bfloat16, device=x.device)
    # Cache index lists for the two projections, on the GPU; this is reference-only.
    rows = [(flat_ids == expert).nonzero().flatten() for expert in range(e)]
    for expert, routes in enumerate(rows):
        if routes.numel():
            gu = a[routes // top_k] @ decoded_weight(w13[expert], s13[expert]).T
            gate, up = gu.chunk(2, dim=-1)
            intermediate[routes] = (gate / (1.0 + torch.exp(-gate)) * up).bfloat16()
    qh, sh = quantize_per_token_group_fp8(intermediate)
    ah = decoded(qh, sh)
    down = torch.zeros((ids.numel(), h), dtype=torch.float32, device=x.device)
    for expert, routes in enumerate(rows):
        if routes.numel():
            down[routes] = (
                ah[routes] @ decoded_weight(w2[expert], s2[expert]).T
            ) * weights.flatten()[routes, None]
    total = torch.zeros_like(x, dtype=torch.float32)
    for route in range(top_k):
        total += down.view(x.shape[0], top_k, h)[:, route]
    return total


def rms(value):
    return value.double().square().mean().sqrt().item()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(191)
    params = random_weights(256, 2048, 512)
    cute = BlockFP8MoE(*params, provider="cute-fp8")
    triton = BlockFP8MoE(*params, provider="triton-fp8")

    @lru_cache(maxsize=None)
    def case(m, skewed):
        generator = torch.Generator(device="cuda").manual_seed(712 + 2 * m + int(skewed))
        x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
        x[:, :128] = 0  # floor scale and padded zeros must agree between quantizers.
        ids = torch.randint(256, (m, 8), device="cuda", dtype=torch.int32, generator=generator)
        if skewed:
            ids[:, :4] = torch.randint(
                8, (m, 4), device="cuda", dtype=torch.int32, generator=generator
            )
        ids.view(-1)[::11] = -1
        weights = torch.rand(m, 8, device="cuda", generator=generator)
        weights /= weights.sum(-1, keepdim=True)
        oracle = reference(x, weights, ids, params)
        baseline = triton(x, weights, ids)
        return x, weights, ids, oracle, baseline

    return cute, case


CASES = [
    (name, m, skewed)
    for name, config in PRESETS.items()
    if config.path == "grouped"
    for m in (1, 2, 3, 4, 64, 300)
    if m != 300 or config.bm >= 64
    for skewed in (False, True)
]


@pytest.mark.parametrize("preset,m,skewed", CASES)
@torch.inference_mode()
def test_presets(model, preset, m, skewed):
    op, case = model
    x, weights, source_ids, oracle, baseline = case(m, skewed)
    ids = source_ids.clone()
    op.set_config(preset)
    actual = op(x, weights, ids)
    repeat = op(x, weights, ids)
    assert torch.equal(actual, repeat), (preset, m, "nondeterministic")
    error, floor = rms(actual.float() - oracle), rms(baseline.float() - oracle)
    assert error <= 1.05 * floor + 1e-30, (
        preset,
        m,
        skewed,
        error,
        floor,
        error / max(floor, 1e-30),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(x, weights, ids)
    graph.replay()
    assert torch.equal(actual, captured)
    # The graph must reread padded counts and skip masked routes on every replay.
    ids.fill_(-1)
    graph.replay()
    assert torch.count_nonzero(captured).item() == 0
    ids.copy_(source_ids)
    graph.replay()
    assert torch.equal(actual, captured)


def route_order(h, sorted_ids, padded, ids):
    r = ids.numel()
    active = (
        (torch.arange(sorted_ids.numel(), device=ids.device) < padded[0])
        & (sorted_ids >= 0)
        & (sorted_ids < r)
    )
    active &= ids.flatten()[sorted_ids.long().clamp(0, r - 1)] >= 0
    output = torch.zeros((r, h.shape[1]), dtype=h.dtype, device=h.device)
    output[sorted_ids[active].long()] = h[active]
    return output


@pytest.mark.parametrize("preset", [name for name, c in PRESETS.items() if c.path == "grouped"])
@torch.inference_mode()
def test_fc1_uses_public_quantizer(model, preset, monkeypatch):
    from ctikernels import _moe_block_fp8_cute as kernels
    from ctikernels.linear import block_fp8

    op, case = model
    x, weights, ids, _, _ = case(4, True)
    expected_q, expected_s = quantize_per_token_group_fp8(x)
    fc1, fc2 = kernels._functions(op, PRESETS[preset])
    calls = []

    def quantize(source):
        assert source is x
        calls.append("quantize")
        return quantize_per_token_group_fp8(source)

    def first(q, scales, *args):
        calls.append("fc1")
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(scales, expected_s)
        return fc1(q, scales, *args)

    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quantize)
    monkeypatch.setattr(kernels, "_functions", lambda *_: (first, fc2))
    kernels.forward(op, x, weights, ids, PRESETS[preset])
    assert calls == ["quantize", "fc1"]


@torch.inference_mode()
def test_masked_reduction_never_reads_poisoned_rows():
    from ctikernels._moe_block_fp8_cute import masked_reduce

    from ..modules._moe_reference import combine_reference

    ids = torch.tensor([[0, -1, 1, -1], [-1, -1, -1, -1]], device="cuda", dtype=torch.int32)
    down = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    down[ids.flatten() < 0] = float("nan")
    weights = torch.rand(2, 4, device="cuda")
    out = torch.empty(2, 256, device="cuda", dtype=torch.bfloat16)
    masked_reduce(down, ids, out, weights)
    expected = combine_reference(down.view(2, 4, 256), weights, ids >= 0)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(out[1]).item() == 0


@torch.inference_mode()
def test_small_aligned_shape_and_prepare():
    params = random_weights(3, 256, 128)
    op = BlockFP8MoE(*params, top_k=2, provider="cute-fp8").set_config("swap16")
    op.prepare([3])
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0, -1], [1, 2], [-1, -1]], device="cuda", dtype=torch.int32)
    weights = torch.full((3, 2), 0.5, device="cuda")
    result = op(x, weights, ids)
    expected = reference(x, weights, ids, params)
    baseline = BlockFP8MoE(*params, top_k=2, provider="triton-fp8")(x, weights, ids)
    assert rms(result.float() - expected) <= 1.05 * rms(baseline.float() - expected) + 1e-30
    assert torch.count_nonzero(result[2]).item() == 0
    assert op(x[:0], weights[:0], ids[:0]).shape == (0, 256)


@torch.inference_mode()
def test_cold_capture_is_rejected(model):
    op, case = model
    x, weights, ids, _, _ = case(1, False)
    op.set_config("swap16")
    with pytest.raises(RuntimeError, match="prepare"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(x, weights, ids)
