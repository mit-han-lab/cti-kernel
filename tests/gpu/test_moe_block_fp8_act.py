# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Fused activation numerical and current-stream/graph qualification on SM12x."""

import pytest
import torch

from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe_block_fp8 import _dsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1))
    or not _dsl_available(),
    reason="requires SM120/121 and CuTe DSL 4.8.0",
)


def ordered_codes(q):
    codes = q.view(torch.uint8).to(torch.int16)
    # Adjacent finite E4M3 values differ by one, including either signed zero.
    return torch.where(codes >= 128, 128 - codes, codes)


@pytest.mark.parametrize("r", [1, 8, 24, 32, 512, 32768])
@pytest.mark.parametrize("kind", ["random", "extreme"])
@torch.inference_mode()
def test_activation_against_split_and_fp32(r, kind):
    from ctikernels._moe_block_fp8_cute_act import silu_mul_quant_fp8

    torch.manual_seed(128)
    i = 512
    gu = (torch.randn(r, 2 * i, device="cuda") * 3).bfloat16()
    if kind == "extreme":
        # Large positive gates have exactly unit sigmoid in FP32. Include
        # both signs, E4M3 ties, a zero row and very large finite products.
        gu[:, :i] = 32
        values = torch.tensor(
            [0, 1, -1, 448, -448, 432, -432, 416], device="cuda", dtype=torch.bfloat16
        )
        gu[:, i:] = values.repeat(i // values.numel())
        gu[:, i + 128 : i + 256] *= 2**50
        gu[:, i + 256 : i + 384] *= 2**-60  # Exercise the 1e-10 amax floor.
        gu[0] = 0
    h = torch.empty(r, i, device="cuda", dtype=torch.bfloat16)
    silu_and_mul(gu, h)
    expected_q, expected_s = quantize_per_token_group_fp8(h)
    q, s = torch.empty_like(expected_q), torch.empty_like(expected_s)
    silu_mul_quant_fp8(gu, q, s)
    assert torch.equal(s.view(torch.int32), expected_s.view(torch.int32))
    if kind == "extreme":
        assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    else:
        # Approximate exp implementations can straddle a BF16 tie; allow
        # one E4M3 step, but no regression versus the FP32 activation oracle.
        assert (ordered_codes(q) - ordered_codes(expected_q)).abs().max().item() <= 1
    g, u = gu.float().chunk(2, -1)
    reference = (g / (1 + torch.exp(-g))) * u
    actual = q.float() * s.repeat_interleave(128, -1)
    split = expected_q.float() * expected_s.repeat_interleave(128, -1)
    fused_error = (actual - reference).double().norm()
    split_error = (split - reference).double().norm()
    assert fused_error <= split_error + 1e-7 * reference.double().norm().clamp_min(1)


@pytest.mark.parametrize("i", [128, 512, 1536])
@pytest.mark.parametrize("offset", [0, 1])
@torch.inference_mode()
def test_current_stream_graph_and_unaligned_views(i, offset):
    from ctikernels._moe_block_fp8_cute_act import silu_mul_quant_fp8

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gu = torch.empty(8 * 2 * i + offset, device="cuda", dtype=torch.bfloat16)[offset:]
        gu = gu.view(8, 2 * i).normal_()
        q = torch.empty(8 * i + offset, device="cuda", dtype=torch.float8_e4m3fn)[offset:]
        q = q.view(8, i)
        s = torch.empty(8, i // 128, device="cuda")
        silu_mul_quant_fp8(gu, q, s)
        eager_q, eager_s = q.clone(), s.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            silu_mul_quant_fp8(gu, q, s)
        graph.replay()
        replay_q, replay_s = q.clone(), s.clone()
        gu.zero_()
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(replay_q.view(torch.uint8), eager_q.view(torch.uint8))
    assert torch.equal(replay_s, eager_s)
    assert torch.count_nonzero(q.float()).item() == 0
    assert (s > 0).all()


@torch.inference_mode()
def test_first_use_during_capture_rejected():
    from ctikernels import _moe_block_fp8_cute_act as act

    gu = torch.zeros(1, 768, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(1, 384, device="cuda", dtype=torch.float8_e4m3fn)
    s = torch.empty(1, 3, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    key = (384, f"sm_{major}{minor}a")
    cached = act._CACHE.pop(key, None)
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            with pytest.raises(RuntimeError, match="prepare outside capture"):
                act.silu_mul_quant_fp8(gu, q, s)
    finally:
        if cached is not None:
            act._CACHE[key] = cached
