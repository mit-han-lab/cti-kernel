# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DeepSeek-V4 linear ops on the GPU (compute capability 8.9+): the MXFP8 activation quantizer
against its stated recipe (and, on 10.x, our Triton kernel bitwise against FlashInfer's), and
``GroupedBlockFP8Linear`` (``wo_a``) per provider against an FP64 oracle on the same quantized
operands, bitwise run to run and under CUDA graph replay."""

from __future__ import annotations

import pytest
import torch

from ctikernels.linear.block_fp8 import GroupedBlockFP8Linear, quantize_per_token_group_fp8
from ctikernels.linear.mxfp8 import dequantize_mxfp8, quantize_mxfp8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="needs compute capability 8.9+",
    ),
]
pytest.importorskip("triton")
GROUPED = ["triton-fp8", "flashinfer-cutlass"]


def _on_10x():
    return torch.cuda.get_device_capability()[0] == 10


def _grouped(provider):
    if provider == "flashinfer-cutlass" and not _on_10x():
        pytest.skip("FlashInfer's grouped GEMM runs on 10.x")
    return provider


def _x(rows, cols, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device="cuda")
    # Spread magnitudes over many binades so every scale exponent is exercised.
    return (
        x
        * torch.exp2(
            torch.randint(-20, 20, (rows, cols // 32, 1), generator=g, device="cuda").float()
        )
        .repeat_interleave(32, -1)
        .view(rows, cols)
    ).to(dtype)


@pytest.mark.parametrize("rows", [1, 4, 37, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_matches_recipe(rows, dtype):
    x = _x(rows, 4096, dtype=dtype)
    if dtype == torch.float16:
        x = x.clamp(-6e4, 6e4)
    q, scale = quantize_mxfp8(x)
    assert q.shape == x.shape and q.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 128) and scale.dtype == torch.float8_e8m0fnu
    groups = x.float().view(rows, 128, 32)
    amax = groups.abs().amax(-1)
    # Power of two rounded up from amax / 448 (the approximate reciprocal can land one
    # binade low only when amax / 448 is within an ULP of a power of two).
    want = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127) + 127
    got = scale.view(torch.uint8).float()
    assert ((got - want).abs() <= 1).all() and (got == want).float().mean() > 0.999
    back = dequantize_mxfp8(q, scale)
    err = (back - x.float()).abs().view(rows, 128, 32)
    # E4M3 round-to-nearest: half an ULP (2^-4 relative at the top binade) of the group scale.
    bound = 2.0 ** (got - 127).unsqueeze(-1) * 16
    assert (err <= bound).all()


@pytest.mark.parametrize("rows", [1, 4, 130, 1000])
@pytest.mark.parametrize("swizzled", [False, True])
def test_mxfp8_triton_is_bitwise_flashinfer(rows, swizzled):
    if not _on_10x():
        pytest.skip("FlashInfer's quantizer runs on 10.x")
    from ctikernels.linear.flashinfer import _mxfp8 as library
    from ctikernels.linear.triton import _mxfp8 as ours

    x = _x(rows, 4096, seed=rows)
    x[0, :64] = 0  # a zero group
    x[-1, 32:64] = 2.0**-130  # FTZ products
    q, scale = library.quantize(x, swizzled=swizzled)
    q2, scale2 = torch.empty_like(q), torch.zeros_like(scale)
    ours.quantize(x, q2, scale2.view(torch.uint8), swizzled=swizzled)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8))
    if swizzled:  # FlashInfer leaves the padding of its fresh buffer unwritten
        mask = torch.zeros_like(scale, dtype=torch.bool)
        valid = (
            torch.arange(rows, device="cuda")[:, None],
            torch.arange(128, device="cuda")[None, :],
        )
        off = ((valid[0] // 128) * 32 * 512 + (valid[1] // 4) * 512 + (valid[0] % 32) * 16
               + (valid[0] % 128 // 32) * 4 + valid[1] % 4)  # fmt: skip
        mask[off.flatten()] = True
        assert torch.equal(scale.view(torch.uint8)[mask], scale2.view(torch.uint8)[mask])
    else:
        assert torch.equal(scale.view(torch.uint8), scale2.view(torch.uint8))


def test_mxfp8_zero_and_swizzled():
    x = _x(130, 256)
    x[3] = 0
    q, scale = quantize_mxfp8(x)
    assert (q[3].float() == 0).all() and (scale[3].view(torch.uint8) == 0).all()
    q2, swizzled = quantize_mxfp8(x, swizzled=True)
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
    assert swizzled.dtype == torch.uint8 and swizzled.numel() == 256 * 8  # 2 x 128-row tiles


def _grouped_weight(groups=8, n=1024, k=4096, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(groups * n, k, generator=g, device="cuda") * 64).to(torch.float8_e4m3fn)
    exps = torch.randint(-14, -8, (groups * n // 128, k // 128), generator=g, device="cuda")
    scale = torch.exp2(exps.float()).to(torch.float8_e8m0fnu)
    return weight, scale


def _oracle(x, weight, scale, groups):
    t = x.shape[0]
    k = weight.shape[1]
    q, s = quantize_per_token_group_fp8(x.view(t * groups, k))
    a = (q.double() * s.double().repeat_interleave(128, 1)).view(t, groups, k)
    w = weight.double() * scale.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return torch.einsum("tgk,gnk->tgn", a, w.view(groups, -1, k)).flatten(1)


@pytest.mark.parametrize("provider", GROUPED)
@pytest.mark.parametrize("tokens", [1, 3, 4, 17, 24, 512])
def test_grouped_matches_oracle(tokens, provider):
    provider = _grouped(provider)
    weight, scale = _grouped_weight()
    op = GroupedBlockFP8Linear(weight, scale, groups=8, provider=provider)
    x = _x(tokens, 8 * 4096, seed=tokens)
    y = op(x)
    want = _oracle(x, weight, scale, 8)
    assert y.shape == (tokens, 8 * 1024) and y.dtype == torch.bfloat16
    # One BF16 rounding of an FP32-accumulated sum.
    assert ((y.double() - want).abs() <= want.abs() * 2.0**-8 + want.abs().max() * 2.0**-16).all()
    assert torch.equal(op(x.view(tokens, 8, 4096)), y)  # bitwise run to run


@pytest.mark.parametrize("provider", GROUPED)
def test_grouped_graph_replay_is_bitwise_eager(provider):
    provider = _grouped(provider)
    weight, scale = _grouped_weight()
    op = GroupedBlockFP8Linear(weight, scale, groups=8, provider=provider)
    op.warm_up(4)
    x = _x(3, 8 * 4096)
    eager = op(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = op(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    unwarmed = _x(9, 8 * 4096)
    with pytest.raises(RuntimeError, match="warm_up"), torch.cuda.graph(torch.cuda.CUDAGraph()):
        op(unwarmed)


def _grouped_reference(op, x):
    """The routing as built before the shared table (per-T ids, ones and alignment), through
    the same grouped kernel: the output must not change bit for bit."""
    from ctikernels.linear.block_fp8 import _GROUPED_CONFIG
    from ctikernels.moe_impl import fused_moe_kernel_triton, moe_align_block_size

    t, g, k, n = x.shape[0], op.groups, op.in_per_group, op.out_per_group
    ids = torch.arange(g, dtype=torch.int32, device=x.device).repeat(t).view(t * g, 1)
    weights = torch.ones((t * g, 1), dtype=torch.float32, device=x.device)
    aligned = moe_align_block_size(ids, _GROUPED_CONFIG["BLOCK_SIZE_M"], g)
    out = torch.empty((t, g * n), dtype=torch.bfloat16, device=x.device)
    q, scale = quantize_per_token_group_fp8(x.view(t * g, k))
    fused_moe_kernel_triton(
        q, op._weight, out.view(t * g, 1, n), weights, ids, *aligned, False, 1,
        _GROUPED_CONFIG, torch.bfloat16, a_scale=scale, b_scale=op._scale,
        block_fp8_provider="triton-fp8",
    )  # fmt: skip
    return out


@pytest.mark.parametrize("tokens", [1, 4, 64, 8192])
def test_grouped_shared_routing_is_bitwise_the_per_t_routing(tokens):
    weight, scale = _grouped_weight()
    op = GroupedBlockFP8Linear(weight, scale, groups=8, provider="triton-fp8")
    x = _x(tokens, 8 * 4096, seed=tokens)
    assert torch.equal(op(x), _grouped_reference(op, x))


def test_grouped_warm_up_memory_is_linear_in_max_t():
    """Two instances warmed to 8192 tokens keep O(max_t * groups) routing bytes between them
    (the per-T tables took O(max_t^2): about 3 GiB per instance here)."""
    weight, scale = _grouped_weight()
    ops = [GroupedBlockFP8Linear(weight, scale, groups=8, provider="triton-fp8") for _ in "ab"]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for op in ops:
        op.warm_up(8192)
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - before
    # ids (int32) + weights (FP32) of 8192 * 8 rows, doubled growth and retired tables included.
    assert grown <= 4 * 8192 * 8 * 8 + (1 << 20), grown
