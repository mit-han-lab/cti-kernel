# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""LoRA expand fused into the Triton block-FP8 GEMM epilogue (``BlockFP8Linear(lora=...)``).

The fused result equals the LoRA-free output's FP32 accumulator plus the FP64 delta, rounded
once: within one BF16/FP16 ULP of (base + delta) beyond the base's own rounding. A zero
delta leaves the output bitwise equal to the LoRA-free path; graph replay equals eager.
"""

from __future__ import annotations

import pytest
import torch

from ctikernels.linear.block_fp8 import BlockFP8Linear
from ctikernels.linear.lora import LoRAEpilogue

from .test_block_fp8 import _random_activation, _random_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
# (K, slice widths): one slice; Qwen3.5 qkv-like slices; narrow slices inside one tile; tails.
SHAPES = (
    (2048, (1024,)),
    (2048, (4096, 512, 512)),
    (512, (96, 32, 16, 200)),
    (4096, (1000,)),
)


def _ulp(value):
    return torch.exp2(torch.floor(torch.log2(value.abs().clamp_min(2.0**-126))) - 7)


def _providers():
    return ["triton-fp8"]


def _epilogue(m, sizes, rank, dtype, low_dtype, seed=0):
    torch.manual_seed(seed)
    padded = max(16, 1 << (rank - 1).bit_length())
    low = (torch.randn(m, len(sizes), padded, device="cuda") * 0.5).to(low_dtype)
    factors = [torch.randn(n, rank, device="cuda", dtype=dtype) * 0.05 for n in sizes]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    lora = LoRAEpilogue(
        low,
        torch.tensor([b.data_ptr() for b in factors], device="cuda", dtype=torch.int64),
        torch.tensor(
            [v for pair in zip(offsets, sizes, strict=True) for v in pair],
            device="cuda",
            dtype=torch.int32,
        ),
        rank,
        rank,
    )
    delta = torch.cat(
        [low[:, s, :rank].double() @ b.double().T for s, b in enumerate(factors)], dim=1
    )
    return lora, factors, delta


@pytest.mark.parametrize("provider", _providers())
@pytest.mark.parametrize("low_dtype", ["rounded", torch.float32])
@pytest.mark.parametrize("m", [1, 4, 33, 200])
@pytest.mark.parametrize("k,sizes", SHAPES)
@pytest.mark.parametrize("rank", [16, 19, 64])
def test_fused_delta_matches_base_plus_delta(provider, low_dtype, m, k, sizes, rank):
    dtype = torch.bfloat16
    low_dtype = dtype if low_dtype == "rounded" else low_dtype
    weight, scale = _random_weight(sum(sizes), k)
    op = BlockFP8Linear(weight, scale, provider=provider)
    x = _random_activation(m, k, dtype)
    base = op(x)
    lora, factors, delta = _epilogue(m, sizes, rank, dtype, low_dtype)
    fused = op(x, lora=lora)
    expected = base.double() + delta.to(base.device)
    # fused = round(acc + delta) and base = round(acc) for one FP32 accumulator acc.
    bound = _ulp(expected) + _ulp(base.double()) + 1e-5 * delta.abs().max()
    assert ((fused.double() - expected).abs() <= bound).all()
    # A zero delta adds exact zeros to the same accumulator: bitwise the LoRA-free path.
    lora.low.zero_()
    torch.testing.assert_close(op(x, lora=lora), base, rtol=0, atol=0)


@pytest.mark.parametrize("provider", _providers())
@pytest.mark.parametrize("m", [1, 4, 64])
def test_graph_replay_equals_eager_and_split_k_is_covered(provider, m):
    k, sizes, rank, dtype = 4096, (1024, 256, 256), 64, torch.bfloat16
    weight, scale = _random_weight(sum(sizes), k)
    op = BlockFP8Linear(weight, scale, provider=provider)
    x = _random_activation(m, k, dtype)
    lora, factors, _ = _epilogue(m, sizes, rank, dtype, dtype, seed=m)
    op.warm_up(batch_sizes=[m])
    out = torch.empty(m, sum(sizes), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, out=out, lora=lora)
    torch.cuda.current_stream().wait_stream(stream)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(x, out=out, lora=lora)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    lora.low.mul_(2)  # device data is read on every replay
    graph.replay()
    assert not torch.equal(out, eager)


@pytest.mark.parametrize("m", [1, 512, 2048])
def test_default_dispatch_runs_lora_on_the_triton_gemm(m):
    """Batches the device table gives CuTe, FlashInfer or CUDA (e.g. RTX 5090 (1024, 2048) at
    M > 256, every M on GB300) still take the LoRA epilogue, on the Triton GEMM."""
    k, sizes, dtype = 2048, (1024,), torch.bfloat16
    weight, scale = _random_weight(sum(sizes), k)
    default, triton = (
        BlockFP8Linear(weight, scale),
        BlockFP8Linear(weight, scale, provider="triton-fp8"),
    )
    x = _random_activation(m, k, dtype)
    lora, factors, _ = _epilogue(m, sizes, 16, dtype, dtype)  # factors: B stays alive
    expected = triton(x, lora=lora)
    torch.testing.assert_close(default(x, lora=lora), expected, rtol=0, atol=0)
    workspace = torch.empty(-(-default.workspace_bytes(m) // 4), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(default(x, lora=lora, workspace=workspace), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 4, 64])
@pytest.mark.parametrize("workspace", [False, True])
def test_ue8m0_input_lora_replays_in_a_cuda_graph(m, workspace):
    """DSV4 projections: power-of-two input scales with a LoRA epilogue, captured after one
    eager LoRA call per M (both the allocator and the workspace= entries)."""
    k, sizes, dtype = 2048, (1024,), torch.bfloat16
    weight, scale = _random_weight(sum(sizes), k)
    op = BlockFP8Linear(weight, scale, scale_ue8m0=True)
    x = _random_activation(m, k, dtype)
    lora, factors, _ = _epilogue(m, sizes, 16, dtype, dtype, seed=m)  # B stays alive
    kwargs = {}
    if workspace:
        kwargs["workspace"] = torch.empty(
            -(-op.workspace_bytes(m) // 4), device="cuda", dtype=torch.float32
        )
    op.warm_up(batch_sizes=[m])
    out = torch.empty(m, sum(sizes), device="cuda", dtype=dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(x, out=out, lora=lora, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(x, out=out, lora=lora, **kwargs)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)


def test_split_k_configs_are_exercised():
    weight, scale = _random_weight(1280, 4096)
    op = BlockFP8Linear(weight, scale, provider=_providers()[0])
    assert op._triton.config_for(1)[2] > 1  # split-K epilogue runs in the split sum


def test_unwarmed_variant_refuses_capture():
    k, sizes, dtype = 512, (256,), torch.bfloat16
    weight, scale = _random_weight(sum(sizes), k)
    op = BlockFP8Linear(weight, scale, provider=_providers()[0])
    x = _random_activation(2, k, dtype)
    lora, _, _ = _epilogue(2, sizes, 32, dtype, dtype)
    op.warm_up(batch_sizes=[2])
    with pytest.raises(RuntimeError, match="before capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(x, lora=lora)


@pytest.mark.parametrize("m", [1, 4, 33])
@pytest.mark.parametrize("row_parallel", [False, True])
def test_lora_shrink_composes_with_the_fp8_base(m, row_parallel):
    """lora_shrink + BlockFP8Linear(lora=...) = base + scaling * (x A^T) B^T; row-parallel
    K shards (FP32 partials) summed like the base all-reduce match the same oracle."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("the Triton LoRA shrink runs on cc >= 10.0")
    from ctikernels.linear.lora import lora_shrink

    k, sizes, rank, dtype, tp = 2048, (1024, 256, 256), 32, torch.bfloat16, 2
    torch.manual_seed(m)
    x = _random_activation(m, k, dtype).clamp(-4, 4)
    factors = tuple(
        t
        for n in sizes
        for t in (
            torch.randn(rank, k, device="cuda", dtype=dtype) * 0.02,
            torch.randn(n, rank, device="cuda", dtype=dtype) * 0.02,
        )
    )
    delta = torch.cat(
        [
            0.5 * (x.double() @ a.double().T) @ b.double().T
            for a, b in zip(factors[::2], factors[1::2], strict=True)
        ],
        dim=1,
    )
    weight, scale = _random_weight(sum(sizes), k)
    if not row_parallel:
        op = BlockFP8Linear(weight, scale, provider=_providers()[0])
        base = op(x)
        lora = lora_shrink(x, factors, scaling=0.5, output_sizes=sizes)
        assert lora.low.dtype == dtype
        fused = op(x, lora=lora)
        expected = base.double() + delta
        bound = _ulp(expected) + _ulp(base.double()) + 2e-2 * delta.abs().max()
        assert ((fused.double() - expected).abs() <= bound).all()
        return
    total = torch.zeros(m, sum(sizes), device="cuda", dtype=torch.float64)
    base_total = torch.zeros_like(total)
    bound = torch.zeros_like(total)  # each rank's two roundings, on that rank's partial
    for r in range(tp):
        ks = slice(r * k // tp, (r + 1) * k // tp)
        op = BlockFP8Linear(
            weight[:, ks].contiguous(),
            scale[:, ks.start // 128 : ks.stop // 128].contiguous(),
            provider=_providers()[0],
        )
        xs = x[:, ks].contiguous()
        local = tuple(f[:, ks].contiguous() if i % 2 == 0 else f for i, f in enumerate(factors))
        lora = lora_shrink(xs, local, scaling=0.5, output_sizes=sizes, row_parallel=True)
        assert lora.low.dtype == torch.float32
        base, fused = op(xs).double(), op(xs, lora=lora).double()
        base_total += base
        total += fused
        bound += _ulp(base) + _ulp(fused)
    expected = base_total + delta
    bound += 2e-2 * delta.abs().max()
    assert ((total - expected).abs() <= bound).all()


@pytest.mark.parametrize("m", [1, 4, 33, 200])
@pytest.mark.parametrize("k,sizes", SHAPES[:3])
def test_bf16_mode_carries_the_same_epilogue(m, k, sizes):
    """The 16-bit mode of the same GEMM (no scales) adds the LoRA tile identically."""
    from ctikernels.linear.triton import _block_fp8
    from ctikernels.linear.triton._block_fp8_provider import _config

    dtype, rank = torch.bfloat16, 32
    torch.manual_seed(m)
    weight = (torch.randn(sum(sizes), k, device="cuda") * 0.02).to(dtype)
    x = (torch.randn(m, k, device="cuda") * 0.5).to(dtype)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    config = _config(m, sum(sizes), k, sms, torch.cuda.get_device_capability())
    base = torch.empty(m, sum(sizes), device="cuda", dtype=dtype)
    _block_fp8.gemm(x, None, weight, None, base, config=config)
    lora, factors, delta = _epilogue(m, sizes, rank, dtype, dtype)
    fused = torch.empty_like(base)
    _block_fp8.gemm(x, None, weight, None, fused, config=config, lora=lora)
    expected = base.double() + delta
    bound = _ulp(expected) + _ulp(base.double()) + 1e-5 * delta.abs().max()
    assert ((fused.double() - expected).abs() <= bound).all()
