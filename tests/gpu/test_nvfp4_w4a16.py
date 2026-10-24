# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""NVFP4 W4A16 linear (training LM head): forward and input gradient against an FP64
oracle on the dequantized weight (FP32-accumulation bound), exact dequantization through
one-hot rows over every E4M3 scale code, run-to-run equality, zero-scale padding rows,
CUDA-graph replay in a GraphWorkspace and the contract checks."""

from __future__ import annotations

import pytest
import torch

from ctikernels import GraphWorkspace
from ctikernels.linear.nvfp4 import NVFP4Weight, dequantize_nvfp4
from ctikernels.linear.training import NVFP4W4A16Linear
from ctikernels.linear.triton import _nvfp4_w4a16

SHAPES = [(256, 128), (512, 1024), (4096, 2048), (1024, 5120), (96, 160), (1000, 176)]  # (N, K)
ROWS = [1, 2, 3, 4, 8, 9, 16, 33, 65, 256, 300]
ALPHA = 0.37  # forward: input scale x weight global; input_grad: weight global


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _weight(n: int, k: int, seed: int = 0) -> NVFP4Weight:
    g = torch.Generator(device=_device()).manual_seed(seed)
    data = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=_device(), generator=g)
    # Every non-negative finite E4M3 code: zero, subnormals and 448.
    codes = torch.randint(0, 127, (n, k // 16), dtype=torch.uint8, device=_device(), generator=g)
    return NVFP4Weight(data, codes.view(torch.float8_e4m3fn), torch.ones(1, device=_device()))


def _dense(w: NVFP4Weight) -> torch.Tensor:
    return dequantize_nvfp4(w.data, w.scale).double()


def _alpha() -> torch.Tensor:
    return torch.tensor([ALPHA], device=_device())


def _assert_bound(result, a, b, rel: float, input_rel: float = 0.0) -> None:
    """|result - alpha a.b| <= rel |exact| + (depth * 2^-24 + input_rel) * alpha |a|.|b|:
    FP32 sums, plus ``input_rel`` for an operand rounded on the way in."""
    exact = (a.double() @ b) * ALPHA
    slack = a.shape[1] * 2**-24 + input_rel
    bound = exact.abs() * rel + slack * ALPHA * (a.double().abs() @ b.abs())
    error = (result.double() - exact).abs()
    assert bool((error <= bound).all()), float((error / bound.clamp_min(1e-300)).max())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n, k", SHAPES)
def test_forward_matches_fp64_and_repeats(n: int, k: int, dtype: torch.dtype) -> None:
    w = _weight(n, k)
    op = NVFP4W4A16Linear(w)
    dense = _dense(w)
    scale = 1.0 if dtype == torch.bfloat16 else 2**-6  # keep FP16 outputs finite
    for m in ROWS:
        x = (torch.randn(m, k, device=_device()) * scale).to(dtype)
        y = op(x, _alpha())
        assert y.dtype == dtype and y.shape == (m, n)
        _assert_bound(y, x, dense.T, 2**-8)
        assert torch.equal(y, op(x, _alpha()))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("n, k", SHAPES)
def test_dequant_cublas_matches_fp64(n: int, k: int, dtype: torch.dtype) -> None:
    w = _weight(n, k)
    op = NVFP4W4A16Linear(w)
    scale = 1.0 if dtype == torch.bfloat16 else 2**-6
    small = NVFP4W4A16Linear(w, chunk_bytes=1 << 20)  # many chunks
    for m in (1, 300, 1030):
        x = (torch.randn(m, k, device=_device()) * scale).to(dtype)
        y = torch.empty(m, n, dtype=dtype, device=_device())
        small._dequant_forward(x, _alpha(), y)
        _assert_bound(y, x, _dense(w).T, 2**-8)
        y = torch.empty(m, n, dtype=dtype, device=_device())
        op._dequant_forward(x, _alpha(), y)
        _assert_bound(y, x, _dense(w).T, 2**-8)
        dy = torch.randn(m, n, device=_device())
        dx = torch.empty(m, k, device=_device())
        small._dequant_input_grad(dy, _alpha(), dx)  # FP32 accumulation across chunks
        _assert_bound(dx, dy, _dense(w), 2**-23, 2**-8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_dequantizes_every_scale_exactly(dtype: torch.dtype) -> None:
    n, k = 1024, 2048
    w = _weight(n, k, seed=1)
    rows = torch.randperm(k, device=_device())[:64]
    x = torch.zeros(64, k, dtype=dtype, device=_device())
    x[torch.arange(64), rows] = 1
    y = NVFP4W4A16Linear(w)(x, torch.ones(1, device=_device()))
    expected = _dense(w)[:, rows].T.float().to(dtype)
    assert torch.equal(y, expected)


# Relative error of FP32 dy entering as 1, 2 or 3 BF16 terms.
DY_TERMS = {1: 2**-8, 2: 2**-16, 3: 0.0}


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "dy_dtype, terms",
    [(torch.float32, 1), (torch.float32, 2), (torch.float32, 3), (torch.bfloat16, 1)],
)
@pytest.mark.parametrize("n, k", SHAPES)
def test_input_grad_matches_fp64(n, k, dy_dtype, terms, out_dtype) -> None:
    w = _weight(n, k)
    op = NVFP4W4A16Linear(w)
    dense = _dense(w)
    input_rel = DY_TERMS[terms] if dy_dtype == torch.float32 else 0.0
    for m in (1, 4, 256, 513):
        dy = torch.randn(m, n, device=_device()).to(dy_dtype)
        dx = op.input_grad(dy, _alpha(), out_dtype=out_dtype, dy_terms=terms)
        assert dx.dtype == out_dtype and dx.shape == (m, k)
        rel = 2**-23 if out_dtype == torch.float32 else 2**-8
        _assert_bound(dx, dy, dense, rel, input_rel)
        assert torch.equal(dx, op.input_grad(dy, _alpha(), out_dtype=out_dtype, dy_terms=terms))


@pytest.mark.parametrize("dy_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n, k", SHAPES)
def test_marlin_input_grad_copy_matches_fp64(n, k, dy_dtype) -> None:
    w = _weight(n, k)
    op = NVFP4W4A16Linear(w, input_grad_copy=True)
    assert op.prepared_bytes > NVFP4W4A16Linear(w).prepared_bytes
    dense = _dense(w)
    input_rel = 2**-8 if dy_dtype == torch.float32 else 0.0
    for m in (1, 4, 9, 256, 513):
        dy = torch.randn(m, n, device=_device()).to(dy_dtype)
        dx = op.input_grad(dy, _alpha(), out_dtype=torch.bfloat16)
        _assert_bound(dx, dy, dense, 2**-8, input_rel)
        assert torch.equal(dx, op.input_grad(dy, _alpha(), out_dtype=torch.bfloat16))


def test_input_grad_staged_dy_equals_in_register_rounding() -> None:
    # Small FP32 dy is rounded to BF16 once into scratch; the result equals rounding it in
    # every program (the kernel run directly on the FP32 dy with the same tile).
    n, k, m = 4096, 2048, 64
    w = _weight(n, k, seed=4)
    op = NVFP4W4A16Linear(w)
    dy = torch.randn(m, n, device=_device())
    staged = op.input_grad(dy, _alpha())
    direct = torch.empty_like(staged)
    config = _nvfp4_w4a16.config_for(m, k, n, op._sms)
    partials = torch.empty(_nvfp4_w4a16.partial_count(op._sms), device=_device())
    _nvfp4_w4a16.input_grad(dy, w.data, w.scale, _alpha(), direct, partials, 1, config)
    assert torch.equal(staged, direct)


def test_input_grad_three_terms_keep_fp32_dy() -> None:
    # One term per row: three terms give the FP32 product (dy * w) * alpha to within the
    # tensor cores' accumulation rounding (2 ulp); a BF16-rounded dy would be ~2^-8 off.
    n, k = 4096, 2048
    w = _weight(n, k, seed=2)
    cols = torch.randint(0, n, (32,), device=_device())
    values = torch.randn(32, device=_device())
    dy = torch.zeros(32, n, device=_device())
    dy[torch.arange(32), cols] = values
    dx = NVFP4W4A16Linear(w).input_grad(dy, _alpha(), dy_terms=3)
    expected = (values[:, None] * _dense(w)[cols].float()) * _alpha()
    assert bool(((dx - expected).abs() <= expected.abs() * 2**-22).all())


def test_zero_scale_padding_rows_contribute_nothing() -> None:
    # A vocabulary shard padded past the vocabulary: padding rows with zero scales (any
    # codes) give zero logits, and their dy columns do not reach dx.
    n, k, vocab = 1024, 512, 1000
    w = _weight(n, k, seed=3)
    w.scale.view(torch.uint8)[vocab:] = 0
    op = NVFP4W4A16Linear(w)
    x = torch.randn(5, k, device=_device()).to(torch.bfloat16)
    assert not op(x, _alpha())[:, vocab:].any()
    dy = torch.randn(7, n, device=_device())
    clean = dy.clone()
    clean[:, vocab:] = 0
    assert torch.equal(op.input_grad(dy, _alpha()), op.input_grad(clean, _alpha()))


def test_zero_rows_launch_nothing() -> None:
    op = NVFP4W4A16Linear(_weight(256, 128))
    x = torch.empty(0, 128, dtype=torch.bfloat16, device=_device())
    assert op(x, _alpha()).shape == (0, 256)
    dy = torch.empty(0, 256, device=_device())
    assert op.input_grad(dy, _alpha()).shape == (0, 128)


def test_graph_replay_in_a_workspace_matches_eager() -> None:
    op = NVFP4W4A16Linear(_weight(1024, 512))
    alpha = _alpha()
    x = torch.randn(5, 512, device=_device()).to(torch.bfloat16)
    dy = torch.randn(5, 1024, device=_device())
    workspace = GraphWorkspace(max_rows=8)  # must outlive the graph
    with workspace:
        y0, _ = op(x, alpha), op.input_grad(dy, alpha)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y1, dx1 = op(x, alpha), op.input_grad(dy, alpha)
    x.copy_(torch.randn_like(x))
    dy.copy_(torch.randn_like(dy))
    alpha.fill_(1.5)
    graph.replay()
    assert torch.equal(y1, op(x, alpha)) and torch.equal(dx1, op.input_grad(dy, alpha))
    assert not torch.equal(y0, y1)


def test_graph_replay_of_every_provider_with_two_objects_in_one_workspace() -> None:
    # Large-M dX (dequant + cuBLAS), the input-grad Marlin copy and two objects of different
    # sizes and chunk budgets sharing the workspace slots.
    small = NVFP4W4A16Linear(_weight(1024, 512), input_grad_copy=True)
    large = NVFP4W4A16Linear(_weight(2048, 1024, seed=1), chunk_bytes=8 << 20)
    alpha = _alpha()
    dy_small = torch.randn(600, 1024, device=_device())
    dy_large = torch.randn(600, 2048, device=_device())
    dy_copy = torch.randn(5, 1024, device=_device()).to(torch.bfloat16)

    def run():
        return (
            small.input_grad(dy_small, alpha),
            large.input_grad(dy_large, alpha),
            small.input_grad(dy_copy, alpha, out_dtype=torch.bfloat16),
            large(torch.ones(600, 1024, device=_device(), dtype=torch.bfloat16), alpha),
        )

    workspace = GraphWorkspace(max_rows=1024)  # must outlive the graph
    with workspace:
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
    for t in (dy_small, dy_large, dy_copy):
        t.copy_(torch.randn_like(t))
    graph.replay()
    for got, want in zip(captured, run(), strict=True):
        assert torch.equal(got, want)


def test_contract_errors() -> None:
    w = _weight(256, 128)
    data = torch.zeros(64, 20, dtype=torch.uint8, device=_device())
    scale = torch.ones(64, 2, device=_device()).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="multiple of 16"):
        NVFP4W4A16Linear(NVFP4Weight(data, scale, w.global_scale))
    op = NVFP4W4A16Linear(w)
    with pytest.raises(TypeError):
        op(torch.randn(2, 128, device=_device()), _alpha())
    with pytest.raises(ValueError):
        op(torch.randn(2, 64, device=_device()).to(torch.bfloat16), _alpha())
    with pytest.raises(TypeError):
        op.input_grad(torch.randn(2, 256, device=_device()), _alpha(), out_dtype=torch.int32)
    with pytest.raises(ValueError, match="dy_terms"):
        op.input_grad(torch.randn(2, 256, device=_device()).bfloat16(), _alpha(), dy_terms=3)
    x = torch.randn(2, 128, device=_device(), dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(RuntimeError, match="autograd"):
        op(x, _alpha())
    for budget in (0, True, 1.5):
        with pytest.raises(ValueError, match="chunk_bytes"):
            NVFP4W4A16Linear(w, chunk_bytes=budget)
    tiny = NVFP4W4A16Linear(w, chunk_bytes=1024)  # less than one weight row at 600 rows
    with pytest.raises(ValueError, match="chunk_bytes holds no weight row"):
        tiny.input_grad(torch.randn(600, 256, device=_device()), _alpha())
