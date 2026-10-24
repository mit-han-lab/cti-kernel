# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""add_layer_norm_ on CPU: the torch law, the contract, and the Triton kernel on the interpreter.

The residual is bitwise torch's add; the norm is within an ulp of F.layer_norm on that stored
sum, the spacing taken at the larger of the two values and max(|weight * normalized|, |bias|),
where the affine terms can cancel.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ctikernels.norm import inference

from ..support.accuracy import assert_bitwise, assert_within_ulps

EPS = 1e-6


def composed(x, residual, weight, bias, eps=EPS):
    """The torch composition (HF/vLLM/SGLang): a dtype add, then nn.LayerNorm in FP32."""
    summed = residual + x
    hidden = x.shape[-1]
    y = F.layer_norm(summed.float(), (hidden,), weight.float(), bias.float(), eps)
    return y.to(x.dtype), summed


def affine_scale(summed, weight, bias, eps=EPS):
    """max(|weight * normalized|, |bias|) in FP64: the magnitude an output ulp is taken at."""
    values = summed.double()
    centered = values - values.mean(-1, keepdim=True)
    normalized = centered / (centered.square().mean(-1, keepdim=True) + eps).sqrt()
    return torch.maximum((weight.double() * normalized).abs(), bias.double().abs())


def make(shape, dtype, weight_dtype, seed, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    hidden = shape[-1]
    x = (torch.randn(shape, generator=g) * 2).to(dtype)
    residual = (torch.randn(shape, generator=g) * 4 + 1).to(dtype)
    weight = (torch.randn(hidden, generator=g) * 0.5 + 1).to(weight_dtype)
    bias = torch.randn(hidden, generator=g).to(weight_dtype)
    return tuple(t.to(device) for t in (x, residual, weight, bias))


@pytest.mark.parametrize("store_residual", [True, False])
@pytest.mark.parametrize("weight_fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (2, 3, 64), (5, 1152)])
def test_cpu_is_the_torch_composition(shape, dtype, weight_fp32, store_residual):
    x, residual, weight, bias = make(shape, dtype, torch.float32 if weight_fp32 else dtype, 0)
    expected, summed = composed(x, residual, weight, bias)
    before = residual.clone()
    out = inference.add_layer_norm_(x, residual, weight, bias, EPS, store_residual=store_residual)
    assert out[0] is x and out[1] is residual
    assert inference.last_standalone_provider["add_layer_norm_"] == "torch"
    assert_bitwise(x, expected, name="normalized")
    assert_bitwise(residual, summed if store_residual else before, name="residual")


def test_cpu_strided_rows_leave_gaps_untouched():
    x_buffer, residual_buffer, weight, bias = make((6, 96), torch.bfloat16, torch.bfloat16, 1)
    x_before, residual_before = x_buffer.clone(), residual_buffer.clone()
    x, residual = x_buffer[:, 16:80], residual_buffer[::2, :64]
    weight, bias = weight[:64].contiguous(), bias[:64].contiguous()
    x_input = x[::2].clone()
    expected, summed = composed(x[::2], residual, weight, bias)
    inference.add_layer_norm_(x[::2], residual, weight, bias, EPS)
    assert_bitwise(x[::2], expected)
    assert_bitwise(residual, summed)
    assert not torch.equal(x[::2], x_input)
    gaps = torch.ones_like(x_buffer, dtype=torch.bool)
    gaps[::2, 16:80] = False
    assert torch.equal(x_buffer[gaps], x_before[gaps])
    residual_gaps = torch.ones_like(residual_buffer, dtype=torch.bool)
    residual_gaps[::2, :64] = False
    assert torch.equal(residual_buffer[residual_gaps], residual_before[residual_gaps])


def test_empty_returns_without_launch():
    x = torch.empty(0, 1152, dtype=torch.bfloat16)
    residual, weight, bias = torch.empty_like(x), torch.ones(1152), torch.zeros(1152)
    out = inference.add_layer_norm_(x, residual, weight, bias, EPS)
    assert out[0] is x and out[1] is residual
    assert inference.last_standalone_provider["add_layer_norm_"] == "empty"


def _base():
    return make((4, 64), torch.bfloat16, torch.bfloat16, 3)


@pytest.mark.parametrize(
    "case,error,match",
    [
        ("x fp32", TypeError, "FP16 or BF16"),
        ("residual dtype", TypeError, "FP16 or BF16"),
        ("weight fp16", TypeError, "x's dtype or FP32"),
        ("bias fp16", TypeError, "x's dtype or FP32"),
        ("residual shape", ValueError, "match x's shape"),
        ("hidden 0", ValueError, "nonempty hidden"),
        ("scalar", ValueError, "nonempty hidden"),
        ("weight shape", ValueError, r"\[hidden\]"),
        ("bias strided", ValueError, r"\[hidden\]"),
        ("weight device", ValueError, "one device"),
        ("hidden stride", ValueError, "unit hidden stride"),
        ("leading dims", ValueError, "one row stride"),
        ("broadcast rows", ValueError, "rows must not overlap"),
        ("same storage", ValueError, "must not overlap"),
        ("residual overlaps x", ValueError, "must not overlap"),
        ("weight inside x", ValueError, "must not overlap"),
        ("bias inside residual", ValueError, "must not overlap"),
        ("x requires grad", RuntimeError, "requires gradients"),
        ("residual requires grad", RuntimeError, "requires gradients"),
        ("weight requires grad", ValueError, "frozen weight and bias"),
        ("bias requires grad", ValueError, "frozen weight and bias"),
    ],
)
def test_contract_rejects(case, error, match):
    x, residual, weight, bias = _base()
    if case == "x fp32":
        x = x.float()
    elif case == "residual dtype":
        residual = residual.half()
    elif case == "weight fp16":
        weight = weight.half()
    elif case == "bias fp16":
        bias = bias.half()
    elif case == "residual shape":
        residual = residual[:3]
    elif case == "hidden 0":
        x, residual = x[:, :0], residual[:, :0]
    elif case == "scalar":
        x, residual = x[0, 0], residual[0, 0]
    elif case == "weight shape":
        weight = weight[:32]
    elif case == "bias strided":
        bias = torch.zeros(128, dtype=bias.dtype)[::2]
    elif case == "weight device":
        weight = weight.to("meta")
    elif case == "hidden stride":
        x = torch.zeros(64, 4, dtype=x.dtype).t()
    elif case == "leading dims":
        x = torch.zeros(4, 2, 64, dtype=x.dtype).transpose(0, 1)
        residual = torch.zeros(2, 4, 64, dtype=x.dtype)
    elif case == "broadcast rows":
        x = torch.zeros(64, dtype=x.dtype).expand(4, 64)
    elif case == "same storage":
        residual = x
    elif case == "residual overlaps x":
        buffer = torch.zeros(5, 64, dtype=x.dtype)
        x, residual = buffer[:4], buffer[1:]
    elif case == "weight inside x":
        weight = x[1]
    elif case == "bias inside residual":
        bias = residual[2]
    elif case == "x requires grad":
        x = x.float().requires_grad_().bfloat16()
    elif case == "residual requires grad":
        residual.requires_grad_()
    elif case == "weight requires grad":
        weight.requires_grad_()
    elif case == "bias requires grad":
        bias.requires_grad_()
    with pytest.raises(error, match=match):
        inference.add_layer_norm_(x, residual, weight, bias, EPS)


def test_rejection_precedes_any_write():
    x, residual, weight, bias = _base()
    before = x.clone(), residual.clone()
    with pytest.raises(ValueError):
        inference.add_layer_norm_(x, residual, weight, bias[:32], EPS)
    assert torch.equal(x, before[0]) and torch.equal(residual, before[1])


# (rows, hidden, dtype, weight dtype, row stride padding, store_residual, max block)
INTERPRETER_CASES = [
    (1, 1, torch.bfloat16, torch.bfloat16, 0, True, None),
    (3, 7, torch.bfloat16, torch.float32, 0, True, None),
    (4, 100, torch.float16, torch.float16, 0, True, None),
    (5, 64, torch.bfloat16, torch.bfloat16, 24, False, None),
    (3, 1152, torch.bfloat16, torch.bfloat16, 0, True, None),
    (2, 100, torch.bfloat16, torch.float32, 12, True, 32),  # the chunked path
    (3, 33, torch.float16, torch.float32, 0, False, 8),
]


def _exercise():
    """Run every interpreter case through the private launcher (CPU tensors). The interpreter
    flushes BF16 subnormals, so the GPU tests own those."""
    from ctikernels.norm.triton import _layer_norm

    default_block = _layer_norm.MAX_BLOCK
    for index, (rows, hidden, dtype, wdtype, pad, store, block) in enumerate(INTERPRETER_CASES):
        _layer_norm.MAX_BLOCK = block or default_block
        x, residual, weight, bias = make((rows, hidden + pad), dtype, wdtype, 10 + index)
        x, residual = x[:, :hidden], residual[:, pad:]
        weight, bias = weight[:hidden].contiguous(), bias[:hidden].contiguous()
        expected, summed = composed(x, residual, weight, bias)
        before = residual.clone()
        _layer_norm.add_layer_norm_(x, residual, weight, bias, EPS, store)
        assert_bitwise(residual, summed if store else before, name=f"residual case {index}")
        assert_within_ulps(
            x, expected, affine_scale(summed, weight, bias), name=f"normalized case {index}"
        )


def test_kernel_on_cpu_interpreter():
    pytest.importorskip("triton")
    command = f"import importlib; importlib.import_module({__spec__.name!r})._exercise()"
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
