# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DeepSeek-V4 linear wrappers on the CPU: lazy imports and the checks that run before any
launch (``quantize_mxfp8``, ``GroupedBlockFP8Linear``); numerics are in tests/gpu."""

from __future__ import annotations

import sys

import pytest
import torch

from ctikernels.linear.block_fp8 import GroupedBlockFP8Linear
from ctikernels.linear.mxfp8 import dequantize_mxfp8, quantize_mxfp8

from ..support.child import run_child


def test_importing_loads_no_library() -> None:
    code = (
        "import sys, ctikernels.linear.mxfp8, ctikernels.linear.block_fp8; "
        "print(sorted(m for m in ('triton', 'flashinfer') if m in sys.modules))"
    )
    out = run_child([sys.executable, "-c", code], timeout=600, check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize(
    ("x", "error", "match"),
    [
        (torch.zeros(4, 64, dtype=torch.float32), TypeError, "BF16 or FP16"),
        (torch.zeros(4, 48, dtype=torch.bfloat16), ValueError, "multiple of 32"),
        (torch.zeros(64, dtype=torch.bfloat16), ValueError, r"\[M, K\]"),
        (torch.zeros(4, 64, dtype=torch.bfloat16), ValueError, "CUDA"),
    ],
)
def test_quantize_mxfp8_rejects(x, error, match) -> None:
    with pytest.raises(error, match=match):
        quantize_mxfp8(x)


def test_dequantize_mxfp8_decodes_the_exponent() -> None:
    q = torch.tensor([[1.5] * 32 + [-2.0] * 32]).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127 + 3, 127 - 2]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert dequantize_mxfp8(q, scale).tolist() == [[12.0] * 32 + [-0.5] * 32]


@pytest.mark.parametrize(
    ("rows", "cols", "groups", "match"),
    [
        (1024, 512, 3, "positive divisor"),
        (8 * 64, 512, 8, "multiples of 128"),
        (8 * 128, 256, 8, "CUDA"),
    ],
)
def test_grouped_rejects_shapes(rows, cols, groups, match) -> None:
    weight = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn)
    scale = torch.ones(-(-rows // 128), -(-cols // 128))
    with pytest.raises(ValueError, match=match):
        GroupedBlockFP8Linear(weight, scale, groups=groups)


def test_grouped_rejects_trainable_weight() -> None:
    weight = torch.zeros(1024, 256).requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        GroupedBlockFP8Linear(weight, torch.ones(8, 2), groups=8)
