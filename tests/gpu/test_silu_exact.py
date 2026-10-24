# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The Triton SiLU/sigmoid helper (``ctikernels.triton.silu``) against torch, bit for bit, on every
finite BF16 and FP16 value, the subnormal-output gate band included; its divide is non-FTZ."""

import pytest
import torch
import triton
import triton.language as tl

from ctikernels.triton.silu import sigmoid, silu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@triton.jit
def _apply(x_ptr, y_ptr, n, SIGMOID: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(x_ptr + i, mask=i < n, other=0.0)
    y = sigmoid(x) if SIGMOID else silu(x)
    tl.store(y_ptr + i, y, mask=i < n)


def gates():
    """Every finite BF16 and FP16 value, plus a dense FP32 sweep of the subnormal-SiLU band."""
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = [codes.view(dtype).float() for dtype in (torch.bfloat16, torch.float16)]
    values.append(torch.linspace(-110.0, -80.0, 1 << 16))
    x = torch.cat(values)
    return x[torch.isfinite(x)].cuda()


@pytest.mark.parametrize("fn", ["silu", "sigmoid"])
def test_bitwise_torch_with_subnormals(fn):
    x = gates()
    y = torch.empty_like(x)
    compiled = _apply[(triton.cdiv(x.numel(), 1024),)](
        x, y, x.numel(), SIGMOID=fn == "sigmoid", B=1024
    )
    want = torch.sigmoid(x) if fn == "sigmoid" else torch.nn.functional.silu(x)
    assert torch.equal(y.view(torch.int32), want.view(torch.int32))
    if fn == "silu":
        subnormal = (want != 0) & (want.abs() < torch.finfo(torch.float32).tiny)
        assert subnormal.sum() > 0  # the band that an FTZ divide flushes is covered
    ptx = compiled.asm["ptx"]
    assert "div.rn.f32" in ptx and "div.rn.ftz" not in ptx
