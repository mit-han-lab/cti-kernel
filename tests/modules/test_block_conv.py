# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DFlash2 block convolution: bitwise against the composed torch form, on the CPU interpreter."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ctikernels.speculative import block_conv


def composed(x, deltas, base, q, dtype=None):
    """The engine's composed form: FP32 coefficients, FP32 products, tap-order FP32 sums,
    rounded to ``dtype`` (default ``x.dtype``)."""
    taps, segments = deltas.shape[1], deltas.shape[2]
    kernel = base.float().view(1, 1, taps, segments, -1)
    kernel = kernel + deltas.float().reshape(-1, q, taps, segments, 1)
    rows = x.float().view(-1, q, segments, kernel.shape[-1])
    out = kernel[:, :, 0] * rows
    for tap in range(1, taps):
        out[:, tap:] += kernel[:, tap:, tap] * rows[:, :-tap]
    return out.view_as(x).to(dtype or x.dtype)


CASES = [
    # (blocks, Q, taps, C, S, x dtype, delta dtype)
    (4, 8, 2, 5120, 320, torch.bfloat16, torch.bfloat16),
    (3, 7, 2, 1024, 64, torch.bfloat16, torch.bfloat16),
    (2, 2, 2, 768, 48, torch.float32, torch.bfloat16),
    (2, 16, 3, 640, 40, torch.bfloat16, torch.float32),
    (5, 1, 2, 256, 16, torch.bfloat16, torch.bfloat16),
    (1, 8, 1, 520, 13, torch.float16, torch.bfloat16),
]


def _case(blocks, q, taps, c, s, dtype, ddtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = blocks * q
    x = (torch.randn(n, c, generator=g) * 3).to(dtype)
    x[0, :8] = -0.0  # signed zeros survive the first-row skip
    if n > 1:
        x[1] = 0.0
    both = (torch.randn(n, 2, taps, s, generator=g) * 0.5).to(ddtype)
    base = (torch.randn(2, taps, c, generator=g) * 0.5).to(torch.bfloat16)
    return x.to(device), both.to(device), base.to(device)


def _truncated_bf16(values):
    # The Triton CPU interpreter casts FP32 to BF16 by truncation (the GPU rounds to nearest);
    # matching its cast checks every operation before it bitwise.
    bits = values.float().view(torch.int32) & -65536
    return bits.view(torch.float32).bfloat16()


def _exercise(device="cpu"):
    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    for index, (blocks, q, taps, c, s, dtype, ddtype) in enumerate(CASES):
        x, both, base = _case(blocks, q, taps, c, s, dtype, ddtype, device, index)
        for side in (0, 1):
            deltas = both[:, side]  # row stride 2 * taps * S, as the engine's projection view
            expected = composed(x, deltas, base[side], q)
            if interpreted and x.dtype == torch.bfloat16:
                expected = _truncated_bf16(composed(x, deltas, base[side], q, torch.float32))
            actual = block_conv(x, deltas, base[side].contiguous(), q)
            assert actual.dtype == x.dtype
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (
                index,
                side,
                (actual.float() - expected.float()).abs().max().item(),
            )
    empty = torch.empty(0, 64, dtype=torch.bfloat16, device=device)
    deltas = torch.empty(0, 2, 4, dtype=torch.bfloat16, device=device)
    base = torch.zeros(2, 64, dtype=torch.bfloat16, device=device)
    assert block_conv(empty, deltas, base, 8).shape == (0, 64)


def test_bitwise_on_cpu_interpreter():
    pytest.importorskip("triton")
    command = f"import importlib; importlib.import_module({__spec__.name!r})._exercise()"
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=os.environ | {"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""},
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "shapes,message",
    [
        (((16, 64), (16, 2, 4), (2, 64), 8), None),
        (((15, 64), (15, 2, 4), (2, 64), 8), "multiple of block_size"),
        (((16, 64), (16, 2, 5), (2, 64), 8), "multiple of S"),
        (((16, 64), (16, 3, 4), (2, 64), 8), "base"),
    ],
)
def test_validation(shapes, message):
    xs, ds, bs, q = shapes
    x, d, b = torch.zeros(xs), torch.zeros(ds), torch.zeros(bs)
    if message is None:
        return
    with pytest.raises(ValueError, match=message):
        block_conv(x, d, b, q)


def test_rejects_scattered_deltas():
    x, b = torch.zeros(8, 64), torch.zeros(2, 64)
    d = torch.zeros(8, 4, 2).transpose(1, 2)  # [8, 2, 4] with stride(2) != 1
    with pytest.raises(ValueError, match="contiguous"):
        block_conv(x, d, b, 8)
