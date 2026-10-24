# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Bilinear position embeddings: the torch provider and the Triton kernel (CPU interpreter)
bitwise against cti-engine's composed Qwen3.5 path in its CUDA rounding, and a scalar oracle."""

import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ctikernels.embedding.inference import bilinear_pos_embed_

# (height, width, merge_size): reciprocal-vs-division points (48: i = 1, 62: i = 61), extreme
# aspect ratios, typical grids, merge 3, and merge 1 with a one-patch axis (max(n - 1, 1)).
GRIDS = [(48, 48, 2), (62, 4, 2), (4, 62, 2), (2, 1024, 2), (1024, 2, 2), (64, 64, 2),
         (32, 48, 2), (6, 9, 3), (1, 5, 1), (7, 1, 1)]  # fmt: skip
DTYPES = [(torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32),
          (torch.bfloat16, torch.float32)]  # fmt: skip


def engine_reference(table, height, width, merge_size, dtype, *, divide=False):
    """cti-engine 21b8c1812 qwen3_5/vision.py (``_merge_window_coordinates``, ``_bilinear_taps``,
    ``Qwen3_5VisionPositionEmbedding.forward``, then ``.to(x.dtype)``), with CUDA torch's
    ``/ host int`` written as its multiply by the FP32 reciprocal; ``divide`` keeps the true
    division CPU torch performs instead."""
    side = int(table.shape[0] ** 0.5)
    shape = (height // merge_size, width // merge_size, merge_size, merge_size)
    rows = torch.arange(height).view(shape[0], 1, merge_size, 1).expand(shape).flatten()
    columns = torch.arange(width).view(1, shape[1], 1, merge_size).expand(shape).flatten()

    def taps(index, size):
        source = index.to(torch.float32) * (side - 1)
        if divide:
            source = source / max(size - 1, 1)
        else:
            source = source * torch.tensor(1.0 / max(size - 1, 1), dtype=torch.float32)
        lower = source.floor()
        offsets = torch.arange(2)
        cells = (lower.long().unsqueeze(-1) + offsets).clamp_(max=side - 1)
        weights = (1 - (source - lower).unsqueeze(-1).sub(offsets).abs()).clamp_(min=0)
        return cells, weights

    row_taps, row_weights = taps(rows, height)
    column_taps, column_weights = taps(columns, width)
    cells = (row_taps.unsqueeze(-1) * side + column_taps.unsqueeze(-2)).flatten(1)
    weights = (row_weights.unsqueeze(-1) * column_weights.unsqueeze(-2)).flatten(1)
    embeddings = table[cells[:, 0]] * weights[:, :1]
    for tap in range(1, cells.shape[1]):
        embeddings += table[cells[:, tap]] * weights[:, tap : tap + 1]
    return embeddings.to(dtype)


def f32(x: float) -> float:
    # A double +, -, * of FP32 values rounded to FP32 is the IEEE FP32 result (53 >= 2 * 24 + 2).
    return struct.unpack("f", struct.pack("f", x))[0]


def scalar_oracle(table, height, width, merge_size, dtype):
    """The docstring's law element by element in Python floats rounded to FP32 after each op."""
    side, values = int(table.shape[0] ** 0.5), table.float().tolist()
    m, out = merge_size, []

    def axis(i, n):
        s = f32(f32(i * (side - 1)) * f32(1.0 / max(n - 1, 1)))
        lo = float(int(s))
        fraction = f32(s - lo)
        weights = [max(f32(1.0 - abs(f32(fraction - k))), 0.0) for k in (0, 1)]
        return [min(int(lo) + k, side - 1) for k in (0, 1)], weights

    for p in range(height * width):
        window, inner = divmod(p, m * m)
        (r_taps, r_w), (c_taps, c_w) = (
            axis((window // (width // m)) * m + inner // m, height),
            axis((window % (width // m)) * m + inner % m, width),
        )
        taps = [
            (r_taps[a] * side + c_taps[b], f32(r_w[a] * c_w[b])) for a in (0, 1) for b in (0, 1)
        ]
        row = [f32(v * taps[0][1]) for v in values[taps[0][0]]]
        for cell, weight in taps[1:]:
            row = [f32(acc + f32(v * weight)) for acc, v in zip(row, values[cell], strict=True)]
        out.append(row)
    return torch.tensor(out, dtype=torch.float32).to(dtype)


def table_of(side, columns, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(side * side, columns, generator=g).to(dtype)


def run(table, grids, dtype, launch):
    """Every grid's rows in one buffer, a gap row between images; returns (out, expected)."""
    rows = sum(h * w + 1 for h, w, _ in grids)
    out = torch.full((rows, table.shape[1]), float("nan"), dtype=dtype)
    expected = out.clone()
    offset = 0
    for height, width, merge_size in grids:
        launch(table, out, offset, height, width, merge_size)
        end = offset + height * width
        expected[offset:end] = engine_reference(table, height, width, merge_size, dtype)
        offset = end + 1
    return out, expected


def public(table, out, offset, height, width, merge_size):
    bilinear_pos_embed_(
        table, out, offset=offset, height=height, width=width, merge_size=merge_size
    )


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("table_dtype,dtype", DTYPES)
def test_torch_provider_matches_engine_reference(table_dtype, dtype):
    out, expected = run(table_of(48, 40, table_dtype), GRIDS, dtype, public)
    assert bits_equal(out, expected)


@pytest.mark.parametrize("grid", [(48, 4, 2), (62, 2, 2), (2, 62, 2), (6, 9, 3), (1, 5, 1)])
def test_engine_reference_matches_scalar_oracle(grid):
    table = table_of(48, 8, torch.bfloat16, seed=1)
    expected = scalar_oracle(table, *grid, torch.bfloat16)
    assert bits_equal(engine_reference(table, *grid, torch.bfloat16), expected)


@pytest.mark.parametrize("height,row", [(48, 1), (62, 61)])
def test_reciprocal_points_follow_cuda_not_true_division(height, row):
    # fl(i * 47) * fp32(1 / (n - 1)) is 0.99999994 at (48, 1) and 46.999996 at (62, 61); true
    # division gives 1 and 47, so the taps differ and so does the row.
    table = table_of(48, 16, torch.float32, seed=2)
    out = torch.empty(height * 2, 16)
    public(table, out, 0, height, 2, 2)
    patches = [2 * (row // 2) * 2 + row % 2 * 2 + c for c in range(2)]  # (row, 0), (row, 1)
    divided = engine_reference(table, height, 2, 2, torch.float32, divide=True)
    assert bits_equal(out, engine_reference(table, height, 2, 2, torch.float32))
    assert not bits_equal(out[patches], divided[patches])


def _exercise():
    from ctikernels.embedding.inference import _check, _reciprocal
    from ctikernels.embedding.triton._bilinear import launch

    def interpreted(table, out, offset, height, width, merge_size):
        side = _check(table, out, offset, height, width, merge_size)
        launch(table, out, offset, height, width, side, merge_size, _reciprocal(height),
               _reciprocal(width))  # fmt: skip

    # FP32 outputs only: the interpreter's FP32 -> BF16 cast truncates (Triton 3.8 cast_impl),
    # where the device rounds to nearest even; tests/gpu checks BF16 outputs.
    for table_dtype in (torch.bfloat16, torch.float32):
        out, expected = run(table_of(48, 40, table_dtype), GRIDS, torch.float32, interpreted)
        assert bits_equal(out, expected), table_dtype
    # Two column tiles (C = 160 > 128) and a strided out.
    table = table_of(48, 160, torch.bfloat16, seed=3)
    out = torch.full((96, 200), float("nan"))[:, :160]
    interpreted(table, out, 0, 48, 2, 2)
    assert bits_equal(out, engine_reference(table, 48, 2, 2, torch.float32))


def test_triton_kernel_on_cpu_interpreter():
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
    "change,error,message",
    [
        (lambda a: a.update(table=torch.zeros(10, 8)), ValueError, "square"),
        (lambda a: a.update(table=torch.zeros(16, 4)), ValueError, r"\[N, C\]"),
        (lambda a: a.update(out=torch.zeros(32, 8, dtype=torch.float64)), TypeError, "FP16"),
        (lambda a: a.update(out=torch.zeros(8, 32).t()), ValueError, "unit column"),
        (lambda a: a.update(out=torch.zeros(70).as_strided((32, 8), (2, 1))), ValueError, "overlap"),
        (lambda a: a.update(height=3), ValueError, "multiples"),
        (lambda a: a.update(merge_size=0), ValueError, "multiples"),
        (lambda a: a.update(offset=20), ValueError, "lie in out"),
        (lambda a: a.update(offset=-1), ValueError, "lie in out"),
        (lambda a: a.update(table=torch.zeros(16, 8, requires_grad=True)), RuntimeError, "autograd"),
    ],
)  # fmt: skip
def test_validation(change, error, message):
    args = {"table": torch.zeros(16, 8), "out": torch.zeros(32, 8), "offset": 0, "height": 4,
            "width": 4, "merge_size": 2}  # fmt: skip
    change(args)
    with pytest.raises(error, match=message):
        bilinear_pos_embed_(args.pop("table"), args.pop("out"), **args)
