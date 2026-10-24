# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 The Qwen Team and The HuggingFace Inc. team
# SPDX-FileCopyrightText: Copyright 2026 The HuggingFace Inc. team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Transformers 93c8b7b485963a10800c91f55304db6be211c2bd
# (`src/transformers/models/qwen3_5/modeling_qwen3_5.py`, `src/transformers/vision_utils.py`)
# via cti-engine 21b8c1812 (`python/ctiengine/models/qwen3_5/vision.py`); see THIRD_PARTY_NOTICES.

"""Bilinear position embeddings on the device: bitwise against cti-engine's composed Qwen3.5
path run on CUDA, against the CPU torch provider, and graph replay."""

import pytest
import torch

from ctikernels.embedding.inference import bilinear_pos_embed_

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SIDE, C = 48, 1152  # Qwen3.5: num_position_embeddings 2304, hidden 1152, merge 2


def engine_composed(table, height, width, merge_size, dtype):
    """cti-engine 21b8c1812 qwen3_5/vision.py verbatim on the table's device:
    ``_merge_window_coordinates``, ``_bilinear_taps``, the position embedding's forward, then
    ``.to(x.dtype)`` (a CUDA ``/ max(size - 1, 1)`` multiplies by the FP32 reciprocal)."""
    side, device = round(table.shape[0] ** 0.5), table.device
    shape = (height // merge_size, width // merge_size, merge_size, merge_size)
    rows = torch.arange(height, device=device).view(shape[0], 1, merge_size, 1)
    columns = torch.arange(width, device=device).view(1, shape[1], 1, merge_size)
    rows, columns = rows.expand(shape).flatten(), columns.expand(shape).flatten()

    def taps(index, size):
        source = index.to(torch.float32) * (side - 1) / max(size - 1, 1)
        lower = source.floor()
        offsets = torch.arange(2, device=index.device)
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


def table_of(columns=C, dtype=torch.bfloat16, seed=0, side=SIDE):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(side * side, columns, generator=g).to(dtype).cuda()


def embed(table, out, offset, height, width, merge_size=2):
    bilinear_pos_embed_(
        table, out, offset=offset, height=height, width=width, merge_size=merge_size
    )


def bits(x):
    return x.contiguous().view(torch.uint8)


def images(table, grids, dtype):
    """Every grid's rows in one NaN-filled buffer, a gap row before each image (the gaps must
    stay NaN); returns (out, expected)."""
    rows = sum(h * w + 1 for h, w in grids)
    out = torch.full((rows, table.shape[1]), float("nan"), dtype=dtype, device="cuda")
    expected, offset = out.clone(), 0
    for height, width in grids:
        offset += 1
        embed(table, out, offset, height, width)
        expected[offset : offset + height * width] = engine_composed(table, height, width, 2, dtype)
        offset += height * width
    return out, expected


@pytest.mark.parametrize(
    "grid", [(48, 48), (62, 62), (2, 1024), (1024, 2), (64, 64), (32, 48), (16, 16), (256, 256)]
)
def test_bitwise_against_engine_composed(grid):
    table = table_of()
    out = torch.empty(grid[0] * grid[1], C, dtype=torch.bfloat16, device="cuda")
    embed(table, out, 0, *grid)
    assert torch.equal(bits(out), bits(engine_composed(table, *grid, 2, torch.bfloat16)))


def test_every_even_size_at_offsets():
    # Rows and columns of every even size up to 512, many images in one buffer.
    sizes = range(2, 513, 2)
    grids = [(s, 2) for s in sizes] + [(2, s) for s in sizes]
    out, expected = images(table_of(16, seed=1), grids, torch.bfloat16)
    assert torch.equal(bits(out), bits(expected))


def test_qwen_shapes_at_offsets():
    out, expected = images(
        table_of(seed=2), [(48, 48), (62, 4), (2, 1024), (32, 48)], torch.bfloat16
    )
    assert torch.equal(bits(out), bits(expected))


@pytest.mark.parametrize(
    "table_dtype,dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float32), (torch.float16, torch.bfloat16)],
)  # fmt: skip
def test_dtypes(table_dtype, dtype):
    table = table_of(dtype=table_dtype, seed=3)
    out, expected = images(table, [(48, 48), (62, 62), (6, 1000)], dtype)
    assert torch.equal(bits(out), bits(expected))


@pytest.mark.parametrize("grid", [(48, 48), (62, 4), (2, 1024)])
def test_bitwise_against_cpu_provider(grid):
    table = table_of(seed=4)
    out = torch.empty(grid[0] * grid[1], C, dtype=torch.bfloat16, device="cuda")
    embed(table, out, 0, *grid)
    expected = torch.empty(out.shape, dtype=torch.bfloat16)
    embed(table.cpu(), expected, 0, *grid)
    assert torch.equal(bits(out.cpu()), bits(expected))


def test_mismatch_points_take_the_reciprocal_on_device():
    """CUDA torch divides by a host int through its FP32 reciprocal, which the op follows."""
    for size, i, value in ((48, 1, 0.99999994), (62, 61, 46.999996)):
        index = torch.tensor([i], device="cuda")
        source = index.to(torch.float32) * (SIDE - 1) / max(size - 1, 1)
        assert source.item() == torch.tensor(value).item()
        assert source.item() != i * (SIDE - 1) / (size - 1)


def test_graph_replay_follows_the_table():
    grids = [(48, 48), (2, 1024), (62, 4)]
    table = table_of(seed=5)
    out = torch.zeros(sum(h * w for h, w in grids), C, dtype=torch.bfloat16, device="cuda")

    def record():
        offset = 0
        for height, width in grids:
            embed(table, out, offset, height, width)
            offset += height * width

    record()  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        record()
    table.copy_(table_of(seed=6))
    out.zero_()
    graph.replay()
    replayed = out.clone()
    record()
    expected = torch.cat([engine_composed(table, h, w, 2, torch.bfloat16) for h, w in grids])
    torch.cuda.synchronize()
    assert torch.equal(bits(replayed), bits(out))
    assert torch.equal(bits(replayed), bits(expected))
