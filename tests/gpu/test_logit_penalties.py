# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Standalone DN4 CUDA qualification; CPU reference comparisons are chunked."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.sampling import LogitPenalties
from ctikernels.sampling._penalties_reference import reference_penalties

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SHAPES = [(1, 1), (2, 1), (3, 1), (4, 1), (4, 16), (1, 64), (32, 2), (1024, 1), (16, 64)]


def rows_for_case(b, width, device):
    indices = torch.arange(b)
    rows = SimpleNamespace(
        **{
            name: value.to(device)
            for name, value in {
                "rows": (indices % 4).int(),
                "prompt_lengths": torch.tensor([0, min(2048, width), width, min(131072, width)])[
                    indices % 4
                ],
                "lengths": torch.full((b,), width, dtype=torch.int32),
                "presence": torch.tensor([0, -2, 2, 0.5])[indices % 4],
                "frequency": torch.tensor([0, 2, -2, 0.25])[indices % 4],
                "repetition": torch.tensor([1, 1e-30, 1e30, 1.2])[indices % 4],
            }.items()
        }
    )
    if b == 1:
        rows.presence.fill_(0.5)
        rows.frequency.fill_(0.3)
        rows.repetition.fill_(1.2)
    return rows


def assert_reference_in_chunks(source, actual, table, rows, drafts):
    cpu_table = table.cpu()
    cpu_rows = SimpleNamespace(**{key: value.cpu() for key, value in vars(rows).items()})
    cpu_drafts = drafts.cpu()
    # At most one request of Q<=64 per reference allocation, even at R=1024.
    for start in range(source.shape[0]):
        sl = slice(start, start + 1)
        local = SimpleNamespace(**{key: value[sl] for key, value in vars(cpu_rows).items()})
        expected = reference_penalties(source[sl].cpu(), cpu_table, local, cpu_drafts[sl])
        got = actual[sl].cpu().contiguous()
        assert torch.equal(got.view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.fixture(scope="module")
def token_table():
    generator = torch.Generator().manual_seed(1701)
    # Reused across the matrix; each vocabulary has a genuine out-of-range ID.
    table = torch.randint(-3, 248325, (4, 131072), dtype=torch.int32, generator=generator)
    table[:, :8] = torch.tensor([-1, 248320, 0, 1, 2, 2, 151936, 151939])
    return table.cuda()


@pytest.mark.parametrize("b,q", SHAPES)
@pytest.mark.parametrize("v", [151936, 248320, 151939])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bitwise_matrix(b, q, v, dtype, token_table):
    # Large-R cases use a short view; other cases scan all 131072 columns.
    table = token_table[:, :257] if b * q == 1024 else token_table
    rows = rows_for_case(b, table.shape[1], "cuda")
    x = torch.randn((b, q, v + 8), device="cuda", dtype=dtype)[..., :v]
    x[..., :3] = torch.tensor([-torch.inf, -0.0, 0.0], device="cuda", dtype=dtype)
    limit = torch.finfo(dtype).max
    x[..., 3:5] = torch.tensor([limit, -limit], device="cuda", dtype=dtype)
    drafts = torch.randint(-2, v + 2, (b, q - 1), device="cuda", dtype=torch.int64)
    if q == 1:
        x = x[:, 0]
    out = torch.empty((*x.shape[:-1], v + 4), device="cuda")[..., :v]
    op = LogitPenalties(table, v)
    op.apply(x, rows, drafts, out=out)
    assert_reference_in_chunks(x, out, table, rows, drafts)
    if dtype == torch.float32:
        op.apply(x, rows, drafts, out=x)
        # Out-of-place has already been checked independently, including zeros.
        for start in range(b):
            assert torch.equal(
                x[start].contiguous().view(torch.int32), out[start].contiguous().view(torch.int32)
            )


def test_fp16_and_262144_token_context():
    v = 151939
    table = (torch.arange(262144, device="cuda", dtype=torch.int32) % (v + 3))[None, :].repeat(2, 1)
    rows = rows_for_case(2, 262144, "cuda")
    rows.rows[:] = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    rows.prompt_lengths[:] = torch.tensor([131072, 262144], device="cuda")
    rows.presence.fill_(0.5)
    rows.frequency.fill_(0.3)
    rows.repetition.fill_(1.2)
    x = torch.randn((2, 2, v), dtype=torch.float16, device="cuda")
    drafts = torch.tensor([[2], [-1]], device="cuda", dtype=torch.int32)
    out = torch.empty_like(x, dtype=torch.float32)
    LogitPenalties(table, v).apply(x, rows, drafts, out=out)
    assert_reference_in_chunks(x, out, table, rows, drafts)


def test_capture_replays_mutated_engine_state():
    b, q, v = 4, 16, 151939
    table = torch.randint(-2, v + 2, (4, 2049), device="cuda", dtype=torch.int32)
    rows = rows_for_case(b, 2049, "cuda")
    x = torch.randn((b, q, v), device="cuda", dtype=torch.bfloat16)
    drafts = torch.randint(0, v, (b, q - 1), device="cuda", dtype=torch.int32)
    out = torch.empty_like(x, dtype=torch.float32)
    op = LogitPenalties(table, v)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            op.apply(x, rows, drafts, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op.apply(x, rows, drafts, out=out)
    for step in range(2):
        table.random_(-2, v + 2)
        rows.rows.copy_(torch.tensor([3, 2, 1, 0], device="cuda", dtype=torch.int32).roll(step))
        rows.prompt_lengths.copy_(torch.tensor([0, 1, 16, 1024], device="cuda"))
        rows.lengths.copy_(torch.tensor([0, 1024, 16, 2049], device="cuda", dtype=torch.int32))
        rows.presence.copy_(torch.tensor([0, -2, 2, 0.5], device="cuda").roll(step))
        rows.frequency.copy_(torch.tensor([0, 2, -2, 0.3], device="cuda").roll(step))
        rows.repetition.copy_(torch.tensor([1, 1e-30, 1e30, 0.7], device="cuda").roll(step))
        drafts.random_(-2, v + 2)
        x.normal_()
        graph.replay()
        assert_reference_in_chunks(x, out, table, rows, drafts)


def test_rows_are_batch_invariant_and_deterministic():
    b, q, v = 6, 4, 151936
    table = torch.randint(-2, v + 2, (b, 3000), device="cuda", dtype=torch.int32)
    rows = rows_for_case(b, 3000, "cuda")
    x = torch.randn((b, q, v), device="cuda")
    drafts = torch.randint(0, v, (b, q - 1), device="cuda", dtype=torch.int32)
    op = LogitPenalties(table, v)
    out = torch.empty_like(x)
    op.apply(x, rows, drafts, out=out)
    again = torch.empty_like(x)
    op.apply(x, rows, drafts, out=again)
    assert torch.equal(out.view(torch.int32), again.view(torch.int32))
    for n in range(b):
        local = SimpleNamespace(**{key: value[n : n + 1] for key, value in vars(rows).items()})
        alone = torch.empty_like(x[n : n + 1])
        op.apply(x[n : n + 1], local, drafts[n : n + 1], out=alone)
        assert torch.equal(alone.view(torch.int32), out[n : n + 1].view(torch.int32))
