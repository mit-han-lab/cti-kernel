# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""paged_kv_metadata against a torch reference: bitwise CSR, capacity bound, graph replay."""

import pytest
import torch

from ctikernels.attention.inference import paged_kv_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(table, rows, lengths, page_size, granularity):
    indptr, indices, last = [0], [], []
    for row, length in zip(rows.tolist(), lengths.tolist(), strict=True):
        pages = table[row, : -(-length // page_size)].long()
        if granularity == "token":
            entries = (pages[:, None] * page_size + torch.arange(page_size)).flatten()[:length]
            last.append(min(length, 1))
        else:
            entries = pages
            last.append((length - 1) % page_size + 1 if length else 0)
        indices.append(entries)
        indptr.append(indptr[-1] + entries.numel())
    as_int = {"dtype": torch.int32}
    return (
        torch.tensor(indptr, **as_int),
        torch.cat(indices).to(torch.int32),
        torch.tensor(last, **as_int),
    )


class Case:
    def __init__(self, lengths, page_size, granularity, width=600, rows_in_table=9, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.page_size, self.granularity = page_size, granularity
        # A padded row stride checks the table's runtime stride.
        backing = torch.randint(
            0, 2**20, (rows_in_table, width + 7), dtype=torch.int32, generator=gen
        )
        self.table = backing.cuda()[:, :width]
        self.rows = torch.randperm(rows_in_table, generator=gen)[: len(lengths)].int().cuda()
        self.lengths = torch.tensor(lengths, dtype=torch.int32).cuda()
        b = len(lengths)
        self.indptr = torch.full((b + 1,), -7, dtype=torch.int32, device="cuda")
        self.indices = torch.full((width * page_size * b,), -7, dtype=torch.int32, device="cuda")
        self.last = torch.full((b,), -7, dtype=torch.int32, device="cuda")

    def run(self, max_kv_len=None, indices=None):
        paged_kv_metadata(
            self.table,
            self.rows,
            self.lengths,
            page_size=self.page_size,
            granularity=self.granularity,
            max_kv_len=max(self.lengths.tolist()) if max_kv_len is None else max_kv_len,
            kv_indptr=self.indptr,
            kv_indices=self.indices if indices is None else indices,
            kv_last_page_len=self.last,
        )

    def check(self):
        indptr, indices, last = reference(
            self.table.cpu(), self.rows.cpu(), self.lengths.cpu(), self.page_size, self.granularity
        )
        assert torch.equal(self.indptr.cpu(), indptr)
        assert torch.equal(self.indices[: indices.numel()].cpu(), indices)
        assert torch.equal(self.last.cpu(), last)
        assert (self.indices[indices.numel() :] == -7).all()


# Decode B=1-4 (one empty request), a prefill batch, and one request longer than a tile.
LENGTHS = [[8192], [5, 0, 77], [1, 2, 3, 4096], [513, 16, 1, 1000, 64, 9], [130000]]


@pytest.mark.parametrize("granularity", ["page", "token"])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("lengths", LENGTHS, ids=lambda x: f"b{len(x)}-max{max(x)}")
def test_matches_reference(lengths, page_size, granularity):
    width = -(-max(lengths) // page_size) + 3
    case = Case(lengths, page_size, granularity, width=width)
    case.run()
    case.check()


def test_capacity_bounds_writes():
    case = Case([300, 300], 16, "token", width=40)
    short = case.indices[:450]
    case.run(indices=short)
    indptr, indices, _ = reference(
        case.table.cpu(), case.rows.cpu(), case.lengths.cpu(), 16, "token"
    )
    assert torch.equal(case.indptr.cpu(), indptr)
    assert torch.equal(short.cpu(), indices[:450])
    assert (case.indices[450:] == -7).all()


@pytest.mark.parametrize("granularity", ["page", "token"])
def test_graph_replay_follows_new_rows_and_lengths(granularity):
    case = Case([100, 2000, 7, 0], 16, granularity, width=200)
    case.run(max_kv_len=3200)  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run(max_kv_len=3200)
    for seed, lengths in ((1, [3200, 1, 0, 64]), (2, [17, 18, 19, 20])):
        other = Case(lengths, 16, granularity, width=200, seed=seed)
        case.table.copy_(other.table)
        case.rows.copy_(other.rows)
        case.lengths.copy_(other.lengths)
        case.indices.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        case.check()


def test_rejects_bad_metadata():
    case = Case([10], 16, "page")
    with pytest.raises(ValueError):
        paged_kv_metadata(
            case.table,
            case.rows.long(),
            case.lengths,
            page_size=16,
            granularity="page",
            max_kv_len=10,
            kv_indptr=case.indptr,
            kv_indices=case.indices,
            kv_last_page_len=case.last,
        )
    with pytest.raises(ValueError):
        paged_kv_metadata(
            case.table,
            case.rows,
            case.lengths,
            page_size=16,
            granularity="slot",
            max_kv_len=10,
            kv_indptr=case.indptr,
            kv_indices=case.indices,
            kv_last_page_len=case.last,
        )
