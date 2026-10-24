# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""kvcache.inference.RowCopy on the GPU: one Triton launch, bitwise, graph == eager."""

import pytest
import torch

from ctikernels.kvcache.inference import RowCopy
from ctikernels.kvcache.triton import _kernels

from ..modules._row_copy_case import GDN_KINDS, KINDS, bits, expected, pool
from ..modules.test_kvcache_row_copy import PAIRS


def _same(tensors, want):
    return all(torch.equal(bits(a), bits(b)) for a, b in zip(tensors, want, strict=True))


@pytest.mark.parametrize("kinds", [KINDS, KINDS[:3], GDN_KINDS], ids=["bytes", "words", "gdn"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_one_launch_copies_every_tensor_bit_for_bit(kinds, dtype):
    tensors = pool("cuda", layers=24 if kinds is not GDN_KINDS else 3, kinds=kinds)
    sources, targets = (torch.tensor(v, dtype=dtype, device="cuda") for v in PAIRS)
    want = expected(tensors, sources, targets)
    op = RowCopy(tensors)
    assert op.provider == "triton"
    op.warm_up((dtype,))
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        op(sources, targets)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 1, [e.name for e in kernels]
    assert _same(tensors, want)


def test_word_class_is_per_tensor():
    """An odd-sized tensor copies by bytes without narrowing the others' words."""
    aligned = torch.zeros(4, 8, 128, device="cuda")  # 4 KiB rows: 16-byte vectors
    odd = torch.zeros(4, 3, dtype=torch.uint8, device="cuda")
    words = RowCopy((aligned, odd))._provider.table[:, 4].tolist()
    assert words == [_kernels.VEC16, _kernels.WORD1]


def test_strided_rows_and_row_counts_per_tensor():
    wide = torch.randn(9, 2048, device="cuda")[:, :1000]
    short = torch.randn(5, 3, dtype=torch.float64, device="cuda")
    sources = torch.tensor([0, 7, 2], device="cuda")
    targets = torch.tensor([3, 1, 8], device="cuda")
    want = expected([wide, short], sources, targets)
    RowCopy((wide, short))(sources, targets)
    assert _same((wide, short), want)


def _compiled():
    return sum(len(cache[0]) for cache in _kernels.copy_rows.device_caches.values())


@pytest.mark.parametrize("n", [1, 3])
def test_unbound_index_rows_capture_without_compiling(n):
    """sources/targets as rows of one [2, n] int64 tensor: targets' address is only 8-byte
    aligned for odd n, which must not select (and compile) another kernel."""
    eager, captured = pool("cuda"), pool("cuda")
    op, graph_op = RowCopy(eager), RowCopy(captured)
    graph_op.warm_up()
    index = torch.full((2, n), -1, dtype=torch.int64, device="cuda")
    before = _compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_op(*index.unbind(0))
    pairs = ([0, 3, 5][:n], [6, 1, 4][:n])
    index.copy_(torch.tensor(pairs, device="cuda"))
    graph.replay()
    op(*index.unbind(0))
    torch.cuda.synchronize()
    assert _compiled() == before
    assert _same(captured, eager)


def test_capturing_a_cold_index_dtype_raises():
    op = RowCopy(pool("cuda"))
    op.warm_up((torch.int64,))
    index = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="warm_up"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(index, index)


@pytest.mark.parametrize(
    ("sources", "targets", "match"),
    [([0, 20], [1, 2], "-1 or in"), ([0, 3], [1, 1], "distinct"), ([0, 3], [3, 1], "disjoint")],
)
def test_engine_checks_enforce_the_index_contract(monkeypatch, sources, targets, match):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", "1")
    op = RowCopy(pool("cuda"))
    with pytest.raises(ValueError, match=match):
        op(torch.tensor(sources, device="cuda"), torch.tensor(targets, device="cuda"))


@pytest.mark.exhaustive
def test_rows_past_two_gib():
    """A tensor over 2**31 bytes: 64-bit row and unit offsets."""
    big = torch.zeros(5, 1 << 29, dtype=torch.float32, device="cuda")  # 2 GiB rows, 10 GiB
    big[4, -8:] = torch.arange(8, device="cuda", dtype=torch.float32)
    RowCopy((big,))(torch.tensor([4], device="cuda"), torch.tensor([1], device="cuda"))
    assert torch.equal(big[1, -8:], big[4, -8:])


def test_graph_replay_equals_eager_with_new_indices():
    eager, captured = pool("cuda"), pool("cuda")
    sources = torch.zeros(3, dtype=torch.int64, device="cuda")
    targets = torch.zeros(3, dtype=torch.int64, device="cuda")
    op, graph_op = RowCopy(eager), RowCopy(captured)
    graph_op.warm_up()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_op(sources, targets)
    for pairs in (([0, 1, 2], [3, 4, 5]), ([8, -1, 6], [0, 2, 1]), ([4, -1, -1], [7, 8, -1])):
        sources.copy_(torch.tensor(pairs[0])), targets.copy_(torch.tensor(pairs[1]))
        op(sources, targets)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(captured, eager)


def test_zero_copies_launch_nothing():
    tensors = pool("cuda")
    before = [t.clone() for t in tensors]
    none = torch.empty(0, dtype=torch.int64, device="cuda")
    op = RowCopy(tensors)  # uploads its row table
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        op(none, none)
        torch.cuda.synchronize()
    assert not [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert _same(tensors, before)
