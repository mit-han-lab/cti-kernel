# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT layer reuse, poison isolation and RNG/capture safety without bookkeeping."""

import pytest
import torch

from .._dflash_reference import draft_attention_reference
from ._dflash import Fixture
from ._exhaustive import ci_cases

MASKS = [(False, None), (True, 4096), (False, 2048), (True, None)]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def allocation_state():
    stats = torch.cuda.memory_stats()
    return torch.cuda.memory_allocated(), stats.get("allocation.all.allocated", 0)


@pytest.mark.parametrize("q,causal,window", ci_cases([4, 8, 16], MASKS, core=[(16, True, 4096)]))
def test_layer_reuse_poison_rng_and_capture(causal, window, q, draft_provider, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    f = Fixture(4, q, 1, 32, 8, causal, window, c=128, provider=draft_provider)
    second = torch.empty_like(f.out)

    def layers():
        f.run()
        f.op.run(q=f.out, paged_kv_cache=(f.k, f.v), out=second)

    f.stage([32, 64, 17, 1], [0, -1, 2, -1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layers()
    stream.synchronize()
    rng = torch.cuda.get_rng_state().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        before = allocation_state()
        layers()
        assert allocation_state() == before
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    saved_k, saved_v = f.k.clone(), f.v.clone()
    for turn, slots in enumerate(([0, -1, 2, -1], [-1, 1, -1, 3], [-1] * 4, [3, 2, 1, 0])):
        snapshots = []
        for poison in (-(2**30), 2**30):
            f.stage([32, 64, 17, 1], slots, turn)
            ptr, indices = [0], []
            for row, slot in enumerate(slots):
                a, b = f.cpu_token_ptr[row : row + 2].tolist()
                indices.extend([poison] * 7 if slot == -1 else f.cpu_token_idx[a:b].tolist())
                ptr.append(len(indices))
                if slot == -1:
                    f.cpu_positions[row * q : (row + 1) * q] = -(2**63) + row
                    if poison > 0:
                        f.query[row * q : (row + 1) * q].fill_(float("inf"))
            f.cpu_token_ptr = torch.tensor(ptr, dtype=torch.int32)
            f.cpu_token_idx.fill_(poison)
            f.cpu_token_idx[: len(indices)] = torch.tensor(indices, dtype=torch.int32)
            f.token_ptr.copy_(f.cpu_token_ptr)
            f.token_idx.copy_(f.cpu_token_idx)
            f.positions.copy_(f.cpu_positions)
            query_before = f.query.clone()
            torch.cuda.synchronize()
            allocated, rng = allocation_state(), torch.cuda.get_rng_state().clone()
            graph.replay()
            torch.cuda.synchronize()
            assert allocation_state() == allocated
            assert torch.equal(torch.cuda.get_rng_state(), rng)
            f.check_metadata()
            expected = f.oracle()
            expected_second = draft_attention_reference(
                expected.bfloat16(),
                f.k,
                f.v,
                slots=f.cpu_slots,
                positions=f.cpu_positions,
                token_indptr=f.cpu_token_ptr,
                token_indices=f.cpu_token_idx,
                query_length=q,
                causal=causal,
                window=window,
            )
            torch.testing.assert_close(f.out.float(), expected, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(second.float(), expected_second, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(f.query, query_before, atol=0, rtol=0, equal_nan=True)
            snapshots.append((f.out.clone(), second.clone()))
            for row, slot in enumerate(slots):
                if slot == -1:
                    for output in (f.out, second):
                        assert torch.count_nonzero(output[row * q : (row + 1) * q]) == 0
        for before, after in zip(*snapshots, strict=True):
            assert torch.equal(before.view(torch.int16), after.view(torch.int16))
    torch.testing.assert_close(f.k, saved_k, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(f.v, saved_v, atol=0, rtol=0, equal_nan=True)
