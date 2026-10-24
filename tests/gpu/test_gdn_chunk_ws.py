# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GP-CP1 correctness and graph replay; no timing runs."""

import pytest
import torch

from ctikernels.gdn import chunk_ws
from ctikernels.gdn.chunk_ws._launch import ChunkPrefillSm120

from ..modules._gdn_chunk_ws_reference import (
    fi_state,
    flashinfer_kernel,
    inputs,
    reference,
    rms_rel,
)
from ._exhaustive import ci_cases


def _reason():
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        return "requires sm120/sm121"
    if not chunk_ws.available():
        return "requires nvidia-cutlass-dsl 4.8.0"
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        return "CuTe DSL or its CUDA libraries are unavailable"
    return None


_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
LENGTHS = ((16,), (64,), (65,), (1, 63, 64, 65, 130, 7, 129), (1024,))


@pytest.fixture(scope="module")
def operators():
    return {}


def _op(operators, hk, hv):
    if (hk, hv) not in operators:
        operators[hk, hv] = ChunkPrefillSm120("cuda", hk, hv)
    return operators[hk, hv]


def _same_bits(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize(
    "hk,hv",
    ci_cases(((16, 32), (16, 48), (8, 16), (8, 24)), core={(16, 32), (16, 48)}),
)
@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("slow", [False, True])
def test_chunk_ws_reference(operators, hk, hv, lengths, slow):
    if hv < hk or hv % hk:
        pytest.skip("GVA geometry only")
    q, k, v, g, beta, base, cu = inputs(hk, hv, lengths, slow=slow)
    n = len(lengths)
    # Single-sequence rows exercise read and ignored initial state across slow/fast.
    slots = [(-1 if i % 3 == 1 else i + 1) for i in range(n)]
    initial = [(i % 3 == 0) if not slow else (i % 3 == 2) for i in range(n)]
    for slot in range(base.shape[0]):
        if not any(s == slot and read for s, read in zip(slots, initial, strict=True)):
            base[slot].fill_(float("nan"))
    pool = base.clone()
    op = _op(operators, hk, hv)
    # A guarded allocation also catches writes past the final packed sequence.
    storage = torch.full((v.shape[0] + 64, hv, 128), 123.0, device=v.device, dtype=v.dtype)
    out = storage[: v.shape[0]]
    slots_t = torch.tensor(slots, dtype=torch.int32, device=q.device)
    initial_t = torch.tensor(initial, dtype=torch.bool, device=q.device)
    scale = 128**-0.5
    op(q, k, v, g, beta, out, pool, slots_t, initial_t, cu, scale)
    assert (storage[v.shape[0] :] == 123.0).all()
    expected_o, expected_s = reference(q, k, v, g, beta, base, lengths, slots, initial, scale)
    written = [(i, slot) for i, slot in enumerate(slots) if slot >= 0]
    seqs, rows = zip(*written, strict=True)
    errors = (rms_rel(out, expected_o), rms_rel(pool[list(rows)], expected_s[list(seqs)]))
    fi = flashinfer_kernel()
    if fi is not None:
        init_fi, indices = fi_state(base, slots, initial)
        state_fi = init_fi.clone()
        out_fi = torch.empty_like(v)
        fi(out_fi, state_fi, q, k, v, init_fi, g.exp(), beta, cu, scale, state_indices=indices)
        fi_errors = (
            rms_rel(out_fi, expected_o),
            rms_rel(state_fi[list(rows)], expected_s[list(seqs)]),
        )
        for error, fi_error in zip(errors, fi_errors, strict=True):
            assert error <= 1.02 * fi_error + 1e-6, (errors, fi_errors)
    else:
        assert max(errors) <= 1e-2, errors
    untouched = [i for i in range(pool.shape[0]) if i not in rows]
    _same_bits(pool[untouched], base[untouched])


def test_chunk_ws_unowned_sequence_writes_no_state(operators):
    q, k, v, g, beta, pool, cu = inputs(16, 32, [65])
    pool.fill_(float("nan"))
    base = pool.clone()
    slots = torch.tensor([-1], device=q.device, dtype=torch.int32)
    initial = torch.tensor([True], device=q.device)
    out = torch.empty_like(v)
    _op(operators, 16, 32)(q, k, v, g, beta, out, pool, slots, initial, cu, 0.5)
    expected, _ = reference(q, k, v, g, beta, base, [65], [-1], [True], 0.5)
    assert rms_rel(out, expected) <= 1e-2
    _same_bits(pool, base)


@pytest.mark.parametrize("hk,hv", [(16, 48), (16, 32)])
def test_chunk_ws_graph_replays_changed_metadata(operators, hk, hv):
    lengths = [1, 63, 65]
    q, k, v, g, beta, base, cu = inputs(hk, hv, lengths, slow=True)
    base[0].fill_(float("nan"))
    slots = torch.tensor([1, -1, 3], dtype=torch.int32, device=q.device)
    initial = torch.tensor([True, False, False], device=q.device)
    op = _op(operators, hk, hv)
    pool = base.clone()
    out = torch.empty_like(v)
    scale = 128**-0.5

    def call():
        op(q, k, v, g, beta, out, pool, slots, initial, cu, scale)

    call()
    compiled = op.compiled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for new_lengths, new_slots, new_initial in (
        ([64, 1, 64], [4, 2, -1], [False, True, True]),
        ([7, 65, 57], [-1, 3, 0], [True, True, False]),
    ):
        cu.copy_(torch.tensor([0, *new_lengths], device=q.device, dtype=torch.int32).cumsum(0))
        slots.copy_(torch.tensor(new_slots, device=q.device, dtype=torch.int32))
        initial.copy_(torch.tensor(new_initial, device=q.device))
        pool.copy_(base)
        graph.replay()
        graph_out, graph_pool = out.clone(), pool.clone()
        pool.copy_(base)
        call()
        _same_bits(out, graph_out)
        _same_bits(pool, graph_pool)
    assert op.compiled is compiled


def test_chunk_ws_symbolic_sizes_reuse_compile(operators):
    op = _op(operators, 16, 32)
    compiled = None
    for lengths in ([16], [64, 65], [1, 63, 64, 65, 130, 7, 129]):
        q, k, v, g, beta, pool, cu = inputs(16, 32, lengths, slow=True)
        base = pool.clone()
        n = len(lengths)
        slots = torch.arange(n, dtype=torch.int32, device=q.device)
        initial = torch.ones(n, dtype=torch.bool, device=q.device)
        out = torch.empty_like(v)
        op(q, k, v, g, beta, out, pool, slots, initial, cu, 0.25)
        expected, state = reference(
            q, k, v, g, beta, base, lengths, list(range(n)), [True] * n, 0.25
        )
        assert rms_rel(out, expected) <= 1e-2
        assert rms_rel(pool[:n], state) <= 1e-2
        if compiled is not None:
            assert op.compiled is compiled
        compiled = op.compiled
