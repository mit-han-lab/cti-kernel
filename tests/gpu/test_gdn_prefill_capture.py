# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""State captures at p = floor(L / 64) * 64 (``capture_rows``) on our Triton prefill and on
the provider the device binds (CuTe v1/v2 or the port); fla-core rejects them."""

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _fla, inference

from .test_gdn import _pools, _prefill_case, _projected, _same, _weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
GRID = GatedDeltaNetPrefill.CAPTURE_GRID


@pytest.fixture(params=["triton", "auto"])
def provider(request):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("captures need the Triton prefill or CuTe (compute capability >= 8.9)")
    return request.param


def _op(hk, hv, provider, monkeypatch):
    if provider == "triton":
        from ctikernels.gdn.triton.prefill import Prefill

        monkeypatch.setattr(
            inference, "_prefill_kernel", lambda geometry, device, **_: Prefill(device, geometry)
        )
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.supports_state_capture
    return op


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize(("hk", "hv"), [(2, 4), (16, 32), (16, 48)])
def test_capture_is_the_state_at_the_floor_point(hk, hv, initial, provider, monkeypatch):
    """y and slots bitwise unchanged; the capture is the state after p tokens (bitwise on
    Triton, whose truncated call computes the same chunks; within 2e-3 of the state's
    magnitude elsewhere), the slot itself at p = L, and history rows p-3..p-1 bitwise."""
    op = _op(hk, hv, provider, monkeypatch)
    lengths, slots = [257, 64, 63, 1, 0, 130, 193], [0, 1, 2, 3, 4, -1, 5]
    rows = [7, 8, 9, 10, 11, 12, -1]
    meta = _prefill_case(lengths, slots, [initial] * len(slots))
    weights = _weights(hk, hv, seed=13700)
    x = _projected(sum(lengths), hk, hv, seed=13701)
    base = _pools(hk, hv, seed=13702, slots=14)
    without, exported = [[p.clone() for p in base] for _ in range(2)]
    capture_rows = torch.tensor(rows, dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        expected_y = op(x, *weights.args(), *meta, *without)
        actual_y = op(x, *weights.args(), *meta, *exported, capture_rows=capture_rows)
    assert _same(actual_y, expected_y)
    written = set()
    begin = 0
    for length, slot, row in zip(lengths, slots, rows, strict=True):
        p = length // GRID * GRID
        if slot >= 0:
            written.add(slot)
            assert all(_same(a[slot], b[slot]) for a, b in zip(exported, without, strict=True))
        if row >= 0 and p >= GRID:
            written.add(row)
            state, history = exported[0][row], exported[1][row]
            assert _same(history, x[begin + p - 3 : begin + p, : history.shape[-1]])
            if p == length and slot >= 0:
                assert _same(state, exported[0][slot])
            scratch = [
                pool[slot : slot + 1].clone() if slot >= 0 else torch.zeros_like(pool[:1])
                for pool in base
            ]
            with torch.inference_mode():
                op(x[begin : begin + p], *weights.args(), *_prefill_case([p], [0], [initial and slot >= 0]), *scratch)  # fmt: skip
            assert _same(history, scratch[1][0])
            if provider == "triton":
                assert _same(state, scratch[0][0])
            else:
                bound = 2e-3 * scratch[0][0].abs().max().item() + 1e-6
                assert (state - scratch[0][0]).abs().max().item() <= bound
        begin += length
    for row in range(base[0].shape[0]):
        if row not in written:
            assert all(_same(a[row], b[row]) for a, b in zip(exported, base, strict=True))


def test_graph_replay_follows_new_rows_and_lengths(provider, monkeypatch):
    op = _op(2, 4, provider, monkeypatch)
    weights = _weights(2, 4, seed=14100)
    x = _projected(200, 2, 4, seed=14101)
    base = _pools(2, 4, seed=14102, slots=8)
    pools = [p.clone() for p in base]
    cu, slots, initial = _prefill_case([129, 71], [0, -1], [True, True])
    rows = torch.tensor([3, 4], dtype=torch.int32, device="cuda")

    def run(target):
        return op(x, *weights.args(), cu, slots, initial, *target, capture_rows=rows)

    with torch.inference_mode():
        op.warm_up(200, max_sequences=2)
        run(pools)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run(pools)
        for ends, targets in (
            ([0, 128, 200], [5, 6]),
            ([0, 64, 200], [-1, 2]),
            ([0, 3, 200], [1, 7]),
        ):
            cu.copy_(torch.tensor(ends, dtype=torch.int32, device="cuda"))
            rows.copy_(torch.tensor(targets, dtype=torch.int32, device="cuda"))
            expected_pools = [p.clone() for p in base]
            expected = run(expected_pools)
            for dst, src in zip(pools, base, strict=True):
                dst.copy_(src)
            graph.replay()
            assert _same(actual, expected)
            assert all(_same(a, b) for a, b in zip(pools, expected_pools, strict=True))


def test_fla_core_rejects_captures(monkeypatch):
    fla = _fla._import("GatedDeltaNetPrefill capture rejection", torch.device("cuda"))
    monkeypatch.setattr(inference, "_prefill_kernel", lambda geometry, device, **_: fla.prefill)
    op = GatedDeltaNetPrefill(num_k_heads=2, num_v_heads=4, device="cuda")
    assert not op.supports_state_capture
    weights = _weights(2, 4, seed=14200)
    with pytest.raises(NotImplementedError), torch.inference_mode():
        op(
            _projected(65, 2, 4, seed=14201),
            *weights.args(),
            *_prefill_case([65], [0], [False]),
            *_pools(2, 4, seed=14202, slots=2),
            capture_rows=torch.ones(1, dtype=torch.int32, device="cuda"),
        )
