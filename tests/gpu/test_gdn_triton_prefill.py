# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Our Triton GDN prefill (``provider == "triton"``) on devices of compute capability
8.9-12.1: every case against the FP64 reference and no worse than unmodified fla-core on the
same inputs, CUDA-graph replay, strided and misaligned inputs, host-sync freedom, 64-bit
addressing and four launches per call."""

from __future__ import annotations

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill
from ctikernels.gdn._contract import Geometry

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import (
    ABSOLUTE_FLOOR,
    GEOMETRIES,
    SLOTS,
    _capture,
    _fla_prefill,
    _inputs,
    _largest_no_worse,
    _no_worse_than_fla,
    _poison,
    _pools,
    _prefill_case,
    _projected,
    _same,
    _weights,
    reference,
)

K = V = 128
# Lengths, slots (-1 stateless) and initial-state flags: single sequences around the 64-token
# chunks, mixed batches with empty-adjacent short ones, and a dominant sequence.
CASES = {
    "one-chunk": ([16], [5], [False]),
    "two-chunks": ([30], [5], [True]),
    "short": ([64], [5], [True]),
    "oep": ([100], [5], [True]),
    "long": ([300], [5], [False]),
    "mixed": ([1, 63, 64, 65, 130, 7, 129], [3, 7, -1, 11, 2, 20, -1], [1, 0, 0, 1, 0, 1, 1]),
    "dominant": ([700, 60, 50], [4, -1, 9], [True, False, True]),
    "average-216": ([300, 200, 150], [6, 13, -1], [False, True, False]),
    "single-520": ([520], [17], [True]),
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="the Triton prefill is validated on compute capability 8.9-12.1",
)


@pytest.fixture(autouse=True)
def _triton(monkeypatch):
    # CuTe off: prefill binds our Triton prefill on every device it is validated on.
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "0")


def _one_seed_no_worse(name: str, ours, fla, exact) -> None:
    """RMS against FP64 no worse than fla-core's; the largest error, which one seed cannot
    judge (see _largest_no_worse), within 1.5x fla-core's. Measured on RTX PRO 6000 over 20
    seeds of the grid's one-chunk and short cases: the largest error was below fla-core's on
    17 and at most 1.40x it (SGLang's above ours on 9 of 10 comparable seeds), while the RMS
    was below fla-core's and SGLang's on all 20."""
    _no_worse_than_fla(name, ours, fla, exact, largest := [])
    ((ours_max, fla_max),) = largest
    assert ours_max <= 1.5 * fla_max + ABSOLUTE_FLOOR, (name, ours_max, fla_max)


def _op(k_heads: int, v_heads: int) -> GatedDeltaNetPrefill:
    op = GatedDeltaNetPrefill(num_k_heads=k_heads, num_v_heads=v_heads, device="cuda")
    assert op.provider == "triton", torch.cuda.get_device_name()
    return op


# CI checks Hv=32 one-chunk ordinary and short heavy-decay precision vs fla-core.
@pytest.mark.parametrize(
    "k_heads,v_heads,case,kind",
    ci_cases(
        GEOMETRIES,
        list(CASES),
        ["ordinary", "slow", "heavy"],
        core=[(16, 32, "one-chunk", "ordinary"), (16, 32, "short", "heavy")],
    ),
)
def test_every_configuration_is_no_worse_than_fla_core_against_fp64(k_heads, v_heads, case, kind):
    """y and the written states no worse than fla-core's against the FP64 reference, the
    history bitwise; fresh sequences' rows never read (NaN there) and no other row written."""
    geometry = Geometry(k_heads, v_heads)
    lengths, slots, initial = CASES[case]
    base_seed = 100 + 7 * list(CASES).index(case) + 1000 * ["ordinary", "slow", "heavy"].index(kind)
    op = _op(k_heads, v_heads)
    largest_errors: dict[str, list[tuple[float, float]]] = {"y": [], "state": []}
    for s in range(5):
        seed = base_seed + 10_000 * s
        weights, projected = _inputs(k_heads, v_heads, lengths, kind, seed)
        cu_seqlens, slot_indices, has_initial_state = _prefill_case(
            lengths, slots, [bool(flag) for flag in initial]
        )
        state, conv_state = _pools(k_heads, v_heads, seed=seed + 2)
        read = {slot for slot, start in zip(slots, initial, strict=True) if slot >= 0 and start}
        written = sorted({slot for slot in slots if slot >= 0})
        case_args = (cu_seqlens, slot_indices, has_initial_state)
        expected_y, expected_state, expected_conv = reference.prefill(
            projected, weights, k_heads, *case_args, state, conv_state
        )
        pools = [_poison(pool, read, "nan", seed + 3) for pool in (state, conv_state)]
        before = [pool.clone() for pool in pools]
        fla_pools = [pool.clone() for pool in (state, conv_state)]
        with torch.inference_mode():
            y = op(projected, *weights.args(), *case_args, *pools)
            fla_y = _fla_prefill(geometry, projected, *weights.args(), *case_args, *fla_pools)
        torch.cuda.synchronize()
        untouched = [row for row in range(SLOTS) if row not in written]
        for pool, old in zip(pools, before, strict=True):
            assert _same(pool[untouched], old[untouched])
        rows, hv = sum(lengths), v_heads
        _no_worse_than_fla(
            f"y seed={seed}",
            y.view(rows, hv, V),
            fla_y.view(rows, hv, V),
            expected_y.view(rows, hv, V),
            largest_errors["y"],
        )
        if written:
            _no_worse_than_fla(
                f"state seed={seed}",
                pools[0][written],
                fla_pools[0][written],
                expected_state[written],
                largest_errors["state"],
            )
            assert _same(pools[1][written], expected_conv[written])
    for name, errors in largest_errors.items():
        if errors:
            _largest_no_worse(name, errors)


@pytest.mark.parametrize("lengths", [[129], [200, 64], [256]])
@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_zero_slot_pools_are_no_worse_than_fla_core_against_fp64(k_heads, v_heads, lengths):
    """Stateless sequences need no pool rows."""
    op = _op(k_heads, v_heads)
    weights, projected = _inputs(k_heads, v_heads, lengths, "ordinary", 180)
    norm_before = weights.norm_weight.clone()
    count = len(lengths)
    case_args = _prefill_case(lengths, [-1] * count, [False] * count)
    state, conv_state = _pools(k_heads, v_heads, seed=182, slots=0)
    expected_y, _, _ = reference.prefill(projected, weights, k_heads, *case_args, state, conv_state)
    arguments = (projected, *weights.args(), *case_args, state, conv_state)
    with torch.inference_mode():
        y = op(*arguments)
        fla_y = _fla_prefill(Geometry(k_heads, v_heads), *arguments)
    torch.cuda.synchronize()
    _one_seed_no_worse("stateless y", y, fla_y, expected_y)
    assert _same(weights.norm_weight, norm_before)
    assert state.shape[0] == conv_state.shape[0] == 0


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_zero_slot_pools_graph_replays_like_eager(k_heads, v_heads):
    op = _op(k_heads, v_heads)
    weights, projected = _inputs(k_heads, v_heads, [256], "ordinary", 190)
    case_args = _prefill_case([256], [-1], [False])
    pools = _pools(k_heads, v_heads, seed=192, slots=0)
    arguments = (projected, *weights.args(), *case_args, *pools)
    with torch.inference_mode():
        op(*arguments)
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(*arguments))
        projected.copy_(_projected(256, k_heads, v_heads, seed=193))
        expected = op(*arguments)
    graph.replay()
    torch.cuda.synchronize()
    assert _same(y, expected)


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_a_graph_follows_new_lengths_slots_and_initial_states_across_configurations(
    k_heads, v_heads
):
    """A graph captured at 1024 tokens in 4 sequences replays every split bitwise like eager
    calls: sequence boundaries are read on the device."""
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=200)
    cu_seqlens, slot_indices, has_initial_state = _prefill_case(
        [256, 256, 256, 256], [1, 2, 3, 4], [False] * 4
    )
    projected = _projected(1024, k_heads, v_heads, seed=201)
    state, conv_state = _pools(k_heads, v_heads, seed=202)
    arguments = (projected, *weights.args(), cu_seqlens, slot_indices, has_initial_state)
    with torch.inference_mode():
        op(*arguments, state, conv_state)
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(*arguments, state, conv_state))
    cases = [
        ([1000, 8, 8, 8], [6, -1, 7, 8], [True, False, True, False]),
        ([1, 1, 1, 1021], [10, 11, -1, 12], [False, True, False, True]),
        ([300, 300, 300, 124], [-1, -1, -1, -1], [True, True, False, False]),
        ([520, 500, 2, 2], [13, 14, 15, 16], [True, True, True, True]),
    ]
    for case, (lengths, slots, initial) in enumerate(cases):
        new_cu, new_slots, new_initial = _prefill_case(lengths, slots, initial)
        new_state, new_conv = _pools(k_heads, v_heads, seed=203 + case)
        new_projected = _projected(1024, k_heads, v_heads, seed=210 + case)
        eager_state, eager_conv = new_state.clone(), new_conv.clone()
        with torch.inference_mode():
            new = (new_projected, *weights.args(), new_cu, new_slots, new_initial)
            expected = op(*new, eager_state, eager_conv)
        statics = (projected, cu_seqlens, slot_indices, has_initial_state, state, conv_state)
        values = (new_projected, new_cu, new_slots, new_initial, new_state, new_conv)
        for static, value in zip(statics, values, strict=True):
            static.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected), lengths
        assert _same(state, eager_state) and _same(conv_state, eager_conv), lengths


@pytest.mark.parametrize("lengths", [[40], [700, 60, 50], [300, 300]])
def test_calls_never_synchronize_with_the_host(lengths):
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=300)
    count = len(lengths)
    cu, slots, initial = _prefill_case(lengths, list(range(1, count + 1)), [True] * count)
    projected = _projected(sum(lengths), k_heads, v_heads, seed=301)
    state, conv_state = _pools(k_heads, v_heads, seed=302)
    arguments = (projected, *weights.args(), cu, slots, initial, state, conv_state)
    with torch.inference_mode():
        op(*arguments)  # compiles
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            op(*arguments)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


@pytest.mark.parametrize("layout", ["aligned-stride", "odd-stride", "misaligned-pools"])
def test_strided_rows_and_misaligned_pools_compute_what_dense_inputs_do(layout):
    """Rows of a wider buffer (a row stride of 16 or 8 elements more) and pools whose storage
    starts 4 bytes past an aligned address, all read in place, give the dense call's results
    bit for bit."""
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=400)
    lengths = [37, 300, 5]
    cu, slots, initial = _prefill_case(lengths, [2, 9, -1], [True, False, False])
    projected = _projected(sum(lengths), k_heads, v_heads, seed=401)
    state, conv_state = _pools(k_heads, v_heads, seed=402)
    dense_state, dense_conv = state.clone(), conv_state.clone()
    with torch.inference_mode():
        expected = op(projected, *weights.args(), cu, slots, initial, dense_state, dense_conv)
    rows = projected
    if layout != "misaligned-pools":
        pad = 16 if layout == "aligned-stride" else 8
        wide = torch.zeros(rows.shape[0], rows.shape[1] + pad, dtype=rows.dtype, device="cuda")
        wide[:, : rows.shape[1]] = rows
        rows = wide[:, : rows.shape[1]]
    else:
        flat = torch.empty(state.numel() + 1, device="cuda")
        state = flat[1:].view(state.shape).copy_(state)
        flat_conv = torch.empty(conv_state.numel() + 2, dtype=conv_state.dtype, device="cuda")
        conv_state = flat_conv[2:].view(conv_state.shape).copy_(conv_state)
        assert state.data_ptr() % 16 and conv_state.data_ptr() % 16
    with torch.inference_mode():
        y = op(rows, *weights.args(), cu, slots, initial, state, conv_state)
    torch.cuda.synchronize()
    assert _same(y, expected)
    assert _same(state, dense_state) and _same(conv_state, dense_conv)


def test_capturing_a_layout_that_never_ran_eagerly_is_refused():
    """The prefill compiles per row stride of ``projected``: after a warm-up on dense rows,
    capturing rows of a wider buffer is refused rather than compiled inside the capture."""
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=500)
    cu, slots, initial = _prefill_case([8, 8], [-1, -1], [False, False])
    state, conv_state = _pools(k_heads, v_heads, seed=501, slots=4)
    projected = _projected(16, k_heads, v_heads, seed=502)
    wide = torch.zeros(16, projected.shape[1] + 64, dtype=torch.bfloat16, device="cuda")
    wide[:, : projected.shape[1]] = projected
    strided = wide[:, : projected.shape[1]]
    with torch.inference_mode():
        op(projected, *weights.args(), cu, slots, initial, state, conv_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match=r"once eagerly"):
            with torch.cuda.graph(graph):
                op(strided, *weights.args(), cu, slots, initial, state, conv_state)
    torch.cuda.synchronize()


def test_warming_misaligned_rows_does_not_warm_their_aligned_stride():
    """The prefill compiles per rows' alignment: capturing aligned rows of a stride warmed only
    misaligned is refused until they run eagerly, then replay matches eager output."""
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=510)
    cu, slots, initial = _prefill_case([8, 8], [-1, -1], [False, False])
    state, conv_state = _pools(k_heads, v_heads, seed=511, slots=4)
    projected = _projected(16, k_heads, v_heads, seed=512)
    width = projected.shape[1]
    wide = torch.empty(16, width + 16, dtype=torch.bfloat16, device="cuda")
    misaligned = wide[:, 1 : width + 1].copy_(projected)
    aligned = torch.empty_like(wide)[:, :width].copy_(projected)
    assert misaligned.data_ptr() % 16 and aligned.data_ptr() % 16 == 0
    assert misaligned.stride(0) == aligned.stride(0) == width + 16
    with torch.inference_mode():
        op(misaligned, *weights.args(), cu, slots, initial, state, conv_state)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match=r"once eagerly"):
            with torch.cuda.graph(graph):
                op(aligned, *weights.args(), cu, slots, initial, state, conv_state)
        expected = op(aligned, *weights.args(), cu, slots, initial, state, conv_state)
        torch.cuda.synchronize()
        graph, y = _capture(
            lambda: op(aligned, *weights.args(), cu, slots, initial, state, conv_state)
        )
        graph.replay()
    torch.cuda.synchronize()
    assert _same(y, expected)


@pytest.mark.parametrize("length", [40, 300])
def test_slots_past_two_to_the_31_state_elements_are_addressed_in_64_bits(length):
    """A pool of 4100 slots at (16, 32) holds more than 2^31 FP32 state elements: a sequence
    in its last slot computes exactly what it does in a one-slot pool (an int32 offset would
    wrap), and no other slot is touched."""
    k_heads, v_heads, slots = 16, 32, 4100
    need = slots * v_heads * V * K * 4 + (4 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=600)
    projected = _projected(length, k_heads, v_heads, seed=601)
    conv_dim = 2 * k_heads * K + v_heads * V
    small_state, small_conv = _pools(k_heads, v_heads, seed=602, slots=1)
    state = torch.zeros(slots, v_heads, V, K, device="cuda")
    conv_state = torch.zeros(slots, 3, conv_dim, dtype=torch.bfloat16, device="cuda")
    state[-1:] = small_state
    conv_state[-1:] = small_conv
    with torch.inference_mode():
        expected = op(projected, *weights.args(), *_prefill_case([length], [0], [True]),
                      small_state, small_conv)  # fmt: skip
        y = op(projected, *weights.args(), *_prefill_case([length], [slots - 1], [True]),
               state, conv_state)  # fmt: skip
    torch.cuda.synchronize()
    assert _same(y, expected)
    assert _same(state[-1:], small_state) and _same(conv_state[-1:], small_conv)
    assert not state[: slots - 1].any() and not conv_state[: slots - 1].any()
    del state, conv_state


def test_rows_past_two_to_the_31_projected_elements_are_addressed_in_64_bits():
    """A batch of 174,000 rows at (16, 32) spans more than 2^31 elements of ``projected``: its
    last sequence, after a stateless one of the rest, is no worse against the FP64 reference
    than fla-core on that sequence alone (a wrapped row offset would read other rows)."""
    k_heads, v_heads, tail = 16, 32, 48
    geometry = Geometry(k_heads, v_heads)
    tokens, width = 174_000, geometry.proj_width
    assert tokens * width > 2**31
    need = tokens * (width + v_heads * V) * 2 + (4 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    op = _op(k_heads, v_heads)
    weights = _weights(k_heads, v_heads, seed=700)
    projected = torch.empty(tokens, width, dtype=torch.bfloat16, device="cuda")
    projected.normal_(generator=torch.Generator("cuda").manual_seed(701))
    projected[-tail:] = _projected(tail, k_heads, v_heads, seed=702)
    state, conv_state = _pools(k_heads, v_heads, seed=703, slots=4)
    alone = _prefill_case([tail], [2], [True])
    expected_y, expected_state, expected_conv = reference.prefill(
        projected[-tail:], weights, k_heads, *alone, state, conv_state
    )
    fla_pools = [state.clone(), conv_state.clone()]
    cu, slots, initial = _prefill_case([tokens - tail, tail], [-1, 2], [False, True])
    with torch.inference_mode():
        fla_y = _fla_prefill(geometry, projected[-tail:], *weights.args(), *alone, *fla_pools)
        y = op(projected, *weights.args(), cu, slots, initial, state, conv_state)
    torch.cuda.synchronize()
    rows = (tail, v_heads, V)
    _one_seed_no_worse("y", y[-tail:].view(rows), fla_y.view(rows), expected_y.view(rows))
    _one_seed_no_worse("state", state[2], fla_pools[0][2], expected_state[2])
    assert _same(conv_state[2], expected_conv[2])


def test_four_launches_per_call():
    from torch.profiler import ProfilerActivity, profile

    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    weights, projected = _inputs(k_heads, v_heads, [300, 0, 64, 700], "ordinary", 800)
    case = _prefill_case([300, 0, 64, 700], [3, 7, -1, 11], [True, False, True, False])
    state, conv_state = _pools(k_heads, v_heads, seed=802)
    rows = torch.tensor([12, -1, 15, 17], dtype=torch.int32, device="cuda")
    for captures in (False, True):
        extra = {"capture_rows": rows} if captures else {}
        with torch.inference_mode():
            op(projected, *weights.args(), *case, state, conv_state, **extra)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as trace:
                op(projected, *weights.args(), *case, state, conv_state, **extra)
                torch.cuda.synchronize()
        launches = sum(e.device_type == torch.autograd.DeviceType.CUDA for e in trace.events())
        assert launches == 4, (captures, launches)


def test_warm_up_compiles_what_a_capture_needs():
    """After warm_up (no eager call) a fresh operation captures calls with and without
    captures, and replays them like eager calls."""
    k_heads, v_heads = 16, 32
    op = _op(k_heads, v_heads)
    assert isinstance(op.warm_up(512), int)
    weights, projected = _inputs(k_heads, v_heads, [300, 100], "ordinary", 900)
    case = _prefill_case([300, 100], [3, 4], [True, False])
    state, conv_state = _pools(k_heads, v_heads, seed=902)
    rows = torch.tensor([12, 14], dtype=torch.int32, device="cuda")
    for captures in (0, 1):
        extra = {"capture_rows": rows} if captures else {}
        pools = [p.clone() for p in (state, conv_state)]
        eager = [p.clone() for p in (state, conv_state)]

        def call(pools=pools, extra=extra):
            return op(projected, *weights.args(), *case, *pools, **extra)

        with torch.inference_mode():
            graph, y = _capture(call)
            expected = op(projected, *weights.args(), *case, *eager, **extra)
            graph.replay()
        torch.cuda.synchronize()
        assert _same(y, expected) and all(map(_same, pools, eager)), captures


@pytest.mark.parametrize(("k_heads", "v_heads"), GEOMETRIES)
def test_batch_invariant_rows_do_not_depend_on_the_batch(k_heads, v_heads, monkeypatch):
    """batch_invariant: one sequence's y rows, final state and history are bitwise the same
    alone and packed with others, across (T, N) buckets the default tier may serve by other
    providers."""
    monkeypatch.delenv("CTI_GDN_PREFILL_CUTE")
    op = GatedDeltaNetPrefill(
        num_k_heads=k_heads, num_v_heads=v_heads, device="cuda", batch_invariant=True
    )
    assert op.provider == "triton"
    weights, rows = _inputs(k_heads, v_heads, [700], "ordinary", 300)
    state, conv_state = _pools(k_heads, v_heads, seed=301)
    results = []
    for before, after in (
        ([], []),
        ([16], []),
        ([2048], [64]),
        ([64, 1024], [5, 3000]),
        ([128] * 3, []),
    ):
        lengths = [*before, 700, *after]
        start = sum(before)
        others = _projected(sum(lengths), k_heads, v_heads, seed=302 + len(lengths))
        projected = torch.cat((others[:start], rows, others[start + 700 :]))
        slots = [4 + n for n in range(len(lengths))]
        slots[len(before)] = 3
        case = _prefill_case(lengths, slots, [True] * len(lengths))
        pools = (state.clone(), conv_state.clone())
        with torch.inference_mode():
            y = op(projected, *weights.args(), *case, *pools)
        results.append((y[start : start + 700], pools[0][3], pools[1][3]))
    torch.cuda.synchronize()
    for result in results[1:]:
        assert all(map(_same, result, results[0]))
