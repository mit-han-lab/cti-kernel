# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Default CuTe product gate; non-default sweeps opt in with CTI_GDN_CUTE_TEST_KNOBS=1."""

from __future__ import annotations

import os
import time

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _cute_prefill, _triton_prefill
from ctikernels.gdn._contract import Geometry

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import (
    ABSOLUTE_FLOOR,
    SLOTS,
    TP2_GEOMETRIES,
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
    reference,
)
from ._gdn_prefill_cases import (
    GEOMETRIES as TP1_GEOMETRIES,
)


def _reason():
    if not torch.cuda.is_available():
        return "CuTe prefill requires a CUDA GPU"
    if not _cute_prefill.available():
        return "CuTe prefill requires nvidia-cutlass-dsl[cu13]==4.8.0"
    if _triton_prefill.device_type(torch.device("cuda")) is None:
        return "CuTe prefill serves the accepted sm120/sm121 devices"
    if _cute_prefill.load(torch.device("cuda")) is None:
        return "nvidia-cutlass-dsl 4.8.0 or its CUDA libraries cannot import"
    return None


_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
GEOMETRIES = TP1_GEOMETRIES + TP2_GEOMETRIES
# Each default execution path, chunk edges, mixed state and a long recurrence.
LENGTHS = [[16], [64], [1, 2, 61], [128], [64, 64], [512], [128, 128, 128, 128], [1024]]
KNOBS = os.environ.get("CTI_GDN_CUTE_TEST_KNOBS") == "1"
knobs_only = pytest.mark.skipif(
    not KNOBS, reason="non-default CuTe knob sweep; set CTI_GDN_CUTE_TEST_KNOBS=1"
)
_REFERENCE_CACHE = {}


@pytest.fixture(scope="module", autouse=True)
def _clear_reference_cache():
    yield
    _REFERENCE_CACHE.clear()


def _precision_reference(hk, hv, lengths, kind, seed_index, x, weights, meta, base, written):
    # Keep only compared rows, on CPU: sweeps must not retain whole FP64 GPU pools.
    key = (hk, hv, tuple(lengths), kind, seed_index)
    if key not in _REFERENCE_CACHE:
        y, state, history = reference.prefill(x, weights, hk, *meta, *base)
        _REFERENCE_CACHE[key] = (y.cpu(), state[written].cpu(), history[written].cpu())
    return tuple(t.to(x.device) for t in _REFERENCE_CACHE[key])


def _schedules(values):
    # CI keeps the default schedule; qualification retains the existing knob opt-in.
    return [
        pytest.param(*case.values, marks=(*case.marks, knobs_only)) if case.values[0] else case
        for case in ci_cases(["", *values], core=[("",)])
    ]


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch):
    # This suite qualifies v1 schedules; the port has its own product suite.
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "0")
    monkeypatch.delenv("CTI_GDN_PREFILL_CUTE", raising=False)
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    # Every bucket v1: the provider table's Triton rows are qualified by their own suites.
    monkeypatch.setattr(_triton_prefill.PROVIDER, "table", lambda device_class: {})


def _op(hk, hv, lengths):
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute"
    assert op.supports_state_capture
    assert op.selected_variant(sum(lengths), len(lengths)) == "cute"
    return op


def _precision_inputs(hk, hv, lengths, kind, seed_index):
    """Shared by the product gate and cute_probe; keep seeds/poison/metadata identical."""
    seed = 4200 + 10000 * seed_index
    count = len(lengths)
    # Sequences past the first four own slots 10-23 in turn, then run stateless (slot -1).
    slots = (
        [7]
        if count == 1
        else [[9, 2, -1, 5][i] if i < 4 else 6 + i if i < 18 else -1 for i in range(count)]
    )
    flags = [bool(seed_index % 2)] if count == 1 else [i % 4 != 1 for i in range(count)]
    meta = _prefill_case(lengths, slots, flags)
    weights, x = _inputs(hk, hv, lengths, kind, seed)
    base = _pools(hk, hv, seed=seed + 2)
    read = {s for s, flag in zip(slots, flags, strict=True) if s >= 0 and flag}
    pools = [_poison(p, read, "nan", seed + 3) for p in base]
    return weights, x, meta, base, pools, sorted(s for s in slots if s >= 0)


# CI checks serial prefill with an aligned Hv=32 output buffer.
@pytest.mark.parametrize(
    "offset,tokens,hk,hv",
    ci_cases([0, 1], [65, 2048], GEOMETRIES, core=[(0, 65, 16, 32)]),
)
def test_preallocated_output_matches_allocating_call(hk, hv, tokens, offset):
    op = _op(hk, hv, [tokens])
    weights, x = _inputs(hk, hv, [tokens], "ordinary", 9400)
    meta = _prefill_case([tokens], [7], [True])
    pools = _pools(hk, hv, seed=9402)
    expected_pools = [p.clone() for p in pools]
    storage = torch.full((tokens * hv * 128 + 2,), -1, dtype=torch.bfloat16, device="cuda")
    out = storage[offset : offset + tokens * hv * 128].view(tokens, hv * 128)
    with torch.inference_mode():
        expected = op(x, *weights.args(), *meta, *expected_pools)
        assert op(x, *weights.args(), *meta, *pools, out=out) is out
    assert _same(out, expected)
    assert all(_same(a, b) for a, b in zip(pools, expected_pools, strict=True))
    assert storage[-1] == -1
    if offset:
        assert storage[0] == -1


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=TP1_GEOMETRIES))
def test_serial_graph_replay_writes_preallocated_output(hk, hv):
    tokens = 65
    op = _op(hk, hv, [tokens])
    weights, x = _inputs(hk, hv, [tokens], "ordinary", 9500)
    meta = _prefill_case([tokens], [7], [True])
    pools = _pools(hk, hv, seed=9502)
    # Element-aligned caller storage is supported, including on replay.
    out = torch.empty(tokens * hv * 128 + 1, dtype=torch.bfloat16, device="cuda")[1:].view(
        tokens, hv * 128
    )
    with torch.inference_mode():
        op(x, *weights.args(), *meta, *pools)  # out=None warms the same executable.
        torch.cuda.current_stream().synchronize()
        graph, captured = _capture(lambda: op(x, *weights.args(), *meta, *pools, out=out))
        assert captured is out
        assert all(not isinstance(c, dict) for c in op._kernel._cute.compiled.values())
        for seed in (9503, 9504):
            new_x = _projected(tokens, hk, hv, seed=seed)
            new_pools = _pools(hk, hv, seed=seed + 2)
            expected_pools = [p.clone() for p in new_pools]
            expected = op(new_x, *weights.args(), *meta, *expected_pools)
            x.copy_(new_x)
            for dst, src in zip(pools, new_pools, strict=True):
                dst.copy_(src)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.current_stream().synchronize()
            assert _same(out, expected)
            assert all(_same(a, b) for a, b in zip(pools, expected_pools, strict=True))


# CI checks overlap ordering and freed sources with a misaligned preallocated Hv=48 output.
@pytest.mark.parametrize(
    "preallocated,hk,hv", ci_cases([False, True], GEOMETRIES, core=[(True, 16, 48)])
)
@pytest.mark.parametrize(
    "drop_sources", ci_cases([False, True], core=[(True,)]), ids=["ordering", "free-sources"]
)
def test_overlap_orders_user_stream_and_protects_sources(
    hk, hv, preallocated, drop_sources, monkeypatch
):
    # Overlap is a geometry default only where it measured faster; force it here.
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "overlap=1")
    tokens = 2048
    op = _op(hk, hv, [tokens])
    user = torch.cuda.Stream()
    assert user != torch.cuda.default_stream()
    with torch.cuda.stream(user), torch.inference_mode():
        weights, x = _inputs(hk, hv, [tokens], "ordinary", 9600)
        meta = _prefill_case([tokens], [7], [True])
        pools = _pools(hk, hv, seed=9602)
        expected_pools = [p.clone() for p in pools]
        expected = op(x, *weights.args(), *meta, *expected_pools)
        user.synchronize()
        launcher = op._kernel._cute
        key = next(iter(launcher.compiled))
        assert isinstance(launcher.compiled[key], dict)
        # All sources originate on the user stream; ordinary stream allocation
        # lifetimes also protect views dropped immediately after submission.
        args = tuple(t.clone() for t in (x, *weights.args(), *meta, *pools))
        out_source = torch.empty(tokens * hv * 128 + 1, dtype=torch.bfloat16, device=x.device)
        out = out_source[1:].view(tokens, hv * 128) if preallocated else None
        result = op(*args, out=out)
        if preallocated:
            assert result is out
        # Consumers run immediately on the caller stream, before any host wait.
        observed = tuple(t.clone() for t in (result, *args[-2:]))
        if drop_sources:
            del args, result, out, out_source, weights, x, meta, pools
            torch.cuda.empty_cache()
            # Exercise reuse of freed input/output-sized allocations as well.
            churn = torch.empty(
                tokens, Geometry(hk, hv).proj_width, dtype=torch.bfloat16, device="cuda"
            ).fill_(float("nan"))
        torch.cuda.current_stream().synchronize()  # Only the user stream is synchronized.
        assert _same(observed[0], expected)
        assert all(_same(a, b) for a, b in zip(observed[1:], expected_pools, strict=True))
        if drop_sources:
            del churn


# CI checks short/long precision at TP1 and TP2, with ordinary and heavy inputs.
@pytest.mark.parametrize(
    "hk,hv,kind",
    ci_cases(
        GEOMETRIES,
        ["ordinary", "slow", "heavy"],
        core=[(16, 32, "ordinary"), (16, 48, "heavy"), (8, 16, "ordinary"), (8, 24, "heavy")],
    ),
)
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param(LENGTHS[case.values[0]], marks=case.marks)
        for case in ci_cases(range(len(LENGTHS)), core=[(0,)])
    ],
)
# CI keeps the default schedule; optional schedules remain exhaustive.
@pytest.mark.parametrize(
    "schedule",
    _schedules(
        [
            f"overlap={overlap},{schedule}"
            for overlap in (0, 1)
            for schedule in [
                "rec_mode=0,prep_mode=0",
                "rec_mode=1,prep_mode=0",
                "rec_mode=2,prep_mode=1",
                "rec_ws=1,bv=32",
                "bv=16,prefetch_depth=1",
            ]
        ]
    ),
)
def test_precision_against_fp64_and_fla(hk, hv, lengths, kind, schedule, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", schedule)
    op = _op(hk, hv, lengths)
    maxima = {"y": [], "state": []}
    for seed_index in range(5):
        weights, x, meta, base, pools, written = _precision_inputs(
            hk, hv, lengths, kind, seed_index
        )
        exact = _precision_reference(
            hk, hv, lengths, kind, seed_index, x, weights, meta, base, written
        )
        before = [p.clone() for p in pools]
        fla_pools = [p.clone() for p in base]
        with torch.inference_mode():
            y = op(x, *weights.args(), *meta, *pools)
            fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
        torch.cuda.synchronize()
        untouched = [s for s in range(SLOTS) if s not in written]
        _no_worse_than_fla("y", y, fy, exact[0], maxima["y"])
        _no_worse_than_fla(
            "state", pools[0][written], fla_pools[0][written], exact[1], maxima["state"]
        )
        assert _same(pools[1][written], exact[2])
        for pool, old in zip(pools, before, strict=True):
            assert _same(pool[untouched], old[untouched])
    # Per-seed RMS strict; the largest errors by the seed-spread rule (fla-core's unmodified
    # chain is the baseline: on the RTX 5090, 8x16 [64] one seed is 1.99x, the rest
    # 0.51-0.94x).
    for name, errors in maxima.items():
        _largest_no_worse(name, errors)


# CI keeps one short zero-slot pool and graph replay per TP1 geometry; TP2 is exhaustive.
@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=TP1_GEOMETRIES))
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param(lengths, marks=case.marks)
        for lengths, case in zip(
            [[16], [64], [128], [512], [128, 128, 128, 128]],
            ci_cases(range(5), core=[(0,)]),
            strict=True,
        )
    ],
)
def test_zero_slot_pool(hk, hv, lengths):
    op = _op(hk, hv, lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 8100)
    meta = _prefill_case(lengths, [-1] * len(lengths), [True] * len(lengths))
    pools = _pools(hk, hv, slots=0, seed=8102)
    expected = reference.prefill(x, weights, hk, *meta, *pools)[0]
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools)
        fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *pools)
    _no_worse_than_fla("stateless y", y, fy, expected)
    assert pools[0].numel() == pools[1].numel() == 0
    with torch.inference_mode():
        graph, captured = _capture(lambda: op(x, *weights.args(), *meta, *pools))
        graph.replay()
        torch.cuda.synchronize()
    assert _same(captured, y)


# CI replays new device metadata on a default-schedule Hv=32 variable-length request.
@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(16, 32)]))
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param(LENGTHS[case.values[0]], marks=case.marks)
        for case in ci_cases(range(len(LENGTHS)), core=[(2,)])
    ],
)
@pytest.mark.parametrize(
    "schedule",
    _schedules(
        [
            f"overlap={overlap},{schedule}"
            for overlap in (0, 1)
            for schedule in [
                "rec_mode=0,prep_mode=0",
                "rec_mode=2,prep_mode=1",
                "rec_ws=1,bv=32",
                "bv=16,prefetch_depth=1",
            ]
        ]
    ),
)
def test_graph_replay_reads_new_device_metadata(hk, hv, lengths, schedule, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", schedule)
    op = _op(hk, hv, lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 9100)
    count = len(lengths)
    meta = _prefill_case(lengths, list(range(count)), [False] * count)
    pools = _pools(hk, hv, seed=9102)
    args = (x, *weights.args(), *meta, *pools)
    with torch.inference_mode():
        op(*args)
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(*args))
        new_meta = _prefill_case(
            [1] * (count - 1) + [sum(lengths) - count + 1],
            [8, -1, 3, 5][:count],
            [True, True, False, True][:count],
        )
        new_pools = _pools(hk, hv, seed=9103)
        expected_pools = [p.clone() for p in new_pools]
        new_x = _projected(sum(lengths), hk, hv, seed=9104)
        expected = op(new_x, *weights.args(), *new_meta, *expected_pools)
        for dst, src in zip((x, *meta, *pools), (new_x, *new_meta, *new_pools), strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
    assert _same(y, expected)
    assert all(_same(a, b) for a, b in zip(pools, expected_pools, strict=True))


# CI checks short and long default dispatch for TP1; TP2 is exhaustive here.
@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=TP1_GEOMETRIES))
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param(lengths, marks=case.marks)
        for lengths, case in zip(
            [[64], [128], [512], [128, 128, 128, 128], [2048]],
            ci_cases(range(5), core=[(0,)]),
            strict=True,
        )
    ],
)
def test_default_dispatch(hk, hv, lengths):
    op = _op(hk, hv, lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 9300)
    count = len(lengths)
    with torch.inference_mode():
        op(
            x,
            *weights.args(),
            *_prefill_case(lengths, [-1] * count, [False] * count),
            *_pools(hk, hv, seed=9302, slots=0),
        )
    launcher = op._kernel._cute
    assert launcher.compiled  # The accepted device actually ran CuTe, with no force flag.
    key = next(iter(launcher.compiled))
    # The row-selected tune (ROW_TUNES) decides; the fused v2 kernel always runs serially.
    selected = launcher._row_tune(sum(lengths), count)[0]
    fused = bool(selected.get("v2", 0))
    expected = "serial" if fused else _cute_prefill.pipeline_variant(sum(lengths), selected)
    assert key[2] == expected
    if fused:
        return
    tune = launcher._effective_tune(hv, key[3], key)
    overlap = isinstance(launcher.compiled[key], dict)
    assert tune["prep_mode"] == int(overlap and hv <= 32)
    # Per-geometry defaults: overlap only where it measured faster.
    assert overlap == (
        bool(selected["overlap"])
        and sum(lengths) >= selected["overlap_min_tokens"]
        and not launcher.launch_blocking
    )


@pytest.mark.parametrize("index", range(4))
def test_frozen_weights_fail_before_launch(index):
    op = _op(16, 32, [64])
    weights, x = _inputs(16, 32, [64], "ordinary", 9400)
    args = list(weights.args())
    args[index].requires_grad_(True)
    op._kernel = lambda *args: pytest.fail("launched before frozen-weight check")
    with pytest.raises(ValueError, match="frozen base weight"):
        op(x, *args, *_prefill_case([64], [-1], [False]), *_pools(16, 32, seed=9402, slots=0))


@pytest.mark.parametrize("input_index", range(3))
def test_grad_inputs_fail_before_launch(input_index):
    op = _op(16, 32, [64])
    weights, x = _inputs(16, 32, [64], "ordinary", 9400)
    pools = _pools(16, 32, seed=9402, slots=0)
    (x, *pools)[input_index].requires_grad_(True)
    op._kernel = lambda *args: pytest.fail("launched before autograd check")
    with pytest.raises(RuntimeError, match="gradient|autograd"):
        op(x, *weights.args(), *_prefill_case([64], [-1], [False]), *pools)


@pytest.mark.parametrize("tokens", [64, pytest.param(128, marks=pytest.mark.exhaustive)])
def test_unwarmed_layout_refused_and_warm_calls_do_not_host_sync(tokens):
    op = _op(16, 32, [tokens])
    weights, x = _inputs(16, 32, [tokens], "ordinary", 9500)
    tail = (
        *weights.args(),
        *_prefill_case([tokens], [-1], [False]),
        *_pools(16, 32, seed=9502, slots=0),
    )
    with torch.inference_mode():
        op(x, *tail)
        torch.cuda.synchronize()
        wide = torch.empty((tokens, x.shape[1] + 8), dtype=x.dtype, device=x.device)
        wide[:, : x.shape[1]].copy_(x)
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="once eagerly"):
            with torch.cuda.graph(graph):
                op(wide[:, : x.shape[1]], *tail)
        torch.cuda.set_sync_debug_mode("error")
        try:
            op(x, *tail)
        finally:
            torch.cuda.set_sync_debug_mode("default")


# CI keeps the default layout's alignment and slot/history checks.
@pytest.mark.parametrize(
    "tune",
    _schedules(
        [
            "split_h=1",
            "u_store=bf16",
            "ut_split=1",
            "w_split=1",
            "ut_split=1,w_split=1,k_split=1,update_split=1,split_h=1",
            "form=tk,t_store=fp32,v_store=fp32",
            "form=tk,t_store=bf16,r_split=0,k_split=0",
            "rec_ws=1,bv=32",
            "rec_ws=1,bv=16,update_split=1,u_store=bf16,o_bf16=1",
            "bv=16,recurrence_warps=2",
            "bv=16,recurrence_warps=4",
            "bv=16,recurrence_warps=8",
            "o_bf16=1",
            "update_split=1",
            "solve_mma=0",
            "solve_mma=1",
            "overlap=0,merge_prepare=0,bv=32,prepare_warps=4",
            "overlap_key=0",
            "rec_mode=2,prep_mode=1,prefetch_depth=1,bv=32",
            "rec_mode=2,prep_mode=1,prefetch_depth=4,bv=32",
            "merge_prepare=0",
            "rec_mode=1",
            "rec_mode=1,k_split=1,update_split=1,split_h=1,w_split=1",
            "rec_mode=1,bv=16,u_store=bf16",
            "bv=64",
            "prepare_warps=8,recurrence_warps=8",
            "mma_m=16,prepare_warps=4",
            "mma_n=16,prepare_warps=4",
        ]
    ),
)
def test_tuned_layouts_and_misalignment(tune, monkeypatch):
    """All layout families preserve slot/history semantics and alignment fallbacks."""
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", tune)
    hk, hv, lengths = 16, 32, [1, 65]
    weights, x = _inputs(hk, hv, lengths, "ordinary", 9700)
    meta = _prefill_case(lengths, [2, -1], [True, True])
    pools = _pools(hk, hv, seed=9702, slots=3)
    exact = reference.prefill(x, weights, hk, *meta, *pools)
    fla_pools = [p.clone() for p in pools]
    aligned_pools = [p.clone() for p in pools]
    op = _op(hk, hv, lengths)
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *aligned_pools)
        fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
    config = _cute_prefill.read_tune(tune)
    reduced_precision = (
        config["u_store"] == "bf16"
        or config["o_bf16"]
        or (config["form"] == "tk" and config["t_store"] == "bf16")
    )
    if reduced_precision:
        # Ablations only: loose FP64 sanity bound, never the product FLA gate.
        for actual, expected in ((y, exact[0]), (aligned_pools[0][2], exact[1][2])):
            torch.testing.assert_close(actual.double(), expected.double(), atol=0.1, rtol=0.1)
    else:
        _no_worse_than_fla("tuned y", y, fy, exact[0])
        _no_worse_than_fla("tuned state", aligned_pools[0][2], fla_pools[0][2], exact[1][2])
    assert _same(aligned_pools[1], exact[2])

    def misalign(t):
        storage = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
        view = storage[1:].view(t.shape)
        view.copy_(t)
        return view

    wide = torch.empty(x.shape[0], x.shape[1] + 1, dtype=x.dtype, device=x.device)
    unaligned_x = wide[:, 1:]
    unaligned_x.copy_(x)
    unaligned_pools = [misalign(p) for p in pools]
    unaligned_args = (
        unaligned_x,
        *(misalign(t) for t in weights.args()),
        *(misalign(t) for t in meta),
        *unaligned_pools,
    )
    with torch.inference_mode():
        uy = op(*unaligned_args)
        assert _same(uy, y)
        assert all(_same(a, b) for a, b in zip(unaligned_pools, aligned_pools, strict=True))
        graph, captured = _capture(lambda: op(*unaligned_args))
        for dst, src in zip(unaligned_pools, pools, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
    assert _same(captured, y)
    assert all(_same(a, b) for a, b in zip(unaligned_pools, aligned_pools, strict=True))


# CI excludes the opt-in knob sweep entirely; default flag replay remains in core.
@knobs_only
@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "overlap,overlap_key,prep_mode,merge_prepare,rec_mode,prefetch_depth",
    [
        (overlap, overlap_key, prep_mode, merge_prepare, rec_mode, prefetch_depth)
        for overlap in (0, 1)
        for overlap_key in (0, 1)
        for prep_mode in (0, 1)
        for merge_prepare in (0, 1)
        for rec_mode in (0, 1, 2)
        for prefetch_depth in (1, 4)
    ],
)
def test_flag_reset_on_replays_with_growing_and_shrinking_chunk_counts(
    overlap, overlap_key, prep_mode, merge_prepare, rec_mode, prefetch_depth, monkeypatch
):
    schedule = (
        f"overlap_min_tokens=0,overlap={overlap},overlap_key={overlap_key},prep_mode={prep_mode},"
        f"merge_prepare={merge_prepare},rec_mode={rec_mode},bv=32,prefetch_depth={prefetch_depth}"
    )
    _flag_replays(monkeypatch, schedule, bool(overlap))


def test_default_flag_reset_on_replays(monkeypatch):
    _flag_replays(monkeypatch, "", True)


def _flag_replays(monkeypatch, schedule, overlap):
    # Fixed T=512, addresses unchanged; active chunk count grows/shrinks 8 -> 9 -> 8.
    hk, hv, lengths = 2, 4, [256, 256]
    weights, x = _inputs(hk, hv, lengths, "ordinary", 10100)
    meta = _prefill_case(lengths, [2, -1], [True, True])
    pools = _pools(hk, hv, slots=3, seed=10102)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "overlap=0")

    # General admission binds this geometry through the real product dispatch.
    def small_op():
        op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
        assert op.provider == "cute"
        assert op.selected_variant(512, 2) == "cute"
        return op

    serial = small_op()
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", schedule)
    op = small_op()
    with torch.inference_mode():
        # Warm the exact captured shape/layout through the op so its provider
        # compiles and the capture guard records the provider's own variant.
        op(x, *weights.args(), *meta, *pools)
        torch.cuda.synchronize()
        graph, captured = _capture(lambda: op(x, *weights.args(), *meta, *pools))
        launcher = op._kernel._cute
        assert any(isinstance(c, dict) for c in launcher.compiled.values()) == bool(overlap)
        for replay, new_lengths in enumerate(([257, 255], [1, 511], [128, 384], [256, 256])):
            new_meta = _prefill_case(new_lengths, [2, -1], [bool(replay % 2), True])
            new_x = _projected(512, hk, hv, seed=10200 + replay)
            new_pools = _pools(hk, hv, slots=3, seed=10300 + replay)
            expected_pools = [p.clone() for p in new_pools]
            expected = serial(new_x, *weights.args(), *new_meta, *expected_pools)
            for dst, src in zip((x, *meta, *pools), (new_x, *new_meta, *new_pools), strict=True):
                dst.copy_(src)
            graph.replay()
            torch.cuda.synchronize()
            assert _same(captured, expected)
            assert all(_same(a, b) for a, b in zip(pools, expected_pools, strict=True))


# CI keeps graph concurrency, the cheaper of the two measured two-stream cases.
@pytest.mark.parametrize("graphs", ci_cases([False, True], core=[(True,)]), ids=["eager", "graphs"])
def test_concurrent_overlap_calls_two_streams_complete(graphs, monkeypatch):
    # Two recurrence grids (768 or 1536 CTAs each) can each fill the GPU. Keep
    # inputs, outputs and captured scratch alive for both independent calls.
    hk, hv, lengths = 16, 48, [2048] * 8
    need = 12 * 2**30
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        pytest.skip("concurrent PDL stress requires CUDA_LAUNCH_BLOCKING disabled")

    def wait_complete(streams):
        deadline = time.monotonic() + 120
        while not all(stream.query() for stream in streams):
            if time.monotonic() >= deadline:
                pytest.fail("concurrent overlap streams did not complete within 120 seconds")
            time.sleep(0.01)
        # Never enter a blocking synchronize until the watchdog observed completion.
        for stream in streams:
            stream.synchronize()

    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "overlap=0")
    serial = _op(hk, hv, lengths)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "overlap=1")
    operations = [_op(hk, hv, lengths) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    cases, expected = [], []
    current = torch.cuda.current_stream()
    with torch.inference_mode():
        for index, operation in enumerate(operations):
            weights, x = _inputs(hk, hv, lengths, "ordinary", 12000 + index * 10)
            meta = _prefill_case(lengths, list(range(8)), [True] * 8)
            base = _pools(hk, hv, slots=8, seed=12002 + index * 10)
            reference_pools = [p.clone() for p in base]
            reference_y = serial(x, *weights.args(), *meta, *reference_pools)
            wait_complete([current])
            expected.append((reference_y, *reference_pools))
            pools = [p.clone() for p in base]
            out = torch.empty_like(reference_y)
            cases.append((x, weights, meta, base, pools, out))
            # Warm compilation and the allocator on each actual launch stream,
            # separately, before submitting any concurrent work.
            stream = streams[index]
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                operation(x, *weights.args(), *meta, *pools, out=out)
            wait_complete([stream])
            launcher = operation._kernel._cute
            assert all(set(c) == {"pipeline", "norm"} for c in launcher.compiled.values())
            records = _cute_prefill.launch_info(
                hk,
                hv,
                sum(lengths),
                len(lengths),
                launcher._effective_tune(hv, next(iter(launcher.compiled))[3]),
            )
            recurrence = next(r for r in records if r["kernel"] == "recurrence")
            assert recurrence["ctas"] >= 2 * launcher.sms

        def run(index):
            x, weights, meta, base, pools, out = cases[index]
            for dst, src in zip(pools, base, strict=True):
                dst.copy_(src)
            return operations[index](x, *weights.args(), *meta, *pools, out=out)

        captured = []
        for index, stream in enumerate(streams):
            stream.wait_stream(current)
            if graphs:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    run(index)
                captured.append(graph)
        for iteration in range(50):
            order = (0, 1) if iteration % 2 == 0 else (1, 0)
            for index in order:
                with torch.cuda.stream(streams[index]):
                    if graphs:
                        captured[index].replay()
                    else:
                        run(index)
            wait_complete(streams)
            for case, reference_values in zip(cases, expected, strict=True):
                actual = (case[-1], *case[-2])
                assert all(torch.equal(a, b) for a, b in zip(actual, reference_values, strict=True))


# CI keeps a long initialized single sequence at Hv=48.
@pytest.mark.parametrize(
    "case,initial,hk,hv",
    ci_cases(
        ["serial-varlen", "long-varlen", "long-single"],
        [False, True],
        GEOMETRIES,
        core=[("long-single", True, 16, 48)],
    ),
)
def test_intermediate_captures_match_independent_truncated_runs(hk, hv, initial, case):
    _check_intermediate_captures(hk, hv, initial, case)


@pytest.mark.parametrize("initial", [False, True])
def test_intermediate_captures_on_gb10_8x16_v2_row(initial):
    # Product dispatch, no tune string: on GB10 this row must take v2 and keep the
    # capture contract (bitwise y, exports equal to truncated runs); sm120 runs v1.
    gb10 = torch.cuda.get_device_capability() == (12, 1)
    _check_intermediate_captures(8, 16, initial, "v2-varlen", 1 if gb10 else None)


@pytest.mark.parametrize(
    "initial,row",
    ci_cases(
        [False, True],
        ["8x24-single", "8x16-quad"],
        core=[(True, "8x24-single"), (False, "8x16-quad")],
    ),
)
def test_intermediate_captures_on_gb10_vsplit2_rows(initial, row):
    # Product dispatch on GB10: one 8x24 sequence of >= 1024 tokens and 8x16 N=4 at
    # T >= 4096 take v2 with vsplit 2 (two-CTA clusters fuse the gated norm).
    gb10 = torch.cuda.get_device_capability() == (12, 1)
    hk, hv, case = (8, 24, "long-single") if row == "8x24-single" else (8, 16, "v2-quad")
    _check_intermediate_captures(hk, hv, initial, case, 2 if gb10 else None)


def _check_intermediate_captures(hk, hv, initial, case, expect_v2=None):
    """expect_v2: the vsplit product dispatch must select (0 = v1), or None to skip."""
    # Keep full pools only for the two product calls; references use one scratch slot.
    # Include compiler scratch and allocator headroom in the existing memory-guard style.
    need = 3 * SLOTS * hv * 128 * 128 * 4 + (1 << 30)
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    if case == "long-single":
        lengths, slots = [2049], [0]
        positions, targets = [[58, 2049]], [[6, 7]]
    elif case == "v2-varlen":
        # T >= 4096 and N >= 5: the GB10 8x16 row that product dispatch routes to v2.
        lengths = [2048, 1024, 640, 257, 130, 64]
        slots = [0, 1, 2, 3, -1, 4]
        positions = [[58, 250], [64, 1024], [1, 2], [3, 257], [7, 1], [-(2**32), 2**32 + 1]]
        targets = [[6, 7], [8, 9], [10, 11], [12, 13], [14, -1], [15, 16]]
    elif case == "v2-quad":
        # T >= 4096 and N = 4: the GB10 8x16 row that product dispatch routes to vsplit 2.
        lengths = [2048, 1024, 640, 520]
        slots = [0, 1, -1, 2]
        positions = [[58, 250], [64, 1024], [1, 2], [3, 520]]
        targets = [[6, 7], [8, 9], [10, 11], [12, 13]]
    else:
        lengths = [257 if case == "serial-varlen" else 2048, 64, 2, 3, 7, 1]
        slots = [0, 1, 2, 3, -1, 4]
        # For initial absolute position 70, p=58 and p=58+3*64 are absolute
        # 64 boundaries. Other entries cover chunk ends, p<3, p=L and invalids.
        positions = [[58, 250], [64, 0], [1, 2], [3, 4], [7, 1], [-(2**32), 2**32 + 1]]
        targets = [[6, 7], [8, 9], [10, 11], [12, 13], [14, -1], [15, 16]]
    # One capture per sequence, at floor(L / 64) * 64 (p < 64 writes nothing).
    positions = [[length // 64 * 64] for length in lengths]
    targets = [r[:1] for r in targets]
    meta = _prefill_case(lengths, slots, [initial] * len(lengths))
    op = _op(hk, hv, lengths)
    if expect_v2 is not None:
        launcher = op._kernel._cute
        tune = launcher._row_tune(sum(lengths), len(lengths))[0]
        assert (tune["vsplit"] if tune.get("v2", 0) else 0) == expect_v2
    weights, x = _inputs(hk, hv, lengths, "ordinary", 10600)
    base = _pools(hk, hv, seed=10602)
    # Poison fresh slots: captures must not read old state/history when initial=False.
    if not initial:
        for pool in base:
            for slot in slots:
                if slot >= 0:
                    pool[slot].fill_(float("nan"))
    without = [p.clone() for p in base]
    exported = [p.clone() for p in base]
    rows = torch.tensor([r[0] for r in targets], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        expected_y = op(x, *weights.args(), *meta, *without)
        actual_y = op(x, *weights.args(), *meta, *exported, capture_rows=rows)
    assert _same(actual_y, expected_y)
    written = {s for s in slots if s >= 0}
    for slot in written:
        assert _same(exported[0][slot], without[0][slot])
        assert _same(exported[1][slot], without[1][slot])
    begin = 0
    for seq, (length, slot) in enumerate(zip(lengths, slots, strict=True)):
        for p, row in zip(positions[seq], targets[seq], strict=True):
            if row < 0 or p <= 0 or p > length:
                continue
            written.add(row)
            # Separate prefill invocation, original history/state, scratch slot 0.
            # A stateless original request becomes a fresh scratch slot so its
            # final state can be observed. No reference reads the captured pools.
            scratch_base = [
                pool[slot : slot + 1].clone() if slot >= 0 else torch.zeros_like(pool[:1])
                for pool in base
            ]
            scratch = [pool.clone() for pool in scratch_base]
            prefix_meta = _prefill_case([p], [0], [initial and slot >= 0])
            prefix = x[begin : begin + p]
            with torch.inference_mode():
                op(prefix, *weights.args(), *prefix_meta, *scratch)
                _, exact, exact_history = reference.prefill(
                    prefix, weights, hk, *prefix_meta, *scratch_base
                )
            got, truncated = exported[0][row], scratch[0][0]
            assert torch.isfinite(got).all()
            capture_error = (got.double() - exact[0]).abs().max().item()
            truncated_error = (truncated.double() - exact[0]).abs().max().item()
            # FMA prefix accumulation and tensor-core MMA have different reduction
            # orders; allow 10% of the truncated run's FP64 error plus FP32 noise.
            assert capture_error <= truncated_error * 1.10 + 1e-6, (
                case,
                seq,
                p,
                capture_error,
                truncated_error,
            )
            # At most 0.2% of the state's max-absolute scale, with a 1e-6 floor
            # for tiny states. This also rules out a vacuous shared reference error.
            scale = exact[0].abs().max().item()
            assert (got - truncated).abs().max().item() <= 2e-3 * scale + 1e-6
            assert _same(exported[1][row], scratch[1][0])
            assert _same(exported[1][row], exact_history[0])
        begin += length
    # Includes positive target rows paired with p=0/p>L, and neighboring guards.
    for row in range(SLOTS):
        if row not in written:
            assert _same(exported[0][row], base[0][row])
            assert _same(exported[1][row], base[1][row])


def _assert_largest_mean(largest):
    """Five-seed mean of the largest errors, as in the FP64 precision matrix."""
    for name, errors in largest.items():
        ours, fla = zip(*errors, strict=True)
        ours_mean, fla_mean = sum(ours) / len(ours), sum(fla) / len(fla)
        assert ours_mean <= fla_mean + ABSOLUTE_FLOOR, (name, ours, fla)


# CI keeps symbolic-N compile sharing on the cheaper Hv=48 overlap geometry.
@pytest.mark.parametrize("overlap,hk,hv", ci_cases([0, 1], GEOMETRIES, core=[(1, 16, 48)]))
@pytest.mark.parametrize("captures", ci_cases([0, 1], core=[(0,)]))
def test_sequence_count_shares_compile(hk, hv, overlap, captures, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"overlap={overlap},overlap_min_tokens=512")
    op = _op(hk, hv, [32])
    launcher = op._kernel._cute
    seen = set()
    # Five counts are five seeds: largest errors compare as a mean (see _no_worse_than_fla).
    largest: dict[str, list[tuple[float, float]]] = {"y": [], "state": []}
    for count in (1, 2, 3, 5, 17):
        lengths = [32] * count
        if overlap:
            lengths[0] += max(0, 512 - sum(lengths))
        weights, x = _inputs(hk, hv, lengths, "ordinary", 10300 + count)
        slots = list(reversed(range(count)))
        meta = _prefill_case(lengths, slots, [bool(n % 2) for n in range(count)])
        base = _pools(hk, hv, seed=10350 + count)
        pools, fla_pools = ([p.clone() for p in base] for _ in range(2))
        capture_args = {}
        if captures:
            # Disabled entries still compile the capture ABI, without aliasing slots.
            capture_args = {
                "capture_rows": torch.full((count,), -1, dtype=torch.int32, device="cuda")
            }
        with torch.inference_mode():
            exact = reference.prefill(x, weights, hk, *meta, *base)
            fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
            y = op(x, *weights.args(), *meta, *pools, **capture_args)
        torch.cuda.synchronize()
        _no_worse_than_fla("y", y, fy, exact[0], largest["y"])
        _no_worse_than_fla(
            "state", pools[0][slots], fla_pools[0][slots], exact[1][slots], largest["state"]
        )
        assert _same(pools[1], exact[2])
        assert _same(pools[0][count:], base[0][count:])
        key = launcher.compile_key(Geometry(hk, hv), x, count, captures)
        seen.add((key[2], key[3]))
        # Each variant compiles with its capture twin.
        assert len(launcher.compiled) == 2 * len(seen) <= 4
        assert len({(k[2], k[3], k[-1]) for k in launcher.compiled}) == len(launcher.compiled)
        assert key[2] == (0 if overlap else "serial")
    _assert_largest_mean(largest)


# CI checks serial graph reuse at a new sequence count on Hv=32.
@pytest.mark.parametrize("overlap,hk,hv", ci_cases([0, 1], GEOMETRIES, core=[(0, 16, 32)]))
def test_new_sequence_count_graph_capture_reuses_warm_variant(hk, hv, overlap, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", f"overlap={overlap},overlap_min_tokens=512")
    op = _op(hk, hv, [32, 32])
    launcher = op._kernel._cute

    def inputs(count):
        lengths = [32] * count
        if overlap:
            lengths[0] += max(0, 512 - sum(lengths))
        weights, x = _inputs(hk, hv, lengths, "ordinary", 10400 + count)
        meta = _prefill_case(lengths, list(range(count)), [True] * count)
        return weights, x, meta

    # N=3 and N=5 are both multi-wave even at TP2 Hv=16 on the 188-SM PRO 6000.
    warm_count, replay_count = (3, 5) if hk == 8 else (2, 3)
    weights2, x2, meta2 = inputs(warm_count)
    weights, x, meta = inputs(replay_count)
    base = _pools(hk, hv, seed=10450)
    pools, fla_pools = ([p.clone() for p in base] for _ in range(2))
    with torch.inference_mode():
        op(x2, *weights2.args(), *meta2, *pools)
        assert launcher.capture_variant(
            Geometry(hk, hv), x2, warm_count
        ) == launcher.capture_variant(Geometry(hk, hv), x, replay_count)
        assert len(launcher.compiled) == 2  # the variant and its capture twin
        torch.cuda.synchronize()
        graph, y = _capture(lambda: op(x, *weights.args(), *meta, *pools))
        assert len(launcher.compiled) == 2
        largest: dict[str, list[tuple[float, float]]] = {"y": [], "state": []}
        for seed in range(10451, 10456):
            x.copy_(_projected(x.shape[0], hk, hv, seed=seed))
            for dst, dst_fla, src in zip(pools, fla_pools, base, strict=True):
                dst.copy_(src)
                dst_fla.copy_(src)
            exact = reference.prefill(x, weights, hk, *meta, *base)
            fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fla_pools)
            graph.replay()
            torch.cuda.synchronize()
            _no_worse_than_fla("y", y, fy, exact[0], largest["y"])
            _no_worse_than_fla(
                "state",
                pools[0][:replay_count],
                fla_pools[0][:replay_count],
                exact[1][:replay_count],
                largest["state"],
            )
            assert _same(pools[1], exact[2])
            assert _same(pools[0][replay_count:], base[0][replay_count:])
        _assert_largest_mean(largest)


# CI checks disabled capture rows and short sequences on a serial Hv=32 request.
@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(16, 32)]))
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param(lengths, marks=case.marks)
        for lengths, case in zip([[200, 63], [600]], ci_cases(range(2), core=[(0,)]), strict=True)
    ],
)
def test_disabled_capture_entries_write_nothing(hk, hv, lengths):
    _check_disabled_capture_entries(hk, hv, lengths)


def _check_disabled_capture_entries(hk, hv, lengths):
    # Row -1 disables a sequence's capture; p = floor(L / 64) * 64 < 64 disables one whose
    # row is a real pool row.
    count = len(lengths)
    slots = list(range(count))
    meta = _prefill_case(lengths, slots, [True] * count)
    op = _op(hk, hv, lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 10700)
    base = _pools(hk, hv, seed=10701)
    without = [p.clone() for p in base]
    exported = [p.clone() for p in base]
    targets = [-1 if length >= 64 else 6 + i for i, length in enumerate(lengths)]
    rows = torch.tensor(targets, dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        expected_y = op(x, *weights.args(), *meta, *without)
        actual_y = op(x, *weights.args(), *meta, *exported, capture_rows=rows)
    torch.cuda.synchronize()
    assert _same(actual_y, expected_y)
    for got, ref in zip(exported, without, strict=True):
        assert _same(got, ref)


_OUTPUT_SPLIT_LENGTHS = [[64], [1, 2, 61], [128, 128, 128, 128], [1024], [2048, 512, 130]]


def test_row_selected_output_split_matches_explicit_fused(monkeypatch):
    hk, hv = 8, 16
    cc = torch.cuda.get_device_capability()
    # Exercise the shipped sm120 row; force a narrow band on devices without an o_split row.
    if not any(row[-1].get("o_split") for row in _cute_prefill.ROW_TUNES.get((cc, (hk, hv)), ())):
        monkeypatch.setitem(
            _cute_prefill.ROW_TUNES,
            (cc, (hk, hv)),
            ((600, 600, None, {"o_split": 1, "bvo": 128}),),
        )
    automatic = _op(hk, hv, [600])
    # The same automatic operation must switch between both schedules. Bitwise
    # equality holds against the explicit fused schedule with the row's other knobs
    # (bv and overlap change the reduction grouping, so they are matched).
    for tokens, split in ((600, 1), (128, 0)):
        selected = automatic._kernel._cute._row_tune(tokens, 1)[0]
        assert selected["o_split"] == split
        monkeypatch.setenv(
            "CTI_GDN_CUTE_TUNE", f"o_split=0,bv={selected['bv']},overlap={selected['overlap']}"
        )
        fused = _op(hk, hv, [tokens])
        assert fused._kernel._cute._row_tune(tokens, 1)[0]["o_split"] == 0
        weights, x = _inputs(hk, hv, [tokens], "ordinary", 14800)
        meta = _prefill_case([tokens], [7], [True])
        pools = _pools(hk, hv, seed=14802)
        expected_pools = [p.clone() for p in pools]
        with torch.inference_mode():
            expected = fused(x, *weights.args(), *meta, *expected_pools)
            actual = automatic(x, *weights.args(), *meta, *pools)
        assert torch.equal(actual, expected)
        assert all(torch.equal(a, b) for a, b in zip(pools, expected_pools, strict=True))


@pytest.mark.parametrize("hk,hv", [(8, 16), (8, 24), (16, 32)])
def test_row_tunes_precision_against_fp64_and_fla(hk, hv, monkeypatch):
    # The shipped default (row tunes active) at a length every sm120 row band covers;
    # on devices without row tunes this is the geometry default.
    test_precision_against_fp64_and_fla(hk, hv, [1024], "ordinary", "", monkeypatch)


@pytest.mark.parametrize(
    "hk,hv,length_index,overlap,aq_out,captures,bv,bvo,prefetch_depth,update_split",
    ci_cases(
        [(8, 16), (8, 24), (16, 32), (16, 48)],
        range(len(_OUTPUT_SPLIT_LENGTHS)),
        [0, 1],
        [0, 1],
        [0, 1],
        [16, 32],
        [32, 64, 128],
        [(1, 0), (1, 1), (4, 0), (4, 1)],
        core=[
            (8, 16, 3, 1, 0, 1, 16, 64, 4, 0),
            (8, 24, 1, 0, 1, 1, 32, 128, 1, 1),
            (8, 16, 0, 0, 1, 0, 32, 32, 1, 0),
        ],
    ),
)
def test_output_split_matches_fused(
    hk,
    hv,
    length_index,
    overlap,
    aq_out,
    captures,
    bv,
    bvo,
    prefetch_depth,
    update_split,
    monkeypatch,
    extra_tune="",
    lengths_override=None,
):
    lengths = _OUTPUT_SPLIT_LENGTHS[length_index] if lengths_override is None else lengths_override
    schedule = (
        f"overlap={overlap},overlap_min_tokens=0,aq_out={aq_out},bv={bv},bvo={bvo},"
        f"prefetch_depth={prefetch_depth},update_split={update_split}{extra_tune}"
    )
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", schedule + ",o_split=0")
    fused = _op(hk, hv, lengths)
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", schedule + ",o_split=1")
    split = _op(hk, hv, lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 14700)
    count = len(lengths)
    meta = _prefill_case(lengths, list(range(count)), [s % 2 == 0 for s in range(count)])
    pools = _pools(hk, hv, seed=14702)
    expected_pools = [p.clone() for p in pools]
    kwargs = {}
    if captures:
        # Every sequence of 64 tokens or more captures at floor(L / 64) * 64.
        kwargs = {"capture_rows": torch.arange(count, 2 * count, dtype=torch.int32, device="cuda")}
    with torch.inference_mode():
        expected = fused(x, *weights.args(), *meta, *expected_pools, **kwargs)
        actual = split(x, *weights.args(), *meta, *pools, **kwargs)
    assert torch.equal(actual, expected)
    # Full pools include final state, history, both capture rows and untouched slots.
    assert all(torch.equal(a, b) for a, b in zip(pools, expected_pools, strict=True))


def test_output_split_precision_against_fp64_and_fla(monkeypatch):
    test_precision_against_fp64_and_fla(
        8,
        16,
        [1024],
        "ordinary",
        "o_split=1,bv=16,bvo=64,aq_out=1,overlap=1,overlap_min_tokens=0",
        monkeypatch,
    )


@pytest.mark.exhaustive
@pytest.mark.parametrize(
    "bv,bvo,aq_out,extra_tune",
    [
        (64, 32, 0, ""),
        (64, 32, 1, ",o_bf16=1"),
        (64, 128, 1, ""),
        (16, 32, 1, ",output_warps=8,u_store=bf16"),
        (32, 128, 1, ",output_warps=8,o_bf16=1"),
        (32, 64, 0, ",prepare_warps=4,recurrence_warps=2,mma_m=16,mma_n=32"),
    ],
)
def test_output_split_extra_layouts(bv, bvo, aq_out, extra_tune, monkeypatch):
    test_output_split_matches_fused(
        8,
        24,
        3,
        1,
        aq_out,
        1,
        bv,
        bvo,
        1,
        1,
        monkeypatch,
        extra_tune,
        lengths_override=[65],
    )


@pytest.mark.exhaustive
def test_output_split_metadata_changing_graph_replays(monkeypatch):
    _flag_replays(monkeypatch, "o_split=1,bv=16,bvo=64,overlap=1,overlap_min_tokens=0", True)


@pytest.mark.parametrize(("hk", "hv"), [(8, 16), (8, 24)])
def test_engine_warm_up_then_run_at_tp2(hk, hv):
    """The engine's warm-up (tokens 1, 2, 4, ... with max_sequences = min(tokens, 16)) primes
    every variant, with and without captures, on valid batches without a fault, and a chunked
    call with captures then runs (finite y; accuracy is the other tests' job)."""
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute"
    for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        op.warm_up(tokens, max_sequences=min(tokens, 16))
        torch.cuda.synchronize()
    lengths = [300, 1, 0, 64]
    weights, projected = _inputs(hk, hv, lengths, "ordinary", 7100)
    case = _prefill_case(lengths, [3, 4, -1, 5], [True, False, False, True])
    state, conv_state = _pools(hk, hv, seed=7101)
    rows = torch.tensor([8, 10, -1, 11], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        y = op(projected, *weights.args(), *case, state, conv_state, capture_rows=rows)
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all()


def test_priming_more_sequences_than_rows_writes_nothing_past_its_buffers(monkeypatch):
    """Deterministic D8 regression: the priming call at T = 1, N = 3 (sm120 TP2 8x16's first
    pair for the 3-sequence wave class). Every buffer the CuTe call allocates (scratch and y)
    gets a canary tail; none may change. Before the fix the batch was [0, 1, 2, 1] and the
    second sequence's row past T wrote output past y and the FP32 o scratch."""
    import math

    from ctikernels.gdn.inference import _PrimeScratch

    guard, real, buffers = 1 << 14, torch, []

    class _Guarded:
        def __getattr__(self, name):
            return getattr(real, name)

        def empty(self, *shape, dtype=None, device=None, **options):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
                shape = tuple(shape[0])
            numel = math.prod(shape)
            base = real.empty(numel + guard, dtype=dtype, device=device, **options)
            base.view(real.uint8).fill_(0xA5)
            buffers.append((base, numel))
            return base[:numel].view(shape)

    op = GatedDeltaNetPrefill(num_k_heads=8, num_v_heads=16, device="cuda")
    assert op.provider == "cute"
    cute = op._kernel._cute
    scratch = _PrimeScratch(op._geometry, torch.device("cuda"), 4, op._geometry.proj_width)
    for captures in (0, 1):
        args, keywords = scratch.args(1, 3, captures)
        cute(op._geometry, *args, **keywords)  # compile outside the guard
        torch.cuda.synchronize()
        monkeypatch.setattr(_cute_prefill, "torch", _Guarded())
        cute(op._geometry, *args, **keywords)
        torch.cuda.synchronize()
        monkeypatch.setattr(_cute_prefill, "torch", real)
        assert buffers
        for base, numel in buffers:
            tail = base.view(torch.uint8)[numel * base.element_size() :]
            assert bool((tail == 0xA5).all()), (captures, base.dtype, numel)
        buffers.clear()
