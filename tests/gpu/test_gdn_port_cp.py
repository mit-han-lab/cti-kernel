# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CP3 aliasing, precision, segment tails, history, graph replay, and IEEE fixup."""

from dataclasses import replace

import pytest
import torch

from ctikernels.gdn.chunk_ws._cp import cp_capacity, cp_plan
from ctikernels.gdn.chunk_ws.prefill import PortPrefill

from . import _gdn_reference as reference
from .test_gdn_chunk_ws import _reason, _same_bits
from .test_gdn_port_prefill import error, make_case

_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")


@pytest.fixture(scope="module")
def operators():
    cache = {}

    def get(hv, segments=None, cp=True, pdl=True):
        key = hv, segments, cp, pdl
        if key not in cache:
            cache[key] = PortPrefill(16, hv, "cuda", cp=cp, cp_segments=segments, pdl=pdl)
        return cache[key]

    return get


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("length", [128, 200, 1000, 1024, 4100])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("decay", ["ordinary", "slow"])
def test_cp_reference(operators, hv, length, initial, decay):
    x, weights, meta, base = make_case(16, hv, (length,))
    if decay == "slow":
        # dt_bias near -8: per-token decay ~1e-4, so segment transfers M stay O(1)
        # (ordinary decay makes M underflow within a few chunks).
        weights = replace(weights, dt_bias=weights.dt_bias - 6)
    meta[2].fill_(initial)
    baseline_pools = tuple(t.clone() for t in base)
    baseline = operators(hv, cp=False)(x, *weights.args(), *meta, *baseline_pools)
    # Share the expensive independent token-by-token oracle across all plans.
    exact_y, exact_state, exact_history = reference.prefill(x, weights, 16, *meta, *base)
    baseline_errors = error(baseline, exact_y), error(baseline_pools[0][1], exact_state[1])
    seen = set()
    for target in (2, 3, 4, 5, None):
        op = operators(hv, target)
        plan = cp_plan(length, hv, op.num_sms, target)
        # Still exercise the default dispatcher, but don't rerun equivalent overrides.
        if plan in seen and target is not None:
            continue
        seen.add(plan)
        pools = tuple(t.clone() for t in base)
        if not initial:
            pools[0][1].fill_(float("nan"))
            pools[1][1].fill_(float("nan"))
        # Guard allocation immediately after the tail; intermediate segments share y.
        storage = torch.full((length + 64, hv * 128), 123, device=x.device, dtype=x.dtype)
        out = storage[:length]
        assert op(x, *weights.args(), *meta, *pools, out=out) is out
        assert torch.isfinite(out).all() and torch.isfinite(pools[0][1]).all()
        assert error(out, exact_y) <= 1.15 * baseline_errors[0] + 1e-7, (target, plan, "y")
        assert error(pools[0][1], exact_state[1]) <= 1.15 * baseline_errors[1] + 1e-7, (
            target,
            plan,
            "state",
        )
        _same_bits(pools[1], baseline_pools[1])
        _same_bits(pools[1], exact_history)
        _same_bits(pools[0][[0, 2]], base[0][[0, 2]])
        assert (storage[length:] == 123).all()


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("empty_pool", [False, True])
def test_cp_unowned(operators, hv, empty_pool):
    x, weights, meta, base = make_case(16, hv, (200,))
    meta[1].fill_(-1)
    if empty_pool:
        base = tuple(t[:0] for t in base)
    for initial in (False, True):
        meta[2].fill_(initial)
        pools = tuple(t.clone() for t in base)
        baseline = operators(hv, cp=False)(x, *weights.args(), *meta, *pools)
        out = operators(hv, 4)(x, *weights.args(), *meta, *pools)
        exact, _, _ = reference.prefill(x, weights, 16, *meta, *base)
        assert error(out, exact) <= 1.15 * error(baseline, exact) + 1e-7
        assert torch.count_nonzero(out) > 0
        for got, expected in zip(pools, base, strict=True):
            _same_bits(got, expected)


@pytest.mark.parametrize("hv", [32, 48])
def test_cp_graph_replay_new_data_and_slot(operators, hv):
    x, weights, meta, base = make_case(16, hv, (1000,))
    op = operators(hv, 4)
    pools = tuple(t.clone() for t in base)
    out = torch.empty(x.shape[0], hv * 128, dtype=x.dtype, device=x.device)
    op(x, *weights.args(), *meta, *pools, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, *weights.args(), *meta, *pools, out=out)
    for slot, initial in ((2, True), (0, False)):
        x.copy_(torch.randn_like(x))
        meta[1].fill_(slot)
        meta[2].fill_(initial)
        eager_pools = tuple(t.clone() for t in base)
        eager = op(x, *weights.args(), *meta, *eager_pools)
        for dst, src in zip(pools, base, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        _same_bits(out, eager)
        for got, expected in zip(pools, eager_pools, strict=True):
            _same_bits(got, expected)
    assert len(op.compiled) == 2


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("length,segments", [(128, 2), (200, 2), (512, 4), (1000, 4)])
def test_cp_pdl_graph_bitwise(operators, hv, length, segments):
    """Direct A->C and A->fixup->C, with new graph inputs and in-place pools."""
    x, weights, meta, base = make_case(16, hv, (length,))
    runs = []
    for pdl in (False, True):
        op = operators(hv, segments, pdl=pdl)
        # Fewer SMs (GB10) cap the forced count; the fixup still runs for every J > 2.
        expected = min(segments, cp_capacity(hv, op.num_sms))
        assert cp_plan(length, hv, op.num_sms, segments)[0] == expected >= 2
        pools = tuple(t.clone() for t in base)
        storage = torch.full((length + 64, hv * 128), 123, dtype=x.dtype, device=x.device)
        out = storage[:length]
        op(x, *weights.args(), *meta, *pools, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, *weights.args(), *meta, *pools, out=out)
        runs.append((graph, pools, storage))
    for slot, initial in ((2, True), (0, False), (1, True), (-1, True)):
        x.copy_(torch.randn_like(x))
        meta[1].fill_(slot)
        meta[2].fill_(initial)
        for graph, pools, storage in runs:
            for dst, src in zip(pools, base, strict=True):
                dst.copy_(src)
            storage[:length].fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        _same_bits(runs[0][2], runs[1][2])
        assert (runs[1][2][length:] == 123).all()
        assert torch.isfinite(runs[1][2][:length]).all()
        for off, on in zip(runs[0][1], runs[1][1], strict=True):
            _same_bits(off, on)


@pytest.mark.parametrize("hv", [32, 48])
@pytest.mark.parametrize("segments", [3, 4, 6])
@pytest.mark.parametrize("pdl", [False, True])
def test_fixup_fp64(hv, segments, pdl):
    from ctikernels.gdn.chunk_ws._cp_fixup import compile_fixup

    torch.manual_seed(19)
    cap = segments + 2
    w = torch.randn(3, cap, hv, 128, 128, device="cuda") * 0.02
    # Non-symmetric M with a strong diagonal detects transposition/layout errors
    # and keeps earlier segment contributions visible in every output.
    w[2].add_(torch.eye(128, device="cuda") * 0.8)
    before = w.clone()
    # Poison the destination rows: later steps must use the state produced by
    # this launch, not the original (or previous graph replay's) W contents.
    w[0, 2:segments].fill_(float("nan"))
    expected = w[0, 1].double()
    outputs = []
    for segment in range(1, segments - 1):
        expected = expected @ w[2, segment].double() + w[1, segment].double()
        outputs.append(expected)
    cc = torch.cuda.get_device_capability()
    fixup = compile_fixup(hv, f"--enable-tvm-ffi --gpu-arch sm_{cc[0]}{cc[1]}a", pdl)
    fixup(w, segments)
    # 128 ordered FP32 FFMAs per dot product, followed by one FP32 add, with
    # FP32 state rounding at each step. The independent oracle stays FP64.
    for segment, expected in enumerate(outputs, 2):
        torch.testing.assert_close(w[0, segment].double(), expected, rtol=2e-5, atol=2e-7)
    _same_bits(w[1:], before[1:])
    _same_bits(w[0, :2], before[0, :2])
    _same_bits(w[0, segments:], before[0, segments:])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fixup(w, segments)
    w[0, 2:segments].fill_(float("nan"))
    graph.replay()
    for segment, expected in enumerate(outputs, 2):
        torch.testing.assert_close(w[0, segment].double(), expected, rtol=2e-5, atol=2e-7)
    _same_bits(w[1:], before[1:])
    _same_bits(w[0, :2], before[0, :2])
    _same_bits(w[0, segments:], before[0, segments:])


def _oracle_bound(actual, exact, *, state=False):
    # Same 2% RMS scale used by the port qualification; CP must additionally
    # stay within 1.15x the measured non-CP oracle error below.
    rms = exact.square().mean().sqrt().item()
    assert error(actual, exact) <= 0.02 * rms + (1e-5 if state else 1e-4)


def _long_context(hk, hv, decay, *, graph=False, packed=False):
    from ctikernels.gdn.chunk_ws.prefill import compile_key

    total = 262144
    lengths = (total - 128, 128) if packed else (total,)
    x, weights, meta, base = make_case(hk, hv, lengths)
    if decay == "heavy":
        weights = replace(weights, dt_bias=weights.dt_bias + 4)
    elif decay in ("slow", "near_zero"):
        weights = replace(weights, dt_bias=weights.dt_bias - (6 if decay == "slow" else 12))
    if packed:
        meta[1].copy_(torch.tensor([1, 2], dtype=torch.int32, device=x.device))
    heads = [0, hv - 1]  # Include the last channel as well as the largest token offsets.
    op = PortPrefill(hk, hv, x.device)
    baseline_op = PortPrefill(hk, hv, x.device, cp=False)
    plans = {op.plan(total, len(lengths)), cp_plan(total, hv, op.num_sms, op.cp_capacity)}
    if packed:
        plans = {(1, total)}
    boundaries = sorted({s * length for j, length in plans for s in range(1, j)})
    exact_y, exact_state, exact_history, starts = reference.prefill_chunked(
        x, weights, hk, *meta, *base, value_heads=heads, boundaries=boundaries
    )
    pools = tuple(t.clone() for t in base)
    out = baseline_op(x, *weights.args(), *meta, *pools)
    baseline_y = out.view(total, hv, 128)[:, heads].clone()
    owned = meta[1].tolist()
    baseline_state = pools[0][owned][:, heads].clone()
    _oracle_bound(baseline_y, exact_y)
    _oracle_bound(baseline_state, exact_state[owned], state=True)
    baseline_errors = error(baseline_y, exact_y), error(baseline_state, exact_state[owned])
    # Non-CP prefixes establish the same error gate at every CP segment start.
    boundary_errors = {}
    for boundary in boundaries:
        prefix_pools = tuple(t.clone() for t in base)
        prefix_meta = (torch.tensor([0, boundary], dtype=torch.int32, device=x.device), *meta[1:])
        baseline_op(x[:boundary], *weights.args(), *prefix_meta, *prefix_pools, out=out[:boundary])
        value = prefix_pools[0][1, heads]
        _oracle_bound(value, starts[0, boundary], state=True)
        boundary_errors[boundary] = error(value, starts[0, boundary])
    del baseline_y, baseline_state
    targets = (None, op.cp_capacity) if not packed else (None,)
    seen = set()
    for target in targets:
        op.port_options = replace(op.port_options, cp_segments=target)
        j, seg_len = op.plan(total, len(lengths))
        if (j, seg_len) in seen:
            continue
        seen.add((j, seg_len))
        # cp_segments affects only host planning. Reuse the already compiled
        # executables when sweeping targets, so the core pays for each once.
        if op.compiled:
            for variant in op.variants(total, len(lengths)):
                previous = next(value for key, value in op.compiled.items() if key[0] == variant)
                key = compile_key(hk, hv, x.stride(0), op.port_options, variant)
                op.compiled[key] = previous
        op._ensure_compiled(x.stride(0), op.variants(total, len(lengths)), False)
        workspaces = []
        if j > 1:
            key = compile_key(hk, hv, x.stride(0), op.port_options, "mn")
            launch, commit = op.compiled[key]

            def record_workspace(*args, workspaces=workspaces, launch=launch):
                workspaces.append(args[12])
                launch(*args)

            op.compiled[key] = record_workspace, commit
        for dst, src in zip(pools, base, strict=True):
            dst.copy_(src)
        op(x, *weights.args(), *meta, *pools, out=out)
        actual_y = out.view(total, hv, 128)[:, heads]
        assert error(actual_y, exact_y) <= 1.15 * baseline_errors[0] + 1e-7
        assert (
            error(pools[0][owned][:, heads], exact_state[owned]) <= 1.15 * baseline_errors[1] + 1e-7
        )
        # Explicit high-address tail gate; a prefix-only check cannot catch wrapping.
        _oracle_bound(actual_y[-64:], exact_y[-64:])
        _same_bits(pools[1], exact_history)
        untouched = [s for s in range(base[0].shape[0]) if s not in owned]
        _same_bits(pools[0][untouched], base[0][untouched])
        if j > 1:
            w = workspaces.pop()
            assert w.shape == (3, j, hv, 128, 128)
            _same_bits(w[0, 0], base[0][1])
            for segment in range(1, j):
                boundary = segment * seg_len
                assert (
                    error(w[0, segment, heads], starts[0, boundary])
                    <= 1.15 * boundary_errors[boundary] + 1e-7
                )
            op.compiled[key] = launch, commit
            del w
    if graph:
        # The final plan is maximum segments, with PDL enabled. Capture once and
        # replay new projected data, slot, and initial-state flags.
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            op(x, *weights.args(), *meta, *pools, out=out)
        for slot, initial in ((2, False), (-1, True), (0, True)):
            x.normal_()
            meta[1].fill_(slot)
            meta[2].fill_(initial)
            eager_pools = tuple(t.clone() for t in base)
            eager = op(x, *weights.args(), *meta, *eager_pools)
            for dst, src in zip(pools, base, strict=True):
                dst.copy_(src)
            captured.replay()
            _same_bits(out, eager)
            for actual, expected in zip(pools, eager_pools, strict=True):
                _same_bits(actual, expected)
            del eager


def test_long_context_ci():
    """CI core: two-head FP64 oracle, default/max CP, all boundaries and far tail."""
    _long_context(8, 16, "ordinary")


@pytest.mark.exhaustive
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
@pytest.mark.parametrize("decay", ["ordinary", "heavy", "slow", "near_zero"])
def test_long_context_exhaustive(hk, hv, decay):
    _long_context(hk, hv, decay, graph=True)


@pytest.mark.exhaustive
@pytest.mark.parametrize("hk,hv", [(16, 32), (16, 48)])
def test_long_context_large_sequence_offset(hk, hv):
    _long_context(hk, hv, "ordinary", packed=True)
