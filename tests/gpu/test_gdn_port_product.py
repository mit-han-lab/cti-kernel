# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GP1 product qualification on sm120; exhaustive cases retain the full precision grid."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetPrefill, _cute_prefill, _triton_prefill
from ctikernels.gdn._contract import Geometry

from . import _gdn_reference as reference
from ._exhaustive import ci_cases
from ._gdn_prefill_cases import (
    ABSOLUTE_FLOOR,
    _fla_prefill,
    _inputs,
    _max_error,
    _no_worse_than_fla,
    _rms_error,
)
from .test_gdn import _capture, _pools, _prefill_case, _same, _weights
from .test_gdn_cute_prefill import _precision_inputs


def _reason():
    if not torch.cuda.is_available():
        return "port product tests require CUDA"
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0) or _triton_prefill.device_type(
        device
    ) not in (_triton_prefill.RTX_5090, _triton_prefill.RTX_PRO_6000):
        return "port product tests require RTX 5090 or RTX PRO 6000 sm120"
    if not _cute_prefill.available():
        return "port product tests require nvidia-cutlass-dsl==4.8.0"
    return None


_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
GEOMETRIES = [(16, 32), (16, 48), (8, 16), (8, 24)]
LENGTHS = [[16], [64], [1, 2, 61], [128], [64, 64], [512], [128, 128, 128, 128], [1024], [2048]]
# Lower edge of each shipped RTX 5090 N = 1 port band (Hv -> min T), written out independently
# of the provider table so a table edit must also edit this test. Hv 32 and 48 run the port at
# every T and every packed batch on both devices.
PORT_BAND = {16: 6144, 24: 3072}
# The qualified lengths of every enabled band (docs, "Dispatch qualification").
QUALIFIED = [
    *((8, 16, t) for t in (6144, 16384)),
    *((8, 24, t) for t in (3072, 4096, 16384)),
    *((16, 32, t) for t in (3072, 16384)),
    *((16, 48, t) for t in (1536, 4096, 16384)),
]


def _expected(hv, lengths):
    """The implementation the shipped table must select for this device and row."""
    kind = _triton_prefill.device_type(torch.device("cuda"))
    if kind not in (_triton_prefill.RTX_5090, _triton_prefill.RTX_PRO_6000):
        return "cute"
    if hv in (32, 48):
        return "port"
    if kind == _triton_prefill.RTX_PRO_6000:
        return "cute"
    return "port" if len(lengths) == 1 and sum(lengths) >= PORT_BAND[hv] else "cute"


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.delenv("CTI_GDN_PREFILL_CUTE", raising=False)
    # The shipped table runs the port on qualified one-sequence bands only; the port's
    # integration mechanics are tested with the port forced.
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "1")
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)


def _op(hk, hv):
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute" and op.supports_state_capture
    assert op._kernel._port is not None, "eligible product must bind the port"
    return op


def _v1(hk, hv, monkeypatch):
    with monkeypatch.context() as context:
        context.setenv("CTI_GDN_PREFILL_PORT", "0")
        op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute" and op._kernel._port is None
    return op


def _oracle_close(y, state, history, exact):
    # Empty/stateless/long smoke checks; the five-seed test below owns the stricter
    # no-worse-than-fla precision gate. Normalize against the reference's RMS/max.
    for got, want in ((y, exact[0]), (state, exact[1])):
        if got.numel():
            error = (got.double() - want.double()).abs()
            assert torch.isfinite(got).all()
            assert (
                error.square().mean().sqrt() <= 0.01 * want.double().square().mean().sqrt() + 1e-6
            )
            assert error.max() <= 0.03 * want.double().abs().max() + 1e-6
    assert _same(history, exact[2])


# The port runs FlashInfer's (vLLM's) rounding points. Its acceptance is a set of
# bounds against fla-core, both measured against FP64 over five seeds: per case, the mean RMS
# error <= 1.05x fla-core's and the largest error <= 1.25x; per test, the geometric means of
# those ratios <= 1.05. v1 keeps the frozen no-worse-than-fla-core gate.
PORT_RMS, PORT_MAX, PORT_GEOMEAN = 1.05, 1.25, 1.05
# Cases whose largest error exceeds 1.25x fla-core's on these rounding points (RTX 5090,
# 2026-10-21, five seeds): expected, held to the measured ratio plus 0.05.
PORT_MAX_EXPECTED = {
    (16, 32, "heavy", (2048,)): 1.34,
    # State max error identical to v1's (1.409e-01, 1.265x fla-core, RTX 5090 and PRO 6000).
    (16, 32, "heavy", (16,)): 1.27,
    (16, 32, "slow", (1024,)): 1.26,
    (16, 48, "heavy", (8192,)): 1.35,
    (16, 48, "heavy", (16384,)): 1.27,
    (16, 48, "ordinary", (512,)): 1.27,
    (16, 48, "ordinary", (4096,)): 1.34,
}
# 16x32 one-sequence T <= 256: the port's largest y error equals v1's on the same BF16 element
# (up to 1.85x fla-core); y's bound there is v1's ratio on the same seeds.
V1_SHARED_MAX = {(16, 32, (t,)) for t in (16, 64, 128, 256)}
# Packed batches past four sequences (sequence buckets 16-256).
MANY = [[64] * 16, [128] * 32, [32] * 64, [16] * 256]
_PORT_RATIOS = defaultdict(list)


@pytest.fixture(scope="module", autouse=True)
def _port_geomeans():
    yield
    for test, ratios in _PORT_RATIOS.items():
        for column, label in enumerate(("y rms", "y max", "state rms", "state max")):
            geomean = math.exp(sum(math.log(r[column]) for r in ratios) / len(ratios))
            assert geomean <= PORT_GEOMEAN, (test, label, geomean)


@pytest.mark.parametrize(
    "hk,hv,case,kind",
    ci_cases(
        GEOMETRIES,
        range(len(LENGTHS)),
        ["ordinary", "slow", "heavy"],
        core=[(hk, hv, 0, "ordinary") for hk, hv in GEOMETRIES]
        + [(8, 16, 8, "ordinary"), (16, 48, 4, "ordinary"), (8, 24, 6, "heavy")],
    ),
)
def test_precision_against_fp64_fla_and_v1(hk, hv, case, kind, monkeypatch):
    _precision_gate(hk, hv, LENGTHS[case], kind, monkeypatch, "fla matrix")


@pytest.mark.parametrize(
    "hk,hv,tokens,kind",
    ci_cases(QUALIFIED, ["ordinary", "slow", "heavy"], core=[(16, 48, 1536, "ordinary")]),
)
def test_qualified_port_bands_precision(hk, hv, tokens, kind, monkeypatch):
    # Every enabled band at its qualified lengths, under the default dispatch.
    _precision_gate(hk, hv, [tokens], kind, monkeypatch, "qualified bands")


@pytest.mark.parametrize(
    "hk,hv,case,kind",
    ci_cases(
        [(16, 32), (16, 48)],
        range(len(MANY)),
        ["ordinary", "slow", "heavy"],
        core=[(16, 32, 3, "ordinary"), (16, 48, 0, "ordinary")],
    ),
)
def test_many_sequence_port_precision(hk, hv, case, kind, monkeypatch):
    _precision_gate(hk, hv, MANY[case], kind, monkeypatch, "many sequences")


def _precision_gate(hk, hv, lengths, kind, monkeypatch, test):
    # The product gate: whatever the default per-row table chooses (port or v1). The forced
    # port is gated over the whole matrix by cp2_precision.py --all.
    monkeypatch.delenv("CTI_GDN_PREFILL_PORT")
    op, v1 = _op(hk, hv), _v1(hk, hv, monkeypatch)
    selected = op.selected_variant(sum(lengths), len(lengths))
    assert selected == _expected(hv, lengths), (selected, lengths)
    calls = []
    monkeypatch.setattr(op._kernel, "_port", _Spy(op._kernel._port, calls))
    errors = {name: [] for name in ("port y", "port state", "v1 y", "v1 state")}
    seeds = {name: [] for name in ("port y", "port state", "v1 y", "v1 state")}
    for seed in range(5):
        weights, x, meta, base, pools, written = _precision_inputs(hk, hv, lengths, kind, seed)
        exact = reference.prefill(x, weights, hk, *meta, *base)
        before = [p.clone() for p in pools]
        fp, vp = ([p.clone() for p in base] for _ in range(2))
        with torch.inference_mode():
            y = op(x, *weights.args(), *meta, *pools)
            fy = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fp)
            vy = v1(x, *weights.args(), *meta, *vp)
        for name, got, baseline, want in (
            ("port y", y, fy, exact[0]),
            ("port state", pools[0][written], fp[0][written], exact[1][written]),
            ("v1 y", vy, fy, exact[0]),
            ("v1 state", vp[0][written], fp[0][written], exact[1][written]),
        ):
            if selected == "port":
                assert torch.isfinite(got).all(), name
                seeds[name].append(
                    (_rms_error(got, want), _rms_error(baseline, want),
                     _max_error(got, want), _max_error(baseline, want))
                )  # fmt: skip
            if not (selected == "port" and name.startswith("port")):
                _no_worse_than_fla(name, got, baseline, want, errors[name])
        # Both product paths must fit the same precision budget, and their difference
        # must fit its triangle bound (no looser independent tolerance).
        bound = 2 * (PORT_RMS if selected == "port" else 1.0)
        assert _rms_error(y, vy) <= bound * _rms_error(fy, exact[0]) + ABSOLUTE_FLOOR
        assert _same(pools[1][written], exact[2][written])
        untouched = [s for s in range(base[0].shape[0]) if s not in written]
        for pool, old in zip(pools, before, strict=True):
            assert _same(pool[untouched], old[untouched])
    # The call ran the selected implementation (the port only when the table chose it).
    assert bool(calls) == (selected == "port"), (selected, len(calls))
    for name, pairs in errors.items():
        if pairs:
            ours, fla = zip(*pairs, strict=True)
            assert sum(ours) / len(ours) <= sum(fla) / len(fla) + ABSOLUTE_FLOOR, (name, pairs)
    if selected == "port":
        ratios = []
        limit = PORT_MAX_EXPECTED.get((hk, hv, kind, tuple(lengths)), PORT_MAX - 0.05) + 0.05

        def contract(rows):
            ours_rms, fla_rms, ours_max, fla_max = zip(*rows, strict=True)
            rms = (sum(ours_rms) + ABSOLUTE_FLOOR) / (sum(fla_rms) + ABSOLUTE_FLOOR)
            return rms, (max(ours_max) + ABSOLUTE_FLOOR) / (max(fla_max) + ABSOLUTE_FLOOR)

        shared = (hk, hv, tuple(lengths)) in V1_SHARED_MAX
        for quantity in ("y", "state"):
            rms, largest = contract(seeds[f"port {quantity}"])
            v1_largest = contract(seeds[f"v1 {quantity}"])[1]
            bound = max(limit, v1_largest) if shared and quantity == "y" else limit
            assert rms <= PORT_RMS, (quantity, rms)
            assert largest <= bound, (quantity, largest, bound)
            # A v1-shared row's y max enters the geomeans against v1 (the same element), not fla.
            ratios += [rms, largest / v1_largest if shared and quantity == "y" else largest]
        _PORT_RATIOS[test].append(ratios)


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(8, 16)]))
@pytest.mark.parametrize("port", [True, False], ids=["port", "v1"])
@pytest.mark.parametrize(
    "lengths",
    [
        [0, 2, 0, 1, 0],
        [0, 0, 0, 0, 0],
        pytest.param([0, 512, 0, 1, 0], marks=pytest.mark.exhaustive),
    ],
)
def test_empty_sequences_keep_or_clear_pools(hk, hv, port, lengths, monkeypatch):
    op = _op(hk, hv) if port else _v1(hk, hv, monkeypatch)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 400)
    meta = _prefill_case(lengths, [0, 1, 2, 3, -1], [True, False, False, True, True])
    base = _pools(hk, hv, 401, slots=5)
    for pool in base:
        pool[2].fill_(float("nan"))  # Fresh empty slots must clear without reading.
    pools = [p.clone() for p in base]
    exact = reference.prefill(x, weights, hk, *meta, *base)
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools)
    _oracle_close(y, *pools, exact)
    for pool, old in zip(pools, base, strict=True):
        assert _same(pool[0], old[0])
        assert torch.count_nonzero(pool[2]) == 0
        assert _same(pool[4], old[4])


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(8, 24)]))
@pytest.mark.parametrize("tokens", ci_cases([16, 2048], core=[(16,)]))
def test_stateless_y_and_pools_untouched(hk, hv, tokens):
    op = _op(hk, hv)
    weights, x = _inputs(hk, hv, [tokens], "ordinary", 500)
    meta = _prefill_case([tokens], [-1], [True])
    base = _pools(hk, hv, 501, slots=2)
    pools = [p.clone() for p in base]
    exact = reference.prefill(x, weights, hk, *meta, *base)
    out = torch.empty(tokens * hv * 128 + 1, dtype=torch.bfloat16, device="cuda")[1:].view(
        tokens, -1
    )
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools, out=out)
    assert y is out
    _oracle_close(y, *pools, exact)
    assert all(_same(p, b) for p, b in zip(pools, base, strict=True))


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(16, 32)]))
@pytest.mark.parametrize("route", ["base", "stride", "disabled"])
def test_v1_routes_are_bitwise_equal(hk, hv, route, monkeypatch):
    op, v1 = _op(hk, hv), _v1(hk, hv, monkeypatch)
    weights, x = _inputs(hk, hv, [17, 64], "ordinary", 600)
    if route in ("base", "stride"):
        padding, offset = (0, 1) if route == "base" else (1, 0)
        storage = torch.empty(
            x.shape[0] * (x.shape[1] + padding) + offset, device="cuda", dtype=x.dtype
        )
        x = storage[offset:].view(x.shape[0], x.shape[1] + padding)[:, : x.shape[1]].copy_(x)
    if route == "disabled":
        op = _v1(hk, hv, monkeypatch)
    meta = _prefill_case([17, 64], [0, 1], [True, False])
    base = _pools(hk, hv, 601, slots=8)
    a, b = ([p.clone() for p in base] for _ in range(2))
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *a)
        vy = v1(x, *weights.args(), *meta, *b)
    assert _same(y, vy) and all(_same(p, q) for p, q in zip(a, b, strict=True))
    assert all(v[0] == "cute" for _, v in op._guard._eager)


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(16, 32)]))
def test_captures_keep_the_port_and_its_output(hk, hv, monkeypatch):
    """Captures do not change dispatch: the captured port call equals the uncaptured one
    (y and slots bitwise); at p = L = 64 the capture is the slot's new state and history."""
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "1")
    op = _op(hk, hv)
    weights, x = _inputs(hk, hv, [17, 64], "ordinary", 600)
    meta = _prefill_case([17, 64], [0, 1], [True, False])
    base = _pools(hk, hv, 601, slots=8)
    a, b = ([p.clone() for p in base] for _ in range(2))
    rows = torch.tensor([4, 5], device="cuda", dtype=torch.int32)
    assert op.selected_variant(81, 2) == "port"
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *a, capture_rows=rows)
        uy = op(x, *weights.args(), *meta, *b)
    assert _same(y, uy)
    for p, q, old in zip(a, b, base, strict=True):
        assert _same(p[:2], q[:2]) and _same(p[5], p[1]) and _same(p[4], old[4])
    assert all(v[0] == "port" for _, v in op._guard._eager)


@pytest.mark.parametrize("hk,hv", ci_cases(GEOMETRIES, core=[(8, 16)]))
@pytest.mark.parametrize("cp", [False, True])
def test_warmup_graph_replays_new_metadata(hk, hv, cp):
    op = _op(hk, hv)
    # Non-CP shapes must select the port on every geometry (Hv 32/48 T <= 512 runs v1).
    lengths = [2048] if cp else [320, 0, 288]
    tokens, count = sum(lengths), len(lengths)
    weights, x = _inputs(hk, hv, lengths, "ordinary", 700)
    base = _pools(hk, hv, 701, slots=5)
    meta = _prefill_case(lengths, [0] if cp else [0, 1, -1], [True] if cp else [True, False, True])
    pools = [p.clone() for p in base]
    out = torch.empty(tokens, hv * 128, device="cuda", dtype=torch.bfloat16)
    op.warm_up(tokens, max_sequences=count)
    expected = ("mn", "seg") if cp else ("full",)
    assert op._kernel._port.variants(tokens, count) == expected
    assert ((), ("port", x.stride(0), expected)) in op._guard._eager
    with torch.inference_mode():
        graph, y = _capture(lambda: op(x, *weights.args(), *meta, *pools, out=out))
        assert y is out
        for replay in range(3):
            next_lengths = [tokens] if cp else ([0, 320, 288] if replay % 2 else [300, 296, 12])
            next_meta = _prefill_case(
                next_lengths,
                [2] if cp else [3, -1, 2],
                [bool(replay % 2)] if cp else [False, True, True],
            )
            for dst, src in zip(meta, next_meta, strict=True):
                dst.copy_(src)
            x.mul_(-1)
            for p, b in zip(pools, base, strict=True):
                p.copy_(b)
            eager_pools = [p.clone() for p in base]
            eager = op(x, *weights.args(), *meta, *eager_pools)
            graph.replay()
            torch.cuda.synchronize()
            assert _same(y, eager)
            assert all(_same(p, e) for p, e in zip(pools, eager_pools, strict=True))


def test_shipped_band_warms_only_reachable_variants(monkeypatch):
    # PORT-CS: at 16x32 on the RTX 5090 (170 SMs) every shipped-band T plans CP, so warm_up
    # builds mn + seg (one shared history commit) and never full; capture then compiles nothing.
    from ctikernels.gdn.chunk_ws import prefill

    if _triton_prefill.device_type(torch.device("cuda")) != _triton_prefill.RTX_5090:
        pytest.skip("the shipped port band is RTX 5090 only")
    monkeypatch.delenv("CTI_GDN_PREFILL_PORT", raising=False)
    op = _op(16, 32)
    port = op._kernel._port
    reachable = op._kernel.port_variants(8192, 1)
    op.warm_up(8192, max_sequences=1)
    if reachable != {("mn", "seg")}:
        pytest.skip(f"this device's CP plan reaches {sorted(reachable)}")
    assert sorted(key[0] for key in port.compiled) == ["mn", "seg"]
    assert len({id(commit) for _, commit in port.compiled.values()}) == 1
    lengths = [4096]
    weights, x = _inputs(16, 32, lengths, "ordinary", 900)
    meta = _prefill_case(lengths, [0], [True])
    pools = _pools(16, 32, 901, slots=1)

    def fail(*a, **kw):
        raise AssertionError("capture compiled")

    monkeypatch.setattr(prefill, "compile_prefill", fail)
    assert op.selected_variant(4096, 1) == "port"
    with torch.inference_mode():
        graph, y = _capture(lambda: op(x, *weights.args(), *meta, *pools))
        graph.replay()
        torch.cuda.synchronize()
    assert len(port.compiled) == 2


def test_unwarmed_port_capture_raises():
    op = _op(8, 16)
    weights, x = _inputs(8, 16, [2048], "ordinary", 800)
    meta = _prefill_case([2048], [0], [False])
    pools = _pools(8, 16, 801, slots=1)
    # Compile only full; global executable caches do not bypass the product guard.
    op.warm_up(16, max_sequences=1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="once eagerly"):
        _capture(lambda: op(x, *weights.args(), *meta, *pools))


def _two_heads(x, weights, hk, hv, heads):
    """Extract two independent value heads with their matching q/k and conv columns."""
    keys = [h // (hv // hk) for h in heads]
    dc = (2 * hk + hv) * 128
    conv = [
        *(k * 128 + j for k in keys for j in range(128)),
        *((hk + k) * 128 + j for k in keys for j in range(128)),
        *((2 * hk + h) * 128 + j for h in heads for j in range(128)),
    ]
    columns = (
        conv
        + [dc + h * 128 + j for h in heads for j in range(128)]
        + [dc + hv * 128 + h for h in heads]
        + [dc + hv * 128 + hv + h for h in heads]
    )
    small = reference.Weights(
        weights.conv_weight[conv], weights.A_log[heads], weights.dt_bias[heads], weights.norm_weight
    )
    return x[:, columns], small, conv


@pytest.mark.parametrize("hk,hv", ci_cases([(8, 16), (16, 32), (16, 48)], core=[(8, 16)]))
def test_262144_tokens_truncated_fp64_two_heads(hk, hv):
    """Bounded long-context smoke: no full-length FP64 loop or fla/v1 scratch.

    Strong decay erases a tail's omitted initial state before the compared rows.
    The deeper slow-decay and CP boundary qualification lives in the port suite.
    """
    tokens = 262144
    op = _op(hk, hv)
    weights = _weights(hk, hv, 900)
    weights.A_log.fill_(1)
    weights.dt_bias.fill_(2)
    generator = torch.Generator(device="cuda").manual_seed(901)
    x = torch.randn(
        tokens, op.geometry.proj_width, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    meta = _prefill_case([tokens], [0], [False])
    pools = _pools(hk, hv, 902, slots=1)
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools)
    assert op._kernel._port.plan(tokens, 1)[0] >= 2
    heads = [0, hv - 1]
    y_heads = y.view(tokens, hv, 128)
    for start, size, compare in ((0, 64, 64), (tokens - 256, 256, 64)):
        small_x, small_w, columns = _two_heads(x[start : start + size], weights, hk, hv, heads)
        small_state = torch.zeros(1, 2, 128, 128, device="cuda")
        small_history = torch.zeros(1, 3, len(columns), dtype=torch.bfloat16, device="cuda")
        if start:
            small_history[0].copy_(x[start - 3 : start, columns])
        small_meta = _prefill_case([size], [0], [bool(start)])
        exact = reference.prefill(small_x, small_w, 2, *small_meta, small_state, small_history)
        got_y = y_heads[start + size - compare : start + size, heads].reshape(compare, -1)
        if start:
            _oracle_close(
                got_y,
                pools[0][:, heads],
                pools[1][:, :, columns],
                (exact[0][-compare:], exact[1], exact[2]),
            )
        else:
            _oracle_close(got_y, exact[1], exact[2], exact)


def test_repacked_batch_runs_the_port(monkeypatch):
    # Hv48 [128]x4 plus a 513-token stateless sequence: packed batches at (16, 48) run the port
    # and match the FP64 oracle.
    monkeypatch.delenv("CTI_GDN_PREFILL_PORT")
    lengths = [128, 128, 128, 128, 513]
    op = _op(16, 48)
    assert op.selected_variant(sum(lengths), len(lengths)) == "port"
    weights, x = _inputs(16, 48, lengths, "ordinary", 600)
    meta = _prefill_case(lengths, [0, 1, 2, 3, -1], [True, True, True, True, True])
    base = _pools(16, 48, 601, slots=5)
    pools = [p.clone() for p in base]
    exact = reference.prefill(x, weights, 16, *meta, *base)
    calls = []
    port = op._kernel._port
    monkeypatch.setattr(op._kernel, "_port", _Spy(port, calls))
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools)
    assert len(calls) == 1
    _oracle_close(y, *pools, exact)


class _Spy:
    def __init__(self, port, calls):
        self._port, self._calls = port, calls

    def __getattr__(self, name):
        return getattr(self._port, name)

    def __call__(self, *args, **kwargs):
        self._calls.append(args)
        return self._port(*args, **kwargs)
