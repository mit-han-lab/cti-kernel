# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GEN-GDN correctness admission grid; CPU-safe collection, GPU execution only.

IDs include local heads and heads01_08/heads09_16/heads17_24/heads25_32 shards.
Use -m 'not exhaustive' for the small CI core; -m exhaustive retains the sweep.
"""

import pytest
import torch

from ctikernels.gdn import (
    GatedDeltaNetDecode,
    GatedDeltaNetPrefill,
    _triton_prefill,
)
from ctikernels.gdn._contract import Geometry

from ._exhaustive import ci_cases
from ._gdn_prefill_cases import (
    ABSOLUTE_FLOOR,
    SLOTS,
    _fla_prefill,
    _inputs,
    _no_worse_than_fla,
    _poison,
    _pools,
    _prefill_case,
    _same,
    reference,
)
from .test_gdn_cute_prefill import _check_intermediate_captures
from .test_gdn_triton import (
    test_every_configuration_matches_the_fp64_reference as _check_decode,
)

GRID = [(hk, r * hk) for r in (1, 2, 3, 4, 6, 8) for hk in range(1, 33)]
CORE = {(1, 1), (2, 4), (2, 6), (4, 8), (4, 12), (3, 24), (5, 30), (32, 32), (32, 256), (16, 128)}
NEW = [(2, 4), (4, 12), (1, 8)]
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or _triton_prefill.device_type(torch.device("cuda")) is None,
    reason="GEN-GDN qualifies RTX 5090, RTX PRO 6000 and GB10",
)


def _id(hk, hv):
    lo = (hk - 1) // 8 * 8 + 1
    return f"hk{hk}-hv{hv}-heads{lo:02}_{lo + 7:02}"


def _grid_cases():
    return [
        pytest.param(*p.values, marks=p.marks, id=_id(*p.values)) for p in ci_cases(GRID, core=CORE)
    ]


@pytest.fixture(autouse=True)
def _product_defaults(monkeypatch):
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "1")
    monkeypatch.delenv("CTI_GDN_CUTE_TUNE", raising=False)
    # Every bucket v1: the provider table's Triton rows are qualified by their own suites.
    monkeypatch.setattr(_triton_prefill.PROVIDER, "table", lambda device_class: {})


@pytest.mark.parametrize("hk,hv", _grid_cases())
def test_decode_grid(hk, hv):
    assert GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda").provider == "triton"
    sms = torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    for batch in sorted({1, 3, 17, 2 * sms // hv + 1}):
        # Reuse every original gate: FP64 ULP, fla RMS/ULP, random padded slots,
        # poisoned unused rows, bitwise conv history and untouched pool rows.
        _check_decode(hk, hv, batch)


def _check_prefill(hk, hv, lengths, initial):
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert op.provider == "cute"
    assert op.selected_variant(sum(lengths), len(lengths)) == "cute"
    slots = [9, 2, -1, 5, 7, 11][: len(lengths)]
    meta = _prefill_case(lengths, slots, [initial] * len(lengths))
    # An external provider need not support empty sequences: remove those rows
    # for the independent baseline, preserving the identical packed token order.
    live = [(length, slot) for length, slot in zip(lengths, slots, strict=True) if length]
    ref_meta = _prefill_case([p[0] for p in live], [p[1] for p in live], [initial] * len(live))
    written = sorted(slot for length, slot in live if slot >= 0)
    zeroed = [
        slot
        for length, slot in zip(lengths, slots, strict=True)
        if not length and slot >= 0 and not initial
    ]
    read = set(written) if initial else set()
    largest = {"y": [], "state": []}
    for seed_index in range(5):
        seed = 4200 + 10000 * seed_index
        weights, x = _inputs(hk, hv, lengths, "ordinary", seed)
        base = _pools(hk, hv, seed=seed + 2)
        pools = [_poison(p, read, "nan", seed + 3) for p in base]
        before = [p.clone() for p in pools]
        exact_y, exact_state, exact_history = reference.prefill(x, weights, hk, *ref_meta, *base)
        fla_pools = [p.clone() for p in base]
        with torch.inference_mode():
            y = op(x, *weights.args(), *meta, *pools)
            fla_y = _fla_prefill(Geometry(hk, hv), x, *weights.args(), *ref_meta, *fla_pools)
        _no_worse_than_fla(
            "y",
            y.view(-1, hv, 128),
            fla_y.view(-1, hv, 128),
            exact_y.view(-1, hv, 128),
            largest["y"],
        )
        _no_worse_than_fla(
            "state",
            pools[0][written],
            fla_pools[0][written],
            exact_state[written],
            largest["state"],
        )
        assert _same(pools[1][written], exact_history[written])
        for pool in pools:
            assert _same(pool[zeroed], torch.zeros_like(pool[zeroed]))
        untouched = [row for row in range(SLOTS) if row not in written + zeroed]
        for got, old in zip(pools, before, strict=True):
            assert _same(got[untouched], old[untouched])
    for name, errors in largest.items():
        ours, fla = zip(*errors, strict=True)
        assert sum(ours) / len(ours) <= sum(fla) / len(fla) + ABSOLUTE_FLOOR, (name, ours, fla)


@pytest.mark.parametrize("hk,hv", _grid_cases())
@pytest.mark.parametrize("initial", [False, True], ids=["fresh", "initial"])
def test_prefill_grid(hk, hv, initial):
    _check_prefill(hk, hv, [1, 63, 64, 65, 300, 0], initial)


@pytest.mark.exhaustive
@pytest.mark.parametrize("hk,hv", [pytest.param(*p, id=_id(*p)) for p in NEW])
@pytest.mark.parametrize("lengths", [[8192], [6000, 1500, 692]], ids=["N1", "N3"])
def test_prefill_long(hk, hv, lengths):
    _check_prefill(hk, hv, lengths, True)


@pytest.mark.parametrize("hk,hv", [pytest.param(*p, id=_id(*p)) for p in NEW])
@pytest.mark.parametrize("initial", [False, True], ids=["fresh", "initial"])
def test_capture(hk, hv, initial):
    _check_intermediate_captures(hk, hv, initial, "serial-varlen", expect_v2=0)


# Keep one mixed/zero-token pair per implementation in CI; larger head counts,
# separate v2 norm and cluster splits remain in the exhaustive sweep.
EMPTY_TUNES = [
    (16, 32, "v1", "v2=0,overlap=0"),
    (2, 6, "v1", "v2=0,overlap=0"),
    (2, 6, "overlap", "overlap=1,overlap_min_tokens=0"),
    (8, 16, "v2", "v2=1"),
    (8, 16, "v2-snorm", "v2=1,v2_snorm=1"),
    (8, 16, "v2-split", "v2=1,vsplit=2"),
]
EMPTY_CASES = [
    pytest.param(*p.values, marks=p.marks, id=f"{p.values[2]}-{_id(*p.values[:2])}")
    for p in ci_cases(EMPTY_TUNES, core=EMPTY_TUNES[1:4])
]


@pytest.mark.parametrize("hk,hv,name,tune", EMPTY_CASES)
@pytest.mark.parametrize("zero_tokens", [False, True], ids=["mixed", "T0"])
def test_empty_sequences(hk, hv, name, tune, zero_tokens, monkeypatch):
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", tune)
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    lengths = [0, 0, 0] if zero_tokens else [0, 0, 65, 1]
    slots = [9, 2, -1] if zero_tokens else [9, 2, 5, -1]
    initial = [False, True, False] if zero_tokens else [False, True, False, False]
    weights, x = _inputs(hk, hv, lengths, "ordinary", 5200)
    pools = _pools(hk, hv, seed=5202)
    before = [p.clone() for p in pools]
    meta = _prefill_case(lengths, slots, initial)
    # Nonnegative capture rows of sequences shorter than 64 (empty ones included) stay
    # untouched; the 65-token sequence's row is disabled.
    rows = [-1 if length >= 64 else 12 + i for i, length in enumerate(lengths)]
    rows = torch.tensor(rows, dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools, capture_rows=rows)
    assert y.shape == (sum(lengths), hv * 128)
    written = {9} if zero_tokens else {9, 5}
    untouched = [i for i in range(SLOTS) if i not in written]
    for got, old in zip(pools, before, strict=True):
        assert _same(got[9], torch.zeros_like(got[9]))
        assert _same(got[untouched], old[untouched])
    # Decode continuation must be exactly the explicit-zero-pool result.
    expected = [p.clone() for p in pools]
    for pool in expected:
        pool[9].zero_()
    _, dx = _inputs(hk, hv, [1, 1, 1], "ordinary", 5300)
    decode_slots = torch.tensor([9, 2, -1], dtype=torch.int32, device="cuda")
    decode = GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda")
    with torch.inference_mode():
        actual_y = decode(dx, *weights.args(), decode_slots, *pools)
        expected_y = decode(dx, *weights.args(), decode_slots, *expected)
    assert _same(actual_y, expected_y)
    assert all(_same(a, b) for a, b in zip(pools, expected, strict=True))
    # N=0 still performs no pool writes.
    before = [p.clone() for p in pools]
    with torch.inference_mode():
        op(x[:0], *weights.args(), *_prefill_case([], [], []), *pools)
    assert all(_same(a, b) for a, b in zip(pools, before, strict=True))


@pytest.mark.parametrize(
    "hk,hv,batch",
    [
        pytest.param(*p.values, marks=p.marks, id=f"{_id(*p.values[:2])}-B{p.values[2]}")
        for p in ci_cases([(16, 48), (4, 12), (1, 8)], [64, 129, 256], core=[(16, 48, 129)])
    ],
)
def test_decode_large_batch(hk, hv, batch):
    _check_decode(hk, hv, batch)


@pytest.mark.parametrize(
    "case",
    ci_cases(["tiny", "N70000", "chunks65600"], core=[("tiny",)]),
)
def test_folded_prefill_grids(case, monkeypatch):
    from ctikernels.gdn._cute_prefill import chunk_capacity

    hk, hv = (1, 1) if case == "chunks65600" else (1, 2)
    lengths = (
        [65600 * 64, 256]
        if case == "chunks65600"
        else [1] * (69999 if case == "N70000" else 3) + [200]
    )
    n, tokens = len(lengths), sum(lengths)
    # Pool rows for the last 64 sequences only: a pool of 70000 rows exceeds the
    # int32 element count of a flattened state tensor. Earlier sequences use slot -1.
    m = min(n, 64)
    # Pools + scratch + input/output and allocator/compiler headroom. The large
    # tests skip devices without enough free memory, before creating inputs.
    need = (
        m * hv * 128 * 128 * 4
        + tokens * (2 * Geometry(hk, hv).proj_width + hv * 256)
        + chunk_capacity(tokens, n) * hv * 200000
        + (2 << 30)
    )
    if torch.cuda.mem_get_info()[0] < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB of free device memory")
    # Pin schedule and one-wave behavior for exact batch-vs-standalone comparison.
    monkeypatch.setenv("CTI_GDN_CUTE_TUNE", "v2=0,overlap=0,bv=64,prefetch_depth=1")
    op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    weights, x = _inputs(hk, hv, lengths, "ordinary", 6100)
    pools = [
        torch.randn(m, hv, 128, 128, device="cuda"),
        torch.randn(m, 3, Geometry(hk, hv).conv_dim, device="cuda", dtype=torch.bfloat16),
    ]
    standalone = [p[-1:].clone() for p in pools]
    slots = [-1] * (n - m) + list(range(m))
    initial = [i % 2 == 1 and slots[i] >= 0 for i in range(n)]
    meta = _prefill_case(lengths, slots, initial)
    last_meta = _prefill_case([lengths[-1]], [0], [initial[-1]])
    with torch.inference_mode():
        y = op(x, *weights.args(), *meta, *pools)
        last_y = op(x[-lengths[-1] :], *weights.args(), *last_meta, *standalone)
    assert _same(y[-lengths[-1] :], last_y)
    assert all(_same(p[-1:], q) for p, q in zip(pools, standalone, strict=True))
