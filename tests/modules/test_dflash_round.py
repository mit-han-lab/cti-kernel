# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU reference vs independent FR composition and index-free extreme arithmetic."""

import sys

import pytest
import torch

from ctikernels.speculative import (
    ALL_VARIANTS,
    RoundVariant,
    commit_block,
    compile_count,
    fixed_query_addresses,
    prepare_block,
    prepare_round,
    warm_up,
)
from ctikernels.speculative._round.op import A_NAMES, C_NAMES, P_NAMES, R_NAMES

from .._dflash_round_reference import (
    ANY_QS,
    BATCHES,
    MODES,
    QS,
    assert_equal,
    clone,
    commit,
    fixture,
    fr_commit,
    fr_prepare,
    prepare,
    round_call,
    semantic_cases,
)
from ..support.child import run_child


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("mode", MODES)
def test_reference_fr_matrix(q, mode):
    for turn in range(max(q, 7)):
        b = 1 + turn % 4
        t = fixture(b, q, 67, mode=mode, s=(0, 1, 3)[turn % 3], turn=turn, optional=turn % 2 == 0)
        # Stops at each i, every a, boundaries/done combinations, varied budgets.
        f = t["frontier_table"][t["request_indices"].long()]
        budgets = (-3, 0, 1, q - 1, q, q + 1, 10**9)
        t["output_ends"].copy_(f + 1 + budgets[turn % 7])
        if turn % 4 == 0:
            t["token_table"][t["request_indices"].long(), f] = 100 + turn * 100
        expected = clone(t)
        fr_prepare(expected)
        fr_commit(expected)
        round_call(t)
        assert_equal(t, expected, fr=True)
        eligible = ~t["round_done"] & (t["output_ends"].clamp(max=67) - 1 - f >= 1)
        assert (t["committed"][eligible] >= 1).all()


@pytest.mark.parametrize("m", (64, 4096, 40960, 131072, 262144))
def test_capacity_axis(m):
    t = fixture(4, 8, m)
    e = clone(t)
    fr_prepare(e)
    fr_commit(e)
    round_call(t)
    assert_equal(t, e, fr=True)


def test_greedy_prefilled_bitwise_and_int32_narrowing():
    for q in QS:
        for a in range(q):
            t = fixture(4, q, 67, turn=a)
            t["target_ids"].add_(2**32)
            e = clone(t)
            e["tokens"].copy_(e["target_ids"].view(4, q))
            e["target_ids"] = None
            round_call(t)
            round_call(e)
            e["target_ids"] = t["target_ids"]
            assert_equal(t, e)


def wrap(x):
    return (x + 2**63) % 2**64 - 2**63


def check_extreme(t):
    """Index-free Python integer expectations independent of the provider."""
    q, m = t["block_size"], t["max_seq_len"]
    original = clone(t)
    prepare(t)
    prefix = 0
    for b, r in enumerate(t["request_indices"].tolist()):
        f = int(original["frontier_table"][r])
        d = bool(original["done_table"][r])
        for i in range(q):
            p = wrap(f + i)
            assert int(t["positions"][b * q + i]) == p
            assert int(t["rope_positions"][b * q + i]) == min(p, m - 1)
            assert int(t["verify_rope_positions"][b * q + i]) == wrap(
                min(p, m - 1) + int(t["rope_offsets"][b])
            )
        assert int(t["cursors"][b]) == wrap(f - int(t["prompt_lengths"][b]) + 1)
        assert int(t["input_ids"][b * q]) == int(original["token_table"][r, min(max(f, 0), m - 1)])
        dummy = d or not 0 <= f < m
        length = q if dummy else f + q
        g = t["guard_start"] + b * q
        for i in range(q):
            expected = (
                g + i
                if dummy
                else int(original["cache_index_table"][r, f + i])
                if f + i < m
                else g + min(f + i - m, q - 1)
            )
            assert int(t["cache_write_indices"][b * q + i]) == expected
        for j in range(min(length, max(0, t["kv_indices"].numel() - prefix))):
            expected = (
                g + j
                if dummy
                else int(original["cache_index_table"][r, j])
                if j < m
                else g + min(j - m, q - 1)
            )
            assert int(t["kv_indices"][prefix + j]) == expected
        prefix += length
        assert int(t["kv_indptr"][b + 1]) == prefix
    commit(t)
    expected_table = original["token_table"].clone()
    for b, r in enumerate(t["request_indices"].tolist()):
        f = int(original["frontier_table"][r])
        a = int(original["accepted"][b])
        d = bool(original["done_table"][r])
        v, fin = 0, True
        if not d and 0 <= f < m:
            e = min(int(t["output_ends"][b]), m)
            budget = max(0, min(e - f - 1, q))  # Python arbitrary integer subtraction.
            v = min(budget, a + 1)
            for i in range(max(0, v)):
                if (
                    int(t["ignore_eos"][b]) == 0
                    and int(t["tokens"][b, i]) in t["stop_token_ids"].tolist()
                ):
                    v = i + 1
                    break
            fin = f + v + 1 >= e or any(
                int(t["tokens"][b, i]) in t["stop_token_ids"].tolist()
                and int(t["ignore_eos"][b]) == 0
                for i in range(max(0, v))
            )
        assert int(t["committed"][b]) == v
        assert int(t["record_done"][b]) == fin
        assert int(t["frontier_table"][r]) == f + v
        assert int(t["accepted"][b]) == (-1 if d else a)
        for i in range(max(0, v)):
            expected_table[r, f + 1 + i] = original["tokens"][b, i]
    assert torch.equal(t["token_table"], expected_table)
    assert torch.equal(t["tokens"], original["tokens"])
    assert torch.equal(t["cache_index_table"], original["cache_index_table"])


def extreme_cases():
    for f in (-(2**63), -1, 67, 2**33 + 5, 2**63 - 1):
        for sign in (-1, 1):
            t = fixture(4, 8, 67, mode="sampled", s=0)
            t["frontier_table"][t["request_indices"].long()] = torch.tensor([f, 1, 65, f])
            t["done_table"][7] = True
            for n in ("output_ends", "rope_offsets", "prompt_lengths"):
                t[n].fill_(-(2**63) if sign < 0 else 2**63 - 1)
            t["accepted"][:] = torch.tensor([-(2**31), -2, 8, 2**31 - 1])
            yield t


def test_extreme_and_short_capacity():
    for t in extreme_cases():
        check_extreme(t)
    for n in (0, 1, 7, 32):
        t = fixture(4, 8, 67, mode="sampled", s=0)
        t["kv_indices"] = t["kv_indices"][:n]
        check_extreme(t)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"window": 1}, "window"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": True}, "block_size"),
        ({"mask_token_id": 2**31}, "mask_token_id"),
        ({"max_seq_len": 0}, "max_seq_len"),
        ({"guard_start": -1}, "guard_start"),
        ({"rope_offsets": None}, "iff"),
        ({"cursors": None}, "iff"),
        ({"use_pdl": 1}, "use_pdl"),
    ],
)
def test_host_options(change, match):
    t = fixture()
    args = {n: t[n] for n in P_NAMES}
    args.update({n: t[n] for n in ("block_size", "max_seq_len", "guard_start", "mask_token_id")})
    args.update(change)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        prepare_block(**args)


@pytest.mark.parametrize(
    "problem",
    ("dtype", "shape", "inner", "vector", "alias", "strided_alias", "device", "grad", "coupling"),
)
def test_metadata_errors(problem):
    t = fixture()
    if problem == "dtype":
        t["positions"] = t["positions"].int()
    if problem == "shape":
        t["round_done"] = t["round_done"][:1]
    if problem == "inner":
        t["token_table"] = t["token_table"][:, ::2]
    if problem == "vector":
        t["positions"] = torch.zeros(64, dtype=torch.int64)[::2]
    if problem == "alias":
        t["positions"] = t["rope_positions"]
    if problem == "strided_alias":
        backing = torch.zeros(10, 512, dtype=torch.int32)
        t["token_table"] = backing[:, :128]
        t["input_ids"] = backing[9, :32]
    if problem == "device":
        t["positions"] = torch.empty(32, device="meta", dtype=torch.int64)
    if problem == "grad":
        t["positions"] = torch.zeros(32, requires_grad=True)
    if problem == "coupling":
        t["proposals"] = None
    with torch.no_grad(), pytest.raises(ValueError):
        (commit if problem == "coupling" else prepare)(t)


def test_disjoint_record_views_and_table_row_counts():
    t = fixture()
    words = torch.zeros(256, dtype=torch.int32)
    t["start_frontiers"] = words[:8].view(torch.int64)
    t["accepted"], t["committed"], t["record_done"] = words[8:12], words[12:16], words[16:20]
    round_call(t)
    before = compile_count()
    warm_up(8, 67, 0, device="cpu")
    assert compile_count() == before


def test_import_without_numpy_or_cuda_provider():
    code = """
import sys
import torch
sys.modules['numpy'] = None
from ctikernels.speculative import prepare_block, commit_block, warm_up
warm_up(8,67,0,device='cpu')
assert 'cutlass' not in sys.modules and 'triton' not in sys.modules
"""
    result = run_child([sys.executable, "-c", code], timeout=600)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_explicit_stop_and_done_axes(mode):
    from .._dflash_round_reference import semantic_cases

    for t in semantic_cases(4, 8, mode):
        expected = clone(t)
        fr_prepare(expected)
        fr_commit(expected)
        round_call(t)
        assert_equal(t, expected, fr=True)


def test_commit_reads_snapshots_and_minimal_table_width():
    t = fixture(mode="sampled", s=0)
    prepare(t)
    expected = clone(t)
    # Simulate live row tables changing after P: C must consume only saved state.
    t["frontier_table"].fill_(2**63 - 1)
    t["done_table"].fill_(True)
    expected["frontier_table"].copy_(t["frontier_table"])
    expected["done_table"].copy_(t["done_table"])
    commit(expected)
    t["token_table"] = t["token_table"][:, :67]
    commit(t)
    expected["token_table"] = expected["token_table"][:, :67]
    assert_equal(t, expected)


@pytest.mark.parametrize(
    "field",
    (
        "tokens",
        "accepted",
        "output_ends",
        "stop_token_ids",
        "token_table",
        "frontier_table",
        "done_table",
        "committed",
        "record_done",
        "k2_lengths",
    ),
)
def test_commit_rejects_wrong_dtype_and_alias(field):
    t = fixture()
    t[field] = t[field].float()
    with pytest.raises(ValueError, match="dtype"):
        commit(t)
    t = fixture()
    t["committed"] = t["accepted"]
    with pytest.raises(ValueError, match="overlap"):
        commit(t)


@pytest.mark.parametrize("q", [pytest.param(1, id="q1"), 8])
def test_warmup_key_excludes_geometry_and_covers_optional_cross_product(monkeypatch, q):
    from ctikernels.speculative._round import cute

    monkeypatch.setattr(cute, "version", lambda _: "4.8.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("Properties", (), {"multi_processor_count": 170})(),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cute, "_COMPILED", {})
    calls = []

    def compile_stub(kind, q, flags, arch):
        calls.append((kind, q, flags, arch))

        def small(tensors, *args):
            assert all(t.numel() > 0 for t in tensors)
            assert tensors[0].numel() <= 4

        def generic(tensors, *args):
            assert all(t.numel() > 0 for t in tensors)
            assert tensors[0].numel() > 4

        return small, generic

    monkeypatch.setattr(cute, "compile_variant", compile_stub)
    provider = cute.Provider(torch.device("cuda:0"))
    # fixed_query_addresses is Triton: the CuTe provider never sees it.
    variants = tuple(v for v in ALL_VARIANTS if v.kind != "fixed_query_addresses")
    provider.warm_up(q, 0, False, variants)
    assert len(calls) == 22
    for b in BATCHES:
        for mode in MODES:
            t = fixture(b, q, 64 + b, mode=mode, s=0)
            provider.prepare(t)
            provider.prepare_round(t)
            provider.commit(t)
    assert len(calls) == 22
    provider.warm_up(q, 3, False, variants)
    assert len(calls) == 28
    provider.warm_up(q, 0, True, variants)
    assert len(calls) == 50


def test_disjoint_storage_in_table_row_gaps():
    t = fixture(mode="sampled")
    backing = torch.zeros(10, 256, dtype=torch.int32)
    backing[:, :128].copy_(t["token_table"])
    t["token_table"] = backing[:, :128]
    t["input_ids"] = backing[0, 128:160]
    t["tokens"] = backing[:4, 192:200]
    t["tokens"].fill_(123)
    expected = clone(t)
    round_call(t)
    round_call(expected)
    assert_equal(t, expected)


def long_round_cases(q, mode):
    for boundary in sorted({0, 31, 32, min(63, q - 1), min(64, q - 1), q - 1}):
        t = fixture(4, q, q + 70, mode=mode, s=3)
        rows = t["request_indices"].long()
        t["frontier_table"][rows] = 1
        t["done_table"].zero_()
        t["accepted"].fill_(boundary)
        if t["proposals"] is not None:
            t["proposals"].copy_(t["tokens"][:, :-1])
            if boundary < q - 1:
                t["proposals"][:, boundary] += 1
        # First stop wins, including when it is in a later warp-sized chunk.
        t["stop_token_ids"][:] = t["tokens"][0, boundary]
        t["stop_token_ids"][1] = t["tokens"][0, min(boundary + 1, q - 1)]
        t["ignore_eos"].zero_()
        t["ignore_eos"][1] = 1
        t["output_ends"][2] = boundary + 2  # Budget truncates before the bonus.
        t["done_table"][rows[3]] = True
        earlier = clone(t)
        earlier["stop_token_ids"][1] = earlier["tokens"][0, max(0, boundary - 32)]
        earlier["stop_token_ids"][2] = earlier["tokens"][1, boundary]
        yield t
        yield earlier
    # Full Q writes, including the final lane; no early stop/mismatch.
    t = fixture(4, q, q + 70, mode=mode, s=0)
    t["frontier_table"].fill_(1)
    t["done_table"].zero_()
    t["accepted"].fill_(q - 1)
    if t["proposals"] is not None:
        t["proposals"].copy_(t["tokens"][:, :-1])
    yield t
    # Prepare guard filling when the Q positions run past table capacity.
    t = fixture(4, q, 67, mode=mode, s=0)
    t["frontier_table"].fill_(66)
    yield t


@pytest.mark.parametrize("q", [33, 64, 65, 128, 129])
@pytest.mark.parametrize("mode", MODES)
def test_long_round_reference(q, mode):
    for t in long_round_cases(q, mode):
        expected = clone(t)
        fr_prepare(expected)
        fr_commit(expected)
        round_call(t)
        assert_equal(t, expected, fr=True)


def test_round_q_and_csr_bounds():
    from ctikernels.speculative._round.op import _geometry

    with pytest.raises(ValueError, match="block_size"):
        _geometry(1, 0, 1)
    _geometry(1, 2**31 - 2, 1)
    with pytest.raises(ValueError, match=r"B\*\(max_seq_len \+ block_size\).*int32 CSR"):
        _geometry(1, 2**31 - 1, 1)


@pytest.mark.parametrize("mode", MODES)
def test_round_q1_bonus_only(mode):
    for turn in range(7):
        t = fixture(4, 1, 67, mode=mode, turn=turn)
        t["frontier_table"].clamp_(max=66)
        if t["proposals"] is not None:
            t["proposals"] = torch.empty_strided((4, 0), (17, 9), dtype=torch.int32)
        expected = clone(t)
        fr_prepare(expected)
        fr_commit(expected)
        round_call(t)
        assert_equal(t, expected, fr=True)
        assert torch.equal(t["accepted"], torch.where(t["round_done"], -1, 0))
        assert ((t["committed"] == 0) | (t["committed"] == 1)).all()


def split_cases():
    for q in QS:
        for mode in MODES:
            yield from semantic_cases(4, q, mode, full=False)
    yield from extreme_cases()
    for n in (0, 1, 7, 32):
        t = fixture(4, 8, 67, mode="sampled", s=0)
        t["kv_indices"] = t["kv_indices"][:n]
        yield t
    for q in (1, 33, 65):
        yield from long_round_cases(q, "greedy") if q > 32 else [fixture(4, q, 67)]


def test_split_prepare_equals_prepare_block():
    """SP7: prepare_round + fixed_query_addresses == prepare_block, bitwise."""
    for t in split_cases():
        fused, split = clone(t), clone(t)
        prepare(fused)
        prepare(split, split=True)
        assert_equal(split, fused)


def test_prepare_round_writes_no_addressing_and_addresses_read_saved_state():
    t = fixture(4, 8, 67)
    poison = {n: t[n].clone() for n in ("cache_write_indices", "kv_indptr", "kv_indices")}
    for n in poison:
        t[n].fill_(-7)
    prepare_round(
        **{n: t[n] for n in R_NAMES},
        **{n: t[n] for n in ("block_size", "max_seq_len", "mask_token_id")},
    )
    for n in poison:
        assert (t[n] == -7).all(), n
    # Live tables changed after P: addresses must follow the saved state only.
    expected = clone(t)
    fixed_query_addresses(
        **{n: expected[n] for n in A_NAMES},
        **{n: expected[n] for n in ("block_size", "max_seq_len", "guard_start")},
    )
    t["frontier_table"].fill_(3)
    t["done_table"].fill_(True)
    fixed_query_addresses(
        **{n: t[n] for n in A_NAMES},
        **{n: t[n] for n in ("block_size", "max_seq_len", "guard_start")},
    )
    for n in poison:
        assert torch.equal(t[n], expected[n]), n


@pytest.mark.parametrize(
    "op,change,match",
    [
        ("round", {"block_size": 0}, "block_size"),
        ("round", {"mask_token_id": 2**31}, "mask_token_id"),
        ("round", {"rope_offsets": None}, "iff"),
        ("round", {"cursors": None}, "iff"),
        ("round", {"use_pdl": 1}, "use_pdl"),
        ("round", {"positions": "int32"}, "dtype"),
        ("round", {"input_ids": "alias"}, "overlap"),
        ("addresses", {"guard_start": -1}, "guard_start"),
        ("addresses", {"guard_start": 2**31 - 32}, "guard_start"),
        ("addresses", {"max_seq_len": 0}, "max_seq_len"),
        ("addresses", {"cache_index_table": "narrow"}, "max_seq_len"),
        ("addresses", {"kv_indptr": "short"}, "shape"),
        ("addresses", {"cache_write_indices": "alias"}, "overlap"),
        ("addresses", {"use_pdl": 1}, "use_pdl"),
    ],
)
def test_split_host_options(op, change, match):
    t = fixture()
    names, scalars = (
        (R_NAMES, ("block_size", "max_seq_len", "mask_token_id"))
        if op == "round"
        else (A_NAMES, ("block_size", "max_seq_len", "guard_start"))
    )
    args = {n: t[n] for n in (*names, *scalars)}
    for n, kind in change.items():
        value = kind
        if kind == "int32":
            value = args[n].int()
        elif kind == "alias":
            value = args["request_indices"].repeat(8)[: args[n].numel()]
            args["request_indices"] = value[:4]
        elif kind == "narrow":
            value = args[n][:, :60]
        elif kind == "short":
            value = args[n][:4]
        args[n] = value
    with pytest.raises(ValueError, match=match):
        (prepare_round if op == "round" else fixed_query_addresses)(**args)


ENGINE_VARIANTS = (
    RoundVariant("prepare_block", rope=True, cursors=True),
    RoundVariant("prepare_block"),
    RoundVariant("commit_block", mode="greedy"),
    RoundVariant("commit_block", mode="sampled"),
)


def test_warm_up_returns_declared_set_in_canonical_order():
    assert len(ALL_VARIANTS) == len(set(ALL_VARIANTS)) == 23
    assert warm_up(8, 67, 0, device="cpu") == ALL_VARIANTS
    declared = warm_up(8, 67, 0, device="cpu", variants=ENGINE_VARIANTS[::-1] * 2)
    assert declared == tuple(v for v in ALL_VARIANTS if v in ENGINE_VARIANTS)
    assert warm_up(8, 67, 0, device="cpu", variants=()) == ()


@pytest.mark.parametrize(
    "variant",
    (
        ("commit_block", False, False, False, "greedy"),
        RoundVariant("prepare"),
        RoundVariant("prepare_block", mode="greedy"),
        RoundVariant("prepare_round", rope=1),
        RoundVariant("fixed_query_addresses", k2=True),
        RoundVariant("commit_block"),
        RoundVariant("commit_block", mode="fused"),
        RoundVariant("commit_block", rope=True, mode="greedy"),
        RoundVariant("commit_block", cursors=True, mode="sampled"),
    ),
)
def test_warm_up_rejects_invalid_variants(variant):
    with pytest.raises(ValueError, match="[Vv]ariant"):
        warm_up(8, 67, 0, device="cpu", variants=(variant,))


def test_declared_warm_up_compiles_only_the_declared_set(monkeypatch):
    from contextlib import nullcontext

    from ctikernels.speculative._round import cute

    monkeypatch.setattr(cute, "version", lambda _: "4.8.0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("Properties", (), {"multi_processor_count": 170})(),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cute, "_COMPILED", {})
    calls = []

    def compile_stub(kind, q, flags, arch):
        calls.append((kind, flags))

        def launch(tensors, *args):
            pass

        return launch, launch

    monkeypatch.setattr(cute, "compile_variant", compile_stub)
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    provider = cute.Provider(torch.device("cuda:0"))
    split = (RoundVariant("prepare_round", rope=True, cursors=True),)
    provider.warm_up(8, 2, False, ENGINE_VARIANTS + split)
    assert sorted(calls, key=repr) == sorted(
        [
            ("prepare", (True, False, True, False, True)),
            ("prepare", (False, False, False, False, True)),
            ("commit", (2, "greedy", False, False, True)),
            ("commit", (2, "sampled", False, False, True)),
            ("round", (True, False, True, False, True)),
        ],
        key=repr,
    )
    t = fixture(4, 8, 67, mode="greedy", s=2)
    t["k2_slots"] = t["k2_lengths"] = None
    capturing[0] = True
    provider.prepare(t)
    provider.prepare_round(t)
    provider.commit(t)
    assert len(calls) == 5
    t["target_ids"] = None
    with pytest.raises(
        RuntimeError,
        match=r"RoundVariant\('commit_block', mode='prefilled', k2=False\) at block_size=8, "
        r"num_stop_tokens=2, use_pdl=False was not declared",
    ):
        provider.commit(t)
    t["prompt_lengths"] = t["cursors"] = None
    with pytest.raises(
        RuntimeError,
        match=r"RoundVariant\('prepare_round', rope=True, k2=False, cursors=False\) at "
        r"block_size=8, use_pdl=False was not declared",
    ):
        provider.prepare_round(t)
    # Compiled by an eager call is not declared: capture still raises.
    capturing[0] = False
    provider.prepare_round(t)
    assert len(calls) == 6
    capturing[0] = True
    with pytest.raises(RuntimeError, match="'prepare_round'.*not declared"):
        provider.prepare_round(t)
    assert len(calls) == 6


class FakeTriton:
    """Stands in for the Triton kernel module: the CI CPU env has no Triton."""

    def __init__(self):
        self.warmed, self.launches, self.loaded = [], [], []
        self.device_caches = {0: ({}, None)}

    def warmup(self, *args, grid, **options):
        self.warmed.append((args, options))
        self.device_caches[0][0][len(self.warmed)] = None
        warmed, loaded = len(self.warmed), self.loaded

        class Compiled:
            def _init_handles(self):
                loaded.append(warmed)

        return Compiled()

    def __getitem__(self, grid):
        return lambda *args, **options: self.launches.append((grid, args, options))


def test_cuda_addresses_provider_is_triton(monkeypatch):
    import types
    from contextlib import nullcontext

    from ctikernels.speculative._round import cute, op, triton
    from ctikernels.speculative._round.reference import Provider as Reference

    assert isinstance(op.provider(torch.device("cpu")), Reference)
    kernel = FakeTriton()
    module = types.ModuleType(triton.__name__ + ".kernels")
    module.fixed_query_addresses = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(triton, "kernels", module, raising=False)
    monkeypatch.setattr(cute, "version", lambda _: "4.8.0")
    capability = [(8, 0)]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability[0])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: type("Properties", (), {"multi_processor_count": 170})(),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(cute, "_COMPILED", {})
    compiled = []
    monkeypatch.setattr(
        cute, "compile_variant", lambda kind, *_: compiled.append(kind) or (None, None)
    )
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    before = compile_count()
    provider = op._Cuda(torch.device("cuda:0"))
    provider.warm_up(8, 0, False, ALL_VARIANTS)
    # 22 CuTe variants; the addresses variant warms one Triton binary per stride width.
    assert len(compiled) == 22 and "addresses" not in compiled
    assert [args[11] for args, _ in kernel.warmed] == [1, 2**31]
    assert kernel.loaded == [1, 2]  # each warmed binary's module is loaded at warm-up
    assert all(o["Q"] == 8 and not o["launch_pdl"] and not o["WAIT"] for _, o in kernel.warmed)
    assert compile_count() == before + 22 + 2
    for b, m, capacity in ((1, 67, 100), (4, 32768, 2**31 + 5), (300, 4096, 0)):
        t = fixture(b, 8, m, s=0)
        t["kv_indices"] = torch.zeros(min(capacity, 1), dtype=torch.int32).expand(capacity)
        capturing[0] = True
        provider.addresses(t)
        capturing[0] = False
        grid, args, options = kernel.launches[-1]
        tiles = -(-(m + 8) // 1024)
        assert grid == (min(b * tiles, 4 * 170),)
        # An empty kv_indices borrows a nonempty int32 output (capacity 0: never written).
        borrowed = {"kv_indices": t["cache_write_indices"]} if capacity == 0 else {}
        assert all(a is borrowed.get(n, t[n]) for a, n in zip(args[:7], A_NAMES, strict=True))
        assert args[7:] == (
            b,
            m,
            t["guard_start"],
            min(capacity, 2**31 - 1),
            args[3].stride(0),
            tiles,
        )
    t["use_pdl"] = True
    capturing[0] = True
    with pytest.raises(
        RuntimeError,
        match=r"RoundVariant\('fixed_query_addresses'\) at block_size=8, use_pdl=True was not "
        r"declared",
    ):
        provider.addresses(t)
    # The private no-wait mutant is never declared and may launch while capturing.
    provider._wait = False
    provider.addresses(t)
    assert kernel.launches[-1][2]["launch_pdl"] and not kernel.launches[-1][2]["WAIT"]
    capturing[0] = False
    provider._wait = True
    provider.warm_up(8, 0, True, (RoundVariant("fixed_query_addresses"),))
    capturing[0] = True
    provider.addresses(t)
    assert kernel.launches[-1][2]["launch_pdl"] and kernel.launches[-1][2]["WAIT"]
    assert len(compiled) == 22
    capturing[0] = False
    capability[0] = (7, 5)
    with pytest.raises(RuntimeError, match="cc >= 8.0"):
        op._Cuda(torch.device("cuda:1")).addresses(t)


# Cover each batch with three modes and several Q values, without a full product.
ANY_BATCH_CORE = [
    (b, ANY_QS[(i + j) % len(ANY_QS)] if b < 1024 else (1, 2, 4)[j], mode, (i + j) % 2 == 0)
    for i, b in enumerate((0, *BATCHES))
    for j, mode in enumerate(MODES)
]


@pytest.mark.parametrize("b,q,mode,optional", ANY_BATCH_CORE)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_any_batch_reference_fr(b, q, mode, optional, split):
    t = fixture(b, q, 67, mode=mode, optional=optional, turn=2)
    t["frontier_table"].clamp_(max=66)  # Q=1 includes F=M in the fixture pattern.
    expected = clone(t)
    fr_prepare(expected)
    fr_commit(expected)
    round_call(t, split)
    assert_equal(t, expected, fr=True)


OPS = (
    (prepare_block, P_NAMES, ("block_size", "max_seq_len", "guard_start", "mask_token_id")),
    (prepare_round, R_NAMES, ("block_size", "max_seq_len", "mask_token_id")),
    (fixed_query_addresses, A_NAMES, ("block_size", "max_seq_len", "guard_start")),
    (commit_block, C_NAMES, ("block_size", "max_seq_len")),
)


def invoke_op(spec, t):
    op, names, scalars = spec
    return op(**{n: t[n] for n in (*names, *scalars)})


def forbid_provider(_):
    raise AssertionError("empty batch reached provider")


@pytest.mark.parametrize("spec", OPS, ids=lambda spec: spec[0].__name__)
@pytest.mark.parametrize("optional", (False, True))
@pytest.mark.parametrize("mode", MODES)
def test_empty_batch_is_validated_noop(monkeypatch, spec, optional, mode):
    from ctikernels.speculative._round import op

    t = fixture(0, mode=mode, optional=optional)
    t["kv_indptr"].fill_(123)
    t["kv_indices"] = torch.full((17,), 456, dtype=torch.int32)
    before = clone(t)
    monkeypatch.setattr(op, "provider", forbid_provider)
    assert invoke_op(spec, t) is None
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(t[name], value), name
    t["block_size"] = 0
    with pytest.raises(ValueError, match="block_size"):
        invoke_op(spec, t)
    t["block_size"] = 8
    t["start_frontiers"] = torch.empty(0, dtype=torch.int32)
    with pytest.raises(ValueError, match="dtype"):
        invoke_op(spec, t)


def test_warm_up_removed_max_batch():
    assert warm_up(8, 67, 0, device="cpu") == ALL_VARIANTS
    with pytest.raises((TypeError, ValueError)):
        warm_up(4, 8, 67, 0, device="cpu")


def test_any_batch_geometry_bounds():
    from ctikernels.speculative._round.op import _geometry

    limit = 2**31 - 1
    _geometry(0, limit, limit)
    _geometry(1, 1, limit - 1)
    _geometry(limit // 2, 1, 1)
    for b, q, m in ((1, 1, limit), (limit // 2 + 1, 1, 1), (limit, 1, 1)):
        with pytest.raises(ValueError, match="int32 CSR"):
            _geometry(b, q, m)
    for b in (-1, 2**31, True):
        with pytest.raises(ValueError, match="batch size"):
            _geometry(b, 1, 1)


@pytest.mark.parametrize("spec", OPS, ids=lambda spec: spec[0].__name__)
def test_op_int32_bounds_without_large_allocations(spec):
    t = fixture(1, 1, 67)
    # The exact CSR limit passes geometry and reaches the table-width check.
    # Unit-inner-stride tables cannot represent this width without huge storage.
    t["max_seq_len"] = 2**31 - 2
    with pytest.raises(ValueError, match="cover.*max_seq_len"):
        invoke_op(spec, t)
    t["max_seq_len"] += 1
    with pytest.raises(ValueError, match="int32 CSR"):
        invoke_op(spec, t)
    t["max_seq_len"] = 1
    # Expanded views cost one element; batch validation precedes layout checks.
    t["request_indices"] = torch.zeros(1, dtype=torch.int32).expand(2**31)
    with pytest.raises(ValueError, match="batch size"):
        invoke_op(spec, t)
    t["request_indices"] = t["request_indices"][: 2**31 - 1]
    with pytest.raises(ValueError, match="int32 CSR"):
        invoke_op(spec, t)


@pytest.mark.parametrize("spec", (OPS[0], OPS[2]), ids=lambda spec: spec[0].__name__)
@pytest.mark.parametrize("b", (0, 1, 8))
def test_guard_int32_bounds(spec, b):
    t = fixture(b)
    t["guard_start"] = 2**31 - 1 - b * t["block_size"]
    invoke_op(spec, t)
    t["guard_start"] += 1
    with pytest.raises(ValueError, match="guard_start"):
        invoke_op(spec, t)


@pytest.mark.parametrize("b", BATCHES)
def test_fixture_request_rows_and_padding(b):
    t = fixture(b)
    rows = t["request_indices"].tolist()
    assert len(set(rows)) == b
    if b <= 4:
        assert rows == [5, 0, 3, 7][:b]
        assert t["frontier_table"].numel() == 9
        assert t["token_table"].shape[0] == 10
        assert t["cache_index_table"].shape[0] == 11
    else:
        assert 0 in rows and b + 4 in rows
        assert rows != sorted(rows)
        assert t["frontier_table"].numel() == b + 5
        assert t["token_table"].shape[0] == b + 6
        assert t["cache_index_table"].shape[0] == b + 7
        done = t["done_table"][t["request_indices"].long()]
        assert done.any() and not done.all()
