# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""R5 exact arithmetic, allocation and graph qualification; no provider fallbacks."""

from itertools import product

import pytest
import torch

from ctikernels.speculative import RoundVariant, compile_count, warm_up

from .._dflash_round_reference import (
    ANY_QS,
    BATCHES,
    MODES,
    QS,
    SENTINEL,
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
from ..modules.test_dflash_round import (
    OPS,
    check_extreme,
    extreme_cases,
    forbid_provider,
    invoke_op,
)
from ._exhaustive import ci_cases

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def stage(dst, src):
    for name, value in src.items():
        if isinstance(value, torch.Tensor):
            dst[name].copy_(value)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def check_case(t, *, fr=True, split=False):
    actual, expected = clone(t, "cuda"), clone(t)
    round_call(expected)
    round_call(actual, split)
    assert_equal(actual, expected)
    if fr:
        oracle = clone(t)
        fr_prepare(oracle)
        fr_commit(oracle)
        assert_equal(actual, oracle, fr=True)
    eligible = ~expected["round_done"] & (
        expected["output_ends"].clamp(max=t["max_seq_len"]) - 1 - expected["start_frontiers"] >= 1
    )
    if fr:
        assert (expected["committed"][eligible] >= 1).all()


@pytest.mark.parametrize(
    "b,q,mode",
    ci_cases(
        range(1, 5),
        QS,
        MODES,
        core=[(4, 8, "greedy"), (1, 16, "sampled"), (3, 5, "prefilled")],
    ),
)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_semantic_matrix(b, q, mode, split):
    for t in semantic_cases(b, q, mode):
        check_case(t, split=split)


@pytest.mark.parametrize("m", (64, 4096, 40960, 131072, 262144, 67))
@pytest.mark.parametrize("q", (8, 16))
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_capacity_widths_and_boundary_frontiers(m, q, split):
    for turn in range(7):
        t = fixture(4, q, m, turn=turn, mode=MODES[turn % 3], s=(0, 1, 3)[turn % 3])
        check_case(t, split=split)
    # Minimal legal width and row-strided storage; C must not write scratch.
    t = fixture(4, q, m)
    for name in ("token_table", "cache_index_table"):
        t[name] = t[name][:, :m]
    check_case(t, fr=False, split=split)


@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_extreme_values_and_short_kv_capacity(split):
    for t in extreme_cases():
        check_extreme(clone(t))
        check_case(t, fr=False, split=split)
    for n in (0, 1, 7, 32):
        t = fixture(mode="sampled", s=0)
        t["kv_indices"] = t["kv_indices"][:n]
        check_case(t, fr=False, split=split)
    # Every hostile acceptance value on a valid, live row.
    for a, end in product((-(2**31), -2, 8, 2**31 - 1), (-(2**63), 2**63 - 1)):
        t = fixture(mode="sampled", s=0)
        t["frontier_table"].fill_(1)
        t["accepted"].fill_(a)
        t["output_ends"].fill_(end)
        check_extreme(clone(t))
        check_case(t, fr=False, split=split)


@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_prefilled_matches_greedy_bitwise(split):
    for q in QS:
        t = fixture(4, q, 67)
        t["target_ids"].add_(2**32)
        actual = clone(t, "cuda")
        prefilled = clone(t, "cuda")
        prefilled["target_ids"] = None
        round_call(actual, split)
        round_call(prefilled, split)
        prefilled["target_ids"] = actual["target_ids"]
        assert_equal(actual, prefilled)


@pytest.mark.parametrize(
    "b,q,mode",
    ci_cases(
        range(1, 5),
        QS,
        MODES,
        core=[(1, 16, "sampled")],
    ),
)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_graph_random_rewrites(b, q, mode, split):
    warm_up(q, 67, 3, device="cuda")
    initial = fixture(b, q, 67, mode=mode)
    actual, eager = clone(initial, "cuda"), clone(initial, "cuda")
    graph = capture(lambda: round_call(actual, split))
    gen = torch.Generator().manual_seed(970 + b + q)
    for turn in range(100):
        t = fixture(b, q, 67, mode=mode, turn=turn)
        rows = t["request_indices"].long()
        t["frontier_table"][rows] = torch.randint(0, 67, (b,), generator=gen)
        t["done_table"][rows] = torch.randint(0, 2, (b,), generator=gen).bool()
        t["output_ends"][:] = torch.randint(1, 100, (b,), generator=gen)
        t["rope_offsets"][:] = torch.randint(-100, 100, (b,), generator=gen)
        t["prompt_lengths"][:] = torch.randint(0, 67, (b,), generator=gen)
        t["cache_index_table"][:8] += turn * 1000
        stage(actual, t)
        stage(eager, t)
        oracle = clone(t)
        fr_prepare(oracle)
        fr_commit(oracle)
        graph.replay()
        round_call(eager, split)
        assert_equal(actual, oracle, fr=True)
        assert_equal(actual, eager)


def prefix(t, b):
    q = t["block_size"]
    result = dict(t)
    for n, v in t.items():
        if not isinstance(v, torch.Tensor) or n.endswith("_table") or n == "stop_token_ids":
            continue
        if n == "kv_indices":
            result[n] = v[: b * (t["max_seq_len"] + q)]
        elif n == "kv_indptr":
            result[n] = v[: b + 1]
        elif v.ndim == 1 and v.numel() == t["request_indices"].numel() * q:
            result[n] = v[: b * q]
        else:
            result[n] = v[:b]
    return result


@pytest.mark.parametrize("batches", ((4, 1, 3), (3, 5, 64)))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_shared_prefix_graphs_preserve_previous_tails(mode, split, batches):
    q = 8
    warm_up(q, 67, 3, device="cuda")
    initial = fixture(max(batches), q, 67, mode=mode)
    actual = clone(initial, "cuda")
    views = {b: prefix(actual, b) for b in batches}
    graphs = {b: capture(lambda t=t: round_call(t, split)) for b, t in views.items()}
    stage(actual, initial)
    expected = clone(initial)
    for turn in range(12):
        for b in batches:
            # Change inputs only. Outputs/tails must retain the preceding round.
            fresh = (
                fixture(b, q, 67, turn=turn, mode=mode)
                if max(batches) <= 4
                else prefix(fixture(max(batches), q, 67, turn=turn, mode=mode), b)
            )
            for n in (
                "frontier_table",
                "done_table",
                "token_table",
                "cache_index_table",
                "output_ends",
                "ignore_eos",
                "proposals",
                "target_ids",
            ):
                if fresh[n] is not None:
                    views[b][n].copy_(fresh[n])
                    prefix(expected, b)[n].copy_(fresh[n])
            graphs[b].replay()
            round_call(prefix(expected, b))
            # Full buffers, including previous larger graph's CSR tail.
            for n, v in actual.items():
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v.cpu(), expected[n]), n


@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_identical_graph_inputs_1000_replays(split):
    t = fixture(4, 8, 67)
    actual = clone(t, "cuda")
    seed = clone(t, "cuda")
    warm_up(8, 67, 3, device="cuda")

    def run():
        stage(actual, seed)
        round_call(actual, split)

    graph = capture(run)
    expected = clone(t)
    round_call(expected)
    for _ in range(1000):
        graph.replay()
        assert_equal(actual, expected)


@pytest.mark.parametrize("use_pdl", (False, pytest.param(True, marks=pytest.mark.exhaustive)))
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_warmup_all_variants_no_allocation_or_sync(use_pdl, split):
    warm_up(8, 262144, 0, device="cuda", use_pdl=use_pdl)
    warm_up(8, 262144, 3, device="cuda", use_pdl=use_pdl)
    count = compile_count()
    for b, mode, rope, k2, cursors, s in product(
        range(1, 5), MODES, (False, True), (False, True), (False, True), (0, 3)
    ):
        t = fixture(b, 8, 67 if b % 2 else 64, mode=mode, s=s)
        if not rope:
            t["rope_offsets"] = t["verify_rope_positions"] = None
        if not k2:
            t["k2_slots"] = t["k2_lengths"] = None
        if not cursors:
            t["prompt_lengths"] = t["cursors"] = None
        t["use_pdl"] = use_pdl
        actual = clone(t, "cuda")
        # First eager call after warm_up is included in allocation audit.
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        old = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            round_call(actual, split)
        finally:
            torch.cuda.set_sync_debug_mode(old)
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        capture(lambda actual=actual: round_call(actual, split)).replay()
    assert compile_count() == count


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param(mode, marks=() if mode in ("sampled", "prefilled") else pytest.mark.exhaustive)
        for mode in MODES
    ],
)
@pytest.mark.parametrize("captured", (pytest.param(False, marks=pytest.mark.exhaustive), True))
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
@pytest.mark.parametrize("b", (4, 8))
def test_d2_delayed_producers_detect_missing_wait(mode, captured, split, b, *, detect_mutants=True):
    from ctikernels.speculative._round.op import provider

    from ._dflash_round_cute import compile_copy

    initial = fixture(b, 8, 67, mode=mode, s=0)
    rows = initial["request_indices"].long()
    initial["frontier_table"][rows] = 1
    initial["done_table"].zero_()
    initial["output_ends"].fill_(60)
    initial["output_ends"][0] = 3
    actual = clone(initial, "cuda")
    producer_seed = clone(initial, "cuda")
    verified = clone(initial)
    verified["tokens"].add_(300)
    if verified["target_ids"] is not None:
        verified["target_ids"].add_(300)
    verified["accepted"].fill_(3)
    verify_seed = clone(verified, "cuda")
    # Ordinary forward publishes a row outside this round between C0 and P1.
    ordinary_src = (
        torch.tensor([23], dtype=torch.int64, device="cuda"),
        torch.tensor([False], dtype=torch.bool, device="cuda"),
    )
    ordinary_row = (
        2 if b <= 4 else next(r for r in range(initial["frontier_table"].numel()) if r not in rows)
    )
    ordinary_dst = tuple(
        actual[n][ordinary_row : ordinary_row + 1] for n in ("frontier_table", "done_table")
    )
    first_names = [n for n, v in initial.items() if isinstance(v, torch.Tensor) and v.numel()]
    verify_names = ["tokens", "accepted"] + (["target_ids"] if mode == "greedy" else [])
    # Next round producer rewrites args, never restores C0's device frontier/done.
    next_names = ["output_ends", "rope_offsets", "prompt_lengths"]
    pairs = [
        (tuple(actual[n] for n in names), tuple(seed[n] for n in names))
        for names, seed in (
            (first_names, producer_seed),
            (verify_names, verify_seed),
            (next_names, producer_seed),
        )
    ]
    copies = [compile_copy(*pair) for pair in pairs]
    ordinary = compile_copy(ordinary_dst, ordinary_src)
    warm_up(8, 67, 0, device="cuda", use_pdl=True)
    actual["use_pdl"] = True
    saved = [clone(initial, "cuda"), clone(initial, "cuda")]
    snapshots = [n for n, v in initial.items() if isinstance(v, torch.Tensor)]
    op = provider(actual["request_indices"].device)

    def sequence(fault=None):
        copies[0](*pairs[0])
        for turn in range(2):
            if turn:
                ordinary(ordinary_dst, ordinary_src)
                copies[2](*pairs[2])
            op._wait = fault != "prepare"
            prepare(actual, split)
            copies[1](*pairs[1])
            op._wait = fault != "commit"
            commit(actual)
            op._wait = True
            for n in snapshots:
                saved[turn][n].copy_(actual[n])

    expected = clone(initial)
    expected_rounds = []
    for turn in range(2):
        if turn:
            expected["frontier_table"][ordinary_row] = 23
            expected["done_table"][ordinary_row] = False
        round_call_inputs = expected
        prepare(round_call_inputs)
        for n in verify_names:
            expected[n].copy_(verified[n])
        commit(expected)
        expected_rounds.append(clone(expected))
    assert expected_rounds[0]["record_done"][0] == 1
    assert expected_rounds[1]["round_done"][0]
    assert expected_rounds[1]["accepted"][0] == -1
    try:

        def eager(fault=None):
            # Hold the stream until the host has queued the whole sequence, so each
            # PDL producer/consumer pair overlaps on the GPU despite Python overhead.
            torch.cuda._sleep(20_000_000)
            sequence(fault)

        good = capture(sequence).replay if captured else eager
        for _ in range(3):
            good()
            for actual_round, wanted in zip(saved, expected_rounds, strict=True):
                assert_equal(actual_round, wanted)
        if not detect_mutants:
            return
        for fault in ("prepare", "commit"):
            # Compile the private mutation outside capture first.
            sequence(fault)
            bad = (
                capture(lambda fault=fault: sequence(fault)).replay
                if captured
                else lambda fault=fault: eager(fault)
            )
            detected = False
            for _ in range(32):
                # Deliberately stale inputs ensure early reads differ from producer data.
                actual["frontier_table"].fill_(40)
                actual["done_table"].fill_(True)
                actual["tokens"].fill_(-100)
                actual["accepted"].zero_()
                if actual["target_ids"] is not None:
                    actual["target_ids"].fill_(-200)
                bad()
                try:
                    for actual_round, wanted in zip(saved, expected_rounds, strict=True):
                        assert_equal(actual_round, wanted)
                except AssertionError:
                    detected = True
                    break
            assert detected, f"PDL {fault} missing-wait mutant escaped 32 attempts"
    finally:
        op._wait = True


@pytest.mark.parametrize("mutant", (False, True))
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
@pytest.mark.parametrize("b", (4, 8))
def test_cross_stream_row_ownership(mutant, split, b):
    from ._dflash_round_cute import compile_checker

    initial = fixture(b, 8, 67, s=0)
    actual, seed = clone(initial, "cuda"), clone(initial, "cuda")
    expected = clone(initial)
    round_call(expected)
    table, cache = actual["token_table"], actual["cache_index_table"]
    last = torch.cat((table.flatten(), cache.flatten()))
    errors = torch.zeros_like(last)
    owned = torch.zeros(max(table.shape[0], cache.shape[0]), dtype=torch.int32)
    owned[initial["request_indices"].long()] = 1
    mutation_row = int((owned[: table.shape[0]] == 0).nonzero()[0])
    owned = owned.to("cuda")
    checker = compile_checker(table, cache, last, errors, owned)
    mutation = compile_checker(
        table, cache, last, errors, owned, mutate=True, mutation_row=mutation_row
    )
    warm_up(8, 67, 0, device="cuda")
    # Restore ONLY round-owned rows and arguments. Prep owns the other rows.
    rows = initial["request_indices"].tolist()

    def run():
        for name in ("token_table", "cache_index_table"):
            for row in rows:
                actual[name][row].copy_(seed[name][row])
        for name in ("frontier_table", "done_table", "accepted"):
            actual[name].copy_(seed[name])
        actual["kv_indices"].fill_(SENTINEL)
        round_call(actual, split)
        if mutant:
            mutation(table, cache, last, errors, owned)

    graph = capture(run)
    # Capture/warmup may inject faults: establish a clean checker baseline.
    last.copy_(torch.cat((table.flatten(), cache.flatten())))
    errors.zero_()
    prep = torch.cuda.Stream()
    prep.wait_stream(torch.cuda.current_stream())
    for _ in range(256):
        with torch.cuda.stream(prep):
            checker(table, cache, last, errors, owned)
        graph.replay()
    prep.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(prep):
        checker(table, cache, last, errors, owned)
    torch.cuda.current_stream().wait_stream(prep)
    if mutant:
        assert errors.any().item(), "adjacent-row RMW mutant escaped ownership checker"
    else:
        assert not errors.any().item()
        # Other table rows belong to prep; compare all round outputs and owned rows.
        snapshot = clone(actual)
        for name in ("token_table", "cache_index_table"):
            for row in range(snapshot[name].shape[0]):
                if row not in rows:
                    snapshot[name][row].copy_(expected[name][row])
        assert_equal(snapshot, expected)


@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_noncontiguous_rows_and_independent_strides(split):
    for mode in MODES:
        t = fixture(mode=mode)
        for name in ("token_table", "cache_index_table", "tokens", "proposals"):
            value = t[name]
            if value is None:
                continue
            backing = torch.full((value.shape[0], value.shape[1] + 19), -999, dtype=value.dtype)
            view = backing[:, : value.shape[1]]
            view.copy_(value)
            t[name] = view
        check_case(t, split=split)
        backing = torch.full((t["kv_indices"].numel() + 1,), SENTINEL, dtype=torch.int32)
        t["kv_indices"] = backing[1:]
        check_case(t, split=split)


@pytest.mark.parametrize("q", [33, 64, 65, 128, 129])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_long_round_reference_graph(q, mode, split):
    from ..modules.test_dflash_round import long_round_cases

    for t in long_round_cases(q, mode):
        check_case(t, split=split)
        # Capture admits declared variants only; eager checks compile lazily.
        warm_up(
            q,
            t["max_seq_len"],
            t["stop_token_ids"].numel(),
            device="cuda",
            use_pdl=t["use_pdl"],
        )
        actual, seed = clone(t, "cuda"), clone(t, "cuda")
        expected = clone(t)
        round_call(expected)

        def run(actual=actual, seed=seed):
            stage(actual, seed)
            round_call(actual, split)

        graph = capture(run)
        for _ in range(2):
            graph.replay()
            assert_equal(actual, expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_round_q1_capture(mode, split):
    t = fixture(4, 1, 67, mode=mode)
    if t["proposals"] is not None:
        t["proposals"] = torch.empty_strided((4, 0), (17, 9), dtype=torch.int32)
    actual = clone(t, "cuda")
    warm_up(1, 67, t["stop_token_ids"].numel(), device="cuda")
    graph = capture(lambda: round_call(actual, split))
    for turn in range(7):
        initial = fixture(4, 1, 67, mode=mode, turn=turn)
        stage(actual, initial)
        expected = clone(initial)
        round_call(expected)
        graph.replay()
        assert_equal(actual, expected)
        assert torch.equal(actual["accepted"].cpu(), torch.where(expected["round_done"], -1, 0))


DECLARED = (
    RoundVariant("prepare_block", rope=True, cursors=True),
    RoundVariant("prepare_block"),
    RoundVariant("commit_block", mode="greedy"),
    RoundVariant("commit_block", mode="sampled"),
    RoundVariant("prepare_round", rope=True, cursors=True),
    RoundVariant("fixed_query_addresses"),
)


def _declared_fixture(b, q, mode, optional, s):
    t = fixture(b, q, 67, mode=mode, s=s)
    t["k2_slots"] = t["k2_lengths"] = None
    if not optional:
        t["rope_offsets"] = t["verify_rope_positions"] = None
        t["prompt_lengths"] = t["cursors"] = None
    return t


def test_declared_warm_up_then_capture_compiles_nothing():
    # Q=10 and S=2 appear in no other warm_up here: only this declaration covers them.
    q, s = 10, 2
    assert set(warm_up(q, 67, s, device="cuda", variants=DECLARED)) == set(DECLARED)
    count = compile_count()
    cases = ((False, True, "greedy"), (False, False, "sampled"), (True, True, "sampled"))
    for (split, optional, mode), b in product(cases, (1, 4)):
        t = _declared_fixture(b, q, mode, optional, s)
        actual, seed = clone(t, "cuda"), clone(t, "cuda")

        def run(actual=actual, seed=seed, split=split):
            stage(actual, seed)
            round_call(actual, split)

        capture(run).replay()
        expected = clone(t)
        round_call(expected)
        assert_equal(actual, expected)
    assert compile_count() == count


def test_capture_of_undeclared_variant_raises_and_names_it():
    q, s = 7, 2
    warm_up(q, 67, s, device="cuda", variants=DECLARED[:3])
    t = _declared_fixture(2, q, "sampled", False, s)
    actual = clone(t, "cuda")
    with pytest.raises(
        RuntimeError,
        match=r"RoundVariant\('commit_block', mode='sampled', k2=False\) at block_size=7, "
        r"num_stop_tokens=2, use_pdl=False was not declared",
    ):
        # capture() runs eagerly first (which compiles lazily), then captures.
        capture(lambda: round_call(actual))


# Two configurations per B cover all Q, modes, options, split, PDL and M axes.
ANY_BATCH_CORE = [
    (
        b,
        ANY_QS[(i + j) % len(ANY_QS)],
        MODES[(i + j) % 3],
        bool((i + j) % 2),
        bool(j),
        bool(i % 2),
        4096 if b <= 16 and j else 67,
    )
    for i, b in enumerate(BATCHES)
    for j in range(2)
]


@pytest.mark.parametrize(
    "b,q,mode,optional,split,pdl,m",
    ci_cases(
        BATCHES,
        ANY_QS,
        MODES,
        (False, True),
        (False, True),
        (False, True),
        (67, 4096),
        core=ANY_BATCH_CORE,
    ),
)
def test_any_batch_matrix(b, q, mode, optional, split, pdl, m):
    t = fixture(b, q, m, mode=mode, optional=optional, turn=2)
    t["frontier_table"].clamp_(max=m - 1)  # FR is a legal-frontier oracle.
    t["use_pdl"] = pdl
    check_case(t, split=split)


@pytest.mark.parametrize("split", (False, True), ids=("fused", "split"))
def test_any_batch_large_m_and_short_capacity(split):
    check_case(fixture(8, 16, 262144), split=split)
    for mode in MODES:
        t = fixture(8, 33, 67, mode=mode)
        oracle = clone(t)
        fr_prepare(oracle)
        total = int(oracle["kv_indptr"][-1])
        for n in (0, 1, total // 2):
            short = clone(t)
            short["kv_indices"] = short["kv_indices"][:n]
            check_case(short, split=split)


@pytest.mark.parametrize("q", (8, 16))
def test_one_warm_up_captures_every_batch(q):
    warm_up(q, 67, 3, device="cuda", variants=DECLARED)
    count = compile_count()
    for i, b in enumerate(BATCHES):
        for split in (False, True):
            t = _declared_fixture(b, q, "greedy" if i % 2 else "sampled", True, 3)
            actual, seed = clone(t, "cuda"), clone(t, "cuda")

            def run(actual=actual, seed=seed, split=split):
                stage(actual, seed)
                round_call(actual, split)

            graph = capture(run)
            assert compile_count() == count
            expected = clone(t)
            round_call(expected)
            for _ in range(2):
                graph.replay()
                assert_equal(actual, expected)
            assert compile_count() == count


@pytest.mark.parametrize("optional", (False, True))
@pytest.mark.parametrize("mode", MODES)
def test_empty_cuda_no_launch_or_declaration(monkeypatch, optional, mode):
    from ctikernels.speculative._round import op

    t = fixture(0, mode=mode, optional=optional)
    t["kv_indptr"].fill_(123)
    t["kv_indices"] = torch.full((17,), 456, dtype=torch.int32)
    actual = clone(t, "cuda")
    monkeypatch.setattr(op, "provider", forbid_provider)
    count = compile_count()
    for spec in OPS:
        assert invoke_op(spec, actual) is None
    graph = capture(lambda: [invoke_op(spec, actual) for spec in OPS])
    graph.replay()
    assert compile_count() == count
    for name, value in t.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[name].cpu(), value), name


ADDRESSES = (RoundVariant("fixed_query_addresses"),)
ADDRESS_OUTPUTS = ("cache_write_indices", "kv_indptr", "kv_indices")
ADDRESS_ARGS = (
    "request_indices",
    "start_frontiers",
    "round_done",
    "cache_index_table",
    "cache_write_indices",
    "kv_indptr",
    "kv_indices",
    "block_size",
    "max_seq_len",
    "guard_start",
    "use_pdl",
)


def addresses_case(b, q, m, *, mix, guard_max, capacity=None, shift=0, seed=0):
    """Saved round state over non-contiguous table rows; the table and kv_indices
    are views at 4-byte (not 16-byte) offsets with row gaps; outputs poisoned."""
    g = torch.Generator().manual_seed(seed)
    rows = 2 * b + 3
    backing = torch.randint(-(2**31), 2**31 - 1, (rows, m + 12), generator=g).to(torch.int32)
    frontiers = (0, m - q - 1, m - q, m - 1, m, -1)
    f = torch.tensor([frontiers[(k + mix) % 6] for k in range(b)], dtype=torch.int64)
    done = torch.tensor([mix % 2 == 1 and k % 3 == 1 for k in range(b)], dtype=torch.bool)
    dummy = done | (f < 0) | (f >= m)
    total = int(torch.where(dummy, q, f + q).sum())
    n = total + 9 if capacity is None else capacity
    return {
        "request_indices": torch.randperm(rows, generator=g)[:b].to(torch.int32),
        "start_frontiers": f,
        "round_done": done,
        "cache_index_table": backing[:, 3 : m + 8],
        "cache_write_indices": torch.full((b * q,), SENTINEL, dtype=torch.int32),
        "kv_indptr": torch.full((b + 1,), SENTINEL, dtype=torch.int32),
        "kv_indices": torch.full((n + shift,), SENTINEL, dtype=torch.int32)[shift:],
        "block_size": q,
        "max_seq_len": m,
        "guard_start": 2**31 - 1 - b * q if guard_max else 0,
        "use_pdl": False,
    }


def to_cuda(t):
    out = dict(t)
    table = t["cache_index_table"]
    backing = table.as_strided((table.shape[0], table.stride(0)), table.stride(), 0)
    offset = table.storage_offset()
    out["cache_index_table"] = backing.cuda()[:, offset : offset + table.shape[1]]
    shift = t["kv_indices"].storage_offset()
    out["kv_indices"] = t["kv_indices"]._base.cuda()[shift:]
    for name in ADDRESS_OUTPUTS[:2] + ADDRESS_ARGS[:3]:
        out[name] = t[name].cuda()
    return out


def call_addresses(t):
    from ctikernels.speculative import fixed_query_addresses

    fixed_query_addresses(**{n: t[n] for n in ADDRESS_ARGS})


def check_addresses(actual, t):
    expected = {**t, **{n: t[n].clone() for n in ADDRESS_OUTPUTS}}
    call_addresses(expected)  # CPU: the torch reference
    for name in ADDRESS_OUTPUTS:
        assert torch.equal(actual[name].cpu(), expected[name]), name


@pytest.mark.parametrize("q", (1, 2, 4, 5, 8, 9, 12, 13, 16, 17, 32, 33))
def test_fixed_query_addresses_grid_bitwise(q):
    """Triton addresses == the torch reference over B, M, boundary frontiers,
    done mixes, guard bounds and short kv_indices capacities."""
    warm_up(q, 32768, 0, device="cuda", variants=ADDRESSES)
    seed = 0
    for m in sorted({1, q, 2 * q + 1, 4096, 32768}):
        for b in (0, 1, 2, 3, 4, 5, 64, 191):
            for mix in range(6 if b <= 5 else 2):
                for guard_max in (False, True):
                    for capacity in (None, 0, 1, 5 * q) if mix == 0 else (None,):
                        seed += 1
                        # Odd seeds shift kv_indices by one element (capacity 0:
                        # a view one past the end of its allocation).
                        t = addresses_case(
                            b,
                            q,
                            m,
                            mix=mix,
                            guard_max=guard_max,
                            capacity=capacity,
                            shift=seed % 2,
                            seed=seed,
                        )
                        actual = to_cuda(t)
                        call_addresses(actual)
                        check_addresses(actual, t)


@pytest.mark.parametrize("use_pdl", (False, True))
@pytest.mark.parametrize("q", (1, 16, 33))
def test_fixed_query_addresses_capture_after_warm_up(q, use_pdl):
    """One warm_up covers every B, M, guard, capacity and 4-byte alignment:
    capture compiles nothing and the graph equals the eager call."""
    warm_up(q, 67, 0, device="cuda", use_pdl=use_pdl, variants=ADDRESSES)
    count = compile_count()
    for b, m, shift in ((1, 67, 0), (4, 4096, 1), (5, 67, 3), (64, 32768, 0), (191, 67, 1)):
        t = addresses_case(b, q, m, mix=b % 6, guard_max=bool(shift), shift=shift, seed=b + q)
        t["use_pdl"] = use_pdl
        eager, graphed = to_cuda(t), to_cuda(t)
        call_addresses(eager)
        graph = capture(lambda graphed=graphed: call_addresses(graphed))
        for name in ADDRESS_OUTPUTS:
            graphed[name].fill_(SENTINEL)
        graph.replay()
        check_addresses(eager, t)
        check_addresses(graphed, t)
    assert compile_count() == count
