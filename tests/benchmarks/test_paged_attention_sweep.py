# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU coverage for sweep selection and cache-footprint planning."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark.kernel_tuning import paged_attention_sweep as sweep
from benchmark.kernel_tuning.paged_attention_sweep import (
    generate_rows,
    layer_count,
    matches_only,
    parse_args,
    parse_only,
)


def test_only_matches_whole_fields_and_combines_selectors():
    selectors = parse_only("decode:qwen3-8b:2:512, verify:*:*:1024, *:qwen35-9b:1:*")
    assert selectors[0] == ("decode", "qwen3-8b", 2, 512)
    for op, model, batch, kv, expected in (
        ("decode", "qwen3-8b", 2, 512, True),
        ("decode", "qwen3-8b", 1, 512, False),
        ("verify", "qwen35-9b", 4, 1024, True),
        ("verify", "qwen3-8b", 2, 512, False),
        ("prefill", "qwen35-9b", 1, 22, True),
    ):
        row = {"op": op, "model": model, "B": batch, "KV": kv}
        assert matches_only(row, selectors) is expected
        assert matches_only(row, None)
        assert matches_only(row, parse_only("*:*:*:*"))


@pytest.mark.parametrize(
    "value",
    [
        "",
        "decode:*:1",
        "decode:*:1:512:extra",
        "other:*:1:512",
        "decode:qwen*:1:512",
        "decode:*:0:512",
        "decode:*:1:-2",
        "decode:*:1:5*",
        "decode:*:1:512,",
    ],
)
def test_only_rejects_malformed_selectors(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_only(value)


@pytest.mark.parametrize(
    "l2,per_layer,budget,comparator,expected",
    [
        (100, 100, 10000, 0, 3),
        (101, 100, 10000, 0, 4),  # ceiling, not floor
        (0, 100, 10000, 0, 2),
        (1, 100, 10000, 0, 2),
        (10000, 100, 10000, 0, 64),
        (1000, 100, 599, 0, 5),
        (1000, 100, 600, 100, 3),  # both resident providers count
        (1000, 100, 400, 100, 2),
    ],
)
def test_layer_count(l2, per_layer, budget, comparator, expected):
    assert layer_count(l2, per_layer, budget, comparator) == expected


@pytest.mark.parametrize(
    "args",
    [
        (100, 100, 199),
        (100, 100, 399, 100),
        (-1, 100, 1000),
        (100, 0, 1000),
        (100, 100, 0),
        (100, 100, 1000, -1),
    ],
)
def test_layer_count_rejects_impossible_budget_and_invalid_sizes(args):
    with pytest.raises(ValueError):
        layer_count(*args)


def test_default_rows_include_fresh_prefill_once():
    args = parse_args([])
    rows = generate_rows(args)
    assert len(rows) == 2 * 4 * 6 + 4
    assert rows[0] == {"op": "decode", "model": "qwen3-8b", "B": 1, "KV": 512, "Q": 1}
    verify = [row for row in rows if row["op"] == "verify"]
    assert len(verify) == 24 and all(row["Q"] == 16 for row in verify)
    prefill = [row for row in rows if row["op"] == "prefill"]
    assert [row["KV"] for row in prefill] == [22, 128, 512, 2048]
    assert all(row["B"] == 1 and row["Q"] == row["KV"] for row in prefill)


def test_custom_rows_filter_models_and_deduplicate_cli_lists():
    args = parse_args(
        [
            "--models",
            "qwen3-8b,qwen35-9b,qwen3-8b",
            "--ops",
            "verify,prefill",
            "--batches",
            "2,3,2",
            "--kv",
            "17,1024",
            "--verify-q",
            "7",
            "--prefill-kv",
            "22,22",
            "--only",
            "verify:qwen35-9b:3:*,prefill:*:*:*",
        ]
    )
    assert generate_rows(args) == [
        {"op": "prefill", "model": "qwen3-8b", "B": 1, "KV": 22, "Q": 22},
        {"op": "verify", "model": "qwen35-9b", "B": 3, "KV": 17, "Q": 7},
        {"op": "verify", "model": "qwen35-9b", "B": 3, "KV": 1024, "Q": 7},
        {"op": "prefill", "model": "qwen35-9b", "B": 1, "KV": 22, "Q": 22},
    ]


def test_filter_can_select_no_rows():
    assert generate_rows(parse_args(["--only", "decode:qwen35-9b:*:*"])) == []


def test_prefill_batches_are_independent_and_selectable():
    args = parse_args(
        [
            "--ops",
            "prefill",
            "--batches",
            "9",
            "--prefill-batches",
            "2,3,2",
            "--prefill-kv",
            "7,11",
        ]
    )
    assert generate_rows(args) == [
        {"op": "prefill", "model": "qwen3-8b", "B": batch, "KV": kv, "Q": kv}
        for batch in (2, 3)
        for kv in (7, 11)
    ]
    args.only = parse_only("prefill:*:3:11")
    assert generate_rows(args) == [
        {"op": "prefill", "model": "qwen3-8b", "B": 3, "KV": 11, "Q": 11}
    ]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--kv", "512,0"),
        ("--batches", ""),
        ("--prefill-batches", "0"),
        ("--prefill-batches", "1,"),
        ("--prefill-batches", "-2"),
        ("--models", "unknown"),
        ("--ops", "decode,"),
        ("--reps", "0"),
        ("--pairs", "-1"),
    ],
)
def test_cli_rejects_invalid_values(flag, value):
    with pytest.raises(SystemExit, match="2"):
        parse_args([flag, value])


def test_paired_timings_rotate_variants_and_use_adjacent_sample_ratios(monkeypatch):
    args = parse_args(["--reps", "3", "--pairs", "2"])
    sequence = []
    candidates = [SimpleNamespace(replay=lambda i=i: sequence.append(i)) for i in range(2)]
    comparator = SimpleNamespace(replay=lambda: sequence.append("vllm"))
    values = iter([1, 1, 4, 2, 8, 4, 100, 2, 101, 100, 12, 6])

    def measure(replay, calls):
        assert calls == 32
        replay()
        return next(values)

    monkeypatch.setattr(sweep, "measure", measure)
    first, second = sweep.time_graphs(candidates, comparator, 16, args)
    assert sequence == [item for i in [0, 1, 1, 0, 0, 1] for item in [i, i, "vllm", "vllm"]]
    assert first["pair_us"] == [[1, 1], [100, 2], [101, 100]]
    assert first["ratio_vs_vllm"] == pytest.approx(1.01)
    assert second["ratio_vs_vllm"] == 2
    assert (first["us"], first["vllm_us"]) == (100, 2)
    assert first["p10"] == pytest.approx(20.8)
    assert first["p90"] == pytest.approx(100.8)


def test_unpaired_timings_take_reps_samples_and_rotate(monkeypatch):
    values = iter([1, 2, 4, 3, 5, 6])
    sequence = []
    candidates = [SimpleNamespace(replay=lambda i=i: sequence.append(i)) for i in range(2)]

    def measure(replay, calls):
        assert calls == 16
        replay()
        return next(values)

    monkeypatch.setattr(sweep, "measure", measure)
    first, second = sweep.time_graphs(candidates, None, 16, parse_args(["--reps", "3"]))
    assert sequence == [0, 1, 1, 0, 0, 1]
    assert (first["us"], second["us"]) == (3, 4)
    assert first["p10"] == pytest.approx(1.4)
    assert first["p90"] == pytest.approx(4.6)
    assert first["vllm_us"] is None and first["ratio_vs_vllm"] is None
    assert sweep.time_graphs([], None, 16, parse_args([])) == []


def test_split_parsers_and_defaults():
    args = parse_args([])
    assert args.decode_split == ["device", "provider"]
    assert args.verify_split == ["tiers", "m128"]
    assert args.prefill_qsplit == ["device", "provider"]
    assert args.prefill_batches == [1]
    assert sweep.decode_splits("device,provider, 128,256,128,device") == [
        "device",
        "provider",
        128,
        256,
    ]
    args = parse_args(["--decode-split", "device,512", "--prefill-qsplit", "1,3,3"])
    assert sweep.row_splits({"op": "decode"}, args) == ["device", 512]
    assert sweep.row_splits({"op": "verify"}, args) == ["tiers", "m128"]
    assert sweep.row_splits({"op": "prefill"}, args) == ["q1", "q3"]


def test_verify_variants_normalize_and_preserve_order():
    args = parse_args(["--verify-split", "tiers,m1, m16,m32,m64,m128,m032,tiers"])
    assert sweep.row_splits({"op": "verify"}, args) == ["tiers", "m1", "m16", "m32", "m64", "m128"]


def test_prefill_variants_normalize_aliases_and_preserve_order():
    args = parse_args(["--prefill-qsplit", "q1, 2, q2,q2+ns,c3,c3+ns,1+ns,q1+ns,c03,c1,q001"])
    assert sweep.row_splits({"op": "prefill"}, args) == [
        "q1",
        "q2",
        "q2+ns",
        "c3",
        "c3+ns",
        "q1+ns",
        "c1",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "q0",
        "c0",
        "q-1",
        "c-2",
        "q",
        "c",
        "q1,",
        "q1+",
        "q1+NS",
        "q1+ns+ns",
        "c2+other",
        "q1.5",
        "x2",
        "+1",
        "1_000",
    ],
)
def test_prefill_parser_rejects_malformed_variants(value):
    with pytest.raises(argparse.ArgumentTypeError):
        sweep.prefill_splits(value)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--decode-split", "auto,"),
        ("--decode-split", "0"),
        ("--decode-split", "1:2"),
        ("--verify-split", ""),
        ("--verify-split", "tiers,"),
        ("--verify-split", "provider"),
        ("--verify-split", "32"),
        ("--verify-split", "m0"),
        ("--verify-split", "m129"),
        ("--verify-split", "m-1"),
        ("--verify-split", "m1.5"),
        ("--verify-split", "m32+ns"),
        ("--prefill-qsplit", "0"),
        ("--prefill-qsplit", "auto"),
    ],
)
def test_split_cli_rejects_invalid_values(flag, value):
    with pytest.raises(SystemExit, match="2"):
        parse_args([flag, value])


@pytest.mark.parametrize(
    "query,parts,expected",
    [
        (7, 1, [(0, 7)]),
        (7, 3, [(0, 3), (3, 5), (5, 7)]),
        (3, 3, [(0, 1), (1, 2), (2, 3)]),
        (8, 2, [(0, 4), (4, 8)]),
    ],
)
def test_qsplit_partition(query, parts, expected):
    assert sweep.qsplit_partition(query, parts) == expected
    for start, end in expected:
        for local in range(end - start):
            # Bottom-right causal alignment: KV length - Q length + local.
            assert end - (end - start) + local == start + local


@pytest.mark.parametrize("query,parts", [(0, 1), (3, 0), (3, 4), (3, -1)])
def test_qsplit_rejects_empty_parts(query, parts):
    with pytest.raises(sweep.Infeasible):
        sweep.qsplit_partition(query, parts)


@pytest.mark.parametrize("suffix", ["", "+ns"])
@pytest.mark.parametrize(
    "query,split,expected",
    [
        (7, "q1", [(0, 7)]),
        (7, "q3", [(0, 3), (3, 5), (5, 7)]),
        (7, "c3", [(0, 3), (3, 6), (6, 7)]),
        (6, "c3", [(0, 3), (3, 6)]),
        (3, "c1", [(0, 1), (1, 2), (2, 3)]),
        (3, "c8", [(0, 3)]),
    ],
)
def test_prefill_partition(query, split, expected, suffix):
    assert sweep.prefill_partition(query, split + suffix) == expected


@pytest.mark.parametrize("query,split", [(0, "c1"), (3, "c0"), (3, "q4"), (3, "q4+ns")])
def test_prefill_partition_rejects_infeasible_ranges(query, split):
    with pytest.raises(sweep.Infeasible):
        sweep.prefill_partition(query, split)


@pytest.fixture
def fake_provider(monkeypatch):
    import torch

    from ctikernels.attention import _flashinfer, inference

    layout = [8, 1, 2, 3, 4] + [0] * (len(_flashinfer._PlanLayout._fields) - 5)
    native = SimpleNamespace(plan=Mock(), _plan_info=layout, use_tensor_cores=True, _backend="fa2")
    monkeypatch.setattr(_flashinfer, "qualified_flashinfer_version", lambda: True)
    wrapper = SimpleNamespace(plan=Mock(), _wrapper=native)
    provider = Mock(return_value=wrapper)
    provider.workspace_size.return_value = (0, 0)
    monkeypatch.setattr(inference, "FlashInferPagedAttention", provider)
    monkeypatch.setattr(inference, "select_flashinfer_prefill_implementation", lambda *_: "fa2")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    return provider, wrapper


def test_decode_fixed_split_replans_same_wrapper_with_native_mapping(fake_provider):
    import torch

    provider, wrapper = fake_provider
    row = {"op": "decode", "model": "qwen3-8b", "B": 2, "KV": 512, "Q": 1}
    actual, plan, _ = sweep.make_operation(row, torch.device("cpu"), 128)
    assert actual is wrapper and provider.call_count == 1
    assert provider.call_args.kwargs["dispatch"] == "provider"
    assert provider.workspace_size.call_args.kwargs["dispatch"] == "provider"
    assert plan == [8, 1, 2, 3, 4]
    original = wrapper.plan.call_args.kwargs
    native = wrapper._wrapper.plan.call_args.kwargs
    for public, private in [
        ("cu_seqlens_k", "indptr"),
        ("indices", "indices"),
        ("last_page_len", "last_page_len"),
        ("seq_lens", "seq_lens"),
    ]:
        assert original[public] is native[private]
    assert set(native) == {
        "indptr",
        "indices",
        "last_page_len",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "page_size",
        "pos_encoding_mode",
        "seq_lens",
        "data_type",
        "q_data_type",
        "kv_data_type",
        "non_blocking",
        "fixed_split_size",
    }
    assert native["fixed_split_size"] == 128 and native["non_blocking"] is True
    assert native["q_data_type"] == native["kv_data_type"] == native["data_type"] == torch.bfloat16


def test_fixed_decode_split_requires_qualified_plan_layout(fake_provider, monkeypatch):
    import torch

    from ctikernels.attention import _flashinfer

    _, wrapper = fake_provider
    monkeypatch.setattr(_flashinfer, "qualified_flashinfer_version", lambda: False)
    row = {"op": "decode", "model": "qwen3-8b", "B": 2, "KV": 512, "Q": 1}
    with pytest.raises(sweep.Infeasible, match="padded CTA count"):
        sweep.make_operation(row, torch.device("cpu"), 128)
    wrapper._wrapper.plan.assert_not_called()
    assert sweep.make_operation(row, torch.device("cpu"), "provider")[1] is None


@pytest.mark.parametrize("installed", ["0.6.11.post3", "0.7.0.post1"])
@pytest.mark.parametrize("split", ["tiers", "m1", "m16", "m32", "m64", "m128"])
def test_verify_split_constructs_fixed_query_provider(monkeypatch, installed, split):
    import torch

    from ctikernels.attention import inference

    wrapper = SimpleNamespace()
    provider = Mock(return_value=wrapper)
    provider.workspace_size.return_value = (0, 0)
    monkeypatch.setattr(inference, "FlashInferFixedQueryAttention", provider)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__=installed))
    monkeypatch.setattr(sweep, "version", lambda _: installed)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    row = {"op": "verify", "model": "qwen3-8b", "B": 2, "KV": 512, "Q": 16}
    actual, plan, _ = sweep.make_operation(row, torch.device("cpu"), split)
    assert actual is wrapper and plan is None
    assert provider.call_count == 1
    options = provider.call_args.kwargs
    if split == "tiers":
        assert "min_chunk_tokens" not in options
    else:
        assert options["min_chunk_tokens"] == int(split[1:])
    assert options["causal"] is True
    assert options["query_length"] == 16 and options["page_size"] == 1
    assert options["kv_indptr"].tolist() == [0, 528, 1056]


@pytest.mark.parametrize("heads,split", [((32, 8, 128), 127), ((8, 8, 128), 512)])
def test_infeasible_decode_does_not_replan(fake_provider, monkeypatch, heads, split):
    import torch

    _, wrapper = fake_provider
    monkeypatch.setitem(sweep.MODELS, "tiny", heads)
    row = {"op": "decode", "model": "tiny", "B": 2, "KV": 512, "Q": 1}
    with pytest.raises(sweep.Infeasible):
        sweep.make_operation(row, torch.device("cpu"), split)
    wrapper._wrapper.plan.assert_not_called()


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("suffix", ["", "+ns"])
@pytest.mark.parametrize(
    "split,ends,kv_ptr",
    [
        ("q1", [7], [0, 7]),
        ("q3", [3, 5, 7], [0, 3, 8, 15]),
        ("c3", [3, 6, 7], [0, 3, 9, 16]),
        ("c8", [7], [0, 7]),
    ],
)
def test_prefill_plan_uses_request_local_page_prefixes(
    fake_provider, batch, suffix, split, ends, kv_ptr
):
    import torch

    provider, wrapper = fake_provider
    row = {"op": "prefill", "model": "qwen3-8b", "B": batch, "KV": 7, "Q": 7}
    sweep.make_operation(row, torch.device("cpu"), split + suffix)
    plan = wrapper.plan.call_args.kwargs
    assert plan["cu_seqlens_q"].tolist() == [0] + [
        request * 7 + end for request in range(batch) for end in ends
    ]
    assert plan["cu_seqlens_k"].tolist() == [0] + [
        request * kv_ptr[-1] + ptr for request in range(batch) for ptr in kv_ptr[1:]
    ]
    assert plan["seq_lens"].tolist() == ends * batch
    assert plan["indices"].tolist() == [
        request * 7 + page for request in range(batch) for end in ends for page in range(end)
    ]
    assert plan["last_page_len"].tolist() == [1] * (batch * len(ends))
    assert plan["causal"] is True
    assert plan.get("disable_split_kv", False) is bool(suffix)
    assert wrapper.plan.call_count == 1
    assert provider.workspace_size.call_args.kwargs["max_batch_size"] == batch * len(ends)
    assert provider.workspace_size.call_args.kwargs["max_query_tokens"] == batch * 7
    assert provider.workspace_size.call_args.kwargs["dispatch"] == "provider"


@pytest.mark.parametrize("op,kv,qlen", [("decode", 5, 1), ("verify", 3, 2), ("prefill", 65, 65)])
def test_reference_grouped_heads_causality_and_query_chunks_on_cpu(monkeypatch, op, kv, qlen):
    import torch

    monkeypatch.setitem(sweep.MODELS, "tiny", (4, 2, 4))
    row = {"op": op, "model": "tiny", "B": 2, "KV": kv, "Q": qlen}
    layer = sweep.make_layers(row, 1, torch.device("cpu"))[0]
    actual = sweep.reference_attention(row, layer)
    q, k, v, _ = layer
    length = kv + (qlen if op == "verify" else 0)
    expected = torch.empty_like(q, dtype=torch.float32)
    # An unvectorized oracle makes each head's grouping and causal boundary explicit.
    for request in range(2):
        for token in range(qlen):
            visible = length - qlen + token + 1
            start = request * length
            for head in range(4):
                query = q[request * qlen + token, head].float()
                keys = k[start : start + visible, 0, head // 2].float()
                values = v[start : start + visible, 0, head // 2].float()
                weights = ((keys @ query) / 2).softmax(0)
                expected[request * qlen + token, head] = weights @ values
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_vllm_padding_preserves_each_requests_keys_on_cpu(monkeypatch):
    import torch

    monkeypatch.setitem(sweep.MODELS, "tiny", (4, 2, 4))
    row = {"op": "decode", "model": "tiny", "B": 2, "KV": 17, "Q": 1}
    source = sweep.make_layers(row, 2, torch.device("cpu"))
    comparison = sweep.make_layers(row, 2, torch.device("cpu"), comparator=True, source=source)
    assert source[0][1].data_ptr() != source[1][1].data_ptr()
    for original, padded in zip(source, comparison, strict=True):
        torch.testing.assert_close(original[0], padded[0], atol=0, rtol=0)
        assert original[0].data_ptr() != padded[0].data_ptr()
        for src, dst in zip(original[1:3], padded[1:3], strict=True):
            assert dst.shape == (4, 16, 2, 4)
            torch.testing.assert_close(
                src.view(2, 17, 2, 4), dst.view(2, 32, 2, 4)[:, :17], atol=0, rtol=0
            )
            assert torch.count_nonzero(dst.view(2, 32, 2, 4)[:, 17:]) == 0


def test_row_builds_all_variant_graphs_before_timing_and_preserves_failures(monkeypatch):
    import torch

    row = {"op": "decode", "model": "qwen3-8b", "B": 1, "KV": 512, "Q": 1}
    args = parse_args(["--decode-split", "device,1,128", "--hot-calls", "1", "--reps", "2"])
    layer = (None, None, None, torch.zeros(1))
    events = []
    monkeypatch.setattr(sweep, "make_layers", lambda *_: [layer])
    monkeypatch.setattr(sweep, "reference_attention", lambda *_: torch.zeros(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    def operation(row, device, split):
        events.append(("plan", split))
        if split == 1:
            raise sweep.Infeasible("grid")
        return SimpleNamespace(run=lambda **_: None), [8], ()

    def capture(call):
        events.append(("capture", None))
        call()
        return SimpleNamespace(replay=lambda: None)

    def measure(replay, calls):
        events.append(("measure", None))
        return 1

    monkeypatch.setattr(sweep, "make_operation", operation)
    monkeypatch.setattr(sweep, "graph_capture", capture)
    monkeypatch.setattr(sweep, "measure", measure)
    results = sweep.run_row(row, "hot", args, torch.device("cpu"), 0)
    assert [(r["split"], r["status"]) for r in results] == [
        ("device", "ok"),
        (1, "infeasible"),
        (128, "ok"),
    ]
    assert (
        events
        == [("plan", "device"), ("capture", None), ("plan", 1), ("plan", 128), ("capture", None)]
        + [("measure", None)] * 4
    )
    assert all(r["correct"] is True for r in results if r["status"] == "ok")


@pytest.mark.parametrize("split", ["device", "provider"])
def test_decode_split_policy_replans_device_variant(fake_provider, split):
    import torch

    provider, wrapper = fake_provider
    row = {"op": "decode", "model": "qwen3-8b", "B": 2, "KV": 512, "Q": 1}
    sweep.make_operation(row, torch.device("cpu"), split)
    assert provider.call_args.kwargs["dispatch"] == split
    assert provider.workspace_size.call_count == 1
    assert provider.workspace_size.call_args.kwargs["dispatch"] == split
    assert wrapper.plan.call_count == (2 if split == "device" else 1)
    wrapper._wrapper.plan.assert_not_called()


@pytest.mark.parametrize("regime", ["hot", "engine"])
def test_verify_row_records_splits_and_times_variants_round_robin(monkeypatch, regime):
    import torch

    args = parse_args(["--verify-split", "tiers,m32,m128", "--hot-calls", "2", "--reps", "3"])
    row = {"op": "verify", "model": "qwen3-8b", "B": 1, "KV": 512, "Q": 16}
    layer = (None, None, None, torch.zeros(1))
    events = []
    wrappers = []
    monkeypatch.setattr(sweep, "make_layers", lambda _, layers, __: [layer] * layers)
    monkeypatch.setattr(sweep, "reference_attention", lambda *_: torch.zeros(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    def operation(row, device, split):
        events.append(("plan", split))
        wrapper = SimpleNamespace(prepare=Mock(), run=Mock())
        wrappers.append(wrapper)
        return wrapper, None, ()

    def capture(call):
        split = args.verify_split[len(wrappers) - 1]
        events.append(("capture", split))
        call()
        return SimpleNamespace(replay=lambda: events.append(("replay", split)))

    def measure(replay, calls):
        assert calls == 2
        replay()
        return 1

    monkeypatch.setattr(sweep, "make_operation", operation)
    monkeypatch.setattr(sweep, "graph_capture", capture)
    monkeypatch.setattr(sweep, "measure", measure)
    results = sweep.run_row(row, regime, args, torch.device("cpu"), 0)
    assert [(r["split"], r["status"]) for r in results] == [
        (split, "ok") for split in args.verify_split
    ]
    assert events[:6] == [
        (event, split) for split in args.verify_split for event in ("plan", "capture")
    ]
    assert events[-9:] == [
        ("replay", split)
        for split in ["tiers", "m32", "m128", "m32", "m128", "tiers", "m128", "tiers", "m32"]
    ]
    for wrapper in wrappers:
        wrapper.prepare.assert_called_once_with()
        assert wrapper.run.call_count == 2
    assert all(r["correct"] is True and r["us"] == 1 for r in results)


@pytest.mark.parametrize("split", ["device", "provider"])
def test_prefill_dispatch_plans_original_requests(fake_provider, split):
    import torch

    provider, wrapper = fake_provider
    row = {"op": "prefill", "model": "qwen3-8b", "B": 2, "KV": 22, "Q": 22}
    sweep.make_operation(row, torch.device("cpu"), split)
    assert provider.call_args.kwargs["dispatch"] == split
    assert provider.workspace_size.call_count == 1
    assert provider.workspace_size.call_args.kwargs["dispatch"] == split
    assert provider.workspace_size.call_args.kwargs["max_batch_size"] == 2
    assert wrapper.plan.call_args.kwargs["cu_seqlens_q"].tolist() == [0, 22, 44]
    assert wrapper.plan.call_args.kwargs["indices"].tolist() == list(range(44))
    assert wrapper.plan.call_count == 1


def test_notc_decode_uses_provider_dispatch(fake_provider):
    import torch

    provider, wrapper = fake_provider
    row = {"op": "decode", "model": "qwen3-8b", "B": 2, "KV": 512, "Q": 1}
    sweep.make_operation(row, torch.device("cpu"), "notc")
    assert provider.call_args.kwargs["dispatch"] == "provider"
    assert provider.workspace_size.call_args.kwargs["dispatch"] == "provider"
    assert provider.call_args.kwargs["use_tensor_cores"] is False
    assert wrapper.plan.call_count == 1
    assert sweep.decode_splits("device,notc,provider") == ["device", "notc", "provider"]
    assert sweep.prefill_splits("device,provider,c8,device") == ["device", "provider", "c8"]


@pytest.mark.parametrize("failure", [None, "setup", "comparator", "row"])
def test_main_records_versions_and_omits_paths(monkeypatch, failure):
    import torch

    records = []
    monkeypatch.setattr(sweep, "emit", records.append)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__="0.7.0.post1"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    def fail(*args, **kwargs):
        raise OSError("cannot open /private/provider/library.so")

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        fail
        if failure == "setup"
        else lambda _: SimpleNamespace(
            name="test GPU",
            multi_processor_count=170,
            L2_cache_size=1024,
        ),
    )
    monkeypatch.setattr(
        sweep, "library_sha256", fail if failure == "comparator" else lambda _: "digest"
    )
    monkeypatch.setattr(torch.ops, "load_library", lambda _: None)
    monkeypatch.setattr(
        sweep,
        "run_row",
        fail if failure == "row" else lambda *args: [{"status": "ok"}],
    )
    sweep.main(
        [
            "--ops",
            "decode",
            "--models",
            "qwen3-8b",
            "--batches",
            "1",
            "--kv",
            "512",
            "--regimes",
            "hot",
            "--vllm-fa2",
            "/private/provider/library.so",
        ]
    )
    assert records[0] == {
        "type": "host",
        "gpu": None if failure == "setup" else "test GPU",
        "sms": None if failure == "setup" else 170,
        "l2": None if failure == "setup" else 1024,
        "flashinfer": "0.7.0.post1",
        "torch": torch.__version__,
        "vllm_so_sha256": None if failure in ("setup", "comparator") else "digest",
    }
    assert "/private" not in str(records)
    if failure:
        assert all(record["error"] == "OSError" for record in records[1:])


def test_sweep_reads_native_plan_only_through_qualified_adapter():
    # native plan layout belongs to the version-qualified adapter.
    assert "_plan_info" not in Path(sweep.__file__).read_text()
