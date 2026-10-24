# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks for PD4 row selection, paired statistics and provider contracts."""

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark.kernel_tuning import paged_attention_range_market as market


def test_default_grid():
    args = market.parse_args([])
    rows = market.generate_rows(args)
    assert len(rows) == 80
    assert rows[0] == {"op": "fp8-decode", "model": "qwen3-8b", "B": 1, "KV": 512, "Q": 1}
    assert len([r for r in rows if r["op"] == "fp8-decode"]) == 32
    assert len([r for r in rows if r["op"] == "graph-prefill"]) == 48
    assert {r["model"] for r in rows} == {"qwen3-8b", "qwen3.5"}
    assert args.rounds == 12 and args.reps == 9 and args.max_bytes == 6 * 1024**3
    assert args.vllm_fa2 is None and not args.pad_random


def test_custom_grid_deduplicates_preserves_order_and_keeps_kv_inclusive():
    args = market.parse_args(
        [
            "--models",
            "qwen3.5,qwen3.5",
            "--ops",
            "graph-prefill,fp8-decode",
            "--batches",
            "3,1,3",
            "--kv",
            "17,17",
            "--prefill-q",
            "7,1,7",
            "--prefill-kv",
            "19,19",
            "--rounds",
            "2",
            "--reps",
            "1",
            "--max-bytes",
            "1234",
            "--seed",
            "-7",
            "--pad-random",
            "--vllm-fa2",
            "library.so",
        ]
    )
    assert market.generate_rows(args) == [
        {"op": op, "model": "qwen3.5", "B": batch, "KV": kv, "Q": query}
        for op, kv, queries in [("graph-prefill", 19, [7, 1]), ("fp8-decode", 17, [1])]
        for batch in (3, 1)
        for query in queries
    ]
    assert args.seed == -7 and args.pad_random and args.vllm_fa2 == Path("library.so")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--models", "unknown"),
        ("--ops", "decode"),
        ("--ops", "fp8-decode,"),
        ("--batches", "0"),
        ("--kv", "1,"),
        ("--prefill-q", "0"),
        ("--prefill-kv", "-1"),
        ("--rounds", "1"),
        ("--rounds", "0"),
        ("--reps", "0"),
        ("--max-bytes", "-1"),
        ("--seed", "no"),
    ],
)
def test_invalid_arguments(flag, value):
    with pytest.raises(SystemExit, match="2"):
        market.parse_args([flag, value])


def test_prefill_cannot_have_more_queries_than_inclusive_kv():
    with pytest.raises(SystemExit, match="2"):
        market.parse_args(["--prefill-q", "64", "--prefill-kv", "32,128"])
    market.parse_args(["--ops", "fp8-decode", "--prefill-q", "64", "--prefill-kv", "32"])


@pytest.mark.parametrize(
    "df,expected",
    [
        (1, 12.706204736),
        (2, 4.302652730),
        (11, 2.200985160),
        (30, 2.042272456),
        (100, 1.983971518),
        (1000, 1.962339081),
    ],
)
def test_student_t_quantile(df, expected):
    assert market.t_critical_95(df) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("df", [0, -1, 1.5])
def test_student_t_rejects_invalid_degrees_of_freedom(df):
    with pytest.raises(ValueError):
        market.t_critical_95(df)


def test_paired_interval_uses_logs_and_round_pairs():
    ours, theirs = [1, 8, 16], [2, 2, 2]
    actual = market.paired_ci(ours, theirs)
    # log ratios = [-log(2), 2log(2), 3log(2)]; independent analytic variance.
    center = 4 * math.log(2) / 3
    variance = 13 * math.log(2) ** 2 / 3
    width = 4.302652729911 * math.sqrt(variance / 3)
    assert actual["ratio"] == pytest.approx(math.exp(center))
    assert actual["ci95"] == pytest.approx([math.exp(center - width), math.exp(center + width)])
    assert actual["round_ratios"] == [0.5, 4, 8] and actual["n"] == 3
    inverse = market.paired_ci(theirs, ours)
    assert inverse["ratio"] == pytest.approx(1 / actual["ratio"])
    assert inverse["ci95"] == pytest.approx([1 / actual["ci95"][1], 1 / actual["ci95"][0]])


def test_constant_ratios_and_twin_noise_floor():
    actual = market.paired_ci([2, 4, 6], [1, 2, 3])
    assert actual["ratio"] == pytest.approx(2)
    assert actual["ci95"] == pytest.approx([2, 2])
    twin = market.paired_ci([2, 4, 6], [2, 4, 6])
    assert twin["ratio"] == 1 and twin["ci95"] == [1, 1]


@pytest.mark.parametrize(
    "a,b",
    [
        ([], []),
        ([1], [1]),
        ([1, 2], [1, 2, 3]),
        ([0, 1], [1, 1]),
        ([1, 2], [-1, 1]),
        ([math.inf, 1], [1, 1]),
        ([1, 1], [math.nan, 1]),
    ],
)
def test_invalid_pairs(a, b):
    with pytest.raises(ValueError):
        market.paired_ci(a, b)


def test_market_faster_is_selected_per_round_and_missing_market_is_explicit():
    samples = {
        "ours": [4, 4, 4],
        "sglang": [2, 8, 2],
        "vllm": [8, 1, 8],
        "ours_cc": [8, 8, 8],
        "ours_twin": [4, 4, 4],
    }
    ratios = market.summarize_ratios(samples)
    assert ratios["ours/max-market-faster"]["round_ratios"] == [2, 4, 2]
    assert ratios["ours_cc/ours"]["ratio"] == pytest.approx(2)
    assert ratios["ours/ours_twin"]["ci95"] == [1, 1]
    del samples["vllm"]
    ratios = market.summarize_ratios(samples)
    assert ratios["ours/vllm"] is None
    assert ratios["ours/max-market-faster"] == ratios["ours/sglang"]
    assert all(value is None for value in market.summarize_ratios({}).values())


@pytest.mark.parametrize("version", ["0.6.11.post3", "0.7.0.post1"])
@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
@pytest.mark.parametrize("heads", list(market.MODELS.values()))
def test_requested_xqa_shapes_are_source_supported(version, capability, heads):
    assert market.xqa_unsupported(version, capability, *heads) is None


@pytest.mark.parametrize(
    "version,capability,heads,page,reason",
    [
        ("unknown", (12, 0), (32, 8, 128), 16, "version"),
        ("0.7.0.post1", (8, 9), (32, 8, 128), 16, "SM120"),
        ("0.7.0.post1", (12, 0), (32, 8, 128), 1, "page size"),
        ("0.7.0.post1", (12, 0), (32, 8, 512), 16, "head dimension 16..256"),
        ("0.7.0.post1", (12, 0), (31, 8, 128), 16, "integral GQA"),
    ],
)
def test_xqa_unqualified_shapes_have_a_reason(version, capability, heads, page, reason):
    assert reason in market.xqa_unsupported(version, capability, *heads, page)


def test_footprint_counts_page_padding_once_across_providers():
    row = {"op": "fp8-decode", "model": "qwen3-8b", "B": 3, "KV": 17, "Q": 1}
    assert market.kv_footprint(row) == 2 * 3 * 32 * 8 * 128
    assert market.kv_footprint({**row, "op": "graph-prefill"}) == 2 * market.kv_footprint(row)
    assert market.engine_layers(100, 100, 1000) == 4  # strict >3x, when budget permits
    assert market.engine_layers(100, 100, 300) == 3
    assert market.engine_layers(10000, 100, 100000) == 64
    with pytest.raises(ValueError):
        market.engine_layers(100, 100, 199)


def test_round_timing_shuffles_seeded_order_warms_once_and_pairs_medians(monkeypatch):
    args = market.parse_args(["--rounds", "3", "--reps", "3"])
    calls, measured = [], []
    graphs = {
        name: SimpleNamespace(replay=lambda n=name: calls.append(n))
        for name in ("ours", "sglang", "ours_twin")
    }
    values = iter([9, 1, 3, 8, 2, 4, 7, 3, 5] * 3)

    def measure(replay, layers):
        assert layers == 7
        replay()
        measured.append(calls[-1])
        return next(values)

    monkeypatch.setattr(market.sweep, "measure", measure)
    result = market.time_graphs(graphs, 7, args, random.Random(11))
    expected = {name: [] for name in graphs}
    rng = random.Random(11)
    order = []
    for _ in range(3):
        names = list(graphs)
        rng.shuffle(names)
        order.extend(names)
        for name, median in zip(names, (3, 4, 5), strict=True):
            expected[name].append(median)
    assert result == expected
    assert calls == [name for name in order for _ in range(4)]
    assert measured == [name for name in order for _ in range(3)]


@pytest.mark.parametrize("op,query", [("fp8-decode", 1), ("graph-prefill", 3)])
def test_reference_handles_fp8_scales_padding_grouping_and_causality(monkeypatch, op, query):
    import torch

    monkeypatch.setitem(market.MODELS, "tiny", (4, 2, 4))
    monkeypatch.setitem(market.sweep.MODELS, "tiny", (4, 2, 4))
    row = {"op": op, "model": "tiny", "B": 2, "KV": 17, "Q": query}
    q, k, v, out = market.make_layers(row, 1, torch.device("cpu"), 42)[0]
    actual = market.reference_attention(row, (q, k, v, out))
    expected = torch.empty_like(q, dtype=torch.float32)
    scale = 0.5 if op == "fp8-decode" else 1.0
    for request in range(2):
        for token in range(query):
            visible = 17 - query + token + 1
            for head in range(4):
                keys = k.float().view(2, 32, 2, 4)[request, :visible, head // 2] * scale
                vals = v.float().view(2, 32, 2, 4)[request, :visible, head // 2] * scale
                weights = ((keys @ q[request * query + token, head].float()) / 2).softmax(0)
                expected[request * query + token, head] = weights @ vals
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["ours", "ours_twin", "ours_cc", "sglang"])
@pytest.mark.parametrize("op,query", [("fp8-decode", 1), ("graph-prefill", 3)])
def test_wrapper_contracts_use_graph_buffers_host_plans_and_float_scales(
    monkeypatch, name, op, query
):
    import torch

    from ctikernels.attention import inference

    if name == "ours_cc" and op == "graph-prefill":
        return
    wrapper = SimpleNamespace(plan=Mock(), run=Mock())
    constructor = Mock(return_value=wrapper)
    constructor.workspace_size.return_value = (0, 0)
    monkeypatch.setattr(inference, "FlashInferPagedAttention", constructor)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=constructor,
            BatchPrefillWithPagedKVCacheWrapper=constructor,
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(market, "SGLANG_WORKSPACE_BYTES", 0)
    row = {"op": op, "model": "qwen3-8b", "B": 2, "KV": 17, "Q": query}
    buffers = [(object(), object(), object(), object())]
    call, retained = market.make_wrapper_call(name, row, buffers, torch.device("cpu"))
    options = constructor.call_args.kwargs
    plan = wrapper.plan.call_args.kwargs
    assert options["use_cuda_graph"] is True and options["backend"] == "fa2"
    assert plan["page_size"] == 16 and plan["sm_scale"] == 128**-0.5
    if name != "sglang":
        assert options["dispatch"] == "device"
        assert options["use_tensor_cores"] is (op == "fp8-decode" and name != "ours_cc")
        assert plan["cu_seqlens_k"].tolist() == [0, 2, 4]
        assert plan["cu_seqlens_q"].tolist() == [0, query, 2 * query]
        assert plan["last_page_len"].tolist() == [1, 1]
        assert plan["dtype"] == torch.bfloat16
    elif op == "fp8-decode":
        assert options["use_tensor_cores"] is True
        assert plan["data_type"] == plan["kv_data_type"] == torch.float8_e4m3fn
        assert plan["q_data_type"] == torch.bfloat16
    else:
        assert options["qo_indptr_buf"].tolist() == [0, 3, 6]
        assert plan["causal"] is True
        assert plan["q_data_type"] == plan["kv_data_type"] == torch.bfloat16
    call()
    assert wrapper.run.call_args.args == ()  # public adapter requires keyword-only calls
    run = wrapper.run.call_args.kwargs
    assert run["q"] is buffers[0][0] and run["out"] is buffers[0][3]
    assert run["paged_kv_cache"] == buffers[0][1:3]
    if op == "fp8-decode":
        assert run["k_scale"] == run["v_scale"] == 0.5
        assert isinstance(run["k_scale"], float)
    else:
        assert "k_scale" not in run and "v_scale" not in run
    assert retained


def test_xqa_matches_engine_scales_layout_metadata_and_zeroed_workspace(monkeypatch):
    import torch

    xqa = Mock()
    monkeypatch.setitem(
        sys.modules, "flashinfer.decode", SimpleNamespace(xqa_batch_decode_with_kv_cache=xqa)
    )
    allocate = torch.zeros
    allocations = []

    def zeros(*size, **kwargs):
        allocations.append((size[0] if len(size) == 1 else size, kwargs))
        return allocate(8 if len(size) == 1 else size, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    row = {"op": "fp8-decode", "model": "qwen3.5", "B": 2, "KV": 17, "Q": 1}
    buffers = [(object(), object(), object(), object()) for _ in range(2)]
    call, retained = market.make_xqa_call(row, buffers, torch.device("cpu"))
    call()
    assert xqa.call_count == 2
    assert (394 * 1024**2, {"dtype": torch.uint8, "device": torch.device("cpu")}) in allocations
    arguments = xqa.call_args.kwargs
    assert arguments["bmm1_scale"] == 0.03125 and arguments["bmm2_scale"] == 0.5
    assert arguments["kv_cache"] == buffers[1][1:3]
    assert arguments["query"] is buffers[1][0] and arguments["out"] is buffers[1][3]
    assert arguments["kv_layout"] == "NHD" and arguments["q_len_per_req"] == 1
    # vLLM capture: max_seq_len = max_model_len, block table cdiv(max_model_len, 16) wide.
    table = arguments["block_tables"]
    assert table.shape == (2, 262144 // 16)
    assert table[:, :2].tolist() == [[0, 1], [2, 3]] and not table[:, 2:].any()
    assert arguments["seq_lens"].tolist() == [17, 17]
    assert arguments["max_seq_len"] == 262144
    assert torch.count_nonzero(arguments["workspace_buffer"]) == 0
    assert retained


def test_fa2_prefill_abi(monkeypatch):
    import torch

    operation = Mock()
    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", operation, raising=False)
    row = {"op": "graph-prefill", "model": "qwen3.5", "B": 2, "KV": 17, "Q": 3}
    buffers = [(object(), object(), object(), object())]
    call, _ = market.make_vllm_prefill_call(row, buffers, torch.device("cpu"))
    call()
    args = operation.call_args.args
    assert args[:4] == buffers[0]
    assert args[4].tolist() == [0, 3, 6] and args[6].tolist() == [17, 17]
    assert args[8].tolist() == [[0, 1], [2, 3]]
    assert args[10:16] == (3, 17, 0.0, 256**-0.5, False, True)


def test_setup_failure_still_emits_one_host_and_one_line_per_row(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    market.main(["--ops", "fp8-decode", "--models", "qwen3-8b", "--batches", "1", "--kv", "17,32"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 3 and records[0]["type"] == "host"
    assert all(record["type"] == "row" and record["status"] == "error" for record in records[1:])
    assert "hostname" not in records[0] and "path" not in records[0]


def test_wrong_provider_is_not_timed_or_used_in_market_ratios(monkeypatch):
    import torch

    monkeypatch.setitem(market.MODELS, "tiny", (4, 2, 4))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(
        market, "reference_attention", lambda row, layer: torch.zeros_like(layer[0]).float()
    )
    captured = {}

    def wrapper(name, row, buffers, device):
        captured[name] = buffers

        def call():
            for _, _, _, out in buffers:
                out.fill_(1 if name == "sglang" else 0)

        return call, ()

    def capture(call):
        call()
        return SimpleNamespace(replay=call)

    def time(graphs, layers, args, rng):
        assert set(graphs) == {"ours", "ours_twin", "ours_nosplit", "ours_eager"}
        assert layers == 2
        return {"ours": [2, 4], "ours_twin": [2, 4]}

    monkeypatch.setattr(market, "make_wrapper_call", wrapper)
    monkeypatch.setattr(market.sweep, "graph_capture", capture)
    monkeypatch.setattr(market, "time_graphs", time)
    row = {"op": "graph-prefill", "model": "tiny", "B": 1, "KV": 17, "Q": 3}
    result = market.run_row(
        row,
        market.parse_args(["--rounds", "2"]),
        torch.device("cpu"),
        1,
        "0.7.0.post1",
        None,
        random.Random(1),
    )
    assert result["providers"]["sglang"]["correct"] is False
    assert result["providers"]["sglang"]["max_error"] == 1
    assert result["providers"]["sglang"]["us"] is None
    assert result["providers"]["vllm"] is None
    assert result["reasons"]["vllm"] == "optional FA2 library not supplied"
    assert result["ratios"]["ours/max-market-faster"] is None
    assert result["ratios"]["ours/ours_twin"]["ci95"] == [1, 1]
    # Per-layer caches differ; providers share those exact tensors, with separate Q/out.
    assert captured["ours"][0][1] is not captured["ours"][1][1]
    for ours, twin in zip(captured["ours"], captured["ours_twin"], strict=True):
        assert ours[1] is twin[1] and ours[2] is twin[2]
        assert ours[0] is not twin[0] and ours[3] is not twin[3]
        torch.testing.assert_close(ours[0], twin[0])
