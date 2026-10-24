# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only benchmark protocol checks; no wheel download or GPU access."""

import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import _moe_lora_vllm as adapter
from benchmark.kernel_tuning import moe_lora_market as market


def test_default_and_quick_grids():
    assert market.parse_args([]).samples == 21
    assert market.parse_args([]).routing == "softmax"
    assert market.parse_args([]).ep_rank == 0
    assert market.parse_args(["--samples", "7"]).samples == 7
    default = market.rows(market.parse_args([]))
    assert len(default) == 140
    assert len(set(default)) == 140
    assert sum(isinstance(row, market.SharedRow) for row in default) == 70
    assert {row.tokens for row in default} == set(market.DEFAULT_TOKENS)
    quick = market.rows(market.parse_args(["--quick", "--ep", "2", "--kinds", "routed"]))
    assert len(quick) == 15
    assert all(row.fields()["E_local"] == 128 for row in quick)
    assert quick[0].fields()["top_k"] == 8


def test_explicit_grid_overrides_quick_and_deduplicates():
    args = market.parse_args(
        ["--quick", "--tokens", "2,2,8", "--ranks", "8,128,8", "--kinds", "routed"]
    )
    assert [(r.tokens, r.rank) for r in market.rows(args)] == [(2, 8), (2, 128), (8, 8), (8, 128)]


@pytest.mark.parametrize(
    "argv",
    [
        ["--tokens", "0"],
        ["--tokens", "1,"],
        ["--tokens", "bad"],
        ["--ranks", "7"],
        ["--ranks", "256"],
        ["--ep", "3"],
        ["--ep-rank", "1"],
        ["--ep", "2", "--ep-rank", "2"],
        ["--ep-rank", "-1"],
        ["--routing", "unknown"],
        ["--sample-ms", "2.49"],
        ["--sample-ms", "nan"],
        ["--samples", "4"],
        ["--pool-size", "1"],
        ["--scaling", "inf"],
        ["--kinds", ""],
        ["--kinds", "shared,"],
        ["--kinds", "unknown"],
        ["--shared-intermediate", "0"],
        ["--shared-intermediate", "-1"],
    ],
)
def test_invalid_protocol_rejected(argv):
    with pytest.raises(SystemExit):
        market.parse_args(argv)


@pytest.mark.parametrize("routing", ["softmax", "skewed", "narrow"])
@pytest.mark.parametrize("ep,ep_rank", [(1, 0), (2, 0), (2, 1)])
def test_routing_options_in_all_row_fields(routing, ep, ep_rank, capsys):
    argv = ["--routing", routing, "--ep", str(ep), "--ep-rank", str(ep_rank)]
    args = market.parse_args(argv)
    assert args.routing == routing and args.ep_rank == ep_rank
    assert market.main([*argv, "--tokens", "1", "--ranks", "8", "--dry-run"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 2
    assert all(row["routing"] == routing and row["ep_rank"] == ep_rank for row in records)


@pytest.mark.parametrize("routing", ["softmax", "skewed", "narrow"])
@pytest.mark.parametrize("ep,ep_rank", [(1, 0), (2, 0), (2, 1)])
def test_routing_shapes_local_mapping_and_determinism(routing, ep, ep_rank):
    kwargs = {"routing": routing, "ep": ep, "ep_rank": ep_rank}
    result = market.make_routing(256, 32, 4, generator=torch.Generator().manual_seed(7), **kwargs)
    repeated = market.make_routing(256, 32, 4, generator=torch.Generator().manual_seed(7), **kwargs)
    for actual, expected in zip(result, repeated, strict=True):
        assert actual.shape == (256, 4) and actual.is_contiguous()
        assert torch.equal(actual, expected)
    ids, local_ids, weights = result
    assert ids.dtype == local_ids.dtype == torch.int32
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights.sum(-1), torch.ones(256))
    local = 32 // ep
    start = ep_rank * local
    is_local = (ids >= start) & (ids < start + local)
    assert torch.equal(local_ids, torch.where(is_local, ids - start, -1))
    assert (weights > 0).all()
    if routing == "narrow":
        touched = ids.unique()
        assert touched.numel() == 8
        assert ((touched >= start) & (touched < start + local)).sum() == round(8 / ep)


def test_default_routing_matches_previous_generator_and_rng_stream():
    generator = torch.Generator().manual_seed(2026)
    previous = torch.Generator().manual_seed(2026)
    for _ in range(3):
        # Interleave activation/cache draws as in the rotating input pool.
        assert torch.equal(
            torch.randn((16, 8), dtype=torch.bfloat16, generator=generator),
            torch.randn((16, 8), dtype=torch.bfloat16, generator=previous),
        )
        ids, local_ids, weights = market.make_routing(16, 32, 4, generator=generator, ep=2)
        logits = torch.randn((16, 32), generator=previous)
        old_weights, old_ids = logits.softmax(-1).topk(4, dim=-1)
        old_weights = (old_weights / old_weights.sum(-1, keepdim=True)).contiguous()
        old_ids = old_ids.to(torch.int32)
        assert torch.equal(ids, old_ids)
        assert torch.equal(weights, old_weights)
        assert torch.equal(local_ids, torch.where(old_ids < 16, old_ids, -1))
        assert torch.equal(generator.get_state(), previous.get_state())


def test_skewed_routing_uses_permuted_zipf_logits():
    generator = torch.Generator().manual_seed(42)
    ids, _, weights = market.make_routing(128, 32, 4, generator=generator, routing="skewed")
    reference = torch.Generator().manual_seed(42)
    logits = torch.randn((128, 32), generator=reference)
    perm = torch.randperm(32, generator=reference)
    logits += 2.0 * (1 / torch.arange(1, 33, dtype=torch.float32)).log()[perm]
    expected_weights, expected_ids = logits.softmax(-1).topk(4, dim=-1)
    assert torch.equal(ids, expected_ids.to(torch.int32))
    assert torch.equal(weights, expected_weights / expected_weights.sum(-1, keepdim=True))


@pytest.mark.parametrize("ep,ep_rank", [(1, 0), (2, 0), (2, 1)])
def test_vllm_expert_map_selects_local_rank_on_cpu(ep, ep_rank):
    model = market.ModelSpec("test", 8, 8, 4, 2, False, (2,), (8,))
    expert = SimpleNamespace(workspace_shapes=lambda *a: ((1,), (1,), (2, 8)))
    runtime = SimpleNamespace(
        module=lambda name: SimpleNamespace(MoEActivation=SimpleNamespace(SILU="silu")),
        config=SimpleNamespace(
            FusedMoEParallelConfig=SimpleNamespace(make_no_parallel=SimpleNamespace),
            FusedMoEConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            RoutingMethodType=SimpleNamespace(Renormalize="renormalize"),
            FUSED_MOE_UNQUANTIZED_CONFIG=None,
        ),
        experts=SimpleNamespace(TritonExperts=lambda *a: expert),
        base=SimpleNamespace(try_get_optimal_moe_config=lambda *a: {}),
    )
    kwargs = {"ep_rank": ep_rank} if ep_rank else {}
    layer = adapter.RoutedLayer(
        runtime,
        torch.zeros(2, 8),
        torch.zeros(4 // ep, 16, 8),
        torch.zeros(4 // ep, 8, 8),
        torch.zeros(2, 2, dtype=torch.int32),
        torch.ones(2, 2),
        (),
        0.5,
        ep,
        model=model,
        base=True,
        **kwargs,
    )
    if ep == 1:
        assert layer.expert_map is None
    else:
        expected = [0, 1, -1, -1] if ep_rank == 0 else [-1, -1, 0, 1]
        assert torch.equal(layer.expert_map, torch.tensor(expected, dtype=torch.int32))


def test_routing_local_ids_partition_fp32_oracle():
    generator = torch.Generator().manual_seed(10)
    x = torch.randn((8, 2), generator=generator)
    w13 = torch.randn((4, 4, 2), generator=generator)
    w2 = torch.randn((4, 2, 2), generator=generator)
    factors = [torch.randn((4, 2, 2), generator=generator) for _ in range(6)]
    ids, _, weights = market.make_routing(8, 4, 2, generator=torch.Generator().manual_seed(7))
    expected = market.fp32_oracle(x, w13, w2, ids, weights, factors, 0.5)
    partials = []
    for ep_rank in (0, 1):
        _, local_ids, local_weights = market.make_routing(
            8, 4, 2, generator=torch.Generator().manual_seed(7), ep=2, ep_rank=ep_rank
        )
        selection = slice(ep_rank * 2, (ep_rank + 1) * 2)
        partials.append(
            market.fp32_oracle(
                x,
                w13[selection],
                w2[selection],
                local_ids,
                local_weights,
                [factor[selection] for factor in factors],
                0.5,
            )
        )
    torch.testing.assert_close(sum(partials), expected)


@pytest.mark.parametrize(
    "tokens,rank,ep,naive,small",
    [
        (1, 8, 1, True, True),
        (2, 64, 1, True, True),
        (4, 32, 1, True, True),
        (4, 64, 1, True, False),
        (4, 128, 1, True, False),
        (8, 8, 1, False, False),
        (1, 8, 2, False, False),
        (2, 16, 2, False, False),
    ],
)
def test_real_punica_dispatch_boundaries(tokens, rank, ep, naive, small):
    assert adapter.path_flags(tokens, rank, ep) == {
        "naive": naive,
        "small_batch": small,
        "one_shot": not small,
    }


def test_ratio_is_paired_and_selects_one_fastest_schedule():
    samples = {"ours": [1, 10, 100], "vllm": [1, 100, 120], "vllm_dual": [2, 200, 240]}
    summary = market.paired_summary(samples, seed=5)
    assert summary["best_vllm"] == "vllm"
    assert summary["ratio"] == 1.2  # Ratio of medians would incorrectly be 10.
    assert summary["interval"][0] <= 1.2 <= summary["interval"][1]
    assert summary["verdict"] == "PASS"
    assert summary == market.paired_summary(samples, seed=5)
    samples["vllm_dual"] = [0.5, 5, 50]
    assert market.paired_summary(samples, seed=5)["ratio"] == 0.5
    assert market.paired_summary(samples, seed=5)["verdict"] == "FAIL"


def test_missing_schedule_cannot_claim_strongest_form_pass():
    assert market.paired_summary({"ours": [1], "vllm": [2]}, seed=1)["verdict"] == "ERROR"
    with pytest.raises(ValueError):
        market.paired_summary({"ours": [1], "vllm": [2, 3], "vllm_dual": [4]}, seed=1)
    with pytest.raises(ValueError):
        market.paired_summary({"ours": [0], "vllm": [2], "vllm_dual": [4]}, seed=1)


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.125, "PASS"), (1.0, "PASS"), (0.99, "TIE"), (0.875, "TIE"), (0.874, "FAIL")],
)
@pytest.mark.parametrize("twin_ratio", [0.875, 1.125])
def test_noise_verdict_boundaries_and_symmetric_bias(ratio, expected, twin_ratio):
    result = market.noise_summary(ratio, [twin_ratio] * 21, [1.0] * 21, seed=7)
    assert result == {
        "aa_ratio": twin_ratio,
        "aa_interval": [twin_ratio, twin_ratio],
        "epsilon": 0.125,
        "verdict_eps": expected,
    }


def test_noise_uses_interval_half_width_when_larger_than_bias():
    result = market.noise_summary(0.9, [0.75, 1.0, 1.25], [1.0] * 3, seed=5)
    assert result["aa_ratio"] == 1.0
    assert result["aa_interval"] == [0.75, 1.25]
    assert result["epsilon"] == 0.25
    assert result["verdict_eps"] == "TIE"
    assert result == market.noise_summary(0.9, [0.75, 1.0, 1.25], [1.0] * 3, seed=5)


def test_noise_ratio_preserves_pairing_and_twin_over_best_direction():
    result = market.noise_summary(0.99, [1.0, 100.0, 120.0], [1.0, 10.0, 100.0], seed=5)
    assert result["aa_ratio"] == 1.2  # Ratio of medians would be 10.
    assert result["aa_interval"][0] <= 1.2 <= result["aa_interval"][1]


@pytest.mark.parametrize("ratio,expected", [(1.0, "PASS"), (0.999, "FAIL")])
def test_zero_noise_keeps_strict_verdict(ratio, expected):
    result = market.noise_summary(ratio, [1.0] * 21, [1.0] * 21, seed=7)
    assert result["epsilon"] == 0
    assert result["verdict_eps"] == expected


def test_noise_tie_does_not_change_strict_verdict():
    samples = {
        "ours": [1 / 0.99] * 21,
        "vllm": [1.0] * 21,
        "vllm_dual": [2.0] * 21,
        "vllm_twin": [1.02] * 21,
    }
    result = market.paired_summary(samples, seed=7)
    assert result["verdict"] == "FAIL"
    assert result["verdict_eps"] == "TIE"
    assert result["epsilon"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "twin,baseline",
    [([], []), ([1], [1, 2]), ([0], [1]), ([1], [-1]), ([math.nan], [1]), ([1], [math.inf])],
)
def test_noise_rejects_invalid_pairs(twin, baseline):
    with pytest.raises(ValueError):
        market.noise_summary(1.0, twin, baseline, seed=7)


def test_interleaving_preserves_one_measurement_per_candidate_per_pair():
    names = ["ours", "vllm", "vllm_dual", "ours_base"]
    orders = market.sample_orders(names, 11, 2026)
    assert orders == market.sample_orders(names, 11, 2026)
    assert len({tuple(order) for order in orders}) > 1
    assert all(Counter(order) == Counter(names) for order in orders)


def test_twins_share_one_shuffled_timing_slot_per_round():
    originals = ["ours", "vllm", "vllm_dual", "ours_base"]
    twins = ["vllm_twin", "vllm_dual_twin"]
    orders = market.sample_orders([*originals, *twins], 21, 2026, twins=twins)
    assert orders == market.sample_orders([*originals, *twins], 21, 2026, twins=twins)
    assert sorted(sum(name in order for order in orders) for name in twins) == [10, 11]
    assert all(len(order) == len(originals) + 1 and set(originals) < set(order) for order in orders)
    assert len({order.index(next(name for name in twins if name in order)) for order in orders}) > 1


@pytest.mark.parametrize("best", ["vllm", "vllm_dual"])
def test_aa_uses_final_winner_and_matching_rounds(best):
    other = "vllm_dual" if best == "vllm" else "vllm"
    samples = {
        "ours": [2, 20, 200, 2000, 20000],
        best: [1, 10, 100, 1000, 10000],
        other: [3, 30, 300, 3000, 30000],
        best + "_twin": [11.25, 1125],
        other + "_twin": [300, 30000, 3000000],
    }
    indices = {name: list(range(5)) for name in ("ours", best, other)}
    indices.update({best + "_twin": [1, 3], other + "_twin": [0, 2, 4]})
    result = market.paired_summary(samples, seed=7, sample_indices=indices)
    assert result["best_vllm"] == best
    assert result["aa_twin"] == best + "_twin"
    assert result["aa_sample_count"] == 2
    assert result["ratio"] == 0.5
    assert result["aa_ratio"] == 1.125
    assert result["epsilon"] == 0.125
    assert result["verdict"] == result["verdict_eps"] == "FAIL"
    del samples[best + "_twin"]
    missing = market.paired_summary(samples, seed=7, sample_indices=indices)
    assert missing["verdict"] == "FAIL"
    assert missing["verdict_eps"] is None
    assert missing["aa_ratio"] is None and missing["epsilon"] is None


@pytest.mark.parametrize("model_name", tuple(market.MODELS))
@pytest.mark.parametrize("ep_rank", [0, 1])
def test_routed_candidates_construct_independent_twin_layers(monkeypatch, model_name, ep_rank):
    import ctikernels.moe.lora as ops
    import ctikernels.moe_impl as base_ops

    instances = []

    class Layer:
        def __init__(self, *args, model, ep_rank, dual=False, base=False, fp8_scales=None):
            assert model is market.MODELS[model_name]
            assert fp8_scales is None
            assert ep_rank == row.ep_rank
            self.dual, self.base = dual, base
            self.workspace13, self.workspace2, self.output = object(), object(), object()
            self.ctx = SimpleNamespace(w13_lora_b_stacked=object(), w2_lora_b_stacked=object())
            self.base_config = {}
            instances.append(self)

        def __call__(self, x, ids, weights):
            return self, x, ids, weights

    monkeypatch.setattr(adapter, "RoutedLayer", Layer)
    monkeypatch.setattr(base_ops, "moe_align_block_size", lambda *a: (None, None, None))
    monkeypatch.setattr(ops, "MoELoRAWeights", SimpleNamespace(separate=lambda *a: None))
    pool = [
        (torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), None)
        for _ in range(2)
    ]
    row = market.Row(1, 8, 2, market.MODELS[model_name], ep_rank=ep_rank)
    candidates, errors, _ = market.build_candidates(
        None,
        row,
        market.parse_args(["--model", model_name]),
        (None, None, (None,) * 6, pool),
    )
    assert not errors
    assert len(instances) == 5
    single, dual, base, single_twin, dual_twin = instances
    for name, layer in (("vllm_twin", single_twin), ("vllm_dual_twin", dual_twin)):
        assert layer.dual is (name == "vllm_dual_twin")
        assert not layer.base
        for attribute in ("workspace13", "workspace2", "output", "ctx"):
            assert all(
                getattr(layer, attribute) is not getattr(other, attribute)
                for other in instances
                if other is not layer
            )
        for call, entry in zip(candidates[name], pool, strict=True):
            actual_layer, x, ids, weights = call()
            assert actual_layer is layer
            assert x is entry[0] and ids is entry[1] and weights is entry[3]
    assert single.workspace13 is dual.workspace13 is base.workspace13


@pytest.mark.parametrize("best", ["vllm", "vllm_dual"])
@pytest.mark.parametrize("fault", [None, "capture", "sample"])
def test_routed_row_times_twins_interleaved_and_handles_failure(monkeypatch, best, fault):
    row, args = market.Row(1, 8, 1), market.parse_args([])
    data = (None, None, (), [(None, None, None, None, None)] * args.pool_size)
    monkeypatch.setattr(market, "make_inputs", lambda *a: data)
    monkeypatch.setattr(market, "fp32_oracle", lambda *a: torch.ones(1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (0, 0))
    names = ["ours", "vllm", "vllm_dual", "vllm_base", "ours_base", "ours_lora", "vllm_lora"]
    candidates = {
        name: [lambda: torch.ones(1) for _ in range(args.pool_size)]
        for name in (*names, "vllm_twin", "vllm_dual_twin")
    }
    monkeypatch.setattr(market, "build_candidates", lambda *a: (candidates, {}, {}))
    captured, timed = [], []

    class Graph:
        def __init__(self, calls, target_ms):
            self.name = next(name for name, entries in candidates.items() if entries is calls)
            self.calls, self.repeats = len(calls), 1
            if self.name == best + "_twin" and fault == "capture":
                raise RuntimeError
            captured.append(self.name)

        def sample(self):
            assert len(captured) == len(candidates) - (fault == "capture")
            timed.append(self.name)
            if self.name == best + "_twin" and fault == "sample":
                raise RuntimeError
            return {"ours": 1 / 0.99, best: 1, best + "_twin": 1.01}.get(self.name, 2)

    monkeypatch.setattr(market, "GraphSample", Graph)
    result = market.benchmark_row(SimpleNamespace(provenance=lambda: {}), row, args)
    assert result["best_vllm"] == best
    assert result["verdict"] == "FAIL"
    assert all(len(order) == len(names) + 1 for order in result["sample_orders"])
    assert all(len(result["samples_us"][name]) == 21 for name in names)
    assert not result["correctness_failed"]
    if fault:
        assert result["status"] == "partial"
        assert result["aa_ratio"] is None and result["verdict_eps"] is None
        assert best + "_twin" not in result["samples_us"]
        assert result["errors"][best + "_twin"]["phase"] == (
            "warmup/capture" if fault == "capture" else "sample"
        )
    else:
        assert result["status"] == "ok"
        assert timed == [name for order in result["sample_orders"] for name in order]
        assert result["aa_ratio"] == 1.01
        assert result["verdict_eps"] == "TIE"
        assert result["aa_sample_count"] in (10, 11)
        for twin in ("vllm_twin", "vllm_dual_twin"):
            assert result["sample_indices"][twin] == [
                i for i, order in enumerate(result["sample_orders"]) if twin in order
            ]


def test_graph_sample_enforces_duration_floor_without_gpu():
    graph = object.__new__(market.GraphSample)
    graph.calls, graph.repeats, graph.target_ms = 8, 1, 2.5
    durations = iter([0.5, 3.0])
    graph.elapsed = lambda repeats: next(durations)
    assert graph.sample() == 3000 / (6 * 8)
    assert graph.repeats == 6


@pytest.mark.parametrize(
    "ours,baseline,failed",
    [
        (0.005, 0.004, False),
        (0.009, 0.004, True),
        (0.021, 0.02, True),
        (math.nan, 0.004, True),
        (0.0, 0.0, False),
        (0.001, 0.0, True),
    ],
)
def test_correctness_gate(ours, baseline, failed):
    assert market.correctness_flag(ours, baseline) is failed


def test_platform_shim_answers_from_torch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert adapter.TorchPlatform.get_device_capability() is None
    assert not adapter.TorchPlatform.has_device_capability(90)
    assert adapter.TorchPlatform.get_device_name() == "CPU"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    assert adapter.TorchPlatform.has_device_capability(120)
    assert adapter.TorchPlatform.has_device_capability((12, 1))
    assert not adapter.TorchPlatform.has_device_capability(122)
    assert adapter.TorchPlatform.is_device_capability(121)
    assert not adapter.TorchPlatform.is_device_capability(120)


def test_off_path_dependencies_fail_closed():
    with pytest.raises(NotImplementedError, match="Off-path"):
        adapter.unavailable()


def test_registration_shim_preserves_function_identity(monkeypatch):
    def operation(value):
        return value + 1

    # Isolate torch.ops attribute changes from the rest of the CPU suite.
    monkeypatch.setattr(torch.ops.vllm, "market_test_op", None, raising=False)
    adapter.direct_register_custom_op("market_test_op", operation, mutates_args=[])
    assert torch.ops.vllm.market_test_op is operation
    assert torch.ops.vllm.market_test_op(4) == 5


def test_file_sha256_streams_bytes(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    assert (
        adapter.file_sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_dry_run_outputs_identical_jsonl_to_stdout_and_file(tmp_path, capsys):
    output = tmp_path / "rows.jsonl"
    assert (
        market.main(
            ["--quick", "--dry-run", "--ep", "2", "--kinds", "routed", "--output", str(output)]
        )
        == 0
    )
    lines = capsys.readouterr().out
    assert lines == output.read_text()
    records = [json.loads(line) for line in lines.splitlines()]
    assert len(records) == 15
    assert all(row["status"] == "dry_run" and row["ep"] == 2 for row in records)


@pytest.mark.parametrize("model", tuple(market.MODELS))
def test_cpu_run_reports_each_row_without_attempting_wheel_load(monkeypatch, capsys, model):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(adapter, "WheelRuntime", lambda *a, **kw: pytest.fail("wheel load on CPU"))
    assert market.main(["--model", model, "--tokens", "1,2", "--ranks", "16"]) == 1
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    has_shared = market.MODELS[model].shared
    assert [record["T"] for record in records] == ([1, 2, 1, 2] if has_shared else [1, 2])
    assert [record.get("kind") for record in records] == (
        [None, None, "shared", "shared"] if has_shared else [None, None]
    )
    assert records[0]["shared"] == ("reported separately (kind=shared)" if has_shared else "none")
    assert all(record["errors"] == {"runtime_load": "RuntimeError"} for record in records)


def test_oracle_handles_remote_experts_and_routed_weights():
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    w13 = torch.ones(1, 16, 8, dtype=torch.bfloat16) / 8
    w2 = torch.eye(8, dtype=torch.bfloat16).unsqueeze(0)
    factors = [torch.zeros(1, 8, 8, dtype=torch.bfloat16) for _ in range(6)]
    ids = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    oracle = market.fp32_oracle(x, w13, w2, ids, weights, factors, 0.5)
    expected = torch.zeros(2, 8)
    expected[0] = torch.nn.functional.silu(torch.tensor(1.0)) * 0.25
    torch.testing.assert_close(oracle, expected)


def test_shared_grid_is_independent_of_ep_and_preserves_routed_fields(capsys):
    assert market.Row(1, 8, 2).fields() == {
        "model": "qwen3.5-35b-a3b",
        "T": 1,
        "rank": 8,
        "ep": 2,
        "routing": "softmax",
        "ep_rank": 0,
        "H": 2048,
        "I": 512,
        "E": 256,
        "E_local": 128,
        "top_k": 8,
        "dtype": "bfloat16",
    }
    argv = ["--tokens", "2,8", "--ranks", "8", "--kinds", "shared,shared", "--ep", "2"]
    assert market.rows(market.parse_args(argv)) == [
        market.SharedRow(2, 8, 512),
        market.SharedRow(8, 8, 512),
    ]
    assert market.main([*argv, "--shared-intermediate", "256", "--dry-run"]) == 0
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records == [dict(market.SharedRow(t, 8, 256).fields(), status="dry_run") for t in (2, 8)]


def test_shared_paired_ratio_requires_both_candidates():
    kwargs = {"seed": 7, "ours": "ours_shared_lora", "baselines": ("vllm_shared_lora",)}
    samples = {"ours_shared_lora": [1, 10, 100], "vllm_shared_lora": [1, 100, 120]}
    result = market.paired_summary(samples, **kwargs)
    assert result["best_vllm"] == "vllm_shared_lora"
    assert result["ratio"] == 1.2
    assert result["interval"][0] <= result["ratio"] <= result["interval"][1]
    assert result == market.paired_summary(samples, **kwargs)
    del samples["vllm_shared_lora"]
    assert market.paired_summary(samples, **kwargs)["verdict"] == "ERROR"


def test_shared_oracle_has_distinct_slices_and_fixed_down_input():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    hidden = torch.tensor([[5], [-2]], dtype=torch.bfloat16)
    factors = (
        torch.tensor([[[1, 0]], [[0, 1]]], dtype=torch.bfloat16),
        torch.tensor([[[2]], [[3]]], dtype=torch.bfloat16),
        torch.tensor([[4]], dtype=torch.bfloat16),
        torch.tensor([[1], [-1]], dtype=torch.bfloat16),
    )
    gate_up, down = market.shared_fp32_oracle(x, hidden, factors, 0.5)
    torch.testing.assert_close(gate_up, torch.tensor([[1.0, 3.0], [3.0, 6.0]]))
    torch.testing.assert_close(down, torch.tensor([[10.0, -10.0], [-4.0, 4.0]]))


def test_shared_input_pool_rotates_both_fixed_inputs():
    row = market.SharedRow(2, 8, 16)
    factors, pool = market.make_shared_inputs(row, 3, 2026, device="cpu")
    assert [tuple(t.shape) for t in factors] == [
        (2, 8, row.model.hidden),
        (2, 16, 8),
        (8, 16),
        (row.model.hidden, 8),
    ]
    assert all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in factors)
    assert all(x.shape == (2, row.model.hidden) and h.shape == (2, 16) for x, h in pool)
    assert all(not torch.equal(pool[0][i], pool[1][i]) for i in (0, 1))
    factors2, pool2 = market.make_shared_inputs(row, 3, 2026, device="cpu")
    for actual, expected in zip((*factors, *pool[0]), (*factors2, *pool2[0]), strict=True):
        torch.testing.assert_close(actual, expected)


def test_shared_candidates_prescale_only_vllm_and_clear_destinations(monkeypatch):
    import ctikernels.moe.lora as ops

    row = market.SharedRow(2, 8, 16)
    data = market.make_shared_inputs(row, 2, 123, device="cpu")
    originals = tuple(value.clone() for value in data[0])
    args = market.parse_args(["--scaling", "0.5"])
    mapping, calls = [], []

    class Metadata:
        @staticmethod
        def make(adapters, tokens, device):
            assert adapters == 1 and tokens == row.tokens and device.type == "cpu"
            return SimpleNamespace(prepare_tensors=lambda value: mapping.append(value))

    class Punica:
        def add_lora_linear(self, y, x, a, b, scale, slices, **kwargs):
            assert self.lora_config.specialize_active_lora is True
            assert scale == 1.0 and kwargs == {"add_inputs": True}
            assert not y.count_nonzero()
            index = 0 if len(slices) == 2 else 2
            expected_a = originals[0].unbind(0) if index == 0 else (originals[2],)
            expected_b = originals[1].unbind(0) if index == 0 else (originals[3],)
            for wa, wb, ea, eb in zip(a, b, expected_a, expected_b, strict=True):
                torch.testing.assert_close(wa, ea.unsqueeze(0))
                torch.testing.assert_close(wb, (eb * args.scaling).unsqueeze(0))
            assert slices == ((16, 16) if index == 0 else (row.model.hidden,))
            calls.append(("vllm", x))
            y.add_(1)

    def ours(y, x, factors, *, scaling, provider):
        assert scaling == args.scaling and provider == "cute"
        assert not y.count_nonzero()
        expected = data[0][:2] if factors[0].ndim == 3 else data[0][2:]
        assert all(actual is original for actual, original in zip(factors, expected, strict=True))
        calls.append(("ours", x))
        y.add_(1)

    monkeypatch.setattr(ops, "shared_lora_w13_", ours)
    monkeypatch.setattr(ops, "shared_lora_w2_", ours)
    runtime = SimpleNamespace(
        punica=SimpleNamespace(PunicaWrapperGPU=Punica),
        module=lambda name: SimpleNamespace(LoRAKernelMeta=Metadata),
    )
    candidates, errors = market.build_shared_candidates(runtime, row, args, data)
    assert not errors
    torch.testing.assert_close(mapping[0], torch.zeros(row.tokens, dtype=torch.int32))
    for name, callbacks in candidates.items():
        for i in (0, 1, 0):
            outputs = callbacks[i]()
            assert all(torch.all(value == 1) for value in outputs)
            assert calls[-2][1] is data[1][i][0]
            assert calls[-1][1] is data[1][i][1]
            assert calls[-1][0] == ("ours" if name.startswith("ours") else "vllm")
    for actual, original in zip(data[0], originals, strict=True):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("fault", [None, "gate_up", "down", "capture", "sample", "nonfinite"])
def test_shared_row_protocol_and_correctness_failures(monkeypatch, fault):
    row = market.SharedRow(2, 8, 16)
    args = market.parse_args(["--samples", "5", "--pool-size", "2"])
    data = market.make_shared_inputs(row, 2, 123, device="cpu")
    monkeypatch.setattr(market, "make_shared_inputs", lambda *a: data)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (0, 0))

    def candidate(entry, ours):
        output = list(market.shared_fp32_oracle(*entry, data[0], args.scaling))
        # A common small baseline error, plus a fault in only one projection.
        for i, projection in enumerate(("gate_up", "down")):
            output[i] = output[i] * (1.1 if ours and fault == projection else 1.001)
        if ours and fault == "nonfinite":
            output[0].fill_(math.nan)
        return tuple(output)

    candidates = {
        name: [lambda entry=entry, ours=ours: candidate(entry, ours) for entry in data[1]]
        for name, ours in (("vllm_shared_lora", False), ("ours_shared_lora", True))
    }
    monkeypatch.setattr(market, "build_shared_candidates", lambda *a: (candidates, {}))
    captured = []

    class Graph:
        def __init__(self, calls, target_ms):
            self.ours = calls is candidates["ours_shared_lora"]
            if self.ours and fault == "capture":
                raise RuntimeError
            captured.append((calls, target_ms))
            self.calls, self.repeats = len(calls), 3

        def sample(self):
            if self.ours and fault == "sample":
                raise RuntimeError
            return 1.0 if self.ours else 2.0

    monkeypatch.setattr(market, "GraphSample", Graph)
    result = market.benchmark_row(SimpleNamespace(provenance=lambda: {}), row, args)
    assert result["kind"] == "shared"
    assert all(target_ms == args.sample_ms and len(calls) == 2 for calls, target_ms in captured)
    assert len(result["sample_orders"]) == args.samples
    assert all(len(values) == args.samples for values in result["samples_us"].values())
    if fault in ("capture", "sample", "nonfinite"):
        assert result["status"] == "partial"
        assert result["verdict"] == "ERROR"
        assert "ours_shared_lora" not in result["samples_us"]
        assert (
            result["errors"]["ours_shared_lora"]["phase"]
            == {"capture": "warmup/capture", "sample": "sample", "nonfinite": "correctness"}[fault]
        )
    else:
        assert all(
            Counter(order) == Counter(candidates.keys()) for order in result["sample_orders"]
        )
        assert result["ratio"] == 2.0
        assert result["correctness_failed"] is (fault is not None)
        assert result["verdict"] == ("INCORRECT" if fault else "PASS")


@pytest.mark.parametrize(
    "model,intermediate,experts",
    [
        ("qwen3.5-35b-a3b", 512, 256),
        ("qwen3-30b-a3b", 768, 128),
    ],
)
@pytest.mark.parametrize("ep", [1, 2])
def test_model_dry_run_fields(model, intermediate, experts, ep, capsys):
    assert (
        market.main(
            ["--model", model, "--ep", str(ep), "--tokens", "1", "--ranks", "16", "--dry-run"]
        )
        == 0
    )
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    routed = records[0]
    assert routed == {
        "model": model,
        "T": 1,
        "rank": 16,
        "ep": ep,
        "routing": "softmax",
        "ep_rank": 0,
        "H": 2048,
        "I": intermediate,
        "E": experts,
        "E_local": experts // ep,
        "shared": "reported separately (kind=shared)" if model == "qwen3.5-35b-a3b" else "none",
        "top_k": 8,
        "dtype": "bfloat16",
        "status": "dry_run",
    }
    assert len(records) == (2 if model == "qwen3.5-35b-a3b" else 1)
    if len(records) == 2:
        assert records[1] == {
            "kind": "shared",
            "routing": "softmax",
            "ep_rank": 0,
            "model": model,
            "T": 1,
            "rank": 16,
            "H": 2048,
            "I": 512,
            "Is": 512,
            "E": 256,
            "E_local": 256,
            "top_k": 8,
            "dtype": "bfloat16",
            "status": "dry_run",
        }


def test_qwen3_defaults_and_explicit_overrides():
    args = market.parse_args(["--model", "qwen3-30b-a3b"])
    assert args.kinds == ("routed",)
    assert args.ranks == (16, 32, 64)
    assert args.tokens == tuple(2**power for power in range(16))
    assert len(market.rows(args)) == 48
    quick = market.parse_args(["--model", "qwen3-30b-a3b", "--quick"])
    assert quick.tokens == market.QUICK_TOKENS and quick.ranks == (16, 32, 64)
    override = market.parse_args(
        ["--model", "qwen3-30b-a3b", "--quick", "--tokens", "65536", "--ranks", "8,128"]
    )
    assert override.tokens == (65536,) and override.ranks == (8, 128)
    assert market.parse_args([]).model == "qwen3.5-35b-a3b"


@pytest.mark.parametrize("model", tuple(market.MODELS))
def test_market_accepts_all_supported_ranks(model):
    ranks = tuple(range(8, 129, 8))
    args = market.parse_args(["--model", model, "--ranks", ",".join(map(str, ranks))])
    assert args.ranks == ranks


@pytest.mark.parametrize("kinds", ["shared", "routed,shared"])
def test_qwen3_refuses_shared(kinds, capsys):
    with pytest.raises(SystemExit) as error:
        market.parse_args(["--model", "qwen3-30b-a3b", "--kinds", kinds])
    assert error.value.code == 2
    assert "qwen3-30b-a3b has no shared expert" in capsys.readouterr().err


@pytest.mark.parametrize("tokens,ep,naive", [(2, 1, True), (4, 1, False), (1, 2, False)])
def test_qwen3_punica_dispatch(tokens, ep, naive):
    assert adapter.path_flags(tokens, 16, ep, experts=128, top_k=8) == {
        "naive": naive,
        "small_batch": naive,
        "one_shot": not naive,
    }
