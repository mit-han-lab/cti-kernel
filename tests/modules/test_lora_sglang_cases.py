# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks for survey geometry, selection and decode routing."""

import pytest

from benchmark.kernel_tuning.lora_sglang import (
    CASES,
    batch_info,
    build_parser,
    chunk_size,
    kernel_mapping,
    selected_cases,
    sol_counts,
)


@pytest.mark.parametrize(
    "model,totals",
    [
        ("9b", (10240, 4096, 12352, 4096, 24576, 4096)),
        ("27b", (14336, 5120, 16480, 5120, 34816, 5120)),
        ("35b-a3b", (9216, 2048, 12352, 2048)),
    ],
)
def test_case_slices(model, totals):
    cases = CASES[model]
    assert tuple(sum(c.output_sizes) for c in cases) == totals
    assert tuple(len(c.output_sizes) for c in cases) == (3, 1, 6, 1, 2, 1)[: len(cases)]
    qkv, o, gdn, out = cases[:4]
    assert qkv.output_sizes[0] == 2 * o.input_size
    assert qkv.output_sizes[1] == qkv.output_sizes[2]
    assert gdn.output_sizes[:2] == (2048, 2048)
    assert gdn.output_sizes[2:4] == (out.input_size, out.input_size)
    assert gdn.output_sizes[4:] == (out.input_size // 128,) * 2
    assert o.output_sizes == out.output_sizes == (qkv.input_size,)
    if model != "35b-a3b":
        gate_up, down = cases[4:]
        assert gate_up.output_sizes == (down.input_size,) * 2
        assert down.output_sizes == (gate_up.input_size,)


def test_cli_defaults_and_selection():
    parser = build_parser()
    args = parser.parse_args(["--sglang-python", "comparator"])
    assert args.ranks == (16, 64, 128)
    assert args.tokens == (1, 2, 3, 4, 8, 16, 32, 64)
    assert args.pairs == 20
    assert args.sglang_max_chunk_size == 16
    assert len(selected_cases(args)) == 16
    args = parser.parse_args(
        [
            "--sglang-python",
            "comparator",
            "--model",
            "27b",
            "--ranks",
            "16,64",
            "--tokens",
            "1,3",
            "--pairs",
            "5",
            "--cases",
            "qkv_gate,in_proj",
            "--ceilings",
            "ceilings.json",
        ]
    )
    assert args.ranks == (16, 64) and args.tokens == (1, 3) and args.pairs == 5
    assert [c.name for _, c in selected_cases(args)] == ["qkv_gate", "in_proj"]
    args.cases = "typo"
    with pytest.raises(ValueError):
        selected_cases(args)


@pytest.mark.parametrize(
    "flag,value", [("--ranks", "0"), ("--tokens", "-1"), ("--pairs", "0"), ("--model", "unknown")]
)
def test_invalid_cli(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--sglang-python", "comparator", flag, value])


def test_decode_routing_and_dispatch():
    chunked = batch_info(64, 16, chunked=True, device="cpu")
    assert chunked.bs == 64 and chunked.num_segments == 4 and chunked.max_len == 16
    assert chunked.seg_indptr.tolist() == [0, 16, 32, 48] + [64] * 61
    assert chunked.weight_indices.shape == (64,)
    assert chunked.permutation.tolist() == list(range(64))
    assert chunked.req_seg_indptr.tolist() == list(range(65))
    merged = batch_info(64, 16, chunked=False, device="cpu")
    assert merged.bs == merged.num_segments == 1
    assert merged.seg_lens.tolist() == [64]
    assert merged.seg_indptr.tolist() == [0, 64]
    assert merged.lora_ranks.tolist() == [16] and merged.scalings.tolist() == [2.0]
    assert [chunk_size(b, 128) for b in (1, 64, 256)] == [16, 32, 128]
    for case in CASES["9b"]:
        last = kernel_mapping(case)["sgl_triton"][-1]
        assert last == (
            "gate_up_lora_b_fwd"
            if case.name == "gate_up"
            else "qkv_lora_b_fwd"
            if case.name in ("in_proj", "qkv_gate")
            else "sgemm_lora_b_fwd"
        )
    operand, traffic, flops = sol_counts(CASES["9b"][1], 16, 2)
    assert traffic - operand == 2 * 2 * 4096
    assert flops == 2 * 2 * (4096 * 4096 + 16 * (4096 + 4096))


def test_pair_protocol_ratio_and_order(monkeypatch):
    from benchmark.kernel_tuning import lora_sglang as bench

    visits = []
    graphs = {name: [(name + str(i), None) for i in range(2)] for name in ("ours", "sgl")}

    def sample(graph, copies):
        assert copies == 3
        visits.append(graph)
        return 10.0 if graph.startswith("ours") else 20.0

    monkeypatch.setattr(bench, "_sample", sample)
    result, _ = bench.compare(graphs, "ours", "sgl", copies=3, pairs=4)
    assert result["ratio_sgl_ours"] == 2.0
    assert result["ratio_iqr"] == (2.0, 2.0)
    assert result["eps"] == 0.0
    assert result["verdict"] == "faster"
    assert visits[:12] == ["ours0", "sgl0", "ours1", "sgl1"] * 3
    assert visits[12:] == [
        "ours0",
        "sgl0",
        "ours1",
        "sgl1",
        "ours0",
        "sgl1",
        "ours1",
        "sgl0",
        "ours1",
        "sgl1",
        "ours0",
        "sgl0",
        "ours1",
        "sgl0",
        "ours0",
        "sgl1",
    ]


def test_extended_cli_and_prefill_routing():
    args = build_parser().parse_args(
        [
            "--sglang-python",
            "comparator",
            "--ranks",
            "256,512",
            "--tokens",
            "128,8192",
            "--dtype",
            "float16",
            "--no-vllm",
            "--vllm-cache",
            "wheel-cache",
        ]
    )
    assert args.ranks == (256, 512) and args.tokens == (128, 8192)
    assert args.dtype == "float16" and args.no_vllm
    assert args.vllm_cache.name == "wheel-cache"
    chunked = batch_info(8192, 512, chunked=True, device="cpu", prefill=True)
    assert chunked.num_segments == 512
    assert chunked.weight_indices.numel() == 513
    assert chunked.req_seg_indptr.tolist() == [0, 8192]
    assert chunked.seg_indptr[512:].eq(8192).all()
    assert kernel_mapping(CASES["9b"][0])["vllm_punica"] == [
        "vllm.lora.ops.triton_ops.lora_shrink",
        "vllm.lora.ops.triton_ops.lora_expand",
    ]


def test_delta_gate_excludes_base_precision_and_rejects_missing_update():
    import torch

    from benchmark.kernel_tuning.lora_sglang import delta_correctness, delta_reference

    x = torch.tensor([[1.0, 2.0]])
    factors = (torch.tensor([[2.0, 1.0]]), torch.tensor([[1.0], [2.0]]))
    delta = delta_reference(x, factors, scaling=0.5)
    torch.testing.assert_close(delta, torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    base = torch.tensor([[100.0, 100.0]])
    base64 = base.double() + 10  # Deliberately inaccurate base: diagnostic only.
    result = delta_correctness(base + delta, base, delta, base64)
    assert result["ok"]
    assert not result["base_precision"]["base_only"]["ok"]
    assert not result["base_precision"]["full_output"]["ok"]
    assert not delta_correctness(base, base, delta, base64)["ok"]
    # A tiny missing update passes the full-output gate but must fail delta gate.
    result = delta_correctness(base, base, delta * 0.001, base64)
    assert result["rel_l2"] < 4e-3 and result["delta_rel_l2"] == 1.0
    assert not result["ok"]
    assert not delta_correctness(base * float("nan"), base, delta, base64)["ok"]


def test_all_variants_share_interleaved_samples(monkeypatch):
    from benchmark.kernel_tuning import lora_sglang as bench

    visits = []
    graphs = {name: [(name, None), (name, None)] for name in bench.VARIANTS}

    def sample(graph, copies):
        visits.append(graph)
        return 10.0 if graph.startswith("ours") else 20.0

    monkeypatch.setattr(bench, "_sample", sample)
    samples = bench.sample_variants(graphs, 2, 20)
    per_round = 2 * len(bench.VARIANTS)
    assert len(visits) == (20 + 3) * per_round
    assert all(len(values) == 20 for values in samples.values())
    # Every round visits each variant twice, with order changing across rounds.
    from collections import Counter

    expected = Counter(dict.fromkeys(bench.VARIANTS, 2))
    assert all(
        Counter(visits[i : i + per_round]) == expected for i in range(0, len(visits), per_round)
    )
    assert visits[:per_round] != visits[per_round : 2 * per_round]
    for ours, market in bench.COMPARISONS:
        result = bench._pair_result(samples, ours, market, 20)
        assert result["pairs"] == 20 and result["eps"] == 0
        assert result["verdict"] == "faster"


def test_faster_market_uses_median_and_marks_incomplete():
    from benchmark.kernel_tuning.lora_sglang import market_verdicts

    row = {
        "variants": {
            "sgl_csgmv": {"ok": True, "us": 3},
            "sgl_triton": {"ok": True, "us": 4},
            "vllm_punica": {"ok": True, "us": 2},
        },
        "comparisons": {
            "ours_prod_vs_vllm_punica": {"verdict": "tie"},
            "ours_cute_vs_vllm_punica": {"verdict": "faster"},
        },
    }
    market_verdicts(row)
    assert row["faster_market"] == "vllm_punica" and row["market_complete"]
    assert row["comparisons"]["ours_prod_vs_faster_market"]["verdict"] == "tie"
    assert row["comparisons"]["ours_cute_vs_faster_market"]["verdict"] == "faster"
    row["variants"]["vllm_punica"]["ok"] = False
    market_verdicts(row)
    assert row["faster_market"] == "sgl_csgmv" and not row["market_complete"]


def test_x925_pinning_reexec_and_already_restricted(monkeypatch):
    from benchmark.kernel_tuning import lora_sglang as bench

    cpuinfo = "processor : 0\nCPU part : 0xd81\n\nprocessor : 1\nCPU part : 0xd85\n\nprocessor : 2\nCPU part : 0xd85\n"
    assert bench.x925_cpus(cpuinfo) == {1, 2}
    monkeypatch.setattr(bench.Path, "read_text", lambda self: cpuinfo)
    monkeypatch.setattr(bench.os, "sched_getaffinity", lambda pid: {0, 1, 2})
    calls = []
    monkeypatch.setattr(bench.os, "sched_setaffinity", lambda pid, cpus: calls.append(cpus))

    class Reexec(Exception):
        pass

    def execute(executable, argv):
        assert argv[1:3] == ["-m", "benchmark.kernel_tuning.lora_sglang"]
        raise Reexec

    monkeypatch.setattr(bench.os, "execv", execute)
    with pytest.raises(Reexec):
        bench.pin_host()
    assert calls == [{1, 2}]
    monkeypatch.setattr(bench.os, "sched_getaffinity", lambda pid: {2})
    assert bench.pin_host() == [2]
    assert len(calls) == 1


def test_punica_uses_real_metadata_contract_and_add_inputs():
    from types import SimpleNamespace

    import torch

    from benchmark.kernel_tuning._lora_vllm import PunicaLinear

    calls = []

    class Metadata:
        @staticmethod
        def make(adapters, tokens, device):
            assert (adapters, tokens, device.type) == (1, 3, "cpu")
            return Metadata()

        def prepare_tensors(self, mapping):
            assert mapping.dtype == torch.int32 and mapping.tolist() == [0, 0, 0]

        def meta_args(self, tokens, specialize_active_lora):
            assert tokens == 3 and specialize_active_lora
            return ("routing",)

    def shrink(x, a, scratch, routing, scale):
        assert routing == "routing" and scale == 1.0
        assert scratch.dtype == torch.float32
        calls.append("shrink")
        scratch.copy_(torch.stack([x @ weight[0].T * scale for weight in a]))

    def expand(scratch, b, output, routing, *, offset_start, add_inputs):
        assert routing == "routing" and offset_start == 0 and add_inputs
        calls.append("expand")
        output.add_(
            torch.cat([low @ weight[0].T for low, weight in zip(scratch, b, strict=True)], -1)
        )

    x, weight = torch.ones(3, 4), torch.ones(5, 4)
    factors = (torch.ones(2, 4), torch.ones(2, 2), torch.ones(2, 4), torch.ones(3, 2))
    runtime = SimpleNamespace(metadata=Metadata, shrink=shrink, expand=expand)
    call = PunicaLinear(runtime, x, weight, factors, 2.0)
    torch.testing.assert_close(call(), torch.full((3, 5), 20.0))
    assert calls == ["shrink", "expand"]


def test_cute_variant_uses_plan_base_and_same_factors(monkeypatch):
    import sys
    from types import ModuleType

    import torch

    import ctikernels.linear as linear
    from benchmark.kernel_tuning import lora_sglang as bench

    _lora_inference = ModuleType("ctikernels.linear._lora_inference")
    monkeypatch.setitem(sys.modules, _lora_inference.__name__, _lora_inference)
    monkeypatch.setattr(linear, "_lora_inference", _lora_inference, raising=False)
    seen = []

    class Plan:
        def __init__(self, factors, **kwargs):
            self.factors = factors
            self.base_provider = "test-base"

        def base(self, x, weight):
            seen.append("base")
            return x @ weight.T

    monkeypatch.setattr(_lora_inference, "SmallBatchLoRAInferencePlan", Plan, raising=False)

    def delta(output, x, factors, *, scaling, output_sizes):
        seen.append("delta")
        assert factors is expected_factors
        assert scaling.dtype == torch.float32 and scaling.ndim == 0
        assert output_sizes == (3,)
        return output.add_(scaling)

    monkeypatch.setattr(linear, "lora_delta_factors_", delta)
    case = bench.ProjectionCase("test", 4, (3,))
    expected_factors = (torch.ones(2, 4), torch.ones(3, 2))
    operands = [(torch.ones(1, 4), torch.ones(3, 4), None, None, expected_factors)]
    calls = bench._calls("ours_cute", operands, case, None, {}, None, None)
    torch.testing.assert_close(calls[0](), torch.full((1, 3), 6.0))
    assert seen == ["base", "delta"]
    assert calls[0].plan.provider == "cute-segmented"
    assert all(("ours_cute", market) in bench.COMPARISONS for market in bench.MARKET_VARIANTS)


def test_triton_report_alias_does_not_duplicate_timing():
    from benchmark.kernel_tuning import lora_sglang as bench

    assert "ours_triton" in bench.VARIANTS and "ours_plan" not in bench.VARIANTS
    assert all(("ours_triton", market) in bench.COMPARISONS for market in bench.MARKET_VARIANTS)
    result = {"ok": True, "us": 12.0, "provider": "triton-small-batch-v1"}
    comparison = {"verdict": "faster"}
    row = {
        "variants": {"ours_triton": result},
        "comparisons": {"ours_triton_vs_faster_market": comparison},
    }
    bench.add_report_aliases(row)
    assert row["variants"]["ours_plan"] is result
    assert row["comparisons"]["ours_plan_vs_faster_market"] is comparison


def test_provider_table_flag_and_output(capsys):
    import json

    from benchmark.kernel_tuning import lora_sglang as bench

    assert not build_parser().parse_args(["--sglang-python", "comparator"]).provider_table
    assert (
        build_parser()
        .parse_args(["--sglang-python", "comparator", "--provider-table"])
        .provider_table
    )
    row = {
        "model": "9b",
        "case": "o",
        "device_class": "other_sm12x",
        "rank": 512,
        "tokens": 1024,
        "K": 4096,
        "output_sizes": [4096],
        "policy_provider": "cute",
    }
    bench.emit_provider_table(row)
    line = capsys.readouterr().out.strip()
    prefix, payload = line.split(" ", 1)
    assert prefix == "PROVIDER"
    assert json.loads(payload) == row
