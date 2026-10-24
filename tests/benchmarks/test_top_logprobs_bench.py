# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only checks of the DN3 benchmark's reporting and semantic baselines."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.logprob import top_bench
from benchmark.logprob.top_bench import cases, parity, summarize, torch_market


def test_case_matrix_and_medians():
    matrix = list(cases())
    assert len(matrix) == 72
    assert {case[2] for case in matrix} == {1, 2, 3, 4, 16, 32, 64}
    assert {case[3] for case in matrix} == {1, 5, 20}
    assert {case[0] for case in matrix} == {151936, 248320}
    result = summarize(
        {
            "ours": [3.0, 1.0, 2.0],
            "vllm_v2": [2.0, 4.0, 3.0],
            "sglang_torch": [4.0, 5.0, 6.0],
            "gather": [0.3, 0.1, 0.2],
        }
    )
    assert result["median_us"]["ours"] == 2
    assert result["ours_over_fastest_market"] == 2 / 3
    assert result["gather_us"] == 0.2
    assert "gather" not in result["median_us"]


def test_semantic_baselines_and_parity_reporting():
    x = torch.tensor([[2.0, 2.0, 0.0, -2.0], [0.0, 1.0, 3.0, 2.0]])
    selected = torch.tensor([0, 2])
    v1 = torch_market(x, 2, selected, ranks=True)
    sgl = torch_market(x, 2, selected, ranks=False)
    assert v1["ranks"].tolist() == [2, 1]
    assert "ranks" not in sgl
    torch.testing.assert_close(v1["selected"], x.log_softmax(-1)[torch.arange(2), selected])
    check = parity(v1, v1)
    assert check["ids_as_sets"] and check["logprobs_close"] and check["ranks_equal"]
    changed = {key: value.clone() for key, value in v1.items()}
    changed["ids"][0, 0] = 3
    changed["top"][0, 0] += 1
    check = parity(v1, changed)
    assert not check["ids_as_sets"] and not check["logprobs_close"]


@pytest.mark.parametrize("mode", ["raw", "processed"])
def test_benchmark_provider_call_on_cpu(mode, monkeypatch):
    # Run the actual benchmark argument wiring and provider, with CPU tensors
    # and an eager measurement stub in place of CUDA timing/capture.
    cpu_torch = SimpleNamespace(**{**vars(torch), "device": lambda *_: torch.device("cpu")})
    monkeypatch.setattr(top_bench, "torch", cpu_torch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU test")

    def measure(callables, replays):
        ours = callables["ours"]()
        market = callables["vllm_v1_torch"]()
        check = parity(ours, market)
        assert all(check[key] for key in ("ids_as_sets", "logprobs_close", "ranks_equal"))
        assert len(callables["gather"]()) == 2
        return summarize(dict.fromkeys(callables, [1.0])), None

    monkeypatch.setattr(top_bench, "measure_interleaved", measure)
    result = top_bench.bench_row(
        None, 17, torch.float32, 3, 5, mode, 1, 2, ["ours", "vllm_v1_torch"]
    )
    assert result["device"] == "CPU test"
    assert result["mode"] == mode


@pytest.mark.parametrize("providers", [("ours",), ("vllm_v1_torch",), ("ours", "vllm_v1_torch")])
def test_provider_subset_summary(providers):
    samples = {name: [2.0, 4.0, 3.0] for name in providers}
    result = summarize({**samples, "gather": [0.1]})
    assert set(result["median_us"]) == set(providers)
    assert ("ours_over_fastest_market" in result) == (len(providers) == 2)


def test_provider_subset_needs_no_vllm_bundle(tmp_path, monkeypatch, capsys):
    providers = ["ours", "vllm_v1_torch"]

    def forbidden(*args, **kwargs):
        raise AssertionError("unselected vllm_v2 must not be loaded")

    def bench(vllm, vocab, dtype, n, k, mode, replays, row_chunk, selected_providers):
        assert vllm is None
        assert selected_providers == providers
        return {
            "V": vocab,
            "dtype": str(dtype),
            "n": n,
            "k": k,
            "mode": mode,
            "median_us": dict.fromkeys(selected_providers, 1.0),
            "gather_us": 0.1,
            "parity_vllm_v2": None,
        }

    monkeypatch.setattr(top_bench, "load_vllm", forbidden)
    monkeypatch.setattr(top_bench, "bench_row", bench)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    output = tmp_path / "timings.jsonl"
    top_bench.main(["--out", str(output), "--providers", *providers, "--limit", "1"])
    row = json.loads(output.read_text())
    assert set(row["median_us"]) == set(providers)
    assert row["parity_vllm_v2"] is None
    console = capsys.readouterr().out
    assert len(console.splitlines()) == 1
    assert "ours=1.0 vllm_v1_torch=1.0" in console


def test_default_providers_require_vllm_bundle(tmp_path):
    with pytest.raises(SystemExit, match="2"):
        top_bench.main(["--out", str(tmp_path / "timings.jsonl")])
