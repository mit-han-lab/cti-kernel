# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks of DN4 timing matrix, prepared state, wiring and reporting."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmark.logprob import penalty_bench as bench


def test_cases_and_market_ratio():
    matrix = list(bench.cases())
    assert len(matrix) == 20
    assert {v for v, _, _, _ in matrix} == {151936, 248320}
    assert {(b, q) for _, b, q, _ in matrix} == {
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (1, 4),
        (1, 16),
        (2, 16),
        (4, 16),
        (4, 8),
        (1, 64),
    }
    assert {width for _, _, _, width in matrix} == {32768}
    wide = list(bench.cases(wide=True))
    assert len(wide) == 28
    assert {(b, q) for _, b, q, width in wide if width == 262144} == {
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
    }
    assert len(list(bench.cases(262144, wide=True))) == 20
    result = bench.summarize(
        {
            "ours": [9, 3, 6],
            "ours_fp32_inplace": [0.1],
            "engine_torch": [0.01],
            "vllm_v2": [4, 2, 3],
            "sglang_penaltylib": [4, 6, 5],
        }
    )
    assert result["median_us"]["ours"] == 6
    assert result["ours_over_fastest_market"] == 2
    assert "ours_over_fastest_market" not in bench.summarize(
        {"ours": [1], "ours_fp32_inplace": [0.1]}
    )


def test_state_packing_and_sglang_baseline():
    table = torch.tensor([[31, 32, 2, 2, -1, 37]], dtype=torch.int32)
    rows = SimpleNamespace(
        rows=torch.tensor([0]),
        prompt_lengths=torch.tensor([2]),
        lengths=torch.tensor([6]),
        presence=torch.tensor([0.5]),
        frequency=torch.tensor([0.25]),
        repetition=torch.tensor([2.0]),
    )
    state = bench.prepare_state(table, rows, 37, torch.tensor([[3, 2]], dtype=torch.int32))
    assert state.prompt_bin_mask.dtype == state.output_bin_counts.dtype == torch.int32
    assert state.prompt_bin_mask.tolist() == [[-2147483648, 1]]
    assert state.output_bin_counts[0, 2] == 2
    assert state.output_bin_counts.sum() == 2
    assert state.token_ids.tolist() == [37, 3, 2]
    assert state.expanded_idx_mapping.tolist() == [0, 0, 0]
    assert state.expanded_local_pos.tolist() == [0, 1, 2]
    x = torch.full((3, 37), 4.0)
    x[1, 2] = -4
    actual = bench.sglang_penaltylib(x, state, 3)
    assert actual[:, 2].tolist() == [1.5, -10, 1.5]
    assert actual[:, 31].tolist() == [4, 4, 4]  # Prompt-only repetition absent.
    assert actual[:, 3].tolist() == [4, 4, 4]  # Drafts not incorporated.


def test_ulp_reporting():
    a = torch.tensor([[-2.0, 1.0, 0.0, -0.0]])
    b = a.clone()
    b[0, 0] = torch.nextafter(a[0, 0], torch.tensor(0.0))
    b[0, 1] = torch.nextafter(a[0, 1], torch.tensor(2.0))
    b[0, 3] = 0.0
    assert bench.parity(a, a) == {"max_ulp_difference": 0, "differing_elements": 0}
    assert bench.parity(a, b) == {"max_ulp_difference": 1, "differing_elements": 3}


@pytest.mark.parametrize("q", [1, 4])
def test_benchmark_wiring_on_cpu(q, monkeypatch):
    cpu_torch = SimpleNamespace(**{**vars(torch), "device": lambda *_: torch.device("cpu")})
    monkeypatch.setattr(bench, "torch", cpu_torch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU test")
    seen = []

    def v2(logits, mapping, token_ids, positions, repetition, frequency, presence, packed, base):
        seen.append(True)
        v = logits.shape[-1]
        for i in range(logits.shape[0]):
            n, j = int(mapping[i]), int(positions[i])
            counts = base[n].clone()
            for token in token_ids[i - j + 1 : i + 1]:
                counts[int(token)] += 1
            ids = torch.arange(v)
            prompt = ((packed[n, ids // 32].long() >> (ids % 32)) & 1).bool()
            scale = torch.where(prompt | (counts > 0), repetition[n], 1.0)
            logits[i].mul_(torch.where(logits[i] > 0, torch.ones_like(scale) / scale, scale))
            logits[i].sub_(frequency[n] * counts.float())
            logits[i].sub_(presence[n] * (counts > 0).float())

    def measure(callables, replays):
        outputs = {name: call() for name, call in callables.items()}
        first = {name: value.clone() for name, value in outputs.items()}
        for name, call in callables.items():
            assert torch.equal(call(), first[name]), "in-place calls must reset every execution"
        assert torch.equal(outputs["ours"], outputs["ours_fp32_inplace"])
        check = bench.output_parity(outputs)
        assert check["parity_vllm_v2"]["differing_elements"] == 0
        assert check["parity_engine_torch"] is not None
        return bench.summarize(dict.fromkeys(callables, [1.0])), check

    monkeypatch.setattr(bench, "measure_interleaved", measure)
    result = bench.bench_row(SimpleNamespace(apply_penalties=v2), 37, 2, q, 32768, 30)
    assert seen and result["table_width"] == 32768
    assert result["device"] == "CPU test"
    assert "excluded" in result["market_state_maintenance"]
    assert "copy included" in result["inplace_reset"]
    assert result["ours_over_fastest_market"] == 1
    assert result["parity_vllm_v2"]["differing_elements"] == 0
    assert result["parity_engine_torch"] is not None
    assert "true division" in result["engine_torch_scope"]


def test_provider_subset_jsonl(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("unselected upstream must not load")

    def row(vllm, v, b, q, width, replays, providers):
        assert vllm is None and providers == ["ours", "ours_fp32_inplace"]
        return {
            "V": v,
            "B": b,
            "Q": q,
            "table_width": width,
            "median_us": dict.fromkeys(providers, 1.0),
            "parity_vllm_v2": None,
            "parity_engine_torch": None,
        }

    monkeypatch.setattr(bench, "load_vllm", forbidden)
    monkeypatch.setattr(bench, "bench_row", row)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    path = tmp_path / "timings.jsonl"
    bench.main(
        ["--out", str(path), "--providers", "ours", "ours_fp32_inplace", "--limit", "1", "--wide"]
    )
    result = json.loads(path.read_text())
    assert result["parity_vllm_v2"] is None
    assert set(result["median_us"]) == {"ours", "ours_fp32_inplace"}
    printed = capsys.readouterr().out
    assert len(printed.splitlines()) == 1
    assert "parity_vllm_v2=n/a parity_engine_torch=n/a" in printed


@pytest.mark.parametrize(
    "args",
    [[], ["--providers", "ours", "--table-width", "12"], ["--providers", "ours", "--replays", "2"]],
)
def test_cli_validation(tmp_path, args):
    with pytest.raises(SystemExit, match="2"):
        bench.main(["--out", str(tmp_path / "timings.jsonl"), *args])


def test_runtime_loader_allowlist(monkeypatch):
    from benchmark.dflash_round import comparators

    def load(path, names, *, module_name, globals):
        assert str(path) == "bundle/vllm/v1/worker/gpu/sample/penalties.py"
        assert names == ("_penalties_kernel", "apply_penalties")
        assert globals["cdiv"](33, 32) == 2
        return "loaded"

    monkeypatch.setattr(comparators, "load_definitions", load)
    assert bench.load_vllm("bundle") == "loaded"


@pytest.mark.parametrize("q", [1, 4])
def test_engine_torch_true_division_and_draft_counts(q):
    table = torch.tensor([[0, 1, 1, -1, 7, 2]], dtype=torch.int32)
    rows = SimpleNamespace(
        rows=torch.tensor([0]),
        prompt_lengths=torch.tensor([1]),
        lengths=torch.tensor([5]),
        presence=torch.tensor([0.5]),
        frequency=torch.tensor([0.25]),
        repetition=torch.tensor([3.0]),
    )
    source = torch.tensor([5.0, -3.0, -0.0, 0.3]).expand(1, q, 4).clone()
    original = source.clone()
    drafts = torch.tensor([[1, 1, -1]])[:, : q - 1]
    result = bench.engine_torch(source, table, rows, drafts)
    expected = source.clone()
    for j in range(q):
        counts = torch.tensor([1, 2, 0, 1], dtype=torch.float32)
        for token in drafts[0, :j]:
            counts[token.clamp(0, 3)] += 1
        seen = counts > 0
        rep = torch.where(source[0, j] > 0, source[0, j] / 3.0, source[0, j] * 3.0)
        expected[0, j] = torch.where(seen, rep, source[0, j]) - 0.25 * counts - 0.5 * (counts > 0)
    assert result.dtype == torch.float32 and result.shape == (1, q, 4)
    assert torch.equal(result.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(source.view(torch.int32), original.view(torch.int32))
    # 5/3 and 5*RN32(1/3) differ before (and here after) subtraction.
    assert result[0, 0, 0] != 5.0 * (torch.ones(()) / 3.0) - 0.75


def test_prints_numeric_parity(tmp_path, monkeypatch, capsys):
    def row(*args):
        return {
            "V": 37,
            "B": 1,
            "Q": 4,
            "table_width": 32768,
            "median_us": {"ours": 1.0, "engine_torch": 2.0},
            "parity_vllm_v2": {"differing_elements": 3, "max_ulp_difference": 2},
            "parity_engine_torch": {"differing_elements": 5, "max_ulp_difference": 4},
        }

    monkeypatch.setattr(bench, "bench_row", row)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    path = tmp_path / "timings.jsonl"
    bench.main(["--out", str(path), "--providers", "ours", "engine_torch", "--limit", "1"])
    printed = capsys.readouterr().out
    assert "engine_torch=2.0" in printed
    assert "parity_vllm_v2=3diff/2ulp parity_engine_torch=5diff/4ulp" in printed
    assert json.loads(path.read_text())["parity_engine_torch"] == row()["parity_engine_torch"]


def test_parity_for_provider_subsets():
    x = torch.ones(1, 3)
    assert bench.output_parity({"engine_torch": x}) == {
        "parity_vllm_v2": None,
        "parity_engine_torch": None,
    }
    checks = bench.output_parity({"ours": x, "engine_torch": x})
    assert checks["parity_vllm_v2"] is None
    assert checks["parity_engine_torch"] == {"differing_elements": 0, "max_ulp_difference": 0}
