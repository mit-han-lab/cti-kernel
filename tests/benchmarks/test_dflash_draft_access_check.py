# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU checks for access matrices, exact storage and sanitizer receipts."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import dflash_draft_access_check as check


def test_cap_256_case_follows_135_completed_replays():
    configurations = [*check.cute_matrix(), *check.large_capacity_matrix()]
    completed = 0
    for case in configurations:
        if case.get("chunk_cap") == 256:
            assert completed == 135
            assert (case["q"], case["b"], case["page_size"], case["capacity"]) == (9, 1, 16, 8192)
            break
        completed += 3 + int(case["requested_splits"] is None)
    else:
        raise AssertionError("missing small-cap case")


def test_construction_exception_receipt_includes_message(monkeypatch):
    def failing_cases(provider):
        raise ValueError("DRAFT CuTe requires SHORT <= CHUNK_CAP")
        yield  # make the construction failure occur during iteration

    monkeypatch.setattr(check, "cute_access_cases", failing_cases)
    receipt = check.access_check()
    assert not receipt["ok"]
    assert receipt["error"] == "ValueError: DRAFT CuTe requires SHORT <= CHUNK_CAP"


def test_execution_exception_receipt_includes_message():
    record = {}
    with check.case_result(record):
        raise ValueError("invalid geometry")
    assert not record["ok"]
    assert record["error"] == "ValueError: invalid geometry"


def test_table_matrix_coverage_and_budget():
    configurations = list(check.table_matrix())
    assert len(configurations) == 32
    assert sum(len(list(check.table_patterns(case))) for case in configurations) == 190
    assert {
        (
            case["requested_provider"],
            case["metadata_index"],
            case["causal"],
            case["window"],
            case["table_width"] - case["max_seq_len"],
        )
        for case in configurations
    } == {
        (provider, metadata, causal, window, padding)
        for provider, metadata, (causal, window), padding in product(
            ("cute", "flashinfer"),
            ("request", "row"),
            ((False, None), (True, None), (True, 4096), (False, 2048)),
            (0, 17),
        )
    }
    for provider, metadata in product(("cute", "flashinfer"), ("request", "row")):
        cases = [
            case
            for case in configurations
            if (case["requested_provider"], case["metadata_index"]) == (provider, metadata)
        ]
        assert {case["b"] for case in cases} == {1, 3, 4}
        assert {case["q"] for case in cases} == {4, 9, 16}
        assert {
            (case["head_dim"], case["num_qo_heads"], case["num_kv_heads"]) for case in cases
        } == {(128, 32, 8), (256, 16, 4), (256, 8, 1)}
    for provider in ("cute", "flashinfer"):
        cases = [case for case in configurations if case["requested_provider"] == provider]
        for name, values in (("b", (1, 3, 4)), ("q", (4, 9, 16))):
            assert {
                (case[name], case["head_dim"], case["num_qo_heads"], case["num_kv_heads"])
                for case in cases
            } == {
                (value, dim, hq, hk)
                for value, (dim, hq, hk) in product(
                    values, ((128, 32, 8), (256, 16, 4), (256, 8, 1))
                )
            }


@pytest.mark.parametrize("configuration", list(check.table_matrix()))
def test_table_patterns_and_exact_host_storage(configuration):
    b, q, maximum, width = (
        configuration[name] for name in ("b", "q", "max_seq_len", "table_width")
    )
    patterns = list(check.table_patterns(configuration))
    states = {state for pattern in patterns for state in pattern}
    assert {
        (1, False),
        (maximum - 1, False),
        (maximum - 1, True),
        (-1, False),
        (maximum, False),
        (2**62, False),
        (-(2**62), False),
    } <= states
    assert any(pattern[-1] == (maximum - 1, False) for pattern in patterns)
    if configuration["window"]:
        window = configuration["window"]
        assert {(window - q + offset, False) for offset in (-1, 0, 1)} <= states
        assert maximum > window

    for pattern in patterns:
        inputs = check.table_host_inputs(configuration, pattern)
        check.exact_storage(inputs)
        requests, frontier, done, table = inputs
        assert requests.shape == (b,)
        assert requests.dtype == table.dtype == torch.int32
        assert frontier.dtype == torch.int64 and done.dtype == torch.bool
        assert int(requests[-1]) == table.shape[0] - 1
        assert requests.tolist() == list(range(1, 2 * b + 1, 2))
        assert table.shape[1] == table.stride(0) == width
        assert table.stride(1) == 1
        count = table.shape[0] + 1 if configuration["metadata_index"] == "request" else b
        assert frontier.shape == done.shape == (count,)
        allowed = torch.zeros_like(table, dtype=torch.bool)
        for row, (f, is_done) in enumerate(pattern):
            request = int(requests[row])
            metadata = request if configuration["metadata_index"] == "request" else row
            assert int(frontier[metadata]) == f
            assert bool(done[metadata]) == is_done
            if not is_done and 0 <= f < maximum:
                allowed[request, : min(f + q, maximum)] = True
        assert torch.all(table[~allowed] == 2**31 - 1)
        assert torch.all((0 <= table[allowed]) & (table[allowed] < maximum + 37))


def test_table_dispatch_and_skip_receipt(monkeypatch, capsys):
    monkeypatch.setattr(check, "table_access_cases", lambda: iter([{"ok": True}]))
    receipt = check.access_check("table")
    assert receipt["ok"] and receipt["cases"] == receipt["passed"] == 1
    assert receipt["skipped"] == [check.TABLE_BOUND_SKIP]
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("errors", [0, 3])
def test_table_sanitizer_cli_forwards_provider_and_reports_errors(monkeypatch, capsys, errors):
    monkeypatch.setattr(check, "find_sanitizer", lambda: Path("compute-sanitizer"))
    monkeypatch.setattr(check.sys, "argv", ["probe", "--provider", "table", "--launch-sanitizer"])

    def run(command, **kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="version 2026.1", stderr="", returncode=0)
        assert command[-2:] == ["--provider", "table"]
        assert command[1:3] == ["--tool", "memcheck"]
        assert kwargs["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
        return SimpleNamespace(
            stdout=check.json.dumps(
                {
                    "type": "access_check",
                    "ok": True,
                    "caching_disabled": True,
                    "cases": 190,
                    "passed": 190,
                    "failed": 0,
                }
            ),
            stderr=f"ERROR SUMMARY: {errors} errors",
            returncode=0,
        )

    monkeypatch.setattr(check.subprocess, "run", run)
    assert check.main() == int(errors != 0)
    receipt = check.json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["requested_provider"] == "table"
    assert receipt["cases"] == 190
    assert receipt["error_summary"] == f"ERROR SUMMARY: {errors} errors"


def test_table_provider_tuning_record():
    names = (
        "splits",
        "key_tile",
        "stages",
        "key_groups",
        "short_union",
        "inline_merge",
        "min_chunk",
        "chunk_cap",
        "merge_fan_in",
        "counter_bytes",
    )
    tuning = dict.fromkeys(names, 1)
    op = SimpleNamespace(provider="cute-table", _implementation=SimpleNamespace(**tuning))
    assert check.provider_record(op) == {"provider": "cute-table", "tuning": tuning}
