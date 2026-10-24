# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT capture API, graph node and replay activity receipts."""

import json
import subprocess

import pytest
import torch

from benchmark.kernel_tuning.dflash_draft_capture_trace import (
    MASKS,
    profile_summary,
    trace_round,
)

from ._exhaustive import ci_cases


# CI captures B=4, Q=16 causal attention with a 4096-token window.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "b,q,causal,window",
    ci_cases(
        [1, 2, 3, 4],
        MASKS,
        core=[(4, 16, True, 4096)],
    ),
)
@pytest.mark.parametrize("fused", [False, True])
def test_draft_attention_capture_trace(b, q, causal, window, fused, draft_provider, monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_INLINE", "1")
    monkeypatch.delenv("CTI_DRAFT_SPLITS", raising=False)
    result = trace_round(b, q, causal, window, provider=draft_provider, fused=fused)
    for name, flag in result["flags"].items():
        assert flag, (name, result)
    for replay in result["replays"]:
        for name, flag in replay["flags"].items():
            assert flag, (name, replay)
    assert result["pass"], result
    if result["provider"] == "cute-splitkv":
        # Both run and fused prepare_and_run capture exactly one attention node.
        assert result["graph_node_counts"]["KERNEL"] == 1
        assert result["graph_memset_count"] == 0


def test_trace_parser_keeps_forbidden_apis_and_async_activities():
    events = [
        {"name": "dflash_round", "ph": "X", "ts": 10, "dur": 10},
        {"name": "cudaDeviceSynchronize", "ph": "X", "ts": 25, "cat": "cuda_runtime"},
        {"name": "cudaGraphLaunch", "ph": "X", "ts": 11, "cat": "cuda_runtime"},
        {"name": "cuMemAlloc_v2", "ph": "X", "ts": 12, "cat": "cuda_driver"},
        {"name": "cudaStreamQuery", "ph": "X", "ts": 13, "cat": "cuda_runtime"},
        {"name": "cudaMemcpyAsync", "ph": "X", "ts": 14, "cat": "cuda_runtime"},
        {"name": "Memcpy DtoD", "ph": "X", "ts": 30, "cat": "gpu_memcpy"},
        {"name": "prepare_draft", "ph": "X", "ts": 35, "cat": "kernel"},
    ]
    result = profile_summary(events)
    assert result["allocation_apis"] == ["cuMemAlloc_v2"]
    assert result["sync_apis"] == ["cudaStreamQuery"]
    assert result["memcpy_apis"] == ["cudaMemcpyAsync"]
    assert result["memcpy_activities"] == ["Memcpy DtoD"]
    assert result["kernel_count"] == 1
    with pytest.raises(ValueError, match="one complete round"):
        profile_summary([])


@pytest.mark.parametrize(
    "summary,rc,passed",
    [
        ("ERROR SUMMARY: 0 errors", 0, True),
        ("ERROR SUMMARY: 1 error", 1, False),
        ("", 0, False),
        ("ERROR SUMMARY: 2 errors\nERROR SUMMARY: 0 errors", 0, False),
    ],
)
def test_sanitizer_wrapper_requires_receipt_and_clean_summary(
    monkeypatch, capsys, tmp_path, summary, rc, passed
):
    from benchmark.kernel_tuning import dflash_draft_access_check as check

    binary = tmp_path / "compute-sanitizer"
    monkeypatch.setattr(check, "find_sanitizer", lambda: binary)

    def run(command, **kwargs):
        assert command[0] == str(binary)
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "Compute Sanitizer version 2026.1.0", "")
        assert command[1:9] == [
            "--tool",
            "memcheck",
            "--padding",
            "4096",
            "--print-limit",
            "50",
            "--error-exitcode",
            "1",
        ]
        assert kwargs["env"]["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"
        assert "--launch-sanitizer" not in command
        receipt = {"type": "access_check", "ok": True, "caching_disabled": True}
        return subprocess.CompletedProcess(command, rc, json.dumps(receipt), summary)

    monkeypatch.setattr(check.subprocess, "run", run)
    assert check.launch_sanitizer() == (0 if passed else 1)
    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    receipt, report = map(json.loads, output.splitlines())
    assert receipt["ok"]
    assert report["sanitizer"] == "compute-sanitizer"
    assert report["version"] == "2026.1.0"
    assert report["rc"] == rc
    assert report["ok"] is passed
