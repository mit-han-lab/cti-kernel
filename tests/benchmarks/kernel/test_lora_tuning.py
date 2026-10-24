# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmark import _support
from benchmark.kernel_tuning import lora as lora_benchmark
from benchmark.kernel_tuning.lora import ProjectionCase, classify_result


def _timing(*, p50: float, cv: float) -> dict[str, float | int]:
    return {"p50": p50, "cv": cv}


def test_projection_case_parses_operation_shape() -> None:
    case = ProjectionCase.parse("qkv=1024:2048,1024,1024")

    assert case == ProjectionCase(
        name="qkv",
        input_size=1024,
        output_sizes=(2048, 1024, 1024),
    )


@pytest.mark.parametrize(
    "raw",
    [
        "qkv",
        "qkv=1024",
        "=1024:2048",
        "qkv=0:2048",
        "qkv=1024:0",
        "qkv=input:2048",
    ],
)
def test_projection_case_rejects_incomplete_or_invalid_shapes(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        ProjectionCase.parse(raw)


@pytest.mark.parametrize(
    ("direct", "candidate", "expected"),
    [
        (_timing(p50=10.0, cv=0.01), _timing(p50=8.0, cv=0.01), "candidate_preferred"),
        (_timing(p50=10.0, cv=0.01), _timing(p50=12.0, cv=0.01), "direct_preferred"),
        (_timing(p50=10.0, cv=0.01), _timing(p50=10.1, cv=0.01), "inconclusive"),
        (_timing(p50=10.0, cv=0.04), _timing(p50=8.0, cv=0.01), "noisy"),
    ],
)
def test_profile_classification_is_advisory_and_noise_aware(
    direct: dict[str, float | int],
    candidate: dict[str, float | int],
    expected: str,
) -> None:
    assert (
        classify_result(
            direct,
            candidate,
            min_speedup_fraction=0.03,
            max_cv=0.03,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("min_speedup_fraction", "max_cv"),
    [(-0.01, 0.03), (1.0, 0.03), (0.03, 0.0), (0.03, 1.0)],
)
def test_profile_classification_rejects_invalid_thresholds(
    min_speedup_fraction: float,
    max_cv: float,
) -> None:
    with pytest.raises(ValueError, match="classification thresholds"):
        classify_result(
            _timing(p50=10.0, cv=0.01),
            _timing(p50=8.0, cv=0.01),
            min_speedup_fraction=min_speedup_fraction,
            max_cv=max_cv,
        )


def test_profile_classification_rejects_nonpositive_latency() -> None:
    with pytest.raises(ValueError, match="latency measurements"):
        classify_result(
            _timing(p50=0.0, cv=0.01),
            _timing(p50=8.0, cv=0.01),
            min_speedup_fraction=0.03,
            max_cv=0.03,
        )


@pytest.mark.parametrize(
    ("direct", "candidate", "message"),
    [
        (_timing(p50=float("nan"), cv=0.01), _timing(p50=8.0, cv=0.01), "latency"),
        (_timing(p50=10.0, cv=-0.01), _timing(p50=8.0, cv=0.01), "variation"),
        (_timing(p50=10.0, cv=0.01), _timing(p50=8.0, cv=float("inf")), "variation"),
    ],
)
def test_profile_classification_rejects_invalid_measurements(
    direct: dict[str, float | int],
    candidate: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_result(
            direct,
            candidate,
            min_speedup_fraction=0.03,
            max_cv=0.03,
        )


def test_profile_distribution_preserves_latency_summary_contract() -> None:
    summary = _support.distribution((1.0, 2.0, 3.0))

    assert summary == {
        "count": 3,
        "mean": 2.0,
        "stddev": pytest.approx(0.816496580927726),
        "cv": pytest.approx(0.408248290463863),
        "min": 1.0,
        "p25": 1.5,
        "p50": 2.0,
        "p90": pytest.approx(2.8),
        "p95": pytest.approx(2.9),
        "p99": pytest.approx(2.98),
        "max": 3.0,
        "iqr": 1.0,
    }


def test_profile_receipt_retains_content_identity(tmp_path: Path) -> None:
    output = tmp_path / "lora-profile.json"
    payload = {"kind": "operation_local_kernel_profile", "results": []}

    _support.write_receipt(output, payload)

    receipt = json.loads(output.read_text())
    identity = receipt.pop("receipt_identity")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    assert receipt == payload
    assert identity == {
        "algorithm": "sha256",
        "canonicalization": "json-sort-keys-compact-v1",
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    assert not output.with_suffix(".json.tmp").exists()


def test_profile_fingerprint_tracks_kernel_repository_sources(monkeypatch) -> None:
    triton = ModuleType("triton")
    triton.__version__ = "3.7.1"
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(lora_benchmark, "git_source_state", lambda: ("revision", True))
    monkeypatch.setattr(lora_benchmark.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        lora_benchmark.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(
            multi_processor_count=170,
            name="test GPU",
            total_memory=32 * 1024**3,
        ),
    )
    monkeypatch.setattr(lora_benchmark.torch.cuda, "get_device_capability", lambda _index: (12, 0))
    sources = (
        Path(_support.__file__).resolve(),
        Path(lora_benchmark.__file__).resolve(),
    )

    fingerprint = lora_benchmark._fingerprint(source_paths=sources)

    assert fingerprint["source"] == {
        "dirty": True,
        "files": {
            "benchmark/_support.py": _support.sha256_file(sources[0]),
            "benchmark/kernel_tuning/lora.py": _support.sha256_file(sources[1]),
        },
        "revision": "revision",
    }
    assert fingerprint["device"] == {
        "capability": [12, 0],
        "index": 0,
        "multi_processor_count": 170,
        "name": "test GPU",
        "total_memory_bytes": 32 * 1024**3,
    }
