# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Minimal measurement and provenance support for kernel benchmarks."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_RECEIPT_IDENTITY_FIELD = "receipt_identity"
_RECEIPT_CANONICALIZATION = "json-sort-keys-compact-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_source_state() -> tuple[str, bool]:
    revision = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "-C", str(REPO_ROOT), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, bool(status)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.mean(values)
    stddev = statistics.pstdev(values)
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    return {
        "count": len(values),
        "mean": mean,
        "stddev": stddev,
        "cv": stddev / mean if mean else 0.0,
        "min": min(values),
        "p25": q1,
        "p50": _percentile(values, 0.5),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "iqr": q3 - q1,
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    if _RECEIPT_IDENTITY_FIELD in payload:
        raise ValueError("receipt payload is already sealed")
    sealed = {
        **payload,
        _RECEIPT_IDENTITY_FIELD: {
            "algorithm": "sha256",
            "canonicalization": _RECEIPT_CANONICALIZATION,
            "payload_sha256": _canonical_json_sha256(payload),
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
