# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Correctness tests do not time anything; latency belongs to benchmark/.

Watchdog deadlines (``time.monotonic``) are allowed: they bound a hang, they do not measure.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
# Not correctness suites: benchmark-harness tests and the repository guards.
SKIP = {"benchmarks", "quality", "ci"}
TIMERS = {"perf_counter", "perf_counter_ns", "process_time", "elapsed_time", "timeit"}
# Existing timing thresholds; each moves to benchmark/ in its op's conversion batch (§3.7).
# This list may only shrink.
ALLOWED = {
    "gpu/test_gdn_replay_handoff_layers.py",  # host median <= 500 us (B2, gdn)
    "gpu/test_moe_expert_map.py",  # align graph replay < 30 us (MoE, paused: D14)
}


def _timing(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in TIMERS:
            lines.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id in TIMERS:
            lines.append(node.lineno)
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in TIMERS:
            lines.append(node.lineno)
        elif isinstance(node, ast.keyword) and node.arg == "enable_timing":
            lines.append(node.value.lineno)
    return lines


def test_correctness_tests_do_not_time() -> None:
    found = {}
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(TESTS).as_posix()
        if rel.split("/")[0] in SKIP or rel in ALLOWED:
            continue
        lines = _timing(ast.parse(path.read_text(), rel))
        if lines:
            found[rel] = lines
    assert not found, f"timing in correctness tests (move it to benchmark/): {found}"


def test_allowed_list_only_names_files_that_still_time() -> None:
    stale = [rel for rel in ALLOWED if not _timing(ast.parse((TESTS / rel).read_text()))]
    assert not stale, f"remove from ALLOWED: {stale}"
