# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Small warm-up qualification; two buckets, two candidates, bounded weight rotation."""

import importlib.metadata
import json
from contextlib import nullcontext
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from ctikernels.linear import (
    Linear,
    autotune,
    autotune_entries,
    release_cublas_workspaces,
    release_scratch,
)
from ctikernels.linear import _autotune as tuning
from ctikernels.linear._dispatch import TUNED, TUNED_GATED, Config


@pytest.fixture(autouse=True)
def dependencies(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("requires sm120/sm121")
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("cutlass-dsl unavailable")
    if version != "4.8.0":
        pytest.skip("requires cutlass-dsl 4.8.0")
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE_CACHE", str(tmp_path / "autotune.json"))
    monkeypatch.setenv("CTI_LINEAR_PROVIDER", "auto")
    monkeypatch.setenv("CTI_LINEAR_AUTOTUNE", "cache")
    monkeypatch.setattr(tuning, "_CACHES", {})


def test_warmup_reload_capture_and_off(monkeypatch):
    n, k = 3072, 4096
    assert all((n, k) not in shapes for shapes in TUNED.values())
    assert all(all(shape[:2] != (n, k) for shape in shapes) for shapes in TUNED_GATED.values())
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    original = weight.clone()
    op = Linear()
    with autotune(ms=[1, 2], budget_seconds=600, top_k=2):
        for m in (1, 2, 3):
            op(x[:m], weight)
    assert torch.equal(weight, original)
    entries = autotune_entries()
    assert len(entries) == 2
    for key, entry in entries.items():
        bucket = tuple(json.loads(key)[-1])
        assert bucket in ((1, 1), (2, 2))
        if entry["provider"] == "cute":
            assert set(bucket).issubset(entry["measured_m"])
    decisions = {m: op.decision(n, k, m) for m in (1, 2)}
    ref, condition = tuning._reference_and_condition(x, weight)
    for m, decision in decisions.items():
        if decision["provider"] == "cute":
            assert decision["record"]["min_speedup"] >= 1.03
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            # CuTe is gated against full precision; fallback is the comparator itself.
            if decision["provider"] == "cute":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            baseline = F.linear(x[:m], weight)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
        assert tuning._accuracy(op(x[:m], weight), baseline, ref[:m], condition[:m], k)["correct"]
    tuning._CACHES.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("reload/capture attempted tuning")

    monkeypatch.setattr(tuning, "tune", forbidden)
    reloaded = Linear()
    for m in (1, 2):
        assert reloaded.decision(n, k, m) == decisions[m]
        reloaded(x[:m], weight)  # Warm compiled kernels or cuBLAS before capture.
    # Warm the unmeasured bucket too, then exercise its None-plan capture guard.
    reloaded(x[:3], weight)
    expected = {m: op(x[:m], weight) for m in (1, 2, 3)}
    for active in (False, True):
        with autotune(ms=[1, 2]) if active else nullcontext():
            for m in (1, 2, 3):
                # The unmeasured key must only be called under capture in a context.
                eager = expected[m]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = reloaded(x[:m], weight)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output.view(torch.int16), eager.view(torch.int16))
    off = Linear(autotune="0")
    off(x[:1], weight)
    assert off.provider == "cublas" and off.decision(n, k, 1)["source"] == "off"


def idle_segment_bytes():
    """Cached segments with no live block; empty_cache() frees all but live graphs' pools."""
    device = torch.cuda.current_device()
    return sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["device"] == device and segment["allocated_size"] == 0
    )


def test_exit_returns_tuning_memory():
    n, k = 3072, 4096  # A table hole, so the context tunes it.
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    op = Linear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # A reusable cache would hide tuning's growth.
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    idle = idle_segment_bytes()
    torch.cuda.reset_peak_memory_stats()
    with autotune(ms=[1], budget_seconds=600, top_k=2):
        op(x, weight)
    tolerance = 16 << 20
    assert torch.cuda.max_memory_reserved() > reserved + 2 * tolerance  # Tuning did allocate.
    # Exit returned every segment tuning left idle (graph pools, copies, inputs); other
    # tests' partially used segments are not idle, so fragmentation cannot mask this.
    assert idle_segment_bytes() <= idle
    release_scratch()  # The op's own eager scratch, not tuning's.
    release_cublas_workspaces()  # Quiescent here: no other cuBLAS work.
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() <= allocated + tolerance
    assert torch.cuda.memory_reserved() <= reserved + tolerance


def test_zero_budget():
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(3072, 4096, device="cuda", dtype=torch.bfloat16)
    op = Linear()
    with autotune(budget_seconds=0):
        op(x, weight)
    assert op.provider == "cublas" and autotune_entries() == {}


def test_warmed_route_frozen_in_capture_under_other_cache(tmp_path):
    """Warm-up decides cuBLAS; capture under a context whose cache holds a compiled
    split-K CuTe plan for the same key keeps cuBLAS and the advertised bytes."""
    n, k = 3072, 4096
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    device = torch.device("cuda", torch.cuda.current_device())
    record = tuning._record((3, 4))
    # 60 pairs with ci_low > 1 qualify on every device, GB10 included.
    point = {"speedup": 1.5, "epsilon": 0.01, "ours_us": 10.0, "cublas_us": 15.0, "pairs": 60}
    point.update(ci_low=1.4, ci_high=1.6)
    record.update(
        provider="cute",
        config=asdict(Config(splits=2)),
        correct=True,
        deterministic=True,
        measured_m=[3, 4],
        regimes={r: {"3": dict(point), "4": dict(point)} for r in tuning.REGIMES},
    )
    other = tmp_path / "other.json"
    key = tuning.cache_key(tuning.identity(device), n, k, 0, (3, 4))
    other.write_text(json.dumps({"format": tuning.FORMAT, "entries": {key: record}}))
    with autotune(cache=str(other)):
        compiled = Linear()
        compiled(x, weight)  # Compiles the cached split-K plan in this process.
        assert compiled.provider == "cute"
    op = Linear()
    expected = op(x, weight)
    assert op.provider == "cublas" and op.workspace_bytes(n, k, 4) == 0
    with autotune(cache=str(other)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(x, weight)
        assert op.provider == "cublas" and op.workspace_bytes(n, k, 4) == 0
        assert op.decision(n, k, 4)["provider"] == "cublas"
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
