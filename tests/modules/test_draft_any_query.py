# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host admission, native plan capacities and tiled-query routing without CUDA."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute, dispatch, flashinfer
from ctikernels.attention._flashinfer import (
    FlashInferFixedQueryAttention,
    chunked_schedule,
    fixed_query_tile,
)
from ctikernels.attention.draft import DraftAttention, _geometry, _provider

from ..support.child import run_child

QUERIES = (1, 33, 64, 129, 1024, 65536)


def geometry(query, **overrides):
    return {
        "batch_size": 3,
        "query_length": query,
        "max_kv_tokens": max(query, 8192),
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 16,
        "causal": True,
        "window": None,
        "dtype": torch.bfloat16,
        "device": torch.device("cuda"),
    } | overrides


@pytest.fixture(autouse=True)
def hardware(monkeypatch):
    for name in ("PROVIDER", "ROWS", "MERGE", "INLINE", "SPLITS"):
        monkeypatch.delenv(f"CTI_DRAFT_{name}", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170),
    )


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("dim", (64, 128, 256))
def test_admission_workspace_and_native_plan(query, dim):
    args = geometry(query, head_dim=dim)
    _geometry(**args)
    flashinfer.check_capacity(**args)
    fixed = {
        k: args[k]
        for k in (
            "batch_size",
            "query_length",
            "num_qo_heads",
            "num_kv_heads",
            "head_dim",
            "device",
        )
    }
    floating, integer = FlashInferFixedQueryAttention.workspace_size(**fixed)
    tile = 16 if query == 1 else 64 if dim == 256 else 128
    assert fixed_query_tile(3, query, 4, dim) == tile
    padded = max(340 // 8, -(-3 * query * 4 // tile) + 2)
    assert floating >= 32 * padded * tile * (dim + 1) * 4
    assert integer >= (3 * padded + 4 + 3 * query + 1) * 4 + padded
    assert floating % 256 == integer % 256 == 0
    assert DraftAttention.workspace_size(**args, provider="flashinfer") == (
        floating,
        integer + flashinfer.fixed_query_bytes(3),
    )


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("dim", (64, 128, 256))
def test_configuration_and_routing(query, dim):
    pytest.importorskip("cutlass")
    args = geometry(query, head_dim=dim, num_qo_heads=16, num_kv_heads=4)
    cute.check_support(**args)
    group, rows, rt, splits = cute.configuration(**args)
    assert rt == -(-query * group // rows)
    assert rows in (16, 32, 48, 64) and splits >= 1
    assert _provider("cute", args) is cute
    assert _provider("flashinfer", args) is flashinfer
    if query > 32:
        assert _provider(None, args) is flashinfer
    else:
        assert dispatch.prefers_flashinfer(**args) == dispatch.prefers_flashinfer(
            **(args | {"query_length": 2})
        )


@pytest.mark.parametrize("query", (33, 64, 129, 1024))
def test_narrow_window_routes_to_cute(query):
    pytest.importorskip("cutlass")
    args = geometry(query, causal=False, window=7)
    assert not dispatch.prefers_flashinfer(**args)
    assert _provider(None, args) is cute
    with pytest.raises(NotImplementedError, match="no right bound"):
        _provider("flashinfer", args)


def test_actual_row_grid_and_batch_fallback(monkeypatch):
    pytest.importorskip("cutlass")
    # Early /64 check passes; actual /32 row geometry must reject before sizing.
    args = geometry(65536, batch_size=1)
    with pytest.raises(NotImplementedError, match="grid-y"):
        cute.check_support(**args)
    assert _provider(None, args) is flashinfer
    monkeypatch.setenv("CTI_DRAFT_ROWS", "16")
    args = geometry(32768)
    with pytest.raises(NotImplementedError, match="grid-y"):
        cute.check_support(**args)
    assert _provider(None, args) is flashinfer
    args = geometry(33, batch_size=65536, max_kv_tokens=33)
    with pytest.raises(NotImplementedError, match="grid-y/grid-z"):
        cute.check_support(**args)
    assert _provider(None, args) is flashinfer


@pytest.mark.parametrize("query", (4096, 65536, 262144))
def test_capacity_does_not_multiply_query_by_unsplit_capacity(query):
    # Old admission used B*ceil(C/128)*Q, although Q tiles force a single split.
    args = geometry(query, batch_size=2, max_kv_tokens=1 << 28, page_size=1)
    flashinfer.check_capacity(**args)
    with pytest.raises(NotImplementedError, match="int32"):
        flashinfer.check_capacity(**(args | {"max_kv_tokens": 1 << 30}))


def test_native_csr_boundary_is_exact():
    args = geometry(
        (1 << 31) - 1,
        batch_size=1,
        max_kv_tokens=(1 << 31) - 1,
        num_qo_heads=1,
        num_kv_heads=1,
        page_size=1,
    )
    flashinfer.check_capacity(**args)
    with pytest.raises(NotImplementedError, match="int32"):
        flashinfer.check_capacity(**(args | {"query_length": 1 << 31}))


@pytest.mark.parametrize("name", ("prepare_draft_chunked", "prepare_split_kv_chunked"))
def test_runtime_window_not_in_compile_key(name):
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime.jit import create_function_from_signature

    from ctikernels.attention import _flashinfer_schedule

    kernel = getattr(_flashinfer_schedule, name)
    binder = create_function_from_signature(
        kernel.signature, kernel.params, CUDABackend(GPUTarget("cuda", 120, 32))
    )
    args = {p.name: 64 for p in kernel.params}
    keys = [binder(**(args | {"WINDOW_LEFT": w}))[1] for w in (-1, 0, 1, 63, 2047)]
    assert all(key == keys[0] for key in keys)


def test_chunked_schedule_interpreter():
    pytest.importorskip("triton")
    result = run_child(
        [sys.executable, str(Path(__file__).with_name("_draft_any_query_schedule.py"))],
        env=os.environ | {"TRITON_INTERPRET": "1"},
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_offline_large_query_lowering(tmp_path):
    pytest.importorskip("triton")
    pytest.importorskip("cutlass")
    result = run_child(
        [sys.executable, str(Path(__file__).with_name("_draft_any_query_compile.py"))],
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "CUTE_DSL_ARCH": "sm_120a",
            "TRITON_CACHE_DIR": str(tmp_path / "triton"),
        },
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_small_query_schedule_is_preserved_up_to_the_compiler_limit():
    for query in (1, 2, 16, 32):
        assert not chunked_schedule(4, query, 512)
        assert not chunked_schedule(2**20 // query, query, 512)
        assert chunked_schedule(2**20 // query + 1, query, 512)
        assert chunked_schedule(1, query, 2**20 + 1)
    assert chunked_schedule(1, 33, 1)
