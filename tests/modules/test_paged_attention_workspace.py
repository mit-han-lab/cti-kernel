# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Properties and frozen planner oracle values; no duplicated native planner."""

from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import torch

from ctikernels.attention import _flashinfer
from ctikernels.attention._flashinfer import FlashInferFixedQueryAttention
from ctikernels.attention._split_tables import (
    XQA_DECODE_SUBSEQ,
    decode_uses_cuda_cores,
    prefill_split_batch_bound,
)
from ctikernels.attention.inference import FlashInferPagedAttention


@pytest.fixture(autouse=True)
def device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12, minor=0, multi_processor_count=188, max_threads_per_multi_processor=1536
        ),
    )


def size(**overrides):
    options = {
        "mode": "prefill",
        "backend": "fa2",
        "max_batch_size": 64,
        "max_query_tokens": 8192,
        "num_qo_heads": 24,
        "num_kv_heads": 4,
        "head_dim": 256,
        "device": torch.device("cuda"),
    }
    return FlashInferPagedAttention.workspace_size(**(options | overrides))


@pytest.mark.parametrize(
    "options,exact",
    [
        ({}, (148426752, 10496)),
        ({"mode": "decode"}, (15494144, 8192)),
        ({"mode": "decode", "use_tensor_cores": True}, (37106688, 1792)),
        ({"mode": "decode", "use_tensor_cores": True, "use_cuda_graph": True}, (37106688, 2048)),
        ({"backend": "fa3"}, (0, 86272)),
        # FA3 tile=192: query lengths [8129] + [1]*63 need 72,000 integer bytes.
        pytest.param({"backend": "fa3", "head_dim": 64}, (0, 72000), id="fa3-tile-192"),
    ],
)
def test_bounds(options, exact):
    assert all(bound >= value for bound, value in zip(size(**options), exact, strict=True))
    sizes = [size(**options, max_batch_size=b) for b in (1, 4, 64, 256, 512)]
    assert all(all(x <= y for x, y in zip(a, b, strict=True)) for a, b in pairwise(sizes))
    sizes = [size(**options, max_query_tokens=t) for t in (1, 16, 64, 8192, 65536)]
    assert all(all(x <= y for x, y in zip(a, b, strict=True)) for a, b in pairwise(sizes))
    assert all(f % 256 == i % 256 == 0 for f, i in sizes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "extend"},
        {"backend": "auto"},
        {"backend": "trtllm-gen"},
        {"max_batch_size": 0},
        {"max_query_tokens": None},
        {"max_query_tokens": 0},
        {"num_qo_heads": 13},
        {"head_dim": 0},
        {"dispatch": "unknown"},
    ],
)
def test_unsupported(overrides):
    with pytest.raises(ValueError):
        size(**overrides)


def test_requires_sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(major=7))
    with pytest.raises(ValueError, match="SM80"):
        size()


@pytest.mark.parametrize(
    "options,exact",
    [
        pytest.param({"mode": "decode"}, (8454144, 3584), id="cuda-empty-kv"),
        pytest.param(
            {"mode": "decode", "use_cuda_graph": True}, (8421120, 3584), id="cuda-graph-grid"
        ),
        # B=254 is just below the 255-request occupancy limit, so it still splits.
        pytest.param(
            {"mode": "decode", "use_cuda_graph": True, "max_batch_size": 254},
            (8421120, 3343),
            id="cuda-graph-grid-cutoff",
        ),
        # One request with 21,760 KV tokens fills 170 tile=64 split chunks.
        pytest.param(
            {"mode": "decode", "use_tensor_cores": True, "num_kv_heads": 2},
            (359301120, 2266),
            id="tensor-core-tile-64",
        ),
        pytest.param({"max_query_tokens": 16}, (177537024, 768), id="query-tile"),
        pytest.param({"max_query_tokens": 64}, (177537024, 1024), id="merge-rows"),
        pytest.param(
            {"mode": "decode", "use_tensor_cores": True, "num_kv_heads": 4, "max_batch_size": 15},
            (44912640, 1536),
            id="output-indptr-scalar-slack",
        ),
        pytest.param(
            {"num_kv_heads": 1, "head_dim": 256, "max_batch_size": 15, "max_query_tokens": 576},
            (1431633920, 7424),
            id="request-tail",
        ),
        # Frozen 3894dcf capacity at the 17B/16 short-tile allowance boundary.
        pytest.param(
            {"max_batch_size": 512, "max_query_tokens": 1088},
            (177537024, 14848),
            id="short-tile-allowance",
        ),
        pytest.param(
            {
                "mode": "decode",
                "use_tensor_cores": True,
                "use_cuda_graph": True,
                "max_batch_size": 31,
            },
            (24305664, 1024),
            id="graph-padding",
        ),
        pytest.param(
            {
                "mode": "decode",
                "use_tensor_cores": True,
                "use_cuda_graph": True,
                "max_batch_size": 64,
            },
            (50196480, 2048),
            id="graph-scalar",
        ),
    ],
)
def test_frozen_planner_boundaries(monkeypatch, options, exact):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12, minor=0, multi_processor_count=170, max_threads_per_multi_processor=1536
        ),
    )
    declared = size(
        **(
            {
                "num_qo_heads": 64,
                "num_kv_heads": 8,
                "head_dim": 128,
                "max_batch_size": 1,
                "max_query_tokens": 1,
            }
            | options
        )
    )
    assert all(bound >= need for bound, need in zip(declared, exact, strict=True))


@pytest.mark.parametrize("sms", [170, 188])
def test_mixed_short_tiles(monkeypatch, sms):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12, minor=0, multi_processor_count=sms, max_threads_per_multi_processor=1536
        ),
    )
    # B<=256, T<=576 includes [288] + [1]*255. Its average packed rows select
    # tile=16, although the total-token bound selects tile=128. The unsplit
    # planner uses 399 tiles; the frozen oracle bounds it with 7424 integer bytes.
    # Using only ceil(T*group/128) plus request tails would declare just 4608 bytes.
    assert (
        size(
            num_qo_heads=64, num_kv_heads=8, head_dim=128, max_batch_size=256, max_query_tokens=576
        )[1]
        >= 7424
    )


@pytest.mark.parametrize(
    "batch,exact",
    [
        (4064, 7339008),
        (4065, 7340800),
        (4066, 115456),
        (4095, 116224),
        (4096, 116224),
        (4097, 116480),
    ],
)
def test_fa3_head_schedule_transition(monkeypatch, batch, exact):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=9, minor=0, multi_processor_count=132, max_threads_per_multi_processor=1536
        ),
    )
    # ceil(B/128)+B-1 crosses 4096 at B=4066; the native planner stops
    # scheduling heads separately. Each graph has a fixed batch size.
    declared = size(
        mode="decode",
        backend="fa3",
        use_tensor_cores=True,
        use_cuda_graph=True,
        max_batch_size=batch,
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=128,
    )
    assert declared[0] == 0
    assert declared[1] >= exact


def test_fa3_eager_bound_covers_smaller_batch(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=9, minor=0, multi_processor_count=132, max_threads_per_multi_processor=1536
        ),
    )
    # Eager limits must cover smaller batches: 4,065 one-token requests still
    # schedule heads separately and need 7,285,024 bytes, even under a 4,066 limit.
    declared = size(
        mode="decode",
        backend="fa3",
        use_tensor_cores=True,
        max_batch_size=4066,
        num_qo_heads=64,
        num_kv_heads=8,
        head_dim=128,
    )
    assert declared[0] == 0
    assert declared[1] >= 7285024


def test_qualification_tracks_dependency_pin(monkeypatch):
    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]
    # Bumping either the adapter qualification or dependency requires rechecking
    # these frozen planner values and the native workspace tests together.
    qualified = "0.7.0.post1"
    assert f"flashinfer-python=={qualified}" in project["dependencies"]
    monkeypatch.setattr(_flashinfer, "version", lambda _: "unqualified")
    with pytest.raises(RuntimeError) as error:
        FlashInferFixedQueryAttention(
            None,
            None,
            query_indptr=None,
            kv_indptr=None,
            kv_indices=None,
            last_page_lengths=None,
            query_length=16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
        )
    assert qualified in _flashinfer.QUALIFIED_VERSIONS
    assert str(error.value).startswith("fixed-query Split-KV is qualified with flashinfer-python==")
    assert all(f"=={known}" in str(error.value) for known in _flashinfer.QUALIFIED_VERSIONS)
    assert str(error.value).endswith("; found unqualified")


@pytest.mark.parametrize("candidate", ["0.6.11.post3", "0.7.0.post1", "0.7.1"])
def test_qualified_versions_pass_the_gate(monkeypatch, candidate):
    """Allow-listed releases get past the version gate (they fail later on the CPU inputs)."""
    monkeypatch.setattr(_flashinfer, "version", lambda _: candidate)
    with pytest.raises(Exception) as error:
        FlashInferFixedQueryAttention(
            None,
            None,
            query_indptr=None,
            kv_indptr=None,
            kv_indices=None,
            last_page_lengths=None,
            query_length=16,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
        )
    assert "qualified with" not in str(error.value)


def test_qualified_versions_are_an_explicit_allow_list():
    assert _flashinfer.QUALIFIED_VERSIONS == ("0.6.11.post3", "0.7.0.post1", "0.7.1")


@pytest.fixture(params=[(12, 0, 170), (12, 0, 188), (12, 1, 48), (9, 0, 132)])
def dispatch_device(request, monkeypatch):
    major, minor, sms = request.param
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=major,
            minor=minor,
            multi_processor_count=sms,
            max_threads_per_multi_processor=1536,
        ),
    )
    return {(12, 0, 170): "rtx5090", (12, 0, 188): "rtxpro6000", (12, 1, 48): "gb10"}.get(
        request.param
    )


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 64])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 4, 256)])
def test_dispatch_decode_sizes(dispatch_device, batch, graph, heads):
    options = {
        "mode": "decode",
        "use_tensor_cores": True,
        "use_cuda_graph": graph,
        "max_batch_size": batch,
        "num_qo_heads": heads[0],
        "num_kv_heads": heads[1],
        "head_dim": heads[2],
    }
    provider = size(**options, dispatch="provider")
    advertised = size(**options)
    for candidate in [batch] if graph else range(1, batch + 1):
        subseq = XQA_DECODE_SUBSEQ.get((dispatch_device, *heads), {}).get(candidate)
        if subseq is not None:
            assert advertised[0] >= _flashinfer.xqa_decode_scratch_size(
                candidate, heads[1], subseq, heads[2], heads[0] // heads[1]
            )
    # Eager plans take the graph kernel of their batch, sized as a graph plan.
    expected = provider
    for candidate in [batch] if graph else range(1, batch + 1):
        if decode_uses_cuda_cores(torch.device("cuda"), *heads, candidate):
            cuda_core = size(
                **options
                | {"use_tensor_cores": False, "use_cuda_graph": True, "max_batch_size": candidate},
                dispatch="provider",
            )
            expected = tuple(max(a, b) for a, b in zip(expected, cuda_core, strict=True))
            assert advertised[1] > provider[1]
    assert advertised == expected


@pytest.mark.parametrize("batch,tokens", [(1, 4), (1, 16), (1, 64), (2, 44), (5, 256), (8, 512)])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_dispatch_prefill_sizes(dispatch_device, batch, tokens, head_dim):
    options = {
        "max_batch_size": batch,
        "max_query_tokens": tokens,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": head_dim,
    }
    bound = prefill_split_batch_bound(dispatch_device, batch, tokens) if head_dim == 128 else None
    provider = size(**options, dispatch="provider")
    routed = size(**(options | {"max_batch_size": max(batch, bound or 0)}), dispatch="provider")
    assert size(**options) == tuple(max(a, b) for a, b in zip(provider, routed, strict=True))
    if bound is None:
        assert size(**options) == provider


@pytest.mark.parametrize(
    "batch,tokens", [(1, 1), (4, 4), (4, 67), (4, 512), (256, 576), (512, 65536)]
)
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_graph_prefill_bound_covers_native_allocations(batch, tokens, head_dim, backend):
    options = {
        "use_cuda_graph": True,
        "max_batch_size": batch,
        "max_query_tokens": tokens,
        "head_dim": head_dim,
        "backend": backend,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
    }
    floating, integer = size(**options, dispatch="provider")
    # Device dispatch may add table routes (padded split / tile steering) on top.
    device_floating, device_integer = size(**options, dispatch="device")
    assert device_floating >= floating and device_integer >= integer
    # Independent allocator walk: graph PrefillPlan always reserves total_rows,
    # even when actual query lengths shrink on replay. Include both KV byte widths'
    # possible tiles, plus the newer large-head tile=32.
    if backend == "fa3":
        tile = 192 if head_dim == 64 else 128
        works = (tokens + tile - 1) // tile + batch - 1
        if works <= 4096:
            works *= 32
        offset = 0
        for count in (works,) * 6 + (189, works):
            offset = (offset + 15) // 16 * 16 + count * 4
        assert floating == 0 and integer >= offset
        return
    # A graph's batch is fixed by its buffers; any fixed batch up to the bound is admitted.
    # Row totals below the count (zero-length requests) wrap the longest request.
    for requests in range(1, batch + 1):
        for total in {tokens, min(requests - 1, tokens)} - {0}:
            _check_graph_prefill_walk(floating, integer, requests, total, head_dim)


def _check_graph_prefill_walk(floating, integer, batch, tokens, head_dim):
    # FA2DetermineCtaTileQ on any replay's longest packed request (<= the declared
    # total_rows - batch + 1, unsigned): 0.7.0.post1 picks 16/32 above D256, else 16/64/128.
    longest = (tokens - batch + 1 if tokens >= batch else 1 << 62) * 4
    if head_dim >= 512:
        choices = {16 if packed <= 32 else 32 for packed in (1, longest)}
    else:
        choices = {16, 64, 128} if head_dim < 256 else {16, 64}
        choices = {t for t in choices if t == 16 or longest > (64 if t == 128 else 16)}
    for tile in sorted(choices):
        count = max(2 * 188 // 8, (tokens * 4 + tile - 1) // tile + batch - 1)
        assert floating >= 32 * count * tile * (head_dim + 1) * 4
        offset = 0
        for count_bytes, alignment in (
            (count * 4, 16),
            (count * 4, 16),
            (count * 4, 16),
            ((batch + 1) * 4, 16),
            (4, 1),
            (4, 16),
            ((tokens + 1) * 4, 16),
            (count, 16),
        ):
            offset = (offset + alignment - 1) // alignment * alignment + count_bytes
        assert integer >= offset


def test_graph_prefill_bound_covers_smaller_fixed_batch():
    # Sized for B64 x 64 tokens (32/8 heads, D128, 188 SMs): a B1 x 64 graph picks
    # tile 128, not the B64 tile 16, and needs ~99.3 MB of float workspace.
    options = {
        "use_cuda_graph": True,
        "max_query_tokens": 64,
        "head_dim": 128,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
    }
    sizes = size(**options, max_batch_size=64)
    assert sizes[0] >= 99_000_000
    for batch in (1, 2, 4, 17, 63):
        smaller = size(**options, max_batch_size=batch)
        assert all(a >= b for a, b in zip(sizes, smaller, strict=True))


@pytest.mark.parametrize("requests,total", [(64, 1), (64, 63), (2, 1), (17, 3)])
def test_graph_prefill_bound_covers_zero_length_requests(requests, total):
    # Sized for B64 x 64 tokens (32/8 heads, D128, 188 SMs): qo_indptr [0, 1, 1, ..., 1]
    # wraps FlashInfer's unsigned longest request to tile 128 (~135.3 MB of float workspace).
    floating, integer = size(
        use_cuda_graph=True,
        max_batch_size=64,
        max_query_tokens=64,
        head_dim=128,
        num_qo_heads=32,
        num_kv_heads=8,
    )
    if (requests, total) == (64, 1):
        assert floating >= 135_266_304
    _check_graph_prefill_walk(floating, integer, requests, total, 128)
