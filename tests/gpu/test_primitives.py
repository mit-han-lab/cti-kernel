# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

import weakref

import pytest
import torch

from ctikernels.index import indexing
from ctikernels.store import _jit_store_module, store_cache
from ctikernels.tensor import weak_tensor_alias
from ctikernels.utils import KernelConfig

# Cover both index types at each byte width, sharing just three JIT specializations
# per kernel. Indexing uses 1, 2, and 4 splits for 512, 1024, and 2048 bytes.
_COPY_CASES = (
    pytest.param(torch.bfloat16, 256, torch.int32, id="bf16-512B-i32"),
    pytest.param(torch.bfloat16, 512, torch.int64, id="bf16-1024B-i64"),
    pytest.param(torch.bfloat16, 1024, torch.int32, id="bf16-2048B-i32"),
    pytest.param(torch.float32, 128, torch.int64, id="fp32-512B-i64"),
    pytest.param(torch.float16, 512, torch.int32, id="fp16-1024B-i32"),
    pytest.param(torch.float32, 512, torch.int64, id="fp32-2048B-i64"),
)


_STORE_CASES = _COPY_CASES + (
    pytest.param(torch.bfloat16, 128, torch.int32, id="bf16-256B-i32"),
    pytest.param(torch.bfloat16, 128, torch.int64, id="bf16-256B-i64"),
    pytest.param(torch.bfloat16, 2048, torch.int32, id="bf16-4096B-i32"),
    pytest.param(torch.bfloat16, 2048, torch.int64, id="bf16-4096B-i64"),
    # 128/384/640-byte rows move one 4-byte vector per lane.
    pytest.param(torch.bfloat16, 64, torch.int32, id="bf16-128B-i32"),
    pytest.param(torch.bfloat16, 192, torch.int64, id="bf16-384B-i64"),
    pytest.param(torch.bfloat16, 320, torch.int32, id="bf16-640B-i32"),
)


def _row_pieces(row_bytes: int) -> int:
    # Warp-sized pieces per row, one 16-, 8- or 4-byte vector per lane.
    unit = next(unit for unit in (16, 8, 4) if row_bytes % (32 * unit) == 0)
    return row_bytes // (32 * unit)


@pytest.mark.parametrize("dtype,width,index_dtype", _COPY_CASES)
def test_indexing_matches_gather_with_allocated_and_owned_output(
    dtype: torch.dtype, width: int, index_dtype: torch.dtype
) -> None:
    torch.manual_seed(41)
    weights = torch.randn((11, width), device="cuda", dtype=dtype)
    indices = torch.tensor([10, 0, 5, 5, 1], device="cuda", dtype=index_dtype)
    expected = weights.index_select(0, indices.long())

    actual = indexing(weights, indices)
    assert actual.data_ptr() != weights.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    output = torch.full_like(expected, float("nan"))
    assert indexing(weights, indices, output=output) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("vocab_range", [None, (0, 11)])
def test_indexing_empty_batch_is_empty(vocab_range) -> None:
    weights = torch.randn((11, 256), device="cuda", dtype=torch.bfloat16)
    indices = torch.empty(0, device="cuda", dtype=torch.int32)
    actual = indexing(weights, indices, vocab_range=vocab_range)
    assert actual.shape == (0, 256) and actual.dtype == weights.dtype
    output = weights.new_empty(0, 256)
    assert indexing(weights, indices, output=output, vocab_range=vocab_range) is output


@pytest.mark.parametrize("dtype,width,index_dtype", _COPY_CASES)
def test_indexing_vocab_range_offsets_local_rows_and_zeros_outside(
    dtype: torch.dtype, width: int, index_dtype: torch.dtype
) -> None:
    torch.manual_seed(42)
    start, length = 17, 11
    weights = torch.randn((length, width), device="cuda", dtype=dtype)
    indices = torch.tensor([16, 17, 27, 28, 22, 22, 0], device="cuda", dtype=index_dtype)
    local_indices = indices.long() - start
    valid = (local_indices >= 0) & (local_indices < length)
    expected = torch.zeros((indices.numel(), width), device="cuda", dtype=dtype)
    expected[valid] = weights.index_select(0, local_indices[valid])
    output = torch.full_like(expected, float("nan"))

    assert indexing(weights, indices, output=output, vocab_range=(start, length)) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,width,index_dtype", _STORE_CASES)
@pytest.mark.parametrize(
    "slots",
    [
        pytest.param([12, 0, 7, 3, 10], id="valid"),
        pytest.param([-1, 0, 7, -1, 10], id="mixed"),
        pytest.param([-1, -1, -1, -1, -1], id="all-negative"),
        pytest.param([-2, 0, -2147483648, -17, 10], id="other-negative"),
    ],
)
def test_store_cache_scatters_strided_kv_and_preserves_untouched_storage(
    dtype: torch.dtype, width: int, index_dtype: torch.dtype, slots: list[int]
) -> None:
    torch.manual_seed(43)
    num_tokens, num_slots, head_dim = 5, 13, 64
    num_heads = width // head_dim
    k_storage = torch.randn((2 * (num_slots + 1), num_heads, head_dim), device="cuda", dtype=dtype)
    v_storage = torch.randn_like(k_storage)
    # Include prefix storage so an erroneous write to slot -1 is observable too.
    k_cache, v_cache = k_storage[2::2], v_storage[2::2]
    # QKV slices have contiguous rows but a larger, matching K/V row stride.
    qkv = torch.randn((num_tokens, 3 * width), device="cuda", dtype=dtype)
    k, v = qkv[:, width : 2 * width], qkv[:, 2 * width :]
    original_qkv = qkv.clone()
    indices = torch.tensor(slots, device="cuda", dtype=index_dtype)
    expected_k, expected_v = k_storage.clone(), v_storage.clone()
    valid = indices >= 0
    expected_k[2 + 2 * indices[valid].long()] = k.reshape(num_tokens, num_heads, head_dim)[valid]
    expected_v[2 + 2 * indices[valid].long()] = v.reshape(num_tokens, num_heads, head_dim)[valid]

    assert not k.is_contiguous() and k.stride() == v.stride()
    assert not k_cache.is_contiguous() and k_cache.stride() == v_cache.stride()
    assert store_cache(k_cache, v_cache, indices, k, v) is None

    # Compare bytes of the full allocations: prefix, gaps and unselected slots,
    # including the last cache row (which Python-style wrapping would overwrite).
    torch.testing.assert_close(
        k_storage.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_storage.view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        qkv.view(torch.uint8), original_qkv.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 4, 16, 64, 256, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("width", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "skip_step", [pytest.param(3, id="mixed"), pytest.param(1, id="all-negative")]
)
def test_store_cache_skips_negative_indices_across_batch_sizes(
    num_tokens: int, width: int, index_dtype: torch.dtype, skip_step: int
) -> None:
    torch.manual_seed(45)
    dev = torch.device("cuda")
    num_slots = num_tokens + 17
    # Guard rows catch slot -1 writes; padding catches incorrect row strides.
    k_storage = torch.randn((num_slots + 2, width + 64), device=dev, dtype=torch.bfloat16)
    v_storage = torch.randn_like(k_storage)
    k_cache, v_cache = k_storage[1:-1, :width], v_storage[1:-1, :width]
    qkv = torch.randint(-32768, 32768, (num_tokens, 3 * width), device=dev, dtype=torch.int16).view(
        torch.bfloat16
    )
    k, v = qkv[:, width : 2 * width], qkv[:, 2 * width :]
    original_qkv = qkv.clone()
    indices = torch.randperm(num_slots, device=dev, dtype=index_dtype)[:num_tokens]
    indices[::skip_step] = -1
    expected_k, expected_v = k_storage.clone(), v_storage.clone()
    valid = indices >= 0
    expected_k[1 + indices[valid].long(), :width] = k[valid]
    expected_v[1 + indices[valid].long(), :width] = v[valid]

    assert k_cache.stride(0) > width and k_cache.stride() == v_cache.stride()
    assert k.stride(0) > width and k.stride() == v.stride()
    store_cache(k_cache, v_cache, indices, k, v)
    # Check the entire backing allocations, including both guards and padding.
    torch.testing.assert_close(
        k_storage.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_storage.view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        qkv.view(torch.uint8), original_qkv.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("width", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("warp_threshold", [1, 2, 4])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("extra_token", [0, 1], ids=["boundary", "boundary-plus-one"])
def test_store_cache_skips_negative_indices_at_dispatch_boundary(
    width: int, index_dtype: torch.dtype, warp_threshold: int, extra_token: int
) -> None:
    torch.manual_seed(46)
    dev = torch.device("cuda")
    splits = max(1, width * 2 // 512)
    boundary = (
        warp_threshold * torch.cuda.get_device_properties(dev).multi_processor_count // splits
    )
    num_tokens, num_slots = boundary + extra_token, boundary + 17
    # Last batch at this block size and first batch at the next one.
    k_cache = torch.randn((num_slots, width), device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    k = torch.randn((num_tokens, width), device=dev, dtype=k_cache.dtype)
    v = torch.randn_like(k)
    indices = torch.randperm(num_slots, device=dev, dtype=index_dtype)[:num_tokens]
    indices[::3] = -1
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    valid = indices >= 0
    expected_k[indices[valid].long()] = k[valid]
    expected_v[indices[valid].long()] = v[valid]
    store_cache(k_cache, v_cache, indices, k, v)
    torch.testing.assert_close(
        k_cache.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_cache.view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("width", [64, 192, 320, 512, 1024, 2048])
@pytest.mark.parametrize("band_warps", [1024, 8192, 32768])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "extra_token", [-1, 0, 1], ids=["below-band-edge", "band-edge", "band-edge-plus-one"]
)
def test_store_cache_skips_negative_indices_at_size_band_edges(
    width: int, band_warps: int, index_dtype: torch.dtype, extra_token: int
) -> None:
    torch.manual_seed(47)
    dev = torch.device("cuda")
    # Batches on both sides of each size band edge (one row per block up to
    # 1024 warps for rows of at least 2 KiB, the K-then-V copy from 8192 to
    # 32768 warps).
    num_tokens = band_warps // _row_pieces(width * 2) + extra_token
    num_slots = num_tokens + 17
    k_storage = torch.randn((num_slots + 2, width + 64), device=dev, dtype=torch.bfloat16)
    v_storage = torch.randn_like(k_storage)
    k_cache, v_cache = k_storage[1:-1, :width], v_storage[1:-1, :width]
    k = torch.randn((num_tokens, width), device=dev, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indices = torch.randperm(num_slots, device=dev, dtype=index_dtype)[:num_tokens]
    indices[::3] = -1
    expected_k, expected_v = k_storage.clone(), v_storage.clone()
    valid = indices >= 0
    expected_k[1 + indices[valid].long(), :width] = k[valid]
    expected_v[1 + indices[valid].long(), :width] = v[valid]
    store_cache(k_cache, v_cache, indices, k, v)
    torch.testing.assert_close(
        k_storage.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        v_storage.view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [128, 512, 1024, 2048])
@pytest.mark.parametrize("num_tokens", [3, 2048])
@pytest.mark.parametrize("use_pdl", [False, True])
def test_store_cache_graph_replay_observes_in_place_index_updates(
    index_dtype: torch.dtype, width: int, num_tokens: int, use_pdl: bool
) -> None:
    torch.manual_seed(44)
    dev = torch.device("cuda")
    num_slots = num_tokens + 4
    k_storage = torch.randn((num_slots + 2, width), device=dev, dtype=torch.bfloat16)
    v_storage = torch.randn_like(k_storage)
    k_cache, v_cache = k_storage[1:-1], v_storage[1:-1]
    qkv = torch.randn((num_tokens, 3 * width), device=dev, dtype=k_cache.dtype)
    k, v = qkv[:, width : 2 * width], qkv[:, 2 * width :]
    launch = _jit_store_module(
        width * 2, config=KernelConfig(num_threads=256, max_occupancy=1, use_pdl=use_pdl)
    ).launch
    indices = torch.randperm(num_slots, device=dev, dtype=index_dtype)[:num_tokens]
    index_pointer = indices.data_ptr()
    # Warm up JIT loading before capture on a nondefault stream.
    launch(k_cache, v_cache, indices, k, v)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch(k_cache, v_cache, indices, k, v)

    # Two mixed replays, then all skipped, then fully valid, with unique slots.
    for skip_step in (3, 2, 1, None):
        indices.copy_(torch.randperm(num_slots, device=dev, dtype=index_dtype)[:num_tokens])
        if skip_step is not None:
            indices[::skip_step] = -1
            indices[0] = torch.iinfo(index_dtype).min
            if index_dtype == torch.int64 and num_tokens > 1:
                indices[-1] = -(1 << 32)
        expected_k, expected_v = k_storage.clone(), v_storage.clone()
        valid = indices >= 0
        expected_k[1 + indices[valid].long()] = k[valid]
        expected_v[1 + indices[valid].long()] = v[valid]
        graph.replay()
        torch.testing.assert_close(
            k_storage.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            v_storage.view(torch.uint8), expected_v.view(torch.uint8), rtol=0, atol=0
        )
        assert indices.data_ptr() == index_pointer


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "width,num_tokens",
    [
        # One row per block (at most 1024 warps, rows of at least 2 KiB).
        (1024, 256),
        (2048, 64),
        # Default copy below and between the bands.
        (512, 64),
        (1024, 1024),
        (512, 4095),
        (64, 8191),
        # K-then-V copy (8192-32768 warps), at both edges and inside.
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (64, 8192),
        (320, 6553),
        # Default copy above the bands.
        (2048, 4097),
        (2048, 8192),
    ],
)
def test_store_cache_early_pdl_trigger_preserves_dependent_graph_copies(
    index_dtype: torch.dtype, width: int, num_tokens: int
) -> None:
    k = torch.zeros((num_tokens, width), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    indices = torch.arange(num_tokens, device="cuda", dtype=index_dtype)
    # Ping-pong cache rows are the next grid's input: an early trigger must not
    # let its loads run before the preceding grid has finished storing.
    keys = [torch.zeros_like(k), torch.zeros_like(k)]
    values = [torch.zeros_like(v), torch.zeros_like(v)]
    store_cache(keys[0], values[0], indices, k, v)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        store_cache(keys[0], values[0], indices, k, v)
        for i in range(1, 50):
            src, dst = (i - 1) % 2, i % 2
            store_cache(keys[dst], values[dst], indices, keys[src], values[src])
    for generation in (1, 2):
        k.fill_(generation)
        v.fill_(-generation)
        graph.replay()
        torch.testing.assert_close(keys[1].view(torch.uint8), k.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(values[1].view(torch.uint8), v.view(torch.uint8), rtol=0, atol=0)


def test_indexing_and_store_cache_capture_on_the_current_stream() -> None:
    weights = torch.zeros((7, 256), device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([6, 0, 3], device="cuda", dtype=torch.int32)
    output = torch.empty((3, 256), device="cuda", dtype=weights.dtype)
    k_cache, v_cache = torch.zeros_like(weights), torch.zeros_like(weights)
    # Warm up JIT loading before capture on a nondefault stream.
    indexing(weights, indices, output=output)
    store_cache(k_cache, v_cache, indices, output, output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        indexing(weights, indices, output=output)
        store_cache(k_cache, v_cache, indices, output, output)

    expected = torch.zeros_like(weights)
    for value in (7, 11):
        weights.fill_(value)
        output.fill_(float("nan"))
        k_cache.zero_()
        v_cache.zero_()
        graph.replay()
        expected[indices.long()] = value
        torch.testing.assert_close(k_cache, expected, rtol=0, atol=0)
        torch.testing.assert_close(v_cache, expected, rtol=0, atol=0)


def test_weak_tensor_alias_shares_strided_data_without_retaining_python_owners() -> None:
    owner = torch.empty((8, 32), device="cuda", dtype=torch.float32)
    owner.copy_(torch.arange(owner.numel(), device="cuda").reshape_as(owner))
    source = owner[1::2, 4:28:2]
    alias = weak_tensor_alias()(source)
    owner_ref, source_ref = weakref.ref(owner), weakref.ref(source)

    assert alias is not source
    assert alias.data_ptr() == source.data_ptr()
    assert alias.shape == source.shape and alias.stride() == source.stride()
    assert alias.dtype == source.dtype and alias.device == source.device
    source.add_(3)
    torch.testing.assert_close(alias, source, rtol=0, atol=0)
    alias.mul_(2)
    torch.testing.assert_close(alias, source, rtol=0, atol=0)

    del source
    assert source_ref() is None
    assert owner_ref() is owner
    torch.testing.assert_close(alias, owner[1::2, 4:28:2], rtol=0, atol=0)
    torch.cuda.current_stream().synchronize()
    del owner
    assert owner_ref() is None
    # The alias is now dangling: do not read it or launch another GPU operation.
    del alias


def test_weak_tensor_alias_cuda_graph_replay_observes_in_place_input_updates() -> None:
    owner = torch.zeros((7, 32), device="cuda", dtype=torch.float32)
    alias = weak_tensor_alias()(owner)
    output = torch.empty_like(owner)
    pointers = owner.data_ptr(), alias.data_ptr(), output.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.add(alias, 3, out=output)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.add(alias, 3, out=output)

    for value in (2, 11):
        owner.fill_(value)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, owner + 3, rtol=0, atol=0)
        assert (owner.data_ptr(), alias.data_ptr(), output.data_ptr()) == pointers
    # Keep the actual storage owner alive until every replay has completed.
    torch.cuda.current_stream().synchronize()
