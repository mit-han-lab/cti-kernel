# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Custom-mask byte ABI, workspace admission and routing, without CUDA."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.attention._draft import cute, flashinfer, flashinfer_mask
from ctikernels.attention.draft import DraftAttention, _provider

from ..support.child import run_child


def geometry(**overrides):
    return {
        "batch_size": 3,
        "query_length": 8,
        "max_kv_tokens": 8192,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 16,
        "dtype": torch.bfloat16,
        "causal": False,
        "window": 7,
        "device": torch.device("cuda"),
    } | overrides


@pytest.fixture(autouse=True)
def hardware(monkeypatch):
    monkeypatch.delenv("CTI_DRAFT_PROVIDER", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12,
            minor=0,
            multi_processor_count=170,
        ),
    )


def reject(**kwargs):
    raise NotImplementedError("CuTe unavailable or grid overflow")


@pytest.mark.parametrize("query", [2, 8, 33, 64, 65536])
def test_routing(monkeypatch, query):
    args = geometry(query_length=query, max_kv_tokens=max(8192, query), window=1)
    monkeypatch.setattr(cute, "check_support", reject)
    assert _provider(None, args) is flashinfer_mask
    assert _provider("flashinfer-mask", args) is flashinfer_mask
    assert DraftAttention.requires_page_indices(**args)
    with pytest.raises(NotImplementedError, match="no right bound"):
        _provider("flashinfer", args)
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "flashinfer-mask")
    for causal, window in [(False, None), (False, 128), (True, 7), (True, None)]:
        assert _provider(None, args | {"causal": causal, "window": window}) is flashinfer_mask
    monkeypatch.delenv("CTI_DRAFT_PROVIDER")
    monkeypatch.setattr(cute, "check_support", lambda **kwargs: None)
    assert _provider(None, args) is cute


@pytest.mark.parametrize("window", [1, 7, 128, None])
@pytest.mark.parametrize("query", [2, 33, 129])
def test_workspace(window, query):
    args = geometry(query_length=query, window=window)
    stride, mask, indptr = flashinfer_mask.mask_layout(**args)
    span = 8192 if window is None else min(8192, window - 1 + query + 15)
    assert stride == (query * span + 7) // 8
    assert mask == (3 * stride + 255) // 256 * 256
    assert indptr == 256
    base = DraftAttention.workspace_size(**(args | {"causal": True}), provider="flashinfer")
    result = DraftAttention.workspace_size(**args, provider="flashinfer-mask")
    assert result == (base[0], base[1] + mask + indptr)
    assert all(n % 256 == 0 for n in result)
    if window is not None:
        assert flashinfer_mask.mask_layout(**(args | {"max_kv_tokens": 131072})) == (
            stride,
            mask,
            indptr,
        )


def test_exact_byte_indptr_boundary():
    # Bit offsets exceed int32 here, but byte offsets still fit. Q=8 makes
    # byte stride equal to C, so B*C=M is the exact admission boundary.
    args = geometry(
        batch_size=1,
        query_length=8,
        max_kv_tokens=2**31 - 1,
        page_size=1,
        window=None,
        query_tile_size=128,
        query_tile_count=1,
        padded_ctas=340,
    )
    # Isolate the new buffer check from the unchanged frozen small-Q schedule.
    from unittest.mock import patch

    with patch.object(flashinfer, "check_capacity"):
        flashinfer_mask.check_capacity(**args)
        with pytest.raises(NotImplementedError, match="byte indptr"):
            flashinfer_mask.check_capacity(**(args | {"query_length": 9}))
    # A real large-Q admitted case whose flattened mask needs >2**31 bits.
    flashinfer_mask.check_capacity(
        **geometry(
            batch_size=1,
            query_length=65536,
            max_kv_tokens=65536,
            page_size=1,
            window=None,
            num_qo_heads=1,
            num_kv_heads=1,
        )
    )


def segment_packbits_python(bits, indptr):
    """Independent implementation of FlashInfer segment_packbits(..., 'little').

    quantization/packbits.py:82-108: segments round up separately to bytes;
    prefill.py passes a flattened [qo_len, kv_len] segment with bitorder=little.
    """
    packed, offsets = [], [0]
    for begin, end in zip(indptr[:-1], indptr[1:], strict=True):
        segment = bits[begin:end]
        for start in range(0, len(segment), 8):
            packed.append(sum(int(v) << bit for bit, v in enumerate(segment[start : start + 8])))
        offsets.append(len(packed))
    return packed, offsets


def test_packbits_documented_layout():
    # Same example segments as FlashInfer's documentation, using little order.
    packed, offsets = segment_packbits_python([1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1], [0, 4, 7, 11])
    assert packed == [13, 4, 11]
    assert offsets == [0, 1, 2, 3]


@pytest.mark.parametrize("name", ["pack_mask", "prepare_draft", "prepare_draft_chunked"])
def test_window_derived_arguments_are_runtime(name):
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.runtime.jit import create_function_from_signature

    from ctikernels.attention import _flashinfer_schedule
    from ctikernels.attention._draft.flashinfer_mask.prepare import pack_mask

    kernel = pack_mask if name == "pack_mask" else getattr(_flashinfer_schedule, name)
    runtime = (
        ("WINDOW_LEFT", "STRIDE", "BYTE_TILES")
        if name == "pack_mask"
        else ("WINDOW_LEFT", "TRIM_WINDOW_LEFT")
    )
    for param in kernel.params:
        if param.name in runtime:
            assert not param.is_constexpr
    binder = create_function_from_signature(
        kernel.signature, kernel.params, CUDABackend(GPUTarget("cuda", 120, 32))
    )
    args = {p.name: 64 for p in kernel.params}
    keys = [
        binder(**(args | dict.fromkeys(runtime, value)))[1] for value in (-1, 0, 1, 2, 16, 31, 2047)
    ]
    assert all(key == keys[0] for key in keys)


def _exercise_device_kernels():
    # Run the actual device algorithms on CPU in a fresh Triton interpreter.
    from ctikernels.attention._draft.flashinfer_mask.prepare import pack_mask
    from ctikernels.attention._flashinfer_schedule import prepare_draft, prepare_draft_chunked

    def ints(n):
        return torch.full((n,), -777, dtype=torch.int32)

    for query in (2, 8, 33, 129):
        for page in (1, 16):
            for window, causal in (
                (1, False),
                (7, False),
                (query - 1, False),
                (7, True),
                (None, False),
                (None, True),
            ):
                batch, capacity, tile, padded = 4, 301, 64, 128
                lengths = [capacity, query, 0, capacity - 1]
                slots = torch.tensor([0, 1, 2, -1], dtype=torch.int32)
                token_ptr = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(
                    0, dtype=torch.int32
                )
                tokens = torch.arange(sum(lengths), dtype=torch.int32)
                qp, pp, fixed = [ints(batch + 1) for _ in range(3)]
                last, total, pages = ints(batch), ints(1), ints(batch * capacity)
                schedule = [ints(padded) for _ in range(3)] + [
                    ints(batch + 1),
                    ints(batch * query + 1),
                    ints(1),
                    torch.zeros(padded, dtype=torch.bool),
                ]
                left = -1 if window is None else window - 1
                kernel = prepare_draft if query <= 32 else prepare_draft_chunked
                kernel[(batch,)](
                    slots,
                    token_ptr,
                    tokens,
                    qp,
                    pp,
                    pages,
                    last,
                    fixed,
                    total,
                    *schedule,
                    BATCH=batch,
                    QUERY=query,
                    PAGE=page,
                    PAGE_TILES=1,
                    GATHER_TILE=1024,
                    QUERY_TILE_COUNT=(query + tile - 1) // tile,
                    MAX_CTAS=padded,
                    PADDED_CTAS=padded,
                    REQUEST_BLOCK=4,
                    TILE_BLOCK=padded,
                    MERGE_BLOCK=1 << (batch * query - 1).bit_length(),
                    SEARCH_STEPS=batch.bit_length(),
                    WINDOW_LEFT=-1,
                    QUERY_TILE_SIZE=tile,
                    TRIM_WINDOW=True,
                    TRIM_WINDOW_LEFT=left,
                    WINDOWED=window is not None,
                )
                stride, _, _ = flashinfer_mask.mask_layout(
                    batch_size=batch,
                    query_length=query,
                    max_kv_tokens=capacity,
                    page_size=page,
                    window=window,
                )
                mask = torch.full((batch * stride,), 255, dtype=torch.uint8)
                indptr = ints(batch + 1)
                tiles = (stride + 255) // 256
                pack_mask[(batch * tiles,)](
                    pp,
                    last,
                    mask,
                    indptr,
                    BATCH=batch,
                    QUERY=query,
                    PAGE=page,
                    STRIDE=stride,
                    BYTE_TILES=tiles,
                    CAUSAL=causal,
                    WINDOWED=window is not None,
                    WINDOW_LEFT=left,
                )
                assert indptr.tolist() == [r * stride for r in range(batch + 1)]
                assert qp.tolist() == [0, query, 2 * query, 2 * query, 2 * query]
                assert schedule[3][-1] == schedule[3][-2] == schedule[3][-3]
                for request, original_length in enumerate(lengths):
                    length = original_length if request < 3 else 0
                    drop = (
                        0 if window is None else max(length - query - window + 1, 0) // page * page
                    )
                    kept = length - drop
                    assert pp[request + 1] - pp[request] == (kept + page - 1) // page
                    assert last[request] == ((length - 1) % page + 1 if length else 0)
                    expected_pages = (
                        tokens[token_ptr[request] + drop : token_ptr[request] + length : page]
                        // page
                    )
                    assert torch.equal(pages[pp[request] : pp[request + 1]], expected_pages)
                    # Build visibility in ORIGINAL request positions, then flatten
                    # the retained [Q, kv_len] matrix as FlashInfer's helper does.
                    bits = [
                        (not causal or key <= length - query + row)
                        and (window is None or abs(length - query + row - key) < window)
                        for row in range(query)
                        for key in range(drop, length)
                    ]
                    expected, _ = segment_packbits_python(bits, [0, len(bits)])
                    actual = mask[request * stride : (request + 1) * stride].tolist()
                    assert actual == expected + [0] * (stride - len(expected))


def test_device_prepare_and_packing_on_cpu():
    pytest.importorskip("triton")
    # By module name from the repository root: the file's package-relative imports need it.
    command = (
        f"import importlib; importlib.import_module({__spec__.name!r})._exercise_device_kernels()"
    )
    result = run_child(
        [sys.executable, "-c", command],
        env=os.environ | {"TRITON_INTERPRET": "1"},
        cwd=Path(__file__).resolve().parents[2],
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
