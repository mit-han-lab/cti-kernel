# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contract/oracle checks; these do not qualify native CUDA execution."""

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from ctikernels._storage import disjoint, reject_grad
from ctikernels.attention.draft import DraftAttention, _geometry

from .._dflash_reference import draft_attention_reference, mask_predicate
from ..support.child import run_child


@pytest.mark.parametrize("q", [4, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [1, 2, 7, 64, 2048, 4096, 8191, None])
def test_geometry_accepts_draft_and_verify_masks_without_cuda(q, causal, window):
    _geometry(
        batch_size=3,
        query_length=q,
        max_kv_tokens=q,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=16,
        dtype=torch.bfloat16,
        causal=causal,
        window=window,
        device="cuda",
    )


@pytest.mark.parametrize("provider", ["cute-splitkv", "flashinfer-fa2-device-schedule"])
@pytest.mark.parametrize("operand", ["q", "k", "v", "out"])
@pytest.mark.parametrize("offset", [1, 8])
def test_draft_operand_alignment_before_dispatch(provider, operand, offset):
    # Exercise the real run validation on CPU, replacing only device execution.
    op = DraftAttention.__new__(DraftAttention)
    op._provider_name = provider
    op._device, op._dtype = torch.device("cpu"), torch.bfloat16
    op._b, op._q, op._p, op._hq, op._hk, op._d = 1, 8, 1, 16, 4, 128
    op._query_row_stride = 16 * 128
    op._metadata = op._readonly = op._writable = ()
    op._bind()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    op._implementation = SimpleNamespace(run=run)
    operands = {
        "q": torch.empty((8, 16, 128), dtype=torch.bfloat16),
        "k": torch.empty((16, 1, 4, 128), dtype=torch.bfloat16),
        "v": torch.empty((16, 1, 4, 128), dtype=torch.bfloat16),
        "out": torch.empty((8, 16, 128), dtype=torch.bfloat16),
    }
    original = operands[operand]
    view = torch.empty(original.numel() + offset, dtype=original.dtype)[offset:].view_as(original)
    assert view.is_contiguous()
    assert view.data_ptr() % 16 == (offset * 2) % 16
    operands[operand] = view
    args = {
        "q": operands["q"],
        "paged_kv_cache": (operands["k"], operands["v"]),
        "out": operands["out"],
    }
    if offset == 1:
        with pytest.raises(ValueError, match="16-byte"):
            op.run(**args)
        assert not calls
    else:
        assert op.run(**args) is operands["out"]
        assert len(calls) == 1


@pytest.mark.parametrize(
    "name",
    [
        "request_indices",
        "token_indptr",
        "token_indices",
        "query_indptr",
        "page_indptr",
        "page_indices",
        "last_page_lengths",
    ],
)
@pytest.mark.parametrize("offset", [1, 4])
def test_draft_int32_metadata_alignment(name, offset):
    # frombuffer can express byte offsets that typed tensor slicing cannot.
    value = torch.frombuffer(bytearray(16), dtype=torch.int32, count=2, offset=offset)
    assert value.is_contiguous()
    assert value.data_ptr() % 4 == offset % 4
    op = DraftAttention.__new__(DraftAttention)
    op._device = torch.device("cpu")
    op._metadata = ((value, (2,), torch.int32, name),)
    if offset == 1:
        with pytest.raises(ValueError, match=f"{name}.*4-byte"):
            op._validate_metadata()
    else:
        op._validate_metadata()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [1, 2, 7, 64, 2048, 4096, 8191, None])
def test_mask_predicate_boundaries_and_overflow(causal, window):
    w = window or 4096
    for position in (w + 1, 2**31 + 20, 2**63 - 1 - w - 1):
        p = torch.tensor([position], dtype=torch.int64)
        distances = [-w - 1, -w, -w + 1, -1, 0, 1, w - 1, w, w + 1]
        keys = torch.tensor([position + d for d in distances], dtype=torch.int64)
        expected = [(window is None or abs(d) < w) and (not causal or d <= 0) for d in distances]
        assert mask_predicate(p, keys, causal=causal, window=window).tolist() == [expected]


def test_fp32_dense_oracle_agrees_with_fp64_on_quantized_inputs():
    gen = torch.Generator().manual_seed(2718)
    q = torch.randn((16, 4, 16), generator=gen).bfloat16()
    k = torch.randn((50, 1, 2, 16), generator=gen).bfloat16()
    v = torch.randn((50, 1, 2, 16), generator=gen).bfloat16()
    for causal, window in ((False, None), (True, 4096), (False, 2048), (True, None)):
        args = {
            "slots": torch.tensor([1, -1]),
            "positions": torch.tensor(list(range(42, 50)) + [-(2**63)] * 8),
            "token_indptr": torch.tensor([0, 50, 50]),
            "token_indices": torch.arange(50).flip(0),
            "query_length": 8,
            "causal": causal,
            "window": window,
        }
        actual = draft_attention_reference(q, k, v, **args)
        reference = draft_attention_reference(q, k, v, precision=torch.float64, **args)
        torch.testing.assert_close(actual.double(), reference, atol=2e-6, rtol=2e-6)
        assert torch.count_nonzero(actual[8:]) == 0


def test_workspace_includes_fixed_query_csr_and_alignment(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, multi_processor_count=170),
    )
    previous = 0
    for b in (1, 2, 3, 4):
        floating, integer = DraftAttention.workspace_size(
            batch_size=b,
            query_length=16,
            max_kv_tokens=4096,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            page_size=16,
            dtype=torch.bfloat16,
            causal=False,
            window=2048,
            device="cuda",
            provider="flashinfer",
        )
        assert floating >= previous
        assert integer >= (b + 1) * 4 + 256
        assert floating % 256 == integer % 256 == 0
        previous = floating


@pytest.mark.parametrize(
    "changes",
    [
        {"query_length": 0},
        {"max_kv_tokens": 4},
        {"batch_size": 2**30},
        {"dtype": torch.float32},
        {"causal": 1},
        {"window": True},
        {"window": 4096.0},
        {"window": 0},
        {"window": -1},
        {"device": "cpu"},
        {"num_qo_heads": 15},
    ],
)
def test_invalid_configuration_fails_before_provider_or_cuda(monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA queried before validation")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    args = {
        "batch_size": 1,
        "query_length": 16,
        "max_kv_tokens": 4096,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": False,
        "window": None,
        "device": "cuda",
    }
    with pytest.raises(ValueError):
        DraftAttention.workspace_size(**(args | changes))


def test_overlap_and_grad_checks_use_host_metadata():
    value = torch.empty(32)
    disjoint((value[:8], value[8:16]), (value[16:],))
    with pytest.raises(ValueError, match="overlap"):
        disjoint((value[:8],), (value[7:16],))
    with torch.no_grad(), pytest.raises(ValueError, match="gradients"):
        reject_grad(value.requires_grad_())


def test_device_metadata_arithmetic_in_cpu_triton_interpreter():
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton interpreter unavailable")
    environment = os.environ.copy()
    environment.update(TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    result = run_child(
        [sys.executable, "-m", "tests.modules._dflash_interpreter"], env=environment, timeout=180
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_provider_support_checks_exact_abi_and_sm12_toolkit(monkeypatch, capability):
    from ctikernels.attention._draft import flashinfer as provider

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(provider, "version", lambda _: "0.6.11.post3")
    provider.check_support(torch.device("cuda"))
    monkeypatch.setattr(torch.version, "cuda", "12.7")
    with pytest.raises(NotImplementedError, match="CUDA"):
        provider.check_support(torch.device("cuda"))
    monkeypatch.setattr(provider, "version", lambda _: "0.6.12")
    with pytest.raises(NotImplementedError, match="ABI"):
        provider.check_support(torch.device("cuda"))
    assert provider.QUALIFIED_VERSIONS == ("0.6.11.post3", "0.7.0.post1", "0.7.1")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    for qualified in provider.QUALIFIED_VERSIONS:
        monkeypatch.setattr(provider, "version", lambda _, release=qualified: release)
        provider.check_support(torch.device("cuda"))


@pytest.mark.parametrize(
    "stride", [(768, 64, 1), (256, 64, 1), (257, 64, 1), (128, 64, 1), (768, 1, 4)]
)
@pytest.mark.parametrize("row_stride", (256, 768))
@pytest.mark.parametrize("method", ("run", "prepare_and_run"))
def test_draft_row_strided_query(stride, row_stride, method):
    op = DraftAttention.__new__(DraftAttention)
    op._device, op._dtype = torch.device("cpu"), torch.bfloat16
    op._b, op._q, op._p, op._hq, op._hk, op._d = 1, 8, 1, 4, 2, 64
    op._query_row_stride = row_stride
    op._metadata = op._readonly = op._writable = ()
    op._bind()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return kwargs["out"]

    op._implementation = SimpleNamespace(run=run, prepare_and_run=run)
    q = torch.empty_strided((8, 4, 64), stride, dtype=op._dtype)
    kv = tuple(torch.empty((16, 1, 2, 64), dtype=op._dtype) for _ in range(2))
    out = torch.empty((8, 4, 64), dtype=op._dtype)
    if stride == (row_stride, 64, 1):
        assert getattr(op, method)(q=q, paged_kv_cache=kv, out=out) is out
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="strides.*query_row_stride"):
            getattr(op, method)(q=q, paged_kv_cache=kv, out=out)
        assert not calls


def test_overlap_uses_strided_byte_span():
    backing = torch.empty(256)
    query = backing.as_strided((4, 2, 8), (64, 8, 1))
    # The final row lies beyond numel(), but is still part of the read operand.
    with pytest.raises(ValueError, match="overlap"):
        disjoint((backing[192:208],), (query,))
    disjoint((backing[208:],), (query,))
    disjoint((backing[:0],), (query,))
