# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""XQA dispatch, staging and launch contracts with a CPU provider fake."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _flashinfer, _split_tables, inference
from ctikernels.attention._flashinfer import xqa_decode_scratch_size
from ctikernels.attention.inference import FlashInferPagedAttention


@pytest.fixture
def provider(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    launch = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=constructor,
            BatchPrefillWithPagedKVCacheWrapper=constructor,
        ),
    )
    monkeypatch.setitem(sys.modules, "flashinfer.xqa", SimpleNamespace(xqa=launch))
    monkeypatch.setattr(_flashinfer, "xqa_flashinfer_version", lambda: True)
    monkeypatch.setattr(inference, "qualified_flashinfer_version", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12,
            minor=0,
            multi_processor_count=170,
            max_threads_per_multi_processor=1536,
        ),
    )
    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, **k: empty(
            *a, **{name: value for name, value in k.items() if name != "pin_memory"}
        ),
    )
    return constructor, native, launch


def operation(*, graph=False, batch=1, dispatch="device", tensor_cores=True):
    return FlashInferPagedAttention(
        torch.empty(2**20, dtype=torch.uint8),
        torch.empty(2**16, dtype=torch.uint8),
        mode="decode",
        backend="fa2",
        use_cuda_graph=graph,
        dispatch=dispatch,
        use_tensor_cores=tensor_cores,
        paged_kv_last_page_len_buffer=torch.empty(batch, dtype=torch.int32),
        paged_kv_indices_buffer=torch.empty(64, dtype=torch.int32),
        paged_kv_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32),
    )


def plan(op, lengths=(17,), **overrides):
    page = overrides.get("page_size", 16)
    pages = [(length + page - 1) // page for length in lengths]
    options = {
        "cu_seqlens_q": torch.arange(len(lengths) + 1, dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, *pages], dtype=torch.int32).cumsum(0, dtype=torch.int32),
        "indices": torch.arange(sum(pages), dtype=torch.int32).flip(0),
        "last_page_len": torch.tensor(
            [(length - 1) % page + 1 for length in lengths], dtype=torch.int32
        ),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": page,
        "pos_encoding_mode": "NONE",
        "dtype": torch.bfloat16,
        "kv_dtype": torch.float8_e4m3fn,
    }
    op.plan(**(options | overrides))
    return options


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("tensor_cores", [False, True])
def test_fp8_routes_without_native_wrapper(provider, graph, tensor_cores):
    constructor, native, launch = provider
    op = operation(graph=graph, tensor_cores=tensor_cores)
    constructor.assert_not_called()
    plan(op)
    assert op._decode_route[0] == "xqa"
    assert op._wrapper is None
    constructor.assert_not_called()
    native.plan.assert_not_called()
    q = torch.zeros(1, 32, 128, dtype=torch.bfloat16)
    cache = (torch.empty(2, 16, 8, 128),) * 2
    out = torch.empty_like(q)
    assert op.run(q=q, paged_kv_cache=cache, out=out) is out
    assert launch.call_args.kwargs["workspace_buffer"].numel() == op._xqa.scratch_bytes
    assert op.run(q=q, paged_kv_cache=cache).shape == q.shape


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_dtype": torch.bfloat16},
        {"logits_soft_cap": 1.0},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"rope_scale": 2.0},
        {"page_size": 1},
        {"head_dim": 512},
        {"num_qo_heads": 24},
    ],
)
def test_off_domain_keeps_native(provider, overrides):
    op = operation()
    plan(op, **overrides)
    assert op._decode_route[0] == "native"
    provider[1].plan.assert_called_once()


def test_provider_dispatch_and_absent_batch(provider):
    for op, lengths in ((operation(dispatch="provider"), (17,)), (operation(batch=5), (17,) * 5)):
        plan(op, lengths)
        assert op._decode_route[0] == "native"


@pytest.mark.parametrize("start_fp8", [False, True])
def test_graph_cannot_switch_route(provider, start_fp8):
    op = operation(graph=True)
    dtypes = (torch.float8_e4m3fn, torch.bfloat16)
    plan(op, kv_dtype=dtypes[not start_fp8])
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op, kv_dtype=dtypes[start_fp8])


@pytest.mark.parametrize("row", [{}, {1: 16}])
def test_graph_cannot_change_split_or_captured_options(provider, monkeypatch, row):
    op = operation(graph=True)
    plan(op)
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op, window_left=32)
    monkeypatch.setitem(_split_tables.XQA_DECODE_SUBSEQ, ("rtx5090", 32, 8, 128), row)
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op)


def test_eager_can_change_routes_and_batch(provider):
    op = operation()
    for lengths, dtype, route in (
        ((17,), torch.float8_e4m3fn, "xqa"),
        ((17,), torch.bfloat16, "native"),
        ((17, 33), torch.float8_e4m3fn, "xqa"),
    ):
        plan(op, lengths, kv_dtype=dtype)
        assert op._decode_route[0] == route
    assert op._xqa.batch == 2


@pytest.mark.parametrize("graph", [False, True])
def test_metadata_snapshot_and_capacity(provider, graph):
    op = operation(graph=graph, batch=2)
    metadata = plan(op, (1, 17))
    xqa = op._xqa
    assert xqa.page_table.shape == (2, 64 if graph else 2)
    assert xqa.page_table[0, :2].tolist() == [2, 0]
    assert xqa.page_table[1, :2].tolist() == [1, 0]
    assert xqa.seq_lens.tolist() == [[1], [17]]
    metadata["indices"].fill_(99)
    assert xqa.page_table[0, 0] == 2
    original = xqa.page_table
    staging = xqa._lengths_host
    plan(op, (65, 33))
    assert (xqa.page_table is original) == graph
    assert xqa._lengths_host is staging
    assert xqa.seq_lens.tolist() == [[65], [33]]
    assert xqa.semaphores.numel() == 16
    assert xqa.semaphores.dtype == torch.uint32
    assert xqa.semaphores.to(torch.int64).count_nonzero() == 0


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("sm_scale,k_scale,v_scale", [(0.125, 0.5, 0.25), (None, None, None)])
@pytest.mark.parametrize("minor,sms", [(0, 170), (0, 188), (1, 48)])
def test_scale_and_sm_count(provider, monkeypatch, batch, sm_scale, k_scale, v_scale, minor, sms):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=minor, multi_processor_count=sms),
    )
    op = operation(batch=batch)
    plan(op, (17,) * batch, sm_scale=sm_scale, window_left=31)
    q = torch.zeros(batch, 32, 128, dtype=torch.bfloat16)
    op.run(q=q, paged_kv_cache=(q, q), k_scale=k_scale, v_scale=v_scale)
    args = provider[2].call_args.kwargs
    assert args["q_scale"] == pytest.approx(
        1.0 if sm_scale is None else 0.125 * 0.5 / 0.25 * 128**0.5
    )
    assert args["kv_scale"] == (1.0 if v_scale is None else v_scale)
    subseq = _split_tables.xqa_decode_subseq(q.device, 32, 8, 128, batch)
    assert args["sm_count"] == subseq * batch * 8
    assert op._xqa.subseq == subseq
    assert op._xqa.scratch_bytes == xqa_decode_scratch_size(batch, 8, subseq, 128, 4)
    assert args["sliding_win_size"] == 32
    assert args["q"].shape == args["output"].shape == (batch, 1, 32, 128)
    assert args["seq_lens"].shape == (batch, 1)
    assert args["enable_pdl"] is None


def test_zero_v_scale_keeps_finite_query_scale(provider):
    op = operation(batch=2)
    plan(op, (17, 17), sm_scale=0.125)
    q = torch.zeros(2, 32, 128, dtype=torch.bfloat16)
    op.run(q=q, paged_kv_cache=(q, q), k_scale=0.5, v_scale=0.0)
    args = provider[2].call_args.kwargs
    assert args["kv_scale"] == 0.0
    assert args["q_scale"] == pytest.approx(0.125 * 0.5 * 128**0.5)


@pytest.mark.parametrize("graph", [False, True])
def test_strided_query_stages_through_dense_rows(provider, graph):
    op = operation(graph=graph, batch=2)
    plan(op, (17, 17))
    q = torch.randn(2, 48, 128).to(torch.bfloat16)[:, :32]
    out = torch.zeros(2, 32, 128, dtype=torch.bfloat16)

    def launch(**args):
        assert args["q"].is_contiguous()
        torch.testing.assert_close(args["q"].squeeze(1), q, atol=0, rtol=0)
        args["output"].copy_(args["q"] + 1)

    provider[2].side_effect = launch
    stage = op._xqa._q_stage.data_ptr()
    assert op.run(q=q, paged_kv_cache=(q, q), out=out, v_scale=0.25) is out
    torch.testing.assert_close(out, q + 1, atol=0, rtol=0)
    assert provider[2].call_args.kwargs["output"].data_ptr() == out.data_ptr()
    assert op._xqa._q_stage.data_ptr() == stage
    # A dense query is passed through without staging.
    dense = q.contiguous()
    op.run(q=dense, paged_kv_cache=(q, q), out=out)
    assert provider[2].call_args.kwargs["q"].data_ptr() == dense.data_ptr()


@pytest.mark.parametrize("fp8", [False, True])
def test_strided_output_is_rejected_on_every_route(provider, fp8):
    op = operation(batch=2)
    plan(op, (17, 17), kv_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16)
    q = torch.zeros(2, 32, 128, dtype=torch.bfloat16)
    out = torch.zeros(2, 48, 128, dtype=torch.bfloat16)[:, :32]
    with pytest.raises(ValueError, match="contiguous out"):
        op.run(q=q, paged_kv_cache=(q, q), out=out)
    provider[2].assert_not_called()
    provider[1].run.assert_not_called()


@pytest.mark.parametrize("graph", [False, True])
def test_workspace_bound_covers_table_and_keeps_integer_size(provider, monkeypatch, graph):
    key = ("rtx5090", 32, 8, 128)
    monkeypatch.setitem(_split_tables.XQA_DECODE_SUBSEQ, key, {1: 16, 4: 2})
    # Make XQA dominate so this checks the exact bound, including eager B1
    # needing more scratch than B4 due to the per-row split counts.
    monkeypatch.setattr(inference, "paged_attention_workspace_size", lambda **_: (256, 256))
    options = {
        "mode": "decode",
        "backend": "fa2",
        "use_tensor_cores": True,
        "use_cuda_graph": graph,
        "max_batch_size": 4,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "device": torch.device("cuda"),
    }
    native = FlashInferPagedAttention.workspace_size(**options, dispatch="provider")
    device = FlashInferPagedAttention.workspace_size(**options)
    expected = xqa_decode_scratch_size(4, 8, 2, 128, 4)
    if not graph:
        expected = max(expected, xqa_decode_scratch_size(1, 8, 16, 128, 4))
    assert device[0] == expected > native[0]
    monkeypatch.setitem(_split_tables.XQA_DECODE_SUBSEQ, key, {})
    assert device[1] == FlashInferPagedAttention.workspace_size(**options)[1]


@pytest.mark.parametrize(
    "axis,values",
    [
        (0, [1, 2, 4, 64]),
        (1, [1, 4, 8, 16]),
        (2, [1, 2, 8, 4096]),
        (3, list(range(16, 257, 16))),
        (4, [1, 2, 4, 8, 16, 32]),
    ],
)
def test_scratch_bound_monotonic(axis, values):
    sizes = []
    for value in values:
        shape = [1, 8, 8, 128, 4]
        shape[axis] = value
        sizes.append(xqa_decode_scratch_size(*shape))
    assert sizes == sorted(sizes)
    assert all(size % 256 == 0 for size in sizes)


@pytest.mark.parametrize("capability", [(9, 0), (12, 0), (12, 1)])
@pytest.mark.parametrize("qualified", [False, True])
def test_version_and_device_gate(provider, monkeypatch, capability, qualified):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    monkeypatch.setattr(_flashinfer, "xqa_flashinfer_version", lambda: qualified)
    op = operation()
    plan(op)
    assert (op._decode_route[0] == "xqa") == (qualified and capability in ((12, 0), (12, 1)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dtype": torch.float32},
        {"kv_dtype": torch.float8_e5m2},
        {"page_size": 8},
        {"head_dim": 0},
        {"head_dim": 24},
        {"num_qo_heads": 30},
        {"num_kv_heads": 0},
        {"custom_mask": torch.ones(1)},
        {"packed_custom_mask": torch.ones(1)},
        {"window_left": -2},
        {"rope_theta": 10000.0},
    ],
)
def test_qualification_rejects_unsupported_options(provider, overrides):
    options = {
        "dtype": torch.bfloat16,
        "kv_dtype": torch.float8_e4m3fn,
        "page_size": 16,
        "head_dim": 128,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": None,
        "window_left": -1,
        "rope_scale": None,
        "rope_theta": None,
    }
    assert not _flashinfer.xqa_decode_qualified(torch.device("cuda"), **(options | overrides))


def test_graph_metadata_batch_and_capacity_are_fixed(provider):
    op = operation(graph=True)
    plan(op)
    with pytest.raises(ValueError, match="fixed batch"):
        plan(op, (17, 17))
    with pytest.raises(ValueError, match="capacity"):
        plan(op, (1041,))


def test_xqa_rejects_device_lengths_without_readback(provider):
    op = operation()
    with pytest.raises(ValueError, match="CPU CSR"):
        plan(op, seq_lens=torch.empty(1, device="meta"))


def test_xqa_rejects_insufficient_scratch(provider):
    op = operation()
    op._workspaces = (torch.empty(0, dtype=torch.uint8), op._workspaces[1])
    with pytest.raises(ValueError, match="borrowed float workspace"):
        plan(op)


@pytest.mark.parametrize(
    "installed,expected",
    [("0.6.11.post3", False), ("0.7.0.post1", True), ("0.7.1", True), ("0.7.2", False)],
)
def test_xqa_version_gate_excludes_0611(monkeypatch, installed, expected):
    _flashinfer.xqa_flashinfer_version.cache_clear()
    monkeypatch.setattr(_flashinfer, "qualified_flashinfer_version", lambda: True)
    monkeypatch.setattr(_flashinfer, "version", lambda _: installed)
    try:
        assert _flashinfer.xqa_flashinfer_version() is expected
    finally:
        _flashinfer.xqa_flashinfer_version.cache_clear()
