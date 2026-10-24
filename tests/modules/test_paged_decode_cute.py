# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CuTe routing and CSR staging contracts without CUDA or a provider installation."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _decode_fp8 as decode
from ctikernels.attention import _split_tables as tables
from ctikernels.attention import inference
from ctikernels.attention.inference import FlashInferPagedAttention


@pytest.fixture
def hardware(monkeypatch):
    for key in ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE"):
        monkeypatch.delenv(f"CTI_FP8DEC_{key}", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12, minor=0, multi_processor_count=170, max_threads_per_multi_processor=1536
        ),
    )
    fake = ModuleType("cutlass")
    fake.cute = ModuleType("cutlass.cute")
    fake.runtime = ModuleType("cutlass.runtime")
    fake.runtime.make_fake_stream = object()
    for name, module in (
        ("cutlass", fake),
        ("cutlass.cute", fake.cute),
        ("cutlass.runtime", fake.runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def provider(hardware, monkeypatch):
    native = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=native,
            BatchPrefillWithPagedKVCacheWrapper=native,
        ),
    )
    monkeypatch.setattr(inference, "qualified_flashinfer_version", lambda: False)
    monkeypatch.setattr(inference, "xqa_decode_qualified", lambda *a, **k: True)
    support = decode.check_support
    monkeypatch.setattr(decode, "check_support", lambda device, **k: support("cuda", **k))
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    cls = decode.FP8PagedDecode

    def construct(workspace, **kw):
        op = cls.__new__(cls)
        op.batch, op.hq, op.hk, op.dim = (
            kw[n] for n in ("batch_size", "num_qo_heads", "num_kv_heads", "head_dim")
        )
        op.page, op.dtype, op.device, op.sm_scale = (
            kw[n] for n in ("page_size", "q_dtype", "device", "sm_scale")
        )
        op._work, op._counters, op._kernel = workspace, object(), Mock()
        op._strided_kernel = Mock()
        return op

    constructor = Mock(side_effect=construct)
    monkeypatch.setattr(decode, "FP8PagedDecode", constructor)
    monkeypatch.setitem(tables.FP8_DECODE_CUTE, ("rtx5090", 16, 4, 256), {1: {}, 2: {}, 4: {}})
    return native, constructor


def operation(*, graph=False, batch=2, dispatch="device", tensor_cores=True):
    return FlashInferPagedAttention(
        torch.empty(2**20, dtype=torch.uint8),
        torch.empty(2**16, dtype=torch.uint8),
        mode="decode",
        backend="fa2",
        use_cuda_graph=graph,
        use_tensor_cores=tensor_cores,
        dispatch=dispatch,
        paged_kv_indptr_buffer=torch.empty(batch + 1, dtype=torch.int32),
        paged_kv_indices_buffer=torch.empty(128, dtype=torch.int32),
        paged_kv_last_page_len_buffer=torch.empty(batch, dtype=torch.int32),
    )


def plan(op, lengths=(17, 33), **overrides):
    page = overrides.get("page_size", 16)
    counts = [(n + page - 1) // page for n in lengths]
    metadata = {
        "cu_seqlens_q": torch.arange(len(lengths) + 1, dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, *counts], dtype=torch.int32).cumsum(0, dtype=torch.int32),
        "indices": torch.arange(sum(counts), dtype=torch.int32).flip(0),
        "last_page_len": torch.tensor([(n - 1) % page + 1 for n in lengths], dtype=torch.int32),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
    }
    op.plan(
        **(
            metadata
            | {
                "num_qo_heads": 16,
                "num_kv_heads": 4,
                "head_dim": 256,
                "page_size": page,
                "pos_encoding_mode": "NONE",
                "dtype": torch.bfloat16,
                "kv_dtype": torch.float8_e4m3fn,
            }
            | overrides
        )
    )
    return metadata


@pytest.mark.parametrize(
    "key,expected", [("rtx5090", {}), ("rtxpro6000", None), ("gb10", None), (None, None)]
)
def test_table_lookup(monkeypatch, key, expected):
    monkeypatch.setattr(tables, "FP8_DECODE_CUTE", {("rtx5090", 16, 4, 256): {1: {}, 2: {}}})
    monkeypatch.setattr(tables, "device_key", lambda _: key)
    assert tables.fp8_decode_cute(None, 16, 4, 256, 1) == expected
    assert tables.fp8_decode_cute(None, 16, 4, 256, 2) == expected
    assert tables.fp8_decode_cute(None, 16, 4, 256, 3) is None
    assert tables.fp8_decode_cute(None, 32, 8, 128, 1) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"window_left": 0},
        {"window_left": -2},
        {"logits_soft_cap": 1.0},
        {"dtype": torch.float32},
        {"kv_dtype": torch.bfloat16},
        {"kv_dtype": torch.float8_e5m2},
        {"page_size": 0},
        {"page_size": -1},
        {"page_size": 1.5},
        {"num_qo_heads": 68},
        {"num_qo_heads": 15},
        {"num_kv_heads": 0},
        {"head_dim": 192},
        {"rope_scale": 1.0},
        {"rope_theta": 10000.0},
        {"pos_encoding_mode": "ROPE_LLAMA"},
    ],
)
def test_route_predicate_rejections(hardware, changes):
    options = {
        "dtype": torch.bfloat16,
        "kv_dtype": torch.float8_e4m3fn,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "page_size": 16,
        "window_left": -1,
        "logits_soft_cap": None,
        "rope_scale": None,
        "rope_theta": None,
        "pos_encoding_mode": "NONE",
    }
    assert inference.fp8_decode_cute_qualified("cuda", **options)
    assert not inference.fp8_decode_cute_qualified("cuda", **(options | changes))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("page", [1, 7, 16])
def test_cute_precedes_xqa_and_stages_csr(provider, graph, page):
    native, constructor = provider
    op = operation(graph=graph)
    metadata = plan(op, page_size=page)
    assert op.decode_route == "cute"
    native.assert_not_called()
    constructor.assert_called_once()
    ptr, idx, last = op._cute_csr
    assert torch.equal(ptr, metadata["cu_seqlens_k"])
    assert torch.equal(last, metadata["last_page_len"])
    assert torch.equal(idx[: metadata["indices"].numel()], metadata["indices"])
    if graph:
        assert ptr.data_ptr() == op._decode_options["paged_kv_indptr_buffer"].data_ptr()
        assert last.data_ptr() == op._decode_options["paged_kv_last_page_len_buffer"].data_ptr()
        assert idx is op._decode_options["paged_kv_indices_buffer"]
        metadata["indices"].fill_(99)
        assert idx[0] != 99
    else:
        assert idx is metadata["indices"]
    pointers = (ptr.data_ptr(), last.data_ptr())
    plan(op, (49, 65), page_size=page)
    assert pointers == (op._cute_csr[0].data_ptr(), op._cute_csr[2].data_ptr())
    assert op._cute_csr[0].tolist() == [
        0,
        (49 + page - 1) // page,
        (49 + page - 1) // page + (65 + page - 1) // page,
    ]
    constructor.assert_called_once()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scales", [{}, {"k_scale": 0.5, "v_scale": 0.0}])
def test_run_preserves_strided_q_output_and_scales(provider, dtype, scales):
    op = operation()
    plan(op, dtype=dtype, sm_scale=0.125)
    q = torch.empty(2, 24, 256, dtype=dtype)[:, :16]
    cache = (torch.empty(5, 16, 4, 256, dtype=torch.float8_e4m3fn),) * 2
    out = torch.empty(q.shape, dtype=q.dtype)
    assert op.run(q=q, paged_kv_cache=cache, out=out, **scales) is out
    args = op._cute._kernel.call_args.args
    assert args[0] is q and args[6] is out
    assert args[-3:] == (0.125, scales.get("k_scale", 1.0), scales.get("v_scale", 1.0))
    allocated = op.run(q=q, paged_kv_cache=cache)
    assert allocated.shape == q.shape and allocated.is_contiguous()
    with pytest.raises(ValueError, match="contiguous out"):
        op.run(q=q, paged_kv_cache=cache, out=q)
    for invalid in (torch.tensor(1.0), float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite host float"):
            op.run(q=q, paged_kv_cache=cache, out=out, k_scale=invalid)


def test_default_scale_config_forwarding_and_graph_freeze(provider, monkeypatch):
    row = {
        2: {
            "splits": 3,
            "min_chunk": 32,
            "warps": 2,
            "keys_per_warp": 32,
            "stages": 2,
            "merge": "last",
        }
    }
    monkeypatch.setitem(tables.FP8_DECODE_CUTE, ("rtx5090", 16, 4, 256), row)
    op = operation(graph=True)
    plan(op)
    assert provider[1].call_args.kwargs.items() >= row[2].items()
    assert op._cute.sm_scale == 256**-0.5
    for change in (
        {"window_left": 31},
        {"sm_scale": 0.5},
        {"page_size": 1},
        {"dtype": torch.float16},
        {"kv_dtype": torch.bfloat16},
    ):
        with pytest.raises(RuntimeError, match="first plan"):
            plan(op, **change)
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "4")
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op)
    monkeypatch.delenv("CTI_FP8DEC_SPLITS")
    row[2]["warps"] = 1
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op)


def test_graph_cannot_switch_to_cute(provider):
    op = operation(graph=True)
    plan(op, kv_dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="first plan"):
        plan(op)


def test_unsupported_kernel_skips_table_lookup(provider, monkeypatch):
    monkeypatch.setattr(decode, "check_support", Mock(side_effect=NotImplementedError))
    lookup = Mock(side_effect=AssertionError("unsupported device queried"))
    monkeypatch.setattr(inference, "fp8_decode_cute", lookup)
    monkeypatch.setattr(inference, "_FlashInferXQADecode", Mock())
    op = operation()
    plan(op)
    assert op.decode_route == "xqa"
    lookup.assert_not_called()
    provider[1].assert_not_called()


def test_cute_plan_rejects_capture_before_construction(provider, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    op = operation()
    with pytest.raises(RuntimeError, match="outside capture"):
        plan(op, indices=SimpleNamespace(device=torch.device("cuda")))
    provider[1].assert_not_called()


@pytest.mark.parametrize("graph", [False, True])
def test_metadata_copies_are_nonblocking(provider, monkeypatch, graph):
    copy = torch.Tensor.copy_
    copies = []

    def record(destination, source, **kwargs):
        copies.append(kwargs)
        return copy(destination, source, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record)
    op = operation(graph=graph)
    plan(op)
    assert copies == [{"non_blocking": True}] * (3 if graph else 2)


def test_eager_grows_and_reuses_metadata(provider):
    op = operation()
    plan(op)
    first = op._cute_metadata
    plan(op, (17, 33, 49, 65))
    grown = op._cute_metadata
    assert grown is not first
    plan(op, (17,))
    assert op._cute_metadata is grown
    assert [t.numel() for t in (op._cute_csr[0], op._cute_csr[2])] == [2, 1]
    plan(op, (17,), kv_dtype=torch.bfloat16)
    assert op.decode_route == "fa2"
    plan(op, (17,), sm_scale=0.125)
    assert op.decode_route == "cute" and op._cute.sm_scale == 0.125


@pytest.mark.parametrize("tensor_cores", [False, True])
def test_provider_dispatch_and_read_only_route(provider, tensor_cores):
    op = operation(dispatch="provider", tensor_cores=tensor_cores)
    plan(op)
    assert op.decode_route == ("fa2" if tensor_cores else "cuda_cores")
    provider[1].assert_not_called()
    with pytest.raises(AttributeError):
        op.decode_route = "cute"


def test_unlisted_batch_keeps_xqa(provider, monkeypatch):
    monkeypatch.setattr(inference, "_FlashInferXQADecode", Mock())
    op = operation(batch=3)
    plan(op, (17, 33, 49))
    assert op.decode_route == "xqa"
    provider[1].assert_not_called()


def test_workspace_and_metadata_errors(provider):
    op = operation()
    op._workspaces = (torch.empty(0, dtype=torch.uint8), op._workspaces[1])
    with pytest.raises(ValueError, match="borrowed float workspace"):
        plan(op)
    op = operation(graph=True)
    with pytest.raises(ValueError, match="fixed batch"):
        plan(op, (17,))
    with pytest.raises(ValueError, match="capacity"):
        plan(op, (4096, 4096))
    with pytest.raises(ValueError, match="CPU CSR"):
        plan(op, last_page_len=torch.empty(2, dtype=torch.int32, device="meta"))
    op._decode_options["paged_kv_indptr_buffer"] = None
    with pytest.raises(ValueError, match="CSR buffers"):
        plan(op)


@pytest.mark.parametrize("graph", [False, True])
def test_workspace_covers_only_admitted_batches(hardware, monkeypatch, graph):
    configs = {
        1: {"splits": 32768, "merge": "last"},
        4: {"splits": 4096, "merge": "last"},
        5: {"splits": 65536, "merge": "last"},
    }
    monkeypatch.setitem(tables.FP8_DECODE_CUTE, ("rtx5090", 16, 4, 256), configs)
    options = {
        "mode": "decode",
        "backend": "fa2",
        "use_tensor_cores": True,
        "use_cuda_graph": graph,
        "max_batch_size": 4,
        "num_qo_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 256,
        "device": "cuda",
    }
    native = FlashInferPagedAttention.workspace_size(**options, dispatch="provider")
    batches = (4,) if graph else (1, 4)
    needed = max(
        decode.workspace_size(
            "cuda", batch_size=b, num_qo_heads=16, num_kv_heads=4, head_dim=256, **configs[b]
        )
        for b in batches
    )
    assert needed > native[0]
    assert FlashInferPagedAttention.workspace_size(**options) == (needed, native[1])
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "65536")
    needed = max(
        decode.workspace_size(
            "cuda", batch_size=b, num_qo_heads=16, num_kv_heads=4, head_dim=256, **configs[b]
        )
        for b in batches
    )
    assert FlashInferPagedAttention.workspace_size(**options) == (needed, native[1])


@pytest.mark.parametrize("layout", ["misaligned", "odd", "heads"])
def test_query_staging_uses_plan_owned_buffer(provider, monkeypatch, layout):
    op = operation(graph=True)
    plan(op)
    assert op.decode_route == "cute"
    if layout == "misaligned":
        q = torch.randn(2 * 16 * 256 + 1, dtype=torch.bfloat16)[1:].view(2, 16, 256)
    else:
        q = torch.randn(2, 32, 257 if layout == "odd" else 256, dtype=torch.bfloat16)[:, ::2, :256]
    cache = torch.empty(8, 2, 16, 4, 256, dtype=torch.float8_e4m3fn)
    out = torch.empty(q.shape, dtype=q.dtype)
    pointer = op._cute_q_stage.data_ptr()
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("run allocation")))
    for _ in range(2):
        q.add_(1)
        op.run(q=q, paged_kv_cache=(cache[:, 0], cache[:, 1]), out=out)
        launched = op._cute._strided_kernel.call_args.args[0]
        torch.testing.assert_close(launched, q)
        assert launched.data_ptr() == pointer
        op._cute._kernel.assert_not_called()
