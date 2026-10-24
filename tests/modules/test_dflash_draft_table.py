# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host-only DF4 admission, workspace ownership and saturation reference checks."""

import pytest
import torch

from ctikernels.attention import draft
from ctikernels.attention._draft import cute, flashinfer
from ctikernels.attention.inference import CacheTableDraftAttention

from .._dflash_table_reference import table_csr


def geometry(**overrides):
    return {
        "batch_size": 2,
        "query_length": 4,
        "max_seq_len": 17,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "causal": True,
        "window": None,
        "device": "cuda",
    } | overrides


def aligned(size):
    backing = torch.empty(size + 255, dtype=torch.uint8)
    offset = -backing.data_ptr() % 256
    return backing[offset : offset + size]


@pytest.fixture
def host(monkeypatch):
    original = draft._geometry
    monkeypatch.setattr(draft, "_geometry", lambda **kw: original(**(kw | {"device": "cuda"})))
    monkeypatch.setattr(draft, "_provider", lambda provider, configuration: cute)
    monkeypatch.setattr(cute, "configuration", lambda **kw: (4, 16, 1, 1))
    monkeypatch.setattr(cute, "workspace_size", lambda **kw: (256, 256))
    launches = []

    class Provider:
        def __init__(self, floating, integer, **kwargs):
            self.sizes = floating.numel(), integer.numel()
            self.metadata_index = kwargs["metadata_index"]

        def run(self, **kwargs):
            launches.append(kwargs)
            return kwargs["out"]

    monkeypatch.setattr(cute, "TableProvider", Provider)
    args = geometry()
    args.pop("batch_size")
    args.pop("device")
    args.update(
        request_indices=torch.tensor([3, 1], dtype=torch.int32),
        frontier_table=torch.zeros(5, dtype=torch.int64),
        done_table=torch.zeros(5, dtype=torch.bool),
        cache_index_table=torch.zeros((7, 29), dtype=torch.int32)[:, :23],
        guard_start=20,
    )
    return args, launches


def construct(args, work=None):
    return CacheTableDraftAttention(*(work or [aligned(256), aligned(256)]), **args)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("max_seq_len", 0, "max_seq_len"),
        ("max_seq_len", True, "max_seq_len"),
        ("max_seq_len", 2**31 - 4, "M \\+ Q"),
        ("query_length", 1, "2 <= Q"),
        ("query_length", 33, "2 <= Q"),
        ("query_length", 4.0, "2 <= Q"),
        ("batch_size", 0, "positive"),
        ("batch_size", 2**30, "int32"),
        ("num_qo_heads", 7, "divisible"),
        ("num_kv_heads", 0, "positive"),
        ("dtype", torch.float32, "FP16 or BF16"),
        ("head_dim", 32, "head dimensions"),
        ("causal", 1, "causal must"),
        ("window", 4.0, "window must"),
        ("window", 0, "window must"),
        ("device", "cpu", "CUDA device"),
    ],
)
def test_geometry_errors(name, value, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        CacheTableDraftAttention.workspace_size(**geometry(**{name: value}))


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("metadata_index", "invalid", "metadata_index"),
        ("metadata_index", "ROW", "metadata_index"),
        ("metadata_index", None, "metadata_index"),
        ("metadata_index", 0, "metadata_index"),
        ("request_indices", torch.zeros((2, 1), dtype=torch.int32), "request_indices"),
        ("request_indices", torch.zeros(0, dtype=torch.int32), "request_indices"),
        ("request_indices", torch.zeros(2, dtype=torch.int64), "request_indices"),
        ("request_indices", torch.zeros(4, dtype=torch.int32)[::2], "request_indices"),
        ("frontier_table", torch.zeros((5, 1), dtype=torch.int64), "frontier_table"),
        ("frontier_table", torch.zeros(0, dtype=torch.int64), "frontier_table"),
        ("frontier_table", torch.zeros(5, dtype=torch.int32), "frontier_table"),
        ("frontier_table", torch.zeros(10, dtype=torch.int64)[::2], "frontier_table"),
        ("done_table", torch.zeros(4, dtype=torch.bool), "done_table"),
        ("done_table", torch.zeros(5, dtype=torch.uint8), "done_table"),
        ("done_table", torch.zeros(10, dtype=torch.bool)[::2], "done_table"),
        ("cache_index_table", torch.zeros(23, dtype=torch.int32), "cache_index_table"),
        ("cache_index_table", torch.zeros((0, 23), dtype=torch.int32), "cache_index_table"),
        ("cache_index_table", torch.zeros((5, 23), dtype=torch.int64), "cache_index_table"),
        ("cache_index_table", torch.zeros((5, 16), dtype=torch.int32), "cache_index_table"),
        ("cache_index_table", torch.zeros((5, 46), dtype=torch.int32)[:, ::2], "cache_index_table"),
        (
            "cache_index_table",
            torch.zeros(23, dtype=torch.int32).expand(5, 23),
            "cache_index_table",
        ),
        ("guard_start", -1, "guard_start"),
        ("guard_start", True, "guard_start"),
        ("guard_start", 2**31 - 8, "guard_start"),
        ("request_indices", torch.zeros(2, requires_grad=True), "gradients"),
    ],
)
def test_metadata_errors(host, name, value, match):
    args, launches = host
    with pytest.raises(ValueError, match=match):
        construct(args | {name: value})
    assert not launches


@pytest.mark.parametrize("name", ["frontier_table", "done_table"])
@pytest.mark.parametrize("shape", [(0,), (1,), (3,), (2, 1)])
def test_row_metadata_shape_errors(host, name, shape):
    args, launches = host
    args |= {
        "metadata_index": "row",
        "frontier_table": torch.zeros(2, dtype=torch.int64),
        "done_table": torch.zeros(2, dtype=torch.bool),
    }
    args[name] = torch.zeros(shape, dtype=args[name].dtype)
    with pytest.raises(ValueError, match="frontier|done"):
        construct(args)
    assert not launches


@pytest.mark.parametrize("length", [1, 3, 5])
def test_row_metadata_matching_lengths_must_equal_batch(host, length):
    args, launches = host
    args |= {
        "metadata_index": "row",
        "frontier_table": torch.zeros(length, dtype=torch.int64),
        "done_table": torch.zeros(length, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        construct(args)
    assert not launches


@pytest.mark.parametrize("name", ["request_indices", "cache_index_table"])
def test_unaligned_metadata(host, name):
    args, _ = host
    shape = args[name].shape
    count = args[name].numel()
    args[name] = torch.frombuffer(bytearray(count * 4 + 1), dtype=torch.int32, offset=1).view(shape)
    with pytest.raises(ValueError, match="4-byte aligned"):
        construct(args)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("kind", ["small", "alignment", "dtype", "shape", "stride", "grad"])
def test_workspace_errors(host, which, kind):
    args, launches = host
    work = [aligned(256), aligned(256)]
    work[which] = {
        "small": aligned(255),
        "alignment": aligned(257)[1:],
        "dtype": torch.empty(256, dtype=torch.int32),
        "shape": aligned(256).view(16, 16),
        "stride": aligned(512)[::2],
        "grad": torch.zeros(256, requires_grad=True),
    }[kind]
    with pytest.raises(ValueError):
        construct(args, work)
    assert not launches


def test_workspace_alias_and_strided_table_span(host):
    args, _ = host
    workspace = aligned(2048)
    with pytest.raises(ValueError, match="overlap"):
        construct(args, [workspace, workspace])
    args["cache_index_table"] = workspace.view(torch.int32).as_strided((5, 23), (64, 1))
    # This aliases the final row beyond numel()*4, which a dense-only check misses.
    with pytest.raises(ValueError, match="overlap"):
        construct(args, [workspace[1024:1280], aligned(256)])


def operands():
    return {
        "q": torch.empty((8, 32, 128), dtype=torch.bfloat16),
        "paged_kv_cache": tuple(
            torch.empty((28, 1, 8, 128), dtype=torch.bfloat16) for _ in range(2)
        ),
        "out": torch.empty((8, 32, 128), dtype=torch.bfloat16),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "q_shape",
        "out_dtype",
        "cache_tuple",
        "cache_shape",
        "cache_short",
        "v_shape",
        "grad",
        "alignment",
        "alias",
    ],
)
def test_run_errors_before_launch(host, kind):
    args, launches = host
    op, values = construct(args), operands()
    if kind == "q_shape":
        values["q"] = values["q"][:4]
    elif kind == "out_dtype":
        values["out"] = values["out"].float()
    elif kind == "cache_tuple":
        values["paged_kv_cache"] = list(values["paged_kv_cache"])
    elif kind == "cache_shape":
        values["paged_kv_cache"] = tuple(t.squeeze(1) for t in values["paged_kv_cache"])
    elif kind == "cache_short":
        values["paged_kv_cache"] = tuple(t[:27] for t in values["paged_kv_cache"])
    elif kind == "v_shape":
        k, v = values["paged_kv_cache"]
        values["paged_kv_cache"] = k, v[:1]
    elif kind == "grad":
        values["q"].requires_grad_()
    elif kind == "alignment":
        q = values["q"]
        values["q"] = torch.empty(q.numel() + 1, dtype=q.dtype)[1:].view_as(q)
    else:
        values["out"] = values["q"]
    with pytest.raises(ValueError):
        op.run(**values)
    assert not launches


@pytest.mark.parametrize("metadata_index", [None, "request", "row"])
def test_workspace_consistency_and_single_run(host, metadata_index):
    args, launches = host
    if metadata_index is not None:
        args["metadata_index"] = metadata_index
    if metadata_index == "row":
        args["frontier_table"] = torch.zeros(2, dtype=torch.int64)
        args["done_table"] = torch.zeros(2, dtype=torch.bool)
    sizes = CacheTableDraftAttention.workspace_size(**geometry())
    op = construct(args, [aligned(n + 256) for n in sizes])
    assert op._implementation.sizes == sizes == (256, 256)
    assert op._implementation.metadata_index == (metadata_index or "request")
    assert op.provider == "cute-table" and op.kernels_per_call == 1
    values = operands()
    assert op.run(**values) is values["out"]
    assert len(launches) == 1
    args["done_table"].resize_(4)
    with pytest.raises(ValueError, match="done_table"):
        op.run(**values)
    assert len(launches) == 1


def test_flashinfer_workspace_and_capacity(monkeypatch):
    monkeypatch.setattr(
        draft.FlashInferFixedQueryAttention, "workspace_size", lambda **kw: (512, 1024)
    )
    sizes = CacheTableDraftAttention.workspace_size(**geometry(), provider="flashinfer")
    assert sizes == (512, 1024 + flashinfer.fixed_query_bytes(2) + flashinfer.table_bytes(2, 21))
    with pytest.raises(NotImplementedError, match="int32"):
        CacheTableDraftAttention.workspace_size(
            **geometry(batch_size=1, max_seq_len=2**31 - 5), provider="flashinfer"
        )


def test_legacy_merge_rejected(host, monkeypatch):
    monkeypatch.setattr(cute, "configuration", lambda **kw: (4, 16, 1, 2))
    monkeypatch.setenv("CTI_DRAFT_MERGE", "legacy")
    with pytest.raises(ValueError, match="inline merge"):
        CacheTableDraftAttention.workspace_size(**geometry())


@pytest.mark.parametrize("frontier", [-(2**63), -(2**62), -1, 0, 1, 16, 17, 22, 2**62, 2**63 - 1])
@pytest.mark.parametrize("done", [False, True])
def test_reference_saturation_and_guard(frontier, done):
    table = torch.full((5, 23), 2**31 - 1, dtype=torch.int32)
    dummy = done or not 0 <= frontier < 17
    prefix = 0 if dummy else min(frontier + 4, 17)
    table[3, :prefix] = torch.arange(prefix, dtype=torch.int32) + 100
    ptr, indices, positions = table_csr(
        torch.tensor([3]),
        torch.tensor([0, 0, 0, frontier]),
        torch.tensor([False, False, False, done]),
        table,
        1000,
        17,
        4,
    )
    length = 4 if dummy else frontier + 4
    assert ptr.tolist() == [0, length]
    assert positions.tolist() == list(range(length - 4, length))
    expected = (
        list(range(1000, 1004))
        if dummy
        else (list(range(100, 100 + prefix)) + list(range(1000, 1000 + max(0, length - 17))))
    )
    assert indices.tolist() == expected


@pytest.mark.parametrize("name", ["frontier_table", "done_table", "cache_index_table"])
def test_metadata_device_mismatch(host, name):
    args, launches = host
    args[name] = args[name].to("meta")
    with pytest.raises(ValueError, match=name):
        construct(args)
    assert not launches


def test_table_layout_change_rejected(host):
    args, launches = host
    op = construct(args)
    args["cache_index_table"].as_strided_((7, 22), (29, 1))
    with pytest.raises(ValueError, match="layout must remain stable"):
        op.run(**operands())
    assert not launches


@pytest.mark.parametrize("provider", [None, "cute", "flashinfer"])
def test_invalid_provider_before_sizing(provider, monkeypatch):
    monkeypatch.setenv("CTI_DRAFT_PROVIDER", "invalid")
    if provider is None:
        with pytest.raises(ValueError, match="provider"):
            CacheTableDraftAttention.workspace_size(**geometry())
    with pytest.raises(ValueError, match="provider"):
        CacheTableDraftAttention.workspace_size(**geometry(), provider="invalid")


@pytest.mark.parametrize("metadata_index", ["request", "row"])
def test_flashinfer_private_csr_and_launch_order(monkeypatch, metadata_index):
    import sys
    from types import SimpleNamespace

    events = []
    state = {}

    def compile_table_csr(query, capacity, arch, row_metadata):
        assert (query, capacity, arch) == (4, 21, "sm_80")
        assert row_metadata == (metadata_index == "row")
        return lambda *args: events.append("expand")

    class Native:
        def __init__(self, floating, integer, **kwargs):
            state.update(kwargs)
            state["workspace"] = integer

        def prepare_and_run(self, **kwargs):
            events.extend(("prepare", "run"))
            return kwargs["out"]

    monkeypatch.setattr(flashinfer, "Provider", Native)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setitem(
        sys.modules,
        "ctikernels.attention._draft.flashinfer._table_csr",
        SimpleNamespace(compile_table_csr=compile_table_csr),
    )
    b, c = 2, 21
    workspace = aligned(256 + flashinfer.table_bytes(b, c))
    table = torch.zeros((7, 29), dtype=torch.int32)[:, :23]
    provider = flashinfer.TableProvider(
        aligned(256),
        workspace,
        request_indices=torch.tensor([3, 1], dtype=torch.int32),
        frontier_table=torch.zeros(b if metadata_index == "row" else 5, dtype=torch.int64),
        done_table=torch.zeros(b if metadata_index == "row" else 5, dtype=torch.bool),
        cache_index_table=table,
        guard_start=20,
        max_seq_len=17,
        query_length=4,
        metadata_index=metadata_index,
    )
    slots, ptr, idx, query, capacity = state["prepare_inputs"]
    assert slots.tolist() == [0, 1] and capacity == c
    assert state["workspace"].numel() == 256
    private = (
        slots,
        ptr,
        idx,
        query,
        state["kv_indptr"],
        state["kv_indices"],
        state["last_page_lengths"],
    )
    draft.disjoint(private, (state["workspace"], table))
    for value in private:
        assert value.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    monkeypatch.setattr(torch, "arange", lambda *args, **kw: pytest.fail("runtime allocation"))
    values = operands()
    assert provider.run(**values) is values["out"]
    assert events == ["expand", "prepare", "run"]
