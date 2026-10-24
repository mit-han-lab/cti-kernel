# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU route tests and an independent graph scheduler/allocation oracle."""

import sys
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _split_tables as tables
from ctikernels.attention import inference
from ctikernels.attention.inference import FlashInferPagedAttention


@pytest.fixture(autouse=True)
def device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(major=12, minor=0, multi_processor_count=170),
    )


def test_prefill_graph_shipped_table():
    assert tables.PREFILL_GRAPH_SPLIT == {
        ("rtx5090", 32, 8, 128): {(b, (129, 256), (0, None)): 4 for b in (2, 3, 4)},
        ("rtx5090", 16, 4, 256): {(b, (129, 256), (0, None)): 4 for b in (2, 3, 4)},
        ("rtxpro6000", 32, 8, 128): {
            (2, (129, 256), (0, None)): 6,
            (3, (129, 256), (0, None)): 4,
            (4, (129, 256), (257, None)): 4,
        },
        ("rtxpro6000", 16, 4, 256): {
            (2, (129, 256), (0, None)): 6,
            (3, (129, 256), (0, None)): 4,
            (4, (129, 256), (0, None)): 4,
        },
        ("gb10", 32, 8, 128): {
            (1, (129, 256), (0, None)): {"chunks": 1, "tile": 64},
            (2, (33, 128), (0, None)): {"chunks": 1, "tile": 64},
            (3, (9, 16), (0, 256)): {"chunks": 1, "tile": 16},
            (4, (9, 16), (0, 256)): {"chunks": 1, "tile": 16},
        },
        ("gb10", 16, 4, 256): {
            (2, (9, 16), (0, None)): {"chunks": 1, "tile": 16},
            (3, (9, 16), (0, None)): {"chunks": 1, "tile": 16},
            (4, (9, 16), (0, None)): {"chunks": 2, "tile": 16},
            (2, (129, 256), (257, None)): 4,
        },
    }
    assert tables.PREFILL_GRAPH_CUTE == {
        ("gb10", 32, 8, 128): {
            (3, (33, 128)): {"block_m": 64},
            (4, (33, 128)): {"block_m": 64, "block_n": 32},
        },
        ("gb10", 16, 4, 256): {(3, (33, 128)): {}},
    }


@pytest.mark.parametrize("batch,short,long", [(2, 4, 6), (3, 6, 6), (4, 6, 8)])
def test_prefill_graph_table(monkeypatch, batch, short, long):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {
            ("rtx5090", 32, 8, 128): {
                (batch, (129, 256), (0, 256)): short,
                (batch, (129, 256), (257, None)): long,
            },
        },
    )
    for query in (129, 200, 256):
        for pages, expected in ((0, short), (1, short), (256, short), (257, long), (8192, long)):
            assert tables.prefill_graph_chunks("cuda", 32, 8, 128, batch, query, pages) == expected
    for query, pages in ((128, 256), (257, 256)):
        assert tables.prefill_graph_chunks("cuda", 32, 8, 128, batch, query, pages) == 1
    assert tables.prefill_graph_max_chunks("cuda", 32, 8, 128, batch) == max(short, long)


@pytest.mark.parametrize(
    "heads,batch", [((16, 4, 128), 2), ((32, 8, 256), 2), ((32, 8, 128), 1), ((16, 4, 256), 1)]
)
def test_prefill_graph_unlisted(heads, batch, monkeypatch):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    assert tables.prefill_graph_chunks("cuda", *heads, batch, 256, 512) == 1
    monkeypatch.setattr(tables, "device_key", lambda _: "gb10")
    assert tables.prefill_graph_chunks("cuda", 32, 8, 128, 3, 256, 512) == 1


def test_prefill_graph_arithmetic(monkeypatch):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    chunks = tables.prefill_graph_max_chunks("cuda", 32, 8, 128, 3)
    layout = tables.prefill_graph_split_layout(768, 3, 4, 128, chunks, 42)
    assert layout == (131, 128, 157)  # tiles=26; P=5*26+1; padded=26+131.
    assert tables.prefill_graph_chunk([256] * 3, [512] * 3, 4, 6, layout) == 86
    assert tables.prefill_graph_chunk([256] * 3, [256] * 3, 4, 6, layout) == 43
    assert tables.prefill_graph_chunk([256] * 3, [512] * 3, 4, 1, layout) is None
    assert tables.prefill_graph_chunk([256] * 3, [1] * 3, 4, 6, layout) is None
    assert tables.prefill_graph_chunk([256] * 3, [0] * 3, 4, 6, layout) is None
    assert tables.prefill_graph_chunk([256] * 3, [512] * 3, 4, 8, layout) is None
    # Empty KV still costs one partition per query tile in FlashInfer.
    tiny = tables.PrefillGraphSplit(0, 128, 2)
    assert tables.prefill_graph_chunk([32, 32], [0, 4], 4, 2, tiny) is None


def test_prefill_graph_tile_guard_and_unsigned_wrap(monkeypatch):
    # These layout guards depend only on their explicit arithmetic inputs.
    monkeypatch.setattr(tables, "PREFILL_GRAPH_SPLIT", {})
    # Natural packed longest=68 ->128, padded longest=56 ->64.
    assert tables.prefill_graph_split_layout(18, 2, 4, 128, 2, 42) is None
    # A uint32 subtraction wraps below zero, but exactly zero selects tile16.
    assert tables.prefill_graph_split_layout(8, 2, 4, 128, 4, 42) is None
    assert tables.prefill_graph_split_layout(8, 2, 4, 128, 5, 42) is None
    layout = tables.prefill_graph_split_layout(18, 2, 4, 128, 12, 42)
    assert layout == (23, 128, 42)


def scheduler_counts(total, batch, group, dim, chunks, target=None):
    """Independent FA2 graph calculation: uint32 longest, then allocator walk."""

    def tile(requests):
        longest = (total - requests + 1) % (2**32)
        if longest * group <= 16:
            return 16
        if dim < 256 and longest * group > 64:
            return 128
        return 64

    natural = tile(batch) if target is None else target
    tiles = -(-total * group // natural) + batch - 1
    padding = (chunks - 1) * tiles + 1 if target is None else total + 1 - target // group - batch
    if padding < 0 or tile(batch + padding) != natural:
        return None
    count = max(340 // 8, -(-total * group // natural) + batch + padding - 1)
    integer = 0
    for size, alignment in (
        (count * 4, 16),
        (count * 4, 16),
        (count * 4, 16),
        ((batch + padding + 1) * 4, 16),
        (4, 1),
        (4, 16),
        ((total + 1) * 4, 16),
        (count, 16),
    ):
        integer = -(-integer // alignment) * alignment + size
    floating = 32 * count * natural * dim * 4 + 32 * count * natural * 4
    return floating, integer


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("max_batch,max_total", [(1, 24), (2, 40), (4, 96), (4, 1024)])
def test_prefill_graph_sizes_cover_every_activated_total(monkeypatch, dim, max_batch, max_total):
    if max_total < 1024 or dim == 256:
        table = {(b, (1, 24), (1, 256)): 2 for b in range(1, 5)} | {
            (b, (8, 24), (257, None)): 4 for b in range(1, 5)
        }
    else:
        table = {
            (b, (129, 256), pages): chunks
            for b, short, long in ((2, 4, 6), (3, 6, 6), (4, 6, 8))
            for pages, chunks in (((0, 256), short), ((257, None), long))
        }
    monkeypatch.setattr(tables, "PREFILL_GRAPH_SPLIT", {("rtx5090", 32, 8, dim): table})
    options = {
        "mode": "prefill",
        "backend": "fa2",
        "use_cuda_graph": True,
        "max_batch_size": max_batch,
        "max_query_tokens": max_total,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": dim,
        "device": torch.device("cuda"),
    }
    declared = FlashInferPagedAttention.workspace_size(**options)
    activated = 0
    for batch in range(1, max_batch + 1):
        cmax = max((c for (b, _, _), c in table.items() if b == batch), default=1)
        for total in range(1, max_total + 1):
            if not any(
                b == batch and c > 1 and lo <= total <= b * hi
                for (b, (lo, hi), _), c in table.items()
            ):
                continue
            required = scheduler_counts(total, batch, 4, dim, cmax)
            if required is not None:
                activated += 1
                assert all(a >= b for a, b in zip(declared, required, strict=True))
    assert activated
    provider = FlashInferPagedAttention.workspace_size(**options, dispatch="provider")
    assert all(a >= b for a, b in zip(declared, provider, strict=True))


@pytest.fixture
def native(monkeypatch):
    wrapper = Mock()
    constructor = Mock(return_value=wrapper)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=constructor,
            BatchPrefillWithPagedKVCacheWrapper=constructor,
        ),
    )
    monkeypatch.setattr(inference, "qualified_flashinfer_version", lambda: True)
    return constructor, wrapper


def operation(**overrides):
    # A mock provider never accesses storage; expanded bytes avoid large CPU allocations.
    return FlashInferPagedAttention(
        torch.empty(1, dtype=torch.uint8).expand(1 << 30),
        torch.empty(1, dtype=torch.uint8).expand(1 << 20),
        **(
            {
                "mode": "prefill",
                "backend": "fa2",
                "use_cuda_graph": True,
                "qo_indptr_buffer": torch.empty(4, dtype=torch.int32),
                "paged_kv_indptr_buffer": torch.empty(4, dtype=torch.int32),
                "paged_kv_indices_buffer": torch.empty(8192, dtype=torch.int32),
                "paged_kv_last_page_len_buffer": torch.empty(3, dtype=torch.int32),
            }
            | overrides
        ),
    )


def plan(op, queries=(256, 256, 256), pages=(512, 512, 512), **overrides):
    op.plan(
        **(
            {
                "cu_seqlens_q": torch.tensor([0, *accumulate(queries)], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, *accumulate(pages)], dtype=torch.int32),
                "indices": torch.arange(sum(pages), dtype=torch.int32),
                "last_page_len": torch.tensor([int(p > 0) for p in pages], dtype=torch.int32),
                "seq_lens": torch.tensor(pages, dtype=torch.int32),
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "page_size": 1,
                "pos_encoding_mode": "NONE",
                "dtype": torch.bfloat16,
            }
            | overrides
        )
    )


def test_prefill_graph_owns_padded_buffers_once_and_replans(native, monkeypatch):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    constructor, wrapper = native
    op = operation()
    constructor.assert_not_called()
    indices = op._prefill_options["paged_kv_indices_buf"]
    plan(op)
    assert op.prefill_route == "padded_split"
    assert constructor.call_args.kwargs["paged_kv_indices_buf"] is indices
    owned = [
        constructor.call_args.kwargs[name]
        for name in (
            "qo_indptr_buf",
            "paged_kv_indptr_buf",
            "paged_kv_last_page_len_buf",
        )
    ]
    assert [x.numel() for x in owned] == [135, 135, 134]
    pointers = [x.data_ptr() for x in owned]
    for queries, pages, chunk in (
        ((256, 256, 256), (256, 128, 0), 43),
        ((256, 256, 256), (1024, 512, 257), 171),
        ((128, 128, 128), (512, 256, 128), None),
        ((256, 256, 256), (512, 512, 512), 86),
    ):
        plan(op, queries, pages)
        args = wrapper.plan.call_args.kwargs
        assert args.get("fixed_split_size") == chunk
        if chunk is None:
            assert "fixed_split_size" not in args
        assert args["qo_indptr"].tolist() == [0, *accumulate(queries)] + [sum(queries)] * 131
        assert args["paged_kv_indptr"].tolist() == [0, *accumulate(pages)] + [sum(pages)] * 131
        assert args["paged_kv_last_page_len"].tolist()[3:] == [0] * 131
        assert args["seq_lens"].tolist() == list(pages) + [0] * 131
        assert [x.data_ptr() for x in owned] == pointers
    constructor.assert_called_once()
    with pytest.raises(AttributeError):
        op.prefill_route = "native"


@pytest.mark.parametrize(
    "options",
    [
        {"causal": False},
        {"disable_split_kv": True},
        {"dtype": torch.float32},
        {"kv_dtype": torch.float16},
        {"o_dtype": torch.float16},
        {"window_left": 1},
        {"logits_soft_cap": 2.0},
        {"rope_scale": 1.0},
        {"rope_theta": 10000.0},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"custom_mask": torch.ones(1, dtype=torch.bool)},
        {"packed_custom_mask": torch.ones(1)},
    ],
)
def test_prefill_graph_rejects_untuned_and_never_switches(native, monkeypatch, options):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    op = operation()
    plan(op, **options)
    assert op.prefill_route == "native"
    plan(op)
    assert op.prefill_route == "native"
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs
    native[0].assert_called_once()


@pytest.mark.parametrize("reason", ["version", "tile", "workspace", "fa3", "provider", "table"])
def test_prefill_graph_route_rejections(native, monkeypatch, reason):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    if reason == "version":
        monkeypatch.setattr(inference, "qualified_flashinfer_version", lambda: False)
    if reason == "tile":
        monkeypatch.setitem(
            tables.PREFILL_GRAPH_SPLIT,
            ("rtx5090", 32, 8, 128),
            {
                (3, (7, 7), (1, None)): 2,
            },
        )
    op = operation(
        **(
            {"backend": "fa3"}
            if reason == "fa3"
            else {"dispatch": "provider"}
            if reason == "provider"
            else {}
        )
    )
    if reason == "workspace":
        op._workspaces = (torch.empty(0), torch.empty(0))
    queries = (7, 7, 7) if reason == "tile" else (128,) * 3 if reason == "table" else (256,) * 3
    plan(op, queries=queries)
    assert op.prefill_route == "native"
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs


@pytest.mark.parametrize(
    "queries,options,message",
    [
        ((256, 256), {}, "batch"),
        ((257, 256, 256), {}, "rows"),
        ((256, 256, 256), {"causal": False}, "geometry/options"),
        ((256, 256, 256), {"dtype": torch.float16}, "geometry/options"),
    ],
)
def test_prefill_graph_first_plan_contract(native, monkeypatch, queries, options, message):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    op = operation()
    plan(op)
    with pytest.raises(ValueError, match=message):
        plan(op, queries=queries, **options)
    assert native[1].plan.call_count == 1


def test_prefill_graph_capacity_fallback_does_not_pass_fixed_split(native, monkeypatch):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    op = operation()
    plan(op)
    monkeypatch.setattr(inference, "prefill_graph_chunks", lambda *args: 100)
    plan(op)
    assert op.prefill_route == "padded_split"
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs


def test_prefill_graph_padding_uses_largest_chunk_count_across_buckets(native, monkeypatch):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {
            ("rtx5090", 32, 8, 128): {
                (2, (129, 256), (0, 256)): 4,
                (2, (129, 256), (257, None)): 6,
            },
        },
    )
    op = operation()
    plan(op, queries=(256, 256), pages=(256, 256))
    assert op._prefill_graph_layout == (86, 128, 103)  # C_max=6, first C=4.
    assert native[1].plan.call_args.kwargs["fixed_split_size"] == 64
    plan(op, queries=(256, 256), pages=(512, 512))
    assert native[1].plan.call_args.kwargs["fixed_split_size"] == 86


@pytest.mark.parametrize("pages", [(0, 0, 0), (1, 1, 1)])
def test_prefill_graph_tiny_kv_uses_auto_split_on_capture_and_replay(native, monkeypatch, pages):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): 6}},
    )
    op = operation()
    plan(op, pages=pages)
    assert op.prefill_route == "padded_split"
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs
    plan(op)
    assert native[1].plan.call_args.kwargs["fixed_split_size"] == 86
    plan(op, pages=pages)
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs


@pytest.mark.parametrize("chunks", [1, 4])
def test_prefill_graph_tile_policy(native, monkeypatch, chunks):
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_SPLIT",
        {("rtx5090", 32, 8, 128): {(3, (129, 256), (0, None)): {"chunks": chunks, "tile": 64}}},
    )
    assert tables.prefill_graph_chunks("cuda", 32, 8, 128, 3, 256, 512) == chunks
    assert tables.prefill_graph_max_chunks("cuda", 32, 8, 128, 3) == chunks
    op = operation()
    plan(op)
    assert op.prefill_route == "padded_split"
    assert op._prefill_graph_layout == (750, 64, 800)
    assert native[1].plan.call_args.kwargs.get("fixed_split_size") == (None if chunks == 1 else 128)
    plan(op, queries=(64, 64, 64))
    assert op._prefill_graph_layout == (750, 64, 800)
    assert "fixed_split_size" not in native[1].plan.call_args.kwargs


@pytest.mark.parametrize(
    "total,batch,group,dim,target,expected",
    [
        (256, 1, 4, 128, 64, (240, 64, 256)),
        (128, 2, 4, 128, 64, (111, 64, 120)),
        (128, 2, 4, 256, 64, (111, 64, 120)),
        (64, 2, 6, 128, 64, (53, 64, 60)),
        (8, 2, 4, 128, 64, None),  # Negative padding.
        (256, 1, 128, 128, 64, None),  # floor(t/g)=0 selects tile16.
        (256, 1, 4, 256, 128, None),  # D256 cannot select tile128.
        (256, 1, 4, 128, 32, None),  # No such provider tile.
    ],
)
def test_prefill_graph_steered_layout(total, batch, group, dim, target, expected):
    assert tables.prefill_graph_split_layout(total, batch, group, dim, 1, 42, target) == expected


@pytest.mark.parametrize("dim", [128, 256])
def test_prefill_graph_steered_workspace_oracle(monkeypatch, dim):
    table = {
        (b, (1, 80), (0, None)): {"chunks": 1 if b % 2 else 4, "tile": 64} for b in range(1, 5)
    }
    monkeypatch.setattr(tables, "PREFILL_GRAPH_SPLIT", {("rtx5090", 32, 8, dim): table})
    sizes = FlashInferPagedAttention.workspace_size(
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        max_batch_size=4,
        max_query_tokens=320,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=dim,
        device="cuda",
    )
    for (batch, (_, hi), _), config in table.items():
        for total in range(1, batch * hi + 1):
            required = scheduler_counts(total, batch, 4, dim, config["chunks"], target=64)
            if required is not None:
                assert all(a >= b for a, b in zip(sizes, required, strict=True))


@pytest.fixture
def cute(monkeypatch):
    import inspect
    import os

    module = inference._prefill_cute
    constructor = Mock(return_value=Mock())
    constructor.__signature__ = inspect.signature(module.CutePagedPrefill)
    snapshots = []
    constructor.side_effect = lambda *a, **kw: (
        snapshots.append(
            {
                key: os.environ.get("CTI_PREFILL_" + key.upper())
                for key in ("block_m", "block_n", "stages", "splits")
            }
        )
        or constructor.return_value
    )
    monkeypatch.setattr(module, "CutePagedPrefill", constructor)
    monkeypatch.setattr(module, "check_support", Mock())
    monkeypatch.setattr(module, "workspace_size", Mock(return_value=1024))
    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_CUTE",
        {("rtx5090", 32, 8, 128): {(3, (129, 256)): {"block_m": 64, "splits": 2}}},
    )
    return constructor, snapshots


def test_cute_prefill_stages_borrowed_buffers_and_runs(native, cute, monkeypatch):
    import os

    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", "128")
    monkeypatch.delenv("CTI_PREFILL_SPLITS", raising=False)
    op = operation()
    buffers = tuple(
        op._prefill_options[name]
        for name in (
            "qo_indptr_buf",
            "paged_kv_indptr_buf",
            "paged_kv_indices_buf",
            "paged_kv_last_page_len_buf",
        )
    )
    original_copy = torch.Tensor.copy_
    copies = []

    def copy(target, source, *, non_blocking=False):
        copies.append((target.data_ptr(), source, non_blocking))
        return original_copy(target, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    plan(op, sm_scale=0.125)
    assert op.prefill_route == "cute"  # Wins over the shipped padded table.
    native[0].assert_not_called()
    kwargs = cute[0].call_args.kwargs
    assert kwargs["batch_size"] == 3
    assert kwargs["max_query_per_request"] == 768
    assert kwargs["sm_scale"] == 0.125
    assert cute[1] == [{"block_m": "64", "block_n": None, "stages": None, "splits": "2"}]
    assert os.environ["CTI_PREFILL_BLOCK_M"] == "128"
    assert "CTI_PREFILL_SPLITS" not in os.environ
    assert [item[0] for item in copies] == [buf.data_ptr() for buf in buffers]
    assert all(item[2] for item in copies)
    plan(op, queries=(400, 0, 1), pages=(4, 0, 2), sm_scale=0.125)
    assert buffers[0].tolist() == [0, 400, 400, 401]
    assert buffers[1].tolist() == [0, 4, 4, 6]
    assert buffers[2][:6].tolist() == list(range(6))
    assert buffers[3].tolist() == [1, 0, 1]
    cute[0].assert_called_once()
    query, key, value, out = (torch.empty(1) for _ in range(4))
    assert (
        op.run(q=query, paged_kv_cache=(key, value), out=out)
        is cute[0].return_value.run.return_value
    )
    args = cute[0].return_value.run.call_args.args
    assert all(a is b for a, b in zip(args, (query, key, value, *buffers, out), strict=True))


@pytest.mark.parametrize("reason", ["support", "table", "dtype", "causal", "provider", "eager"])
def test_cute_prefill_fallback_freezes(native, cute, monkeypatch, reason):
    if reason == "support":
        inference._prefill_cute.check_support.side_effect = NotImplementedError
    if reason == "table":
        monkeypatch.setattr(tables, "PREFILL_GRAPH_CUTE", {})
    op = operation(
        **(
            {"dispatch": "provider"}
            if reason == "provider"
            else {"use_cuda_graph": False}
            if reason == "eager"
            else {}
        )
    )
    plan(
        op,
        **(
            {"dtype": torch.float32}
            if reason == "dtype"
            else {"causal": False}
            if reason == "causal"
            else {}
        ),
    )
    assert op.prefill_route != "cute"
    inference._prefill_cute.check_support.side_effect = None
    plan(op)
    assert op.prefill_route != "cute"
    cute[0].assert_not_called()


@pytest.mark.parametrize(
    "queries,options,message",
    [
        ((256, 256), {}, "batch"),
        ((257, 256, 256), {}, "rows"),
        ((256, 256, 256), {"causal": False}, "geometry/options"),
        ((256, 256, 256), {"sm_scale": 0.5}, "geometry/options"),
    ],
)
def test_cute_prefill_first_plan_limits(native, cute, queries, options, message):
    op = operation()
    plan(op)
    with pytest.raises(ValueError, match=message):
        plan(op, queries=queries, **options)
    cute[0].assert_called_once()


def test_cute_prefill_small_workspace_and_failed_compile_restore_env(native, cute, monkeypatch):
    import os

    monkeypatch.setenv("CTI_PREFILL_SPLITS", "7")
    op = operation()
    op._workspaces = (torch.empty(1, dtype=torch.uint8), op._workspaces[1])
    with pytest.raises(ValueError, match="1024 borrowed float workspace bytes"):
        plan(op)
    assert os.environ["CTI_PREFILL_SPLITS"] == "7"
    cute[0].assert_not_called()
    cute[0].side_effect = RuntimeError("compile failure")
    with pytest.raises(RuntimeError, match="compile failure"):
        plan(operation())
    assert os.environ["CTI_PREFILL_SPLITS"] == "7"


def test_cute_prefill_workspace_covers_every_table_total(native, cute, monkeypatch):
    import os

    monkeypatch.setattr(
        tables,
        "PREFILL_GRAPH_CUTE",
        {
            ("rtx5090", 32, 8, 128): {
                (1, (4, 8)): {"splits": 2},
                (2, (3, 7)): {"splits": 3},
                (4, (1, 8)): {"splits": 4},
            },
        },
    )
    seen = set()

    def size(device, **bounds):
        b, t = bounds["batch_size"], bounds["max_query_per_request"]
        splits = int(os.environ["CTI_PREFILL_SPLITS"])
        seen.add((b, t, splits))
        return (1000 - t) * (1 << 20)  # Interior/smaller totals can dominate.

    monkeypatch.setattr(inference._prefill_cute, "workspace_size", size)
    sizes = FlashInferPagedAttention.workspace_size(
        mode="prefill",
        backend="fa2",
        use_cuda_graph=True,
        max_batch_size=3,
        max_query_tokens=12,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        device="cuda",
    )
    assert seen == {(1, t, 2) for t in range(4, 9)} | {(2, t, 3) for t in range(3, 13)}
    assert sizes[0] >= 997 * (1 << 20)


@pytest.mark.parametrize("layout", ["misaligned", "odd", "heads", "fused"])
def test_cute_prefill_query_staging_and_scales(native, cute, monkeypatch, layout):
    op = operation()
    plan(op)
    assert op.prefill_route == "cute"
    if layout == "misaligned":
        q = torch.randn(768 * 32 * 128 + 1, dtype=torch.bfloat16)[1:].view(768, 32, 128)
    elif layout == "fused":
        q = torch.randn(768, 48, 128, dtype=torch.bfloat16)[:, :32]
    else:
        q = torch.randn(768, 64, 129 if layout == "odd" else 128, dtype=torch.bfloat16)[
            :, ::2, :128
        ]
    out = torch.empty(q.shape, dtype=q.dtype)
    cache = torch.empty(8, 2, 16, 8, 128, dtype=q.dtype)
    pointer = op._cute_q_stage.data_ptr()
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("run allocation")))
    for k_scale, v_scale in ((0.37, 1.9), (1.0, 0.0), (-0.5, 1.0), (0.0, 2.0)):
        q.add_(1)
        op.run(
            q=q,
            paged_kv_cache=(cache[:, 0], cache[:, 1]),
            out=out,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        call = cute[0].return_value.run.call_args
        torch.testing.assert_close(call.args[0], q)
        assert call.args[0].data_ptr() == (
            q.data_ptr() if layout in ("heads", "fused") else pointer
        )
        assert call.kwargs == {"k_scale": k_scale, "v_scale": v_scale}
