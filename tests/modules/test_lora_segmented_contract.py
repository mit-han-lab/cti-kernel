# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Host-only contract checks; no CUDA driver or CuTe dependency required."""

import pytest
import torch

from ctikernels.linear import _lora_segmented as impl
from ctikernels.linear import lora_segmented_delta_, lora_segmented_projection


def operands():
    return {
        "output": torch.empty(3, 12, dtype=torch.bfloat16),
        "x": torch.empty(3, 16, dtype=torch.bfloat16),
        "lora_a": torch.empty(2, 16, 16, dtype=torch.bfloat16),
        "lora_b": torch.empty(2, 12, 8, dtype=torch.bfloat16),
        "seg_indptr": torch.tensor([0, 1, 3], dtype=torch.int32),
        "weight_indices": torch.tensor([0, 1], dtype=torch.int32),
        "ranks": torch.tensor([8, 3], dtype=torch.int32),
        "scalings": torch.ones(2),
        "output_sizes": (8, 4),
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("x", None, "must be tensors"),
        ("x", torch.empty(3, 16), "BF16 or FP16"),
        ("x", torch.empty(16, 3, dtype=torch.bfloat16).t(), "contiguous"),
        ("x", torch.empty(3, 1, 16, dtype=torch.bfloat16), "matrices"),
        ("output", torch.empty(3, 13, dtype=torch.bfloat16), "output and B"),
        ("output", torch.empty(12, 3, dtype=torch.bfloat16).t(), "unit last stride"),
        ("output", torch.empty(1, 12, dtype=torch.bfloat16).expand(3, 12), "nonoverlapping"),
        ("lora_a", torch.empty(2, 8, 16, dtype=torch.bfloat16), "A must have shape"),
        ("lora_b", torch.empty(2, 12, 513, dtype=torch.bfloat16), "max_rank"),
        ("seg_indptr", torch.tensor([0, 3], dtype=torch.int32), "S\\+1"),
        ("seg_indptr", torch.tensor([0, 1, 3]), "int32"),
        ("weight_indices", torch.zeros(2, 1, dtype=torch.int32), "S\\+1"),
        ("ranks", torch.ones(1, dtype=torch.int32), "one entry"),
        ("ranks", torch.ones(2), "int32"),
        ("scalings", torch.ones(2, dtype=torch.float64), "float32"),
        ("scalings", torch.ones(2, requires_grad=True), "inference-only"),
        ("scalings", torch.ones(4)[::2], "contiguous"),
        ("scalings", torch.ones(2, device="meta"), "same CUDA device"),
        ("output_sizes", (), "positive integer"),
        ("output_sizes", (8, 0, 4), "positive integer"),
        ("output_sizes", (8, 4.0), "positive integer"),
        ("output_sizes", (True, 11), "positive integer"),
        ("output_sizes", (8, 3), "output and B"),
        ("max_seq_len", -1, "nonnegative"),
        ("max_seq_len", 1.5, "nonnegative"),
    ],
)
def test_contract_errors(field, value, message):
    args = operands()
    args[field] = value
    with pytest.raises(ValueError, match=message):
        lora_segmented_delta_(**args)


def test_cpu_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        lora_segmented_delta_(**operands())


def test_alias_rejected():
    args = operands()
    args["output"] = args["x"][:, :12]
    with pytest.raises(ValueError, match="share storage"):
        lora_segmented_delta_(**args)


def test_row_strided_output_passes_layout_validation():
    args = operands()
    args["output"] = torch.empty(3, 20, dtype=torch.bfloat16)[:, 3:15]
    with pytest.raises(ValueError, match="requires CUDA"):
        lora_segmented_delta_(**args)


def test_no_tensor_values_are_read(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("host metadata scan")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with pytest.raises(ValueError, match="requires CUDA"):
        lora_segmented_delta_(**operands())


def test_chunk_grid_bound():
    # Enumerate short compositions including empty segments and ragged chunks.
    from itertools import product

    for lengths in product((0, 1, 15, 16, 17, 33), repeat=4):
        m = sum(lengths)
        assert sum((v + 15) // 16 for v in lengths) <= (m + 15) // 16 + len(lengths)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("weight", torch.empty(12, 15, dtype=torch.bfloat16), "expected x"),
        ("weight", torch.empty(12, 16), "match x"),
        ("bias", torch.empty(13, dtype=torch.bfloat16), "bias must"),
    ],
)
def test_projection_validation(field, value, message):
    args = operands()
    del args["output"]
    args.update(weight=torch.empty(12, 16, dtype=torch.bfloat16), bias=None)
    args[field] = value
    with pytest.raises(ValueError, match=message):
        lora_segmented_projection(**args)


def test_benchmark_request_and_chunk_metadata_match():
    from benchmark.kernel_tuning.lora_segmented_bench import cases, metadata

    configurations = list(cases())
    assert len(configurations) == 24
    for _mode, m, segments, adapters in configurations:
        ours, sgl = metadata(m, segments, adapters, 64, device="cpu")
        request_ids = torch.repeat_interleave(
            sgl.req_weight_indices, sgl.req_seg_indptr.diff().long()
        )
        ours_ids = torch.repeat_interleave(ours["weight_indices"], ours["seg_indptr"].diff().long())
        torch.testing.assert_close(ours_ids, request_ids[ours["permutation"].long()])
        torch.testing.assert_close(ours["permutation"], sgl.permutation)
        assert int(ours["seg_indptr"].diff().max()) <= 16
        chunk_ids = torch.repeat_interleave(sgl.weight_indices, sgl.seg_indptr.diff().long())
        torch.testing.assert_close(request_ids[sgl.permutation.long()], chunk_ids)
        assert len(request_ids) == m
        assert int(sgl.seg_lens.max()) <= sgl.max_len
        assert sgl.seg_indptr[0] == 0 and sgl.seg_indptr[-1] == m
        assert torch.unique(request_ids).numel() == adapters
        assert sgl.lora_ranks is ours["ranks"]
        assert sgl.scalings is ours["scalings"]


@pytest.mark.parametrize(
    "value,message",
    [
        (torch.arange(3), "int32"),
        (torch.arange(4, dtype=torch.int32), "int32"),
        (torch.arange(6, dtype=torch.int32)[::2], "contiguous"),
        (torch.empty(3, dtype=torch.int32, device="meta"), "same CUDA device"),
        ([], "must be tensors"),
    ],
)
def test_permutation_contract(value, message):
    with pytest.raises(ValueError, match=message):
        lora_segmented_delta_(**operands(), permutation=value)


@pytest.mark.parametrize("stride", ["rank", "max_rank"])
def test_pool_layout_modes_pass_validation(stride):
    with pytest.raises(ValueError, match="requires CUDA"):
        lora_segmented_delta_(
            **operands(), a_slice_stride=stride, permutation=torch.arange(3, dtype=torch.int32)
        )


def test_bad_pool_layout():
    with pytest.raises(ValueError, match="a_slice_stride"):
        lora_segmented_delta_(**operands(), a_slice_stride="padded")


def test_grouped_metadata_stable_and_crosses_requests():
    meta = impl.lora_segment_metadata([2, 1, 2, -1, 1], [2, 1, 2, 1, 2], device="cpu")
    assert meta["permutation"].tolist() == [5, 2, 6, 7, 0, 1, 3, 4]
    assert meta["seg_indptr"].tolist() == [0, 1, 4, 8]
    assert meta["weight_indices"].tolist() == [-1, 1, 2]
    meta = impl.lora_segment_metadata([0] * 64, [1] * 64)
    assert meta["seg_indptr"].tolist() == [0, 16, 32, 48, 64]
    assert meta["weight_indices"].numel() == 4
    assert meta["permutation"].tolist() == list(range(64))
    empty = impl.lora_segment_metadata([1, 2], [0, 0])
    assert empty["seg_indptr"].tolist() == [0]
    assert empty["permutation"].numel() == empty["weight_indices"].numel() == 0


@pytest.mark.parametrize("ids,lengths", [([1], [1, 2]), ([1], [-1]), ([1.5], [2]), ([1], [2.5])])
def test_bad_request_metadata(ids, lengths):
    with pytest.raises(ValueError):
        impl.lora_segment_metadata(ids, lengths)


def factor_operands():
    return {
        "output": torch.empty(3, 12, dtype=torch.bfloat16),
        "x": torch.empty(3, 16, dtype=torch.bfloat16),
        "factors": (
            torch.empty(8, 16, dtype=torch.bfloat16),
            torch.empty(8, 8, dtype=torch.bfloat16),
            torch.empty(8, 16, dtype=torch.bfloat16),
            torch.empty(4, 8, dtype=torch.bfloat16),
        ),
        "scaling": torch.tensor(2.0),
        "output_sizes": (8, 4),
    }


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("factors", (), "nonempty tuple"),
        ("output_sizes", (12,), "per factor pair"),
        ("scaling", torch.tensor(2.0, dtype=torch.float64), "float32 scalar"),
        ("scaling", torch.ones(2), "float32 scalar"),
        ("scaling", torch.tensor(2.0, requires_grad=True), "inference-only"),
        ("output", torch.empty(3, 13, dtype=torch.bfloat16), "output must match"),
        ("x", torch.empty(3, 16), "BF16 or FP16"),
    ],
)
def test_factor_contract(field, value, message):
    args = factor_operands()
    args[field] = value
    with pytest.raises(ValueError, match=message):
        impl.lora_delta_factors_(**args)


def test_factor_no_host_reads_and_shared_rank(monkeypatch):
    args = factor_operands()

    def forbidden(*args, **kwargs):
        raise AssertionError("host scan")

    for method in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    with pytest.raises(ValueError, match="requires CUDA"):
        impl.lora_delta_factors_(**args)
    args["factors"] = (*args["factors"][:2], torch.empty(4, 16), torch.empty(4, 4))
    with pytest.raises(ValueError, match="shared rank"):
        impl.lora_delta_factors_(**args)


def test_pointer_table_cache_and_cross_stream_capture(monkeypatch):
    # Exercise real address arithmetic without a CUDA context or GPU allocations.
    from types import SimpleNamespace

    real_tensor, real_empty = torch.tensor, torch.empty
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: real_tensor(*a, **dict(kw, device="cpu")))
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: real_empty(*a, **dict(kw, device="cpu")))
    waits = []
    stream = SimpleNamespace(cuda_stream=1, wait_event=lambda event: waits.append(event))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)

    class Event:
        def record(self, stream):
            self.stream = stream.cuda_stream

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(impl, "_POINTER_TABLES", {})
    a, b = torch.empty(2, 16, 16, dtype=torch.bfloat16), torch.empty(2, 12, 8, dtype=torch.bfloat16)
    offsets = ((0, 0, 256, 256), (0, 64, 96, 160))
    tables = impl._pointer_tables((a,), (b,), a.device, False, sizes=(8, 4), offsets=offsets)
    assert tables[0].dtype == tables[1].dtype == torch.int64
    assert tables[0].tolist() == [a.data_ptr() + 2 * n for n in offsets[0]]
    assert tables[1].tolist() == [b.data_ptr() + 2 * n for n in offsets[1]]
    assert tables[2].numel() == 0
    assert tables[3].tolist() == [0, 8, 8, 4]
    assert impl._pointer_tables((a,), (b,), a.device, True, sizes=(8, 4), offsets=offsets) is tables
    assert not waits
    stream.cuda_stream = 2
    assert impl._pointer_tables((a,), (b,), a.device, True, sizes=(8, 4), offsets=offsets) is tables
    assert len(waits) == 0
    # Capture waits cannot mark a stream as initialized for later eager calls.
    impl._pointer_tables((a,), (b,), a.device, False, sizes=(8, 4), offsets=offsets)
    assert len(waits) == 1
    impl._pointer_tables((a,), (b,), a.device, False, sizes=(8, 4), offsets=offsets)
    assert len(waits) == 1
    with pytest.raises(RuntimeError, match="warm it eagerly"):
        impl._pointer_tables((a.clone(),), (b,), a.device, True, sizes=(8, 4), offsets=offsets)
    separate = impl._pointer_tables((a[0], a[1]), (b[0], b[1]), a.device, False, sizes=(12, 12))
    assert separate[0].tolist() == [a[0].data_ptr(), a[1].data_ptr()]
    assert separate[1].tolist() == [b[0].data_ptr(), b[1].data_ptr()]


@pytest.mark.skipif(
    not torch.backends.cuda.is_built(), reason="fake CUDA tensors need a CUDA-built PyTorch"
)
@pytest.mark.parametrize("chunk", [16, 32, 64])
@pytest.mark.parametrize("m", [0, 65])
def test_device_metadata_no_readback_with_fake_tensors(m, monkeypatch, chunk):
    from torch._subclasses.fake_tensor import FakeTensorMode

    def forbidden(*args, **kwargs):
        raise AssertionError("device metadata readback")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with FakeTensorMode():
        ids = torch.empty(4, device="cuda", dtype=torch.int32)
        lengths = torch.empty_like(ids)
        meta = impl.lora_segment_metadata(ids, lengths, total_tokens=m, chunk_tokens=chunk)
        assert meta["seg_indptr"].shape == (m + 1,)
        assert meta["weight_indices"].shape == meta["permutation"].shape == (m,)
        tensors = [meta[key] for key in ("seg_indptr", "weight_indices", "permutation")]
        tensors.extend(meta["chunks"][:3])
        assert all(t.dtype == torch.int32 and t.device == ids.device for t in tensors)
        assert all(t.shape == ((m + chunk - 1) // chunk + m,) for t in meta["chunks"][:3])


@pytest.mark.parametrize(
    "ids,lengths",
    [
        ([2, 1, 2, -1, 1], [2, 1, 19, 1, 20]),
        ([0] * 64, [1] * 64),
        (list(range(64)), [1] * 64),
        ([1, 2], [0, 0]),
        ([], []),
    ],
)
@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_chunk_table_covers_sorted_rows_and_padding(ids, lengths, chunk):
    meta = impl.lora_segment_metadata(ids, lengths, chunk_tokens=chunk)
    table = meta["chunks"]
    m, s = sum(lengths), meta["weight_indices"].numel()
    assert table.count == s
    assert table.chunk_tokens == chunk
    assert (table.chunk_end[:s] - table.chunk_start[:s]).le(chunk).all()
    assert all(t.shape == ((m + chunk - 1) // chunk + s,) for t in table[:3])
    torch.testing.assert_close(table.chunk_start[:s], meta["seg_indptr"][:-1])
    torch.testing.assert_close(table.chunk_end[:s], meta["seg_indptr"][1:])
    torch.testing.assert_close(table.chunk_adapter[:s], meta["weight_indices"])
    assert table.chunk_adapter[s:].eq(-1).all()
    assert table.chunk_start[s:].eq(m).all() and table.chunk_end[s:].eq(m).all()
    assert all(t.dtype == torch.int32 and t.is_contiguous() for t in table[:3])


@pytest.mark.parametrize(
    "bad,message",
    [
        (None, "LoRAChunkTable"),
        ("shape", "chunk arrays"),
        ("dtype", "chunk arrays"),
        ("count", "chunk count"),
        ("stride", "contiguous"),
        ("device", "same CUDA device"),
    ],
)
def test_chunk_table_validation(bad, message):
    table = impl.lora_segment_metadata([0, 1], [1, 2])["chunks"]
    if bad is None:
        table = tuple(table[:3])
    elif bad == "shape":
        table = table._replace(chunk_start=table.chunk_start[:2])
    elif bad == "dtype":
        table = table._replace(chunk_end=table.chunk_end.long())
    elif bad == "count":
        table = table._replace(count=4)
    elif bad == "stride":
        table = table._replace(chunk_start=torch.zeros(6, dtype=torch.int32)[::2])
    elif bad == "device":
        table = table._replace(chunk_start=torch.empty(3, dtype=torch.int32, device="meta"))
    with pytest.raises(ValueError, match=message):
        lora_segmented_delta_(**operands(), chunks=table)


def test_chunk_table_validation_does_not_read_values(monkeypatch):
    table = impl.lora_segment_metadata([0, 1], [1, 2])["chunks"]

    def forbidden(*args, **kwargs):
        raise AssertionError("host metadata scan")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, forbidden)
    with pytest.raises(ValueError, match="requires CUDA"):
        lora_segmented_delta_(**operands(), chunks=table)


@pytest.mark.parametrize("chunk", [0, 8, 128, True, 32.0, "32", None])
def test_invalid_metadata_chunk(chunk):
    with pytest.raises(ValueError, match="chunk_tokens"):
        impl.lora_segment_metadata([0], [65], chunk_tokens=chunk)


def test_chunk_override_matches_table():
    chunks = impl.lora_segment_metadata([0], [65], chunk_tokens=32)["chunks"]
    assert impl._launch_chunk((None,) * 4, chunks) == 32
    assert impl._launch_chunk((None, None, None, 32), chunks) == 32
    with pytest.raises(ValueError, match="rebuild metadata"):
        impl._launch_chunk((None, None, None, 64), chunks)
    assert impl._launch_chunk((None,) * 4, None) == 16
    assert impl._launch_chunk((None, None, None, 64), None) == 64
