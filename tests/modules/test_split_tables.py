# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Device table selection and decode graph layout guards, without CUDA."""

import sys
from itertools import combinations, pairwise
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _flashinfer, inference
from ctikernels.attention import _split_tables as tables
from ctikernels.attention.inference import FlashInferPagedAttention


@pytest.fixture(autouse=True)
def fresh_qualification():
    # The qualification is cached per process; tests patch the versions it reads.
    _flashinfer.qualified_flashinfer_version.cache_clear()
    yield
    _flashinfer.qualified_flashinfer_version.cache_clear()


@pytest.mark.parametrize(
    "tokens,expected",
    [(0, 0), (512, 0), (1000, 0), (3000, 2), (1000000, 5)]
    + [(edge - 1, i) for i, edge in enumerate((1024, 2048, 4096, 8192, 16384))]
    + [(edge, i + 1) for i, edge in enumerate((1024, 2048, 4096, 8192, 16384))],
)
def test_kv_bucket(tokens, expected):
    assert tables.kv_bucket(tokens) == expected


@pytest.mark.parametrize(
    "major,minor,sms,expected",
    [
        (12, 0, 170, "rtx5090"),
        (12, 0, 188, "rtxpro6000"),
        (12, 1, 48, "gb10"),
        (12, 0, 48, None),
        (12, 1, 170, None),
        (9, 0, 132, None),
    ],
)
def test_device_key(monkeypatch, major, minor, sms, expected):
    properties = Mock(
        return_value=SimpleNamespace(major=major, minor=minor, multi_processor_count=sms)
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    device = torch.device("cuda", 2)
    assert tables.device_key(device) == expected
    properties.assert_called_once_with(device)


@pytest.mark.parametrize(
    "key,op,heads,batch,expected",
    [
        ("rtx5090", "decode", (32, 8, 128), 1, (8, 0, 0, 0, 16, 16)),
        ("rtx5090", "decode", (32, 8, 128), 4, (0,) * 6),
        ("rtxpro6000", "decode", (32, 8, 128), 1, (8, 16, 0, 0, 0, 0)),
        ("rtx5090", "decode", (16, 4, 256), 2, (8, 8, 8, 8, 16, 16)),
        ("rtxpro6000", "decode", (16, 4, 256), 4, (0, 0, 8, 8, 8, 8)),
        ("rtx5090", "prefill", (32, 8, 128), 1, None),
        ("rtx5090", "decode", (32, 8, 64), 1, None),
        ("gb10", "decode", (32, 8, 128), 1, None),
        ("gb10", "decode", (16, 4, 256), 1, (0, 0, 0, 0, 8, 8)),
        ("gb10", "decode", (16, 4, 256), 4, (0, 0, 0, 0, 2, 2)),
        (None, "decode", (32, 8, 128), 1, None),
        ("rtx5090", "decode", (32, 8, 128), 0, None),
        ("rtx5090", "decode", (32, 8, 128), 5, None),
    ],
)
def test_lookup(monkeypatch, key, op, heads, batch, expected):
    monkeypatch.setattr(tables, "device_key", lambda _: key)
    assert tables.lookup(torch.device("cuda"), op, *heads, batch) == expected


@pytest.mark.parametrize(
    "batch,expected",
    [(1, {"min_chunk": 64}), (2, {"min_chunk": 64}), (3, {"min_chunk": 64}), (4, {}), (5, None)],
)
def test_pro6000_qwen35_fp8_decode(monkeypatch, batch, expected):
    monkeypatch.setattr(tables, "device_key", lambda _: "rtxpro6000")
    assert tables.fp8_decode_cute(torch.device("cuda"), 16, 4, 256, batch) == expected


def test_decode_table_shapes():
    assert len(tables._TABLES) == 5
    for table in tables._TABLES.values():
        assert len(table) == 4
        for row in table:
            assert len(row) == 6
            assert all(value >= 0 for value in row)


@pytest.mark.parametrize(
    "row,pages,tokens,capacity,expected",
    [
        (None, [512], 512, 42, None),
        ((0,) * 6, [512], 512, 42, None),
        ((8,) * 6, [], 0, 42, None),
        ((8,) * 6, [0], 0, 42, None),
        ((8,) * 6, [1, 1], 1, 42, None),  # Must retain at least two chunks.
        ((1,) * 6, [512], 512, 42, None),
        ((8,) * 6, [512], 512, 8, 64),
        ((8,) * 6, [1000, 3], 1000, 9, 125),  # Ragged sum, not B*S.
        ((8,) * 6, [1000, 3], 1000, 8, None),
        ((8,) * 6, [1001, 3], 1001, 9, 126),
        ((8, 0, 0, 0, 0, 0), [1024], 1023, 8, 128),
        ((8, 0, 0, 0, 0, 0), [1024], 1024, 8, None),
    ],
)
def test_decode_chunk(row, pages, tokens, capacity, expected):
    assert tables.decode_chunk(row, pages, tokens, capacity) == expected


@pytest.fixture
def decode_native(monkeypatch):
    native = Mock(_plan_info=[42, 1, 2, 3] + [0] * 11, use_tensor_cores=True, _backend="fa2")
    monkeypatch.setattr(_flashinfer, "version", lambda _: "0.6.11.post3")
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(
            __version__="0.6.11.post3",
            BatchDecodeWithPagedKVCacheWrapper=Mock(return_value=native),
            BatchPrefillWithPagedKVCacheWrapper=Mock(return_value=native),
        ),
    )
    monkeypatch.setattr(tables, "device_key", lambda _: "rtx5090")
    # RTX 5090: 2 * 170 SMs // 8 KV heads.
    monkeypatch.setattr(inference, "decode_split_capacity", lambda device, kv_heads: 42)
    return native


def operation(*, graph=True, policy="device", tensor_cores=True):
    return FlashInferPagedAttention(
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
        mode="decode",
        backend="fa2",
        use_tensor_cores=tensor_cores,
        use_cuda_graph=graph,
        dispatch=policy,
        paged_kv_last_page_len_buffer=torch.ones(1, dtype=torch.int32) if graph else None,
    )


def plan(wrapper, lengths=(512,)):
    wrapper.plan(
        cu_seqlens_q=torch.arange(len(lengths) + 1, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32),
        indices=torch.arange(sum(lengths), dtype=torch.int32),
        last_page_len=torch.ones(len(lengths), dtype=torch.int32),
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=1,
        pos_encoding_mode="NONE",
        dtype=torch.bfloat16,
    )


def test_graph_layout_is_immutable(decode_native):
    wrapper = operation()
    plan(wrapper, (1,))
    assert "fixed_split_size" not in decode_native.plan.call_args.kwargs
    plan(wrapper)
    assert decode_native.plan.call_args.kwargs["fixed_split_size"] == 64
    decode_native._plan_info[1] = 99  # In-place provider changes must not change our snapshot.
    with pytest.raises(RuntimeError, match="captured plan layout"):
        plan(wrapper)


@pytest.mark.parametrize("lengths", [(512,), (1,), (2048,), (16384,)])
def test_eager_and_graph_plans_pick_the_same_chunk(decode_native, lengths):
    chunks = []
    for graph in (False, True):
        wrapper = operation(graph=graph)
        plan(wrapper, lengths)  # The graph's first (capture) plan uses the table too.
        plan(wrapper, lengths)
        chunks.append(decode_native.plan.call_args.kwargs.get("fixed_split_size"))
    assert chunks[0] == chunks[1]
    assert chunks[0] == {(512,): 64, (16384,): 1024}.get(lengths)


@pytest.mark.parametrize("graph", [False, True])
def test_split_capacity_bounds_eager_and_graph_alike(monkeypatch, decode_native, graph):
    # 512 pages / S=8 gives 8 chunks: over a 7-item capacity in both modes.
    monkeypatch.setattr(inference, "decode_split_capacity", lambda device, kv_heads: 7)
    wrapper = operation(graph=graph)
    plan(wrapper)
    assert "fixed_split_size" not in decode_native.plan.call_args.kwargs
    monkeypatch.setattr(inference, "decode_split_capacity", lambda device, kv_heads: 8)
    plan(wrapper)
    assert decode_native.plan.call_args.kwargs["fixed_split_size"] == 64


@pytest.mark.parametrize("batches", [(1, 2, 3, 4, 1), (2, 1, 4)])
def test_eager_decode_takes_the_graph_kernel_per_batch(monkeypatch, decode_native, batches):
    constructor = sys.modules["flashinfer"].BatchDecodeWithPagedKVCacheWrapper
    monkeypatch.setattr(FlashInferPagedAttention, "_workspace_fits", lambda *a, **k: True)
    eager = operation(graph=False)
    for batch in batches:
        plan(eager, (512,) * batch)
        graph = FlashInferPagedAttention(
            torch.zeros(0, dtype=torch.uint8),
            torch.zeros(0, dtype=torch.uint8),
            mode="decode",
            backend="fa2",
            use_tensor_cores=True,
            use_cuda_graph=True,
            paged_kv_last_page_len_buffer=torch.ones(batch, dtype=torch.int32),
        )
        plan(graph, (512,) * batch)
        assert eager.decode_route == graph.decode_route
        assert eager.decode_route == ("fa2" if batch == 1 else "cuda_cores")
    # Eager keeps one native wrapper per kernel: at most two constructions, plus
    # one per graph operation.
    kernels = {batch == 1 for batch in batches}
    assert constructor.call_count == len(kernels) + len(batches)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("policy,tensor_cores", [("provider", True), ("device", False)])
def test_provider_and_non_tensor_core_decode_never_consult_tables(
    monkeypatch, decode_native, graph, policy, tensor_cores
):
    monkeypatch.setattr(tables, "device_key", Mock(side_effect=AssertionError("no table lookup")))
    wrapper = operation(graph=graph, policy=policy, tensor_cores=tensor_cores)
    plan(wrapper)
    plan(wrapper)
    assert "fixed_split_size" not in decode_native.plan.call_args.kwargs


def test_eager_table_applies_on_first_plan(decode_native):
    wrapper = operation(graph=False)
    plan(wrapper)
    assert decode_native.plan.call_args.kwargs["fixed_split_size"] == 64


def test_device_lengths_keep_provider_default_without_readback(decode_native):
    wrapper = operation(graph=False)
    wrapper.plan(
        cu_seqlens_q=None,
        cu_seqlens_k=torch.empty(2, device="meta"),
        indices=None,
        last_page_len=None,
        seq_lens=None,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=1,
        pos_encoding_mode="NONE",
        dtype=torch.bfloat16,
    )
    assert "fixed_split_size" not in decode_native.plan.call_args.kwargs


@pytest.mark.parametrize("key", ["rtx5090", "rtxpro6000", "gb10", None])
@pytest.mark.parametrize("batch", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("heads", [(32, 8, 128), (16, 4, 256), (16, 8, 128)])
def test_decode_cuda_core_routing(monkeypatch, key, batch, heads):
    monkeypatch.setattr(tables, "device_key", lambda _: key)
    # GB10 keeps tensor cores at every batch.
    expected = heads == (32, 8, 128) and key in ("rtx5090", "rtxpro6000") and 2 <= batch <= 4
    assert tables.decode_uses_cuda_cores(torch.device("cuda"), *heads, batch) == expected


def prefill_split(lengths, *, key="rtx5090", kv_lengths=None, **options):
    from itertools import accumulate

    return tables.prefill_query_split(
        key,
        [0, *accumulate(lengths)],
        [0, *accumulate(kv_lengths or lengths)],
        list(kv_lengths or lengths),
        **{"head_dim": 128, "causal": True, "page_size": 1, **options},
    )


@pytest.mark.parametrize(
    "key,lengths,chunk",
    [
        ("rtx5090", [22], 4),
        ("rtx5090", [65], 8),
        ("rtx5090", [256], 16),
        ("rtxpro6000", [17], 8),
        ("rtxpro6000", [129], 16),
        ("gb10", [49], 16),
        ("gb10", [128], 16),  # next power of two is 32, capped at 16.
        ("rtx5090", [5, 22, 40], 8),
        ("rtx5090", [8] * 4, 4),
        ("gb10", [22, 22], 8),
    ],
)
def test_prefill_split_preserves_order_prefixes_and_remainders(key, lengths, chunk):
    split = prefill_split(lengths, key=key)
    assert split is not None
    assert split.qo_indptr[-1] == sum(lengths)
    query = request_start = pseudo = 0
    for length in lengths:
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            assert split.qo_indptr[pseudo : pseudo + 2] == [query + start, query + end]
            assert split.seq_lens[pseudo] == end
            positions = split.gather_positions[
                split.kv_indptr[pseudo] : split.kv_indptr[pseudo + 1]
            ]
            assert positions == list(range(request_start, request_start + end))
            # Bottom-right alignment preserves the original causal query position.
            assert end - (end - start) == start
            pseudo += 1
        query += length
        request_start += length
    assert split.last_page_len == [1] * pseudo


@pytest.mark.parametrize(
    "lengths,options",
    [
        ([22], {"key": None}),
        ([22], {"head_dim": 256}),
        ([22], {"causal": False}),
        ([22], {"page_size": 2}),
        ([22, 22], {"kv_lengths": [22, 23]}),
        ([0], {}),
        ([4], {}),
        ([257], {}),
        ([8] * 5, {}),
        ([16], {"key": "rtxpro6000"}),
        ([257], {"key": "rtxpro6000"}),
        ([129], {"key": "gb10"}),
        ([22] * 3, {"key": "gb10"}),
    ],
)
def test_prefill_split_guards(lengths, options):
    assert prefill_split(lengths, **options) is None


@pytest.mark.parametrize("required", [(8, 8), (9, 8), (8, 9)])
def test_lazy_decode_constructs_once_and_checks_both_workspaces(
    monkeypatch, decode_native, required
):
    constructor = sys.modules["flashinfer"].BatchDecodeWithPagedKVCacheWrapper
    sizing = Mock(return_value=required)
    monkeypatch.setattr(inference, "paged_attention_workspace_size", sizing)
    wrapper = FlashInferPagedAttention(
        torch.zeros(8, dtype=torch.uint8),
        torch.zeros(8, dtype=torch.uint8),
        mode="decode",
        backend="fa2",
        use_tensor_cores=True,
        use_cuda_graph=True,
        paged_kv_last_page_len_buffer=torch.ones(2, dtype=torch.int32),
    )
    constructor.assert_not_called()
    with pytest.raises(RuntimeError, match=r"plan\(\) before run"):
        wrapper.run(q=None, paged_kv_cache=None)
    plan(wrapper, (512, 512))
    expected_tc = required != (8, 8)
    assert wrapper._use_tensor_cores is expected_tc
    assert constructor.call_args.kwargs["use_tensor_cores"] is expected_tc
    decode_native.reset_workspace_buffer.assert_called_once_with(*wrapper._workspaces)
    assert sizing.call_args.kwargs["use_tensor_cores"] is False
    assert sizing.call_args.kwargs["max_batch_size"] == 2
    plan(wrapper, (512, 512))
    constructor.assert_called_once()
    if not expected_tc:
        assert "fixed_split_size" not in decode_native.plan.call_args.kwargs


@pytest.mark.parametrize(
    "graph,policy,tensor_cores",
    [(False, "device", True), (True, "provider", True), (True, "device", False)],
)
def test_decode_construction_defers_device_routes(decode_native, graph, policy, tensor_cores):
    wrapper = operation(graph=graph, policy=policy, tensor_cores=tensor_cores)
    if policy == "device":
        # Q/KV dtype is first known at plan time; FP8 can select XQA even
        # without requested tensor cores, and must not allocate a native wrapper.
        sys.modules["flashinfer"].BatchDecodeWithPagedKVCacheWrapper.assert_not_called()
        decode_native.reset_workspace_buffer.assert_not_called()
        plan(wrapper)
    sys.modules["flashinfer"].BatchDecodeWithPagedKVCacheWrapper.assert_called_once()
    decode_native.reset_workspace_buffer.assert_called_once()


def prefill_plan(wrapper, lengths=(22,), *, kv_lengths=None, **options):
    from itertools import accumulate

    kv_lengths = kv_lengths or lengths
    wrapper.plan(
        cu_seqlens_q=torch.tensor([0, *accumulate(lengths)], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, *accumulate(kv_lengths)], dtype=torch.int32),
        indices=torch.arange(sum(kv_lengths), dtype=torch.int32).flip(0),
        last_page_len=torch.ones(len(lengths), dtype=torch.int32),
        seq_lens=torch.tensor(kv_lengths, dtype=torch.int32),
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        page_size=1,
        pos_encoding_mode="NONE",
        dtype=torch.bfloat16,
        **options,
    )


@pytest.mark.parametrize("required", [(8, 8), (9, 8), (8, 9)])
def test_prefill_workspace_fallback(monkeypatch, decode_native, required):
    monkeypatch.setattr(inference, "device_key", lambda _: "rtx5090")
    sizing = Mock(return_value=required)
    monkeypatch.setattr(inference, "paged_attention_workspace_size", sizing)
    wrapper = FlashInferPagedAttention(
        torch.zeros(8, dtype=torch.uint8),
        torch.zeros(8, dtype=torch.uint8),
        mode="prefill",
        backend="fa2",
    )
    gather = Mock(side_effect=lambda indices, positions: indices[positions])
    monkeypatch.setattr(wrapper, "_gather_prefill_indices", gather)
    prefill_plan(wrapper)
    actual = decode_native.plan.call_args.kwargs
    assert sizing.call_args.kwargs["max_batch_size"] == 6
    assert sizing.call_args.kwargs["max_query_tokens"] == 22
    if required == (8, 8):
        expected = prefill_split([22])
        assert actual["qo_indptr"].tolist() == expected.qo_indptr
        assert actual["paged_kv_indptr"].tolist() == expected.kv_indptr
        assert actual["seq_lens"].tolist() == expected.seq_lens
        assert actual["paged_kv_indices"].tolist() == [21 - i for i in expected.gather_positions]
    else:
        gather.assert_not_called()
        assert actual["qo_indptr"].tolist() == [0, 22]
        assert actual["paged_kv_indices"].tolist() == list(reversed(range(22)))


def test_prefill_gather_reuses_capacity(monkeypatch, decode_native):
    # CPU simulation exercises the gather and reuse; GPU coverage checks pinned H2D.
    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: empty(
            *args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}
        ),
    )
    wrapper = FlashInferPagedAttention(
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
        mode="prefill",
        backend="fa2",
    )
    indices = torch.tensor([19, 3, 11, 5], dtype=torch.int32)
    for positions in ([0, 1, 0, 1, 2, 3], [2, 0, 2], [0] * 7):
        previous = wrapper._split_indices
        result = wrapper._gather_prefill_indices(indices, positions)
        assert result.tolist() == [indices[i].item() for i in positions]
        if previous is not None:
            assert (previous is wrapper._split_indices) == (len(positions) <= previous.numel())


@pytest.mark.parametrize(
    "policy,options",
    [("provider", {}), ("device", {"causal": False}), ("device", {"kv_lengths": (23,)})],
)
def test_prefill_unsplit_plans_never_size_or_gather(monkeypatch, decode_native, policy, options):
    monkeypatch.setattr(inference, "device_key", lambda _: "rtx5090")
    sizing = Mock(side_effect=AssertionError("no split workspace check"))
    monkeypatch.setattr(inference, "paged_attention_workspace_size", sizing)
    wrapper = FlashInferPagedAttention(
        torch.zeros(0, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.uint8),
        mode="prefill",
        backend="fa2",
        dispatch=policy,
    )
    prefill_plan(wrapper, **options)
    assert decode_native.plan.call_args.kwargs["qo_indptr"].tolist() == [0, 22]
    assert wrapper._gather_host is None


@pytest.mark.parametrize("key", ["rtx5090", "rtxpro6000", "gb10", None])
@pytest.mark.parametrize("max_batch,max_total", [(5, 24), (2, 140)])
def test_prefill_split_bound_brute_force(monkeypatch, key, max_batch, max_total):
    minor, sms = {"rtx5090": (0, 170), "rtxpro6000": (0, 188), "gb10": (1, 48), None: (0, 48)}[key]
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            major=12,
            minor=minor,
            multi_processor_count=sms,
            max_threads_per_multi_processor=1536,
        ),
    )
    maxima = {}
    for total in range(1, max_total + 1):
        for batch in range(1, min(max_batch, total) + 1):
            largest = 0
            bound = tables.prefill_split_batch_bound(key, batch, total)
            for cuts in combinations(range(1, total), batch - 1):
                lengths = [end - start for start, end in pairwise((0, *cuts, total))]
                split = prefill_split(lengths, key=key)
                if split is not None:
                    assert bound is not None and len(split.seq_lens) <= bound
                    assert split.qo_indptr[-1] <= total
                    largest = max(largest, len(split.seq_lens))
            maxima[batch, total] = max(
                largest, maxima.get((batch - 1, total), 0), maxima.get((batch, total - 1), 0)
            )
            assert bound == (maxima[batch, total] or None)
            if largest:
                options = {
                    "mode": "prefill",
                    "backend": "fa2",
                    "max_query_tokens": total,
                    "num_qo_heads": 32,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "device": torch.device("cuda"),
                }
                advertised = FlashInferPagedAttention.workspace_size(
                    max_batch_size=batch, **options
                )
                required = FlashInferPagedAttention.workspace_size(
                    max_batch_size=largest,
                    dispatch="provider",
                    **options,
                )
                assert all(a >= r for a, r in zip(advertised, required, strict=True))


@pytest.mark.parametrize("candidate", ["0.6.11.post3", "0.7.0.post1", "unqualified"])
@pytest.mark.parametrize("module_version", ["0.6.11.post3", "0.7.0.post1", "unqualified", None])
def test_decode_plan_capacity_version_gate(monkeypatch, candidate, module_version):
    monkeypatch.setattr(_flashinfer, "version", lambda _: candidate)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__=module_version))
    values = [42, 1, 2, 3] + [0] * 11
    native = SimpleNamespace(_plan_info=values, use_tensor_cores=True, _backend="fa2")
    result = _flashinfer.decode_plan_capacity(native)
    if candidate in _flashinfer.QUALIFIED_VERSIONS and candidate == module_version:
        assert result == (42, tuple(values))
        values[0] = 99
        assert result[0] == result[1][0] == 42
    else:
        assert result is None


@pytest.mark.parametrize("graph", [False, True])
def test_unqualified_decode_keeps_provider_default(monkeypatch, decode_native, graph):
    monkeypatch.setattr(_flashinfer, "version", lambda _: "unqualified")
    monkeypatch.setattr(inference, "lookup", Mock(side_effect=AssertionError("no table lookup")))
    wrapper = operation(graph=graph)
    plan(wrapper)
    plan(wrapper)
    assert "fixed_split_size" not in decode_native.plan.call_args.kwargs
    assert wrapper._capture_plan_info is None


@pytest.mark.parametrize(
    "overrides",
    [{"use_tensor_cores": False}, {"_backend": "fa3"}, {"_plan_info": None}, {"_plan_info": [42]}],
)
def test_decode_plan_capacity_rejects_other_layouts(decode_native, overrides):
    for name, value in overrides.items():
        setattr(decode_native, name, value)
    assert _flashinfer.decode_plan_capacity(decode_native) is None


@pytest.mark.parametrize("key", ["rtx5090", "rtxpro6000", "gb10", None])
@pytest.mark.parametrize(
    "heads",
    [
        (32, 8, 128),
        (16, 4, 128),
        (16, 4, 256),
        (24, 4, 256),
        (8, 2, 256),
        (12, 2, 256),
        (32, 8, 64),
    ],
)
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5])
def test_xqa_split_returns_table_value_without_sms(monkeypatch, key, heads, batch):
    monkeypatch.setattr(tables, "device_key", lambda _: key)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("no SM lookup"))
    )
    expected = tables.XQA_DECODE_SUBSEQ.get((key, *heads), {}).get(batch)
    assert tables.xqa_decode_subseq(None, *heads, batch) == expected


@pytest.mark.parametrize("batch,expected", [(1, 21), (2, 10), (3, 7), (4, 5)])
def test_xqa_rtx5090_qwen3_split_matches_vllm(monkeypatch, batch, expected):
    monkeypatch.setattr(tables, "device_key", lambda _: "rtx5090")
    assert tables.xqa_decode_subseq(None, 32, 8, 128, batch) == expected == 170 // (batch * 8)


def test_xqa_split_uses_custom_table_count(monkeypatch):
    monkeypatch.setattr(tables, "device_key", lambda _: "test")
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", Mock(side_effect=AssertionError("no SM lookup"))
    )
    monkeypatch.setitem(tables.XQA_DECODE_SUBSEQ, ("test", 32, 8, 128), {4: 3})
    assert tables.xqa_decode_subseq(None, 32, 8, 128, 4) == 3
