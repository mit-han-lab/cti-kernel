# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contracts for fixed launch geometry, scratch sizing, and host validation."""

import builtins
import contextlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from ctikernels.attention import _decode_fp8 as decode


@pytest.fixture(autouse=True)
def hardware(monkeypatch):
    for key in ("SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES", "MERGE"):
        monkeypatch.delenv(f"CTI_FP8DEC_{key}", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=170)
    )


def geometry(**changes):
    return {"batch_size": 2, "num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128} | changes


@pytest.mark.parametrize(
    "length,minimum,limit,expected",
    [
        (8192, 256, 21, (400, 21)),
        (2048, 64, 10, (208, 10)),
        (2048, 256, 10, (256, 8)),
        (5, 1, 7, (16, 1)),
        (17, 1, 7, (16, 2)),
        (257, 256, 21, (144, 2)),
        (0, 1, 7, (16, 0)),
    ],
)
def test_runtime_split_geometry(length, minimum, limit, expected):
    assert decode._split_geometry(length, minimum, limit) == expected


@pytest.mark.parametrize("minimum", [1, 17, 64, 256])
@pytest.mark.parametrize("limit", [1, 7, 10, 21, 170])
def test_runtime_splits_cover_sequence(minimum, limit):
    for length in (*range(1, 514), 2048, 8192, 8193, 2**32 + 17, (2**31 - 1) ** 2):
        chunk, count = decode._split_geometry(length, minimum, limit)
        requested = min(max((length + minimum - 1) // minimum, 1), limit)
        assert 1 <= count <= requested
        assert chunk % 16 == 0
        assert (chunk - 16) * requested < length <= chunk * requested
        intervals = [(split * chunk, min((split + 1) * chunk, length)) for split in range(count)]
        assert intervals[0][0] == 0
        assert intervals[-1][1] == length
        assert all(begin % 16 == 0 and begin < end for begin, end in intervals)
        assert all(
            left[1] == right[0] for left, right in zip(intervals, intervals[1:], strict=False)
        )


def supported(**changes):
    return {
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 16,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.float8_e4m3fn,
    } | changes


@pytest.mark.parametrize("batch,heads,splits", [(1, 1, 170), (2, 8, 10), (7, 4, 6), (1024, 8, 1)])
def test_occupancy_target(batch, heads, splits):
    config = decode.configuration(
        "cuda", **geometry(batch_size=batch, num_qo_heads=heads * 4, num_kv_heads=heads)
    )
    assert config == {
        "splits": splits,
        "min_chunk": 256,
        "warps": 4,
        "keys_per_warp": 16,
        "stages": 3,
        "merge": "distributed",
    }


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("splits", [1, 2, 7, 340])
def test_workspace(monkeypatch, dim, splits):
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", str(splits))
    options = geometry(batch_size=3, num_qo_heads=24, num_kv_heads=4, head_dim=dim)
    size = decode.workspace_size("cuda", **options)
    payload = 3 * 24 * splits * (dim + 1) * 4 if splits > 1 else 0
    assert size == max(256, (payload + 255) // 256 * 256)
    assert size % 256 == 0
    assert size >= payload


def test_overrides(monkeypatch):
    for key, value in {"SPLITS": 3, "MIN_CHUNK": 17, "WARPS": 2, "KEYS": 32, "STAGES": 4}.items():
        monkeypatch.setenv(f"CTI_FP8DEC_{key}", str(value))
    assert decode.configuration("cuda", **geometry()) == {
        "splits": 3,
        "min_chunk": 17,
        "warps": 2,
        "keys_per_warp": 32,
        "stages": 4,
        "merge": "distributed",
    }


def test_configuration_kwargs_and_environment_precedence(monkeypatch):
    overrides = {
        "splits": 7,
        "min_chunk": 32,
        "warps": 2,
        "keys_per_warp": 32,
        "stages": 2,
        "merge": "last",
    }
    assert decode.configuration("cuda", **geometry(), **overrides) == overrides
    assert decode.workspace_size("cuda", **geometry(), **overrides) == decode._workspace_bytes(
        2, 32, 128, 7
    )
    for key, value in {
        "SPLITS": 3,
        "MIN_CHUNK": 17,
        "WARPS": 4,
        "KEYS": 16,
        "STAGES": 3,
        "MERGE": "distributed",
    }.items():
        monkeypatch.setenv(f"CTI_FP8DEC_{key}", str(value))
    assert decode.configuration("cuda", **geometry(), **overrides) == {
        "splits": 3,
        "min_chunk": 17,
        "warps": 4,
        "keys_per_warp": 16,
        "stages": 3,
        "merge": "distributed",
    }
    assert decode.workspace_size("cuda", **geometry(), **overrides) == decode._workspace_bytes(
        2, 32, 128, 3
    )


@pytest.mark.parametrize("environment", [False, True])
def test_constructor_configuration_overrides(monkeypatch, environment):
    overrides = {
        "splits": 7,
        "min_chunk": 32,
        "warps": 2,
        "keys_per_warp": 32,
        "stages": 2,
        "merge": "last",
    }
    if environment:
        monkeypatch.setenv("CTI_FP8DEC_SPLITS", "3")
    expected = overrides | {"splits": 3 if environment else 7}
    workspace = MagicMock()
    workspace.device, workspace.dtype, workspace.ndim = torch.device("cuda:0"), torch.uint8, 1
    workspace.numel.return_value = decode.workspace_size("cuda", **geometry(), **overrides)
    workspace.is_contiguous.return_value = True
    workspace.data_ptr.return_value = 256
    monkeypatch.setattr(decode, "check_support", lambda *a, **k: None)
    compile_stub = Mock(return_value=((Mock(), Mock()), None))
    monkeypatch.setattr(decode, "_compile_resident", compile_stub)
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: zeros(*a, **(k | {"device": "cpu"})))
    op = decode.FP8PagedDecode(
        workspace, device="cuda:0", q_dtype=torch.bfloat16, page_size=16, **geometry(), **overrides
    )
    assert op.config == expected
    assert (op._kernel, op._strided_kernel) == compile_stub.return_value[0]
    assert compile_stub.call_args.args[-1] == expected


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_configuration_kwargs(value):
    with pytest.raises(ValueError, match="CTI_FP8DEC_SPLITS"):
        decode.configuration("cuda", **geometry(), splits=value)


@pytest.mark.parametrize("key", ["SPLITS", "MIN_CHUNK", "WARPS", "KEYS", "STAGES"])
@pytest.mark.parametrize("value", ["0", "-1", "x", "1.5", ""])
def test_invalid_positive_overrides(monkeypatch, key, value):
    monkeypatch.setenv(f"CTI_FP8DEC_{key}", value)
    with pytest.raises(ValueError, match=f"CTI_FP8DEC_{key}"):
        decode.configuration("cuda", **geometry())


@pytest.mark.parametrize(
    "key,value",
    [
        ("WARPS", "3"),
        ("KEYS", "8"),
        ("STAGES", "1"),
        ("STAGES", "5"),
        ("SPLITS", str(2**31)),
        ("MIN_CHUNK", str(2**31)),
    ],
)
def test_invalid_tuning_domain(monkeypatch, key, value):
    monkeypatch.setenv(f"CTI_FP8DEC_{key}", value)
    with pytest.raises(ValueError):
        decode.configuration("cuda", **geometry())


@pytest.mark.parametrize("dim,arena", [(64, 49152), (128, 49152), (256, 98304)])
def test_resource_bound(dim, arena, monkeypatch):
    assert decode._shared_bytes(dim, 4, 16, 3) == arena + 1024
    monkeypatch.setenv("CTI_FP8DEC_WARPS", "8")
    monkeypatch.setenv("CTI_FP8DEC_KEYS", "32")
    with pytest.raises(ValueError, match="shared-memory"):
        decode.configuration("cuda", **geometry(head_dim=dim))


@pytest.mark.parametrize(
    "warps,keys,page_size,metadata",
    [
        (4, 32, None, 1032),
        (4, 32, 1, 1024),
        (8, 32, None, 2056),
        (8, 32, 1, 2048),
        (8, 32, 2, 1032),
        (8, 32, 3, 1024),
        (8, 32, 16, 1024),
    ],
)
@pytest.mark.parametrize("splits", [1, 10000])
def test_page_cache_and_merge_storage(warps, keys, page_size, metadata, splits):
    arena = max(warps * 2 * 2 * keys * 128, splits * 4 * 4)
    assert (
        decode._shared_bytes(128, warps, keys, 2, splits, 4, page_size=page_size)
        == arena + metadata
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"batch_size": 65536},
        {"num_kv_heads": 0},
        {"num_qo_heads": 0},
        {"num_qo_heads": 33},
        {"num_qo_heads": 136},
        {"num_kv_heads": 65536, "num_qo_heads": 65536},
        {"head_dim": 80},
    ],
)
def test_geometry_rejections(changes):
    with pytest.raises(NotImplementedError):
        decode.configuration("cuda", **geometry(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"page_size": 0},
        {"page_size": 1.5},
        {"head_dim": 192},
        {"q_dtype": torch.float32},
        {"kv_dtype": torch.float8_e5m2},
        {"num_qo_heads": 129},
        {"num_kv_heads": 0},
    ],
)
def test_support_rejections(changes):
    with pytest.raises(NotImplementedError):
        decode.check_support("cuda", **supported(**changes))


@pytest.mark.parametrize(
    "device,capability", [("cpu", (12, 0)), ("cuda", (9, 0)), ("cuda", (12, 2))]
)
def test_device_rejections(monkeypatch, device, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: capability)
    with pytest.raises(NotImplementedError, match="sm_120"):
        decode.check_support(device, **supported())


@pytest.mark.parametrize("minor", [0, 1])
@pytest.mark.parametrize("group", range(1, 17))
def test_supported_without_cuda(monkeypatch, minor, group):
    # Only the import is needed by check_support; never initialize CUDA here.
    fake = ModuleType("cutlass")
    fake.cute = ModuleType("cutlass.cute")
    fake.runtime = ModuleType("cutlass.runtime")
    fake.runtime.make_fake_stream = object()
    monkeypatch.setitem(sys.modules, "cutlass", fake)
    monkeypatch.setitem(sys.modules, "cutlass.cute", fake.cute)
    monkeypatch.setitem(sys.modules, "cutlass.runtime", fake.runtime)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    decode.check_support("cuda", **supported(num_qo_heads=group * 3, num_kv_heads=3, page_size=7))


def test_missing_dsl(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "cutlass.cute":
            raise ImportError("test missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(NotImplementedError, match="cutlass-dsl"):
        decode.check_support("cuda", **supported())


def cpu_operation():
    # Bypass only construction's GPU allocation/compilation to exercise the
    # real run validation on CPU tensor metadata and a recording launch stub.
    op = decode.FP8PagedDecode.__new__(decode.FP8PagedDecode)
    op.batch, op.hq, op.hk, op.dim, op.page = 2, 8, 2, 64, 7
    op.dtype, op.device, op.sm_scale = torch.float16, torch.device("cpu"), 0.125
    op._work, op._counters, op._kernel = object(), object(), Mock()
    op._strided_kernel = Mock()
    q = torch.empty(2, 24, 64, dtype=torch.float16)[:, :8]
    k = torch.empty(5, 7, 2, 64, dtype=torch.float8_e4m3fn)
    ptr, idx, last = (torch.empty(n, dtype=torch.int32) for n in (3, 10, 2))
    return op, [q, k, k, ptr, idx, last, torch.empty(2, 8, 64, dtype=torch.float16)]


def test_strided_launch_and_runtime_scales(monkeypatch):
    op, args = cpu_operation()
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    assert op.run(*args, k_scale=0.5, v_scale=0.0) is args[-1]
    call = op._kernel.call_args.args
    assert all(a is b for a, b in zip(call[:7], args, strict=True))
    assert call[-3:] == (0.125, 0.5, 0.0)


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, lambda x: x[:1]),
        (0, lambda x: x.float()),
        (0, lambda x: torch.empty(2, 16, 64, dtype=x.dtype)[:, ::2]),
        (0, lambda x: torch.empty(2, 8, 65, dtype=x.dtype)[:, :, :64]),
        (1, lambda x: x.float()),
        (2, lambda x: x[:4]),
        (3, lambda x: x.to(torch.int64)),
        (4, lambda x: x[::2]),
        (5, lambda x: x[:1]),
        (6, lambda x: x.transpose(1, 2)),
    ],
)
def test_run_rejects_invalid_metadata(index, replacement):
    op, args = cpu_operation()
    args[index] = replacement(args[index])
    with pytest.raises(ValueError):
        op.run(*args)
    op._kernel.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e40, torch.tensor(1.0), "1"])
def test_scales_are_host_fp32(value):
    with pytest.raises(ValueError, match="host float"):
        decode._scale(value, "test")


def test_benchmark_hook_retains_graph_metadata(monkeypatch):
    from benchmark.kernel_tuning import paged_attention_range_market as market

    constructor = Mock()
    monkeypatch.setattr(decode, "FP8PagedDecode", constructor)
    monkeypatch.setattr(decode, "workspace_size", lambda *args, **kwargs: 256)
    row = {"op": "fp8-decode", "model": "qwen3-8b", "B": 2, "KV": 257, "Q": 1}
    buffers = [
        (torch.empty(1, dtype=torch.bfloat16), object(), object(), object()) for _ in range(2)
    ]
    call, retained = market.make_cute_fp8_call(row, buffers, "cpu")
    _, workspace, host, gpu = retained
    assert workspace.dtype == torch.uint8
    assert gpu["indices"].numel() == 2 * (
        (market.MODEL_MAX_LEN[row["model"]] + market.PAGE - 1) // market.PAGE
    )
    torch.testing.assert_close(gpu["indices"][: host["indices"].numel()], host["indices"])
    call()
    op = constructor.return_value
    assert op.run.call_count == 2
    for recorded, (q, k, v, out) in zip(op.run.call_args_list, buffers, strict=True):
        assert all(
            left is right
            for left, right in zip(
                recorded.args, (q, k, v, gpu["k"], gpu["indices"], gpu["last"], out), strict=True
            )
        )
        assert recorded.kwargs == {"k_scale": market.K_SCALE, "v_scale": market.V_SCALE}
    assert "cute_fp8" in market._provider_names(row)
    assert "cute_fp8" not in market._provider_names(row | {"op": "graph-prefill"})


@pytest.mark.parametrize("mode", ["distributed", "last"])
def test_merge_modes(monkeypatch, mode):
    monkeypatch.setenv("CTI_FP8DEC_MERGE", mode)
    assert decode.configuration("cuda", **geometry())["merge"] == mode


def test_invalid_merge(monkeypatch):
    monkeypatch.setenv("CTI_FP8DEC_MERGE", "serial")
    with pytest.raises(ValueError, match="CTI_FP8DEC_MERGE"):
        decode.configuration("cuda", **geometry())


@pytest.mark.parametrize("sms", [170, 188, 48])
@pytest.mark.parametrize("dim,resident", [(128, 2), (256, 1)])
def test_construction_residency(monkeypatch, sms, dim, resident):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            multi_processor_count=sms,
            shared_memory_per_multiprocessor=100 * 1024,
            max_threads_per_multi_processor=1536,
        ),
    )
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "10000")
    config = decode.configuration("cuda", **geometry(batch_size=1, head_dim=dim))
    compiled_splits = []

    def compile_stub(*args, **kwargs):
        compiled_splits.append((kwargs["splits"], kwargs["kv_dense"]))
        return object()

    monkeypatch.setattr(decode, "_compile", compile_stub)
    monkeypatch.setattr(decode._cooperative, "supported", lambda _: True)
    occupancy = Mock(return_value=resident)
    monkeypatch.setattr(decode._cooperative, "blocks_per_sm", occupancy)
    _, blocks = decode._compile_resident(
        torch.device("cuda:0"), 1, 8, 4, dim, 16, torch.bfloat16, "sm_120a", config
    )
    assert blocks == resident
    assert config["splits"] == sms * resident // 8
    assert compiled_splits == [(config["splits"], True), (config["splits"], False)]
    assert occupancy.call_count == 2
    assert config["splits"] * 8 <= sms * blocks
    assert occupancy.call_args.args[-1] == decode._shared_bytes(dim, 4, 16, 3, page_size=16) + 1024


def test_residency_rechecked_after_recompile(monkeypatch):
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "1000")
    config = decode.configuration("cuda", **geometry())
    compiled = []

    def compile_stub(*args, **kwargs):
        compiled.append((kwargs["splits"], kwargs["kv_dense"]))
        return object()

    monkeypatch.setattr(decode, "_compile", compile_stub)
    monkeypatch.setattr(decode._cooperative, "supported", lambda _: True)
    # The strided binary is the tighter occupancy bound; both are rechecked.
    occupancy = Mock(side_effect=[2, 1, 1, 2])
    monkeypatch.setattr(decode._cooperative, "blocks_per_sm", occupancy)
    decode._compile_resident(
        torch.device("cuda:0"), 2, 8, 4, 128, 16, torch.bfloat16, "sm_120a", config
    )
    assert compiled == [(21, True), (21, False), (10, True), (10, False)]
    assert occupancy.call_count == 4
    assert config["splits"] == 10


def test_distributed_merge_budget_includes_metadata(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(multi_processor_count=10000)
    )
    config = decode.configuration("cuda", **geometry(), splits=10000)
    compile_stub = Mock(return_value=object())
    monkeypatch.setattr(decode, "_compile", compile_stub)
    monkeypatch.setattr(decode._cooperative, "supported", lambda _: True)
    occupancy = Mock(return_value=1)
    monkeypatch.setattr(decode._cooperative, "blocks_per_sm", occupancy)
    decode._compile_resident(
        torch.device("cuda:0"), 1, 1, 4, 128, 16, torch.bfloat16, "sm_120a", config
    )
    assert config["splits"] == (99 * 1024 - 1024 - 1024) // 16
    assert occupancy.call_args.args[-1] == 99 * 1024
    assert compile_stub.call_args.kwargs["splits"] == config["splits"]


@pytest.mark.parametrize("merge", ["distributed", "last"])
def test_no_cooperative_requirement_for_independent_grid(monkeypatch, merge):
    config = decode.configuration("cuda", **geometry(batch_size=1024))
    config["merge"] = merge
    compile_stub = Mock(return_value=object())
    monkeypatch.setattr(decode, "_compile", compile_stub)
    support = Mock(side_effect=AssertionError("unneeded support query"))
    occupancy = Mock(side_effect=AssertionError("unneeded occupancy query"))
    monkeypatch.setattr(decode._cooperative, "supported", support)
    monkeypatch.setattr(decode._cooperative, "blocks_per_sm", occupancy)
    decode._compile_resident(
        torch.device("cuda:0"), 1024, 8, 4, 128, 16, torch.bfloat16, "sm_120a", config
    )
    assert config["splits"] == 1
    support.assert_not_called()
    occupancy.assert_not_called()


def test_unsupported_cooperative_launch(monkeypatch):
    config = decode.configuration("cuda", **geometry())
    monkeypatch.setattr(decode._cooperative, "supported", lambda _: False)
    with pytest.raises(NotImplementedError, match="CTI_FP8DEC_MERGE=last"):
        decode._compile_resident(
            torch.device("cuda:0"), 2, 8, 4, 128, 16, torch.bfloat16, "sm_120a", config
        )


def test_last_arrival_preserves_oversubscribed_override(monkeypatch):
    monkeypatch.setenv("CTI_FP8DEC_MERGE", "last")
    monkeypatch.setenv("CTI_FP8DEC_SPLITS", "1000")
    config = decode.configuration("cuda", **geometry())
    compile_stub = Mock(return_value=object())
    monkeypatch.setattr(decode, "_compile", compile_stub)
    query = Mock(side_effect=AssertionError("last-arrival must not require co-residency"))
    monkeypatch.setattr(decode._cooperative, "supported", query)
    monkeypatch.setattr(decode._cooperative, "blocks_per_sm", query)
    decode._compile_resident(
        torch.device("cuda:0"), 2, 8, 4, 128, 16, torch.bfloat16, "sm_120a", config
    )
    assert config["splits"] == compile_stub.call_args.kwargs["splits"] == 1000
    query.assert_not_called()


@pytest.mark.parametrize("layout", ["fused", "broadcast"])
@pytest.mark.parametrize(
    "packed_k,packed_v", [(False, False), (True, False), (False, True), (True, True)]
)
def test_runtime_query_and_independent_page_strides(monkeypatch, layout, packed_k, packed_v):
    op, args = cpu_operation()
    q = torch.empty(2, 24, 64, dtype=op.dtype)
    args[0] = {"fused": q[:, :8], "broadcast": q[:1, :8].expand(2, -1, -1)}[layout]
    if packed_k:
        args[1] = torch.empty(5, 2, 7, 2, 64, dtype=torch.float8_e4m3fn)[:, 0]
    if packed_v:
        args[2] = torch.empty(5, 3, 7, 2, 64, dtype=torch.float8_e4m3fn)[:, 1]
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    assert op.run(*args) is args[-1]
    selected = op._strided_kernel if packed_k or packed_v else op._kernel
    unused = op._kernel if packed_k or packed_v else op._strided_kernel
    assert all(a is b for a, b in zip(selected.call_args.args[:7], args, strict=True))
    unused.assert_not_called()


def test_e4m3_bf16_embedding_exact_for_finite_codes():
    codes = torch.arange(256, dtype=torch.int32)
    expected = codes.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
    bits = ((codes & 0x7F) << 4) | ((codes & 0x80) << 8)
    embedded = bits.to(torch.int16).view(torch.bfloat16)
    actual = (embedded.double() * 2.0**120).to(torch.bfloat16)
    finite = torch.isfinite(expected)
    assert torch.equal(actual.view(torch.int16)[finite], expected.view(torch.int16)[finite])


def test_packed_e4m3_bf16_bit_mapping():
    # Model prmt's sign-extension selectors for every possible byte pair.
    words = torch.arange(65536, dtype=torch.int64)
    lo, hi = words & 0xFF, words >> 8
    permuted = lo | ((lo >> 7) * 0xFF << 8) | (hi << 16) | ((hi >> 7) * 0xFF << 24)
    actual = (permuted << 4) & 0x87F087F0
    expected_lo = ((lo & 0x7F) << 4) | ((lo & 0x80) << 8)
    expected_hi = ((hi & 0x7F) << 4) | ((hi & 0x80) << 8)
    assert torch.equal(actual, expected_lo | (expected_hi << 16))


@pytest.mark.parametrize("layout", ["heads", "padded", "mixed", "packed"])
def test_aligned_kv_views(monkeypatch, layout):
    from ..gpu._attention import kv_views

    op, args = cpu_operation()
    args[1:3] = kv_views((args[1], args[2]), layout)
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocation")))
    assert op.run(*args) is args[-1]
    kernel = op._strided_kernel
    kernel.assert_called_once()
    assert kernel.call_args.args[1] is args[1]
    assert kernel.call_args.args[2] is args[2]


@pytest.mark.parametrize("layout", ["base", "page", "token", "head", "inner"])
@pytest.mark.parametrize("index", [1, 2])
def test_unaligned_kv_rejected_before_launch(index, layout):
    op, args = cpu_operation()
    cache = args[index]
    if layout == "base":
        cache = torch.empty(cache.numel() + 1, dtype=cache.dtype)[1:].view(cache.shape)
    elif layout == "inner":
        cache = torch.empty(*cache.shape[:-1], 2 * cache.shape[-1], dtype=cache.dtype)[..., ::2]
    else:
        strides = list(cache.stride())
        strides[{"page": 0, "token": 1, "head": 2}[layout]] += 1
        cache = torch.empty_strided(cache.shape, strides, dtype=cache.dtype)
    args[index] = cache
    with pytest.raises(ValueError, match="KV requires"):
        op.run(*args)
    op._kernel.assert_not_called()
    op._strided_kernel.assert_not_called()
