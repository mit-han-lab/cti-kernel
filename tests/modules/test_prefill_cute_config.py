# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU contracts for the fixed-bound prefill provider and graph benchmark hook."""

import builtins
import contextlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _prefill_cute as prefill


@pytest.fixture(autouse=True)
def configuration_environment(monkeypatch):
    for key in ("BLOCK_M", "BLOCK_N", "STAGES", "SPLITS", "MIN_CHUNK"):
        monkeypatch.delenv("CTI_PREFILL_" + key, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: Mock(multi_processor_count=170)
    )


def geometry(**changes):
    return {
        "batch_size": 3,
        "max_query_per_request": 100,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
    } | changes


def support(**changes):
    return {
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 16,
        "dtype": torch.bfloat16,
    } | changes


@pytest.mark.parametrize(
    "dim,rows,keys,splits", [(64, 64, 64, 3), (128, 64, 64, 3), (256, 64, 64, 2)]
)
def test_defaults_are_host_static(monkeypatch, dim, rows, keys, splits):
    monkeypatch.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("host tensor read")))
    assert prefill.configuration("cuda", **geometry(head_dim=dim)) == {
        "block_m": rows,
        "block_n": keys,
        "stages": 2,
        "splits": splits,
        "min_chunk": 256 if dim == 256 else 64,
    }


@pytest.mark.parametrize("minor,sms,rows", [(0, 170, 64), (0, 188, 64), (1, 48, 128)])
@pytest.mark.parametrize("batch,query", [(3, 64), (1, 256), (64, 256)])
@pytest.mark.parametrize("dim", [128, 256])
def test_device_defaults_and_grid_target(monkeypatch, minor, sms, rows, batch, query, dim):
    rows = 64 if dim == 256 else rows
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: Mock(multi_processor_count=sms)
    )
    cfg = prefill.configuration(
        "cuda", **geometry(batch_size=batch, max_query_per_request=query, head_dim=dim)
    )
    assert cfg["block_m"] == rows
    grid = batch * 8 * ((query * 4 + rows - 1) // rows)
    target = sms * (1 if dim == 256 else 2)
    assert cfg["splits"] == max(1, (target + grid - 1) // grid)
    shared = prefill._shared_bytes(dim, rows, cfg["block_n"], cfg["stages"]) + 1024
    # 48 KiB D=128 arena plus metadata fits twice in a 100 KiB carveout.
    assert shared * (2 if dim == 128 else 1) <= 100 * 1024
    monkeypatch.setenv("CTI_PREFILL_SPLITS", "1")
    assert prefill.configuration("cuda", **geometry())["splits"] == 1


@pytest.mark.parametrize("batch,query,splits", [(1, 64, 3), (3, 64, 1), (2, 256, 1), (4, 256, 1)])
def test_d256_gb10_screen_defaults(monkeypatch, batch, query, splits):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 1))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: Mock(multi_processor_count=48)
    )
    assert prefill.configuration(
        "cuda",
        **geometry(
            batch_size=batch,
            max_query_per_request=query,
            num_qo_heads=16,
            num_kv_heads=4,
            head_dim=256,
        ),
    ) == {"block_m": 64, "block_n": 64, "stages": 2, "splits": splits, "min_chunk": 256}


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("rows", [64, 128])
@pytest.mark.parametrize("splits", [1, 2, 7])
def test_workspace_includes_padded_tiles_and_counters(monkeypatch, dim, rows, splits):
    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", str(rows))
    monkeypatch.setenv("CTI_PREFILL_SPLITS", str(splits))
    count = 3 * 8 * ((100 * 4 + rows - 1) // rows)
    payload = count * splits * rows * (dim + 1) * 4 if splits > 1 else 0
    counters = (payload + 255) // 256 * 256
    size = counters + (count * 4 if splits > 1 else 0)
    assert prefill.workspace_size("cuda", **geometry(head_dim=dim)) == max(
        256, (size + 255) // 256 * 256
    )


def test_overrides(monkeypatch):
    values = {"BLOCK_M": 128, "BLOCK_N": 32, "STAGES": 3, "SPLITS": 5, "MIN_CHUNK": 17}
    for key, value in values.items():
        monkeypatch.setenv("CTI_PREFILL_" + key, str(value))
    assert prefill.configuration("cuda", **geometry()) == {
        key.lower(): value for key, value in values.items()
    }


@pytest.mark.parametrize("key", ["BLOCK_M", "BLOCK_N", "STAGES", "SPLITS", "MIN_CHUNK"])
@pytest.mark.parametrize("value", ["0", "-1", "bad", "1.5", "", str(2**31)])
def test_bad_env(monkeypatch, key, value):
    monkeypatch.setenv("CTI_PREFILL_" + key, value)
    with pytest.raises(ValueError, match="CTI_PREFILL_" + key):
        prefill.configuration("cuda", **geometry())


@pytest.mark.parametrize("key,value", [("BLOCK_M", "32"), ("BLOCK_N", "16"), ("STAGES", "4")])
def test_bad_choices(monkeypatch, key, value):
    monkeypatch.setenv("CTI_PREFILL_" + key, value)
    with pytest.raises(ValueError):
        prefill.configuration("cuda", **geometry())


def test_resource_limit(monkeypatch):
    monkeypatch.setenv("CTI_PREFILL_BLOCK_N", "64")
    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", "128")
    with pytest.raises(ValueError, match="shared-memory"):
        prefill.configuration("cuda", **geometry(head_dim=256))


def test_persistent_q_three_stage_budget(monkeypatch):
    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", "128")
    monkeypatch.setenv("CTI_PREFILL_STAGES", "3")
    with pytest.raises(ValueError, match="shared-memory"):
        prefill.configuration("cuda", **geometry(head_dim=256))
    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", "64")
    assert prefill.configuration("cuda", **geometry(head_dim=256))["stages"] == 3


def test_split_grid_limit(monkeypatch):
    monkeypatch.setenv("CTI_PREFILL_SPLITS", str(2**30))
    with pytest.raises(ValueError, match="grid"):
        prefill.configuration("cuda", **geometry())


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"batch_size": 65536},
        {"max_query_per_request": 0},
        {"max_query_per_request": True},
        {"num_qo_heads": 33},
        {"num_kv_heads": 0},
        {"head_dim": 96},
        {"max_query_per_request": 2**31},
    ],
)
def test_geometry_rejections(changes):
    with pytest.raises(NotImplementedError):
        prefill.configuration("cuda", **geometry(**changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"page_size": 0},
        {"page_size": 1.5},
        {"page_size": 2**31},
        {"dtype": torch.float32},
        {"num_qo_heads": 33},
    ],
)
def test_unsupported_geometry(changes):
    with pytest.raises(NotImplementedError):
        prefill.check_support("cuda", **support(**changes))


@pytest.mark.parametrize("device,cap", [("cpu", (12, 0)), ("cuda", (9, 0)), ("cuda", (12, 2))])
def test_unsupported_device(monkeypatch, device, cap):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: cap)
    with pytest.raises(NotImplementedError, match="sm_120"):
        prefill.check_support(device, **support())


@pytest.mark.parametrize("group", [1, 3, 4, 9, 17, 33])
@pytest.mark.parametrize("minor", [0, 1])
def test_supported_without_cuda(monkeypatch, group, minor):
    fake = ModuleType("cutlass")
    fake.cute, fake.runtime = ModuleType("cutlass.cute"), ModuleType("cutlass.runtime")
    fake.runtime.make_fake_stream = object()
    for name, module in (
        ("cutlass", fake),
        ("cutlass.cute", fake.cute),
        ("cutlass.runtime", fake.runtime),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, minor))
    prefill.check_support("cuda", **support(num_qo_heads=group * 3, num_kv_heads=3, page_size=7))


def test_missing_dsl(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "cutlass.cute":
            raise ImportError("optional dependency absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(NotImplementedError, match="cutlass-dsl"):
        prefill.check_support("cuda", **support())


def cpu_operation():
    op = prefill.CutePagedPrefill.__new__(prefill.CutePagedPrefill)
    op.batch, op.qmax, op.hq, op.hk, op.dim, op.page = 2, 16, 8, 2, 64, 7
    op.q_row_stride = 24 * 64
    op.device, op.dtype, op.sm_scale = torch.device("cpu"), torch.float16, 0.125
    op._kernel, op._work, op._counters = Mock(), object(), object()
    op._strided_kernel = Mock()
    q = torch.empty(32, 24, 64, dtype=op.dtype)[:, :8]
    cache = torch.empty(11, 7, 2, 64, dtype=op.dtype)
    qp, kp, ids, last = [torch.empty(n, dtype=torch.int32) for n in (3, 3, 22, 2)]
    return op, [q, cache, cache, qp, kp, ids, last, torch.empty(32, 8, 64, dtype=op.dtype)]


@pytest.mark.parametrize("rows", [0, 3, 32])
def test_run_only_passes_buffers(monkeypatch, rows):
    op, args = cpu_operation()
    args[0], args[-1] = args[0][:rows], args[-1][:rows]
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("host read")))
    monkeypatch.setattr(torch.cuda, "synchronize", Mock(side_effect=AssertionError("sync")))
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocation")))
    assert op.run(*args) is args[-1]
    recorded = op._kernel.call_args.args
    assert all(left is right for left, right in zip(recorded[:8], args, strict=True))
    assert recorded[8:] == (op._work, op._counters, 0.125, 1.0)


@pytest.mark.parametrize(
    "index,change",
    [
        (0, lambda x: x.float()),
        (0, lambda x: torch.empty(32, 8, 65, dtype=x.dtype)[:, :, :64]),
        (0, lambda x: x[:, :, :32]),
        (1, lambda x: x.float()),
        (2, lambda x: x[:3]),
        (3, lambda x: x.to(torch.int64)),
        (4, lambda x: x[:2]),
        (5, lambda x: x[::2]),
        (6, lambda x: x[:1]),
        (7, lambda x: x.transpose(1, 2)),
    ],
)
def test_run_metadata_rejection(index, change):
    op, args = cpu_operation()
    args[index] = change(args[index])
    with pytest.raises(ValueError):
        op.run(*args)
    op._kernel.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"sm_scale": float("inf")},
        {"sm_scale": float("nan")},
        {"sm_scale": 1e40},
        {"sm_scale": torch.tensor(1.0)},
        {"q_row_stride": 1},
        {"q_row_stride": 4097},
        {"q_row_stride": 4096.0},
    ],
)
def test_construction_rejects_invalid_host_options(monkeypatch, changes):
    monkeypatch.setattr(prefill, "check_support", Mock())
    with pytest.raises(ValueError):
        prefill.CutePagedPrefill(
            None, **geometry(), page_size=16, dtype=torch.bfloat16, device="cuda:0", **changes
        )


def test_benchmark_hook(monkeypatch):
    from benchmark.kernel_tuning import paged_attention_range_market as market

    constructor = Mock()
    monkeypatch.setattr(prefill, "CutePagedPrefill", constructor)
    monkeypatch.setattr(prefill, "workspace_size", lambda *args, **kwargs: 256)
    row = {"op": "graph-prefill", "model": "qwen3-8b", "B": 3, "Q": 64, "KV": 2048}
    buffers = [
        (torch.empty(192, 32, 128, dtype=torch.bfloat16), object(), object(), object())
        for _ in range(2)
    ]
    call, retained = market.make_cute_prefill_call(row, buffers, "cpu")
    _, workspace, host, gpu = retained
    assert workspace.numel() == 256
    assert constructor.call_args.kwargs["max_query_per_request"] == 64
    torch.testing.assert_close(host["q"], gpu["q"])
    call()
    assert constructor.return_value.run.call_count == 2
    for recorded, (q, k, v, out) in zip(
        constructor.return_value.run.call_args_list, buffers, strict=True
    ):
        expected = (q, k, v, gpu["q"], gpu["k"], gpu["indices"], gpu["last"], out)
        assert all(left is right for left, right in zip(recorded.args, expected, strict=True))
    assert "cute_prefill" in market._provider_names(row)
    assert "cute_prefill" not in market._provider_names(row | {"op": "fp8-decode"})
    monkeypatch.setattr(market, "DRAFT_VARIANTS", ("cp_m64_s1",))
    assert "cp_m64_s1" in market._provider_names(row)
    assert "cp_m64_s1" not in market._provider_names(row | {"op": "fp8-decode"})


def test_variant_scope_and_restore(monkeypatch):
    from benchmark.kernel_tuning import paged_attention_range_market as market

    expected = {
        "CTI_PREFILL_BLOCK_M": "128",
        "CTI_PREFILL_BLOCK_N": "32",
        "CTI_PREFILL_STAGES": "3",
        "CTI_PREFILL_SPLITS": "2",
        "CTI_PREFILL_MIN_CHUNK": "17",
    }
    assert market.prefill_variant_env("cp_m128_n32_st3_s2_c17") == expected
    import os

    monkeypatch.setenv("CTI_PREFILL_BLOCK_M", "64")

    def maker(*args):
        assert all(os.environ[key] == value for key, value in expected.items())
        raise ValueError("construction failed")

    monkeypatch.setattr(market, "make_cute_prefill_call", maker)
    with pytest.raises(ValueError, match="construction failed"):
        market.make_draft_variant_call({}, [], "cpu", "cp_m128_n32_st3_s2_c17")
    assert os.environ["CTI_PREFILL_BLOCK_M"] == "64"
    assert "CTI_PREFILL_SPLITS" not in os.environ


@pytest.mark.parametrize("layout", ["dense", "fused", "heads", "broadcast"])
def test_runtime_query_and_independent_page_strides(monkeypatch, layout):
    op, args = cpu_operation()
    q = torch.empty(32, 24, 64, dtype=op.dtype)
    args[0] = {
        "dense": q[:, :8].contiguous(),
        "fused": q[:, :8],
        "heads": q[:, ::3],
        "broadcast": q[:1, :8].expand(32, -1, -1),
    }[layout]
    args[1] = torch.empty(11, 2, 7, 2, 64, dtype=op.dtype)[:, 0]
    args[2] = torch.empty(11, 3, 7, 2, 64, dtype=op.dtype)[:, 1]
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    for k_scale, v_scale in ((0.37, 1.9), (1.0, 0.0), (-0.5, 1.0), (0.0, 2.0)):
        assert op.run(*args, k_scale=k_scale, v_scale=v_scale) is args[-1]
        assert op._kernel.call_args.args[-2:] == (0.125 * k_scale, v_scale)


@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_runtime_scales_reject_nonfinite(name, value):
    op, args = cpu_operation()
    with pytest.raises(ValueError, match=name):
        op.run(*args, **{name: value})
    op._kernel.assert_not_called()


@pytest.mark.parametrize("layout", ["heads", "padded", "mixed", "packed"])
def test_aligned_kv_views(monkeypatch, layout):
    from ..gpu._attention import kv_views

    op, args = cpu_operation()
    args[1:3] = kv_views((args[1], args[2]), layout)
    monkeypatch.setattr(torch.cuda, "device", lambda _: contextlib.nullcontext())
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocation")))
    assert op.run(*args) is args[-1]
    kernel = op._kernel if layout == "packed" else op._strided_kernel
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
