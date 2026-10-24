# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU route address proofs, dispatch tests and GPU-free CuTe compilation."""

import importlib.metadata
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels import _moe_block_fp8_triton_table as table
from ctikernels.moe_block_fp8 import BlockFP8MoE


@pytest.mark.parametrize("routes", [1, 8, 24, 32, 64, 65])
@pytest.mark.parametrize("bm", [16, 64])
@pytest.mark.parametrize("group_m", [1, 8])
def test_route_layout_addresses_and_cache(routes, bm, group_m):
    op = object.__new__(BlockFP8MoE)
    op._route_blocks = {}
    ids = (torch.arange(routes, dtype=torch.int32) % 4).view(-1, 1)
    ids[::3] = -1
    sorted_ids, experts, padded = op._route_layout(ids, bm)
    assert sorted_ids.dtype == experts.dtype == padded.dtype == torch.int32
    assert sorted_ids.is_contiguous() and experts.is_contiguous()
    assert experts.data_ptr() == ids.data_ptr()
    assert padded.tolist() == [routes * bm]
    assert torch.equal(sorted_ids.view(routes, bm)[:, 0], torch.arange(routes))
    assert (sorted_ids.view(routes, bm)[:, 1:] == routes).all()
    # Model the unchanged Triton kernel's grouped grid, including a partial
    # final group and a partial N tile. Every valid (route,column) is written once.
    n, bn = 128, 48
    num_n = -(-n // bn)
    writes = torch.zeros(routes, n, dtype=torch.int32)
    mapping = torch.tensor([-1, 1, -1, 0], dtype=torch.int32)
    for pid in range(routes * num_n):
        first_m = (pid // (group_m * num_n)) * group_m
        size_m = min(routes - first_m, group_m)
        pid_m = first_m + ((pid % (group_m * num_n)) % size_m)
        pid_n = (pid % (group_m * num_n)) // size_m
        assert pid_m * bm < padded.item()
        offsets = pid_m * bm + torch.arange(bm)
        assert offsets.min() >= 0 and offsets.max() < sorted_ids.numel()
        tokens = sorted_ids[offsets]
        assert 0 <= pid_m < experts.numel()
        expert = experts[pid_m].item()
        local = mapping[expert].item() if expert >= 0 else -1
        assert local in (-1, 0, 1)  # -1 takes REMOTE before forming weight addresses.
        live = tokens[tokens < routes].long()
        assert live.tolist() == [pid_m]
        for top_k in (1, 8):
            # FC1 indexes source row route//top_k; FC2 has top_k=1.
            assert (live // top_k).max() < -(-routes // top_k)
        assert live.max() < ids.numel()  # FC2 routing-weight load.
        writes[live, pid_n * bn : min((pid_n + 1) * bn, n)] += 1
    assert (writes == 1).all()
    again, new_experts, same_padded = op._route_layout(ids.clone(), bm)
    assert again is sorted_ids and same_padded is padded
    assert new_experts.data_ptr() != experts.data_ptr()
    new_experts.fill_(-1)
    assert (experts >= 0).any() or routes == 1
    op._route_layout(ids, bm * 2)
    assert len(op._route_blocks) == 2


def test_route_policy_overrides(monkeypatch):
    device = (12, 0, 170)
    for m, expected in ((1, True), (8, True), (9, False)):
        assert all(c["route_blocks"] == expected for c in table.selection(device, m, 256))
    fc1, fc2 = table.selection(device, 1, 256)
    for enabled in (True, False):
        row = tuple(dict(c, route_blocks=enabled) for c in (fc1, fc2))
        monkeypatch.setitem(table.TABLE, device, {(256, 2048, 512, 8): [(4096, *row)]})
        assert table.selection(device, 9, 256) == row
    fc2["route_blocks"] = False
    monkeypatch.setitem(table.TABLE, device, {(256, 2048, 512, 8): [(4096, fc1, fc2)]})
    with pytest.raises(ValueError, match="share route_blocks"):
        table.selection(device, 1, 256)
    # A default BM=64 config does not enable route blocks even below 64 routes.
    assert not table.selection((99, 0, 1), 4, 1)[0]["route_blocks"]


@pytest.mark.parametrize("available,enabled", [(True, True), (False, True), (True, False)])
def test_activation_dispatch(available, enabled):
    op = object.__new__(BlockFP8MoE)
    op.device_class, op.provider = (12, 0, 170), "triton-fp8"
    op._available, op._enabled = available, enabled
    op.e, op.h, op.i, op.top_k = 256, 2048, 512, 8
    configs = op.config_for(4)
    assert configs[1]["act"] == (
        "cute-silu-quant" if available and enabled else "triton-silu-quant"
    )
    assert all(c["route_blocks"] for c in configs)


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("route_blocks", [False, True])
@pytest.mark.parametrize("act", ["split", "cute-silu-quant", "triton-silu-quant"])
def test_triton_chain_layout_and_launches(route_blocks, act, mapped, monkeypatch):
    from ctikernels import moe_block_fp8, moe_impl
    from ctikernels.activation import inference
    from ctikernels.linear import block_fp8

    op = object.__new__(BlockFP8MoE)
    op.i, op.h, op.top_k, op.num_global_experts = 128, 256, 8, 4
    op._route_blocks = {}
    op.expert_map = torch.tensor([-1, 1, -1, 0], dtype=torch.int32) if mapped else None
    op.w13 = op.w2 = op.s13 = op.s2 = None
    x = torch.zeros(3, op.h, dtype=torch.bfloat16)
    ids = torch.arange(24, dtype=torch.int32).reshape(3, op.top_k) % 4
    weights = torch.ones_like(ids, dtype=torch.float32)
    configs = table.selection((12, 0, 170), 3, 256)
    for c in configs:
        c["route_blocks"] = route_blocks
    configs[1]["act"] = act
    calls, layouts = [], []

    def align(ids, bm, experts):
        calls.append("align")
        assert experts == 4
        return op._route_layout(ids, bm)

    def quantize(source):
        calls.append("quantize")
        return (
            torch.empty(source.shape, dtype=torch.float8_e4m3fn),
            torch.empty(source.shape[0], source.shape[1] // 128),
        )

    def silu(source, out):
        calls.append("silu")
        out.zero_()

    def fused(source, q, s, act_out=None, *, name="fused"):
        calls.append(name)
        assert source.shape == (24, 256) and act_out is None
        assert q.shape == (24, 128) and s.shape == (24, 1)

    def triton_fused(source, *, outputs, act):
        fused(source, *outputs, act, name="triton-fused")

    def gemm(
        q, w, out, weights, ids, sorted_ids, experts, padded, routed, top_k, config, dtype, **kw
    ):
        second = q.shape == (24, 128)  # FC2 reads the routed activation rows
        calls.append("fc2" if second else "fc1")
        assert not routed  # FC2 rows stay unweighted: the reduction weights them
        assert "act" not in config and "route_blocks" not in config
        assert kw["expert_map"] is op.expert_map
        assert kw["num_global_experts"] == 4
        assert top_k == (1 if second else op.top_k)
        assert q.shape == ((24, 128) if second else (3, 256))
        assert experts.data_ptr() == ids.data_ptr()
        layouts.append((sorted_ids, experts, padded))
        out.zero_()

    def reduce(down, out, *, topk_ids, expert_map, topk_weights, shared, shared_gate):
        calls.append("reduce")
        assert topk_ids is ids and expert_map is op.expert_map and topk_weights is weights
        out.copy_(down.sum(1))

    def masked_reduce(*args, **kwargs):
        raise AssertionError("without a down hook the plain route sum runs")

    monkeypatch.setattr(moe_impl, "moe_align_block_size", align)
    monkeypatch.setattr(moe_impl, "fused_moe_kernel_triton", gemm)
    monkeypatch.setattr(moe_impl, "moe_sum_reduce_triton", reduce)
    monkeypatch.setitem(
        sys.modules,
        "ctikernels._moe_block_fp8_reduce",
        SimpleNamespace(masked_reduce=masked_reduce),
    )
    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quantize)
    monkeypatch.setattr(inference, "silu_and_mul", silu)
    monkeypatch.setitem(
        sys.modules, "ctikernels._moe_block_fp8_cute_act", SimpleNamespace(silu_mul_quant_fp8=fused)
    )
    monkeypatch.setattr(moe_block_fp8, "silu_and_mul_quantize_fp8", triton_fused)
    assert torch.equal(op._triton(x, weights, ids, configs), x)
    expected = ([] if route_blocks else ["align"]) + ["quantize", "fc1"]
    expected += {
        "split": ["silu", "quantize"],
        "cute-silu-quant": ["fused"],
        "triton-silu-quant": ["triton-fused"],
    }[act]
    assert calls == expected + ["fc2", "reduce"]
    assert all(a is b for a, b in zip(*layouts, strict=True))
    assert configs[0]["route_blocks"] == route_blocks  # Caller configs are not mutated.
    assert "act" in configs[1]


def test_prepare_and_config_capture_readiness(monkeypatch):
    op = object.__new__(BlockFP8MoE)
    op.e, op.h, op.i, op.top_k = 2, 128, 128, 8
    op.w13 = torch.empty(2, 256, 128, dtype=torch.float8_e4m3fn)
    op.w2 = torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn)
    op.s13, op.s2 = torch.ones(2, 2, 1), torch.ones(2, 1, 1)
    op.device, op.device_class, op.provider = torch.device("cpu"), (12, 0, 170), "triton-fp8"
    op._available, op._enabled, op._preset = False, False, None
    op._ready, op._route_blocks = set(), {}
    config = op.config_for(1)
    monkeypatch.setattr(op, "config_for", lambda _: config)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    def forward(x, weights, ids, config, **shared):
        op._route_layout(ids, config[0]["BLOCK_SIZE_M"])
        return x.clone()

    monkeypatch.setattr(op, "_triton", forward)
    args = (
        torch.zeros(1, 128, dtype=torch.bfloat16),
        torch.ones(1, 8),
        torch.zeros(1, 8, dtype=torch.int32),
    )
    capturing = True
    with pytest.raises(RuntimeError, match="prepare"):
        op(*args)
    with pytest.raises(RuntimeError, match="outside capture"):
        op.prepare([1])
    # Even a missing constant cache after warm-up must fail before allocation.
    fake_ids = SimpleNamespace(device=torch.device("cuda:0"), is_cuda=True, numel=lambda: 8)
    with pytest.raises(RuntimeError, match="prepare"):
        op._route_layout(fake_ids, 16)
    capturing = False
    op.prepare([1])
    assert len(op._route_blocks) == 1
    capturing = True
    assert torch.equal(op(*args), args[0])
    config[1]["act"] = "cute-silu-quant"
    with pytest.raises(RuntimeError, match="prepare"):
        op(*args)


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize("i", [128, 512, 1536])
def test_offline_activation_compilation(arch, i, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", arch)
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    pytest.importorskip("cutlass.cute")
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.0":
        pytest.skip("requires DSL 4.8.0")
    from ctikernels._moe_block_fp8_cute_act import compile_kernel

    fn = compile_kernel(i, arch=arch, options=f"--keep-ptx --dump-dir={tmp_path}")
    ptx = fn.__ptx__
    if "\n" not in ptx:
        ptx = Path(ptx).read_text()
    assert f".target {arch}" in ptx
    assert "div.rn.f32" in ptx and "ex2.approx" in ptx
    assert "cvt.rn.bf16.f32" in ptx
    assert "e4m3" in ptx and "shfl.sync.bfly" in ptx
    assert "ld.global.v4.b16" in ptx or "ld.global.v2.b32" in ptx or "ld.global.b64" in ptx
    assert "st.global.b32" in ptx
    assert "atom." not in ptx and "bar.sync" not in ptx
