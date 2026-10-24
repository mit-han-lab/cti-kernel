# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Our Triton frozen block-FP8 expert dX (``triton.fused_moe.block_fp8_moe_dx``): the provider
off sm120/121, held here on every compute capability 8.9+ device to the served-point oracle,
the BF16-scratch baseline's accuracy, determinism and EP2 partial sums."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ctikernels._moe_block_fp8_dx import _forward_cache
from ctikernels.moe_block_fp8 import BlockFP8MoE
from ctikernels.moe_impl import moe_align_block_size
from ctikernels.triton import fused_moe

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires compute capability 8.9+",
)


def _reference():
    path = Path(__file__).resolve().parents[2] / "benchmark/kernel_tuning/block_fp8_moe_dx.py"
    spec = importlib.util.spec_from_file_location("dx_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _reference()


def _triton_dx(op, x, weights, ids, dy):
    gu, (q, qs), _ = _forward_cache(op, x, weights, ids, op.config_for(x.shape[0]), None)
    layout = moe_align_block_size(ids, fused_moe.DX_BLOCK_M, op.num_global_experts)
    return fused_moe.block_fp8_moe_dx(op, dy, weights, ids, gu, q, qs, layout)


@pytest.mark.parametrize("shape", [(64, 2048, 512), (32, 1024, 768)])
@pytest.mark.parametrize("m", [1, 4, 64, 300])
def test_triton_dx_matches_the_served_point_oracle(shape, m):
    e, h, i = shape
    params, args = REF.make_case(e, h, i, m, seed=m)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    x, weights, ids, dy = args
    actual = _triton_dx(op, x, weights, ids, dy)
    assert all(torch.equal(a, b) for a, b in zip(actual, _triton_dx(op, *args), strict=True))
    point = REF.served_point(op, x, weights, ids)
    oracle = REF.reference(op, *args, exact=True, formula=True, point=point)
    baseline = REF.reference(op, *args)
    a, b = REF.errors(actual, oracle), REF.errors(baseline, oracle)
    print(f"triton dx E={e} H={h} I={i} M={m} ours={a.tolist()} baseline={b.tolist()}")
    assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == torch.float32
    assert (actual[1][ids == -1] == 0).all()
    tau = 3 * REF.permutation_noise(op, args, oracle, b, 4)
    assert torch.all(a <= b * (1 + tau) + 1e-6 * oracle[1].abs().max().item()), (a, b, tau)


@pytest.mark.parametrize("m", [3, 64])
def test_triton_dx_ep2_partials_sum_to_the_full_result(m):
    params, args = REF.make_case(64, 1024, 512, m, seed=7)
    full = BlockFP8MoE(*params, provider="triton-fp8")
    x, weights, ids, dy = args
    dx, dw = _triton_dx(full, x, weights, ids, dy)
    total_dx = torch.zeros_like(dx, dtype=torch.float32)
    total_dw = torch.zeros_like(dw)
    for rank in range(2):
        lo, hi = rank * 32, (rank + 1) * 32
        mapping = torch.full((64,), -1, dtype=torch.int32, device="cuda")
        mapping[lo:hi] = torch.arange(32, dtype=torch.int32, device="cuda")
        local = BlockFP8MoE(
            *(t[lo:hi] for t in params), provider="triton-fp8", expert_map=mapping,
            num_global_experts=64,
        )  # fmt: skip
        part_dx, part_dw = _triton_dx(local, x, weights, ids, dy)
        remote = (ids < lo) | (ids >= hi)
        assert (part_dw[remote] == 0).all()
        total_dx += part_dx.float()
        total_dw += part_dw
    assert torch.equal(total_dw, dw)
    torch.testing.assert_close(total_dx, dx.float(), rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("m", [3, 64])
def test_public_dx_selects_the_triton_provider(m):
    params, args = REF.make_case(32, 1024, 512, m, seed=5)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    expected = _triton_dx(op, *args)
    actual = op.dx(*args, provider="triton")
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        assert all(torch.equal(a, b) for a, b in zip(op.dx(*args), expected, strict=True))
        with pytest.raises(RuntimeError, match="sm120/121"):
            op.dx(*args, provider="cute")


def test_triton_dx_zero_rows():
    params, args = REF.make_case(8, 1024, 512, 0)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    dx, dw = op.dx(*args)
    assert dx.shape == (0, 1024) and dw.shape == (0, 8)


@pytest.mark.parametrize("row_divisor", [8, 1])
@pytest.mark.parametrize("mapped", [False, True])
@torch.inference_mode()
def test_grouped_dgrad_primitive(row_divisor, mapped):
    """``block_fp8_moe_dgrad``: each local route's row times its expert's decoded weight,
    against FP64; remote and -1 routes untouched; two calls equal bit for bit."""
    e, n, k, m, top_k = 16, 1024, 512, 37, 8
    torch.manual_seed(row_divisor + mapped)
    w = (torch.randn(e, n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
    s = torch.exp2(torch.rand(e, n // 128, k // 128, device="cuda") * 2 - 11)
    ids = torch.randint(e, (m, top_k), device="cuda", dtype=torch.int32)
    ids.flatten()[::9] = -1
    a = torch.randn(m * top_k // row_divisor, n + 64, device="cuda").bfloat16()[:, :n]
    expert_map, local = None, w
    if mapped:
        expert_map = torch.full((e,), -1, dtype=torch.int32, device="cuda")
        expert_map[4:12] = torch.arange(8, dtype=torch.int32, device="cuda")
        local, s = w[4:12], s[4:12]
    layout = moe_align_block_size(ids, fused_moe.DX_BLOCK_M, e)
    out = torch.full((m * top_k, k), 7.0, device="cuda")
    fused_moe.block_fp8_moe_dgrad(
        a, local, s, ids, layout, row_divisor=row_divisor, expert_map=expert_map, out=out
    )
    again = fused_moe.block_fp8_moe_dgrad(
        a, local, s, ids, layout, row_divisor=row_divisor, expert_map=expert_map
    )
    decoded = local.double() * s.double().repeat_interleave(128, 1).repeat_interleave(128, 2)
    for route, expert in enumerate(ids.flatten().tolist()):
        slot = -1 if expert < 0 else (expert if expert_map is None else expert_map[expert].item())
        if slot < 0:
            assert (out[route] == 7.0).all()
            continue
        expected = a[route // row_divisor].double() @ decoded[slot]
        error = (out[route].double() - expected).abs().max() / expected.abs().max()
        assert error < 1e-5, (route, error)
        assert torch.equal(again[route], out[route])
