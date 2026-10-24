# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen FP8 dX qualification. Requires sm120/121; never run during offline work."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ctikernels import _moe_block_fp8_triton_table as triton_policy
from ctikernels._moe_block_fp8_dx import _forward_cache, block_fp8_moe_dx, prepare_dx
from ctikernels.moe_block_fp8 import BlockFP8MoE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="requires sm120/121",
)


def load_reference():
    path = Path(__file__).resolve().parents[2] / "benchmark/kernel_tuning/block_fp8_moe_dx.py"
    spec = importlib.util.spec_from_file_location("dx_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference()
SEEDS, PERMUTATIONS = (11, 23, 37, 41), 8


@pytest.fixture(scope="module", params=[(256, 2048, 512), (128, 2048, 768)])
def op(request):
    e, h, i = request.param
    params, _ = REF.make_case(e, h, i, 1)
    return BlockFP8MoE(*params, provider="triton-fp8")


@pytest.fixture(scope="module")
def rank_ops(op):
    # Reuse each shape/rank's compiled and prepared state across all cases.
    ranks = []
    for rank in range(2):
        lo, hi = rank * op.e // 2, (rank + 1) * op.e // 2
        mapping = torch.full((op.e,), -1, device=op.device, dtype=torch.int32)
        mapping[lo:hi] = torch.arange(hi - lo, device=op.device, dtype=torch.int32)
        ranks.append(
            BlockFP8MoE(
                op.w13[lo:hi],
                op.s13[lo:hi],
                op.w2[lo:hi],
                op.s2[lo:hi],
                provider="triton-fp8",
                expert_map=mapping,
                num_global_experts=op.e,
            )
        )
    return ranks


def assert_ep2_dispatch(op, local, m, rank, record_property):
    shape = (local.e, local.h, local.i, local.top_k)
    measured = triton_policy.MEASURED.get(local.device_class, {}).get(shape, ())
    assert measured, (
        f"EP2 rank {rank}: no exact measured key {shape} for device {local.device_class}; "
        "placeholder/heuristic dispatch cannot qualify EP2 dX"
    )
    # Past the last measured row production uses block_fp8_moe_config; qualify that too.
    covered = any(m <= limit for limit, _, _ in measured)
    record_property("measured_row_covers_m", covered)
    assert triton_policy.TABLE[local.device_class][shape] == triton_policy._measured_rows(measured)
    expected = triton_policy.selection(local.device_class, m, *shape)
    expected[1]["act"] = "cute-silu-quant" if local._available and local._enabled else "split"
    config = local.config_for(m)
    assert config == expected
    different = [a != b for a, b in zip(config, op.config_for(m), strict=True)]
    record_property("ep2_fc1", config[0])
    record_property("ep2_fc2", config[1])
    record_property("differs_from_ep1", different)
    print(
        f"EP2 E_local={local.e} I={local.i} rank={rank} M={m} "
        f"FC1={config[0]} FC2={config[1]} differs_from_ep1={different} measured={covered}"
    )


def inputs(op, m, skewed=False, seed=11):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, op.h, device="cuda", generator=generator).bfloat16()
    dy = torch.randn(m, op.h, device="cuda", generator=generator).bfloat16()
    ids = torch.randint(op.e, (m, op.top_k), device="cuda", dtype=torch.int32, generator=generator)
    if skewed:
        ids[:, :6] %= 4
    ids.flatten()[seed % 7 :: 11] = -1
    weights = torch.rand(m, op.top_k, device="cuda", generator=generator)
    weights /= weights.sum(-1, keepdim=True)
    weights.flatten()[::13] = 0
    return x, weights, ids, dy


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64, 512])
@pytest.mark.parametrize("skewed", [False, True])
def test_accuracy_no_worse_than_bf16_scratch(op, m, skewed):
    records = []
    for seed in SEEDS:
        args = inputs(op, m, skewed, seed)
        point = REF.served_point(op, *args[:3])
        actual = block_fp8_moe_dx(op, *args, saved=point[1])
        baseline = REF.reference(op, *args)
        oracle = REF.reference(op, *args, exact=True, formula=True, point=point)
        a, b = REF.errors(actual, oracle), REF.errors(baseline, oracle)
        # FP32-scale accuracy at the served point, independent of baseline/tau.
        # A cancellation-aware forward-error bound is checked separately below.
        assert a[2] <= 1e-4 * oracle[1].abs().max().item() + 1e-7
        tau = 3 * REF.permutation_noise(op, args, oracle, b, PERMUTATIONS)
        unquantized = REF.reference(op, *args, exact=True, formula=True)
        floor = REF.errors((oracle[0].bfloat16(), oracle[1].float()), oracle)
        print(
            f"dx E={op.e} I={op.i} M={m} skewed={skewed} seed={seed} "
            f"served_ours={a.tolist()} served_baseline={b.tolist()} tau={tau.tolist()} "
            f"unquantized_ours={REF.errors(actual, unquantized).tolist()} "
            f"unquantized_baseline={REF.errors(baseline, unquantized).tolist()} "
            f"output_rounding_floor={floor.tolist()}"
        )
        records.append((seed, a, b, tau))
        assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == torch.float32
        assert (actual[1][args[2] == -1] == 0).all()
        # Router gradients must survive a zero routing weight.
        zero_weight = (args[1] == 0) & (args[2] >= 0)
        assert (actual[1][zero_weight] != 0).all()
    # Per-seed 3-sigma tolerance is measured, including an exactly zero spread.
    for seed, a, b, tau in records:
        assert torch.all(a <= b * (1 + tau)), (seed, a, b, tau)
    ratios = torch.stack([REF.error_ratios(a, b) for _, a, b, _ in records])
    assert torch.all(ratios.mean(0) <= 1), ratios.mean(0)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64, 512])
def test_saved_determinism_and_graph(op, m):
    args = inputs(op, m, True)
    prepare_dx(op, [m])
    eager = block_fp8_moe_dx(op, *args)
    again = block_fp8_moe_dx(op, *args)
    assert all(torch.equal(a, b) for a, b in zip(eager, again, strict=True))
    gu, _, _ = _forward_cache(op, *args[:3], op.config_for(m), None)
    cached = block_fp8_moe_dx(op, *args, saved={"gate_up": gu})
    assert all(torch.equal(a, b) for a, b in zip(eager, cached, strict=True))
    for saved in (None, gu):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = block_fp8_moe_dx(op, *args, saved=saved)
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(eager, captured, strict=True))
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(eager, captured, strict=True))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64, 512])
@pytest.mark.parametrize("skewed", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("rank", [0, 1])
def test_ep2_production_accuracy(op, rank_ops, m, skewed, rank, record_property):
    local = rank_ops[rank]
    assert_ep2_dispatch(op, local, m, rank, record_property)
    records = []
    for seed in SEEDS:
        args = inputs(op, m, skewed, seed)  # Global IDs, including remote and invalid routes.
        point = REF.served_point(local, *args[:3])
        actual = block_fp8_moe_dx(local, *args, saved=point[1])
        baseline = REF.reference(local, *args)
        oracle = REF.reference(local, *args, exact=True, formula=True, point=point)
        a, b = REF.errors(actual, oracle), REF.errors(baseline, oracle)
        # FP32-scale accuracy at the served point, independent of baseline/tau.
        # A cancellation-aware forward-error bound is checked separately below.
        assert a[2] <= 1e-4 * oracle[1].abs().max().item() + 1e-7
        tau = 3 * REF.permutation_noise(local, args, oracle, b, PERMUTATIONS)
        print(
            f"EP2 dx E_local={local.e} I={local.i} rank={rank} M={m} "
            f"skewed={skewed} seed={seed} served_ours={a.tolist()} "
            f"served_baseline={b.tolist()} tau={tau.tolist()}"
        )
        records.append((seed, a, b, tau))
        remote = (args[2] < rank * local.e) | (args[2] >= (rank + 1) * local.e)
        assert actual[0].dtype == torch.bfloat16 and actual[1].dtype == torch.float32
        for result in (actual, baseline, oracle):
            assert (result[1][remote] == 0).all()
            assert (result[0][remote.all(dim=1)] == 0).all()
        zero_weight = (args[1] == 0) & ~remote
        assert (actual[1][zero_weight] != 0).all()
    for seed, a, b, tau in records:
        assert torch.all(a <= b * (1 + tau)), (seed, a, b, tau)
    ratios = torch.stack([REF.error_ratios(a, b) for _, a, b, _ in records])
    assert torch.all(ratios.mean(0) <= 1), ratios.mean(0)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64, 512])
@pytest.mark.parametrize("skewed", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("rank", [0, 1])
def test_ep2_production_saved_determinism_and_graph(op, rank_ops, m, skewed, rank, record_property):
    local = rank_ops[rank]
    assert_ep2_dispatch(op, local, m, rank, record_property)
    args = inputs(op, m, skewed)
    prepare_dx(local, [m])
    eager = block_fp8_moe_dx(local, *args)
    again = block_fp8_moe_dx(local, *args)
    assert all(torch.equal(a, b) for a, b in zip(eager, again, strict=True))
    gu, _, _ = _forward_cache(local, *args[:3], local.config_for(m), None)
    cached = block_fp8_moe_dx(local, *args, saved={"gate_up": gu})
    assert all(torch.equal(a, b) for a, b in zip(eager, cached, strict=True))
    for saved in (None, gu):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = block_fp8_moe_dx(local, *args, saved=saved)
        graph.replay()
        first = tuple(t.clone() for t in captured)
        assert all(torch.equal(a, b) for a, b in zip(eager, first, strict=True))
        graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(first, captured, strict=True))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 64, 512])
@pytest.mark.parametrize("skewed", [False, True])
def test_ep2_rank_sum(op, m, skewed):
    args = inputs(op, m, skewed)
    whole = block_fp8_moe_dx(op, *args)
    ranks = []
    config = op.config_for(m)
    for rank in range(2):
        lo, hi = rank * op.e // 2, (rank + 1) * op.e // 2
        mapping = torch.full((op.e,), -1, device="cuda", dtype=torch.int32)
        mapping[lo:hi] = torch.arange(hi - lo, device="cuda", dtype=torch.int32)
        local = BlockFP8MoE(
            op.w13[lo:hi],
            op.s13[lo:hi],
            op.w2[lo:hi],
            op.s2[lo:hi],
            provider="triton-fp8",
            expert_map=mapping,
            num_global_experts=op.e,
        )
        # Preserve forward arithmetic: EP's different local E must not retune FC1.
        local.config_for = lambda _, config=config: config
        result = block_fp8_moe_dx(local, *args)
        remote = (args[2] < lo) | (args[2] >= hi)
        assert (result[1][remote] == 0).all()
        ranks.append(result)
        rank_ids = torch.where(remote, -1, args[2])
        full_rank = block_fp8_moe_dx(op, args[0], args[1], rank_ids, args[3])
        assert all(torch.equal(a, b) for a, b in zip(result, full_rank, strict=True))
    summed = ranks[0][0].float() + ranks[1][0].float()
    # Three final BF16 roundings (two ranks and EP1), plus FP32 reduction order.
    bound = (
        torch.finfo(torch.bfloat16).eps
        / 2
        * (ranks[0][0].float().abs() + ranks[1][0].float().abs() + whole[0].float().abs())
    )
    torch.testing.assert_close(summed, whole[0].float(), atol=bound.max().item() + 1e-7, rtol=0)
    torch.testing.assert_close(ranks[0][1] + ranks[1][1], whole[1], atol=0, rtol=0)


def test_empty_and_all_remote(op):
    args = inputs(op, 0)
    dx, dw = block_fp8_moe_dx(op, *args)
    assert dx.shape == args[0].shape and dw.shape == args[1].shape
    args = inputs(op, 3)
    args[2].fill_(-1)
    dx, dw = block_fp8_moe_dx(op, *args)
    assert torch.count_nonzero(dx) == torch.count_nonzero(dw) == 0
    mapping = torch.full((op.e,), -1, dtype=torch.int32, device="cuda")
    local = BlockFP8MoE(op.w13, op.s13, op.w2, op.s2, provider="triton-fp8", expert_map=mapping)
    args[2].zero_()
    dx, dw = block_fp8_moe_dx(local, *args)
    assert torch.count_nonzero(dx) == torch.count_nonzero(dw) == 0


def test_served_ste_formula_with_autograd():
    params, args = REF.make_case(2, 128, 128, 2)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    point = REF.served_point(op, *args[:3])
    formula = REF.reference(op, *args, exact=True, formula=True, point=point)
    autograd = REF.reference(op, *args, exact=True, point=point)
    for a, b in zip(formula, autograd, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
def test_served_gate_up_backward_formula(scale_dtype):
    params, args = REF.make_case(4, 128, 128, 3)
    params[1], params[3] = params[1].to(scale_dtype), params[3].to(scale_dtype)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    x, weights, ids, dy = args
    point = REF.served_point(op, x, weights, ids)
    gu, aq = point[1], point[2]
    actual = block_fp8_moe_dx(op, *args, saved=gu)
    w13 = REF.dequantize(op.w13, op.s13, torch.float64)
    w2 = REF.dequantize(op.w2, op.s2, torch.float64)
    route_dx = torch.zeros(ids.numel(), op.h, device="cuda", dtype=torch.float64)
    dw = torch.zeros_like(weights).flatten()
    for e, routes in REF.route_groups(op, ids):
        tokens = routes // op.top_k
        g, u = gu[routes].double().chunk(2, -1)
        dd = (dy[tokens].float() * weights.flatten()[routes, None]).bfloat16().double()
        da = dd @ w2[e]
        sigmoid = torch.sigmoid(g)
        dg = da * u * sigmoid * (1 + g * (1 - sigmoid))
        du = da * g * sigmoid
        dgdu = torch.cat((dg, du), -1).bfloat16().double()
        route_dx[routes] = dgdu @ w13[e]
        dw[routes] = (aq[routes] * (dy[tokens].double() @ w2[e])).sum(-1).float()
    expected_dx = route_dx.view(3, op.top_k, op.h).sum(1).bfloat16()
    # Allow BF16 ties from FP32 MMA / approximate exp; the full-shape gate
    # separately enforces the measured baseline-relative max/RMS requirement.
    torch.testing.assert_close(actual[0], expected_dx, rtol=0.016, atol=2e-4)
    torch.testing.assert_close(actual[1], dw.view_as(weights), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("i", [512, 768])
@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fused", [False, True])
def test_router_fp32_forward_error_bound(i, scale_dtype, fused):
    # Small E/M keeps this direct check cheap, with production H/I and both
    # served activation quantizers. NaN routing weights on masked/remote
    # routes catch accidental reads; live zero/negative weights test independence.
    params, args = REF.make_case(4, 2048, i, 3)
    for j in (1, 3):
        params[j] = params[j].to(scale_dtype)
        if scale_dtype == torch.float32:
            params[j] *= 1.0012345  # Exercise scale bits not representable in BF16.
    mapping = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device="cuda")
    op = BlockFP8MoE(
        *(p[:2] for p in params),
        provider="triton-fp8",
        expert_map=mapping,
        num_global_experts=4,
    )
    config = op.config_for(3)
    config[1]["act"] = "cute-silu-quant" if fused else "split"
    op.config_for = lambda _: tuple(c.copy() for c in config)
    x, weights, ids, dy = args
    ids[:] = torch.tensor([0, 2, 0, 2, -1, 1, 3, -1], device="cuda", dtype=torch.int32)
    weights[:, :4] = torch.tensor([0, -0.7, 0.3, 1.2], device="cuda")
    weights[:, 4:] = float("nan")
    point = REF.served_point(op, x, weights, ids)
    actual = block_fp8_moe_dx(op, *args, saved=point[1])
    oracle = REF.reference(op, *args, exact=True, formula=True, point=point)
    w2 = REF.dequantize(op.w2, op.s2, torch.float64)
    eps = torch.finfo(torch.float32).eps / 2

    def gamma(n):
        return n * eps / (1 - n * eps)

    # Each 128-term raw BF16*FP8 dot has exact FP32 products; gamma(128)
    # bounds its sum. H/128 scale FMAs add that many roundings. For K=I,
    # dequantization + product + any fixed sum tree use <= K+2 roundings
    # along a term's path (our 64-column tree is shorter). Summing absolute
    # terms, rather than |dW|, makes the bound valid under cancellation.
    gg, gd = gamma(128 + op.h // 128), gamma(i + 2)
    for e, routes in REF.route_groups(op, ids):
        a = point[2][routes]
        d = dy[routes // op.top_k].double()
        g = d @ w2[e]
        g_error = gg * (d.abs() @ w2[e].abs())
        bound = (a.abs() * g_error).sum(-1) * (1 + gd) + gd * (a * g).abs().sum(-1)
        error = (actual[1].flatten()[routes].double() - oracle[1].flatten()[routes]).abs()
        assert torch.all(error <= bound + 1e-10), (error, bound)
    assert (actual[1][:, 4:] == 0).all()
    assert (actual[1][:, 0] != 0).all()
    assert REF.errors(actual, oracle)[2] <= 1e-4 * oracle[1].abs().max().item() + 1e-7
    # Identical token/expert/activation routes produce identical unweighted dW.
    assert torch.equal(actual[1][:, 0], actual[1][:, 2])
    assert torch.equal(actual[1][:, 1], actual[1][:, 3])


def test_method_matches_function(op):
    args = inputs(op, 3, True)
    op.prepare_dx([3])
    for a, b in zip(op.dx(*args), block_fp8_moe_dx(op, *args), strict=True):
        assert torch.equal(a, b)
