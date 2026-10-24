# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Metadata contracts, transposed tile address proofs and GPU-free compilation."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE


@pytest.fixture(autouse=True)
def require_cute():
    # These tests import the dX kernels and benchmark, which use the optional DSL.
    pytest.importorskip("cuda.bindings.driver")
    pytest.importorskip("cutlass.cute")


def load_reference():
    path = Path(__file__).resolve().parents[2] / "benchmark/kernel_tuning/block_fp8_moe_dx.py"
    spec = importlib.util.spec_from_file_location("dx_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transposed_tiles_and_pipeline():
    from ctikernels._moe_block_fp8_dx_gemm import BK, BM, BN, STAGES, DxGemm

    # Layout addresses cover the stored [K,N] rectangle once, with one block
    # scale per tile, even across the gate/up boundary and I=768.
    for k, n in ((2048, 512), (2048, 768), (1024, 2048), (1536, 2048)):
        assert k % BK == n % BN == 0
        for kt in range(k // BK):
            assert kt * BK // 128 == (kt * BK + BK - 1) // 128
            assert ((kt + STAGES - 1) % STAGES) != kt % STAGES
        for nt in range(n // BN):
            assert nt * BN // 128 == (nt * BN + BN - 1) // 128
        offsets = [
            (lane % (BN // 16) * 16 + j, lane // (BN // 16))
            for lane in range(128)
            for j in range(16)
        ]
        assert len(set(offsets)) == BN * BK
        assert set(offsets) == {(n, k) for n in range(BN) for k in range(BK)}
    assert BM == 16
    assert 128 % BK == 128 % BN == 0
    with pytest.raises(ValueError, match="128-wide scale blocks"):
        DxGemm(192, 128, 8, activation=True, mapped=False)


def test_raw_weight_conversion_and_block_promotion():
    # E4M3 -> BF16 is exact, including subnormals. Scale multiplication before
    # BF16 MMA loses bits; scaling the FP32 block partial preserves them.
    raw = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    raw = raw[torch.isfinite(raw)]
    assert torch.equal(raw, raw.bfloat16().float())
    a = torch.zeros(1, 256, dtype=torch.bfloat16)
    a[0, 0], a[0, 128] = 1, 2
    q = torch.full((256, 128), 1.125, dtype=torch.float8_e4m3fn)
    scales = torch.tensor([0.3, 0.7], dtype=torch.float32)
    weight = q.double() * scales.repeat_interleave(128)[:, None].double()
    oracle = a.double() @ weight
    scratch = a.float() @ weight.bfloat16().float()
    promoted = torch.zeros_like(oracle, dtype=torch.float32)
    for block in range(2):
        tile = slice(block * 128, (block + 1) * 128)
        partial = a[:, tile].float() @ q[tile].bfloat16().float()
        # FP64 intermediate emulates a single FP32 FMA for these exact products.
        promoted = (scales[block].double() * partial.double() + promoted.double()).float()
    assert (promoted.double() - oracle).abs().max() < (scratch.double() - oracle).abs().max()
    torch.testing.assert_close(promoted.double(), oracle, rtol=1e-7, atol=0)


def test_saved_contract_and_provider():
    from ctikernels._moe_block_fp8_dx import _config, _validate

    op = SimpleNamespace(w13=torch.empty(1), h=128, i=128, top_k=2)
    x = torch.zeros(3, 128, dtype=torch.bfloat16)
    weights, ids = torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int32)
    saved = torch.empty(6, 256, dtype=torch.bfloat16)
    assert _validate(op, x, weights, ids, x, {"gate_up": saved}) is saved
    for bad in (torch.empty(6, 128, dtype=torch.bfloat16), {"down": saved}):
        with pytest.raises(ValueError, match="saved"):
            _validate(op, x, weights, ids, x, bad)
    with pytest.raises(TypeError, match="dy"):
        _validate(op, x, weights, ids, x.float(), None)
    op.provider_for = lambda _: "cute-fp8"
    with pytest.raises(ValueError, match="triton-fp8 forward"):
        _config(op, 3)


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_recompute_uses_served_config_without_fc2(monkeypatch, saved, fused):
    from ctikernels import _moe_block_fp8_dx as dx_module
    from ctikernels import moe_impl
    from ctikernels._moe_block_fp8_dx import _forward_cache
    from ctikernels.linear import block_fp8

    op = object.__new__(BlockFP8MoE)
    op.h, op.i, op.top_k, op.num_global_experts = 128, 128, 2, 4
    op.expert_map, op._route_blocks = None, {}
    op.w13, op.w2, op.s13, op.s2 = (object() for _ in range(4))
    x, ids = torch.zeros(3, 128, dtype=torch.bfloat16), torch.zeros(3, 2, dtype=torch.int32)
    weights = torch.zeros(3, 2)
    fc1 = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64, "BLOCK_SIZE_N": 128, "route_blocks": True}
    fc2 = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_N": 64,
        "route_blocks": True,
        "act": "cute-silu-quant" if fused else "split",
    }
    calls = []

    def gemm(a, w, out, routing, ids, slots, experts, padded, weighted, top_k, config, dtype, **kw):
        calls.append((w, weighted, top_k, config, kw))
        out.zero_()

    monkeypatch.setattr(moe_impl, "fused_moe_kernel_triton", gemm)
    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", lambda x: (x, x))
    packed = torch.zeros(6, 128, dtype=torch.float8_e4m3fn)
    scales = torch.ones(6, 1)

    def activation(gu, act_kind):
        assert gu.shape == (6, 256) and act_kind == ("cute-silu-quant" if fused else "split")
        return packed, scales

    monkeypatch.setattr(dx_module, "_quantize_activation", activation)
    cache = torch.zeros(6, 256, dtype=torch.bfloat16) if saved else None
    gu, (q, qs), layout = _forward_cache(op, x, weights, ids, (fc1, fc2), cache)
    assert len(calls) == (0 if saved else 1)
    if not saved:
        assert calls[0][:3] == (op.w13, False, 2)
        assert calls[0][3]["BLOCK_SIZE_K"] == 64
    assert all("route_blocks" not in c[3] and "act" not in c[3] for c in calls)
    assert fc1["route_blocks"] and fc2["act"] == ("cute-silu-quant" if fused else "split")
    assert gu.shape == (6, 256) and q is packed and qs is scales and layout is not None


@pytest.mark.parametrize("i", [512, 768])
def test_backward_launch_plan_and_partial_scratch(monkeypatch, i):
    from ctikernels import _moe_block_fp8_dx as dx_module
    from ctikernels import _moe_block_fp8_dx_gemm as cute_module
    from ctikernels._moe_block_fp8_dx_gemm import BN

    op = SimpleNamespace(
        h=2048,
        i=i,
        top_k=2,
        expert_map=None,
        w13=torch.empty(1),
        w2=torch.empty(1),
        s13=torch.empty(1),
        s2=torch.empty(1),
        provider_for=lambda _: "triton-fp8",
        config_for=lambda _: ({}, {}),
    )
    x = torch.zeros(3, op.h, dtype=torch.bfloat16)
    ids, weights = torch.zeros(3, 2, dtype=torch.int32), torch.ones(3, 2)
    gu = torch.empty(6, 2 * i, dtype=torch.bfloat16)
    q, qs = torch.empty(6, i, dtype=torch.float8_e4m3fn), torch.empty(6, i // 128)
    layout = (object(), object(), object())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dx_module, "_forward_cache", lambda *a: (gu, (q, qs), layout))
    calls = []
    monkeypatch.setattr(dx_module, "_cute", cute_module)
    monkeypatch.setattr(
        cute_module,
        "_compiled",
        lambda *a: tuple(
            lambda *args, name=name: calls.append((name, args)) for name in ("da", "dx", "finish")
        ),
    )
    dx, dw = dx_module.block_fp8_moe_dx(op, x, weights, ids, x)
    assert [name for name, _ in calls] == ["da", "dx", "finish"]
    da, backward, finish = [args for _, args in calls]
    assert da[8] is gu and da[9] is q and da[10] is qs
    scratch = da[11]
    assert scratch.shape == (6, i // BN) and scratch.dtype == torch.float32
    assert backward[11] is scratch and finish[1] is scratch
    assert da[12].shape == gu.shape and da[12].dtype == torch.bfloat16
    assert backward[12].shape == (6, op.h) and backward[12].dtype == torch.float32
    assert finish[0] is backward[12] and finish[4] is dx
    assert dx.shape == x.shape and dw.shape == weights.shape


def test_reference_formula_and_ste_autograd():
    ref = load_reference()
    params, inputs = ref.make_case(4, 128, 128, 3, device="cpu")
    op = SimpleNamespace(
        w13=params[0],
        s13=params[1],
        w2=params[2],
        s2=params[3],
        h=128,
        i=128,
        top_k=8,
        expert_map=None,
    )
    actual = ref.reference(op, *inputs, exact=True)
    formula = ref.reference(op, *inputs, exact=True, formula=True)
    for a, b in zip(actual, formula, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    x = torch.randn(2, 128, requires_grad=True)
    upstream = torch.randn_like(x)
    (dx,) = torch.autograd.grad((ref.quant_ste(x) * upstream).sum(), (x,))
    assert torch.equal(dx, upstream)
    for dtype in (torch.bfloat16, torch.float32):
        value = x.detach().to(dtype)
        groups = value.float().reshape(2, 1, 128)
        peak = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
        packed = (groups * (448.0 / peak)).clamp(-448, 448).to(torch.float8_e4m3fn)
        expected = (packed.float() * (peak * (1.0 / 448.0))).reshape_as(value).to(dtype)
        assert torch.equal(ref.quant_ste(value), expected)
    baseline = ref.reference(op, *inputs)
    assert baseline[0].dtype == torch.bfloat16 and baseline[1].dtype == torch.float32
    assert all(torch.isfinite(t).all() for t in baseline)
    # Multiplying scales (including BF16) and rounding to the BF16 scratch.
    q = torch.ones(1, 128, 128, dtype=torch.float8_e4m3fn)
    s = torch.tensor([[[0.3]]], dtype=torch.bfloat16)
    assert torch.equal(
        ref.dequantize(q, s, torch.float32).bfloat16(),
        torch.full((1, 128, 128), s.item(), dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("mapped", [False, True])
def test_served_point_formula_and_ste_autograd(mapped):
    ref = load_reference()
    params, args = ref.make_case(4, 128, 128, 3, device="cpu")
    op = SimpleNamespace(
        w13=params[0],
        s13=params[1],
        w2=params[2],
        s2=params[3],
        h=128,
        i=128,
        top_k=8,
        expert_map=torch.tensor([0, -1, 1, -1]) if mapped else None,
    )
    x, weights, ids, dy = args
    ids[0, :3] = torch.tensor([0, -1, 1])
    weights[0, 0] = 0
    # Synthetic saved values deliberately differ from a recomputed forward.
    # The GPU test captures the real shipped quantizers and FC1 instead.
    generator = torch.Generator().manual_seed(53)
    point = (
        ref.quant_ste(x.double()),
        torch.randn(ids.numel(), 256, generator=generator).bfloat16(),
        ref.quant_ste(torch.randn(ids.numel(), 128, generator=generator).double()),
    )
    formula = ref.reference(op, *args, exact=True, formula=True, point=point)
    autograd = ref.reference(op, *args, exact=True, point=point)
    for a, b in zip(formula, autograd, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-12)
    assert formula[1][0, 0] != 0 and formula[1][0, 1] == 0
    if mapped:
        assert (formula[1][(ids == 1) | (ids == 3)] == 0).all()
    changed = ref.reference(
        op, *args, exact=True, formula=True, point=(point[0] * 2, point[1], point[2] * 2)
    )
    assert torch.equal(changed[0], formula[0])  # STE dX uses saved g/u, not xq or aq.
    assert torch.equal(changed[1], formula[1] * 2)
    with pytest.raises(ValueError, match="exact=True"):
        ref.reference(op, *args, point=point)


@pytest.mark.parametrize("fused", [False, True])
def test_served_point_uses_shipped_quantizers(monkeypatch, fused):
    from ctikernels.linear import block_fp8

    ref = load_reference()
    x, weights, ids = (
        torch.zeros(1, 128, dtype=torch.bfloat16),
        torch.ones(1, 2),
        torch.zeros(1, 2, dtype=torch.int32),
    )
    gu = torch.zeros(2, 256, dtype=torch.bfloat16)
    op = SimpleNamespace(
        config_for=lambda _: ({}, {"act": "cute-silu-quant" if fused else "split"})
    )

    def quant(value):
        assert value is x
        return torch.ones_like(x, dtype=torch.float8_e4m3fn), torch.full((1, 1), 0.3)

    def cache(actual_op, actual_x, actual_weights, actual_ids, config, saved):
        assert actual_op is op and actual_x is x and actual_weights is weights and actual_ids is ids
        assert saved is None
        return gu, activation(gu, fused), None

    def activation(value, fused_act):
        assert value is gu and fused_act == fused
        return torch.full((2, 128), 1.125, dtype=torch.float8_e4m3fn), torch.full((2, 1), 0.7)

    monkeypatch.setattr(block_fp8, "quantize_per_token_group_fp8", quant)
    monkeypatch.setattr(ref, "_forward_cache", cache)
    xq, saved, aq = ref.served_point(op, x, weights, ids)
    assert saved is gu and xq.dtype == aq.dtype == torch.float64
    assert torch.equal(xq, torch.full_like(xq, torch.tensor(0.3).item()))
    assert torch.equal(aq, torch.full_like(aq, 1.125 * torch.tensor(0.7).item()))


@pytest.mark.parametrize("noisy", [False, True])
def test_measured_permutation_and_sum_noise(monkeypatch, noisy):
    ref = load_reference()
    op = SimpleNamespace(top_k=3)
    routes = torch.tensor([2**24, 1, -(2**24)] if noisy else [1, 1, 1]).bfloat16()
    routes = routes.view(1, 3, 1).expand(1, 3, 128)
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    weights, ids = torch.ones(1, 3), torch.arange(3).view(1, 3)
    oracle = (routes.double().sum(1), torch.zeros(1, 3, dtype=torch.float64))
    if not noisy:
        oracle = (oracle[0] + 0.3, oracle[1])  # Constant nonzero error must give zero spread.

    def baseline(op, x, weights, ids, dy, return_routes=False):
        reordered = routes[:, ids[0]]
        result = (ref.sum_slots(reordered), torch.zeros_like(weights))
        return (result, reordered) if return_routes else result

    monkeypatch.setattr(ref, "reference", baseline)
    args = (x, weights, ids, x)
    error = ref.errors(baseline(op, *args), oracle)
    sigma = ref.permutation_noise(op, args, oracle, error)
    assert (sigma[:2] > 0).all() if noisy else (sigma == 0).all()
    assert (sigma[2:] == 0).all()
    ratios = ref.error_ratios(torch.tensor([0.0, 1.0]), torch.zeros(2))
    assert ratios[0] == 1 and ratios[1].isinf()


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize("i,mapped,bf16", [(512, False, False), (768, True, True)])
def test_offline_compile(arch, i, mapped, bf16, tmp_path, monkeypatch):
    from ctikernels._moe_block_fp8_dx_gemm import compile_finish, compile_gemm

    # With a GPU visible, an earlier test of this process may have initialized CUDA.
    initialized = torch.cuda.is_initialized()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUTE_DSL_ARCH", arch)
    monkeypatch.setenv("CUTE_DSL_DISABLE_FILE_CACHING", "1")
    options = f"--keep-ptx --dump-dir={tmp_path}"
    for activation in (True, False):
        fn = compile_gemm(
            2048,
            i,
            8,
            activation=activation,
            mapped=mapped,
            scale_bf16=bf16,
            arch=arch,
            options=options,
        )
        ptx = fn.__ptx__
        if "\n" not in ptx:
            ptx = Path(ptx).read_text()
        for instruction in (
            f".target {arch}",
            "cp.async.cg.shared.global",
            "cp.async.wait_group",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16",
            "cvt.rn.bf16.f32",
            "fma.rn.f32",
        ):
            assert instruction in ptx
        assert "atom." not in ptx
        assert ("shfl.sync.bfly" in ptx) == activation
        assert "ld.local" not in ptx and "st.local" not in ptx
    fn = compile_finish(2048, i, 8, mapped=mapped, arch=arch, options=options)
    ptx = fn.__ptx__
    if "\n" not in ptx:
        ptx = Path(ptx).read_text()
    assert "atom." not in ptx and "shfl.sync" not in ptx
    assert torch.cuda.is_initialized() == initialized
