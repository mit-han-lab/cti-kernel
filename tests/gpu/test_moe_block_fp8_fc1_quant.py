# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""FC1 with the SiLU*up + 1x128 quantization in its epilogue (``swiglu_quant_scale``, a
256-column tile): the bits of the unfused FC1 + activation chain, with and without EP."""

import importlib.util
from pathlib import Path

import pytest
import torch

from ctikernels.moe_block_fp8 import BlockFP8MoE
from ctikernels.moe_impl import fused_moe_kernel_triton, moe_align_block_size

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires compute capability 8.9+",
)


def _case(e, h, i, m, seed):
    path = Path(__file__).resolve().parents[2] / "benchmark/kernel_tuning/block_fp8_moe_dx.py"
    spec = importlib.util.spec_from_file_location("dx_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_case(e, h, i, m, seed=seed)


def _fused_config(op, m):
    fc1, fc2 = (dict(c) for c in op.config_for(m))
    fc1.update(BLOCK_SIZE_N=256, num_stages=2)  # fits sm12x shared memory at every BLOCK_SIZE_M
    return fc1, fc2


@pytest.mark.parametrize("shape", [(64, 2048, 512), (32, 1024, 768)])
@pytest.mark.parametrize("m", [1, 5, 300])
@torch.inference_mode()
def test_fused_fc1_matches_the_unfused_chain_bitwise(shape, m):
    params, (x, weights, ids, _) = _case(*shape, m, seed=m)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    config = _fused_config(op, m)
    fused = op._triton(x, weights, ids, config)
    unfused, _ = op._triton(x, weights, ids, config, return_act=True)  # gate_up materialized
    assert torch.equal(fused, unfused)


@pytest.mark.parametrize("m", [3, 64])
@torch.inference_mode()
def test_fused_fc1_with_expert_map_matches_the_unfused_chain(m):
    params, (x, weights, ids, _) = _case(64, 1024, 512, m, seed=11)
    mapping = torch.full((64,), -1, dtype=torch.int32, device="cuda")
    mapping[16:48] = torch.arange(32, dtype=torch.int32, device="cuda")
    op = BlockFP8MoE(
        *(t[16:48] for t in params), provider="triton-fp8", expert_map=mapping,
        num_global_experts=64,
    )  # fmt: skip
    config = _fused_config(op, m)
    fused = op._triton(x, weights, ids, config)
    unfused, _ = op._triton(x, weights, ids, config, return_act=True)
    assert torch.equal(fused, unfused)


@torch.inference_mode()
def test_fused_fc1_requires_a_256_column_tile():
    params, (x, weights, ids, _) = _case(8, 256, 256, 2, seed=1)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    fc1, _ = op.config_for(2)
    fc1 = {k: v for k, v in fc1.items() if k != "route_blocks"} | {"BLOCK_SIZE_N": 128}
    routes = ids.numel()
    q = torch.zeros(2, op.top_k, op.i, dtype=torch.float8_e4m3fn, device="cuda")
    scale = torch.empty(routes, op.i // 128, device="cuda")
    layout = moe_align_block_size(ids, fc1["BLOCK_SIZE_M"], op.num_global_experts)
    with pytest.raises(ValueError, match="BLOCK_SIZE_N 256"):
        fused_moe_kernel_triton(
            x.to(torch.float8_e4m3fn), op.w13, q, weights, ids, *layout, False, op.top_k, fc1,
            x.dtype, a_scale=torch.ones(2, op.h // 128, device="cuda"), b_scale=op.s13,
            block_fp8_provider="triton-fp8", swiglu_quant_scale=scale,
        )  # fmt: skip


def _fc1(op, x, weights, ids, *, lora=None, act=False, fused=True):
    """FC1 through the block-FP8 grouped GEMM: fused (q, scale, act) or BF16 gate_up."""
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8

    m, routes = x.shape[0], ids.numel()
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256 if fused else 128, "BLOCK_SIZE_K": 128,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}  # fmt: skip
    layout = moe_align_block_size(ids, 16, op.num_global_experts)
    q, qs = quantize_per_token_group_fp8(x)
    kwargs = {}
    if fused:
        out = torch.empty(m, op.top_k, op.i, dtype=torch.float8_e4m3fn, device="cuda")
        kwargs["swiglu_quant_scale"] = torch.empty(routes, op.i // 128, device="cuda")
        if act:
            kwargs["swiglu_act"] = torch.empty(routes, op.i, dtype=x.dtype, device="cuda")
        if lora is not None:
            kwargs["gate_up_lora"] = lora
    else:
        out = torch.empty(m, op.top_k, 2 * op.i, dtype=x.dtype, device="cuda")
    fused_moe_kernel_triton(
        q, op.w13, out, weights, ids, *layout, False, op.top_k, config, x.dtype, a_scale=qs,
        b_scale=op.s13, block_fp8_provider="triton-fp8", expert_map=op.expert_map,
        num_global_experts=op.num_global_experts, **kwargs,
    )  # fmt: skip
    if not fused:
        return out.view(routes, 2 * op.i)
    return out.view(routes, op.i), kwargs["swiglu_quant_scale"], kwargs.get("swiglu_act")


@pytest.mark.parametrize("m", [1, 5, 64])
@pytest.mark.parametrize("rank", [8, 16, 64])
@torch.inference_mode()
def test_fused_fc1_lora_and_act_output(m, rank):
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.linear.block_fp8_gated import silu_and_mul_quantize_fp8
    from ctikernels.moe_impl import GateUpLoRA

    params, (x, weights, ids, _) = _case(16, 1024, 256, m, seed=m + rank)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    routes, local = ids.numel(), ids.view(-1).clamp(min=0).long()
    torch.manual_seed(rank)
    low = torch.randn(routes, 2, rank, device="cuda") * 0.5
    b_gate = torch.randn(op.e, op.i, rank, device="cuda", dtype=x.dtype) * 0.05
    b_up = torch.randn(op.e, op.i, rank, device="cuda", dtype=x.dtype) * 0.05  # separate leaf
    valid = ids.view(-1) >= 0
    # A zero delta and the act output leave the fused bits unchanged; act is the chain's.
    zero = GateUpLoRA(low, b_gate, b_up, torch.zeros(1, device="cuda"))
    q0, s0, act0 = _fc1(op, x, weights, ids, lora=zero, act=True)
    q1, s1, _ = _fc1(op, x, weights, ids)
    assert torch.equal(q0[valid].view(torch.uint8), q1[valid].view(torch.uint8))
    assert torch.equal(s0[valid], s1[valid])
    expected_act = torch.empty_like(act0)
    silu_and_mul_quantize_fp8(_fc1(op, x, weights, ids, fused=False), act=expected_act)
    assert torch.equal(act0[valid], expected_act[valid])
    # A real delta: FP32 add before the single rounding (oracle: the unfused FP32 sum).
    lora = GateUpLoRA(low, b_gate, b_up, torch.full((1,), 0.75, device="cuda"))
    q, s, act = _fc1(op, x, weights, ids, lora=lora, act=True)
    gate_up = _fc1(op, x, weights, ids, fused=False).float()
    delta = torch.cat(
        [torch.einsum("rk,rnk->rn", low[:, j], b[local].float()) for j, b in enumerate((b_gate, b_up))],
        dim=1,
    )  # fmt: skip
    g, u = (gate_up + 0.75 * delta).to(x.dtype).float().chunk(2, -1)
    reference = (g / (1 + torch.exp(-g)) * u)[valid]
    torch.testing.assert_close(act[valid].float(), reference, rtol=2e-2, atol=1e-2)
    # q and scale are the 1x128 quantization of the act it stored, bit for bit.
    eq, es = quantize_per_token_group_fp8(act)
    assert torch.equal(q[valid].view(torch.uint8), eq[valid].view(torch.uint8))
    assert torch.equal(s[valid], es[valid])


@pytest.mark.parametrize("m,rank", [(1, 16), (4, 16), (64, 64)])
@torch.inference_mode()
def test_fused_fc1_lora_is_no_less_accurate_than_the_chain(m, rank):
    """Against the FP64 W8A8 oracle, the single-rounding LoRA epilogue's activation error is at
    most the old chain's (BF16 gate_up, BF16 expand added, activation)."""
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.linear.block_fp8_gated import silu_and_mul_quantize_fp8
    from ctikernels.moe_impl import GateUpLoRA

    params, (x, weights, ids, _) = _case(16, 2048, 512, m, seed=m + rank)
    ids = ids.clamp(min=0)
    op = BlockFP8MoE(*params, provider="triton-fp8")
    routes, local = ids.numel(), ids.view(-1).long()
    torch.manual_seed(rank)
    low = torch.randn(routes, 2, rank, device="cuda")
    b = [torch.randn(op.e, op.i, rank, device="cuda", dtype=x.dtype) * 0.05 for _ in range(2)]
    lora = GateUpLoRA(low, b[0], b[1], torch.full((1,), 2.0, device="cuda"))
    _, _, fused = _fc1(op, x, weights, ids, lora=lora, act=True)
    delta = 2.0 * torch.cat(
        [torch.einsum("rk,rnk->rn", low[:, j].double(), b[j][local].double()) for j in (0, 1)], 1
    )
    chain_gate_up = _fc1(op, x, weights, ids, fused=False).float() + delta.to(x.dtype).float()
    chain = torch.empty_like(fused)
    silu_and_mul_quantize_fp8(chain_gate_up.to(x.dtype), act=chain)
    q, scale = quantize_per_token_group_fp8(x)
    rows = (q.double() * scale.double().repeat_interleave(128, 1))[
        torch.arange(routes, device="cuda") // op.top_k
    ]
    w = op.w13.double() * op.s13.double().repeat_interleave(128, 1).repeat_interleave(128, 2)
    g, u = (torch.einsum("rk,rnk->rn", rows, w[local]) + delta).chunk(2, -1)
    oracle = g / (1 + torch.exp(-g)) * u
    fused_error = (fused.double() - oracle).norm() / oracle.norm()
    chain_error = (chain.double() - oracle).norm() / oracle.norm()
    assert fused_error <= chain_error * 1.05, (fused_error.item(), chain_error.item())
