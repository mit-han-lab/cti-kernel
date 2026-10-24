# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""End-to-end block-FP8 MoE accuracy: the shipped forward versus the original Triton chain.

Both chains see identical operands (BF16 input, FP8 weights and block scales, routing) and
are compared against an FP64 reference of the routed MoE (FC1 -> SiLU*up -> FC2 -> weighted
route sum) on max and RMS error. The original chain is the #28 composition: alignment, the
Triton FP8 group quantizer, the split SiLU*up and the default block_fp8_moe_config launch.
The shared expert is outside this operator, so it is outside this gate.
"""

import pytest
import torch

from ctikernels.activation.inference import silu_and_mul
from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
from ctikernels.moe import moe_align_block_size_local
from ctikernels.moe_block_fp8 import BlockFP8MoE
from ctikernels.moe_impl import block_fp8_moe_config, fused_moe_kernel_triton, moe_sum_reduce_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires compute capability 8.0 or newer",
)

E, H, INTERMEDIATE, TOP_K = 256, 2048, 512, 8
SEEDS = (11, 23, 37, 41)
PERMUTATIONS = 8


@pytest.fixture(scope="module")
def params():
    torch.manual_seed(7)
    tensors = []
    for n, k in ((2 * INTERMEDIATE, H), (H, INTERMEDIATE)):
        w = (torch.randn(E, n, k, device="cuda") * 16).to(torch.float8_e4m3fn)
        s = torch.exp2(torch.rand(E, n // 128, k // 128, device="cuda") * 2 - 11)
        tensors.extend((w, s.bfloat16()))
    return tuple(tensors)


def dequantize(q, s):
    return q.double() * s.double().repeat_interleave(128, -2).repeat_interleave(128, -1)


def fp64_reference(x, weights, ids, params):
    """Routed MoE in FP64 on the exact operands; -1 routes contribute nothing."""
    w13, s13, w2, s2 = params
    flat = ids.flatten()
    out = torch.zeros(x.shape[0], H, dtype=torch.float64, device=x.device)
    for e in flat.unique().tolist():
        if e < 0:
            continue
        routes = (flat == e).nonzero().flatten()
        tokens = routes // TOP_K
        gate, up = (x[tokens].double() @ dequantize(w13[e], s13[e]).T).chunk(2, -1)
        h = torch.nn.functional.silu(gate) * up
        down = (h @ dequantize(w2[e], s2[e]).T) * weights.flatten()[routes, None].double()
        out.index_add_(0, tokens, down)
    return out


def original_chain(x, weights, ids, params):
    """The #28 Triton chain: align, quantize, FC1, split SiLU*up, quantize, FC2, reduce."""
    w13, s13, w2, s2 = params
    m = x.shape[0]
    # Masked routes become expert 0 with zero routing weight: identical contribution (zero).
    weights = torch.where(ids >= 0, weights, 0.0).contiguous()
    ids = ids.clamp_min(0).contiguous()
    config = block_fp8_moe_config(
        m, E, 2 * INTERMEDIATE, H, TOP_K, capability=torch.cuda.get_device_capability()
    )
    identity = torch.arange(E, dtype=torch.int32, device=x.device)
    sorted_ids, experts, padded = moe_align_block_size_local(
        ids, config["BLOCK_SIZE_M"], identity, E
    )
    gate_up = torch.empty((m, TOP_K, 2 * INTERMEDIATE), dtype=x.dtype, device=x.device)
    act = torch.empty((m * TOP_K, INTERMEDIATE), dtype=x.dtype, device=x.device)
    down = torch.empty((m, TOP_K, H), dtype=x.dtype, device=x.device)
    for second, (source, target, weight, scale) in enumerate(
        ((x, gate_up, w13, s13), (act, down, w2, s2))
    ):
        if second:
            silu_and_mul(gate_up.view(-1, 2 * INTERMEDIATE), act)
        q, qs = quantize_per_token_group_fp8(source)
        fused_moe_kernel_triton(
            q,
            weight,
            target,
            weights,
            ids,
            sorted_ids,
            experts,
            padded,
            False,
            1 if second else TOP_K,
            config,
            x.dtype,
            a_scale=qs,
            b_scale=scale,
            block_fp8_provider="triton-fp8",
        )
    out = torch.empty_like(x)
    moe_sum_reduce_triton(down, out, topk_ids=ids, topk_weights=weights)
    return out


def case(m, skewed, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
    ids = torch.randint(E, (m, TOP_K), device="cuda", dtype=torch.int32, generator=g)
    if skewed:
        ids[:, :6] %= 4
    ids.flatten()[seed % 7 :: 11] = -1
    weights = torch.rand(m, TOP_K, device="cuda", generator=g)
    return x, weights / weights.sum(-1, keepdim=True), ids


def errors(actual, reference):
    d = actual.double() - reference
    return d.abs().max().item(), d.pow(2).mean().sqrt().item()


def triton_noise(x, weights, ids, params, base_errors, reference):
    """Relative spread of the original chain's own error under top-k slot permutations.

    Permuting the top-k slots of (ids, weights) is the same MoE on identical operands; only
    the alignment order and the route-sum order change. The spread of the chain's max and RMS
    error over these equivalent orders is the comparison noise floor (BF16 ties).
    """
    samples = [base_errors]
    g = torch.Generator(device="cuda").manual_seed(1234)
    for _ in range(PERMUTATIONS):
        perm = torch.randperm(TOP_K, device="cuda", generator=g)
        out = original_chain(x, weights[:, perm].contiguous(), ids[:, perm].contiguous(), params)
        samples.append(errors(out, reference))
    t = torch.tensor(samples, dtype=torch.float64)
    mean = t.mean(0)
    # A rank with no local routes is exact (zero error) in every order: zero spread.
    return torch.where(mean > 0, t.std(0) / mean.clamp_min(1e-300), 0.0).tolist()


@pytest.mark.parametrize("ep", [1, 2])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 64])
@pytest.mark.parametrize("skewed", [False, True])
@torch.inference_mode()
def test_end_to_end_no_worse_than_original_triton(params, ep, m, skewed):
    rows, spreads = [], []
    for seed in SEEDS:
        x, weights, ids = case(m, skewed, seed)
        for rank in range(ep):
            start, end = rank * E // ep, (rank + 1) * E // ep
            mapping = torch.full((E,), -1, dtype=torch.int32, device="cuda")
            mapping[start:end] = torch.arange(end - start, dtype=torch.int32, device="cuda")
            local = tuple(t[start:end] for t in params)
            op = BlockFP8MoE(*local, expert_map=mapping if ep > 1 else None, num_global_experts=E)
            ours = op(x, weights, ids)
            # The same rank's work on the full stack: remote routes carry zero routing weight.
            on_rank = (ids >= start) & (ids < end)
            rank_ids = torch.where(on_rank, ids, -1)
            base = original_chain(x, weights, rank_ids, params)
            reference = fp64_reference(x, weights, rank_ids, params)
            a, b = errors(ours, reference), errors(base, reference)
            rows.append((seed, rank, a, b))
            spreads.append(triton_noise(x, weights, rank_ids, params, b, reference))
    # tau = 3 sigma of the original chain's relative error spread, pooled (RMS) over seeds/ranks.
    s = torch.tensor(spreads, dtype=torch.float64)
    tau_max, tau_rms = (3 * s.pow(2).mean(0).sqrt()).tolist()
    for seed, rank, a, b in rows:
        print(
            f"E2E ep={ep} m={m} skewed={skewed} seed={seed} rank={rank} ours max={a[0]:.4e} rms={a[1]:.4e} triton max={b[0]:.4e} rms={b[1]:.4e} tau max={tau_max:.3e} rms={tau_rms:.3e}"
        )
    # Each seed and each rank: ours <= Triton * (1 + tau) on max and RMS.
    for seed, rank, a, b in rows:
        assert a[0] <= b[0] * (1 + tau_max), ("max", seed, rank, a, b, tau_max)
        assert a[1] <= b[1] * (1 + tau_rms), ("rms", seed, rank, a, b, tau_rms)
    # Pooled: the mean ratio over seeds and ranks is <= 1 on max and RMS.
    ratios = torch.tensor([(a[0], a[1], b[0], b[1]) for _, _, a, b in rows], dtype=torch.float64)
    ratios = torch.where(ratios[:, 2:] > 0, ratios[:, :2] / ratios[:, 2:].clamp_min(1e-300), 1.0)
    mean_max, mean_rms = ratios.mean(0).tolist()
    assert mean_max <= 1.0 and mean_rms <= 1.0, (mean_max, mean_rms)
