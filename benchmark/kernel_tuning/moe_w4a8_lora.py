# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""W4A8 routed experts + per-expert LoRA at DSV4 Flash shapes: fused CuTe vs the separate form.

Baseline (the competitor-equivalent): FlashInfer W4A8 without LoRA, plus a separate LoRA
stage (gate/up and down LoRA on BF16 caches, the CuTe moe.lora kernels). Ours: the CuTe W4A8
kernels with the LoRA deltas inside FC1/FC2. Same card, CUDA graphs, A-B-A order, cold L2
(a read flush) unless --warm. One JSON row per (M, rank), speedup = baseline / ours.

    python benchmark/kernel_tuning/moe_w4a8_lora.py --ms 1 --ranks 16 --ep 2
"""

import argparse
import json

import torch

from benchmark.kernel_tuning.moe_w4a8 import EXPERTS, INTER, TOP_K, H, routes, time_graph, weights
from benchmark.sol.sol import Call, Kernel, estimate

SOL_DEVICES = {"NVIDIA GB10": "gb10", "RTX 5090": "rtx5090", "RTX PRO 6000": "rtx-pro-6000"}


def lora_weights(local, rank, seed=4):
    from ctikernels.moe.lora import MoELoRAWeights

    g = torch.Generator(device="cuda").manual_seed(seed)

    def a(*shape):
        return ((torch.rand(shape, generator=g, device="cuda") * 2 - 1) * shape[-1] ** -0.5).to(
            torch.bfloat16
        )

    def b(*shape):
        return (torch.randn(shape, generator=g, device="cuda") * 0.5 * rank**-0.5).to(
            torch.bfloat16
        )

    return MoELoRAWeights.separate(
        a(local, rank, H),
        a(local, rank, H),
        b(local, INTER, rank),
        b(local, INTER, rank),
        a(local, rank, INTER),
        b(local, H, rank),
    )


def sol(m, ids, local, rank):
    """Deterministic floor (sol.py, the SOL ceilings): touched experts' MXFP4 weights
    and scales, the touched LoRA leaves, x and out, and the BF16 per-route y written and read
    once (compulsory for the slot-ordered combine); FC1/FC2 on the mixed
    mxf8f6f4 pipe, the LoRA terms on BF16."""
    name = torch.cuda.get_device_name()
    device = next((key for prefix, key in SOL_DEVICES.items() if prefix in name), None)
    if device is None:
        return None
    flat = ids.flatten()
    routes_local = int(((flat >= 0) & (flat < local)).sum())
    experts = int(flat[(flat >= 0) & (flat < local)].unique().numel())
    per_expert = 2 * INTER * H // 2 + H * INTER // 2  # E2M1 bytes of w13 and w2
    leaves = 2 * rank * (2 * H + 2 * INTER + INTER + H)  # six BF16 leaves per expert
    call = Call(
        [
            Kernel(
                "w4a8_lora",
                dram_bytes={
                    "weights": experts * (per_expert + leaves),
                    "scales": experts * per_expert // 16,
                    "activations": m * H * 2,
                    "outputs": m * H * 2,
                    "intermediates": 2 * routes_local * H * 2,
                },
                flops={
                    "mxf8f6f4_e2m1_e4m3_m16n8k32": routes_local * 2 * (2 * INTER * H + H * INTER),
                    "bf16_m16n8k16": routes_local * 2 * rank * (2 * H + 2 * INTER + INTER + H),
                },
            )
        ]
    )
    result = estimate(call, device)
    return {
        "us": result.t_ref_us,
        "limit": result.limiting_resource,
        "routes": routes_local,
        "experts": experts,
    }


def row(m, rank, ep, cold, breakdown=False):
    from ctikernels.moe import MXFP4Experts, moe_align_block_size_local, to_local_topk_ids
    from ctikernels.moe.lora import moe_lora_w2_, moe_lora_w13_

    local = EXPERTS // ep
    w = weights(local)
    rw, ids = routes(m)
    expert_map = torch.full((EXPERTS,), -1, dtype=torch.int32, device="cuda")
    expert_map[:local] = torch.arange(local, dtype=torch.int32, device="cuda")
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    lora = lora_weights(local, rank)
    out_a, out_b = torch.empty_like(x), torch.empty_like(x)
    fused = MXFP4Experts(*w, swiglu_limit=10.0, scales_layout="canonical")
    # Each op takes over (rewrites in place) its tensors.
    base = MXFP4Experts(
        *[t.clone() for t in w], swiglu_limit=10.0, provider="flashinfer", scales_layout="canonical"
    )
    local_ids = to_local_topk_ids(ids, expert_map)
    cache13 = torch.zeros(m * TOP_K, 2 * INTER, dtype=torch.bfloat16, device="cuda")
    cache2 = torch.zeros(m * TOP_K, INTER, dtype=torch.bfloat16, device="cuda")
    cache3 = torch.zeros(m * TOP_K, H, dtype=torch.bfloat16, device="cuda")

    def ours():
        fused(x, rw, ids, expert_map=expert_map, out=out_a, lora=lora, scaling=2.0)

    def lora_stage(block_m):
        # The engine aligns the local routes once per step (capture needs it passed in).
        aligned = moe_align_block_size_local(ids, block_m, expert_map, local)
        names = ("sorted_token_ids", "expert_ids", "num_tokens_post_padded")
        kw = dict(zip(names, aligned, strict=True))
        kw.update(block_m=block_m, scaling=2.0, top_k=TOP_K)
        moe_lora_w13_(cache13, x, lora, local_ids, **kw)
        moe_lora_w2_(cache3, cache2, lora, local_ids, rw, **kw)

    def separate_with(block_m):
        def separate():
            base(x, rw, ids, expert_map=expert_map, out=out_b)
            lora_stage(block_m)

        return separate

    # Strongest form of the baseline: its fastest alignment block per row.
    block_m = min(
        (16, 32, 64), key=lambda b: time_graph(separate_with(b), cold=cold, samples=5)["us"]
    )
    separate = separate_with(block_m)
    first = time_graph(separate, cold=cold)
    mine = time_graph(ours, cold=cold)
    last = time_graph(separate, cold=cold)
    baseline = (first["us"] + last["us"]) / 2
    parts = {}
    if breakdown:
        from ctikernels.moe._lora_weights import FactorSlices
        from ctikernels.moe.cute import _lora_provider

        cute = fused._w4a8_lora
        z13 = torch.zeros(m * TOP_K, 2, rank, device="cuda")
        stages = {
            "cute_base": lambda: super(type(cute), cute).__call__(x, local_ids, rw, out=out_a),
            "flashinfer_base": lambda: base(x, rw, ids, expert_map=expert_map, out=out_b),
            "lora_stage": lambda: lora_stage(block_m),
            "shrink13": lambda: _lora_provider.shrink_(
                z13, x, FactorSlices((lora.gate_a, lora.up_a)), local_ids, TOP_K
            ),
        }
        parts = {name: time_graph(fn, cold=cold)["us"] for name, fn in stages.items()}
    touched = int(((ids >= 0) & (ids < local)).any(0).sum()) if m == 1 else None
    floor = sol(m, ids, local, rank)
    return {
        "M": m,
        "rank": rank,
        "ep": ep,
        "cold": cold,
        "device": torch.cuda.get_device_name(),
        "ours_us": mine["us"],
        "baseline_us": [first["us"], last["us"]],
        "baseline_block_m": block_m,
        "speedup": baseline / mine["us"],
        "local_experts_touched": touched,
        "parts_us": parts,
        "sol": floor,
        "sol_fraction": None if floor is None else floor["us"] / mine["us"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", type=int, nargs="+", default=[1])
    parser.add_argument("--ranks", type=int, nargs="+", default=[16])
    parser.add_argument("--ep", type=int, choices=(1, 2), default=2)
    parser.add_argument("--breakdown", action="store_true", help="time each stage too")
    parser.add_argument("--warm", action="store_true", help="keep L2 warm between calls")
    args = parser.parse_args(argv)
    for m in args.ms:
        for rank in args.ranks:
            result = row(m, rank, args.ep, cold=not args.warm, breakdown=args.breakdown)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
