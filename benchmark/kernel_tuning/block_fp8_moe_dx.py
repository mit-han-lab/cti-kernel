# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Frozen expert dX: CuTe block-scale promotion versus BF16 scratch + autograd.

Run with PYTHONPATH=python. Routing indices for the baseline are grouped once
outside timing/capture; weight dequantization is inside each timed invocation.
"""

import argparse
import json
from functools import partial

import torch

from ctikernels._moe_block_fp8_dx import (
    _forward_cache,
    block_fp8_moe_dx,
    prepare_dx,
)
from ctikernels.moe_block_fp8 import BlockFP8MoE


def dequantize(q, scales, dtype):
    e, n, k = q.shape
    return (
        q.to(dtype).view(e, n // 128, 128, k // 128, 128) * scales.to(dtype)[:, :, None, :, None]
    ).reshape(e, n, k)


def quant_ste(x):
    """The group quantizer's forward value; identity derivative, including at zero."""
    groups = x.float().reshape(*x.shape[:-1], -1, 128)
    peak = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = peak * (1.0 / 448.0)
    q = (groups * (448.0 / peak)).clamp(-448, 448).to(torch.float8_e4m3fn)
    value = (q.float() * scale).reshape_as(x).to(x.dtype)
    return value.detach() + (x - x.detach())


def route_groups(op, ids):
    """Host grouping belongs to the baseline setup, never the production path."""
    local = ids.clone().long()
    if op.expert_map is not None:
        local = torch.where(ids >= 0, op.expert_map[ids.clamp_min(0).long()], -1)
    return [
        (e, (local.flatten() == e).nonzero().flatten()) for e in local.unique().tolist() if e >= 0
    ]


def served_point(op, x, weights, ids):
    """Capture actual quantizer values and the served BF16 FC1 boundary."""
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8

    def dequant(q, scale):
        return (q.double().reshape(*scale.shape, 128) * scale.double()[..., None]).reshape(q.shape)

    config = op.config_for(x.shape[0])
    xq = dequant(*quantize_per_token_group_fp8(x))
    gu, quantized, _ = _forward_cache(op, x, weights, ids, config, None)
    aq = dequant(*quantized)
    return xq, gu, aq


def sum_slots(routes, order=None):
    """Explicit FP32 reduction of baseline BF16 route derivatives."""
    acc = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    for slot in range(routes.shape[1]) if order is None else order:
        acc = acc + routes[:, slot].float()
    return acc.bfloat16()


def reference(
    op,
    x,
    weights,
    ids,
    dy,
    *,
    groups=None,
    exact=False,
    formula=False,
    point=None,
    return_routes=False,
):
    """FP64 oracle or the engine's BF16-scratch STE autograd baseline.

    formula=True independently evaluates the real-arithmetic derivative; use
    exact=True and point=served_point(...) for the served STE oracle. Omitting
    point gives the informational unquantized oracle. The baseline retains its
    own forward. return_routes exposes its BF16 route gradients for sum-order
    noise measurements without repeating autograd. No floating-point atomics.
    """
    if point is not None and not exact:
        raise ValueError("a served linearization point requires exact=True")
    if return_routes and (exact or formula):
        raise ValueError("return_routes requires the BF16 autograd baseline")
    if groups is None:
        groups = route_groups(op, ids)
    dtype = torch.float64 if exact else torch.bfloat16
    # Multiplication is FP32 even for a BF16 scale; round only its result.
    w13 = dequantize(op.w13, op.s13, torch.float64 if exact else torch.float32).to(dtype)
    w2 = dequantize(op.w2, op.s2, torch.float64 if exact else torch.float32).to(dtype)
    r = ids.numel()
    if formula:
        dxr = torch.zeros((r, op.h), dtype=dtype, device=x.device)
        dw = torch.zeros_like(weights, dtype=dtype).flatten()
        for e, routes in groups:
            tokens = routes // op.top_k
            g, u = (
                x[tokens].to(dtype) @ w13[e].T if point is None else point[1][routes].double()
            ).chunk(2, -1)
            sigmoid = torch.sigmoid(g)
            a = g * sigmoid * u
            down = (a if point is None else point[2][routes]) @ w2[e].T
            dd = weights.flatten()[routes, None].to(dtype) * dy[tokens].to(dtype)
            da = dd @ w2[e]
            dg = da * u * sigmoid * (1 + g * (1 - sigmoid))
            du = da * g * sigmoid
            dxr[routes] = torch.cat((dg, du), -1) @ w13[e]
            dw[routes] = (dy[tokens].to(dtype) * down).sum(-1)
        return dxr.view(x.shape[0], op.top_k, op.h).sum(1), dw.view_as(weights)
    with torch.enable_grad():
        # Independent route leaves avoid index_select backward's atomic token sum.
        xr = x[:, None, :].expand(-1, op.top_k, -1).reshape(r, op.h).to(dtype).detach()
        xr.requires_grad_(True)
        wr = weights.to(torch.float64 if exact else torch.float32).detach().requires_grad_(True)
        down = torch.zeros((r, op.h), dtype=dtype, device=x.device)
        for e, routes in groups:
            inp = xr[routes] if exact else quant_ste(xr[routes])
            if point is not None:
                inp = point[0][routes // op.top_k] + (inp - inp.detach())
            gu = inp @ w13[e].T
            if point is not None:
                gu = point[1][routes].double() + (gu - gu.detach())
            g, u = gu.chunk(2, -1)
            a = (
                (torch.nn.functional.silu(g.float()) * u.float()).to(dtype)
                if not exact
                else torch.nn.functional.silu(g) * u
            )
            if point is not None:
                a = point[2][routes] + (a - a.detach())
            down[routes] = (a if exact else quant_ste(a)) @ w2[e].T
        # Empty ranks still have well-defined zero derivatives.
        y = (down.to(wr.dtype).view(x.shape[0], op.top_k, op.h) * wr[:, :, None]).sum(1)
        loss = (y * dy.to(y.dtype)).sum() + xr.sum() * 0
        dr, dw = torch.autograd.grad(loss, (xr, wr))
        routes = dr.view(x.shape[0], op.top_k, op.h)
        if not exact:
            result = (sum_slots(routes), dw)
            return (result, routes) if return_routes else result
        acc = torch.zeros_like(x, dtype=torch.float64)
        for slot in range(op.top_k):
            acc = acc + dr.view(x.shape[0], op.top_k, op.h)[:, slot].to(acc.dtype)
        return acc.to(dtype), dw


def errors(actual, oracle):
    metrics = []
    for a, b in zip(actual, oracle, strict=True):
        delta = a.double() - b
        metrics.extend((delta.abs().max().item(), delta.square().mean().sqrt().item()))
    return torch.tensor(metrics, dtype=torch.float64)


def permutation_noise(op, args, oracle, base_error, permutations=8):
    """Relative sample sigma from slot permutations and FP32 sum orders."""
    samples = [base_error]
    x, weights, ids, dy = args
    generator = torch.Generator(device=x.device).manual_seed(1234)
    for _ in range(permutations):
        permutation = torch.randperm(op.top_k, device=x.device, generator=generator)
        result, routes = reference(
            op,
            x,
            weights[:, permutation].contiguous(),
            ids[:, permutation].contiguous(),
            dy,
            return_routes=True,
        )
        dw = result[1][:, permutation.argsort()]
        samples.append(errors((result[0], dw), oracle))
        order = torch.randperm(op.top_k, device=x.device, generator=generator).tolist()
        samples.append(errors((sum_slots(routes, order), dw), oracle))
    samples = torch.stack(samples)
    mean = samples.mean(0)
    varying = samples.amax(0) != samples.amin(0)
    return torch.where(varying, samples.std(0) / mean.clamp_min(1e-300), 0)


def error_ratios(actual, baseline):
    return torch.where(
        baseline > 0,
        actual / baseline.clamp_min(1e-300),
        torch.where(actual == 0, 1.0, float("inf")),
    )


def make_case(e, h, i, m, *, seed=11, skewed=False, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    params = []
    for n, k in ((2 * i, h), (h, i)):
        q = (torch.randn(e, n, k, device=device, generator=generator) * 16).to(torch.float8_e4m3fn)
        s = torch.exp2(
            torch.rand(e, n // 128, k // 128, device=device, generator=generator) * 2 - 11
        ).bfloat16()
        params.extend((q, s))
    x = torch.randn(m, h, device=device, generator=generator).bfloat16()
    dy = torch.randn(m, h, device=device, generator=generator).bfloat16()
    ids = torch.randint(e, (m, 8), dtype=torch.int32, device=device, generator=generator)
    if skewed:
        ids[:, :6] %= 4
    ids.flatten()[seed % 7 :: 11] = -1
    weights = torch.rand(m, 8, device=device, generator=generator)
    weights /= weights.sum(-1, keepdim=True)
    weights.flatten()[::13] = 0
    return params, (x, weights, ids, dy)


def graph_time(call, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=("qwen35", "qwen3"), default="qwen35")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--skewed", action="store_true")
    parser.add_argument(
        "--accuracy", action="store_true", help="print served and unquantized errors"
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    e, h, i = (256, 2048, 512) if args.shape == "qwen35" else (128, 2048, 768)
    for m in args.batch_sizes:
        params, inputs = make_case(e, h, i, m, seed=args.seed, skewed=args.skewed)
        op = BlockFP8MoE(*params, top_k=8, provider="triton-fp8")
        groups = route_groups(op, inputs[2])
        sm12x = torch.cuda.get_device_capability() in ((12, 0), (12, 1))
        prepare_dx(op, [m], provider="triton")
        timed = {}
        # Same card A-B-A: CuTe (sm120/121 only), our Triton chain, CuTe again.
        for name in ("cute", "triton", "cute") if sm12x else ("triton",):
            if name == "cute":
                prepare_dx(op, [m], provider="cute")
            call = partial(block_fp8_moe_dx, op, *inputs, provider=name)
            us, actual = graph_time(call, args.repeats)
            timed.setdefault(name, []).append(us)
        ours = min(timed.get("cute", timed["triton"]))
        baseline, base = graph_time(partial(reference, op, *inputs, groups=groups), args.repeats)
        accuracy = {}
        if args.accuracy:
            point = served_point(op, *inputs[:3])
            oracle = reference(op, *inputs, exact=True, formula=True, point=point)
            unquantized = reference(op, *inputs, exact=True, formula=True)
            a, b = errors(actual, oracle), errors(base, oracle)
            tau = 3 * permutation_noise(op, inputs, oracle, b)
            accuracy = {
                "seed": args.seed,
                "metrics": ["dx_max", "dx_rms", "dw_max", "dw_rms"],
                "served_ours": a.tolist(),
                "served_baseline": b.tolist(),
                "tau": tau.tolist(),
                "per_seed_pass": bool(torch.all(a <= b * (1 + tau))),
                "unquantized_ours": errors(actual, unquantized).tolist(),
                "unquantized_baseline": errors(base, unquantized).tolist(),
                "output_rounding_floor": errors(
                    (oracle[0].bfloat16(), oracle[1].float()), oracle
                ).tolist(),
            }
        # Logical unique expert bytes, not measured DRAM traffic. Ours reads W2
        # only in da, W13 for FC1 and dx; shared/cache reuse is hardware-dependent.
        per_expert = 3 * h * i + (3 * h * i // (128 * 128)) * params[1].element_size()
        touched = len(groups) * per_expert
        print(
            json.dumps(
                {
                    "shape": args.shape,
                    "m": m,
                    "active_experts": len(groups),
                    "weight_bytes": touched,
                    "cute_us": min(timed["cute"]) if "cute" in timed else None,
                    "cute_aba": timed.get("cute"),
                    "triton_us": timed["triton"][0],
                    "cute_over_triton": (
                        min(timed["cute"]) / timed["triton"][0] if "cute" in timed else None
                    ),
                    "reference_us": baseline,
                    "cute_GBs": touched / ours / 1000,
                    "reference_GBs": touched / baseline / 1000,
                    "reference_dequant_bytes": e * per_expert,
                    **accuracy,
                }
            )
        )


if __name__ == "__main__":
    main()
