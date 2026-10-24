# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
"""Paired Qwen3-30B-A3B BF16 routed experts, one EP partial without collectives.

Routing is prepared once outside timing. Every provider consumes the same FP32
weights and global int32 IDs. CUDA graphs follow the block market's deployed
limits; JSONL uses its provider/summary record style. No shared expert is present.
"""

from __future__ import annotations

import argparse
import gc
import math
import random
import statistics
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__:
    from . import moe_block_market as common
    from .moe_triton_tune import expert_mapping
else:
    import moe_block_market as common
    from moe_triton_tune import expert_mapping

H, E, INTER, TOP_K = 2048, 128, 768, 8
TOKENS = (1, 2, 3, 4, 5, 8, 9, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PROVIDERS = ("ours", "sglang", "vllm", "vllm:graph")


def row_tokens(rows, explicit=None):
    if rows not in ("decode", "prefill", "all"):
        raise ValueError("unknown row preset")
    tokens = (
        sorted({int(t) for t in explicit.split(",")})
        if explicit is not None
        else [t for t in TOKENS if rows == "all" or (t <= 64 if rows == "decode" else t >= 128)]
    )
    if not tokens or tokens[0] <= 0:
        raise ValueError("token counts must be positive")
    return tokens


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--sglang-path", type=Path)
    parser.add_argument("--ep", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rank", type=int, help="default: measure all ranks separately")
    parser.add_argument("--ep-map", choices=("auto", "always"), default="auto")
    parser.add_argument("--norm-topk-prob", choices=("true", "false", "both"), default="true")
    parser.add_argument("--rows", choices=("decode", "prefill", "all"), default="all")
    parser.add_argument("--t", help="explicit comma-separated token counts, overrides --rows")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true", help="check both normalization modes")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--allow-fi-version", action="store_true")
    args = parser.parse_args(argv)
    args.providers = list(dict.fromkeys(args.providers.split(",")))
    if any(p not in PROVIDERS for p in args.providers):
        parser.error("unknown provider")
    try:
        args.tokens = row_tokens(args.rows, args.t)
    except ValueError:
        parser.error("--t requires positive comma-separated integers")
    if args.rank is not None and not 0 <= args.rank < args.ep:
        parser.error("rank must be in [0, ep)")
    if args.rounds < 1 or (args.layers is not None and args.layers < 1):
        parser.error("layers and rounds must be positive")
    args.norms = (
        [True, False]
        if args.check_only or args.norm_topk_prob == "both"
        else [args.norm_topk_prob == "true"]
    )
    return args


@dataclass
class Weights:
    router: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    expert_map: torch.Tensor
    up_gate: torch.Tensor | None = None


@dataclass
class Inputs:
    x: torch.Tensor
    scales: torch.Tensor
    ids: torch.Tensor


def routing(x, router, norm_topk_prob):
    """HF Qwen3 router through top-k; retain FP32 scales for the experts API."""
    logits = F.linear(x, router.to(x.dtype))
    scales, ids = logits.softmax(-1, dtype=torch.float32).topk(TOP_K, dim=-1)
    if norm_topk_prob:
        scales = scales / scales.sum(-1, keepdim=True)
    return scales, ids.to(torch.int32)


def make_weights(args, rank, need_vllm):
    local = E // args.ep
    result = []
    for layer in range(args.layers):
        gen = torch.Generator(device="cuda")

        def normal(shape, gen=gen):
            return torch.randn(shape, generator=gen, device="cuda").mul_(0.02).bfloat16()

        gen.manual_seed(args.seed + 1009 * layer)
        router = normal((E, H))
        gu = torch.empty((local, 2 * INTER, H), device="cuda", dtype=torch.bfloat16)
        down = torch.empty((local, H, INTER), device="cuda", dtype=torch.bfloat16)
        # Global-expert seeds make EP1 and every EP partition the same model.
        for slot in range(local):
            gen.manual_seed(args.seed + 1009 * layer + 1000003 * (rank * local + slot + 1))
            gu[slot].copy_(normal((2 * INTER, H)))
            down[slot].copy_(normal((H, INTER)))
        up_gate = torch.cat((gu[:, INTER:], gu[:, :INTER]), dim=1) if need_vllm else None
        result.append(Weights(router, gu, down, expert_mapping(local, E, rank, "cuda"), up_gate))
    return result


def make_inputs(t, k, seed, weights, norm):
    xs = common.make_inputs(t, k, seed)
    return [
        Inputs(x, *routing(x, weights[j % len(weights)].router, norm)) for j, x in enumerate(xs)
    ]


def reference(inputs, w):
    """FP32 expert loop on BF16 weights, conditioned on the shared routing."""
    x, ids, scales = inputs.x.float(), inputs.ids, inputs.scales
    local = torch.full_like(ids, -1)
    valid = ids >= 0
    local[valid] = w.expert_map[ids[valid].long()]
    out = torch.zeros_like(x)
    for expert in local.unique().tolist():
        if expert < 0:
            continue
        rows, slots = torch.where(local == expert)
        gate, up = F.linear(x[rows], w.gate_up[expert].float()).chunk(2, -1)
        y = F.linear(F.silu(gate) * up, w.down[expert].float())
        out.index_add_(0, rows, y * scales[rows, slots, None])
    return out


def work_counts(ids, mapping, hidden=H, inter=INTER):
    local = torch.full_like(ids, -1)
    valid = ids >= 0
    local[valid] = mapping[ids[valid].long()]
    hits = local[local >= 0]
    experts, routes = hits.unique().numel(), hits.numel()
    return {
        "bytes": experts * 3 * hidden * inter * 2 + 4 * ids.shape[0] * hidden,
        "flops": routes * 2 * 3 * hidden * inter,
        "unique_local_experts_mean": experts,
        "local_routes_mean": routes,
    }


def sol(inputs, weights, roof):
    counts = [
        work_counts(x.ids, weights[j % len(weights)].expert_map) for j, x in enumerate(inputs)
    ]
    notes = {key: statistics.mean(c[key] for c in counts) for key in counts[0]}
    floor = (
        max(notes["bytes"] / roof["read_bytes_per_s"], notes["flops"] / roof["bf16_flops_per_s"])
        * 1e6
    )
    notes["flops_convention"] = "local routes only; bytes include hit experts and BF16 input/output"
    return floor, notes


class Experts:
    def __init__(self, name, args, rank, t, sgl=None, fi=None):
        self.name, self.args, self.rank, self.sgl, self.fi = name, args, rank, sgl, fi
        self.graph_mode = (
            name in ("ours", "vllm:graph")
            or (name == "sglang" and t <= 2048)
            or (name == "vllm" and t <= 512)
        )
        self.op = None
        if name == "ours":
            from ctikernels.moe import Experts as Op

            self.op = Op(provider="auto")

    def prepare(self, inputs):
        return (torch.empty_like(inputs.x),)

    def __call__(self, inputs, w, slot):
        x, scales, ids = inputs.x, inputs.scales, inputs.ids
        if self.op is not None:
            mapping = w.expert_map if self.args.ep > 1 or self.args.ep_map == "always" else None
            return self.op(x, w.gate_up, w.down, scales, ids, expert_map=mapping, out=slot[0])
        if self.name == "sglang":
            # SGLang 0.5.22's standard dispatcher translates global IDs with its
            # local-expert map (token_dispatcher/standard.py) and its Triton runner then
            # filters -1 routes (moe_runner/triton.py); importing those modules pulls in
            # Transformers, so the same two steps run here on SGLang's own kernels.
            if self.args.ep > 1:
                ids = w.expert_map[ids]
            return self.sgl.fused.fused_experts_impl(
                x,
                w.gate_up,
                w.down,
                scales,
                ids,
                inplace=False,
                activation="silu",
                is_gated=True,
                filter_expert=self.args.ep > 1,
            )
        self.fi.fused(
            input=x,
            token_selected_experts=ids,
            token_final_scales=scales,
            fc1_expert_weights=w.up_gate,
            fc2_expert_weights=w.down,
            output_dtype=torch.bfloat16,
            quant_scales=[],
            output=slot[0],
            tp_size=1,
            tp_rank=0,
            ep_size=self.args.ep,
            ep_rank=self.rank,
            activation_type=self.fi.activation.Swiglu,
        )
        return slot[0]


def tune(fi, args, weights, rank):
    inputs = make_inputs(max(args.tokens), len(weights), args.seed + 71, weights, args.norms[0])
    block = Experts("vllm", args, rank, max(args.tokens), fi=fi)
    slots = [block.prepare(x) for x in inputs]
    fi.tune_calls(
        args,
        [
            lambda x=x, w=w, slot=slot: block(x, w, slot)
            for x, w, slot in zip(inputs, weights, slots, strict=True)
        ],
    )


def import_checks(args, reporter):
    active, sgl, fi = list(args.providers), None, None
    reporter.emit(
        {
            "type": "header",
            "model": "Qwen3-30B-A3B",
            "providers": active,
            "torch": torch.__version__,
            "ep": args.ep,
            "ep_map": args.ep_map,
            "seed": args.seed,
            "tokens": args.tokens,
            "layers": args.layers,
            "rounds": args.rounds,
            "norm_topk_prob": args.norms,
            "error_metric": "max absolute error / FP32 reference row RMS",
            "error_bound": 0.03,
        }
    )
    for name in ("ours", "sglang", "vllm"):
        if name not in active and not (name == "vllm" and "vllm:graph" in active):
            continue
        try:
            if name == "ours":
                from ctikernels.moe import Experts as Op

                Op(provider="auto")
            elif name == "sglang":
                if args.sglang_path is None:
                    raise ValueError("--sglang-path is required")
                sgl = common.load_sglang(args.sglang_path)
                with common.sglang_topology(sgl, args.ep, args.rank or 0):
                    assert not sgl.runtime.get_exec().moe.enable_fused_moe_sum_all_reduce
                    Experts(name, args, args.rank or 0, 1, sgl=sgl)
            else:
                fi = common.FlashInferBackend(args, precomputed_routing=True)
            reporter.emit(
                {
                    "type": "imports",
                    "provider": name,
                    "status": "ok",
                    "version": sgl.version
                    if name == "sglang"
                    else fi.version
                    if name == "vllm"
                    else None,
                }
            )
        except Exception as exc:
            for p in list(active):
                if p == name or (name == "vllm" and p == "vllm:graph"):
                    reporter.fail(p, "import/topology", exc)
                    active.remove(p)
    return sgl, fi, active


def check(blocks, inputs, weights, reporter, row):
    errors, outputs = {}, {}
    for j, w in enumerate(weights):
        target = reference(inputs[j], w)
        rms = target.square().mean(-1).sqrt().clamp_min(1e-12)
        for name, block in blocks.items():
            if name in errors and errors[name] is None:
                continue
            try:
                actual = block(inputs[j], w, block.prepare(inputs[j])).float()
                error = float(((actual - target).abs().amax(-1) / rms).max())
                errors[name] = max(errors.get(name, 0.0), error) if math.isfinite(error) else None
                if math.isfinite(error):
                    outputs.setdefault(name, []).append(actual)
            except Exception as exc:
                reporter.fail(name, "correctness", exc)
                errors[name] = None
    for name, error in errors.items():
        reporter.emit(
            {
                "type": "check",
                **row,
                "provider": name,
                "max_rel_err": error,
                "launch_config": blocks[name].op.launch_config if blocks[name].op else None,
                "row_flagged": error is None or error > 0.03,
            }
        )
    return errors, outputs


def run_row(args, rank, t, norm, weights, sgl, fi, roof, reporter, active, inputs=None):
    row = {
        "device": torch.cuda.get_device_name(),
        "ep": args.ep,
        "rank": rank,
        "T": t,
        "norm_topk_prob": norm,
    }
    seed = args.seed + 7919 * t
    inputs = inputs if inputs is not None else make_inputs(t, args.layers, seed, weights, norm)
    blocks, unavailable = {}, {}
    sconfig = None
    for name in active:
        try:
            if name == "sglang":
                sconfig = common.sglang_config(sgl, t, weights[0])
            blocks[name] = Experts(name, args, rank, t, sgl, fi)
        except Exception as exc:
            reporter.fail(name, "construction/config", exc)
            unavailable[name] = "construction/config failed"
    errors, outputs = check(blocks, inputs, weights, reporter, row)
    if args.check_only:
        return outputs
    # Only check-only needs to retain the partials for the EP sum. Large prefill
    # graphs need their pools, not extra FP32 copies of all provider outputs.
    outputs.clear()
    for name in list(blocks):
        if errors.get(name) is None:
            del blocks[name]
            unavailable[name] = "correctness execution failed"
    k = args.layers
    loops = {}
    while blocks:
        durations = {}
        for name, block in list(blocks.items()):
            try:
                loops[name] = common.TimedLoop(block, inputs, weights)
                durations[name] = min(loops[name].time() for _ in range(3))
            except Exception as exc:
                reporter.fail(name, "warmup/capture", exc)
                unavailable[name] = "warmup/capture failed"
                blocks.pop(name)
                loops.pop(name, None)
        if not durations or min(durations.values()) >= 2000:
            break
        k = max(
            k + args.layers,
            math.ceil(k * 2200 / max(min(durations.values()), 1) / args.layers) * args.layers,
        )
        loops.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        inputs = make_inputs(t, k, seed, weights, norm)
    rounds = {name: [] for name in loops}
    rng = random.Random(seed + rank)
    for _ in range(args.rounds):
        order = list(loops)
        rng.shuffle(order)
        for name in order:
            try:
                rounds[name].append(loops[name].time() / k)
            except Exception as exc:
                reporter.fail(name, "timing", exc)
                unavailable[name] = "timing failed"
    rounds = {p: times for p, times in rounds.items() if len(times) == args.rounds}
    medians = {p: statistics.median(times) for p, times in rounds.items()}
    floor, sol_notes = sol(inputs, weights, roof)
    bad = [p for p, error in errors.items() if error is None or error > 0.03]
    config = blocks["ours"].op.launch_config if "ours" in blocks else None
    for name in args.providers:
        reporter.emit(
            {
                "type": "provider",
                **row,
                "provider": name,
                "median_us": medians.get(name),
                "min_us": min(rounds[name]) if name in rounds else None,
                "rounds_us": rounds.get(name, []),
                "sol_us": floor,
                "pct_sol": 100 * floor / medians[name] if name in medians else None,
                "max_rel_err": errors.get(name),
                "config": config if name == "ours" else sconfig if name == "sglang" else None,
                "notes": {
                    "K": k,
                    "layers": args.layers,
                    "sol": sol_notes,
                    "timing": "graph" if name in blocks and blocks[name].graph_mode else "eager",
                    "status": "ok" if name in rounds else unavailable.get(name, "unavailable"),
                    "row_flagged": bool(bad),
                    "sglang_config": sconfig,
                    "autotune_ran": fi.tuned if fi else False,
                    "tuning_buckets": fi.tuning_buckets if fi else None,
                    "autotune_seconds": fi.tune_seconds if fi else None,
                    "flashinfer_version": fi.version if fi else None,
                    "sglang_version": sgl.version if sgl else None,
                    "selected_provider": str(blocks[name].op.provider)
                    if name in blocks and blocks[name].op
                    else None,
                },
            }
        )
    markets = {p: medians[p] for p in common.MARKETS if p in medians}
    best = min(markets, key=markets.get) if markets else None
    ours = medians.get("ours")
    paired = {"ours:auto" if p == "ours" else p: times for p, times in rounds.items()}
    lo, hi = common.bootstrap(paired, seed + rank)
    ratio = markets[best] / ours if best and ours else None
    reporter.emit(
        {
            "type": "summary",
            **row,
            "ours_auto_us": ours,
            "market_best_name": best,
            "market_best_us": markets.get(best),
            "ratio": ratio,
            "ci_lo": lo,
            "ci_hi": hi,
            "row_flagged": bool(bad),
            "market_complete": all(p in rounds for p in common.MARKETS),
            "launch_config": config,
        }
    )
    cells = " ".join(f"{p}={medians[p]:.2f}us" if p in medians else f"{p}=n/a" for p in PROVIDERS)
    pct = f"{100 * floor / ours:.1f}%" if ours else "n/a"
    print(
        f"ROW ep={args.ep} rank={rank} T={t} norm={norm} {cells} market={best} ratio={ratio} CI95=[{lo},{hi}] SOL={pct} config={config}"
        + (" FLAG" if bad else ""),
        flush=True,
    )
    return outputs


def partition(weights, ep, rank):
    local = E // ep
    sl = slice(rank * local, (rank + 1) * local)
    return [
        replace(
            w,
            gate_up=w.gate_up[sl],
            down=w.down[sl],
            up_gate=w.up_gate[sl] if w.up_gate is not None else None,
            expert_map=expert_mapping(local, E, rank, w.router.device),
        )
        for w in weights
    ]


def check_all_ranks(args, sgl, fi, reporter, active):
    full_args = argparse.Namespace(**vars(args))
    full_args.ep = 1
    weights = make_weights(full_args, 0, any(p.startswith("vllm") for p in active))
    for t in args.tokens:
        for norm in args.norms:
            inputs = make_inputs(t, args.layers, args.seed + 7919 * t, weights, norm)
            totals, scales, counts = {}, {}, {}
            for rank in range(args.ep):
                topology = (
                    common.sglang_topology(sgl, args.ep, rank)
                    if "sglang" in active
                    else nullcontext()
                )
                with topology:
                    outputs = run_row(
                        args,
                        rank,
                        t,
                        norm,
                        partition(weights, args.ep, rank),
                        sgl,
                        fi,
                        None,
                        reporter,
                        active,
                        inputs,
                    )
                for name, values in outputs.items():
                    if len(values) != args.layers:
                        continue
                    stacked = torch.stack(values)
                    totals[name] = totals.get(name, 0) + stacked
                    scales[name] = max(scales.get(name, 0), float(stacked.abs().max()))
                    counts[name] = counts.get(name, 0) + 1
            topology = common.sglang_topology(sgl, 1, 0) if "sglang" in active else nullcontext()
            with topology:
                for name in active:
                    try:
                        block = Experts(name, full_args, 0, t, sgl, fi)
                        full = torch.stack(
                            [
                                block(x, w, block.prepare(x)).float()
                                for x, w in zip(inputs, weights, strict=True)
                            ]
                        )
                        if counts.get(name) != args.ep:
                            raise RuntimeError("incomplete EP partials")
                        # Each partial and the full output rounds to BF16 once.
                        bound = (args.ep + 1) * 2.0**-8 * max(scales[name], float(full.abs().max()))
                        error = float((totals[name] - full).abs().max())
                        reporter.emit(
                            {
                                "type": "ep_sum",
                                "provider": name,
                                "ep": args.ep,
                                "T": t,
                                "norm_topk_prob": norm,
                                "max_abs_err": error,
                                "bf16_bound": bound,
                                "row_flagged": error > bound,
                            }
                        )
                    except Exception as exc:
                        reporter.fail(name, "EP sum", exc)
                        reporter.emit(
                            {
                                "type": "ep_sum",
                                "provider": name,
                                "ep": args.ep,
                                "T": t,
                                "norm_topk_prob": norm,
                                "status": "unavailable",
                                "row_flagged": True,
                            }
                        )


def main(argv=None):
    args = parse_args(argv)
    reporter = common.Reporter(args.out)
    try:
        sgl, fi, active = import_checks(args, reporter)
        if args.check_imports:
            return
        if not active:
            return
        if not torch.cuda.is_available():
            print("UNAVAILABLE CUDA no visible CUDA device", flush=True)
            return
        torch.backends.cuda.matmul.allow_tf32 = False
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        l2 = getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0))
        if args.layers is None:
            args.layers = max(3, math.floor(2 * l2 / ((E // args.ep) * 3 * H * INTER * 2)) + 1)
        reporter.emit(
            {
                "type": "weight_set",
                "layers": args.layers,
                "local_weight_bytes": args.layers * (E // args.ep) * 3 * H * INTER * 2,
                "l2_bytes": l2 or None,
            }
        )
        with torch.inference_mode():
            if args.check_only:
                check_all_ranks(args, sgl, fi, reporter, active)
                return
            roof = common.ceilings()
            reporter.emit({"type": "ceilings", "device": torch.cuda.get_device_name(), **roof})
            for rank in [args.rank] if args.rank is not None else range(args.ep):
                weights = make_weights(args, rank, any(p.startswith("vllm") for p in active))
                if fi is not None and any(p.startswith("vllm") for p in active):
                    try:
                        tune(fi, args, weights, rank)
                    except Exception as exc:
                        for p in list(active):
                            if p.startswith("vllm"):
                                reporter.fail(p, "autotune", exc)
                                active.remove(p)
                topology = (
                    common.sglang_topology(sgl, args.ep, rank)
                    if "sglang" in active
                    else nullcontext()
                )
                with topology:
                    for t in args.tokens:
                        for norm in args.norms:
                            run_row(args, rank, t, norm, weights, sgl, fi, roof, reporter, active)
                            gc.collect()
                            torch.cuda.empty_cache()
                del weights
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
