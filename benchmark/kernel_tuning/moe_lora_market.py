# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
# Failures are isolated by row/candidate and reported by type, without raw diagnostics.
"""Paired BF16 MoE LoRA benchmark for Qwen3.5-35B-A3B and Qwen3-30B-A3B.

Run with ``python -m benchmark.kernel_tuning.moe_lora_market --quick``.
The default Qwen3.5 model uses H2048/I512/E256/top-8, with shared rows.
Select --model qwen3-30b-a3b for H2048/I768/E128/top-8, routed only,
ranks 16/32/64 and tokens 1..32768 (powers of two). Explicit --ranks
accepts multiples of eight in 8..128; --tokens also permits 65536 and beyond.
The comparator executes the official wheel, including TritonExperts.apply.
EP2 reports local partial outputs, excluding collectives on both sides.
Use --routing {softmax,skewed,narrow} for routing patterns (default softmax).
Use --ep-rank {0,1} to select the local EP rank (default 0; must be < ep).
Shared rows compare gate/up and down LoRA deltas on fixed dense inputs.

Each graph contains a rotating pool of distinct activations AND routing choices.
Weights belong to one layer; routing rotates the touched experts. CUDA event
samples replay enough graphs to exceed 2.5ms. Candidates run interleaved in a
seeded shuffled order. The fastest vLLM schedule is selected by median latency;
paired ratios use that same schedule for every sample. The reported interval is
a percentile bootstrap confidence interval for the median paired speedup.

Routed rows also capture independent twins of both vLLM schedules up front.
Only one twin is timed per round, alternating modes in a seeded order, then
shuffled with every other candidate. A/A uses twin / best on the subset of
rounds matching the final best schedule (10 or 11 pairs at 21 samples), without
an extra timing pass. epsilon = max(abs(median_AA - 1), (CI_hi - CI_lo) / 2).
verdict_eps is PASS at ratio >= 1, TIE at 1 - epsilon <= ratio < 1, else FAIL.
Shared rows omit A/A. Use --tokens/--ranks to restrict the grid when needed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_RANKS = (8, 16, 32, 64, 128)
QUICK_TOKENS = (1, 4, 8, 64, 1024)
QUICK_RANKS = (16, 64, 128)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    shared: bool
    tokens: tuple[int, ...]
    ranks: tuple[int, ...]

    def fields(self, ep=1):
        return {
            "model": self.name,
            "H": self.hidden,
            "I": self.intermediate,
            "E": self.experts,
            "E_local": self.experts // ep,
            "top_k": self.top_k,
        }


DEFAULT_MODEL = ModelSpec("qwen3.5-35b-a3b", 2048, 512, 256, 8, True, DEFAULT_TOKENS, DEFAULT_RANKS)
QWEN3_MODEL = ModelSpec(
    "qwen3-30b-a3b", 2048, 768, 128, 8, False, (*DEFAULT_TOKENS, 16384, 32768), (16, 32, 64)
)
MODELS = {model.name: model for model in (DEFAULT_MODEL, QWEN3_MODEL)}


@dataclass(frozen=True)
class Row:
    tokens: int
    rank: int
    ep: int
    model: ModelSpec = DEFAULT_MODEL
    routing: str = "softmax"
    ep_rank: int = 0

    def fields(self):
        return {
            "T": self.tokens,
            "rank": self.rank,
            "ep": self.ep,
            "routing": self.routing,
            "ep_rank": self.ep_rank,
            **self.model.fields(self.ep),
            "dtype": "bfloat16",
        }


@dataclass(frozen=True)
class SharedRow:
    tokens: int
    rank: int
    intermediate: int
    model: ModelSpec = DEFAULT_MODEL
    routing: str = "softmax"
    ep_rank: int = 0

    def fields(self):
        return {
            "kind": "shared",
            "routing": self.routing,
            "ep_rank": self.ep_rank,
            "T": self.tokens,
            "rank": self.rank,
            **self.model.fields(),
            "Is": self.intermediate,
            "dtype": "bfloat16",
        }


def positive_list(value):
    try:
        values = tuple(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default=DEFAULT_MODEL.name)
    parser.add_argument("--tokens", type=positive_list)
    parser.add_argument("--ranks", type=positive_list)
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ep-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--routing", choices=("softmax", "skewed", "narrow"), default="softmax")
    parser.add_argument("--kinds", help="routed,shared or either kind")
    parser.add_argument("--shared-intermediate", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vllm-cache", type=Path, default=Path("/var/tmp/moe-lora-vllm-cache"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--sample-ms", type=float, default=3.0)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--scaling", type=float, default=0.5)
    parser.add_argument("--ours-torch", action="store_true", help="diagnostic for T <= 64")
    parser.add_argument(
        "--provider", choices=("cute", "triton"), default="cute", help="our MoE LoRA provider"
    )
    parser.add_argument(
        "--base",
        choices=("bf16", "fp8"),
        default="bf16",
        help="routed expert weights: BF16, or block FP8 (E4M3, 128x128 scales; W8A8 1x128)",
    )
    parser.add_argument(
        "--compare-provider",
        choices=("cute", "triton"),
        help="also time our layer on this provider (candidate ours_alt, ratio_vs_alt = alt / ours)",
    )
    parser.add_argument(
        "--sglang-python", type=Path, help="SGLang source tree (holds sglang/): adds SGLang rows"
    )
    args = parser.parse_args(argv)
    if args.ep_rank >= args.ep:
        parser.error("ep-rank must be < ep")
    model = MODELS[args.model]
    kinds = args.kinds if args.kinds is not None else "routed,shared" if model.shared else "routed"
    args.kinds = tuple(dict.fromkeys(kinds.split(",")))
    if any(kind not in ("routed", "shared") for kind in args.kinds):
        parser.error("kinds must be drawn from routed,shared")
    if "shared" in args.kinds and not model.shared:
        parser.error(f"{model.name} has no shared expert; use --kinds routed")
    if args.shared_intermediate < 1:
        parser.error("shared-intermediate must be positive")
    args.tokens = args.tokens or (QUICK_TOKENS if args.quick else model.tokens)
    args.ranks = args.ranks or (QUICK_RANKS if args.quick and model.shared else model.ranks)
    if args.provider == "cute" and any(not 8 <= rank <= 128 or rank % 8 for rank in args.ranks):
        parser.error("cute ranks must be multiples of 8 in [8, 128]")
    if args.samples < 5 or args.pool_size < 2:
        parser.error("need at least 5 samples and 2 distinct inputs")
    if not math.isfinite(args.sample_ms) or args.sample_ms < 2.5:
        parser.error("sample-ms must be finite and >= 2.5")
    if not math.isfinite(args.scaling):
        parser.error("scaling must be finite")
    return args


def rows(args):
    return [
        Row(t, r, args.ep, MODELS[args.model], args.routing, args.ep_rank)
        if kind == "routed"
        else SharedRow(
            t, r, args.shared_intermediate, MODELS[args.model], args.routing, args.ep_rank
        )
        for kind in args.kinds
        for t in args.tokens
        for r in args.ranks
    ]


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)


def paired_ratio(numerator, denominator, *, seed):
    """Median paired ratio and its deterministic 95% percentile bootstrap CI."""
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired samples must have equal nonzero lengths")
    if any(not math.isfinite(v) or v <= 0 for values in (numerator, denominator) for v in values):
        raise ValueError("timings must be finite and positive")
    ratios = [a / b for a, b in zip(numerator, denominator, strict=True)]
    rng = random.Random(seed)
    boot = [statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
    return statistics.median(ratios), [quantile(boot, 0.025), quantile(boot, 0.975)]


def noise_summary(ratio, twin_samples, baseline_samples, *, seed):
    """A/A = twin / best; epsilon = max(abs(median_AA - 1), (CI_hi - CI_lo) / 2).

    PASS preserves the strict median >= 1 rule. A slower result is TIE at or
    above 1 - epsilon, and FAIL below it. The interval uses the same bootstrap
    method as the main ratio, with sample pairing preserved.
    """
    aa_ratio, aa_interval = paired_ratio(twin_samples, baseline_samples, seed=seed)
    epsilon = max(abs(aa_ratio - 1), (aa_interval[1] - aa_interval[0]) / 2)
    return {
        "aa_ratio": aa_ratio,
        "aa_interval": aa_interval,
        "epsilon": epsilon,
        "verdict_eps": "PASS" if ratio >= 1 else "TIE" if ratio >= 1 - epsilon else "FAIL",
    }


def paired_summary(
    samples, *, seed, ours="ours", baselines=("vllm", "vllm_dual"), sample_indices=None
):
    """Select one strongest vLLM variant per row, then preserve sample pairing."""
    noise = (
        {
            "aa_ratio": None,
            "aa_interval": None,
            "epsilon": None,
            "verdict_eps": None,
            "aa_sample_count": 0,
        }
        if sample_indices is not None or any(name + "_twin" in samples for name in baselines)
        else {}
    )
    required = (ours, *baselines)
    if any(name not in samples for name in required):
        return {"verdict": "ERROR", "ratio": None, "interval": None, **noise}
    if len({len(samples[name]) for name in required}) != 1 or not samples[ours]:
        raise ValueError("paired samples must have equal nonzero lengths")
    if any(not math.isfinite(v) or v <= 0 for name in required for v in samples[name]):
        raise ValueError("timings must be finite and positive")
    best = min(baselines, key=lambda name: statistics.median(samples[name]))
    ratio, interval = paired_ratio(samples[best], samples[ours], seed=seed)
    twin = best + "_twin"
    if twin in samples:
        baseline_samples = samples[best]
        if sample_indices is not None:
            baseline_by_index = dict(zip(sample_indices[best], samples[best], strict=True))
            baseline_samples = [baseline_by_index[index] for index in sample_indices[twin]]
        noise.update(noise_summary(ratio, samples[twin], baseline_samples, seed=seed))
        noise.update(aa_sample_count=len(samples[twin]), aa_twin=twin)
    return {
        "best_vllm": best,
        "ratio": ratio,
        "interval": interval,
        "interval_method": "95% percentile bootstrap of median paired ratios (2000 draws)",
        "verdict": "PASS" if ratio >= 1.0 else "FAIL",
        **noise,
    }


def sample_orders(names, count, seed, *, twins=()):
    rng = random.Random(seed)
    twins = list(twins)
    rng.shuffle(twins)
    names = [name for name in names if name not in twins]
    result = []
    for index in range(count):
        order = list(names)
        if twins:
            order.append(twins[index % len(twins)])
        rng.shuffle(order)
        result.append(order)
    return result


def correctness_flag(ours_error, vllm_error):
    return (
        not math.isfinite(ours_error)
        or not math.isfinite(vllm_error)
        or ours_error > 2e-2
        or ours_error > 2 * vllm_error
    )


def fp32_oracle(x, w13, w2, ids_local, weights, factors, scaling):
    """Independent FP32 expert-by-expert reference, with no BF16 rounds.

    Disjoint pair outputs avoid nondeterministic index_add reductions. Disable
    TF32 outside this helper so the reference's matrix products remain FP32.
    """
    import torch
    import torch.nn.functional as F

    ga, gb, ua, ub, da, db = factors
    t, k = ids_local.shape
    output = torch.zeros(t * k, x.shape[-1], dtype=torch.float32, device=x.device)
    routes = ids_local.cpu().reshape(-1).tolist()
    buckets = [[] for _ in range(w13.shape[0])]
    for pair, expert in enumerate(routes):
        if expert >= 0:
            buckets[expert].append(pair)
    for expert, pairs in enumerate(buckets):
        if not pairs:
            continue
        pair_ids = torch.tensor(pairs, dtype=torch.long, device=x.device)
        inputs = x[pair_ids // k].float()
        gate, up = (inputs @ w13[expert].float().T).chunk(2, dim=-1)
        gate = gate + scaling * ((inputs @ ga[expert].float().T) @ gb[expert].float().T)
        up = up + scaling * ((inputs @ ua[expert].float().T) @ ub[expert].float().T)
        hidden = F.silu(gate) * up
        down = hidden @ w2[expert].float().T
        down = down + scaling * ((hidden @ da[expert].float().T) @ db[expert].float().T)
        output[pair_ids] = down * weights.reshape(-1)[pair_ids, None]
    return output.view(t, k, -1).sum(1)


def rel_l2(actual, expected):
    import torch

    error = torch.linalg.vector_norm(actual.float() - expected)
    norm = torch.linalg.vector_norm(expected).clamp_min(1e-30)
    return (error / norm).item()


class GraphSample:
    """Warm all pool entries, capture once, and time only CUDA graph work."""

    def __init__(self, calls, target_ms):
        import torch

        self.calls = len(calls)
        self.target_ms = target_ms
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for call in calls:
                    call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            with torch.cuda.graph(self.graph, stream=stream):
                for call in calls:
                    call()
        finally:
            if enabled:
                gc.enable()
        self.graph.replay()
        torch.cuda.synchronize()
        elapsed = self.elapsed(1)
        self.repeats = max(1, math.ceil(target_ms * 1.25 / max(elapsed, 1e-6)))

    def elapsed(self, repeats):
        import torch

        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(repeats):
            self.graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    def sample(self):
        # Recalibrate if clocks changed; every retained sample meets the floor.
        for _ in range(8):
            elapsed = self.elapsed(self.repeats)
            if elapsed >= self.target_ms:
                return elapsed * 1000 / (self.repeats * self.calls)
            self.repeats = max(
                self.repeats + 1,
                math.ceil(self.repeats * self.target_ms * 1.2 / max(elapsed, 1e-6)),
            )
        raise RuntimeError("CUDA graph sample did not meet duration floor")


def make_routing(tokens, experts, top_k, *, generator, routing="softmax", ep=1, ep_rank=0):
    """Generate global IDs, local IDs and normalized weights on the generator's device."""
    import torch

    device = generator.device
    local = experts // ep
    start = ep_rank * local
    logits = torch.randn((tokens, experts), device=device, dtype=torch.float32, generator=generator)
    if routing == "skewed":
        popularity = 1 / torch.arange(1, experts + 1, device=device, dtype=torch.float32)
        perm = torch.randperm(experts, device=device, generator=generator)
        logits += 2.0 * popularity.log()[perm]
    elif routing == "narrow":
        count = 2 * top_k
        local_count = max(1, round(count * local / experts))
        local_chosen = torch.randperm(local, device=device, generator=generator)[:local_count]
        remote_chosen = torch.randperm(experts - local, device=device, generator=generator)[
            : count - local_count
        ]
        remote_chosen = torch.where(remote_chosen < start, remote_chosen, remote_chosen + local)
        chosen = torch.cat((local_chosen + start, remote_chosen))
        logits[:, chosen] += 50.0
    elif routing != "softmax":
        raise ValueError(f"unknown routing: {routing}")
    weights, ids = logits.softmax(-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
    ids = ids.to(torch.int32)
    local_ids = torch.where((ids >= start) & (ids < start + local), ids - start, -1).contiguous()
    return ids, local_ids, weights


def make_inputs(row, pool_size, seed):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape, scale=1.0):
        return (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * scale
        ).contiguous()

    local = row.model.experts // row.ep
    w13 = normal((local, 2 * row.model.intermediate, row.model.hidden), row.model.hidden**-0.5)
    w2 = normal((local, row.model.hidden, row.model.intermediate), row.model.intermediate**-0.5)
    rank = row.rank
    factors = [
        normal(shape, scale)
        for shape, scale in (
            ((local, rank, row.model.hidden), row.model.hidden**-0.5),
            ((local, row.model.intermediate, rank), 0.1 / rank**0.5),
            ((local, rank, row.model.hidden), row.model.hidden**-0.5),
            ((local, row.model.intermediate, rank), 0.1 / rank**0.5),
            ((local, rank, row.model.intermediate), row.model.intermediate**-0.5),
            ((local, row.model.hidden, rank), 0.1 / rank**0.5),
        )
    ]
    pool = []
    for _ in range(pool_size):
        x = normal((row.tokens, row.model.hidden))
        ids, local_ids, weights = make_routing(
            row.tokens,
            row.model.experts,
            row.model.top_k,
            generator=generator,
            routing=row.routing,
            ep=row.ep,
            ep_rank=row.ep_rank,
        )
        # Fixed identical post-activation inputs for the isolated w2 delta.
        cache2 = normal((row.tokens * row.model.top_k, row.model.intermediate))
        pool.append((x, ids, local_ids, weights, cache2))
    return w13, w2, factors, pool


def block_fp8(w):
    """E4M3 [E, N, K] with FP32 128x128 block scales (amax / 448), and the BF16 dequantization."""
    import torch

    e, n, k = w.shape
    blocks = w.float().reshape(e, n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-10) / 448
    q = (blocks / scale[:, :, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = (q.float() * scale[:, :, None, :, None]).reshape(e, n, k).to(w.dtype)
    return q.reshape(e, n, k), scale, dequant


def build_candidates(runtime, row, args, data, fp8=None):
    import torch

    from benchmark.kernel_tuning._moe_lora_vllm import RoutedLayer

    w13, w2, factors, pool = data
    x, ids, _, weights, _ = pool[0]
    single = RoutedLayer(
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        args.scaling,
        row.ep,
        model=row.model,
        ep_rank=row.ep_rank,
        fp8_scales=fp8,
    )
    dual = RoutedLayer(
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        args.scaling,
        row.ep,
        model=row.model,
        ep_rank=row.ep_rank,
        dual=True,
        fp8_scales=fp8,
    )
    base = RoutedLayer(
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        args.scaling,
        row.ep,
        model=row.model,
        ep_rank=row.ep_rank,
        base=True,
        fp8_scales=fp8,
    )
    # Construct both possible controls before timing. Each owns its workspaces,
    # output, Punica metadata, and (for dual-stream) synchronization objects.
    single_twin = RoutedLayer(
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        args.scaling,
        row.ep,
        model=row.model,
        ep_rank=row.ep_rank,
        fp8_scales=fp8,
    )
    dual_twin = RoutedLayer(
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        args.scaling,
        row.ep,
        model=row.model,
        ep_rank=row.ep_rank,
        dual=True,
        fp8_scales=fp8,
    )
    # Graphs execute sequentially. Sharing workspaces across schedules avoids
    # multiplying large prefill workspace memory by candidate/pool count.
    for layer in (dual, base):
        layer.workspace13, layer.workspace2, layer.output = (
            single.workspace13,
            single.workspace2,
            single.output,
        )
    dual.ctx.w13_lora_b_stacked = single.ctx.w13_lora_b_stacked
    dual.ctx.w2_lora_b_stacked = single.ctx.w2_lora_b_stacked
    cache1 = torch.empty(
        (row.tokens * row.model.top_k, 2 * row.model.intermediate), device=x.device, dtype=x.dtype
    )
    cache3 = torch.empty(
        (row.tokens * row.model.top_k, row.model.hidden), device=x.device, dtype=x.dtype
    )

    def vllm_delta(entry):
        x, ids, _, weights, cache2 = entry
        # Both delta candidates include identical output clearing, so repeated
        # graph replays never accumulate or overflow an in-place destination.
        cache1.zero_()
        cache3.zero_()
        single.delta(cache1, cache2, cache3, x, ids, weights)
        return cache1, cache3

    candidates = {
        "vllm": [lambda entry=entry: single(entry[0], entry[1], entry[3]) for entry in pool],
        "vllm_dual": [lambda entry=entry: dual(entry[0], entry[1], entry[3]) for entry in pool],
        "vllm_twin": [
            lambda entry=entry: single_twin(entry[0], entry[1], entry[3]) for entry in pool
        ],
        "vllm_dual_twin": [
            lambda entry=entry: dual_twin(entry[0], entry[1], entry[3]) for entry in pool
        ],
        "vllm_base": [lambda entry=entry: base(entry[0], entry[1], entry[3]) for entry in pool],
        "vllm_lora": [lambda entry=entry: vllm_delta(entry) for entry in pool],
    }
    errors = {}
    try:
        from ctikernels.moe.lora import (
            MoELoRAWeights,
            fused_experts_lora,
            moe_lora_w2_,
            moe_lora_w13_,
        )
        from ctikernels.moe_impl import moe_align_block_size

        gate_a, gate_b, up_a, up_b, down_a, down_b = factors
        lora = MoELoRAWeights.separate(gate_a, up_a, gate_b, up_b, down_a, down_b)
        # These are the base alignment inputs already available to an in-layer
        # LoRA update. vLLM's isolated candidate still runs its own LoRA alignment.
        aligned = []
        for entry in pool:
            local_ids = torch.where(entry[2] >= 0, entry[2], 0).contiguous()
            sorted_ids, expert_ids, padded = moe_align_block_size(
                local_ids, 32, row.model.experts // row.ep
            )
            aligned.append(
                {
                    "sorted_token_ids": sorted_ids,
                    "expert_ids": expert_ids,
                    "num_tokens_post_padded": padded,
                    "block_m": 32,
                }
            )

        def ours(entry, provider=args.provider, base_only=False):
            return fused_experts_lora(
                entry[0],
                w13,
                w2,
                entry[3],
                entry[2],
                None if base_only else lora,
                scaling=args.scaling,
                provider=provider,
                local_only=row.ep == 1,
                num_global_experts=row.model.experts,
                **({} if fp8 is None else {"w13_scale": fp8[0], "w2_scale": fp8[1]}),
            )

        candidates["ours"] = [lambda entry=entry: ours(entry) for entry in pool]
        if args.compare_provider is not None:
            candidates["ours_alt"] = [
                lambda entry=entry: ours(entry, args.compare_provider) for entry in pool
            ]
        candidates["ours_base"] = [
            lambda entry=entry: ours(entry, base_only=True) for entry in pool
        ]
        if args.ours_torch and row.tokens <= 64:
            candidates["ours_torch"] = [lambda entry=entry: ours(entry, "torch") for entry in pool]

        def ours_delta(entry, alignment):
            x, _, local_ids, weights, cache2 = entry
            cache1.zero_()
            cache3.zero_()
            moe_lora_w13_(
                cache1,
                x,
                lora,
                local_ids,
                scaling=args.scaling,
                top_k=row.model.top_k,
                provider=args.provider,
                **alignment,
            )
            moe_lora_w2_(
                cache3,
                cache2,
                lora,
                local_ids,
                weights,
                scaling=args.scaling,
                top_k=row.model.top_k,
                provider=args.provider,
                **alignment,
            )
            return cache1, cache3

        candidates["ours_lora"] = [
            lambda entry=entry, alignment=alignment: ours_delta(entry, alignment)
            for entry, alignment in zip(pool, aligned, strict=True)
        ]
    except Exception as error:
        errors["ours_import"] = type(error).__name__
    if args.sglang_python is not None:
        try:
            candidates.update(sglang_candidates(row, args, data, cache1, cache3, fp8))
        except Exception as error:
            errors["sglang_import"] = type(error).__name__
    return candidates, errors, single.base_config


def sglang_candidates(row, args, data, cache1, cache3, fp8=None):
    """SGLang's MoE LoRA op around the SGLang-port base pipeline (moe_impl), and its delta alone."""
    import torch

    from benchmark.kernel_tuning._moe_lora_sglang import SGLangMoELoRA, load
    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.linear.block_fp8_gated import silu_and_mul_quantize_fp8
    from ctikernels.moe.lora import base_config_for
    from ctikernels.moe_impl import (
        block_fp8_moe_config,
        block_fp8_moe_provider,
        fused_moe_kernel_triton,
        moe_align_block_size,
        moe_sum_reduce_triton,
    )

    w13, w2, factors, pool = data
    t, k, e = row.tokens, row.model.top_k, row.model.experts // row.ep
    h, i = row.model.hidden, row.model.intermediate
    sgl = SGLangMoELoRA(load(args.sglang_python), factors, e, k, w13.device, args.scaling)
    cfg = base_config_for(t)
    gemm = {}
    if fp8 is not None:
        cfg = block_fp8_moe_config(
            t, e, 2 * i, h, k, capability=torch.cuda.get_device_capability(w13.device)
        )
        gemm = {"block_fp8_provider": block_fp8_moe_provider(w13.device)}
    # Remote routes: expert 0 at routing weight 0 (the weights below), as the engine does.
    safe = [torch.where(entry[2] >= 0, entry[2], 0).to(torch.int32) for entry in pool]
    masked = [torch.where(entry[2] >= 0, entry[3], 0.0) for entry in pool]

    def full(entry, ids, weights):
        x = entry[0]
        sorted_ids, experts, padded = moe_align_block_size(ids, cfg["BLOCK_SIZE_M"], e)
        c1 = torch.empty((t, k, 2 * i), device=x.device, dtype=x.dtype)
        a1, s1 = (x, None) if fp8 is None else quantize_per_token_group_fp8(x)
        scales1 = {} if fp8 is None else {"a_scale": s1, "b_scale": fp8[0], **gemm}
        fused_moe_kernel_triton(
            a1, w13, c1, weights, ids, sorted_ids, experts, padded, False, k, cfg, x.dtype,
            **scales1,
        )  # fmt: skip
        lora_align = sgl.alignment(ids)
        sgl._call(c1, x, sgl.a13, sgl.b13, weights, lora_align, False)
        if fp8 is None:
            c2 = a2 = silu_and_mul(c1.view(t * k, 2 * i))
            scales2 = {}
        else:
            c2 = torch.empty((t * k, i), device=x.device, dtype=x.dtype)
            a2, s2 = silu_and_mul_quantize_fp8(c1.view(t * k, 2 * i), act=c2)
            scales2 = {"a_scale": s2, "b_scale": fp8[1], **gemm}
        c3 = torch.empty((t, k, h), device=x.device, dtype=x.dtype)
        fused_moe_kernel_triton(
            a2, w2, c3, weights, ids, sorted_ids, experts, padded, True, 1, cfg, x.dtype,
            **scales2,
        )  # fmt: skip
        sgl._call(c3, c2, sgl.a2, sgl.b2, weights, lora_align, True)
        out = torch.empty_like(x)
        moe_sum_reduce_triton(c3, out)
        return out

    def delta(entry, ids, weights):
        cache1.zero_()
        cache3.zero_()
        sgl(cache1.view(t, k, -1), entry[4], cache3.view(t, k, -1), entry[0], ids, weights)
        return cache1, cache3

    return {
        "sglang": [
            lambda a=entry, b=ids, c=w: full(a, b, c)
            for entry, ids, w in zip(pool, safe, masked, strict=True)
        ],
        "sglang_lora": [
            lambda a=entry, b=ids, c=w: delta(a, b, c)
            for entry, ids, w in zip(pool, safe, masked, strict=True)
        ],
    }


def make_shared_inputs(row, pool_size, seed, *, device="cuda"):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, scale=1.0):
        return (
            torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) * scale
        ).contiguous()

    intermediate, rank = row.intermediate, row.rank
    factors = (
        normal((2, rank, row.model.hidden), row.model.hidden**-0.5),
        normal((2, intermediate, rank), 0.1 / rank**0.5),
        normal((rank, intermediate), intermediate**-0.5),
        normal((row.model.hidden, rank), 0.1 / rank**0.5),
    )
    pool = [
        (normal((row.tokens, row.model.hidden)), normal((row.tokens, intermediate)))
        for _ in range(pool_size)
    ]
    return factors, pool


def shared_fp32_oracle(x, hidden, factors, scaling):
    """Independent gate/up and down deltas; hidden is fixed, not derived from x."""
    import torch

    a13, b13, a2, b2 = factors
    gate_up = torch.cat(
        [(x.float() @ a.float().T) @ b.float().T for a, b in zip(a13, b13, strict=True)],
        dim=-1,
    )
    down = (hidden.float() @ a2.float().T) @ b2.float().T
    return gate_up * scaling, down * scaling


def build_shared_candidates(runtime, row, args, data):
    import torch

    from benchmark.kernel_tuning._moe_lora_vllm import SharedLayer

    factors, pool = data
    x, _ = pool[0]
    layer = SharedLayer(runtime, x, factors, args.scaling)
    cache = torch.empty((row.tokens, 2 * row.intermediate), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)

    def vllm_delta(entry):
        # Both candidates clear identical destinations inside the graph so
        # replay cannot accumulate deltas. Punica owns its shrink buffers.
        cache.zero_()
        out.zero_()
        return layer.delta(cache, out, *entry)

    candidates = {"vllm_shared_lora": [lambda entry=entry: vllm_delta(entry) for entry in pool]}
    errors = {}
    try:
        from ctikernels.moe.lora import shared_lora_w2_, shared_lora_w13_

        a13, b13, a2, b2 = factors

        def ours_delta(entry):
            cache.zero_()
            out.zero_()
            shared_lora_w13_(
                cache, entry[0], (a13, b13), scaling=args.scaling, provider=args.provider
            )
            shared_lora_w2_(out, entry[1], (a2, b2), scaling=args.scaling, provider=args.provider)
            return cache, out

        candidates["ours_shared_lora"] = [lambda entry=entry: ours_delta(entry) for entry in pool]
    except Exception as error:
        errors["ours_import"] = type(error).__name__
    return candidates, errors


def benchmark_shared_row(runtime, row, args):
    import torch

    seed = args.seed + row.tokens * 1009 + row.rank * 17 + 1
    data = make_shared_inputs(row, args.pool_size, seed)
    candidates, errors = build_shared_candidates(runtime, row, args, data)
    factors, pool = data
    oracle = shared_fp32_oracle(*pool[0], factors, args.scaling)
    accuracy, outputs = {}, {}
    for name, calls in candidates.items():
        try:
            output = calls[0]()
            values = {
                projection: rel_l2(actual, expected)
                for projection, actual, expected in zip(
                    ("gate_up", "down"), output, oracle, strict=True
                )
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError("nonfinite output")
            accuracy[name] = values
            outputs[name] = tuple(value.clone() for value in output)
        except Exception as error:
            errors[name] = {"phase": "correctness", "type": type(error).__name__}
    ours, baseline = "ours_shared_lora", "vllm_shared_lora"
    differences, accuracy_failed = {}, None
    if ours in accuracy and baseline in accuracy:
        accuracy_failed = any(
            correctness_flag(accuracy[ours][projection], accuracy[baseline][projection])
            for projection in ("gate_up", "down")
        )
        for projection, actual, expected in zip(
            ("gate_up", "down"), outputs[ours], outputs[baseline], strict=True
        ):
            value = rel_l2(actual, expected.float())
            differences[projection] = value if math.isfinite(value) else None
    del outputs
    graphs = {}
    for name, calls in candidates.items():
        if name in errors:
            continue
        try:
            graphs[name] = GraphSample(calls, args.sample_ms)
        except Exception as error:
            errors[name] = {"phase": "warmup/capture", "type": type(error).__name__}
    samples = {name: [] for name in graphs}
    orders = sample_orders(graphs, args.samples, seed)
    for order in orders:
        for name in order:
            if name in errors:
                continue
            try:
                samples[name].append(graphs[name].sample())
            except Exception as error:
                errors[name] = {"phase": "sample", "type": type(error).__name__}
    samples = {name: values for name, values in samples.items() if len(values) == args.samples}
    result = dict(
        row.fields(),
        **paired_summary(samples, seed=seed, ours=ours, baselines=(baseline,)),
        status=("partial" if samples else "error") if errors else "ok",
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        seed=seed,
        scaling=args.scaling,
        delta_mode="gate_up + down on identical fixed inputs; destinations cleared",
        adapter_id=0,
        pool_size=args.pool_size,
        sample_ms_floor=args.sample_ms,
        times_us={name: statistics.median(values) for name, values in samples.items()},
        samples_us=samples,
        sample_orders=orders,
        rel_l2=accuracy,
        ours_vs_vllm_rel_l2=differences,
        correctness_failed=accuracy_failed,
        errors=errors,
        graph_calls={name: graph.calls for name, graph in graphs.items()},
        graph_replays_per_sample={name: graph.repeats for name, graph in graphs.items()},
        vllm=dict(runtime.provenance(), entrypoint="PunicaWrapperGPU.add_lora_linear"),
    )
    if accuracy_failed:
        result["performance_verdict"] = result["verdict"]
        result["verdict"] = "INCORRECT"
    return result


def benchmark_row(runtime, row, args):
    if isinstance(row, SharedRow):
        return benchmark_shared_row(runtime, row, args)

    import torch

    from benchmark.kernel_tuning._moe_lora_vllm import path_flags

    seed = args.seed + row.tokens * 1009 + row.rank * 17 + row.ep
    data = make_inputs(row, args.pool_size, seed)
    fp8 = dequant = None
    if args.base == "fp8":
        (q13, s13, d13), (q2, s2, d2) = block_fp8(data[0]), block_fp8(data[1])
        data, fp8, dequant = (q13, q2, *data[2:]), (s13, s2), (d13, d2)
    candidates, errors, config = build_candidates(runtime, row, args, data, fp8)
    result = dict(
        row.fields(),
        status="ok",
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        seed=seed,
        base_config=config,
        vllm_path=path_flags(
            row.tokens, row.rank, row.ep, experts=row.model.experts, top_k=row.model.top_k
        ),
        shared="reported separately (kind=shared)" if row.model.shared else "none",
        ep_scope="local partial output; no collectives",
        scaling=args.scaling,
        ours_base_mode="lora=None",
        delta_mode="w13 + w2 on identical fixed cache2; vLLM LoRA alignment included; ours reuses base alignment; destinations cleared",
        ours_delta_block_m=32,
        pool_size=args.pool_size,
        sample_ms_floor=args.sample_ms,
        base=args.base,
    )
    w13, w2, factors, pool = data
    if dequant is not None:
        w13, w2 = dequant  # the oracle runs the FP8 weights' exact BF16 values
    x, _, local_ids, weights, _ = pool[0]
    oracle = fp32_oracle(x, w13, w2, local_ids, weights, factors, args.scaling)
    accuracy = {}
    outputs = {}
    for name in (
        "vllm",
        "vllm_dual",
        "vllm_twin",
        "vllm_dual_twin",
        "ours",
        "ours_alt",
        "ours_torch",
        "sglang",
    ):
        if name in candidates:
            try:
                output = candidates[name][0]()
                value = rel_l2(output, oracle)
                if not math.isfinite(value):
                    raise FloatingPointError("nonfinite output")
                accuracy[name] = value
                outputs[name] = output.clone()
            except Exception as error:
                errors[name] = {"phase": "correctness", "type": type(error).__name__}
    output_differences = {}
    if "ours" in outputs:
        for name in ("vllm", "vllm_dual"):
            if name in outputs:
                value = rel_l2(outputs["ours"], outputs[name].float())
                output_differences[name] = value if math.isfinite(value) else None
    del outputs
    graphs = {}
    for name, calls in candidates.items():
        if name in errors:
            continue
        try:
            graphs[name] = GraphSample(calls, args.sample_ms)
        except Exception as error:
            errors[name] = {"phase": "warmup/capture", "type": type(error).__name__}
    samples = {name: [] for name in graphs}
    sample_indices = {name: [] for name in graphs}
    twins = [name for name in ("vllm_twin", "vllm_dual_twin") if name in graphs]
    orders = sample_orders(graphs, args.samples, seed, twins=twins)
    for index, order in enumerate(orders):
        for name in order:
            if name in errors:
                continue
            try:
                samples[name].append(graphs[name].sample())
                sample_indices[name].append(index)
            except Exception as error:
                errors[name] = {"phase": "sample", "type": type(error).__name__}
    samples = {name: values for name, values in samples.items() if name not in errors}
    sample_indices = {name: sample_indices[name] for name in samples}
    summary = paired_summary(samples, seed=seed, sample_indices=sample_indices)
    best = summary.get("best_vllm", "vllm")
    accuracy_failed = None
    if "ours" in accuracy and best in accuracy:
        accuracy_failed = correctness_flag(accuracy["ours"], accuracy[best])
    for name in ("sglang", "sglang_lora", "ours_alt"):
        ours_name = "ours_lora" if name == "sglang_lora" else "ours"
        if name in samples and ours_name in samples:
            ratio, interval = paired_ratio(samples[name], samples[ours_name], seed=seed)
            label = name.removeprefix("ours_")
            summary[f"ratio_vs_{label}"] = ratio
            summary[f"interval_vs_{label}"] = interval
    result.update(
        summary,
        times_us={name: statistics.median(values) for name, values in samples.items()},
        samples_us=samples,
        sample_indices=sample_indices,
        sample_orders=orders,
        rel_l2=accuracy,
        ours_vs_vllm_rel_l2=output_differences,
        correctness_failed=accuracy_failed,
        errors=errors,
        graph_calls={name: graph.calls for name, graph in graphs.items()},
        graph_replays_per_sample={name: graph.repeats for name, graph in graphs.items()},
        vllm=runtime.provenance(),
    )
    if errors:
        result["status"] = "partial" if samples else "error"
    if accuracy_failed:
        result["performance_verdict"] = result["verdict"]
        result["verdict"] = "INCORRECT"
    for full, base_name, label in (("ours", "ours_base", "ours"), (best, "vllm_base", "vllm")):
        if full in samples and base_name in samples:
            result.setdefault("lora_increment_us", {})[label] = statistics.median(
                a - b for a, b in zip(samples[full], samples[base_name], strict=True)
            )
    return result


def main(argv=None):
    args = parse_args(argv)
    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("w")

    def emit(record):
        line = json.dumps(record, allow_nan=False, sort_keys=True)
        print(line, flush=True)
        if output:
            output.write(line + "\n")
            output.flush()

    try:
        if args.dry_run:
            for row in rows(args):
                record = dict(row.fields(), status="dry_run")
                if isinstance(row, Row):
                    record["shared"] = (
                        "reported separately (kind=shared)" if row.model.shared else "none"
                    )
                emit(record)
            return 0
        import torch

        from benchmark.kernel_tuning._moe_lora_vllm import WheelRuntime
        from benchmark.kernel_tuning._prologue_market import quiet_external

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime, load_error = None, None
        device_info = {"device": None, "capability": None}
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for timing")
            device_info = {
                "device": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
            with quiet_external():
                runtime = WheelRuntime(args.vllm_cache)
        except Exception as error:
            load_error = type(error).__name__
        failed = False
        for row in rows(args):
            record = dict(
                row.fields(),
                **device_info,
                status="error",
                verdict="ERROR",
            )
            if isinstance(row, Row):
                record["shared"] = (
                    "reported separately (kind=shared)" if row.model.shared else "none"
                )
            try:
                if runtime is None:
                    record["errors"] = {"runtime_load": load_error}
                else:
                    with quiet_external(), torch.inference_mode():
                        record = benchmark_row(runtime, row, args)
            except Exception as error:
                record["errors"] = {"row": type(error).__name__}
                if runtime is not None:
                    record["vllm"] = runtime.provenance()
            failed |= (
                record.get("status") != "ok" or record.get("correctness_failed", False) is True
            )
            emit(record)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return int(failed)
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
