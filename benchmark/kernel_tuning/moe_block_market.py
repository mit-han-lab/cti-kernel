# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
# External providers must fail independently; diagnostics report exception types,
# never arbitrary exception text (which can contain credentials or machine paths).
"""Paired BF16 Qwen3.5-35B-A3B MoE-block comparison, one EP rank at a time.

Run as a script or module with --sglang-path and --vllm-lib supplied explicitly.
The SGLang path contains its `sglang` directory. Optional --check-imports checks
all market imports, the published topology, and the vLLM op schema without CUDA.
SGLang's ordinary Python import dependencies must be installed or on PYTHONPATH.

Each provider has independent graph memory and distinct retained outputs. All
providers consume the same input slots and cycle the same independent layers.
K is a multiple of L and is calibrated until every loop takes at least 2 ms.
CUDA events include launch gaps in eager forms. No collective is performed.
The CI resamples paired rounds and reselects the fastest market median in each
bootstrap sample; it estimates min(market medians) / median(ours:auto).
"""

from __future__ import annotations

import argparse
import builtins
import gc
import importlib
import importlib.metadata
import inspect
import json
import math
import random
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

if __package__:
    from ._moe_market_imports import load_sglang
else:
    from _moe_market_imports import load_sglang

H, E, EXPERT_I, SHARED_I, TOP_K = 2048, 256, 512, 512, 8
PROVIDERS = ("ours:auto", "ours:triton_block", "ours:composed", "sglang", "vllm", "vllm:graph")
MARKETS = ("sglang", "vllm", "vllm:graph")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default=",".join(PROVIDERS))
    parser.add_argument("--sglang-path", type=Path)
    parser.add_argument("--vllm-lib", type=Path)
    parser.add_argument("--ep", type=int, choices=(1, 2), default=1)
    parser.add_argument("--rank", type=int, help="default: measure all EP ranks separately")
    parser.add_argument("--rows", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--t", help="explicit comma-separated token counts")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="correctness lines, no timing")
    parser.add_argument("--allow-fi-version", action="store_true")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args(argv)
    args.providers = list(dict.fromkeys(args.providers.split(",")))
    if any(p not in PROVIDERS for p in args.providers):
        parser.error("unknown provider")
    try:
        args.tokens = (
            [int(t) for t in args.t.split(",")]
            if args.t
            else (list(range(1, 9)) if args.rows == "decode" else [512, 1024, 2048, 4096, 8192])
        )
    except ValueError:
        parser.error("--t requires comma-separated integers")
    args.layers = args.layers if args.layers is not None else (3 if args.rows == "decode" else 2)
    if args.layers < 1 or args.rounds < 1 or not args.tokens or min(args.tokens) < 1:
        parser.error("layers, rounds, and token counts must be positive")
    if args.rank is not None and not 0 <= args.rank < args.ep:
        parser.error("rank must be in [0, ep)")
    return args


class Reporter:
    def __init__(self, path):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8") if path else None
        self.unavailable = set()

    def emit(self, record):
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(line, flush=True)
        if self.file:
            self.file.write(line + "\n")
            self.file.flush()

    def fail(self, name, stage, exc):
        if name not in self.unavailable:
            reason = type(exc).__name__
            if (
                isinstance(exc, ModuleNotFoundError)
                and exc.name
                and all(c.isalnum() or c in "_." for c in exc.name)
            ):
                reason += " (missing module " + exc.name + ")"
            print(f"UNAVAILABLE {name} {stage}: {reason}", flush=True)
            self.unavailable.add(name)

    def close(self):
        if self.file:
            self.file.close()


@dataclass
class Weights:
    router: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    shared_gate_up: torch.Tensor
    shared_down: torch.Tensor
    shared_gate: torch.Tensor
    expert_map: torch.Tensor
    up_gate: torch.Tensor | None = None


def make_weights(args, rank, need_vllm):
    result = []
    local, si = E // args.ep, SHARED_I // args.ep
    for layer in range(args.layers):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + 1009 * layer)

        def normal(shape, gen=gen):
            return (
                torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
                .mul_(0.02)
                .bfloat16()
            )

        router = normal((E, H))
        shared_gate = normal((1, H))
        full_gu, full_d = normal((2, SHARED_I, H)), normal((H, SHARED_I))
        sh_gu = full_gu[:, rank * si : (rank + 1) * si].reshape(2 * si, H).contiguous()
        sh_d = full_d[:, rank * si : (rank + 1) * si].contiguous()
        del full_gu, full_d
        # Separate per-rank expert RNG; router/shared tensors above are identical
        # across ranks, with genuine complementary shared intermediate slices.
        gen.manual_seed(args.seed + 1009 * layer + 1000003 * (rank + 1))
        gu, down = normal((local, 2 * EXPERT_I, H)), normal((local, H, EXPERT_I))
        mapping = torch.full((E,), -1, device="cuda", dtype=torch.int32)
        mapping[rank * local : (rank + 1) * local] = torch.arange(
            local, device="cuda", dtype=torch.int32
        )
        up_gate = torch.cat((gu[:, EXPERT_I:], gu[:, :EXPERT_I]), dim=1) if need_vllm else None
        result.append(Weights(router, gu, down, sh_gu, sh_d, shared_gate, mapping, up_gate))
    return result


def make_inputs(t, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Allocate separate blocks; the first L inputs stay identical when K grows.
    return [
        torch.randn((t, H), generator=gen, device="cuda", dtype=torch.bfloat16) for _ in range(k)
    ]


def routing(x, w):
    logits = F.linear(x, w.router)  # BF16 logits are part of the contract.
    weights, ids = logits.float().softmax(-1).topk(TOP_K, dim=-1)
    return weights / weights.sum(-1, keepdim=True), ids


def reference(x, w, ids=None):
    """FP32 block on BF16 weights. With ``ids`` (a provider's own selection), the weights are
    the renormalized softmax at those ids, so router near-ties do not count as error."""
    if ids is None:
        weights, ids = routing(x, w)
    else:
        probs = F.linear(x, w.router).float().softmax(-1).gather(1, ids)
        weights = probs / probs.sum(-1, keepdim=True)
    local_ids = w.expert_map[ids.long()]
    xf = x.float()
    out = torch.zeros_like(xf)
    for expert in local_ids.unique().tolist():
        if expert < 0:
            continue
        token, slot = torch.where(local_ids == expert)
        gu = F.linear(xf[token], w.gate_up[expert].float())
        gate, up = gu.chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, w.down[expert].float())
        out.index_add_(0, token, y * weights[token, slot, None])
    gate, up = F.linear(xf, w.shared_gate_up.float()).chunk(2, dim=-1)
    shared = F.linear(F.silu(gate) * up, w.shared_down.float())
    return out + torch.sigmoid(F.linear(xf, w.shared_gate.float())) * shared


@contextmanager
def sglang_topology(s, ep, rank):
    # Same public publication sequence as test_utils.published_topology, without
    # importing its HTTP clients, model downloaders, and process-control helpers.
    s.runtime.reset_context()
    args = s.server.ServerArgs(
        model_path="dummy", tp_size=ep, ep_size=ep, moe_runner_backend="triton"
    )
    s.runtime.publish(args, role="test", ranks=s.runtime.SpawnRanks(world_rank=rank, dp_rank=None))
    # This benchmark stops before all-reduce and never initializes process groups.
    s.runtime.get_parallel().override_permanently(tp_group=None)
    try:
        yield
    finally:
        s.runtime.reset_context()


def sglang_config(s, t, w):
    opened = []
    real_open = builtins.open

    def observe(file, *args, **kwargs):
        p = Path(file)
        if p.suffix == ".json":
            opened.append(str(Path(p.parent.name) / p.name))
        return real_open(file, *args, **kwargs)

    # Observe actual reads by SGLang's getter, including version fallback and
    # the down-projection's fallback to the up-projection config.
    s.config.get_moe_configs.cache_clear()
    with patch("builtins.open", observe):
        config, (down, max_block) = s.config.try_get_optimal_moe_config(
            w.gate_up.shape, w.down.shape, TOP_K, None, t, return_down_config=True
        )
    return {
        "up": dict(config),
        "down": dict(down) if down else None,
        "max_block_m": max_block,
        "files": list(dict.fromkeys(opened)) or ["default"],
        "up_source": next((p for p in opened if not p.endswith("_down.json")), "default"),
        "down_source": next(
            (p for p in opened if p.endswith("_down.json")),
            next((p for p in opened if not p.endswith("_down.json")), "default")
            if down
            else "default",
        ),
    }


def load_topk(lib):
    if lib is None:
        raise ValueError("--vllm-lib is required")
    core = lib / "_C_stable_libtorch.abi3.so"
    if core.is_file():
        torch.ops.load_library(str(core))
    torch.ops.load_library(str(lib / "_moe_C_stable_libtorch.abi3.so"))
    op = torch.ops._moe_C.topk_softmax.default
    schema = op._schema
    names = [a.name for a in schema.arguments]
    required = [
        "topk_weights",
        "topk_indices",
        "token_expert_indices",
        "gating_output",
        "renormalize",
    ]
    if names[:5] != required:
        raise ValueError("unsupported topk_softmax required arguments")
    tail = []
    for arg in schema.arguments[5:]:
        if arg.name in ("bias", "is_padding"):
            tail.append(None)
        elif arg.has_default_value():
            tail.append(arg.default_value)
        else:
            raise ValueError("unsupported topk_softmax extra argument")
    return op, tail, str(schema)


def native_silu_mul(gu):
    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


class FlashInferBackend:
    def __init__(self, args, *, precomputed_routing=False):
        fi = importlib.import_module("flashinfer")
        self.version = fi.__version__
        if self.version != "0.7.0.post1" and not args.allow_fi_version:
            raise ValueError("FlashInfer version differs from 0.7.0.post1; use --allow-fi-version")
        moe = importlib.import_module("flashinfer.fused_moe")
        core = importlib.import_module("flashinfer.fused_moe.core")
        self.activation = getattr(core, "ActivationType", getattr(moe, "ActivationType", None))
        if self.activation is None:
            raise ImportError("FlashInfer ActivationType is unavailable")
        self.fused = moe.cutlass_fused_moe
        self.signature = inspect.signature(self.fused)
        expected = {
            "input",
            "token_selected_experts",
            "token_final_scales",
            "fc1_expert_weights",
            "fc2_expert_weights",
            "output_dtype",
            "quant_scales",
            "output",
            "tp_size",
            "tp_rank",
            "ep_size",
            "ep_rank",
            "activation_type",
        }
        if not expected.issubset(self.signature.parameters):
            raise ValueError("unsupported FlashInfer CUTLASS signature")
        if not precomputed_routing:
            self.topk, self.tail, self.schema = load_topk(args.vllm_lib)
        self.autotune = importlib.import_module("flashinfer.autotuner").autotune
        utils = importlib.import_module("flashinfer.fused_moe.utils")
        self.buckets = getattr(utils, "get_hybrid_num_tokens_buckets", None)
        # One compiled native activation, specialized by shape, warmed before capture.
        self.silu = torch.compile(native_silu_mul, dynamic=False, fullgraph=True)
        self.tuned = False
        self.tune_seconds = None
        self.tuning_buckets = None

    def tune(self, args, weights, rank):
        if self.tuned:
            return
        t = max(args.tokens)
        x = make_inputs(t, 1, args.seed + 71)[0]
        block = Block("vllm", args, rank, t, fi=self)
        slot = block.prepare(x)
        self.tune_calls(args, [lambda w=w: block(x, w, slot) for w in weights])

    def tune_calls(self, args, calls):
        """Autotune a prepared workload with the deployed hybrid token buckets."""
        if self.tuned:
            return
        t = max(args.tokens)
        start = time.perf_counter()
        self.tuning_buckets = self.buckets(t) if self.buckets else None
        inner = (
            self.autotune(tuning_buckets=self.tuning_buckets)
            if (
                self.tuning_buckets
                and "tuning_buckets" in inspect.signature(self.autotune).parameters
            )
            else nullcontext()
        )
        # vLLM's outer tune_mode=True context encloses a bucket-scoped dummy run.
        with self.autotune(tune_mode=True), inner:
            for call in calls:
                call()
        torch.cuda.synchronize()
        self.tune_seconds = time.perf_counter() - start
        self.tuned = True


class Block:
    def __init__(self, name, args, rank, t, sgl=None, fi=None):
        self.name, self.args, self.rank, self.t = name, args, rank, t
        self.sgl, self.fi = sgl, fi
        self.graph_mode = (
            name.startswith("ours:")
            or name == "vllm:graph"
            or (name == "sglang" and t <= 2048)
            or (name == "vllm" and t <= 512)
        )
        self.alt = torch.cuda.Stream() if name in MARKETS else None
        self.op = None
        if name.startswith("ours:"):
            from ctikernels.moe import MoEBlock

            self.op = MoEBlock(provider=name.split(":", 1)[1])

    def prepare(self, x):
        out = torch.empty_like(x)
        if self.name.startswith("vllm"):
            return (
                out,
                torch.empty((self.t, TOP_K), device=x.device, dtype=torch.float32),
                torch.empty((self.t, TOP_K), device=x.device, dtype=torch.int32),
                torch.empty((self.t, TOP_K), device=x.device, dtype=torch.int32),
            )
        return (out,)

    def shared(self, x, w):
        gu = F.linear(x, w.shared_gate_up)
        activation = (
            self.sgl.activation.silu_and_mul(gu) if self.name == "sglang" else self.fi.silu(gu)
        )
        y = F.linear(activation, w.shared_down)
        if self.name.startswith("vllm"):
            y = torch.sigmoid(F.linear(x, w.shared_gate)) * y
        return y

    def sgl_routed(self, x, w):
        logits = F.linear(x, w.router)
        scales, ids = self.sgl.gate.moe_fused_gate(
            logits, None, TOP_K, scoring_func="softmax", renormalize=True
        )
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

    def __call__(self, x, w, slot):
        if self.op is not None:
            return self.op(
                x,
                w.router,
                w.gate_up,
                w.down,
                w.shared_gate_up,
                w.shared_down,
                w.shared_gate,
                # EP1 owns every expert: the engine passes no map (no -1 routes to skip).
                expert_map=w.expert_map if self.args.ep > 1 else None,
                out=slot[0],
            )
        current = torch.cuda.current_stream()
        if self.name == "sglang":
            if self.graph_mode:
                self.alt.wait_stream(current)
                # SGLang's dual-stream implementation clones the shared input.
                shared = self.shared(x.clone(), w)
                with torch.cuda.stream(self.alt):
                    routed = self.sgl_routed(x, w)
                current.wait_stream(self.alt)
            else:
                shared = self.shared(x, w)
                routed = self.sgl_routed(x, w)
            self.sgl.elementwise.fused_gate_sigmoid_mul_add(
                x, w.shared_gate.view(-1), shared, routed
            )
            return routed
        if self.t <= 256:
            self.alt.wait_stream(current)
            with torch.cuda.stream(self.alt):
                shared = self.shared(x, w)
        logits = F.linear(x, w.router)
        if self.t > 256:
            shared = self.shared(x, w)
        out, scales, ids, indices = slot
        self.fi.topk(scales, ids, indices, logits, True, *self.fi.tail)
        self.fi.fused(
            input=x,
            token_selected_experts=ids,
            token_final_scales=scales,
            fc1_expert_weights=w.up_gate,
            fc2_expert_weights=w.down,
            output_dtype=torch.bfloat16,
            quant_scales=[],
            output=out,
            tp_size=1,
            tp_rank=0,
            ep_size=self.args.ep,
            ep_rank=self.rank,
            activation_type=self.fi.activation.Swiglu,
        )
        if self.t <= 256:
            current.wait_stream(self.alt)
        # Some FlashInfer releases return a one-element list; the explicitly
        # supplied output tensor is the stable API contract in either case.
        return shared + out


def event_us(call):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000


@contextmanager
def capture_gc():
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class TimedLoop:
    def __init__(self, block, xs, weights):
        self.block, self.xs, self.weights = block, xs, weights
        self.slots = [block.prepare(x) for x in xs]
        self.outputs = [None] * len(xs)
        self.graph = None
        # Side-stream warmup also warms library handles used during capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.eager()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        if block.graph_mode:
            self.outputs = [None] * len(xs)
            with capture_gc():
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.eager()
            self.graph.replay()
            torch.cuda.synchronize()

    def eager(self):
        for j, (x, slot) in enumerate(zip(self.xs, self.slots, strict=True)):
            self.outputs[j] = self.block(x, self.weights[j % len(self.weights)], slot)

    def run(self):
        if self.graph is None:
            self.eager()
        else:
            self.graph.replay()

    def time(self):
        return event_us(self.run)


def ceilings():
    props = torch.cuda.get_device_properties(0)
    # 512 MiB minimum exceeds 4x L2 on target devices, even when the property is
    # unavailable. The reduction reads each FP32 value and writes tiny partials.
    l2 = getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0))
    nbytes = max(512 << 20, 4 * l2)
    data = torch.ones(nbytes // 4, device="cuda", dtype=torch.float32)
    reduction = lambda data=data: data.sum()
    for _ in range(3):
        reduction()
    bw = nbytes / (min(event_us(reduction) for _ in range(7)) * 1e-6)
    del reduction, data
    a = torch.randn((8192, 8192), device="cuda", dtype=torch.bfloat16)
    b, c = torch.randn_like(a), torch.empty_like(a)
    mm = lambda: torch.matmul(a, b, out=c)
    for _ in range(3):
        mm()
    peak = (2 * 8192**3) / (min(event_us(mm) for _ in range(7)) * 1e-6)
    return {
        "read_bytes_per_s": bw,
        "bf16_flops_per_s": peak,
        "read_buffer_bytes": nbytes,
        "l2_bytes": l2 or None,
        "matmul_shape": [8192, 8192, 8192],
    }


def sol(xs, weights, ep, roof):
    counts = []
    local_routes = []
    for j, x in enumerate(xs):
        w = weights[j % len(weights)]
        _, ids = routing(x, w)
        local = w.expert_map[ids.long()]
        counts.append(int(local[local >= 0].unique().numel()))
        local_routes.append(int((local >= 0).sum().item()))
    t, si = xs[0].shape[0], SHARED_I // ep
    nbytes = E * H * 2 + statistics.mean(counts) * 3 * EXPERT_I * H * 2 + 3 * si * H * 2 + 4 * t * H
    # Only the routes this rank computes count: t * TOP_K at EP1, about half at EP2.
    flops = 2 * t * (E * H + 3 * si * H) + 2 * statistics.mean(local_routes) * 3 * EXPERT_I * H
    floor = max(nbytes / roof["read_bytes_per_s"], flops / roof["bf16_flops_per_s"]) * 1e6
    return floor, {
        "bytes": nbytes,
        "flops": flops,
        "unique_local_experts_mean": statistics.mean(counts),
        "local_routes_mean": statistics.mean(local_routes),
        "flops_convention": "router + shared I-slice for every token, routed experts for local routes",
    }


def bootstrap(rounds, seed, samples=10000):
    if "ours:auto" not in rounds:
        return None, None
    markets = [p for p in MARKETS if p in rounds]
    n = len(rounds["ours:auto"])
    if not markets or n < 2:
        return None, None
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        ours = statistics.median(rounds["ours:auto"][i] for i in indices)
        market = min(statistics.median(rounds[p][i] for i in indices) for p in markets)
        ratios.append(market / ours)
    ratios.sort()
    return ratios[int(0.025 * samples)], ratios[int(0.975 * samples)]


def check_blocks(blocks, xs, weights, reporter):
    from ctikernels.moe._routing_check import classify_topk

    errors = dict.fromkeys(blocks, 0.0)
    ties = dict.fromkeys(blocks, 0)
    mismatches = dict.fromkeys(blocks, 0)
    routed = set()  # providers that expose their routing record
    failed = set()
    nonfinite = set()
    # Check every independent layer on exactly the first L measured input slots.
    for j, w in enumerate(weights):
        ref = reference(xs[j], w)
        rms = ref.square().mean().sqrt().clamp_min(1e-12)
        for name, block in blocks.items():
            if name in failed or name in nonfinite:
                continue
            try:
                actual = block(xs[j], w, block.prepare(xs[j])).float()
                target = ref
                if block.op is not None and block.op.routing_ids is not None:
                    # Our providers expose their selection: check it against an independent
                    # FP64 top-k (only genuine BF16-logit boundary ties may differ, the GPU
                    # tests' rule), then condition the output reference on it.
                    chosen = block.op.routing_ids.long()
                    tie, mismatch = classify_topk(xs[j], w.router, chosen)
                    routed.add(name)
                    ties[name] += tie
                    mismatches[name] += mismatch
                    target = reference(xs[j], w, chosen)
                err = float(((actual - target).abs().max() / rms).item())
                errors[name] = max(errors[name], err) if math.isfinite(err) else None
                if errors[name] is None:
                    nonfinite.add(name)
            except (ValueError, NotImplementedError) as exc:
                failed.add(name)
                errors[name] = None
                if name != "ours:triton_block":
                    reporter.fail(name, "correctness execution", exc)
            except Exception as exc:
                failed.add(name)
                errors[name] = None
                reporter.fail(name, "correctness execution", exc)
    for name, err in errors.items():
        if name not in failed:
            bad = err is None or err > 0.03 or mismatches[name] > 0
            print(
                f"CHECK {name} max_rel_err={err} route_ties={ties[name]} "
                f"route_mismatches={mismatches[name]} {'FLAG' if bad else 'ok'}",
                flush=True,
            )
    # Nonfinite outputs must still be timed and flagged, like large finite errors.
    return errors, failed, {n: (ties[n], mismatches[n]) for n in routed}


def run_row(args, rank, t, weights, sgl, fi, roof, reporter, active):
    seed = args.seed + 7919 * t
    blocks, unavailable = {}, {}
    sconfig = None
    if "sglang" in active:
        try:
            sconfig = sglang_config(sgl, t, weights[0])
        except Exception as exc:
            reporter.fail("sglang", "config lookup", exc)
            unavailable["sglang"] = "config lookup failed"
    for name in active:
        if name in unavailable:
            continue
        try:
            blocks[name] = Block(name, args, rank, t, sgl, fi)
        except Exception as exc:
            reporter.fail(name, "construction", exc)
            unavailable[name] = "construction failed"
    xs = make_inputs(t, args.layers, seed)
    errors = dict.fromkeys(blocks)
    routes = {}
    if not args.skip_check:
        errors, failed, routes = check_blocks(blocks, xs, weights, reporter)
        for name in failed:
            blocks.pop(name)
            unavailable[name] = (
                "n/a: unsupported shape"
                if name == "ours:triton_block"
                else "check execution failed"
            )
    if args.check_only:
        return None, {}
    k = args.layers
    # A shared K, determined from actual graph/eager loops, makes each round a
    # paired comparison on exactly the same layer and input sequence.
    while blocks:
        loops, durations = {}, {}
        for name, block in list(blocks.items()):
            try:
                loops[name] = TimedLoop(block, xs, weights)
                durations[name] = min(loops[name].time() for _ in range(3))
            except (ValueError, NotImplementedError) as exc:
                if name != "ours:triton_block":
                    reporter.fail(name, "warmup/capture", exc)
                unavailable[name] = (
                    "n/a: unsupported shape"
                    if name == "ours:triton_block"
                    else "warmup/capture failed"
                )
                blocks.pop(name)
                loops.pop(name, None)
            except Exception as exc:
                reporter.fail(name, "warmup/capture", exc)
                unavailable[name] = "warmup/capture failed"
                blocks.pop(name)
                loops.pop(name, None)
        if not durations or min(durations.values()) >= 2000:
            break
        shortest = min(durations.values())
        k = max(k + args.layers, math.ceil(k * 2200 / max(shortest, 1) / args.layers) * args.layers)
        loops.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        xs = make_inputs(t, k, seed)
    else:
        loops = {}
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
    floor, sol_notes = sol(xs, weights, args.ep, roof)
    medians = {p: statistics.median(times) for p, times in rounds.items()}
    bad = [
        p
        for p in errors
        if not args.skip_check
        and (
            (errors[p] is None and p in blocks)
            or (errors[p] is not None and errors[p] > 0.03)
            or routes.get(p, (0, 0))[1] > 0
        )
    ]
    common = {"device": torch.cuda.get_device_name(), "ep": args.ep, "rank": rank, "T": t}
    for name in args.providers:
        notes = {
            "K": k,
            "layers": args.layers,
            "timing": "graph" if name in blocks and blocks[name].graph_mode else "eager",
            "status": "ok" if name in rounds else unavailable.get(name, "unavailable"),
            "check": (
                "skipped"
                if args.skip_check
                else "unavailable"
                if name not in errors or name in unavailable
                else "flagged"
                if name in bad
                else "checked"
            ),
            "row_flagged": bool(bad),
            "route_ties_mismatches": routes.get(name),
            "sol": sol_notes,
            "sglang_config": sconfig,
        }
        if name.startswith("vllm") and fi is not None:
            notes.update(
                flashinfer_version=fi.version,
                autotune_ran=fi.tuned,
                autotune_seconds=fi.tune_seconds,
                tuning_buckets=fi.tuning_buckets,
                topk_schema=fi.schema,
            )
        if name == "sglang" and sgl is not None:
            notes.update(shims=sgl.shims, sglang_version=sgl.version)
        if name in blocks and blocks[name].op is not None:
            notes["selected_provider"] = str(blocks[name].op.provider)
        reporter.emit(
            {
                "type": "provider",
                **common,
                "provider": name,
                "median_us": medians.get(name),
                "min_us": min(rounds[name]) if name in rounds else None,
                "rounds_us": rounds.get(name, []),
                "sol_us": floor,
                "pct_sol": 100 * floor / medians[name] if name in medians else None,
                "max_rel_err": errors.get(name),
                "config": sconfig if name == "sglang" else None,
                "notes": notes,
            }
        )
    markets = {p: medians[p] for p in MARKETS if p in medians}
    ours = {p: v for p, v in medians.items() if p.startswith("ours:")}
    best = min(markets, key=markets.get) if markets else None
    ours_best = min(ours, key=ours.get) if ours else None
    auto = medians.get("ours:auto")
    lo, hi = bootstrap(rounds, seed + rank)
    summary = {
        "type": "summary",
        **common,
        "ours_auto_us": auto,
        "ours_provider": str(blocks["ours:auto"].op.provider) if "ours:auto" in rounds else None,
        "ours_best_name": ours_best,
        "ours_best_us": ours.get(ours_best),
        "market_best_us": markets.get(best),
        "market_best_name": best,
        "ratio": markets[best] / auto if best and auto else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "row_flagged": bool(bad),
        "flagged_providers": bad,
        "market_complete": all(p in rounds for p in MARKETS),
    }
    reporter.emit(summary)
    # Destruction happens at row return; graphs and their pools stay alive through
    # all paired rounds and the correctness/summary reporting.
    return summary, medians


def import_checks(args, reporter):
    sgl, fi = None, None
    active = list(args.providers)
    try:
        fi_version = importlib.import_module("flashinfer").__version__
    except Exception:
        try:
            fi_version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            fi_version = "unavailable"
    reporter.emit(
        {
            "type": "header",
            "torch": torch.__version__,
            "flashinfer": fi_version,
            "required_flashinfer": "0.7.0.post1",
            "providers": active,
            "seed": args.seed,
            "ep": args.ep,
            "tokens": args.tokens,
            "layers": args.layers,
            "rounds": args.rounds,
            "error_metric": "max absolute error / FP32 reference RMS",
            "error_bound": 0.03,
            "ci_method": "paired round bootstrap of ratio of medians; reselect market minimum",
        }
    )
    if "sglang" in active:
        try:
            if args.sglang_path is None:
                raise ValueError("--sglang-path is required")
            sgl = load_sglang(args.sglang_path)
            with sglang_topology(sgl, args.ep, args.rank or 0):
                assert sgl.runtime.get_exec().moe.enable_fused_moe_sum_all_reduce is False
            print("IMPORT sglang real kernels and published topology OK", flush=True)
            reporter.emit(
                {
                    "type": "imports",
                    "provider": "sglang",
                    "version": sgl.version,
                    "shims": sgl.shims,
                }
            )
        except Exception as exc:
            reporter.fail("sglang", "import/topology (requires SGLang Python dependencies)", exc)
            active.remove("sglang")
    if any(p.startswith("vllm") for p in active):
        try:
            # Schema inspection is meaningful even when the FI version is rejected.
            _, _, schema = load_topk(args.vllm_lib)
            print("SCHEMA " + schema, flush=True)
            fi = FlashInferBackend(args)
            reporter.emit(
                {
                    "type": "imports",
                    "provider": "vllm",
                    "flashinfer_version": fi.version,
                    "topk_schema": fi.schema,
                    "cutlass_signature": str(fi.signature),
                }
            )
        except Exception as exc:
            for p in list(active):
                if p.startswith("vllm"):
                    reason = (
                        "requires FlashInfer 0.7.0.post1 or --allow-fi-version"
                        if fi_version != "0.7.0.post1" and not args.allow_fi_version
                        else "import/schema"
                    )
                    reporter.fail(p, reason, exc)
                    active.remove(p)
    return sgl, fi, active


def main(argv=None):
    args = parse_args(argv)
    reporter = Reporter(args.out)
    try:
        sgl, fi, active = import_checks(args, reporter)
        if args.check_imports:
            return
        if not torch.cuda.is_available():
            print("UNAVAILABLE CUDA no visible CUDA device", flush=True)
            return
        # Keep the FP32 reference honest; this does not change BF16 GEMM inputs.
        torch.backends.cuda.matmul.allow_tf32 = False
        summaries = []
        with torch.inference_mode():
            roof = ceilings()
            reporter.emit({"type": "ceilings", "device": torch.cuda.get_device_name(), **roof})
            ranks = [args.rank] if args.rank is not None else range(args.ep)
            for rank in ranks:
                weights = make_weights(args, rank, fi is not None)
                if fi is not None:
                    try:
                        fi.tune(args, weights, rank)
                        reporter.emit(
                            {
                                "type": "autotune",
                                "ran": fi.tuned,
                                "seconds": fi.tune_seconds,
                                "max_T": max(args.tokens),
                                "tuning_buckets": fi.tuning_buckets,
                            }
                        )
                    except Exception as exc:
                        for p in list(active):
                            if p.startswith("vllm"):
                                reporter.fail(p, "autotune", exc)
                                active.remove(p)
                        fi = None
                topology = (
                    sglang_topology(sgl, args.ep, rank) if "sglang" in active else nullcontext()
                )
                with topology:
                    for t in args.tokens:
                        print(f"ROW ep={args.ep} rank={rank} T={t}", flush=True)
                        summaries.append(
                            run_row(args, rank, t, weights, sgl, fi, roof, reporter, active)
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                del weights
                gc.collect()
                torch.cuda.empty_cache()
        print("\nMedian microseconds (n/a = unavailable); ratio > 1 favors ours:auto", flush=True)
        print("EP rank T " + " ".join(args.providers) + " ratio [95% CI]", flush=True)
        for summary, medians in summaries:
            if summary is None:
                continue
            cells = [f"{medians[p]:.2f}" if p in medians else "n/a" for p in args.providers]
            ratio = f"{summary['ratio']:.3f}" if summary["ratio"] is not None else "n/a"
            ci = (
                f"[{summary['ci_lo']:.3f}, {summary['ci_hi']:.3f}]"
                if summary["ci_lo"] is not None
                else "[n/a]"
            )
            flag = " FLAG" if summary["row_flagged"] else ""
            print(
                f"{summary['ep']} {summary['rank']} {summary['T']} "
                + " ".join(cells)
                + f" {ratio} {ci}{flag}",
                flush=True,
            )
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
