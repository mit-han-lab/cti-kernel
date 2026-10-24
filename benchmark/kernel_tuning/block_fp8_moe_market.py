# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Paired, single-GPU Qwen block-FP8 expert timing (no router or collective).

Each graph has eight independent inputs/routings. SGLang uses eight private
in-place buffers PER graph, restored from immutable inputs before EVERY replay,
including warmup. Copies precede the start event and are excluded from timing.
EP2 measures rank zero's partial result; SGLang's dispatcher map lookup IS timed.
The other sides consume global IDs and the expert map in their normal kernels.

Two graphs per side provide self-pair noise estimates. All graph variants are
shuffled independently each round. Verdicts concern timing only; inspect rel-L2
errors separately. Reference math uses original checkpoint scales, FP32 GEMMs
(TF32 disabled), and no activation quantization. DeepGEMM's requantization is
performed on private weights, outside timing, never on the reference weights.

SOL is a cold, once-per-touched-expert weight-read floor plus minimal BF16
activation traffic (input read, intermediate write/read, output write). It omits
quantization, padding, routing, extra scratch traffic and launch costs; it is
not a prediction of measured DRAM traffic. Cache residency can exceed this floor.
The deterministic floor (``floor_*``) is the bar for every side: weights and scales, the
BF16 input read and output write, plus the per-route BF16 down-projection buffer written and
read once (``4 * routes * H`` bytes). A route-major grouped GEMM cannot sum a token's top-k
products inside one program, so a deterministic ordered sum needs that buffer; the
intermediate activation is not counted (FC1 can feed FC2 its quantized input directly).
The row-level floor uses FP32 checkpoint scales; each side's floor uses its
actual weight/scale tensor sizes, including packed DeepGEMM scales.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import sys
import traceback
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

from benchmark.kernel_tuning.block_fp8_moe_sol import SHAPES

BATCHES = "1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096"
SIDES = ("ours", "sglang", "vllm", "vllm-triton")


def emit(prefix, record):
    print(
        prefix + json.dumps(record, separators=(",", ":"), default=str, allow_nan=False), flush=True
    )


def safe_error(error):
    # Exception text can contain paths or environment-derived credentials.
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__)[-3:])
    for key, value in os.environ.items():
        if value and ("SECRET" in key.upper() or "TOKEN" in key.upper()):
            text = text.replace(value, "<redacted>")
    text = re.sub(r"https?://\S+", "<url>", text)
    return text[-1800:]


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextmanager
def vllm_mode(triton_only):
    """Scope cached Python choices; captured graphs contain no Python dispatch.

    Both vLLM sides share one module tree. Clear both E8M0 caches at boundaries,
    including the scale-format oracle, so Triton never inherits packed scales.
    No kernels are substituted. This standalone harness is single-threaded.
    """
    key = "VLLM_USE_DEEP_GEMM_E8M0"
    previous = os.environ.get(key)
    dg = envs = None

    def clear():
        if envs is not None and hasattr(envs.__getattr__, "cache_clear"):
            envs.__getattr__.cache_clear()
        if dg is not None:
            dg.is_deep_gemm_e8m0_used.cache_clear()
            if hasattr(dg.DeepGemmQuantScaleFMT, "_oracle_cache"):
                delattr(dg.DeepGemmQuantScaleFMT, "_oracle_cache")

    try:
        if triton_only:
            os.environ[key] = "0"
        import vllm.envs as envs
        from vllm.utils import deep_gemm as dg

        clear()
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
        clear()


@dataclass
class Side:
    run: object
    scope: object = nullcontext
    inplace: bool = False
    info: dict = field(default_factory=dict)
    op: object = None
    describe: object = None


def expert_bytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors) / tensors[0].shape[0]


def sglang_config_source(module, e, i, down=False):
    configs = module.get_moe_configs(e, i, "fp8_w8a8", 128, 128, down_moe=down)
    if not configs:
        return {"source": "default"}
    import triton

    root = Path(os.environ.get("SGLANG_MOE_CONFIG_DIR", Path(module.__file__).parent)) / "configs"
    current = root / ("triton_" + triton.__version__.replace(".", "_"))
    versions = sorted(
        root.glob("triton_*"),
        key=lambda path: tuple(map(int, path.name.removeprefix("triton_").split("_"))),
        reverse=True,
    )
    # The resolver can reuse an up-projection JSON when a down JSON is absent.
    for is_down in (True, False) if down else (False,):
        filename = module.get_config_file_name(e, i, "fp8_w8a8", [128, 128], down_moe=is_down)
        for directory in dict.fromkeys([current, *versions]):
            if (directory / filename).is_file():
                return {"source": "tuned_json", "file": f"{directory.name}/{filename}"}
    raise RuntimeError("SGLang returned tuned configs but their source JSON is missing")


def sglang_dispatch(w13_shape, w2_shape, top_k, m):
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as config,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils import get_config

    up, (down, _) = config.try_get_optimal_moe_config(
        w13_shape,
        w2_shape,
        top_k,
        "fp8_w8a8",
        m,
        block_shape=[128, 128],
        return_down_config=True,
    )
    e, _, i = w2_shape
    sources = (
        [{"source": "override"}] * 2
        if get_config()
        else [sglang_config_source(config, e, i, down=d) for d in (False, True)]
    )
    return {
        "config_lookup": {
            "E": e,
            "N": i,
            "dtype": "fp8_w8a8",
            "block_shape": [128, 128],
            "fc1": {**sources[0], "config": up},
            "fc2": {**sources[1], "config": down or up},
        }
    }


def build_side(name, full, ep, batches, processed, shape="qwen3.5"):
    E, H, INTERMEDIATE, K = SHAPES[shape]
    n = E // ep
    w13, s13, w2, s2 = (t[:n] for t in full)
    emap = None
    if ep == 2:
        emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        emap[:n] = torch.arange(n, device="cuda", dtype=torch.int32)
    if name == "ours":
        from ctikernels.moe_block_fp8 import BlockFP8MoE

        op = BlockFP8MoE(w13, s13, w2, s2, top_k=K, expert_map=emap)
        op.prepare(batches)
        return Side(
            op,
            op=op,
            info={
                "scale_dtype": str(s13.dtype),
                "weight_scale_bytes_per_expert": expert_bytes(w13, s13, w2, s2),
            },
        )
    if name == "sglang":
        from sglang.srt.runtime_context import get_context

        def scope():
            return get_context().override_server_args(
                moe_runner_backend="triton",
                moe_a2a_backend="none",
                enable_fused_moe_sum_all_reduce=False,
                enable_deterministic_inference=False,
            )

        with scope():
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
            from sglang.srt.layers.moe.topk import StandardTopKOutput

            cfg = MoeRunnerConfig(
                num_experts=E,
                num_local_experts=n,
                hidden_size=H,
                intermediate_size_per_partition=INTERMEDIATE,
                top_k=K,
                activation="silu",
                is_gated=True,
                inplace=True,
                routed_scaling_factor=1.0,
            )
        fs13, fs2 = s13.float(), s2.float()

        def run(x, weights, ids):
            # StandardDispatcher's exact map operation, included in capture.
            local_ids = emap[ids] if emap is not None else ids
            top = StandardTopKOutput(weights, local_ids, None)
            return fused_experts(
                x,
                w13,
                w2,
                top,
                cfg,
                use_fp8_w8a8=True,
                w1_scale=fs13,
                w2_scale=fs2,
                block_shape=[128, 128],
            )

        return Side(
            run,
            scope,
            True,
            {
                "backend": "triton",
                "executed_experts": ["fused_experts"],
                "weight_scale_bytes_per_expert": expert_bytes(w13, fs13, w2, fs2),
                "scale_dtype": "torch.float32",
                "ep_remap_timed": ep == 2,
                "input_refresh_timed": False,
            },
            describe=lambda m: sglang_dispatch(w13.shape, w2.shape, K, m),
        )

    triton_only = name == "vllm-triton"
    scope = lambda: vllm_mode(triton_only)
    with scope():
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )
        from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            Fp8MoeBackend,
            make_fp8_moe_quant_config,
            select_fp8_moe_backend,
        )
        from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            prepare_fp8_moe_layer_for_deepgemm,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8Dynamic128Sym,
            kFp8Static128BlockSym,
        )
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
        from vllm.v1.worker.workspace import init_workspace_manager

        p = FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=1,
            ep_size=ep,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=ep > 1,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        )
        c = FusedMoEConfig(
            num_experts=E,
            experts_per_token=K,
            hidden_dim=H,
            intermediate_size=INTERMEDIATE,
            num_local_experts=n,
            num_logical_experts=E,
            activation=MoEActivation.SILU,
            device=w13.device,
            in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.RenormalizeNaive,
            moe_parallel_config=p,
            moe_backend="triton" if triton_only else "auto",
        )
        backend, cls = select_fp8_moe_backend(c, kFp8Static128BlockSym, kFp8Dynamic128Sym)
        if backend == Fp8MoeBackend.DEEPGEMM:
            if "deepgemm" not in processed:
                # The loader mutates weights/scales; process one private global copy.
                # EP1/EP2 then share storage, including the processed representation.
                a, sa, b, sb = full
                processed["deepgemm"] = prepare_fp8_moe_layer_for_deepgemm(
                    a.clone(), b.clone(), sa.float().clone(), sb.float().clone(), (128, 128)
                )
            w13, w2, fs13, fs2 = (t[:n] for t in processed["deepgemm"])
        else:
            fs13, fs2 = s13.float(), s2.float()
        q = make_fp8_moe_quant_config(backend, fs13, fs2, None, None, block_shape=[128, 128])
        experts = cls(c, q)
        info = {
            "backend": backend.name,
            "experts_class": cls.__name__,
            "loader_scale_dtype": "torch.float32",
            "weight_scale_bytes_per_expert": expert_bytes(w13, fs13, w2, fs2),
            "runtime_scale_dtype": str(fs13.dtype),
            "e8m0": is_deep_gemm_e8m0_used(),
            "executed_experts": [],
        }

        def describe(m):
            if backend != Fp8MoeBackend.DEEPGEMM:
                return {}
            from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
                _valid_deep_gemm_shape,
            )
            from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

            alignment = get_mk_alignment_for_contiguous_layout()[0]
            return {
                "deepgemm_shape_check": {
                    "alignment": alignment,
                    "fc1_aligned": _valid_deep_gemm_shape(alignment, 2 * INTERMEDIATE, H),
                    "fc2_aligned": _valid_deep_gemm_shape(alignment, H, INTERMEDIATE),
                    "runtime_shape_valid": _valid_deep_gemm_shape(m, INTERMEDIATE, H),
                    "ue8m0_bypasses_small_m_fallback": info["e8m0"],
                }
            }

        # Observe the real runtime pick with the actual quantized input. Recording
        # happens in eager/capture Python, and adds no nodes to a replay.
        if hasattr(experts, "_select_experts_impl"):
            select = experts._select_experts_impl

            def observed(*args, **kwargs):
                chosen = select(*args, **kwargs)
                label = type(chosen).__name__
                if label not in info["executed_experts"]:
                    info["executed_experts"].append(label)
                return chosen

            experts._select_experts_impl = observed
        else:
            info["executed_experts"].append(cls.__name__)
        init_workspace_manager(w13.device)
        kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), experts)

    def run(x, weights, ids):
        return kernel.apply(x, w13, w2, weights, ids, MoEActivation.SILU, E, emap, False)

    return Side(run, scope, info=info, describe=describe)


def random_weight(n, d, scale_dtype, shape="qwen3.5"):
    E, _, _, _ = SHAPES[shape]
    # Generate one expert at a time to avoid a multi-GiB FP32 temporary on GB10.
    w = torch.empty((E, n, d), device="cuda", dtype=torch.float8_e4m3fn)
    for expert in range(E):
        w[expert].copy_((torch.randn(n, d, device="cuda") * 96).clamp_(-448, 448))
    # BF16 checkpoint precision is shared by ALL sides, including --scales fp32.
    s = torch.exp2(torch.rand(E, n // 128, d // 128, device="cuda") * 6 - 14)
    return w, s.bfloat16().to(scale_dtype)


def make_cases(m, routing, seed, shape="qwen3.5"):
    E, H, _, K = SHAPES[shape]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bias = torch.zeros(E, device="cuda")
    if routing == "skewed":
        bias[:16] = 5.0
    cases = []
    for _ in range(8):
        probs = (torch.randn(m, E, device="cuda", generator=gen) + bias).softmax(-1)
        weights, ids = probs.topk(K, dim=-1)
        x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16, generator=gen)
        cases.append((x, weights / weights.sum(-1, keepdim=True), ids.int()))
    return cases


def reference(cases, full, ep, shape="qwen3.5"):
    """FP32 partial rank result, expert-at-a-time dequantization, CPU results."""
    E, _, _, _ = SHAPES[shape]
    w13, s13, w2, s2 = full
    m = cases[0][0].shape[0]
    x = torch.cat([c[0] for c in cases]).float()
    weights = torch.cat([c[1] for c in cases])
    ids = torch.cat([c[2] for c in cases])
    out = torch.zeros_like(x)
    for e in range(E // ep):
        rows, slots = torch.where(ids == e)
        if not rows.numel():
            continue
        a = w13[e].float() * s13[e].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        b = w2[e].float() * s2[e].float().repeat_interleave(128, 0).repeat_interleave(128, 1)
        gate, up = (x[rows] @ a.T).chunk(2, dim=-1)
        result = (torch.nn.functional.silu(gate) * up) @ b.T
        out.index_add_(0, rows, result * weights[rows, slots, None])
    return list(out.cpu().split(m))


def rel_l2(actual, expected):
    numerator = sum(
        (a.double() - b.double()).square().sum().item()
        for a, b in zip(actual, expected, strict=True)
    )
    denominator = sum(b.double().square().sum().item() for b in expected)
    value = math.sqrt(numerator / max(denominator, 1e-300))
    return value if math.isfinite(value) else None


class Replay:
    def __init__(self, side, cases, stream):
        self.side, self.cases = side, cases
        self.inputs = [c[0].clone() for c in cases] if side.inplace else [c[0] for c in cases]
        self.graph = torch.cuda.CUDAGraph()
        self.refresh()
        with side.scope(), torch.cuda.graph(self.graph, stream=stream):
            self.outputs = [
                side.run(x, weights, ids)
                for x, (_, weights, ids) in zip(self.inputs, cases, strict=True)
            ]
        self.refresh()
        self.graph.replay()
        torch.cuda.synchronize()

    def refresh(self):
        if self.side.inplace:
            for target, (source, _, _) in zip(self.inputs, self.cases, strict=True):
                target.copy_(source)

    def sample(self, replays):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        times = []
        for _ in range(replays):
            self.refresh()  # same stream; copy completes BEFORE the start event
            start.record()
            self.graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / len(self.cases))
        return statistics.median(times)


def eps_of(a, b):
    ratios = [x / y for x, y in zip(a, b, strict=True)]
    q1, _, q3 = statistics.quantiles(ratios, n=4)
    return max(abs(statistics.median(ratios) - 1), (q3 - q1) / 2)


def speedup_statistics(ours_a, ours_b, side_a, side_b, *, seed, m, ep, side, bootstrap=10000):
    """Percentile bootstrap of median side/ours speedup, using both graph pairs.

    Each ratio pairs the same variant and interleaved round. Resample the pooled
    2*pairs ratios with replacement; interpolate the 2.5/97.5 percentiles.
    Pair count and X925 affinity qualification remain the launcher's responsibility.
    """
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    samples = []
    for oa, ob, sa, sb in zip(ours_a, ours_b, side_a, side_b, strict=True):
        if any(not math.isfinite(t) or t <= 0 for t in (oa, ob, sa, sb)):
            raise ValueError("timings must be positive and finite")
        samples.extend((sa / oa, sb / ob))
    if not samples:
        raise ValueError("at least one timing pair required")
    # String seeding is stable across processes (unlike Python's hash()).
    rng = random.Random(json.dumps([seed, m, ep, side], separators=(",", ":")))
    medians = sorted(
        statistics.median(rng.choices(samples, k=len(samples))) for _ in range(bootstrap)
    )

    def percentile(q):
        index = (len(medians) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return medians[lo] + (medians[hi] - medians[lo]) * (index - lo)

    lo, hi = percentile(0.025), percentile(0.975)
    return {
        "speedup_vs_" + side: statistics.median(samples),
        "speedup_ci95_vs_" + side: [lo, hi],
        "gb10_rule_vs_" + side: "pass" if lo > 1 else "fail" if hi < 1 else "inconclusive",
        "rule_pairs": len(samples) // 2,
    }


def bandwidth(args):
    bw = args.dram_gbps
    if bw is None and args.ceilings:
        data = json.loads(Path(args.ceilings).read_text())
        bw = data.get("dram_gbps", data.get("memory", {}).get("dram_read_gbs"))
        if bw is None:
            raise ValueError("ceilings requires dram_gbps or memory.dram_read_gbs")
    if bw is not None and (not math.isfinite(bw) or bw <= 0):
        raise ValueError("DRAM bandwidth must be positive and finite")
    return bw


def sol(cases, ep, scale_bytes, bw, runtime_expert_bytes=None, shape="qwen3.5"):
    E, H, INTERMEDIATE, _ = SHAPES[shape]
    local = [ids[ids < E // ep] for _, _, ids in cases]
    unique = statistics.mean(ids.unique().numel() for ids in local)
    routes = statistics.mean(ids.numel() for ids in local)
    m = cases[0][0].shape[0]
    weight_bytes = unique * (
        3 * H * INTERMEDIATE + scale_bytes * (2 * INTERMEDIATE * H + H * INTERMEDIATE) // 128**2
    )
    if runtime_expert_bytes is not None:
        weight_bytes = unique * runtime_expert_bytes
    activation_bytes = 4 * m * H + 4 * routes * INTERMEDIATE
    total = weight_bytes + activation_bytes
    floor = weight_bytes + 4 * m * H + 4 * routes * H
    return {
        "unique_local_experts": unique,
        "local_routes": routes,
        "weight_scale_read_bytes": weight_bytes,
        "activation_traffic_bytes": activation_bytes,
        "bytes": total,
        "dram_gbps": bw,
        "sol_us": total / (bw * 1000) if bw else None,
        "floor_down_buffer_bytes": 4 * routes * H,
        "floor_bytes": floor,
        "floor_us": floor / (bw * 1000) if bw else None,
    }


def run_row(args, ep, m, full, sides, status, bw):
    cases = make_cases(m, args.routing, args.seed + m, args.shape)
    refs = reference(cases, full, ep, args.shape)
    graphs, outputs, timing = {}, {}, {}
    e, h, i, top_k = SHAPES[args.shape]
    record = {
        "shape": args.shape,
        "E_global": e,
        "E_local": e // ep,
        "H": h,
        "I": i,
        "top_k": top_k,
        "block_shape": [128, 128],
        "M": m,
        "ep": ep,
        "rank": 0,
        "routing": args.routing,
        "pairs": args.pairs,
        "rule_pairs": args.pairs,
        "bootstrap": args.bootstrap,
        "replays": args.replays,
        "calls_per_graph": 8,
        "seed": args.seed,
        "scales": args.scales,
        "correctness": {},
        "sides": {},
        "sol": sol(cases, ep, 4, bw, shape=args.shape),
        "sol_model": "cold weights + minimal BF16 traffic",
    }

    def drop(name, error):
        status[name] = {"status": "dropped", "ep": ep, "M": m, "error": safe_error(error)}
        emit("", {"side": name, **status[name]})
        sides.pop(name, None)
        graphs.pop((name, "A"), None)
        graphs.pop((name, "B"), None)
        outputs.pop(name, None)
        record["sides"].pop(name, None)
        record["correctness"].pop(name, None)

    manager = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        # vLLM shares a process-wide scratch allocation. Grow it for EVERY side
        # before capture, then lock it so later captures cannot invalidate pointers.
        if any(name.startswith("vllm") for name in sides):
            from vllm.v1.worker.workspace import current_workspace_manager

            manager = current_workspace_manager()
            manager.unlock()
        for name, side in list(sides.items()):
            try:
                with side.scope(), torch.cuda.stream(stream):
                    for x, weights, ids in cases:
                        side.run(x.clone() if side.inplace else x, weights, ids)
                torch.cuda.synchronize()
            except Exception as error:  # noqa: BLE001 - optional side failures must be isolated
                drop(name, error)
        if manager is not None:
            manager.lock()
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for name, side in list(sides.items()):
                try:
                    if "executed_experts" in side.info and hasattr(side, "run"):
                        # Reset per-row runtime observations for fallback wrappers.
                        if side.info.get("experts_class") == "TritonOrDeepGemmExperts":
                            side.info["executed_experts"].clear()
                    graphs[name, "A"] = Replay(side, cases, stream)
                    graphs[name, "B"] = Replay(side, cases, stream)
                    outputs[name] = [y.float().cpu() for y in graphs[name, "A"].outputs]
                    other_outputs = [y.float().cpu() for y in graphs[name, "B"].outputs]
                    record["correctness"][name] = {
                        "rel_l2_fp32": rel_l2(outputs[name], refs),
                        "rel_l2_self": rel_l2(other_outputs, outputs[name]),
                    }
                    record["sides"][name] = dict(side.info)
                    if side.describe is not None:
                        with side.scope():
                            record["sides"][name].update(side.describe(m))
                    if name == "ours":
                        record["sides"][name].update(
                            provider=side.op.provider_for(m), config=side.op.config_for(m)
                        )
                except Exception as error:  # noqa: BLE001 - optional side failures must be isolated
                    drop(name, error)
        finally:
            if was_enabled:
                gc.enable()
        rng = random.Random(args.seed + 100003 * ep + m)
        for round_index in range(3 + args.pairs):
            order = list(graphs)
            rng.shuffle(order)
            for key in order:
                if key not in graphs:
                    continue
                try:
                    us = graphs[key].sample(args.replays)
                    if not math.isfinite(us) or us <= 0:
                        raise ValueError("invalid CUDA event duration")
                    if round_index >= 3:
                        timing.setdefault(key, []).append(us)
                except Exception as error:  # noqa: BLE001 - optional side failures must be isolated
                    drop(key[0], error)
        for name in sides:
            a, b = timing[name, "A"], timing[name, "B"]
            us = statistics.median(a + b)
            stats = record["sides"][name]
            stats.update(us=us, self_eps=eps_of(b, a))
            floor = sol(
                cases,
                ep,
                2 if name == "ours" and args.scales == "bf16" else 4,
                bw,
                stats.get("weight_scale_bytes_per_expert"),
                args.shape,
            )
            stats["sol_bytes"] = floor["bytes"]
            stats["floor_bytes"] = floor["floor_bytes"]
            if bw:
                stats.update(sol_us=floor["sol_us"], sol_util=floor["sol_us"] / us)
                stats.update(floor_us=floor["floor_us"], floor_util=floor["floor_us"] / us)
            if "ours" in outputs:
                record["correctness"][name]["rel_l2_ours"] = rel_l2(outputs[name], outputs["ours"])
        if "ours" in sides:
            for name in sides:
                if name == "ours":
                    continue
                ratio = statistics.median(
                    a / b for a, b in zip(timing["ours", "A"], timing[name, "A"], strict=True)
                )
                eps = max(record["sides"]["ours"]["self_eps"], record["sides"][name]["self_eps"])
                record["ratio_ours_vs_" + name] = ratio
                record["eps_ours_vs_" + name] = eps
                record["verdict_ours_vs_" + name] = (
                    "faster" if ratio < 1 - eps else "slower" if ratio > 1 + eps else "inconclusive"
                )
                record.update(
                    speedup_statistics(
                        timing["ours", "A"],
                        timing["ours", "B"],
                        timing[name, "A"],
                        timing[name, "B"],
                        seed=args.seed,
                        m=m,
                        ep=ep,
                        side=name,
                        bootstrap=args.bootstrap,
                    )
                )
        if args.verbose:
            record["samples_us"] = {
                name + "/" + variant: values for (name, variant), values in timing.items()
            }
        emit("ROW ", record)
    finally:
        graphs.clear()
        outputs.clear()
        if manager is not None:
            manager.unlock()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    if os.environ.get("SGLANG_PATH"):
        # A package tree from the release wheel, without its client initializer.
        from benchmark.kernel_tuning._moe_market_imports import load_sglang

        load_sglang(Path(os.environ["SGLANG_PATH"]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="qwen3.5")
    parser.add_argument("--batches", default=BATCHES)
    parser.add_argument("--ep", default="1,2")
    parser.add_argument("--sides", default=",".join(SIDES))
    parser.add_argument("--scales", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ceilings")
    parser.add_argument("--dram-gbps", type=float)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    E, H, INTERMEDIATE, K = SHAPES[args.shape]
    try:
        args.batches = list(dict.fromkeys(int(x) for x in args.batches.split(",")))
        eps = list(dict.fromkeys(int(x) for x in args.ep.split(",")))
        names = list(dict.fromkeys(args.sides.split(",")))
        if min(args.batches) < 1 or not set(eps) <= {1, 2} or not set(names) <= set(SIDES):
            raise ValueError("invalid batches, EP or sides")
        if args.pairs < 2 or args.replays < 1:
            raise ValueError("pairs >= 2 and replays >= 1 required")
        if args.bootstrap < 1:
            raise ValueError("bootstrap >= 1 required")
        bw = bandwidth(args)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    status = {n: {"status": "pending"} for n in names}
    available = torch.cuda.is_available()
    host = {
        "shape": args.shape,
        "E_global": E,
        "H": H,
        "I": INTERMEDIATE,
        "top_k": K,
        "gpu": torch.cuda.get_device_name() if available else None,
        "cc": torch.cuda.get_device_capability() if available else None,
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count
        if available
        else None,
        "versions": {
            n: version(n)
            for n in (
                "torch",
                "triton",
                "flashinfer-python",
                "sglang",
                "sglang-kernel",
                "vllm",
                "deep-gemm",
                "transformers",
                "compressed-tensors",
                "torchvision",
                "apache-tvm-ffi",
            )
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "side_status": status,
    }
    emit("HOST ", host)
    if not available:
        parser.exit(2, "CUDA is required; no timings were collected.\n")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        dtype = torch.bfloat16 if args.scales == "bf16" else torch.float32
        w13, s13 = random_weight(2 * INTERMEDIATE, H, dtype, args.shape)
        w2, s2 = random_weight(H, INTERMEDIATE, dtype, args.shape)
        full, processed = (w13, s13, w2, s2), {}
        for ep in eps:
            sides = {}
            for name in names:
                if status[name]["status"] == "dropped":
                    continue
                try:
                    sides[name] = build_side(name, full, ep, args.batches, processed, args.shape)
                    status[name] = {"status": "ready", "ep": ep, **sides[name].info}
                except Exception as error:  # noqa: BLE001 - optional side failures must be isolated
                    status[name] = {"status": "dropped", "ep": ep, "error": safe_error(error)}
                    emit("", {"side": name, **status[name]})
            emit("HOST ", host)
            for m in args.batches:
                if not sides:
                    break
                run_row(args, ep, m, full, sides, status, bw)
            sides.clear()
            gc.collect()
            torch.cuda.empty_cache()
        host["versions"].pop("deep-gemm", None)
        dg = sys.modules.get("deep_gemm") or sys.modules.get("vllm.third_party.deep_gemm")
        host["versions"]["deep_gemm"] = (
            version("deep-gemm")
            or getattr(dg, "__version__", None)
            or ("vendored in vllm " + str(version("vllm")) if dg is not None else None)
        )
        emit("HOST ", host)
    return 0 if any(s["status"] == "ready" for s in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
