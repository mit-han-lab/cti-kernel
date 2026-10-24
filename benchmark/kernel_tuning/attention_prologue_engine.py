# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
# Dependency failures are reported with safe source locations, never raw diagnostics.

"""Paired engine-regime prologue benchmark; imports are CPU-safe.

Dispatch audit of the supplied vLLM 0.31.0 wheel (line numbers refer to that wheel):
* model_executor/models/qwen3.py:153-169 splits Q/K/V, views per head, calls
  RMSNorm on Q/K, then get_rope's RotaryEmbedding and Attention.
* models/qwen3_next.py:31 aliases GemmaRMSNorm as Qwen3NextRMSNorm;
  models/qwen3_5.py:154-162 uses Qwen3NextAttention for full attention.
  qwen3_next.py:361-446 has TWO paths. Gate-disabled attention calls the native
  norm/rotary sequence. Default gated CUDA BF16 text attention instead calls
  layers/fused_qk_norm_rope.py:fused_qk_rmsnorm_rope_gate (norm_beta=1).
  This model-level fusion is independent of compiler fusion flags. We import
  that real kernel for default hd256, including its gate copy. Ours/compose
  read the engine's merged [Q | K | V | gate] projection (cti-engine Qwen3.5
  qkv_gate_proj merges q, k, v, output_gate in that order): the same GEMM with
  permuted weight rows, then a strided Q|K|V view with no packing copy.
* layers/layernorm.py:74-94,159-168 forward_native calls ir.ops.rms_norm;
  ir/ops/layernorm.py:9-21 owns the real native expression.
  platforms/cuda.py:738 defaults IR priority to native when using Inductor.
  We expose it through public set_default(['native']) / set_default_torch_wrap(False) APIs
  to Inductor, avoiding replacement arithmetic or private compiler lowerings.
* config/vllm.py:1788-1795 defaults custom_ops to 'none' with Inductor compiling;
  model_executor/custom_op.py:189-192 therefore selects forward_native.
  config/vllm.py:267,290,313,336 disables enable_qk_norm_rope_fusion at every
  optimization level; config/compilation.py:296-307 restricts rope/cache fusion
  to ROCm. config/vllm.py also defaults use_inductor_graph_partition=False.
* layers/rotary_embedding/__init__.py:110-153 selects RotaryEmbedding for
  default text RoPE (MRotaryEmbedding with mrope_section; 1D text positions
  reduce to the same base rotation). base.py:160-219 calls common.py's real
  ApplyRotaryEmb.forward_static. Native tables match BF16 query dtype.
* layers/attention/attention.py:548-578,724-761 calls opaque
  unified_kv_cache_update -> FlashAttentionImpl.do_kv_cache_update;
  v1/attention/backends/flash_attn.py:1507-1542 calls reshape_and_cache_flash.
  config/compilation.py:1162-1189 adds unified_kv_cache_update to splitting_ops
  without Inductor graph partition. Here the real wheel cache op runs AFTER
  the compiled native region, inside the enclosing full CUDA graph.
* utils/torch_utils.py:39-41,425-438 maps fp8/fp8_e4m3/fp8_e5m2 to uint8
  storage. _custom_ops.py:2544-2563 takes key, value, key_cache, value_cache,
  slot_mapping, kv_cache_dtype, k_scale, v_scale. flash_attn.py:1532-1541
  passes the layer's FP32 device scale tensors. We use "fp8" for E4M3 and
  "fp8_e5m2" for E5M2, with uint8 views of the same FP8 cache layout.

SGLang 72730d6 audit (python/sglang/): srt/models/qwen3_5.py:1538-1542
selects forward_prepare_cuda_fused when attn_output_gate=True. Lines 1428-1456
call kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py's real Triton kernel:
zero-centered Gemma norm, partial NeoX RoPE, gate deinterleave. Gate-disabled
lines 1393-1426,1458-1479 call two sgl_kernel.gemma_rmsnorm operations then
JIT RoPE. Both use the existing MHATokenToKVPool -> JIT store_cache path.
The hd128 path remains fused_inplace_qknorm -> JIT RoPE -> JIT store_cache.
Additional hd256 dispatch sources are pinned by Git blob SHA-1, as are the
existing hd128 JIT sources. No third-party source is copied into this repo.
FP8 audit: srt/mem_cache/memory_pool.py:2860-2870 divides BF16 K/V in place
by optional Python float scales, casts to the pool dtype, then views its store
dtype (uint8). The scaled `sglang` stack reproduces those BF16 divisions;
`sglang_noscale` reproduces a checkpoint without KV scales (casts only).
Ours/compose and vLLM use FP32 device scales 0.0478/0.0213; the scaled
SGLang stack uses the same values. All FP8 checks use clipped FP64 K oracles:
ours <=1 FP8 ulp, compose/markets <=2; V is bitwise except scaled SGLang
(<=1 ulp). The gate uses elementwise spacing, including subnormals and saturation.

One graph runs L projections, prologues and fillers on one stream. All layers
reuse ONE projection activation buffer; out-of-place Q/K outputs are released
after each layer so the graph private pool can reuse their storage. KV pools
and weights are distinct per layer; indices, positions and tables are shared.
Default filler is BF16 o_proj + gate_up + down GEMMs at the row's token count,
with distinct model-sized weights per layer and shared activation scratch.
No attention/cache read or synthetic L2 eviction is added. --filler=legacy
enables the old synthetic GEMM/eviction diagnostic; --filler-width and
--evict-mib apply only to it. Diagnostic runs produce no faster/slower verdict.
Default --timing=span profiles one replay per CUDA-only profiler session (no
shapes/stacks). Projection-only and baseline graphs calibrate P and F kernels
per layer once per row. Each stack calibrates K once; every replay must have
exactly L*(P+K+F) kernels and matching projection/filler name multisets in each
layer. Invalid samples are reported and the entire pair is excluded. Sorting
GPU timestamps gives end(last prologue) - end(last projection), averaged over
layers: launch gaps into/within the prologue count, filler time does not.
Projection output is L2-hot here; SOL is a DRAM-bound reference only.

Each rotated/interleaved pair has its own baseline and an independently built
ours self graph. Self-pairs calibrate p95 relative noise; per-pair market/ours
ratios, exact binomial median intervals and a fresh >=40-pair retry apply.
The fastest median of real SGLang (both FP8 scale modes)/vLLM is the market;
surrogates never count.
--timing=subtract retains diagnostic graph-minus-baseline timing (including
allocation/layout effects). Span mode also retains these differences using
CUDA events around the same profiled replays. Nonpositive differences in
subtract mode are unresolved noise, never clamped into a speedup.
This is an isolated engine-regime measurement, not full serving throughput.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import statistics
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmark.kernel_tuning.attention_prologue_paired import (  # noqa: E402
    CEILINGS,
    diagnostic_verdicts,
    median_ci,
    percentile95,
)

KV_DTYPES = {"bf16": "bfloat16", "fp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}
KV_SCALES = (0.0478, 0.0213)

ROWS = (1, 2, 3, 4, 16, 32, 48, 64, 128, 256, 512, 1024, 4096)
VARIANTS = {
    "hd128": (32, 8, 128, 128, False),
    "hd256": (16, 4, 256, 64, True),
    "hd128-moe": (32, 4, 128, 128, False),
    "hd256-27b": (24, 4, 256, 64, True),
}
# Qwen3-8B, Qwen3.5-9B full attention, Qwen3-30B-A3B (hidden 2048; the filler MLP is the
# 8 routed experts' 8 x 768 intermediate as one dense GEMM pair) and Qwen3.8-27B full
# attention (TP1), respectively.
MODEL_SIZES = {
    "hd128": (4096, 12288),
    "hd256": (4096, 12288),
    "hd128-moe": (2048, 6144),
    "hd256-27b": (5120, 17408),
}
LIFETIME_MODEL = "shared projection buffer, outputs released per layer, per-layer KV pools"
# The SGLang 72730d6 python/ source tree (--sglang-python or SGLANG_PYTHON).
DEFAULT_SGLANG = Path(os.environ["SGLANG_PYTHON"]) if os.environ.get("SGLANG_PYTHON") else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, default="hd128")
    p.add_argument("--kv-dtype", choices=KV_DTYPES, default="bf16")
    p.add_argument("--rows", default=",".join(map(str, ROWS)))
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--timing", choices=("span", "subtract"), default="span")
    p.add_argument("--json", type=Path)
    p.add_argument("--sglang-python", type=Path, default=DEFAULT_SGLANG)
    p.add_argument("--vllm-cache", type=Path, default=Path("/tmp/attention-prologue-vllm-cache"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument(
        "--filler",
        choices=("model", "legacy"),
        default="model",
        help="model o_proj + MLP GEMMs, or legacy cache-pressure diagnostic",
    )
    p.add_argument(
        "--evict-mib",
        type=float,
        default=0,
        help="legacy diagnostic only: eviction buffer MiB, at least twice device L2",
    )
    p.add_argument(
        "--filler-width",
        type=int,
        default=12288,
        help="legacy diagnostic only: synthetic GEMM output width",
    )
    p.add_argument(
        "--no-output-gate", action="store_true", help="hd256 gate-disabled configuration"
    )
    p.add_argument(
        "--masked", action="store_true", help="ours is the masked (slot < 0 skipping) variant"
    )
    p.add_argument(
        "--check-vllm-source", action="store_true", help="CPU source import/forward only"
    )
    return p


def filler_shapes(variant):
    hidden, intermediate = MODEL_SIZES[variant]
    qh, _, hd, _, _ = VARIANTS[variant]
    return {
        "o_proj": (qh * hd, hidden),
        "gate_up": (hidden, 2 * intermediate),
        "down": (intermediate, hidden),
    }


def protocol_metadata(args, *, tokens=None, eviction_bytes=None):
    shapes = filler_shapes(args.variant)
    if args.filler == "legacy":
        shapes = {"synthetic": (MODEL_SIZES[args.variant][0], args.filler_width)}
    weights = 2 * sum(m * n for m, n in shapes.values())
    flags = {
        "legacy_filler": args.filler == "legacy",
        "filler_width_override": args.filler_width != 12288,
        "evict_mib_override": args.evict_mib != 0,
        "subtract_timing": args.timing == "subtract",
    }
    # Logical GEMM traffic: weights + input reads + output writes, not measured
    # DRAM traffic (cache residency and GEMM tiling can change the actual bytes).
    if tokens is None:
        moved = {
            value: protocol_metadata(args, tokens=int(value), eviction_bytes=eviction_bytes)[
                "filler_bytes_per_layer"
            ]
            for value in args.rows.split(",")
        }
    else:
        moved = weights + 2 * tokens * sum(m + n for m, n in shapes.values())
        if args.filler == "legacy":
            moved = None if eviction_bytes is None else moved + 2 * eviction_bytes
    return {
        "kv_dtype": args.kv_dtype,
        "k_scale": KV_SCALES[0] if args.kv_dtype != "bf16" else None,
        "v_scale": KV_SCALES[1] if args.kv_dtype != "bf16" else None,
        "sglang_noscale": args.kv_dtype != "bf16",
        "sglang_noscale_scales": {"k_scale": None, "v_scale": None},
        "sampling": "interleaved per-pair baseline + stacks + ours self-pair; static full CUDA graphs",
        "lifetime_model": LIFETIME_MODEL,
        "hidden": MODEL_SIZES[args.variant][0],
        "intermediate": MODEL_SIZES[args.variant][1],
        "num_q_heads": VARIANTS[args.variant][0],
        "num_kv_heads": VARIANTS[args.variant][1],
        "head_dim": VARIANTS[args.variant][2],
        "layers": args.layers,
        "timing": args.timing,
        "initial_pairs": args.pairs,
        "retry_pairs": max(40, 2 * args.pairs),
        "filler_kind": args.filler,
        "filler_dtype": "bfloat16",
        "filler_weight_shapes": shapes,
        "filler_weights": "distinct per layer" if args.filler == "model" else "shared (legacy)",
        "filler_weight_bytes_per_layer": weights,
        "filler_bytes_per_layer": moved,
        "filler_bytes_scope": "logical weight/input reads + output writes + legacy eviction read/write",
        "filler_width": args.filler_width,
        "evict_mib": args.evict_mib,
        "eviction_bytes": eviction_bytes if args.filler == "legacy" else 0,
        "diagnostic_flags": flags,
        "diagnostic": any(flags.values()),
    }


def print_protocol(protocol):
    print(
        f"Protocol: {protocol['lifetime_model']}; layers={protocol['layers']}; "
        f"kv_dtype={protocol['kv_dtype']}; "
        f"k_scale={protocol['k_scale']}; v_scale={protocol['v_scale']}; "
        f"sglang_noscale={protocol['sglang_noscale']}; "
        f"filler={protocol['filler_kind']}; "
        f"filler_weights={protocol['filler_weight_bytes_per_layer']} bytes/layer; "
        f"timing={protocol['timing']}; diagnostic={protocol['diagnostic']}; "
        f"flags={json.dumps(protocol['diagnostic_flags'], sort_keys=True)}",
        flush=True,
    )


def fresh_compile(function):
    # Avoid Dynamo's per-code-object specialization limit across 12 static sizes.
    clone = types.FunctionType(
        function.__code__.replace(),
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    return torch.compile(clone, backend="inductor", fullgraph=True, dynamic=False)


class Inputs:
    masked = False

    def __init__(self, tokens, args):
        self.qh, self.kh, self.hd, self.rd, self.zero = VARIANTS[args.variant]
        self.gated = self.zero and not args.no_output_gate
        self.kv_dtype = args.kv_dtype
        self.masked = args.masked
        self.fp8 = args.kv_dtype != "bf16"
        self.cache_dtype = getattr(torch, KV_DTYPES[args.kv_dtype])
        self.cache_kind = {"bf16": "auto", "fp8_e4m3": "fp8", "fp8_e5m2": "fp8_e5m2"}[args.kv_dtype]
        self.tokens = tokens
        self.qsize, self.ksize = self.qh * self.hd, self.kh * self.hd
        self.width = self.qsize * (2 if self.gated else 1) + 2 * self.ksize
        device, dtype = f"cuda:{args.device}", torch.bfloat16
        gen = torch.Generator(device=device).manual_seed(args.seed + tokens)

        def random(*shape):
            return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

        self.positions = torch.arange(tokens, device=device, dtype=torch.int64) + 1024
        frequency = 1e6 ** (-torch.arange(0, self.rd, 2, dtype=torch.float64) / self.rd)
        angles = torch.arange(tokens + 1024, dtype=torch.float64)[:, None] * frequency
        self.table = torch.cat((angles.cos(), angles.sin()), -1).to(device, torch.float32)
        self.slots = max(256, math.ceil((tokens + 1) / 16) * 16)
        self.indices64 = torch.randperm(self.slots - 1, generator=gen, device=device)[:tokens] + 1
        self.indices = self.indices64.to(torch.int32)
        self.scale = torch.ones((), device=device)
        self.k_scale, self.v_scale = (
            [torch.full((1,), value, device=device, dtype=torch.float32) for value in KV_SCALES]
            if self.fp8
            else (self.scale, self.scale)
        )
        self.vllm_table = self.table.to(torch.bfloat16)
        hidden, intermediate = MODEL_SIZES[args.variant]
        self.raw = torch.empty(tokens, self.width, device=device, dtype=dtype)
        x = random(tokens, hidden)
        self.filler_kind = args.filler
        if args.filler == "model":
            self.filler_input = random(tokens, self.qsize)
            self.filler_hidden = torch.empty(tokens, hidden, device=device, dtype=dtype)
            self.filler_gate_up = torch.empty(tokens, 2 * intermediate, device=device, dtype=dtype)
            self.filler_down_input = self.filler_gate_up[:, :intermediate]
        else:
            self.filler_output = torch.empty(tokens, args.filler_width, device=device, dtype=dtype)
            self.filler_weight = random(hidden, args.filler_width)
        self.layers = []
        for _ in range(args.layers):
            self.layers.append(
                types.SimpleNamespace(
                    x=x,
                    w=random(self.width, hidden) / 64,
                    raw=self.raw,
                    qw=random(self.hd) * 0.2 + (0 if self.zero else 1),
                    kw=random(self.hd) * 0.2 + (0 if self.zero else 1),
                )
            )
            if args.filler == "model":
                for name, shape in filler_shapes(args.variant).items():
                    setattr(self.layers[-1], name, random(*shape))
        if self.gated:
            for layer in self.layers:
                layer.w_engine = self.engine_layout(layer.w)
                layer.raw_engine = self.raw
        self.eviction_bytes = 0
        if args.filler == "legacy":
            l2 = torch.cuda.get_device_properties(args.device).L2_cache_size
            if l2 <= 0:
                raise RuntimeError("Device did not report L2 size")
            self.eviction_bytes = math.ceil(max(2 * l2, args.evict_mib * 1024**2) / 2) * 2
            self.eviction = torch.zeros(self.eviction_bytes // 2, device=device, dtype=dtype)

    def split(self, raw):
        q, k, v = raw.split((self.qsize * (2 if self.gated else 1), self.ksize, self.ksize), -1)
        if self.gated:
            q = q.view(self.tokens, self.qh, 2 * self.hd)[..., : self.hd]
        return (
            q.reshape(self.tokens, self.qh, self.hd),
            k.view(-1, self.kh, self.hd),
            v.view(-1, self.kh, self.hd),
        )

    def engine_layout(self, rows):
        """Market [q0 g0 q1 g1 ... | K | V] rows -> the engine's [Q | K | V | gate]."""
        qg = rows[: 2 * self.qsize].unflatten(0, (self.qh, 2, self.hd))
        rest = rows[2 * self.qsize :]
        q, g = qg[:, 0].flatten(0, 1), qg[:, 1].flatten(0, 1)
        return torch.cat((q, rest, g), 0).contiguous()

    def project(self, layer, engine=False):
        if engine:
            torch.matmul(layer.x, layer.w_engine.T, out=layer.raw_engine)
        else:
            torch.matmul(layer.x, layer.w.T, out=layer.raw)

    def filler(self, layer):
        if self.filler_kind == "model":
            torch.matmul(self.filler_input, layer.o_proj, out=self.filler_hidden)
            torch.matmul(self.filler_hidden, layer.gate_up, out=self.filler_gate_up)
            # Only GEMM weight traffic is modeled: omit the activation kernel.
            torch.matmul(self.filler_down_input, layer.down, out=self.filler_hidden)
        else:
            torch.matmul(layer.x, self.filler_weight, out=self.filler_output)
            self.eviction.add_(1)

    def cache(self):
        shape = (self.slots, self.kh, self.hd)
        return [
            torch.zeros(shape, device=self.table.device, dtype=self.cache_dtype) for _ in range(2)
        ]


def reference(b, layer):
    q, k, v = b.split(layer.raw)
    dtype = torch.float64 if b.fp8 else torch.float32
    c, s = b.table[b.positions].cpu().to(dtype).unsqueeze(1).chunk(2, -1)
    outputs, bounds = [], []
    for heads, weight in ((q, layer.qw), (k, layer.kw)):
        x = heads.cpu().to(dtype)
        x = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        x = x * (weight.cpu().to(dtype) + (1 if b.zero else 0))
        a, d = x[..., : b.rd].chunk(2, -1)
        outputs.append(torch.cat((a * c - d * s, d * c + a * s, x[..., b.rd :]), -1))
        bounds.append(
            torch.cat(
                (
                    (a * c).abs() + (d * s).abs(),
                    (d * c).abs() + (a * s).abs(),
                    x[..., b.rd :].abs(),
                ),
                -1,
            )
        )
    return (*outputs, v.cpu().float()), (*bounds, v.cpu().float().abs())


def fp8_ulp(reference, dtype):
    """Spacing at the clipped magnitude, with a constant subnormal spacing."""
    info = torch.finfo(dtype)
    magnitude = reference.double().abs().clamp(min=info.tiny, max=info.max)
    return torch.exp2(torch.floor(torch.log2(magnitude))) * info.eps


def fp8_error(got, reference, scale, dtype, *, limit, bitwise=False, bound=None):
    """CPU-testable FP8 gate; byte equality includes the sign of zero. ``bound`` (the
    unscaled magnitude bound of a BF16-rounded K, as for Q) adds 4 of its BF16 ulps to the
    allowed error: stacks that round K to BF16 before quantizing lose relative accuracy
    near RoPE cancellation."""
    from benchmark.kernel_tuning.attention_prologue_paired import bf16_ulp

    info = torch.finfo(dtype)
    scaled = (reference.double() / scale.double()).clamp(-info.max, info.max)
    diff = (got.double() - scaled).abs()
    ulps = diff / fp8_ulp(scaled, dtype)
    allowed = limit * fp8_ulp(scaled, dtype)
    if bound is not None:
        allowed = allowed + 4 * bf16_ulp(bound.double()) / scale.double()
    finite = bool(torch.isfinite(ulps).all())
    expected = (reference.float() / scale.float()).clamp(-info.max, info.max).to(dtype)
    exact = torch.equal(got.view(torch.uint8), expected.view(torch.uint8))
    return {
        "passed": finite and (exact if bitwise else bool((diff <= allowed).all())),
        "max_abs": diff.max().item() if finite else None,
        "max_fp8_ulp": ulps.max().item() if finite else None,
        "ulp_limit": limit,
        "bitwise": exact,
        "requires_bitwise": bitwise,
    }


def correctness(b, operations, caches, stack="ours"):
    from benchmark.kernel_tuning.attention_prologue_paired import bf16_ulp

    errors, passed = [], True
    for layer, op, (kc, vc) in zip(b.layers, operations, caches, strict=True):
        b.project(layer)
        expected, bounds = reference(b, layer)
        if getattr(op, "engine_layout", False):
            # Bitwise the same values as the market projection, engine column order.
            layer.raw_engine.copy_(b.engine_layout(layer.raw.T).T)
        kc.view(torch.uint8).zero_()
        vc.view(torch.uint8).zero_()
        q = op()
        torch.cuda.synchronize()
        written = [
            cache.view(torch.uint8)[b.indices64].cpu().view(b.cache_dtype) for cache in (kc, vc)
        ]
        actual = (q.cpu().float(), *written)
        record = {}
        for name, output, ref, bound in zip(("q", "k", "v"), actual, expected, bounds, strict=True):
            got = output.reshape_as(ref)
            if b.fp8 and name != "q":
                scale = b.k_scale if name == "k" else b.v_scale
                scale = torch.ones(1) if stack == "sglang_noscale" else scale.cpu()
                limit = 1 if stack in ("ours", "self") or name == "v" else 2
                record[name] = fp8_error(
                    got,
                    ref,
                    scale,
                    b.cache_dtype,
                    limit=limit,
                    bitwise=name == "v" and stack != "sglang",
                    bound=None if stack in ("ours", "self") or name == "v" else bound,
                )
                passed &= record[name]["passed"]
                continue
            diff = (got.double() - ref.double()).abs()
            finite = bool(torch.isfinite(diff).all())
            ulp = diff / bf16_ulp(bound.double())
            okay = finite and bool((diff == 0).all() if name == "v" else (ulp <= 4).all())
            record[name] = {
                "passed": okay,
                "max_abs": diff.max().item() if finite else None,
                "max_ulp": (diff / bf16_ulp(ref.double())).max().item() if finite else None,
                "max_conditioned_ulp": ulp.max().item() if finite else None,
            }
            passed &= okay
        mask = torch.ones(b.slots, device=kc.device, dtype=torch.bool)
        mask[b.indices64] = False
        untouched = bool(
            (kc.view(torch.uint8)[mask] == 0).all() and (vc.view(torch.uint8)[mask] == 0).all()
        )
        record["untouched_cache_rows"] = untouched
        passed &= untouched
        errors.append(record)
    return passed, errors


def native_region(b, layer, modules):
    norm, rotary = modules
    cls = norm.GemmaRMSNorm if b.zero else norm.RMSNorm
    qnorm, knorm = cls(b.hd, eps=1e-6), cls(b.hd, eps=1e-6)
    qnorm.weight = torch.nn.Parameter(layer.qw, requires_grad=False)
    knorm.weight = torch.nn.Parameter(layer.kw, requires_grad=False)
    rope = rotary.RotaryEmbedding(b.hd, b.rd, b.table.shape[0], 1e6, True, torch.bfloat16)
    rope.cos_sin_cache = b.vllm_table

    def run(raw):
        q, k, _ = b.split(raw)
        q = qnorm.forward_native(q).reshape(b.tokens, b.qsize)
        k = knorm.forward_native(k).reshape(b.tokens, b.ksize)
        return rope.forward_native(b.positions, q, k)

    return fresh_compile(run)


def surrogate_region(b, layer):
    from benchmark.kernel_tuning._prologue_market import native_vllm

    def run(raw):
        if not b.zero:
            return native_vllm(raw, layer.qw, layer.kw, b.vllm_table, b.positions)
        # Explicit hd256 generalization of the existing independent surrogate.
        q, k, _ = b.split(raw)
        coeff = b.vllm_table[b.positions].unsqueeze(1)
        c, s = coeff.chunk(2, -1)
        outputs = []
        for x, weight in ((q, layer.qw), (k, layer.kw)):
            fp = x.float()
            n = (
                fp * (fp.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weight.float())
            ).to(x.dtype)
            a, d = n[..., : b.rd].chunk(2, -1)
            outputs.append(torch.cat((a * c - d * s, d * c + a * s, n[..., b.rd :]), -1))
        return outputs[0], outputs[1]

    return fresh_compile(run)


def operations(name, b, caches, providers):
    from ctikernels.attention.inference import Prologue
    from ctikernels.norm.inference import rms_norm_
    from ctikernels.rotary.inference import apply_rotary_
    from ctikernels.store import store_cache

    ours = None
    if name == "ours":
        ours = Prologue(
            num_q_heads=b.qh,
            num_kv_heads=b.kh,
            eps=1e-6,
            device=b.table.device,
            head_dim=b.hd,
            rotary_dim=b.rd,
            zero_centered=b.zero,
            cache_dtype=b.cache_dtype,
            masked=b.masked,
        )
    result = []
    for layer, (kc, vc) in zip(b.layers, caches, strict=True):
        native = None
        if name == "vllm_surrogate":
            native = surrogate_region(b, layer)
        elif name == "vllm" and not b.gated:
            native = native_region(b, layer, providers["vllm"])

        def run(layer=layer, kc=kc, vc=vc, native=native):
            raw = layer.raw
            q, k, v = b.split(raw)
            if name in ("ours", "compose"):
                # Gated: a strided [tokens, Q|K|V] view of the engine projection.
                packed = layer.raw_engine[:, : b.qsize + 2 * b.ksize] if b.gated else raw
                if ours is not None:
                    scales = {"k_scale": b.k_scale, "v_scale": b.v_scale} if b.fp8 else {}
                    return ours(
                        packed,
                        layer.qw,
                        layer.kw,
                        b.table,
                        b.positions,
                        kc,
                        vc,
                        b.indices,
                        **scales,
                    )
                q, k, v = packed.unflatten(1, (b.qh + 2 * b.kh, b.hd)).split((b.qh, b.kh, b.kh), 1)
                rms_norm_(q, layer.qw, 1e-6, zero_centered=b.zero)
                rms_norm_(k, layer.kw, 1e-6, zero_centered=b.zero)
                apply_rotary_(q.flatten(1), k.flatten(1), b.table, b.positions, head_dim=b.hd)
                if b.fp8:
                    from ctikernels.attention._prologue import _quantize

                    k = _quantize(k, b.k_scale, b.cache_dtype)
                    v = _quantize(v, b.v_scale, b.cache_dtype)
                    kc, vc = kc.view(torch.uint8), vc.view(torch.uint8)
                store_cache(kc, vc, b.indices, k.flatten(1), v.flatten(1))
            elif name in ("sglang", "sglang_noscale"):
                norm, rope, cache, *extra = providers["sglang"]
                if b.gated:
                    q_gate = raw[:, : 2 * b.qsize]
                    q, k, gate = extra[0](
                        q_gate,
                        k.flatten(1),
                        layer.qw,
                        layer.kw,
                        b.table,
                        b.positions,
                        1e-6,
                        b.qh,
                        b.kh,
                        b.hd,
                        b.rd,
                        has_gate=True,
                    )
                elif b.zero:
                    q = extra[0](q.reshape(-1, b.hd), layer.qw, 1e-6).view(b.tokens, b.qh, b.hd)
                    k = extra[0](k.reshape(-1, b.hd), layer.kw, 1e-6).view(b.tokens, b.kh, b.hd)
                else:
                    norm.fused_inplace_qknorm(q, k, layer.qw, layer.kw, 1e-6, head_dim=b.hd)
                if not b.gated:
                    rope.apply_rope_with_cos_sin_cache_inplace(
                        q, k, b.table, b.positions, is_neox=True, fused_args=None
                    )
                if b.fp8:
                    if name == "sglang":
                        k.div_(KV_SCALES[0])
                        v.div_(KV_SCALES[1])
                    k = k.to(b.cache_dtype).view(torch.uint8)
                    v = v.to(b.cache_dtype).view(torch.uint8)
                    kc, vc = kc.view(torch.uint8), vc.view(torch.uint8)
                cache.store_cache(
                    k.reshape(b.tokens, -1),
                    v.flatten(1),
                    kc.flatten(1),
                    vc.flatten(1),
                    b.indices,
                    row_bytes=b.ksize * (1 if b.fp8 else 2),
                    v_row_bytes=b.ksize * (1 if b.fp8 else 2),
                    size_limit=b.slots,
                )
            else:
                if name == "vllm" and b.gated:
                    q, k, gate = providers["vllm_fused"].fused_qk_rmsnorm_rope_gate(
                        raw[:, : 2 * b.qsize],
                        k.flatten(1),
                        layer.qw,
                        layer.kw,
                        providers["vllm_table"],
                        b.positions,
                        1e-6,
                        b.qh,
                        b.kh,
                        b.hd,
                        b.rd,
                        norm_beta=1.0,
                    )
                else:
                    q, k = native(raw)
                if b.fp8:
                    kc, vc = kc.view(torch.uint8), vc.view(torch.uint8)
                providers["cache_op"](
                    k.reshape(b.tokens, b.kh, b.hd),
                    v,
                    kc.view(-1, 16, b.kh, b.hd),
                    vc.view(-1, 16, b.kh, b.hd),
                    b.indices64,
                    b.cache_kind,
                    b.k_scale,
                    b.v_scale,
                )
            return q

        run.engine_layout = b.gated and name in ("ours", "compose")
        result.append(run)
    return result, ours.provider if ours is not None else name


class InvalidSample(ValueError):
    """A replay cannot be sliced safely; messages contain only local diagnostics."""


@dataclass(frozen=True)
class KernelEvent:
    name: str
    start: float
    end: float


def ordered_kernels(events):
    events = sorted(events, key=lambda event: event.start)
    if not events:
        raise InvalidSample("no GPU kernels recorded")
    if any(
        not math.isfinite(event.start) or not math.isfinite(event.end) or event.end <= event.start
        for event in events
    ):
        raise InvalidSample("invalid GPU kernel timestamps")
    # PDL lets a dependent kernel start before its primary ends; start order stays serial.
    return events


def kernels_per_layer(events, layers):
    if layers < 1 or not events or len(events) % layers:
        raise InvalidSample(f"kernel count {len(events)} is not a positive multiple of {layers}")
    return len(events) // layers


@dataclass(frozen=True)
class SpanLayout:
    projection: tuple[str, ...]
    filler: tuple[str, ...]

    @classmethod
    def calibrate(cls, projection_events, baseline_events, layers):
        projection = ordered_kernels(projection_events)
        baseline = ordered_kernels(baseline_events)
        p = kernels_per_layer(projection, layers)
        total = kernels_per_layer(baseline, layers)
        if total <= p:
            raise InvalidSample("baseline must contain filler kernels")
        names = tuple(event.name for event in projection[:p])
        for offset in range(0, len(projection), p):
            if Counter(event.name for event in projection[offset : offset + p]) != Counter(names):
                raise InvalidSample("projection name multiset differs between layers")
        layout = cls(names, tuple(event.name for event in baseline[p:total]))
        layout.spans(baseline, layers, 0)
        return layout

    def prologue_count(self, events, layers):
        count = kernels_per_layer(events, layers) - len(self.projection) - len(self.filler)
        if count < 1:
            raise InvalidSample("stack must contain prologue kernels")
        self.spans(events, layers, count)
        return count

    def spans(self, events, layers, prologue_count):
        p, f = len(self.projection), len(self.filler)
        if layers < 1 or p < 1 or f < 1 or prologue_count < 0:
            raise InvalidSample("invalid layer or kernel counts")
        stride = p + f + prologue_count
        expected = layers * stride
        if len(events) != expected:
            raise InvalidSample(f"kernel count mismatch: expected {expected}, got {len(events)}")
        events = ordered_kernels(events)
        spans = []
        for offset in range(0, expected, stride):
            layer = events[offset : offset + stride]
            # Slice by count first: prologue kernels may share GEMM/add names.
            if Counter(event.name for event in layer[:p]) != Counter(self.projection):
                raise InvalidSample("projection name multiset mismatch")
            if Counter(event.name for event in layer[-f:]) != Counter(self.filler):
                raise InvalidSample("filler name multiset mismatch")
            projection_end = max(event.end for event in layer[:p])
            prologue_end = max((event.end for event in layer[p : p + prologue_count]), default=0)
            spans.append(max(prologue_end, projection_end) - projection_end)
        return spans


def kernel_counts(events, layers):
    total = kernels_per_layer(events, layers)
    return {
        "total_per_layer": total,
        "names_per_layer": {
            key: value / layers for key, value in Counter(event.name for event in events).items()
        },
        "method": "CUDA-only profiler of captured replay",
    }


class Graph:
    def __init__(self, b, ops, *, filler=True):
        self.layers = len(b.layers)

        def run():
            for index, layer in enumerate(b.layers):
                b.project(layer, engine=bool(ops) and getattr(ops[index], "engine_layout", False))
                if ops:
                    # Discard before filler/next projection, including in capture:
                    # the private graph pool can reuse out-of-place Q/K storage.
                    ops[index]()
                if filler:
                    b.filler(layer)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            run()
        self.start, self.end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for _ in range(3):
            self.sample()

    def sample(self):
        self.start.record()
        self.graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000 / self.layers

    def profile(self):
        from torch.profiler import ProfilerActivity, profile

        # Drain earlier work before opening the session. Exactly one replay is
        # enclosed, so events from other stacks cannot enter this sample.
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=False
        ) as prof:
            graph_us = self.sample()
            torch.cuda.synchronize()
        events = [
            KernelEvent(event.name, event.time_range.start, event.time_range.end)
            for event in prof.events()
            if event.device_type.name == "CUDA"
            and not event.name.lower().startswith(("memcpy", "memset", "[memory"))
        ]
        return events, graph_us

    def span_sample(self):
        events, graph_us = self.profile()
        spans = self.span_layout.spans(events, self.layers, self.prologue_kernels)
        return statistics.mean(spans), graph_us


def collect(graphs, baseline, count, *, self_graph, timing="subtract"):
    from benchmark.kernel_tuning._prologue_market import safe_failure

    samples = []
    names = list(graphs)
    for index in range(count):
        # Rotate and reverse stack order; alternate the position of the paired baseline.
        order = names[index % len(names) :] + names[: index % len(names)]
        if index % 2:
            order.reverse()
        if "ours" in graphs:
            order.append("self")
        order.insert(len(order) if index % 2 else 0, "baseline")
        replay_graphs = {**graphs, "baseline": baseline, "self": self_graph}
        record = {}
        if timing == "span":
            record.update(graph_us_per_layer={}, invalid={})
        for name in order:
            graph = replay_graphs[name]
            if timing == "subtract":
                record[name] = graph.sample()
                continue
            try:
                record[name], record["graph_us_per_layer"][name] = graph.span_sample()
            except InvalidSample as error:
                record[name] = None
                record["invalid"][name] = {"reason": str(error)}
            except Exception as error:
                record[name] = None
                record["invalid"][name] = safe_failure(error, "replay profiler")
        samples.append(record)
    return samples


def summarize(samples, names, *, timing="subtract", kv_dtype="bf16"):
    valid = [sample for sample in samples if not sample.get("invalid")]
    adjusted = {
        name: [s[name] - s["baseline"] if timing == "subtract" else s[name] for s in valid]
        for name in names
    }
    result = {
        "timing": timing,
        "us_per_layer": {
            name: statistics.median(values) for name, values in adjusted.items() if values
        },
        "pairs": len(valid),
        "attempted_pairs": len(samples),
        "invalid_pairs": len(samples) - len(valid),
        "samples": samples,
        "samples_graph_us_per_layer": (
            samples if timing == "subtract" else [s["graph_us_per_layer"] for s in samples]
        ),
        "samples_prologue_us_per_layer": adjusted,
        "verdict": "noise",
        "speedup": None,
        "ci95": None,
        "epsilon": None,
        "market": None,
    }
    if timing == "span":
        result["samples_subtraction_us_per_layer"] = [
            {
                name: value - s["graph_us_per_layer"]["baseline"]
                for name, value in s["graph_us_per_layer"].items()
                if name != "baseline"
            }
            if "baseline" in s["graph_us_per_layer"]
            else {}
            for s in samples
        ]
    if len(valid) < 6:
        result["reason"] = "fewer than six valid pairs; no finite exact 95% median CI"
        return result
    required = ("sglang", "vllm") if kv_dtype == "bf16" else ("sglang", "sglang_noscale", "vllm")
    market = [name for name in required if name in names]
    # Surrogate measurements are visible but cannot stand in for the real market bar.
    if "ours" not in names or len(market) != len(required):
        result["reason"] = "all real market stacks and ours must pass correctness"
        return result
    best = min(market, key=lambda name: result["us_per_layer"][name])
    result["market"] = best
    self_times = [s["self"] - s["baseline"] if timing == "subtract" else s["self"] for s in valid]
    if any(t <= 0 for name in ("ours", *market) for t in adjusted[name]) or any(
        t <= 0 for t in self_times
    ):
        result["reason"] = (
            "nonpositive baseline-subtracted samples; subtraction unresolved"
            if timing == "subtract"
            else "nonpositive prologue spans"
        )
        return result
    epsilon = percentile95(
        [abs(y / x - 1) for x, y in zip(adjusted["ours"], self_times, strict=True)]
    )
    ratios = [y / x for x, y in zip(adjusted["ours"], adjusted[best], strict=True)]
    ci = median_ci(ratios)
    result.update(
        epsilon=epsilon,
        speedup=statistics.median(ratios),
        ci95=ci,
        verdict="faster" if ci[0] > 1 + epsilon else "slower" if ci[1] < 1 - epsilon else "noise",
    )
    return result


def source_preflight(cache, *, gated=False):
    from benchmark.kernel_tuning._prologue_market import vllm_fused_module, vllm_source_modules

    wheels = sorted(cache.glob("vllm-0.31.0-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError("Place one vLLM 0.31.0 wheel in --vllm-cache")
    modules, info = vllm_source_modules(wheels[0])
    # Exercise both real native norm classes and partial/full RoPE on CPU only.
    for hd, rd, cls in ((128, 128, modules[0].RMSNorm), (256, 64, modules[0].GemmaRMSNorm)):
        norm = cls(hd, eps=1e-6).to(dtype=torch.bfloat16)
        norm.requires_grad_(False)
        rotary = modules[1].RotaryEmbedding(hd, rd, 16, 1e6, True, torch.bfloat16)
        x = torch.ones(2, 1, hd, dtype=torch.bfloat16)
        q = norm.forward_native(x).flatten(1)
        out, _ = rotary.forward_native(torch.tensor([1, 2]), q, q)
        if not bool(torch.isfinite(out).all()) or out.shape != (2, hd):
            raise RuntimeError("Real native source CPU preflight failed")
    if gated:
        fused = vllm_fused_module()
        if not callable(fused.fused_qk_rmsnorm_rope_gate):
            raise RuntimeError("Missing real gated entrypoint")
        info["gated_kernel"] = "real fused_qk_rmsnorm_rope_gate imported; not launched"
    info.update(status="real modules imported; CPU forwards passed", cuda_launched=False)
    return info


def load_providers(args):
    from benchmark.kernel_tuning._prologue_market import (
        load_vllm_cache,
        quiet_external,
        safe_failure,
        sglang_engine_modules,
        sglang_modules,
        vllm_fused_module,
        vllm_source_modules,
    )

    providers, audit = {}, {}
    with quiet_external():
        try:
            if VARIANTS[args.variant][4]:
                modules, info = sglang_engine_modules(
                    args.sglang_python, gated=not args.no_output_gate
                )
            else:
                modules, info = sglang_modules(args.sglang_python)
            if not info["revision_verified"]:
                raise RuntimeError("SGLang revision mismatch")
            providers["sglang"], audit["sglang"] = modules, info
            if args.kv_dtype != "bf16":
                providers["sglang_noscale"] = modules
                audit["sglang_noscale"] = dict(info, k_scale=None, v_scale=None)
        except Exception as error:
            audit["sglang"] = safe_failure(error, "sglang source import/revision verification")
        cache_op, info = load_vllm_cache(args.vllm_cache)
        audit["vllm"] = info
        if cache_op is not None:
            providers["cache_op"] = cache_op
            try:
                modules, source = vllm_source_modules(args.vllm_cache / info["wheel"])
                if VARIANTS[args.variant][4] and not args.no_output_gate:
                    providers["vllm_fused"] = vllm_fused_module()
                    source["norm_rope"] = (
                        "real wheel fused_qk_rmsnorm_rope_gate (default gated model dispatch)"
                    )
                    source["off_path_stubs"].append("vllm.triton_utils (real triton/tl re-exports)")
                    import hashlib
                    import zipfile

                    member = "vllm/model_executor/layers/fused_qk_norm_rope.py"
                    with zipfile.ZipFile(args.vllm_cache / info["wheel"]) as archive:
                        source["source_sha256"][member] = hashlib.sha256(
                            archive.read(member)
                        ).hexdigest()
                providers["vllm"] = modules
                info.update(source)
            except Exception as error:
                info["source_import_failure"] = safe_failure(
                    error, "vllm real Python source import"
                )
                info["norm_rope"] = "vllm_surrogate: real Python modules unavailable"
                providers["vllm_surrogate"] = True
    return providers, audit


def run_row(tokens, args, providers):
    from benchmark.kernel_tuning._prologue_market import quiet_external, safe_failure

    record = {"tokens": tokens, "timing": args.timing, "stacks": {}}
    graphs = {}
    with quiet_external():
        b = Inputs(tokens, args)
        providers["vllm_table"] = b.vllm_table
        record["projection_width"] = b.width
        record["eviction_bytes"] = b.eviction_bytes
        record["protocol"] = protocol_metadata(args, tokens=tokens, eviction_bytes=b.eviction_bytes)
        baseline = Graph(b, [])
        layout = None
        try:
            baseline_events, _ = baseline.profile()
            record["baseline_kernels"] = kernel_counts(baseline_events, args.layers)
            if args.timing == "span":
                projection = Graph(b, [], filler=False)
                projection_events, _ = projection.profile()
                record["projection_kernels"] = kernel_counts(projection_events, args.layers)
                layout = SpanLayout.calibrate(projection_events, baseline_events, args.layers)
                record["span_layout"] = {
                    "projection_per_layer": len(layout.projection),
                    "filler_per_layer": len(layout.filler),
                }
                baseline.span_layout, baseline.prologue_kernels = layout, 0
        except Exception as error:
            record["timing_failure"] = (
                {"reason": str(error)}
                if isinstance(error, InvalidSample)
                else safe_failure(error, "projection/baseline graph profiler")
            )
        names = ["ours", "compose", "sglang", "vllm", "vllm_surrogate", "self"]
        if args.kv_dtype != "bf16":
            names.insert(3, "sglang_noscale")
        for name in names:
            if name == "self" and "ours" not in graphs:
                continue
            if name not in ("ours", "compose", "self") and name not in providers:
                record["stacks"][name] = {"status": "unavailable"}
                continue
            entry = record["stacks"][name] = {}
            phase = "construct"
            try:
                caches = [b.cache() for _ in b.layers]
                ops, provider = operations("ours" if name == "self" else name, b, caches, providers)
                entry["provider"] = provider
                phase = "correctness"
                okay, errors = correctness(b, ops, caches, name)
                entry["correctness"] = errors
                if args.kv_dtype != "bf16":
                    entry["max_fp8_ulp"] = {
                        kind: max(layer[kind]["max_fp8_ulp"] for layer in errors)
                        if errors
                        and all(layer[kind]["max_fp8_ulp"] is not None for layer in errors)
                        else None
                        for kind in ("k", "v")
                    }
                if not okay:
                    entry["status"] = "excluded: correctness failed"
                    continue
                phase = "capture"
                graph = Graph(b, ops)
                entry["status"] = "ok"
                try:
                    events, _ = graph.profile()
                    counts = kernel_counts(events, args.layers)
                    base_count = record.get("baseline_kernels", {}).get("total_per_layer")
                    counts["prologue_per_layer"] = (
                        counts["total_per_layer"] - base_count if base_count is not None else None
                    )
                    entry["kernels"] = counts
                    if args.timing == "span":
                        if layout is None:
                            raise InvalidSample("projection/baseline calibration unavailable")
                        graph.prologue_kernels = layout.prologue_count(events, args.layers)
                        graph.span_layout = layout
                except Exception as error:
                    entry["timing_failure"] = (
                        {"reason": str(error)}
                        if isinstance(error, InvalidSample)
                        else safe_failure(error, "stack graph profiler")
                    )
                    if args.timing == "span":
                        entry["status"] = "excluded: invalid span calibration"
                        continue
                # Captured graphs keep tensor addresses alive; retain explicit owners too.
                graph.owners = (ops, caches)
                graphs[name] = graph
            except Exception as error:
                entry.update(
                    status="excluded: execution failed", failure=safe_failure(error, phase)
                )
        self_graph = graphs.pop("self", None)
        if "ours" in graphs and self_graph is None:
            record["comparison"] = {
                "verdict": "noise",
                "reason": "independent self graph unavailable",
            }
        elif graphs:

            def measure(count):
                return summarize(
                    collect(graphs, baseline, count, self_graph=self_graph, timing=args.timing),
                    list(graphs),
                    timing=args.timing,
                    kv_dtype=args.kv_dtype,
                )

            initial = measure(args.pairs)
            result = initial
            required = {"ours", "sglang", "vllm"}
            if args.kv_dtype != "bf16":
                required.add("sglang_noscale")
            if initial["verdict"] == "noise" and required <= graphs.keys():
                result = measure(max(40, 2 * args.pairs))
                result["initial"] = initial
            record["comparison"] = result
        device_name = torch.cuda.get_device_name(args.device)
        bandwidth = next((value for key, value in CEILINGS.items() if key in device_name), None)
        # Ideal fused QKV read + Q and KV writes, plus weights/table/indices.
        moved = prologue_bytes(tokens, b.qsize, b.ksize, b.hd, b.rd, args.kv_dtype)
        record["sol"] = {
            "bytes_per_layer": moved,
            "dram_GB_s": bandwidth,
            "us_per_layer": moved / (bandwidth * 1000) if bandwidth else None,
            "scope": (
                "DRAM-bound reference only (projection output is L2-hot); "
                "ideal ungated fused prologue; excludes gate packing and filler"
            ),
        }
        if record["protocol"]["diagnostic"]:
            diagnostic_verdicts(record)
    return record


def prologue_bytes(tokens, qsize, ksize, head_dim, rotary_dim, kv_dtype):
    cache_bytes = 2 if kv_dtype == "bf16" else 1
    return (
        tokens * (4 * qsize + (4 + 2 * cache_bytes) * ksize)
        + 4 * head_dim
        + tokens * (4 * rotary_dim + 12)
        + (8 if kv_dtype != "bf16" else 0)
    )


def print_row(row):
    comp = row.get("comparison", {})
    times = comp.get("us_per_layer", {})
    values = []
    for name in ("ours", "compose", "sglang", "sglang_noscale", "vllm", "vllm_surrogate"):
        if (
            name in ("sglang_noscale", "vllm_surrogate")
            and name not in row.get("stacks", {})
            and name not in times
        ):
            continue
        entry = row.get("stacks", {}).get(name, {})
        provider = entry.get("provider", "NA")
        count = entry.get("kernels", {}).get("prologue_per_layer")
        value = f"{times[name]:.3f}" if name in times else "NA"
        status = f"[{entry.get('status', 'unavailable')}]" if name not in times else ""
        values.append(
            f"{name}({provider})={value}us/layer {count if count is not None else 'NA'}k/layer{status}"
        )
    ratio, ci, eps = comp.get("speedup"), comp.get("ci95"), comp.get("epsilon")
    speedup = f"{ratio:.3f}x CI95=[{ci[0]:.3f},{ci[1]:.3f}]" if ratio is not None else "NA CI95=NA"
    sol = row.get("sol", {}).get("us_per_layer")
    print(
        f"T={row['tokens']} timing={row.get('timing', 'NA')} "
        + " ".join(values)
        + f" speedup={speedup} market={comp.get('market')}"
        + f" epsilon={f'{eps:.4f}' if eps is not None else 'NA'}"
        + f" verdict={comp.get('verdict', 'noise')}"
        + f" invalid_pairs={comp.get('invalid_pairs', 0)}/{comp.get('attempted_pairs', 0)}"
        + f" SOL={f'{sol:.3f}' if sol is not None else 'NA'}us/layer",
        flush=True,
    )
    # Include failed calibration and individual invalid replay diagnostics even
    # without --json. External failures contain safe locations only.
    if "reason" in comp:
        print(f"  {comp['reason']}")
    if "timing_failure" in row:
        print(f"  timing calibration: {json.dumps(row['timing_failure'], sort_keys=True)}")
    for name, entry in row.get("stacks", {}).items():
        if "timing_failure" in entry:
            print(
                f"  {name} timing calibration: {json.dumps(entry['timing_failure'], sort_keys=True)}"
            )
    for phase, comparison in (("initial", comp.get("initial", {})), ("final", comp)):
        for index, sample in enumerate(comparison.get("samples", [])):
            if sample.get("invalid"):
                print(
                    f"  {phase} pair={index} invalid={json.dumps(sample['invalid'], sort_keys=True)}"
                )


def main(argv=None):
    global torch
    args = parser().parse_args(argv)
    rows = [int(value) for value in args.rows.split(",")]
    if (
        min(rows, default=0) < 1
        or args.layers < 1
        or args.pairs < 6
        or args.filler_width < 1
        or args.evict_mib < 0
    ):
        raise ValueError(
            "Positive rows/layers/filler, nonnegative eviction and at least six pairs required"
        )
    torch = importlib.import_module("torch")
    from benchmark.kernel_tuning._prologue_market import quiet_external, safe_failure

    report = {
        "variant": args.variant,
        "layers": args.layers,
        "pairs": args.pairs,
        "timing": args.timing,
        "output_gate": VARIANTS[args.variant][4] and not args.no_output_gate,
        "masked": args.masked,
        "filler_width": args.filler_width,
        "seed": args.seed,
        "rows": [],
        "protocol": protocol_metadata(args),
        "dispatch_audit": __doc__,
    }
    # Row protocols contain token-dependent logical filler bytes (and the
    # device-dependent eviction size for legacy diagnostics).
    print_protocol(report["protocol"])
    if args.check_vllm_source:
        try:
            with quiet_external(), torch.no_grad():
                report["vllm"] = source_preflight(args.vllm_cache, gated=report["output_gate"])
        except Exception as error:
            report["vllm"] = safe_failure(error, "CPU vLLM source preflight")
        print(json.dumps(report["vllm"], sort_keys=True))
    else:
        torch.cuda.set_device(args.device)
        report["device"] = torch.cuda.get_device_name(args.device)
        providers, report["market_audit"] = load_providers(args)
        with torch.no_grad():
            for tokens in rows:
                try:
                    row = run_row(tokens, args, providers)
                except Exception as error:
                    row = {"tokens": tokens, "failure": safe_failure(error, "row setup")}
                report["rows"].append(row)
                if "protocol" in row:
                    report["protocol"]["filler_bytes_per_layer"][str(tokens)] = row["protocol"][
                        "filler_bytes_per_layer"
                    ]
                    report["protocol"]["eviction_bytes"] = row["protocol"]["eviction_bytes"]
                print_row(row)
                if args.json:
                    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                gc.collect()
                torch.cuda.empty_cache()
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    try:
        result = main()
        if "exception" in result.get("vllm", {}):
            raise SystemExit(1)
    except Exception as error:
        # Never print external exception text, tracebacks, environment or credentials.
        print(
            f"Engine benchmark failed ({type(error).__name__}); no raw diagnostics emitted.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
