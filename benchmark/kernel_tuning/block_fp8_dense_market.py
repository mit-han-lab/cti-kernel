# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Same-card block-FP8 dense GEMM rows (Qwen3.5-35B-A3B-FP8 shapes; DeepSeek V4 Flash ones by
``--shapes dsv4-...``): ours against SGLang 0.5.22 and vLLM 0.31.0 in their default form on this
device, plus their Triton kernels.

- ``ours``: ``BlockFP8Linear`` with its default dispatch (warmed before capture).
- ``sglang``: SGLang's auto backend for the device (``fp8_utils``): CUTLASS blockwise
  (sgl-kernel) on sm120/sm121, FlashInfer groupwise on sm10x; ``sglang-triton`` its Triton.
- ``vllm``: vLLM's CUTLASS block-scaled ``cutlass_scaled_mm`` after its column-major
  per-token-group quantizer; ``vllm-triton`` its ``w8a8_triton_block_scaled_mm``.

Every side quantizes the BF16 activation itself. Weights rotate over copies exceeding four
times the L2 cache; one sample is a CUDA-graph replay of one call per copy. Rounds alternate
the side order (A-B-...-B-A); the median per side is reported with ours/side speedups and the
relative L2 error against the FP32 dequantized reference. One JSON line per (shape, batch).
usage: python -m benchmark.kernel_tuning.block_fp8_dense_market --sglang-path DIR
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
from pathlib import Path

import torch

SHAPES = {
    "qkv-gate": (9216, 2048),
    "out": (2048, 4096),
    "gdn-qkvz": (12288, 2048),
    "shared-gate-up": (1024, 2048),
    "shared-down": (2048, 512),
    # DeepSeek V4 Flash dense projections (TP=1).
    "dsv4-wqkv_a": (1536, 4096),
    "dsv4-wq_b": (32768, 1024),
    "dsv4-idx_wq_b": (8192, 1024),
    "dsv4-wo_b": (4096, 8192),
    "dsv4-shared_gate_up": (4096, 4096),
    "dsv4-shared_down": (4096, 2048),
}
QWEN = ("qkv-gate", "out", "gdn-qkvz", "shared-gate-up", "shared-down")
SIDES = ("ours", "sglang", "sglang-triton", "vllm", "vllm-triton")


def _sglang(path):
    from benchmark.kernel_tuning._moe_market_imports import load_sglang

    load_sglang(path)
    return importlib.import_module("sglang.srt.layers.quantization.fp8_utils")


def factories(sglang_path):
    capability = torch.cuda.get_device_capability()

    def ours(w, s):
        from ctikernels.linear.block_fp8 import BlockFP8Linear

        op = BlockFP8Linear(w, s)
        return op, op

    def sglang(w, s):
        fp8 = _sglang(sglang_path)
        name = (
            "cutlass_w8a8_block_fp8_linear_with_fallback"
            if capability[0] == 12
            else "flashinfer_gemm_w8a8_block_fp8_linear_with_fallback"
        )
        linear = getattr(fp8, name)
        return (lambda x: linear(x, w, [128, 128], s)), name

    def sglang_triton(w, s):
        linear = _sglang(sglang_path).triton_w8a8_block_fp8_linear
        return (lambda x: linear(x, w, [128, 128], s)), "triton_w8a8_block_fp8_linear"

    def vllm(w, s):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        def run(x):
            q, xs = per_token_group_quant_fp8(x, 128, column_major_scales=True)
            return ops.cutlass_scaled_mm(q, w.T, out_dtype=x.dtype, scale_a=xs, scale_b=s.T)

        return run, "cutlass_scaled_mm(block)"

    def vllm_triton(w, s):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
            w8a8_triton_block_scaled_mm,
        )

        def run(x):
            q, xs = per_token_group_quant_fp8(x, 128)
            return w8a8_triton_block_scaled_mm(q, w, xs, s, [128, 128], output_dtype=x.dtype)

        return run, "w8a8_triton_block_scaled_mm"

    return {
        "ours": ours,
        "sglang": sglang,
        "sglang-triton": sglang_triton,
        "vllm": vllm,
        "vllm-triton": vllm_triton,
    }


def _weights(n, k, copies, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for _ in range(copies):
        w = (torch.randn(n, k, device="cuda", generator=generator) * 64).clamp(-448, 448)
        exponent = torch.rand(n // 128, k // 128, device="cuda", generator=generator) * 4 - 12
        out.append((w.to(torch.float8_e4m3fn), torch.exp2(exponent)))
    return out


def _sample(graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-path", type=Path, default=os.environ.get("SGLANG_PATH"))
    parser.add_argument("--shapes", default=",".join(QWEN))
    parser.add_argument("--batches", default="1,2,3,4,8,16,64,512,4096")
    parser.add_argument("--sides", default=",".join(SIDES))
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    make = factories(args.sglang_path)
    sides = args.sides.split(",")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    device = torch.cuda.get_device_name()
    with torch.inference_mode():
        for shape in args.shapes.split(","):
            n, k = SHAPES[shape]
            copies = max(2, -(-4 * l2 // (n * k)))
            weights = _weights(n, k, copies, seed=n + k)
            ops = {side: [make[side](w, s) for w, s in weights] for side in sides}
            names = {side: ops[side][0][1] if side != "ours" else None for side in sides}
            if "ours" in ops:
                for op, _ in ops["ours"]:
                    op.warm_up(max(map(int, args.batches.split(","))))
            for m in map(int, args.batches.split(",")):
                x = (torch.randn(m, k, device="cuda") * 2).bfloat16()
                w0, s0 = weights[0]
                reference = (
                    x.float()
                    @ (w0.float() * s0.repeat_interleave(128, 0).repeat_interleave(128, 1)).T
                )
                graphs, errors = {}, {}
                for side in sides:
                    call = ops[side][0][0]
                    result = call(x).float()
                    errors[side] = ((result - reference).norm() / reference.norm()).item()
                    for c, _ in ops[side]:
                        c(x)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for c, _ in ops[side]:
                            c(x)
                    graph.replay()
                    graphs[side] = graph
                times = {side: [] for side in sides}
                for r in range(args.rounds):
                    for side in sides if r % 2 == 0 else reversed(sides):
                        times[side].append(_sample(graphs[side]) / copies)
                median = {side: statistics.median(values) for side, values in times.items()}
                record = {
                    "device": device,
                    "shape": shape,
                    "n": n,
                    "k": k,
                    "m": m,
                    "us": {side: round(value, 3) for side, value in median.items()},
                    "rel_l2": {side: round(value, 5) for side, value in errors.items()},
                    "kernels": names,
                }
                if "ours" in median:
                    record["speedup"] = {
                        side: round(median[side] / median["ours"], 3)
                        for side in sides
                        if side != "ours"
                    }
                print(json.dumps(record), flush=True)
                del graphs


if __name__ == "__main__":
    main()
