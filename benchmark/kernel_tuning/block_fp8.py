# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Time block-FP8 (W8A8) inference against its baselines at Qwen3.5-35B-A3B-FP8 shapes.

Dense projections (merged q/k/v/gate, attention/GDN output, merged GDN qkvz, shared-expert
gate_up and down) and the routed-expert MoE forward (256 experts, top 8, hidden 2048,
intermediate 512: quantize, gate_up GEMM, SiLU x up, quantize, down GEMM, combine). Every path
runs on the same operands; the weights rotate over enough copies to exceed the L2 cache, and
each sample is one CUDA-graph replay of a call per copy. One JSON line per (shape, batch,
path) with the median microseconds per call.

Built-in paths: ``bf16`` (the existing BF16 GEMM or MoE kernel on decoded weights),
``ours`` (``BlockFP8Linear`` / the block-FP8 ``fused_moe_kernel_triton``) and, for dense,
``flashinfer`` (sgl-kernel's quantizer and FlashInfer's groupwise CUTLASS GEMM). ``--extra
MODULE`` adds paths from ``MODULE.DENSE_PATHS`` / ``MODULE.MOE_PATHS`` (name -> factory with
the built-in factories' signatures), for baselines that are not dependencies of this package.

usage: python -m benchmark.kernel_tuning.block_fp8 [--dense] [--moe] [--batches 1,2,4]
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
from collections.abc import Callable
from functools import partial

import torch

DENSE_SHAPES = {
    "qkv-gate": (9216, 2048),
    "out": (2048, 4096),
    "gdn-qkvz": (12288, 2048),
    "shared-gate-up": (1024, 2048),
    "shared-down": (2048, 512),
}
EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 256, 8, 2048, 512
DEFAULT_BATCHES = (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
_ROTATION_BYTES = 512 << 20


def _decoded(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from ctikernels.linear.block_fp8 import dequantize_block_fp8

    return dequantize_block_fp8(weight, scale)


def _sgl_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import sgl_per_token_group_quant_fp8

    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(x.shape[0], x.shape[1] // 128, device=x.device, dtype=torch.float32)
    sgl_per_token_group_quant_fp8(x, q, scale, 128, 1e-10, -448.0, 448.0)
    return q, scale


def dense_bf16(weight: torch.Tensor, scale: torch.Tensor) -> Callable:
    decoded = _decoded(weight, scale)
    return lambda x: torch.nn.functional.linear(x, decoded)


def dense_ours(weight: torch.Tensor, scale: torch.Tensor) -> Callable:
    from ctikernels.linear.block_fp8 import BlockFP8Linear

    return BlockFP8Linear(weight, scale)


def dense_flashinfer(weight: torch.Tensor, scale: torch.Tensor) -> Callable:
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    def run(x: torch.Tensor) -> torch.Tensor:
        q, x_scale = _sgl_quantize(x)
        return gemm_fp8_nt_groupwise(
            q, weight, x_scale, scale, scale_major_mode="K", out_dtype=x.dtype, backend="cutlass"
        )

    return run


def moe_config_bf16(tokens: int) -> dict:
    """The engine's fused-MoE configuration (Mini-SGLang's defaults)."""
    if tokens <= EXPERTS:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    return {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}


def moe_forward(
    launch: Callable,
    quantize: Callable | None,
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    s13: torch.Tensor | None,
    s2: torch.Tensor | None,
    weights: torch.Tensor,
    ids: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """The routed-expert forward with ``launch`` (a ``fused_moe_kernel_triton``)."""
    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.moe_impl import moe_align_block_size, moe_sum_reduce_triton

    m, top_k = ids.shape
    two_i = w13.shape[1]
    sorted_ids, block_experts, padded = moe_align_block_size(
        ids, config["BLOCK_SIZE_M"], w13.shape[0]
    )
    gate_up = torch.empty(m, top_k, two_i, device=x.device, dtype=x.dtype)
    act = torch.empty(m * top_k, two_i // 2, device=x.device, dtype=x.dtype)
    down = torch.empty(m, top_k, w2.shape[1], device=x.device, dtype=x.dtype)
    for down_projection, (source, target, weight, scale) in enumerate(
        ((x, gate_up, w13, s13), (act, down, w2, s2))
    ):
        if down_projection:
            silu_and_mul(gate_up.view(-1, two_i), act)
        operand, extra = source, {}
        if quantize is not None:
            operand, a_scale = quantize(source)
            extra = {"a_scale": a_scale, "b_scale": scale}
        launch(
            operand, weight, target, weights, ids, sorted_ids, block_experts, padded,
            False, 1 if down_projection else top_k, config, x.dtype, **extra,
        )  # fmt: skip
    out = torch.empty(m, w2.shape[1], device=x.device, dtype=x.dtype)
    moe_sum_reduce_triton(down, out, topk_ids=ids, topk_weights=weights)  # vLLM's combine
    return out


def moe_bf16(w13, s13, w2, s2) -> Callable:
    from ctikernels.moe_impl import fused_moe_kernel_triton

    d13, d2 = _decoded(w13, s13), _decoded(w2, s2)

    def run(x, weights, ids):
        config = moe_config_bf16(x.shape[0])
        return moe_forward(
            fused_moe_kernel_triton, None, x, d13, d2, None, None, weights, ids, config
        )

    return run


def moe_ours(w13, s13, w2, s2) -> Callable:
    from ctikernels.linear.block_fp8 import quantize_per_token_group_fp8
    from ctikernels.moe_impl import block_fp8_moe_config, fused_moe_kernel_triton

    capability = torch.cuda.get_device_capability(w13.device)

    def run(x, weights, ids):
        config = block_fp8_moe_config(
            x.shape[0], w13.shape[0], w13.shape[1], x.shape[1], TOP_K, capability=capability
        )
        return moe_forward(
            fused_moe_kernel_triton, quantize_per_token_group_fp8, x, w13, w2, s13, s2,
            weights, ids, config,
        )  # fmt: skip

    return run


# Keep the established factory as the exact Triton baseline.
moe_triton = moe_ours


def moe_cute(w13, s13, w2, s2) -> Callable:
    from ctikernels.moe_block_fp8 import BlockFP8MoE

    return BlockFP8MoE(w13, s13, w2, s2, top_k=TOP_K, provider="cute-fp8")


MOE_PATHS = {"triton-fp8": moe_triton, "cute-fp8": moe_cute}


def _random_weight(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = (torch.randn(*shape, device="cuda") * 96).clamp(-448, 448).to(torch.float8_e4m3fn)
    blocks = (*shape[:-2], -(-shape[-2] // 128), -(-shape[-1] // 128))
    scale = torch.exp2(torch.rand(blocks, device="cuda") * 6 - 14)
    return weight, scale


def _median_us(calls: list[Callable], iterations: int, *, warmups=()) -> float:
    # Eager calls do not satisfy the block/static FP8 capture contract.
    # Callers supply explicit warm_up calls for every op and layout they capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for warm in warmups:
            warm()
        for call in calls:
            call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for call in calls:
            call()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    return statistics.median(samples)


def _emit(record: dict) -> None:
    print("BENCH " + json.dumps(record), flush=True)


@torch.inference_mode()
def bench_dense(paths: dict, batches, iterations: int, base: dict) -> None:
    for name, (n, k) in DENSE_SHAPES.items():
        copies = max(2, -(-_ROTATION_BYTES // (n * k)))
        weights = [_random_weight(n, k) for _ in range(copies)]
        ops = {path: [factory(*w) for w in weights] for path, factory in paths.items()}
        for m in batches:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for path, path_ops in ops.items():
                calls = [lambda op=op, x=x: op(x) for op in path_ops]
                warmups = [
                    partial(op.warm_up, batch_sizes=[m], dtype=x.dtype)
                    for op in path_ops
                    if hasattr(op, "warm_up")
                ]
                us = _median_us(calls, iterations, warmups=warmups)
                _emit({**base, "op": "dense", "shape": name, "N": n, "K": k, "M": m,
                       "path": path, "us": us})  # fmt: skip
        del ops, weights
        torch.cuda.empty_cache()


@torch.inference_mode()
def bench_moe(paths: dict, batches, iterations: int, base: dict) -> None:
    w13, s13 = _random_weight(EXPERTS, 2 * INTERMEDIATE, HIDDEN)
    w2, s2 = _random_weight(EXPERTS, HIDDEN, INTERMEDIATE)
    runs = {path: factory(w13, s13, w2, s2) for path, factory in paths.items()}
    rotations = 8  # routings per sample, so that small batches touch new experts each call
    for m in batches:
        cases = []
        for _ in range(rotations):
            probabilities = torch.randn(m, EXPERTS, device="cuda").softmax(-1)
            weights, ids = probabilities.topk(TOP_K, dim=-1)
            x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
            cases.append((x, weights / weights.sum(-1, keepdim=True), ids.to(torch.int32)))
        for path, run in runs.items():
            calls = [lambda case=case, run=run: run(*case) for case in cases]
            us = _median_us(calls, iterations)
            _emit({**base, "op": "moe", "M": m, "path": path, "us": us})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--extra", help="module with DENSE_PATHS / MOE_PATHS baselines")
    args = parser.parse_args()
    batches = [int(m) for m in args.batches.split(",")]
    dense = {"bf16": dense_bf16, "flashinfer": dense_flashinfer, "ours": dense_ours}
    moe = {"bf16": moe_bf16, "ours": moe_ours, **MOE_PATHS}
    if args.extra:
        extra = importlib.import_module(args.extra)
        dense.update(getattr(extra, "DENSE_PATHS", {}))
        moe.update(getattr(extra, "MOE_PATHS", {}))
    import triton

    base = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }
    torch.manual_seed(0)
    if args.dense or not args.moe:
        bench_dense(dense, batches, args.iterations, base)
    if args.moe or not args.dense:
        bench_moe(moe, batches, args.iterations, base)


if __name__ == "__main__":
    main()
