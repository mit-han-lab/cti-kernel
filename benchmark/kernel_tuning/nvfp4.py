# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Time the NVFP4 gated-MLP input projection and linear against their baselines at Qwen3.5
dense and shared-expert shapes.

Gated (``--gated``, ``x [M, H] -> [M, I]``): ``bf16`` (cuBLAS GEMM on the merged BF16 weight,
then ``silu_and_mul``), and ``NVFP4GateUpSiLU`` on its ``unfused`` path (x quantization, QuACK
W4A4 GEMM to BF16, Triton SiLU * up quantizer) and ``fused`` path (x quantization, QuACK W4A4
GEMM with SiLU * up and NVFP4 quantization in the epilogue), each under every candidate QuACK
configuration (``--configs``), up to 64 rows the ``triton`` small-batch GEMM under every
candidate configuration (``--small``), and ``ours`` (the plan ``NVFP4GateUpSiLU``
dispatches). Linear (``--linear``, the down projection ``[M, I] -> [M, H]``): ``bf16``
(cuBLAS) and ``NVFP4Linear`` including input quantization from BF16. Every path runs on the same
operands; the weights rotate over enough copies to exceed the L2 cache, and each sample is one
CUDA-graph replay of a call per copy. One JSON line per (shape, batch, path, configuration)
with the median microseconds per call.

``sglang`` imports pinned SGLang 72730d6 at runtime, preserving its qwen3x NVFP4 GEMM probe and auto
FlashInfer dispatch. SGLang rounds gate_up GEMM output to BF16 before SiLU*up+quant;
our fused path does not; gate 2 checks that it is no worse vs FP64. The shared expert uses
separate SGLang activation and down-input quantization. All gated paths, including bf16,
end at NVFP4 + scales.

The optional ``flashinfer`` configuration uses FlashInfer 0.6.11.post3 CUTLASS with
our activation quantizer: linear path ``flashinfer``, gated path ``unfused``. Its
weights are prepared outside timing. Unavailable backends emit an error, not a timing.

The Triton path uses per-operation arrival counters for split-K; each object's calls and
graph replays run in order on one stream.

usage: python -m benchmark.kernel_tuning.nvfp4 [--gated] [--linear] [--batches 1,2,4]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from collections.abc import Callable

import torch

# Qwen3.5 (hidden, intermediate): 9B and 27B dense MLPs, 35B-A3B shared expert.
SHAPES = {"9b": (4096, 12288), "27b": (5120, 17408), "35b-shared": (2048, 512)}
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
CONFIGS = {
    "p128x128": (128, 128, True, True),
    "c128x128": (128, 128, False, True),
    "c128x256": (128, 256, False, True),
    "c256x128": (256, 128, False, True),
    "s128x128": (128, 128, True, False),
    "flashinfer": None,
}
# Triton small-batch configurations (splits, block N, block K, warps, stages, prequantize),
# labelled t<splits>n<N>k<K><q: quantize first | i: in the GEMM>.
SMALL = {
    f"t{splits}n{n}k{k}{'q' if pre else 'i'}": (splits, n, k, 4, 2, pre)
    for splits in (1, 2, 4, 8, 16)
    for n in (32, 64, 128)
    for k in (128, 256)
    for pre in (False, True)
}
PATHS = ("bf16", "unfused", "fused", "quack", "triton", "ours", "sglang", "flashinfer")
SMALL_ROWS = 64
_ROTATION_BYTES = 512 << 20


def _weight(rows: int, cols: int):
    from ctikernels.linear.nvfp4 import NVFP4Weight

    data = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda")
    scale = (torch.rand(rows, cols // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    return NVFP4Weight(data, scale, torch.full((1,), 0.02 / 448 / 6, device="cuda"))


def _decoded(weight) -> torch.Tensor:
    from ctikernels.linear.nvfp4 import dequantize_nvfp4

    return dequantize_nvfp4(*weight).to(torch.bfloat16)


def _median_us(calls: list[Callable], iterations: int) -> float:
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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


def _forced(op, path: str | None, config) -> None:
    op.plan_for = (lambda rows: config) if path is None else (lambda rows: (path, config))


def _dispatch_metadata(op, rows: int, *, gated: bool) -> dict:
    """Record both enabled provider and the actual row's plan in a mixed dispatch table."""
    from ctikernels.linear._nvfp4_dispatch import Cute, FlashInfer, Small

    plan = op.plan_for(rows)
    config = plan[1] if gated else plan
    if config is None:
        provider = "dequant"
    elif isinstance(config, FlashInfer):
        provider = "flashinfer"
    elif isinstance(config, Small):
        provider = "triton"
    elif isinstance(config, Cute):
        provider = "cute"
    else:
        provider = "quack"
    fields = None if config is None else list(config)
    return {
        "provider": op.provider,
        "dispatched_provider": provider,
        "implementation_language": provider,
        "plan_type": type(config).__name__,
        "plan": [plan[0], fields] if gated else fields,
    }


def _candidates(configs: dict, small: dict, m: int, k: int):
    """(path suffix, label, configuration) of every candidate for ``m`` rows and depth ``k``."""
    from ctikernels.linear._nvfp4_dispatch import FlashInfer, Small
    from ctikernels.linear._nvfp4_quack import Config

    for label, fields in configs.items():
        if label == "flashinfer":
            yield "flashinfer", label, FlashInfer()
        else:
            yield "quack", label, Config(*fields)
    if m <= SMALL_ROWS:
        for label, fields in small.items():
            if k % (fields[0] * fields[2]) == 0:
                yield "triton", label, Small(*fields)


def _flashinfer_ops(weights, factory, enabled):
    if not enabled:
        return [], None
    try:
        return [factory(w) for w in weights], None
    except Exception as error:  # noqa: BLE001 - record optional backend failures in the sweep
        return [], repr(error)[:200]


@torch.inference_mode()
def bench_gated(shapes, batches, configs, small, paths, iterations: int, base: dict) -> None:
    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.linear.nvfp4 import NVFP4GateUpSiLU, quantize_nvfp4

    from .nvfp4_sglang import SGLangGated

    for name in shapes:
        h, i = SHAPES[name]
        copies = max(2, -(-_ROTATION_BYTES // (i * h)))
        scale = torch.full((1,), 4.0 / 2688.0, device="cuda")
        weights = [(_weight(i, h), _weight(i, h)) for _ in range(copies)]
        # The benchmark recipe gives both weights the same constant descale. Share
        # its storage so FlashInfer can project the merged weight with one alpha.
        weights = [(g, u._replace(global_scale=g.global_scale)) for g, u in weights]
        flash_ops, flash_error = _flashinfer_ops(
            weights,
            lambda pair, scale=scale: NVFP4GateUpSiLU(
                *pair, input_scale=scale, output_scale=scale, provider="flashinfer"
            ),
            "flashinfer" in configs and "unfused" in paths,
        )
        ops = [NVFP4GateUpSiLU(g, u, input_scale=scale, output_scale=scale) for g, u in weights]
        sglang = (
            [SGLangGated(g, u, scale, scale, shared=name == "35b-shared") for g, u in weights]
            if "sglang" in paths
            else []
        )
        bf16_copies = max(2, -(-_ROTATION_BYTES // (4 * i * h)))
        merged = [
            torch.cat((_decoded(op_gate), _decoded(op_up)))
            for op_gate, op_up in weights[:bf16_copies]
        ]
        for m in batches:
            x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16)
            row = {**base, "op": "gated", "shape": name, "H": h, "I": i, "M": m}

            def bf16(weight, x=x, scale=scale):
                return quantize_nvfp4(silu_and_mul(torch.nn.functional.linear(x, weight)), scale)

            if sglang:
                sglang[0].probe(x)
                us = _median_us([lambda op=op, x=x: op(x) for op in sglang], iterations)
                _emit({**row, "path": "sglang", "config": None, "us": us, **sglang[0].metadata(m)})
            if "bf16" in paths:
                us = _median_us([lambda w=w: bf16(w) for w in merged], iterations)
                _emit({**row, "path": "bf16", "config": None, "us": us})
            for kind, label, config in _candidates(configs, small, m, h):
                candidates = {
                    "quack": ("unfused", "fused"),
                    "flashinfer": ("unfused",),
                    "triton": ("triton",),
                }[kind]
                for path in (path for path in candidates if path in paths):
                    selected = flash_ops if kind == "flashinfer" else ops
                    if kind == "flashinfer" and flash_error is not None:
                        _emit({**row, "path": path, "config": label, "error": flash_error})
                        continue
                    for op in selected:
                        _forced(op, path if kind == "triton" else f"quack-{path}", config)
                    try:
                        us = _median_us([lambda op=op, x=x: op(x) for op in selected], iterations)
                    except Exception as error:  # noqa: BLE001 - a configuration may not compile
                        _emit({**row, "path": path, "config": label, "error": repr(error)[:200]})
                        continue
                    _emit({**row, "path": path, "config": label, "us": us})
            for op in ops:
                op.__dict__.pop("plan_for", None)
            if "ours" in paths:
                us = _median_us([lambda op=op, x=x: op(x) for op in ops], iterations)
                _emit({**row, "path": "ours", "config": None, "us": us,
                       **_dispatch_metadata(ops[0], m, gated=True)})  # fmt: skip
        del ops, flash_ops, merged, weights, sglang
        torch.cuda.empty_cache()


@torch.inference_mode()
def bench_linear(shapes, batches, configs, small, paths, iterations: int, base: dict) -> None:
    from ctikernels.linear.nvfp4 import NVFP4Linear

    from .nvfp4_sglang import SGLangLinear

    for name in shapes:
        h, i = SHAPES[name]
        n, k = h, i  # the down projection
        copies = max(2, -(-_ROTATION_BYTES // (n * k // 2)))
        scale = torch.full((1,), 4.0 / 2688.0, device="cuda")
        weights = [_weight(n, k) for _ in range(copies)]
        ops = [NVFP4Linear(w, scale) for w in weights]
        flash_ops, flash_error = _flashinfer_ops(
            weights,
            lambda w, scale=scale: NVFP4Linear(w, scale, provider="flashinfer"),
            "flashinfer" in configs and "flashinfer" in paths,
        )
        sglang = [SGLangLinear((w,), scale) for w in weights] if "sglang" in paths else []
        bf16_copies = max(2, -(-_ROTATION_BYTES // (2 * n * k)))
        decoded = [_decoded(w) for w in weights[:bf16_copies]]
        for m in batches:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            row = {**base, "op": "linear", "shape": name, "N": n, "K": k, "M": m}
            if sglang:
                sglang[0].probe(x)
                us = _median_us([lambda op=op, x=x: op(x) for op in sglang], iterations)
                _emit({**row, "path": "sglang", "config": None, "us": us, **sglang[0].metadata(m)})
            if "bf16" in paths:
                calls = [lambda w=w, x=x: torch.nn.functional.linear(x, w) for w in decoded]
                us = _median_us(calls, iterations)
                _emit({**row, "path": "bf16", "config": None, "us": us})
            for kind, label, config in _candidates(configs, small, m, k):
                path = "unfused" if kind == "quack" and "unfused" in paths else kind
                if path not in paths:
                    continue
                selected = flash_ops if kind == "flashinfer" else ops
                if kind == "flashinfer" and flash_error is not None:
                    _emit({**row, "path": path, "config": label, "error": flash_error})
                    continue
                for op in selected:
                    _forced(op, None, config)
                # Include input quantization: every linear path starts from the same BF16 x.
                try:
                    us = _median_us([lambda op=op, x=x: op(x) for op in selected], iterations)
                except Exception as error:  # noqa: BLE001 - a configuration may not compile
                    _emit({**row, "path": path, "config": label, "error": repr(error)[:200]})
                    continue
                _emit({**row, "path": path, "config": label, "us": us})
            for op in ops:
                op.__dict__.pop("plan_for", None)
            if "ours" in paths:
                us = _median_us([lambda op=op, x=x: op(x) for op in ops], iterations)
                _emit({**row, "path": "ours", "config": None, "us": us,
                       **_dispatch_metadata(ops[0], m, gated=False)})  # fmt: skip
        del ops, flash_ops, decoded, weights, sglang
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument(
        "--configs", default=",".join(CONFIGS), help="QuACK labels and flashinfer ('' for none)"
    )
    parser.add_argument("--small", default=",".join(SMALL), help="Triton ('' for none)")
    parser.add_argument("--paths", default=",".join(p for p in PATHS if p != "sglang"))
    parser.add_argument("--sglang-python", help="unmodified pinned SGLang python subtree")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    shapes = args.shapes.split(",")
    batches = [int(m) for m in args.batches.split(",")]
    configs = {label: CONFIGS[label] for label in args.configs.split(",") if label}
    small = {label: SMALL[label] for label in args.small.split(",") if label}
    paths = set(args.paths.split(","))
    import triton

    base = {
        "boundary_version": 2,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }
    if "sglang" in paths:
        from .nvfp4_sglang import load

        load(args.sglang_python)
    torch.manual_seed(0)
    if args.gated or not args.linear:
        bench_gated(shapes, batches, configs, small, paths, args.iterations, base)
    if args.linear or not args.gated:
        bench_linear(shapes, batches, configs, small, paths, args.iterations, base)


if __name__ == "__main__":
    main()
