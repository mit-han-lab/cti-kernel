# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Time the block-FP8 (W8A8) gated MLP against the unfused path at Qwen3.5 FP8 shapes.

Shapes: Qwen3.5-35B-A3B-FP8's shared expert (hidden 2048, intermediate 512) and
Qwen3.5-27B-FP8's dense MLP (hidden 5120, intermediate 17408), each at TP 1, 2 and 4 (the
intermediate split). Two operations per path: ``fc1`` (the quantized SiLU x up the down
projection consumes) and ``mlp`` (``fc1`` and the down projection). The weights rotate over
enough copies to exceed the L2 cache and each sample is one CUDA-graph replay of a call per
copy; one JSON line per (shape, batch, operation, path) with the median microseconds per call.

Paths: ``unfused`` (``BlockFP8Linear`` gate_up, FlashInfer's ``silu_and_mul``,
``quantize_per_token_group_fp8``), ``cute-act-quant``, ``cute-fp8-fused`` (M <= 16),
and, where a qualified FlashInfer FC1
is installed, ``flashinfer-cute-dsl`` (``BlockFP8GateUpSiLU`` providers). The down projection
is ``BlockFP8Linear.quantized`` with its default dispatch for every path.

usage: python -m benchmark.kernel_tuning.block_fp8_gated [--shapes a,b] [--batches 1,2,4]
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from functools import partial

import torch

from benchmark.kernel_tuning.block_fp8 import _median_us, _random_weight

SHAPES = {  # name -> (intermediate I, hidden K)
    "35b-shared": (512, 2048),
    "35b-shared-tp2": (256, 2048),
    "35b-shared-tp4": (128, 2048),
    "27b-mlp": (17408, 5120),
    "27b-mlp-tp2": (8704, 5120),
    "27b-mlp-tp4": (4352, 5120),
}
DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
_ROTATION_BYTES = 512 << 20


def fc1_unfused(w13: torch.Tensor, s13: torch.Tensor) -> Callable:
    from ctikernels.activation.inference import silu_and_mul
    from ctikernels.linear.block_fp8 import BlockFP8Linear, quantize_per_token_group_fp8

    gate_up = BlockFP8Linear(w13, s13)

    def run(x):
        return quantize_per_token_group_fp8(silu_and_mul(gate_up(x)))

    run.warm_up = gate_up.warm_up
    return run


def _fc1_provider(provider: str) -> Callable:
    def factory(w13: torch.Tensor, s13: torch.Tensor) -> Callable:
        from ctikernels.linear.block_fp8_gated import BlockFP8GateUpSiLU

        return BlockFP8GateUpSiLU(w13, s13, provider=provider)

    return factory


def _flashinfer_available(i: int, k: int) -> bool:
    from ctikernels.linear.flashinfer._block_fp8_gated import _flashinfer_fc1

    try:
        _flashinfer_fc1("benchmark", torch.cuda.get_device_capability(), i, k)
    except (RuntimeError, ValueError):
        return False
    return True


PATHS = {
    "unfused": fc1_unfused,
    "cute-act-quant": _fc1_provider("cute-act-quant"),
    "cute-fp8-fused": _fc1_provider("cute-fp8-fused"),
    "flashinfer-cute-dsl": _fc1_provider("flashinfer-cute-dsl"),
}


def _emit(record: dict) -> None:
    print("BENCH " + json.dumps(record), flush=True)


def _warm_mlp(fc1, down, x):
    fc1.warm_up(batch_sizes=[x.shape[0]], dtype=x.dtype)
    q, scale = fc1(x)
    down.warm_up(
        batch_sizes=[x.shape[0]],
        quantized=True,
        input_layouts=[
            ((q.stride(0), bool(q.data_ptr() % 16)), (*scale.stride(), bool(scale.data_ptr() % 16)))
        ],
    )


@torch.inference_mode()
def bench(shapes, paths, batches, iterations: int, base: dict) -> None:
    from ctikernels.linear.block_fp8 import BlockFP8Linear

    for name in shapes:
        i, k = SHAPES[name]
        copies = max(2, -(-_ROTATION_BYTES // (3 * i * k)))
        weights = [(_random_weight(2 * i, k), _random_weight(k, i)) for _ in range(copies)]
        names = [p for p in paths if p != "flashinfer-cute-dsl" or _flashinfer_available(i, k)]
        fc1 = {p: [PATHS[p](*w13) for w13, _ in weights] for p in names}
        down = [BlockFP8Linear(*w2) for _, w2 in weights]
        for m in batches:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            for path in names:
                if path == "cute-fp8-fused" and m > 16:
                    continue  # Do not label fallback GEMM timings as fused FC1.
                ops = list(zip(fc1[path], down, strict=True))
                warmups = [partial(_warm_mlp, f, d, x) for f, d in ops]
                calls = {
                    "fc1": [lambda f=f, x=x: f(x) for f, _ in ops],
                    "mlp": [lambda f=f, d=d, x=x: d.quantized(*f(x)) for f, d in ops],
                }
                for operation, call in calls.items():
                    us = _median_us(call, iterations, warmups=warmups)
                    _emit({**base, "op": operation, "shape": name, "I": i, "K": k, "M": m,
                           "path": path, "us": us})  # fmt: skip
        del fc1, down, weights
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--paths", default=",".join(PATHS))
    parser.add_argument("--batches", default=",".join(map(str, DEFAULT_BATCHES)))
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    import triton

    base = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "python": platform.python_version(),
    }
    torch.manual_seed(0)
    bench(
        args.shapes.split(","),
        args.paths.split(","),
        [int(m) for m in args.batches.split(",")],
        args.iterations,
        base,
    )


if __name__ == "__main__":
    main()
