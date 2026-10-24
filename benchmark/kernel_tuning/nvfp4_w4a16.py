# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""NVFP4 W4A16 linear vs cuBLAS BF16, paired A-B-A on one GPU (CUDA graphs), against two
baselines: cuBLAS on a resident BF16 copy of the dequantized weight (2 bytes per element
extra), and the same after dequantizing the weight on every call (no extra copy). Rows:
LM-head shapes (vocab x hidden); ``forward`` is ``x . W^T`` (BF16 x), ``input_grad`` is
``dy . W`` (cuBLAS after casting FP32 dy to BF16; ours reads FP32 dy, one term). Speedup =
baseline time / ours.

    python -m benchmark.kernel_tuning.nvfp4_w4a16 [--shapes 151936x4096,...] [--rows 1,2,...]
"""

from __future__ import annotations

import argparse
import functools

import torch
import triton
import triton.language as tl

from benchmark._harness import paired, table, timing
from ctikernels.linear.nvfp4 import NVFP4Weight, dequantize_nvfp4
from ctikernels.linear.training import NVFP4W4A16Linear
from ctikernels.linear.triton._nvfp4_w4a16 import _decode_e2m1, _decode_e4m3

SHAPES = ((151936, 4096), (151936, 5120), (248320, 4096), (248320, 5120))
ROWS = (1, 2, 3, 4, 64, 256, 512, 4096)


def _weight(n: int, k: int) -> NVFP4Weight:
    data = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(48, 100, (n, k // 16), dtype=torch.uint8, device="cuda")
    return NVFP4Weight(data, scale.view(torch.float8_e4m3fn), torch.ones(1, device="cuda"))


@triton.jit
def _dequant_kernel(data, scale, out, pairs, k, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < pairs
    packed = tl.load(data + i, mask=mask, other=0)
    s = _decode_e4m3(tl.load(scale + (i // (k // 2)) * (k // 16) + i % (k // 2) // 8, mask=mask))
    tl.store(out + 2 * i, (_decode_e2m1(packed & 15) * s).to(tl.bfloat16), mask)
    tl.store(out + 2 * i + 1, (_decode_e2m1(packed >> 4) * s).to(tl.bfloat16), mask)


def _dequant(w: NVFP4Weight, out: torch.Tensor) -> None:
    pairs = w.data.numel()
    block = 1024
    _dequant_kernel[(triton.cdiv(pairs, block),)](
        w.data, w.scale.view(torch.uint8), out, pairs, out.shape[1], BLOCK=block
    )


def _bf16_dense(w: NVFP4Weight) -> torch.Tensor:
    n, k = w.data.shape[0], w.data.shape[1] * 2
    dense = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    for start in range(0, n, 8192):  # chunked: the FP32 decode of the whole weight is large
        rows = slice(start, start + 8192)
        dense[rows] = dequantize_nvfp4(w.data[rows], w.scale[rows]).to(torch.bfloat16)
    return dense


def _rows(op, w, dense, m: int, timer, rounds: int) -> list[tuple]:
    n, k = dense.shape
    alpha = torch.ones(1, device="cuda")
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(m, n, device="cuda")
    dx = torch.empty(m, k, device="cuda")
    dx16 = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
    cases = {
        "forward": (
            lambda: torch.mm(x, dense.T, out=y),
            lambda: (_dequant(w, dense), torch.mm(x, dense.T, out=y)),
            lambda: op(x, alpha, out=y),
        ),
        "input_grad": (
            lambda: torch.mm(dy.to(torch.bfloat16), dense, out=dx16),
            lambda: (_dequant(w, dense), torch.mm(dy.to(torch.bfloat16), dense, out=dx16)),
            lambda: op.input_grad(dy, alpha, out=dx),
        ),
    }
    out = []
    for name, (resident, dequant, ours) in cases.items():
        cells = [name, f"{n}x{k}", m]
        for base in (resident, dequant):
            pair = paired.aba(base, ours, timer, rounds=rounds)
            speedup = table.ratio_cell(1 / pair.ratio, (1 / pair.ci[1], 1 / pair.ci[0]))
            cells += [f"{pair.a_us:.1f}", speedup]
        cells.insert(5, f"{pair.b_us:.1f}")
        out.append(tuple(cells))
        print(out[-1], flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=",".join(f"{n}x{k}" for n, k in SHAPES))
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    timer = functools.partial(timing.graph, samples=5, calls=5, warmup=2)
    rows = []
    for shape in args.shapes.split(","):
        w = _weight(*map(int, shape.split("x")))
        op, dense = NVFP4W4A16Linear(w), _bf16_dense(w)
        for m in map(int, args.rows.split(",")):
            rows += _rows(op, w, dense, m, timer, args.rounds)
        del op, dense
        torch.cuda.empty_cache()
    print(
        table.render(
            f"NVFP4 W4A16 vs cuBLAS BF16 ({torch.cuda.get_device_name()})",
            (
                "op",
                "N x K",
                "M",
                "resident us",
                "vs resident",
                "ours us",
                "dequant+cuBLAS us",
                "vs dequant",
            ),
            rows,
        )
    )


if __name__ == "__main__":
    main()
