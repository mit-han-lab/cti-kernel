# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Gated-MLP LoRA training step (forward + backward) against the three-node path.

    python -m benchmark.mlp.bench_mlp_lora [--rows 1024,4096,16384] [--ranks 8,64] [--cores auto]
        [--unsloth-path DIR]

Comparators, each a paired A-B-A against gated_mlp_lora (speedup = comparator time / ours,
above 1 means ours is faster):
- three-node: what the engine ran before, lora_projection_training(gate_up) ->
  activation.training.silu_and_mul -> lora_projection_training(down);
- unsloth: Unsloth's LoRA_MLP (unsloth/kernels/fast_lora.py) with its default in-place dX, BF16
  adapters. Imported from ``--unsloth-path`` (a directory holding the ``unsloth`` package or
  its kernels with a shim for the unquantized path), never bundled.
"saved" is autograd's saved activation bytes per token (factors and weights excluded), measured
with a saved-tensors hook; "peak" is the step's peak allocation above the inputs.
"""

from __future__ import annotations

import argparse
import functools
import sys

import torch

from benchmark._harness import paired, table, timing
from ctikernels.activation import training as activation
from ctikernels.linear.training import DenseProduct, lora_projection_training
from ctikernels.mlp.training import gated_mlp_lora

HIDDEN, INTERMEDIATE, SCALING = 4096, 12288, 2.0  # Qwen3-8B


def _case(rows: int, rank: int):
    torch.manual_seed(0)

    def tensor(*shape, grad=False, scale=0.02):
        return (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale).requires_grad_(
            grad
        )

    x = tensor(rows, HIDDEN, grad=True, scale=1.0)
    w_gu, w_d = tensor(2 * INTERMEDIATE, HIDDEN), tensor(HIDDEN, INTERMEDIATE)
    gate_up = tuple(
        tensor(*shape, grad=True) for shape in ((rank, HIDDEN), (INTERMEDIATE, rank)) * 2
    )
    down = (tensor(rank, INTERMEDIATE, grad=True), tensor(HIDDEN, rank, grad=True))
    grad = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    return x, w_gu, w_d, gate_up, down, grad


def three_nodes(x, w_gu, w_d, gate_up, down):
    merged = lora_projection_training(x, DenseProduct(w_gu), gate_up, scaling=SCALING)
    h = activation.silu_and_mul(merged)
    return lora_projection_training(h, DenseProduct(w_d), down, scaling=SCALING)


def ours(x, w_gu, w_d, gate_up, down):
    return gated_mlp_lora(x, DenseProduct(w_gu), DenseProduct(w_d), gate_up, down, scaling=SCALING)


def unsloth_mlp(lora_mlp):
    def fn(x, w_gu, w_d, gate_up, down):
        from unsloth.kernels.swiglu import swiglu_DWf_DW_dfg_kernel, swiglu_fg_kernel

        w_g, w_u = w_gu.chunk(2)
        a_g, b_g, a_u, b_u = gate_up
        a_d, b_d = down
        y = lora_mlp.apply(
            x.view(1, *x.shape),
            w_g, None, a_g, b_g, SCALING,
            w_u, None, a_u, b_u, SCALING,
            w_d, None, a_d, b_d, SCALING,
            swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel, True,
        )  # fmt: skip
        return y.view(x.shape[0], -1)

    return fn


def _step(fn, x, w_gu, w_d, gate_up, down, grad):
    params = (x, *gate_up, *down)
    torch.autograd.grad(fn(x, w_gu, w_d, gate_up, down), params, grad)


def saved_bytes_per_token(fn, x, w_gu, w_d, gate_up, down, grad) -> float:
    excluded = {t.data_ptr() for t in (w_gu, w_d, *gate_up, *down)}
    seen: dict[int, int] = {}

    def pack(t):
        if t.data_ptr() not in excluded:
            seen[t.data_ptr()] = max(seen.get(t.data_ptr(), 0), t.nbytes)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = fn(x, w_gu, w_d, gate_up, down)
    del y
    return sum(seen.values()) / x.shape[0]


def _peak(fn, case) -> float:
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _step(fn, *case)
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="1024,4096,16384")
    parser.add_argument("--ranks", default="8,64")
    parser.add_argument("--cores", default="auto")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--unsloth-path")
    args = parser.parse_args()
    paired.pin(paired.parse_cores(args.cores))
    comparators = {"three-node": three_nodes}
    if args.unsloth_path:
        sys.path.insert(0, args.unsloth_path)
        from unsloth.kernels.fast_lora import LoRA_MLP

        comparators["unsloth"] = unsloth_mlp(LoRA_MLP)
    timer = functools.partial(timing.eager, samples=8, calls=3, warmup=2)
    rows_out = []
    for rank in map(int, args.ranks.split(",")):
        for rows in map(int, args.rows.split(",")):
            case = _case(rows, rank)
            for name, fn in comparators.items():
                # Unsloth writes dX over its input: give it its own copy.
                own = (case[0].detach().clone().requires_grad_(), *case[1:])
                pair = paired.aba(
                    functools.partial(_step, fn, *own),
                    functools.partial(_step, ours, *case),
                    timer,
                    rounds=args.rounds,
                )
                rows_out.append(
                    (
                        rank,
                        rows,
                        name,
                        f"{pair.a_us / 1e3:.3f}",
                        f"{pair.b_us / 1e3:.3f}",
                        table.ratio_cell(1 / pair.ratio, (1 / pair.ci[1], 1 / pair.ci[0])),
                        f"{saved_bytes_per_token(fn, *own):.0f}",
                        f"{saved_bytes_per_token(ours, *case):.0f}",
                        f"{_peak(fn, own):.0f}",
                        f"{_peak(ours, case):.0f}",
                    )
                )
    print(
        table.render(
            f"gated-MLP LoRA fwd+bwd, H={HIDDEN} I={INTERMEDIATE} bf16, "
            f"{torch.cuda.get_device_name()}",
            (
                "rank",
                "T",
                "comparator",
                "comparator ms",
                "ours ms",
                "speedup [95% CI]",
                "saved B/tok comparator",
                "saved B/tok ours",
                "peak MiB comparator",
                "peak MiB ours",
            ),
            rows_out,
        )
    )


if __name__ == "__main__":
    main()
