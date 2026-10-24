# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""RMS norms against the reference implementations: one table per op.

    python -m benchmark.norm.bench_norm [--timer graph|eager] [--cores 16-19] [--zero-centered]

Rows are engine shapes (decode B=1-4 and prefill rows) at each hidden size. Each comparator
column is a paired A-B-A against ours (benchmark/_harness/paired.py): its median time and the
ratio comparator/ours with a 95% bootstrap interval, so a ratio above 1 means ours is faster.
Markets that are not installed print ``n/a``.
"""

from __future__ import annotations

import argparse
import functools

import torch

from benchmark._harness import markets, paired, table, timing
from ctikernels.norm import inference as norm

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-6


def torch_rms_norm(x, weight, eps, zero_centered):
    """The frameworks' eager formula (vLLM ``forward_native``; Gemma for zero-centered)."""
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    if zero_centered:
        return (values * (1.0 + weight.float())).to(x.dtype)
    return values.to(x.dtype) * weight


def torch_add_rms_norm(x, residual, weight, eps, zero_centered):
    summed = x.float() + residual.float()
    residual.copy_(summed)
    x.copy_(torch_rms_norm(summed, weight, eps, zero_centered).to(x.dtype))


def _kernel(market, name):
    """The market's kernel ``name`` (top level or ``.norm``), or why it is unavailable."""
    if market.module is None:
        return market.reason
    for owner in (market.module, getattr(market.module, "norm", None)):
        if hasattr(owner, name):
            return getattr(owner, name)
    return f"{market.name} has no {name}"


def implementations(op, zero):
    """name -> ``make(x, residual, weight)`` returning a no-argument call, or why it is n/a.

    Every call writes its own outputs; in-place ops keep mutating the same buffers, which
    changes values but not the work.
    """
    sgl, fi, vllm = (markets.load(name) for name in ("sglang", "flashinfer", "vllm"))
    if op == "add_rms_norm_":
        found = {
            "ours": lambda x, r, w: lambda: norm.add_rms_norm_(x, r, w, EPS, zero_centered=zero),
            "torch": lambda x, r, w: lambda: torch_add_rms_norm(x, r, w, EPS, zero),
        }
        fused = "gemma_fused_add_rmsnorm" if zero else "fused_add_rmsnorm"
        kernels = {"sglang": _kernel(sgl, fused), "flashinfer": _kernel(fi, fused)}
        kernels["vllm"] = "no zero-centered op" if zero else _kernel(vllm, "fused_add_rms_norm")

        def bind(kernel):
            return lambda x, r, w: lambda: kernel(x, r, w, EPS)

    else:
        if op == "rms_norm":
            ours = lambda x, r, w: lambda: norm.rms_norm(x, w, EPS, zero_centered=zero)
        else:  # residual_rms_norm: the residual=None branch, a plain norm in the markets
            ours = lambda x, r, w: (
                lambda: norm.residual_rms_norm(r, w, EPS, zero_centered=zero, out_dtype=x.dtype)
            )
        found = {"ours": ours, "torch": lambda x, r, w: lambda: torch_rms_norm(x, w, EPS, zero)}
        plain = "gemma_rmsnorm" if zero else "rmsnorm"
        kernels = {"sglang": _kernel(sgl, plain), "flashinfer": _kernel(fi, plain)}
        kernels["vllm"] = "no zero-centered op" if zero else _kernel(vllm, "rms_norm")

        def bind(kernel):
            if kernel is kernels["vllm"]:  # vLLM writes into a caller-provided output
                return lambda x, r, w: functools.partial(kernel, torch.empty_like(x), x, w, EPS)
            return lambda x, r, w: lambda: kernel(x, w, EPS)

    for name, kernel in kernels.items():
        found[name] = kernel if isinstance(kernel, str) else bind(kernel)
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ops", default="rms_norm,add_rms_norm_,residual_rms_norm")
    parser.add_argument("--hidden", default="2048,4096,5120")
    parser.add_argument("--rows", default="1,2,3,4,64,256,1024")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--zero-centered", action="store_true")
    parser.add_argument("--timer", choices=timing.TIMERS, default="graph")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--cores", help="pin host work: 16-19, or auto (default: current mask)")
    args = parser.parse_args(argv)
    paired.pin(paired.parse_cores(args.cores))
    dtype, zero = DTYPES[args.dtype], args.zero_centered
    timer = timing.TIMERS[args.timer]
    torch.manual_seed(0)
    device = torch.device("cuda")
    print(
        f"# {torch.cuda.get_device_name(device)}, {args.dtype}, zero_centered={zero}, "
        f"timer={args.timer}, rounds={args.rounds}, cores={args.cores or 'unpinned'}"
    )
    for name in ("sglang", "flashinfer", "vllm"):
        m = markets.load(name)
        print(f"# {name}: {m.version if m.module else 'n/a (' + m.reason + ')'}")
    for op in args.ops.split(","):
        impls = implementations(op, zero)
        others = [k for k in impls if k != "ours"]
        rows = []
        for hidden in map(int, args.hidden.split(",")):
            weight = torch.randn(hidden, device=device, dtype=dtype)
            for m in map(int, args.rows.split(",")):
                x = torch.randn(m, hidden, device=device, dtype=dtype)
                residual = torch.randn_like(x)
                ours = impls["ours"](x, residual, weight)
                if op == "add_rms_norm_":
                    norm.warm_up_add_rms_norm(m, hidden=hidden, dtype=dtype, device=device)
                row = [f"{m}x{hidden}", None]
                ours_us = []  # our median from each pair; the column shows the smallest
                for key in others:
                    impl = impls[key]
                    if isinstance(impl, str):
                        row.append("n/a")
                        continue
                    result = paired.aba(ours, impl(x, residual, weight), timer, rounds=args.rounds)
                    ours_us.append(result.a_us)
                    row.append(f"{result.b_us:.2f} us, {table.ratio_cell(result.ratio, result.ci)}")
                row[1] = f"{min(ours_us):.2f}" if ours_us else "-"
                rows.append(row)
        notes = {k: impls[k] for k in others if isinstance(impls[k], str)}
        header = ["rows x hidden", "ours us", *(f"{k} (us, {k}/ours)" for k in others)]
        print(table.render(f"{op} ({args.dtype}, zero_centered={zero})", header, rows))
        for key, reason in notes.items():
            print(f"{key}: n/a - {reason}")
        print()


if __name__ == "__main__":
    main()
