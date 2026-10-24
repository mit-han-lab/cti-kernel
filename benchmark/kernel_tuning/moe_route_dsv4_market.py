# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DeepSeek-V4 Flash router market: ctikernels.moe.route(scoring="sqrtsoftplus") vs SGLang's
moe_fused_gate (the file given by $SGLANG_GATE, executed as shipped; only its logging, JIT-utility
and radix imports are stubbed, none of which its sqrtsoftplus path uses) after the gate GEMM
(BF16 x BF16 -> FP32, torch.mm), both against an FP64 reference. Shapes: E=256, top-6, H=4096,
FP32 bias, routed scaling 1.5. Reports ids agreement (rows outside FP64 near-ties), weight error,
the exact-tie rule, and graph-replayed time per call."""

import importlib.util
import os
import statistics
import sys
import types

import torch
import torch.nn.functional as F

from ctikernels.moe import route

E, K, H, SCALE = 256, 6, 4096, 1.5


def load_sglang(path):
    """The SGLang module, with its infra imports stubbed (not its kernel code)."""
    for name in ("sglang", "sglang.kernels", "sglang.kernels.jit", "sglang.kernels.ops",
                 "sglang.kernels.ops.moe"):  # fmt: skip
        sys.modules.setdefault(name, types.ModuleType(name))
    logging_mod = types.ModuleType("sglang.kernel_api_logging")
    logging_mod.debug_kernel_api = lambda fn: fn
    utils = types.ModuleType("sglang.kernels.jit.utils")
    utils.cache_once = lambda fn: fn
    utils.load_jit = None
    utils.is_arch_support_pdl = lambda: torch.cuda.get_device_capability()[0] >= 9
    radix = types.ModuleType("sglang.kernels.ops.moe.moe_route_radix")
    radix.covered = lambda *args: False
    sys.modules.update({
        "sglang.kernel_api_logging": logging_mod,
        "sglang.kernels.jit.utils": utils,
        "sglang.kernels.ops.moe.moe_route_radix": radix,
    })  # fmt: skip
    sys.modules["sglang.kernels.ops.moe"].moe_route_radix = radix
    spec = importlib.util.spec_from_file_location("sglang_moe_fused_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.moe_fused_gate


def graph_us(fn, inner=20, outer=15):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(outer):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / inner)
    return statistics.median(out)


def reference(x, w, bias):
    """FP64: scores, selection keys sorted descending, ids (ties to the lower id)."""
    s = F.softplus(x.double() @ w.double().T).sqrt()
    key = s + bias.double()
    order = torch.sort(key, dim=1, descending=True, stable=True)
    return s, order.values, order.indices[:, :K]


def compare(name, w_got, ids_got, s, keys, ids_ref):
    margin = keys[:, K - 1] - keys[:, K]
    clear = margin > 1e-5
    same = (ids_got.long().sort(1).values == ids_ref.sort(1).values).all(1)
    want = s.gather(1, ids_got.long().clamp_min(0))
    want = want / want.sum(1, keepdim=True) * SCALE
    err = (w_got.double() - want).abs().max().item()
    return (
        f"{name}: ids same {int(same.sum())}/{len(same)} (clear rows "
        f"{int(same[clear].sum())}/{int(clear.sum())}) weight err {err:.1e}"
    )


def main():
    from ctikernels.moe import _route

    if "SCORE_GEMM_ROWS" in os.environ:
        _route.SCORE_GEMM_ROWS = int(os.environ["SCORE_GEMM_ROWS"])
    if "SCORE_TILE" in os.environ:
        _route.SCORE_TILE = tuple(int(v) for v in os.environ["SCORE_TILE"].split(","))
    print("SCORE_GEMM_ROWS", _route.SCORE_GEMM_ROWS, "SCORE_TILE", _route.SCORE_TILE, flush=True)
    gate = load_sglang(os.environ["SGLANG_GATE"])
    print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(E, H, device="cuda", generator=gen) * H**-0.5).bfloat16()
    bias = torch.randn(E, device="cuda", generator=gen) * 0.02

    def sglang(x):
        logits = torch.mm(x, w.t(), out_dtype=torch.float32)
        return gate(logits, bias, K, scoring_func="sqrtsoftplus", renormalize=True,
                    routed_scaling_factor=SCALE, apply_routed_scaling_factor_on_output=True)  # fmt: skip

    for t in (1, 4, 64, 512, 4096, 8192):
        x = torch.randn(t, H, device="cuda", generator=gen).bfloat16()

        def ours(x=x):
            return route(x, w, K, scoring="sqrtsoftplus", bias=bias, scale=SCALE)

        s, keys, ids_ref = reference(x, w, bias)
        ow, oi = ours()
        sw, si = sglang(x)
        a1, m, a2 = graph_us(ours), graph_us(lambda x=x: sglang(x)), graph_us(ours)
        o = (a1 + a2) / 2
        print(
            f"T={t} ours {o:.2f}us sglang {m:.2f}us ({m / o:.2f}x) | "
            + compare("ours", ow, oi, s, keys, ids_ref)
            + " | "
            + compare("sglang", sw, si, s, keys, ids_ref),
            flush=True,
        )
    # Exact ties: experts 3, 7 and 200 share one router row and bias, so their scores and
    # selection keys are bit-identical; both must keep the lower ids first.
    wt = w.clone()
    wt[7] = wt[3]
    wt[200] = wt[3]
    bt = bias.clone()
    bt[[3, 7, 200]] = 10.0  # always selected
    x = torch.randn(64, H, device="cuda", generator=gen).bfloat16()
    _, oi = route(x, wt, 2, scoring="sqrtsoftplus", bias=bt, scale=SCALE)
    logits = torch.mm(x, wt.t(), out_dtype=torch.float32)
    _, si = gate(logits, bt, 2, scoring_func="sqrtsoftplus", renormalize=True,
                 routed_scaling_factor=SCALE, apply_routed_scaling_factor_on_output=True)  # fmt: skip
    print("tie top-2 of {3, 7, 200}: ours", sorted(set(map(tuple, oi.sort(1).values.tolist()))),
          "sglang", sorted(set(map(tuple, si.sort(1).values.tolist()))))  # fmt: skip


if __name__ == "__main__":
    main()
