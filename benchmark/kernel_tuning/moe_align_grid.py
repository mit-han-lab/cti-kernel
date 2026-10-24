# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""moe_align_block_size grid: the provider production picks (``moe_align_block_size`` for EP1,
``moe_align_block_size_local`` with the rank's map for EP) vs the faster market comparator.

Market, each in its strongest released form:
- SGLang: ``sgl_kernel.moe_align_block_size`` (E + 1 buckets, padded), and, where SGLang's
  wrapper takes it (<= 64 routes, > 64 buckets), its Triton ``moe_align_small_numel``
  (``--sglang-path``, a checkout's ``python/`` directory). EP: ids already local (-1 remote).
- vLLM 0.31.0 ``_moe_C.moe_align_block_size`` (``--vllm-vendor``: directory holding the
  ``vllm-0.31.0-ops`` and ``vllm-0.31.0-moe`` library trees); EP: its expert_map in-kernel.

Also timed: our Triton and our CuTe on their own (CuTe where it runs), for the provider table.
CUDA-graph replay, a non-PDL one-element kernel before every call (its twin subtracted), paired
A-B-A on ours, median us per call. Correct = the same routes per expert block as a reference.
One JSON line per row; speedup = faster market / ours (auto).

    python moe_align_grid.py --e 128 256 --mode ep1 ep2 --k 6 8 [--t ...] [--block ...]
"""

import argparse
import importlib.machinery
import json
import os
import platform
import statistics
import sys
import types

import torch

from ctikernels import moe_align_block_size
from ctikernels.moe import moe_align_block_size_local
from ctikernels.moe.triton._slot_layout import align_block_size

TS = (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

_SEP = []


def separator():
    if not _SEP:
        _SEP.append(torch.zeros(1, device="cuda"))
    _SEP[0].add_(1)


def graph_us(fns, inner=20, outer=15):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(outer):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(2_000_000)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / inner)
    return statistics.median(times)


def timed(fn):
    return max(graph_us((separator, fn)) - graph_us((separator,)), 1e-3)


def load_vllm(vendor):
    for tree, lib in (
        ("vllm-0.31.0-ops", "_C_stable_libtorch.abi3.so"),
        ("vllm-0.31.0-moe", "_moe_C_stable_libtorch.abi3.so"),
    ):
        torch.ops.load_library(os.path.join(vendor, tree, platform.machine(), lib))

    def align(ids, bs, e, expert_map=None):  # vLLM's wrapper; EP with ignore_invalid_experts
        cap = ids.numel() + e * (bs - 1)
        if ids.numel() < e:
            cap = min(ids.numel() * bs, cap)
        sorted_ids = torch.empty(cap, dtype=torch.int32, device=ids.device)
        experts = torch.empty(-(-cap // bs), dtype=torch.int32, device=ids.device)
        total = torch.empty(1, dtype=torch.int32, device=ids.device)
        torch.ops._moe_C.moe_align_block_size(ids, e, bs, sorted_ids, experts, total, expert_map)
        return sorted_ids, experts, total

    return align


def load_sglang_small(path):
    """SGLang's tiny-batch Triton align, imported without the package's client initializer."""
    if "sglang" not in sys.modules:
        package = types.ModuleType("sglang")
        package.__path__ = [os.path.join(path, "sglang")]
        package.__spec__ = importlib.machinery.ModuleSpec("sglang", None, is_package=True)
        sys.modules["sglang"] = package
    from sglang.kernels.ops.moe.moe_align_small_numel import (
        SMALL_NUMEL_LIMIT,
        moe_align_small_numel,
    )

    return moe_align_small_numel, SMALL_NUMEL_LIMIT


def sgl_buffers(ids, e, bs):
    """SGLang wrapper's buffers for E + 1 buckets."""
    n = ids.numel()
    cap = n * bs if n < e + 1 else n + (e + 1) * (bs - 1)
    cap = (cap + 3) // 4 * 4
    opts = {"dtype": torch.int32, "device": ids.device}
    return (
        torch.empty(cap, **opts),
        torch.empty(-(-cap // bs), **opts),
        torch.empty(1, **opts),
        torch.empty(e + 2, **opts),
    )


def reference(ids, e, bs, expert_map):
    """Per-route local expert (-1 dropped) -> sorted multiset of (expert, route) and the total."""
    flat = ids.view(-1).long()
    local = expert_map.long()[flat.clamp(min=0)].where(flat >= 0, -1)
    counts = torch.bincount(local[local >= 0], minlength=e)
    total = int(((counts + bs - 1) // bs * bs).sum())
    keep = (local >= 0).nonzero().view(-1)
    return total, (local[keep] * 2**32 + keep).sort().values


def same_routes(out, ref, bs, n):
    total, key = ref
    got = int(out[2].view(-1)[0])
    if got != total:
        return False
    rows = out[0][:total].long()
    blocks = out[1][: total // bs].long().repeat_interleave(bs)
    real = rows < n
    return torch.equal((blocks[real] * 2**32 + rows[real]).sort().values, key)


def row(e, t, k, bs, ep, vllm, sgl_kernel, small):
    gen = torch.Generator(device="cuda").manual_seed(e * 100003 + t * 31 + k * 7 + bs)
    ids = torch.rand(t, e, generator=gen, device="cuda").topk(k, 1).indices.int()
    n = ids.numel()
    if ep == 1:
        e_local = e
        gmap = torch.arange(e, dtype=torch.int32, device="cuda")

        def auto():
            return moe_align_block_size(ids, bs, e)

        def triton():
            return align_block_size(ids, bs, e, invalid_bucket=True)

        def cute():
            return moe_align_block_size_local(
                ids, bs, gmap, e, capacity_experts=e + 1, identity=True, invalid_bucket=True
            )

        cute_runs = n <= 4096
        sgl_ids, vllm_args = ids, (ids, bs, e)
    else:  # rank 0 of EP `ep`: the first E / ep experts are local
        e_local = e // ep
        gmap = torch.full((e,), -1, dtype=torch.int32, device="cuda")
        gmap[:e_local] = torch.arange(e_local, dtype=torch.int32, device="cuda")

        def auto():
            return moe_align_block_size_local(ids, bs, gmap, e_local)

        def triton():
            return align_block_size(ids, bs, e_local, gmap)

        cute, cute_runs = None, False  # auto is CuTe here
        sgl_ids = gmap[ids.long()].int()
        vllm_args = (ids, bs, e, gmap)
    out = auto()
    provider = (
        "triton"
        if out[0].untyped_storage().data_ptr() == out[1].untyped_storage().data_ptr()
        else "cute"
    )
    ref = reference(ids, e_local, bs, gmap)
    correct = same_routes(out, ref, bs, n) and same_routes(triton(), ref, bs, n)
    sbuf = sgl_buffers(sgl_ids, e_local, bs)
    market = {
        "sgl_kernel": lambda: sgl_kernel.moe_align_block_size(
            sgl_ids, e_local + 1, bs, *sbuf, True
        ),
        "vllm": lambda: vllm(*vllm_args),
    }
    if small is not None and n <= small[1] and e_local + 1 > 64:
        market["sgl_small"] = lambda: small[0](sgl_ids, e_local + 1, bs, *sbuf[:3])
    a = timed(auto)
    us = {name: timed(fn) for name, fn in market.items()}
    ours = {"triton": timed(triton)}
    if cute_runs:
        ours["cute"] = timed(cute)
    b = timed(auto)
    mine = (a + b) / 2
    if provider == "cute":
        ours.setdefault("cute", mine)
    best = min(us, key=us.get)
    return {
        "E": e, "T": t, "k": k, "block": bs, "ep": ep, "routes": n, "provider": provider,
        "correct": bool(correct), "ours_us": round(mine, 3), "aba_spread": round(abs(a - b) / mine, 3),
        "triton_us": round(ours["triton"], 3),
        "cute_us": None if ours.get("cute") is None else round(ours["cute"], 3),
        "market_us": {kk: round(v, 3) for kk, v in us.items()},
        "best": best, "speedup": round(us[best] / mine, 3),
    }  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--t", type=int, nargs="+", default=list(TS))
    p.add_argument("--k", type=int, nargs="+", default=[6, 8])
    p.add_argument("--block", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--ep", type=int, nargs="+", default=[1, 2])
    p.add_argument("--vllm-vendor", required=True)
    p.add_argument("--sglang-path")
    args = p.parse_args()
    import sgl_kernel

    vllm = load_vllm(args.vllm_vendor)
    small = None
    if args.sglang_path:
        try:
            small = load_sglang_small(args.sglang_path)
        except ImportError as exc:
            print(json.dumps({"sgl_small": f"unavailable: {exc}"}), flush=True)
    print(json.dumps({"device": torch.cuda.get_device_name()}), flush=True)
    for ep in args.ep:
        for e in args.e:
            for k in args.k:
                for bs in args.block:
                    for t in args.t:
                        print(json.dumps(row(e, t, k, bs, ep, vllm, sgl_kernel, small)), flush=True)


if __name__ == "__main__":
    main()
