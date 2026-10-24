# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CUDA-graph timings of individual CuTe and Triton expert GEMMs.

Examples (run from the project root with PYTHONPATH=python:.):
  python -m benchmark.kernel_tuning.moe_cute_gemm_bench --out out/moe.jsonl
  python -m benchmark.kernel_tuning.moe_cute_gemm_bench --shapes qwen35-ep2 \
      --kinds down --configs 'down:16x64x64s6w4g1x4|64x128x64s3w8g2x4n|128x128x32s3w8g4x2'

Total-T times are sums of measured chunk times, excluding routing/layout work.
Repeated equal-size chunks reuse one measured layout. FLOPs count real local
and shared slots only. The three peak values are measured dense BF16 ceilings.
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch

from ctikernels.moe.cute._provider import KINDS, CuteBackend, overrides
from ctikernels.moe.triton import _kernels as TK
from ctikernels.moe.triton._provider import _blocks, _launch
from ctikernels.moe.triton._slot_layout import build_layout

PEAKS = {"5090": 247.4, "PRO 6000": 466.1, "GB10": 124.0}
SHAPES = {
    "qwen35-ep1": (2048, 512, 512, 256, 1),
    "qwen35-ep2": (2048, 512, 256, 256, 2),
    "qwen35-ep4": (2048, 512, 128, 256, 4),
    "qwen3": (2048, 768, 0, 128, 1),
}
TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


class Inputs:
    def __init__(self, shape, tokens, seed):
        self.h, self.ni, self.nis, global_e, ep = SHAPES[shape]
        self.e, self.tokens = global_e // ep, tokens
        self.kp1 = 9 if self.nis else 8
        self.nmax = max(self.ni, self.nis)
        self.device = torch.device("cuda")
        gen = torch.Generator(device="cuda").manual_seed(seed)

        def rand(*dims, scale=1):
            return (torch.randn(dims, generator=gen, device=self.device) * scale).bfloat16()

        self.x, self.dy = rand(tokens, self.h), rand(tokens, self.h)
        self.w13 = rand(self.e, 2 * self.ni, self.h, scale=self.h**-0.5)
        self.w2 = rand(self.e, self.h, self.ni, scale=self.ni**-0.5)
        self.s13 = rand(2 * self.nis, self.h, scale=self.h**-0.5) if self.nis else None
        self.s2 = rand(self.h, self.nis, scale=max(self.nis, 1) ** -0.5) if self.nis else None
        # Uniform distinct global top-8, with rank-zero local placement and remote masks.
        cpu_gen = torch.Generator().manual_seed(seed)
        sel = torch.rand(tokens, global_e, generator=cpu_gen).topk(8, dim=-1).indices
        self.local_rows = int((sel < self.e).sum())
        self.shared_rows = tokens if self.nis else 0
        self.sel = sel.to(self.device, torch.int32)
        self.live = torch.ones(tokens, dtype=torch.bool, device=self.device)
        self.map = torch.arange(global_e, dtype=torch.int32, device=self.device)
        self.map[self.e :] = -1
        self.ws = (
            torch.rand(tokens, self.kp1, generator=gen, device=self.device).softmax(-1).flatten()
        )

    def buffers(self, bm, bn):
        lay = build_layout(self.sel, self.live, self.e, bm, self.map, has_shared=bool(self.nis))
        rows = lay.sorted_slot.numel()

        def bf(width):
            return torch.randn(rows, width, device=self.device, dtype=torch.bfloat16)

        return {
            "lay": lay,
            "a": bf(self.nmax),
            "gs": bf(self.nmax),
            "us": bf(self.nmax),
            "y": bf(self.h),
            "gg": bf(2 * self.nmax),
            "dx": bf(self.h),
            "dw": torch.empty(rows, self.nmax // bn, device=self.device),
        }

    def flops(self, kind, saved):
        rn = self.local_rows * self.ni + self.shared_rows * self.nis
        factor = (
            4 if kind in ("gate_up", "dx_pair") else 6 if kind == "bwd_slot" and not saved else 2
        )
        return factor * self.h * rn


def cute_launch(p, be, buf, kind, saved):
    lay = buf["lay"]
    kw = {"ni": p.ni, "nis": p.nis}
    if kind == "gate_up":
        return lambda: be.gate_up(
            p.x, p.w13, p.s13, lay, buf["a"], buf["gs"], buf["us"], **kw, kp1=p.kp1, save=saved
        )
    if kind == "down":
        return lambda: be.down(buf["a"], p.w2, p.s2, lay, buf["y"], **kw)
    if kind == "dx_pair":
        return lambda: be.dx_pair(buf["gg"], p.w13, p.s13, lay, buf["dx"], **kw)
    return lambda: be.bwd_slot(
        p.x,
        p.dy,
        p.w13,
        p.s13,
        p.w2,
        p.s2,
        lay,
        p.ws,
        buf["gs"],
        buf["us"],
        buf["gg"],
        buf["dw"],
        **kw,
        kp1=p.kp1,
        saved=saved,
    )


def triton_config(p):
    # Mirror the provider's column divisibility guards, including 64-wide shards.
    blk = _blocks(p.device, p.ni, p.nis or p.ni)
    bn = min(blk["bn"], p.ni, p.nis or p.ni)
    if p.ni % bn or p.nis % bn:
        bn = 64
    bnh = min(blk["bn"], p.h)
    if p.h % bnh:
        bnh = 64
    return blk, bn, bnh


def triton_launch(p, buf, kind, saved):
    blk, bn, bnh = triton_config(p)
    bm, bk = blk["bm"], blk["bk"]
    lay = buf["lay"]
    ss, ee = lay.sorted_slot, lay.block_expert
    s13, s2 = p.s13 if p.s13 is not None else p.w13, p.s2 if p.s2 is not None else p.w2
    common = (p.h, p.ni, p.nis, p.nmax, p.e)
    launch = _launch(blk)
    if kind == "gate_up":
        return lambda: TK._fwd_gate_up_kernel[(lay.num_blocks, p.nmax // bn)](
            p.x,
            p.w13,
            s13,
            ss,
            ee,
            buf["a"],
            buf["gs"],
            buf["us"],
            *common,
            p.kp1,
            saved,
            bm,
            bn,
            min(bk, p.h),
            **launch,
        )
    if kind == "down":
        return lambda: TK._fwd_down_kernel[(lay.num_blocks, p.h // bnh)](
            buf["a"],
            p.w2,
            s2,
            ss,
            ee,
            buf["y"],
            *common,
            bm,
            bnh,
            min(bk, p.ni, p.nis or p.ni),
            **launch,
        )
    if kind == "dx_pair":
        return lambda: TK._bwd_dx_pair_kernel[(lay.num_blocks, p.h // bnh)](
            buf["gg"],
            p.w13,
            s13,
            ss,
            ee,
            buf["dx"],
            *common,
            bm,
            bnh,
            min(bk, p.ni, p.nis or p.ni),
            **launch,
        )
    return lambda: TK._bwd_slot_kernel[(lay.num_blocks, p.nmax // bn)](
        p.x,
        p.dy,
        p.w13,
        s13,
        p.w2,
        s2,
        ss,
        ee,
        p.ws,
        buf["gs"],
        buf["us"],
        buf["gg"],
        buf["dw"],
        *common,
        p.kp1,
        p.nmax // bn,
        saved,
        bm,
        bn,
        min(bk, p.h),
        **launch,
    )


def graph_pair(cute_fn, triton_fn, launches, repeats):
    """Capture back-to-back launches, then interleave timed graph replays."""
    graphs = []
    for fn in (cute_fn, triton_fn):
        for _ in range(3):
            fn()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(launches):
                fn()
        graphs.append(graph)
    torch.cuda.synchronize()
    samples = [[], []]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        for graph in graphs:
            graph.replay()
    for r in range(repeats):
        for i in (0, 1) if r % 2 == 0 else (1, 0):
            start.record()
            graphs[i].replay()
            end.record()
            end.synchronize()
            samples[i].append(start.elapsed_time(end) * 1000 / launches)
    return tuple(statistics.median(s) for s in samples)


def config_sweep(text, saved="both"):
    """kind:spec|spec,kind:spec|spec; specs accept the optional TMA suffix t."""
    result = {k: [None] for k in KINDS}
    if text:
        for entry in text.split(","):
            kind, specs = entry.split(":", 1)
            if kind not in KINDS:
                raise ValueError(f"Unknown GEMM kind {kind!r}")
            result[kind] = [overrides(f"{kind}:{spec}") for spec in specs.split("|")]
            if kind == "bwd_slot" and saved == "yes":
                # Saved backward has one B operand and can use a larger tile
                # than recompute. Do not apply its override to gate/up parsing.
                for override in result[kind]:
                    override.pop("gate_up", None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--configs", default="")
    parser.add_argument("--saved", choices=("both", "yes", "no"), default="both")
    parser.add_argument("--launches", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    args = parser.parse_args()
    shapes, kinds = args.shapes.split(","), args.kinds.split(",")
    tokens_list = [int(t) for t in args.tokens.split(",")]
    if any(s not in SHAPES for s in shapes) or any(k not in KINDS for k in kinds):
        parser.error("Unknown shape or kind")
    if min(tokens_list + [args.launches, args.repeats]) <= 0:
        parser.error("Tokens, launches and repeats must be positive")
    sweep = config_sweep(args.configs, args.saved)
    name = torch.cuda.get_device_name()
    peak = args.peak_tflops or next((v for k, v in PEAKS.items() if k in name), None)
    output = args.out.open("w") if args.out else None
    try:
        for shape in shapes:
            for total in tokens_list:
                step = args.chunk_tokens
                counts = Counter(min(step, total - start) for start in range(0, total, step))
                # Allocate weights once per chunk geometry, outside capture/timing.
                cases = {c: Inputs(shape, c, args.seed) for c in counts}
                for kind in kinds:
                    modes = (False, True) if kind in ("gate_up", "bwd_slot") else (False,)
                    if kind in ("gate_up", "bwd_slot") and args.saved != "both":
                        modes = (args.saved == "yes",)
                    for saved in modes:
                        for override in sweep[kind]:
                            chunks = []
                            try:
                                for c, multiplicity in counts.items():
                                    p = cases[c]
                                    be = CuteBackend(p.device, p.ni, p.nis, p.h)
                                    if override is not None:
                                        be.override = override
                                    bm = be.block_m(p.device, c, p.ni, p.nis)
                                    bn = p.nmax // be.dw_tiles(
                                        p.device, c, p.ni, p.nis, saved=saved
                                    )
                                    cb = p.buffers(bm, bn)
                                    tb_cfg, tb_bn, tb_bnh = triton_config(p)
                                    tb = p.buffers(tb_cfg["bm"], tb_bn)
                                    cu, tr = graph_pair(
                                        cute_launch(p, be, cb, kind, saved),
                                        triton_launch(p, tb, kind, saved),
                                        args.launches,
                                        args.repeats,
                                    )
                                    chunks.append(
                                        {
                                            "tokens": c,
                                            "multiplicity": multiplicity,
                                            "cute_us": cu,
                                            "triton_us": tr,
                                            "cute_config": str(be._config(kind, cb["lay"], saved)),
                                            "cute_shared_bytes": be._config(
                                                kind, cb["lay"], saved
                                            ).shared_bytes(kind, saved),
                                            "cute_two_ctas_smem": be._config(
                                                kind, cb["lay"], saved
                                            ).two_ctas_smem(kind, saved),
                                            "triton_config": {
                                                **tb_cfg,
                                                "bn": tb_bn
                                                if kind in ("gate_up", "bwd_slot")
                                                else tb_bnh,
                                            },
                                            "local_rows": p.local_rows,
                                            "shared_rows": p.shared_rows,
                                            "flops": p.flops(kind, saved),
                                        }
                                    )
                                    del cb, tb
                            except ValueError as exc:
                                # An invalid sweep entry (shape or shared-memory limits) is skipped.
                                print(json.dumps({"kind": kind, "skipped": str(exc)}), flush=True)
                                continue
                            cu_total = sum(x["cute_us"] * x["multiplicity"] for x in chunks)
                            tr_total = sum(x["triton_us"] * x["multiplicity"] for x in chunks)
                            flops = sum(x["flops"] * x["multiplicity"] for x in chunks)
                            cu_tf, tr_tf = flops / cu_total / 1e6, flops / tr_total / 1e6
                            row = {
                                "shape": shape,
                                "tokens": total,
                                "kind": kind,
                                "saved": saved,
                                "device": name,
                                "peak_tflops": peak,
                                "chunks": chunks,
                                "cute_per_chunk_us": chunks[0]["cute_us"],
                                "triton_per_chunk_us": chunks[0]["triton_us"],
                                "cute_per_T_us": cu_total,
                                "triton_per_T_us": tr_total,
                                "total_time_method": "sum_of_chunk_medians",
                                "speedup": tr_total / cu_total,
                                "cute_tflops": cu_tf,
                                "triton_tflops": tr_tf,
                                "cute_peak_percent": 100 * cu_tf / peak if peak else None,
                                "triton_peak_percent": 100 * tr_tf / peak if peak else None,
                                "launches": args.launches,
                                "replays": args.repeats,
                            }
                            line = json.dumps(row, allow_nan=False)
                            print(line, flush=True)
                            if output:
                                output.write(line + "\n")
                                output.flush()
                del cases
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
