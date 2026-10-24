# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Row-parallel GEMM + LoRA + all-reduce + residual + RMSNorm across W local GPUs, per call
in CUDA graphs, rounds interleaved A-B-A across arms:

  fused                  our Triton GEMM + LoRA epilogue publishes its tiles; Exchange.reduce
  cublas+lora+cute       the engine's path: cuBLAS, the segmented LoRA delta, the CuTe exit
  cublas+lora+nccl+norm  cuBLAS + LoRA delta, PyNCCL all-reduce, norm.inference.add_rms_norm_
  ours+exchange / ours+lora+cute, and the GEMM-only rows (SOL: gemm_floor_us)

python -m benchmark.collective.exchange --world 2 --tokens 1 4 --k 2048 --n 4096 --rank 16
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import traceback
from datetime import timedelta


def _worker(rank, args, port, queue):
    import torch
    import torch.distributed as dist

    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        world = args.world
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(minutes=10),
        )  # fmt: skip
        from ctikernels import init_pynccl
        from ctikernels.collective.inference import AllReduceAddRMSNorm, Exchange
        from ctikernels.linear._lora_segmented import lora_delta_factors_
        from ctikernels.linear.lora import lora_shrink
        from ctikernels.linear.triton._block_fp8 import gemm
        from ctikernels.norm.inference import add_rms_norm_

        group = dist.group.WORLD
        comm = init_pynccl(tp_rank=rank, tp_size=world, tp_cpu_group=group, max_size_bytes=256)
        xch = Exchange(
            rank=rank, world_size=world, cpu_group=group, device=device,
            max_tokens=max(args.tokens), max_hidden=args.n, communicator=comm,
        )  # fmt: skip
        cute = None
        if args.cute:
            cute = AllReduceAddRMSNorm(
                rank=rank, world_size=world, cpu_group=group, device=device, hidden=args.n,
                max_tokens=max(args.tokens), eps=1e-6, provider="cute",
            )  # fmt: skip
        g = torch.Generator(device=device).manual_seed(1 + rank)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, device=device) * scale).bfloat16()

        # Weights rotate through >= 256 MiB per graph, so every call reads them from DRAM.
        copies = max(1, -(-(256 << 20) // (args.n * args.k * 2)))
        weights = [randn(args.n, args.k, scale=0.02) for _ in range(copies)]
        factors = (randn(args.rank, args.k, scale=0.02), randn(args.n, args.rank, scale=0.02))
        gain = randn(args.n)
        stream = torch.cuda.Stream()
        rows = []
        calls = max(args.calls, copies)

        def make_arms(tokens):
            x = randn(tokens, args.k)
            out = torch.empty(tokens, args.n, dtype=torch.bfloat16, device=device)
            residual = randn(tokens, args.n)
            config = tuple(args.config) if args.config else (16, 64, 1, 4, 3)

            def ours(i, exchange=None, lora=True):
                low = None
                if lora:
                    low = lora_shrink(x, factors, scaling=0.5, output_sizes=(args.n,),
                                      row_parallel=True)  # fmt: skip
                w = weights[i % copies]
                return gemm(x, None, w, None, out, config=config, lora=low, exchange=exchange)

            def cublas(i, lora=True):
                torch.nn.functional.linear(x, weights[i % copies], out=out)
                if lora:
                    lora_delta_factors_(out, x, factors, scaling=0.5, output_sizes=(args.n,),
                                        row_parallel=True)  # fmt: skip

            def fused(i):
                tile = ours(i, xch.epilogue)
                xch.reduce(out, tile=tile, residual=residual, norm=(gain, 1e-6, True))

            def fused_nolora(i):
                tile = ours(i, xch.epilogue, lora=False)
                xch.reduce(out, tile=tile, residual=residual, norm=(gain, 1e-6, True))

            def exchange(i):
                ours(i)
                xch.add_rms_norm(out, residual, gain, 1e-6, True)

            def via_nccl(i):
                cublas(i)
                comm.all_reduce(out, "sum")
                add_rms_norm_(out, residual, gain, 1e-6, zero_centered=True)

            def cublas_cute(i):
                cublas(i)
                cute(out, residual, gain)

            def ours_cute(i):
                ours(i)
                cute(out, residual, gain)

            arms = {
                "fused": fused,
                "fused(no LoRA)": fused_nolora,
                "ours+exchange": exchange,
                "cublas+lora+nccl+norm": via_nccl,
                "ours+lora": ours,
                "ours+lora+publish": lambda i: ours(i, xch.epilogue),
                "ours": lambda i: ours(i, lora=False),
                "cublas+lora": cublas,
                "cublas": lambda i: cublas(i, lora=False),
            }
            if cute is not None:
                arms["cublas+lora+cute"] = cublas_cute
                arms["ours+lora+cute"] = ours_cute
            return arms

        for tokens in args.tokens:
            arms = make_arms(tokens)
            graphs = {}
            for name, call in arms.items():
                with torch.cuda.stream(stream):
                    for i in range(3):
                        call(i)
                stream.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for i in range(calls):
                        call(i)
                graphs[name] = graph
            xch.check()
            times = {name: [] for name in graphs}
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            order = list(graphs)
            for round_ in range(args.rounds):
                # A-B-A: alternate the arm order every round.
                for name in order if round_ % 2 == 0 else order[::-1]:
                    dist.barrier()
                    with torch.cuda.stream(stream):
                        start.record()
                        graphs[name].replay()
                        end.record()
                    end.synchronize()
                    per_rank = [None] * world
                    dist.all_gather_object(per_rank, start.elapsed_time(end) * 1000 / calls)
                    times[name].append(max(per_rank))
            xch.check()
            # The weight-read floor of one rank's GEMM: D = 2NK + 2MK + 2MN bytes at --read-gbs.
            floor = 2 * (args.n * args.k + tokens * (args.k + args.n)) / (args.read_gbs * 1e3)
            row = {
                "tokens": tokens, "k": args.k, "n": args.n, "rank": args.rank, "world": world,
                "transport": xch.transport, "device": torch.cuda.get_device_name(device),
                "gemm_floor_us": floor,
                "us": {name: statistics.median(v) for name, v in times.items()},
            }  # fmt: skip
            rows.append(row)
            del graphs
        dist.barrier()
        xch.close()
        if cute is not None:
            cute.close()
        queue.put((rank, rows, None))
    except Exception:  # noqa: BLE001  (reported to the parent)
        queue.put((rank, None, traceback.format_exc()))


def main():
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1])
    parser.add_argument("--k", type=int, default=2048, help="K per rank")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--config", type=int, nargs="*")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--cute", action="store_true", help="also the CuTe exit (PRO pair)")
    parser.add_argument("--read-gbs", type=float, default=1533.1, help="DRAM read GB/s (SOL)")
    args = parser.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, args, port, queue)) for r in range(args.world)]
    for p in procs:
        p.start()
    results = {r: (rows, err) for r, rows, err in (queue.get(timeout=900) for _ in procs)}
    for p in procs:
        p.join(timeout=60)
    errors = {r: err for r, (_, err) in results.items() if err}
    if errors:
        raise SystemExit("\n".join(f"rank {r}: {e}" for r, e in errors.items()))
    out = os.environ.get("OUT")
    for row in results[0][0]:
        line = json.dumps(row)
        print(line)
        if out:
            with open(os.path.join(out, "exchange.jsonl"), "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
