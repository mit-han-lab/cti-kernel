# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Report-only two-GPU NCCL market shapes; run with python -m benchmark.collective.pynccl."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import traceback
from datetime import timedelta
from pathlib import Path


def _measure(call, *, torch, dist, stream, rounds, calls, graph_mode):
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    dist.barrier()
    graph = None
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                call()
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph.replay()
        stream.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(rounds):
        dist.barrier()
        with torch.cuda.stream(stream):
            start.record()
            if graph is not None:
                graph.replay()
            else:
                for _ in range(calls):
                    call()
            end.record()
        end.synchronize()
        per_rank = [None, None]
        dist.all_gather_object(per_rank, start.elapsed_time(end) * 1000 / calls)
        times.append(max(per_rank))
    return statistics.median(times)


def _worker(rank, port, args, queue):
    import torch
    import torch.distributed as dist

    from ctikernels.pynccl import init_pynccl, nccl_runtime_version

    try:
        torch.set_num_threads(1)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(seconds=120),
        )
        common = {"tp_rank": rank, "tp_size": 2, "tp_cpu_group": dist.group.WORLD}
        direct = init_pynccl(**common)
        probed = init_pynccl(**common, max_size_bytes=1 << 20)
        staged = init_pynccl(**common, max_size_bytes=1 << 20, _force_window=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        report = {
            "nccl": nccl_runtime_version(),
            "device": torch.cuda.get_device_name(),
            "rounds": args.rounds,
            "calls_per_round": args.calls,
            "statistic": "median of per-round maximum rank latency, microseconds per op",
            "probe": {
                "mode": probed.window_mode(),
                "reason": probed.window_reason(),
                "bytes": probed.window_bytes(),
            },
            "forced": {
                "mode": staged.window_mode(),
                "reason": staged.window_reason(),
                "bytes": staged.window_bytes(),
            },
            "raw_in_place": "same measurement as in_place: one raw ncclAllReduce, no copies",
            "rows": [],
        }

        def record(operation, shape, candidates):
            row = {"operation": operation, **shape}
            for graph_mode in (False, True):
                mode = "graph" if graph_mode else "eager"
                row[mode] = {
                    name: _measure(
                        call,
                        torch=torch,
                        dist=dist,
                        stream=stream,
                        rounds=args.rounds,
                        calls=args.calls,
                        graph_mode=graph_mode,
                    )
                    for name, call in candidates.items()
                }
                if operation == "all_reduce":
                    row[mode]["raw_in_place"] = row[mode]["in_place"]
            report["rows"].append(row)

        # Zero is invariant under repeated in-place sums, preventing overflow.
        for tokens in range(1, 5):
            for hidden in (4096, 5120):
                tensor = torch.zeros((tokens, hidden), dtype=torch.bfloat16, device="cuda")
                stream.wait_stream(torch.cuda.current_stream())
                record(
                    "all_reduce",
                    {"T": tokens, "H": hidden, "dtype": "bfloat16"},
                    {
                        "window_forced": lambda t=tensor: staged.all_reduce(t, "sum"),
                        "in_place": lambda t=tensor: direct.all_reduce(t, "sum"),
                    },
                )

        for length in (64, 256, 1024, 4096, 16384, 65536):
            padded = length + (length & 1)
            tensor = torch.zeros(padded, dtype=torch.uint8, device="cuda")
            gathered = torch.empty(2 * padded, dtype=torch.uint8, device="cuda")
            stream.wait_stream(torch.cuda.current_stream())

            def emulated(gathered=gathered, tensor=tensor, length=length):
                direct.all_gather(gathered.view(torch.float16), tensor.view(torch.float16))
                return gathered[:length]  # Root 0's slice, as in the engine emulation.

            record(
                "broadcast",
                {"bytes": length, "root": 0},
                {
                    "native": lambda t=tensor[:length]: direct.broadcast(t, 0),
                    "all_gather_emulation": emulated,
                },
            )

        for tokens in range(1, 5):
            for local_vocab in (75968, 124160):
                for dtype in (torch.bfloat16, torch.float32):
                    tensor = torch.zeros((tokens, local_vocab), dtype=dtype, device="cuda")
                    gathered = torch.empty((2, tokens, local_vocab), dtype=dtype, device="cuda")
                    stream.wait_stream(torch.cuda.current_stream())
                    record(
                        "logits_all_gather",
                        {"T": tokens, "V_local": local_vocab, "dtype": str(dtype)},
                        {
                            "all_gather": lambda dst=gathered, src=tensor: direct.all_gather(
                                dst, src
                            ),
                        },
                    )
        queue.put((rank, report, None))
    except Exception:  # noqa: BLE001 -- send failures to the parent, without terminating peers.
        queue.put((rank, None, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--out", type=Path, help="Optional JSON output; otherwise only stdout")
    args = parser.parse_args()
    if args.rounds < 1 or args.calls < 1:
        parser.error("--rounds and --calls must be positive")
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(rank, port, args, queue)) for rank in range(2)]
    for worker in workers:
        worker.start()
    results = [queue.get() for _ in workers]
    for worker in workers:
        worker.join()
    failures = [f"rank {rank}: {error}" for rank, _, error in results if error]
    failures.extend(f"worker exitcode={worker.exitcode}" for worker in workers if worker.exitcode)
    if failures:
        raise RuntimeError("\n".join(failures))
    report = next(report for rank, report, _ in results if rank == 0)
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
