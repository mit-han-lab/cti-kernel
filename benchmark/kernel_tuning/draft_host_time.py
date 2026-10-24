# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Host time of DraftAttention.run in the engine's mixed-step call pattern.

Mirrors the engine's fixed-query rows of a DFlash mixed step: one operation per
(batch size, mask), prepare() once per step, then one run() per layer on fresh
views (q/out rows after the paged rows, the layer's K/V pool viewed as pages of
one token). Arms are name=path (python/ctikernels under path), imported as
isolated packages (draft_window_ab.load_arm) so trees run side by side on the
same buffers. Per arm and batch size it reports the per-call host time of
run() (perf_counter_ns around the call only, launches queued; the step is
synchronized outside the timed region) and the per-step sum over the layers.
It also checks that every arm writes bit-identical outputs for every layer,
eagerly and replayed from a CUDA graph of prepare() plus all layers.

JSONL on stdout: one "host" record per (arm, batch size, provider) and one
"bitwise" record per (batch size, provider).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def emit(value):
    print(json.dumps(value), flush=True)


def build(DraftAttention, *, batch, args, provider, device, torch):
    q_len, capacity = args.query, args.capacity
    hq, hk, d = args.heads
    dtype = torch.bfloat16
    lengths = [min(capacity, args.context + q_len * (i + 1)) for i in range(batch)]
    # Request b's keys are pool rows [b*C, b*C + L_b), packed at the front of token_indices.
    token_indices = torch.zeros(batch * capacity, dtype=torch.int32, device=device)
    packed = torch.cat(
        [
            torch.arange(b * capacity, b * capacity + n, dtype=torch.int32, device=device)
            for b, n in enumerate(lengths)
        ]
    )
    token_indices[: packed.numel()] = packed
    token_indptr = torch.tensor(
        [0] + [sum(lengths[: i + 1]) for i in range(batch)], dtype=torch.int32, device=device
    )
    positions = torch.cat(
        [torch.arange(n - q_len, n, dtype=torch.int64, device=device) for n in lengths]
    )
    request = torch.arange(batch, dtype=torch.int32, device=device)
    csr = torch.zeros(2 * (batch + 1) + batch + batch * capacity, dtype=torch.int32, device=device)
    query_indptr, page_indptr, last, pages = csr.split(
        [batch + 1, batch + 1, batch, batch * capacity]
    )
    geometry = {
        "query_length": q_len,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": d,
        "page_size": 1,
        "dtype": dtype,
        "causal": args.causal,
        "window": args.window,
    }
    sizes = DraftAttention.workspace_size(
        batch_size=batch, max_kv_tokens=capacity, device=device, provider=provider, **geometry
    )
    floating = torch.zeros(sizes[0] + 256, dtype=torch.uint8, device=device)
    integer = torch.zeros(sizes[1] + 256, dtype=torch.uint8, device=device)
    return DraftAttention(
        floating[: sizes[0]] if sizes[0] else floating[:256],
        integer[: sizes[1]] if sizes[1] else integer[:256],
        request_indices=request,
        positions=positions,
        token_indptr=token_indptr,
        token_indices=token_indices,
        query_indptr=query_indptr,
        page_indptr=page_indptr,
        page_indices=pages,
        last_page_lengths=last,
        provider=provider,
        **geometry,
    )


def step(op, q_rows, out_rows, caches, split, timings=None):
    op.prepare()
    clock = time.perf_counter_ns
    for k_cache, v_cache, out in zip(caches[0], caches[1], out_rows, strict=True):
        q = q_rows[split:]
        pages = (
            k_cache.view(-1, 1, k_cache.shape[1], k_cache.shape[2]),
            v_cache.view(-1, 1, v_cache.shape[1], v_cache.shape[2]),
        )
        o = out[split:]
        start = clock()
        op.run(q=q, paged_kv_cache=pages, out=o)
        if timings is not None:
            timings.append((clock() - start) / 1e3)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from draft_window_ab import load_arm, pin_cores

    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="name=path")
    ap.add_argument("--batches", default="1,2,3,4")
    ap.add_argument("--query", type=int, default=16)
    ap.add_argument("--heads", default="32,8,128")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--capacity", type=int, default=8192)
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--split", type=int, default=64, help="paged rows before the fixed rows")
    ap.add_argument("--causal", type=int, default=1)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--providers", default="auto")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    args.heads = tuple(int(x) for x in args.heads.split(","))
    args.causal = bool(args.causal)
    cores = pin_cores()
    device = torch.device("cuda")
    hq, hk, d = args.heads
    arms = [(spec.split("=", 1)[0], load_arm(*spec.split("=", 1))) for spec in args.arms]
    emit(
        {
            "type": "env",
            "device": torch.cuda.get_device_name(),
            "cores": cores,
            "arms": [name for name, _ in arms],
            "torch": torch.__version__,
        }
    )
    gen = torch.Generator(device=device).manual_seed(0)
    max_b = max(int(b) for b in args.batches.split(","))
    pool = max_b * args.capacity
    caches = tuple(
        [
            torch.randn((pool, hk, d), generator=gen, device=device, dtype=torch.bfloat16)
            for _ in range(args.layers)
        ]
        for _ in range(2)
    )
    for provider in args.providers.split(","):
        selected = None if provider == "auto" else provider
        for batch in (int(b) for b in args.batches.split(",")):
            rows = args.split + batch * args.query
            q_rows = torch.randn((rows, hq, d), generator=gen, device=device, dtype=torch.bfloat16)
            outputs = {}
            for name, DraftAttention in arms:
                op = build(
                    DraftAttention,
                    batch=batch,
                    args=args,
                    provider=selected,
                    device=device,
                    torch=torch,
                )
                outs = [
                    torch.zeros((rows, hq, d), device=device, dtype=torch.bfloat16)
                    for _ in range(args.layers)
                ]
                for _ in range(args.warmup):
                    step(op, q_rows, outs, caches, args.split)
                torch.cuda.synchronize()
                timings, per_step = [], []
                for _ in range(args.steps):
                    before = len(timings)
                    step(op, q_rows, outs, caches, args.split, timings)
                    per_step.append(sum(timings[before:]))
                    torch.cuda.synchronize()
                eager = torch.stack([o[args.split :] for o in outs]).clone()
                for o in outs:
                    o.zero_()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step(op, q_rows, outs, caches, args.split)
                torch.cuda.current_stream().wait_stream(stream)
                for o in outs:
                    o.zero_()
                with torch.cuda.graph(graph):
                    step(op, q_rows, outs, caches, args.split)
                graph.replay()
                torch.cuda.synchronize()
                replay = torch.stack([o[args.split :] for o in outs]).clone()
                outputs[name] = (eager, replay)
                emit(
                    {
                        "type": "host",
                        "arm": name,
                        "provider": op.provider,
                        "requested": provider,
                        "B": batch,
                        "Q": args.query,
                        "heads": args.heads,
                        "layers": args.layers,
                        "us_per_call_median": statistics.median(timings),
                        "us_per_call_mean": statistics.fmean(timings),
                        "us_per_call_p90": sorted(timings)[int(0.9 * len(timings))],
                        "ms_per_step_median": statistics.median(per_step) / 1e3,
                        "eager_equals_graph": bool(torch.equal(eager, replay)),
                    }
                )
                del op, graph
            reference = next(iter(outputs.values()))
            emit(
                {
                    "type": "bitwise",
                    "provider": provider,
                    "B": batch,
                    "equal": {
                        name: bool(torch.equal(e, reference[0]) and torch.equal(r, reference[1]))
                        for name, (e, r) in outputs.items()
                    },
                }
            )


if __name__ == "__main__":
    main()
