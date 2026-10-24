# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT provider rows: CuTe vs our FA2 vs host-planned FlashInfer (SGLang's form).

One construction per (geometry, provider); runtime lengths vary. Times are CUDA-graph
replays of run() (prepare outside the graph); precision is against an FP32 reference.

usage: python -m benchmark.kernel_tuning.draft_provider_rows GEO [GEO ...]
GEO = Hq/Hkv/D/Q/mask/B/C/L1+L2+...   mask: n (full) or c (causal); L = keys per request.
Prints one "ROW {json}" line per (geometry, L). On GB10 the process pins to the X925 cores.
"""

import json
import os
import platform
import sys

import torch

from ctikernels.attention.inference import DraftAttention

PROVIDERS = ("cute", "flashinfer")


def pin_x925():
    if platform.machine() != "aarch64":
        return
    cpus, current = [], None
    with open("/proc/cpuinfo") as info:
        for line in info:
            if line.startswith("processor"):
                current = int(line.split(":")[1])
            if line.startswith("CPU part") and line.split(":")[1].strip() == "0xd85":
                cpus.append(current)
    if cpus:
        os.sched_setaffinity(0, cpus)


def graph_us(fn, calls=20, replays=7):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return sorted(times)[len(times) // 2]


def reference(q, k, v, length, hq, hk, causal):
    """q [Q,Hq,D], k/v [L,Hkv,D] -> FP32 [Q,Hq,D], bottom-right aligned."""
    query = q.shape[0]
    qf = q.float().view(query, hk, hq // hk, -1)
    scores = torch.einsum("qhgd,khd->hgqk", qf, k.float()) * q.shape[-1] ** -0.5
    if causal:
        rows = torch.arange(query, device=q.device)[:, None] + (length - query)
        keys = torch.arange(length, device=q.device)[None, :]
        scores = scores.masked_fill(keys > rows, float("-inf"))
    return torch.einsum("hgqk,khd->qhgd", torch.softmax(scores, -1), v.float()).reshape(
        query, hq, -1
    )


def precision(out, ref):
    exact = (out.float() == ref.to(torch.bfloat16).float()).float().mean().item()
    error = out.float() - ref
    return {
        "exact": round(exact, 4),
        "rms": float(error.square().mean().sqrt()),
        "maxabs": float(error.abs().max()),
    }


def construct(geometry, metadata, provider):
    device, batch, capacity = geometry["device"], geometry["batch_size"], geometry["max_kv_tokens"]
    sizes = DraftAttention.workspace_size(**geometry, provider=provider)
    workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]

    def ints(count):
        return torch.zeros(count, dtype=torch.int32, device=device)

    pages = ints(batch * capacity)
    if not DraftAttention.requires_page_indices(**geometry, provider=provider):
        pages = None
    keys = ("query_length", "num_qo_heads", "num_kv_heads", "head_dim", "page_size", "dtype")
    return DraftAttention(
        *workspaces,
        **metadata,
        query_indptr=ints(batch + 1),
        page_indptr=ints(batch + 1),
        page_indices=pages,
        last_page_lengths=ints(batch),
        causal=geometry["causal"],
        window=None,
        provider=provider,
        **{key: geometry[key] for key in keys},
    )


def run_geometry(spec):
    hq, hk, dim, query, mask, batch, capacity, lengths = spec.split("/")
    hq, hk, dim, query, batch, capacity = map(int, (hq, hk, dim, query, batch, capacity))
    lengths = [int(x) for x in lengths.split("+")]
    causal, device = mask == "c", torch.device("cuda")
    torch.manual_seed(1)
    k = torch.randn(batch * capacity + 8, 1, hk, dim, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    q = torch.randn(batch * query, hq, dim, dtype=torch.bfloat16, device=device)
    geometry = {
        "batch_size": batch,
        "query_length": query,
        "max_kv_tokens": capacity,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": None,
        "device": device,
    }
    metadata = {
        "request_indices": torch.arange(batch, dtype=torch.int32, device=device),
        "positions": torch.zeros(batch * query, dtype=torch.int64, device=device),
        "token_indptr": torch.zeros(batch + 1, dtype=torch.int32, device=device),
        "token_indices": torch.zeros(batch * capacity, dtype=torch.int32, device=device),
    }
    ops = {provider: construct(geometry, metadata, provider) for provider in PROVIDERS}
    import flashinfer

    host = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device=device), "NHD", backend="fa2"
    )
    for length in lengths:
        # Request b owns cache rows [b*C, b*C+L).
        indices = torch.cat(
            [
                torch.arange(b * capacity, b * capacity + length, dtype=torch.int32)
                for b in range(batch)
            ]
        ).to(device)
        metadata["token_indptr"].copy_(torch.arange(batch + 1, dtype=torch.int32) * length)
        metadata["token_indices"][: batch * length].copy_(indices)
        metadata["positions"].copy_(torch.arange(length - query, length).repeat(batch))
        ref = torch.cat(
            [
                reference(
                    q[b * query : (b + 1) * query],
                    k[b * capacity : b * capacity + length, 0],
                    v[b * capacity : b * capacity + length, 0],
                    length,
                    hq,
                    hk,
                    causal,
                )
                for b in range(batch)
            ]
        )
        row = {"geo": spec.rsplit("/", 1)[0], "L": length, "gpu": torch.cuda.get_device_name()}
        for provider, op in ops.items():
            out = torch.empty_like(q)
            op.prepare()
            op.run(q=q, paged_kv_cache=(k, v), out=out)
            torch.cuda.synchronize()
            row[provider] = {
                "provider": op.provider,
                "run_us": round(
                    graph_us(lambda op=op, out=out: op.run(q=q, paged_kv_cache=(k, v), out=out)), 2
                ),
                "prep_us": round(graph_us(op.prepare), 2),
                **precision(out, ref),
            }
        host.plan(
            (torch.arange(batch + 1, dtype=torch.int32) * query).to(device),
            (torch.arange(batch + 1, dtype=torch.int32) * length).to(device),
            indices,
            torch.ones(batch, dtype=torch.int32, device=device),
            hq,
            hk,
            dim,
            1,
            causal=causal,
            q_data_type=torch.bfloat16,
        )
        out = torch.empty_like(q)
        host.run(q, (k, v), out=out)
        torch.cuda.synchronize()
        row["fi_hostplan"] = {
            "run_us": round(graph_us(lambda out=out: host.run(q, (k, v), out=out)), 2),
            **precision(out, ref),
        }
        print("ROW " + json.dumps(row), flush=True)


def main(specs):
    pin_x925()
    print("GPU", torch.cuda.get_device_name(), flush=True)
    for spec in specs:
        run_geometry(spec)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
