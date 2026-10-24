# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paired ragged-batch timing of CuTe DRAFT with the spread grid on/off and FA2.

Each row fixes B, Q, heads, mask, capacity and per-request contexts. Arms:
``off`` (CTI_DRAFT_SPREAD=0, per-request grid), ``on`` (default selection) and
``fa2`` (provider="flashinfer"). Every arm captures a graph of 10 run() calls after
one prepare(); each pair replays every arm once in alternating order. JSONL
records hold every pair's per-call microseconds (type: pair) and a summary
(type: row) with medians and the bootstrap 95% CI of the median off/on and fa2/on
ratios (values above one favor ``on``). GB10 timing is pinned to Cortex-X925.
usage: draft_spread_ragged.py [row,...|all] [pairs]
"""

import json
import os
import random
import statistics
import sys

# name: (B, Q, Hq, Hkv, D, causal, window, capacity, contexts)
ROWS = {
    "b3q18_199k": (3, 18, 32, 8, 128, False, None, 241957, [199000, 300, 50]),
    "b3q18_199k_u": (3, 18, 32, 8, 128, False, None, 241957, [199000] * 3),
    "b2q16_full_ragged32k": (2, 16, 32, 8, 128, False, None, 32768, [32752, 100]),
    "b4q16_hd256_ragged32k": (4, 16, 16, 4, 256, True, None, 32768, [32752, 100, 40, 1000]),
    "b8q16_ragged32k": (8, 16, 16, 4, 128, True, None, 32768, [32752] + [200] * 7),
    "b5q16_hd256_ragged32k": (5, 16, 16, 4, 256, True, None, 32768, [32752, 50, 50, 50, 50]),
    "b4q16_full_ragged131k": (4, 16, 32, 8, 128, False, None, 131072, [131056, 2000, 300, 30]),
    "u_b2q16_full_c32k_100": (2, 16, 32, 8, 128, False, None, 32768, [100] * 2),
    "u_b3q16_full_c32k_1000": (3, 16, 32, 8, 128, False, None, 32768, [1000] * 3),
    "u_b4q16_full_c32k_4000": (4, 16, 32, 8, 128, False, None, 32768, [4000] * 4),
    "u_b4q16_hd256_c32k_400": (4, 16, 16, 4, 256, True, None, 32768, [400] * 4),
    "u_b3q16_full_c32k_32k": (3, 16, 32, 8, 128, False, None, 32768, [32752] * 3),
    # hd256 causal full past the B<=4 CuTe cap (Qwen3.5 16/4 and 8/2 VERIFY), C=32768.
    "cap_b8_hd256_ragged": (8, 16, 16, 4, 256, True, None, 32768, [32752] + [200] * 7),
    "cap_b16_hd256_ragged": (16, 16, 16, 4, 256, True, None, 32768, [32752] + [200] * 15),
    "cap_b32_hd256_ragged": (32, 16, 16, 4, 256, True, None, 32768, [32752] + [200] * 31),
    "cap_b16_hd256_mixed": (
        16,
        16,
        16,
        4,
        256,
        True,
        None,
        32768,
        [100, 20000, 300, 5000, 1000, 64, 12000, 700, 2500, 40, 9000, 150, 3000, 450, 16000, 800],
    ),
    "cap_b8_hd256_u512": (8, 16, 16, 4, 256, True, None, 32768, [512] * 8),
    "cap_b16_hd256_u512": (16, 16, 16, 4, 256, True, None, 32768, [512] * 16),
    "cap_b32_hd256_u512": (32, 16, 16, 4, 256, True, None, 32768, [512] * 32),
    "cap_b8_hd256_u8k": (8, 16, 16, 4, 256, True, None, 32768, [8192] * 8),
    "cap_b16_hd256_u8k": (16, 16, 16, 4, 256, True, None, 32768, [8192] * 16),
    "cap_b32_hd256_u2k": (32, 16, 16, 4, 256, True, None, 32768, [2048] * 32),
    "cap_b16_hd256_82_ragged": (16, 16, 8, 2, 256, True, None, 32768, [32752] + [200] * 15),
    "cap_b8_hd256_82_u4k": (8, 16, 8, 2, 256, True, None, 32768, [4096] * 8),
}
ARMS = (("off", "cute", "0"), ("on", "cute", None), ("fa2", "flashinfer", None))


def pin_x925():
    cpus, current = [], None
    with open("/proc/cpuinfo") as info:
        for line in info:
            if line.startswith("processor"):
                current = int(line.split(":")[1])
            if line.startswith("CPU part") and line.split(":")[1].strip() == "0xd85":
                cpus.append(current)
    if cpus:
        os.sched_setaffinity(0, cpus)
    return cpus


def build(row, provider, spread):
    import torch

    from ctikernels.attention.inference import DraftAttention

    batch, query, hq, hk, dim, causal, window, capacity, contexts = row
    if spread is None:
        os.environ.pop("CTI_DRAFT_SPREAD", None)
    else:
        os.environ["CTI_DRAFT_SPREAD"] = spread
    device = "cuda"
    lengths = [context + query for context in contexts]
    generator = torch.Generator(device=device).manual_seed(0)
    keys = torch.randn(
        sum(lengths) + 8, 1, hk, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    values = torch.randn(keys.shape, dtype=torch.bfloat16, device=device, generator=generator)
    q = torch.randn(
        batch * query, hq, dim, dtype=torch.bfloat16, device=device, generator=generator
    )
    token_indptr = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0).to(torch.int32)
    token_indices = torch.zeros(batch * capacity, dtype=torch.int32)
    token_indices[: sum(lengths)] = torch.arange(sum(lengths), dtype=torch.int32)
    positions = torch.cat([torch.arange(n - query, n, dtype=torch.int64) for n in lengths])
    geometry = {
        "query_length": query,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
        "page_size": 1,
        "dtype": torch.bfloat16,
        "causal": causal,
        "window": window,
    }
    sizing = {
        **geometry,
        "batch_size": batch,
        "max_kv_tokens": capacity,
        "device": torch.device(device),
    }
    workspaces = [
        torch.zeros(size, dtype=torch.uint8, device=device)
        for size in DraftAttention.workspace_size(**sizing, provider=provider)
    ]
    pages = DraftAttention.requires_page_indices(**sizing, provider=provider)

    def ints(count):
        return torch.zeros(count, dtype=torch.int32, device=device)

    op = DraftAttention(
        *workspaces,
        request_indices=torch.arange(batch, dtype=torch.int32, device=device),
        positions=positions.to(device),
        token_indptr=token_indptr.to(device),
        token_indices=token_indices.to(device),
        query_indptr=ints(batch + 1),
        page_indptr=ints(batch + 1),
        page_indices=ints(batch * capacity) if pages else None,
        last_page_lengths=ints(batch),
        provider=provider,
        **geometry,
    )
    out = torch.empty_like(q)
    op.prepare()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op.run(q=q, paged_kv_cache=(keys, values), out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(10):
            op.run(q=q, paged_kv_cache=(keys, values), out=out)
    graph.replay()
    torch.cuda.synchronize()
    return op, graph, out, (keys, values, q, workspaces)


def replay_us(graph):
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 100.0


def median_ci(ratios, rng):
    middle = statistics.median(ratios)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return round(middle, 4), round(boot[50], 4), round(boot[1949], 4)


def main():
    import torch

    cores = pin_x925()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(ROWS)
    pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    print(json.dumps({"type": "host", "gpu": torch.cuda.get_device_name(), "x925": cores}))
    rng = random.Random(1)
    for name in names:
        arms = {}
        for arm, provider, spread in ARMS:
            try:
                arms[arm] = build(ROWS[name], provider, spread)
            except Exception as error:  # noqa: BLE001 - keep a receipt per failed arm.
                print(
                    json.dumps(
                        {"type": "error", "row": name, "arm": arm, "error": str(error)[:300]}
                    )
                )
        os.environ.pop("CTI_DRAFT_SPREAD", None)
        times = {arm: [] for arm in arms}
        for index in range(pairs):
            order = list(arms) if index % 2 == 0 else list(reversed(list(arms)))
            sample = {arm: replay_us(arms[arm][1]) for arm in order}
            for arm, value in sample.items():
                times[arm].append(value)
            print(
                json.dumps(
                    {"type": "pair", "row": name, "us": {k: round(v, 3) for k, v in sample.items()}}
                )
            )
        record = {"type": "row", "row": name, "spec": ROWS[name][:8], "contexts": ROWS[name][8]}
        for arm in arms:
            record[arm + "_us"] = round(statistics.median(times[arm]), 2)
        if "on" in arms:
            implementation = arms["on"][0]._implementation
            record["on_spread"] = getattr(implementation, "spread", None)
            record["on_splits"] = getattr(implementation, "splits", None)
            record["on_merge"] = arms["on"][0].merge
            for other in ("off", "fa2"):
                if other in arms:
                    ratios = [a / b for a, b in zip(times[other], times["on"], strict=True)]
                    record[other + "/on"] = median_ci(ratios, rng)
                    difference = arms[other][2].float() - arms["on"][2].float()
                    record[other + "_max_abs_diff"] = float(difference.abs().max())
        print(json.dumps(record), flush=True)
        del arms
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
