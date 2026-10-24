# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Engine per-layer DFlash attention latency, with JSONL receipts on stdout.

Use n distinct layer K/V buffers rotating over >= 2x L2 (counting allocated BF16
K/V bytes). n is at least the shape layer count: 36 for q3_target, 5 for draft_noncausal,
or 8 for q35_target (Qwen3.5 hd256 verify). Capture one
CUDA graph of n calls, time replays with CUDA events, and report per-call latency
as elapsed / n: median of 25 replays, plus the minimum. kvlen = ctx + Q; physical
K/V storage is rounded to 256 tokens. Q and output buffers are shared across
layers, as are immutable metadata and workspace. TF32 is disabled for the
independent float32 dense GQA correctness check (atol = rtol = 2e-2).

q3_target and draft_noncausal use Hq/Hkv = 32/8 and D = 128; q35_target
uses 16/4 and D = 256 with causal full attention. CuTe supports D = 128/256; q35_target explicitly opts into CuTe. q3_target uses DraftAttention's public causal-full
mask (causal=True, window=None); draft_noncausal is full non-causal attention.
draft_fi uses DraftAttention(provider="flashinfer") at D=128 or D=256, with
the same capacity and prepare-once handling as cute.
auto uses DraftAttention(provider=None), recording the selected dispatch and tuning.
cute runs prepared metadata; cute_fused folds prepare into only the first of n calls. vllm_fa2 (vLLM's vllm_flash_attn dispatch) uses P=16;
flash_attn (the flash-attn 2.8.3 package, ``flash_attn_with_kvcache``) uses
P=256, its smallest paged block; the other providers use P=1. flash_attn emits
SKIP when the package is absent. fi_fixed uses the device's default minimum
Split-KV chunk; fi_fixed_s16/s32/s64 override it to 16/32/64 tokens, while the
CTA bound still selects larger chunks for long contexts. All fi_fixed providers
prepare the device schedule once before timing.

sglang reproduces SGLang 72730d6's FlashInfer 0.7.1 fa2 dispatch, using a
256 MiB workspace and one host plan with exact lengths before graph capture.
--sglang-flashinfer PATH supplies a directory containing that flashinfer package;
without it sglang emits SKIP. The worker records and checks version 0.7.1.
--providers selects a comma list (default: all). With --sglang-flashinfer, each
combo runs separate fresh workers for the selected non-sglang providers and
sglang alone, so FlashInfer 0.6.11 and 0.7.1 never share a process.

Run from the repository root with the desired ctikernels wheel installed::

    python -m benchmark.kernel_tuning.dflash_draft_layers --query 16 --capacity 32768
    python -m benchmark.kernel_tuning.dflash_draft_layers --combos '[{"Q":4},{"Q":9,"CAP":32768}]'

--vllm-fa2 supplies the vLLM _vllm_fa2_C.abi3.so directly; when absent that
provider emits SKIP with a reason. --combos is a JSON list of Q/CAP and CuTe
tuning overrides (MERGE is a string; other values are integers; see --help).
Each combo runs in a fresh subprocess, before
CUDA or provider initialization; a default run is one combo. GB10 timing is
pinned to available Cortex-X925 cores. No CUDA imports occur until a worker runs.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

SHAPES = {
    "q3_target": {"hq": 32, "hk": 8, "d": 128, "causal": True, "layers": 36},
    "draft_noncausal": {"hq": 32, "hk": 8, "d": 128, "causal": False, "layers": 5},
    "q35_target": {"hq": 16, "hk": 4, "d": 256, "causal": True, "layers": 8},
    # Qwen3.5-35B-A3B target VERIFY (GQA 8) and its TP2 shard.
    "q35moe_target": {"hq": 16, "hk": 2, "d": 256, "causal": True, "layers": 10},
    "q35moe_tp2": {"hq": 8, "hk": 1, "d": 256, "causal": True, "layers": 10},
    # K-Q7: Qwen3-30B-A3B DFlash draft geometry (GQA 8, D=128), causal layers.
    "q3moe_draft": {"hq": 32, "hk": 4, "d": 128, "causal": True, "layers": 5},
}
MIN_CHUNKS = {
    "fi_fixed": None,
    "fi_fixed_s16": 16,
    "fi_fixed_s32": 32,
    "fi_fixed_s64": 64,
    "fi_fixed_s128": 128,
}
COMPARATOR_LABELS = {
    "auto": "ctikernels DraftAttention automatic dispatch",
    "cute": "ctikernels DraftAttention CuTe",
    "cute_fused": "ctikernels DraftAttention CuTe",
    "draft_fi": "ctikernels DraftAttention FlashInfer",
    "table": "ctikernels CacheTableDraftAttention automatic dispatch",
    "table_row": "ctikernels CacheTableDraftAttention automatic dispatch, row metadata",
    "vllm_fa2": "vllm_flash_attn (vLLM 0.31.0 dispatch, P=16)",
    "flash_attn": "flash_attn 2.8.3 (P=256)",
    **dict.fromkeys(MIN_CHUNKS, "ctikernels FlashInferFixedQueryAttention"),
    "sglang": "SGLang 72730d6 dispatch: FlashInfer 0.7.1 fa2",
}
# Preserve the original default sweep; automatic dispatch is opt-in here.
PROVIDERS = tuple(p for p in COMPARATOR_LABELS if p not in ("auto", "table", "table_row"))
TUNING_KEYS = (
    "ROWS",
    "MIN_CHUNK",
    "SPLITS",
    "TARGET_CTAS",
    "CHUNK_CAP",
    "KEY_TILE",
    "KEY_GROUPS",
    "STAGES",
    "PIECES",
    "INLINE",
    "MERGE",
    "COOP_PDL",
    "SHORT",
)


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def pin_cores():
    """Restrict Linux affinity to available Cortex-X925 cores (CPU part 0xd85)."""
    if not hasattr(os, "sched_getaffinity"):
        return {"x925_cores": [], "cores": [], "affinity_status": "unavailable"}
    cores = []
    status = "unchanged"
    try:
        for stanza in re.split(r"\n\s*\n", Path("/proc/cpuinfo").read_text().strip()):
            fields = dict(line.split(":", 1) for line in stanza.splitlines() if ":" in line)
            fields = {k.strip().lower(): v.strip() for k, v in fields.items()}
            if int(fields.get("cpu part", "0"), 16) == 0xD85:
                cores.append(int(fields["processor"]))
        cores = sorted(set(cores) & os.sched_getaffinity(0))
        if cores:
            os.sched_setaffinity(0, cores)
            status = "pinned"
    except (OSError, ValueError, KeyError):
        status = "failed"
    return {
        "x925_cores": cores,
        "cores": sorted(os.sched_getaffinity(0)),
        "affinity_status": status,
    }


@functools.lru_cache(None)
def load_vllm(path):
    import torch

    torch.ops.load_library(str(path))


def capture_graph(fn, n):
    """Warm up and capture one chain of n distinct layer calls."""
    import torch

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    return g


def time_graph(fn, n):
    import torch

    g = capture_graph(fn, n)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(25):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0 / n)
    del g
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def reference(q, k, v, B, lpad, kvlen, causal, Q, hq, hk, d):
    import torch

    # Float32 dense GQA, with TF32 disabled in main; no provider reference.
    qq = q.float().view(B, Q, hk, hq // hk, d).permute(0, 2, 3, 1, 4)
    kk = k.view(B, lpad, hk, d)[:, :kvlen].float().permute(0, 2, 3, 1)
    vv = v.view(B, lpad, hk, d)[:, :kvlen].float().permute(0, 2, 1, 3)
    scores = (qq @ kk.unsqueeze(2)) * (d**-0.5)
    if causal:
        mask = torch.arange(kvlen, device=q.device)[None, :] > (
            kvlen - Q + torch.arange(Q, device=q.device)[:, None]
        )
        scores.masked_fill_(mask, float("-inf"))
    out = scores.softmax(-1) @ vv.unsqueeze(2)
    return out.permute(0, 3, 1, 2, 4).reshape(B * Q, hq, d)


def build_provider(provider, q, kc, vc, B, ctx, lpad, causal, hq, hk, d, row, args):
    """Return fn(i), output, and resources to retain through graph destruction."""
    import torch

    resources = []
    Q = args.query
    dev, kvlen = q.device, ctx + Q
    o = torch.empty_like(q)
    qi_cpu = torch.arange(B + 1, dtype=torch.int32) * Q
    ki_cpu = torch.arange(B + 1, dtype=torch.int32) * kvlen
    slots = torch.arange(B, dtype=torch.int32)
    idx_cpu = (slots[:, None] * lpad + torch.arange(kvlen, dtype=torch.int32)).flatten()
    kp = [k.view(-1, 1, hk, d) for k in kc]
    vp = [v.view(-1, 1, hk, d) for v in vc]
    geometry = {
        "query_length": Q,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": d,
        "page_size": 1,
        "dtype": q.dtype,
        "causal": causal,
    }
    if provider in ("auto", "cute", "cute_fused", "draft_fi"):
        from ctikernels.attention.draft import DraftAttention

        draft_provider = {"auto": None, "draft_fi": "flashinfer"}.get(provider, "cute")
        geometry["window"] = None
        cap = args.capacity or kvlen
        padded = torch.zeros(B * cap, dtype=torch.int32)
        padded[: idx_cpu.numel()] = idx_cpu
        idx_cpu = padded
        sizes = DraftAttention.workspace_size(
            batch_size=B, max_kv_tokens=cap, device=dev, provider=draft_provider, **geometry
        )
        ws = [torch.zeros(size, dtype=torch.uint8, device=dev) for size in sizes]
        op = DraftAttention(
            *ws,
            provider=draft_provider,
            request_indices=slots.to(dev),
            positions=torch.arange(ctx, kvlen, dtype=torch.int64, device=dev).repeat(B),
            token_indptr=ki_cpu.to(dev),
            token_indices=idx_cpu.to(dev),
            query_indptr=torch.empty(B + 1, dtype=torch.int32, device=dev),
            page_indptr=torch.empty(B + 1, dtype=torch.int32, device=dev),
            page_indices=torch.empty(B * cap, dtype=torch.int32, device=dev),
            last_page_lengths=torch.empty(B, dtype=torch.int32, device=dev),
            **geometry,
        )
        row["cute_tuning"] = {
            name: getattr(op._implementation, name)
            for name in (
                "splits",
                "min_chunk",
                "key_tile",
                "short_union",
                "inline_merge",
                "merge_mode",
                "coop_pdl",
                "cooperative_blocks_per_sm",
                "counter_bytes",
            )
            if hasattr(op._implementation, name)
        }
        row["op.provider"] = op.provider
        resources.extend([ws, op])
        op.prepare()

        def fn(i):
            call = op.prepare_and_run if provider == "cute_fused" and i == 0 else op.run
            call(q=q, paged_kv_cache=(kp[i], vp[i]), out=o)

    elif provider in ("table", "table_row"):
        from ctikernels.attention.draft import CacheTableDraftAttention

        # Engine layout: M = C - Q table columns per request, frontier F = ctx, no done rows.
        geometry["window"] = None
        geometry.pop("page_size")
        maximum = (args.capacity or kvlen + Q) - Q
        if maximum < kvlen:
            raise ValueError("table candidate requires capacity >= ctx + 2Q")
        table = torch.zeros(B, maximum, dtype=torch.int32)
        table[:, :kvlen] = idx_cpu.view(B, kvlen)
        tensors = {
            "request_indices": slots.to(dev),
            "frontier_table": torch.full((B,), ctx, dtype=torch.int64, device=dev),
            "done_table": torch.zeros(B, dtype=torch.bool, device=dev),
            "cache_index_table": table.to(dev),
        }
        sizes = CacheTableDraftAttention.workspace_size(
            batch_size=B, max_seq_len=maximum, device=dev, **geometry
        )
        ws = [torch.zeros(size, dtype=torch.uint8, device=dev) for size in sizes]
        op = CacheTableDraftAttention(
            *ws,
            guard_start=0,
            max_seq_len=maximum,
            metadata_index="row" if provider == "table_row" else "request",
            **tensors,
            **geometry,
        )
        row["op.provider"] = op.provider
        row["cute_tuning"] = {
            name: getattr(op._implementation, name)
            for name in ("splits", "key_tile", "short_union", "merge_mode")
            if hasattr(op._implementation, name)
        }
        resources.extend([ws, op, tensors])

        def fn(i):
            op.run(q=q, paged_kv_cache=(kp[i], vp[i]), out=o)

    elif provider == "vllm_fa2":
        load_vllm(args.vllm_fa2)
        npages = lpad // 16
        kp = [k.view(-1, 16, hk, d) for k in kc]
        vp = [v.view(-1, 16, hk, d) for v in vc]
        cu_q = qi_cpu.to(dev)
        dummy = torch.zeros_like(cu_q)
        seqused = torch.full((B,), kvlen, dtype=torch.int32, device=dev)
        table = torch.arange(B * npages, dtype=torch.int32, device=dev).reshape(B, npages)

        def fn(i):
            torch.ops._vllm_fa2_C.varlen_fwd(
                q,
                kp[i],
                vp[i],
                o,
                cu_q,
                dummy,
                seqused,
                None,
                table,
                None,
                Q,
                kvlen,
                0.0,
                d**-0.5,
                False,
                causal,
                -1,
                -1,
                0.0,
                False,
                0,
                None,
            )

    elif provider == "flash_attn":
        import flash_attn

        row["flash_attn_version"] = flash_attn.__version__
        npages = lpad // 256
        kp = [k.view(-1, 256, hk, d) for k in kc]
        vp = [v.view(-1, 256, hk, d) for v in vc]
        qb = q.view(B, Q, hq, d)
        seqlens = torch.full((B,), kvlen, dtype=torch.int32, device=dev)
        table = torch.arange(B * npages, dtype=torch.int32, device=dev).reshape(B, npages)

        def fn(i):
            # The package allocates its output; the correctness call copies it.
            result = flash_attn.flash_attn_with_kvcache(
                qb,
                kp[max(i, 0)],
                vp[max(i, 0)],
                cache_seqlens=seqlens,
                block_table=table,
                causal=causal,
            )
            if i < 0:
                o.view(B, Q, hq, d).copy_(result)

    elif provider == "sglang":
        import flashinfer

        ws = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
        csr = {
            "qo_indptr_buf": torch.empty(B + 1, dtype=torch.int32, device=dev),
            "paged_kv_indptr_buf": torch.empty(B + 1, dtype=torch.int32, device=dev),
            "paged_kv_indices_buf": torch.empty(B * kvlen, dtype=torch.int32, device=dev),
            "paged_kv_last_page_len_buf": torch.empty(B, dtype=torch.int32, device=dev),
        }
        op = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", backend="fa2", use_cuda_graph=True, **csr
        )
        resources.extend([ws, csr, op])
        op.plan(
            qo_indptr=qi_cpu,
            paged_kv_indptr=ki_cpu,
            paged_kv_indices=idx_cpu.to(dev),
            paged_kv_last_page_len=torch.ones(B, dtype=torch.int32),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim_qk=d,
            page_size=1,
            causal=causal,
            window_left=-1,
            q_data_type=q.dtype,
            kv_data_type=q.dtype,
            disable_split_kv=False,
            non_blocking=False,
        )

        def fn(i):
            op.run(q=q, paged_kv_cache=(kp[i], vp[i]), out=o)

    else:
        from ctikernels.attention.inference import FlashInferFixedQueryAttention as FQ

        sizes = FQ.workspace_size(
            batch_size=B, query_length=Q, num_qo_heads=hq, num_kv_heads=hk, head_dim=d, device=dev
        )
        ws = [torch.zeros(size, dtype=torch.uint8, device=dev) for size in sizes]
        qi = qi_cpu.to(dev)
        ki = torch.zeros(B + 1, dtype=torch.int32, device=dev)
        kx = torch.zeros(max(B * lpad, B * Q), dtype=torch.int32, device=dev)
        lp = torch.ones(B, dtype=torch.int32, device=dev)
        op = FQ(
            *ws,
            query_indptr=qi,
            kv_indptr=ki,
            kv_indices=kx,
            last_page_lengths=lp,
            min_chunk_tokens=MIN_CHUNKS[provider],
            **geometry,
        )
        resources.extend([ws, op])
        ki.copy_(ki_cpu)
        kx[: idx_cpu.numel()].copy_(idx_cpu)
        op.prepare()

        def fn(i):
            op.run(q=q, paged_kv_cache=(kp[i], vp[i]), out=o)

    return fn, o, resources


def benchmark_provider(provider, q, kc, vc, B, ctx, lpad, causal, hq, hk, d, row, args):
    import torch

    fn, o, resources = build_provider(
        provider, q, kc, vc, B, ctx, lpad, causal, hq, hk, d, row, args
    )
    if provider != "auto":
        row.pop("op.provider", None)
    row["us"], row["us_min"] = time_graph(fn, row["n"])
    fn(-1 if provider == "flash_attn" else 0)
    torch.cuda.synchronize()
    expected = reference(q, kc[0], vc[0], B, lpad, ctx + args.query, causal, args.query, hq, hk, d)
    row["correct"] = "PASS" if torch.allclose(o.float(), expected, atol=2e-2, rtol=2e-2) else "FAIL"


def layer_geometry(name, B, ctx, Q, L2):
    shape = SHAPES[name]
    lpad = math.ceil((ctx + Q) / 256) * 256  # Match attn_bench.py, even for P=16.
    per_layer = 2 * B * lpad * shape["hk"] * shape["d"] * 2
    n = max(shape["layers"], math.ceil(2 * L2 / per_layer))
    # Long contexts: cap the rotating K/V set (bytes, from the environment so paired
    # workers inherit it) while keeping >= 2 layers and >= 2x L2.
    budget = int(os.environ.get("DFLASH_BENCH_MAX_ROTATING_BYTES", "0"))
    if budget:
        n = max(2, math.ceil(2 * L2 / per_layer), min(n, budget // per_layer))
    return {**shape, "lpad": lpad, "n": n, "rotating_kv_bytes": n * per_layer}


def layer_buffers(B, Q, geometry):
    """Identical seeded allocation order for standalone and paired workers."""
    import torch

    hq, hk, d = (geometry[key] for key in ("hq", "hk", "d"))
    n, lpad = geometry["n"], geometry["lpad"]
    torch.manual_seed(0)
    kc = [torch.randn(B * lpad, hk, d, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    vc = [torch.randn(B * lpad, hk, d, device="cuda", dtype=torch.bfloat16) for _ in range(n)]
    q = torch.randn(B * Q, hq, d, device="cuda", dtype=torch.bfloat16)
    return q, kc, vc


def run_case(name, B, ctx, L2, args):
    Q = args.query
    geometry = layer_geometry(name, B, ctx, Q, L2)
    hq, hk, d = (geometry[key] for key in ("hq", "hk", "d"))
    causal, lpad, n = (geometry[key] for key in ("causal", "lpad", "n"))
    rows = [
        {
            "shape": name,
            "B": B,
            "ctx": ctx,
            "provider": p,
            "hq": hq,
            "hk": hk,
            "d": d,
            "min_chunk_tokens": MIN_CHUNKS.get(p),
            "comparator_label": COMPARATOR_LABELS[p],
            "us": None,
            "us_min": None,
            "n": n,
            "Q": Q,
            "capacity": args.capacity or ctx + Q,
            "combo": args.combo_index,
            "kvlen": ctx + Q,
            "rotating_kv_bytes": geometry["rotating_kv_bytes"],
            "correct": "FAIL",
        }
        for p in args.providers
    ]
    passed = True
    ready = []
    for row in rows:
        if row["provider"] == "sglang" and args.sglang_flashinfer is None:
            row.update(status="SKIP", correct="SKIP", reason="--sglang-flashinfer directory absent")
            emit(row)
            continue
        if row["provider"] == "vllm_fa2" and (args.vllm_fa2 is None or not args.vllm_fa2.is_file()):
            row.update(status="SKIP", correct="SKIP", reason="--vllm-fa2 shared library absent")
            emit(row)
            continue
        if row["provider"] == "flash_attn" and importlib.util.find_spec("flash_attn") is None:
            row.update(status="SKIP", correct="SKIP", reason="flash-attn package absent")
            emit(row)
            continue
        if row["provider"] == "sglang":
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    import flashinfer

                row["flashinfer_version"] = flashinfer.__version__
                if flashinfer.__version__ != "0.7.1":
                    row.update(status="ERROR", reason="SGLang requires FlashInfer 0.7.1")
            except Exception as exc:  # noqa: BLE001 - retain a receipt for import failures.
                row.update(
                    status="ERROR", reason="FlashInfer import failed", error_type=type(exc).__name__
                )
            if row.get("status") == "ERROR":
                emit(row)
                passed = False
                continue
        ready.append(row)
    if not ready:
        return passed
    try:
        q, kc, vc = layer_buffers(B, Q, geometry)
    except Exception as exc:  # noqa: BLE001 - record allocation failures for every provider.
        for row in ready:
            row.update(
                status="ERROR",
                reason="layer buffer allocation failed",
                error_type=type(exc).__name__,
            )
            emit(row)
        return False
    for row in ready:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                benchmark_provider(
                    row["provider"], q, kc, vc, B, ctx, lpad, causal, hq, hk, d, row, args
                )
            row["status"] = row["correct"]
            passed = passed and row["correct"] == "PASS"
        except Exception as exc:  # noqa: BLE001 - retain a JSONL receipt and continue providers.
            row.update(
                status="ERROR", reason="provider benchmark failed", error_type=type(exc).__name__
            )
            passed = False
        emit(row)
    return passed


def csv_ints(value):
    try:
        return [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shapes",
        type=lambda value: value.split(","),
        default=list(SHAPES),
        help="comma-separated q3_target,draft_noncausal,q35_target",
    )
    parser.add_argument(
        "--batches",
        type=csv_ints,
        default=[1, 2, 3, 4],
        help="comma-separated batch sizes (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--contexts",
        type=csv_ints,
        default=[100, 300, 1000, 4000],
        help="comma-separated context lengths, excluding Q",
    )
    parser.add_argument("--query", type=int, default=16, help="query length Q, 2..32 (default: 16)")
    parser.add_argument(
        "--capacity", type=int, default=0, help="DraftAttention token capacity C; 0 = exact ctx + Q"
    )
    parser.add_argument(
        "--vllm-fa2",
        type=Path,
        metavar="PATH",
        help="path to _vllm_fa2_C.abi3.so; required for vllm_fa2, otherwise SKIP",
    )
    parser.add_argument(
        "--providers",
        type=lambda value: value.split(","),
        default=list(PROVIDERS),
        help="comma-separated providers (default: all)",
    )
    parser.add_argument(
        "--sglang-flashinfer",
        type=Path,
        metavar="PATH",
        help="directory containing FlashInfer 0.7.1; required for sglang, otherwise SKIP",
    )
    parser.add_argument(
        "--combos",
        metavar="JSON",
        help="fresh subprocess per object in JSON list; Q/CAP override CLI; other keys set CTI_DRAFT_: "
        + ", ".join(TUNING_KEYS)
        + "; MERGE=distributed/tree/legacy; COOP_PDL=0/1",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--combo-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        any(shape not in SHAPES for shape in args.shapes)
        or min(args.batches) < 1
        or min(args.contexts) < 0
    ):
        parser.error("expected known shapes, positive batches, and nonnegative contexts")
    if any(provider not in COMPARATOR_LABELS for provider in args.providers):
        parser.error("expected known providers: " + ",".join(COMPARATOR_LABELS))
    if (
        args.worker
        and args.sglang_flashinfer is not None
        and "sglang" in args.providers
        and args.providers != ["sglang"]
    ):
        parser.error("FlashInfer 0.7.1 requires an sglang-only worker")
    if args.worker and args.combos is not None:
        parser.error("worker cannot launch combos")
    try:
        combos = json.loads(args.combos) if args.combos is not None else [{}]
    except json.JSONDecodeError:
        parser.error("combos must be valid JSON")
    if not isinstance(combos, list) or not combos:
        parser.error("combos must be a nonempty list")
    for combo in combos:
        if not isinstance(combo, dict) or any(
            key not in ("Q", "CAP", *TUNING_KEYS)
            or (
                value not in ("distributed", "tree", "legacy")
                if key == "MERGE"
                else type(value) is not int
            )
            for key, value in combo.items()
        ):
            parser.error(
                "combo keys must be Q, CAP or tuning keys; MERGE is distributed/tree/legacy, others integers"
            )
        query, capacity = combo.get("Q", args.query), combo.get("CAP", args.capacity)
        if not 2 <= query <= 32 or capacity < 0:
            parser.error("expected 2 <= Q <= 32 and C >= 0")
        if capacity and capacity < max(args.contexts) + query:
            parser.error("nonzero capacity must hold ctx + Q for every selected context")
        if any(
            value < (0 if key in ("PIECES", "INLINE", "SHORT", "COOP_PDL") else 1)
            for key, value in combo.items()
            if key in TUNING_KEYS and key != "MERGE"
        ):
            parser.error(
                "tuning values must be positive (PIECES, INLINE, SHORT and COOP_PDL may be zero)"
            )
        if "COOP_PDL" in combo and combo["COOP_PDL"] not in (0, 1):
            parser.error("COOP_PDL must be 0 or 1")
    return args, combos


def run_combos(args, combos):
    failed = False
    for index, combo in enumerate(combos):
        env = os.environ.copy()
        for key in TUNING_KEYS:
            if key in combo:
                env["CTI_DRAFT_" + key] = str(combo[key])
        query, capacity = combo.get("Q", args.query), combo.get("CAP", args.capacity)
        emit(
            {
                "type": "combo",
                "combo": index,
                "Q": query,
                "capacity": capacity,
                "tuning": {key: value for key, value in combo.items() if key in TUNING_KEYS},
            }
        )
        command = [
            sys.executable,
            "-m",
            "benchmark.kernel_tuning.dflash_draft_layers",
            "--worker",
            "--combo-index",
            str(index),
            "--shapes",
            ",".join(args.shapes),
            "--batches",
            ",".join(map(str, args.batches)),
            "--contexts",
            ",".join(map(str, args.contexts)),
            "--query",
            str(query),
            "--capacity",
            str(capacity),
        ]
        if args.vllm_fa2 is not None:
            command.extend(["--vllm-fa2", str(args.vllm_fa2)])
        groups = [args.providers]
        if args.sglang_flashinfer is not None:
            groups = [[p for p in args.providers if p != "sglang"]]
            if "sglang" in args.providers:
                groups.append(["sglang"])
        for providers in groups:
            if not providers:
                continue
            worker_command = [*command, "--providers", ",".join(providers)]
            if providers == ["sglang"] and args.sglang_flashinfer is not None:
                worker_command.extend(["--sglang-flashinfer", str(args.sglang_flashinfer)])
            result = subprocess.run(worker_command, env=env, check=False)
            emit(
                {
                    "type": "combo_result",
                    "combo": index,
                    "providers": providers,
                    "returncode": result.returncode,
                }
            )
            failed |= result.returncode != 0
    return int(failed)


def run_worker(args):
    if args.providers == ["sglang"] and args.sglang_flashinfer is not None:
        sys.path.insert(0, str(args.sglang_flashinfer.resolve()))
    # Imports and provider chatter go to stderr; stdout is JSONL only.
    with contextlib.redirect_stdout(sys.stderr):
        import torch

    affinity = pin_cores()
    if not torch.cuda.is_available():
        emit(
            {"type": "error", "combo": args.combo_index, "reason": "CUDA is required to benchmark"}
        )
        return 1
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    reported_l2 = getattr(props, "L2_cache_size", 0)
    l2 = reported_l2 or (96 << 20)
    emit(
        {
            "type": "host",
            "combo": args.combo_index,
            "device": props.name,
            "sms": props.multi_processor_count,
            "l2_bytes": l2,
            "l2_source": "device" if reported_l2 else "conservative 96 MiB fallback",
            **affinity,
        }
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    passed = True
    with torch.no_grad():
        for name in args.shapes:
            for batch in args.batches:
                for ctx in args.contexts:
                    success = run_case(name, batch, ctx, l2, args)
                    passed = passed and success
                    torch.cuda.empty_cache()
    return 0 if passed else 1


def main(argv=None):
    args, combos = parse_args(argv)
    if not args.worker:
        return run_combos(args, combos)
    try:
        return run_worker(args)
    except Exception as exc:  # noqa: BLE001 - emit a failure receipt without environment contents.
        emit(
            {
                "type": "error",
                "combo": args.combo_index,
                "reason": "worker initialization failed",
                "error_type": type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
