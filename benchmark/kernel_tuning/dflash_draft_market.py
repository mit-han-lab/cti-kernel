# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT market gate; run with python -m benchmark.kernel_tuning.dflash_draft_market.

One isolated subprocess per comparator measures dispatched DraftAttention
(provider=None for auto) as a complete-op CUDA graph replay, against a
fixed-plan/varlen comparator graph. The gated complete op is prepare_and_run(),
the first-layer call (CuTe overlaps prepare on a joined stream branch); the
two-call prepare(); run() composition is also paired and reported. Both use identical BF16 Q/K/V and the same
page size, and must pass the independent dense oracle at atol=rtol=2e-2.
Host planning is outside graphs. After warmup, 20 reference self-pairs freeze
epsilon at nearest-rank p95 (the 19th sorted deviation); alternating candidate/
reference pairs pass when median(candidate/reference) <= 1+epsilon. Wall-clock
intervals synchronize before/after repeated graph calls. These are immutable,
active rows, not engine end-to-end timings. A market claim requires passing
both SGLang and vLLM for every selected row; stock FlashInfer is also reported.

Dispatch provenance (source research; versions below are qualification targets):
* vLLM v0.31.0, db9527a46873454610df6dbedf79a36d6bf1a7f6:
  vllm/v1/attention/backends/flash_attn.py:1450 dispatches DRAFT on sm_120/
  sm_121 to FlashAttentionBackend;
  vllm/v1/attention/backends/fa_utils.py:98 selects FA2 on 12.x.
  vllm/vllm_flash_attn/flash_attn_interface.py:347 calls
  torch.ops._vllm_fa2_C.varlen_fwd from vllm-flash-attn fork
  9cd61de38763d712bb6ce56e2a02cc2bf718c89f. flash_attn.py:355 requires paged
  blocks divisible by 16 (default 16). flash_attn.py:587 maps full to (-1,-1),
  v1 causal sliding to (W-1,0), and DFlash2 noncausal sliding to (W-1,W-1).
  The released FA2 wheel extension contains sm_80 SASS + sm_80 PTX only;
  the driver JIT-compiles PTX for sm_12x. This harness loads that .so directly.
* SGLang 72730d6360a5894b167500d5d1ad82c34c6bb116:
  python/sglang/srt/speculative/dflash_worker_v2.py:2587 selects DRAFT mode.
  python/sglang/srt/layers/attention/flashinfer_backend.py:310, :516, :985
  and runtime flags :1372 dispatch FlashInferAttnBackend to
  BatchPrefillWithPagedKVCacheWrapper(backend="fa2"), page size 1, NHD.
  python/pyproject.toml:42 pins flashinfer_python 0.7.1. flashinfer_backend.py:1970
  retains min(L,W-1) context tokens plus the query block for sliding layers.
  :1134 uses persistent CSR buffers with host planning outside graph replay.
  :2055, :2335 plan window_left=-1 then pass causal/window_left at forward time.
  Deviation: this comparator freezes causal and window_left=W-1 in plan(),
  which is semantically equivalent for these immutable rows (Q <= 32 < W).

Only JSONL goes to stdout. Package locations are basenames; no host identity or
input paths are emitted. Supply the extracted FA2 library and its wheel filename;
the latter can also be read from an adjacent wheel.txt or a single adjacent wheel.
No packages are installed or downloaded by this script. Pure helpers and CLI
parsing do not import torch, FlashInfer, or initialize CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

MASKS = {"full": (False, None), "causal_sliding": (True, 4096), "noncausal_sliding": (False, 2048)}
# Opt-in K-Q7 rows (Qwen3-30B-A3B DFlash drafts): causal full and causal W2048 layers.
EXTRA_MASKS = {"causal_full": (True, None), "causal_sliding_2048": (True, 2048)}
ALL_MASKS = {**MASKS, **EXTRA_MASKS}
COMPARATORS = ("flashinfer", "sglang", "vllm")
REVISIONS = {
    "vllm": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    "vllm_flash_attn": "9cd61de38763d712bb6ce56e2a02cc2bf718c89f",
    "sglang": "72730d6360a5894b167500d5d1ad82c34c6bb116",
}
SGLANG_DEVIATION = (
    "SGLang plans window_left=-1 then supplies causal/window_left at forward time "
    "(flashinfer_backend.py:2055, :2335); comparator freezes causal/window_left in "
    "plan, semantically equivalent for immutable rows with Q < W."
)
PROVENANCE = {
    "flashinfer": "environment FlashInfer BatchPrefillWithPagedKVCacheWrapper fa2, P=1, NHD, full KV",
    "sglang": (
        "python/sglang/srt/speculative/dflash_worker_v2.py:2587 DRAFT -> "
        "python/sglang/srt/layers/attention/flashinfer_backend.py:310,:516,:985,:1372 "
        "FlashInferAttnBackend -> BatchPrefillWithPagedKVCacheWrapper fa2, P=1, NHD; "
        "python/pyproject.toml:42 flashinfer_python==0.7.1; "
        "flashinfer_backend.py:1970 trim context to min(L,W-1); :1134 persistent CSR"
    ),
    "vllm": (
        "vllm/v1/attention/backends/flash_attn.py:1450 DRAFT -> FlashAttentionBackend; "
        "vllm/v1/attention/backends/fa_utils.py:98 sm_120/sm_121 -> FA2; "
        "vllm/vllm_flash_attn/flash_attn_interface.py:347 -> torch.ops._vllm_fa2_C.varlen_fwd; "
        "flash_attn.py:355 P=16 (multiple of 16), :587 full/causal/symmetric windows; "
        "released wheel sm_80 SASS + sm_80 PTX, driver JIT to sm_12x"
    ),
}
SELF_PAIRS = 20
ATOL = RTOL = 2e-2


def parse_shard(value):
    """Zero-based I/N, with stable row selection rows[I::N]."""
    try:
        index, count = map(int, value.split("/"))
        if count > 0 and 0 <= index < count:
            return index, count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("shard must be zero-based I/N with 0 <= I < N")


def parse_heads(value):
    try:
        hq, hk = map(int, value.split("/"))
        if hq > 0 and hk > 0 and hq % hk == 0:
            return hq, hk
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("heads must be positive Hq/Hkv with Hq divisible by Hkv")


def generate_rows(*, queries, batches, contexts, masks, heads, shard=(0, 1)):
    rows = []
    for mask in dict.fromkeys(masks):
        causal, window = ALL_MASKS[mask]
        boundary = window or 4096
        lengths = sorted(set(contexts) | {boundary - 1, boundary, boundary + 1})
        for q, b, context, (hq, hk) in itertools.product(
            dict.fromkeys(queries), dict.fromkeys(batches), lengths, dict.fromkeys(heads)
        ):
            rows.append(
                {
                    "row_id": len(rows),
                    "mask": mask,
                    "causal": causal,
                    "window": window,
                    "query_length": q,
                    "batch_size": b,
                    "context": context,
                    "num_qo_heads": hq,
                    "num_kv_heads": hk,
                    "head_dim": 128,
                }
            )
    index, count = shard
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard")
    return rows[index::count]


def parse_row_key(value):
    """mask:B:Q:context:Hq/Hkv, matching generate_rows fields."""
    try:
        mask, b, q, context, heads = value.split(":")
        if mask in ALL_MASKS:
            return (mask, int(b), int(q), int(context), *parse_heads(heads))
    except (ValueError, argparse.ArgumentTypeError):
        pass
    raise argparse.ArgumentTypeError("row key must be mask:B:Q:context:Hq/Hkv")


def row_key(row):
    return (
        row["mask"],
        row["batch_size"],
        row["query_length"],
        row["context"],
        row["num_qo_heads"],
        row["num_kv_heads"],
    )


def x925_cores(cpuinfo):
    """Parse Linux processor stanzas; Cortex-X925 has CPU part 0xd85."""
    cores = set()
    for stanza in re.split(r"\n\s*\n", cpuinfo.strip()):
        fields = dict(line.split(":", 1) for line in stanza.splitlines() if ":" in line)
        fields = {key.strip().lower(): value.strip() for key, value in fields.items()}
        try:
            if int(fields.get("cpu part", "0"), 16) == 0xD85:
                cores.add(int(fields["processor"]))
        except (KeyError, ValueError):
            continue
    return sorted(cores)


def pin_x925(enabled):
    if not enabled:
        return {"requested": False, "cores": sorted(os.sched_getaffinity(0))}
    cores = set(x925_cores(Path("/proc/cpuinfo").read_text())) & os.sched_getaffinity(0)
    if not cores:
        raise RuntimeError("no allowed Cortex-X925 cores found")
    os.sched_setaffinity(0, cores)
    return {"requested": True, "cpu_part": "0xd85", "cores": sorted(os.sched_getaffinity(0))}


def vllm_window(causal, window):
    if window is None:
        return -1, -1
    return (window - 1, 0) if causal else (window - 1, window - 1)


def paired_timings(candidate, reference, *, pairs, calls):
    from benchmark.kernel_tuning import dflash_draft_attention as protocol

    if pairs == SELF_PAIRS:
        return protocol.paired_timings(candidate, reference, calls=calls)
    # --pairs changes measured candidate pairs, never the 20-pair noise calibration.
    self_pairs = [
        [protocol.measure(reference, calls), protocol.measure(reference, calls)]
        for _ in range(SELF_PAIRS)
    ]
    epsilon = sorted(abs(a / b - 1) for a, b in self_pairs)[18]
    samples = []
    for index in range(pairs):
        if index % 2:
            ref, cand = protocol.measure(reference, calls), protocol.measure(candidate, calls)
        else:
            cand, ref = protocol.measure(candidate, calls), protocol.measure(reference, calls)
        samples.append([cand, ref])
    ratios = [cand / ref for cand, ref in samples]
    median = statistics.median(ratios)
    return {
        "reference_self_pairs_us": self_pairs,
        "epsilon": epsilon,
        "candidate_reference_pairs_us": samples,
        "ratios": ratios,
        "median_ratio": median,
        "candidate_median_us": statistics.median(p[0] for p in samples),
        "reference_median_us": statistics.median(p[1] for p in samples),
        "speed": "PASS" if median <= 1 + epsilon else "FAIL",
        "pairs": pairs,
        "calls_per_interval": calls,
    }


def vllm_artifact(args):
    library = Path(args.vllm_fa2)
    digest = hashlib.sha256()
    with library.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if args.vllm_fa2_sha256 and args.vllm_fa2_sha256.lower() != sha:
        raise ValueError("vLLM FA2 SHA256 mismatch")
    wheel = args.vllm_wheel
    if not wheel:
        receipt = library.parent / "wheel.txt"
        wheels = sorted(library.parent.glob("vllm-*.whl"))
        wheel = receipt.read_text().strip() if receipt.is_file() else None
        if wheel is None and len(wheels) == 1:
            wheel = wheels[0].name
    if (
        not wheel
        or not Path(wheel).name.startswith("vllm-0.31.0-")
        or not Path(wheel).name.endswith(".whl")
    ):
        raise ValueError("provide --vllm-wheel v0.31.0 filename or an adjacent wheel.txt/wheel")
    return {"wheel": Path(wheel).name, "so_sha256": sha, "version": "0.31.0"}


def run_case(kind, row, args):
    import torch

    from benchmark.kernel_tuning.dflash_draft_attention import (
        graph_capture,
        launch_count,
        provider_report,
    )
    from ctikernels.attention.draft import DraftAttention
    from tests._dflash_reference import draft_attention_reference

    b, qlen, context = row["batch_size"], row["query_length"], row["context"]
    hq, hk = row["num_qo_heads"], row["num_kv_heads"]
    causal, window = row["causal"], row["window"]
    page = 16 if kind == "vllm" else 1
    length = context + qlen
    npages = math.ceil(length / page)
    stride = npages * page
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20261014)
    q = torch.randn((b * qlen, hq, 128), device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn((b * npages, page, hk, 128), device=device, dtype=q.dtype, generator=generator)
    v = torch.randn(k.shape, device=device, dtype=q.dtype, generator=generator)
    slots = torch.arange(b, dtype=torch.int32)
    positions = torch.arange(context, length, dtype=torch.int64).repeat(b)
    token_ptr = torch.arange(b + 1, dtype=torch.int32) * length
    token_idx = (slots[:, None] * stride + torch.arange(length, dtype=torch.int32)).flatten()
    geometry = {
        "query_length": qlen,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": 128,
        "page_size": page,
        "dtype": q.dtype,
        "causal": causal,
        "window": window,
    }
    provider = None if args.provider == "auto" else args.provider
    sizes = DraftAttention.workspace_size(
        batch_size=b, max_kv_tokens=length, device=device, provider=provider, **geometry
    )
    workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
    # Engine regime: a provider that does not read the page CSR gets no buffer, so
    # prepare does not write B*C/P page indices it never uses (CuTe).
    needs_pages = DraftAttention.requires_page_indices(
        batch_size=b, max_kv_tokens=length, device=device, provider=provider, **geometry
    )
    op = DraftAttention(
        *workspaces,
        provider=provider,
        request_indices=slots.cuda(),
        positions=positions.cuda(),
        token_indptr=token_ptr.cuda(),
        token_indices=token_idx.cuda(),
        query_indptr=torch.empty(b + 1, dtype=torch.int32, device=device),
        page_indptr=torch.empty(b + 1, dtype=torch.int32, device=device),
        page_indices=(
            torch.empty(b * npages, dtype=torch.int32, device=device) if needs_pages else None
        ),
        last_page_lengths=torch.empty(b, dtype=torch.int32, device=device),
        **geometry,
    )
    out, ref_out = torch.empty_like(q), torch.empty_like(q)

    def candidate():
        op.prepare_and_run(q=q, paged_kv_cache=(k, v), out=out)

    def two_call():
        op.prepare()
        op.run(q=q, paged_kv_cache=(k, v), out=out)

    if kind == "vllm":
        cu_q = (torch.arange(b + 1, dtype=torch.int32) * qlen).cuda()
        dummy = torch.zeros_like(cu_q)
        seqused = torch.full((b,), length, dtype=torch.int32, device=device)
        table = torch.arange(b * npages, dtype=torch.int32, device=device).reshape(b, npages)
        wl, wr = vllm_window(causal, window)

        def comparator():
            torch.ops._vllm_fa2_C.varlen_fwd(
                q,
                k,
                v,
                ref_out,
                cu_q,
                dummy,
                seqused,
                None,
                table,
                None,
                qlen,
                length,
                0.0,
                128**-0.5,
                False,
                causal,
                wl,
                wr,
                0.0,
                False,
                0,
                None,
            )

        info = {"call": "torch.ops._vllm_fa2_C.varlen_fwd", "window_size": [wl, wr]}
    else:
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        start = max(0, context - (window - 1)) if kind == "sglang" and window else 0
        nk = length - start
        indices = (
            (slots[:, None] * stride + torch.arange(start, length, dtype=torch.int32))
            .flatten()
            .cuda()
        )
        workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
        csr = {
            "qo_indptr_buf": torch.empty(b + 1, dtype=torch.int32, device=device),
            "paged_kv_indptr_buf": torch.empty(b + 1, dtype=torch.int32, device=device),
            "paged_kv_indices_buf": torch.empty(b * nk, dtype=torch.int32, device=device),
            "paged_kv_last_page_len_buf": torch.empty(b, dtype=torch.int32, device=device),
        }
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            backend="fa2",
            use_cuda_graph=True,
            **csr,
        )
        wrapper.plan(
            qo_indptr=torch.arange(b + 1, dtype=torch.int32) * qlen,
            paged_kv_indptr=torch.arange(b + 1, dtype=torch.int32) * nk,
            paged_kv_indices=indices,
            paged_kv_last_page_len=torch.ones(b, dtype=torch.int32),
            num_qo_heads=hq,
            num_kv_heads=hk,
            head_dim_qk=128,
            page_size=1,
            causal=causal,
            window_left=-1 if window is None else window - 1,
            q_data_type=q.dtype,
            kv_data_type=q.dtype,
            disable_split_kv=False,
            non_blocking=False,
        )

        def comparator():
            wrapper.run(q=q, paged_kv_cache=(k, v), out=ref_out)

        info = {
            "call": "BatchPrefillWithPagedKVCacheWrapper fa2 fixed-plan",
            "keys": nk,
            "window_left": -1 if window is None else window - 1,
            "planning_deviation": SGLANG_DEVIATION if kind == "sglang" else None,
        }

    # All objects, especially wrapper/workspace/CSR, remain local and alive until
    # both graphs are destroyed. Closures alone do not retain borrowed workspaces.
    candidate()
    comparator()
    expected = draft_attention_reference(
        q,
        k,
        v,
        slots=slots,
        positions=positions,
        token_indptr=token_ptr,
        token_indices=token_idx,
        query_length=qlen,
        causal=causal,
        window=window,
    )

    def correctness():
        result = {}
        for name, actual in (("candidate", out), ("comparator", ref_out)):
            try:
                torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)
                result[name] = "PASS"
            except AssertionError:
                result[name] = "FAIL"
        return result

    eager_correctness = correctness()
    two_call()
    two_call_correctness = correctness()["candidate"]
    cg, _ = graph_capture(candidate)
    tg, _ = graph_capture(two_call)
    rg, _ = graph_capture(comparator)
    try:
        for _ in range(args.warmup):
            cg.replay()
            tg.replay()
            rg.replay()
        timings = paired_timings(cg.replay, rg.replay, pairs=args.pairs, calls=args.calls)
        two = paired_timings(tg.replay, rg.replay, pairs=args.pairs, calls=args.calls)
        counts = launch_count(cg.replay)
        if counts["kernels"] < 1:
            raise RuntimeError("profiler did not observe candidate CUDA kernels")
        graph_correctness = correctness()
        correct = {
            name: "PASS" if eager_correctness[name] == graph_correctness[name] == "PASS" else "FAIL"
            for name in eager_correctness
        }
        passed = timings["speed"] == "PASS" and all(v == "PASS" for v in correct.values())
        return {
            **timings,
            "correctness": correct,
            "eager_correctness": eager_correctness,
            "graph_correctness": graph_correctness,
            "status": "PASS" if passed else "FAIL",
            "comparator_info": info,
            "page_size": page,
            **provider_report(op),
            "page_indices_buffer": needs_pages,
            "kernels_per_call": counts,
            "two_call": {
                "call": "prepare(); run()",
                "correctness": two_call_correctness,
                **{
                    key: two[key]
                    for key in ("epsilon", "median_ratio", "candidate_median_us", "speed")
                },
            },
        }
    finally:
        torch.cuda.synchronize()
        del cg, tg, rg


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def failed_row(error_type, stage):
    # Keep failure records machine-readable without exposing exception paths.
    return {
        "status": "FAIL",
        "speed": "UNMEASURED",
        "error_type": error_type,
        "error_stage": stage,
        "correctness": {"candidate": "UNMEASURED", "comparator": "UNMEASURED"},
        "epsilon": None,
        "median_ratio": None,
        "candidate_median_us": None,
        "reference_median_us": None,
        "reference_self_pairs_us": [],
        "candidate_reference_pairs_us": [],
        "kernels_per_call": None,
        "provider": None,
        "cute_tuning": None,
    }


def worker(kind, rows, args):
    # Set search order before torch, ctikernels, or benchmark-helper imports.
    if kind == "sglang":
        sys.path.insert(0, str(Path(args.sglang_flashinfer).resolve()))
    affinity = {"requested": args.pin_x925, "cores": None}
    versions = dict.fromkeys(("torch", "cuda", "flashinfer", "flashinfer_package_dir", "vllm"))
    setup_error = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import torch

            versions.update(torch=torch.__version__, cuda=torch.version.cuda)
            affinity = pin_x925(args.pin_x925)
            if kind == "vllm":
                versions["vllm"] = vllm_artifact(args)
                torch.ops.load_library(str(args.vllm_fa2))
            else:
                import flashinfer

                package = Path(flashinfer.__file__).resolve().parent
                versions.update(
                    flashinfer=flashinfer.__version__, flashinfer_package_dir=package.name
                )
                if kind == "sglang" and (
                    flashinfer.__version__ != "0.7.1"
                    or package != (Path(args.sglang_flashinfer) / "flashinfer").resolve()
                ):
                    raise ValueError("SGLang requires FlashInfer 0.7.1 from --sglang-flashinfer")
            torch.backends.cuda.matmul.allow_tf32 = False
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        setup_error = type(error).__name__
    source = os.environ.get("CTIK_SOURCE_SHA", "")
    emit(
        {
            "type": "header",
            "comparator": kind,
            "versions": versions,
            "setup_error": setup_error,
            "revisions": REVISIONS,
            "dispatch_provenance": PROVENANCE[kind],
            "planning_deviation": SGLANG_DEVIATION if kind == "sglang" else None,
            "ctikernels_source_sha": source if re.fullmatch(r"[0-9a-fA-F]{40}", source) else None,
            "provider_requested": args.provider,
            "affinity": affinity,
            "shard": list(args.shard),
            "rows": len(rows),
            "pairs": args.pairs,
            "reference_self_pairs": SELF_PAIRS,
            "calls_per_interval": args.calls,
            "warmup_calls": args.warmup,
            "atol": ATOL,
            "rtol": RTOL,
            "seed": 20261014,
            "dtype": "bfloat16",
            "page_size": 16 if kind == "vllm" else 1,
            "timing": "synchronized wall-clock us/call; complete candidate graph vs fixed comparator graph",
            "gate": "median(candidate/reference) <= 1+epsilon and both oracle checks pass",
        }
    )
    counts = {"PASS": 0, "FAIL": 0}
    for row in rows:
        if setup_error:
            result = failed_row(setup_error, "setup")
        else:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    result = run_case(kind, row, args)
            except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as error:
                result = failed_row(type(error).__name__, "measurement")
        counts[result["status"]] += 1
        emit(
            {
                "type": "row",
                "comparator": kind,
                "page_size": 16 if kind == "vllm" else 1,
                **row,
                **result,
            }
        )
    emit(
        {
            "type": "summary",
            "comparator": kind,
            "pass": counts["PASS"],
            "fail": counts["FAIL"],
            "status": "FAIL" if counts["FAIL"] or setup_error else ("PASS" if rows else "EMPTY"),
        }
    )
    return int(bool(counts["FAIL"] or setup_error))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparators", default=",".join(COMPARATORS))
    parser.add_argument("--queries", nargs="+", type=int, default=[4, 5, 6, 8, 9, 12, 16])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 512, 4096])
    parser.add_argument("--masks", nargs="+", choices=ALL_MASKS, default=list(MASKS))
    parser.add_argument("--heads", nargs="+", type=parse_heads, default=[(32, 8), (16, 4)])
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sglang-flashinfer", type=Path)
    parser.add_argument("--vllm-fa2", type=Path)
    parser.add_argument("--vllm-fa2-sha256")
    parser.add_argument("--vllm-wheel")
    parser.add_argument("--provider", choices=("auto", "cute", "flashinfer"), default="auto")
    parser.add_argument("--shard", type=parse_shard, default=(0, 1), help="zero-based I/N")
    parser.add_argument("--pin-x925", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        type=parse_row_key,
        help="keep only rows mask:B:Q:context:Hq/Hkv (after sharding)",
    )
    parser.add_argument("--worker", choices=COMPARATORS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.comparators = list(dict.fromkeys(part.strip() for part in args.comparators.split(",")))
    if any(kind not in COMPARATORS for kind in args.comparators):
        parser.error("comparators must be a comma list of flashinfer,sglang,vllm")
    if (
        args.pairs < 1
        or args.calls < 100
        or args.warmup < 20
        or any(q not in range(2, 33) for q in args.queries)
        or any(b not in range(1, 5) for b in args.batches)
        or min(args.contexts) < 0
    ):
        parser.error("require pairs>=1, calls>=100, warmup>=20, Q=2..32, B=1..4, contexts>=0")
    selected = [args.worker] if args.worker else args.comparators
    if "sglang" in selected and (
        args.sglang_flashinfer is None or not (args.sglang_flashinfer / "flashinfer").is_dir()
    ):
        parser.error("--sglang-flashinfer must contain a flashinfer 0.7.1 package")
    if "vllm" in selected and (args.vllm_fa2 is None or not args.vllm_fa2.is_file()):
        parser.error("--vllm-fa2 must point to the extracted FA2 shared library")
    if args.vllm_fa2_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.vllm_fa2_sha256):
        parser.error("--vllm-fa2-sha256 must contain 64 hexadecimal digits")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    rows = generate_rows(
        queries=args.queries,
        batches=args.batches,
        contexts=args.contexts,
        masks=args.masks,
        heads=args.heads,
        shard=args.shard,
    )
    if args.only:
        keep = set(args.only)
        rows = [row for row in rows if row_key(row) in keep]
    if args.worker:
        return worker(args.worker, rows, args)
    failed = False
    # Sequential workers avoid concurrent GPU load while isolating module names.
    for kind in args.comparators:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.kernel_tuning.dflash_draft_market",
                *argv,
                "--worker",
                kind,
            ],
            check=False,
        )
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
