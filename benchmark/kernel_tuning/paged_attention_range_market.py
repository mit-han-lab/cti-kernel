# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""PD4 engine-regime market rows; run with PYTHONPATH=python python -m
benchmark.kernel_tuning.paged_attention_range_market.

One host record and one JSON record per row. Timings are microseconds per layer:
median of --reps event samples per round, then median across --rounds. Ratios
pair round medians; their point estimate is the geometric mean and their 95%
Student-t interval is computed in log space. Raw round medians are retained.
--max-bytes caps shared resident KV only (not Q/output/workspaces or optional
placement padding). Cache footprints that cannot exceed 3x L2 are identified.
Provider chatter is discarded, including native/JIT output, to avoid exposing
machine paths. Errors contain only exception types or fixed diagnostic text.
GPU dependencies are imported only at execution time.

Source contracts inspected locally (no third-party implementation copied):
* SGLang 72730d6 flashinfer_backend.py:1057,1444,2377: graph wrappers and scales.
* vLLM 0.31 v1/attention/backends/flashinfer.py:109,1980,2053,2534: XQA
  workspace, model-dtype Q scaling, value scaling and decode call;
  utils/flashinfer.py:292 reexports the FlashInfer function, envs.py:1782 sets
  its default workspace to 394 MiB. Environment overrides are not replicated.
* FlashInfer 0.6.11.post3 / 0.7.0.post1: decode.py XQA entry point, xqa.py,
  jit/xqa.py and data/csrc/xqa/mha.h,mha.cu. Both accept NHD, page 16 and GQA 4
  with BF16 Q/output and e4m3 KV on SM120/121. The older JIT accepts D=16..256
  in steps of 16; the newer accepts up to 512 (D>256 requires GQA<=16 and Q=1).
  Neither JIT imposes a GQA whitelist for ordinary D<=256 decode; this bench
  conservatively qualifies only the requested GQA=4, D=128/256 shapes.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import math
import os
import random
import statistics
import sys
from pathlib import Path

from benchmark.kernel_tuning import paged_attention_sweep as sweep

# Per-rank (q heads, KV heads, head dim): Qwen3-8B, Qwen3.5-9B, Qwen3.5/3.8-27B and TP2 ranks.
MODELS = {
    "qwen3-8b": sweep.MODELS["qwen3-8b"],
    "qwen3.5": sweep.MODELS["qwen35-9b"],
    "qwen3.5-27b": (24, 4, 256),
    "qwen3-8b-tp2": (16, 4, 128),
    "qwen3.5-tp2": (8, 2, 256),
    "qwen3.5-27b-tp2": (12, 2, 256),
}
# The sweep's FP32 oracle and FA2 ABI read geometry by model name.
for _name, _geometry in MODELS.items():
    sweep.MODELS.setdefault(_name, _geometry)
OPS = ("fp8-decode", "graph-prefill")
# Model context lengths (config max_position_embeddings) for vLLM's capture-time max_seq_len.
MODEL_MAX_LEN = {
    "qwen3-8b": 40960,
    "qwen3.5": 262144,
    "qwen3.5-27b": 262144,
    "qwen3-8b-tp2": 40960,
    "qwen3.5-tp2": 262144,
    "qwen3.5-27b-tp2": 262144,
}
PAGE = 16
K_SCALE = V_SCALE = 0.5
XQA_WORKSPACE_BYTES = 394 * 1024**2
SGLANG_WORKSPACE_BYTES = 384 * 1024**2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=sweep.comma_choices(OPS), default=",".join(OPS))
    parser.add_argument("--models", type=sweep.comma_choices(MODELS), default="qwen3-8b,qwen3.5")
    parser.add_argument("--batches", type=sweep.comma_ints, default="1,2,3,4")
    parser.add_argument("--kv", type=sweep.comma_ints, default="512,2048,8192,32768")
    parser.add_argument("--prefill-q", type=sweep.comma_ints, default="16,64,256")
    parser.add_argument("--prefill-kv", type=sweep.comma_ints, default="2048,8192")
    parser.add_argument("--rounds", type=sweep.positive_int, default=12)
    parser.add_argument("--reps", type=sweep.positive_int, default=9)
    parser.add_argument("--max-bytes", type=sweep.positive_int, default=6 * 1024**3)
    parser.add_argument("--vllm-fa2", type=Path)
    parser.add_argument("--seed", type=int, default=sweep.SEED)
    parser.add_argument("--pad-random", action="store_true")
    parser.add_argument("--xqa-subseq", type=sweep.comma_ints)
    parser.add_argument("--draft-variants", type=lambda value: tuple(value.split(",")))
    parser.add_argument("--padsplit", type=lambda value: tuple(value.split(",")))
    parser.add_argument("--providers", type=lambda value: tuple(value.split(",")))
    args = parser.parse_args(argv)
    if args.rounds < 2:
        parser.error("--rounds must be at least 2 for a t interval")
    if "graph-prefill" in args.ops and max(args.prefill_q) > min(args.prefill_kv):
        parser.error("every --prefill-kv must include every --prefill-q")
    return args


def generate_rows(args):
    return [
        {"op": op, "model": model, "B": batch, "KV": kv, "Q": query}
        for model in args.models
        for op in args.ops
        for batch in args.batches
        for kv in (args.kv if op == "fp8-decode" else args.prefill_kv)
        for query in ([1] if op == "fp8-decode" else args.prefill_q)
    ]


@functools.cache
def t_critical_95(df):
    """Two-sided 95% Student-t critical value, without a SciPy dependency.

    Substitute t=sqrt(df)*tan(theta) into the density. Its integral is a
    normalized integral of cos(theta)**(df-1), evaluated by the elementary
    reduction formula I_n=(sin(theta)*cos(theta)**(n-1)+(n-1)*I_(n-2))/n.
    """
    if not isinstance(df, int) or df < 1:
        raise ValueError("degrees of freedom must be a positive integer")
    normalization = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)
    low, high = 0.0, math.pi / 2
    for _ in range(64):
        theta = (low + high) / 2
        sine, cosine = math.sin(theta), math.cos(theta)
        power = df - 1
        integral = theta if power % 2 == 0 else sine
        for exponent in range(2 + power % 2, power + 1, 2):
            integral = (sine * cosine ** (exponent - 1) + (exponent - 1) * integral) / exponent
        if normalization * integral < 0.475:
            low = theta
        else:
            high = theta
    return math.sqrt(df) * math.tan((low + high) / 2)


def paired_ci(numerator, denominator):
    """Paired geometric mean ratio and t-CI; never use ratios of aggregate medians."""
    if len(numerator) != len(denominator) or len(numerator) < 2:
        raise ValueError("paired CI needs at least two matched rounds")
    if any(not math.isfinite(x) or x <= 0 for x in (*numerator, *denominator)):
        raise ValueError("timings must be finite and positive")
    logs = [math.log(a) - math.log(b) for a, b in zip(numerator, denominator, strict=True)]
    mean = statistics.mean(logs)
    half_width = t_critical_95(len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs))
    return {
        "ratio": math.exp(mean),
        "ci95": [math.exp(mean - half_width), math.exp(mean + half_width)],
        "round_ratios": [a / b for a, b in zip(numerator, denominator, strict=True)],
        "n": len(logs),
    }


def summarize_ratios(samples):
    pairs = {
        "ours/sglang": ("ours", "sglang"),
        "ours/vllm": ("ours", "vllm"),
        "ours_cc/ours": ("ours_cc", "ours"),
        "ours/ours_twin": ("ours", "ours_twin"),
    }
    result = {
        label: paired_ci(samples[a], samples[b]) if a in samples and b in samples else None
        for label, (a, b) in pairs.items()
    }
    market = [samples[name] for name in ("sglang", "vllm") if name in samples]
    fastest = [min(values) for values in zip(*market, strict=True)] if market else None
    result["ours/max-market-faster"] = (
        paired_ci(samples["ours"], fastest) if "ours" in samples and fastest else None
    )
    return result


def xqa_unsupported(version, capability, hq, hk, dim, page=PAGE):
    """Conservative source qualification, not a claim about all XQA configurations."""
    if version not in ("0.6.11.post3", "0.7.0.post1"):
        return "XQA source contract not inspected for this FlashInfer version"
    if capability not in ((12, 0), (12, 1)):
        return "this market route is qualified only for SM120/SM121"
    if page not in (16, 32, 64, 128):
        return "XQA requires page size 16, 32, 64 or 128"
    # jit/xqa.py compiles HEAD_GRP_SIZE = hq // hk; both versions take D in 16..256, step 16.
    if hk <= 0 or hq % hk or not 1 <= hq // hk <= 32 or dim % 16 or not 16 <= dim <= 256:
        return "benchmark qualifies integral GQA <= 32 with head dimension 16..256 (step 16)"
    return None


def kv_footprint(row):
    _, hk, dim = MODELS[row["model"]]
    element_bytes = 1 if row["op"] == "fp8-decode" else 2
    return 2 * row["B"] * ((row["KV"] + PAGE - 1) // PAGE) * PAGE * hk * dim * element_bytes


def engine_layers(l2, per_layer, max_bytes):
    layers = sweep.layer_count(l2, per_layer, max_bytes)
    # The sweep reaches >=3x L2; prefer strict > when equality can be avoided.
    if layers * per_layer == 3 * l2 and layers < min(64, max_bytes // per_layer):
        layers += 1
    return layers


@contextlib.contextmanager
def quiet_providers():
    """Suppress Python and native provider diagnostics, which may contain paths."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    finally:
        for fd, original in zip((1, 2), saved, strict=True):
            os.dup2(original, fd)
            os.close(original)


def make_metadata(row, device):
    import torch

    b, length = row["B"], row["KV"]
    pages = (length + PAGE - 1) // PAGE
    host = {
        "q": torch.arange(b + 1, dtype=torch.int32) * row["Q"],
        "k": torch.arange(b + 1, dtype=torch.int32) * pages,
        "last": torch.full((b,), (length - 1) % PAGE + 1, dtype=torch.int32),
        "lengths": torch.full((b,), length, dtype=torch.int32),
        "indices": torch.arange(b * pages, dtype=torch.int32),
    }
    return host, {name: value.to(device) for name, value in host.items()}


def make_layers(row, layers, device, seed):
    import torch

    hq, hk, dim = MODELS[row["model"]]
    pages = (row["KV"] + PAGE - 1) // PAGE
    dtype = torch.float8_e4m3fn if row["op"] == "fp8-decode" else torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed)
    buffers = []
    for _ in range(layers):
        q = torch.randn(
            (row["B"] * row["Q"], hq, dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        kv = [
            torch.randn(
                (row["B"] * pages, PAGE, hk, dim),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ).to(dtype)
            for _ in range(2)
        ]
        buffers.append((q, *kv, torch.full_like(q, math.nan)))
    return buffers


def reference_attention(row, layer):
    """Reuse the sweep's FP32 oracle after removing page padding and dequantizing."""
    q, k, v, out = layer
    _, hk, dim = MODELS[row["model"]]
    scale = K_SCALE if row["op"] == "fp8-decode" else 1.0
    keys = k.float().view(row["B"], -1, hk, dim)[:, : row["KV"]].contiguous() * scale
    scale = V_SCALE if row["op"] == "fp8-decode" else 1.0
    values = v.float().view(row["B"], -1, hk, dim)[:, : row["KV"]].contiguous() * scale
    return sweep.reference_attention(row, (q, keys, values, out))


def make_wrapper_call(name, row, buffers, device):
    import flashinfer
    import torch

    from ctikernels.attention.inference import FlashInferPagedAttention

    decode = row["op"] == "fp8-decode"
    hq, hk, dim = MODELS[row["model"]]
    heads = {"num_qo_heads": hq, "num_kv_heads": hk, "head_dim": dim}
    host, gpu = make_metadata(row, device)
    kv_dtype = torch.float8_e4m3fn if decode else torch.bfloat16
    options = {
        "mode": "decode" if decode else "prefill",
        "backend": "fa2",
        "use_tensor_cores": decode and name != "ours_cc",
        # ours_eager: provider eager plan (exact CTA count), execution still captured.
        "use_cuda_graph": name != "ours_eager",
        "dispatch": "device",
    }
    if decode:
        # Graph indices capacity as an engine sizes it: every request up to the model
        # context (the XQA route's page-table width follows this buffer).
        capacity = capacity_table(row, gpu, device).view(-1)
        capacity[: gpu["indices"].numel()] = gpu["indices"]
        gpu["indices"] = capacity
    graph_buffers = {
        "paged_kv_indptr_buffer": gpu["k"],
        "paged_kv_indices_buffer": gpu["indices"],
        "paged_kv_last_page_len_buffer": gpu["last"],
    }
    if name == "sglang":
        workspaces = [torch.empty(SGLANG_WORKSPACE_BYTES, dtype=torch.uint8, device=device)]
        if decode:
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspaces[0],
                kv_layout="NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                backend="fa2",
                **graph_buffers,
            )
            wrapper.plan(
                indptr=host["k"],
                indices=gpu["indices"],
                last_page_len=host["last"],
                page_size=PAGE,
                **heads,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_dtype,
                data_type=kv_dtype,
                sm_scale=dim**-0.5,
                non_blocking=True,
            )
        else:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspaces[0],
                kv_layout="NHD",
                use_cuda_graph=True,
                backend="fa2",
                qo_indptr_buf=gpu["q"],
                **{
                    key.removesuffix("buffer") + "buf": value
                    for key, value in graph_buffers.items()
                },
            )
            wrapper.plan(
                qo_indptr=host["q"],
                paged_kv_indptr=host["k"],
                paged_kv_indices=gpu["indices"],
                paged_kv_last_page_len=host["last"],
                num_qo_heads=hq,
                num_kv_heads=hk,
                head_dim_qk=dim,
                page_size=PAGE,
                causal=True,
                sm_scale=dim**-0.5,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_dtype,
                non_blocking=True,
            )
    else:
        sizes = FlashInferPagedAttention.workspace_size(
            **options,
            max_batch_size=row["B"],
            max_query_tokens=row["B"] * row["Q"],
            device=device,
            **heads,
        )
        workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
        wrapper = FlashInferPagedAttention(
            *workspaces,
            **options,
            **(graph_buffers if options["use_cuda_graph"] else {}),
            **({"qo_indptr_buffer": gpu["q"]} if not decode and options["use_cuda_graph"] else {}),
        )
        wrapper.plan(
            cu_seqlens_q=host["q"],
            cu_seqlens_k=host["k"],
            indices=gpu["indices"],
            last_page_len=host["last"],
            seq_lens=host["lengths"],
            page_size=PAGE,
            pos_encoding_mode="NONE",
            dtype=torch.bfloat16,
            kv_dtype=kv_dtype,
            sm_scale=dim**-0.5,
            causal=True,
            disable_split_kv=name == "ours_nosplit",
            **heads,
        )
    torch.cuda.synchronize(device)

    def call():
        for q, k, v, out in buffers:
            wrapper.run(
                q=q,
                paged_kv_cache=(k, v),
                out=out,
                **({"k_scale": K_SCALE, "v_scale": V_SCALE} if decode else {}),
            )

    return call, (wrapper, workspaces, host, gpu)


DRAFT_KNOBS = {
    "r": "ROWS",
    "s": "SPLITS",
    "t": "KEY_TILE",
    "g": "KEY_GROUPS",
    "st": "STAGES",
    "x": "TARGET_CTAS",
    "m": "MIN_CHUNK",
    "c": "CHUNK_CAP",
}


def draft_variant_env(name):
    """cd_r64_s1_t64 -> CTI_DRAFT_ROWS=64, CTI_DRAFT_SPLITS=1, CTI_DRAFT_KEY_TILE=64."""
    import re

    env = {}
    for part in name.split("_")[1:]:
        key, value = re.fullmatch(r"([a-z]+)(\d+)", part).groups()
        env["CTI_DRAFT_" + DRAFT_KNOBS[key]] = value
    return env


FP8_KNOBS = {"s": "SPLITS", "m": "MIN_CHUNK", "w": "WARPS", "k": "KEYS", "st": "STAGES"}


def fp8_variant_env(name):
    """cf_s42_st4 -> CTI_FP8DEC_SPLITS=42, CTI_FP8DEC_STAGES=4; cf_last -> last-arrival merge."""
    import re

    env = {}
    for part in name.split("_")[1:]:
        if part == "last":
            env["CTI_FP8DEC_MERGE"] = "last"
            continue
        key, value = re.fullmatch(r"([a-z]+)(\d+)", part).groups()
        env["CTI_FP8DEC_" + FP8_KNOBS[key]] = value
    return env


def make_draft_variant_call(row, buffers, device, name):
    if name.startswith("cp_"):
        variant, maker = prefill_variant_env(name), make_cute_prefill_call
    elif name.startswith("cf_"):
        variant, maker = fp8_variant_env(name), make_cute_fp8_call
    else:
        variant, maker = draft_variant_env(name), make_draft_call
    saved = {key: os.environ.get(key) for key in variant}
    os.environ.update(variant)
    try:
        call, retained = maker(row, buffers, device)
        call()  # compile and warm under the variant environment
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return call, retained


PREFILL_KNOBS = {"m": "BLOCK_M", "n": "BLOCK_N", "st": "STAGES", "s": "SPLITS", "c": "MIN_CHUNK"}


def prefill_variant_env(name):
    """cp_m64_n32_st3_s2_c128 -> construction-time CTI_PREFILL overrides."""
    import re

    env = {}
    for part in name.split("_")[1:]:
        match = re.fullmatch(r"([a-z]+)(\d+)", part)
        if match is None or match[1] not in PREFILL_KNOBS:
            raise ValueError(f"invalid CuTe prefill variant: {name}")
        env["CTI_PREFILL_" + PREFILL_KNOBS[match[1]]] = match[2]
    return env


def make_cute_prefill_call(row, buffers, device):
    """One fixed-grid paged prefill per layer, retaining graph CSR and workspace."""
    import torch

    from ctikernels.attention._prefill_cute import CutePagedPrefill, workspace_size

    hq, hk, dim = MODELS[row["model"]]
    host, gpu = make_metadata(row, device)
    geometry = {
        "batch_size": row["B"],
        "max_query_per_request": row["Q"],
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
    }
    workspace = torch.empty(workspace_size(device, **geometry), dtype=torch.uint8, device=device)
    op = CutePagedPrefill(
        workspace,
        **geometry,
        page_size=PAGE,
        dtype=buffers[0][0].dtype,
        device=device,
        q_row_stride=buffers[0][0].stride(0),
    )

    def call():
        for q, k, v, out in buffers:
            op.run(q, k, v, gpu["q"], gpu["k"], gpu["indices"], gpu["last"], out)

    return call, (op, workspace, host, gpu)


def make_draft_call(row, buffers, device):
    """PD5 screen: the CuTe DRAFT kernel (packed GQA, split-KV) on extend rows.

    Uniform Q per request with bottom-right causal alignment is the DRAFT contract;
    token indices address the page-16 cache rows directly.
    """
    import torch

    from ctikernels.attention.draft import DraftAttention

    hq, hk, dim = MODELS[row["model"]]
    b, qlen, length = row["B"], row["Q"], row["KV"]
    pages = (length + PAGE - 1) // PAGE
    geometry = {
        "query_length": qlen,
        "num_qo_heads": hq,
        "num_kv_heads": hk,
        "head_dim": dim,
        "page_size": PAGE,
        "dtype": torch.bfloat16,
        "causal": True,
        "window": None,
    }
    sizes = DraftAttention.workspace_size(
        batch_size=b, max_kv_tokens=length, device=device, provider="cute", **geometry
    )
    workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
    tokens = torch.arange(length, dtype=torch.int32)
    host = {
        "request_indices": torch.arange(b, dtype=torch.int32),
        "positions": torch.arange(length - qlen, length, dtype=torch.int64).repeat(b),
        "token_indptr": torch.arange(b + 1, dtype=torch.int32) * length,
        "token_indices": torch.cat([r * pages * PAGE + tokens for r in range(b)]),
    }
    gpu = {name: value.to(device) for name, value in host.items()}
    scratch = {
        "query_indptr": torch.empty(b + 1, dtype=torch.int32, device=device),
        "page_indptr": torch.empty(b + 1, dtype=torch.int32, device=device),
        "page_indices": torch.empty(b * pages, dtype=torch.int32, device=device),
        "last_page_lengths": torch.empty(b, dtype=torch.int32, device=device),
    }
    op = DraftAttention(*workspaces, provider="cute", **gpu, **scratch, **geometry)

    def call():
        op.prepare()
        for q, k, v, out in buffers:
            op.run(q=q, paged_kv_cache=(k, v), out=out)

    return call, (op, workspaces, gpu, scratch)


def make_padsplit_call(row, buffers, device, pad, chunk):
    """PD5 screen: graph prefill planned with `pad` empty trailing requests (raising
    FlashInfer's graph CTA bound, scheduler.cuh PrefillSplitQOKVIndptr) and a fixed KV
    split of `chunk` pages, so B*tiles*splits CTAs fit the captured grid."""
    import flashinfer
    import torch

    hq, hk, dim = MODELS[row["model"]]
    host, _ = make_metadata(row, device)
    b = row["B"]
    tail = lambda t, value: torch.cat([t, torch.full((pad,), value, dtype=t.dtype)])
    qo = tail(host["q"], int(host["q"][-1]))
    kv = tail(host["k"], int(host["k"][-1]))
    last = tail(host["last"], 0)
    gpu = {
        "qo": qo.to(device),
        "kv": kv.to(device),
        "last": last.to(device),
        "indices": host["indices"].to(device),
    }
    workspace = torch.zeros(SGLANG_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        use_cuda_graph=True,
        backend="fa2",
        qo_indptr_buf=gpu["qo"],
        paged_kv_indptr_buf=gpu["kv"],
        paged_kv_indices_buf=gpu["indices"],
        paged_kv_last_page_len_buf=gpu["last"],
    )
    wrapper.plan(
        qo_indptr=qo,
        paged_kv_indptr=kv,
        paged_kv_indices=gpu["indices"],
        paged_kv_last_page_len=last,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=dim,
        page_size=PAGE,
        causal=True,
        sm_scale=dim**-0.5,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        non_blocking=True,
        **({"fixed_split_size": chunk} if chunk else {}),
    )
    torch.cuda.synchronize(device)
    del b

    def call():
        for q, k, v, out in buffers:
            wrapper.run(q=q, paged_kv_cache=(k, v), out=out)

    return call, (wrapper, workspace, gpu)


PADSPLIT = ()  # --padsplit pad:chunk screens for graph prefill


def make_cute_fp8_call(row, buffers, device):
    """Fixed-grid FP8 decode; CSR contents select the active splits on device."""
    import torch

    from ctikernels.attention._decode_fp8 import FP8PagedDecode, workspace_size

    hq, hk, dim = MODELS[row["model"]]
    host, gpu = make_metadata(row, device)
    capacity = capacity_table(row, gpu, device).view(-1)
    capacity[: gpu["indices"].numel()] = gpu["indices"]
    gpu["indices"] = capacity
    geometry = {"batch_size": row["B"], "num_qo_heads": hq, "num_kv_heads": hk, "head_dim": dim}
    workspace = torch.empty(workspace_size(device, **geometry), dtype=torch.uint8, device=device)
    op = FP8PagedDecode(
        workspace, **geometry, page_size=PAGE, q_dtype=buffers[0][0].dtype, device=device
    )

    def call():
        for q, k, v, out in buffers:
            op.run(
                q,
                k,
                v,
                gpu["k"],
                gpu["indices"],
                gpu["last"],
                out,
                k_scale=K_SCALE,
                v_scale=V_SCALE,
            )

    return call, (op, workspace, host, gpu)


def make_xqa_call(row, buffers, device):
    import torch
    from flashinfer.decode import xqa_batch_decode_with_kv_cache

    _, gpu = make_metadata(row, device)
    # vLLM captures decode with max_seq_len = max_model_len and a block table of
    # cdiv(max_model_len, page) columns (gpu_model_runner.py:2313, block_table.py:112);
    # XQA sizes its sub-sequence grid from that host value.
    max_len = max(MODEL_MAX_LEN[row["model"]], row["KV"])
    table = torch.zeros(row["B"], -(-max_len // PAGE), dtype=torch.int32, device=device)
    pages = gpu["indices"].view(row["B"], -1)
    table[:, : pages.shape[1]] = pages
    workspace = torch.zeros(XQA_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    dim = MODELS[row["model"]][2]

    def call():
        for q, k, v, out in buffers:
            xqa_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k, v),
                workspace_buffer=workspace,
                block_tables=table,
                seq_lens=gpu["lengths"],
                max_seq_len=max_len,
                bmm1_scale=dim**-0.5 * K_SCALE,
                bmm2_scale=V_SCALE,
                window_left=-1,
                out=out,
                sinks=None,
                kv_layout="NHD",
                q_len_per_req=1,
                mask=None,
            )

    return call, (workspace, gpu, table)


def make_vllm_prefill_call(row, buffers, device):
    """Sweep's FA2 paged ABI with uniform Q>1 and bottom-right causal alignment."""
    import torch

    # Q=1 is already implemented by the sweep with the identical paged ABI.
    if row["Q"] == 1:
        return sweep.make_vllm_call(row, buffers, len(buffers), device), ()
    _, gpu = make_metadata(row, device)
    dummy_k = torch.zeros_like(gpu["q"])
    table = gpu["indices"].view(row["B"], -1)
    dim = MODELS[row["model"]][2]

    def call():
        for q, k, v, out in buffers:
            torch.ops._vllm_fa2_C.varlen_fwd(
                q,
                k,
                v,
                out,
                gpu["q"],
                dummy_k,
                gpu["lengths"],
                None,
                table,
                None,
                row["Q"],
                row["KV"],
                0.0,
                dim**-0.5,
                False,
                True,
                -1,
                -1,
                0.0,
                False,
                0,
                None,
            )

    return call, (gpu, dummy_k, table)


def time_graphs(graphs, layers, args, rng):
    samples = {name: [] for name in graphs}
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            graph = graphs[name]
            graph.replay()  # one untimed warm replay before this provider's samples
            samples[name].append(
                statistics.median(sweep.measure(graph.replay, layers) for _ in range(args.reps))
            )
    return samples


XQA_SUBSEQ = ()  # --xqa-subseq screens: fixed XQA sub-sequence counts per (request, KV head)
PROVIDERS = None  # --providers keeps only these names
DRAFT_VARIANTS = ()  # --draft-variants cd_r64_s1_t64,... (CuTe DRAFT tuning screens)


def provider_names(row):
    return [name for name in _provider_names(row) if PROVIDERS is None or name in PROVIDERS]


def _provider_names(row):
    variants = [
        name
        for name in DRAFT_VARIANTS
        if not name.startswith("cp_") or row["op"] == "graph-prefill"
    ]
    return ["ours", "ours_twin", "sglang", "vllm", *variants] + (
        ["ours_cc", "draft_bf16", "cute_fp8", *(f"xqa_s{count}" for count in XQA_SUBSEQ)]
        if row["op"] == "fp8-decode"
        else [
            "ours_nosplit",
            "ours_eager",
            "cute_draft",
            "cute_prefill",
            *(f"ps{p}_{c}" for p, c in PADSPLIT),
            *(f"pc{c}" for c in (1, 2, 3, 4, 6, 8)),
            *(f"pt64c{c}" for c in (1, 2, 4)),
            *(f"pt16c{c}" for c in (1, 2)),
        ]
    )


def capacity_table(row, gpu, device):
    import torch

    max_len = max(MODEL_MAX_LEN[row["model"]], row["KV"])
    table = torch.zeros(row["B"], -(-max_len // PAGE), dtype=torch.int32, device=device)
    pages = gpu["indices"].view(row["B"], -1)
    table[:, : pages.shape[1]] = pages
    return table


def make_xqa_subseq_call(row, buffers, device, count):
    """XQA through its public launcher with sm_count fixing the sub-sequence grid.

    FlashInfer's launcher (data/csrc/xqa/mha.cu, launchMHAFlashInfer) splits each
    (request, KV head) into min(sm_count // (B * Hkv), ceil(max_seq_len / tile)) CTAs.
    """
    import torch
    from flashinfer.xqa import xqa

    hq, hk, dim = MODELS[row["model"]]
    _, gpu = make_metadata(row, device)
    # Engine-shaped capacity: a block table as wide as the model context (as vLLM's),
    # so the launcher's ceil(width * page / 256) bound never trims the fixed split.
    table = capacity_table(row, gpu, device)
    semaphores = torch.zeros(8 * 1024**2, dtype=torch.uint8, device=device)
    scratch = torch.empty(256 * 1024**2, dtype=torch.uint8, device=device)
    lengths = gpu["lengths"].unsqueeze(1)

    def call():
        for q, k, v, out in buffers:
            xqa(
                q.unsqueeze(1),
                k,
                v,
                table,
                lengths,
                out.unsqueeze(1),
                scratch,
                semaphores,
                hk,
                PAGE,
                q_scale=dim**-0.5 * K_SCALE / V_SCALE * dim**0.5,
                kv_scale=V_SCALE,
                kv_layout="NHD",
                sm_count=count * row["B"] * hk,
            )

    return call, (semaphores, scratch, gpu, table, lengths)


def run_row(row, args, device, l2, flashinfer_version, fa2_error, rng):
    import torch

    per_layer = kv_footprint(row)
    layers = engine_layers(l2, per_layer, args.max_bytes)
    pad_bytes = rng.randint(0, 1024**3) if args.pad_random else 0
    pad = torch.empty(pad_bytes, dtype=torch.uint8, device=device)
    source = make_layers(row, layers, device, args.seed)
    expected = reference_attention(row, source[0])
    providers, reasons, graphs, owners = {}, {}, {}, [pad, source]
    for name in provider_names(row):
        reason = None
        if name == "vllm":
            if row["op"] == "fp8-decode":
                reason = xqa_unsupported(
                    flashinfer_version,
                    torch.cuda.get_device_capability(device),
                    *MODELS[row["model"]],
                )
            else:
                reason = fa2_error or (
                    "optional FA2 library not supplied" if args.vllm_fa2 is None else None
                )
        providers[name] = None
        if reason:
            reasons[name] = reason
            continue
        result = {"correct": None, "max_error": None, "us": None, "round_us": []}
        providers[name] = result
        try:
            buffers = (
                source
                if name == "ours"
                else [(q.clone(), k, v, torch.full_like(out, math.nan)) for q, k, v, out in source]
            )
            owners.append(buffers)
            if name.startswith("xqa_s"):
                call, retained = make_xqa_subseq_call(row, buffers, device, int(name[5:]))
            elif name == "cute_fp8":
                call, retained = make_cute_fp8_call(row, buffers, device)
            elif name == "cute_prefill":
                call, retained = make_cute_prefill_call(row, buffers, device)
            elif name.startswith("pt"):
                # pt64c2: pad so FlashInfer's graph tile (from T - B' + 1 packed rows)
                # becomes 64, then split KV into C chunks.
                tile, chunks = (int(x) for x in name[2:].split("c"))
                hq, hk, _ = MODELS[row["model"]]
                group, total = hq // hk, row["B"] * row["Q"]
                pad = total + 1 - tile // group - row["B"]
                if pad < 0:
                    raise ValueError("too few rows to steer the tile")
                pages = -(-row["KV"] // PAGE)
                call, retained = make_padsplit_call(
                    row, buffers, device, pad, -(-pages // chunks) if chunks > 1 else 0
                )
            elif name.startswith("pc"):
                # C KV chunks per request; pad empty requests so the captured grid
                # (ceil(T*g/128) + B' - 1 tiles) holds every split tile.
                chunks = int(name[2:])
                hq, hk, dim = MODELS[row["model"]]
                tile = 64 if dim == 256 else 128  # FA2DetermineCtaTileQ for packed rows > 64
                tiles = -(-row["B"] * row["Q"] * (hq // hk) // tile) + row["B"] - 1
                pages = -(-row["KV"] // PAGE)
                call, retained = make_padsplit_call(
                    row, buffers, device, (chunks - 1) * tiles + 1, -(-pages // chunks)
                )
            elif name.startswith("ps"):
                pad, chunk = (int(x) for x in name[2:].split("_"))
                call, retained = make_padsplit_call(row, buffers, device, pad, chunk)
            elif name.startswith(("cd_", "cf_", "cp_")):
                call, retained = make_draft_variant_call(row, buffers, device, name)
            elif name == "cute_draft":
                call, retained = make_draft_call(row, buffers, device)
            elif name == "draft_bf16":
                # PD5 calibration: the BF16 CuTe split-KV kernel on dequantized copies.
                buffers = [
                    (
                        q,
                        (k.float() * K_SCALE).to(torch.bfloat16),
                        (v.float() * V_SCALE).to(torch.bfloat16),
                        out,
                    )
                    for q, k, v, out in buffers
                ]
                owners.append(buffers)
                call, retained = make_draft_call(row, buffers, device)
            elif name != "vllm":
                call, retained = make_wrapper_call(name, row, buffers, device)
            elif row["op"] == "fp8-decode":
                call, retained = make_xqa_call(row, buffers, device)
            else:
                call, retained = make_vllm_prefill_call(row, buffers, device)
            owners.append((call, retained))
            graph = sweep.graph_capture(call)
            actual = buffers[0][3].float()
            error = (actual - expected).abs().max().item()
            result["max_error"] = error if math.isfinite(error) else None
            result["correct"] = bool(torch.allclose(actual, expected, atol=2e-2, rtol=2e-2))
            if not result["correct"]:
                result["status"] = "incorrect"
                reasons[name] = (
                    "output is nonfinite" if not math.isfinite(error) else "FP32 reference mismatch"
                )
                continue
            graphs[name] = graph
            result["status"] = "ok"
        except Exception as error:  # noqa: BLE001 -- isolate provider failures; never emit exception text
            result["status"] = "error"
            reasons[name] = type(error).__name__
    del expected
    torch.cuda.synchronize(device)
    samples = time_graphs(graphs, layers, args, rng)
    for name, values in samples.items():
        providers[name].update(us=statistics.median(values), round_us=values)
    torch.cuda.synchronize(device)
    return {
        "layers": layers,
        "kv_bytes": layers * per_layer,
        "kv_exceeds_3x_l2": layers * per_layer > 3 * l2,
        "pad_bytes": pad_bytes,
        "providers": providers,
        "reasons": reasons,
        "ratios": summarize_ratios(samples),
    }


def main(argv=None):
    global XQA_SUBSEQ, PROVIDERS
    args = parse_args(argv)
    XQA_SUBSEQ = tuple(args.xqa_subseq or ())
    global PADSPLIT, DRAFT_VARIANTS
    DRAFT_VARIANTS = tuple(args.draft_variants or ())
    PADSPLIT = tuple(tuple(int(x) for x in item.split(":")) for item in (args.padsplit or ()))
    PROVIDERS = args.providers
    host = dict.fromkeys(("gpu", "sms", "l2", "torch", "flashinfer", "vllm_so_sha256"))
    setup_error = fa2_error = torch_module = None
    try:
        with quiet_providers():
            import torch

            torch_module = torch
            host["torch"] = torch.__version__
            import flashinfer

            host["flashinfer"] = flashinfer.__version__
            device = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(device)
            host.update(
                gpu=properties.name,
                sms=properties.multi_processor_count,
                l2=properties.L2_cache_size,
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            if args.vllm_fa2 is not None:
                try:
                    host["vllm_so_sha256"] = sweep.library_sha256(args.vllm_fa2)
                    torch.ops.load_library(str(args.vllm_fa2))
                except Exception as error:  # noqa: BLE001 -- optional provider must not disable others
                    fa2_error = type(error).__name__
    except Exception as error:  # noqa: BLE001 -- structured errors, without paths
        setup_error = type(error).__name__
    sweep.emit({"type": "host", **host})
    rng = random.Random(args.seed)
    for row in generate_rows(args):
        record = {
            "type": "row",
            **row,
            "regime": "engine",
            "seed": args.seed,
            "rounds": args.rounds,
            "reps": args.reps,
            "page_size": PAGE,
        }
        try:
            if setup_error:
                record.update(status="error", error=setup_error)
            else:
                with quiet_providers(), torch.inference_mode():
                    record.update(
                        run_row(row, args, device, host["l2"], host["flashinfer"], fa2_error, rng)
                    )
        except Exception as error:  # noqa: BLE001 -- isolate row failures without diagnostics
            record.update(status="error", error=type(error).__name__)
        sweep.emit(record)
        gc.collect()
        if torch_module is not None and torch_module.cuda.is_initialized():
            with quiet_providers():
                torch_module.cuda.empty_cache()


if __name__ == "__main__":
    main()
