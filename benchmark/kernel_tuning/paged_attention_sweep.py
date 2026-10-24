# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Paged attention sweep; run with PYTHONPATH=python python -m benchmark.kernel_tuning.paged_attention_sweep.

JSONL timings are CUDA-event microseconds per layer call. Each verify graph has
one shared prepare followed by calls runs, so its prepare cost is amortized.
Variants run round-robin with the starting variant rotated each rep. Decode
samples pair each variant with an adjacent vLLM sample; ratios are the median
of those sample ratios. --pairs batches graph replays within each paired sample.
--max-bytes bounds resident K/V across both providers,
including vLLM block padding, but excludes queries, outputs and workspaces.
No GPU providers are imported until execution.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import statistics
import sys
from importlib.metadata import version
from pathlib import Path

MODELS = {"qwen3-8b": (32, 8, 128), "qwen35-9b": (16, 4, 256)}
OPS = ("decode", "verify", "prefill")
REGIMES = ("hot", "engine")
SEED = 20261015


class UnsupportedVersion(RuntimeError):
    """The fixed-query provider's private layout is not qualified."""


class Infeasible(ValueError):
    """A split cannot be represented by this row's launch geometry."""


def positive_int(value):
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a positive integer")


def comma_ints(value):
    return list(dict.fromkeys(positive_int(part.strip()) for part in value.split(",")))


def decode_splits(value):
    return list(
        dict.fromkeys(
            part.strip()
            if part.strip() in ("device", "provider", "notc")
            else positive_int(part.strip())
            for part in value.split(",")
        )
    )


def verify_splits(value):
    variants = []
    for part in value.split(","):
        variant = part.strip()
        if variant == "tiers":
            variants.append(variant)
            continue
        match = re.fullmatch(r"m([0-9]+)", variant)
        if match is None or not 1 <= int(match[1]) <= 128:
            raise argparse.ArgumentTypeError(
                "expected verify variants tiers or mN with N in [1, 128]"
            )
        variants.append(f"m{int(match[1])}")
    return list(dict.fromkeys(variants))


def prefill_splits(value):
    variants = []
    for part in value.split(","):
        if part.strip() in ("device", "provider"):
            variants.append(part.strip())
            continue
        match = re.fullmatch(r"([qc]?)([0-9]+)(\+ns)?", part.strip())
        if match is None:
            raise argparse.ArgumentTypeError(
                "expected prefill variants device, provider, qN or cN, optionally +ns"
            )
        kind, size, suffix = match.groups()
        variants.append(f"{kind or 'q'}{positive_int(size)}{suffix or ''}")
    return list(dict.fromkeys(variants))


def qsplit_partition(query, parts):
    """Consecutive nonempty [start, end) ranges, with remainder assigned first."""
    if query < 1 or not 1 <= parts <= query:
        raise Infeasible("prefill qsplit requires 1 <= N <= Q")
    width, remainder = divmod(query, parts)
    boundaries = [i * width + min(i, remainder) for i in range(parts + 1)]
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def prefill_partition(query, split):
    """Per-request query ranges; each range [s, e) uses that request's KV [0, e)."""
    variant = split.removesuffix("+ns")
    size = int(variant[1:])
    if variant[0] == "q":
        return qsplit_partition(query, size)
    if variant[0] != "c" or query < 1 or size < 1:
        raise Infeasible("prefill chunks require positive Q and chunk size")
    return [(start, min(start + size, query)) for start in range(0, query, size)]


def row_splits(row, args):
    if row["op"] == "decode":
        return args.decode_split
    if row["op"] == "verify":
        return args.verify_split
    return args.prefill_qsplit


def comma_choices(choices):
    def parse(value):
        parts = list(dict.fromkeys(part.strip() for part in value.split(",")))
        if any(part not in choices for part in parts):
            raise argparse.ArgumentTypeError(f"expected a comma list from {','.join(choices)}")
        return parts

    return parse


def parse_only(value):
    """Parse OR-ed op:model:B:KV selectors; * matches an entire field."""
    selectors = []
    for item in value.split(","):
        fields = tuple(part.strip() for part in item.split(":"))
        if len(fields) != 4:
            raise argparse.ArgumentTypeError("--only requires op:model:B:KV selectors")
        op, model, batch, kv = fields
        if op not in (*OPS, "*") or model not in (*MODELS, "*"):
            raise argparse.ArgumentTypeError("unknown op/model in --only")
        selectors.append(
            (
                op,
                model,
                "*" if batch == "*" else positive_int(batch),
                "*" if kv == "*" else positive_int(kv),
            )
        )
    return selectors


def matches_only(row, selectors):
    key = (row["op"], row["model"], row["B"], row["KV"])
    return not selectors or any(
        all(
            pattern == "*" or pattern == field for pattern, field in zip(selector, key, strict=True)
        )
        for selector in selectors
    )


def generate_rows(args):
    rows = []
    for model in args.models:
        for op in args.ops:
            batches = args.prefill_batches if op == "prefill" else args.batches
            lengths = args.prefill_kv if op == "prefill" else args.kv
            for batch in batches:
                for kv in lengths:
                    query = kv if op == "prefill" else (args.verify_q if op == "verify" else 1)
                    row = {"op": op, "model": model, "B": batch, "KV": kv, "Q": query}
                    if matches_only(row, args.only):
                        rows.append(row)
    return rows


def layer_count(l2, per_layer_kv_bytes, max_bytes, comparator_kv_bytes=0):
    """Reach 3x L2 where possible, subject to [2,64] and the resident KV cap."""
    if l2 < 0 or per_layer_kv_bytes <= 0 or max_bytes <= 0 or comparator_kv_bytes < 0:
        raise ValueError("invalid cache sizes")
    desired = max(2, min(64, (3 * l2 + per_layer_kv_bytes - 1) // per_layer_kv_bytes))
    capacity = max_bytes // (per_layer_kv_bytes + comparator_kv_bytes)
    if capacity < 2:
        raise ValueError("--max-bytes cannot hold the minimum two engine layers")
    return min(desired, capacity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=comma_choices(MODELS), default="qwen3-8b")
    parser.add_argument("--ops", type=comma_choices(OPS), default="decode,verify,prefill")
    parser.add_argument("--batches", type=comma_ints, default="1,2,3,4")
    parser.add_argument("--kv", type=comma_ints, default="512,1024,2048,4096,8192,16384")
    parser.add_argument("--prefill-kv", type=comma_ints, default="22,128,512,2048")
    parser.add_argument("--prefill-batches", type=comma_ints, default="1")
    parser.add_argument("--verify-q", type=positive_int, default=16)
    parser.add_argument("--decode-split", type=decode_splits, default="device,provider")
    parser.add_argument(
        "--verify-split",
        type=verify_splits,
        default="tiers,m128",
        help="comma list: tiers uses device tiers; mN sets minimum chunk tokens (1..128; m128 is native)",
    )
    parser.add_argument(
        "--prefill-qsplit",
        type=prefill_splits,
        default="device,provider",
        help="comma list: device, provider, qN equal parts or cN-query chunks per request; +ns disables Split-KV",
    )
    parser.add_argument("--only", type=parse_only)
    parser.add_argument("--regimes", type=comma_choices(REGIMES), default="hot,engine")
    parser.add_argument("--hot-calls", type=positive_int, default=16)
    parser.add_argument("--max-bytes", type=positive_int, default=6 * 1024**3)
    parser.add_argument("--reps", type=positive_int, default=9)
    parser.add_argument(
        "--pairs",
        type=positive_int,
        default=7,
        help="graph replays per timed sample when pairing decode with vLLM",
    )
    parser.add_argument("--vllm-fa2", type=Path)
    return parser.parse_args(argv)


@contextlib.contextmanager
def provider_output():
    # Redirect native/JIT subprocess chatter too, not just Python print calls.
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def library_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_operation(row, device, split="device"):
    import torch

    from ctikernels.attention.inference import (
        FlashInferFixedQueryAttention,
        FlashInferPagedAttention,
        select_flashinfer_prefill_implementation,
    )

    op, b, qlen = row["op"], row["B"], row["Q"]
    length = row["KV"] + (qlen if op == "verify" else 0)
    hq, hk, dim = MODELS[row["model"]]
    heads = {"num_qo_heads": hq, "num_kv_heads": hk, "head_dim": dim}
    cpu_q = torch.arange(b + 1, dtype=torch.int32, device="cpu") * qlen
    cpu_k = torch.arange(b + 1, dtype=torch.int32, device="cpu") * length
    cpu_last = torch.ones(b, dtype=torch.int32, device="cpu")
    cpu_lengths = torch.full((b,), length, dtype=torch.int32, device="cpu")
    indices = torch.arange(b * length, dtype=torch.int32, device=device)
    if op == "prefill" and split not in ("device", "provider"):
        parts = prefill_partition(qlen, split)
        cpu_q = torch.tensor(
            [0] + [request * qlen + end for request in range(b) for _, end in parts],
            dtype=torch.int32,
        )
        cpu_lengths = torch.tensor([end for _, end in parts] * b, dtype=torch.int32)
        cpu_k = torch.cat(
            (torch.zeros(1, dtype=torch.int32), cpu_lengths.cumsum(0, dtype=torch.int32))
        )
        # Pseudo-requests share only their original request's physical page prefix.
        indices = torch.cat(
            [
                indices[request * length : request * length + end]
                for request in range(b)
                for _, end in parts
            ]
        )
        b *= len(parts)
        cpu_last = torch.ones(b, dtype=torch.int32)
    ptr, last = cpu_k.to(device), cpu_last.to(device)
    if op == "verify":
        import flashinfer

        # The adapter gates distribution metadata. Also gate the imported module:
        # a driver can select another installation using sys.path alone.
        from ctikernels.attention._flashinfer import QUALIFIED_VERSIONS

        installed = version("flashinfer-python")
        if flashinfer.__version__ not in QUALIFIED_VERSIONS or installed != flashinfer.__version__:
            raise UnsupportedVersion(
                f"fixed-query Split-KV is qualified with {QUALIFIED_VERSIONS}; "
                f"found module {flashinfer.__version__}, distribution {installed}"
            )
        sizes = FlashInferFixedQueryAttention.workspace_size(
            batch_size=b, query_length=qlen, device=device, **heads
        )
        workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
        query_ptr = cpu_q.to(device)
        wrapper = FlashInferFixedQueryAttention(
            *workspaces,
            query_indptr=query_ptr,
            kv_indptr=ptr,
            kv_indices=indices,
            last_page_lengths=last,
            query_length=qlen,
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
            **({"min_chunk_tokens": int(split[1:])} if split.startswith("m") else {}),
            **heads,
        )
        # Construction plans dummy Q-length keys and overwrites borrowed buffers.
        query_ptr.copy_(cpu_q)
        ptr.copy_(cpu_k)
        last.copy_(cpu_last)
        plan = None
    else:
        decode = op == "decode"
        backend = (
            "fa2" if decode else select_flashinfer_prefill_implementation(device, torch.bfloat16)
        )
        options = {
            "mode": op,
            "backend": backend,
            # "notc" measures FlashInfer's CUDA-core decode kernel for the same row.
            "use_tensor_cores": decode and hq // hk >= 4 and split != "notc",
            "use_cuda_graph": decode,
            "dispatch": "device" if split == "device" else "provider",
        }
        sizes = FlashInferPagedAttention.workspace_size(
            **options, max_batch_size=b, max_query_tokens=row["B"] * qlen, device=device, **heads
        )
        workspaces = [torch.zeros(size, dtype=torch.uint8, device=device) for size in sizes]
        wrapper = FlashInferPagedAttention(
            *workspaces,
            **options,
            paged_kv_indptr_buffer=ptr if decode else None,
            paged_kv_indices_buffer=indices if decode else None,
            paged_kv_last_page_len_buffer=last if decode else None,
        )
        plan_arguments = dict(
            cu_seqlens_q=cpu_q,
            cu_seqlens_k=cpu_k,
            indices=indices,
            last_page_len=cpu_last,
            seq_lens=cpu_lengths,
            page_size=1,
            pos_encoding_mode="NONE",
            dtype=torch.bfloat16,
            causal=True,
            **heads,
        )
        if op == "prefill" and split.endswith("+ns"):
            plan_arguments["disable_split_kv"] = True
        wrapper.plan(**plan_arguments)
        if decode and split == "device":
            torch.cuda.synchronize(device)
            wrapper.plan(**plan_arguments)
        # Complete async host staging even if feasibility rejects the variant.
        torch.cuda.synchronize(device)
        # Native plan fields are read only through the version-qualified adapter.
        from ctikernels.attention._flashinfer import decode_plan_capacity

        capacity = decode_plan_capacity(wrapper._wrapper) if decode else None
        if decode and isinstance(split, int):
            if not options["use_tensor_cores"]:
                raise Infeasible("fixed decode splits require use_tensor_cores=True")
            if capacity is None:
                raise Infeasible("provider decode plan does not expose its padded CTA count")
            if b * ((length + split - 1) // split) > capacity[0]:
                raise Infeasible("fixed decode split exceeds the provider plan's padded CTA count")
            # Match FlashInferPagedAttention.plan's decode mapping, adding only
            # fixed_split_size. Replan this wrapper before capturing its graph.
            wrapper._wrapper.plan(
                indptr=cpu_k,
                indices=indices,
                last_page_len=cpu_last,
                **heads,
                page_size=1,
                pos_encoding_mode="NONE",
                seq_lens=cpu_lengths,
                data_type=torch.bfloat16,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                non_blocking=True,
                fixed_split_size=split,
            )
            capacity = decode_plan_capacity(wrapper._wrapper)
        plan = list(capacity[1])[:5] if capacity is not None else None
        query_ptr = None
    # Retain all caller-owned buffers, including host inputs to async planning.
    retained = (workspaces, cpu_q, cpu_k, cpu_last, cpu_lengths, indices, ptr, last, query_ptr)
    torch.cuda.synchronize(device)
    return wrapper, plan, retained


def make_layers(row, layers, device, *, comparator=False, source=None):
    import torch

    hq, hk, dim = MODELS[row["model"]]
    b, qlen = row["B"], row["Q"]
    length = row["KV"] + (qlen if row["op"] == "verify" else 0)
    page = 16 if comparator else 1
    blocks = (length + page - 1) // page
    generator = torch.Generator(device=device).manual_seed(SEED)
    buffers = []
    for index in range(layers):
        shape = (b * blocks, page, hk, dim)
        if source is None:
            q = torch.randn(
                (b * qlen, hq, dim), dtype=torch.bfloat16, device=device, generator=generator
            )
            k = torch.randn(shape, dtype=q.dtype, device=device, generator=generator)
            v = torch.randn(shape, dtype=q.dtype, device=device, generator=generator)
        else:
            source_q, source_k, source_v, _ = source[index]
            q = source_q.clone()
            k, v = (torch.zeros(shape, dtype=q.dtype, device=device) for _ in range(2))
            # Preserve request boundaries even for KV not divisible by 16.
            k.view(b, blocks * page, hk, dim)[:, :length].copy_(source_k.view(b, length, hk, dim))
            v.view(b, blocks * page, hk, dim)[:, :length].copy_(source_v.view(b, length, hk, dim))
        buffers.append((q, k, v, torch.empty_like(q)))
    return buffers


def make_vllm_call(row, buffers, calls, device):
    import torch

    b, length = row["B"], row["KV"]
    dim = MODELS[row["model"]][2]
    blocks = (length + 15) // 16
    cu_q = torch.arange(b + 1, dtype=torch.int32, device=device)
    dummy_cu_k = torch.zeros_like(cu_q)
    seqused_k = torch.full((b,), length, dtype=torch.int32, device=device)
    block_table = torch.arange(b * blocks, dtype=torch.int32, device=device).reshape(b, blocks)

    def call():
        for index in range(calls):
            q, k, v, out = buffers[index % len(buffers)]
            torch.ops._vllm_fa2_C.varlen_fwd(
                q,
                k,
                v,
                out,
                cu_q,
                dummy_cu_k,
                seqused_k,
                None,
                block_table,
                None,
                1,
                length,
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

    return call


def reference_attention(row, layer):
    """Independent FP32 math, chunked over queries to bound prefill scratch."""
    import torch

    q, k, v, _ = layer
    hq, hk, dim = MODELS[row["model"]]
    b, qlen = row["B"], row["Q"]
    length = row["KV"] + (qlen if row["op"] == "verify" else 0)
    q = q.float().view(b, qlen, hk, hq // hk, dim)
    k = k.float().view(b, length, hk, dim)
    v = v.float().view(b, length, hk, dim)
    expected = torch.empty((b, qlen, hq, dim), dtype=torch.float32, device=q.device)
    keys = torch.arange(length, device=q.device)
    for request in range(b):
        kr = k[request].permute(1, 2, 0).unsqueeze(1)
        vr = v[request].permute(1, 0, 2).unsqueeze(1)
        for start in range(0, qlen, 64):
            end = min(start + 64, qlen)
            qr = q[request, start:end].permute(1, 2, 0, 3)
            scores = (qr @ kr) * dim**-0.5
            positions = torch.arange(start, end, device=q.device) + length - qlen
            scores.masked_fill_(keys[None, :] > positions[:, None], -torch.inf)
            result = (scores.softmax(-1) @ vr).permute(2, 0, 1, 3)
            expected[request, start:end] = result.reshape(end - start, hq, dim)
    return expected.view(b * qlen, hq, dim)


def graph_capture(call):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # Eager warmup/JIT before capture on the same side stream.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    stream.synchronize()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(replay, calls):
    import torch

    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Events initialize lazily on first record. Do that before the measured
    # interval so host-side event creation cannot lengthen a short graph sample.
    begin.record()
    end.record()
    end.synchronize()
    begin.record()
    replay()
    end.record()
    end.synchronize()
    elapsed = begin.elapsed_time(end) * 1000 / calls
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid CUDA event timing")
    return elapsed


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def time_graphs(candidates, comparator, calls, args):
    """Time all variants together; each comparator sample is adjacent to its variant."""
    samples = [[] for _ in candidates]
    comparison = [[] for _ in candidates]
    replays = args.pairs if comparator is not None else 1

    def sample(graph):
        def replay():
            for _ in range(replays):
                graph.replay()

        return measure(replay, calls * replays)

    for rep in range(args.reps):
        # Rotate the starting variant to balance cache/clock drift across reps.
        for offset in range(len(candidates)):
            index = (rep + offset) % len(candidates)
            samples[index].append(sample(candidates[index]))
            if comparator is not None:
                comparison[index].append(sample(comparator))
    results = []
    for ours, theirs in zip(samples, comparison, strict=True):
        pairs = list(map(list, zip(ours, theirs, strict=True))) if theirs else []
        results.append(
            {
                "us": statistics.median(ours),
                "p10": percentile(ours, 0.1),
                "p90": percentile(ours, 0.9),
                "vllm_us": statistics.median(theirs) if theirs else None,
                "ratio_vs_vllm": statistics.median(a / b for a, b in pairs) if pairs else None,
                "pair_us": pairs,
            }
        )
    return results


def empty_result(split):
    return {
        **dict.fromkeys(
            (
                "layers",
                "calls",
                "us",
                "p10",
                "p90",
                "plan",
                "correct",
                "vllm_us",
                "ratio_vs_vllm",
                "vllm_correct",
            )
        ),
        "pair_us": [],
        "split": split,
    }


def run_row(row, regime, args, device, l2):
    import torch

    paired = row["op"] == "decode" and args.vllm_fa2 is not None
    _, hk, dim = MODELS[row["model"]]
    length = row["KV"] + (row["Q"] if row["op"] == "verify" else 0)
    kv_bytes = 2 * row["B"] * length * hk * dim * 2
    comparator_bytes = 2 * row["B"] * ((length + 15) // 16) * 16 * hk * dim * 2 if paired else 0
    layers = (
        layer_count(l2, kv_bytes, args.max_bytes, comparator_bytes) if regime == "engine" else 1
    )
    if layers * (kv_bytes + comparator_bytes) > args.max_bytes:
        raise ValueError("--max-bytes cannot hold the hot K/V buffers")
    calls = layers if regime == "engine" else args.hot_calls
    buffers = make_layers(row, layers, device)
    comparator_graph = None
    comparison = None
    if paired:
        comparison = make_layers(row, layers, device, comparator=True, source=buffers)
        comparator_call = make_vllm_call(row, comparison, calls, device)
        comparator_graph = graph_capture(comparator_call)
    expected = None
    vllm_correct = "skipped" if paired else None
    if row["B"] * row["KV"] <= 16384 * 2:
        expected = reference_attention(row, buffers[0])
        if paired:
            torch.testing.assert_close(comparison[0][3].float(), expected, atol=2e-2, rtol=2e-2)
            vllm_correct = True

    results, graphs, owners, timed_results = [], [], [], []
    for split in row_splits(row, args):
        result = empty_result(split)
        result.update(layers=layers, calls=calls, vllm_correct=vllm_correct)
        results.append(result)
        try:
            wrapper, plan, retained = make_operation(row, device, split)
            result["plan"] = plan
            # Keep independent schedule/workspace storage alive for every graph.
            owners.append((wrapper, retained))

            def candidate(wrapper=wrapper):
                if row["op"] == "verify":
                    wrapper.prepare()
                for index in range(calls):
                    q, k, v, out = buffers[index % layers]
                    wrapper.run(q=q, paged_kv_cache=(k, v), out=out)

            graph = graph_capture(candidate)
            if expected is not None:
                # qsplit still writes the original full output in query order.
                torch.testing.assert_close(buffers[0][3].float(), expected, atol=2e-2, rtol=2e-2)
            result.update(correct=True if expected is not None else "skipped", status="ok")
            graphs.append(graph)
            timed_results.append(result)
        except Infeasible as error:
            result.update(status="infeasible", error=str(error))
        except UnsupportedVersion as error:
            result.update(status="unsupported", error=str(error))
        except Exception as error:  # noqa: BLE001 -- isolate variant planning/check failures
            result.update(status="error", error=type(error).__name__)
    del expected
    # All graphs exist before any timing, with shared row K/V and stable plans.
    for _ in range(10):
        for graph in graphs:
            graph.replay()
        if comparator_graph is not None:
            comparator_graph.replay()
    torch.cuda.synchronize(device)
    timings = time_graphs(graphs, comparator_graph, calls, args)
    for result, timing in zip(timed_results, timings, strict=True):
        result.update(timing)
    torch.cuda.synchronize(device)
    return results


def main(argv=None):
    args = parse_args(argv)
    host = dict.fromkeys(("gpu", "sms", "l2", "flashinfer", "torch", "vllm_so_sha256"))
    setup_error = None
    comparator_error = None
    torch_module = None
    try:
        with provider_output():
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
                    host["vllm_so_sha256"] = library_sha256(args.vllm_fa2)
                    torch.ops.load_library(str(args.vllm_fa2))
                except Exception as error:  # noqa: BLE001 -- report setup failures as JSONL
                    comparator_error = type(error).__name__
    except Exception as error:  # noqa: BLE001 -- keep worker output machine-readable
        setup_error = type(error).__name__
    emit({"type": "host", **host})
    for row in generate_rows(args):
        for regime in args.regimes:
            results = [empty_result(split) for split in row_splits(row, args)]
            error_text = setup_error or (comparator_error if row["op"] == "decode" else None)
            if error_text:
                for result in results:
                    result.update(status="error", error=error_text)
            else:
                try:
                    with provider_output(), torch.inference_mode():
                        results = run_row(row, regime, args, device, host["l2"])
                except Exception as error:  # noqa: BLE001 -- isolate failures to one row/regime
                    for result in results:
                        result.update(status="error", error=type(error).__name__)
            for result in results:
                emit({"type": "row", **row, "regime": regime, **result})
            # The run_row frame (and any exception traceback) is gone before GC.
            gc.collect()
            if torch_module is not None and torch_module.cuda.is_initialized():
                with provider_output():
                    torch_module.cuda.empty_cache()


if __name__ == "__main__":
    main()
