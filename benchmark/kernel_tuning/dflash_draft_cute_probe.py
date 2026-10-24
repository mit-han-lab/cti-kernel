# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Strict oracle checks, single-call and amortized CUDA graph timings.

Run from the repository root with python -m benchmark.kernel_tuning.dflash_draft_cute_probe.
Fixture imports only torch and the independent oracle; this probe needs no pytest.
"""

import argparse
import contextlib
import itertools
import json
import os
import statistics
import sys

import torch

from tests.gpu._dflash import Fixture

ATOL = RTOL = 2e-2


def graph_latency(call, warmup, *, calls=1, replays=200):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(replays)
    ]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) * 1000 / calls for start, end in events)


def fixed_plan(f, hq, hk):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(f.query_ptr),
        paged_kv_indptr_buf=torch.empty_like(f.page_ptr),
        paged_kv_indices_buf=torch.empty_like(f.page_idx),
        paged_kv_last_page_len_buf=torch.empty_like(f.last),
    )
    wrapper.plan(
        qo_indptr=f.expected_query_ptr,
        paged_kv_indptr=f.expected_page_ptr,
        paged_kv_indices=f.expected_pages,
        paged_kv_last_page_len=f.expected_last,
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=128,
        page_size=f.p,
        causal=f.causal,
        window_left=-1 if f.window is None else f.window - 1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        disable_split_kv=False,
        non_blocking=False,
    )
    return wrapper


def tuning():
    return {
        "splits_override": os.environ.get("CTI_DRAFT_SPLITS"),
        "min_chunk": int(os.environ.get("CTI_DRAFT_MIN_CHUNK", "128")),
        "target_ctas": int(
            os.environ.get(
                "CTI_DRAFT_TARGET_CTAS",
                str(2 * torch.cuda.get_device_properties(0).multi_processor_count),
            )
        ),
        "key_tile": int(os.environ.get("CTI_DRAFT_KEY_TILE", "32")),
        "key_groups": int(os.environ.get("CTI_DRAFT_KEY_GROUPS", "2")),
        "stages": int(os.environ.get("CTI_DRAFT_STAGES", "3")),
    }


def sweep_shape(b, length, hq, hk, q, causal, window, page, warmup):
    # One oracle and one stock fixed-plan timing per shape, shared by all 48 candidates.
    with contextlib.redirect_stdout(sys.stderr):
        stock = Fixture(b, q, page, hq, hk, causal, window, c=length, provider="flashinfer")
        stock.stage([length - q] * b, list(range(b)))
        expected = stock.oracle()
        fixed = fixed_plan(stock, hq, hk)

        def stock_call():
            fixed.run(q=stock.query, paged_kv_cache=(stock.k, stock.v), out=stock.out)

        stock_call()
        torch.testing.assert_close(stock.out.float(), expected, atol=ATOL, rtol=RTOL)
        stock_us = graph_latency(stock_call, warmup, calls=20, replays=15)
        sms = torch.cuda.get_device_properties(0).multi_processor_count
    # CTI_SWEEP_GRID='[[64,128],[1,2],[32],[1,2],[2]]' narrows the grid for long shapes.
    grid = json.loads(
        os.environ.get("CTI_SWEEP_GRID", "[[64, 128, 256], [1, 2], [32, 64], [1, 2], [2, 3]]")
    )
    for minimum, multiple, key_tile, groups, stages in itertools.product(*grid):
        for name, value in (
            ("MIN_CHUNK", minimum),
            ("TARGET_CTAS", multiple * sms),
            ("KEY_TILE", key_tile),
            ("KEY_GROUPS", groups),
            ("STAGES", stages),
        ):
            os.environ[f"CTI_DRAFT_{name}"] = str(value)
        with contextlib.redirect_stdout(sys.stderr):
            f = Fixture(b, q, page, hq, hk, causal, window, c=length, provider="cute")
            f.stage([length - q] * b, list(range(b)))

            def run(f=f):
                f.op.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

            def complete(f=f, run=run):
                f.op.prepare()
                run()

            complete()
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
            run_us = graph_latency(run, warmup, calls=20, replays=15)
            complete_us = graph_latency(complete, warmup, calls=20, replays=15)
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
        print(
            json.dumps(
                {
                    "b": b,
                    "length": length,
                    "hq": hq,
                    "hk": hk,
                    "q": q,
                    "causal": causal,
                    "window": window,
                    "page": page,
                    **tuning(),
                    "splits": f.op._implementation.splits,
                    "cute_run_only_us": run_us,
                    "cute_prepare_run_us": complete_us,
                    "flashinfer_fixed_plan_run_us": stock_us,
                    "amortized_calls_per_graph": 20,
                    "amortized_replays": 15,
                    "warmup": warmup,
                }
            ),
            flush=True,
        )


def probe(b, length, hq, hk, q, causal, window, page, warmup):
    # Diagnostics from provider compilation belong on stderr, preserving JSON stdout.
    with contextlib.redirect_stdout(sys.stderr):
        fixtures = [
            Fixture(b, q, page, hq, hk, causal, window, c=length, provider=provider)
            for provider in ("cute", "flashinfer")
        ]
        for f in fixtures:
            f.stage([length - q] * b, list(range(b)))
        expected = fixtures[0].oracle()
        fixed = fixed_plan(fixtures[1], hq, hk)
        results = []
        for f, name, prepare in (
            (fixtures[0], "cute_prepare_run", True),
            (fixtures[0], "cute_run_only", False),
            (fixtures[1], "flashinfer_prepare_run", True),
            (fixtures[1], "flashinfer_run_only", False),
            (fixtures[1], "flashinfer_fixed_plan_run", False),
        ):
            runner = fixed if name == "flashinfer_fixed_plan_run" else f.op

            def call(f=f, prepare=prepare, runner=runner):
                if prepare:
                    runner.prepare()
                runner.run(q=f.query, paged_kv_cache=(f.k, f.v), out=f.out)

            f.out.fill_(float("nan"))
            call()
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
            error = (f.out.float() - expected).abs()
            maximum = error.max().item()
            scaled = (error / (ATOL + RTOL * expected.abs())).max().item()
            latency = graph_latency(call, warmup)
            amortized = graph_latency(call, warmup, calls=20, replays=50)
            torch.testing.assert_close(f.out.float(), expected, atol=ATOL, rtol=RTOL)
            results.append(
                {
                    "operation": name,
                    "provider": "stock_flashinfer" if runner is fixed else f.op.provider,
                    "splits": getattr(f.op._implementation, "splits", None),
                    "max_abs_error": maximum,
                    "max_tolerance_fraction": scaled,
                    "graph_median_us": latency,
                    "replays": 200,
                    "amortized_graph_median_us": amortized,
                    "amortized_calls_per_graph": 20,
                    "amortized_replays": 50,
                    "warmup": warmup,
                }
            )
    geometry = {
        "b": b,
        "length": length,
        "q": q,
        "hq": hq,
        "hk": hk,
        "page": page,
        "causal": causal,
        "window": window,
    }
    for result in results:
        print(json.dumps(geometry | tuning() | result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 1024, 4096])
    parser.add_argument("--page-size", type=int, choices=[1, 16], default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--splits", type=int)
    parser.add_argument("--min-chunk", type=int)
    parser.add_argument("--key-tile", type=int, choices=[32, 64])
    parser.add_argument("--key-groups", type=int, choices=[1, 2])
    parser.add_argument("--stages", type=int, choices=[2, 3])
    parser.add_argument(
        "--target-ctas", type=int, help="absolute CTA target (not an SM multiplier)"
    )
    parser.add_argument("--sweep", action="store_true", help="48 candidates per probe shape")
    parser.add_argument(
        "--queries", nargs="+", type=int, default=[8, 16], help="run every mask at each Q"
    )
    args = parser.parse_args()
    for name, value in (
        ("SPLITS", args.splits),
        ("MIN_CHUNK", args.min_chunk),
        ("TARGET_CTAS", args.target_ctas),
        ("KEY_TILE", args.key_tile),
        ("KEY_GROUPS", args.key_groups),
        ("STAGES", args.stages),
    ):
        if value is not None:
            if value < 1:
                parser.error(f"--{name.lower().replace('_', '-')} must be positive")
            os.environ[f"CTI_DRAFT_{name}"] = str(value)
    if args.sweep and any(
        value is not None
        for value in (
            args.splits,
            args.min_chunk,
            args.target_ctas,
            args.key_tile,
            args.key_groups,
            args.stages,
        )
    ):
        parser.error("--sweep supplies its own tuning grid; omit tuning overrides")
    if args.warmup < 1 or any(b not in range(1, 5) for b in args.b):
        parser.error("warmup must be positive and batch sizes must be 1..4")
    if min(args.lengths) < 16:
        parser.error("lengths must be at least 16")
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.sweep:
        os.environ.pop("CTI_DRAFT_SPLITS", None)
    masks = [(16, False, None), (16, True, 4096), (8, False, 2048)]
    if args.queries:
        masks = [(q, causal, window) for q in args.queries for _, causal, window in masks]
    for b, length, (hq, hk), (q, causal, window) in itertools.product(
        args.b, args.lengths, [(32, 8), (16, 4)], masks
    ):
        if args.sweep:
            sweep_shape(b, length, hq, hk, q, causal, window, args.page_size, min(args.warmup, 3))
        else:
            probe(b, length, hq, hk, q, causal, window, args.page_size, args.warmup)


if __name__ == "__main__":
    main()
