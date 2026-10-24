# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DRAFT access probes with standalone exact-size CUDA memcheck storage."""

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ctikernels.attention.inference import CacheTableDraftAttention
from tests._dflash_reference import draft_attention_reference
from tests._dflash_table_reference import table_csr
from tests.gpu._dflash import Fixture

from .dflash_draft_capture_trace import diagnostic_stdout


def exact_storage(tensors):
    assert len({value.data_ptr() for value in tensors}) == len(tensors)
    assert all(
        value.storage_offset() == 0
        and value.untyped_storage().nbytes() == value.numel() * value.element_size()
        for value in tensors
    )


def provider_record(op):
    return {
        "provider": op.provider,
        "tuning": {
            name: getattr(op._implementation, name)
            for name in (
                "splits",
                "key_tile",
                "stages",
                "key_groups",
                "short_union",
                "inline_merge",
                "min_chunk",
                "chunk_cap",
                "merge_fan_in",
                "counter_bytes",
            )
        }
        if op.provider in ("cute-splitkv", "cute-table")
        else None,
    }


@contextlib.contextmanager
def split_setting(splits):
    previous = os.environ.get("CTI_DRAFT_SPLITS")
    try:
        if splits is None:
            os.environ.pop("CTI_DRAFT_SPLITS", None)
        else:
            os.environ["CTI_DRAFT_SPLITS"] = str(splits)
        yield
    finally:
        if previous is None:
            os.environ.pop("CTI_DRAFT_SPLITS", None)
        else:
            os.environ["CTI_DRAFT_SPLITS"] = previous


@contextlib.contextmanager
def case_result(record):
    record.update(type="access_case", status="FAIL", ok=False)
    try:
        yield
    except (RuntimeError, ValueError, OSError, ImportError, AssertionError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    else:
        record.update(status="PASS", ok=True)


class CuteAccessFixture(Fixture):
    """Reuse paged staging, replacing the fixture's guarded output with exact storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = torch.empty_like(self.query)
        del self.out_backing
        counters = self.op._implementation._counters
        exact_storage(
            [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
            + self.workspaces
            + ([counters] if counters.numel() else [])
        )

    def check_access_outputs(self):
        expected = draft_attention_reference(
            self.query.cpu(),
            self.k.cpu(),
            self.v.cpu(),
            slots=self.cpu_slots,
            positions=self.cpu_positions,
            token_indptr=self.cpu_token_ptr,
            token_indices=self.cpu_token_idx,
            query_length=self.q,
            causal=self.causal,
            window=self.window,
        )
        actual = self.out.cpu().float()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        for row, slot in enumerate(self.cpu_slots.tolist()):
            if slot == -1:
                assert torch.count_nonzero(actual[row * self.q : (row + 1) * self.q]) == 0
        # Both entry points refresh the engine CSR (fused: from the attention CTAs).
        for name, value, reference in (
            ("query_indptr", self.query_ptr, self.expected_query_ptr),
            ("page_indptr", self.page_ptr, self.expected_page_ptr),
            ("page_indices", self.page_idx, self.expected_pages),
            ("last_page_lengths", self.last, self.expected_last),
        ):
            torch.testing.assert_close(value.cpu(), reference, atol=0, rtol=0, msg=name)

    def call(self, entry):
        if entry == "fused":
            return self.op.prepare_and_run(
                q=self.query, paged_kv_cache=(self.k, self.v), out=self.out
            )
        return self.run()


def cute_matrix():
    # 36 DRAFT constructions: every Q/mask/split combination, other axes sampled.
    # Three length replays each plus nine all-inactive replays = 117 DRAFT cases.
    # Entry points alternate so each Q/mask runs both prepare(); run() and the fused call.
    for qi, q in enumerate((4, 9, 16)):
        for mi, (causal, window) in enumerate(((False, None), (True, 4096), (False, 2048))):
            for si, splits in enumerate((1, 2, 7, None)):
                variant = (qi * 3 + mi + si) % 8
                hq, hk = (16, 4) if variant & 2 else (32, 8)
                yield {
                    "q": q,
                    "causal": causal,
                    "window": window,
                    "b": 3 if variant & 1 else 1,
                    "num_qo_heads": hq,
                    "num_kv_heads": hk,
                    "page_size": 16 if variant & 4 else 1,
                    "entry": "fused" if (qi + mi + si) % 2 else "two_call",
                    "requested_splits": splits,
                }

    # Four additive VERIFY constructions: both query widths, pages and entry points.
    for q in (4, 16):
        for page, entry, splits in ((1, "two_call", 2), (16, "fused", None)):
            yield {
                "q": q,
                "causal": True,
                "window": None,
                "b": 3,
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "page_size": page,
                "entry": entry,
                "requested_splits": splits,
            }


def large_capacity_matrix():
    # Full-attention capacity above the widest window union: occupancy-sized splits,
    # the count-aware merge, and (with a 256-key index buffer) the piece loop.
    for q, b, heads, page, entry, cap in (
        (16, 3, (32, 8), 1, "fused", None),
        (9, 1, (16, 4), 16, "two_call", 256),
        (4, 3, (32, 8), 16, "fused", 256),
    ):
        yield {
            "q": q,
            "causal": False,
            "window": None,
            "b": b,
            "num_qo_heads": heads[0],
            "num_kv_heads": heads[1],
            "page_size": page,
            "entry": entry,
            "requested_splits": None,
            "chunk_cap": cap,
            "capacity": 8192,
        }


@contextlib.contextmanager
def chunk_cap_setting(cap):
    previous = os.environ.get("CTI_DRAFT_CHUNK_CAP")
    try:
        if cap is not None:
            os.environ["CTI_DRAFT_CHUNK_CAP"] = str(cap)
        yield
    finally:
        if previous is None:
            os.environ.pop("CTI_DRAFT_CHUNK_CAP", None)
        else:
            os.environ["CTI_DRAFT_CHUNK_CAP"] = previous


def cute_access_cases(provider):
    for configuration in [*cute_matrix(), *large_capacity_matrix()]:
        q, b, window = (configuration[name] for name in ("q", "b", "window"))
        lengths = [q + 1, q + 17, q + 33] if window is None else [window - 1, window, window + 1]
        if "capacity" in configuration:
            lengths = [q + 1, 1500, 5000]
        # Sizing and construction must see the same override; execution must not.
        with (
            split_setting(configuration["requested_splits"]),
            chunk_cap_setting(configuration.get("chunk_cap")),
        ):
            f = CuteAccessFixture(
                b,
                q,
                configuration["page_size"],
                configuration["num_qo_heads"],
                configuration["num_kv_heads"],
                configuration["causal"],
                window,
                c=configuration.get("capacity", max(lengths)),
                provider=provider,
            )
        assert f.op.provider == "cute-splitkv"
        # Warm up and capture once, then change lengths, physical pages and slots.
        f.stage([0] * b, list(range(b)))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            f.call(configuration["entry"])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            f.call(configuration["entry"])
        torch.cuda.current_stream().wait_stream(stream)
        patterns = [(length, list(range(b))) for length in lengths]
        if b > 1:
            patterns[1][1][0] = -1
            patterns[2][1][-1] = -1
        if configuration["requested_splits"] is None:
            patterns.append((lengths[-1], [-1] * b))
        for turn, (length, slots) in enumerate(patterns):
            record = {
                **configuration,
                **provider_record(f.op),
                "suite": "cute_matrix",
                "capacity": f.c,
                "lengths": [length if slot != -1 else 0 for slot in slots],
                "slots": slots,
                "graph_replay": True,
            }
            with case_result(record):
                f.stage([length - q] * b, slots, turn)
                graph.replay()
                torch.cuda.synchronize()
                f.check_access_outputs()
            yield record


TABLE_BOUND_SKIP = (
    "M + Q near 2**31 - 1: skipped; the public interface requires a real table "
    "with at least M int32 columns (about 8 GiB per row), incompatible with "
    "a probe that avoids allocating that table."
)


def table_matrix():
    # 32 constructions, 190 executions. Cross providers/metadata/masks/widths;
    # sample B/Q/head geometry, and cover every frontier path per construction.
    for provider in ("cute", "flashinfer"):
        for mi, metadata in enumerate(("request", "row")):
            for wi, (causal, window) in enumerate(
                ((False, None), (True, None), (True, 4096), (False, 2048))
            ):
                for padding in (0, 17):
                    variant = mi * 8 + wi * 2 + int(bool(padding))
                    hq, hk, dim = ((32, 8, 128), (16, 4, 256), (8, 1, 256))[
                        (variant + variant // 3) % 3
                    ]
                    yield {
                        "requested_provider": provider,
                        "metadata_index": metadata,
                        "causal": causal,
                        "window": window,
                        "max_seq_len": 4101,
                        "table_width": 4101 + padding,
                        "b": (1, 3, 4)[variant % 3],
                        "q": (4, 9, 16)[(variant // 3) % 3],
                        "num_qo_heads": hq,
                        "num_kv_heads": hk,
                        "head_dim": dim,
                    }


def table_patterns(configuration):
    b, q, maximum = (configuration[name] for name in ("b", "q", "max_seq_len"))
    boundary = configuration["window"] or 2048
    states = [
        (1, False),
        (maximum - 1, False),
        (maximum - 1, True),
        (-1, False),
        (maximum, False),
        (2**62, False),
        (-(2**62), False),
        (boundary - q - 1, False),
        (boundary - q, False),
        (boundary - q + 1, False),
    ]
    # Wrap the final group: for B=1,3,4 the spill state also lands in the last
    # execution row, whose request always selects the final physical table row.
    for start in range(0, len(states), b):
        yield [states[(start + row) % len(states)] for row in range(b)]


def table_host_inputs(configuration, pattern):
    """CPU staging shared with matrix tests; padding and unused rows stay poisoned."""
    b, q, maximum, width = (
        configuration[name] for name in ("b", "q", "max_seq_len", "table_width")
    )
    requests = torch.arange(1, 2 * b + 1, 2, dtype=torch.int32)
    table_rows = int(requests[-1]) + 1
    # Request metadata deliberately has one more row than the mapping, and > B.
    count = table_rows + 1 if configuration["metadata_index"] == "request" else b
    frontier = torch.full((count,), -(2**62), dtype=torch.int64)
    done = torch.ones(count, dtype=torch.bool)
    table = torch.full((table_rows, width), 2**31 - 1, dtype=torch.int32)
    # Dense width is also the row stride: no allocator tail after the final W.
    # For W > M the in-allocation padding stays poisoned, just like unused keys.
    guard_start = maximum + 37
    for row, (f, is_done) in enumerate(pattern):
        request = int(requests[row])
        metadata = request if configuration["metadata_index"] == "request" else row
        frontier[metadata], done[metadata] = f, is_done
        if not is_done and 0 <= f < maximum:
            prefix = min(f + q, maximum)
            table[request, :prefix] = (
                torch.arange(prefix, dtype=torch.int32) * 7 + request * 13
            ) % guard_start
    return requests, frontier, done, table


class TableAccessFixture:
    def __init__(self, configuration):
        self.configuration = configuration
        b, q, maximum, hq, hk, dim = (
            configuration[name]
            for name in ("b", "q", "max_seq_len", "num_qo_heads", "num_kv_heads", "head_dim")
        )
        self.guard_start = maximum + 37
        self.host = table_host_inputs(configuration, next(table_patterns(configuration)))
        self.metadata = [value.to("cuda") for value in self.host]
        generator = torch.Generator().manual_seed(1739)
        self.cpu_query, self.cpu_k, self.cpu_v = [
            torch.randn(shape, dtype=torch.bfloat16, generator=generator)
            for shape in (
                (b * q, hq, dim),
                (self.guard_start + b * q, 1, hk, dim),
                (self.guard_start + b * q, 1, hk, dim),
            )
        ]
        self.query, self.k, self.v = [
            value.to("cuda") for value in (self.cpu_query, self.cpu_k, self.cpu_v)
        ]
        self.out = torch.empty_like(self.query)
        geometry = {
            "query_length": q,
            "max_seq_len": maximum,
            "num_qo_heads": hq,
            "num_kv_heads": hk,
            "head_dim": dim,
            "dtype": torch.bfloat16,
            "causal": configuration["causal"],
            "window": configuration["window"],
            "provider": configuration["requested_provider"],
        }
        sizes = CacheTableDraftAttention.workspace_size(
            batch_size=b, device=self.query.device, **geometry
        )
        self.workspaces = [torch.empty(size, dtype=torch.uint8, device="cuda") for size in sizes]
        requests, frontier, done, table = self.metadata
        self.op = CacheTableDraftAttention(
            *self.workspaces,
            request_indices=requests,
            frontier_table=frontier,
            done_table=done,
            cache_index_table=table,
            guard_start=self.guard_start,
            metadata_index=configuration["metadata_index"],
            **geometry,
        )
        expected_provider = (
            "cute-table"
            if configuration["requested_provider"] == "cute"
            else "flashinfer-fa2-table-csr"
        )
        assert self.op.provider == expected_provider
        tensors = [*self.metadata, self.query, self.k, self.v, self.out, *self.workspaces]
        if self.op.provider == "cute-table":
            tensors.append(self.op._implementation._counters)
        exact_storage([value for value in tensors if value.numel()])

    def run(self, pattern):
        self.host = table_host_inputs(self.configuration, pattern)
        for device, host in zip(self.metadata, self.host, strict=True):
            device.copy_(host)
        # A missing output store must fail parity even after a successful prior call.
        self.out.fill_(float("nan"))
        self.op.run(q=self.query, paged_kv_cache=(self.k, self.v), out=self.out)
        torch.cuda.synchronize()
        requests, frontier, done, table = self.host
        if self.configuration["metadata_index"] == "request":
            frontier, done = frontier[requests.long()], done[requests.long()]
        b, q = self.configuration["b"], self.configuration["q"]
        slots = torch.arange(b, dtype=torch.int32)
        ptr, indices, positions = table_csr(
            slots,
            frontier,
            done,
            table[requests.long()],
            self.guard_start,
            self.configuration["max_seq_len"],
            q,
        )
        expected = draft_attention_reference(
            self.cpu_query,
            self.cpu_k,
            self.cpu_v,
            slots=slots,
            positions=positions,
            token_indptr=ptr,
            token_indices=indices,
            query_length=q,
            causal=self.configuration["causal"],
            window=self.configuration["window"],
        )
        torch.testing.assert_close(self.out.cpu().float(), expected, atol=2e-2, rtol=2e-2)


def table_access_cases():
    for configuration in table_matrix():
        fixture = TableAccessFixture(configuration)
        for pattern in table_patterns(configuration):
            record = {
                **configuration,
                **provider_record(fixture.op),
                "suite": "table_matrix",
                "frontiers": [f for f, _ in pattern],
                "done": [done for _, done in pattern],
                "graph_replay": False,
            }
            with case_result(record):
                fixture.run(pattern)
            yield record


def access_check(provider="cute"):
    cases, error = [], None
    iterator = table_access_cases() if provider == "table" else cute_access_cases(provider)
    while True:
        try:
            with diagnostic_stdout():
                record = next(iterator)
        except StopIteration:
            break
        except (RuntimeError, ValueError, OSError, ImportError, AssertionError) as failure:
            error = f"{type(failure).__name__}: {failure}"
            break
        cases.append(record)
        print(json.dumps(record), flush=True)
    passed = sum(case["ok"] for case in cases)
    return {
        "type": "access_check",
        "cases": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "standalone_exact_storage": True,
        "caching_disabled": os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1",
        "error": error,
        **({"skipped": [TABLE_BOUND_SKIP]} if provider == "table" else {}),
        "ok": bool(cases) and passed == len(cases) and error is None,
    }


def find_sanitizer():
    found = shutil.which("compute-sanitizer")
    if found:
        return Path(found)
    roots = [Path("/usr/local/cuda")]
    if os.environ.get("CUDA_HOME"):
        roots.insert(0, Path(os.environ["CUDA_HOME"]))
    for root in roots:
        candidate = root / "bin" / "compute-sanitizer"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("compute-sanitizer is unavailable")


def launch_sanitizer(provider="cute"):
    binary = find_sanitizer()
    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    match = re.search(r"[Vv]ersion\s+([\d.]+)", version.stdout + version.stderr)
    env = os.environ.copy()
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    options = [
        "--tool",
        "memcheck",
        "--padding",
        "4096",
        "--print-limit",
        "50",
        "--error-exitcode",
        "1",
        "--report-api-errors",
        "no",
    ]
    command = [
        str(binary),
        *options,
        sys.executable,
        "-m",
        "benchmark.kernel_tuning.dflash_draft_access_check",
        "--provider",
        provider,
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    # Forward only our structured receipt; package/sanitizer output can contain paths.
    child_ok, counts = False, {}
    for line in result.stdout.splitlines():
        try:
            receipt = json.loads(line)
        except ValueError:
            continue
        if isinstance(receipt, dict) and receipt.get("type") in ("access_case", "access_check"):
            print(json.dumps(receipt), flush=True)
            if receipt["type"] == "access_check":
                child_ok = receipt.get("ok") is True and receipt.get("caching_disabled") is True
                counts = {name: receipt.get(name) for name in ("cases", "passed", "failed")}
    summaries = re.findall(r"ERROR SUMMARY: (\d+) errors?", result.stdout + result.stderr)
    errors = max(map(int, summaries)) if summaries else None
    ok = result.returncode == 0 and errors == 0 and child_ok and match is not None
    print(
        json.dumps(
            {
                "type": "sanitizer",
                "sanitizer": binary.name,
                "version": match.group(1) if match else None,
                "options": options,
                "requested_provider": provider,
                **counts,
                "error_summary": f"ERROR SUMMARY: {errors} errors" if errors is not None else None,
                "rc": result.returncode,
                "ok": ok,
            }
        ),
        flush=True,
    )
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-sanitizer", action="store_true")
    parser.add_argument("--provider", choices=("cute", "table"), default="cute")
    args = parser.parse_args()
    if args.launch_sanitizer:
        return launch_sanitizer(args.provider)
    result = access_check(args.provider)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
