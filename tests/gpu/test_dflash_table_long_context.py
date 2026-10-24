# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""131K/262K CacheTableDraftAttention and FixedQuery correctness with bounded-memory FP32 references.

Each GPU row covers eager execution and mutable-metadata CUDA graph replay.
With M=C-Q, the table contract admits F<M: its longest live length is C-1,
including Q-1 guard keys. F=M is separately checked as a guard-only done row.
FixedQuery admits the full C keys. B=1/4 are core; B=2/3 are exhaustive.
Table geometries explicitly cover both CuTe and FlashInfer providers.
"""

import sys
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ctikernels.attention import _flashinfer, draft
from ctikernels.attention._draft import flashinfer
from ctikernels.attention.inference import CacheTableDraftAttention, FlashInferFixedQueryAttention

from ._exhaustive import ci_cases

# Same BF16 tolerance as test_dflash_draft_attention.py; FixedQuery uses 1e-2.
ATOL = RTOL = 2e-2
CHUNK = 16384
CAPACITIES = (131072, 262144)
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@dataclass(frozen=True)
class Geometry:
    interface: str
    q: int
    hq: int
    hk: int
    dim: int
    causal: bool
    window: int | None = None

    def id(self, provider):
        mask = "causal" if self.causal else "full"
        if self.window is not None:
            mask += f"-w{self.window}"
        interface = f"table-{provider}" if self.interface == "table" else self.interface
        return f"{interface}-q{self.q}-{mask}-{self.hq}x{self.hk}-d{self.dim}"

    def arguments(self):
        return {
            "query_length": self.q,
            "num_qo_heads": self.hq,
            "num_kv_heads": self.hk,
            "head_dim": self.dim,
        }


GEOMETRIES = (
    Geometry("table", 16, 32, 8, 128, False),
    Geometry("table", 16, 32, 8, 128, True, 4096),
    Geometry("table", 8, 32, 8, 128, False, 2048),
    Geometry("table", 16, 32, 8, 128, True),
    Geometry("table", 16, 16, 4, 256, True),
    Geometry("fixed", 16, 32, 8, 128, True),
    Geometry("fixed", 16, 16, 4, 256, True),
)
GEOMETRY_PROVIDERS = tuple(
    (geometry, provider)
    for geometry in GEOMETRIES
    for provider in (("cute", "flashinfer") if geometry.interface == "table" else (None,))
)


def chunked_reference(query, cache, key_rows, *, causal, window, chunk=CHUNK):
    """Online softmax per request/KV group; never expand KV over query heads.

    key_rows is an independently constructed logical-to-physical key list per
    request. FP32 temporaries are O(chunk * (D + Q * Hq/Hkv)), under 64 MiB
    for these geometries, regardless of engine capacity or batch size.
    """
    k, v = cache
    b, hk = len(key_rows), k.shape[2]
    q, hq, dim = query.shape[0] // b, query.shape[1], query.shape[2]
    group = hq // hk
    output = torch.empty_like(query, dtype=torch.float32)
    for row, indices in enumerate(key_rows):
        length = indices.numel()
        positions = length - q + torch.arange(q, device=query.device)
        # Entire chunks before this union are masked for every query.
        start = 0 if window is None else max(0, length - q - window + 1)
        for head in range(hk):
            heads = slice(head * group, (head + 1) * group)
            queries = query[row * q : (row + 1) * q, heads].float().transpose(0, 1)
            maximum = torch.full(
                (group, q, 1), -torch.inf, dtype=torch.float32, device=query.device
            )
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((group, q, dim), dtype=torch.float32, device=query.device)
            for lo in range(start, length, chunk):
                hi = min(lo + chunk, length)
                keys = torch.arange(lo, hi, device=query.device)
                allowed = torch.ones((q, hi - lo), dtype=torch.bool, device=query.device)
                if causal:
                    allowed &= keys[None, :] <= positions[:, None]
                if window is not None:
                    allowed &= (keys[None, :] - positions[:, None]).abs() < window
                physical = indices[lo:hi]
                scores = (queries @ k[physical, 0, head].float().T) * dim**-0.5
                scores.masked_fill_(~allowed[None, :, :], -torch.inf)
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
                # Fully masked early chunks contribute zero, including when both
                # running and chunk maxima are -inf (avoid -inf - -inf).
                safe_maximum = next_maximum.masked_fill(~next_maximum.isfinite(), 0)
                correction = (maximum - safe_maximum).exp()
                weights = (scores - safe_maximum).exp()
                denominator = denominator * correction + weights.sum(dim=-1, keepdim=True)
                numerator = numerator * correction + weights @ v[physical, 0, head].float()
                maximum = next_maximum
            output[row * q : (row + 1) * q, heads] = (numerator / denominator).transpose(0, 1)
    return output


class LongContextCase:
    def __init__(self, geometry, provider, capacity, batch):
        self.g, self.c, self.b = geometry, capacity, batch
        g, c, b = geometry, capacity, batch
        self.maximum = c - g.q
        self.guard = b * c
        top = c - 1 if g.interface == "table" else c
        self.lengths = (
            [top, c // 2 + 13, 4096 + 7, 100][:b],
            list(reversed([top - 37, c // 2 + 31, 4096 - 23, 129][:b])),
        )
        if b == 1:
            self.lengths[1][0] = c // 2 + 31
        self.requests = (list(range(b)), list(reversed(range(b))))
        self.query = torch.empty((b * g.q, g.hq, g.dim), dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty_like(self.query)
        self.workspaces = []
        arguments = g.arguments()
        if g.interface == "table":
            self.slots = torch.empty(b, dtype=torch.int32, device="cuda")
            self.frontier = torch.empty(b, dtype=torch.int64, device="cuda")
            self.done = torch.zeros(b, dtype=torch.bool, device="cuda")
            self.table = (
                torch.arange(b, dtype=torch.int32, device="cuda")[:, None] * c
                + torch.arange(self.maximum, dtype=torch.int32, device="cuda")[None, :]
            )
            arguments.update(
                max_seq_len=self.maximum,
                dtype=torch.bfloat16,
                causal=g.causal,
                window=g.window,
                provider=provider,
            )
            sizes = CacheTableDraftAttention.workspace_size(
                batch_size=b, device="cuda", **arguments
            )
            self.workspaces = [torch.zeros(n, dtype=torch.uint8, device="cuda") for n in sizes]
            self.op = CacheTableDraftAttention(
                *self.workspaces,
                request_indices=self.slots,
                frontier_table=self.frontier,
                done_table=self.done,
                cache_index_table=self.table,
                guard_start=self.guard,
                metadata_index="row",
                **arguments,
            )
            assert self.op.provider.startswith(provider)
        else:
            self.query_ptr = torch.empty(b + 1, dtype=torch.int32, device="cuda")
            self.kv_ptr = torch.empty_like(self.query_ptr)
            self.indices = torch.empty(b * c, dtype=torch.int32, device="cuda")
            self.last = torch.empty(b, dtype=torch.int32, device="cuda")
            sizes = FlashInferFixedQueryAttention.workspace_size(
                batch_size=b, device=self.query.device, **arguments
            )
            self.workspaces = [torch.zeros(n, dtype=torch.uint8, device="cuda") for n in sizes]
            self.op = FlashInferFixedQueryAttention(
                *self.workspaces,
                query_indptr=self.query_ptr,
                kv_indptr=self.kv_ptr,
                kv_indices=self.indices,
                last_page_lengths=self.last,
                page_size=1,
                dtype=torch.bfloat16,
                causal=True,
                **arguments,
            )

        rows = self.guard + (b * g.q if g.interface == "table" else 0)
        self.cache = tuple(
            torch.empty((rows, 1, g.hk, g.dim), dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        )
        generator = torch.Generator(device="cuda").manual_seed(7163 + c + b + g.dim + g.q)
        self.query.normal_(generator=generator)
        # Allocate capacity once, but initialize only prefixes used by either
        # replay. No CPU cache mirror, repeated KV heads, or full-cache clone.
        needed = [0] * b
        for requests, lengths in zip(self.requests, self.lengths, strict=True):
            for request, length in zip(requests, lengths, strict=True):
                needed[request] = max(needed[request], length)
        for value in self.cache:
            for request, length in enumerate(needed):
                count = min(length, self.maximum) if g.interface == "table" else length
                for lo in range(0, count, CHUNK):
                    value[request * c + lo : request * c + min(count, lo + CHUNK)].normal_(
                        generator=generator
                    )
            if g.interface == "table":
                value[self.guard :].normal_(generator=generator)
        # Largest configuration: just over 4 GiB KV, plus small metadata/scratch.
        owned = (*self.cache, self.query, self.out, *self.workspaces)
        assert sum(t.numel() * t.element_size() for t in owned) < 10 * 1024**3

    def stage(self, turn, *, boundary=False):
        g = self.g
        requests, lengths = self.requests[turn], self.lengths[turn]
        keys = []
        for row, (request, length) in enumerate(zip(requests, lengths, strict=True)):
            if boundary and row == 0:
                physical = self.guard + torch.arange(g.q, device="cuda")
            else:
                logical = torch.arange(length, device="cuda")
                physical = request * self.c + logical
                if g.interface == "table":
                    physical = torch.where(
                        logical < self.maximum,
                        physical,
                        self.guard + row * g.q + logical - self.maximum,
                    )
            keys.append(physical)
        if g.interface == "table":
            frontiers = [length - g.q for length in lengths]
            if boundary:
                frontiers[0] = self.maximum
            self.slots.copy_(torch.tensor(requests, dtype=torch.int32))
            self.frontier.copy_(torch.tensor(frontiers, dtype=torch.int64))
        else:
            ptr = [0]
            for length in lengths:
                ptr.append(ptr[-1] + length)
            self.kv_ptr.copy_(torch.tensor(ptr, dtype=torch.int32))
            self.indices[: ptr[-1]].copy_(torch.cat(keys))
            self.last.fill_(1)
        return keys

    def run(self):
        if self.g.interface == "fixed":
            # FixedQuery's mutable schedule must be refreshed inside the graph.
            self.op.prepare()
        return self.op.run(q=self.query, paged_kv_cache=self.cache, out=self.out)

    def reference(self, keys):
        return chunked_reference(
            self.query, self.cache, keys, causal=self.g.causal, window=self.g.window
        )


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "geometry,provider,capacity,batch",
    [
        pytest.param(
            *case.values,
            marks=case.marks,
            id=(
                f"{case.values[0].id(case.values[1])}"
                f"-C{case.values[2]}-B{case.values[3]}-eager-graph"
            ),
        )
        for case in ci_cases(
            GEOMETRY_PROVIDERS,
            CAPACITIES,
            (1, 2, 3, 4),
            core=(
                (geometry, provider, capacity, batch)
                for (geometry, provider), capacity, batch in product(
                    GEOMETRY_PROVIDERS, CAPACITIES, (1, 4)
                )
            ),
        )
    ],
)
def test_long_context(geometry, provider, capacity, batch, monkeypatch):
    assert batch * capacity < 2**31
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        case = LongContextCase(geometry, provider, capacity, batch)
        keys = case.stage(0)
        expected = case.reference(keys)
        first_expected = expected.clone()
        tolerance = 0.01 if geometry.interface == "fixed" else ATOL
        case.run()  # Construction and compilation/warmup are outside capture.
        stream.synchronize()
        torch.testing.assert_close(case.out.float(), expected, atol=tolerance, rtol=tolerance)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            case.run()
        for turn in (0, 1):
            if turn:
                keys = case.stage(turn)
                expected = case.reference(keys)
            case.out.fill_(torch.nan)
            # Replay BEFORE eager can refresh any private CSR/schedule state.
            graph.replay()
            torch.testing.assert_close(case.out.float(), expected, atol=tolerance, rtol=tolerance)
            replayed = case.out.clone()
            case.out.fill_(torch.nan)
            case.run()
            torch.testing.assert_close(case.out.float(), expected, atol=tolerance, rtol=tolerance)
            # Random long-context means can differ by less than BF16 tolerance
            # after a small frontier shift. Exact same-input parity also detects
            # stale graph metadata in those cases.
            assert torch.equal(replayed, case.out)
        if geometry.interface == "table":
            # F=M (nominal F+Q=C) is done, not a live C-token request.
            keys = case.stage(0, boundary=True)
            first_expected[: geometry.q] = chunked_reference(
                case.query[: geometry.q],
                case.cache,
                keys[:1],
                causal=geometry.causal,
                window=geometry.window,
            )
            case.out.fill_(torch.nan)
            graph.replay()
            torch.testing.assert_close(case.out.float(), first_expected, atol=ATOL, rtol=RTOL)
            replayed = case.out.clone()
            case.out.fill_(torch.nan)
            case.run()
            torch.testing.assert_close(case.out.float(), first_expected, atol=ATOL, rtol=RTOL)
            assert torch.equal(replayed, case.out)
        stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)


def table_geometry(batch, capacity):
    return {
        "batch_size": batch,
        "max_seq_len": capacity - 16,
        "query_length": 16,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "causal": True,
        "window": None,
        "device": "cuda",
    }


def test_long_context_host_workspace_size(monkeypatch):
    # Only hardware properties are substituted; real sizing and admission run.
    # No CUDA initialization, provider compilation, or workspace allocation.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=8, multi_processor_count=80),
    )
    for g in GEOMETRIES:
        arguments = dict(batch_size=4, device="cuda", **g.arguments())
        if g.interface == "table":
            sizes = CacheTableDraftAttention.workspace_size(
                **arguments,
                max_seq_len=262144 - g.q,
                dtype=torch.bfloat16,
                causal=g.causal,
                window=g.window,
                provider="flashinfer",
            )
        else:
            # FixedQuery sizing is independent of KV capacity; GPU rows exercise
            # real construction with B*C index capacity, including B4/C262144.
            sizes = FlashInferFixedQueryAttention.workspace_size(**arguments)
        assert all(0 < size < 1024**3 and size % 256 == 0 for size in sizes)


@pytest.mark.parametrize(
    "batch,capacity",
    [(1, 2**31), (1, 2**31 + 1), (4, 2**29), (4, 2**29 + 1)],
    ids=("M+Q-equal-int32", "M+Q-above-int32", "B*C-equal-int32", "B*C-above-int32"),
)
def test_table_int32_rejected_before_allocation(batch, capacity, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("int32 rejection must precede provider selection or any launch")

    monkeypatch.setattr(draft, "_provider", forbidden)
    arguments = table_geometry(batch, capacity)
    with pytest.raises(ValueError, match="signed int32"):
        CacheTableDraftAttention.workspace_size(**arguments)
    arguments.pop("batch_size")
    arguments.pop("device")
    # These tiny CPU tensors suffice: scalar bounds precede CUDA/layout checks.
    with pytest.raises(ValueError, match="signed int32"):
        CacheTableDraftAttention(
            torch.empty(0, dtype=torch.uint8),
            torch.empty(0, dtype=torch.uint8),
            request_indices=torch.empty(batch, dtype=torch.int32),
            frontier_table=torch.empty(batch, dtype=torch.int64),
            done_table=torch.empty(batch, dtype=torch.bool),
            cache_index_table=torch.empty((1, 1), dtype=torch.int32),
            guard_start=0,
            metadata_index="row",
            **arguments,
        )


def test_fa2_host_capacity_guard():
    # The existing guard belongs to the DRAFT FA2 adapter and raises
    # NotImplementedError. Direct FixedQuery has no capacity argument/guard;
    # do not misrepresent this as a FixedQuery constructor ValueError test.
    arguments = {
        "batch_size": 4,
        "max_kv_tokens": 262144,
        "page_size": 1,
        "query_length": 16,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "causal": True,
        "window": None,
    }
    # Admit every explicit FlashInfer table row in the GPU matrix. Unexpected
    # construction failures must fail those rows, not become silent skips.
    for geometry, capacity, batch in product(GEOMETRIES, CAPACITIES, (1, 2, 3, 4)):
        if geometry.interface == "table":
            flashinfer.check_capacity(
                **(
                    arguments
                    | geometry.arguments()
                    | {
                        "batch_size": batch,
                        "max_kv_tokens": capacity,
                        "causal": geometry.causal,
                        "window": geometry.window,
                    }
                )
            )
    for batch, capacity in ((4, 2**29), (1, 2**30 + 1)):
        with pytest.raises(NotImplementedError, match="int32"):
            flashinfer.check_capacity(
                **(arguments | {"batch_size": batch, "max_kv_tokens": capacity})
            )


def test_fixed_query_initial_capacity_guard_before_launch(monkeypatch):
    # This is FixedQuery's actual capacity guard: indices must hold B*ceil(Q/P).
    # Tensor metadata stand-ins avoid CUDA and any capacity-sized allocation.
    monkeypatch.setattr(_flashinfer, "version", lambda name: _flashinfer.QUALIFIED_VERSIONS[0])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=8, multi_processor_count=80),
    )
    native = Mock()
    native.return_value.plan.side_effect = AssertionError("must reject before native planning")
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=native)
    )

    def buffer(count, dtype=torch.int32):
        return SimpleNamespace(
            device=torch.device("cuda"),
            dtype=dtype,
            ndim=1,
            is_contiguous=lambda: True,
            numel=lambda: count,
        )

    arguments = GEOMETRIES[-2].arguments()
    sizes = FlashInferFixedQueryAttention.workspace_size(
        batch_size=4, device=torch.device("cuda"), **arguments
    )
    with pytest.raises(ValueError, match="indices cannot hold the initial query blocks"):
        FlashInferFixedQueryAttention(
            *(buffer(n, torch.uint8) for n in sizes),
            query_indptr=buffer(5),
            kv_indptr=buffer(5),
            kv_indices=buffer(4 * 16 - 1),
            last_page_lengths=buffer(4),
            page_size=1,
            dtype=torch.bfloat16,
            causal=True,
            min_chunk_tokens=128,
            **arguments,
        )
    native.return_value.plan.assert_not_called()


def test_chunked_reference_matches_dense_fp64():
    """Validate chunk merges and masks on CPU, including all-masked chunks."""
    generator = torch.Generator().manual_seed(1739)
    query = torch.randn((8, 4, 8), generator=generator, dtype=torch.bfloat16)
    cache = tuple(
        torch.randn((64, 1, 2, 8), generator=generator, dtype=torch.bfloat16) for _ in range(2)
    )
    rows = [torch.randperm(64, generator=generator)[:n] for n in (37, 23)]
    for causal, window in ((False, None), (True, None), (True, 7), (False, 3)):
        actual = chunked_reference(query, cache, rows, causal=causal, window=window, chunk=2)
        expected = torch.empty_like(query, dtype=torch.float64)
        for row, indices in enumerate(rows):
            length = len(indices)
            for i in range(4):
                position = length - 4 + i
                allowed = [
                    j
                    for j in range(length)
                    if (not causal or j <= position)
                    and (window is None or abs(j - position) < window)
                ]
                for head in range(4):
                    keys = cache[0][indices[allowed], 0, head // 2].double()
                    values = cache[1][indices[allowed], 0, head // 2].double()
                    scores = keys @ query[row * 4 + i, head].double() / 8**0.5
                    expected[row * 4 + i, head] = scores.softmax(dim=0) @ values
        torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=1e-6)
