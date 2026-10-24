# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""TritonPagedAttention against the CPU FP32 oracle: accuracy, FP8 KV, batch invariance, graphs."""

import pytest
import torch

from ctikernels.attention.inference import TritonPagedAttention

from ._attention import reference_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (Hq, Hkv, D): Qwen3-30B-A3B, Qwen3.5-27B, Qwen3.5-35B-A3B, its TP2 shard.
GEOMETRIES = [(32, 4, 128), (24, 4, 256), (16, 2, 256), (8, 1, 256)]
PAGE = 16
FP8 = torch.float8_e4m3fn
SCALES = (0.75, 1.5)


class Case:
    """Sequences of (query, key) lengths over shuffled pages of one cache."""

    def __init__(self, lengths, hq, hk, d, dtype=torch.bfloat16, kv_dtype=None, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.lengths, self.hq, self.hk, self.d = lengths, hq, hk, d
        counts = [-(-kv // PAGE) for _, kv in lengths]
        order = torch.randperm(sum(counts) + 3, device="cuda", generator=gen)
        self.pages = order[: sum(counts)].int()
        slots, start = [], 0
        for (_, kv), count in zip(lengths, counts, strict=True):
            page = order[start : start + count]
            start += count
            slots.append((page[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()[:kv])
        self.indices = torch.cat(slots).int()
        self.kv_indptr = (
            torch.tensor([0] + [kv for _, kv in lengths], device="cuda").cumsum(0).int()
        )
        self.q_indptr = torch.tensor([0] + [q for q, _ in lengths], device="cuda").cumsum(0).int()

        def rand(*shape, dtype):
            return torch.randn(*shape, device="cuda", generator=gen).to(dtype)

        kv_dtype = dtype if kv_dtype is None else kv_dtype
        self.k, self.v = (rand(order.numel(), PAGE, hk, d, dtype=kv_dtype) for _ in range(2))
        self.q = rand(int(self.q_indptr[-1]), hq, d, dtype=dtype)

    def reference(self, causal=True, window_left=-1, scales=(1.0, 1.0)):
        return reference_attention(
            self.q.cpu(),
            (self.k.cpu(), self.v.cpu()),
            [q for q, _ in self.lengths],
            [kv for _, kv in self.lengths],
            self.pages.cpu(),
            causal=causal,
            window_left=window_left,
            k_scale=scales[0],
            v_scale=scales[1],
        )

    def run(self, op, max_q_len=None, out=None, scales=(None, None), **mask):
        max_q_len = max(q for q, _ in self.lengths) if max_q_len is None else max_q_len
        op.plan(
            q_indptr=self.q_indptr,
            kv_indptr=self.kv_indptr,
            kv_indices=self.indices,
            max_q_len=max_q_len,
            **mask,
        )
        out = torch.empty_like(self.q) if out is None else out
        k_scale, v_scale = scales
        return op.run(
            q=self.q, paged_kv_cache=(self.k, self.v), out=out, k_scale=k_scale, v_scale=v_scale
        )


def operation(case, **options):
    return TritonPagedAttention(
        num_qo_heads=case.hq,
        num_kv_heads=case.hk,
        head_dim=case.d,
        dtype=case.q.dtype,
        kv_dtype=case.k.dtype,
        device="cuda",
        **options,
    )


def close(out, expected):
    torch.testing.assert_close(out.float().cpu(), expected, atol=1e-2, rtol=1e-2)


DECODE = [(1, 1), (1, 77), (1, 4096), (1, 300)]
# Extend after a cached prefix, a prefill without one, and a decode row in one batch.
EXTEND = [(37, 37), (128, 1100), (1, 513), (200, 333)]
KV = {"bf16": (torch.bfloat16, None), "fp16": (torch.float16, None), "fp8": (torch.bfloat16, FP8)}


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: "x".join(map(str, g)))
@pytest.mark.parametrize("lengths", [DECODE, EXTEND], ids=["decode", "extend"])
@pytest.mark.parametrize("kv", list(KV))
def test_matches_reference(geometry, lengths, kv):
    case = Case(lengths, *geometry, *KV[kv])
    scales = SCALES if kv == "fp8" else (1.0, 1.0)
    out = case.run(operation(case), scales=scales)
    close(out, case.reference(scales=scales))


@pytest.mark.parametrize("causal,window_left", [(True, 63), (False, -1), (False, 39)])
def test_masks(causal, window_left):
    # Fixed-query verify geometry: B=3, Q=16 over prefixes, one shorter than its window.
    case = Case([(16, 500), (16, 16), (16, 33)], 32, 4, 128)
    out = case.run(operation(case), causal=causal, window_left=window_left)
    close(out, case.reference(causal, window_left))


@pytest.mark.parametrize("lengths", [DECODE, EXTEND], ids=["decode", "extend"])
def test_batch_invariant_rows(lengths):
    case = Case(lengths, 16, 2, 256)
    op = operation(case, batch_invariant=True, max_rows=64)
    batched = case.run(op)
    close(batched, case.reference())
    for s, length in enumerate(lengths):
        alone = Case([length], 16, 2, 256)
        alone.k, alone.v = case.k, case.v
        rows = slice(int(case.q_indptr[s]), int(case.q_indptr[s + 1]))
        alone.q = case.q[rows]
        kv = slice(int(case.kv_indptr[s]), int(case.kv_indptr[s + 1]))
        alone.indices = case.indices[kv].clone()
        # Same max_q_len, so the row takes the same path as in the batch.
        out = alone.run(op, max_q_len=max(q for q, _ in lengths))
        assert torch.equal(out, batched[rows])


@pytest.mark.parametrize("lengths", [DECODE, EXTEND], ids=["decode", "extend"])
@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_graph_replay_matches_eager(lengths, kv):
    case = Case(lengths, 24, 4, 256, *KV[kv])
    scales = SCALES if kv == "fp8" else (None, None)
    op = operation(case)
    eager = case.run(op, scales=scales).clone()
    static = {name: getattr(case, name).clone() for name in ("q_indptr", "kv_indptr", "indices")}
    op.plan(
        q_indptr=static["q_indptr"],
        kv_indptr=static["kv_indptr"],
        kv_indices=static["indices"],
        max_q_len=max(q for q, _ in lengths),
    )
    out = torch.empty_like(case.q)

    def call():
        op.run(
            q=case.q, paged_kv_cache=(case.k, case.v), out=out, k_scale=scales[0], v_scale=scales[1]
        )

    call()  # compile outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.zero_()
    graph.replay()
    assert torch.equal(out, eager)
    # New lengths in the same buffers (same row counts) replay to the new eager result.
    other = Case([(q, max(q, kv // 3 + 5)) for q, kv in lengths], 24, 4, 256, *KV[kv], seed=1)
    other.k, other.v, other.q = case.k, case.v, case.q
    other.indices = case.indices[: int(other.kv_indptr[-1])].clone()
    expected = other.run(op, scales=scales)
    static["kv_indptr"].copy_(other.kv_indptr)
    static["q_indptr"].copy_(other.q_indptr)
    static["indices"][: other.indices.numel()].copy_(other.indices)
    graph.replay()
    assert torch.equal(out, expected)


def test_empty_rows_and_keys():
    case = Case([(1, 0), (1, 40)], 8, 1, 256)
    out = case.run(operation(case))
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    close(out[1:], case.reference()[1:])
    empty = Case([(0, 0)], 8, 1, 256)
    assert empty.run(operation(empty), max_q_len=2).numel() == 0
