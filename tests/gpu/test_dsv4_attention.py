# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DeepSeek-V4 attention entries against float32 reference math: the packed-row store, the
query prologue, inverse rotary, row selection and sparse attention (window, c4 top-k and
c128 rows, sinks), eager and CUDA-graph replay."""

from __future__ import annotations

import math

import pytest
import torch

from ctikernels import GraphWorkspace
from ctikernels.attention.inference import (
    DSV4Pages,
    DSV4Rows,
    DSV4SparseAttention,
    dsv4_inverse_rope_,
    dsv4_page_rows,
    dsv4_prologue,
    dsv4_store_kv,
    dsv4_topk_rows,
)

pytestmark = pytest.mark.gpu
EPS = 1e-6
HEADS = 64  # DeepSeek-V4-Flash query heads
WINDOW = DSV4Pages(64)
C128 = DSV4Pages(2)
# Rel-L2 bound vs the FP32 reference per provider. TRTLLM-GEN/FlashMLA read BF16 rows with BF16
# queries (~2.7e-3). FI's sm12x sparse backend quantizes Q to FP8 in-kernel (~3e-2 decode,
# ~1.8e-2 extend), as SGLang ships there; looser than the official reference's BF16 Q.
TOLERANCE = {"flashinfer-trtllm": 1e-2, "flashinfer-sparse": 5e-2}


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _table(positions: int = 4096, theta: float = 10000.0) -> torch.Tensor:
    freq = 1.0 / theta ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64)
    angles = torch.arange(positions, dtype=torch.float32, device="cuda")[:, None] * freq
    return torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()


def _rotate(x: torch.Tensor, table: torch.Tensor, positions: torch.Tensor, inverse=False):
    cos, sin = table[positions].float().chunk(2, dim=-1)
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    sin = -sin if inverse else sin
    pairs = x[..., -64:].float().unflatten(-1, (-1, 2))
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
    return torch.cat((x[..., :-64].float(), rotated), dim=-1)


def _rms(x, weight=None):
    x = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + EPS)
    return x if weight is None else x * weight.float()


def _fp8_ue8m0(x: torch.Tensor) -> torch.Tensor:
    blocks = x.float().unflatten(-1, (-1, 64))
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax * torch.tensor(1 / 448.0))))
    q = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).flatten(-2)


def _latent(kv, weight, table, positions):
    """The reference's stored row: norm, rotary, FP8 on the non-rotary channels."""
    x = _rotate(_rms(kv, weight), table, positions)
    return torch.cat((_fp8_ue8m0(x[:, :448]), x[:, 448:].bfloat16().float()), dim=-1)


def _read(cache: torch.Tensor, pages: DSV4Pages, rows: torch.Tensor) -> torch.Tensor:
    """Float32 ``[N, 512]`` of packed rows ``rows`` (the page ABI, decoded in Torch)."""
    flat = cache.view(-1, pages.page_bytes)
    page, slot = (rows // pages.rows).long(), (rows % pages.rows).long()
    data = flat[page[:, None], slot[:, None] * 576 + torch.arange(576, device=cache.device)]
    scale_at = pages.rows * 576 + slot[:, None] * 8 + torch.arange(7, device=cache.device)
    exponents = flat[page[:, None], scale_at].float()
    nope = data[:, :448].view(torch.float8_e4m3fn).float().unflatten(-1, (7, 64))
    nope = (nope * torch.exp2(exponents - 127)[..., None]).flatten(-2)
    rope = data[:, 448:].contiguous().view(torch.bfloat16).float()
    return torch.cat((nope, rope), dim=-1)


def _pool(pages: DSV4Pages, count: int, table, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """A pool with ``count`` random rows written by the kernel, and their reference values."""
    cache = torch.zeros(
        -(-count // pages.rows) * pages.page_bytes, dtype=torch.uint8, device="cuda"
    )
    kv = torch.randn(count, 512, generator=gen, device="cuda").bfloat16() * 3
    weight = torch.rand(512, generator=gen, device="cuda").bfloat16() + 0.5
    positions = torch.randint(0, table.shape[0], (count,), generator=gen, device="cuda")
    rows = torch.arange(count, device="cuda")
    dsv4_store_kv(kv, weight, EPS, table, positions, rows, cache, pages)
    return cache, _latent(kv, weight, table, positions)


@pytest.mark.parametrize("pages", [WINDOW, C128], ids=["rows64", "rows2"])
def test_store_matches_reference_rows(pages):
    table, gen = _table(), _gen(1)
    cache, ref = _pool(pages, 301, table, gen)
    got = _read(cache, pages, torch.arange(301, device="cuda"))
    # One BF16 ulp; the absolute floor covers rotations that cancel to near zero.
    torch.testing.assert_close(got[:, 448:], ref[:, 448:], rtol=2**-7, atol=1e-6)
    # FP8 codes may differ by one step where the fp32 norm rounds at a boundary.
    step = got[:, :448].abs().unflatten(-1, (7, 64)).amax(-1, keepdim=True) / 448 * 2**-2
    err = (got[:, :448] - ref[:, :448]).abs().unflatten(-1, (7, 64))
    assert (err <= step * 1.01).all()
    assert (err > 0).float().mean() < 1e-3
    scales = cache.view(-1, pages.page_bytes)[:, pages.rows * 576 : pages.rows * 584]
    pad = scales.reshape(-1, pages.rows, 8)[..., 7]
    assert (pad == 0).all()


def test_store_skips_negative_rows():
    table, gen = _table(), _gen(2)
    cache = torch.zeros(WINDOW.page_bytes, dtype=torch.uint8, device="cuda")
    kv = torch.randn(3, 512, generator=gen, device="cuda").bfloat16()
    rows = torch.tensor([5, -1, 7], device="cuda")
    dsv4_store_kv(kv, torch.ones(512, device="cuda"), EPS, table, rows.clone(), rows, cache, WINDOW)
    written = cache[: 64 * 576].view(64, 576).any(-1)
    assert written.nonzero().flatten().tolist() == [5, 7]


@pytest.mark.parametrize("tokens", [1, 4, 20, 130])  # inverse-rope head groups 1, 1, 2, 4 (sm120)
def test_prologue_and_inverse_rope(tokens):
    table, gen = _table(), _gen(3)
    q = torch.randn(tokens, HEADS, 512, generator=gen, device="cuda").bfloat16() * 2
    kv = torch.randn(tokens, 512, generator=gen, device="cuda").bfloat16() * 3
    weight = torch.rand(512, generator=gen, device="cuda").bfloat16() + 0.5
    positions = torch.randint(0, 4096, (tokens,), generator=gen, device="cuda")
    rows = torch.randperm(256, device="cuda")[:tokens]
    rows[0] = -1  # skipped
    fused = torch.zeros(4 * WINDOW.page_bytes, dtype=torch.uint8, device="cuda")
    alone = torch.zeros_like(fused)
    got = dsv4_prologue(q.clone(), kv, weight, EPS, table, positions, rows, fused, WINDOW)
    dsv4_store_kv(kv, weight, EPS, table, positions, rows, alone, WINDOW)
    assert torch.equal(fused, alone)
    torch.testing.assert_close(
        got.float(), _rotate(_rms(q), table, positions), rtol=2**-7, atol=1e-2
    )
    back = dsv4_inverse_rope_(got.clone(), table, positions)
    torch.testing.assert_close(
        back.float(), _rotate(got, table, positions, inverse=True), rtol=2**-7, atol=1e-2
    )


@pytest.mark.parametrize("rows,entries", [(4, 2048), (1, 4096), (4, 16384)])
def test_topk_rows_lower_index_ties(rows, entries):
    gen = _gen(4)
    scores = torch.randn(rows, entries, generator=gen, device="cuda")
    scores[:, 300:900] = 5.0  # 600 tied scores, 512 slots
    lengths = torch.tensor([entries, 700, 512, 37][:rows], dtype=torch.int32, device="cuda")
    page_table = torch.randperm(entries // 64, device="cuda").int().repeat(rows, 1).contiguous()
    got = dsv4_topk_rows(scores, page_table, lengths, k=512, pages=WINDOW)
    for _ in range(5):  # bitwise, including the order
        assert torch.equal(dsv4_topk_rows(scores, page_table, lengths, k=512, pages=WINDOW), got)
    out = torch.empty_like(got)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dsv4_topk_rows(scores, page_table, lengths, k=512, pages=WINDOW, out=out)
    graph.replay()
    assert torch.equal(out, got)
    raw = torch.empty_like(got)
    both = dsv4_topk_rows(scores, page_table, lengths, k=512, pages=WINDOW, entries=raw)
    assert torch.equal(both, got)
    mapped = page_table.gather(1, (raw // 64).clamp_min(0).long()) * 64 + raw % 64
    assert torch.equal(torch.where(raw >= 0, mapped, -1), got)
    for row in range(rows):
        n = int(lengths[row])
        order = torch.sort(scores[row, :n], descending=True, stable=True).indices[:512]
        want = page_table[row, order // 64] * 64 + order % 64
        have = got[row][got[row] >= 0]
        assert sorted(have.tolist()) == sorted(want.int().tolist())
        assert (got[row] < 0).sum() == max(512 - n, 0)


@pytest.mark.parametrize("entries", [1024, 16384])  # register path; larger than any candidate cap
def test_topk_rows_huge_tie_takes_the_lowest_entries(entries):
    scores = torch.zeros(2, entries, device="cuda")  # one tie group
    scores[1, entries - 7] = 1.0
    lengths = torch.full((2,), entries, dtype=torch.int32, device="cuda")
    page_table = torch.arange(entries // 64, device="cuda").int().repeat(2, 1).contiguous()
    raw = torch.empty(2, 512, dtype=torch.int32, device="cuda")
    dsv4_topk_rows(scores, page_table, lengths, k=512, pages=WINDOW, entries=raw)
    assert raw[0].tolist() == list(range(512))
    assert raw[1].tolist() == [*range(511), entries - 7]


def _reference(q, sinks, window_kv, window_idx, window_len, extra_kv, extra_idx, extra_len):
    tokens = q.shape[0]
    keys = []
    for t in range(tokens):
        w = window_idx[t, : window_len[t]]
        e = extra_idx[t, : extra_len[t]] if extra_idx is not None else w[:0]
        w, e = w[w >= 0], e[e >= 0]
        keys.append(
            torch.cat((window_kv[w.long()], extra_kv[e.long()] if e.numel() else window_kv[:0]))
        )
    out = torch.empty(tokens, q.shape[1], 512, device="cuda")
    for t, k in enumerate(keys):
        logits = q[t].float() @ k.T / math.sqrt(512)
        logits = torch.cat((logits, sinks[:, None].float()), dim=-1)
        out[t] = logits.softmax(-1)[:, :-1] @ k
    return out


def _case(tokens, kind, gen, short=False, heads=HEADS):
    table = _table()
    window_cache, window_kv = _pool(WINDOW, 1024, table, gen)
    q = torch.randn(tokens, heads, 512, generator=gen, device="cuda").bfloat16()
    sinks = torch.randn(heads, generator=gen, device="cuda")
    pick = lambda n, k: torch.stack(
        [torch.randperm(n, device="cuda")[:k] for _ in range(tokens)]
    ).int()
    window_idx = pick(1024, 128)
    window_len = torch.full((tokens,), 128, dtype=torch.int32, device="cuda")
    if short:
        window_len = torch.randint(1, 129, (tokens,), generator=gen, device="cuda").int()
    window = DSV4Rows(window_cache, WINDOW, window_idx, window_len)
    extra, extra_kv, extra_idx, extra_len = None, None, None, None
    if kind != "swa":
        pages, count, width = (WINDOW, 4096, 512) if kind == "c4" else (C128, 300, 256)
        extra_cache, extra_kv = _pool(pages, count, table, gen)
        extra_idx = pick(count, width)
        extra_len = torch.full((tokens,), width, dtype=torch.int32, device="cuda")
        if short:
            extra_len = torch.randint(0, width + 1, (tokens,), generator=gen, device="cuda").int()
        extra = DSV4Rows(extra_cache, pages, extra_idx, extra_len)
    ref = _reference(q, sinks, window_kv, window_idx, window_len, extra_kv, extra_idx, extra_len)
    return q, sinks, window, extra, ref


@pytest.mark.parametrize("kind", ["swa", "c4", "c128"])
@pytest.mark.parametrize("tokens,short", [(1, False), (4, True), (37, True)])
def test_sparse_attention_matches_reference(kind, tokens, short):
    q, sinks, window, extra, ref = _case(tokens, kind, _gen(10 + tokens), short)
    attention = DSV4SparseAttention(
        q.device, num_heads=HEADS, max_tokens=64, max_extra=512, scale=512**-0.5
    )
    got = attention(q, sinks, window, extra)
    err = (got.float() - ref).norm() / ref.norm()
    assert err < TOLERANCE[attention.provider], err


def test_sparse_attention_graph_replay():
    gen = _gen(20)
    q, sinks, window, extra, ref = _case(4, "c4", gen)
    attention = DSV4SparseAttention(
        q.device, num_heads=HEADS, max_tokens=4, max_extra=512, scale=512**-0.5
    )
    out = torch.empty_like(q)
    attention(q, sinks, window, extra, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attention(q, sinks, window, extra, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert (out.float() - ref).norm() / ref.norm() < TOLERANCE[attention.provider]


def test_sparse_attention_graphs_share_one_workspace_largest_first():
    """The engine's capture order: warm up and capture B = 4..1 (largest first) in one
    GraphWorkspace, c4 and c128 layers sharing it; every replay equals its eager call bitwise."""
    gen = _gen(21)
    attention = DSV4SparseAttention(
        "cuda", num_heads=HEADS, max_tokens=4, max_extra=512, scale=512**-0.5
    )
    cases = [(b, kind, _case(b, kind, gen, True)) for b in (4, 3, 2, 1) for kind in ("c4", "c128")]
    eager, graphs, outs = {}, [], {}
    with GraphWorkspace(max_rows=4):
        for b, kind, (q, sinks, window, extra, _) in cases:
            eager[b, kind] = attention(q, sinks, window, extra).clone()
        for b, kind, (q, sinks, window, extra, _) in cases:
            outs[b, kind] = torch.empty_like(q)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                attention(q, sinks, window, extra, out=outs[b, kind])
            graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for key, expected in eager.items():
        assert torch.equal(outs[key], expected), key


def _attend(q, sinks, keys):
    logits = torch.einsum("hd,kd->hk", q.float(), keys) / math.sqrt(512)
    probs = torch.cat((logits, sinks[:, None].float()), dim=-1).softmax(-1)[:, :-1]
    return probs @ keys


class _Paged:
    """Two requests over one page table (256-token pages, random ids), every pool row written
    by the store kernel; ``value(kind, request, item)`` is the reference row of a window
    position or a c4/c128 entry, read through the K-d mapping."""

    PER_PAGE = {"window": 256, "c4": 64, "c128": 2}
    POOLS = {"window": WINDOW, "c4": WINDOW, "c128": C128}

    def __init__(self, ends, gen):
        self.ends = ends
        pages = max(-(-end // 256) for end in ends)
        ids = torch.randperm(pages * len(ends), generator=torch.Generator().manual_seed(7))
        self.table = ids.view(len(ends), pages).int().cuda()
        self.cache, self.values = {}, {}
        table = _table(1 << 16)
        for kind, per_page in self.PER_PAGE.items():
            rows = pages * len(ends) * per_page
            pool = self.POOLS[kind]
            cache = torch.zeros(
                -(-rows // pool.rows) * pool.page_bytes, dtype=torch.uint8, device="cuda"
            )
            kv = torch.randn(rows, 512, generator=gen, device="cuda").bfloat16() * 3
            weight = torch.rand(512, generator=gen, device="cuda").bfloat16() + 0.5
            positions = torch.randint(0, 1 << 16, (rows,), generator=gen, device="cuda")
            dsv4_store_kv(
                kv, weight, EPS, table, positions, torch.arange(rows, device="cuda"), cache, pool
            )
            self.cache[kind] = cache
            self.values[kind] = _read(cache, pool, torch.arange(rows, device="cuda"))

    def value(self, kind, request, item):
        per = self.PER_PAGE[kind]
        return self.values[kind][int(self.table[request, item // per]) * per + item % per]


@pytest.mark.parametrize("chunk", [(3200, 150), (3030, 10)])  # 350 / 40 queries (<= 64: split-K)
@pytest.mark.parametrize("heads", [64, 32])
@pytest.mark.parametrize("kind", ["window", "c4", "c128"])
def test_sparse_attention_extend_plan(kind, heads, chunk):
    gen = _gen(30)
    starts, ends = [3000, 0], list(chunk)  # a later chunk and a first chunk
    paged = _Paged(ends, gen)
    query_request = torch.cat(
        [torch.full((e - s,), r) for r, (s, e) in enumerate(zip(starts, ends, strict=True))]
    )
    positions = torch.cat([torch.arange(s, e) for s, e in zip(starts, ends, strict=True)])
    tokens = positions.numel()
    counts = (
        sum(e - max(s - 127, 0) for s, e in zip(starts, ends, strict=True)),
        sum(e // 4 for e in ends),
        sum(e // 128 for e in ends),
    )
    attention = DSV4SparseAttention(
        "cuda",
        num_heads=heads,
        max_tokens=4,
        max_extra=512,
        scale=512**-0.5,
        max_extend_tokens=512,
        max_seq_len=4096,
        max_requests=4,
    )
    plan = attention.plan_extend(
        paged.table,
        torch.tensor(starts).cuda(),
        torch.tensor(ends).cuda(),
        positions.cuda(),
        query_request.int().cuda(),
        counts=counts,
    )
    q = torch.randn(tokens, heads, 512, generator=gen, device="cuda").bfloat16()
    sinks = torch.randn(heads, generator=gen, device="cuda")
    raw, rows, extra = None, None, None
    if kind == "c4":
        visible = ((positions + 1) // 4).clamp(max=512)
        raw = torch.full((tokens, 512), -1, dtype=torch.int32)
        for t in range(tokens):
            n = (int(positions[t]) + 1) // 4
            raw[t, : int(visible[t])] = torch.randperm(
                n, generator=torch.Generator().manual_seed(t)
            )[:512]
        raw = raw.cuda()
        page = paged.table[query_request.cuda().long()[:, None], (raw // 64).clamp_min(0).long()]
        rows = torch.where(raw >= 0, page * 64 + raw % 64, -1).int()
    if kind != "window":
        extra = (paged.cache[kind], _Paged.POOLS[kind])
    got = attention.extend(
        q, sinks, plan, paged.cache["window"], extra, c4_rows=rows, c4_entries=raw
    )
    for t in range(0, tokens, 7):
        r, p = int(query_request[t]), int(positions[t])
        keys = [paged.value("window", r, x) for x in range(max(p - 127, 0), p + 1)]
        if kind == "c4":
            keys += [paged.value("c4", r, int(e)) for e in raw[t] if e >= 0]
        elif kind == "c128":
            keys += [paged.value("c128", r, e) for e in range((p + 1) // 128)]
        ref = _attend(q[t], sinks, torch.stack(keys))
        err = (got[t].float() - ref).norm() / ref.norm()
        assert err < TOLERANCE[attention.provider], (t, err)
    again = attention.extend(
        q, sinks, plan, paged.cache["window"], extra, c4_rows=rows, c4_entries=raw
    )
    assert torch.equal(again, got)


def test_page_rows_follow_the_page_table():
    gen = _gen(31)
    paged = _Paged([3000, 300], gen)
    positions = torch.tensor([2999, 299, 5]).cuda()
    query_request = torch.tensor([0, 1, 1], dtype=torch.int32).cuda()
    rows = torch.empty(3, 128, dtype=torch.int32, device="cuda")
    lengths = torch.empty(3, dtype=torch.int32, device="cuda")
    c128 = torch.empty(3, 64, dtype=torch.int32, device="cuda")
    c128_len = torch.empty(3, dtype=torch.int32, device="cuda")
    dsv4_page_rows(
        paged.table, positions, query_request, window=(rows, lengths), c128=(c128, c128_len)
    )
    for t in range(3):
        r, p = int(query_request[t]), int(positions[t])
        want = [
            int(paged.table[r, x // 256]) * 256 + x % 256 for x in range(p, max(p - 128, -1), -1)
        ]
        assert rows[t, : len(want)].tolist() == want and int(lengths[t]) == len(want)
        assert (rows[t, len(want) :] == -1).all()
        entries = (p + 1) // 128
        want = [int(paged.table[r, e // 2]) * 2 + e % 2 for e in range(entries)]
        assert c128[t, :entries].tolist() == want and int(c128_len[t]) == entries


@pytest.mark.parametrize("queries", [2, 5])
def test_page_rows_masked_lanes_stay_inside_the_table(queries):
    # An 8-request, 16-page table read for c128 at width 64 (more columns than the table holds)
    # and for windows that start before position 0: masked lanes read no table cell (run under
    # compute-sanitizer with PYTORCH_NO_CUDA_MEMORY_CACHING=1 so the table is its own allocation).
    gen = _gen(47)
    table = torch.randint(0, 1 << 12, (8, 16), dtype=torch.int32, device="cuda", generator=gen)
    positions = torch.tensor([4095, 3, 127, 2047, 0][:queries]).cuda()
    query_request = torch.tensor([7, 0, 3, 7, 5][:queries], dtype=torch.int32).cuda()
    rows = torch.empty(queries, 128, dtype=torch.int32, device="cuda")
    lengths = torch.empty(queries, dtype=torch.int32, device="cuda")
    c128 = torch.empty(queries, 64, dtype=torch.int32, device="cuda")
    c128_len = torch.empty(queries, dtype=torch.int32, device="cuda")
    dsv4_page_rows(table, positions, query_request, window=(rows, lengths), c128=(c128, c128_len))
    host = table.cpu()
    for t in range(queries):
        r, p = int(query_request[t]), int(positions[t])
        window = [int(host[r, (p - j) // 256]) * 256 + (p - j) % 256 if p - j >= 0 else -1
                  for j in range(128)]  # fmt: skip
        entries = [int(host[r, e // 2]) * 2 + e % 2 if e < (p + 1) // 128 else -1
                   for e in range(64)]  # fmt: skip
        assert rows[t].tolist() == window and c128[t].tolist() == entries


def test_sparse_attention_bitwise_repeatable():
    q, sinks, window, extra, _ = _case(4, "c4", _gen(40), True)
    attention = DSV4SparseAttention(
        q.device, num_heads=HEADS, max_tokens=4, max_extra=512, scale=512**-0.5
    )
    first = attention(q, sinks, window, extra)
    assert all(torch.equal(attention(q, sinks, window, extra), first) for _ in range(3))


def _null_paged(ends, gen, prepared):
    """``_Paged`` laid out as the engine lays it out: request pages from id 1, and the null page
    0 of every pool never written (0xFF bytes decode to NaN), then prepared or not."""
    paged = _Paged(ends, gen)
    paged.table = paged.table + 1
    for kind, per in _Paged.PER_PAGE.items():
        pool = _Paged.POOLS[kind]
        null = torch.full(((per // pool.rows) * pool.page_bytes,), 0xFF, dtype=torch.uint8,
                          device="cuda")  # fmt: skip
        paged.cache[kind] = torch.cat((null, paged.cache[kind]))
        if prepared:
            pool.prepare(paged.cache[kind])
        paged.values[kind] = torch.cat((paged.values[kind][:per] * 0, paged.values[kind]))
    return paged


def _row0_check(got, ref_rows, prepared, provider):
    """Prepared: finite and matching the reference. Unprepared: FlashInfer 0.7.0.post1's sm12x
    kernels gather row 0 of a pool for every masked candidate slot (decode_dsv4_kernel.cuh
    ~:406, prefill_common.cuh:193) and multiply its value by a zero probability, so a
    non-finite row 0 poisons the output: the reason pools are prepared."""
    if not prepared:
        assert not torch.isfinite(got).all(), "FlashInfer no longer reads row 0 for masked slots"
        return
    assert torch.isfinite(got).all()
    for g, ref in ref_rows:
        assert (g.float() - ref).norm() / ref.norm() < TOLERANCE[provider]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="FlashInfer's packed sm12x kernels",
)
@pytest.mark.parametrize(
    ("kind", "prepared"),
    [("window", True), ("c4", True), ("c128", True), ("window", False)],
    ids=["window", "c4", "c128", "window-unprepared"],
)
def test_null_page_row0_decode(kind, prepared):
    gen = _gen(60)
    ends = [3200, 150]
    paged = _null_paged(ends, gen, prepared)
    # Request 1 at position 10 (11 window rows: masked window slots) and at 149, request 0 at 3199.
    query_request = torch.tensor([1, 1, 0], dtype=torch.int32, device="cuda")
    positions = torch.tensor([10, 149, 3199], device="cuda")
    tokens, heads = 3, 64
    wrows = torch.empty(tokens, 128, dtype=torch.int32, device="cuda")
    wlen = torch.empty(tokens, dtype=torch.int32, device="cuda")
    c128 = torch.empty(tokens, 64, dtype=torch.int32, device="cuda")
    c128_len = torch.empty(tokens, dtype=torch.int32, device="cuda")
    dsv4_page_rows(paged.table, positions, query_request, window=(wrows, wlen),
                   c128=(c128, c128_len))  # fmt: skip
    extra, entries = None, [[] for _ in range(tokens)]
    if kind == "c4":
        visible = ((positions + 1) // 4).clamp(max=512)
        raw = torch.full((tokens, 512), -1, dtype=torch.int32, device="cuda")
        for t in range(tokens):
            pick = torch.randperm(int(visible[t]), generator=torch.Generator().manual_seed(t))
            raw[t, : int(visible[t])] = pick.int().cuda()
            entries[t] = pick.tolist()
        page = paged.table[query_request.long()[:, None], (raw // 64).clamp_min(0).long()]
        rows = torch.where(raw >= 0, page * 64 + raw % 64, -1).int()
        extra = DSV4Rows(paged.cache["c4"], WINDOW, rows, visible.int())
    elif kind == "c128":
        entries = [list(range((int(p) + 1) // 128)) for p in positions]
        extra = DSV4Rows(paged.cache["c128"], C128, c128, c128_len)
    q = torch.randn(tokens, heads, 512, generator=gen, device="cuda").bfloat16()
    sinks = torch.randn(heads, generator=gen, device="cuda")
    attention = DSV4SparseAttention(
        "cuda", num_heads=heads, max_tokens=4, max_extra=512, scale=512**-0.5
    )
    got = attention(q, sinks, DSV4Rows(paged.cache["window"], WINDOW, wrows, wlen), extra)
    refs = []
    for t in range(tokens):
        r, p = int(query_request[t]), int(positions[t])
        keys = [paged.value("window", r, x) for x in range(max(p - 127, 0), p + 1)]
        if kind != "window":
            keys += [paged.value(kind, r, e) for e in entries[t]]
        refs.append((got[t], _attend(q[t], sinks, torch.stack(keys))))
    _row0_check(got, refs, prepared, attention.provider)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="FlashInfer's packed sm12x kernels",
)
@pytest.mark.parametrize("chunk", [(3200, 150), (3030, 10)])  # 350 / 40 queries
@pytest.mark.parametrize("kind", ["window", "c4", "c128"])
def test_null_page_row0_extend(kind, chunk):
    gen = _gen(61)
    starts, ends = [3000, 0], list(chunk)
    paged = _null_paged(ends, gen, True)
    query_request = torch.cat(
        [torch.full((e - s,), r) for r, (s, e) in enumerate(zip(starts, ends, strict=True))]
    )
    positions = torch.cat([torch.arange(s, e) for s, e in zip(starts, ends, strict=True)])
    tokens, heads = positions.numel(), 64
    counts = (
        sum(e - max(s - 127, 0) for s, e in zip(starts, ends, strict=True)),
        sum(e // 4 for e in ends),
        sum(e // 128 for e in ends),
    )
    attention = DSV4SparseAttention(
        "cuda", num_heads=heads, max_tokens=4, max_extra=512, scale=512**-0.5,
        max_extend_tokens=512, max_seq_len=4096, max_requests=4,
    )  # fmt: skip
    plan = attention.plan_extend(
        paged.table, torch.tensor(starts).cuda(), torch.tensor(ends).cuda(), positions.cuda(),
        query_request.int().cuda(), counts=counts,
    )  # fmt: skip
    q = torch.randn(tokens, heads, 512, generator=gen, device="cuda").bfloat16()
    sinks = torch.randn(heads, generator=gen, device="cuda")
    raw, rows, extra = None, None, None
    if kind == "c4":
        visible = ((positions + 1) // 4).clamp(max=512)
        raw = torch.full((tokens, 512), -1, dtype=torch.int32)
        for t in range(tokens):
            n = (int(positions[t]) + 1) // 4
            raw[t, : int(visible[t])] = torch.randperm(
                n, generator=torch.Generator().manual_seed(t)
            )[:512]
        raw = raw.cuda()
        page = paged.table[query_request.cuda().long()[:, None], (raw // 64).clamp_min(0).long()]
        rows = torch.where(raw >= 0, page * 64 + raw % 64, -1).int()
    if kind != "window":
        extra = (paged.cache[kind], _Paged.POOLS[kind])
    got = attention.extend(
        q, sinks, plan, paged.cache["window"], extra, c4_rows=rows, c4_entries=raw
    )
    refs = []
    for t in range(0, tokens, 7):
        r, p = int(query_request[t]), int(positions[t])
        keys = [paged.value("window", r, x) for x in range(max(p - 127, 0), p + 1)]
        if kind == "c4":
            keys += [paged.value("c4", r, int(e)) for e in raw[t] if e >= 0]
        elif kind == "c128":
            keys += [paged.value("c128", r, e) for e in range((p + 1) // 128)]
        refs.append((got[t], _attend(q[t], sinks, torch.stack(keys))))
    _row0_check(got, refs, True, attention.provider)
