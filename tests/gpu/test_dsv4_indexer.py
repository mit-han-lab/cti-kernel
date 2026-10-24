# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""DeepSeek-V4 compressor and indexer on the GPU against PyTorch references of the same
rounding: ragged prefill chunks, single- and multi-token decode chained through caller-owned
rings (no length counter: a fresh request in a dirty slot, chunked prefill, decode across window
boundaries), and the indexer's BF16-exact logits with lower-entry ties through FlashInfer's
top-k."""

from __future__ import annotations

import importlib

import pytest
import torch

from ctikernels.dsv4_indexer.inference import DSV4Compressor, DSV4Indexer

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
pytest.importorskip("triton")

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def bf16(x):
    return x.to(torch.bfloat16).float()


def rope(x, pos, cos, sin):
    """Rotate the trailing 64 channels of ``x [..., D]`` in adjacent pairs at ``pos``."""
    head, tail = x[..., :-64], x[..., -64:].unflatten(-1, (32, 2))
    c, s = cos[pos], sin[pos]
    while c.dim() < tail.dim() - 1:
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    even, odd = tail.unbind(-1)
    rotated = torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2)
    return torch.cat((head, rotated), -1)


def hadamard(x):
    out, span = x.clone(), 1
    while span < x.shape[-1]:
        pairs = out.unflatten(-1, (-1, 2, span))
        out = torch.stack((pairs[..., 0, :] + pairs[..., 1, :], pairs[..., 0, :] - pairs[..., 1, :]),
                          -2).flatten(-3)  # fmt: skip
        span *= 2
    return bf16(out * x.shape[-1] ** -0.5)


def ceil_pow2_exponent(raw):
    bits = raw.contiguous().view(torch.int32)
    return ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()


def fp4_round_trip(x):
    """E2M1 with UE8M0 per 32 (ties to even), decoded."""
    g = x.unflatten(-1, (-1, 32))
    e = ceil_pow2_exponent(g.abs().amax(-1).clamp_min(6 * 2.0**-126) * (1 / 6))
    a = g / torch.exp2(e.float()).unsqueeze(-1)
    grid = E2M1.to(x.device)
    m = a.abs()
    # Nearest magnitude, ties to the even code (the kernel's comparison chain).
    idx = sum((m > t).long() for t in (0.25, 1.25, 2.5, 5)) + sum(
        (m >= t).long() for t in (0.75, 1.75, 3.5)
    )
    return (grid[idx] * a.sign() * torch.exp2(e.float()).unsqueeze(-1)).flatten(-2)


# Compressor reference ---------------------------------------------------------------------


def reference_entries(kv, score, ape, weight, cos, sin, ratio, dim):
    """Every complete entry of one request's whole token stream ``[S, coff * dim]``."""
    overlap = ratio == 4
    blocks = kv.shape[0] // ratio
    score = score + ape.repeat(blocks + 1, 1)[: kv.shape[0]]
    out = []
    for b in range(blocks):
        cur = slice(b * ratio, (b + 1) * ratio)
        if overlap:
            if b:
                prev = slice((b - 1) * ratio, b * ratio)
                pk, ps = kv[prev, :dim], score[prev, :dim]
            else:
                pk = torch.zeros(ratio, dim, device=kv.device)
                ps = torch.full((ratio, dim), float("-inf"), device=kv.device)
            k = torch.cat((pk, kv[cur, dim:]))
            s = torch.cat((ps, score[cur, dim:]))
        else:
            k, s = kv[cur], score[cur]
        pooled = (k * s.softmax(0)).sum(0)
        if dim == 128:  # index keys keep the official BF16 casts
            pooled = bf16(pooled)
            normed = bf16(pooled * torch.rsqrt(pooled.square().mean() + 1e-6) * weight.float())
            out.append(hadamard(bf16(rope(normed, b * ratio, cos, sin))))
        else:  # latent entries: FP32 intermediates (dsv4_store_kv's recipe)
            normed = pooled * torch.rsqrt(pooled.square().mean() + 1e-6) * weight.float()
            out.append(rope(normed, b * ratio, cos, sin))
    return torch.stack(out) if out else torch.zeros(0, dim, device=kv.device)


def decode_latent(cache, rows, rows_per_page):
    page, row = rows // rows_per_page, rows % rows_per_page
    base = cache[page]
    data = torch.stack([base[i, r * 576 : (r + 1) * 576] for i, r in enumerate(row.tolist())])
    foot = torch.stack([base[i, rows_per_page * 576 + r * 8:rows_per_page * 576 + r * 8 + 7]
                        for i, r in enumerate(row.tolist())])  # fmt: skip
    nope = data[:, :448].contiguous().view(torch.float8_e4m3fn).float().unflatten(-1, (7, 64))
    nope = (nope * torch.exp2(foot.float() - 127).unsqueeze(-1)).flatten(-2)
    return torch.cat((nope, data[:, 448:].contiguous().view(torch.bfloat16).float()), -1)


def decode_index(cache, rows):
    page, row = rows // 64, rows % 64
    out = []
    for p, r in zip(page.tolist(), row.tolist(), strict=True):
        payload = cache[p, r * 64 : (r + 1) * 64].int()
        codes = torch.stack((payload & 15, payload >> 4), -1).flatten()
        value = E2M1.to(cache.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
        scale = torch.exp2(cache[p, 4096 + r * 4 : 4096 + r * 4 + 4].float() - 127)
        out.append(value * scale.repeat_interleave(32))
    return torch.stack(out)


@pytest.mark.parametrize(("ratio", "dim"), [(4, 512), (4, 128), (128, 512)])
def test_compressor_chains_prefill_and_decode(ratio, dim):
    torch.manual_seed(ratio + dim)
    dev = "cuda"
    coff = 2 if ratio == 4 else 1
    lengths_total = [700, 37, 263]
    ids = [5, 0, 2]
    op = DSV4Compressor(ratio=ratio, dim=dim, max_tokens=1024, max_batch=4, device=dev)
    slots = 6
    state_kv = torch.empty(op.state_shape(slots), device=dev)
    state_score = torch.empty_like(state_kv)
    pages = torch.randperm(slots * 4, dtype=torch.int32, device=dev).view(slots, 4)
    cache = torch.zeros(slots * 4, op.page_bytes, dtype=torch.uint8, device=dev)
    ape = torch.randn(ratio, coff * dim, device=dev)
    weight = (1 + 0.1 * torch.randn(dim, device=dev)).bfloat16()
    angles = torch.rand(1024, 32, device=dev) * 6.3
    cos, sin = angles.cos(), angles.sin()
    stream_kv = [torch.randn(n, coff * dim, device=dev) for n in lengths_total]
    stream_score = [torch.randn(n, coff * dim, device=dev) for n in lengths_total]
    done = [0, 0, 0]
    written = [[], [], []]

    def step(counts):
        live = [i for i, c in enumerate(counts) if c]
        kv = torch.cat([stream_kv[i][done[i] : done[i] + counts[i]] for i in live])
        sc = torch.cat([stream_score[i][done[i] : done[i] + counts[i]] for i in live])
        offs = torch.tensor([0] + [counts[i] for i in live], device=dev).cumsum(0).int()
        rows = op(
            kv, sc, state_kv=state_kv, state_score=state_score,
            request_ids=torch.tensor([ids[i] for i in live], dtype=torch.int32, device=dev),
            offsets=offs, starts=torch.tensor([done[i] for i in live], device=dev),
            pages=pages, cache=cache, ape=ape, norm_weight=weight, rope=torch.cat((cos, sin), -1),
        )  # fmt: skip
        for k, i in enumerate(live):
            completed = (done[i] + counts[i]) // ratio - done[i] // ratio
            got = rows[k].tolist()
            assert all(r == -1 for r in got[completed:])
            written[i] += got[:completed]
            done[i] += counts[i]

    step([300, 20, 129])  # prefill: ragged, HCA completes one entry for the third request
    for _ in range(5):  # single-token decode
        step([1, 1, 1])
    step([3, 0, 5])  # multi-token decode crossing window boundaries
    step([392, 12, 124])  # prefill to the end

    for i in range(3):
        want = reference_entries(stream_kv[i], stream_score[i], ape, weight, cos, sin, ratio, dim)
        entry = torch.arange(want.shape[0], device=dev)
        rows_per_page = 256 // ratio
        expect = (
            pages[ids[i]][entry // rows_per_page].long() * rows_per_page + entry % rows_per_page
        )
        assert written[i] == expect.tolist()
        if not written[i]:
            continue
        rows = torch.tensor(written[i], dtype=torch.int64, device=dev)
        if dim == 128:
            got, ref = decode_index(cache, rows), fp4_round_trip(want)
            # Pooling sums in another order can move a BF16 rounding and flip an E2M1 code.
            assert (got == ref).float().mean() > 0.98
        else:
            got = decode_latent(cache, rows, rows_per_page)
            # BF16 rotary channels from FP32 math: one rounding, sums in another order.
            rope_err = (got[:, 448:] - bf16(want[:, 448:])).abs()
            bound = want.abs().amax(-1, keepdim=True) * 2.0**-7
            assert (rope_err <= bound).all(), (rope_err / bound).max()
            assert (rope_err == 0).float().mean() > 0.98, (rope_err == 0).float().mean()


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_bytes_match_dsv4_store_kv(ratio):
    """K-d contract: latent entries are packed as dsv4_store_kv packs rows. Dyadic KV with
    constant scores makes every pooled sum exact in any order and a {0, +-1} RoPE table makes
    every product exact, so both writers see bit-identical entries."""
    attention = importlib.import_module("ctikernels.attention.inference")
    if not hasattr(attention, "dsv4_store_kv"):
        pytest.skip("needs ctikernels.attention.inference.dsv4_store_kv (combined DSV4 tree)")
    torch.manual_seed(7)
    dev, coff, rows_per_page = "cuda", 2 if ratio == 4 else 1, 256 // ratio
    tokens = 8 * ratio
    # Positive values keep every rotated channel nonzero (no signed-zero ambiguity).
    kv = torch.randint(1, 9, (tokens, coff * 512), device=dev).float() * 0.25
    score = torch.zeros_like(kv)
    op = DSV4Compressor(ratio=ratio, dim=512, max_tokens=tokens, max_batch=1, device=dev)
    state = torch.zeros(op.state_shape(1), device=dev)
    pages = torch.tensor([[3, 1, 0, 2]], dtype=torch.int32, device=dev)
    cache = torch.zeros(4, op.page_bytes, dtype=torch.uint8, device=dev)
    weight = (1 + 0.25 * torch.randint(-2, 3, (512,), device=dev)).bfloat16()
    quarter = torch.randint(0, 4, (4 * tokens, 32), device=dev).float() * (torch.pi / 2)
    table = torch.cat((quarter.cos().round(), quarter.sin().round()), -1)
    rows = op(
        kv, score, state_kv=state, state_score=state.clone(),
        request_ids=torch.zeros(1, dtype=torch.int32, device=dev),
        offsets=torch.tensor([0, tokens], dtype=torch.int32, device=dev),
        starts=torch.zeros(1, dtype=torch.int64, device=dev), pages=pages, cache=cache,
        ape=torch.zeros(ratio, coff * 512, device=dev), norm_weight=weight, rope=table,
    )  # fmt: skip
    entries = tokens // ratio
    rows = rows[0, :entries]
    pooled = []
    for b in range(entries):
        cur = kv[b * ratio : (b + 1) * ratio, -512:]
        if ratio == 4:
            prev = kv[(b - 1) * ratio : b * ratio, :512] if b else None
            pooled.append(cur.mean(0) if prev is None else torch.cat((prev, cur)).mean(0))
        else:
            pooled.append(cur.mean(0))
    other = torch.zeros_like(cache)
    window = torch.arange(entries, device=dev) * ratio
    attention.dsv4_store_kv(torch.stack(pooled), weight, 1e-6, table, window, rows, other,
                            attention.DSV4Pages(rows_per_page))  # fmt: skip
    assert torch.equal(other, cache)


@pytest.mark.parametrize(("ratio", "dim"), [(4, 512), (4, 128), (128, 512)])
def test_compressor_padding_rows_are_inert(ratio, dim):
    """Request id -1 pads a batch (extend and decode): the real rows' outputs, state and cache
    are bitwise those of the unpadded batch; padding returns rows -1."""
    torch.manual_seed(ratio + dim + 7)
    dev = "cuda"
    width = (2 if ratio == 4 else 1) * dim
    op = DSV4Compressor(ratio=ratio, dim=dim, max_tokens=1024, max_batch=8, device=dev)
    slots, ids = 4, [3, 1]
    pages = torch.randperm(slots * 4, dtype=torch.int32, device=dev).view(slots, 4)
    ape = torch.randn(ratio, width, device=dev)
    weight = (1 + 0.1 * torch.randn(dim, device=dev)).bfloat16()
    angles = torch.rand(1024, 32, device=dev) * 6.3
    rope = torch.cat((angles.cos(), angles.sin()), -1)
    state = [torch.randn(op.state_shape(slots), device=dev) for _ in range(2)]
    copies = [
        {
            "state_kv": state[0].clone(), "state_score": state[1].clone(),
            "cache": torch.zeros(slots * 4, op.page_bytes, dtype=torch.uint8, device=dev),
        }
        for _ in range(2)
    ]  # fmt: skip
    done = [0, 0]

    def step(counts, pad_counts):
        real = [
            (torch.randn(c, width, device=dev), torch.randn(c, width, device=dev)) for c in counts
        ]
        pad = [
            (torch.randn(c, width, device=dev), torch.randn(c, width, device=dev))
            for c in pad_counts
        ]
        # Padded order: pad, real 0, pad, real 1, pad ...; padding starts are deliberately stale.
        order = []
        for i in range(max(len(real), len(pad))):
            if i < len(pad):
                order.append(("pad", i))
            if i < len(real):
                order.append(("real", i))
        results = []
        for copy, layout in zip(
            copies, ([("real", i) for i in range(len(real))], order), strict=True
        ):
            chunks = [(real if kind == "real" else pad)[i] for kind, i in layout]
            offs = torch.tensor([0] + [c[0].shape[0] for c in chunks], device=dev).cumsum(0).int()
            rows = op(
                torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]),
                request_ids=torch.tensor([ids[i] if k == "real" else -1 for k, i in layout],
                                         dtype=torch.int32, device=dev),
                offsets=offs,
                starts=torch.tensor([done[i] if k == "real" else 999 for k, i in layout], device=dev),
                pages=pages, ape=ape, norm_weight=weight, rope=rope, **copy,
            )  # fmt: skip
            results.append((layout, rows.clone()))
        (_, rows0), (layout, rows1) = results
        for pos, (kind, i) in enumerate(layout):
            if kind == "real":
                assert torch.equal(rows1[pos], rows0[i])
            else:
                assert (rows1[pos] == -1).all()
        for i, c in enumerate(counts):
            done[i] += c

    step([2 * ratio + 3, ratio + 1], [5, 9, 2])  # extend (prefill path), entries completed
    step([1, 1], [1, 1])  # decode
    step([ratio - 1, 2], [1])  # decode path when T <= max(16, B), else extend; window ends
    for name in ("state_kv", "state_score", "cache"):
        assert torch.equal(copies[0][name], copies[1][name]), name


def _compressor_case(ratio, dim, tokens, slots=3, seed=0):
    """One request's token stream and a compressor over ``slots`` NaN-filled (dirty) ring slots."""
    torch.manual_seed(seed)
    dev, coff = "cuda", 2 if ratio == 4 else 1
    op = DSV4Compressor(ratio=ratio, dim=dim, max_tokens=tokens, max_batch=2, device=dev)
    angles = torch.rand(tokens + 1, 32, device=dev) * 6.3
    case = {
        "op": op,
        "kv": torch.randn(tokens, coff * dim, device=dev),
        "score": torch.randn(tokens, coff * dim, device=dev),
        "pages": torch.randperm(slots * 8, dtype=torch.int32, device=dev).view(slots, 8),
        "ape": torch.randn(ratio, coff * dim, device=dev),
        "weight": (1 + 0.1 * torch.randn(dim, device=dev)).bfloat16(),
        "cos": angles.cos(),
        "sin": angles.sin(),
    }
    return case


def _fresh_state(case, slots=3):
    op = case["op"]
    nan = torch.full(op.state_shape(slots), float("nan"), device="cuda")
    cache = torch.zeros(slots * 8, op.page_bytes, dtype=torch.uint8, device="cuda")
    return nan, nan.clone(), cache


def _feed(case, state_kv, state_score, cache, slot, start, end):
    """Tokens ``[start, end)`` of the case's stream as one call for request ``slot``."""
    dev = "cuda"
    rows = case["op"](
        case["kv"][start:end], case["score"][start:end], state_kv=state_kv,
        state_score=state_score, request_ids=torch.tensor([slot], dtype=torch.int32, device=dev),
        offsets=torch.tensor([0, end - start], dtype=torch.int32, device=dev),
        starts=torch.tensor([start], device=dev), pages=case["pages"], cache=cache,
        ape=case["ape"], norm_weight=case["weight"],
        rope=torch.cat((case["cos"], case["sin"]), -1),
    )  # fmt: skip
    return rows[0].tolist()


@pytest.mark.parametrize(("ratio", "dim"), [(4, 512), (4, 128), (128, 512)])
def test_compressor_fresh_request_in_dirty_slot_decodes_across_boundaries(ratio, dim):
    """A new request in a NaN-filled slot (no reset) prefills, then decodes one token at a time
    across window boundaries (4 and 128): every entry is finite, matches the one-shot
    reference, and is written where the page map says; nothing else in the cache changes."""
    tokens = 2 * 128 + 6
    case = _compressor_case(ratio, dim, tokens, seed=ratio + dim)
    state_kv, state_score, cache = _fresh_state(case)
    slot, written = 1, []
    prefill = 128 - 3  # decode then crosses 128 and several 4-boundaries
    written += [r for r in _feed(case, state_kv, state_score, cache, slot, 0, prefill) if r >= 0]
    for p in range(prefill, tokens):
        written += [r for r in _feed(case, state_kv, state_score, cache, slot, p, p + 1) if r >= 0]
    want = reference_entries(case["kv"], case["score"], case["ape"], case["weight"], case["cos"],
                             case["sin"], ratio, dim)  # fmt: skip
    rows_per_page = 256 // ratio
    entry = torch.arange(want.shape[0], device="cuda")
    expect = (
        case["pages"][slot][entry // rows_per_page].long() * rows_per_page + entry % rows_per_page
    )
    assert written == expect.tolist()
    rows = torch.tensor(written, dtype=torch.int64, device="cuda")
    if dim == 128:
        got = decode_index(cache, rows)
        assert torch.isfinite(got).all()
        assert (got == fp4_round_trip(want)).float().mean() > 0.98
    else:
        got = decode_latent(cache, rows, rows_per_page)
        assert torch.isfinite(got).all()
        rope_err = (got[:, 448:] - bf16(want[:, 448:])).abs()
        assert (rope_err <= want.abs().amax(-1, keepdim=True) * 2.0**-7).all()
    # Only the request's own entries were written: every other page row is still zero.
    untouched = torch.ones(cache.shape[0] * rows_per_page, dtype=torch.bool, device="cuda")
    untouched[rows] = False
    page_rows = untouched.view(cache.shape[0], rows_per_page)
    for page in torch.nonzero(page_rows.all(-1)).flatten().tolist():
        assert (cache[page] == 0).all(), page


@pytest.mark.parametrize(("ratio", "dim"), [(4, 512), (4, 128), (128, 512)])
@pytest.mark.parametrize("cuts", [(37,), (128, 200), (17, 64, 131, 245)])
def test_compressor_chunked_prefill_equals_one_call(ratio, dim, cuts):
    """A prefill split into chunks (chunks over 16 tokens, the prefill path) at several offsets
    writes the same cache bytes and rows as one call, from NaN-filled slots."""
    tokens = 262
    case = _compressor_case(ratio, dim, tokens, seed=ratio * 3 + dim)
    one = _fresh_state(case)
    rows_one = _feed(case, *one, 2, 0, tokens)
    split = _fresh_state(case)
    bounds, rows_split = (0, *cuts, tokens), []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        rows_split += [r for r in _feed(case, *split, 2, start, end) if r >= 0]
    assert rows_split == [r for r in rows_one if r >= 0]
    assert torch.equal(split[2], one[2])


# Indexer ----------------------------------------------------------------------------------


PROVIDERS = ["triton", "flashinfer"]


def _provider(name):
    """Skip a provider this device cannot run (collection never touches CUDA)."""
    if name == "flashinfer":
        from ctikernels.dsv4_indexer.flashinfer import _logits

        if not _logits.available(torch.device("cuda")):
            pytest.skip("FlashInfer indexer needs compute capability 10.x")
    return name


def _index_case(t=5, entries=704, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    pages_per = -(-entries // 128) * 2  # the table width the indexer requires
    pages = torch.randperm(2 * pages_per, dtype=torch.int32, device=dev).view(2, pages_per)
    cache = torch.randint(0, 256, (2 * pages_per, 4352), dtype=torch.uint8, device=dev)
    cache[:, 4096:] = torch.randint(122, 130, (2 * pages_per, 256), dtype=torch.uint8, device=dev)
    q = torch.randn(t, 64, 128, device=dev).bfloat16()
    w = torch.randn(t, 64, device=dev).bfloat16()
    positions = torch.randint(0, 4 * entries + 40, (t,), device=dev)
    ids = torch.randint(0, 2, (t,), dtype=torch.int32, device=dev)
    angles = torch.rand(4 * entries + 64, 32, device=dev) * 6.3
    return q, w, positions, ids, pages, cache, angles.cos(), angles.sin()


def reference_scores(q, w, positions, ids, pages, cache, cos, sin, n):
    """FP64 scores of the FP4 operands; ``lengths`` = visible entries per query."""
    qq = fp4_round_trip(hadamard(bf16(rope(q.float(), positions, cos, sin)))).double()
    entry = torch.arange(n, device=q.device)
    out = []
    for t in range(q.shape[0]):
        rows = pages[ids[t].long()][entry // 64].long() * 64 + entry % 64
        dots = qq[t] @ decode_index(cache, rows).double().T
        weight = w[t].double() * (128**-0.5 * 64**-0.5)
        out.append((dots.relu() * weight[:, None]).sum(0))
    return torch.stack(out), ((positions + 1) // 4).clamp(max=n)


def _run(op, case, out):
    q, w, positions, ids, pages, cache, cos, sin = case
    return op(q, w, positions=positions, request_ids=ids, pages=pages, cache=cache,
              rope=torch.cat((cos, sin), -1), out=out)  # fmt: skip


@pytest.mark.parametrize("provider", PROVIDERS)
def test_indexer_matches_reference_deterministic_and_graph_safe(provider):
    provider = _provider(provider)
    case = _index_case()
    n = 704
    op = DSV4Indexer(max_queries=8, max_entries=n, device="cuda", provider=provider)
    assert op.provider == provider
    out = torch.full((5, op.columns + 7), float("nan"), device="cuda")[:, : op.columns]
    got = _run(op, case, out).clone()
    want, lengths = reference_scores(*case, n)
    for t, length in enumerate(lengths.tolist()):
        # FP32 sums of exact FP4 products in another order than the FP64 oracle.
        bound = 1e-5 * (want[t, :length].abs().max() + 1)
        assert ((got[t, :length].double() - want[t, :length]).abs() <= bound).all()
    assert _same_visible(got, _run(op, case, out), lengths)  # bitwise run to run
    graph = torch.cuda.CUDAGraph()
    replay = torch.zeros_like(out)
    with torch.cuda.graph(graph):
        _run(op, case, replay)
    graph.replay()
    torch.cuda.synchronize()
    assert _same_visible(got, replay[:, :n], lengths)


def _same_visible(a, b, lengths):
    return all(torch.equal(a[t, :k], b[t, :k]) for t, k in enumerate(lengths.tolist()))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_indexer_ties_select_the_lower_entry(provider):
    provider = _provider(provider)
    from flashinfer.topk import top_k_page_table_transform

    q, w, positions, ids, pages, cache, cos, sin = case = _index_case(t=3, entries=2048, seed=1)
    # Repeat one page's keys everywhere: every visible score has many exact ties.
    cache[:] = cache[pages[0, 0].long()]
    positions[:] = 4 * 2048 - 1
    op = DSV4Indexer(max_queries=3, max_entries=2048, device="cuda", provider=provider)
    out = torch.empty(3, op.columns, device="cuda")
    scores = _run(op, case, out)
    lengths = ((positions + 1) // 4).int()
    # tie_break picks the selected set; deterministic only fixes its order (and is not
    # supported on 12.x), so the sets are compared.
    got = top_k_page_table_transform(scores, pages, lengths, 512, row_to_batch=ids, tie_break=1,
                                     page_size=64)  # fmt: skip
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :512]
    want = pages[ids.long()].gather(1, order // 64).long() * 64 + order % 64
    assert torch.equal(got.sort(-1).values.long(), want.sort(-1).values)


@pytest.mark.parametrize(("ratio", "dim"), [(128, 512), (4, 512), (4, 128)])
@pytest.mark.parametrize("prompt", [127, 129, 200, 255, 257, 300])
def test_compressor_prefill_then_one_decode_equals_one_call(ratio, dim, prompt):
    """The engine's failing shape: prefill tokens [0, prompt) in one call at start 0 from a
    NaN-filled slot, then decode token ``prompt``; the cache equals one call over
    [0, prompt + 1] bitwise (every entry finite)."""
    tokens = prompt + 1
    case = _compressor_case(ratio, dim, tokens, seed=prompt + ratio + dim)
    one = _fresh_state(case)
    _feed(case, *one, 2, 0, tokens)
    split = _fresh_state(case)
    _feed(case, *split, 2, 0, prompt)
    _feed(case, *split, 2, prompt, tokens)
    assert torch.equal(split[2], one[2])
    assert torch.isfinite(split[0][2]).any()


def test_indexer_never_reads_null_page_row0():
    """The index pool needs no prepare: Triton masks key and scale loads past each query's
    visible entries, so a NaN-scaled (0xFF) never-written null page 0 leaves the scores
    bitwise unchanged."""
    q, w, positions, ids, pages, cache, cos, sin = _index_case(seed=3)
    pages = pages + 1  # request pages from id 1; page 0 is the null page
    n = 704
    op = DSV4Indexer(max_queries=8, max_entries=n, device="cuda", provider="triton")
    scores = []
    for fill in (0x00, 0xFF):
        null = torch.full((1, 4352), fill, dtype=torch.uint8, device="cuda")
        case = (q, w, positions, ids, pages, torch.cat((null, cache)), cos, sin)
        out = torch.full((q.shape[0], op.columns), float("nan"), device="cuda")
        scores.append(_run(op, case, out).clone())
    lengths = ((positions + 1) // 4).clamp(max=n)
    assert all(torch.isfinite(scores[1][t, :k]).all() for t, k in enumerate(lengths.tolist()))
    assert _same_visible(scores[0], scores[1], lengths)
