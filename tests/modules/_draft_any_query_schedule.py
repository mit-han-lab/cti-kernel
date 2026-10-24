# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Execute real Triton prepare code on CPU via its isolated interpreter."""

import torch

from ctikernels.attention._flashinfer_schedule import (
    prepare_draft_chunked,
    prepare_split_kv,
    prepare_split_kv_chunked,
)


def prefix(values):
    return torch.tensor([0, *values], dtype=torch.int64).cumsum(0).to(torch.int32)


def run(batch, query, window, draft=False, minimum=128, short=0):
    page, tile, max_ctas = 16, 128, 42
    query_tiles = -(-query * 4 // tile)
    padded = max(max_ctas, -(-batch * query * 4 // tile) + batch - 1)
    lengths = [query + (i * 19) % 513 for i in range(batch)]
    active = [not draft or i % 3 != 1 for i in range(batch)]
    pages = [-(-n // page) if live else 0 for n, live in zip(lengths, active, strict=True)]
    kv_ptr = prefix(pages)
    schedule = [
        torch.full((n,), -777, dtype=dtype)
        for n, dtype in (
            (padded, torch.int32),
            (padded, torch.int32),
            (padded, torch.int32),
            (batch + 1, torch.int32),
            (batch * query + 1, torch.int32),
            (1, torch.int32),
            (padded, torch.bool),
        )
    ]
    args = {
        "BATCH": batch,
        "QUERY": query,
        "PAGE": page,
        "QUERY_TILE_COUNT": query_tiles,
        "MAX_CTAS": max_ctas,
        "PADDED_CTAS": padded,
        "REQUEST_BLOCK": 1 << (batch - 1).bit_length(),
        "TILE_BLOCK": 1 << (padded - 1).bit_length(),
        "MERGE_BLOCK": 1 << (batch * query - 1).bit_length(),
        "SEARCH_STEPS": batch.bit_length(),
        "WINDOW_LEFT": window,
        "QUERY_TILE_SIZE": tile,
        "MIN_CHUNK": minimum,
        "SHORT_TOKENS": short,
    }
    if draft:
        slots = torch.tensor(
            [i if live else -1 for i, live in enumerate(active)], dtype=torch.int32
        )
        token_ptr = prefix([n if live else 0 for n, live in zip(lengths, active, strict=True)])
        # Each logical page begins at an aligned physical page, with gaps.
        tokens = torch.cat(
            [
                torch.arange(n, dtype=torch.int32) + i * (1 << 20)
                for i, (n, live) in enumerate(zip(lengths, active, strict=True))
                if live
            ]
        )
        qp, pp, fixed = [torch.empty(batch + 1, dtype=torch.int32) for _ in range(3)]
        last, total = torch.empty(batch, dtype=torch.int32), torch.empty(1, dtype=torch.int32)
        pi = torch.empty(sum(pages), dtype=torch.int32)
        page_tiles = -(-max(pages) // 1024)
        prepare_draft_chunked[(batch * page_tiles,)](
            slots,
            token_ptr,
            tokens,
            qp,
            pp,
            pi,
            last,
            fixed,
            total,
            *schedule,
            PAGE_TILES=page_tiles,
            GATHER_TILE=1024,
            **args,
        )
        assert torch.equal(pp, kv_ptr)
        assert torch.equal(qp, prefix([query if live else 0 for live in active]))
        assert torch.equal(fixed, prefix([query] * batch))
        assert total.item() == batch * query
        for i in range(batch):
            assert last[i] == ((lengths[i] - 1) % page + 1 if active[i] else 0)
            expected = tokens[token_ptr[i] : token_ptr[i + 1] : page] // page
            assert torch.equal(pi[kv_ptr[i] : kv_ptr[i + 1]], expected)
    else:
        prepare_split_kv_chunked[(1,)](kv_ptr, *schedule, **args)
        if query <= 32:
            old = [torch.empty_like(t) for t in schedule]
            prepare_split_kv[(1,)](kv_ptr, *old, **args)
            assert all(torch.equal(a, b) for a, b in zip(schedule, old, strict=True))

    span = window + max(query, tile) + (page - 1 if query > 32 else 0)
    effective = [min(n, -(-span // page)) if window >= 0 else n for n in pages]
    high = max(1, *effective)
    low = max(minimum // page, 1) if high * page <= short else 128 // page
    if query > 32 and window >= 0:
        low = max(low, high)
    while low < high:
        middle = (low + high) // 2
        if sum(-(-n // middle) for n in effective) * query_tiles > max_ctas:
            low = middle + 1
        else:
            high = middle
    chunks = [-(-n // low) for n in effective]
    assert schedule[5].item() == low * page
    if query > 32 and window >= 0:
        for length, live, kept, splits in zip(lengths, active, effective, chunks, strict=True):
            if not live:
                continue
            # Count retained tokens from the real final key, not page capacity.
            last = (length - 1) % page + 1
            assert (kept - 1) * page + last >= min(length, window + query)
            # Native prefill.cuh uses W-1+tile to stride partial output/LSE.
            # Each query tile's chunks must cover through the final key and
            # agree with that stride, including partially filled final pages.
            native_splits = -(-min(length, window + tile) // schedule[5].item())
            assert native_splits == splits
            assert splits * schedule[5].item() >= min(length, window + query)
    assert torch.equal(schedule[3], prefix([n * query for n in chunks]))
    assert torch.equal(schedule[4], prefix([n for n in chunks for _ in range(query)]))
    tiles = [(r, q, k) for r, n in enumerate(chunks) for q in range(query_tiles) for k in range(n)]
    assert schedule[6].sum() == len(tiles)
    actual = torch.stack(schedule[:3], dim=1)
    assert torch.equal(actual[: len(tiles)], torch.tensor(tiles, dtype=torch.int32).reshape(-1, 3))
    assert not actual[len(tiles) :].any()


if __name__ == "__main__":
    for query in (1, 2, 32, 33, 64, 129, 512, 4096):
        for window in (-1, 0, 63, 2047):
            run(3, query, window)
            run(3, query, window, draft=True)
    for query in (33, 129):
        run(3, query, 63, minimum=16, short=512)
        run(3, query, 63, draft=True, minimum=16, short=512)
    run(5, 262144, -1)  # B*Q > Triton's maximum block size.
    run(1025, 33, 63)  # Request scan crosses a chunk boundary.
    run(1025, 33, 63, draft=True)
