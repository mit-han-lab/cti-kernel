# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only Triton interpreter checks, isolated from CUDA and mocked imports."""

import torch
import triton

from ctikernels.attention._flashinfer_schedule import prepare_draft


def check_schedule():
    for b in (1, 2, 3, 4):
        for q in (8, 16):
            for p in (1, 16):
                for active_bits in range(1 << b):
                    slots = torch.tensor(
                        [r if active_bits & (1 << r) else -1 for r in range(b)], dtype=torch.int32
                    )
                    lens = [q + 1700 * r if slot != -1 else 0 for r, slot in enumerate(slots)]
                    token_ptr = torch.tensor([0] + lens, dtype=torch.int32).cumsum(
                        0, dtype=torch.int32
                    )
                    token_idx = torch.full((b * 6000,), -123, dtype=torch.int32)
                    expected_pages = []
                    for r, length in enumerate(lens):
                        page_ids = (
                            torch.arange((length + p - 1) // p, dtype=torch.int32).flip(0)
                            + r * 6000
                        )
                        values = (page_ids[:, None] * p + torch.arange(p)[None, :]).flatten()[
                            :length
                        ]
                        start = int(token_ptr[r])
                        token_idx[start : start + length] = values
                        expected_pages.extend(page_ids.tolist())
                    for window_left, query_tiles in ((-1, 1), (2047, 1), (4095, 2)):
                        query_ptr = torch.full((b + 1,), -123, dtype=torch.int32)
                        page_ptr, fixed = query_ptr.clone(), query_ptr.clone()
                        last = torch.full((b,), -123, dtype=torch.int32)
                        page_idx = torch.full((b * 6000,), -123, dtype=torch.int32)
                        arrays = [torch.full((32,), -123, dtype=torch.int32) for _ in range(3)]
                        offsets = torch.full((b + 1,), -123, dtype=torch.int32)
                        merge = torch.full((b * q + 1,), -123, dtype=torch.int32)
                        chunk = torch.zeros(1, dtype=torch.int32)
                        valid = torch.ones(32, dtype=torch.bool)
                        total = torch.full((1,), -123, dtype=torch.int32)
                        page_tiles = triton.cdiv(triton.cdiv(6000, p), 1024)
                        prepare_draft[(b * page_tiles,)](
                            slots,
                            token_ptr,
                            token_idx,
                            query_ptr,
                            page_ptr,
                            page_idx,
                            last,
                            fixed,
                            total,
                            *arrays,
                            offsets,
                            merge,
                            chunk,
                            valid,
                            BATCH=b,
                            QUERY=q,
                            PAGE=p,
                            PAGE_TILES=page_tiles,
                            GATHER_TILE=1024,
                            QUERY_TILE_COUNT=query_tiles,
                            MAX_CTAS=32,
                            PADDED_CTAS=32,
                            REQUEST_BLOCK=triton.next_power_of_2(b),
                            TILE_BLOCK=32,
                            MERGE_BLOCK=triton.next_power_of_2(b * q),
                            SEARCH_STEPS=b.bit_length(),
                            WINDOW_LEFT=window_left,
                            QUERY_TILE_SIZE=128,
                        )
                        assert query_ptr.tolist() == [0] + [
                            sum(x != -1 for x in slots[:r]) * q for r in range(1, b + 1)
                        ]
                        assert fixed.tolist() == [r * q for r in range(b + 1)]
                        assert int(total) == b * q
                        assert page_idx[: len(expected_pages)].tolist() == expected_pages
                        assert (page_idx[len(expected_pages) :] == -123).all()
                        assert last.tolist() == [(n - 1) % p + 1 if n else 0 for n in lens]
                        effective_pages = [(length + p - 1) // p for length in lens]
                        assert (
                            page_ptr.tolist()
                            == [0] + torch.tensor(effective_pages).cumsum(0).tolist()
                        )
                        if window_left >= 0:
                            effective_pages = [
                                min(n, (window_left + 128 + p - 1) // p) for n in effective_pages
                            ]
                        chunks = [
                            (n + int(chunk) // p - 1) // (int(chunk) // p) for n in effective_pages
                        ]
                        # Fixed rows: an inactive row's Q outputs own empty merge ranges.
                        expected_merge = [0]
                        for count in chunks:
                            for _ in range(q):
                                expected_merge.append(expected_merge[-1] + count)
                        assert merge.tolist() == expected_merge
                        assert valid.sum() == sum(chunks) * query_tiles
                        assert arrays[0][valid].tolist() == [
                            r for r, n in enumerate(chunks) for _ in range(query_tiles * n)
                        ]
                        assert arrays[1][valid].tolist() == [
                            tile for n in chunks for tile in range(query_tiles) for _ in range(n)
                        ]
                        assert arrays[2][valid].tolist() == [
                            j for n in chunks for _ in range(query_tiles) for j in range(n)
                        ]


def check_fixed_query_tiers():
    from ctikernels.attention._flashinfer import FIXED_QUERY_MIN_CHUNK
    from ctikernels.attention._flashinfer_schedule import prepare_split_kv

    cases = [
        (
            FIXED_QUERY_MIN_CHUNK[12, 0, 256],
            ((512, 16), (513, 32), (768, 32), (769, 64), (1280, 64), (1281, 128)),
        ),
        (FIXED_QUERY_MIN_CHUNK[12, 1, 256], ((256, 32), (257, 128))),
        (((128, 0),), ((256, 128), (1281, 128))),
    ]
    for minimum in (16, 32, 64):
        cases.append((((minimum, (1 << 31) - 1),), ((512, minimum), (4096, minimum))))
    for tiers, boundaries in cases:
        padded = tiers + ((128, 0),) * (3 - len(tiers))
        for length, expected in boundaries:
            # Choose the longest request, regardless of its batch position.
            for lengths in ((8, length), (length, 8)):
                indptr = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
                arrays = [torch.empty(512, dtype=torch.int32) for _ in range(3)]
                offsets = torch.empty(3, dtype=torch.int32)
                merge = torch.empty(9, dtype=torch.int32)
                chunk = torch.zeros(1, dtype=torch.int32)
                valid = torch.empty(512, dtype=torch.bool)
                prepare_split_kv[(1,)](
                    indptr,
                    *arrays,
                    offsets,
                    merge,
                    chunk,
                    valid,
                    BATCH=2,
                    QUERY=4,
                    PAGE=1,
                    QUERY_TILE_COUNT=1,
                    MAX_CTAS=512,
                    PADDED_CTAS=512,
                    REQUEST_BLOCK=2,
                    TILE_BLOCK=512,
                    MERGE_BLOCK=8,
                    SEARCH_STEPS=2,
                    MIN_CHUNK=padded[0][0],
                    SHORT_TOKENS=padded[0][1],
                    MIN_CHUNK_2=padded[1][0],
                    SHORT_TOKENS_2=padded[1][1],
                    MIN_CHUNK_3=padded[2][0],
                    SHORT_TOKENS_3=padded[2][1],
                )
                assert chunk.item() == expected, (tiers, lengths, chunk.item())


if __name__ == "__main__":
    assert not torch.cuda.is_initialized()
    check_schedule()
    check_fixed_query_tiers()
    assert not torch.cuda.is_initialized()
