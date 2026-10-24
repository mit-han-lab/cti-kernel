# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU address/fragment model of the original small-M PTX kernel.

No CUDA/DSL dependency. Separately audit the orthogonal row, K and stage axes
rather than enumerate every element of the vocabulary head. Addresses are in
BF16 elements (FP32 elements for partials). This checks the source mathematics,
not the compiler's lowering or CUDA graph allocation lifetimes.
"""

from itertools import product

import pytest
import torch

from ctikernels.linear._dispatch import QWEN_SHAPES, Config, config_for, tune_configs


def configs():
    for rows, tile, stages in product((16, 32, 64, 128), (64, 128, 256, 512), (2, 3, 4, 5, 6)):
        try:
            yield Config(rows, tile, stages)
            if rows == 128:
                yield Config(rows, tile, stages, warps=8)
        except ValueError:
            pass


@pytest.mark.parametrize("mtiles", range(1, 9))
@pytest.mark.parametrize("cfg", list(configs()))
def test_shared_addresses_and_fragments(cfg, mtiles):
    pitch = cfg.tile_k + 8
    xrows = 8 * mtiles
    stage_size = (cfg.rows + xrows) * pitch
    threads = cfg.warps * 32
    initialized = set()
    for row_count, start in ((cfg.rows, 0), (xrows, cfg.rows * pitch)):
        for tid in range(threads):
            for j in range((row_count * cfg.tile_k // 8 + threads - 1) // threads):
                vec = tid + j * threads
                row, col = divmod(vec, cfg.tile_k // 8)
                col *= 8
                if row >= row_count:
                    continue
                assert row < row_count
                offset = start + row * pitch + col
                assert offset % 8 == 0
                assert 0 <= offset < offset + 8 <= stage_size
                for element in range(offset, offset + 8):
                    assert element not in initialized
                    initialized.add(element)
                for stage in range(cfg.stages):
                    assert 2 * (stage * stage_size + offset + 8) <= cfg.shared_bytes_for(mtiles)
    assert len(initialized) == (cfg.rows + xrows) * cfg.tile_k
    for warp, r, kk, t in product(
        range(cfg.warps),
        range(cfg.rows // (16 * cfg.warps)),
        range(cfg.tile_k // 16),
        range(mtiles),
    ):
        row = warp * 16 + r * cfg.warps * 16
        a_addresses = [(row + lane % 16) * pitch + lane // 16 * 8 + kk * 16 for lane in range(32)]
        b_addresses = [
            cfg.rows * pitch + (t * 8 + lane % 8) * pitch + (lane // 8) % 2 * 8 + kk * 16
            for lane in range(32)
        ]
        # ldmatrix non-transposed: lane//4 selects a matrix row; lane%4
        # selects its packed pair. Matrix i takes addresses from lanes 8*i..8*i+7.
        for lane in range(32):
            for reg, half in product(range(4), range(2)):
                address = a_addresses[reg * 8 + lane // 4] + (lane % 4) * 2 + half
                assert address in initialized
                expected_row = row + lane // 4 + (reg % 2) * 8
                expected_k = kk * 16 + (reg // 2) * 8 + (lane % 4) * 2 + half
                assert divmod(address, pitch) == (expected_row, expected_k)
            for reg, half in product(range(2), range(2)):
                address = b_addresses[reg * 8 + lane // 4] + (lane % 4) * 2 + half
                assert address in initialized
                # Physical x[M,K] is mathematical B[K,M]: both K halves
                # match the K halves of A's register pairs.
                assert divmod(address - cfg.rows * pitch, pitch) == (
                    t * 8 + lane // 4,
                    kk * 16 + reg * 8 + (lane % 4) * 2 + half,
                )


def cases():
    result = set()
    for n, k in (*QWEN_SHAPES, (1, 8), (67, 264), (129, 520)):
        for kind, sms in (("rtx5090", 170), ("pro6000", 188), ("gb10", 48)):
            for cfg in tune_configs(config_for(kind, sms, n, k), n, k, kind):
                result.add((n, k, cfg))
    # Previously misreported child exit: exactly 34 chunks in each split.
    result.add((5120, 17408, Config(64, 256, 2, 2, 4)))
    # Cover every accepted shared layout, empty splits, excess prefetches, and
    # K shorter than a stage. Rows=128 needs two MMA operations per warp.
    for cfg in configs():
        for n, k, splits in ((1, 8, 64), (67, 264, 16), (129, 520, 4), (64, 4096, 32)):
            result.add((n, k, Config(cfg.rows, cfg.tile_k, cfg.stages, splits, cfg.warps)))
    return sorted(result, key=lambda item: (item[0], item[1], repr(item[2])))


@pytest.mark.parametrize("n,k,cfg", cases())
def test_global_addresses_and_split_coverage(n, k, cfg):
    chunks = (k + cfg.splits * cfg.tile_k - 1) // (cfg.splits * cfg.tile_k)
    covered = []
    for split in range(cfg.splits):
        slots = [None] * cfg.stages
        for chunk in range(chunks + cfg.stages - 1):
            # Includes the last stages-1 zero-fill prefetches.
            base = (split * chunks + chunk) * cfg.tile_k
            slots[chunk % cfg.stages] = chunk
            for col in range(0, cfg.tile_k, 8):
                valid = chunk < chunks and base + col < k
                address = base + col if valid else 0
                assert 0 <= address <= k - 8  # even a masked source is valid
                if valid:
                    covered.append(address)
            consumed = chunk - cfg.stages + 1
            if consumed >= 0:
                assert slots[consumed % cfg.stages] == consumed
    assert sorted(covered) == list(range(0, k, 8))
    # N is affine in ntile: first and last blocks cover the address extrema.
    for ntile in {0, (n - 1) // cfg.rows}:
        written = set()
        for warp, r, lane, v in product(
            range(cfg.warps), range(cfg.rows // (16 * cfg.warps)), range(32), range(4)
        ):
            nr = ntile * cfg.rows + warp * 16 + r * cfg.warps * 16 + lane // 4 + 8 * (v // 2)
            mr = 2 * (lane % 4) + v % 2
            for m in range(1, 9):
                # Model the last valid K vector and the masked N/M base pointer.
                w_address = nr * k + k - 8 if nr < n else 0
                x_address = mr * k + k - 8 if mr < m else 0
                assert 0 <= w_address <= n * k - 8
                assert 0 <= x_address <= m * k - 8
                if nr < n and mr < m:
                    assert (m, mr, nr) not in written
                    written.add((m, mr, nr))
                    for split in {0, cfg.splits - 1}:
                        offset = split * m * n + mr * n + nr
                        assert 0 <= offset < cfg.splits * m * n
        assert len(written) == sum(range(1, 9)) * min(cfg.rows, n - ntile * cfg.rows)
    for m in range(1, 9):
        # All reduction lanes are linear, so check both ends of the last CTA.
        grid = (m * n + 255) // 256
        for idx in ((grid - 1) * 256, grid * 256 - 1, m * n - 1):
            if idx < m * n:
                assert 0 <= idx < m * n
                assert (cfg.splits - 1) * m * n + idx < cfg.splits * m * n


@pytest.mark.parametrize("n", [2, 64, 66, 96, 4096, 5120, 248320])
@pytest.mark.parametrize("m", range(1, 65))
def test_vector_reduction_alignment_and_tails(n, m):
    total = n * m
    blocks = (total + 63) // 64
    for block in {0, blocks - 1}:
        written = []
        for tid in range(32):
            idx = block * 64 + tid * 2
            if idx < total:
                assert idx + 1 < total
                assert idx * 2 % 4 == 0  # packed BF16 output store
                for split in (0, 1, 16, 33, 63):
                    address = split * total + idx
                    assert address * 4 % 8 == 0  # v2.f32 load
                    assert address + 1 < (split + 1) * total
                written.extend((idx, idx + 1))
        assert written == list(range(block * 64, min(total, (block + 1) * 64)))


@pytest.mark.parametrize("k", [8, 24, 32, 40, 264, 4096, 17408])
@pytest.mark.parametrize(
    "m,n", [(8 * t - tail, 13 if tail else 16) for t in range(1, 9) for tail in (0, 7)]
)
def test_register_fragment_permutation_dot_product(k, m, n):
    rng = torch.Generator().manual_seed(7)
    w = torch.randn((n, k), generator=rng, dtype=torch.float64)
    x = torch.randn((m, k), generator=rng, dtype=torch.float64)
    blocks = (k + 31) // 32
    # Zero-fill K, N and M exactly as the predicated kernel loads do.
    wp = torch.zeros((16, blocks * 32), dtype=torch.float64)
    xrows = 8 * ((m + 7) // 8)
    xp = torch.zeros((xrows, blocks * 32), dtype=torch.float64)
    wp[:n, :k], xp[:m, :k] = w, x
    wp, xp = wp.reshape(16, blocks, 32), xp.reshape(xrows, blocks, 32)
    a = torch.zeros((blocks, 2, 16, 16), dtype=torch.float64)
    b = torch.zeros((blocks, 2, 16, xrows), dtype=torch.float64)
    visited = set()
    for lane in range(32):
        g, t = divmod(lane, 4)
        # Two 128-bit vectors, each four packed BF16 pairs, from global W.
        loads = wp[[g, g + 8], :, t * 8 : t * 8 + 8]
        for s in range(2):
            aregs = loads[:, :, s * 4 : s * 4 + 4].reshape(2, blocks, 2, 2)
            for reg, h in product(range(4), range(2)):
                # Public PTX A fragment: packed regs alternate row halves,
                # then advance by eight logical K positions.
                row, col = g + (reg % 2) * 8, t * 2 + (reg // 2) * 8 + h
                a[:, s, row, col] = aregs[reg % 2, :, reg // 2, h]
            # ld.shared.v2.b32 reads the same four contiguous x values.
            for tile in range(xrows // 8):
                bregs = xp[tile * 8 + g, :, t * 8 + s * 4 : t * 8 + s * 4 + 4].reshape(blocks, 2, 2)
                for q, h in product(range(2), range(2)):
                    j = t * 2 + q * 8 + h
                    b[:, s, j, tile * 8 + g] = bregs[:, q, h]
                    physical = t * 8 + s * 4 + q * 2 + h
                    if g == 0 and tile == 0:
                        assert physical not in visited
                        visited.add(physical)
    assert visited == set(range(32))
    actual = (a @ b).sum(dim=(0, 1)).T[:m, :n]
    torch.testing.assert_close(actual, x @ w.T, rtol=1e-11, atol=1e-10)
    # A mismatched B permutation must be observable on the same random data.
    wrong = (a @ b.flip(2)).sum(dim=(0, 1)).T[:m, :n]
    assert not torch.allclose(wrong, x @ w.T, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("k", [8, 24, 64, 72, 264, 4096])
@pytest.mark.parametrize("m,n", [(1, 16), (2, 13), (8, 16), (9, 16), (16, 11)])
def test_wide_exchange_fragment_dot_product(k, m, n):
    # Wide variant: lane L loads row L//8 + 4*i, columns 8*(L%8)..+8 of a 64-column
    # block (4 rows x 128 bytes per instruction); _exchange swaps one row pair with L^4.
    rng = torch.Generator().manual_seed(11)
    w = torch.randn((n, k), generator=rng, dtype=torch.float64)
    x = torch.randn((m, k), generator=rng, dtype=torch.float64)
    blocks = (k + 63) // 64
    wp = torch.zeros((16, blocks * 64), dtype=torch.float64)
    xrows = 8 * ((m + 7) // 8)
    xp = torch.zeros((xrows, blocks * 64), dtype=torch.float64)
    wp[:n, :k], xp[:m, :k] = w, x
    wp, xp = wp.reshape(16, blocks, 64), xp.reshape(xrows, blocks, 64)
    loads = [
        [wp[lane // 8 + 4 * i, :, lane % 8 * 8 : lane % 8 * 8 + 8] for i in range(4)]
        for lane in range(32)
    ]
    a = torch.zeros((blocks, 2, 2, 16, 16), dtype=torch.float64)
    b = torch.zeros((blocks, 2, 2, 16, xrows), dtype=torch.float64)
    for lane in range(32):
        g, t = divmod(lane, 4)
        odd, partner = g % 2, lane ^ 4
        frag = [[None, None], [None, None]]
        for q in range(2):
            # PTX selp/shfl semantics of _exchange.
            sent = loads[partner][2 * q + (0 if (partner // 4) % 2 else 1)]
            keep = loads[lane][2 * q + odd]
            frag[0][q], frag[1][q] = (sent, keep) if odd else (keep, sent)
        for half, s in product(range(2), range(2)):
            for reg, h in product(range(4), range(2)):
                label, col = g + (reg % 2) * 8, t * 2 + (reg // 2) * 8 + h
                a[:, half, s, label, col] = frag[half][reg % 2][:, s * 4 + (reg // 2) * 2 + h]
            for tile in range(xrows // 8):
                for q, h in product(range(2), range(2)):
                    physical = half * 32 + t * 8 + s * 4 + q * 2 + h
                    b[:, half, s, t * 2 + q * 8 + h, tile * 8 + g] = xp[tile * 8 + g, :, physical]
    labels = [label % 8 // 2 + 4 * (label % 2) + 8 * (label // 8) for label in range(16)]
    assert sorted(labels) == list(range(16))  # epilogue row of each MMA label
    actual = torch.zeros((16, xrows), dtype=torch.float64)
    actual[labels] = (a @ b).sum(dim=(0, 1, 2))
    torch.testing.assert_close(actual[:n, :m].T, x @ w.T, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("warps", range(1, 9))
@pytest.mark.parametrize("rows_per_warp", [16, 32])
@pytest.mark.parametrize("depth", range(2, 9))
def test_register_prefetch_addresses(warps, rows_per_warp, depth):
    cfg = Config(warps * rows_per_warp, 128, splits=3, warps=warps, variant="reg", prefetch=depth)
    n, k = cfg.rows - 3, 264
    chunks = (k + cfg.splits * cfg.tile_k - 1) // (cfg.splits * cfg.tile_k)
    covered = set()
    for split, chunk, warp, r, lane in product(
        range(cfg.splits), range(chunks), range(warps), range(rows_per_warp // 16), range(32)
    ):
        base = (split * chunks + chunk) * cfg.tile_k
        for batch, d, half in product(
            range((cfg.tile_k // 32 + depth - 1) // depth), range(depth), range(2)
        ):
            block = batch * depth + d
            if block >= cfg.tile_k // 32:
                continue
            nr = warp * 16 + r * warps * 16 + lane // 4 + half * 8
            col = base + block * 32 + (lane % 4) * 8
            valid = nr < n and col < k
            address = nr * k + col if valid else 0
            assert address % 8 == 0 and 0 <= address <= n * k - 8
            if valid:
                assert (nr, col) not in covered
                covered.add((nr, col))
    assert covered == set(product(range(n), range(0, k, 8)))
    assert cfg.shared_bytes == 8 * (cfg.tile_k + 8) * 2


@pytest.mark.parametrize("variant", ["reg", "smem"])
@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("rows_per_warp", [16, 32])
def test_bxq_output_and_x_staging(variant, warps, rows_per_warp):
    rows = warps * rows_per_warp
    if variant == "smem" and rows > 128:
        return  # outside the smem config space
    cfg = Config(rows, 128, warps=warps, variant=variant)
    n = rows + 3  # N tail, multiple CTAs
    threads = warps * 32
    for mtiles in range(1, 9):
        staged = set()
        for tid in range(threads):
            for j in range((8 * mtiles * cfg.tile_k // 8 + threads - 1) // threads):
                vec = tid + j * threads
                row, col = divmod(vec, cfg.tile_k // 8)
                if row < 8 * mtiles:
                    for offset in range(8):
                        address = (row, col * 8 + offset)
                        assert address not in staged
                        staged.add(address)
        assert staged == set(product(range(8 * mtiles), range(cfg.tile_k)))
        for m in range(8 * (mtiles - 1) + 1, 8 * mtiles + 1):
            written = set()
            for ntile, warp, r, lane, t, v in product(
                range((n + rows - 1) // rows),
                range(warps),
                range(rows_per_warp // 16),
                range(32),
                range(mtiles),
                range(4),
            ):
                nr = ntile * rows + warp * 16 + r * warps * 16 + lane // 4 + (v // 2) * 8
                mr = t * 8 + (lane % 4) * 2 + v % 2
                if nr < n and mr < m:
                    assert (mr, nr) not in written
                    written.add((mr, nr))
            assert written == set(product(range(m), range(n)))
