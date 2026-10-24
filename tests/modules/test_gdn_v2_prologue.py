# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU models of CP1 ownership/rounding; no CUDA or kernel compilation.

These check the math and memory ownership, not generated PTX or hardware
rounding. The optional identity-partition tests check the actual CuTe layout.
"""

import pytest
import torch


def _split_mma(a, b, acc):
    ah, bh = a.bfloat16().float(), b.bfloat16().float()
    al, bl = (a - ah).bfloat16().float(), (b - bh).bfloat16().float()
    # Production _small_mma's four-term accumulation order.
    for left, right in ((al, bl), (al, bh), (ah, bl), (ah, bh)):
        acc = acc + left @ right
    return acc


def _tf32(x):
    # cvt.rna.tf32.f32, same conversion as cute.arch.cvt_f32_tf32.
    return ((x.contiguous().view(torch.int32) + 4096) & -8192).view(torch.float32)


def _diagonal_mma(a, b, acc):
    ah, bh = _tf32(a), _tf32(b)
    al, bl = _tf32(a - ah), _tf32(b - bh)
    for kk in (0, 8):
        for left, right in ((al, bh), (ah, bl), (ah, bh)):
            acc = acc + left[:, kk : kk + 8] @ right[kk : kk + 8]
    return acc


def _inverse(lower, *, scalar_diagonal=False):
    inverse = torch.zeros_like(lower)
    eye = torch.eye(16)
    for block in range(4):
        sl = slice(block * 16, (block + 1) * 16)
        block_lower = lower[sl, sl]
        r, c = torch.arange(16)[:, None], torch.arange(16)[None, :]
        diag = eye - torch.where((r == c + 1) & (r // 2 == c // 2), block_lower, 0)
        for span in (2, 4, 8):
            mask = (
                (r // (2 * span) == c // (2 * span))
                & (r % (2 * span) >= span)
                & (c % (2 * span) < span)
            )
            cross = torch.where(mask, block_lower, 0)
            tmp = -_diagonal_mma(diag, cross, torch.zeros_like(diag))
            diag = _diagonal_mma(tmp, diag, diag)
        if scalar_diagonal:
            # Original v1 pivot order, retaining the identical off-diagonal path.
            for row in range(16):
                x = -block_lower[row].clone()
                x[row] = 1
                for pivot in range(row - 1, 0, -1):
                    x[:pivot] -= x[pivot] * block_lower[pivot, :pivot]
                diag[row] = x
        inverse[sl, sl] = diag
    for distance in (1, 2, 3):
        for j in range(4 - distance):
            i = j + distance
            rows, cols = slice(i * 16, (i + 1) * 16), slice(j * 16, (j + 1) * 16)
            acc = torch.zeros(16, 16)
            for offset in range(1, distance + 1):
                inner = slice((j + offset) * 16, (j + offset + 1) * 16)
                acc = _split_mma(inverse[rows, inner], lower[inner, cols], acc)
            inverse[rows, cols] = _split_mma(-acc, inverse[cols, cols], torch.zeros_like(acc))
    return inverse


@pytest.mark.parametrize("length", [1, 15, 16, 17, 63, 64])
@pytest.mark.parametrize("kind", ["random", "correlated", "unit"])
def test_recursive_blocked_inverse_fp64(length, kind):
    generator = torch.Generator().manual_seed(451 + length)
    k = torch.randn(64, 128, generator=generator)
    if kind == "correlated":
        k = 0.1 * k + torch.randn(1, 128, generator=generator)
    if kind == "unit":
        k.fill_(0)
        k[:, 0] = 1
    k = (k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16().float()
    k[length:] = 0
    beta = torch.rand(64, generator=generator) if kind != "unit" else torch.ones(64)
    decay = (
        torch.zeros(64) if kind == "unit" else -torch.rand(64, generator=generator).cumsum(0) * 0.01
    )
    lower = torch.tril((k @ k.T) * beta[:, None] * (decay[:, None] - decay[None, :]).exp(), -1)
    actual = _inverse(lower).double()
    triangular = torch.eye(64, dtype=torch.float64) + lower.double()
    expected = torch.linalg.solve_triangular(
        triangular, torch.eye(64, dtype=torch.float64), upper=False
    )
    assert torch.equal(actual.triu(1), torch.zeros_like(actual))
    assert torch.equal(actual.diag(), torch.ones(64, dtype=torch.float64))
    baseline = _inverse(lower, scalar_diagonal=True).double()
    # The unchanged off-diagonal BF16 path can exceed 8e-6 on correlated
    # inputs; require its error scale, and a separate tight diagonal bound.
    assert (actual - expected).abs().max() <= max(8e-6, 1.02 * (baseline - expected).abs().max())
    baseline_residual = (triangular @ baseline - torch.eye(64)).abs().max()
    assert (triangular @ actual - torch.eye(64)).abs().max() <= max(8e-6, 1.02 * baseline_residual)
    for block in range(4):
        sl = slice(block * 16, (block + 1) * 16)
        assert (actual[sl, sl] - expected[sl, sl]).abs().max() < 1e-6


@pytest.mark.parametrize("columns", [32, 64, 128])
@pytest.mark.parametrize("length", [1, 2, 3, 17, 64, 65, 129])
@pytest.mark.parametrize("history_live", [False, True])
def test_vector_halo_and_sliding_conv(columns, length, history_live):
    generator = torch.Generator().manual_seed(length + columns)
    begin = 5
    x = torch.randn(begin + length + 3, columns, generator=generator).bfloat16().float()
    history = torch.randn(3, columns, generator=generator).bfloat16().float()
    weights = torch.randn(columns, 4, generator=generator).bfloat16().float()
    for start in range(begin, begin + length, 64):
        raw = torch.full((67, columns), float("nan"))
        copied = set()
        for tid in range(256):
            for v in range(tid, 67 * (columns // 8), 256):
                row, col = v // (columns // 8), v % (columns // 8) * 8
                token = start - 3 + row
                use_history = token < begin and history_live
                valid = begin <= token < begin + length or use_history
                source = (
                    history[token - begin + 3] if use_history else x[min(max(token, 0), len(x) - 1)]
                )
                raw[row, col : col + 8] = source[col : col + 8] if valid else 0
                assert (row, col) not in copied
                copied.add((row, col))
        assert len(copied) == 67 * (columns // 8)
        got = torch.full((64, columns), float("nan"))
        owned = set()
        for tid in range(columns * 2):
            first, col = tid // (columns // 8) * 4, tid % (columns // 8) * 8
            ring = torch.empty(4, 8)
            ring[:3] = raw[first : first + 3, col : col + 8]
            for step in range(4):
                row = first + step
                ring[(step + 3) % 4] = raw[row + 3, col : col + 8]
                value = torch.zeros(8)
                for tap in range(4):
                    value += ring[(step + tap) % 4] * weights[col : col + 8, tap]
                got[row, col : col + 8] = value
                assert (row, col) not in owned
                owned.add((row, col))
        expected = sum(raw[tap : tap + 64] * weights[:, tap] for tap in range(4))
        assert torch.equal(got, expected)
        assert len(owned) == 64 * (columns // 8)
        # Same stable activation form and tail zeroing. rcp.approx itself needs GPU validation.
        t = (-got.abs()).exp()
        silu = got * torch.where(got < 0, t, 1) / (1 + t)
        live = min(64, begin + length - start)
        silu[live:] = 0
        reference = torch.nn.functional.silu(expected.double())
        reference[live:] = 0
        torch.testing.assert_close(silu.double(), reference, atol=2e-6, rtol=2e-6)
        scale = (
            (silu.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
            if columns == 128
            else torch.rand(64, 1, generator=generator)
        )
        result = silu * scale
        hi = result.bfloat16().float()
        lo = (result - hi).bfloat16().float()
        torch.testing.assert_close(hi + lo, result, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_register_norm_partials_and_output_ownership(vsplit):
    generator = torch.Generator().manual_seed(77)
    output = torch.randn(64, 128, generator=generator)
    vs, rows = 128 // vsplit, 16 // vsplit
    partial = torch.full((64, 4), float("nan"))
    reconstructed = torch.full_like(output, float("nan"))
    for part in range(vsplit):
        norm = torch.full((8, 64), float("nan"))
        for wid in range(8):
            # Poison padded MMA rows: they must never contribute to the norm.
            fragments = torch.full((32, 32), float("nan"))
            for lane in range(32):
                for e in range(32):
                    vi, ti = lane // 4 + 8 * (e // 2 % 2), 2 * (lane % 4) + e % 2 + 8 * (e // 4)
                    if vi < rows:
                        v = part * vs + wid * rows + vi
                        fragments[lane, e] = output[ti, v]
                        reconstructed[ti, v] = fragments[lane, e]
            for n in range(8):
                for c in range(2):
                    square = fragments[:, n * 4 + c].square()
                    if rows == 16:
                        square += fragments[:, n * 4 + c + 2].square()
                    for shift in range(2 if rows == 4 else 3):
                        square = square + square[torch.arange(32) ^ (4 << shift)]
                    for lane in range(4):
                        norm[wid, n * 8 + 2 * lane + c] = square[lane]
        for group in range(vs // 32):
            partial[:, part * vs // 32 + group] = norm[
                group * 2 * vsplit : (group + 1) * 2 * vsplit
            ].sum(0)
    torch.testing.assert_close(reconstructed, output, atol=0, rtol=0)
    torch.testing.assert_close(
        partial, output.reshape(64, 4, 32).square().sum(-1), atol=1e-5, rtol=2e-7
    )
