# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only CuTe identity partitions; requires the optional CuTe DSL package.

These tests build layout IR only: no compilation, CUDA context or GPU launch.
They exercise the production chain MMA and swizzle, not handwritten stand-ins.
"""

import pytest


@pytest.fixture
def layouts():
    cute = pytest.importorskip("cutlass.cute")
    from cutlass._mlir import ir

    from ctikernels.gdn._cute_prefill_kernels import _layout, _mma

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield cute, _mma(1, 16, 8), _layout(64, 128)


@pytest.mark.parametrize("columns", [64, 128])
def test_chain_c_to_a_identity_order(layouts, columns):
    cute, mma, _ = layouts
    for lane in range(32):
        thread = mma.get_slice(lane)
        identity = cute.make_identity_tensor((16, columns))
        ca, cc = thread.partition_A(identity), thread.partition_C(identity)
        assert cute.size(ca) == cute.size(cc) == columns // 2
        for e in range(cute.size(ca)):
            # Covers full residual C->A and compact H's e//4*2+e%4 mapping.
            assert ca[e] == cc[e]
            row, col = cc[e]
            assert (row, col) == (
                lane // 4 + 8 * (e // 2 % 2),
                2 * (lane % 4) + e % 2 + 8 * (e // 4),
            )
            assert (e % 4 < 2) == (row < 8)


def test_update_b_identity_order_and_swizzled_addresses(layouts):
    cute, mma, sk_layout = layouts
    addresses = set()
    for lane in range(32):
        thread = mma.get_slice(lane)
        cb = thread.partition_B(cute.make_identity_tensor((128, 64)))
        rb = thread.make_fragment_B(cb.shape)
        assert rb.shape == cb.shape
        for kk in range(4):
            ck = cb[None, None, kk]
            assert cute.size(ck) == cute.size(rb[None, None, kk]) == 64
            for e in range(cute.size(ck)):
                ki, ti = ck[e]
                j, b = divmod(e, 4)
                # SM80 BLayout ((4,8),(2,2)):((16,1),(8,64)).
                assert (ki, ti) == (
                    8 * j + lane // 4,
                    16 * kk + 2 * (lane % 4) + b % 2 + 8 * (b // 2),
                )
                addresses.add(sk_layout((ti, ki)))
    assert addresses == set(range(64 * 128))


@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_state_and_h1_identity_coordinates(layouts, vsplit):
    cute, mma, _ = layouts
    components, rows, vs = (4 if vsplit == 1 else 2), 16 // vsplit, 128 // vsplit
    state_coordinates, dump_coordinates = set(), set()
    for lane in range(32):
        thread = mma.get_slice(lane)
        ch = thread.partition_C(cute.make_identity_tensor((16, 128)))
        ca = thread.partition_A(cute.make_identity_tensor((16, 128)))
        for n in range(16):
            for c in range(components):
                e, compact_e = n * 4 + c, n * components + c
                vi, ki = ch[e]
                assert ch[e] == ch[((c % 2, c // 2), 0, n)]
                assert (vi, ki) == (lane // 4 + 8 * (c // 2), n * 8 + 2 * (lane % 4) + c % 2)
                assert ca[e] == ch[e]
                assert compact_e == (e if vsplit == 1 else e // 4 * 2 + e % 4)
                if vi >= rows:
                    continue
                for part in range(vsplit):
                    for wid in range(8):
                        vrow = wid * rows + vi
                        state_coord = (part * vs + vrow, ki)
                        dump_coord = (part, 10 + vrow // 64, vrow % 64, ki)
                        assert state_coord not in state_coordinates
                        assert dump_coord not in dump_coordinates
                        state_coordinates.add(state_coord)
                        dump_coordinates.add(dump_coord)
                        # H1 slots reconstruct slice-local V rows, then global V.
                        assert (
                            part * vs + (dump_coord[1] - 10) * 64 + dump_coord[2],
                            ki,
                        ) == state_coord
    assert state_coordinates == {(v, k) for v in range(128) for k in range(128)}
    assert len(dump_coordinates) == 128 * 128


def test_diagonal_tf32_and_bf16_c_partitions_match(layouts):
    cute, _, _ = layouts
    from cutlass.cute.nvgpu import warp

    from ctikernels.gdn._cute_prefill_kernels import _mma

    bf16 = _mma(1, 16, 16)
    tf32 = cute.make_tiled_mma(
        warp.MmaTF32Op((16, 8, 8)),
        cute.make_layout((1, 1, 1)),
        permutation_mnk=(16, 16, 8),
    )
    identity = cute.make_identity_tensor((16, 16))
    for lane in range(32):
        bc = bf16.get_slice(lane).partition_C(identity)
        tc = tf32.get_slice(lane).partition_C(identity)
        assert bc.shape == tc.shape
        assert [bc[e] for e in range(cute.size(bc))] == [tc[e] for e in range(cute.size(tc))]


@pytest.mark.parametrize("vsplit", [1, 2, 4])
def test_norm_lane_groups_own_exact_v_columns(layouts, vsplit):
    cute, mma, _ = layouts
    rows = 16 // vsplit
    fragments = [
        mma.get_slice(lane).partition_C(cute.make_identity_tensor((16, 64))) for lane in range(32)
    ]
    for n in range(8):
        for c in range(2):
            for writer in range(4):
                owned = set()
                for group in range(min(rows, 8)):
                    lane = writer + 4 * group
                    for e in [4 * n + c, 4 * n + c + 2] if rows == 16 else [4 * n + c]:
                        vi, ti = fragments[lane][e]
                        assert ti == 8 * n + 2 * writer + c
                        owned.add(vi)
                assert owned == set(range(rows))
