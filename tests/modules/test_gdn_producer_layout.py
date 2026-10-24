# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU proofs for the producer's address and SiLU simplifications."""

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("output", [False, True])
def test_eight_row_address_phases_match_cute_layout(output):
    cutlass = pytest.importorskip("cutlass")
    import cutlass.cute as cute
    from cutlass._mlir import ir
    from cutlass.cute.nvgpu import warpgroup

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            kind = (
                warpgroup.SmemLayoutAtomKind.MN_SW32
                if output
                else warpgroup.SmemLayoutAtomKind.K_SW128
            )
            atom = warpgroup.make_smem_layout_atom(kind, cutlass.BFloat16)
            layout = cute.tile_to_shape(
                atom,
                (128, 64, 1) if output else (64, 128, 2),
                order=(1, 0, 2) if output else (0, 1, 2),
            )
            # Evaluate CuTe's actual swizzle as a byte-address oracle. In
            # particular, an allocation aligned to 128 need not align to 1024.
            byte_map = cute.make_composed_layout(layout.inner, 0, cute.make_layout((1 << 18,)))
            mask, row8_bytes = (0x10, 256) if output else (0x70, 1024)
            for stage, role, lane in itertools.product(
                range(1 if output else 2), range(4), range(32)
            ):
                offsets = [
                    2
                    * layout.outer(
                        (lane * 4, role * 16 + r, stage)
                        if output
                        else (role * 16 + r, lane * 4, stage)
                    )
                    for r in range(16)
                ]
                for allocation in range(0x4000, 0x4400, 128):
                    phases = [
                        (allocation + offset) ^ (((allocation + offset) >> 3) & mask)
                        for offset in offsets[:8]
                    ]
                    for r in range(16):
                        address = phases[r % 8] + (r // 8) * row8_bytes
                        assert address == byte_map(allocation + offsets[r])
                        assert address % 8 == 0  # one aligned LDS/STS.64
                        oracle = (
                            banks.o_address(lane * 4, role * 16 + r, allocation)
                            if output
                            else banks.k_address(
                                role * 16 + r, lane * 4, allocation + stage * 16384
                            )
                        )
                        assert address == oracle


def test_silu_exp2_ftz_does_not_change_fp32_denominator():
    # Every BF16 bit pattern covers signed subnormals, normals, infinities and
    # NaNs; include dense FP32 samples around exp2's output-underflow boundary.
    exponent = torch.cat(
        (
            (torch.arange(65536, dtype=torch.int32) << 16).view(torch.float32),
            torch.linspace(-150, -125, 4096),
        )
    )
    tiny = torch.finfo(torch.float32).tiny
    ftz_input = torch.where(exponent.abs() < tiny, 0.0, exponent)
    ftz_output = torch.exp2(ftz_input)
    ftz_output = torch.where(ftz_output.abs() < tiny, 0.0, ftz_output)
    torch.testing.assert_close(
        1 + torch.exp2(exponent), 1 + ftz_output, rtol=0, atol=0, equal_nan=True
    )


path = Path(__file__).resolve().parents[2] / "scripts/analyze_gdn_producer_banks.py"
spec = importlib.util.spec_from_file_location("gdn_banks", path)
banks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(banks)


@pytest.mark.parametrize("base", range(0, 1024, 128))
def test_shared_accesses_all_swizzle_phases(base):
    for layout in (
        lambda row: banks.k_address(row, 0, base),
        lambda row: banks.o_address(0, row, base),
        lambda row: base + row * 16,
    ):
        assert banks.access_report([layout(row) for row in range(8)], 16)["excess_wavefronts"] == 0
    for row in range(64):
        addresses = [banks.k_address(row, lane * 4, base) for lane in range(32)]
        assert banks.access_report(addresses)["excess_wavefronts"] == 0
        assert (
            banks.access_report([banks.o_address(lane * 4, row, base) for lane in range(32)])[
                "predicted_wavefronts"
            ]
            == 8
        )
    for role in range(4):
        seen = set()
        for group in range(4):
            for quarter in range(4):
                addresses = []
                for lane in range(32):
                    d, token = banks.y_coordinates(lane, group, quarter)
                    token += role * 16
                    address = banks.o_address(d, token, base)
                    # Device computes two real pointers, then adds constant offsets.
                    origin_d, origin_t = banks.y_coordinates(lane, group % 2, 0)
                    origin = banks.o_address(origin_d, role * 16 + origin_t, base)
                    assert address == origin + (group // 2) * 256 + quarter * 4096
                    addresses.append(address)
                    for j in range(4):
                        assert (d + j, token) not in seen
                        seen.add((d + j, token))
                assert banks.access_report(addresses)["excess_wavefronts"] == 0
        assert seen == {(d, t) for d in range(128) for t in range(role * 16, role * 16 + 16)}


def test_remapped_reduction_preserves_fp32_tree_and_token_isolation():
    gen = torch.Generator().manual_seed(319)
    scale = torch.empty(16, 128).uniform_(-8, 8, generator=gen).exp()
    values = torch.randn(16, 128, generator=gen) * scale
    # A common four-element accumulation; test only the changed reduction tree.
    partial = (values.reshape(16, 32, 4) ** 2).sum(dim=2)
    lanes = torch.arange(32)
    original = partial.clone()
    for shift in (1, 2, 4, 8, 16):
        original = original + original[:, lanes ^ shift]
    for group in range(4):
        remapped = torch.empty(32, 4)
        for lane in range(32):
            for quarter in range(4):
                d, t = banks.y_coordinates(lane, group, quarter)
                remapped[lane, quarter] = partial[t, d // 4]
        for shift in (1, 2, 16):
            remapped = remapped + remapped[lanes ^ shift]
        total = (remapped[:, 0] + remapped[:, 1]) + (remapped[:, 2] + remapped[:, 3])
        for lane in range(32):
            _, t = banks.y_coordinates(lane, group, 0)
            assert total[lane] == original[t, 0]


@pytest.mark.parametrize("option", ["q_lookahead", "remap_y"])
def test_producer_experiments_specialize_cache(option):
    from ctikernels.gdn.chunk_ws.prefill import PortOptions, compile_key

    normal, ahead = PortOptions(), PortOptions(**{option: True})
    assert not getattr(normal, option)
    assert compile_key(16, 48, 16480, normal) != compile_key(16, 48, 16480, ahead)
    with pytest.raises(TypeError, match=option):
        PortOptions(**{option: 1})
