#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Integer address oracle for CP2's unchanged shared layouts (no CUDA required).

Counts distinct 32-bit words per bank, treating reads of the same word as a
broadcast. LDS/STS.64 has two ideal 128-byte wavefronts, one per half warp;
report excess wavefronts separately from the whole-warp bank multiplicity.
"""

import json
from collections import defaultdict


def swizzle(address, mask):
    return address ^ ((address >> 3) & mask)


def k_address(row, d, base=256):
    return swizzle(base + row * 128 + (d % 64) * 2 + (d // 64) * 8192, 0x70)


def o_address(d, token, base=82176):
    return swizzle(base + (d % 16) * 2 + token * 32 + (d // 16) * 2048, 0x10)


def y_coordinates(lane, group, quarter):
    return (lane % 4) * 4 + (lane // 16) * 16 + quarter * 32, group * 4 + (lane // 4) % 4


def bank_degree(addresses, width=8):
    banks = defaultdict(set)
    for address in addresses:
        for offset in range(0, width, 4):
            word = (address + offset) // 4
            banks[word % 32].add(word)
    return max(map(len, banks.values()))


def access_report(addresses, width=8):
    lanes_per_wave = 128 // width
    ideal = (len(addresses) + lanes_per_wave - 1) // lanes_per_wave
    actual = sum(
        bank_degree(addresses[i : i + lanes_per_wave], width)
        for i in range(0, len(addresses), lanes_per_wave)
    )
    return {
        "bank_multiplicity": bank_degree(addresses, width),
        "ideal_wavefronts": ideal,
        "predicted_wavefronts": actual,
        "excess_wavefronts": actual - ideal,
    }


def main():
    print(
        json.dumps(
            {
                "k_q_v_row_and_boundary": access_report(
                    [k_address(13, lane * 4) for lane in range(32)]
                ),
                "y_before": access_report([o_address(lane * 4, 0) for lane in range(32)]),
                "y_after": access_report(
                    [o_address(*y_coordinates(lane, 0, 0)) for lane in range(32)]
                ),
                "gate_write_half_warp": access_report([lane * 4 for lane in range(16)], 4),
                "gate_scan": access_report([lane * 4 for lane in range(32)], 4),
                "math_k_matrix_wave": access_report([k_address(row, 0) for row in range(8)], 16),
                "math_o_matrix_wave": access_report([o_address(0, row) for row in range(8)], 16),
                "math_qk_inverse_matrix_wave": access_report([row * 16 for row in range(8)], 16),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
