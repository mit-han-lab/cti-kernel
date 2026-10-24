# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Whole-block crossover data; larger or unknown devices use composition."""

_BLOCK_TABLE = {
    "rtx5090": [(64, "triton_block")],
    "pro6000": [(64, "triton_block")],
    "gb10": [(128, "triton_block")],
}
