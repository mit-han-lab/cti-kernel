# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""DFlash round bookkeeping and the DFlash2 block convolution."""

from .inference import (
    ALL_VARIANTS,
    RoundVariant,
    block_conv,
    commit_block,
    compile_count,
    fixed_query_addresses,
    prepare_block,
    prepare_round,
    warm_up,
)

__all__ = [
    "block_conv",
    "prepare_block",
    "prepare_round",
    "fixed_query_addresses",
    "commit_block",
    "warm_up",
    "RoundVariant",
    "ALL_VARIANTS",
    "compile_count",
]
