# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Benchmark aliases for the package's CPU-importable mHC reference math."""

from ctikernels.mhc._reference import (
    MhcConfig,
    error_stats,
    exit_fp64,
    exit_torch_bf16,
    make_state,
    make_weights,
    output_dict,
    sinkhorn,
)

__all__ = [
    "MhcConfig",
    "error_stats",
    "exit_fp64",
    "exit_torch_bf16",
    "make_state",
    "make_weights",
    "output_dict",
    "sinkhorn",
]
