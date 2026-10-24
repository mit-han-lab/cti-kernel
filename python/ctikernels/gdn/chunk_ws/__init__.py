# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Lazy entry point for the SM12x warp-specialized chunk prefill provider."""

import importlib.metadata

__all__ = ["available", "load"]


def available() -> bool:
    """Check distribution metadata without importing CuTe or initializing CUDA."""
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except importlib.metadata.PackageNotFoundError:
        return False


def load(device, num_k_heads=16, num_v_heads=48):
    if not available():
        return None
    from ._launch import ChunkPrefillSm120

    return ChunkPrefillSm120(device, num_k_heads, num_v_heads)
