# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""BF16 variant registration without importing the CuTe DSL or inspecting CUDA.

CuTe supports M=1..64 in eight-row buckets.
TABLES preserves inclusive measured M ranges and their holes. supports() is a
table admission query, not a dependency/layout check. This is not a block-FP8
provider: inputs and frozen weights are BF16, with no quantization scales.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from .._dispatch import TUNED, Config
from ..inference import Linear

DEVICE_KEYS = {"rtx5090": (12, 0, 170), "pro6000": (12, 0, 188), "gb10": (12, 1, 48)}
TABLES = {
    key: {shape: dict(runs) for shape, runs in TUNED[kind].items()}
    for kind, key in DEVICE_KEYS.items()
}


def supports(device_key, n, k, m):
    """Whether the shipped table admits this device/shape/M (no interpolation)."""
    return 1 <= m <= 64 and any(
        lo <= m <= hi for lo, hi in TABLES.get(device_key, {}).get((n, k), {})
    )


@lru_cache(maxsize=None)
def _operation(config):
    # Explicit policy avoids process-wide environment changes during integration.
    return Linear(config=config, provider="auto" if config is None else "cute")


def smallm_bf16_linear(x, weight, *, out=None, config=None):
    """Frozen BF16 [M,K] @ [N,K].T; optional Config explicitly selects CuTe.

    Without config, use measured dispatch with cuBLAS fallback. Warm every
    selected config eagerly before graph capture. Output/scratch ownership and
    stream ordering follow Linear. No autograd or block-FP8 operands.
    """
    if x.requires_grad or weight.requires_grad:
        raise ValueError("smallm_bf16_linear requires inputs with requires_grad=False")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("smallm_bf16_linear requires BF16 inputs and weights")
    if config is not None and not isinstance(config, Config):
        raise TypeError("config must be a Config")
    return _operation(config)(x, weight, out)


@dataclass(frozen=True)
class Variant:
    call: Callable = smallm_bf16_linear
    dtype: str = "bf16"
    max_m: int = 64
    supports: Callable = supports

    def __call__(self, x, weight, *, out=None, config=None):
        return self.call(x, weight, out=out, config=config)


VARIANTS = {"smallm-bf16": Variant()}

__all__ = ["Config", "DEVICE_KEYS", "TABLES", "VARIANTS", "smallm_bf16_linear", "supports"]
