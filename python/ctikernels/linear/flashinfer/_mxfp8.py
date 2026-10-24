# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Reference: FlashInfer's mxfp8_quantize (CUDA backend) is the called library.

"""FlashInfer MXFP8 activation quantizer: the library's compiled entry, resolved once per
process (version check, JIT module load) and called with our own outputs, so an eager call
costs one launch and two allocations of host time."""

from __future__ import annotations

import functools

import torch

from ._block_fp8 import FLASHINFER_VERSION, QUALIFIED_VERSIONS

# flashinfer.tllm_enums.SfLayout values.
_LAYOUT_128X4 = 0
_LAYOUT_LINEAR = 2


@functools.cache
def _module():
    import importlib.metadata

    try:
        version = importlib.metadata.version("flashinfer-python")
        from flashinfer.jit.fp8_quantization import gen_mxfp8_quantization_sm100_module
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError(
            f"quantize_mxfp8: needs flashinfer-python=={FLASHINFER_VERSION}"
        ) from error
    if version not in QUALIFIED_VERSIONS:
        raise RuntimeError(
            f"quantize_mxfp8: qualified on FlashInfer {QUALIFIED_VERSIONS}, found {version}"
        )
    return gen_mxfp8_quantization_sm100_module().build_and_load()


def quantize(x: torch.Tensor, *, swizzled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, K] E4M3, scale)`` (UE8M0 ``[M, K / 32]``, or flat uint8 swizzled) for contiguous ``x`` ``[M, K]`` (K % 32 == 0), on
    compute capability 10.0+ (programmatic dependent launch, as the library's default)."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    if swizzled:
        sf = torch.empty(
            -(-m // 128) * 128 * (-(-k // 128) * 4), dtype=torch.uint8, device=x.device
        )
    else:
        # Row-major UE8M0 [M, K / 32]: the library writes bytes, the dtype is the caller's.
        sf = torch.empty((m, k // 32), dtype=torch.float8_e8m0fnu, device=x.device)
    _module().mxfp8_quantize(x, q, sf, _LAYOUT_128X4 if swizzled else _LAYOUT_LINEAR, 32, True)
    return q, sf
