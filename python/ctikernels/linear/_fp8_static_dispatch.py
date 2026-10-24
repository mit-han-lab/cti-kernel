# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Static FP8 dispatch, measured per device for the Qwen3.8-27B-NVFP4 FP8 modules.

Rows: the provider with the best paired speedup against the faster of SGLang 72730d6 /
vLLM 0.31.0 at M = 1, 2, 3, 4 and 8 ... 4096; the
existing block-FP8 kernels ("block-fp8...") are kept only where they beat the best
library path (cuBLASLt "torch-scaled-mm...", FlashInfer "flashinfer-bmm...").

Device (major, minor, SMs) -> (N, K) -> increasing (last_M, provider) runs.
The last run extends upward; absent shapes use block-fp8.
"""

from .cute._block_fp8_provider import CONFIGS

TABLE: dict[tuple[int, int, int], dict[tuple[int, int], tuple[tuple[int, str], ...]]] = {
    (12, 0, 170): {  # RTX 5090
        (5120, 3072): (
            (1, "block-fp8"),
            (4, "block-fp8:cute-fp8:swap64x32s3c2-persistent-ovl-pdl-mx"),
            (512, "block-fp8"),
            (4096, "flashinfer-bmm:cublas"),
        ),
        (5120, 6144): (
            (3, "block-fp8:triton-fp8"),
            (1024, "block-fp8"),
            (4096, "flashinfer-bmm:cublas"),
        ),
        (7168, 5120): (
            (1, "block-fp8"),
            (2, "block-fp8:triton-fp8"),
            (3, "block-fp8"),
            (4, "block-fp8:cute-fp8:swap64x32s3c2-persistent-ovl-pdl-mx"),
            (4096, "block-fp8"),
        ),
        (8192, 5120): ((4096, "block-fp8"),),
        (14336, 5120): (
            (1, "block-fp8:cute-fp8:swap128x32s4-persistent-ovl-pdl-mx"),
            (2, "block-fp8:triton-fp8"),
            (4096, "block-fp8"),
        ),
        (16384, 5120): ((1, "block-fp8"), (2, "block-fp8:triton-fp8"), (4096, "block-fp8")),
    },
    (12, 0, 188): {  # RTX PRO 6000
        (5120, 3072): (
            (32, "block-fp8"),
            (64, "flashinfer-bmm:cublas"),
            (128, "block-fp8"),
            (256, "flashinfer-bmm:cublas"),
            (512, "torch-scaled-mm:rowwise"),
            (2048, "flashinfer-bmm:cublas"),
            (4096, "torch-scaled-mm:rowwise"),
        ),
        (5120, 6144): (
            (128, "block-fp8"),
            (512, "torch-scaled-mm:rowwise"),
            (2048, "flashinfer-bmm:cublas"),
            (4096, "torch-scaled-mm:rowwise"),
        ),
        (7168, 5120): ((128, "block-fp8"), (4096, "torch-scaled-mm:rowwise")),
        (8192, 5120): (
            (3, "block-fp8:triton-fp8"),
            (128, "block-fp8"),
            (4096, "torch-scaled-mm:rowwise"),
        ),
        (14336, 5120): (
            (1, "block-fp8:cute-fp8:swap128x32s3-plain-pdl"),
            (128, "block-fp8"),
            (4096, "torch-scaled-mm:rowwise"),
        ),
        (16384, 5120): (
            (1, "block-fp8"),
            (4, "block-fp8:triton-fp8"),
            (128, "block-fp8"),
            (4096, "torch-scaled-mm:rowwise"),
        ),
    },
    (12, 1, 48): {  # GB10
        (5120, 3072): ((256, "block-fp8"), (4096, "flashinfer-bmm:cublas")),
        (5120, 6144): (
            (1, "block-fp8:triton-fp8"),
            (256, "block-fp8"),
            (512, "flashinfer-bmm:cublas"),
            (2048, "torch-scaled-mm:rowwise"),
            (4096, "flashinfer-bmm:cublas"),
        ),
        (7168, 5120): ((256, "block-fp8"), (2048, "torch-scaled-mm:rowwise"), (4096, "block-fp8")),
        (8192, 5120): (
            (1, "block-fp8"),
            (2, "block-fp8:cute-fp8:swap128x32s4-persistent-ovl-pdl-mx"),
            (512, "block-fp8"),
            (2048, "torch-scaled-mm:rowwise"),
            (4096, "block-fp8"),
        ),
        (14336, 5120): (
            (2, "block-fp8"),
            (3, "block-fp8:cute-fp8:swap128x32s4-persistent-ovl-pdl-mx"),
            (512, "block-fp8"),
            (2048, "torch-scaled-mm:rowwise"),
            (4096, "block-fp8"),
        ),
        (16384, 5120): ((512, "block-fp8"), (2048, "torch-scaled-mm:rowwise"), (4096, "block-fp8")),
    },
}
DEFAULT = "block-fp8"


def parse_provider(name):
    """Return (outer, inner, config), validating without importing CUDA/DSL code."""
    if name in ("block-fp8", "torch-scaled-mm"):
        return name, None, None
    if name in ("torch-scaled-mm:rowwise", "torch-scaled-mm:slots"):
        return "torch-scaled-mm", name.split(":")[1], None
    if name in ("flashinfer-bmm:cublas", "flashinfer-bmm:cutlass", "flashinfer-bmm:auto"):
        return "flashinfer-bmm", name.split(":")[1], None
    if isinstance(name, str) and name.startswith("block-fp8:"):
        inner = name.removeprefix("block-fp8:")
        if inner in ("triton-fp8", "flashinfer-cutlass", "cuda-sm12x"):
            return "block-fp8", inner, None
        if inner.startswith("cute-fp8:"):
            config = inner.removeprefix("cute-fp8:")
            if config in CONFIGS:
                return "block-fp8", "cute-fp8", config
    raise ValueError(f"invalid static FP8 provider: {name!r}")


def validate_table():
    for shapes in TABLE.values():
        for runs in shapes.values():
            for _, name in runs:
                parse_provider(name)


validate_table()


def buckets(capability, sms, n, k):
    return TABLE.get((*capability, sms), {}).get((n, k), ((0, DEFAULT),))


def provider(runs, m):
    return next((name for last, name in runs if m <= last), runs[-1][1])
