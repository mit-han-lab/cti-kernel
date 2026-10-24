# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Dense block-FP8 dispatch between the ``triton-fp8`` and ``flashinfer-cutlass`` providers
(and ``cuda-sm12x`` on GB10).

``TABLE`` holds, per device (compute capability and SM count) and packed weight shape
``(N, K)``, the provider measured fastest over a batch-size grid, as runs ``(last_m,
provider)`` in increasing ``last_m``: a batch of ``rows`` runs the provider of the first run
with ``rows <= last_m``, and the last run's above the grid. The runs come from ``python -m
benchmark.kernel_tuning.block_fp8_dispatch`` (CUDA-graph replay, weights rotating beyond L2)
over the Qwen3.5-35B-A3B-FP8 projection shapes and their equal TP=2 and TP=4 shards; the
grid holds every multiple of 128 up to 1024, the band within which FlashInfer's time is flat.
A shape or device the table lacks uses the size rule ``FLASHINFER_FLOPS``: FlashInfer from
``2 M N K`` FLOPs on.

``QUANTIZER`` holds, per device, the activation quantizer's tile (rows per program, warps) as
runs ``(last_units, tile)`` over ``units = rows * groups``, its program count at one row per
program, from ``python -m benchmark.kernel_tuning.block_fp8_quantizer``; a device it lacks
takes 4, 8 or 16 rows per program by batch size. A tile changes the launch, never a value.
Only this module decides; it imports no provider.
"""

from __future__ import annotations

CUTE = "cute-fp8"
TRITON = "triton-fp8"
FLASHINFER = "flashinfer-cutlass"
SM12X = "cuda-sm12x"

# Per compute capability, the GEMM size (2 M N K FLOPs) from which FlashInfer's CUTLASS GEMM
# beats the Triton one on average over the projection shapes (RTX 5090 / RTX PRO 6000: 12.0,
# GB10: 12.1), for shapes the table lacks.
# GB300 (10.3) has no size rule: shapes its table lacks stay on the Triton GEMM.
FLASHINFER_FLOPS = {(12, 0): 16e9, (12, 1): 2e9}

# (major, minor, SMs) -> {(N, K): ((last_m, provider), ...)}: the RTX 5090 (12.0, 170 SMs),
# RTX PRO 6000 (12.0, 188 SMs) and GB10 (12.1, 48 SMs; three-way with cuda-sm12x, emitted
# with --margin 0.02).
_T, _F, _C, _Q = TRITON, FLASHINFER, SM12X, CUTE
TABLE: dict[tuple[int, int, int], dict[tuple[int, int], tuple[tuple[int, str], ...]]] = {
    (12, 0, 170): {  # RTX 5090
        (256, 2048): ((4096, _T),),
        (512, 512): ((2048, _T), (4096, _F)),
        (512, 2048): ((2048, _T), (4096, _F)),
        (512, 4096): ((2048, _T), (4096, _F)),
        (1024, 512): ((1024, _T), (2048, _F), (3072, _T), (4096, _F)),
        (1024, 1024): ((1024, _T), (2048, _F), (3072, _T), (4096, _F)),
        (1024, 2048): ((256, _T), (4096, _Q)),
        (1024, 4096): ((1024, _T), (2048, _F), (3072, _T), (4096, _F)),
        (2048, 128): ((4096, _T),),
        (2048, 256): ((640, _T), (1024, _F), (1536, _T), (2048, _F), (4096, _T)),
        (2048, 512): ((256, _T), (4096, _Q)),
        (2048, 1024): ((640, _T), (1024, _F), (1536, _T), (2048, _F), (4096, _T)),
        (2048, 2048): ((640, _T), (1024, _F), (1536, _T), (2048, _F), (4096, _T)),
        (2048, 4096): ((128, _T), (4096, _Q)),
        (2304, 2048): ((512, _T), (1024, _F), (1536, _T), (4096, _F)),
        (3072, 2048): ((512, _T), (896, _F), (1024, _T), (1536, _F), (2048, _T), (4096, _F)),
        (4608, 2048): ((448, _T), (512, _F), (896, _T), (1536, _F), (2048, _T), (4096, _F)),
        (5120, 2048): ((8, _T), (4096, _Q)),
        (5120, 3072): ((16, _T), (4096, _Q)),
        (5120, 6144): ((16, _T), (32, _Q), (64, _T), (4096, _Q)),
        (5120, 8704): ((16, _T), (32, _Q), (64, _T), (4096, _Q)),
        (5120, 17408): ((64, _T), (4096, _Q)),
        (6144, 2048): ((320, _T), (384, _F), (640, _T), (896, _F), (1536, _T), (4096, _F)),
        (7168, 5120): ((8, _T), (4096, _Q)),
        (8192, 5120): ((8, _T), (4096, _Q)),
        (9216, 512): ((224, _T), (256, _F), (448, _T), (512, _F), (640, _T), (4096, _F)),
        (9216, 1024): ((192, _T), (256, _F), (448, _T), (512, _F), (640, _T), (4096, _F)),
        (9216, 2048): ((16, _T), (4096, _Q)),
        (12288, 512): ((320, _T), (384, _F), (512, _T), (4096, _F)),
        (12288, 1024): ((320, _T), (384, _F), (512, _T), (4096, _F)),
        (12288, 2048): ((16, _T), (4096, _Q)),
        (14336, 5120): ((8, _T), (4096, _Q)),
        (16384, 5120): ((16, _T), (4096, _Q)),
        (17408, 5120): ((16, _T), (4096, _Q)),
        (34816, 5120): ((4, _T), (4096, _Q)),
    },
    (12, 0, 188): {  # RTX PRO 6000
        (256, 2048): ((4096, _T),),
        (512, 512): ((2048, _T), (4096, _F)),
        (512, 2048): ((2048, _T), (4096, _F)),
        (512, 4096): ((2048, _T), (4096, _F)),
        (1024, 512): ((1024, _T), (2048, _F), (4096, _T)),
        (1024, 1024): ((1024, _T), (4096, _F)),
        (1024, 2048): ((256, _T), (1024, _Q), (4096, _F)),
        (1024, 4096): ((1024, _T), (4096, _F)),
        (1536, 4096): ((128, _T), (1024, _F)),
        (2048, 128): ((4096, _T),),
        (2048, 256): ((640, _T), (1024, _F), (2048, _T), (4096, _F)),
        (2048, 512): ((256, _T), (2048, _Q), (4096, _F)),
        (2048, 1024): ((640, _T), (4096, _F)),
        (2048, 2048): ((640, _T), (4096, _F)),
        (2048, 4096): ((128, _T), (512, _Q), (4096, _F)),
        (2304, 2048): ((640, _T), (4096, _F)),
        (3072, 2048): ((448, _T), (4096, _F)),
        (4096, 2048): ((512, _T), (2048, _F)),
        (4096, 8192): ((256, _T), (2048, _F)),
        (4608, 2048): ((320, _T), (4096, _F)),
        (5120, 2048): ((8, _T), (256, _Q), (4096, _F)),
        (5120, 3072): ((16, _T), (256, _Q), (4096, _F)),
        (5120, 6144): ((16, _T), (256, _Q), (4096, _F)),
        (5120, 8704): ((16, _T), (64, _Q), (128, _T), (256, _Q), (4096, _F)),
        (5120, 17408): ((64, _T), (256, _Q), (4096, _F)),
        (6144, 2048): ((192, _T), (384, _F), (448, _T), (4096, _F)),
        (7168, 5120): ((16, _T), (128, _Q), (4096, _F)),
        (8192, 5120): ((16, _T), (128, _Q), (4096, _F)),
        (9216, 512): ((192, _T), (256, _F), (320, _T), (4096, _F)),
        (9216, 1024): ((160, _T), (4096, _F)),
        (9216, 2048): ((16, _T), (64, _Q), (128, _T), (4096, _F)),
        (12288, 512): ((256, _T), (4096, _F)),
        (12288, 1024): ((192, _T), (4096, _F)),
        (12288, 2048): ((32, _Q), (128, _T), (256, _Q), (4096, _F)),
        (14336, 5120): ((8, _T), (64, _Q), (128, _T), (4096, _F)),
        (16384, 5120): ((8, _T), (64, _Q), (128, _T), (4096, _F)),
        (17408, 5120): ((16, _T), (64, _Q), (128, _T), (4096, _F)),
        (32768, 1024): ((4, _T), (512, _F)),
        (34816, 5120): ((16, _T), (256, _Q), (4096, _F)),
    },
    (12, 1, 48): {  # GB10
        (256, 2048): ((768, _T), (1536, _F), (4096, _C)),
        (512, 512): ((768, _T), (4096, _F)),
        (512, 2048): ((512, _T), (2048, _F), (4096, _C)),
        (512, 4096): ((384, _T), (1024, _F), (4096, _C)),
        (1024, 512): ((384, _T), (4096, _F)),
        (1024, 1024): ((320, _T), (4096, _F)),
        (1024, 2048): ((16, _T), (256, _Q), (512, _F), (1024, _Q), (2048, _F), (4096, _C)),
        (1024, 4096): ((192, _T), (1024, _F), (4096, _C)),
        (1536, 4096): ((4, _T), (512, _Q), (4096, _F)),
        (2048, 128): ((320, _T), (384, _F), (512, _T), (1536, _F), (4096, _T)),
        (2048, 256): ((256, _T), (2048, _F), (4096, _T)),
        (2048, 512): ((32, _T), (4096, _Q)),
        (2048, 1024): ((192, _T), (4096, _F)),
        (2048, 2048): ((96, _T), (128, _F), (192, _T), (4096, _F)),
        (2048, 4096): ((512, _Q), (2048, _F), (4096, _C)),
        (2304, 2048): ((64, _T), (256, _F), (320, _C), (4096, _F)),
        (3072, 2048): ((64, _T), (4096, _F)),
        (4096, 2048): ((4, _T), (256, _Q), (512, _F), (2048, _Q), (4096, _F)),
        (4096, 4096): ((4, _T), (256, _Q), (4096, _F)),
        (4096, 8192): ((4, _T), (512, _Q), (2048, _F), (4096, _Q)),
        (4608, 2048): ((64, _T), (128, _C), (4096, _F)),
        (5120, 2048): ((128, _Q), (512, _F), (4096, _Q)),
        (5120, 3072): ((2048, _Q), (4096, _F)),
        (5120, 6144): ((512, _Q), (2048, _F), (4096, _Q)),
        (5120, 8704): ((512, _Q), (1024, _F), (4096, _Q)),
        (5120, 17408): ((64, _Q), (256, _F), (4096, _Q)),
        (6144, 2048): ((64, _T), (4096, _F)),
        (7168, 5120): ((16, _T), (256, _Q), (2048, _F), (4096, _Q)),
        (8192, 1024): ((16, _T), (2048, _Q), (4096, _F)),
        (8192, 5120): ((1, _Q), (2, _T), (4, _Q), (16, _T), (512, _Q), (2048, _F), (4096, _Q)),
        (9216, 512): ((64, _T), (2048, _F), (3072, _T), (4096, _F)),
        (9216, 1024): ((64, _T), (4096, _F)),
        (9216, 2048): ((16, _T), (256, _Q), (512, _F), (4096, _Q)),
        (12288, 512): ((64, _T), (2048, _F), (3072, _T), (4096, _F)),
        (12288, 1024): ((64, _T), (4096, _F)),
        (12288, 2048): ((16, _T), (128, _Q), (512, _F), (2048, _Q), (4096, _F)),
        (14336, 5120): ((512, _Q), (2048, _F), (4096, _Q)),
        (16384, 5120): ((512, _Q), (2048, _F), (4096, _Q)),
        (17408, 5120): (
            (1, _T),
            (3, _Q),
            (4, _T),
            (128, _Q),
            (256, _F),
            (512, _Q),
            (2048, _F),
            (4096, _Q),
        ),
        (32768, 1024): ((64, _T), (2048, _Q), (4096, _F)),
        (34816, 5120): ((8, _Q), (16, _T), (512, _Q), (2048, _F), (4096, _Q)),
    },
    # GB300 (10.3, 152 SMs), DeepSeek V4 Flash dense projections (2026-10-19 grid, weights
    # rotating past L2). FlashInfer only from M = 33: its sm100 GEMM is exact there and, with
    # FlashInferCutlassGemm.rows' padding, below; it loses to Triton at small M.
    (10, 3, 152): {
        (1536, 4096): ((384, _T), (4096, _F)),
        (4096, 2048): ((33, _T), (4096, _F)),
        (4096, 4096): ((33, _T), (4096, _F)),
        (4096, 8192): ((32, _T), (4096, _F)),
        (8192, 1024): ((33, _T), (64, _F), (128, _T), (4096, _F)),
        (32768, 1024): ((32, _T), (4096, _F)),
    },
}

Buckets = tuple[tuple[int, str], ...]

# (major, minor, SMs) -> ((last_units, (block_m, num_warps)), ...): the RTX 5090, RTX PRO 6000
# and GB10, from sweeps with the activation in L2.
QUANTIZER: dict[tuple[int, int, int], tuple[tuple[int, tuple[int, int]], ...]] = {
    (12, 0, 170): (
        (512, (1, 1)),
        (1024, (8, 8)),
        (2048, (16, 8)),
        (16384, (8, 4)),
        (131072, (32, 16)),
    ),
    (12, 0, 188): ((512, (1, 1)), (8192, (8, 4)), (16384, (16, 8)), (131072, (16, 4))),
    (12, 1, 48): (
        (256, (1, 1)),
        (1024, (8, 4)),
        (4096, (32, 8)),
        (65536, (64, 16)),
        (131072, (1, 1)),
    ),
}


def quantizer_tile(capability: tuple[int, int], sms: int, rows: int, groups: int):
    """``(block_m, num_warps)`` of the quantizer on ``rows`` x ``groups`` 128-column groups."""
    runs = QUANTIZER.get((*capability, sms))
    if runs is None:  # by rows alone, with 4 warps
        return (4 if rows <= 64 else 8 if rows <= 1024 else 16), 4
    units = rows * groups
    for last, tile in runs:
        if units <= last:
            return tile
    return runs[-1][1]


def buckets(capability: tuple[int, int], sms: int, n: int, k: int) -> Buckets | None:
    """The runs of a packed ``[N, K]`` weight on this device: the measured ones, else the
    size rule's, else None (the device has no measured FlashInfer advantage)."""
    measured = TABLE.get((*capability, sms), {}).get((n, k))
    if measured is not None:
        return measured
    flops = FLASHINFER_FLOPS.get(tuple(capability))
    if flops is None:
        return None
    first = max(1, -(-int(flops) // (2 * n * k)))
    return ((first - 1, TRITON), (first, FLASHINFER))


def restrict(table: Buckets, available: set[str]) -> Buckets:
    """``table`` with the runs of unavailable providers given to ``triton-fp8`` (merged)."""
    runs: list[tuple[int, str]] = []
    for last, measured in table:
        name = measured if measured in available else TRITON
        if runs and runs[-1][1] == name:
            runs[-1] = (last, name)
        else:
            runs.append((last, name))
    return tuple(runs)


def provider(table: Buckets, rows: int) -> str:
    """The provider of the run holding ``rows``."""
    for last, name in table:
        if rows <= last:
            return name
    return table[-1][1]


# Parallel to TABLE: config ceilings for runs selecting cute-fp8. No provider import.
CUTE_CONFIG: dict[tuple[int, int, int], dict[tuple[int, int], tuple[tuple[int, str], ...]]] = {
    (12, 0, 170): {  # RTX 5090
        (1024, 2048): (
            (512, "swap128x32s4-persistent-ovl-pdl-mx"),
            (1024, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx"),
        ),
        (2048, 512): (
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (2048, 4096): (
            (256, "swap128x32s4-mx"),
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s3-mx"),
            (4096, "normal128x128s2-persistent-mx"),
        ),
        (5120, 2048): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "swap128x32s4w8-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (512, "normal128x128s2-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 3072): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 6144): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x64s4-mx"),
            (256, "swap128x64s3-persistent-ovl-pdl-mx"),
            (512, "normal128x128s2-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 8704): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x64s4-mx"),
            (256, "swap128x64s3-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 17408): (
            (128, "swap128x64s4-mx"),
            (256, "swap128x64s3-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (7168, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "swap128x32s4-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (256, "normal128x128s3-mx-pdl"),
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (8192, 5120): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (512, "normal128x128s3-mx-pdl"),
            (1024, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (9216, 2048): (
            (64, "swap128x32s4-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
        ),
        (12288, 2048): (
            (32, "swap128x32s4-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
        ),
        (14336, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
            (128, "normal128x128s3-mx-pdl"),
            (256, "swap128x64s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (16384, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (17408, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (34816, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
            (128, "normal128x128s3-mx-pdl"),
            (256, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
    },
    (12, 0, 188): {  # RTX PRO 6000
        (1024, 2048): ((512, "swap128x32s4-mx"), (1024, "normal128x64s3-mx")),
        (2048, 512): (
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2-persistent-ovl-pdl-mx"),
            (2048, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (2048, 4096): ((256, "swap128x32s4-mx"), (512, "normal64x128s3-mx")),
        (5120, 2048): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s3-plain-pdl"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (5120, 3072): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s3-plain-pdl"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (5120, 6144): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s3-plain-pdl"),
            (256, "swap128x64s3-mx"),
        ),
        (5120, 8704): ((64, "swap64x32s3c2-persistent-ovl-pdl-mx"), (256, "swap128x64s3w8-mx")),
        (5120, 17408): ((128, "swap128x32s3-plain-pdl"), (256, "swap128x64s3-mx")),
        (7168, 5120): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (8192, 5120): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (9216, 2048): ((64, "swap128x32s4-persistent-ovl-pdl-mx"),),
        (12288, 2048): (
            (16, "swap128x32s3-plain-pdl"),
            (32, "swap128x32s4-persistent-ovl-pdl-plain"),
            (256, "normal64x128s3-persistent-ovl-pdl-plain"),
        ),
        (14336, 5120): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (16384, 5120): ((64, "swap64x32s3c2-persistent-ovl-pdl-mx"),),
        (17408, 5120): ((64, "swap64x32s3c2-persistent-ovl-pdl-mx"),),
        (34816, 5120): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (256, "normal128x128s3-mx-pdl"),
        ),
    },
    (12, 1, 48): {  # GB10
        (1024, 2048): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (1024, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (1536, 4096): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (512, "normal128x128s3-mx-pdl"),
        ),
        (2048, 512): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (2048, 4096): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (256, "normal128x128s3-mx-pdl"),
            (512, "swap128x64s3-persistent-ovl-pdl-mx"),
        ),
        (4096, 2048): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (4096, 4096): (
            (32, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
        ),
        (4096, 8192): (
            (192, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "swap128x64s4-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 2048): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (1024, "normal128x128s2-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (5120, 3072): (
            (1, "swap128x32s4-persistent-ovl-pdl-mx"),
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (512, "normal128x128s3-mx-pdl"),
            (1024, "normal128x128s2w8g8-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (5120, 6144): (
            (1, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (64, "swap128x32s4-persistent-ovl-pdl-mx"),
            (128, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 8704): (
            (64, "swap128x32s4-persistent-ovl-pdl-mx"),
            (128, "swap128x64s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (5120, 17408): (
            (64, "swap128x32s4-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (7168, 5120): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (8192, 1024): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (256, "normal64x128s3-persistent-ovl-pdl-mx"),
            (512, "swap128x64s3-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (8192, 5120): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (9216, 2048): (
            (128, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (256, "swap128x64s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (12288, 2048): (
            (64, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (14336, 5120): (
            (2, "swap128x32s4-persistent-ovl-pdl-mx"),
            (3, "swap64x32s3c2-persistent-ovl-pdl-mx"),
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (16384, 5120): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "swap128x64s3-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (17408, 5120): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
        (32768, 1024): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (512, "normal64x128s3-persistent-ovl-pdl-mx"),
            (2048, "normal128x128s2-persistent-ovl-pdl-mx"),
        ),
        (34816, 5120): (
            (128, "swap128x32s4-persistent-ovl-pdl-mx"),
            (256, "swap128x64s4-mx"),
            (4096, "normal128x128s3-mx-pdl"),
        ),
    },
}


def cute_config(capability: tuple[int, int], sms: int, n: int, k: int, rows: int) -> str:
    """Measured config for this M range, else the gated-fp8 heuristic without tail splitting."""
    runs = CUTE_CONFIG.get((*capability, sms), {}).get((n, k))
    if runs:
        return provider(runs, rows)
    if rows <= 128:
        return "swap128x32s4-mx" if n <= 1024 else "swap128x64s3-mx"
    if n >= 8192:
        return "normal128x128s2-mx"
    return "normal128x64s3-mx"
