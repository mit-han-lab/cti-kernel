# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Measured full-fusion FC1 overrides, keyed by (device class, GEMM N=2I, K).

Keep the original M=64 cells (27B TP=1); remove a cell here to disable it.
These cells were measured at revision 950aa9e. SMALL_M overrides the
small-M policy per (device class, I, H). Remaining cells use package BlockFP8Linear
dispatch plus the cute-fp8 activation variant.
"""

# Small-M policy per (device class, I, H): inclusive M ceilings covering 1..last;
# above last, TABLE / the package GEMM plus activation. Fused specs pick the
# engine: "fma" (CUDA-core GEMV) or "mma" (m16n8k32 E4M3 tiles). Generated from
# paired cold-weight CUDA-graph sweeps of every variant against the unfused path
# (median of ABBABAAB octets, 2026-10-16). RTX 5090 rows use the MLP boundary
# (FC1 + down): FC1-only loops overstate the fused kernel, whose PDL prologue
# overlaps the previous fused call but not the Triton down GEMM. GB10 rows are
# FC1-boundary sweeps until the MLP sweep lands.
# The RTX PRO 6000 rows keep the conservative default until measured.
SMALL_M = {
    ("rtx5090", 256, 2048): ((2, "cute-fp8-fused", "fma:r=2,w=4"), (4, "cute-act-quant", None)),
    ("rtx5090", 512, 2048): ((1, "cute-fp8-fused", "fma:r=2,w=4"), (4, "cute-act-quant", None)),
    ("rtx5090", 2048, 4096): ((1, "cute-fp8-fused", "fma:r=2,w=4"), (4, "cute-act-quant", None)),
    ("rtx5090", 8704, 5120): ((4, "cute-act-quant", None),),
    ("rtx5090", 17408, 5120): ((1, "cute-fp8-fused", "fma:r=4,w=4"), (4, "cute-act-quant", None)),
    ("gb10", 256, 2048): (
        (1, "cute-fp8-fused", "fma:r=4,w=8"),
        (2, "cute-fp8-fused", "fma:r=2,w=4"),
        (4, "cute-fp8-fused", "mma:wn=1,wk=8"),
    ),
    ("gb10", 512, 2048): (
        (1, "cute-fp8-fused", "fma:r=2,w=4"),
        (2, "cute-fp8-fused", "fma:r=8,w=8"),
        (4, "cute-fp8-fused", "mma:wn=1,wk=8"),
    ),
    ("gb10", 2048, 4096): (
        (1, "cute-fp8-fused", "fma:r=8,w=8"),
        (2, "cute-fp8-fused", "fma:r=4,w=4"),
        (4, "cute-fp8-fused", "mma:wn=1,wk=8"),
    ),
    ("gb10", 8704, 5120): (
        (1, "cute-fp8-fused", "fma:r=8,w=8"),
        (2, "cute-fp8-fused", "fma:r=2,w=4"),
        (3, "cute-fp8-fused", "fma:r=4,w=4"),
        (4, "cute-fp8-fused", "mma:wn=1,wk=8"),
    ),
    ("gb10", 17408, 5120): (
        (1, "cute-fp8-fused", "fma:r=8,w=8"),
        (4, "cute-fp8-fused", "fma:r=4,w=4"),
    ),
    **{
        ("rtxpro6000", i, h): ((1, "cute-fp8-fused", "fma:r=4,w=4"), (4, "cute-act-quant", None))
        for i, h in ((512, 2048), (256, 2048), (2048, 4096), (17408, 5120), (8704, 5120))
    },
}

TABLE = {
    ("rtx5090", 34816, 5120): (
        (63, "cute-act-quant"),
        (64, "flashinfer-cute-dsl"),
        (4096, "cute-act-quant"),
    ),
    ("rtxpro6000", 34816, 5120): (
        (63, "cute-act-quant"),
        (64, "flashinfer-cute-dsl"),
        (4096, "cute-act-quant"),
    ),
}
