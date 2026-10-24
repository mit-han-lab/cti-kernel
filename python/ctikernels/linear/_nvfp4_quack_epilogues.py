# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The QuACK GEMM epilogues of ``_nvfp4_quack`` (imports quack-kernels and CuTe DSL). They are
module globals so QuACK's compile cache can resolve them by import.
The fused gated epilogue applies SiLU * up to FP32 accumulators, without intermediate
BF16 gate/up rounding. SGLang rounds gate_up GEMM output to BF16 before SiLU*up+quant;
our fused path does not; gate 2 checks that it is no worse vs FP64. The unfused Triton
quantizer instead consumes an already rounded BF16 projection.

QuACK 0.6.5 BlockScaleFactorStore delegates to quantize_sf_slots: E4M3 of
amax * (S * (1/6)), factor min(rcp_approx(sf) * S, FLT_MAX), then hardware
conversion. That external recipe differs from the local FlashInfer market
recipe (nested FTZ reciprocals, amax-zero guard); the package is not modified.
"""

from __future__ import annotations

from quack.activation import swiglu
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.math import unpack
from quack.epilogue.ops import RowVecLoad, Scalar, TileStore
from quack.epilogue.quantize_out import BlockScaleFactorStore


@gemm_epilogue(
    ops={"column_scale": RowVecLoad("column_scale"), "input_scale": Scalar("input_scale")}
)
def linear(acc, column_scale, input_scale):
    return {"D": acc * column_scale * input_scale}


@gemm_epilogue(
    outputs=(
        TileStore(
            "postact", gated=True, quant=BlockScaleFactorStore("postact_sf", output="postact")
        ),
    ),
    ops={"column_scale": RowVecLoad("column_scale"), "input_scale": Scalar("input_scale")},
    extra_ops=(Scalar("sfd_norm_const"),),
    mode="acc_pair",
)
def gated(acc, column_scale, input_scale):
    gate, up = unpack(acc * column_scale * input_scale)
    return {"postact": swiglu(gate, up)}
