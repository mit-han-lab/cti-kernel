# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""torch's FP32 SiLU and sigmoid in Triton, bit for bit (SGLang's form): ``x / (1 + expf(-x))``
and ``1 / (1 + expf(-x))`` with libdevice's accurate expf and an IEEE divide that keeps
subnormal results (inline ``div.rn.f32``: Triton links libdevice with FTZ, so its divide flushes
the subnormal SiLU of gates near -87..-104 that torch keeps). A subnormal expf result only
occurs where ``1 + expf`` rounds to 1, so expf's FTZ does not change either function."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def silu(x):
    return _div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def sigmoid(x):
    return _div_rn(tl.full(x.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-x))
