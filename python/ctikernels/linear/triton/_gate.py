# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Linear's unfused gate, ``x * sigmoid(gate)``, in one launch.

FP32 math rounded once to ``x.dtype``, with the fused CuTe prologue's instruction sequence
(``linear._gate_ptx``), so the fused and unfused paths give the same bits for every element.
That sequence's sigmoid is approximate (ex2.approx, rcp.approx): within a few FP32 ULP of
PyTorch's, so the result is within one ``x.dtype`` ULP of ``torch.sigmoid(gate.float()).mul_(x)``.
Any row and column strides; contiguous output. Integer arguments and pointer alignment are not
specialized, so one compiled kernel serves every shape and layout of a dtype pair (warm it
eagerly before graph capture).
"""

import torch
import triton
import triton.language as tl

from .._gate_ptx import gate_ptx

_GATE = tl.constexpr(
    "{ .reg .f32 x, g, s; .reg .pred p; mov.b32 x, $1; mov.b32 g, $2; "
    + gate_ptx(("x", "g", "s", "p"))
    + "mov.b32 $0, x; }"
)

BLOCK = 1024


@triton.jit(
    do_not_specialize=["M", "K", "XS0", "XS1", "GS0", "GS1"],
    do_not_specialize_on_alignment=["X", "G", "A"],
)
def _gate_kernel(X, G, A, M: tl.int64, K: tl.int64, XS0, XS1, GS0, GS1, BLOCK: tl.constexpr):
    blocks = (K + BLOCK - 1) // BLOCK
    pid = tl.program_id(0).to(tl.int64)
    row = pid // blocks
    col = (pid % blocks) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = col < K
    x = tl.load(X + row * XS0 + col * XS1, valid, other=0).to(tl.float32)
    g = tl.load(G + row * GS0 + col * GS1, valid, other=0).to(tl.float32)
    a = tl.inline_asm_elementwise(_GATE, "=f,f,f", [x, g], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(A + row * K + col, a.to(A.dtype.element_ty), valid)


def gate_multiply(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(gate)`` for CUDA [M, K] ``x`` and ``gate`` of the same shape, FP32 math,
    returned contiguous in ``x.dtype``."""
    m, k = x.shape
    a = torch.empty(m, k, dtype=x.dtype, device=x.device)
    if m and k:
        with torch.cuda.device(x.device):
            _gate_kernel[(m * triton.cdiv(k, BLOCK),)](
                x, gate, a, m, k, *x.stride(), *gate.stride(), BLOCK=BLOCK
            )
    return a
