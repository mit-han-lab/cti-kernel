# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The gate's instruction sequence, shared by the fused CuTe prologue and the unfused Triton gate
so that both give the same bits for every element.

``x <- x * sigmoid(g)`` in FP32 (x and g already FP32): 0fbfb8aa3b is -log2(e) rounded to FP32;
e = ex2.approx(-g log2 e), sigmoid = rcp.approx(1 + e). For g < -80, 1 / (1 + e) nears the
subnormal range that rcp.approx.ftz flushes, so both x and 1 + e are scaled by s = 2^-64 (x * s
is exact whenever the result can be nonzero in BF16); otherwise s = 1.0 and every scaling step is
exact. Within a few FP32 ULP of the exact sigmoid; rounded once to BF16 by the caller.
"""


def gate_ptx(*elements: tuple[str, str, str, str]) -> str:
    """PTX computing ``x <- x * sigmoid(g)`` for each ``(x, g, s, p)``: FP32 registers ``x`` and
    ``g`` (clobbered), an FP32 scratch ``s`` and a predicate ``p``. Each step is issued for every
    element before the next (the fused prologue's pair interleaving)."""
    steps = (
        "setp.lt.f32 {p}, {g}, 0fc2a00000;",
        "selp.f32 {s}, 0f1f800000, 0f3f800000, {p};",
        "mul.rn.f32 {x}, {x}, {s};",
        "mul.rn.f32 {g}, {g}, 0fbfb8aa3b;",
        "ex2.approx.ftz.f32 {g}, {g};",
        "fma.rn.f32 {g}, {g}, {s}, {s};",
        "rcp.approx.ftz.f32 {g}, {g};",
        "mul.rn.f32 {x}, {x}, {g};",
    )
    return (
        " ".join(step.format(x=x, g=g, s=s, p=p) for step in steps for x, g, s, p in elements) + " "
    )
