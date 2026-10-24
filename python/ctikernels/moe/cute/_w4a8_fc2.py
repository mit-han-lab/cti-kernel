# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# kernel_moe_mxfp8_mxfp4_fc2_finalize.py, moe_mxfp8_mxfp4_fc2_finalize.py).
# Local modifications: K32 input, no route weight argument, masked BF16 atomics,
# CTI runtime, and optional FP32 diagnostics absent from production launches.

"""FC2 specialization with BF16 row stores or atomic scatter-add."""

from ._w4a8_fc1 import compile_gemm


def arguments(ws, weight, scales, out, trace=None, *, stamps=None, combine="atomic", lora=None):
    """``lora`` = (z2 [P,1,r] FP32, down B [E,H,r], scale [1]) or None."""
    z2, down_b, scale = (None,) * 3 if lora is None else lora
    return (
        ws.q1,
        ws.q1_scales,
        weight,
        scales,
        ws.expert_offsets,
        ws.scale_offsets,
        ws.row_token,
        ws.y if combine == "rows" else out,
        None,
        None if trace is None else trace["fc2"],
        None,
        None,
        stamps,
        z2,
        down_b,
        None,
        scale,
        None if lora is None else ws.row_pair,
        None,
        None,
        None,
        None,
    )


def compile_fc2(
    runtime,
    tile,
    grid,
    limit,
    ws,
    weight,
    scales,
    out,
    *,
    pdl=True,
    trace=None,
    debug_mode=None,
    stamps=None,
    combine="atomic",
    lora=None,
    lora_rows=False,
):
    if tile.stage != 2:
        raise ValueError("FC2 requires a stage-2 tile")
    return compile_gemm(
        runtime,
        tile,
        grid,
        limit,
        arguments(ws, weight, scales, out, trace, stamps=stamps, combine=combine, lora=lora),
        pdl=pdl,
        debug=trace is not None,
        debug_mode=debug_mode,
        combine=combine,
        lora_rows=lora_rows,
    )
