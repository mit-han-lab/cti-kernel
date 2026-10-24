# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# kernel_moe_mxfp8_mxfp4_fc1_act_q1.py, moe_mxfp8_mxfp4_fc1_act_q1.py).
# Local modifications: CTI runtime/cache, native gate/up, K32 Q1, optional traces.

"""FC1 specialization and launch arguments; no DSL import at module import."""

import torch

from ._w4a8_debug import DebugLaunch, validate_debug


def compile_gemm(
    runtime,
    tile,
    grid,
    limit,
    args,
    *,
    pdl=True,
    debug=False,
    debug_mode=None,
    combine="atomic",
    up_first=False,
    lora_rows=False,
):
    """Static TMA descriptors, cached by device and complete tensor metadata."""
    if combine not in ("atomic", "rows"):
        raise ValueError("combine must be atomic or rows")
    debug_stamps = len(args) > 12 and args[12] is not None
    validate_debug(debug_mode, debug_stamps)
    if debug_stamps:
        stamps = args[12]
        if stamps.dtype != torch.int64 or tuple(stamps.shape) != (grid, 4):
            raise ValueError("W4A8 debug stamps must be int64 [grid, 4]")
        if stamps.device != args[0].device:
            raise ValueError("W4A8 debug stamps must share the operand device")
    k, weight = args[0].shape[1], args[2]
    # Canonical [E, N, K/2] weights, read through native 3-D TMA boxes.
    if weight.ndim != 3:
        raise ValueError("W4A8 weights must be canonical [E, N, K/2]")
    e, n, half_k = weight.shape
    if half_k * 2 != k:
        raise ValueError("W4A8 weight K must match activations")
    if k % 128 or n % 128:
        raise ValueError("W4A8 K and N must be multiples of 128")
    lora_rank = args[13].shape[2] if len(args) > 13 and args[13] is not None else 0
    tile.check_smem(limit)
    if debug_stamps and tile.smem_bytes + 128 > limit:
        raise ValueError("W4A8 debug stamps exceed the shared memory limit")
    for index, tensor in enumerate(args):
        if tensor is not None:
            if not tensor.is_contiguous():
                raise ValueError("W4A8 GEMM requires contiguous tensors")
            # Check actual pointers even on a metadata-only compilation cache hit.
            alignment = (
                16 if index < 4 or (index == 7 and combine == "rows") else tensor.element_size()
            )
            if tensor.data_ptr() % alignment:
                raise ValueError("W4A8 TMA operands must be 16-byte aligned")
    signature = tuple(
        None if tensor is None else (tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
        for tensor in args
    )

    def build():
        import cutlass
        import cutlass.cute as cute
        from cutlass.runtime import make_fake_stream

        from ._w4a8_gemm import GroupedGemm
        from ._w4a8_watchdog import allocate_records

        types = {
            torch.uint8: cutlass.Uint8,
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
            torch.float32: cutlass.Float32,
            torch.bfloat16: cutlass.BFloat16,
        }
        fake = []
        for index, tensor in enumerate(args):
            if tensor is None:
                fake.append(None)
            else:
                # TMA operands and private row output promise 16-byte alignment.
                # Caller-owned atomic output promises only one element.
                alignment = (
                    16 if index < 4 or (index == 7 and combine == "rows") else tensor.element_size()
                )
                fake.append(
                    cute.runtime.make_fake_tensor(
                        types[tensor.dtype],
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        assumed_align=alignment,
                    )
                )
        sink = None
        fake_sink = None
        if debug_mode in ("no_epilogue", "fragments_only"):
            sink = torch.empty((grid, 256), dtype=torch.int32, device=args[0].device)
            fake_sink = cute.runtime.make_fake_tensor(
                cutlass.Int32,
                (grid, 256),
                (256, 1),
                assumed_align=4,
            )
        records = allocate_records(grid) if tile.watchdog else None
        compiled = cute.compile(
            GroupedGemm(
                tile,
                grid,
                limit,
                pdl=pdl,
                debug=debug,
                debug_mode=debug_mode,
                debug_stamps=debug_stamps,
                combine=combine,
                lora_rank=lora_rank,
                up_first=up_first,
                lora_rows=lora_rows,
            ),
            *fake,
            fake_sink,
            cutlass.Int64(0) if tile.watchdog else None,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options=runtime.options,
        )
        return DebugLaunch(compiled, records, sink)

    return runtime.compile_cached(
        (
            "w4a8_gemm",
            tile,
            grid,
            limit,
            pdl,
            debug,
            debug_mode,
            debug_stamps,
            combine,
            up_first,
            lora_rows,
            signature,
        ),
        build,
    )


def arguments(ws, weight, scales, trace=None, *, stamps=None, lora=None):
    """``lora`` = (z13 [P,2,r] FP32, gate B, up B [E,I,r], scale [1], A2 [E,r,I],
    z2 parts [P,I/BN,r] FP32, z2 [P,1,r] FP32, counters [P*r] int32, zero) or None."""
    trace = trace or {}
    z, gate_b, up_b, scale, a2, parts, z2, count = (None,) * 8 if lora is None else lora
    return (
        ws.q0,
        ws.q0_scales,
        weight,
        scales,
        ws.expert_offsets,
        ws.scale_offsets,
        ws.row_weight,
        ws.q1,
        ws.q1_scales,
        trace.get("gate"),
        trace.get("up"),
        trace.get("h"),
        stamps,
        z,
        gate_b,
        up_b,
        scale,
        None if lora is None else ws.row_pair,
        a2,
        parts,
        z2,
        count,
    )


def compile_fc1(
    runtime,
    tile,
    grid,
    limit,
    ws,
    weight,
    scales,
    *,
    pdl=True,
    trace=None,
    debug_mode=None,
    stamps=None,
    lora=None,
    up_first=False,
    lora_rows=False,
):
    if tile.stage != 1:
        raise ValueError("FC1 requires a stage-1 tile")
    return compile_gemm(
        runtime,
        tile,
        grid,
        limit,
        arguments(ws, weight, scales, trace, stamps=stamps, lora=lora),
        pdl=pdl,
        debug=trace is not None,
        debug_mode=debug_mode,
        up_first=up_first,
        lora_rows=lora_rows,
    )
