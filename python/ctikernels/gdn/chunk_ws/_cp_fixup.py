# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py,
#   flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""FP32 IEEE SIMT fixup over row-independent segment states in one launch.

No token addresses occur here: W has at most 3 * 16 * Hv * 128 * 128
elements (37,748,736 at Hv=48), independent of the context length.

M/N come from our gated-before-inverse chunk math, identity init / zero V,
without upstream T precompute. Segment zero's init is copied to workspace before
final pool writes. The planner uses our own provisional cost constants.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from ._fused import ffma

ROWS = 16
THREADS = 128
SLAB = 16
M_BUFFERS = 4
PREFETCH = M_BUFFERS - 1
SLABS = 128 // SLAB
STATE_PITCH = 132  # Preserve float4 alignment, skew successive rows by four banks.
SMEM_BYTES = (ROWS * STATE_PITCH + M_BUFFERS * SLAB * 128 + ROWS * 128) * 4


@cute.jit
def copy16(dst: cute.Pointer, src: cute.Pointer):
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16;",
        read_only_args=[cutlass.Int32(dst.toint()), src.toint()],
    )


@cute.jit
def load4(src: cute.Pointer, shared: cutlass.Constexpr):
    return cute.arch.inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];"
        if shared
        else "ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[cutlass.Int32(src.toint()) if shared else src.toint()],
    )


@cute.jit
def store4(dst: cute.Pointer, values, shared: cutlass.Constexpr):
    cute.arch.inline_ptx(
        "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};"
        if shared
        else "st.global.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[
            cutlass.Int32(dst.toint()) if shared else dst.toint(),
            values[0],
            values[1],
            values[2],
            values[3],
        ],
    )


@cute.jit
def stage_slab(matrix, additive, w, head, rows, tid, index, total):
    # A continuous M stream, index = step * 8 + slab, including across epilogues.
    # Empty tail groups keep wait_group(PREFETCH - 1) valid while draining.
    index = cutlass.Uint32(index)
    if index < cutlass.Uint32(total):
        segment, slab = 1 + index // SLABS, index % SLABS
        src = w.iterator + w.layout((2, segment, head, slab * SLAB, 0))
        dst = matrix + index % M_BUFFERS * SLAB * 128
        for part in cutlass.range_constexpr(SLAB * 128 // (THREADS * 4)):
            offset = tid * 4 + part * THREADS * 4
            copy16(dst + offset, src + offset)
        # Fetch N with slab 3, issued at the start of the step: the previous
        # epilogue has consumed the single N buffer by then. M slabs 0/1/2 of the
        # next step are already in flight during the current epilogue.
        if slab == PREFETCH:
            src_n = w.iterator + w.layout((1, segment, head, rows * ROWS, 0))
            for part in cutlass.range_constexpr(ROWS * 128 // (THREADS * 4)):
                offset = tid * 4 + part * THREADS * 4
                copy16(additive + offset, src_n + offset)
    cute.arch.cp_async_commit_group()


@cute.kernel
def cp_fixup(w: cute.Tensor, num_segments: cutlass.Int32, pdl: cutlass.Constexpr):
    head, rows, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    row = tid // 32 * 4
    col = lane * 4
    storage = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(SMEM_BYTES // 4), byte_alignment=16
    )
    state = storage.iterator
    matrix = state + ROWS * STATE_PITCH
    additive = matrix + M_BUFFERS * SLAB * 128
    if cutlass.const_expr(pdl):
        # Let C start its independent input prologue immediately. Every thread
        # waits before any W access; the hint does not publish unfinished W.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
    initial = w.iterator + w.layout((0, 1, head, rows * ROWS, 0))
    for part in cutlass.range_constexpr(ROWS * 128 // (THREADS * 4)):
        offset = tid * 4 + part * THREADS * 4
        copy16(state + offset // 128 * STATE_PITCH + offset % 128, initial + offset)
    total = (num_segments - 2) * SLABS
    # Initial state shares the first M group, so the first slab wait covers it.
    for index in cutlass.range_constexpr(PREFETCH):
        stage_slab(matrix, additive, w, head, rows, tid, cutlass.Int32(index), total)

    # Each warp owns four [V,K] rows, each lane a 4x4 output tile. CTAs only
    # consume their own state rows, so all J-2 steps stay in one launch without
    # a grid rendezvous. M is fetched once per row block, not once per row.
    for segment in cutlass.range(1, num_segments - 1, unroll=1):
        result = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        result.fill(cutlass.Float32(0))
        for slab in cutlass.range(SLABS, unroll=1):
            index = cutlass.Uint32((segment - 1) * SLABS + slab)
            cute.arch.cp_async_wait_group(PREFETCH - 1)
            # Publish the current slab and finish all previous-slab readers
            # before reusing that buffer for the slab three positions ahead.
            cute.arch.sync_threads()
            buffer = index % M_BUFFERS
            stage_slab(matrix, additive, w, head, rows, tid, index + PREFETCH, total)
            for group in cutlass.range_constexpr(SLAB // 4):
                # Keep M's 4x4 tile and stream one state row at a time. This
                # keeps the original 64-register budget and the exact per-dot
                # FMA order, without reloading M for each state row.
                tile = cute.make_rmem_tensor((4, 4), cutlass.Float32)
                for k in cutlass.range_constexpr(4):
                    m_values = load4(
                        matrix + buffer * SLAB * 128 + (group * 4 + k) * 128 + col, True
                    )
                    for c in cutlass.range_constexpr(4):
                        tile[k, c] = m_values[c]
                for r in cutlass.range_constexpr(4):
                    s_values = load4(
                        state + (row + r) * STATE_PITCH + slab * SLAB + group * 4, True
                    )
                    for k in cutlass.range_constexpr(4):
                        for c in cutlass.range_constexpr(4):
                            result[r, c] = ffma(s_values[k], tile[k, c], result[r, c])
        # All old S reads finish before publishing the next S. Global W is
        # only for the subsequent launch; the next step uses shared memory.
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(4):
            coord = (rows * ROWS + row + r, col)
            n = load4(additive + (row + r) * 128 + col, True)
            values = cute.make_rmem_tensor((4,), cutlass.Float32)
            for c in cutlass.range_constexpr(4):
                values[c] = result[r, c] + n[c]
            store4(w.iterator + w.layout((0, segment + 1, head, *coord)), values, False)
            store4(state + (row + r) * STATE_PITCH + col, values, True)
        cute.arch.sync_threads()
    cute.arch.cp_async_wait_group(0)


@cute.jit
def launch_fixup(w, num_segments: cutlass.Int32, stream, pdl: cutlass.Constexpr = False):
    cp_fixup(w, num_segments, pdl).launch(
        grid=(w.shape[2], 128 // ROWS, 1),
        block=(THREADS, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
        use_pdl=pdl,
    )


def compile_fixup(hv, options, pdl=False):
    from cutlass.runtime import make_fake_stream

    cap = cute.sym_int()
    w = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (3, cap, hv, 128, 128),
        (cap * hv * 16384, hv * 16384, 16384, 128, 1),
        assumed_align=16,
    )
    return cute.compile(
        launch_fixup,
        w,
        cutlass.Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        pdl,
        options=options,
    )
