# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py,
#   flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""Fused producers and CP using our gated chunk math, without upstream T precompute.

M/N reuse identity init / zero V; the gate precedes the inverse. Segment zero's
initial state is copied to W so the final pool write never aliases a launch read.
Fixup uses FP32 IEEE; the planner uses our own provisional constants.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T

from . import _watchdog
from ._gates import GateProcessor
from ._kernel import ChunkKernelSm120, NamedBarrier, WorkDesc
from ._o_store import OutputStore


@cute.jit
def ex2(x, ftz: cutlass.Constexpr = False):
    # PTX ex2 without .ftz preserves subnormal results, unlike fastmath exp2.
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value()],
            "ex2.approx.ftz.f32 $0, $1;" if ftz else "ex2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def rcp(x):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value()],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def sigmoid(x):
    t = ex2(-abs(x) * cutlass.Float32(1.4426950408889634))
    numerator = cutlass.Float32(1.0) if x >= 0 else t
    return numerator * rcp(cutlass.Float32(1.0) + t)


@cute.jit
def silu(x):
    # Non-FTZ exp2 keeps subnormals; rcp(+inf) is zero for large negative x.
    return x * rcp(cutlass.Float32(1) + ex2(-x * cutlass.Float32(1.4426950408889634)))


@cute.jit
def silu_four(values):
    exponent = cute.make_rmem_tensor((4,), cutlass.Float32)
    for r in cutlass.range_constexpr(4):
        exponent[r] = -values[r] * cutlass.Float32(1.4426950408889634)
    for r in cutlass.range_constexpr(4):
        # Underflowed exp2 is added to 1: both zero and a subnormal round
        # to exactly 1. FTZ here preserves SiLU, and avoids ptxas's otherwise
        # mandatory half-exponent/square fixup for each of the four outputs.
        exponent[r] = ex2(exponent[r], True)
    for r in cutlass.range_constexpr(4):
        exponent[r] = cutlass.Float32(1) + exponent[r]
    for r in cutlass.range_constexpr(4):
        exponent[r] = rcp(exponent[r])
    for r in cutlass.range_constexpr(4):
        values[r] = values[r] * exponent[r]


@cute.jit
def norm_four(values, inv, rms: cutlass.Constexpr):
    inv.fill(cutlass.Float32(0))
    for j in cutlass.range_constexpr(4):
        for r in cutlass.range_constexpr(4):
            inv[r] = inv[r] + values[j, r] * values[j, r]
    for shift in cutlass.range_constexpr(5):
        for r in cutlass.range_constexpr(4):
            inv[r] = inv[r] + cute.arch.shuffle_sync_bfly(inv[r], 1 << shift)
    for r in cutlass.range_constexpr(4):
        ss = inv[r]
        if cutlass.const_expr(rms):
            ss = ss * cutlass.Float32(1 / 128)
        inv[r] = cute.math.rsqrt(ss + cutlass.Float32(1e-6), fastmath=True)


@cute.jit
def swizzled_address(ptr, mask: cutlass.Constexpr):
    # ptrtoint returns the unswizzled address. Match K_SW128 / MN_SW32's
    # byte-level S<B,4,3> mapping before dropping the pointer's swizzle type.
    address = ptr.toint()
    return address ^ ((address >> 3) & mask)


@cute.jit
def shared_row(base, offset: cutlass.Constexpr):
    address = base + offset
    # Address is already swizzled; do not attach a second swizzle to the pointer.
    ptr = cute.make_ptr(cutlass.BFloat16, address, cute.AddressSpace.smem, assumed_align=8)
    return cute.make_tensor(ptr, cute.make_layout((4,)))


@cute.jit
def unpack(value, high: cutlass.Constexpr):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value()],
            "and.b32 $0, $1, 0xffff0000;" if high else "shl.b32 $0, $1, 16;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def pack(lo, hi):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [lo.ir_value(), hi.ir_value()],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def ffma(a, b, c):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value() for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def current_cta():
    # Volatile rematerialization keeps CP metadata out of long register ranges.
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %ctaid.x;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def load_raw_vector(source, dst):
    if source.toint() % 8 == 0:
        src = cute.make_tensor(source.align(8), cute.make_layout((4,)))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
        )
        cute.copy(atom, src, dst)
    else:
        # Projected/history/weights may have only BF16 alignment.
        for j in cutlass.range_constexpr(4):
            dst[j] = source[j]


@dataclass
class CPWorkDesc(WorkDesc):
    segment: cutlass.Int32
    kind: cutlass.Int32  # 0=N, 1=M


class FusedChunkKernel(ChunkKernelSm120):
    def __init__(
        self,
        hk,
        hv,
        producer_registers=72,
        iket=False,
        prefetch=2,
        watchdog=False,
        variant="full",
        pdl=True,
        q_lookahead=False,
        remap_y=False,
    ):
        super().__init__(hk, hv)
        self.fused = True
        self.variant = variant
        self.pdl = pdl
        self.remap_y = remap_y
        self.q_lookahead = q_lookahead and variant != "mn"
        self.no_output = variant == "mn"
        self.watchdog = watchdog
        self.prefetch = prefetch
        # MN has no Q or norm work. Return its surplus producer registers to
        # math; the default 56/224 allocation fits the same 168*384 budget.
        self.producer_registers = (
            min(producer_registers, 56) if self.no_output else producer_registers
        )
        self.iket = iket
        self.conv_dim = (2 * hk + hv) * 128

    def get_register_requirements(self, *args):
        # The launch allocates 168 registers x 384 threads = 64512: 128 p + 256 m must fit,
        # or setmaxnreg.inc in the math warpgroups never returns.
        return self.producer_registers, ((504 - self.producer_registers) // 2 // 8) * 8

    @cute.jit
    def __call__(
        self,
        projected,
        weights,
        alog,
        bias,
        norm,
        cu,
        slots,
        initial,
        state,
        history,
        y,
        tensormaps,
        workspace,
        seg_len: cutlass.Int32,
        num_segments: cutlass.Int32,
        rows,
        capture: cutlass.Int32,
        stream,
    ):
        y3 = cute.make_tensor(
            y.iterator, cute.make_layout((y.shape[0], self.hv, 128), stride=(self.hv * 128, 128, 1))
        )
        self.row_stride = projected.stride[0]
        bundle = (
            projected,
            weights,
            alog,
            bias,
            norm,
            history,
            workspace,
            seg_len,
            num_segments,
            rows,
            capture,
        )
        super().__call__(
            None,
            None,
            None,
            bundle,
            None,
            y3,
            state,
            slots,
            initial,
            cu,
            cutlass.Float32(128**-0.5),
            tensormaps,
            stream,
        )

    @cute.jit
    def get_next_work(self, cu_seqlens, num_q_heads, num_v_heads, num_sab_heads):
        if cutlass.const_expr(self.variant == "full"):
            return super().get_next_work(cu_seqlens, num_q_heads, num_v_heads, num_sab_heads)
        else:
            bx, _, _ = cute.arch.block_idx()
            r, head = bx // self.hv, bx % self.hv
            segment, kind = r, cutlass.Int32(0)
            if cutlass.const_expr(self.no_output):
                segment = cutlass.Int32(0)
                if r > 0:
                    segment = 1 + (r - 1) // 2
                    kind = (r - 1) % 2
            start = cu_seqlens[0] + segment * self.seg_len
            length = cutlass.min(self.seg_len, cu_seqlens[1] - start)
            return CPWorkDesc(
                cutlass.Int32(0),
                head * self.hk // self.hv,
                head,
                start,
                length,
                cutlass.Int32(0),
                segment,
                kind,
            )

    @cute.jit
    def workspace_view(self, plane, segment, head):
        w = self.cp_workspace
        view = cute.make_tensor(w.iterator, cute.select(w.layout, mode=[4, 3, 2, 1, 0]))
        return view[None, None, head, segment, plane]

    @cute.jit
    def load_state(self, acc, mma, state, initial, slots, work):
        if cutlass.const_expr(self.variant == "full"):
            super().load_state(acc, mma, state, initial, slots, work)
        elif cutlass.const_expr(self.variant == "seg"):
            # Both math warpgroups execute this before any W load or pool store.
            # Independent first-chunk KK/QK work can already have run.
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            self.kv_load(acc, self.workspace_view(0, work.segment, work.v_head_idx()), mma)
            # A last segment shorter than one chunk starts at p: capture its loaded state.
            self.maybe_store_capture(acc, mma, state, work, cutlass.Int32(0))
        else:
            if work.segment == 0:
                super().load_state(acc, mma, state, initial, slots, work)
                self.kv_store(acc, self.workspace_view(0, 0, work.v_head_idx()), mma)
            elif work.kind == 1:
                coords = mma.partition_C(cute.make_identity_tensor((128, 128)))
                for i in cutlass.range_constexpr(cute.size(acc)):
                    v, k = coords[i]
                    acc[i] = cutlass.Float32(v == k)

    @cute.jit
    def store_state(self, acc, mma, state, slots, work):
        if cutlass.const_expr(self.variant == "full"):
            super().store_state(acc, mma, state, slots, work)
        else:
            # Re-read CTA coordinates at the tail: keeping CP coordinates alive
            # through all chunks otherwise increases math register spills.
            bx = current_cta()
            r, head = bx // self.hv, bx % self.hv
            if cutlass.const_expr(self.variant == "seg"):
                slot = slots[0]
                if r == self.num_segments - 1 and slot >= 0:
                    self.kv_store(acc, self.pool_view(state, slot, head), mma)
            else:
                plane, segment = cutlass.Int32(0), cutlass.Int32(1)
                if r > 0:
                    plane, segment = 1 + (r - 1) % 2, 1 + (r - 1) // 2
                self.kv_store(acc, self.workspace_view(plane, segment, head), mma)

    @cute.jit
    def maybe_store_capture(self, acc, mma, state, work, block_end):
        """FlashInfer's state checkpoint, at the one point p = L // 64 * 64.

        ``block_end`` is local to the CTA's tokens and ``acc`` holds the state there. Stores
        to ``state[capture_rows[n]]`` when capturing, ``block_end`` is p, the row is >= 0
        and p >= 64. The callers only reach p >= 64: before the final chunk (full chunks
        before it), after it (L % 64 == 0), or, in the last CP segment, at its loaded start.
        """
        if cutlass.const_expr(not self.no_output):
            # Read work before branching: a branch that reads it carries it out, rebinding
            # the caller's work_desc to values local to an enclosing branch (site A).
            n, head = work.seq_idx, work.v_head_idx()
            p = work.seq_len // self.BLK_KV * self.BLK_KV
            if self.capture != 0 and block_end == p:
                if cutlass.const_expr(self.variant == "seg"):
                    bx = current_cta()
                    row = self.capture_rows[0]
                    if bx // self.hv == self.num_segments - 1 and row >= 0:
                        self.kv_store(acc, self.pool_view(state, row, bx % self.hv), mma)
                else:
                    row = self.capture_rows[n]
                    if row >= 0:
                        self.kv_store(acc, self.pool_view(state, row, head), mma)

    @cute.jit
    def raw_row(self, dst, x, history, tok, col, start, length, slot, initial):
        dst.fill(cutlass.BFloat16(0))
        # 64-bit row addressing: projected's stride is a specialization, T is symbolic.
        if tok >= 0 and tok < length:
            load_raw_vector(x.iterator + x.layout((cutlass.Int64(start + tok), col)), dst)
        elif tok < 0 and slot >= 0 and initial != 0:
            load_raw_vector(
                history.iterator + history.layout((cutlass.Int64(slot), tok + 3, col)), dst
            )

    @cute.jit
    def issue_raw(
        self,
        atom,
        tensor,
        smem,
        pipe,
        ps,
        raw_bar,
        work,
        blk,
        col,
        role,
        wait_name: cutlass.Constexpr,
    ):
        # Only the issuing warp needs to acquire the math-release barrier.
        # The other producers synchronize through raw completion before reading.
        if role == 0:
            self.trace_begin(wait_name)
            self.producer_acquire(
                pipe,
                ps,
                {"producer.wait_q": 10, "producer.wait_k": 11, "producer.wait_v": 12}[wait_name],
                blk,
            )
            self.trace_end()
            self.trace_begin("producer.tma_issue")
            # The acquire makes previous math reads and producer writes visible
            # to this issuing warp before the generic -> async proxy fence.
            cute.arch.fence_view_async_shared()
            barrier = raw_bar + ps.index
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 64 * 128 * 2)
            # CuTe TMA copy elects internally. An enclosing elect_one nests
            # elect.sync under a single-lane branch and can deadlock on SM120.
            # Reconverge the warp after expect_tx before either internal issuer.
            cute.arch.sync_warp()
            src = cute.domain_offset((cutlass.Int64(work.tok_offset) + blk * 64, col), tensor)
            tile = cute.zipped_divide(src, (64, 128))[
                ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
            ]
            dst, src = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(smem[None, None, ps.index], 0, 2),
                cute.group_modes(tile, 0, 2),
            )
            copies = cute.size(dst) // cute.size(atom.layout_dst_tv, mode=[1])
            tile_bytes = cute.size(dst) * self.dtype.width // 8
            assert copies == 2 and tile_bytes == 16384
            cute.copy(atom, src, dst, tma_bar_ptr=barrier)
            self.trace_end()

    @cute.jit
    def produce_tile(
        self,
        x,
        weights,
        history,
        work,
        slot,
        initial,
        col_base,
        smem,
        pipe,
        ps,
        raw_bar,
        blk,
        role,
        normalize: cutlass.Constexpr,
        name: cutlass.Constexpr,
        wait_name: cutlass.Constexpr,
    ):
        lane = cute.arch.lane_idx()
        col = col_base + lane * 4
        first = blk * 64 + role * 16
        prev = cute.make_rmem_tensor((4, 3), cutlass.Float32)
        # Define every lane on every outer-loop iteration. Complementary
        # role branches alone leave artificial loop-carried SSA live ranges.
        prev.fill(cutlass.Float32(0))
        taps = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        scratch = cute.make_rmem_tensor((2,), cutlass.Uint32)
        cp = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
        )
        # Warp zero's previous-chunk/initial history is independent of raw TMA.
        # Fetch it and the weights before waiting for completion.
        if role == 0:
            for h in cutlass.range_constexpr(3):
                self.raw_row(
                    cute.recast_tensor(scratch, cutlass.BFloat16),
                    x,
                    history,
                    first - 3 + h + (work.tok_offset if self.variant != "full" else 0),
                    col,
                    (cutlass.Int32(0) if self.variant != "full" else work.tok_offset),
                    (work.tok_offset + work.seq_len if self.variant != "full" else work.seq_len),
                    slot,
                    initial,
                )
                for j in cutlass.range_constexpr(4):
                    prev[j, h] = unpack(scratch[j // 2], j % 2 == 1)
        for j in cutlass.range_constexpr(4):
            load_raw_vector(
                weights.iterator + weights.layout((col + j, 0)),
                cute.recast_tensor(scratch, cutlass.BFloat16),
            )
            for h in cutlass.range_constexpr(4):
                taps[j, h] = unpack(scratch[h // 2], h % 2 == 1)
        self.trace_begin(wait_name)
        # Raw barriers start in phase zero; PipelineAsync producer starts in one.
        if cutlass.const_expr(self.watchdog):
            _watchdog.wait(
                raw_bar + ps.index,
                ps.phase ^ 1,
                self.watchdog_words,
                {"producer.wait_raw_q": 30, "producer.wait_raw_k": 31, "producer.wait_raw_v": 32}[
                    wait_name
                ],
                blk,
                ps.index,
            )
        else:
            cute.arch.mbarrier_wait(raw_bar + ps.index, ps.phase ^ 1)
        self.trace_end()
        self.trace_begin(name)
        if role != 0:
            for h in cutlass.range_constexpr(3):
                tile = cute.local_tile(
                    smem[None, None, ps.index], (1, 4), (role * 16 - 3 + h, lane)
                )
                cute.copy(
                    cp, cute.group_modes(tile, 0, 2), cute.recast_tensor(scratch, cutlass.BFloat16)
                )
                for j in cutlass.range_constexpr(4):
                    prev[j, h] = unpack(scratch[j // 2], j % 2 == 1)
        # All cross-warp boundary reads must finish before any in-place store.
        self.mark_progress({"producer.q": 50, "producer.k": 51, "producer.v": 52}[name], blk)
        cute.arch.barrier(barrier_id=NamedBarrier.PRODUCER, number_of_threads=128)
        # Tile allocation is only 128-byte aligned: its base can carry into
        # the SW128 phase bits. Materialize all eight phases from actual CuTe
        # pointers, then advance by 1024 bytes without changing the phase.
        addresses = cute.make_rmem_tensor((8,), cutlass.Int32)
        for r in cutlass.range_constexpr(8):
            tile = cute.local_tile(smem[None, None, ps.index], (1, 4), (role * 16 + r, lane))
            addresses[r] = swizzled_address(tile.iterator, 0x70)
        row8_bytes = cutlass.const_expr(2 * smem.layout((8, 0, 0)))
        assert row8_bytes == 1024
        for group in cutlass.range_constexpr(4):
            raw = cute.make_rmem_tensor((2, 4), cutlass.Uint32)
            mixed = cute.make_rmem_tensor((4, 4), cutlass.Float32)
            for r in cutlass.range_constexpr(4):
                row = group * 4 + r
                tile = shared_row(addresses[row % 8], (row // 8) * row8_bytes)
                cute.copy(cp, tile, cute.recast_tensor(raw[None, r], cutlass.BFloat16))
            # Four independent convolution chains, followed by four independent
            # SiLUs. Keep raw rows packed until their column is consumed.
            for j in cutlass.range_constexpr(4):
                window = cute.make_rmem_tensor((7,), cutlass.Float32)
                for h in cutlass.range_constexpr(3):
                    window[h] = prev[j, h]
                for r in cutlass.range_constexpr(4):
                    window[3 + r] = unpack(raw[j // 2, r], j % 2 == 1)
                values = cute.make_rmem_tensor((4,), cutlass.Float32)
                values.fill(cutlass.Float32(0))
                for h in cutlass.range_constexpr(4):
                    tap = taps[j, h]
                    for r in cutlass.range_constexpr(4):
                        values[r] = ffma(tap, window[r + h], values[r])
                silu_four(values)
                for r in cutlass.range_constexpr(4):
                    mixed[j, r] = values[r]
                for h in cutlass.range_constexpr(3):
                    prev[j, h] = window[4 + h]
            if cutlass.const_expr(normalize):
                inv = cute.make_rmem_tensor((4,), cutlass.Float32)
                norm_four(mixed, inv, False)
                for j in cutlass.range_constexpr(4):
                    for r in cutlass.range_constexpr(4):
                        mixed[j, r] = mixed[j, r] * inv[r]
            for r in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(2):
                    packed = pack(mixed[j * 2, r], mixed[j * 2 + 1, r])
                    if first + group * 4 + r >= work.seq_len:
                        packed = cutlass.Uint32(0)
                    scratch[j] = packed
                row = group * 4 + r
                tile = shared_row(addresses[row % 8], (row // 8) * row8_bytes)
                cute.copy(cp, cute.recast_tensor(scratch, cutlass.BFloat16), tile)
        # A second full barrier publishes generic stores to math's ldmatrix.
        # Raw TMA completion alone must never release the math warpgroups.
        pipe.producer_commit(ps)
        self.trace_end()

    @cute.jit
    def produce_gates(
        self, x, alog, bias, work, slot, vh, blk, role, sAlpha, sBeta, ap, bp, aps, bps, scale
    ):
        self.trace_begin("producer.wait_gates")
        self.producer_acquire(ap, aps, 14, blk)
        self.producer_acquire(bp, bps, 15, blk)
        self.trace_end()
        self.trace_begin("producer.gates")
        lane = cute.arch.lane_idx()
        if lane < 16:
            row = role * 16 + lane
            tok = blk * 64 + row
            g, beta = cutlass.Float32(0), cutlass.Float32(0)
            if tok < work.seq_len:
                b = cutlass.Float32(
                    x[cutlass.Int64(work.tok_offset + tok), self.conv_dim + self.hv * 128 + vh]
                )
                a = (
                    cutlass.Float32(
                        x[
                            cutlass.Int64(work.tok_offset + tok),
                            self.conv_dim + self.hv * 128 + self.hv + vh,
                        ]
                    )
                    + bias[vh]
                )
                t = ex2(-abs(a) * cutlass.Float32(1.4426950408889634))
                u = cutlass.Float32(1) + t
                sp = (a if a > 0 else cutlass.Float32(0)) + (
                    cute.math.log(u, fastmath=False) - ((u - cutlass.Float32(1)) - t)
                )
                g = -ex2(alog[vh] * cutlass.Float32(1.4426950408889634)) * sp
                beta = sigmoid(b)
            sAlpha[row, GateProcessor.CUMSUM_LOG, aps.index] = g * cutlass.Float32(
                1.4426950408889634
            )
            sBeta[row, bps.index] = beta
        self.mark_progress(53, blk)
        cute.arch.barrier(barrier_id=NamedBarrier.PRODUCER, number_of_threads=128)
        if role == 1:
            GateProcessor().run(sAlpha[None, None, aps.index], scale)
        # Publish scan results before ANY producer can commit or reuse this barrier.
        self.mark_progress(54, blk)
        cute.arch.barrier(barrier_id=NamedBarrier.PRODUCER, number_of_threads=128)
        ap.producer_commit(aps)
        bp.producer_commit(bps)
        self.trace_end()

    @cute.jit
    def produce_q(
        self,
        raw_atom,
        raw_tensor,
        sQ,
        qp,
        qs,
        raw_bar,
        work,
        blk,
        kh,
        role,
        x,
        weights,
        history,
        slot,
        read,
    ):
        self.issue_raw(
            raw_atom,
            raw_tensor,
            sQ,
            qp,
            qs,
            raw_bar,
            work,
            blk,
            kh * 128,
            role,
            "producer.wait_q",
        )
        self.produce_tile(
            x,
            weights,
            history,
            work,
            slot,
            read,
            kh * 128,
            sQ,
            qp,
            qs,
            raw_bar,
            blk,
            role,
            True,
            "producer.q",
            "producer.wait_raw_q",
        )

    @cute.jit
    def run_fused_producer(
        self,
        raw_atom,
        raw_tensor,
        raw_bar,
        bundle,
        initial,
        slots,
        work,
        blocks,
        role,
        sQ,
        sK,
        sV,
        sAlpha,
        sBeta,
        sO,
        qp,
        kp,
        vp,
        ap,
        bp,
        op,
        atom_o,
        tensor_o,
        tensormaps,
        num_seqs,
        scale,
    ):
        x, weights, alog, bias, norm, history = bundle[:6]
        slot = slots[work.seq_idx]
        read = initial[work.seq_idx]
        kh, vh = work.q_head_idx(), work.v_head_idx()
        qs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.q_stage)
        ks = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.k_stage)
        vs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.v_stage)
        aps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        bps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        os = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        norm_regs = cute.make_rmem_tensor((4,), cutlass.Float32)
        if cutlass.const_expr(not self.no_output and not self.remap_y):
            for j in cutlass.range_constexpr(4):
                norm_regs[j] = norm[cute.arch.lane_idx() * 4 + j]
        # Raw K has its own issue cursor, one chunk ahead of transformation.
        ki = ks.clone()
        self.issue_raw(
            raw_atom,
            raw_tensor,
            sK,
            kp,
            ki,
            raw_bar + 1,
            work,
            0,
            (self.hk + kh) * 128,
            role,
            "producer.wait_k",
        )
        ki.advance()
        # K(0) and gates(0) first; afterwards K(c + 1) and gates(c + 1) end iteration c, so
        # Q(c), the only single-stage tile gated by a math release, starts as soon as
        # Q(c - 1) is released.
        self.produce_tile(
            x,
            weights,
            history,
            work,
            slot,
            read,
            (self.hk + kh) * 128,
            sK,
            kp,
            ks,
            raw_bar + 1,
            cutlass.Int32(0),
            role,
            True,
            "producer.k",
            "producer.wait_raw_k",
        )
        ks.advance()
        self.produce_gates(
            x,
            alog,
            bias,
            work,
            slot,
            vh,
            cutlass.Int32(0),
            role,
            sAlpha,
            sBeta,
            ap,
            bp,
            aps,
            bps,
            scale,
        )
        aps.advance()
        bps.advance()
        if cutlass.const_expr(self.q_lookahead):
            self.produce_q(
                raw_atom,
                raw_tensor,
                sQ,
                qp,
                qs,
                raw_bar,
                work,
                cutlass.Int32(0),
                kh,
                role,
                x,
                weights,
                history,
                slot,
                read,
            )
            qs.advance()
        for blk in cutlass.range(blocks + 1, unroll=1):
            if cutlass.const_expr(self.pdl and self.no_output):
                # Each CTA signals once, at the start of its final chunk.
                # The runtime releases the dependent grid after all CTAs signal.
                if blk == blocks - 1 and role == 0:
                    cute.arch.griddepcontrol_launch_dependents()
            if blk < blocks:
                if cutlass.const_expr(not self.no_output):
                    if cutlass.const_expr(not self.q_lookahead):
                        self.produce_q(
                            raw_atom,
                            raw_tensor,
                            sQ,
                            qp,
                            qs,
                            raw_bar,
                            work,
                            blk,
                            kh,
                            role,
                            x,
                            weights,
                            history,
                            slot,
                            read,
                        )
                        qs.advance()
                if cutlass.const_expr(self.no_output):
                    r = current_cta() // self.hv
                    if r > 0 and (r - 1) % 2 == 1:
                        self.producer_acquire(vp, vs, 12, blk)
                        lane = cute.arch.lane_idx()
                        zeros = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                        zeros.fill(cutlass.BFloat16(0))
                        atom = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
                        )
                        # Keep zero-fill addresses short-lived (unrolling all
                        # 64 scalar stores spills producer registers).
                        for row in cutlass.range(16, unroll=1):
                            tile = cute.local_tile(
                                sV[None, None, vs.index], (1, 4), (role * 16 + row, lane)
                            )
                            cute.copy(atom, zeros, cute.group_modes(tile, 0, 2))
                        vp.producer_commit(vs)
                    else:
                        self.issue_raw(
                            raw_atom,
                            raw_tensor,
                            sV,
                            vp,
                            vs,
                            raw_bar + 3,
                            work,
                            blk,
                            (2 * self.hk + vh) * 128,
                            role,
                            "producer.wait_v",
                        )
                        self.produce_tile(
                            x,
                            weights,
                            history,
                            work,
                            slot,
                            read,
                            (2 * self.hk + vh) * 128,
                            sV,
                            vp,
                            vs,
                            raw_bar + 3,
                            blk,
                            role,
                            False,
                            "producer.v",
                            "producer.wait_raw_v",
                        )
                else:
                    self.issue_raw(
                        raw_atom,
                        raw_tensor,
                        sV,
                        vp,
                        vs,
                        raw_bar + 3,
                        work,
                        blk,
                        (2 * self.hk + vh) * 128,
                        role,
                        "producer.wait_v",
                    )
                    self.produce_tile(
                        x,
                        weights,
                        history,
                        work,
                        slot,
                        read,
                        (2 * self.hk + vh) * 128,
                        sV,
                        vp,
                        vs,
                        raw_bar + 3,
                        blk,
                        role,
                        False,
                        "producer.v",
                        "producer.wait_raw_v",
                    )
                vs.advance()
            if cutlass.const_expr(self.q_lookahead):
                # V(c) must be published first: the global FP16 guard defers
                # Q(c)'s final read/release until NewV(c). No dependence on
                # y(c-1) or K(c+1), whose work follows this lookahead.
                if blk + 1 < blocks:
                    self.produce_q(
                        raw_atom,
                        raw_tensor,
                        sQ,
                        qp,
                        qs,
                        raw_bar,
                        work,
                        blk + 1,
                        kh,
                        role,
                        x,
                        weights,
                        history,
                        slot,
                        read,
                    )
                    qs.advance()
            if blk > 0:
                if cutlass.const_expr(not self.no_output):
                    # All producer threads wait once before the first y handling.
                    # Raw TMA, conv/gates, and the independent input prologue above
                    # are free to overlap A/fixup. History is only committed later.
                    if cutlass.const_expr(self.pdl and self.variant == "seg"):
                        if blk == 1:
                            cute.arch.griddepcontrol_wait()
                    if cutlass.const_expr(self.remap_y):
                        self.normalize_output_remapped(
                            x, norm, sO, op, os, work, slot, vh, blk - 1, role
                        )
                    else:
                        self.normalize_output(
                            x, norm_regs, sO, op, os, work, slot, vh, blk - 1, role
                        )
                    # Only warp 1 releases O (32 arrivals). Every warp advances its
                    # own consumer phase, including those that do not issue TMA.
                    if role == 1:
                        os = OutputStore(64, 128).step(
                            sO,
                            atom_o,
                            tensor_o,
                            tensormaps,
                            op,
                            os,
                            work,
                            vh,
                            blk - 1,
                            blocks,
                            num_seqs,
                            already_waited=True,
                        )
                        if cutlass.const_expr(self.watchdog):
                            cute.arch.mbarrier_arrive(self.watchdog_store)
                    else:
                        os.advance()
                    if cutlass.const_expr(self.watchdog):
                        # Bulk-group completion cannot be polled. Other producer
                        # warps monitor warp 1, including the final output store.
                        if role != 1:
                            _watchdog.wait(
                                self.watchdog_store,
                                (blk - 1) % 2,
                                self.watchdog_words,
                                43,
                                blk - 1,
                                0,
                            )
            # K(blk + 1) waits for math to release K(blk - 1), the end of chunk blk - 1.
            if blk + 1 < blocks:
                self.issue_raw(
                    raw_atom,
                    raw_tensor,
                    sK,
                    kp,
                    ki,
                    raw_bar + 1,
                    work,
                    blk + 1,
                    (self.hk + kh) * 128,
                    role,
                    "producer.wait_k",
                )
                ki.advance()
                self.produce_tile(
                    x,
                    weights,
                    history,
                    work,
                    slot,
                    read,
                    (self.hk + kh) * 128,
                    sK,
                    kp,
                    ks,
                    raw_bar + 1,
                    blk + 1,
                    role,
                    True,
                    "producer.k",
                    "producer.wait_raw_k",
                )
                ks.advance()
                self.produce_gates(
                    x,
                    alog,
                    bias,
                    work,
                    slot,
                    vh,
                    blk + 1,
                    role,
                    sAlpha,
                    sBeta,
                    ap,
                    bp,
                    aps,
                    bps,
                    scale,
                )
                aps.advance()
                bps.advance()
        if cutlass.const_expr(not self.no_output):
            self.producer_tail(qp, qs, 60, blocks)
        self.producer_tail(kp, ks, 61, blocks)
        self.producer_tail(vp, vs, 62, blocks)
        self.producer_tail(ap, aps, 64, blocks)
        self.producer_tail(bp, bps, 65, blocks)

    @cute.jit
    def load_z_group(self, z, x, work, slot, vh, blk, role, group):
        token = (cute.arch.lane_idx() // 4) % 4
        d = (cute.arch.lane_idx() % 4) * 4 + (cute.arch.lane_idx() // 16) * 16
        for batch in cutlass.range_constexpr(4 // min(self.prefetch, 4)):
            for r in cutlass.range_constexpr(min(self.prefetch, 4)):
                quarter = batch * min(self.prefetch, 4) + r
                tok = blk * 64 + role * 16 + group * 4 + token
                row = cute.recast_tensor(z[None, quarter], cutlass.BFloat16)
                row.fill(cutlass.BFloat16(0))
                if tok < work.seq_len:
                    source = x.iterator + x.layout(
                        (
                            cutlass.Int64(work.tok_offset + tok),
                            self.conv_dim + vh * 128 + d + quarter * 32,
                        )
                    )
                    src = cute.make_tensor(source.align(8), cute.make_layout((4,)))
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
                    )
                    cute.copy(atom, src, row)

    @cute.jit
    def normalize_output_remapped(self, x, norm, sO, op, os, work, slot, vh, blk, role):
        lane = cute.arch.lane_idx()
        # Four tokens per warp, eight lanes per token. Each half warp covers
        # four distinct 32-byte token segments in MN_SW32: LDS/STS.64 need
        # exactly two wavefronts, instead of eight for a single-token warp.
        d = (lane % 4) * 4 + (lane // 16) * 16
        token = (lane // 4) % 4
        cp = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
        )
        z = cute.make_rmem_tensor((2, 4), cutlass.Uint32)
        self.trace_begin("producer.y")
        self.load_z_group(z, x, work, slot, vh, blk, role, 0)
        norm_values = cute.make_rmem_tensor((4, 4), cutlass.Float32)
        for quarter in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(4):
                norm_values[j, quarter] = norm[d + quarter * 32 + j]
        self.trace_end()
        self.trace_begin("producer.wait_o")
        self.consumer_wait(op, os, 23, blk)
        self.trace_end()
        self.trace_begin("producer.y")
        # Two pointers cover both SW32 phases; token+8 preserves the phase.
        addresses = cute.make_rmem_tensor((2,), cutlass.Int32)
        for g in cutlass.range_constexpr(2):
            tile = cute.local_tile(
                sO[None, None, os.index], (4, 1), (d // 4, role * 16 + token + g * 4)
            )
            addresses[g] = swizzled_address(tile.iterator, 0x10)
        for group in cutlass.range_constexpr(4):
            if cutlass.const_expr(group > 0):
                self.load_z_group(z, x, work, slot, vh, blk, role, group)
            values = cute.make_rmem_tensor((4, 4), cutlass.Float32)
            packed = cute.make_rmem_tensor((2,), cutlass.Uint32)
            sums = cute.make_rmem_tensor((4,), cutlass.Float32)
            sums.fill(cutlass.Float32(0))
            for quarter in cutlass.range_constexpr(4):
                tile = shared_row(addresses[group % 2], (group // 2) * 256 + quarter * 4096)
                cute.copy(cp, tile, cute.recast_tensor(packed, cutlass.BFloat16))
                for j in cutlass.range_constexpr(4):
                    value = unpack(packed[j // 2], j % 2 == 1)
                    values[j, quarter] = value
                    sums[quarter] = sums[quarter] + value * value
            # Same reduction tree over d as the original xor 1,2,4,8,16:
            # lane xor 16 now joins adjacent 16-d blocks; registers join 32/64.
            for shift in cutlass.range_constexpr(3):
                for quarter in cutlass.range_constexpr(4):
                    sums[quarter] = sums[quarter] + cute.arch.shuffle_sync_bfly(
                        sums[quarter], (1, 2, 16)[shift]
                    )
            ss = (sums[0] + sums[1]) + (sums[2] + sums[3])
            inv = cute.math.rsqrt(
                ss * cutlass.Float32(1 / 128) + cutlass.Float32(1e-6), fastmath=True
            )
            for quarter in cutlass.range_constexpr(4):
                gates = cute.make_rmem_tensor((4,), cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    gates[j] = unpack(z[j // 2, quarter], j % 2 == 1)
                silu_four(gates)
                for j in cutlass.range_constexpr(4):
                    values[j, quarter] = (
                        norm_values[j, quarter] * values[j, quarter] * inv * gates[j]
                    )
                for j in cutlass.range_constexpr(2):
                    value = pack(values[j * 2, quarter], values[j * 2 + 1, quarter])
                    if blk * 64 + role * 16 + group * 4 + token >= work.seq_len:
                        value = cutlass.Uint32(0)
                    packed[j] = value
                tile = shared_row(addresses[group % 2], (group // 2) * 256 + quarter * 4096)
                cute.copy(cp, cute.recast_tensor(packed, cutlass.BFloat16), tile)
        cute.arch.fence_view_async_shared()
        self.trace_end()
        self.trace_begin("producer.wait_y")
        self.mark_progress(55, blk)
        cute.arch.barrier(barrier_id=NamedBarrier.PRODUCER, number_of_threads=128)
        self.trace_end()

    @cute.jit
    def load_z_batch(self, dst, x, work, slot, vh, blk, role, batch):
        for r in cutlass.range_constexpr(self.prefetch):
            tok = blk * 64 + role * 16 + batch * self.prefetch + r
            row = cute.recast_tensor(dst[None, r], cutlass.BFloat16)
            row.fill(cutlass.BFloat16(0))
            if tok < work.seq_len:
                load_raw_vector(
                    x.iterator
                    + x.layout(
                        (
                            cutlass.Int64(work.tok_offset + tok),
                            self.conv_dim + vh * 128 + cute.arch.lane_idx() * 4,
                        )
                    ),
                    row,
                )

    @cute.jit
    def normalize_output(self, x, norm, sO, op, os, work, slot, vh, blk, role):
        lane = cute.arch.lane_idx()
        # All sixteen raw z rows (32 packed registers) precede the O wait.
        # PF now only groups z loads; K/Q/V have no software prefetch ring.
        z = cute.make_rmem_tensor((2, self.prefetch, 16 // self.prefetch), cutlass.Uint32)
        self.trace_begin("producer.y")
        for batch in cutlass.range_constexpr(16 // self.prefetch):
            self.load_z_batch(z[None, None, batch], x, work, slot, vh, blk, role, batch)
        self.trace_end()
        z_rows = cute.make_tensor(z.iterator, cute.make_layout((2, 16)))
        self.trace_begin("producer.wait_o")
        self.consumer_wait(op, os, 23, blk)
        self.trace_end()
        self.trace_begin("producer.y")
        # Constant indices consume Z in place: no register FIFO copies and no
        # loop-carried live range for already consumed rows.
        addresses = cute.make_rmem_tensor((8,), cutlass.Int32)
        for r in cutlass.range_constexpr(8):
            tile = cute.local_tile(sO[None, None, os.index], (4, 1), (lane, role * 16 + r))
            addresses[r] = swizzled_address(tile.iterator, 0x10)
        row8_bytes = cutlass.const_expr(2 * sO.layout((0, 8, 0)))
        assert row8_bytes == 256
        cp = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
        )
        for group in cutlass.range_constexpr(4):
            values = cute.make_rmem_tensor((4, 4), cutlass.Float32)
            packed = cute.make_rmem_tensor((2,), cutlass.Uint32)
            for r in cutlass.range_constexpr(4):
                tile = shared_row(addresses[(group * 4 + r) % 8], (group // 2) * row8_bytes)
                cute.copy(cp, tile, cute.recast_tensor(packed, cutlass.BFloat16))
                for j in cutlass.range_constexpr(4):
                    value = unpack(packed[j // 2], j % 2 == 1)
                    values[j, r] = value
            inv = cute.make_rmem_tensor((4,), cutlass.Float32)
            norm_four(values, inv, True)
            for j in cutlass.range_constexpr(4):
                gates = cute.make_rmem_tensor((4,), cutlass.Float32)
                for r in cutlass.range_constexpr(4):
                    gates[r] = unpack(z_rows[j // 2, group * 4 + r], j % 2 == 1)
                silu_four(gates)
                for r in cutlass.range_constexpr(4):
                    value = norm[j] * values[j, r] * inv[r] * gates[r]
                    values[j, r] = value
            for r in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(2):
                    value = pack(values[j * 2, r], values[j * 2 + 1, r])
                    if blk * 64 + role * 16 + group * 4 + r >= work.seq_len:
                        value = cutlass.Uint32(0)
                    packed[j] = value
                tile = shared_row(addresses[(group * 4 + r) % 8], (group // 2) * row8_bytes)
                cute.copy(cp, cute.recast_tensor(packed, cutlass.BFloat16), tile)
        cute.arch.fence_view_async_shared()
        self.trace_end()
        self.trace_begin("producer.wait_y")
        self.mark_progress(55, blk)
        cute.arch.barrier(barrier_id=NamedBarrier.PRODUCER, number_of_threads=128)
        self.trace_end()


@cute.kernel
def commit_history(
    x: cute.Tensor,
    cu: cute.Tensor,
    slots: cute.Tensor,
    initial: cute.Tensor,
    history: cute.Tensor,
    state: cute.Tensor,
    rows: cute.Tensor,
    capture: cutlass.Int32,
):
    n, block, _ = cute.arch.block_idx()
    lane, _, _ = cute.arch.thread_idx()
    col = block * 128 + lane
    # A capture row gets the history at p = L // 64 * 64 >= 64: rows p-3..p-1.
    if capture != 0 and col < history.shape[2]:
        row = rows[n]
        start = cu[n]
        p = (cu[n + 1] - start) // 64 * 64
        if row >= 0 and p >= 64:
            for h in cutlass.range_constexpr(3):
                history[cutlass.Int64(row), h, col] = x[cutlass.Int64(start + p - 3 + h), col]
    slot = slots[n]
    # Empty extends from zero must commit zero recurrent state too. Use the
    # history grid's columns as a strided partition of the state row.
    if slot >= 0 and cu[n] == cu[n + 1] and initial[n] == 0:
        for index in cutlass.range(
            col, state.shape[1] * 16384, cute.ceil_div(history.shape[2], 128) * 128
        ):
            state[cutlass.Int64(slot), index // 16384, index // 128 % 128, index % 128] = (
                cutlass.Float32(0)
            )
    if slot >= 0 and col < history.shape[2]:
        start, end = cu[n], cu[n + 1]
        length = end - start
        # Load ALL source rows before writing any destination (L=1,2 overlaps).
        values = cute.make_rmem_tensor((3,), cutlass.BFloat16)
        for h in cutlass.range_constexpr(3):
            tok = length - 3 + h
            value = cutlass.BFloat16(0)
            if tok >= 0:
                value = x[cutlass.Int64(start + tok), col]
            elif initial[n] != 0:
                value = history[cutlass.Int64(slot), tok + 3, col]
            values[h] = value
        for h in cutlass.range_constexpr(3):
            history[cutlass.Int64(slot), h, col] = values[h]


@cute.jit
def launch_history(x, cu, slots, initial, history, state, rows, capture: cutlass.Int32, stream):
    commit_history(x, cu, slots, initial, history, state, rows, capture).launch(
        grid=(slots.shape[0], cute.ceil_div(history.shape[2], 128), 1),
        block=(128, 1, 1),
        stream=stream,
    )
