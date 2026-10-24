# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Head-local raw replay with optional section timestamps.

Original implementation, arithmetic from this repository's product decode.
Atomic/fence instructions follow https://docs.nvidia.com/cuda/parallel-thread-execution/.
The trusted one-launch policy joins each row's readers with a per-row arrival.
Strict mode joins mutable S1/G1 control readers with a grid arrival. The explicit
snapshot policy uses an ordered control kernel followed by counter-free compute.
All use head-local output normalization.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.utils import SmemAllocator

from .._layout import Layout
from ._index import wide_pool_offsets


@cute.jit
def mul_rn(a, b):
    """A rounding boundary, including when its consumer is an explicit FMA."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def product_exp(x):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value()],
            "{ .reg .f32 t; mul.rn.f32 t, $1, 0f3FB8AA3B; ex2.approx.f32 $0, t; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def sigmoid(x):
    e = product_exp(-cute.math.abs(x))
    # Match Triton's PTX instruction, not an assumed reciprocal refinement.
    r = Float32(
        llvm.inline_asm(
            T.f32(),
            [(Float32(1) + e).ir_value()],
            "div.full.f32 $0, 0f3F800000, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return r if x >= 0 else e * r


@cute.jit
def product_log1p(x):
    # Triton's linked __nv_log1pf uses FTZ internally. Its argument here is
    # exp(-abs(a)) in [0,1]. Flush a subnormal argument before calling the
    # system implementation; never transcribe a vendor polynomial.
    x = Float32(
        llvm.inline_asm(
            T.f32(),
            [x.ir_value()],
            "mul.ftz.f32 $0, $1, 0f3F800000;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return cute.math.log1p(x)


@cute.jit
def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@cute.jit
def sum_group(x, lanes: cutlass.Constexpr):
    for d in cutlass.range_constexpr(lanes.bit_length() - 1):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=(lanes // 2) >> d)
    return x


@cute.jit
def sum32(x):
    for d in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> d)
    return x


@cute.jit
def product_dot(a, b, chunks: cutlass.Constexpr):
    # TILE's register order is c=0,1,2,3 then the next 16-column block.
    # LLVM contracts the first add as fma(p0, rounded(p1)), then p2..p31.
    total = mul_rn(a[0, 1], b[0, 1])
    total = cute.math.fma(a[0, 0], b[0, 0], total)
    for i in cutlass.range_constexpr(2, chunks * 4):
        total = cute.math.fma(a[i // 4, i % 4], b[i // 4, i % 4], total)
    return total


@cute.jit
def product_norm(a, b, ncol: cutlass.Constexpr = False):
    # VEC: four 32-element warps, shuffle 16,8,4,2,1 within each,
    # then warp sums (0+2)+(1+3). Emulate those warps in four registers.
    if cutlass.const_expr(ncol):
        tid, _, _ = cute.arch.thread_idx()
        aa = cute.make_rmem_tensor((8, 4), Float32)
        bb = cute.make_rmem_tensor((8, 4), Float32)
        for i in cutlass.range_constexpr(32):
            source = ((i // 4) % 2) * 16 + (tid % 4) * 4 + i % 4
            aa[i // 4, i % 4] = cute.arch.shuffle_sync(a[i // 8], source)
            bb[i // 4, i % 4] = cute.arch.shuffle_sync(b[i // 8], source)
        result = sum_group(product_dot(aa, bb, 8), 4)
    else:
        partial = cute.make_rmem_tensor((4,), Float32)
        for j in cutlass.range_constexpr(4):
            v = mul_rn(a[j], b[j])
            v = cute.math.fma(a[j], b[j], cute.arch.shuffle_sync_bfly(v, offset=16))
            for d in cutlass.range_constexpr(4):
                v = v + cute.arch.shuffle_sync_bfly(v, offset=8 >> d)
            partial[j] = v
        result = (partial[0] + partial[2]) + (partial[1] + partial[3])
    return result


@cute.jit
def arrive(ptr):
    # The preceding CTA barrier joins control readers. An acq_rel RMW chain
    # orders those reads before the last CTA replaces mutable controls. This
    # election runs BEFORE state stores; no per-thread GPU fence is needed.
    # Stream completion publishes the disjoint state tiles to the next op.
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint().ir_value()],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cluster_sync():
    # Public PTX ISA 8.0: all threads participate, including inactive rows.
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.release; barrier.cluster.wait.acquire;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def cluster_store(ptr, rank, value):
    # ptr is already in the shared address space (as in state_vector).
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), rank.ir_value(), value.ir_value()],
        "{ .reg .u64 remote; mapa.shared::cluster.u64 remote, $0, $1; "
        "st.shared::cluster.f32 [remote], $2; }",
        "l,r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def load_checkpoint_position(ptr):
    # Atomic relaxed access: the late-reader rule needs one whole int64 value,
    # but no ordering with another CTA's disjoint state tile is required.
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def store_checkpoint_position(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.relaxed.gpu.global.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def vector_load(src, offset, frag):
    dtype = frag.element_type
    ptr = cute.make_ptr(
        dtype,
        src.iterator.toint() + Int64(offset) * (dtype.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), dtype, num_bits_per_copy=128)
    cute.copy(atom, cute.make_tensor(ptr, cute.make_layout(cute.size(frag))), frag)


@cute.jit
def state_vector(ptr, offset, space: cutlass.Constexpr):
    aligned = cute.make_ptr(Float32, (ptr + offset).toint(), space, assumed_align=16)
    return cute.make_tensor(aligned, cute.make_layout(4))


@cute.jit
def vector_store(dst, offset, frag):
    dtype = frag.element_type
    ptr = cute.make_ptr(
        dtype,
        dst.iterator.toint() + Int64(offset) * (dtype.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
    cute.copy(atom, frag, cute.make_tensor(ptr, cute.make_layout(cute.size(frag))))


class Step:
    def __init__(
        self,
        hk,
        hv,
        window,
        batch,
        stride,
        config,
        slots,
        complete,
        advance,
        arithmetic,
        layout=None,
    ):
        self.hk, self.hv, self.window = hk, hv, window
        self.batch, self.stride, self.slots = batch, stride, slots
        self.complete, self.advance = complete, advance
        self.dc = (2 * hk + hv) * 128
        self.width = self.dc + 2 * hv
        self.layout = layout or Layout(
            hv * 16384, 3 * self.dc, hv * 16384, 3 * self.dc, out_batch=hv * 128, out_token=hv * 128
        )
        self.cp_stride = self.layout.checkpoint
        self.h_stride = self.layout.history
        self.index_stride = self.layout.indices
        self.position_stride = self.layout.positions
        self.out_stride = self.layout.out_batch
        # Keep the original operand order and integer widths for each dense
        # component. These flags are specialization constants, never device tests.
        self.dense_checkpoint = self.cp_stride == hv * 16384
        self.dense_history = self.h_stride == 3 * self.dc
        self.dense_indices = self.index_stride == 1
        self.dense_positions = self.position_stride == 1
        self.dense_out = self.out_stride == hv * 128
        self.aligned_x = self.layout.x_alignment == 16
        self.dense_batch_offsets = self.dense_out and self.dense_indices and self.dense_positions
        if self.dense_checkpoint and self.dense_history:
            self.wide_pool = wide_pool_offsets(slots, hv, window, self.width)
        else:
            self.wide_pool = wide_pool_offsets(
                slots, hv, window, self.width, self.cp_stride, self.h_stride
            )
        self.tile, self.warps, self.prefetch = config.value_tile, config.warps, config.prefetch
        self.lanes, self.sections = config.lanes, config.sections
        self.trusted, self.flush_always = config.trusted, config.flush_always
        self.immediate = config.immediate
        self.phase_features = False
        self.tokens = 1 if self.immediate else window
        self.current_row = 3 if self.immediate else window + 3
        self.snapshot_controls = config.snapshot_controls
        self.threads = self.warps * 32
        self.replay_threads = min(self.threads, self.tile * self.lanes)
        self.rows = self.tile // (self.replay_threads // self.lanes)
        self.chunks = 32 // self.lanes
        self.cols, self.nv = 256 + self.tile, 128 // self.tile
        self.clustered = self.nv > 1 and self.batch > 0
        self.feature_cols = 256 + 2 * self.tile
        self.speculative_ring = config.stages_ring(window)
        (
            self.product_bv,
            self.product_warps,
            self.product_fin,
            self.product_wsum,
            self.product_bfly,
            self.product_a3,
            self.product_ncol,
        ) = arithmetic

    @cute.jit
    def load_projected(self, x, offset, frag):
        if cutlass.const_expr(self.layout.x_alignment == 16):
            vector_load(x, offset, frag)
        else:
            for j in cutlass.range_constexpr(cute.size(frag)):
                frag[j] = x[Int64(offset) + j]

    @cute.jit
    def stamp(self, debug, b, h, tid, section: cutlass.Constexpr):
        if cutlass.const_expr(self.sections):
            tile, _, _ = cute.arch.block_idx()
            if (tid == 0) & (tile == 0):
                # Public PTX ISA special register, nanoseconds. The memory clobber
                # keeps memory work on its side of this diagnostic boundary.
                tick = Int64(
                    llvm.inline_asm(
                        T.i64(),
                        [],
                        "mov.u64 $0, %globaltimer;",
                        "=l,~{memory}",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                debug[(b * self.hv + h) * 7 + section] = tick

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cw: cute.Tensor,
        al: cute.Tensor,
        bias: cute.Tensor,
        nw: cute.Tensor,
        slots: cute.Tensor,
        positions: cute.Tensor,
        checkpoint: cute.Tensor,
        history: cute.Tensor,
        ring: cute.Tensor,
        metadata: cute.Tensor,
        epoch: cute.Tensor,
        record: cute.Tensor,
        sequence: cute.Tensor,
        poison: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        counters: cute.Tensor,
        debug: cute.Tensor,
        frozen: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # All indexing bounds come from the specialization; only validity has a
        # runtime length. Static views remove unused int64 extents from the CUDA
        # parameter block while preserving the public tensor/TVM calling ABI.
        x = cute.make_tensor(x.iterator, cute.make_layout(max(8, self.batch * self.stride)))
        cw = cute.make_tensor(cw.iterator, cute.make_layout(self.dc * 4))
        al = cute.make_tensor(al.iterator, cute.make_layout(self.hv))
        bias = cute.make_tensor(bias.iterator, cute.make_layout(self.hv))
        nw = cute.make_tensor(nw.iterator, cute.make_layout(128))
        if cutlass.const_expr(self.dense_indices):
            slots = cute.make_tensor(slots.iterator, cute.make_layout(max(1, self.batch)))
        else:
            slots = cute.make_tensor(
                slots.iterator, cute.make_layout(max(1, self.batch * self.index_stride))
            )
        if cutlass.const_expr(self.dense_positions):
            positions = cute.make_tensor(positions.iterator, cute.make_layout(max(1, self.batch)))
        else:
            positions = cute.make_tensor(
                positions.iterator, cute.make_layout(max(1, self.batch * self.position_stride))
            )
        if cutlass.const_expr(self.dense_checkpoint):
            checkpoint = cute.make_tensor(
                checkpoint.iterator, cute.make_layout(max(4, self.slots * self.hv * 16384))
            )
        else:
            checkpoint = cute.make_tensor(
                checkpoint.iterator, cute.make_layout(max(4, self.slots * self.cp_stride))
            )
        if cutlass.const_expr(self.dense_history):
            history = cute.make_tensor(
                history.iterator, cute.make_layout(max(8, self.slots * 3 * self.dc))
            )
        else:
            history = cute.make_tensor(
                history.iterator, cute.make_layout(max(8, self.slots * self.h_stride))
            )
        ring = cute.make_tensor(
            ring.iterator, cute.make_layout(max(8, self.slots * (self.window + 3) * self.width))
        )
        metadata = cute.make_tensor(metadata.iterator, cute.make_layout(max(8, self.slots * 8)))
        epoch = cute.make_tensor(epoch.iterator, cute.make_layout(1))
        record = cute.make_tensor(record.iterator, cute.make_layout(16))
        sequence = cute.make_tensor(sequence.iterator, cute.make_layout(1))
        poison = cute.make_tensor(poison.iterator, cute.make_layout(1))
        if cutlass.const_expr(self.dense_out):
            out = cute.make_tensor(
                out.iterator, cute.make_layout(max(8, self.batch * self.hv * 128))
            )
        else:
            out = cute.make_tensor(
                out.iterator, cute.make_layout(max(8, self.batch * self.out_stride))
            )
        counters = cute.make_tensor(
            counters.iterator, cute.make_layout(max(1, self.batch) if self.trusted else 1)
        )
        debug = cute.make_tensor(debug.iterator, cute.make_layout((self.batch * self.hv + 1) * 7))
        frozen = cute.make_tensor(frozen.iterator, cute.make_layout(2 * self.batch + 1))
        if cutlass.const_expr(self.snapshot_controls):
            self.snapshot(
                slots, positions, metadata, epoch, record, sequence, poison, frozen, history, ring
            ).launch(
                grid=(1, 1, 1),
                block=(32, 1, 1),
                stream=stream,
            )
        self.kernel(
            x,
            cw,
            al,
            bias,
            nw,
            slots,
            positions,
            checkpoint,
            history,
            ring,
            metadata,
            epoch,
            record,
            sequence,
            poison,
            valid,
            out,
            counters,
            debug,
            frozen,
        ).launch(
            grid=(self.nv if self.batch else 1, self.hv if self.batch else 1, max(1, self.batch)),
            block=(self.threads, 1, 1),
            cluster=(self.nv, 1, 1) if self.clustered else None,
            min_blocks_per_mp=1,
            stream=stream,
        )

        if cutlass.const_expr(self.immediate and self.snapshot_controls):
            self.snapshot_history(x, history, slots, frozen).launch(
                grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream
            )

    @cute.jit
    def column(self, c, kh, h, tile):
        col = kh * 128 + c
        if c >= 128:
            col = self.hk * 128 + kh * 128 + c - 128
        if c >= 256:
            col = 2 * self.hk * 128 + h * 128 + tile * self.tile + c - 256
        return col

    @cute.jit
    def validate(
        self,
        slots,
        positions,
        metadata,
        epoch,
        record,
        sequence,
        poison,
        lane,
        saved,
        bid,
        freeze: cutlass.Constexpr,
    ):
        # All first-hop reads are independent. Do not predicate slot/metadata
        # loads on status or poison: only the final error selection needs them.
        seq, current_epoch, poisoned = sequence[0], epoch[0], poison[0]
        # Scalar typed views avoid a vector-fragment MOV that made metadata
        # issue wait for the status LDG in the incoming SASS.
        words = cute.recast_tensor(record, Int32)
        old, code, done = cute.recast_tensor(record, Int64)[0], words[2], words[3]
        error = Int32(0)
        for b in range(lane, self.batch, 32):
            (slot, requested) = (
                (
                    Int64(slots[b])
                    if cutlass.const_expr(self.dense_indices)
                    else Int64(slots[Int64(b) * self.index_stride])
                ),
                (
                    Int64(positions[b])
                    if cutlass.const_expr(self.dense_positions)
                    else Int64(positions[Int64(b) * self.position_stride])
                ),
            )
            if slot != -1:
                if (slot < 0) | (slot >= self.slots):
                    error = error | Int32(1)
                else:
                    fields = cute.make_rmem_tensor((8,), Int64)
                    vector_load(metadata, Int64(slot) * 8, fields)
                    cp, pos, cursor, n = fields[0], fields[1], fields[2], fields[3]
                    good = (fields[4] == 1) & (fields[7] == current_epoch)
                    if cutlass.const_expr(self.immediate):
                        good = good & (n == 0) & (pos == cp)
                    else:
                        good = good & (n >= 0) & (n < self.window) & (pos - cp == n)
                        good = good & (cursor >= 0) & (cursor < self.window + 3)
                    if not good:
                        error = error | Int32(1)
                    if (requested != pos) | (pos == Int64(9223372036854775807)):
                        error = error | Int32(2)
                    if cutlass.const_expr(freeze):
                        saved[1 + b * 2] = pos
                        if cutlass.const_expr(not self.immediate):
                            if cutlass.const_expr(self.flush_always):
                                saved[2 + b * 2] = cursor + n * (self.window + 3)
                            else:
                                saved[2 + b * 2] = cursor
                    else:
                        if cutlass.const_expr(not self.immediate):
                            if b == bid:
                                saved[1], saved[2] = Int32(cursor), Int32(n)
        if cutlass.const_expr(self.complete):
            good_status = (seq > 0) & (old == 0) & (code == 0) & (done == 0)
        else:
            good_status = (seq > 0) & (old == seq) & (done == 0)
        error = error | code
        if poisoned != 0:
            error = code | Int32(32)
        if not good_status:
            error = Int32(64)
        return error

    @cute.jit
    def ring_row(self, t, cursor, n):
        return (cursor + t - n - 3 + 2 * (self.window + 3)) % (self.window + 3)

    @cute.jit
    def convolve(self, raw, weights, t, c, n, cursor=0):
        taps = cute.make_rmem_tensor((4,), Float32)
        for tap in cutlass.range_constexpr(4):
            rr = t + tap
            if cutlass.const_expr(self.speculative_ring):
                rr = self.ring_row(t + tap, cursor, n)
            if t + tap == n + 3:
                rr = Int32(self.current_row)
            taps[tap] = Float32(weights[c, tap]) * Float32(raw[rr, c])
        v, e1 = two_sum(taps[0], taps[1])
        v, e2 = two_sum(v, taps[2])
        v, e3 = two_sum(v, taps[3])
        v = v + ((e1 + e2) + e3)
        return v

    @cute.jit
    def prepare_gates(self, q, k, raw_gates, gates, al, bias, t, h, lane):
        kk = product_norm(k, k, self.product_ncol)
        qq = product_norm(q, q, self.product_ncol)
        qk = product_norm(q, k, self.product_ncol)
        if lane == 0:
            rk = cute.math.rsqrt(kk + 1e-6, fastmath=True)
            rq = cute.math.rsqrt(qq + 1e-6, fastmath=True) * Float32(128**-0.5)
            a, beta = raw_gates[t, 1], raw_gates[t, 0]
            a = a + Float32(bias[h])
            sp = cute.math.max(a, Float32(0)) + product_log1p(product_exp(-cute.math.abs(a)))
            gates[t, 0], gates[t, 3], gates[t, 4] = rk, rq, qk * rq * rk
            gates[t, 1] = product_exp(mul_rn(-product_exp(Float32(al[h])), sp))
            gates[t, 2] = sigmoid(beta)

    @cute.jit
    def replay_token(
        self,
        matrix,
        features,
        gates,
        values,
        t,
        tid,
        state_head,
        flush,
        final: cutlass.Constexpr,
        retain: cutlass.Constexpr = False,
        row_base=0,
    ):
        if tid < self.replay_threads:
            # One aligned broadcast carries all historical scalar controls.
            scalars = cute.make_rmem_tensor((4,), Float32)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
            cute.copy(atom, state_vector(gates.iterator, t * 8, cute.AddressSpace.smem), scalars)
            rk, alpha, beta, rq = scalars[0], scalars[1], scalars[2], scalars[3]
            if cutlass.const_expr(final):
                kq = gates[t, 4]
            k = cute.make_rmem_tensor((self.chunks, 4), Float32)
            for j in cutlass.range_constexpr(self.chunks):
                for c in cutlass.range_constexpr(4):
                    col = (tid % self.lanes) * 4 + j * self.lanes * 4 + c
                    k[j, c] = features[t, 128 + col]
            for r in cutlass.range_constexpr(self.rows):
                row = tid // self.lanes + r * (self.replay_threads // self.lanes)
                feature_row = row_base + row if cutlass.const_expr(self.phase_features) else row
                feature_tile = 128 if cutlass.const_expr(self.phase_features) else self.tile
                sv = matrix[r, None, None].load()
                sk = sum_group(product_dot(sv, k, self.chunks), self.lanes) * rk
                val = features[t, 256 + feature_row]
                # Product folds v's SiLU multiplication into the residual FMA.
                delta = beta * cute.math.fma(
                    val, features[t, 256 + feature_tile + feature_row], -mul_rn(alpha, sk)
                )
                if cutlass.const_expr(final):
                    q = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    for j in cutlass.range_constexpr(self.chunks):
                        for c in cutlass.range_constexpr(4):
                            col = (tid % self.lanes) * 4 + j * self.lanes * 4 + c
                            q[j, c] = features[t, col]
                    sq = (
                        sum_group(
                            product_dot(sv, q, self.chunks),
                            self.lanes,
                        )
                        * rq
                    )
                    if cutlass.const_expr(self.product_bv == 32):
                        if cutlass.const_expr(self.product_a3):
                            ov = cute.math.fma(alpha, sq, mul_rn(delta, kq))
                            if row_base + row < 64:
                                ov = cute.math.fma(delta, kq, mul_rn(alpha, sq))
                        else:
                            ov = cute.math.fma(alpha, sq, mul_rn(delta, kq))
                    elif cutlass.const_expr(self.product_bv == 64 and self.product_warps == 8):
                        # One row per product thread: its backend rounds delta*kq first.
                        ov = cute.math.fma(alpha, sq, mul_rn(delta, kq))
                    else:
                        ov = cute.math.fma(delta, kq, mul_rn(alpha, sq))
                    if tid % self.lanes == 0:
                        values[row_base + row] = ov
                if cutlass.const_expr(final):
                    if flush:
                        updated = cute.make_rmem_tensor((4,), Float32)
                        for j in cutlass.range_constexpr(self.chunks):
                            col = (tid % self.lanes) * 4 + j * self.lanes * 4
                            for c in cutlass.range_constexpr(4):
                                updated[c] = cute.math.fma(
                                    alpha, sv[j, c], mul_rn(delta * rk, k[j, c])
                                )
                            vector_store(state_head, row * 128 + col, updated)
                if cutlass.const_expr(not final or retain):
                    for j in cutlass.range_constexpr(self.chunks):
                        for c in cutlass.range_constexpr(4):
                            matrix[r, j, c] = cute.math.fma(
                                alpha, sv[j, c], mul_rn(delta * rk, k[j, c])
                            )

    @cute.jit
    def output_sum(self, ov):
        """Emulate the product's row reductions without a cross-CTA exchange."""
        part = cute.make_rmem_tensor((4,), Float32)
        if cutlass.const_expr(self.product_wsum):
            for j in cutlass.range_constexpr(4):
                v = mul_rn(ov[j], ov[j])
                for d in cutlass.range_constexpr(3):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << d)
                part[j] = v
            if cutlass.const_expr(self.product_bfly and not self.product_a3):
                for j in cutlass.range_constexpr(4):
                    v = part[j]
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=8)
                    part[j] = v + cute.arch.shuffle_sync_bfly(v, offset=16)
                total = (part[0] + part[1]) + (part[2] + part[3])
            else:
                total = Float32(0)
                for j in cutlass.range_constexpr(4):
                    for group in cutlass.range_constexpr(4):
                        total = total + cute.arch.shuffle_sync(part[j], group * 8)
        elif cutlass.const_expr(self.product_fin):
            if cutlass.const_expr(self.hv // self.hk > self.product_warps // 2):
                total = mul_rn(ov[1], ov[1])
                total = cute.math.fma(ov[0], ov[0], total)
                total = cute.math.fma(ov[2], ov[2], total)
                total = cute.math.fma(ov[3], ov[3], total)
                total = sum32(total)
            else:
                total = product_norm(ov, ov)
        else:
            # Each product warp owns eight rows. BV/warps determines whether
            # its initial partial has one or two products before row shuffles.
            for tile in cutlass.range_constexpr(128 // self.product_bv):
                if cutlass.const_expr(self.product_bv == 128):
                    for j in cutlass.range_constexpr(2):
                        v = cute.math.fma(ov[j], ov[j], mul_rn(ov[j + 2], ov[j + 2]))
                        part[j] = sum_group(v, 8)
                    v = part[0] + part[1]
                elif cutlass.const_expr(self.product_warps == 4):
                    v = cute.math.fma(
                        ov[2 * tile], ov[2 * tile], mul_rn(ov[2 * tile + 1], ov[2 * tile + 1])
                    )
                    v = sum_group(v, 8)
                else:
                    for j in cutlass.range_constexpr(2):
                        x = ov[2 * tile + j]
                        v = mul_rn(x, x)
                        v = cute.math.fma(x, x, cute.arch.shuffle_sync_bfly(v, offset=4))
                        v = v + cute.arch.shuffle_sync_bfly(v, offset=2)
                        part[j] = v + cute.arch.shuffle_sync_bfly(v, offset=1)
                    v = part[0] + part[1]
                v = v + cute.arch.shuffle_sync_bfly(v, offset=16)
                v = v + cute.arch.shuffle_sync_bfly(v, offset=8)
                part[2 + tile] = v
            total = part[2]
            if cutlass.const_expr(self.product_bv == 64):
                total = total + part[3]
        # Product publishes lane zero of each reduction; first-stage FMA
        # contraction can make other butterfly lanes differ by one ULP.
        return cute.arch.shuffle_sync(total, 0)

    @cute.kernel
    def snapshot(
        self,
        slots: cute.Tensor,
        positions: cute.Tensor,
        metadata: cute.Tensor,
        epoch: cute.Tensor,
        record: cute.Tensor,
        sequence: cute.Tensor,
        poison: cute.Tensor,
        frozen: cute.Tensor,
        history: cute.Tensor,
        ring: cute.Tensor,
    ):
        lane, _, _ = cute.arch.thread_idx()
        error = self.validate(
            slots, positions, metadata, epoch, record, sequence, poison, lane, frozen, 0, True
        )
        error = cute.arch.warp_redux_sync(error, "or")
        if lane == 0:
            frozen[0] = Int64(error)

        if cutlass.const_expr(not self.immediate):
            # Ordered preparation makes empty-ring seeds authoritative before any
            # speculative DMA or history writer in the counter-free compute grid.
            if error == 0:
                for b in range(self.batch):
                    if cutlass.const_expr(self.dense_indices):
                        slot = Int64(slots[b])
                    else:
                        slot = Int64(slots[Int64(b) * self.index_stride])
                    if (slot >= 0) & (slot < self.slots):
                        if metadata[slot * 8 + 3] == 0:
                            cursor = Int32(metadata[slot * 8 + 2])
                            # Conv columns only: seed gates are never read.
                            for i in range(lane, 3 * self.dc, 32):
                                t, col = i // self.dc, i % self.dc
                                rr = (cursor + t + self.window) % (self.window + 3)
                                if cutlass.const_expr(self.dense_history):
                                    ring[(slot * (self.window + 3) + rr) * self.width + col] = (
                                        history[(slot * 3 + t) * self.dc + col]
                                    )
                                else:
                                    ring[(slot * (self.window + 3) + rr) * self.width + col] = (
                                        history[slot * self.h_stride + t * self.dc + col]
                                    )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        cw: cute.Tensor,
        al: cute.Tensor,
        bias: cute.Tensor,
        nw: cute.Tensor,
        slots: cute.Tensor,
        positions: cute.Tensor,
        checkpoint: cute.Tensor,
        history: cute.Tensor,
        ring: cute.Tensor,
        metadata: cute.Tensor,
        epoch: cute.Tensor,
        record: cute.Tensor,
        sequence: cute.Tensor,
        poison: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        counters: cute.Tensor,
        debug: cute.Tensor,
        frozen: cute.Tensor,
    ):
        tile, h, bid = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.dense_batch_offsets):
            b = Int32(bid) if cutlass.const_expr(self.batch * self.stride < 2**31) else Int64(bid)
        else:
            b = (
                Int32(bid)
                if cutlass.const_expr(
                    self.batch
                    * max(self.stride, self.out_stride, self.index_stride, self.position_stride)
                    < 2**31
                )
                else Int64(bid)
            )
        self.stamp(debug, b, h, tid, 0)
        lane, warp = tid % 32, tid // 32
        smem = SmemAllocator()
        control = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        if cutlass.const_expr(not self.immediate and self.window <= 8):
            staged_state = smem.allocate_tensor(
                Float32, cute.make_layout(self.tile * 128), byte_alignment=16
            )
        elif cutlass.const_expr(self.sections):
            probe = smem.allocate_tensor(Float32, cute.make_layout(4), byte_alignment=16)
        raw = smem.allocate_tensor(
            BFloat16,
            cute.make_layout((self.current_row + 1, self.cols), stride=(self.cols, 1)),
            byte_alignment=16,
        )
        features = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.tokens, self.feature_cols), stride=(self.feature_cols, 1)),
            byte_alignment=16,
        )
        gates = smem.allocate_tensor(
            Float32, cute.make_layout((self.tokens, 8), stride=(8, 1)), byte_alignment=16
        )
        raw_gates = smem.allocate_tensor(
            Float32, cute.make_layout((self.tokens, 2), stride=(2, 1)), byte_alignment=16
        )
        values = smem.allocate_tensor(Float32, cute.make_layout(128), byte_alignment=16)
        if cutlass.const_expr(self.speculative_ring):
            physical_gates = smem.allocate_tensor(
                Float32, cute.make_layout((self.window + 3, 2), stride=(2, 1)), byte_alignment=16
            )
        slot = Int64(-1)
        if cutlass.const_expr(self.batch > 0):
            if cutlass.const_expr(self.dense_indices):
                slot = Int64(slots[b])
            else:
                slot = Int64(slots[Int64(b) * self.index_stride])
        readable = (slot >= 0) & (slot < self.slots)
        pool_slot = slot
        if cutlass.const_expr(self.wide_pool):
            pool_slot = Int64(slot)
        # One wide base-pointer calculation per head; all per-thread matrix
        # offsets below are bounded by 128*128 and stay in 32-bit registers.
        if cutlass.const_expr(self.dense_checkpoint):
            state_head = cute.make_tensor(
                checkpoint.iterator + (pool_slot * self.hv + h) * 16384 + tile * self.tile * 128,
                cute.make_layout(16384),
            )
        else:
            state_head = cute.make_tensor(
                checkpoint.iterator
                + pool_slot * self.cp_stride
                + h * 16384
                + tile * self.tile * 128,
                cute.make_layout(16384),
            )
        row0 = tid // self.lanes
        kh = h // (self.hv // self.hk)
        matrix = cute.make_rmem_tensor(
            cute.make_layout((self.rows, self.chunks, 4), stride=(self.chunks * 4, 4, 1)),
            Float32,
        )
        if cutlass.const_expr(self.immediate):
            if readable & (tid < self.replay_threads):
                for r in cutlass.range_constexpr(self.rows):
                    row = row0 + r * (self.threads // self.lanes)
                    for j in cutlass.range_constexpr(self.chunks):
                        vector_load(
                            state_head,
                            row * 128 + (tid % self.lanes) * 4 + j * self.lanes * 4,
                            matrix[r, j, None],
                        )
        # The control warp and checkpoint copy engine progress independently.
        # L<=8 fits a complete tile plus token scratch in opt-in shared memory.
        # Larger windows retain register loads to stay below the shared limit.
        if readable:
            if cutlass.const_expr(self.prefetch):
                llvm.inline_asm(
                    None,
                    [(state_head.iterator + tid * 4).toint().ir_value()],
                    "prefetch.global.L2 [$0];",
                    "l",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            if cutlass.const_expr(not self.immediate and self.window <= 8):
                atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=128
                )
                for i in cutlass.range_constexpr(self.tile * 128 // (self.threads * 4)):
                    offset = (tid + i * self.threads) * 4
                    src = state_vector(state_head.iterator, offset, cute.AddressSpace.gmem)
                    dst = state_vector(staged_state.iterator, offset, cute.AddressSpace.smem)
                    cute.copy(atom, src, dst)
                cute.arch.cp_async_commit_group()
        # Start physical ring DMA before metadata validation. Only the slot bounds
        # control these reads; stale physical rows never enter the arithmetic.
        if cutlass.const_expr(self.speculative_ring):
            if readable:
                for i in range(tid, (self.window + 3) * (self.cols // 8), self.threads):
                    rr, c = i // (self.cols // 8), (i % (self.cols // 8)) * 8
                    col = self.column(c, kh, h, tile)
                    base = (pool_slot * (self.window + 3) + rr) * self.width + col
                    if cutlass.const_expr(self.width % 8 == 0):
                        src_ptr = cute.make_ptr(
                            BFloat16,
                            (ring.iterator + base).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        dst_ptr = cute.make_ptr(
                            BFloat16,
                            (raw.iterator + rr * self.cols + c).toint(),
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        )
                        atom = cute.make_copy_atom(
                            cute.nvgpu.cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128
                        )
                        cute.copy(
                            atom,
                            cute.make_tensor(src_ptr, cute.make_layout(8)),
                            cute.make_tensor(dst_ptr, cute.make_layout(8)),
                        )
                    else:
                        for j in cutlass.range_constexpr(8):
                            raw[rr, c + j] = ring[base + j]
                cute.arch.cp_async_commit_group()
                # Keep the validation warp free of a dependent gate load/store.
                for i in range(tid, 32 + (self.window + 3) * 2, self.threads):
                    if i >= 32:
                        rr, gate = (i - 32) // 2, (i - 32) % 2
                        base = (pool_slot * (self.window + 3) + rr) * self.width
                        physical_gates[rr, gate] = Float32(
                            ring[base + self.dc + gate * self.hv + h]
                        )
        row_pos = Int64(0)
        if cutlass.const_expr(self.trusted):
            error = Int32(0)
            if readable & (tid == 0):
                fields = cute.make_rmem_tensor((4,), Int64)
                vector_load(metadata, Int64(slot) * 8, fields)
                row_pos = fields[1]
                control[1], control[2] = Int32(fields[2]), Int32(fields[3])
        elif cutlass.const_expr(self.snapshot_controls):
            error = Int32(frozen[0])
        else:
            if tid < 32:
                error = self.validate(
                    slots,
                    positions,
                    metadata,
                    epoch,
                    record,
                    sequence,
                    poison,
                    lane,
                    control,
                    bid,
                    False,
                )
                error = cute.arch.warp_redux_sync(error, "or")
                if lane == 0:
                    control[0] = error
        if cutlass.const_expr(not self.immediate and self.window > 8):
            if readable & (tid < self.replay_threads):
                for r in cutlass.range_constexpr(self.rows):
                    row = row0 + r * (self.threads // self.lanes)
                    for j in cutlass.range_constexpr(self.chunks):
                        vector_load(
                            state_head,
                            row * 128 + (tid % self.lanes) * 4 + j * self.lanes * 4,
                            matrix[r, j, None],
                        )
        # Current input and output gate do not depend on metadata. Put the current
        # raw row at a fixed shared offset, distinct from the staged ring rows.
        pack = cute.make_rmem_tensor((8,), BFloat16)
        if cutlass.const_expr(self.batch > 0):
            for c8 in range(tid, self.cols // 8, self.threads):
                col = self.column(c8 * 8, kh, h, tile)
                if cutlass.const_expr(self.aligned_x):
                    vector_load(x, b * self.stride + col, pack)
                else:
                    self.load_projected(x, b * self.stride + col, pack)
                for j in cutlass.range_constexpr(8):
                    raw[self.current_row, c8 * 8 + j] = pack[j]
        conv_weights = smem.allocate_tensor(
            BFloat16, cute.make_layout((self.cols, 4), stride=(4, 1)), byte_alignment=16
        )
        for c2 in range(tid, self.cols // 2, self.threads):
            col = self.column(c2 * 2, kh, h, tile)
            vector_load(cw, col * 4, pack)
            for j in cutlass.range_constexpr(8):
                conv_weights[c2 * 2 + j // 4, j % 4] = pack[j]
        output_gate = smem.allocate_tensor(BFloat16, cute.make_layout(self.tile), byte_alignment=16)
        if cutlass.const_expr(self.batch > 0):
            if tid < self.tile // 8:
                if cutlass.const_expr(self.aligned_x):
                    vector_load(
                        x, b * self.stride + self.dc + h * 128 + tile * self.tile + tid * 8, pack
                    )
                else:
                    self.load_projected(
                        x, b * self.stride + self.dc + h * 128 + tile * self.tile + tid * 8, pack
                    )
                for j in cutlass.range_constexpr(8):
                    output_gate[tid * 8 + j] = pack[j]
        if cutlass.const_expr(self.immediate):
            if readable:
                # Only the three authoritative history seeds are staged. All CTAs
                # finish their own reads before elected publication rolls them.
                for i in range(tid, 3 * (self.cols // 8), self.threads):
                    t, c = i // (self.cols // 8), (i % (self.cols // 8)) * 8
                    col = self.column(c, kh, h, tile)
                    if cutlass.const_expr(self.dense_history):
                        vector_load(history, (pool_slot * 3 + t) * self.dc + col, pack)
                    else:
                        vector_load(history, pool_slot * self.h_stride + t * self.dc + col, pack)
                    for j in cutlass.range_constexpr(8):
                        raw[t, c + j] = pack[j]
        if cutlass.const_expr(not self.snapshot_controls):
            cute.arch.sync_threads()
            if cutlass.const_expr(not self.trusted):
                error = control[0]
        if cutlass.const_expr(not self.immediate and not self.snapshot_controls):
            if readable & (error == 0) & (control[2] == 0):
                if cutlass.const_expr(self.speculative_ring):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                # An empty ring has no authoritative seeds. Stage only this CTA's
                # columns, and finish all history reads before the reader election.
                cursor0 = control[1]
                for i in range(tid, 3 * (self.cols // 8), self.threads):
                    t, c = i // (self.cols // 8), (i % (self.cols // 8)) * 8
                    col = self.column(c, kh, h, tile)
                    rr = (cursor0 + t + self.window) % (self.window + 3)
                    if cutlass.const_expr(self.dense_history):
                        vector_load(history, (pool_slot * 3 + t) * self.dc + col, pack)
                    else:
                        vector_load(history, pool_slot * self.h_stride + t * self.dc + col, pack)
                    staged = t
                    if cutlass.const_expr(self.speculative_ring):
                        staged = rr
                    for j in cutlass.range_constexpr(8):
                        raw[staged, c + j] = pack[j]
                    owns = (c >= 256) | ((tile == 0) & (h % (self.hv // self.hk) == 0))
                    if owns:
                        base = (pool_slot * (self.window + 3) + rr) * self.width + col
                        if cutlass.const_expr(self.width % 8 == 0):
                            vector_store(ring, base, pack)
                        else:
                            for j in cutlass.range_constexpr(8):
                                ring[base + j] = pack[j]
                # Seed gate columns are never read (raw gates come from records).
                cute.arch.sync_threads()
        if cutlass.const_expr(not self.snapshot_controls):
            # All controls were consumed by validation and its CTA barrier.
            # Elect before mutable history/metadata writes; empty-ring seed
            # stores above touch only dead rows. control[3] is consumed after
            # the common epilogue barrier.
            if cutlass.const_expr(self.trusted):
                # Each CTA arrives only after its metadata and history seed reads
                # join locally. The last arriver therefore cannot publish before
                # a late reader of this row finishes. No CTA consumes peer state
                # stores; stream completion joins those disjoint stores.
                if readable & (tid == 0):
                    control[3] = arrive(counters.iterator + b)
            else:
                if tid == 0:
                    control[3] = arrive(counters.iterator)
        if cutlass.const_expr(self.speculative_ring):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        self.stamp(debug, b, h, tid, 1)
        active = readable & (error == 0)
        p, cursor, n = Int64(0), Int32(0), Int32(0)
        if active:
            if cutlass.const_expr(self.immediate):
                if cutlass.const_expr(self.snapshot_controls):
                    p = frozen[1 + b * 2]
                if tid < 2:
                    raw_gates[0, tid] = Float32(
                        x[b * self.stride + self.dc + self.hv * 128 + tid * self.hv + h]
                    )
                cute.arch.sync_threads()
                self.stamp(debug, b, h, tid, 2)
                for c in range(tid, self.cols, self.threads):
                    v = self.convolve(raw, conv_weights, 0, c, 0)
                    sig = sigmoid(v)
                    if c < 256:
                        features[0, c] = mul_rn(v, sig)
                    else:
                        features[0, c], features[0, c + self.tile] = v, sig
                cute.arch.sync_threads()
                if warp == 0:
                    q = cute.make_rmem_tensor((4,), Float32)
                    k = cute.make_rmem_tensor((4,), Float32)
                    for j in cutlass.range_constexpr(4):
                        q[j], k[j] = features[0, lane + j * 32], features[0, 128 + lane + j * 32]
                    self.prepare_gates(q, k, raw_gates, gates, al, bias, 0, h, lane)
                cute.arch.sync_threads()
                self.stamp(debug, b, h, tid, 3)
                self.replay_token(
                    matrix,
                    features,
                    gates,
                    values,
                    0,
                    tid,
                    state_head,
                    True,
                    True,
                    row_base=tile * self.tile,
                )
                self.stamp(debug, b, h, tid, 4)
            else:
                if cutlass.const_expr(self.snapshot_controls):
                    p = frozen[1 + b * 2]
                    cursor = Int32(frozen[2 + b * 2])
                    if cutlass.const_expr(self.flush_always):
                        n = cursor // (self.window + 3)
                        cursor = cursor % (self.window + 3)
                    else:
                        m = p - load_checkpoint_position(metadata.iterator + Int64(slot) * 8)
                        # The ordered snapshot validated p-c in [0,L). During compute,
                        # only head 0 can replace c with p+1, and only on flush. Thus a
                        # reader sees either L-1 or -1 on flush; both map to L-1. Every
                        # head owns its state tile, so an early c store cannot change a
                        # late head's checkpoint data. Ring cursor and p are immutable
                        # snapshots, append overwrites the one row no reader consumes,
                        # and history seeds were prepared by the ordered snapshot.
                        # Stream completion joins all state stores before any next op.
                        n = Int32(m + self.window if m < 0 else m)
                else:
                    cursor, n = control[1], control[2]
                if cutlass.const_expr(self.sections and self.window > 8):
                    # Thread 0 consumes every state vector through a volatile shared
                    # store before its timestamp. Empty asm operands would disappear
                    # in PTX and would not force the hardware load scoreboard to wait.
                    # This diagnostic dependency (16/32 vector stores) is absent off.
                    if tid == 0:
                        for r in cutlass.range_constexpr(self.rows):
                            for j in cutlass.range_constexpr(self.chunks):
                                llvm.inline_asm(
                                    None,
                                    [probe.iterator.toint().ir_value()]
                                    + [matrix[r, j, c].ir_value() for c in range(4)],
                                    "st.volatile.shared.v4.f32 [$0], {$1, $2, $3, $4};",
                                    "l,f,f,f,f,~{memory}",
                                    has_side_effects=True,
                                    is_align_stack=False,
                                    asm_dialect=llvm.AsmDialect.AD_ATT,
                                )
                self.stamp(debug, b, h, tid, 2)
                # Stage n history records plus three seeds and the current raw row.
                # Ring rows are 16-byte aligned on production head geometries. Tiny
                # contract-test geometries can have unaligned ring strides.
                if cutlass.const_expr(not self.speculative_ring):
                    if (n > 0) | self.snapshot_controls:
                        for i in range(tid, (n + 3) * (self.cols // 8), self.threads):
                            t, c = i // (self.cols // 8), (i % (self.cols // 8)) * 8
                            col = self.column(c, kh, h, tile)
                            rr = (cursor + t - n - 3 + 2 * (self.window + 3)) % (self.window + 3)
                            base = (pool_slot * (self.window + 3) + rr) * self.width + col
                            if cutlass.const_expr(self.width % 8 == 0):
                                vector_load(ring, base, pack)
                            else:
                                for j in cutlass.range_constexpr(8):
                                    pack[j] = ring[base + j]
                            for j in cutlass.range_constexpr(8):
                                raw[t, c + j] = pack[j]
                # Stage raw beta/a in aligned eight-head vectors. Each CTA needs only
                # its head; tiny geometries use naturally aligned scalar loads.
                for t in range(tid, n + 1, self.threads):
                    for j in cutlass.range_constexpr(2):
                        if t == n:
                            base = b * self.stride + self.dc + self.hv * 128 + j * self.hv
                            if cutlass.const_expr(self.hv % 8 == 0):
                                if cutlass.const_expr(self.aligned_x):
                                    vector_load(x, base + (h // 8) * 8, pack)
                                else:
                                    self.load_projected(x, base + (h // 8) * 8, pack)
                                chosen = Float32(0)
                                for u in cutlass.range_constexpr(8):
                                    if h % 8 == u:
                                        chosen = Float32(pack[u])
                                raw_gates[t, j] = chosen
                            else:
                                raw_gates[t, j] = Float32(x[base + h])
                        else:
                            if cutlass.const_expr(self.speculative_ring):
                                rr = self.ring_row(t + 3, cursor, n)
                                raw_gates[t, j] = physical_gates[rr, j]
                            else:
                                rr = (cursor + t - n + self.window + 3) % (self.window + 3)
                                base = (
                                    (pool_slot * (self.window + 3) + rr) * self.width
                                    + self.dc
                                    + j * self.hv
                                )
                                if cutlass.const_expr(self.hv % 8 == 0):
                                    vector_load(ring, base + (h // 8) * 8, pack)
                                    chosen = Float32(0)
                                    for u in cutlass.range_constexpr(8):
                                        if h % 8 == u:
                                            chosen = Float32(pack[u])
                                    raw_gates[t, j] = chosen
                                else:
                                    raw_gates[t, j] = Float32(ring[base + h])
                cute.arch.sync_threads()
                if n == 0:
                    # Use the entire CTA for the single token's convolution.
                    for c in range(tid, self.cols, self.threads):
                        v = self.convolve(raw, conv_weights, 0, c, n, cursor)
                        sig = sigmoid(v)
                        if c < 256:
                            features[0, c] = mul_rn(v, sig)
                        else:
                            features[0, c], features[0, c + self.tile] = v, sig
                    cute.arch.sync_threads()
                    if warp == 0:
                        q = cute.make_rmem_tensor((4,), Float32)
                        k = cute.make_rmem_tensor((4,), Float32)
                        for j in cutlass.range_constexpr(4):
                            q[j], k[j] = (
                                features[0, lane + j * 32],
                                features[0, 128 + lane + j * 32],
                            )
                        self.prepare_gates(q, k, raw_gates, gates, al, bias, 0, h, lane)
                else:
                    # L<=warps: one token per warp, no serial token loop. Larger
                    # supported windows use independent warp-strided waves. q/k
                    # stay in registers through normalization; only replay consumes
                    # the shared features. No intermediate CTA barrier is needed.
                    for t in range(warp, n + 1, self.warps):
                        q = cute.make_rmem_tensor((4,), Float32)
                        k = cute.make_rmem_tensor((4,), Float32)
                        for j in cutlass.range_constexpr(4):
                            c = lane + j * 32
                            qraw = self.convolve(raw, conv_weights, t, c, n, cursor)
                            kraw = self.convolve(raw, conv_weights, t, 128 + c, n, cursor)
                            q[j] = mul_rn(qraw, sigmoid(qraw))
                            k[j] = mul_rn(kraw, sigmoid(kraw))
                            features[t, c], features[t, 128 + c] = q[j], k[j]
                            if c < self.tile:
                                v = self.convolve(raw, conv_weights, t, 256 + c, n, cursor)
                                features[t, 256 + c] = v
                                features[t, 256 + self.tile + c] = sigmoid(v)
                        self.prepare_gates(q, k, raw_gates, gates, al, bias, t, h, lane)
                if cutlass.const_expr(not self.immediate and self.window <= 8):
                    # No checkpoint consumer until every token's prologue is ready.
                    # All threads wait for their own async copies, then the CTA join
                    # makes the complete tile available to its register layout.
                    cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if cutlass.const_expr(not self.immediate and self.window <= 8):
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                    )
                    if tid < self.replay_threads:
                        for r in cutlass.range_constexpr(self.rows):
                            row = row0 + r * (self.threads // self.lanes)
                            for j in cutlass.range_constexpr(self.chunks):
                                offset = row * 128 + (tid % self.lanes) * 4 + j * self.lanes * 4
                                src = state_vector(
                                    staged_state.iterator, offset, cute.AddressSpace.smem
                                )
                                cute.copy(atom, src, matrix[r, j, None])
                self.stamp(debug, b, h, tid, 3)
                # The historical loop has no final-token branch or output dot.
                for t in range(n):
                    self.replay_token(
                        matrix, features, gates, values, t, tid, state_head, False, False
                    )
                self.replay_token(
                    matrix,
                    features,
                    gates,
                    values,
                    n,
                    tid,
                    state_head,
                    (n + 1 == self.window) | self.flush_always,
                    True,
                    row_base=tile * self.tile,
                )
                self.stamp(debug, b, h, tid, 4)
                # Unique owners append raw channels and maintain canonical convolution
                # history on EVERY step (the recurrence checkpoint alone is delayed).
                for c8 in range(tid, self.cols // 8, self.threads):
                    c = c8 * 8
                    owns = (c >= 256) | ((tile == 0) & (h % (self.hv // self.hk) == 0))
                    if owns:
                        col = self.column(c, kh, h, tile)
                        for j in cutlass.range_constexpr(8):
                            pack[j] = raw[self.current_row, c + j]
                        base = (pool_slot * (self.window + 3) + cursor) * self.width + col
                        if cutlass.const_expr(self.width % 8 == 0):
                            vector_store(ring, base, pack)
                        else:
                            for j in cutlass.range_constexpr(8):
                                ring[base + j] = pack[j]
                        if (n > 0) | self.snapshot_controls:
                            for j in cutlass.range_constexpr(3):
                                rr = n + 1 + j
                                if cutlass.const_expr(self.speculative_ring):
                                    rr = self.ring_row(n + 1 + j, cursor, n)
                                if j == 2:
                                    rr = Int32(self.current_row)
                                for u in cutlass.range_constexpr(8):
                                    pack[u] = raw[rr, c + u]
                                if cutlass.const_expr(self.dense_history):
                                    vector_store(history, (pool_slot * 3 + j) * self.dc + col, pack)
                                else:
                                    vector_store(
                                        history, pool_slot * self.h_stride + j * self.dc + col, pack
                                    )
                if (tid == 0) & (tile == 0):
                    base = (pool_slot * (self.window + 3) + cursor) * self.width + self.dc
                    ring[base + h] = x[b * self.stride + self.dc + self.hv * 128 + h]
                    ring[base + self.hv + h] = x[
                        b * self.stride + self.dc + self.hv * 128 + self.hv + h
                    ]
        else:
            if cutlass.const_expr(
                (not self.immediate and self.window <= 8) or self.speculative_ring
            ):
                cute.arch.cp_async_wait_group(0)
            self.stamp(debug, b, h, tid, 2)
            self.stamp(debug, b, h, tid, 3)
            self.stamp(debug, b, h, tid, 4)
            if cutlass.const_expr(self.batch > 0):
                if tid < self.tile:
                    if cutlass.const_expr(self.dense_out):
                        out[(b * self.hv + h) * 128 + tile * self.tile + tid] = BFloat16(0)
                    else:
                        out[Int64(b) * self.out_stride + h * 128 + tile * self.tile + tid] = (
                            BFloat16(0)
                        )
        if cutlass.const_expr(self.trusted):
            cute.arch.sync_threads()
            if readable:
                if control[3] == self.hv * self.nv - 1:
                    if tid == 0:
                        metadata[Int64(slot) * 8 + 1] = row_pos + 1
                        if cutlass.const_expr(self.immediate):
                            metadata[Int64(slot) * 8] = row_pos + 1
                        else:
                            metadata[Int64(slot) * 8 + 2] = (cursor + 1) % (self.window + 3)
                            metadata[Int64(slot) * 8 + 3] = (
                                Int64(0)
                                if (n + 1 == self.window) | self.flush_always
                                else Int64(n + 1)
                            )
                            if (n + 1 == self.window) | self.flush_always:
                                metadata[Int64(slot) * 8] = row_pos + 1
                        if cutlass.const_expr(self.advance):
                            if cutlass.const_expr(self.dense_positions):
                                positions[b] = row_pos + 1
                            else:
                                positions[Int64(b) * self.position_stride] = row_pos + 1
                        counters[b] = Int32(0)
                    if self.immediate | (n == 0):
                        self.roll_row_history(x, history, b, Int64(slot), tid)
        elif cutlass.const_expr(self.snapshot_controls):
            # No peer reads the fields below in compute; p/cursor/error came
            # from the ordered snapshot. Only c has readers (negative-m rule).
            if active & (h == 0) & (tid == 0):
                metadata[Int64(slot) * 8 + 1] = p + 1
                if cutlass.const_expr(self.immediate):
                    store_checkpoint_position(metadata.iterator + Int64(slot) * 8, p + 1)
                else:
                    metadata[Int64(slot) * 8 + 2] = (cursor + 1) % (self.window + 3)
                    metadata[Int64(slot) * 8 + 3] = (
                        Int64(0) if (n + 1 == self.window) | self.flush_always else Int64(n + 1)
                    )
                    if (n + 1 == self.window) | self.flush_always:
                        store_checkpoint_position(metadata.iterator + Int64(slot) * 8, p + 1)
                if cutlass.const_expr(self.advance):
                    if cutlass.const_expr(self.dense_positions):
                        positions[b] = p + 1
                    else:
                        positions[Int64(b) * self.position_stride] = p + 1
            publish = (bid == 0) & (h == 0)
        else:
            # Publish shared output values and the early reader-election result.
            # There is no GPU fence here: no CTA consumes another's state stores.
            cute.arch.sync_threads()
            publish = control[3] == max(1, self.batch * self.hv * self.nv) - 1
            if publish:
                if cutlass.const_expr(not self.immediate):
                    if error == 0:
                        for row in range(self.batch):
                            if cutlass.const_expr(self.dense_indices):
                                s = Int64(slots[row])
                            else:
                                s = Int64(slots[Int64(row) * self.index_stride])
                            if s >= 0:
                                if metadata[s * 8 + 3] == 0:
                                    self.roll_row_history(x, history, row, s, tid)
                        cute.arch.sync_threads()
                for i in range(tid, self.batch, self.threads):
                    if cutlass.const_expr(self.dense_indices):
                        s = Int64(slots[i])
                    else:
                        s = Int64(slots[Int64(i) * self.index_stride])
                    if (error == 0) & (s >= 0):
                        pos = metadata[s * 8 + 1]
                        metadata[s * 8 + 1] = pos + 1
                        if cutlass.const_expr(self.immediate):
                            metadata[s * 8] = pos + 1
                        else:
                            count = metadata[s * 8 + 3]
                            metadata[s * 8 + 2] = (metadata[s * 8 + 2] + 1) % (self.window + 3)
                            metadata[s * 8 + 3] = (
                                Int64(0)
                                if (count + 1 == self.window) | self.flush_always
                                else count + 1
                            )
                            if (count + 1 == self.window) | self.flush_always:
                                metadata[s * 8] = pos + 1
                        if cutlass.const_expr(self.advance):
                            if cutlass.const_expr(self.dense_positions):
                                positions[i] = pos + 1
                            else:
                                positions[Int64(i) * self.position_stride] = pos + 1
        if cutlass.const_expr(not self.trusted):
            if publish:
                if cutlass.const_expr(self.immediate and not self.snapshot_controls):
                    if error == 0:
                        self.roll_history(x, history, slots, tid)
                if cutlass.const_expr(self.complete):
                    if error & 64 == 0:
                        for i in range(tid, cute.size(valid), self.threads):
                            valid[i] = False
                # Publication is consumed only after engine stream completion. No
                # status reader in this grid needs a release fence or completion join.
                if tid == 0:
                    if cutlass.const_expr(not self.snapshot_controls):
                        counters[0] = Int32(0)
                    words = cute.recast_tensor(record, Int32)
                    if error != 0:
                        llvm.inline_asm(
                            T.i32(),
                            [poison.iterator.toint().ir_value(), error.ir_value()],
                            "atom.relaxed.gpu.global.or.b32 $0, [$1], $2;",
                            "=r,l,r,~{memory}",
                            has_side_effects=True,
                            is_align_stack=False,
                            asm_dialect=llvm.AsmDialect.AD_ATT,
                        )
                    # A colliding invocation never overwrites an unsettled record.
                    if error & 64 == 0:
                        completion_code = error
                        if cutlass.const_expr(self.complete):
                            if poison[0] != 0:
                                completion_code = completion_code | Int32(32)
                        words[2] = completion_code
                        if cutlass.const_expr(self.complete):
                            cute.recast_tensor(record, Int64)[0] = sequence[0]
                            words[3] = Int32(1)
        self.stamp(debug, b, h, tid, 5)
        if cutlass.const_expr(self.clustered):
            # Join all CTAs before any remote access, even inactive/error CTAs.
            # Each producer broadcasts disjoint rows into every peer's values.
            cluster_sync()
            if active & (tid < self.tile):
                row = tile * self.tile + tid
                value = values[row]
                for peer in cutlass.range_constexpr(self.nv):
                    if tile != peer:
                        cluster_store(values.iterator + row, Int32(peer), value)
            cluster_sync()
        if active:
            if cutlass.const_expr(self.snapshot_controls):
                cute.arch.sync_threads()
            # Each warp forms the same 128-row reduction. One CTA barrier replaces
            # the global head counter/load and the seven shared reduction barriers.
            ov = cute.make_rmem_tensor((4,), Float32)
            for j in cutlass.range_constexpr(4):
                ov[j] = values[lane + j * 32]
            total = self.output_sum(ov)
            if tid < self.tile:
                row = tile * self.tile + tid
                z, weight = Float32(output_gate[tid]), Float32(nw[row])
                rho = cute.math.rsqrt(total / 128 + 1e-6, fastmath=True)
                y = (weight * z * sigmoid(z)) * (values[row] * rho)
                if cutlass.const_expr(self.dense_out):
                    out[(b * self.hv + h) * 128 + row] = y.to(BFloat16)
                else:
                    out[Int64(b) * self.out_stride + h * 128 + row] = y.to(BFloat16)
        self.stamp(debug, b, h, tid, 6)
        if cutlass.const_expr(self.clustered):
            # Keep every shared allocation alive until all peers finish.
            cluster_sync()

    @cute.jit
    def roll_row_history(self, x, history, b, slot, tid):
        # Callers pass int32 batch rows (the strict delayed publisher loops on
        # an int32 index outside the main kernel's widening guard).
        b = Int64(b)
        pack0 = cute.make_rmem_tensor((8,), BFloat16)
        pack1 = cute.make_rmem_tensor((8,), BFloat16)
        pack2 = cute.make_rmem_tensor((8,), BFloat16)
        for c8 in range(tid, self.dc // 8, self.threads):
            c = c8 * 8
            if cutlass.const_expr(self.dense_history):
                vector_load(history, (slot * 3 + 1) * self.dc + c, pack0)
            else:
                vector_load(history, slot * self.h_stride + 1 * self.dc + c, pack0)
            if cutlass.const_expr(self.dense_history):
                vector_load(history, (slot * 3 + 2) * self.dc + c, pack1)
            else:
                vector_load(history, slot * self.h_stride + 2 * self.dc + c, pack1)
            if cutlass.const_expr(self.aligned_x):
                vector_load(x, b * self.stride + c, pack2)
            else:
                self.load_projected(x, b * self.stride + c, pack2)
            if cutlass.const_expr(self.dense_history):
                vector_store(history, slot * 3 * self.dc + c, pack0)
            else:
                vector_store(history, slot * self.h_stride + c, pack0)
            if cutlass.const_expr(self.dense_history):
                vector_store(history, (slot * 3 + 1) * self.dc + c, pack1)
            else:
                vector_store(history, slot * self.h_stride + 1 * self.dc + c, pack1)
            if cutlass.const_expr(self.dense_history):
                vector_store(history, (slot * 3 + 2) * self.dc + c, pack2)
            else:
                vector_store(history, slot * self.h_stride + 2 * self.dc + c, pack2)

    @cute.jit
    def roll_history(self, x, history, slots, tid):
        # The last reader CTA owns publication for all rows. Every seed load
        # precedes the acq_rel election, so shared q/k channels cannot race.
        pack0 = cute.make_rmem_tensor((8,), BFloat16)
        pack1 = cute.make_rmem_tensor((8,), BFloat16)
        pack2 = cute.make_rmem_tensor((8,), BFloat16)
        for i in range(tid, self.batch * (self.dc // 8), self.threads):
            b, c = i // (self.dc // 8), (i % (self.dc // 8)) * 8
            if cutlass.const_expr(self.dense_batch_offsets):
                if cutlass.const_expr(self.batch * self.stride >= 2**31):
                    b = Int64(b)
            else:
                if cutlass.const_expr(
                    self.batch
                    * max(self.stride, self.out_stride, self.index_stride, self.position_stride)
                    >= 2**31
                ):
                    b = Int64(b)
            if cutlass.const_expr(self.dense_indices):
                slot = Int64(slots[b])
            else:
                slot = Int64(slots[Int64(b) * self.index_stride])
            if slot >= 0:
                if cutlass.const_expr(self.dense_history):
                    vector_load(history, (slot * 3 + 1) * self.dc + c, pack0)
                else:
                    vector_load(history, slot * self.h_stride + 1 * self.dc + c, pack0)
                if cutlass.const_expr(self.dense_history):
                    vector_load(history, (slot * 3 + 2) * self.dc + c, pack1)
                else:
                    vector_load(history, slot * self.h_stride + 2 * self.dc + c, pack1)
                if cutlass.const_expr(self.aligned_x):
                    vector_load(x, b * self.stride + c, pack2)
                else:
                    self.load_projected(x, b * self.stride + c, pack2)
                if cutlass.const_expr(self.dense_history):
                    vector_store(history, slot * 3 * self.dc + c, pack0)
                else:
                    vector_store(history, slot * self.h_stride + c, pack0)
                if cutlass.const_expr(self.dense_history):
                    vector_store(history, (slot * 3 + 1) * self.dc + c, pack1)
                else:
                    vector_store(history, slot * self.h_stride + 1 * self.dc + c, pack1)
                if cutlass.const_expr(self.dense_history):
                    vector_store(history, (slot * 3 + 2) * self.dc + c, pack2)
                else:
                    vector_store(history, slot * self.h_stride + 2 * self.dc + c, pack2)

    @cute.kernel
    def snapshot_history(self, x, history, slots, frozen):
        tid, _, _ = cute.arch.thread_idx()
        if frozen[0] == 0:
            self.roll_history(x, history, slots, tid)


class RingEmpty:
    """One warp reduces active slot metadata; no atomics or host synchronization."""

    def __init__(self, batch, slots, index_stride=1):
        self.batch, self.slots = batch, slots
        self.index_stride = index_stride
        self.dense_indices = index_stride == 1

    @cute.jit
    def __call__(
        self, metadata: cute.Tensor, slots: cute.Tensor, flag: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(metadata, slots, flag).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, metadata, slots, flag):
        lane, _, _ = cute.arch.thread_idx()
        bad = Int32(0)
        for b in range(lane, self.batch, 32):
            if cutlass.const_expr(self.dense_indices):
                slot = Int64(slots[b])
            else:
                slot = Int64(slots[Int64(b) * self.index_stride])
            if slot != -1:
                if (slot < 0) | (slot >= self.slots):
                    bad = Int32(1)
                else:
                    if (metadata[slot * 8 + 3] != 0) | (
                        metadata[slot * 8] != metadata[slot * 8 + 1]
                    ):
                        bad = Int32(1)
        bad = cute.arch.warp_redux_sync(bad, "or")
        if lane == 0:
            flag[0] = bad == 0


class AdvancePositions:
    """Trusted #25-route metadata: one warp, per-row; -1 and out-of-range rows are masked.

    Ringless slots keep n=0 and cursor=0, so checkpoint and committed positions
    both become committed+1, exactly as the trusted immediate CuTe step publishes.
    """

    def __init__(self, batch, slots, index_stride=1):
        self.batch, self.slots = batch, slots
        self.index_stride = index_stride
        self.dense_indices = index_stride == 1

    @cute.jit
    def __call__(self, metadata: cute.Tensor, slots: cute.Tensor, stream: cuda.CUstream):
        self.kernel(metadata, slots).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, metadata, slots):
        lane, _, _ = cute.arch.thread_idx()
        for b in range(lane, self.batch, 32):
            if cutlass.const_expr(self.dense_indices):
                slot = Int64(slots[b])
            else:
                slot = Int64(slots[Int64(b) * self.index_stride])
            if (slot >= 0) & (slot < self.slots):
                pos = metadata[slot * 8 + 1] + 1
                metadata[slot * 8 + 1] = pos
                metadata[slot * 8] = pos
