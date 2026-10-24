# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordered replay phases with the unchanged product arithmetic from Step.

Strict mode validates the whole batch in one control CTA before compute; a separate
publication launch owns descriptors and raw history. Trusted mode publishes verify
rows while staging inputs and elects the last compute CTA per row for other phases. Compute splits each head into value-row
tiles held in registers and stages at most ``warps`` input tokens. Output values
are retained for up to 32 tokens, then broadcast in blocks across the CTA cluster
before the unchanged full-head normalization.
"""

import math
from dataclasses import replace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T
from cutlass.utils import SmemAllocator

from ._kernel import Step, arrive, mul_rn, sigmoid, vector_load, vector_store


@cute.jit
def cluster_sync():
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()


@cute.jit
def cluster_store(pointer, peer, value):
    remote = cute.arch.map_dsmem_ptr(pointer, peer)
    llvm.inline_asm(
        None,
        [remote.toint().ir_value(), value.ir_value()],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def status_or(pointer, value):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [pointer.toint().ir_value(), value.ir_value()],
            "atom.relaxed.gpu.global.or.b32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


OUTPUT_RESIDENT = 32


class Phases(Step):
    def __init__(
        self,
        hk,
        hv,
        window,
        batch,
        stride,
        config,
        slots,
        arithmetic,
        mode,
        q,
        ns,
        pb,
        capture,
        vec=False,
        tile=128,
        trusted=False,
        pdl=True,
        fused=True,
        delayed=False,
        layout=None,
        slot_layout=(),
    ):
        config = replace(config, value_tile=128, speculative_ring=False)
        super().__init__(
            hk, hv, window, batch, stride, config, slots, False, False, arithmetic, layout
        )
        self.slot_layout = slot_layout
        self.phase_features = True
        self.mode, self.q, self.ns, self.pb, self.capture = mode, q, ns, pb, capture
        self.proj = self.dc + self.hv * 128 + 2 * self.hv
        self.token_stride = self.layout.token or self.proj
        if layout is None:
            self.index_stride = 2 if mode == "export_snapshot" else 1
            self.position_stride = q if mode == "verify" else 1
            self.out_stride = (q if mode == "verify" else 1) * hv * 128
        self.out_token_stride = self.layout.out_token
        self.trusted, self.pdl = trusted, pdl
        # Launch bound: >1 caps registers so that CTAs co-reside on one SM.
        self.min_blocks = config.min_blocks
        self.fused = trusted and fused and mode != "reject"
        if delayed and not (mode == "commit" and trusted and fused):
            raise ValueError("delayed commit requires trusted fused commit")
        self.delayed = delayed
        self.vec = vec
        self.tile, self.nv = tile, 128 // tile
        self.replay_threads = min(self.threads, tile * self.lanes)
        self.rows = tile // (self.replay_threads // self.lanes)
        self.output_tokens = q if mode == "verify" else 1
        self.dense_indices = self.index_stride == (2 if mode == "export_snapshot" else 1) and (
            mode != "export_snapshot" or self.layout.pair == 1
        )
        self.dense_positions = self.position_stride == (q if mode == "verify" else 1)
        self.dense_token = self.token_stride == self.proj
        self.dense_out = (
            self.out_stride == self.output_tokens * hv * 128 and self.out_token_stride == hv * 128
        )
        self.dense_accepted = self.layout.accepted == 1
        self.dense_snapshots = self.layout.snapshots == 1
        self.dense_offsets = self.layout.offsets == 1
        self.dense_valid = self.layout.valid == 1
        self.chunked_output = self.output_tokens > OUTPUT_RESIDENT
        self.output_rows = (
            OUTPUT_RESIDENT + self.warps if self.chunked_output else self.output_tokens
        )
        if self.dense_token and self.out_token_stride == hv * 128:
            self.wide_tokens = (
                max(q * self.width, q * self.proj, q * hv * 128, 4 * (window + 3)) >= 2**31
            )
        else:
            self.wide_tokens = (
                max(
                    q * self.width,
                    q * self.token_stride,
                    q * self.out_token_stride,
                    4 * (window + 3),
                )
                >= 2**31
            )
        # Verify publication is a read-only copy of Q raw rows: spread it over
        # CTAs (about four 16-byte vectors per thread). Other modes own per-row
        # metadata and keep one CTA per row.
        self.copy_ctas = 1
        if mode == "verify" and vec:
            self.copy_ctas = min(65535, max(1, -(-(q * self.width // 8) // (128 * 4))))

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cw: cute.Tensor,
        al: cute.Tensor,
        bias: cute.Tensor,
        nw: cute.Tensor,
        indices: cute.Tensor,
        positions: cute.Tensor,
        checkpoint: cute.Tensor,
        history: cute.Tensor,
        ring: cute.Tensor,
        metadata: cute.Tensor,
        epoch: cute.Tensor,
        snapshots: cute.Tensor,
        pending: cute.Tensor,
        pm: cute.Tensor,
        accepted: cute.Tensor,
        snap_rows: cute.Tensor,
        offsets: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        record: cute.Tensor,
        sequence: cute.Tensor,
        poison: cute.Tensor,
        arrivals: cute.Tensor,
        stream: cuda.CUstream,
        status_slot: cute.Tensor = None,
    ):
        if cutlass.const_expr(not self.trusted):
            self.control(
                indices,
                positions,
                metadata,
                epoch,
                pm,
                accepted,
                snap_rows,
                offsets,
                record,
                sequence,
                poison,
                status_slot,
            ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        if cutlass.const_expr(self.batch > 0):
            if cutlass.const_expr(self.mode != "reject"):
                self.compute(
                    x,
                    cw,
                    al,
                    bias,
                    nw,
                    indices,
                    checkpoint,
                    ring,
                    metadata,
                    pending,
                    pm,
                    accepted,
                    snap_rows,
                    offsets,
                    out,
                    record,
                    poison,
                    history,
                    epoch,
                    snapshots,
                    valid,
                    arrivals,
                    status_slot,
                ).launch(
                    grid=(self.nv, self.hv, self.batch),
                    cluster=(self.nv, 1, 1) if cutlass.const_expr(self.nv > 1) else None,
                    block=(self.threads, 1, 1),
                    use_pdl=self.pdl,
                    min_blocks_per_mp=self.min_blocks,
                    stream=stream,
                )
            if cutlass.const_expr(not self.fused):
                self.publish_phase(
                    x,
                    indices,
                    history,
                    ring,
                    metadata,
                    epoch,
                    snapshots,
                    pending,
                    pm,
                    accepted,
                    snap_rows,
                    offsets,
                    valid,
                    record,
                    poison,
                    status_slot,
                ).launch(
                    grid=(self.batch, self.copy_ctas, 1),
                    block=(128, 1, 1),
                    stream=stream,
                    use_pdl=self.pdl,
                )

    def invocation_view(self, tensor, invocation, stride):
        """Slot ``invocation`` of a stacked argument; ``stride`` is in elements.

        Keeps the bound alignment wherever the slot stride preserves it, so the
        slotted specialization emits the same vector accesses as a slot view.
        """
        if cutlass.const_expr(stride == 0):
            # Unused stacked argument (no capture/arrivals): keep the bound view.
            return tensor
        offset = Int64(invocation) * Int64(stride)
        if cutlass.const_expr(tensor.element_type is Boolean):
            # Byte-aligned flags: plain iterator arithmetic keeps the bool memref type.
            return cute.make_tensor(tensor.iterator + offset, tensor.layout)
        size = max(1, tensor.element_type.width // 8)
        align = math.gcd(tensor.iterator.alignment, stride * size)
        ptr = cute.make_ptr(
            tensor.element_type,
            tensor.iterator.toint() + offset * size,
            tensor.iterator.memspace,
            assumed_align=max(align, size),
        )
        return cute.make_tensor(ptr, tensor.layout)

    def invocation(self, status_slot):
        """(valid, clamped slot): views always point inside the stacked buffers."""
        invocation = Int64(status_slot[0])
        valid = (invocation >= 0) & (invocation < self.slot_layout[0])
        return valid, invocation * Int64(valid)

    @cute.jit
    def source(self, indices, b):
        if cutlass.const_expr(self.mode == "export_snapshot"):
            if cutlass.const_expr(self.dense_indices):
                source, target = Int64(indices[b * 2]), Int64(indices[b * 2 + 1])
            else:
                source, target = (
                    Int64(indices[Int64(b) * self.index_stride]),
                    Int64(indices[Int64(b) * self.index_stride + self.layout.pair]),
                )
        else:
            if cutlass.const_expr(self.dense_indices):
                source = Int64(indices[b])
            else:
                source = Int64(indices[Int64(b) * self.index_stride])
            target = source
        return source, target

    @cute.jit
    def pending_row(self, pm, slot, b):
        row = Int64(b)
        if cutlass.const_expr(self.mode == "export_snapshot" and self.pb > 0):
            row = Int64(-1)
            for j in range(self.pb):
                if (pm[j * 8] == slot) & ((pm[j * 8 + 5] == 1) | (pm[j * 8 + 5] == 2)):
                    row = Int64(j)
        return row

    @cute.jit
    def row_active(self, slot, target, pb):
        active = (slot != -1) & (target != -1)
        if cutlass.const_expr(self.trusted):
            active = (slot >= 0) & (slot < self.slots)
            if cutlass.const_expr(self.mode == "export_snapshot"):
                active = active & (target >= self.slots) & (target < self.slots + self.ns)
            if cutlass.const_expr(
                self.mode in ("commit", "reject")
                or (self.mode == "export_snapshot" and self.pb > 0)
            ):
                active = active & (pb >= 0) & (pb < self.pb)
        return active

    @cute.jit
    def validate_row(
        self, indices, positions, metadata, epoch, pm, accepted, snap_rows, offsets, b
    ):
        # Widen before any batch-row stride multiply (snapshot/offset strides
        # are independent of B*Q, so no size guard can keep int32 safe).
        b = Int64(b)
        source, target = self.source(indices, b)
        error = Int32(0)
        if (source != -1) & (target != -1):
            if (source < 0) | (source >= self.slots):
                error = Int32(1)
            else:
                base = source * 8
                cp, pos = metadata[base], metadata[base + 1]
                cursor, n = metadata[base + 2], metadata[base + 3]
                good = (metadata[base + 4] == 1) & (metadata[base + 7] == epoch[0])
                good = good & (n >= 0) & (n < self.window) & (pos - cp == n)
                good = good & (cursor >= 0) & (cursor < self.window + 3)
                if not good:
                    error = error | Int32(1)
                if cutlass.const_expr(self.mode in ("bridge_step", "delayed_step")):
                    if cutlass.const_expr(self.dense_positions):
                        if (Int64(positions[b]) != pos) | (pos == Int64(9223372036854775807)):
                            error = error | Int32(2)
                    else:
                        if (Int64(positions[Int64(b) * self.position_stride]) != pos) | (
                            pos == Int64(9223372036854775807)
                        ):
                            error = error | Int32(2)
                if cutlass.const_expr(self.mode == "verify"):
                    for j in range(self.token_index(0), self.q):
                        if cutlass.const_expr(self.dense_positions):
                            if (Int64(positions[b * self.q + j]) != pos + j) | (
                                pos > Int64(9223372036854775807 - self.q)
                            ):
                                error = error | Int32(2)
                        else:
                            if (
                                Int64(positions[Int64(b) * self.position_stride + j]) != pos + j
                            ) | (pos > Int64(9223372036854775807 - self.q)):
                                error = error | Int32(2)
                found = Int32(1)
                if cutlass.const_expr(
                    self.mode in ("commit", "reject")
                    or (self.mode == "export_snapshot" and self.pb > 0)
                ):
                    pb = self.pending_row(pm, source, b)
                    if pb < 0:
                        # Portable validate returns immediately with ONLY descriptor
                        # error when its export search fails (even if live metadata
                        # was also invalid).
                        error, found = Int32(8), Int32(0)
                    else:
                        p = pb * 8
                        tx = pm[p + 5]
                        good = (pm[p] == source) & (pm[p + 1] == metadata[base + 5])
                        good = good & (pm[p + 2] == cp) & (pm[p + 3] == pos) & (pm[p + 4] == self.q)
                        good = good & ((tx == 1) | (tx == 2)) & (pm[p + 7] == epoch[0])
                        if not good:
                            error = error | Int32(8)
                        if cutlass.const_expr(self.mode != "reject"):
                            if cutlass.const_expr(self.dense_accepted):
                                m = Int64(accepted[pb])
                            else:
                                m = Int64(accepted[pb * self.layout.accepted])
                            if (m < 0) | (m > self.q):
                                error = error | Int32(4)
                            if (tx == 2) & (pm[p + 6] != m):
                                error = error | Int32(8)
                            positive = m if m >= 0 else Int64(0)
                            if pos > Int64(9223372036854775807) - positive:
                                error = error | Int32(2)
                if cutlass.const_expr(self.mode == "export_snapshot" or self.capture):
                    if found != 0:
                        snap = target
                        if cutlass.const_expr(self.capture):
                            if cutlass.const_expr(self.dense_snapshots):
                                snap = Int64(snap_rows[b])
                            else:
                                snap = Int64(snap_rows[b * self.layout.snapshots])
                        if snap != -1:
                            if (snap < self.slots) | (snap >= self.slots + self.ns):
                                error = error | Int32(16)
                            if cutlass.const_expr(self.capture or self.pb > 0):
                                if cutlass.const_expr(self.dense_offsets):
                                    if Int64(offsets[b]) < 0:
                                        error = error | Int32(4)
                                else:
                                    if Int64(offsets[b * self.layout.offsets]) < 0:
                                        error = error | Int32(4)
        return error

    @cute.kernel
    def control(
        self,
        indices,
        positions,
        metadata,
        epoch,
        pm,
        accepted,
        snap_rows,
        offsets,
        record,
        sequence,
        poison,
        status_slot,
    ):
        tid, _, _ = cute.arch.thread_idx()
        invocation_ok = Boolean(True)
        if cutlass.const_expr(bool(self.slot_layout)):
            invocation_ok, invocation = self.invocation(status_slot)
            record = self.invocation_view(record, invocation, self.slot_layout[1])
            sequence = self.invocation_view(sequence, invocation, self.slot_layout[2])
            pm = self.invocation_view(pm, invocation, self.slot_layout[4])
        if tid == 0:
            if cutlass.const_expr(bool(self.slot_layout)):
                if not invocation_ok:
                    status_or(poison.iterator, Int32(64))
            if invocation_ok:
                words = cute.recast_tensor(record, Int32)
                seq, old, done = sequence[0], cute.recast_tensor(record, Int64)[0], words[3]
                code, poisoned = words[2], poison[0]
                if (seq <= 0) | (old != seq) | (done != 0):
                    poisoned = poisoned | Int32(64)
                elif poisoned != 0:
                    code = code | Int32(32)
                # Guard precedes validation; propagation follows ALL batch rows.
                # A prior code alone does not suppress validation until propagated.
                if cutlass.const_expr(self.batch > 0):
                    if poisoned == 0:
                        for b in range(self.batch):
                            row = Int64(b)
                            code = code | self.validate_row(
                                indices,
                                positions,
                                metadata,
                                epoch,
                                pm,
                                accepted,
                                snap_rows,
                                offsets,
                                row,
                            )
                # Preserve any error already installed by another invocation record
                # (or a colliding call), as portable guard/propagate's atomics do.
                code = code | status_or(words.iterator + 2, code)
                if cutlass.const_expr(self.batch > 0):
                    poisoned = poisoned | code
                status_or(poison.iterator, poisoned)

    @cute.jit
    def raw_value(self, x, history, ring, pending, slot, b, pb, empty, cursor, t, col):
        if cutlass.const_expr(self.wide_tokens):
            t = Int64(t)
        value = BFloat16(0)
        if t < 0:
            if cutlass.const_expr(self.immediate):
                if (t >= -3) & (col < self.dc):
                    if cutlass.const_expr(self.dense_history):
                        value = history[(slot * 3 + t + 3) * self.dc + col]
                    else:
                        value = history[slot * self.h_stride + (t + 3) * self.dc + col]
            else:
                if empty:
                    if (t >= -3) & (col < self.dc):
                        if cutlass.const_expr(self.dense_history):
                            value = history[(slot * 3 + t + 3) * self.dc + col]
                        else:
                            value = history[slot * self.h_stride + (t + 3) * self.dc + col]
                else:
                    rr = ((cursor + t) % (self.window + 3) + self.window + 3) % (self.window + 3)
                    value = ring[(slot * (self.window + 3) + rr) * self.width + col]
        else:
            if cutlass.const_expr(self.mode in ("verify", "bridge_step", "delayed_step")):
                xc = col if col < self.dc else col + self.hv * 128
                if cutlass.const_expr(self.dense_token):
                    value = x[Int64(b) * self.stride + t * self.proj + xc]
                else:
                    value = x[Int64(b) * self.stride + Int64(t) * self.token_stride + xc]
            elif cutlass.const_expr(self.mode in ("commit", "export_snapshot") and self.pb > 0):
                value = pending[(pb * self.q + t) * self.width + col]
        return value

    @cute.jit
    def raw_vector(self, x, history, ring, pending, slot, b, pb, empty, cursor, t, col, frag):
        """Eight raw values starting at col (col % 8 == 0), as raw_value."""
        if cutlass.const_expr(self.wide_tokens):
            t = Int64(t)
        if t < 0:
            if cutlass.const_expr(self.immediate):
                for j in cutlass.range_constexpr(8):
                    frag[j] = BFloat16(0)
                if (t >= -3) & (col < self.dc):
                    if cutlass.const_expr(self.dense_history):
                        vector_load(history, (slot * 3 + t + 3) * self.dc + col, frag)
                    else:
                        vector_load(history, slot * self.h_stride + (t + 3) * self.dc + col, frag)
            else:
                if empty:
                    for j in cutlass.range_constexpr(8):
                        frag[j] = BFloat16(0)
                    if (t >= -3) & (col < self.dc):
                        if cutlass.const_expr(self.dense_history):
                            vector_load(history, (slot * 3 + t + 3) * self.dc + col, frag)
                        else:
                            vector_load(
                                history, slot * self.h_stride + (t + 3) * self.dc + col, frag
                            )
                else:
                    rr = ((cursor + t) % (self.window + 3) + self.window + 3) % (self.window + 3)
                    vector_load(ring, (slot * (self.window + 3) + rr) * self.width + col, frag)
        else:
            if cutlass.const_expr(self.mode in ("verify", "bridge_step", "delayed_step")):
                xc = col if col < self.dc else col + self.hv * 128
                if cutlass.const_expr(self.dense_token):
                    if cutlass.const_expr(self.aligned_x):
                        vector_load(x, Int64(b) * self.stride + t * self.proj + xc, frag)
                    else:
                        self.load_projected(x, Int64(b) * self.stride + t * self.proj + xc, frag)
                else:
                    self.load_projected(
                        x, Int64(b) * self.stride + Int64(t) * self.token_stride + xc, frag
                    )
            elif cutlass.const_expr(self.mode in ("commit", "export_snapshot") and self.pb > 0):
                vector_load(pending, (pb * self.q + t) * self.width + col, frag)
            else:
                for j in cutlass.range_constexpr(8):
                    frag[j] = BFloat16(0)

    @cute.jit
    def token_index(self, value):
        if cutlass.const_expr(self.wide_tokens):
            return Int64(value)
        else:
            return Int32(value)

    @cute.jit
    def bounds(self, accepted, snap_rows, offsets, target, pb, b):
        m, end = self.token_index(Int32(0)), self.token_index(Int32(0))
        if cutlass.const_expr(
            self.mode == "commit" or (self.mode == "export_snapshot" and self.pb > 0)
        ):
            if cutlass.const_expr(self.dense_accepted):
                length = Int64(accepted[pb])
            else:
                length = Int64(accepted[pb * self.layout.accepted])
            if cutlass.const_expr(self.trusted):
                # Clamp before narrowing to the bounded token-index specialization.
                length = min(max(length, Int64(0)), Int64(self.q))
            m = self.token_index(length)
        if cutlass.const_expr(self.mode == "verify"):
            end = self.token_index(self.q)
        elif cutlass.const_expr(self.mode in ("bridge_step", "delayed_step")):
            end = self.token_index(1)
        elif cutlass.const_expr(self.mode == "commit"):
            end = m
        snap, k = Int64(-1), Int64(0)
        if cutlass.const_expr(self.mode == "export_snapshot"):
            snap = target
            if cutlass.const_expr(self.pb > 0):
                if cutlass.const_expr(self.dense_offsets):
                    k = Int64(offsets[b])
                else:
                    k = Int64(offsets[b * self.layout.offsets])
                end = self.token_index(k) if (k >= 0) & (k <= m) else self.token_index(0)
        elif cutlass.const_expr(self.capture):
            (snap, k) = (
                (
                    Int64(snap_rows[b])
                    if cutlass.const_expr(self.dense_snapshots)
                    else Int64(snap_rows[b * self.layout.snapshots])
                ),
                (
                    Int64(offsets[b])
                    if cutlass.const_expr(self.dense_offsets)
                    else Int64(offsets[b * self.layout.offsets])
                ),
            )
        return m, end, snap, k

    @cute.jit
    def lazy_row(self, n, m, capture):
        lazy = Boolean(False)
        if cutlass.const_expr(self.delayed):
            lazy = (n + m < self.window) & (not capture)
        return lazy

    @cute.jit
    def append_rows(self, ring, pending, slot, pb, cursor, m, cta, tid):
        """Disjoint dead ring rows, with one writer per element across all CTAs."""
        worker = cta * self.threads + tid
        workers = self.nv * self.hv * self.threads
        if cutlass.const_expr(self.vec):
            frag = cute.make_rmem_tensor((8,), BFloat16)
            per_row = self.width // 8
            for i in range(worker, m * per_row, workers):
                t, c = i // per_row, (i % per_row) * 8
                rr = (cursor + t) % (self.window + 3)
                vector_load(pending, (pb * self.q + t) * self.width + c, frag)
                vector_store(ring, (slot * (self.window + 3) + rr) * self.width + c, frag)
        else:
            for i in range(worker, m * self.width, workers):
                t, c = i // self.width, i % self.width
                rr = (cursor + t) % (self.window + 3)
                ring[(slot * (self.window + 3) + rr) * self.width + c] = pending[
                    (pb * self.q + t) * self.width + c
                ]

    @cute.jit
    def store_matrix(self, matrix, destination, tid, row_base):
        if tid < self.replay_threads:
            for r in cutlass.range_constexpr(self.rows):
                row = row_base + tid // self.lanes + r * (self.replay_threads // self.lanes)
                for j in cutlass.range_constexpr(self.chunks):
                    col = (tid % self.lanes) * 4 + j * self.lanes * 4
                    vector_store(destination, row * 128 + col, matrix[r, j, None])

    @cute.kernel
    def compute(
        self,
        x,
        cw,
        al,
        bias,
        nw,
        indices,
        checkpoint,
        ring,
        metadata,
        pending,
        pm,
        accepted,
        snap_rows,
        offsets,
        out,
        record,
        poison,
        history,
        epoch,
        snapshots,
        valid,
        arrivals,
        status_slot,
    ):
        tile, h, bid = cute.arch.block_idx()
        row_base = tile * self.tile
        b = Int64(bid)
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = SmemAllocator()
        raw = smem.allocate_tensor(
            BFloat16, cute.make_layout((self.warps + 3, 384), stride=(384, 1)), byte_alignment=16
        )
        weights = smem.allocate_tensor(
            BFloat16, cute.make_layout((384, 4), stride=(4, 1)), byte_alignment=16
        )
        features = smem.allocate_tensor(
            Float32, cute.make_layout((self.warps, 512), stride=(512, 1)), byte_alignment=16
        )
        raw_gates = smem.allocate_tensor(
            Float32, cute.make_layout((self.warps, 2), stride=(2, 1)), byte_alignment=16
        )
        gates = smem.allocate_tensor(
            Float32, cute.make_layout((self.warps, 8), stride=(8, 1)), byte_alignment=16
        )
        values = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.output_rows, 128), stride=(128, 1)),
            byte_alignment=16,
        )
        if cutlass.const_expr(self.fused and self.mode != "verify"):
            election = smem.allocate_tensor(Int32, cute.make_layout(1))
        # The selector is written before the graph/stream work that launches this
        # kernel (never by its PDL predecessor), so it may be read before the PDL
        # wait; its latency then overlaps the prologue instead of the first
        # slot-dependent loads.
        invocation_ok = Boolean(True)
        if cutlass.const_expr(bool(self.slot_layout)):
            invocation_ok, invocation = self.invocation(status_slot)
            if cutlass.const_expr(not self.trusted):
                record = self.invocation_view(record, invocation, self.slot_layout[1])
            pending = self.invocation_view(pending, invocation, self.slot_layout[3])
            pm = self.invocation_view(pm, invocation, self.slot_layout[4])
            valid = self.invocation_view(valid, invocation, self.slot_layout[5])
            arrivals = self.invocation_view(arrivals, invocation, self.slot_layout[6])
        kh = h // (self.hv // self.hk)
        if cutlass.const_expr(self.vec):
            # Column pairs are adjacent within each 128-column segment.
            pack = cute.make_rmem_tensor((8,), BFloat16)
            for c2 in range(tid, 192, self.threads):
                col = self.column(c2 * 2, kh, h, 0)
                vector_load(cw, col * 4, pack)
                for j in cutlass.range_constexpr(8):
                    weights[c2 * 2 + j // 4, j % 4] = pack[j]
        else:
            for i in range(tid, 384 * 4, self.threads):
                col = self.column(i // 4, kh, h, 0)
                weights[i // 4, i % 4] = cw[col * 4 + i % 4]
        if cutlass.const_expr(self.pdl):
            # Only immutable weights/shared scratch precede this wait. It also
            # protects writes from readers in the previous stream kernel.
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if invocation_ok:
            slot, target = self.source(indices, b)
            pb = self.pending_row(pm, slot, b)
            active = self.row_active(slot, target, pb)
            if cutlass.const_expr(not self.trusted):
                active = active & (cute.recast_tensor(record, Int32)[2] == 0) & (poison[0] == 0)
            run_row, read_row = active, Boolean(True)
            if cutlass.const_expr(self.chunked_output):
                # Masked clusters execute the same exchange schedule without reading state.
                run_row, read_row = Boolean(True), active
            if run_row:
                n, cursor = self.token_index(0), self.token_index(0)
                if read_row:
                    n = self.token_index(metadata[slot * 8 + 3])
                    cursor = self.token_index(metadata[slot * 8 + 2])
                empty = n == 0
                m, end, snap, k = self.bounds(accepted, snap_rows, offsets, target, pb, b)
                capture = (snap >= self.slots) & (snap < self.slots + self.ns) & (k >= 0) & (k <= m)
                lazy = self.lazy_row(n, m, capture)
                if lazy:
                    self.append_rows(ring, pending, slot, pb, cursor, m, h * self.nv + tile, tid)
                else:
                    if cutlass.const_expr(self.dense_checkpoint):
                        state_head = cute.make_tensor(
                            checkpoint.iterator + (slot * self.hv + h) * 16384,
                            cute.make_layout(16384),
                        )
                    else:
                        state_head = cute.make_tensor(
                            checkpoint.iterator + slot * self.cp_stride + h * 16384,
                            cute.make_layout(16384),
                        )
                    matrix = cute.make_rmem_tensor(
                        cute.make_layout(
                            (self.rows, self.chunks, 4), stride=(self.chunks * 4, 4, 1)
                        ),
                        Float32,
                    )
                    if (tid < self.replay_threads) & read_row:
                        for r in cutlass.range_constexpr(self.rows):
                            row = (
                                row_base
                                + tid // self.lanes
                                + r * (self.replay_threads // self.lanes)
                            )
                            for j in cutlass.range_constexpr(self.chunks):
                                vector_load(
                                    state_head,
                                    row * 128 + (tid % self.lanes) * 4 + j * self.lanes * 4,
                                    matrix[r, j, None],
                                )
                    emitted = self.token_index(0)
                    for start in range(-n, end, self.warps):
                        count = self.token_index(self.warps)
                        if end - start < count:
                            count = end - start
                        if read_row:
                            if cutlass.const_expr(self.vec):
                                # 16-byte staging: each 8-column group lies inside one segment.
                                frag = cute.make_rmem_tensor((8,), BFloat16)
                                for i in range(tid, (count + 3) * 48, self.threads):
                                    rr, c = i // 48, (i % 48) * 8
                                    col = self.column(c, kh, h, 0)
                                    self.raw_vector(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        start + rr - 3,
                                        col,
                                        frag,
                                    )
                                    for j in cutlass.range_constexpr(8):
                                        raw[rr, c + j] = frag[j]
                                    if cutlass.const_expr(self.fused and self.mode == "verify"):
                                        # Only the current chunk's tokens (not its three seeds).
                                        # Each key head has one owner; every value head owns V.
                                        if (tile == 0) & (rr >= 3) & (start + rr - 3 >= 0):
                                            if (c >= 256) | (h % (self.hv // self.hk) == 0):
                                                vector_store(
                                                    pending,
                                                    (b * self.q + start + rr - 3) * self.width
                                                    + col,
                                                    frag,
                                                )
                            else:
                                for i in range(tid, (count + 3) * 384, self.threads):
                                    col = self.column(i % 384, kh, h, 0)
                                    value = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        start + i // 384 - 3,
                                        col,
                                    )
                                    raw[i // 384, i % 384] = value
                                    if cutlass.const_expr(self.fused and self.mode == "verify"):
                                        t = start + i // 384 - 3
                                        if (tile == 0) & (i // 384 >= 3) & (t >= 0):
                                            if (i % 384 >= 256) | (h % (self.hv // self.hk) == 0):
                                                pending[(b * self.q + t) * self.width + col] = value
                            for i in range(tid, count * 2, self.threads):
                                col = self.dc + (i % 2) * self.hv + h
                                value = self.raw_value(
                                    x,
                                    history,
                                    ring,
                                    pending,
                                    slot,
                                    b,
                                    pb,
                                    empty,
                                    cursor,
                                    start + i // 2,
                                    col,
                                )
                                raw_gates[i // 2, i % 2] = Float32(value)
                                if cutlass.const_expr(self.fused and self.mode == "verify"):
                                    if (tile == 0) & (start + i // 2 >= 0):
                                        # Gates are isolated BF16 columns owned by head h;
                                        # a 16-byte store would overwrite other CTA owners.
                                        pending[
                                            (b * self.q + start + i // 2) * self.width + col
                                        ] = value
                            cute.arch.sync_threads()
                            if warp < count:
                                q = cute.make_rmem_tensor((4,), Float32)
                                key = cute.make_rmem_tensor((4,), Float32)
                                for j in cutlass.range_constexpr(4):
                                    c = lane + j * 32
                                    # n=warps disables Step's special current-token remap:
                                    # this chunk is laid out contiguously, including its seeds.
                                    qr = self.convolve(raw, weights, warp, c, self.warps)
                                    kr = self.convolve(raw, weights, warp, 128 + c, self.warps)
                                    q[j], key[j] = mul_rn(qr, sigmoid(qr)), mul_rn(kr, sigmoid(kr))
                                    v = self.convolve(raw, weights, warp, 256 + c, self.warps)
                                    features[warp, c], features[warp, 128 + c] = q[j], key[j]
                                    features[warp, 256 + c], features[warp, 384 + c] = v, sigmoid(v)
                                self.prepare_gates(
                                    q, key, raw_gates, gates, al, bias, warp, h, lane
                                )
                            cute.arch.sync_threads()
                            for t in range(count):
                                if capture & (Int64(start + t) == k):
                                    if cutlass.const_expr(self.dense_checkpoint):
                                        dest = cute.make_tensor(
                                            checkpoint.iterator + (snap * self.hv + h) * 16384,
                                            cute.make_layout(16384),
                                        )
                                    else:
                                        dest = cute.make_tensor(
                                            checkpoint.iterator + snap * self.cp_stride + h * 16384,
                                            cute.make_layout(16384),
                                        )
                                    self.store_matrix(matrix, dest, tid, row_base)
                                if tid < self.replay_threads:
                                    if cutlass.const_expr(
                                        self.mode in ("verify", "bridge_step", "delayed_step")
                                    ):
                                        if start + t >= 0:
                                            output_row = start + t
                                            if cutlass.const_expr(self.chunked_output):
                                                output_row = output_row % self.output_rows
                                            self.replay_token(
                                                matrix,
                                                features,
                                                gates,
                                                values[output_row, None],
                                                t,
                                                tid,
                                                state_head,
                                                False,
                                                True,
                                                True,
                                                row_base,
                                            )
                                        else:
                                            self.replay_token(
                                                matrix,
                                                features,
                                                gates,
                                                values[0, None],
                                                t,
                                                tid,
                                                state_head,
                                                False,
                                                False,
                                                row_base=row_base,
                                            )
                                    else:
                                        self.replay_token(
                                            matrix,
                                            features,
                                            gates,
                                            values[0, None],
                                            t,
                                            tid,
                                            state_head,
                                            False,
                                            False,
                                            row_base=row_base,
                                        )
                        # All consumers finish before the next chunk replaces scratch.
                        cute.arch.sync_threads()
                        if cutlass.const_expr(self.chunked_output):
                            completed = start + count
                            if completed >= emitted + OUTPUT_RESIDENT:
                                self.output_block(
                                    x,
                                    nw,
                                    out,
                                    values,
                                    b,
                                    h,
                                    tile,
                                    tid,
                                    active,
                                    emitted,
                                    emitted + OUTPUT_RESIDENT,
                                )
                                emitted += OUTPUT_RESIDENT
                            if (completed == end) & (emitted < end):
                                self.output_block(
                                    x, nw, out, values, b, h, tile, tid, active, emitted, end
                                )
                    if capture & (k == Int64(end)):
                        if cutlass.const_expr(self.dense_checkpoint):
                            dest = cute.make_tensor(
                                checkpoint.iterator + (snap * self.hv + h) * 16384,
                                cute.make_layout(16384),
                            )
                        else:
                            dest = cute.make_tensor(
                                checkpoint.iterator + snap * self.cp_stride + h * 16384,
                                cute.make_layout(16384),
                            )
                        self.store_matrix(matrix, dest, tid, row_base)
                    if cutlass.const_expr(self.mode in ("commit", "flush", "bridge_step")):
                        self.store_matrix(matrix, state_head, tid, row_base)
                    elif cutlass.const_expr(self.mode == "delayed_step"):
                        if (n + 1 == self.window) | self.flush_always:
                            self.store_matrix(matrix, state_head, tid, row_base)
            if cutlass.const_expr(
                self.mode in ("verify", "bridge_step", "delayed_step") and not self.chunked_output
            ):
                # Every CTA, including masked/poisoned rows, reaches both barriers.
                # The first joins local writers and guarantees peers' SMEM lifetime;
                # the second completes all remote stores before any CTA can exit.
                if cutlass.const_expr(self.nv > 1):
                    cluster_sync()
                    if active:
                        for i in range(tid, self.output_tokens * self.tile, self.threads):
                            t, row = i // self.tile, row_base + i % self.tile
                            value = values[t, row]
                            for peer in cutlass.range_constexpr(self.nv):
                                if peer != tile:
                                    cluster_store(values.iterator + t * 128 + row, peer, value)
                    cluster_sync()
                else:
                    cute.arch.sync_threads()
                if active:
                    for t in range(warp, self.output_tokens, self.warps):
                        # One warp per token; all lanes retain the original 128-value reduction order.
                        ov = cute.make_rmem_tensor((4,), Float32)
                        for j in cutlass.range_constexpr(4):
                            ov[j] = values[t, lane + j * 32]
                        total = self.output_sum(ov)
                        for r in cutlass.range_constexpr(self.tile // 32):
                            row = row_base + lane + r * 32
                            if cutlass.const_expr(self.dense_token):
                                z = Float32(
                                    x[
                                        Int64(b) * self.stride
                                        + t * self.proj
                                        + self.dc
                                        + h * 128
                                        + row
                                    ]
                                )
                            else:
                                z = Float32(
                                    x[
                                        Int64(b) * self.stride
                                        + Int64(t) * self.token_stride
                                        + self.dc
                                        + h * 128
                                        + row
                                    ]
                                )
                            rho = cute.math.rsqrt(total / 128 + 1e-6, fastmath=True)
                            y = (Float32(nw[row]) * z * sigmoid(z)) * (values[t, row] * rho)
                            if cutlass.const_expr(self.dense_out):
                                out[((b * self.output_tokens + t) * self.hv + h) * 128 + row] = (
                                    y.to(BFloat16)
                                )
                            else:
                                out[
                                    Int64(b) * self.out_stride
                                    + Int64(t) * self.out_token_stride
                                    + h * 128
                                    + row
                                ] = y.to(BFloat16)
                else:
                    for i in range(tid, self.output_tokens * self.tile, self.threads):
                        t, row = i // self.tile, row_base + i % self.tile
                        if cutlass.const_expr(self.dense_out):
                            out[((b * self.output_tokens + t) * self.hv + h) * 128 + row] = (
                                BFloat16(0)
                            )
                        else:
                            out[
                                Int64(b) * self.out_stride
                                + Int64(t) * self.out_token_stride
                                + h * 128
                                + row
                            ] = BFloat16(0)

            if cutlass.const_expr(self.fused):
                if cutlass.const_expr(self.mode == "verify"):
                    if (tile == 0) & (h == 0):
                        self.publish_row(
                            x,
                            indices,
                            history,
                            ring,
                            metadata,
                            epoch,
                            snapshots,
                            pending,
                            pm,
                            accepted,
                            snap_rows,
                            offsets,
                            valid,
                            b,
                            Int32(0),
                            tid,
                            slot,
                            target,
                            pb,
                            active,
                            Int32(1),
                            self.threads,
                        )
                else:
                    # Every row-b read of ring/history/metadata/pending precedes its
                    # CTA barrier and acq_rel arrival. The last arriver acquires the
                    # chain of all CTAs, so no CTA of row b can still read state that
                    # this publisher overwrites. Cluster barriers above are reached
                    # even by masked rows, before any CTA can publish or exit.
                    cute.arch.sync_threads()
                    if active:
                        if tid == 0:
                            election[0] = arrive(arrivals.iterator + b)
                        cute.arch.sync_threads()
                        if election[0] == self.nv * self.hv - 1:
                            self.publish_row(
                                x,
                                indices,
                                history,
                                ring,
                                metadata,
                                epoch,
                                snapshots,
                                pending,
                                pm,
                                accepted,
                                snap_rows,
                                offsets,
                                valid,
                                b,
                                Int32(0),
                                tid,
                                slot,
                                target,
                                pb,
                                active,
                                Int32(1),
                                self.threads,
                            )
                            cute.arch.sync_threads()
                            if tid == 0:
                                arrivals[b] = Int32(0)
                    elif (tile == 0) & (h == 0):
                        if cutlass.const_expr(self.mode == "export_snapshot" or self.capture):
                            if tid == 0:
                                if cutlass.const_expr(self.dense_valid):
                                    valid[b] = False
                                else:
                                    valid[b * self.layout.valid] = False

    @cute.jit
    def output_block(self, x, nw, out, values, b, h, tile, tid, active, first, end):
        row_base = tile * self.tile
        lane, warp = tid % 32, tid // 32
        if cutlass.const_expr(self.nv > 1):
            cluster_sync()
            if active:
                for i in range(tid, (end - first) * self.tile, self.threads):
                    t, row = first + i // self.tile, row_base + i % self.tile
                    rr = t % self.output_rows
                    value = values[rr, row]
                    for peer in cutlass.range_constexpr(self.nv):
                        if peer != tile:
                            cluster_store(values.iterator + rr * 128 + row, peer, value)
            cluster_sync()
        else:
            cute.arch.sync_threads()
        if active:
            for t in range(first + warp, end, self.warps):
                rr = t % self.output_rows
                ov = cute.make_rmem_tensor((4,), Float32)
                for j in cutlass.range_constexpr(4):
                    ov[j] = values[rr, lane + j * 32]
                total = self.output_sum(ov)
                for r in cutlass.range_constexpr(self.tile // 32):
                    row = row_base + lane + r * 32
                    if cutlass.const_expr(self.dense_token):
                        z = Float32(
                            x[Int64(b) * self.stride + t * self.proj + self.dc + h * 128 + row]
                        )
                    else:
                        z = Float32(
                            x[
                                Int64(b) * self.stride
                                + Int64(t) * self.token_stride
                                + self.dc
                                + h * 128
                                + row
                            ]
                        )
                    rho = cute.math.rsqrt(total / 128 + 1e-6, fastmath=True)
                    y = (Float32(nw[row]) * z * sigmoid(z)) * (values[rr, row] * rho)
                    if cutlass.const_expr(self.dense_out):
                        out[((b * self.output_tokens + t) * self.hv + h) * 128 + row] = y.to(
                            BFloat16
                        )
                    else:
                        out[
                            Int64(b) * self.out_stride
                            + Int64(t) * self.out_token_stride
                            + h * 128
                            + row
                        ] = y.to(BFloat16)
        else:
            for i in range(tid, (end - first) * self.tile, self.threads):
                t, row = first + i // self.tile, row_base + i % self.tile
                if cutlass.const_expr(self.dense_out):
                    out[((b * self.output_tokens + t) * self.hv + h) * 128 + row] = BFloat16(0)
                else:
                    out[
                        Int64(b) * self.out_stride
                        + Int64(t) * self.out_token_stride
                        + h * 128
                        + row
                    ] = BFloat16(0)
        # No producer may recycle a row until all peer consumers finish reading it.
        if cutlass.const_expr(self.nv > 1):
            cluster_sync()
        else:
            cute.arch.sync_threads()

    @cute.jit
    def publish_vectors(
        self,
        x,
        ring,
        pending,
        history,
        slot,
        b,
        pb,
        empty,
        cursor,
        end,
        capture,
        snap,
        k,
        tid,
        workers,
        lazy,
    ):
        """16-byte version of the column loop below; dc % 8 == 0, so no vector
        straddles the history/gate boundary."""
        h0 = cute.make_rmem_tensor((8,), BFloat16)
        h1 = cute.make_rmem_tensor((8,), BFloat16)
        h2 = cute.make_rmem_tensor((8,), BFloat16)
        s0 = cute.make_rmem_tensor((8,), BFloat16)
        for c8 in range(tid, self.width // 8, workers):
            c = c8 * 8
            if c < self.dc:
                if cutlass.const_expr(self.mode != "export_snapshot"):
                    self.raw_vector(
                        x, history, ring, pending, slot, b, pb, empty, cursor, end - 3, c, h0
                    )
                    self.raw_vector(
                        x, history, ring, pending, slot, b, pb, empty, cursor, end - 2, c, h1
                    )
                    self.raw_vector(
                        x, history, ring, pending, slot, b, pb, empty, cursor, end - 1, c, h2
                    )
                if capture:
                    for j in cutlass.range_constexpr(3):
                        self.raw_vector(
                            x, history, ring, pending, slot, b, pb, empty, cursor, k - 3 + j, c, s0
                        )
                        if cutlass.const_expr(self.dense_history):
                            vector_store(history, (snap * 3 + j) * self.dc + c, s0)
                        else:
                            vector_store(history, snap * self.h_stride + j * self.dc + c, s0)
                if cutlass.const_expr(self.mode != "export_snapshot"):
                    if cutlass.const_expr(self.dense_history):
                        vector_store(history, (slot * 3) * self.dc + c, h0)
                    else:
                        vector_store(history, slot * self.h_stride + c, h0)
                    if cutlass.const_expr(self.dense_history):
                        vector_store(history, (slot * 3 + 1) * self.dc + c, h1)
                    else:
                        vector_store(history, slot * self.h_stride + self.dc + c, h1)
                    if cutlass.const_expr(self.dense_history):
                        vector_store(history, (slot * 3 + 2) * self.dc + c, h2)
                    else:
                        vector_store(history, slot * self.h_stride + 2 * self.dc + c, h2)
            else:
                for j in cutlass.range_constexpr(8):
                    h0[j], h1[j], h2[j] = BFloat16(0), BFloat16(0), BFloat16(0)
            if cutlass.const_expr(not self.immediate and self.mode != "export_snapshot"):
                if not lazy:
                    base = (slot * (self.window + 3) + self.window) * self.width + c
                    vector_store(ring, base, h0)
                    vector_store(ring, base + self.width, h1)
                    vector_store(ring, base + 2 * self.width, h2)

    @cute.jit
    def publish_step(self, x, history, ring, metadata, slot, b, cursor, empty, tid, workers):
        # Each column owns its three history reads and writes. Seed an empty ring
        # from authoritative history before appending, including rotated cursors.
        for c in range(tid, self.width, workers):
            h0, h1, h2 = BFloat16(0), BFloat16(0), BFloat16(0)
            if c < self.dc:
                if cutlass.const_expr(self.dense_history):
                    h0 = history[(slot * 3) * self.dc + c]
                else:
                    h0 = history[slot * self.h_stride + c]
                if cutlass.const_expr(self.dense_history):
                    h1 = history[(slot * 3 + 1) * self.dc + c]
                else:
                    h1 = history[slot * self.h_stride + self.dc + c]
                if cutlass.const_expr(self.dense_history):
                    h2 = history[(slot * 3 + 2) * self.dc + c]
                else:
                    h2 = history[slot * self.h_stride + 2 * self.dc + c]
            if empty:
                ring[
                    (slot * (self.window + 3) + (cursor + self.window) % (self.window + 3))
                    * self.width
                    + c
                ] = h0
                ring[
                    (slot * (self.window + 3) + (cursor + self.window + 1) % (self.window + 3))
                    * self.width
                    + c
                ] = h1
                ring[
                    (slot * (self.window + 3) + (cursor + self.window + 2) % (self.window + 3))
                    * self.width
                    + c
                ] = h2
            xc = c if c < self.dc else c + self.hv * 128
            current = x[Int64(b) * self.stride + xc]
            ring[(slot * (self.window + 3) + cursor) * self.width + c] = current
            if c < self.dc:
                if cutlass.const_expr(self.dense_history):
                    history[(slot * 3) * self.dc + c] = h1
                else:
                    history[slot * self.h_stride + c] = h1
                if cutlass.const_expr(self.dense_history):
                    history[(slot * 3 + 1) * self.dc + c] = h2
                else:
                    history[slot * self.h_stride + self.dc + c] = h2
                if cutlass.const_expr(self.dense_history):
                    history[(slot * 3 + 2) * self.dc + c] = current
                else:
                    history[slot * self.h_stride + 2 * self.dc + c] = current
        cute.arch.sync_threads()
        if tid == 0:
            base = slot * 8
            pos, n = metadata[base + 1] + 1, metadata[base + 3] + 1
            flush = (n == self.window) | self.flush_always
            metadata[base + 1] = pos
            metadata[base + 2] = (cursor + 1) % (self.window + 3)
            metadata[base + 3] = Int64(0) if flush else n
            if flush:
                metadata[base] = pos

    @cute.kernel
    def publish_phase(
        self,
        x,
        indices,
        history,
        ring,
        metadata,
        epoch,
        snapshots,
        pending,
        pm,
        accepted,
        snap_rows,
        offsets,
        valid,
        record,
        poison,
        status_slot,
    ):
        bid, chunk, _ = cute.arch.block_idx()
        b = Int64(bid)
        tid, _, _ = cute.arch.thread_idx()
        # The selector is written before the graph/stream work that launches this
        # kernel (never by its PDL predecessor), so it may be read before the PDL
        # wait; its latency then overlaps the prologue instead of the first
        # slot-dependent loads.
        invocation_ok = Boolean(True)
        if cutlass.const_expr(bool(self.slot_layout)):
            invocation_ok, invocation = self.invocation(status_slot)
            if cutlass.const_expr(not self.trusted):
                record = self.invocation_view(record, invocation, self.slot_layout[1])
            pending = self.invocation_view(pending, invocation, self.slot_layout[3])
            pm = self.invocation_view(pm, invocation, self.slot_layout[4])
            valid = self.invocation_view(valid, invocation, self.slot_layout[5])
        if cutlass.const_expr(self.pdl):
            # Compute still reads ring/history/metadata: wait before ANY global
            # access, including valid-bit clears and publication overwrites.
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        if invocation_ok:
            slot, target = self.source(indices, b)
            pb = self.pending_row(pm, slot, b)
            active = self.row_active(slot, target, pb)
            allowed = Int32(1)
            if cutlass.const_expr(not self.trusted):
                allowed = Int32(poison[0] == 0)
                active = active & (cute.recast_tensor(record, Int32)[2] == 0)
            self.publish_row(
                x,
                indices,
                history,
                ring,
                metadata,
                epoch,
                snapshots,
                pending,
                pm,
                accepted,
                snap_rows,
                offsets,
                valid,
                b,
                chunk,
                tid,
                slot,
                target,
                pb,
                active,
                allowed,
                128,
            )

    @cute.jit
    def publish_row(
        self,
        x,
        indices,
        history,
        ring,
        metadata,
        epoch,
        snapshots,
        pending,
        pm,
        accepted,
        snap_rows,
        offsets,
        valid,
        b,
        chunk,
        tid,
        slot,
        target,
        pb,
        active,
        allowed,
        workers,
    ):
        if allowed != 0:
            if cutlass.const_expr(self.mode == "export_snapshot" or self.capture):
                if (tid == 0) & (chunk == 0):
                    if cutlass.const_expr(self.dense_valid):
                        valid[b] = False
                    else:
                        valid[b * self.layout.valid] = False
            if active:
                base = slot * 8
                pos = metadata[base + 1]
                cursor = (
                    Int64(metadata[base + 2])
                    if cutlass.const_expr(self.wide_tokens)
                    else Int32(metadata[base + 2])
                )
                empty = metadata[base + 3] == 0
                if cutlass.const_expr(self.mode == "delayed_step"):
                    self.publish_step(
                        x, history, ring, metadata, slot, b, cursor, empty, tid, workers
                    )
                elif cutlass.const_expr(self.mode == "reject"):
                    if tid == 0:
                        pm[pb * 8 + 5] = Int64(4)
                elif cutlass.const_expr(self.mode == "flush"):
                    if tid == 0:
                        metadata[base], metadata[base + 3] = pos, Int64(0)
                elif cutlass.const_expr(self.mode == "verify"):
                    if cutlass.const_expr(not self.fused):
                        if cutlass.const_expr(self.vec):
                            frag = cute.make_rmem_tensor((8,), BFloat16)
                            per_row = self.width // 8
                            for v in range(
                                self.token_index(chunk * 128 + tid),
                                self.q * per_row,
                                self.copy_ctas * 128,
                            ):
                                t, c = v // per_row, (v % per_row) * 8
                                self.raw_vector(
                                    x,
                                    history,
                                    ring,
                                    pending,
                                    slot,
                                    b,
                                    pb,
                                    empty,
                                    cursor,
                                    t,
                                    c,
                                    frag,
                                )
                                vector_store(pending, (b * self.q + t) * self.width + c, frag)
                        else:
                            for i in range(self.token_index(tid), self.q * self.width, 128):
                                pending[b * self.q * self.width + i] = self.raw_value(
                                    x,
                                    history,
                                    ring,
                                    pending,
                                    slot,
                                    b,
                                    pb,
                                    empty,
                                    cursor,
                                    i // self.width,
                                    i % self.width,
                                )
                    cute.arch.sync_threads()
                    if (tid == 0) & (chunk == 0):
                        pm[b * 8], pm[b * 8 + 1] = slot, metadata[base + 5]
                        pm[b * 8 + 2], pm[b * 8 + 3] = metadata[base], pos
                        pm[b * 8 + 4], pm[b * 8 + 5] = Int64(self.q), Int64(1)
                        pm[b * 8 + 6], pm[b * 8 + 7] = Int64(-1), epoch[0]
                else:
                    m, end, snap, k = self.bounds(accepted, snap_rows, offsets, target, pb, b)
                    capture = (
                        (snap >= self.slots) & (snap < self.slots + self.ns) & (k >= 0) & (k <= m)
                    )
                    n = (
                        Int64(metadata[base + 3])
                        if cutlass.const_expr(self.wide_tokens)
                        else Int32(metadata[base + 3])
                    )
                    lazy = self.lazy_row(n, m, capture)
                    if lazy & empty & (m > 0):
                        for i in range(tid, 3 * self.width, workers):
                            t, c = i // self.width, i % self.width
                            rr = (cursor + t + self.window) % (self.window + 3)
                            value = BFloat16(0)
                            if c < self.dc:
                                if cutlass.const_expr(self.dense_history):
                                    value = history[(slot * 3 + t) * self.dc + c]
                                else:
                                    value = history[slot * self.h_stride + t * self.dc + c]
                            ring[(slot * (self.window + 3) + rr) * self.width + c] = value
                        cute.arch.sync_threads()
                    if (not lazy) | (m > 0):
                        # Columns have unique owners. Each thread loads all three
                        # history seeds and any capture before replacing its column.
                        if cutlass.const_expr(self.vec):
                            self.publish_vectors(
                                x,
                                ring,
                                pending,
                                history,
                                slot,
                                b,
                                pb,
                                empty,
                                cursor,
                                end,
                                capture,
                                snap,
                                k,
                                tid,
                                workers,
                                lazy,
                            )
                        for c in range(tid, self.width if not self.vec else 0, workers):
                            h0, h1, h2 = BFloat16(0), BFloat16(0), BFloat16(0)
                            if c < self.dc:
                                if cutlass.const_expr(self.mode != "export_snapshot"):
                                    h0 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        end - 3,
                                        c,
                                    )
                                    h1 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        end - 2,
                                        c,
                                    )
                                    h2 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        end - 1,
                                        c,
                                    )
                                if capture:
                                    s0 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        k - 3,
                                        c,
                                    )
                                    s1 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        k - 2,
                                        c,
                                    )
                                    s2 = self.raw_value(
                                        x,
                                        history,
                                        ring,
                                        pending,
                                        slot,
                                        b,
                                        pb,
                                        empty,
                                        cursor,
                                        k - 1,
                                        c,
                                    )
                                    if cutlass.const_expr(self.dense_history):
                                        history[(snap * 3) * self.dc + c] = s0
                                    else:
                                        history[snap * self.h_stride + c] = s0
                                    if cutlass.const_expr(self.dense_history):
                                        history[(snap * 3 + 1) * self.dc + c] = s1
                                    else:
                                        history[snap * self.h_stride + self.dc + c] = s1
                                    if cutlass.const_expr(self.dense_history):
                                        history[(snap * 3 + 2) * self.dc + c] = s2
                                    else:
                                        history[snap * self.h_stride + 2 * self.dc + c] = s2
                                if cutlass.const_expr(self.mode != "export_snapshot"):
                                    if cutlass.const_expr(self.dense_history):
                                        history[(slot * 3) * self.dc + c] = h0
                                    else:
                                        history[slot * self.h_stride + c] = h0
                                    if cutlass.const_expr(self.dense_history):
                                        history[(slot * 3 + 1) * self.dc + c] = h1
                                    else:
                                        history[slot * self.h_stride + self.dc + c] = h1
                                    if cutlass.const_expr(self.dense_history):
                                        history[(slot * 3 + 2) * self.dc + c] = h2
                                    else:
                                        history[slot * self.h_stride + 2 * self.dc + c] = h2
                            if cutlass.const_expr(
                                not self.immediate and self.mode != "export_snapshot"
                            ):
                                if not lazy:
                                    ring[
                                        (slot * (self.window + 3) + self.window) * self.width + c
                                    ] = h0
                                    ring[
                                        (slot * (self.window + 3) + self.window + 1) * self.width
                                        + c
                                    ] = h1
                                    ring[
                                        (slot * (self.window + 3) + self.window + 2) * self.width
                                        + c
                                    ] = h2
                    cute.arch.sync_threads()
                    if tid == 0:
                        if capture:
                            sm = (snap - self.slots) * 4
                            snapshots[sm], snapshots[sm + 1] = pos + k, metadata[base + 6]
                            snapshots[sm + 2], snapshots[sm + 3] = epoch[0], Int64(1)
                            if cutlass.const_expr(self.dense_valid):
                                valid[b] = True
                            else:
                                valid[b * self.layout.valid] = True
                        if cutlass.const_expr(self.mode == "export_snapshot"):
                            if cutlass.const_expr(self.pb > 0):
                                first = Int32(1)
                                for j in range(b):
                                    if cutlass.const_expr(self.dense_indices):
                                        prior = Int64(indices[j * 2 + 1])
                                    else:
                                        prior = Int64(
                                            indices[Int64(j) * self.index_stride + self.layout.pair]
                                        )
                                    prior_active = prior != -1
                                    if cutlass.const_expr(self.trusted):
                                        prior_active = (prior >= self.slots) & (
                                            prior < self.slots + self.ns
                                        )
                                    if cutlass.const_expr(self.dense_indices):
                                        if (Int64(indices[j * 2]) == slot) & prior_active:
                                            first = Int32(0)
                                    else:
                                        if (
                                            Int64(indices[Int64(j) * self.index_stride]) == slot
                                        ) & prior_active:
                                            first = Int32(0)
                                if first != 0:
                                    pm[pb * 8 + 5], pm[pb * 8 + 6] = Int64(2), Int64(m)
                        else:
                            next_pos = pos + Int64(end)
                            if lazy:
                                if m > 0:
                                    metadata[base + 1] = next_pos
                                    metadata[base + 2] = Int64((cursor + m) % (self.window + 3))
                                    metadata[base + 3] = Int64(n + m)
                                    metadata[base + 4], metadata[base + 7] = Int64(1), epoch[0]
                            else:
                                metadata[base], metadata[base + 1] = next_pos, next_pos
                                metadata[base + 2], metadata[base + 3] = Int64(0), Int64(0)
                                metadata[base + 4], metadata[base + 7] = Int64(1), epoch[0]
                            if cutlass.const_expr(self.mode == "commit"):
                                pm[pb * 8 + 5], pm[pb * 8 + 6] = Int64(3), Int64(m)
