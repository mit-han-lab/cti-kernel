# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Ordered all-layer handoff control and disjoint, bit-preserving payload copies."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Boolean, Float32, Int32, Int64

from ._kernel import vector_load, vector_store
from ._phases import status_or


@cute.jit
def table_tensor(
    table, layer, field: cutlass.Constexpr, dtype: cutlass.Constexpr, size: cutlass.Constexpr
):
    ptr = cute.make_ptr(dtype, table[layer * 8 + field], cute.AddressSpace.gmem, assumed_align=16)
    return cute.make_tensor(ptr, cute.make_layout(size))


class Handoff:
    def __init__(
        self,
        hk,
        hv,
        window,
        batch,
        layers,
        slots,
        snapshots,
        record_stride,
        strict,
        chunks,
        ringless=False,
        pitches=(),
        index_stride=1,
        position_stride=1,
    ):
        self.ringless = ringless
        self.hv, self.window, self.batch, self.layers = hv, window, batch, layers
        self.slots, self.snapshots = slots, snapshots
        self.record_stride, self.strict, self.chunks = record_stride, strict, chunks
        self.dc = (2 * hk + hv) * 128
        self.width = self.dc + 2 * hv
        self.state_size = hv * 128 * 128
        self.pitches = (
            pitches or ((self.state_size, 3 * self.dc, self.state_size, 3 * self.dc),) * layers
        )
        self.max_pitches = tuple(max(p[i] for p in self.pitches) for i in range(4))
        self.index_stride, self.position_stride = index_stride, position_stride
        self.dense_indices = index_stride == 1
        self.dense_positions = position_stride == 1
        self.dense_checkpoint = all(p[0] == self.state_size for p in self.pitches)
        self.dense_history = all(p[1] == 3 * self.dc for p in self.pitches)
        self.dense_canonical = all(p[2] == self.state_size for p in self.pitches)
        self.dense_source = all(p[3] == 3 * self.dc for p in self.pitches)

    @cute.jit
    def __call__(
        self,
        table: cute.Tensor,
        positions: cute.Tensor,
        indices: cute.Tensor,
        records: cute.Tensor,
        sequence: cute.Tensor,
        poison: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.strict):
            self.control(table, positions, indices, records, sequence, poison).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream
            )
        if cutlass.const_expr(self.batch > 0):
            self.payload(table, positions, indices, records, sequence).launch(
                grid=(self.chunks, self.batch, self.layers),
                block=(256, 1, 1),
                use_pdl=True,
                stream=stream,
            )

    @cute.kernel
    def control(self, table, positions, indices, records, sequence, poison):
        tid, _, _ = cute.arch.thread_idx()
        error, inplace_error = Int32(0), Int32(0)
        # Two reductions amortize row validation over all layers. Invalid slots
        # contribute only bit 1, and padding contributes no error, even at p < 0.
        if poison[0] == 0:
            for b in range(tid, self.batch, 32):
                if cutlass.const_expr(self.dense_indices):
                    slot = Int64(indices[b])
                else:
                    slot = Int64(indices[Int64(b) * self.index_stride])
                if slot != -1:
                    if (slot < 0) | (slot >= self.slots):
                        error = error | Int32(1)
                        inplace_error = inplace_error | Int32(1)
                    else:
                        row_error = Int32(0)
                        if cutlass.const_expr(self.dense_positions):
                            if Int64(positions[b]) < 0:
                                row_error = Int32(2)
                        else:
                            if Int64(positions[Int64(b) * self.position_stride]) < 0:
                                row_error = Int32(2)
                        error = error | row_error
                        inplace_error = inplace_error | row_error
                        if slot != b:
                            inplace_error = inplace_error | Int32(1)
        for d in cutlass.range_constexpr(5):
            error = error | cute.arch.shuffle_sync_bfly(error, offset=16 >> d)
            inplace_error = inplace_error | cute.arch.shuffle_sync_bfly(
                inplace_error, offset=16 >> d
            )
        if tid == 0:
            words = cute.recast_tensor(records, Int32)
            longs = cute.recast_tensor(records, Int64)
            seq = sequence[0]
            for j in range(self.layers):
                base = j * (self.record_stride // 4)
                old, done = longs[j * (self.record_stride // 8)], words[base + 3]
                if (seq <= 0) | (old != seq) | (done != 0):
                    status_or(poison.iterator, Int32(64))
                elif poison[0] != 0:
                    status_or(words.iterator + base + 2, Int32(32))
                if cutlass.const_expr(self.batch > 0):
                    if poison[0] == 0:
                        code = error
                        if (table[j * 8 + 7] & Int64(1)) != 0:
                            code = inplace_error
                        status_or(words.iterator + base + 2, code)
                    status_or(poison.iterator, words[base + 2])

    @cute.kernel
    def payload(self, table, positions, indices, records, sequence):
        # This must precede EVERY global access, including the pointer table.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        chunk, b, j = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        allowed = Boolean(True)
        if cutlass.const_expr(self.strict):
            words = cute.recast_tensor(records, Int32)
            longs = cute.recast_tensor(records, Int64)
            seq = sequence[0]
            # Final poison cannot gate payload: a later layer may fail after an
            # earlier layer earned its write. Guard success plus final code zero
            # for the entire prefix exactly recovers that earlier permission.
            for i in range(j + 1):
                base = i * (self.record_stride // 4)
                allowed = allowed & (seq > 0) & (longs[i * (self.record_stride // 8)] == seq)
                allowed = allowed & (words[base + 3] == 0) & (words[base + 2] == 0)
        if cutlass.const_expr(self.dense_indices):
            slot = Int64(indices[b])
        else:
            slot = Int64(indices[Int64(b) * self.index_stride])
        inplace = (table[j * 8 + 7] & Int64(1)) != 0
        has_ring = (table[j * 8 + 7] & Int64(2)) == 0
        if cutlass.const_expr(self.dense_positions):
            active = allowed & (slot >= 0) & (slot < self.slots) & (Int64(positions[b]) >= 0)
        else:
            active = (
                allowed
                & (slot >= 0)
                & (slot < self.slots)
                & (Int64(positions[Int64(b) * self.position_stride]) >= 0)
            )
        active = active & ((not inplace) | (slot == b))
        if active:
            total = self.slots + self.snapshots
            if cutlass.const_expr(self.dense_checkpoint):
                checkpoint = table_tensor(table, j, 0, Float32, total * self.state_size)
            else:
                checkpoint = table_tensor(table, j, 0, Float32, total * self.max_pitches[0])
            if cutlass.const_expr(self.dense_history):
                history = table_tensor(table, j, 1, BFloat16, total * 3 * self.dc)
            else:
                history = table_tensor(table, j, 1, BFloat16, total * self.max_pitches[1])
            ring = table_tensor(table, j, 2, BFloat16, self.slots * (self.window + 3) * self.width)
            metadata = table_tensor(table, j, 3, Int64, self.slots * 8)
            epoch = table_tensor(table, j, 4, Int64, 1)
            if cutlass.const_expr(self.dense_canonical):
                canonical = table_tensor(table, j, 5, Float32, self.batch * self.state_size)
            else:
                canonical = table_tensor(table, j, 5, Float32, self.batch * self.max_pitches[2])
            if cutlass.const_expr(self.dense_source):
                source = table_tensor(table, j, 6, BFloat16, self.batch * 3 * self.dc)
            else:
                source = table_tensor(table, j, 6, BFloat16, self.batch * self.max_pitches[3])
            start = chunk * 256 + tid
            stride = self.chunks * 256
            if cutlass.const_expr(not self.dense_checkpoint):
                cp_pitch = Int64(self.pitches[0][0])
                for layer in cutlass.range_constexpr(1, self.layers):
                    if cutlass.const_expr(self.pitches[layer][0] != self.pitches[0][0]):
                        if j == layer:
                            cp_pitch = Int64(self.pitches[layer][0])
            if cutlass.const_expr(not self.dense_history):
                h_pitch = Int64(self.pitches[0][1])
                for layer in cutlass.range_constexpr(1, self.layers):
                    if cutlass.const_expr(self.pitches[layer][1] != self.pitches[0][1]):
                        if j == layer:
                            h_pitch = Int64(self.pitches[layer][1])
            if cutlass.const_expr(not self.dense_canonical):
                src_cp_pitch = Int64(self.pitches[0][2])
                for layer in cutlass.range_constexpr(1, self.layers):
                    if cutlass.const_expr(self.pitches[layer][2] != self.pitches[0][2]):
                        if j == layer:
                            src_cp_pitch = Int64(self.pitches[layer][2])
            if cutlass.const_expr(not self.dense_source):
                src_h_pitch = Int64(self.pitches[0][3])
                for layer in cutlass.range_constexpr(1, self.layers):
                    if cutlass.const_expr(self.pitches[layer][3] != self.pitches[0][3]):
                        if j == layer:
                            src_h_pitch = Int64(self.pitches[layer][3])
            if cutlass.const_expr(self.dense_canonical):
                src_cp = Int64(b) * self.state_size
            else:
                src_cp = Int64(b) * src_cp_pitch
            if cutlass.const_expr(self.dense_checkpoint):
                dst_cp = slot * self.state_size
            else:
                dst_cp = slot * cp_pitch
            if (canonical.iterator + src_cp).toint() != (checkpoint.iterator + dst_cp).toint():
                frag = cute.make_rmem_tensor((4,), Float32)
                for v in range(start, self.state_size // 4, stride):
                    vector_load(canonical, src_cp + v * 4, frag)
                    vector_store(checkpoint, dst_cp + v * 4, frag)
            if cutlass.const_expr(self.dense_source):
                src_h = Int64(b) * 3 * self.dc
            else:
                src_h = Int64(b) * src_h_pitch
            if cutlass.const_expr(self.dense_history):
                dst_h = slot * 3 * self.dc
            else:
                dst_h = slot * h_pitch
            copy_history = (source.iterator + src_h).toint() != (history.iterator + dst_h).toint()
            hfrag = cute.make_rmem_tensor((8,), BFloat16)
            for v in range(start, 3 * self.dc // 8, stride):
                row, col = v // (self.dc // 8), (v % (self.dc // 8)) * 8
                vector_load(source, src_h + v * 8, hfrag)
                if copy_history:
                    vector_store(history, dst_h + v * 8, hfrag)
                if cutlass.const_expr(not self.ringless):
                    if has_ring:
                        offset = (slot * (self.window + 3) + self.window + row) * self.width + col
                        if cutlass.const_expr(self.hv % 4 == 0):
                            vector_store(ring, offset, hfrag)
                        else:
                            # Non-multiple-of-eight BF16 ring strides are not 16B aligned.
                            for k in cutlass.range_constexpr(8):
                                ring[offset + k] = hfrag[k]
            if cutlass.const_expr(not self.ringless):
                if has_ring:
                    if cutlass.const_expr(self.hv % 4 == 0):
                        hfrag.fill(BFloat16(0))
                        for v in range(start, 3 * (2 * self.hv // 8), stride):
                            row, col = v // (2 * self.hv // 8), (v % (2 * self.hv // 8)) * 8
                            offset = (
                                (slot * (self.window + 3) + self.window + row) * self.width
                                + self.dc
                                + col
                            )
                            vector_store(ring, offset, hfrag)
                    else:
                        for v in range(start, 3 * 2 * self.hv, stride):
                            row, col = v // (2 * self.hv), v % (2 * self.hv)
                            ring[
                                (slot * (self.window + 3) + self.window + row) * self.width
                                + self.dc
                                + col
                            ] = BFloat16(0)
            if (chunk == 0) & (tid == 0):
                base = slot * 8
                current_epoch = epoch[0]
                fresh = (metadata[base + 4] != 1) | (metadata[base + 7] != current_epoch)
                generation = metadata[base + 5] + Int64(fresh)
                if cutlass.const_expr(self.dense_positions):
                    position = Int64(positions[b])
                else:
                    position = Int64(positions[Int64(b) * self.position_stride])
                metadata[base] = position
                metadata[base + 1] = position
                metadata[base + 2] = Int64(0)
                metadata[base + 3] = Int64(0)
                metadata[base + 4] = Int64(1)
                metadata[base + 5] = generation
                metadata[base + 7] = current_epoch
