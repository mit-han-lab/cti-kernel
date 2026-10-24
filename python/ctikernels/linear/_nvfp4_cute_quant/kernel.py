# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Recipe follows FlashInfer 0.7.0.post1 (csrc/nv_internal/tensorrt_llm/kernels/
# quantization_utils.cuh, cvt_warp_fp16_to_fp4), design reference only (Apache-2.0).

"""Original CuTe NVFP4 quantizers. Two adjacent lanes own a 16-value scale block.

Each lane loads eight BF16/FP16 activations (sixteen interleaved gate/up values),
then writes four packed bytes. Only the NaN-ignoring amax crosses lanes. Larger
batches use whole-row CTAs with all lane loads issued before quantization and
an occupancy-sized grid striding over rows. Padding is written separately,
without quantizing fictitious rows.
Scale/pack follows the market recipe in _nvfp4_cute_recipe.
Inline assembly uses the public PTX ISA (div, rcp, mul, setp, cvt, add); no external kernel
implementation is incorporated here.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr

from .._nvfp4_cute_recipe import maximum as _maximum
from .._nvfp4_cute_recipe import mul as _mul
from .._nvfp4_cute_recipe import pack as _pack
from .._nvfp4_cute_recipe import reciprocal as _reciprocal
from .._nvfp4_cute_recipe import scale as _scale
from . import _launch_bounds


@cute.jit
def _silu_product(gate: Float32, up: Float32):
    # SGLang's activation: gate / (1 + expf(-gate)) * up (accurate exp, IEEE divide).
    silu = cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[gate, Float32(1.0) + cute.math.exp(-gate, fastmath=False)],
    )
    return silu * up


@cute.jit
def _turns(start: Int64, stop: Int64, step: Int64):
    # Grid-stride trip count. A unit-step loop avoids the DSL's dynamic-sign step
    # rewrite, whose start/stop selects mix integer widths.
    return cutlass.max((stop - start + step - Int64(1)) // step, Int64(0))


@cute.jit
def _offset(row: Int64, block: Int32, atoms_k: cutlass.Constexpr):
    local_row = Int32(row % Int64(128))
    return (
        (row // Int64(128)) * Int64(atoms_k * 512)
        + Int64(block // Int32(4)) * Int64(512)
        + Int64(
            (local_row % Int32(32)) * Int32(16)
            + (local_row // Int32(32)) * Int32(4)
            + block % Int32(4)
        )
    )


@cute.jit
def _fill(target: cute.Tensor, offset: Int64, value: Uint8, size: cutlass.Constexpr):
    packed = cute.make_rmem_tensor(size, Uint8)
    for i in cutlass.range_constexpr(size):
        packed[i] = value
    ptr = cute.make_ptr(
        Uint8, target.iterator.toint() + offset, cute.AddressSpace.gmem, assumed_align=size
    )
    dst = cute.make_tensor(ptr, cute.make_layout(size))
    atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), Uint8, num_bits_per_copy=size * 8)
    cute.copy(atom, packed, dst)


@cute.jit
def _columns(
    column: cute.Tensor, offset: Int64, gate: Float32, up: Float32, alignment: cutlass.Constexpr
):
    values = cute.make_rmem_tensor(4, Float32)
    values[0], values[1] = gate, up
    values[2], values[3] = gate, up
    ptr = cute.make_ptr(
        Float32,
        column.iterator.toint() + offset * Int64(4),
        cute.AddressSpace.gmem,
        assumed_align=alignment,
    )
    dst = cute.make_tensor(ptr, cute.make_layout(4))
    atom = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), Float32, num_bits_per_copy=alignment * 8)
    cute.copy(atom, values, dst)


class Kernel:
    def __init__(
        self,
        cols,
        gated,
        flags,
        columns,
        clear,
        aligned_x,
        aligned_out,
        geometry,
        gated_row_blocks=False,
        early_trigger=False,
        pdl=True,
    ):
        self.cols, self.gated, self.flags = cols, gated, flags
        self.early_trigger = early_trigger
        self.pdl = pdl
        self.gated_row_blocks = gated_row_blocks
        self.columns, self.clear = columns, clear
        # Scalar side outputs (alpha, reciprocal, column scales) feed GEMM epilogues, some of
        # which read them before their own grid-dependency wait (FlashInfer's sm120 CUTLASS FP4
        # GEMM, always launched with PDL, reads alpha so). Their dependents launch at exit.
        self.trigger = not (flags[2] or flags[4] or columns)
        self.aligned_x, self.aligned_out = aligned_x, aligned_out
        (self.threads, self.threads_per_row, self.groups, self.rows_per_cta, self.row_loop) = (
            geometry
        )
        self.ctas_per_sm = max(1, 2048 // self.threads)
        self.min_blocks = _launch_bounds(self.threads) if self.row_loop else 0
        self.atoms_k = (cols + 63) // 64
        self.source_count = 16 if gated else 8
        self.input_alignment, self.load_bits = (16, 128) if aligned_x else (2, 16)
        self.output_alignment, self.store_bits = (4, 32) if aligned_out else (1, 8)
        self.missing = self.atoms_k * 4 - cols // 16
        self.halves = cols // 8
        self.tiles_per_row = (self.halves + self.threads * self.groups - 1) // (
            self.threads * self.groups
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        data: cute.Tensor,
        scale: cute.Tensor,
        global_scale: cute.Tensor,
        weight: cute.Tensor,
        up_weight: cute.Tensor,
        alpha: cute.Tensor,
        output: cute.Tensor,
        reciprocal: cute.Tensor,
        column: cute.Tensor,
        gate_factor: cute.Tensor,
        up_factor: cute.Tensor,
        clear: cute.Tensor,
        scale_alignment: Int32,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        rows = Int64(x.shape[0])
        row_chunks = (rows + Int64(self.rows_per_cta - 1)) // Int64(self.rows_per_cta)
        quant_ctas = row_chunks * Int64(self.tiles_per_row)
        if const_expr(self.row_loop):
            quant_ctas = Int64(cutlass.min(rows, Int64(sm_count) * Int64(self.ctas_per_sm)))
        # One padding task owns a 16-byte chunk; omit chunks with no padded rows.
        remainder = Int32(rows % Int64(128))
        first_chunk = Int32(cutlass.max(remainder - Int32(96), Int32(0)))
        chunks = Int32(cutlass.select_(remainder == Int32(0), Int32(0), Int32(32) - first_chunk))
        padding_tasks = Int64(chunks) * Int64(self.atoms_k)
        tail_tasks = Int64(0)
        if const_expr(self.missing):
            tail_tasks = rows
        pad_ctas = (padding_tasks + tail_tasks + Int64(self.threads - 1)) // Int64(self.threads)
        if const_expr(self.flags[2] or self.flags[4] or self.columns or self.clear):
            pad_ctas = Int64(cutlass.max(pad_ctas, Int64(1)))
        self.kernel(
            x,
            data,
            scale,
            global_scale,
            weight,
            up_weight,
            alpha,
            output,
            reciprocal,
            column,
            gate_factor,
            up_factor,
            clear,
            scale_alignment,
            rows,
            quant_ctas,
            padding_tasks,
            first_chunk,
            chunks,
        ).launch(
            grid=[quant_ctas + pad_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        data: cute.Tensor,
        scale: cute.Tensor,
        global_scale: cute.Tensor,
        weight: cute.Tensor,
        up_weight: cute.Tensor,
        alpha: cute.Tensor,
        output: cute.Tensor,
        reciprocal: cute.Tensor,
        column: cute.Tensor,
        gate_factor: cute.Tensor,
        up_factor: cute.Tensor,
        clear: cute.Tensor,
        scale_alignment: Int32,
        rows: Int64,
        quant_ctas: Int64,
        padding_tasks: Int64,
        first_chunk: Int32,
        chunks: Int32,
    ):
        cute.arch.griddepcontrol_wait()
        if const_expr(self.trigger and self.early_trigger):
            # Large grids: let the consumer's prologue overlap our tail. Consumers still
            # wait for this entire grid before reading outputs or overwriting x.
            cute.arch.griddepcontrol_launch_dependents()
        tid = Int32(cute.arch.thread_idx()[0])
        bid = Int64(cute.arch.block_idx()[0])
        inverse = _reciprocal(Float32(global_scale[0]))
        gf, uf = Float32(1.0), Float32(1.0)
        if const_expr(self.flags[6]):
            gf, uf = Float32(gate_factor[0]), Float32(up_factor[0])
        if bid < quant_ctas:
            if const_expr(self.row_loop):
                for turn in cutlass.range(_turns(bid, rows, quant_ctas)):
                    row = bid + Int64(turn) * quant_ctas
                    self._quantize_row(x, data, scale, inverse, gf, uf, row, tid)
            else:
                row = (bid // Int64(self.tiles_per_row)) * Int64(self.rows_per_cta)
                tile = Int32(bid % Int64(self.tiles_per_row))
                self._quantize_tile(x, data, scale, inverse, gf, uf, row, tile, tid, rows)
        else:
            pad_bid = bid - quant_ctas
            index = pad_bid * Int64(self.threads) + Int64(tid)
            if pad_bid == Int64(0):
                self._side_outputs(
                    global_scale, weight, up_weight, alpha, output, reciprocal, column, clear, tid
                )
            # FlashInfer writes literal zero to padded scale rows/columns.
            pad_byte = Uint8(0)
            if index < padding_tasks:
                self._padding(scale, rows, index, first_chunk, chunks, pad_byte, scale_alignment)
            if const_expr(self.missing):
                tail_row = index - padding_tasks
                if (tail_row >= Int64(0)) & (tail_row < rows):
                    self._tail(scale, tail_row, pad_byte, scale_alignment)
        if const_expr(self.trigger and not self.early_trigger):
            # Smaller grids trigger after this CTA's work: an entry trigger let consumer
            # GEMM CTAs occupy SMs while quantizer CTAs were still unscheduled.
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def _side_outputs(
        self,
        global_scale: cute.Tensor,
        weight: cute.Tensor,
        up_weight: cute.Tensor,
        alpha: cute.Tensor,
        output: cute.Tensor,
        reciprocal: cute.Tensor,
        column: cute.Tensor,
        clear: cute.Tensor,
        tid: Int32,
    ):
        if tid == Int32(0):
            if const_expr(self.flags[2]):
                alpha[0] = Float32(global_scale[0]) * Float32(weight[0])
            if const_expr(self.flags[4]):
                reciprocal[0] = _reciprocal(Float32(output[0]))
        if const_expr(self.columns):
            weight_value = Float32(weight[0])
            up_value = weight_value
            if const_expr(self.flags[1]):
                up_value = Float32(up_weight[0])
            # Four adjacent FP32 columns per lane, including alternating gate/up.
            for turn in cutlass.range(
                _turns(Int64(tid) * Int64(4), Int64(self.columns), Int64(self.threads * 4))
            ):
                col = (Int64(tid) + Int64(turn) * Int64(self.threads)) * Int64(4)
                if col + Int64(4) <= Int64(self.columns):
                    if column.iterator.toint() % Int64(16) == Int64(0):
                        _columns(column, col, weight_value, up_value, 16)
                    else:
                        _columns(column, col, weight_value, up_value, 4)
                else:
                    for j in cutlass.range_constexpr(4):
                        if col + Int64(j) < Int64(self.columns):
                            if const_expr(j % 2 == 0):
                                column[col + Int64(j)] = weight_value
                            else:
                                column[col + Int64(j)] = up_value
        if const_expr(self.clear):
            # Runtime length: the padded scale buffer grows per 128-row bucket.
            clear_size = Int64(clear.shape[0])
            for turn in cutlass.range(
                _turns(Int64(tid) * Int64(16), clear_size, Int64(self.threads * 16))
            ):
                pos = (Int64(tid) + Int64(turn) * Int64(self.threads)) * Int64(16)
                if pos + Int64(16) <= clear_size:
                    if clear.iterator.toint() % Int64(16) == Int64(0):
                        _fill(clear, pos, Uint8(0), 16)
                    else:
                        if clear.iterator.toint() % Int64(4) == Int64(0):
                            for j in cutlass.range_constexpr(4):
                                _fill(clear, pos + Int64(j * 4), Uint8(0), 4)
                        else:
                            for j in cutlass.range_constexpr(16):
                                clear[pos + Int64(j)] = Uint8(0)
                else:
                    for j in cutlass.range_constexpr(16):
                        if pos + Int64(j) < clear_size:
                            clear[pos + Int64(j)] = Uint8(0)

    @cute.jit
    def _padding(
        self,
        scale: cute.Tensor,
        rows: Int64,
        index: Int64,
        first_chunk: Int32,
        chunks: Int32,
        value: Uint8,
        alignment: Int32,
    ):
        # Narrow only bounded local indices; all global byte offsets stay i64.
        if const_expr(self.atoms_k * 32 < 2**31):
            item = Int32(index)
            chunk = item % Int32(32)
            atom = item // Int32(32)
            if chunks != Int32(32):
                chunk = item % chunks + first_chunk
                atom = item // chunks
        else:
            chunk = Int32(index % Int64(chunks)) + first_chunk
            atom = Int32(index // Int64(chunks))
        address = (rows // Int64(128)) * Int64(self.atoms_k * 512)
        address = address + Int64(atom) * Int64(512) + Int64(chunk * Int32(16))
        remainder = Int32(rows % Int64(128))
        if (chunk >= remainder) & (alignment >= Int32(16)):
            _fill(scale, address, value, 16)
        else:
            for j in cutlass.range_constexpr(4):
                if chunk + Int32(j * 32) >= remainder:
                    if alignment >= Int32(4):
                        _fill(scale, address + Int64(j * 4), value, 4)
                    else:
                        for b in cutlass.range_constexpr(4):
                            scale[address + Int64(j * 4 + b)] = value

    @cute.jit
    def _tail(self, scale: cute.Tensor, row: Int64, value: Uint8, alignment: Int32):
        address = _offset(row, Int32(self.cols // 16), self.atoms_k)
        if const_expr(self.missing == 1):
            scale[address] = value
        else:
            # Missing 2/3 bytes end at a four-byte boundary, so the last pair
            # admits one aligned b16 store without racing a real block's scale.
            if const_expr(self.missing == 3):
                scale[address] = value
            if alignment >= Int32(4):
                _fill(scale, address + Int64(self.missing - 2), value, 2)
            else:
                for j in cutlass.range_constexpr(2):
                    scale[address + Int64(self.missing - 2 + j)] = value

    @cute.jit
    def _quantize_row(
        self,
        x: cute.Tensor,
        data: cute.Tensor,
        scale: cute.Tensor,
        inverse: Float32,
        gf: Float32,
        uf: Float32,
        row: Int64,
        tid: Int32,
    ):
        source = cute.make_rmem_tensor((self.source_count, self.groups), x.element_type)
        # Keep only native 16-bit inputs live across loads (1..5 chunks for
        # K <= 17408); convert one eight-value chunk at a time, with no FP32 row
        # buffer or second row in flight. Only the final chunk needs a lane mask.
        for group in cutlass.range_constexpr(self.groups):
            half_block = tid + Int32(group * self.threads)
            if const_expr((group + 1) * self.threads <= self.halves):
                self._load_lane(x, source[None, group], row, half_block)
            else:
                if half_block < Int32(self.halves):
                    self._load_lane(x, source[None, group], row, half_block)
        for group in cutlass.range_constexpr(self.groups):
            half_block = tid + Int32(group * self.threads)
            if const_expr((group + 1) * self.threads <= self.halves):
                self._quantize_lane(
                    source[None, group], data, scale, inverse, gf, uf, row, half_block, tid
                )
            else:
                if half_block < Int32(self.halves):
                    self._quantize_lane(
                        source[None, group], data, scale, inverse, gf, uf, row, half_block, tid
                    )

    @cute.jit
    def _quantize_tile(
        self,
        x: cute.Tensor,
        data: cute.Tensor,
        scale: cute.Tensor,
        inverse: Float32,
        gf: Float32,
        uf: Float32,
        base_row: Int64,
        tile: Int32,
        tid: Int32,
        rows: Int64,
    ):
        row = base_row
        half = tile * Int32(self.threads * self.groups) + tid
        if const_expr(self.rows_per_cta > 1):
            row = base_row + Int64(tid // Int32(self.halves))
            half = tid % Int32(self.halves)
        active = (row < rows) & (tid < Int32(self.rows_per_cta * self.halves))
        source = cute.make_rmem_tensor((self.source_count, self.groups), x.element_type)
        # Issue independent vector loads before any reduction/conversion work.
        for group in cutlass.range_constexpr(self.groups):
            half_block = half + Int32(group * self.threads)
            if active & (half_block < Int32(self.halves)):
                self._load_lane(x, source[None, group], row, half_block)
        for group in cutlass.range_constexpr(self.groups):
            half_block = half + Int32(group * self.threads)
            if active & (half_block < Int32(self.halves)):
                self._quantize_lane(
                    source[None, group], data, scale, inverse, gf, uf, row, half_block, tid
                )

    @cute.jit
    def _load_lane(self, x: cute.Tensor, source: cute.Tensor, row: Int64, half_block: Int32):
        if const_expr(self.gated_row_blocks):
            # Two contiguous [M, I] GEMM outputs in one allocation, no interleave copy.
            for slot in cutlass.range_constexpr(2):
                offset = (Int64(slot) * Int64(x.shape[0]) + row) * Int64(self.cols)
                offset = offset + Int64(half_block) * Int64(8)
                ptr = cute.make_ptr(
                    x.element_type,
                    x.iterator.toint() + offset * Int64(2),
                    cute.AddressSpace.gmem,
                    assumed_align=self.input_alignment,
                )
                src = cute.make_tensor(ptr, cute.make_layout(8))
                dst = cute.make_tensor(source.iterator + slot * 8, cute.make_layout(8))
                atom = cute.make_copy_atom(
                    cute.nvgpu.CopyG2ROp(), x.element_type, num_bits_per_copy=self.load_bits
                )
                cute.copy(atom, src, dst)
        else:
            self._load_interleaved_lane(x, source, row, half_block)

    @cute.jit
    def _load_interleaved_lane(
        self, x: cute.Tensor, source: cute.Tensor, row: Int64, half_block: Int32
    ):
        offset = row * Int64(x.stride[0]) + Int64(half_block) * Int64(self.source_count)
        ptr = cute.make_ptr(
            x.element_type,
            x.iterator.toint() + offset * Int64(2),
            cute.AddressSpace.gmem,
            assumed_align=self.input_alignment,
        )
        src = cute.make_tensor(ptr, cute.make_layout(self.source_count))
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(), x.element_type, num_bits_per_copy=self.load_bits
        )
        cute.copy(atom, src, source)

    @cute.jit
    def _quantize_lane(
        self,
        source: cute.Tensor,
        data: cute.Tensor,
        scale: cute.Tensor,
        inverse: Float32,
        gf: Float32,
        uf: Float32,
        row: Int64,
        half_block: Int32,
        tid: Int32,
    ):
        values = cute.make_rmem_tensor(8, Float32)
        for i in cutlass.range_constexpr(8):
            if const_expr(self.gated):
                if const_expr(self.gated_row_blocks):
                    gate, up = source[i].to(Float32), source[i + 8].to(Float32)
                else:
                    gate, up = source[2 * i].to(Float32), source[2 * i + 1].to(Float32)
                if const_expr(self.flags[6]):
                    gate = (gate * gf).to(BFloat16).to(Float32)
                    up = (up * uf).to(BFloat16).to(Float32)
                values[i] = _silu_product(gate, up)
            else:
                values[i] = source[i].to(Float32)
        maximum = cute.math.abs(Float32(values[0]))
        for i in cutlass.range_constexpr(8):
            v = Float32(values[i])
            magnitude = cute.math.abs(v)
            maximum = _maximum(maximum, magnitude)
        pair = Uint32(3) << Uint32(tid % 32 // 2 * 2)
        maximum = _maximum(maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1, mask=pair))
        sf, factor = _scale(maximum, inverse)
        packed = cute.make_rmem_tensor(4, Uint8)
        for i in cutlass.range_constexpr(4):
            packed[i] = _pack(_mul(values[2 * i], factor), _mul(values[2 * i + 1], factor))
        target = row * Int64(data.stride[0]) + Int64(half_block) * Int64(4)
        ptr_out = cute.make_ptr(
            Uint8,
            data.iterator.toint() + target,
            cute.AddressSpace.gmem,
            assumed_align=self.output_alignment,
        )
        dst = cute.make_tensor(ptr_out, cute.make_layout(4))
        store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), Uint8, num_bits_per_copy=self.store_bits
        )
        cute.copy(store, packed, dst)
        if half_block % Int32(2) == Int32(0):
            scale[_offset(row, half_block // Int32(2), self.atoms_k)] = sf.bitcast(Uint8)
