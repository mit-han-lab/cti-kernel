# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Original swap-AB BF16 MMA kernel. W is MMA A; x rows are MMA B in 8-row tiles.

mtiles (compile time, 1..8) covers M <= 8*mtiles; each W fragment feeds every x tile.
Nonzero gate_stride adds an FP32 sigmoid/product prologue with one BF16 rounding, the
instruction sequence of ``linear._gate_ptx`` shared with the unfused Triton gate.

128-bit cp.async loads feed a shared-memory ring; each x tile is loaded once per
CTA. Padding each shared row by eight BF16s avoids ldmatrix bank conflicts.
The reg variant loads W fragments straight into registers (8 rows x 64 bytes per
warp load); the wide variant loads 4 rows x 128 bytes and rebuilds the same
fragments with one xor-4 shuffle (GB10 streams wider rows faster).
With Config.cluster the split CTAs of one N tile form a thread-block cluster and CTA r
owns rows [r*rows/S, (r+1)*rows/S) of the N tile; after one cluster barrier each owner
sums its rows over the splits in split order and writes the output, in one launch.
L2 mode reads the partials from the split-K scratch. DSMEM mode needs no scratch: each
CTA stages its partials in local shared memory grouped by owner and copies each owner's
contiguous run into that CTA's shared memory with 16-byte remote stores (mapa +
st.shared::cluster). DSMEM traffic grows with M (slow all-to-all), so it suits small M.
Instruction and fragment definitions: NVIDIA public PTX ISA, sections
mma.m16n8k16 (floating point), ldmatrix, cp.async, shfl.sync, mapa, st.shared::cluster,
ld.global.cg and barrier.cluster. No vendor kernel is copied.
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm

from .._dispatch import DSMEM, L2, NONE
from .._gate_ptx import gate_ptx


@cute.jit
def _copy16(dst: cute.Pointer, src: cute.Pointer, offset: Int64, valid):
    # src-size=0 zero-fills padded N, K and M; use a valid base even for masked loads.
    address = src.toint() + (offset if valid else Int64(0)) * 2
    size = Int32(16) if valid else Int32(0)
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[Int32(dst.toint()), address, size],
    )


def _store_cluster(address, rank, values):
    """1/2/4 FP32 values (one vector) to shared offset ``address`` of cluster CTA ``rank``;
    side effects keep the store between the cluster barriers."""
    width = len(values)
    vector = "" if width == 1 else f".v{width}"
    operands = ", ".join(f"${i + 2}" for i in range(width))
    operands = operands if width == 1 else "{" + operands + "}"
    llvm.inline_asm(
        None,
        [Int32(address).ir_value(), Int32(rank).ir_value()]
        + [Float32(v).ir_value() for v in values],
        "{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; "
        f"st.shared::cluster{vector}.f32 [a], {operands}; }}",
        "r,r" + ",f" * width + ",~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _load_l2(src, offset):
    """FP32 global load at L2 (.cg) of another CTA's partial, after the cluster barrier."""
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [Int64(src.toint() + offset * 4).ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _mma(a, b, c):
    return cute.arch.inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]],
    )


@cute.jit
def _load_w16(src: cute.Pointer, offset: Int64, valid):
    # K is a multiple of eight at the provider boundary: each 16-byte vector
    # is wholly valid or wholly padding, including the final 32-element block.
    address = src.toint() + (offset if valid else Int64(0)) * 2
    return cute.arch.inline_ptx(
        "{ .reg .pred p; setp.ne.b32 p, {$r1}, 0; "
        "mov.b32 {$w0}, 0; mov.b32 {$w1}, 0; mov.b32 {$w2}, 0; mov.b32 {$w3}, 0; "
        "@p ld.global.nc.L1::no_allocate.v4.b32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}]; }",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[address, Int32(valid)],
    )


@cute.jit
def _load_w16_l2(src: cute.Pointer, offset: Int64, valid):
    # Plain (L2-allocating) load for the wide variant: on GB10, L1::no_allocate
    # weight lines do not stay L2-resident for the next call that reads them.
    address = src.toint() + (offset if valid else Int64(0)) * 2
    return cute.arch.inline_ptx(
        "{ .reg .pred p; setp.ne.b32 p, {$r1}, 0; "
        "mov.b32 {$w0}, 0; mov.b32 {$w1}, 0; mov.b32 {$w2}, 0; mov.b32 {$w3}, 0; "
        "@p ld.global.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}]; }",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[address, Int32(valid)],
    )


@cute.jit
def _exchange(odd, even_word, odd_word):
    # Lanes L and L^4 own the two 64-byte halves of the same rows. Each keeps
    # one word, sends the other, and returns (K[0:32] block word, K[32:64] word).
    return cute.arch.inline_ptx(
        "{ .reg .pred p; .reg .b32 s, k, r; setp.ne.b32 p, {$r0}, 0; "
        "selp.b32 s, {$r1}, {$r2}, p; selp.b32 k, {$r2}, {$r1}, p; "
        "shfl.sync.bfly.b32 r, s, 4, 0x1f, 0xffffffff; "
        "selp.b32 {$w0}, r, k, p; selp.b32 {$w1}, k, r, p; }",
        write_only_types=[Int32, Int32],
        read_only_args=[odd, even_word, odd_word],
    )


_GATE_PAIR = (
    "{ .reg .b32 bits; .reg .f32 x0, x1, g0, g1, s0, s1; .reg .pred p0, p1; "
    "shl.b32 bits, {$r0}, 16; mov.b32 x0, bits; "
    "and.b32 bits, {$r0}, 0xffff0000; mov.b32 x1, bits; "
    "shl.b32 bits, {$r1}, 16; mov.b32 g0, bits; "
    "and.b32 bits, {$r1}, 0xffff0000; mov.b32 g1, bits; "
    + gate_ptx(("x0", "g0", "s0", "p0"), ("x1", "g1", "s1", "p1"))
    + "cvt.rn.bf16x2.f32 {$w0}, x1, x0; }"
)


@cute.jit
def _gate_pair(x, gate):
    # Expand BF16 bits exactly; the shared gate sequence; round once. First cvt source is high.
    return cute.arch.inline_ptx(_GATE_PAIR, write_only_types=[Int32], read_only_args=[x, gate])


@cute.jit
def _gate16(x: cute.Pointer, gate: cute.Pointer):
    xv = cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[Int32(x.toint())],
    )
    gv = cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[Int32(gate.toint())],
    )
    values = cute.make_rmem_tensor(4, Int32)
    for i in cutlass.range_constexpr(4):
        values[i] = _gate_pair(xv[i], gv[i])
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[Int32(x.toint()), values[0], values[1], values[2], values[3]],
    )


class Kernel:
    def __init__(
        self, n, k, config, streaming=False, mtiles=1, gate_stride=0, fp32=False, out_stride=0
    ):
        self.n, self.k, self.config, self.mtiles = n, k, config, mtiles
        # Output row stride; 0 (contiguous) keeps the measured offset arithmetic.
        self.ldy = out_stride or n
        self.fp32 = fp32
        self.gate_stride = gate_stride
        if gate_stride and config.variant == "wide":
            raise ValueError("the wide variant has no gated prologue")
        self.xrows = 8 * mtiles
        self.chunks = (k + config.splits * config.tile_k - 1) // (config.splits * config.tile_k)
        # Two adjacent outputs per lane need aligned pairs in every split.
        # Keep sm120 and odd-N launches identical to their measured versions.
        self.vector_reduce = (streaming or fp32) and n % 2 == 0
        self.cluster = config.cluster

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        y: cute.Tensor,
        partial: cute.Tensor,
        stream: cuda.CUstream,
        gate=None,
    ):
        if const_expr(self.gate_stride != 0):
            self.launch_gated(x, w, y, partial, stream, gate)
        elif const_expr(self.config.variant == "reg"):
            self.main_reg(x.iterator, w.iterator, y.iterator, partial.iterator, x.shape[0]).launch(
                grid=[cute.ceil_div(self.n, self.config.rows), self.config.splits, 1],
                block=[self.config.warps * 32, 1, 1],
                cluster=self.cluster_shape(),
                smem=self.config.shared_bytes_for(self.mtiles),
                stream=stream,
            )
        elif const_expr(self.config.variant == "wide"):
            self.main_wide(x.iterator, w.iterator, y.iterator, partial.iterator, x.shape[0]).launch(
                grid=[cute.ceil_div(self.n, self.config.rows), self.config.splits, 1],
                block=[self.config.warps * 32, 1, 1],
                cluster=self.cluster_shape(),
                smem=self.config.shared_bytes_for(self.mtiles),
                stream=stream,
            )
        else:
            self.launch_smem(x, w, y, partial, stream)
        if const_expr(self.config.splits > 1 and self.cluster == NONE):
            self.launch_reduce(x, y, partial, stream)

    def cluster_shape(self):
        return [1, self.config.splits, 1] if self.cluster else None

    @cute.jit
    def launch_gated(self, x, w, y, partial, stream, gate):
        if const_expr(self.config.variant == "reg"):
            self.main_reg(
                x.iterator, w.iterator, y.iterator, partial.iterator, x.shape[0], gate.iterator
            ).launch(
                grid=[cute.ceil_div(self.n, self.config.rows), self.config.splits, 1],
                block=[self.config.warps * 32, 1, 1],
                cluster=self.cluster_shape(),
                smem=self.config.shared_bytes_for(self.mtiles, gated=True),
                stream=stream,
            )
        else:
            self.main(
                x.iterator, w.iterator, y.iterator, partial.iterator, x.shape[0], gate.iterator
            ).launch(
                grid=[cute.ceil_div(self.n, self.config.rows), self.config.splits, 1],
                block=[self.config.warps * 32, 1, 1],
                cluster=self.cluster_shape(),
                smem=self.config.shared_bytes_for(self.mtiles, gated=True),
                stream=stream,
            )

    @cute.jit
    def launch_smem(self, x, w, y, partial, stream):
        self.main(x.iterator, w.iterator, y.iterator, partial.iterator, x.shape[0]).launch(
            grid=[cute.ceil_div(self.n, self.config.rows), self.config.splits, 1],
            block=[self.config.warps * 32, 1, 1],
            cluster=self.cluster_shape(),
            # SmemAllocator infers this too; keep the launch contract explicit,
            # including configurations above the default 48 KiB limit.
            smem=self.config.shared_bytes_for(self.mtiles),
            stream=stream,
        )

    @cute.jit
    def launch_reduce(self, x, y, partial, stream):
        if const_expr(self.config.splits > 1):
            if const_expr(self.vector_reduce):
                # 64 outputs/CTA gives 64 CTAs even for N=4096, M=1.
                self.reduce_pairs(partial.iterator, y.iterator, x.shape[0]).launch(
                    grid=[cute.ceil_div(self.n * x.shape[0], 64), 1, 1],
                    block=[32, 1, 1],
                    stream=stream,
                )
            else:
                self.reduce(partial.iterator, y.iterator, x.shape[0]).launch(
                    grid=[cute.ceil_div(self.n * x.shape[0], 256), 1, 1],
                    block=[256, 1, 1],
                    stream=stream,
                )

    @cute.jit
    def stage(self, sw, sx, x, w, m, ntile, split, chunk, tid, gate=None):
        cfg = self.config
        pitch = cfg.tile_k + 8
        base = (split * self.chunks + chunk) * cfg.tile_k
        for j in cutlass.range_constexpr(cute.ceil_div(cfg.rows * cfg.tile_k // 8, cfg.warps * 32)):
            vec = tid + j * cfg.warps * 32
            row, col = vec // (cfg.tile_k // 8), vec % (cfg.tile_k // 8) * 8
            nr = ntile * cfg.rows + row
            _copy16(
                sw + row * pitch + col,
                w,
                Int64(nr) * self.k + base + col,
                (nr < self.n) & (base + col < self.k) & (chunk < self.chunks),
            )
        for j in cutlass.range_constexpr(
            cute.ceil_div(self.xrows * cfg.tile_k // 8, cfg.warps * 32)
        ):
            vec = tid + j * cfg.warps * 32
            row, col = vec // (cfg.tile_k // 8), vec % (cfg.tile_k // 8) * 8
            if row < self.xrows:
                _copy16(
                    sx + row * pitch + col,
                    x,
                    Int64(row) * self.k + base + col,
                    (row < m) & (base + col < self.k) & (chunk < self.chunks),
                )
                if const_expr(self.gate_stride != 0):
                    _copy16(
                        sx + self.xrows * pitch + row * pitch + col,
                        gate,
                        Int64(row) * self.gate_stride + base + col,
                        (row < m) & (base + col < self.k) & (chunk < self.chunks),
                    )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def transform(self, sx, m, tid):
        cfg = self.config
        pitch = cfg.tile_k + 8
        for j in cutlass.range_constexpr(
            cute.ceil_div(self.xrows * cfg.tile_k // 8, cfg.warps * 32)
        ):
            vec = tid + j * cfg.warps * 32
            row, col = vec // (cfg.tile_k // 8), vec % (cfg.tile_k // 8) * 8
            if (row < self.xrows) & (row < m):
                address = sx + row * pitch + col
                _gate16(address, address + self.xrows * pitch)
        cute.arch.sync_threads()

    @cute.kernel
    def main(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        y: cute.Pointer,
        partial: cute.Pointer,
        m: Int32,
        gate=None,
    ):
        cfg = self.config
        tid, _, _ = cute.arch.thread_idx()
        ntile, split, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        pitch = cfg.tile_k + 8
        stage_size = (cfg.rows + self.xrows) * pitch
        if const_expr(self.gate_stride != 0):
            stage_size += self.xrows * pitch
        allocator = utils.SmemAllocator()
        smem = allocator.allocate_tensor(
            BFloat16, cute.make_layout(cfg.stages * stage_size), byte_alignment=16
        )
        red = self.partial_tile(allocator)
        if const_expr(self.cluster == DSMEM):
            # Peers must be running before their shared memory is written; this
            # phase completes long before the epilogue waits on it.
            cute.arch.cluster_arrive_relaxed()
        acc = cute.make_rmem_tensor((cfg.rows // (16 * cfg.warps), self.mtiles, 4), Float32)
        acc.fill(0.0)
        bfrag = cute.make_rmem_tensor((self.mtiles, 2), Int32)
        # Config admits up to six stages. Commit even masked tail stages so
        # wait_group(stages-1) always retires the group about to be consumed.
        for s in cutlass.range_constexpr(cfg.stages - 1):
            prefetch = smem.iterator + s * stage_size
            self.stage(prefetch, prefetch + cfg.rows * pitch, x, w, m, ntile, split, s, tid, gate)
        for chunk in range(self.chunks):
            future = chunk + cfg.stages - 1
            sw_next = smem.iterator + (future % cfg.stages) * stage_size
            self.stage(
                sw_next, sw_next + cfg.rows * pitch, x, w, m, ntile, split, future, tid, gate
            )
            cute.arch.cp_async_wait_group(cfg.stages - 1)
            cute.arch.sync_threads()
            sw = smem.iterator + (chunk % cfg.stages) * stage_size
            sx = sw + cfg.rows * pitch
            if const_expr(self.gate_stride != 0):
                self.transform(sx, m, tid)
            for kk in cutlass.range_constexpr(cfg.tile_k // 16):
                # x is stored [M,K], i.e. the column-major MMA B operand.
                # x2 loads K[0:8], then K[8:16]; no transpose is needed.
                for t in cutlass.range_constexpr(self.mtiles):
                    bp = sx + (t * 8 + lane % 8) * pitch + ((lane // 8) % 2) * 8 + kk * 16
                    b = cute.arch.inline_ptx(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{$w0}, {$w1}}, [{$r0}];",
                        write_only_types=[Int32, Int32],
                        read_only_args=[Int32(bp.toint())],
                    )
                    bfrag[t, 0] = b[0]
                    bfrag[t, 1] = b[1]
                for r in cutlass.range_constexpr(cfg.rows // (16 * cfg.warps)):
                    row = warp * 16 + r * cfg.warps * 16
                    # x4 order: (row0,K0), (row8,K0), (row0,K8), (row8,K8).
                    ap = sw + (row + lane % 16) * pitch + (lane // 16) * 8 + kk * 16
                    a = cute.arch.inline_ptx(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                        write_only_types=[Int32, Int32, Int32, Int32],
                        read_only_args=[Int32(ap.toint())],
                    )
                    for t in cutlass.range_constexpr(self.mtiles):
                        values = _mma(
                            a,
                            (bfrag[t, 0], bfrag[t, 1]),
                            (acc[r, t, 0], acc[r, t, 1], acc[r, t, 2], acc[r, t, 3]),
                        )
                        for v in cutlass.range_constexpr(4):
                            acc[r, t, v] = values[v]
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        for r in cutlass.range_constexpr(cfg.rows // (16 * cfg.warps)):
            for t in cutlass.range_constexpr(self.mtiles):
                for v in cutlass.range_constexpr(4):
                    nr = (
                        ntile * cfg.rows + warp * 16 + r * cfg.warps * 16 + lane // 4 + (v // 2) * 8
                    )
                    mr = t * 8 + (lane % 4) * 2 + v % 2
                    if const_expr(self.cluster == DSMEM):
                        if mr < m:
                            self.push(red, mr, nr - ntile * cfg.rows, acc[r, t, v])
                    elif (nr < self.n) & (mr < m):
                        offset = Int64(mr) * self.n + nr
                        if const_expr(cfg.splits > 1):
                            partial[Int64(split) * m * self.n + offset] = acc[r, t, v]
                        elif const_expr(self.fp32):
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v]
                        else:
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v].to(BFloat16)
        if const_expr(self.cluster == DSMEM):
            self.cluster_reduce(red, y, m, ntile, split, tid)
        elif const_expr(self.cluster == L2):
            self.l2_reduce(partial, y, m, ntile, split, tid)

    @cute.kernel
    def main_reg(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        y: cute.Pointer,
        partial: cute.Pointer,
        m: Int32,
        gate=None,
    ):
        cfg = self.config
        tid, _, _ = cute.arch.thread_idx()
        ntile, split, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        group, thread = lane // 4, lane % 4
        pitch = cfg.tile_k + 8
        buffer_size = self.xrows * pitch
        if const_expr(self.gate_stride != 0):
            buffer_size += self.xrows * pitch
        allocator = utils.SmemAllocator()
        sx = allocator.allocate_tensor(BFloat16, cute.make_layout(buffer_size), byte_alignment=16)
        red = self.partial_tile(allocator)
        if const_expr(self.cluster == DSMEM):
            # Peers must be running before their shared memory is written; this
            # phase completes long before the epilogue waits on it.
            cute.arch.cluster_arrive_relaxed()
        row_pairs = cfg.rows // (16 * cfg.warps)
        acc = cute.make_rmem_tensor((row_pairs, self.mtiles, 4), Float32)
        acc.fill(0.0)
        bfrag = cute.make_rmem_tensor((self.mtiles, 2), Int32)
        weights = cute.make_rmem_tensor((cfg.prefetch, row_pairs, 2, 4), Int32)
        for chunk in range(self.chunks):
            base = (split * self.chunks + chunk) * cfg.tile_k
            # X is loaded once per CTA tile; W never enters shared memory.
            for j in cutlass.range_constexpr(
                cute.ceil_div(self.xrows * cfg.tile_k // 8, cfg.warps * 32)
            ):
                vec = tid + j * cfg.warps * 32
                row, col = vec // (cfg.tile_k // 8), vec % (cfg.tile_k // 8) * 8
                if row < self.xrows:
                    _copy16(
                        sx.iterator + row * pitch + col,
                        x,
                        Int64(row) * self.k + base + col,
                        (row < m) & (base + col < self.k),
                    )
                    if const_expr(self.gate_stride != 0):
                        _copy16(
                            sx.iterator + self.xrows * pitch + row * pitch + col,
                            gate,
                            Int64(row) * self.gate_stride + base + col,
                            (row < m) & (base + col < self.k),
                        )
            cute.arch.cp_async_commit_group()
            for batch in cutlass.range_constexpr(cute.ceil_div(cfg.tile_k // 32, cfg.prefetch)):
                # All independent global vectors are issued before consuming
                # any of them: 2*prefetch 16-byte loads per lane and row pair.
                for d in cutlass.range_constexpr(cfg.prefetch):
                    block = batch * cfg.prefetch + d
                    if const_expr(block < cfg.tile_k // 32):
                        col = base + block * 32 + thread * 8
                        for r in cutlass.range_constexpr(row_pairs):
                            for half in cutlass.range_constexpr(2):
                                nr = (
                                    ntile * cfg.rows
                                    + warp * 16
                                    + r * cfg.warps * 16
                                    + group
                                    + half * 8
                                )
                                values = _load_w16(
                                    w, Int64(nr) * self.k + col, (nr < self.n) & (col < self.k)
                                )
                                for v in cutlass.range_constexpr(4):
                                    weights[d, r, half, v] = values[v]
                if const_expr(batch == 0):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    if const_expr(self.gate_stride != 0):
                        self.transform(sx.iterator, m, tid)
                for d in cutlass.range_constexpr(cfg.prefetch):
                    block = batch * cfg.prefetch + d
                    if const_expr(block < cfg.tile_k // 32):
                        for s in cutlass.range_constexpr(2):
                            # PTX logical j=2*t+h+8*q maps to physical
                            # phi(s,j)=8*t+4*s+2*q+h within this K32 block.
                            # A packed order: row g, g+8, g, g+8; q=0,0,1,1.
                            # B uses the SAME phi, column g (x row g).
                            for t in cutlass.range_constexpr(self.mtiles):
                                bp = (
                                    sx.iterator
                                    + (t * 8 + group) * pitch
                                    + block * 32
                                    + thread * 8
                                    + s * 4
                                )
                                b = cute.arch.inline_ptx(
                                    "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
                                    write_only_types=[Int32, Int32],
                                    read_only_args=[Int32(bp.toint())],
                                )
                                bfrag[t, 0] = b[0]
                                bfrag[t, 1] = b[1]
                            for r in cutlass.range_constexpr(row_pairs):
                                a = (
                                    weights[d, r, 0, s * 2],
                                    weights[d, r, 1, s * 2],
                                    weights[d, r, 0, s * 2 + 1],
                                    weights[d, r, 1, s * 2 + 1],
                                )
                                for t in cutlass.range_constexpr(self.mtiles):
                                    values = _mma(
                                        a,
                                        (bfrag[t, 0], bfrag[t, 1]),
                                        (acc[r, t, 0], acc[r, t, 1], acc[r, t, 2], acc[r, t, 3]),
                                    )
                                    for v in cutlass.range_constexpr(4):
                                        acc[r, t, v] = values[v]
            cute.arch.sync_threads()
        for r in cutlass.range_constexpr(row_pairs):
            for t in cutlass.range_constexpr(self.mtiles):
                for v in cutlass.range_constexpr(4):
                    nr = ntile * cfg.rows + warp * 16 + r * cfg.warps * 16 + group + (v // 2) * 8
                    mr = t * 8 + thread * 2 + v % 2
                    if const_expr(self.cluster == DSMEM):
                        if mr < m:
                            self.push(red, mr, nr - ntile * cfg.rows, acc[r, t, v])
                    elif (nr < self.n) & (mr < m):
                        offset = Int64(mr) * self.n + nr
                        if const_expr(cfg.splits > 1):
                            partial[Int64(split) * m * self.n + offset] = acc[r, t, v]
                        elif const_expr(self.fp32):
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v]
                        else:
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v].to(BFloat16)
        if const_expr(self.cluster == DSMEM):
            self.cluster_reduce(red, y, m, ntile, split, tid)
        elif const_expr(self.cluster == L2):
            self.l2_reduce(partial, y, m, ntile, split, tid)

    @cute.kernel
    def main_wide(
        self,
        x: cute.Pointer,
        w: cute.Pointer,
        y: cute.Pointer,
        partial: cute.Pointer,
        m: Int32,
    ):
        # Register streaming with 128 bytes per W row per warp instruction: in a
        # 64-column block, lane L loads row L//8 + 4*i, columns 8*(L%8)..+8.
        # One xor-4 exchange gives lane (g, t) columns 8t..8t+8 and 32+8t..32+8t+8
        # of rows g//2 + 4*(g%2) and that + 8 (its MMA row labels g and g + 8),
        # i.e. the reg variant's per-thread K layout in each 32-column half.
        cfg = self.config
        tid, _, _ = cute.arch.thread_idx()
        ntile, split, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        group, thread = lane // 4, lane % 4
        odd = group % 2
        pitch = cfg.tile_k + 8
        allocator = utils.SmemAllocator()
        sx = allocator.allocate_tensor(
            BFloat16, cute.make_layout(self.xrows * pitch), byte_alignment=16
        )
        red = self.partial_tile(allocator)
        if const_expr(self.cluster == DSMEM):
            # Peers must be running before their shared memory is written; this
            # phase completes long before the epilogue waits on it.
            cute.arch.cluster_arrive_relaxed()
        row_groups = cfg.rows // (16 * cfg.warps)
        acc = cute.make_rmem_tensor((row_groups, self.mtiles, 4), Float32)
        acc.fill(0.0)
        bfrag = cute.make_rmem_tensor((self.mtiles, 2), Int32)
        weights = cute.make_rmem_tensor((cfg.prefetch, row_groups, 4, 4), Int32)
        frag = cute.make_rmem_tensor((row_groups, 2, 2, 4), Int32)
        for chunk in range(self.chunks):
            base = (split * self.chunks + chunk) * cfg.tile_k
            for j in cutlass.range_constexpr(
                cute.ceil_div(self.xrows * cfg.tile_k // 8, cfg.warps * 32)
            ):
                vec = tid + j * cfg.warps * 32
                row, col = vec // (cfg.tile_k // 8), vec % (cfg.tile_k // 8) * 8
                if row < self.xrows:
                    _copy16(
                        sx.iterator + row * pitch + col,
                        x,
                        Int64(row) * self.k + base + col,
                        (row < m) & (base + col < self.k),
                    )
            cute.arch.cp_async_commit_group()
            for batch in cutlass.range_constexpr(cute.ceil_div(cfg.tile_k // 64, cfg.prefetch)):
                for d in cutlass.range_constexpr(cfg.prefetch):
                    block = batch * cfg.prefetch + d
                    if const_expr(block < cfg.tile_k // 64):
                        col = base + block * 64 + (lane % 8) * 8
                        for r in cutlass.range_constexpr(row_groups):
                            for i in cutlass.range_constexpr(4):
                                nr = (
                                    ntile * cfg.rows
                                    + warp * 16
                                    + r * cfg.warps * 16
                                    + lane // 8
                                    + 4 * i
                                )
                                values = _load_w16_l2(
                                    w, Int64(nr) * self.k + col, (nr < self.n) & (col < self.k)
                                )
                                for v in cutlass.range_constexpr(4):
                                    weights[d, r, i, v] = values[v]
                if const_expr(batch == 0):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                for d in cutlass.range_constexpr(cfg.prefetch):
                    block = batch * cfg.prefetch + d
                    if const_expr(block < cfg.tile_k // 64):
                        # Even lanes keep i = 0 / 2 (labels g, g + 8), odd lanes i = 1 / 3.
                        for r in cutlass.range_constexpr(row_groups):
                            for q in cutlass.range_constexpr(2):
                                for v in cutlass.range_constexpr(4):
                                    halves = _exchange(
                                        odd, weights[d, r, 2 * q, v], weights[d, r, 2 * q + 1, v]
                                    )
                                    frag[r, 0, q, v] = halves[0]
                                    frag[r, 1, q, v] = halves[1]
                        for b in cutlass.range_constexpr(2):
                            for s in cutlass.range_constexpr(2):
                                # Same phi(s, j) K permutation as main_reg per 32 columns.
                                for t in cutlass.range_constexpr(self.mtiles):
                                    bp = (
                                        sx.iterator
                                        + (t * 8 + group) * pitch
                                        + block * 64
                                        + b * 32
                                        + thread * 8
                                        + s * 4
                                    )
                                    bv = cute.arch.inline_ptx(
                                        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
                                        write_only_types=[Int32, Int32],
                                        read_only_args=[Int32(bp.toint())],
                                    )
                                    bfrag[t, 0] = bv[0]
                                    bfrag[t, 1] = bv[1]
                                for r in cutlass.range_constexpr(row_groups):
                                    a = (
                                        frag[r, b, 0, s * 2],
                                        frag[r, b, 1, s * 2],
                                        frag[r, b, 0, s * 2 + 1],
                                        frag[r, b, 1, s * 2 + 1],
                                    )
                                    for t in cutlass.range_constexpr(self.mtiles):
                                        values = _mma(
                                            a,
                                            (bfrag[t, 0], bfrag[t, 1]),
                                            (
                                                acc[r, t, 0],
                                                acc[r, t, 1],
                                                acc[r, t, 2],
                                                acc[r, t, 3],
                                            ),
                                        )
                                        for v in cutlass.range_constexpr(4):
                                            acc[r, t, v] = values[v]
            cute.arch.sync_threads()
        for r in cutlass.range_constexpr(row_groups):
            for t in cutlass.range_constexpr(self.mtiles):
                for v in cutlass.range_constexpr(4):
                    nr = (
                        ntile * cfg.rows
                        + warp * 16
                        + r * cfg.warps * 16
                        + group // 2
                        + 4 * odd
                        + (v // 2) * 8
                    )
                    mr = t * 8 + thread * 2 + v % 2
                    if const_expr(self.cluster == DSMEM):
                        if mr < m:
                            self.push(red, mr, nr - ntile * cfg.rows, acc[r, t, v])
                    elif (nr < self.n) & (mr < m):
                        offset = Int64(mr) * self.n + nr
                        if const_expr(cfg.splits > 1):
                            partial[Int64(split) * m * self.n + offset] = acc[r, t, v]
                        elif const_expr(self.fp32):
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v]
                        else:
                            y[self._row_offset(offset, mr, nr)] = acc[r, t, v].to(BFloat16)
        if const_expr(self.cluster == DSMEM):
            self.cluster_reduce(red, y, m, ntile, split, tid)
        elif const_expr(self.cluster == L2):
            self.l2_reduce(partial, y, m, ntile, split, tid)

    def partial_tile(self, allocator):
        """FP32 receive tile [splits, xrows, rows/splits] and local staging tile
        [owners, xrows * rows/splits + 4] of a cluster reduction (None otherwise)."""
        if const_expr(self.cluster != DSMEM):
            return None
        return allocator.allocate_tensor(
            Float32,
            cute.make_layout(2 * self.xrows * self.config.rows + 4 * self.config.splits),
            byte_alignment=16,
        )

    def owner_run(self):
        """Staged floats per owner, padded by four so owners start in different banks."""
        return self.xrows * (self.config.rows // self.config.splits) + 4

    @cute.jit
    def push(self, red, mr, row, value):
        # Stage [owner][mr][row % share], owner = row // share, after the receive tile.
        share = self.config.rows // self.config.splits
        slot = row // share * self.owner_run() + mr * share + row % share
        red[self.xrows * self.config.rows + slot] = value

    @cute.jit
    def cluster_reduce(self, red, y, m, ntile, split, tid):
        # Every CTA reaches every barrier (no early exits); after the last one, only
        # local shared memory is read, so CTAs may exit independently.
        cfg = self.config
        threads = cfg.warps * 32
        share = cfg.rows // cfg.splits
        vectors = self.xrows * share // 4  # 16-byte vectors per owner run (xrows % 8 == 0).
        cute.arch.sync_threads()
        # Peers must be running before their shared memory is written (relaxed
        # arrive at kernel start).
        cute.arch.cluster_wait()
        # Owner o's first m*share staged values go to [split][0:m][:] of CTA o. The last
        # vector may carry rows >= m into slots the owner never reads.
        base = Int32(red.iterator.toint())
        stage = self.xrows * cfg.rows
        for j in cutlass.range_constexpr(cute.ceil_div(cfg.splits * vectors, threads)):
            q = j * threads + tid
            owner, i = q // vectors, q % vectors
            if (owner < cfg.splits) & (i * 4 < m * share):
                source = stage + owner * self.owner_run() + i * 4
                _store_cluster(
                    base + (split * self.xrows * share + i * 4) * 4,
                    owner,
                    [red[source + v] for v in range(4)],
                )
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for j in cutlass.range_constexpr(cute.ceil_div(self.xrows * share, threads)):
            e = j * threads + tid
            mr = e // share
            nr = ntile * cfg.rows + split * share + e % share
            if (mr < m) & (nr < self.n):
                value = Float32(0.0)
                for s in cutlass.range_constexpr(cfg.splits):
                    # Identical FP32 addition order to the reduce kernels.
                    value += red[(s * self.xrows + mr) * share + e % share]
                offset = self._row_offset(Int64(mr) * self.n + nr, mr, nr)
                if const_expr(self.fp32):
                    y[offset] = value
                else:
                    y[offset] = value.to(BFloat16)

    @cute.jit
    def l2_reduce(self, partial, y, m, ntile, split, tid):
        # Partials are in the scratch ([split][M][N], written above); the cluster
        # barrier's release/acquire makes them visible, then owner CTA r sums rows
        # [r*share, (r+1)*share) of the N tile over the splits (L2 loads, split order).
        cfg = self.config
        threads = cfg.warps * 32
        share = cfg.rows // cfg.splits
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for j in cutlass.range_constexpr(cute.ceil_div(self.xrows * share, threads)):
            e = j * threads + tid
            mr = e // share
            nr = ntile * cfg.rows + split * share + e % share
            if (mr < m) & (nr < self.n):
                value = Float32(0.0)
                for s in cutlass.range_constexpr(cfg.splits):
                    # Identical FP32 addition order to the reduce kernels.
                    value += _load_l2(partial, Int64(s) * m * self.n + Int64(mr) * self.n + nr)
                offset = self._row_offset(Int64(mr) * self.n + nr, mr, nr)
                if const_expr(self.fp32):
                    y[offset] = value
                else:
                    y[offset] = value.to(BFloat16)

    def _row_offset(self, offset, mr, nr):
        """Output element offset; trace-time choice, so contiguous output is unchanged."""
        return offset if self.ldy == self.n else Int64(mr) * self.ldy + nr

    def _flat_offset(self, idx):
        """Output offset of flat index ``idx`` of [M,N] (pairs never straddle rows: N even)."""
        return idx if self.ldy == self.n else (idx // self.n) * self.ldy + idx % self.n

    @cute.kernel
    def reduce(self, partial: cute.Pointer, y: cute.Pointer, m: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        idx = Int64(block) * 256 + tid
        if idx < m * self.n:
            value = Float32(0.0)
            for split in cutlass.range_constexpr(self.config.splits):
                value += partial[Int64(split) * m * self.n + idx]
            if const_expr(self.fp32):
                y[self._flat_offset(idx)] = value
            else:
                y[self._flat_offset(idx)] = value.to(BFloat16)

    @cute.kernel
    def reduce_pairs(self, partial: cute.Pointer, y: cute.Pointer, m: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        idx = Int64(block) * 64 + tid * 2
        if idx < m * self.n:
            first, second = Float32(0.0), Float32(0.0)
            for split in cutlass.range_constexpr(self.config.splits):
                address = partial.toint() + (Int64(split) * m * self.n + idx) * 4
                values = cute.arch.inline_ptx(
                    "ld.global.v2.f32 {{$w0}, {$w1}}, [{$r0}];",
                    write_only_types=[Float32, Float32],
                    read_only_args=[address],
                )
                # Identical FP32 addition order to the scalar reduction.
                first += values[0]
                second += values[1]
            if const_expr(self.fp32):
                cute.arch.inline_ptx(
                    "st.global.v2.f32 [{$r0}], {{$r1}, {$r2}};",
                    read_only_args=[y.toint() + self._flat_offset(idx) * 4, first, second],
                )
            else:
                # PTX cvt packs its first source in the high half, second in low.
                packed = cute.arch.inline_ptx(
                    "cvt.rn.bf16x2.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Int32],
                    read_only_args=[second, first],
                )
                cute.arch.inline_ptx(
                    "st.global.b32 [{$r0}], {$r1};",
                    read_only_args=[y.toint() + self._flat_offset(idx) * 2, packed],
                )
