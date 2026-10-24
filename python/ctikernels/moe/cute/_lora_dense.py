# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Dense shared LoRA: split-K shrink followed by tiled expand.

Shrink atomically reduces K splits into compact FP32 rows. Across splits the
addition order is nondeterministic; determinism is lower priority than latency.
Single-split shrink stores directly, without initializing the workspace.
"""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.runtime import make_fake_stream
from cutlass.utils import SmemAllocator

from .. import _lora_common
from ._lora_block import add_pair, copy16, load_a, load_b, mma, store16

# Config: (shrink BM, split K, expand BM, BN, shrink stages); legacy four-tuples use 2 stages.
DENSE_OVERRIDE = {}
_COMPILED = {}


def _shrink_tiles(total_r):
    return (total_r + 31) // 32 if total_r >= 32 else total_r // 8


def _shrink_bm(block_m, total_r):
    while block_m // 16 * _shrink_tiles(total_r) * 4 > 64:
        block_m //= 2
    return block_m


def _dense_config(config):
    config = tuple(config)
    if len(config) == 4:
        config += (2,)
    if len(config) != 5:
        raise ValueError("dense config must contain 4 or 5 entries")
    return config


def dense_config(rows, k, n, rank, slices, sms, config=None):
    """Metadata-only launch policy; explicit configs must satisfy resource guards."""
    if DENSE_OVERRIDE:
        return _dense_config(DENSE_OVERRIDE["config"])
    if config is not None:
        return _dense_config(config)
    bm = 16 if rows <= 128 else 32 if rows <= 1024 else 64
    shrink_bm = _shrink_bm(64, rank * slices)
    blocks = max(1, (rows + shrink_bm - 1) // shrink_bm)
    tiles = (k + 63) // 64
    splits = max(1, min((2 * sms + blocks - 1) // blocks, tiles))
    stages = max(2, min(4, (tiles + splits - 1) // splits + 1))
    while stages > 2:
        try:
            DenseShrink(k, rank * slices, shrink_bm, splits, stages=stages)
        except ValueError:
            stages -= 1
        else:
            break
    bn = 128 if (rows + bm - 1) // bm * ((n + 127) // 128) * slices >= sms else 64
    return shrink_bm, splits, bm, bn, stages


@cute.jit
def load_global4(ptr, valid):
    # Predicate inside PTX to avoid branch-merge moves consuming each load early.
    return cute.arch.inline_ptx(
        "{ .reg .pred p; setp.ne.u32 p, {$r1}, 0; "
        "mov.f32 {$w0}, 0f00000000; mov.f32 {$w1}, 0f00000000; "
        "mov.f32 {$w2}, 0f00000000; mov.f32 {$w3}, 0f00000000; "
        "@p ld.global.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}]; }",
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[ptr.toint(), Int32(valid)],
    )


@cute.jit
def reduce16(dst, src):
    cute.arch.inline_ptx(
        "{ .reg .f32 v0, v1, v2, v3; "
        "ld.shared.v4.f32 {v0, v1, v2, v3}, [{$r1}]; "
        "red.global.add.v4.f32 [{$r0}], {v0, v1, v2, v3}; }",
        read_only_args=[dst.toint(), Int32(src.toint())],
    )


class DenseShrink:
    def __init__(self, k, total_r, block_m, split_k, block_k=64, stages=2):
        if k <= 0 or k % 8 or not 8 <= total_r <= 256 or total_r % 8:
            raise ValueError("K and total_r must be positive multiples of 8; total_r <= 256")
        if block_m not in (16, 32, 64) or block_k != 64:
            raise ValueError("shrink block_m must be 16, 32, or 64; block_k must be 64")
        if type(split_k) is not int or split_k < 1 or stages not in (2, 3, 4):
            raise ValueError("split_k must be positive and stages must be 2, 3, or 4")
        self.k, self.r, self.bm, self.sk = k, total_r, block_m, split_k
        self.bk, self.stages = block_k, stages
        self.rt = _shrink_tiles(total_r)
        if block_m // 16 * self.rt * 4 > 64:
            raise ValueError("shrink configuration exceeds 64 FP32 accumulator registers")
        self.kwarps = total_r < 32
        self.arows = total_r if self.kwarps else self.rt * 32
        self.pitch = block_k + 8
        self.xsize, self.asize = block_m * self.pitch, self.arows * self.pitch
        self.tiles = (k + block_k - 1) // block_k
        self.per_split = (self.tiles + split_k - 1) // split_k
        ring_bytes = 2 * stages * (self.xsize + self.asize)
        scratch_bytes = 4 * (4 if self.kwarps else 1) * block_m * total_r
        # The drained ring becomes the coalesced-store/reduction workspace.
        self.smem_bytes = max(ring_bytes, scratch_bytes)
        if self.smem_bytes + 1024 > 99 * 1024:
            raise ValueError("dense shrink exceeds the 99 KiB shared-memory limit")

    @cute.jit
    def __call__(self, x: cute.Tensor, a: cute.Tensor, ws: cute.Tensor, stream: cuda.CUstream):
        self.main(x.iterator, a.iterator, ws.iterator, x.shape[0]).launch(
            grid=(cute.ceil_div(x.shape[0], self.bm), self.sk, 1),
            block=(128, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
            preferred_smem_carveout=100,
        )

    @cute.jit
    def stage(self, sx, sa, x, a, rows, block, split, unit, tid):
        tile = split * self.per_split + unit
        valid_tile = (unit < self.per_split) & (tile < self.tiles)
        for j in cutlass.range((self.bm * 8 + 127) // 128, unroll=1):
            v = tid + j * 128
            m, col = v // 8, v % 8 * 8
            if m < self.bm:
                row = block * self.bm + m
                copy16(
                    sx + (unit % self.stages) * self.xsize + m * self.pitch + col,
                    x,
                    Int64(row) * self.k + tile * self.bk + col,
                    valid_tile & (row < rows) & (tile * self.bk + col < self.k),
                )
        for j in cutlass.range((self.arows * 8 + 127) // 128, unroll=1):
            v = tid + j * 128
            r, col = v // 8, v % 8 * 8
            if r < self.arows:
                copy16(
                    sa + (unit % self.stages) * self.asize + r * self.pitch + col,
                    a,
                    Int64(r) * self.k + tile * self.bk + col,
                    valid_tile & (r < self.r) & (tile * self.bk + col < self.k),
                )
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def main(self, x: cute.Pointer, a: cute.Pointer, ws: cute.Pointer, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        block, split, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        buf = SmemAllocator().allocate_tensor(
            BFloat16, cute.make_layout(self.smem_bytes // 2), byte_alignment=16
        )
        sx = buf.iterator
        sa = sx + self.stages * self.xsize
        acc = cute.make_rmem_tensor((self.bm // 16, self.rt, 4), Float32)
        acc.fill(0)
        cute.arch.griddepcontrol_wait()
        for ahead in cutlass.range(self.stages - 1, unroll=1):
            self.stage(sx, sa, x, a, rows, block, split, Int32(ahead), tid)
        for unit in range(self.per_split):
            self.stage(sx, sa, x, a, rows, block, split, unit + self.stages - 1, tid)
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            ax = sx + (unit % self.stages) * self.xsize
            aa = sa + (unit % self.stages) * self.asize
            for kk in cutlass.range(1 if self.kwarps else 4, unroll=1):
                step = warp * 16 if const_expr(self.kwarps) else kk * 16
                for m in cutlass.range_constexpr(self.bm // 16):
                    af = load_a(ax + (m * 16 + lane % 16) * self.pitch + lane // 16 * 8 + step)
                    for r in cutlass.range_constexpr(self.rt):
                        rn = r * 8 if const_expr(self.kwarps) else warp * 8 + r * 32
                        bf = load_b(aa + (rn + lane % 8) * self.pitch + lane // 8 % 2 * 8 + step)
                        result = mma(
                            af, bf, (acc[m, r, 0], acc[m, r, 1], acc[m, r, 2], acc[m, r, 3])
                        )
                        for v in cutlass.range_constexpr(4):
                            acc[m, r, v] = result[v]
            cute.arch.sync_threads()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        scratch = cute.recast_ptr(sx, dtype=Float32)
        for m in cutlass.range_constexpr(self.bm // 16):
            for r in cutlass.range_constexpr(self.rt):
                for v in cutlass.range_constexpr(4):
                    mr = m * 16 + lane // 4 + v // 2 * 8
                    rr = (
                        (r * 8 if const_expr(self.kwarps) else warp * 8 + r * 32)
                        + lane % 4 * 2
                        + v % 2
                    )
                    plane = warp if const_expr(self.kwarps) else Int32(0)
                    if rr < self.r:
                        scratch[(plane * self.bm + mr) * self.r + rr] = acc[m, r, v]
        cute.arch.sync_threads()
        if const_expr(self.kwarps):
            for j in cutlass.range((self.bm * self.r + 127) // 128, unroll=1):
                index = tid + j * 128
                if index < self.bm * self.r:
                    value = Float32(0)
                    for w in cutlass.range_constexpr(4):
                        value += scratch[w * self.bm * self.r + index]
                    scratch[index] = value
            cute.arch.sync_threads()
        for j in cutlass.range((self.bm * (self.r // 4) + 127) // 128, unroll=1):
            index = (tid + j * 128) * 4
            m, r = index // self.r, index % self.r
            row = block * self.bm + m
            if (m < self.bm) & (row < rows):
                if const_expr(self.sk == 1):
                    store16(ws + Int64(row) * self.r + r, scratch + index)
                else:
                    reduce16(ws + Int64(row) * self.r + r, scratch + index)
        cute.arch.griddepcontrol_launch_dependents()


class DenseExpand:
    def __init__(self, n, rank, slices, block_m, block_n, save_z):
        if n <= 0 or n % 8 or not 8 <= rank <= 128 or rank % 8 or slices not in (1, 2):
            raise ValueError("invalid dense expand dimensions")
        if block_m not in (16, 32, 64) or block_n not in (64, 128):
            raise ValueError("expand block_m must be 16, 32, or 64 and block_n 64 or 128")
        self.n, self.r, self.slices = n, rank, slices
        self.bm, self.bn, self.save_z = block_m, block_n, save_z
        self.rpad = (rank + 15) // 16 * 16
        self.pitch = self.rpad + 8
        # Padded cache rows keep the epilogue's eight-row accesses on distinct banks.
        self.cpitch = block_n + 8
        self.zsize, self.bsize = block_m * self.pitch, block_n * self.pitch
        self.smem_bytes = 2 * (self.zsize + self.bsize + block_m * self.cpitch)
        if self.smem_bytes + 1024 > 99 * 1024:
            raise ValueError("dense expand exceeds the 99 KiB shared-memory limit")

    @cute.jit
    def __call__(
        self,
        ws: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        scale: cute.Tensor,
        z,
        stream: cuda.CUstream,
    ):
        zp = z.iterator if const_expr(self.save_z) else out.iterator
        self.main(ws.iterator, b.iterator, out.iterator, scale.iterator, zp, out.shape[0]).launch(
            grid=(
                cute.ceil_div(out.shape[0], self.bm),
                cute.ceil_div(self.n, self.bn),
                self.slices,
            ),
            block=(128, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
            preferred_smem_carveout=100,
        )

    @cute.kernel
    def main(
        self,
        ws: cute.Pointer,
        b: cute.Pointer,
        out: cute.Pointer,
        scale: cute.Pointer,
        z: cute.Pointer,
        rows: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, nt, s = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        buf = SmemAllocator().allocate_tensor(
            BFloat16, cute.make_layout(self.smem_bytes // 2), byte_alignment=16
        )
        sz = buf.iterator
        sb, sc = sz + self.zsize, sz + self.zsize + self.bsize
        # Shrink only writes ws and signals dependents after its own wait.
        # For split-K its immediate predecessor is the workspace zero-fill.
        # All shrink predecessors (including cache/B producers before that fill)
        # are complete when this CTA starts, so B/cache prefetch before our wait is safe.
        for j in cutlass.range((self.bn * (self.rpad // 8) + 127) // 128, unroll=1):
            v = tid + j * 128
            col, r = v // (self.rpad // 8), v % (self.rpad // 8) * 8
            if col < self.bn:
                n = nt * self.bn + col
                copy16(
                    sb + col * self.pitch + r,
                    b,
                    (Int64(s) * self.n + n) * self.r + r,
                    (n < self.n) & (r < self.r),
                )
        for j in cutlass.range((self.bm * (self.bn // 8) + 127) // 128, unroll=1):
            v = tid + j * 128
            m, col = v // (self.bn // 8), v % (self.bn // 8) * 8
            if m < self.bm:
                row, n = block * self.bm + m, nt * self.bn + col
                copy16(
                    sc + m * self.cpitch + col,
                    out,
                    Int64(row) * self.slices * self.n + s * self.n + n,
                    (row < rows) & (n < self.n),
                )
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        # Rank 8 needs only one vector per thread even at BM64. Fill its padded
        # half locally, avoiding a second masked load and early conversion scheduling.
        load_r = const_expr(8 if self.r == 8 else self.rpad)
        vectors = const_expr((self.bm * (load_r // 4) + 127) // 128)
        # Issue every load before conversion/use; cap each unrolled batch at 64 FP32s.
        for batch in cutlass.range_constexpr((vectors + 15) // 16):
            count = const_expr(min(16, vectors - batch * 16))
            values = cute.make_rmem_tensor((count, 4), Float32)
            for j in cutlass.range_constexpr(count):
                v = tid + (batch * 16 + j) * 128
                m, r = v // (load_r // 4), v % (load_r // 4) * 4
                row = block * self.bm + m
                loaded = load_global4(
                    ws + (Int64(row) * self.slices + s) * self.r + r,
                    (m < self.bm) & (row < rows) & (r < self.r),
                )
                for q in cutlass.range_constexpr(4):
                    values[j, q] = loaded[q]
            for j in cutlass.range_constexpr(count):
                v = tid + (batch * 16 + j) * 128
                m, r = v // (load_r // 4), v % (load_r // 4) * 4
                row = block * self.bm + m
                if m < self.bm:
                    for q in cutlass.range_constexpr(4):
                        sz[m * self.pitch + r + q] = values[j, q].to(BFloat16)
                        if const_expr(self.r == 8):
                            sz[m * self.pitch + r + q + 8] = BFloat16(0)
                        if const_expr(self.save_z):
                            if (nt == 0) & (row < rows) & (r < self.r):
                                z[(Int64(row) * self.slices + s) * self.r + r + q] = values[
                                    j, q
                                ].to(z.dtype)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        acc = cute.make_rmem_tensor((self.bm // 16, self.bn // 32, 4), Float32)
        acc.fill(0)
        for kk in cutlass.range(self.rpad // 16, unroll=1):
            for m in cutlass.range_constexpr(self.bm // 16):
                af = load_a(sz + (m * 16 + lane % 16) * self.pitch + lane // 16 * 8 + kk * 16)
                for n in cutlass.range_constexpr(self.bn // 32):
                    bf = load_b(
                        sb
                        + (warp * (self.bn // 4) + n * 8 + lane % 8) * self.pitch
                        + lane // 8 % 2 * 8
                        + kk * 16
                    )
                    result = mma(af, bf, (acc[m, n, 0], acc[m, n, 1], acc[m, n, 2], acc[m, n, 3]))
                    for q in cutlass.range_constexpr(4):
                        acc[m, n, q] = result[q]
        multiplier = scale[0]
        for m in cutlass.range_constexpr(self.bm // 16):
            for n in cutlass.range_constexpr(self.bn // 32):
                for half in cutlass.range_constexpr(2):
                    row = m * 16 + lane // 4 + half * 8
                    col = warp * (self.bn // 4) + n * 8 + lane % 4 * 2
                    add_pair(
                        sc + row * self.cpitch + col,
                        acc[m, n, half * 2],
                        acc[m, n, half * 2 + 1],
                        multiplier,
                    )
        cute.arch.sync_threads()
        for j in cutlass.range((self.bm * (self.bn // 8) + 127) // 128, unroll=1):
            v = tid + j * 128
            m, col = v // (self.bn // 8), v % (self.bn // 8) * 8
            row, n = block * self.bm + m, nt * self.bn + col
            if (m < self.bm) & (row < rows) & (n < self.n):
                store16(
                    out + Int64(row) * self.slices * self.n + s * self.n + n,
                    sc + m * self.cpitch + col,
                )
        cute.arch.griddepcontrol_launch_dependents()


@lru_cache(maxsize=None)
def compile_dense(k, n, rank, slices, capability, config, z_dtype=None):
    """Fake-tensor compilation; row counts stay symbolic and launches use the caller stream."""

    bm_s, sk, bm_e, bn, stages = _dense_config(config)
    shrink = DenseShrink(k, slices * rank, bm_s, sk, stages=stages)
    expand = DenseExpand(n, rank, slices, bm_e, bn, z_dtype is not None)
    fake, t = cute.runtime.make_fake_tensor, cute.sym_int()
    x = fake(BFloat16, (t, k), (k, 1), assumed_align=16)
    a = fake(BFloat16, (slices * rank, k), (k, 1), assumed_align=16)
    b = fake(BFloat16, (slices, n, rank), (n * rank, rank, 1), assumed_align=16)
    ws = fake(Float32, (t, slices * rank), (slices * rank, 1), assumed_align=16)
    out = fake(BFloat16, (t, slices * n), (slices * n, 1), assumed_align=16)
    scale = fake(Float32, (1,), (1,))
    z = (
        None
        if z_dtype is None
        else fake(
            Float32 if z_dtype == torch.float32 else BFloat16,
            (t, slices, rank),
            (slices * rank, rank, 1),
            assumed_align=16,
        )
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]}a"
    return (
        cute.compile(shrink, x, a, ws, stream, options=options),
        cute.compile(expand, ws, b, out, scale, z, stream, options=options),
    )


def dense_apply_(out, x, a2d, b, scale, *, rank, slices, z_out=None, config=None):
    """Apply validated shared factors A [slices*rank,K], B [slices,N,rank]."""
    capability, sms = _lora_common.device_info(x.device)
    if capability not in ((12, 0), (12, 1)) or not _lora_common.available():
        raise RuntimeError("cute requires sm_120/sm_121 and nvidia-cutlass-dsl 4.8.0")
    rows, k = x.shape
    if rows == 0:
        return out
    n = b.shape[1]
    config = dense_config(rows, k, n, rank, slices, sms, config=config)
    zdtype = None if z_out is None else z_out.dtype
    key = (x.device, k, n, rank, slices, capability, config, zdtype)
    kernels = _COMPILED.get(key)
    if kernels is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "warm up this dense LoRA config and z_out dtype before CUDA graph capture"
            )
        with _lora_common.LOCK:
            kernels = _COMPILED.get(key)
            if kernels is None:
                kernels = compile_dense(k, n, rank, slices, capability, config, zdtype)
                _COMPILED[key] = kernels
    allocate = torch.empty if config[1] == 1 else torch.zeros
    ws = allocate((rows, slices * rank), dtype=torch.float32, device=x.device)
    kernels[0](x, a2d, ws)
    kernels[1](ws, b, out, scale, z_out)
    return out
