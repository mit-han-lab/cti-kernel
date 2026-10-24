# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""W8A16 grouped backward GEMM with FP32 block-scale promotion.

Design references: NVIDIA CUTLASS CuTe DSL Ampere tensorop_gemm (MMA
partitions and multistage buffering), and this repository's BF16 moe-dx
branch. Independently implemented; no upstream source is incorporated.
https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/ampere
Manual asynchronous copies use only the public NVIDIA PTX ISA.
"""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64
from cutlass.cute.nvgpu import warp
from cutlass.memory import SmemAllocator
from cutlass.runtime import make_fake_stream

BM, BN, BK, STAGES, THREADS = 16, 64, 32, 3, 128


@cute.jit
def _copy_weight(dst, src, valid):
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[Int32(dst.toint()), src.toint(), Int32(16) if valid else Int32(0)],
    )


class DxGemm:
    def __init__(self, h, i, top_k, *, activation, mapped):
        if 128 % BN or 128 % BK or h % 128 or i % 128:
            raise ValueError("backward tiles must divide aligned 128-wide scale blocks")
        self.h, self.i, self.top_k = h, i, top_k
        self.activation, self.mapped = activation, mapped
        self.k, self.n = (h, i) if activation else (2 * i, h)

    @cute.jit
    def __call__(
        self,
        a,
        w,
        scale,
        routes,
        experts,
        padded,
        mapping,
        weights,
        gu,
        q,
        qs,
        dw_partials,
        out,
        stream: cuda.CUstream,
    ):
        la = cute.tile_to_shape(
            cute.make_composed_layout(
                cute.make_swizzle(2, 3, 3), 0, cute.make_layout((8, BK), stride=(BK, 1))
            ),
            (BM, BK, STAGES),
            order=(0, 1, 2),
        )
        lb = cute.tile_to_shape(
            cute.make_composed_layout(
                cute.make_swizzle(3, 3, 3), 0, cute.make_layout((BN, 8), stride=(1, BN))
            ),
            (BN, BK),
            order=(0, 1),
        )
        mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            cute.make_layout((1, 4, 1)),
            permutation_mnk=(16, 64, 16),
        )
        self.run(
            a,
            w,
            scale,
            routes,
            experts,
            padded,
            mapping,
            weights,
            gu,
            q,
            qs,
            dw_partials,
            out,
            la,
            lb,
            mma,
        ).launch(
            grid=(cute.ceil_div(routes.shape[0], BM), self.n // BN, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.jit
    def issue(self, a, w, weights, slots, sa, sa_unweighted, raw, expert, nt, kt, tid):
        stage = kt % STAGES
        # The last prefetched tiles are zero-filled without forming OOB pointers.
        for j in cutlass.range_constexpr(BN * BK // (16 * THREADS)):
            vector = tid + j * THREADS
            col, kk = (vector % (BN // 16)) * 16, vector // (BN // 16)
            offset = Int64(expert) * self.k * self.n
            if kt < self.k // BK:
                offset += Int64(kt * BK + kk) * self.n + nt * BN + col
            _copy_weight(
                raw.iterator + raw.layout((col, kk, stage)), w.iterator + offset, kt < self.k // BK
            )
        for j in cutlass.range_constexpr(BM * BK // THREADS):
            linear = tid + j * THREADS
            row, kk = linear // BK, linear % BK
            route = slots[row]
            value = Float32(0)
            unweighted = Float32(0)
            if (route >= 0) & (route < weights.shape[0]) & (kt < self.k // BK):
                if cutlass.const_expr(self.activation):
                    unweighted = Float32(a[route // self.top_k, kt * BK + kk])
                    value = unweighted * weights[route]
                else:
                    value = Float32(a[route, kt * BK + kk])
            sa[row, kk, stage] = value.to(BFloat16)
            if cutlass.const_expr(self.activation):
                sa_unweighted[row, kk, stage] = unweighted.to(BFloat16)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def run(
        self,
        a: cute.Tensor,
        w: cute.Tensor,
        scale: cute.Tensor,
        routes: cute.Tensor,
        experts: cute.Tensor,
        padded: cute.Tensor,
        mapping: cute.Tensor,
        weights: cute.Tensor,
        gu: cute.Tensor,
        q: cute.Tensor,
        qs: cute.Tensor,
        dw_partials: cute.Tensor,
        out: cute.Tensor,
        la: cute.ComposedLayout,
        lb: cute.ComposedLayout,
        mma: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        mt, nt, _ = cute.arch.block_idx()
        if mt * BM < padded[0]:
            expert = experts[mt]
            if cutlass.const_expr(self.mapped):
                if expert >= 0:
                    expert = mapping[expert]
            if expert >= 0:
                alloc = SmemAllocator()
                slots = alloc.allocate_tensor(Int32, cute.make_layout(BM), byte_alignment=16)
                sa = alloc.allocate_tensor(BFloat16, la, byte_alignment=128)
                # Preserve the original BF16(dy*w) MMA for bitwise dX. The
                # second A ring holds exact BF16 dy, sharing all W2 B loads.
                sa_unweighted = sa
                if cutlass.const_expr(self.activation):
                    sa_unweighted = alloc.allocate_tensor(BFloat16, la, byte_alignment=128)
                    products = alloc.allocate_tensor(
                        Float32, cute.make_layout((BM, BN), stride=(BN, 1)), byte_alignment=128
                    )
                raw = alloc.allocate_tensor(
                    Float8E4M3FN,
                    cute.make_layout((BN, BK, STAGES), stride=(1, BN, BN * BK)),
                    byte_alignment=128,
                )
                sb = alloc.allocate_tensor(BFloat16, lb, byte_alignment=128)
                if tid < BM:
                    route = Int32(-1)
                    if mt * BM + tid < routes.shape[0]:
                        route = routes[mt * BM + tid]
                    slots[tid] = route
                cute.arch.sync_threads()
                thread = mma.get_slice(tid)
                ra = mma.make_fragment_A(thread.partition_A(sa)[None, None, None, 0])
                rb = mma.make_fragment_B(thread.partition_B(sb))
                coord = thread.partition_C(cute.make_identity_tensor((BM, BN)))
                accum = cute.make_rmem_tensor(coord.shape, Float32)
                accum.fill(0)
                partial = cute.make_rmem_tensor(coord.shape, Float32)
                partial.fill(0)
                if cutlass.const_expr(self.activation):
                    ra_unweighted = mma.make_fragment_A(
                        thread.partition_A(sa_unweighted)[None, None, None, 0]
                    )
                    unweighted_accum = cute.make_rmem_tensor(coord.shape, Float32)
                    unweighted_partial = cute.make_rmem_tensor(coord.shape, Float32)
                    unweighted_accum.fill(0)
                    unweighted_partial.fill(0)
                copy_a = cute.make_tiled_copy_A(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma
                )
                copy_b = cute.make_tiled_copy_B(
                    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BFloat16), mma
                )
                ta, tb = copy_a.get_slice(tid), copy_b.get_slice(tid)
                source_a, source_b = ta.partition_S(sa), tb.partition_S(sb)
                dest_a, dest_b = ta.retile(ra), tb.retile(rb)
                if cutlass.const_expr(self.activation):
                    source_unweighted = ta.partition_S(sa_unweighted)
                    dest_unweighted = ta.retile(ra_unweighted)
                for p in cutlass.range_constexpr(STAGES - 1):
                    self.issue(
                        a, w, weights, slots, sa, sa_unweighted, raw, expert, nt, Int32(p), tid
                    )
                for kt in cutlass.range(self.k // BK, unroll=1):
                    stage = kt % STAGES
                    cute.arch.cp_async_wait_group(STAGES - 2)
                    cute.arch.sync_threads()
                    for j in cutlass.range_constexpr(BN * BK // THREADS):
                        linear = tid + j * THREADS
                        col, kk = linear % BN, linear // BN
                        sb[col, kk] = raw[col, kk, stage].to(BFloat16)
                    cute.arch.sync_threads()
                    self.issue(
                        a,
                        w,
                        weights,
                        slots,
                        sa,
                        sa_unweighted,
                        raw,
                        expert,
                        nt,
                        kt + STAGES - 1,
                        tid,
                    )
                    for kk in cutlass.range_constexpr(BK // 16):
                        cute.copy(copy_a, source_a[None, None, kk, stage], dest_a[None, None, kk])
                        cute.copy(copy_b, source_b[None, None, kk], dest_b[None, None, kk])
                        cute.gemm(mma, partial, ra[None, None, kk], rb[None, None, kk], partial)
                        if cutlass.const_expr(self.activation):
                            cute.copy(
                                copy_a,
                                source_unweighted[None, None, kk, stage],
                                dest_unweighted[None, None, kk],
                            )
                            cute.gemm(
                                mma,
                                unweighted_partial,
                                ra_unweighted[None, None, kk],
                                rb[None, None, kk],
                                unweighted_partial,
                            )
                    # Raw E4M3 is exact in BF16. Promote once per scale block,
                    # after all four BK tiles, without rounding scaled weights.
                    if (kt + 1) % (128 // BK) == 0:
                        factor = Float32(scale[expert, kt * BK // 128, nt * BN // 128])
                        for j in cutlass.range_constexpr(cute.size(coord)):
                            accum[j] = cute.math.fma(factor, partial[j], accum[j])
                        partial.fill(0)
                        if cutlass.const_expr(self.activation):
                            for j in cutlass.range_constexpr(cute.size(coord)):
                                unweighted_accum[j] = cute.math.fma(
                                    factor, unweighted_partial[j], unweighted_accum[j]
                                )
                            unweighted_partial.fill(0)
                cute.arch.cp_async_wait_group(0)
                for j in cutlass.range_constexpr(cute.size(coord)):
                    row, col = coord[j][0], nt * BN + coord[j][1]
                    route = slots[row]
                    if (route >= 0) & (route < weights.shape[0]):
                        if cutlass.const_expr(self.activation):
                            # FP8 q is exact in FP32; scale/product round only
                            # in FP32. Shared storage bridges the MMA C mapping
                            # to a fixed row-wise warp reduction (no atomics).
                            aq = Float32(q[route, col]) * Float32(qs[route, col // 128])
                            products[row, coord[j][1]] = aq * unweighted_accum[j]
                            g, u = Float32(gu[route, col]), Float32(gu[route, self.i + col])
                            # torch's silu backward: accurate expf, IEEE divide.
                            sigmoid = Float32(1) / (Float32(1) + cute.math.exp(-g, fastmath=False))
                            out[route, col] = (accum[j] * u * sigmoid * (1 + g * (1 - sigmoid))).to(
                                BFloat16
                            )
                            out[route, self.i + col] = (accum[j] * g * sigmoid).to(BFloat16)
                        else:
                            out[route, col] = accum[j]

                if cutlass.const_expr(self.activation):
                    cute.arch.sync_threads()
                    lane, warp_id = tid % 32, tid // 32
                    for j in cutlass.range_constexpr(BM // (THREADS // 32)):
                        row = warp_id + j * (THREADS // 32)
                        route = slots[row]
                        if (route >= 0) & (route < weights.shape[0]):
                            value = products[row, lane] + products[row, lane + 32]
                            for shift in cutlass.range_constexpr(5):
                                value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
                            if lane == 0:
                                dw_partials[route, nt] = value


def compile_gemm(h, i, top_k, *, activation, mapped, scale_bf16=False, arch="sm_120a", options=""):
    """Compile a symbolic grouped launch without a device or CUDA context."""
    fake = cute.runtime.make_fake_tensor

    def tensor(dtype, shape):
        stride, strides = 1, []
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= extent
        return fake(dtype, shape, tuple(strides), assumed_align=dtype.width // 8)

    rows, blocks, extent, e = (cute.sym_int64() for _ in range(4))
    k, n = (h, i) if activation else (2 * i, h)
    return cute.compile(
        DxGemm(h, i, top_k, activation=activation, mapped=mapped),
        tensor(BFloat16, (rows, k)),
        tensor(Float8E4M3FN, (e, k, n)),
        tensor(BFloat16 if scale_bf16 else Float32, (e, k // 128, n // 128)),
        tensor(Int32, (extent,)),
        tensor(Int32, (blocks,)),
        tensor(Int32, (1,)),
        tensor(Int32, (cute.sym_int64(),)),
        tensor(Float32, (cute.sym_int64(),)),
        tensor(BFloat16, (cute.sym_int64(), 2 * i)),
        tensor(Float8E4M3FN, (cute.sym_int64(), i)),
        tensor(Float32, (cute.sym_int64(), i // 128)),
        tensor(Float32, (cute.sym_int64(), i // BN)),
        tensor(BFloat16 if activation else Float32, (cute.sym_int64(), 2 * i if activation else h)),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch} {options}",
    )


# ----------------------------------------------------------------- the sm120/sm121 dX chain

_CACHE = {}
_LOCK = threading.Lock()


class FinishDx:
    def __init__(self, h, i, top_k, mapped):
        self.h, self.tiles, self.top_k, self.mapped = h, i // BN, top_k, mapped

    @cute.jit
    def __call__(self, route_dx, dw_partials, ids, mapping, dx, dw, stream: cuda.CUstream):
        self.reduce(route_dx, ids, mapping, dx).launch(
            grid=(cute.ceil_div(dx.shape[0] * self.h, 128), 1, 1), block=(128, 1, 1), stream=stream
        )
        self.dot(dw_partials, ids, mapping, dw).launch(
            grid=(cute.ceil_div(ids.shape[0], 128), 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.jit
    def local(self, ids, mapping, route):
        expert = ids[route]
        if cutlass.const_expr(self.mapped):
            if expert >= 0:
                expert = mapping[expert]
        return expert >= 0

    @cute.kernel
    def reduce(
        self, route_dx: cute.Tensor, ids: cute.Tensor, mapping: cute.Tensor, dx: cute.Tensor
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        linear = bid * 128 + tid
        if linear < dx.shape[0] * self.h:
            token, col = linear // self.h, linear % self.h
            value = Float32(0)
            for slot in cutlass.range_constexpr(self.top_k):
                route = token * self.top_k + slot
                if self.local(ids, mapping, route):
                    value += route_dx[route, col]
            dx[token, col] = value.to(BFloat16)

    @cute.kernel
    def dot(
        self,
        dw_partials: cute.Tensor,
        ids: cute.Tensor,
        mapping: cute.Tensor,
        dw: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        route = bid * 128 + tid
        if route < ids.shape[0]:
            value = Float32(0)
            if self.local(ids, mapping, route):
                for tile in cutlass.range_constexpr(self.tiles):
                    value += dw_partials[route, tile]
            dw[route] = value


def compile_finish(h, i, top_k, *, mapped, arch="sm_120a", options=""):
    fake = cute.runtime.make_fake_tensor

    def matrix(dtype):
        return fake(dtype, (cute.sym_int64(), h), (h, 1), assumed_align=dtype.width // 8)

    def vector(dtype):
        return fake(dtype, (cute.sym_int64(),), (1,), assumed_align=dtype.width // 8)

    return cute.compile(
        FinishDx(h, i, top_k, mapped),
        matrix(Float32),
        fake(Float32, (cute.sym_int64(), i // BN), (i // BN, 1), assumed_align=4),
        vector(Int32),
        vector(Int32),
        matrix(BFloat16),
        vector(Float32),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch} {options}",
    )


def _compiled(op, arch):
    key = (op.h, op.i, op.top_k, arch, op.expert_map is not None, op.s2.dtype, op.s13.dtype)
    if key not in _CACHE:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("block_fp8_moe_dx: call prepare_dx outside capture")
        with _LOCK:
            if key not in _CACHE:
                common = {"mapped": op.expert_map is not None, "arch": arch}
                _CACHE[key] = (
                    compile_gemm(
                        op.h,
                        op.i,
                        op.top_k,
                        activation=True,
                        scale_bf16=op.s2.dtype == torch.bfloat16,
                        **common,
                    ),
                    compile_gemm(
                        op.h,
                        op.i,
                        op.top_k,
                        activation=False,
                        scale_bf16=op.s13.dtype == torch.bfloat16,
                        **common,
                    ),
                    compile_finish(op.h, op.i, op.top_k, **common),
                )
    return _CACHE[key]


def cute_dx(op, x, topk_weights, topk_ids, dy, gu, q, qs, layout, key, cc):
    """The CuTe sm120/sm121 chain over the forward cache and its ``BM`` route layout
    (validated and prepared by the dispatcher, ``_moe_block_fp8_dx.block_fp8_moe_dx``)."""
    ready = getattr(op, "_dx_ready", set())
    if torch.cuda.is_current_stream_capturing() and key not in ready:
        raise RuntimeError("block_fp8_moe_dx: call prepare_dx([M]) outside capture")
    da_gemm, dx_gemm, finish = _compiled(op, f"sm_{cc[0]}{cc[1]}a")
    mapping = op.expert_map if op.expert_map is not None else topk_ids.view(-1)
    weights = topk_weights.view(-1)
    dgdu = torch.empty_like(gu)
    route_dx = torch.empty((topk_ids.numel(), op.h), dtype=torch.float32, device=x.device)
    dw_partials = torch.empty((topk_ids.numel(), op.i // BN), dtype=torch.float32, device=x.device)
    da_gemm(dy, op.w2, op.s2, *layout, mapping, weights, gu, q, qs, dw_partials, dgdu)
    dx_gemm(dgdu, op.w13, op.s13, *layout, mapping, weights, gu, q, qs, dw_partials, route_dx)
    dx, dw = torch.empty_like(x), torch.empty_like(topk_weights)
    finish(route_dx, dw_partials, topk_ids.view(-1), mapping, dx, dw.view(-1))
    ready.add(key)
    op._dx_ready = ready
    return dx, dw
