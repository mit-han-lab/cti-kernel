# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Two-launch small-M routed MoE. No sorting, counters, atomics or clearing.

FC1 owns (first route naming an expert, intermediate tile), keeps eight token
slots resident, and scatters to every matching route, including duplicates.
FC2 owns (token, output tile) and combines routes in order as FlashInfer's finalize:
``acc = fma(w, BF16(y), acc)`` in FP32, rounded once. Its producer
runs ahead across route boundaries; stage phases never reset between experts.
Only public PTX and the installed CuTe DSL API are used. No upstream kernel
source is incorporated; operand scheduling follows our dense CuTe design.
"""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as hopper
import torch
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int8, Int32, Int64
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.memory import SmemAllocator
from cutlass.runtime import make_fake_stream
from cutlass.tensor_utils import LayoutEnum
from cutlass.utils import blackwell_helpers, blockscaled_layout

_CACHE = {}
_LOCK = threading.Lock()


@cute.jit
def divide_rn(numerator: Float32, denominator: Float32):
    return cute.arch.inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[numerator, denominator],
    )


def unit_scales(mma, tid, pointer, repetitions, operand_a):
    """Build the scale register layout, without loading a scale operand."""
    if operand_a:
        layout = blockscaled_layout.sm120_make_smem_layout_sfa(mma, (128, 128, 128), 32, 1)
        partition = blackwell_helpers.partition_fragment_SFA
    else:
        layout = blockscaled_layout.sm120_make_smem_layout_sfb(mma, (128, 128, 128), 32, 1)
        partition = blackwell_helpers.partition_fragment_SFB
    view = cute.make_tensor(cute.recast_ptr(pointer, dtype=Float8E8M0FNU), layout)
    fragment = partition(view[None, None, 0], mma.get_slice(tid), tid)
    k_mode = next(j for j in range(2, cute.rank(fragment)) if cute.size(fragment.shape[j:]) == 4)
    fragment = cute.group_modes(fragment, k_mode, cute.rank(fragment))
    fragment = cute.group_modes(fragment, 1, k_mode)
    fragment = cute.composition(fragment, (None, cute.make_layout(repetitions), None))
    result = fragment[None, None, 0]
    cute.recast_tensor(result, Int8).fill(127)
    return result


class DecodeKernel:
    def __init__(self, config, *, fc1, top_k, external_quant=False):
        self.fc1, self.top_k = fc1, top_k
        self.width = config.bni if fc1 else config.bn
        self.halves = 2 if fc1 else 1
        self.warps = self.width // 16
        self.threads = 32 * (self.warps + 1)
        self.stages, self.residency = config.stages, config.ctas_per_sm
        self.external_quant = external_quant
        self.pdl = config.use_pdl

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        sa: cute.Tensor,
        w: cute.Tensor,
        sw: cute.Tensor,
        ids: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            hopper.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, Float8E4M3FN, 128),
            Float8E4M3FN,
        )
        # FC1's K groups are resident. FC2's activation shares the weight ring.
        a_planes = a.shape[1] // 128 if self.fc1 else self.stages
        la = cute.tile_to_shape(atom, (8, 128, a_planes), order=(0, 1, 2))
        lw = cute.tile_to_shape(atom, (self.width, 128, self.stages), order=(0, 1, 2))
        tma, gw = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w,
            cute.slice_(lw, (None, None, 0)),
            (self.width, 128),
        )
        mma = cute.make_tiled_mma(
            warp.MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU),
            cute.make_layout((self.warps, 1, 1)),
            permutation_mnk=(self.width, 8, 32),
        )

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[Int64, 2 * self.stages]
            activation_scales: cute.struct.MemRange[Float32, 8 * a_planes]
            weight_scales: cute.struct.MemRange[Float32, self.halves * self.stages]
            activation: cute.struct.Align[cute.struct.MemRange[Float8E4M3FN, cute.cosize(la)], 128]
            weight: cute.struct.Align[
                cute.struct.MemRange[Float8E4M3FN, self.halves * cute.cosize(lw)], 128
            ]

        self.storage = Storage
        assert Storage.size_in_bytes() <= 99 * 1024
        rows = cute.size(ids) if self.fc1 else out.shape[0]
        self.run(a, sa, tma, gw, sw, ids, routing, out, la, lw, mma).launch(
            grid=(out.shape[1] // self.width, rows, 1),
            block=(self.threads, 1, 1),
            smem=Storage.size_in_bytes(),
            min_blocks_per_mp=self.residency,
            stream=stream,
            # Avoid DSL 4.8 querying a live device for its carveout heuristic.
            preferred_smem_carveout=100,
            use_pdl=self.pdl and not self.fc1,
        )

    @cute.jit
    def quant_group(self, a, sa, shared, scales, source, row, kb, plane, lane, valid):
        """Eight lanes own a 128-element group; RN matches the public quantizer."""
        column = (lane % 8) * 16
        if cutlass.const_expr(self.external_quant):
            scale = Float32(0)
            if valid:
                scale = sa[source, kb]
            if lane % 8 == 0:
                scales[row, plane] = scale
            for j in cutlass.range_constexpr(16):
                value = Float8E4M3FN(0)
                if valid:
                    value = a[source, kb * 128 + column + j]
                shared[row, column + j, plane] = value
        else:
            values = cute.make_rmem_tensor((16,), Float32)
            values.fill(0)
            peak, nonfinite = Float32(1e-10), Int32(0)
            if valid:
                for j in cutlass.range_constexpr(16):
                    value = Float32(a[source, kb * 128 + column + j])
                    values[j] = value
                    peak = cutlass.max(peak, cute.math.abs(value))
                    nonfinite |= Int32(not cute.math.isfinite(value))
            for shift in cutlass.range_constexpr(3):
                peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, offset=1 << shift))
                nonfinite |= cute.arch.shuffle_sync_bfly(nonfinite, offset=1 << shift)
            if nonfinite != 0:
                peak = Float32(float("nan"))
            multiplier = divide_rn(Float32(448), peak)
            if lane % 8 == 0:
                scales[row, plane] = peak * Float32(1.0 / 448.0)
            for j in cutlass.range_constexpr(16):
                value = cutlass.min(
                    Float32(448), cutlass.max(Float32(-448), values[j] * multiplier)
                )
                if nonfinite != 0:
                    value = Float32(float("nan"))
                shared[row, column + j, plane] = value.to(Float8E4M3FN)

    @cute.kernel
    def run(
        self,
        a: cute.Tensor,
        sa: cute.Tensor,
        tma: cute.CopyAtom,
        gw: cute.Tensor,
        sw: cute.Tensor,
        ids: cute.Tensor,
        routing: cute.Tensor,
        out: cute.Tensor,
        la: cute.ComposedLayout,
        lw: cute.ComposedLayout,
        mma: cute.TiledMma,
    ):
        tid, _, _ = cute.arch.thread_idx()
        nt, owner, _ = cute.arch.block_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        if cutlass.const_expr(self.pdl and self.fc1):
            # Every CTA signals, including duplicate and masked owners. FC2's
            # weights/ids are independent; h reads must wait for grid completion.
            cute.arch.griddepcontrol_launch_dependents()
        live = Int32(1)
        expert = Int32(0)
        if cutlass.const_expr(self.fc1):
            expert = ids[owner]
            if expert < 0:
                live = Int32(0)
            for earlier in cutlass.range(owner, unroll=1):
                if ids[earlier] == expert:
                    live = Int32(0)
        if live != 0:
            storage = SmemAllocator().allocate(self.storage)
            full = storage.barriers.data_ptr()
            empty = full + self.stages
            asc = storage.activation_scales.get_tensor(
                cute.make_layout((8, cute.size(la, mode=[2])), stride=(1, 8))
            )
            wsc = storage.weight_scales.get_tensor(
                cute.make_layout((self.halves, self.stages), stride=(1, self.halves))
            )
            activation = storage.activation.get_tensor(la.outer, swizzle=la.inner)
            weight = storage.weight.get_tensor(lw.outer, swizzle=lw.inner)
            weight_up = cute.make_tensor(
                cute.recast_ptr(
                    storage.weight.data_ptr() + (self.halves - 1) * cute.cosize(lw), lw.inner
                ),
                lw.outer,
            )
            if wid == self.warps:
                with cute.arch.elect_one():
                    for s in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(full + s, 32)
                        cute.arch.mbarrier_init(empty + s, self.warps * 32)
                cpasync.prefetch_descriptor(tma)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(self.fc1):
                # Stable token slots avoid a shared compaction table. Each expert
                # loads x once; only tokens routed to it are quantized/consumed.
                for group in cutlass.range(tid // 8, 8 * sw.shape[2], self.threads // 8, unroll=1):
                    row, kb = group // sw.shape[2], group % sw.shape[2]
                    used = Int32(0)
                    if row < a.shape[0]:
                        for r in cutlass.range(self.top_k, unroll=1):
                            if ids[row * self.top_k + r] == expert:
                                used = Int32(1)
                    self.quant_group(a, sa, activation, asc, row, row, kb, kb, lane, used != 0)
            cute.arch.sync_threads()

            thread = mma.get_slice(tid)
            ra = mma.make_fragment_A(thread.partition_A(weight)[None, None, None, 0])
            rb = mma.make_fragment_B(thread.partition_B(activation)[None, None, None, 0])
            ua = unit_scales(mma, tid, weight.iterator, cute.size(ra, mode=[1]), True)
            ub = unit_scales(mma, tid, activation.iterator, cute.size(rb, mode=[1]), False)
            load_w = cute.make_tiled_copy_A(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), Float8E4M3FN), mma
            )
            load_a = cute.make_tiled_copy_B(
                cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), Float8E4M3FN), mma
            )
            tw, ta = load_w.get_slice(tid), load_a.get_slice(tid)
            pw, pu, pa = (
                tw.partition_S(weight),
                tw.partition_S(weight_up),
                ta.partition_S(activation),
            )
            vw, va = tw.retile(ra), ta.retile(rb)
            shape = mma.partition_shape_C((self.width, 8))
            coords = thread.partition_C(cute.make_identity_tensor((self.width, 8)))
            acc = cute.make_rmem_tensor(shape, Float32)
            up = cute.make_rmem_tensor(shape, Float32)
            partial = cute.make_rmem_tensor(shape, Float32)
            total = cute.make_rmem_tensor(shape, Float32)
            total.fill(0)
            stage, phase = Int32(0), Int32(0)
            dependency_ready = Int32(0)
            route_count = 1 if self.fc1 else self.top_k
            for r in cutlass.range(route_count, unroll=1):
                route = owner * self.top_k + r
                if cutlass.const_expr(not self.fc1):
                    expert = ids[route]
                if expert >= 0:
                    weight_row = Int64(expert) * sw.shape[1] * 128 + nt * self.width
                    tile_w = cute.local_tile(
                        gw, (self.width, 128), (weight_row // self.width, None)
                    )
                    ws, wg = cpasync.tma_partition(
                        tma,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(weight, 0, 2),
                        cute.group_modes(tile_w, 0, 2),
                    )
                    if cutlass.const_expr(self.fc1):
                        tile_u = cute.local_tile(
                            gw, (self.width, 128), ((weight_row + out.shape[1]) // self.width, None)
                        )
                        us, ug = cpasync.tma_partition(
                            tma,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(weight_up, 0, 2),
                            cute.group_modes(tile_u, 0, 2),
                        )
                    if wid == self.warps:
                        for kb in cutlass.range(sw.shape[2], unroll=1):
                            cute.arch.mbarrier_wait(empty + stage, phase ^ 1)
                            if lane == 0:
                                cute.arch.mbarrier_expect_tx(
                                    full + stage, self.halves * self.width * 128
                                )
                            cute.arch.sync_warp()
                            # Weight prefetch precedes h loads/quantization.
                            cute.copy(tma, wg[None, kb], ws[None, stage], tma_bar_ptr=full + stage)
                            if cutlass.const_expr(self.fc1):
                                cute.copy(
                                    tma, ug[None, kb], us[None, stage], tma_bar_ptr=full + stage
                                )
                            else:
                                if cutlass.const_expr(self.pdl):
                                    if dependency_ready == 0:
                                        cute.arch.griddepcontrol_wait()
                                        dependency_ready = Int32(1)
                                for group in cutlass.range_constexpr(2):
                                    row = group * 4 + lane // 8
                                    self.quant_group(
                                        a,
                                        sa,
                                        activation,
                                        asc,
                                        route,
                                        row,
                                        kb,
                                        stage,
                                        lane,
                                        row == 0,
                                    )
                            if lane < self.halves:
                                scale_row = (nt * self.width + lane * out.shape[1]) // 128
                                wsc[lane, stage] = sw[expert, scale_row, kb]
                            cute.arch.mbarrier_arrive(full + stage)
                            stage += 1
                            if stage == self.stages:
                                stage = Int32(0)
                                phase ^= 1
                    else:
                        acc.fill(0)
                        up.fill(0)
                        for kb in cutlass.range(sw.shape[2], unroll=1):
                            cute.arch.mbarrier_wait(full + stage, phase)
                            plane = kb if self.fc1 else stage
                            # Each thread's C atom has two token columns and two
                            # weight rows. Cache only the distinct scale products.
                            factors = cute.make_rmem_tensor((2, self.halves), Float32)
                            for half in cutlass.range_constexpr(self.halves):
                                ws_value = wsc[half, stage]
                                for c in cutlass.range_constexpr(2):
                                    factors[c, half] = asc[coords[c][1], plane] * ws_value
                            for half in cutlass.range_constexpr(self.halves):
                                source = pw if half == 0 else pu
                                partial.fill(0)
                                cute.copy(load_w, source[None, None, 0, stage], vw[None, None, 0])
                                cute.copy(load_a, pa[None, None, 0, plane], va[None, None, 0])
                                for kk in cutlass.range_constexpr(4):
                                    if cutlass.const_expr(kk < 3):
                                        nxt = (kk + 1) % 2
                                        cute.copy(
                                            load_w,
                                            source[None, None, kk + 1, stage],
                                            vw[None, None, nxt],
                                        )
                                        cute.copy(
                                            load_a,
                                            pa[None, None, kk + 1, plane],
                                            va[None, None, nxt],
                                        )
                                    if cutlass.const_expr(kk == 2 and half == self.halves - 1):
                                        # Last shared read is complete; reuse the
                                        # ring while the last two MMAs execute.
                                        cute.arch.mbarrier_arrive(empty + stage)
                                    cute.gemm(
                                        mma,
                                        partial,
                                        [ra[None, None, kk % 2], ua],
                                        [rb[None, None, kk % 2], ub],
                                        partial,
                                    )
                                for j in cutlass.range_constexpr(cute.size(acc)):
                                    if cutlass.const_expr(half == 0):
                                        acc[j] = acc[j] + partial[j] * factors[j % 2, half]
                                    else:
                                        up[j] = up[j] + partial[j] * factors[j % 2, half]
                            stage += 1
                            if stage == self.stages:
                                stage = Int32(0)
                                phase ^= 1
                        if cutlass.const_expr(self.fc1):
                            for j in cutlass.range_constexpr(cute.size(acc)):
                                col, token = coords[j][0], coords[j][1]
                                if token < a.shape[0]:
                                    value = (
                                        divide_rn(acc[j], Float32(1) + cute.math.exp(-acc[j]))
                                        * up[j]
                                    )
                                    for slot in cutlass.range(self.top_k, unroll=1):
                                        target = token * self.top_k + slot
                                        if ids[target] == expert:
                                            out[target, nt * self.width + col] = value.to(BFloat16)
                        else:
                            # FlashInfer's finalize (vLLM's default): the unweighted row,
                            # rounded to BF16, enters one FMA per route in route order.
                            rw = routing[route]
                            for j in cutlass.range_constexpr(cute.size(total)):
                                total[j] = cute.arch.inline_ptx(
                                    "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                                    write_only_types=[Float32],
                                    read_only_args=[rw, Float32(acc[j].to(BFloat16)), total[j]],
                                )
            if cutlass.const_expr(not self.fc1):
                if cutlass.const_expr(self.pdl):
                    # Includes all-masked CTAs, whose producer never loaded h.
                    cute.arch.griddepcontrol_wait()
                if wid < self.warps:
                    for j in cutlass.range_constexpr(cute.size(total)):
                        if coords[j][1] == 0:
                            out[owner, nt * self.width + coords[j][0]] = total[j].to(BFloat16)


def compile_kernel(
    config, e, h, i, top_k=8, *, fc1, external_quant=False, arch="sm_120a", options=""
):
    """Offline compile using symbolic M (1..8); never allocate or launch CUDA."""
    if config.path != "decode":
        raise ValueError("expected a decode config")
    if external_quant and not fc1:
        raise ValueError("external quantization is an FC1 qualification option")
    if h % 128 or i % 128 or h < 128 or i < 128 or e < 1 or not 1 <= top_k <= 8:
        raise ValueError("decode requires H/I multiples of 128 and top_k in 1..8")
    fake = cute.runtime.make_fake_tensor
    k, n = (h, 2 * i) if fc1 else (i, h)
    dtype = Float8E4M3FN if external_quant else BFloat16
    # Public BF16 inputs may be contiguous views with a two-byte storage offset.
    a = fake(dtype, (cute.sym_int64(), k), (k, 1), assumed_align=16 if external_quant else 2)
    sa = fake(Float32, (cute.sym_int64(), k // 128), (k // 128, 1), assumed_align=4)
    w = fake(Float8E4M3FN, (e * n, k), (k, 1), assumed_align=16)
    sw = fake(
        Float32, (e, n // 128, k // 128), (n // 128 * (k // 128), k // 128, 1), assumed_align=4
    )
    ids = fake(Int32, (cute.sym_int64(),), (1,), assumed_align=4)
    routing = fake(Float32, (cute.sym_int64(),), (1,), assumed_align=4)
    width = i if fc1 else h
    out = fake(BFloat16, (cute.sym_int64(), width), (width, 1), assumed_align=16)
    return cute.compile(
        DecodeKernel(config, fc1=fc1, top_k=top_k, external_quant=external_quant),
        a,
        sa,
        w,
        sw,
        ids,
        routing,
        out,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch {arch} {options}",
    )


def _functions(op, config, external_quant):
    result = []
    for fc1 in (True, False):
        source = external_quant and fc1
        key = (config, op.device, op.device_class[:2], op.e, op.h, op.i, op.top_k, fc1, source)
        if key not in _CACHE:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("decode: call prepare outside capture")
            with _LOCK:
                if key not in _CACHE:
                    major, minor = op.device_class[:2]
                    _CACHE[key] = compile_kernel(
                        config,
                        op.e,
                        op.h,
                        op.i,
                        op.top_k,
                        fc1=fc1,
                        external_quant=source,
                        arch=f"sm_{major}{minor}a",
                    )
        result.append(_CACHE[key])
    return result


def forward(op, x, weights, ids, config, *, return_intermediates=False, quantized_x=None):
    """Normal execution launches exactly FC1 and FC2. Debug may supply (xq, xs)."""
    if not 1 <= x.shape[0] <= 8 or not 1 <= op.top_k <= 8:
        raise ValueError("decode supports 1..8 tokens and top_k in 1..8")
    fc1, fc2 = _functions(op, config, quantized_x is not None)
    h = torch.empty((ids.numel(), op.i), dtype=torch.bfloat16, device=x.device)
    out = torch.empty_like(x)
    a, sa = (x, op.s13.view(-1, op.h // 128)) if quantized_x is None else quantized_x
    routes, routing = ids.view(-1), weights.view(-1)
    fc1(a, sa, op.w13.view(-1, op.h), op.s13, routes, routing, h)
    fc2(h, op.s2.view(-1, op.i // 128), op.w2.view(-1, op.i), op.s2, routes, routing, out)
    return (out, h) if return_intermediates else out
