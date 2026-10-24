# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The attention prologue as one CuTe DSL kernel for sm_120a / sm_121a.

Programs (CTAs) are of one kind each: Q programs normalize and rotate Q heads in place in
``qkv``, K programs write normalized, rotated K heads to ``k_cache[slot]``, V programs copy V
heads to ``v_cache[slot]``. A warp owns one token's group of ``rows`` heads, so the position,
the cos/sin row and the weight are loaded once per warp. In a head row each lane holds two
rotary dims (2 lane, 2 lane + 1; the NeoX partner d +- 32 is lane ^ 16) and lanes 8-31 hold
eight of the other 192 dims (one 16-byte access). The math is FP32 with one rounding at the
store. With 128-dim heads all dims rotate: each lane holds two dims and their partners
64 dims later, so rotation needs no shuffle.

With an FP8 cache (``cache`` "e4m3" or "e5m2") K and V are quantized as they are stored: the
FP32 K result (V's BF16 value) is divided by the per-tensor FP32 scale (``div.rn``) and rounded
once to FP8 with saturation to the finite range (``cvt.rn.satfinite``), vLLM's
``reshape_and_cache_flash`` FP8 semantics. Q stays BF16 in ``qkv``.

The masked kernel (``masked``) skips the cache writes of a token whose slot is negative
(padding): K and V programs finish their values in registers, then one warp-uniform branch on
``slot >= 0`` holds every cache address computation and store. Q is processed as usual. The
unmasked kernel has no such branch and is the kernel used for calls with valid slots only.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int16, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

HEAD_DIM = 256
ROTARY_DIM = 64
# rsqrt.approx.f32 overshoots 1/sqrt by at most 2^-22.9 (relative); scaled by 1 - 2^-21 it
# never exceeds it, so a row whose eps term is below FP32 resolution cannot round across a
# BF16 tie.
_RSQRT_NO_OVERSHOOT = 1.0 - 2.0**-21

_G2R = cute.nvgpu.CopyG2ROp()
_R2G = cute.nvgpu.CopyR2GOp()


def _l2_policy(kind):
    """An L2 cache policy (createpolicy) for ``.L2::cache_hint`` accesses."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            f"createpolicy.fractional.L2::{kind}.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _quantize(values, scale, cache):
    """FP8 bytes of FP32 ``values`` (2 or a multiple of 4) divided by ``scale``: an Int16 for
    two values, otherwise one Int32 per four (element i in byte i)."""
    quotients = [
        cute.arch.inline_ptx(
            "div.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],
            read_only_args=[Float32(value), scale],
        )
        for value in values
    ]
    pairs = [
        # cvt puts its first source in the upper byte.
        cute.arch.inline_ptx(
            f"cvt.rn.satfinite.{cache}x2.f32 {{$w0}}, {{$r0}}, {{$r1}};",
            write_only_types=[Int16],
            read_only_args=[quotients[i + 1], quotients[i]],
        )
        for i in range(0, len(quotients), 2)
    ]
    if len(pairs) == 1:
        return pairs
    return [
        cute.arch.inline_ptx(
            "mov.b32 {$w0}, {{$r0}, {$r1}};",
            write_only_types=[Int32],
            read_only_args=[pairs[i], pairs[i + 1]],
        )
        for i in range(0, len(pairs), 2)
    ]


def _store_bytes(words, base, offset, pred=None):
    """``words`` (one Int16, or 1-2 Int32) to byte ``base + offset`` as one ``st.global.cs``,
    predicated on ``pred`` when given."""
    address = base.toint() + Int64(offset)
    if len(words) == 2:
        kind, values = ".v2.b32", "{{$r1}, {$r2}}"
    else:
        kind, values = (".b16" if isinstance(words[0], Int16) else ".b32"), "{$r1}"
    cute.arch.inline_ptx(
        f"st.global.cs{kind} [{{$r0}}], {values};", read_only_args=[address, *words], predicate=pred
    )


def _load_scale(scale):
    """The per-tensor FP32 scale at ``scale``."""
    value = cute.make_rmem_tensor((1,), Float32)
    _load(cute.make_copy_atom(_G2R, Float32, num_bits_per_copy=32), value, scale, 0)
    return value[0]


def _gptr(dtype, base, offset, align):
    """``base + offset`` (elements) as a global pointer aligned to ``align`` bytes."""
    return cute.make_ptr(
        dtype,
        base.toint() + Int64(offset) * (dtype.width // 8),
        cute.AddressSpace.gmem,
        assumed_align=align,
    )


def _load(atom, frag, base, offset, policy=None):
    """``size(frag)`` contiguous elements at ``base + offset`` into the register fragment."""
    n = cute.size(frag)
    dtype = frag.element_type
    src = cute.make_tensor(_gptr(dtype, base, offset, n * dtype.width // 8), cute.make_layout(n))
    if policy is None:
        cute.copy(atom, src, frag)
    else:
        cute.copy(atom, src, frag, cache_policy=policy)


def _store(atom, frag, base, offset, policy=None):
    n = cute.size(frag)
    dtype = frag.element_type
    dst = cute.make_tensor(_gptr(dtype, base, offset, n * dtype.width // 8), cute.make_layout(n))
    if policy is None:
        cute.copy(atom, frag, dst)
    else:
        cute.copy(atom, frag, dst, cache_policy=policy)


def _store_if(frag, base, offset, pred, streaming, policy=None):
    """A 16-byte fragment to ``base + offset`` as one predicated ``st.global.v4`` (a CuTe copy
    under a lane condition becomes a branch region, which also fences the scheduler)."""
    words = cute.recast_tensor(frag, Int32)
    address = base.toint() + Int64(offset) * (frag.element_type.width // 8)
    mode = ".cs" if streaming else ""
    if policy is None:
        cute.arch.inline_ptx(
            f"st.global{mode}.v4.b32 [{{$r0}}], {{{{$r1}}, {{$r2}}, {{$r3}}, {{$r4}}}};",
            read_only_args=[address, words[0], words[1], words[2], words[3]],
            predicate=pred,
        )
    else:
        cute.arch.inline_ptx(
            f"st.global{mode}.L2::cache_hint.v4.b32 [{{$r0}}], "
            f"{{{{$r1}}, {{$r2}}, {{$r3}}, {{$r4}}}}, {{$r5}};",
            read_only_args=[address, words[0], words[1], words[2], words[3], policy],
            predicate=pred,
        )


def _st_at(address, words, pred=None):
    """``words`` (one Int16, or 1, 2 or 4 Int32) at byte ``address`` as one ``st.global.cs``."""
    if isinstance(words[0], Int16):
        kind, values = ".b16", "{$r1}"
    else:
        n = len(words)
        kind = {1: ".b32", 2: ".v2.b32", 4: ".v4.b32"}[n]
        values = ", ".join(f"{{$r{i + 1}}}" for i in range(n))
        values = values if n == 1 else "{" + values + "}"
    cute.arch.inline_ptx(
        f"st.global.cs{kind} [{{$r0}}], {values};", read_only_args=[address, *words], predicate=pred
    )


def _words(frag):
    """A register fragment as its 32-bit words."""
    words = cute.recast_tensor(frag, Int32)
    return [words[i] for i in range(cute.size(words))]


class Kernel:
    """One kernel per head counts and launch configuration: Q, K and V heads per warp
    (``q_rows``, ``k_rows``, ``v_rows``), ``warps`` per program, and ``stream`` (L2 hints for
    calls that move a large share of L2). ``cache`` is None (BF16 caches) or the FP8 format
    ("e4m3", "e5m2"); FP8 caches are bytes, their strides in bytes. Strides are 32-bit; rows
    and slots are addressed in 64 bits. ``masked`` kernels skip negative slots' cache writes."""

    def __init__(
        self, q_heads, kv_heads, q_rows, k_rows, v_rows, warps, stream,
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM, zero_centered=True, cache=None, masked=False,
        axes=None,
    ):  # fmt: skip
        self.head_dim, self.rotary_dim, self.zero_centered = head_dim, rotary_dim, zero_centered
        self.cache, self.masked, self.axes = cache, masked, axes
        # Per-frequency position axes (multi-axis rotary) as two bit masks: axis bit 0, bit 1.
        if axes is not None:
            self.axis_bits = tuple(
                sum(((axis >> bit) & 1) << f for f, axis in enumerate(axes)) for bit in (0, 1)
            )
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.q_rows, self.k_rows, self.v_rows = q_rows, k_rows, v_rows
        self.warps, self.stream = warps, stream

    @cute.jit
    def __call__(
        self,
        qkv: cute.Tensor,
        q_weight: cute.Tensor,
        k_weight: cute.Tensor,
        cos_sin: cute.Tensor,
        positions: cute.Tensor,
        slots: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        k_scale: cute.Tensor,
        v_scale: cute.Tensor,
        eps: Float32,
        q_programs: Int32,
        k_programs: Int32,
        programs: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            qkv.iterator, q_weight.iterator, k_weight.iterator, cos_sin.iterator,
            positions.iterator, slots.iterator, k_cache.iterator, v_cache.iterator,
            k_scale.iterator, v_scale.iterator, eps,
            qkv.shape[0], q_programs, k_programs, qkv.stride[0], cos_sin.stride[0],
            k_cache.stride[0], k_cache.stride[1], v_cache.stride[0], v_cache.stride[1],
        ).launch(
            grid=[programs, 1, 1], block=[32 * self.warps, 1, 1], stream=stream, use_pdl=True
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        qkv: cute.Pointer,
        q_weight: cute.Pointer,
        k_weight: cute.Pointer,
        cos_sin: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        k_cache: cute.Pointer,
        v_cache: cute.Pointer,
        k_scale: cute.Pointer,
        v_scale: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        q_programs: Int32,
        k_programs: Int32,
        qkv_stride: Int32,
        cos_sin_stride: Int32,
        k_slot_stride: Int32,
        k_head_stride: Int32,
        v_slot_stride: Int32,
        v_head_stride: Int32,
    ):
        # Programmatic dependent launch: nothing is read before the previous kernel completes.
        cute.arch.griddepcontrol_wait()
        program, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if program < q_programs:
            self.norm_rope(
                qkv, q_weight, cos_sin, positions, slots, qkv, k_scale, eps, tokens,
                program * self.warps + warp, lane, qkv_stride, cos_sin_stride, k_slot_stride,
                k_head_stride, self.q_heads, self.q_rows, 0, True,
            )  # fmt: skip
        elif program < q_programs + k_programs:
            self.norm_rope(
                qkv, k_weight, cos_sin, positions, slots, k_cache, k_scale, eps, tokens,
                (program - q_programs) * self.warps + warp, lane, qkv_stride, cos_sin_stride,
                k_slot_stride, k_head_stride, self.kv_heads, self.k_rows, self.q_heads, False,
            )  # fmt: skip
        else:
            self.copy_v(
                qkv, slots, v_cache, v_scale, tokens,
                (program - q_programs - k_programs) * self.warps + warp, lane, qkv_stride,
                v_slot_stride, v_head_stride,
            )  # fmt: skip
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def norm_rope(
        self,
        qkv: cute.Pointer,
        weight: cute.Pointer,
        cos_sin: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        out: cute.Pointer,
        scale_ptr: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        group: Int32,
        lane: Int32,
        qkv_stride,
        cos_sin_stride,
        out_slot_stride,
        out_head_stride,
        heads: cutlass.Constexpr,
        rows: cutlass.Constexpr,
        head_offset: cutlass.Constexpr,
        in_place: cutlass.Constexpr,
    ):
        if const_expr(self.head_dim == 128):
            self.norm_rope_128(
                qkv, weight, cos_sin, positions, slots, out, scale_ptr, eps, tokens, group, lane,
                qkv_stride, cos_sin_stride, out_slot_stride, out_head_stride,
                heads, rows, head_offset, in_place,
            )  # fmt: skip
        else:
            groups_per_token = heads // rows
            if group < tokens * groups_per_token:
                # Unsigned: group >= 0, so no floor-division correction before the first load.
                token = Int32(cutlass.Uint32(group) // cutlass.Uint32(groups_per_token))
                head0 = (group - token * groups_per_token) * rows
                half = ROTARY_DIM // 2
                ld_position = cute.make_copy_atom(_G2R, Int64, num_bits_per_copy=64)
                ld_index = cute.make_copy_atom(_G2R, Int32, num_bits_per_copy=32)
                ld_table = cute.make_copy_atom(_G2R, Float32, num_bits_per_copy=64)
                ld128_w = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=128)
                ld32_w = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=32)

                wide = lane >= 8
                wide_dim = ROTARY_DIM + 8 * (lane - 8)
                # Branch-free loads: lanes 0-7 (no wide dims) read the first wide chunk too and drop
                # it by a select, so no load sits in a branch ahead of the cos/sin loads.
                load_dim = ROTARY_DIM + 8 * cutlass.max(lane - 8, 0)
                rot_dim = 2 * lane
                angle = rot_dim % half

                row = Int64(token) * Int64(qkv_stride) + Int64((head_offset + head0) * HEAD_DIM)
                # Issue order: the dependent chain first (position -> cos/sin row), then the rows,
                # the slot and the weight; the cos/sin loads last (they wait for the position).
                index = cute.make_rmem_tensor((1,), Int64)
                if const_expr(self.axes is None):
                    _load(ld_position, index, positions, token)
                position = index[0]
                cos = cute.make_rmem_tensor((2,), Float32)
                sin = cute.make_rmem_tensor((2,), Float32)
                w_wide = cute.make_rmem_tensor((8,), BFloat16)
                w_rot = cute.make_rmem_tensor((2,), BFloat16)
                x_wide = cute.make_rmem_tensor(
                    cute.make_ordered_layout((rows, 8), order=(1, 0)), BFloat16
                )
                x_rot = cute.make_rmem_tensor(
                    cute.make_ordered_layout((rows, 2), order=(1, 0)), BFloat16
                )
                self.load_rows(qkv, x_wide, x_rot, row, load_dim, rot_dim, rows)
                slot = Int64(0)
                scale = Float32(1.0)
                if const_expr(not in_place):
                    slot_index = cute.make_rmem_tensor((1,), Int32)
                    _load(ld_index, slot_index, slots, token)
                    slot = Int64(slot_index[0])
                    if const_expr(self.cache is not None):
                        scale = _load_scale(scale_ptr)
                _load(ld128_w, w_wide, weight, load_dim)
                _load(ld32_w, w_rot, weight, rot_dim)
                if const_expr(self.axes is None):
                    table = position * Int64(cos_sin_stride) + angle
                    _load(ld_table, cos, cos_sin, table)
                    _load(ld_table, sin, cos_sin, table + half)
                else:
                    self.axis_cos_sin(
                        cos, sin, cos_sin, positions, token, tokens, angle, cos_sin_stride, half
                    )
                ww = w_wide.load().to(Float32) + 1.0
                wr = w_rot.load().to(Float32) + 1.0
                if const_expr(self.masked and not in_place):
                    self.finish_k_masked(
                        x_wide, x_rot, ww, wr, cos, sin, out, scale, eps, lane, wide, wide_dim,
                        rot_dim, slot, head0, out_slot_stride, out_head_stride, rows,
                    )  # fmt: skip
                else:
                    self.finish_rows(
                        x_wide, x_rot, ww, wr, cos, sin, out, scale, eps, lane, wide, wide_dim,
                        rot_dim, row, slot, head0, out_slot_stride, out_head_stride, rows,
                        in_place,
                    )  # fmt: skip

    @cute.jit
    def load_rows(
        self,
        qkv: cute.Pointer,
        x_wide: cute.Tensor,
        x_rot: cute.Tensor,
        row: Int64,
        load_dim: Int32,
        rot_dim: Int32,
        rows: cutlass.Constexpr,
    ):
        # Streaming calls: the qkv heads are dead once read (K copied out, Q rewritten), so they
        # are loaded evict-first.
        load_policy = _l2_policy("evict_first") if const_expr(self.stream) else None
        ld128 = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=128)
        ld32 = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=32)
        for r in cutlass.range_constexpr(rows):
            _load(ld128, x_wide[r, None], qkv, row + r * HEAD_DIM + load_dim, load_policy)
            _load(ld32, x_rot[r, None], qkv, row + r * HEAD_DIM + rot_dim, load_policy)

    @cute.jit
    def finish_rows(
        self,
        x_wide: cute.Tensor,
        x_rot: cute.Tensor,
        ww: cute.TensorSSA,
        wr: cute.TensorSSA,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Pointer,
        scale: Float32,
        eps: Float32,
        lane: Int32,
        wide: cutlass.Boolean,
        wide_dim: Int32,
        rot_dim: Int32,
        row: Int64,
        slot: Int64,
        head0: Int32,
        out_slot_stride,
        out_head_stride,
        rows: cutlass.Constexpr,
        in_place: cutlass.Constexpr,
    ):
        store_mode = (
            cute.nvgpu.StoreCacheMode.WRITE_BACK
            if const_expr(in_place)
            else cute.nvgpu.StoreCacheMode.STREAMING
        )
        st32 = cute.make_copy_atom(
            _R2G, BFloat16, num_bits_per_copy=32, store_cache_mode=store_mode
        )
        # Streaming calls store Q, which attention reads next, evict-last.
        q_policy = _l2_policy("evict_last") if const_expr(self.stream and in_place) else None

        # Per-lane partial sums of the squares, then one butterfly over the warp.
        inv = cute.make_rmem_tensor((rows,), Float32)
        for r in cutlass.range_constexpr(rows):
            xw = x_wide[r, None].load().to(Float32)
            xr = x_rot[r, None].load().to(Float32)
            part = (xw * xw).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            part = (part if wide else Float32(0.0)) + xr[0] * xr[0] + xr[1] * xr[1]
            for k in cutlass.range_constexpr(5):
                part = part + cute.arch.shuffle_sync_bfly(part, offset=16 >> k)
            inv[r] = (
                cute.math.rsqrt(part * (1.0 / HEAD_DIM) + eps, fastmath=True) * _RSQRT_NO_OVERSHOOT
            )

        for r in cutlass.range_constexpr(rows):
            if const_expr(in_place):
                target = row + r * HEAD_DIM
            else:
                target = slot * Int64(out_slot_stride) + Int64((head0 + r) * out_head_stride)
            # Dims past the rotary part: done before the table row is needed.
            values = x_wide[r, None].load().to(Float32) * inv[r] * ww
            if const_expr(not in_place and self.cache is not None):
                words = _quantize([values[i] for i in range(8)], scale, self.cache)
                _store_bytes(words, out, target + wide_dim, wide)
            else:
                out_wide = cute.make_rmem_tensor((8,), BFloat16)
                out_wide.store(values.to(BFloat16))
                _store_if(out_wide, out, target + wide_dim, wide, not in_place, q_policy)

        for r in cutlass.range_constexpr(rows):
            if const_expr(in_place):
                target = row + r * HEAD_DIM
            else:
                target = slot * Int64(out_slot_stride) + Int64((head0 + r) * out_head_stride)
            # Rotary dims: norm and partner (d +- 32, lane ^ 16) first, then the table.
            n0 = Float32(x_rot[r, 0]) * inv[r] * wr[0]
            n1 = Float32(x_rot[r, 1]) * inv[r] * wr[1]
            p0 = cute.arch.shuffle_sync_bfly(n0, offset=16)
            p1 = cute.arch.shuffle_sync_bfly(n1, offset=16)
            if lane < 16:
                p0 = -p0
                p1 = -p1
            r0 = n0 * cos[0] + p0 * sin[0]
            r1 = n1 * cos[1] + p1 * sin[1]
            if const_expr(not in_place and self.cache is not None):
                _store_bytes(_quantize([r0, r1], scale, self.cache), out, target + rot_dim)
            else:
                out_rot = cute.make_rmem_tensor((2,), BFloat16)
                out_rot[0] = r0.to(BFloat16)
                out_rot[1] = r1.to(BFloat16)
                _store(st32, out_rot, out, target + rot_dim, q_policy)

    @cute.jit
    def finish_k_masked(
        self,
        x_wide: cute.Tensor,
        x_rot: cute.Tensor,
        ww: cute.TensorSSA,
        wr: cute.TensorSSA,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Pointer,
        scale: Float32,
        eps: Float32,
        lane: Int32,
        wide: cutlass.Boolean,
        wide_dim: Int32,
        rot_dim: Int32,
        slot: Int64,
        head0: Int32,
        out_slot_stride,
        out_head_stride,
        rows: cutlass.Constexpr,
    ):
        """``finish_rows`` for K heads of the masked kernel. The cache addresses and stores
        are under branches on ``slot >= 0`` (warp-uniform: one token)."""
        size = 1 if const_expr(self.cache is not None) else 2
        dst = cute.make_rmem_tensor((rows, 2), Int64)
        if const_expr(rows == 1):
            # One head per warp (small, latency-bound calls): the address first, off the
            # critical path; with more heads per warp it waits for the stores (measured).
            if slot >= 0:
                self.k_addresses(dst, out, slot, head0, wide_dim, rot_dim, size, out_slot_stride,
                                 out_head_stride, rows)  # fmt: skip
        inv = cute.make_rmem_tensor((rows,), Float32)
        for r in cutlass.range_constexpr(rows):
            xw = x_wide[r, None].load().to(Float32)
            xr = x_rot[r, None].load().to(Float32)
            part = (xw * xw).reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
            part = (part if wide else Float32(0.0)) + xr[0] * xr[0] + xr[1] * xr[1]
            for k in cutlass.range_constexpr(5):
                part = part + cute.arch.shuffle_sync_bfly(part, offset=16 >> k)
            inv[r] = (
                cute.math.rsqrt(part * (1.0 / HEAD_DIM) + eps, fastmath=True) * _RSQRT_NO_OVERSHOOT
            )
        wide_out, rot_out = [], []
        for r in cutlass.range_constexpr(rows):
            values = x_wide[r, None].load().to(Float32) * inv[r] * ww
            n0 = Float32(x_rot[r, 0]) * inv[r] * wr[0]
            n1 = Float32(x_rot[r, 1]) * inv[r] * wr[1]
            p0 = cute.arch.shuffle_sync_bfly(n0, offset=16)
            p1 = cute.arch.shuffle_sync_bfly(n1, offset=16)
            if lane < 16:
                p0 = -p0
                p1 = -p1
            r0 = n0 * cos[0] + p0 * sin[0]
            r1 = n1 * cos[1] + p1 * sin[1]
            if const_expr(self.cache is not None):
                wide_out.append(_quantize([values[i] for i in range(8)], scale, self.cache))
                rot_out.append(_quantize([r0, r1], scale, self.cache))
            else:
                out_wide = cute.make_rmem_tensor((8,), BFloat16)
                out_wide.store(values.to(BFloat16))
                out_rot = cute.make_rmem_tensor((2,), BFloat16)
                out_rot[0] = r0.to(BFloat16)
                out_rot[1] = r1.to(BFloat16)
                wide_out.append(_words(out_wide))
                rot_out.append(_words(out_rot))
        if slot >= 0:
            if const_expr(rows != 1):
                self.k_addresses(dst, out, slot, head0, wide_dim, rot_dim, size, out_slot_stride,
                                 out_head_stride, rows)  # fmt: skip
            for r in cutlass.range_constexpr(rows):
                _st_at(dst[r, 0], wide_out[r], wide)
                _st_at(dst[r, 1], rot_out[r])

    @cute.jit
    def k_addresses(
        self,
        dst: cute.Tensor,
        out: cute.Pointer,
        slot: Int64,
        head0: Int32,
        first: Int32,
        second: Int32,
        size: cutlass.Constexpr,
        out_slot_stride,
        out_head_stride,
        rows: cutlass.Constexpr,
    ):
        """Byte addresses of each K head row's two fragments (dims ``first``, ``second``) in
        the cache slot ``slot``; called only for ``slot >= 0``."""
        for r in cutlass.range_constexpr(rows):
            head = slot * Int64(out_slot_stride) + Int64((head0 + r) * out_head_stride)
            dst[r, 0] = out.toint() + (head + first) * size
            dst[r, 1] = out.toint() + (head + second) * size

    @cute.jit
    def copy_v(
        self,
        qkv: cute.Pointer,
        slots: cute.Pointer,
        v_out: cute.Pointer,
        scale_ptr: cute.Pointer,
        tokens: Int32,
        group: Int32,
        lane: Int32,
        qkv_stride,
        v_slot_stride,
        v_head_stride,
    ):
        width = self.head_dim // 32
        rows = self.v_rows
        groups_per_token = self.kv_heads // rows
        if group < tokens * groups_per_token:
            token = Int32(cutlass.Uint32(group) // cutlass.Uint32(groups_per_token))
            head0 = (group - token * groups_per_token) * rows
            ld128 = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=width * 16)
            load_policy = _l2_policy("evict_first") if const_expr(self.stream) else None
            ld_index = cute.make_copy_atom(_G2R, Int32, num_bits_per_copy=32)
            row = Int64(token) * Int64(qkv_stride) + Int64(
                (self.q_heads + self.kv_heads + head0) * self.head_dim
            )
            v_raw = cute.make_rmem_tensor(
                cute.make_ordered_layout((rows, width), order=(1, 0)), BFloat16
            )
            for r in cutlass.range_constexpr(rows):
                _load(
                    ld128, v_raw[r, None], qkv, row + r * self.head_dim + width * lane, load_policy
                )
            slot_index = cute.make_rmem_tensor((1,), Int32)
            _load(ld_index, slot_index, slots, token)
            slot = Int64(slot_index[0])
            scale = Float32(1.0)
            if const_expr(self.cache is not None):
                scale = _load_scale(scale_ptr)
            if const_expr(self.masked):
                # A negative slot (padding): no address, no store (warp-uniform branch).
                if slot >= 0:
                    self.store_v(
                        v_raw, v_out, scale, slot, head0, lane, v_slot_stride, v_head_stride
                    )
            else:
                self.store_v(v_raw, v_out, scale, slot, head0, lane, v_slot_stride, v_head_stride)

    @cute.jit
    def store_v(
        self,
        v_raw: cute.Tensor,
        v_out: cute.Pointer,
        scale: Float32,
        slot: Int64,
        head0: Int32,
        lane: Int32,
        v_slot_stride,
        v_head_stride,
    ):
        width = self.head_dim // 32
        st128 = cute.make_copy_atom(
            _R2G,
            BFloat16,
            num_bits_per_copy=width * 16,
            store_cache_mode=cute.nvgpu.StoreCacheMode.STREAMING,
        )
        for r in cutlass.range_constexpr(self.v_rows):
            target = slot * Int64(v_slot_stride) + Int64((head0 + r) * v_head_stride)
            if const_expr(self.cache is not None):
                values = v_raw[r, None].load().to(Float32)
                words = _quantize([values[i] for i in range(width)], scale, self.cache)
                _store_bytes(words, v_out, target + width * lane)
            else:
                _store(st128, v_raw[r, None], v_out, target + width * lane)

    @cute.jit
    def axis_cos_sin(
        self, cos, sin, cos_sin, positions, token, tokens, frequency, cos_sin_stride, half
    ):
        """cos/sin of frequencies ``frequency`` and ``frequency + 1`` from the table rows of
        their own position axes: ``positions`` is the flat contiguous [3, tokens]."""
        ld_position = cute.make_copy_atom(_G2R, Int64, num_bits_per_copy=64)
        ld_one = cute.make_copy_atom(_G2R, Float32, num_bits_per_copy=32)
        low, high = cutlass.Uint64(self.axis_bits[0]), cutlass.Uint64(self.axis_bits[1])
        for i in cutlass.range_constexpr(2):
            f = frequency + i
            shift = cutlass.Uint64(f)
            axis = ((low >> shift) & 1) | (((high >> shift) & 1) << 1)
            index = cute.make_rmem_tensor((1,), Int64)
            _load(ld_position, index, positions, Int64(axis) * Int64(tokens) + Int64(token))
            value = cute.make_rmem_tensor((1,), Float32)
            table = index[0] * Int64(cos_sin_stride) + f
            _load(ld_one, value, cos_sin, table)
            cos[i] = value[0]
            _load(ld_one, value, cos_sin, table + half)
            sin[i] = value[0]

    @cute.jit
    def norm_rope_128(
        self,
        qkv: cute.Pointer,
        weight: cute.Pointer,
        cos_sin: cute.Pointer,
        positions: cute.Pointer,
        slots: cute.Pointer,
        out: cute.Pointer,
        scale_ptr: cute.Pointer,
        eps: Float32,
        tokens: Int32,
        group: Int32,
        lane: Int32,
        qkv_stride,
        cos_sin_stride,
        out_slot_stride,
        out_head_stride,
        heads: cutlass.Constexpr,
        rows: cutlass.Constexpr,
        head_offset: cutlass.Constexpr,
        in_place: cutlass.Constexpr,
    ):
        groups_per_token = heads // rows
        if group < tokens * groups_per_token:
            token = Int32(cutlass.Uint32(group) // cutlass.Uint32(groups_per_token))
            head0 = (group - token * groups_per_token) * rows
            dim = 2 * lane
            half = self.rotary_dim // 2
            row = Int64(token) * Int64(qkv_stride) + Int64((head_offset + head0) * self.head_dim)
            ld_position = cute.make_copy_atom(_G2R, Int64, num_bits_per_copy=64)
            ld_index = cute.make_copy_atom(_G2R, Int32, num_bits_per_copy=32)
            ld32 = cute.make_copy_atom(_G2R, BFloat16, num_bits_per_copy=32)
            ld_table = cute.make_copy_atom(_G2R, Float32, num_bits_per_copy=64)
            load_policy = _l2_policy("evict_first") if const_expr(self.stream) else None

            # Issue the dependent position chain first, then rows, slot, weights and table.
            index = cute.make_rmem_tensor((1,), Int64)
            if const_expr(self.axes is None):
                _load(ld_position, index, positions, token)
            position = index[0]
            layout = cute.make_ordered_layout((rows, 2), order=(1, 0))
            x_lo = cute.make_rmem_tensor(layout, BFloat16)
            x_hi = cute.make_rmem_tensor(layout, BFloat16)
            for r in cutlass.range_constexpr(rows):
                _load(ld32, x_lo[r, None], qkv, row + r * self.head_dim + dim, load_policy)
                _load(ld32, x_hi[r, None], qkv, row + r * self.head_dim + half + dim, load_policy)
            slot = Int64(0)
            scale = Float32(1.0)
            if const_expr(not in_place):
                slot_index = cute.make_rmem_tensor((1,), Int32)
                _load(ld_index, slot_index, slots, token)
                slot = Int64(slot_index[0])
                if const_expr(self.cache is not None):
                    scale = _load_scale(scale_ptr)
            w_lo = cute.make_rmem_tensor((2,), BFloat16)
            w_hi = cute.make_rmem_tensor((2,), BFloat16)
            _load(ld32, w_lo, weight, dim)
            _load(ld32, w_hi, weight, half + dim)
            cos = cute.make_rmem_tensor((2,), Float32)
            sin = cute.make_rmem_tensor((2,), Float32)
            if const_expr(self.axes is None):
                table = position * Int64(cos_sin_stride) + dim
                _load(ld_table, cos, cos_sin, table)
                _load(ld_table, sin, cos_sin, table + half)
            else:
                self.axis_cos_sin(
                    cos, sin, cos_sin, positions, token, tokens, dim, cos_sin_stride, half
                )
            wl = w_lo.load().to(Float32)
            wh = w_hi.load().to(Float32)

            store_mode = (
                cute.nvgpu.StoreCacheMode.WRITE_BACK
                if const_expr(in_place)
                else cute.nvgpu.StoreCacheMode.STREAMING
            )
            st32 = cute.make_copy_atom(
                _R2G, BFloat16, num_bits_per_copy=32, store_cache_mode=store_mode
            )
            q_policy = _l2_policy("evict_last") if const_expr(self.stream and in_place) else None
            size = 1 if const_expr(self.cache is not None) else 2
            dst = cute.make_rmem_tensor((rows, 2), Int64)
            if const_expr(self.masked and not in_place and rows == 1):
                # The cache addresses early for one head per warp (see finish_k_masked).
                if slot >= 0:
                    self.k_addresses(dst, out, slot, head0, dim, half + dim, size,
                                     out_slot_stride, out_head_stride, rows)  # fmt: skip
            inv = cute.make_rmem_tensor((rows,), Float32)
            for r in cutlass.range_constexpr(rows):
                lo = x_lo[r, None].load().to(Float32)
                hi = x_hi[r, None].load().to(Float32)
                part = lo[0] * lo[0] + lo[1] * lo[1] + hi[0] * hi[0] + hi[1] * hi[1]
                for k in cutlass.range_constexpr(5):
                    part = part + cute.arch.shuffle_sync_bfly(part, offset=16 >> k)
                inv[r] = (
                    cute.math.rsqrt(part * (1.0 / self.head_dim) + eps, fastmath=True)
                    * _RSQRT_NO_OVERSHOOT
                )
            if const_expr(self.masked and not in_place):
                # K of the masked kernel: every value first, then the addresses and stores in
                # one warp-uniform branch on slot >= 0.
                lo_out, hi_out = [], []
                for r in cutlass.range_constexpr(rows):
                    lo = x_lo[r, None].load().to(Float32) * inv[r] * wl
                    hi = x_hi[r, None].load().to(Float32) * inv[r] * wh
                    rot_lo = lo * cos.load() - hi * sin.load()
                    rot_hi = hi * cos.load() + lo * sin.load()
                    if const_expr(self.cache is not None):
                        lo_out.append(_quantize([rot_lo[0], rot_lo[1]], scale, self.cache))
                        hi_out.append(_quantize([rot_hi[0], rot_hi[1]], scale, self.cache))
                    else:
                        out_lo = cute.make_rmem_tensor((2,), BFloat16)
                        out_hi = cute.make_rmem_tensor((2,), BFloat16)
                        out_lo.store(rot_lo.to(BFloat16))
                        out_hi.store(rot_hi.to(BFloat16))
                        lo_out.append(_words(out_lo))
                        hi_out.append(_words(out_hi))
                if slot >= 0:
                    if const_expr(rows != 1):
                        self.k_addresses(dst, out, slot, head0, dim, half + dim, size,
                                         out_slot_stride, out_head_stride, rows)  # fmt: skip
                    for r in cutlass.range_constexpr(rows):
                        _st_at(dst[r, 0], lo_out[r])
                        _st_at(dst[r, 1], hi_out[r])
            else:
                for r in cutlass.range_constexpr(rows):
                    if const_expr(in_place):
                        target = row + r * self.head_dim
                    else:
                        target = slot * Int64(out_slot_stride) + Int64(
                            (head0 + r) * out_head_stride
                        )
                    lo = x_lo[r, None].load().to(Float32) * inv[r] * wl
                    hi = x_hi[r, None].load().to(Float32) * inv[r] * wh
                    rot_lo = lo * cos.load() - hi * sin.load()
                    rot_hi = hi * cos.load() + lo * sin.load()
                    if const_expr(not in_place and self.cache is not None):
                        words = _quantize([rot_lo[0], rot_lo[1]], scale, self.cache)
                        _store_bytes(words, out, target + dim)
                        words = _quantize([rot_hi[0], rot_hi[1]], scale, self.cache)
                        _store_bytes(words, out, target + half + dim)
                    else:
                        out_lo = cute.make_rmem_tensor((2,), BFloat16)
                        out_hi = cute.make_rmem_tensor((2,), BFloat16)
                        out_lo.store(rot_lo.to(BFloat16))
                        out_hi.store(rot_hi.to(BFloat16))
                        _store(st32, out_lo, out, target + dim, q_policy)
                        _store(st32, out_hi, out, target + half + dim, q_policy)
