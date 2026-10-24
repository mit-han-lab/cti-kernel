# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Flat, vectorized SwiGLU. Original CuTe implementation; design references:
SGLang 72730d6360a5894b167500d5d1ad82c34c6bb116 activation.cuh (flat 16/32-byte
vectors/PDL), vLLM 0.31.0 SiluAndMul.forward_native (fused pointwise work).
Only documented CuTe operations and public PTX are used.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T


@cute.jit
def divide(a, b):
    # Explicit IEEE division, with gradual underflow, regardless of compiler flags.
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def vector_at(base, offset, elems: cutlass.Constexpr):
    itemsize = base.value_type.width // 8
    ptr = cute.make_ptr(
        base.value_type,
        base.toint() + Int64(offset) * itemsize,
        cute.AddressSpace.gmem,
        assumed_align=elems * itemsize,
    )
    return cute.make_tensor(ptr, cute.make_layout(elems))


def l2_policy(kind):
    # Same public createpolicy pattern as attention/_prologue_cute.py; sm_80+.
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


def load_with_policy(base, offset, frag, policy):
    # Recast pairs without conversion, as in the attention prologue's vector stores.
    # These Python helpers build a fixed asm signature at trace time (1/2/4 words).
    words = cute.recast_tensor(frag, Int32)
    count = cute.size(words)
    address = base.toint() + Int64(offset) * 2
    vector = "" if count == 1 else f"v{count}."
    registers = "$0" if count == 1 else "{" + ", ".join(f"${i}" for i in range(count)) + "}"
    loaded = llvm.inline_asm(
        T.i32() if count == 1 else llvm.StructType.get_literal([T.i32()] * count),
        [address.ir_value(), policy.ir_value()],
        f"ld.global.L2::cache_hint.{vector}b32 {registers}, [${count}], ${count + 1};",
        ",".join(["=r"] * count + ["l", "l", "~{memory}"]),
        # Keep global reads after griddepcontrol.wait, including with early PDL.
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    for i in range(count):
        words[i] = Int32(loaded if count == 1 else llvm.extractvalue(T.i32(), loaded, [i]))


def store_with_policy(base, offset, frag, policy):
    words = cute.recast_tensor(frag, Int32)
    count = cute.size(words)
    address = base.toint() + Int64(offset) * 2
    vector = "" if count == 1 else f"v{count}."
    registers = "$1" if count == 1 else "{" + ", ".join(f"${i + 1}" for i in range(count)) + "}"
    llvm.inline_asm(
        None,
        [address.ir_value()] + [words[i].ir_value() for i in range(count)] + [policy.ir_value()],
        f"st.global.L2::cache_hint.{vector}b32 [$0], {registers}, ${count + 1};",
        ",".join(["l"] + ["r"] * count + ["l", "~{memory}"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def silu_mul(g: Float32, u: Float32):
    # SGLang's activation: g / (1 + expf(-g)) * u (accurate expf, IEEE division).
    return divide(g, 1.0 + cute.math.exp(-g, fastmath=False)) * u


@cute.jit
def gated(g: Float32, u: Float32, kind: cutlass.Constexpr):
    if cutlass.const_expr(kind == "silu"):
        value = silu_mul(g, u)
    else:
        value = activate(g, kind) * u
    return value


@cute.jit
def activate(g: Float32, kind: cutlass.Constexpr):
    # Math/rounding reference: SGLang 72730d6,
    # python/sglang/kernels/jit/csrc/elementwise/activation.cuh:25-39,104-117.
    # Ordinary activations stay FP32 through the up multiply, then round once.
    if cutlass.const_expr(kind == "relu2"):
        positive = g if g > 0.0 else Float32(0.0)
        value = positive * positive
    elif cutlass.const_expr(kind == "gelu"):
        # GELU(x) = x Phi(x) with Phi(x) = erfc(-x/sqrt(2))/2: unlike 1 + erf, erfc keeps
        # full relative precision in the negative tail (as torch's normcdf form does).
        value = g * (0.5 * cute.math.erfc(-g * 0.7071067811865475, fastmath=False))
    elif cutlass.const_expr(kind == "gelu_tanh"):
        # (1 + tanh(z))/2 == 1/(1 + exp(-2z)): the same function without 1 + tanh's
        # cancellation for negative z. Overflowing exp gives the signed zero limit.
        cubic = ((0.044715 * g) * g) * g
        z = 0.7978845608028654 * (g + cubic)
        value = divide(g, 1.0 + cute.math.exp(-2.0 * z, fastmath=False))
    else:
        value = silu_mul(g, Float32(1.0))
    return value


class Kernel:
    def __init__(self, width, config, *, kind="silu", general=False, lanes=1):
        self.width = width
        self.kind, self.general, self.lanes = kind, general, lanes
        self.general_elems = lanes if lanes > 1 else 4
        self.input_width = width if kind == "relu2" else 2 * width
        self.threads, self.vectors, self.pdl = config.threads, config.vectors, config.pdl
        self.elems = config.elems
        self.trigger = config.trigger
        self.cache = config.cache

    @cute.jit
    def __call__(self, x: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream):
        if cutlass.const_expr(self.general):
            count = Int64(x.shape[0]) * (
                (self.width + self.general_elems - 1) // self.general_elems
            )
            self.general_kernel(
                x.iterator, y.iterator, count, Int64(x.stride[0]), Int64(y.stride[0])
            ).launch(
                grid=[(count + self.threads - 1) // self.threads, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
            )
        else:
            count = Int64(x.shape[0]) * (self.width // self.elems)
            tile = self.threads * self.vectors
            self.kernel(x.iterator, y.iterator, count).launch(
                grid=[(count + tile - 1) // tile, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
                use_pdl=self.pdl,
            )

    @cute.kernel
    def general_kernel(
        self, x: cute.Pointer, y: cute.Pointer, count: Int64, x_stride: Int64, y_stride: Int64
    ):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        index = Int64(block) * self.threads + thread
        n = self.general_elems
        per_row = (self.width + n - 1) // n
        if index < count:
            row, col = index // per_row, (index % per_row) * n
            offset = row * x_stride + col
            if cutlass.const_expr(self.lanes > 1):
                bits = n * x.value_type.width
                load = cute.make_copy_atom(
                    cute.nvgpu.CopyG2ROp(), x.value_type, num_bits_per_copy=bits
                )
                store = cute.make_copy_atom(
                    cute.nvgpu.CopyR2GOp(), y.value_type, num_bits_per_copy=bits
                )
                gate = cute.make_rmem_tensor(n, x.value_type)
                result = cute.make_rmem_tensor(n, y.value_type)
                cute.copy(load, vector_at(x, offset, n), gate)
                if cutlass.const_expr(self.kind != "relu2"):
                    up = cute.make_rmem_tensor(n, x.value_type)
                    cute.copy(load, vector_at(x, offset + self.width, n), up)
                for j in cutlass.range_constexpr(n):
                    if cutlass.const_expr(self.kind == "relu2"):
                        value = activate(Float32(gate[j]), self.kind)
                    else:
                        value = gated(Float32(gate[j]), Float32(up[j]), self.kind)
                    result[j] = value.to(y.value_type)
                cute.copy(store, result, vector_at(y, row * y_stride + col, n))
            else:
                # Scalar accesses still cover several elements per thread, with a tail guard.
                for j in cutlass.range_constexpr(n):
                    if col + j < self.width:
                        g = Float32(vector_at(x, offset + j, 1)[0])
                        if cutlass.const_expr(self.kind == "relu2"):
                            value = activate(g, self.kind)
                        else:
                            u = Float32(vector_at(x, offset + j + self.width, 1)[0])
                            value = gated(g, u, self.kind)
                        destination = vector_at(y, row * y_stride + col + j, 1)
                        destination[0] = value.to(y.value_type)

    @cute.kernel
    def kernel(self, x: cute.Pointer, y: cute.Pointer, count: Int64):
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.pdl and self.trigger == "early"):
            # Earliest safe point: the dependent may start, but its wait still
            # requires this entire grid to complete and flush its output stores.
            cute.arch.griddepcontrol_launch_dependents()
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        n = self.elems
        access = min(n, 8)
        parts = n // access
        bits = access * x.value_type.width
        load = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), x.value_type, num_bits_per_copy=bits)
        store = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), y.value_type, num_bits_per_copy=bits)
        # One policy per thread, reused by both input loads and every vector/half.
        if cutlass.const_expr(self.cache in ("ld_el", "both")):
            load_policy = l2_policy("evict_last")
        if cutlass.const_expr(self.cache in ("st_ef", "both")):
            store_policy = l2_policy("evict_first")
        per_row = self.width // n
        for v in cutlass.range_constexpr(self.vectors):
            index = Int64(block) * (self.threads * self.vectors) + v * self.threads + thread
            # One bounds branch per vector, never a per-element dynamic branch.
            if index < count:
                row = index // per_row
                col = (index % per_row) * n
                # e16 is one contiguous 32-byte vector, backed by two 128-bit
                # copies into adjacent register halves. Each half assumes only
                # the provider's 16-byte alignment, including offset input views.
                gate = cute.make_rmem_tensor((access, parts), x.value_type)
                up = cute.make_rmem_tensor((access, parts), x.value_type)
                result = cute.make_rmem_tensor((access, parts), y.value_type)
                for part in cutlass.range_constexpr(parts):
                    offset = row * (2 * self.width) + col + part * access
                    if cutlass.const_expr(self.cache in ("ld_el", "both")):
                        load_with_policy(x, offset, gate[None, part], load_policy)
                        load_with_policy(x, offset + self.width, up[None, part], load_policy)
                    else:
                        cute.copy(load, vector_at(x, offset, access), gate[None, part])
                        cute.copy(load, vector_at(x, offset + self.width, access), up[None, part])
                for part in cutlass.range_constexpr(parts):
                    gate_part, up_part = gate[None, part], up[None, part]
                    result_part = result[None, part]
                    for j in cutlass.range_constexpr(access):
                        g, u = Float32(gate_part[j]), Float32(up_part[j])
                        result_part[j] = silu_mul(g, u).to(y.value_type)
                for part in cutlass.range_constexpr(parts):
                    if cutlass.const_expr(self.cache in ("st_ef", "both")):
                        store_with_policy(
                            y, index * n + part * access, result[None, part], store_policy
                        )
                    else:
                        cute.copy(
                            store,
                            result[None, part],
                            vector_at(y, index * n + part * access, access),
                        )
        if cutlass.const_expr(self.pdl and self.trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()
