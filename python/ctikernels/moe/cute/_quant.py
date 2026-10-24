# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Group-32 integer BF16 -> E4M3FN/UE8M0, with no FP denormal dependency.

Four lanes own a group; each loads eight BF16 bits in one 128-bit copy.
The exponent/significand algorithm matches the package's Triton quant_bf16.
Finite inputs only: NaN/Inf have no portable output in the declared ABI.
"""

from functools import lru_cache

import torch

from .._common import Quantized, frozen
from ._runtime import require


@lru_cache(maxsize=1)
def _encoder():
    """Four adjacent lanes encode one K32 group from eight BF16 bits each.

    All warp lanes must participate, including masked groups. Integer arithmetic
    preserves BF16 subnormals and signed underflow without FP denormal modes.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint8

    @cute.jit
    def encode(raw: cute.Tensor):
        peak = Int32(0)
        for j in cutlass.range_constexpr(8):
            peak = cutlass.max(peak, Int32(raw[j]) & 32767)
        # Every lane executes both shuffles, including out-of-bounds groups.
        peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, offset=1))
        peak = cutlass.max(peak, cute.arch.shuffle_sync_bfly(peak, offset=2))
        scale = cutlass.min(cutlass.max((peak >> 7) - 8 + Int32((peak & 127) > 96), 0), 254)
        if peak == 0:
            scale = Int32(127)
        packed = cute.make_rmem_tensor((8,), Uint8)
        for j in cutlass.range_constexpr(8):
            bits = Int32(raw[j])
            mag = bits & 32767
            ef = mag >> 7
            sig = (mag & 127) | (Int32(ef > 0) * 128)
            shift = Int32(cute.arch.clz(sig)) - 24
            sig = sig << shift
            exponent = cutlass.max(ef, 1) - scale - shift
            discard = cutlass.min(cutlass.max(4, -exponent - 2), 24)
            kept = sig >> discard
            rest = sig & ((Int32(1) << discard) - 1)
            half = Int32(1) << (discard - 1)
            rounded = kept + Int32((rest > half) | ((rest == half) & ((kept & 1) != 0)))
            code = rounded
            if exponent >= -6:
                code = (exponent + 6) * 8 + rounded
            code = cutlass.min(code, 126) | ((bits >> 8) & 128)
            if peak == 0:
                code = Int32(0)
            packed[j] = Uint8(code)
        return packed, scale

    return encode


@lru_cache(maxsize=1)
def _bf16_encoder():
    """Per-value form of the same integer encoder, for fused W4A8 Q1 epilogues."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint8

    @cute.jit
    def scale_from_peak(peak):
        scale = cutlass.min(cutlass.max((peak >> 7) - 8 + Int32((peak & 127) > 96), 0), 254)
        if peak == 0:
            scale = Int32(127)
        return scale

    @cute.jit
    def encode_bits(bits, peak, scale):
        mag = bits & 32767
        ef = mag >> 7
        sig = (mag & 127) | (Int32(ef > 0) * 128)
        shift = Int32(cute.arch.clz(sig)) - 24
        sig = sig << shift
        exponent = cutlass.max(ef, 1) - scale - shift
        discard = cutlass.min(cutlass.max(4, -exponent - 2), 24)
        kept = sig >> discard
        rest = sig & ((Int32(1) << discard) - 1)
        half = Int32(1) << (discard - 1)
        rounded = kept + Int32((rest > half) | ((rest == half) & ((kept & 1) != 0)))
        code = rounded
        if exponent >= -6:
            code = (exponent + 6) * 8 + rounded
        code = cutlass.min(code, 126) | ((bits >> 8) & 128)
        if peak == 0:
            code = Int32(0)
        return Uint8(code)

    return scale_from_peak, encode_bits


@lru_cache(maxsize=1)
def _kernel_type():
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint8, Uint16

    encode = _encoder()

    class QuantKernel:
        def __init__(self, threads=128, aligned=True):
            self.threads = threads
            self.aligned = aligned

        @cute.jit
        def __call__(self, x: cute.Tensor, q: cute.Tensor, s: cute.Tensor, stream: cuda.CUstream):
            groups = cute.size(x) // 32
            self.kernel(x.iterator, q.iterator, s.iterator, groups).launch(
                grid=[cute.ceil_div(groups, self.threads // 4), 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, x: cute.Pointer, q: cute.Pointer, s: cute.Pointer, groups: Int32):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            group = block * (self.threads // 4) + thread // 4
            offset = group * 32 + (thread % 4) * 8
            raw = cute.make_rmem_tensor((8,), Uint16)
            raw.fill(0)
            if group < groups:
                if cutlass.const_expr(self.aligned):
                    src = cute.make_tensor(
                        cute.make_ptr(
                            Uint16,
                            x.toint() + Int64(offset) * 2,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(8),
                    )
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyG2ROp(), Uint16, num_bits_per_copy=128
                    )
                    cute.copy(atom, src, raw)
                else:
                    for j in cutlass.range_constexpr(8):
                        raw[j] = x[offset + j]
            packed, scale = encode(raw)
            if group < groups:
                dst = cute.make_tensor(
                    cute.make_ptr(
                        Uint8,
                        q.toint() + Int64(offset),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    ),
                    cute.make_layout(8),
                )
                store = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), Uint8, num_bits_per_copy=64)
                cute.copy(store, packed, dst)
                if thread % 4 == 0:
                    s[group] = Uint8(scale)

    return QuantKernel


def mxfp8_quantize(x):
    """Same Quantized(values, scales) ABI as triton._launch.quantize."""
    frozen(x)
    if x.ndim != 2 or x.shape[1] <= 0 or x.shape[1] % 32:
        raise ValueError("quantizer requires [M,K], K a positive multiple of 32")
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("quantizer requires contiguous BF16")
    runtime = require(x.device)
    values = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((x.shape[0], x.shape[1] // 32), dtype=torch.uint8, device=x.device)
    if x.numel():
        args = (x.view(torch.uint16), values.view(torch.uint8), scales)
        aligned = x.data_ptr() % 16 == 0
        kernel = _kernel_type()(128, aligned)
        compiled = runtime.compile(("quant", 128, x.shape[1], aligned), kernel, args)
        compiled(*args)
    return Quantized(values, scales)
