# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe port of our probe's ar_step; two slots, device sequence, rank-order sum."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir.dialects import llvm

from ..._host_exchange.cute.kernels import _ptr, load_raw, put, store_raw


def _add_rn(a, b):
    """Keep the FP32 rounding step: NVVM otherwise folds casts into half adds."""
    return Float32(
        llvm.inline_asm(
            Float32.mlir_type,
            [a.ir_value(), b.ir_value()],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class Step:
    def __init__(self, dtype, ctas, threads, *, gather=False):
        self.dtype, self.ctas, self.threads = dtype, ctas, threads
        self.gather = gather

    @cute.jit
    def __call__(
        self,
        x: Int64,
        out: Int64,
        words: cute.Tensor,
        host: Int64,
        stride: Int64,
        n: Int32,
        rank: Int32,
        stream,
    ):
        self.kernel(x, out, words, host, stride, n, rank).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        out: Int64,
        words: cute.Tensor,
        host: Int64,
        stride: Int64,
        n: Int32,
        rank: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        shared = cutlass.utils.SmemAllocator().allocate_tensor(Int64, cute.make_layout(2), 8)
        word_base = words.iterator.toint()
        seq_ptr = _ptr(Int64, word_base, 8)
        status = _ptr(Int64, word_base + 24, 8)
        if tid == 0:
            put(shared, 0, cute.arch.load(seq_ptr, Int64, sem="acquire", scope="gpu"))
            put(shared, 1, Int64(0))
        cute.arch.sync_threads()
        seq = shared[0]
        slot = seq & Int64(1)
        send = host + slot * stride
        recv = host + (slot + 2) * stride
        flags = host + 4 * stride
        doorbell = _ptr(Int64, flags + slot * 128, 8)
        rflag = _ptr(Int64, flags + (slot + 2) * 128, 8)
        abort = _ptr(Int64, flags + 512, 8)
        i = bid * self.threads + tid
        while i < n:
            off = Int64(i) * 16
            if const_expr(self.gather):
                value = load_raw(x + off)
                store_raw(send + off, value, cop="wt")
                store_raw(out + Int64(rank) * Int64(n) * 16 + off, value)
            else:
                store_raw(send + off, load_raw(x + off), cop="wt")
            i += self.ctas * self.threads
        # Every writer fences, then leaders form a device-scope acq_rel chain.
        cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()
        if tid == 0:
            arrive = _ptr(Int64, word_base + 8, 8)
            ticket = cute.arch.atomic_add(arrive, Int64(1), sem="acq_rel", scope="gpu")
            if ticket == self.ctas - 1:
                cute.arch.store(arrive, Int64(0), sem="release", scope="gpu")
                cute.arch.fence_acq_rel_sys()
                cute.arch.store(doorbell, (seq << 24) | (Int64(n) * 16), sem="release", scope="sys")
            seen = cute.arch.load(rflag, Int64, sem="acquire", scope="sys")
            failed = cute.arch.load(abort, Int64, sem="acquire", scope="sys")
            while (seen != seq) & (failed == 0):
                seen = cute.arch.load(rflag, Int64, sem="acquire", scope="sys")
                failed = cute.arch.load(abort, Int64, sem="acquire", scope="sys")
            put(shared, 1, failed)
            if failed != 0:
                cute.arch.atomic_exch(status, Int64(1), sem="relaxed", scope="gpu")
        cute.arch.sync_threads()
        if shared[1] == 0:
            i = bid * self.threads + tid
            while i < n:
                off = Int64(i) * 16
                if const_expr(self.gather):
                    store_raw(
                        out + Int64(1 - rank) * Int64(n) * 16 + off,
                        load_raw(recv + off, cop="cv"),
                    )
                else:
                    local = cute.TensorSSA(load_raw(x + off), (4,), Int32)
                    peer = cute.TensorSSA(load_raw(recv + off, cop="cv"), (4,), Int32)
                    a = local.bitcast(self.dtype).to(Float32)
                    b = peer.bitcast(self.dtype).to(Float32)
                    rank0, rank1 = a, b
                    if rank != 0:
                        rank0, rank1 = b, a
                    sums = cute.make_rmem_tensor((128 // self.dtype.width,), Float32)
                    for j in cutlass.range_constexpr(128 // self.dtype.width):
                        put(sums, j, _add_rn(rank0[j], rank1[j]))
                    value = sums.load()
                    if const_expr(self.dtype == Float32):
                        store_raw(out + off, value.bitcast(Int32))
                    else:
                        store_raw(out + off, value.to(self.dtype).bitcast(Int32))
                i += self.ctas * self.threads
        # Finish all reads before the last CTA advances the device sequence.
        cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()
        if tid == 0:
            finish = _ptr(Int64, word_base + 16, 8)
            ticket = cute.arch.atomic_add(finish, Int64(1), sem="acq_rel", scope="gpu")
            if ticket == self.ctas - 1:
                cute.arch.store(finish, Int64(0), sem="release", scope="gpu")
                cute.arch.store(seq_ptr, seq + 1, sem="release", scope="gpu")


def compile_step(dtype, ctas, threads, *, gather=False, options=""):
    import torch
    from cutlass import BFloat16, Float16

    kind = (
        None
        if gather
        else {torch.bfloat16: BFloat16, torch.float16: Float16, torch.float32: Float32}[dtype]
    )
    return cute.compile(
        Step(kind, ctas, threads, gather=gather),
        Int64(0),
        Int64(0),
        cute.runtime.make_fake_tensor(Int64, (4,), stride=(1,), assumed_align=8),
        Int64(0),
        Int64(0),
        Int32(1),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi {options}",
    )
