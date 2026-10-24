# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (
# flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py,
# flashinfer/gdn_kernels/delta_rule_dsl/schedule.py,
# flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py,
# flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""Shared chunk recurrence with compile-time output elision for CP M/N.

CP uses our gate-before-inverse math, identity init / zero V (no upstream T
precompute); segment zero's init is saved to workspace before final pool writes.
The separate fixup is FP32 IEEE and our planner constants are provisional.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.experimental import iket
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cutlass_dsl import T

from . import _watchdog
from ._gates import GateProcessor
from ._inverse import ChunkInverse
from ._mma import WarpMma, round_down
from ._o_store import OutputStore


@dataclass
class WorkDesc:
    # coord
    seq_idx: cutlass.Int32
    private_q_head_idx: cutlass.Int32
    private_v_head_idx: cutlass.Int32
    tok_offset: Any

    # shape
    seq_len: cutlass.Int32

    # update by mainloop
    tile_idx: cutlass.Int32

    def is_valid(self, num_seqs: cutlass.Int32):
        return self.seq_idx >= cutlass.Int32(0) and self.seq_idx < num_seqs

    def q_head_idx(self):
        return self.private_q_head_idx

    @staticmethod
    @cute.jit
    def is_gva(num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        return num_v_heads > num_q_heads

    @cute.jit
    def k_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        k_head_idx = self.private_v_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            k_head_idx = self.private_q_head_idx
        return k_head_idx

    def v_head_idx(self):
        return self.private_v_head_idx

    @cute.jit
    def o_head_idx(self, num_q_heads: cutlass.Int32, num_v_heads: cutlass.Int32):
        o_head_idx = self.private_q_head_idx
        if WorkDesc.is_gva(num_q_heads, num_v_heads):
            o_head_idx = self.private_v_head_idx
        return o_head_idx


# ─── Named-barrier IDs used by the compute kernel ────────────────────────────
# Must not conflict with each other or with pipeline barrier storage.


class NamedBarrier(IntEnum):
    PRODUCER = 11  # fused WG0 gate scan / output rendezvous, 128 threads
    MATH_WG0 = 4  # OrderedMathBarriers: StreamkBarrier0
    MATH_WG1 = 5  # OrderedMathBarriers: StreamkBarrier1
    KK_SYNC = 13  # sync all 128 WG0 threads before collective_inverse


class WarpGroupRole(IntEnum):
    LDST = 0
    MATH_KK = 1
    MATH_QK = 2


class LoadStoreWarpRole(IntEnum):
    LOAD_QKV = 0
    STORE_O = 1
    LOAD_BETA = 2
    LOAD_ALPHA = 3


class MathWarpGroupRole(IntEnum):
    KK = 0
    QK = 1


# ─── Warp-specialized delta-rule kernel ───────────────────────────────────────
# Grid: (num_seqs * num_sab_heads, 1, 1)
# Block: 384 threads → WG0=[0,127], WG1=[128,255], WG2=[256,383]
#
# needs_alpha / needs_beta / needs_init_state are class attributes set in __init__.
# The JIT compiler specialises per instance, so they are compile-time booleans
# inside the kernel without any parameter-passing trickery.


class ChunkKernelSm120:
    @staticmethod
    def get_register_requirements(
        max_threads_per_block: int,
        min_blocks_per_multiprocessor: int,
        num_mma_warp_groups: int,
        threads_per_warp_group: int,
    ) -> tuple[int, int]:
        reg_alloc_granularity = 8
        load_registers = 40 - 2 * reg_alloc_granularity
        total_registers = (
            round_down(
                64 * 1024 // min_blocks_per_multiprocessor,
                max_threads_per_block * reg_alloc_granularity,
            )
            // threads_per_warp_group
        )
        mma_registers = round_down(
            (total_registers - load_registers) // num_mma_warp_groups,
            reg_alloc_granularity,
        )
        return min(248, load_registers), min(248, mma_registers)

    def __init__(self, num_k_heads: int, num_v_heads: int):
        self.fused = False
        self.no_output = False
        self.variant = "full"
        self.pdl = False
        self.watchdog = False
        self.iket = False
        self.hk = num_k_heads
        self.hv = num_v_heads
        self.needs_alpha = True
        self.needs_beta = True
        self.needs_init_state = True
        self.dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        # FlashInfer's (vLLM's) rounding: T is inverted in FP16, then beta-scaled to BF16.
        self.inverse_dtype = cutlass.Float16
        self.BLK_Q = 64
        self.BLK_KV = 64
        self.D = 128
        self.q_stage = 1
        self.k_stage = 2
        self.v_stage = 1
        self.o_stage = 1
        self.alpha_beta_stage = 2

    @cute.jit
    def mark_progress(self, tag, blk):
        if cutlass.const_expr(self.watchdog):
            _watchdog.progress(self.watchdog_words, tag, blk)

    @cute.jit
    def producer_acquire(self, pipe, state, tag: cutlass.Constexpr, blk):
        if cutlass.const_expr(self.watchdog):
            _watchdog.wait(
                pipe.consumer_get_barrier(state),
                state.phase,
                self.watchdog_words,
                tag,
                blk,
                state.index,
            )
        else:
            pipe.producer_acquire(state)

    @cute.jit
    def consumer_wait(self, pipe, state, tag: cutlass.Constexpr, blk):
        if cutlass.const_expr(self.watchdog):
            _watchdog.wait(
                pipe.producer_get_barrier(state),
                state.phase,
                self.watchdog_words,
                tag,
                blk,
                state.index,
            )
        else:
            pipe.consumer_wait(state)

    @cute.jit
    def producer_tail(self, pipe, state, tag: cutlass.Constexpr, blk):
        if cutlass.const_expr(self.watchdog):
            for _ in cutlass.range_constexpr(pipe.num_stages):
                self.producer_acquire(pipe, state, tag, blk)
                state.advance()
        else:
            pipe.producer_tail(state)

    @cute.jit
    def trace_begin(self, name: cutlass.Constexpr):
        if cutlass.const_expr(self.iket):
            iket.range_push(name)

    @cute.jit
    def trace_end(self):
        if cutlass.const_expr(self.iket):
            iket.range_pop()

    def get_next_work(
        self,
        cu_seqlens: cute.Tensor,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ) -> WorkDesc:
        bx, _, _ = cute.arch.block_idx()
        seq_idx = bx // num_sab_heads
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        tok_start = cu_seqlens[seq_idx]
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1] - tok_start)

        return WorkDesc(
            seq_idx=seq_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_start,
            seq_len=seq_len,
            tile_idx=cutlass.Int32(0),
        )

    # ─── Ordered 2-WG math barriers ───────────────────────────────────────────
    # Translates flat::OrderedNamedBarriers<UseReservedNB, NB0, NB1>.
    # wg_idx: MathWarpGroupRole.KK or MathWarpGroupRole.QK.

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Pre-arrive at WG0's barrier so WG0 is unblocked on the first wait."""
        if cutlass.const_expr(self.watchdog):
            if wg_idx == MathWarpGroupRole.QK:
                cute.arch.mbarrier_arrive(self.watchdog_order)
        elif wg_idx == MathWarpGroupRole.QK:
            cute.arch.barrier_arrive(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32, blk):
        """Arrive+wait on this WG's own ordered barrier."""
        if cutlass.const_expr(self.watchdog):
            barrier = self.watchdog_order + wg_idx
            cute.arch.mbarrier_arrive(barrier)
            _watchdog.wait(
                barrier, self.watchdog_phase[0], self.watchdog_words, 40 + wg_idx, blk, wg_idx
            )
            phase = self.watchdog_phase
            phase[0] = phase[0] ^ 1
        elif wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)
        else:
            cute.arch.barrier(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Arrive at the other WG's barrier to unblock it."""
        if cutlass.const_expr(self.watchdog):
            cute.arch.mbarrier_arrive(self.watchdog_order + (1 - wg_idx))
        elif wg_idx == MathWarpGroupRole.KK:
            cute.arch.barrier_arrive(barrier_id=NamedBarrier.MATH_WG1, number_of_threads=256)
        else:
            cute.arch.barrier_arrive(barrier_id=NamedBarrier.MATH_WG0, number_of_threads=256)

    # ─── kk_store_and_inv ─────────────────────────────────────────────────────

    @cute.jit
    def _kk_store_and_inv(
        self,
        tKKrKK: cute.Tensor,  # fp32 KK accumulator (from 128-thread kk_tiled_mma)
        kk_tiled_mma,
        kk_thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element for MMA operand
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) - used when needs_beta
        beta_pipe_idx: cutlass.Int32,
        tKKcMkk: cute.Tensor,  # coordinate mapping for KK fragment
    ):
        """Store tKKrKK → sKK_inv in FP16, invert, then reload as the BF16 operand (beta-scaled)."""
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_store = cute.make_tiled_copy_C(stsm_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_inv = cute.make_fragment_like(tKKrKK, self.inverse_dtype)
        tKKrKK_cv = thr_store.retile(tKKrKK_inv)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrKK_inv[i] = self.inverse_dtype(tKKrKK[i])
        cute.copy(tiled_store, tKKrKK_cv, tKKsKK)

        cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
        ChunkInverse().run(sKK_inv, NamedBarrier.KK_SYNC)

        # Reload FP16 T and store it back as the BF16 NewV operand, beta-scaled.
        cute.arch.barrier(barrier_id=NamedBarrier.KK_SYNC, number_of_threads=128)
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.inverse_dtype,
        )
        tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
        thr_load = tiled_load.get_slice(kk_thread_idx)
        tKKrKK_cpy = cute.make_fragment_like(tKKrKK_inv)
        tKKrKK_cvt = cute.make_fragment_like(tKKrKK_inv, self.dtype)
        tKKrKK_cv2 = thr_load.retile(tKKrKK_cpy)
        cute.copy(tiled_load, thr_load.partition_S(sKK_inv), tKKrKK_cv2)
        tKKcMkk_cv = thr_load.retile(tKKcMkk)

        for i in cutlass.range_constexpr(cute.size(tKKrKK_cpy)):
            if cutlass.const_expr(self.needs_beta):
                _, t = tKKcMkk_cv[i]
                tKKrKK_cvt[i] = self.dtype(
                    cutlass.Float32(tKKrKK_cpy[i]) * cutlass.Float32(sBeta[t, beta_pipe_idx])
                )
            else:
                tKKrKK_cvt[i] = self.dtype(tKKrKK_cpy[i])

        tKKsKK2 = thr_store.partition_D(sKK_opd)
        tKKrKK_cv3 = thr_store.retile(tKKrKK_cvt)
        cute.copy(tiled_store, tKKrKK_cv3, tKKsKK2)

    # ─── kk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        alpha_stage: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, GateProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, t = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cute.math.exp2(
                    cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                    fastmath=True,
                )
        if cutlass.const_expr(self.needs_beta):
            beta_row = sBeta[None, beta_stage]
            for i in cutlass.range_constexpr(cute.size(tKKrKK)):
                s, _ = tKKcMkk[i]
                tKKrKK[i] = tKKrKK[i] * cutlass.Float32(beta_row[s])

    # ─── qk_or_kk_mask ────────────────────────────────────────────────────────

    @cute.jit
    def qk_or_kk_mask(
        self,
        frag: cute.Tensor,
        coord_tensor: cute.Tensor,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        for i in cutlass.range_constexpr(cute.size(frag)):
            s, t = coord_tensor[i]
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and (s < B and t < B)
            if not pred:
                frag[i] = cutlass.Float32(0.0)

    # ─── qk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def qk_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, GateProcessor.CUMSUM_LOG, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                s, t = tQKcMqk[i]
                tQKrQK[i] = (
                    tQKrQK[i]
                    * cute.math.exp2(
                        cutlass.Float32(alpha_cumlog[s]) - cutlass.Float32(alpha_cumlog[t]),
                        fastmath=True,
                    )
                    * scale
                )
        else:
            for i in cutlass.range_constexpr(cute.size(tQKrQK)):
                tQKrQK[i] = tQKrQK[i] * scale

    # ─── qk_store ─────────────────────────────────────────────────────────────

    @cute.jit
    def qk_store(
        self,
        tQKrQK: cute.Tensor,
        sQK: cute.Tensor,
        qk_tiled_mma,
        qk_thread_idx: cutlass.Int32,
    ):
        # qk_epi applies exp(cumlog[s] - cumlog[t]) * scale before this cast.
        # For normalized q/k, nonpositive gates and the causal mask, |QK| is
        # bounded by |scale| (up to input BF16 rounding), safely inside FP16.
        stsm_atom = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), sQK.element_type
        )
        qk_tiled_copy = cute.make_tiled_copy_C(stsm_atom, qk_tiled_mma)
        qk_thr_copy = qk_tiled_copy.get_slice(qk_thread_idx)
        tQKsQK = qk_thr_copy.partition_D(sQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, sQK.element_type)
        tQKrQK_cvt_cv = qk_thr_copy.retile(tQKrQK_cvt)
        for i in cutlass.range_constexpr(cute.size(tQKrQK)):
            tQKrQK_cvt[i] = sQK.element_type(tQKrQK[i])
        cute.copy(qk_tiled_copy, tQKrQK_cvt_cv, tQKsQK)

    # ─── o1_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def o1_epi(
        self,
        tOrO: cute.Tensor,
        tOcO: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        scale: cutlass.Float32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cpscale = sAlpha[None, GateProcessor.CUMPROD_SCALE, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                _, tok_q = tOcO[i]
                tOrO[i] = cutlass.Float32(alpha_cpscale[tok_q]) * tOrO[i]
        else:
            for i in cutlass.range_constexpr(cute.size(tOrO)):
                tOrO[i] = scale * tOrO[i]

    # ─── sk_epi ───────────────────────────────────────────────────────────────

    @cute.jit
    def sk_epi(
        self,
        tSKrSK: cute.Tensor,
        tSKcSK: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cp = sAlpha[None, GateProcessor.CUMPROD, alpha_stage]
            for i in cutlass.range_constexpr(cute.size(tSKrSK)):
                _, tok_kv = tSKcSK[i]
                tSKrSK[i] = tSKrSK[i] * cutlass.Float32(alpha_cp[tok_kv])

    # ─── sk_load_v ────────────────────────────────────────────────────────────

    @cute.jit
    def sk_load_v(
        self,
        tSKrSK: cute.Tensor,
        sV_DS: cute.Tensor,
        sk_tiled_copy_C,
        sk_thr_copy_C,
        v_stage: cutlass.Int32,
    ) -> cute.Tensor:
        tSKrV = cute.make_fragment_like(tSKrSK, self.dtype)
        tSKrV_cv = sk_thr_copy_C.retile(tSKrV)
        tSKsV = sk_thr_copy_C.partition_S(sV_DS)
        cute.copy(sk_tiled_copy_C, tSKsV[None, None, None, v_stage], tSKrV_cv)
        return tSKrV

    # ─── kv_decay_v ───────────────────────────────────────────────────────────

    @cute.jit
    def kv_decay_v(
        self,
        tKVrV: cute.Tensor,
        tKVcV: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
    ):
        if cutlass.const_expr(self.needs_alpha):
            alpha_cumlog = sAlpha[None, GateProcessor.CUMSUM_LOG, alpha_stage]
            block_log = cutlass.Float32(alpha_cumlog[B - cutlass.Int32(1)])
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                coeff = cute.math.exp2(
                    block_log - cutlass.Float32(alpha_cumlog[tok]), fastmath=True
                )
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        coeff = cutlass.Float32(0.0)
                tKVrV[i] = tKVrV.element_type(cutlass.Float32(tKVrV[i]) * coeff)
        else:
            for i in cutlass.range_constexpr(cute.size(tKVrV)):
                _, tok = tKVcV[i]
                if cutlass.const_expr(is_final_block):
                    if tok >= B:
                        tKVrV[i] = tKVrV.element_type(0.0)

    # ─── o_store ──────────────────────────────────────────────────────────────

    @cute.jit
    def o_store(
        self,
        tOrO: cute.Tensor,
        tOsO: cute.Tensor,
        o_tiled_copy_r2s,
        o_thr_copy_r2s,
    ):
        tOrO_f16 = cute.make_fragment_like(tOrO, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tOrO)):
            tOrO_f16[i] = self.dtype(tOrO[i])
        tOrO_cv = o_thr_copy_r2s.retile(tOrO_f16)
        cute.arch.fence_view_async_shared()
        cute.copy(o_tiled_copy_r2s, tOrO_cv, tOsO)
        cute.arch.fence_view_async_shared()

    # ─── TMA load helpers ────────────────────────────────────────────────────

    @cute.jit
    def load_qkv_tma(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        blk: cutlass.Int32,
        tok_start,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)

        sK = sK_DS[None, None, k_producer_state.index]
        mK = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_k[None, None, k_head_idx],
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        k_pipeline.producer_acquire(k_producer_state)
        cute.copy(
            tma_atom_k,
            tKgK,
            tKsK,
            tma_bar_ptr=k_pipeline.producer_get_barrier(k_producer_state),
        )
        k_pipeline.producer_commit(k_producer_state)
        k_producer_state.advance()

        sQ = sQ_SD[None, None, q_producer_state.index]
        mQ = cute.domain_offset(
            (blk_tok, cutlass.Int32(0)),
            tma_tensor_q[None, None, q_head_idx],
        )
        gQ = cute.zipped_divide(mQ, (self.BLK_Q, self.D))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(gQ, 0, 2),
        )
        q_pipeline.producer_acquire(q_producer_state)
        cute.copy(
            tma_atom_q,
            tQgQ,
            tQsQ,
            tma_bar_ptr=q_pipeline.producer_get_barrier(q_producer_state),
        )
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        sV = sV_DS[None, None, v_producer_state.index]
        mV = cute.domain_offset(
            (cutlass.Int32(0), blk_tok),
            tma_tensor_v[None, None, v_head_idx],
        )
        gV = cute.zipped_divide(mV, (self.D, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        tVsV, tVgV = cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gV, 0, 2),
        )
        v_pipeline.producer_acquire(v_producer_state)
        cute.copy(
            tma_atom_v,
            tVgV,
            tVsV,
            tma_bar_ptr=v_pipeline.producer_get_barrier(v_producer_state),
        )
        v_pipeline.producer_commit(v_producer_state)
        v_producer_state.advance()
        return q_producer_state, k_producer_state, v_producer_state

    # ─── load_alpha ───────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_alpha (scalar load).
    # Caller must sync before calling GateProcessor on the loaded data.

    @cute.jit
    def load_alpha(
        self,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        alpha_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sAlpha_k = sAlpha[None, None, alpha_stage]
        num_iters = self.BLK_Q // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sAlpha_k[row, GateProcessor.CUMSUM_LOG] = g_alpha[
                    tok * num_sab_heads + sab_head_idx
                ] * cutlass.Float32(1.4426950408889634)
            else:
                sAlpha_k[row, GateProcessor.CUMSUM_LOG] = cutlass.Float32(0.0)

    # ─── load_beta ────────────────────────────────────────────────────────────
    # Translates FlatMainloopTmaWarpSpecializedDeltaRule::load_beta.

    @cute.jit
    def load_beta(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        blk_tok: cutlass.Int32,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        beta_stage: cutlass.Int32,
    ):
        lane_id = cute.arch.lane_idx()
        sBeta_k = sBeta[None, beta_stage]
        num_iters = self.BLK_KV // 32
        for i in cutlass.range_constexpr(num_iters):
            row = cutlass.Int32(i * 32) + lane_id
            tok = blk_tok + row
            if tok < tok_end:
                sBeta_k[row] = g_beta[tok * num_sab_heads + sab_head_idx]
            else:
                sBeta_k[row] = cutlass.Float32(0.0)

    # ─── kv_load / kv_store ───────────────────────────────────────────────────

    @cute.jit
    def kv_load(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            tKVrKV[i] = gKV[k_idx, v_idx].to(self.acc_dtype)

    @cute.jit
    def kv_store(
        self,
        tKVrKV: cute.Tensor,
        gKV: cute.Tensor,
        kv_thr_mma,
    ):
        c_kv = cute.make_identity_tensor((self.D, self.D))
        tKVcKV = kv_thr_mma.partition_C(c_kv)
        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            v_idx, k_idx = tKVcKV[i]
            gKV[k_idx, v_idx] = tKVrKV[i].to(gKV.element_type)

    # ─── compute_loop_body ───────────────────────────────────────────────────
    # Translates the C++ compute_loop_body lambda captured inside compute().
    # Called by Math WGs (tidx >= 128) for one block iteration.

    @cute.jit
    def compute_loop_body(
        self,
        blk: cutlass.Int32,
        # Smem tensors (staged; caller indexes the active stage)
        sQ_SD: cute.Tensor,  # (BlkQ, D, StagesQ)  – row-major atom, swizzled
        sK_SD: cute.Tensor,  # (BlkKV, D, StagesK) – same atom
        sK_DS: cute.Tensor,  # (D, BlkKV, StagesK) – K transposed
        sV_DS: cute.Tensor,  # (D, BlkKV, StagesV) – V transposed
        sQK: cute.Tensor,  # (BlkQ, BlkKV)
        sKK_inv: cute.Tensor,  # (BlkKV, BlkKV)
        sKK_opd: cute.Tensor,  # sKK_inv storage recast as Element
        sO: cute.Tensor,  # O output smem (staged)
        sAlpha: cute.Tensor,  # (BlkQ, GateProcessor.NUM_CHANNELS, StagesAlpha) or zero-shaped
        sBeta: cute.Tensor,  # (BlkKV, StagesBeta) or zero-shaped
        kv_tiled_mma,
        # Mainloop pipelines and active read states
        q_pipeline,
        q_consumer_state,
        k_pipeline,
        k_consumer_state,
        v_pipeline,
        v_consumer_state,
        o_pipeline,
        o_producer_state,
        alpha_pipeline,
        alpha_consumer_state,
        beta_pipeline,
        beta_consumer_state,
        # Compile-time flags
        is_first_block: bool,
        is_final_block: bool,
        # Valid token count for masking on final block
        B: cutlass.Int32,
        # Running KV state (D×D fp32, in registers across all blocks)
        tKVrKV: cute.Tensor,
        # Scale factor
        scale: cutlass.Float32,
        # WG role: MathWarpGroupRole.KK or MathWarpGroupRole.QK.
        wg_idx: cutlass.Int32,
        initial_context=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx - cutlass.Int32(128)  # relative to compute threads
        kk_thread_idx = thread_idx % cutlass.Int32(128)
        qk_thread_idx = thread_idx % cutlass.Int32(128)

        # ── TiledMMAs ─────────────────────────────────────────────────────────
        blk_q = cute.size(sQ_SD, mode=[0])
        blk_kv = cute.size(sK_SD, mode=[0])
        d = cute.size(sQ_SD, mode=[1])
        tile_shape_qk = (blk_q, blk_kv, d)
        tile_shape_kk = tile_shape_qk
        tile_shape_o1 = (d, blk_q, d)
        tile_shape_o2 = (d, blk_q, blk_kv)
        tile_shape_sk = (d, blk_kv, d)
        tile_shape_newv = (d, blk_kv, blk_kv)
        k_stage = k_consumer_state.index
        q_stage = q_consumer_state.index
        v_stage = v_consumer_state.index
        o_stage = cutlass.Int32(0)
        alpha_stage = alpha_consumer_state.index
        beta_stage = beta_consumer_state.index

        mma_atom_4w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))
        mma_atom_8w = warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16))

        # QK/KK: 4 warps × 16M = 64M  (1 warpgroup, 128 threads)
        qk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_qk
        )
        kk_tiled_mma = cute.make_tiled_mma(
            mma_atom_4w, cute.make_layout((4, 1, 1)), permutation_mnk=tile_shape_kk
        )

        # O1/O2/SK/NewV: 8 warps × 16M = 128M (both warpgroups, 256 threads)
        o1_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o1
        )
        o2_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_o2
        )
        sk_tiled_mma = cute.make_tiled_mma(
            mma_atom_8w, cute.make_layout((8, 1, 1)), permutation_mnk=tile_shape_sk
        )
        newv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=tile_shape_newv,
        )

        # ── Thread slices ─────────────────────────────────────────────────────
        qk_thr_mma = qk_tiled_mma.get_slice(qk_thread_idx)
        kk_thr_mma = kk_tiled_mma.get_slice(kk_thread_idx)
        sk_thr_mma = sk_tiled_mma.get_slice(thread_idx)
        newv_thr_mma = newv_tiled_mma.get_slice(thread_idx)
        o1_thr_mma = o1_tiled_mma.get_slice(thread_idx)
        o2_thr_mma = o2_tiled_mma.get_slice(thread_idx)
        kv_thr_mma = kv_tiled_mma.get_slice(thread_idx)

        # ── Copy atoms ────────────────────────────────────────────────────────
        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        ldsm_t4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )

        # ── Active smem slices (extract 2D from staged tensors) ───────────────
        sQ_k = sQ_SD[None, None, q_stage]  # (BlkQ, D)
        sK_SD_k = sK_SD[None, None, k_stage]  # (BlkKV, D)
        sK_DS_k = sK_DS[None, None, k_stage]  # (D, BlkKV)

        # ── QK copies ─────────────────────────────────────────────────────────
        qk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, qk_tiled_mma)
        qk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, qk_tiled_mma)
        qk_thr_copy_A = qk_tiled_copy_A.get_slice(qk_thread_idx)
        qk_thr_copy_B = qk_tiled_copy_B.get_slice(qk_thread_idx)

        tQKrQ = qk_thr_mma.make_fragment_A(qk_thr_mma.partition_A(sQ_k))
        tQKrQ_cv = qk_thr_copy_A.retile(tQKrQ)
        tQKsQ = qk_thr_copy_A.partition_S(sQ_SD)
        tQKrK = qk_thr_mma.make_fragment_B(qk_thr_mma.partition_B(sK_SD_k))
        tQKrK_cv = qk_thr_copy_B.retile(tQKrK)
        tQKsK = qk_thr_copy_B.partition_S(sK_SD)

        # ── KK copies (same atom as QK) ───────────────────────────────────────
        kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_n4, kk_tiled_mma)
        kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, kk_tiled_mma)
        kk_thr_copy_A = kk_tiled_copy_A.get_slice(kk_thread_idx)
        kk_thr_copy_B = kk_tiled_copy_B.get_slice(kk_thread_idx)

        tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD_k))
        tKKrA_cv = kk_thr_copy_A.retile(tKKrA)
        tKKsA = kk_thr_copy_A.partition_S(sK_SD)
        tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD_k))
        tKKrB_cv = kk_thr_copy_B.retile(tKKrB)
        tKKsB = kk_thr_copy_B.partition_S(sK_SD)

        # ── SK copies ─────────────────────────────────────────────────────────
        # SK B: K loaded from sK_SD (row-major BlkKV×D) with LDSM_N — matches C++ SK B-operand
        # SK C: V loaded from sV_DS (col-major D×BlkKV) with LDSM_T
        sk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, sk_tiled_mma)
        sk_tiled_copy_C = cute.make_tiled_copy_C(ldsm_t4, sk_tiled_mma)
        sk_thr_copy_B = sk_tiled_copy_B.get_slice(thread_idx)
        sk_thr_copy_C = sk_tiled_copy_C.get_slice(thread_idx)

        # Work around DSL make_fragment_B not accepting partition_shape_B output directly.
        tSKrK = cute.make_rmem_tensor(
            sk_thr_mma.partition_shape_B(cute.slice_(tile_shape_sk, (0, None, None))),
            self.dtype,
        )
        tSKrK_cv = sk_thr_copy_B.retile(tSKrK)
        tSKsK = sk_thr_copy_B.partition_S(sK_SD)

        # ── NewV copies ───────────────────────────────────────────────────────
        newv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, newv_tiled_mma)
        newv_thr_copy_B = newv_tiled_copy_B.get_slice(thread_idx)
        tNewVrB = newv_thr_mma.make_fragment_B(newv_thr_mma.partition_B(sKK_opd))
        tNewVrB_cv = newv_thr_copy_B.retile(tNewVrB)
        tNewVsB = newv_thr_copy_B.partition_S(sKK_opd)

        # ── KV copies ─────────────────────────────────────────────────────────
        kv_tiled_copy_B = cute.make_tiled_copy_B(ldsm_t4, kv_tiled_mma)
        kv_thr_copy_B = kv_tiled_copy_B.get_slice(thread_idx)
        tKVrK = kv_thr_mma.make_fragment_B(kv_thr_mma.partition_B(sK_DS_k))
        tKVrK_cv = kv_thr_copy_B.retile(tKVrK)
        tKVsK = kv_thr_copy_B.partition_S(sK_DS)

        # ── O1/O2 copies ──────────────────────────────────────────────────────
        o1_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o1_tiled_mma)
        o2_tiled_copy_B = cute.make_tiled_copy_B(ldsm_n4, o2_tiled_mma)
        o1_thr_copy_B = o1_tiled_copy_B.get_slice(thread_idx)
        o2_thr_copy_B = o2_tiled_copy_B.get_slice(thread_idx)

        # Direct partition_B(sQ_k) preserves the swizzled Q layout here and produces
        # a non-C++ B fragment shape; derive the fragment from TileShapeO1 instead.
        tOrQ = cute.make_rmem_tensor(
            o1_thr_mma.partition_shape_B(cute.slice_(tile_shape_o1, (0, None, None))),
            self.dtype,
        )
        tOrQ_cv = o1_thr_copy_B.retile(tOrQ)
        tOsQ = o1_thr_copy_B.partition_S(sQ_SD)
        tOrQK = o2_thr_mma.make_fragment_B(o2_thr_mma.partition_B(sQK))
        tOrQK_cv = o2_thr_copy_B.retile(tOrQK)
        tOsQK = o2_thr_copy_B.partition_S(sQK)

        # ── O store (R→S STSM) ────────────────────────────────────────────────
        o_stsm = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        o_tiled_copy_r2s = cute.make_tiled_copy_C(o_stsm, o1_tiled_mma)
        o_thr_copy_r2s = o_tiled_copy_r2s.get_slice(thread_idx)
        tOsO = o_thr_copy_r2s.partition_D(sO)

        # ── Coordinate tensors for masking / alpha/beta indexing ──────────────
        cMqk = cute.make_identity_tensor((blk_q, blk_kv))
        tQKcMqk = qk_thr_mma.partition_C(cMqk)
        cMkk = cMqk  # same shape (BlkKV == BlkQ == 64)
        tKKcMkk = kk_thr_mma.partition_C(cMkk)
        cO = cute.make_identity_tensor((d, blk_q))
        tOcO = o1_thr_mma.partition_C(cO)
        cSK = cute.make_identity_tensor((d, blk_kv))
        tSKcSK = sk_thr_mma.partition_C(cSK)
        cV = cute.make_identity_tensor((d, blk_kv))
        tKVcV = kv_thr_mma.partition_A(cV)

        # ── KK GEMM (WG0 only) ────────────────────────────────────────────────
        self.trace_begin("math.wait_k_gates")
        self.consumer_wait(k_pipeline, k_consumer_state, 21, blk)
        if cutlass.const_expr(self.needs_alpha):
            self.consumer_wait(alpha_pipeline, alpha_consumer_state, 24, blk)
            cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.needs_beta):
            self.consumer_wait(beta_pipeline, beta_consumer_state, 25, blk)
            cute.arch.fence_view_async_shared()
        self.trace_end()
        # Match the C++ reject-non-role-first shape; ptxas keeps BRA.U around
        # the role body instead of predicating the HMMA/LDSM/STSM sequence.
        if wg_idx != MathWarpGroupRole.KK:
            cute.arch.sync_warp()
        else:
            self.trace_begin("math.KK_inverse")
            cute.copy(kk_tiled_copy_A, tKKsA[None, None, None, k_stage], tKKrA_cv)
            cute.copy(kk_tiled_copy_B, tKKsB[None, None, None, k_stage], tKKrB_cv)
            tKKrKK = cute.make_rmem_tensor(
                kk_thr_mma.partition_shape_C((blk_kv, blk_kv)), self.acc_dtype
            )
            tKKrKK.fill(self.acc_dtype(0.0))
            cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)
            self.kk_epi(tKKrKK, tKKcMkk, sAlpha, sBeta, alpha_stage, beta_stage)
            self.qk_or_kk_mask(tKKrKK, tKKcMkk, is_final_block, B)
            self._kk_store_and_inv(
                tKKrKK,
                kk_tiled_mma,
                kk_thread_idx,
                sKK_inv,
                sKK_opd,
                sBeta,
                beta_stage,
                tKKcMkk,
            )
            self.trace_end()
        if cutlass.const_expr(self.needs_beta):
            beta_pipeline.consumer_release(beta_consumer_state)
            beta_consumer_state.advance()

        if cutlass.const_expr(initial_context is not None and self.no_output):
            self.load_state(tKVrKV, kv_thr_mma, *initial_context)

        if cutlass.const_expr(not self.no_output):
            self.trace_begin("math.wait_q")
            self.consumer_wait(q_pipeline, q_consumer_state, 20, blk)
            self.trace_end()
            if wg_idx != MathWarpGroupRole.QK:
                cute.arch.sync_warp()
            else:
                self.trace_begin("math.QK")
                cute.copy(qk_tiled_copy_A, tQKsQ[None, None, None, q_stage], tQKrQ_cv)
                cute.copy(qk_tiled_copy_B, tQKsK[None, None, None, k_stage], tQKrK_cv)
                tQKrQK = cute.make_rmem_tensor(
                    qk_thr_mma.partition_shape_C((blk_q, blk_kv)), self.acc_dtype
                )
                tQKrQK.fill(self.acc_dtype(0.0))
                cute.gemm(qk_tiled_mma, tQKrQK, tQKrQ, tQKrK, tQKrQK)
                self.qk_epi(tQKrQK, tQKcMqk, sAlpha, alpha_stage, scale)
                self.qk_or_kk_mask(tQKrQK, tQKcMqk, is_final_block, B)
                self.qk_store(tQKrQK, sQK, qk_tiled_mma, qk_thread_idx)
                self.trace_end()

            # Load CP state after first KK/QK, shortening its live range.
            if cutlass.const_expr(initial_context is not None):
                self.load_state(tKVrKV, kv_thr_mma, *initial_context)

            # ── O1: KV_state @ Q (both WGs, skip on first block) ─────────────────
            self.trace_begin("math.O1")
            tOrO = cute.make_rmem_tensor(o1_thr_mma.partition_shape_C((d, blk_q)), self.acc_dtype)
            tOrO.fill(self.acc_dtype(0.0))
            if cutlass.const_expr(not is_first_block):
                cute.copy(o1_tiled_copy_B, tOsQ[None, None, None, q_stage], tOrQ_cv)
                tOrKV = WarpMma.make_acc_into_op(tKVrKV, o1_tiled_mma, self.dtype)
                cute.gemm(o1_tiled_mma, tOrO, tOrKV, tOrQ, tOrO)
                self.o1_epi(tOrO, tOcO, sAlpha, alpha_stage, scale)
            q_pipeline.consumer_release(q_consumer_state)
            q_consumer_state.advance()

            self.trace_end()
        self.trace_begin("math.SK_NewV")
        # ── SK: KV_state @ K^T (result negated below via V - SK) ─────────────
        tSKrSK = cute.make_rmem_tensor(sk_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype)
        tSKrSK.fill(self.acc_dtype(0.0))
        if cutlass.const_expr(not is_first_block):
            tSKrS = WarpMma.make_acc_into_op(tKVrKV, sk_tiled_mma, self.dtype)
            cute.copy(sk_tiled_copy_B, tSKsK[None, None, None, k_stage], tSKrK_cv)
            cute.gemm(sk_tiled_mma, tSKrSK, tSKrS, tSKrK, tSKrSK)

        # ── Load V from smem ──────────────────────────────────────────────────
        self.trace_begin("math.wait_v")
        self.consumer_wait(v_pipeline, v_consumer_state, 22, blk)
        self.trace_end()
        tSKrV = self.sk_load_v(tSKrSK, sV_DS, sk_tiled_copy_C, sk_thr_copy_C, v_stage)

        # sk_epi + V - SK  (SK=0 on first block, so V - SK = V)
        if cutlass.const_expr(not is_first_block):
            self.sk_epi(tSKrSK, tSKcSK, sAlpha, alpha_stage)
            for i in cutlass.range_constexpr(cute.size(tSKrV)):
                tSKrV[i] = tSKrV[i] - self.dtype(tSKrSK[i])

        # ── NewV = (V - SK) @ T^T  (ordered: WG0 first) ──────────────────────
        tNewVrA = WarpMma.make_acc_into_op(tSKrV, newv_tiled_mma, self.dtype)
        tNewVrC = cute.make_rmem_tensor(newv_thr_mma.partition_shape_C((d, blk_kv)), self.acc_dtype)
        self._math_order_wait(wg_idx, blk)
        cute.copy(newv_tiled_copy_B, tNewVsB, tNewVrB_cv)
        tNewVrC.fill(self.acc_dtype(0.0))
        cute.gemm(newv_tiled_mma, tNewVrC, tNewVrA, tNewVrB, tNewVrC)
        self._math_order_notify(wg_idx)
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()

        self.trace_end()
        if cutlass.const_expr(not self.no_output):
            self.trace_begin("math.O2_norm")
            # ── O2 = O1 + NewV @ QK (ordered: WG0 first) ────────────────────────
            tOrNewV = WarpMma.make_acc_into_op(tNewVrC, o2_tiled_mma, self.dtype)
            self._math_order_wait(wg_idx, blk)
            cute.copy(o2_tiled_copy_B, tOsQK, tOrQK_cv)
            cute.gemm(o2_tiled_mma, tOrO, tOrNewV, tOrQK, tOrO)
            self._math_order_notify(wg_idx)

            # ── O store to smem ───────────────────────────────────────────────────
            self.producer_acquire(o_pipeline, o_producer_state, 13, blk)
            self.o_store(
                tOrO,
                tOsO[None, None, None, o_stage],
                o_tiled_copy_r2s,
                o_thr_copy_r2s,
            )
            o_pipeline.producer_commit(o_producer_state)
            o_producer_state.advance()

            self.trace_end()
        if cutlass.const_expr(self.no_output):
            # O2 normally supplies this second ordered phase. Retain the
            # rendezvous without O2: next chunk's KK must not overwrite sKK
            # until BOTH WGs have loaded the current inverse for NewV.
            self._math_order_wait(wg_idx, blk)
            self._math_order_notify(wg_idx)
            tOrNewV = WarpMma.make_acc_into_op(tNewVrC, kv_tiled_mma, self.dtype)
        self.trace_begin("math.KV_update")
        # ── KV state update ───────────────────────────────────────────────────
        block_coeff = cutlass.Float32(1.0)
        if cutlass.const_expr(self.needs_alpha):
            block_coeff = cutlass.Float32(
                sAlpha[B - cutlass.Int32(1), GateProcessor.CUMPROD, alpha_stage]
            )

        for i in cutlass.range(cute.size(tKVrKV), unroll_full=True):
            tKVrKV[i] = block_coeff * tKVrKV[i]

        self.kv_decay_v(tOrNewV, tKVcV, sAlpha, alpha_stage, is_final_block, B)

        # KV += NewV @ K
        cute.copy(kv_tiled_copy_B, tKVsK[None, None, None, k_stage], tKVrK_cv)
        cute.gemm(kv_tiled_mma, tKVrKV, tOrNewV, tKVrK, tKVrKV)
        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.consumer_release(alpha_consumer_state)
            alpha_consumer_state.advance()
        self.trace_end()
        return (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        )

    # ─── Warp role entry points ──────────────────────────────────────────────
    # The current DSL bridge still uses CTA-wide sync epochs, but each role owns
    # its own loop, matching the C++ warp-specialized dispatch shape.

    @cute.jit
    def run_load_qkv_role(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        num_blocks: cutlass.Int32,
        tok_start,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
    ):
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
            ) = self.load_qkv_tma(
                sQ_SD,
                sK_DS,
                sV_DS,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_v,
                tma_tensor_v,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                blk,
                tok_start,
                q_head_idx,
                k_head_idx,
                v_head_idx,
            )

    @cute.jit
    def run_load_alpha_role(
        self,
        sAlpha: cute.Tensor,
        g_alpha: cute.Tensor,
        alpha_pipeline,
        scale: cutlass.Float32,
        num_blocks: cutlass.Int32,
        tok_start,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
            if cutlass.const_expr(self.needs_alpha):
                alpha_pipeline.producer_acquire(alpha_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_alpha(
                    sAlpha,
                    g_alpha,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                GateProcessor().run(sAlpha[None, None, alpha_producer_state.index], scale)
                cute.arch.fence_view_async_shared()
                alpha_pipeline.producer_commit(alpha_producer_state)
                alpha_producer_state.advance()

    @cute.jit
    def run_load_beta_role(
        self,
        sBeta: cute.Tensor,
        g_beta: cute.Tensor,
        beta_pipeline,
        num_blocks: cutlass.Int32,
        tok_start,
        tok_end,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
    ):
        beta_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            blk_tok = tok_start + blk * cutlass.Int32(self.BLK_KV)
            if cutlass.const_expr(self.needs_beta):
                beta_pipeline.producer_acquire(beta_producer_state)
                cute.arch.fence_view_async_shared()
                self.load_beta(
                    sBeta,
                    g_beta,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    beta_producer_state.index,
                )
                cute.arch.fence_view_async_shared()
                beta_pipeline.producer_commit(beta_producer_state)
                beta_producer_state.advance()

    @cute.jit
    def run_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        o_pipeline,
        alpha_pipeline,
        beta_pipeline,
        g_state: cute.Tensor,
        g_initial: cute.Tensor,
        g_state_indices: cute.Tensor,
        work_desc: WorkDesc,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        o_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.o_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((8, 1, 1)),
            permutation_mnk=(self.D, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        initial_context = None
        if cutlass.const_expr(self.variant == "full"):
            self.load_state(tKVrKV, kv_thr_mma, g_state, g_initial, g_state_indices, work_desc)
        else:
            initial_context = (g_state, g_initial, g_state_indices, work_desc)

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        (
            q_consumer_state,
            k_consumer_state,
            v_consumer_state,
            o_producer_state,
            alpha_consumer_state,
            beta_consumer_state,
        ) = self.compute_loop_body(
            cutlass.Int32(0),
            sQ_SD,
            sK_SD,
            sK_DS,
            sV_DS,
            sQK,
            sKK_inv,
            sKK_opd,
            sO,
            sAlpha,
            sBeta,
            kv_tiled_mma,
            q_pipeline,
            q_consumer_state,
            k_pipeline,
            k_consumer_state,
            v_pipeline,
            v_consumer_state,
            o_pipeline,
            o_producer_state,
            alpha_pipeline,
            alpha_consumer_state,
            beta_pipeline,
            beta_consumer_state,
            False,
            True,
            first_B,
            tKVrKV,
            scale,
            wg_idx,
            initial_context,
        )

        for _blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                _blk,
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                wg_idx,
            )

        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            if cutlass.const_expr(self.fused):
                self.maybe_store_capture(
                    tKVrKV, kv_thr_mma, g_state, work_desc, last_blk * cutlass.Int32(self.BLK_KV)
                )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                o_producer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                last_blk,
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                o_pipeline,
                o_producer_state,
                alpha_pipeline,
                alpha_consumer_state,
                beta_pipeline,
                beta_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
            )
        self.store_state(tKVrKV, kv_thr_mma, g_state, g_state_indices, work_desc)
        if cutlass.const_expr(self.fused):
            self.maybe_store_capture(tKVrKV, kv_thr_mma, g_state, work_desc, work_desc.seq_len)

    @cute.jit
    def pool_view(self, state, slot, head):
        # kv_load/store expect [K,V], while the pool is physically [V,K].
        view = cute.make_tensor(state.iterator, cute.select(state.layout, mode=[3, 2, 1, 0]))
        # Int64 slot: slot * Hv * 16384 passes 2^31 from slot 2731 at Hv = 48.
        return view[None, None, head, cutlass.Int64(slot)]

    @cute.jit
    def load_state(self, acc, mma, state, initial, slots, work):
        slot = slots[work.seq_idx]
        if slot >= 0 and initial[work.seq_idx] != 0:
            self.kv_load(acc, self.pool_view(state, slot, work.v_head_idx()), mma)

    @cute.jit
    def store_state(self, acc, mma, state, slots, work):
        slot = slots[work.seq_idx]
        if slot >= 0:
            self.kv_store(acc, self.pool_view(state, slot, work.v_head_idx()), mma)

    # ─── Kernel entry point ───────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        g_o: cute.Tensor,
        g_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_initial: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        g_tensormaps: cute.Tensor,
        stream,
    ):
        num_q_heads = self.hk
        num_k_heads = self.hk
        num_v_heads = self.hv
        num_sab_heads = self.hv
        num_seqs = g_state_indices.shape[0]
        grid_x = num_seqs * self.hv
        if cutlass.const_expr(self.variant == "mn"):
            grid_x = (2 * g_alpha[8] - 3) * self.hv
        elif cutlass.const_expr(self.variant == "seg"):
            grid_x = g_alpha[8] * self.hv
        if cutlass.const_expr(not self.fused):
            g_q = cute.make_tensor(g_q.iterator, cute.select(g_q.layout, [0, 2, 1]))
            g_k = cute.make_tensor(g_k.iterator, cute.select(g_k.layout, [2, 0, 1]))
            g_v = cute.make_tensor(g_v.iterator, cute.select(g_v.layout, [2, 0, 1]))
            g_alpha = cute.make_tensor(g_alpha.iterator, cute.make_layout(cute.size(g_alpha)))
            g_beta = cute.make_tensor(g_beta.iterator, cute.make_layout(cute.size(g_beta)))
        g_o = cute.make_tensor(g_o.iterator, cute.select(g_o.layout, [2, 0, 1]))
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        q_smem_layout = cute.slice_(q_storage_layout, (None, None, 0))
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_ds = cute.select(k_storage_layout_sd, [1, 0, 2])
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_ds = cute.select(v_storage_layout_sd, [1, 0, 2])
        k_smem_layout = cute.slice_(k_storage_layout_ds, (None, None, 0))
        v_smem_layout = cute.slice_(v_storage_layout_ds, (None, None, 0))
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        o_smem_layout = cute.slice_(o_storage_layout, (None, None, 0))

        if cutlass.const_expr(self.fused):
            # One descriptor over raw projected [T, P], shared by Q/K/V.
            # All three stages have the same row-major K_SW128 tile layout.
            tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                g_alpha[0],
                q_smem_layout,
                (self.BLK_Q, self.D),
            )
            tma_atom_k, tma_tensor_k = tma_atom_q, tma_tensor_q
            tma_atom_v, tma_tensor_v = tma_atom_q, tma_tensor_q
        else:
            tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
                tma_load_op, g_q, q_smem_layout, (self.BLK_Q, self.D)
            )
            tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
                tma_load_op, g_k, k_smem_layout, (self.D, self.BLK_KV)
            )
            tma_atom_v, tma_tensor_v = cpasync.make_tiled_tma_atom(
                tma_load_op, g_v, v_smem_layout, (self.D, self.BLK_KV)
            )

        if cutlass.const_expr(self.no_output):
            tma_atom_o, tma_tensor_o = None, g_o
        else:
            tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_o, tma_tensor_o = cpasync.make_tiled_tma_atom(
                tma_store_op, g_o, o_smem_layout, (self.D, self.BLK_Q)
            )

        dtype_bytes = self.dtype.width // 8
        self.tma_load_q_bytes = cute.size(q_smem_layout) * dtype_bytes
        self.tma_load_k_bytes = cute.size(k_smem_layout) * dtype_bytes
        self.tma_load_v_bytes = cute.size(v_smem_layout) * dtype_bytes

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, GateProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        beta_storage_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))

        @cute.struct
        class SharedStorage:
            watchdog_words: cute.struct.MemRange[cutlass.Int32, 12 if self.watchdog else 0]
            watchdog_barriers: cute.struct.MemRange[cutlass.Int64, 3 if self.watchdog else 0]
            raw_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4 if self.fused else 0]
            q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_stage * 2]
            v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.v_stage * 2]
            o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.o_stage * 2]
            alpha_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.alpha_beta_stage * 2]
            beta_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.alpha_beta_stage * 2]

            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.inverse_dtype, cute.cosize(kk_storage_layout)],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_storage_layout)],
                16,
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_storage_layout)],
                16,
            ]

        assert SharedStorage.size_in_bytes() <= 101376, "sm120 shared-memory budget exceeded"
        self.shared_storage = SharedStorage

        self.kernel(
            g_alpha,
            g_beta,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            g_state,
            g_initial,
            g_state_indices,
            g_tensormaps,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(384, 1, 1),
            max_number_threads=(384, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.pdl and self.variant == "seg",
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha,
        g_beta,
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_v,
        tma_tensor_v,
        tma_atom_o,
        tma_tensor_o,
        g_state,
        g_initial,
        g_state_indices,
        g_tensormaps,
        cu_seqlens,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        NUM_LOAD_WARP_GROUPS = 1
        NUM_MMA_WARP_GROUPS = 2
        THREADS_PER_WARP_GROUP = 128
        WARPS_PER_WARP_GROUP = 4
        MIN_BLOCKS_PER_MP = 1
        MAX_THREADS_PER_BLOCK = (
            NUM_LOAD_WARP_GROUPS + NUM_MMA_WARP_GROUPS
        ) * THREADS_PER_WARP_GROUP
        load_registers, mma_registers = self.get_register_requirements(
            MAX_THREADS_PER_BLOCK,
            MIN_BLOCKS_PER_MP,
            NUM_MMA_WARP_GROUPS,
            THREADS_PER_WARP_GROUP,
        )

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // cutlass.Int32(THREADS_PER_WARP_GROUP))
        ldst_warp_role = cute.arch.make_warp_uniform(warp_idx % cutlass.Int32(WARPS_PER_WARP_GROUP))

        if warp_idx == LoadStoreWarpRole.LOAD_QKV:
            cpasync.prefetch_descriptor(tma_atom_q)
            if cutlass.const_expr(not self.fused):
                cpasync.prefetch_descriptor(tma_atom_k)
                cpasync.prefetch_descriptor(tma_atom_v)
            if cutlass.const_expr(not self.no_output):
                cpasync.prefetch_descriptor(tma_atom_o)

        if cutlass.const_expr(self.variant != "full"):
            self.cp_workspace, self.seg_len, self.num_segments = g_alpha[6:9]
        if cutlass.const_expr(self.fused):
            self.capture_rows, self.capture = g_alpha[9:11]
        work_desc = self.get_next_work(
            cu_seqlens,
            num_q_heads,
            num_v_heads,
            num_sab_heads,
        )
        tok_end = work_desc.tok_offset + work_desc.seq_len
        num_blocks = (
            work_desc.seq_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # math_tidx / wg_idx: valid for Math WG threads; LdSt WG gets negative values (unused)
        math_tidx = tidx - cutlass.Int32(THREADS_PER_WARP_GROUP)
        wg_idx = math_tidx // cutlass.Int32(THREADS_PER_WARP_GROUP)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0))
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0))
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, GateProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        beta_layout = cute.make_layout((self.BLK_KV, self.alpha_beta_stage))
        sBeta = storage.smem_beta.get_tensor(beta_layout)

        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        load_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        vector_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 128 if self.fused else 32
        )
        vector_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARP_GROUPS * THREADS_PER_WARP_GROUP
        )
        o_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        if cutlass.const_expr(self.fused):
            q_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.q_mbar_ptr.data_ptr(),
                num_stages=self.q_stage,
                producer_group=vector_producer_group,
                consumer_group=vector_consumer_group,
            )
            k_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                num_stages=self.k_stage,
                producer_group=vector_producer_group,
                consumer_group=vector_consumer_group,
            )
            v_pipeline = pipeline.PipelineAsync.create(
                barrier_storage=storage.v_mbar_ptr.data_ptr(),
                num_stages=self.v_stage,
                producer_group=vector_producer_group,
                consumer_group=vector_consumer_group,
            )
        else:
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar_ptr.data_ptr(),
                num_stages=self.q_stage,
                producer_group=load_producer_group,
                consumer_group=load_consumer_group,
                tx_count=self.tma_load_q_bytes,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            k_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                num_stages=self.k_stage,
                producer_group=load_producer_group,
                consumer_group=load_consumer_group,
                tx_count=self.tma_load_k_bytes,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            v_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.v_mbar_ptr.data_ptr(),
                num_stages=self.v_stage,
                producer_group=load_producer_group,
                consumer_group=load_consumer_group,
                tx_count=self.tma_load_v_bytes,
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
        o_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.o_mbar_ptr.data_ptr(),
            num_stages=self.o_stage,
            producer_group=o_producer_group,
            consumer_group=o_consumer_group,
        )
        alpha_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.alpha_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
        )
        beta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.beta_mbar_ptr.data_ptr(),
            num_stages=self.alpha_beta_stage,
            producer_group=vector_producer_group,
            consumer_group=vector_consumer_group,
        )
        if cutlass.const_expr(self.fused):
            if tidx == 0:
                for stage in cutlass.range_constexpr(4):
                    cute.arch.mbarrier_init(storage.raw_mbar_ptr.data_ptr() + stage, 1)
        if cutlass.const_expr(self.watchdog):
            self.watchdog_words = storage.watchdog_words.data_ptr()
            self.watchdog_order = storage.watchdog_barriers.data_ptr()
            self.watchdog_store = self.watchdog_order + 2
            self.watchdog_phase = cute.make_rmem_tensor((1,), cutlass.Int32)
            phase = self.watchdog_phase
            phase[0] = 0
            words = self.watchdog_words
            if tidx < 12:
                words[tidx] = 0
            if tidx == 0:
                cute.arch.mbarrier_init(self.watchdog_order, 256)
                cute.arch.mbarrier_init(self.watchdog_order + 1, 256)
                cute.arch.mbarrier_init(self.watchdog_store, 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if work_desc.seq_len != cutlass.Int32(0) and warp_group_idx == WarpGroupRole.LDST:
            cute.arch.setmaxregister_decrease(load_registers)
            if cutlass.const_expr(self.fused):
                self.run_fused_producer(
                    tma_atom_q,
                    tma_tensor_q,
                    storage.raw_mbar_ptr.data_ptr(),
                    g_alpha,
                    g_initial,
                    g_state_indices,
                    work_desc,
                    num_blocks,
                    ldst_warp_role,
                    sQ_SD,
                    sK_SD,
                    storage.smem_v.get_tensor(v_layout_sd.outer, swizzle=v_layout_sd.inner),
                    sAlpha,
                    sBeta,
                    sO,
                    q_pipeline,
                    k_pipeline,
                    v_pipeline,
                    alpha_pipeline,
                    beta_pipeline,
                    o_pipeline,
                    tma_atom_o,
                    tma_tensor_o,
                    g_tensormaps,
                    num_seqs,
                    scale,
                )
            else:
                if ldst_warp_role == LoadStoreWarpRole.LOAD_QKV:
                    self.run_load_qkv_role(
                        sQ_SD,
                        sK_DS,
                        sV_DS,
                        tma_atom_q,
                        tma_tensor_q,
                        tma_atom_k,
                        tma_tensor_k,
                        tma_atom_v,
                        tma_tensor_v,
                        q_pipeline,
                        k_pipeline,
                        v_pipeline,
                        num_blocks,
                        work_desc.tok_offset,
                        work_desc.q_head_idx(),
                        work_desc.k_head_idx(num_q_heads, num_v_heads),
                        work_desc.v_head_idx(),
                    )
                elif ldst_warp_role == LoadStoreWarpRole.STORE_O:
                    OutputStore(self.BLK_Q, self.D).run(
                        sO,
                        tma_atom_o,
                        tma_tensor_o,
                        g_tensormaps,
                        o_pipeline,
                        num_blocks,
                        work_desc,
                        num_seqs,
                        self.o_stage,
                        num_q_heads,
                        num_v_heads,
                    )
                elif ldst_warp_role == LoadStoreWarpRole.LOAD_BETA:
                    self.run_load_beta_role(
                        sBeta,
                        g_beta,
                        beta_pipeline,
                        num_blocks,
                        work_desc.tok_offset,
                        tok_end,
                        work_desc.o_head_idx(num_q_heads, num_v_heads),
                        num_sab_heads,
                    )
                elif ldst_warp_role == LoadStoreWarpRole.LOAD_ALPHA:
                    self.run_load_alpha_role(
                        sAlpha,
                        g_alpha,
                        alpha_pipeline,
                        scale,
                        num_blocks,
                        work_desc.tok_offset,
                        tok_end,
                        work_desc.o_head_idx(num_q_heads, num_v_heads),
                        num_sab_heads,
                    )
        elif work_desc.seq_len != cutlass.Int32(0):
            cute.arch.setmaxregister_increase(mma_registers)

            self.run_math_role(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sBeta,
                q_pipeline,
                k_pipeline,
                v_pipeline,
                o_pipeline,
                alpha_pipeline,
                beta_pipeline,
                g_state,
                g_initial,
                g_state_indices,
                work_desc,
                scale,
                wg_idx,
                math_tidx,
                num_blocks,
                num_q_heads,
                num_v_heads,
                num_sab_heads,
                num_seqs,
            )


@cute.kernel
def _read_nsmid(result: cute.Tensor):
    # PTX nsmid is an exclusive bound, including holes in SM numbering.
    result[0] = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %nsmid;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def read_nsmid(result, stream):
    _read_nsmid(result).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
