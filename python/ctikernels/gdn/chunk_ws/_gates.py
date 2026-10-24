# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/alpha.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

import cutlass
import cutlass.cute as cute

# ─── GateProcessor ──────────────────────────────────────────────────────────
# Translates FlatMainloopTmaWarpSpecializedDeltaRule::GateProcessor.


class GateProcessor:
    CUMSUM_LOG = 0
    CUMPROD = 1
    CUMPROD_SCALE = 2
    NUM_CHANNELS = 3

    @cute.jit
    def run(
        self,
        vecs: cute.Tensor,
        scale: cutlass.Float32,
    ):
        WARP_SIZE = 32
        blk_q = cute.size(vecs.shape[0])
        num_iters = blk_q // WARP_SIZE

        lane_id = cute.arch.lane_idx()

        # vecs shape (blk_q, NUM_CHANNELS) col-major.
        vecs_32 = cute.flat_divide(vecs, (WARP_SIZE,))

        frag = cute.make_rmem_tensor(num_iters, cutlass.Float32)
        for i in cutlass.range_constexpr(num_iters):
            raw = cutlass.Float32(vecs_32[lane_id, i, GateProcessor.CUMSUM_LOG])
            frag[i] = raw  # Loader supplies g * log2(e); padding contributes zero.

        for log_off in cutlass.range_constexpr(5):  # offsets 1, 2, 4, 8, 16
            off = 1 << log_off
            for i in cutlass.range_constexpr(num_iters):
                v = cute.arch.shuffle_sync_up(frag[i], off, mask_and_clamp=0)
                if lane_id >= off:
                    frag[i] = frag[i] + v

        for i in cutlass.range_constexpr(1, num_iters):
            carry = cute.arch.shuffle_sync(frag[i - 1], 31)
            frag[i] = frag[i] + carry

        for i in cutlass.range_constexpr(num_iters):
            vecs_32[lane_id, i, GateProcessor.CUMSUM_LOG] = frag[i]
            cumprod = cute.math.exp2(frag[i], fastmath=True)
            vecs_32[lane_id, i, GateProcessor.CUMPROD] = cumprod
            vecs_32[lane_id, i, GateProcessor.CUMPROD_SCALE] = cumprod * scale
