# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Signed-zero fix-up after a smaller-index top-k that ranks +0 above -0 (FlashInfer CUB).

Only a row whose selection boundary lies at zero can differ from the rule that treats -0 and
+0 as equal: then every selected value is >= 0, and the selected zeros must be the row's
smallest-ID zeros of either sign. Other rows return after one 16-value load.
"""

import triton
import triton.language as tl


@triton.jit
def zero_boundary_fixup(X, Vals, Ids, V, K: tl.constexpr, KP: tl.constexpr,
                        BLOCK: tl.constexpr):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, KP)
    slot = j < K
    v = tl.load(Vals + row * K + j, slot, float("inf")).to(tl.float32)
    if tl.min(v, 0) == 0.0:
        # Selected values above zero (NaN ranks above +inf) stay; the zeros are re-chosen.
        keep = (v > 0.0) | (v != v) | ~slot
        rank = tl.cumsum((~keep).to(tl.int32), 0) - 1  # r-th replaced slot takes the r-th zero
        need = tl.sum((~keep).to(tl.int32), 0)
        ids = tl.load(Ids + row * K + j, slot, 0)
        found = need * 0
        for start in range(0, V, BLOCK):
            if found < need:
                c = start + tl.arange(0, BLOCK)
                zero = (tl.load(X + row * V + c, c < V, 1.0).to(tl.float32) == 0.0) & (c < V)
                count = tl.sum(zero.to(tl.int32), 0)
                if count > 0:  # the rank match runs only on chunks that hold zeros
                    order = found + tl.cumsum(zero.to(tl.int32), 0) - 1
                    hit = zero[None, :] & (order[None, :] == rank[:, None]) & ~keep[:, None]
                    chosen = tl.sum(tl.where(hit, c[None, :], 0), 1).to(ids.dtype)
                    ids = tl.where(tl.sum(hit.to(tl.int32), 1) > 0, chosen, ids)
                found += count
        tl.store(Ids + row * K + j, ids, slot)
