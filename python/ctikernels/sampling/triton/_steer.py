# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from vLLM 0.31.0 db9527a46873454610df6dbedf79a36d6bf1a7f6
# (`vllm/v1/worker/gpu/sample/logit_bias.py`); see THIRD_PARTY_NOTICES.

"""Sparse steering kernel and launch for ``ctikernels.sampling.steer_logits_``: one program
per logits row adds its request's biases, then, while the row is short of min_tokens, writes
-inf to its stop tokens (vLLM's ``_bias_kernel`` without allowed ids or the all-masked
restore; CSR lists instead of capped per-slot arrays)."""

import triton
import triton.language as tl

_BLOCK = 128


@triton.jit
def steer_kernel(L, row_stride, Rows, Generated, MinTokens, StopIndptr, StopIds, BiasIndptr,
                 BiasIds, BiasValues, BLOCK: tl.constexpr):  # fmt: skip
    r = tl.program_id(0)
    row = L + r.to(tl.int64) * row_stride
    n = tl.load(Rows + r).to(tl.int64)
    j = tl.arange(0, BLOCK)
    lo, hi = tl.load(BiasIndptr + n), tl.load(BiasIndptr + n + 1)
    for start in range(lo, hi, BLOCK):
        k = start + j
        valid = k < hi
        ids = tl.load(BiasIds + k, valid, 0).to(tl.int64)
        x = tl.load(row + ids, valid)
        # One FP32 add, nothing to contract into an FMA: round to nearest.
        tl.store(row + ids, x + tl.load(BiasValues + k, valid, 0.0), valid)
    # A biased stop token: its -inf must land after the add from another thread.
    tl.debug_barrier()
    if tl.load(Generated + r).to(tl.int64) < tl.load(MinTokens + n).to(tl.int64):
        lo, hi = tl.load(StopIndptr + n), tl.load(StopIndptr + n + 1)
        for start in range(lo, hi, BLOCK):
            k = start + j
            valid = k < hi
            ids = tl.load(StopIds + k, valid, 0).to(tl.int64)
            tl.store(row + ids, tl.full([BLOCK], float("-inf"), tl.float32), valid)


def launch(logits, rows, generated, min_tokens, stop_indptr, stop_ids, bias_indptr, bias_ids,
           bias_values):  # fmt: skip
    """Launch over validated tensors with R > 0 (see ``ctikernels.sampling.steer_logits_``)."""
    steer_kernel[(logits.shape[0],)](
        logits, logits.stride(0), rows, generated, min_tokens, stop_indptr, stop_ids,
        bias_indptr, bias_ids, bias_values, BLOCK=_BLOCK, num_warps=4,
    )  # fmt: skip
