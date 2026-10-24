# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 FlashInfer team
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from flashinfer 0.7.0.post1 (flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py,
#   flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py)
# FlashInfer revision 946200de1ae94fc93fdd0926f0a13afd1fa7f0f1; see THIRD_PARTY_NOTICES.

"""Bounded single-sequence planner with our provisional cost constants.

Unlike upstream T precompute, M/N use our chunk math (gate before inverse,
identity init / zero V). Segment zero's init is saved before any final pool write;
fixup is FP32 IEEE. No upstream T precompute implementation is shared.
"""

# Chunk units, fitted to paired whole-path runner sweeps (cp3_bench, RTX 5090 Hv 16/24/32/48;
# within ~6% of the measured CP times): an MN chunk costs 0.6 of a full chunk, each fixup step
# 0.4 (latency-bound FP32), every CTA pays about one chunk of pipeline fill/drain, and the extra
# launches about half a chunk.
A_COST = 0.6
FIX_COST = 0.4
SEGMENT_OVERHEAD = 1.0
LAUNCH_COST = 0.5
MAX_SEGMENTS = 16


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def cp_capacity(hv: int, num_sms: int) -> int:
    """Maximum candidate count over all T, also the allocated workspace bound."""
    _positive("hv", hv)
    _positive("num_sms", num_sms)
    return min(MAX_SEGMENTS, max(2, 2 * num_sms // hv + 1))


def cp_cost(j: int, length: int, hv: int, num_sms: int) -> float:
    """Estimated A + C waves, serial fixup, and incremental launch cost."""
    per_cta = length // 64 + SEGMENT_OVERHEAD
    return (
        ((2 * j - 3) * hv + num_sms - 1) // num_sms * per_cta * A_COST
        + (j * hv + num_sms - 1) // num_sms * per_cta
        + FIX_COST * max(j - 2, 0)
        + LAUNCH_COST
    )


def noncp_cost(chunks: int) -> float:
    """The single-launch estimate cp_cost competes with, in the same units."""
    return chunks + SEGMENT_OVERHEAD


def cp_plan(t: int, hv: int, num_sms: int, segments: int | None = None) -> tuple[int, int]:
    """Return actual (J,L), or (1,T) when CP loses to C non-CP chunks.

    An override forces the bounded candidate count, bypassing the cost gate.
    Alignment can reduce the actual J (e.g. T=200, target=3 gives J=2,L=128).
    This avoids empty segments and always preserves J=ceil(T/L).
    """
    _positive("t", t)
    cap = cp_capacity(hv, num_sms)
    if segments is not None:
        _positive("segments", segments)
    chunks = (t + 63) // 64
    limit = min(chunks, cap)
    if segments == 1 or limit < 2:
        return 1, t
    candidates = range(2, limit + 1) if segments is None else (min(segments, limit),)
    best, cost = (1, t), noncp_cost(chunks)
    for candidate in candidates:
        length = 64 * ((chunks + candidate - 1) // candidate)
        j = (t + length - 1) // length
        estimate = cp_cost(j, length, hv, num_sms)
        if segments is not None or estimate < cost:
            best, cost = (j, length), estimate
    return best
