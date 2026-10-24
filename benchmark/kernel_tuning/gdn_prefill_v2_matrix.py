# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Frozen V2P0 acceptance contract. Pure Python; no device imports or dispatch."""

from benchmark.kernel_tuning.gdn_prefill_sol import LONG, SMALL

GEOMETRIES = {
    (16, 32): {"models": "Qwen3.5-9B / 35B-A3B", "tp": 1, "optional": False},
    (16, 48): {"models": "Qwen3.5-27B", "tp": 1, "optional": False},
    (8, 16): {"models": "Qwen3.5-9B / 35B-A3B", "tp": 2, "optional": False},
    (8, 24): {"models": "Qwen3.5-27B", "tp": 2, "optional": False},
    (4, 8): {"models": "Qwen3.5-9B / 35B-A3B", "tp": 4, "optional": True},
    (4, 12): {"models": "Qwen3.5-27B", "tp": 4, "optional": True},
}
CASES = tuple((tuple(lengths), tuple(initial)) for lengths, initial in SMALL + LONG)
DEVICES = {
    "rtx5090": {"name": "RTX 5090", "arch": "sm120"},
    "rtx-pro-6000": {"name": "RTX PRO 6000", "arch": "sm120"},
    "gb10": {"name": "GB10", "arch": "sm121"},
}

TIMING_BOUNDARY = """Whole GDN block: conv -> gates -> chunk recurrence -> gated norm.
Input projected is BF16 [T, P], with q|k|v|z|b|a column widths
Hk*128, Hk*128, Hv*128, Hv*128, Hv, Hv; P = 2*Hk*128 + 2*Hv*128 + 2*Hv.
Inputs also include BF16 conv weights [conv_channels, 4], FP32 A_log/dt_bias
[Hv], FP32 norm weight [128], int32 cu_seqlens [N+1], int32 slots [N], and
bool has_initial_state [N]. The FP32 state pool [S, Hv, 128, 128] and BF16
conv-history pool [S, 3, conv_channels] update in place; output y is BF16
[T, Hv*128]. conv_channels = 2*Hk*128 + Hv*128. Frozen rows use distinct valid
slots, no captures, and the listed initial-state flags. Norm uses swish, eps=1e-6.
Projection GEMMs, engine scheduling, metadata preparation, and the 256 MiB
L2 flush are outside timing. Time CUDA-graph replay; fresh-slot initialization,
provider copies and the whole block stay inside. Pools evolve between replays.
Comparators are the faster of vLLM 0.31.0 (FlashInfer 0.7.0.post1, native
separate qkvz/ba buffers prepared outside timing) and SGLang 72730d6 (vendored
FLA Triton), each with native history layout. Use 20 interleaved pairs;
speedup is median(min(SGLang_i, vLLM_i) / ours_i). Noise eps is
p95(abs(ratio-1)) of 20 self-pairs of the faster comparator, with linear
percentile interpolation. eps > 0.02 is inconclusive; otherwise pass at
speedup >= 1-eps.
TP2 parity (8x16, 8x24) is against the faster of vLLM and SGLang, with 20 pairs.
"""


def rows():
    """Return 68 acceptance rows per device (204 over DEVICES); exclude optional TP4."""
    return [
        {"hk": hk, "hv": hv, "case": case, "lengths": list(lengths), "initial": list(initial)}
        for (hk, hv), geometry in GEOMETRIES.items()
        if not geometry["optional"]
        for case, (lengths, initial) in enumerate(CASES)
    ]


def self_test():
    """Check the frozen counts, row identities and sequence contracts on CPU."""
    assert len(GEOMETRIES) == 6
    assert sum(not g["optional"] for g in GEOMETRIES.values()) == 4
    assert len(CASES) == len(set(CASES)) == 17
    assert len(DEVICES) == 3
    acceptance = rows()
    assert len(acceptance) == 68
    assert len(acceptance) * len(DEVICES) == 204
    assert len({(r["hk"], r["hv"], r["case"]) for r in acceptance}) == 68
    for row in acceptance:
        assert row["lengths"] and len(row["lengths"]) == len(row["initial"])
        assert all(type(length) is int and length >= 1 for length in row["lengths"])
        assert all(type(flag) is bool for flag in row["initial"])
