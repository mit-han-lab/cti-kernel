# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Per-row DRAFT provider dispatch keyed by the hardware tuple.

For Q>32 auto prefers FA2, except symmetric noncausal W<Q requires CuTe and
CUTE_LONG_QUERY_ROWS (B=1 D=128 full/causal rows on the RTX 5090 and RTX PRO 6000).
Q=1 uses the Q=2 rules. For 2<=Q<=32 the measured rules remain unchanged:
for D=64 and D=128 the default provider is CuTe wherever it supports the geometry.
D=256 defaults to CuTe only for qualified causal-full rows in HD256_CUTE_ROWS;
other rows use FlashInfer. Explicit providers override these defaults.
For D=128 a rule routes a row to FlashInfer only where paired measurements found the
fallback's complete op faster than CuTe's (see
benchmark/kernel_tuning/results/dflash_df2_receipts.md and, at engine capacity,
dflash_dispatch_df2fix.csv: a row needs a >2% win at ctx 100 and at most a 2%
loss at ctx 400/1000, since one construction serves every runtime length; no
measured row on RTX 5090, RTX PRO 6000 or GB10 qualifies). Rows are keyed by the
construction-time geometry: B, Q, Hq/Hkv, the mask and the key union bound
C without a window, otherwise min(C, W + Q - 1); runtime lengths cannot change
the provider (graph capture).
"""

import torch

# (major, minor, SMs) -> rules; a rule matches when every given field matches.
# Fields: masks (set of "full"/"causal_full"/"causal_sliding"/"noncausal_sliding"), batch (set),
# query (inclusive range), group (set of Hq/Hkv), union (inclusive range).
FLASHINFER_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {}

# Q>32 rows (D=128) that take CuTe; other Q>32 rows use FA2. Paired graph rows, B=1
# 16/4 and 32/8 heads, Q50/Q147, full and causal: CuTe beat our FA2 device schedule at
# 500-2400 keys (RTX PRO 6000 B1 Q50 ctx 2398: 15.1 vs 18.4 us; ctx 500: 9.3 vs 17.0)
# and is more precise at every length (FP32 partials; FA2's BF16 partials equal SGLang
# FlashInfer bit for bit, which vLLM's unsplit FA2 can dominate). It is slower at long
# contexts (B1 Q50 ctx 16000: 52.8 vs 36.7 us; Q147 ctx 4000: 1.16x on the RTX 5090).
# B>=2 stays on FA2: 32/8 B4 Q50 ctx 4000 measured CuTe 1.97x (RTX PRO 6000).
# Receipt: benchmark/kernel_tuning/results/draft_window_dwr.md ("D=64 and Q>32").
CUTE_LONG_QUERY_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {
    (12, 0, 170): ({"masks": {"full", "causal_full"}, "batch": {1}},),
    (12, 0, 188): ({"masks": {"full", "causal_full"}, "batch": {1}},),
}

# D=64 rows that prefer FlashInfer, keyed like FLASHINFER_ROWS (masks, batch, query).
FLASHINFER_HD64_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {}

# D=256 per-layer graph chains, C=32768, Qwen3.5 16/4 heads, causal full,
# ctx=100/200/400/1000/4000: fewer losing contexts, then lower geomean ratio
# against min(vLLM vllm_flash_attn, SGLang FlashInfer 0.7.1).
# Receipt: benchmark/kernel_tuning/results/dflash_df3_hd256_dispatch.csv.
# batch_range is inclusive; None means no upper bound. These rules cover Q<=32.
# Every rule stops at B=4, the measured band. Beyond it CuTe's construction-time split
# count (target CTAs // (B*Hkv*row tiles)) falls to one, so a ragged batch's longest
# request is walked by one CTA per head, while FA2's device schedule chunks each request
# by its runtime length: RTX 5090 Q16 16/4 ragged B=5/8/16 at ctx 32768 measured CuTe
# 3.0-10.6x FA2's time, uniform B=8 ctx 8192 0.97x, uniform B=64 ctx 512 1.17x; GB10
# (X925-pinned, 60 pairs) ragged B=5/8/16 1.87-1.92x, uniform ctx 8192 1.01-1.05x, uniform
# B=64 ctx 512 0.87x (kept on FA2: B alone cannot tell it from a ragged batch); a
# random-shape evaluation row (8/2, B=83, P=172) 20x.
# Receipt: draft_window_dwr.md ("Large batches").
HD256_CUTE_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {
    (12, 0, 170): ({"masks": {"causal_full"}, "query": (12, 32), "batch_range": (1, 4)},),
    (12, 0, 188): (
        {"masks": {"causal_full"}, "query": (12, 32), "batch_range": (3, 4)},
        # Measured (dflash_df4_receipts.md, section 7): at B=2, Q16 CuTe's worst
        # context over ctx 100/400/1000/4000 is 1.017x the faster market, FlashInfer's
        # 1.148x (ctx 100). B=1 stays on FlashInfer (worst 1.046 vs CuTe 1.069).
        {"masks": {"causal_full"}, "query": (16, 32), "batch": {2}},
    ),
    # GB10 B=1 Q<=8 moved to FlashInfer in DF4: with PDL after the first call per
    # prepare, FlashInfer measured 0.89-0.97x CuTe at ctx 100/400/1000 in one call
    # (dflash_df4_receipts.md, GB10 section). Q>=9 keeps CuTe (mixed at Q16).
    (12, 1, 48): (
        {"masks": {"causal_full"}, "batch_range": (3, 4)},
        {"masks": {"causal_full"}, "query": (9, 32), "batch_range": (1, 4)},
    ),
}


# CuTe merge mode per row: "legacy" (attention + merge kernel) where paired A/B
# chains found DF3's single-launch inline merge slower than cti-kernels 0.1.12 at
# L=512/4096 (it wins only at L=128, B>=2). Runtime lengths cannot pick the mode,
# so two DFlash draft pairs (32/8 heads, D=128: Q16 causal sliding, Q8 noncausal
# sliding) take the 0.1.12 legacy merge on every device.
# Receipt: benchmark/kernel_tuning/results/draft_window_dw1.md.
# The Q16 full pair uses the distributed (inline) merge: per-layer chains at engine
# capacity C=32768 (ctx 100/300/1000/4000, B=1-4) against the faster of vLLM 0.31
# vllm_flash_attn and SGLang FlashInfer 0.7.1 found legacy 1.15-1.17x slower at
# ctx 100 (B2-4 on the RTX 5090, B1-2 on GB10) and distributed at most 0.93x
# (RTX 5090) / 1.02x (GB10 B1 ctx 100) of the market elsewhere.
# Receipt: benchmark/kernel_tuning/results/draft_window_dwr.md.
_DRAFT_PAIRS = (
    {"masks": {"causal_sliding"}, "query": (16, 16), "group": {4}},
    {"masks": {"noncausal_sliding"}, "query": (8, 8), "group": {4}},
)
LEGACY_MERGE_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {
    (12, 0, 170): _DRAFT_PAIRS,
    (12, 0, 188): _DRAFT_PAIRS,
    (12, 1, 48): _DRAFT_PAIRS,
}


# D=256 with GQA group >= 8 (Qwen3.5-35B-A3B VERIFY 16/2 and its TP2 shard 8/1),
# per-layer graph chains at C=32768, ctx 100/1000/4000, B 1-4, Q 8/16: the smallest
# worst ratio over contexts against the faster of vLLM vllm_flash_attn and SGLang
# FlashInfer 0.7.1 (dflash_df4_receipts.md, DFMOE). Rows here select CuTe; other
# group-8 rows use FlashInfer. Hardware without an entry, and other group sizes, use
# HD256_CUTE_ROWS: on the RTX 5090, 64/4 and 64/1 Q16 at B=1/3 (ctx 100/1000, paired
# CuTe vs FlashInfer graph chains) take CuTe by the smaller worst ratio except 64/1
# B=1, where FlashInfer is 6-10% faster (dflash_hd256_gqa_receipts.md).
# Like HD256_CUTE_ROWS, every rule stops at B=4 (the DFMOE band was B 1-4).
HD256_GQA8_CUTE_ROWS: dict[tuple[int, int, int], tuple[dict, ...]] = {
    (12, 0, 170): (
        {"masks": {"causal_full"}, "kv_heads": {2}, "batch": {1}, "query": (12, 32)},
        {"masks": {"causal_full"}, "kv_heads": {1}, "batch": {3}, "query": (12, 32)},
    ),
    (12, 0, 188): (
        {"masks": {"causal_full"}, "kv_heads": {2}, "query": (8, 11), "batch_range": (2, 4)},
        {"masks": {"causal_full"}, "kv_heads": {2}, "query": (12, 32), "batch_range": (1, 4)},
    ),
}


def mask_name(causal, window):
    if window is None:
        return "causal_full" if causal else "full"
    return "causal_sliding" if causal else "noncausal_sliding"


def hardware(device):
    device = torch.device(device)
    major, minor = torch.cuda.get_device_capability(device)
    return major, minor, torch.cuda.get_device_properties(device).multi_processor_count


def _matches(rule, row):
    if "batch_range" in rule:
        minimum, maximum = rule["batch_range"]
        if row["batch"] < minimum or (maximum is not None and row["batch"] > maximum):
            return False
    for field in ("masks", "batch", "group", "kv_heads"):
        if field in rule and row[field] not in rule[field]:
            return False
    for field in ("query", "union"):
        if field in rule and not rule[field][0] <= row[field] <= rule[field][1]:
            return False
    return True


def prefers_flashinfer(
    *,
    batch_size,
    query_length,
    max_kv_tokens,
    num_qo_heads,
    num_kv_heads,
    causal,
    window,
    device,
    head_dim=128,
    **kwargs,
):
    if query_length > 32:
        if not (causal or window is None or window >= query_length):
            return False
        row = {"masks": mask_name(causal, window), "batch": batch_size, "query": query_length}
        rules = CUTE_LONG_QUERY_ROWS.get(hardware(device), ()) if head_dim == 128 else ()
        return not any(_matches(rule, row) for rule in rules)
    query_length = max(query_length, 2)
    if head_dim == 64:
        # D=64 defaults to CuTe (FLASHINFER_HD64_ROWS lists measured exceptions; none).
        # Paired graph rows, 16/4 and 32/8 heads, Q16, B 1-16, ctx 128-16000: CuTe beat
        # our FA2 device schedule on every RTX 5090 row and on 8 of 9 RTX PRO 6000 rows
        # (B4 full ctx 4000: 1.06x), and is more precise (FP32 partials vs FA2's BF16).
        # Receipt: benchmark/kernel_tuning/results/draft_window_dwr.md ("D=64 and Q>32").
        row = {"masks": mask_name(causal, window), "batch": batch_size, "query": query_length}
        rules = FLASHINFER_HD64_ROWS.get(hardware(device), ())
        return any(_matches(rule, row) for rule in rules)
    if head_dim == 256:
        row = {
            "masks": mask_name(causal, window),
            "batch": batch_size,
            "query": query_length,
            "kv_heads": num_kv_heads,
        }
        table = HD256_CUTE_ROWS
        if num_qo_heads // num_kv_heads == 8 and hardware(device) in HD256_GQA8_CUTE_ROWS:
            table = HD256_GQA8_CUTE_ROWS
        return not any(_matches(rule, row) for rule in table.get(hardware(device), ()))
    if not FLASHINFER_ROWS:
        return False
    rules = FLASHINFER_ROWS.get(hardware(device), ())
    if not rules:
        return False
    union = max_kv_tokens if window is None else min(max_kv_tokens, window + query_length - 1)
    row = {
        "masks": mask_name(causal, window),
        "batch": batch_size,
        "query": query_length,
        "group": num_qo_heads // num_kv_heads,
        "union": union,
    }
    return any(_matches(rule, row) for rule in rules)


def prefers_legacy_merge(
    *, batch_size, query_length, num_qo_heads, num_kv_heads, head_dim, causal, window, device
):
    if head_dim != 128:
        return False
    row = {
        "masks": mask_name(causal, window),
        "batch": batch_size,
        "query": query_length,
        "group": num_qo_heads // num_kv_heads,
    }
    return any(_matches(rule, row) for rule in LEGACY_MERGE_ROWS.get(hardware(device), ()))
