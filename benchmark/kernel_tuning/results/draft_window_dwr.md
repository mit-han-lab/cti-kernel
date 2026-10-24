<!-- SPDX-License-Identifier: MIT -->
# DW-R: DraftAttention SGLang range (A4, A6), long-context and K-Q7 rows (2026-10-16)

Branch `kernels/draft-window-rc`, on top of the DW1 merge (28fd62d1). It is merged with
`kernels/rc-0.1.13` 84364af (which contains DF4). Raw JSONL for every run below is in
`draft_window_dwr/`. Market rows keep per-pair timings; per-layer rows keep their median
and minimum.

## What changed

| item | change |
|---|---|
| A4: any Q | `DraftAttention` and `FlashInferFixedQueryAttention` accept any fixed Q >= 1 with C >= Q, on both CuTe and FA2. The FA2/FixedQuery prepares loop over 1024-lane chunks when Q > 32. The Q <= 32 compiled code is unchanged. CuTe already tiles packed query rows. `check_support` validates every compiled tier's grid, so auto routing falls back before sizing. Q > 32 prefers FA2, except noncausal W < Q, which needs CuTe. Q = 1 uses the Q = 2 rules. |
| A4 fix found on GPU | The FA2/FixedQuery sliding-window page trim kept W-1+tile keys. For Q > tile, the leftmost query needs W-1+Q, plus P-1 for a partial last page. The span is a constexpr `max(Q, tile)`, so the small-Q code is unchanged. |
| A6: noncausal W < Q off SM12x | A new provider, `provider="flashinfer-mask"` (reported as `flashinfer-fa2-custom-mask`). It trims each request's page CSR to the window union and packs a little-endian custom mask on device. It runs native `window_left=-1` with `MaskMode.CUSTOM`. The FlashInfer custom-mask ABI (byte-offset `mask_indptr`, bit `qo*kv_len+kv`) is the same on 0.6.11.post3, 0.7.0.post1 and 0.7.1; `docs/dflash_kernels.md` cites the source lines. Auto routing uses it when CuTe rejects: off sm_12x, or at CuTe grid limits. W, the mask stride and the byte-tile count are runtime Triton arguments, so nothing compiles per W. With DF4's steady-state FA2 PDL, the first run after each `prepare()` (which writes the mask) is serialized. |
| Market harness | `dflash_draft_market.py` passes `page_indices` only when the selected provider reads it (`requires_page_indices`). CuTe never reads the page CSR. With a buffer, its fused publish writes all B*C/P indices per call, which dominated short sliding-window rows at C=131072/262144 (GB10 noncausal W2048 B2: 2.1x before, 0.83x after). Opt-in K-Q7 masks: `causal_full`, `causal_sliding_2048`. |
| Q16 full pair merge (engine/dflash-kernels report) | `dispatch.LEGACY_MERGE_ROWS` drops the Q16 full pair. It now uses the distributed (inline) merge on every device; the causal W4096 and noncausal W2048 pairs keep the 0.1.12 legacy merge. Receipts below. On the RTX PRO 6000, the RC's 32-row rule keys on the effective inline merge (ae6048b), so the PRO full pair also changes tiling there. That leg is not measured yet (PRO reserved). |
| Doc fixes (review of 29f43c5) | `draft_window_sglang_range.md`: the Q row is closed by A4, and the window row discloses and then closes A6. |

## Correctness (FP32 reference, eager + graph replay with restaged lengths)

| suite | RTX 5090 | GB10 | RTX PRO 6000 |
|---|---|---|---|
| A4/A6 new files (`test_draft_any_query.py`, `test_draft_custom_mask.py`), not exhaustive, at 6517b11 | 162 passed | 162 passed | A4 file 71 passed (at 6e50347); exhaustive A4 rows found the Q > tile bug fixed in 6517b11; the rest is deferred |
| A4/A6 exhaustive rows at 6517b11 | 512 passed (3 shards) | - | deferred |
| draft core (attention, capture trace, cute, inline, integration, window, fixed query) at 6517b11 | 446 passed | 446 passed | deferred |
| long context C=131072/262144 + chunked-reference audit at 6517b11 | 96 passed | 96 passed | deferred |
| after the RC merge (b8a2309 + ab94ddb test fix): custom mask, any Q, DF4 table, window, attention (not exhaustive) | 515 passed + custom mask 42 passed | 515 passed + custom mask 42 passed | deferred |
| after the merge onto RC b4dce8a2 (ba8de206): DF4 table, large, window, attention (not exhaustive) | 387 passed | 387 passed | waived |

CPU suite at ab94ddb (the DW-R code head; 8018293b adds only these receipts): 14851 passed, 5 skipped, 12 failed. At 936b62bf
(after merging RC aeba9e39, which added tests): 15284 passed, 7 skipped, 22 failed. At
ba8de206 (merge onto RC b4dce8a2 plus these receipts): 16325 passed, 5 skipped, 20 failed, all 20
`test_moe_block_fp8_*` (venv-side; they do not reproduce in CI). b4dce8a2 fixes the two
`test_dflash_draft_hd256` failures counted in the earlier runs. With the venv used here, the RC-side
tests (`test_moe_block_fp8_*`, and before b4dce8a2 two `test_dflash_draft_hd256` provider overrides) also fail on
plain `kernels/rc-0.1.13`; this branch does not touch them. The two hd256 ones expect CuTe for 64/4
heads (group 16), while DF4's GQA-8 table routes group >= 8 at that per-rank head count to FlashInfer.

## Q16 full pair: legacy vs distributed merge (per-layer chains at engine capacity)

`dflash_draft_layers.py --shapes draft_noncausal --query 16 --capacity 32768`. n distinct layer
K/V rotate over at least 2x L2. Comparators: vLLM 0.31.0 `vllm_flash_attn` (P=16) and SGLang 72730d6
FlashInfer 0.7.1 fa2 (P=1). The legacy columns are the RC default before this change.

RTX 5090 (per-layer chains, C=32768, 32/8 hd128 Q16 full noncausal; ratio = ours / faster of vLLM and SGLang):

| B | ctx | vLLM us | SGLang us | legacy (0.1.13 RC) us | ratio | distributed (this branch) us | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 5.74 | 7.30 | 5.05 | 0.889 | 5.00 | 0.870 |
| 1 | 300 | 12.83 | 10.15 | 5.57 | 0.557 | 7.36 | 0.725 |
| 1 | 1000 | 30.92 | 12.87 | 7.98 | 0.628 | 9.43 | 0.733 |
| 1 | 4000 | 116.49 | 18.35 | 15.50 | 0.837 | 16.71 | 0.911 |
| 2 | 100 | 5.79 | 11.52 | 6.65 | 1.161 | 5.05 | 0.873 |
| 2 | 300 | 12.82 | 18.93 | 7.39 | 0.583 | 8.39 | 0.655 |
| 2 | 1000 | 30.98 | 20.11 | 11.08 | 0.557 | 12.19 | 0.606 |
| 2 | 4000 | 117.30 | 33.19 | 26.17 | 0.834 | 27.05 | 0.815 |
| 3 | 100 | 6.16 | 18.37 | 6.79 | 1.156 | 5.28 | 0.858 |
| 3 | 300 | 13.23 | 19.76 | 7.98 | 0.627 | 9.70 | 0.734 |
| 3 | 1000 | 31.60 | 21.46 | 13.95 | 0.652 | 15.59 | 0.726 |
| 3 | 4000 | 117.82 | 40.84 | 36.53 | 0.879 | 38.16 | 0.934 |
| 4 | 100 | 6.44 | 18.79 | 6.98 | 1.148 | 5.42 | 0.841 |
| 4 | 300 | 13.17 | 20.54 | 8.54 | 0.661 | 10.21 | 0.775 |
| 4 | 1000 | 31.87 | 22.71 | 16.39 | 0.719 | 17.87 | 0.787 |
| 4 | 4000 | 118.27 | 55.27 | 45.97 | 0.838 | 48.92 | 0.885 |

GB10 (same; X925-pinned):

| B | ctx | vLLM us | SGLang us | legacy (0.1.13 RC) us | ratio | distributed (this branch) us | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 5.71 | 9.49 | 6.63 | 1.171 | 5.67 | 0.993 |
| 1 | 300 | 11.68 | 14.50 | 10.67 | 0.867 | 10.39 | 0.890 |
| 1 | 1000 | 30.22 | 27.53 | 23.45 | 0.848 | 24.55 | 0.892 |
| 1 | 4000 | 105.26 | 82.35 | 73.24 | 0.875 | 74.97 | 0.910 |
| 2 | 100 | 9.93 | 14.16 | 11.09 | 1.151 | 6.30 | 0.634 |
| 2 | 300 | 21.12 | 22.73 | 18.07 | 0.840 | 13.55 | 0.642 |
| 2 | 1000 | 55.00 | 48.88 | 42.35 | 0.871 | 41.69 | 0.853 |
| 2 | 4000 | 232.19 | 162.87 | 165.62 | 1.008 | 147.30 | 0.904 |
| 3 | 100 | 10.33 | 15.89 | 10.81 | 1.035 | 8.23 | 0.797 |
| 3 | 300 | 23.19 | 31.46 | 20.18 | 0.896 | 18.11 | 0.781 |
| 3 | 1000 | 62.23 | 70.15 | 58.65 | 0.940 | 62.67 | 1.007 |
| 3 | 4000 | 232.81 | 238.23 | 232.01 | 0.972 | 233.77 | 1.004 |
| 4 | 100 | 15.32 | 16.18 | 15.10 | 1.034 | 10.19 | 0.665 |
| 4 | 300 | 32.34 | 40.20 | 27.02 | 0.851 | 24.31 | 0.752 |
| 4 | 1000 | 85.86 | 92.31 | 74.89 | 0.880 | 71.16 | 0.829 |
| 4 | 4000 | 318.14 | 313.34 | 310.25 | 0.989 | 284.11 | 0.907 |

Distributed is at most 0.934 of the faster market time on the RTX 5090 (all 16 rows). On GB10 it
is 0.634-0.993 on 14 rows and ties vLLM on two rows (B3 ctx 1000/4000: 1.007/1.004; unpaired
chains, about 1% run-to-run). Legacy lost at ctx 100 on both devices (1.15-1.17x). Rejected
alternatives, measured in the same harness: legacy with the short-union single split (`SHORT=288`)
was still 1.34-1.39x on the RTX 5090 at B2-4 ctx 100. `ROWS=64` was 1.17-1.60x on the RTX 5090 at ctx 100 and 1.43x on GB10 at B1.

## Long context, C = 131072 / 262144 (market harness, capacity = length, B = 1-4)

`dflash_draft_market.py`, 20 self-pairs to set epsilon, then 20 alternating pairs of 100 calls.
Each cell is the median of ours/comparator. "tie" means within the comparator's A/A epsilon.
vLLM's FA2 (sm_80 PTX JIT on sm_12x) failed at C=262144 on the RTX 5090 (AcceleratorError in its
worker).

RTX 5090, full mask (32/8 hd128 Q16):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 131072 | 32/8 | 336.6 | 1.002 | 1.009 | 0.087 | **no** |
| full | 2 | 16 | 131072 | 32/8 | 659.6 | 0.953 | 0.944 | 0.168 | yes |
| full | 3 | 16 | 131072 | 32/8 | 989.7 | 0.953 | 0.945 | 0.250 | yes |
| full | 4 | 16 | 131072 | 32/8 | 1324.6 | 0.917 | 0.913 | 0.335 | yes |
| full | 1 | 16 | 262144 | 32/8 | 655.1 | 0.999 | 0.979 | AcceleratorError | yes |
| full | 2 | 16 | 262144 | 32/8 | 1316.0 | 0.925 | 0.892 | AcceleratorError | yes |
| full | 3 | 16 | 262144 | 32/8 | 2003.0 | 0.921 | 0.863 | AcceleratorError | yes |
| full | 4 | 16 | 262144 | 32/8 | 2995.0 | 0.893 | 0.931 | AcceleratorError | yes |

RTX PRO 6000, full mask C=131072 (the C=262144 and sliding legs were not run: PRO reserved):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 131072 | 32/8 | 369.2 | 0.996 | 0.993 | 0.130 | yes |
| full | 2 | 16 | 131072 | 32/8 | 717.2 | 0.984 | 0.984 | 0.254 | yes |
| full | 3 | 16 | 131072 | 32/8 | 1071.5 | 0.989 | 0.989 | 0.380 | yes |
| full | 4 | 16 | 131072 | 32/8 | 1422.5 | 0.994 | 0.993 | 0.501 | yes |

GB10, full mask (measured with the legacy merge, before 30ced7f; see the B=2 recheck below):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 131072 | 32/8 | 2304.1 | 1.013 | 0.998 | 0.630 | yes |
| full | 2 | 16 | 131072 | 32/8 | 4955.7 | 1.091 | 1.094 | 0.744 | **no** |
| full | 3 | 16 | 131072 | 32/8 | 6887.4 | 1.009 | 1.008 | 0.922 | **no** |
| full | 4 | 16 | 131072 | 32/8 | 9135.1 | 1.007 | 1.009 | 0.836 | **no** |
| full | 1 | 16 | 262144 | 32/8 | 4477.1 | 1.016 | 1.016 | 0.621 | **no** |
| full | 2 | 16 | 262144 | 32/8 | 9638.7 | 1.103 | 1.108 | 0.710 | **no** |
| full | 3 | 16 | 262144 | 32/8 | 13312.2 | 1.010 | 1.013 | - | **no** |
| full | 4 | 16 | 262144 | 32/8 | 17686.2 | 1.015 | 1.027 | - | **no** |

RTX 5090, sliding masks (causal W4096 Q16, noncausal W2048 Q8):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| causal_sliding | 1 | 16 | 131072 | 32/8 | 14.4 | 0.876 | 0.876 | 0.119 | yes |
| causal_sliding | 1 | 16 | 262144 | 32/8 | 14.4 | 0.876 | 0.875 | 0.119 | yes |
| causal_sliding | 2 | 16 | 131072 | 32/8 | 21.8 | 0.659 | 0.626 | 0.183 | yes |
| causal_sliding | 2 | 16 | 262144 | 32/8 | 21.5 | 0.656 | 0.628 | 0.184 | yes |
| causal_sliding | 3 | 16 | 131072 | 32/8 | 26.8 | 0.622 | 0.622 | 0.218 | yes |
| causal_sliding | 3 | 16 | 262144 | 32/8 | 26.7 | 0.621 | 0.628 | 0.218 | yes |
| causal_sliding | 4 | 16 | 131072 | 32/8 | 32.8 | 0.592 | 0.617 | 0.267 | yes |
| causal_sliding | 4 | 16 | 262144 | 32/8 | 32.8 | 0.586 | 0.616 | 0.267 | yes |
| noncausal_sliding | 1 | 8 | 131072 | 32/8 | 8.3 | 0.575 | 0.574 | 0.129 | yes |
| noncausal_sliding | 1 | 8 | 262144 | 32/8 | 8.3 | 0.575 | 0.574 | 0.129 | yes |
| noncausal_sliding | 2 | 8 | 131072 | 32/8 | 10.3 | 0.716 | 0.716 | 0.158 | yes |
| noncausal_sliding | 2 | 8 | 262144 | 32/8 | 10.3 | 0.716 | 0.715 | 0.162 | yes |
| noncausal_sliding | 3 | 8 | 131072 | 32/8 | 12.3 | 0.501 | 0.501 | 0.188 | yes |
| noncausal_sliding | 3 | 8 | 262144 | 32/8 | 12.3 | 0.501 | 0.501 | 0.188 | yes |
| noncausal_sliding | 4 | 8 | 131072 | 32/8 | 14.4 | 0.467 | 0.468 | 0.220 | yes |
| noncausal_sliding | 4 | 8 | 262144 | 32/8 | 14.4 | 0.468 | 0.468 | 0.220 | yes |

GB10, sliding masks:

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| causal_sliding | 1 | 16 | 131072 | 32/8 | 26.7 | 1.000 | 1.011 | 0.325 | tie (<=1+eps) |
| causal_sliding | 1 | 16 | 262144 | 32/8 | 26.7 | 1.000 | 1.012 | 0.304 | tie (<=1+eps) |
| causal_sliding | 2 | 16 | 131072 | 32/8 | 152.0 | 0.970 | 0.973 | 0.700 | yes |
| causal_sliding | 2 | 16 | 262144 | 32/8 | 150.2 | 0.953 | 0.955 | 0.706 | yes |
| causal_sliding | 3 | 16 | 131072 | 32/8 | 231.9 | 0.967 | 0.951 | 0.916 | yes |
| causal_sliding | 3 | 16 | 262144 | 32/8 | 229.3 | 0.968 | 0.950 | 0.922 | yes |
| causal_sliding | 4 | 16 | 131072 | 32/8 | 308.8 | 0.997 | 0.979 | 0.906 | yes |
| causal_sliding | 4 | 16 | 262144 | 32/8 | 307.9 | 0.984 | 0.979 | 0.915 | yes |
| noncausal_sliding | 1 | 8 | 131072 | 32/8 | 14.4 | 0.875 | 0.875 | 0.280 | yes |
| noncausal_sliding | 1 | 8 | 262144 | 32/8 | 14.4 | 0.875 | 0.875 | 0.281 | yes |
| noncausal_sliding | 2 | 8 | 131072 | 32/8 | 20.5 | 0.834 | 0.835 | 0.206 | yes |
| noncausal_sliding | 2 | 8 | 262144 | 32/8 | 20.4 | 0.827 | 0.834 | 0.216 | yes |
| noncausal_sliding | 3 | 8 | 131072 | 32/8 | 68.1 | 0.933 | 0.970 | 0.674 | yes |
| noncausal_sliding | 3 | 8 | 262144 | 32/8 | 83.7 | 0.917 | 0.925 | 0.600 | yes |
| noncausal_sliding | 4 | 8 | 131072 | 32/8 | 143.8 | 0.920 | 0.895 | 0.815 | yes |
| noncausal_sliding | 4 | 8 | 262144 | 32/8 | 142.2 | 0.884 | 0.896 | 0.811 | yes |

GB10 full rows above predate the merge change. Re-measured on the current code, every GB10 full
row at C=131072/262144 is within the SGLang A/A epsilon (0.989-1.008), and B2 dropped from 1.094 to 0.997:

GB10, full mask on the current code (vLLM omitted at C=262144 to fit the queue cap; it is 0.62-0.71 in the earlier run):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 131072 | 32/8 | 2293.3 | - | 1.008 | 0.624 | tie (<=1+eps) |
| full | 2 | 16 | 131072 | 32/8 | 4534.6 | - | 0.997 | 0.685 | yes |
| full | 3 | 16 | 131072 | 32/8 | 6867.8 | - | 1.004 | 0.841 | tie (<=1+eps) |
| full | 4 | 16 | 131072 | 32/8 | 9002.1 | - | 0.993 | 0.840 | yes |
| full | 1 | 16 | 262144 | 32/8 | 4514.5 | - | 0.991 | - | yes |
| full | 2 | 16 | 262144 | 32/8 | 9078.5 | - | 1.001 | - | tie (<=1+eps) |
| full | 3 | 16 | 262144 | 32/8 | 13662.8 | - | 1.004 | - | tie (<=1+eps) |
| full | 4 | 16 | 262144 | 32/8 | 17922.4 | - | 0.989 | - | yes |
 RTX 5090 B1 C=131072 is 1.009 vs SGLang in this run; the RTX PRO 6000
measures the same row at 0.993.

## K-Q7 (Qwen3-30B-A3B DFlash draft, 32/4 hd128), market harness

These rows do not touch the merge change (group 8). The losses concentrate in Q16 at
ctx 32768 (all B, both devices) and in GB10 B1-2 Q8 at ctx 512-4096.

RTX 5090, K-Q7 (Qwen3-30B-A3B DFlash draft geometry, 32/4 hd128):

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.800 | yes |
| full | 1 | 16 | 512 | 32/4 | 8.3 | 0.402 | 0.402 | 0.402 | yes |
| full | 1 | 16 | 4096 | 32/4 | 16.4 | 0.727 | 0.727 | 0.137 | yes |
| full | 2 | 16 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.799 | yes |
| full | 2 | 16 | 512 | 32/4 | 12.3 | 0.548 | 0.552 | 0.600 | yes |
| full | 2 | 16 | 4096 | 32/4 | 22.6 | 0.917 | 0.917 | 0.188 | yes |
| full | 3 | 16 | 128 | 32/4 | 8.2 | 0.396 | 0.399 | 0.800 | yes |
| full | 3 | 16 | 512 | 32/4 | 14.4 | 0.637 | 0.637 | 0.700 | yes |
| full | 3 | 16 | 4096 | 32/4 | 28.8 | 0.998 | 1.000 | 0.238 | tie (<=1+eps) |
| full | 4 | 16 | 128 | 32/4 | 8.2 | 0.365 | 0.365 | 0.800 | yes |
| full | 4 | 16 | 512 | 32/4 | 14.4 | 0.637 | 0.637 | 0.683 | yes |
| full | 4 | 16 | 4096 | 32/4 | 34.9 | 0.945 | 0.945 | 0.288 | yes |
| causal_full | 1 | 16 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.798 | yes |
| causal_full | 1 | 16 | 512 | 32/4 | 8.3 | 0.402 | 0.402 | 0.402 | yes |
| causal_full | 1 | 16 | 4096 | 32/4 | 16.4 | 0.727 | 0.725 | 0.136 | yes |
| causal_full | 2 | 16 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.799 | yes |
| causal_full | 2 | 16 | 512 | 32/4 | 12.3 | 0.555 | 0.547 | 0.589 | yes |
| causal_full | 2 | 16 | 4096 | 32/4 | 23.8 | 0.967 | 0.960 | 0.203 | yes |
| causal_full | 3 | 16 | 128 | 32/4 | 8.2 | 0.389 | 0.373 | 0.798 | yes |
| causal_full | 3 | 16 | 512 | 32/4 | 14.4 | 0.637 | 0.637 | 0.637 | yes |
| causal_full | 3 | 16 | 4096 | 32/4 | 30.8 | 1.003 | 1.026 | 0.254 | tie (<=1+eps) |
| causal_full | 4 | 16 | 128 | 32/4 | 8.2 | 0.365 | 0.365 | 0.799 | yes |
| causal_full | 4 | 16 | 512 | 32/4 | 14.4 | 0.637 | 0.637 | 0.645 | yes |
| causal_full | 4 | 16 | 4096 | 32/4 | 36.5 | 0.987 | 0.991 | 0.303 | yes |
| causal_full | 1 | 8 | 128 | 32/4 | 8.2 | 0.666 | 0.666 | 0.799 | yes |
| causal_full | 1 | 8 | 512 | 32/4 | 8.2 | 0.666 | 0.618 | 0.401 | yes |
| causal_full | 1 | 8 | 4096 | 32/4 | 12.3 | 0.856 | 0.765 | 0.102 | yes |
| causal_full | 2 | 8 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.799 | yes |
| causal_full | 2 | 8 | 512 | 32/4 | 10.3 | 0.502 | 0.502 | 0.542 | yes |
| causal_full | 2 | 8 | 4096 | 32/4 | 16.5 | 0.729 | 0.729 | 0.138 | yes |
| causal_full | 3 | 8 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.798 | yes |
| causal_full | 3 | 8 | 512 | 32/4 | 11.4 | 0.557 | 0.560 | 0.546 | yes |
| causal_full | 3 | 8 | 4096 | 32/4 | 20.5 | 0.769 | 0.769 | 0.170 | yes |
| causal_full | 4 | 8 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.799 | yes |
| causal_full | 4 | 8 | 512 | 32/4 | 12.3 | 0.600 | 0.600 | 0.546 | yes |
| causal_full | 4 | 8 | 4096 | 32/4 | 22.6 | 0.689 | 0.689 | 0.188 | yes |
| causal_sliding_2048 | 1 | 8 | 128 | 32/4 | 8.2 | 0.666 | 0.666 | 0.799 | yes |
| causal_sliding_2048 | 1 | 8 | 512 | 32/4 | 8.2 | 0.661 | 0.660 | 0.401 | yes |
| causal_sliding_2048 | 1 | 8 | 4096 | 32/4 | 10.3 | 0.714 | 0.715 | 0.157 | yes |
| causal_sliding_2048 | 2 | 8 | 128 | 32/4 | 8.2 | 0.402 | 0.401 | 0.800 | yes |
| causal_sliding_2048 | 2 | 8 | 512 | 32/4 | 10.3 | 0.502 | 0.502 | 0.502 | yes |
| causal_sliding_2048 | 2 | 8 | 4096 | 32/4 | 14.3 | 0.696 | 0.698 | 0.219 | yes |
| causal_sliding_2048 | 3 | 8 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.799 | yes |
| causal_sliding_2048 | 3 | 8 | 512 | 32/4 | 11.5 | 0.561 | 0.563 | 0.597 | yes |
| causal_sliding_2048 | 3 | 8 | 4096 | 32/4 | 16.4 | 0.736 | 0.798 | 0.250 | yes |
| causal_sliding_2048 | 4 | 8 | 128 | 32/4 | 8.2 | 0.401 | 0.401 | 0.798 | yes |
| causal_sliding_2048 | 4 | 8 | 512 | 32/4 | 12.3 | 0.600 | 0.600 | 0.550 | yes |
| causal_sliding_2048 | 4 | 8 | 4096 | 32/4 | 18.5 | 0.818 | 0.818 | 0.282 | yes |
| full | 1 | 16 | 32768 | 32/4 | 70.2 | 1.260 | 1.286 | 0.075 | **no** |
| full | 2 | 16 | 32768 | 32/4 | 137.3 | 1.338 | 1.344 | 0.141 | **no** |
| full | 3 | 16 | 32768 | 32/4 | 219.3 | 1.510 | 1.518 | 0.225 | **no** |
| full | 4 | 16 | 32768 | 32/4 | 323.2 | 1.746 | 1.740 | 0.332 | **no** |
| causal_full | 1 | 16 | 32768 | 32/4 | 71.7 | 1.272 | 1.275 | 0.077 | **no** |
| causal_full | 2 | 16 | 32768 | 32/4 | 139.1 | 1.343 | 1.340 | 0.143 | **no** |
| causal_full | 3 | 16 | 32768 | 32/4 | 221.5 | 1.521 | 1.520 | 0.227 | **no** |
| causal_full | 4 | 16 | 32768 | 32/4 | 326.5 | 1.760 | 1.755 | 0.334 | **no** |
| causal_full | 1 | 8 | 32768 | 32/4 | 38.9 | 1.117 | 1.118 | 0.042 | tie (<=1+eps) |
| causal_full | 2 | 8 | 32768 | 32/4 | 94.2 | 0.958 | 0.957 | 0.096 | yes |
| causal_full | 3 | 8 | 32768 | 32/4 | 133.5 | 0.949 | 0.947 | 0.136 | yes |
| causal_full | 4 | 8 | 32768 | 32/4 | 175.2 | 0.959 | 0.953 | 0.178 | yes |
| causal_sliding_2048 | 1 | 8 | 32768 | 32/4 | 10.3 | 0.714 | 0.715 | 0.156 | yes |
| causal_sliding_2048 | 2 | 8 | 32768 | 32/4 | 14.4 | 0.639 | 0.637 | 0.213 | yes |
| causal_sliding_2048 | 3 | 8 | 32768 | 32/4 | 16.4 | 0.728 | 0.728 | 0.243 | yes |
| causal_sliding_2048 | 4 | 8 | 32768 | 32/4 | 18.5 | 0.818 | 0.818 | 0.273 | yes |

GB10, K-Q7:

| mask | B | Q | ctx | heads | ours us | / FA2 0.6.11 | / SGLang | / vLLM | ours <= best market |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 16 | 128 | 32/4 | 8.2 | 0.445 | 0.445 | 0.889 | yes |
| full | 1 | 16 | 512 | 32/4 | 16.4 | 0.850 | 0.848 | 0.999 | yes |
| full | 1 | 16 | 4096 | 32/4 | 28.7 | 1.056 | 1.049 | 0.327 | tie (<=1+eps) |
| full | 2 | 16 | 128 | 32/4 | 8.2 | 0.390 | 0.392 | 0.535 | yes |
| full | 2 | 16 | 512 | 32/4 | 18.5 | 0.818 | 0.818 | 0.601 | yes |
| full | 2 | 16 | 4096 | 32/4 | 45.3 | 1.104 | 1.054 | 0.260 | tie (<=1+eps) |
| full | 3 | 16 | 128 | 32/4 | 10.3 | 0.418 | 0.418 | 0.627 | yes |
| full | 3 | 16 | 512 | 32/4 | 20.5 | 0.769 | 0.768 | 0.648 | yes |
| full | 3 | 16 | 4096 | 32/4 | 93.0 | 0.860 | 0.838 | 0.578 | yes |
| full | 4 | 16 | 128 | 32/4 | 10.3 | 0.380 | 0.384 | 0.457 | yes |
| full | 4 | 16 | 512 | 32/4 | 18.5 | 0.640 | 0.638 | 0.410 | yes |
| full | 4 | 16 | 4096 | 32/4 | 166.6 | 0.897 | 0.869 | 0.525 | yes |
| causal_full | 1 | 16 | 128 | 32/4 | 8.2 | 0.446 | 0.446 | 0.803 | yes |
| causal_full | 1 | 16 | 512 | 32/4 | 16.4 | 0.871 | 0.857 | 0.929 | yes |
| causal_full | 1 | 16 | 4096 | 32/4 | 28.8 | 1.050 | 1.044 | 0.322 | tie (<=1+eps) |
| causal_full | 2 | 16 | 128 | 32/4 | 8.2 | 0.395 | 0.390 | 0.502 | yes |
| causal_full | 2 | 16 | 512 | 32/4 | 18.5 | 0.819 | 0.818 | 0.592 | yes |
| causal_full | 2 | 16 | 4096 | 32/4 | 47.2 | 1.150 | 1.106 | 0.264 | tie (<=1+eps) |
| causal_full | 3 | 16 | 128 | 32/4 | 10.3 | 0.418 | 0.418 | 0.626 | yes |
| causal_full | 3 | 16 | 512 | 32/4 | 20.5 | 0.769 | 0.770 | 0.625 | yes |
| causal_full | 3 | 16 | 4096 | 32/4 | 93.9 | 0.804 | 0.807 | 0.513 | yes |
| causal_full | 4 | 16 | 128 | 32/4 | 10.3 | 0.386 | 0.386 | 0.465 | yes |
| causal_full | 4 | 16 | 512 | 32/4 | 20.5 | 0.695 | 0.697 | 0.436 | yes |
| causal_full | 4 | 16 | 4096 | 32/4 | 166.7 | 0.905 | 0.848 | 0.552 | yes |
| causal_full | 1 | 8 | 128 | 32/4 | 8.2 | 0.667 | 0.660 | 0.804 | yes |
| causal_full | 1 | 8 | 512 | 32/4 | 14.4 | 1.144 | 1.081 | 0.802 | tie (<=1+eps) |
| causal_full | 1 | 8 | 4096 | 32/4 | 21.6 | 1.169 | 1.167 | 0.243 | **no** |
| causal_full | 2 | 8 | 128 | 32/4 | 8.2 | 0.446 | 0.446 | 0.502 | yes |
| causal_full | 2 | 8 | 512 | 32/4 | 16.4 | 0.889 | 0.888 | 0.522 | yes |
| causal_full | 2 | 8 | 4096 | 32/4 | 29.9 | 0.831 | 0.846 | 0.162 | yes |
| causal_full | 3 | 8 | 128 | 32/4 | 10.3 | 0.533 | 0.542 | 0.626 | yes |
| causal_full | 3 | 8 | 512 | 32/4 | 18.5 | 0.900 | 0.900 | 0.563 | yes |
| causal_full | 3 | 8 | 4096 | 32/4 | 89.4 | 0.966 | 0.896 | 0.491 | yes |
| causal_full | 4 | 8 | 128 | 32/4 | 10.3 | 0.501 | 0.498 | 0.456 | yes |
| causal_full | 4 | 8 | 512 | 32/4 | 18.5 | 0.822 | 0.819 | 0.394 | yes |
| causal_full | 4 | 8 | 4096 | 32/4 | 170.7 | 1.032 | 0.999 | 0.583 | yes |
| causal_sliding_2048 | 1 | 8 | 128 | 32/4 | 8.2 | 0.661 | 0.667 | 0.805 | yes |
| causal_sliding_2048 | 1 | 8 | 512 | 32/4 | 14.4 | 1.088 | 1.103 | 0.803 | **no** |
| causal_sliding_2048 | 1 | 8 | 4096 | 32/4 | 18.5 | 1.197 | 1.230 | 0.374 | **no** |
| causal_sliding_2048 | 2 | 8 | 128 | 32/4 | 8.2 | 0.446 | 0.446 | 0.502 | yes |
| causal_sliding_2048 | 2 | 8 | 512 | 32/4 | 16.4 | 0.889 | 0.888 | 0.523 | yes |
| causal_sliding_2048 | 2 | 8 | 4096 | 32/4 | 22.6 | 0.916 | 0.917 | 0.233 | yes |
| causal_sliding_2048 | 3 | 8 | 128 | 32/4 | 10.3 | 0.547 | 0.556 | 0.626 | yes |
| causal_sliding_2048 | 3 | 8 | 512 | 32/4 | 18.5 | 0.900 | 0.901 | 0.563 | yes |
| causal_sliding_2048 | 3 | 8 | 4096 | 32/4 | 26.7 | 0.813 | 0.866 | 0.266 | yes |
| causal_sliding_2048 | 4 | 8 | 128 | 32/4 | 10.3 | 0.501 | 0.501 | 0.456 | yes |
| causal_sliding_2048 | 4 | 8 | 512 | 32/4 | 18.5 | 0.824 | 0.821 | 0.393 | yes |
| causal_sliding_2048 | 4 | 8 | 4096 | 32/4 | 32.4 | 0.829 | 0.884 | 0.227 | yes |
| full | 1 | 16 | 32768 | 32/4 | 342.2 | 1.067 | 1.075 | 0.452 | **no** |
| full | 2 | 16 | 32768 | 32/4 | 787.9 | 1.290 | 1.296 | 0.528 | **no** |
| full | 3 | 16 | 32768 | 32/4 | 917.6 | 1.021 | 1.023 | 0.603 | tie (<=1+eps) |
| full | 4 | 16 | 32768 | 32/4 | 1225.3 | 1.029 | 1.022 | 0.532 | **no** |
| causal_full | 1 | 16 | 32768 | 32/4 | 343.4 | 1.069 | 1.061 | 0.435 | **no** |
| causal_full | 2 | 16 | 32768 | 32/4 | 792.9 | 1.286 | 1.292 | 0.513 | **no** |
| causal_full | 3 | 16 | 32768 | 32/4 | 916.9 | 1.014 | 1.022 | 0.600 | **no** |
| causal_full | 4 | 16 | 32768 | 32/4 | 1226.7 | 1.024 | 1.023 | 0.529 | **no** |
| causal_full | 1 | 8 | 32768 | 32/4 | 318.8 | 1.023 | 1.026 | 0.403 | **no** |
| causal_full | 2 | 8 | 32768 | 32/4 | 617.3 | 1.030 | 1.021 | 0.398 | tie (<=1+eps) |
| causal_full | 3 | 8 | 32768 | 32/4 | 907.8 | 1.008 | 1.013 | 0.586 | tie (<=1+eps) |
| causal_full | 4 | 8 | 32768 | 32/4 | 1204.5 | 1.018 | 1.014 | 0.524 | **no** |
| causal_sliding_2048 | 1 | 8 | 32768 | 32/4 | 18.5 | 1.211 | 1.201 | 0.370 | **no** |
| causal_sliding_2048 | 2 | 8 | 32768 | 32/4 | 22.6 | 0.919 | 0.922 | 0.230 | yes |
| causal_sliding_2048 | 3 | 8 | 32768 | 32/4 | 26.7 | 0.822 | 0.866 | 0.267 | yes |
| causal_sliding_2048 | 4 | 8 | 32768 | 32/4 | 35.6 | 0.901 | 0.987 | 0.226 | yes |

Diagnosis: Q16 x G8 = 128 packed query rows. The CuTe tile has 32 rows (64 at most), so 2-4
row tiles each run the full QK/PV over the same keys. On the RTX 5090 at B4 ctx 32768, the
attention math runs at about 106 TFLOP/s, against about 180 for FA2 (128-row tiles).
- Launching the row tiles adjacently (row-tile-major grid) changed nothing on either device, so
  L2 reuse is not the limit.
- `ROWS=64` recovers GB10 B2 at ctx 32768 (1.276 -> 1.022) but loses at short contexts for the
  32/8 pair.
- The legacy merge recovers GB10 ctx 512-4096 (for example Q16 B1 ctx 4096: 1.05 -> 0.89) but
  is 1.2-1.7x slower at ctx 32768 on the RTX PRO 6000 and RTX 5090 (`*_kq7_legacy.jsonl`).

Engine regime (per-layer chains at the model's 40960 capacity) shows the same split. The default wins ctx
100-1000 everywhere and loses at ctx 32000; `ROWS=64` trades short-context wins for long-context ones:

RTX 5090, K-Q7 per-layer chains (`--shapes q3moe_draft`, 32/4 hd128 causal, Q16, C=40960):

| B | ctx | vLLM us | SGLang us | default us | ratio | ROWS=64 us | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 5.81 | 17.95 | 5.03 | 0.865 | 6.73 | 1.159 |
| 1 | 300 | 13.12 | 19.34 | 7.28 | 0.555 | 10.31 | 0.787 |
| 1 | 1000 | 31.23 | 19.71 | 8.58 | 0.436 | 10.18 | 0.517 |
| 1 | 4000 | 116.79 | 21.85 | 14.72 | 0.674 | 15.60 | 0.714 |
| 1 | 32000 | 898.36 | 53.06 | 69.52 | 1.310 | 58.14 | 1.096 |
| 2 | 100 | 5.84 | 18.66 | 5.01 | 0.858 | 6.70 | 1.147 |
| 2 | 300 | 13.13 | 20.03 | 8.36 | 0.637 | 10.83 | 0.825 |
| 2 | 1000 | 31.28 | 20.82 | 11.30 | 0.543 | 12.36 | 0.594 |
| 2 | 4000 | 117.41 | 24.54 | 22.22 | 0.905 | 23.15 | 0.944 |
| 2 | 32000 | 948.83 | 97.59 | 133.45 | 1.367 | 116.21 | 1.189 |
| 3 | 100 | 6.24 | 19.14 | 5.07 | 0.813 | 6.77 | 1.087 |
| 3 | 300 | 13.51 | 20.74 | 9.12 | 0.675 | 11.48 | 0.849 |
| 3 | 1000 | 32.00 | 21.54 | 13.89 | 0.645 | 15.29 | 0.706 |
| 3 | 4000 | 118.45 | 30.44 | 32.71 | 1.075 | 27.81 | 0.911 |
| 3 | 32000 | 949.88 | 138.50 | 214.61 | 1.550 | 160.49 | 1.160 |
| 4 | 100 | 6.30 | 19.59 | 5.07 | 0.804 | 6.75 | 1.073 |
| 4 | 300 | 13.61 | 21.53 | 9.97 | 0.732 | 12.20 | 0.897 |
| 4 | 1000 | 32.44 | 22.62 | 16.13 | 0.713 | 16.67 | 0.738 |
| 4 | 4000 | 118.60 | 34.19 | 45.00 | 1.316 | 35.73 | 1.043 |
| 4 | 32000 | 950.80 | 179.74 | 318.06 | 1.770 | 218.73 | 1.220 |

GB10, same:

| B | ctx | vLLM us | SGLang us | default us | ratio | ROWS=64 us | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 5.24 | 13.97 | 5.09 | 0.971 | 6.74 | 1.220 |
| 1 | 300 | 11.09 | 17.31 | 8.73 | 0.788 | 10.94 | 0.991 |
| 1 | 1000 | 24.48 | 19.60 | 17.69 | 0.903 | 19.91 | 1.001 |
| 1 | 4000 | 85.56 | 49.42 | 45.81 | 0.927 | 44.60 | 0.905 |
| 1 | 32000 | 733.59 | 318.28 | 334.36 | 1.051 | 324.00 | 1.031 |
| 2 | 100 | 9.50 | 14.92 | 5.40 | 0.569 | 6.97 | 0.726 |
| 2 | 300 | 20.28 | 20.71 | 9.99 | 0.493 | 12.11 | 0.599 |
| 2 | 1000 | 48.62 | 32.10 | 26.74 | 0.833 | 29.55 | 0.924 |
| 2 | 4000 | 172.27 | 88.40 | 94.25 | 1.066 | 82.20 | 0.926 |
| 2 | 32000 | 1432.36 | 603.13 | 749.53 | 1.243 | 622.82 | 1.046 |
| 3 | 100 | 10.00 | 16.05 | 7.21 | 0.721 | 7.42 | 0.740 |
| 3 | 300 | 20.84 | 25.83 | 13.79 | 0.662 | 13.43 | 0.637 |
| 3 | 1000 | 49.88 | 44.33 | 36.69 | 0.828 | 37.26 | 0.840 |
| 3 | 4000 | 178.16 | 136.99 | 132.56 | 0.968 | 130.40 | 0.959 |
| 3 | 32000 | 1456.81 | 879.86 | 930.66 | 1.058 | 924.09 | 1.044 |
| 4 | 100 | 14.33 | 16.02 | 7.83 | 0.547 | 8.09 | 0.565 |
| 4 | 300 | 30.36 | 30.60 | 14.63 | 0.482 | 14.22 | 0.471 |
| 4 | 1000 | 73.36 | 56.83 | 37.77 | 0.665 | 42.04 | 0.742 |
| 4 | 4000 | 265.48 | 179.58 | 152.84 | 0.851 | 156.63 | 0.871 |
| 4 | 32000 | 2259.23 | 1172.17 | 1199.48 | 1.023 | 1218.71 | 1.037 |

No ctx-independent rule wins every row, so routing for group 8 is unchanged. The fix is
kernel-side: 128-row tiles (or a two-tile K/V share) for Q*G >= 128.

## Large batches (randomized draft-attention rows, B > 4)

The randomized draft-attention evaluation samples the whole SGLang range (B 1-256, Q 1-256, P 1-256,
ragged and inactive requests). Its rows at B >= 60 lost by up to 17x. A repro of three such
rows on the RTX 5090 timed each provider in a 10-call CUDA graph, prepare + run, against
FlashInfer 0.6.11 FA2 planned on the host with the actual lengths (SGLang's kind of plan).
Raw rows: `draft_window_dwr/rtx5090_large_batch.jsonl`.

| row | auto picks | CuTe us | our FA2 us | host-planned FlashInfer us |
|---|---|---|---|---|
| 8/2 hd256 causal full, B83 Q16 P172, one 57k request + 82 short | CuTe | 2083.6 | 105.6 | 101.2 |
| 16/4 hd128 causal full, B60 Q16 P1, one 2k request + 59 short | CuTe | 52.8 | 27.4 | 17.1 |
| 32/8 hd128 full, B84 Q18 P1, 23 active, ragged up to 235k | CuTe | 4810.8 | 1897.8 | 1877.4 |

Cause: CuTe fixes its split count at construction, as target CTAs // (B * Hkv * row tiles). At
B=83 that is one split, so one CTA per head walks the 57k-key request while the rest of the
grid idles. FA2's device schedule chunks every request by its runtime length.

A sweep on the RTX 5090 (P=1, C = top + Q; pattern 0 = every request at the top context,
pattern 2 = one request at the top context and the others at 0-64) shows that no rule over B
alone wins every hd128 row:

| geometry | B | pattern, top | CuTe us | our FA2 us | CuTe / FA2 |
|---|---|---|---|---|---|
| 32/8 hd128 causal full Q16 | 5 | 2, 32768 | 317.6 | 105.8 | 3.00 |
| 32/8 hd128 causal full Q16 | 8 | 2, 32768 | 615.9 | 115.3 | 5.34 |
| 32/8 hd128 causal full Q16 | 16 | 2, 32768 | 618.9 | 143.2 | 4.32 |
| 32/8 hd128 causal full Q16 | 64 | 2, 32768 | 661.2 | 1848.0 | 0.36 |
| 32/8 hd128 causal full Q16 | 8 | 0, 8192 | 175.2 | 191.4 | 0.92 |
| 32/8 hd128 causal full Q16 | 16 | 0, 8192 | 377.8 | 446.9 | 0.85 |
| 32/8 hd128 causal full Q16 | 64 | 0, 8192 | 1431.8 | 1735.9 | 0.82 |
| 16/4 hd256 causal full Q16 | 5 | 2, 32768 | 305.9 | 100.2 | 3.05 |
| 16/4 hd256 causal full Q16 | 8 | 2, 32768 | 597.3 | 117.9 | 5.07 |
| 16/4 hd256 causal full Q16 | 16 | 2, 32768 | 1187.1 | 112.0 | 10.60 |
| 16/4 hd256 causal full Q16 | 8 | 0, 8192 | 178.5 | 184.1 | 0.97 |
| 16/4 hd256 causal full Q16 | 64 | 0, 512 | 126.6 | 108.0 | 1.17 |
| 32/8 hd128 noncausal W2048 Q8 | 8 | 2, 32768 | 16.3 | 21.9 | 0.74 |
| 32/8 hd128 noncausal W2048 Q8 | 16 | 0, 8192 | 91.9 | 127.6 | 0.72 |
| 32/8 hd128 noncausal W2048 Q8 | 64 | 2, 32768 | 55.7 | 150.0 | 0.37 |
| 32/8 hd128 causal W4096 Q16 | 8 | 2, 32768 | 38.1 | 24.3 | 1.57 |
| 32/8 hd128 causal W4096 Q16 | 16 | 0, 8192 | 180.6 | 241.3 | 0.75 |

GB10, same sweep, hd128 and noncausal rows (not X925-pinned, so they are context only and no
routing rule rests on them; raw rows in `draft_window_dwr/gb_large_batch.jsonl`):

| geometry | B | pattern, top | CuTe us | our FA2 us | CuTe / FA2 |
|---|---|---|---|---|---|
| 32/8 hd128 causal full Q16 | 8 | 2, 32768 | 836.8 | 617.4 | 1.36 |
| 32/8 hd128 causal full Q16 | 16 | 2, 32768 | 823.7 | 1959.5 | 0.42 |
| 32/8 hd128 causal full Q16 | 64 | 2, 32768 | 1035.5 | 1809.4 | 0.57 |
| 32/8 hd128 causal full Q16 | 8 | 0, 8192 | 1168.1 | 1212.2 | 0.96 |
| 32/8 hd128 causal full Q16 | 64 | 0, 8192 | 9147.3 | 9606.3 | 0.95 |
| 32/8 hd128 noncausal W2048 Q8 | 8 | 2, 32768 | 40.1 | 41.3 | 0.97 |
| 32/8 hd128 noncausal W2048 Q8 | 64 | 2, 32768 | 198.9 | 186.3 | 1.07 |

GB10, hd256 rows behind the routing change, rerun X925-pinned at 8a3bf70f (the merge onto
the RC tip b4dce8a2): 60 interleaved pairs per row in random order, CUDA-graph replay of 10
prepare+run calls per sample, engine regime (page indices only where the provider reads them),
bootstrap 95% CI of the median ratio. Raw per-pair timings: `draft_window_dwr/gb_hd256_pinned.jsonl`.

| geometry | B | pattern, top | CuTe us | our FA2 us | CuTe / FA2 median | 95% CI |
|---|---|---|---|---|---|---|
| 16/4 hd256 causal full Q16 | 5 | 2, 32768 | 1210.7 | 617.1 | 1.919 | 1.907-2.024 |
| 16/4 hd256 causal full Q16 | 8 | 2, 32768 | 1176.3 | 623.8 | 1.886 | 1.883-1.889 |
| 16/4 hd256 causal full Q16 | 16 | 2, 32768 | 1184.1 | 632.3 | 1.870 | 1.864-1.874 |
| 16/4 hd256 causal full Q16 | 8 | 0, 8192 | 1231.2 | 1175.2 | 1.047 | 1.047-1.049 |
| 16/4 hd256 causal full Q16 | 16 | 0, 8192 | 2284.3 | 2272.1 | 1.006 | 1.005-1.006 |
| 16/4 hd256 causal full Q16 | 64 | 0, 512 | 667.2 | 762.6 | 0.874 | 0.873-0.876 |

The pinned rows confirm the unpinned sweep on the ragged rows (1.87-1.92x, CIs above 1) and
on uniform ctx 8192 (FA2 0.6-4.7% faster). One row goes the other way: uniform B=64 at
ctx 512, where CuTe is 13% faster on GB10 (on the RTX 5090 the same row favours FA2, 1.17x).
The rule keeps FA2 there: construction sees only B, not the length distribution, and a
ragged batch at the same B is expected to cost CuTe about 1.9x (extrapolated from the measured
ragged B=5/8/16 rows; B=64 ragged hd256 was not measured). The B=64 short uniform row is a disclosed GB10
loss until the length-aware CuTe split schedule lands.

Change: every hd256 CuTe rule (`dispatch.HD256_CUTE_ROWS`) now stops at B=4, its measured
band. Before, three rules had no upper bound on B. Past B=4, hd256 causal full rows use FA2:
of the six hd256 rows measured above B=4 on the RTX 5090, CuTe is 3.0-20x slower on four, 1.17x
slower on one and 0.97x on one. On GB10 (X925-pinned, above) it is 1.87-1.92x slower on the
ragged rows, 1.006-1.047x on uniform ctx 8192, and 0.87x on uniform B=64 ctx 512. The group-8 hd256 table (DF4,
`dispatch.HD256_GQA8_CUTE_ROWS`) replaces the 16/4 table for group 8 and was measured at B 1-4
only; its two RTX PRO 6000 rules had no upper bound on B (16/2 Q16 B5 selected CuTe), so they
now stop at B=4 too. `test_hd256_cute_rules_stop_at_b4` checks B 5-64 on every hardware key and
head layout, `test_hd256_b4_b5_boundary` the B=4/B=5 step.

hd128 and the windowed masks keep CuTe. There, uniform rows favour CuTe by 8-28%, ragged
B 5-16 favour FA2 by 1.6-5.3x, and at B=64 ragged CuTe wins, because our FA2 schedule
degrades too (1848 us vs 1019 us host-planned). The B=64 ragged statement rests on one
measured hd128 row (32/8 causal full, pattern 2) and one noncausal W2048 row; for hd256 and
the other masks it is extrapolated from the measured B=5/8/16 rows, not measured. Construction cannot see the length
distribution, so the fix is in the kernel: a runtime, length-aware split schedule in CuTe,
as FA2's device schedule has. This is open.

The same limit applies inside B <= 4. A randomized round after this change (RTX PRO 6000, 8 random
rows) had 16/4 hd128 causal full, B3 Q18, with one request at ctx 198,959 and two short ones.
CuTe took 750.5 us against SGLang's 294.4 us (0.39x): 4 splits per request, so the long
request runs on 48 CTAs. The same round's 24/4 hd256 B22 row, now on FA2, measured 1.018x
SGLang (a win). Its other losses: Q178 (FA2, 0.61x vs vLLM), B7 Q16 P4 causal W4096 (0.66x),
B1 Q16 full ctx 3439 (0.88x).

## Randomized precision verdicts (go-live round and the round at this head)

The go-live round marked three rows "precision failed"; a round at this head (RTX PRO 6000,
8 rows) marked six. Per seed, the records show that the gate fails the market kernels on the
same checks:
- Every row where our output fails also fails at least one market kernel (vLLM FA2 or SGLang
  FlashInfer) on the same seed and check.
- On three rows our output hash is bit-identical to a market kernel's on the failing seed
  (24/4 hd256 B3 ctx 98k and 12/2 hd256 B4 ctx 5k vs SGLang FlashInfer; 32/8 hd128 B4 Q7
  vs vLLM FA2), and both fail together.
- The remaining failures are max-ULP 1.057-1.221 against a best of 0.795-1.175. vLLM FA2
  shows the same max ULP on the same seeds.
Each candidate is judged against the best market value per check, so two different
correctly-rounding kernels fail each other. These verdicts come from the gate, not from our
kernels.

## D=64 and Q>32 (provider rows, CuTe vs our FA2 vs host-planned FlashInfer)

Probe: `benchmark/kernel_tuning/draft_provider_rows.py`. One construction per geometry and
provider, uniform runtime lengths L (keys per request, queries included), page size 1, CUDA graph
of 20 run() calls (prepare outside), median of 7 replays. "Host plan" is FlashInfer's
BatchPrefillWithPagedKVCacheWrapper planned on the host per length (SGLang's form). Precision
against an FP32 reference: exact = share of BF16 outputs equal to the rounded reference.
Raw rows: `draft_window_dwr/rtx5090_provider_rows.jsonl`, `draft_window_dwr/rtx_pro_6000_provider_rows.jsonl`
(`cute_env` marks the 64/128-row tile variants). Kernels as of 57fa4c5d.

D=64 (Q16; times in us, CuTe / our FA2 / host plan):

| device | heads, mask, B | L=128 | L=1000 | L=2245-4000 | L=16000 |
|---|---|---|---|---|---|
| RTX 5090 | 16/4 full B1 (C 262160) | 3.5 / 7.6 / 4.1 | | 6.3 / 9.6 / 6.5 (2245) | 13.0 / 14.7 / 14.6 |
| RTX 5090 | 16/4 causal B1 | 3.6 / 7.5 / 4.2 | 5.4 / 8.4 / 6.0 | 7.3 / 9.7 / 7.0 | |
| RTX 5090 | 16/4 full B4 | 4.7 / 11.1 / 4.2 | 9.8 / 13.2 / 8.1 | 15.3 / 17.4 / 15.2 | |
| RTX 5090 | 16/4 causal B8 | 5.0 / 12.3 / 4.5 | 10.0 / 15.6 / 12.5 | 21.6 / 33.1 / 24.1 | |
| RTX 5090 | 16/4 full B16 | 5.2 / 12.7 / 4.5 | 13.6 / 26.6 / 21.2 | 38.5 / 57.7 / 39.1 | |
| RTX 5090 | 32/8 causal B2 | 4.5 / 8.2 / 4.3 | 10.0 / 13.6 / 8.2 | 15.6 / 17.9 / 15.5 | |
| RTX PRO 6000 | 16/4 full B1 (C 262160) | 4.0 / 7.5 / 4.2 | | 7.2 / 9.6 / 7.1 (2245) | 13.5 / 14.8 / 14.6 |
| RTX PRO 6000 | 16/4 causal B1 | 4.0 / 7.4 / 4.2 | 6.5 / 8.4 / 6.7 | 8.2 / 9.6 / 7.4 | |
| RTX PRO 6000 | 16/4 full B4 | 4.3 / 9.8 / 4.3 | 11.9 / 12.2 / 8.8 | 16.3 / 15.4 / 15.1 | |

CuTe is faster than our FA2 device schedule on all 21 RTX 5090 rows (the table omits 16/4 causal B4, also a CuTe win) and 8 of 9 RTX PRO 6000 rows
(B4 full L4000: 1.06x). Exact share: CuTe 0.60-0.66, FA2 0.45-0.63 (BF16 split partials; equal to
the host plan's). D=64 now defaults to CuTe on every device (`FLASHINFER_HD64_ROWS` is empty). The
randomized row 16/4 d64 full B1 Q16 ctx 2245 was FA2 at 0.78x SGLang. Open: B4 L1000 is 1.22x (RTX 5090)
and 1.36x (RTX PRO 6000) the host plan.

GB10 (X925-pinned; `draft_window_dwr/gb10_provider_rows.jsonl`), D=64 Q16, CuTe / our FA2 / host plan:
16/4 full B1 (C 262160) L128 3.6 / 6.5 / 3.7, L2245 8.7 / 10.1 / 9.9, L16000 30.5 / 23.6 / 22.8;
16/4 causal B4 L128 4.6 / 9.1 / 4.0, L1000 12.3 / 19.1 / 17.9, L4000 24.6 / 36.9 / 28.1;
16/4 full B16 L128 5.9 / 11.6 / 6.2, L1000 24.5 / 45.1 / 23.6, L4000 303.8 / 312.2 / 285.8.
CuTe beats our FA2 on 8 of 9 GB10 rows; the exception is B1 at 16000 keys (1.29x), disclosed:
a construction-time rule cannot separate it from the B1 rows CuTe wins (L128 0.56x, L2245 0.86x).

Q>32, D=128 (times in us, CuTe default tiles / our FA2 / host plan):

| device | heads, mask, B, Q | L=500 | L=2398-4000 | L=16000 |
|---|---|---|---|---|
| RTX 5090 | 16/4 full B1 Q50 (C 131122) | 8.2 / 23.9 / 12.6 | 14.4 / 21.1 / 14.1 (2398) | 56.5 / 49.3 / 49.5 |
| RTX 5090 | 32/8 causal B1 Q50 | 10.0 / 20.5 / 13.5 | 32.2 / 30.6 / 30.5 | |
| RTX 5090 | 16/4 full B1 Q147 | 11.0 / 21.9 / 14.2 | 43.8 / 37.7 / 37.5 | |
| RTX 5090 | 16/4 full B1 Q33 | 7.4 / 18.6 / 12.1 | 14.8 / 21.1 / 20.3 | 42.5 / 54.8 / 47.1 |
| RTX 5090 | 16/4 full B1 Q48 | 7.9 / 19.3 / 12.3 | 16.6 / 27.1 / 21.2 | 48.6 / 48.4 / 47.8 |
| RTX 5090 | 16/4 causal B4 Q33 | 11.1 / 21.0 / 14.0 | 43.7 / 52.9 / 50.8 | |
| RTX 5090 | 32/8 full B4 Q33 | 12.2 / 23.9 / 23.6 | 77.1 / 91.8 / 91.6 | |
| RTX 5090 | 32/8 causal B2 Q40 | 11.1 / 21.6 / 14.6 | 43.3 / 51.9 / 51.7 | 160.5 / 175.8 / 176.5 |
| RTX 5090 | 32/8 full B4 Q50 | 23.1 / 27.7 / 27.1 | 155.5 / 97.0 / 96.2 | |
| RTX 5090 | 32/8 full B4 Q147 | 45.8 / 58.3 / 58.1 | 306.1 / 237.9 / 238.3 | |
| RTX PRO 6000 | 16/4 full B1 Q50 (C 131122) | 9.3 / 17.0 / 12.1 | 15.1 / 18.4 / 13.2 (2398) | 52.8 / 36.7 / 36.4 |
| RTX PRO 6000 | 32/8 full B4 Q50 | 22.4 / 26.3 / 25.2 | 143.9 / 73.1 / 73.5 | |

- Precision: CuTe exact 0.59-0.66 on every row, FA2 0.45-0.53 (bit-identical to the host plan).
  The randomized row 16/4 d128 full B1 Q50 ctx 2398 failed precision because SGLang FlashInfer (which our
  FA2 equals) is dominated there by vLLM's unsplit FA2; CuTe's FP32 partials are more exact than both.
- Rule (`CUTE_LONG_QUERY_ROWS`), D=128 full/causal rows with Q>32:
  - B=1 takes CuTe on the RTX 5090 and RTX PRO 6000. Disclosed cost: past ~200 packed rows
    (Q*G) at 4k-16k keys CuTe is 1.05-1.44x our FA2 (RTX PRO 6000 B1 Q50 ctx 16000: 52.8 vs
    36.7 us; RTX 5090 B1 Q147 ctx 4000: 1.16x). Up to 192 packed rows it wins or ties at every
    measured length (RTX 5090 B1 Q48 ctx 16000: 48.6 vs 48.4 us).
  - B>=2 stays on FA2 on every device. The B=2-4 rows above (CuTe faster up to Q*G = 192 on the
    RTX 5090; at 200 packed rows, 32/8 B4 Q50 ctx 4000, CuTe 1.6x on the RTX 5090 and 1.97x on
    the RTX PRO 6000) are recorded only: a shape-specific speed rule for them is out of the
    current scope (no per-shape speed tuning), so they are a reported, unchased gap.
  - Sliding windows, D=64/256 with Q>32, and GB10 stay on FA2. GB10 (X925-pinned): 16/4 full B1 Q50
    CuTe / FA2 / host plan L500 11.8 / 18.3 / 13.7, L2398 38.3 / 26.1 / 25.7, L16000 285.7 /
    155.6 / 162.8, so B=1 stays on FA2 there; 16/4 causal B4 Q33 L500 15.1 / 29.5 / 29.3, L4000
    148.2 / 178.4 / 172.2.
- Larger row tiles do not close the long-context gap: RTX PRO 6000 B1 Q50 ctx 16000 is 47.9 us at
  64 rows and 47.4 us at 128 rows (FA2 36.7); 128 rows win only at 32/8 B4 Q147 ctx 4000 (163.7 vs
  167.5 us PRO, 214.5 vs 234.1 us RTX 5090) and lose at short contexts. The Q>32 long-context gap
  is a mainloop-efficiency gap for later CuTe work (IKET first), not a tile-count choice.
- Our FA2 run is 3-11 us slower than the host plan at short lengths (graph-mode padded grid and
  the always-on merge kernel); this is the remaining FA2-path loss to SGLang.

## Deferred and open

- RTX PRO 6000: every DW-R correctness and perf leg after the A4 file (the device was not
  available).
- K-Q7 group-8 rows above: a kernel change (128-row tiles for Q*G >= 128) then a randomized
  round.
- Large ragged batches with hd128 or windowed masks: a runtime, length-aware CuTe split schedule
  (see "Large batches"). The FA2 device schedule's B=64 ragged loss to host-planned FlashInfer
  (1.8x) is open too.
