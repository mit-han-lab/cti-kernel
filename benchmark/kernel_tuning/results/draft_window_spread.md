<!-- SPDX-License-Identifier: MIT -->
# DraftAttention: length-aware CuTe split grid (spread)

Design: `docs/dflash_kernels.md`, section "Length-aware split grid (spread)".
Code: `python/ctikernels/attention/_draft/cute/` (`spread_grid`, `Attention.spread_schedule`).
Kernel commit c9a02f7a; the receipts below ran the tree at be7b8e1a, which has the same
kernel (later commits change only tests, docs and receipts).

## Problem

A per-request grid `(s, Hkv * row_tiles, B)` fixes the split count s per request at
construction. In a ragged batch the long request is walked by s CTAs per row tile while
the short ones finish at once. Examples: the randomized row 32/8 Q18 B3 at a 199k context was
0.39x SGLang, and ragged hd256 B5/8/16 rows were 1.9-10.6x FA2's time. FlashInfer
plans per-request chunks from the runtime lengths. CuTe cannot plan on the host
(graph capture), so the kernel now computes its own schedule from the CSR on the device.

## Ragged and uniform rows

Harness: `benchmark/kernel_tuning/draft_spread_ragged.py all 60`. Each pair replays
three graphs of 10 `run()` calls, alternating the arm order. Arms: `off` = per-request
grid (CTI_DRAFT_SPREAD=0), `on` = default selection, FA2 = provider `flashinfer`
(our device schedule). Ratios are time/`on`, so values above 1 favour `on`. The CI is a
bootstrap 95% CI of the median ratio over 60 pairs. GB10 is X925-pinned.
Raw pairs: `draft_window_spread/ragged_<device>.jsonl` (type `pair`).
`spread/S`: spread B (0 = per-request grid) and CTAs per row tile.

### RTX 5090

| row | B | Q | Hq/Hkv | D | mask | C | contexts | off us | on us | FA2 us | off/on [95% CI] | FA2/on [95% CI] | spread/S |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| b3q18_199k | 3 | 18 | 32/8 | 128 | full | 241957 | 199000, 300, 50 | 1841.71 | 742.79 | 525.4 | 2.479 [2.478, 2.481] | 0.707 [0.707, 0.708] | 3/7 |
| b3q18_199k_u | 3 | 18 | 32/8 | 128 | full | 241957 | 199000 x3 | 1987.3 | 1979.91 | 1576.51 | 1.004 [1.003, 1.004] | 0.796 [0.795, 0.796] | 3/7 |
| b2q16_full_ragged32k | 2 | 16 | 32/8 | 128 | full | 32768 | 32752, 100 | 128.62 | 91.66 | 96.09 | 1.402 [1.401, 1.404] | 1.049 [1.048, 1.050] | 2/10 |
| b4q16_hd256_ragged32k | 4 | 16 | 16/4 | 256 | causal full | 32768 | 32752, 100, 40, 1000 | 241.25 | 97.43 | 98.6 | 2.476 [2.473, 2.480] | 1.011 [1.010, 1.013] | 4/21 |
| b8q16_ragged32k | 8 | 16 | 16/4 | 128 | causal full | 32768 | 32752, 200, 200, 200, 200, 200, 200, 200 | 303.52 | 50.81 | 54.7 | 5.979 [5.959, 5.998] | 1.072 [1.067, 1.076] | 8/21 |
| b5q16_hd256_ragged32k | 5 | 16 | 16/4 | 256 | causal full | 32768 | 32752, 50, 50, 50, 50 | 297.94 | 94.09 | 97.82 | 3.166 [3.162, 3.171] | 1.039 [1.038, 1.042] | 5/21 |
| b4q16_full_ragged131k | 4 | 16 | 32/8 | 128 | full | 131072 | 131056, 2000, 300, 30 | 1209.59 | 362.57 | 362.05 | 3.335 [3.333, 3.336] | 0.998 [0.997, 0.999] | 4/10 |
| u_b2q16_full_c32k_100 | 2 | 16 | 32/8 | 128 | full | 32768 | 100 x2 | 5.11 | 5.31 | 11.49 | 0.962 [0.958, 0.967] | 2.162 [2.158, 2.170] | 2/10 |
| u_b3q16_full_c32k_1000 | 3 | 16 | 32/8 | 128 | full | 32768 | 1000 x3 | 13.13 | 13.32 | 20.24 | 0.986 [0.984, 0.988] | 1.519 [1.514, 1.521] | 3/10 |
| u_b4q16_full_c32k_4000 | 4 | 16 | 32/8 | 128 | full | 32768 | 4000 x4 | 43.74 | 43.63 | 51.27 | 0.990 [0.985, 1.005] | 1.157 [1.153, 1.176] | 4/10 |
| u_b4q16_hd256_c32k_400 | 4 | 16 | 16/4 | 256 | causal full | 32768 | 400 x4 | 12.31 | 12.68 | 13.68 | 0.976 [0.972, 0.983] | 1.082 [1.077, 1.091] | 4/21 |
| u_b3q16_full_c32k_32k | 3 | 16 | 32/8 | 128 | full | 32768 | 32752 x3 | 254.22 | 253.54 | 266.15 | 1.002 [1.001, 1.003] | 1.047 [1.044, 1.050] | 3/10 |

### RTX PRO 6000 (30 pairs, subset; one call)

| row | B | Q | Hq/Hkv | D | mask | C | contexts | off us | on us | FA2 us | off/on [95% CI] | FA2/on [95% CI] | spread/S |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| b3q18_199k | 3 | 18 | 32/8 | 128 | full | 241957 | 199000, 300, 50 | 1731.5 | 702.22 | 561.11 | 2.464 [2.463, 2.467] | 0.799 [0.798, 0.799] | 3/7 |
| b2q16_full_ragged32k | 2 | 16 | 32/8 | 128 | full | 32768 | 32752, 100 | 120.57 | 71.98 | 72.74 | 1.675 [1.671, 1.680] | 1.003 [0.986, 1.030] | 2/11 |
| b4q16_hd256_ragged32k | 4 | 16 | 16/4 | 256 | causal full | 32768 | 32752, 100, 40, 1000 | 225.05 | 89.77 | 96.39 | 2.512 [2.505, 2.515] | 1.074 [1.062, 1.084] | 4/23 |
| b8q16_ragged32k | 8 | 16 | 16/4 | 128 | causal full | 32768 | 32752, 200, 200, 200, 200, 200, 200, 200 | 278.37 | 46.04 | 38.82 | 6.051 [6.038, 6.074] | 0.837 [0.811, 0.880] | 8/23 |
| u_b2q16_full_c32k_100 | 2 | 16 | 32/8 | 128 | full | 32768 | 100 x2 | 5.63 | 5.86 | 15.41 | 0.959 [0.954, 0.966] | 2.614 [2.601, 2.639] | 2/11 |
| u_b4q16_hd256_c32k_400 | 4 | 16 | 16/4 | 256 | causal full | 32768 | 400 x4 | 14.04 | 14.27 | 15.04 | 0.984 [0.980, 0.986] | 1.057 [1.052, 1.059] | 4/23 |
| u_b3q16_full_c32k_32k | 3 | 16 | 32/8 | 128 | full | 32768 | 32752 x3 | 278.5 | 277.21 | 287.45 | 1.005 [1.004, 1.006] | 1.037 [1.036, 1.038] | 3/11 |

### GB10

| row | B | Q | Hq/Hkv | D | mask | C | contexts | off us | on us | FA2 us | off/on [95% CI] | FA2/on [95% CI] | spread/S |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| b3q18_199k | 3 | 18 | 32/8 | 128 | full | 241957 | 199000, 300, 50 | 4718.56 | 4708.89 | 3432.73 | 1.003 [0.988, 1.006] | 0.730 [0.718, 0.735] | 0/1 |
| b3q18_199k_u | 3 | 18 | 32/8 | 128 | full | 241957 | 199000 x3 | 10192.68 | 10074.18 | 10199.46 | 1.008 [1.008, 1.012] | 1.012 [1.011, 1.013] | 0/1 |
| b2q16_full_ragged32k | 2 | 16 | 32/8 | 128 | full | 32768 | 32752, 100 | 795.5 | 603.45 | 586.36 | 1.320 [1.314, 1.325] | 0.970 [0.968, 0.974] | 2/3 |
| b4q16_hd256_ragged32k | 4 | 16 | 16/4 | 256 | causal full | 32768 | 32752, 100, 40, 1000 | 1173.48 | 620.51 | 612.94 | 1.885 [1.879, 1.892] | 0.988 [0.986, 0.990] | 4/6 |
| b8q16_ragged32k | 8 | 16 | 16/4 | 128 | causal full | 32768 | 32752, 200, 200, 200, 200, 200, 200, 200 | 748.81 | 340.78 | 320.8 | 2.164 [2.158, 2.225] | 0.942 [0.938, 0.944] | 8/12 |
| b5q16_hd256_ragged32k | 5 | 16 | 16/4 | 256 | causal full | 32768 | 32752, 50, 50, 50, 50 | 1169.36 | 776.29 | 598.04 | 1.508 [1.455, 1.515] | 0.771 [0.768, 0.773] | 5/6 |
| b4q16_full_ragged131k | 4 | 16 | 32/8 | 128 | full | 131072 | 131056, 2000, 300, 30 | 3136.93 | 2338.15 | 2335.76 | 1.361 [1.338, 1.373] | 1.004 [0.994, 1.024] | 4/6 |
| u_b2q16_full_c32k_100 | 2 | 16 | 32/8 | 128 | full | 32768 | 100 x2 | 5.64 | 5.83 | 14.66 | 0.968 [0.966, 0.970] | 2.515 [2.513, 2.517] | 2/3 |
| u_b3q16_full_c32k_1000 | 3 | 16 | 32/8 | 128 | full | 32768 | 1000 x3 | 28.0 | 27.62 | 36.09 | 0.912 [0.828, 1.022] | 1.278 [1.184, 1.337] | 3/6 |
| u_b4q16_full_c32k_4000 | 4 | 16 | 32/8 | 128 | full | 32768 | 4000 x4 | 290.54 | 288.83 | 308.28 | 1.007 [1.005, 1.009] | 1.065 [1.063, 1.067] | 4/6 |
| u_b4q16_hd256_c32k_400 | 4 | 16 | 16/4 | 256 | causal full | 32768 | 400 x4 | 18.54 | 18.17 | 22.95 | 0.979 [0.974, 1.015] | 1.183 [1.174, 1.280] | 4/6 |
| u_b3q16_full_c32k_32k | 3 | 16 | 32/8 | 128 | full | 32768 | 32752 x3 | 1770.32 | 1761.02 | 1724.82 | 1.004 [1.001, 1.007] | 0.994 [0.981, 0.999] | 3/6 |

## Uniform rows in the engine regime (RTX 5090)

Harness: `benchmark/kernel_tuning/draft_window_ab.py --arm on=. --arm off=.@CTI_DRAFT_SPREAD=0
--arm on2=. --masks full,q3moe --batches 2,3,4 --ctxs 128,512,4096 --pairs 60 --rounds 3`.
This uses per-layer graph chains over rotating K/V buffers (>= 2x L2) at C=32768. Speed is
off_us / on_us (below 1 = spread slower); on2 is an identical second arm (the A/A floor).
`full` is the Qwen3-8B Q16 full noncausal pair (32/8 hd128); `q3moe` is K-Q7 (32/4 Q16).
The DW1 sliding pairs use the legacy merge and never take the spread grid.
Raw pairs: `draft_window_spread/uniform_ab_rtx5090.jsonl`.

| mask | B | ctx | on/off speed [95% CI] | A/A on/on2 [95% CI] |
| --- | ---: | ---: | --- | --- |
| full | 2 | 128 | 0.974 [0.974, 0.975] | 1.000 [0.998, 1.001] |
| full | 2 | 512 | 0.990 [0.987, 0.992] | 0.992 [0.988, 1.004] |
| full | 2 | 4096 | 0.991 [0.990, 0.993] | 0.998 [0.995, 1.001] |
| full | 3 | 128 | 0.953 [0.953, 0.954] | 1.000 [1.000, 1.002] |
| full | 3 | 512 | 0.990 [0.988, 0.991] | 1.000 [0.999, 1.000] |
| full | 3 | 4096 | 0.999 [0.997, 1.000] | 1.000 [1.000, 1.002] |
| full | 4 | 128 | 0.962 [0.962, 0.962] | 1.001 [1.000, 1.002] |
| full | 4 | 512 | 0.990 [0.988, 0.991] | 1.000 [0.998, 1.000] |
| full | 4 | 4096 | 1.015 [1.012, 1.016] | 1.000 [0.999, 1.001] |
| q3moe | 2 | 128 | 0.972 [0.970, 0.973] | 1.001 [1.000, 1.002] |
| q3moe | 2 | 512 | 0.980 [0.969, 0.981] | 1.005 [1.000, 1.011] |
| q3moe | 2 | 4096 | 0.999 [0.998, 1.000] | 1.000 [0.999, 1.000] |
| q3moe | 3 | 128 | 0.970 [0.970, 0.970] | 1.005 [1.003, 1.005] |
| q3moe | 3 | 512 | 0.981 [0.979, 0.982] | 1.004 [1.001, 1.005] |
| q3moe | 3 | 4096 | 1.000 [0.999, 1.000] | 1.000 [1.000, 1.001] |
| q3moe | 4 | 128 | 0.967 [0.967, 0.968] | 1.000 [1.000, 1.000] |
| q3moe | 4 | 512 | 0.987 [0.985, 0.990] | 0.998 [0.995, 1.000] |
| q3moe | 4 | 4096 | 1.009 [1.007, 1.011] | 1.000 [1.000, 1.002] |

Strict rule against the per-request grid (CI_high >= 1): 4 of 18 rows pass. The cost is
the schedule prologue: 2.6-4.7% at ctx 128, 0.9-2.0% at 512, and at most 0.1% (some rows
gain up to 1.5%) at 4096. The A/A arm passes 17 of 18 (worst CI_high 0.9999). Runtime
lengths cannot choose the grid (graph capture), so this trades a 0.1-0.2 us prologue on
short uniform rows for the 1.4-6x ragged gains above. `CTI_DRAFT_SPREAD=0` restores the
per-request grid.

## Uniform rows against the faster market (spread on by default)

Condition: spread stays on for 2 <= B <= 32 unless a uniform row
falls below the faster of vLLM and SGLang because of its prologue. Harness:
`dflash_draft_layers_paired.py --candidate auto --queries 16 --capacities 32768 --batches 2,3,4
--contexts 100,400,1000,4000 --pairs 60 --calls 3` (engine regime: per-layer graphs over n
rotating K/V sets >= 2x L2; 20 self-pairs per comparator, then 60 alternating
candidate/comparator pairs). Comparators: vLLM 0.31.0 `vllm_flash_attn` (P=16) and SGLang
72730d6 FlashInfer 0.7.1 fa2 (P=1). Speed = comparator / ours per pair against the faster
comparator (above 1: ours faster); the CI is the bootstrap 95% CI of the median (2000
resamples). `gate` is the harness's own epsilon gate. Shapes: `draft_noncausal` (Qwen3-8B
Q16 full pair, 32/8 hd128), `q3_target` (32/8 hd128 causal full), `q35_target` (Qwen3.5
16/4 hd256 causal full), `q3moe_draft` (K-Q7, 32/4 hd128 causal). Every row passes the FP32
dense oracle (atol = rtol = 2e-2). GB10 is X925-pinned. Code: 96a0da76 (same kernel as c9a02f7a).
Raw rows: `draft_window_spread/market_uniform_<device>.jsonl` and `*_spread_off.jsonl`.

### RTX 5090 (spread on)

| shape | B | ctx | ours us | vLLM us | SGLang us | faster | speed vs faster [95% CI] | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q3_target | 2 | 100 | 5.22 | 5.89 | 11.63 | vLLM | 1.120 [1.117, 1.126] | PASS |
| q3_target | 2 | 400 | 9.28 | 16.58 | 19.23 | vLLM | 1.788 [1.786, 1.788] | PASS |
| q3_target | 2 | 1000 | 12.04 | 31.15 | 20.01 | SGLang | 1.661 [1.659, 1.662] | PASS |
| q3_target | 2 | 4000 | 26.64 | 116.63 | 31.12 | SGLang | 1.167 [1.167, 1.168] | PASS |
| q3_target | 3 | 100 | 5.26 | 6.00 | 18.43 | vLLM | 1.128 [1.126, 1.139] | PASS |
| q3_target | 3 | 400 | 10.81 | 16.59 | 20.17 | vLLM | 1.536 [1.535, 1.537] | PASS |
| q3_target | 3 | 1000 | 15.11 | 31.47 | 21.42 | SGLang | 1.418 [1.416, 1.418] | PASS |
| q3_target | 3 | 4000 | 37.81 | 117.08 | 41.23 | SGLang | 1.090 [1.090, 1.091] | PASS |
| q3_target | 4 | 100 | 5.32 | 6.22 | 18.76 | vLLM | 1.165 [1.163, 1.171] | PASS |
| q3_target | 4 | 400 | 12.06 | 16.72 | 21.18 | vLLM | 1.386 [1.384, 1.388] | PASS |
| q3_target | 4 | 1000 | 17.37 | 31.70 | 22.65 | SGLang | 1.304 [1.303, 1.304] | PASS |
| q3_target | 4 | 4000 | 48.17 | 116.98 | 54.50 | SGLang | 1.132 [1.132, 1.133] | PASS |
| draft_noncausal | 2 | 100 | 5.25 | 5.88 | 11.66 | vLLM | 1.114 [1.111, 1.120] | PASS |
| draft_noncausal | 2 | 400 | 9.45 | 16.58 | 19.32 | vLLM | 1.756 [1.753, 1.758] | PASS |
| draft_noncausal | 2 | 1000 | 12.19 | 31.42 | 20.22 | SGLang | 1.658 [1.658, 1.660] | PASS |
| draft_noncausal | 2 | 4000 | 27.17 | 118.66 | 31.82 | SGLang | 1.169 [1.168, 1.171] | PASS |
| draft_noncausal | 3 | 100 | 5.30 | 6.02 | 18.46 | vLLM | 1.135 [1.123, 1.135] | PASS |
| draft_noncausal | 3 | 400 | 10.94 | 16.60 | 20.36 | vLLM | 1.516 [1.515, 1.518] | PASS |
| draft_noncausal | 3 | 1000 | 15.33 | 31.76 | 21.41 | SGLang | 1.395 [1.394, 1.397] | PASS |
| draft_noncausal | 3 | 4000 | 38.59 | 119.16 | 41.77 | SGLang | 1.083 [1.081, 1.084] | PASS |
| draft_noncausal | 4 | 100 | 5.36 | 6.24 | 18.82 | vLLM | 1.159 [1.153, 1.161] | PASS |
| draft_noncausal | 4 | 400 | 12.20 | 16.70 | 21.27 | vLLM | 1.370 [1.367, 1.372] | PASS |
| draft_noncausal | 4 | 1000 | 17.69 | 32.01 | 22.70 | SGLang | 1.284 [1.283, 1.285] | PASS |
| draft_noncausal | 4 | 4000 | 49.02 | 119.40 | 54.90 | SGLang | 1.120 [1.119, 1.121] | PASS |
| q35_target | 2 | 100 | 8.48 | 9.63 | 18.68 | vLLM | 1.135 [1.133, 1.136] | PASS |
| q35_target | 2 | 400 | 11.81 | 27.34 | 19.86 | SGLang | 1.682 [1.681, 1.682] | PASS |
| q35_target | 2 | 1000 | 13.62 | 59.15 | 20.59 | SGLang | 1.512 [1.511, 1.513] | PASS |
| q35_target | 2 | 4000 | 28.90 | 224.37 | 28.25 | SGLang | 0.979 [0.977, 0.980] | PASS |
| q35_target | 3 | 100 | 8.52 | 9.73 | 18.87 | vLLM | 1.141 [1.140, 1.142] | PASS |
| q35_target | 3 | 400 | 12.84 | 27.56 | 20.37 | SGLang | 1.586 [1.585, 1.587] | PASS |
| q35_target | 3 | 1000 | 17.10 | 59.34 | 21.11 | SGLang | 1.234 [1.233, 1.235] | PASS |
| q35_target | 3 | 4000 | 39.39 | 224.90 | 39.60 | SGLang | 1.006 [1.005, 1.007] | PASS |
| q35_target | 4 | 100 | 8.57 | 9.71 | 19.24 | vLLM | 1.133 [1.132, 1.133] | PASS |
| q35_target | 4 | 400 | 13.82 | 27.53 | 20.99 | SGLang | 1.519 [1.518, 1.520] | PASS |
| q35_target | 4 | 1000 | 19.56 | 59.43 | 21.88 | SGLang | 1.119 [1.116, 1.120] | PASS |
| q35_target | 4 | 4000 | 49.16 | 225.32 | 50.06 | SGLang | 1.019 [1.018, 1.020] | PASS |
| q3moe_draft | 2 | 100 | 5.27 | 5.90 | 18.87 | vLLM | 1.116 [1.112, 1.120] | PASS |
| q3moe_draft | 2 | 400 | 8.97 | 16.79 | 20.29 | vLLM | 1.871 [1.871, 1.872] | PASS |
| q3moe_draft | 2 | 1000 | 11.51 | 31.69 | 20.96 | SGLang | 1.821 [1.820, 1.822] | PASS |
| q3moe_draft | 2 | 4000 | 22.59 | 118.59 | 24.46 | SGLang | 1.083 [1.082, 1.084] | PASS |
| q3moe_draft | 3 | 100 | 5.35 | 6.05 | 19.43 | vLLM | 1.125 [1.122, 1.130] | PASS |
| q3moe_draft | 3 | 400 | 10.46 | 16.83 | 21.01 | vLLM | 1.610 [1.609, 1.610] | PASS |
| q3moe_draft | 3 | 1000 | 14.18 | 32.01 | 21.88 | SGLang | 1.542 [1.541, 1.543] | PASS |
| q3moe_draft | 3 | 4000 | 33.26 | 119.28 | 30.78 | SGLang | 0.925 [0.924, 0.926] | FAIL |
| q3moe_draft | 4 | 100 | 5.38 | 6.23 | 19.71 | vLLM | 1.159 [1.157, 1.159] | PASS |
| q3moe_draft | 4 | 400 | 11.47 | 16.88 | 22.05 | vLLM | 1.472 [1.471, 1.473] | PASS |
| q3moe_draft | 4 | 1000 | 16.55 | 32.39 | 22.73 | SGLang | 1.374 [1.373, 1.375] | PASS |
| q3moe_draft | 4 | 4000 | 45.71 | 119.95 | 34.60 | SGLang | 0.757 [0.757, 0.758] | FAIL |

Spread off (CTI_DRAFT_SPREAD=0), the ctx-4000 rows, separate processes:

| shape | B | ctx | ours us | vLLM us | SGLang us | faster | speed vs faster [95% CI] | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q35_target | 2 | 4000 | 28.46 | 218.31 | 28.51 | SGLang | 1.002 [1.001, 1.003] | PASS |
| q35_target | 3 | 4000 | 39.07 | 219.16 | 38.88 | SGLang | 0.994 [0.994, 0.995] | PASS |
| q35_target | 4 | 4000 | 48.67 | 219.52 | 50.07 | SGLang | 1.028 [1.028, 1.029] | PASS |
| q3moe_draft | 2 | 4000 | 21.95 | 115.56 | 24.00 | SGLang | 1.094 [1.093, 1.095] | PASS |
| q3moe_draft | 3 | 4000 | 32.33 | 115.93 | 30.03 | SGLang | 0.929 [0.928, 0.930] | FAIL |
| q3moe_draft | 4 | 4000 | 44.52 | 116.35 | 33.67 | SGLang | 0.756 [0.755, 0.757] | FAIL |

### GB10 (spread on)

| shape | B | ctx | ours us | vLLM us | SGLang us | faster | speed vs faster [95% CI] | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_noncausal | 2 | 100 | 6.40 | 10.09 | 14.49 | vLLM | 1.525 [1.521, 1.539] | PASS |
| draft_noncausal | 2 | 400 | 20.54 | 27.55 | 26.55 | SGLang | 1.303 [1.287, 1.311] | PASS |
| draft_noncausal | 2 | 1000 | 43.11 | 54.14 | 48.01 | SGLang | 1.117 [1.114, 1.120] | PASS |
| draft_noncausal | 2 | 4000 | 153.57 | 200.27 | 158.80 | SGLang | 1.033 [1.030, 1.035] | PASS |
| draft_noncausal | 3 | 100 | 8.59 | 10.97 | 15.36 | vLLM | 1.213 [1.199, 1.242] | PASS |
| draft_noncausal | 3 | 400 | 28.48 | 31.07 | 36.26 | vLLM | 1.094 [1.091, 1.100] | PASS |
| draft_noncausal | 3 | 1000 | 65.42 | 63.24 | 68.40 | vLLM | 0.970 [0.965, 0.972] | PASS |
| draft_noncausal | 3 | 4000 | 234.31 | 233.21 | 233.59 | vLLM | 0.998 [0.995, 1.001] | PASS |
| draft_noncausal | 4 | 100 | 10.00 | 15.47 | 16.62 | vLLM | 1.507 [1.502, 1.540] | PASS |
| draft_noncausal | 4 | 400 | 36.25 | 42.11 | 47.24 | vLLM | 1.166 [1.156, 1.176] | PASS |
| draft_noncausal | 4 | 1000 | 78.17 | 85.31 | 89.54 | vLLM | 1.094 [1.090, 1.098] | PASS |
| draft_noncausal | 4 | 4000 | 290.26 | 334.14 | 306.41 | SGLang | 1.059 [1.052, 1.061] | PASS |
| q3moe_draft | 2 | 100 | 5.78 | 9.62 | 15.35 | vLLM | 1.664 [1.659, 1.670] | PASS |
| q3moe_draft | 2 | 400 | 14.14 | 25.23 | 21.63 | SGLang | 1.531 [1.528, 1.534] | PASS |
| q3moe_draft | 2 | 1000 | 28.44 | 48.09 | 30.62 | SGLang | 1.075 [1.072, 1.080] | PASS |
| q3moe_draft | 2 | 4000 | 99.46 | 174.89 | 88.03 | SGLang | 0.888 [0.883, 0.891] | FAIL |
| q3moe_draft | 3 | 100 | 7.78 | 10.22 | 15.65 | vLLM | 1.299 [1.295, 1.314] | PASS |
| q3moe_draft | 3 | 400 | 18.53 | 26.19 | 26.66 | vLLM | 1.418 [1.412, 1.420] | PASS |
| q3moe_draft | 3 | 1000 | 39.65 | 49.60 | 43.58 | SGLang | 1.100 [1.097, 1.106] | PASS |
| q3moe_draft | 3 | 4000 | 132.99 | 180.86 | 133.79 | SGLang | 1.009 [1.003, 1.012] | PASS |
| q3moe_draft | 4 | 100 | 8.05 | 14.48 | 16.07 | vLLM | 1.803 [1.799, 1.820] | PASS |
| q3moe_draft | 4 | 400 | 22.24 | 38.41 | 32.31 | SGLang | 1.456 [1.450, 1.458] | PASS |
| q3moe_draft | 4 | 1000 | 45.79 | 73.99 | 55.69 | SGLang | 1.217 [1.214, 1.219] | PASS |
| q3moe_draft | 4 | 4000 | 163.63 | 267.28 | 175.30 | SGLang | 1.072 [1.065, 1.076] | PASS |
| q3_target | 2 | 100 | 8.22 | 11.13 | 15.12 | vLLM | 1.353 [1.346, 1.357] | PASS |
| q3_target | 2 | 400 | 19.95 | 27.63 | 26.62 | SGLang | 1.334 [1.331, 1.337] | PASS |
| q3_target | 2 | 1000 | 41.66 | 53.48 | 48.11 | SGLang | 1.155 [1.153, 1.157] | PASS |
| q3_target | 2 | 4000 | 157.26 | 207.65 | 164.96 | SGLang | 1.041 [1.033, 1.055] | PASS |
| q3_target | 3 | 100 | 11.73 | 12.81 | 16.39 | vLLM | 1.102 [1.095, 1.105] | PASS |
| q3_target | 3 | 400 | 28.43 | 31.01 | 36.47 | vLLM | 1.093 [1.088, 1.097] | PASS |
| q3_target | 3 | 1000 | 64.96 | 62.31 | 67.97 | vLLM | 0.959 [0.957, 0.964] | PASS |
| q3_target | 3 | 4000 | 238.19 | 240.78 | 241.20 | vLLM | 1.008 [1.001, 1.015] | PASS |
| q3_target | 4 | 100 | 13.44 | 17.30 | 17.88 | vLLM | 1.292 [1.288, 1.295] | PASS |
| q3_target | 4 | 400 | 35.33 | 42.91 | 47.68 | vLLM | 1.196 [1.190, 1.223] | PASS |
| q3_target | 4 | 1000 | 77.48 | 89.30 | 92.11 | vLLM | 1.117 [1.099, 1.148] | PASS |
| q3_target | 4 | 4000 | 297.39 | 326.75 | 315.22 | SGLang | 1.059 [1.055, 1.068] | PASS |
| q35_target | 2 | 100 | 9.12 | 8.76 | 15.31 | vLLM | 0.952 [0.949, 0.954] | PASS |
| q35_target | 2 | 400 | 24.52 | 23.74 | 23.06 | SGLang | 0.940 [0.938, 0.944] | PASS |
| q35_target | 2 | 1000 | 45.43 | 48.60 | 44.45 | SGLang | 0.982 [0.978, 0.984] | PASS |
| q35_target | 2 | 4000 | 155.14 | 186.45 | 163.78 | SGLang | 1.053 [1.051, 1.055] | PASS |
| q35_target | 3 | 100 | 9.92 | 9.75 | 15.79 | vLLM | 0.963 [0.959, 0.965] | PASS |
| q35_target | 3 | 400 | 31.82 | 28.46 | 30.94 | vLLM | 0.897 [0.889, 0.909] | FAIL |
| q35_target | 3 | 1000 | 61.67 | 60.70 | 64.00 | vLLM | 0.986 [0.983, 0.988] | PASS |
| q35_target | 3 | 4000 | 224.71 | 231.05 | 236.14 | vLLM | 1.032 [1.029, 1.035] | PASS |
| q35_target | 4 | 100 | 10.04 | 16.34 | 16.47 | vLLM | 1.631 [1.602, 1.636] | PASS |
| q35_target | 4 | 400 | 34.77 | 45.78 | 39.27 | SGLang | 1.133 [1.125, 1.134] | PASS |
| q35_target | 4 | 1000 | 75.84 | 98.36 | 86.72 | SGLang | 1.144 [1.142, 1.147] | PASS |
| q35_target | 4 | 4000 | 283.33 | 391.80 | 307.87 | SGLang | 1.088 [1.080, 1.091] | PASS |

Spread off, the B=3 hd128 rows:

| shape | B | ctx | ours us | vLLM us | SGLang us | faster | speed vs faster [95% CI] | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| draft_noncausal | 3 | 100 | 8.69 | 11.20 | 15.54 | vLLM | 1.235 [1.214, 1.278] | PASS |
| draft_noncausal | 3 | 400 | 28.19 | 30.97 | 36.57 | vLLM | 1.102 [1.098, 1.105] | PASS |
| draft_noncausal | 3 | 1000 | 65.76 | 62.98 | 69.41 | vLLM | 0.960 [0.955, 0.963] | PASS |
| draft_noncausal | 3 | 4000 | 235.70 | 235.10 | 237.16 | vLLM | 0.997 [0.994, 1.040] | PASS |
| q3_target | 3 | 100 | 11.36 | 12.97 | 16.49 | vLLM | 1.146 [1.142, 1.148] | PASS |
| q3_target | 3 | 400 | 28.10 | 31.12 | 36.76 | vLLM | 1.109 [1.105, 1.110] | PASS |
| q3_target | 3 | 1000 | 65.19 | 63.27 | 71.63 | vLLM | 0.966 [0.953, 0.972] | PASS |
| q3_target | 3 | 4000 | 238.27 | 236.68 | 244.48 | vLLM | 0.993 [0.984, 1.001] | PASS |

### In-process spread on/off, Qwen3.5 hd256 (RTX 5090)

`draft_window_ab.py --arm on=. --arm off=.@CTI_DRAFT_SPREAD=0 --arm on2=. --masks cfull
--heads 16,4,256 --batches 2,3,4 --ctxs 128,512,1000,4096` (60 pairs x 3 rounds, per-layer
graph chains; on2 is the A/A arm). Raw: `draft_window_spread/uniform_ab_hd256_rtx5090.jsonl`.

| mask | heads | B | ctx | on us | off us | on/off speed [95% CI] | A/A on/on2 [95% CI] |
|---|---|---|---|---|---|---|---|
| cfull | 16/4/256 | 2 | 128 | 9.70 | 9.44 | 0.973 [0.972, 0.974] | 0.996 [0.996, 0.996] |
| cfull | 16/4/256 | 2 | 512 | 11.82 | 11.56 | 0.981 [0.978, 0.983] | 0.996 [0.995, 0.999] |
| cfull | 16/4/256 | 2 | 1000 | 13.54 | 13.31 | 0.982 [0.981, 0.986] | 0.994 [0.993, 0.997] |
| cfull | 16/4/256 | 2 | 4096 | 29.36 | 29.20 | 0.994 [0.992, 0.996] | 1.000 [0.998, 1.000] |
| cfull | 16/4/256 | 3 | 128 | 9.80 | 9.44 | 0.963 [0.963, 0.964] | 1.000 [1.000, 1.001] |
| cfull | 16/4/256 | 3 | 512 | 13.84 | 13.47 | 0.974 [0.973, 0.975] | 1.001 [0.999, 1.001] |
| cfull | 16/4/256 | 3 | 1000 | 17.06 | 16.86 | 0.989 [0.988, 0.990] | 1.002 [1.000, 1.004] |
| cfull | 16/4/256 | 3 | 4096 | 39.74 | 39.73 | 0.999 [0.999, 1.000] | 1.000 [0.999, 1.002] |
| cfull | 16/4/256 | 4 | 128 | 9.75 | 9.43 | 0.967 [0.966, 0.968] | 1.002 [1.002, 1.003] |
| cfull | 16/4/256 | 4 | 512 | 14.81 | 14.55 | 0.983 [0.982, 0.984] | 1.000 [1.000, 1.001] |
| cfull | 16/4/256 | 4 | 1000 | 19.43 | 19.30 | 0.993 [0.991, 0.994] | 1.001 [1.000, 1.002] |
| cfull | 16/4/256 | 4 | 4096 | 49.71 | 49.59 | 0.998 [0.997, 1.000] | 1.000 [0.999, 1.003] |

### Reading

- **Rows below the faster market (CI_low <= 1):** 3 of 48 on the RTX 5090, 10 of 48 on GB10.
- **Not caused by spread:**
  - RTX 5090 K-Q7 B3/B4 at ctx 4000 (0.925/0.757 vs SGLang). With spread off they are the
    same (0.929/0.756). Cause: the 32-row tile at Q*G = 128 (work item b).
  - GB10 hd128 B=3 at ctx 1000/4000 (`draft_noncausal`, `q3_target`, 0.959-1.008). With
    spread off they are 0.960-0.997, the same within noise. At 32/8, GB10 B=3 has 48 CTAs on
    48 SMs either way. The DW-R per-layer table had these rows at 1.004-1.007x vLLM's time
    before spread.
- **Partly spread:** RTX 5090 `q35_target` B2 at ctx 4000 is 0.979 [0.977, 0.980] vs SGLang
  with spread on and 1.002 [1.001, 1.003] with spread off (separate processes). The
  in-process A/B puts spread's cost on that row at 0.6% (0.994 [0.992, 0.996]); the rest is
  process placement. It ties SGLang within about 2% in either mode.
- **GB10 rows attributed (spread off, X925-pinned, 60 pairs, code 5c61361a):** the same
  rows with `CTI_DRAFT_SPREAD=0` land within noise of spread on, in both directions, so
  spread does not cause them. They are base-kernel gaps at these shapes.

  | shape | B | ctx | spread on | spread off |
  |---|---|---|---|---|
  | q35_target | 2 | 100 | 0.952 [0.949, 0.954] | 0.967 [0.957, 0.981] |
  | q35_target | 2 | 400 | 0.940 [0.938, 0.944] | 0.925 [0.922, 0.927] |
  | q35_target | 2 | 1000 | 0.982 [0.978, 0.984] | 0.989 [0.986, 0.992] |
  | q35_target | 3 | 100 | 0.963 [0.959, 0.965] | 0.973 [0.969, 0.980] |
  | q35_target | 3 | 400 | 0.897 [0.889, 0.909] | 0.913 [0.906, 0.918] |
  | q35_target | 3 | 1000 | 0.986 [0.983, 0.988] | 0.970 [0.965, 0.973] |
  | q3moe_draft | 2 | 4000 | 0.888 [0.883, 0.891] | 0.863 [0.860, 0.867] |

  Speed vs the faster of vLLM 0.31 / SGLang (ratio > 1 = ours faster). The `q3moe_draft`
  row is the 32-row tile at Q*G = 128 (work item b); the hd256 Qwen3.5 rows at ctx <= 1000
  are the CuTe hd256 kernel itself. Raw rows: the second header block of
  `draft_window_spread/market_uniform_gb10_spread_off.jsonl`.
- **No rule added.** A rule from host-known inputs (B, configured capacity, head dim, device)
  that turns spread off for the RTX 5090 hd256 B=2 row would also turn it off for ragged
  hd256 batches with the same construction. There, spread is 2.5-3.2x faster (rows above).
  `CTI_DRAFT_SPREAD=0` stays the override.

## hd256 past the B<=4 CuTe cap (RTX 5090)

`draft_spread_ragged.py` rows `cap_*` (60 pairs; one K/V set, graph of 10 calls after one
prepare). FA2/CuTe above 1 favours CuTe with spread.
Raw: `draft_window_spread/hd256_large_batch_rtx5090.jsonl`.

| row | B | heads | contexts | CuTe spread us | FA2 us | FA2/CuTe [95% CI] | spread/S |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| cap_b8_hd256_u512 | 8 | 16/4 | 512 x8 | 18.40 | 17.08 | 0.932 [0.928, 0.935] | 8/21 |
| cap_b16_hd256_u512 | 16 | 16/4 | 512 x16 | 23.72 | 25.18 | 1.008 [0.997, 1.072] | 16/21 |
| cap_b32_hd256_u512 | 32 | 16/4 | 512 x32 | 46.04 | 44.11 | 0.940 [0.936, 0.979] | 0/1 |
| cap_b8_hd256_u8k | 8 | 16/4 | 8192 x8 | 176.50 | 183.13 | 1.036 [1.035, 1.038] | 8/21 |
| cap_b16_hd256_u8k | 16 | 16/4 | 8192 x16 | 333.07 | 347.90 | 1.044 [1.043, 1.045] | 16/21 |
| cap_b32_hd256_u2k | 32 | 16/4 | 2048 x32 | 191.56 | 189.41 | 0.988 [0.987, 0.989] | 0/1 |
| cap_b8_hd256_82_u4k | 8 | 8/2 | 4096 x8 | 39.94 | 35.09 | 0.810 [0.804, 0.881] | 8/42 |
| cap_b8_hd256_ragged | 8 | 16/4 | 32752 + 200 x7 | 103.03 | 100.30 | 0.973 [0.972, 0.975] | 8/21 |
| cap_b16_hd256_ragged | 16 | 16/4 | 32752 + 200 x15 | 212.06 | 107.34 | 0.505 [0.505, 0.506] | 16/21 |
| cap_b32_hd256_ragged | 32 | 16/4 | 32752 + 200 x31 | 1181.39 | 118.28 | 0.100 [0.100, 0.100] | 0/1 |
| cap_b16_hd256_mixed | 16 | 16/4 | mixed 40-20000 | 446.48 | 195.86 | 0.438 [0.438, 0.439] | 16/21 |
| b5q16_hd256_ragged32k | 5 | 16/4 | 32752 + 50 x4 | 94.96 | 98.41 | 1.037 [1.035, 1.039] | 5/21 |
| cap_b16_hd256_82_ragged | 16 | 8/2 | 32752 + 200 x15 | 55.18 | 39.59 | 0.714 [0.670, 0.717] | 16/42 |

Verdict: the cap stays on the RTX 5090. Past B=5, FA2 is faster on:
- the ragged rows (16/4 B8 0.97x, B16 0.51x, mixed B16 0.44x, B32 0.10x; 8/2 B16 0.71x);
- short uniform rows (16/4 B8 and B32 at ctx 512, 0.93-0.94x; 8/2 B8 ctx 4096, 0.81x).

Only 16/4 uniform ctx 8192 (1.04x) and ragged B5 (1.04x) favour CuTe. The spread grid
holds one wave of CTAs (S = target // row-tile CTAs). At B >= 16 the long request
gets too few pieces, and at B=32 S = B, so spread is off. Lifting the cap needs a
multi-wave spread grid. The RTX PRO 6000 leg did not run: the device was not available.

## 128-row tiles for Q*G >= 128 (probe, GB10)

`CTI_DRAFT_ROWS=128` (a local env change, not committed) compiles at hd128 (224 registers,
no spills) and passes the FP32 check on K-Q7 32/4 Q16, B=1-4. Per layer it is slower
than the default tiles: B=1 12.80 vs 5.16 us at ctx 100, 31.5 vs 16.7 us at ctx 1000, and
335.8 vs 315.9 us at ctx 32000 (`draft_window_spread/kq7_rows128_gb10.jsonl`, combo 1).
Halving the row tiles halves the CTAs. A 128-row tile needs a split schedule that
restores the CTA count, not only the larger MMA tile.

## Reading

- **Ragged rows.** Spread is 1.4-6.0x faster on the RTX 5090, 1.7-6.1x on the RTX PRO
  6000 and 1.3-2.2x on GB10 wherever S > B. The RTX 5090 rows now tie or beat FA2: hd256 B4/B5 at 1.01/1.04x,
  B8 at 1.07x, B4 at a 131k context at 1.00x. The randomized 199k row went from 0.28x to
  0.71x FA2's time. Its uniform 199k twin is also 0.80x FA2 with either grid: that
  gap is the 32-row tile at Q*G=72 (three row tiles re-read K/V), work item (b) below.
- **GB10 199k row.** It has S = B (s = 1, 3 x 16 row CTAs = 48 SMs), so the spread grid
  could not give the long request a second CTA and stays off. Remaining gap: 0.73x FA2.
  Slack past the occupancy target (`CTI_DRAFT_SPREAD_SLACK=1`) would add CTAs, but
  past cooperative residency it forces the tree merge, which cost 2.1x on a uniform
  RTX 5090 hd256 row. Left opt-in.
- **Uniform rows.** Output is bit-identical (max_abs_diff 0). The schedule prologue
  (one lane per request, 4-5 warp reductions, one dependent reload) costs:
  - RTX 5090: 3.8% at ctx 100 (B2, 5.1 -> 5.3 us), 1.4% at ctx 1000, 2.4% for hd256 ctx 400;
  - RTX PRO 6000: 4.1% at ctx 100 and 1.6% for hd256 ctx 400;
  - GB10: 3.2% at ctx 100 and 2.1% for hd256 ctx 400;
  - nothing measurable at ctx >= 4000 on either device.
  Even with that cost, these short rows stay 1.08-2.5x faster than FA2.
- **Datapath.** The first version took the CTA's request from warp shuffles. ptxas then
  moved the per-CTA index math off the uniform datapath (+550 SASS, MOV +259, U* ops
  down). Taking every owner value from `redux.sync` and reloading the lengths at a
  uniform address restored it: 5728 vs 5584 instructions at hd256 B4.
  Int64 schedule math cost 8% at ctx 100; Int32 fixed that.
- **Per-request grid unchanged.** With spread off the attention SASS is identical
  to the parent commit (3312 instructions, no diff). The DW1 draft pairs (legacy merge)
  and the table path keep it.

## Wide spread: length-aware CuTe split grid past B = 32

### Problem
For B > 32 the CuTe grid was per request: each request got the construction-time split
count s (usually 1 there), so a ragged batch's long request was walked by one CTA per KV
head. RTX PRO 6000, randomized row 32/8 hd128 noncausal Q18 B=84 (61 inactive, one 235,617-token
request): CuTe 4169-4209 us, FlashInfer's own host plan 1250-1259 us (go-live round: 0.284x
SGLang). RTX 5090, B=33 with one long request: CuTe 2.5x (ctx 2048) and 2.9x (ctx 16384)
FlashInfer FA2's time.

### Design (`_draft/cute/_kernel.py`, `spread_schedule_wide`)
- B > 32 spreads by default (`CTI_DRAFT_SPREAD=0` still turns it off). Grid row
  S = B * s + max(1, target // row-tile CTAs): one extra wave of CTAs that long requests
  share. That exceeds co-residency, so these grids use the tree merge directly (reported by
  `.merge`, no fallback entry). A request that fits one CTA writes its output directly.
- The schedule: lane l owns requests l*G .. l*G+G-1 (G = ceil(B/32) <= 8), loads them once
  and pre-sums them, so each quantity takes one `redux.sync`; when every request fits one
  CTA, CTA x is request x (its words load with the blocks), with no owner search. Beyond
  8 groups, runtime loops over 32-request groups reload. Only redux results feed the index
  math (uniform datapath).
- First chunk divides the summed unions by the active requests, not B, so inactive requests
  do not split uniform active ones. No request is cut below 256 keys (`WIDE_MIN_CHUNK`).
- Head-major launch (x = KV head and row tile): a request's head CTAs run together and share
  its paged K/V rows, as in the per-request grid. Request-major cost 12% on a memory-bound
  uniform B=174 row (PRO 6000, 155.7 vs 139.0 us); head-major 142.3 us.
- B <= 32 code is unchanged: the B=4 spread attention SASS is identical (4768 instructions).

### Grid sizing (PRO 6000, B=84 row above, `CTI_DRAFT_TARGET_CTAS`)
One wave of slack 2049 us; two and four waves 2085-2192 us; eight waves 1364 us but the
uniform B=82 row then costs 51 vs 44 us. Kept at one wave. The remaining gap to FA2 on this
row is the row tile: Q18 x G4 = 72 rows take two 64-row CuTe tiles per head, each walking
the 235k keys, against one 128-row FA2 tile (work item b).

### Minimum piece (RTX 5090, us; per-request grid = code before this change)
| row | per-request grid | min 256 | min 512 | min 1024 | FA2 |
|---|---|---|---|---|---|
| h128c:33:2:2048 | 66.8 | 35.0 | 36.8 | 46.5 | 27.1 |
| h128n:64:2:16384 | 26.8 | 27.5 | 27.4 | 27.4 | 155.6 |
| h128c:33:2:16384 | 302.8 | 61.0 | 61.0 | 61.3 | 106.4 |
| h128c:64:2:2048 | 76.6 | 46.7 | 47.0 | 56.6 | 135.4 |
| h128c:64:1:2048 | 86.6 | 76.2 | 76.2 | 76.6 | 215.0 |
| h128c:128:2:2048 | 83.4 | 56.2 | 57.3 | 66.8 | 155.4 |
| h128c:128:1:128 | 42.8 | 45.2 | 45.1 | 45.3 | 47.8 |
| h128n:64:1:128 | 36.7 | 13.7 | 13.7 | 13.7 | 26.2 |
| h128c:33:1:2048 | 67.0 | 60.1 | 60.0 | 56.9 | 105.0 |
| h128c:64:2:16384 | 321.0 | 81.1 | 81.0 | 81.6 | 922.8 |

Row names: `h128c` = 32/8 hd128 causal full Q16, `h128n` = 32/8 hd128 noncausal W2048 Q8;
`geometry:B:pattern:top`, pattern 0 uniform, 1 log-uniform ragged, 2 one long request plus
short ones; capacity = top + Q. 256 is best or tied on every row but `h128c:33:1:2048`
(60.1 vs 56.9 at 1024). With 64-key pieces the windowed `h128n:64:2:16384` row was 43.8 us
(merge-bound). One row still loses to FA2: `h128c:33:2:2048` (35.0 vs 27.1 us).
\* The 256 rows came first in the call; the runner's stream trim dropped their marker line.

### Sweep, RTX 5090 (per-request grid vs wide spread at the 1024-key minimum, us)
| row | CuTe old us | CuTe new us | new/old | FA2 us | new/FA2 |
|---|---|---|---|---|---|
| h128c:33:0:128 | 22.7 | 23.6 | 1.04 | 25.8 | 0.91 |
| h128c:33:1:128 | 16.1 | 16.8 | 1.04 | 31.8 | 0.53 |
| h128c:33:2:128 | 13.1 | 14.2 | 1.08 | 19.9 | 0.71 |
| h128c:33:0:2048 | 183.4 | 183.4 | 1.00 | 230.4 | 0.80 |
| h128c:33:1:2048 | 67.0 | 55.9 | 0.83 | 102.8 | 0.54 |
| h128c:33:2:2048 | 66.8 | 45.7 | 0.68 | 26.6 | 1.72 |
| h128c:33:0:16384 | 1563.8 | 1564.3 | 1.00 | 1691.2 | 0.92 |
| h128c:33:1:16384 | 379.7 | 366.1 | 0.96 | 627.8 | 0.58 |
| h128c:33:2:16384 | 302.8 | 60.5 | 0.20 | 104.9 | 0.58 |
| h128c:64:0:128 | 40.1 | 39.6 | 0.99 | 44.2 | 0.90 |
| h128c:64:1:128 | 22.6 | 23.2 | 1.03 | 39.2 | 0.59 |
| h128c:64:2:128 | 23.0 | 23.6 | 1.03 | 30.8 | 0.77 |
| h128c:64:0:2048 | 393.1 | 393.2 | 1.00 | 458.2 | 0.86 |
| h128c:64:1:2048 | 86.6 | 75.8 | 0.88 | 213.3 | 0.36 |
| h128c:64:2:2048 | 76.6 | 55.9 | 0.73 | 133.7 | 0.42 |
| h128c:64:0:16384 | 2816.7 | 2819.1 | 1.00 | 3417.9 | 0.82 |
| h128c:64:1:16384 | 456.6 | 410.5 | 0.90 | 1495.6 | 0.27 |
| h128c:64:2:16384 | 321.0 | 80.7 | 0.25 | 905.1 | 0.09 |
| h128c:128:0:128 | 81.8 | 82.0 | 1.00 | 96.8 | 0.85 |
| h128c:128:1:128 | 42.8 | 44.6 | 1.04 | 46.8 | 0.95 |
| h128c:128:2:128 | 44.2 | 45.9 | 1.04 | 47.9 | 0.96 |
| h128c:128:0:2048 | 720.0 | 717.5 | 1.00 | 803.0 | 0.89 |
| h128c:128:1:2048 | 125.8 | 115.5 | 0.92 | 248.4 | 0.46 |
| h128c:128:2:2048 | 83.4 | 65.9 | 0.79 | 158.3 | 0.42 |
| h128n:33:0:128 | 11.9 | 12.4 | 1.04 | 36.9 | 0.34 |
| h128n:33:1:128 | 8.2 | 8.6 | 1.05 | 19.8 | 0.43 |
| h128n:33:2:128 | 7.8 | 8.4 | 1.08 | 17.0 | 0.49 |
| h128n:33:0:2048 | 179.2 | 180.1 | 1.01 | 221.7 | 0.81 |
| h128n:33:1:2048 | 32.2 | 32.1 | 1.00 | 94.0 | 0.34 |
| h128n:33:2:2048 | 21.1 | 21.1 | 1.00 | 29.9 | 0.71 |
| h128n:33:0:16384 | 179.9 | 180.2 | 1.00 | 221.4 | 0.81 |
| h128n:33:1:16384 | 88.9 | 88.9 | 1.00 | 220.9 | 0.40 |
| h128n:33:2:16384 | 15.4 | 15.5 | 1.01 | 29.8 | 0.52 |
| h128n:64:0:128 | 22.2 | 22.4 | 1.01 | 40.2 | 0.56 |
| h128n:64:1:128 | 36.7 | 13.5 | 0.37 | 26.1 | 0.52 |
| h128n:64:2:128 | 12.6 | 13.0 | 1.03 | 25.0 | 0.52 |
| h128n:64:0:2048 | 343.0 | 343.2 | 1.00 | 443.9 | 0.77 |
| h128n:64:1:2048 | 44.6 | 43.8 | 0.98 | 197.0 | 0.22 |
| h128n:64:2:2048 | 33.5 | 33.6 | 1.00 | 153.8 | 0.22 |
| h128n:64:0:16384 | 354.4 | 356.1 | 1.00 | 444.7 | 0.80 |
| h128n:64:1:16384 | 120.9 | 119.5 | 0.99 | 235.1 | 0.51 |
| h128n:64:2:16384 | 26.8 | 26.9 | 1.00 | 153.7 | 0.18 |
| h128n:128:0:128 | 40.4 | 39.3 | 0.97 | 67.8 | 0.58 |
| h128n:128:1:128 | 22.2 | 23.1 | 1.04 | 36.6 | 0.63 |
| h128n:128:2:128 | 22.3 | 23.1 | 1.04 | 36.7 | 0.63 |
| h128n:128:0:2048 | 697.5 | 696.4 | 1.00 | 780.1 | 0.89 |
| h128n:128:1:2048 | 101.2 | 101.2 | 1.00 | 230.4 | 0.44 |
| h128n:128:2:2048 | 40.1 | 39.8 | 0.99 | 148.1 | 0.27 |

At the 256 minimum the ragged rows above improve further (minimum-piece table); uniform rows
take the one-CTA-per-request path, which does not depend on the minimum.

### GB10 before the change (per-request grid vs FA2)
CuTe was already faster than FA2 on all 48 rows of the same sweep (0.31-0.97x FA2's time,
`wide_sweep_gb10_before.jsonl`): at 48 SMs the requests fill the GPU. GB10 rows with the wide
grid: not run yet (no GB10 was available).

### PRO 6000 paired rows (same call, code before -> wide spread at the 1024-key minimum, us)
| row | before | after |
|---|---|---|
| B=99 Q7 16/4 noncausal, ctx <= 122 ragged (randomized row) | 12.0 | 13.4 |
| B=174 Q16 16/4 noncausal, uniform 625, 50 inactive (randomized row) | 139.2 | 141.4 |
| B=82 Q1 32/8 causal, uniform 385 (randomized row) | 41.5 | 42.4 |
| B=84 Q18 32/8 noncausal, one 235k request, 61 inactive (randomized row) | 4169 | 1512 |

## CuTe prepare with a page-index buffer
`Prepare` launched ceil(capacity / 128) CTAs per request when the caller passes a page-index
buffer (the randomized evaluation always does), and every CTA first summed the earlier requests
serially. Now tiles without pages leave first, at most 32 CTAs per request stride over its
pages, and the prefix runs on all 128 threads (warp redux + shared reduction). PRO 6000,
page buffer passed (`FORCE_PAGES=1`):

| row | before | after |
|---|---|---|
| B=99, C=131,079 | 287.5 | 4.2 |
| B=174, C=262,160 | 895.8 | 6.8 |
| B=82, C=131,073 | 202.1 | 3.8 |
| B=84, C=241,957 | 388.2 | 11.4 |

Without a page buffer (engine regime) prepare is 1.6-2.5 us (was 4.7-16.7 us). The randomized PRO
round at c9e8bafc had charged this: B=174 Q16 at 1042 us vs SGLang 158 us (0.152x).

Raw rows: `draft_window_spread/wide_*.jsonl`.

## Randomized draft-attention rounds (tree at be7b8e1a)

- **RTX 5090: void.** The vLLM FA2 comparator failed to
  load (its vendored-file checksum check failed), so 0 of 8 rows were
  complete. Against SGLang FlashInfer alone, 3 rows missed precision; in the one
  CuTe+spread row (16/4 d256 causal, B3 Q16, ragged), RMS was 0.000538 against
  SGLang's 0.000533. A precision verdict needs the vLLM level.
- **RTX PRO 6000: DROP on speed.** Every row passes precision; geomean 1.034x
  (95% CI 0.984-1.089). The three rows below 0.98 do not use the spread grid:
  | row | ratio | why not spread |
  | --- | ---: | --- |
  | 16/4 d256 noncausal W2048, B198 Q5 | 0.900 | B > 32, FA2 |
  | 16/4 d128 noncausal, B1 Q1 at 35.5k | 0.959 | B = 1 |
  | 16/4 d128 noncausal, B57 Q47 | 0.974 | Q > 32, FA2 |

  Rows on the spread grid: 32/8 d128 noncausal W2048 B3 Q2 at 1.075x, and 8/2 d256
  causal B2 Q16 ragged at 18k at 1.009x.
