# Small-M Linear: R1b holes, hot-L2 re-pair, host overhead (2026-10-15)

Measured with `linear_smallm_bxq_sweep.py --configs` (5 rotating-weight pairs) and
`linear_smallm.py --accept` (20 pairs, ε from self-pairs). SOL = weight-read
roofline against the read-only DRAM ceiling (RTX 5090 1661 GB/s).

## R1b: RTX 5090 holes at M = 24-64 stay on cuBLAS

The R1b gate is ≥ 1.03× over F.linear **and** ≥ 95 % SOL. The configuration sweep
added candidates for these rows: 128-row tiles, deeper rings, register streaming
with 2-22 deterministic splits sized to fill 170 SMs. No row passes both gates
reliably. A CTA count-limited kernel at large M is the cause (ncu, 4096x12288,
M=64: 128 CTAs at 1 CTA/SM, 8 % occupancy, 67 % DRAM throughput; cuBLAS 85 %).
F.linear itself sits at 83-95 % SOL on these rows. Where F.linear is above
~92 % SOL, 1.03× would need > 95 % SOL from us. The best candidate per row:

| N | K | Projection | M | F.linear us | F.linear SOL % | best CuTe us | CuTe SOL % | paired speedup | best config (rows,tile_k,stages,splits,warps,variant,prefetch) | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 12288 | mlp.down | 24 | 68.43 | 89.3 | 66.41 | 92.0 | 1.032 | 64,128,2,5,2,smem,4 | < 95% SOL |
| 4096 | 12288 | mlp.down | 32 | 69.00 | 88.7 | 67.72 | 90.4 | 1.019 | 64,64,4,5,4,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 12288 | mlp.down | 36 | 79.88 | 76.8 | 69.10 | 88.7 | 1.150 | 128,64,3,5,8,smem,4 | < 95% SOL |
| 4096 | 12288 | mlp.down | 48 | 70.37 | 87.5 | 69.66 | 88.4 | 1.006 | 64,64,4,5,4,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 12288 | mlp.down | 64 | 89.26 | 69.3 | 74.07 | 83.5 | 1.113 | 128,64,3,5,8,smem,4 | < 95% SOL |
| 4096 | 20480 | dflash-qwen3-8b fc (context) | 24 | 110.27 | 92.2 | 108.37 | 93.9 | 1.013 | 64,128,2,5,2,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 20480 | dflash-qwen3-8b fc (context) | 32 | 111.12 | 91.8 | 110.58 | 92.2 | 1.002 | 64,64,4,5,4,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 20480 | dflash-qwen3-8b fc (context) | 36 | 111.04 | 91.9 | 110.54 | 92.3 | 1.010 | 64,64,4,5,4,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 20480 | dflash-qwen3-8b fc (context) | 48 | 113.17 | 90.5 | 116.32 | 88.1 | 1.013 | 128,64,3,5,4,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 20480 | dflash-qwen3-8b fc (context) | 64 | 113.04 | 91.0 | 114.74 | 89.7 | 0.983 | 128,64,3,5,8,smem,4 | < 1.03x and < 95% SOL |
| 4096 | 32768 | dflash-qwen3.5-9b fc (context) | 24 | 171.54 | 94.8 | 167.09 | 97.4 | 1.029 | 128,64,2,5,4,reg,2 | < 1.03x |
| 4096 | 32768 | dflash-qwen3.5-9b fc (context) | 32 | 172.42 | 94.6 | 167.97 | 97.1 | 1.026 | 128,64,2,5,4,reg,2 | < 1.03x |
| 4096 | 32768 | dflash-qwen3.5-9b fc (context) | 36 | 174.24 | 93.7 | 170.56 | 95.7 | 1.016 | 64,64,4,5,4,smem,4 | < 1.03x |
| 4096 | 32768 | dflash-qwen3.5-9b fc (context) | 48 | 174.91 | 93.6 | 167.82 | 97.6 | 1.042 | 128,64,2,5,4,reg,2 | probe passes; acceptance did not hold |
| 4096 | 32768 | dflash-qwen3.5-9b fc (context) | 64 | 176.18 | 93.3 | 177.87 | 92.5 | 0.992 | 64,64,4,5,4,smem,4 | < 1.03x and < 95% SOL |
| 6144 | 4096 | attention.qkv (8B / drafts) | 24 | 36.10 | 84.8 | 34.10 | 89.7 | 1.057 | 128,64,3,7,4,smem,4 | < 95% SOL |
| 6144 | 4096 | attention.qkv (8B / drafts) | 32 | 36.02 | 85.2 | 34.90 | 88.0 | 1.033 | 128,64,3,7,4,smem,4 | < 95% SOL |
| 6144 | 4096 | attention.qkv (8B / drafts) | 36 | 36.30 | 84.7 | 35.70 | 86.1 | 1.020 | 128,64,3,7,8,smem,4 | < 1.03x and < 95% SOL |
| 6144 | 4096 | attention.qkv (8B / drafts) | 48 | 36.55 | 84.5 | 36.52 | 84.6 | 1.001 | 128,64,2,4,4,reg,2 | < 1.03x and < 95% SOL |
| 6144 | 4096 | attention.qkv (8B / drafts) | 64 | 37.29 | 83.4 | 40.11 | 77.5 | 0.928 | 64,64,4,4,4,smem,4 | < 1.03x and < 95% SOL |
| 10240 | 4096 | attention.qkv (9B, qkv+gate) | 24 | 55.13 | 92.4 | 53.83 | 94.6 | 1.024 | 64,64,4,1,4,smem,4 | < 1.03x and < 95% SOL |
| 10240 | 4096 | attention.qkv (9B, qkv+gate) | 32 | 55.04 | 92.8 | 54.61 | 93.5 | 1.010 | 64,64,4,1,4,smem,4 | < 1.03x and < 95% SOL |
| 10240 | 4096 | attention.qkv (9B, qkv+gate) | 36 | 54.96 | 93.0 | 55.22 | 92.6 | 0.995 | 64,64,4,1,4,smem,4 | < 1.03x and < 95% SOL |
| 10240 | 4096 | attention.qkv (9B, qkv+gate) | 48 | 55.36 | 92.7 | 56.02 | 91.6 | 0.990 | 128,64,2,2,4,reg,2 | < 1.03x and < 95% SOL |
| 10240 | 4096 | attention.qkv (9B, qkv+gate) | 64 | 55.57 | 92.9 | 58.55 | 88.1 | 0.952 | 64,64,4,2,4,smem,4 | < 1.03x and < 95% SOL |
| 12352 | 4096 | gdn.in_proj_qkvzba | 24 | 67.34 | 91.2 | 65.63 | 93.5 | 1.029 | 64,128,2,1,2,smem,4 | < 1.03x and < 95% SOL |
| 12352 | 4096 | gdn.in_proj_qkvzba | 32 | 67.40 | 91.3 | 67.59 | 91.1 | 1.001 | 128,64,3,4,8,smem,4 | < 1.03x and < 95% SOL |
| 12352 | 4096 | gdn.in_proj_qkvzba | 36 | 67.29 | 91.6 | 68.18 | 90.4 | 0.988 | 128,64,3,4,4,smem,4 | < 1.03x and < 95% SOL |
| 12352 | 4096 | gdn.in_proj_qkvzba | 48 | 67.51 | 91.6 | 70.37 | 87.9 | 0.961 | 128,64,2,2,4,reg,2 | < 1.03x and < 95% SOL |
| 12352 | 4096 | gdn.in_proj_qkvzba | 64 | 78.17 | 79.6 | 76.11 | 81.7 | 0.907 | 128,64,3,4,4,smem,4 | < 1.03x and < 95% SOL |

The 4096x32768 rows at M=24/32/48 (97 % SOL) were re-accepted with 20 pairs at
1.026-1.031×, inside the 3 % rule, so they stay on cuBLAS. Filling these rows needs
a new large-M variant: more resident CTAs per SM, x double-buffered with W
register streaming, and a fused fixed-order split-K reduction.

## Dispatch per M bucket, and the hot-L2 veto

Dispatch is per M bucket (1, 2, 3-4, 5-8, then 8-row buckets to 64). A bucket keeps
CuTe only if every measured M in it qualifies the same launch configuration, cold
(rotating weights). The engine-named shapes (24576x4096, 4096x4096, 2048x4096) at
all M, and every shape at M=64, were also re-paired with the weight hot in L2: the
same weight is called 16-256 times per graph, with no flush. A hot row below
1.03× vetoes its bucket. Hot L2 favours cuBLAS once M ≥ 17 for weights that fit in
L2: on the RTX 5090, 4096x4096 runs 0.74-0.87× hot and 2048x4096 0.86-1.03×. Those
buckets now use cuBLAS. Kept buckets: RTX 5090 35/46, RTX PRO 6000 92/102, GB10 7.

## GB10 table

Re-tuned on the reference GB10 and confirmed on a second host (X925-pinned).
These rows are kept:
- M=1 for N = 4096/6144/10240/12288 at K=4096: 1.16-1.25×, where cuBLAS M=1 takes a slow GEMV path;
- 64x4096 at M=2-8: 5.5×.

2048x4096 M=1 (1.21× cold) loses with the weight hot in L2 (0.55×), so it stays on cuBLAS.

## Host overhead per call (RTX 5090, median, Linear minus F.linear)

`linear_host_overhead.py`, 20 rounds x 100 back-to-back calls:

| Path | Linear − F.linear |
| --- | --- |
| CuTe, single kernel (splits = 1) | -1.3 to +0.4 µs |
| cuBLAS fallback | +1.7 to +2.1 µs |
| CuTe split-K (main + reduce launch) | +2.7 to +4.7 µs |

These deltas scale with host CPU speed. A slower host measured +4.5 to +6.0 µs (cuBLAS fallback),
+1.8 to +1.9 µs (single kernel) and +7.4 to +10.7 µs (split-K).
Split-K rows exceed the +2 µs target because of the second launch. A fused
in-kernel reduction removes it. Graph replay is unaffected.
