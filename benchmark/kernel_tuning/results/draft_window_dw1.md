<!-- SPDX-License-Identifier: MIT -->
# DW1 receipts: DraftAttention with a runtime window

> rc2 receipts. The DW1 completion for rc3 (review fixes, strict-rule results on the
> final code, SASS identity against 0.1.12, IKET) is in `draft_window_dw1_rc3.md`;
> its performance tables supersede the ones below.

Branch `kernels/draft-window`. Brief: any per-layer `window` (W >= 1 or None) and
`causal`, head dim 64/128/256, any dividing GQA ratio including MQA.

## What changed

- **API.** `DraftAttention` accepts any int W >= 1 or None and any bool `causal`
  per layer. The fixed four-mask whitelist is gone. The predicate stays the
  symmetric K0.17 one: `abs(p-k) < W`, intersected with `k <= p` when causal.
- **CuTe.** W is a runtime Int32 kernel argument. Kernels specialize only on
  `(causal, windowed)` and on tiling. Tiling (splits, index buffer, piece loop)
  uses `min(C, pow2ceil(W) + Q - 1)`, so every W in one power-of-two bucket
  shares the compiled kernels. W=2048 and W=4096 keep their previous tiling.
  Runtime split bounds and masks use the actual W.
- **FA2 fallback.** Every causal W maps to `window_left = W-1`. A noncausal W maps
  exactly only when W >= Q, because `window_left` has no right bound. FA2 rejects
  noncausal W < Q at construction and workspace sizing; CuTe serves those rows.
- **Mask arithmetic.** The mask is back on 0.1.12's absolute Int64 keys.
  - DF3's Int32 request-relative distances moved the CTA-uniform tile math off
    the uniform datapath: the legacy kt32 kernel had 30 uniform instructions,
    against 285 in 0.1.12. That kernel ran 1-2% slower than 0.1.12 on the RTX 5090.
  - Causal layers skip the implied right bound.
  - The causal-window kernel is 2664 SASS instructions; 0.1.12's is 2704.
- **Merge routing.** `dispatch.LEGACY_MERGE_ROWS` routes the three draft pairs to
  the legacy attention + merge kernels on all three devices: hd128, Hq/Hkv=4, Q16
  full or causal sliding, and Q8 noncausal sliding. In the paired chains below,
  DF3's single-launch inline merge was:
  - 4-37% faster than 0.1.12 only at L=128 with B>=2;
  - 3-28% slower at L=512 and L=4096 on every device.

  Runtime lengths cannot choose the mode, so these rows take the mode that is
  never slower. `CTI_DRAFT_MERGE`, or an explicitly set `CTI_DRAFT_INLINE`,
  overrides the rule.

## Correctness (FP32 dense reference, eager and CUDA-graph replay)

`tests/gpu/test_dflash_draft_window.py` covers:
- the grid: W in {1, 2, 7, 64, 2048, 4096, 8191, None} x causal x hd {64, 128, 256}
  x (Hq, Hkv) in {(2,2), (8,2), (64,1)};
- three providers per point: auto, explicit CuTe and explicit FA2;
- C=8256, P=16 and B=4 with an inactive slot;
- runtime lengths Q / 100 / 3000 / C, changed between replays inside one graph;
- graph output bit-identical to eager output, both checked against the reference;
- bucket reuse: W=3000 after W=4096 compiles nothing;
- a three-layer graph with W=64, 4096 and 3000.

Rows are skipped only where an explicit provider does not support the
geometry: CuTe needs D 128/256 and Q*G <= 128.

The suites run per device:
- the window grid (433 tests);
- the core draft GPU suites, `-m "not exhaustive"` (147 tests: attention, cute,
  inline, large, capacity, capture-trace, integration).

| device | window grid + core draft suites (final code) |
|---|---|
| RTX 5090 | 532 passed, 48 skipped |
| GB10 | 532 passed, 48 skipped |
| RTX PRO 6000 | 532 passed, 48 skipped (two calls) |

RTX 5090 also ran the exhaustive draft suites: 3645 tests in 16 shards, all
passing. `test_multiple_pieces_and_replay` now counts new compile misses instead
of exact totals. It had failed 16/16 at `a343142` because distributed
construction also compiles its tree fallback; all 16 now pass.

The CPU suite (`tests/modules tests/benchmarks tests/quality`) passes. Offline
CuTe compiles of the new signature pass: 26 sm_120a configurations and one
sm_121a configuration (`tests/modules/_dflash_draft_window_compile.py`).

## No regression against 0.1.12: the three draft pairs

Method (`benchmark/kernel_tuning/draft_window_ab.py`):
- All arms run in one process and share the same Q/K/V buffers.
  - The candidate is this branch.
  - The baseline is the 0.1.12 tree (`12f42d8`) and a second identical copy of it.
  - Each tree is imported under its own package name.
  - Earlier two-process pairing showed up to 10% A/A bias at L=4096, from
    per-process placement. A single process with shared buffers removes most of it.
- Shapes:
  - 32/8 heads, D=128, P=1, C=32768;
  - L = 128/512/4096 keys (ctx = L - Q), B = 1-4;
  - graph chains of at least 5 layers rotating K/V over at least 2x L2.
- Sampling:
  - 12 rounds; each round rebuilds every arm's op and graph in shuffled order;
  - 5 pairs per round, 60 pairs in total;
  - each sample is the median of 5 replays.
- Statistic: the speed ratio is 0.1.12 time / candidate time, with a bootstrap
  95% CI of the median.
- Pinning: GB10 runs on the X925 cores.
- A/A column: the same statistic for the second 0.1.12 copy against the first.
  It is the noise floor of the strict rule.

Strict rule: CI_high >= 1.
- Candidate: 33/36 rows on the RTX 5090, 32/36 on the RTX PRO 6000, 35/36 on the GB10.
- A/A, two identical 0.1.12 copies: 33/36, 32/36 and 32/36.
- So the candidate misses 8 of 108 rows, and identical code misses 11.
- Every candidate miss is within 0.9%. The worst is RTX PRO 6000 W2048 B2 L512
  at 0.991; the A/A copy reads 0.998 on that row.

### RTX 5090

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | candidate CI_high>=1 | max abs err |
|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 5.21 | 5.20 | 0.998 (0.997-1.000) | 0.997 (0.996-0.998) | yes | 0.0038 |
| full | 1 | 512 | 6.35 | 6.35 | 0.999 (0.997-1.000) | 1.000 (0.999-1.000) | yes | 0.0011 |
| full | 1 | 4096 | 15.50 | 15.50 | 1.000 (0.999-1.001) | 1.000 (1.000-1.001) | yes | 0.0003 |
| full | 2 | 128 | 6.73 | 6.73 | 0.997 (0.994-1.004) | 1.000 (0.994-1.004) | yes | 0.0021 |
| full | 2 | 512 | 8.62 | 8.64 | 1.001 (1.000-1.002) | 1.001 (1.000-1.003) | yes | 0.0011 |
| full | 2 | 4096 | 26.06 | 26.18 | 1.002 (1.000-1.006) | 1.004 (1.002-1.007) | yes | 0.0003 |
| full | 3 | 128 | 6.81 | 6.81 | 0.999 (0.996-1.004) | 0.997 (0.995-1.005) | yes | 0.0034 |
| full | 3 | 512 | 9.91 | 9.91 | 1.001 (1.000-1.002) | 1.001 (1.000-1.001) | yes | 0.0011 |
| full | 3 | 4096 | 36.34 | 36.35 | 1.000 (1.000-1.002) | 0.999 (0.998-1.003) | yes | 0.0004 |
| full | 4 | 128 | 7.03 | 7.07 | 1.011 (1.006-1.014) | 1.008 (1.001-1.012) | yes | 0.0021 |
| full | 4 | 512 | 11.00 | 10.99 | 1.000 (0.998-1.000) | 1.000 (0.999-1.001) | yes | 0.0016 |
| full | 4 | 4096 | 45.98 | 46.00 | 1.000 (0.999-1.001) | 1.001 (1.000-1.002) | yes | 0.0005 |
| c4096 | 1 | 128 | 5.56 | 5.51 | 0.992 (0.989-0.996) | 0.992 (0.991-0.999) | no | 0.0028 |
| c4096 | 1 | 512 | 6.33 | 6.33 | 1.000 (1.000-1.001) | 1.000 (0.999-1.000) | yes | 0.0011 |
| c4096 | 1 | 4096 | 15.47 | 15.48 | 1.002 (1.001-1.004) | 1.010 (1.003-1.010) | yes | 0.0003 |
| c4096 | 2 | 128 | 7.51 | 8.04 | 1.071 (1.062-1.074) | 0.997 (0.992-1.001) | yes | 0.0023 |
| c4096 | 2 | 512 | 8.87 | 9.43 | 1.064 (1.062-1.067) | 1.000 (1.000-1.001) | yes | 0.0011 |
| c4096 | 2 | 4096 | 27.01 | 27.59 | 1.025 (1.023-1.027) | 1.001 (1.000-1.002) | yes | 0.0003 |
| c4096 | 3 | 128 | 10.90 | 11.25 | 1.033 (1.032-1.035) | 0.999 (0.998-1.001) | yes | 0.0038 |
| c4096 | 3 | 512 | 12.66 | 13.11 | 1.036 (1.035-1.036) | 0.999 (0.998-1.001) | yes | 0.0011 |
| c4096 | 3 | 4096 | 39.04 | 39.45 | 1.011 (1.010-1.013) | 1.000 (0.998-1.000) | yes | 0.0004 |
| c4096 | 4 | 128 | 10.91 | 11.29 | 1.035 (1.035-1.038) | 1.000 (0.999-1.000) | yes | 0.0024 |
| c4096 | 4 | 512 | 13.31 | 13.41 | 1.008 (1.007-1.010) | 1.000 (0.999-1.002) | yes | 0.0020 |
| c4096 | 4 | 4096 | 48.55 | 49.00 | 1.009 (1.008-1.010) | 0.999 (0.998-1.001) | yes | 0.0004 |
| n2048 | 1 | 128 | 5.56 | 5.58 | 1.003 (1.000-1.009) | 1.000 (0.994-1.003) | yes | 0.0038 |
| n2048 | 1 | 512 | 6.33 | 6.33 | 1.000 (1.000-1.001) | 1.000 (1.000-1.000) | yes | 0.0011 |
| n2048 | 1 | 4096 | 8.03 | 8.02 | 0.999 (0.997-1.001) | 1.020 (1.000-1.026) | yes | 0.0005 |
| n2048 | 2 | 128 | 5.51 | 5.52 | 1.004 (0.998-1.007) | 1.000 (0.997-1.004) | yes | 0.0021 |
| n2048 | 2 | 512 | 7.16 | 7.16 | 1.000 (0.999-1.001) | 1.002 (1.000-1.004) | yes | 0.0011 |
| n2048 | 2 | 4096 | 10.94 | 10.95 | 1.000 (0.977-1.026) | 1.019 (1.000-1.031) | yes | 0.0005 |
| n2048 | 3 | 128 | 7.39 | 7.41 | 1.003 (0.999-1.003) | 1.002 (0.999-1.003) | yes | 0.0022 |
| n2048 | 3 | 512 | 9.65 | 9.62 | 0.997 (0.995-0.999) | 1.001 (1.000-1.003) | no | 0.0020 |
| n2048 | 3 | 4096 | 21.22 | 21.25 | 1.000 (0.999-1.003) | 1.000 (0.998-1.003) | yes | 0.0006 |
| n2048 | 4 | 128 | 7.35 | 7.37 | 1.005 (1.000-1.007) | 1.000 (1.000-1.002) | yes | 0.0022 |
| n2048 | 4 | 512 | 10.82 | 10.77 | 0.996 (0.995-0.998) | 1.000 (0.999-1.001) | no | 0.0011 |
| n2048 | 4 | 4096 | 25.90 | 25.91 | 1.000 (0.999-1.002) | 1.000 (0.999-1.001) | yes | 0.0005 |

Candidate: 33/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 33/36.

### RTX PRO 6000

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | candidate CI_high>=1 | max abs err |
|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 6.05 | 6.07 | 1.004 (1.001-1.005) | 1.000 (1.000-1.001) | yes | 0.0021 |
| full | 1 | 512 | 7.43 | 7.43 | 1.001 (0.999-1.003) | 1.004 (1.004-1.005) | yes | 0.0011 |
| full | 1 | 4096 | 17.29 | 17.28 | 0.999 (0.998-1.000) | 1.000 (0.998-1.001) | yes | 0.0003 |
| full | 2 | 128 | 7.45 | 7.45 | 0.996 (0.995-1.004) | 1.004 (1.004-1.006) | yes | 0.0021 |
| full | 2 | 512 | 9.75 | 9.76 | 1.000 (0.999-1.001) | 1.001 (1.000-1.001) | yes | 0.0011 |
| full | 2 | 4096 | 28.78 | 28.78 | 1.001 (0.999-1.002) | 1.000 (0.998-1.002) | yes | 0.0003 |
| full | 3 | 128 | 7.58 | 7.65 | 1.011 (1.003-1.012) | 1.002 (1.000-1.002) | yes | 0.0026 |
| full | 3 | 512 | 11.05 | 11.04 | 0.999 (0.998-1.001) | 0.999 (0.997-1.000) | yes | 0.0011 |
| full | 3 | 4096 | 39.78 | 39.74 | 1.000 (0.998-1.001) | 1.000 (0.999-1.001) | yes | 0.0004 |
| full | 4 | 128 | 7.88 | 7.85 | 0.994 (0.992-1.007) | 0.994 (0.992-0.994) | yes | 0.0038 |
| full | 4 | 512 | 12.36 | 12.37 | 1.001 (1.000-1.002) | 0.999 (0.998-1.000) | yes | 0.0012 |
| full | 4 | 4096 | 50.67 | 50.66 | 1.000 (0.998-1.001) | 1.000 (0.999-1.001) | yes | 0.0004 |
| c4096 | 1 | 128 | 6.76 | 6.75 | 0.998 (0.997-1.002) | 1.003 (0.999-1.005) | yes | 0.0022 |
| c4096 | 1 | 512 | 7.90 | 7.90 | 0.998 (0.996-1.000) | 1.000 (0.996-1.001) | no | 0.0010 |
| c4096 | 1 | 4096 | 17.66 | 17.73 | 1.003 (1.002-1.005) | 1.000 (0.999-1.000) | yes | 0.0003 |
| c4096 | 2 | 128 | 8.51 | 9.42 | 1.097 (1.096-1.106) | 0.999 (0.978-1.000) | yes | 0.0035 |
| c4096 | 2 | 512 | 10.07 | 10.56 | 1.050 (1.048-1.051) | 1.001 (1.000-1.001) | yes | 0.0010 |
| c4096 | 2 | 4096 | 29.33 | 30.11 | 1.028 (1.026-1.029) | 1.003 (1.002-1.004) | yes | 0.0003 |
| c4096 | 3 | 128 | 11.37 | 11.92 | 1.048 (1.047-1.049) | 1.001 (0.998-1.004) | yes | 0.0024 |
| c4096 | 3 | 512 | 13.67 | 14.02 | 1.025 (1.021-1.027) | 1.003 (1.000-1.004) | yes | 0.0012 |
| c4096 | 3 | 4096 | 42.28 | 42.57 | 1.008 (1.006-1.008) | 1.000 (1.000-1.002) | yes | 0.0005 |
| c4096 | 4 | 128 | 11.60 | 12.15 | 1.048 (1.043-1.049) | 1.000 (0.999-1.005) | yes | 0.0038 |
| c4096 | 4 | 512 | 15.07 | 15.31 | 1.017 (1.016-1.017) | 1.000 (0.999-1.000) | yes | 0.0011 |
| c4096 | 4 | 4096 | 53.20 | 53.56 | 1.007 (1.006-1.007) | 1.000 (1.000-1.001) | yes | 0.0003 |
| n2048 | 1 | 128 | 6.39 | 6.38 | 1.001 (0.996-1.002) | 0.997 (0.996-0.998) | yes | 0.0020 |
| n2048 | 1 | 512 | 7.47 | 7.43 | 0.998 (0.995-0.998) | 1.005 (1.002-1.007) | no | 0.0011 |
| n2048 | 1 | 4096 | 9.68 | 9.66 | 0.996 (0.995-0.999) | 0.999 (0.998-1.002) | no | 0.0005 |
| n2048 | 2 | 128 | 6.39 | 6.39 | 1.000 (0.996-1.003) | 0.997 (0.995-1.000) | yes | 0.0020 |
| n2048 | 2 | 512 | 8.51 | 8.43 | 0.991 (0.989-0.993) | 0.998 (0.997-0.999) | no | 0.0010 |
| n2048 | 2 | 4096 | 12.07 | 12.07 | 1.002 (0.998-1.007) | 1.001 (0.999-1.003) | yes | 0.0005 |
| n2048 | 3 | 128 | 8.09 | 8.13 | 1.006 (1.000-1.008) | 1.003 (0.994-1.006) | yes | 0.0028 |
| n2048 | 3 | 512 | 11.04 | 11.06 | 1.001 (1.000-1.002) | 0.999 (0.999-1.000) | yes | 0.0011 |
| n2048 | 3 | 4096 | 20.52 | 20.44 | 0.996 (0.989-1.005) | 1.005 (0.998-1.041) | yes | 0.0005 |
| n2048 | 4 | 128 | 8.29 | 8.33 | 1.006 (1.002-1.008) | 1.000 (0.999-1.003) | yes | 0.0021 |
| n2048 | 4 | 512 | 12.25 | 12.27 | 1.003 (1.000-1.004) | 1.002 (1.000-1.004) | yes | 0.0019 |
| n2048 | 4 | 4096 | 25.32 | 25.35 | 1.001 (1.000-1.003) | 0.999 (0.997-1.001) | yes | 0.0005 |

Candidate: 32/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 32/36.

### GB10

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | candidate CI_high>=1 | max abs err |
|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 7.54 | 7.38 | 0.985 (0.975-1.009) | 0.992 (0.979-1.000) | yes | 0.0023 |
| full | 1 | 512 | 15.48 | 15.47 | 1.000 (0.998-1.001) | 1.002 (1.000-1.003) | yes | 0.0011 |
| full | 1 | 4096 | 81.45 | 81.38 | 0.998 (0.995-1.001) | 1.002 (1.000-1.003) | yes | 0.0003 |
| full | 2 | 128 | 12.20 | 12.14 | 1.001 (0.994-1.013) | 0.992 (0.978-0.995) | yes | 0.0022 |
| full | 2 | 512 | 26.36 | 26.24 | 0.995 (0.994-0.996) | 0.996 (0.993-0.997) | no | 0.0010 |
| full | 2 | 4096 | 168.37 | 168.27 | 1.000 (0.998-1.002) | 1.002 (1.001-1.004) | yes | 0.0005 |
| full | 3 | 128 | 13.09 | 13.16 | 1.024 (1.007-1.026) | 1.003 (0.996-1.023) | yes | 0.0023 |
| full | 3 | 512 | 34.48 | 34.44 | 1.000 (0.997-1.002) | 1.000 (0.998-1.001) | yes | 0.0015 |
| full | 3 | 4096 | 235.88 | 236.01 | 1.000 (1.000-1.002) | 1.000 (1.000-1.001) | yes | 0.0005 |
| full | 4 | 128 | 17.28 | 17.34 | 1.006 (1.000-1.009) | 1.008 (1.002-1.019) | yes | 0.0030 |
| full | 4 | 512 | 44.64 | 44.63 | 1.000 (0.996-1.005) | 1.006 (1.003-1.008) | yes | 0.0011 |
| full | 4 | 4096 | 314.52 | 314.67 | 1.000 (0.999-1.002) | 1.000 (0.999-1.001) | yes | 0.0004 |
| c4096 | 1 | 128 | 9.29 | 9.15 | 1.002 (0.983-1.012) | 1.003 (0.980-1.008) | yes | 0.0023 |
| c4096 | 1 | 512 | 16.77 | 16.81 | 1.003 (1.001-1.003) | 0.998 (0.997-1.000) | yes | 0.0010 |
| c4096 | 1 | 4096 | 80.07 | 80.12 | 1.000 (0.998-1.002) | 0.997 (0.996-1.000) | yes | 0.0003 |
| c4096 | 2 | 128 | 11.17 | 11.19 | 1.015 (1.002-1.027) | 1.007 (0.989-1.017) | yes | 0.0023 |
| c4096 | 2 | 512 | 25.07 | 25.16 | 1.004 (1.001-1.006) | 0.999 (0.997-1.001) | yes | 0.0012 |
| c4096 | 2 | 4096 | 162.62 | 162.92 | 1.001 (1.000-1.003) | 1.001 (0.999-1.003) | yes | 0.0003 |
| c4096 | 3 | 128 | 13.70 | 13.53 | 0.995 (0.984-1.016) | 0.996 (0.988-1.015) | yes | 0.0023 |
| c4096 | 3 | 512 | 35.14 | 35.26 | 1.003 (1.002-1.005) | 0.999 (0.996-1.001) | yes | 0.0014 |
| c4096 | 3 | 4096 | 237.13 | 236.92 | 0.997 (0.995-1.001) | 0.998 (0.996-1.001) | yes | 0.0005 |
| c4096 | 4 | 128 | 17.34 | 17.49 | 1.008 (1.002-1.016) | 0.986 (0.982-1.002) | yes | 0.0036 |
| c4096 | 4 | 512 | 45.43 | 45.58 | 1.004 (1.001-1.006) | 0.999 (0.997-1.001) | yes | 0.0011 |
| c4096 | 4 | 4096 | 315.05 | 315.44 | 1.001 (0.999-1.003) | 1.001 (0.999-1.003) | yes | 0.0004 |
| n2048 | 1 | 128 | 7.43 | 7.48 | 0.991 (0.984-1.031) | 1.005 (0.994-1.011) | yes | 0.0021 |
| n2048 | 1 | 512 | 15.14 | 15.16 | 1.001 (1.001-1.002) | 1.000 (0.999-1.001) | yes | 0.0010 |
| n2048 | 1 | 4096 | 40.98 | 40.93 | 0.999 (0.998-1.000) | 1.000 (0.997-1.003) | yes | 0.0004 |
| n2048 | 2 | 128 | 8.65 | 8.61 | 0.996 (0.982-1.010) | 0.972 (0.962-0.982) | yes | 0.0023 |
| n2048 | 2 | 512 | 23.54 | 23.55 | 1.000 (0.999-1.003) | 0.999 (0.999-1.001) | yes | 0.0009 |
| n2048 | 2 | 4096 | 78.80 | 78.62 | 0.998 (0.997-1.000) | 1.000 (0.999-1.002) | yes | 0.0005 |
| n2048 | 3 | 128 | 9.41 | 9.55 | 1.017 (1.013-1.053) | 1.006 (0.971-1.046) | yes | 0.0023 |
| n2048 | 3 | 512 | 30.88 | 31.03 | 1.004 (1.002-1.007) | 1.002 (1.000-1.004) | yes | 0.0011 |
| n2048 | 3 | 4096 | 116.28 | 116.11 | 0.998 (0.997-1.000) | 0.998 (0.998-1.000) | yes | 0.0008 |
| n2048 | 4 | 128 | 8.35 | 8.67 | 1.039 (1.011-1.040) | 1.036 (0.982-1.040) | yes | 0.0023 |
| n2048 | 4 | 512 | 38.85 | 38.85 | 0.999 (0.997-1.001) | 1.004 (1.000-1.006) | yes | 0.0011 |
| n2048 | 4 | 4096 | 150.15 | 150.26 | 1.001 (0.999-1.002) | 1.000 (0.999-1.001) | yes | 0.0005 |

Candidate: 35/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 32/36.

## ncu evidence (RTX 5090, legacy-mode attention kernel, one eager call, three launches each)

| row | arm | instructions executed | uniform-pipe instructions | cycles (max SM) |
|---|---|---|---|---|
| W2048 B1 L512 | candidate | 511,083 | 73,120 | 16,744-16,809 |
| W2048 B1 L512 | pre-change HEAD | 516,075 | 7,744 | 17,341-17,463 |
| W2048 B1 L512 | 0.1.12 | 514,507 | 71,840 | 16,932-17,244 |
| causal W4096 B1 L128 | candidate | 292,939 | 3,840 | 15,970-16,392 |
| causal W4096 B1 L128 | pre-change HEAD | 304,651 | 3,712 | 16,549-16,989 |
| causal W4096 B1 L128 | 0.1.12 | 299,467 | 3,456 | 15,547-16,086 |
| W2048 B4 L4096 | candidate | 4,986,924 | 338,560 | 64,802-65,830 |
| W2048 B4 L4096 | pre-change HEAD | 5,103,404 | 103,680 | 64,490-66,381 |
| W2048 B4 L4096 | 0.1.12 | 5,040,684 | 320,640 | 65,488-69,275 |

The Int64 key form restores 0.1.12's uniform-datapath use. The causal-W4096 B1
L128 row executes 2% fewer instructions than 0.1.12 but uses about 1% more
cycles; the paired chains place it at the A/A floor.

## Why the legacy merge for these rows

Pre-change HEAD (DF3 defaults) against 0.1.12, same harness:
- **RTX 5090, inline merge:**
  - 1.10-1.25x faster at L=128 for B>=2;
  - 0.76-0.96x at L=512 and L=4096;
  - 0.87-0.90x at L=128 for B=1.
- **RTX PRO 6000 (causal W4096) and GB10 (W2048):** the same shape.

Knob sweeps on the RTX 5090 (KEY_TILE 32, SHORT=0, cooperative PDL off) did not
recover the L>=512 rows:
- Inline mode's co-residency target halves the split count at B>=2.
- The distributed merge costs more than the legacy merge kernel at B=1.

The window change itself is neutral against pre-change HEAD: 0.99-1.01 on every
row and device, with SASS register counts unchanged.

## Reproduce

```
python -m benchmark.kernel_tuning.draft_window_ab --arm cand=. --arm base=<0.1.12 tree> \
    --arm base2=<copy of the 0.1.12 tree> --masks c4096 --lens 128,512,4096 --pairs 60 --rounds 12
```
