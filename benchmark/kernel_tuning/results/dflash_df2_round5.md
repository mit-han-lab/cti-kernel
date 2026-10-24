<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DF2: round 5 receipts (0.1.11rc10, 0.1.11rc11)

This round applies three decisions: per-row provider
selection for GB10's losing rows, an SOL floor that includes the measured graph-replay
step, and the R4 stopgap check against FA2. Earlier rounds:
[round 4](dflash_df2_round4.md), [DF2 market gate](dflash_df2_market_gate.md).

Summary:

- **RTX 5090 and RTX PRO 6000:** every row is at or above the faster of vLLM 0.31.0
  and SGLang 72730d6 dispatch (896/896 each, rc8 gate; the sm_120 paths are
  unchanged in rc10).
- **GB10:** 3 rows (full, B=3, Q=4, 16/4, L=4095/4096/4097) still lose to SGLang's
  FlashInfer 0.7.1 dispatch, and 2 more are borderline. None of our providers wins
  them. See [GB10 rows](#gb10-rows).
- **SOL ≥ 90%:** not met on any row, even against the corrected floor. See
  [SOL](#sol-against-the-measured-replay-floor).
- **R4 stopgap (rc11):** short-context verify/draft attention goes to our fastest own
  provider per row: the CuTe kernel for hd128 (DraftAttention now accepts causal full
  for target VERIFY) and FlashInferFixedQueryAttention with small Split-KV chunks for
  short requests (hd256). Rows that still lose are listed in
  [`dflash_r4_stopgap.csv`](dflash_r4_stopgap.csv). See [R4 stopgap](#r4-stopgap).
  The FA2 (flash-attn 2.8.3) provider is dropped; see [FA2](#fa2-dropped).

Per-row table for all 2688 gate rows (device, mask, Q, B, L, heads, provider, paired
ratios against SGLang and vLLM with ε, the gate result and notes):
[`dflash_df2_rows.csv`](dflash_df2_rows.csv). `ratio_vs_faster_market` is the larger
of the two paired ratios, which is the ratio against the faster framework. Gate
results come from the rc8 gate (RTX 5090, RTX PRO 6000) and the rc9 gate (GB10).

## Candidate

| wheel | sha256 | source | change vs rc9 |
| --- | --- | --- | --- |
| 0.1.11rc10 | `6cf004cd62f43ff126ee47d827cc69668a5594930485c7cd39b3d0fdd18530e2` | `4b5167d91013be34a91b1bfdb0c69d6387938c48` | GB10 only: three splits for full B=3 Q=4 16/4 rows over 4097–4128 keys |
| 0.1.11rc11 | `a3accefcc241777dd1001a3e13f207f1a7918c77ded53cec994085c5349c1a67` | `1a6fc1833e557a3fa9093d61adba39e08f017b58` | R4 stopgap: causal full in DraftAttention; per-device short-request Split-KV chunk table in FlashInferFixedQueryAttention |

Installed-wheel checks:

| device | rc10 GPU tests | rc10 memcheck | rc11 GPU tests | rc11 memcheck |
| --- | --- | --- | --- | --- |
| RTX 5090 | 889 passed | 141/141, `ERROR SUMMARY: 0 errors` | 1028 passed | 163/163, `ERROR SUMMARY: 0 errors` |
| RTX PRO 6000 | 889 passed | 141/141, `ERROR SUMMARY: 0 errors` | 1028 passed | 163/163, `ERROR SUMMARY: 0 errors` |
| GB10 | 889 passed | 141/141, `ERROR SUMMARY: 0 errors` | 1028 passed | 163/163, `ERROR SUMMARY: 0 errors` |

rc11 adds 139 GPU tests: causal full across the DRAFT suites, and
`min_chunk_tokens` against the dense reference. Its memcheck matrix adds causal-full
cases (163 in total). The first rc11 memcheck pass failed 4 cases with a harness
`RuntimeError`, with 0 sanitizer errors. At Q=4 the inactive-row poison segments
(7 entries per row) overflowed the capacity-5 index table. The harness now writes
min(7, capacity) entries (`2e90e8e`, test code only). The wheel is unchanged, and
the rerun is in the rc11 memcheck column.

## GB10 rows

Each session is 3 independent processes on one dedicated GB10 (pinned to the
Cortex-X925 cores), 40 interleaved pairs per process, ε from self-pairs. Cells give
the paired median ratio against SGLang for each process; `!` marks a failure. The
`S=k` cells force k splits through the environment; rc10 builds the S=3 rule in.

| row (mask:B:Q:L:Hq/Hkv) | default (S=4 or S=3), session 1 | S=3 | default, session 2 | rc10 (rule) |
| --- | --- | --- | --- | --- |
| full:3:4:4095:16/4 | 1.102 1.117! 1.142! | 1.092 1.001 0.974 | 1.143! 0.970 0.980 | 1.292! 1.118! 1.123! |
| full:3:4:4096:16/4 | 1.093! 1.115! 1.157! | 1.087! 1.008 0.970 | 1.122! 0.968 0.971 | 1.297! 1.129! 1.126! |
| full:3:4:4097:16/4 | 1.105 1.119! 1.186! | 1.089 1.006 0.978 | 1.124! 0.957 0.981 | 1.293! 1.109! 1.122! |
| causal_sliding:4:4:4095:16/4 | 1.048! 1.012 1.034 | (default is S=3) | 1.032 1.016 1.019 | 1.036! 1.018 1.038 |
| noncausal_sliding:3:4:2047:32/8 | 1.061 1.059 1.110! | 1.516! 1.203! 1.542! | 1.075! 0.991 0.935 | 0.966 0.971 1.096! |
| causal_sliding:3:9:4096:16/4 | 0.913 1.179! 0.941 | 1.148! 1.098! 1.069! | 1.025 0.972 1.016 | 0.981 1.020 0.853 |

Other variants measured in the same sessions: S=2, S=4, two CTAs per SM
(`TARGET_CTAS=96`) and our FlashInfer 0.6.11 provider. None beat the default in a
majority of processes. The FlashInfer provider's SGLang pairs are N/A (both load
FlashInfer in one process), and against stock FlashInfer 0.6.11 it was
1.00–1.21×, no better than CuTe. Absolute candidate times for one configuration
swing 52–86 µs between processes. Within session 1, S=3 was the fastest variant in
every process for the three full rows, so rc10 keeps the rule. The rc10 session
still lost all three full rows to SGLang.

The faster market kernel on these rows is SGLang's FlashInfer 0.7.1 fa2 prefill
with a host-side fixed plan. It is not a dependency of this package, which pins
FlashInfer 0.6.11.post3, and its plan does not follow changing device metadata
inside a capture. Routing these rows to it would need a separate decision.

## SOL against the measured replay floor

`dflash_replay_floor.py` times graphs holding k one-element kernels with the gate's
protocol: back-to-back `CUDAGraph.replay()`, 30 intervals of 100 replays, with
graph sizes interleaved. k = 0 is an empty capture. Medians in µs:

| device | k=0 | 1 | 2 | 3 | 4 | 8 | slope over k≥1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | 0.47 | 2.08 | 4.12 | 4.13 | 4.76 | 8.22 | 0.81 |
| RTX PRO 6000 | 0.62 | 2.12 | 4.16 | 5.14 | 6.21 | 10.31 | 1.12 |
| GB10 (X925-pinned) | 0.55 | 3.09 | 4.13 | 6.13 | 6.18 | 10.27 | 1.00 |

Replays advance in ~2 µs steps (~1 µs on GB10 beyond two kernels). A complete op
with k kernels cannot go below replay(k). Consecutive replays overlap host launch
with device work, so `dflash_draft_sol_floor.py` uses
`t_ref = max(useful-work t_ref, replay(k))`. Utilization is t_ref divided by the
gate's candidate latency.

| device | rows | util (corrected) | ≥ 90% | launch-bound rows (replay floor ≥ work) | their candidate − floor |
| --- | --- | --- | --- | --- | --- |
| RTX 5090 | 896 | 28.6–66.5% | 0 | 610, util 29–67% | 2.1–10.3 µs |
| RTX PRO 6000 | 896 | 27.2–67.8% | 0 | 630, util 27–50% | 4.1–11.1 µs |
| GB10 | 896 | 16.9–68.4% | 0 | 322, util 29–57% | 3.1–10.3 µs |

The corrected floor still leaves 2–11 µs on launch-bound rows. A real attention op
serializes dependent global round trips: lengths and indices, then K/V, then the
partial write, then the merge's reads. The one-element kernels used for the floor
issue none of these. That dependent-latency chain is the overhead this model still
omits. Measuring it as a floor (a minimal kernel pair with the same dependent loads)
is the next model fix. DF3's single launch removes one kernel and its round trips
from short rows.

## R4 stopgap

Decision: no FA2 provider. Each row goes to our own fastest provider,
measured per layer with the engine methodology (`dflash_draft_layers.py`: n layers
rotating over at least 2× L2, one graph, per-call time = elapsed / n). The
comparators are vLLM 0.31.0's real dispatch, `vllm_flash_attn` (fork 9cd61de,
P=16), and SGLang 72730d6's dispatch, FlashInfer 0.7.1 fa2 with a host plan, run in
its own process. The bar is the faster of the two.

Providers (rc11):

- **hd128 verify (q3_target, causal full) and draft (full non-causal):** the
  `DraftAttention` CuTe kernel. `DraftAttention(causal=True, window=None)` is new in
  rc11 and additive. The engine must construct it for verify to use it; its current
  verify op is `FlashInferFixedQueryAttention`.
- **hd256 verify (q35_target):** `FlashInferFixedQueryAttention`, whose device
  schedule now uses a smaller minimum Split-KV chunk while the longest request is
  short. The table is per (capability, head_dim): sm_120 hd128 32 tokens and hd256
  16 tokens up to 512 tokens, GB10 hd256 32 tokens up to 256. FlashInfer's
  `fixed_split_size` is the equivalent. This applies without engine changes.
  Applied at every length, small chunks cost up to 1.2–1.45× at ctx ≥ 1000, hence
  the length bound.

Result per device, shape and ctx (Q ∈ {4, 5, 6, 8, 9, 12, 16}, B = 1–4, exact
capacity, plus Q=16 at C=32768). PASS means at or below the faster framework:

| device | shape | ctx=100 | ctx=400 | ctx=1000 | worst losing ratio |
| --- | --- | --- | --- | --- | --- |
| RTX 5090 | q3_target | 29/32 | 32/32 | 32/32 | 1.15 (Q16 B2–4 at C=32768) |
| RTX 5090 | draft_noncausal | 29/32 | 32/32 | 32/32 | 1.18 (Q16 B2–4 at C=32768) |
| RTX 5090 | q35_target | 31/32 | 32/32 | 18/32 | 1.01 (ties with SGLang at ctx=1000) |
| RTX PRO 6000 | q3_target | 7/32 | 32/32 | 32/32 | 1.36 |
| RTX PRO 6000 | draft_noncausal | 1/32 | 32/32 | 32/32 | 1.39 |
| RTX PRO 6000 | q35_target | 19/32 | 32/32 | 23/32 | 1.27 (ctx=100); 1.01 at ctx=1000 |
| GB10 | q3_target | 20/32 | 32/32 | 32/32 | 1.17 |
| GB10 | draft_noncausal | 21/32 | 32/32 | 32/32 | 1.27 |
| GB10 | q35_target | 11/32 | 18/32 | 14/32 | 1.53 (ctx=100); 1.11 at ctx≥400 |

Against the engine's current verify path (default `FlashInferFixedQueryAttention`),
hd128 ctx=100 rows go from 1.0–3.1× the faster framework to 0.65–1.39×. The rest of
the ctx=100 gap is the DF3 target: a runtime-adaptive single launch. The hd256 rows
at ctx ≥ 400 on GB10, and the ~1.00–1.01 ties at ctx=1000 on sm_120, are FlashInfer
0.6.11 against SGLang's FlashInfer 0.7.1 running the same FA2 kernel family.

## FA2 dropped

Recorded for the page-size decision:

- flash-attn is absent from every test environment. The RTX 5090, the
  RTX PRO 6000 and the GB10 all fail with `ModuleNotFoundError: flash_attn`.
  The only build found locally is the engine env's flash-attn 2.8.3.post1, x86_64,
  with sm_120 SASS only.
- flash-attn 2.8.3's paged KV requires `page_block_size % 256 == 0`; the engine's
  DFlash pool is page_size=1.
- `dflash_draft_layers.py` keeps a `flash_attn` column (P=256), which SKIPs when
  the package is absent.
