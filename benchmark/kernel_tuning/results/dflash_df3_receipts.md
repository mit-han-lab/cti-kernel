<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DF3 receipts: single-launch CuTe DRAFT/verify attention

Branch `kernels/dflash-df3` on top of the rebased DF2 branch (`kernels/dflash-df2`).
Design: `docs/dflash_df3_design.md`.

## What changed

- **One kernel per call.** The CuTe DRAFT kernel picks its split count at run time
  from the device lengths. Unions up to a per-device threshold (`SHORT`) use a
  single split and write the output directly. Longer unions write FP32 partials and
  merge inside the same launch.
- **Merge.** The default is a distributed merge under a cooperative launch with PDL.
  Every split CTA of a (request, KV head, row tile) arrives on an op-owned counter
  and waits on a generation flag, then merges a slice of the rows. The cooperative
  launch guarantees co-residency, so the wait cannot deadlock behind kernels on
  other streams or under MPS (forward-progress test: two instances on two streams
  plus a long concurrent kernel).
  - A last-arriver tree merge (no waiting) is the automatic fallback when the grid
    cannot be co-resident.
  - The legacy two-kernel merge stays available (`CTI_DRAFT_MERGE`).
  - The counters are a dedicated zeroed tensor owned by the op, not part of the
    shared float workspace.
- **Defaults.** KEY_TILE 64 (hd128) with the inline merge. `SHORT` per device and
  batch size:

  | device | single-split threshold |
  |---|---|
  | RTX 5090 | 192 keys at B=1, else 288 |
  | RTX PRO 6000 | 288 keys at B<=2, else 416 |
  | GB10 | 416 keys |
  | hd256 | 160 keys (RTX 5090), 288 keys (RTX PRO 6000 and GB10) |

  Receipt: `dflash_df3_short_sweep.csv`.
- **head_dim 256** (Qwen3.5 VERIFY) in the CuTe kernel (swizzle `S<3,3,4>`).
  - Routed by default only on measured rows: RTX 5090 Q>=12; RTX PRO 6000 Q>=12
    and B>=3; GB10 everywhere except B=2 with Q<=8 (`dflash_df3_hd256_dispatch.csv`).
  - Elsewhere hd256 uses FlashInferFixedQueryAttention.
- **FixedQuery** passes `enable_pdl=False`. FlashInfer's prefill reads the device
  schedule before `griddepcontrol.wait`, so PDL raced with the preceding prepare.
  - Cost on RTX PRO 6000: geomean +2.0%, up to +7.3% at ctx 100
    (`dflash_fixedquery_pdl_ab.csv`).
- **hd128 FixedQuery short-request chunk tier** (32 tokens up to 512 keys on
  sm_120): confirmed on RTX PRO 6000. It is neutral to slightly better (0.96–1.005×
  the alternatives). GB10 correctly has no hd128 tier: smaller chunks cost up to 1.5×
  (`dflash_fixedquery_hd128_chunks.csv`).

## Market table (per-layer graph chains)

Method:
- Shapes:
  - Qwen3 draft: full non-causal attention, 32/8 heads, D=128.
  - Qwen3 target VERIFY: causal full attention, 32/8 heads, D=128.
  - Qwen3.5 target VERIFY: causal full attention, 16/4 heads, D=256.
- Matrix: Q 4/5/6/8/9/16, B 1–4, ctx 100/200/400/1000/4000, both exact capacity
  and C=32768.
- Timing: rotating K/V buffers over at least 2× L2, per-call graph-chain medians
  (`dflash_draft_layers.py`).
- Candidate: the dispatched provider.
- Bar: the faster of vLLM 0.31.0 `vllm_flash_attn` (P=16) and SGLang 72730d6's
  FlashInfer 0.7.1 fa2 dispatch, measured in the same call on the same GPU.
- Every candidate row is correct against the FP32 reference.

Full rows: `dflash_df3_table.csv`. Every device covers Q 4/5/6/8/9/16. GB10 timing
processes run pinned to the Cortex-X925 cores (5–9, 15–19).
Cells are rows at or below the bar / rows, with the worst losing ratio in parentheses:

| device | head dim | ctx 100 | ctx 200 | ctx 400 | ctx 1000 | ctx 4000 |
|---|---|---|---|---|---|---|
| RTX 5090 | 128 | 96/96 | 92/96 (worst 1.005) | 96/96 | 96/96 | 93/96 (worst 1.022) |
| RTX 5090 | 256 | 48/48 | 48/48 | 48/48 | 48/48 | 9/48 (worst 1.027) |
| RTX PRO 6000 | 128 | 90/96 (worst 1.021) | 96/96 | 96/96 | 93/96 (worst 1.024) | 94/96 (worst 1.016) |
| RTX PRO 6000 | 256 | 28/48 (worst 1.200) | 48/48 | 48/48 | 35/48 (worst 1.069) | 6/48 (worst 1.018) |
| GB10 | 128 | 83/96 (worst 1.114) | 80/96 (worst 1.068) | 93/96 (worst 1.055) | 90/96 (worst 1.069) | 85/96 (worst 1.047) |
| GB10 | 256 | 20/48 (worst 1.167) | 42/48 (worst 1.025) | 29/48 (worst 1.398) | 31/48 (worst 1.086) | 26/48 (worst 1.032) |

The DF3 target rows are hd128 at ctx 100/400/1000:
- RTX 5090: 288/288.
- RTX PRO 6000: 279/288, with 9 losses within 1.021–1.024×.
- GB10: 266/288 (worst 1.114).

Remaining losses:
- **hd256 at ctx 4000** (RTX 5090, RTX PRO 6000): FixedQuery is 0.2–2.7% behind
  SGLang's FlashInfer 0.7.1 in this table, and up to 6.5% behind in the FlashInfer pin A/B
  below. The FlashInfer version does not explain it (see the A/B). hd256 CuTe is not
  faster there except at Q16 B=1.
- **hd256 at ctx 100 on RTX PRO 6000** (Q<=8, FixedQuery after the PDL-off fix):
  up to 1.20×.
- **GB10 hd128 B=1 at ctx 100–1000:** up to 1.11× (vLLM at ctx 100, SGLang at
  ctx 200–1000). GB10 hd256 loses on many rows (up to 1.40× at ctx 400).

## FlashInfer pin A/B: hd256 ctx 4000 (RTX 5090, RTX PRO 6000)

Question: do the hd256 ctx 4000 losses change if the product moves from FlashInfer
0.6.11.post3 to 0.7.0.post1 or 0.7.1?

Method:
- One call per device and Q pair (Q 4/5, 6/8, 9/16), each on one GPU.
- Every call runs 3 rounds. Each round runs the three versions in a rotated order,
  each in fresh worker processes, with exact capacity and C=32768, B 1–4.
- FlashInfer 0.7.x is side-installed per node. For this test only, the shipped copy
  accepts the side-installed version in the two FixedQuery/DraftAttention version
  checks, and the SGLang worker accepts it too. A passing FixedQuery row therefore
  proves the side-installed version was imported.
- Ratio: our dispatched provider over SGLang's FlashInfer 0.7.1 dispatch in the same
  round, median of the 3 rounds. vLLM's `vllm_flash_attn` is far slower on these rows
  (about 220 µs), so SGLang is the bar.

Rows: `dflash_df3_fi_pin_ab.csv`. Every row is correct.

| device | rows at or below SGLang FI 0.7.1, ours on 0.6.11 / 0.7.0.post1 / 0.7.1 | worst ratio |
|---|---|---|
| RTX 5090 | 4/48, 4/48, 4/48 | 1.050, 1.049, 1.053 |
| RTX PRO 6000 | 1/48, 1/48, 1/48 | 1.072, 1.064, 1.065 |

- Under 0.7.1, losing rows are 0.6–5.3% behind on the RTX 5090 and 0.3–6.5% behind on the RTX PRO
  6000 (largest at B=1 and 2). The best of our two providers changes only Q16 rows.
- Our FixedQuery time moves by less than 1% between the three versions, and so does
  SGLang's (0.7.0.post1 vs 0.7.1).
- The gap therefore comes from our FixedQuery schedule, not the FlashInfer version:
  the device-side plan with PDL off, against SGLang's host plan with exact lengths.
- Under either 0.7.x pin the rows stay behind, so they are disclosed. No new provider
  work was done for them.


## SOL (informational)

The reference time is the larger of:
- the latency chain: the in-graph launch intercept, plus the measured dependent
  round trips (two L2, one DRAM, and, with a merge, one atomic, a flag poll and the
  partial loads);
- the streaming time of K/V, Q, O and index bytes at the shared sol-ceilings DRAM
  rate, or the BF16 MMA time.

Round-trip slopes come from `dflash_dependent_rt.jsonl`.

| device | min % | median % | max % | rows >= 90% |
|---|---|---|---|---|
| RTX 5090 | 15.0 | 28.1 | 86.8 | 0/520 |
| RTX PRO 6000 | 16.3 | 29.9 | 86.1 | 0/500 |
| GB10 | 25.0 | 84.1 | 149.4 | 226/680 |

GB10 values above 100% mean the rotating buffers are partly served by its system
cache, so the DRAM model does not hold there. On sm_120 the ctx 100 rows sit about
4 µs above the latency chain.
- One CTA per KV head at B=1 streams its K/V with limited per-SM memory-level
  parallelism.
- The model does not count this, and the SOL >= 90% goal is not met.

## Correctness and receipts (4a76c1f code; 9667a25/85a48d9 only relax two test bounds; the hd128 CuTe shards and GB10 memcheck rerun at b314ca6)

- **CI core** (`-m "not exhaustive"`, 11 DFlash/FixedQuery GPU files): 245/245 on
  RTX 5090, RTX PRO 6000 and GB10, 4.5–6 min per device.
- **Full sweeps at 85a48d9**, sharded under the runners' 30-minute cap. Counts are
  passed/run, with 0 failures in every shard:

  | shard | RTX 5090 | RTX PRO 6000 | GB10 |
  |---|---|---|---|
  | inline merge, hd128 | 656 | 656 | 656 |
  | inline merge, hd256 | 111 | 111 | 111 |
  | CuTe split boundaries/replay, hd256 | 395 | 395 | 395 |
  | CuTe split boundaries/replay, hd128 | 1179 (b314ca6, shards) | 1179 (b314ca6, shards) | 1179 |
  | capture trace + VERIFY parity + int64 bounds | 784 | 784 | 784 |

  - The inline sweeps include the two-stream forward-progress test, run under the
    distributed and tree merges.
  - On sm_120 the hd128 CuTe shard runs as contiguous index shards of the collected
    `-k 128` items (a pytest plugin in the test runner), each 6–13 minutes. GB10 ran
    it in one call.
- **Memcheck** (compute-sanitizer memcheck, CuTe DRAFT access matrix 143 cases, and
  the int64-bound cases at L=C=2^31−1):
  - 0 errors on RTX 5090 and RTX PRO 6000 at 4a76c1f.
  - GB10 0 errors at b314ca6: the 143-case matrix and the int64-bound cases.
- **Capture trace:** one kernel node per CuTe call, for both `run` and
  `prepare_and_run`.
- **CPU suite:** 5823 passed; pre-commit clean.
