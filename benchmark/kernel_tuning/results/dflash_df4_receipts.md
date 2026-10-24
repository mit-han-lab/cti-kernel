<!-- SPDX-License-Identifier: MIT -->

# DF4 receipts: cache-table interface and the remaining losing rows

DF4 order: 1. the direct `cache_index_table` / frontier / done / guard
interface; 2. the RTX PRO 6000 hd128 ctx 1000 B=3-4 and Q8 B=1 rows; 3. hd256
ctx 4000 on sm_120; 4. GB10 hd128 B=1 and hd256. Then DFMOE.

Timing method throughout: per-layer CUDA-graph chains over rotating K/V buffers
(>= 2x L2), `dflash_draft_layers_paired.py`: 20 reference self-pairs fix
epsilon (p95), then 20 interleaved candidate/reference pairs per reference, one
call on one GPU. Bar: the faster of vLLM 0.31.0 `vllm_flash_attn` (P=16) and
SGLang 72730d6's FlashInfer 0.7.1 dispatch. GB10 workers are pinned to the
Cortex-X925 cores (5-9, 15-19).

## 1. Direct cache-table interface (`CacheTableDraftAttention`)

API and semantics: `docs/dflash_kernels.md`, "Direct cache-table interface
(DF4)". The CuTe kernel reads `cache_index_table[r, j]` by row stride, with
`request_indices`, the int64 frontier, the done flags and the guard rule (R5
§3.1-3.3), and needs no token CSR and no prepare: **one kernel per call**. The
FlashInfer fallback (rows the dispatch routes to FlashInfer) expands the tables
into a private CSR with one CuTe kernel, then runs the existing FlashInfer path.

`metadata_index`:
- `"request"`: `frontier_table[r]`, `done_table[r]` (R5's tables).
- `"row"`: `frontier_table[b]`, `done_table[b]` of shape [B] (R5 P's
  `start_frontiers` / `round_done`). Recommended: the metadata loads issue
  together with `request_indices[b]`, one dependent load fewer.

Paired A/B against the CSR `DraftAttention` (same CuTe schedule, C=32768,
draft mask, 32/8 heads, D=128), `dflash_df4_table_ab.csv`:

| build | RTX 5090 ctx 100 | RTX 5090 ctx 400-1000 | RTX PRO 6000 ctx 100 | RTX PRO 6000 ctx 400-1000 |
|---|---|---|---|---|
| request, per-key staging branch (7fcc4a9) | 1.034-1.043 | 1.001-1.031 | 1.031-1.051 | 0.997-1.023 |
| row, per-key staging branch (7fcc4a9) | 1.012-1.024 | 0.987-1.023 | 1.008-1.029 | 0.989-1.015 |
| **row, split staging (bcd5522, HEAD)** | **0.995-1.012** | **0.963-1.003** | **0.996-1.009** | **0.965-1.002** |

- The per-key copy-or-store branch in index staging cost 1-2% at ctx 100. A
  bisect build without it ran at 0.98-0.99x the CSR path. HEAD stages keys below
  the live prefix with the CSR copy loop and writes guard keys in a second,
  CTA-uniform pass that runs only when a chunk reaches past the prefix.
- Row mode is now at parity at ctx 100 and 0-4% faster at ctx 400-1000: the CSR
  path loads `request_indices[b]` before its row lengths (inactive-row check),
  the row-mode table path loads them together.

Correctness (GPU, from the source tree; HEAD code):
- `tests/gpu/test_dflash_draft_table.py`, 1288 tests (both modes, both providers,
  four masks with window boundaries, done / guard-spill / out-of-range frontier
  rows, poisoned table entries outside each live prefix, graph capture and replay
  with changing metadata, bitwise parity with the CSR path, D128 32/8, D256 16/4,
  16/2 and 8/1): RTX 5090 1288/1288 (2323721); RTX PRO 6000 968/968 (bcd5522)
  and 1288/1288 in three shards (644/644 at 2323721; items 0-643 as 322 + 322
  at 2e8f8b8; items 644-1287 were run at 2323721 only); GB10 968/968 (bcd5522).
- Memcheck (`dflash_draft_access_check.py --provider table --launch-sanitizer`,
  190 cases: both metadata modes, four masks, D128 32/8 and D256 16/4 and 8/1,
  CuTe and the FlashInfer fallback, exact-size storage): `ERROR SUMMARY: 0 errors`,
  190/190 correct, on RTX 5090 and GB10 (fb43c9f). The M+Q near 2^31 case is
  skipped (it needs an ~8 GiB table row).
- Other suites at the early-trigger build: CuTe DRAFT cute+inline exhaustive
  2653/2653 on RTX 5090; DFlash core suite (198) on RTX 5090, RTX PRO 6000 and
  GB10; FixedQuery 80/80 and DraftAttention FlashInfer 1027/1027 on all three
  devices at 879b2f2.

## 2. RTX PRO 6000 hd128 rows

Sweeps: `dflash_df4_sweeps.csv` (variant and env per row).

- **Q9-16, B>=3, unions 513-4128 keys:** 32-row tiles beat 48/64-row tiles on
  every measured row (exact capacity, ctx 1000/2000/4000: 0.88-0.997x SGLang
  against 0.96-1.05x). HEAD sets `ROWS32_MIN_BATCH = {(12, 0, 188): 3}`.
- **Early PDL trigger** (all devices): the CuTe kernel now triggers its dependent
  launch right after `griddepcontrol.wait` instead of at exit. The next kernel
  can only launch once every CTA here has executed the trigger (so the whole
  cooperative grid is already resident) and it still waits for this grid's
  completion. RTX 5090 1-4% faster at ctx 1000; RTX PRO 6000 0.5-4% faster at
  ctx 100-1000 (`CTI_DRAFT_EARLY_TRIGGER=0` restores the exit trigger).
- **Q4/Q5, B=2-3, ctx 1000 (still behind SGLang by up to ~2%):** measured and
  rejected:
  - 32-key tiles (more splits): the cooperative grid then needs two CTAs per SM,
    shared memory is ~53 KB per CTA (just over half of 100 KB), so the merge falls
    back to the tree merge: 1.8-1.9x slower (ncu: 36.8 vs 19.9 us eager).
  - 32-key tiles with one key group (fits two CTAs per SM): not better.
  - Tree merge, no cooperative PDL, 3 stages: slower.
  - Legacy two-kernel merge: 3-12% faster at ctx 1000 but 13-43% slower at
    ctx 100-400 (one construction serves every runtime length).
  - Root cause: with 64-key tiles one CTA fits per SM, so the cooperative grid is
    capped at 188 CTAs; at B=3 (24 head tiles) that leaves 3 tiles per CTA, where
    FlashInfer runs two smaller CTAs per SM.
- **ctx 100, B=1-2 vs vLLM:** within 0.4-2.1% on the server-edition GPU before the
  early trigger; see the recount.

## 3. hd256 ctx 4000 on sm_120 (FixedQuery)

Analysis (source, both FlashInfer versions): our FixedQuery device schedule
already chooses exactly the split SGLang's FlashInfer 0.7.1 graph plan chooses
for these rows (same kv chunk, same CTA tile, same padded grid; split on). The
only execution difference was PDL: FlashInfer reads its schedule before
`griddepcontrol.wait`, so every call ran with PDL off. HEAD runs the **first call
after each `prepare()` without PDL** and every later call with PDL (safe: later
calls start after that call started). `CTI_FIXED_QUERY_PDL=0` restores PDL-off.

Result (`dflash_df4_hd256_pdl.csv`, C=32768): 0-2% faster at ctx 400 and at
ctx 4000 B=3-4. B=1-2 at ctx 4000 remain 0.3-1.7% behind with an identical
schedule and PDL. That gap is not explained by the schedule; the remaining known
difference is the FlashInfer build (0.6.11 here, 0.7.1 in SGLang), not isolated
in this round. See the recount below.

## 4. GB10

GB10 noise on the GB10 devices available (the reference-bandwidth GB10 was unreachable): self-pair epsilon
5-44%, median ~0.18, X925-pinned. Rows within that noise cannot be resolved; the
strict ratio is reported beside the epsilon gate.
- hd128 B=1 (C=32768, 42 rows at 879b2f2, before the early trigger): 41/42 pass the epsilon gate;
  the strict worst is 1.045 (ctx 400 vs SGLang). A lower single-split threshold
  for B=1 (352 instead of 416 keys) was measured and is not better.
- hd256 (Qwen3.5 16/4, C=32768): CuTe rows are 1-10% behind SGLang at several
  (B, ctx) points (`dflash_df4_gb10.csv`, variant g2). A same-call A/B at B=1
  (variants auto/fi): FlashInfer, now with PDL after the first call, ran at
  0.89-0.97x CuTe for Q4 and Q8 at ctx 100/400/1000 and at 0.97-1.006x the faster
  market; Q16 is mixed (CuTe faster at ctx 100/400). The dispatch now routes GB10
  hd256 B=1 Q<=8 to FlashInfer (B=2 Q<=8 already was); Q>=9 keeps CuTe. The same
  direction appeared in an earlier separate-call run (variant fi).
- Remaining GB10 hd256 losses (B>=3 CuTe rows at ctx 400-1000, Q16 B=1 at
  ctx 1000) are within or near this noise and are not resolved this round.

## 5. Recount (paired, `dflash_df4_paired.csv`; build per row in the `build` column)

hd128, 336 rows per device (draft and Qwen3 VERIFY, Q 4/5/6/8/9/12/16, B 1-4,
ctx 100/400/1000, exact and C=32768), dispatched provider (CSR `DraftAttention`):

| device | build | epsilon gate | strict ratio <= 1 | correct | worst failing | DF3 (7ea435c) |
|---|---|---|---|---|---|---|
| RTX 5090 | ea78a3d | 336/336 | 336/336 | 336/336 | none | 336/336, 336/336 |
| RTX PRO 6000 | ea78a3d; Q5/Q12 at 2e8f8b8 | 328/336 | 328/336 | 336/336 | 1.026 | 320/336, 311/336, worst 1.045 |

RTX PRO 6000 failing rows (8): Q4 B=3 ctx 1000 against SGLang (1.005-1.026,
section 2), and Q8 ctx 100 B=1-2 against vLLM (1.006-1.022; on the
server-edition GPU, where vLLM's ctx 100 time is relatively faster). The Q5 and
Q12 rows were rerun at 2e8f8b8 (after the early trigger): 96/96 on both gates,
so the earlier Q5 B=2-3 ctx 1000 losses are gone.

GB10 hd128 B=1 (879b2f2, C=32768, 42 rows): 41/42 epsilon gate, strict worst
1.045 (section 4). Superseded by the GB10-rule re-measurement below.

GB10 hd128 B=1 under the GB10 rule (2e8f8b8, `dflash_df4_gb10_rule.csv`): one
GB10 host per call (the same physical GPU for both calls), workers pinned to
the X925 cores 5-9 and 15-19, 60 interleaved candidate/market pairs per row,
24 graph replays of n >= 12 layer calls per sample (>= 1.4 ms per sample).
Per-pair speedup = faster-market time / candidate time; median and bootstrap
95% CI (10000 resamples); PASS only if CI_low > 1, FAIL if CI_high < 1, else
INCONCLUSIVE. All 42 rows correct.

| ctx | PASS | INCONCLUSIVE | FAIL | worst median speedup (faster market) |
|---|---|---|---|---|
| 100 | 0 | 2 | 12 | 0.970 (vLLM vllm_flash_attn, draft Q8) |
| 400 | 5 | 3 | 6 | 0.953 (SGLang FI 0.7.1, draft Q5) |
| 1000 | 8 | 4 | 2 | 0.985 (SGLang, draft Q9) |
| all | 13/42 | 9/42 | 20/42 | |

The ctx 100 losses are 0.6-3.0% against vllm_flash_attn (SGLang is 1.5-1.8x
slower there); the ctx 400 losses are 0.4-4.7% against SGLang. These rows are
deferred to DF5; no GB10 dispatch change was made from them.

hd256 Qwen3.5 VERIFY (16/4, C=32768, Q 4/5/6/8/9/16, B 1-4):
- RTX 5090 (ea78a3d): ctx 100/400/1000 72/72; ctx 4000 22/24 (worst 1.008;
  DF3 worst 1.027).
- RTX PRO 6000 (Q6/Q8 at ea78a3d, Q4/5/9/16 at 2e8f8b8, all on the
  server-edition GPU): 83/96 epsilon gate, 77/96 strict. By context: ctx 100
  21/24 (Q9 B4, Q16 B1-2 behind vllm_flash_attn by 1.2-13.7%), ctx 400 24/24,
  ctx 1000 18/24 (B=3-4 behind SGLang by up to 6.6%), ctx 4000 20/24 (within
  1.1% of SGLang). All 96 rows correct.

## 6. DFMOE (Qwen3.5-35B-A3B VERIFY: 16/2 and TP2 8/1, D=256)

- GPU tests (`tests/gpu/test_dflash_moe_shapes.py`, the table suite's 8/1 rows,
  and both geometries in `test_attention_prologue.py`): RTX 5090 362/362 (moe +
  prologue), RTX PRO 6000 311 + GB10 305 passed for the moe file; the prologue
  failures in those calls were the test runner not shipping `csrc` (ninja could not
  find `store.cu`); rerun with `csrc`: GB10 66/66 prologue; RTX PRO 6000 (2e8f8b8, CI core `-m "not exhaustive"` of the prologue and moe files) 57/57.
- The 16/4 hd256 dispatch did not carry over to group 8 (`dflash_df4_moe_sweeps.csv`):
  CuTe lost 8-28% at ctx 4000 for B>=2 on RTX 5090; on RTX PRO 6000 FlashInfer
  lost 10-20% at ctx 100 and CuTe lost at ctx 4000. `HD256_GQA8_CUTE_ROWS` picks
  per (device, KV heads, B, Q) the provider with the smaller worst ratio.
- RTX 5090 after the change (f9518b5): 16/2 36/36 epsilon gate (was 33/36);
  8/1 31/36 (FlashInfer at ctx 4000, 1.006-1.018).
- RTX PRO 6000 16/2 before the change: 21/36; after (2e8f8b8, server-edition
  GPU): 20/36 epsilon gate, 18/36 strict, 36/36 correct. ctx 1000 12/12;
  ctx 100 4/12 (Q8/Q16 behind vllm_flash_attn by up to 5.9%); ctx 4000 4/12
  (Q8/Q16 B>=2 behind SGLang by up to 18.8%). The dispatch change did not help
  on this GPU; these rows need kernel work (deferred, see STATUS).
- FixedQuery tier tables: unchanged (`docs/dflash_moe_shapes.md`); the 64/128-token
  minimum chunks lose 23-95% at ctx 100 for group 8.

## 7. Close-out: our providers on the deferred rows (RTX PRO 6000)

`dflash_df4_closeout.csv`: the same paired protocol (20 interleaved pairs per
reference, epsilon from 20 self-pairs, one GPU per call), C=32768, one row per
(variant, shape, Q, B, ctx). The candidate is forced per variant: `cute` (CuTe
split-KV, distributed merge), `legacy` (CuTe with the two-launch legacy merge),
`draft_fi` (FlashInfer device schedule), `fi_fixed_s16`/`fi_fixed_s32`
(FlashInfer fixed split, 16/32-token minimum chunk). Product code is unchanged
from calf2f92 except the `auto` rows, which ran the new dispatch (6e46873).
Routing is construction-time only, so a row moves only if one provider is
better across every context it serves (worst ratio over ctx 100/400/1000/4000).

| deferred row | shipped | best alternative of ours | decision |
|---|---|---|---|
| hd256 16/4 Q16 B=2 | FlashInfer, worst 1.148 (ctx 100) | CuTe, worst 1.017 (ctx 4000); ctx 100 0.990 | **routed to CuTe** (auto at 6e46873: ctx 100 0.983, 400 0.764, 1000 0.846, 4000 1.016, correct) |
| hd256 16/4 Q16 B=1 | FlashInfer, worst 1.046 (ctx 100) | CuTe, ctx 100 1.000 but ctx 4000 1.069 | stays FlashInfer (deferred) |
| hd256 16/4 Q4/Q5/Q9 B=1-4 ctx 100/1000 | FlashInfer, worst 1.013 (Q9 B4 ctx 100/1000, Q4 B4 ctx 1000 1.011) | CuTe loses ctx 1000 by up to 1.153; fi_fixed_s16 and legacy lose more | stays FlashInfer (deferred) |
| hd256 16/4 Q16 B=3-4 ctx 1000 | CuTe 1.006 / 1.064 | FlashInfer 0.955 / 1.046 at ctx 1000 but 1.183 / 1.221 at ctx 100 | stays CuTe (deferred) |
| hd128 Q8 B=1-2 ctx 100 | CuTe 1.006-1.022 (section 5) | legacy 1.09-1.12, draft_fi 1.37-1.44, fi_fixed_s32 1.37-1.46 | stays CuTe (deferred) |
| hd128 Q4/Q5 B=2-3 ctx 1000 | CuTe 0.99-1.026 | legacy 0.91-0.98 at ctx 1000 but 1.08-1.28 at ctx 100 | stays CuTe (deferred) |

The hd256 PRO ctx 100 row singled out for follow-up (Q16, 13.7% in section 5) was
Q16 B=2: it is now on CuTe and passes at ctx 100. The remaining rows above have
no provider of ours that wins at every context they serve; they stay in DF5.
