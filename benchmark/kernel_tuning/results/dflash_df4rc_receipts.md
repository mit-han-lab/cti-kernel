# DF4 on rc-0.1.13 (kernels/dflash-df4-rc): validation receipts

Base: `kernels/rc-0.1.13` @ 59d0762 (DW1 core: runtime window W and per-layer
causal, legacy merge for the three DFlash draft pairs). DF4 (kernels/dflash-df4
7ea435c..8eccb17) was applied as one squashed patch, and the conflicts were
resolved as follows:

- The CuTe table path (`CacheTableDraftAttention`) takes DW1's runtime `window`
  kernel argument and `split_bounds(length, window)`.
- Table attention keeps DF4's inline merge. DW1's `LEGACY_MERGE_ROWS` apply to
  the CSR `DraftAttention` provider only. Explicitly forcing
  `CTI_DRAFT_MERGE=legacy` with splits still raises for table instances.
- `dispatch.py` keeps DW1's `LEGACY_MERGE_ROWS` next to DF4's
  `HD256_GQA8_CUTE_ROWS`.
- `test_table_csr_parity_and_graph` pins `CTI_DRAFT_MERGE=distributed`, so that
  its CSR reference uses the table's schedule on the legacy-merge rows.
- Table geometry tests accept any positive window (DW1).
- The paired benchmark tooling that DF3/DF4 use (`dflash_draft_layers_paired*`,
  `build_provider`/`capture_graph`) is restored, because the RC did not carry it.

Not re-measured on this branch: the DF4 table interface (`cute-table`) on the
three DW1 draft pairs. It uses the inline merge, whereas the CSR provider uses
DW1's legacy merge. DW1's paired A/B found the legacy merge faster for those
pairs at L=512/4096.

## Correctness (devices: RTX 5090, RTX PRO 6000, GB10)

| Suite | RTX 5090 | RTX PRO 6000 | GB10 |
|---|---|---|---|
| CPU suite (`tests/modules tests/benchmarks tests/quality`) | 10782 passed, 0 failed | | |
| DRAFT/FixedQuery GPU core (`-m "not exhaustive"`, 11 files incl. table, MoE shapes, prologue) | 182/182 | 182/182 | 182/182 |
| `test_dflash_draft_window.py` (full grid) | 385 passed, 48 skipped | 385 passed, 48 skipped | 385 passed, 48 skipped |
| `test_dflash_draft_table.py` full (1288) | 1288/1288 | – | – |
| `test_dflash_draft_table.py -k "cute and 32-8-128"` (DW1 draft pairs) | in full | 160/160 | 160/160 |
| `test_dflash_long_context.py` (96 GPU + 8 host) | 56 core + 48 exhaustive pass | same | same |

The 48 skips are CuTe-unsupported geometries; for those, the window test's
`auto`/`flashinfer` parametrizations run the same masks.

Long-context rows: C = 131072 and 262144, B = 1–4 (B = 2/3 exhaustive), eager
plus changing-metadata graph replay, against a chunked FP32 reference.
- `CacheTableDraftAttention` with provider `cute` and `flashinfer`, on the
  three DFlash draft masks (32/8 D128: Q16 full non-causal, Q16 causal W4096,
  Q8 non-causal W2048) and on causal-full VERIFY (32/8 D128, 16/4 D256).
- `FlashInferFixedQueryAttention` VERIFY (32/8 D128, 16/4 D256).
- Host checks: B=4 at C=262144 is admitted, and capacities crossing signed int32
  are rejected before any launch.

## Long-context paired latency (Q=16, exact capacity C = ctx + Q)

Raw rows: `dflash_df4rc_longctx.csv` and `dflash_df4rc_longctx_gb10_60pairs.csv`.

Method:
- Candidate: `DraftAttention` automatic dispatch.
- References, measured in the same call: vLLM 0.31.0 `vllm_flash_attn` (P=16)
  and SGLang 72730d6's FlashInfer 0.7.1 FA2 dispatch.
- 20 interleaved pairs, with ε from 20 reference self-pairs; GB10 workers are
  pinned to the X925 cores.
- Rotating K/V: `DFLASH_BENCH_MAX_ROTATING_BYTES` = 8 GiB, or the two-layer
  minimum on the RTX 5090 rows at B≥3/131k and B=2/262k.

| Device | Rows | Correct | vs vLLM (candidate/ref) | vs SGLang (candidate/ref) | Market gate |
|---|---|---|---|---|---|
| RTX 5090 | 18 (131k B1–4, 262k B1–2) | 18 | 0.04–0.34 | 0.94–0.99 | 18/18 PASS |
| RTX PRO 6000 | 18 (131k B1–4, 262k B1–2) | 18 | 0.07–0.50 | 0.975–1.000 | 18/18 PASS |
| GB10 | 18 | 18 | 0.40–0.85 | 0.977–1.084 | 14/18 PASS |

GB10 rows behind SGLang (DF5, GB10 long-context split tuning):

| Row | 20 pairs | 60 pairs (GB10 rule) |
|---|---|---|
| draft B=2 C=131072 | 1.078 | 1.078; speedup CI95 0.924–0.931, FAIL |
| draft B=2 C=262144 | 1.084 | not rerun |
| q3 VERIFY B=2 C=131072 | 1.053 | 1.010; speedup CI95 0.987–0.995, FAIL |
| q3 VERIFY B=1 C=262144 | 1.038 | not rerun |

Diagnosis: the short-context split table gives the B=2 draft rows 2 splits,
which is 32 CTAs on 48 SMs. At these lengths the kernel is DRAM-bound, and
SGLang's plan fills the GPU.

## Rebase onto rc-0.1.13 with DW1 rc3 (28fd62d1)

The branch was rebased onto the RC tip that contains DW1 rc3. rc3 changed the
CuTe DRAFT kernel: the mask became per-row inclusive key bounds, the q row
stride became a constant, it supports any Q·G and hd64, and it added capacity
tiers. The rebase made these changes:

- The DF4 table path is ported onto rc3's per-tier `_configure_tier`
  compilation; `table_mode` is threaded through it.
- `CacheTableDraftAttention` sets the provider name behind rc3's read-only
  `provider` property, and its contiguous query row stride.
- FlashInfer DRAFT `pdl` reports DF4's steady state: on after the first call
  per `prepare()`, unless `CTI_FIXED_QUERY_PDL=0`. The FlashInfer table
  provider reports `pdl`/`merge`.
- The RTX PRO 6000 B≥3 32-row rule applies only to inline-merge rows. DW1's
  frozen legacy-merge draft pairs keep their 0.1.12 tiles (`test_frozen_pairs`).
- The long-context files are split: `test_dflash_long_context.py` (DW1:
  DraftAttention CuTe/FA2 and FixedQuery) and `test_dflash_table_long_context.py`
  (CacheTableDraftAttention CuTe/FA2, and FixedQuery with prepare inside the graph).

| Suite | RTX 5090 | RTX PRO 6000 | GB10 |
|---|---|---|---|
| CPU suite | 12296 passed, 0 failed | | |
| Core + window grid (`-m "not exhaustive"`, 12 files) | 524/524 | 524/524 (at ae6048b) | 524/524 |
| Long context, both files, core | 136/136 | 136/136 | 136/136 |
| Long context, exhaustive (B=2/3) | 48/48 | – | – |
| `test_dflash_draft_table.py` full | 1288/1288 | – | – |
| Table, DW1 draft pairs (`-k "cute and 32-8-128"`) | in full | 160/160 (at ae6048b) | 160/160 |

The RTX 5090 and GB10 rows ran at 510deed. The following commit changes only
the RTX PRO 6000 B≥3 tile rule.

Paired gate after the rebase:
- Rows: hd128 draft (32/8, full non-causal) and VERIFY (causal), C=32768,
  ctx 100/1000, Q 4/8/16, B 1–4.
- Comparators: vLLM 0.31.0 `vllm_flash_attn` and SGLang 72730d6 FlashInfer 0.7.1;
  20 pairs, with ε from self-pairs.

| Candidate | RTX 5090 | RTX PRO 6000 | GB10 |
|---|---|---|---|
| `table_row` (engine interface, inline merge) | 48/48 | 46/48 | 48/48 |
| `auto` (CSR DraftAttention) | 45/48 | 20/24 (B=3–4) | 46/48 |

`table_row` misses on PRO, both DF5 residual rows:
- draft Q8 B=1 ctx100: 1.001 vs vLLM;
- VERIFY Q4 B=3 ctx1000: 1.008 vs SGLang.

`auto` misses are, apart from the DF5 Q4 B=3 ctx1000 rows (1.026), all the same
row: DW1's legacy-merge Q16 full non-causal draft pair at ctx 100, slower than
vLLM `vllm_flash_attn`.

| Device | B | vs vLLM |
|---|---|---|
| RTX 5090 | 2–4 | 1.15–1.20 |
| RTX PRO 6000 | 3–4 | 1.34–1.37 |
| GB10 | 1–2 | 1.12–1.22 |

These are DW1's routing (`LEGACY_MERGE_ROWS`). Their compile constants are the
frozen 0.1.12 ones, which this rebase does not change. The inline-merge table
path wins the same rows.
