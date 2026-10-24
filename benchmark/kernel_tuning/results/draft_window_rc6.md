<!-- SPDX-License-Identifier: MIT -->
# DraftAttention on the rc6 head: B > 4 correctness (DW-RC6)

## Branch

`kernels/draft-window-rc6` = `kernels/rc6-0.1.13` at de48a72c plus:

- The DW-R follow-up delta (the 22 commits from the handoff2 base up to b65ebcac),
  cherry-picked as 21 commits: the spread grid for 2 <= B <= 32, the wide spread for B > 32, the CuTe prepare fixes, D=64 to
  CuTe by default, the Q>32 B=1 D=128 precision routing to CuTe, the RTX 5090 B=2-4 rule
  added and then removed (net: no rule), and their receipts. 59220969 was dropped
  as empty: rc6 already holds the same change (handoff2 head 062a8018), through the
  handoff2 merge 75a390d2.
- 42ea9211: the CuTe split and inline-merge GPU tests pin the per-request grid
  (`CTI_DRAFT_SPREAD=0`). Spread is on by default for 2 <= B <= 32, and under spread a
  provider reports the grid size as `splits`. Six tests in `test_dflash_draft_cute.py`
  and `test_dflash_draft_inline.py` asserted per-request split counts and failed on GB10
  at B >= 2. The spread grid has its own suite (`test_draft_spread.py`). This is a fix to
  the tests only; no kernel output changed.
- 51eebe86: the B > 4 grid (`tests/gpu/test_draft_large_batch.py`) and the
  workspace-size report (`benchmark/kernel_tuning/draft_workspace_sizes.py`).

The DraftAttention code (`python/ctikernels/attention/draft.py`, `_draft/`,
`_cute_prepare.py`) is byte-identical to b65ebcac. Nothing in rc6's own changes touches it.

## B > 4 grid

`tests/gpu/test_draft_large_batch.py`. The test cases are:

- B in {5, 8, 16, 32, 48, 64, 191, 300}.
- D in {64, 128, 256}.
- Q in {1, 2, 5, 16, 32, 33, 64}.
- Four masks: full, causal, causal W=1000, and noncausal W=1000.
- Heads 8/2, C = 2048 tokens per request including Q, and page size 16.
- Extra auto rows:
  - noncausal W = Q // 2 (W < Q) for Q in {2, 5, 16, 33, 64};
  - an MQA 64/1 D=256 causal W=1000 arm (DeepSeek-V4-like draft) for Q in {1, 2, 8, 16}.

Arms:

| arm | provider | what it pins |
|---|---|---|
| auto | default routing | must never raise or skip on a valid geometry |
| cute | CuTe | per-request split grid (`CTI_DRAFT_SPREAD=0`) |
| spread | CuTe | spread grid for B <= 32, wide spread (lane blocks, tree merge) for B > 32 (`CTI_DRAFT_SPREAD=1`) |
| flashinfer | FA2 device schedule | |
| table, table-cute, table-flashinfer | `CacheTableDraftAttention` (auto, CuTe, FA2 table CSR) | table path |

Each case constructs the op once and stages uniform lengths (C - Q per request). It
captures one CUDA graph and then runs four turns:

1. uniform;
2. ragged: contexts 0 / 7 / C/2 / C-Q, with every seventh request inactive (slot -1), so B=5
   replays inactive to active;
3. one long request (the last row) with the rest at 7;
4. uniform again.

Every turn checks four things:

- The eager output matches the FP32 oracle (`draft_attention_reference`) at
  `ATOL = RTOL = 2e-2`, the window suite's tolerance.
- The graph replay is bit-equal to eager.
- Inactive rows are zero.
- Metadata and the output guard bands are intact.

Each case clears the compile caches, so every construction compiles its own kernels.

Only the table arms skip: `CacheTableDraftAttention` admits 2 <= Q <= 32 (its
constructor raises `ValueError("DRAFT requires 2 <= Q <= 32")`). That is 3 arms x 8 B x 3 D
x 3 Q x 4 masks = 864 cases per device. The CSR `DraftAttention` arms run every Q.

The run shipped the snapshot of 51eebe86 to each device and ran it in 16 strided shards
(`SHARD=i/16`), with no exhaustive filter.

| device | cases | passed | skipped (table Q outside 2..32) | failed | summed shard time |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 4856 | 3992 | 864 | 0 | 208 min |
| RTX PRO 6000 | 4856 | 3992 | 864 | 0 | 146 min |
| GB10 | 4856 | 3992 | 864 | 0 | 129 min |

Every skip is one of the 864 table-arm rows with Q outside 2..32, so no forced provider
skipped an admission check on these geometries. Per arm, each device passed:

- auto: 824 (672 grid rows, 120 noncausal W < Q rows and 32 MQA 64/1 rows);
- cute, spread and flashinfer: 672 each;
- table, table-cute and table-flashinfer: 384 each.

The core suites also ran at 42ea9211, with `-m "not exhaustive"`, on all three devices:
`test_dflash_draft_{attention,capture_trace,cute,inline,integration,window,table}.py`,
`test_draft_spread.py`, `test_draft_any_query.py` and `test_draft_custom_mask.py`. Each
device passed 641 tests: RTX 5090 in 846 s, RTX PRO 6000 in 695 s and GB10 in 588 s.
`test_draft_spread.py` (33 cases) covers B=2-300 ragged spread, wide spread and 131k
contexts, and is part of those 641.

## Workspace size vs B

`benchmark/kernel_tuning/draft_workspace_sizes.py` reports `workspace_size()` for each arm.
It covers 16,632 geometries per device: B in {1, 2, 4, 5, 8, 16, 32, 48, 64, 191, 300},
D in {64, 128, 256}, Q in {1, 16, 64}, the four masks, heads 8/2 and 32/8, and
C in {2048, 32768, 262144}. The script only routes and sizes; it compiles nothing, and
each device runs it in seconds. The table arms report `n/a` for Q outside 2..32. In the
auto rows, `c` marks CuTe and `f` marks FA2.

The tables give float + int bytes in MiB:

- **CuTe:** the workspace holds split partials. It falls to 0 (only 256 bytes of int
  workspace) once the construction-time split count reaches 1 (per-request grid at large
  B, or the spread grid when S = B).
- **Wide spread:** grows linearly with B past 32.
- **FA2:** its device schedule sizes partial buffers for the worst runtime split, so the
  workspace grows with B * Q * heads * D past the minimum. The largest is 3010 MiB, for
  B=300 32/8 D=256 Q=64 full.

### RTX 5090

32/8 D=128 Q=16 causal full C=32768: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 2.52 c | 2.52 c | 2.52 c | 2.52 c | 2.52 c | 0.00 c | 0.00 c | 14.61 c | 18.65 c | 50.64 c | 78.11 c |
| cute | 2.52 | 2.52 | 2.02 | 2.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 2.52 | 2.52 | 2.52 | 2.52 | 2.52 | 0.00 | 0.00 | 14.61 | 18.65 | 50.64 | 78.11 |
| flashinfer | 84.66 | 84.66 | 84.66 | 84.66 | 84.66 | 84.66 | 94.74 | 143.12 | 191.49 | 576.49 | 905.06 |
| table-cute | 2.52 | 2.52 | 2.02 | 2.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 84.91 | 85.16 | 85.66 | 85.91 | 86.66 | 88.66 | 102.74 | 155.12 | 207.50 | 624.27 | 980.10 |

8/2 D=256 Q=16 causal W=1000 C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 170.67 f | 287.14 f | 450.79 f |
| cute | 5.27 | 5.27 | 5.02 | 5.02 | 5.02 | 4.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 5.27 | 5.27 | 5.27 | 5.27 | 5.27 | 5.27 | 5.27 | 11.29 | 13.30 | 29.24 | 42.92 |
| flashinfer | 170.67 | 170.67 | 170.67 | 170.67 | 170.67 | 170.67 | 170.67 | 170.67 | 170.67 | 287.14 | 450.79 |
| table-cute | 5.27 | 5.27 | 5.02 | 5.02 | 5.02 | 4.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 172.67 | 174.67 | 178.67 | 180.67 | 186.67 | 202.67 | 234.67 | 266.68 | 298.68 | 669.17 | 1050.84 |

8/2 D=64 Q=64 noncausal full C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 43.17 f | 43.17 f | 43.17 f | 43.17 f | 43.17 f | 43.17 f | 43.17 f | 43.18 f | 48.52 f | 145.29 f | 228.35 f |
| cute | 1.27 | 2.54 | 2.54 | 2.54 | 2.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 1.27 | 2.67 | 2.67 | 2.67 | 2.67 | 2.67 | 0.00 | 8.76 | 10.79 | 26.92 | 40.75 |
| flashinfer | 43.17 | 43.17 | 43.17 | 43.17 | 43.17 | 43.17 | 43.17 | 43.18 | 48.52 | 145.29 | 228.35 |
| table-cute | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| table-flashinfer | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Largest workspace per arm over the whole report (all geometries, C up to 262144):

| arm | MiB | geometry |
|---|---:|---|
| auto | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| cute | 377.93 | B=300 32/8 D=128 Q=16 causal=True W=1000 C=32768 |
| spread | 606.36 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| flashinfer | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| table-cute | 5.35 | B=5 8/2 D=128 Q=16 causal=False W=None C=2048 |
| table-flashinfer | 2403.10 | B=300 32/8 D=256 Q=16 causal=False W=None C=262144 |

### RTX PRO 6000

32/8 D=128 Q=16 causal full C=32768: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 2.77 c | 2.77 c | 2.77 c | 2.77 c | 2.77 c | 0.00 c | 0.00 c | 14.87 c | 18.90 c | 50.90 c | 78.36 c |
| cute | 2.77 | 2.52 | 2.02 | 2.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 2.77 | 2.77 | 2.77 | 2.77 | 2.77 | 0.00 | 0.00 | 14.87 | 18.90 | 50.90 | 78.36 |
| flashinfer | 94.74 | 94.74 | 94.74 | 94.74 | 94.74 | 94.74 | 94.74 | 143.12 | 191.49 | 576.49 | 905.06 |
| table-cute | 2.77 | 2.52 | 2.02 | 2.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 94.99 | 95.24 | 95.74 | 95.99 | 96.74 | 98.74 | 102.74 | 155.12 | 207.50 | 624.27 | 980.10 |

8/2 D=256 Q=16 causal W=1000 C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 188.74 f | 287.14 f | 450.79 f |
| cute | 5.90 | 5.77 | 5.52 | 5.65 | 5.02 | 4.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 5.90 | 11.92 | 13.93 | 29.87 | 43.55 |
| flashinfer | 188.74 | 188.74 | 188.74 | 188.74 | 188.74 | 188.74 | 188.74 | 188.74 | 188.74 | 287.14 | 450.79 |
| table-cute | 5.90 | 5.77 | 5.52 | 5.65 | 5.02 | 4.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 190.74 | 192.74 | 196.74 | 198.74 | 204.74 | 220.74 | 252.74 | 284.75 | 316.75 | 669.17 | 1050.84 |

8/2 D=64 Q=64 noncausal full C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 47.74 f | 47.74 f | 47.74 f | 47.74 f | 47.74 f | 47.74 f | 47.75 f | 47.75 f | 48.52 f | 145.29 f | 228.35 f |
| cute | 1.40 | 2.79 | 2.54 | 2.54 | 2.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 1.40 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 | 0.00 | 9.01 | 11.05 | 27.17 | 41.01 |
| flashinfer | 47.74 | 47.74 | 47.74 | 47.74 | 47.74 | 47.74 | 47.75 | 47.75 | 48.52 | 145.29 | 228.35 |
| table-cute | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| table-flashinfer | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Largest workspace per arm over the whole report (all geometries, C up to 262144):

| arm | MiB | geometry |
|---|---:|---|
| auto | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| cute | 377.93 | B=300 32/8 D=128 Q=16 causal=True W=1000 C=32768 |
| spread | 606.36 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| flashinfer | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| table-cute | 5.92 | B=2 8/2 D=128 Q=16 causal=True W=1000 C=32768 |
| table-flashinfer | 2403.10 | B=300 32/8 D=256 Q=16 causal=False W=None C=262144 |

### GB10

32/8 D=128 Q=16 causal full C=32768: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 0.76 c | 0.76 c | 1.51 c | 1.51 c | 0.00 c | 0.00 c | 0.00 c | 13.61 c | 17.64 c | 49.64 c | 77.10 c |
| cute | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 0.76 | 0.76 | 1.51 | 1.51 | 0.00 | 0.00 | 0.00 | 13.61 | 17.64 | 49.64 | 77.10 |
| flashinfer | 24.19 | 24.19 | 24.19 | 24.19 | 24.19 | 46.36 | 94.74 | 143.12 | 191.49 | 576.49 | 905.06 |
| table-cute | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 24.44 | 24.69 | 25.19 | 25.44 | 26.19 | 50.36 | 102.74 | 155.12 | 207.50 | 624.27 | 980.10 |

8/2 D=256 Q=16 causal W=1000 C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 48.19 f | 48.19 f | 48.19 f | 48.19 f | 48.19 f | 48.19 f | 48.19 f | 71.28 f | 95.38 f | 287.14 f | 450.79 f |
| cute | 1.51 | 1.51 | 1.51 | 1.26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 1.51 | 1.51 | 1.51 | 1.51 | 1.51 | 0.00 | 0.00 | 7.53 | 9.54 | 25.48 | 39.15 |
| flashinfer | 48.19 | 48.19 | 48.19 | 48.19 | 48.19 | 48.19 | 48.19 | 71.28 | 95.38 | 287.14 | 450.79 |
| table-cute | 1.51 | 1.51 | 1.51 | 1.26 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| table-flashinfer | 50.19 | 52.19 | 56.19 | 58.19 | 64.19 | 80.19 | 112.20 | 167.29 | 223.39 | 669.17 | 1050.84 |

8/2 D=64 Q=64 noncausal full C=262144: float + int workspace in MiB (selected provider for auto)

| arm | B=1 | B=2 | B=4 | B=5 | B=8 | B=16 | B=32 | B=48 | B=64 | B=191 | B=300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | 12.19 f | 12.19 f | 12.19 f | 12.19 f | 12.19 f | 12.19 f | 24.13 f | 36.32 f | 48.52 f | 145.29 f | 228.35 f |
| cute | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| spread | 0.38 | 0.38 | 0.76 | 0.76 | 0.00 | 0.00 | 0.00 | 6.86 | 8.89 | 25.01 | 38.85 |
| flashinfer | 12.19 | 12.19 | 12.19 | 12.19 | 12.19 | 12.19 | 24.13 | 36.32 | 48.52 | 145.29 | 228.35 |
| table-cute | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| table-flashinfer | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Largest workspace per arm over the whole report (all geometries, C up to 262144):

| arm | MiB | geometry |
|---|---:|---|
| auto | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| cute | 151.17 | B=300 32/8 D=128 Q=16 causal=True W=1000 C=32768 |
| spread | 604.35 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| flashinfer | 3009.81 | B=300 32/8 D=256 Q=64 causal=False W=None C=2048 |
| table-cute | 1.51 | B=1 8/2 D=128 Q=16 causal=False W=None C=2048 |
| table-flashinfer | 2403.10 | B=300 32/8 D=256 Q=16 causal=False W=None C=262144 |

## B = 1-4 speed

No new speed work. The DraftAttention code is byte-identical to b65ebcac, so the B=1-4
rows are the DW-R receipts (`draft_window_dwr.md`, `draft_window_spread.md`).

A randomized draft-attention round was not run on this tree.

## Review points (no change to results or range)

- The table path (`CacheTableDraftAttention`) admits 2 <= Q <= 32; the CSR path admits
  any Q >= 1. The engine's table rounds use the DFlash block size (Q = 16).
- D=256 rows other than causal-full B <= 4 route to FA2 under auto (the hd256 CuTe rules).
  Its workspace is larger than CuTe's: for 8/2 D=256 Q=16 causal W=1000 at C=262144 it is
  189-451 MiB for B=1-300 on the RTX PRO 6000, against 6-44 MiB for CuTe spread. The
  largest auto workspace in the report is 3010 MiB (B=300, 32/8, D=256, Q=64, full,
  C=2048, FA2).
