<!-- SPDX-License-Identifier: MIT -->
# DW1 completion (rc3): DraftAttention with a runtime window

Branch `kernels/draft-window`, commits `03f61eb..HEAD` on top of the rc2 core
(`d6fe743`). This file supersedes the performance sections of
`draft_window_dw1.md`, which keeps the rc2 history.

## Review findings and what changed

| finding | change |
|---|---|
| P1: no provider for noncausal W<Q with hd64 or Q*G>128 | CuTe now covers D 64/128/256 and any dividing GQA (MQA included) up to the grid limits. FA2 still rejects noncausal W<Q (`window_left` has no right bound). Auto routing falls back to CuTe whenever FA2 cannot express the mask, and to FA2 whenever CuTe rejects. The default routing rules are unchanged: hd64 goes to FA2, hd128 to CuTe, hd256 per `HD256_CUTE_ROWS`. |
| P1: compilation depends on W | The CuTe compile keys depend only on (causal, windowed) and construction-time capacity values. A windowed layer compiles the fixed tiers `{min(C, 4128), C}` at construction and picks one from W. `workspace_size` is the maximum over the tiers, so it does not depend on W. Inside the kernel, a CTA-uniform runtime branch runs the unwindowed walk when the request length L <= W. W never selects compiled code. |
| P2: FA2 compiles per W | `prepare_draft` / `prepare_split_kv` take `WINDOW_LEFT` as a runtime argument with `do_not_specialize`. Seven W values (including W-1 = 0, 1 and multiples of 16) add no Triton compile (GPU test). |
| P2: FA2 int32 overflow on huge W | `DraftAttention` treats a finite W >= C as the full mask before provider selection, and `check_capacity` clamps W to C. GPU tests cover W = C, C+1 and 2^31-1 on both providers. |
| P1: silent PDL downgrade | The eager cooperative-PDL retry warns once (`RuntimeWarning`) and is recorded. Construction fallbacks (distributed to tree) are recorded too. Read-only `DraftAttention.pdl` and `.merge` sit next to `.provider`, and `Provider.fallbacks` lists every change. |
| P1: row-strided q | `DraftAttention(query_row_stride=...)` declares a fused-QKV column view (strides `(s, D, 1)`, `s >= Hq*D`, 16-byte rows). The declaration happens once, at construction: the CuTe kernel folds the row stride into its address arithmetic. Contiguous q (the default) compiles 0.1.12's exact address expression. A runtime-stride form changed the whole kernel's code generation (see SASS below). FA2 reads strides per call. |
| P2: unused page CSR | `page_indices=None` is legal with CuTe: Prepare and the fused publish skip page-index writes (a constexpr flag). `DraftAttention.requires_page_indices(...)` reports whether the selected provider needs it. |
| P2: raw paired measurements | `draft_window_ab.py` emits one `type: pair` record per pair (per-arm us and ratio), plus the bootstrap seed and resample count. The raw JSONL of every A/B run below is in `draft_window_dw1_rc3/`. |
| P2: IKET evidence | CuTe ranges (`attn.prologue/walk/partial/merge/merge_wait/epilogue`, `merge.all`) compile only with `CTI_DRAFT_IKET=1`. Driver: `draft_iket.py` / `draft_iket_node.py`. Results below. |
| K-Q7 | Qwen3-30B-A3B rows (32/4 heads, hd128; Q16 full and causal full, Q8 causal full and W2048) are in the GPU grid on all providers. The A/B harness has `q3moe` masks. |
| test gaps | No provider skips. Auto must construct and match the FP32 reference at every grid point. Explicit FA2 must raise exactly for noncausal W<Q. Explicit CuTe supports every point. |

## Correctness (FP32 reference; eager plus graph replay with changing lengths; graph bit-identical to eager)

Final code `9b8fe1a`:

| device | `test_dflash_draft_window.py` | core draft suites (`-m "not exhaustive"`) |
|---|---|---|
| RTX 5090 | 342 passed | 147 passed |
| RTX PRO 6000 | 342 passed | 147 passed |
| GB10 | 342 passed | 147 passed |

What the window suite covers:
- the grid: W {1, 2, 7, 64, 2048, 4096, 8191, None} x causal x hd {64, 128, 256} x (Hq, Hkv) {(2,2), (8,2), (64,1)}, with auto routing over the full grid and explicit CuTe/FA2 arms;
- MQA 64:1 with Q 8/16 (512/1024 packed rows per KV head) x hd {64, 128, 256} x {causal W64, noncausal W3 (CuTe only), full};
- the DSV4 draft geometry (64/1 heads, hd256, Q8, causal W2048, B=1-4);
- W >= C and W = 2^31-1;
- strided q (bit-identical to contiguous, and stride mismatches raise);
- `page_indices=None`, FA2 cache reuse, CuTe compile reuse across W, `.pdl`/`.merge`, and the K-Q7 rows.

The RTX 5090 also ran the exhaustive draft suites (`-m exhaustive`, the seven core files, sharded): 3647 passed, 0 failed.

The CPU suite (`tests/modules tests/benchmarks tests/quality`) has 9600 passed, 4 skipped.
`pre-commit run --all-files` is clean.

## No regression against 0.1.12: the three draft pairs (strict rule)

The method is unchanged from `draft_window_dw1.md`:
- `draft_window_ab.py`, with all arms in one process sharing the Q/K/V buffers;
- 32/8 heads, D=128, P=1, C=32768, L = 128/512/4096, B = 1-4;
- 12 rebuild rounds, 60 pairs, each pair the median of 5 replays;
- the bootstrap 95% CI of the median 0.1.12/candidate ratio, plus an A/A arm (a second 0.1.12 copy);
- GB10 pinned to the X925 cores.

Rule: a row passes when CI_high >= 1. The final code is `9b8fe1a` on every row.

| device | rows measured | candidate CI_high >= 1 | A/A CI_high >= 1 |
|---|---|---|---|
| RTX 5090 | 36 | 34 | 34 |
| RTX PRO 6000 | 36 | 29 | 34 |
| GB10 | 36 | 28 | 31 |

Total: candidate 91/108 rows pass; the A/A copy of identical 0.1.12 code passes 99/108.

### Rows that miss the strict rule

| device | pair | B | L | speed | CI_high | A/A CI_high | kernels vs 0.1.12 |
|---|---|---|---|---|---|---|---|
| RTX 5090 | full | 3 | 512 | 0.999 | 0.9998 | 1.0007 | identical |
| RTX 5090 | c4096 | 4 | 512 | 0.998 | 0.9988 | 1.0012 | attention differs (3544 vs 2728 SASS), merge identical |
| RTX PRO 6000 | full | 1 | 512 | 0.998 | 0.9984 | 1.0000 | identical |
| RTX PRO 6000 | full | 2 | 512 | 0.999 | 0.9998 | 1.0001 | identical |
| RTX PRO 6000 | n2048 | 1 | 512 | 0.996 | 0.9975 | 1.0043 | attention differs (3384 vs 2704 SASS), merge identical |
| RTX PRO 6000 | n2048 | 1 | 4096 | 0.997 | 0.9988 | 1.0008 | attention differs (3384 vs 2704 SASS), merge identical |
| RTX PRO 6000 | n2048 | 2 | 512 | 0.998 | 0.9999 | 1.0053 | attention differs (3384 vs 2704 SASS), merge identical |
| RTX PRO 6000 | n2048 | 2 | 4096 | 0.992 | 0.9974 | 1.0006 | attention differs (3384 vs 2704 SASS), merge identical |
| RTX PRO 6000 | n2048 | 3 | 512 | 0.998 | 0.9987 | 1.0021 | attention differs (3392 vs 2704 SASS), merge identical |
| GB10 | full | 2 | 512 | 0.995 | 0.9966 | 0.9972 | identical |
| GB10 | full | 3 | 128 | 0.995 | 0.9991 | 1.0103 | identical |
| GB10 | c4096 | 2 | 4096 | 0.997 | 0.9979 | 0.9993 | attention differs (3840 vs 3000 SASS), merge identical |
| GB10 | n2048 | 1 | 4096 | 0.997 | 0.9976 | 0.9990 | attention differs (4096 vs 3152 SASS), merge identical |
| GB10 | n2048 | 2 | 4096 | 0.997 | 0.9985 | 1.0008 | attention differs (4096 vs 3176 SASS), merge identical |
| GB10 | n2048 | 3 | 512 | 0.995 | 0.9959 | 0.9994 | attention differs (4096 vs 3160 SASS), merge identical |
| GB10 | n2048 | 3 | 4096 | 0.996 | 0.9991 | 1.0009 | attention differs (4096 vs 3160 SASS), merge identical |
| GB10 | n2048 | 4 | 128 | 0.984 | 0.9997 | 0.9981 | attention differs (4096 vs 3024 SASS), merge identical |

Classification:
- **SASS-identical to 0.1.12.** The candidate's attention and merge kernels for these rows match 0.1.12 instruction for instruction; only kernel-parameter constant-bank offsets differ (table below). The miss is harness noise or bias. The A/A copy shows the same: on GB10 full B2 L512 the candidate reads 0.995 and the A/A copy 0.995.
- **Windowed rows (code differs by design).** 0.1.12 compiled W as a constant. A runtime W changes the attention kernel. These rows are not fixed, and no exception is claimed here.

### SASS identity against 0.1.12 (offline `cute.compile` for each pair configuration and hardware tuple, `cuobjdump -sass`)

| hardware | pair | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|
| RTX 5090 | full | identical | identical | identical | identical |
| RTX 5090 | c4096 | attention differs (3408 vs 2704 SASS), merge identical | attention differs (3544 vs 2728 SASS), merge identical | attention differs (3536 vs 2720 SASS), merge identical | attention differs (3544 vs 2728 SASS), merge identical |
| RTX 5090 | n2048 | attention differs (3384 vs 2696 SASS), merge identical | attention differs (3392 vs 2704 SASS), merge identical | attention differs (3384 vs 2704 SASS), merge identical | attention differs (3392 vs 2704 SASS), merge identical |
| RTX PRO 6000 | full | identical | identical | identical | identical |
| RTX PRO 6000 | c4096 | attention differs (3400 vs 2696 SASS), merge identical | attention differs (3536 vs 2720 SASS), merge identical | attention differs (3536 vs 2720 SASS), merge identical | attention differs (3536 vs 2720 SASS), merge identical |
| RTX PRO 6000 | n2048 | attention differs (3384 vs 2704 SASS), merge identical | attention differs (3384 vs 2704 SASS), merge identical | attention differs (3392 vs 2704 SASS), merge identical | attention differs (3384 vs 2704 SASS), merge identical |
| GB10 | full | identical | identical | identical | identical |
| GB10 | c4096 | attention differs (3816 vs 2976 SASS), merge identical | attention differs (3840 vs 3000 SASS), merge identical | attention differs (3816 vs 2976 SASS), merge identical | attention differs (3816 vs 2976 SASS), merge identical |
| GB10 | n2048 | attention differs (4096 vs 3152 SASS), merge identical | attention differs (4096 vs 3176 SASS), merge identical | attention differs (4096 vs 3160 SASS), merge identical | attention differs (4096 vs 3024 SASS), merge identical |

Full-mask pair kernels (attention and merge) are identical to 0.1.12 on all 12 configurations of each hardware tuple. Windowed kernels differ by design (runtime W, plus the L <= W walk branch). Their merge kernels are identical.

### What moved the rows (rc2 `c1a5403` to `9b8fe1a`)

- **Contiguous q keeps 0.1.12's address expression.** The first strided-q version made the row stride a runtime kernel argument. It rewrote the q address as `row * stride + (head*G + g) * D`, which changed code generation for the whole attention kernel. On the GB10 full-mask kernel: 3416 to 3440 instructions, 30 to 119 uniform-datapath instructions, 231 to 221 registers. GB10 full B1 then measured 0.992-0.998. Restoring the expression for contiguous q, and making a declared stride a compile constant, brings these kernels back to SASS identity. GB10 full B1 now measures 1.000-1.007.
- **Requests no longer than W take the unwindowed walk.** A CTA-uniform runtime branch on L <= W skips the window compares when every key is inside the window. RTX 5090 n2048 L128/L512 moved from 0.996-0.998 to 0.999-1.015, and c4096 B1 L128 from 0.997 to 1.012.
- **Tried and not kept:**
  - Per-row inclusive key bounds hoisted out of the tile loop (`ff69d69`). ptxas then spilled the sm_120 legacy kernels (16 B stack, 6-9 LDL), and RTX 5090 c4096 B1 L128 fell to 0.978. Reverted.
  - A per-tile uniform skip of the window compares for tiles inside the window. GB10 c4096 went to 12/12 rows, but GB10 n2048 did not improve (6 misses before and after). On sm_120 the skip dropped the n2048 kernel's uniform-datapath instructions from 293 to 17, which risks the RTX 5090 rows that now pass. Not committed.

### Why the pairs keep the legacy merge: IKET (RTX 5090, causal W4096, Q16, one sampled CTA)

| row | merge | attention CTA span | prologue | walk | partial store | merge phase (of which arrival wait) | separate merge kernel |
|---|---|---|---|---|---|---|---|
| B1 L4096 | legacy | 11.39 us | 0.99 | 9.25 | 0.80 | - | 1.06 us |
| B1 L4096 | distributed | 14.14 us | 0.99 | 8.96 | 0.80 | 3.04 (1.54) | - |
| B2 L512 | legacy | 5.06 us | 1.12 | 2.46 | 0.93 | - | 1.50 us |
| B2 L512 | distributed | 10.24 us | 1.15 | 2.75 | 0.96 | 4.83 (1.57) | - |

The walk itself is the same in both modes. In the distributed mode, a CTA's merge phase costs 3.0-4.8 us, of which 1.5 us is spent waiting for the other splits' arrivals. The legacy mode's separate merge kernel costs 1.1-1.5 us and overlaps through PDL. This is why the three pair shapes keep `LEGACY_MERGE_ROWS`, as the rc2 ncu counts suggested.

Tools: `run-iket` from nvidia-cutlass-dsl 4.8.0 on the RTX 5090 runner, via `draft_iket_node.py`, with one sampled cluster. Method per the CUTLASS IKET guide (https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/iket_profiling.html). SASS comparisons used `cuobjdump -sass`, `-res-usage` and uniform-instruction counts.

## Compile set and on-demand compilation

- **Compile keys.** Compilation is keyed by (causal, windowed), the capacity tier, B, Q, heads, D, page, dtype, arch, merge mode and the declared q row stride. W is never a key.
- **Cost.** A windowed layer compiles two tiers when C > 4128. For C <= 4128 there is one tier.
- **Reuse.** GPU tests construct W in {64, 2048, 3000, 4096, 8191} at one capacity. After the first windowed construction, `compile_kernels` records no cache miss.
- **Disk cache.** CuTe DSL 4.8 already keeps a file cache of compiled kernels (`CUTE_DSL_CACHE_DIR`; `CUTE_DSL_DISABLE_FILE_CACHING` turns it off). Its key covers the DSL version, the function and its compile options. Provider selection happens on the host before compilation, so the cache cannot change a selection across nodes.
- **Which variants to compile.** That is decided by the engine's warm-up: the kernel side compiles only at `DraftAttention` construction.

## Reproduce

```
python -m benchmark.kernel_tuning.draft_window_ab --arm cand=. --arm base=<0.1.12 tree> \
    --arm base2=<copy of the 0.1.12 tree> --masks n2048 --lens 128,512,4096 --pairs 60 --rounds 12
python benchmark/kernel_tuning/draft_iket_node.py --heads 32/8 --hd 128 --query 16 --causal \
    --window 4096 --batch 2 --length 512 --capacity 32768 --merge legacy
```

## Per-row tables (final code)

### RTX 5090

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | CI_high>=1 | SASS vs 0.1.12 | max abs err |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 5.28 | 5.28 | 0.998 (0.997-1.000) | 0.998 (0.997-1.001) | yes | identical | 0.0038 |
| full | 1 | 512 | 6.40 | 6.40 | 1.000 (0.999-1.002) | 1.000 (0.999-1.002) | yes | identical | 0.0011 |
| full | 1 | 4096 | 15.51 | 15.53 | 1.001 (1.000-1.003) | 1.003 (1.001-1.004) | yes | identical | 0.0003 |
| full | 2 | 128 | 6.82 | 6.82 | 0.996 (0.992-1.004) | 1.006 (0.996-1.008) | yes | identical | 0.0021 |
| full | 2 | 512 | 8.72 | 8.71 | 0.999 (0.998-1.001) | 1.000 (0.998-1.000) | yes | identical | 0.0011 |
| full | 2 | 4096 | 26.27 | 26.28 | 1.001 (0.999-1.004) | 1.001 (0.999-1.005) | yes | identical | 0.0003 |
| full | 3 | 128 | 6.91 | 6.87 | 0.993 (0.985-1.004) | 0.991 (0.986-1.002) | yes | identical | 0.0034 |
| full | 3 | 512 | 10.02 | 10.01 | 0.999 (0.998-1.000) | 1.000 (0.999-1.001) | **no** | identical | 0.0011 |
| full | 3 | 4096 | 36.38 | 36.45 | 1.000 (0.999-1.002) | 1.001 (1.000-1.005) | yes | identical | 0.0004 |
| full | 4 | 128 | 7.08 | 7.11 | 1.003 (1.002-1.005) | 1.002 (0.999-1.005) | yes | identical | 0.0021 |
| full | 4 | 512 | 11.07 | 11.07 | 1.000 (0.999-1.001) | 1.000 (0.999-1.003) | yes | identical | 0.0016 |
| full | 4 | 4096 | 46.03 | 46.06 | 1.001 (1.000-1.002) | 1.001 (1.000-1.002) | yes | identical | 0.0005 |
| c4096 | 1 | 128 | 5.37 | 5.43 | 1.012 (1.010-1.015) | 0.994 (0.992-0.995) | yes | attention differs (3408 vs 2704 SASS), merge identical | 0.0028 |
| c4096 | 1 | 512 | 6.25 | 6.30 | 1.007 (1.006-1.008) | 0.997 (0.996-1.000) | yes | attention differs (3408 vs 2704 SASS), merge identical | 0.0011 |
| c4096 | 1 | 4096 | 15.32 | 15.33 | 1.001 (1.000-1.002) | 1.001 (0.999-1.003) | yes | attention differs (3408 vs 2704 SASS), merge identical | 0.0003 |
| c4096 | 2 | 128 | 7.21 | 7.85 | 1.084 (1.081-1.093) | 1.007 (1.005-1.008) | yes | attention differs (3544 vs 2728 SASS), merge identical | 0.0023 |
| c4096 | 2 | 512 | 8.74 | 9.42 | 1.078 (1.077-1.080) | 1.000 (1.000-1.001) | yes | attention differs (3544 vs 2728 SASS), merge identical | 0.0011 |
| c4096 | 2 | 4096 | 26.73 | 27.46 | 1.026 (1.025-1.029) | 1.000 (0.997-1.003) | yes | attention differs (3544 vs 2728 SASS), merge identical | 0.0003 |
| c4096 | 3 | 128 | 10.96 | 11.12 | 1.011 (1.009-1.016) | 0.996 (0.994-1.002) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0038 |
| c4096 | 3 | 512 | 12.59 | 12.90 | 1.026 (1.024-1.026) | 1.000 (0.999-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0011 |
| c4096 | 3 | 4096 | 39.03 | 39.44 | 1.010 (1.007-1.011) | 1.000 (0.997-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0004 |
| c4096 | 4 | 128 | 10.65 | 11.04 | 1.036 (1.033-1.039) | 1.010 (0.999-1.012) | yes | attention differs (3544 vs 2728 SASS), merge identical | 0.0024 |
| c4096 | 4 | 512 | 13.31 | 13.29 | 0.998 (0.997-0.999) | 1.000 (0.999-1.001) | **no** | attention differs (3544 vs 2728 SASS), merge identical | 0.0020 |
| c4096 | 4 | 4096 | 48.66 | 48.95 | 1.006 (1.005-1.007) | 1.000 (0.999-1.001) | yes | attention differs (3544 vs 2728 SASS), merge identical | 0.0004 |
| n2048 | 1 | 128 | 5.45 | 5.54 | 1.015 (1.014-1.017) | 0.997 (0.995-1.001) | yes | attention differs (3384 vs 2696 SASS), merge identical | 0.0038 |
| n2048 | 1 | 512 | 6.30 | 6.34 | 1.007 (1.006-1.008) | 1.000 (0.998-1.000) | yes | attention differs (3384 vs 2696 SASS), merge identical | 0.0011 |
| n2048 | 1 | 4096 | 8.07 | 8.05 | 0.999 (0.994-1.002) | 1.000 (0.999-1.002) | yes | attention differs (3384 vs 2696 SASS), merge identical | 0.0005 |
| n2048 | 2 | 128 | 5.33 | 5.39 | 1.012 (1.004-1.016) | 1.007 (0.998-1.008) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0021 |
| n2048 | 2 | 512 | 7.13 | 7.16 | 1.003 (1.002-1.004) | 1.002 (1.001-1.002) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0011 |
| n2048 | 2 | 4096 | 11.23 | 11.25 | 1.000 (0.973-1.025) | 1.023 (1.003-1.029) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0005 |
| n2048 | 3 | 128 | 7.28 | 7.39 | 1.015 (1.013-1.017) | 1.001 (0.999-1.003) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0022 |
| n2048 | 3 | 512 | 9.60 | 9.59 | 0.999 (0.998-1.000) | 0.999 (0.998-1.001) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0020 |
| n2048 | 3 | 4096 | 21.64 | 21.64 | 1.001 (0.998-1.004) | 0.999 (0.996-1.001) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0006 |
| n2048 | 4 | 128 | 7.32 | 7.42 | 1.016 (1.013-1.020) | 1.003 (1.000-1.006) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0022 |
| n2048 | 4 | 512 | 10.84 | 10.83 | 0.999 (0.998-1.000) | 1.000 (0.999-1.000) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0011 |
| n2048 | 4 | 4096 | 26.30 | 26.31 | 1.000 (0.999-1.001) | 1.000 (0.999-1.001) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0005 |

Strict rule: candidate 34/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 34/36.

### RTX PRO 6000

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | CI_high>=1 | SASS vs 0.1.12 | max abs err |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 6.16 | 6.41 | 1.004 (1.002-1.025) | 1.038 (1.005-1.043) | yes | identical | 0.0021 |
| full | 1 | 512 | 7.88 | 7.85 | 0.998 (0.996-0.998) | 0.996 (0.993-1.000) | **no** | identical | 0.0011 |
| full | 1 | 4096 | 17.75 | 17.75 | 1.000 (1.000-1.001) | 1.000 (0.999-1.000) | yes | identical | 0.0003 |
| full | 2 | 128 | 7.56 | 7.58 | 1.001 (0.976-1.005) | 1.000 (0.971-1.005) | yes | identical | 0.0021 |
| full | 2 | 512 | 10.23 | 10.23 | 0.999 (0.998-1.000) | 1.000 (1.000-1.000) | **no** | identical | 0.0011 |
| full | 2 | 4096 | 29.51 | 29.54 | 1.000 (0.999-1.001) | 1.001 (0.999-1.001) | yes | identical | 0.0003 |
| full | 3 | 128 | 8.02 | 7.73 | 0.991 (0.965-1.006) | 1.022 (1.009-1.027) | yes | identical | 0.0026 |
| full | 3 | 512 | 11.60 | 11.60 | 1.000 (0.999-1.000) | 1.000 (1.000-1.000) | yes | identical | 0.0011 |
| full | 3 | 4096 | 40.66 | 40.66 | 0.999 (0.999-1.001) | 1.000 (0.999-1.000) | yes | identical | 0.0004 |
| full | 4 | 128 | 8.20 | 8.31 | 1.007 (0.995-1.009) | 1.026 (1.010-1.028) | yes | identical | 0.0038 |
| full | 4 | 512 | 12.84 | 12.85 | 1.000 (1.000-1.002) | 1.001 (1.000-1.002) | yes | identical | 0.0012 |
| full | 4 | 4096 | 51.48 | 51.47 | 1.000 (0.999-1.000) | 1.000 (0.999-1.001) | yes | identical | 0.0004 |
| c4096 | 1 | 128 | 6.30 | 6.43 | 1.021 (1.010-1.022) | 1.003 (0.988-1.007) | yes | attention differs (3400 vs 2696 SASS), merge identical | 0.0022 |
| c4096 | 1 | 512 | 7.36 | 7.45 | 1.011 (1.011-1.012) | 1.001 (1.000-1.002) | yes | attention differs (3400 vs 2696 SASS), merge identical | 0.0010 |
| c4096 | 1 | 4096 | 17.06 | 17.16 | 1.005 (1.004-1.007) | 0.999 (0.998-1.000) | yes | attention differs (3400 vs 2696 SASS), merge identical | 0.0003 |
| c4096 | 2 | 128 | 8.24 | 9.36 | 1.137 (1.128-1.145) | 1.023 (0.992-1.030) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0035 |
| c4096 | 2 | 512 | 9.99 | 10.63 | 1.063 (1.062-1.065) | 0.998 (0.997-1.000) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0010 |
| c4096 | 2 | 4096 | 29.10 | 29.75 | 1.024 (1.022-1.026) | 1.000 (0.998-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0003 |
| c4096 | 3 | 128 | 10.65 | 11.39 | 1.070 (1.069-1.070) | 1.000 (1.000-1.000) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0024 |
| c4096 | 3 | 512 | 13.60 | 13.88 | 1.021 (1.020-1.021) | 1.000 (0.999-1.000) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0012 |
| c4096 | 3 | 4096 | 41.91 | 42.05 | 1.005 (1.004-1.007) | 1.000 (0.999-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0005 |
| c4096 | 4 | 128 | 11.18 | 11.88 | 1.064 (1.061-1.066) | 1.000 (0.999-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0038 |
| c4096 | 4 | 512 | 14.74 | 14.98 | 1.017 (1.016-1.018) | 0.999 (0.999-1.000) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0011 |
| c4096 | 4 | 4096 | 52.79 | 53.09 | 1.007 (1.005-1.007) | 1.000 (0.999-1.001) | yes | attention differs (3536 vs 2720 SASS), merge identical | 0.0003 |
| n2048 | 1 | 128 | 6.76 | 6.90 | 1.020 (1.015-1.022) | 1.010 (1.006-1.013) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0020 |
| n2048 | 1 | 512 | 8.00 | 8.00 | 0.996 (0.995-0.997) | 1.002 (0.996-1.004) | **no** | attention differs (3384 vs 2704 SASS), merge identical | 0.0011 |
| n2048 | 1 | 4096 | 9.80 | 9.77 | 0.997 (0.995-0.999) | 0.998 (0.994-1.001) | **no** | attention differs (3384 vs 2704 SASS), merge identical | 0.0005 |
| n2048 | 2 | 128 | 6.65 | 6.66 | 1.001 (0.981-1.005) | 0.996 (0.974-1.000) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0020 |
| n2048 | 2 | 512 | 9.21 | 9.19 | 0.998 (0.996-1.000) | 1.004 (1.002-1.005) | **no** | attention differs (3384 vs 2704 SASS), merge identical | 0.0010 |
| n2048 | 2 | 4096 | 13.88 | 13.78 | 0.992 (0.990-0.997) | 0.999 (0.998-1.001) | **no** | attention differs (3384 vs 2704 SASS), merge identical | 0.0005 |
| n2048 | 3 | 128 | 8.58 | 8.39 | 1.007 (0.975-1.013) | 1.017 (1.012-1.035) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0028 |
| n2048 | 3 | 512 | 11.70 | 11.67 | 0.998 (0.997-0.999) | 1.001 (1.000-1.002) | **no** | attention differs (3392 vs 2704 SASS), merge identical | 0.0011 |
| n2048 | 3 | 4096 | 24.09 | 24.09 | 0.999 (0.997-1.000) | 0.999 (0.998-1.001) | yes | attention differs (3392 vs 2704 SASS), merge identical | 0.0005 |
| n2048 | 4 | 128 | 8.58 | 8.79 | 1.021 (1.017-1.023) | 1.022 (1.015-1.023) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0021 |
| n2048 | 4 | 512 | 12.97 | 12.98 | 1.000 (1.000-1.001) | 1.000 (0.999-1.000) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0019 |
| n2048 | 4 | 4096 | 29.40 | 29.41 | 1.000 (0.999-1.001) | 1.000 (0.999-1.001) | yes | attention differs (3384 vs 2704 SASS), merge identical | 0.0005 |

Strict rule: candidate 29/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 34/36.

### GB10

| pair | B | L | candidate us | 0.1.12 us | candidate speed (95% CI) | A/A: 0.1.12 copy (95% CI) | CI_high>=1 | SASS vs 0.1.12 | max abs err |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 128 | 7.35 | 7.39 | 1.007 (1.000-1.012) | 0.985 (0.978-1.005) | yes | identical | 0.0023 |
| full | 1 | 512 | 15.04 | 15.05 | 1.000 (0.999-1.002) | 1.001 (0.999-1.002) | yes | identical | 0.0011 |
| full | 1 | 4096 | 81.86 | 81.75 | 1.000 (0.998-1.002) | 1.002 (0.998-1.004) | yes | identical | 0.0003 |
| full | 2 | 128 | 11.86 | 11.94 | 1.010 (0.993-1.034) | 0.993 (0.988-1.018) | yes | identical | 0.0022 |
| full | 2 | 512 | 26.61 | 26.50 | 0.995 (0.994-0.997) | 0.995 (0.994-0.997) | **no** | identical | 0.0010 |
| full | 2 | 4096 | 170.11 | 170.36 | 1.001 (0.999-1.003) | 1.004 (1.002-1.006) | yes | identical | 0.0005 |
| full | 3 | 128 | 12.80 | 12.72 | 0.995 (0.972-0.999) | 1.006 (0.977-1.010) | **no** | identical | 0.0023 |
| full | 3 | 512 | 34.70 | 34.74 | 1.002 (1.000-1.004) | 1.002 (1.001-1.004) | yes | identical | 0.0015 |
| full | 3 | 4096 | 235.65 | 235.65 | 1.000 (0.999-1.001) | 1.000 (0.999-1.001) | yes | identical | 0.0005 |
| full | 4 | 128 | 16.78 | 17.31 | 1.028 (0.995-1.040) | 1.012 (0.997-1.033) | yes | identical | 0.0030 |
| full | 4 | 512 | 44.81 | 44.86 | 1.000 (0.999-1.004) | 1.002 (1.000-1.004) | yes | identical | 0.0011 |
| full | 4 | 4096 | 311.99 | 312.14 | 1.000 (0.999-1.001) | 1.001 (0.999-1.001) | yes | identical | 0.0004 |
| c4096 | 1 | 128 | 9.34 | 9.50 | 1.019 (1.013-1.023) | 0.999 (0.995-1.003) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0023 |
| c4096 | 1 | 512 | 16.85 | 16.97 | 1.007 (1.005-1.008) | 0.998 (0.997-1.000) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0010 |
| c4096 | 1 | 4096 | 83.57 | 83.43 | 1.001 (0.997-1.005) | 1.000 (0.997-1.004) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0003 |
| c4096 | 2 | 128 | 11.06 | 11.26 | 1.017 (1.011-1.022) | 1.001 (0.997-1.005) | yes | attention differs (3840 vs 3000 SASS), merge identical | 0.0023 |
| c4096 | 2 | 512 | 25.94 | 26.12 | 1.007 (1.004-1.010) | 1.003 (1.000-1.004) | yes | attention differs (3840 vs 3000 SASS), merge identical | 0.0012 |
| c4096 | 2 | 4096 | 166.80 | 166.40 | 0.997 (0.995-0.998) | 0.998 (0.997-0.999) | **no** | attention differs (3840 vs 3000 SASS), merge identical | 0.0003 |
| c4096 | 3 | 128 | 12.91 | 13.24 | 1.023 (0.980-1.031) | 0.997 (0.949-1.010) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0023 |
| c4096 | 3 | 512 | 35.18 | 35.19 | 1.001 (0.999-1.003) | 1.005 (1.003-1.007) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0014 |
| c4096 | 3 | 4096 | 238.05 | 238.33 | 1.001 (1.000-1.002) | 1.000 (0.999-1.001) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0005 |
| c4096 | 4 | 128 | 17.00 | 17.58 | 1.035 (1.025-1.044) | 1.007 (0.995-1.016) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0036 |
| c4096 | 4 | 512 | 45.03 | 45.29 | 1.004 (1.003-1.008) | 0.997 (0.993-1.000) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0011 |
| c4096 | 4 | 4096 | 316.18 | 316.99 | 1.003 (1.002-1.004) | 1.001 (1.000-1.003) | yes | attention differs (3816 vs 2976 SASS), merge identical | 0.0004 |
| n2048 | 1 | 128 | 7.46 | 7.38 | 1.008 (1.000-1.034) | 0.992 (0.979-1.000) | yes | attention differs (4096 vs 3152 SASS), merge identical | 0.0021 |
| n2048 | 1 | 512 | 15.22 | 15.30 | 1.006 (1.005-1.007) | 1.000 (0.999-1.001) | yes | attention differs (4096 vs 3152 SASS), merge identical | 0.0010 |
| n2048 | 1 | 4096 | 42.82 | 42.66 | 0.997 (0.995-0.998) | 0.998 (0.996-0.999) | **no** | attention differs (4096 vs 3152 SASS), merge identical | 0.0004 |
| n2048 | 2 | 128 | 8.79 | 8.88 | 1.017 (0.989-1.029) | 0.997 (0.983-1.020) | yes | attention differs (4096 vs 3176 SASS), merge identical | 0.0023 |
| n2048 | 2 | 512 | 22.86 | 23.05 | 1.007 (1.005-1.010) | 1.005 (1.003-1.007) | yes | attention differs (4096 vs 3176 SASS), merge identical | 0.0009 |
| n2048 | 2 | 4096 | 79.37 | 79.17 | 0.997 (0.995-0.999) | 0.999 (0.998-1.001) | **no** | attention differs (4096 vs 3176 SASS), merge identical | 0.0005 |
| n2048 | 3 | 128 | 9.91 | 10.19 | 1.011 (0.979-1.060) | 1.009 (0.958-1.055) | yes | attention differs (4096 vs 3160 SASS), merge identical | 0.0023 |
| n2048 | 3 | 512 | 33.40 | 33.19 | 0.994 (0.993-0.996) | 0.998 (0.996-0.999) | **no** | attention differs (4096 vs 3160 SASS), merge identical | 0.0011 |
| n2048 | 3 | 4096 | 119.78 | 119.70 | 0.996 (0.993-0.999) | 0.998 (0.994-1.001) | **no** | attention differs (4096 vs 3160 SASS), merge identical | 0.0008 |
| n2048 | 4 | 128 | 8.90 | 8.77 | 0.984 (0.964-1.000) | 0.991 (0.981-0.998) | **no** | attention differs (4096 vs 3024 SASS), merge identical | 0.0023 |
| n2048 | 4 | 512 | 39.46 | 39.41 | 0.998 (0.995-1.001) | 0.999 (0.998-1.001) | yes | attention differs (4096 vs 3024 SASS), merge identical | 0.0011 |
| n2048 | 4 | 4096 | 152.98 | 153.15 | 1.002 (1.001-1.002) | 1.000 (0.999-1.001) | yes | attention differs (4096 vs 3024 SASS), merge identical | 0.0005 |

Strict rule: candidate 28/36 rows with CI_high >= 1. A/A (two identical 0.1.12 copies): 31/36.
