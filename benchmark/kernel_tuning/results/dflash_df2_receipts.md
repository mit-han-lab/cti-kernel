<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DF2 receipts: CuTe DSL DRAFT attention

DF2 replaces FlashInfer underneath `DraftAttention` with a CuTe DSL split-KV
provider (`op.provider == "cute-splitkv"`, `python/ctikernels/attention/_draft/cute/`).
It is the default on sm120/sm121 for head dimension 128 and Q·(Hq/Hkv) ≤ 128;
FlashInfer FA2 with the frozen Triton device schedule stays the fallback
(`provider="flashinfer"` or `CTI_DRAFT_PROVIDER=flashinfer`).

**Superseded for speed and kernel counts by `dflash_df2_market_gate.md`** (installed
0.1.11rc2 wheel, gate against stock FlashInfer, SGLang 72730d6 and vLLM 0.31.0;
prepare folded into the first layer's attention).

## Candidate

| field | value |
| --- | --- |
| branch | `kernels/dflash-kernels` (local, not pushed) |
| GPU test and GB10 benchmark source | `6e7c110f484bd6ab568988875c027efac6eb8f5a` (worktree source, not an installed wheel) |
| RTX 5090 / RTX PRO 6000 benchmark source | `afe9996463f89a963bf860745d5e6d12253240e4` (+ RTX 5090 Q9 rerun at the same SHA); the sm120 provider code is unchanged since (later commits: a test fix, recording tuning attributes, and a GB10-only change that was reverted) |
| environment | torch 2.13.0+cu130, CUDA 13.0, nvidia-cutlass-dsl 4.8.0, flashinfer-python 0.6.11.post3 (stock comparator and fallback), BF16, page size 1 for speed (tests also P=16) |
| devices | RTX 5090 (sm120), RTX PRO 6000 Blackwell Server (sm120), GB10 (sm121) |

## Design

- **prepare** (1 kernel, shared by the layers of a round): writes the engine-owned
  query/page CSR, page indices and last-page lengths exactly as in DF1. Attention
  does not read them; it reads the token CSR directly.
- **attention** (1 kernel): grid (S, Hkv·row tiles, B). The Q·G query rows of a
  (request, KV head) are packed (row = qi·G + g). Each CTA owns one key range of the
  request's visible union. Its token indices are staged in shared memory once at
  the start (chunk ≤ 1024 keys), then K/V tiles are gathered with 16-byte
  `cp.async` into a 2-stage ring. S = QKᵀ and O += PV run on `mma.sync`
  m16n8k16 BF16 with FP32 accumulation and an online log2-domain softmax.
  Masks are applied per element from the frozen K0.17 predicates. With few rows,
  two key groups split each tile and are combined in shared memory. A single
  non-empty split writes BF16 output directly; otherwise normalized FP32 partials
  and LSE are written with 16-byte stores.
- **merge** (1 kernel, compiled only when S > 1): one warp per (row, KV head), with
  lane-parallel LSE and vectorized partial loads. It writes zeros for slot −1 rows
  and does nothing for rows that ran a single split.
- Every kernel launches with PDL and reads metadata only after `griddepcontrol.wait`.
  S is fixed at construction; lengths, slots and physical tokens are device inputs
  on every replay (capture-safe with changing metadata).
- Defaults (from RTX 5090 / RTX PRO 6000 / GB10 sweeps): S = min(⌈union/64⌉,
  ⌊SMs / (B·Hkv·row tiles)⌋) so the CTAs fit one wave, raised to ⌈union/1024⌉.
  Key tile 32 (sm120) or 64 (GB10); 2 stages. Two key groups for ≤ 48 packed rows
  on sm120, at every row count on GB10. All of these can be overridden through
  `CTI_DRAFT_*` environment variables for sweeps.

Kernels per call (this round): complete op 3 (prepare, attention, merge) when S > 1,
else 2; per-layer `run()` 2 or 1. Since `bfc554b` the complete op `prepare_and_run()` is
2 (1 when S = 1): the attention CTAs write the prepare outputs. Stock FlashInfer fixed-plan graph: 2 (attention, merge).

## Correctness (GPU, source 6e7c110)

Suites: `test_dflash_draft_attention.py`, `test_dflash_draft_cute.py`,
`test_dflash_padding.py`, `test_dflash_integration.py`, `test_fixed_query_attention.py`,
`test_dflash_capture_trace.py`, `test_dflash_cute_prepare.py`. The first two run both
providers. They cover Q ∈ {4,5,8,9,12,13,16,17}, the three masks with their
2047/2048/2049 and 4095/4096/4097 boundaries, B=1–4, 32/8 and 16/4 heads,
P = 1 and 16, forced split counts 1/2/7/auto, graph replay with changing lengths,
slots and positions, all-inactive batches, NaN-poisoned unused pages, and the
frozen 2e-2 oracle tolerance. Capture traces and integration run the default
(CuTe) provider. The CuTe speculative helpers are tested bitwise against the
frozen Triton kernels.

| device | result |
| --- | --- |
| RTX 5090 | 825 passed |
| RTX PRO 6000 | 825 passed |
| GB10 | 825 passed |

CPU: 1572 passed; pre-commit clean.

## Speed

Protocol: `benchmark/kernel_tuning/dflash_draft_attention.py --queries Q --contexts 128 512 4096`
(window boundaries added automatically), B=1–4, 32/8 and 16/4 heads, 20 warm-up
calls, then 20 interleaved candidate/reference pairs of 100 synchronized calls.
ε is the p95 |ratio−1| over 20 reference self-pairs, fixed before the pairs.
**Graph gate (DF2):** the candidate's complete op (device prepare + run, CUDA-graph
replay) against the stock FlashInfer `BatchPrefillWithPagedKVCacheWrapper` fa2
fixed-plan CUDA graph (host plan outside timing; it cannot follow changing
metadata). A row passes when median(candidate/reference) ≤ 1 + ε. Run-only is the
per-layer cost when one prepare serves the round (informational). The eager gate
against stock plan+run passes on every row on all three devices.
SGLang `72730d6` DRAFT comparator: N/A (engine-bound path with host-known replay
lengths; no standalone device-prepare adapter), never counted as PASS.
SOL is informational: the shared sol-ceilings tables, with an L2-reuse reference
(ceiling definition recorded per row).

**RTX 5090** — source `afe9996463f89a963bf860745d5e6d12253240e4`, torch 2.13.0+cu130, CUDA 13.0, flashinfer-python 0.6.11.post3, capability (12, 0).

| Q | mask | rows | correctness | graph gate PASS | complete/stock | run-only/stock | SOL util (graph complete) | limiting | kernels/call graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40/40 | 40/40 | 0.57–1.00 | 0.55–1.00 | 14–53% | floor/l2 | 3 |
| 4 | full | 40 | 40/40 | 40/40 | 0.57–1.00 | 0.51–1.00 | 14–53% | floor/l2 | 3 |
| 4 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.58–1.00 | 0.54–1.00 | 14–43% | floor/l2 | 3 |
| 5 | causal_sliding | 40 | 40/40 | 40/40 | 0.40–0.86 | 0.40–0.86 | 14–53% | floor/l2 | 3 |
| 5 | full | 40 | 40/40 | 40/40 | 0.40–0.86 | 0.40–0.86 | 14–53% | floor/l2 | 3 |
| 5 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.40–0.86 | 0.40–0.86 | 14–42% | floor/l2 | 3 |
| 8 | causal_sliding | 40 | 40/40 | 40/40 | 0.40–0.86 | 0.40–0.86 | 14–53% | floor/l2 | 3 |
| 8 | full | 40 | 40/40 | 40/40 | 0.40–1.00 | 0.36–0.86 | 14–54% | floor/l2 | 3 |
| 8 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.40–1.00 | 0.39–0.87 | 14–43% | floor/l2 | 3 |
| 9 | causal_sliding | 40 | 40/40 | 37/40 | 0.41–1.12 | 0.40–0.98 | 11–40% | floor/l2 | 3 |
| 9 | full | 40 | 40/40 | 37/40 | 0.41–1.14 | 0.40–1.00 | 11–43% | floor/l2 | 3 |
| 9 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.42–0.86 | 0.40–0.86 | 11–34% | floor/l2 | 3 |
| 12 | causal_sliding | 40 | 40/40 | 40/40 | 0.40–1.00 | 0.40–1.00 | 11–42% | floor/l2/tensor | 3 |
| 12 | full | 40 | 40/40 | 40/40 | 0.40–1.00 | 0.40–1.00 | 11–44% | floor/l2/tensor | 3 |
| 12 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.44–0.86 | 0.40–0.86 | 11–35% | floor/l2/tensor | 3 |
| 16 | causal_sliding | 40 | 40/40 | 40/40 | 0.46–1.00 | 0.40–1.00 | 11–48% | floor/tensor | 3 |
| 16 | full | 40 | 40/40 | 40/40 | 0.46–1.00 | 0.38–1.00 | 11–49% | floor/tensor | 3 |
| 16 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.46–1.00 | 0.40–0.86 | 11–41% | floor/tensor | 3 |

Total RTX 5090: 768 rows, correctness 768/768, graph gate 762/768 PASS.


**RTX PRO 6000** — source `afe9996463f89a963bf860745d5e6d12253240e4`, torch 2.13.0+cu130, CUDA 13.0, flashinfer-python 0.6.11.post3, capability (12, 0).

| Q | mask | rows | correctness | graph gate PASS | complete/stock | run-only/stock | SOL util (graph complete) | limiting | kernels/call graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40/40 | 36/40 | 0.83–1.09 | 0.67–1.00 | 14–53% | floor/l2 | 3 |
| 4 | full | 40 | 40/40 | 38/40 | 0.81–1.09 | 0.67–1.02 | 14–57% | floor/l2 | 3 |
| 4 | noncausal_sliding | 48 | 48/48 | 43/48 | 0.83–1.08 | 0.67–1.08 | 14–41% | floor/l2 | 3 |
| 5 | causal_sliding | 40 | 40/40 | 39/40 | 0.63–1.12 | 0.50–1.00 | 14–53% | floor/l2 | 3 |
| 5 | full | 40 | 40/40 | 36/40 | 0.63–1.14 | 0.50–1.00 | 14–57% | floor/l2 | 3 |
| 5 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.63–1.00 | 0.50–1.00 | 14–43% | floor/l2 | 3 |
| 8 | causal_sliding | 40 | 40/40 | 40/40 | 0.56–1.00 | 0.45–0.88 | 14–53% | floor/l2 | 3 |
| 8 | full | 40 | 40/40 | 40/40 | 0.56–1.00 | 0.45–0.88 | 14–56% | floor/l2 | 3 |
| 8 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.57–1.00 | 0.46–0.86 | 14–43% | floor/l2 | 3 |
| 9 | causal_sliding | 40 | 40/40 | 35/40 | 0.58–1.04 | 0.50–1.00 | 14–42% | floor/l2 | 3 |
| 9 | full | 40 | 40/40 | 35/40 | 0.57–1.06 | 0.49–1.00 | 14–45% | floor/l2 | 3 |
| 9 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.62–1.00 | 0.50–0.92 | 14–34% | floor/l2 | 3 |
| 12 | causal_sliding | 40 | 40/40 | 40/40 | 0.56–1.00 | 0.50–1.00 | 14–43% | floor/l2 | 3 |
| 12 | full | 40 | 40/40 | 40/40 | 0.56–1.00 | 0.50–1.00 | 14–46% | floor/l2 | 3 |
| 12 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.56–0.95 | 0.52–0.94 | 14–35% | floor/l2 | 3 |
| 16 | causal_sliding | 40 | 40/40 | 37/40 | 0.56–1.13 | 0.56–1.00 | 13–40% | floor/l2 | 3 |
| 16 | full | 40 | 40/40 | 38/40 | 0.56–1.11 | 0.50–1.00 | 13–43% | floor/l2 | 3 |
| 16 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.56–0.93 | 0.56–0.86 | 13–35% | floor/l2 | 3 |

Total RTX PRO 6000: 768 rows, correctness 768/768, graph gate 737/768 PASS.


**GB10** — source `6e7c110f484bd6ab568988875c027efac6eb8f5a`, torch 2.13.0+cu130, CUDA 13.0, flashinfer-python 0.6.11.post3, capability (12, 1).

| Q | mask | rows | correctness | graph gate PASS | complete/stock | run-only/stock | SOL util (graph complete) | limiting | kernels/call graph |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40/40 | 26/40 | 0.70–1.14 | 0.61–1.12 | 15–55% | dram/floor/l2 | 2/3 |
| 4 | full | 40 | 40/40 | 26/40 | 0.67–1.13 | 0.62–1.09 | 15–55% | dram/floor/l2 | 2/3 |
| 4 | noncausal_sliding | 48 | 48/48 | 46/48 | 0.70–1.11 | 0.63–1.09 | 15–63% | dram/floor/l2 | 2/3 |
| 5 | causal_sliding | 40 | 40/40 | 27/40 | 0.50–1.14 | 0.45–1.12 | 15–57% | dram/floor/l2 | 2/3 |
| 5 | full | 40 | 40/40 | 27/40 | 0.48–1.15 | 0.45–1.14 | 15–57% | dram/floor/l2 | 2/3 |
| 5 | noncausal_sliding | 48 | 48/48 | 46/48 | 0.50–1.09 | 0.45–1.07 | 15–58% | dram/floor/l2 | 2/3 |
| 8 | causal_sliding | 40 | 40/40 | 38/40 | 0.51–1.05 | 0.49–1.04 | 8–54% | dram/floor/l2 | 2/3 |
| 8 | full | 40 | 40/40 | 38/40 | 0.46–1.06 | 0.50–1.05 | 11–58% | dram/floor/l2 | 2/3 |
| 8 | noncausal_sliding | 48 | 48/48 | 48/48 | 0.51–1.05 | 0.48–1.05 | 8–56% | dram/floor/l2 | 2/3 |
| 9 | causal_sliding | 40 | 40/40 | 29/40 | 0.59–1.25 | 0.49–1.17 | 12–53% | dram/floor/l2 | 2/3 |
| 9 | full | 40 | 40/40 | 27/40 | 0.53–1.13 | 0.49–1.13 | 12–53% | dram/floor/l2 | 2/3 |
| 9 | noncausal_sliding | 48 | 48/48 | 42/48 | 0.59–1.22 | 0.50–1.13 | 12–46% | dram/floor/l2 | 2/3 |
| 12 | causal_sliding | 40 | 40/40 | 34/40 | 0.46–1.13 | 0.43–1.12 | 12–53% | dram/floor/l2 | 2/3 |
| 12 | full | 40 | 40/40 | 34/40 | 0.43–1.12 | 0.42–1.05 | 12–53% | dram/floor/l2 | 2/3 |
| 12 | noncausal_sliding | 48 | 48/48 | 45/48 | 0.45–1.11 | 0.44–1.10 | 12–49% | dram/floor/l2 | 2/3 |
| 16 | causal_sliding | 40 | 40/40 | 33/40 | 0.46–1.13 | 0.39–1.07 | 12–52% | dram/floor/l2 | 2/3 |
| 16 | full | 40 | 40/40 | 32/40 | 0.42–1.16 | 0.39–1.07 | 12–52% | dram/floor/l2 | 2/3 |
| 16 | noncausal_sliding | 48 | 48/48 | 44/48 | 0.46–1.13 | 0.39–1.01 | 12–46% | dram/floor/l2 | 2/3 |

Total GB10: 768 rows, correctness 768/768, graph gate 642/768 PASS.


#### Rows failing the graph gate

| device | Q | mask | B | L | Hq/Hkv | complete/stock | ε | candidate µs | stock µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | 9 | causal_sliding | 1 | 4095 | 32/8 | 1.055 | 0.016 | 16.5 | 15.6 |
| RTX 5090 | 9 | causal_sliding | 1 | 4096 | 32/8 | 1.103 | 0.012 | 16.5 | 14.9 |
| RTX 5090 | 9 | causal_sliding | 1 | 4097 | 32/8 | 1.122 | 0.016 | 16.5 | 14.7 |
| RTX 5090 | 9 | full | 1 | 4095 | 32/8 | 1.126 | 0.006 | 16.5 | 14.6 |
| RTX 5090 | 9 | full | 1 | 4096 | 32/8 | 1.136 | 0.006 | 16.5 | 14.5 |
| RTX 5090 | 9 | full | 1 | 4097 | 32/8 | 1.134 | 0.006 | 16.4 | 14.5 |
| RTX PRO 6000 | 4 | causal_sliding | 3 | 4095 | 32/8 | 1.003 | 0.002 | 20.6 | 20.6 |
| RTX PRO 6000 | 4 | causal_sliding | 4 | 4095 | 32/8 | 1.091 | 0.013 | 24.6 | 22.6 |
| RTX PRO 6000 | 4 | causal_sliding | 4 | 4096 | 32/8 | 1.091 | 0.026 | 24.6 | 22.6 |
| RTX PRO 6000 | 4 | causal_sliding | 4 | 4097 | 32/8 | 1.091 | 0.019 | 24.6 | 22.6 |
| RTX PRO 6000 | 4 | full | 4 | 4095 | 32/8 | 1.090 | 0.001 | 24.6 | 22.6 |
| RTX PRO 6000 | 4 | full | 4 | 4096 | 32/8 | 1.023 | 0.013 | 22.6 | 22.2 |
| RTX PRO 6000 | 4 | noncausal_sliding | 4 | 2047 | 32/8 | 1.041 | 0.002 | 17.1 | 16.5 |
| RTX PRO 6000 | 4 | noncausal_sliding | 4 | 2048 | 32/8 | 1.049 | 0.003 | 17.3 | 16.5 |
| RTX PRO 6000 | 4 | noncausal_sliding | 4 | 2048 | 16/4 | 1.035 | 0.019 | 13.4 | 12.9 |
| RTX PRO 6000 | 4 | noncausal_sliding | 4 | 2049 | 32/8 | 1.076 | 0.002 | 17.7 | 16.5 |
| RTX PRO 6000 | 4 | noncausal_sliding | 4 | 4096 | 32/8 | 1.062 | 0.005 | 17.5 | 16.5 |
| RTX PRO 6000 | 5 | causal_sliding | 2 | 4096 | 16/4 | 1.116 | 0.003 | 14.1 | 12.6 |
| RTX PRO 6000 | 5 | full | 2 | 4095 | 16/4 | 1.036 | 0.012 | 14.4 | 13.9 |
| RTX PRO 6000 | 5 | full | 2 | 4096 | 32/8 | 1.137 | 0.023 | 16.4 | 14.5 |
| RTX PRO 6000 | 5 | full | 2 | 4096 | 16/4 | 1.052 | 0.014 | 14.4 | 13.7 |
| RTX PRO 6000 | 5 | full | 2 | 4097 | 32/8 | 1.140 | 0.013 | 16.4 | 14.4 |
| RTX PRO 6000 | 9 | causal_sliding | 1 | 4095 | 32/8 | 1.004 | 0.001 | 14.5 | 14.4 |
| RTX PRO 6000 | 9 | causal_sliding | 1 | 4095 | 16/4 | 1.022 | 0.010 | 14.4 | 14.1 |
| RTX PRO 6000 | 9 | causal_sliding | 1 | 4096 | 16/4 | 1.043 | 0.024 | 14.4 | 13.8 |
| RTX PRO 6000 | 9 | causal_sliding | 1 | 4097 | 32/8 | 1.002 | 0.002 | 14.4 | 14.4 |
| RTX PRO 6000 | 9 | causal_sliding | 1 | 4097 | 16/4 | 1.018 | 0.015 | 14.4 | 14.1 |
| RTX PRO 6000 | 9 | full | 1 | 4095 | 32/8 | 1.002 | 0.001 | 14.4 | 14.4 |
| RTX PRO 6000 | 9 | full | 1 | 4095 | 16/4 | 1.045 | 0.018 | 14.4 | 13.8 |
| RTX PRO 6000 | 9 | full | 1 | 4096 | 32/8 | 1.002 | 0.001 | 14.4 | 14.4 |
| RTX PRO 6000 | 9 | full | 1 | 4096 | 16/4 | 1.063 | 0.019 | 14.4 | 13.6 |
| RTX PRO 6000 | 9 | full | 1 | 4097 | 16/4 | 1.063 | 0.014 | 14.4 | 13.6 |
| RTX PRO 6000 | 16 | causal_sliding | 1 | 4095 | 32/8 | 1.005 | 0.002 | 16.5 | 16.4 |
| RTX PRO 6000 | 16 | causal_sliding | 1 | 4096 | 32/8 | 1.135 | 0.007 | 16.5 | 14.5 |
| RTX PRO 6000 | 16 | causal_sliding | 1 | 4097 | 32/8 | 1.120 | 0.016 | 16.4 | 14.7 |
| RTX PRO 6000 | 16 | full | 1 | 4096 | 32/8 | 1.111 | 0.017 | 16.4 | 14.8 |
| RTX PRO 6000 | 16 | full | 1 | 4097 | 32/8 | 1.087 | 0.013 | 16.4 | 15.1 |
| GB10 | 4 | causal_sliding | 2 | 4095 | 32/8 | 1.115 | 0.046 | 143.7 | 128.8 |
| GB10 | 4 | causal_sliding | 2 | 4096 | 32/8 | 1.122 | 0.046 | 144.1 | 128.6 |
| GB10 | 4 | causal_sliding | 2 | 4097 | 32/8 | 1.120 | 0.039 | 145.5 | 130.4 |
| GB10 | 4 | causal_sliding | 3 | 4095 | 32/8 | 1.038 | 0.026 | 236.9 | 228.3 |
| GB10 | 4 | causal_sliding | 3 | 4096 | 32/8 | 1.032 | 0.032 | 236.7 | 229.4 |
| GB10 | 4 | causal_sliding | 3 | 4096 | 16/4 | 1.054 | 0.051 | 78.5 | 73.7 |
| GB10 | 4 | causal_sliding | 3 | 4097 | 32/8 | 1.034 | 0.007 | 236.6 | 228.6 |
| GB10 | 4 | causal_sliding | 3 | 4097 | 16/4 | 1.042 | 0.018 | 77.3 | 73.9 |
| GB10 | 4 | causal_sliding | 4 | 4095 | 32/8 | 1.051 | 0.034 | 316.0 | 300.5 |
| GB10 | 4 | causal_sliding | 4 | 4095 | 16/4 | 1.140 | 0.042 | 150.3 | 129.9 |
| GB10 | 4 | causal_sliding | 4 | 4096 | 32/8 | 1.052 | 0.023 | 317.9 | 300.8 |
| GB10 | 4 | causal_sliding | 4 | 4096 | 16/4 | 1.132 | 0.038 | 147.5 | 130.6 |
| GB10 | 4 | causal_sliding | 4 | 4097 | 32/8 | 1.054 | 0.025 | 316.2 | 299.9 |
| GB10 | 4 | causal_sliding | 4 | 4097 | 16/4 | 1.135 | 0.045 | 147.2 | 129.9 |
| GB10 | 4 | full | 1 | 4095 | 32/8 | 1.129 | 0.123 | 27.4 | 24.2 |
| GB10 | 4 | full | 1 | 4096 | 32/8 | 1.105 | 0.028 | 27.4 | 24.8 |
| GB10 | 4 | full | 2 | 4095 | 32/8 | 1.120 | 0.035 | 143.7 | 128.3 |
| GB10 | 4 | full | 2 | 4096 | 32/8 | 1.118 | 0.041 | 144.2 | 129.0 |
| GB10 | 4 | full | 2 | 4097 | 32/8 | 1.124 | 0.034 | 145.7 | 130.0 |
| GB10 | 4 | full | 3 | 4095 | 32/8 | 1.040 | 0.024 | 237.6 | 228.5 |
| GB10 | 4 | full | 3 | 4096 | 32/8 | 1.037 | 0.019 | 237.4 | 228.9 |
| GB10 | 4 | full | 3 | 4097 | 32/8 | 1.033 | 0.029 | 237.2 | 229.5 |
| GB10 | 4 | full | 4 | 4095 | 32/8 | 1.072 | 0.023 | 322.2 | 300.1 |
| GB10 | 4 | full | 4 | 4095 | 16/4 | 1.111 | 0.024 | 147.5 | 133.0 |
| GB10 | 4 | full | 4 | 4096 | 32/8 | 1.055 | 0.036 | 316.3 | 299.8 |
| GB10 | 4 | full | 4 | 4096 | 16/4 | 1.103 | 0.027 | 146.8 | 133.1 |
| GB10 | 4 | full | 4 | 4097 | 32/8 | 1.056 | 0.022 | 316.4 | 300.0 |
| GB10 | 4 | full | 4 | 4097 | 16/4 | 1.105 | 0.045 | 146.9 | 132.8 |
| GB10 | 4 | noncausal_sliding | 3 | 4096 | 32/8 | 1.110 | 0.052 | 76.1 | 68.6 |
| GB10 | 4 | noncausal_sliding | 4 | 4096 | 32/8 | 1.077 | 0.042 | 149.3 | 139.0 |
| GB10 | 5 | causal_sliding | 2 | 4095 | 32/8 | 1.108 | 0.040 | 157.3 | 140.9 |
| GB10 | 5 | causal_sliding | 2 | 4096 | 32/8 | 1.116 | 0.038 | 157.5 | 140.2 |
| GB10 | 5 | causal_sliding | 2 | 4097 | 32/8 | 1.123 | 0.043 | 159.8 | 140.6 |
| GB10 | 5 | causal_sliding | 3 | 4095 | 32/8 | 1.031 | 0.030 | 239.4 | 232.1 |
| GB10 | 5 | causal_sliding | 3 | 4095 | 16/4 | 1.098 | 0.073 | 74.0 | 67.3 |
| GB10 | 5 | causal_sliding | 3 | 4096 | 32/8 | 1.035 | 0.025 | 239.5 | 231.3 |
| GB10 | 5 | causal_sliding | 3 | 4096 | 16/4 | 1.084 | 0.046 | 73.8 | 68.2 |
| GB10 | 5 | causal_sliding | 3 | 4097 | 32/8 | 1.043 | 0.040 | 246.0 | 234.4 |
| GB10 | 5 | causal_sliding | 3 | 4097 | 16/4 | 1.095 | 0.051 | 74.1 | 68.1 |
| GB10 | 5 | causal_sliding | 4 | 4095 | 16/4 | 1.137 | 0.027 | 162.6 | 141.4 |
| GB10 | 5 | causal_sliding | 4 | 4096 | 32/8 | 1.055 | 0.040 | 327.3 | 311.0 |
| GB10 | 5 | causal_sliding | 4 | 4096 | 16/4 | 1.135 | 0.026 | 163.1 | 143.5 |
| GB10 | 5 | causal_sliding | 4 | 4097 | 16/4 | 1.130 | 0.054 | 162.7 | 142.4 |
| GB10 | 5 | full | 2 | 4095 | 32/8 | 1.135 | 0.031 | 157.4 | 138.2 |
| GB10 | 5 | full | 2 | 4096 | 32/8 | 1.134 | 0.037 | 159.2 | 138.6 |
| GB10 | 5 | full | 2 | 4097 | 32/8 | 1.152 | 0.042 | 157.3 | 137.1 |
| GB10 | 5 | full | 3 | 4095 | 32/8 | 1.031 | 0.022 | 240.4 | 234.4 |
| GB10 | 5 | full | 3 | 4095 | 16/4 | 1.103 | 0.047 | 73.5 | 66.6 |
| GB10 | 5 | full | 3 | 4096 | 32/8 | 1.049 | 0.021 | 244.7 | 232.3 |
| GB10 | 5 | full | 3 | 4096 | 16/4 | 1.101 | 0.055 | 73.9 | 67.0 |
| GB10 | 5 | full | 3 | 4097 | 32/8 | 1.038 | 0.023 | 241.4 | 232.1 |
| GB10 | 5 | full | 3 | 4097 | 16/4 | 1.107 | 0.040 | 76.0 | 67.2 |
| GB10 | 5 | full | 4 | 4095 | 32/8 | 1.042 | 0.038 | 323.1 | 310.7 |
| GB10 | 5 | full | 4 | 4095 | 16/4 | 1.113 | 0.060 | 160.6 | 144.8 |
| GB10 | 5 | full | 4 | 4096 | 16/4 | 1.108 | 0.054 | 159.8 | 145.0 |
| GB10 | 5 | full | 4 | 4097 | 16/4 | 1.127 | 0.049 | 162.3 | 143.8 |
| GB10 | 5 | noncausal_sliding | 3 | 4096 | 32/8 | 1.093 | 0.047 | 86.0 | 78.8 |
| GB10 | 5 | noncausal_sliding | 4 | 4096 | 32/8 | 1.082 | 0.042 | 161.3 | 148.2 |
| GB10 | 8 | causal_sliding | 3 | 4095 | 32/8 | 1.033 | 0.032 | 250.9 | 244.7 |
| GB10 | 8 | causal_sliding | 4 | 4097 | 32/8 | 1.038 | 0.036 | 335.6 | 322.6 |
| GB10 | 8 | full | 4 | 4095 | 32/8 | 1.043 | 0.035 | 332.4 | 319.0 |
| GB10 | 8 | full | 4 | 4096 | 32/8 | 1.037 | 0.029 | 333.9 | 318.2 |
| GB10 | 9 | causal_sliding | 1 | 4095 | 16/4 | 1.248 | 0.163 | 20.5 | 16.4 |
| GB10 | 9 | causal_sliding | 1 | 4096 | 16/4 | 1.248 | 0.114 | 20.5 | 16.4 |
| GB10 | 9 | causal_sliding | 1 | 4097 | 16/4 | 1.249 | 0.070 | 20.5 | 16.4 |
| GB10 | 9 | causal_sliding | 2 | 4095 | 32/8 | 1.062 | 0.042 | 155.5 | 147.1 |
| GB10 | 9 | causal_sliding | 2 | 4096 | 32/8 | 1.061 | 0.037 | 158.4 | 148.1 |
| GB10 | 9 | causal_sliding | 3 | 4095 | 32/8 | 1.052 | 0.043 | 253.0 | 240.3 |
| GB10 | 9 | causal_sliding | 3 | 4096 | 32/8 | 1.040 | 0.028 | 250.4 | 241.5 |
| GB10 | 9 | causal_sliding | 3 | 4097 | 32/8 | 1.044 | 0.033 | 253.4 | 243.2 |
| GB10 | 9 | causal_sliding | 4 | 4095 | 32/8 | 1.067 | 0.022 | 334.2 | 315.3 |
| GB10 | 9 | causal_sliding | 4 | 4096 | 32/8 | 1.062 | 0.036 | 333.5 | 315.9 |
| GB10 | 9 | causal_sliding | 4 | 4097 | 32/8 | 1.072 | 0.036 | 335.4 | 314.3 |
| GB10 | 9 | full | 1 | 4096 | 16/4 | 1.128 | 0.054 | 18.5 | 16.4 |
| GB10 | 9 | full | 1 | 4097 | 16/4 | 1.127 | 0.045 | 18.5 | 16.4 |
| GB10 | 9 | full | 2 | 4095 | 32/8 | 1.066 | 0.033 | 158.2 | 147.5 |
| GB10 | 9 | full | 2 | 4096 | 32/8 | 1.053 | 0.048 | 155.7 | 149.9 |
| GB10 | 9 | full | 2 | 4097 | 32/8 | 1.071 | 0.048 | 157.8 | 146.8 |
| GB10 | 9 | full | 3 | 4095 | 16/4 | 1.116 | 0.070 | 93.1 | 83.0 |
| GB10 | 9 | full | 3 | 4096 | 32/8 | 1.042 | 0.028 | 251.4 | 241.0 |
| GB10 | 9 | full | 3 | 4096 | 16/4 | 1.127 | 0.071 | 95.0 | 82.6 |
| GB10 | 9 | full | 3 | 4097 | 32/8 | 1.032 | 0.031 | 250.8 | 243.2 |
| GB10 | 9 | full | 3 | 4097 | 16/4 | 1.116 | 0.074 | 92.6 | 83.6 |
| GB10 | 9 | full | 4 | 4095 | 32/8 | 1.063 | 0.022 | 334.8 | 315.3 |
| GB10 | 9 | full | 4 | 4096 | 32/8 | 1.053 | 0.020 | 334.1 | 315.4 |
| GB10 | 9 | full | 4 | 4097 | 32/8 | 1.049 | 0.026 | 333.4 | 315.9 |
| GB10 | 9 | noncausal_sliding | 1 | 2047 | 32/8 | 1.213 | 0.056 | 19.9 | 16.4 |
| GB10 | 9 | noncausal_sliding | 1 | 2047 | 16/4 | 1.130 | 0.018 | 14.6 | 12.9 |
| GB10 | 9 | noncausal_sliding | 1 | 2048 | 32/8 | 1.210 | 0.146 | 19.9 | 16.4 |
| GB10 | 9 | noncausal_sliding | 1 | 2049 | 32/8 | 1.219 | 0.163 | 20.0 | 16.4 |
| GB10 | 9 | noncausal_sliding | 1 | 4096 | 32/8 | 1.197 | 0.089 | 19.7 | 16.4 |
| GB10 | 9 | noncausal_sliding | 1 | 4096 | 16/4 | 1.196 | 0.095 | 15.2 | 12.7 |
| GB10 | 12 | causal_sliding | 3 | 4096 | 32/8 | 1.030 | 0.030 | 252.4 | 244.7 |
| GB10 | 12 | causal_sliding | 3 | 4097 | 32/8 | 1.036 | 0.030 | 252.4 | 241.4 |
| GB10 | 12 | causal_sliding | 4 | 4095 | 32/8 | 1.057 | 0.035 | 336.3 | 318.6 |
| GB10 | 12 | causal_sliding | 4 | 4096 | 32/8 | 1.054 | 0.023 | 334.3 | 319.1 |
| GB10 | 12 | causal_sliding | 4 | 4096 | 16/4 | 1.052 | 0.035 | 162.0 | 151.7 |
| GB10 | 12 | causal_sliding | 4 | 4097 | 32/8 | 1.047 | 0.020 | 337.0 | 320.0 |
| GB10 | 12 | full | 1 | 4095 | 16/4 | 1.124 | 0.074 | 18.5 | 16.4 |
| GB10 | 12 | full | 4 | 4095 | 32/8 | 1.054 | 0.026 | 335.5 | 318.3 |
| GB10 | 12 | full | 4 | 4096 | 32/8 | 1.057 | 0.021 | 336.4 | 318.2 |
| GB10 | 12 | full | 4 | 4096 | 16/4 | 1.075 | 0.039 | 161.4 | 149.5 |
| GB10 | 12 | full | 4 | 4097 | 32/8 | 1.048 | 0.026 | 334.5 | 320.0 |
| GB10 | 12 | full | 4 | 4097 | 16/4 | 1.056 | 0.047 | 161.0 | 152.5 |
| GB10 | 12 | noncausal_sliding | 1 | 2047 | 32/8 | 1.109 | 0.081 | 18.5 | 16.7 |
| GB10 | 12 | noncausal_sliding | 1 | 2049 | 32/8 | 1.110 | 0.078 | 18.5 | 16.7 |
| GB10 | 12 | noncausal_sliding | 3 | 4096 | 32/8 | 1.095 | 0.072 | 90.8 | 83.2 |
| GB10 | 16 | causal_sliding | 1 | 4095 | 16/4 | 1.129 | 0.102 | 20.5 | 18.2 |
| GB10 | 16 | causal_sliding | 1 | 4096 | 16/4 | 1.128 | 0.103 | 20.5 | 18.2 |
| GB10 | 16 | causal_sliding | 1 | 4097 | 16/4 | 1.131 | 0.019 | 20.5 | 18.2 |
| GB10 | 16 | causal_sliding | 3 | 4095 | 32/8 | 1.041 | 0.028 | 258.8 | 251.4 |
| GB10 | 16 | causal_sliding | 3 | 4096 | 32/8 | 1.037 | 0.027 | 258.9 | 251.0 |
| GB10 | 16 | causal_sliding | 4 | 4095 | 32/8 | 1.067 | 0.055 | 345.8 | 326.5 |
| GB10 | 16 | causal_sliding | 4 | 4097 | 32/8 | 1.064 | 0.041 | 349.4 | 327.7 |
| GB10 | 16 | full | 1 | 4095 | 16/4 | 1.163 | 0.118 | 20.0 | 17.2 |
| GB10 | 16 | full | 1 | 4096 | 16/4 | 1.164 | 0.040 | 20.1 | 17.3 |
| GB10 | 16 | full | 1 | 4097 | 16/4 | 1.162 | 0.129 | 20.0 | 17.3 |
| GB10 | 16 | full | 3 | 4095 | 32/8 | 1.036 | 0.026 | 257.9 | 249.9 |
| GB10 | 16 | full | 4 | 4095 | 32/8 | 1.067 | 0.022 | 347.8 | 326.2 |
| GB10 | 16 | full | 4 | 4096 | 32/8 | 1.063 | 0.045 | 348.2 | 326.1 |
| GB10 | 16 | full | 4 | 4096 | 16/4 | 1.059 | 0.043 | 165.4 | 155.5 |
| GB10 | 16 | full | 4 | 4097 | 32/8 | 1.059 | 0.050 | 345.9 | 326.7 |
| GB10 | 16 | noncausal_sliding | 1 | 2047 | 32/8 | 1.111 | 0.095 | 20.5 | 18.5 |
| GB10 | 16 | noncausal_sliding | 1 | 2049 | 32/8 | 1.110 | 0.036 | 20.5 | 18.5 |
| GB10 | 16 | noncausal_sliding | 1 | 2049 | 16/4 | 1.126 | 0.068 | 16.2 | 14.4 |
| GB10 | 16 | noncausal_sliding | 1 | 4096 | 16/4 | 1.133 | 0.081 | 16.3 | 14.4 |

## Remaining distance (not met in this step)

- **SOL ≥ 90 % at B=1–4 is not reached:** graph-complete utilization is 8–63 %.
  Short contexts are bounded by the 3 graph nodes; at L≈4096 the kernel runs at
  roughly a third of the BF16 MMA rate. An ncu SASS profile showed one warp per
  scheduler with dependent HMMA chains and softmax that does not overlap the MMAs.
- **Graph-gate failures** (table above) cluster at L≈4096:
  - B=1 with 32/8 heads and Q ≥ 9, on all devices. The extra prepare node costs
    about 1 µs; run-only is ≤ 1.00× on the RTX 5090.
  - DRAM-bound B ≥ 2 rows on GB10 (1.03–1.14×).
  - B=3–4 rows with Q4/Q5 on RTX PRO 6000 (≤ 1.14×).
- Next levers: overlap softmax with MMA (two warpgroups ping-ponging, or 8 warps
  with deeper key groups), fold the prepare work into the first layer's attention
  launch, and give GB10 more bytes in flight per CTA (a 2×SMs target regressed B=1
  and was reverted).
- The CuTe port of FlashInfer's FA2 split-KV schedule was removed: it never
  completed on GPU (a bisect showed a hang even when it returned right after its
  first block reduction). The FlashInfer fallback keeps the frozen Triton schedule.

These historical counts include the retired K0 CuTe port. DF2 now retains
only DRAFT attention coverage; see `docs/dflash_draft_attention.md` for
the independent test, capture and access-check entry points. R5 owns
round bookkeeping. The historical GPU counts do not qualify this rewrite.
