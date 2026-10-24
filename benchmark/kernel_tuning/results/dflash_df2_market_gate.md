<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DF2: DRAFT market gate (0.1.11rc2)

This round qualifies the CuTe DSL DRAFT provider from an installed wheel against
the kernels each market stack actually dispatches, and records the per-row
provider dispatch decision. Earlier DF2 receipts: `dflash_df2_receipts.md`.

## Candidate

| field | value |
| --- | --- |
| wheel | `cti_kernels-0.1.11rc2-py3-none-any.whl`, sha256 `3b854611e00ae36aecfed4cfeb5f5f30190af249194ea63c72ae8a12af4a1b55` |
| source | `bfc554b30b28984e182186fbcbc34b5a5b060e8f` (branch `kernels/dflash-kernels`, local) |
| environment | torch 2.13.0+cu130, CUDA 13.0, nvidia-cutlass-dsl 4.8.0, BF16 |
| devices | RTX 5090 (sm120), RTX PRO 6000 Blackwell Server (sm120), GB10 (sm121; timing processes pinned to the Cortex-X925 cores) |

The GPU tests, the memcheck run and the gate import `ctikernels` from the installed
wheel (the worktree's `python/` is removed on the node); only `tests/` and
`benchmark/` come from the source tree.

## Change in this round: prepare folded into the first layer's attention

`prepare_and_run()` no longer forks a prepare launch. After its PDL wait, each
request's attention CTAs write that request's query/page CSR, page indices and
last-page lengths with Prepare's arithmetic (a runtime flag; `run()` passes 0, so
per-layer calls are unchanged). Attention still never reads that CSR.
Kernels per complete op: **2** (attention, merge), or 1 when S = 1; `prepare(); run()`
stays 3 (2). Every gate row below measured 2 kernels per call.

## Correctness and memory checks (installed rc2 wheel)

| device | GPU tests | compute-sanitizer memcheck (CuTe access matrix + K0 rounds) |
| --- | --- | --- |
| RTX 5090 | 877 passed | 129/129 cases, `ERROR SUMMARY: 0 errors` |
| RTX PRO 6000 | 877 passed | 129/129 cases, `ERROR SUMMARY: 0 errors` |
| GB10 | 877 passed | 129/129 cases, `ERROR SUMMARY: 0 errors` |

Test files: `test_dflash_draft_attention.py`, `test_dflash_draft_cute.py`,
`test_dflash_padding.py`, `test_dflash_integration.py`, `test_fixed_query_attention.py`,
`test_dflash_capture_trace.py`, `test_dflash_cute_prepare.py`. The fused-entry test now
covers P = 1 and 16 and a forced single split, so the page-index loop strides
(more pages than publishing threads). The memcheck matrix alternates
`prepare(); run()` and `prepare_and_run()` per construction and checks the CSR on
every graph replay. Misaligned (16-byte) Q/K/V/out views are rejected before any
launch (`test_operand_base_pointer_alignment`, storage-offset views).
CPU suite: 1648 passed at `e59cf6a` (a tests-only commit after the wheel source); pre-commit clean.

## Market gate protocol

`python -m benchmark.kernel_tuning.dflash_draft_market` (one subprocess per comparator).
Candidate: the dispatched `DraftAttention` complete op (`prepare_and_run()`, CUDA-graph
replay with device metadata). Comparators, each a fixed CUDA graph with host planning
outside timing (they cannot follow changing metadata):

- **stock FlashInfer** 0.6.11.post3 `BatchPrefillWithPagedKVCacheWrapper` fa2, P = 1;
- **SGLang 72730d6** DRAFT dispatch: FlashInfer 0.7.1 fa2, P = 1, sliding layers keep
  the last W−1 context tokens plus the block (`flashinfer_backend.py:310,:516,:985,:1372,:1970`);
- **vLLM 0.31.0** (db9527a) DRAFT dispatch on sm12x: `vllm_flash_attn` FA2 `varlen_fwd`
  (fork 9cd61de), P = 16, windows (−1,−1) / (W−1,0) / (W−1,W−1)
  (`flash_attn.py:1450,:587`, `fa_utils.py:98`).

Rows: masks full / causal sliding W=4096 / non-causal sliding W=2048, Q ∈ {4,5,6,8,9,12,16},
B = 1–4, contexts 128, 512, 4096 plus the window boundaries (4095/4096/4097,
2047/2048/2049), heads 32/8 and 16/4, head_dim 128: 896 rows per device. After 20
warm-up calls, ε is the p95 |ratio − 1| of 20 reference self-pairs; 20 interleaved
candidate/reference pairs of 100 synchronized calls each; a row passes against a
comparator when median(candidate/reference) ≤ 1 + ε and both outputs pass the
independent dense oracle (atol = rtol = 2e-2). **The bar is every comparator,
i.e. the faster of stock FlashInfer, SGLang and vLLM.** SOL is informational.

Timing note: on every device, per-replay times fall on discrete steps (about 2.05 µs on
the sm120 hosts, about 1 µs on GB10), for the candidate and every comparator alike.
Short rows where two stacks land one step apart therefore flip between runs and
processes (for example, the vLLM reference at L = 128 on RTX PRO 6000 measured 8.3 µs
in one process and 10.3 µs in another).

## Gate results (rc2 wheel)

Gate: last eligible record per key/comparator, in lexical file and line order; `.log.try` follows `.log`. All three comparators are required; missing records fail. Correctness counts candidate PASS in all three correctness fields.

Candidate latency uses FlashInfer's candidate measurement, else SGLang, else vLLM. Fastest market is the minimum measured reference latency. SOL uses the same candidate record and kernel count. N/A denotes missing measurements; ! marks failure or SOL above 100%.

Ceiling definition: Informational useful-work empirical reference from benchmark/sol/ceilings/{rtx5090,rtx-pro-6000,gb10}.json; serial graph kernels, no PDL: sum(max(graph floor, DRAM time, BF16 MMA time)). One attention node carries BF16 Q reads, union-visible K/V reads per KV head, one int32 page index per union key and BF16 output writes; FLOPs = 4*B*Hq*128*sum(query visible-key counts). Remaining launches are floor-only nodes. The benchmark replays identical buffers, so min(1, L2 size/working set) of every attention byte is served at the measured L2 read rate (warm_l2 when it all fits, else partial_l2), the rest at the sustained DRAM read rate as an aggregate traffic proxy; cold_t_ref_us keeps the all-DRAM reference. Excludes padding, scalar work, scratch traffic and eager host overhead. Utilization is t_ref/measured latency, with ratios above one flagged.

This receipt uses the L2-aware **t_ref_us / candidate_median_us** (warm_l2 or partial_l2).

### rtx5090

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl; vllm_so_sha256=db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31

| Q | mask | rows | correctness | PASS vs FlashInfer | PASS vs SGLang | PASS vs vLLM | PASS vs all | candidate/fastest-market ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5200–0.9632 | 10.0%–62.0% | {2} |
| 4 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5032–0.9983 | 9.5%–62.1% | {2} |
| 4 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.6041–0.8207 | 9.6%–45.9% | {2} |
| 5 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4019–0.8570 | 12.1%–62.2% | {2} |
| 5 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4027–0.8573 | 9.5%–62.2% | {2} |
| 5 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4034–0.7512 | 11.0%–46.0% | {2} |
| 6 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4021–0.8565 | 12.2%–62.2% | {2} |
| 6 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4019–0.8569 | 12.5%–62.2% | {2} |
| 6 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4016–0.7511 | 11.3%–46.1% | {2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4010–0.8559 | 9.7%–62.1% | {2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4011–0.8474 | 9.5%–62.3% | {2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4011–0.7994 | 9.8%–46.3% | {2} |
| 9 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4017–0.9988 | 9.4%–44.7% | {2} |
| 9 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4018–0.9973 | 9.4%–46.5% | {2} |
| 9 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4018–0.9963 | 9.5%–36.1% | {2} |
| 12 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4021–0.9979 | 9.5%–45.8% | {2} |
| 12 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4018–0.9976 | 9.5%–46.4% | {2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4017–0.8028 | 9.4%–36.7% | {2} |
| 16 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4015–0.8774 | 9.5%–53.0% | {2} |
| 16 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4021–0.8771 | 9.5%–53.4% | {2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4021–0.8573 | 9.4%–39.6% | {2} |
| **Total** | all | 896 | 896 | 896 | 896 | 896 | 896 | 0.4010–0.9988 | 9.4%–62.3% | {2} |

Totals: rows=896, correct=896, pass_all=896, fail_all=0.

### rtx-pro-6000

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl; vllm_so_sha256=db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7

| Q | mask | rows | correctness | PASS vs FlashInfer | PASS vs SGLang | PASS vs vLLM | PASS vs all | candidate/fastest-market ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.6690–0.9996 | 11.7%–67.4% | {2} |
| 4 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.6684–0.9974 | 11.7%–67.4% | {2} |
| 4 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.6688–0.9984 | 11.7%–45.3% | {2} |
| 5 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5020–0.9974 | 11.7%–66.9% | {2} |
| 5 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5017–0.9975 | 11.7%–67.6% | {2} |
| 5 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5025–0.9978 | 11.7%–45.4% | {2} |
| 6 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5017–0.8763 | 11.7%–67.7% | {2} |
| 6 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5021–0.9963 | 11.7%–67.7% | {2} |
| 6 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5019–0.9980 | 11.7%–46.0% | {2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5004–0.9564 | 11.7%–67.4% | {2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4994–0.9969 | 11.7%–67.7% | {2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5022–0.9999 | 11.7%–45.6% | {2} |
| 9 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5032–1.0004 | 9.6%–51.0% | {2} |
| 9 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5032–1.0006 | 9.6%–54.7% | {2} |
| 9 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.6112–1.0007 | 9.4%–40.0% | {2} |
| 12 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5167–1.0006 | 9.5%–51.2% | {2} |
| 12 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5032–1.0006 | 10.0%–51.2% | {2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5580–1.0008 | 9.4%–40.3% | {2} |
| 16 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5572–1.0019 | 9.4%–46.3% | {2} |
| 16 | full | 40 | 40 | 40 | 40 | 38 | 38 | 0.5581–1.2435 | 9.4%–47.5% | {2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 45 | 45 | 0.5573–1.2445 | 9.4%–36.1% | {2} |
| **Total** | all | 896 | 896 | 896 | 896 | 891 | 891 | 0.4994–1.2445 | 9.4%–67.7% | {2} |

Totals: rows=896, correct=896, pass_all=891, fail_all=5.

### gb10

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_aarch64.whl; vllm_so_sha256=be52450ee26044a3903082a726e05d2f0f0acbdcce994e94e12c6f54274a2246; ctikernels_source_sha=bfc554b30b28984e182186fbcbc34b5a5b060e8f; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19

| Q | mask | rows | correctness | PASS vs FlashInfer | PASS vs SGLang | PASS vs vLLM | PASS vs all | candidate/fastest-market ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 35 | 35 | 40 | 33 | 0.6169–1.1863 | 10.0%–66.9% | {1, 2} |
| 4 | full | 40 | 40 | 37 | 35 | 40 | 34 | 0.5757–1.1766 | 10.0%–67.6% | {1, 2} |
| 4 | noncausal_sliding | 48 | 48 | 48 | 47 | 48 | 47 | 0.6385–1.2825 | 10.0%–67.6% | {1, 2} |
| 5 | causal_sliding | 40 | 40 | 36 | 38 | 40 | 35 | 0.4612–1.2258 | 10.0%–67.6% | {1, 2} |
| 5 | full | 40 | 40 | 37 | 37 | 40 | 36 | 0.4610–1.2492 | 10.0%–67.6% | {1, 2} |
| 5 | noncausal_sliding | 48 | 48 | 48 | 47 | 48 | 47 | 0.4674–1.2217 | 10.0%–68.0% | {1, 2} |
| 6 | causal_sliding | 40 | 40 | 39 | 37 | 40 | 36 | 0.4847–1.1448 | 10.0%–66.4% | {1, 2} |
| 6 | full | 40 | 40 | 40 | 38 | 40 | 38 | 0.4457–1.1768 | 10.0%–67.7% | {1, 2} |
| 6 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4780–1.1760 | 10.0%–68.1% | {1, 2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4076–1.0991 | 9.9%–65.4% | {1, 2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3983–1.0895 | 10.0%–67.9% | {1, 2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4015–1.1198 | 10.0%–66.2% | {1, 2} |
| 9 | causal_sliding | 40 | 40 | 37 | 38 | 40 | 36 | 0.5010–1.1765 | 9.9%–59.2% | {1, 2} |
| 9 | full | 40 | 40 | 38 | 39 | 40 | 38 | 0.4562–1.1239 | 9.9%–59.0% | {1, 2} |
| 9 | noncausal_sliding | 48 | 48 | 44 | 45 | 48 | 42 | 0.5035–1.1341 | 9.4%–51.5% | {1, 2} |
| 12 | causal_sliding | 40 | 40 | 40 | 39 | 40 | 39 | 0.4557–1.0974 | 9.9%–58.3% | {1, 2} |
| 12 | full | 40 | 40 | 38 | 38 | 40 | 37 | 0.4571–1.1875 | 9.9%–58.8% | {1, 2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4559–1.0947 | 9.3%–51.9% | {1, 2} |
| 16 | causal_sliding | 40 | 40 | 38 | 39 | 40 | 38 | 0.5390–1.1103 | 8.0%–59.9% | {1, 2} |
| 16 | full | 40 | 40 | 39 | 40 | 38 | 37 | 0.4720–1.1738 | 8.2%–57.6% | {1, 2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5004–1.0534 | 8.0%–48.3% | {1, 2} |
| **Total** | all | 896 | 896 | 866 | 864 | 894 | 845 | 0.3983–1.2825 | 8.0%–68.1% | {1, 2} |

Totals: rows=896, correct=896, pass_all=845, fail_all=51.

### Failing gate rows

Comparator cells are median paired ratio (ε); ! means status != PASS. Reruns never change the gate. Override device comes from JSON when available, else filename markers 5090/pro/gb. Only AB provider cute/auto with empty env qualifies.

| device | mask | B | Q | L | Hq/Hkv | FlashInfer ratio (ε) | SGLang ratio (ε) | vLLM ratio (ε) | candidate µs | fastest market µs | rerun |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rtx-pro-6000 | full | 3 | 16 | 128 | 32/8 | 0.5569 (0.0010) | 0.5568 (0.0016) | 1.2430 (0.0028) ! | 10.3290 | 8.3064 | — |
| rtx-pro-6000 | full | 4 | 16 | 128 | 16/4 | 0.5564 (0.0040) | 0.5570 (0.0013) | 1.1276 (0.0353) ! | 10.3009 | 9.1414 | — |
| rtx-pro-6000 | noncausal_sliding | 2 | 16 | 128 | 32/8 | 0.5614 (0.0081) | 0.5598 (0.0067) | 1.2428 (0.0021) ! | 10.3084 | 8.3053 | — |
| rtx-pro-6000 | noncausal_sliding | 4 | 16 | 128 | 16/4 | 0.5925 (0.0139) | 0.5770 (0.0106) | 1.1634 (0.0475) ! | 10.2960 | 8.8718 | — |
| rtx-pro-6000 | noncausal_sliding | 4 | 16 | 128 | 32/8 | 0.5572 (0.0012) | 0.5569 (0.0011) | 1.2433 (0.0024) ! | 10.3318 | 8.3021 | — |
| gb10 | causal_sliding | 1 | 9 | 4095 | 16/4 | 1.0858 (0.0363) ! | 1.1063 (0.0428) ! | 0.1948 (0.0301) | 17.8423 | 16.4236 | — |
| gb10 | causal_sliding | 1 | 9 | 4096 | 16/4 | 1.1237 (0.0623) ! | 1.1238 (0.1626) | 0.2002 (0.0338) | 18.4718 | 16.4291 | — |
| gb10 | causal_sliding | 1 | 9 | 4097 | 32/8 | 1.0342 (0.1973) | 1.1465 (0.1087) ! | 0.2628 (0.0331) | 24.6107 | 23.8147 | — |
| gb10 | causal_sliding | 1 | 12 | 4095 | 32/8 | 0.9991 (0.1680) | 1.1265 (0.1101) ! | 0.2595 (0.0360) | 24.6194 | 24.6358 | — |
| gb10 | causal_sliding | 1 | 16 | 4095 | 16/4 | 1.1101 (0.0662) ! | 1.1100 (0.2099) | 0.2168 (0.0390) | 20.5179 | 18.4801 | — |
| gb10 | causal_sliding | 1 | 16 | 4096 | 16/4 | 1.1100 (0.0789) ! | 1.1059 (0.0992) ! | 0.2150 (0.0289) | 20.5063 | 18.4693 | — |
| gb10 | causal_sliding | 2 | 4 | 4095 | 32/8 | 1.0550 (0.0487) ! | 1.0173 (0.0452) | 0.6878 (0.0443) | 145.8246 | 124.9530 | — |
| gb10 | causal_sliding | 2 | 5 | 4097 | 32/8 | 1.0306 (0.0281) ! | 1.1042 (0.0365) ! | 0.6823 (0.0448) | 146.0839 | 139.0169 | — |
| gb10 | causal_sliding | 2 | 6 | 4095 | 32/8 | 0.9953 (0.0310) | 1.0699 (0.0424) ! | 0.7043 (0.0455) | 146.6827 | 143.9219 | — |
| gb10 | causal_sliding | 3 | 4 | 4096 | 16/4 | 1.1880 (0.0742) ! | 1.1356 (0.0506) ! | 0.6652 (0.1264) | 73.1327 | 61.6478 | — |
| gb10 | causal_sliding | 3 | 4 | 4097 | 16/4 | 1.1264 (0.0670) ! | 0.9645 (0.0763) | 0.6415 (0.0936) | 83.1738 | 70.6308 | — |
| gb10 | causal_sliding | 3 | 5 | 4096 | 16/4 | 1.0783 (0.0740) ! | 1.0093 (0.0467) | 0.6324 (0.0928) | 74.2903 | 68.6147 | — |
| gb10 | causal_sliding | 3 | 5 | 4097 | 16/4 | 1.2244 (0.0577) ! | 0.9890 (0.0617) | 0.6456 (0.0888) | 78.9973 | 64.4433 | — |
| gb10 | causal_sliding | 3 | 6 | 4096 | 16/4 | 1.0877 (0.0871) ! | 0.9315 (0.0653) | 0.7470 (0.0861) | 70.6606 | 65.0134 | — |
| gb10 | causal_sliding | 3 | 9 | 4096 | 16/4 | 1.1801 (0.0957) ! | 0.9090 (0.0902) | 0.7835 (0.0847) | 83.4419 | 70.9252 | — |
| gb10 | causal_sliding | 4 | 4 | 4095 | 16/4 | 1.0185 (0.0444) | 1.0261 (0.0053) ! | 0.7039 (0.0421) | 150.2216 | 147.4684 | — |
| gb10 | causal_sliding | 4 | 4 | 4095 | 32/8 | 1.0259 (0.0234) ! | 1.0231 (0.0140) ! | 0.9216 (0.0560) | 307.6484 | 299.7380 | — |
| gb10 | causal_sliding | 4 | 4 | 4096 | 16/4 | 1.0805 (0.0210) ! | 1.1024 (0.0218) ! | 0.7130 (0.0804) | 147.5879 | 133.1770 | — |
| gb10 | causal_sliding | 4 | 4 | 4097 | 16/4 | 1.0026 (0.0378) | 1.0369 (0.0354) ! | 0.7178 (0.0470) | 148.2777 | 147.8911 | — |
| gb10 | causal_sliding | 4 | 5 | 4095 | 16/4 | 1.0542 (0.0311) ! | 1.0028 (0.0483) | 0.6676 (0.0449) | 160.0522 | 147.4485 | — |
| gb10 | causal_sliding | 4 | 5 | 4096 | 16/4 | 1.0297 (0.0570) | 1.0781 (0.0415) ! | 0.7115 (0.0471) | 146.0148 | 141.7166 | — |
| gb10 | causal_sliding | 4 | 6 | 4097 | 16/4 | 0.9976 (0.0465) | 1.0525 (0.0463) ! | 0.6997 (0.0441) | 152.4763 | 150.3949 | — |
| gb10 | causal_sliding | 4 | 6 | 4097 | 32/8 | 1.0196 (0.0239) | 1.0240 (0.0193) ! | 0.9243 (0.0303) | 315.1279 | 307.8731 | — |
| gb10 | full | 1 | 9 | 4096 | 16/4 | 1.1230 (0.0413) ! | 1.1233 (0.0996) ! | 0.1990 (0.0294) | 18.4425 | 16.4240 | — |
| gb10 | full | 1 | 12 | 4095 | 16/4 | 1.1087 (0.0272) ! | 1.1219 (0.0963) ! | 0.1969 (0.0355) | 18.2367 | 16.4365 | — |
| gb10 | full | 1 | 12 | 4096 | 16/4 | 1.1243 (0.0467) ! | 1.1216 (0.1548) | 0.1987 (0.0385) | 18.4679 | 16.4260 | — |
| gb10 | full | 1 | 12 | 4097 | 32/8 | 0.9907 (0.1791) | 1.1292 (0.1015) ! | 0.2564 (0.0766) | 24.4086 | 24.6111 | — |
| gb10 | full | 1 | 16 | 128 | 16/4 | 0.8061 (0.0489) | 0.8298 (0.1314) | 1.1185 (0.0058) ! | 9.9735 | 9.1703 | — |
| gb10 | full | 1 | 16 | 128 | 32/8 | 0.7639 (0.1091) | 0.7161 (0.0288) | 1.1101 (0.0293) ! | 10.2760 | 9.2540 | — |
| gb10 | full | 2 | 4 | 4095 | 32/8 | 1.0293 (0.0353) | 1.1035 (0.0185) ! | 0.6848 (0.0505) | 145.5199 | 139.3011 | — |
| gb10 | full | 2 | 4 | 4096 | 32/8 | 1.0753 (0.0389) ! | 1.1020 (0.0405) ! | 0.7095 (0.0766) | 147.7767 | 125.5914 | — |
| gb10 | full | 2 | 6 | 4097 | 32/8 | 0.9931 (0.0403) | 1.0905 (0.0285) ! | 0.7403 (0.0477) | 146.9667 | 140.9891 | — |
| gb10 | full | 3 | 4 | 4095 | 16/4 | 1.0369 (0.0630) | 1.0633 (0.0592) ! | 0.7279 (0.1516) | 68.5240 | 66.7013 | — |
| gb10 | full | 3 | 4 | 4096 | 16/4 | 1.1009 (0.0713) ! | 1.1272 (0.0658) ! | 0.7073 (0.0855) | 72.7421 | 66.0679 | — |
| gb10 | full | 3 | 5 | 4096 | 16/4 | 1.2856 (0.0946) ! | 1.1040 (0.0693) ! | 0.7952 (0.0839) | 75.4071 | 60.3642 | — |
| gb10 | full | 3 | 5 | 4097 | 16/4 | 1.0783 (0.0576) ! | 0.9470 (0.0810) | 0.6565 (0.0933) | 77.7835 | 64.2579 | — |
| gb10 | full | 3 | 6 | 4096 | 16/4 | 1.0423 (0.0778) | 1.0925 (0.0819) ! | 0.6254 (0.0921) | 67.1006 | 57.7839 | — |
| gb10 | full | 3 | 9 | 4096 | 16/4 | 1.0962 (0.0887) ! | 1.0248 (0.0701) | 0.7833 (0.0861) | 77.6336 | 69.3585 | — |
| gb10 | full | 4 | 4 | 4097 | 16/4 | 1.0289 (0.0300) | 1.0486 (0.0247) ! | 0.7287 (0.0489) | 152.4018 | 146.5752 | — |
| gb10 | full | 4 | 4 | 4097 | 32/8 | 1.0252 (0.0191) ! | 1.0232 (0.0236) | 0.9223 (0.0298) | 307.9135 | 300.0867 | — |
| gb10 | full | 4 | 5 | 4095 | 32/8 | 1.0279 (0.0258) ! | 1.0213 (0.0201) ! | 0.9307 (0.0819) | 315.5639 | 307.3768 | — |
| gb10 | full | 4 | 5 | 4096 | 16/4 | 1.0275 (0.0541) | 1.0581 (0.0433) ! | 0.7263 (0.0461) | 151.1298 | 145.6512 | — |
| gb10 | full | 4 | 16 | 4096 | 16/4 | 1.0494 (0.0310) ! | 1.0222 (0.0362) | 0.7665 (0.0406) | 167.3676 | 158.8203 | — |
| gb10 | noncausal_sliding | 1 | 9 | 2047 | 16/4 | 1.1121 (0.0244) ! | 1.1345 (0.1240) ! | 0.2923 (0.0593) | 14.3745 | 12.6746 | — |
| gb10 | noncausal_sliding | 1 | 9 | 2048 | 16/4 | 1.1219 (0.0776) ! | 1.0904 (0.1255) | 0.2923 (0.0422) | 14.3765 | 12.8217 | — |
| gb10 | noncausal_sliding | 1 | 9 | 2049 | 32/8 | 1.0689 (0.1910) | 1.1154 (0.0416) ! | 0.3418 (0.0480) | 18.4638 | 16.4356 | — |
| gb10 | noncausal_sliding | 1 | 9 | 4096 | 16/4 | 1.0599 (0.0504) ! | 1.0771 (0.1391) | 0.2807 (0.0554) | 14.3942 | 13.3567 | — |
| gb10 | noncausal_sliding | 1 | 9 | 4096 | 32/8 | 1.1136 (0.0526) ! | 1.1230 (0.1252) | 0.3381 (0.0537) | 18.3435 | 16.4436 | — |
| gb10 | noncausal_sliding | 3 | 4 | 2048 | 32/8 | 0.9780 (0.0679) | 1.1252 (0.0765) ! | 0.8071 (0.0910) | 69.3456 | 64.3146 | — |
| gb10 | noncausal_sliding | 3 | 5 | 2048 | 32/8 | 1.0111 (0.0659) | 1.1572 (0.0691) ! | 0.7505 (0.1316) | 84.0503 | 70.3875 | — |
| gb10 | noncausal_sliding | 3 | 9 | 2048 | 32/8 | 0.9149 (0.0570) | 1.1423 (0.0939) ! | 0.7325 (0.0813) | 84.5448 | 79.5457 | — |

### Rerun receipts (failed gate keys only)

| device | mask | B | Q | L | Hq/Hkv | rerun vs all (ratios and ε) |
| --- | --- | --- | --- | --- | --- | --- |

## Per-row dispatch decision

`python/ctikernels/attention/_draft/dispatch.py` routes a row (keyed by device
capability and SM count, B, Q, Hq/Hkv, mask and the key-union bound) to the FlashInfer
fallback only where paired measurements show it faster. **The table is empty: every row
dispatches CuTe on all three devices.** Evidence (worktree source `f172e55`, before the
prepare fold, which leaves the FlashInfer provider unchanged; same-process pairs against
the stock FlashInfer graph):

- RTX 5090 (11 rows) and RTX PRO 6000 (15 rows) where CuTe was within 3 % of stock
  FlashInfer: the FlashInfer provider (Triton device prepare + FA2) measured
  1.12–1.28× stock on every row; CuTe 0.86–1.00×.
- GB10, the 35 rows failing the rc1 gate: the FlashInfer provider was worse than CuTe
  on 29 by paired ratio. The other 6 were rerun three times each. Only full B=3 Q=4
  16/4 at L≈4096 favored it (1.05/1.01/0.99× vs CuTe 1.18/1.12/0.98×). Through the
  dispatched path that row then measured 1.056–1.062× stock (no better than CuTe),
  and its SGLang pair cannot be measured in-process (the SGLang worker's FlashInfer
  0.7.1 shadows the provider's FlashInfer). The rule was reverted.
- Forcing a single split (one kernel) on all 168 L = 128 rows: worse on 114 rows on the
  RTX 5090 and on 46 on the RTX PRO 6000, tied elsewhere; not adopted.

The "CuTe only where ≥ 1.03× faster than FlashInfer" rule cannot be met by
routing: the FlashInfer *provider* (complete op, device prepare included) is slower
than the stock fixed-plan graph on every measured row, so a routed row would lose.


### Paired provider evidence

| device | mask | B | Q | L | Hq/Hkv | CuTe / stock FI | FI provider / stock FI | CuTe µs | FI provider µs | faster |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | causal_sliding | 1 | 9 | 4095 | 32/8 | 1.001 | 1.188 | 14.4 | 17.1 | CuTe |
| RTX 5090 | causal_sliding | 1 | 9 | 4096 | 32/8 | 0.998 | 1.253 | 14.4 | 18.1 | CuTe |
| RTX 5090 | causal_sliding | 1 | 9 | 4097 | 32/8 | 0.997 | 1.269 | 14.4 | 18.3 | CuTe |
| RTX 5090 | causal_sliding | 1 | 12 | 4095 | 32/8 | 0.880 | 1.195 | 14.4 | 18.5 | CuTe |
| RTX 5090 | causal_sliding | 1 | 12 | 4097 | 32/8 | 0.899 | 1.182 | 14.4 | 18.5 | CuTe |
| RTX 5090 | full | 1 | 4 | 4095 | 16/4 | 0.844 | 1.276 | 10.3 | 14.4 | CuTe |
| RTX 5090 | full | 1 | 4 | 4095 | 32/8 | 0.838 | 1.169 | 10.3 | 14.4 | CuTe |
| RTX 5090 | full | 1 | 4 | 4096 | 16/4 | 0.879 | 1.218 | 10.3 | 14.4 | CuTe |
| RTX 5090 | full | 1 | 9 | 128 | 32/8 | 0.833 | 1.004 | 8.2 | 10.3 | CuTe |
| RTX 5090 | full | 1 | 9 | 4096 | 32/8 | 1.001 | 1.164 | 14.4 | 16.8 | CuTe |
| RTX 5090 | full | 1 | 12 | 4095 | 32/8 | 0.882 | 1.208 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | causal_sliding | 1 | 6 | 4097 | 16/4 | 0.961 | 1.142 | 12.4 | 16.4 | CuTe |
| RTX PRO 6000 | causal_sliding | 1 | 9 | 4096 | 32/8 | 0.932 | 1.275 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | causal_sliding | 1 | 9 | 4097 | 32/8 | 0.936 | 1.269 | 14.5 | 18.5 | CuTe |
| RTX PRO 6000 | causal_sliding | 1 | 12 | 4097 | 32/8 | 0.879 | 1.126 | 14.5 | 18.5 | CuTe |
| RTX PRO 6000 | full | 1 | 6 | 4097 | 16/4 | 0.859 | 1.144 | 12.4 | 16.5 | CuTe |
| RTX PRO 6000 | full | 1 | 9 | 4095 | 32/8 | 0.995 | 1.282 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | full | 1 | 9 | 4096 | 32/8 | 0.988 | 1.281 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | full | 1 | 12 | 4096 | 32/8 | 0.878 | 1.126 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | full | 1 | 12 | 4097 | 32/8 | 0.880 | 1.126 | 14.4 | 18.5 | CuTe |
| RTX PRO 6000 | full | 2 | 4 | 4096 | 16/4 | 0.860 | 1.155 | 12.4 | 16.6 | CuTe |
| RTX PRO 6000 | noncausal_sliding | 1 | 9 | 2047 | 16/4 | 0.869 | 1.147 | 12.4 | 14.4 | CuTe |
| RTX PRO 6000 | noncausal_sliding | 1 | 9 | 2048 | 16/4 | 0.918 | 1.161 | 12.4 | 16.4 | CuTe |
| RTX PRO 6000 | noncausal_sliding | 1 | 9 | 2049 | 16/4 | 0.936 | 1.153 | 12.4 | 16.4 | CuTe |
| RTX PRO 6000 | noncausal_sliding | 3 | 16 | 128 | 32/8 | 0.557 | 1.198 | 10.3 | 22.2 | CuTe |
| RTX PRO 6000 | noncausal_sliding | 4 | 16 | 128 | 32/8 | 0.547 | 1.214 | 10.3 | 22.6 | CuTe |
| GB10 | causal_sliding | 1 | 9 | 4095 | 32/8 | 0.993 | 1.086 | 24.4 | 26.7 | CuTe |
| GB10 | causal_sliding | 1 | 9 | 4096 | 16/4 | 1.067 | 1.187 | 17.5 | 19.5 | CuTe |
| GB10 | causal_sliding | 1 | 9 | 4097 | 16/4 | 1.121 | 1.146 | 18.4 | 18.8 | CuTe |
| GB10 | causal_sliding | 1 | 9 | 4097 | 32/8 | 1.087 | 1.074 | 24.7 | 26.7 | FlashInfer |
| GB10 | causal_sliding | 1 | 12 | 4096 | 16/4 | 1.121 | 1.203 | 18.5 | 20.5 | CuTe |
| GB10 | causal_sliding | 1 | 16 | 4097 | 32/8 | 1.131 | 1.077 | 28.7 | 28.8 | FlashInfer |
| GB10 | causal_sliding | 3 | 4 | 4096 | 16/4 | 1.005 | 1.021 | 69.2 | 79.4 | CuTe |
| GB10 | causal_sliding | 4 | 4 | 4095 | 16/4 | 1.018 | 1.041 | 136.0 | 141.5 | CuTe |
| GB10 | causal_sliding | 4 | 4 | 4095 | 32/8 | 1.015 | 1.031 | 304.4 | 310.3 | CuTe |
| GB10 | causal_sliding | 4 | 4 | 4096 | 16/4 | 1.011 | 1.039 | 139.2 | 141.6 | CuTe |
| GB10 | causal_sliding | 4 | 4 | 4097 | 16/4 | 1.019 | 1.050 | 143.6 | 154.5 | CuTe |
| GB10 | causal_sliding | 4 | 9 | 4096 | 32/8 | 1.009 | 1.029 | 320.2 | 320.1 | CuTe |
| GB10 | full | 1 | 9 | 4097 | 32/8 | 1.083 | 1.152 | 26.7 | 26.1 | CuTe |
| GB10 | full | 1 | 12 | 4095 | 32/8 | 0.990 | 1.082 | 23.7 | 26.7 | CuTe |
| GB10 | full | 1 | 12 | 4096 | 16/4 | 1.093 | 1.189 | 18.0 | 19.5 | CuTe |
| GB10 | full | 1 | 12 | 4097 | 32/8 | 0.992 | 1.081 | 23.8 | 26.7 | CuTe |
| GB10 | full | 1 | 16 | 4095 | 32/8 | 1.076 | 1.099 | 27.2 | 28.8 | CuTe |
| GB10 | full | 1 | 16 | 4097 | 32/8 | 1.033 | 0.938 | 25.6 | 28.8 | FlashInfer |
| GB10 | full | 2 | 4 | 4096 | 16/4 | 0.933 | 1.117 | 21.3 | 24.6 | CuTe |
| GB10 | full | 2 | 5 | 4096 | 32/8 | 1.012 | 1.031 | 143.2 | 152.2 | CuTe |
| GB10 | full | 2 | 6 | 4097 | 32/8 | 1.032 | 1.027 | 141.1 | 147.3 | FlashInfer |
| GB10 | full | 2 | 8 | 4096 | 32/8 | 0.985 | 1.037 | 145.2 | 155.2 | CuTe |
| GB10 | full | 2 | 9 | 128 | 16/4 | 0.565 | 1.124 | 9.3 | 18.5 | CuTe |
| GB10 | full | 2 | 9 | 4096 | 32/8 | 0.999 | 1.029 | 154.1 | 156.6 | CuTe |
| GB10 | full | 2 | 16 | 128 | 16/4 | 0.557 | 1.111 | 10.3 | 20.5 | CuTe |
| GB10 | full | 3 | 4 | 4096 | 16/4 | 1.125 | 0.957 | 83.7 | 69.7 | FlashInfer |
| GB10 | full | 3 | 5 | 4096 | 16/4 | 1.052 | 1.123 | 72.7 | 78.7 | CuTe |
| GB10 | full | 3 | 9 | 4096 | 16/4 | 1.056 | 1.100 | 89.7 | 80.7 | CuTe |
| GB10 | full | 4 | 4 | 4097 | 16/4 | 1.016 | 1.046 | 143.7 | 154.4 | CuTe |
| GB10 | full | 4 | 9 | 4095 | 16/4 | 1.083 | 1.022 | 152.7 | 154.6 | FlashInfer |
| GB10 | noncausal_sliding | 1 | 9 | 2048 | 16/4 | 1.158 | 1.249 | 14.4 | 16.4 | CuTe |
| GB10 | noncausal_sliding | 1 | 16 | 128 | 32/8 | 0.722 | 1.143 | 10.3 | 16.4 | CuTe |
| GB10 | noncausal_sliding | 3 | 4 | 2047 | 32/8 | 0.918 | 1.100 | 70.4 | 77.3 | CuTe |
| GB10 | noncausal_sliding | 3 | 5 | 2047 | 32/8 | 1.051 | 1.082 | 84.5 | 77.9 | CuTe |
| GB10 | noncausal_sliding | 3 | 9 | 2047 | 32/8 | 0.890 | 0.996 | 82.1 | 88.6 | CuTe |

Totals (faster provider by paired ratio against stock FlashInfer): {"RTX 5090": {"CuTe": 11, "FlashInfer": 0}, "RTX PRO 6000": {"CuTe": 15, "FlashInfer": 0}, "GB10": {"CuTe": 29, "FlashInfer": 6}}

## Remaining distance (not met)

- **Rows below the faster market stack (rc2 gate):** RTX 5090 0/896; RTX PRO 6000
  5/896; GB10 51/896.
  - RTX PRO 6000: all five are L = 128, Q = 16, B ≥ 2, against vLLM. vLLM's single FA2
    kernel lands on the 8.3 µs step, CuTe (attention + merge) on 10.3 µs.
  - GB10, B = 1 at L ≈ 2048/4096 (mostly 16/4 heads): 1.06–1.15× stock FlashInfer
    (FlashInfer about 16.4 µs, CuTe about 18.4 µs). This is persistent across rc1, rc2 and the
    reruns; no split, tile or key-group setting closes it.
  - GB10, B = 2–4 at L ≈ 4096: 1.02–1.29×. These rows are DRAM-bound (B = 4 32/8:
    about 300 µs on both stacks) or sit at the L2 capacity (B = 3 16/4), and their reference
    times are bimodal across processes, so many flip between runs (27 of the 35 rc1
    GB10 failures passed on rerun).
  - GB10, L = 128 Q = 16 B = 1 against vLLM: one 1 µs step (9.2 vs 10.3 µs).
- **SOL ≥ 90 % at B = 1–4 is not reached:** L2-aware utilization 9–62 % (RTX 5090),
  9–68 % (RTX PRO 6000), 8–68 % (GB10). Short contexts are bounded by graph-node and
  replay-step floors that no single op can reach 90 % of. At L ≈ 4096 the kernel runs at
  about a third of the BF16 MMA rate (softmax does not overlap the MMAs).
- Next levers: overlap softmax with MMA (two warpgroups or deeper key groups), more
  bytes in flight per CTA on GB10, and a single-kernel path (S = 1 without merge) for short
  unions that does not lose at B = 1.

These historical counts include the retired K0 CuTe port. DF2 now retains
only DRAFT attention coverage; see `docs/dflash_draft_attention.md` for
the independent test, capture and access-check entry points. R5 owns
round bookkeeping. The historical GPU counts do not qualify this rewrite.
