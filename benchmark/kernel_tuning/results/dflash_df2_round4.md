<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DF2: round 4 receipts (0.1.11rc8 / 0.1.11rc9)

This round reduces shared index storage, moves fused CSR publication after loads,
and adapts split walks, merge reads and row tiles to the key union. The market bar
is met on RTX 5090 (896/896) and RTX PRO 6000 (896/896). GB10 reaches 878/896:
12 of the 18 losses are classified as process noise, 5 are real and 1 is
unclassified. SOL >= 90% is **NOT met**. The R4 per-layer floor is met at
ctx >= 300 on RTX 5090 and RTX PRO 6000 after the two noisy rows pass a separate
rerun; it is not met at ctx=100. Earlier receipts: [DF2 market gate](dflash_df2_market_gate.md).

## Candidates

| wheel version | wheel sha256 | source | market gate |
| --- | --- | --- | --- |
| 0.1.11rc8 | `cc79105588387fc4e4b2f89d28ff2736b5fe08dfc1bfdf95cd2ab8419ca03668` | `9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2` | RTX 5090 and RTX PRO 6000 |
| 0.1.11rc9 (final candidate) | `195cbafaab8a3b820f6109af36cfed7a53289fb6189e0f26a2aa79806f38fdfb` | `9146556427bd14c5296550566c5d0a3aeae422dc` | GB10 |

rc9 differs from rc8 only in GB10's row-tile rule in
`python/ctikernels/attention/_draft/cute/__init__.py` `configuration()`.
Every sm_120 code path and configuration is identical. The rc8 sm_120 gate
therefore applies to rc9.

Environment: torch 2.13.0+cu130, CUDA 13.0, nvidia-cutlass-dsl 4.8.0, BF16.
Comparators: stock FlashInfer 0.6.11.post3 (informational); SGLang 72730d6 dispatch
(FlashInfer 0.7.1 fa2); vLLM 0.31.0 dispatch (`vllm_flash_attn` FA2 `varlen_fwd`,
fork 9cd61de, P=16). GB10 timing processes are pinned to Cortex-X925 cores.

## Changes this round (kernel/provider)

| change | measured effect |
| --- | --- |
| Shared index buffer sized to the real per-CTA chunk bound instead of the chunk cap (16 KiB on GB10). | GB10 B=1 L~4096 per-kernel time dropped. Full Q12 16/4 fell from 18.6 to 16.5 us, tying FlashInfer. |
| Fused CSR publish moved after Q/index loads are issued. | Short sm_120 rows moved from the 10.3 to the 8.3 us replay step. |
| A split walks its chunk in index pieces. Capacity no longer forces splits beyond the widest window union (4128 keys). The piece loop is compiled only where needed. | Removes capacity-driven extra splits. An always-compiled loop cost one replay step, up to 1.14x. |
| Above 4128 keys, merge reads only filled splits and uses 32-row tiles. | Engine-like C=32768, B=4 Q16 ctx100 per-layer latency fell from 10.6 to 7.1 us on RTX 5090. |
| 32-row tiles at sm_120 B=1 and for unions <=512 keys. GB10 uses them at B<=2 and unions <=512 or >4128. Legacy 48/64-row tiles remain elsewhere. | Avoids the measured regressions: 32-row tiles cost up to 1.2x at B>=2 over 513–4128 keys and up to 1.5x on GB10 B=1 L~4096 in hot-L2 replays. |
| Unions <=512 keys size splits for one CTA per SM on sm_120. | RTX PRO 6000 B=4 Q16 L=128 rows fell from 10.3 to 8.3 us. |

Kernels per complete op: **2** (attention + merge), **1** when there is one split.

## Correctness and memory

These are GPU checks from the installed wheels, for **both rc8 and rc9**.

| device | GPU tests per wheel | compute-sanitizer memcheck per wheel |
| --- | --- | --- |
| RTX 5090 | 889 passed | 141/141 cases, `ERROR SUMMARY: 0 errors` |
| RTX PRO 6000 | 889 passed | 141/141 cases, `ERROR SUMMARY: 0 errors` |
| GB10 | 889 passed | 141/141 cases, `ERROR SUMMARY: 0 errors` |

The 12 new tests cover multi-piece walks at C=8192 with 64/128-key index pieces,
inactive slots and graph replays across piece boundaries. The memcheck matrix
adds three C=8192 constructions covering occupancy splits, the count-aware merge
and the piece loop. CPU suite: 1968 passed. Pre-commit: clean.

One RTX 5090 gate shard attempt on rc5 (superseded) lost its GPU mid-run:
`CUDA initialization: unknown error`. Every later row in that process errored.
The automatic retry on the same node passed all rows. This was environmental,
not a kernel fault.

## Market gate

The bar is the faster of SGLang and vLLM actual dispatch. Stock FlashInfer is
informational. The following rc8 RTX 5090 / RTX PRO 6000 sections and rc9 GB10
section are reproduced verbatim, with their shared header and protocol once.

# DRAFT market gate receipts

Gate: last eligible record per key/comparator, in lexical file and line order; `.log.try` follows `.log`. Rule: candidate correctness and speed must PASS against BOTH SGLang and vLLM actual dispatch; missing framework records fail. Correctness counts candidate PASS in both framework records. Stock FlashInfer persistent/fixed-plan results are informational only and never affect market pass/fail.

Candidate latency uses FlashInfer's candidate measurement, else SGLang, else vLLM. Faster framework is the minimum measured SGLang/vLLM reference latency. SOL uses the same candidate record and kernel count. N/A denotes missing measurements; ! marks failure or SOL above 100%.

Comparator cells are median paired ratio (ε); ! means status != PASS. Reruns never change the gate. Override device comes from JSON when available, else filename markers 5090/pro/gb. Only AB provider cute/auto with empty env qualifies.

Ceiling definition: Informational useful-work empirical reference from benchmark/sol/ceilings/{rtx5090,rtx-pro-6000,gb10}.json; serial graph kernels, no PDL: sum(max(graph floor, DRAM time, BF16 MMA time)). One attention node carries BF16 Q reads, union-visible K/V reads per KV head, one int32 page index per union key and BF16 output writes; FLOPs = 4*B*Hq*128*sum(query visible-key counts). Remaining launches are floor-only nodes. The benchmark replays identical buffers, so min(1, L2 size/working set) of every attention byte is served at the measured L2 read rate (warm_l2 when it all fits, else partial_l2), the rest at the sustained DRAM read rate as an aggregate traffic proxy; cold_t_ref_us keeps the all-DRAM reference. Excludes padding, scalar work, scratch traffic and eager host overhead. Utilization is t_ref/measured latency, with ratios above one flagged.

This receipt uses the L2-aware **t_ref_us / candidate_median_us** (warm_l2 or partial_l2).

## rtx5090

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl; vllm_so_sha256=db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-15 / 0-31

| Q | mask | rows | correctness | PASS vs SGLang | PASS vs vLLM | PASS market bar (both) | FlashInfer informational PASS | FlashInfer informational ratio range | candidate/faster-framework ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4365–0.9362 | 0.5028–0.9401 | 12.5%–62.1% | {2} |
| 4 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4322–0.9978 | 0.5025–0.9891 | 12.0%–62.1% | {2} |
| 4 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4669–0.8023 | 0.5031–0.8024 | 12.5%–45.9% | {2} |
| 5 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.3087–0.8408 | 0.3804–0.8439 | 12.5%–62.2% | {2} |
| 5 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3027–0.8238 | 0.3195–0.8243 | 12.5%–62.2% | {2} |
| 5 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3352–0.7497 | 0.4025–0.7496 | 12.5%–46.0% | {2} |
| 6 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 39 | 0.3024–0.7615 | 0.3495–0.7618 | 12.5%–62.3% | {2} |
| 6 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3024–0.7503 | 0.3043–0.7454 | 12.5%–62.3% | {2} |
| 6 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3030–0.7514 | 0.3595–0.7511 | 12.5%–46.1% | {2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 35 | 0.3166–0.7373 | 0.4007–0.8554 | 12.0%–62.3% | {2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3027–0.7276 | 0.3514–0.7167 | 12.5%–62.4% | {2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3038–0.7156 | 0.3359–0.7898 | 11.0%–46.3% | {2} |
| 9 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 35 | 0.3025–0.9995 | 0.4013–0.9993 | 9.8%–44.7% | {2} |
| 9 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3021–0.9995 | 0.4013–0.9992 | 9.8%–47.6% | {2} |
| 9 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3027–0.7164 | 0.4023–0.7984 | 9.8%–36.1% | {2} |
| 12 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.3028–0.9985 | 0.4012–0.9994 | 10.0%–45.8% | {2} |
| 12 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3023–0.9980 | 0.4014–0.9977 | 10.0%–49.1% | {2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3030–0.7160 | 0.4014–0.7986 | 10.0%–36.7% | {2} |
| 16 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.3024–0.8764 | 0.4015–0.8766 | 9.5%–53.1% | {2} |
| 16 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3027–0.8766 | 0.4016–0.8770 | 10.5%–55.8% | {2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3150–0.7154 | 0.4016–0.8003 | 9.5%–39.6% | {2} |
| **Total** | all | 896 | 896 | 896 | 896 | 896 | 885 | 0.3021–0.9995 | 0.3043–0.9994 | 9.5%–62.4% | {2} |

Totals: rows=896, correct=896, pass_market=896, fail_market=0, flashinfer_informational_fail=11.

### Rows below the market bar

| mask | B | Q | L | Hq/Hkv | SGLang ratio (ε) | vLLM ratio (ε) | candidate µs | faster framework µs | FlashInfer informational ratio (ε) | rerun market bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

None.


### Rerun receipts (failed gate keys only)

| mask | B | Q | L | Hq/Hkv | rerun market bar (ratios and ε) |
| --- | --- | --- | --- | --- | --- |

## rtx-pro-6000

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_x86_64.whl; vllm_so_sha256=db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9; ctikernels_source_sha=9ef48ff2ca0753c9e00be42e792f1bf44b4f83c2; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=0-29 / 0-59 / 0-7

| Q | mask | rows | correctness | PASS vs SGLang | PASS vs vLLM | PASS market bar (both) | FlashInfer informational PASS | FlashInfer informational ratio range | candidate/faster-framework ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.6677–0.9963 | 0.6675–0.9547 | 11.7%–67.5% | {2} |
| 4 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.6678–0.9972 | 0.6674–0.9977 | 11.7%–67.5% | {2} |
| 4 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.6680–0.9962 | 0.6677–0.9985 | 11.7%–52.3% | {2} |
| 5 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5017–0.9839 | 0.5024–0.9903 | 11.7%–67.6% | {2} |
| 5 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.5015–0.9410 | 0.5033–0.9883 | 11.7%–67.6% | {2} |
| 5 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5018–0.8570 | 0.5016–0.9980 | 11.7%–52.5% | {2} |
| 6 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.5013–0.8757 | 0.5016–0.8757 | 11.7%–67.7% | {2} |
| 6 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4933–0.8756 | 0.5019–0.9969 | 11.7%–67.7% | {2} |
| 6 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.5013–0.8345 | 0.5013–0.9968 | 11.7%–46.6% | {2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4473–0.8886 | 0.4855–0.8889 | 11.7%–67.2% | {2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4471–0.8749 | 0.4637–0.9270 | 11.7%–67.8% | {2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4758–0.8561 | 0.4975–0.9984 | 11.7%–45.6% | {2} |
| 9 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4473–0.9999 | 0.5027–1.0002 | 11.7%–51.0% | {2} |
| 9 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4470–0.9998 | 0.5026–1.0004 | 11.7%–55.3% | {2} |
| 9 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4500–0.8329 | 0.5164–0.9988 | 11.4%–40.1% | {2} |
| 12 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4473–0.9945 | 0.4678–0.9996 | 11.7%–51.2% | {2} |
| 12 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4470–0.9975 | 0.4943–0.9998 | 11.7%–51.2% | {2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4471–0.7999 | 0.5031–0.9994 | 11.2%–40.3% | {2} |
| 16 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4466–0.9272 | 0.5570–0.9551 | 9.4%–47.2% | {2} |
| 16 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4381–0.9374 | 0.5579–0.9994 | 9.7%–48.1% | {2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4473–0.8293 | 0.5809–1.0005 | 9.4%–36.1% | {2} |
| **Total** | all | 896 | 896 | 896 | 896 | 896 | 896 | 0.4381–0.9999 | 0.4637–1.0005 | 9.4%–67.8% | {2} |

Totals: rows=896, correct=896, pass_market=896, fail_market=0, flashinfer_informational_fail=0.

### Rows below the market bar

| mask | B | Q | L | Hq/Hkv | SGLang ratio (ε) | vLLM ratio (ε) | candidate µs | faster framework µs | FlashInfer informational ratio (ε) | rerun market bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

None.


### Rerun receipts (failed gate keys only)

| mask | B | Q | L | Hq/Hkv | rerun market bar (ratios and ε) |
| --- | --- | --- | --- | --- | --- |

## gb10

- flashinfer: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.6.11.post3; ctikernels_source_sha=9146556427bd14c5296550566c5d0a3aeae422dc; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19
- sglang: torch=2.13.0+cu130; cuda=13.0; flashinfer=0.7.1; ctikernels_source_sha=9146556427bd14c5296550566c5d0a3aeae422dc; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19
- vllm: torch=2.13.0+cu130; cuda=13.0; vllm_version=0.31.0; vllm_wheel=vllm-0.31.0-cp38-abi3-manylinux_2_28_aarch64.whl; vllm_so_sha256=be52450ee26044a3903082a726e05d2f0f0acbdcce994e94e12c6f54274a2246; ctikernels_source_sha=9146556427bd14c5296550566c5d0a3aeae422dc; vllm_revision=db9527a46873454610df6dbedf79a36d6bf1a7f6; vllm_flash_attn_revision=9cd61de38763d712bb6ce56e2a02cc2bf718c89f; sglang_revision=72730d6360a5894b167500d5d1ad82c34c6bb116; affinity_cores=5-9,15-17,19 / 5-9,15-19

| Q | mask | rows | correctness | PASS vs SGLang | PASS vs vLLM | PASS market bar (both) | FlashInfer informational PASS | FlashInfer informational ratio range | candidate/faster-framework ratio range | SOL util range (L2-aware) | kernels/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | causal_sliding | 40 | 40 | 37 | 40 | 37 | 38 | 0.5736–1.0750 | 0.5735–1.2474 | 10.0%–67.5% | {1, 2} |
| 4 | full | 40 | 40 | 37 | 40 | 37 | 36 | 0.6020–1.1169 | 0.5876–1.1669 | 10.2%–67.6% | {1, 2} |
| 4 | noncausal_sliding | 48 | 48 | 46 | 48 | 46 | 48 | 0.5913–1.0221 | 0.6121–1.1661 | 10.0%–67.8% | {1, 2} |
| 5 | causal_sliding | 40 | 40 | 39 | 40 | 39 | 39 | 0.4456–1.1004 | 0.4458–1.0970 | 10.0%–67.6% | {1, 2} |
| 5 | full | 40 | 40 | 39 | 40 | 39 | 38 | 0.4457–1.1428 | 0.4456–1.1772 | 10.4%–67.7% | {1, 2} |
| 5 | noncausal_sliding | 48 | 48 | 47 | 48 | 47 | 48 | 0.4462–1.0448 | 0.4462–1.1131 | 10.0%–67.9% | {1, 2} |
| 6 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4452–1.0298 | 0.4456–1.1165 | 10.1%–67.6% | {1, 2} |
| 6 | full | 40 | 40 | 38 | 40 | 38 | 40 | 0.4451–1.0297 | 0.4451–1.1059 | 10.3%–67.7% | {1, 2} |
| 6 | noncausal_sliding | 48 | 48 | 47 | 48 | 47 | 48 | 0.4458–1.0342 | 0.4458–1.0636 | 10.1%–68.1% | {1, 2} |
| 8 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4015–1.0232 | 0.4012–1.1331 | 10.0%–67.2% | {1, 2} |
| 8 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.4011–1.0950 | 0.4009–1.1595 | 10.0%–67.6% | {1, 2} |
| 8 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4012–1.0264 | 0.4010–1.0668 | 10.0%–68.4% | {1, 2} |
| 9 | causal_sliding | 40 | 40 | 38 | 40 | 38 | 39 | 0.4457–1.2220 | 0.4818–1.2158 | 10.0%–55.5% | {1, 2} |
| 9 | full | 40 | 40 | 39 | 40 | 39 | 38 | 0.4457–1.2115 | 0.4872–1.3237 | 10.0%–56.7% | {1, 2} |
| 9 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4569–1.1077 | 0.5011–1.1496 | 10.0%–47.6% | {1, 2} |
| 12 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.4387–1.0809 | 0.4553–1.2102 | 10.0%–56.5% | {1, 2} |
| 12 | full | 40 | 40 | 39 | 40 | 39 | 40 | 0.4323–1.1872 | 0.4568–1.2426 | 10.0%–55.9% | {1, 2} |
| 12 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.4455–1.0452 | 0.4726–1.2088 | 10.0%–51.8% | {1, 2} |
| 16 | causal_sliding | 40 | 40 | 40 | 40 | 40 | 40 | 0.3860–1.1327 | 0.4560–1.1977 | 10.0%–55.9% | {1, 2} |
| 16 | full | 40 | 40 | 40 | 40 | 40 | 40 | 0.3856–1.0415 | 0.4621–1.1796 | 10.0%–56.3% | {1, 2} |
| 16 | noncausal_sliding | 48 | 48 | 48 | 48 | 48 | 48 | 0.3864–1.0003 | 0.4608–1.0292 | 10.0%–48.3% | {1, 2} |
| **Total** | all | 896 | 896 | 878 | 896 | 878 | 884 | 0.3856–1.2220 | 0.4009–1.3237 | 10.0%–68.4% | {1, 2} |

Totals: rows=896, correct=896, pass_market=878, fail_market=18, flashinfer_informational_fail=12.

### Rows below the market bar

| mask | B | Q | L | Hq/Hkv | SGLang ratio (ε) | vLLM ratio (ε) | candidate µs | faster framework µs | FlashInfer informational ratio (ε) | rerun market bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| causal_sliding | 1 | 9 | 4095 | 32/8 | 1.5283 (0.1596) ! | 0.3474 (0.0777) | 30.1536 | 24.8014 | 1.2220 (0.1461) ! | — |
| causal_sliding | 2 | 4 | 4095 | 32/8 | 1.0293 (0.0241) ! | 0.6992 (0.0483) | 157.8394 | 143.5039 | 1.0311 (0.0417) | — |
| causal_sliding | 3 | 5 | 4097 | 16/4 | 1.0944 (0.0684) ! | 0.6948 (0.0800) | 70.5546 | 79.4456 | 0.9811 (0.0723) | — |
| causal_sliding | 3 | 9 | 4096 | 16/4 | 1.1746 (0.1067) ! | 0.7364 (0.0766) | 80.4822 | 73.8274 | 1.0765 (0.1027) | — |
| causal_sliding | 4 | 4 | 4095 | 16/4 | 1.0415 (0.0387) ! | 0.7105 (0.0474) | 139.7947 | 151.9158 | 0.9788 (0.0349) | — |
| causal_sliding | 4 | 4 | 4097 | 16/4 | 1.0387 (0.0358) ! | 0.7171 (0.0442) | 133.3618 | 132.8779 | 0.9619 (0.0405) | — |
| full | 1 | 9 | 4097 | 32/8 | 1.5838 (0.1646) ! | 0.2966 (0.0355) | 29.9170 | 23.6715 | 1.2115 (0.1874) ! | — |
| full | 1 | 12 | 4095 | 32/8 | 1.4975 (0.1701) ! | 0.3096 (0.0337) | 31.0100 | 24.9558 | 1.1872 (0.1936) | — |
| full | 2 | 5 | 4097 | 32/8 | 1.0335 (0.0181) ! | 0.6849 (0.0447) | 153.2718 | 139.2745 | 1.0048 (0.0252) | — |
| full | 2 | 6 | 4095 | 32/8 | 1.0392 (0.0252) ! | 0.6699 (0.0428) | 153.6011 | 138.8921 | 1.0228 (0.0260) | — |
| full | 2 | 6 | 4096 | 32/8 | 1.0791 (0.0387) ! | 0.6937 (0.0622) | 139.9461 | 131.7329 | 0.9878 (0.0295) | — |
| full | 3 | 4 | 4095 | 16/4 | 1.1077 (0.0485) ! | 0.7185 (0.0843) | 67.0204 | 73.1320 | 1.1169 (0.0756) ! | — |
| full | 3 | 4 | 4096 | 16/4 | 1.0764 (0.0630) ! | 0.7686 (0.0862) | 70.6768 | 67.4151 | 1.1064 (0.0728) ! | — |
| full | 3 | 4 | 4097 | 16/4 | 1.0496 (0.0480) ! | 0.7284 (0.0883) | 79.8268 | 83.3312 | 1.0735 (0.0749) | — |
| noncausal_sliding | 3 | 4 | 2047 | 32/8 | 1.0651 (0.0429) ! | 0.6454 (0.0880) | 67.5678 | 96.2481 | 1.0079 (0.0671) | — |
| noncausal_sliding | 3 | 4 | 4096 | 32/8 | 1.0823 (0.0506) ! | 0.6899 (0.0917) | 72.7488 | 69.8652 | 0.9827 (0.0791) | — |
| noncausal_sliding | 3 | 5 | 4096 | 32/8 | 1.1174 (0.0581) ! | 0.7222 (0.0925) | 73.5464 | 70.3971 | 0.9724 (0.0708) | — |
| noncausal_sliding | 3 | 6 | 4096 | 32/8 | 1.1065 (0.0596) ! | 0.7662 (0.0862) | 75.3438 | 70.8355 | 0.9719 (0.0638) | — |

### Rerun receipts (failed gate keys only)

| mask | B | Q | L | Hq/Hkv | rerun market bar (ratios and ε) |
| --- | --- | --- | --- | --- | --- |


## GB10 classification

Each failing row was rerun in 3 independent processes on one dedicated GB10.
Timing processes were pinned to Cortex-X925 cores. Each process used 40
interleaved pairs, with epsilon from 20 self-pairs. A row is **real** when it
fails against either framework in at least 2 of 3 processes; otherwise it is
**noise**. The table covers the rc9 gate's 19 failing rows seen before completion.
Row keys are mask:B:Q:L:Hq/Hkv. Cells are paired ratio (epsilon); ! marks failure.

| row | process 1 SGLang / vLLM | process 2 | process 3 | failing processes | class |
| --- | --- | --- | --- | --- | --- |
| causal_sliding:1:9:4095:32/8 | 1.005 (0.095) / 0.312 (0.028) | 1.084 (0.107) / 0.242 (0.030) | 0.996 (0.089) / 0.242 (0.031) | 0/3 | noise |
| causal_sliding:2:4:4095:32/8 | 1.063 (0.051)! / 0.605 (0.047) | 0.983 (0.036) / 0.680 (0.053) | 1.022 (0.038) / 0.624 (0.049) | 1/3 | noise |
| causal_sliding:3:5:4097:16/4 | 1.194 (0.074)! / 0.681 (0.098) | 0.928 (0.062) / 0.654 (0.084) | 1.060 (0.072) / 0.618 (0.089) | 1/3 | noise |
| causal_sliding:4:4:4095:16/4 | 1.086 (0.043)! / 0.617 (0.049) | 0.992 (0.041) / 0.704 (0.045) | 1.041 (0.038)! / 0.642 (0.049) | 2/3 | real |
| causal_sliding:4:4:4097:16/4 | 1.067 (0.037)! / 0.615 (0.049) | 0.985 (0.016) / 0.707 (0.047) | 1.036 (0.039) / 0.644 (0.048) | 1/3 | noise |
| full:1:12:4095:32/8 | 1.000 (0.167) / 0.335 (0.029) | 1.081 (0.118) / 0.252 (0.030) | 0.986 (0.134) / 0.252 (0.030) | 0/3 | noise |
| full:1:9:4097:32/8 | 0.988 (0.091) / 0.323 (0.029) | 1.093 (0.026)! / 0.250 (0.039) | 0.980 (0.031) / 0.251 (0.042) | 1/3 | noise |
| full:2:5:4097:32/8 | 1.050 (0.031)! / 0.620 (0.050) | 0.952 (0.033) / 0.704 (0.049) | 1.018 (0.040) / 0.654 (0.050) | 1/3 | noise |
| full:2:6:4095:32/8 | 1.047 (0.028)! / 0.596 (0.082) | 0.950 (0.051) / 0.701 (0.046) | 1.015 (0.041) / 0.654 (0.049) | 1/3 | noise |
| full:2:6:4096:32/8 | 1.047 (0.043)! / 0.595 (0.048) | 0.949 (0.010) / 0.697 (0.053) | 1.013 (0.020) / 0.653 (0.062) | 1/3 | noise |
| full:3:4:4095:16/4 | 1.196 (0.053)! / 0.704 (0.073) | 0.937 (0.077) / 0.664 (0.088) | 1.092 (0.058)! / 0.614 (0.087) | 2/3 | real |
| full:3:4:4096:16/4 | 1.197 (0.073)! / 0.686 (0.104) | 0.934 (0.061) / 0.673 (0.090) | 1.086 (0.041)! / 0.639 (0.092) | 2/3 | real |
| full:3:4:4097:16/4 | 1.186 (0.075)! / 0.697 (0.085) | 0.942 (0.048) / 0.664 (0.094) | 1.096 (0.059)! / 0.645 (0.149) | 2/3 | real |
| full:4:12:4097:16/4 | 0.976 (0.036) / 0.692 (0.048) | 1.006 (0.039) / 0.755 (0.052) | 0.975 (0.039) / 0.715 (0.051) | 0/3 | noise |
| full:4:4:4095:16/4 | 1.084 (0.052)! / 0.638 (0.052) | 0.991 (0.059) / 0.725 (0.049) | 1.042 (0.048) / 0.661 (0.052) | 1/3 | noise |
| noncausal_sliding:3:4:2047:32/8 | 1.175 (0.084)! / 0.621 (0.082) | 1.056 (0.064) / 0.664 (0.096) | 1.083 (0.080)! / 0.627 (0.097) | 2/3 | real |
| noncausal_sliding:3:4:4096:32/8 | 1.042 (0.086) / 0.455 (0.076) | 1.006 (0.060) / 0.662 (0.083) | 0.878 (0.056) / 0.632 (0.086) | 0/3 | noise |
| noncausal_sliding:3:5:4096:32/8 | 1.007 (0.035) / 0.447 (0.086) | 1.004 (0.080) / 0.647 (0.090) | 0.893 (0.077) / 0.623 (0.088) | 0/3 | noise |
| noncausal_sliding:3:6:4096:32/8 | 0.897 (0.050) / 0.459 (0.085) | 1.006 (0.075) / 0.658 (0.090) | 0.904 (0.071) / 0.609 (0.090) | 0/3 | noise |

The final gate has 18 losses. Their mapping is:

| classification | count | final loss keys (mask:B:Q:L:Hq/Hkv) |
| --- | --- | --- |
| noise | 12 | causal_sliding:1:9:4095:32/8; causal_sliding:2:4:4095:32/8; causal_sliding:3:5:4097:16/4; causal_sliding:4:4:4097:16/4; full:1:9:4097:32/8; full:1:12:4095:32/8; full:2:5:4097:32/8; full:2:6:4095:32/8; full:2:6:4096:32/8; noncausal_sliding:3:4:4096:32/8; noncausal_sliding:3:5:4096:32/8; noncausal_sliding:3:6:4096:32/8 |
| real | 5 | causal_sliding:4:4:4095:16/4; full:3:4:4095:16/4; full:3:4:4096:16/4; full:3:4:4097:16/4; noncausal_sliding:3:4:2047:32/8 |
| unclassified | 1 | causal_sliding:3:9:4096:16/4 (first seen in the rerun shard) |

The two classified noise rows full:4:12:4097:16/4 and full:4:4:4095:16/4 are
absent from the final loss set. Classification does not change the gate counts.
All five real losses have Q=4 at B>=3. Their K/V working sets are ~25–34 MB,
around GB10's 24 MiB L2 capacity. Per-row split/tile sweeps used splits 3/6/8,
32-key tiles, 96 target CTAs and 2 repetitions. No setting won in every process.
The candidate and SGLang both swing ~56–82 us between processes.

The earlier rc6 classification of rc4's 18 GB10 failures found **17 noise,
1 real**. The real row was causal_sliding B3 Q4 L4096 16/4 (2/3 failing
processes). All other rows failed in at most one process.

## R4 per-layer

Reproduce with `python -m benchmark.kernel_tuning.dflash_draft_layers`.
The engine methodology rotates n distinct layer K/V buffers over at least 2x L2.
One CUDA graph contains n calls. CUDA events measure each replay, and per-call
latency is elapsed / n. Report the median of 25 replays. kvlen = ctx + Q.
n is at least the model layer count (36 target, 5 draft); metadata is prepared
once, and `cute_fused` folds prepare into the first layer call only.

Shapes are q3_target (32/8 heads, causal full, benchmark-only mask) and
draft_noncausal (32/8 heads, full non-causal), both with head dimension 128.
DraftAttention's public masks do not include causal full attention. The benchmark
uses a geometry-check wrapper that verifies mask forwarding before relaxing only
validation; the actual causal flag is preserved.
Q is {4,5,8,9,12,16} at exact capacity and {4,9,16} at C=32768.
B is 1–4; ctx is 100/300/1000/4000. The floor is
min(vLLM FA2, FlashInfer fixed-query), the latter being the engine's current path.

Supply `--vllm-fa2 PATH` to the vLLM `_vllm_fa2_C.abi3.so`; without it, that
provider is recorded as skipped and the full floor cannot be assessed.
`--shapes`, `--batches` and `--contexts` accept comma-separated lists.
`--query` selects Q; `--capacity 0` uses exact kvlen. `--combos` runs each
Q/CAP/tuning combination in a fresh subprocess. For example:

```sh
python -m benchmark.kernel_tuning.dflash_draft_layers \
  --vllm-fa2 _vllm_fa2_C.abi3.so \
  --combos '[{"Q":4},{"Q":5},{"Q":8},{"Q":9},{"Q":12},{"Q":16},{"Q":4,"CAP":32768},{"Q":9,"CAP":32768},{"Q":16,"CAP":32768}]'
```

The original R4 summary and failing rows follow verbatim. The summary retains
the two RTX 5090 losses from the first measurement; the separate reruns are
recorded below it.

| device | rows | at or below min(vLLM FA2, FlashInfer fixed-query) | ctx100 losses (exact / C=32768) | other losses |
| --- | --- | --- | --- | --- |
| RTX 5090 | 288 | 270 | 0 / 16 | 2 |
| RTX PRO 6000 | 288 | 222 | 42 / 24 | 0 |
| GB10 | 288 | 246 | 13 / 18 | 11 |

| device | shape | Q | B | ctx | capacity | CuTe us | vLLM FA2 us | FlashInfer fixed us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | draft_noncausal | 4 | 2 | 4000 | exact | 38.19 | 114.24 | 26.87 |
| RTX 5090 | draft_noncausal | 4 | 3 | 100 | C=32768 | 6.41 | 5.76 | 10.49 |
| RTX 5090 | draft_noncausal | 4 | 4 | 100 | C=32768 | 6.61 | 6.02 | 10.81 |
| RTX 5090 | draft_noncausal | 9 | 2 | 100 | C=32768 | 6.47 | 5.71 | 10.74 |
| RTX 5090 | draft_noncausal | 9 | 3 | 100 | C=32768 | 6.56 | 5.8 | 17.15 |
| RTX 5090 | draft_noncausal | 9 | 4 | 100 | C=32768 | 7.0 | 6.1 | 17.74 |
| RTX 5090 | draft_noncausal | 12 | 1 | 4000 | exact | 19.6 | 113.68 | 17.46 |
| RTX 5090 | draft_noncausal | 16 | 2 | 100 | C=32768 | 6.8 | 5.81 | 11.41 |
| RTX 5090 | draft_noncausal | 16 | 3 | 100 | C=32768 | 7.11 | 5.96 | 17.95 |
| RTX 5090 | draft_noncausal | 16 | 4 | 100 | C=32768 | 7.37 | 6.19 | 18.6 |
| RTX 5090 | q3_target | 4 | 3 | 100 | C=32768 | 6.37 | 5.83 | 10.56 |
| RTX 5090 | q3_target | 4 | 4 | 100 | C=32768 | 6.48 | 6.03 | 10.77 |
| RTX 5090 | q3_target | 9 | 2 | 100 | C=32768 | 6.49 | 5.81 | 10.84 |
| RTX 5090 | q3_target | 9 | 3 | 100 | C=32768 | 6.66 | 5.88 | 17.16 |
| RTX 5090 | q3_target | 9 | 4 | 100 | C=32768 | 6.86 | 6.16 | 17.88 |
| RTX 5090 | q3_target | 16 | 2 | 100 | C=32768 | 6.71 | 5.89 | 11.45 |
| RTX 5090 | q3_target | 16 | 3 | 100 | C=32768 | 6.87 | 6.02 | 18.08 |
| RTX 5090 | q3_target | 16 | 4 | 100 | C=32768 | 7.19 | 6.3 | 18.68 |
| RTX PRO 6000 | draft_noncausal | 4 | 1 | 100 | C=32768 | 6.04 | 5.32 | 8.35 |
| RTX PRO 6000 | draft_noncausal | 4 | 1 | 100 | exact | 5.56 | 5.21 | 8.09 |
| RTX PRO 6000 | draft_noncausal | 4 | 2 | 100 | C=32768 | 5.86 | 5.43 | 9.54 |
| RTX PRO 6000 | draft_noncausal | 4 | 2 | 100 | exact | 5.61 | 5.3 | 9.3 |
| RTX PRO 6000 | draft_noncausal | 4 | 3 | 100 | C=32768 | 7.01 | 5.52 | 9.7 |
| RTX PRO 6000 | draft_noncausal | 4 | 3 | 100 | exact | 5.59 | 5.38 | 9.42 |
| RTX PRO 6000 | draft_noncausal | 4 | 4 | 100 | C=32768 | 7.28 | 5.7 | 9.94 |
| RTX PRO 6000 | draft_noncausal | 4 | 4 | 100 | exact | 5.63 | 5.5 | 9.63 |
| RTX PRO 6000 | draft_noncausal | 5 | 1 | 100 | exact | 5.4 | 5.22 | 9.17 |
| RTX PRO 6000 | draft_noncausal | 5 | 2 | 100 | exact | 5.47 | 5.31 | 9.37 |
| RTX PRO 6000 | draft_noncausal | 5 | 3 | 100 | exact | 5.41 | 5.38 | 9.5 |
| RTX PRO 6000 | draft_noncausal | 8 | 1 | 100 | exact | 5.7 | 5.27 | 9.5 |
| RTX PRO 6000 | draft_noncausal | 8 | 2 | 100 | exact | 5.77 | 5.35 | 9.66 |
| RTX PRO 6000 | draft_noncausal | 8 | 3 | 100 | exact | 5.72 | 5.43 | 13.98 |
| RTX PRO 6000 | draft_noncausal | 8 | 4 | 100 | exact | 5.82 | 5.5 | 14.33 |
| RTX PRO 6000 | draft_noncausal | 9 | 1 | 100 | C=32768 | 5.97 | 5.32 | 9.67 |
| RTX PRO 6000 | draft_noncausal | 9 | 1 | 100 | exact | 5.47 | 5.24 | 9.54 |
| RTX PRO 6000 | draft_noncausal | 9 | 2 | 100 | C=32768 | 7.51 | 5.48 | 14.2 |
| RTX PRO 6000 | draft_noncausal | 9 | 2 | 100 | exact | 5.63 | 5.41 | 13.93 |
| RTX PRO 6000 | draft_noncausal | 9 | 3 | 100 | C=32768 | 7.58 | 5.62 | 14.48 |
| RTX PRO 6000 | draft_noncausal | 9 | 3 | 100 | exact | 5.59 | 5.44 | 14.13 |
| RTX PRO 6000 | draft_noncausal | 9 | 4 | 100 | C=32768 | 7.71 | 5.71 | 14.83 |
| RTX PRO 6000 | draft_noncausal | 9 | 4 | 100 | exact | 5.75 | 5.56 | 14.52 |
| RTX PRO 6000 | draft_noncausal | 12 | 1 | 100 | exact | 5.57 | 5.33 | 9.85 |
| RTX PRO 6000 | draft_noncausal | 12 | 2 | 100 | exact | 5.75 | 5.41 | 14.27 |
| RTX PRO 6000 | draft_noncausal | 12 | 3 | 100 | exact | 5.69 | 5.5 | 14.67 |
| RTX PRO 6000 | draft_noncausal | 12 | 4 | 100 | exact | 5.9 | 5.59 | 15.11 |
| RTX PRO 6000 | draft_noncausal | 16 | 1 | 100 | C=32768 | 6.17 | 5.46 | 10.47 |
| RTX PRO 6000 | draft_noncausal | 16 | 1 | 100 | exact | 5.86 | 5.4 | 10.19 |
| RTX PRO 6000 | draft_noncausal | 16 | 2 | 100 | C=32768 | 7.88 | 5.67 | 15.02 |
| RTX PRO 6000 | draft_noncausal | 16 | 2 | 100 | exact | 6.15 | 5.51 | 14.62 |
| RTX PRO 6000 | draft_noncausal | 16 | 3 | 100 | C=32768 | 8.07 | 5.92 | 15.84 |
| RTX PRO 6000 | draft_noncausal | 16 | 3 | 100 | exact | 6.32 | 5.73 | 15.33 |
| RTX PRO 6000 | draft_noncausal | 16 | 4 | 100 | C=32768 | 8.3 | 6.07 | 16.33 |
| RTX PRO 6000 | draft_noncausal | 16 | 4 | 100 | exact | 6.82 | 5.97 | 15.86 |
| RTX PRO 6000 | q3_target | 4 | 1 | 100 | C=32768 | 5.93 | 5.5 | 8.45 |
| RTX PRO 6000 | q3_target | 4 | 1 | 100 | exact | 5.43 | 5.38 | 8.2 |
| RTX PRO 6000 | q3_target | 4 | 2 | 100 | C=32768 | 5.86 | 5.53 | 9.6 |
| RTX PRO 6000 | q3_target | 4 | 2 | 100 | exact | 5.62 | 5.4 | 9.31 |
| RTX PRO 6000 | q3_target | 4 | 3 | 100 | C=32768 | 7.18 | 5.65 | 9.76 |
| RTX PRO 6000 | q3_target | 4 | 3 | 100 | exact | 5.58 | 5.49 | 9.47 |
| RTX PRO 6000 | q3_target | 4 | 4 | 100 | C=32768 | 7.22 | 5.75 | 10.0 |
| RTX PRO 6000 | q3_target | 4 | 4 | 100 | exact | 5.65 | 5.59 | 9.7 |
| RTX PRO 6000 | q3_target | 5 | 2 | 100 | exact | 5.5 | 5.42 | 9.45 |
| RTX PRO 6000 | q3_target | 8 | 1 | 100 | exact | 5.52 | 5.49 | 9.5 |
| RTX PRO 6000 | q3_target | 8 | 2 | 100 | exact | 5.81 | 5.5 | 9.87 |
| RTX PRO 6000 | q3_target | 8 | 3 | 100 | exact | 5.72 | 5.52 | 14.12 |
| RTX PRO 6000 | q3_target | 8 | 4 | 100 | exact | 5.83 | 5.63 | 14.49 |
| RTX PRO 6000 | q3_target | 9 | 1 | 100 | C=32768 | 5.84 | 5.55 | 9.87 |
| RTX PRO 6000 | q3_target | 9 | 2 | 100 | C=32768 | 7.62 | 5.6 | 14.39 |
| RTX PRO 6000 | q3_target | 9 | 2 | 100 | exact | 5.64 | 5.58 | 14.2 |
| RTX PRO 6000 | q3_target | 9 | 3 | 100 | C=32768 | 7.6 | 5.69 | 14.66 |
| RTX PRO 6000 | q3_target | 9 | 3 | 100 | exact | 5.58 | 5.54 | 14.31 |
| RTX PRO 6000 | q3_target | 9 | 4 | 100 | C=32768 | 7.7 | 5.75 | 15.06 |
| RTX PRO 6000 | q3_target | 9 | 4 | 100 | exact | 5.74 | 5.65 | 14.71 |
| RTX PRO 6000 | q3_target | 12 | 2 | 100 | exact | 5.73 | 5.69 | 14.63 |
| RTX PRO 6000 | q3_target | 12 | 3 | 100 | exact | 5.69 | 5.58 | 14.81 |
| RTX PRO 6000 | q3_target | 12 | 4 | 100 | exact | 5.89 | 5.68 | 15.23 |
| RTX PRO 6000 | q3_target | 16 | 1 | 100 | C=32768 | 5.98 | 5.77 | 10.74 |
| RTX PRO 6000 | q3_target | 16 | 1 | 100 | exact | 5.72 | 5.65 | 10.46 |
| RTX PRO 6000 | q3_target | 16 | 2 | 100 | C=32768 | 7.73 | 5.77 | 15.21 |
| RTX PRO 6000 | q3_target | 16 | 2 | 100 | exact | 6.16 | 5.66 | 14.96 |
| RTX PRO 6000 | q3_target | 16 | 3 | 100 | C=32768 | 8.07 | 5.76 | 15.67 |
| RTX PRO 6000 | q3_target | 16 | 3 | 100 | exact | 6.29 | 5.66 | 15.33 |
| RTX PRO 6000 | q3_target | 16 | 4 | 100 | C=32768 | 8.02 | 5.84 | 16.27 |
| RTX PRO 6000 | q3_target | 16 | 4 | 100 | exact | 6.61 | 5.87 | 15.92 |
| GB10 | draft_noncausal | 4 | 1 | 100 | C=32768 | 5.98 | 5.27 | 8.04 |
| GB10 | draft_noncausal | 4 | 1 | 100 | exact | 5.58 | 5.24 | 7.85 |
| GB10 | draft_noncausal | 4 | 2 | 4000 | C=32768 | 144.15 | 195.78 | 142.48 |
| GB10 | draft_noncausal | 4 | 4 | 100 | C=32768 | 11.22 | 14.31 | 10.4 |
| GB10 | draft_noncausal | 4 | 4 | 4000 | C=32768 | 279.13 | 311.66 | 276.32 |
| GB10 | draft_noncausal | 8 | 1 | 100 | exact | 5.74 | 5.27 | 9.0 |
| GB10 | draft_noncausal | 8 | 2 | 4000 | exact | 149.57 | 192.83 | 144.79 |
| GB10 | draft_noncausal | 9 | 1 | 100 | C=32768 | 6.36 | 5.45 | 9.39 |
| GB10 | draft_noncausal | 9 | 1 | 100 | exact | 5.71 | 5.37 | 9.07 |
| GB10 | draft_noncausal | 9 | 1 | 4000 | exact | 76.51 | 94.51 | 75.63 |
| GB10 | draft_noncausal | 9 | 2 | 100 | C=32768 | 10.38 | 9.66 | 14.01 |
| GB10 | draft_noncausal | 9 | 2 | 4000 | C=32768 | 152.7 | 192.6 | 148.72 |
| GB10 | draft_noncausal | 16 | 1 | 100 | C=32768 | 6.46 | 5.84 | 10.27 |
| GB10 | draft_noncausal | 16 | 1 | 100 | exact | 6.21 | 5.52 | 9.79 |
| GB10 | draft_noncausal | 16 | 2 | 100 | C=32768 | 11.42 | 10.08 | 14.87 |
| GB10 | draft_noncausal | 16 | 3 | 100 | C=32768 | 11.89 | 11.15 | 16.11 |
| GB10 | draft_noncausal | 16 | 3 | 100 | exact | 11.2 | 10.81 | 15.81 |
| GB10 | draft_noncausal | 16 | 4 | 100 | C=32768 | 16.74 | 15.27 | 16.89 |
| GB10 | q3_target | 4 | 1 | 100 | C=32768 | 5.65 | 5.33 | 7.83 |
| GB10 | q3_target | 4 | 1 | 100 | exact | 5.41 | 5.36 | 7.9 |
| GB10 | q3_target | 4 | 2 | 4000 | C=32768 | 142.6 | 196.06 | 141.84 |
| GB10 | q3_target | 4 | 2 | 4000 | exact | 142.95 | 195.46 | 142.22 |
| GB10 | q3_target | 4 | 3 | 4000 | C=32768 | 208.29 | 225.21 | 207.83 |
| GB10 | q3_target | 4 | 3 | 4000 | exact | 208.05 | 226.24 | 207.26 |
| GB10 | q3_target | 4 | 4 | 100 | C=32768 | 14.89 | 17.09 | 14.87 |
| GB10 | q3_target | 4 | 4 | 4000 | C=32768 | 276.66 | 312.36 | 273.93 |
| GB10 | q3_target | 8 | 1 | 100 | exact | 5.74 | 5.41 | 9.21 |
| GB10 | q3_target | 9 | 1 | 100 | C=32768 | 6.18 | 5.36 | 9.2 |
| GB10 | q3_target | 9 | 1 | 100 | exact | 5.54 | 5.44 | 9.07 |
| GB10 | q3_target | 9 | 2 | 100 | C=32768 | 12.94 | 11.21 | 14.96 |
| GB10 | q3_target | 9 | 2 | 4000 | C=32768 | 153.5 | 196.25 | 148.6 |
| GB10 | q3_target | 9 | 3 | 100 | C=32768 | 13.07 | 12.83 | 15.82 |
| GB10 | q3_target | 9 | 3 | 100 | exact | 12.74 | 12.73 | 16.49 |
| GB10 | q3_target | 9 | 4 | 100 | C=32768 | 17.58 | 17.34 | 17.57 |
| GB10 | q3_target | 12 | 1 | 100 | exact | 5.88 | 5.56 | 9.38 |
| GB10 | q3_target | 12 | 3 | 100 | exact | 12.96 | 12.9 | 16.95 |
| GB10 | q3_target | 16 | 1 | 100 | C=32768 | 6.78 | 5.61 | 9.98 |
| GB10 | q3_target | 16 | 1 | 100 | exact | 6.39 | 5.68 | 9.75 |
| GB10 | q3_target | 16 | 2 | 100 | C=32768 | 12.73 | 11.6 | 16.14 |
| GB10 | q3_target | 16 | 3 | 100 | C=32768 | 14.53 | 13.12 | 17.34 |
| GB10 | q3_target | 16 | 3 | 100 | exact | 14.37 | 13.19 | 17.4 |
| GB10 | q3_target | 16 | 4 | 100 | C=32768 | 19.56 | 17.7 | 18.93 |

The two RTX 5090 exact-capacity ctx=4000 losses were rerun in a separate call
and passed: draft Q4 B2 measured 25.0 us; draft Q12 B1 measured 15.1 us.
They are noise. Every row at ctx >=300 is at or below the floor on RTX 5090 and
RTX PRO 6000. At ctx >=1000 the candidate is 2–15x faster than vLLM FA2.

ctx=100 remains below target: RTX 5090 has 16 rows at C=32768, B>=2
(10–17% slower than FA2); RTX PRO 6000 has 66 rows (1–40%). Optional GB10 has
42 losses overall, including DRAM-bound ctx=4000 near-ties with FlashInfer.
FA2 is one kernel launch. Our split op is attention + merge, and at large
capacity the occupancy-sized grid keeps idle splits.

## Negative results this round (tried, measured, reverted)

| experiment | result |
| --- | --- |
| Split K/V single-buffer pipeline, STAGES=1, FA-style staggered K/V loads. | Slower on GB10 and RTX PRO 6000. GB10 B=1 causal Q9 16/4 L4095: 22.7 vs 16.5 us. |
| Inline last-CTA merge with gpu-scope fence, atomic arrival and serial partial reads. | Correct but slower everywhere by 1–15 us per layer. RTX PRO 6000 Q16 B=1 ctx100: 8.6 vs 6.0 us. |
| Runtime single split for unions <=128 keys. | Slower at engine capacity: Q16 B4 ctx100 8.3 vs 7.1 us. The merge launch remains and the walk doubles. |
| Always-compiled piece loop. | Costs one replay step, up to 1.14x. Compile it only where needed. |

## SOL (L2-aware, informational)

| device | utilization |
| --- | --- |
| RTX 5090 | 9.5–62.4% |
| RTX PRO 6000 | 9.4–67.8% |
| GB10 | 10.0–68.4% |

The DF2 goal of SOL >=90% is **not met**. Short rows are bounded by graph-node
and replay-step floors. These are informational L2-aware useful-work estimates,
using the ceiling definition in the market gate protocol.

## Remaining distance and next steps

- GB10: 5 real L2-capacity rows remain, plus 1 unclassified final loss.
  Classify the new row and seek a schedule that wins across independent processes.
- R4 ctx=100: try a runtime-adaptive single-launch schedule. Persistent CTAs
  choose split/rows from the device length, with merge folded into that schedule.
  The naive inline last-CTA merge was slower.
- SOL: close the distance to 90%. Measure useful-work gains separately from
  graph-node and replay-step floors.
