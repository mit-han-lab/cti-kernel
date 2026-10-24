<!-- SPDX-License-Identifier: MIT -->
# GDN replay decode: test receipts (K2 companion, target ctikernels 0.1.11)

Contract: `docs/gdn_replay_decode.md` (r2 API accepted by the engine; r2.1 = N1 docs).
Wheel SHA256: filled at release (not cut yet).

## Portable K2 (`GatedDeltaNetReplay`, Triton provider + torch reference), commit bea8d87

Suite `tests/gpu/test_gdn_replay.py` (S1 status ring and poison, G1 static graph buffers,
bridge, Q=16 and Q=8 zero/partial/full acceptance across wraparound and flush boundaries,
CUDA-graph capture B=1..4 with changing slots/cursors/accept counts, partial failure and
recovery, TP-sharded heads), run on 2026-10-14:

| Device | Driver | Result |
|---|---|---|
| RTX 5090 | 615.36 | 20 passed |
| RTX PRO 6000 Blackwell Server Edition | 580.126.09 | 20 passed |
| GB10 | 580.78 | 20 passed |

CPU suite at the same commit: 818 passed.

These are correctness receipts only. Precision (§10) and speed (§11) qualification are
reported separately and do not gate the portable delivery.

## K2 for 0.1.11: default dispatch and CuTe transaction phases

Default dispatch (`provider=None` on CUDA): on RTX 5090 and RTX PRO 6000 with
nvidia-cutlass-dsl 4.8.0, `verify`, `commit`, `reject`, `flush`, `bridge_step` and
`export_snapshot` run on the CuTe provider; `step` and the lifecycle phases stay on the
portable Triton provider. Every other device (GB10 included) uses the portable provider.
The CuTe phases reuse the product decode's arithmetic recipe: verify outputs, committed,
flushed and exported state are bitwise equal to consecutive product decode steps at the
same batch specialization (`test_product_bitwise_transaction_phases`: Q=8/16, zero/
partial/full acceptance, prefixes L-1, L, L+1, 3L+1, B=1/2/4, (16,32) and (16,48),
ordinary and heavy inputs).

GPU suite (`tests/gpu/test_gdn_replay.py` + `tests/gpu/test_gdn_replay_cute.py`, run in
a bitwise and a non-bitwise shard), branch kernels/gdn-k2:

| Device | Commit | Bitwise shard | Non-bitwise shard |
|---|---|---|---|
| RTX 5090 | 4742dd8 | passed | passed |
| RTX PRO 6000 Blackwell Server Edition | 4742dd8 | passed | passed |
| GB10 (portable default; CuTe tests run explicitly) | 4742dd8 | passed | passed at 89f979f; 4742dd8 pending |

4742dd8 accepts any verify length Q in [2, 32]. Its bitwise shard includes
`test_product_bitwise_any_q_per_row_acceptance` for Q=4,5,8,9,12,13,16,17 (every
accepted length m=0..Q per row, plus reject), and its contract shard runs the
all-acceptance and CUDA-graph tests at the new Q values. Earlier heads: a62e4af,
d1214a7 and 89f979f passed the same shards on RTX PRO 6000 and GB10.

The same CuTe phase code passed on RTX 5090 and RTX PRO 6000 in the development branch
(kernels/gdn-replay-cute ab0f3ec), and the full CuTe suite passed on GB10 there.

Precision (frozen FP64 judge, RTX PRO 6000, L=8, Q=8/16, position 2^33+7, fresh handoff,
16 completed full-schedule cases across (16,32) B=4, (16,48) B=1 and B=4, seeds 11/23,
ordinary and heavy): every CuTe output and state RMS/max ratio against the product decode
is exactly 1.0. The full ≥5-seed matrix on every device is still to run; this is not a
qualification claim. Devices on the portable path (GB10) keep the earlier portable
precision gap (committed-state RMS 1.2164785218284655x, max 2.820042502346004x of the
product decode; section below).

CPU suite (torch 2.13.0 CPU + cpu-ci group, no numpy, no Triton): 911 passed.

## CuTe DSL decode step (opt-in `provider="cute"`; not qualified, not promoted)

One fused launch per decode step (validation, status, ring append, replay from the FP32
checkpoint, gated RMSNorm, device-predicated flush). The history below predates the CuTe
transaction phases (section above).

GPU tests (`tests/gpu/test_gdn_replay.py` + `tests/gpu/test_gdn_replay_cute.py`, 81 tests):
RTX 5090 81 passed; RTX PRO 6000 79 passed, 2 failed (`test_product_bitwise_multiwindow`
B=4 (16,48): 1-ulp output mismatches against the product decode, whose dispatch on that
device differs; open). Decode outputs and flushed state are bitwise equal to the product
decode at B=1..4 on the RTX 5090.

Precision (frozen judge, RTX 5090, (16,32) B=1 seed 11, L=8 Q=8, ordinary): decode output
max ratio 1.0; worst output RMS 1.0000017190238517 (verify path, Triton fallback); state
RMS 1.2164785218284655 and state max 2.820042502346004 (materialized/committed state via
the Triton fallback arithmetic). Gate not passed: the remaining gaps sit in the portable
Triton phase kernels, which do not yet follow the product decode's arithmetic recipe.

Timing (cold regime, CUDA graphs, full L=8 windows, 20 interleaved pairs; median
product/cute ratio, >1 = CuTe faster), commit 5d10767 kernel:

| Device | Shape | B=1 | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|---|---|
| RTX 5090 | (16,32) | 0.449 | 0.607 | 0.864 | 0.807 | 0.982 |
| RTX 5090 | (16,48) | 0.558 | 0.729 | 0.814 | 1.053 | 1.059 |
| GB10 | (16,32) | 1.149 | 1.191 | 1.324 | – | – |
| GB10 | (16,48) | 1.277 | 1.354 | 1.379 | – | – |

RTX 5090 self-pair ε was large on that node at small B (0.12–0.72), so the 5090 crossover
is provisional: (16,48) B=6, (16,32) not observed through B=8. Warm-L2 kernel profile
(RTX 5090, B=1): CuTe 6.1–6.8 µs at window phase 0 vs product 2.2–3.6 µs; controls
1.8–2.8 µs and ~0.35 µs per replayed token dominate. SGLang comparators not yet run.

## K2 publication fix (vectorized multi-CTA publish), 2026-10-14

Root cause of ~74 us per replay call in the DFlash trace: verify's publish copied
Q raw rows per batch row with one 128-thread CTA and scalar BF16 accesses
(Q*W/128 dependent iterations per thread). Publish now uses 16-byte copies, and
verify's copy spreads over ceil(Q*W/4096) CTAs per row. Arithmetic is unchanged.

GPU tests (tests/gpu/test_gdn_replay.py + test_gdn_replay_cute.py, kernel code of
this commit): RTX 5090 205 passed (88 bitwise + 117 other), RTX PRO 6000 205
passed (88 + 117), GB10 205 passed; 0 failed. Bitwise tests compare CuTe phase
outputs/state against the product decode, so outputs and state are unchanged.
CPU tests/modules: 794 passed.

RTX 5090, shape (16,32), verify + commit (accept Q//2), device time per layer per
round (`benchmark/kernel_tuning/gdn_k2_round.py`, eager, torch profiler), scalar ->
vector publication, in us:

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 67.3 -> 24.8 | 68.4 -> 26.0 | 70.9 -> 27.3 | 72.6 -> 29.1 |
| 5 | 77.5 -> 26.1 | 78.7 -> 27.2 | 81.0 -> 28.6 | 83.3 -> 30.1 |
| 8 | 108.1 -> 31.4 | 108.7 -> 32.1 | 112.9 -> 34.0 | 114.5 -> 35.5 |
| 9 | 119.1 -> 33.7 | 119.8 -> 34.3 | 123.9 -> 36.3 | 125.6 -> 37.8 |
| 12 | 149.7 -> 38.8 | 150.2 -> 39.5 | 155.8 -> 41.7 | 156.9 -> 42.7 |
| 16 | 190.9 -> 45.7 | 191.6 -> 45.8 | 197.8 -> 48.1 | 199.2 -> 48.5 |

Publish is now 5.6-6.5 us per round (both phases); compute (16-38 us) dominates.

## K2 verify comparators (vLLM 0.31.0, SGLang 72730d6), 2026-10-14

`benchmark/kernel_tuning/gdn_verify_compare.py`: one layer-round of speculative
verify for Qwen3.5 GDN, shape (16,32), accepted Q//2 per row, each provider
captured as a CUDA graph; 20 interleaved pairs per comparator, ε from 20
self-pairs; warm L2 with the fixture reset outside the timed interval. Ours =
K2 verify + commit with checks="strict" (control + compute + publish per phase,
six launches per round; commit 6972a8f kernels). Trusted mode was not timed here. vLLM dispatch:
`qwen_gdn_linear_attn.py:_forward_core` -> `causal_conv1d_update` ->
`fused_sigmoid_gating_delta_rule_update` (per-position state cache; selection at
the next verify) -> `layer_norm_fwd`. SGLang dispatch: `GDNAttnBackend.forward_extend`
/ `TritonGDNKernel.target_verify` -> `causal_conv1d_update` ->
`fused_sigmoid_gating_delta_rule_update` -> `_layer_norm_fwd` ->
`scatter_mamba_states_after_mtp_verify`. Source hashes are in each JSON row.
Outputs agree with both (max |diff| 0.031 in BF16, allclose in every row).

### RTX 5090

us per layer-round (ours / vLLM / SGLang), speedup vs the faster comparator, eps:

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 21.5 / 23.2 / 19.9: **0.957** (ε 0.095) | 25.1 / 24.6 / 24.0: **0.967** (ε 0.105) | 27.6 / 32.8 / 30.2: **1.077** (ε 0.085) | 30.7 / 34.8 / 36.7: **1.125** (ε 0.111) |
| 5 | 21.8 / 23.7 / 22.0: **1.020** (ε 0.091) | 25.3 / 30.7 / 28.1: **1.104** (ε 0.094) | 28.4 / 34.8 / 33.6: **1.130** (ε 0.071) | 32.8 / 38.9 / 41.0: **1.119** (ε 0.191) |
| 8 | 24.3 / 28.6 / 30.2: **1.167** (ε 0.076) | 28.0 / 36.9 / 37.2: **1.246** (ε 0.066) | 32.8 / 43.0 / 46.1: **1.168** (ε 0.086) | 38.9 / 49.2 / 60.4: **1.264** (ε 0.166) |
| 9 | 25.6 / 30.7 / 32.8: **1.196** (ε 0.059) | 29.5 / 38.9 / 41.1: **1.187** (ε 0.069) | 38.9 / 45.1 / 51.2: **1.196** (ε 0.180) | 41.0 / 55.3 / 69.6: **1.350** (ε 0.159) |
| 12 | 27.6 / 37.9 / 43.4: **1.284** (ε 0.039) | 32.1 / 45.1 / 50.1: **1.223** (ε 0.061) | 43.0 / 59.4 / 68.6: **1.382** (ε 0.169) | 45.0 / 79.8 / 93.2: **1.773** (ε 0.161) |
| 16 | 31.0 / 45.1 / 54.6: **1.285** (ε 0.052) | 38.2 / 54.3 / 67.7: **1.351** (ε 0.132) | 47.1 / 84.0 / 93.2: **1.826** (ε 0.177) | 49.1 / 110.6 / 124.9: **2.272** (ε 0.147) |

### RTX PRO 6000

us per layer-round (ours / vLLM / SGLang), speedup vs the faster comparator, eps:

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 27.1 / 28.2 / 24.6: **0.920** (ε 0.176) | 30.6 / 30.7 / 29.1: **0.975** (ε 0.125) | 32.6 / 36.9 / 33.8: **1.041** (ε 0.074) | 35.0 / 41.0 / 39.1: **1.109** (ε 0.160) |
| 5 | 27.2 / 30.1 / 26.7: **1.017** (ε 0.072) | 30.6 / 32.8 / 32.9: **1.078** (ε 0.048) | 33.8 / 41.0 / 40.6: **1.166** (ε 0.071) | 36.0 / 43.0 / 44.3: **1.106** (ε 0.052) |
| 8 | 30.6 / 34.8 / 36.4: **1.110** (ε 0.050) | 34.7 / 43.0 / 45.3: **1.175** (ε 0.060) | 38.8 / 49.2 / 52.5: **1.191** (ε 0.049) | 43.0 / 53.3 / 63.5: **1.175** (ε 0.118) |
| 9 | 33.0 / 36.8 / 39.7: **1.110** (ε 0.048) | 35.9 / 45.1 / 49.4: **1.187** (ε 0.074) | 41.7 / 51.2 / 56.9: **1.131** (ε 0.048) | 49.2 / 57.3 / 69.6: **1.208** (ε 0.157) |
| 12 | 34.9 / 41.0 / 49.8: **1.163** (ε 0.034) | 39.4 / 51.2 / 60.8: **1.237** (ε 0.068) | 45.6 / 61.5 / 73.6: **1.228** (ε 0.068) | 53.2 / 75.8 / 93.7: **1.408** (ε 0.148) |
| 16 | 39.0 / 51.2 / 67.0: **1.292** (ε 0.046) | 43.6 / 61.5 / 79.3: **1.327** (ε 0.044) | 57.3 / 79.9 / 94.2: **1.397** (ε 0.157) | 59.4 / 112.6 / 131.6: **1.833** (ε 0.161) |

### GB10

us per layer-round (ours / vLLM / SGLang), speedup vs the faster comparator, eps:

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 34.7 / 42.0 / 66.9: **1.230** (ε 0.254) | 53.2 / 92.9 / 132.7: **1.841** (ε 0.278) | 74.4 / 152.6 / 203.4: **2.041** (ε 0.599) | 104.1 / 207.5 / 282.5: **1.975** (ε 0.035) |
| 5 | 34.6 / 55.3 / 77.8: **1.716** (ε 0.181) | 55.3 / 118.6 / 158.5: **2.154** (ε 0.169) | 74.6 / 190.6 / 252.9: **2.521** (ε 0.056) | 109.6 / 256.0 / 339.1: **2.343** (ε 0.296) |
| 8 | 40.8 / 92.7 / 111.5: **2.206** (ε 0.226) | 64.2 / 194.5 / 235.4: **3.156** (ε 0.128) | 84.8 / 296.3 / 368.3: **3.513** (ε 0.050) | 119.6 / 388.4 / 488.6: **3.246** (ε 0.077) |
| 9 | 44.5 / 105.5 / 122.9: **2.393** (ε 0.229) | 71.4 / 218.8 / 263.3: **3.260** (ε 0.207) | 91.1 / 326.8 / 405.8: **3.583** (ε 0.059) | 131.2 / 433.3 / 541.7: **3.299** (ε 0.094) |
| 12 | 48.6 / 140.3 / 161.8: **3.072** (ε 0.231) | 78.1 / 287.7 / 343.2: **3.689** (ε 0.126) | 97.3 / 429.1 / 522.2: **4.411** (ε 0.072) | 147.5 / 569.3 / 693.2: **3.843** (ε 0.442) |
| 16 | 53.2 / 188.5 / 214.7: **3.524** (ε 0.220) | 88.1 / 381.6 / 450.8: **4.312** (ε 0.314) | 110.6 / 562.3 / 670.7: **5.036** (ε 0.244) | 161.9 / 746.5 / 897.6: **4.618** (ε 0.107) |

Ours trails only at Q=4, B=1–2 on RTX 5090 and RTX PRO 6000 (0.92–0.975, within ε):
at the smallest window our fixed per-round cost (two launches plus the commit
publication) is not yet amortized.

## K2 for 0.1.12: GPU tests and three-variant comparator (kernels/gdn-k2-compute 8671924), 2026-10-15

GPU tests at 8671924 (`tests/gpu/test_gdn_replay*.py`, seven shards): RTX 5090
371 passed, RTX PRO 6000 371 passed, GB10 371 passed (shards 55/55/54/53/52/51/51);
0 failed on every device.

Comparator: `benchmark/kernel_tuning/gdn_verify_compare.py` with the fair fixtures
of a1f82fe (each provider times exactly its own dispatch, including its own q/k/v
packing; copy audit asserts no extra copies). Rows were measured at a1f82fe; the
kernel sources (`cute/_kernel.py`, `cute/_phase_launch.py`, `cute/_phases.py`) are
byte-identical at 8671924, which changed only the phase-arrivals binding in
`_api.py`. Shape (16,32), accepted Q//2 per row, CUDA graph per provider, warm L2,
20 interleaved pairs, ε = p95 |ratio − 1| over 20 self-pairs. Comparators: vLLM
0.31.0 and SGLang 72730d6 (source hashes in every JSON row). Every row passes the
output correctness check against both comparators. Cells: µs per layer-round
(ours / vLLM / SGLang): speedup over the faster comparator (bold when ≥ 1 + ε).
GB10 runs are pinned to the X925 cores. Paired A/B always runs inside one call;
the three variants on one device ran in separate calls and may differ in GPU.

Summary, speedup over the faster of vLLM/SGLang at B=1–4 × Q=4/5/6/8/9/12/16:

| variant | RTX 5090 | RTX PRO 6000 | GB10 |
|---|---|---|---|
| strict (control + compute + publish) | 1.099–2.374; 9 rows within ε (wide ε on a shared node) | 0.971–2.000; Q=4 B=1–2 and Q=5 B=1 within ε | 1.542–5.269 |
| trusted, publication not fused | 1.312–2.875 | 1.196–2.467 | 1.664–5.708 |
| trusted + fused (default) | 1.331–2.723 | 1.428–2.629 | 1.941–5.712 |

Trusted and trusted + fused clear 1 + ε in every row on all three devices.

### RTX 5090: trusted, fused=off (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 16.0 / 25.8 / 22.0: **1.381** (ε 0.113) | 19.0 / 27.6 / 25.2: **1.312** (ε 0.099) | 22.0 / 34.8 / 31.2: **1.432** (ε 0.060) | 23.1 / 38.9 / 37.3: **1.537** (ε 0.042) |
| 5 | 17.4 / 27.6 / 23.6: **1.359** (ε 0.089) | 19.8 / 29.7 / 28.6: **1.453** (ε 0.113) | 22.4 / 38.9 / 35.2: **1.524** (ε 0.086) | 24.6 / 43.0 / 43.0: **1.708** (ε 0.084) |
| 6 | 18.5 / 29.0 / 26.6: **1.441** (ε 0.053) | 20.5 / 36.9 / 32.8: **1.567** (ε 0.095) | 24.2 / 43.0 / 38.5: **1.516** (ε 0.045) | 26.6 / 47.1 / 50.2: **1.769** (ε 0.113) |
| 8 | 20.1 / 31.7 / 31.8: **1.565** (ε 0.225) | 22.5 / 41.0 / 38.7: **1.667** (ε 0.091) | 26.8 / 47.1 / 49.0: **1.714** (ε 0.100) | 28.7 / 53.2 / 63.5: **1.800** (ε 0.146) |
| 9 | 21.8 / 31.8 / 35.5: **1.485** (ε 0.045) | 24.3 / 43.0 / 42.7: **1.676** (ε 0.036) | 30.4 / 49.2 / 54.3: **1.600** (ε 0.111) | 32.8 / 63.5 / 72.2: **2.000** (ε 0.135) |
| 12 | 23.8 / 41.0 / 45.4: **1.641** (ε 0.081) | 26.8 / 47.1 / 52.9: **1.714** (ε 0.074) | 34.8 / 63.5 / 69.6: **1.824** (ε 0.129) | 36.9 / 86.0 / 96.3: **2.333** (ε 0.123) |
| 16 | 26.3 / 49.2 / 57.7: **1.643** (ε 0.068) | 30.7 / 56.3 / 68.6: **1.767** (ε 0.070) | 39.8 / 90.1 / 96.3: **2.200** (ε 0.114) | 40.9 / 116.7 / 129.0: **2.875** (ε 0.132) |

range 1.312-2.875; rows not above 1+ε: []

### RTX 5090: strict, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 22.6 / 25.3 / 23.1: 1.099 (ε 0.165) | 24.7 / 28.8 / 27.0: 1.160 (ε 0.637) | 27.9 / 30.7 / 32.4: 1.129 (ε 0.152) | 29.9 / 34.8 / 37.2: **1.138** (ε 0.135) |
| 5 | 22.2 / 26.8 / 25.0: **1.190** (ε 0.133) | 24.7 / 30.7 / 29.2: 1.252 (ε 0.857) | 29.0 / 34.8 / 35.5: 1.112 (ε 0.140) | 31.2 / 38.9 / 41.4: **1.188** (ε 0.059) |
| 6 | 23.1 / 29.0 / 28.4: 1.280 (ε 2.571) | 26.0 / 32.8 / 33.3: **1.296** (ε 0.157) | 30.5 / 38.9 / 38.8: **1.196** (ε 0.100) | 33.0 / 43.0 / 48.7: **1.302** (ε 0.100) |
| 8 | 24.6 / 31.0 / 32.8: **1.266** (ε 0.198) | 28.8 / 36.9 / 38.4: **1.251** (ε 0.102) | 33.5 / 43.0 / 47.0: **1.250** (ε 0.075) | 35.8 / 51.2 / 62.0: **1.426** (ε 0.180) |
| 9 | 25.8 / 32.9 / 35.3: **1.320** (ε 0.241) | 30.7 / 38.9 / 42.7: 1.250 (ε 0.335) | 36.8 / 45.1 / 50.4: 1.262 (ε 0.437) | 38.9 / 55.3 / 70.7: **1.424** (ε 0.120) |
| 12 | 28.4 / 35.8 / 44.8: **1.302** (ε 0.114) | 32.9 / 45.1 / 51.6: **1.325** (ε 0.141) | 40.8 / 59.4 / 67.5: 1.482 (ε 2.248) | 42.0 / 79.9 / 96.3: **1.858** (ε 0.101) |
| 16 | 31.2 / 45.1 / 57.1: **1.361** (ε 0.127) | 35.2 / 53.2 / 68.1: **1.427** (ε 0.102) | 45.2 / 86.0 / 95.5: **1.896** (ε 0.108) | 47.1 / 110.6 / 129.2: **2.374** (ε 0.514) |

range 1.099-2.374; rows not above 1+ε: [(4, 1, 1.099, 0.165), (4, 2, 1.16, 0.637), (4, 3, 1.129, 0.152), (5, 2, 1.252, 0.857), (5, 3, 1.112, 0.14), (6, 1, 1.28, 2.571), (9, 2, 1.25, 0.335), (9, 3, 1.262, 0.437), (12, 3, 1.482, 2.248)]

### RTX 5090: trusted, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 17.5 / 26.2 / 23.0: **1.331** (ε 0.103) | 19.0 / 28.1 / 26.4: **1.391** (ε 0.108) | 21.0 / 31.4 / 32.1: **1.471** (ε 0.065) | 23.3 / 34.8 / 37.4: **1.417** (ε 0.087) |
| 5 | 18.6 / 27.8 / 26.2: **1.410** (ε 0.100) | 20.3 / 32.0 / 30.2: **1.488** (ε 0.093) | 22.4 / 34.8 / 36.3: **1.477** (ε 0.088) | 24.5 / 38.9 / 42.8: **1.510** (ε 0.134) |
| 6 | 18.8 / 29.4 / 28.5: **1.509** (ε 0.089) | 20.7 / 31.7 / 33.3: **1.549** (ε 0.069) | 23.7 / 38.9 / 39.1: **1.522** (ε 0.070) | 26.0 / 43.0 / 49.5: **1.647** (ε 0.108) |
| 8 | 20.1 / 32.1 / 33.9: **1.608** (ε 0.054) | 23.1 / 36.9 / 40.0: **1.504** (ε 0.097) | 26.9 / 43.0 / 47.7: **1.594** (ε 0.134) | 28.4 / 49.2 / 63.6: **1.691** (ε 0.150) |
| 9 | 21.7 / 33.7 / 36.0: **1.565** (ε 0.086) | 25.0 / 38.4 / 42.7: **1.477** (ε 0.048) | 30.2 / 45.1 / 51.8: **1.533** (ε 0.112) | 31.6 / 55.3 / 71.9: **1.761** (ε 0.136) |
| 12 | 23.2 / 36.8 / 44.7: **1.553** (ε 0.061) | 27.6 / 43.0 / 53.8: **1.538** (ε 0.040) | 33.5 / 61.4 / 70.8: **1.834** (ε 0.136) | 34.9 / 79.9 / 95.7: **2.288** (ε 0.161) |
| 16 | 27.2 / 45.0 / 57.3: **1.547** (ε 0.053) | 31.4 / 53.2 / 69.0: **1.596** (ε 0.073) | 38.4 / 84.0 / 95.1: **2.177** (ε 0.125) | 39.7 / 110.6 / 128.2: **2.723** (ε 0.144) |

range 1.331-2.723; rows not above 1+ε: []

### RTX PRO 6000: strict, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 24.4 / 28.7 / 23.9: 0.971 (ε 0.085) | 28.1 / 30.7 / 28.8: 1.036 (ε 0.083) | 30.2 / 36.9 / 33.3: **1.111** (ε 0.070) | 32.1 / 41.0 / 38.2: **1.174** (ε 0.088) |
| 5 | 25.0 / 30.9 / 26.8: 1.068 (ε 0.083) | 27.9 / 32.8 / 31.3: **1.133** (ε 0.061) | 31.2 / 41.0 / 38.9: **1.248** (ε 0.051) | 33.6 / 45.1 / 43.1: **1.223** (ε 0.067) |
| 6 | 26.2 / 33.0 / 29.9: **1.156** (ε 0.056) | 29.8 / 36.9 / 35.9: **1.208** (ε 0.081) | 32.5 / 45.1 / 43.1: **1.297** (ε 0.054) | 34.8 / 47.1 / 47.6: **1.264** (ε 0.075) |
| 8 | 27.9 / 35.3 / 36.4: **1.261** (ε 0.076) | 31.8 / 43.0 / 43.7: **1.313** (ε 0.082) | 36.4 / 51.2 / 51.3: **1.276** (ε 0.067) | 39.9 / 55.3 / 58.4: **1.318** (ε 0.095) |
| 9 | 29.9 / 36.9 / 39.5: **1.227** (ε 0.084) | 32.8 / 47.1 / 47.7: **1.334** (ε 0.046) | 39.0 / 53.3 / 56.3: **1.228** (ε 0.049) | 42.9 / 59.4 / 69.6: **1.261** (ε 0.060) |
| 12 | 32.2 / 43.0 / 50.5: **1.293** (ε 0.071) | 36.5 / 53.3 / 60.4: **1.351** (ε 0.052) | 43.0 / 63.5 / 71.6: **1.292** (ε 0.070) | 53.2 / 75.8 / 90.1: **1.462** (ε 0.182) |
| 16 | 35.9 / 51.2 / 65.3: **1.411** (ε 0.065) | 40.7 / 63.5 / 75.1: **1.410** (ε 0.035) | 55.1 / 87.1 / 92.2: **1.554** (ε 0.175) | 59.4 / 118.8 / 130.5: **2.000** (ε 0.177) |

range 0.971-2.000; rows not above 1+ε: [(4, 1, 0.971, 0.085), (4, 2, 1.036, 0.083), (5, 1, 1.068, 0.083)]

### RTX PRO 6000: trusted, fused=off (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 19.4 / 27.4 / 23.2: **1.196** (ε 0.067) | 21.4 / 29.3 / 28.2: **1.317** (ε 0.050) | 23.4 / 34.7 / 32.9: **1.353** (ε 0.090) | 25.4 / 36.9 / 37.2: **1.434** (ε 0.095) |
| 5 | 19.3 / 29.2 / 27.2: **1.387** (ε 0.096) | 21.8 / 32.9 / 31.6: **1.420** (ε 0.105) | 25.1 / 38.1 / 36.5: **1.459** (ε 0.088) | 25.7 / 41.0 / 42.7: **1.473** (ε 0.062) |
| 6 | 21.3 / 30.4 / 29.4: **1.388** (ε 0.067) | 23.4 / 35.0 / 35.6: **1.507** (ε 0.061) | 26.2 / 40.3 / 40.7: **1.505** (ε 0.074) | 27.9 / 45.1 / 48.4: **1.455** (ε 0.057) |
| 8 | 23.1 / 33.7 / 35.7: **1.452** (ε 0.080) | 25.3 / 38.1 / 42.3: **1.516** (ε 0.048) | 29.5 / 47.1 / 51.1: **1.502** (ε 0.055) | 31.1 / 52.4 / 62.8: **1.623** (ε 0.078) |
| 9 | 25.2 / 35.1 / 39.5: **1.394** (ε 0.091) | 27.5 / 42.6 / 48.3: **1.474** (ε 0.055) | 31.9 / 49.2 / 54.9: **1.477** (ε 0.057) | 35.8 / 55.3 / 68.1: **1.549** (ε 0.175) |
| 12 | 27.3 / 41.3 / 49.6: **1.500** (ε 0.044) | 29.9 / 49.2 / 59.0: **1.598** (ε 0.067) | 36.3 / 60.5 / 67.7: **1.483** (ε 0.053) | 42.0 / 75.8 / 90.6: **1.774** (ε 0.188) |
| 16 | 31.2 / 49.1 / 64.6: **1.564** (ε 0.055) | 34.0 / 59.4 / 74.7: **1.643** (ε 0.074) | 45.6 / 77.8 / 90.9: **1.650** (ε 0.169) | 47.1 / 115.3 / 127.0: **2.467** (ε 0.168) |

range 1.196-2.467; rows not above 1+ε: []

### RTX PRO 6000: trusted, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 16.4 / 28.4 / 23.9: **1.428** (ε 0.124) | 18.0 / 30.7 / 28.5: **1.542** (ε 0.071) | 19.8 / 34.8 / 33.1: **1.676** (ε 0.089) | 21.6 / 41.0 / 38.2: **1.720** (ε 0.100) |
| 5 | 16.0 / 29.4 / 26.3: **1.672** (ε 0.067) | 18.0 / 32.0 / 31.2: **1.723** (ε 0.098) | 20.3 / 41.0 / 38.5: **1.860** (ε 0.082) | 22.5 / 43.0 / 42.8: **1.738** (ε 0.108) |
| 6 | 17.2 / 31.5 / 29.5: **1.714** (ε 0.070) | 19.4 / 34.8 / 34.2: **1.793** (ε 0.099) | 21.5 / 41.0 / 43.0: **1.802** (ε 0.059) | 23.6 / 47.1 / 47.6: **1.869** (ε 0.055) |
| 8 | 19.0 / 33.8 / 35.3: **1.826** (ε 0.129) | 19.7 / 41.0 / 42.5: **1.971** (ε 0.170) | 24.1 / 49.2 / 49.9: **1.923** (ε 0.114) | 27.7 / 55.3 / 59.4: **1.934** (ε 0.122) |
| 9 | 21.3 / 36.7 / 38.1: **1.724** (ε 0.046) | 22.9 / 45.1 / 47.1: **1.918** (ε 0.079) | 27.5 / 53.2 / 54.5: **1.817** (ε 0.059) | 30.7 / 59.4 / 64.7: **1.813** (ε 0.148) |
| 12 | 22.4 / 41.0 / 48.7: **1.845** (ε 0.069) | 25.2 / 53.2 / 58.6: **1.979** (ε 0.078) | 31.6 / 61.4 / 69.1: **1.781** (ε 0.058) | 36.8 / 84.0 / 92.2: **2.263** (ε 0.155) |
| 16 | 28.3 / 51.2 / 64.0: **1.766** (ε 0.068) | 31.9 / 61.4 / 73.8: **1.791** (ε 0.057) | 40.7 / 81.9 / 93.2: **1.952** (ε 0.163) | 43.0 / 114.7 / 129.0: **2.629** (ε 0.154) |

range 1.428-2.629; rows not above 1+ε: []

### GB10: strict, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 32.8 / 49.1 / 67.1: **1.542** (ε 0.229) | 51.2 / 92.6 / 134.1: **1.792** (ε 0.176) | 71.6 / 152.0 / 199.6: **2.102** (ε 0.028) | 99.3 / 210.9 / 283.7: **2.120** (ε 0.029) |
| 5 | 32.7 / 59.0 / 77.2: **1.698** (ε 0.217) | 53.2 / 123.9 / 156.9: **2.352** (ε 0.185) | 73.7 / 193.5 / 254.2: **2.631** (ε 0.099) | 103.1 / 254.9 / 332.8: **2.465** (ε 0.010) |
| 6 | 36.8 / 71.6 / 90.2: **2.009** (ε 0.208) | 56.8 / 150.5 / 183.4: **2.644** (ε 0.163) | 76.8 / 230.0 / 286.4: **2.994** (ε 0.032) | 113.9 / 301.0 / 379.9: **2.625** (ε 0.360) |
| 8 | 40.9 / 96.2 / 111.6: **2.362** (ε 0.233) | 63.4 / 197.6 / 233.4: **3.130** (ε 0.162) | 82.0 / 295.9 / 365.9: **3.596** (ε 0.061) | 117.8 / 385.2 / 481.3: **3.255** (ε 0.034) |
| 9 | 44.4 / 108.2 / 121.3: **2.401** (ε 0.232) | 69.6 / 221.2 / 264.2: **3.137** (ε 0.174) | 89.8 / 330.0 / 406.8: **3.664** (ε 0.045) | 128.0 / 430.6 / 534.5: **3.368** (ε 0.087) |
| 12 | 48.1 / 145.4 / 159.7: **2.930** (ε 0.220) | 78.7 / 292.6 / 342.5: **3.716** (ε 0.140) | 95.3 / 434.2 / 518.1: **4.519** (ε 0.158) | 147.3 / 569.0 / 684.7: **3.868** (ε 0.112) |
| 16 | 53.8 / 192.2 / 212.8: **3.532** (ε 0.201) | 91.5 / 387.1 / 451.2: **4.257** (ε 0.184) | 107.2 / 565.9 / 671.5: **5.269** (ε 0.030) | 169.0 / 749.7 / 893.9: **4.414** (ε 0.205) |

range 1.542-5.269; rows not above 1+ε: []

### GB10: trusted, fused=off (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 23.6 / 43.0 / 66.6: **1.664** (ε 0.132) | 48.1 / 94.2 / 143.4: **1.932** (ε 0.453) | 62.5 / 153.9 / 203.0: **2.465** (ε 0.036) | 94.2 / 213.0 / 285.0: **2.282** (ε 0.362) |
| 5 | 29.0 / 60.6 / 78.5: **2.127** (ε 0.249) | 47.0 / 127.0 / 157.2: **2.660** (ε 0.145) | 63.8 / 194.5 / 252.9: **3.047** (ε 0.037) | 98.5 / 252.9 / 335.0: **2.563** (ε 0.075) |
| 6 | 31.8 / 70.6 / 91.1: **2.193** (ε 0.287) | 51.4 / 149.6 / 181.9: **2.921** (ε 0.124) | 69.6 / 227.8 / 280.6: **3.280** (ε 0.043) | 105.5 / 300.1 / 382.8: **2.846** (ε 0.030) |
| 8 | 34.9 / 95.2 / 112.8: **2.798** (ε 0.258) | 56.8 / 198.9 / 234.5: **3.495** (ε 0.086) | 72.7 / 292.9 / 365.6: **4.067** (ε 0.028) | 112.6 / 388.1 / 484.4: **3.461** (ε 0.027) |
| 9 | 37.9 / 107.1 / 126.0: **2.833** (ε 0.246) | 64.5 / 222.6 / 265.7: **3.449** (ε 0.064) | 80.9 / 328.7 / 405.7: **4.083** (ε 0.053) | 124.9 / 438.8 / 539.9: **3.502** (ε 0.686) |
| 12 | 40.9 / 146.1 / 161.8: **3.442** (ε 0.222) | 74.3 / 294.4 / 340.5: **4.006** (ε 0.385) | 91.2 / 428.8 / 521.6: **4.731** (ε 0.397) | 131.9 / 569.0 / 686.1: **4.278** (ε 0.107) |
| 16 | 47.4 / 193.2 / 214.0: **4.012** (ε 0.225) | 85.0 / 385.0 / 449.8: **4.532** (ε 0.138) | 99.3 / 568.0 / 672.6: **5.708** (ε 0.023) | 149.3 / 748.6 / 895.6: **4.994** (ε 0.179) |

range 1.664-5.708; rows not above 1+ε: []

### GB10: trusted, fused=on (28 rows, correctness_ok all=True)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 22.7 / 46.3 / 66.5: **1.941** (ε 0.181) | 43.0 / 96.2 / 133.7: **2.246** (ε 0.136) | 62.7 / 153.6 / 197.6: **2.439** (ε 0.034) | 93.4 / 210.9 / 284.7: **2.250** (ε 0.022) |
| 5 | 24.8 / 56.7 / 78.6: **2.322** (ε 0.210) | 43.6 / 122.8 / 156.6: **2.966** (ε 0.357) | 65.8 / 194.5 / 248.8: **2.966** (ε 0.051) | 94.2 / 254.4 / 336.0: **2.702** (ε 0.029) |
| 6 | 27.9 / 71.5 / 88.2: **2.549** (ε 0.249) | 47.1 / 150.5 / 182.7: **3.239** (ε 0.129) | 67.6 / 230.4 / 287.4: **3.398** (ε 0.031) | 102.4 / 299.1 / 382.9: **2.902** (ε 0.030) |
| 8 | 30.6 / 97.2 / 112.8: **3.008** (ε 0.208) | 53.2 / 196.6 / 232.5: **3.685** (ε 0.161) | 74.7 / 294.8 / 362.2: **3.974** (ε 0.042) | 110.6 / 387.7 / 484.6: **3.508** (ε 0.015) |
| 9 | 35.3 / 106.5 / 124.5: **2.988** (ε 0.263) | 61.4 / 223.2 / 263.8: **3.651** (ε 0.195) | 77.8 / 328.7 / 405.5: **4.216** (ε 0.027) | 120.8 / 432.1 / 534.7: **3.575** (ε 0.091) |
| 12 | 39.0 / 144.5 / 161.8: **3.492** (ε 0.215) | 68.6 / 292.8 / 343.7: **4.292** (ε 0.147) | 87.9 / 432.8 / 520.6: **4.907** (ε 0.048) | 139.2 / 570.4 / 687.2: **4.105** (ε 0.281) |
| 16 | 44.2 / 195.3 / 211.7: **4.264** (ε 0.209) | 77.6 / 387.0 / 451.6: **4.975** (ε 0.113) | 99.3 / 568.3 / 671.2: **5.712** (ε 0.090) | 151.6 / 749.6 / 894.8: **4.944** (ε 0.215) |

range 1.941-5.712; rows not above 1+ε: []


## K2 R3b: phase launch policy for B=3–4 occupancy (kernels/gdn-k2-compute), 2026-10-15

What changed: the launch shape of the CuTe transaction phases (verify, commit, reject,
flush, bridge, export). Arithmetic, contract and trusted/strict semantics are
unchanged. Every launch variant is bitwise equal to the 8-warp baseline (GPU tests
below). Policy (`cute/_phase_launch.py:phase_launch`, per-device `cute/_config.py`):

- one wave of 32/64-row tiles (`B·Hv·128/tile ≤ SMs`): 8 warps, or 16 warps when
  Q > 8 (all Q tokens staged in one chunk);
- full-head CTAs that fit one wave (`B·Hv ≤ SMs`): 16 warps (512 threads, 103
  registers, no spills): one shared conv/gate prologue, one row per thread;
- more full-head CTAs than SMs (Hv=48, B=4 on RTX 5090/PRO 6000): 64-row tiles with
  launch bound 2 (≤128 registers, no spills; two CTAs per SM) instead of a second wave;
- GB10 never splits (DRAM-bound at B ≥ 2): full-head 16-warp CTAs.

Why not "≥ 2 CTAs per SM" everywhere: each CTA recomputes the conv/gate prologue
for its head, so 32/64-row tiles at 2–3 CTAs per SM lose to one 128-row CTA once
the head CTAs fill a wave (sweep `benchmark/kernel_tuning/gdn_k2_round.py --tiles
--warps --min-blocks`; e.g. RTX 5090 (16,32) B=4 Q=16: 128/8/1 29.2 µs, 64/8/2
30.6, 32/8/2 47.6, 128/16/1 26.1). A 16-warp full-head CTA reaches the same warp
occupancy as two 8-warp CTAs without duplicating the prologue. Registers (sm_120,
`cuobjdump --dump-resource-usage`): verify 128/8 178, 128/16 103, 64/8 131
(launch bound 2: 128, no spill); commit 128/8 252, 128/16 95–97, 64/8 114.

GPU tests at 2896df0 (`tests/gpu/test_gdn_replay.py`, `test_gdn_replay_cute.py`,
`test_gdn_replay_trusted_k2.py`; 732 tests incl. 72 new launch-variant bitwise and 24
graph launch-variant tests; 20 shards): RTX 5090 732 passed, RTX PRO 6000 732 passed,
GB10 732 passed; 0 failed on every device. Review follow-up d152fe9 (4-byte arrivals warm signature,
two-stream concurrent rounds): 19/19 selected tests pass on each device. CPU suite:
1058 passed, 2 skipped.

### Same-call A/B, legacy policy (tile by wave, 8 warps, 1 CTA/SM) vs shipped policy

`gdn_k2_round.py --tiles legacy auto auto legacy` at 17d17d6: trusted + fused + PDL,
device time per verify+commit round (accepted Q/2), 50 rounds per pass, two passes
per setting in ABBA order, one call per device and shape. Each row pins the
effective (tile, warps, min CTAs/SM) it reports. (Before 17d17d6 the benchmark's
`--min-blocks` default of 1 overrode the policy in `auto` rows, so an `auto` row at
Hv=48, B=4 ran 64/8/1 while reporting 64/8/2. The earlier sweep tables used explicit
settings and were not affected, but they were re-timed here with the fixed tool.)
Devices: one retail RTX 5090, one PRO 6000, one GB10 pinned to X925
cores. Cells use the faster of the two passes.

**RTX 5090** (legacy → shipped policy, µs per verify+commit round, min of 2 ABBA passes × 50 rounds; speedup; shipped tile/warps/min CTAs per SM)

| shape, Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| (16,32) Q=4 | 10.6 → 10.6 (1.00×, 32/8/1) | 12.4 → 12.4 (1.00×, 64/8/1) | 14.8 → 14.5 (1.02×, 128/16/1) | 15.9 → 15.6 (1.02×, 128/16/1) |
| (16,32) Q=5 | 11.2 → 11.3 (1.00×, 32/8/1) | 13.1 → 13.1 (1.00×, 64/8/1) | 15.4 → 15.1 (1.02×, 128/16/1) | 16.5 → 16.3 (1.02×, 128/16/1) |
| (16,32) Q=6 | 12.0 → 11.9 (1.00×, 32/8/1) | 13.7 → 13.7 (1.00×, 64/8/1) | 16.8 → 16.1 (1.04×, 128/16/1) | 17.8 → 17.1 (1.04×, 128/16/1) |
| (16,32) Q=8 | 13.3 → 13.4 (1.00×, 32/8/1) | 15.1 → 15.1 (1.00×, 64/8/1) | 18.8 → 18.0 (1.05×, 128/16/1) | 19.7 → 19.0 (1.03×, 128/16/1) |
| (16,32) Q=9 | 15.1 → 13.6 (1.11×, 32/16/1) | 17.2 → 15.4 (1.12×, 64/16/1) | 21.1 → 18.6 (1.14×, 128/16/1) | 22.1 → 19.6 (1.13×, 128/16/1) |
| (16,32) Q=12 | 16.8 → 15.5 (1.08×, 32/16/1) | 18.9 → 17.4 (1.09×, 64/16/1) | 24.1 → 21.3 (1.13×, 128/16/1) | 25.0 → 22.2 (1.13×, 128/16/1) |
| (16,32) Q=16 | 19.3 → 18.3 (1.06×, 32/16/1) | 22.1 → 20.4 (1.08×, 64/16/1) | 28.4 → 24.9 (1.14×, 128/16/1) | 29.0 → 25.8 (1.12×, 128/16/1) |
| (16,48) Q=4 | 12.2 → 12.3 (1.00×, 64/8/1) | 15.7 → 15.3 (1.02×, 128/16/1) | 17.2 → 17.1 (1.01×, 128/16/1) | 28.1 → 26.5 (1.06×, 64/8/2) |
| (16,48) Q=5 | 12.7 → 12.8 (1.00×, 64/8/1) | 16.2 → 15.8 (1.02×, 128/16/1) | 17.6 → 17.6 (1.00×, 128/16/1) | 29.3 → 27.7 (1.06×, 64/8/2) |
| (16,48) Q=6 | 13.7 → 13.8 (0.99×, 64/8/1) | 17.4 → 16.9 (1.03×, 128/16/1) | 19.0 → 18.7 (1.02×, 128/16/1) | 31.9 → 29.9 (1.07×, 64/8/2) |
| (16,48) Q=8 | 15.3 → 15.3 (1.00×, 64/8/1) | 19.8 → 18.9 (1.05×, 128/16/1) | 20.8 → 20.5 (1.02×, 128/16/1) | 35.9 → 33.3 (1.08×, 64/8/2) |
| (16,48) Q=9 | 17.2 → 15.5 (1.11×, 64/16/1) | 22.2 → 19.5 (1.14×, 128/16/1) | 23.5 → 21.3 (1.10×, 128/16/1) | 40.4 → 37.4 (1.08×, 64/8/2) |
| (16,48) Q=12 | 19.0 → 17.5 (1.09×, 64/16/1) | 25.2 → 22.2 (1.14×, 128/16/1) | 26.3 → 23.9 (1.10×, 128/16/1) | 46.3 → 41.2 (1.13×, 64/8/2) |
| (16,48) Q=16 | 22.3 → 20.5 (1.09×, 64/16/1) | 29.6 → 25.8 (1.15×, 128/16/1) | 30.4 → 27.7 (1.10×, 128/16/1) | 54.6 → 48.2 (1.13×, 64/8/2) |

B=3–4 range 0.999–1.138×; all rows 0.994–1.145×; rows < 0.98: []

**RTX PRO 6000** (legacy → shipped policy, µs per verify+commit round, min of 2 ABBA passes × 50 rounds; speedup; shipped tile/warps/min CTAs per SM)

| shape, Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| (16,32) Q=4 | 13.3 → 13.3 (1.00×, 32/8/1) | 14.9 → 14.9 (1.00×, 64/8/1) | 17.7 → 17.5 (1.01×, 128/16/1) | 18.7 → 18.5 (1.01×, 128/16/1) |
| (16,32) Q=5 | 13.9 → 13.9 (1.00×, 32/8/1) | 15.5 → 15.5 (1.00×, 64/8/1) | 18.5 → 18.2 (1.02×, 128/16/1) | 19.5 → 19.4 (1.01×, 128/16/1) |
| (16,32) Q=6 | 15.0 → 15.1 (1.00×, 32/8/1) | 16.7 → 16.7 (1.00×, 64/8/1) | 20.1 → 19.5 (1.03×, 128/16/1) | 21.2 → 20.6 (1.03×, 128/16/1) |
| (16,32) Q=8 | 16.8 → 16.8 (1.00×, 32/8/1) | 18.6 → 18.6 (1.00×, 64/8/1) | 22.6 → 21.8 (1.03×, 128/16/1) | 23.6 → 22.9 (1.03×, 128/16/1) |
| (16,32) Q=9 | 18.9 → 17.0 (1.11×, 32/16/1) | 21.0 → 19.0 (1.11×, 64/16/1) | 25.5 → 22.5 (1.13×, 128/16/1) | 26.4 → 23.7 (1.11×, 128/16/1) |
| (16,32) Q=12 | 21.0 → 19.6 (1.07×, 32/16/1) | 23.3 → 21.5 (1.09×, 64/16/1) | 29.2 → 25.9 (1.13×, 128/16/1) | 30.1 → 27.1 (1.11×, 128/16/1) |
| (16,32) Q=16 | 24.4 → 23.2 (1.05×, 32/16/1) | 27.1 → 25.1 (1.08×, 64/16/1) | 34.3 → 30.4 (1.13×, 128/16/1) | 35.2 → 31.7 (1.11×, 128/16/1) |
| (16,48) Q=4 | 14.9 → 14.9 (1.00×, 64/8/1) | 18.0 → 18.1 (1.00×, 128/16/1) | 19.8 → 19.7 (1.01×, 128/16/1) | 32.8 → 29.5 (1.11×, 64/8/2) |
| (16,48) Q=5 | 15.5 → 15.5 (1.00×, 64/8/1) | 18.8 → 18.8 (1.00×, 128/16/1) | 20.6 → 20.5 (1.00×, 128/16/1) | 34.6 → 31.0 (1.11×, 64/8/2) |
| (16,48) Q=6 | 16.6 → 16.6 (1.00×, 64/8/1) | 20.4 → 20.1 (1.01×, 128/16/1) | 22.2 → 21.8 (1.02×, 128/16/1) | 37.7 → 33.9 (1.11×, 64/8/2) |
| (16,48) Q=8 | 18.4 → 18.4 (1.00×, 64/8/1) | 22.9 → 22.5 (1.02×, 128/16/1) | 24.7 → 24.1 (1.02×, 128/16/1) | 43.2 → 38.0 (1.14×, 64/8/2) |
| (16,48) Q=9 | 20.6 → 18.8 (1.10×, 64/16/1) | 25.8 → 23.2 (1.11×, 128/16/1) | 27.5 → 24.8 (1.11×, 128/16/1) | 48.8 → 43.1 (1.13×, 64/8/2) |
| (16,48) Q=12 | 23.2 → 21.4 (1.08×, 64/16/1) | 29.5 → 26.7 (1.11×, 128/16/1) | 31.3 → 28.3 (1.10×, 128/16/1) | 56.7 → 49.2 (1.15×, 64/8/2) |
| (16,48) Q=16 | 26.9 → 25.0 (1.08×, 64/16/1) | 34.6 → 31.4 (1.10×, 128/16/1) | 36.4 → 33.2 (1.10×, 128/16/1) | 67.3 → 58.1 (1.16×, 64/8/2) |

B=3–4 range 1.004–1.158×; all rows 0.998–1.158×; rows < 0.98: []

**GB10** (legacy → shipped policy, µs per verify+commit round, min of 2 ABBA passes × 50 rounds; speedup; shipped tile/warps/min CTAs per SM)

| shape, Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| (16,32) Q=4 | 16.7 → 16.7 (1.00×, 128/16/1) | 31.3 → 31.3 (1.00×, 128/16/1) | 38.1 → 41.3 (0.92×, 128/16/1) | 62.2 → 61.1 (1.02×, 128/16/1) |
| (16,32) Q=5 | 17.5 → 17.3 (1.01×, 128/16/1) | 32.9 → 32.6 (1.01×, 128/16/1) | 42.6 → 41.9 (1.02×, 128/16/1) | 65.0 → 63.8 (1.02×, 128/16/1) |
| (16,32) Q=6 | 19.0 → 18.5 (1.03×, 128/16/1) | 35.9 → 35.0 (1.03×, 128/16/1) | 46.0 → 44.5 (1.04×, 128/16/1) | 69.5 → 67.0 (1.04×, 128/16/1) |
| (16,32) Q=8 | 21.4 → 20.8 (1.03×, 128/16/1) | 40.9 → 39.5 (1.04×, 128/16/1) | 52.9 → 50.5 (1.05×, 128/16/1) | 79.9 → 76.9 (1.04×, 128/16/1) |
| (16,32) Q=9 | 24.2 → 21.4 (1.13×, 128/16/1) | 47.0 → 41.1 (1.14×, 128/16/1) | 58.4 → 52.2 (1.12×, 128/16/1) | 86.5 → 77.5 (1.12×, 128/16/1) |
| (16,32) Q=12 | 27.9 → 24.9 (1.12×, 128/16/1) | 53.7 → 48.0 (1.12×, 128/16/1) | 65.1 → 58.1 (1.12×, 128/16/1) | 98.1 → 88.3 (1.11×, 128/16/1) |
| (16,32) Q=16 | 32.5 → 29.4 (1.11×, 128/16/1) | 63.4 → 56.7 (1.12×, 128/16/1) | 77.1 → 69.8 (1.10×, 128/16/1) | 116.2 → 106.7 (1.09×, 128/16/1) |
| (16,48) Q=4 | 19.3 → 19.1 (1.01×, 128/16/1) | 41.9 → 41.3 (1.01×, 128/16/1) | 68.7 → 66.9 (1.03×, 128/16/1) | 99.5 → 98.4 (1.01×, 128/16/1) |
| (16,48) Q=5 | 19.9 → 19.8 (1.00×, 128/16/1) | 43.1 → 42.2 (1.02×, 128/16/1) | 70.0 → 69.2 (1.01×, 128/16/1) | 105.6 → 104.4 (1.01×, 128/16/1) |
| (16,48) Q=6 | 21.4 → 21.0 (1.02×, 128/16/1) | 46.8 → 45.3 (1.03×, 128/16/1) | 74.7 → 72.3 (1.03×, 128/16/1) | 113.1 → 110.0 (1.03×, 128/16/1) |
| (16,48) Q=8 | 23.9 → 23.3 (1.02×, 128/16/1) | 53.3 → 51.3 (1.04×, 128/16/1) | 82.6 → 79.9 (1.03×, 128/16/1) | 125.2 → 120.8 (1.04×, 128/16/1) |
| (16,48) Q=9 | 26.6 → 24.0 (1.10×, 128/16/1) | 60.0 → 53.3 (1.12×, 128/16/1) | 91.9 → 83.1 (1.11×, 128/16/1) | 138.0 → 125.0 (1.10×, 128/16/1) |
| (16,48) Q=12 | 30.2 → 27.5 (1.10×, 128/16/1) | 65.5 → 58.8 (1.11×, 128/16/1) | 107.0 → 97.1 (1.10×, 128/16/1) | 148.5 → 135.9 (1.09×, 128/16/1) |
| (16,48) Q=16 | 35.1 → 32.0 (1.10×, 128/16/1) | 72.7 → 64.8 (1.12×, 128/16/1) | 123.6 → 112.7 (1.10×, 128/16/1) | 174.4 → 161.3 (1.08×, 128/16/1) |

B=3–4 range 0.924–1.120×; all rows 0.924–1.141×; rows < 0.98: [(32, 4, 3, 0.924)]

GB10 (16,32) Q=4 B=3 is noise: the passes are legacy 38.1/42.2 µs and shipped
49.6/41.3 µs. A separate 100-round ABBA GB10 run (8 vs 16 warps at tile 128)
measured that cell at 36.1–36.8 µs with 8 warps and 34.9 µs with 16, i.e. 1.044×
in favour of 16, with 0.986–1.052× over Q=4/5/6/8 × B=1–4.

### Comparator (vLLM 0.31.0, SGLang 72730d6), trusted + fused, new policy

`gdn_verify_compare.py` as in the 0.1.12 section (shape (16,32), CUDA graph per
provider, warm L2, 20 interleaved pairs, ε = p95 |ratio − 1| over 20 self-pairs);
each device's two calls (B=1–2, B=3–4) ran on one RTX 5090, one RTX PRO 6000
and one GB10 (X925-pinned). Cells: µs per layer-round (ours / vLLM /
SGLang): speedup over the faster comparator (bold when ≥ 1 + ε), then the chosen
tile/warps/min CTAs per SM. Earlier tables ran on other nodes, so they are not an
A/B for this change; use the same-call table above.

#### RTX 5090 (28 rows, correctness_ok all=True, gpu NVIDIA GeForce RTX 5090)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.9 / 22.4 / 19.9: **1.445** (ε 0.115) 32/8/1 | 15.9 / 24.6 / 23.2: **1.490** (ε 0.099) 64/8/1 | 16.9 / 30.4 / 27.7: **1.641** (ε 0.123) 128/16/1 | 18.9 / 34.8 / 33.4: **1.717** (ε 0.096) 128/16/1 |
| 5 | 14.1 / 24.2 / 22.2: **1.578** (ε 0.313) 32/8/1 | 16.1 / 28.5 / 26.8: **1.698** (ε 0.065) 64/8/1 | 18.4 / 32.8 / 32.5: **1.696** (ε 0.196) 128/16/1 | 20.3 / 36.9 / 38.6: **1.823** (ε 0.355) 128/16/1 |
| 6 | 14.4 / 26.1 / 24.2: **1.694** (ε 0.121) 32/8/1 | 16.4 / 32.1 / 30.0: **1.757** (ε 0.499) 64/8/1 | 18.5 / 36.9 / 35.5: **1.815** (ε 0.092) 128/16/1 | 21.0 / 41.0 / 44.0: **1.951** (ε 0.263) 128/16/1 |
| 8 | 16.0 / 28.5 / 30.5: **1.768** (ε 0.094) 32/8/1 | 18.6 / 35.6 / 36.1: **1.837** (ε 0.035) 64/8/1 | 21.1 / 43.0 / 44.5: **1.945** (ε 0.101) 128/16/1 | 23.0 / 47.1 / 58.4: **1.997** (ε 0.110) 128/16/1 |
| 9 | 16.3 / 30.7 / 32.5: **1.850** (ε 0.376) 32/16/1 | 18.4 / 38.6 / 40.5: **1.947** (ε 0.836) 64/16/1 | 22.2 / 44.8 / 48.5: **1.965** (ε 0.098) 128/16/1 | 24.1 / 55.3 / 66.6: **2.263** (ε 0.146) 128/16/1 |
| 12 | 18.2 / 36.6 / 42.2: **1.970** (ε 0.066) 32/16/1 | 20.4 / 43.0 / 48.9: **2.096** (ε 0.077) 64/16/1 | 25.9 / 57.3 / 65.5: **2.179** (ε 0.138) 128/16/1 | 26.7 / 79.9 / 93.2: **2.916** (ε 0.226) 128/16/1 |
| 16 | 20.5 / 42.7 / 53.1: **1.919** (ε 0.274) 32/16/1 | 23.6 / 53.2 / 65.2: **2.212** (ε 0.756) 64/16/1 | 29.6 / 86.0 / 90.1: **2.852** (ε 0.157) 128/16/1 | 30.7 / 108.5 / 124.4: **3.538** (ε 0.110) 128/16/1 |

range 1.445-3.538; rows not above 1+ε: []

#### RTX PRO 6000 (28 rows, correctness_ok all=True, gpu NVIDIA RTX PRO 6000 Blackwell Server Edition)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 15.5 / 27.8 / 23.2: **1.513** (ε 0.095) 32/8/1 | 17.6 / 30.7 / 27.7: **1.555** (ε 0.144) 64/8/1 | 19.3 / 34.8 / 32.7: **1.654** (ε 0.056) 128/16/1 | 21.3 / 41.0 / 37.3: **1.703** (ε 0.091) 128/16/1 |
| 5 | 17.1 / 29.8 / 26.9: **1.547** (ε 0.103) 32/8/1 | 19.6 / 32.8 / 31.9: **1.662** (ε 0.136) 64/8/1 | 19.6 / 43.0 / 39.5: **1.864** (ε 0.102) 128/16/1 | 22.5 / 44.0 / 43.0: **1.756** (ε 0.083) 128/16/1 |
| 6 | 18.2 / 32.8 / 30.5: **1.652** (ε 0.042) 32/8/1 | 20.4 / 36.9 / 37.0: **1.748** (ε 0.111) 64/8/1 | 21.5 / 45.1 / 43.0: **1.946** (ε 0.078) 128/16/1 | 23.6 / 49.2 / 49.3: **1.847** (ε 0.123) 128/16/1 |
| 8 | 19.6 / 34.8 / 36.7: **1.829** (ε 0.096) 32/8/1 | 20.5 / 42.0 / 43.6: **1.988** (ε 0.080) 64/8/1 | 23.8 / 51.2 / 50.4: **1.965** (ε 0.038) 128/16/1 | 26.6 / 55.3 / 59.4: **1.930** (ε 0.118) 128/16/1 |
| 9 | 19.6 / 36.9 / 40.3: **1.921** (ε 0.103) 32/16/1 | 21.6 / 47.1 / 49.1: **2.073** (ε 0.081) 64/16/1 | 25.2 / 53.3 / 55.3: **1.931** (ε 0.111) 128/16/1 | 27.6 / 59.4 / 67.6: **2.035** (ε 0.133) 128/16/1 |
| 12 | 22.2 / 40.0 / 49.2: **1.772** (ε 0.116) 32/16/1 | 24.7 / 53.3 / 59.5: **1.986** (ε 0.074) 64/16/1 | 28.1 / 59.4 / 69.2: **1.904** (ε 0.054) 128/16/1 | 34.8 / 79.9 / 92.1: **2.384** (ε 0.191) 128/16/1 |
| 16 | 25.7 / 51.2 / 65.3: **1.972** (ε 0.106) 32/16/1 | 28.0 / 63.2 / 75.8: **2.102** (ε 0.073) 64/16/1 | 36.8 / 84.0 / 90.1: **2.273** (ε 0.158) 128/16/1 | 38.9 / 116.8 / 129.0: **3.003** (ε 0.177) 128/16/1 |

range 1.513-3.003; rows not above 1+ε: []

#### GB10 (28 rows, correctness_ok all=True, gpu NVIDIA GB10)

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 24.5 / 48.3 / 63.2: **2.004** (ε 0.229) 128/16/1 | 43.0 / 94.9 / 131.1: **2.216** (ε 0.150) 128/16/1 | 58.0 / 154.4 / 202.5: **2.651** (ε 0.372) 128/16/1 | 90.8 / 211.6 / 288.8: **2.322** (ε 0.047) 128/16/1 |
| 5 | 24.4 / 57.3 / 75.2: **2.359** (ε 0.192) 128/16/1 | 44.6 / 124.9 / 155.6: **2.815** (ε 0.144) 128/16/1 | 59.9 / 196.6 / 250.6: **3.250** (ε 0.051) 128/16/1 | 93.9 / 258.1 / 340.0: **2.755** (ε 0.080) 128/16/1 |
| 6 | 26.5 / 69.8 / 87.0: **2.654** (ε 0.182) 128/16/1 | 46.9 / 146.5 / 181.2: **3.162** (ε 0.125) 128/16/1 | 64.5 / 230.4 / 288.8: **3.562** (ε 0.043) 128/16/1 | 97.3 / 306.2 / 389.1: **3.148** (ε 0.023) 128/16/1 |
| 8 | 28.5 / 94.4 / 109.6: **3.195** (ε 0.175) 128/16/1 | 50.5 / 199.3 / 235.5: **4.055** (ε 0.089) 128/16/1 | 69.6 / 298.0 / 366.8: **4.281** (ε 0.030) 128/16/1 | 104.1 / 394.0 / 488.6: **3.782** (ε 0.251) 128/16/1 |
| 9 | 28.8 / 106.7 / 122.8: **3.515** (ε 0.209) 128/16/1 | 51.2 / 220.2 / 266.2: **4.400** (ε 0.107) 128/16/1 | 72.7 / 332.5 / 409.6: **4.606** (ε 0.062) 128/16/1 | 110.6 / 439.0 / 539.3: **3.988** (ε 0.049) 128/16/1 |
| 12 | 32.6 / 139.3 / 157.4: **4.109** (ε 0.146) 128/16/1 | 57.3 / 292.9 / 345.1: **5.066** (ε 0.107) 128/16/1 | 79.6 / 431.8 / 519.6: **5.394** (ε 0.290) 128/16/1 | 117.7 / 576.3 / 694.0: **4.884** (ε 0.537) 128/16/1 |
| 16 | 38.4 / 189.4 / 210.2: **4.843** (ε 0.194) 128/16/1 | 65.7 / 385.0 / 450.6: **5.972** (ε 0.271) 128/16/1 | 90.0 / 576.1 / 680.4: **6.404** (ε 0.798) 128/16/1 | 135.2 / 755.4 / 899.0: **5.600** (ε 0.215) 128/16/1 |

range 2.004-6.404; rows not above 1+ε: []

DRAM floor (cold, state only: verify reads S, commit reads and writes S, S = B·Hv·64 KiB;
sol-ceilings 4522f5a read/copy GB/s): RTX 5090 4.0/8.1/12.1/16.1 µs at B=1/2/3/4,
PRO 6000 4.3/8.5/12.8/17.0, GB10 27.7/55.5/83.2/111.0. The comparator runs warm-L2
(the B ≤ 4 state fits in L2), so GB10 rows can sit below this floor. At Q=4, B=4
the discrete GPUs are within 1.2–1.3× of it; what remains at B=3–4 is the state
traffic of separate verify and commit (R3c) and the per-token replay chain at large Q.

## K2 R3c delayed commit and R3d fused handoff, 2026-10-15

R3c (`d5b34dc`) adds lazy append of accepted rows, folding at ring capacity or
snapshot capture, capturable trusted flush, and commit-mode selection. The API
was updated after that commit: `commit_mode=None` (default) and `"eager"` retain
eager commit; `"auto"` selects the measured `cute._config.COMMIT_POLICY` per batch
bucket, and `"delayed"` forces delayed. Existing engine bindings remain eager.
Opt in only after every direct checkpoint consumer calls `flush(slots)` first.

R3d (`6767272`) adds a bound layer pointer table and fused handoff with one control
and one payload launch, bitwise equal to the per-layer path including status.
`279d16c` fixes the out-of-place test's destination uniqueness; `1996ca5` checks
exact successful prefixes for later-layer failures and detects swapped table rows.
`309bfd1` stubs the product launch plan in the compile-cache test for CPU CI
without Triton.

RTX 5090 measurements use retail cards, one call per table. RTX PRO 6000
is the Blackwell Server Edition. GB10 measurements are pinned to its X925 cores.

### R3c same-process eager/delayed A/B

ABBA A/B uses 20 pairs over K-round graphs for both modes, where K is the fold
period at accepted `Q//2`; timings are µs per round. Self-pair calibration is
`ε = p95 |self-pair ratio − 1|`. Cells retain eager/delayed µs, the paired median
eager/delayed ratio, ε and K; ratios need not equal the quotient of rounded times.
Bold ratios clear 1+ε in that individual measurement, not necessarily the policy.

For each device's (batch bucket, Q, window) key, delayed qualifies only if the
ratio is ≥1 in every measured shape and >1+ε in at least one, with K>1. Shapes
are (16,32) and (16,48), except GB10 uses (16,32) only. The populated `_DELAYED`
table in `cute/_config.py` implements this rule. Unmeasured keys (B≥5, window 16,
Q outside 4/5/6/8/9/12/16) stay eager. Qualifying rows cluster at Q=4–6 for window
4 and Q≤9 for window 8; K=1 folds every commit and does the same work as eager.

#### RTX 5090, (16,32), window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 12.4/11.2: **1.103** (ε0.083, K2) | 14.7/13.2: 1.115 (ε1.867, K2) | 16.2/14.3: 1.132 (ε0.189, K2) | 17.4/15.3: **1.142** (ε0.100, K2) |
| 5 | 13.1/12.0: 1.099 (ε0.435, K2) | 15.1/13.2: **1.130** (ε0.062, K2) | 16.6/15.2: 1.091 (ε0.682, K2) | 18.0/15.3: **1.155** (ε0.060, K2) |
| 6 | 13.5/13.4: 1.010 (ε0.063, K2) | 15.9/15.4: 1.032 (ε0.066, K2) | 17.6/16.4: **1.088** (ε0.069, K2) | 19.2/17.4: **1.095** (ε0.062, K2) |
| 8 | 16.2/16.1: 0.994 (ε0.085, K1) | 18.4/18.4: 1.000 (ε0.074, K1) | 20.4/20.4: 1.003 (ε0.044, K1) | 22.4/22.3: 1.003 (ε0.065, K1) |
| 9 | 16.2/16.2: 1.004 (ε0.062, K1) | 18.4/18.4: 1.002 (ε0.163, K1) | 20.6/22.0: 0.934 (ε0.097, K1) | 23.4/22.7: 1.002 (ε0.078, K1) |
| 12 | 18.3/18.1: 1.017 (ε0.048, K1) | 20.3/20.3: 1.007 (ε0.313, K1) | 24.2/24.4: 0.995 (ε0.061, K1) | 25.2/26.2: 0.976 (ε0.231, K1) |
| 16 | 20.0/20.4: 0.994 (ε0.075, K1) | 22.5/22.6: 0.994 (ε0.071, K1) | 27.6/27.7: 0.987 (ε0.070, K1) | 28.5/29.9: 0.969 (ε0.084, K1) |

#### RTX 5090, (16,32), window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 11.7/11.1: **1.046** (ε0.045, K4) | 13.9/12.3: 1.125 (ε2.046, K4) | 15.4/13.3: **1.155** (ε0.033, K4) | 16.4/14.0: **1.181** (ε0.050, K4) |
| 5 | 12.2/11.8: 1.024 (ε0.047, K4) | 14.3/13.3: **1.076** (ε0.060, K4) | 16.0/14.0: **1.156** (ε0.033, K4) | 17.3/14.6: **1.181** (ε0.057, K4) |
| 6 | 13.1/13.7: 0.951 (ε0.037, K3) | 15.1/15.1: 1.000 (ε0.648, K3) | 17.2/15.8: **1.087** (ε0.070, K3) | 18.4/16.5: **1.118** (ε0.038, K3) |
| 8 | 14.4/14.6: 0.992 (ε0.218, K2) | 17.3/17.2: 1.000 (ε0.050, K2) | 19.6/18.5: **1.063** (ε0.036, K2) | 20.7/19.5: **1.086** (ε0.058, K2) |
| 9 | 15.1/14.3: 1.047 (ε0.071, K2) | 16.4/16.3: 1.011 (ε0.042, K2) | 20.5/18.7: 1.065 (ε0.093, K2) | 21.6/20.0: **1.078** (ε0.053, K2) |
| 12 | 17.3/17.2: 1.008 (ε0.098, K2) | 18.6/18.9: 0.992 (ε0.043, K2) | 23.5/23.2: 1.004 (ε0.174, K2) | 24.4/23.5: 1.012 (ε0.049, K2) |
| 16 | 20.8/22.0: 0.966 (ε1.616, K1) | 23.1/23.0: 0.991 (ε0.656, K1) | 28.6/28.6: 0.991 (ε0.063, K1) | 29.8/30.4: 0.972 (ε0.087, K1) |

#### RTX 5090, (16,48), window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.7/12.6: **1.095** (ε0.036, K2) | 16.4/14.2: **1.146** (ε0.064, K2) | 17.7/15.4: **1.147** (ε0.060, K2) | 26.0/22.5: **1.137** (ε0.045, K2) |
| 5 | 14.3/13.2: **1.094** (ε0.074, K2) | 16.9/14.8: **1.143** (ε0.065, K2) | 18.4/15.4: **1.182** (ε0.107, K2) | 27.6/24.6: **1.125** (ε0.048, K2) |
| 6 | 15.4/15.1: 1.011 (ε0.049, K2) | 17.4/16.4: **1.095** (ε0.083, K2) | 19.5/17.4: **1.118** (ε0.057, K2) | 29.7/27.6: **1.073** (ε0.036, K2) |
| 8 | 17.4/16.4: 1.005 (ε0.084, K1) | 20.5/20.5: 1.000 (ε0.053, K1) | 22.5/22.5: 1.000 (ε0.084, K1) | 34.8/34.8: 1.000 (ε0.063, K1) |
| 9 | 17.4/17.3: 1.002 (ε0.088, K1) | 20.5/20.5: 1.000 (ε0.091, K1) | 24.6/24.6: 1.000 (ε0.091, K1) | 38.9/38.9: 1.000 (ε0.056, K1) |
| 12 | 18.6/18.6: 1.000 (ε0.086, K1) | 24.6/24.6: 1.000 (ε0.044, K1) | 25.6/26.6: 1.000 (ε0.082, K1) | 43.0/43.0: 1.000 (ε0.053, K1) |
| 16 | 22.4/22.5: 1.000 (ε0.049, K1) | 27.6/27.8: 0.978 (ε0.039, K1) | 30.7/30.7: 1.000 (ε0.071, K1) | 49.2/49.2: 1.000 (ε0.001, K1) |

#### RTX 5090, (16,48), window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.1/12.1: **1.076** (ε0.032, K4) | 15.6/13.3: **1.178** (ε0.019, K4) | 16.9/14.1: **1.220** (ε0.044, K4) | 25.3/22.5: **1.138** (ε0.020, K4) |
| 5 | 13.8/13.2: **1.047** (ε0.019, K4) | 16.4/13.8: **1.165** (ε0.021, K4) | 17.9/14.8: **1.207** (ε0.056, K4) | 26.6/24.1: **1.106** (ε0.037, K4) |
| 6 | 14.7/15.3: 0.976 (ε0.024, K3) | 17.2/15.7: **1.109** (ε0.040, K3) | 19.1/16.6: **1.134** (ε0.037, K3) | 28.7/28.0: 1.024 (ε0.025, K3) |
| 8 | 15.8/15.4: 1.002 (ε0.038, K2) | 19.5/17.9: **1.084** (ε0.056, K2) | 21.5/19.5: **1.106** (ε0.091, K2) | 33.2/30.7: **1.067** (ε0.040, K2) |
| 9 | 15.9/14.9: **1.068** (ε0.044, K2) | 20.0/18.9: **1.066** (ε0.049, K2) | 21.5/20.5: **1.100** (ε0.048, K2) | 36.9/33.8: **1.090** (ε0.029, K2) |
| 12 | 18.3/18.3: 1.002 (ε0.045, K2) | 22.5/22.5: 1.000 (ε0.045, K2) | 24.6/23.6: 1.040 (ε0.044, K2) | 41.0/42.0: 0.976 (ε0.024, K2) |
| 16 | 22.5/21.5: 1.001 (ε0.074, K1) | 27.6/27.6: 1.000 (ε0.040, K1) | 29.5/30.4: 1.000 (ε0.073, K1) | 49.2/49.2: 1.000 (ε0.001, K1) |

#### RTX PRO 6000, (16,32), window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 15.3/14.1: **1.085** (ε0.076, K2) | 17.6/15.6: **1.128** (ε0.077, K2) | 18.4/16.5: **1.119** (ε0.063, K2) | 20.5/17.7: **1.129** (ε0.087, K2) |
| 5 | 14.9/14.0: **1.064** (ε0.025, K2) | 18.1/16.8: **1.091** (ε0.058, K2) | 19.0/17.3: **1.096** (ε0.067, K2) | 21.0/18.7: **1.121** (ε0.104, K2) |
| 6 | 16.0/16.0: 0.990 (ε0.078, K2) | 18.8/18.2: 1.035 (ε0.067, K2) | 20.0/19.0: **1.059** (ε0.033, K2) | 22.0/20.4: **1.085** (ε0.075, K2) |
| 8 | 19.8/19.2: 1.008 (ε0.108, K1) | 20.7/20.8: 1.005 (ε0.082, K1) | 23.9/23.7: 1.002 (ε0.049, K1) | 25.6/25.7: 0.994 (ε0.083, K1) |
| 9 | 19.7/19.7: 0.998 (ε0.055, K1) | 21.5/21.3: 0.998 (ε0.067, K1) | 24.8/24.6: 1.011 (ε0.101, K1) | 26.6/26.8: 0.979 (ε0.091, K1) |
| 12 | 21.7/21.7: 1.002 (ε0.044, K1) | 23.7/23.9: 0.998 (ε0.081, K1) | 27.9/28.6: 0.984 (ε0.054, K1) | 29.7/30.0: 0.987 (ε0.073, K1) |
| 16 | 25.6/25.3: 1.001 (ε0.057, K1) | 27.4/27.4: 0.994 (ε0.071, K1) | 32.3/33.0: 0.993 (ε0.066, K1) | 34.0/35.1: 0.973 (ε0.066, K1) |

#### RTX PRO 6000, (16,32), window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.9/13.4: **1.040** (ε0.022, K4) | 16.0/15.2: **1.062** (ε0.045, K4) | 17.2/15.7: **1.107** (ε0.026, K4) | 19.6/16.9: **1.150** (ε0.034, K4) |
| 5 | 14.5/14.4: 1.018 (ε0.024, K4) | 16.8/16.3: **1.047** (ε0.043, K4) | 18.2/16.3: **1.101** (ε0.033, K4) | 20.1/17.9: **1.143** (ε0.035, K4) |
| 6 | 16.1/16.9: 0.952 (ε0.046, K3) | 18.5/18.8: 0.974 (ε0.028, K3) | 19.8/18.7: **1.058** (ε0.041, K3) | 21.6/19.6: **1.095** (ε0.038, K3) |
| 8 | 17.4/17.5: 0.989 (ε0.044, K2) | 19.2/19.6: 0.984 (ε0.170, K2) | 22.6/21.6: **1.046** (ε0.044, K2) | 24.5/23.4: **1.068** (ε0.049, K2) |
| 9 | 18.3/17.4: **1.048** (ε0.047, K2) | 19.7/19.3: 1.046 (ε0.053, K2) | 23.5/22.5: **1.043** (ε0.030, K2) | 25.7/24.2: 1.065 (ε0.080, K2) |
| 12 | 20.6/21.3: 0.979 (ε0.050, K2) | 22.6/22.6: 0.998 (ε0.051, K2) | 26.5/27.5: 0.966 (ε0.019, K2) | 29.3/28.8: 1.018 (ε0.043, K2) |
| 16 | 26.4/26.3: 1.008 (ε0.088, K1) | 27.9/27.9: 1.004 (ε0.095, K1) | 33.1/33.1: 0.996 (ε0.040, K1) | 35.4/35.5: 1.002 (ε0.080, K1) |

#### RTX PRO 6000, (16,48), window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 17.7/16.0: **1.111** (ε0.048, K2) | 19.0/17.1: **1.116** (ε0.037, K2) | 21.3/19.1: **1.106** (ε0.055, K2) | 30.4/27.8: **1.102** (ε0.039, K2) |
| 5 | 17.8/16.6: **1.077** (ε0.075, K2) | 19.9/17.9: **1.109** (ε0.045, K2) | 22.1/20.0: **1.109** (ε0.079, K2) | 32.2/29.1: **1.107** (ε0.033, K2) |
| 6 | 19.8/19.1: 1.031 (ε0.078, K2) | 20.8/19.7: **1.056** (ε0.032, K2) | 24.0/21.4: **1.105** (ε0.057, K2) | 35.0/33.3: 1.041 (ε0.056, K2) |
| 8 | 21.3/21.4: 0.989 (ε0.137, K1) | 25.5/25.4: 0.997 (ε0.048, K1) | 27.9/28.0: 0.992 (ε0.096, K1) | 41.3/42.0: 0.984 (ε0.067, K1) |
| 9 | 21.6/21.8: 1.005 (ε0.054, K1) | 25.4/26.8: 0.964 (ε0.067, K1) | 28.1/29.0: 0.995 (ε0.112, K1) | 46.1/46.1: 0.998 (ε0.019, K1) |
| 12 | 24.4/25.1: 0.980 (ε0.074, K1) | 29.4/29.7: 0.995 (ε0.036, K1) | 32.2/33.3: 0.991 (ε0.081, K1) | 52.4/52.5: 1.000 (ε0.039, K1) |
| 16 | 27.4/27.6: 0.983 (ε0.090, K1) | 33.7/33.9: 0.993 (ε0.090, K1) | 36.6/36.8: 0.998 (ε0.064, K1) | 61.2/61.7: 0.993 (ε0.039, K1) |

#### RTX PRO 6000, (16,48), window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 16.7/15.6: **1.072** (ε0.021, K4) | 18.1/16.5: **1.099** (ε0.033, K4) | 20.5/17.6: **1.177** (ε0.037, K4) | 29.5/27.4: **1.082** (ε0.012, K4) |
| 5 | 17.2/16.7: **1.039** (ε0.035, K4) | 19.0/17.3: **1.094** (ε0.032, K4) | 21.2/18.5: **1.154** (ε0.034, K4) | 31.0/29.5: **1.051** (ε0.016, K4) |
| 6 | 18.8/19.5: 0.969 (ε0.025, K3) | 20.6/19.5: **1.050** (ε0.033, K3) | 22.8/20.8: **1.096** (ε0.050, K3) | 34.0/34.4: 0.996 (ε0.017, K3) |
| 8 | 19.9/19.9: 1.001 (ε0.033, K2) | 23.2/22.9: 1.029 (ε0.049, K2) | 26.3/24.9: 1.066 (ε0.085, K2) | 41.9/38.4: **1.093** (ε0.061, K2) |
| 9 | 20.0/19.5: 1.034 (ε0.035, K2) | 24.1/23.2: **1.040** (ε0.037, K2) | 26.4/25.1: 1.054 (ε0.076, K2) | 44.3/41.6: **1.063** (ε0.033, K2) |
| 12 | 23.0/23.1: 0.999 (ε0.032, K2) | 28.0/28.3: 0.984 (ε0.040, K2) | 30.3/30.3: 1.011 (ε0.039, K2) | 49.9/52.0: 0.964 (ε0.022, K2) |
| 16 | 27.6/27.7: 1.005 (ε0.119, K1) | 33.8/33.9: 0.999 (ε0.029, K1) | 37.2/37.8: 0.982 (ε0.064, K1) | 61.0/61.4: 0.994 (ε0.029, K1) |

#### GB10, (16,32), window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 18.9/17.1: **1.109** (ε0.032, K2) | 35.7/29.6: **1.264** (ε0.038, K2) | 47.1/40.3: **1.179** (ε0.022, K2) | 76.1/60.0: **1.265** (ε0.039, K2) |
| 5 | 20.0/16.9: **1.185** (ε0.072, K2) | 37.0/33.0: **1.113** (ε0.029, K2) | 48.4/42.0: **1.152** (ε0.019, K2) | 74.3/62.0: **1.205** (ε0.014, K2) |
| 6 | 21.1/19.0: **1.114** (ε0.038, K2) | 39.7/37.6: **1.077** (ε0.028, K2) | 50.4/47.7: **1.057** (ε0.011, K2) | 78.7/71.7: **1.100** (ε0.014, K2) |
| 8 | 25.5/25.2: 1.007 (ε0.093, K1) | 51.2/50.6: 1.036 (ε0.048, K1) | 69.7/67.6: **1.046** (ε0.028, K1) | 103.5/108.6: 0.953 (ε0.017, K1) |
| 9 | 26.2/25.6: **1.045** (ε0.044, K1) | 51.3/53.1: 0.997 (ε0.041, K1) | 71.8/73.5: 0.984 (ε0.641, K1) | 105.1/115.7: 0.908 (ε0.021, K1) |
| 12 | 29.7/28.4: **1.066** (ε0.064, K1) | 59.3/60.8: 0.980 (ε0.038, K1) | 79.7/78.4: 1.018 (ε0.028, K1) | 121.3/120.8: 1.010 (ε0.022, K1) |
| 16 | 34.2/33.8: 1.026 (ε0.059, K1) | 69.6/68.6: 1.005 (ε0.033, K1) | 88.1/88.0: 1.006 (ε0.024, K1) | 134.1/140.3: 0.956 (ε0.016, K1) |

#### GB10, (16,32), window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 18.2/15.6: **1.167** (ε0.012, K4) | 32.5/28.2: **1.170** (ε0.016, K4) | 39.9/34.3: **1.173** (ε0.018, K4) | 68.6/51.4: **1.332** (ε0.016, K4) |
| 5 | 18.7/16.2: **1.156** (ε0.050, K4) | 33.7/30.9: **1.107** (ε0.018, K4) | 42.4/36.1: **1.171** (ε0.164, K4) | 67.1/53.5: **1.256** (ε0.016, K4) |
| 6 | 20.1/18.3: **1.098** (ε0.009, K3) | 36.9/35.3: **1.053** (ε0.023, K3) | 47.2/42.9: **1.099** (ε0.015, K3) | 75.0/63.4: **1.184** (ε0.014, K3) |
| 8 | 23.0/21.5: **1.064** (ε0.041, K2) | 44.1/43.4: **1.027** (ε0.025, K2) | 56.3/51.6: **1.093** (ε0.030, K2) | 87.8/79.2: **1.110** (ε0.012, K2) |
| 9 | 23.2/22.0: **1.052** (ε0.038, K2) | 46.9/44.4: **1.062** (ε0.026, K2) | 57.7/53.9: **1.070** (ε0.019, K2) | 91.0/83.2: **1.096** (ε0.018, K2) |
| 12 | 27.2/26.6: 1.020 (ε0.031, K2) | 52.3/54.7: 0.970 (ε0.021, K2) | 63.8/62.8: 1.017 (ε0.024, K2) | 102.8/96.2: 1.070 (ε0.154, K2) |
| 16 | 33.6/32.8: 1.008 (ε0.064, K1) | 68.7/73.0: 0.957 (ε0.026, K1) | 88.3/85.7: **1.034** (ε0.024, K1) | 137.0/142.0: 0.965 (ε0.020, K1) |

### Comparator: delayed versus vLLM and SGLang

External versions are vLLM 0.31.0 and SGLang 72730d6. Ours uses forced delayed
commit with K-round graphs spanning a complete fold period, divided by K for
per-round timings; external graphs retain one round. The faster external is the
bar. Cells are ours / vLLM / SGLang µs per round, then paired speedup against that
bar and self-pair ε. Ratios are retained as measured, not recomputed from the
rounded times. Bold is retained from the input tables; qualification requires
speedup >1+ε.

Two RTX 5090 window-4 rows, (Q6,B1) and (Q16,B4), are 1.84× / 3.47× faster but
have self-pair ε 2.4 / 3.4 (2.410 / 3.423 in the table) from a noisy cluster node.
Re-measured on another RTX 5090 (local, same protocol, window 4, ours delayed),
both rows clear 1+ε: (Q6,B1) ours 13.2 µs vs vLLM 26.6 / SGLang 24.2, 1.866×
(ε 0.059); (Q16,B4) 30.7 vs 112.6 / 122.9, 3.700× (ε 0.155); correctness passes.
All other supplied comparator rows clear 1+ε. The supplied GB10 comparator
covers window 8 only.

#### RTX 5090, window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 11.2 / 22.3 / 20.2: **1.793** (ε 0.085) | 12.8 / 24.6 / 23.7: **1.848** (ε 0.383) | 14.3 / 30.2 / 28.5: **1.998** (ε 0.045) | 15.3 / 34.7 / 33.3: **2.126** (ε 0.074) |
| 5 | 12.0 / 24.5 / 22.0: **1.832** (ε 0.087) | 13.2 / 28.2 / 26.4: **2.016** (ε 0.035) | 15.1 / 34.3 / 32.8: **2.120** (ε 0.085) | 16.2 / 36.9 / 38.9: **2.264** (ε 0.076) |
| 6 | 13.2 / 26.6 / 24.4: **1.836** (ε 2.410) | 15.2 / 30.7 / 29.4: **1.926** (ε 0.045) | 16.3 / 36.8 / 35.7: **2.138** (ε 0.060) | 17.4 / 41.0 / 45.1: **2.359** (ε 0.111) |
| 8 | 16.2 / 28.5 / 30.3: **1.761** (ε 0.092) | 18.5 / 34.8 / 36.3: **1.854** (ε 0.105) | 21.8 / 42.7 / 44.8: **1.919** (ε 0.160) | 24.5 / 49.2 / 59.4: **2.042** (ε 0.166) |
| 9 | 16.1 / 28.7 / 32.5: **1.821** (ε 0.136) | 18.4 / 36.9 / 38.7: **1.922** (ε 0.068) | 23.1 / 44.9 / 48.4: **1.859** (ε 0.197) | 24.5 / 55.3 / 68.2: **2.253** (ε 0.119) |
| 12 | 18.2 / 36.5 / 41.3: **1.926** (ε 0.146) | 20.3 / 42.7 / 48.7: **1.953** (ε 0.047) | 25.5 / 59.4 / 66.3: **2.314** (ε 0.142) | 28.6 / 79.9 / 92.2: **2.786** (ε 0.158) |
| 16 | 20.9 / 42.8 / 52.8: **1.933** (ε 0.084) | 23.9 / 53.0 / 65.2: **2.182** (ε 0.102) | 29.7 / 83.8 / 91.1: **2.814** (ε 0.141) | 30.8 / 109.3 / 123.9: **3.466** (ε 3.423) |

#### RTX 5090, window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 12.0 / 26.7 / 23.2: **1.929** (ε 0.031) | 13.5 / 29.1 / 27.2: **2.025** (ε 0.033) | 14.4 / 31.3 / 32.3: **2.188** (ε 0.032) | 15.1 / 34.8 / 38.0: **2.268** (ε 0.040) |
| 5 | 12.9 / 28.5 / 26.4: **2.049** (ε 0.037) | 14.1 / 32.6 / 30.6: **2.167** (ε 0.024) | 15.1 / 34.8 / 36.6: **2.300** (ε 0.031) | 15.7 / 38.9 / 42.5: **2.406** (ε 0.042) |
| 6 | 15.1 / 30.3 / 29.0: **1.923** (ε 0.040) | 16.6 / 32.8 / 33.5: **1.985** (ε 0.027) | 17.4 / 38.9 / 40.2: **2.167** (ε 0.056) | 18.2 / 43.0 / 49.4: **2.363** (ε 0.048) |
| 8 | 16.8 / 32.8 / 34.0: **1.944** (ε 0.038) | 19.3 / 36.9 / 39.9: **1.879** (ε 0.049) | 20.9 / 43.0 / 47.7: **2.002** (ε 0.053) | 22.0 / 49.2 / 63.7: **2.238** (ε 0.089) |
| 9 | 16.4 / 34.4 / 36.4: **2.050** (ε 0.039) | 18.0 / 38.9 / 42.4: **2.086** (ε 0.050) | 21.7 / 45.1 / 52.0: **2.103** (ε 0.065) | 22.7 / 55.3 / 71.4: **2.398** (ε 0.084) |
| 12 | 19.6 / 39.2 / 45.6: **2.003** (ε 0.051) | 21.0 / 45.1 / 53.1: **2.058** (ε 0.042) | 25.8 / 59.4 / 68.7: **2.289** (ε 0.053) | 26.9 / 79.9 / 96.8: **2.932** (ε 0.063) |
| 16 | 25.6 / 45.1 / 56.8: **1.685** (ε 0.057) | 28.0 / 52.2 / 68.3: **1.806** (ε 0.078) | 33.8 / 86.0 / 95.7: **2.520** (ε 0.089) | 35.4 / 110.6 / 129.5: **3.106** (ε 0.129) |

#### RTX PRO 6000, window 4

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.9 / 28.5 / 23.8: **1.724** (ε 0.039) | 15.7 / 30.7 / 27.7: **1.756** (ε 0.071) | 16.5 / 35.2 / 33.5: **2.036** (ε 0.067) | 17.8 / 41.0 / 37.9: **2.108** (ε 0.046) |
| 5 | 13.9 / 29.7 / 26.6: **1.868** (ε 0.053) | 16.6 / 33.8 / 32.0: **1.949** (ε 0.068) | 17.1 / 42.0 / 38.7: **2.190** (ε 0.045) | 18.9 / 45.1 / 43.0: **2.196** (ε 0.068) |
| 6 | 16.0 / 31.7 / 29.6: **1.835** (ε 0.072) | 18.4 / 36.9 / 35.7: **1.913** (ε 0.083) | 19.0 / 45.1 / 45.0: **2.204** (ε 0.041) | 20.5 / 49.2 / 48.7: **2.239** (ε 0.049) |
| 8 | 19.6 / 35.8 / 35.8: **1.806** (ε 0.345) | 21.2 / 43.0 / 44.6: **1.952** (ε 0.119) | 25.2 / 51.2 / 51.5: **1.869** (ε 0.093) | 27.6 / 57.3 / 59.4: **1.960** (ε 0.109) |
| 9 | 19.6 / 36.9 / 39.8: **1.887** (ε 0.086) | 21.5 / 47.1 / 48.7: **2.014** (ε 0.070) | 25.5 / 53.3 / 55.6: **1.932** (ε 0.057) | 28.6 / 59.4 / 65.5: **1.985** (ε 0.114) |
| 12 | 21.6 / 41.0 / 49.8: **1.934** (ε 0.068) | 23.9 / 53.3 / 60.7: **2.139** (ε 0.060) | 29.8 / 63.5 / 70.7: **1.940** (ε 0.090) | 34.8 / 84.0 / 94.2: **2.415** (ε 0.149) |
| 16 | 25.9 / 51.2 / 65.1: **1.956** (ε 0.088) | 28.3 / 62.2 / 76.8: **2.078** (ε 0.067) | 36.6 / 82.0 / 92.1: **2.160** (ε 0.138) | 38.9 / 118.8 / 129.2: **3.002** (ε 0.140) |

#### RTX PRO 6000, window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 13.5 / 28.5 / 23.8: **1.765** (ε 0.050) | 15.2 / 30.7 / 27.9: **1.843** (ε 0.040) | 15.7 / 35.8 / 33.1: **2.132** (ε 0.029) | 16.7 / 41.0 / 37.6: **2.230** (ε 0.039) |
| 5 | 14.2 / 30.8 / 27.6: **1.939** (ε 0.029) | 16.2 / 32.8 / 32.5: **2.007** (ε 0.030) | 16.7 / 42.0 / 39.0: **2.337** (ε 0.035) | 17.8 / 45.1 / 42.7: **2.374** (ε 0.041) |
| 6 | 16.7 / 32.7 / 30.2: **1.842** (ε 0.044) | 18.9 / 34.8 / 35.1: **1.829** (ε 0.045) | 18.8 / 45.1 / 42.6: **2.284** (ε 0.035) | 19.8 / 49.2 / 48.3: **2.385** (ε 0.040) |
| 8 | 17.8 / 35.3 / 35.8: **1.978** (ε 0.068) | 19.4 / 43.0 / 43.7: **2.103** (ε 0.039) | 22.2 / 51.2 / 51.3: **2.198** (ε 0.051) | 23.5 / 56.3 / 59.4: **2.335** (ε 0.098) |
| 9 | 17.7 / 37.9 / 39.8: **2.130** (ε 0.052) | 19.1 / 47.1 / 48.1: **2.275** (ε 0.063) | 22.8 / 55.3 / 56.0: **2.301** (ε 0.066) | 24.6 / 59.4 / 66.9: **2.345** (ε 0.075) |
| 12 | 21.1 / 41.0 / 49.9: **1.950** (ε 0.030) | 23.0 / 53.3 / 61.1: **2.240** (ε 0.040) | 27.4 / 63.5 / 69.8: **2.286** (ε 0.033) | 30.7 / 84.0 / 92.1: **2.801** (ε 0.098) |
| 16 | 25.8 / 51.2 / 65.9: **1.944** (ε 0.064) | 28.2 / 63.2 / 76.6: **2.069** (ε 0.061) | 36.8 / 84.0 / 94.2: **2.200** (ε 0.146) | 38.9 / 116.8 / 129.0: **2.976** (ε 0.172) |

#### GB10, window 8

| Q | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|
| 4 | 15.5 / 43.0 / 63.1: **2.773** (ε 0.089) | 27.6 / 94.4 / 129.7: **3.447** (ε 0.058) | 33.0 / 154.5 / 198.7: **4.692** (ε 0.023) | 51.1 / 214.2 / 287.9: **4.173** (ε 0.070) |
| 5 | 16.4 / 61.1 / 77.1: **3.767** (ε 0.076) | 28.6 / 123.3 / 157.7: **4.333** (ε 0.036) | 34.6 / 194.3 / 252.8: **5.607** (ε 0.372) | 53.4 / 262.7 / 341.0: **4.870** (ε 0.221) |
| 6 | 19.1 / 71.7 / 88.1: **3.810** (ε 0.093) | 33.4 / 147.5 / 179.9: **4.360** (ε 0.043) | 40.6 / 229.0 / 289.3: **5.640** (ε 0.019) | 62.8 / 307.9 / 392.9: **4.925** (ε 0.177) |
| 8 | 23.3 / 93.6 / 108.8: **4.093** (ε 0.120) | 40.6 / 196.9 / 234.3: **4.921** (ε 0.050) | 49.7 / 298.1 / 366.7: **6.016** (ε 0.119) | 79.4 / 396.4 / 490.2: **4.993** (ε 0.228) |
| 9 | 23.6 / 106.9 / 122.9: **4.491** (ε 0.138) | 41.9 / 220.8 / 262.1: **5.269** (ε 0.048) | 52.8 / 331.8 / 408.1: **6.274** (ε 0.288) | 82.9 / 439.3 / 541.7: **5.276** (ε 0.133) |
| 12 | 27.8 / 143.6 / 158.7: **5.056** (ε 0.262) | 51.2 / 290.8 / 341.3: **5.696** (ε 0.047) | 61.4 / 434.2 / 522.7: **7.033** (ε 0.192) | 97.8 / 575.1 / 695.3: **5.884** (ε 0.212) |
| 16 | 36.9 / 191.5 / 207.2: **5.010** (ε 0.178) | 67.6 / 384.9 / 449.1: **5.754** (ε 0.118) | 85.0 / 569.3 / 672.3: **6.690** (ε 0.356) | 143.4 / 758.8 / 902.2: **5.293** (ε 0.267) |

### R3d fused handoff

All calls use 24 layers and strict checks. Times are µs per call; graph replay
columns measure the fused path. Slots and active rows are both 4 for (16,32),
and both 8 for (16,48). Launch counts are per-layer loop / fused call.

| Device | (Hk,Hv) | Per-layer host µs | Fused host µs | Per-layer device µs | Fused device µs | Graph host µs | Graph device µs | Launches per call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5090 | (16,32) | 5286.04 | 12.76 | 5837.74 | 18.36 | 1.74 | 20.48 | 120 / 2 |
| RTX 5090 | (16,48) | 5395.62 | 12.71 | 6015.19 | 37.68 | 1.84 | 22.53 | 120 / 2 |
| RTX PRO 6000 | (16,32) | 5851.41 | 17.52 | 6104.43 | 24.43 | 1.80 | 24.58 | 120 / 2 |
| RTX PRO 6000 | (16,48) | 5548.00 | 13.72 | 5776.96 | 24.71 | 2.18 | 24.70 | 120 / 2 |
| GB10 | (16,32) | 4934.21 | 13.14 | 4991.06 | 24.69 | 1.95 | 25.51 | 120 / 2 |
| GB10 | (16,48) | 5973.86 | 13.23 | 5996.56 | 89.42 | 1.97 | 81.59 | 120 / 2 |

### GPU tests

Reported GPU runs of `test_gdn_replay_delayed_commit.py` plus
`test_gdn_replay_handoff_layers.py`, and the R3d fix tests:

| Device | Delayed commit + handoff layers | R3d fix tests |
|---|---:|---:|
| RTX 5090 | 96/96 | 73/73 |
| RTX PRO 6000 | 96/96 | 73/73 |
| GB10 | 96/96 | 73/73 |

Existing trusted K2 suite regression:

Existing trusted K2 suite (`tests/gpu/test_gdn_replay_trusted_k2.py`, with the
R3c kernels in place; the delayed mode is opt-in, so these cover the unchanged eager
paths): 475/475 on the RTX 5090 (12 shards) and on the RTX PRO 6000 (12 shards);
GB10 475/475 (the 35-case CI core plus 440 exhaustive cases in 6 shards). The
product-bitwise replay CuTe cases (`-m bitwise tests/gpu/test_gdn_replay_cute.py`)
pass 88/88 on GB10.

R3c retains eager compatibility and offers measured delayed opt-in after consumer
flushes, with per-round gains about 1.04–1.33× where accepted rows fit the ring;
K=1 ties. The engine uses window 4; window 8 roughly doubles the qualifying Q
range. Ring storage is `(L+3)·W·2` bytes per slot and layer: 115584 → 181632 bytes
for (16,32), and 144704 → 227392 bytes for (16,48). The comparator clears the
faster external bar beyond ε on every row (two RTX 5090 rows after a re-measure).
R3d reduces strict 24-layer handoff from 120 launches to two and host submission
to 12.71–17.52 µs; the reported GPU suites pass on all three devices, with the
trusted K2 regression receipt awaiting the orchestrator.
