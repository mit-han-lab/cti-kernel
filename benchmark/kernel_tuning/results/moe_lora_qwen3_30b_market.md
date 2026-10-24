<!-- SPDX-License-Identifier: MIT -->
# Per-expert MoE LoRA forward vs vLLM 0.31.0: Qwen3-30B-A3B (ML1b)

Produced by `python -m benchmark.kernel_tuning.moe_lora_market --model qwen3-30b-a3b`
(21 paired samples per row on the RTX 5090, 61 on GB10, host process pinned to the X925 cores on GB10).
Shape: Qwen3-30B-A3B routed MoE (E=128, top_k=8, H=2048, I=768, BF16), no shared
expert. Each cell is the median paired speedup `t(vLLM strongest) / t(ours)`. vLLM's
side is 0.31.0's own `TritonExperts.apply` with a one-adapter `MoELoRAContext`,
executed from the official wheel in single or dual stream mode, whichever is faster
for that row. Ours is `fused_experts_lora` (base Triton grouped GEMMs, with a
CuTe rewrite pending, plus CuTe LoRA kernels). "(tie)" means the
median is below 1.0 but within the same-run A/A epsilon of two identical vLLM
instances; "(loss)" is below that. No row raised a correctness flag: each row also
checks ours and vLLM against an FP32 oracle.

Providers (backticked, per row): `pair` is the clustered CuTe pair kernel (split-K
shrink, DSMEM reduce); `blk16`/`blk32` is the CuTe mma.sync block kernel with that
LoRA row tile (fused SwiGLU epilogue on gate/up). EP2 rows hold 64 local experts,
mask remote routes, and pass `num_global_experts=128` so LoRA row tiles are sized by
the expected local routes; they are local partial outputs (no collectives).

Rows: T=1..32768 at ranks 16/32/64 (EP1 on the RTX 5090 and GB10; EP2 on the RTX
5090 at T=1/8/64/512/1024/4096) and a flatness row at T=65536 (ranks 16 and 64,
two-input pool). On the RTX 5090 the T=65536 row does not fit beside the
comparator's graphs and workspaces in 32 GB (ours hits OutOfMemoryError at capture)
and is measured on GB10 only; its correctness is covered by the GPU long-context
tests. RTX PRO 6000 rows are deferred: that device was reserved for other work
during this round.

Snapshots: RTX 5090 EP1 T=64 was re-measured after the 33-64 token base-tile change
(16-row, 4-warp, 3-stage tiles with LoRA); EP2 rows were measured after the
`num_global_experts` row-tile change. The other rows predate those changes, which
do not touch them (different device or EP1 only).

### RTX 5090 — EP1

48 rows: PASS 48, TIE 0, FAIL 0, correctness flags 0; speedup median 1.077, min 1.003, max 1.779.

| T \ rank | 16 | 32 | 64 |
|---|---|---|---|
| 1 | 1.26 `pair` | 1.50 `pair` | 1.78 `pair` |
| 2 | 1.19 `pair` | 1.37 `pair` | 1.62 `pair` |
| 4 | 1.13 `pair` | 1.11 `pair` | 1.25 `pair` |
| 8 | 1.08 `pair` | 1.05 `pair` | 1.13 `pair` |
| 16 | 1.04 `pair` | 1.04 `pair` | 1.06 `pair` |
| 32 | 1.02 `pair` | 1.01 `blk16` | 1.07 `blk16` |
| 64 | 1.00 `pair` | 1.02 `blk16` | 1.13 `blk16` |
| 128 | 1.01 `blk16` | 1.04 `blk16` | 1.09 `blk16` |
| 256 | 1.01 `blk16` | 1.03 `blk16` | 1.07 `blk16` |
| 512 | 1.03 `blk16` | 1.04 `blk16` | 1.08 `blk16` |
| 1024 | 1.05 `blk16` | 1.06 `blk32` | 1.32 `blk32` |
| 2048 | 1.05 `blk16` | 1.07 `blk32` | 1.36 `blk32` |
| 4096 | 1.06 `blk16` | 1.04 `blk32` | 1.20 `blk32` |
| 8192 | 1.09 `blk16` | 1.07 `blk32` | 1.23 `blk32` |
| 16384 | 1.12 `blk16` | 1.10 `blk32` | 1.25 `blk32` |
| 32768 | 1.14 `blk16` | 1.11 `blk32` | 1.22 `blk32` |

### RTX 5090 — EP2 (64 local experts)

18 rows: PASS 18, TIE 0, FAIL 0, correctness flags 0; speedup median 1.147, min 1.025, max 1.673.

| T \ rank | 16 | 32 | 64 |
|---|---|---|---|
| 1 | 1.51 `pair` | 1.40 `pair` | 1.67 `pair` |
| 8 | 1.18 `pair` | 1.15 `pair` | 1.16 `pair` |
| 64 | 1.05 `pair` | 1.03 `blk16` | 1.22 `blk16` |
| 512 | 1.04 `blk16` | 1.06 `blk16` | 1.14 `blk16` |
| 1024 | 1.07 `blk16` | 1.03 `blk32` | 1.23 `blk32` |
| 4096 | 1.04 `blk16` | 1.04 `blk32` | 1.18 `blk32` |

### GB10 — EP1

50 rows: PASS 50, TIE 0, FAIL 0, correctness flags 0; speedup median 1.055, min 1.006, max 1.375.

| T \ rank | 16 | 32 | 64 |
|---|---|---|---|
| 1 | 1.04 `pair` | 1.09 `pair` | 1.16 `pair` |
| 2 | 1.03 `pair` | 1.04 `pair` | 1.04 `pair` |
| 4 | 1.02 `pair` | 1.03 `pair` | 1.01 `pair` |
| 8 | 1.01 `pair` | 1.02 `pair` | 1.03 `pair` |
| 16 | 1.01 `pair` | 1.02 `pair` | 1.02 `pair` |
| 32 | 1.01 `pair` | 1.02 `blk16` | 1.03 `blk16` |
| 64 | 1.01 `pair` | 1.02 `blk16` | 1.04 `blk16` |
| 128 | 1.02 `pair` | 1.06 `blk16` | 1.07 `blk16` |
| 256 | 1.04 `blk16` | 1.06 `blk16` | 1.07 `blk16` |
| 512 | 1.04 `blk16` | 1.06 `blk16` | 1.06 `blk16` |
| 1024 | 1.03 `blk16` | 1.03 `blk32` | 1.13 `blk32` |
| 2048 | 1.06 `blk16` | 1.05 `blk32` | 1.15 `blk32` |
| 4096 | 1.10 `blk16` | 1.09 `blk32` | 1.21 `blk32` |
| 8192 | 1.15 `blk16` | 1.14 `blk32` | 1.27 `blk32` |
| 16384 | 1.19 `blk16` | 1.18 `blk32` | 1.28 `blk32` |
| 32768 | 1.23 `blk16` | 1.23 `blk32` | 1.34 `blk32` |
| 65536 | 1.26 `blk16` | - | 1.38 `blk32` |
