<!-- SPDX-License-Identifier: MIT -->
# Per-expert MoE LoRA forward vs vLLM 0.31.0 (ML1)

Produced by `python -m benchmark.kernel_tuning.moe_lora_market` (default preset, 21
paired samples per row). Shape: Qwen3.5-35B-A3B routed MoE (E=256, top_k=8, H=2048,
I=512, BF16), shared expert Is=512. Each cell is the median paired speedup
`t(vLLM strongest) / t(ours)`. vLLM's side is 0.31.0's own `TritonExperts.apply`
with a one-adapter `MoELoRAContext`, executed from the official wheel in single or
dual stream mode, whichever is faster for that row. Ours is `fused_experts_lora`
(base Triton grouped GEMMs, with a CuTe rewrite pending, plus
CuTe LoRA kernels). Shared rows compare the shared-expert gate/up + down LoRA
deltas against vLLM's punica shrink/expand.
"(tie)" means the median is below 1.0 but within the same-run A/A epsilon of two
identical vLLM instances; "(loss)" is below that. No row raised a correctness flag:
each row also checks ours and vLLM against an FP32 oracle.

Providers (backticked, per row):
- Routed: `pair` is the clustered CuTe pair kernel (split-K shrink, DSMEM
  reduce); `blk16`/`blk32` is the CuTe mma.sync block kernel with that LoRA row
  tile (fused SwiGLU epilogue on gate/up).
- Shared: `gate/up / down` modes from `_shared_policy.SHARED_TABLE`. `C` is the
  CuTe pair kernel at T<=16, otherwise the CuTe block kernel. `D<shrink BM>.<split
  K>.<expand BM>.<BN>.<stages>` is the dense two-launch CuTe kernel pair
  (`_cute_dense`). `L` is the library skinny GEMMs (FP32 z, in-place `addmm_`).
  The RTX PRO 6000 uses the RTX 5090 table (same sm_120 table class).

Rows: T=1..8192 at ranks 8/16/32/64/128, plus long bands T=16384 and 32768 at every
rank (EP1 on all devices, EP2 on the RTX 5090) and a flatness row at T=65536 (all
ranks on GB10; rank 16 on the RTX PRO 6000, where the rank-128 rerun at the final
code did not run this round and is pending (its pre-fix run read 0.984); on
the RTX 5090 the routed T=65536 row does not fit beside the comparator's graphs and
workspaces in 32 GB and is omitted there; the shared T=65536 rows run on all three).
Routed T=65536 rows use a two-input pool.

Snapshots: dispatch changes landed in steps (grid trim for decode batches, routed
pair thresholds, per-device base tiles, register shrink accumulators for large
ranks, 32-row rank-128 blocks). Rows those changes touch were re-measured and
replace the earlier runs: every routed rank-64/128 block row (T>=32 on all
devices and EP2), RTX 5090 EP1 T=16-32 and T=256-1024, EP2 T=16 and T=512,
RTX PRO 6000 T=16-32, GB10 T<=64 and T=512. So every row ran on code that is
identical, for that device, batch size and rank, to the final commit. GB10 T=64
ranks 8 and 16 use 61 interleaved samples (X925-pinned); the GB10 shared rows at
T=512-2048, ranks 16/32, use 41 samples (GB10 noise is high).
### RTX 5090 — routed, EP1

80 rows: PASS 80, TIE 0, FAIL 0, correctness flags 0; speedup median 1.101, min 1.003, max 1.934.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 1.24 `pair` | 1.32 `pair` | 1.53 `pair` | 1.93 `pair` | 1.83 `pair` |
| 2 | 1.16 `pair` | 1.21 `pair` | 1.39 `pair` | 1.55 `pair` | 1.45 `pair` |
| 4 | 1.10 `pair` | 1.12 `pair` | 1.23 `pair` | 1.16 `pair` | 1.27 `pair` |
| 8 | 1.08 `pair` | 1.10 `pair` | 1.06 `pair` | 1.17 `pair` | 1.23 `pair` |
| 16 | 1.04 `pair` | 1.03 `pair` | 1.03 `pair` | 1.08 `pair` | 1.20 `pair` |
| 32 | 1.04 `pair` | 1.03 `pair` | 1.03 `blk16` | 1.08 `blk16` | 1.41 `blk16` |
| 64 | 1.01 `pair` | 1.01 `pair` | 1.01 `blk16` | 1.14 `blk16` | 1.48 `blk16` |
| 128 | 1.01 `blk16` | 1.01 `blk16` | 1.02 `blk16` | 1.08 `blk16` | 1.75 `blk16` |
| 256 | 1.01 `blk16` | 1.01 `blk16` | 1.02 `blk16` | 1.08 `blk16` | 1.86 `blk16` |
| 512 | 1.01 `blk16` | 1.00 `blk16` | 1.02 `blk16` | 1.07 `blk16` | 1.81 `blk16` |
| 1024 | 1.07 `blk16` | 1.06 `blk16` | 1.11 `blk16` | 1.48 `blk16` | 1.41 `blk16` |
| 2048 | 1.07 `blk16` | 1.05 `blk16` | 1.07 `blk32` | 1.43 `blk32` | 1.27 `blk32` |
| 4096 | 1.08 `blk16` | 1.07 `blk16` | 1.08 `blk32` | 1.36 `blk32` | 1.16 `blk32` |
| 8192 | 1.10 `blk16` | 1.09 `blk16` | 1.08 `blk32` | 1.31 `blk32` | 1.10 `blk32` |
| 16384 | 1.12 `blk16` | 1.11 `blk16` | 1.10 `blk32` | 1.32 `blk32` | 1.07 `blk32` |
| 32768 | 1.17 `blk16` | 1.16 `blk16` | 1.13 `blk32` | 1.32 `blk32` | 1.05 `blk32` |

### RTX PRO 6000 — routed, EP1

81 rows: PASS 81, TIE 0, FAIL 0, correctness flags 0; speedup median 1.121, min 1.010, max 1.988.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 1.26 `pair` | 1.34 `pair` | 1.59 `pair` | 1.99 `pair` | 1.84 `pair` |
| 2 | 1.17 `pair` | 1.25 `pair` | 1.43 `pair` | 1.62 `pair` | 1.47 `pair` |
| 4 | 1.13 `pair` | 1.17 `pair` | 1.27 `pair` | 1.18 `pair` | 1.26 `pair` |
| 8 | 1.07 `pair` | 1.08 `pair` | 1.05 `pair` | 1.14 `pair` | 1.20 `pair` |
| 16 | 1.09 `pair` | 1.09 `pair` | 1.06 `pair` | 1.12 `pair` | 1.10 `pair` |
| 32 | 1.04 `pair` | 1.04 `pair` | 1.02 `blk16` | 1.09 `blk16` | 1.21 `blk16` |
| 64 | 1.01 `pair` | 1.01 `pair` | 1.01 `blk16` | 1.14 `blk16` | 1.40 `blk16` |
| 128 | 1.02 `blk16` | 1.02 `blk16` | 1.04 `blk16` | 1.08 `blk16` | 1.55 `blk16` |
| 256 | 1.02 `blk16` | 1.01 `blk16` | 1.03 `blk16` | 1.08 `blk16` | 1.64 `blk16` |
| 512 | 1.04 `blk16` | 1.03 `blk16` | 1.04 `blk16` | 1.09 `blk16` | 1.64 `blk16` |
| 1024 | 1.07 `blk16` | 1.06 `blk16` | 1.09 `blk16` | 1.37 `blk16` | 1.28 `blk16` |
| 2048 | 1.07 `blk16` | 1.05 `blk16` | 1.06 `blk32` | 1.32 `blk32` | 1.16 `blk32` |
| 4096 | 1.08 `blk16` | 1.07 `blk16` | 1.07 `blk32` | 1.31 `blk32` | 1.11 `blk32` |
| 8192 | 1.13 `blk16` | 1.12 `blk16` | 1.13 `blk32` | 1.38 `blk32` | 1.11 `blk32` |
| 16384 | 1.17 `blk16` | 1.16 `blk16` | 1.16 `blk32` | 1.40 `blk32` | 1.09 `blk32` |
| 32768 | 1.20 `blk16` | 1.19 `blk16` | 1.19 `blk32` | 1.41 `blk32` | 1.08 `blk32` |
| 65536 | - | 1.23 `blk16` | - | - | - |

### GB10 — routed, EP1

85 rows: PASS 85, TIE 0, FAIL 0, correctness flags 0; speedup median 1.062, min 1.005, max 1.362.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 1.05 `pair` | 1.06 `pair` | 1.13 `pair` | 1.20 `pair` | 1.11 `pair` |
| 2 | 1.03 `pair` | 1.03 `pair` | 1.06 `pair` | 1.06 `pair` | 1.05 `pair` |
| 4 | 1.03 `pair` | 1.02 `pair` | 1.06 `pair` | 1.02 `pair` | 1.03 `pair` |
| 8 | 1.02 `pair` | 1.01 `pair` | 1.02 `pair` | 1.02 `pair` | 1.12 `pair` |
| 16 | 1.01 `pair` | 1.01 `pair` | 1.03 `pair` | 1.03 `pair` | 1.13 `pair` |
| 32 | 1.01 `pair` | 1.01 `pair` | 1.03 `blk16` | 1.02 `blk16` | 1.08 `blk16` |
| 64 | 1.01 `pair` | 1.01 `pair` | 1.01 `blk16` | 1.03 `blk16` | 1.09 `blk16` |
| 128 | 1.04 `pair` | 1.03 `pair` | 1.06 `blk16` | 1.06 `blk16` | 1.30 `blk16` |
| 256 | 1.04 `blk16` | 1.03 `blk16` | 1.06 `blk16` | 1.06 `blk16` | 1.28 `blk16` |
| 512 | 1.05 `blk16` | 1.05 `blk16` | 1.07 `blk16` | 1.07 `blk16` | 1.28 `blk16` |
| 1024 | 1.02 `blk16` | 1.02 `blk16` | 1.04 `blk16` | 1.17 `blk16` | 1.15 `blk16` |
| 2048 | 1.03 `blk16` | 1.03 `blk16` | 1.04 `blk32` | 1.15 `blk32` | 1.13 `blk32` |
| 4096 | 1.08 `blk16` | 1.07 `blk16` | 1.08 `blk32` | 1.19 `blk32` | 1.14 `blk32` |
| 8192 | 1.13 `blk16` | 1.11 `blk16` | 1.10 `blk32` | 1.25 `blk32` | 1.15 `blk32` |
| 16384 | 1.16 `blk16` | 1.16 `blk16` | 1.15 `blk32` | 1.27 `blk32` | 1.14 `blk32` |
| 32768 | 1.20 `blk16` | 1.19 `blk16` | 1.18 `blk32` | 1.31 `blk32` | 1.15 `blk32` |
| 65536 | 1.27 `blk16` | 1.26 `blk16` | 1.25 `blk32` | 1.36 `blk32` | 1.19 `blk32` |

### RTX 5090 — routed, EP2 (128 local experts)

80 rows: PASS 80, TIE 0, FAIL 0, correctness flags 0; speedup median 1.146, min 1.024, max 2.263.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 1.48 `pair` | 1.69 `pair` | 1.55 `pair` | 1.83 `pair` | 2.26 `pair` |
| 2 | 1.37 `pair` | 1.46 `pair` | 1.41 `pair` | 1.57 `pair` | 1.82 `pair` |
| 4 | 1.22 `pair` | 1.33 `pair` | 1.25 `pair` | 1.34 `pair` | 1.46 `pair` |
| 8 | 1.17 `pair` | 1.20 `pair` | 1.16 `pair` | 1.21 `pair` | 1.25 `pair` |
| 16 | 1.14 `pair` | 1.13 `pair` | 1.14 `pair` | 1.21 `pair` | 1.37 `pair` |
| 32 | 1.16 `pair` | 1.14 `pair` | 1.10 `blk16` | 1.16 `blk16` | 1.50 `blk16` |
| 64 | 1.07 `pair` | 1.06 `pair` | 1.05 `blk16` | 1.17 `blk16` | 1.52 `blk16` |
| 128 | 1.02 `blk16` | 1.03 `blk16` | 1.06 `blk16` | 1.13 `blk16` | 1.89 `blk16` |
| 256 | 1.03 `blk16` | 1.03 `blk16` | 1.05 `blk16` | 1.13 `blk16` | 1.87 `blk16` |
| 512 | 1.05 `blk16` | 1.04 `blk16` | 1.05 `blk16` | 1.12 `blk16` | 1.80 `blk16` |
| 1024 | 1.09 `blk16` | 1.07 `blk16` | 1.05 `blk32` | 1.38 `blk32` | 1.21 `blk32` |
| 2048 | 1.06 `blk16` | 1.05 `blk16` | 1.07 `blk32` | 1.38 `blk32` | 1.24 `blk32` |
| 4096 | 1.07 `blk16` | 1.06 `blk16` | 1.09 `blk32` | 1.41 `blk32` | 1.20 `blk32` |
| 8192 | 1.08 `blk16` | 1.07 `blk16` | 1.05 `blk32` | 1.23 `blk32` | 1.05 `blk32` |
| 16384 | 1.12 `blk16` | 1.11 `blk16` | 1.09 `blk32` | 1.27 `blk32` | 1.07 `blk32` |
| 32768 | 1.15 `blk16` | 1.14 `blk16` | 1.12 `blk32` | 1.28 `blk32` | 1.06 `blk32` |

### RTX 5090 — shared

85 rows: PASS 85, TIE 0, FAIL 0, correctness flags 0; speedup median 1.665, min 1.039, max 4.060.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 4.06 `C/C` | 3.17 `C/C` | 2.69 `D16.32.16.64.2/C` | 2.48 `D16.32.16.64.2/D16.4.16.64.2` | 2.79 `D16.32.16.64.2/D16.8.16.64.2` |
| 2 | 3.22 `C/C` | 2.61 `C/C` | 2.14 `D16.32.16.64.2/C` | 2.04 `D16.32.16.64.2/D16.4.16.64.2` | 2.47 `D16.32.16.64.2/D16.8.16.64.2` |
| 4 | 3.12 `C/C` | 2.49 `C/C` | 2.12 `D16.32.16.64.2/C` | 2.01 `D16.32.16.64.2/D16.4.16.64.2` | 2.29 `D16.32.16.64.2/D16.8.16.64.2` |
| 8 | 3.00 `C/C` | 2.40 `C/C` | 2.05 `D16.32.16.64.2/C` | 1.97 `D16.32.16.64.2/D16.4.16.64.2` | 2.18 `D16.32.16.64.2/D16.8.16.64.2` |
| 16 | 2.75 `C/C` | 2.14 `C/C` | 1.96 `D16.32.16.64.2/C` | 1.96 `D16.32.16.64.2/D16.4.16.64.2` | 2.17 `D16.32.16.64.2/D16.8.16.64.2` |
| 32 | 2.37 `D16.16.16.128.2/C` | 2.16 `D32.32.16.64.2/D16.1.16.64.4` | 1.93 `D16.32.16.64.2/D16.8.16.128.2` | 1.97 `D16.32.16.64.2/D16.4.16.64.2` | 2.22 `D16.32.16.64.2/D16.8.16.64.2` |
| 64 | 2.46 `D16.16.16.128.3/D16.1.16.128.4` | 2.16 `D16.32.16.64.2/D16.1.16.64.3` | 2.00 `D16.16.16.64.2/D16.8.16.64.2` | 2.15 `D16.16.16.64.3/D16.8.16.128.2` | 2.28 `D16.16.16.64.2/D16.8.16.128.2` |
| 128 | 2.19 `D16.16.16.128.2/D16.1.16.128.2` | 1.99 `D32.32.16.64.2/D16.1.16.64.3` | 1.67 `D16.16.16.128.2/D16.8.16.128.2` | 1.58 `D16.16.16.64.3/D16.8.16.128.2` | 1.50 `D16.16.32.64.2/D16.8.32.64.2` |
| 256 | 1.94 `D16.16.16.64.2/D16.1.16.128.2` | 1.66 `D16.16.16.128.2/D16.1.16.64.3` | 1.47 `D32.16.16.64.2/D16.8.32.64.2` | 1.37 `D32.16.16.64.2/D16.8.32.64.2` | 1.30 `D32.16.32.64.2/D16.8.32.64.2` |
| 512 | 1.69 `D32.16.16.64.2/D16.1.16.64.3` | 1.44 `D32.8.32.128.2/D16.1.64.64.3` | 1.27 `D64.16.32.64.2/D16.4.64.64.2` | 1.22 `D32.8.64.64.2/D32.8.64.128.2` | 1.22 `D32.8.64.64.2/D32.8.64.64.2` |
| 1024 | 1.49 `D32.16.16.64.2/D16.1.32.64.2` | 1.23 `D64.16.32.64.2/D16.1.64.64.2` | 1.18 `D64.8.64.64.2/D32.4.32.64.2` | 1.21 `D32.4.32.64.2/D32.4.64.64.2` | 1.18 `L/L` |
| 2048 | 1.63 `D64.4.32.64.2/D32.1.64.64.3` | 1.20 `D64.8.64.64.2/D16.1.64.64.2` | 1.16 `D64.8.64.64.2/D64.4.64.64.2` | 1.23 `D64.4.64.128.2/D64.4.64.64.3` | 1.53 `L/L` |
| 4096 | 1.39 `D64.4.64.64.2/D64.1.64.64.2` | 1.11 `D64.4.64.64.2/D32.1.64.64.2` | 1.19 `D64.4.32.128.2/L` | 1.39 `L/L` | 1.58 `L/L` |
| 8192 | 1.39 `D64.4.64.64.2/D64.1.64.64.2` | 1.04 `D64.4.64.64.2/D32.1.64.64.2` | 1.16 `D64.4.32.128.2/L` | 1.43 `L/L` | 1.50 `L/L` |
| 16384 | 1.18 `D64.4.64.64.2/D64.1.64.64.2` | 1.09 `D64.4.64.64.2/D32.1.64.64.2` | 1.13 `D64.4.32.128.2/L` | 1.46 `L/L` | 1.56 `L/L` |
| 32768 | 1.18 `D64.4.64.64.2/D64.1.64.64.2` | 1.11 `D64.4.64.64.2/D32.1.64.64.2` | 1.22 `D64.4.32.128.2/L` | 1.43 `L/L` | 1.68 `L/L` |
| 65536 | 1.13 `D64.4.64.64.2/D64.1.64.64.2` | 1.07 `D64.4.64.64.2/D32.1.64.64.2` | 1.20 `D64.4.32.128.2/L` | 1.42 `L/L` | 1.70 `L/L` |

### RTX PRO 6000 — shared

85 rows: PASS 85, TIE 0, FAIL 0, correctness flags 0; speedup median 1.688, min 1.042, max 3.999.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 4.00 `C/C` | 3.14 `C/C` | 2.68 `D16.32.16.64.2/C` | 2.47 `D16.32.16.64.2/D16.4.16.64.2` | 2.76 `D16.32.16.64.2/D16.8.16.64.2` |
| 2 | 3.20 `C/C` | 2.59 `C/C` | 2.14 `D16.32.16.64.2/C` | 2.03 `D16.32.16.64.2/D16.4.16.64.2` | 2.30 `D16.32.16.64.2/D16.8.16.64.2` |
| 4 | 3.24 `C/C` | 2.60 `C/C` | 2.07 `D16.32.16.64.2/C` | 1.97 `D16.32.16.64.2/D16.4.16.64.2` | 2.17 `D16.32.16.64.2/D16.8.16.64.2` |
| 8 | 2.97 `C/C` | 2.56 `C/C` | 2.03 `D16.32.16.64.2/C` | 1.95 `D16.32.16.64.2/D16.4.16.64.2` | 2.15 `D16.32.16.64.2/D16.8.16.64.2` |
| 16 | 2.95 `C/C` | 2.38 `C/C` | 1.97 `D16.32.16.64.2/C` | 1.92 `D16.32.16.64.2/D16.4.16.64.2` | 2.06 `D16.32.16.64.2/D16.8.16.64.2` |
| 32 | 2.38 `D16.16.16.128.2/C` | 2.14 `D32.32.16.64.2/D16.1.16.64.4` | 1.91 `D16.32.16.64.2/D16.8.16.128.2` | 1.96 `D16.32.16.64.2/D16.4.16.64.2` | 2.11 `D16.32.16.64.2/D16.8.16.64.2` |
| 64 | 2.48 `D16.16.16.128.3/D16.1.16.128.4` | 2.24 `D16.32.16.64.2/D16.1.16.64.3` | 2.03 `D16.16.16.64.2/D16.8.16.64.2` | 2.05 `D16.16.16.64.3/D16.8.16.128.2` | 2.18 `D16.16.16.64.2/D16.8.16.128.2` |
| 128 | 2.23 `D16.16.16.128.2/D16.1.16.128.2` | 2.03 `D32.32.16.64.2/D16.1.16.64.3` | 1.69 `D16.16.16.128.2/D16.8.16.128.2` | 1.69 `D16.16.16.64.3/D16.8.16.128.2` | 1.54 `D16.16.32.64.2/D16.8.32.64.2` |
| 256 | 2.06 `D16.16.16.64.2/D16.1.16.128.2` | 1.74 `D16.16.16.128.2/D16.1.16.64.3` | 1.56 `D32.16.16.64.2/D16.8.32.64.2` | 1.47 `D32.16.16.64.2/D16.8.32.64.2` | 1.39 `D32.16.32.64.2/D16.8.32.64.2` |
| 512 | 1.79 `D32.16.16.64.2/D16.1.16.64.3` | 1.52 `D32.8.32.128.2/D16.1.64.64.3` | 1.32 `D64.16.32.64.2/D16.4.64.64.2` | 1.29 `D32.8.64.64.2/D32.8.64.128.2` | 1.27 `D32.8.64.64.2/D32.8.64.64.2` |
| 1024 | 1.61 `D32.16.16.64.2/D16.1.32.64.2` | 1.37 `D64.16.32.64.2/D16.1.64.64.2` | 1.20 `D64.8.64.64.2/D32.4.32.64.2` | 1.19 `D32.4.32.64.2/D32.4.64.64.2` | 1.19 `L/L` |
| 2048 | 1.66 `D64.4.32.64.2/D32.1.64.64.3` | 1.11 `D64.8.64.64.2/D16.1.64.64.2` | 1.12 `D64.8.64.64.2/D64.4.64.64.2` | 1.21 `D64.4.64.128.2/D64.4.64.64.3` | 1.47 `L/L` |
| 4096 | 1.40 `D64.4.64.64.2/D64.1.64.64.2` | 1.09 `D64.4.64.64.2/D32.1.64.64.2` | 1.10 `D64.4.32.128.2/L` | 1.37 `L/L` | 1.75 `L/L` |
| 8192 | 1.44 `D64.4.64.64.2/D64.1.64.64.2` | 1.10 `D64.4.64.64.2/D32.1.64.64.2` | 1.14 `D64.4.32.128.2/L` | 1.47 `L/L` | 1.94 `L/L` |
| 16384 | 1.21 `D64.4.64.64.2/D64.1.64.64.2` | 1.04 `D64.4.64.64.2/D32.1.64.64.2` | 1.04 `D64.4.32.128.2/L` | 1.22 `L/L` | 1.64 `L/L` |
| 32768 | 1.18 `D64.4.64.64.2/D64.1.64.64.2` | 1.09 `D64.4.64.64.2/D32.1.64.64.2` | 1.11 `D64.4.32.128.2/L` | 1.27 `L/L` | 1.64 `L/L` |
| 65536 | 1.18 `D64.4.64.64.2/D64.1.64.64.2` | 1.12 `D64.4.64.64.2/D32.1.64.64.2` | 1.17 `D64.4.32.128.2/L` | 1.26 `L/L` | 1.65 `L/L` |

### GB10 — shared

85 rows: PASS 85, TIE 0, FAIL 0, correctness flags 0; speedup median 1.404, min 1.012, max 3.501.

| T \ rank | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 1 | 3.05 `D16.16.16.64.2/D16.1.16.64.4` | 2.48 `D16.32.16.64.2/D16.1.16.64.2` | 2.47 `D16.16.16.64.2/D16.4.16.64.2` | 2.93 `D16.16.16.64.2/D16.8.16.64.2` | 3.50 `D16.16.16.64.2/D16.8.16.64.2` |
| 2 | 2.52 `D16.16.16.64.2/D16.1.16.64.4` | 2.07 `D16.32.16.64.2/D16.1.16.64.2` | 2.05 `D16.16.16.64.2/D16.4.16.64.2` | 2.53 `D16.16.16.64.2/D16.8.16.64.2` | 3.10 `D16.16.16.64.2/D16.8.16.64.2` |
| 4 | 2.49 `D16.16.16.64.2/D16.1.16.64.4` | 2.05 `D16.32.16.64.2/D16.1.16.64.2` | 2.00 `D16.16.16.64.2/D16.4.16.64.2` | 2.48 `D16.16.16.64.2/D16.8.16.64.2` | 2.99 `D16.16.16.64.2/D16.8.16.64.2` |
| 8 | 2.46 `D16.16.16.64.2/D16.1.16.64.4` | 2.00 `D16.32.16.64.2/D16.1.16.64.2` | 1.97 `D16.16.16.64.2/D16.4.16.64.2` | 2.43 `D16.16.16.64.2/D16.8.16.64.2` | 2.91 `D16.16.16.64.2/D16.8.16.64.2` |
| 16 | 2.42 `D16.16.16.64.2/D16.1.16.64.4` | 1.92 `D16.32.16.64.2/D16.1.16.64.2` | 2.00 `D16.16.16.64.2/D16.4.16.64.2` | 2.53 `D16.16.16.64.2/D16.8.16.64.2` | 2.83 `D16.16.16.64.2/D16.8.16.64.2` |
| 32 | 2.29 `D16.16.16.64.2/D16.1.32.64.4` | 2.07 `D16.16.16.128.2/D16.1.16.64.3` | 2.31 `D16.16.16.64.2/D16.8.16.128.2` | 2.46 `D16.16.16.64.2/D16.8.16.64.2` | 2.54 `D16.16.16.64.2/D16.8.16.128.2` |
| 64 | 2.50 `D16.16.16.128.2/D16.1.16.64.4` | 2.38 `D32.16.32.64.3/D16.1.16.64.4` | 2.37 `D16.32.16.64.2/D16.8.16.64.2` | 2.67 `D32.16.16.64.2/D16.8.16.64.2` | 3.19 `D16.8.32.64.2/D16.8.32.128.2` |
| 128 | 1.75 `D16.16.16.64.2/D16.1.64.64.4` | 1.52 `D64.16.32.64.3/D16.1.32.128.3` | 1.33 `D64.16.32.64.2/D32.8.16.64.2` | 1.32 `D32.8.64.64.2/D16.4.32.64.2` | 1.24 `D32.8.64.64.2/D16.4.32.64.3` |
| 256 | 1.82 `D32.16.32.64.2/D16.1.32.64.4` | 1.36 `D64.16.16.64.2/D16.1.64.64.4` | 1.26 `D64.8.32.64.2/D16.8.32.64.2` | 1.25 `D64.8.64.128.2/D32.4.64.64.3` | 1.21 `L/L` |
| 512 | 1.47 `D16.1.32.64.4/D16.1.64.64.3` | 1.17 `D64.16.64.64.2/D16.1.64.64.4` | 1.10 `D64.8.64.64.2/D16.1.32.64.3` | 1.17 `D64.4.64.64.2/L` | 1.30 `L/L` |
| 1024 | 1.24 `D64.6.64.64.4/D32.1.64.64.4` | 1.03 `D64.8.64.128.2/D32.1.64.64.3` | 1.06 `D64.4.64.128.2/D32.1.64.64.4` | 1.22 `L/L` | 1.57 `L/L` |
| 2048 | 1.23 `C/D32.1.64.64.3` | 1.08 `C/D64.8.64.64.2` | 1.01 `L/D64.4.64.64.2` | 1.11 `L/L` | 1.29 `L/L` |
| 4096 | 1.09 `C/D32.1.64.128.2` | 1.05 `C/D32.1.64.128.2` | 1.05 `C/D16.1.64.128.4` | 1.09 `L/D16.1.64.128.4` | 1.25 `L/L` |
| 8192 | 1.21 `C/L` | 1.19 `C/L` | 1.19 `C/L` | 1.17 `L/L` | 1.29 `L/L` |
| 16384 | 1.22 `C/L` | 1.21 `C/L` | 1.21 `C/L` | 1.13 `L/L` | 1.27 `L/L` |
| 32768 | 1.23 `C/L` | 1.12 `C/L` | 1.15 `C/L` | 1.20 `L/L` | 1.57 `L/L` |
| 65536 | 1.24 `C/L` | 1.27 `C/L` | 1.34 `C/L` | 1.40 `L/L` | 1.76 `L/L` |

GB10 routed T=64, rank 16 (a tie at 0.999 in the previous round): with GB10 base
tiles BM16/8 warps/4 stages at 33-64 tokens it measures 1.013 (95% CI
1.012-1.014, 61 interleaved pairs, X925-pinned); the same row at the previous
code read 0.9994 (CI 0.998-1.001) over 61 pairs.

Rank-128 prefill rows (previously 0.979 at T=16384 and 0.964 at T=32768 on the RTX
5090, 0.994 at T=32768 on the RTX PRO 6000; now 1.075, 1.053 and 1.075): the block
shrink now keeps large-rank FP32 accumulators in registers instead of round-tripping
them through shared memory per 32-row factor panel, and rank 128 takes 32-row blocks
once experts average 64 rows. RTX 5090 LoRA increment at T=32768: 9.46 ms before,
8.97 ms with register accumulators, 8.02 ms with 32-row blocks (vLLM 8.6 ms).

GPU suites at code identical to the final commit: `pytest
tests/gpu/test_moe_lora.py` (including exhaustive cases and the long-context
bands T=16384/65536/262144) passed 3566/3566 on the RTX PRO 6000 and GB10 and
3565/3566 on the RTX 5090, where the T=262144 routed rank-128 band skipped for
memory (12 GiB free on that node; it passed on the other two devices).

Profiling evidence for the shared-expert dense kernels is in
`moe_lora_shared_profile.md`.
