<!-- SPDX-License-Identifier: MIT -->
# Per-expert MoE LoRA training step vs Unsloth (ML3)

Produced by `python benchmark/kernel_tuning/moe_lora_unsloth_bench.py` (15 paired,
interleaved samples per routed row, 21 per shared row; host process pinned to the
X925 cores on GB10). Each row times one training step of one projection group:
forward plus backward to `x`, the routing weights (routed rows) and every LoRA
factor, with frozen BF16 base weights and LoRA scale 1 (alpha = rank).

* **Ours:** `fused_experts_lora` / `shared_expert_lora` with trainable inputs (the
  ML2 autograd functions; CuTe LoRA shrink/expand, SwiGLU/routing-weight backward
  and grouped factor-gradient kernels; base GEMMs on the Triton provider; the
  shared expert on dense library GEMMs around the CuTe SwiGLU backward).
* **Unsloth:** `unsloth==2026.10.6`, `unsloth_zoo==2026.10.6`, `transformers==5.17.0`,
  `peft==0.21.2`, torch 2.13.0, Triton 3.7.1, in a separate environment. The bench
  imports Unsloth first (its patches), builds the transformers expert / MLP module of
  the model, and wraps it with PEFT the way Unsloth's `get_peft_model` does for MoE
  models (`target_parameters` on the expert `gate_up_proj` / `down_proj`,
  `target_modules` on the shared MLP). Every candidate below is timed on every row,
  and the fastest one is the bar:

  | Label | Kernel path |
  | --- | --- |
  | `grouped_mm` | `UNSLOTH_MOE_BACKEND=grouped_mm` (Unsloth's default): `unsloth_zoo/temporary_patches/moe_utils.py` `forward_native_grouped_mm`, separated LoRA, Triton grouped GEMMs from `moe_grouped_fp16.py` up to Unsloth's measured sm_12x rows-per-expert limits, `torch._grouped_mm` above them |
  | `grouped_mm_triton` | the same with `UNSLOTH_MOE_GROUPED_TRITON=1` (Triton grouped GEMMs at every size) |
  | `unsloth_triton` | `UNSLOTH_MOE_BACKEND=unsloth_triton`: `unsloth/kernels/moe/grouped_gemm` (fused permute grouped GEMM) |
  | `peft` | shared MLP: PEFT `lora.Linear` on gate/up/down (cuBLAS) |
  | `fast_lora` | shared MLP: `unsloth/kernels/fast_lora.py` `apply_lora_mlp_swiglu` (Unsloth's fused LoRA MLP autograd function) |

  Each candidate runs with BF16 adapters (`bf16`) and with FP32 adapters under BF16
  autocast (`fp32+autocast`, Unsloth's default `float32_mixed_precision` for BF16
  models). `native_torch` (a per-expert loop, ~50-80x slower) is excluded from the
  default grid. Gradient checkpointing is off on both sides (it only adds recompute).

Ours and every Unsloth candidate ran as CUDA graphs, except `grouped_mm` and
`unsloth_triton` on the same three rank-64 rows on both devices (Qwen3.5 T=8192, Qwen3
T=2048 and T=8192): there Unsloth's auto policy sends rank-64 factor GEMMs to
`torch._grouped_mm`, which copies group offsets to the host off sm_90/sm_100, so capture
fails and those two were timed with CUDA events; the fastest candidate on those rows
(`grouped_mm_triton`) was graph-captured. All candidates and ours read the
same base-weight storage, so no candidate gets warmer L2 than another under the
interleaving (on GB10 the whole T=1 shared working set fits in L2). `fast_lora`'s
backward is in place by default (it writes dX into its input `X`), so each shared
candidate has its own input tensor, restored from the same `x` before every warmup,
capture and timed replay (outside the timed region); Unsloth and ours run on
identical inputs, with `fast_lora` kept in its default in-place form. `Unsloth/ours` is
the median paired ratio against the fastest candidate; a row passes above 1.0.
Sanity checks in every row: the LoRA branch moves Unsloth's routed output (9.8-23%
relative change with the random adapters, so the adapters are applied), and on the
shared rows our forward matches PEFT's within 0.7% relative Frobenius with the same
adapters. Shapes: Qwen3.5-35B-A3B (E=256, H=2048, I=512, top-8, shared Is=512)
and Qwen3-30B-A3B (E=128, H=2048, I=768, top-8, no shared expert), BF16, EP1.

**Void rows:** EP2. Unsloth has no expert parallelism (every expert is local to its
module), so no EP2 Unsloth row exists; our EP2 path is unchanged from ML2.
**Owed:** RTX PRO 6000 rows (not measured in this round).

## Summary

* RTX 5090: 45/45, median 1.14x, min 1.016x
* GB10: 45/45, median 1.26x, min 1.042x

### RTX 5090

| model | part | T | rank | ours ms | Unsloth best ms | Unsloth path / adapters | Unsloth/ours | pass |
|---|---|---:|---:|---:|---:|---|---:|---|
| Qwen3-30B-A3B | routed | 1 | 16 | 0.308 | 0.336 | grouped_mm / bf16 | 1.098 | yes |
| Qwen3-30B-A3B | routed | 1 | 32 | 0.449 | 0.481 | grouped_mm_triton / bf16 | 1.073 | yes |
| Qwen3-30B-A3B | routed | 1 | 64 | 0.826 | 0.876 | grouped_mm / bf16 | 1.061 | yes |
| Qwen3-30B-A3B | routed | 64 | 16 | 1.754 | 1.783 | grouped_mm_triton / bf16 | 1.017 | yes |
| Qwen3-30B-A3B | routed | 64 | 32 | 1.879 | 1.944 | grouped_mm / bf16 | 1.034 | yes |
| Qwen3-30B-A3B | routed | 64 | 64 | 2.326 | 2.379 | grouped_mm_triton / bf16 | 1.023 | yes |
| Qwen3-30B-A3B | routed | 512 | 16 | 1.943 | 2.055 | grouped_mm_triton / bf16 | 1.058 | yes |
| Qwen3-30B-A3B | routed | 512 | 32 | 2.092 | 2.270 | grouped_mm / bf16 | 1.085 | yes |
| Qwen3-30B-A3B | routed | 512 | 64 | 2.613 | 2.749 | grouped_mm_triton / bf16 | 1.052 | yes |
| Qwen3-30B-A3B | routed | 2048 | 16 | 3.398 | 4.287 | grouped_mm_triton / bf16 | 1.262 | yes |
| Qwen3-30B-A3B | routed | 2048 | 32 | 3.663 | 4.479 | grouped_mm_triton / bf16 | 1.225 | yes |
| Qwen3-30B-A3B | routed | 2048 | 64 | 4.315 | 4.977 | grouped_mm_triton / bf16 | 1.152 | yes |
| Qwen3-30B-A3B | routed | 8192 | 16 | 11.497 | 13.631 | grouped_mm_triton / bf16 | 1.186 | yes |
| Qwen3-30B-A3B | routed | 8192 | 32 | 11.973 | 13.846 | grouped_mm_triton / bf16 | 1.157 | yes |
| Qwen3-30B-A3B | routed | 8192 | 64 | 13.234 | 14.614 | grouped_mm_triton / bf16 | 1.104 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 16 | 0.341 | 0.398 | grouped_mm_triton / bf16 | 1.170 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 32 | 0.542 | 0.640 | grouped_mm_triton / bf16 | 1.182 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 64 | 1.078 | 1.337 | grouped_mm_triton / bf16 | 1.238 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 16 | 2.054 | 2.087 | grouped_mm / bf16 | 1.016 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 32 | 2.266 | 2.362 | grouped_mm_triton / bf16 | 1.043 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 64 | 2.959 | 3.126 | grouped_mm_triton / bf16 | 1.056 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 16 | 2.514 | 2.649 | grouped_mm / bf16 | 1.054 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 32 | 2.779 | 2.985 | grouped_mm_triton / bf16 | 1.074 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 64 | 3.581 | 3.775 | grouped_mm / bf16 | 1.055 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 16 | 3.477 | 4.248 | grouped_mm / bf16 | 1.222 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 32 | 3.792 | 4.577 | grouped_mm / bf16 | 1.206 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 64 | 4.684 | 5.392 | grouped_mm / bf16 | 1.153 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 16 | 9.205 | 11.087 | grouped_mm_triton / bf16 | 1.204 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 32 | 9.703 | 11.425 | grouped_mm / bf16 | 1.177 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 64 | 11.117 | 12.276 | grouped_mm_triton / bf16 | 1.103 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 16 | 0.039 | 0.057 | fast_lora / bf16 | 1.476 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 32 | 0.041 | 0.059 | fast_lora / bf16 | 1.461 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 64 | 0.043 | 0.057 | fast_lora / bf16 | 1.340 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 16 | 0.061 | 0.080 | fast_lora / bf16 | 1.308 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 32 | 0.063 | 0.082 | peft / bf16 | 1.299 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 64 | 0.065 | 0.086 | peft / bf16 | 1.326 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 16 | 0.104 | 0.133 | fast_lora / bf16 | 1.274 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 32 | 0.109 | 0.139 | peft / bf16 | 1.284 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 64 | 0.117 | 0.145 | peft / bf16 | 1.246 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 16 | 0.261 | 0.299 | fast_lora / bf16 | 1.144 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 32 | 0.268 | 0.282 | fast_lora / bf16 | 1.056 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 64 | 0.291 | 0.300 | fast_lora / bf16 | 1.035 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 16 | 0.899 | 0.923 | fast_lora / bf16 | 1.028 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 32 | 0.926 | 0.946 | fast_lora / bf16 | 1.021 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 64 | 0.983 | 1.034 | fast_lora / bf16 | 1.052 | yes |

### GB10

| model | part | T | rank | ours ms | Unsloth best ms | Unsloth path / adapters | Unsloth/ours | pass |
|---|---|---:|---:|---:|---:|---|---:|---|
| Qwen3-30B-A3B | routed | 1 | 16 | 1.344 | 1.577 | grouped_mm_triton / bf16 | 1.172 | yes |
| Qwen3-30B-A3B | routed | 1 | 32 | 1.978 | 2.404 | grouped_mm_triton / bf16 | 1.215 | yes |
| Qwen3-30B-A3B | routed | 1 | 64 | 3.265 | 4.250 | grouped_mm / bf16 | 1.302 | yes |
| Qwen3-30B-A3B | routed | 64 | 16 | 11.353 | 11.997 | grouped_mm_triton / bf16 | 1.058 | yes |
| Qwen3-30B-A3B | routed | 64 | 32 | 12.037 | 13.072 | grouped_mm / bf16 | 1.085 | yes |
| Qwen3-30B-A3B | routed | 64 | 64 | 14.383 | 15.568 | grouped_mm_triton / bf16 | 1.091 | yes |
| Qwen3-30B-A3B | routed | 512 | 16 | 13.371 | 17.116 | grouped_mm / bf16 | 1.280 | yes |
| Qwen3-30B-A3B | routed | 512 | 32 | 14.520 | 18.301 | grouped_mm_triton / bf16 | 1.262 | yes |
| Qwen3-30B-A3B | routed | 512 | 64 | 16.447 | 20.611 | grouped_mm / bf16 | 1.252 | yes |
| Qwen3-30B-A3B | routed | 2048 | 16 | 21.058 | 36.957 | grouped_mm_triton / bf16 | 1.745 | yes |
| Qwen3-30B-A3B | routed | 2048 | 32 | 22.107 | 37.977 | grouped_mm_triton / bf16 | 1.712 | yes |
| Qwen3-30B-A3B | routed | 2048 | 64 | 24.526 | 40.347 | grouped_mm_triton / bf16 | 1.642 | yes |
| Qwen3-30B-A3B | routed | 8192 | 16 | 54.029 | 76.312 | grouped_mm_triton / bf16 | 1.409 | yes |
| Qwen3-30B-A3B | routed | 8192 | 32 | 56.005 | 77.619 | grouped_mm_triton / bf16 | 1.383 | yes |
| Qwen3-30B-A3B | routed | 8192 | 64 | 60.582 | 81.944 | grouped_mm_triton / bf16 | 1.346 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 16 | 1.512 | 1.866 | grouped_mm_triton / bf16 | 1.231 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 32 | 2.641 | 3.323 | grouped_mm_triton / bf16 | 1.258 | yes |
| Qwen3.5-35B-A3B | routed | 1 | 64 | 4.774 | 6.460 | grouped_mm_triton / bf16 | 1.354 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 16 | 14.073 | 14.632 | grouped_mm / bf16 | 1.042 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 32 | 15.573 | 16.422 | grouped_mm_triton / bf16 | 1.055 | yes |
| Qwen3.5-35B-A3B | routed | 64 | 64 | 18.716 | 20.336 | grouped_mm_triton / bf16 | 1.082 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 16 | 17.574 | 20.029 | grouped_mm_triton / bf16 | 1.148 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 32 | 19.182 | 22.181 | grouped_mm / bf16 | 1.144 | yes |
| Qwen3.5-35B-A3B | routed | 512 | 64 | 23.117 | 25.832 | grouped_mm / bf16 | 1.139 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 16 | 24.021 | 34.546 | grouped_mm / bf16 | 1.454 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 32 | 26.126 | 36.675 | grouped_mm / bf16 | 1.411 | yes |
| Qwen3.5-35B-A3B | routed | 2048 | 64 | 29.372 | 40.744 | grouped_mm_triton / bf16 | 1.376 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 16 | 50.907 | 68.660 | grouped_mm_triton / bf16 | 1.358 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 32 | 53.352 | 70.918 | grouped_mm_triton / bf16 | 1.327 | yes |
| Qwen3.5-35B-A3B | routed | 8192 | 64 | 59.139 | 75.258 | grouped_mm_triton / bf16 | 1.271 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 16 | 0.054 | 0.081 | fast_lora / bf16 | 1.512 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 32 | 0.056 | 0.082 | fast_lora / bf16 | 1.475 | yes |
| Qwen3.5-35B-A3B | shared | 1 | 64 | 0.070 | 0.085 | fast_lora / bf16 | 1.212 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 16 | 0.078 | 0.111 | fast_lora / bf16 | 1.423 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 32 | 0.083 | 0.112 | fast_lora / bf16 | 1.353 | yes |
| Qwen3.5-35B-A3B | shared | 64 | 64 | 0.095 | 0.117 | fast_lora / bf16 | 1.244 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 16 | 0.212 | 0.257 | fast_lora / bf16 | 1.220 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 32 | 0.215 | 0.252 | fast_lora / bf16 | 1.174 | yes |
| Qwen3.5-35B-A3B | shared | 512 | 64 | 0.238 | 0.269 | fast_lora / bf16 | 1.122 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 16 | 0.703 | 0.856 | fast_lora / bf16 | 1.220 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 32 | 0.684 | 0.779 | fast_lora / bf16 | 1.133 | yes |
| Qwen3.5-35B-A3B | shared | 2048 | 64 | 0.747 | 0.846 | fast_lora / bf16 | 1.132 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 16 | 3.340 | 5.069 | fast_lora / fp32+autocast | 1.498 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 32 | 3.578 | 4.990 | fast_lora / bf16 | 1.416 | yes |
| Qwen3.5-35B-A3B | shared | 8192 | 64 | 3.728 | 5.092 | fast_lora / bf16 | 1.432 | yes |


## Open gaps

None: every measured row is faster than the fastest Unsloth candidate. The thinnest
margins are the routed T=64 rows on the RTX 5090 (1.015-1.06x), where both sides
stream all expert weights twice (forward and dX) and sit near the DRAM floor.

## Changes made for this comparison

* **Shared expert step, small batches:** the shared backward is launch-bound
  at T<=64, so it now uses a dense SwiGLU backward without routing reads (no `zeros` /
  `ones` routing tensors in forward or backward), folds the LoRA scale into one
  multiply-and-round, writes BF16 factor gradients straight from the GEMMs (no FP32
  copies and casts) and runs gate/up as one batched GEMM (no `cat` / `stack`). GB10
  T=1 rank 64 (torch.profiler, eager): 42 -> 23 kernel launches per step, summed kernel
  time 72.5 -> 54.3 us.
* **RC alignment fix:** the RC's CuTe `moe_align_block_size` puts routes
  outside the local experts into trailing blocks of expert -1, which broke the binary
  search for per-expert block ranges in the grouped factor-gradient kernel (routed
  gradients up to 42% off at <=4096 routes with an invalid or remote route). The search
  now orders expert -1 blocks after every local expert.
* **Measurement:** an earlier shared run gave Unsloth's candidates one base-weight copy
  and ours another; on GB10 that made T=1 rank 64 read 0.93x although ours is faster in
  every isolated regime (back-to-back graph replays 55 vs 88 us, single replay after an
  L2 flush 127 vs 148 us, single warm replay 86 vs 92 us against `fast_lora`). The bench
  now aliases one copy for everyone.
