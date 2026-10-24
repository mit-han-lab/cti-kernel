<!-- SPDX-License-Identifier: MIT -->
# Shared-expert LoRA: profiling evidence for the dense kernels

Shape: Qwen3.5-35B-A3B shared expert (H=2048, Is=512, BF16). gate/up = K 2048,
N 2x512; down = K 512, N 2048. RTX 5090.

## Per-kernel device time before the dense kernels (torch profiler, eager, warm L2)

Per-row sums over both projections; vLLM 0.31.0 punica shrink/expand for reference.

| row | ours: library shrink (+splitK reduce) | ours: BlockKernel expand | vLLM shrink | vLLM expand |
|---|---|---|---|---|
| T=512, r=16 | 7.4 us | 8.4 us | 5.3 us | 6.3 us |
| T=512, r=64 | 8.1 us | 15.5 us | 8.6 us | 7.6 us |

The expand stage was the gap: about 2x vLLM's expand at rank 64.

## Nsight Compute, T=512, r=64 (base clocks, `--cache-control all`)

| kernel | grid | duration | achieved occupancy | top stall |
|---|---|---|---|---|
| BlockKernel expand, gate/up (old) | 256 | 12.67 us | 12.6% | short scoreboard, 66% of 15.8 cycles/issue |
| BlockKernel expand, down (old) | 352 | 15.26 us | 16.3% | short scoreboard, 50% of 12.2 cycles/issue |
| DenseExpand, gate/up (new, cfg 32/8/64/64/2) | 128 | 5.15 us | 8.3% | short scoreboard, 48% of 6.5 cycles/issue |
| DenseExpand, down (new, same cfg) | 256 | 6.78 us | 12.9% | short scoreboard, 48% of 8.3 cycles/issue |
| DenseShrink, gate/up (new) | 128 | 9.34 us (cold L2) | 8.3% | - |
| DenseShrink, down (new) | 128 | 4.16 us | 8.3% | short scoreboard, 45% of 10.1 cycles/issue |

The old expand walks every N tile of its row block serially, with two CTA barriers
per tile and a scalar shared-memory epilogue. The new expand gives each CTA one
[BM, BN] tile and issues its B/cache prefetch before the PDL wait. It issues all
z loads before using them, which cuts the expand time 2.2-2.5x.

## Shrink/expand split after the rework (torch profiler, eager)

Before the rework, the dense expand summed split-K partials in a loop whose
loads were dependent across iterations. Rank 128, T=512, down: expand 10.9 us,
against a ~2.3 us output read-modify-write floor. The current kernels reduce splits
with vector `red.global.add` into compact FP32 z, and the expand loads z with all
loads in flight. The probe results that produced `_shared_policy.SHARED_TABLE`
(`benchmark/kernel_tuning/moe_lora_shared_probe.py`) record per-projection times
for every mode and config.
