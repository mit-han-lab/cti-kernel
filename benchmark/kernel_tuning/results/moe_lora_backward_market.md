<!-- SPDX-License-Identifier: MIT -->
# Per-expert MoE LoRA training forward + backward vs torch (ML2)

Produced by `python -m benchmark.kernel_tuning.moe_lora_backward_bench` (9 paired,
interleaved samples per routed row, 15 per shared row; CUDA graphs for ours and the
BF16 torch loops, CUDA events for the FP32 reference; host process pinned to the
X925 cores on GB10). Each row times one complete training step of the layer:
forward with saved state plus backward for `x`, routing weights and every LoRA
factor (A and B of gate, up and down), with base weights frozen.

* **Ours:** `fused_experts_lora` / `shared_expert_lora` with trainable inputs:
  ML1 forward kernels saving `c1` and the shrink outputs, base backward GEMMs on
  transposed weight views (Triton, a CuTe rewrite pending), CuTe LoRA shrink/expand,
  CuTe SwiGLU + routing-weight backward and CuTe grouped factor-gradient GEMMs;
  the shared expert uses dense library GEMMs around the CuTe activation backward.
* **Torch BF16:** a PEFT-style composition (per-expert loops, BF16 GEMMs, FP32
  activation and routing arithmetic) differentiated by autograd. Neither
  vLLM 0.31.0 nor SGLang 0.5.22 ships a MoE LoRA backward, so this is the bar.
* **FP32 ref:** the FP32 differentiable reference composition (the oracle).

`Torch/ours` is the median paired ratio. `Max grad rel. err` is the largest
relative Frobenius error over all requested gradients against FP32 autograd of the
reference on FP32 copies (the GPU tests gate each gradient at 0.025). EP2 rows hold
half the experts locally, mask remote routes, and are local partial results.
Shapes: Qwen3.5-35B-A3B (E=256, H=2048, I=512, top-8, shared Is=512) and
Qwen3-30B-A3B (E=128, H=2048, I=768, top-8, no shared expert), BF16.
RTX PRO 6000 rows are pending (the device was not available during this round).

## RTX 5090

| Model | Kind | EP | T | Rank | Ours ms | Torch BF16 ms | FP32 ref ms | Torch/ours | Max grad rel. err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35-35b-a3b | routed | 1 | 1 | 16 | 0.345 | 1.897 | 9.9 | 5.49 | 0.0058 |
| qwen35-35b-a3b | routed | 1 | 1 | 32 | 0.559 | 4.132 | 11.7 | 7.41 | 0.0053 |
| qwen35-35b-a3b | routed | 1 | 1 | 64 | 1.099 | 8.474 | 19.4 | 7.71 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 64 | 16 | 2.089 | 67.837 | 317.9 | 32.49 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 64 | 32 | 2.353 | 122.558 | 341.5 | 52.41 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 64 | 64 | 3.033 | 240.913 | 455.5 | 79.49 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 512 | 16 | 2.535 | 83.921 | 369.2 | 33.12 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 512 | 32 | 2.807 | 144.079 | 414.4 | 51.35 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 512 | 64 | 3.634 | 280.482 | 550.0 | 77.16 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 16 | 3.516 | 80.229 | 542.7 | 22.83 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 32 | 3.833 | 140.403 | 598.4 | 36.66 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 64 | 4.722 | 278.034 | 748.1 | 58.91 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 16 | 9.349 | 116.438 | 1381.2 | 12.46 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 32 | 9.903 | 177.217 | 1464.7 | 17.89 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 64 | 11.347 | 317.284 | 1628.5 | 27.97 | 0.0050 |
| qwen35-35b-a3b | shared | 1 | 1 | 16 | 0.063 | 0.143 | 2.6 | 2.26 | 0.0067 |
| qwen35-35b-a3b | shared | 1 | 1 | 32 | 0.071 | 0.143 | 2.6 | 2.22 | 0.0055 |
| qwen35-35b-a3b | shared | 1 | 1 | 64 | 0.067 | 0.140 | 2.6 | 2.10 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 64 | 16 | 0.082 | 0.162 | 2.6 | 2.01 | 0.0052 |
| qwen35-35b-a3b | shared | 1 | 64 | 32 | 0.085 | 0.168 | 2.6 | 1.97 | 0.0053 |
| qwen35-35b-a3b | shared | 1 | 64 | 64 | 0.090 | 0.171 | 2.5 | 1.94 | 0.0054 |
| qwen35-35b-a3b | shared | 1 | 512 | 16 | 0.135 | 0.270 | 2.7 | 2.02 | 0.0051 |
| qwen35-35b-a3b | shared | 1 | 512 | 32 | 0.141 | 0.278 | 2.6 | 1.95 | 0.0051 |
| qwen35-35b-a3b | shared | 1 | 512 | 64 | 0.148 | 0.285 | 2.3 | 1.91 | 0.0050 |
| qwen35-35b-a3b | shared | 1 | 2048 | 16 | 0.282 | 0.624 | 2.5 | 2.23 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 2048 | 32 | 0.292 | 0.614 | 2.5 | 2.12 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 2048 | 64 | 0.312 | 0.632 | 2.5 | 2.03 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 16 | 0.894 | 2.695 | 8.0 | 3.02 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 32 | 0.929 | 2.715 | 8.1 | 2.94 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 64 | 0.972 | 2.786 | 8.1 | 2.87 | 0.0048 |
| qwen3-30b-a3b | routed | 1 | 1 | 16 | 0.313 | 1.567 | 17.8 | 5.02 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 1 | 32 | 0.451 | 2.332 | 18.1 | 5.17 | 0.0052 |
| qwen3-30b-a3b | routed | 1 | 1 | 64 | 0.823 | 4.326 | 18.4 | 5.25 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 64 | 16 | 1.759 | 34.714 | 198.3 | 19.74 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 64 | 32 | 1.859 | 44.612 | 202.3 | 24.00 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 64 | 64 | 2.338 | 77.259 | 218.7 | 33.04 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 16 | 1.940 | 28.808 | 220.2 | 14.87 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 32 | 2.105 | 39.462 | 208.9 | 18.77 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 64 | 2.608 | 71.702 | 232.5 | 27.49 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 16 | 3.381 | 33.659 | 310.4 | 9.96 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 32 | 3.592 | 44.282 | 312.7 | 12.33 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 64 | 4.239 | 77.157 | 353.6 | 18.20 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 16 | 11.460 | 63.046 | 829.0 | 5.50 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 32 | 11.899 | 71.501 | 847.7 | 6.01 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 64 | 13.119 | 104.844 | 893.5 | 7.99 | 0.0050 |
| qwen35-35b-a3b | routed | 2 | 64 | 16 | 1.154 | 28.187 | 185.3 | 24.43 | 0.0050 |
| qwen35-35b-a3b | routed | 2 | 64 | 64 | 1.577 | 60.356 | 183.4 | 38.25 | 0.0050 |
| qwen35-35b-a3b | routed | 2 | 2048 | 16 | 2.020 | 29.911 | 284.8 | 14.82 | 0.0050 |
| qwen35-35b-a3b | routed | 2 | 2048 | 64 | 2.635 | 69.908 | 312.7 | 26.53 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 64 | 16 | 0.928 | 15.013 | 91.3 | 16.25 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 64 | 64 | 1.235 | 21.712 | 95.4 | 17.58 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 2048 | 16 | 2.008 | 14.829 | 152.3 | 7.39 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 2048 | 64 | 2.471 | 22.159 | 159.4 | 8.96 | 0.0050 |

53 rows; torch/ours median 9.96, min 1.91; 53/53 faster; 0 rows over the 0.02 error gate.

## GB10

| Model | Kind | EP | T | Rank | Ours ms | Torch BF16 ms | FP32 ref ms | Torch/ours | Max grad rel. err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen35-35b-a3b | routed | 1 | 1 | 16 | 1.486 | 13.710 | 23.2 | 9.24 | 0.0065 |
| qwen35-35b-a3b | routed | 1 | 1 | 32 | 2.591 | 30.971 | 45.4 | 11.98 | 0.0071 |
| qwen35-35b-a3b | routed | 1 | 1 | 64 | 4.667 | 62.129 | 82.8 | 13.30 | 0.0051 |
| qwen35-35b-a3b | routed | 1 | 64 | 16 | 13.353 | 424.978 | 665.4 | 31.61 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 64 | 32 | 15.128 | 852.628 | 1195.5 | 56.05 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 64 | 64 | 18.422 | 1717.312 | 2226.7 | 93.21 | 0.0051 |
| qwen35-35b-a3b | routed | 1 | 512 | 16 | 17.372 | 516.052 | 1229.8 | 29.69 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 512 | 32 | 19.199 | 1000.702 | 1819.0 | 52.25 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 512 | 64 | 22.656 | 1982.441 | 2983.0 | 87.50 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 16 | 23.856 | 584.246 | 2628.1 | 24.42 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 32 | 25.763 | 1074.598 | 3229.6 | 41.65 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 2048 | 64 | 29.535 | 2048.913 | 4393.7 | 69.31 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 16 | 50.749 | 876.523 | 8052.3 | 17.29 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 32 | 52.848 | 1373.554 | 8615.5 | 25.92 | 0.0050 |
| qwen35-35b-a3b | routed | 1 | 8192 | 64 | 58.596 | 2353.496 | 9777.8 | 40.23 | 0.0050 |
| qwen35-35b-a3b | shared | 1 | 1 | 16 | 0.089 | 0.176 | 3.5 | 2.07 | 0.0061 |
| qwen35-35b-a3b | shared | 1 | 1 | 32 | 0.096 | 0.173 | 2.8 | 1.97 | 0.0045 |
| qwen35-35b-a3b | shared | 1 | 1 | 64 | 0.119 | 0.183 | 2.7 | 1.71 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 64 | 16 | 0.155 | 0.260 | 2.6 | 1.77 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 64 | 32 | 0.162 | 0.275 | 2.7 | 1.74 | 0.0047 |
| qwen35-35b-a3b | shared | 1 | 64 | 64 | 0.164 | 0.264 | 2.5 | 1.72 | 0.0047 |
| qwen35-35b-a3b | shared | 1 | 512 | 16 | 0.271 | 0.866 | 3.5 | 3.18 | 0.0049 |
| qwen35-35b-a3b | shared | 1 | 512 | 32 | 0.271 | 0.853 | 3.3 | 3.18 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 512 | 64 | 0.296 | 0.871 | 3.2 | 2.95 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 2048 | 16 | 0.734 | 4.219 | 12.1 | 5.74 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 2048 | 32 | 0.750 | 4.252 | 12.4 | 5.61 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 2048 | 64 | 0.809 | 4.332 | 12.3 | 5.36 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 16 | 3.367 | 22.673 | 54.3 | 6.63 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 32 | 3.415 | 23.046 | 54.5 | 6.57 | 0.0048 |
| qwen35-35b-a3b | shared | 1 | 8192 | 64 | 3.615 | 23.116 | 55.2 | 6.22 | 0.0048 |
| qwen3-30b-a3b | routed | 1 | 1 | 16 | 1.331 | 8.939 | 17.7 | 6.71 | 0.0051 |
| qwen3-30b-a3b | routed | 1 | 1 | 32 | 2.010 | 17.870 | 28.9 | 8.81 | 0.0052 |
| qwen3-30b-a3b | routed | 1 | 1 | 64 | 3.255 | 31.792 | 45.3 | 9.82 | 0.0056 |
| qwen3-30b-a3b | routed | 1 | 64 | 16 | 11.461 | 142.402 | 265.1 | 12.45 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 64 | 32 | 12.296 | 285.032 | 441.6 | 23.17 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 64 | 64 | 14.295 | 563.372 | 773.6 | 39.45 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 16 | 13.575 | 147.205 | 562.6 | 10.75 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 32 | 14.409 | 289.236 | 737.2 | 20.05 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 512 | 64 | 16.573 | 574.712 | 1068.6 | 34.51 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 16 | 20.967 | 196.305 | 1455.0 | 9.36 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 32 | 22.203 | 333.185 | 1612.8 | 15.09 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 2048 | 64 | 24.935 | 625.013 | 1925.0 | 25.07 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 16 | 53.509 | 399.934 | 4934.4 | 7.45 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 32 | 55.271 | 540.634 | 5099.2 | 9.75 | 0.0050 |
| qwen3-30b-a3b | routed | 1 | 8192 | 64 | 59.861 | 821.282 | 5426.8 | 13.74 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 64 | 16 | 5.756 | 41.471 | 87.5 | 7.21 | 0.0051 |
| qwen3-30b-a3b | routed | 2 | 64 | 64 | 7.147 | 143.346 | 224.6 | 20.01 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 2048 | 16 | 11.772 | 67.070 | 694.7 | 5.69 | 0.0050 |
| qwen3-30b-a3b | routed | 2 | 2048 | 64 | 13.477 | 166.757 | 821.0 | 12.32 | 0.0050 |

49 rows; torch/ours median 10.75, min 1.71; 49/49 faster; 0 rows over the 0.02 error gate.
