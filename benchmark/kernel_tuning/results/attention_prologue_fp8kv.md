# N4: FP8 KV cache prologue — engine-regime pairing

Commit d2948a5 (kernel unchanged since 40760bd). `python -m benchmark.kernel_tuning.attention_prologue_engine
--variant {hd128,hd256} --kv-dtype {fp8_e4m3,fp8_e5m2} --pairs 20`: 8 layers per graph, model-sized
filler, time = in-graph kernel spans of the prologue per layer (see the script's docstring). hd128 =
Qwen3-8B / DFlash drafts (32 Q / 8 KV heads), hd256 = Qwen3.5-9B gated full attention (16 / 4).
Per-tensor scales k_scale 0.0478, v_scale 0.0213 (FP32 device tensors for ours and vLLM, Python floats
for SGLang).

Markets, as each dispatches them with an FP8 KV cache:
- SGLang 72730d6: its BF16 prologue kernels (fused_inplace_qknorm + JIT RoPE, or the fused Triton
  norm+RoPE+gate kernel for gated hd256), then MHATokenToKVPool.set_kv_buffer: in-place BF16 division by
  the scales, `.to(fp8)`, uint8 view, JIT store_cache (6-7 kernels). `SGLang no-scale` is the same
  without KV scales (casts only, 4-5 kernels), the default for checkpoints without KV scales.
- vLLM 0.31.0: Inductor-compiled native RMSNorm/RoPE (or the fused gated kernel) + reshape_and_cache_flash
  with kv_cache_dtype "fp8" / "fp8_e5m2" and the scale tensors (2-3 kernels).
- Ours: one CuTe kernel.

Speedup = fastest market / ours with an exact 95% median CI over 20 interleaved pairs; ε = p95 self-pair
noise; verdict "faster" needs the CI above 1 + ε (GB10 rule: CI_low > 1). Correctness gates per stack and
layer: Q as in BF16 mode; V bitwise equal to fp8(V / v_scale); ours' K within 1 FP8 ulp of the FP64
oracle's quantization (measured max 0.5 ulp, i.e. the oracle's rounding); market K (rounded to BF16
before quantizing) within 2 FP8 ulps plus 4 BF16 ulps of the conditioned bound. GPU tests
(tests/gpu/test_attention_prologue.py -m "not exhaustive") at b8371df, which adds every dense geometry
(Qwen3, Qwen3.5, Qwen3.8 TP1/TP2, DFlash drafts) and capped model-max-context bands: 126 passed on the
RTX 5090 and GB10 (PRO 6000: see its section). The extension sections at the end add Qwen3.8-27B
(hd256-27b, 24 Q / 4 KV heads, gated) and T=4096 for every variant, plus the RTX PRO 6000.

### RTX 5090, fp8_e4m3

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd128 | 1 | 1.12 | 5.92 | 4.34 | 4.26 | vllm | 3.805 [3.789, 3.814] | 0.014 | faster |
| hd128 | 2 | 1.51 | 9.95 | 6.80 | 4.74 | vllm | 3.089 [3.058, 3.282] | 0.034 | faster |
| hd128 | 3 | 1.52 | 10.12 | 6.93 | 4.59 | vllm | 3.024 [2.992, 3.199] | 0.031 | faster |
| hd128 | 4 | 1.55 | 10.27 | 7.00 | 4.80 | vllm | 3.090 [3.028, 3.212] | 0.033 | faster |
| hd128 | 16 | 1.61 | 10.52 | 7.21 | 5.01 | vllm | 3.040 [2.975, 3.274] | 0.041 | faster |
| hd128 | 32 | 1.17 | 10.33 | 7.03 | 5.93 | vllm | 5.113 [4.937, 5.187] | 0.081 | faster |
| hd128 | 48 | 1.17 | 10.39 | 7.12 | 5.97 | vllm | 5.052 [4.986, 5.269] | 0.058 | faster |
| hd128 | 64 | 1.23 | 10.61 | 7.32 | 5.70 | vllm | 4.557 [4.482, 4.875] | 0.043 | faster |
| hd128 | 128 | 1.92 | 11.93 | 8.49 | 5.74 | vllm | 2.978 [2.944, 3.028] | 0.052 | faster |
| hd128 | 256 | 2.58 | 13.66 | 10.25 | 6.51 | vllm | 2.526 [2.508, 2.551] | 0.034 | faster |
| hd128 | 512 | 3.59 | 16.23 | 12.51 | 10.97 | vllm | 3.062 [3.044, 3.078] | 0.019 | faster |
| hd128 | 1024 | 5.65 | 21.68 | 16.95 | 18.12 | sglang_noscale | 2.998 [2.980, 3.025] | 0.077 | faster |
| hd256 | 1 | 1.24 | 5.50 | 3.88 | 3.25 | vllm | 2.627 [2.621, 2.631] | 0.020 | faster |
| hd256 | 2 | 1.70 | 8.17 | 5.63 | 4.25 | vllm | 2.488 [2.481, 2.499] | 0.023 | faster |
| hd256 | 3 | 1.67 | 8.32 | 5.73 | 4.14 | vllm | 2.472 [2.456, 2.493] | 0.032 | faster |
| hd256 | 4 | 1.69 | 8.32 | 5.73 | 4.12 | vllm | 2.455 [2.434, 2.467] | 0.019 | faster |
| hd256 | 16 | 1.74 | 8.56 | 5.89 | 4.35 | vllm | 2.500 [2.474, 2.529] | 0.027 | faster |
| hd256 | 32 | 2.02 | 8.78 | 6.12 | 4.41 | vllm | 2.182 [2.169, 2.193] | 0.027 | faster |
| hd256 | 48 | 2.08 | 9.05 | 6.40 | 4.47 | vllm | 2.140 [2.134, 2.157] | 0.065 | faster |
| hd256 | 64 | 2.16 | 9.36 | 6.69 | 4.72 | vllm | 2.181 [2.165, 2.198] | 0.037 | faster |
| hd256 | 128 | 2.51 | 11.24 | 8.23 | 6.21 | vllm | 2.479 [2.447, 2.493] | 0.028 | faster |
| hd256 | 256 | 2.38 | 13.14 | 10.28 | 7.22 | vllm | 3.028 [3.003, 3.056] | 0.032 | faster |
| hd256 | 512 | 3.55 | 18.76 | 15.76 | 11.54 | vllm | 3.248 [3.238, 3.258] | 0.017 | faster |
| hd256 | 1024 | 6.12 | 30.36 | 26.47 | 19.31 | vllm | 3.149 [3.140, 3.162] | 0.008 | faster |

### GB10 (X925-pinned), fp8_e4m3

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd128 | 1 | 1.32 | 7.51 | 5.51 | 4.80 | vllm | 3.644 [3.622, 3.665] | 0.084 | faster |
| hd128 | 2 | 1.89 | 13.67 | 9.41 | 6.32 | vllm | 3.343 [3.328, 3.363] | 0.128 | faster |
| hd128 | 3 | 1.92 | 13.72 | 9.54 | 6.44 | vllm | 3.353 [3.329, 3.368] | 0.313 | faster |
| hd128 | 4 | 1.86 | 13.62 | 9.46 | 6.42 | vllm | 3.450 [3.439, 3.463] | 0.136 | faster |
| hd128 | 16 | 1.99 | 13.88 | 9.59 | 6.55 | vllm | 3.284 [3.269, 3.306] | 0.463 | faster |
| hd128 | 32 | 2.20 | 14.59 | 10.47 | 6.94 | vllm | 3.152 [3.134, 3.439] | 0.374 | faster |
| hd128 | 48 | 2.07 | 14.66 | 10.37 | 7.12 | vllm | 3.437 [3.400, 3.463] | 0.961 | faster |
| hd128 | 64 | 2.22 | 12.84 | 9.33 | 6.39 | vllm | 2.885 [2.857, 3.154] | 0.382 | faster |
| hd128 | 128 | 3.66 | 17.87 | 13.34 | 11.40 | vllm | 3.109 [3.078, 3.134] | 0.432 | faster |
| hd128 | 256 | 5.79 | 22.64 | 17.55 | 16.63 | vllm | 2.875 [2.854, 2.882] | 0.189 | faster |
| hd128 | 512 | 12.86 | 36.19 | 28.67 | 36.56 | sglang_noscale | 2.237 [2.175, 2.435] | 0.144 | faster |
| hd128 | 1024 | 52.74 | 149.65 | 134.16 | 93.70 | vllm | 1.782 [1.767, 1.806] | 0.080 | faster |
| hd256 | 1 | 1.51 | 6.76 | 4.77 | 3.79 | vllm | 2.511 [2.497, 2.516] | 0.085 | faster |
| hd256 | 2 | 2.12 | 11.29 | 7.66 | 5.42 | vllm | 2.549 [2.543, 2.561] | 0.229 | faster |
| hd256 | 3 | 2.11 | 11.27 | 7.64 | 5.32 | vllm | 2.526 [2.511, 2.535] | 0.301 | faster |
| hd256 | 4 | 2.09 | 11.31 | 7.72 | 5.41 | vllm | 2.584 [2.569, 2.594] | 0.359 | faster |
| hd256 | 16 | 2.40 | 11.60 | 7.94 | 5.70 | vllm | 2.373 [2.357, 2.383] | 0.181 | faster |
| hd256 | 32 | 3.14 | 12.51 | 8.94 | 6.84 | vllm | 2.182 [2.141, 2.208] | 0.718 | faster |
| hd256 | 48 | 3.28 | 13.00 | 9.41 | 7.35 | vllm | 2.254 [2.216, 2.407] | 0.323 | faster |
| hd256 | 64 | 3.46 | 11.57 | 8.64 | 7.55 | vllm | 2.178 [2.164, 2.200] | 0.411 | faster |
| hd256 | 128 | 4.06 | 17.12 | 13.08 | 12.25 | vllm | 3.036 [3.017, 3.078] | 0.226 | faster |
| hd256 | 256 | 7.23 | 28.38 | 23.11 | 22.87 | vllm | 3.184 [3.145, 3.205] | 0.362 | faster |
| hd256 | 512 | 20.73 | 75.39 | 69.90 | 71.15 | sglang_noscale | 3.452 [3.361, 3.554] | 0.156 | faster |
| hd256 | 1024 | 67.91 | 201.93 | 190.59 | 184.97 | vllm | 2.722 [2.690, 2.747] | 0.055 | faster |

### GB10 (X925-pinned), fp8_e5m2

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd128 | 1 | 1.33 | 7.91 | 5.94 | 4.91 | vllm | 3.707 [3.685, 3.729] | 0.070 | faster |
| hd128 | 2 | 1.91 | 14.01 | 9.81 | 6.25 | vllm | 3.274 [3.262, 3.284] | 0.274 | faster |
| hd128 | 3 | 1.93 | 13.91 | 9.77 | 6.42 | vllm | 3.330 [3.317, 3.351] | 0.226 | faster |
| hd128 | 4 | 1.85 | 14.03 | 9.80 | 6.42 | vllm | 3.459 [3.443, 3.482] | 0.264 | faster |
| hd128 | 16 | 2.03 | 14.17 | 9.99 | 6.57 | vllm | 3.246 [3.206, 3.260] | 0.232 | faster |
| hd128 | 32 | 2.23 | 14.84 | 10.59 | 7.02 | vllm | 3.147 [3.112, 3.164] | 0.300 | faster |
| hd128 | 48 | 2.08 | 14.92 | 10.87 | 7.14 | vllm | 3.432 [3.333, 3.447] | 0.397 | faster |
| hd128 | 64 | 2.12 | 13.12 | 9.70 | 6.45 | vllm | 3.039 [3.008, 3.057] | 0.263 | faster |
| hd128 | 128 | 3.64 | 18.06 | 13.79 | 11.40 | vllm | 3.138 [3.123, 3.161] | 0.217 | faster |
| hd128 | 256 | 5.60 | 22.38 | 17.79 | 16.71 | vllm | 3.000 [2.972, 3.073] | 0.247 | faster |
| hd128 | 512 | 11.00 | 35.79 | 28.32 | 33.82 | sglang_noscale | 2.576 [2.371, 2.834] | 0.259 | faster |
| hd128 | 1024 | 51.01 | 150.87 | 135.25 | 93.50 | vllm | 1.831 [1.818, 1.851] | 0.090 | faster |
| hd256 | 1 | 1.52 | 7.28 | 5.26 | 3.83 | vllm | 2.514 [2.510, 2.520] | 0.128 | faster |
| hd256 | 2 | 2.11 | 11.60 | 8.08 | 5.31 | vllm | 2.513 [2.497, 2.531] | 0.254 | faster |
| hd256 | 3 | 2.11 | 11.74 | 8.24 | 5.40 | vllm | 2.562 [2.543, 2.580] | 0.331 | faster |
| hd256 | 4 | 2.17 | 11.69 | 8.22 | 5.37 | vllm | 2.477 [2.452, 2.494] | 0.328 | faster |
| hd256 | 16 | 2.39 | 11.93 | 8.48 | 5.65 | vllm | 2.369 [2.352, 2.384] | 0.250 | faster |
| hd256 | 32 | 3.30 | 13.14 | 9.57 | 6.89 | vllm | 2.099 [2.062, 2.110] | 0.423 | faster |
| hd256 | 48 | 3.33 | 13.46 | 9.91 | 7.20 | vllm | 2.195 [2.122, 2.215] | 0.478 | faster |
| hd256 | 64 | 3.71 | 12.01 | 9.14 | 7.58 | vllm | 2.050 [2.030, 2.063] | 0.191 | faster |
| hd256 | 128 | 3.86 | 17.29 | 13.48 | 12.26 | vllm | 3.184 [3.155, 3.252] | 0.253 | faster |
| hd256 | 256 | 7.64 | 27.97 | 23.86 | 23.01 | vllm | 3.020 [2.984, 3.047] | 0.050 | faster |
| hd256 | 512 | 20.16 | 75.30 | 69.48 | 70.13 | sglang_noscale | 3.458 [3.396, 3.627] | 0.145 | faster |
| hd256 | 1024 | 67.09 | 200.93 | 189.43 | 182.71 | vllm | 2.705 [2.692, 2.749] | 0.024 | faster |

## Extension: Qwen3.8-27B and T=4096 (commit b1bbf8a, kernel unchanged)

Same protocol and gates, fp8_e4m3. `--variant hd256-27b` = Qwen3.8-27B gated full attention at TP1
(24 Q / 4 KV heads, hd256, hidden 5120); T=4096 rows use `--rows 4096`.

### RTX 5090, fp8_e4m3

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd256-27b | 1 | 1.54 | 6.47 | 4.77 | 4.12 | vllm | 2.668 [2.648, 2.687] | 0.043 | faster |
| hd256-27b | 2 | 1.02 | 8.13 | 5.57 | 4.29 | vllm | 4.224 [4.188, 4.257] | 0.032 | faster |
| hd256-27b | 3 | 0.99 | 8.21 | 5.60 | 4.29 | vllm | 4.333 [4.276, 4.376] | 0.052 | faster |
| hd256-27b | 4 | 1.01 | 8.26 | 5.62 | 4.29 | vllm | 4.268 [4.215, 4.363] | 0.054 | faster |
| hd256-27b | 16 | 1.63 | 8.63 | 5.88 | 4.09 | vllm | 2.509 [2.472, 2.545] | 0.032 | faster |
| hd256-27b | 32 | 1.96 | 8.99 | 6.25 | 4.36 | vllm | 2.222 [2.210, 2.232] | 0.025 | faster |
| hd256-27b | 48 | 1.95 | 9.22 | 6.49 | 4.63 | vllm | 2.372 [2.356, 2.386] | 0.021 | faster |
| hd256-27b | 64 | 2.12 | 9.60 | 6.88 | 4.86 | vllm | 2.290 [2.285, 2.306] | 0.028 | faster |
| hd256-27b | 128 | 2.40 | 11.77 | 8.98 | 6.36 | vllm | 2.641 [2.636, 2.651] | 0.018 | faster |
| hd256-27b | 256 | 3.18 | 15.32 | 12.46 | 8.46 | vllm | 2.659 [2.653, 2.662] | 0.007 | faster |
| hd256-27b | 512 | 4.89 | 22.82 | 19.75 | 13.88 | vllm | 2.841 [2.828, 2.853] | 0.009 | faster |
| hd256-27b | 1024 | 8.61 | 39.35 | 35.28 | 26.34 | vllm | 3.057 [3.050, 3.069] | 0.025 | faster |
| hd256-27b | 4096 | 51.39 | 161.56 | 151.57 | 153.51 | sglang_noscale | 2.948 [2.936, 2.959] | 0.015 | faster |
| hd128 | 4096 | 28.66 | 61.92 | 49.41 | 76.85 | sglang_noscale | 1.725 [1.719, 1.736] | 0.012 | faster |
| hd256 | 4096 | 30.33 | 112.46 | 102.21 | 100.65 | vllm | 3.319 [3.312, 3.326] | 0.007 | faster |

### GB10 (X925-pinned), fp8_e4m3

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd256-27b | 1 | 2.40 | 9.81 | 7.20 | 5.64 | vllm | 2.349 [2.329, 2.366] | 0.251 | faster |
| hd256-27b | 2 | 2.33 | 11.68 | 7.99 | 5.61 | vllm | 2.410 [2.389, 2.420] | 0.161 | faster |
| hd256-27b | 3 | 2.57 | 11.72 | 8.00 | 5.64 | vllm | 2.194 [2.143, 2.215] | 0.153 | faster |
| hd256-27b | 4 | 1.98 | 9.43 | 6.44 | 4.85 | vllm | 2.460 [2.400, 2.479] | 0.291 | faster |
| hd256-27b | 16 | 3.25 | 12.25 | 8.55 | 6.30 | vllm | 1.939 [1.926, 1.949] | 0.195 | faster |
| hd256-27b | 32 | 3.64 | 13.70 | 10.09 | 8.44 | vllm | 2.315 [2.303, 2.343] | 0.267 | faster |
| hd256-27b | 48 | 4.88 | 15.00 | 11.37 | 9.47 | vllm | 1.942 [1.928, 1.947] | 0.245 | faster |
| hd256-27b | 64 | 6.06 | 14.16 | 11.38 | 10.31 | vllm | 1.699 [1.692, 1.710] | 0.145 | faster |
| hd256-27b | 128 | 4.99 | 20.75 | 16.34 | 15.86 | vllm | 3.223 [2.954, 3.304] | 0.425 | faster |
| hd256-27b | 256 | 11.46 | 44.21 | 39.18 | 41.07 | sglang_noscale | 3.520 [3.253, 3.665] | 0.335 | faster |
| hd256-27b | 512 | 36.35 | 127.71 | 121.56 | 117.41 | vllm | 3.146 [3.081, 3.252] | 0.137 | faster |
| hd256-27b | 1024 | 98.77 | 279.18 | 263.11 | 258.08 | vllm | 2.619 [2.604, 2.633] | 0.033 | faster |
| hd256-27b | 4096 | 548.77 | 1198.02 | 1119.07 | 1088.24 | vllm | 1.985 [1.978, 1.986] | 0.010 | faster |
| hd128 | 4096 | 402.69 | 1012.51 | 910.89 | 468.62 | vllm | 1.161 [1.157, 1.164] | 0.022 | faster |
| hd256 | 4096 | 403.73 | 880.63 | 804.91 | 780.94 | vllm | 1.933 [1.929, 1.948] | 0.028 | faster |

### RTX PRO 6000, fp8_e4m3 (hd128, hd256, hd256-27b, all rows)

| variant | T | ours us | SGLang us | SGLang no-scale us | vLLM us | market | ours/market speedup [95% CI] | ε | verdict |
|---|---:|---:|---:|---:|---:|---|---|---:|---|
| hd128 | 1 | 2.37 | 11.24 | 8.47 | 6.92 | vllm | 2.917 [2.905, 2.936] | 0.029 | faster |
| hd128 | 2 | 1.84 | 14.31 | 9.73 | 6.87 | vllm | 3.754 [3.714, 3.813] | 0.029 | faster |
| hd128 | 3 | 1.80 | 14.65 | 9.86 | 7.04 | vllm | 3.884 [3.863, 3.937] | 0.034 | faster |
| hd128 | 4 | 2.24 | 15.11 | 10.37 | 6.85 | vllm | 3.084 [3.025, 3.166] | 0.025 | faster |
| hd128 | 16 | 2.33 | 15.65 | 10.75 | 7.18 | vllm | 3.105 [3.057, 3.149] | 0.030 | faster |
| hd128 | 32 | 1.52 | 14.83 | 10.03 | 8.11 | vllm | 5.304 [5.224, 5.422] | 0.051 | faster |
| hd128 | 48 | 2.44 | 15.93 | 11.04 | 7.30 | vllm | 2.992 [2.946, 3.088] | 0.035 | faster |
| hd128 | 64 | 2.48 | 16.02 | 11.17 | 7.38 | vllm | 2.997 [2.958, 3.110] | 0.038 | faster |
| hd128 | 128 | 1.62 | 16.32 | 11.38 | 8.49 | vllm | 5.248 [5.162, 5.397] | 0.041 | faster |
| hd128 | 256 | 3.15 | 19.63 | 14.76 | 9.21 | vllm | 2.881 [2.861, 2.991] | 0.034 | faster |
| hd128 | 512 | 4.16 | 20.83 | 16.43 | 14.06 | vllm | 3.302 [3.272, 3.328] | 0.044 | faster |
| hd128 | 1024 | 6.15 | 25.55 | 19.85 | 20.57 | sglang_noscale | 3.228 [3.213, 3.239] | 0.028 | faster |
| hd128 | 4096 | 28.30 | 61.75 | 47.69 | 70.29 | sglang_noscale | 1.686 [1.675, 1.690] | 0.002 | faster |
| hd256 | 1 | 1.48 | 10.09 | 7.32 | 5.81 | vllm | 3.897 [3.865, 3.922] | 0.035 | faster |
| hd256 | 2 | 2.51 | 12.01 | 8.11 | 5.72 | vllm | 2.277 [2.261, 2.286] | 0.021 | faster |
| hd256 | 3 | 2.52 | 12.14 | 8.21 | 5.68 | vllm | 2.263 [2.242, 2.277] | 0.024 | faster |
| hd256 | 4 | 2.54 | 12.13 | 8.16 | 5.62 | vllm | 2.221 [2.197, 2.237] | 0.028 | faster |
| hd256 | 16 | 2.47 | 12.70 | 8.70 | 5.93 | vllm | 2.403 [2.396, 2.426] | 0.026 | faster |
| hd256 | 32 | 2.73 | 12.76 | 8.62 | 5.82 | vllm | 2.129 [2.115, 2.155] | 0.042 | faster |
| hd256 | 48 | 2.89 | 13.16 | 9.14 | 6.07 | vllm | 2.099 [2.076, 2.111] | 0.021 | faster |
| hd256 | 64 | 2.93 | 13.51 | 9.51 | 6.33 | vllm | 2.167 [2.135, 2.180] | 0.018 | faster |
| hd256 | 128 | 3.35 | 15.00 | 10.96 | 7.36 | vllm | 2.186 [2.175, 2.206] | 0.020 | faster |
| hd256 | 256 | 3.34 | 18.14 | 14.07 | 9.60 | vllm | 2.868 [2.836, 2.875] | 0.045 | faster |
| hd256 | 512 | 4.16 | 23.34 | 19.38 | 13.74 | vllm | 3.308 [3.297, 3.319] | 0.016 | faster |
| hd256 | 1024 | 5.99 | 33.20 | 28.95 | 20.57 | vllm | 3.437 [3.420, 3.443] | 0.012 | faster |
| hd256 | 4096 | 35.43 | 126.53 | 114.03 | 108.19 | vllm | 3.055 [3.052, 3.063] | 0.008 | faster |
| hd256-27b | 1 | 2.39 | 9.53 | 6.80 | 5.11 | vllm | 2.139 [2.123, 2.153] | 0.022 | faster |
| hd256-27b | 2 | 2.65 | 12.13 | 8.28 | 5.71 | vllm | 2.152 [2.144, 2.158] | 0.037 | faster |
| hd256-27b | 3 | 2.61 | 12.15 | 8.25 | 5.47 | vllm | 2.094 [2.076, 2.115] | 0.036 | faster |
| hd256-27b | 4 | 2.62 | 12.04 | 8.20 | 5.42 | vllm | 2.064 [2.045, 2.076] | 0.019 | faster |
| hd256-27b | 16 | 2.38 | 12.35 | 8.44 | 5.62 | vllm | 2.356 [2.341, 2.376] | 0.030 | faster |
| hd256-27b | 32 | 2.52 | 13.11 | 9.05 | 6.73 | vllm | 2.653 [2.641, 2.679] | 0.034 | faster |
| hd256-27b | 48 | 2.80 | 13.13 | 9.28 | 6.22 | vllm | 2.227 [2.197, 2.234] | 0.023 | faster |
| hd256-27b | 64 | 2.96 | 13.68 | 9.76 | 6.26 | vllm | 2.113 [2.105, 2.130] | 0.031 | faster |
| hd256-27b | 128 | 3.33 | 15.20 | 11.33 | 7.70 | vllm | 2.303 [2.287, 2.313] | 0.039 | faster |
| hd256-27b | 256 | 3.63 | 19.21 | 15.31 | 10.11 | vllm | 2.788 [2.775, 2.800] | 0.013 | faster |
| hd256-27b | 512 | 4.95 | 27.06 | 23.02 | 16.02 | vllm | 3.239 [3.233, 3.248] | 0.014 | faster |
| hd256-27b | 1024 | 8.26 | 43.82 | 39.28 | 25.84 | vllm | 3.134 [3.126, 3.142] | 0.010 | faster |
| hd256-27b | 4096 | 54.35 | 166.70 | 155.94 | 147.75 | vllm | 2.718 [2.715, 2.726] | 0.010 | faster |
