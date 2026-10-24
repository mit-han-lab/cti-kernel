# Frozen BF16 small-M linear measurements

Generated from supplied RESULT logs; no GPU work was run to produce this report.
Missing measurements remain pending. Error rows are retained as errors, never
replaced by inferred timings. See operations for integration
and kernel design for register layouts and tuning.

## Scope, baseline and protocol

Qwen3.5-9B, Qwen3.5-27B and the dense/shared portion of Qwen3.5-35B-A3B,
TP=1, BF16 text decode. Routed experts, vision and MTP are excluded; router and
scalar gate rows explicitly use BF16. Market and acceptance use M=1,2,3,4,6,8;
probe winner measurements include M=1..8. Fused GDN qkvzba and split qkvz plus
ba are alternative layouts; never add both to a model total.

cuBLAS is biasless `torch.nn.functional.linear`, the ordinary unquantized CUDA
BF16 path in the supplied SGLang 0.5.21 / 72730d6 source export. The export label
is not independent verification of a pristine revision. cuBLASLt uses the same
call with the preferred BLAS backend changed during warmup and capture; a
preference does not prove which vendor kernel actually ran. QuACK 0.6.5 calls
`gemm(x, weight.T, tuned=False)` without benchmark-time autotuning.

Cold timing reads an eviction buffer at least twice L2 before the start event.
Each CUDA graph captures one call; pairs alternate replay order. Compilation,
FP64 correctness references and profiling are outside timing. Market defaults
to 20 pairs, five single replays per sample (median); acceptance uses 20 pairs,
one replay per sample and 20 baseline self-pairs. Epsilon is linear p95 of
`abs(self_A/self_B - 1)`. Ratios are medians of paired ratios, so may differ
from ratios of the displayed median times.

The fixed market hook measures shipped auto dispatch, including cuBLAS fallback,
and records `actual_provider` and `logical_kernels_per_call`. Older market logs
raised `RuntimeError` whenever dispatch correctly chose cuBLAS. All 15 RTX 5090,
13 PRO 6000 and 44 GB10 original error rows matched table holes; their timings
remain pending until rerun. GPU tests permit these fallbacks, explaining why
those tests passed. This was not evidence of a kernel accuracy failure.
Warmup already happened before capture. The market now shares the probe's
K-aware accuracy gate and records exception type plus message, capped at 300
characters. Legacy successful ours rows were CuTe-only.

Correctness uses FP64 reference R and condition C=sum(abs(x*w)), with three
independent gates: maximum ULP error <= max(cuBLAS ULP error, 0.5)+0.5; maximum
condition-normalized error <= 1.5*cuBLAS normalized error +
2^-9*max(abs(R)/C)+2*eps32; RMS error <= 1.05*cuBLAS RMS, with no additive RMS
tolerance. The elementwise ULP is floored at
`2*eps32*C*max(1,sqrt(K/1024))`. This is a statistical accumulation allowance,
not a worst-case error guarantee. Missing-chunk and swapped-fragment CPU
regressions still fail. Absolute errors are diagnostic metrics.

The table generator admits correct cold configurations with median paired
speedup >=1.03 (3 %). Any incorrect observation vetoes that configuration/M.
A configuration is dispatched only if the current acceptance round measured that exact configuration (Config fields, M bucket, CTA threads) at paired speedup >=1.03 for that device/shape/M; evidence for another configuration never qualifies it, and rows without such acceptance stay on cuBLAS.
Identical adjacent measured M values coalesce; holes are preserved. Probe
selection is not independent acceptance. Acceptance PASS requires correctness
and paired speedup >=1-epsilon. Displayed kernels/call in acceptance are actual
profiled CUDA activities; the operation's cuBLAS property is one logical call.

## SOL and utilization

This report uses the SOL ceilings' **read-only DRAM ceilings**: RTX 5090
1661 GB/s, RTX PRO 6000 1543 GB/s, GB10 199 GB/s (decimal).
`SOL_us = 2*(N*K + M*K + M*N)/(read_GBps*1000)`;
utilization = SOL_us/measured_us, unclipped. All four providers use the same
reference. The benchmark logs may use max(read,copy), including 219 GB/s on
GB10; the report deliberately recomputes SOL and utilization from read-only
ceilings for consistency with this comparison. This compulsory-byte estimate
excludes scratch, repeated reads, compute cost and launch floors; it is not a
measured traffic counter or the full multi-resource SOL estimator.

## Engine evidence

The supplied RTX 5090 Qwen3.5-9B engine profile has complete decode graphs only
at B=1 and B=4; B=2 and B=3 were not measured. CTI GEMM time rises from
9.784697 ms to 10.599599 ms: **+0.814902 ms**, mostly MLP down
(**+0.634860 ms**) and gate_up (+0.204870 ms). At M=4, cuBLAS selects
CUTLASS WMMA BF16 kernels; B=1 predominantly uses GEMV. SGLang rises from
9.935388 ms to 11.848763 ms (+1.913375 ms), including +1.350684 ms for its
separate small GDN ba projection. CTI's fused layout has 129 linear calls per
step; SGLang's split layout has 153.

These are decode-only, complete-graph measurements, excluding truncated windows
and prefill. Projection shapes were inferred from verified configuration,
source order and grid checks, not captured GEMM arguments. They establish the
motivation for small-M optimization, not an engine speedup from this kernel.

## Limitations

- GB10 large shapes still run about 10% slower than cuBLAS GEMV in the supplied
  tuning evidence. Unqualified shapes fall back to cuBLAS through PyTorch.
  cuBLAS can stream above the measured copy ceiling; utilization may exceed one.
- gate_up often ties; no uniform improvement or 90%-of-bandwidth claim is made.
- RTX 5090 event timing is quantized at roughly 1 us in these measurements,
  making small-call near-ties noisy.
- Fused gate_up plus SiLU is not implemented. This milestone is BF16 only;
  there is no block-FP8 path or quantization-scale API.
- A previous acceptance round does not qualify the current dispatch table.
  Missing or incomplete current acceptance remains an integration limitation.
  Summed isolated cold GEMM times are not end-to-end decode latency.

## Regeneration (CPU only)

Run from the repository root, substituting public local log filenames:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python benchmark/kernel_tuning/linear_smallm_table.py \
  --markdown --market market_rtx5090.log market_pro6000.log market_gb10.log \
  --probe probe_rtx5090.log probe_pro6000.log probe_gb10.log probe_sm120_previous.log \
  --accept accept_rtx5090.log accept_pro6000.log accept_gb10.log \
  --previous-accept previous_accept_rtx5090.log previous_accept_pro6000.log \
  > benchmark/kernel_tuning/results/linear_smallm.md
```

Named input groups tolerate absent files so the same command works when final
logs arrive. Within market/acceptance, the last supplied run per device wins;
current acceptance takes precedence over `--previous-accept`. Completion status
is reported separately from available rows; incomplete runs are not declared
accepted. Probe groups aggregate supplied observations, including earlier
sm120 probes. Do not combine incompatible kernel versions. No source filenames,
paths, environment dictionaries or exception diagnostics are copied into this
report. Static report text lives beside the table tool in
`linear_smallm_report.md`; regenerate after changing it.

Dispatch regeneration is separate and does not run during `--markdown`.
It qualifies rows only from the current `--accept` logs; `--previous-accept`
rows are reported but never qualify a dispatch entry. Pass the cold, engine
(`--l2 engine`) and hot (`--l2 hot`) acceptance logs, measured at every bucket
edge (`linear_smallm.py --m accept`): a bucket needs passing cold and engine
records at both edges, and cold logs alone yield an empty table:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python benchmark/kernel_tuning/linear_smallm_table.py \
  --probe probe_rtx5090.log probe_pro6000.log probe_gb10.log probe_sm120_previous.log \
  --accept cold_rtx5090.log cold_pro6000.log cold_gb10.log \
  engine_rtx5090.log engine_pro6000.log engine_gb10.log hot_rtx5090.log hot_pro6000.log \
  --rewrite
```

Then format the generated Python. Without `--probe`, `--rewrite` re-gates the
committed table against the `--accept` logs. Without `--rewrite`, output is a
Python dictionary literal.

## Market measurements

Cold, read-flush; Qwen3.5-9B. Each provider cell is **microseconds / utilization**.

### RTX 5090

Run completion marker: True; error rows: 0; pairs: [20]; replays/sample: [5].

| Projection | N | K | M | cuBLAS | cuBLASLt | QuACK 0.6.5 | Ours (dispatch) | SOL us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdn.in_proj_qkvz | 12288 | 4096 | 1 | 63.488 / 0.955 | 64.000 / 0.947 | 104.448 / 0.580 | 63.488 / 0.955 (CuTe) | 60.6238 |
| gdn.in_proj_qkvz | 12288 | 4096 | 2 | 65.536 / 0.925 | 65.536 / 0.925 | 106.496 / 0.569 | 63.488 / 0.955 (CuTe) | 60.6435 |
| gdn.in_proj_qkvz | 12288 | 4096 | 3 | 65.536 / 0.926 | 65.536 / 0.926 | 104.448 / 0.581 | 63.488 / 0.956 (CuTe) | 60.6632 |
| gdn.in_proj_qkvz | 12288 | 4096 | 4 | 65.536 / 0.926 | 65.536 / 0.926 | 104.448 / 0.581 | 65.536 / 0.926 (CuTe) | 60.6829 |
| gdn.in_proj_qkvz | 12288 | 4096 | 6 | 65.536 / 0.927 | 65.536 / 0.927 | 106.496 / 0.570 | 65.536 / 0.927 (CuTe) | 60.7224 |
| gdn.in_proj_qkvz | 12288 | 4096 | 8 | 65.536 / 0.927 | 65.536 / 0.927 | 100.352 / 0.605 | 65.536 / 0.927 (cuBLAS) | 60.7619 |
| gdn.in_proj_ba | 64 | 4096 | 1 | 6.144 / 0.052 | 6.144 / 0.052 | 98.304 / 0.003 | 6.144 / 0.052 (CuTe) | 0.3207 |
| gdn.in_proj_ba | 64 | 4096 | 2 | 26.624 / 0.012 | 26.624 / 0.012 | 98.304 / 0.003 | 6.144 / 0.053 (CuTe) | 0.3257 |
| gdn.in_proj_ba | 64 | 4096 | 3 | 26.624 / 0.012 | 26.624 / 0.012 | 98.304 / 0.003 | 6.144 / 0.054 (CuTe) | 0.3307 |
| gdn.in_proj_ba | 64 | 4096 | 4 | 26.624 / 0.013 | 26.624 / 0.013 | 98.304 / 0.003 | 6.144 / 0.055 (CuTe) | 0.3357 |
| gdn.in_proj_ba | 64 | 4096 | 6 | 26.624 / 0.013 | 26.624 / 0.013 | 98.304 / 0.004 | 6.144 / 0.056 (CuTe) | 0.3457 |
| gdn.in_proj_ba | 64 | 4096 | 8 | 8.192 / 0.043 | 6.144 / 0.058 | 98.304 / 0.004 | 6.144 / 0.058 (CuTe) | 0.3557 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 1 | 63.488 / 0.960 | 63.488 / 0.960 | 104.448 / 0.583 | 63.488 / 0.960 (cuBLAS) | 60.9395 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 2 | 65.536 / 0.930 | 65.536 / 0.930 | 106.496 / 0.572 | 65.536 / 0.930 (CuTe) | 60.9593 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 3 | 65.536 / 0.930 | 65.536 / 0.930 | 106.496 / 0.573 | 65.536 / 0.930 (CuTe) | 60.9791 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 4 | 65.536 / 0.931 | 65.536 / 0.931 | 106.496 / 0.573 | 65.536 / 0.931 (CuTe) | 60.9989 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 6 | 65.536 / 0.931 | 65.536 / 0.931 | 106.496 / 0.573 | 65.536 / 0.931 (CuTe) | 61.0385 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 8 | 65.536 / 0.932 | 65.536 / 0.932 | 102.400 / 0.596 | 65.536 / 0.932 (CuTe) | 61.0781 |
| gdn.out_proj | 4096 | 4096 | 1 | 26.624 / 0.759 | 26.624 / 0.759 | 98.304 / 0.206 | 24.576 / 0.822 (CuTe) | 20.2112 |
| gdn.out_proj | 4096 | 4096 | 2 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.823 (CuTe) | 20.2211 |
| gdn.out_proj | 4096 | 4096 | 3 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.823 (CuTe) | 20.2309 |
| gdn.out_proj | 4096 | 4096 | 4 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.824 (CuTe) | 20.2408 |
| gdn.out_proj | 4096 | 4096 | 6 | 26.624 / 0.761 | 26.624 / 0.761 | 98.304 / 0.206 | 24.576 / 0.824 (CuTe) | 20.2605 |
| gdn.out_proj | 4096 | 4096 | 8 | 26.624 / 0.762 | 26.624 / 0.762 | 98.304 / 0.206 | 24.576 / 0.825 (CuTe) | 20.2803 |
| attention.qkv | 10240 | 4096 | 1 | 53.248 / 0.949 | 53.248 / 0.949 | 98.304 / 0.514 | 53.248 / 0.949 (cuBLAS) | 50.5206 |
| attention.qkv | 10240 | 4096 | 2 | 55.296 / 0.914 | 55.296 / 0.914 | 100.352 / 0.504 | 55.296 / 0.914 (cuBLAS) | 50.5379 |
| attention.qkv | 10240 | 4096 | 3 | 55.296 / 0.914 | 55.296 / 0.914 | 100.352 / 0.504 | 55.296 / 0.914 (cuBLAS) | 50.5551 |
| attention.qkv | 10240 | 4096 | 4 | 55.296 / 0.915 | 55.296 / 0.915 | 98.304 / 0.514 | 55.296 / 0.915 (CuTe) | 50.5724 |
| attention.qkv | 10240 | 4096 | 6 | 55.296 / 0.915 | 55.296 / 0.915 | 100.352 / 0.504 | 53.248 / 0.950 (CuTe) | 50.6069 |
| attention.qkv | 10240 | 4096 | 8 | 55.296 / 0.916 | 55.296 / 0.916 | 98.304 / 0.515 | 53.248 / 0.951 (CuTe) | 50.6415 |
| attention.o | 4096 | 4096 | 1 | 26.624 / 0.759 | 26.624 / 0.759 | 98.304 / 0.206 | 24.576 / 0.822 (CuTe) | 20.2112 |
| attention.o | 4096 | 4096 | 2 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.823 (CuTe) | 20.2211 |
| attention.o | 4096 | 4096 | 3 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.823 (CuTe) | 20.2309 |
| attention.o | 4096 | 4096 | 4 | 26.624 / 0.760 | 26.624 / 0.760 | 98.304 / 0.206 | 24.576 / 0.824 (CuTe) | 20.2408 |
| attention.o | 4096 | 4096 | 6 | 26.624 / 0.761 | 26.624 / 0.761 | 98.304 / 0.206 | 24.576 / 0.824 (CuTe) | 20.2605 |
| attention.o | 4096 | 4096 | 8 | 26.624 / 0.762 | 26.624 / 0.762 | 98.304 / 0.206 | 24.576 / 0.825 (CuTe) | 20.2803 |
| mlp.gate_up | 24576 | 4096 | 1 | 122.880 / 0.987 | 122.880 / 0.987 | 245.760 / 0.493 | 122.880 / 0.987 (cuBLAS) | 121.2426 |
| mlp.gate_up | 24576 | 4096 | 2 | 124.928 / 0.971 | 124.928 / 0.971 | 249.856 / 0.485 | 124.928 / 0.971 (cuBLAS) | 121.2771 |
| mlp.gate_up | 24576 | 4096 | 3 | 124.928 / 0.971 | 124.928 / 0.971 | 245.760 / 0.494 | 124.928 / 0.971 (CuTe) | 121.3116 |
| mlp.gate_up | 24576 | 4096 | 4 | 124.928 / 0.971 | 124.928 / 0.971 | 247.808 / 0.490 | 123.648 / 0.981 (CuTe) | 121.3462 |
| mlp.gate_up | 24576 | 4096 | 6 | 124.928 / 0.972 | 124.928 / 0.972 | 248.832 / 0.488 | 124.928 / 0.972 (cuBLAS) | 121.4152 |
| mlp.gate_up | 24576 | 4096 | 8 | 124.928 / 0.972 | 124.928 / 0.972 | 239.616 / 0.507 | 124.928 / 0.972 (cuBLAS) | 121.4843 |
| mlp.down | 4096 | 12288 | 1 | 66.048 / 0.918 | 65.536 / 0.925 | 284.384 / 0.213 | 65.536 / 0.925 (cuBLAS) | 60.6238 |
| mlp.down | 4096 | 12288 | 2 | 67.584 / 0.897 | 67.584 / 0.897 | 284.672 / 0.213 | 65.536 / 0.925 (CuTe) | 60.6435 |
| mlp.down | 4096 | 12288 | 3 | 77.824 / 0.779 | 76.560 / 0.792 | 284.384 / 0.213 | 65.536 / 0.926 (CuTe) | 60.6632 |
| mlp.down | 4096 | 12288 | 4 | 77.824 / 0.780 | 77.824 / 0.780 | 284.336 / 0.213 | 65.536 / 0.926 (CuTe) | 60.6829 |
| mlp.down | 4096 | 12288 | 6 | 77.824 / 0.780 | 77.824 / 0.780 | 284.672 / 0.213 | 65.536 / 0.927 (CuTe) | 60.7224 |
| mlp.down | 4096 | 12288 | 8 | 77.824 / 0.781 | 77.824 / 0.781 | 284.672 / 0.213 | 65.536 / 0.927 (CuTe) | 60.7619 |
| lm_head | 248320 | 4096 | 1 | 1265.520 / 0.968 | 1263.616 / 0.969 | 1496.944 / 0.818 | 1198.080 / 1.022 (CuTe) | 1225.0104 |
| lm_head | 248320 | 4096 | 2 | 1204.224 / 1.018 | 1204.224 / 1.018 | 1549.312 / 0.791 | 1204.224 / 1.018 (cuBLAS) | 1225.3143 |
| lm_head | 248320 | 4096 | 3 | 1204.224 / 1.018 | 1204.224 / 1.018 | 1495.040 / 0.820 | 1204.224 / 1.018 (cuBLAS) | 1225.6183 |
| lm_head | 248320 | 4096 | 4 | 1206.272 / 1.016 | 1206.272 / 1.016 | 1568.768 / 0.781 | 1206.272 / 1.016 (cuBLAS) | 1225.9222 |
| lm_head | 248320 | 4096 | 6 | 1206.272 / 1.017 | 1206.272 / 1.017 | 1607.680 / 0.763 | 1206.272 / 1.017 (cuBLAS) | 1226.5301 |
| lm_head | 248320 | 4096 | 8 | 1210.080 / 1.014 | 1210.368 / 1.014 | 1456.128 / 0.843 | 1208.320 / 1.016 (cuBLAS) | 1227.1379 |

### RTX PRO 6000

Run completion marker: True; error rows: 0; pairs: [20]; replays/sample: [5].

| Projection | N | K | M | cuBLAS | cuBLASLt | QuACK 0.6.5 | Ours (dispatch) | SOL us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdn.in_proj_qkvz | 12288 | 4096 | 1 | 71.648 / 0.911 | 71.136 / 0.917 | 114.176 / 0.572 | 69.600 / 0.938 (CuTe) | 65.2599 |
| gdn.in_proj_qkvz | 12288 | 4096 | 2 | 71.648 / 0.911 | 71.648 / 0.911 | 114.688 / 0.569 | 69.600 / 0.938 (CuTe) | 65.2812 |
| gdn.in_proj_qkvz | 12288 | 4096 | 3 | 71.648 / 0.911 | 71.648 / 0.911 | 114.688 / 0.569 | 69.600 / 0.938 (CuTe) | 65.3024 |
| gdn.in_proj_qkvz | 12288 | 4096 | 4 | 71.648 / 0.912 | 71.648 / 0.912 | 114.176 / 0.572 | 69.600 / 0.939 (CuTe) | 65.3236 |
| gdn.in_proj_qkvz | 12288 | 4096 | 6 | 71.648 / 0.912 | 71.648 / 0.912 | 114.688 / 0.570 | 69.600 / 0.939 (CuTe) | 65.3661 |
| gdn.in_proj_qkvz | 12288 | 4096 | 8 | 71.648 / 0.913 | 71.648 / 0.913 | 108.544 / 0.603 | 69.600 / 0.940 (CuTe) | 65.4086 |
| gdn.in_proj_ba | 64 | 4096 | 1 | 6.144 / 0.056 | 6.144 / 0.056 | 62.976 / 0.005 | 6.144 / 0.056 (CuTe) | 0.3452 |
| gdn.in_proj_ba | 64 | 4096 | 2 | 28.640 / 0.012 | 28.160 / 0.012 | 63.488 / 0.006 | 6.144 / 0.057 (CuTe) | 0.3506 |
| gdn.in_proj_ba | 64 | 4096 | 3 | 28.640 / 0.012 | 28.400 / 0.013 | 63.488 / 0.006 | 6.144 / 0.058 (CuTe) | 0.3560 |
| gdn.in_proj_ba | 64 | 4096 | 4 | 28.640 / 0.013 | 28.672 / 0.013 | 63.488 / 0.006 | 6.144 / 0.059 (CuTe) | 0.3614 |
| gdn.in_proj_ba | 64 | 4096 | 6 | 28.640 / 0.013 | 28.640 / 0.013 | 63.488 / 0.006 | 6.144 / 0.061 (CuTe) | 0.3721 |
| gdn.in_proj_ba | 64 | 4096 | 8 | 8.192 / 0.047 | 7.680 / 0.050 | 62.976 / 0.006 | 6.144 / 0.062 (CuTe) | 0.3829 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 1 | 71.648 / 0.916 | 71.408 / 0.919 | 118.288 / 0.555 | 71.648 / 0.916 (cuBLAS) | 65.5998 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 2 | 71.648 / 0.916 | 71.648 / 0.916 | 120.832 / 0.543 | 71.648 / 0.916 (cuBLAS) | 65.6211 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 3 | 71.648 / 0.916 | 71.648 / 0.916 | 118.272 / 0.555 | 71.648 / 0.916 (CuTe) | 65.6424 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 4 | 71.648 / 0.916 | 71.648 / 0.916 | 118.272 / 0.555 | 71.648 / 0.916 (CuTe) | 65.6638 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 6 | 71.648 / 0.917 | 71.648 / 0.917 | 120.832 / 0.544 | 71.648 / 0.917 (CuTe) | 65.7064 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 8 | 71.648 / 0.918 | 71.648 / 0.918 | 112.640 / 0.584 | 71.648 / 0.918 (CuTe) | 65.7490 |
| gdn.out_proj | 4096 | 4096 | 1 | 30.112 / 0.723 | 28.640 / 0.760 | 63.456 / 0.343 | 26.592 / 0.818 (CuTe) | 21.7568 |
| gdn.out_proj | 4096 | 4096 | 2 | 28.640 / 0.760 | 28.640 / 0.760 | 63.456 / 0.343 | 27.968 / 0.778 (CuTe) | 21.7675 |
| gdn.out_proj | 4096 | 4096 | 3 | 28.640 / 0.760 | 28.640 / 0.760 | 63.456 / 0.343 | 28.640 / 0.760 (CuTe) | 21.7781 |
| gdn.out_proj | 4096 | 4096 | 4 | 28.640 / 0.761 | 30.688 / 0.710 | 63.456 / 0.343 | 28.032 / 0.777 (CuTe) | 21.7887 |
| gdn.out_proj | 4096 | 4096 | 6 | 28.672 / 0.761 | 30.176 / 0.723 | 63.488 / 0.344 | 28.640 / 0.762 (CuTe) | 21.8099 |
| gdn.out_proj | 4096 | 4096 | 8 | 28.640 / 0.762 | 30.176 / 0.723 | 63.488 / 0.344 | 26.592 / 0.821 (CuTe) | 21.8312 |
| attention.qkv | 10240 | 4096 | 1 | 61.440 / 0.885 | 59.360 / 0.916 | 95.648 / 0.569 | 59.360 / 0.916 (CuTe) | 54.3842 |
| attention.qkv | 10240 | 4096 | 2 | 59.360 / 0.916 | 60.896 / 0.893 | 96.256 / 0.565 | 59.360 / 0.916 (cuBLAS) | 54.4027 |
| attention.qkv | 10240 | 4096 | 3 | 60.896 / 0.894 | 59.360 / 0.917 | 94.176 / 0.578 | 59.360 / 0.917 (CuTe) | 54.4213 |
| attention.qkv | 10240 | 4096 | 4 | 60.880 / 0.894 | 60.896 / 0.894 | 94.176 / 0.578 | 59.360 / 0.917 (CuTe) | 54.4399 |
| attention.qkv | 10240 | 4096 | 6 | 60.896 / 0.895 | 61.408 / 0.887 | 96.224 / 0.566 | 59.360 / 0.918 (CuTe) | 54.4771 |
| attention.qkv | 10240 | 4096 | 8 | 60.896 / 0.895 | 61.408 / 0.888 | 90.080 / 0.605 | 59.360 / 0.918 (CuTe) | 54.5142 |
| attention.o | 4096 | 4096 | 1 | 30.112 / 0.723 | 28.640 / 0.760 | 63.456 / 0.343 | 26.592 / 0.818 (CuTe) | 21.7568 |
| attention.o | 4096 | 4096 | 2 | 28.640 / 0.760 | 28.640 / 0.760 | 63.456 / 0.343 | 27.968 / 0.778 (CuTe) | 21.7675 |
| attention.o | 4096 | 4096 | 3 | 28.640 / 0.760 | 28.640 / 0.760 | 63.456 / 0.343 | 28.640 / 0.760 (CuTe) | 21.7781 |
| attention.o | 4096 | 4096 | 4 | 28.640 / 0.761 | 30.688 / 0.710 | 63.456 / 0.343 | 28.032 / 0.777 (CuTe) | 21.7887 |
| attention.o | 4096 | 4096 | 6 | 28.672 / 0.761 | 30.176 / 0.723 | 63.488 / 0.344 | 28.640 / 0.762 (CuTe) | 21.8099 |
| attention.o | 4096 | 4096 | 8 | 28.640 / 0.762 | 30.176 / 0.723 | 63.488 / 0.344 | 26.592 / 0.821 (CuTe) | 21.8312 |
| mlp.gate_up | 24576 | 4096 | 1 | 140.672 / 0.928 | 141.280 / 0.924 | 229.376 / 0.569 | 137.184 / 0.951 (CuTe) | 130.5145 |
| mlp.gate_up | 24576 | 4096 | 2 | 136.672 / 0.955 | 139.232 / 0.938 | 235.520 / 0.554 | 136.512 / 0.956 (cuBLAS) | 130.5517 |
| mlp.gate_up | 24576 | 4096 | 3 | 136.672 / 0.955 | 139.232 / 0.938 | 231.424 / 0.564 | 136.672 / 0.955 (cuBLAS) | 130.5889 |
| mlp.gate_up | 24576 | 4096 | 4 | 136.672 / 0.956 | 139.968 / 0.933 | 229.376 / 0.569 | 136.608 / 0.956 (cuBLAS) | 130.6260 |
| mlp.gate_up | 24576 | 4096 | 6 | 136.672 / 0.956 | 139.232 / 0.939 | 235.520 / 0.555 | 136.672 / 0.956 (cuBLAS) | 130.7004 |
| mlp.gate_up | 24576 | 4096 | 8 | 136.672 / 0.957 | 139.232 / 0.939 | 219.152 / 0.597 | 136.672 / 0.957 (cuBLAS) | 130.7747 |
| mlp.down | 4096 | 12288 | 1 | 75.776 / 0.861 | 73.184 / 0.892 | 176.128 / 0.371 | 73.728 / 0.885 (CuTe) | 65.2599 |
| mlp.down | 4096 | 12288 | 2 | 73.728 / 0.885 | 75.248 / 0.868 | 178.848 / 0.365 | 73.696 / 0.886 (CuTe) | 65.2812 |
| mlp.down | 4096 | 12288 | 3 | 73.728 / 0.886 | 75.264 / 0.868 | 176.112 / 0.371 | 73.696 / 0.886 (CuTe) | 65.3024 |
| mlp.down | 4096 | 12288 | 4 | 73.728 / 0.886 | 73.728 / 0.886 | 176.128 / 0.371 | 73.696 / 0.886 (CuTe) | 65.3236 |
| mlp.down | 4096 | 12288 | 6 | 73.728 / 0.887 | 75.776 / 0.863 | 176.112 / 0.371 | 75.088 / 0.871 (CuTe) | 65.3661 |
| mlp.down | 4096 | 12288 | 8 | 73.728 / 0.887 | 75.264 / 0.869 | 176.112 / 0.371 | 73.696 / 0.888 (CuTe) | 65.4086 |
| lm_head | 248320 | 4096 | 1 | 1378.304 / 0.957 | 1367.552 / 0.964 | 1515.248 / 0.870 | 1326.544 / 0.994 (CuTe) | 1318.6923 |
| lm_head | 248320 | 4096 | 2 | 1327.072 / 0.994 | 1331.200 / 0.991 | 1576.960 / 0.836 | 1327.088 / 0.994 (cuBLAS) | 1319.0195 |
| lm_head | 248320 | 4096 | 3 | 1329.120 / 0.993 | 1339.392 / 0.985 | 1517.568 / 0.869 | 1329.152 / 0.993 (cuBLAS) | 1319.3467 |
| lm_head | 248320 | 4096 | 4 | 1331.168 / 0.991 | 1341.440 / 0.984 | 1518.592 / 0.869 | 1331.200 / 0.991 (cuBLAS) | 1319.6739 |
| lm_head | 248320 | 4096 | 6 | 1335.296 / 0.989 | 1335.296 / 0.989 | 1589.248 / 0.831 | 1335.296 / 0.989 (cuBLAS) | 1320.3282 |
| lm_head | 248320 | 4096 | 8 | 1339.392 / 0.986 | 1337.344 / 0.988 | 1472.512 / 0.897 | 1339.392 / 0.986 (cuBLAS) | 1320.9826 |

### GB10

Run completion marker: True; error rows: 0; pairs: [20]; replays/sample: [5].

| Projection | N | K | M | cuBLAS | cuBLASLt | QuACK 0.6.5 | Ours (dispatch) | SOL us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdn.in_proj_qkvz | 12288 | 4096 | 1 | 598.880 / 0.845 | 597.888 / 0.846 | 463.712 / 1.091 | 492.816 / 1.027 (CuTe) | 506.0104 |
| gdn.in_proj_qkvz | 12288 | 4096 | 2 | 441.184 / 1.147 | 441.392 / 1.147 | 463.648 / 1.092 | 442.144 / 1.145 (cuBLAS) | 506.1750 |
| gdn.in_proj_qkvz | 12288 | 4096 | 3 | 441.184 / 1.148 | 442.208 / 1.145 | 464.736 / 1.090 | 442.080 / 1.145 (cuBLAS) | 506.3397 |
| gdn.in_proj_qkvz | 12288 | 4096 | 4 | 443.200 / 1.143 | 442.512 / 1.145 | 463.712 / 1.092 | 442.368 / 1.145 (cuBLAS) | 506.5044 |
| gdn.in_proj_qkvz | 12288 | 4096 | 6 | 442.336 / 1.146 | 442.752 / 1.145 | 464.736 / 1.091 | 443.200 / 1.144 (cuBLAS) | 506.8337 |
| gdn.in_proj_qkvz | 12288 | 4096 | 8 | 444.256 / 1.142 | 443.232 / 1.144 | 465.744 / 1.089 | 444.224 / 1.142 (cuBLAS) | 507.1630 |
| gdn.in_proj_ba | 64 | 4096 | 1 | 8.032 / 0.333 | 8.032 / 0.333 | 61.280 / 0.044 | 8.032 / 0.333 (cuBLAS) | 2.6764 |
| gdn.in_proj_ba | 64 | 4096 | 2 | 30.560 / 0.089 | 28.512 / 0.095 | 61.280 / 0.044 | 8.128 / 0.334 (CuTe) | 2.7182 |
| gdn.in_proj_ba | 64 | 4096 | 3 | 28.992 / 0.095 | 28.512 / 0.097 | 61.280 / 0.045 | 8.128 / 0.340 (CuTe) | 2.7600 |
| gdn.in_proj_ba | 64 | 4096 | 4 | 30.496 / 0.092 | 28.512 / 0.098 | 61.280 / 0.046 | 8.032 / 0.349 (CuTe) | 2.8018 |
| gdn.in_proj_ba | 64 | 4096 | 6 | 30.560 / 0.094 | 28.512 / 0.101 | 61.280 / 0.047 | 8.160 / 0.354 (CuTe) | 2.8855 |
| gdn.in_proj_ba | 64 | 4096 | 8 | 10.080 / 0.295 | 10.080 / 0.295 | 61.248 / 0.048 | 8.224 / 0.361 (CuTe) | 2.9691 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 1 | 436.704 / 1.165 | 436.176 / 1.166 | 508.752 / 1.000 | 436.144 / 1.166 (cuBLAS) | 508.6456 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 2 | 444.256 / 1.145 | 444.256 / 1.145 | 505.728 / 1.006 | 444.256 / 1.145 (cuBLAS) | 508.8109 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 3 | 444.320 / 1.146 | 444.256 / 1.146 | 507.552 / 1.003 | 444.224 / 1.146 (cuBLAS) | 508.9762 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 4 | 445.488 / 1.143 | 445.360 / 1.143 | 510.816 / 0.997 | 445.264 / 1.143 (cuBLAS) | 509.1415 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 6 | 446.304 / 1.142 | 446.304 / 1.142 | 511.840 / 0.995 | 446.304 / 1.142 (cuBLAS) | 509.4722 |
| gdn.in_proj_qkvzba | 12352 | 4096 | 8 | 448.352 / 1.137 | 448.352 / 1.137 | 513.888 / 0.992 | 448.352 / 1.137 (cuBLAS) | 509.8028 |
| gdn.out_proj | 4096 | 4096 | 1 | 205.632 / 0.820 | 204.784 / 0.824 | 160.576 / 1.051 | 173.920 / 0.970 (CuTe) | 168.6976 |
| gdn.out_proj | 4096 | 4096 | 2 | 151.392 / 1.115 | 151.392 / 1.115 | 161.408 / 1.046 | 151.344 / 1.115 (cuBLAS) | 168.7799 |
| gdn.out_proj | 4096 | 4096 | 3 | 151.824 / 1.112 | 151.392 / 1.115 | 160.608 / 1.051 | 151.392 / 1.115 (cuBLAS) | 168.8622 |
| gdn.out_proj | 4096 | 4096 | 4 | 151.392 / 1.116 | 151.392 / 1.116 | 160.608 / 1.052 | 151.392 / 1.116 (cuBLAS) | 168.9446 |
| gdn.out_proj | 4096 | 4096 | 6 | 151.632 / 1.115 | 151.424 / 1.117 | 160.576 / 1.053 | 151.392 / 1.117 (cuBLAS) | 169.1092 |
| gdn.out_proj | 4096 | 4096 | 8 | 151.392 / 1.118 | 151.392 / 1.118 | 160.608 / 1.054 | 151.392 / 1.118 (cuBLAS) | 169.2739 |
| attention.qkv | 10240 | 4096 | 1 | 501.568 / 0.841 | 500.576 / 0.842 | 391.008 / 1.078 | 417.792 / 1.009 (CuTe) | 421.6822 |
| attention.qkv | 10240 | 4096 | 2 | 368.448 / 1.145 | 367.648 / 1.147 | 389.984 / 1.082 | 367.600 / 1.148 (cuBLAS) | 421.8263 |
| attention.qkv | 10240 | 4096 | 3 | 367.456 / 1.148 | 367.424 / 1.148 | 388.944 / 1.085 | 367.456 / 1.148 (cuBLAS) | 421.9703 |
| attention.qkv | 10240 | 4096 | 4 | 368.480 / 1.146 | 368.480 / 1.146 | 390.928 / 1.080 | 368.448 / 1.146 (cuBLAS) | 422.1144 |
| attention.qkv | 10240 | 4096 | 6 | 367.744 / 1.149 | 367.616 / 1.149 | 392.656 / 1.076 | 367.456 / 1.150 (cuBLAS) | 422.4026 |
| attention.qkv | 10240 | 4096 | 8 | 368.496 / 1.147 | 368.480 / 1.147 | 391.024 / 1.081 | 368.480 / 1.147 (cuBLAS) | 422.6907 |
| attention.o | 4096 | 4096 | 1 | 205.632 / 0.820 | 204.784 / 0.824 | 160.576 / 1.051 | 173.920 / 0.970 (CuTe) | 168.6976 |
| attention.o | 4096 | 4096 | 2 | 151.392 / 1.115 | 151.392 / 1.115 | 161.408 / 1.046 | 151.344 / 1.115 (cuBLAS) | 168.7799 |
| attention.o | 4096 | 4096 | 3 | 151.824 / 1.112 | 151.392 / 1.115 | 160.608 / 1.051 | 151.392 / 1.115 (cuBLAS) | 168.8622 |
| attention.o | 4096 | 4096 | 4 | 151.392 / 1.116 | 151.392 / 1.116 | 160.608 / 1.052 | 151.392 / 1.116 (cuBLAS) | 168.9446 |
| attention.o | 4096 | 4096 | 6 | 151.632 / 1.115 | 151.424 / 1.117 | 160.576 / 1.053 | 151.392 / 1.117 (cuBLAS) | 169.1092 |
| attention.o | 4096 | 4096 | 8 | 151.392 / 1.118 | 151.392 / 1.118 | 160.608 / 1.054 | 151.392 / 1.118 (cuBLAS) | 169.2739 |
| mlp.gate_up | 24576 | 4096 | 1 | 866.176 / 1.168 | 866.176 / 1.168 | 920.416 / 1.099 | 866.176 / 1.168 (cuBLAS) | 1011.9796 |
| mlp.gate_up | 24576 | 4096 | 2 | 875.360 / 1.156 | 875.360 / 1.156 | 921.424 / 1.099 | 875.360 / 1.156 (cuBLAS) | 1012.2677 |
| mlp.gate_up | 24576 | 4096 | 3 | 876.384 / 1.155 | 876.368 / 1.155 | 920.736 / 1.100 | 876.384 / 1.155 (cuBLAS) | 1012.5559 |
| mlp.gate_up | 24576 | 4096 | 4 | 877.408 / 1.154 | 877.408 / 1.154 | 924.544 / 1.096 | 877.072 / 1.155 (cuBLAS) | 1012.8441 |
| mlp.gate_up | 24576 | 4096 | 6 | 877.456 / 1.155 | 878.432 / 1.154 | 922.512 / 1.099 | 878.416 / 1.154 (cuBLAS) | 1013.4204 |
| mlp.gate_up | 24576 | 4096 | 8 | 877.440 / 1.156 | 877.440 / 1.156 | 922.480 / 1.099 | 877.312 / 1.156 (cuBLAS) | 1013.9967 |
| mlp.down | 4096 | 12288 | 1 | 441.184 / 1.147 | 441.168 / 1.147 | 465.984 / 1.086 | 441.184 / 1.147 (cuBLAS) | 506.0104 |
| mlp.down | 4096 | 12288 | 2 | 451.408 / 1.121 | 450.624 / 1.123 | 476.192 / 1.063 | 450.400 / 1.124 (cuBLAS) | 506.1750 |
| mlp.down | 4096 | 12288 | 3 | 447.328 / 1.132 | 447.312 / 1.132 | 470.880 / 1.075 | 447.328 / 1.132 (cuBLAS) | 506.3397 |
| mlp.down | 4096 | 12288 | 4 | 448.496 / 1.129 | 448.976 / 1.128 | 471.056 / 1.075 | 448.352 / 1.130 (cuBLAS) | 506.5044 |
| mlp.down | 4096 | 12288 | 6 | 444.256 / 1.141 | 443.232 / 1.143 | 466.880 / 1.086 | 443.232 / 1.143 (cuBLAS) | 506.8337 |
| mlp.down | 4096 | 12288 | 8 | 446.304 / 1.136 | 446.320 / 1.136 | 464.064 / 1.093 | 446.304 / 1.136 (cuBLAS) | 507.1630 |
| lm_head | 248320 | 4096 | 1 | 11749.232 / 0.870 | 11749.200 / 0.870 | 9212.208 / 1.110 | 10260.048 / 0.997 (CuTe) | 10224.8355 |
| lm_head | 248320 | 4096 | 2 | 8766.304 / 1.167 | 8765.328 / 1.167 | 9202.592 / 1.111 | 8764.768 / 1.167 (cuBLAS) | 10227.3724 |
| lm_head | 248320 | 4096 | 3 | 8776.528 / 1.166 | 8775.536 / 1.166 | 9212.256 / 1.110 | 8775.552 / 1.166 (cuBLAS) | 10229.9092 |
| lm_head | 248320 | 4096 | 4 | 8784.768 / 1.165 | 8779.728 / 1.165 | 9236.688 / 1.108 | 8785.408 / 1.165 (cuBLAS) | 10232.4461 |
| lm_head | 248320 | 4096 | 6 | 8799.168 / 1.163 | 8798.080 / 1.164 | 9266.032 / 1.105 | 8803.056 / 1.163 (cuBLAS) | 10237.5198 |
| lm_head | 248320 | 4096 | 8 | 8810.368 / 1.163 | 8807.872 / 1.163 | 9249.120 / 1.107 | 8810.352 / 1.163 (cuBLAS) | 10242.5934 |

## Probe best configurations

Best correct cold candidate per device/N/K/M across supplied probe observations, including losers to cuBLAS. Repeats use median paired speedup; any incorrect observation vetoes that candidate/M. These selection data are not independent acceptance. `Admit` means speedup >=1.03, not that this report rewrote dispatch. Config = (rows, tile K, stages, splits, warps, variant, prefetch).

### RTX 5090

Supplied probe runs: 4; runs with successful completion markers: 4.

#### qwen3.5-9b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 4096 | 1 | (16, 128, 2, 1, 1, 'reg', 4) | 63.4880 | 65.5360 | 1.0323 | yes |
| 12288 | 4096 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 63.4880 | 65.5360 | 1.0323 | yes |
| 12288 | 4096 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5200 | 65.5360 | 1.0239 | no |
| 12288 | 4096 | 4 | (64, 64, 3, 2, 4, 'smem', 4) | 63.4880 | 65.5360 | 1.0300 | no |
| 12288 | 4096 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5120 | 65.5360 | 1.0161 | no |
| 12288 | 4096 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5120 | 65.5360 | 1.0149 | no |
| 12288 | 4096 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5120 | 65.5360 | 1.0140 | no |
| 12288 | 4096 | 8 | (16, 128, 2, 1, 1, 'reg', 4) | 65.5360 | 65.5360 | 1.0000 | no |
| 64 | 4096 | 1 | (16, 64, 3, 32, 1, 'smem', 4) | 5.5600 | 6.1440 | 1.1197 | yes |
| 64 | 4096 | 2 | (16, 64, 3, 32, 1, 'smem', 4) | 5.1200 | 26.6240 | 5.4167 | yes |
| 64 | 4096 | 3 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 26.6240 | 4.3333 | yes |
| 64 | 4096 | 4 | (16, 64, 3, 32, 1, 'smem', 4) | 5.6320 | 26.6240 | 4.8750 | yes |
| 64 | 4096 | 5 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 26.6240 | 4.3333 | yes |
| 64 | 4096 | 6 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 26.6240 | 4.3333 | yes |
| 64 | 4096 | 7 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 26.6240 | 4.3333 | yes |
| 64 | 4096 | 8 | (16, 128, 2, 32, 1, 'reg', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 12352 | 4096 | 1 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5040 | 65.0240 | 1.0011 | no |
| 12352 | 4096 | 2 | (64, 64, 3, 2, 4, 'smem', 4) | 63.4880 | 65.5360 | 1.0318 | yes |
| 12352 | 4096 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5120 | 65.5360 | 1.0309 | yes |
| 12352 | 4096 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 64.9200 | 66.5600 | 1.0313 | yes |
| 12352 | 4096 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 64.5120 | 65.5360 | 1.0163 | no |
| 12352 | 4096 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 65.4560 | 65.5360 | 1.0249 | no |
| 12352 | 4096 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 65.5360 | 66.4880 | 1.0146 | no |
| 12352 | 4096 | 8 | (64, 128, 2, 1, 4, 'smem', 4) | 65.5360 | 66.5600 | 1.0241 | no |
| 4096 | 4096 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 2 | (64, 64, 2, 8, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 4 | (64, 64, 2, 8, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 4096 | 4096 | 8 | (64, 64, 3, 8, 4, 'smem', 4) | 24.5760 | 26.6240 | 1.0833 | yes |
| 10240 | 4096 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 54.2720 | 54.2720 | 1.0000 | no |
| 10240 | 4096 | 2 | (64, 64, 3, 4, 4, 'smem', 4) | 53.2480 | 55.2960 | 1.0000 | no |
| 10240 | 4096 | 3 | (64, 128, 2, 2, 4, 'smem', 4) | 55.2960 | 55.2960 | 1.0000 | no |
| 10240 | 4096 | 4 | (64, 128, 2, 8, 4, 'smem', 4) | 55.2960 | 55.2960 | 1.0184 | no |
| 10240 | 4096 | 5 | (64, 128, 2, 2, 4, 'smem', 4) | 55.2960 | 55.2960 | 1.0000 | no |
| 10240 | 4096 | 6 | (64, 128, 2, 2, 4, 'smem', 4) | 55.2960 | 56.3200 | 1.0370 | yes |
| 10240 | 4096 | 7 | (64, 128, 2, 2, 4, 'smem', 4) | 54.2720 | 57.3440 | 1.0377 | yes |
| 10240 | 4096 | 8 | (64, 128, 2, 2, 4, 'smem', 4) | 55.2960 | 55.2960 | 1.0341 | yes |
| 24576 | 4096 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 122.8640 | 122.8800 | 1.0001 | no |
| 24576 | 4096 | 2 | (64, 128, 3, 1, 4, 'smem', 4) | 123.3200 | 124.9200 | 1.0047 | no |
| 24576 | 4096 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 122.8800 | 124.9280 | 1.0167 | no |
| 24576 | 4096 | 4 | (64, 128, 4, 1, 4, 'smem', 4) | 122.8800 | 124.7200 | 1.0133 | no |
| 24576 | 4096 | 5 | (64, 128, 3, 1, 4, 'smem', 4) | 124.9280 | 124.7840 | 1.0000 | no |
| 24576 | 4096 | 6 | (64, 128, 3, 1, 4, 'smem', 4) | 124.9280 | 124.9280 | 0.9928 | no |
| 24576 | 4096 | 7 | (64, 128, 3, 1, 4, 'smem', 4) | 124.9280 | 124.9280 | 1.0000 | no |
| 24576 | 4096 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 124.9280 | 124.9280 | 1.0000 | no |
| 4096 | 12288 | 1 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 66.0480 | 1.0078 | no |
| 4096 | 12288 | 2 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 69.6320 | 1.0625 | yes |
| 4096 | 12288 | 3 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 77.8240 | 1.1875 | yes |
| 4096 | 12288 | 4 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 79.7600 | 1.2094 | yes |
| 4096 | 12288 | 5 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 78.8480 | 1.1875 | yes |
| 4096 | 12288 | 6 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 79.8720 | 1.2187 | yes |
| 4096 | 12288 | 7 | (16, 128, 2, 3, 1, 'reg', 4) | 65.5360 | 79.8720 | 1.1875 | yes |
| 4096 | 12288 | 8 | (16, 128, 2, 3, 1, 'reg', 4) | 66.0480 | 79.8720 | 1.1862 | yes |
| 248320 | 4096 | 1 | (128, 64, 3, 1, 4, 'smem', 4) | 1197.0560 | 1265.1440 | 1.0564 | yes |
| 248320 | 4096 | 2 | (128, 64, 3, 1, 4, 'smem', 4) | 1198.0799 | 1202.6800 | 1.0034 | no |
| 248320 | 4096 | 3 | (128, 128, 2, 1, 4, 'smem', 4) | 1202.1760 | 1208.3200 | 1.0060 | no |
| 248320 | 4096 | 4 | (128, 128, 2, 1, 4, 'smem', 4) | 1202.1760 | 1208.3200 | 1.0051 | no |
| 248320 | 4096 | 5 | (128, 128, 2, 1, 4, 'smem', 4) | 1204.2240 | 1210.3680 | 1.0060 | no |
| 248320 | 4096 | 6 | (128, 128, 2, 1, 4, 'smem', 4) | 1204.2240 | 1212.4161 | 1.0060 | no |
| 248320 | 4096 | 7 | (128, 64, 3, 1, 4, 'smem', 4) | 1203.2000 | 1208.3200 | 1.0043 | no |
| 248320 | 4096 | 8 | (128, 128, 2, 1, 4, 'smem', 4) | 1206.2720 | 1212.4161 | 1.0051 | no |

#### qwen3.5-27b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 5120 | 1 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4480 | 104.4480 | 1.0000 | no |
| 16384 | 5120 | 2 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4480 | 114.6880 | 1.0980 | yes |
| 16384 | 5120 | 3 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4480 | 114.7040 | 1.0980 | yes |
| 16384 | 5120 | 4 | (128, 128, 2, 1, 8, 'smem', 4) | 105.4720 | 114.6880 | 1.0971 | yes |
| 16384 | 5120 | 5 | (128, 128, 2, 1, 8, 'smem', 4) | 104.4480 | 115.7120 | 1.0980 | yes |
| 16384 | 5120 | 6 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4320 | 114.6880 | 1.0984 | yes |
| 16384 | 5120 | 7 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4480 | 114.6880 | 1.0980 | yes |
| 16384 | 5120 | 8 | (128, 128, 2, 1, 4, 'smem', 4) | 104.4480 | 114.6880 | 1.0980 | yes |
| 96 | 5120 | 1 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 96 | 5120 | 2 | (16, 64, 3, 32, 1, 'smem', 4) | 4.9760 | 32.7680 | 6.7978 | yes |
| 96 | 5120 | 3 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 32.7680 | 5.3333 | yes |
| 96 | 5120 | 4 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 32.7680 | 5.3333 | yes |
| 96 | 5120 | 5 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 96 | 5120 | 6 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 96 | 5120 | 7 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 96 | 5120 | 8 | (16, 64, 2, 32, 1, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 16480 | 5120 | 1 | (128, 128, 2, 1, 8, 'smem', 4) | 105.3280 | 104.4480 | 0.9917 | no |
| 16480 | 5120 | 2 | (128, 128, 2, 1, 4, 'smem', 4) | 106.4960 | 115.7120 | 1.0900 | yes |
| 16480 | 5120 | 3 | (128, 128, 2, 1, 8, 'smem', 4) | 105.4560 | 114.6880 | 1.0971 | yes |
| 16480 | 5120 | 4 | (128, 128, 2, 1, 4, 'smem', 4) | 106.4960 | 116.4960 | 1.0962 | yes |
| 16480 | 5120 | 5 | (128, 128, 2, 1, 8, 'smem', 4) | 105.4560 | 114.6880 | 1.0864 | yes |
| 16480 | 5120 | 6 | (128, 128, 2, 1, 4, 'smem', 4) | 106.4960 | 116.7360 | 1.0962 | yes |
| 16480 | 5120 | 7 | (128, 128, 2, 1, 4, 'smem', 4) | 106.4960 | 116.7360 | 1.0962 | yes |
| 16480 | 5120 | 8 | (128, 128, 2, 1, 8, 'smem', 4) | 106.4800 | 114.6880 | 1.0877 | yes |
| 5120 | 6144 | 1 | (16, 128, 2, 3, 1, 'reg', 4) | 43.0080 | 43.0080 | 1.0000 | no |
| 5120 | 6144 | 2 | (16, 128, 2, 3, 1, 'reg', 4) | 43.0080 | 49.1520 | 1.1429 | yes |
| 5120 | 6144 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 43.0080 | 48.1280 | 1.1190 | yes |
| 5120 | 6144 | 4 | (16, 128, 2, 3, 1, 'reg', 4) | 43.0080 | 49.1520 | 1.1429 | yes |
| 5120 | 6144 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 43.0080 | 48.1280 | 1.1190 | yes |
| 5120 | 6144 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 43.0080 | 48.1280 | 1.1190 | yes |
| 5120 | 6144 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 43.0080 | 48.1280 | 1.1190 | yes |
| 5120 | 6144 | 8 | (64, 128, 2, 2, 4, 'smem', 4) | 43.0080 | 48.1200 | 1.1070 | yes |
| 14336 | 5120 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 92.5600 | 92.1520 | 0.9885 | no |
| 14336 | 5120 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 93.1840 | 98.3040 | 1.0551 | yes |
| 14336 | 5120 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 94.2080 | 98.3040 | 1.0435 | yes |
| 14336 | 5120 | 4 | (64, 64, 3, 2, 4, 'smem', 4) | 94.2080 | 98.3040 | 1.0435 | yes |
| 14336 | 5120 | 5 | (64, 128, 3, 2, 4, 'smem', 4) | 94.2080 | 112.6400 | 1.1957 | yes |
| 14336 | 5120 | 6 | (64, 128, 3, 2, 4, 'smem', 4) | 94.2080 | 112.6400 | 1.1957 | yes |
| 14336 | 5120 | 7 | (64, 128, 3, 2, 4, 'smem', 4) | 94.2080 | 112.6400 | 1.1957 | yes |
| 14336 | 5120 | 8 | (64, 128, 3, 2, 4, 'smem', 4) | 94.2080 | 112.6400 | 1.1921 | yes |
| 34816 | 5120 | 1 | (64, 128, 4, 1, 4, 'smem', 4) | 219.1360 | 214.5200 | 0.9767 | no |
| 34816 | 5120 | 2 | (64, 128, 4, 1, 4, 'smem', 4) | 219.1360 | 221.1840 | 1.0047 | no |
| 34816 | 5120 | 3 | (128, 128, 2, 1, 4, 'smem', 4) | 220.1600 | 221.1840 | 1.0047 | no |
| 34816 | 5120 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 221.1840 | 221.1840 | 1.0000 | no |
| 34816 | 5120 | 5 | (64, 128, 4, 1, 4, 'smem', 4) | 220.0160 | 221.1840 | 1.0001 | no |
| 34816 | 5120 | 6 | (128, 128, 2, 1, 4, 'smem', 4) | 221.1840 | 221.1840 | 1.0000 | no |
| 34816 | 5120 | 7 | (64, 128, 4, 1, 4, 'smem', 4) | 221.1840 | 221.1840 | 1.0000 | no |
| 34816 | 5120 | 8 | (128, 128, 2, 1, 4, 'smem', 4) | 221.1840 | 221.1840 | 0.9989 | no |
| 5120 | 17408 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 110.5920 | 112.6400 | 1.0185 | no |
| 5120 | 17408 | 2 | (64, 128, 2, 4, 4, 'smem', 4) | 110.5920 | 114.6880 | 1.0370 | yes |
| 5120 | 17408 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 110.5920 | 114.6880 | 1.0370 | yes |
| 5120 | 17408 | 4 | (64, 128, 2, 4, 4, 'smem', 4) | 110.5920 | 115.7120 | 1.0376 | yes |
| 5120 | 17408 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 111.0400 | 115.7120 | 1.0415 | yes |
| 5120 | 17408 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 111.1040 | 125.9520 | 1.1337 | yes |
| 5120 | 17408 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 112.5680 | 124.9280 | 1.1233 | yes |
| 5120 | 17408 | 8 | (64, 64, 3, 4, 4, 'smem', 4) | 111.4720 | 125.9520 | 1.1311 | yes |
| 248320 | 5120 | 1 | (64, 128, 4, 1, 4, 'smem', 4) | 1517.0560 | 1628.6720 | 1.0736 | yes |
| 248320 | 5120 | 2 | (64, 128, 4, 1, 4, 'smem', 4) | 1518.0800 | 1527.2960 | 1.0064 | no |
| 248320 | 5120 | 3 | (64, 128, 4, 1, 4, 'smem', 4) | 1519.6160 | 1529.8560 | 1.0074 | no |
| 248320 | 5120 | 4 | (64, 128, 4, 1, 4, 'smem', 4) | 1521.1440 | 1530.3680 | 1.0057 | no |
| 248320 | 5120 | 5 | (64, 128, 4, 1, 4, 'smem', 4) | 1521.6640 | 1531.9040 | 1.0061 | no |
| 248320 | 5120 | 6 | (64, 128, 4, 1, 4, 'smem', 4) | 1521.6640 | 1531.9040 | 1.0061 | no |
| 248320 | 5120 | 7 | (64, 128, 4, 1, 4, 'smem', 4) | 1521.6640 | 1531.9040 | 1.0067 | no |
| 248320 | 5120 | 8 | (64, 128, 4, 1, 4, 'smem', 4) | 1524.7360 | 1533.4400 | 1.0054 | no |

#### qwen3.5-35b-a3b-dense

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 2048 | 1 | (16, 128, 2, 1, 1, 'reg', 4) | 34.8160 | 36.8640 | 1.0588 | yes |
| 12288 | 2048 | 2 | (64, 64, 3, 2, 4, 'smem', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12288 | 2048 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12288 | 2048 | 4 | (16, 128, 2, 1, 1, 'reg', 4) | 34.8160 | 34.8160 | 1.0005 | no |
| 12288 | 2048 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8080 | 34.8160 | 1.0000 | no |
| 12288 | 2048 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 36.8640 | 1.0170 | no |
| 12288 | 2048 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 35.3280 | 1.0147 | no |
| 12288 | 2048 | 8 | (64, 128, 3, 2, 4, 'smem', 4) | 35.7200 | 35.8400 | 1.0150 | no |
| 64 | 2048 | 1 | (16, 128, 3, 16, 1, 'smem', 4) | 5.1200 | 5.1200 | 1.0000 | no |
| 64 | 2048 | 2 | (16, 64, 3, 16, 1, 'smem', 4) | 4.6080 | 14.7760 | 3.2917 | yes |
| 64 | 2048 | 3 | (16, 128, 2, 8, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 4 | (16, 64, 3, 16, 1, 'smem', 4) | 4.9360 | 14.7360 | 3.0833 | yes |
| 64 | 2048 | 5 | (16, 64, 3, 16, 1, 'smem', 4) | 6.1440 | 15.1360 | 2.6667 | yes |
| 64 | 2048 | 6 | (16, 64, 3, 16, 1, 'smem', 4) | 6.1440 | 14.3360 | 2.3333 | yes |
| 64 | 2048 | 7 | (16, 128, 2, 8, 1, 'smem', 4) | 6.1440 | 15.3600 | 2.5061 | yes |
| 64 | 2048 | 8 | (16, 256, 2, 16, 1, 'smem', 4) | 4.0960 | 14.3360 | 3.5000 | yes |
| 12352 | 2048 | 1 | (16, 128, 2, 1, 1, 'reg', 4) | 34.8160 | 36.8640 | 1.0588 | yes |
| 12352 | 2048 | 2 | (64, 64, 2, 2, 4, 'smem', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12352 | 2048 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12352 | 2048 | 4 | (16, 128, 2, 1, 1, 'reg', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12352 | 2048 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 34.8160 | 1.0000 | no |
| 12352 | 2048 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 34.8000 | 36.8640 | 1.0593 | yes |
| 12352 | 2048 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 34.8160 | 36.8640 | 1.0588 | yes |
| 12352 | 2048 | 8 | (64, 128, 2, 4, 4, 'smem', 4) | 34.8160 | 36.3520 | 1.0313 | yes |
| 2048 | 4096 | 1 | (64, 128, 2, 16, 4, 'smem', 4) | 15.3600 | 16.3840 | 1.0714 | yes |
| 2048 | 4096 | 2 | (64, 64, 3, 16, 4, 'smem', 4) | 14.3360 | 24.5760 | 1.7143 | yes |
| 2048 | 4096 | 3 | (64, 128, 2, 16, 4, 'smem', 4) | 14.3360 | 25.6000 | 1.7170 | yes |
| 2048 | 4096 | 4 | (64, 64, 2, 16, 4, 'smem', 4) | 14.3360 | 24.5760 | 1.7143 | yes |
| 2048 | 4096 | 5 | (64, 128, 2, 16, 4, 'smem', 4) | 16.2400 | 26.6240 | 1.6395 | yes |
| 2048 | 4096 | 6 | (64, 128, 2, 16, 4, 'smem', 4) | 15.3600 | 24.5760 | 1.6596 | yes |
| 2048 | 4096 | 7 | (64, 64, 2, 16, 4, 'smem', 4) | 14.3360 | 26.6240 | 1.7857 | yes |
| 2048 | 4096 | 8 | (64, 64, 2, 16, 4, 'smem', 4) | 15.3600 | 26.6240 | 1.6696 | yes |
| 9216 | 2048 | 1 | (64, 128, 2, 8, 4, 'smem', 4) | 26.6240 | 28.6720 | 1.0769 | yes |
| 9216 | 2048 | 2 | (64, 64, 3, 4, 4, 'smem', 4) | 26.6240 | 26.6240 | 1.0000 | no |
| 9216 | 2048 | 3 | (64, 128, 2, 8, 4, 'smem', 4) | 26.6240 | 26.6240 | 1.0000 | no |
| 9216 | 2048 | 4 | (64, 64, 3, 4, 4, 'smem', 4) | 26.6240 | 26.6240 | 1.0000 | no |
| 9216 | 2048 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 28.6720 | 28.6720 | 1.0000 | no |
| 9216 | 2048 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 28.6720 | 28.6720 | 1.0000 | no |
| 9216 | 2048 | 7 | (64, 128, 2, 8, 4, 'smem', 4) | 26.6240 | 28.6720 | 1.0385 | yes |
| 9216 | 2048 | 8 | (16, 128, 2, 2, 1, 'reg', 4) | 28.6720 | 28.6720 | 1.0000 | no |
| 1024 | 2048 | 1 | (64, 128, 2, 8, 4, 'smem', 4) | 8.1120 | 8.1920 | 1.0193 | no |
| 1024 | 2048 | 2 | (64, 64, 3, 16, 4, 'smem', 4) | 7.1520 | 14.3360 | 2.0451 | yes |
| 1024 | 2048 | 3 | (64, 128, 2, 16, 4, 'reg', 4) | 8.1920 | 14.3360 | 1.7500 | yes |
| 1024 | 2048 | 4 | (64, 128, 2, 8, 4, 'smem', 4) | 8.1920 | 14.3360 | 1.8183 | yes |
| 1024 | 2048 | 5 | (64, 128, 2, 16, 4, 'reg', 4) | 8.1920 | 14.3360 | 1.7500 | yes |
| 1024 | 2048 | 6 | (64, 128, 2, 16, 4, 'reg', 4) | 8.1920 | 14.3360 | 1.7856 | yes |
| 1024 | 2048 | 7 | (64, 128, 2, 16, 4, 'reg', 4) | 8.1920 | 14.3360 | 1.7500 | yes |
| 1024 | 2048 | 8 | (64, 128, 2, 16, 4, 'reg', 4) | 8.1920 | 15.3600 | 1.8826 | yes |
| 2048 | 512 | 1 | (16, 128, 2, 4, 1, 'reg', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 2048 | 512 | 2 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 8.1920 | 1.1667 | yes |
| 2048 | 512 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 7.1680 | 1.1667 | yes |
| 2048 | 512 | 4 | (16, 128, 2, 4, 1, 'reg', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 2048 | 512 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 7.1680 | 1.0000 | no |
| 2048 | 512 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 2048 | 512 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 2048 | 512 | 8 | (64, 128, 2, 4, 4, 'smem', 4) | 6.1440 | 8.1920 | 1.1667 | yes |
| 256 | 2048 | 1 | (32, 256, 2, 8, 1, 'reg', 8) | 6.1440 | 6.1440 | 1.0000 | no |
| 256 | 2048 | 2 | (64, 64, 3, 16, 4, 'smem', 4) | 5.1200 | 15.3600 | 3.0000 | yes |
| 256 | 2048 | 3 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 256 | 2048 | 4 | (64, 128, 2, 8, 4, 'smem', 4) | 6.1280 | 6.1440 | 1.0123 | no |
| 256 | 2048 | 5 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 256 | 2048 | 6 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1440 | 8.0320 | 1.3073 | yes |
| 256 | 2048 | 7 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1440 | 8.0000 | 1.3021 | yes |
| 256 | 2048 | 8 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1440 | 8.1920 | 1.3333 | yes |
| 1 | 2048 | 1 | (16, 64, 2, 16, 1, 'smem', 4) | 4.0960 | 11.2640 | 2.5178 | yes |
| 1 | 2048 | 2 | (16, 64, 3, 16, 1, 'smem', 4) | 4.0960 | 4.0960 | 1.0000 | no |
| 1 | 2048 | 3 | (16, 128, 2, 16, 1, 'smem', 4) | 5.1200 | 6.1440 | 1.2500 | yes |
| 1 | 2048 | 4 | (16, 64, 3, 16, 1, 'smem', 4) | 4.0960 | 4.0960 | 1.0000 | no |
| 1 | 2048 | 5 | (16, 128, 2, 8, 1, 'smem', 4) | 5.1200 | 6.1440 | 1.2500 | yes |
| 1 | 2048 | 6 | (16, 128, 2, 16, 1, 'smem', 4) | 4.0960 | 4.0960 | 1.0000 | no |
| 1 | 2048 | 7 | (16, 128, 2, 16, 1, 'smem', 4) | 4.8960 | 6.1440 | 1.2385 | yes |
| 1 | 2048 | 8 | (16, 128, 2, 16, 1, 'smem', 4) | 4.0960 | 4.0960 | 1.0000 | no |
| 248320 | 2048 | 1 | (64, 128, 2, 1, 4, 'smem', 4) | 601.0880 | 626.6880 | 1.0426 | yes |
| 248320 | 2048 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 602.1120 | 608.2560 | 1.0102 | no |
| 248320 | 2048 | 3 | (128, 128, 2, 1, 8, 'smem', 4) | 604.1600 | 608.2560 | 1.0068 | no |
| 248320 | 2048 | 4 | (128, 64, 3, 1, 4, 'smem', 4) | 604.1600 | 608.2560 | 1.0068 | no |
| 248320 | 2048 | 5 | (128, 128, 2, 1, 8, 'smem', 4) | 604.1600 | 610.3040 | 1.0102 | no |
| 248320 | 2048 | 6 | (128, 128, 2, 1, 8, 'smem', 4) | 606.2080 | 610.3040 | 1.0068 | no |
| 248320 | 2048 | 7 | (128, 128, 2, 1, 8, 'smem', 4) | 606.2080 | 610.3040 | 1.0068 | no |
| 248320 | 2048 | 8 | (128, 128, 2, 1, 8, 'smem', 4) | 606.2080 | 611.3280 | 1.0084 | no |

### RTX PRO 6000

Supplied probe runs: 4; runs with successful completion markers: 4.

#### qwen3.5-9b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 4096 | 1 | (64, 128, 2, 1, 4, 'smem', 4) | 69.6000 | 71.6480 | 1.0232 | no |
| 12288 | 4096 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 70.6240 | 71.6480 | 1.0291 | no |
| 12288 | 4096 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 70.6240 | 71.6480 | 1.0147 | no |
| 12288 | 4096 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 72.4720 | 1.0286 | no |
| 12288 | 4096 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 70.9360 | 71.6480 | 1.0247 | no |
| 12288 | 4096 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 70.9280 | 72.6720 | 1.0290 | no |
| 12288 | 4096 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 72.9600 | 1.0183 | no |
| 12288 | 4096 | 8 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 72.4560 | 1.0184 | no |
| 64 | 4096 | 1 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 8.1760 | 1.3294 | yes |
| 64 | 4096 | 2 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 28.6720 | 4.6667 | yes |
| 64 | 4096 | 3 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1440 | 28.6720 | 4.6667 | yes |
| 64 | 4096 | 4 | (16, 128, 3, 32, 1, 'smem', 4) | 7.1600 | 28.6560 | 4.0492 | yes |
| 64 | 4096 | 5 | (16, 128, 2, 16, 1, 'smem', 4) | 7.6960 | 28.6400 | 3.7389 | yes |
| 64 | 4096 | 6 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 28.6400 | 4.6615 | yes |
| 64 | 4096 | 7 | (16, 64, 3, 32, 1, 'smem', 4) | 6.1120 | 28.6720 | 4.6911 | yes |
| 64 | 4096 | 8 | (16, 256, 2, 32, 1, 'smem', 4) | 6.1120 | 7.1680 | 1.2485 | yes |
| 12352 | 4096 | 1 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 71.6480 | 1.0000 | no |
| 12352 | 4096 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 72.1680 | 1.0071 | no |
| 12352 | 4096 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 72.6720 | 1.0143 | no |
| 12352 | 4096 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 73.6960 | 1.0286 | no |
| 12352 | 4096 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 73.6960 | 1.0286 | no |
| 12352 | 4096 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6480 | 73.1920 | 1.0173 | no |
| 12352 | 4096 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 72.6720 | 73.6960 | 1.0143 | no |
| 12352 | 4096 | 8 | (64, 128, 2, 1, 4, 'smem', 4) | 71.6560 | 73.6960 | 1.0211 | no |
| 4096 | 4096 | 1 | (64, 128, 3, 8, 4, 'smem', 4) | 28.6400 | 30.6880 | 1.0869 | yes |
| 4096 | 4096 | 2 | (64, 64, 2, 8, 4, 'smem', 4) | 28.6400 | 30.7120 | 1.0726 | yes |
| 4096 | 4096 | 3 | (64, 128, 2, 8, 4, 'smem', 4) | 28.6400 | 30.7040 | 1.0721 | yes |
| 4096 | 4096 | 4 | (64, 64, 2, 8, 4, 'smem', 4) | 28.6400 | 30.7200 | 1.0726 | yes |
| 4096 | 4096 | 5 | (64, 128, 2, 8, 4, 'smem', 4) | 28.6400 | 30.7200 | 1.0726 | yes |
| 4096 | 4096 | 6 | (64, 128, 2, 8, 4, 'smem', 4) | 28.6400 | 30.7200 | 1.0726 | yes |
| 4096 | 4096 | 7 | (64, 128, 3, 8, 4, 'smem', 4) | 28.6400 | 30.7040 | 1.0721 | yes |
| 4096 | 4096 | 8 | (64, 128, 2, 4, 4, 'smem', 4) | 28.4400 | 30.6880 | 1.0718 | yes |
| 10240 | 4096 | 1 | (64, 64, 2, 4, 4, 'smem', 4) | 61.4080 | 63.4880 | 1.0339 | yes |
| 10240 | 4096 | 2 | (64, 64, 2, 4, 4, 'smem', 4) | 61.4080 | 61.4080 | 1.0000 | no |
| 10240 | 4096 | 3 | (64, 128, 2, 2, 4, 'smem', 4) | 60.3840 | 61.4080 | 1.0173 | no |
| 10240 | 4096 | 4 | (64, 128, 2, 8, 4, 'smem', 4) | 60.3840 | 61.4080 | 1.0173 | no |
| 10240 | 4096 | 5 | (64, 64, 2, 4, 4, 'smem', 4) | 61.4080 | 61.4080 | 1.0000 | no |
| 10240 | 4096 | 6 | (64, 64, 2, 4, 4, 'smem', 4) | 61.4080 | 61.4080 | 1.0334 | yes |
| 10240 | 4096 | 7 | (64, 64, 2, 4, 4, 'smem', 4) | 61.4080 | 61.4080 | 1.0334 | yes |
| 10240 | 4096 | 8 | (64, 128, 2, 2, 4, 'smem', 4) | 61.4080 | 62.4320 | 1.0334 | yes |
| 24576 | 4096 | 1 | (64, 128, 4, 1, 4, 'smem', 4) | 138.2240 | 141.2800 | 1.0148 | no |
| 24576 | 4096 | 2 | (64, 128, 4, 1, 4, 'smem', 4) | 139.2320 | 137.1840 | 0.9853 | no |
| 24576 | 4096 | 3 | (64, 128, 4, 2, 4, 'smem', 4) | 139.2320 | 137.1840 | 0.9853 | no |
| 24576 | 4096 | 4 | (64, 128, 4, 1, 4, 'smem', 4) | 139.2320 | 137.1840 | 0.9853 | no |
| 24576 | 4096 | 5 | (64, 128, 4, 2, 4, 'smem', 4) | 139.2320 | 137.1840 | 0.9853 | no |
| 24576 | 4096 | 6 | (64, 128, 4, 2, 4, 'smem', 4) | 139.2640 | 137.1840 | 0.9851 | no |
| 24576 | 4096 | 7 | (64, 128, 4, 2, 4, 'smem', 4) | 141.3120 | 137.1840 | 0.9737 | no |
| 24576 | 4096 | 8 | (64, 128, 4, 2, 4, 'smem', 4) | 140.7840 | 137.1840 | 0.9745 | no |
| 4096 | 12288 | 1 | (64, 128, 2, 16, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 2 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 3 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 4 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 5 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 6 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 4096 | 12288 | 7 | (64, 128, 3, 8, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0280 | no |
| 4096 | 12288 | 8 | (64, 128, 2, 16, 4, 'smem', 4) | 73.6960 | 75.7760 | 1.0282 | no |
| 248320 | 4096 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 1327.0720 | 1380.8640 | 1.0403 | yes |
| 248320 | 4096 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 1329.1200 | 1329.1200 | 1.0000 | no |
| 248320 | 4096 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 1331.1681 | 1331.1681 | 1.0008 | no |
| 248320 | 4096 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 1331.1681 | 1333.2160 | 1.0015 | no |
| 248320 | 4096 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 1333.2160 | 1333.2160 | 1.0000 | no |
| 248320 | 4096 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 1335.2640 | 1337.3120 | 1.0023 | no |
| 248320 | 4096 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 1337.3120 | 1337.3120 | 1.0008 | no |
| 248320 | 4096 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 1337.3120 | 1339.3600 | 1.0015 | no |

#### qwen3.5-27b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 5120 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 122.8480 | 1.0526 | yes |
| 16384 | 5120 | 2 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8160 | 1.0352 | yes |
| 16384 | 5120 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8000 | 1.0351 | yes |
| 16384 | 5120 | 4 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8240 | 1.0353 | yes |
| 16384 | 5120 | 5 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 135.1360 | 1.1579 | yes |
| 16384 | 5120 | 6 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 136.1600 | 1.1579 | yes |
| 16384 | 5120 | 7 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8000 | 1.0351 | yes |
| 16384 | 5120 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 117.2160 | 120.2960 | 1.0263 | no |
| 96 | 5120 | 1 | (32, 128, 2, 32, 2, 'smem', 4) | 6.7040 | 10.2400 | 1.3342 | yes |
| 96 | 5120 | 2 | (64, 128, 2, 40, 4, 'reg', 4) | 7.1520 | 34.8160 | 4.9667 | yes |
| 96 | 5120 | 3 | (16, 64, 2, 32, 1, 'smem', 4) | 8.1600 | 34.7840 | 4.2627 | yes |
| 96 | 5120 | 4 | (32, 128, 2, 32, 2, 'smem', 4) | 6.7520 | 34.8160 | 5.1230 | yes |
| 96 | 5120 | 5 | (16, 64, 2, 32, 1, 'smem', 4) | 8.1600 | 10.2400 | 1.2549 | yes |
| 96 | 5120 | 6 | (16, 64, 2, 32, 1, 'smem', 4) | 8.1600 | 10.2400 | 1.2549 | yes |
| 96 | 5120 | 7 | (16, 64, 2, 32, 1, 'smem', 4) | 8.1760 | 10.2400 | 1.2500 | yes |
| 96 | 5120 | 8 | (16, 256, 2, 32, 1, 'smem', 4) | 6.1120 | 8.1760 | 1.3351 | yes |
| 16480 | 5120 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 122.8480 | 1.0526 | yes |
| 16480 | 5120 | 2 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8000 | 1.0351 | yes |
| 16480 | 5120 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 118.7520 | 120.8000 | 1.0175 | no |
| 16480 | 5120 | 4 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8000 | 1.0351 | yes |
| 16480 | 5120 | 5 | (64, 128, 3, 1, 4, 'smem', 4) | 116.7040 | 120.8000 | 1.0351 | yes |
| 16480 | 5120 | 6 | (64, 128, 3, 1, 4, 'smem', 4) | 118.7520 | 137.1840 | 1.1552 | yes |
| 16480 | 5120 | 7 | (64, 128, 3, 1, 4, 'smem', 4) | 118.3360 | 120.8000 | 1.0263 | no |
| 16480 | 5120 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 118.5200 | 120.8160 | 1.0262 | no |
| 5120 | 6144 | 1 | (64, 128, 2, 8, 4, 'smem', 4) | 48.3360 | 49.1520 | 1.0169 | no |
| 5120 | 6144 | 2 | (64, 128, 2, 16, 4, 'smem', 4) | 48.0960 | 49.1520 | 1.0429 | yes |
| 5120 | 6144 | 3 | (64, 128, 2, 8, 4, 'smem', 4) | 49.1200 | 49.1200 | 1.0089 | no |
| 5120 | 6144 | 4 | (64, 128, 2, 4, 4, 'smem', 4) | 49.1360 | 49.1360 | 1.0149 | no |
| 5120 | 6144 | 5 | (64, 128, 2, 8, 4, 'smem', 4) | 49.1200 | 49.1520 | 1.0007 | no |
| 5120 | 6144 | 6 | (64, 64, 2, 8, 4, 'smem', 4) | 49.1200 | 49.1520 | 1.0007 | no |
| 5120 | 6144 | 7 | (64, 64, 2, 8, 4, 'smem', 4) | 48.7200 | 49.1200 | 1.0429 | yes |
| 5120 | 6144 | 8 | (64, 64, 2, 8, 4, 'smem', 4) | 49.1200 | 49.1520 | 1.0212 | no |
| 14336 | 5120 | 1 | (64, 128, 2, 1, 4, 'smem', 4) | 104.4160 | 102.3680 | 0.9804 | no |
| 14336 | 5120 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 104.4160 | 106.4800 | 1.0196 | no |
| 14336 | 5120 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 106.4640 | 107.0880 | 1.0155 | no |
| 14336 | 5120 | 4 | (64, 128, 2, 4, 4, 'smem', 4) | 104.4160 | 106.4960 | 1.0196 | no |
| 14336 | 5120 | 5 | (64, 128, 3, 2, 4, 'smem', 4) | 104.4160 | 106.4960 | 1.0199 | no |
| 14336 | 5120 | 6 | (64, 128, 3, 2, 4, 'smem', 4) | 104.4160 | 106.4960 | 1.0199 | no |
| 14336 | 5120 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 106.4640 | 108.5440 | 1.0195 | no |
| 14336 | 5120 | 8 | (64, 128, 3, 2, 4, 'smem', 4) | 104.4160 | 106.4960 | 1.0198 | no |
| 34816 | 5120 | 1 | (64, 256, 2, 1, 4, 'smem', 4) | 241.6320 | 251.8720 | 1.0509 | yes |
| 34816 | 5120 | 2 | (64, 128, 3, 1, 4, 'smem', 4) | 239.5840 | 242.6720 | 1.0129 | no |
| 34816 | 5120 | 3 | (64, 128, 4, 1, 4, 'smem', 4) | 240.6240 | 241.6320 | 1.0025 | no |
| 34816 | 5120 | 4 | (64, 128, 4, 1, 4, 'smem', 4) | 240.6080 | 241.6560 | 1.0077 | no |
| 34816 | 5120 | 5 | (64, 128, 4, 1, 4, 'smem', 4) | 241.6320 | 241.6640 | 1.0001 | no |
| 34816 | 5120 | 6 | (64, 128, 4, 1, 4, 'smem', 4) | 241.6320 | 241.6480 | 1.0043 | no |
| 34816 | 5120 | 7 | (64, 128, 4, 1, 4, 'smem', 4) | 241.6320 | 241.6480 | 1.0001 | no |
| 34816 | 5120 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 241.6320 | 242.6560 | 1.0085 | no |
| 5120 | 17408 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 123.3760 | 124.9120 | 1.0109 | no |
| 5120 | 17408 | 2 | (64, 128, 2, 4, 4, 'smem', 4) | 124.9200 | 125.4400 | 1.0164 | no |
| 5120 | 17408 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 123.8960 | 125.9520 | 1.0165 | no |
| 5120 | 17408 | 4 | (64, 128, 2, 4, 4, 'smem', 4) | 124.9200 | 126.7680 | 1.0165 | no |
| 5120 | 17408 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 124.9280 | 126.9760 | 1.0164 | no |
| 5120 | 17408 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 124.9280 | 126.9760 | 1.0164 | no |
| 5120 | 17408 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 124.9280 | 126.9680 | 1.0164 | no |
| 5120 | 17408 | 8 | (64, 128, 2, 4, 4, 'smem', 4) | 124.7200 | 126.7760 | 1.0206 | no |
| 248320 | 5120 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 1669.0880 | 1695.2320 | 1.0150 | no |
| 248320 | 5120 | 2 | (64, 128, 3, 1, 4, 'smem', 4) | 1672.7600 | 1685.5040 | 1.0075 | no |
| 248320 | 5120 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 1675.2321 | 1687.5520 | 1.0086 | no |
| 248320 | 5120 | 4 | (64, 128, 3, 1, 4, 'smem', 4) | 1677.2799 | 1691.6240 | 1.0082 | no |
| 248320 | 5120 | 5 | (64, 128, 3, 1, 4, 'smem', 4) | 1679.3280 | 1691.6480 | 1.0073 | no |
| 248320 | 5120 | 6 | (64, 128, 3, 1, 4, 'smem', 4) | 1681.3760 | 1695.7600 | 1.0085 | no |
| 248320 | 5120 | 7 | (64, 128, 3, 1, 4, 'smem', 4) | 1683.4240 | 1697.3600 | 1.0077 | no |
| 248320 | 5120 | 8 | (64, 128, 4, 1, 4, 'smem', 4) | 1683.4240 | 1699.8399 | 1.0104 | no |

#### qwen3.5-35b-a3b-dense

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 2048 | 1 | (16, 128, 2, 1, 1, 'reg', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12288 | 2048 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12288 | 2048 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0186 | no |
| 12288 | 2048 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12288 | 2048 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12288 | 2048 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 38.6720 | 38.8800 | 1.0002 | no |
| 12288 | 2048 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12288 | 2048 | 8 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 64 | 2048 | 1 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1360 | 6.1440 | 1.0013 | no |
| 64 | 2048 | 2 | (16, 64, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 3 | (16, 128, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 4 | (16, 64, 2, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 5 | (16, 128, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 6 | (16, 128, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 7 | (16, 128, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 64 | 2048 | 8 | (16, 64, 3, 16, 1, 'smem', 4) | 6.1440 | 16.3840 | 2.6667 | yes |
| 12352 | 2048 | 1 | (16, 128, 2, 1, 1, 'reg', 4) | 38.8800 | 38.8800 | 1.0004 | no |
| 12352 | 2048 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12352 | 2048 | 3 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 0.9942 | no |
| 12352 | 2048 | 4 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12352 | 2048 | 5 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12352 | 2048 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12352 | 2048 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 12352 | 2048 | 8 | (64, 128, 2, 1, 4, 'smem', 4) | 38.8800 | 38.8800 | 1.0000 | no |
| 2048 | 4096 | 1 | (64, 128, 2, 16, 4, 'smem', 4) | 16.3520 | 18.4000 | 1.1252 | yes |
| 2048 | 4096 | 2 | (16, 128, 2, 6, 1, 'reg', 4) | 18.4000 | 18.4000 | 1.1130 | yes |
| 2048 | 4096 | 3 | (128, 256, 2, 16, 4, 'reg', 8) | 18.4000 | 18.4000 | 1.1130 | yes |
| 2048 | 4096 | 4 | (64, 64, 3, 16, 4, 'smem', 4) | 16.6640 | 19.4240 | 1.1344 | yes |
| 2048 | 4096 | 5 | (64, 64, 3, 16, 4, 'smem', 4) | 18.4000 | 18.4320 | 1.0017 | no |
| 2048 | 4096 | 6 | (64, 64, 3, 16, 4, 'smem', 4) | 17.9840 | 20.4480 | 1.1183 | yes |
| 2048 | 4096 | 7 | (128, 256, 2, 16, 4, 'reg', 8) | 18.4000 | 20.4800 | 1.1130 | yes |
| 2048 | 4096 | 8 | (64, 64, 2, 16, 4, 'smem', 4) | 17.8880 | 20.2480 | 1.1353 | yes |
| 9216 | 2048 | 1 | (64, 128, 2, 2, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 2 | (64, 64, 3, 4, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 3 | (64, 128, 2, 2, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 4 | (64, 64, 3, 4, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 5 | (64, 128, 3, 4, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 6 | (64, 128, 3, 4, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 7 | (64, 128, 3, 4, 4, 'smem', 4) | 31.7120 | 30.6880 | 1.0000 | no |
| 9216 | 2048 | 8 | (64, 128, 2, 8, 4, 'smem', 4) | 30.6880 | 30.6880 | 1.0000 | no |
| 1024 | 2048 | 1 | (128, 128, 2, 16, 8, 'reg', 4) | 8.1600 | 10.2080 | 1.1255 | yes |
| 1024 | 2048 | 2 | (64, 256, 2, 16, 4, 'smem', 4) | 9.2000 | 10.2080 | 1.2249 | yes |
| 1024 | 2048 | 3 | (64, 64, 3, 16, 4, 'smem', 4) | 10.2080 | 10.2080 | 1.1034 | yes |
| 1024 | 2048 | 4 | (64, 128, 2, 32, 4, 'smem', 4) | 9.1840 | 10.2080 | 1.2274 | yes |
| 1024 | 2048 | 5 | (64, 64, 3, 16, 4, 'smem', 4) | 10.2080 | 10.2240 | 1.0031 | no |
| 1024 | 2048 | 6 | (64, 64, 3, 16, 4, 'smem', 4) | 10.2080 | 10.2080 | 1.0425 | yes |
| 1024 | 2048 | 7 | (64, 64, 3, 16, 4, 'smem', 4) | 10.2080 | 10.2080 | 1.1034 | yes |
| 1024 | 2048 | 8 | (64, 128, 2, 16, 2, 'reg', 4) | 8.1600 | 10.2080 | 1.2510 | yes |
| 2048 | 512 | 1 | (64, 128, 2, 4, 4, 'smem', 4) | 7.1360 | 8.1600 | 1.0838 | yes |
| 2048 | 512 | 2 | (64, 256, 2, 4, 4, 'smem', 4) | 8.1600 | 8.1600 | 1.0592 | yes |
| 2048 | 512 | 3 | (64, 128, 2, 4, 4, 'smem', 4) | 8.1600 | 8.1600 | 1.0321 | yes |
| 2048 | 512 | 4 | (64, 64, 3, 4, 4, 'smem', 4) | 7.1360 | 8.1600 | 1.1669 | yes |
| 2048 | 512 | 5 | (64, 128, 2, 4, 4, 'smem', 4) | 8.1600 | 8.1600 | 1.0000 | no |
| 2048 | 512 | 6 | (64, 128, 2, 4, 4, 'smem', 4) | 7.9520 | 8.1600 | 1.0911 | yes |
| 2048 | 512 | 7 | (64, 128, 2, 4, 4, 'smem', 4) | 8.1600 | 7.9600 | 0.9755 | no |
| 2048 | 512 | 8 | (128, 128, 2, 4, 4, 'smem', 4) | 7.7440 | 8.1600 | 1.1209 | yes |
| 256 | 2048 | 1 | (64, 64, 2, 16, 4, 'smem', 4) | 7.1520 | 6.1280 | 1.0000 | no |
| 256 | 2048 | 2 | (64, 64, 3, 16, 4, 'smem', 4) | 6.1280 | 16.3840 | 2.6667 | yes |
| 256 | 2048 | 3 | (64, 128, 2, 8, 4, 'smem', 4) | 8.1600 | 16.3520 | 2.0039 | yes |
| 256 | 2048 | 4 | (64, 64, 3, 16, 4, 'smem', 4) | 6.1120 | 7.1360 | 1.1641 | yes |
| 256 | 2048 | 5 | (64, 64, 2, 16, 4, 'smem', 4) | 6.1440 | 8.1760 | 1.3307 | yes |
| 256 | 2048 | 6 | (64, 64, 2, 16, 4, 'smem', 4) | 8.1600 | 8.1920 | 1.0561 | yes |
| 256 | 2048 | 7 | (64, 64, 2, 16, 4, 'smem', 4) | 8.1760 | 8.1600 | 1.1230 | yes |
| 256 | 2048 | 8 | (64, 256, 2, 16, 4, 'smem', 4) | 6.1280 | 8.1920 | 1.3333 | yes |
| 1 | 2048 | 1 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 7.6800 | 1.2500 | yes |
| 1 | 2048 | 2 | (16, 64, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 3 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 4 | (16, 64, 3, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 5 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 6 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 7 | (16, 128, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 1 | 2048 | 8 | (16, 64, 2, 16, 1, 'smem', 4) | 6.1440 | 6.1440 | 1.0000 | no |
| 248320 | 2048 | 1 | (64, 128, 3, 1, 4, 'smem', 4) | 667.6160 | 680.9280 | 1.0195 | no |
| 248320 | 2048 | 2 | (64, 128, 2, 1, 4, 'smem', 4) | 669.6640 | 669.6800 | 1.0000 | no |
| 248320 | 2048 | 3 | (64, 128, 3, 1, 4, 'smem', 4) | 671.7120 | 671.7120 | 1.0007 | no |
| 248320 | 2048 | 4 | (64, 128, 3, 1, 4, 'smem', 4) | 671.7120 | 672.2240 | 1.0000 | no |
| 248320 | 2048 | 5 | (64, 128, 3, 1, 4, 'smem', 4) | 673.7600 | 675.8400 | 1.0031 | no |
| 248320 | 2048 | 6 | (64, 128, 2, 1, 4, 'smem', 4) | 673.7600 | 675.8240 | 1.0031 | no |
| 248320 | 2048 | 7 | (64, 128, 2, 1, 4, 'smem', 4) | 673.7600 | 675.8080 | 1.0030 | no |
| 248320 | 2048 | 8 | (64, 128, 3, 1, 4, 'smem', 4) | 675.8080 | 677.8800 | 1.0031 | no |

### GB10

Supplied probe runs: 1; runs with successful completion markers: 1.

#### qwen3.5-9b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 4096 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 479.9680 | 588.9920 | 1.2273 | yes |
| 12288 | 4096 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 478.8080 | 439.8880 | 0.9192 | no |
| 12288 | 4096 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 479.0720 | 440.6720 | 0.9206 | no |
| 12288 | 4096 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 480.0720 | 442.2880 | 0.9220 | no |
| 12288 | 4096 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 479.5840 | 442.1600 | 0.9223 | no |
| 12288 | 4096 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 479.3760 | 442.1920 | 0.9248 | no |
| 12288 | 4096 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 482.1120 | 442.3840 | 0.9201 | no |
| 12288 | 4096 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 481.4480 | 442.4400 | 0.9173 | no |
| 64 | 4096 | 1 | (16, 256, 3, 8, 1, 'smem', 4) | 8.0080 | 8.0080 | 1.0081 | no |
| 64 | 4096 | 2 | (16, 512, 3, 8, 1, 'smem', 4) | 8.0000 | 28.5600 | 3.6000 | yes |
| 64 | 4096 | 3 | (16, 256, 3, 8, 1, 'smem', 4) | 8.0000 | 28.5280 | 3.5660 | yes |
| 64 | 4096 | 4 | (16, 64, 2, 64, 1, 'reg', 2) | 7.9520 | 28.5760 | 3.6146 | yes |
| 64 | 4096 | 5 | (16, 256, 3, 8, 1, 'smem', 4) | 8.0160 | 28.6080 | 3.5669 | yes |
| 64 | 4096 | 6 | (16, 256, 3, 8, 1, 'smem', 4) | 8.0160 | 28.6240 | 3.5999 | yes |
| 64 | 4096 | 7 | (16, 256, 3, 8, 1, 'smem', 4) | 8.0000 | 28.5120 | 3.5700 | yes |
| 64 | 4096 | 8 | (16, 128, 3, 8, 1, 'smem', 4) | 8.0160 | 10.0800 | 1.2600 | yes |
| 12352 | 4096 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 495.6480 | 433.4640 | 0.8739 | no |
| 12352 | 4096 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 492.3760 | 442.9760 | 0.9000 | no |
| 12352 | 4096 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 491.5840 | 443.2320 | 0.9030 | no |
| 12352 | 4096 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 493.5760 | 443.8160 | 0.8996 | no |
| 12352 | 4096 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 493.2480 | 445.2640 | 0.9012 | no |
| 12352 | 4096 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 502.0960 | 447.7280 | 0.8997 | no |
| 12352 | 4096 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 497.4720 | 443.9520 | 0.8911 | no |
| 12352 | 4096 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 500.3120 | 444.2480 | 0.8874 | no |
| 4096 | 4096 | 1 | (64, 256, 2, 16, 2, 'reg', 8) | 168.5200 | 202.6960 | 1.2039 | yes |
| 4096 | 4096 | 2 | (64, 256, 2, 16, 2, 'reg', 8) | 168.2480 | 153.2320 | 0.9106 | no |
| 4096 | 4096 | 3 | (64, 256, 2, 16, 2, 'reg', 8) | 171.8880 | 156.9600 | 0.9160 | no |
| 4096 | 4096 | 4 | (32, 128, 2, 11, 1, 'reg', 4) | 171.3600 | 154.9760 | 0.9014 | no |
| 4096 | 4096 | 5 | (64, 256, 2, 16, 2, 'reg', 8) | 172.3680 | 155.4720 | 0.9051 | no |
| 4096 | 4096 | 6 | (64, 256, 2, 16, 2, 'reg', 8) | 169.8240 | 153.4240 | 0.9061 | no |
| 4096 | 4096 | 7 | (64, 256, 2, 16, 2, 'reg', 8) | 169.8240 | 153.4240 | 0.9079 | no |
| 4096 | 4096 | 8 | (64, 256, 2, 16, 2, 'reg', 8) | 170.7760 | 153.9520 | 0.9026 | no |
| 10240 | 4096 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 406.5440 | 491.3440 | 1.2089 | yes |
| 10240 | 4096 | 2 | (16, 128, 2, 2, 1, 'reg', 4) | 409.4160 | 367.0000 | 0.8968 | no |
| 10240 | 4096 | 3 | (16, 128, 2, 2, 1, 'reg', 4) | 409.4240 | 364.5280 | 0.8922 | no |
| 10240 | 4096 | 4 | (16, 128, 2, 2, 1, 'reg', 4) | 410.7200 | 370.3040 | 0.9038 | no |
| 10240 | 4096 | 5 | (16, 128, 2, 2, 1, 'reg', 4) | 408.9120 | 365.3920 | 0.8929 | no |
| 10240 | 4096 | 6 | (16, 128, 2, 2, 1, 'reg', 4) | 409.4400 | 366.9440 | 0.8950 | no |
| 10240 | 4096 | 7 | (16, 128, 2, 2, 1, 'reg', 4) | 410.4160 | 366.4000 | 0.8939 | no |
| 10240 | 4096 | 8 | (16, 128, 2, 2, 1, 'reg', 4) | 409.2720 | 366.1760 | 0.8937 | no |
| 24576 | 4096 | 1 | (16, 64, 2, 1, 1, 'reg', 2) | 975.0480 | 857.2400 | 0.8801 | no |
| 24576 | 4096 | 2 | (16, 64, 2, 1, 1, 'reg', 2) | 977.5680 | 876.0160 | 0.8989 | no |
| 24576 | 4096 | 3 | (16, 64, 2, 1, 1, 'reg', 2) | 974.3040 | 870.7520 | 0.8930 | no |
| 24576 | 4096 | 4 | (16, 64, 2, 1, 1, 'reg', 2) | 977.4720 | 870.7520 | 0.8917 | no |
| 24576 | 4096 | 5 | (16, 64, 2, 1, 1, 'reg', 2) | 977.7920 | 873.9840 | 0.8931 | no |
| 24576 | 4096 | 6 | (16, 64, 2, 1, 1, 'reg', 2) | 986.4480 | 884.0640 | 0.8944 | no |
| 24576 | 4096 | 7 | (16, 64, 2, 1, 1, 'reg', 2) | 986.8960 | 874.0160 | 0.8836 | no |
| 24576 | 4096 | 8 | (16, 64, 2, 1, 1, 'reg', 2) | 1002.5760 | 895.0880 | 0.8956 | no |
| 4096 | 12288 | 1 | (16, 64, 2, 5, 1, 'reg', 2) | 492.9040 | 436.3200 | 0.8857 | no |
| 4096 | 12288 | 2 | (16, 64, 2, 5, 1, 'reg', 2) | 495.7840 | 440.0960 | 0.8875 | no |
| 4096 | 12288 | 3 | (16, 64, 2, 5, 1, 'reg', 2) | 494.8800 | 439.1200 | 0.8891 | no |
| 4096 | 12288 | 4 | (16, 128, 2, 5, 1, 'reg', 4) | 498.0320 | 438.6240 | 0.8806 | no |
| 4096 | 12288 | 5 | (16, 64, 2, 5, 1, 'reg', 2) | 494.0640 | 439.3600 | 0.8898 | no |
| 4096 | 12288 | 6 | (16, 64, 2, 5, 1, 'reg', 2) | 494.1280 | 440.4160 | 0.8898 | no |
| 4096 | 12288 | 7 | (16, 64, 2, 5, 1, 'reg', 2) | 495.1200 | 438.5760 | 0.8857 | no |
| 4096 | 12288 | 8 | (32, 128, 2, 9, 1, 'reg', 4) | 528.6080 | 482.6880 | 0.8914 | no |
| 248320 | 4096 | 1 | (32, 128, 2, 1, 2, 'reg', 4) | 9873.2481 | 11812.7356 | 1.1964 | yes |
| 248320 | 4096 | 2 | (16, 64, 2, 1, 1, 'reg', 2) | 9664.0959 | 8870.1122 | 0.9088 | no |
| 248320 | 4096 | 3 | (16, 64, 2, 1, 1, 'reg', 2) | 9653.0724 | 8761.6801 | 0.9087 | no |
| 248320 | 4096 | 4 | (16, 128, 2, 1, 1, 'reg', 4) | 10022.0480 | 9162.6244 | 0.9113 | no |
| 248320 | 4096 | 5 | (16, 64, 2, 1, 1, 'reg', 2) | 9626.4639 | 8759.2001 | 0.9099 | no |
| 248320 | 4096 | 6 | (16, 64, 2, 1, 1, 'reg', 2) | 9649.2481 | 8774.5118 | 0.9093 | no |
| 248320 | 4096 | 7 | (16, 64, 2, 1, 1, 'reg', 2) | 9707.3760 | 8835.4883 | 0.9098 | no |
| 248320 | 4096 | 8 | (16, 64, 2, 1, 1, 'reg', 2) | 9724.8962 | 8889.2319 | 0.9103 | no |

#### qwen3.5-27b

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16384 | 5120 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 789.4640 | 715.3360 | 0.9061 | no |
| 16384 | 5120 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 809.6080 | 753.2480 | 0.9305 | no |
| 16384 | 5120 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 792.8960 | 746.8160 | 0.9324 | no |
| 16384 | 5120 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 792.1200 | 741.0080 | 0.9356 | no |
| 16384 | 5120 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 792.9280 | 739.6480 | 0.9322 | no |
| 16384 | 5120 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 793.9200 | 739.7760 | 0.9330 | no |
| 16384 | 5120 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 792.6240 | 739.8080 | 0.9325 | no |
| 16384 | 5120 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 796.2240 | 741.7040 | 0.9321 | no |
| 96 | 5120 | 1 | (16, 128, 3, 10, 1, 'smem', 4) | 10.0800 | 12.1120 | 1.2032 | yes |
| 96 | 5120 | 2 | (32, 128, 2, 40, 2, 'reg', 4) | 9.9840 | 36.6880 | 3.6706 | yes |
| 96 | 5120 | 3 | (16, 64, 2, 40, 1, 'reg', 2) | 10.0800 | 34.9440 | 3.5004 | yes |
| 96 | 5120 | 4 | (16, 64, 2, 40, 1, 'reg', 2) | 10.0800 | 35.8000 | 3.5430 | yes |
| 96 | 5120 | 5 | (16, 64, 2, 40, 1, 'reg', 2) | 10.0640 | 12.1280 | 1.2054 | yes |
| 96 | 5120 | 6 | (16, 64, 2, 40, 1, 'reg', 2) | 10.0800 | 12.1280 | 1.2035 | yes |
| 96 | 5120 | 7 | (16, 64, 2, 40, 1, 'reg', 2) | 10.0960 | 13.1520 | 1.3025 | yes |
| 96 | 5120 | 8 | (16, 256, 2, 20, 1, 'reg', 8) | 10.0480 | 12.1280 | 1.2054 | yes |
| 16480 | 5120 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 795.2400 | 723.6000 | 0.9074 | no |
| 16480 | 5120 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 795.7280 | 738.3760 | 0.9282 | no |
| 16480 | 5120 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 796.5760 | 739.1520 | 0.9292 | no |
| 16480 | 5120 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 797.4800 | 744.6560 | 0.9331 | no |
| 16480 | 5120 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 799.4880 | 741.2000 | 0.9289 | no |
| 16480 | 5120 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 799.5520 | 739.7920 | 0.9261 | no |
| 16480 | 5120 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 799.0240 | 743.2320 | 0.9276 | no |
| 16480 | 5120 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 805.2640 | 747.4400 | 0.9272 | no |
| 5120 | 6144 | 1 | (16, 64, 2, 4, 1, 'reg', 2) | 300.6800 | 271.6400 | 0.9042 | no |
| 5120 | 6144 | 2 | (16, 64, 2, 4, 1, 'reg', 2) | 301.4560 | 275.9120 | 0.9149 | no |
| 5120 | 6144 | 3 | (16, 64, 2, 4, 1, 'reg', 2) | 302.0160 | 275.9200 | 0.9134 | no |
| 5120 | 6144 | 4 | (16, 64, 2, 4, 1, 'reg', 2) | 303.1120 | 275.3280 | 0.9092 | no |
| 5120 | 6144 | 5 | (16, 64, 2, 4, 1, 'reg', 2) | 303.5040 | 275.2640 | 0.9072 | no |
| 5120 | 6144 | 6 | (16, 64, 2, 4, 1, 'reg', 2) | 303.4880 | 276.3040 | 0.9122 | no |
| 5120 | 6144 | 7 | (16, 64, 2, 4, 1, 'reg', 2) | 305.4720 | 276.2560 | 0.9057 | no |
| 5120 | 6144 | 8 | (16, 64, 2, 4, 1, 'reg', 2) | 305.8560 | 278.3520 | 0.9112 | no |
| 14336 | 5120 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 695.6960 | 631.6480 | 0.9097 | no |
| 14336 | 5120 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 696.4640 | 641.0480 | 0.9182 | no |
| 14336 | 5120 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 690.1280 | 634.7200 | 0.9172 | no |
| 14336 | 5120 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 710.3920 | 636.8480 | 0.9087 | no |
| 14336 | 5120 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 693.2640 | 634.9760 | 0.9156 | no |
| 14336 | 5120 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 696.2080 | 632.6880 | 0.9094 | no |
| 14336 | 5120 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 701.2640 | 638.8320 | 0.9110 | no |
| 14336 | 5120 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 702.2720 | 642.6640 | 0.9104 | no |
| 34816 | 5120 | 1 | (16, 64, 2, 1, 1, 'reg', 2) | 1722.9120 | 1522.2560 | 0.8827 | no |
| 34816 | 5120 | 2 | (16, 64, 2, 1, 1, 'reg', 2) | 1733.9760 | 1555.8320 | 0.8991 | no |
| 34816 | 5120 | 3 | (16, 64, 2, 1, 1, 'reg', 2) | 1719.6320 | 1549.1680 | 0.9010 | no |
| 34816 | 5120 | 4 | (16, 64, 2, 1, 1, 'reg', 2) | 1719.9520 | 1549.9360 | 0.9026 | no |
| 34816 | 5120 | 5 | (16, 64, 2, 1, 1, 'reg', 2) | 1717.9840 | 1542.7520 | 0.8977 | no |
| 34816 | 5120 | 6 | (16, 64, 2, 1, 1, 'reg', 2) | 1720.8640 | 1547.3120 | 0.8968 | no |
| 34816 | 5120 | 7 | (16, 64, 2, 1, 1, 'reg', 2) | 1873.1681 | 1652.9280 | 0.8856 | no |
| 34816 | 5120 | 8 | (16, 64, 2, 1, 1, 'reg', 2) | 1720.6640 | 1547.8720 | 0.8986 | no |
| 5120 | 17408 | 1 | (16, 64, 2, 4, 1, 'reg', 2) | 856.9360 | 774.5520 | 0.9022 | no |
| 5120 | 17408 | 2 | (16, 64, 2, 4, 1, 'reg', 2) | 854.2000 | 770.6400 | 0.9013 | no |
| 5120 | 17408 | 3 | (16, 64, 2, 4, 1, 'reg', 2) | 854.3680 | 769.3600 | 0.9005 | no |
| 5120 | 17408 | 4 | (16, 64, 2, 4, 1, 'reg', 2) | 855.5040 | 772.0400 | 0.9020 | no |
| 5120 | 17408 | 5 | (16, 64, 2, 4, 1, 'reg', 2) | 858.4640 | 771.4240 | 0.8987 | no |
| 5120 | 17408 | 6 | (16, 64, 2, 4, 1, 'reg', 2) | 858.5920 | 771.4080 | 0.8991 | no |
| 5120 | 17408 | 7 | (16, 64, 2, 4, 1, 'reg', 2) | 859.4720 | 775.6640 | 0.9039 | no |
| 5120 | 17408 | 8 | (16, 64, 2, 4, 1, 'reg', 2) | 862.0400 | 773.9760 | 0.8981 | no |
| 248320 | 5120 | 1 | (16, 256, 2, 1, 1, 'reg', 8) | 12448.5760 | 15251.2960 | 1.2250 | yes |
| 248320 | 5120 | 2 | (16, 64, 2, 1, 1, 'reg', 2) | 12278.8479 | 10901.3357 | 0.8881 | no |
| 248320 | 5120 | 3 | (16, 64, 2, 1, 1, 'reg', 2) | 12279.1038 | 10916.4000 | 0.8907 | no |
| 248320 | 5120 | 4 | (16, 64, 2, 1, 1, 'reg', 2) | 12261.1837 | 10924.6321 | 0.8918 | no |
| 248320 | 5120 | 5 | (16, 64, 2, 1, 1, 'reg', 2) | 12353.8399 | 11081.0399 | 0.8965 | no |
| 248320 | 5120 | 6 | (16, 64, 2, 1, 1, 'reg', 2) | 12272.2077 | 11030.3683 | 0.8927 | no |
| 248320 | 5120 | 7 | (16, 64, 2, 1, 1, 'reg', 2) | 12274.6878 | 10955.7281 | 0.8923 | no |
| 248320 | 5120 | 8 | (16, 64, 2, 1, 1, 'reg', 2) | 12481.1120 | 11163.5363 | 0.8879 | no |

#### qwen3.5-35b-a3b-dense

| N | K | M | Config | Ours us | cuBLAS us | Paired speedup | Admit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12288 | 2048 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 242.7200 | 291.1760 | 1.1986 | yes |
| 12288 | 2048 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 242.8320 | 229.7120 | 0.9438 | no |
| 12288 | 2048 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 243.5360 | 228.2400 | 0.9392 | no |
| 12288 | 2048 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 244.8720 | 228.4160 | 0.9327 | no |
| 12288 | 2048 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 247.8400 | 228.1920 | 0.9191 | no |
| 12288 | 2048 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 247.1520 | 229.2000 | 0.9235 | no |
| 12288 | 2048 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 247.1360 | 222.6080 | 0.8989 | no |
| 12288 | 2048 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 249.1040 | 224.1360 | 0.9008 | no |
| 64 | 2048 | 1 | (16, 64, 2, 32, 1, 'reg', 2) | 6.0640 | 6.0000 | 0.9937 | no |
| 64 | 2048 | 2 | (16, 128, 2, 16, 1, 'reg', 4) | 6.1760 | 16.2240 | 2.6269 | yes |
| 64 | 2048 | 3 | (16, 64, 2, 32, 1, 'reg', 2) | 6.2880 | 16.2240 | 2.5802 | yes |
| 64 | 2048 | 4 | (16, 64, 2, 32, 1, 'reg', 2) | 6.0720 | 16.2320 | 2.6719 | yes |
| 64 | 2048 | 5 | (16, 64, 2, 32, 1, 'reg', 2) | 6.3840 | 16.2400 | 2.5414 | yes |
| 64 | 2048 | 6 | (16, 64, 2, 32, 1, 'reg', 2) | 6.1760 | 16.2240 | 2.6272 | yes |
| 64 | 2048 | 7 | (16, 64, 2, 32, 1, 'reg', 2) | 6.3360 | 16.2720 | 2.5607 | yes |
| 64 | 2048 | 8 | (16, 64, 2, 32, 1, 'reg', 2) | 6.7360 | 16.2320 | 2.4954 | yes |
| 12352 | 2048 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 245.6400 | 296.8960 | 1.2085 | yes |
| 12352 | 2048 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 247.4320 | 232.5840 | 0.9433 | no |
| 12352 | 2048 | 3 | (16, 64, 2, 2, 1, 'reg', 2) | 247.1200 | 231.7440 | 0.9358 | no |
| 12352 | 2048 | 4 | (16, 64, 2, 2, 1, 'reg', 2) | 249.5120 | 227.6320 | 0.9117 | no |
| 12352 | 2048 | 5 | (16, 64, 2, 2, 1, 'reg', 2) | 248.3360 | 233.3120 | 0.9344 | no |
| 12352 | 2048 | 6 | (16, 64, 2, 2, 1, 'reg', 2) | 249.6640 | 231.2000 | 0.9242 | no |
| 12352 | 2048 | 7 | (16, 64, 2, 2, 1, 'reg', 2) | 250.3200 | 228.8800 | 0.9183 | no |
| 12352 | 2048 | 8 | (16, 64, 2, 2, 1, 'reg', 2) | 250.7040 | 224.9120 | 0.8979 | no |
| 2048 | 4096 | 1 | (64, 256, 2, 16, 2, 'reg', 8) | 86.8000 | 106.9120 | 1.2353 | yes |
| 2048 | 4096 | 2 | (16, 256, 2, 16, 1, 'reg', 8) | 87.8720 | 81.7280 | 0.9299 | no |
| 2048 | 4096 | 3 | (64, 256, 2, 16, 2, 'reg', 8) | 85.8720 | 80.7520 | 0.9364 | no |
| 2048 | 4096 | 4 | (64, 256, 2, 16, 2, 'reg', 8) | 86.4640 | 79.8720 | 0.9250 | no |
| 2048 | 4096 | 5 | (64, 256, 2, 16, 2, 'reg', 8) | 87.8880 | 81.7280 | 0.9352 | no |
| 2048 | 4096 | 6 | (64, 256, 2, 16, 2, 'reg', 8) | 86.5920 | 79.6960 | 0.9205 | no |
| 2048 | 4096 | 7 | (64, 256, 2, 16, 2, 'reg', 8) | 86.9120 | 79.7280 | 0.9176 | no |
| 2048 | 4096 | 8 | (64, 256, 2, 16, 2, 'reg', 8) | 87.8720 | 81.0160 | 0.9268 | no |
| 9216 | 2048 | 1 | (16, 64, 2, 2, 1, 'reg', 2) | 185.3280 | 220.7040 | 1.1917 | yes |
| 9216 | 2048 | 2 | (16, 64, 2, 2, 1, 'reg', 2) | 186.3520 | 172.6080 | 0.9249 | no |
| 9216 | 2048 | 3 | (16, 256, 2, 2, 1, 'reg', 8) | 189.2640 | 170.4800 | 0.9027 | no |
| 9216 | 2048 | 4 | (16, 128, 2, 2, 1, 'reg', 4) | 188.8160 | 171.3600 | 0.9080 | no |
| 9216 | 2048 | 5 | (16, 256, 2, 2, 1, 'reg', 8) | 190.3680 | 169.2800 | 0.8889 | no |
| 9216 | 2048 | 6 | (16, 256, 2, 2, 1, 'reg', 8) | 190.3040 | 169.2960 | 0.8818 | no |
| 9216 | 2048 | 7 | (16, 256, 2, 2, 1, 'reg', 8) | 193.4720 | 172.5440 | 0.8992 | no |
| 9216 | 2048 | 8 | (16, 256, 2, 2, 1, 'reg', 8) | 190.8640 | 176.2480 | 0.9247 | no |
| 1024 | 2048 | 1 | (64, 256, 2, 8, 2, 'reg', 8) | 24.4480 | 30.3680 | 1.2380 | yes |
| 1024 | 2048 | 2 | (16, 64, 2, 32, 1, 'reg', 2) | 24.4160 | 26.5760 | 1.0892 | yes |
| 1024 | 2048 | 3 | (64, 256, 2, 8, 2, 'reg', 8) | 24.5600 | 26.4320 | 1.0776 | yes |
| 1024 | 2048 | 4 | (64, 256, 2, 8, 2, 'reg', 8) | 24.4240 | 26.4560 | 1.0825 | yes |
| 1024 | 2048 | 5 | (64, 256, 2, 8, 2, 'reg', 8) | 24.7680 | 26.4480 | 1.0678 | yes |
| 1024 | 2048 | 6 | (64, 256, 2, 8, 2, 'reg', 8) | 24.4800 | 26.4640 | 1.0813 | yes |
| 1024 | 2048 | 7 | (64, 256, 2, 8, 2, 'reg', 8) | 24.6560 | 26.4640 | 1.0747 | yes |
| 1024 | 2048 | 8 | (16, 64, 2, 32, 1, 'reg', 2) | 24.9760 | 26.6880 | 1.0785 | yes |
| 2048 | 512 | 1 | (32, 128, 2, 4, 1, 'reg', 4) | 14.1840 | 14.1760 | 0.9994 | no |
| 2048 | 512 | 2 | (32, 128, 2, 4, 1, 'reg', 4) | 14.2160 | 14.4320 | 1.0169 | no |
| 2048 | 512 | 3 | (32, 128, 2, 4, 1, 'reg', 4) | 14.3360 | 14.4640 | 1.0146 | no |
| 2048 | 512 | 4 | (32, 128, 2, 4, 1, 'reg', 4) | 14.2400 | 14.1920 | 0.9966 | no |
| 2048 | 512 | 5 | (32, 128, 2, 4, 1, 'reg', 4) | 14.4480 | 14.5440 | 1.0122 | no |
| 2048 | 512 | 6 | (32, 128, 2, 4, 1, 'reg', 4) | 14.3520 | 14.4320 | 1.0055 | no |
| 2048 | 512 | 7 | (32, 128, 2, 4, 1, 'reg', 4) | 14.6080 | 14.3520 | 0.9868 | no |
| 2048 | 512 | 8 | (32, 128, 2, 4, 1, 'reg', 4) | 14.9840 | 14.4400 | 0.9962 | no |
| 256 | 2048 | 1 | (32, 128, 2, 16, 2, 'reg', 4) | 10.0720 | 10.0720 | 1.0008 | no |
| 256 | 2048 | 2 | (16, 64, 2, 32, 1, 'reg', 2) | 10.0800 | 18.2720 | 1.8127 | yes |
| 256 | 2048 | 3 | (32, 128, 2, 16, 2, 'reg', 4) | 10.0480 | 18.2560 | 1.8124 | yes |
| 256 | 2048 | 4 | (16, 64, 2, 32, 1, 'reg', 2) | 10.0800 | 12.1120 | 1.2051 | yes |
| 256 | 2048 | 5 | (32, 128, 2, 16, 2, 'reg', 4) | 10.0480 | 11.9360 | 1.1841 | yes |
| 256 | 2048 | 6 | (32, 128, 2, 16, 2, 'reg', 4) | 10.0800 | 12.1120 | 1.2035 | yes |
| 256 | 2048 | 7 | (32, 128, 2, 16, 2, 'reg', 4) | 10.0480 | 12.0960 | 1.2035 | yes |
| 256 | 2048 | 8 | (16, 64, 2, 32, 1, 'reg', 2) | 10.0480 | 12.1280 | 1.2054 | yes |
| 1 | 2048 | 1 | (16, 128, 3, 4, 1, 'smem', 4) | 5.9680 | 14.1760 | 2.3790 | yes |
| 1 | 2048 | 2 | (16, 256, 3, 4, 1, 'smem', 4) | 5.9520 | 5.9680 | 1.0027 | no |
| 1 | 2048 | 3 | (16, 128, 2, 16, 1, 'reg', 4) | 5.9520 | 6.0000 | 1.0054 | no |
| 1 | 2048 | 4 | (16, 128, 2, 16, 1, 'reg', 4) | 5.9760 | 5.9840 | 1.0096 | no |
| 1 | 2048 | 5 | (16, 128, 2, 16, 1, 'reg', 4) | 5.9840 | 5.9840 | 1.0027 | no |
| 1 | 2048 | 6 | (16, 128, 2, 16, 1, 'reg', 4) | 5.9840 | 5.9840 | 1.0000 | no |
| 1 | 2048 | 7 | (16, 128, 2, 16, 1, 'reg', 4) | 5.9680 | 6.1120 | 1.0241 | no |
| 1 | 2048 | 8 | (16, 256, 2, 8, 1, 'reg', 8) | 5.9680 | 5.9840 | 1.0054 | no |
| 248320 | 2048 | 1 | (16, 64, 2, 1, 1, 'reg', 2) | 4825.4160 | 5762.9360 | 1.1936 | yes |
| 248320 | 2048 | 2 | (16, 64, 2, 1, 1, 'reg', 2) | 4828.5679 | 4385.8720 | 0.9080 | no |
| 248320 | 2048 | 3 | (16, 64, 2, 1, 1, 'reg', 2) | 4827.4720 | 4391.9840 | 0.9108 | no |
| 248320 | 2048 | 4 | (16, 128, 2, 1, 1, 'reg', 4) | 4896.5759 | 4408.5121 | 0.9005 | no |
| 248320 | 2048 | 5 | (16, 64, 2, 1, 1, 'reg', 2) | 5134.9280 | 4538.3360 | 0.8977 | no |
| 248320 | 2048 | 6 | (16, 64, 2, 1, 1, 'reg', 2) | 4870.4801 | 4415.3600 | 0.9055 | no |
| 248320 | 2048 | 7 | (16, 64, 2, 1, 1, 'reg', 2) | 4870.4960 | 4410.3200 | 0.9056 | no |
| 248320 | 2048 | 8 | (64, 256, 2, 1, 2, 'reg', 8) | 4953.9518 | 4463.4881 | 0.9057 | no |

## Acceptance

20 interleaved pairs and 20 cuBLAS self-pairs per row; epsilon is linear p95 of abs(self ratio - 1). PASS requires correctness and paired speedup >= 1-epsilon. Previous-round rows do not qualify a newer table.

### RTX 5090

current round; 174/174 rows PASS; successful completion marker: True.

| Model | Projection | N | K | M | Provider | Kernels/call | Dispatched us | cuBLAS us | Paired speedup | epsilon | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 1 | CuTe | 1 | 63.4560 | 63.4560 | 1.0000 | 0.0073 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 2 | CuTe | 1 | 63.4720 | 65.5360 | 1.0323 | 0.0005 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 3 | cuBLAS | 1 | 65.5360 | 65.5360 | 1.0000 | 0.0005 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 4 | cuBLAS | 1 | 65.5360 | 65.5360 | 1.0000 | 0.0025 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 6 | cuBLAS | 1 | 65.5360 | 65.5360 | 1.0000 | 2.1409 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 8 | cuBLAS | 1 | 65.5360 | 65.5360 | 1.0000 | 0.6329 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 1 | CuTe | 2 | 6.1440 | 6.1440 | 1.0026 | 0.0265 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 2 | CuTe | 2 | 6.1440 | 26.6240 | 4.3333 | 0.0024 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 3 | CuTe | 2 | 6.1440 | 26.6240 | 4.3307 | 0.0776 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 4 | CuTe | 2 | 6.1440 | 26.6240 | 4.3333 | 0.0166 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 6 | CuTe | 2 | 6.1440 | 26.6240 | 4.3333 | 0.0012 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 8 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 1.3070 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 1 | cuBLAS | 1 | 65.5040 | 65.5360 | 1.0005 | 0.0354 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 2 | CuTe | 2 | 65.5040 | 65.5040 | 1.0000 | 0.0308 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 3 | CuTe | 1 | 65.5360 | 65.5360 | 1.0005 | 0.1069 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 4 | CuTe | 1 | 65.5360 | 65.5360 | 1.0000 | 0.5096 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 6 | cuBLAS | 1 | 65.5040 | 65.5360 | 1.0005 | 0.0357 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 8 | cuBLAS | 1 | 65.5040 | 65.5360 | 1.0005 | 0.0313 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 1 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0093 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 1 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0093 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 2 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0061 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 2 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0061 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 3 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0849 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 3 | CuTe | 2 | 26.6240 | 26.5920 | 0.9988 | 0.0849 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 4 | CuTe | 2 | 24.5440 | 26.6240 | 1.0847 | 0.0824 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 4 | CuTe | 2 | 24.5440 | 26.6240 | 1.0847 | 0.0824 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 6 | CuTe | 2 | 24.5760 | 26.6240 | 1.0833 | 0.0019 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 6 | CuTe | 2 | 24.5760 | 26.6240 | 1.0833 | 0.0019 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 8 | CuTe | 2 | 24.5440 | 26.5920 | 1.0834 | 0.0718 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 8 | CuTe | 2 | 24.5440 | 26.5920 | 1.0834 | 0.0718 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 1 | cuBLAS | 1 | 55.2640 | 55.2960 | 1.0006 | 0.3129 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 2 | cuBLAS | 1 | 55.2640 | 55.2960 | 1.0003 | 0.7331 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 3 | cuBLAS | 1 | 55.2960 | 55.2960 | 1.0000 | 0.0402 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 4 | cuBLAS | 1 | 55.2640 | 55.2640 | 1.0000 | 0.0404 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 6 | CuTe | 2 | 55.2640 | 55.2960 | 1.0006 | 1.3518 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 8 | CuTe | 2 | 55.2960 | 56.3040 | 1.0182 | 0.0376 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 1 | cuBLAS | 1 | 122.8800 | 122.8480 | 0.9997 | 0.0338 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 2 | cuBLAS | 1 | 122.8800 | 122.8800 | 1.0003 | 1.2153 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 3 | cuBLAS | 1 | 122.8800 | 122.8800 | 1.0000 | 0.0181 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 4 | cuBLAS | 1 | 124.9280 | 122.8800 | 0.9997 | 0.0182 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 6 | cuBLAS | 1 | 122.8480 | 122.8800 | 1.0003 | 0.0169 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 8 | cuBLAS | 1 | 123.8880 | 123.9040 | 1.0000 | 0.0178 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 1 | cuBLAS | 1 | 67.5840 | 67.5840 | 1.0000 | 0.0318 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 2 | CuTe | 2 | 65.5360 | 67.5840 | 1.0313 | 0.0318 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 3 | CuTe | 2 | 65.5040 | 75.7760 | 1.1568 | 0.0259 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 4 | CuTe | 2 | 65.5040 | 75.7760 | 1.1568 | 0.7164 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 6 | CuTe | 2 | 65.5040 | 75.7760 | 1.1568 | 0.0270 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 8 | CuTe | 2 | 65.5360 | 75.7760 | 1.1563 | 0.0270 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 1 | CuTe | 1 | 1198.0799 | 1264.6240 | 1.0547 | 0.1198 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 2 | cuBLAS | 1 | 1200.1280 | 1200.1280 | 1.0000 | 0.1841 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 3 | cuBLAS | 1 | 1202.1760 | 1200.1280 | 1.0000 | 0.0063 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 4 | cuBLAS | 1 | 1202.1760 | 1202.1760 | 1.0000 | 0.0213 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 6 | cuBLAS | 1 | 1204.2240 | 1204.2240 | 1.0000 | 0.1616 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 8 | cuBLAS | 1 | 1206.2720 | 1206.2720 | 1.0000 | 0.1086 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 1 | cuBLAS | 1 | 102.9120 | 102.4320 | 1.0000 | 0.0230 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 2 | CuTe | 1 | 104.4160 | 114.6880 | 1.0984 | 1.4003 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 3 | CuTe | 1 | 104.4160 | 114.6880 | 1.0984 | 0.0349 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 4 | CuTe | 1 | 104.4160 | 114.6880 | 1.0981 | 0.0179 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 6 | CuTe | 1 | 105.4560 | 114.6880 | 1.0979 | 0.0187 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 8 | CuTe | 1 | 106.4960 | 114.6880 | 1.0864 | 0.0179 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 1 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.3333 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 2 | CuTe | 2 | 6.1440 | 32.7680 | 5.3333 | 0.0667 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 3 | CuTe | 2 | 6.1440 | 32.7680 | 5.3333 | 0.0657 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 4 | CuTe | 2 | 6.1440 | 32.7680 | 5.3333 | 0.8037 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 6 | CuTe | 2 | 6.1440 | 8.1920 | 1.3333 | 5.2772 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 8 | CuTe | 2 | 6.1440 | 8.1920 | 1.3333 | 0.0039 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 1 | cuBLAS | 1 | 104.4480 | 104.4480 | 1.0000 | 0.6220 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 2 | CuTe | 1 | 106.4800 | 114.6880 | 1.0772 | 0.0182 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 3 | CuTe | 1 | 106.4960 | 114.6880 | 1.0769 | 0.0195 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 4 | CuTe | 1 | 106.4960 | 114.6880 | 1.0772 | 0.0182 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 6 | CuTe | 1 | 106.4640 | 114.6880 | 1.0772 | 0.0179 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 8 | CuTe | 1 | 106.4640 | 114.6880 | 1.0772 | 0.0428 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 1 | cuBLAS | 1 | 43.0080 | 43.0080 | 1.0000 | 0.0484 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 1 | cuBLAS | 1 | 43.0080 | 43.0080 | 1.0000 | 0.0484 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 2 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 6.5667 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 2 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 6.5667 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 3 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 0.0418 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 3 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 0.0418 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 4 | CuTe | 2 | 43.0080 | 47.1040 | 1.0952 | 0.0442 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 4 | CuTe | 2 | 43.0080 | 47.1040 | 1.0952 | 0.0442 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 6 | CuTe | 2 | 43.0080 | 47.1040 | 1.0961 | 0.0513 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 6 | CuTe | 2 | 43.0080 | 47.1040 | 1.0961 | 0.0513 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 8 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 0.0442 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 8 | CuTe | 2 | 42.9760 | 47.1040 | 1.0961 | 0.0442 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 1 | cuBLAS | 1 | 91.1200 | 90.0800 | 0.9998 | 0.5752 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 2 | CuTe | 1 | 94.2080 | 98.3040 | 1.0435 | 0.0003 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 3 | CuTe | 2 | 94.2080 | 98.3040 | 1.0435 | 0.0040 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 4 | CuTe | 2 | 96.2560 | 98.3040 | 1.0213 | 0.1181 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 6 | CuTe | 2 | 96.2240 | 112.6400 | 1.1704 | 0.0182 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 8 | CuTe | 2 | 96.2560 | 112.6400 | 1.1702 | 0.0185 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 1 | cuBLAS | 1 | 212.9760 | 212.9920 | 1.0000 | 0.0114 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 2 | cuBLAS | 1 | 221.1840 | 221.1840 | 1.0000 | 1.0940 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 3 | cuBLAS | 1 | 221.1840 | 221.1840 | 1.0000 | 0.4239 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 4 | cuBLAS | 1 | 221.1840 | 221.1840 | 1.0000 | 0.0097 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 6 | cuBLAS | 1 | 221.2160 | 223.2160 | 1.0001 | 0.7584 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 8 | cuBLAS | 1 | 223.2320 | 221.1840 | 1.0000 | 0.3068 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 1 | cuBLAS | 1 | 112.6400 | 112.6400 | 1.0000 | 0.0203 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 2 | CuTe | 2 | 110.5600 | 114.6880 | 1.0373 | 0.0014 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 3 | CuTe | 2 | 110.5600 | 114.6880 | 1.0373 | 0.0006 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 4 | CuTe | 2 | 110.5600 | 114.6880 | 1.0373 | 0.0099 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 6 | CuTe | 2 | 110.5920 | 124.9280 | 1.1296 | 0.0174 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 8 | CuTe | 2 | 112.6400 | 124.9280 | 1.1091 | 0.3431 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 1 | CuTe | 1 | 1515.5040 | 1628.1281 | 1.0744 | 0.0730 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 2 | cuBLAS | 1 | 1525.7601 | 1525.7601 | 1.0000 | 0.0034 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 3 | cuBLAS | 1 | 1525.7601 | 1527.8080 | 1.0013 | 0.0028 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 4 | cuBLAS | 1 | 1525.7601 | 1527.8080 | 1.0013 | 0.0063 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 6 | cuBLAS | 1 | 1529.8560 | 1529.8560 | 1.0000 | 0.0028 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 8 | cuBLAS | 1 | 1530.8640 | 1531.9040 | 1.0007 | 0.0094 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 1 | CuTe | 1 | 34.8160 | 35.7920 | 1.0280 | 0.0853 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 2 | cuBLAS | 1 | 34.8160 | 34.8160 | 1.0000 | 0.1323 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 3 | cuBLAS | 1 | 34.8160 | 34.8160 | 1.0000 | 0.0039 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 4 | cuBLAS | 1 | 34.8160 | 34.7840 | 0.9991 | 0.0299 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 6 | cuBLAS | 1 | 34.8160 | 34.7840 | 0.9991 | 0.2061 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 8 | cuBLAS | 1 | 34.8160 | 34.8160 | 1.0009 | 0.0018 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 1 | cuBLAS | 1 | 4.0960 | 4.0960 | 1.0000 | 1.7180 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 2 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.0023 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 3 | CuTe | 2 | 6.1280 | 14.3360 | 2.3394 | 0.0114 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 4 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.0092 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 6 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.0114 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 8 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.4258 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 1 | CuTe | 1 | 34.7840 | 36.8320 | 1.0589 | 0.0598 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 2 | cuBLAS | 1 | 34.8160 | 34.8160 | 1.0000 | 0.0009 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 3 | cuBLAS | 1 | 34.8160 | 34.7840 | 0.9991 | 0.0039 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 4 | cuBLAS | 1 | 34.8160 | 34.8000 | 1.0000 | 4.0817 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 6 | CuTe | 2 | 34.7840 | 34.8160 | 1.0009 | 0.0589 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 8 | CuTe | 2 | 34.7840 | 34.8160 | 1.0009 | 0.5975 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 1 | CuTe | 2 | 14.3360 | 16.3840 | 1.1429 | 0.1429 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 1 | CuTe | 2 | 14.3360 | 16.3840 | 1.1429 | 0.1429 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 2 | CuTe | 2 | 16.3840 | 24.5760 | 1.5000 | 0.0126 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 2 | CuTe | 2 | 16.3840 | 24.5760 | 1.5000 | 0.0126 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 3 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.0833 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 3 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.0833 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 4 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.3091 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 4 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.3091 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 6 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.0784 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 6 | CuTe | 2 | 14.3360 | 24.5760 | 1.7143 | 0.0784 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 8 | CuTe | 2 | 14.3520 | 24.5760 | 1.7124 | 0.5414 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 8 | CuTe | 2 | 14.3520 | 24.5760 | 1.7124 | 0.5414 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 1 | CuTe | 2 | 26.5920 | 28.6720 | 1.0782 | 0.0886 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 2 | cuBLAS | 1 | 28.6720 | 26.5920 | 0.9988 | 0.0758 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 3 | cuBLAS | 1 | 26.6240 | 26.5920 | 0.9988 | 0.0783 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 4 | cuBLAS | 1 | 26.5920 | 26.6240 | 1.0012 | 0.0782 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 6 | cuBLAS | 1 | 28.6720 | 28.6720 | 1.0000 | 0.0782 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 8 | cuBLAS | 1 | 28.6720 | 28.6720 | 1.0000 | 0.0770 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 1 | cuBLAS | 1 | 8.1920 | 8.1920 | 1.0000 | 0.0039 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 2 | CuTe | 2 | 8.1920 | 14.3360 | 1.7500 | 0.0256 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 3 | CuTe | 2 | 8.1920 | 14.3360 | 1.7500 | 0.1507 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 4 | CuTe | 2 | 8.1920 | 14.3360 | 1.7500 | 3.2850 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 6 | CuTe | 2 | 8.1920 | 14.3360 | 1.7500 | 0.0045 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 8 | CuTe | 2 | 8.1920 | 14.3360 | 1.7500 | 0.8512 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 1 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.3333 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 2 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 2.2177 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 3 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 3.7468 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 4 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.5417 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 6 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.2063 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 8 | CuTe | 2 | 6.1600 | 6.1440 | 0.9974 | 0.0265 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 1 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.3417 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 2 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.1553 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 3 | CuTe | 2 | 6.1440 | 14.3360 | 2.3333 | 0.9213 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 4 | cuBLAS | 2 | 6.1440 | 6.1440 | 1.0000 | 0.5098 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 6 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.0255 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 8 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.3333 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 1 | CuTe | 2 | 4.0960 | 10.2400 | 2.5000 | 0.1974 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 2 | cuBLAS | 1 | 4.0960 | 4.0960 | 1.0000 | 0.3070 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 3 | CuTe | 2 | 4.0960 | 6.1440 | 1.5000 | 0.0116 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 4 | cuBLAS | 1 | 4.0960 | 4.0960 | 1.0000 | 0.1780 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 6 | cuBLAS | 1 | 4.0960 | 4.0960 | 1.0000 | 0.0327 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 8 | cuBLAS | 1 | 4.0960 | 4.0960 | 1.0000 | 0.5070 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 1 | CuTe | 1 | 600.0320 | 626.6880 | 1.0444 | 0.0533 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 2 | cuBLAS | 1 | 604.1600 | 604.1600 | 1.0000 | 0.0596 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 3 | cuBLAS | 1 | 604.1600 | 604.1600 | 1.0000 | 0.0089 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 4 | cuBLAS | 1 | 606.2080 | 606.2080 | 1.0000 | 0.0102 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 6 | cuBLAS | 1 | 606.2080 | 606.2080 | 1.0000 | 0.0140 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 8 | cuBLAS | 1 | 606.2080 | 606.2080 | 1.0000 | 0.0071 | PASS |

### RTX PRO 6000

current round; 174/174 rows PASS; successful completion marker: True.

| Model | Projection | N | K | M | Provider | Kernels/call | Dispatched us | cuBLAS us | Paired speedup | epsilon | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 1 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0291 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 2 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0347 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 3 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0072 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 4 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0119 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 6 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0286 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 8 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0290 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 1 | CuTe | 2 | 6.1440 | 8.1920 | 1.3333 | 0.3333 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 2 | CuTe | 2 | 6.1440 | 28.6720 | 4.6250 | 0.0714 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 3 | CuTe | 2 | 8.1920 | 28.6720 | 3.5000 | 0.0200 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 4 | CuTe | 2 | 8.1920 | 28.6720 | 3.5000 | 0.0020 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 6 | CuTe | 2 | 8.1920 | 28.6720 | 3.5000 | 0.0192 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 8 | CuTe | 2 | 8.1920 | 8.1920 | 1.0000 | 0.0719 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 1 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0068 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 2 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0290 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 3 | cuBLAS | 1 | 71.6480 | 71.6480 | 1.0000 | 0.0290 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 4 | cuBLAS | 1 | 73.6960 | 71.6480 | 0.9965 | 0.0282 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 6 | cuBLAS | 1 | 73.1840 | 71.6480 | 1.0000 | 0.0283 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 8 | cuBLAS | 1 | 73.6960 | 73.6960 | 1.0000 | 0.0286 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 1 | CuTe | 2 | 28.6400 | 30.6880 | 1.0715 | 0.0995 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 1 | CuTe | 2 | 28.6400 | 30.6880 | 1.0715 | 0.0995 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 2 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0001 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 2 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0001 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 3 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0046 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 3 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0046 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 4 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0018 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 4 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0018 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 6 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0010 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 6 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0010 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 8 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0044 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 8 | CuTe | 2 | 28.6400 | 30.7200 | 1.0726 | 0.0044 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 1 | CuTe | 2 | 61.4080 | 63.4880 | 1.0339 | 0.0339 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 2 | cuBLAS | 1 | 61.4080 | 61.4080 | 1.0000 | 0.0339 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 3 | cuBLAS | 1 | 61.4080 | 61.4080 | 1.0000 | 0.0096 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 4 | cuBLAS | 1 | 61.4080 | 61.4080 | 1.0000 | 0.0106 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 6 | CuTe | 2 | 61.4080 | 61.4080 | 1.0000 | 0.0097 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 8 | CuTe | 2 | 61.4080 | 61.4080 | 1.0000 | 0.0078 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 1 | cuBLAS | 1 | 141.2800 | 141.2800 | 1.0000 | 0.0147 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 2 | cuBLAS | 1 | 137.1840 | 137.1840 | 1.0000 | 0.0149 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 3 | cuBLAS | 1 | 137.1840 | 137.1840 | 1.0000 | 0.0196 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 4 | cuBLAS | 1 | 137.1840 | 137.1840 | 1.0000 | 0.0004 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 6 | cuBLAS | 1 | 137.1840 | 137.1840 | 1.0000 | 0.0010 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 8 | cuBLAS | 1 | 137.1840 | 137.1840 | 1.0000 | 0.0050 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 1 | cuBLAS | 2 | 75.7760 | 75.7760 | 1.0000 | 0.0271 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 2 | cuBLAS | 2 | 75.7760 | 73.7280 | 0.9998 | 0.0278 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 3 | cuBLAS | 2 | 75.7760 | 75.7760 | 1.0000 | 0.0278 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 4 | cuBLAS | 2 | 75.7760 | 75.7760 | 1.0000 | 0.0271 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 6 | cuBLAS | 2 | 75.7760 | 75.7760 | 1.0000 | 0.0282 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 8 | cuBLAS | 2 | 75.7760 | 75.7760 | 1.0000 | 0.0278 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 1 | CuTe | 1 | 1327.0720 | 1380.3520 | 1.0401 | 0.0031 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 2 | cuBLAS | 1 | 1329.1200 | 1327.0880 | 1.0000 | 0.0015 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 3 | cuBLAS | 1 | 1329.1200 | 1329.1360 | 1.0000 | 0.0015 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 4 | cuBLAS | 1 | 1331.1681 | 1331.1681 | 1.0000 | 0.0031 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 6 | cuBLAS | 1 | 1337.3280 | 1337.3441 | 1.0000 | 0.0016 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 8 | cuBLAS | 1 | 1339.3919 | 1339.3919 | 1.0000 | 0.0031 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 1 | CuTe | 1 | 116.7040 | 122.8480 | 1.0526 | 0.0174 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 2 | CuTe | 1 | 116.7040 | 120.8320 | 1.0354 | 0.0170 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 3 | CuTe | 1 | 118.7520 | 120.8320 | 1.0175 | 0.0170 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 4 | CuTe | 1 | 118.7840 | 120.8320 | 1.0172 | 0.0049 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 6 | CuTe | 1 | 118.7840 | 135.1680 | 1.1552 | 0.0192 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 8 | cuBLAS | 1 | 120.8320 | 120.8320 | 1.0000 | 0.0175 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 1 | CuTe | 2 | 8.1920 | 10.2400 | 1.2500 | 0.0130 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 2 | CuTe | 2 | 8.1920 | 34.8160 | 4.2500 | 0.0667 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 3 | CuTe | 2 | 8.1920 | 34.8160 | 4.2500 | 0.0285 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 4 | CuTe | 2 | 8.1920 | 34.8160 | 4.2500 | 0.0588 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 6 | CuTe | 2 | 8.1600 | 10.2400 | 1.2549 | 0.0002 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 8 | CuTe | 2 | 8.1920 | 10.2400 | 1.2500 | 0.0031 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 1 | CuTe | 1 | 118.7520 | 122.8480 | 1.0345 | 0.0169 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 2 | CuTe | 1 | 118.7520 | 122.8800 | 1.0348 | 0.0172 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 3 | cuBLAS | 1 | 122.8800 | 122.6400 | 1.0000 | 0.0169 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 4 | CuTe | 1 | 118.7520 | 120.8000 | 1.0174 | 0.0170 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 6 | CuTe | 1 | 118.7520 | 135.9360 | 1.1382 | 0.0154 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 8 | cuBLAS | 1 | 120.8000 | 120.8000 | 1.0000 | 0.0170 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 1 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0027 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 1 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0027 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 2 | CuTe | 2 | 49.1200 | 49.1520 | 1.0007 | 0.0401 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 2 | CuTe | 2 | 49.1200 | 49.1520 | 1.0007 | 0.0401 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 3 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0417 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 3 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0417 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 4 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0423 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 4 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0423 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 6 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0423 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 6 | cuBLAS | 2 | 49.1520 | 49.1520 | 1.0000 | 0.0423 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 8 | cuBLAS | 2 | 51.2000 | 51.2000 | 1.0000 | 0.0417 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 8 | cuBLAS | 2 | 51.2000 | 51.2000 | 1.0000 | 0.0417 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 1 | cuBLAS | 1 | 102.3680 | 102.3680 | 1.0000 | 0.0203 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 2 | cuBLAS | 1 | 104.4480 | 104.4320 | 1.0000 | 0.0199 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 3 | cuBLAS | 1 | 105.4720 | 105.1840 | 1.0000 | 0.0199 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 4 | cuBLAS | 1 | 104.4160 | 104.4160 | 1.0000 | 0.0202 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 6 | cuBLAS | 1 | 104.4480 | 104.4320 | 1.0000 | 0.0244 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 8 | cuBLAS | 1 | 106.4960 | 106.4960 | 1.0000 | 0.0195 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 1 | CuTe | 1 | 241.6640 | 251.8720 | 1.0424 | 0.0084 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 2 | cuBLAS | 1 | 241.6640 | 242.6880 | 1.0000 | 0.0086 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 3 | cuBLAS | 1 | 243.7120 | 241.6640 | 0.9948 | 0.0085 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 4 | cuBLAS | 1 | 242.6880 | 241.6320 | 1.0000 | 0.0086 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 6 | cuBLAS | 1 | 242.6880 | 241.6640 | 1.0000 | 0.0085 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 8 | cuBLAS | 1 | 243.7120 | 243.7120 | 1.0000 | 0.0086 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 1 | cuBLAS | 1 | 122.8480 | 124.9280 | 1.0039 | 0.0169 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 2 | cuBLAS | 2 | 124.9280 | 124.9280 | 1.0000 | 0.0164 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 3 | cuBLAS | 2 | 124.9280 | 124.9280 | 1.0000 | 0.0164 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 4 | cuBLAS | 2 | 126.9760 | 124.9280 | 1.0000 | 0.0164 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 6 | cuBLAS | 2 | 126.9760 | 126.9760 | 1.0000 | 0.0164 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 8 | cuBLAS | 2 | 126.9760 | 126.9760 | 1.0000 | 0.0172 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 1 | cuBLAS | 1 | 1697.7921 | 1697.7921 | 1.0000 | 0.0024 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 2 | cuBLAS | 1 | 1687.5520 | 1687.5520 | 1.0000 | 0.0024 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 3 | cuBLAS | 1 | 1689.6000 | 1689.6000 | 1.0005 | 0.0025 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 4 | cuBLAS | 1 | 1691.6480 | 1693.6960 | 1.0012 | 0.0024 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 6 | cuBLAS | 1 | 1697.7921 | 1697.7921 | 1.0000 | 0.0025 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 8 | cuBLAS | 1 | 1699.8399 | 1701.8720 | 1.0000 | 0.0024 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 1 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0007 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 2 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0837 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 3 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0163 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 4 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0151 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 6 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0000 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 8 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0151 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 1 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.0784 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 2 | CuTe | 2 | 6.1440 | 16.3840 | 2.6667 | 0.0037 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 3 | CuTe | 2 | 6.1440 | 16.3840 | 2.6667 | 0.0323 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 4 | CuTe | 2 | 6.1440 | 16.3840 | 2.6667 | 0.0036 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 6 | CuTe | 2 | 6.1440 | 16.3840 | 2.6667 | 0.0324 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 8 | CuTe | 2 | 6.1440 | 16.3840 | 2.6667 | 0.0578 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 1 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0034 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 2 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0134 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 3 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0174 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 4 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0144 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 6 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0183 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 8 | cuBLAS | 1 | 38.8800 | 38.8800 | 1.0000 | 0.0154 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 1 | CuTe | 2 | 18.4000 | 18.4000 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 1 | CuTe | 2 | 18.4000 | 18.4000 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 2 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1021 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 2 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1021 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 3 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.0001 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 3 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.0001 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 4 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1006 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 4 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1006 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 6 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1130 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 6 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.1130 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 8 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.0241 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 8 | CuTe | 2 | 18.4000 | 20.4800 | 1.1130 | 0.0241 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 1 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0228 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 2 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0429 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 3 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0018 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 4 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0222 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 6 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0715 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 8 | cuBLAS | 1 | 30.6880 | 30.6880 | 1.0000 | 0.0192 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 1 | CuTe | 2 | 10.2400 | 8.1600 | 0.7969 | 0.2502 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 2 | CuTe | 2 | 10.2400 | 10.2400 | 1.0000 | 0.0130 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 3 | CuTe | 2 | 10.2400 | 10.2400 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 4 | CuTe | 2 | 10.2400 | 10.2400 | 1.0000 | 0.0130 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 6 | CuTe | 2 | 10.2400 | 10.2400 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 8 | CuTe | 2 | 10.2080 | 10.2400 | 1.0000 | 0.0128 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 1 | CuTe | 2 | 8.1920 | 8.1600 | 0.9980 | 0.0796 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 2 | CuTe | 2 | 8.1920 | 8.1600 | 0.9961 | 0.2472 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 3 | CuTe | 2 | 8.1920 | 8.1600 | 0.9980 | 0.3333 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 4 | CuTe | 2 | 8.1920 | 8.1760 | 0.9981 | 0.0867 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 6 | CuTe | 2 | 8.1600 | 8.1600 | 1.0000 | 0.0764 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 8 | CuTe | 2 | 8.1600 | 8.1600 | 1.0000 | 0.0976 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 1 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.3403 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 2 | CuTe | 2 | 8.1920 | 16.3840 | 2.0000 | 0.0369 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 3 | CuTe | 2 | 8.1920 | 16.3840 | 2.0000 | 0.0034 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 4 | CuTe | 2 | 8.1920 | 8.1920 | 1.0000 | 0.0039 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 6 | CuTe | 2 | 8.1920 | 8.1920 | 1.0000 | 0.0165 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 8 | CuTe | 2 | 8.1920 | 8.1920 | 1.0000 | 0.0200 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 1 | CuTe | 2 | 6.1440 | 6.1440 | 1.0000 | 0.3333 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 2 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.1010 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 3 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.1119 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 4 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.1057 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 6 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.0912 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 8 | cuBLAS | 1 | 6.1440 | 6.1440 | 1.0000 | 0.0094 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 1 | cuBLAS | 1 | 679.9360 | 679.9360 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 2 | cuBLAS | 1 | 669.6640 | 669.6960 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 3 | cuBLAS | 1 | 671.7440 | 671.7440 | 1.0000 | 0.0032 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 4 | cuBLAS | 1 | 671.7120 | 671.7120 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 6 | cuBLAS | 1 | 673.7600 | 673.7600 | 1.0000 | 0.0031 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 8 | cuBLAS | 1 | 677.3120 | 675.8080 | 1.0000 | 0.0031 | PASS |

### GB10

current round; 174/174 rows PASS; successful completion marker: True.

| Model | Projection | N | K | M | Provider | Kernels/call | Dispatched us | cuBLAS us | Paired speedup | epsilon | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 1 | CuTe | 2 | 495.4240 | 599.9520 | 1.2163 | 0.0747 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 2 | cuBLAS | 1 | 442.5280 | 442.2720 | 0.9983 | 0.0188 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 3 | cuBLAS | 1 | 443.3760 | 443.1040 | 1.0023 | 0.0185 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 4 | cuBLAS | 1 | 442.6880 | 443.3440 | 1.0002 | 0.0397 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 6 | cuBLAS | 1 | 445.3440 | 445.2480 | 1.0009 | 0.0095 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvz | 12288 | 4096 | 8 | cuBLAS | 1 | 448.8160 | 448.3360 | 0.9971 | 0.0284 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 1 | cuBLAS | 1 | 8.0000 | 8.0000 | 1.0000 | 0.3081 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 2 | CuTe | 2 | 8.1920 | 30.4800 | 3.7049 | 0.6256 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 3 | CuTe | 2 | 8.0480 | 28.9120 | 3.6000 | 0.0602 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 4 | CuTe | 2 | 8.0000 | 28.9120 | 3.6132 | 0.0658 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 6 | CuTe | 2 | 8.1440 | 28.8800 | 3.5663 | 1.1541 | PASS |
| qwen3.5-9b | gdn.in_proj_ba | 64 | 4096 | 8 | CuTe | 2 | 10.0480 | 10.0480 | 1.0000 | 0.0293 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 1 | cuBLAS | 1 | 439.1040 | 441.1200 | 1.0029 | 0.0165 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 2 | cuBLAS | 1 | 443.2000 | 443.2800 | 1.0000 | 0.0121 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 3 | cuBLAS | 1 | 444.2400 | 443.2320 | 0.9974 | 0.0140 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 4 | cuBLAS | 1 | 446.3840 | 444.7680 | 0.9979 | 0.0395 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 6 | cuBLAS | 1 | 445.2480 | 445.1840 | 1.0013 | 0.0094 | PASS |
| qwen3.5-9b | gdn.in_proj_qkvzba | 12352 | 4096 | 8 | cuBLAS | 1 | 445.2480 | 446.2720 | 1.0018 | 0.0094 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 1 | CuTe | 2 | 173.9200 | 206.1920 | 1.1826 | 0.0107 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 1 | CuTe | 2 | 173.9200 | 206.1920 | 1.1826 | 0.0107 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 2 | cuBLAS | 1 | 151.3600 | 151.3600 | 0.9999 | 0.0302 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 2 | cuBLAS | 1 | 151.3600 | 151.3600 | 0.9999 | 0.0302 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 3 | cuBLAS | 1 | 151.3120 | 151.2800 | 1.0001 | 0.0218 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 3 | cuBLAS | 1 | 151.3120 | 151.2800 | 1.0001 | 0.0218 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 4 | cuBLAS | 1 | 151.3600 | 150.7840 | 0.9990 | 0.2346 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 4 | cuBLAS | 1 | 151.3600 | 150.7840 | 0.9990 | 0.2346 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 6 | cuBLAS | 1 | 151.3600 | 151.3600 | 0.9938 | 0.0192 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 6 | cuBLAS | 1 | 151.3600 | 151.3600 | 0.9938 | 0.0192 | PASS |
| qwen3.5-9b | gdn.out_proj | 4096 | 4096 | 8 | cuBLAS | 1 | 152.3840 | 152.3680 | 1.0000 | 0.0151 | PASS |
| qwen3.5-9b | attention.o | 4096 | 4096 | 8 | cuBLAS | 1 | 152.3840 | 152.3680 | 1.0000 | 0.0151 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 1 | CuTe | 2 | 422.1600 | 500.5600 | 1.1945 | 0.0206 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 2 | cuBLAS | 1 | 368.4480 | 369.5520 | 1.0063 | 0.0119 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 3 | cuBLAS | 1 | 367.4560 | 367.4240 | 1.0007 | 0.0067 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 4 | cuBLAS | 1 | 368.4480 | 367.9360 | 0.9974 | 0.0144 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 6 | cuBLAS | 1 | 367.4240 | 368.1440 | 1.0000 | 0.0089 | PASS |
| qwen3.5-9b | attention.qkv | 10240 | 4096 | 8 | cuBLAS | 1 | 368.4800 | 367.7920 | 0.9976 | 0.0673 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 1 | cuBLAS | 1 | 866.7360 | 864.5760 | 0.9989 | 0.1827 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 2 | cuBLAS | 1 | 874.3040 | 875.3280 | 1.0024 | 0.0068 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 3 | cuBLAS | 1 | 875.3280 | 874.4000 | 0.9995 | 0.1351 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 4 | cuBLAS | 1 | 875.3440 | 876.3680 | 1.0013 | 0.0083 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 6 | cuBLAS | 1 | 875.4080 | 877.3760 | 1.0007 | 0.0047 | PASS |
| qwen3.5-9b | mlp.gate_up | 24576 | 4096 | 8 | cuBLAS | 1 | 878.4000 | 878.4000 | 1.0035 | 0.0060 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 1 | cuBLAS | 1 | 439.8720 | 440.3040 | 1.0027 | 0.0146 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 2 | cuBLAS | 1 | 456.0480 | 455.0880 | 1.0011 | 0.0533 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 3 | cuBLAS | 1 | 450.4160 | 451.9200 | 1.0013 | 0.0121 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 4 | cuBLAS | 1 | 449.3920 | 450.3680 | 1.0022 | 0.0164 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 6 | cuBLAS | 1 | 450.3680 | 449.6960 | 0.9991 | 0.0209 | PASS |
| qwen3.5-9b | mlp.down | 4096 | 12288 | 8 | cuBLAS | 1 | 451.3920 | 451.6480 | 1.0000 | 0.0226 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 1 | CuTe | 1 | 10253.6483 | 11755.5041 | 1.1466 | 0.0370 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 2 | cuBLAS | 1 | 8757.2799 | 8760.1280 | 0.9997 | 0.0051 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 3 | cuBLAS | 1 | 8766.8800 | 8766.7999 | 0.9998 | 0.0063 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 4 | cuBLAS | 1 | 8776.5760 | 8779.0718 | 1.0005 | 0.0117 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 6 | cuBLAS | 1 | 8790.2079 | 8795.9838 | 1.0009 | 0.0205 | PASS |
| qwen3.5-9b | lm_head | 248320 | 4096 | 8 | cuBLAS | 1 | 8809.3600 | 8811.3441 | 1.0007 | 0.0184 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 1 | cuBLAS | 1 | 731.7440 | 727.5680 | 0.9979 | 0.0678 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 2 | cuBLAS | 1 | 727.8880 | 727.7920 | 0.9999 | 0.0101 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 3 | cuBLAS | 1 | 728.3840 | 729.3600 | 1.0013 | 0.0121 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 4 | cuBLAS | 1 | 736.1120 | 736.0800 | 1.0002 | 0.0116 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 6 | cuBLAS | 1 | 729.9360 | 730.0000 | 1.0002 | 0.0102 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvz | 16384 | 5120 | 8 | cuBLAS | 1 | 729.9200 | 730.4000 | 1.0010 | 0.0103 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 1 | CuTe | 2 | 11.0720 | 12.0960 | 1.0925 | 0.0239 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 2 | CuTe | 2 | 10.0480 | 36.6720 | 3.6497 | 0.0103 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 3 | CuTe | 2 | 10.0640 | 36.6720 | 3.6397 | 0.0061 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 4 | CuTe | 2 | 10.0960 | 36.6880 | 3.6297 | 0.0057 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 6 | CuTe | 2 | 10.1120 | 12.0960 | 1.1933 | 0.0213 | PASS |
| qwen3.5-27b | gdn.in_proj_ba | 96 | 5120 | 8 | CuTe | 2 | 10.3840 | 12.0960 | 1.1649 | 0.0238 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 1 | cuBLAS | 1 | 731.5360 | 728.0000 | 0.9952 | 0.1670 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 2 | cuBLAS | 1 | 743.2480 | 742.2400 | 0.9982 | 0.0179 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 3 | cuBLAS | 1 | 737.0880 | 736.3200 | 0.9996 | 0.0122 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 4 | cuBLAS | 1 | 738.2400 | 741.2160 | 1.0048 | 0.0139 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 6 | cuBLAS | 1 | 739.1360 | 740.1600 | 1.0013 | 0.0106 | PASS |
| qwen3.5-27b | gdn.in_proj_qkvzba | 16480 | 5120 | 8 | cuBLAS | 1 | 743.3280 | 743.7760 | 0.9994 | 0.0100 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 1 | cuBLAS | 1 | 278.0320 | 279.5680 | 1.0037 | 0.0231 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 1 | cuBLAS | 1 | 278.0320 | 279.5680 | 1.0037 | 0.0231 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 2 | cuBLAS | 1 | 281.0080 | 281.4080 | 1.0021 | 0.0138 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 2 | cuBLAS | 1 | 281.0080 | 281.4080 | 1.0021 | 0.0138 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 3 | cuBLAS | 1 | 280.4640 | 281.4400 | 1.0010 | 0.0164 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 3 | cuBLAS | 1 | 280.4640 | 281.4400 | 1.0010 | 0.0164 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 4 | cuBLAS | 1 | 280.5120 | 280.4000 | 1.0000 | 0.0106 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 4 | cuBLAS | 1 | 280.5120 | 280.4000 | 1.0000 | 0.0106 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 6 | cuBLAS | 1 | 280.4960 | 280.3840 | 1.0000 | 0.0147 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 6 | cuBLAS | 1 | 280.4960 | 280.3840 | 1.0000 | 0.0147 | PASS |
| qwen3.5-27b | gdn.out_proj | 5120 | 6144 | 8 | cuBLAS | 1 | 282.5280 | 282.4480 | 0.9980 | 0.0184 | PASS |
| qwen3.5-27b | attention.o | 5120 | 6144 | 8 | cuBLAS | 1 | 282.5280 | 282.4480 | 0.9980 | 0.0184 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 1 | cuBLAS | 1 | 630.1120 | 631.7600 | 1.0026 | 0.1787 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 2 | cuBLAS | 1 | 634.7040 | 634.7200 | 0.9996 | 0.0096 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 3 | cuBLAS | 1 | 635.2160 | 635.7440 | 1.0015 | 0.0253 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 4 | cuBLAS | 1 | 636.9120 | 637.7600 | 0.9993 | 0.0085 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 6 | cuBLAS | 1 | 641.0240 | 639.7920 | 1.0007 | 0.0162 | PASS |
| qwen3.5-27b | attention.qkv | 14336 | 5120 | 8 | cuBLAS | 1 | 640.8320 | 639.8080 | 0.9998 | 0.0115 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 1 | cuBLAS | 1 | 1525.5680 | 1523.6320 | 0.9990 | 0.0894 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 2 | cuBLAS | 1 | 1547.6640 | 1549.1199 | 1.0003 | 0.0073 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 3 | cuBLAS | 1 | 1549.8240 | 1549.1040 | 0.9999 | 0.0214 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 4 | cuBLAS | 1 | 1550.2080 | 1549.7440 | 0.9993 | 0.0118 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 6 | cuBLAS | 1 | 1554.7520 | 1557.2960 | 1.0019 | 0.0114 | PASS |
| qwen3.5-27b | mlp.gate_up | 34816 | 5120 | 8 | cuBLAS | 1 | 1556.3040 | 1556.2880 | 1.0007 | 0.0120 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 1 | cuBLAS | 1 | 781.1360 | 780.3520 | 1.0008 | 0.1589 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 2 | cuBLAS | 1 | 779.0880 | 777.0400 | 0.9974 | 0.0172 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 3 | cuBLAS | 1 | 806.7840 | 809.5840 | 0.9995 | 0.0593 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 4 | cuBLAS | 1 | 777.5040 | 777.0560 | 0.9988 | 0.0172 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 6 | cuBLAS | 1 | 778.0480 | 778.9440 | 1.0001 | 0.0115 | PASS |
| qwen3.5-27b | mlp.down | 5120 | 17408 | 8 | cuBLAS | 1 | 779.0240 | 778.0480 | 1.0001 | 0.0093 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 1 | CuTe | 1 | 12963.3756 | 15228.7679 | 1.1756 | 0.0198 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 2 | cuBLAS | 1 | 10947.4239 | 10941.2961 | 0.9998 | 0.0043 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 3 | cuBLAS | 1 | 10953.7759 | 10950.0160 | 0.9994 | 0.0095 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 4 | cuBLAS | 1 | 10961.2479 | 10965.3759 | 1.0005 | 0.0176 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 6 | cuBLAS | 1 | 10975.0404 | 10977.7761 | 1.0004 | 0.0186 | PASS |
| qwen3.5-27b | lm_head | 248320 | 5120 | 8 | cuBLAS | 1 | 10996.5763 | 10996.2239 | 0.9998 | 0.0314 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 1 | CuTe | 2 | 251.6960 | 294.8320 | 1.1746 | 0.0149 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 2 | cuBLAS | 1 | 232.2080 | 230.2240 | 0.9951 | 0.1344 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 3 | cuBLAS | 1 | 230.2400 | 228.4000 | 0.9893 | 0.0319 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 4 | cuBLAS | 1 | 229.4240 | 228.6880 | 0.9913 | 0.0407 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 6 | cuBLAS | 1 | 230.2400 | 230.1440 | 0.9953 | 0.0349 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvz | 12288 | 2048 | 8 | cuBLAS | 1 | 223.6000 | 224.0960 | 1.0040 | 0.0146 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 1 | cuBLAS | 1 | 5.9840 | 5.9520 | 1.0000 | 0.0567 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 2 | CuTe | 2 | 8.0320 | 16.2240 | 2.0199 | 0.0266 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 3 | CuTe | 2 | 8.0000 | 16.2240 | 2.0379 | 0.0295 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 4 | CuTe | 2 | 7.8720 | 16.3520 | 2.0852 | 0.5090 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 6 | CuTe | 2 | 8.0000 | 16.4160 | 2.0478 | 0.1196 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_ba | 64 | 2048 | 8 | CuTe | 2 | 8.0320 | 16.2400 | 2.0280 | 0.0304 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 1 | CuTe | 2 | 252.2880 | 298.8320 | 1.1867 | 0.0140 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 2 | cuBLAS | 1 | 229.2160 | 228.7040 | 0.9996 | 0.0404 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 3 | cuBLAS | 1 | 229.7920 | 230.2080 | 1.0049 | 0.0371 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 4 | cuBLAS | 1 | 228.5600 | 229.2160 | 1.0068 | 0.0361 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 6 | cuBLAS | 1 | 229.7120 | 229.4080 | 1.0001 | 0.0359 | PASS |
| qwen3.5-35b-a3b-dense | gdn.in_proj_qkvzba | 12352 | 2048 | 8 | cuBLAS | 1 | 225.1200 | 225.1200 | 0.9994 | 0.0137 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 1 | CuTe | 2 | 88.0320 | 107.3440 | 1.2095 | 0.3477 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 1 | CuTe | 2 | 88.0320 | 107.3440 | 1.2095 | 0.3477 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 2 | cuBLAS | 1 | 79.6000 | 78.8800 | 0.9990 | 0.0282 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 2 | cuBLAS | 1 | 79.6000 | 78.8800 | 0.9990 | 0.0282 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 3 | cuBLAS | 1 | 79.7120 | 79.6800 | 0.9994 | 0.0495 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 3 | cuBLAS | 1 | 79.7120 | 79.6800 | 0.9994 | 0.0495 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 4 | cuBLAS | 1 | 79.6800 | 79.6800 | 1.0000 | 0.0546 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 4 | cuBLAS | 1 | 79.6800 | 79.6800 | 1.0000 | 0.0546 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 6 | cuBLAS | 1 | 79.6800 | 79.6800 | 0.9998 | 0.0537 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 6 | cuBLAS | 1 | 79.6800 | 79.6800 | 0.9998 | 0.0537 | PASS |
| qwen3.5-35b-a3b-dense | gdn.out_proj | 2048 | 4096 | 8 | cuBLAS | 1 | 79.6960 | 79.7120 | 1.0000 | 0.0257 | PASS |
| qwen3.5-35b-a3b-dense | attention.o | 2048 | 4096 | 8 | cuBLAS | 1 | 79.6960 | 79.7120 | 1.0000 | 0.0257 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 1 | CuTe | 2 | 191.8240 | 226.1120 | 1.1813 | 0.0703 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 2 | cuBLAS | 1 | 173.3920 | 172.3520 | 0.9993 | 0.0353 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 3 | cuBLAS | 1 | 170.8160 | 170.8160 | 1.0000 | 0.0364 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 4 | cuBLAS | 1 | 171.8720 | 172.4160 | 1.0055 | 0.0312 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 6 | cuBLAS | 1 | 171.3440 | 170.9760 | 1.0009 | 0.0329 | PASS |
| qwen3.5-35b-a3b-dense | attention.qkv | 9216 | 2048 | 8 | cuBLAS | 1 | 171.4240 | 171.8400 | 1.0032 | 0.0301 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 1 | CuTe | 2 | 24.3840 | 30.5280 | 1.2520 | 0.0609 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 2 | CuTe | 2 | 24.5760 | 26.4320 | 1.0694 | 0.0813 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 3 | CuTe | 2 | 24.3840 | 26.4320 | 1.0840 | 0.0814 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 4 | CuTe | 2 | 24.3840 | 26.4320 | 1.0839 | 0.1530 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 6 | CuTe | 2 | 24.3840 | 26.4320 | 1.0846 | 0.0819 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate_up | 1024 | 2048 | 8 | CuTe | 2 | 26.4320 | 26.4320 | 1.0012 | 0.0773 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 1 | cuBLAS | 1 | 14.1440 | 14.1440 | 1.0000 | 0.0267 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 2 | CuTe | 2 | 14.2720 | 16.1920 | 1.1337 | 0.1259 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 3 | CuTe | 2 | 14.4160 | 14.5600 | 1.0225 | 0.1270 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 4 | cuBLAS | 1 | 16.1920 | 16.1920 | 1.0000 | 0.8077 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 6 | cuBLAS | 1 | 16.1920 | 16.1920 | 1.0000 | 0.1333 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.down | 2048 | 512 | 8 | cuBLAS | 1 | 16.2240 | 16.2240 | 1.0000 | 0.1090 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 1 | cuBLAS | 1 | 10.0640 | 10.0800 | 1.0000 | 0.0319 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 2 | CuTe | 2 | 10.0960 | 18.2720 | 1.8098 | 0.0093 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 3 | CuTe | 2 | 10.0480 | 18.2400 | 1.8153 | 0.0096 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 4 | CuTe | 2 | 10.1120 | 12.0960 | 1.1959 | 0.0212 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 6 | CuTe | 2 | 10.0480 | 12.0960 | 1.2032 | 0.0238 | PASS |
| qwen3.5-35b-a3b-dense | router.gate | 256 | 2048 | 8 | CuTe | 2 | 10.0480 | 12.0960 | 1.2038 | 0.0111 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 1 | CuTe | 2 | 5.9840 | 14.1440 | 2.3511 | 0.0291 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 2 | cuBLAS | 1 | 5.9520 | 5.9520 | 1.0000 | 0.0698 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 3 | cuBLAS | 2 | 6.1760 | 6.0960 | 0.9922 | 0.0656 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 4 | cuBLAS | 1 | 5.9520 | 5.9520 | 1.0000 | 1.2939 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 6 | cuBLAS | 1 | 5.9520 | 5.9520 | 1.0000 | 0.0599 | PASS |
| qwen3.5-35b-a3b-dense | shared_expert.gate | 1 | 2048 | 8 | cuBLAS | 1 | 5.9520 | 5.9520 | 1.0000 | 0.0478 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 1 | CuTe | 1 | 4986.1920 | 5822.9282 | 1.1679 | 0.0471 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 2 | cuBLAS | 1 | 4383.2159 | 4386.1120 | 1.0011 | 0.0043 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 3 | cuBLAS | 1 | 4399.4560 | 4398.4001 | 1.0007 | 0.0052 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 4 | cuBLAS | 1 | 4405.1359 | 4398.9439 | 0.9991 | 0.0137 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 6 | cuBLAS | 1 | 4419.4081 | 4419.5039 | 1.0015 | 0.0060 | PASS |
| qwen3.5-35b-a3b-dense | lm_head | 248320 | 2048 | 8 | cuBLAS | 1 | 4423.5201 | 4427.6321 | 1.0002 | 0.0092 | PASS |

## Decode-step GEMM estimates

Sum of isolated median GEMM times times calls/step; excludes attention, recurrent kernels, scheduling and communication. Fused and split GDN layouts are alternatives. Delta = step_ms(B) - step_ms(1). This is not measured engine latency.

### RTX 5090

current round.

| Model | Layout | B | Calls/step | Dispatched ms | cuBLAS ms | Dispatched delta ms | cuBLAS delta ms | All rows PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | fused | 1 | 129 | 10.1591 | 10.2246 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | fused | 2 | 129 | 10.0956 | 10.1604 | -0.0635 | -0.0642 | True |
| qwen3.5-9b | fused | 3 | 129 | 10.0977 | 10.4233 | -0.0614 | 0.1987 | True |
| qwen3.5-9b | fused | 4 | 129 | 10.0964 | 10.4261 | -0.0627 | 0.2015 | True |
| qwen3.5-9b | fused | 6 | 129 | 10.0321 | 10.4284 | -0.1270 | 0.2038 | True |
| qwen3.5-9b | fused | 8 | 129 | 10.0677 | 10.4703 | -0.0914 | 0.2456 | True |
| qwen3.5-9b | split | 1 | 153 | 10.2574 | 10.3222 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | split | 2 | 153 | 10.1943 | 10.8001 | -0.0631 | 0.4780 | True |
| qwen3.5-9b | split | 3 | 153 | 10.2451 | 11.0623 | -0.0123 | 0.7401 | True |
| qwen3.5-9b | split | 4 | 153 | 10.2438 | 11.0651 | -0.0136 | 0.7429 | True |
| qwen3.5-9b | split | 6 | 153 | 10.1804 | 11.0674 | -0.0771 | 0.7452 | True |
| qwen3.5-9b | split | 8 | 153 | 10.2159 | 10.6177 | -0.0415 | 0.2956 | True |
| qwen3.5-27b | fused | 1 | 257 | 31.5789 | 31.6759 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | fused | 2 | 257 | 32.1262 | 33.1141 | 0.5473 | 1.4382 | True |
| qwen3.5-27b | fused | 3 | 257 | 32.1270 | 33.1162 | 0.5481 | 1.4403 | True |
| qwen3.5-27b | fused | 4 | 257 | 32.1618 | 33.1162 | 0.5829 | 1.4403 | True |
| qwen3.5-27b | fused | 6 | 257 | 32.1679 | 34.1330 | 0.5891 | 2.4571 | True |
| qwen3.5-27b | fused | 8 | 257 | 32.4275 | 34.0050 | 0.8486 | 2.3291 | True |
| qwen3.5-27b | split | 1 | 305 | 31.8000 | 31.8740 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | split | 2 | 305 | 32.3220 | 34.6870 | 0.5220 | 2.8130 | True |
| qwen3.5-27b | split | 3 | 305 | 32.3220 | 34.6890 | 0.5220 | 2.8150 | True |
| qwen3.5-27b | split | 4 | 305 | 32.3569 | 34.6890 | 0.5568 | 2.8150 | True |
| qwen3.5-27b | split | 6 | 305 | 32.4145 | 34.5262 | 0.6144 | 2.6522 | True |
| qwen3.5-27b | split | 8 | 305 | 32.7240 | 34.3982 | 0.9239 | 2.5242 | True |
| qwen3.5-35b-a3b-dense | fused | 1 | 241 | 3.4660 | 3.9025 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | fused | 2 | 241 | 3.5738 | 4.4541 | 0.1078 | 0.5516 | True |
| qwen3.5-35b-a3b-dense | fused | 3 | 241 | 3.4714 | 4.5350 | 0.0054 | 0.6325 | True |
| qwen3.5-35b-a3b-dense | fused | 4 | 241 | 3.4731 | 4.1283 | 0.0071 | 0.2258 | True |
| qwen3.5-35b-a3b-dense | fused | 6 | 241 | 3.4929 | 4.1492 | 0.0270 | 0.2467 | True |
| qwen3.5-35b-a3b-dense | fused | 8 | 241 | 3.4942 | 4.1492 | 0.0283 | 0.2467 | True |
| qwen3.5-35b-a3b-dense | split | 1 | 271 | 3.5898 | 3.9942 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | split | 2 | 271 | 3.7581 | 4.8842 | 0.1683 | 0.8900 | True |
| qwen3.5-35b-a3b-dense | split | 3 | 271 | 3.6552 | 4.9661 | 0.0654 | 0.9719 | True |
| qwen3.5-35b-a3b-dense | split | 4 | 271 | 3.6574 | 4.5579 | 0.0676 | 0.5637 | True |
| qwen3.5-35b-a3b-dense | split | 6 | 271 | 3.6782 | 4.5784 | 0.0884 | 0.5842 | True |
| qwen3.5-35b-a3b-dense | split | 8 | 271 | 3.6795 | 4.5793 | 0.0897 | 0.5851 | True |

### RTX PRO 6000

current round.

| Model | Layout | B | Calls/step | Dispatched ms | cuBLAS ms | Dispatched delta ms | cuBLAS delta ms | All rows PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | fused | 1 | 129 | 11.4002 | 11.5356 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | fused | 2 | 129 | 11.2711 | 11.2701 | -0.1290 | -0.2655 | True |
| qwen3.5-9b | fused | 3 | 129 | 11.2711 | 11.3377 | -0.1290 | -0.1979 | True |
| qwen3.5-9b | fused | 4 | 129 | 11.3223 | 11.3397 | -0.0778 | -0.1959 | True |
| qwen3.5-9b | fused | 6 | 129 | 11.3162 | 11.3459 | -0.0840 | -0.1897 | True |
| qwen3.5-9b | fused | 8 | 129 | 11.3306 | 11.3971 | -0.0696 | -0.1385 | True |
| qwen3.5-9b | split | 1 | 153 | 11.5476 | 11.7322 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | split | 2 | 153 | 11.4186 | 11.9583 | -0.1290 | 0.2260 | True |
| qwen3.5-9b | split | 3 | 153 | 11.4677 | 12.0258 | -0.0799 | 0.2936 | True |
| qwen3.5-9b | split | 4 | 153 | 11.4698 | 12.0279 | -0.0778 | 0.2956 | True |
| qwen3.5-9b | split | 6 | 153 | 11.4760 | 12.0340 | -0.0717 | 0.3018 | True |
| qwen3.5-9b | split | 8 | 153 | 11.4780 | 11.5446 | -0.0696 | -0.1876 | True |
| qwen3.5-27b | fused | 1 | 257 | 35.5103 | 36.4933 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | fused | 2 | 257 | 35.6644 | 35.9299 | 0.1541 | -0.5635 | True |
| qwen3.5-27b | fused | 3 | 257 | 36.0141 | 35.8669 | 0.5038 | -0.6264 | True |
| qwen3.5-27b | fused | 4 | 257 | 35.8666 | 35.7683 | 0.3564 | -0.7250 | True |
| qwen3.5-27b | fused | 6 | 257 | 35.8733 | 36.6323 | 0.3630 | 0.1390 | True |
| qwen3.5-27b | fused | 8 | 257 | 36.2030 | 36.2050 | 0.6927 | -0.2883 | True |
| qwen3.5-27b | split | 1 | 305 | 35.8052 | 36.9848 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | split | 2 | 305 | 35.9593 | 37.5027 | 0.1541 | 0.5179 | True |
| qwen3.5-27b | split | 3 | 305 | 36.2092 | 37.4513 | 0.4040 | 0.4664 | True |
| qwen3.5-27b | split | 4 | 305 | 36.2614 | 37.4410 | 0.4562 | 0.4562 | True |
| qwen3.5-27b | split | 6 | 305 | 36.2665 | 37.0870 | 0.4613 | 0.1021 | True |
| qwen3.5-27b | split | 8 | 305 | 36.5978 | 36.6981 | 0.7926 | -0.2867 | True |
| qwen3.5-35b-a3b-dense | fused | 1 | 241 | 4.1180 | 4.0335 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | fused | 2 | 241 | 4.1897 | 4.5993 | 0.0716 | 0.5658 | True |
| qwen3.5-35b-a3b-dense | fused | 3 | 241 | 4.1917 | 4.6013 | 0.0737 | 0.5678 | True |
| qwen3.5-35b-a3b-dense | fused | 4 | 241 | 4.1917 | 4.2743 | 0.0737 | 0.2407 | True |
| qwen3.5-35b-a3b-dense | fused | 6 | 241 | 4.1925 | 4.2757 | 0.0745 | 0.2421 | True |
| qwen3.5-35b-a3b-dense | fused | 8 | 241 | 4.1948 | 4.2777 | 0.0767 | 0.2442 | True |
| qwen3.5-35b-a3b-dense | split | 1 | 271 | 4.3023 | 4.2179 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | split | 2 | 271 | 4.3740 | 5.0908 | 0.0716 | 0.8730 | True |
| qwen3.5-35b-a3b-dense | split | 3 | 271 | 4.3761 | 5.0929 | 0.0737 | 0.8750 | True |
| qwen3.5-35b-a3b-dense | split | 4 | 271 | 4.3760 | 4.7658 | 0.0737 | 0.5479 | True |
| qwen3.5-35b-a3b-dense | split | 6 | 271 | 4.3768 | 4.7672 | 0.0745 | 0.5493 | True |
| qwen3.5-35b-a3b-dense | split | 8 | 271 | 4.3791 | 4.7692 | 0.0767 | 0.5514 | True |

### GB10

current round.

| Model | Layout | B | Calls/step | Dispatched ms | cuBLAS ms | Dispatched delta ms | cuBLAS delta ms | All rows PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-9b | fused | 1 | 129 | 71.5463 | 74.7012 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | fused | 2 | 129 | 69.7564 | 69.7721 | -1.7899 | -4.9291 | True |
| qwen3.5-9b | fused | 3 | 129 | 69.6341 | 69.6270 | -1.9122 | -5.0742 | True |
| qwen3.5-9b | fused | 4 | 129 | 69.6724 | 69.6776 | -1.8739 | -5.0235 | True |
| qwen3.5-9b | fused | 6 | 129 | 69.6839 | 69.7354 | -1.8624 | -4.9658 | True |
| qwen3.5-9b | fused | 8 | 129 | 69.8728 | 69.9015 | -1.6735 | -4.7996 | True |
| qwen3.5-9b | split | 1 | 153 | 73.0900 | 78.7051 | 0.0000 | 0.0000 | True |
| qwen3.5-9b | split | 2 | 153 | 69.9369 | 70.4794 | -3.1531 | -8.2257 | True |
| qwen3.5-9b | split | 3 | 153 | 69.8065 | 70.3178 | -3.2835 | -8.3874 | True |
| qwen3.5-9b | split | 4 | 153 | 69.7757 | 70.3373 | -3.3143 | -8.3678 | True |
| qwen3.5-9b | split | 6 | 153 | 69.8817 | 70.4300 | -3.2083 | -8.2751 | True |
| qwen3.5-9b | split | 8 | 153 | 70.1996 | 70.1922 | -2.8904 | -8.5129 | True |
| qwen3.5-27b | fused | 1 | 257 | 223.5820 | 225.6283 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | fused | 2 | 257 | 223.6752 | 223.6087 | 0.0932 | -2.0196 | True |
| qwen3.5-27b | fused | 3 | 257 | 225.2701 | 225.4335 | 1.6881 | -0.1948 | True |
| qwen3.5-27b | fused | 4 | 257 | 223.5137 | 223.6087 | -0.0683 | -2.0196 | True |
| qwen3.5-27b | fused | 6 | 257 | 223.9609 | 224.2061 | 0.3789 | -1.4222 | True |
| qwen3.5-27b | fused | 8 | 257 | 224.4724 | 224.4086 | 0.8904 | -1.2197 | True |
| qwen3.5-27b | split | 1 | 305 | 224.1234 | 226.1881 | 0.0000 | 0.0000 | True |
| qwen3.5-27b | split | 2 | 305 | 223.4203 | 224.6754 | -0.7032 | -1.5127 | True |
| qwen3.5-27b | split | 3 | 305 | 225.3353 | 226.8596 | 1.2119 | 0.6715 | True |
| qwen3.5-27b | split | 4 | 305 | 223.8962 | 225.1232 | -0.2273 | -1.0649 | True |
| qwen3.5-27b | split | 6 | 305 | 224.0047 | 224.2990 | -0.1188 | -1.8891 | True |
| qwen3.5-27b | split | 8 | 305 | 224.3273 | 224.3471 | 0.2038 | -1.8410 | True |
| qwen3.5-35b-a3b-dense | fused | 1 | 241 | 20.1774 | 24.0986 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | fused | 2 | 241 | 18.3735 | 18.7999 | -1.8039 | -5.2987 | True |
| qwen3.5-35b-a3b-dense | fused | 3 | 241 | 18.3908 | 18.8131 | -1.7866 | -5.2855 | True |
| qwen3.5-35b-a3b-dense | fused | 4 | 241 | 18.4335 | 18.6137 | -1.7439 | -5.4849 | True |
| qwen3.5-35b-a3b-dense | fused | 6 | 241 | 18.4744 | 18.6256 | -1.7029 | -5.4730 | True |
| qwen3.5-35b-a3b-dense | fused | 8 | 241 | 18.4254 | 18.5163 | -1.7520 | -5.5823 | True |
| qwen3.5-35b-a3b-dense | split | 1 | 271 | 20.3392 | 24.1572 | 0.0000 | 0.0000 | True |
| qwen3.5-35b-a3b-dense | split | 2 | 271 | 18.7042 | 19.3322 | -1.6350 | -4.8250 | True |
| qwen3.5-35b-a3b-dense | split | 3 | 271 | 18.6443 | 19.2456 | -1.6949 | -4.9116 | True |
| qwen3.5-35b-a3b-dense | split | 4 | 271 | 18.6955 | 19.0884 | -1.6436 | -5.0688 | True |
| qwen3.5-35b-a3b-dense | split | 6 | 271 | 18.7303 | 19.1401 | -1.6089 | -5.0170 | True |
| qwen3.5-35b-a3b-dense | split | 8 | 271 | 18.6208 | 18.9728 | -1.7184 | -5.1844 | True |
