# Frozen BF16 small-M linear measurements

Generated from supplied RESULT logs; no GPU work was run to produce this report.
Missing measurements remain pending. Error rows are retained as errors, never
replaced by inferred timings.

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
