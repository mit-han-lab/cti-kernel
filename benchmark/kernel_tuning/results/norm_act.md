<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors -->

# Norm/activation paired qualification

Rows C3 (SiLU x mul), C4 (Qwen3 residual add + RMSNorm), C5 (Qwen3.5 zero-centered
prefill residual norm) and D2 (aux-hidden `sum_out`) against the kernels vLLM 0.31.0 and
SGLang 72730d6 actually dispatch, paired in one call on the same GPU. Ours is the
shipped default (`static_weight=False`, PDL-safe weight loads after the wait); the
`static` column is `AddRMSNorm(static_weight=True)`.

FlashInfer provider support is restricted to the pin `0.7.0.post1` and a successful
import. Standard weights and activation auto run on our kernels without FlashInfer.
Auto reroutes retain a `.reason` on `.provider` (standalone norms use
`last_standalone_provider`); `CTI_ENGINE_CHECKS` rejects reroutes before launch.
Inference norm weights must be frozen: `requires_grad=True` raises `ValueError`
in both grad modes. Historical comparisons below retain their measured versions.

## Activation dispatch (NA-ACT)

Auto uses accurate CuTe for SiLU, erf GELU and tanh GELU in FP16/BF16/FP32,
for every width >= 1 on capability 12.x (sm_120/121) with DSL 4.8.0;
outside 12.x auto uses FlashInfer, then a visible Triton/torch reroute. On capability major 12,
contiguous 16-byte-aligned SiLU FP16/BF16 table hits select the tuned kernel;
unknown SM counts and all other cases select general CuTe. Device classes select
speed only. Existing table configurations now set `fast=False` (IEEE `div.rn`);
the historical timing receipts below predate that precision change.

Both kernels share negative-tail-safe SiLU arithmetic and round once on output.
The general kernel uses runtime Int64 row strides and selects the largest legal
power-of-two vector up to 16 bytes by width, strides and pointer alignment.
N-D leading dimensions collapse to rows without copies where possible. Other
layouts visibly pack/copy; `.provider.reason` explains the reroute and
`CTI_ENGINE_CHECKS` rejects it. ReLU2 uses the same native-stride general kernel.
Zero rows and width zero do not launch. `.dispatched_config` is a tuned Config
or `cute_general_nN`; both kernels report `.provider == "cute"`.

On capability 12.x FlashInfer runs only when asked for. Outside 12.x, auto takes
FlashInfer for FP16/BF16 inputs it can take; otherwise SiLU visibly reroutes to
Triton and GELU kinds to the torch pointwise form (reason logged once, rejected
under `CTI_ENGINE_CHECKS`). Explicit `provider="triton"` is honored; forcing CuTe
outside 12.x raises (unqualified there). On 12.x without the DSL, FP16/BF16 SiLU
reroutes to Triton and GELU kinds to FlashInfer; FP32 raises. Forced CuTe configs retain tuned-kernel
admission checks.
GELU uses `g * erfc(-g/sqrt(2))/2` and tanh GELU uses the identity
`(1 + tanh(z))/2 = 1/(1 + exp(-2z))`: the same functions without the cancellation of
`1 + erf`/`1 + tanh` in the negative tail, where the FP32 composition (torch eager,
Inductor's forward_native) returns zero for representable results.

NA-ACT qualification (`tests/gpu/test_activation.py`, `test_na_*`): an FP64 oracle per
element (tail-accurate forms) over widths {8, 16, 24, 1000, 3072, 4352, 6144, 8704,
12288, 17408, 25600} x M {0, 1-5, 16, 64, 65, 512, 4096, 16384} x FP16/BF16/FP32 x
silu/gelu/gelu_tanh x contiguous/row-strided buffers (core subset in CI, the full grid
`exhaustive`), odd widths 1-9, N-D layouts, an int64-indexed case (x numel > 2^31),
graph capture after warm_up, and tuned-vs-general bitwise SiLU. 16-bit outputs must be
within 1 ulp of FP64 wherever the FP32 composition is; FP32 outputs within the
composition's worst element (or 8 ulp); and the exact-rounding rate must not be worse
than the composition's (one-sided exact-binomial McNemar, p < 0.01). The two rows the
10-17 randomized rounds failed on FlashInfer (silu BF16 D=17408, gelu_tanh BF16 D=12288) are
also compared against a real `torch.compile`d forward_native. All of it passes on the
RTX 5090, RTX PRO 6000 and GB10. Measured on one 512 x 1000 draw (GB10), the share of
16-bit outputs equal to the rounded FP64 value: silu 0.99995 (FP16) / 0.999998 (BF16),
equal to the composition; gelu 0.99988 / 0.99996 against 0.947 / 0.932; gelu_tanh
0.99986 / 0.99966 against 0.955 / 0.927.

NA-ACT speed, reported only (BF16 SiLU, M=1-4, paired in one process, hot and cold L2,
20 rounds; `norm_act_paired --op silu_mul`). The ratio is the faster market time over
ours: SGLang 72730d6's JIT `act_and_mul` everywhere, and vLLM 0.31.0's captured Inductor
kernel at width 12288 (the only width with a captured vLLM kernel; it was the faster
market there). Width 12288 runs the tuned table entries, now with IEEE division; every
other width runs the general kernel (no PDL, both SiLU forms per element), which sits at
a ~1.3-1.8 µs floor against ~0.8-1.2 µs for SGLang. Not chased under the current scope.

| Width (kernel) | RTX 5090 ours / SGLang µs, M=1 hot | RTX 5090 ratio hot / cold, M=1-4 | RTX PRO 6000 ours / SGLang µs, M=1 hot | RTX PRO 6000 ratio hot / cold, M=1-4 | GB10 ours / SGLang µs, M=1 hot | GB10 ratio hot / cold, M=1-4 |
|---|---|---|---|---|---|---|
| 12288 (tuned) | 0.71 / 0.90 | 1.17-1.30 / 1.14-1.17 | 0.80 / 1.20 | 1.31-1.46 / 1.35-1.43 | 0.72 / 1.10 | 1.14-1.33 / 1.19-1.38 |
| 6144 (general) | 1.36 / 0.83 | 0.61-0.66 / 0.75-0.76 | 1.79 / 1.16 | 0.65-0.67 / 0.75-0.79 | 1.65 / 1.09 | 0.66-0.67 / 0.74-0.78 |
| 3072 (general) | 1.34 / 0.79 | 0.59-0.67 / 0.75-0.77 | 1.74 / 1.03 | 0.60-0.67 / 0.75-0.79 | 1.61 / 1.00 | 0.62-0.68 / 0.74-0.80 |
| 17408 (general) | 1.36 / 0.91 | 0.67-0.69 / 0.70-0.76 | 1.77 / 1.19 | 0.67-0.68 / 0.74-0.80 | 1.62 / 1.09 | 0.61-0.67 / 0.77-0.81 |
| 8704 (general) | 1.35 / 0.90 | 0.66-0.67 / 0.73-0.76 | 1.74 / 1.15 | 0.65-0.67 / 0.75-0.80 | 1.53 / 1.03 | 0.63-0.67 / 0.75-0.80 |
| 4352 (general) | 1.35 / 0.79 | 0.58-0.67 / 0.74-0.77 | 1.77 / 1.08 | 0.61-0.66 / 0.75-0.79 | 1.50 / 1.03 | 0.63-0.69 / 0.75-0.82 |

## Warm-up (NA-WU)

Before CUDA graph capture, call `AddRMSNorm.warm_up(max_m, hidden=4096,
dtype=torch.bfloat16, device="cuda", zero_centered=False)` or
`SiluAndMul.warm_up(max_m, width=12288, dtype=torch.bfloat16, device="cuda")`.
The norm API also accepts `residual_dtype`, `weight_dtype`, `store_residual` and
`sum_out=True` for a contiguous residual-dtype sum buffer. Warm-up resolves every
contiguous, 16-byte-aligned M=1..max_m shape using the current overrides and table,
compiles each selected CuTe config once without launching, and launches
FlashInfer/Triton routes once per M on zeroed scratch to trigger their lazy
compilation and specialization. All scratch buffers are zero-filled, with
`eps=1e-6` for norm; norm warms the requested `sum_out`/`store_residual` variant,
and SiLU warms both `out=None` and supplied-output variants. There is no timing.
Warm-up returns maximal `{"m": (lo, hi), "provider": ..., "config": name_or_None}`
runs. All general CuTe activation routes are compile-only and report their
`cute_general_nN` descriptor.
Plans stay pinned independently of the 128-entry ordinary cache; the inference
module helpers `warm_up_add_rms_norm`, `warm_up_silu_and_mul`,
`warm_up_gelu_and_mul` and `warm_up_gelu_tanh_and_mul` pin the shared singletons.
Activation row-strided layouts reuse warmed vector lanes; other lane counts need
an eager call. During capture, an uncompiled
selected CuTe config raises with a warm-up instruction instead of silently
changing provider. Activation compilation failures raise; norm retains its
recorded eager fallback. Warm-up
itself must run outside capture. This applies to RTX 5090, RTX PRO 6000 and GB10.

## Findings that shaped this round

- The real SGLang market is FlashInfer 0.7.1's CuTe DSL `FusedAddRMSNormKernel`, called
  through sglang-kernel 0.4.9 (not `sgl_kernel`'s own CUDA). For H=4096 on sm_120/121
  its kernel source equals FlashInfer 0.7.0.post1's (the 0.7.1 changes are for H > 8192
  and sm_100/103/107), so with a 0.7.0.post1 environment our FlashInfer provider IS the
  SGLang kernel.
- vLLM 0.31.0 runs two different residual-norm kernels per decoder layer: site A stores
  only the normalized output, and site B rematerializes the residual and stores it. The
  vLLM column is that layer mean. It is taken from vLLM's own compile cache (Qwen3-8B,
  Qwen3.5-9B), and its launch geometry matches the nsys trace.
- Any `torch.profiler` use slows L2-hot CUDA-graph replays by 16-20 % for the rest of
  the process. A same-kernel cross-process control showed it: `xproc` was 0.80-0.84
  before the fix and is about 1.00 after. Kernel listings and geometry proofs now run in
  short-lived probe processes, and timing processes refuse profiler activation.
- An early PDL trigger (`griddepcontrol.launch_dependents` right after the wait and the
  loads, `_te` configs) won every decode/DFlash sweep row for both CuTe kernels. The
  dependent's wait still covers this grid's completion, so it is safe.
- Cold prefill rows sit at the DRAM floor. vLLM's two-site layer reads more and writes
  less than an op that stores the residual on every call. At large M its layer mean
  stays 1-3 % ahead of any per-op residual norm (ours, FlashInfer and SGLang alike).
  Beating it there needs layer-level fusion.

## Method

Each graph has L=32 calls, bounded by captured external timing events to exclude
Python enqueue gaps. Hot repeats one input/output. Cold rotates distinct
inputs whose combined size exceeds twice the device-reported L2; for small M,
several 32-call graphs cover the pool. Per-round time sums those graph times and
divides by all calls. Candidates have separate graphs, share the input pool, and
shuffle order (seeded) every round across 30 rounds. Each round's faster-market ratio is
`min(vLLM, SGLang) / ours`; the median and exact distribution-free 95% interval
use binomial order statistics. A separate identical ours graph measures A/A;
`epsilon = max(.005, abs(1-median_AA) + half_width_AA_CI)`. PASS requires the
faster-ratio interval lower endpoint to be at least `1-epsilon`. FlashInfer's
baseline is also recorded. A timing PASS does not qualify failed numerics.

**Numerical gate.** `--check` evaluates every distinct input against unrounded
FP64, reporting maximum output ULP, RMS, and RMS ULP. For **each metric**,
`ours <= min(comparator errors) * (1 + TAU[metric])`; comparators include actual
markets and the FP32 torch formulas. The best comparator may differ by metric.
Only an exactly zero best error gets an absolute floor of `1e-12`.
[`norm_act_tau.py`](../norm_act_tau.py) derives TAU from the largest relative
excess `max(a/b - 1, b/a - 1, 0)` between FP32 implementations that round once
to BF16/FP16 and differ only in evaluation/reduction order or rsqrt versus
reciprocal sqrt. The CPU run (`torch 2.13.0+cpu`, one thread, 208.1 seconds)
covered these BF16/FP16 shape sets:

- SiLU widths 8/128/12288 × M=1/3/4/16/17/64/256/1024.
- Standard and zero-centered add-RMSNorm hidden 8/136/4096 ×
  M=1/2/3/4/16/17/64/256/300/1024/2048, using native, pairwise, and
  chunk8/32/128 sums of squares with rsqrt and reciprocal sqrt. Zero padding
  handles non-power-of-two hidden sizes; division uses the original hidden size.
- Every finite gate encoding at width 12288, with up multipliers
  0/0.25/1/-3/128, plus shuffled gates (seed 20261016) paired with every finite
  up encoding, matching the shipped-config GPU test layouts.

Each random shape uses seeds 0–31, reduced to 0–7 above 262144 output elements,
plus the exact GPU test seed `20261015 + M` in every case (33 or 9 samples).
SiLU variants share the kernel's split-exponent negative-tail formula and avoid
premature multiplication overflow, so correctness defects do not inflate the
measured evaluation-order noise. The 180 random shape cases and 12 exhaustive
layouts measured:

| Metric | Maximum relative excess | TAU (rounded up to a power of two) | Maximum's shape set and seed |
| --- | ---: | ---: | --- |
| max_ulp | 0.0010644159089683303 | 2^-9 = 0.001953125 | Random zero-centered norm, FP16, M=4, hidden=136, seed 18 |
| rms | 3.2340738394420754e-5 | 2^-14 = 0.00006103515625 | Random zero-centered norm, FP16, M=2, hidden=136, seed 24 |
| rms_ulp | 9.633508800677149e-6 | 2^-16 = 0.0000152587890625 | Random SiLU, FP16, M=1, width=128, seed 21 |

Both norm maxima compare native/rsqrt against pairwise/rsqrt; the SiLU maximum
compares divide-then-multiply against multiply-then-divide.
Reproduce with `python -m benchmark.kernel_tuning.norm_act_tau`; `--device cuda`
uses the same CPU-generated FP32 draws rounded to each input dtype. Matching
seed numbers do not reproduce CUDA-native random draws. These are empirical
noise floors, not universal bounds on other distributions or device math
libraries. The CPU regression test measures a small subset including all
finite-encoding layouts and checks that the constants cover it; the reported
M=300/hidden=8 FP16 norm failure is also retained as a regression. The
counterexample that is worse than the best comparator still fails the gate.
The GPU suite additionally covers empty/odd M, other widths, both dtypes, output
aliasing, graph replay, and dynamic-M compile reuse; zero gates have a bitwise
check. Shipped-config exhaustive bitwise comparison against e8 applies only to
e8/e16: e16 chooses the negative-tail path per 8-element half, while e2/e4 may
legitimately route neighboring lanes differently. Every config retains the
numerical gate. Earlier timing results require requalification with this gate
before claiming a numerical PASS.

**N1/N2 FP32 activation calibration (measured on CUDA, see below).** FP32 SiLU,
GELU, tanh-GELU and relu2 support-range tests use the same per-metric best-comparator
rule, with a separate `TAU_FP32[kind][metric]`. The BF16/FP16 formulas and `TAU`
above are unchanged. FP32 comparators, shared between the tests and calibration,
are the torch FP32 eager expression of SGLang's math, native `F.silu` or
`F.gelu(approximate="none"/"tanh")` followed by the up multiply, and an
algebraically equivalent FP32 ordering. The latter multiplies gate by up before
the final activation factor; tanh-GELU also factors the cubic as
`gate * (1 + 0.044715 * gate.square())`. relu2 compares square, multiply and
power-of-two expressions of the comparison-masked positive input; these have
the same single-multiply rounding and measured zero spread. All four kinds
use the unrounded FP64 oracle and the same `no_worse` gate; no CuTe result is
used to choose a tolerance.

Run on **each target GPU** (GB10, RTX 5090 and RTX PRO 6000):

```bash
python -m benchmark.kernel_tuning.norm_act_tau --fp32-activation
```

This mode defaults to CUDA and covers the full grids, including exhaustive-only
and empty cases: M=0/1/3/64/257; SiLU/GELU/tanh-GELU widths
4/8/12/13/6144/12288; relu2 widths 1/13/6144/12288 (110 kind/shape cases).
Each shape uses device-native standard-normal FP32 draws with seeds 0–31,
or 0–7 above 262144 output elements, plus its GPU test seed
`20261016 + M + width` (33 or 9 samples). `--seeds N` requires N >= 8;
partial shape overrides are rejected. Unlike the older 16-bit calibration,
CUDA mode reproduces the tests' CUDA RNG draws. CPU draws with the same seed
are different.

For every identical sample and each of max ULP, RMS and RMS ULP, measure
`max(a/b - 1, b/a - 1, 0)` over every unordered comparator pair, using `1e-12`
only for an exactly zero denominator. Maximize over seeds and shapes **within
each kind**, then round each metric UP to a power of two. The command prints
per-shape maxima, each final maximum's shape/seed/comparator-pair witness, and
a copyable `TAU_FP32` dictionary. For a shared three-GPU gate, use the maximum
derived value across those GPUs independently for each kind and metric, and
record their measured maxima and witnesses here. A nearly exact comparator
on a tiny shape can make the relative spread large; this is not the same as
a large absolute output error.

Measured on CUDA: `python -m benchmark.kernel_tuning.norm_act_tau --fp32-activation`
on the RTX 5090, RTX PRO 6000 and GB10 (torch 2.13.0+cu130, full default sweep, about
3 s per device). The three devices gave identical maxima. Rows of fewer than 64 outputs
(M=1 with width 4-13) set a far larger relative spread than every other shape (a
4-element RMS is dominated by one value), so the gate uses two size bands instead of
loosening the large one with the tiny rows' floor:

| Kind | >= 64 outputs: maxima (max ULP / RMS / RMS ULP) | TAU_FP32 | < 64 outputs: maxima | TAU_FP32_SMALL |
| --- | --- | --- | --- | --- |
| SiLU | 0.588 / 0.880 / 0.196 | 1 / 1 / 0.25 | 1.518 / 11.58 / 2.292 | 2 / 16 / 4 |
| GELU | 0.044 / 0.882 / 0.042 | 0.0625 / 1 / 0.0625 | 3.007 / 37.50 / 2.832 | 4 / 64 / 4 |
| tanh-GELU | 0.410 / 0.971 / 0.099 | 0.5 / 1 / 0.125 | 3.569 / 36.70 / 2.176 | 4 / 64 / 4 |
| relu2 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Every small-band witness is an M=1 row of width 4 or 12, sglang_eager against the
reordered comparator. With these values `tests/gpu/test_activation.py` passes on all
three devices (FlashInfer 0.7.0.post1). Empty forced-CuTe SiLU calls now assert provider `"empty"`
(nothing launched); the dispatch test exercises real FP32 CuTe output, while
the three inadmissible BF16 layout cases retain their FlashInfer routing checks.

In the GPU suite, `test_shipped_norm_configurations` applies this gate to every
shipped norm dispatch-table entry (CuTe, FlashInfer and Triton alike) at its band's
lowest M, against FlashInfer, vLLM's Inductor kernel and the FP32 torch formulas.

The GPU suites re-ran this gate for every shipped configuration on RTX 5090,
RTX PRO 6000 and GB10 under both FlashInfer 0.7.0.post1 and 0.6.11.post3:
1480 tests per run at `0e14b13` (earlier 1276 at `9b98736`).

## Market columns

Market source is supplied at runtime via `--market-root`; no vendor source is
bundled here. The default `--vllm-mode shipped` loads vLLM 0.31.0's own Inductor
cache modules by file path, substitutes current device properties in host
metadata, and launches the JIT bodies in the supplied layer wrapper's argument
order. Bodies are unchanged except for the explicit D2 aux layout specialization
described below. RTX 5090 runs pin the shipped `.best_config`; other GPUs autotune
and record their selected configurations. Every site is profiled for kernel name,
grid and block. RTX 5090 geometry must match the nsys trace: norm A/B grid M and
512 threads; SiLU XBLOCK=1024, four warps (grids 192/768 at M=16/64).

`--vllm-mode compiled` imports the shipped vLLM IR using a minimal registration
stub and compiles dynamic-M fragments with vLLM's recorded Inductor options.
Receipts compare normalized JIT definitions, preserving arithmetic, casts,
loads/stores and tiling while ignoring names and host metadata. A mismatch is
MARKET_ERROR. `auto` retains the failed proof and falls back to shipped modules;
`--allow-vllm-mismatch` produces diagnostic results only. The norm `vllm` timing
alternates site A (normalized output only) and site B (rematerialized previous
residual plus normalized and residual outputs), with equal site counts. Separate
`vllm_a`/`vllm_b` columns are diagnostic; the market verdict uses the layer mean.
Site B also reads h1 (the o_proj output from before the MLP); it is hot in the hot
regime and rotates with the residual (L2-cold) in the engine and cold regimes, restored
outside timing. The shipped B
keeps the rematerialized intermediate in FP32 while the public IR rounds it to
BF16. Receipts report errors against both references; arithmetic is not changed
to conceal that difference.

D2 (`add_norm_sum` / `gemma_add_norm_sum`) uses the DFlash capturing-layer
subgraph: even layers run A, odd layers run B-aux. Qwen3 selects the
`triton_red_fused_add_fused_add_rms_norm_2.ciqqjvb7...` module, and Qwen3.5
selects `triton_red_fused__to_copy_add_fused_add_rms_norm_2.cxy7p5q7...`.
The capturing wrapper calls
`triton_red_fused_add_fused_add_rms_norm_2.run(buf5, buf0, arg4_1, arg7_1, buf6, buf8, buf15, s72, 4096)`:
inputs are h2, h1, old residual, weight; outputs are new residual, normalized,
and aux (h2 + s1). The compiled equality attempt returns
`(normalized, new_residual, h2 + s1)` from the supplied IR. There is no zero-h1
adaptation and no requirement for a standalone two-input D2 snapshot.

**D2 layout qualification:** both supplied 3-store modules hardcode contiguous
aux addressing (`out_ptr3 + r0_1 + 4096*x0`). Passing a column slice directly would
corrupt adjacent columns at M > 1. The shipped adapter therefore specializes
only that aux store's row stride from H to 3*H in memory, preserving all
arithmetic, other addresses and the three stores. No third-party source file is
written. Receipts explicitly mark this timed JIT body as different from the
original, record the adapted source hash, and profile its geometry and chosen
configuration separately. The original kernel's source-equality receipt covers
the contiguous body only. This is a layout adaptation of the capturing dispatch,
not a claim that the unmodified dump supports strided aux output.

Compiled mode still attempts equality against the original 3-store kernel using
the IR graph returning all three outputs. If that proof succeeds, packing its
contiguous aux into the requested slice adds a timed copy; the receipt identifies
that overhead. `auto` falls back to the shipped stride specialization after a
failed compiled proof. Shipped mode needs no mismatch override.

Our D2 candidate alternates plain `add_rms_norm_` and the same call with
`sum_out=aux_slice`. The `unfused` control alternates plain norm and a separate
`torch.add` plus plain norm; the add observes the original operands before
in-place normalization. All layer means require even L. The default rows are
M = 1, 4, 16, 64, 256, 1024, regime `engine`, H=4096, BF16, L=36, 30 rounds.
The faster-market ratio uses the minimum of the vLLM and SGLang layer means
within each paired round; `unfused_ratio` remains a separate paired comparison.
Aux checks use a middle column slice of `[M, 3*H]` with guard columns and test
both layer parities. vLLM B-aux gets separate normalized/residual/aux metrics.

The norm `sglang` column loads sglang-kernel 0.4.9's supplied `utils.py` and
`elementwise.py` without importing its extension. This is SGLang 72730d6's
RMSNorm/GemmaRMSNorm residual dispatch to FlashInfer 0.7.1 with `enable_pdl=None`.
A private worker asserts that exact version, the wrapper's FlashInfer branch,
and usable CuTe DSL >=4.7; receipts include source hashes, dispatch citations,
actual Torch/CuTe versions and profiled kernels. The worker maps the parent's
pools through CUDA IPC, so both processes time the same physical buffers. Synchronous JSON-line requests
place `sglang` and `fi071_direct` in the same seeded shuffled rounds as local
candidates, with CUDA synchronization on both sides of every process handoff.
Both restore outside timed graph events. `fi071_direct` controls wrapper overhead;
`flashinfer` remains the environment provider. `--fi07` separately qualifies our
provider on 0.7.0.post1 and never stands in for SGLang.

For D2, the worker alternates plain norm and norm followed by
`aux_slice.copy_(residual)`, including the copy in the odd-layer timed graph.
This follows SGLang 72730d6's `layer_boundary/residual/access.py:final_norm_pair`
and `layer_boundary/stage.py`, which call `capture(residual)` after fused norm.
`layers/aux_hidden_states.py:AuxHiddenStateList.capture` clones that updated
residual; the preallocated packer copies it into `[tokens, K*hidden]`.
The direct-FI control uses the same alternation. Receipts profile both parities,
and the worker checks aux values and guard columns before returning metrics.

Runner requirements: the external market root supplies Python IR, Triton
modules, `.best_config` siblings, layer wrappers and `sgl_kernel` Python wrappers.
The shipped specializations require BF16 and H=4096 for norms (12288 for SiLU).
The worker accepts `--fi071 /path/to/overlay`; otherwise it installs a private
FI 0.7.1 overlay with `pip --no-deps --target` in a temporary directory. The parent keeps
its own FI provider. The supplied runner uses Torch 2.13.0+cu130, Triton 3.7.1
and CuTe DSL 4.8.0; upstream SGLang pins Torch 2.14.1, and actual versions are
recorded. The worker omits unused h1 storage, shares the parent's seeded
x/residual/weight values, and exits cooperatively without process signals.

Example D2 command (repeat with `gemma_add_norm_sum` for Qwen3.5):

```sh
PYTHONPATH=python:. python -m benchmark.kernel_tuning.norm_act_paired \
  --op add_norm_sum --market-root ../market --vllm-mode shipped \
  --check --profile
```

For C4, use `--op add_norm --rows 16 --regimes engine` and optionally
`--candidates ours,static,cute,flashinfer,fi_default,triton`. Unsupported
standard-weight Triton is recorded as skipped. `--sweep --static-weight 0`
includes both static-weight settings across the CuTe configurations.

SiLU keeps SGLang 72730d6's own `act_and_mul` JIT kernel through the tvm-ffi loader
(`--sglang-root`), with its CUDA flags and PDL defaults; its vLLM column is the shipped,
source- and geometry-proven Inductor kernel above.

## Receipts (2026-10-15/16, 30 rounds)

The default path (`static_weight=False`), H=4096 (norm) and 12288 (SiLU), BF16.
Each row is paired in one call: L=36 norm layers (32 SiLU) per graph. The ratio is
min(vLLM, SGLang) / ours, with the distribution-free 95 % CI low end and
eps = max(0.005, A/A noise). `xproc` is the same-kernel cross-process control
(remote/local; 1.000 means the two-process pairing has no bias).

C4 and D2 receipts are from the tables at 9556504 (code 9556504..48fdd88), except the
zero-centered D2 M=1024 rows of the RTX 5090 and PRO 6000, re-measured with the new C5
bands. C3 and C5 were re-measured on 2026-10-16 with the 0.1.12 baseline, FlashInfer
0.7.0.post1, in the timing process (`HEAD` tables). On 0.7.0.post1 our FlashInfer provider runs the
same norm kernel as SGLang's FlashInfer 0.7.1 (the `flashinfer` and `sglang` columns
agree within noise), and each C5 band takes our fastest provider from a paired
10-round sweep of every CuTe config, FlashInfer and Triton (`_PREFILL_ZERO` in
`norm/_dispatch.py`).

Totals: C3 37/38 and C5 42/54 rows pass (re-measured); C4 54/54 and D2 36/36 (D2
zero-centered M=1024 re-measured on the 5090 and PRO 6000).
The remaining misses:
- C5 prefill rows at 0.971-0.995 (DRAM-floor or identical-kernel ties). Cold/engine
  rows at large M lose to vLLM's site A/site B layer pair: site A writes no residual
  and site B re-adds it, a read-heavier DRAM mix than any per-op residual norm, so
  FlashInfer (= SGLang), Triton and CuTe lose by the same margin; beating it needs
  layer-level fusion. The others (5090 M=1024 cold and M=2048 engine, PRO M=256 hot,
  M=1024 engine/cold and M=8192 engine) trail SGLang's kernel by 0.5-3 %; our FlashInfer
  provider runs that same kernel at ratio ~1.00 there, within the sweep's run-to-run
  spread.
  The 5090 M=1024 hot row that read 0.921 on FlashInfer 0.6.11 now routes to CuTe
  (t1024, 2 rows/CTA) and passes at 1.028.
- C3 GB10 M=256, multi-process rule: >= 60 interleaved
  pairs over >= 3 fresh X925-pinned processes; PASS iff the pooled per-pair median
  ratio (faster market of the same round / candidate) is >= 1 and the pooled
  bootstrap 95% CI_low >= 1 - tol. tol is the row's A/A noise:
  max(0.005, |1 - median(aa/ours)| + CI95 half-width of that median), measured in
  the same processes (0.005 here). Driver: `benchmark/kernel_tuning/norm_act_multiproc.py`
  (FlashInfer 0.7.0.post1). Pooled over 12 processes / 240 pairs on GB10:

  | candidate | hot median [CI lo] | cold median [CI lo] |
  |---|---|---|
  | t64/e8 (shipped) | 0.998 [0.981] | 1.000 [0.994] |
  | t256/e2 `st_ef` (previous) | 0.893 [0.856] | 1.052 [1.032] |
  | t256/e2 | 0.808 [0.796] | 1.035 [1.013] |
  | FlashInfer | 0.998 [0.953] | 0.959 [0.947] |

  No provider passes both regimes; t64/e8 has the best worst-regime median and ships
  (fix-or-replace), disclosed as FAIL by a median 0.2 % below 1. Engine-regime A/B
  (activation then the 12288x4096 down-proj GEMM in one graph, 3 processes, 60 pairs):
  total time vs `st_ef` is 0.975 (t256/e2) and 0.978 (t64/e8) hot, 0.976/0.978 cold
  at M=256, and 0.989/0.996 hot at M=384, so evict-first output stores are slower
  with the consumer and no band ships them. M=384-511 uses t256/e2 (no cache hint):
  4-process receipt hot 1.022 [1.019], cold 1.004 [1.002] PASS.

### C3 SiLU x mul (width 12288)

| device | M | regime | ours us | vLLM us (A/B) | SGLang us | ratio vs faster [CI95 lo] | eps | provider | verdict | xproc |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 | 1 | hot | 0.649 | 0.828 | 1.011 | 1.279 [1.259] | 0.008 | cute | PASS |  |
| GB10 | 1 | cold | 1.244 | 1.763 | 1.895 | 1.417 [1.414] | 0.005 | cute | PASS |  |
| GB10 | 4 | hot | 0.878 | 1.024 | 1.098 | 1.174 [1.153] | 0.009 | cute | PASS |  |
| GB10 | 4 | cold | 2.149 | 2.683 | 2.613 | 1.215 [1.212] | 0.005 | cute | PASS |  |
| GB10 | 16 | hot | 1.163 | 1.675 | 1.233 | 1.073 [1.059] | 0.019 | cute | PASS |  |
| GB10 | 16 | cold | 5.947 | 6.406 | 6.037 | 1.015 [1.013] | 0.005 | cute | PASS |  |
| GB10 | 64 | hot | 2.751 | 8.118 | 2.817 | 1.024 [1.013] | 0.008 | cute | PASS |  |
| GB10 | 64 | cold | 21.357 | 21.819 | 21.570 | 1.011 [1.008] | 0.005 | cute | PASS |  |
| GB10 | 256 | hot | 17.718 | 56.434 | 16.752 | 0.998 [0.959] | 0.005 | cute | FAIL (multi-process, disclosed) |  |
| GB10 | 256 | cold | 83.477 | 83.590 | 84.594 | 1.000 [0.968] | 0.005 | cute | FAIL (multi-process, disclosed) |  |
| GB10 | 384 | hot | 97.810 | 118.361 | 98.955 | 1.022 [1.019] | 0.005 | cute | PASS |  |
| GB10 | 384 | cold | 118.261 | 118.749 | 126.017 | 1.004 [1.002] | 0.005 | cute | PASS |  |
| GB10 | 1024 | hot | 322.833 | 322.007 | 334.973 | 0.997 [0.997] | 0.005 | cute | PASS |  |
| GB10 | 1024 | cold | 309.464 | 308.835 | 327.762 | 0.998 [0.997] | 0.005 | cute | PASS |  |
| PRO 6000 | 1 | hot | 0.715 | 1.031 | 1.138 | 1.435 [1.414] | 0.021 | cute | PASS |  |
| PRO 6000 | 1 | cold | 1.114 | 1.409 | 1.547 | 1.265 [1.264] | 0.005 | cute | PASS |  |
| PRO 6000 | 4 | hot | 0.790 | 0.935 | 1.164 | 1.161 [1.153] | 0.008 | cute | PASS |  |
| PRO 6000 | 4 | cold | 1.261 | 1.516 | 1.586 | 1.204 [1.201] | 0.005 | cute | PASS |  |
| PRO 6000 | 16 | hot | 1.031 | 1.078 | 1.158 | 1.050 [1.039] | 0.016 | cute | PASS |  |
| PRO 6000 | 16 | cold | 1.830 | 1.961 | 1.927 | 1.051 [1.047] | 0.005 | cute | PASS |  |
| PRO 6000 | 64 | hot | 1.417 | 1.609 | 1.602 | 1.130 [1.126] | 0.007 | cute | PASS |  |
| PRO 6000 | 64 | cold | 4.243 | 4.366 | 4.248 | 1.002 [1.000] | 0.005 | cute | PASS |  |
| PRO 6000 | 256 | hot | 3.200 | 4.062 | 3.319 | 1.042 [1.032] | 0.005 | cute | PASS |  |
| PRO 6000 | 256 | cold | 13.700 | 13.879 | 13.729 | 1.002 [0.999] | 0.005 | cute | PASS |  |
| PRO 6000 | 1024 | hot | 9.421 | 31.447 | 10.115 | 1.075 [1.068] | 0.013 | cute | PASS |  |
| PRO 6000 | 1024 | cold | 50.742 | 51.386 | 51.080 | 1.006 [1.005] | 0.005 | cute | PASS |  |
| RTX 5090 | 1 | hot | 0.612 | 0.843 | 0.907 | 1.361 [1.320] | 0.028 | cute | PASS |  |
| RTX 5090 | 1 | cold | 0.954 | 1.174 | 1.288 | 1.232 [1.227] | 0.005 | cute | PASS |  |
| RTX 5090 | 4 | hot | 0.642 | 0.849 | 0.952 | 1.331 [1.318] | 0.032 | cute | PASS |  |
| RTX 5090 | 4 | cold | 1.075 | 1.290 | 1.330 | 1.200 [1.197] | 0.005 | cute | PASS |  |
| RTX 5090 | 16 | hot | 0.887 | 1.058 | 0.958 | 1.081 [1.069] | 0.038 | cute | PASS |  |
| RTX 5090 | 16 | cold | 1.529 | 1.719 | 1.608 | 1.052 [1.050] | 0.006 | cute | PASS |  |
| RTX 5090 | 64 | hot | 1.225 | 1.408 | 1.459 | 1.153 [1.142] | 0.009 | cute | PASS |  |
| RTX 5090 | 64 | cold | 3.380 | 3.521 | 3.401 | 1.008 [1.005] | 0.005 | cute | PASS |  |
| RTX 5090 | 256 | hot | 3.263 | 3.606 | 3.523 | 1.081 [1.076] | 0.010 | cute | PASS |  |
| RTX 5090 | 256 | cold | 11.572 | 11.580 | 11.732 | 1.000 [0.993] | 0.010 | cute | PASS |  |
| RTX 5090 | 1024 | hot | 10.294 | 45.316 | 11.190 | 1.087 [1.071] | 0.010 | cute | PASS |  |
| RTX 5090 | 1024 | cold | 47.673 | 47.728 | 47.874 | 1.001 [0.998] | 0.005 | cute | PASS |  |

### C4 residual add + RMSNorm, standard weight (H 4096)

| device | M | regime | ours us | vLLM us (A/B) | SGLang us | ratio vs faster [CI95 lo] | eps | provider | verdict | xproc |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 | 1 | hot | 1.277 | 1.795 (1.914/1.667) | 1.479 | 1.173 [1.155] | 0.019 | cute | PASS | 0.994 |
| GB10 | 1 | engine | 1.483 | 2.132 (2.047/2.196) | 1.701 | 1.148 [1.145] | 0.010 | cute | PASS | 1.005 |
| GB10 | 1 | cold | 1.560 | 2.210 (2.129/2.302) | 1.779 | 1.144 [1.140] | 0.008 | cute | PASS | 1.007 |
| GB10 | 2 | hot | 1.372 | 1.720 (1.651/1.776) | 1.821 | 1.250 [1.239] | 0.008 | cute | PASS | 0.996 |
| GB10 | 2 | engine | 1.569 | 2.764 (3.268/2.261) | 2.181 | 1.391 [1.387] | 0.007 | cute | PASS | 1.006 |
| GB10 | 2 | cold | 1.678 | 2.306 (2.210/2.367) | 2.377 | 1.375 [1.373] | 0.015 | cute | PASS | 1.007 |
| GB10 | 3 | hot | 1.391 | 1.733 (1.650/1.784) | 1.820 | 1.239 [1.225] | 0.030 | cute | PASS | 0.994 |
| GB10 | 3 | engine | 1.584 | 2.198 (2.136/2.286) | 2.081 | 1.315 [1.311] | 0.011 | cute | PASS | 1.009 |
| GB10 | 3 | cold | 1.763 | 2.875 (3.327/2.417) | 2.392 | 1.360 [1.355] | 0.032 | cute | PASS | 1.030 |
| GB10 | 4 | hot | 1.394 | 1.737 (1.653/1.780) | 2.057 | 1.245 [1.229] | 0.017 | cute | PASS | 0.984 |
| GB10 | 4 | engine | 1.624 | 2.803 (3.251/2.339) | 2.304 | 1.426 [1.417] | 0.007 | cute | PASS | 1.001 |
| GB10 | 4 | cold | 1.892 | 2.415 (2.292/2.540) | 2.558 | 1.281 [1.275] | 0.040 | cute | PASS | 1.019 |
| GB10 | 16 | hot | 1.428 | 1.971 (2.024/1.869) | 2.104 | 1.360 [1.346] | 0.009 | cute | PASS | 0.989 |
| GB10 | 16 | engine | 2.485 | 3.719 (3.706/3.740) | 2.913 | 1.199 [1.172] | 0.040 | cute | PASS | 1.022 |
| GB10 | 16 | cold | 3.553 | 3.878 (3.411/4.394) | 4.020 | 1.103 [1.092] | 0.061 | cute | PASS | 1.037 |
| GB10 | 64 | hot | 2.468 | 2.788 (2.450/3.085) | 2.788 | 1.117 [1.102] | 0.005 | cute | PASS | 0.990 |
| GB10 | 64 | engine | 5.528 | 8.210 (6.260/10.182) | 5.765 | 1.056 [1.039] | 0.028 | cute | PASS | 1.032 |
| GB10 | 64 | cold | 10.009 | 10.313 (8.076/12.542) | 10.212 | 1.023 [1.017] | 0.027 | cute | PASS | 1.005 |
| PRO 6000 | 1 | hot | 1.365 | 1.764 (1.663/1.860) | 1.536 | 1.125 [1.109] | 0.011 | cute | PASS | 1.000 |
| PRO 6000 | 1 | engine | 1.524 | 2.239 (2.167/2.323) | 1.752 | 1.152 [1.149] | 0.005 | cute | PASS | 1.004 |
| PRO 6000 | 1 | cold | 1.616 | 2.309 (2.221/2.405) | 1.833 | 1.135 [1.131] | 0.012 | cute | PASS | 1.009 |
| PRO 6000 | 2 | hot | 1.479 | 2.029 (2.080/1.942) | 1.934 | 1.308 [1.305] | 0.009 | cute | PASS | 1.000 |
| PRO 6000 | 2 | engine | 1.637 | 2.288 (2.213/2.349) | 2.175 | 1.332 [1.326] | 0.007 | cute | PASS | 1.005 |
| PRO 6000 | 2 | cold | 1.713 | 2.374 (2.288/2.443) | 2.465 | 1.387 [1.383] | 0.007 | cute | PASS | 1.005 |
| PRO 6000 | 3 | hot | 1.479 | 1.870 (1.764/1.939) | 1.934 | 1.256 [1.247] | 0.008 | cute | PASS | 1.000 |
| PRO 6000 | 3 | engine | 1.652 | 2.271 (2.221/2.351) | 2.155 | 1.302 [1.290] | 0.006 | cute | PASS | 1.004 |
| PRO 6000 | 3 | cold | 1.738 | 2.359 (2.278/2.446) | 2.381 | 1.361 [1.356] | 0.008 | cute | PASS | 1.007 |
| PRO 6000 | 4 | hot | 1.533 | 1.912 (1.833/2.002) | 2.147 | 1.260 [1.244] | 0.020 | cute | PASS | 0.959 |
| PRO 6000 | 4 | engine | 1.678 | 2.304 (2.250/2.357) | 2.284 | 1.361 [1.359] | 0.005 | cute | PASS | 1.002 |
| PRO 6000 | 4 | cold | 1.758 | 2.388 (2.300/2.462) | 2.531 | 1.360 [1.358] | 0.008 | cute | PASS | 1.007 |
| PRO 6000 | 16 | hot | 1.532 | 1.975 (1.877/2.048) | 2.219 | 1.289 [1.274] | 0.007 | cute | PASS | 1.000 |
| PRO 6000 | 16 | engine | 1.782 | 2.383 (2.309/2.456) | 2.432 | 1.337 [1.335] | 0.007 | cute | PASS | 1.007 |
| PRO 6000 | 16 | cold | 1.933 | 2.464 (2.392/2.574) | 2.694 | 1.280 [1.273] | 0.008 | cute | PASS | 1.005 |
| PRO 6000 | 64 | hot | 1.637 | 2.088 (2.002/2.173) | 2.275 | 1.277 [1.267] | 0.011 | cute | PASS | 1.008 |
| PRO 6000 | 64 | engine | 2.465 | 2.954 (2.685/3.235) | 2.776 | 1.127 [1.121] | 0.014 | cute | PASS | 1.012 |
| PRO 6000 | 64 | cold | 3.107 | 3.303 (2.976/3.616) | 3.431 | 1.071 [1.060] | 0.027 | cute | PASS | 1.022 |
| RTX 5090 | 1 | hot | 1.110 | 1.446 (1.358/1.509) | 1.252 | 1.130 [1.125] | 0.013 | cute | PASS | 0.978 |
| RTX 5090 | 1 | engine | 1.269 | 1.872 (1.808/1.929) | 1.469 | 1.158 [1.154] | 0.005 | cute | PASS | 1.002 |
| RTX 5090 | 1 | cold | 1.346 | 1.934 (1.861/1.991) | 1.527 | 1.135 [1.133] | 0.011 | cute | PASS | 1.010 |
| RTX 5090 | 2 | hot | 1.195 | 1.536 (1.476/1.593) | 1.536 | 1.285 [1.278] | 0.007 | cute | PASS | 1.005 |
| RTX 5090 | 2 | engine | 1.358 | 1.904 (1.843/1.961) | 1.857 | 1.368 [1.364] | 0.009 | cute | PASS | 1.008 |
| RTX 5090 | 2 | cold | 1.413 | 1.966 (1.905/2.038) | 2.076 | 1.393 [1.391] | 0.011 | cute | PASS | 1.009 |
| RTX 5090 | 3 | hot | 1.192 | 1.520 (1.454/1.587) | 1.536 | 1.284 [1.276] | 0.006 | cute | PASS | 1.011 |
| RTX 5090 | 3 | engine | 1.365 | 1.903 (1.853/1.960) | 1.784 | 1.309 [1.306] | 0.006 | cute | PASS | 1.005 |
| RTX 5090 | 3 | cold | 1.416 | 1.966 (1.915/2.026) | 2.018 | 1.390 [1.388] | 0.008 | cute | PASS | 1.006 |
| RTX 5090 | 4 | hot | 1.204 | 1.536 (1.469/1.593) | 1.736 | 1.271 [1.258] | 0.008 | cute | PASS | 0.985 |
| RTX 5090 | 4 | engine | 1.386 | 1.914 (1.867/1.965) | 1.917 | 1.382 [1.380] | 0.005 | cute | PASS | 1.002 |
| RTX 5090 | 4 | cold | 1.445 | 1.977 (1.925/2.036) | 2.124 | 1.366 [1.361] | 0.009 | cute | PASS | 1.007 |
| RTX 5090 | 16 | hot | 1.247 | 1.586 (1.517/1.643) | 1.764 | 1.273 [1.263] | 0.008 | cute | PASS | 1.004 |
| RTX 5090 | 16 | engine | 1.496 | 1.984 (1.928/2.078) | 2.003 | 1.325 [1.321] | 0.008 | cute | PASS | 1.005 |
| RTX 5090 | 16 | cold | 1.633 | 2.060 (1.987/2.138) | 2.263 | 1.263 [1.258] | 0.014 | cute | PASS | 1.012 |
| RTX 5090 | 64 | hot | 1.362 | 1.707 (1.650/1.767) | 1.820 | 1.271 [1.251] | 0.009 | cute | PASS | 1.006 |
| RTX 5090 | 64 | engine | 2.002 | 2.429 (2.204/2.706) | 2.298 | 1.154 [1.144] | 0.013 | cute | PASS | 1.013 |
| RTX 5090 | 64 | cold | 2.546 | 2.723 (2.493/3.016) | 2.845 | 1.070 [1.066] | 0.026 | cute | PASS | 1.023 |

### C5 residual add + RMSNorm, zero-centered weight (H 4096)

| device | M | regime | ours us | vLLM us (A/B) | SGLang us | ratio vs faster [CI95 lo] | eps | provider | verdict | xproc |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 | 256 | hot | 5.856 | 6.140 (5.286/7.164) | 6.714 | 1.049 [1.046] | 0.009 | cute | PASS | 1.005 |
| GB10 | 256 | engine | 19.812 | 30.223 (19.740/40.704) | 19.401 | 0.981 [0.977] | 0.031 | cute | PASS | 1.008 |
| GB10 | 256 | cold | 38.428 | 39.003 (29.074/48.683) | 38.084 | 0.995 [0.990] | 0.013 | cute | PASS | 1.016 |
| GB10 | 512 | hot | 16.441 | 20.591 (9.557/29.739) | 16.143 | 1.129 [0.984] | 0.188 | cute | PASS | 1.154 |
| GB10 | 512 | engine | 44.349 | 65.266 (41.492/89.303) | 44.322 | 1.000 [0.998] | 0.044 | cute | PASS | 1.049 |
| GB10 | 512 | cold | 75.378 | 77.497 (58.311/96.668) | 76.075 | 1.009 [1.006] | 0.007 | cute | PASS | 1.003 |
| GB10 | 1024 | hot | 90.980 | 141.412 (57.701/188.231) | 88.339 | 0.979 [0.970] | 0.173 | triton | PASS | 1.121 |
| GB10 | 1024 | engine | 109.951 | 147.418 (106.413/185.292) | 109.552 | 0.998 [0.995] | 0.037 | triton | PASS | 1.006 |
| GB10 | 1024 | cold | 144.270 | 147.115 (110.460/184.068) | 149.404 | 1.020 [1.017] | 0.005 | triton | PASS | 1.000 |
| GB10 | 2048 | hot | 272.350 | 298.360 (223.412/373.762) | 280.661 | 1.046 [1.030] | 0.026 | triton | PASS | 1.038 |
| GB10 | 2048 | engine | 284.492 | 292.869 (220.020/366.806) | 291.640 | 1.026 [1.023] | 0.008 | triton | PASS | 1.005 |
| GB10 | 2048 | cold | 285.558 | 289.524 (217.178/362.310) | 297.134 | 1.013 [1.012] | 0.005 | triton | PASS | 1.000 |
| GB10 | 4096 | hot | 578.640 | 598.555 (448.208/749.083) | 602.623 | 1.036 [1.034] | 0.036 | triton | PASS | 1.035 |
| GB10 | 4096 | engine | 575.407 | 580.338 (434.193/725.817) | 598.499 | 1.008 [1.007] | 0.005 | triton | PASS | 1.001 |
| GB10 | 4096 | cold | 564.252 | 571.904 (425.471/719.208) | 588.680 | 1.012 [1.011] | 0.005 | triton | PASS | 1.001 |
| GB10 | 8192 | hot | 1158.360 | 1193.047 (894.421/1490.845) | 1209.982 | 1.031 [1.028] | 0.017 | triton | PASS | 1.010 |
| GB10 | 8192 | engine | 1162.455 | 1158.747 (860.960/1461.826) | 1202.189 | 0.997 [0.996] | 0.005 | triton | PASS | 1.000 |
| GB10 | 8192 | cold | 1139.300 | 1132.822 (842.211/1423.498) | 1171.110 | 0.994 [0.994] | 0.005 | triton | FAIL | 1.000 |
| PRO 6000 | 256 | hot | 2.476 | 2.944 (2.762/3.073) | 2.445 | 0.992 [0.987] | 0.005 | flashinfer | FAIL | 0.992 |
| PRO 6000 | 256 | engine | 4.784 | 6.107 (4.891/7.430) | 4.805 | 1.006 [1.001] | 0.017 | flashinfer | PASS | 1.011 |
| PRO 6000 | 256 | cold | 7.674 | 7.645 (6.280/8.926) | 7.673 | 0.996 [0.989] | 0.021 | flashinfer | PASS | 1.017 |
| PRO 6000 | 512 | hot | 3.868 | 4.153 (3.812/4.437) | 4.096 | 1.059 [1.053] | 0.005 | cute | PASS | 1.000 |
| PRO 6000 | 512 | engine | 7.190 | 10.476 (7.702/13.149) | 7.361 | 1.024 [1.016] | 0.022 | cute | PASS | 1.016 |
| PRO 6000 | 512 | cold | 12.486 | 13.482 (10.695/15.928) | 12.685 | 1.018 [1.009] | 0.028 | cute | PASS | 1.019 |
| PRO 6000 | 1024 | hot | 6.599 | 6.955 (5.760/8.123) | 6.827 | 1.034 [1.034] | 0.005 | cute | PASS | 0.995 |
| PRO 6000 | 1024 | engine | 12.458 | 18.772 (13.994/23.485) | 12.060 | 0.973 [0.968] | 0.018 | cute | FAIL | 1.014 |
| PRO 6000 | 1024 | cold | 22.516 | 24.877 (19.797/29.668) | 22.356 | 0.995 [0.990] | 0.010 | cute | FAIL | 1.008 |
| PRO 6000 | 2048 | hot | 12.459 | 12.876 (11.150/14.336) | 13.028 | 1.035 [1.032] | 0.005 | cute | PASS | 1.004 |
| PRO 6000 | 2048 | engine | 23.068 | 34.112 (23.383/45.581) | 23.766 | 1.033 [1.030] | 0.010 | cute | PASS | 1.011 |
| PRO 6000 | 2048 | cold | 45.283 | 46.852 (36.750/56.945) | 45.046 | 0.996 [0.993] | 0.010 | cute | PASS | 1.008 |
| PRO 6000 | 4096 | hot | 24.803 | 50.916 (16.032/94.450) | 24.916 | 1.007 [1.005] | 0.026 | flashinfer | PASS | 1.021 |
| PRO 6000 | 4096 | engine | 46.383 | 86.745 (51.257/111.672) | 46.393 | 1.001 [1.000] | 0.013 | flashinfer | PASS | 1.014 |
| PRO 6000 | 4096 | cold | 90.851 | 90.897 (69.717/113.282) | 90.937 | 1.001 [0.998] | 0.005 | flashinfer | PASS | 1.002 |
| PRO 6000 | 8192 | hot | 90.216 | 172.032 (133.517/227.000) | 90.283 | 1.001 [0.998] | 0.101 | cute | PASS | 0.997 |
| PRO 6000 | 8192 | engine | 149.760 | 180.281 (135.925/227.556) | 148.081 | 0.988 [0.933] | 0.032 | cute | FAIL | 0.996 |
| PRO 6000 | 8192 | cold | 183.397 | 181.446 (135.822/227.896) | 183.381 | 0.990 [0.989] | 0.005 | cute | FAIL | 1.000 |
| RTX 5090 | 256 | hot | 2.413 | 2.503 (2.492/2.550) | 2.617 | 1.046 [1.038] | 0.008 | cute | PASS | 0.990 |
| RTX 5090 | 256 | engine | 3.483 | 4.905 (4.639/5.227) | 3.672 | 1.061 [1.049] | 0.011 | cute | PASS | 1.006 |
| RTX 5090 | 256 | cold | 5.219 | 6.222 (5.826/6.566) | 5.228 | 1.004 [1.000] | 0.017 | cute | PASS | 1.015 |
| RTX 5090 | 512 | hot | 3.868 | 4.209 (3.581/4.835) | 3.868 | 1.000 [1.000] | 0.005 | flashinfer | PASS | 0.998 |
| RTX 5090 | 512 | engine | 5.769 | 8.069 (6.480/10.235) | 5.797 | 1.007 [1.001] | 0.007 | flashinfer | PASS | 1.006 |
| RTX 5090 | 512 | cold | 10.345 | 10.741 (8.756/13.027) | 10.354 | 1.006 [1.000] | 0.025 | flashinfer | PASS | 1.018 |
| RTX 5090 | 1024 | hot | 6.940 | 7.111 (6.087/8.192) | 7.566 | 1.028 [1.025] | 0.005 | cute | PASS | 1.000 |
| RTX 5090 | 1024 | engine | 10.524 | 15.921 (10.240/21.589) | 11.036 | 1.044 [1.038] | 0.013 | cute | PASS | 1.011 |
| RTX 5090 | 1024 | cold | 21.444 | 21.020 (15.132/26.694) | 20.792 | 0.971 [0.968] | 0.011 | cute | FAIL | 1.010 |
| RTX 5090 | 2048 | hot | 12.344 | 12.345 (10.695/13.815) | 13.426 | 1.000 [0.999] | 0.005 | cute | PASS | 1.000 |
| RTX 5090 | 2048 | engine | 21.642 | 33.575 (21.301/48.996) | 21.191 | 0.980 [0.974] | 0.012 | cute | FAIL | 1.009 |
| RTX 5090 | 2048 | cold | 43.715 | 42.605 (31.143/53.675) | 43.492 | 0.975 [0.973] | 0.005 | cute | FAIL | 1.003 |
| RTX 5090 | 4096 | hot | 23.910 | 67.413 (19.342/93.824) | 25.771 | 1.081 [1.078] | 0.005 | cute | PASS | 1.006 |
| RTX 5090 | 4096 | engine | 50.533 | 84.651 (60.217/107.065) | 51.001 | 1.010 [1.008] | 0.031 | cute | PASS | 1.024 |
| RTX 5090 | 4096 | cold | 87.467 | 85.420 (63.346/107.463) | 87.808 | 0.977 [0.975] | 0.005 | cute | FAIL | 1.001 |
| RTX 5090 | 8192 | hot | 175.154 | 169.074 (115.016/215.347) | 176.441 | 0.966 [0.965] | 0.051 | triton | PASS | 1.053 |
| RTX 5090 | 8192 | engine | 174.848 | 170.610 (126.628/215.230) | 176.327 | 0.976 [0.975] | 0.007 | triton | FAIL | 1.001 |
| RTX 5090 | 8192 | cold | 175.189 | 171.435 (127.659/215.353) | 176.326 | 0.979 [0.978] | 0.005 | triton | FAIL | 0.999 |

### D2 add + RMSNorm with sum_out, standard weight

| device | M | regime | ours us | vLLM us (A/B) | SGLang us | ratio vs faster [CI95 lo] | eps | provider | verdict | xproc |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 | 1 | engine | 1.456 | 2.140 | 2.228 | 1.470 [1.469] | 0.005 | cute | PASS | 1.000 |
| GB10 | 4 | engine | 1.595 | 2.874 | 3.162 | 1.803 [1.799] | 0.005 | cute | PASS | 1.004 |
| GB10 | 16 | engine | 2.569 | 3.901 | 3.755 | 1.462 [1.458] | 0.007 | cute | PASS | 1.001 |
| GB10 | 64 | engine | 6.803 | 9.191 | 7.599 | 1.123 [1.113] | 0.020 | cute | PASS | 1.001 |
| GB10 | 256 | engine | 24.917 | 34.546 | 26.254 | 1.054 [1.048] | 0.011 | flashinfer | PASS | 1.005 |
| GB10 | 1024 | engine | 139.037 | 169.278 | 146.181 | 1.052 [1.050] | 0.005 | flashinfer | PASS | 1.001 |
| PRO 6000 | 1 | engine | 1.594 | 2.352 | 2.437 | 1.476 [1.475] | 0.005 | cute | PASS | 1.000 |
| PRO 6000 | 4 | engine | 1.704 | 2.452 | 3.413 | 1.439 [1.438] | 0.005 | cute | PASS | 1.000 |
| PRO 6000 | 16 | engine | 1.787 | 2.495 | 3.462 | 1.396 [1.394] | 0.005 | cute | PASS | 1.001 |
| PRO 6000 | 64 | engine | 2.419 | 3.618 | 3.766 | 1.495 [1.492] | 0.005 | cute | PASS | 1.002 |
| PRO 6000 | 256 | engine | 5.580 | 6.819 | 5.988 | 1.073 [1.071] | 0.005 | flashinfer | PASS | 1.001 |
| PRO 6000 | 1024 | engine | 14.279 | 20.138 | 15.360 | 1.077 [1.075] | 0.007 | flashinfer | PASS | 1.000 |
| RTX 5090 | 1 | engine | 1.302 | 1.932 | 2.212 | 1.484 [1.483] | 0.005 | cute | PASS | 1.000 |
| RTX 5090 | 4 | engine | 1.413 | 2.015 | 2.839 | 1.427 [1.425] | 0.005 | cute | PASS | 1.000 |
| RTX 5090 | 16 | engine | 1.527 | 2.095 | 2.897 | 1.372 [1.371] | 0.005 | cute | PASS | 1.001 |
| RTX 5090 | 64 | engine | 1.982 | 2.529 | 3.156 | 1.274 [1.273] | 0.007 | cute | PASS | 1.000 |
| RTX 5090 | 256 | engine | 3.938 | 4.636 | 4.848 | 1.175 [1.173] | 0.006 | flashinfer | PASS | 1.001 |
| RTX 5090 | 1024 | engine | 13.363 | 18.716 | 14.108 | 1.059 [1.055] | 0.008 | flashinfer | PASS | 1.005 |

### D2 add + RMSNorm with sum_out, zero-centered

| device | M | regime | ours us | vLLM us (A/B) | SGLang us | ratio vs faster [CI95 lo] | eps | provider | verdict | xproc |
|---|---|---|---|---|---|---|---|---|---|---|
| GB10 | 1 | engine | 1.470 | 2.724 | 2.226 | 1.515 [1.514] | 0.005 | cute | PASS | 1.000 |
| GB10 | 4 | engine | 1.622 | 2.870 | 3.162 | 1.768 [1.767] | 0.005 | cute | PASS | 1.001 |
| GB10 | 16 | engine | 2.600 | 3.419 | 3.787 | 1.318 [1.313] | 0.005 | cute | PASS | 1.002 |
| GB10 | 64 | engine | 6.848 | 9.129 | 7.672 | 1.123 [1.120] | 0.007 | cute | PASS | 1.002 |
| GB10 | 256 | engine | 26.396 | 35.016 | 27.676 | 1.049 [1.046] | 0.009 | cute | PASS | 1.004 |
| GB10 | 1024 | engine | 143.005 | 169.431 | 148.480 | 1.039 [1.038] | 0.005 | triton | PASS | 1.002 |
| PRO 6000 | 1 | engine | 1.630 | 2.334 | 2.436 | 1.432 [1.431] | 0.005 | cute | PASS | 1.000 |
| PRO 6000 | 4 | engine | 1.725 | 3.515 | 3.416 | 1.979 [1.977] | 0.005 | cute | PASS | 1.003 |
| PRO 6000 | 16 | engine | 1.805 | 2.476 | 3.461 | 1.372 [1.370] | 0.005 | cute | PASS | 1.002 |
| PRO 6000 | 64 | engine | 2.424 | 3.601 | 3.766 | 1.488 [1.484] | 0.005 | cute | PASS | 1.002 |
| PRO 6000 | 256 | engine | 5.584 | 6.462 | 5.991 | 1.073 [1.070] | 0.005 | flashinfer | PASS | 1.004 |
| PRO 6000 | 1024 | engine | 14.301 | 20.024 | 15.359 | 1.074 [1.071] | 0.005 | cute | PASS | 1.004 |
| RTX 5090 | 1 | engine | 1.328 | 2.437 | 2.215 | 1.668 [1.666] | 0.005 | cute | PASS | 1.001 |
| RTX 5090 | 4 | engine | 1.417 | 2.488 | 2.839 | 1.756 [1.754] | 0.005 | cute | PASS | 1.001 |
| RTX 5090 | 16 | engine | 1.532 | 2.557 | 2.899 | 1.668 [1.666] | 0.005 | cute | PASS | 1.002 |
| RTX 5090 | 64 | engine | 1.977 | 2.980 | 3.151 | 1.505 [1.502] | 0.005 | cute | PASS | 1.008 |
| RTX 5090 | 256 | engine | 3.706 | 5.246 | 4.848 | 1.312 [1.307] | 0.005 | cute | PASS | 1.005 |
| RTX 5090 | 1024 | engine | 13.824 | 18.599 | 14.076 | 1.019 [1.016] | 0.009 | cute | PASS | 1.002 |


## RMSNORM-W: standard-weight FP32 residual

Scope: the residual-stream norms of Qwen3 dense and
Qwen3-MoE (scale by `weight`) with an FP32 residual, `store_residual=False` and a
BF16 output, under the zero-centered family's numerics contract. Every
residual-stream form now takes `zero_centered=False`: inference `add_rms_norm_` /
`AddRMSNorm` with an FP32 residual (and `sum_out`, `store_residual=False`,
`normalize="stored_sum"`), inference `residual_rms_norm`, and the training
`add_rms_norm` (frozen weight, FP32 residual or `save="inputs"`) and
`residual_rms_norm`. The CuTe kernel loads and stores an FP32 residual as two
128-bit vectors per 8-element chunk; the Triton residual-stream kernels take a
`ZERO_CENTERED` constexpr (zero-centered bits unchanged).

**Dispatch.** `_FP32_RESIDUAL` in `norm/_dispatch.py` holds the best
worst-regime provider per device, H in {2048, 4096}, M band and static flag, from
paired `--residual-dtype fp32 --sweep` runs (20 rounds; CuTe configs and Triton).
Both weight kinds and both x dtypes share the bands (the kernels differ only by
the +1 on the scale). FlashInfer has no mixed-dtype norm and is never selected.

Cluster row splitting is opt-in through `AddRMSNorm(config=Config(..., split=...))`
or `CTI_NORM_CONFIG`, never auto-dispatched. Paired 30-round FP32-residual,
BF16-x measurements with PDL early trigger in the same process found it slower:

| Device | H | M | Config | cold / engine / hot (µs) |
| --- | --- | --- | --- | --- |
| GB10 | 2048 | 64 | table cute_t256_r2 | 7.68 / 5.40 / 2.49 |
| GB10 | 2048 | 64 | c2, 128 threads | 8.49 / 6.45 / 2.86 |
| GB10 | 2048 | 64 | c3, 64 threads (former rule) | 8.98 / 6.86 / 3.41 |
| GB10 | 2048 | 64 | c3, 128 threads | 8.97 / — / — |
| GB10 | 2048 | 64 | c4 | 9.05 / — / — |
| RTX 5090 | 2048 | 256 | table cute_t512_r1 | 3.99 / 3.24 / 2.06 |
| RTX 5090 | 2048 | 256 | c2 | 4.81 / 4.08 / 2.79 |
| RTX 5090 | 2048 | 256 | c4 | 6.04 / 5.32 / 3.72 |
| RTX 5090 | 2048 | 256 | c7 (former rule) | 8.96 / 8.26 / 5.93 |
| RTX 5090 | 2048 | 256 | c8 | 9.38 / — / — |

Cluster launch and DSMEM barrier costs exceed the balance gain. On GB10, the
32-CTA (two rows/CTA) and 64-CTA (one row/CTA) layouts take the same time,
contradicting the SM-imbalance hypothesis there. Forced configs and the kernel's
cluster path remain available for experiments.

**Market and bar.** Neither market has an FP32-residual kernel on its default path for
these models. vLLM 0.31.0 has no FP32-residual mode. SGLang 72730d6 enables
`fp32_residual=True` for Qwen3 only in its RL on-policy mode, where
`RMSNorm.forward_cuda` falls back to `forward_native`; Qwen3-MoE has no such mode.
Rule for `residual_dtype=float32` rows only:

- (a) every row must be >= the like-for-like market, SGLang's FP32-residual path in
  its strongest form, on every device. That path has two columns: `sgl_fp32`, the
  `forward_native` arithmetic in torch eager (the default), and `sgl_fp32c`, the same
  function under `torch.compile(fullgraph=True, dynamic=False)` with default Inductor
  options (what SGLang runs with `--enable-torch-compile`). The bar is ours against
  the faster of the two, per paired round (`fp32_market_ratio`), with median >= 1 and
  CI95 low >= 1 - eps.
- (b) every traffic-bound row reaches the SOL bar, against the measured device
  bandwidth, for the bytes the FP32 contract moves: 12 B per hidden element
  (x 2 + residual in 4 + residual out 4 + out 2) plus the weight (2 H). The bar is
  CuTe >= 0.95 and Triton >= 0.90. Bandwidth is measured once per bench process: the
  median of 20 CUDA-event reps on cold buffers of max(512 MiB, 4*L2) for an FP32
  copy, an FP32 add and a read-only sum. This op writes half its traffic, so the
  SOL bound uses the measured mixed read/write witness: `bw_gbs = max(copy, add)`. The (b) bound for this read+write op is the slower
  of bytes / that mixed bandwidth and the paired, size-matched `rw_copy` witness
  of the same bytes in the same regime: an
  op that must read and write those bytes cannot beat a copy of them. The bar
  (CuTe >= 0.95, Triton >= 0.90) applies to ours against that bound.
  GB10 sustains 216–222 GB/s for those witnesses versus 235 GB/s for read-only sum.
  All three witnesses remain in receipts; HEADER/ROW also record the sum separately
  as `bw_read_gbs`. The read-only sum is a recorded witness only; its rate does not
  enter t_dram or the SOL gate.
  A row is traffic-bound when its DRAM floor (bytes / bandwidth) is >= 4 µs; smaller
  rows are latency-bound. Hot rows are L2-resident and are not judged by (b).
- Each market's real BF16-residual kernel stays disclosed per row (vLLM Inductor;
  SGLang -> FlashInfer 0.7.1 CuTe norm). It moves 8 B per element against our 12, so in
  traffic-bound rows it is about 0.67 of our cost or less. It is not the gate for this
  mode. The `zero_centered=False` BF16-residual rows keep the normal bar.

The paired `rw_copy` column is a diagnostic, excluded from market ratios,
verdicts and bar (a). Enabled by default for `--op add_norm --residual-dtype fp32`
(disable with `--no-rw-copy`), it times one FP32 `torch.Tensor.copy_` per layer,
with `ceil(3*M*H/2)` elements: 6*M*H bytes read and 6*M*H bytes written for
aligned benchmark shapes. It uses the same L-call graph chain, paired rounds and
pool rotation/restoration regime as ours: hot reuses one resident pair across L
calls; engine/cold rotate pairs across the same number of graphs. Restoration
runs outside timing before each candidate's sample. This measures the time a pure streaming
read+write kernel needs for the same bytes in the same regime, including the
per-kernel DRAM latency and launch/drain cost every implementation pays. Receipts
report `us.rw_copy` and `sol_witness = {name: t_rw_copy / t_name}` for ours,
static, Triton and local CuTe candidates. The weight vector is excluded from this
size-matched diagnostic; the existing bandwidth SOL formula still includes it.

**vLLM column at H=2048.** It runs the real vLLM 0.31.0 Inductor output for
Qwen3-30B-A3B (hidden 2048), captured from vLLM's own compile cache (2-layer
dummy-weight engine, default compilation config). Site A is the post-attention norm
that feeds `moe_forward`: it stores the normalized output twice and no residual. Site B
is the next layer's input norm, which rematerializes the residual; its body equals the
H=4096 site B with 2048 in place of 4096. The capture has no `.best_config` (the engine
stopped before its first forward), so both kernels autotune at runtime from their own
decorator, as vLLM does on first use. The launch geometry check requires the probed
grid/block to equal the launcher the autotuner selected: grid = ceil(M / XBLOCK) and
block = 32 x num_warps. Compiled-expression stand-ins are no longer accepted at any
width: compiled mode requires source/ABI equality with the captured file, and geometry
verification is never disabled.

**Numerics.** The stored FP32 sum equals torch's FP32 add bitwise (so
`fp32_sum` == `stored_sum`); the normalized output passes the FP64
best-comparator gate against torch FP32 eager and SGLang's native FP32-residual
arithmetic. Triton inference equals the training forward bitwise; CuTe rows are
qualified by accuracy only (train/infer bitwise invariance is not required,
decision 2026-10-15).

Receipts (2026-10-17): shipped dispatch (`ours`, `static_weight=False`) and the
`static_weight=True` opt-in, 30 rounds, paired in one process, FlashInfer 0.7.0.post1
overlay; devices in this round: GB10, RTX 5090, RTX PRO 6000. (a) = ours vs the faster SGLang
FP32-residual column (median, CI95 low). These are the same 30-round receipts,
restated under the 2026-10-17 rule: the DRAM floor and (b) SOL use the
measured mixed read/write bandwidth, max of the FP32 copy and add witnesses. The
read-only sum remains recorded as `bw_read_gbs` only. The bar is given for
traffic-bound rows only. The BF16 columns are
disclosed and are not the gate for this mode (vLLM at H=2048 = Qwen3-30B-A3B capture,
at H=4096 = Qwen3-8B).

| Device | H | M | Regime | ours (µs) | static (µs) | SGLang FP32 eager (µs) | SGLang FP32 compiled (µs) | (a) ours vs faster FP32 market (CI95 low) | DRAM floor (µs) | (b) SOL ours | (b) bar | vLLM BF16 (µs) | SGLang BF16 (µs) | vs faster BF16 market (disclosed) | provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GB10 | 4096 | 1 | hot | 1.40 | 1.28 | 12.06 | 1.73 | 1.24 (1.22) | 0.26 | 0.184 | latency-bound | 1.54 | 1.39 | 1.004 (0.982) | cute |
| GB10 | 4096 | 1 | engine | 1.61 | 1.50 | 13.44 | 2.24 | 1.39 (1.39) | 0.26 | 0.161 | latency-bound | 2.65 | 1.65 | 1.027 (1.025) | cute |
| GB10 | 4096 | 1 | cold | 1.69 | 1.57 | 14.47 | 2.33 | 1.38 (1.38) | 0.26 | 0.152 | latency-bound | 2.14 | 1.73 | 1.019 (1.017) | cute |
| GB10 | 4096 | 4 | hot | 1.60 | 1.39 | 13.71 | 1.99 | 1.24 (1.23) | 0.92 | 0.579 | latency-bound | 1.65 | 1.99 | 1.033 (1.027) | cute |
| GB10 | 4096 | 4 | engine | 1.86 | 1.76 | 15.08 | 2.51 | 1.34 (1.33) | 0.92 | 0.496 | latency-bound | 2.18 | 2.22 | 1.169 (1.154) | cute |
| GB10 | 4096 | 4 | cold | 2.17 | 2.01 | 16.03 | 2.68 | 1.24 (1.23) | 0.92 | 0.425 | latency-bound | 2.35 | 2.46 | 1.085 (1.078) | cute |
| GB10 | 4096 | 16 | hot | 1.64 | 1.39 | 18.66 | 2.84 | 1.74 (1.70) | 3.58 | 2.186 | latency-bound | 2.13 | 1.99 | 1.221 (1.211) | cute |
| GB10 | 4096 | 16 | engine | 3.43 | 3.39 | 20.38 | 4.19 | 1.22 (1.21) | 3.58 | 1.043 | latency-bound | 3.69 | 2.87 | 0.836 (0.831) | cute |
| GB10 | 4096 | 16 | cold | 4.52 | 4.46 | 21.46 | 4.66 | 1.03 (1.02) | 3.58 | 0.792 | latency-bound | 4.34 | 3.96 | 0.883 (0.864) | cute |
| GB10 | 4096 | 64 | hot | 3.38 | 3.36 | 25.79 | 11.27 | 3.35 (3.28) | 14.21 | 4.210 | 0.95 pass | 2.50 | 2.63 | 0.744 (0.733) | cute |
| GB10 | 4096 | 64 | engine | 10.21 | 10.26 | 31.12 | 12.68 | 1.25 (1.24) | 14.21 | 1.392 | 0.95 pass | 7.90 | 5.85 | 0.573 (0.568) | cute |
| GB10 | 4096 | 64 | cold | 14.58 | 14.60 | 33.93 | 14.96 | 1.03 (1.03) | 14.21 | 0.975 | 0.95 pass | 10.58 | 10.21 | 0.706 (0.700) | cute |
| GB10 | 4096 | 256 | hot | 13.39 | 16.68 | 135.13 | 50.69 | 3.86 (3.39) | 56.74 | 4.237 | 0.90 pass | 5.92 | 6.83 | 0.494 (0.357) | triton |
| GB10 | 4096 | 256 | engine | 40.20 | 40.21 | 140.44 | 51.87 | 1.34 (1.30) | 56.74 | 1.411 | 0.90 pass | 29.95 | 19.68 | 0.499 (0.489) | triton |
| GB10 | 4096 | 256 | cold | 57.32 | 57.36 | 144.62 | 57.81 | 1.01 (1.00) | 56.74 | 0.990 | 0.90 pass | 38.84 | 38.09 | 0.666 (0.663) | triton |
| GB10 | 4096 | 1024 | hot | 164.14 | 163.75 | 944.91 | 224.24 | 1.36 (1.33) | 226.84 | 1.382 | 0.95 pass | 144.90 | 81.44 | 0.496 (0.488) | cute |
| GB10 | 4096 | 1024 | engine | 203.01 | 203.14 | 905.98 | 226.48 | 1.11 (1.10) | 226.84 | 1.117 | 0.95 pass | 149.30 | 111.74 | 0.554 (0.549) | cute |
| GB10 | 4096 | 1024 | cold | 224.47 | 225.18 | 890.29 | 227.20 | 1.01 (1.01) | 226.84 | 1.011 | 0.95 pass | 150.83 | 151.60 | 0.672 (0.671) | cute |
| GB10 | 2048 | 1 | hot | 1.06 | 0.97 | 11.87 | 1.48 | 1.40 (1.39) | 0.13 | 0.122 | latency-bound | 1.64 | 1.14 | 1.098 (1.075) | cute |
| GB10 | 2048 | 1 | engine | 1.33 | 1.23 | 13.26 | 2.64 | 1.99 (1.97) | 0.13 | 0.098 | latency-bound | 2.65 | 1.51 | 1.140 (1.139) | cute |
| GB10 | 2048 | 1 | cold | 1.36 | 1.28 | 14.29 | 2.65 | 1.95 (1.95) | 0.13 | 0.095 | latency-bound | 2.69 | 1.54 | 1.134 (1.133) | cute |
| GB10 | 2048 | 4 | hot | 1.09 | 1.02 | 13.06 | 2.33 | 2.13 (2.11) | 0.46 | 0.425 | latency-bound | 1.99 | 1.64 | 1.511 (1.499) | cute |
| GB10 | 2048 | 4 | engine | 1.40 | 1.34 | 14.05 | 3.34 | 2.38 (2.37) | 0.46 | 0.330 | latency-bound | 2.93 | 1.97 | 1.403 (1.400) | cute |
| GB10 | 2048 | 4 | cold | 1.51 | 1.45 | 15.08 | 3.39 | 2.25 (2.23) | 0.46 | 0.306 | latency-bound | 3.00 | 2.18 | 1.444 (1.432) | cute |
| GB10 | 2048 | 16 | hot | 1.19 | 1.05 | 15.14 | 2.41 | 2.02 (2.01) | 1.79 | 1.502 | latency-bound | 2.03 | 1.93 | 1.624 (1.618) | cute |
| GB10 | 2048 | 16 | engine | 2.29 | 2.20 | 16.44 | 3.61 | 1.58 (1.57) | 1.79 | 0.785 | latency-bound | 3.10 | 2.26 | 0.987 (0.982) | cute |
| GB10 | 2048 | 16 | cold | 2.79 | 2.75 | 17.37 | 3.83 | 1.37 (1.35) | 1.79 | 0.642 | latency-bound | 3.46 | 2.79 | 0.999 (0.978) | cute |
| GB10 | 2048 | 64 | hot | 2.37 | 1.62 | 18.72 | 5.83 | 2.46 (2.42) | 7.12 | 3.005 | 0.95 pass | 2.20 | 1.94 | 0.825 (0.815) | cute |
| GB10 | 2048 | 64 | engine | 5.41 | 5.52 | 21.39 | 7.32 | 1.35 (1.33) | 7.12 | 1.317 | 0.95 pass | 4.94 | 3.79 | 0.700 (0.700) | cute |
| GB10 | 2048 | 64 | cold | 7.63 | 7.73 | 22.66 | 8.06 | 1.06 (1.05) | 7.12 | 0.933 | 0.95 GAP | 6.06 | 6.00 | 0.795 (0.785) | cute |
| GB10 | 2048 | 256 | hot | 5.91 | 5.58 | 41.07 | 23.68 | 3.99 (3.95) | 28.43 | 4.809 | 0.95 pass | 3.92 | 3.69 | 0.625 (0.624) | cute |
| GB10 | 2048 | 256 | engine | 19.06 | 19.16 | 50.55 | 27.22 | 1.35 (1.31) | 28.43 | 1.491 | 0.95 pass | 15.22 | 10.42 | 0.545 (0.544) | cute |
| GB10 | 2048 | 256 | cold | 28.39 | 28.61 | 55.74 | 29.44 | 1.04 (1.03) | 28.43 | 1.002 | 0.95 pass | 19.97 | 19.48 | 0.690 (0.683) | cute |
| GB10 | 2048 | 1024 | hot | 36.09 | 47.99 | 326.95 | 111.74 | 3.13 (2.46) | 113.67 | 3.150 | 0.90 pass | 43.64 | 18.33 | 0.527 (0.382) | triton |
| GB10 | 2048 | 1024 | engine | 88.28 | 88.12 | 317.90 | 117.40 | 1.34 (1.29) | 113.67 | 1.288 | 0.90 pass | 76.64 | 44.15 | 0.507 (0.498) | triton |
| GB10 | 2048 | 1024 | cold | 113.01 | 113.00 | 321.24 | 118.31 | 1.05 (1.04) | 113.67 | 1.006 | 0.90 pass | 85.94 | 75.89 | 0.671 (0.670) | triton |
| RTX 5090 | 4096 | 1 | hot | 1.29 | 1.16 | 11.49 | 1.62 | 1.26 (1.23) | 0.04 | 0.028 | latency-bound | 1.45 | 1.25 | 0.969 (0.957) | cute |
| RTX 5090 | 4096 | 1 | engine | 1.42 | 1.36 | 12.82 | 2.03 | 1.43 (1.43) | 0.04 | 0.026 | latency-bound | 1.88 | 1.47 | 1.039 (1.037) | cute |
| RTX 5090 | 4096 | 1 | cold | 1.52 | 1.41 | 13.61 | 2.08 | 1.38 (1.37) | 0.04 | 0.024 | latency-bound | 1.95 | 1.53 | 1.012 (1.006) | cute |
| RTX 5090 | 4096 | 4 | hot | 1.44 | 1.27 | 13.03 | 1.82 | 1.27 (1.26) | 0.13 | 0.090 | latency-bound | 1.55 | 1.76 | 1.074 (1.065) | cute |
| RTX 5090 | 4096 | 4 | engine | 1.57 | 1.44 | 14.32 | 2.13 | 1.36 (1.35) | 0.13 | 0.083 | latency-bound | 1.93 | 1.93 | 1.225 (1.224) | cute |
| RTX 5090 | 4096 | 4 | cold | 1.63 | 1.52 | 15.13 | 2.20 | 1.34 (1.34) | 0.13 | 0.080 | latency-bound | 1.98 | 2.14 | 1.213 (1.208) | cute |
| RTX 5090 | 4096 | 16 | hot | 1.46 | 1.30 | 16.59 | 1.86 | 1.27 (1.26) | 0.51 | 0.345 | latency-bound | 1.59 | 1.76 | 1.081 (1.068) | triton |
| RTX 5090 | 4096 | 16 | engine | 1.80 | 1.57 | 17.81 | 2.16 | 1.20 (1.20) | 0.51 | 0.281 | latency-bound | 1.99 | 2.01 | 1.106 (1.105) | triton |
| RTX 5090 | 4096 | 16 | cold | 1.94 | 1.69 | 18.67 | 2.27 | 1.17 (1.16) | 0.51 | 0.260 | latency-bound | 2.07 | 2.28 | 1.068 (1.064) | triton |
| RTX 5090 | 4096 | 64 | hot | 1.60 | 1.32 | 18.47 | 2.21 | 1.39 (1.35) | 2.00 | 1.251 | latency-bound | 1.73 | 1.85 | 1.078 (1.072) | cute |
| RTX 5090 | 4096 | 64 | engine | 2.54 | 2.41 | 19.50 | 2.82 | 1.13 (1.11) | 2.00 | 0.789 | latency-bound | 2.44 | 2.30 | 0.907 (0.905) | cute |
| RTX 5090 | 4096 | 64 | cold | 2.99 | 2.78 | 20.32 | 3.16 | 1.06 (1.05) | 2.00 | 0.671 | latency-bound | 2.74 | 2.85 | 0.917 (0.912) | cute |
| RTX 5090 | 4096 | 256 | hot | 3.47 | 3.24 | 26.73 | 5.73 | 1.66 (1.65) | 8.00 | 2.306 | 0.95 pass | 2.44 | 2.67 | 0.705 (0.699) | cute |
| RTX 5090 | 4096 | 256 | engine | 5.16 | 5.21 | 28.76 | 6.93 | 1.35 (1.34) | 8.00 | 1.550 | 0.95 pass | 4.34 | 3.74 | 0.725 (0.722) | cute |
| RTX 5090 | 4096 | 256 | cold | 7.54 | 7.65 | 30.55 | 7.81 | 1.04 (1.03) | 8.00 | 1.061 | 0.95 pass | 5.63 | 5.22 | 0.688 (0.683) | cute |
| RTX 5090 | 4096 | 1024 | hot | 12.80 | 12.34 | 64.21 | 27.34 | 2.13 (2.11) | 31.99 | 2.500 | 0.95 pass | 6.83 | 7.68 | 0.533 (0.533) | cute |
| RTX 5090 | 4096 | 1024 | engine | 21.25 | 21.29 | 68.96 | 28.88 | 1.37 (1.33) | 31.99 | 1.506 | 0.95 pass | 15.87 | 11.21 | 0.526 (0.523) | cute |
| RTX 5090 | 4096 | 1024 | cold | 32.06 | 32.17 | 73.76 | 32.53 | 1.01 (1.01) | 31.99 | 0.998 | 0.95 pass | 21.10 | 20.82 | 0.649 (0.647) | cute |
| RTX 5090 | 2048 | 1 | hot | 0.94 | 0.87 | 10.88 | 1.37 | 1.44 (1.42) | 0.02 | 0.019 | latency-bound | 1.49 | 1.02 | 1.073 (1.059) | cute |
| RTX 5090 | 2048 | 1 | engine | 1.16 | 1.09 | 12.17 | 2.35 | 2.03 (2.03) | 0.02 | 0.016 | latency-bound | 2.43 | 1.31 | 1.133 (1.131) | cute |
| RTX 5090 | 2048 | 1 | cold | 1.19 | 1.14 | 12.99 | 2.37 | 2.00 (1.98) | 0.02 | 0.015 | latency-bound | 2.46 | 1.34 | 1.128 (1.126) | cute |
| RTX 5090 | 2048 | 4 | hot | 1.02 | 0.94 | 12.12 | 2.13 | 2.07 (2.06) | 0.07 | 0.063 | latency-bound | 1.81 | 1.42 | 1.398 (1.389) | cute |
| RTX 5090 | 2048 | 4 | engine | 1.23 | 1.17 | 13.32 | 2.97 | 2.43 (2.41) | 0.07 | 0.053 | latency-bound | 2.64 | 1.76 | 1.437 (1.433) | cute |
| RTX 5090 | 2048 | 4 | cold | 1.25 | 1.21 | 14.18 | 3.03 | 2.41 (2.40) | 0.07 | 0.052 | latency-bound | 2.71 | 1.99 | 1.586 (1.582) | cute |
| RTX 5090 | 2048 | 16 | hot | 1.10 | 0.97 | 13.88 | 2.24 | 2.04 (2.02) | 0.25 | 0.230 | latency-bound | 1.89 | 1.71 | 1.547 (1.535) | cute |
| RTX 5090 | 2048 | 16 | engine | 1.33 | 1.30 | 14.97 | 2.99 | 2.25 (2.22) | 0.25 | 0.190 | latency-bound | 2.67 | 1.89 | 1.421 (1.418) | cute |
| RTX 5090 | 2048 | 16 | cold | 1.41 | 1.36 | 15.84 | 3.03 | 2.14 (2.13) | 0.25 | 0.179 | latency-bound | 2.77 | 2.08 | 1.474 (1.471) | cute |
| RTX 5090 | 2048 | 64 | hot | 1.16 | 1.05 | 14.45 | 2.26 | 1.96 (1.94) | 1.00 | 0.864 | latency-bound | 1.93 | 1.71 | 1.478 (1.460) | cute |
| RTX 5090 | 2048 | 64 | engine | 1.75 | 1.70 | 15.54 | 3.14 | 1.80 (1.79) | 1.00 | 0.572 | latency-bound | 2.80 | 2.02 | 1.156 (1.151) | cute |
| RTX 5090 | 2048 | 64 | cold | 2.02 | 1.96 | 16.37 | 3.25 | 1.61 (1.60) | 1.00 | 0.495 | latency-bound | 2.94 | 2.32 | 1.148 (1.138) | cute |
| RTX 5090 | 2048 | 256 | hot | 2.07 | 1.76 | 18.18 | 3.33 | 1.61 (1.60) | 4.00 | 1.932 | latency-bound | 2.34 | 1.76 | 0.862 (0.853) | cute |
| RTX 5090 | 2048 | 256 | engine | 3.23 | 3.31 | 19.46 | 4.50 | 1.43 (1.38) | 4.00 | 1.236 | latency-bound | 3.63 | 2.74 | 0.847 (0.843) | cute |
| RTX 5090 | 2048 | 256 | cold | 3.96 | 4.09 | 20.30 | 4.97 | 1.25 (1.24) | 4.00 | 1.008 | latency-bound | 4.20 | 3.46 | 0.877 (0.868) | cute |
| RTX 5090 | 2048 | 1024 | hot | 6.60 | 6.01 | 35.71 | 11.99 | 1.82 (1.80) | 15.98 | 2.422 | 0.95 pass | 4.78 | 3.98 | 0.603 (0.602) | cute |
| RTX 5090 | 2048 | 1024 | engine | 10.14 | 10.17 | 38.41 | 13.59 | 1.35 (1.34) | 15.98 | 1.575 | 0.95 pass | 7.92 | 5.62 | 0.554 (0.553) | cute |
| RTX 5090 | 2048 | 1024 | cold | 15.18 | 15.50 | 41.49 | 16.18 | 1.06 (1.06) | 15.98 | 1.052 | 0.95 pass | 10.47 | 10.41 | 0.684 (0.679) | cute |
| RTX PRO 6000 | 4096 | 1 | hot | 1.56 | 1.42 | 13.43 | 1.95 | 1.25 (1.24) | 0.04 | 0.025 | latency-bound | 1.76 | 1.48 | 0.950 (0.948) | cute |
| RTX PRO 6000 | 4096 | 1 | engine | 1.72 | 1.63 | 14.76 | 2.42 | 1.41 (1.41) | 0.04 | 0.022 | latency-bound | 2.24 | 1.76 | 1.023 (1.021) | cute |
| RTX PRO 6000 | 4096 | 1 | cold | 1.80 | 1.69 | 15.60 | 2.50 | 1.39 (1.39) | 0.04 | 0.021 | latency-bound | 2.31 | 1.83 | 1.019 (1.012) | cute |
| RTX PRO 6000 | 4096 | 4 | hot | 1.79 | 1.56 | 15.59 | 2.24 | 1.25 (1.24) | 0.14 | 0.077 | latency-bound | 1.89 | 2.10 | 1.058 (1.047) | cute |
| RTX PRO 6000 | 4096 | 4 | engine | 1.87 | 1.74 | 16.59 | 2.51 | 1.34 (1.34) | 0.14 | 0.074 | latency-bound | 2.29 | 2.26 | 1.213 (1.213) | cute |
| RTX PRO 6000 | 4096 | 4 | cold | 1.95 | 1.82 | 17.47 | 2.61 | 1.34 (1.33) | 0.14 | 0.071 | latency-bound | 2.35 | 2.51 | 1.209 (1.202) | cute |
| RTX PRO 6000 | 4096 | 16 | hot | 1.88 | 1.59 | 19.91 | 2.30 | 1.22 (1.21) | 0.53 | 0.284 | latency-bound | 1.93 | 2.19 | 1.028 (1.011) | cute |
| RTX PRO 6000 | 4096 | 16 | engine | 1.98 | 1.90 | 20.75 | 2.61 | 1.32 (1.32) | 0.53 | 0.270 | latency-bound | 2.35 | 2.41 | 1.187 (1.186) | cute |
| RTX PRO 6000 | 4096 | 16 | cold | 2.19 | 2.01 | 21.66 | 2.74 | 1.25 (1.25) | 0.53 | 0.244 | latency-bound | 3.03 | 2.72 | 1.240 (1.237) | cute |
| RTX PRO 6000 | 4096 | 64 | hot | 1.95 | 1.82 | 21.60 | 2.50 | 1.26 (1.26) | 2.12 | 1.088 | latency-bound | 2.10 | 2.27 | 1.077 (1.060) | cute |
| RTX PRO 6000 | 4096 | 64 | engine | 3.02 | 2.85 | 22.66 | 3.48 | 1.17 (1.15) | 2.12 | 0.703 | latency-bound | 2.98 | 2.76 | 0.916 (0.912) | cute |
| RTX PRO 6000 | 4096 | 64 | cold | 3.68 | 3.54 | 23.57 | 3.93 | 1.07 (1.06) | 2.12 | 0.576 | latency-bound | 3.27 | 3.42 | 0.888 (0.881) | cute |
| RTX PRO 6000 | 4096 | 256 | hot | 3.64 | 3.08 | 28.91 | 6.79 | 1.88 (1.86) | 8.47 | 2.329 | 0.95 pass | 3.24 | 2.50 | 0.688 (0.686) | cute |
| RTX PRO 6000 | 4096 | 256 | engine | 7.00 | 7.02 | 31.61 | 8.52 | 1.23 (1.22) | 8.47 | 1.210 | 0.95 pass | 5.91 | 4.78 | 0.684 (0.678) | cute |
| RTX PRO 6000 | 4096 | 256 | cold | 9.72 | 9.70 | 33.04 | 9.81 | 1.01 (1.00) | 8.47 | 0.872 | 0.95 GAP | 8.04 | 7.67 | 0.789 (0.777) | cute |
| RTX PRO 6000 | 4096 | 1024 | hot | 10.87 | 10.41 | 64.71 | 27.53 | 2.53 (2.50) | 33.88 | 3.118 | 0.95 pass | 6.66 | 6.88 | 0.613 (0.613) | cute |
| RTX PRO 6000 | 4096 | 1024 | engine | 22.67 | 22.84 | 74.21 | 29.30 | 1.29 (1.29) | 33.88 | 1.495 | 0.95 pass | 18.62 | 12.06 | 0.532 (0.531) | cute |
| RTX PRO 6000 | 4096 | 1024 | cold | 33.71 | 33.85 | 83.20 | 33.71 | 1.00 (1.00) | 33.88 | 1.005 | 0.95 pass | 24.69 | 22.36 | 0.662 (0.660) | cute |
| RTX PRO 6000 | 2048 | 1 | hot | 1.13 | 1.08 | 13.09 | 1.69 | 1.50 (1.49) | 0.02 | 0.017 | latency-bound | 1.83 | 1.25 | 1.111 (1.100) | cute |
| RTX PRO 6000 | 2048 | 1 | engine | 1.36 | 1.30 | 14.34 | 2.78 | 2.04 (2.03) | 0.02 | 0.014 | latency-bound | 2.91 | 1.57 | 1.156 (1.155) | cute |
| RTX PRO 6000 | 2048 | 1 | cold | 1.41 | 1.35 | 15.19 | 2.80 | 1.99 (1.98) | 0.02 | 0.014 | latency-bound | 2.95 | 1.59 | 1.125 (1.123) | cute |
| RTX PRO 6000 | 2048 | 4 | hot | 1.30 | 1.16 | 14.85 | 2.62 | 2.02 (2.02) | 0.07 | 0.053 | latency-bound | 2.25 | 1.82 | 1.390 (1.374) | cute |
| RTX PRO 6000 | 2048 | 4 | engine | 1.47 | 1.41 | 15.77 | 3.55 | 2.40 (2.40) | 0.07 | 0.047 | latency-bound | 3.14 | 2.07 | 1.401 (1.399) | cute |
| RTX PRO 6000 | 2048 | 4 | cold | 1.53 | 1.46 | 16.70 | 3.58 | 2.35 (2.34) | 0.07 | 0.045 | latency-bound | 3.23 | 2.36 | 1.548 (1.544) | cute |
| RTX PRO 6000 | 2048 | 16 | hot | 1.31 | 1.18 | 16.89 | 2.73 | 2.09 (2.06) | 0.27 | 0.204 | latency-bound | 2.32 | 2.08 | 1.579 (1.560) | cute |
| RTX PRO 6000 | 2048 | 16 | engine | 1.57 | 1.52 | 17.75 | 3.56 | 2.27 (2.24) | 0.27 | 0.170 | latency-bound | 3.15 | 2.27 | 1.447 (1.447) | cute |
| RTX PRO 6000 | 2048 | 16 | cold | 1.65 | 1.61 | 18.67 | 3.58 | 2.17 (2.17) | 0.27 | 0.162 | latency-bound | 3.30 | 2.50 | 1.512 (1.510) | cute |
| RTX PRO 6000 | 2048 | 64 | hot | 1.40 | 1.25 | 17.41 | 2.79 | 2.02 (1.98) | 1.06 | 0.759 | latency-bound | 2.38 | 2.10 | 1.516 (1.497) | cute |
| RTX PRO 6000 | 2048 | 64 | engine | 2.11 | 2.01 | 18.39 | 3.77 | 1.79 (1.78) | 1.06 | 0.503 | latency-bound | 3.31 | 2.42 | 1.146 (1.145) | cute |
| RTX PRO 6000 | 2048 | 64 | cold | 2.44 | 2.38 | 19.23 | 3.93 | 1.61 (1.60) | 1.06 | 0.435 | latency-bound | 3.50 | 2.75 | 1.128 (1.117) | cute |
| RTX PRO 6000 | 2048 | 256 | hot | 2.41 | 1.83 | 21.08 | 4.24 | 1.76 (1.74) | 4.23 | 1.757 | 0.95 pass | 2.77 | 2.22 | 0.923 (0.916) | cute |
| RTX PRO 6000 | 2048 | 256 | engine | 4.23 | 4.32 | 22.80 | 5.67 | 1.37 (1.33) | 4.23 | 1.000 | 0.95 pass | 4.43 | 3.24 | 0.766 (0.763) | cute |
| RTX PRO 6000 | 2048 | 256 | cold | 5.57 | 5.63 | 23.71 | 6.29 | 1.13 (1.12) | 4.23 | 0.761 | 0.95 GAP | 5.08 | 4.64 | 0.831 (0.825) | cute |
| RTX PRO 6000 | 2048 | 1024 | hot | 5.18 | 5.40 | 38.71 | 11.92 | 2.30 (2.26) | 16.92 | 3.269 | 0.95 pass | 4.78 | 4.10 | 0.791 (0.791) | cute |
| RTX PRO 6000 | 2048 | 1024 | engine | 11.99 | 11.98 | 41.89 | 14.34 | 1.21 (1.19) | 16.92 | 1.412 | 0.95 pass | 10.71 | 7.28 | 0.608 (0.605) | cute |
| RTX PRO 6000 | 2048 | 1024 | cold | 16.92 | 16.78 | 45.07 | 17.35 | 1.03 (1.03) | 16.92 | 1.000 | 0.95 pass | 13.71 | 12.74 | 0.751 (0.744) | cute |

- measured mixed read/write bandwidth GB10: 221 GB/s (max of FP32 copy, add)
- measured mixed read/write bandwidth RTX 5090: 1575 GB/s (max of FP32 copy, add)
- measured mixed read/write bandwidth RTX PRO 6000: 1487 GB/s (max of FP32 copy, add)

**Verdict.** (a) is unchanged: every row of every device passes, at 1.00-3.99x
against the faster SGLang FP32-residual column. Under the mixed bound, (b) passes
on every RTX 5090 row. GB10 H2048 M256 cold (1.002) and H4096 M1024 cold (1.011)
now pass; H4096 M64 cold also passes (0.975 here, 0.964 in Round 3 below).
Three cold rows are below 0.95 against bytes / mixed bandwidth alone: GB10 H2048 M64 (0.933), RTX PRO 6000
H2048 M256 (0.761), and RTX PRO 6000 H4096 M256 (0.872); they pass the accepted bound (Round 3).

### Round 3 (2026-10-17): row split and size-matched witness

Row split was implemented with a CuTe cluster of 2-8 CTAs per row, DSMEM push of
per-CTA sums of squares, `Config.split`, and a launch-time balance rule. It was
slower than the table configs on every measured row of GB10, RTX 5090 and
RTX PRO 6000. See the split table in **Dispatch** above for GB10 and RTX 5090.
On RTX PRO 6000, H2048 M256 cold took 6.03 µs (c2) / 7.18 µs (c4) versus
5.53 µs; H4096 M256 cold took 10.25 µs (c2) / 10.33 µs (c4) versus 9.79 µs.
The balance rule was removed; split remains opt-in only.

The paired size-matched `rw_copy` witness is an FP32 copy of the same 6·M·H
bytes read and 6·M·H bytes written, using the same L-layer chain and regime.
These receipts use 30 rounds, one process per device and width, and the
FlashInfer 0.7.0.post1 overlay.

| Device | H | M | Regime | ours (µs) | rw_copy witness (µs) | ours vs witness | DRAM floor, mixed (µs) | (b) SOL ours | Triton (µs) | best forced split (µs) | provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GB10 | 4096 | 64 | hot | 3.36 | 2.63 | 0.785 | 14.20 | 4.230 | 3.14 | — | cute_t256_r2_p1_s0_te |
| GB10 | 4096 | 64 | engine | 10.08 | 14.58 | 1.447 | 14.20 | 1.409 | 11.03 | — | cute_t256_r2_p1_s0_te |
| GB10 | 4096 | 64 | cold | 14.72 | 14.81 | 1.006 | 14.20 | 0.964 | 15.69 | — | cute_t256_r2_p1_s0_te |
| GB10 | 2048 | 64 | hot | 2.39 | 1.76 | 0.736 | 7.14 | 2.987 | 2.39 | — | cute_t256_r2_p1_s0_te |
| GB10 | 2048 | 64 | engine | 5.41 | 7.66 | 1.414 | 7.14 | 1.319 | 5.98 | — | cute_t256_r2_p1_s0_te |
| GB10 | 2048 | 64 | cold | 7.67 | 7.73 | 1.008 | 7.14 | 0.931 | 8.03 | — | cute_t256_r2_p1_s0_te |
| GB10 | 2048 | 256 | hot | 5.97 | 4.62 | 0.773 | 28.51 | 4.776 | 5.23 | — | cute_t256_r1_p0_s0_tn |
| GB10 | 2048 | 256 | engine | 19.18 | 28.39 | 1.480 | 28.51 | 1.486 | 19.33 | — | cute_t256_r1_p0_s0_tn |
| GB10 | 2048 | 256 | cold | 28.51 | 28.52 | 1.000 | 28.51 | 1.000 | 28.63 | — | cute_t256_r1_p0_s0_tn |
| RTX 5090 | 4096 | 64 | hot | 1.61 | 1.51 | 0.937 | 2.00 | 1.244 (latency-bound) | 1.51 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 4096 | 64 | engine | 2.52 | 3.00 | 1.192 | 2.00 | 0.796 (latency-bound) | 2.57 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 4096 | 64 | cold | 3.00 | 3.01 | 1.001 | 2.00 | 0.666 (latency-bound) | 3.03 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 4096 | 256 | hot | 3.44 | 3.18 | 0.926 | 7.99 | 2.323 | 3.12 | — | cute_t256_r1_p0_s0_tn |
| RTX 5090 | 4096 | 256 | engine | 5.14 | 7.84 | 1.525 | 7.99 | 1.554 | 5.94 | — | cute_t256_r1_p0_s0_tn |
| RTX 5090 | 4096 | 256 | cold | 7.60 | 7.85 | 1.032 | 7.99 | 1.052 | 7.79 | — | cute_t256_r1_p0_s0_tn |
| RTX 5090 | 2048 | 64 | hot | 1.16 | 1.13 | 0.972 | 1.00 | 0.862 (latency-bound) | 1.65 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 2048 | 64 | engine | 1.75 | 2.29 | 1.312 | 1.00 | 0.574 (latency-bound) | 2.24 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 2048 | 64 | cold | 2.02 | 2.25 | 1.112 | 1.00 | 0.495 (latency-bound) | 2.61 | — | cute_t256_r1_p1_s0_te |
| RTX 5090 | 2048 | 256 | hot | 2.07 | 2.05 | 0.987 | 4.00 | 1.928 (latency-bound) | 2.33 | — | cute_t512_r1_p1_s0_te |
| RTX 5090 | 2048 | 256 | engine | 3.25 | 4.22 | 1.296 | 4.00 | 1.229 (latency-bound) | 3.73 | — | cute_t512_r1_p1_s0_te |
| RTX 5090 | 2048 | 256 | cold | 3.97 | 4.25 | 1.070 | 4.00 | 1.008 (latency-bound) | 4.35 | — | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 4096 | 256 | hot | 3.57 | 2.77 | 0.777 | 8.47 | 2.371 | 3.58 | 4.72 (c4) | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 4096 | 256 | engine | 7.03 | 9.51 | 1.351 | 8.47 | 1.204 | 7.89 | 7.56 (c2) | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 4096 | 256 | cold | 9.79 | 9.58 | 0.979 | 8.47 | 0.865 | 10.03 | 10.25 (c2) | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 2048 | 256 | hot | 2.44 | 1.70 | 0.698 | 4.23 | 1.732 | 2.37 | 3.18 (c2) | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 2048 | 256 | engine | 4.24 | 5.51 | 1.299 | 4.23 | 0.999 | 4.84 | 4.97 (c2) | cute_t512_r1_p1_s0_te |
| RTX PRO 6000 | 2048 | 256 | cold | 5.53 | 5.55 | 1.003 | 4.23 | 0.765 | 5.79 | 6.03 (c2) | cute_t512_r1_p1_s0_te |

In the cold regime, a pure streaming copy of the same bytes takes as long as
ours on the three remaining rows: ours vs witness is 1.008 (GB10 H2048 M64),
1.003 (RTX PRO 6000 H2048 M256), and 0.979 (RTX PRO 6000 H4096 M256).
The deficit against bytes/bandwidth is a per-kernel cost—DRAM latency and
launch/drain of a one-wave chain—that every implementation pays, not SM imbalance.
Hot rows are L2-resident and not judged by (b), so their witness ratio is not
meaningful for this bound.

The three open cold rows below use mixed-bandwidth SOL from the main receipt
and size-matched copy ratios from Round 3. Under the accepted (b) bound (the
slower of bytes / mixed bandwidth and the size-matched witness) all three pass. The row split stays built (`Config.split`, opt-in)
and is not dispatched.

| Device | H | M | SOL vs mixed bandwidth | ours vs size-matched copy witness | (b) vs accepted bound | status |
| --- | --- | --- | --- | --- | --- | --- |
| GB10 | 2048 | 64 | 0.933 | 1.008 | 1.008 | pass |
| RTX PRO 6000 | 2048 | 256 | 0.761 | 1.003 | 1.003 | pass |
| RTX PRO 6000 | 4096 | 256 | 0.872 | 0.979 | 0.979 | pass |

## Support range vs SGLang 72730d6

N1/N2/N5 source audit and implementation, 2026-10-16. `S/` below means
`python/sglang/` at **72730d6** (read with `git show`, not the moving checkout).
`F/` means the installed FlashInfer **0.7.0.post1** package in
`../envs/engine-kfi07/lib/python3.12/site-packages/flashinfer/`.
`O/` means this repository's `python/ctikernels/`.
`D` is the output width for gated activations, the input width for ReLU2;
`M` is the product of prefix dimensions. `V16 = 16 / sizeof(dtype)`; gated
AOT kernels cover multiples of V16, while the Blackwell JIT requires 2*V16.
Layout copies are allowed; none of
these new paths performs activation or normalization arithmetic through torch.

| Op | SGLang range: dtype, rank, width, kind, zero rows | Ours before | Ours now | Gap | Status / ETA |
| --- | --- | --- | --- | --- | --- |
| `silu_and_mul` | FP16/BF16/FP32; rank >= 1 flattened to contiguous rows; SiLU; `D % V16 == 0` across JIT/AOT routes; M=0 JIT returns without launch [S1,S2] | FP16/BF16 native; FP32 reached FlashInfer's FP16-only dispatch [F1]. Existing tuned CuTe admission required input width divisible by 16 and 16-byte pointers. | FP16/BF16 nonempty dispatch/config/math unchanged. FP32 uses CuTe for any D, including D=4/12/13 and unaligned views; all zero-element calls bypass provider resolution [O1,O2]. | N1 | Closed: GPU tests pass on RTX 5090, PRO 6000, GB10 (exhaustive, FlashInfer 0.7.0.post1); FP32 speed vs SGLang's kernel not yet measured (ETA 10-18) |
| `gelu_and_mul` | FP16/BF16/FP32; rank >= 1; `D % V16 == 0` across JIT/AOT routes; erf GELU; M=0 JIT accepted [S1,S2] | FlashInfer only; FP32 rejected, M=0 could launch an invalid grid [F1] | FP16/BF16 FlashInfer; FP32 accurate-erf CuTe with any D; M=0 no launch [O2,O3] | N1 | Closed: GPU tests pass on RTX 5090, PRO 6000, GB10 (exhaustive, FlashInfer 0.7.0.post1); FP32 speed vs SGLang's kernel not yet measured (ETA 10-18) |
| `gelu_tanh_and_mul` | FP16/BF16/FP32; rank >= 1; `D % V16 == 0` across JIT/AOT routes; 0.044715 cubic, FP32 activation and multiplication then one output rounding; M=0 JIT accepted [S1,S2] | No inference entry | Public `(x, out=None)`; FP16/BF16 FlashInfer kernel, FP32 CuTe, any FP32 D; M=0 no launch [O2,O3] | N2 | Closed: FP16/BF16 runs FlashInfer's kernel; GPU tests pass on three devices; FP32 speed vs SGLang not yet measured (ETA 10-18) |
| `relu2` | FP16/BF16/FP32; rank >= 1; **M*D**, not D, must be divisible by V; strict positive comparison, FP32 square then one output rounding; M=0 accepted [S1,S3] | No inference entry | Public unary `(x, out=None)`; same shape; CuTe all three dtypes, any D/M, scalar tails and unaligned views; M=0 no launch [O2,O3] | N2 | Closed: GPU tests pass on three devices; speed vs SGLang's relu2 not yet measured (ETA 10-18) |
| Ordinary `rms_norm` / `rms_norm_` | FP16/BF16 on the FlashInfer route; rank >= 1 via contiguous flatten/restore; positive H with provider vector selection; no zero-centered weight offset; layer handles empty input [S4,F2] | Forwarded every rank to a rank-2/3-only FlashInfer ABI [F2] | Every rank returns empty output (or x for mutation) without launch. Vector-aligned rank-2/3 inputs with unit hidden stride keep their direct provider ABI, including QK-norm; other ranks/layouts flatten through a compatible view or packed copy, restore shape and copy back for mutation. | N5 | Closed: rank-2/3 empty and hidden-stride-2 regression tests pass on RTX 5090, RTX PRO 6000 and GB10 (FlashInfer 0.7.0.post1) |

Source evidence:

- **S1:** `S/kernels/ops/activation/activation.py:21,57,65,158,188` selects
  dtype-specialized JIT code and flattens arbitrary prefixes. The AOT dispatch
  explicitly includes FP32 at `S/kernels/aot/csrc/elementwise/activation.cu:93,115,137`
  via the float case in `S/kernels/aot/include/utils.h:306-314`. The layer routes
  widths outside JIT admission to AOT at `S/srt/layers/activation.py:77-95`;
  its ReLU2 CUDA entry calls JIT directly at `:290-291`.
- **S2:** `S/kernels/jit/csrc/elementwise/activation.cuh:25-39` defines all four
  formulas; `:104-117` shows the ordinary path has **no intermediate dtype
  rounding**; `:204-219` matches contiguous rows and checks gated width;
  `:217` returns early for zero rows. `V = kMaxVecBytes / sizeof(dtype)`;
  `S/kernels/jit/include/sgl_kernel/utils.cuh:140` sets 32 bytes on Blackwell
  and 16 bytes otherwise. Thus FP32 JIT widths are multiples of 8/4 and
  FP16/BF16 widths multiples of 16/8, respectively. The AOT path uses 16-byte
  vectors (`activation.cu:94-101`); a non-divisible width is not a reliable
  accepted tail path (the vector loop cannot produce every element).
  Vector loads imply corresponding natural alignment. The new paths are
  broader: element-aligned pointers and arbitrary widths are safe.
- **S3:** `S/kernels/jit/csrc/elementwise/activation.cuh:346-365` checks total
  elements and returns early for empty ReLU2. For example M=64,D=13 is accepted
  for every dtype on both vector architectures; odd gated D is not JIT-admitted.
- **S4:** `S/srt/layers/layernorm.py:102-107,520-525,562-568,624-631` establishes
  ordinary provider selection, empty handling, contiguous flatten and reshape
  back. `S/kernels/aot/python/sgl_kernel/elementwise.py:13,115-122` routes
  FP16/BF16 to FlashInfer. This N5 change concerns that route's rank adaptation;
  it does not claim coverage of all alternative SGLang norm providers/dtypes.
- **F1:** `F/jit/activation.py:43` uses FP16/BF16-only dispatch;
  `F/activation.py:103-119` checks input byte width then launches without an
  empty guard. `F/jit/activation.py:93-105` defines erf/tanh GELU; its tanh
  expression has the same cubic and output rounding convention as SGLang.
- **F2:** `F/data/csrc/norm.cu:25-78` admits only ranks 2 and 3, unit hidden
  stride, FP16/BF16. `F/data/include/flashinfer/norm.cuh:140-169` selects vector
  width with gcd(16/itemsize,H), rather than imposing H divisible by 8.
- **O1:** `O/activation/inference.py:83-91` handles empty and FP32 SiLU before
  the unchanged low-precision dispatcher.
- **O2:** `O/activation/inference.py:273-349` defines tanh GELU/ReLU2, empty
  handling, dtype routing, layout copies and native launch.
- **O3:** `O/activation/cute/_kernel.py:117-134,170-211` defines compile-time
  activation kinds, accurate erf/tanh and 128-bit or scalar accesses.
  `O/activation/cute/_provider.py:81-126` caches dtype/kind/width/alignment/device,
  uses dynamic int64 M, disables contraction in the new path, and requires
  warmup before capture. Existing SiLU configs/compilation flags are unchanged.
- **O4:** `O/norm/inference.py` ordinary standalone entry points and
  `_ordinary_rms_direct` / `_ordinary_rms_rows` keep compatible rank-2/3 calls
  direct and adapt every other rank/layout. Non-collapsible prefixes, strided
  hidden axes and vector-misaligned pointers or row pitches copy; in-place
  results copy back. Empty standalone, residual and fused add inputs return
  without a provider launch.

Validation added in `tests/modules/test_inference_activation.py`,
`tests/norm/test_api.py`, `tests/gpu/test_activation.py`,
`tests/norm/test_accuracy.py`, `tests/norm/test_training.py` and
`tests/norm/test_contract.py`: public signatures/exports; provider routing and empty
no-launch behavior; FP64 accuracy gates (norms now use the shared PyTorch
precision baseline); FP16/BF16 FlashInfer tanh GELU identity; ReLU2 rounding;
unaligned/strided buffers; graph replay; and standalone norm rank/layout
bitwise equality to the rank-2 call. Large cross-products are marked exhaustive.
Offline CuTe DSL 4.8.0 compilation passed 56 cases across SM80/120/121, including
vector/tail/unaligned branches and representative original FP16/BF16 kernels.
Eight representative original FP16/BF16 SiLU configurations also produced
identical PTX instruction streams against HEAD (normalizing only module-qualified
symbol names), including precise/fast math, early PDL, multiple vectors and e16.
The GPU test modules collect successfully in the installed GPU dependency
environment (including existing tests and exhaustive cases).
GPU numerical results and performance are **not measured on this host**.

CPU validation: `PYTHONPATH=python:. python -m pytest -q
tests/modules tests/benchmarks tests/quality -x` completed with **12,737 passed,
721 skipped**, plus 13 warnings from existing FP8 interpreter/autotune tests.
Focused activation/norm/version/cache checks covering the final row-alignment
fix passed **289 tests**. `pre-commit run --all-files` passed.

Empty inputs: every rank, including rank-2/3, returns without a provider launch
(the earlier rank-2/3 FlashInfer delegation of empty inputs is fixed).


## NA-NRANGE: correctness admission

Inference RMSNorm now admits FP16/BF16/FP32, both weight modes, FP32 weights,
all output dtypes, every H >= 1, empty M and int64 element offsets. Standalone
rank-2/3 unit-hidden-stride views (including merged QKV heads) run natively;
higher-rank prefixes may collapse to two strides. Add+norm and residual-only
norm accept arbitrary row strides without alignment restrictions. Residual dtype,
normalization and store/no-store/sum_out modes are independent. Unsupported native
layouts use a visible `pack` reroute and copy modified buffers back.

The ordered predicate in operations.md is shared
by eager and graph resolution. On capability 12.0/12.1, admitted off-table widths
use CuTe `Config()` (1024 threads if the default's staging bound is insufficient).
Other inputs take generic Triton. Existing tables and measured rows are unchanged;
this adds no tuning results. Missing DSL, recorded compilation failure, unusable
FlashInfer and rejected table configurations report reroute reasons. Warm-up also
compiles and pins off-table configurations. Training layout restrictions remain.

`tests/norm/test_accuracy.py` pairs the full H/M range with semantic variants,
checks a GPU FP64 oracle against the plain PyTorch implementation at the same precision,
checks stored sums and claimed cross-entry equalities bitwise, and includes
memory-guarded unmarked >2^31-element tests. CPU tests validate selection,
packing, overrides and cache behavior only. Historical qualification before the
shared accuracy gate migration covered RTX 5090, RTX PRO 6000 and GB10 with the
tests now in `test_accuracy.py`, `test_training.py` and `test_contract.py`: the
full suite including the exhaustive grids under FlashInfer 0.7.0.post1 (2817 passed; GB10 2816 and one
>2^31 case skipped by its memory guard), and the core subset with FlashInfer blocked.

#### NA-NRANGE speed report (M=1-4, report only)

Paired add+norm (`norm_act_paired.py`, 20 rounds, default `static_weight=False`, FlashInfer 0.7.0.post1 in-process; SGLang column = its FlashInfer 0.7.1 kernel in the market worker). Each cell is ours µs / ratio vs the faster market (hot | cold). vLLM's Inductor capture exists only for H=4096; H=5120 compares with SGLang only.

| Device | Weight | H | Provider | M=1 | M=2 | M=3 | M=4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | standard | 4096 | `cute_t512_r1_p1_s0_te` | 1.122/1.110 \| 1.309/1.149 | 1.195/1.281 \| 1.413/1.378 | 1.206/1.253 \| 1.413/1.380 | 1.209/1.262 \| 1.424/1.376 |
| RTX 5090 | standard | 5120 | `cute_t512_r1_p0_s0_tn` | 1.534/1.001 \| 1.749/0.972 | 1.628/1.119 \| 1.846/1.264 | 1.630/1.119 \| 1.857/1.247 | 1.650/1.275 \| 1.868/1.345 |
| RTX 5090 | 1+w | 4096 | `cute_t512_r1_p1_s0_te` | 1.129/1.118 \| 1.337/1.142 | 1.220/1.293 \| 1.414/1.468 | 1.227/1.263 \| 1.424/1.418 | 1.223/1.379 \| 1.449/1.476 |
| RTX 5090 | 1+w | 5120 | `cute_t512_r1_p0_s0_tn` | 1.536/1.001 \| 1.750/0.970 | 1.641/1.109 \| 1.847/1.262 | 1.650/1.104 \| 1.859/1.243 | 1.649/1.282 \| 1.868/1.342 |
| RTX PRO 6000 | standard | 4096 | `cute_t512_r1_p1_s0_te` | 1.389/1.116 \| 1.596/1.148 | 1.472/1.261 \| 1.702/1.392 | 1.498/1.263 \| 1.720/1.368 | 1.498/1.267 \| 1.742/1.365 |
| RTX PRO 6000 | standard | 5120 | `cute_t512_r1_p0_s0_tn` | 1.877/1.000 \| 2.092/0.975 | 1.991/1.133 \| 2.237/1.221 | 1.991/1.115 \| 2.239/1.217 | 1.991/1.314 \| 2.265/1.328 |
| RTX PRO 6000 | 1+w | 4096 | `cute_t512_r1_p1_s0_te` | 1.377/1.107 \| 1.610/1.135 | 1.486/1.260 \| 1.721/1.381 | 1.493/1.294 \| 1.732/1.364 | 1.528/1.360 \| 1.752/1.361 |
| RTX PRO 6000 | 1+w | 5120 | `cute_t512_r1_p0_s0_tn` | 1.868/1.001 \| 2.101/0.973 | 2.007/1.132 \| 2.243/1.220 | 1.998/1.114 \| 2.243/1.217 | 1.991/1.314 \| 2.267/1.328 |
| GB10 | standard | 4096 | `cute_t512_r1_p1_s0_te` | 1.302/1.153 \| 1.549/1.150 | 1.383/1.241 \| 1.652/1.433 | 1.404/1.227 \| 1.713/1.351 | 1.411/1.230 \| 1.845/1.305 |
| GB10 | standard | 5120 | `cute_t512_r1_p0_s0_tn` | 1.722/1.031 \| 2.009/0.993 | 1.877/1.143 \| 2.148/1.264 | 1.878/1.182 \| 2.267/1.209 | 1.872/1.245 \| 2.453/1.175 |
| GB10 | 1+w | 4096 | `cute_t512_r1_p1_s0_te` | 1.303/1.145 \| 1.554/1.145 | 1.396/1.226 \| 1.664/1.388 | 1.402/1.237 \| 1.716/1.348 | 1.391/1.238 \| 1.825/1.320 |
| GB10 | 1+w | 5120 | `cute_t512_r1_p0_s0_tn` | 1.733/1.017 \| 2.009/0.992 | 1.877/1.127 \| 2.157/1.263 | 1.878/1.176 \| 2.274/1.204 | 1.872/1.251 \| 2.451/1.171 |

Below 1.0: H=5120 M=1 cold on every device (0.970-0.993), the generic `Config()` without PDL. Reported, not chased (scope rule).

Per-head Q/K norm on merged-qkv views (`rms_norm_` on q then k; one CUDA graph of 50 calls, median of 20 replays, µs per q+k pair). SGLang's JIT `fused_inplace_qknorm` is not installable in our env and is not measured; the FlashInfer column is SGLang's non-JIT path.

| Device | Shape | T | Provider | ours | FlashInfer | Inductor stand-in |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | qwen3-8b D128 std | 1 | `flashinfer` | 1.56 | 1.55 | 1.64 |
| RTX 5090 | qwen3-8b D128 std | 2 | `flashinfer` | 1.80 | 1.80 | 1.93 |
| RTX 5090 | qwen3-8b D128 std | 3 | `flashinfer` | 1.80 | 1.80 | 2.79 |
| RTX 5090 | qwen3-8b D128 std | 4 | `flashinfer` | 1.81 | 1.80 | 2.36 |
| RTX 5090 | qwen3.5-9b D256 zc | 1 | `triton` | 2.05 | 1.76 | n/a |
| RTX 5090 | qwen3.5-9b D256 zc | 2 | `triton` | 2.08 | 1.77 | n/a |
| RTX 5090 | qwen3.5-9b D256 zc | 3 | `triton` | 2.09 | 1.77 | n/a |
| RTX 5090 | qwen3.5-9b D256 zc | 4 | `triton` | 2.12 | 1.76 | n/a |
| RTX PRO 6000 | qwen3-8b D128 std | 1 | `flashinfer` | 1.83 | 1.83 | 2.01 |
| RTX PRO 6000 | qwen3-8b D128 std | 2 | `flashinfer` | 1.88 | 1.89 | 2.30 |
| RTX PRO 6000 | qwen3-8b D128 std | 3 | `flashinfer` | 1.89 | 1.89 | 2.29 |
| RTX PRO 6000 | qwen3-8b D128 std | 4 | `flashinfer` | 1.91 | 1.92 | 2.30 |
| RTX PRO 6000 | qwen3.5-9b D256 zc | 1 | `triton` | 2.45 | 1.81 | n/a |
| RTX PRO 6000 | qwen3.5-9b D256 zc | 2 | `triton` | 2.49 | 2.14 | n/a |
| RTX PRO 6000 | qwen3.5-9b D256 zc | 3 | `triton` | 2.53 | 2.14 | n/a |
| RTX PRO 6000 | qwen3.5-9b D256 zc | 4 | `triton` | 2.55 | 2.14 | n/a |
| GB10 | qwen3-8b D128 std | 1 | `flashinfer` | 1.74 | 1.74 | 1.79 |
| GB10 | qwen3-8b D128 std | 2 | `flashinfer` | 1.74 | 1.74 | 1.80 |
| GB10 | qwen3-8b D128 std | 3 | `flashinfer` | 1.75 | 1.75 | 1.98 |
| GB10 | qwen3-8b D128 std | 4 | `flashinfer` | 1.74 | 1.74 | 2.04 |
| GB10 | qwen3.5-9b D256 zc | 1 | `triton` | 2.16 | 1.71 | n/a |
| GB10 | qwen3.5-9b D256 zc | 2 | `triton` | 2.17 | 1.71 | n/a |
| GB10 | qwen3.5-9b D256 zc | 3 | `triton` | 2.17 | 1.70 | n/a |
| GB10 | qwen3.5-9b D256 zc | 4 | `triton` | 2.21 | 1.92 | n/a |

D128 standard runs FlashInfer itself (ties). D256 zero-centered runs our Triton kernel at 0.78-0.86 of FlashInfer `gemma_rmsnorm`; reported, not chased. The torch.compile stand-in for D256 (about 23 µs) is not representative of vLLM's fused capture and is not reported.
