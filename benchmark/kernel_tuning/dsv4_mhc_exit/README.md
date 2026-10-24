# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

# EA5 phase 1: DeepSeek-V4-Flash mHC exit

Phase 1 measures market implementations; phase 2 also measures the CuTe package
provider described below. The target is checkpoint `60d8d70770c6776ff598c94bb586a859a38244f1`, H=4096, four residual
streams, 20 Sinkhorn iterations, and all epsilons 1e-6. There are 43 layers and
two exits per layer, hence 86 exits per step. Each FP32 fn matrix is
24 × 16384 × 4 bytes = 1.5 MiB; the 86 matrices occupy 129 MiB.
Checkpoint `config.json` supplies these dimensions and epsilons.

## Boundary and references

With two tensor/expert ranks, first sum the BF16 partials and round to BF16:
`x = bf16(x0.float() + x1.float())`. For each token and destination stream j:

```text
Rnew[j,h] = post[j] * x[h] + sum_i comb[i,j] * R[i,h]
flat      = flatten(Rnew)
mix       = (flat @ fn.T) / sqrt(mean(flat²) + rms_eps)
pre       = sigmoid(mix[:4] * scale[0] + base[:4]) + hc_eps
post_new  = 2 * sigmoid(mix[4:8] * scale[1] + base[4:8])
C         = softmax(reshape(mix[8:] * scale[2] + base[8:], 4, 4), axis=-1) + hc_eps
C         = C / (column_sums(C) + hc_eps)
repeat 19 times: C /= row_sums(C) + hc_eps; C /= column_sums(C) + hc_eps
y[h]      = sum_j pre[j] * Rnew[j,h]
out[h]    = y[h] / sqrt(mean(y²) + norm_eps) * norm_weight[h]
```

The independent FP64 oracle starts from the stored BF16/FP32 inputs and weights,
without intermediate rounding. The torch composition rounds Rnew and y to BF16,
uses FP32 arithmetic, and rounds out to BF16. These boundaries follow the official
`inference/model.py:674-700`; Sinkhorn order follows `inference/kernel.py:398-424`.
The oracle starts after the collective's BF16 rounding; pair.py separately checks
the collective against the exact BF16 sum of the two partials.

Synthetic fn entries have standard deviation `1/sqrt(4H)`, scales are one, base
offsets have standard deviation 0.1, and norm weights are BF16 near one. Normalized
projection logits are therefore order one without deliberate sigmoid saturation.
Initial comb uses the same actual Sinkhorn procedure on random normal logits.
These are representative synthetic inputs, not measured checkpoint distributions.

## Source dispatch audit

Paths below are relative to each named source tree. Counts exclude all-reduce,
allocations, and first-use compilation. They are source-derived counts; use
`--count-kernels` to record actual warmed CUDA activities on the target machine.

| Provider, H=4096, T=1–4 | Eager compute launches | Graph compute launches per exit | Final norm |
| --- | ---: | ---: | --- |
| vLLM 0.31.0 TileLang | 2 | 2 | Fused into second kernel |
| SGLang 72730d6 defaults | 5 | 5 | Separate RMSNorm (dispatch below) |
| FlashInfer 0.7.1 composition here | expected 5 | expected 5 | Separate RMSNorm (dispatch below) |
| torch composition | profiler count | profiler count | Separate FP32 torch operations |
| optional torch_compile | compiler dependent | compiler dependent | Compiler dependent |

### vLLM

`vllm/models/deepseek_v4/nvidia/model.py:1366-1418` invokes
`mhc_fused_post_pre_tilelang` with `n_splits=1, tile_n=1` and the next norm's
weight and epsilon. The first layer instead uses standalone/broadcast pre
(`model.py:1330-1365`), outside this steady-state exit. CUDA plus installed
TileLang enables the TileLang implementation, without an sm_100-only gate
(`model_executor/layers/mhc.py:16-31,738-775`). The torch and Triton alternatives
in `model_executor/kernels/mhc/{torch,triton}.py` are not this model's decode path.

The call-site split/tile arguments are compatibility arguments: the wrapper
selects configuration internally (`model_executor/kernels/mhc/tilelang.py:814-918`).
At H=4096, mult=4, T=1–4 the effective configuration is **tile_n=2, n_splits=8,
128 threads**, selected by `tilelang_kernels.py:67-95`. The first kernel fuses
post and projection/squared-sum partials. The second reduces the eight splits,
does Sinkhorn, collapses streams, and applies the next RMSNorm
(`tilelang.py:918-941`; `tilelang_kernels.py:299-502,1282-1287`). It rounds the
collapsed input to BF16 inside the fused norm path (`tilelang_kernels.py:432-450`).

There is a numerical distinction: the first fused kernel projects and squares
its **unrounded FP32** post values while also storing a BF16 residual
(`tilelang_kernels.py:741-755`). Its coefficients can differ from the torch
composition, which re-reads rounded BF16 Rnew. The error floor is reported,
never imposed on market providers.

The fused first kernel covers T<=32 and H divisible by 128; T=16–32 selects
tile_n=6, still eight splits here. Prefill uses separate post, projection/stats,
and big-fuse/norm: three launches (`tilelang.py:942-975`). Projection selects
DeepGEMM when enabled and present, otherwise TileLang (`tilelang.py:22-71`).
sm_121 is admitted by vLLM's capability-family-120 check
(`platforms/cuda.py:709-715`); availability of the particular DeepGEMM build still
matters. `VLLM_USE_DEEP_GEMM=0` selects the upstream TileLang fallback for prefill.
External DeepGEMM takes precedence over the wheel's vendored copy
(`utils/import_utils.py:500-508`, `utils/deep_gemm.py:182-215`).

The adapter imports the real wrapper, kernels, TileLang utilities, and warmup
helpers. Minimal package/platform/registration shims avoid vLLM server imports;
no numerical kernel is replaced. Wrapper-owned allocations remain wrapper-owned.
CUDA graph capture makes those addresses stable. Use this adapter in a standalone
process, since the minimal namespaces are not a usable vLLM server environment.

### SGLang

`sglang/srt/runtime_context.py:1824-1828` maps platform facts to
`srt/utils/common.py:290-307`: sm_121 is Blackwell and sm_120-family, **not sm_100**
(with CUDA >=12.8). Defaults in `srt/environ.py:1609,1617-1621` enable TileLang
pre/post and DeepGEMM pre-normalization, disable FlashInfer mHC, and disable Mega
mHC prefill. At H=4096 the actual chain is:

1. TileLang `mhc_post`: `srt/models/deepseek_v4_mhc.py:481-517` and
   `kernels/ops/layernorm/mhc.py:1311-1383` — one launch.
2. Triton partial projection/squared sum, then fused reduction/Sinkhorn:
   `deepseek_v4_mhc.py:274-338`; `mhc.py:2462-2551` — two launches. K=16384
   selects 64 slices, block_k=64, block_m=8 at decode, tf32x3 dot precision
   (`mhc.py:2300-2336`). The compensated DeepGEMM branches require sm_100 or
   sm_90, so they do not apply on sm_121, including prefill.
3. Triton `hc_combine` — one launch (`mhc.py:2740-2765`), then the normal
   `sgl_kernel.rmsnorm` dispatcher — one launch
   (`srt/models/deepseek_v4_mhc.py:419-445`; `srt/layers/layernorm.py:102-107,631-632`;
   `kernels/aot/python/sgl_kernel/elementwise.py:76-122`).

There is another dispatch inside `sgl_kernel`: with FlashInfer installed and
BF16 input, it forwards to `flashinfer.norm.rmsnorm`; otherwise it uses its AOT
CUDA kernel (`elementwise.py:6-13,16-28,115-122`). FlashInfer 0.7.1 defaults to
CuTe DSL norm when that stack supports the device, otherwise JIT CUDA; setting
`FLASHINFER_USE_CUDA_NORM=1` selects JIT CUDA (`flashinfer/norm/__init__.py:60-101,188-198`).
The adapter records this runtime norm backend. A standalone SGLang JIT norm
exists, but is not selected by this model's ordinary RMSNorm path. There are no
eager torch arithmetic kernels in the default H=4096 chain.

This five-launch chain is the same in eager execution and CUDA graph capture.
`compile_in_capture_mode` compiles a torch helper only when it is defined in
model capture mode (`srt/model_executor/runner_utils/capture_mode.py:48-58`).
That helper is the *fallback* post at `deepseek_v4_mhc.py:528-539`; default
TileLang post returns before it. Capturing a raw CUDA graph does not itself set
the model capture flag. The adapter fixes the default chain and does not emulate
nondefault `SGLANG_OPT_*` overrides.

The split-H post, fused combine/norm, and `can_fuse_post` policies require H=5120
(`deepseek_v4_mhc.py:419-445,481-490,543-558`). Split-H and combine/norm are Triton
(`kernels/ops/layernorm/mhc_post_split_h.py:8-38`, `hc_combine_norm.py:10-80`).
Mega mHC is opt-in sm_100 and uses shifted pre coefficients for V4.1
(`deepseek_v4_mhc.py:581-587`; `mhc_mega.py:30-35`). The single-node fused AR+mHC
implementation is **JIT CUDA**, fixed at H=5120 and dependent on registered
cluster communication (`kernels/ops/communication/all_reduce_mhc.py:7-46`). None
of these specialized JIT CUDA/mega paths replaces the H=4096 default chain.
The generic residual boundary merely carries coefficients and invokes callbacks
(`srt/layers/layer_boundary/residual/mhc.py:27-85`).

The adapter loads real kernel-level functions and the installed `sgl_kernel`
RMSNorm. It bypasses model/server imports with explicit no-CP, nonsymmetric,
CUDA runtime policy shims. It preallocates the RMSNorm output; other public
wrappers allocate their own outputs. No server or distributed SGLang runtime
is initialized. The dispatch also applies to RTX PRO 6000 and RTX 5090 at this
shape; this is a source observation, not a performance claim.

### FlashInfer

The supplied local distribution identifies as **0.7.0.post1**, not 0.7.1.
The public v0.7.1 tag was checked separately: `flashinfer/mhc.py`,
`flashinfer/jit/mhc.py`, `csrc/mhc/mhc_post.cu`, and
`csrc/mhc/mhc_pre_big_fuse.cu` are byte-identical to that local copy.
The architecture-selection file differs; the 0.7.1 rule below is from the tag.

`mhc_post` is one CUDA launch; it requires HC=4, BF16 x/R and FP32 coefficients
(`flashinfer/mhc.py:41-132`). `mhc_pre_big_fuse` consumes external dot products
and squared sums; `mhc_pre_big_fuse_with_prenorm` consumes external dot products
but computes squared sums itself (`mhc.py:176-332,334-451`). Neither computes the
fn projection or the *next sublayer's* RMSNorm. Both return BF16 layer input.
The adapter uses post, BF16-to-FP32 cast, torch/cuBLAS FP32 GEMM, pre-with-prenorm,
and FlashInfer RMSNorm (CuTe DSL or JIT CUDA): expected five compute launches. cuBLAS implementation
details can change that count, hence profiling support. The post multiplier is
explicitly set to **2**; the FlashInfer API default is 1.

The JIT module compiles ordinary CUDA sources and has no sm_100-only restriction
([`flashinfer/jit/mhc.py:21-28`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/flashinfer/jit/mhc.py#L21)).
The v0.7.1 target selector preserves sm_121 and chooses `compute_121a`, requiring
CUDA toolkit >=12.9
([`flashinfer/compilation_context.py:80-107`](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/flashinfer/compilation_context.py#L80)).
Thus sm_121 is supported in source; successful JIT/build/capture remains a runtime
check. H=4096 is explicitly specialized by post (`csrc/mhc/mhc_post.cu:524-526`).
Pre requires H divisible by 8 and HC=4 (`mhc_pre_big_fuse.cu:365-406`); the external
sum variant admits split counts 1,2,4,8,16 (`mhc_pre_big_fuse.cu:429-431`).
The adapter requires installed FlashInfer 0.7.1 rather than silently labeling an
older installation as the requested version. GEMM and norm outputs are preallocated.
The audited v0.7.1 `flashinfer/norm/__init__.py` also matches the local copy.

## Measurement protocol

`Provider.prepare(cfg, weights, T, device)` returns a callable taking
`(x, R, post, comb)` and returning `(Rnew, post_new, comb_new, out)`, or None
with `status="unavailable: ..."`. GPU providers return unavailable on CPU without
probing CUDA. Optional import/compile/capture failures are reported by stage and
exception type, without echoing environment values or local paths.

Decode rows are T=1,2,3,4. Prefill rows 128,1024,8192 are disclosed and ungated;
no phase-1 provider is rejected for exceeding the torch error floor. Every row
compares all exposed outputs to the FP64 oracle, reporting max-abs and RMS errors
and the strict `provider <= torch_bf16` result. Providers do not expose pre or
unnormalized y; the JSON names those omissions rather than reconstructing and
pretending to test hidden kernel intermediates. CPU oracle work is chunked to
bound memory; the torch floor runs on the provider's device with the full batch
so its GEMM reduction order matches the torch provider. TF32 is disabled for torch GEMMs.

The default rotation has at least 16 distinct weight sets **and distinct residual
buffers**. Using the device's reported L2 size, the harness increases this count
until fn+R alone exceeds twice L2. It refuses a cold run if the L2 size is unknown
or `--calls` cannot visit all sets; increase calls in the latter case. `--hot`
uses one set. This models reuse distance, not cache locking or guaranteed eviction.
Prefill buffers can consume several GiB, especially with graph-owned temporaries.

Each mode warms first. Graph mode captures 86 consecutive rotating exits by
default on a non-default stream; eager launches the same sequence. CUDA event
time is divided by calls. Pilot runs calibrate about 3 ms per sample. Thirty
paired samples shuffle provider order with a fixed seed; ratios are per-pair
provider time / fastest provider time, with a percentile bootstrap 95% CI of the
median ratio. The fastest is selected by sample median, including torch baselines;
the CI is conditional on that selection, not corrected for winner selection.
Kernel profiling is outside timing and separates compute from all CUDA activities.

pair.py imports Session/barriers, discovery/pinning helpers, and slower-rank
timing from `two_node_allreduce/bench.py`, and initializes PyNccl identically.
It reports AR alone, mHC alone, and B1/NCCL plus each provider, with ratios within
each group. Each rank uses the same seeds/order; samples retain the slower rank.
Random partials test the exact BF16 reduction and whole-exit error. Timing uses
zero partials with random R so repeated in-place NCCL sums stay stable, matching
the existing all-reduce benchmark and avoiding reset-copy costs. Both eager and
graph outputs are checked by exchanging SHA256 hashes of every output's exact
bytes. The graphs include 86 full exits; JSON includes both per-exit and per-step
time. Slow captured NCCL runs are still measured. B1 fallback is rejected so a
row labeled B1 cannot silently measure another collective.

## Running

Run from the repository root with a CUDA-enabled environment and the matching
source trees. Examples use generic paths; select one visible GPU per process.

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --vllm-root artifacts/vllm-0.31.0/vllm \
  --sglang-python vendor/sglang-72730d6/python \
  --count-kernels --out mhc-exit.json

PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers torch torch_compile --rows 1 2 3 4 --hot --out mhc-hot.json

# Start on each node, using its rank and the same rank-0 address/port.
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python -m benchmark.kernel_tuning.dsv4_mhc_exit.pair \
  --rank 0 --store <rank-0-address>:29500 \
  --vllm-root artifacts/vllm-0.31.0/vllm \
  --sglang-python vendor/sglang-72730d6/python --out mhc-pair.json
# On the second node use --rank 1 and the same other options.

PYTHONPATH=python python -m pytest -q tests/benchmarks/test_dsv4_mhc_exit_reference.py
```

Runtime requirements: torch needs CUDA-enabled PyTorch; torch_compile also needs
its compatible compiler stack. vLLM needs the extracted 0.31.0 package tree,
TileLang (the audited setup uses 0.1.12), its CUDA compiler dependencies, and
optionally DeepGEMM for prefill. SGLang needs the 72730d6 Python tree, compatible
Triton/TileLang and `sgl_kernel` wheel. FlashInfer needs `flashinfer-python==0.7.1`,
its JIT dependencies, and a toolkit supporting the device target. pair.py also
needs the B1 RoCE prerequisites and the same NCCL setup as the existing two-node
benchmark; `NCCL_MAX_NCHANNELS` defaults to 2 unless explicitly configured.
No framework server is needed. GPU latency, actual kernel counts, graph capture,
and two-node execution have not been validated by the CPU-only implementation run.
Tests use the repository pytest configuration directly with `PYTHONPATH=python`.

## Results (phase 1 headroom, GB10, 2026-10-16)

All rows: H=4096, cold fn (rotating weight sets past 2x L2), median of 20 paired
samples, per exit; "per step" is 86 exits. Full table with errors and CIs:
`benchmark/kernel_tuning/results/dsv4_mhc_exit_headroom.csv`.

Single GB10 (bench.py, X925-pinned), CUDA graph, microseconds per exit:

| T | torch | vLLM TileLang | SGLang | FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 260 | **13.7** | 21.1 | 21.6 |
| 2 | 360 | **14.1** | 23.2 | 117.6 |
| 3 | 393 | **14.0** | 23.1 | 151.1 |
| 4 | 314 | **13.6** | 21.1 | 120.1 |
| 128 (prefill) | 695 | n/a | **79** | 125 |
| 1024 (prefill) | 6582 | n/a | **745** | 1565 |
| 8192 (prefill) | 54211 | n/a | **6887** | 13932 |

Eager launches cost 70-80 us per exit for every framework chain (host-bound);
the engine runs decode in a graph. vLLM prefill rows need its DeepGEMM
projection, which the harness could not import here (reported as unavailable).

Cabled GB10 pair (pair.py, tp=2), CUDA graph, microseconds per exit
(per-step milliseconds in parentheses):

| T | B1 AR | NCCL AR | vLLM mHC | B1 + vLLM | B1 + SGLang | NCCL + vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.6 | 50.2 | 14.4 | **25.4 (2.19)** | 32.4 | 48.2 (4.14) |
| 2 | 11.5 | 51.9 | 14.4 | **27.7 (2.38)** | 36.3 | 47.0 (4.04) |
| 3 | 12.8 | 50.2 | 14.5 | **28.7 (2.47)** | 36.6 | 48.7 (4.19) |
| 4 | 14.1 | 52.2 | 14.6 | **31.2 (2.68)** | 39.8 | 54.3 (4.67) |

Eager NCCL is 17.4-21.2 us per all-reduce, but the eager framework mHC chains
add 70-80 us, so the best market exit in either mode is graph NCCL + vLLM
(47-54 us). Output error: SGLang and FlashInfer match the torch BF16 composition; vLLM is
up to 1.34x its max error and 1.02x its RMS error on some rows (it projects
unrounded FP32 residuals and rounds y before the norm). All pair outputs
matched bitwise across ranks.

## Phase 2 status (GB10, 2026-10-17)

`ctikernels.mhc.MhcExit` is correct but not yet faster than the market for decode.
All 427 GPU tests pass (measured floor vs the FP64 oracle in exit/pre/post modes,
H = 1024/4096/4104, T = 1..1000, graph replay, PDL chains). Paired, CUDA graph,
86 cold exits, microseconds per exit:

| T | ours (grid) | ours (cluster) | vLLM | SGLang |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.5 | 20.8 | **14.0-14.1** | 18.8 |
| 2 | 18.3 | 19.8 | **14.6-14.7** | 24-25 |
| 4 | 22.5 | 26.5 | **14.7-14.8** | 23.4-23.7 |
| 8 | n/a | 45.5 | **17.7** | 23.6 |
| 128 | 278-575 | | n/a | **79-82** |
| 8192 | **6130-6147** | | n/a | 6993-7193 |

With a 4 MiB non-PDL spacer between exits (no overlap with a previous exit),
net of the spacer: vLLM 16.1-16.8, ours (grid) 21.3-29.7 at T = 1-4.
IKET (grid kernel, T = 1, per CTA after the PDL wait): residual and partials
1.4 us, exchange arrival/skew 3.9-4.3, partial read 3.5, coefficients 2.2-2.5,
output 0.2; fn staging before the wait 9.2. Approximate division saves 0.5 us;
128-column slices and a late PDL trigger are slower. The cluster-of-8 variant
(L2 prefetch, DSMEM exchange) is slower still and stays an A/B option
(`CTI_MHC_DECODE=cluster`).

## Plan (phases 2 and 3)

The sum of the separate pieces is the bar: B1 + vLLM costs 25-31 us per exit.
Two costs dominate the mHC part: reading the cold 1.5 MiB FP32 fn per exit
(about 6 us of DRAM time on GB10) and the launches and grid-wide reductions of a
multi-kernel chain.

1. Single-GPU mHC exit in CuTe DSL (decode): one launch, the H dimension split
   across CTAs. Each CTA loads its fn slice (24 x 4 x H/G FP32) into shared
   memory before the programmatic-dependent-launch wait, so the fn read overlaps
   the previous kernel. After the wait it forms its slice of R' (stored BF16),
   and partial sums of the 24 projections, the squared sum and the 10 Gram
   products of R'. A deterministic grid exchange (fixed-order sum of per-CTA
   partials, monotonic arrival counter, graph-replay safe) lets every CTA
   redundantly run Sinkhorn and compute the RMS of y = sum_j pre_j R'_j from the
   Gram products, so RMSNorm needs no second pass. Prefill uses token tiles with
   whole rows per CTA (no grid exchange). Target: below vLLM's 13.6-14.1 us.
2. Fused two-node exit: the same CTAs run the B1 protocol. While the partial is
   on the wire they load fn, R, post and comb; the sum arrives in rank order
   (bitwise equal to B1), then the step-1 epilogue runs. Target: B1 time plus a
   few microseconds (about 12-17 us per exit, 1.0-1.5 ms per step), against
   B1 + our separate kernel and B1 + vLLM.

## Phase 2: `ctikernels.mhc` (GPU qualification pending)

The `cti` provider now calls the package's `MhcExit` with preallocated outputs.
The package owns the reference math; this directory reexports it, so the installed
package has no dependency on `benchmark/`. `reference` is the BF16 market
composition, `exact_reference` is the FP64 oracle, and `measured_floor` gates
max-absolute **and** RMS errors independently for every exposed output. The
references accept `mode="exit"`, `"pre"`, or `"post"` and return output tuples
(the post reference returns a one-element tuple).

Residual products use FP32 TwoProduct (`mul.rn` plus FMA error), TwoSum and
compensation, followed by renormalization. BF16 conversion examines the low part
at an exact BF16 midpoint and nudges the FP32 high part toward it before rounding.
The unchanged high part supplies projection, squared-sum and Gram statistics.
Pre mode uses Dot2 compensated products, converts both components to FP64 before
addition, and performs all subsequent reductions and mix normalization in FP64.
Pre mixes, sigmoid, softmax, Sinkhorn, gains and rstd now remain FP64; only
post_out/comb_out stores convert once with `cvt.rn.f32.f64`. Exit statistics and
coefficients stay FP32. FP64 computation greatly reduces rounding error; it is
not a proof of correct rounding of arbitrary real-valued transcendental chains.
The FP32 coefficient chain uses accurate exp/sqrt and explicit `div.rn.f32`;
scale/base operations do not contract. Sinkhorn follows softmax + eps, initial
column division, then 19 row/column divisions. Norm weight is multiplied directly.
No external implementation code was copied.

- Decode uses 128 threads and capacity buckets 4/16/32, covering every T<=32.
  Warp w processes tokens w, w+4, ...; lane 4g+j owns eight columns of stream j.
  All lanes participate in projections and Gram statistics, with fixed XOR
  reductions across column groups and then streams. Residual vectors remain in
  shared FP32 storage through the output pass; y remains unrounded.
- S=64 is preferred when the grid is co-resident. At H=4096 on 48 SMs, capacities
  4/16 use S=64, G=64 (two CTAs/SM suffice); capacity 32 uses S=128, G=32.
  The S=128 fn slice is staged in two 64-column pieces, allowing the full T=32
  residual cache to fit. The 24 KiB fn allocation becomes totals after all fn
  readers finish. `prepare` checks shared memory, SM count, thread count and a
  conservative register allocation of 256 registers/thread before warming.
  `decode_slice=64` or `128` can override the selection when residency permits.
- Partials are [T, 35, padded-G], FP32 for exit and FP64 for pre; G is padded to
  a multiple of four for aligned 16-byte reads. Padding is initialized. Writers
  fence at GPU scope and synchronize; one leader atomically increments an Int64
  counter and acquire-spins to `(old // G + 1) * G`. Each capacity has its own
  counter because grids may differ between calls. Counters never reset during
  graph replay. Each reader issues coherent relaxed.gpu vector loads before
  adding partials in ascending G order, with no stale L1 reads.
- Thread v reduces statistic v, v+128, ... into shared totals. By default 16
  lanes per token own one matrix entry each, using shuffles for ascending-order
  row/column sums. One lane derives rstd from the Gram matrix. CTA zero writes
  post/comb. All threads share the final eight-column output vectors.
- Prefill uses 256 threads. The default chooses TT from 1/2/4/8/16 to supply
  at least two CTAs per SM when T permits. Explicit `prefill_tile` accepts
  1/2/4/8/16/32. Tiles up to four use 64-column chunks; larger tiles use 32.
  Double-buffered fn staging overlaps the next chunk with computation. TT=1/2
  additionally split columns over four/two groups of statistic owners and
  reduce their totals once, in fixed order. Other tiles use one owner per
  statistic. The output pass recomputes the same unrounded residual high parts
  from x/R/post/comb, with duplicate residual stores disabled. Both passes use
  the same error-free arithmetic helper. Pre mode reads R and keeps y in FP64.
- Every launch uses PDL. Decode stages the first fn piece, norm weights, scale
  and base before the wait; prefill stages its first fn chunk and scale/base.
  By default the dependent trigger immediately follows the wait. A successor can prefetch
  immutable weights, but its own wait blocks activation/workspace/counter use
  until this grid completes and flushes. The trigger releases only after every
  CTA executes it, so every exchange CTA is already resident. Post-only uses
  the same residual rounding helper and one destination vector per thread.

One instance owns its counters/workspace and serves **one stream at a time**,
including graphs. Switching streams requires an event/wait ordering the calls.
Keep the instance and all input/output tensors alive until work completes.
`close()` synchronizes before freeing its workspace; do not replay its graphs
subsequently. `prepare(token_counts)` compiles/warms all three modes and allocates
workspace outside capture. Token count is a runtime argument; counts in the same
bucket share one compiled specialization. In capture, supply every output;
eager calls allocate omitted outputs. Outputs cannot overlap inputs or one
another. There is no torch fallback and no autograd support.

First device checks (strict numerical floors remain qualification gates):

```bash
# Midpoint rounding, pre-mode projection accuracy, and changing exchange grids.
PYTHONPATH=python python -m pytest -q tests/gpu/test_mhc.py \
  -k 'residual_midpoint or graph_switches or (H4096 and pre and T1)' -x
# Full correctness, determinism and mutable graph replay matrix.
PYTHONPATH=python python -m pytest -q tests/gpu/test_mhc.py -x
# Cold-fn decode, then repeat with a non-PDL predecessor between exits.
PYTHONPATH=python python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti torch vllm --rows 1 2 3 4 --modes graph --count-kernels \
  --vllm-root artifacts/vllm-0.31.0/vllm --out mhc-phase2.json
PYTHONPATH=python python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti torch vllm --rows 1 2 3 4 --modes graph --spacer-mib 16 \
  --vllm-root artifacts/vllm-0.31.0/vllm --out mhc-spacer.json
```

`--spacer-mib N` defaults to zero. A positive value allocates two unrelated FP32
buffers and inserts one ordinary non-PDL `torch.add(source, 0, out=target)`
between exits, identically for all providers in eager and graph sequences.
There are calls-1 spacers per step. The report prints standalone spacer latency
and records read/write bytes, standalone step cost, amortized cost per exit and
provider latency minus that estimate. Standalone subtraction is an estimate:
cache interference and overlap can differ in the combined sequence. Kernel counts
include spacers when enabled. The two-rank benchmark does not expose this flag.

The GPU suite covers H=1024/4096/4104, all three modes, T=1/2/3/4/7/16/17/31/32/33/128/1000,
three replays with changed inputs, dependent pre/exit/post chains, positive and
negative midpoint rounding, and alternating decode grids in a graph. It compares
against full-batch GPU torch with TF32 disabled and a CPU FP64 oracle. Runtime
numerical floors and latency still require device qualification.

Round-2 baseline verification (superseded by the audit below): 61 CPU tests
and pre-commit passed. Offline sm_121a compilation covered 39 configurations:
H=1024/4096/4104, exit/pre capacities 4/16/32, prefill TT=8/16/32, and post.
CUDA device entry points are blocked during compilation. PTX confirms 16-byte
shared loads, cp.async before the PDL wait, the immediate dependent trigger,
coherent vector partial reads, and explicit FP32/FP64 rounding operations.

Round-2 H=4096 resource budgets follow (before the parallel chain): Registers and static shared memory come from
`cuobjdump -res-usage`; dynamic shared allocations are summed from the compiled
shared layouts, including alignment. The shared column includes the cubin's
additional 1 KiB static allocation (post has none).

| Kernel/bucket | Registers/thread, exit / pre | Shared bytes/CTA, exit / pre |
| --- | ---: | ---: |
| Decode capacity 4, S=64 | 80 / 166 | 30,144 / 30,144 |
| Decode capacity 16, S=64 | 80 / 166 | 42,672 / 42,672 |
| Decode capacity 32, S=128 | 128 / 128 | 92,400 / 92,400 |
| Prefill TT=8 | 72 / 64 | 31,088 / 32,208 |
| Prefill TT=16 (default) | 173 / 169 | 36,464 / 38,704 |
| Prefill TT=32 | 174 / 166 | 47,216 / 51,696 |
| Post | 40 | 0 |

All 39 cubins report LOCAL=0; the largest register count across shapes is 175.
At H=4096, decode exit capacities 4/16 report a 32-byte stack frame, and decode
pre capacity 32 reports 24 bytes; the other listed configurations report zero.
These compiler results establish resource bounds, not GPU correctness or speed.
Compare T=1..4 first for the former per-token latency slope, then prefill
T=128/1024/8192 and spacer-enabled decode. TT=8 has a substantially smaller
register footprint and is worth comparing with the default TT=16 on device.


## Round 3: fixed-cost diagnosis and IKET

The supplied GB10 measurements are the reproduction: decode rose from 13.6 us
at T=1 to roughly 64–67 us across T=1..4. No GPU was available for this revision;
the changes below are candidates requiring the commands here to qualify latency
and strict numerical floors. In particular, <79 us at T=128 and <768 us at T=1024
are targets, not measured results.

The CPU oracle now has `round_bf16` in `ctikernels.mhc._reference`. It restores
sticky information only at BF16 midpoints after FP64-to-FP32 conversion. Tests
cover both signs, even/odd ties, subnormals, large exponents and special values.
The residual midpoint GPU test uses it. `measured_floor` and `error_stats` already
subtract the unrounded FP64 oracle; they intentionally do not round the oracle.

### Evidence and falsifiable suspects

The following static SASS counts are for H=4096, S=64, capacity=4, exit mode,
sm_121a, without IKET. Counts cover the whole disassembly, including slow paths;
they are **not executed-instruction counts**.

| SASS/resource | Round-2 serial | Round-3 lane16 |
| --- | ---: | ---: |
| CALL sites | 654 | 48 |
| Static instructions (including helper functions) | 13,872 | 4,728 |
| MUFU.RCP sites | 651 | 45 |
| MUFU.EX2 sites | 24 | 4 |
| LDL / STL sites | 4 / 4 | 0 / 0 |
| Stack bytes/thread | 32 | 0 |
| Registers/thread | 80 | 96 |

1. **Serial coefficients are the leading suspect.** Twenty Sinkhorn iterations
   expand to 624 matrix divisions per token (16 initial row + 20*16 column +
   19*16 row). Their normal paths include reciprocal, correction FMAs, FCHK,
   branch/reconvergence machinery; the exceptional paths contain
   `__cuda_sm3x_div_rn_noftz_f32_slowpath` calls. Sqrt also has a guarded slow
   path; exp lowers to MUFU.EX2 plus arithmetic. The presence of a CALL does not
   establish that it is taken. The 16-lane implementation reduces serial work
   and instruction footprint without replacing accurate division. Prediction:
   `mhc.coefficients` shrinks sharply versus `CTI_MHC_CHAIN=serial`.
2. **Early PDL overlap may compete for issue/shared-memory resources.** The
   successor stages immutable weights before waiting. Its exchange still needs
   a fully resident grid. `CTI_MHC_EARLY_TRIGGER=0` moves the trigger to the end
   of each kernel, after all output work. Prediction: late triggering improves
   exchange/prologue or total chain time if overlap is harmful. Eager launch
   gaps can conceal this effect; compare the CUDA-graph benchmark too.
3. **Exchange arrival imbalance.** Only thread zero spins; all warps wait at a
   CTA barrier afterward. The new `mhc.exchange` range includes publication,
   arrival, spin and that barrier. G=64/P=64 at H=4096/S=64 has no padding to
   clear. At H=4104 only three padded columns are initialized, not the entire
   workspace. Readers sum G actual columns and use padded pitch only for
   alignment. Prediction: a large exchange range relative to residual work
   implicates imbalance/residency, not an unconditional 50 us zeroing pass.
4. **Staging and shared accesses.** cp.async waits happen after the PDL wait and
   before shared fn consumption; all threads reach the CTA barriers. Decode fn
   reads are 16-byte vectors in bank-friendly layouts. Residual-cache scalar
   stores have a strided stream layout and may conflict; no bank-conflict claim
   can be confirmed from SASS alone. Prediction: `mhc.async_wait` or
   `mhc.residual_partials`, rather than coefficients, dominates. There is no
   speculative bank-layout rewrite in this revision.

### Compile-time switches

Set these **before prepare/compilation**, in a fresh process for each comparison.
Prepared instances keep their existing compiled variants when the environment
changes. No configuration values are printed wholesale.

| Variable | Default | Effect |
| --- | --- | --- |
| `CTI_MHC_IKET` | `0` | `1` emits CuTe IKET push/pop ranges; default emits none |
| `CTI_MHC_CHAIN` | `lane16` | `serial` selects the old ownership pattern, still FP64 in pre |
| `CTI_MHC_PUBLISH` | `early` | For `last` exchange, publish gains and write output before Sinkhorn; `late` waits for Sinkhorn before publication |
| `CTI_MHC_EARLY_TRIGGER` | `1` | `0` moves dependent launch to kernel end |
| `CTI_MHC_PREFILL_TILE` | adaptive | Force 1/2/4/8/16/32; explicit constructor argument wins |

Trace ranges are `mhc.prologue`, `mhc.grid_wait`, `mhc.async_wait`,
`mhc.residual_partials`, `mhc.partial_stores`, `mhc.padding`, `mhc.exchange`,
`mhc.partial_read`, `mhc.coefficients`, and `mhc.output`. Prefill has no grid
exchange/padding/partial-read phase. Decode residual/partial ranges repeat per
warp token; prefill ranges repeat per chunk. They use the shipped IKET API,
with a constexpr flag guarding every call. No third-party source is vendored.

`run-iket` appends `iket` to `CUTE_DSL_COMPILER_OPT` (and the experimental DSL
counterpart) in its workload process. This keeps the emitted operations during
lowering. The CTI flag emits the operations; the CLI's compiler flag preserves
them and its injection pass instruments the resulting cubin. Ordinary production
compilation has neither markers nor timing operations. The offline audit uses
`--iket` explicitly for instrumented variants.

### GPU commands

Run from the repository root with nvidia-cutlass-dsl 4.8.0 and its IKET runner
installed. Choose fresh output directories for each trace run.

```bash
PYTHONPATH=python python -m pytest -q tests/gpu/test_mhc.py

# Default candidate, 86 exits per graph step, cold rotation.
PYTHONPATH=python python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti torch vllm sglang --rows 1 2 3 4 128 1024 8192 \
  --modes graph --calls 86 --weight-sets 16 --count-kernels \
  --vllm-root artifacts/vllm-0.31.0/vllm \
  --sglang-python vendor/sglang-72730d6/python --out mhc-round3.json

# Scheduling comparisons; keep the other switch at its default in each run.
PYTHONPATH=python CTI_MHC_CHAIN=serial python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti --rows 1 2 3 4 --modes graph --calls 86 --out mhc-serial.json
PYTHONPATH=python CTI_MHC_EARLY_TRIGGER=0 python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti --rows 1 2 3 4 --modes graph --calls 86 --out mhc-late.json
PYTHONPATH=python CTI_MHC_PREFILL_TILE=16 python -m benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti --rows 128 1024 --modes graph --out mhc-tt16.json

# Eager traces: run-iket requires a literal -- before the Python command.
for t in 1 4; do
  PYTHONPATH=python CTI_MHC_IKET=1 run-iket --output-dir out/mhc-iket-t$t \
    profile --postprocess all --keep -- \
    python -m benchmark.kernel_tuning.dsv4_mhc_exit.iket_probe --tokens "$t" --calls 32
  python -m benchmark.kernel_tuning.dsv4_mhc_exit.iket_summary \
    out/mhc-iket-t$t/*.trace.json --last-launches 32 --ns-per-tick 1
done
# Repeat those trace commands in fresh directories with CTI_MHC_CHAIN=serial,
# then CTI_MHC_EARLY_TRIGGER=0. --tokens 128 also probes prefill.
```

Adjust only the optional provider checkout paths to match their local installs.
The probe prepares and validates all buffers, then invokes bound compiled ABI
calls without per-exit allocation or validation overhead. It rotates at least
16 independent weights/states and uses the same >2*L2 cold-footprint policy as
`bench.py`. If 32 calls cannot cover the required rotation it fails explicitly;
increase both `--calls` and summary `--last-launches` together. It synchronizes
before/after the short eager chain, never between exits.

`--postprocess all` writes `iket_pid_*.pftrace`, `iket_pid_*.trace.json` and an
HTML viewer in the output directory; `--keep` also retains tracker/raw captures.
The JSON is IKET's compact decoded format with `stringTable`, `locationTable`
and per-launch ranges. The summary selects the last N instrumented launches,
sums repeated durations per warp, then takes the maximum warp total per CTA.
It prints mean/median over launch/CTA samples. Add `--by-cta` for a separate
mean/median per CTA coordinate across launches, useful for exchange imbalance.
Thus it does not multiply elapsed
time by the number of parallel warps. Summed phase values are diagnostic rather
than a reconstruction of wall time. `--ns-per-tick 1` assumes globaltimer
nanoseconds; omit it to report raw ticks for captures using a different timer.
IKET instrumentation and eager launch gaps perturb timing: the graph benchmark
remains the latency gate. Runtime JSON/Perfetto decoding remains GPU-unverified.

### Small-prefill occupancy and traffic

For 48 SMs the adaptive choices are TT=1 at T=128 (128 CTAs), TT=8 at T=1024
(128 CTAs), and TT=16 at T=8192 (512 CTAs). TT=1 uses four column shards and
64-column double buffers, halving the chunk/barrier count compared with round 2.
TT=8 has fewer statistics per thread than TT=16. Token tiles are part of the
compile cache key, so preparing T=128 cannot lock later T=1024 into its tile.

At H=4096 one fn matrix is 1,572,864 bytes. Each prefill CTA stages it once:
128 CTAs request 201,326,592 bytes from the cache hierarchy at both T=128 and
T=1024. These are not all DRAM bytes: CTAs reuse the same fn matrix during one
exit even though successive exits rotate cold matrices. The two residual passes
issue up to 80*T*H input bytes before cache reuse (each destination stream reads
four source streams and x); residual/output stores add 10*T*H and norm reads up
to 2*T*H. Repeated post/comb scalar reads add up to 20*T*H before
cache/loop-invariant reuse. Distinct x/R inputs across both passes total only 10*T*H bytes. Repeated
source-stream loads are L2-hot, and the second pass avoids loading rounded R'.
For T=128 this bounds requested fn+activation traffic at about 260 MB; T=1024
at about 671 MB. At an assumed effective cache delivery rate of 4 TB/s these
amounts alone take about 65/168 us; at 2 TB/s, about 130/336 us. These are traffic
estimates, not latency predictions: cache reuse can reduce transactions while
barriers, shared-memory traffic, arithmetic and occupancy add time. The small-T
speed target therefore needs the trace/benchmark, particularly at T=128.

Offline reproduction (no CUDA device calls):

```bash
PYTHONPATH=python python -m benchmark.kernel_tuning.dsv4_mhc_exit.compile_audit \
  --out out/mhc-sm121a-audit
PYTHONPATH=python python -m pytest -q \
  tests/modules/test_mhc_api.py tests/benchmarks/test_dsv4_mhc_exit_reference.py
```

The audit compiles every decode capacity, every prefill tile, all three modes
at H=1024/4096/4104 with and without IKET. Explicit S=128 overrides cover the smaller decode capacities too. Additional
H=4096 cases cover serial,
late-trigger and their combination on both paths, including FP64 pre. Cubin/PTX
artifacts are written outside the source tree for resource/SASS inspection.

Final CPU validation: 105 tests passed and pre-commit passed on the affected
files. All 168 offline sm_121a configurations passed with CUDA driver access
blocked; the audit also checks each cubin's IKET instrumentation status through
the shipped `iket.Context` API. Instrumented decode exposes all ten named ranges;
production cubins report no instrumentation. The GPU suite collects 127 cases;
none were executed in this CPU environment. GPU latency, numerical floors and
real trace readback remain for device qualification.


## Cluster decode and small-prefill revision

Exit now defaults to one cluster of eight 256-thread CTAs for T=1..8.
`CTI_MHC_DECODE=grid` restores the grid-exchange decode for those counts;
`CTI_MHC_DECODE=cluster` explicitly selects the default. Set the switch before
constructing the provider. T=9..32 retains the existing grid decode. Pre mode
retains compensated FP32 dots, FP64 reductions/coefficients and its existing
routing; post mode retains its existing kernel. No numerical floor was relaxed.

Each cluster CTA owns S=ceil(H/512)*64 columns per stream. All eight warps
participate in reductions; a second masked 512-column iteration covers S=576
at H=4104. Before the PDL wait, each warp elects one lane to issue bounded
`cp.async.bulk.prefetch.L2.global` requests for its share of the 96 fn rows.
Norm weights, scale and base are staged in shared memory. Bulk prefetch is
available on sm_121a; no plain-load fallback was needed. It is only a cache hint,
so it does not guarantee L2 hits or overlap on the device. See the
[NVIDIA PTX prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch).

After the wait and early dependent trigger, lanes retain the residual high parts
for all tokens in registers and shared memory. Projection processes four rows
at a time, reusing each pair of 16-byte global fn loads across the token bucket.
Warp totals reduce in order 0..7; every CTA then sums DSMEM publications in CTA
order 0..7. Publications and final totals have separate shared allocations.
A release/acquire cluster barrier publishes partials, and a second barrier
finishes all peer reads before any CTA can exit. The kernel does not access
arrival counters or the global partial workspace. The common ABI keeps a tensor
argument for stream selection; the provider still allocates workspace for its
other modes outside capture.

Every full-mask shuffle executes with all 32 lanes. Odd token counts use the
existing clamped-token/valid-store rule in the lane16 coefficient chain. Output
reloads each lane's own residual high parts and gathers all four streams with
whole-warp shuffles, avoiding the compiler's earlier output-pass spills.
The cluster IKET ranges are `mhc.prefetch`, `mhc.grid_wait`,
`mhc.loads_residual`, `mhc.projection`, `mhc.cta_reduce`,
`mhc.cluster_exchange`, `mhc.coefficients`, and `mhc.output`.
Gram work is included in projection. Instrumentation is absent by default.

Exit T=33..256 uses the grid decode structure with capacities 64/128/256.
It stages each fn column slice once and reuses it for every token. It recomputes
the identical unrounded residual high part for output instead of retaining all
T residuals in shared memory. Each capacity has an independent monotonic arrival
counter, so switching G during graph replay preserves epochs. Shared vector
loads use volatile PTX to prevent all 192 fn values from being hoisted across
the token loop and spilling. `prefill_tile` or `CTI_MHC_PREFILL_TILE` explicitly
restores the previous token-tiled prefill for T>32; T>256 keeps adaptive prefill.

At H=4096/T=128, fn requests to the global cache hierarchy fall from 192 MiB
to 1.5 MiB. The 192 MiB of per-token fn reuse now comes from shared memory.
The conservative launch policy selects S=64/G=64 for capacity 128 and
S=128/G=32 for capacity 256. The remaining all-to-all partial reads are 70 MiB
at T=128 and 35 MiB at T=256; each writes about 1.09 MiB of partials.
Two residual passes request up to 80*T*H activation input bytes before reuse,
and up to 20*T*H coefficient bytes; output stores add 10*T*H bytes.
Including fn and partial exchange gives roughly 128/148 MiB of requested global
traffic at T=128/256, or 34–67/39–78 us at an assumed 2–4 TB/s cache delivery
rate. Shared-memory and arithmetic costs still matter. Distinct activation
inputs/outputs plus fn are about 11.5/21.5 MiB, corresponding to about 52/98 us
at 230 GB/s if each reaches DRAM once. These overlapping bandwidth estimates
make 79/150 us plausible targets, not verified timings or guaranteed bounds.

The offline audit now accepts `--new-only --sass` and `--fastdiv 0|1`.
It covers H=1024/4096/4104, cluster capacities 1/2/4/8, grid capacities
64/128/256, distinct S=128 overrides, IKET on/off, and serial/late-trigger
controls. It blocks CUDA driver access and saves PTX, cubin, SASS and resource
reports outside the source tree. SASS checks reject LDL/STL, nonzero stack/local
storage, and CALL targets other than guarded div/sqrt exceptional slow paths.
For cluster it also checks prefetch before the dependency wait, mapped DSMEM
loads, and both cluster barriers. Accurate FP32 division still has static
slow-path CALL sites; this does not mean the normal arithmetic path calls them.

H=4096 production resource measurements (FASTDIV=0, lane16, early trigger):

| Variant | Registers/thread | Shared bytes/CTA, including static allocation |
| --- | ---: | ---: |
| Cluster capacity 1 | 48 | 12,804 |
| Cluster capacity 2 | 72 | 22,424 |
| Cluster capacity 4 | 128 | 41,632 |
| Cluster capacity 8 | 172 | 80,080 |
| Grid capacity 64, S=64 | 128 | 36,208 |
| Grid capacity 128, S=64 | 128 | 46,448 |
| Grid capacity 256, S=128 | 128 | 67,184 |

Shared totals sum aligned dynamic allocations and the cubin's 1 KiB static
allocation. SASS contains `UBLKPF.L2` before `ACQBULK`, ordinary `LDG.E.128`
projection loads, mapped `LD.E` DSMEM reads and `UCGABAR_ARV/UCGABAR_WAIT`.
FP32 fn loads appear outside the token reuse operations. The exact chain has
48 guarded div/sqrt CALL sites; FASTDIV keeps only the two sqrt slow paths.

Run CPU checks and offline audits with their installed dependencies:

```bash
PYTHONPATH=python python -m pytest -q \
  tests/modules/test_mhc_api.py tests/benchmarks/test_dsv4_mhc_exit_reference.py
PYTHONPATH=python CUTE_DSL_CUDA_VERSION=13 python -m \
  benchmark.kernel_tuning.dsv4_mhc_exit.compile_audit \
  --new-only --sass --out out/mhc-cluster-audit
PYTHONPATH=python CUTE_DSL_CUDA_VERSION=13 python -m \
  benchmark.kernel_tuning.dsv4_mhc_exit.compile_audit \
  --new-only --sass --fastdiv 1 --out out/mhc-cluster-audit-fastdiv
```

GPU qualification and timing commands:

```bash
# The fixture exercises both grid and cluster, all three modes, all T=1..8,
# H=1024/4096/4104, small-prefill boundaries, mutable replay and PDL chains.
PYTHONPATH=python python -m pytest -q tests/gpu/test_mhc.py

for variant in cluster grid; do
  PYTHONPATH=python CTI_MHC_DECODE="$variant" python -m \
    benchmark.kernel_tuning.dsv4_mhc_exit.bench \
    --providers cti --rows 1 2 3 4 5 6 7 8 16 32 128 256 1024 8192 \
    --modes graph --calls 86 --weight-sets 86 --out "mhc-$variant.json"
done
PYTHONPATH=python CTI_MHC_DECODE=cluster python -m \
  benchmark.kernel_tuning.dsv4_mhc_exit.bench \
  --providers cti --rows 1 2 4 8 --modes graph --calls 86 --weight-sets 86 \
  --spacer-mib 16 --out mhc-cluster-spacer.json

PYTHONPATH=python CTI_MHC_IKET=1 CTI_MHC_DECODE=cluster \
  run-iket --output-dir out/mhc-cluster-iket profile --postprocess all --keep -- \
  python -m benchmark.kernel_tuning.dsv4_mhc_exit.iket_probe --tokens 1 --calls 86
```

Use a fresh trace directory when repeating with `--tokens 4` or `--tokens 128`.
Repeat timings with `CTI_MHC_FASTDIV=1` separately. Device correctness, hangs,
latency and the T=128/256 targets remain GPU qualification gates.

Validation for this revision: 128 CPU tests passed, and pre-commit passed on
all seven changed files. All 136 new-variant sm_121a compilation/SASS audits
passed (68 each for FASTDIV=0/1), including instrumented variants. Every audited
cubin has STACK=0 and LOCAL=0 and contains no LDL/STL. Maximum register use is
176/thread across this matrix. The GPU suite collects 434 cases; none were run
in the CPU-only environment.

### Publish gains before Sinkhorn

For grid decode with `CTI_MHC_EXCHANGE=last`, the last CTA now reduces the
partials, computes RMS/pre/post and Gram rstd, publishes the five gains per
token, and release-stores the epoch flag. It writes its own output before
running softmax and Sinkhorn. Other CTAs acquire the flag and write output as
before. `CTI_MHC_PUBLISH=late` keeps publication behind Sinkhorn for comparison;
the knob participates in the compile cache key. Both coefficient ownership
patterns retain their arithmetic and reduction order.

`mhc.last_gains` covers partial read, gains, and publication;
`mhc.last_sinkhorn` covers softmax, Sinkhorn, and comb stores. In late mode the
Sinkhorn range is nested inside the publication range. The timeline summary
includes the final Sinkhorn range when reporting kernel end.

Totals alias the expired fn staging allocation only in the cached capacities.
The output pass reads cache/norm/gains, or recomputes residuals in registers;
it does not overwrite totals or small parameters. RMS has a separate shared
allocation that stays live until Sinkhorn. The fixed ascending-G partial
addition chain remains unchanged: distributing partial sums across threads
would change rounding.

The GPU exchange test compares explicit early and late publication bitwise
against `all` in exit/pre and lane16/serial, including odd token tails and
T=33 to exercise recompute output. The unset default retains mutable graph
replay coverage. GPU correctness and latency still require device validation.

Offline audit examples (with the CuTe DSL environment configured):

```bash
python benchmark/kernel_tuning/dsv4_mhc_exit/compile_audit.py \
  --out out/mhc-publish-compile --hidden 4096 --exchange-only \
  --caps 4 16 32 --publish early
python benchmark/kernel_tuning/dsv4_mhc_exit/compile_audit.py \
  --out out/mhc-publish-sass --hidden 4096 --exchange-only \
  --caps 4 --modes exit --publish early --sass
```

The small-exit SASS audit checks absence of local memory and publication before
the output stores and Sinkhorn exponential/division sequence, with IKET off/on.


### Decode projection and exchange scheduling

Cached grid decode capacities 4/16/32 default to `CTI_MHC_PROJ=split`.
Residual and Gram work retain token-owned warps. After a CTA barrier, warp w
projects rows w, w+4, ..., w+20 for all tokens in blocks of four, retaining
12 plain shared-load vectors per piece. Each lane keeps the original stream,
columns, product order, and butterfly order. Tail token masks guard loads,
products, and stores; all lanes execute every projection shuffle.
`CTI_MHC_PROJ=token` retains the previous mapping for bitwise comparison.
Recompute capacities and prefill retain their projection mapping. For cached
two-piece slices, the second fn staging loop stays rolled to prevent staging
addresses from spilling across the projection tile; first-piece staging
remains before the PDL wait.

Both grid exchanges default to `CTI_MHC_FENCE=single`: a CTA barrier precedes
thread 0's GPU acq_rel arrival RMW, and a CTA barrier precedes the last CTA's
thread 0 release-store of the gain flag. Acquire waits are unchanged. The
design follows CUTLASS's `GenericBarrier::arrive_inc` / `wait_eq` in
[include/cutlass/barrier.h](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/barrier.h).
`CTI_MHC_FENCE=all` retains the earlier all-thread fences. Both new switches
participate in the prepared-kernel cache key.

For more than 128 partial values, each reduction thread handles two values
per round, loading both before interleaving their independent ascending-G
addition chains. The smaller reduction retains one value per thread.

The sm_121a compiler lowers the thread-0 acq_rel RMW to a thread-0-only
`MEMBAR.ALL.GPU` plus `ATOMG`; release stores also retain a thread-0 fence.
Thus `single` removes the all-thread fence, but does not produce literally
zero SASS MEMBAR instructions. The audit checks that no MEMBAR precedes the
arrival CTA barrier, no explicit GPU fence remains in PTX, projection loads
are nonvolatile, and no local-memory spill instructions occur.

The GPU scheduling test covers split/token and single/all with both exchanges,
exit/pre, H=4096/4104 and T=1/2/3/4/5/8/17/32. It also captures defaults on a
non-default stream and replays changed inputs. Device tests and timings still
require GPU qualification.

```bash
python benchmark/kernel_tuning/dsv4_mhc_exit/compile_audit.py \
  --out out/mhc-split-compile --hidden 4096 4104 --exchange-only \
  --caps 4 16 32 --modes exit pre --proj split --fence single --sass
PYTHONPATH=python python -m pytest -q tests/gpu/test_mhc.py \
  -k 'decode_projection_and_fence_bits or last_arriver_matches_all'
```

Validation: 143 CPU tests passed. The 48-case sm_121a matrix (both exchanges,
exit/pre, capacities 4/16/32, IKET off/on, H=4096/4104) passed compilation and
SASS audit. Sixteen cap-4 control cases also passed for split/all-fence and
token/all-fence. All audited cubins have zero stack/local memory and no LDL/STL.
Default `last` exchange register counts per thread, across H and IKET:

| Mode | Capacity 4 | Capacity 16 | Capacity 32 |
| --- | ---: | ---: | ---: |
| exit | 128–130 | 128–130 | 128 |
| pre | 168 | 168 | 144–150 |

The `all` exchange peaks at 184 registers in pre/cap-32. Split projection
emits plain LDS.128 vectors without volatile serialization. GPU tests were
added but not run; bitwise equality, graph replay, and latency remain unmeasured.

## Round 4 results: last-arriver decode (GB10, 2026-10-17)

CUDA graph, 86 back-to-back exits with cold fn rotation, microseconds per exit,
same GB10 for every column (single-provider runs; paired market rows below).

| T | grid, all-CTA exchange | + last arriver | + gains before Sinkhorn | + split projection, single release | same + `CTI_MHC_FASTDIV=1` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.7 | 15.2 | 14.2 | 14.2 | 13.9 |
| 2 | 18.4 | 15.9 | 15.3 | 15.2 | 14.8 |
| 4 | 23.4 | 18.1 | 17.2 | 17.0 | 16.6 |
| 8 | 28.4 | 24.3 | 21.9 | 21.6-21.8 | 21.8 |

All rows pass the measured floor (ours <= torch composition on max and RMS error
vs the FP64 oracle). The variants are bitwise identical to each other, except
`CTI_MHC_FASTDIV=1`, which changes the Sinkhorn rounding and still passes the floor.
Split projection alone is neutral (14.15 vs 14.16 at T=1); the single-thread
release arrival saves about 0.4 us.

Trigger placement (`CTI_MHC_EARLY_TRIGGER`, T = 1/2/4/8): after the wait
14.2/15.2/17.0/21.6; after the first residual phase 15.3/16.4/18.3/20.5; before
the arrival 13.7/15.9/18.9/22.0. No placement wins every row, so the default
stays after the wait.

IKET timeline (eager probe, instrumented, T = 1, microseconds after the PDL wait,
CTA median [max]): residual loads end 3.1 [5.1]; partial stores end 4.9 [7.8];
arrival start 6.2 [8.7]; last arrival 9.3; last CTA partial read 0.6, gains and
publish 0.8, other CTAs see the flag at 11.2-11.4 and finish output by 11.6;
the last CTA's Sinkhorn runs 10.7-13.4 and ends the kernel. At T = 4 the
partial read is 1.7 us (140 values on one SM) and the last arrival is at 10.5.
The remaining critical path is therefore (a) the time to the last arrival
(activation loads queue behind the successor's 1.5 MiB fn prefetch, and 16 of
48 SMs hold two CTAs), (b) the 2.7 us Sinkhorn on one CTA, and (c) at T >= 4 the
single-SM partial read. ncu is not available on GB10 runners; the evidence is
IKET only.

Paired against vLLM 0.31.0 (2 TileLang kernels), same GB10, CUDA graph, 86 cold
exits, 10 pairs; ratio = time / faster provider, with its interval:

| T | ours (default) | ours (`CTI_MHC_FASTDIV=1`) | vLLM | default vs vLLM | FASTDIV vs vLLM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.19 | **13.71** | 14.02-14.03 | 1.011 [1.010, 1.013] | **vLLM 1.023 [1.023, 1.025] slower** |
| 2 | 15.22 | 14.78 | 14.21-14.26 | 1.068 [1.066, 1.071] | 1.040 [1.037, 1.041] |
| 4 | 17.02 | 16.47 | 14.38-14.44 | 1.184 [1.168, 1.187] | 1.141 [1.139, 1.150] |
| 8 | 21.96 | 22.00 | 17.49-17.51 | 1.257 [1.243, 1.345] | 1.255 [1.194, 1.328] |

Ours passes the measured floor in every row; vLLM's own output does not at
T = 1, 4 and 8 (its error exceeds the torch composition's; reported, not imposed).

## Round 5 (fused, pair)

`pair.py` adds **B1+mHC fused** to the composed-provider timing group, using the
same paired/interleaved samples as B1 + ours and B1 + vLLM. pair.py times T=1-4
in eager and graph modes; the two-node test (`tests/multi_node/test_two_node_mhc_exit.py`)
also qualifies T=8, 16 and 32. Random-input checks compare all four outputs
bitwise with B1 followed by the separate exit kernel, exchange rank digests, and
apply the existing FP64-oracle/torch-BF16 floor checks.

Fusion requires grid decode and a fitting active B1 provider. Preparation
compiles without issuing fused collectives; both ranks enter the first real
call together. `CTI_MHC_AR_PREFETCH=1` (default) prefetches each CTA's residual
rows while communication is pending; `0` disables that hint for comparison.

GB10 pair (two GB10 hosts, one RoCE cable), H=4096, cold fn, big-core pinned,
60 paired samples of 120 exits each (2.9-4.0 ms per sample), median per exit;
ratio = provider / fused with a bootstrap 95% CI of the median paired ratio
(GB10 rule: pass iff CI low > 1). Per step = 86 exits. NCCL 2.30.7 + vLLM is
shown for reference.

| T | Mode | Fused µs | Fused per step ms | B1 + ours µs | ratio [CI] | B1 + vLLM µs | ratio [CI] | NCCL + vLLM µs |
|---|---|---|---|---|---|---|---|---|
| 1 | eager | 32.67 | 2.81 | 39.48 | 1.208 [1.206, 1.211] | 78.07 | 2.389 [2.385, 2.393] | 77.59 |
| 1 | graph | 24.19 | 2.08 | 28.07 | 1.159 [1.157, 1.160] | 25.05 | 1.034 [1.018, 1.036] | 43.88 |
| 2 | eager | 33.38 | 2.87 | 40.61 | 1.213 [1.212, 1.217] | 80.31 | 2.407 [2.403, 2.411] | 80.37 |
| 2 | graph | 26.51 | 2.28 | 30.84 | 1.166 [1.163, 1.171] | 26.52 | 1.002 [1.000, 1.005] | 46.99 |
| 3 | eager | 33.19 | 2.85 | 40.66 | 1.224 [1.223, 1.228] | 79.88 | 2.406 [2.402, 2.409] | 79.95 |
| 3 | graph | 28.18 | 2.42 | 33.80 | 1.188 [1.185, 1.202] | 28.98 | 1.025 [1.004, 1.030] | 47.73 |
| 4 | eager | 33.55 | 2.89 | 40.96 | 1.218 [1.215, 1.221] | 80.19 | 2.387 [2.380, 2.390] | 80.29 |
| 4 | graph | 29.81 | 2.56 | 36.35 | 1.218 [1.204, 1.221] | 30.69 | 1.027 [1.005, 1.031] | 49.32 |

Every row passes against both composed paths. Correctness in the same run: the
fused outputs are bitwise equal to B1 followed by our separate exit and across
ranks, and within the torch-BF16 floor against the FP64 oracle, for every T and
mode. The fused launch costs about B1 plus our standalone exit (graph T=1:
9.5 + 14.6 µs measured separately), and the composed B1 + ours pays 4-7 µs
more. The network wait does not yet overlap exit work beyond the fn prologue
and the residual L2 prefetch. The T=2 graph margin over B1 + vLLM is small
(CI low 1.0003).
