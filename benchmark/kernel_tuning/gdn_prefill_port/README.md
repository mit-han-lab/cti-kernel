# Whole-layer GDN prefill comparison

`whole_prefill.py` and `whole_parity.py` are adapted from an earlier GDN prefill
benchmark (`bench_prefill.py` and `bench_parity.py`). They use this directory's
`memory_guard.py`. No comparator implementation is copied into this directory:
`GDNPP_BUNDLE` points to a separate comparator bundle.

The timed boundary is the whole prefill layer after projection: convolution,
gate/QKV preparation, recurrence, gated normalization, output and pool writes.
Preparation of each comparator's native inputs stays outside timing. Market
providers are SGLang **72730d6** (`sglang72`) and vLLM **0.31.0** (`vllm31`),
including vLLM's upstream backend selection. Product/v1 is this branch's
`GatedDeltaNetPrefill`; sm120 requires its **cute** provider. The checker accepts
v1's fixed `provider` property and the newer row dispatch API.

## Flags

- `whole_parity.py --ours {product,port}` selects the measured implementation
  (default `product`). `--extra-ours` requires `--ours port` and additionally
  pairs port against product v1 in the same invocation.
- `--port-options 'k=v,k=v'` on either script passes constructor options to
  `ctikernels.gdn.chunk_ws.prefill.PortPrefill`. Values are decimal integers,
  case-insensitive `true`/`false`, or strings. Keys must be unique identifiers;
  `num_k_heads`, `num_v_heads`, and `device` are reserved. Options are never
  echoed. The import is lazy; a missing entry point produces
  `PortPrefillUnavailableError` and a fixed explanatory message.
- `--set all --geoms 16x32,16x48` selects all **17 rows per geometry** (34 rows).
  Parity `--cases` uses global indices 0–16 and intersects `--set`.
- `--pairs 20` is the default. Self-calibration always uses exactly 20 pairs,
  independently of `--pairs`. `--no-vllm` disables that comparator.
- `whole_prefill.py --providers port,product,sglang72,vllm31` is the original
  single-provider median-replay driver; `--check` also compares each provider's
  output to the first provider. Port/v1 correctness runs even without `--check`.
- Parity retains optional `--trace`, `--roof`, `--reps-per-sample`,
  `--min-sample-us`, and `--out` JSONL output. Keep output files in the checkout.
  The inherited `--with-fi` requires a product with FlashInfer dispatch support;
  v1 has no such dispatch and fails that check explicitly.

## Unchanged measurement protocol

The paired ABAB protocol retains the source's rotating order: with two providers,
successive pairs run AB, BA; with market comparators, the starting provider rotates
each round. Each provider has a separate CUDA graph and pools, three graph warmups,
and a **256 MiB** L2 flush **outside** CUDA events around one replay. Pools reset
before each experiment and evolve between replays. Parity never falls back to
eager timing on graph capture failure (the inherited optional FlashInfer side
is explicitly eager). The standalone driver retains its labeled eager fallback.

Ratios use the median of full-precision paired `comparator_us / ours_us` samples;
values above 1 favor ours. The best-market ratio uses the faster comparator in
each round. Epsilon is linear-interpolated p95 of `abs(A/B - 1)` from 20 self
pairs of the comparator; best-market epsilon comes from the comparator with
the lower median latency. Verdicts are unchanged: `inconclusive` if eps > 0.02,
otherwise `pass` if ratio >= 1 - eps, else `fail`. Only emitted sample arrays
round to 0.01 us.

`--extra-ours` runs port/v1 after releasing market graphs, keeping the original
market interleaving and peak instance allowance. Its independent port samples
are `ours_v1_us` / `ours_v1_median_us`; product samples are `v1_us` /
`v1_median_us`, with `v1A_us` / `v1B_us`, `speedup_v1`, `eps_v1`, and
`verdict_v1`. The table's `ours us` is the market experiment's median; its v1
ratio uses the separate paired experiment, not the quotient of table medians.

Before any timing of a port row, fresh identically seeded port and product calls
record `port_vs_product`: output, written recurrent-state rows, and written
convolution-state rows each have `max_abs` and `rms_relative`, where the latter
is RMS(error)/RMS(product). Equal zero-reference tensors give zero; nonzero error
against an all-zero reference gives null. Nonfinite differences give null with
`finite: false`. Numerical differences are recorded without assertions.
`untouched_rows` records exact equality for each side's state and convolution
rows outside `slot_indices`; `untouched_rows_ok` combines those checks. Pools
are restored afterwards. Timing verdicts describe performance only.

Each row emits JSON with medians, ratios, epsilons, verdicts and correctness.
A final compact table shows row, T, N, ours/v1/vLLM/SGLang microseconds and ratios.
Diagnostics identify devices by GPU name and do not echo hostnames, paths,
arguments, environment values, or exception text.

## Runs

On an sm_120 GPU (RTX 5090 or RTX PRO 6000), from the repository root, after
`PortPrefill` is implemented:

```bash
GDNPP_BUNDLE=<comparator bundle> PYTHONPATH=python:. \
  python benchmark/kernel_tuning/gdn_prefill_port/whole_parity.py \
  --set all --geoms 16x32,16x48 --pairs 20 --ours port --extra-ours
```

The runs need CuTe DSL 4.8.0 and FlashInfer 0.7.0.post1, which the vLLM
comparator requires on supported CUDA 13 sm120 configurations.
`--port-options` can be appended once the port's options are defined.

## GP-CP3: single-sequence context parallelism

`PortPrefill(..., cp=True)` enables the bounded host planner for N=1. Use
`cp=False` or `cp_segments=1` for the previous fused path. `cp_segments=J`
forces a bounded target; 64-token alignment can reduce the actual J. The plan
is available as `ctikernels.gdn.chunk_ws.prefill.cp_plan(T, Hv, num_sms, segments)`.
Workspace is allocated on the first eager CP call. Warm every variant/row stride
before graph capture; warmed calls with `out` allocate no device storage or sync.

Run correctness on each device, then the paired cold-L2 graph benchmark:

```bash
export PYTHONPATH=python:.
python -m pytest -q tests/gpu/test_gdn_port_cp.py
python benchmark/kernel_tuning/gdn_prefill_port/cp3_bench.py --pairs 30
python benchmark/kernel_tuning/gdn_prefill_port/cp3_bench.py \
  --segments planner,2,3,4,5 --lengths 256,512,1024,2048,4096,8192,16384 --pairs 30
```

`cp3_bench.py` defaults to 16x32 and 16x48. It emits one JSON line per row and
then a compact table: actual `(J,L)`, paired CP/non-CP times, self-pair eps, and
A/fixup/C/history kernel times from one in-process Kineto replay. A Chrome-activity
fallback handles PyTorch builds that omit graph nodes from FunctionEvents. No subprocess
profiler is used. For an eager watchdog CP call, run
`cp2_debug.py --lengths 1000 --cp-segments 4 --check`. See [CP2.md](CP2.md) for the CP3 protocol and offline checks.
