# Test and benchmark standard

Norm is the pilot; every other op converts to this standard in turn. The pattern follows flash-attention
(`tests/test_flash_attn.py`, `benchmarks/benchmark_flash_attention.py`,
`flash_attn/utils/benchmark.py`). The code here is our own.

## What we take from flash-attention

- **Accuracy is relative to PyTorch.** flash-attention computes `attention_ref(upcast=True)`
  in FP32 as the reference. It also computes the same formula at the input precision
  (`upcast=False, reorder_ops=True`), which measures the error that precision costs. The test
  then asserts `(out - out_ref).abs().max() <= 2 * (out_pt - out_ref).abs().max()`. No tolerance
  is tuned per test.
- **Grids come from parametrize.** Each axis is listed once (dtype, head dim, sequence length,
  causal, ...), and the test runs on the product.
- **Benchmarks are separate scripts.** They share a few timing helpers and compare against
  the reference implementations, one table per op. Correctness tests never time anything.

## Layout

| Path | Contents |
|---|---|
| `tests/<op>/oracle.py` | Reference formulas with an `upcast` flag. `upcast=True` returns FP64. `upcast=False` is plain PyTorch at the op's precision, written as frameworks write it. |
| `tests/<op>/test_api.py` | CPU only: the interface, the selection truth table, and error types. |
| `tests/<op>/test_<kind>.py` | GPU only (`pytestmark = pytest.mark.gpu`). `test_accuracy.py` holds the correctness grid. Other kinds, such as `test_contract.py` (aliasing, graphs, streams, PDL, bitwise invariants) and `test_training.py` (autograd), hold what is not a tolerance. |
| `tests/support/accuracy.py` | The shared gate: `assert_accurate`, `assert_bitwise`, `errors`, and `TOLERANCES`. |
| `benchmark/_harness/` | `timing.py` (eager and CUDA-graph per-call time), `paired.py` (A-B-A pairing on fixed cores), `stats.py` (bootstrap), `table.py`, `markets.py` (optional vLLM, SGLang and FlashInfer). |
| `benchmark/<op>/bench_<op>.py` | One script per op. It prints one table per op against the reference implementations. |

## Accuracy gate

`assert_accurate(ours, reference, baseline, family=...)` measures the maximum and the mean
absolute error over the finite elements. It passes when both satisfy

    error(ours vs reference) <= k * error(baseline vs reference) + atol

- `reference` is the FP64 oracle. Use FP32 only where FP64 does not exist, and say so.
- `baseline` is the oracle with `upcast=False`, in the same dtype as our output.
- `k` and `atol` are set once per op family and output dtype, in `TOLERANCES`. For the norm
  family: k = 2 for FP16/BF16 outputs, as in flash-attention; k = 4 for FP32 outputs, because
  only the reduction order differs; atol is 1e-5 for FP16/BF16 and 1e-6 for FP32, which only
  covers exact zeros. A per-call `tolerance=` override needs a measured reason at the call
  site.
- **Non-finite values.** The default `nonfinite="match"` requires NaN and ±inf exactly where
  the reference has them. `nonfinite="forbid"` requires every output element to be finite.
- **Failure report.** A failure gives the shape, dtype, family, k and atol; both errors for
  ours and for torch; and the worst element with its three values.
- **Gradients.** Compare FP64 autograd through the `upcast=True` oracle with autograd through
  the `upcast=False` oracle, under the same gate.
- **Exact contracts.** Results that must be exact use `assert_bitwise`, never a tolerance:
  stored residual sums, eager equal to graph, training equal to inference.

## Grids

- Axes come from the op's support: dtypes, weight conventions, layouts, widths, rows, and
  providers forced with `provider=` (or the provider variable for function APIs). The support
  is written once at the top of `test_accuracy.py`. Grids must not be generated from tuning tables.
- **Size.** Large products run in CI as an explicit core plus pairwise cover
  (`tests/gpu/_exhaustive.py`). Rows outside the core are marked `exhaustive`. Every value
  and every pair of values still runs in qualification.
- **Test ids.** Ids are stable. Explain any added, removed or moved id in the change.

## Benchmarks

- **Timing.** `timing.graph` captures N calls and reports the per-call device time of each
  replay, as an engine running CUDA graphs sees it. `timing.eager` adds host launch cost.
- **Pairing.** `paired.aba(ours, other, timer)` times ours, the other, then ours again in
  each round, and divides by the mean of the two flanks, so drift cancels. It reports the
  median ratio over rounds with a seeded 95% bootstrap interval. Pin the host with
  `--cores` (for example `16-19`, or `auto` for the first 4 cores the process may use).
- **Rows and columns.** Rows are engine shapes: decode B = 1-4 plus prefill rows, at each
  hidden size. Columns are ours, then each comparator with its time and the ratio
  comparator/ours. A ratio above 1 means ours is faster. A missing market prints `n/a` with
  the reason.
- Run with `python -m benchmark.norm.bench_norm --cores 16-19`.
- **CPU tests.** CPU tests of harness code live in `tests/benchmarks/`.

## Test tiers

| Tier | What runs |
|---|---|
| CPU | `tests/modules tests/benchmarks tests/quality tests/ci tests/<op> -m "not gpu"` |
| GPU | `tests/gpu tests/<op> -m "not exhaustive"`, where every collected test must pass. Exhaustive rows run in device qualification. |
