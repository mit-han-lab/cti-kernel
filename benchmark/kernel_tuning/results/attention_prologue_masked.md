# PM1: masked KV store in the fused attention prologue — valid-row cost

Design: `Prologue(masked=True)` selects a compile-time masked CuTe kernel; the
default `Prologue` keeps the N4 kernel (244fe48a). The unmasked SASS is byte-identical to 244fe48a's
for all 17 compiled configurations checked (hd256/hd128, BF16/e4m3/e5m2, 1/2/4 heads per warp,
2/4/8 warps, streaming on/off), so calls that do not ask for the mask keep today's kernel and speed.

In the masked kernel a negative `cache_indices` entry (graph padding) skips that token's K and V
cache work behind warp-uniform branches on `slot >= 0` (one token per warp): no destination address
is formed and nothing is stored; Q is still normalized and rotated. K computes every value first;
its cache addresses are computed in a `slot >= 0` branch (early, off the critical path, for one
head per warp; at the stores for more heads per warp), and the stores sit in a second `slot >= 0`
branch. V computes its addresses and stores in one `slot >= 0` branch. No host sync; capturable.

Correctness: tests/gpu/test_attention_prologue.py (masked: mixed and all-negative indices with
guard slots, byte-exact untouched bytes, BF16 and FP8, T in {1, 3, 4, 24, 64, 256, 512, 2048},
graph replay with indices rewritten in place and no recompilation; masked == unmasked byte for
byte on valid slots for T in {1, 4, 64, 512, 2048}).

## Valid-row timing: masked variant against the unmasked kernel

Protocol as `python -m benchmark.kernel_tuning.attention_prologue_masked_paired` (see its
docstring): 20 rounds per process, every buffer re-allocated per round, one CUDA graph of 256
chained calls (8 layers' buffers) per candidate, rotated order, A/A control (the unmasked kernel
compiled twice). Rounds of all processes pooled; ε = A/A 95% half-width + |A/A − 1|; a row passes
iff the pooled lower bound of unmasked/masked ≥ 1 − ε. Ratios > 1 mean the masked kernel is
faster.

### RTX 5090 (6 processes on 3 nodes)

| geometry | cache | T | unmasked µs | masked µs | unmasked/masked [95% CI] | ε | verdict |
|---|---|---:|---:|---:|---|---:|---|
| hd256 16/4 | bf16 | 1 | 0.712 | 0.728 | 0.983 [0.973, 0.994] | 0.0156 | **below** |
| hd256 16/4 | bf16 | 2 | 0.735 | 0.751 | 0.981 [0.973, 0.989] | 0.0078 | **below** |
| hd256 16/4 | bf16 | 3 | 0.735 | 0.747 | 0.990 [0.974, 1.007] | 0.0340 | pass |
| hd256 16/4 | bf16 | 4 | 0.764 | 0.772 | 0.990 [0.981, 0.998] | 0.0118 | **below** |
| hd256 16/4 | bf16 | 64 | 1.027 | 1.032 | 0.997 [0.989, 1.005] | 0.0080 | **below** |
| hd256 16/4 | bf16 | 128 | 1.344 | 1.343 | 1.002 [0.998, 1.006] | 0.0073 | pass |
| hd256 16/4 | bf16 | 256 | 1.913 | 1.881 | 1.016 [1.012, 1.020] | 0.0047 | pass |
| hd256 16/4 | bf16 | 512 | 3.061 | 3.167 | 0.967 [0.965, 0.969] | 0.0044 | **below** |
| hd256 16/4 | e4m3 | 1 | 0.903 | 0.915 | 0.989 [0.985, 0.992] | 0.0040 | **below** |
| hd256 16/4 | e4m3 | 2 | 0.920 | 0.942 | 0.981 [0.968, 0.995] | 0.0165 | **below** |
| hd256 16/4 | e4m3 | 3 | 0.922 | 0.944 | 0.978 [0.976, 0.980] | 0.0031 | **below** |
| hd256 16/4 | e4m3 | 4 | 0.947 | 0.968 | 0.979 [0.971, 0.987] | 0.0108 | **below** |
| hd256 16/4 | e4m3 | 64 | 1.471 | 1.487 | 0.987 [0.983, 0.992] | 0.0018 | **below** |
| hd256 16/4 | e4m3 | 128 | 1.636 | 1.640 | 0.994 [0.989, 1.000] | 0.0083 | **below** |
| hd256 16/4 | e4m3 | 256 | 1.961 | 1.972 | 0.992 [0.991, 0.994] | 0.0033 | **below** |
| hd256 16/4 | e4m3 | 512 | 2.998 | 2.970 | 1.009 [1.007, 1.012] | 0.0033 | pass |
| hd128 32/8 | bf16 | 1 | 0.721 | 0.743 | 0.960 [0.938, 0.981] | 0.0892 | pass |
| hd128 32/8 | bf16 | 2 | 0.755 | 0.779 | 0.964 [0.942, 0.986] | 0.0257 | **below** |
| hd128 32/8 | bf16 | 3 | 0.788 | 0.807 | 0.970 [0.946, 0.994] | 0.0583 | pass |
| hd128 32/8 | bf16 | 4 | 0.768 | 0.785 | 1.004 [0.973, 1.036] | 0.0411 | pass |
| hd128 32/8 | bf16 | 64 | 1.063 | 1.064 | 1.001 [0.997, 1.005] | 0.0048 | pass |
| hd128 32/8 | bf16 | 128 | 1.291 | 1.276 | 1.011 [1.008, 1.015] | 0.0007 | pass |
| hd128 32/8 | bf16 | 256 | 1.845 | 1.878 | 0.982 [0.980, 0.984] | 0.0036 | **below** |
| hd128 32/8 | bf16 | 512 | 2.984 | 3.044 | 0.978 [0.977, 0.980] | 0.0015 | **below** |
| hd128 32/8 | e4m3 | 1 | 0.806 | 0.831 | 0.968 [0.959, 0.977] | 0.0025 | **below** |
| hd128 32/8 | e4m3 | 2 | 0.840 | 0.840 | 1.006 [0.996, 1.015] | 0.0078 | pass |
| hd128 32/8 | e4m3 | 3 | 0.865 | 0.856 | 1.018 [1.004, 1.032] | 0.0223 | pass |
| hd128 32/8 | e4m3 | 4 | 0.848 | 0.848 | 1.010 [1.001, 1.019] | 0.0090 | pass |
| hd128 32/8 | e4m3 | 64 | 1.196 | 1.207 | 0.991 [0.986, 0.996] | 0.0024 | **below** |
| hd128 32/8 | e4m3 | 128 | 1.342 | 1.345 | 0.995 [0.987, 1.003] | 0.0087 | **below** |
| hd128 32/8 | e4m3 | 256 | 1.844 | 2.029 | 0.905 [0.902, 0.909] | 0.0058 | **below** |
| hd128 32/8 | e4m3 | 512 | 2.816 | 2.890 | 0.974 [0.972, 0.976] | 0.0033 | **below** |

12 of 32 rows pass; worst 0.905.

### GB10 (3 processes, X925-pinned)

| geometry | cache | T | unmasked µs | masked µs | unmasked/masked [95% CI] | ε | verdict |
|---|---|---:|---:|---:|---|---:|---|
| hd256 16/4 | bf16 | 1 | 0.751 | 0.772 | 0.983 [0.977, 0.988] | 0.0091 | **below** |
| hd256 16/4 | bf16 | 2 | 0.772 | 0.788 | 0.979 [0.970, 0.988] | 0.0154 | **below** |
| hd256 16/4 | bf16 | 3 | 0.772 | 0.780 | 0.993 [0.981, 1.005] | 0.0165 | **below** |
| hd256 16/4 | bf16 | 4 | 0.797 | 0.804 | 0.992 [0.983, 1.002] | 0.0074 | **below** |
| hd256 16/4 | bf16 | 64 | 2.063 | 2.090 | 0.991 [0.957, 1.027] | 0.0192 | **below** |
| hd256 16/4 | bf16 | 128 | 5.525 | 5.525 | 1.000 [0.997, 1.003] | 0.0120 | pass |
| hd256 16/4 | bf16 | 256 | 20.356 | 20.544 | 0.992 [0.989, 0.996] | 0.0049 | **below** |
| hd256 16/4 | bf16 | 512 | 60.018 | 59.923 | 1.001 [1.001, 1.002] | 0.0011 | pass |
| hd256 16/4 | e4m3 | 1 | 0.972 | 0.972 | 0.999 [0.991, 1.008] | 0.0094 | pass |
| hd256 16/4 | e4m3 | 2 | 0.981 | 0.997 | 0.984 [0.984, 0.984] | 0.0064 | **below** |
| hd256 16/4 | e4m3 | 3 | 0.996 | 0.996 | 1.000 [0.985, 1.014] | 0.0030 | **below** |
| hd256 16/4 | e4m3 | 4 | 1.020 | 1.015 | 0.997 [0.989, 1.005] | 0.0002 | **below** |
| hd256 16/4 | e4m3 | 64 | 1.967 | 1.956 | 1.009 [0.992, 1.025] | 0.0152 | pass |
| hd256 16/4 | e4m3 | 128 | 4.736 | 4.655 | 1.023 [1.013, 1.034] | 0.0137 | pass |
| hd256 16/4 | e4m3 | 256 | 18.056 | 18.042 | 1.000 [0.998, 1.003] | 0.0029 | pass |
| hd256 16/4 | e4m3 | 512 | 54.816 | 54.820 | 1.001 [1.000, 1.002] | 0.0019 | pass |
| hd128 32/8 | bf16 | 1 | 0.724 | 0.748 | 0.966 [0.961, 0.972] | 0.0092 | **below** |
| hd128 32/8 | bf16 | 2 | 0.785 | 0.809 | 0.972 [0.970, 0.975] | 0.0133 | **below** |
| hd128 32/8 | bf16 | 3 | 0.772 | 0.791 | 0.974 [0.970, 0.978] | 0.0025 | **below** |
| hd128 32/8 | bf16 | 4 | 0.780 | 0.807 | 0.970 [0.966, 0.974] | 0.0132 | **below** |
| hd128 32/8 | bf16 | 64 | 1.828 | 1.820 | 1.002 [0.987, 1.018] | 0.0213 | pass |
| hd128 32/8 | bf16 | 128 | 5.725 | 5.736 | 0.991 [0.980, 1.002] | 0.0105 | **below** |
| hd128 32/8 | bf16 | 256 | 20.410 | 20.406 | 1.000 [0.999, 1.001] | 0.0018 | pass |
| hd128 32/8 | bf16 | 512 | 59.568 | 59.563 | 1.001 [1.000, 1.002] | 0.0017 | pass |
| hd128 32/8 | e4m3 | 1 | 0.822 | 0.892 | 0.925 [0.918, 0.932] | 0.0080 | **below** |
| hd128 32/8 | e4m3 | 2 | 0.876 | 0.876 | 1.001 [0.992, 1.010] | 0.0009 | **below** |
| hd128 32/8 | e4m3 | 3 | 0.863 | 0.860 | 1.020 [1.010, 1.031] | 0.0178 | pass |
| hd128 32/8 | e4m3 | 4 | 0.884 | 0.868 | 1.009 [1.004, 1.014] | 0.0013 | pass |
| hd128 32/8 | e4m3 | 64 | 1.781 | 1.725 | 1.023 [1.012, 1.035] | 0.0267 | pass |
| hd128 32/8 | e4m3 | 128 | 4.489 | 4.479 | 1.001 [0.986, 1.016] | 0.0112 | **below** |
| hd128 32/8 | e4m3 | 256 | 17.996 | 17.967 | 1.004 [1.001, 1.008] | 0.0054 | pass |
| hd128 32/8 | e4m3 | 512 | 55.209 | 55.112 | 1.001 [1.000, 1.002] | 0.0020 | pass |

15 of 32 rows pass; worst 0.925.

### RTX PRO 6000

Pending: an RTX PRO 6000 was not available for this run.

## Where the masked kernel's time goes

- Unmasked calls: none (identical SASS).
- V branch: free. A diagnostic build with only V masked matches the unmasked kernel at T = 1–4
  on the RTX 5090. In the first masked build ptxas sank the V row load below the slot test (load
  after load); the shipped V keeps it ahead.
- K branch: the whole small-T cost (K-only masked build = full masked cost, 0.94–0.96 on the RTX 5090
  for hd256 at T = 1–4). The slot test now gates the K cache-address chain (constant loads of
  the cache pointer and strides, 64-bit slot × stride, LEA) that the unmasked kernel computes in
  the shadow of the cos/sin loads. Computing the address in its own early `slot >= 0` branch
  (ptxas if-converts it to predicated instructions) halves the loss for one head per warp
  (hd256 0.95 → 0.98–0.99 on the RTX 5090); for two or four heads per warp the early address
  moves rows by ±5–9% either way (ptxas scheduling), so those configurations compute it at the
  stores.
- Remaining rows below are 1–4% at T ≤ 4 (about 0.01–0.03 µs per call; GB10 hd128 e4m3 T = 1 is 0.925, about 8%), plus single
  configuration rows that flip with ptxas scheduling (RTX 5090 hd128 e4m3 T = 256 at 0.905 runs
  the same configuration as T = 512 at 0.974; the early-address build gave 1.015 and 0.94 there).

## Disclosed residual (speed reported, not gated)

The masked variant ships with the residual above; it is not tuned further (no per-device launch
table). Calls without `masked=True` are unaffected. Rows below 1 − ε, as unmasked/masked:

| device | rows below / measured | T ≤ 4 range | 64–512 rows range | worst row |
|---|---:|---|---|---|
| RTX 5090 | 20 / 32 | 0.964–0.990 | 0.905–0.997 | hd128 e4m3 T = 256: 0.905 |
| GB10 | 17 / 32 | 0.925–1.001 | 0.991–1.001 | hd128 e4m3 T = 1: 0.925 |
| RTX PRO 6000 | owed | | | |

Market contract for padding: vLLM's `reshape_and_cache_flash` skips `PAD_SLOT_ID = -1` (the
masked variant's contract); SGLang writes padded tokens to its reserved slot 0, a sink slot
(the default kernel's contract).
