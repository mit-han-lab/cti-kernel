# Warm-up autotune vs the offline table

`benchmark/kernel_tuning/linear_autotune_check.py` hides each Qwen3.5-9B (N, K) from this
device's table, runs `autotune(ms=..., top_k=3)` on it and compares the decision per M
bucket with the shipped table. Engine Ms: 1, 2, 4, 8, 16, 24, 32 on the RTX 5090 and RTX PRO
6000 (6 shapes, 42 keys each); 1, 2, 4, 8 on GB10 (4 shapes, 16 keys, X925-pinned). Gate:
correct vs FP64, bitwise repeatable and bucket-invariant, and every (regime, M) point with
speedup >= max(1.03, 1 + epsilon) over default-flag F.linear in the cold, engine and hot regimes.

| Device | Keys | Agree | Both CuTe | Autotune-only CuTe | Table-only CuTe | Autotuned CuTe min speedup | Tuning s/key (median, max) | Median epsilon |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 | 42 | 31 | 7 | 1 | 10 | 1.043–1.215 | 0.96, 1.75 | 1.4% |
| RTX PRO 6000 | 42 | 32 | 4 | 0 | 10 | 1.032–1.137 | 0.76, 1.44 | 1.5% |
| GB10 | 16 | 16 | 4 | 0 | 0 | 1.144–1.190 | 0.13, 2.15 | 7.3% |

Every table-only row loses the hot-L2 regime at runtime (speedup 0.66–0.98) or sits
inside the margin; the runtime gate always measures hot, while offline hot evidence is
optional, so warm-up errs toward cuBLAS. The autotune-only key
(RTX 5090, 4096x4096, M=1, reg 16x128, 3 splits) passed 1.06x cold, 1.17x engine, 1.5x hot.
A full 42-key pass takes about 35 s, compilation included.

One-command pipeline demo (RTX PRO 6000, hole 3072x4096, engine Ms 1, 2, 3, 4, 8): probe,
candidate, cold/engine/hot acceptance and re-gate finished; CuTe won cold (1.12–1.20x) but
not the engine regime (0.98–1.04x) or hot (0.79–0.84x), so `_dispatch.py` stayed unchanged.
