# DraftAttention.run host time (HF-AD)

Base is tag v0.1.13rc6 (8eb02cc2). New is branch kernels/draft-hf, which binds
the operand contract once (docs).
Both trees run in one process on the same buffers through
`benchmark/kernel_tuning/draft_host_time.py`.

**Workload.** This is the engine's fixed-query rows of a Qwen3-8B + DFlash
mixed step: 32/8 heads, head dim 128, BF16, page size 1, causal full VERIFY,
Q=16, capacity 8192, contexts of about 1000 keys, and 36 layers. Each step calls
`prepare()` once, then `run()` once per layer on fresh views, as the engine does.
The views are the q/out rows after 64 paged rows and the layer's K/V pool viewed
as one-token pages.

**Timing.** Each call to `run()` is timed with perf_counter_ns, with launches
queued. The step is synchronized outside the timed region. The table gives the
median over 30 steps × 36 calls, and ms/step is the median per-step sum over the
36 calls. GB10 runs pinned to its Cortex-X925 cores.

**Outputs.** For every row, both trees write bit-identical outputs for all 36
layers, eager and replayed from a CUDA graph of prepare() plus the 36 calls.
Eager equals graph for both trees. Raw records:
[draft_host_time_rtx5090.jsonl](draft_host_time_rtx5090.jsonl),
[draft_host_time_rtx_pro_6000.jsonl](draft_host_time_rtx_pro_6000.jsonl) and
[draft_host_time_gb10.jsonl](draft_host_time_gb10.jsonl).

| Device | Provider | B | base us/call | new us/call | base ms/step | new ms/step |
|---|---|---:|---:|---:|---:|---:|
| RTX 5090 | cute-splitkv (auto) | 1 | 84.5 | 7.9 | 3.07 | 0.29 |
| RTX 5090 | cute-splitkv (auto) | 2 | 85.0 | 8.1 | 3.09 | 0.29 |
| RTX 5090 | cute-splitkv (auto) | 3 | 84.9 | 7.9 | 3.07 | 0.29 |
| RTX 5090 | cute-splitkv (auto) | 4 | 85.6 | 8.0 | 3.10 | 0.29 |
| RTX 5090 | flashinfer-fa2-device-schedule | 1 | 88.9 | 11.8 | 3.23 | 0.44 |
| RTX 5090 | flashinfer-fa2-device-schedule | 2 | 89.9 | 12.0 | 3.28 | 0.45 |
| RTX 5090 | flashinfer-fa2-device-schedule | 3 | 87.6 | 12.1 | 3.19 | 0.45 |
| RTX 5090 | flashinfer-fa2-device-schedule | 4 | 89.4 | 12.2 | 3.26 | 0.46 |
| RTX PRO 6000 | cute-splitkv (auto) | 1 | 80.6 | 7.3 | 2.95 | 0.27 |
| RTX PRO 6000 | cute-splitkv (auto) | 2 | 80.5 | 7.3 | 2.92 | 0.27 |
| RTX PRO 6000 | cute-splitkv (auto) | 3 | 80.3 | 7.1 | 2.94 | 0.26 |
| RTX PRO 6000 | cute-splitkv (auto) | 4 | 80.5 | 7.2 | 2.93 | 0.27 |
| RTX PRO 6000 | flashinfer-fa2-device-schedule | 1 | 84.5 | 11.1 | 3.09 | 0.41 |
| RTX PRO 6000 | flashinfer-fa2-device-schedule | 2 | 84.4 | 10.9 | 3.07 | 0.40 |
| RTX PRO 6000 | flashinfer-fa2-device-schedule | 3 | 83.9 | 10.9 | 3.06 | 0.40 |
| RTX PRO 6000 | flashinfer-fa2-device-schedule | 4 | 84.1 | 11.1 | 3.06 | 0.41 |
| GB10 | cute-splitkv (auto) | 1 | 78.4 | 9.6 | 2.83 | 0.35 |
| GB10 | cute-splitkv (auto) | 2 | 77.9 | 9.5 | 2.81 | 0.34 |
| GB10 | cute-splitkv (auto) | 3 | 78.2 | 9.7 | 2.82 | 0.35 |
| GB10 | cute-splitkv (auto) | 4 | 78.7 | 9.8 | 2.84 | 0.35 |
| GB10 | flashinfer-fa2-device-schedule | 1 | 83.5 | 14.7 | 3.02 | 0.53 |
| GB10 | flashinfer-fa2-device-schedule | 2 | 83.8 | 14.7 | 3.02 | 0.53 |
| GB10 | flashinfer-fa2-device-schedule | 3 | 81.4 | 14.7 | 2.94 | 0.53 |
| GB10 | flashinfer-fa2-device-schedule | 4 | 85.2 | 14.7 | 3.07 | 0.53 |

## Where the time went (RTX 5090, B=4, µs per call)

The new tree's figures are medians of 2000 calls of each part in isolation.
The base `run()` figures come from the table above. "Full operand check" is
base's per-call check (`_check_operands`, which the new tree keeps as its
fallback), timed in the same process.

| Part | base | new |
|---|---:|---:|
| `run()`, CuTe | 85.6 | 7.8 |
| `run()`, FA2 | 89.4 | 11.3 |
| operand checks | 74.1 (full operand check) | 3.8 |
| provider `run()`, CuTe launch | - | 3.9 |
| provider `run()`, FA2 `paged_run` | - | 7.7 |

Base's operand check spent most of its time in `_storage.disjoint`, about 77
`_byte_span` evaluations per call (cProfile). That check covers the call's
q/k/v/out together with the bound metadata and workspaces. The bound buffers'
intervals are now computed and checked at construction and by every
`prepare()`. Each call compares only q/k/v/out against them.

**Remaining launch cost.** The CuTe kernel's TVM-FFI call takes 3.6 µs with
torch tensors. Pre-converting the bound arguments to TVM-FFI tensors would save
about 0.8 µs. That was not adopted, because the call is already well under
target.

**Engine profile.** An engine profile saw about 165 µs per call and about
6.4 ms per step for the same 36 calls. That profile ran under the Python stack
tracer, which inflates Python-heavy frames. The unprofiled base figure here is
85 µs per call and 3.1 ms per step.
