<!-- SPDX-License-Identifier: MIT -->
# DFlash K0 DRAFT attention receipts

## 0.1.10 candidate (0.1.10rc2, round 3, installed wheel)

| field | value |
| --- | --- |
| package / version | `cti-kernels` 0.1.10rc2 (candidate build; release bump and publication remain separate) |
| package source SHA | `5809aa2d9913ef651802933e488b52f86ef72b2a` (`git archive`, only `pyproject.toml` version patched to 0.1.10rc2) |
| benchmark / test SHA | `992b32bf86fb953006e74c72c64b976539f0d13b`; its `python/` package tree is identical to the package source SHA; manifests record this benchmark SHA |
| wheel | `cti_kernels-0.1.10rc2-py3-none-any.whl`, SHA256 `bd42cd9fb75dad4da3a3b9552c6c3ce7fb6617ac4db28e6721e9ac7e38579461` |
| providers (all six manifests) | flashinfer-python 0.6.11.post3 (FA2 paged prefill + merge, private device schedule); Triton 3.7.1 (prepare) |
| environment (all six manifests) | torch 2.13.0+cu130, CUDA 13.0, BF16, page size 1, seed 20261014, atol=rtol=0.02; GPU tests also cover P=16 |
| devices | RTX 5090 (sm120); RTX PRO 6000 Blackwell Server (sm120); GB10 (sm121, aarch64, optional) |
| `op.provider` | `flashinfer-fa2-device-schedule` |
| scope | K0 only: padding helpers + `DraftAttention` (DFlash full, DFlash causal W4096, DFlash2 non-causal W2048). K1 sampling is out of scope. |
| acceptance | kernel: this package; integration acceptance (M2/M3/M5, E2E speed gate): the engine; version bump and publication: kernel release step |
| engine pin | stays at 0.1.8 until the engine accepts this candidate (contract K0.28) |

Pinned DFlash2 config (K0.28 precondition): `z-lab/Qwen3.8-27B-DFlash2@50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
`config.json` (SHA256 `873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`) has `is_causal: false`, `sliding_window: 2048`,
five `sliding_attention` layers, `dflash_config.block_size: 8`, 32/8 heads, head_dim 128:
the non-causal W2048 Q8 mask, TP1 32/8 and TP2 16/4 geometries tested here.

The GPU runs used the installed wheel with the benchmark and tests from the SHA above.
History: rc1 (round 2) passed 178 default tests per device; the extended `CTIK_DFLASH_EXTENDED=1` suite was last run on rc1 (366/366 on RTX 5090 and RTX PRO 6000); the only package code change since rc1 is the gather capacity check.

### Correctness (installed wheel)

Suite: `tests/gpu/test_dflash_draft_attention.py`, `test_dflash_padding.py`,
`test_dflash_integration.py`, `test_fixed_query_attention.py`, and
`test_dflash_capture_trace.py`.

| device | five-file GPU suite result |
| --- | --- |
| RTX 5090 | 197 passed, 1 warning |
| RTX PRO 6000 | 197 passed, 1 warning |
| GB10 | 197 passed, 1 warning |

`test_dflash_integration.py` covers independent exact-B1/2/3/4 captures of a whole
draft round, B4→B3 on shared buffers, inactive-row poison isolation, CUDA RNG
non-interference, no allocation during capture/replay, no sync in eager
(sync-debug `error`), and TP2 local head slices (16/4 of 32/8) against the TP1 oracle.
This is kernel-side evidence; real engine lifecycle, two-GPU TP2 and E2E acceptance
remain separate.

### Capture traces and sanitizer access checks

`benchmark/kernel_tuning/dflash_capture_trace.py` emits 12/12 passing cases per
device: B1–4 × full Q16, causal W4096 Q16 and non-causal W2048 Q8. Each complete
round contains helpers and two attention layers sharing one prepare. Three replays
per case change stable device metadata, including all-inactive and reused slots.
Graph node kinds are enumerated through the CUDA driver API
(`cuGraphGetNodes`/`cuGraphNodeGetType`) on the kept graph.

| device | B1–4 × three masks | graph nodes per kept graph | replay APIs | memcpy (capture/replay) | allocation counters (capture/replay) |
| --- | --- | --- | --- | --- | --- |
| RTX 5090 | 12/12 PASS | 10 KERNEL; 0 MEMSET; no other kinds | 36/36: only `cudaGraphLaunch` (1/replay) | 0 APIs / activities | all deltas 0 |
| RTX PRO 6000 | 12/12 PASS | 10 KERNEL; 0 MEMSET; no other kinds | 36/36: only `cudaGraphLaunch` (1/replay) | 0 APIs / activities | all deltas 0 |
| GB10 | 12/12 PASS | 10 KERNEL; 0 MEMSET; no other kinds | 36/36: only `cudaGraphLaunch` (1/replay) | 0 APIs / activities | all deltas 0 |

Every replay has 10 CUDA kernel activities matching the graph. Capture and replay
report no allocation, synchronization or memcpy APIs. `memory_allocated`,
`allocation.all.allocated`, `allocated_bytes.all.allocated` and `num_device_alloc`
are unchanged from before the capture body through all replays. Capture begin/end,
metadata staging and profiler teardown are outside the measured operation scopes.

`benchmark/kernel_tuning/dflash_access_check.py` ran all helpers and all three
masks with mixed/all-inactive/reused slots, poisoned metadata and exact-capacity
boundaries, standalone exact-size storage and allocator caching disabled.
Command options: `compute-sanitizer --tool memcheck --padding 4096 --print-limit 50`.

| device | access cases | Compute Sanitizer version | memcheck summary | return code |
| --- | --- | --- | --- | --- |
| RTX 5090 | 12/12 PASS | 2025.3.1.0 | ERROR SUMMARY: 0 errors | 0 |
| RTX PRO 6000 | 12/12 PASS | 2025.3.1.0 | ERROR SUMMARY: 0 errors | 0 |
| GB10 | 12/12 PASS | 2025.3.1.0 | ERROR SUMMARY: 0 errors | 0 |

Memcheck detects out-of-allocation accesses; the source dominance review and
poison tests supply complementary evidence for forbidden reads within live storage.
See Access review.

### Kernels per call

| path | kernels |
| --- | --- |
| candidate complete op (graph / eager) | 3 / 3: `prepare_draft` (Triton), FA2 `BatchPrefillWithPagedKVCacheKernel`, `PersistentVariableLengthMergeStatesKernel` |
| candidate prepare / run | 1 / 2 (one prepare serves every layer sharing the layout and mask) |
| FlashInfer eager plan + run / fixed-plan graph | 2 / 2 (host plan adds no CUDA kernels) |
| round 1 (769bcaf) | 7 (3 prepare + 2 query pack/scatter) |

### Speed (`benchmark/kernel_tuning/dflash_draft_attention.py`, installed wheel)

The protocol fix makes the gate matched eager-vs-eager: candidate device
prepare + run eager versus stock FlashInfer host plan + run eager. Synchronized
wall-clock intervals include completed GPU work and host planning, with synchronization
outside each operation. After 20 warm-up calls, use 20 interleaved pairs of 100 calls
per interval, alternating pair order; ε is p95 |ratio − 1| over 20 reference self-pairs.
PASS means median(candidate/reference) ≤ 1 + ε. All rows pass the numerical oracle.

Graph gate: **N/A**, never counted as PASS. Stock FlashInfer has no capture-safe
path following changing device lengths/page maps: `plan()` is host work, while the
fixed-plan graph replays a frozen plan. The candidate's capture qualification does
not supply a semantically equivalent stock graph comparator.
SGLang `72730d6360a5894b167500d5d1ad82c34c6bb116` DRAFT path: N/A (requires
host-known replay lengths; no standalone K0 device-prepare adapter).

`complete/fixed` and `run-only/fixed` are informational paired median ratios to
the FlashInfer fixed-plan graph floor, with the same timing protocol. Run-only
is the per-layer cost when one prepare serves the round. Neither ratio gates parity.

| device | mask | rows | correctness | eager gate PASS | eager ratio | complete/fixed | run-only/fixed | SOL util (graph complete) | SOL limiting resource | SOL regime counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | full | 40 | 40/40 PASS | 40/40 | 0.32–0.65 | 1.03–1.21 | 1.00–1.06 | 5.2–32.3% | floor/tensor | warm_l2: 40, partial_l2: 0 |
| RTX 5090 | causal_sliding | 40 | 40/40 PASS | 40/40 | 0.32–0.57 | 1.01–1.20 | 1.00–1.12 | 5.2–32.1% | floor/tensor | warm_l2: 40, partial_l2: 0 |
| RTX 5090 | noncausal_sliding | 48 | 48/48 PASS | 48/48 | 0.33–0.58 | 1.00–1.29 | 1.00–1.14 | 5.2–23.8% | floor/l2 | warm_l2: 48, partial_l2: 0 |
| RTX PRO 6000 | full | 40 | 40/40 PASS | 40/40 | 0.61–0.75 | 1.08–1.33 | 1.00–1.11 | 7.0–32.1% | floor/l2 | warm_l2: 40, partial_l2: 0 |
| RTX PRO 6000 | causal_sliding | 40 | 40/40 PASS | 40/40 | 0.62–0.70 | 1.09–1.25 | 1.00–1.07 | 7.1–32.0% | floor/l2 | warm_l2: 40, partial_l2: 0 |
| RTX PRO 6000 | noncausal_sliding | 48 | 48/48 PASS | 48/48 | 0.61–0.73 | 1.08–1.33 | 0.95–1.12 | 7.1–26.6% | floor/l2 | warm_l2: 48, partial_l2: 0 |
| GB10 | full | 40 | 40/40 PASS | 40/40 | 0.71–0.90 | 1.01–1.22 | 0.97–1.03 | 7.0–54.6% | dram/floor/l2 | warm_l2: 25, partial_l2: 15 |
| GB10 | causal_sliding | 40 | 40/40 PASS | 40/40 | 0.71–0.90 | 1.01–1.23 | 0.98–1.05 | 7.0–54.5% | dram/floor/l2 | warm_l2: 25, partial_l2: 15 |
| GB10 | noncausal_sliding | 48 | 48/48 PASS | 48/48 | 0.63–0.86 | 0.99–1.30 | 0.93–1.07 | 8.4–48.9% | dram/floor/l2 | warm_l2: 40, partial_l2: 8 |

### SOL (informational)

The rows' `ceiling_definition` uses the empirical useful-work reference from the
SOL ceilings in `benchmark/sol/ceilings/rtx5090.json`, `rtx-pro-6000.json` and
`gb10.json`. For serial graph kernels without PDL,
`t_ref = sum(max(graph-node floor, DRAM time, L2 time, BF16 MMA time))`.
One attention node accounts for BF16 Q reads, union-visible K/V reads per KV head,
one int32 page index per union key, and BF16 output writes;
`FLOPs = 4 * B * Hq * 128 * sum(query visible-key counts)`.
The other two launches are floor-only nodes. The benchmark replays identical
buffers, so `min(1, L2 size / working set)` of every attention byte is served at
the measured L2 read rate, with the rest at the sustained DRAM read rate as an
aggregate read/write traffic proxy. The working set is the attention bytes
accounted for above. The regime is `warm_l2` when all of it fits, otherwise
`partial_l2`. The ceiling tables give L2 sizes of 96 MiB for RTX 5090, 128 MiB for
RTX PRO 6000 and 24 MiB for GB10. `cold_t_ref_us` retains the all-DRAM reference.
The model excludes padding, scalar work, scratch traffic and eager host overhead.
These are measured ceilings, not theoretical hardware peaks.

The SOL fields were recomputed offline on CPU from the logged GPU latencies using
the committed `sol_estimate`, `l2_fraction` and `sol_report` helper functions;
the input `.sol.jsonl` rows record `sol_recomputed_offline: true`. No GPU timings
were rerun. The old cold-only reference overstated achievable time for L2-resident
working sets, which is why utilization ratios above 100% appeared. No rows exceed
100% with the L2-reuse reference.

`SOL util = 100 * t_ref / candidate graph complete latency`; it is informational.
At short contexts the three graph-node floors dominate; at long contexts an 8/16-row
query block per KV head gives too few CTAs to reach the L2 read rate, so FA2 is
latency-bound and the L2-rate reference stays optimistic.
The model's limiting resource is shown above; this does not profile the actual
bottleneck. SOL utilization does not count as a speed or qualification pass.

Remaining distance: complete/fixed is 1.00–1.29× on RTX 5090, 1.08–1.33× on
RTX PRO 6000 and 0.99–1.30× on GB10; run-only/fixed is respectively 1.00–1.14×,
0.95–1.12× and 0.93–1.07×. Round 1 was 1.11–1.66× for the complete op.
The complete op still pays one `prepare_draft` launch per round (three nodes versus
the floor's two). The floor launches attention and merge with PDL (FlashInfer's
default on sm90+). The candidate disables PDL because native FA2 reads its device
schedule (CTA validity, request/tile indices, chunk size) before `griddepcontrol.wait`;
otherwise it could overlap prepare and read a stale schedule. Closing this requires
a kernel that waits before reading device metadata (a CuTe DSL DRAFT kernel),
not adopted this round.

### Sizing and costs (K0.19/K0.27 recommendation)

| item | RTX 5090 | RTX PRO 6000 | GB10 |
| --- | --- | --- | --- |
| float workspace (payload bound = aligned = allocator backing) | 88.8–89.8 MB | 99.3 MB | 25.4 MB |
| int workspace (incl. 256 B fixed query CSR) | 1024–1792 B | 1280–2048 B | 768–1024 B |
| alignment | 256 B | 256 B | 256 B |
| first use (construction + first prepare/run, incl. JIT load) | 1.02–515.33 ms | 1.46–554.02 ms | 1.15–24982.30 ms |
| capture (prepare + run) | 0.20–7.16 ms | 0.36–4.97 ms | 0.26–3.93 ms |
| benchmark-process resident peak (includes two 128 MiB reference workspaces) | 392.6–577.2 MiB | 401.5–587.3 MiB | 331.5–516.8 MiB |

The float workspace is FlashInfer's conservative Split-KV partial-output bound
(`Hq × max(2·SMs/Hkv·128, …) × (d+1) × 4`), independent of the KV length.
Integer payload, aligned size and allocator backing also match in these rows.
First-use ranges retain the observed GB10 initialization/JIT costs up to 24.98 s;
these are outside steady-state timing. Resident peaks describe the benchmark
process, including reference allocations, rather than candidate-only memory.

### Rows

L is the verified context length (`context`); total KV length is L+Q. Timings are
paired marginal medians, while ratios are medians of paired ratios (rounding or
pairing can make their quotient differ). FI fixed-plan graph time is taken from
the complete/fixed pair, and SOL uses that pair's candidate graph complete time.
All rows have correctness PASS; graph gate N/A counts as no graph parity result.

| device | mask | Q | B | L | Hq/Hkv | candidate eager µs | FI eager µs | ε | eager gate | graph gate | candidate graph complete µs | FI fixed-plan graph µs | complete/fixed | run-only/fixed | SOL t_ref µs | SOL util (graph complete) | SOL regime | kernels/call graph/eager |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | full | 16 | 1 | 128 | 32/8 | 54.9 | 85.7 | 0.495 | PASS | N/A | 12.3 | 10.3 | 1.20 | 1.00 | 1.17 | 9.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 128 | 16/4 | 59.1 | 104.4 | 0.856 | PASS | N/A | 14.4 | 12.6 | 1.14 | 1.04 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 512 | 32/8 | 59.6 | 104.7 | 0.800 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.33 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 512 | 16/4 | 59.1 | 143.7 | 0.792 | PASS | N/A | 16.4 | 13.5 | 1.21 | 1.06 | 1.17 | 7.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4095 | 32/8 | 59.6 | 104.1 | 0.804 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.05 | 27.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4095 | 16/4 | 60.1 | 105.4 | 0.787 | PASS | N/A | 18.5 | 16.3 | 1.13 | 1.01 | 2.92 | 15.8% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4096 | 32/8 | 60.8 | 104.6 | 0.807 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.05 | 27.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4096 | 16/4 | 61.6 | 104.5 | 0.793 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 2.92 | 15.8% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4097 | 32/8 | 59.6 | 104.7 | 0.785 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.05 | 27.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 1 | 4097 | 16/4 | 59.3 | 103.7 | 0.798 | PASS | N/A | 18.5 | 16.3 | 1.14 | 1.01 | 2.92 | 15.8% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 128 | 32/8 | 60.0 | 109.2 | 0.761 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 128 | 16/4 | 59.1 | 187.1 | 0.752 | PASS | N/A | 21.2 | 20.5 | 1.03 | 1.00 | 1.17 | 5.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 512 | 32/8 | 59.1 | 111.3 | 0.747 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.88 | 8.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 512 | 16/4 | 59.4 | 190.7 | 0.747 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.33 | 5.9% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4095 | 32/8 | 59.7 | 190.4 | 0.738 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.32 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4095 | 16/4 | 59.3 | 112.1 | 0.750 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.05 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4096 | 32/8 | 60.1 | 112.4 | 0.745 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.32 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4096 | 16/4 | 59.5 | 151.0 | 0.742 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.05 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4097 | 32/8 | 61.2 | 112.4 | 0.746 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.33 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 2 | 4097 | 16/4 | 59.5 | 189.9 | 0.747 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.05 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 128 | 32/8 | 59.7 | 112.9 | 0.732 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.25 | 5.5% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 128 | 16/4 | 59.2 | 190.3 | 0.732 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 512 | 32/8 | 60.9 | 113.6 | 0.728 | PASS | N/A | 24.6 | 21.7 | 1.14 | 1.03 | 2.43 | 9.9% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 512 | 16/4 | 59.1 | 112.5 | 0.731 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.60 | 7.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4095 | 32/8 | 60.3 | 118.3 | 0.727 | PASS | N/A | 45.1 | 43.0 | 1.05 | 1.00 | 13.59 | 30.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4095 | 16/4 | 60.3 | 113.6 | 0.741 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.19 | 25.0% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4096 | 32/8 | 60.1 | 113.9 | 0.727 | PASS | N/A | 45.1 | 43.0 | 1.05 | 1.00 | 13.59 | 30.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4096 | 16/4 | 59.8 | 169.9 | 0.735 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.19 | 25.0% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4097 | 32/8 | 61.0 | 113.5 | 0.740 | PASS | N/A | 45.1 | 43.0 | 1.05 | 1.00 | 13.60 | 30.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 3 | 4097 | 16/4 | 61.7 | 163.0 | 0.562 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.19 | 25.0% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 128 | 32/8 | 61.3 | 164.8 | 0.386 | PASS | N/A | 22.8 | 20.8 | 1.10 | 1.06 | 1.41 | 6.2% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 128 | 16/4 | 61.7 | 155.3 | 0.470 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 512 | 32/8 | 60.9 | 162.4 | 0.544 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 2.98 | 12.1% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 512 | 16/4 | 61.6 | 164.8 | 0.329 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.88 | 8.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4095 | 32/8 | 61.9 | 181.8 | 0.550 | PASS | N/A | 55.4 | 53.3 | 1.04 | 1.00 | 17.86 | 32.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4095 | 16/4 | 61.6 | 165.6 | 0.361 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.32 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4096 | 32/8 | 61.0 | 146.6 | 0.565 | PASS | N/A | 55.4 | 53.3 | 1.04 | 1.00 | 17.86 | 32.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4096 | 16/4 | 61.7 | 173.2 | 0.497 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.32 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4097 | 32/8 | 61.7 | 165.6 | 0.418 | PASS | N/A | 55.4 | 53.3 | 1.04 | 1.00 | 17.87 | 32.3% | warm_l2 | 3/3 |
| RTX 5090 | full | 16 | 4 | 4097 | 16/4 | 60.5 | 116.0 | 0.713 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.33 | 26.7% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 128 | 32/8 | 59.0 | 104.7 | 0.793 | PASS | N/A | 12.3 | 10.3 | 1.20 | 1.00 | 1.17 | 9.5% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 128 | 16/4 | 58.9 | 122.2 | 0.794 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.02 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 512 | 32/8 | 60.3 | 105.6 | 0.784 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.32 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 512 | 16/4 | 59.9 | 104.3 | 0.794 | PASS | N/A | 14.4 | 12.8 | 1.12 | 1.12 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4095 | 32/8 | 59.7 | 183.3 | 0.791 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.04 | 27.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4095 | 16/4 | 59.5 | 105.4 | 0.790 | PASS | N/A | 16.5 | 16.4 | 1.01 | 1.00 | 2.91 | 17.7% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4096 | 32/8 | 59.8 | 144.0 | 0.794 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.04 | 27.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4096 | 16/4 | 59.8 | 105.1 | 0.789 | PASS | N/A | 16.5 | 16.2 | 1.01 | 1.01 | 2.91 | 17.7% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4097 | 32/8 | 59.9 | 105.0 | 0.796 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 5.04 | 27.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 1 | 4097 | 16/4 | 59.5 | 104.8 | 0.797 | PASS | N/A | 16.5 | 16.1 | 1.03 | 1.02 | 2.91 | 17.7% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 128 | 32/8 | 60.5 | 109.3 | 0.761 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 128 | 16/4 | 59.8 | 110.7 | 0.713 | PASS | N/A | 22.4 | 20.5 | 1.09 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 512 | 32/8 | 60.2 | 110.9 | 0.751 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.86 | 8.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 512 | 16/4 | 60.3 | 110.6 | 0.739 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.32 | 5.9% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4095 | 32/8 | 59.9 | 111.3 | 0.742 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4095 | 16/4 | 60.7 | 110.5 | 0.754 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.04 | 20.4% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4096 | 32/8 | 60.4 | 152.0 | 0.755 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4096 | 16/4 | 59.8 | 111.8 | 0.750 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.04 | 20.4% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4097 | 32/8 | 59.5 | 111.2 | 0.748 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 2 | 4097 | 16/4 | 59.9 | 190.4 | 0.754 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 5.04 | 20.4% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 128 | 32/8 | 59.0 | 153.0 | 0.747 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.25 | 5.5% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 128 | 16/4 | 59.2 | 113.8 | 0.740 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 512 | 32/8 | 59.9 | 189.3 | 0.746 | PASS | N/A | 24.6 | 21.9 | 1.13 | 1.02 | 2.40 | 9.8% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 512 | 16/4 | 59.9 | 177.3 | 0.729 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.59 | 7.0% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4095 | 32/8 | 60.5 | 113.4 | 0.744 | PASS | N/A | 45.1 | 43.1 | 1.05 | 1.00 | 13.54 | 30.0% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4095 | 16/4 | 59.9 | 191.2 | 0.725 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.16 | 24.9% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4096 | 32/8 | 59.7 | 113.5 | 0.737 | PASS | N/A | 45.1 | 43.1 | 1.05 | 1.00 | 13.54 | 30.0% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4096 | 16/4 | 61.4 | 154.4 | 0.727 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.16 | 24.9% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4097 | 32/8 | 61.3 | 141.5 | 0.567 | PASS | N/A | 45.1 | 43.1 | 1.05 | 1.00 | 13.54 | 30.0% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 3 | 4097 | 16/4 | 61.4 | 160.0 | 0.482 | PASS | N/A | 28.7 | 26.7 | 1.08 | 1.00 | 7.16 | 24.9% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 128 | 32/8 | 62.1 | 146.4 | 0.506 | PASS | N/A | 22.8 | 21.6 | 1.06 | 1.03 | 1.41 | 6.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 128 | 16/4 | 61.5 | 143.9 | 0.489 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 512 | 32/8 | 61.4 | 157.3 | 0.472 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 2.94 | 11.9% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 512 | 16/4 | 62.8 | 149.4 | 0.500 | PASS | N/A | 22.7 | 20.5 | 1.10 | 1.00 | 1.86 | 8.2% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4095 | 32/8 | 62.3 | 153.9 | 0.569 | PASS | N/A | 55.4 | 53.3 | 1.04 | 1.00 | 17.80 | 32.1% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4095 | 16/4 | 61.8 | 169.4 | 0.325 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4096 | 32/8 | 62.3 | 178.6 | 0.547 | PASS | N/A | 55.4 | 53.3 | 1.04 | 1.00 | 17.80 | 32.1% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4096 | 16/4 | 61.9 | 153.1 | 0.589 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4097 | 32/8 | 61.8 | 176.1 | 0.418 | PASS | N/A | 56.4 | 53.3 | 1.06 | 1.00 | 17.80 | 31.5% | warm_l2 | 3/3 |
| RTX 5090 | causal_sliding | 16 | 4 | 4097 | 16/4 | 62.1 | 149.1 | 0.552 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 9.29 | 26.6% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 128 | 32/8 | 54.3 | 94.2 | 0.464 | PASS | N/A | 10.3 | 10.3 | 1.00 | 1.00 | 1.17 | 11.4% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 128 | 16/4 | 59.8 | 112.1 | 0.631 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 512 | 32/8 | 60.1 | 123.7 | 0.657 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.14 | 1.20 | 8.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 512 | 16/4 | 60.6 | 121.8 | 0.724 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2047 | 32/8 | 59.9 | 149.0 | 0.419 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2047 | 16/4 | 59.8 | 112.3 | 0.558 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.57 | 9.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2048 | 32/8 | 60.5 | 124.8 | 0.637 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2048 | 16/4 | 60.1 | 161.7 | 0.574 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.57 | 9.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2049 | 32/8 | 60.4 | 147.3 | 0.573 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 2049 | 16/4 | 60.1 | 142.6 | 0.736 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.57 | 9.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 4096 | 32/8 | 60.0 | 125.0 | 0.586 | PASS | N/A | 16.2 | 14.4 | 1.12 | 1.00 | 2.35 | 14.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 1 | 4096 | 16/4 | 61.0 | 138.4 | 0.627 | PASS | N/A | 14.4 | 14.4 | 1.00 | 1.00 | 1.57 | 10.9% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 128 | 32/8 | 59.4 | 175.2 | 0.594 | PASS | N/A | 14.4 | 12.5 | 1.15 | 1.08 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 128 | 16/4 | 59.9 | 132.3 | 0.579 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.17 | 8.1% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 512 | 32/8 | 60.5 | 145.8 | 0.546 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.61 | 9.8% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 512 | 16/4 | 59.8 | 177.1 | 0.546 | PASS | N/A | 15.9 | 12.4 | 1.29 | 1.09 | 1.20 | 7.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2047 | 32/8 | 60.0 | 141.6 | 0.638 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 3.92 | 23.7% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2047 | 16/4 | 69.0 | 141.5 | 0.640 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2048 | 32/8 | 59.7 | 177.2 | 0.506 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 3.92 | 23.7% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2048 | 16/4 | 65.7 | 135.1 | 0.512 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2049 | 32/8 | 59.8 | 180.1 | 0.696 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 3.92 | 23.7% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 2049 | 16/4 | 60.1 | 138.7 | 0.484 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 4096 | 32/8 | 60.2 | 169.5 | 0.476 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 3.92 | 23.8% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 2 | 4096 | 16/4 | 60.0 | 156.6 | 0.691 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.35 | 14.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 128 | 32/8 | 60.6 | 179.2 | 0.521 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.17 | 5.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 128 | 16/4 | 60.2 | 177.7 | 0.504 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.17 | 5.7% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 512 | 32/8 | 60.4 | 136.5 | 0.462 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 2.02 | 9.0% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 512 | 16/4 | 60.3 | 147.0 | 0.615 | PASS | N/A | 22.6 | 20.4 | 1.11 | 1.01 | 1.40 | 6.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2047 | 32/8 | 69.8 | 139.9 | 0.533 | PASS | N/A | 26.7 | 24.6 | 1.08 | 1.00 | 5.48 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2047 | 16/4 | 68.6 | 136.7 | 0.565 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.13 | 13.9% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2048 | 32/8 | 60.5 | 154.2 | 0.499 | PASS | N/A | 26.7 | 24.6 | 1.08 | 1.00 | 5.48 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2048 | 16/4 | 60.2 | 137.8 | 0.547 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.13 | 13.9% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2049 | 32/8 | 60.5 | 150.9 | 0.527 | PASS | N/A | 26.7 | 24.6 | 1.08 | 1.00 | 5.48 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 2049 | 16/4 | 60.0 | 147.9 | 0.535 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.13 | 13.9% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 4096 | 32/8 | 61.0 | 159.7 | 0.384 | PASS | N/A | 26.7 | 24.6 | 1.08 | 1.00 | 5.48 | 20.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 3 | 4096 | 16/4 | 61.1 | 144.4 | 0.575 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.13 | 13.9% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 128 | 32/8 | 60.6 | 140.4 | 0.470 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.29 | 5.7% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 128 | 16/4 | 77.7 | 134.6 | 0.613 | PASS | N/A | 21.1 | 20.2 | 1.05 | 1.02 | 1.17 | 5.5% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 512 | 32/8 | 60.6 | 175.5 | 0.623 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 2.44 | 10.8% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 512 | 16/4 | 61.2 | 155.1 | 0.479 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 1.61 | 7.1% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2047 | 32/8 | 60.9 | 159.0 | 0.566 | PASS | N/A | 34.9 | 30.8 | 1.13 | 1.00 | 7.05 | 20.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2047 | 16/4 | 60.8 | 172.9 | 0.515 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.92 | 17.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2048 | 32/8 | 61.1 | 147.8 | 0.525 | PASS | N/A | 34.9 | 30.8 | 1.13 | 1.00 | 7.05 | 20.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2048 | 16/4 | 61.4 | 139.2 | 0.389 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.92 | 17.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2049 | 32/8 | 60.9 | 149.8 | 0.561 | PASS | N/A | 34.8 | 30.8 | 1.13 | 1.00 | 7.05 | 20.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 2049 | 16/4 | 61.5 | 160.2 | 0.496 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.92 | 17.3% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 4096 | 32/8 | 61.2 | 146.3 | 0.407 | PASS | N/A | 34.8 | 30.8 | 1.13 | 1.00 | 7.05 | 20.2% | warm_l2 | 3/3 |
| RTX 5090 | noncausal_sliding | 8 | 4 | 4096 | 16/4 | 61.3 | 157.7 | 0.516 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.92 | 17.3% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 128 | 32/8 | 53.4 | 71.3 | 0.163 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 128 | 16/4 | 58.2 | 84.8 | 0.081 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 512 | 32/8 | 58.1 | 84.4 | 0.082 | PASS | N/A | 16.5 | 12.4 | 1.33 | 1.00 | 1.45 | 8.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 512 | 16/4 | 58.0 | 83.9 | 0.093 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4095 | 32/8 | 58.6 | 84.4 | 0.061 | PASS | N/A | 18.5 | 16.5 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4095 | 16/4 | 58.5 | 84.0 | 0.085 | PASS | N/A | 18.5 | 14.4 | 1.28 | 1.00 | 2.49 | 13.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4096 | 32/8 | 58.3 | 83.9 | 0.087 | PASS | N/A | 18.5 | 16.5 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4096 | 16/4 | 58.1 | 84.6 | 0.035 | PASS | N/A | 18.5 | 14.4 | 1.28 | 1.00 | 2.49 | 13.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4097 | 32/8 | 58.3 | 83.8 | 0.073 | PASS | N/A | 18.5 | 16.5 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 1 | 4097 | 16/4 | 58.7 | 87.1 | 0.079 | PASS | N/A | 18.5 | 14.4 | 1.28 | 1.00 | 2.49 | 13.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 128 | 32/8 | 57.9 | 87.3 | 0.075 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 128 | 16/4 | 57.9 | 88.2 | 0.078 | PASS | N/A | 20.3 | 16.5 | 1.23 | 1.00 | 1.45 | 7.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 512 | 32/8 | 59.0 | 87.3 | 0.047 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.83 | 8.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 512 | 16/4 | 58.1 | 87.8 | 0.085 | PASS | N/A | 20.6 | 16.5 | 1.24 | 1.11 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4095 | 32/8 | 58.4 | 86.6 | 0.089 | PASS | N/A | 26.8 | 24.6 | 1.09 | 1.00 | 7.06 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4095 | 16/4 | 58.0 | 88.0 | 0.058 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.09 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4096 | 32/8 | 58.3 | 86.7 | 0.086 | PASS | N/A | 26.7 | 24.6 | 1.09 | 1.00 | 7.06 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4096 | 16/4 | 58.1 | 86.0 | 0.089 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.02 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4097 | 32/8 | 59.0 | 88.2 | 0.099 | PASS | N/A | 26.7 | 24.6 | 1.09 | 1.00 | 7.07 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 2 | 4097 | 16/4 | 57.7 | 87.2 | 0.073 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.02 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 128 | 32/8 | 60.3 | 90.7 | 0.103 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 128 | 16/4 | 58.1 | 91.0 | 0.080 | PASS | N/A | 20.6 | 16.8 | 1.23 | 1.08 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 512 | 32/8 | 58.9 | 90.9 | 0.093 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.01 | 2.27 | 10.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 512 | 16/4 | 58.4 | 90.2 | 0.065 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.62 | 7.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4095 | 32/8 | 58.4 | 90.4 | 0.132 | PASS | N/A | 33.4 | 30.8 | 1.08 | 1.00 | 10.11 | 30.3% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4095 | 16/4 | 58.0 | 90.2 | 0.098 | PASS | N/A | 24.7 | 22.1 | 1.12 | 1.02 | 5.54 | 22.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4096 | 32/8 | 58.2 | 89.7 | 0.078 | PASS | N/A | 33.4 | 30.8 | 1.08 | 1.00 | 10.11 | 30.3% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4096 | 16/4 | 58.1 | 89.9 | 0.082 | PASS | N/A | 24.7 | 22.3 | 1.11 | 1.02 | 5.55 | 22.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4097 | 32/8 | 58.3 | 89.7 | 0.084 | PASS | N/A | 34.3 | 30.8 | 1.11 | 1.00 | 10.11 | 29.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 3 | 4097 | 16/4 | 58.0 | 89.4 | 0.076 | PASS | N/A | 24.7 | 21.5 | 1.15 | 1.05 | 5.55 | 22.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 128 | 32/8 | 58.0 | 92.1 | 0.082 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.00 | 1.58 | 7.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 128 | 16/4 | 57.9 | 93.0 | 0.039 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.01 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 512 | 32/8 | 58.1 | 91.8 | 0.072 | PASS | N/A | 22.6 | 18.7 | 1.21 | 1.03 | 2.70 | 11.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 512 | 16/4 | 58.5 | 93.9 | 0.085 | PASS | N/A | 21.3 | 18.5 | 1.15 | 1.00 | 1.83 | 8.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4095 | 32/8 | 59.6 | 95.2 | 0.073 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4095 | 16/4 | 59.1 | 97.0 | 0.103 | PASS | N/A | 26.8 | 24.6 | 1.09 | 1.00 | 7.07 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4096 | 32/8 | 58.6 | 93.0 | 0.072 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4096 | 16/4 | 58.7 | 93.4 | 0.097 | PASS | N/A | 26.8 | 24.6 | 1.09 | 1.00 | 7.07 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4097 | 32/8 | 59.4 | 94.0 | 0.077 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | full | 16 | 4 | 4097 | 16/4 | 58.7 | 93.5 | 0.086 | PASS | N/A | 26.8 | 23.8 | 1.13 | 1.04 | 7.07 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 128 | 32/8 | 57.4 | 83.0 | 0.077 | PASS | N/A | 14.5 | 12.4 | 1.17 | 1.00 | 1.45 | 10.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 128 | 16/4 | 57.1 | 82.6 | 0.098 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 512 | 32/8 | 58.4 | 85.1 | 0.068 | PASS | N/A | 16.5 | 13.3 | 1.24 | 1.04 | 1.45 | 8.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 512 | 16/4 | 57.7 | 82.3 | 0.077 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4095 | 32/8 | 59.7 | 85.0 | 0.122 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4095 | 16/4 | 57.6 | 83.9 | 0.089 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4096 | 32/8 | 58.4 | 85.9 | 0.137 | PASS | N/A | 18.5 | 16.5 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4096 | 16/4 | 58.0 | 84.1 | 0.084 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4097 | 32/8 | 58.0 | 83.8 | 0.076 | PASS | N/A | 18.5 | 16.5 | 1.13 | 1.00 | 4.02 | 21.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 1 | 4097 | 16/4 | 58.5 | 83.7 | 0.089 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 128 | 32/8 | 57.1 | 85.7 | 0.137 | PASS | N/A | 20.6 | 18.4 | 1.12 | 1.01 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 128 | 16/4 | 57.1 | 86.6 | 0.095 | PASS | N/A | 19.0 | 16.5 | 1.15 | 1.00 | 1.45 | 7.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 512 | 32/8 | 58.3 | 87.7 | 0.046 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.83 | 8.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 512 | 16/4 | 58.6 | 86.3 | 0.065 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.07 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4095 | 32/8 | 57.9 | 85.3 | 0.084 | PASS | N/A | 26.8 | 24.7 | 1.09 | 1.00 | 7.06 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4095 | 16/4 | 59.0 | 85.7 | 0.087 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.00 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4096 | 32/8 | 59.7 | 93.6 | 0.124 | PASS | N/A | 26.9 | 24.7 | 1.09 | 1.00 | 7.06 | 26.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4096 | 16/4 | 58.8 | 91.0 | 0.076 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.00 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4097 | 32/8 | 60.6 | 91.7 | 0.104 | PASS | N/A | 26.8 | 24.7 | 1.09 | 1.00 | 7.06 | 26.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 2 | 4097 | 16/4 | 57.7 | 87.7 | 0.068 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.00 | 4.02 | 17.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 128 | 32/8 | 58.7 | 91.3 | 0.227 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 128 | 16/4 | 57.4 | 89.5 | 0.065 | PASS | N/A | 20.6 | 16.6 | 1.24 | 1.02 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 512 | 32/8 | 57.8 | 90.3 | 0.064 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.00 | 2.27 | 10.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 512 | 16/4 | 59.1 | 94.7 | 0.067 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.62 | 7.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4095 | 32/8 | 58.0 | 91.3 | 0.076 | PASS | N/A | 34.9 | 30.8 | 1.13 | 1.00 | 10.11 | 29.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4095 | 16/4 | 58.4 | 91.3 | 0.138 | PASS | N/A | 24.7 | 22.5 | 1.10 | 1.01 | 5.54 | 22.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4096 | 32/8 | 57.8 | 89.2 | 0.095 | PASS | N/A | 34.9 | 30.8 | 1.13 | 1.00 | 10.11 | 29.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4096 | 16/4 | 58.0 | 89.8 | 0.123 | PASS | N/A | 24.7 | 22.6 | 1.09 | 1.00 | 5.54 | 22.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4097 | 32/8 | 61.2 | 92.9 | 0.121 | PASS | N/A | 34.9 | 30.8 | 1.13 | 1.00 | 10.11 | 29.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 3 | 4097 | 16/4 | 58.7 | 93.0 | 0.127 | PASS | N/A | 24.7 | 22.6 | 1.09 | 1.00 | 5.54 | 22.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 128 | 32/8 | 59.0 | 93.6 | 0.235 | PASS | N/A | 22.3 | 18.5 | 1.20 | 1.00 | 1.58 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 128 | 16/4 | 59.9 | 96.0 | 0.062 | PASS | N/A | 20.6 | 18.4 | 1.12 | 1.00 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 512 | 32/8 | 59.8 | 94.3 | 0.094 | PASS | N/A | 22.6 | 18.5 | 1.22 | 1.03 | 2.70 | 11.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 512 | 16/4 | 59.3 | 94.1 | 0.082 | PASS | N/A | 20.6 | 18.5 | 1.11 | 1.00 | 1.83 | 8.9% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4095 | 32/8 | 63.2 | 100.1 | 0.230 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4095 | 16/4 | 57.1 | 90.0 | 0.049 | PASS | N/A | 27.0 | 24.6 | 1.10 | 1.00 | 7.07 | 26.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4096 | 32/8 | 56.5 | 90.0 | 0.086 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4096 | 16/4 | 56.3 | 91.4 | 0.035 | PASS | N/A | 27.5 | 24.6 | 1.12 | 1.00 | 7.07 | 25.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4097 | 32/8 | 58.4 | 94.3 | 0.116 | PASS | N/A | 41.1 | 37.0 | 1.11 | 1.00 | 13.16 | 32.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | causal_sliding | 16 | 4 | 4097 | 16/4 | 56.5 | 89.7 | 0.077 | PASS | N/A | 27.8 | 24.6 | 1.13 | 1.00 | 7.07 | 25.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 128 | 32/8 | 52.9 | 72.6 | 0.109 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 128 | 16/4 | 56.8 | 84.4 | 0.112 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 512 | 32/8 | 57.5 | 84.5 | 0.090 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 512 | 16/4 | 55.7 | 80.1 | 0.148 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2047 | 32/8 | 56.3 | 81.6 | 0.075 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.49 | 15.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2047 | 16/4 | 57.0 | 82.9 | 0.087 | PASS | N/A | 14.4 | 13.4 | 1.08 | 0.95 | 1.73 | 12.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2048 | 32/8 | 58.2 | 85.6 | 0.064 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2048 | 16/4 | 56.7 | 83.9 | 0.125 | PASS | N/A | 14.4 | 13.3 | 1.09 | 0.99 | 1.73 | 12.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2049 | 32/8 | 57.7 | 84.8 | 0.068 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 2049 | 16/4 | 57.2 | 84.0 | 0.081 | PASS | N/A | 14.4 | 13.4 | 1.08 | 0.95 | 1.73 | 12.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 4096 | 32/8 | 57.7 | 85.8 | 0.062 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 1 | 4096 | 16/4 | 56.4 | 85.0 | 0.087 | PASS | N/A | 15.7 | 13.6 | 1.15 | 1.04 | 1.73 | 11.0% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 128 | 32/8 | 56.7 | 87.6 | 0.079 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 128 | 16/4 | 56.7 | 86.3 | 0.082 | PASS | N/A | 14.4 | 12.4 | 1.17 | 1.00 | 1.45 | 10.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 512 | 32/8 | 58.3 | 88.2 | 0.058 | PASS | N/A | 16.5 | 12.4 | 1.33 | 1.07 | 1.77 | 10.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 512 | 16/4 | 56.7 | 86.5 | 0.089 | PASS | N/A | 16.4 | 12.4 | 1.33 | 1.00 | 1.45 | 8.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2047 | 32/8 | 56.8 | 86.5 | 0.080 | PASS | N/A | 17.3 | 14.4 | 1.20 | 1.00 | 4.02 | 23.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2047 | 16/4 | 59.2 | 89.0 | 0.047 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2048 | 32/8 | 56.8 | 87.0 | 0.105 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 4.02 | 24.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2048 | 16/4 | 57.2 | 88.9 | 0.074 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2049 | 32/8 | 56.7 | 86.5 | 0.079 | PASS | N/A | 16.5 | 14.4 | 1.14 | 1.00 | 4.02 | 24.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 2049 | 16/4 | 56.9 | 86.7 | 0.072 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 4096 | 32/8 | 56.7 | 88.1 | 0.093 | PASS | N/A | 16.6 | 14.4 | 1.15 | 1.00 | 4.02 | 24.2% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 2 | 4096 | 16/4 | 57.5 | 86.7 | 0.171 | PASS | N/A | 16.5 | 14.3 | 1.15 | 1.01 | 2.49 | 15.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 128 | 32/8 | 56.9 | 90.1 | 0.069 | PASS | N/A | 20.6 | 16.5 | 1.24 | 1.10 | 1.45 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 128 | 16/4 | 56.9 | 90.6 | 0.067 | PASS | N/A | 18.7 | 16.5 | 1.13 | 1.00 | 1.45 | 7.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 512 | 32/8 | 57.1 | 89.9 | 0.080 | PASS | N/A | 20.6 | 17.5 | 1.18 | 1.05 | 2.18 | 10.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 512 | 16/4 | 56.9 | 89.7 | 0.058 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.00 | 1.57 | 7.7% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2047 | 32/8 | 57.8 | 90.1 | 0.074 | PASS | N/A | 22.7 | 20.5 | 1.11 | 1.00 | 5.54 | 24.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2047 | 16/4 | 57.4 | 89.4 | 0.069 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 3.26 | 15.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2048 | 32/8 | 57.5 | 92.4 | 0.034 | PASS | N/A | 22.7 | 20.5 | 1.10 | 1.00 | 5.54 | 24.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2048 | 16/4 | 56.9 | 90.3 | 0.060 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 3.26 | 15.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2049 | 32/8 | 56.9 | 89.8 | 0.080 | PASS | N/A | 22.7 | 20.5 | 1.10 | 1.00 | 5.54 | 24.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 2049 | 16/4 | 57.0 | 90.1 | 0.032 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.00 | 3.26 | 15.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 4096 | 32/8 | 57.1 | 90.9 | 0.076 | PASS | N/A | 23.0 | 20.5 | 1.12 | 1.00 | 5.54 | 24.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 3 | 4096 | 16/4 | 56.6 | 90.1 | 0.084 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 3.26 | 15.8% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 128 | 32/8 | 56.9 | 91.1 | 0.067 | PASS | N/A | 20.6 | 16.9 | 1.22 | 1.05 | 1.46 | 7.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 128 | 16/4 | 56.9 | 93.8 | 0.057 | PASS | N/A | 19.6 | 16.5 | 1.19 | 1.00 | 1.45 | 7.4% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 512 | 32/8 | 57.4 | 94.4 | 0.057 | PASS | N/A | 20.6 | 18.1 | 1.13 | 1.02 | 2.58 | 12.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 512 | 16/4 | 56.8 | 93.3 | 0.066 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.05 | 1.78 | 8.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2047 | 32/8 | 56.9 | 91.9 | 0.068 | PASS | N/A | 26.5 | 22.6 | 1.17 | 1.00 | 7.06 | 26.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2047 | 16/4 | 57.3 | 92.6 | 0.069 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 4.02 | 19.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2048 | 32/8 | 57.0 | 91.5 | 0.076 | PASS | N/A | 26.6 | 22.6 | 1.18 | 1.00 | 7.06 | 26.6% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2048 | 16/4 | 59.2 | 96.0 | 0.042 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 4.02 | 19.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2049 | 32/8 | 57.0 | 93.4 | 0.057 | PASS | N/A | 26.7 | 22.6 | 1.18 | 1.00 | 7.06 | 26.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 2049 | 16/4 | 57.3 | 93.5 | 0.057 | PASS | N/A | 20.6 | 16.5 | 1.24 | 1.12 | 4.02 | 19.5% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 4096 | 32/8 | 56.8 | 93.5 | 0.070 | PASS | N/A | 28.2 | 24.6 | 1.14 | 1.00 | 7.06 | 25.1% | warm_l2 | 3/3 |
| RTX PRO 6000 | noncausal_sliding | 8 | 4 | 4096 | 16/4 | 56.8 | 92.4 | 0.083 | PASS | N/A | 20.6 | 16.5 | 1.25 | 1.12 | 4.02 | 19.5% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 128 | 32/8 | 54.4 | 63.8 | 0.008 | PASS | N/A | 16.4 | 14.2 | 1.15 | 1.01 | 1.43 | 8.7% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 128 | 16/4 | 59.2 | 74.8 | 0.009 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 1.23 | 8.5% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 512 | 32/8 | 58.8 | 75.4 | 0.008 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 2.56 | 15.6% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 512 | 16/4 | 58.7 | 75.1 | 0.052 | PASS | N/A | 15.5 | 12.7 | 1.22 | 1.03 | 1.69 | 10.9% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4095 | 32/8 | 58.6 | 75.5 | 0.062 | PASS | N/A | 28.7 | 26.7 | 1.08 | 0.98 | 13.06 | 45.5% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4095 | 16/4 | 58.9 | 75.4 | 0.014 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 6.95 | 33.8% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4096 | 32/8 | 59.0 | 75.6 | 0.037 | PASS | N/A | 28.8 | 26.6 | 1.08 | 1.00 | 13.07 | 45.4% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4096 | 16/4 | 58.6 | 75.4 | 0.061 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 6.95 | 33.8% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4097 | 32/8 | 58.5 | 75.4 | 0.039 | PASS | N/A | 28.8 | 26.6 | 1.08 | 1.00 | 13.07 | 45.5% | warm_l2 | 3/3 |
| GB10 | full | 16 | 1 | 4097 | 16/4 | 58.6 | 75.1 | 0.062 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 6.95 | 33.8% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 128 | 32/8 | 58.6 | 77.9 | 0.007 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 2.04 | 9.0% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 128 | 16/4 | 58.8 | 78.3 | 0.009 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 1.43 | 7.0% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 512 | 32/8 | 58.7 | 78.7 | 0.048 | PASS | N/A | 24.6 | 22.5 | 1.09 | 1.00 | 4.29 | 17.4% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 512 | 16/4 | 58.8 | 78.9 | 0.052 | PASS | N/A | 21.5 | 18.5 | 1.16 | 1.00 | 2.56 | 11.9% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4095 | 32/8 | 174.4 | 212.5 | 0.028 | PASS | N/A | 165.9 | 163.4 | 1.01 | 0.98 | 38.06 | 22.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4095 | 16/4 | 58.8 | 78.6 | 0.075 | PASS | N/A | 39.1 | 35.0 | 1.12 | 1.01 | 13.07 | 33.4% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4096 | 32/8 | 172.7 | 212.4 | 0.028 | PASS | N/A | 166.0 | 163.7 | 1.01 | 0.98 | 38.09 | 22.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4096 | 16/4 | 58.8 | 78.6 | 0.069 | PASS | N/A | 38.9 | 34.9 | 1.12 | 1.00 | 13.08 | 33.6% | warm_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4097 | 32/8 | 172.2 | 212.4 | 0.029 | PASS | N/A | 166.3 | 164.3 | 1.01 | 0.98 | 38.13 | 22.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 2 | 4097 | 16/4 | 58.7 | 78.6 | 0.065 | PASS | N/A | 38.9 | 34.9 | 1.12 | 1.00 | 13.08 | 33.6% | warm_l2 | 3/3 |
| GB10 | full | 16 | 3 | 128 | 32/8 | 58.8 | 80.8 | 0.036 | PASS | N/A | 26.7 | 23.9 | 1.12 | 1.03 | 2.65 | 9.9% | warm_l2 | 3/3 |
| GB10 | full | 16 | 3 | 128 | 16/4 | 58.7 | 80.9 | 0.004 | PASS | N/A | 21.7 | 19.4 | 1.12 | 1.00 | 1.73 | 8.0% | warm_l2 | 3/3 |
| GB10 | full | 16 | 3 | 512 | 32/8 | 58.8 | 80.7 | 0.019 | PASS | N/A | 30.8 | 27.1 | 1.14 | 1.00 | 6.03 | 19.6% | warm_l2 | 3/3 |
| GB10 | full | 16 | 3 | 512 | 16/4 | 58.6 | 80.7 | 0.043 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.43 | 15.2% | warm_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4095 | 32/8 | 263.5 | 300.7 | 0.013 | PASS | N/A | 252.9 | 245.1 | 1.03 | 1.00 | 108.35 | 42.8% | partial_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4095 | 16/4 | 107.9 | 140.8 | 0.011 | PASS | N/A | 101.2 | 94.8 | 1.07 | 0.98 | 18.82 | 18.6% | partial_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4096 | 32/8 | 262.9 | 301.5 | 0.021 | PASS | N/A | 252.6 | 245.1 | 1.03 | 1.00 | 108.40 | 42.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4096 | 16/4 | 108.6 | 140.1 | 0.042 | PASS | N/A | 101.2 | 94.0 | 1.08 | 0.99 | 18.82 | 18.6% | partial_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4097 | 32/8 | 260.9 | 301.9 | 0.025 | PASS | N/A | 252.8 | 245.2 | 1.03 | 1.00 | 108.45 | 42.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 3 | 4097 | 16/4 | 106.6 | 142.4 | 0.006 | PASS | N/A | 100.4 | 94.1 | 1.07 | 0.99 | 18.82 | 18.7% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 128 | 32/8 | 58.9 | 83.0 | 0.014 | PASS | N/A | 29.0 | 26.7 | 1.09 | 1.00 | 3.26 | 11.2% | warm_l2 | 3/3 |
| GB10 | full | 16 | 4 | 128 | 16/4 | 58.8 | 83.0 | 0.011 | PASS | N/A | 23.6 | 20.5 | 1.15 | 1.03 | 2.04 | 8.6% | warm_l2 | 3/3 |
| GB10 | full | 16 | 4 | 512 | 32/8 | 59.1 | 83.4 | 0.049 | PASS | N/A | 34.9 | 32.8 | 1.06 | 1.00 | 7.76 | 22.2% | warm_l2 | 3/3 |
| GB10 | full | 16 | 4 | 512 | 16/4 | 59.0 | 83.4 | 0.059 | PASS | N/A | 24.6 | 22.5 | 1.09 | 1.00 | 4.29 | 17.4% | warm_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4095 | 32/8 | 336.5 | 376.6 | 0.043 | PASS | N/A | 327.2 | 319.7 | 1.02 | 1.00 | 178.65 | 54.6% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4095 | 16/4 | 174.9 | 215.8 | 0.017 | PASS | N/A | 167.1 | 165.2 | 1.01 | 0.97 | 38.20 | 22.9% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4096 | 32/8 | 336.6 | 377.2 | 0.004 | PASS | N/A | 327.5 | 319.4 | 1.03 | 1.00 | 178.71 | 54.6% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4096 | 16/4 | 172.4 | 216.3 | 0.024 | PASS | N/A | 167.3 | 165.4 | 1.01 | 0.97 | 38.23 | 22.8% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4097 | 32/8 | 337.3 | 375.5 | 0.009 | PASS | N/A | 327.3 | 320.0 | 1.02 | 1.00 | 178.78 | 54.6% | partial_l2 | 3/3 |
| GB10 | full | 16 | 4 | 4097 | 16/4 | 173.0 | 216.1 | 0.030 | PASS | N/A | 167.3 | 165.3 | 1.01 | 0.98 | 38.26 | 22.9% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 128 | 32/8 | 58.6 | 74.6 | 0.006 | PASS | N/A | 16.4 | 14.4 | 1.14 | 1.00 | 1.43 | 8.7% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 128 | 16/4 | 58.8 | 75.0 | 0.005 | PASS | N/A | 14.6 | 12.4 | 1.17 | 1.03 | 1.23 | 8.4% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 512 | 32/8 | 58.8 | 75.5 | 0.007 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 2.56 | 15.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 512 | 16/4 | 59.0 | 75.2 | 0.072 | PASS | N/A | 16.4 | 13.3 | 1.23 | 1.02 | 1.69 | 10.3% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4095 | 32/8 | 59.1 | 75.6 | 0.086 | PASS | N/A | 28.7 | 26.7 | 1.08 | 0.99 | 13.06 | 45.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4095 | 16/4 | 59.4 | 75.4 | 0.022 | PASS | N/A | 21.9 | 18.5 | 1.18 | 1.00 | 6.95 | 31.8% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4096 | 32/8 | 58.6 | 75.6 | 0.068 | PASS | N/A | 28.7 | 26.7 | 1.08 | 0.99 | 13.06 | 45.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4096 | 16/4 | 59.1 | 75.5 | 0.061 | PASS | N/A | 21.9 | 18.5 | 1.19 | 1.00 | 6.95 | 31.7% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4097 | 32/8 | 58.9 | 75.7 | 0.068 | PASS | N/A | 28.7 | 26.7 | 1.08 | 0.99 | 13.06 | 45.4% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 1 | 4097 | 16/4 | 59.1 | 75.3 | 0.054 | PASS | N/A | 21.7 | 18.5 | 1.18 | 1.00 | 6.95 | 32.0% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 128 | 32/8 | 58.9 | 77.9 | 0.043 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 2.04 | 9.0% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 128 | 16/4 | 58.8 | 78.2 | 0.038 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 1.43 | 7.0% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 512 | 32/8 | 58.9 | 78.6 | 0.060 | PASS | N/A | 24.6 | 22.4 | 1.10 | 1.01 | 4.29 | 17.4% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 512 | 16/4 | 59.0 | 78.2 | 0.051 | PASS | N/A | 21.3 | 18.5 | 1.16 | 1.00 | 2.56 | 12.0% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4095 | 32/8 | 171.7 | 212.1 | 0.006 | PASS | N/A | 166.1 | 160.3 | 1.04 | 1.00 | 38.06 | 22.9% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4095 | 16/4 | 58.8 | 78.3 | 0.059 | PASS | N/A | 39.0 | 35.4 | 1.10 | 1.00 | 13.07 | 33.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4096 | 32/8 | 172.5 | 212.8 | 0.027 | PASS | N/A | 166.5 | 160.7 | 1.04 | 1.00 | 38.06 | 22.9% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4096 | 16/4 | 58.8 | 78.4 | 0.064 | PASS | N/A | 39.0 | 35.2 | 1.11 | 1.00 | 13.07 | 33.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4097 | 32/8 | 176.1 | 213.4 | 0.031 | PASS | N/A | 166.7 | 160.9 | 1.04 | 1.00 | 38.06 | 22.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 2 | 4097 | 16/4 | 58.7 | 78.4 | 0.074 | PASS | N/A | 39.0 | 35.4 | 1.10 | 1.00 | 13.07 | 33.5% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 128 | 32/8 | 59.0 | 80.8 | 0.043 | PASS | N/A | 26.7 | 24.0 | 1.11 | 1.03 | 2.65 | 9.9% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 128 | 16/4 | 59.1 | 80.8 | 0.030 | PASS | N/A | 21.6 | 19.3 | 1.12 | 1.00 | 1.73 | 8.0% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 512 | 32/8 | 59.2 | 81.3 | 0.063 | PASS | N/A | 30.8 | 28.2 | 1.09 | 0.99 | 6.03 | 19.6% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 512 | 16/4 | 59.2 | 80.9 | 0.065 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 3.43 | 15.2% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4095 | 32/8 | 263.1 | 302.1 | 0.033 | PASS | N/A | 253.4 | 246.3 | 1.03 | 1.00 | 108.35 | 42.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4095 | 16/4 | 108.7 | 141.7 | 0.041 | PASS | N/A | 100.2 | 89.3 | 1.12 | 1.05 | 18.82 | 18.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4096 | 32/8 | 263.1 | 302.0 | 0.025 | PASS | N/A | 253.2 | 246.1 | 1.03 | 1.00 | 108.35 | 42.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4096 | 16/4 | 107.5 | 142.4 | 0.014 | PASS | N/A | 100.3 | 89.4 | 1.12 | 1.05 | 18.82 | 18.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4097 | 32/8 | 259.5 | 301.5 | 0.018 | PASS | N/A | 252.6 | 245.7 | 1.03 | 1.00 | 108.35 | 42.9% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 3 | 4097 | 16/4 | 106.3 | 142.5 | 0.006 | PASS | N/A | 99.9 | 89.0 | 1.12 | 1.05 | 18.82 | 18.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 128 | 32/8 | 59.3 | 83.2 | 0.047 | PASS | N/A | 29.2 | 26.7 | 1.09 | 1.00 | 3.26 | 11.2% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 128 | 16/4 | 59.0 | 83.1 | 0.011 | PASS | N/A | 23.7 | 20.6 | 1.15 | 1.03 | 2.04 | 8.6% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 512 | 32/8 | 59.4 | 83.4 | 0.067 | PASS | N/A | 36.0 | 32.8 | 1.10 | 1.00 | 7.76 | 21.6% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 512 | 16/4 | 59.2 | 83.2 | 0.073 | PASS | N/A | 24.6 | 22.6 | 1.09 | 1.00 | 4.29 | 17.4% | warm_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4095 | 32/8 | 335.4 | 377.3 | 0.028 | PASS | N/A | 327.8 | 320.4 | 1.02 | 1.00 | 178.65 | 54.5% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4095 | 16/4 | 172.7 | 216.0 | 0.007 | PASS | N/A | 167.6 | 164.9 | 1.02 | 0.98 | 38.20 | 22.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4096 | 32/8 | 336.9 | 376.8 | 0.017 | PASS | N/A | 327.9 | 319.8 | 1.03 | 1.00 | 178.65 | 54.5% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4096 | 16/4 | 173.4 | 216.7 | 0.029 | PASS | N/A | 167.8 | 165.3 | 1.02 | 0.98 | 38.20 | 22.8% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4097 | 32/8 | 337.3 | 376.9 | 0.017 | PASS | N/A | 328.0 | 320.0 | 1.03 | 1.00 | 178.65 | 54.5% | partial_l2 | 3/3 |
| GB10 | causal_sliding | 16 | 4 | 4097 | 16/4 | 173.9 | 216.7 | 0.025 | PASS | N/A | 167.7 | 165.4 | 1.01 | 0.98 | 38.20 | 22.8% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 128 | 32/8 | 54.7 | 63.3 | 0.010 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 1.31 | 9.1% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 128 | 16/4 | 59.0 | 74.2 | 0.008 | PASS | N/A | 14.3 | 11.8 | 1.22 | 1.05 | 1.23 | 8.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 512 | 32/8 | 58.7 | 74.4 | 0.039 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 2.44 | 16.9% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 512 | 16/4 | 58.6 | 74.5 | 0.038 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 1.63 | 11.3% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2047 | 32/8 | 59.0 | 74.5 | 0.044 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 6.94 | 37.5% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2047 | 16/4 | 58.6 | 74.4 | 0.012 | PASS | N/A | 16.4 | 12.6 | 1.30 | 1.07 | 3.88 | 23.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2048 | 32/8 | 58.6 | 74.6 | 0.005 | PASS | N/A | 18.7 | 16.4 | 1.14 | 1.00 | 6.94 | 37.2% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2048 | 16/4 | 58.8 | 74.5 | 0.048 | PASS | N/A | 16.4 | 12.6 | 1.30 | 1.06 | 3.88 | 23.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2049 | 32/8 | 58.7 | 74.6 | 0.032 | PASS | N/A | 18.6 | 16.4 | 1.13 | 1.00 | 6.94 | 37.2% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 2049 | 16/4 | 59.0 | 74.6 | 0.054 | PASS | N/A | 16.4 | 12.6 | 1.30 | 1.06 | 3.88 | 23.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 4096 | 32/8 | 59.0 | 74.3 | 0.030 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 6.94 | 37.5% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 1 | 4096 | 16/4 | 59.3 | 74.5 | 0.010 | PASS | N/A | 16.1 | 12.4 | 1.30 | 1.06 | 3.88 | 24.1% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 128 | 32/8 | 58.4 | 77.4 | 0.009 | PASS | N/A | 15.8 | 13.4 | 1.18 | 1.05 | 1.80 | 11.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 128 | 16/4 | 58.7 | 77.4 | 0.012 | PASS | N/A | 14.4 | 12.3 | 1.17 | 1.00 | 1.31 | 9.1% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 512 | 32/8 | 58.7 | 77.6 | 0.012 | PASS | N/A | 16.5 | 14.4 | 1.15 | 1.00 | 4.06 | 24.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 512 | 16/4 | 58.6 | 77.9 | 0.050 | PASS | N/A | 15.9 | 12.3 | 1.29 | 1.05 | 2.44 | 15.3% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2047 | 32/8 | 58.9 | 77.5 | 0.081 | PASS | N/A | 26.7 | 24.6 | 1.09 | 1.00 | 13.06 | 48.9% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2047 | 16/4 | 58.6 | 77.6 | 0.041 | PASS | N/A | 19.4 | 16.4 | 1.18 | 1.00 | 6.95 | 35.9% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2048 | 32/8 | 58.6 | 77.6 | 0.065 | PASS | N/A | 26.7 | 24.6 | 1.09 | 1.00 | 13.06 | 48.9% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2048 | 16/4 | 59.2 | 77.5 | 0.057 | PASS | N/A | 19.3 | 16.4 | 1.18 | 1.00 | 6.95 | 36.0% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2049 | 32/8 | 58.5 | 77.4 | 0.045 | PASS | N/A | 26.7 | 24.6 | 1.08 | 1.00 | 13.06 | 48.9% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 2049 | 16/4 | 59.0 | 77.9 | 0.062 | PASS | N/A | 19.5 | 16.4 | 1.18 | 1.00 | 6.95 | 35.7% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 4096 | 32/8 | 59.1 | 77.5 | 0.054 | PASS | N/A | 28.7 | 26.7 | 1.08 | 0.93 | 13.06 | 45.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 2 | 4096 | 16/4 | 58.7 | 77.5 | 0.025 | PASS | N/A | 18.7 | 16.4 | 1.14 | 1.00 | 6.95 | 37.1% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 128 | 32/8 | 58.7 | 80.3 | 0.023 | PASS | N/A | 20.5 | 18.5 | 1.11 | 1.00 | 2.30 | 11.2% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 128 | 16/4 | 58.5 | 79.9 | 0.009 | PASS | N/A | 18.5 | 16.4 | 1.13 | 1.00 | 1.56 | 8.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 512 | 32/8 | 58.6 | 80.3 | 0.006 | PASS | N/A | 24.6 | 21.9 | 1.12 | 1.03 | 5.68 | 23.0% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 512 | 16/4 | 58.7 | 80.2 | 0.050 | PASS | N/A | 20.5 | 17.1 | 1.20 | 1.03 | 3.25 | 15.8% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2047 | 32/8 | 84.4 | 129.2 | 0.045 | PASS | N/A | 78.9 | 79.0 | 1.00 | 0.94 | 18.82 | 23.9% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2047 | 16/4 | 59.3 | 80.8 | 0.054 | PASS | N/A | 31.7 | 28.7 | 1.10 | 1.00 | 10.01 | 31.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2048 | 32/8 | 85.0 | 131.3 | 0.046 | PASS | N/A | 79.3 | 79.1 | 1.00 | 0.94 | 18.82 | 23.7% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2048 | 16/4 | 58.9 | 80.6 | 0.054 | PASS | N/A | 31.8 | 28.7 | 1.11 | 1.00 | 10.01 | 31.5% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2049 | 32/8 | 84.5 | 130.6 | 0.040 | PASS | N/A | 77.5 | 77.9 | 0.99 | 0.95 | 18.82 | 24.3% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 2049 | 16/4 | 58.5 | 80.2 | 0.065 | PASS | N/A | 31.6 | 28.7 | 1.10 | 1.00 | 10.01 | 31.6% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 4096 | 32/8 | 79.0 | 124.8 | 0.009 | PASS | N/A | 71.4 | 64.9 | 1.10 | 1.04 | 18.82 | 26.4% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 3 | 4096 | 16/4 | 58.6 | 80.7 | 0.023 | PASS | N/A | 31.3 | 28.7 | 1.09 | 1.00 | 10.01 | 32.0% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 128 | 32/8 | 58.8 | 82.9 | 0.026 | PASS | N/A | 22.6 | 20.5 | 1.10 | 1.00 | 2.79 | 12.3% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 128 | 16/4 | 58.9 | 82.5 | 0.012 | PASS | N/A | 20.5 | 17.7 | 1.16 | 1.04 | 1.81 | 8.8% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 512 | 32/8 | 58.9 | 83.1 | 0.011 | PASS | N/A | 28.7 | 25.8 | 1.11 | 1.03 | 7.29 | 25.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 512 | 16/4 | 58.5 | 82.9 | 0.050 | PASS | N/A | 20.7 | 18.5 | 1.12 | 1.00 | 4.06 | 19.7% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2047 | 32/8 | 164.4 | 206.1 | 0.019 | PASS | N/A | 158.0 | 154.4 | 1.02 | 0.98 | 38.03 | 24.1% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2047 | 16/4 | 58.7 | 83.1 | 0.063 | PASS | N/A | 36.9 | 33.1 | 1.12 | 1.01 | 13.07 | 35.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2048 | 32/8 | 165.4 | 206.3 | 0.027 | PASS | N/A | 158.3 | 154.4 | 1.03 | 0.99 | 38.03 | 24.0% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2048 | 16/4 | 58.5 | 83.3 | 0.068 | PASS | N/A | 36.9 | 33.1 | 1.12 | 1.02 | 13.07 | 35.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2049 | 32/8 | 167.0 | 206.4 | 0.026 | PASS | N/A | 158.5 | 154.5 | 1.03 | 0.99 | 38.03 | 24.0% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 2049 | 16/4 | 58.7 | 83.0 | 0.054 | PASS | N/A | 37.0 | 33.2 | 1.11 | 1.01 | 13.07 | 35.4% | warm_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 4096 | 32/8 | 160.7 | 204.3 | 0.028 | PASS | N/A | 157.2 | 144.1 | 1.09 | 1.05 | 38.03 | 24.2% | partial_l2 | 3/3 |
| GB10 | noncausal_sliding | 8 | 4 | 4096 | 16/4 | 59.1 | 83.5 | 0.046 | PASS | N/A | 39.0 | 34.9 | 1.12 | 1.04 | 13.07 | 33.5% | warm_l2 | 3/3 |

## Round 1 (source tree, 769bcaf)

RTX 5090 and RTX PRO 6000: 135 passed each (PRO 6000 extended Q2/8/9/16/32: 321 passed);
128/128 parity rows per device; complete op 1.11–1.66× the fixed-plan floor with 7 kernels
per call. Superseded by round 2.
