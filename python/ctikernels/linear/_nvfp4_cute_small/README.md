# Experimental small-M CuTe provider

Standalone provider for DSL 4.8.0, RTX 5090 / RTX PRO 6000 (sm120), and GB10
(sm121), M=1..32. It is experimental and absent from the dispatch tables.
RTX 5090 development measurements are recorded below; complete device and
numerical qualification remains pending.

## Status (2026-10-14)

The RTX 5090 9b measurements in `cute_results.txt` cover successive development
versions, not a qualified dispatch candidate. The initial implementation used
per-row bulk copies. TMA B/SFB loads and an all-warp A-quantization prologue then
reached `load_only=19.0 us` for linear M=1: about 97% of the measured copy BW.
The full-kernel versus `no_reduce` probes exposed roughly 5–10 us of split-K
reduction/epilogue cost. This was not primarily an MMA throughput limit.

| Version | 9b linear M=1 | M=8 | Interpretation |
| --- | --- | --- | --- |
| TMA + all-warp prologue (RTX 5090) | `load_only` 19.0 us; full 24.3 us for the recorded N32/Ktile512/stages3/split8 configuration | Not in this probe breakdown | Weight loading approaches the copy ceiling; reduction adds cost. |
| Split-one / cluster development (RTX 5090) | Best 21.4 us versus SGLang 21.0 us | Best 29.9 us versus SGLang 21.6 us | Near parity at M=1 did not extend to M=8. |
| Consumer JIT A quantization (RTX 5090) | 28.4–46.8 us (approximately 28–47 us) | 50.6–93.5 us | Consumers load A from L2 and quantize it in the mainloop, delaying MMA and B-ring consumption. |
| A-ring quantizer warps, first version (RTX PRO 6000) | 46.3 us (q1) / 27.7 us (q2+PDL), versus SGLang 23.5 us | 327 / 182 us, versus SGLang 24.1 us | Loads serialized. |
| Batched one-lane-per-block quantizer: unmeasured | Unmeasured | Unmeasured | Unconditional vector load batches and lane-local quantization. |

The consumer JIT revision regressed. The first A-ring revision compiled and passed
353 GPU tests on RTX 5090, but RTX PRO 6000 timings exposed serialized quantizer
loads. This revision batches unconditional loads ahead of lane-local quantization
into the same independent shared-memory A ring. Flat M1..8 latency and copy
bandwidth remain measurement targets, not established results. The `stamps`
probe is removed because the earlier revision failed NVVM compilation; it does
not establish a measured phase timeline.

```python
from ctikernels.linear import _nvfp4_cute_small as cute_small

cfg = cute_small.Config(32, 512, 2, 4, 3, 1024, reduce_mode="cluster")
w = cute_small.prepare(weight)
work = cute_small.Workspace(w, x.shape[0], cfg)
y = cute_small.linear(x, w, input_scale, cfg, workspace=work)
```

`Workspace(..., outputs=False)` holds no output storage, so every call binds `out`;
`scratch=(counters, partials)` takes zeroed, self-resetting split-K storage of at least
`Workspace.scratch_shapes(...)` elements from a caller (e.g. a `GraphWorkspace` slot).

Prepare weights/workspaces and prewarm outside capture. Each warmed call launches
one kernel. Calls and graph replays sharing a workspace must be serialized. Global
scales retain caller-owned device storage and are read every call/replay. Weight
fields and scales are checked for grad requirements before any launch, including
changes after preparation. Kernels contain no FP64; address offsets use int64.
Inputs require a 16-byte-aligned base and a row stride divisible by eight elements.
The standalone provider rejects other alignments. `NVFP4Linear` and
`NVFP4GateUpSiLU` select a non-CuTe plan for that call using pointer/stride metadata,
including when `provider="cute"` was requested. No host synchronization is needed.

## Mainloop and reduction

A CTA handles up to 16 real M rows, one N tile, and a contiguous split of K. M17..32
uses two CTA rows and reads weights twice. A/SFA are quantized into shared memory;
only B/SFB use the transaction-counted TMA ring. A uses 16-byte sectors across rows,
SFA is K64-major, and prepared SFB packs eight rows of four scales into each sector.
Tensor-map bounds zero-fill N/K tails. Each producer stage issues two TMA copies,
with full barriers and consumer-counted empty barriers. The last use of a stage
needs no empty arrival because no refill follows it.

There are three warp roles. The legacy `warps` field still includes the producer:
`C = warps - 1` consumers occupy warp IDs 0..C-1, `Q = qwarps` quantizers occupy
C..C+Q-1, and the last warp produces B/SFB. Total launch size is `(warps+qwarps)*32`.
Consumers never quantize or load global activations in their mainloop. There is
no named consumer barrier.

Each of `a_stages` A slots holds `8*Ktile` packed bytes and `Ktile` SFA bytes in
the existing sector and K64-major layouts. Padded rows are zeroed once before the
role split. Quantizers distribute real rows and K16 blocks across all Q warps,
with one lane per block and no quantization shuffles. Each lane owns
`ceil(min(M,16)*(Ktile/16)/(32*Q))` blocks, processed in batches of at most eight.
All loads of a batch precede its conversions/math: two unconditional 16-byte
`ld.global.v4.b32` loads retain eight packed registers per block, at most 64 per
batch. Load addresses clamp to the last live row and last full K16 block; separate
validity predicates select zero values after conversion. There is no runtime
alignment branch. Padded rows skip stores, while K-tail blocks in live rows
quantize zero values with the existing scale arithmetic.

Each block reduces its NaN-ignoring amax locally (all-NaN stays NaN), computes
its scale with the shared market recipe in `_nvfp4_cute_recipe.py`, and uses
hardware RN-even E2M1 pair conversion directly on FTZ products. No nonfinite
flag, finite-factor cap, or NaN/signed-zero substitution remains. The recipe
follows FlashInfer 0.7.0.post1 `quantization_utils.cuh` as a design reference
only (Apache-2.0). Each block
writes one aligned eight-byte shared vector and its scale byte. Debug data/scales
keep their original layouts. Total register usage and generated load scheduling
still require device measurement. Consumer stage/lane address terms are hoisted
out of the K64 loop; MMA fragments and synchronization remain unchanged.

`a_full[slot]` starts with Q arrivals. Every quantizer warp finishes its stores,
executes `sync_warp`, then lane zero arrives with release semantics. Consumers
acquire-wait on both B full and A full before reading either fragment. After the
last K64 group, every consumer lane arrives on the relevant empty barrier if
that slot will be reused. Both empty rings count C*32 arrivals.

For either ring of depth D, consumer step s waits on full parity `(s//D)%2`.
The writer skips the empty wait on its first D steps, then waits on empty parity
`(s//D-1)%2` before refill. Empty arrivals occur only when `s+D<count`.
Thus short/empty splits have no waits for nonexistent refills, and A and B can
wrap independently. The producer starts B/SFB immediately after initialization;
it never waits for A. Every role reaches the post-mainloop CTA join, after which
the combined A/SFA allocation is reused as FP32 result scratch. The kernel ignores
`max_slice_k`; shared storage is independent of total K and split length.

With `pdl=True`, the DSL launch uses `use_pdl=True`. All non-producer threads
execute `griddepcontrol_wait()` before reading x/scales or writing debug data.
The producer may prefetch immutable prepared B/SFB first, then executes the wait
after its loop and before the CTA join or any global epilogue access. Its timeout
path also waits before writing diagnostics. After the join, thread zero issues
`griddepcontrol_launch_dependents()` once per CTA before the epilogue.

This is safe for serialized graph replays: early launch is only a scheduling hint;
a dependent kernel must wait for predecessor completion before consuming outputs
or overwriting storage that the predecessor still reads (WAR). In this kernel all
writes to output, partials, counters and diagnostics are ordered after the wait,
including writes by the producer during reduction. Only immutable B/SFB reads and
CTA-private initialization can precede it. Workspace sharing still requires
serialized calls/replays; mutable x and scale updates belong on the ordered stream.
These ordering rules follow the [public PTX grid dependency instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol).

The gated epilogue uses all CTA threads over flattened real rows and columns;
groups of 16 lanes compute SiLU*up, reduce NaN-ignoring FP32 amax, quantize, and pack
two output words. Reduction protocols and MMA fragments are unchanged.

- **split_k=1:** no global partial storage, counters, reduction atomics, or reduction
  fences. The owning CTA writes BF16 output, or performs FP32 SiLU(gate)*up followed
  by NVFP4 output quantization. Only real rows enter the epilogue.
- **reduce_mode="global":** partials are `[N tile, split, M, n_tile]`, with no
  padded M rows. All CTA threads store/load aligned FP32 vectors of four elements.
  A CTA join orders every writer before the leader's GPU acquire-release atomic
  arrival. The last arrival acquires the RMW chain and broadcasts through a CTA
  join; all threads sum splits in fixed ascending order. The counter is reset with
  a release exchange after the epilogue. There is no separate GPU fence.
- **reduce_mode="cluster":** launch shape `(1, split_k, 1)` groups the K splits.
  Each CTA publishes only real accumulator rows in its own shared memory, reusing
  A storage. A release/acquire cluster barrier publishes these rows. Rank zero's
  threads use `mapa.shared::cluster` and vector DSMEM loads, summing ranks in
  ascending order. A final cluster barrier keeps every peer alive through the
  epilogue. There are no global partials or counters, including diagnostic modes.

Clusters accept only 2/4/8 CTAs. NVIDIA documents eight as the portable limit;
nonportable maxima require a device/kernel occupancy query. The documented B200
16-CTA opt-in is not assumed for RTX 5090 / RTX PRO 6000 / GB10. No nonportable
size is enabled. Actual occupancy on the target devices remains unmeasured.
See the [cluster programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-block-clusters),
[Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html#thread-block-clusters),
and [public PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
No third-party implementation was copied or adapted.

## Configuration and sweeps

`Config(N, Ktile, stages, splits, warps, max_slice_k=3072, ctas_per_sm=1,
probe=None, reduce_mode="global", qwarps=1, a_stages=4, pdl=False)` supports
N16/32/64/128, Ktile128/256/512/1024, stages1..8, warps2/3/5, qwarps1..4 and
a_stages2..8, with `(warps+qwarps)*32 <= 256`. Consumers must divide N/8; gated tiles preserve pairs and require
`N/2 % 16 == 0`. N96 is omitted because the existing tile/ownership layout only
represents the listed N values. Shared budget remains 99/49/32/24 KiB for residency
targets 1/2/3/4. Actual occupancy and register spilling need device measurement.

Shared bytes are bounded by
`max(a_stages*9*Ktile,64*N) + stages*N*Ktile*9/16 + 16*(stages+a_stages) + 16`.
The first term is one allocation for packed A, SFA, and later result scratch.
Both public `Config` and the kernel validate this budget. The final 16 bytes cover
the four-byte result/abort flag with conservative alignment headroom.

Public `sweep` and the dev script share one candidate source:

| Shape | N | K |
| --- | ---: | ---: |
| 9b-linear | 4096 | 12288 |
| 9b-gated | 24576 | 4096 |
| 27b-linear | 5120 | 17408 |
| 27b-gated | 34816 | 5120 |
| 35b-shared-linear | 2048 | 512 |
| 35b-shared-gated | 1024 | 2048 |

Gated N counts both interleaved projections. Each 9B/27B sweep has at most 24
candidates: four families crossed with qwarps1/2/4 and a_stages4/6.

- RTX 5090 (170 SMs) / RTX PRO 6000 (188 SMs) 9B/27B linear: split-one
  N64/Ktile512/stages4 and N32/Ktile256/stages4, plus N32 cluster2 at
  Ktile256/stages4 and cluster4 at Ktile512/stages3.
- GB10 (48 SMs) linear: the same four families, all at Ktile256.
- 9B/27B gated: split-one N64/stages4 and N128/stages3 at residency1,
  N64/stages5 and N128/stages4 at residency2.
  Choose the largest fitting Ktile512/256/128 on RTX 5090 / RTX PRO 6000,
  or Ktile256/128 on GB10. Omit combinations that cannot fit Ktile128.
  N128 uses warps5 for qwarps1/2 and warps3 for qwarps4 to stay within 256 threads.
- 35B-shared: cluster splits2/4/8, N16/N32 linear or N32/N64 gated,
  Ktile128, stages2, warps3, qwarps1/2, a_stages4 (12 candidates). Residency targets remain three CTAs per SM
  on RTX 5090 / RTX PRO 6000 and two on GB10.

`presets` selects s0/s3/s4 for 9B/27B and s0/s5/s10 for 35B-shared. These are
unmeasured candidates. The legacy report field `a_slice_k` and `slice_k(k)` now
report `a_stages*Ktile`; `max_slice_k` remains an ignored integer compatibility
field. Config strings from earlier revisions retain their positional meaning.
`smem_bytes(k)` is independent of total K and split count. Dispatch plans carry the
three new fields while retaining defaults for older tables.

## Probes and development

`Config(..., probe=...)` selects a separately compiled diagnostic variant:

| Probe | Behavior |
| --- | --- |
| load_only | B/SFB ring only; quantizer warps idle; records consumed K-stage count per CTA in `probe_counts`. No partials are written. |
| no_prologue | Zero all A/SFA slots once; quantizers skip quantization but still publish full barriers. |
| no_mma | Keep operand loads live through a bounded checksum instead of MMA. |
| no_reduce | Mainloop and local publication; global mode also writes compact partials. Skip all reduction, counters, cluster barriers, and epilogue. |

Load-only/no-reduce leave outputs untouched. Timing and occupancy estimates are
reported only after CUDA events complete; requested residency is not measured
cluster occupancy.

`CTI_CUTE_SMALL_DEBUG_TIMEOUT=1` bounds each mbarrier wait to 2^26 attempts.
Kernel failures record diagnostics and trap, invalidating the device context:
local exits could strand the CTA join or invalidate cluster
DSMEM. The standalone `debug_wait` helper retains whole-warp exit by default.
Discard the workspace/context after a kernel failure. Debug
checks synchronize only outside capture; production calls have no host sync.

The dev script selects all shapes above and M1..32. It uses fresh subprocesses,
>512 MiB weight rotation, graph timing, FP64 correctness, and optional SGLang.
`--config-start` / `--config-count` partition sweeps into bounded invocations.
Sweep counts are queried in an owned child for the actual device, so partitioning
does not depend on obsolete fixed counts. Explicit configurations accept
`N:Ktile:stages:split:warps[:max_slice_k[:ctas_per_sm]][:global|cluster][:qN][:aN][:pdl]`,
for example `64:512:4:1:3:6144:1:q2:a4:pdl`. Trailing options may be reordered;
duplicates and invalid fields are rejected. Old strings still parse.
A 740-second parent budget and three cases at 240 seconds each leave headroom toward
850 seconds per runner call. Runner setup is external to this budget; any exhausted
budget or timeout is an incomplete measurement, not a passing sweep. Only owned
child processes are terminated. `CUTE_COMPILED` is emitted before the first launch.

```bash
python -m py_compile python/ctikernels/linear/_nvfp4_cute_small/{__init__,kernel}.py tests/gpu/test_nvfp4_cute_small.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest -q tests/gpu/test_nvfp4_cute_small.py tests/modules/test_nvfp4.py
```

CPU tests check lazy imports, frozen weights, compact scales, scratch sizes,
public shape/device sweep coverage and budgets, old/new config strings, ignored legacy limits, cluster validation, and the isolated
Python kernel constructor's A-ring budget without importing the DSL. GPU tests cover split-one,
A/B ring wraps, global and cluster2/4/8 reduction, empty splits/tails, bit-exact
activation/output quantization, FP64 linear/gated oracles, repeated identical graph
replays, PDL on/off identity, A-ring wrap/short splits at qwarps1/2 and a_stages2/4,
and in-place x/global-scale updates at M1/2/3/5/8/16/17/32. New batch tests cover
M1/3/8/16/17/32, qwarps1/2/4, BF16/FP16, K1056 tails at Ktile256, and multiple
load batches at Ktile1024. CPU tests cover alignment rejection and per-call fallback
for both public operators, including forced CuTe. Probes cover
all three reduction paths. GPU tests skip locally; no CUDA/JIT builds run locally.
