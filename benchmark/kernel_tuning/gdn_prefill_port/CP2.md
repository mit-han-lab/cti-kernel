# GP-CP2 fused port: raw TMA and in-place transforms

`PortPrefill` launches the fused chunk kernel followed by the unchanged raw-history
commit kernel. Defaults remain **f16nvs / producer 72 / math 216 / PF2**.
CP1's standalone default and math implementation are unchanged. PF now groups
only the z preloads; it does not control K/Q/V traffic or the number of z rows
loaded before O. Existing producer budgets and PF2/4/8 remain accepted.

## Status

Measured RTX 5090 baseline for the previous row-load producer: **1577 us**
at P72/PF4, versus **537 us** for bare CP1, with 16x48, `[4096,4096]`, f16nvs.
Producer K/Q/V were 3.3/3.4/2.9 us per warp/chunk and math waited 5.6 us for
K/gates, 6.1 us for Q and 3.2 us for V. Those measurements describe the previous
implementation; there are **no GPU latency or correctness results for this rewrite**.

Offline validation: Ruff check/format passes; the GDN CPU suite reports
**1833 passed, 2 skipped**. All **57 GPU cases collect and skip**
locally. Existing accuracy, graph, padding, varlen, history and tuning cases remain;
the padded-row accuracy cases now use a TMA-compatible stride/base. Two GPU tests
and two CPU tests reject misaligned projected bases and strides at call time.
The 5% error margin relative to matching-precision CP1 remains unchanged.

The final offline matrix covers base/f16nvs, P56/P72, 16x32/16x48 at PF2.
Additional P72 base/f16nvs IKET builds retain all 24 ranges, and standalone CP1
base/f16nvs builds pass their existing instruction checks for both geometries.
PTX checks verify raw TMA loads, tx arrivals, proxy fences, 128-thread barriers,
8-byte global/shared copies, packed BF16 conversion and register-allocation bounds.

## Design as built

- One TMA descriptor addresses raw projected `[T, logical_width]`, with physical
  row stride `P*2` bytes. Host validation requires **P % 8 == 0** and a **16-byte
  aligned base**, before compilation, allocation or launch. Invalid inputs raise
  a clear `ValueError`. The offline fake tensor promises the same alignment.
  Aligned padded rows and aligned nonzero storage offsets remain supported.
- The descriptor uses CP1's **K_SW128** layout and logical **64x128 BF16** tile.
  The source origin is `(tok_offset + 64*chunk, region_offset + head*128)`.
  It writes directly into sQ, one of two sK stages, or sV. As in CP1, CuTe lowers
  one logical tile copy to **two 8 KiB transfers**, each covering 64 columns,
  sharing one **16384-byte transaction barrier**. There is no raw scratch tile.
- Four extra raw-completion mbarriers cover Q, K0, K1 and V. Warp 0 elects one
  issuer, waits for the existing math-release/empty barrier, executes
  `fence.proxy.async.shared::cta`, arms the raw tx count and issues the copy.
  K0 is primed before the loop; a separate issue cursor loads K(c+1) before
  transforming K(c), as soon as that K stage is released. Q/V are issued after
  publishing K(c), before the gate scan, so their transfers overlap gates.
- Warp `w` owns rows `[16w, 16w+16)`, and lane `l` owns four adjacent columns.
  Warp 0 preloads three prior raw rows from projected, initial conv history or
  zeros before waiting for TMA. All warps also preload the current tile's FP32
  convolution taps. After raw completion, warps 1–3 read their three preceding
  rows from the raw shared tile into FP32 history registers. Named barrier **11**
  gathers all **128 producer threads** before any row is overwritten.
- Each warp walks its rows top-down: one 8-byte LDS, convolution with FP32 taps
  and history, non-FTZ SiLU, q/k L2 normalization with five xor shuffles, then
  packed BF16 and one 8-byte STS to the same row. The history shifts using the
  original raw row. Invalid output rows and slot -1 are explicitly zeroed.
  TMA zero-fills rows beyond total T; raw rows crossing a sequence end affect
  only masked rows, never preceding valid rows.
- The original `PipelineAsync` full barriers are **transformed-full** barriers:
  all 128 producers arrive after their generic shared writes, and the math
  warpgroups wait on those barriers before ldmatrix. The raw tx barrier is never
  used as the math-ready signal. Math releases and math operations are unchanged.
  Raw phase is `producer.phase ^ 1`, since tx barriers start at phase zero while
  the empty-wait producer state starts at phase one.
- TMA completion observation orders async writes before generic reads. The
  transformed-full release/acquire orders generic stores before math reads.
  The issuer's proxy fence after the empty acquire orders reuse of a buffer
  previously written through the generic proxy. These follow NVIDIA's
  [PTX async-proxy rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-copying-async-proxy).
- All four warps preload **all 16 z rows** (32 packed registers per lane) before
  waiting for O. PF only groups those 8-byte load instructions. A constant-index
  register FIFO feeds a single-row epilogue loop, avoiding dynamic register-array
  indexing and sixteen copies of the arithmetic. BF16 O is expanded for FP32
  RMS norm and `weight * o * inv * silu(z)`, then packed back to the same O tile.
  All producers fence and rendezvous before warp 1 uses the existing output TMA
  store and releases O. The full/tail output descriptor protocol is unchanged.
- Gate arithmetic, scan and two stages are unchanged. T/N/S remain symbolic;
  geometry, physical row stride, precision, budget and z grouping specialize the
  cache. Warm each specialization before graph capture. Supplied-output calls
  allocate no device storage or host synchronization. Cross-stream calls remain
  ordered by the caller; slot -1 touches no pools; the separate history launch
  retains short-sequence overlap handling and exact raw BF16 history bits.

## Compile matrix (CuTe DSL 4.8.0, sm_120a)

All entries use PF2, 168 launch registers and **100608 bytes** dynamic shared
memory. The four raw barriers fit existing alignment padding. The budget rule is
`128*p + 256*m <= 168*384 = 64512`; 56/224 and 72/216 both exactly fit.

Stack and spill bytes are **whole-kernel ptxas diagnostics**. LDL/STL are static
SASS instruction sites, not dynamic traffic or per-warp spill bytes. Producer
covers DEALLOC through TRY_ALLOC; math + tail starts at TRY_ALLOC. Prelude sites
are excluded from these region columns and recorded separately in resources.json.

| Precision | Geometry | Producer/math | Stack B | Spill store/load B | Producer LDL/STL | Math + tail LDL/STL |
|---|---|---:|---:|---:|---:|---:|
| base | 16x32 | 56/224 | 248 | 1412/1312 | 99/162 | 202/102 |
| base | 16x48 | 56/224 | 240 | 1372/1296 | 97/155 | 200/100 |
| base | 16x32 | 72/216 | 168 | 1244/1276 | 71/103 | 215/110 |
| base | 16x48 | 72/216 | 168 | 1244/1276 | 71/103 | 215/110 |
| f16nvs | 16x32 | 56/224 | 248 | 1284/1072 | 99/162 | 138/86 |
| f16nvs | 16x48 | 56/224 | 240 | 1244/1056 | 97/155 | 136/84 |
| f16nvs | 16x32 | 72/216 | 168 | 996/908 | 71/103 | 129/80 |
| f16nvs | 16x48 | 72/216 | 168 | 996/908 | 71/103 | 129/80 |

**Neither budget is spill-free.** P72 has 71 producer LDL and 103 producer STL
sites in these builds. All-z preloading and the expanded producer state retain
register pressure; this rewrite establishes the TMA data path, not a demonstrated
speedup. Compiler scheduling also changes math spills despite unchanged math
source. Benchmark both budgets rather than inferring latency from static counts.
Every history build uses 16 registers with zero stack, spills or shared memory.
IKET affects scheduling and resources; use uninstrumented builds for timing.

Final evidence is under `.worklog/gdn-cp2-tma/final/`, `iket/` and `cp1/`:
PTX, cubins, ptxas logs, SASS, resource dumps and `resources.json`. The `probe`,
`matrix`, `z-batches` and `z-fifo` directories are exploratory builds; use `final`
for the table above. All artifacts and caches are inside the checkout.

## Raw TMA hang investigation and watchdog

The first-launch hang at `30f2329` has a concrete protocol defect:
`issue_raw()` enclosed `cute.copy()` in `cute.arch.elect_one()`. CuTe DSL 4.8.0's
`CopyBulkTensorTileG2SOp` documentation explicitly warns against this because
TMA copies elect internally. The old PTX contains an outer election enclosing
both internal elections. The fix elects only the barrier arrival, then
reconverges warp 0 before calling `cute.copy()`. This is a suspected cause of
the observed hang; the fix has been checked offline, not on the RTX 5090.

Protocol audit:

- Each raw issue partitions into two 64-row x 64-column BF16 copies, 8,192 bytes
  each. `expect_tx` is 16,384 bytes. Compilation prints `GDN_RAW_TMA` with the
  copy count, total bytes and partition shapes, and asserts these sizes. K
  priming, K lookahead, Q and V produce eight static TMA instruction sites.
- All four raw barriers initialize with arrival count one and phase zero,
  before the initialization fence and CTA rendezvous. Q and V each have one
  stage; K uses two. Pipeline producers start at phase one, so raw completion
  waits use `producer.phase ^ 1`. K's independent issue cursor leads its
  transformation cursor by one chunk; each cursor toggles after two chunks.
- Warp 0 acquires the math-release barrier before reuse. All its lanes execute
  the async shared-memory proxy fence; one lane arrives with `expect_tx`.
  The warp rendezvous precedes CuTe's internal elections. All four producer
  warps wait for raw completion, read their boundary history, rendezvous before
  in-place writes, then commit 128 arrivals to the full pipeline barrier.
- The descriptor covers the projected `[T, P]` BF16 view with physical stride
  `[row_stride, 1]`, a 16-byte-aligned base and 16-byte-multiple row stride.
  Q/K/V share the same `K_SW128` layout and 64x128 logical tile. The two boxes
  have column origins `col` and `col + 64`; their shared destinations differ
  by 8,192 bytes. Tail rows are masked during transformation; raw reads across
  a packed sequence boundary cannot contribute to live causal rows.

`PortPrefill(..., watchdog=True)` is a separate compile/cache specialization.
It uses `mbarrier.try_wait.parity` with a two-second `%globaltimer` budget for
all pipeline acquire/wait/tail paths and raw waits. Ordered math barriers use
debug-only 256-arrival mbarriers with per-thread parity. On expiry, one lane
prints `GDN_WATCHDOG tag=... blk=... warp=... lane=... stage=... parity=... bx=...`
and all 12 warp progress words, then traps. Default builds emit no watchdog
timer, printf, progress access, or extra barrier initialization.

Progress words are the last reached marker, encoded as `(blk + 1) * 256 + tag`;
zero means no marker yet. Warps 0–3 are producers, 4–7 KK math, 8–11 QK math.

| Tags | Meaning |
| --- | --- |
| 10, 11, 12, 13, 14, 15 | Acquire empty Q, K, V, O, alpha, beta |
| 20, 21, 22, 23, 24, 25 | Wait full Q, K, V, O, alpha, beta |
| 30, 31, 32 | Raw Q, K, V completion |
| 40, 41 | Ordered KK, QK math wait |
| 43 | Producer peers waiting for output TMA store completion |
| 50, 51, 52 | Before producer Q, K, V boundary-read rendezvous |
| 53, 54, 55 | Before gate scan, gate publication, output norm rendezvous |
| 60, 61, 62, 64, 65 | Q, K, V, alpha, beta producer tails |

Named `bar.sync` barriers cannot be polled; every producer named rendezvous
has a progress marker. CTA initialization, intra-warp synchronization and
intra-math-group named barriers retain their hardware waits. Output TMA's
bulk-group wait also has no polling form: in watchdog builds, the other three
producer warps monitor a completion mbarrier signaled by the store warp. This
covers the final store too. Watchdog mode changes synchronization and register
pressure and is for diagnosis, not timing.

Run each case eagerly on an RTX 5090 (no graphs, no timing):

```bash
PYTHONPATH=python:. python benchmark/kernel_tuning/gdn_prefill_port/cp2_debug.py --lengths '[64]' --check
```

Replace `[64]` with `[128]`, `[16]`, or `[100]`. Omit `--lengths` to run those
four batches sequentially, or repeat it to select batches. `--check` compares
y and recurrent state against CP1 plus FP32 torch glue (2% relative RMS with
absolute floors of 1e-4 and 1e-5), and checks history and untouched slots exactly.
Synchronization is inside the exception handler; stdout/stderr are flushed
and the first trap/error exits nonzero without another CUDA launch.

Offline watchdog compile: add `--watchdog` to `compile_gdn_chunk_ws.py --fused`.
The compile harness checks both raw-copy reconvergence and emitted wait paths.
Artifacts for this investigation are under `.worklog/gdn-watchdog/`.
Validation: Ruff lint/format and whitespace checks pass; CPU GDN tests report
1,844 passed and two skipped. The default `f16nvs`, producer budget 72,
prefetch 2 kernel compiles for both 16x32 and 16x48, with watchdog on and off.
All four builds pass the PTX protocol checks. No GPU launches were performed.

## Offline reproduction

```bash
export PYTHONPATH="$PWD/python:$PWD"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p .worklog/gdn-cp2-tma/tmp
export TMPDIR="$PWD/.worklog/gdn-cp2-tma/tmp"
export XDG_CACHE_HOME="$PWD/.worklog/gdn-cp2-tma/xdg-cache"
export CUDA_CACHE_PATH="$PWD/.worklog/gdn-cp2-tma/cuda-cache"
export CUTE_DSL_CACHE_DIR="$PWD/.worklog/gdn-cp2-tma/cache"
CP2_PY=<engine env>/bin/python

$CP2_PY -m ruff check --no-cache python/ctikernels/gdn/chunk_ws scripts/compile_gdn_chunk_ws.py benchmark/kernel_tuning/gdn_prefill_port/cp2_iket.py tests/modules/test_gdn_chunk_ws.py tests/gpu/test_gdn_port_prefill.py
$CP2_PY -m pytest tests/modules/test_gdn*.py -q --basetemp=.worklog/gdn-cp2-tma/pytest-final
$CP2_PY -m pytest tests/gpu/test_gdn_port_prefill.py -q --basetemp=.worklog/gdn-cp2-tma/pytest-gpu
$CP2_PY scripts/compile_gdn_chunk_ws.py --fused --variants base,f16nvs --producer-registers 56,72 --prefetch 2 --output .worklog/gdn-cp2-tma/final
$CP2_PY scripts/compile_gdn_chunk_ws.py --fused --iket --variants base,f16nvs --producer-registers 72 --prefetch 2 --output .worklog/gdn-cp2-tma/iket
$CP2_PY scripts/compile_gdn_chunk_ws.py --variants base,f16nvs --output .worklog/gdn-cp2-tma/cp1
```

## RTX 5090 commands

These commands have **not** been run during offline validation. Run correctness
first, then compare the fused whole graph (including history commit) against bare
CP1 with preparation outside timing. Explicit budgets avoid a redundant K/Q/V
prefetch sweep. They need CuTe DSL 4.8.0 and FlashInfer 0.7.0.post1.

```bash
export PYTHONPATH="$PWD/python:$PWD"
DIR=benchmark/kernel_tuning/gdn_prefill_port

python -m pytest -q tests/gpu/test_gdn_port_prefill.py

for variant in base f16nvs; do
  for budget in 56 72; do
    python "$DIR/cp2_kernel.py" --hk 16 --hv 48 --lengths 4096,4096 --pairs 30 \
      --variant "$variant" --producer-registers "$budget" --prefetch 2 --with-fi
  done
done

for budget in 56 72; do
  python "$DIR/cp2_iket_node.py" --hk 16 --hv 48 --lengths 1024 --variant f16nvs \
    --producer-registers "$budget" --prefetch 2
done
```

## IKET interpretation

The 24 names include all nine unchanged math ranges. `producer.k/q/v` measure
only the in-place transform (including boundary shared reads and rendezvous),
once per producer warp/chunk. Tap loads and warp-0 global history preloads occur
before `producer.wait_raw_k/q/v` and outside transform ranges. Raw waits measure
TMA completion separately from math-release waits `producer.wait_k/q/v`.

`producer.tma_issue` occurs three times per chunk on the issuing warp; K priming
plus the lookahead together give one K issue per chunk. `producer.wait_k/q/v`
are also issuer-warp ranges. Gates, y, raw waits and output waits involve all four
producer warps. Y retains two ranges, preload and compute; wait_y is the output
rendezvous. The summarizer detects TMA traces and uses the new multiplicities,
while retaining support for older row-load traces. Aggregate wait fractions mix
these different participation counts and are sampled warp time, not utilization.

Nested `math.wait_v` is excluded from its enclosing denominator a second time.
`cp2_iket_node.py` traces CTA 0: `[1024]` covers 16 chunks of one sequence/head.
Compare raw waits, transforms, math waits and the y epilogue against the historical
CP1 target, using the same precision and budget in tracing and timing.

## GP-CP3: N=1 CP extension

The full fused path remains the baseline (`cp=False`). The default N=1 planner
compares `C=ceil(T/64)` against A/C CTA waves and a serial fixup estimate, with
provisional chunk-unit constants **A=0.6, fixup=0.15, launch=0.5** in `_cp.py`.
They require calibration with `cp3_bench.py` on both GPUs. Candidate J is bounded
by `min(16, max(2, floor(2*SMs/Hv)+1))`; the FP32 workspace has shape
`[3,Jcap,Hv,128,128]`. `cp_segments` overrides the cost gate and clamps to that
bound and the chunk count. Exact J can be impossible with equal, 64-aligned
segments (T=200, target=3 produces J=2,L=128); returned J always equals ceil(T/L).

A (`mn`) launches `(2J-3)*Hv` CTAs. Segment zero runs from the original pool
state and saves both the original state and segment one's start. Intermediate
segments compute N from zero and M from identity/zero V, using the same gated
inverse and precision options as the full kernel. Q loads, QK, O1/O2, norm,
output stores and Q/O tails are absent. The second ordered math phase remains
as an empty rendezvous: it protects inverse scratch reuse after both WGs have
read NewV's operands. V zero-fill acquires/commits exactly once per chunk.

A single FP32 IEEE CuTe SIMT fixup (omitted for J=2) owns four independent V rows
per CTA, keeps each row in warp registers, and iterates over segment boundaries.
C (`seg`) reads every initial state from W, and only its final segment writes the
pool. No C CTA reads the pool, eliminating initial/final aliasing. Segment-zero
convolution uses the original history; later segments read the preceding three
raw projected rows. The existing history commit runs once with the real cu.
Intermediate segments are complete 64-row tiles; the last segment ends at the
allocation end, so OutputStore's N=1 fast path handles its tail safely.
Slot -1 retains zero y and untouched pools, including empty pools.

CP workspace and variants must be warmed eagerly before graph capture; captured
replays can change projected data, slot and initial flag. The CPU tests mock
allocation, compilation and scalar extraction to ensure warmed supplied-output
calls perform none of them. GPU tests share each FP64 oracle across forced plans,
compare RMS to non-CP within 1.15x plus 1e-7, check history/untouched rows exactly,
and cover fixup and graph replay. GPU execution is validated on device.

Offline compile command (use the cache/TMPDIR setup above, with a CP3 directory):

```bash
$CP2_PY scripts/compile_gdn_chunk_ws.py --fused --cp-variants full,mn,seg \
  --variants f16nvs --producer-registers 72 --prefetch 2 --output .worklog/gdn-cp3/final
```

Add `--watchdog` or `--iket` to compile those specializations. CP2's debug harness
accepts `--cp-segments 4 --lengths 1000 --check`; its default remains CP-disabled.
GPU commands and benchmark arguments are in [README.md](README.md#gp-cp3-single-sequence-context-parallelism).

### CP3 offline results (2026-10-16)

Ruff check/format and whitespace checks pass. GDN CPU tests: **1,872 passed,
2 skipped**. All **87** CP2/CP3 GPU tests collect and skip without a GPU.
`full`, `mn`, `seg` and FP32 fixup compile for both 16x32 and 16x48 with f16nvs;
MN/SEG also compile with watchdog and IKET. IKET retains 14 MN ranges (no Q/O)
and all 24 SEG ranges. Additional 16x32 MN builds cover base, all BF16 hi/lo
options, global/warp FP16 guards, and FP16 O2-only. No GPU runs were performed.

Fresh uninstrumented builds of HEAD **e480cc1** establish today's baseline;
the older CP2 resource table above describes an earlier producer. Both geometries
have identical figures in this CP3 matrix. All chunk variants use 168 launch
registers and 100608 bytes of shared memory.

| Variant | Producer/math budgets | Stack B | Spill store/load B | Producer LDL/STL | Math + tail LDL/STL |
|---|---:|---:|---:|---:|---:|
| HEAD/full | 72/216 | 40 | 56/112 | 2/2 | 24/8 |
| MN | 56/224 | 16 | 12/40 | 0/0 | 10/2 |
| SEG | 72/216 | 40 | 44/100 | 2/2 | 23/8 |

The no-output MN producer caps `producer_registers` at 56 and gives the surplus
to math. This is the only register-budget change; the exact bound
`128*p + 256*m <= 168*384` still holds. A loop of 64-bit zero stores avoids
zero-fill address spills. Initial CP state loads follow the first KK/QK work to
reduce live ranges, and CTA coordinates are rematerialized for final stores.
Unowned SEG CTAs retain zero state without loading W; they still write zero y.

Fixup uses 36 registers with zero stack/spills/shared memory; history uses 16
registers with zero stack/spills/shared memory. Spill figures are static ptxas
bytes and SASS instruction sites, not measured traffic. Both CP variants meet
the producer and math spill limits against today's full kernel. Instrumented
builds change scheduling/resources and must not be used for timing.

Artifacts: `.worklog/gdn-cp3/{baseline-build,final,watchdog,iket}/resources.json`,
PTX, SASS, cubins and ptxas logs. Extra precision logs are under `precision/`.
The bounded target rounding and MN budget cap are the implementation adjustments;
the requested A → IEEE fixup → C → unchanged history-commit decomposition is retained.

## Producer efficiency experiments after a569b1b

The default now masks packed pairs, reducing producer instructions without
changing valid-row arithmetic. `PortOptions.remap_y` and `q_lookahead` are separate
compile-time flags, default false. CP2 timing, IKET and debug CLIs accept
`--remap-y` and `--q-lookahead`; the precision harness accepts the corresponding
boolean port options. Existing watchdog tags and IKET range multiplicities remain.
The remapped y preloads one four-token group of z, then streams the remaining
groups, unlike the historical all-z preload described above.
