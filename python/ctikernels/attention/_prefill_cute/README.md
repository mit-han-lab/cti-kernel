<!-- SPDX-License-Identifier: MIT -->

# CuTe paged prefill on sm_120/sm_121

`CutePagedPrefill` implements causal NHD paged attention with bottom-right
alignment, BF16/FP16 Q/K/V, D=64/128/256, and arbitrary dividing GQA and positive
page sizes. Packed rows are `query_index * group_size + head_in_group`.
Each warp owns 16 packed rows and traverses its full KV chunk.

Construction compiles against fixed batch/query bounds and initializes arrival
counters in caller-supplied workspace. `run` validates tensor metadata and launches
one kernel through TVM-FFI; it does not allocate tensors or read CSR values on the
host. Q row/head strides are runtime values: the last dimension must be contiguous,
row/head strides multiples of eight elements, and the base 16-byte aligned.
The legacy `q_row_stride` constructor keyword remains accepted but does not fix
runtime strides. Out is contiguous and 16-byte aligned. K and V may have different
16-byte-aligned page strides, including views of packed `[pages,2,P,Hkv,D]`
storage; each inner `[P,Hkv,D]` page must be dense NHD. Q/out may have any common
row capacity up to `batch_size * max_query_per_request`.
Only rows described by `qo_indptr` are written.

`run(k_scale=1.0, v_scale=1.0)` accepts finite host FP32 scales of either sign,
including zero. Logits use `sm_scale * k_scale`; zero produces uniform weights
over visible keys, with masked keys excluded. V scaling happens in FP32 in the
direct or split-merge epilogue before the single output rounding. Scales are
runtime arguments (captured by value in CUDA graphs), with no sign specialization.

CSR contents must be valid: monotone int32 pointers, in-range physical page IDs,
last-page lengths consistent with the page counts, and query lengths within the
construction bound and Q/out capacity. Empty requests use repeated query/page
pointers and last-page length zero. When Q exceeds KV length, rows with negative
causal positions produce zero. This is device masking, not a host validation
error. Metadata may change in place between graph replays. One instance and its
workspace require stream-ordered, nonconcurrent use, including construction.

## D=256 changes and defaults, 2026-10-17

D=256 now defaults to **M=64, N=64, two stages, minimum split chunk 256**
on both architectures. It targets one wave:
`max(1, ceil(SM_count / (B * Hkv * query_tiles)))` splits. The device rounds
chunks to N and applies the minimum chunk, so short contexts use fewer active
splits. Construction reads device properties only, never CSR contents.

D=64/128 defaults are unchanged: M=64 on sm_120 and M=128 on sm_121,
N=64, two stages, minimum chunk 64, and a two-wave split target.
`CTI_PREFILL_BLOCK_M`, `CTI_PREFILL_BLOCK_N`, `CTI_PREFILL_STAGES`,
`CTI_PREFILL_SPLITS`, and `CTI_PREFILL_MIN_CHUNK` remain available.
For D=256, overriding M=128 or stages=3 selects N=32 unless N is explicitly
set; configurations exceeding 99 KiB still fail validation. Workspace sizing
and construction must use the same device and environment.

GB10 (48 SMs), Hq/Hkv=16/4, KV=2048:

| B | Q | Base CTAs, new M64 | Previous M128 split bound | New splits | New launched CTAs |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 16 | 12 | 3 | 48 |
| 3 | 64 | 48 | 4 | 1 | 48 |
| 2 | 256 | 128 | 2 | 1 | 128 |
| 4 | 256 | 256 | 1 | 1 | 256 |

The old M64 override used six splits for B1 Q64. It now uses three. Relative
to the old M128 default, B1 FP32 partial-output writes fall from 12 MiB to
3 MiB per layer, with matching reductions in merge reads (excluding LSE).
B3 Q64 and B2 Q256 now avoid partial-output traffic and merging altogether.
These are static traffic estimates, not measured latency improvements.

### What was expensive

Q was already in shared memory for D=256; the previous kernel did not keep a
full register Q tile. The output accumulator still needs 128 FP32 registers
per thread. Increasing N without controlling fragment lifetimes caused external
ptxas spills. QK now processes four k-chunks per rolled iteration, loading Q
with `ldmatrix`; PV issues statically indexed 16-column V slices. Fully unrolling
the gather also spilled, so the common dividing-page gather instead loads one
page ID and computes one global base for several copies, keeping its outer loop
rolled. Arbitrary page sizes retain the general gather.

K and V no longer alias on D=256. The same shared-memory budget holds persistent
Q, `stages-1` K slots, and one V tile. V copies start before QK. After all warps
finish QK and read current page IDs, a barrier permits refilling the consumed K
slot and the current ID slot. V and future IDs share an async group; future K
has the next group. `wait_group(1)` plus a CTA barrier publishes V/IDs before
PV. The post-PV barrier protects the V tile from the next iteration. With
three stages, the K ring has two slots and advances independently of the
three-slot ID ring. The final wait drains copies before arena reuse.

N=64 amortizes softmax and synchronization across twice as many keys. For a
full tile, the score reduction still uses two row maxima and two row sums per
thread, each reduced with two shuffles. Unmasked scores use one FMA and one
`ex2.approx.ftz.f32`; only partial/diagonal tiles mask scores. Negative and zero
scales use runtime sign handling, and empty rows remain zero. Output
rescaling is skipped when its factor is one. The loop clips its end to the
last live query's causal position.

The split epilogue already distributed dimensions across all threads. Its
remaining waste was recomputing LSE maxima, exponentials, and normalization for
every eight-element output vector. The D=256 winner now computes normalized
weights once per row, reuses the LSE workspace to publish them, and synchronizes
before the vector merge. Two `ld.global.cg.v4.f32` loads cover each eight-element
partial. The merge writes output vectors directly, removing the shared-memory
round trip. A final barrier precedes counter reset. No spinning, cooperative
launch, or co-residency guarantee is needed. Single-split output still uses
shared staging for coalesced vector stores. D=64/128 keep their previous path.

All instruction use follows the [public PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
No third-party kernel code was copied.

## Offline evidence

The receipts below describe the D256 optimization baseline, before runtime Q/KV
strides and scale handling. They are historical evidence, not fresh compile or
instruction-count results for those later changes.

CPU configuration tests: **131 passed**. Ruff check and format: passed.
The compile matrix and the four exact GB10 screen bounds cover **62 offline
compiles** across sm_120a/sm_121a, CuTe 4.8 and external CUDA 13.0 ptxas.
All have zero native stack/local bytes and zero external stack/spill-store/
spill-load bytes. CUDA remains uninitialized. D=128 resource usage and complete
per-opcode full-tile counts match the previous receipts for every existing case.

The compile script writes fresh CSV/JSON files to its output directory. CuTe emits PTX 9.4;
`--ptx-version 9.0` changes only a separate external-assembler copy.

The following page-16 BF16 configurations use B=3, Qmax=256, Hkv=4, GQA=4,
except the exact screen cases. Both architectures produce identical counts.
Registers are native / external. Shared totals include arena, worst-case page
ID capacity, winner flag, and the 1,024-byte static reservation.

| D | M | N | Stages | Splits | Registers | Shared bytes/CTA | CTAs / warps per SM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 64 | 2 | 1 | 212 / 210 | 50,704 | 2 / 8 |
| 128 | 128 | 64 | 2 | 2 | 210 / 212 | 50,704 | 1 / 8 |
| 256 | 64 | 32 | 2 | 1 | 206 / 206 | 66,832 | 1 / 4 |
| 256 | 64 | 32 | 2 | 8 | 212 / 214 | 66,832 | 1 / 4 |
| 256 | 128 | 32 | 2 | 1 | 206 / 206 | 99,600 | 1 / 8 |
| 256 | 128 | 32 | 2 | 4, 15, 16 | 212 / 214 | 99,600 | 1 / 8 |
| 256 | 64 | 64 | 2 | 1 | 234 / 250 | 99,856 | 1 / 4 |
| 256 | 64 | 64 | 2 | 3, 11 | 238 / 254 | 99,856 | 1 / 4 |

The exact B1 Q64 screen specialization has 238 / 254 registers; the other three
have 234 / 250. All use 99,856 shared bytes. The page-7, GQA-9, three-stage
N32/split7 case uses 218 / 220 registers and 83,344 shared bytes. N64/page7
and page1 split cases use 236 / 254 and 238 / 254 registers respectively.

Occupancy is a resource bound: 65,536 registers/SM, register allocation rounded
to 256 per warp, 48 warps/SM, and a conservative 100 KiB shared carveout.
See NVIDIA's [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/#occupancy).
D256 M64 has 4 resident warps (8.3%); M128 has 8 (16.7%). All D256 cases fit
one CTA/SM: 170, 188, or 48 resident CTAs on those SM counts. The new default
trades the old M128 warp occupancy for smaller query tiles, lower split traffic,
and fewer KV-loop instructions. Hardware timing must assess that tradeoff.

### Static SASS per full KV tile per warp

Counts follow the shortest/longest unmasked control-flow paths, expanding
partially unrolled gather and QK loops by their actual trip counts. Predicated
instruction slots count as issued, including assembler-inserted predicated-off
LDS. Ranges cover rescaling and page-ID-copy participation. Prologue, diagonal/
tail paths, and epilogue are excluded. HMMA fractions are static instruction
fractions, not measured tensor utilization or latency predictions.

| D256 configuration | Instructions/tile/warp | HMMA | HMMA fraction |
|---|---:|---:|---:|
| M64 N32, split1 | 712–866 | 128 | 14.8–18.0% |
| M64 N32, split8 | 709–863 | 128 | 14.8–18.1% |
| M128 N32, split1 | 680–834 | 128 | 15.3–18.8% |
| M128 N32, split4/15/16 | 677–831 | 128 | 15.4–18.9% |
| M64 N64, split1 | 1,093–1,247 | 256 | 20.5–23.4% |
| M64 N64, split3/11, or FP16 split7 | 1,092–1,246 | 256 | 20.5–23.4% |
| M64 N64, page1 split4 | 1,433–1,562 | 256 | 16.4–17.9% |
| M64 N64, page7 GQA3 split4 | 2,307–2,473 | 256 | 10.4–11.1% |
| M64 N32, stages3, page7 GQA9 split7 | 1,322–1,488 | 128 | 8.6–9.7% |

For equal work (64 keys), the old M64 N32 kernel needed 1,610–1,914
instructions versus the new M64 N64's 1,093–1,247: 32–35% fewer. The old M64
N64 override needed 1,475–1,629 instructions. Old M128 N32 needed 1,232–1,536
per 64 keys; its copy work was shared among twice as many warps.

| Opcode per 64 keys, split1 | Old M64 N32 (two tiles) | New M64 N64 (one tile) |
|---|---:|---:|
| HMMA | 256 | 256 |
| BAR | 10 | 4 |
| LDSM | 160 | 144 |
| LDS | 128–134 | 32–35 |
| LDGSTS | 32–34 | 32–33 |
| SHFL | 16 | 8 |
| MUFU | 36 | 34 |
| FMNMX | 44 | 38 |
| FFMA | 36 | 34 |
| FMUL | 8–264 | 4–132 |
| IADD | 268–274 | 124–126 |
| IMAD | 134–136 | 78–80 |

GPU correctness, race checking, graph replay, and timing remain unverified on
this revision. All 49 non-exhaustive GPU cases skip without CUDA. Eighteen new
D256 cases cover separate K/V storage, two/three-stage wraps, N32/N64, FP16/BF16,
arbitrary pages, scale signs, all-empty rows, and split-count changes under
replay. Run correctness and race checks before using the timing results.

```bash
PYTHONPATH=python python -m pytest -q tests/modules/test_prefill_cute_config.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m tests.modules._prefill_cute_compile \
  --output prefill-receipts --ptx-version 9.0
PYTHONPATH=python python -m pytest -q tests/gpu/test_prefill_cute.py -m 'not exhaustive'
ruff check python/ctikernels/attention/_prefill_cute \
  tests/modules/test_prefill_cute_config.py tests/modules/_prefill_cute_compile.py \
  tests/gpu/test_prefill_cute.py
```
