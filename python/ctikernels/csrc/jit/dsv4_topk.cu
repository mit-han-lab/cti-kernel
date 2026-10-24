// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Reference: SGLang deepseek_v4 topk_v1 and FlashInfer radix top-k
// (Apache-2.0). Original deterministic implementation; design references only,
// no adapted code.

// Row top-K (K = 512) of FP32 scores, one CTA per row: a 12-bit histogram of
// the order-preserving key's top digit finds the threshold bin; one pass over
// the row takes every key above it and collects the bin's entries in shared
// memory; two 10-bit passes over those settle the exact threshold key, whose
// ties go to the lowest indices (sorted when they overflow). The selected set
// is therefore fixed, and a bitonic sort fixes its order (ascending entry
// ids). Bins larger than shared memory take the same passes over the whole
// row. Rows of at most 2048 entries run the same rounds over keys held in
// registers and rank the chosen ones with ballots. Entries past `length` are
// ignored; the output holds min(length, K) page-mapped rows (and, when asked,
// the entry ids in a second output), then -1.

#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kK = 512;
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / 32;
constexpr int kCap = 8192;    // compacted entries per buffer (two buffers)
constexpr int kShortKeys = 4; // rows up to kShortKeys * kThreads entries keep
                              // their keys in registers (the short path)
constexpr int kMapBits =
    2 * kCap * 32; // rows up to this length rank via a bitmap

struct TopKParams {
  const float *scores;
  int64_t score_stride;
  const int32_t *lengths;
  const int32_t *table;
  int64_t table_stride;
  const int32_t *batch; // row -> request, or nullptr (identity)
  int32_t *out;
  int64_t out_stride;
  int32_t page_bits;
  int32_t *entries; // the selected entry ids too, or nullptr
  int64_t entries_stride;
};

__device__ __forceinline__ uint32_t order_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Digits of the order-preserving key: 12 bits (sign, exponent, 3 mantissa
// bits), then 10 and 10.
constexpr int kBits0 = 12, kBits1 = 10;
constexpr int kBins0 = 1 << kBits0, kBins1 = 1 << kBits1;

struct alignas(16) Shared {
  uint32_t hist0[kBins0]; // pass 1; reused for the second refine digit
  uint32_t hist1[kBins1];
  uint32_t start[kThreads]; // ranking: chosen entries in lower threads' words
  int32_t page[kThreads];   // the row's page ids, when it has <= kThreads pages
  int32_t warp_sum[2][kWarps];
  int32_t slab_sum[kShortKeys][kWarps]; // short rows: flags per warp and slab
  int32_t bin;
  int32_t above;
};

// Exclusive block scan of two flags: each thread's rank and the block totals.
__device__ __forceinline__ void scan2(Shared &sh, bool a, bool b, int &ra,
                                      int &rb, int &ta, int &tb) {
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned ma = __ballot_sync(0xffffffffu, a);
  const unsigned mb = __ballot_sync(0xffffffffu, b);
  const unsigned below = (1u << lane) - 1u;
  if (lane == 0) {
    sh.warp_sum[0][warp] = __popc(ma);
    sh.warp_sum[1][warp] = __popc(mb);
  }
  __syncthreads();
  int pa = 0, pb = 0, sa = 0, sb = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const int va = sh.warp_sum[0][w], vb = sh.warp_sum[1][w];
    pa += w < static_cast<int>(warp) ? va : 0;
    pb += w < static_cast<int>(warp) ? vb : 0;
    sa += va;
    sb += vb;
  }
  ra = pa + __popc(ma & below);
  rb = pb + __popc(mb & below);
  ta = sa;
  tb = sb;
  __syncthreads();
}

// The bin of `hist` (kBins bins) holding the `remaining`-th largest key
// (sh.bin) and how many keys lie in higher bins (sh.above).
// Sum of `x` over this thread and all higher threads of the block.
__device__ __forceinline__ uint32_t block_suffix(Shared &sh, uint32_t x) {
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t v = x;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const uint32_t up = __shfl_down_sync(0xffffffffu, v, off);
    if (lane + off < 32)
      v += up;
  }
  if (lane == 0)
    sh.warp_sum[0][warp] = static_cast<int32_t>(v);
  __syncthreads();
  uint32_t later = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    later += w > static_cast<int>(warp) ? sh.warp_sum[0][w] : 0;
  __syncthreads();
  return v + later;
}

// The bin of `hist` (kBins bins) holding the `remaining`-th largest key
// (sh.bin) and how many keys lie in higher bins (sh.above).
template <int kBins>
__device__ __forceinline__ void find_bin(Shared &sh, const uint32_t *hist,
                                         int remaining) {
  constexpr int kPer = kBins / kThreads;
  static_assert(kPer % 2 == 0, "bins are read in pairs");
  const int tx = threadIdx.x;
  uint32_t bins[kPer]; // vector loads: fewer bank conflicts than kPer scalars
  if constexpr (kPer % 4 == 0) {
#pragma unroll
    for (int i = 0; i < kPer; i += 4) {
      const uint4 v = reinterpret_cast<const uint4 *>(hist + tx * kPer)[i / 4];
      bins[i] = v.x;
      bins[i + 1] = v.y;
      bins[i + 2] = v.z;
      bins[i + 3] = v.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPer; i += 2) {
      const uint2 v = reinterpret_cast<const uint2 *>(hist + tx * kPer)[i / 2];
      bins[i] = v.x;
      bins[i + 1] = v.y;
    }
  }
  uint32_t mine = 0;
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    mine += bins[i];
  uint32_t acc = block_suffix(sh, mine) - mine; // keys in higher threads' bins
  const uint32_t need = static_cast<uint32_t>(remaining);
#pragma unroll
  for (int i = kPer - 1; i >= 0; --i) {
    const uint32_t h = bins[i];
    if (acc < need && acc + h >= need) {
      sh.bin = tx * kPer + i;
      sh.above = static_cast<int>(acc);
    }
    acc += h;
  }
  __syncthreads();
}

// `pages`: the row's page ids prefetched to shared memory, or nullptr.
__device__ __forceinline__ int32_t value_of(const TopKParams &p, int row,
                                            const int32_t *pages, int32_t idx) {
  const int request = p.batch != nullptr ? p.batch[row] : row;
  const int at = idx >> p.page_bits;
  const int32_t page =
      pages != nullptr ? pages[at] : p.table[request * p.table_stride + at];
  return (page << p.page_bits) | (idx & ((1 << p.page_bits) - 1));
}

// Output slot `slot` of `row`: entry `idx` (-1: none) as its row, and its id.
__device__ __forceinline__ void emit(const TopKParams &p, int row,
                                     const int32_t *pages, int slot,
                                     int32_t idx) {
  p.out[row * p.out_stride + slot] =
      idx < 0 ? -1 : value_of(p, row, pages, idx);
  if (p.entries != nullptr)
    p.entries[row * p.entries_stride + slot] = idx;
}

// Ascending bitonic sort of keys[0, n) in shared memory (n a power of two).
__device__ __forceinline__ void bitonic(int32_t *keys, int n) {
  for (int size = 2; size <= n; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < n; i += kThreads) {
        const int j = i ^ stride;
        if (j > i) {
          const bool up = (i & size) == 0;
          const int32_t a = keys[i], b = keys[j];
          if ((a > b) == up) {
            keys[i] = b;
            keys[j] = a;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Exclusive ranks of `flag` in entry order (entry i = tx + r * kThreads is
// slab r) and the total.
__device__ __forceinline__ int short_ranks(Shared &sh,
                                           const bool (&flag)[kShortKeys],
                                           int (&rank)[kShortKeys]) {
  const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned mask[kShortKeys];
#pragma unroll
  for (int r = 0; r < kShortKeys; ++r) {
    mask[r] = __ballot_sync(0xffffffffu, flag[r]);
    if (lane == 0)
      sh.slab_sum[r][warp] = __popc(mask[r]);
  }
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int r = 0; r < kShortKeys; ++r) {
    int before = 0, total = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const int v = sh.slab_sum[r][w];
      before += w < static_cast<int>(warp) ? v : 0;
      total += v;
    }
    rank[r] = base + before + __popc(mask[r] & ((1u << lane) - 1u));
    base += total;
  }
  __syncthreads();
  return base;
}

// Rows of kK < length <= kShortKeys * kThreads entries: the radix rounds run
// over keys held in registers (no candidate lists), and the chosen entries'
// output slots come from ballot ranks, already in ascending entry order.
__device__ __forceinline__ void short_topk(const TopKParams &p, Shared &sh,
                                           int row, const float *scores,
                                           int length, bool prefetch,
                                           int32_t my_page) {
  const int tx = threadIdx.x;
  uint32_t key[kShortKeys];
  bool valid[kShortKeys];
#pragma unroll
  for (int r = 0; r < kShortKeys; ++r) {
    const int i = tx + r * kThreads;
    valid[r] = i < length;
    key[r] = valid[r] ? order_key(scores[i]) : 0u;
  }
  int remaining = kK;
  uint32_t prefix = 0;
  int done = 0; // settled high bits of the threshold key
  bool all = false;
#pragma unroll
  for (int round = 0; round < 3 && !all; ++round) {
    const int bits = round == 0 ? kBits0 : kBits1;
    const int shift = 32 - done - bits;
    uint32_t *hist = round == 1 ? sh.hist1 : sh.hist0;
    for (int i = tx; i < (1 << bits); i += kThreads)
      hist[i] = 0;
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kShortKeys; ++r)
      if (valid[r] &&
          (done == 0 || (key[r] >> (32 - done)) == (prefix >> (32 - done))))
        atomicAdd(&hist[(key[r] >> shift) & ((1u << bits) - 1u)], 1u);
    __syncthreads();
    if (round == 0)
      find_bin<kBins0>(sh, hist, remaining);
    else
      find_bin<kBins1>(sh, hist, remaining);
    const uint32_t bin = static_cast<uint32_t>(sh.bin);
    remaining -= sh.above;
    all = hist[bin] == static_cast<uint32_t>(remaining);
    prefix |= bin << shift;
    done += bits;
  }
  // Chosen: above the threshold's settled bits, or equal to them (all of
  // them, or the `remaining` lowest entries of an exact tie).
  bool chosen[kShortKeys], eq[kShortKeys];
#pragma unroll
  for (int r = 0; r < kShortKeys; ++r) {
    const uint32_t high = done == 32 ? key[r] : key[r] >> (32 - done);
    const uint32_t top = done == 32 ? prefix : prefix >> (32 - done);
    chosen[r] = valid[r] && high > top;
    eq[r] = valid[r] && high == top;
  }
  int rank[kShortKeys];
  if (!all) {
    short_ranks(sh, eq, rank);
#pragma unroll
    for (int r = 0; r < kShortKeys; ++r)
      chosen[r] = chosen[r] || (eq[r] && rank[r] < remaining);
  } else {
#pragma unroll
    for (int r = 0; r < kShortKeys; ++r)
      chosen[r] = chosen[r] || eq[r];
  }
  if (prefetch)
    sh.page[tx] = my_page;
  short_ranks(sh, chosen, rank); // its barriers also publish sh.page
  const int32_t *pages = prefetch ? sh.page : nullptr;
#pragma unroll
  for (int r = 0; r < kShortKeys; ++r)
    if (chosen[r])
      emit(p, row, pages, rank[r], tx + r * kThreads);
}

__global__ void __launch_bounds__(kThreads)
    dsv4_topk_kernel(const __grid_constant__ TopKParams p) {
  extern __shared__ int32_t buffers[]; // [2][kCap] entry ids
  __shared__ Shared sh;
  __shared__ int32_t chosen[kK];
  __shared__ int32_t s_taken, s_count, s_next;
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  device::PDL::wait<true>(); // the scores come from the previous kernel
  const int length = p.lengths[row];
  const float *scores = p.scores + row * p.score_stride;
  const int want = length < kK ? length : kK;
  // Rows with <= kThreads pages load their page ids now; the loads overlap the
  // passes below.
  const bool prefetch = length > kK && ((length - 1) >> p.page_bits) < kThreads;
  int32_t my_page = 0;
  if (prefetch && tx <= ((length - 1) >> p.page_bits)) {
    const int request = p.batch != nullptr ? p.batch[row] : row;
    my_page = p.table[request * p.table_stride + tx];
  }
  const int32_t *pages = nullptr;

  if (length <= kK) {
    for (int i = tx; i < length; i += kThreads)
      emit(p, row, pages, i, i);
    for (int j = want + tx; j < kK; j += kThreads)
      emit(p, row, pages, j, -1);
    return;
  }
  if (length <= kShortKeys * kThreads) {
    short_topk(p, sh, row, scores, length, prefetch, my_page);
    return;
  }
  int remaining = want;
  for (int i = tx; i < kBins0; i += kThreads)
    sh.hist0[i] = 0;
  for (int i = tx; i < kBins1; i += kThreads)
    sh.hist1[i] = 0;
  if (tx == 0)
    s_taken = s_count = 0;
  __syncthreads();
  for (int i = tx; i < length; i += kThreads)
    atomicAdd(&sh.hist0[order_key(scores[i]) >> (32 - kBits0)], 1u);
  __syncthreads();
  find_bin<kBins0>(sh, sh.hist0, remaining);
  const uint32_t top = static_cast<uint32_t>(sh.bin);
  remaining -= sh.above;
  // A bin holding exactly the entries still needed is taken whole: no refine.
  const bool whole = sh.hist0[top] == static_cast<uint32_t>(remaining);

  // Above the bin: chosen; in it: candidates (with the next digit's histogram).
  for (int i = tx; i < length; i += kThreads) {
    const uint32_t key = order_key(scores[i]);
    const uint32_t digit = key >> (32 - kBits0);
    if (digit > top || (whole && digit == top)) {
      chosen[atomicAdd(&s_taken, 1)] = i;
    } else if (digit == top) {
      const int at = atomicAdd(&s_count, 1);
      if (at < kCap)
        buffers[at] = i;
      atomicAdd(&sh.hist1[(key >> kBits1) & (kBins1 - 1)], 1u);
    }
  }
  __syncthreads();

  const int count0 = s_count;
  const bool spill = count0 > kCap; // the bin did not fit: walk the whole row
  int count = count0;
  int *source = buffers, *target = buffers + kCap;
  uint32_t prefix = top << (32 - kBits0);
  if (whole)
    remaining = 0;
  for (int round = 0; round < 2 && remaining > 0; ++round) {
    const int shift = round == 0 ? kBits1 : 0;
    const uint32_t *hist = round == 0 ? sh.hist1 : sh.hist0;
    find_bin<kBins1>(sh, hist, remaining);
    const uint32_t bin = static_cast<uint32_t>(sh.bin);
    remaining -= sh.above;
    const bool all = hist[bin] == static_cast<uint32_t>(remaining);
    if (round == 0)
      for (int i = tx; i < kBins1; i += kThreads)
        sh.hist0[i] = 0;
    if (tx == 0)
      s_next = 0;
    __syncthreads();
    const uint32_t high = 0xffffffffu << (shift + kBits1);
    const int n = spill ? length : count;
    for (int j = tx; j < n; j += kThreads) {
      const int i = spill ? j : source[j];
      const uint32_t key = order_key(scores[i]);
      if (spill && (key & high) != prefix)
        continue;
      const uint32_t digit = (key >> shift) & (kBins1 - 1);
      if (digit > bin || (all && digit == bin)) {
        chosen[atomicAdd(&s_taken, 1)] = i;
      } else if (digit == bin) {
        const int at = atomicAdd(&s_next, 1);
        if (!spill)
          target[at] = i;
        if (round == 0)
          atomicAdd(&sh.hist0[key & (kBins1 - 1)], 1u);
      }
    }
    __syncthreads();
    prefix |= bin << shift;
    if (!spill) {
      int *swap = source;
      source = target;
      target = swap;
      count = s_next;
    }
    __syncthreads();
    if (all)
      remaining = 0;
  }
  // `prefix` is the exact threshold key: its `remaining` lowest-index entries.
  const int taken = s_taken;
  if (!spill) {
    if (remaining > 0 &&
        count > remaining) { // more ties than room: lowest indices win
      int n = 1;
      while (n < count)
        n <<= 1;
      for (int j = count + tx; j < n; j += kThreads)
        source[j] = 0x7fffffff;
      __syncthreads();
      bitonic(source, n);
    }
    for (int j = tx; j < remaining; j += kThreads)
      chosen[taken + j] = source[j];
  } else {
    int ties = 0;
    for (int base = 0; base < length && ties < remaining; base += kThreads) {
      const int i = base + tx;
      const bool eq = i < length && order_key(scores[i]) == prefix;
      int ra, rb, ta, tb;
      scan2(sh, eq, false, ra, rb, ta, tb);
      if (eq && ties + ra < remaining)
        chosen[taken + ties + ra] = i;
      ties += ta;
    }
  }
  __syncthreads();
  if (prefetch) {
    sh.page[tx] = my_page;
    pages = sh.page;
  }
  // A fixed order: ascending entry ids (want == kK here).
  if (length <= kMapBits) { // rank = chosen entries below, from a bitmap
    uint32_t *map = reinterpret_cast<uint32_t *>(buffers);
    const int words = (length + 31) >> 5;
    for (int w = tx; w < words; w += kThreads)
      map[w] = 0;
    __syncthreads();
    const int mine = chosen[tx]; // kThreads == kK
    atomicOr(&map[mine >> 5], 1u << (mine & 31));
    __syncthreads();
    const int per = (words + kThreads - 1) / kThreads; // words per thread
    uint32_t sum = 0;
    for (int i = 0; i < per; ++i) {
      const int w = tx * per + i;
      sum += w < words ? __popc(map[w]) : 0;
    }
    // Exclusive prefix over threads = total - suffix(this and higher).
    sh.start[tx] = kK - block_suffix(sh, sum);
    __syncthreads();
    // Each chosen entry's rank: chosen entries in lower words, then lower bits.
    const int w = mine >> 5, owner = w / per;
    int rank = static_cast<int>(sh.start[owner]) +
               __popc(map[w] & ((1u << (mine & 31)) - 1u));
    for (int v = owner * per; v < w; ++v)
      rank += __popc(map[v]);
    emit(p, row, pages, rank, mine);
  } else {
    bitonic(chosen, kK);
    for (int j = tx; j < kK; j += kThreads)
      emit(p, row, pages, j, chosen[j]);
  }
}

bool is(const tvm::ffi::TensorView t, std::uint8_t code, std::uint8_t bits) {
  return t.dtype().code == code && t.dtype().bits == bits;
}

} // namespace

struct Dsv4TopK {
  // scores FP32 [R, N] (unit column stride), lengths int32 [R], table int32
  // [B, P] (unit column stride), batch int32 [R] (empty: row -> row), out int32
  // [R, 512] rows, entries int32 [R, 512] entry ids (empty: not written).
  static void run(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView lengths,
                  const tvm::ffi::TensorView table,
                  const tvm::ffi::TensorView batch,
                  const tvm::ffi::TensorView out,
                  const tvm::ffi::TensorView entries, std::int64_t page_rows) {
    using host::RuntimeCheck;
    const auto rows = scores.size(0);
    RuntimeCheck(scores.ndim() == 2 && is(scores, kDLFloat, 32) &&
                     scores.stride(1) == 1,
                 "DSV4 top-k: FP32 scores [R, N]");
    RuntimeCheck(is(lengths, kDLInt, 32) && lengths.size(0) == rows &&
                     is(table, kDLInt, 32) && table.stride(1) == 1 &&
                     is(out, kDLInt, 32) && out.size(0) == rows &&
                     out.size(1) == kK && out.stride(1) == 1,
                 "DSV4 top-k: int32 lengths/table/out [R, 512]");
    const bool with_entries = entries.size(0) > 0;
    RuntimeCheck(!with_entries ||
                     (is(entries, kDLInt, 32) && entries.size(0) == rows &&
                      entries.size(1) == kK && entries.stride(1) == 1),
                 "DSV4 top-k: int32 entries [R, 512]");
    RuntimeCheck(page_rows > 0 && (page_rows & (page_rows - 1)) == 0,
                 "DSV4 top-k: page rows is a power of two");
    if (rows == 0)
      return;
    int bits = 0;
    while ((1 << bits) < page_rows)
      ++bits;
    const TopKParams params{
        static_cast<const float *>(scores.data_ptr()),
        scores.stride(0),
        static_cast<const int32_t *>(lengths.data_ptr()),
        static_cast<const int32_t *>(table.data_ptr()),
        table.stride(0),
        batch.size(0) > 0 ? static_cast<const int32_t *>(batch.data_ptr())
                          : nullptr,
        static_cast<int32_t *>(out.data_ptr()),
        out.stride(0),
        bits,
        with_entries ? static_cast<int32_t *>(entries.data_ptr()) : nullptr,
        with_entries ? entries.stride(0) : 0,
    };
    constexpr std::size_t kSmem = 2 * kCap * sizeof(int32_t);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        dsv4_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSmem));
    RuntimeCheck(attribute == cudaSuccess, "DSV4 top-k: shared memory size");
    const auto device = scores.device();
    const auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
    host::LaunchKernel(static_cast<unsigned>(rows), kThreads, stream, kSmem)
        .with_attr(true)(dsv4_topk_kernel, params);
    RuntimeCheck(cudaGetLastError() == cudaSuccess, "DSV4 top-k: launch");
  }
};
