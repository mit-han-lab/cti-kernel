// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Reference: CUTLASS/CuTe warp-specialized TMA (BSD-3-Clause); public PTX ISA.
// Original pipeline implementation; design references only, no adapted code.
// Tensor maps use cuTensorMapEncodeTiled through
// cudaGetDriverEntryPointByVersion (12.0 ABI).

// Block-FP8 dense GEMM for compute capability 12.x (GB10): out[m, n] = sum over
// 128-column groups g of (q[m, g] . w[n, g]) * a_scale[m, g] * w_scale[n / 128,
// g]. Each group's dot is accumulated in FP32 by E4M3 tensor-core MMAs
// (mma.sync m16n8k32), then added to the FP32 result with one fused
// multiply-add by (a_scale * w_scale), the Triton kernels' order; the result is
// rounded once to BF16 or FP16.
//
// Persistent CTAs (one per SM) walk BM x BN output tiles in bands of GROUP_M
// row tiles. A ring of STAGES shared-memory slots holds one 128-column group of
// the activation and weight tiles each: the weight tile arrives by TMA
// (128-byte swizzle). The activation tile either arrives by TMA as well (In =
// uint8: E4M3 q with FP32 scales from the caller) or is quantized here (In =
// BF16 or FP16): kQuantWarps warps read the activation rows, apply the
// per-token-group recipe of ctikernels.linear.block_fp8 (bit for bit) and store
// E4M3 values and scales into the slot. WARPS_M x WARPS_N MMA warps consume the
// slots; full/empty mbarriers are the only synchronization, so loads of the
// next tile overlap the current tile's MMAs and stores. The host checks N % BN
// == 0, K % 128 == 0 and 16-byte aligned rows.

#include <ctiengine/tensor.h>
#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

constexpr int kGroup =
    128; // scale group width = bytes of one E4M3 tile row per slot

struct GemmParams {
  const void *__restrict__ x; // [M, K] BF16/FP16 activations (quantized here)
  const float *__restrict__ a_scale; // [M, K / 128] (E4M3 activations)
  const float *__restrict__ b_scale; // [N / 128, K / 128]
  void *__restrict__ out;            // [M, N]
  std::int64_t ldx, ldo;             // row strides in elements
  std::int64_t sam, sag, sbn, sbg;   // scale strides in elements
  int m, n, k;
  int tiles_m, tiles_n;
};

__device__ __forceinline__ std::uint32_t smem_addr(const void *p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(std::uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar),
               "r"(count)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(std::uint32_t bar,
                                          std::uint32_t parity) {
  asm volatile("{\n"
               ".reg .pred done;\n"
               "wait_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
               "@!done bra wait_%=;\n"
               "}\n" ::"r"(bar),
               "r"(parity)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(std::uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar)
               : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(std::uint32_t bar,
                                               std::uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar),
      "r"(bytes)
      : "memory");
}

// 2-D TMA load of the box at (x = column byte, y = row) into shared memory,
// completing on bar. The CTA destination (as CUTLASS's sm120 path): sm_12x
// lowers a shared::cluster one to a driver syscall whose 16000-byte per-thread
// stack the driver then reserves for the whole device.
__device__ __forceinline__ void tma_load_2d(std::uint32_t dst,
                                            const CUtensorMap *map,
                                            std::uint32_t bar, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
               "complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];\n" ::"r"(dst),
               "l"(reinterpret_cast<std::uint64_t>(map)), "r"(bar), "r"(x),
               "r"(y)
               : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t (&r)[4],
                                            std::uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

// d = a . b (no accumulator input: the first k-step of a group).
__device__ __forceinline__ void mma_e4m3_zero(float (&d)[4],
                                              const std::uint32_t (&a)[4],
                                              std::uint32_t b0,
                                              std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f));
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4],
                                         const std::uint32_t (&a)[4],
                                         std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two FP32 values to two E4M3 bytes (round to nearest even, saturating to
// +-448, NaN kept), `lo` in the low byte.
__device__ __forceinline__ std::uint32_t e4m3x2(float lo, float hi) {
  std::uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// Byte offset of 16-byte chunk `chunk` of tile row `row` in TMA's 128-byte
// swizzle: rows are 128 bytes and chunk c of row r is stored at chunk c ^ (r %
// 8), so the 8 rows an ldmatrix phase reads at one logical chunk fall in
// distinct banks.
__device__ __forceinline__ std::uint32_t tile_offset(int row, int chunk) {
  return static_cast<std::uint32_t>(row * kGroup + ((chunk ^ (row & 7)) << 4));
}

template <typename Out>
__device__ __forceinline__ void store_pair(Out *p, float x, float y);

template <>
__device__ __forceinline__ void store_pair<__nv_bfloat16>(__nv_bfloat16 *p,
                                                          float x, float y) {
  *reinterpret_cast<__nv_bfloat162 *>(p) = __floats2bfloat162_rn(x, y);
}

template <>
__device__ __forceinline__ void store_pair<__half>(__half *p, float x,
                                                   float y) {
  *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(x, y);
}

__device__ __forceinline__ void tile_coords(const GemmParams &p, int group_m,
                                            int t, int &tm, int &tn) {
  const int band = group_m * p.tiles_n;
  const int first_m = (t / band) * group_m;
  const int band_m = min(p.tiles_m - first_m, group_m);
  tm = first_m + (t % band) % band_m;
  tn = (t % band) / band_m;
}

template <typename In> struct Pair;
template <> struct Pair<__nv_bfloat16> {
  using T = __nv_bfloat162;
  static __device__ __forceinline__ float2 to_float2(T v) {
    return __bfloat1622float2(v);
  }
};
template <> struct Pair<__half> {
  using T = __half2;
  static __device__ __forceinline__ float2 to_float2(T v) {
    return __half22float2(v);
  }
};

// Quantizer warps take 8 rows of a group per iteration, 4 lanes per row and 32
// consecutive values (64 bytes) per lane: one iteration's slice is 4 16-byte
// vectors per lane. Rows past M read as zeros (and quantize to zeros).
template <typename In>
__device__ __forceinline__ void load_slice(const GemmParams &p, int row, int g,
                                           int lane, uint4 (&raw)[4]) {
  const uint4 *src = reinterpret_cast<const uint4 *>(
      static_cast<const In *>(p.x) +
      static_cast<std::int64_t>(row < p.m ? row : 0) * p.ldx +
      static_cast<std::int64_t>(g) * kGroup + (lane & 3) * 32);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    raw[i] = row < p.m ? __ldg(src + i) : make_uint4(0u, 0u, 0u, 0u);
}

// Quantize one loaded slice (tile row `row`) into slot memory with the recipe
// of ctikernels.linear.block_fp8, bit for bit: amax = max(|x|, 1e-10) (NaN if
// any value is NaN), scale = amax * fp32(1 / 448), q = E4M3_RN(saturate(x *
// (448 / amax))).
template <typename In>
__device__ __forceinline__ void quantize_slice(const uint4 (&raw)[4], int row,
                                               std::uint32_t a_tile,
                                               float *scales, int lane) {
  using P = Pair<In>;
  using T = typename P::T;
  const T *v = reinterpret_cast<const T *>(raw);
  // |x| maxima as packed pairs (exact; NaN propagates), then over the row's 4
  // lanes.
  T m = __habs2(v[0]);
#pragma unroll
  for (int i = 1; i < 16; ++i)
    m = __hmax2_nan(m, __habs2(v[i]));
#pragma unroll
  for (int d = 2; d >= 1; d >>= 1) {
    std::uint32_t bits = *reinterpret_cast<std::uint32_t *>(&m);
    bits = __shfl_xor_sync(0xffffffffu, bits, d);
    m = __hmax2_nan(m, *reinterpret_cast<T *>(&bits));
  }
  const float2 mf = P::to_float2(m);
  float amax = fmaxf(fmaxf(mf.x, mf.y), 1e-10f);
  if (mf.x != mf.x || mf.y != mf.y)
    amax = __int_as_float(0x7fffffff);
  const float multiplier = __fdiv_rn(448.f, amax);
  std::uint32_t q[8];
#pragma unroll
  for (int i = 0; i < 16; i += 2) {
    const float2 a = P::to_float2(v[i]);
    const float2 b = P::to_float2(v[i + 1]);
    const std::uint32_t lo =
        e4m3x2(__fmul_rn(a.x, multiplier), __fmul_rn(a.y, multiplier));
    const std::uint32_t hi =
        e4m3x2(__fmul_rn(b.x, multiplier), __fmul_rn(b.y, multiplier));
    q[i / 2] = lo | (hi << 16);
  }
  const int chunk = (lane & 3) * 2;
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(
                   a_tile + tile_offset(row, chunk)),
               "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3])
               : "memory");
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(
                   a_tile + tile_offset(row, chunk + 1)),
               "r"(q[4]), "r"(q[5]), "r"(q[6]), "r"(q[7])
               : "memory");
  if ((lane & 3) == 0)
    scales[row] = __fmul_rn(amax, 1.f / 448.f);
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M,
          typename In, typename Out>
struct Kernel {
  static constexpr bool kFused = !std::is_same_v<In, std::uint8_t>;
  static constexpr int kMma = WARPS_M * WARPS_N;
  static constexpr int kQuantWarps = kFused ? 4 : 0;
  static constexpr int kThreads = (kMma + (kFused ? kQuantWarps : 1)) * 32;
  static constexpr int kATile = BM * kGroup;
  static constexpr int kBTile = BN * kGroup;
  static constexpr int kStage = kATile + kBTile;
  // 1024 bytes of alignment slack, the ring, the activation scales of each
  // slot, barriers.
  static constexpr int kSmem =
      1024 + STAGES * kStage + (kFused ? STAGES * BM * 4 : 0) + 16 * STAGES;
  static_assert(kSmem <= 101376, "GB10 shared memory per block");
};

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M,
          typename In, typename Out>
__global__ __launch_bounds__(
    Kernel<BM, BN, WARPS_M, WARPS_N, STAGES, GROUP_M, In, Out>::kThreads,
    1) void block_fp8_gemm_kernel(const __grid_constant__ CUtensorMap a_map,
                                  const __grid_constant__ CUtensorMap b_map,
                                  const __grid_constant__ GemmParams p) {
  using K = Kernel<BM, BN, WARPS_M, WARPS_N, STAGES, GROUP_M, In, Out>;
  constexpr bool kFused = K::kFused;
  constexpr int kMma = K::kMma;
  constexpr int kWarpM = BM / WARPS_M;
  constexpr int kWarpN = BN / WARPS_N;
  constexpr int kMT = kWarpM / 16; // m16 tiles per warp
  constexpr int kNT = kWarpN / 8;  // n8 tiles per warp
  constexpr int kATile = K::kATile;
  constexpr int kStage = K::kStage;
  constexpr int kScaleBlocks = (kWarpN + kGroup - 1) / kGroup;
  constexpr int kQuantRows = kFused ? BM / K::kQuantWarps : 8;
  static_assert(kWarpM % 16 == 0 && kWarpN % 16 == 0,
                "warp tile is m16 x n16 multiples");
  static_assert(kATile % 1024 == 0 && K::kBTile % 1024 == 0,
                "1024-byte aligned swizzled tiles");
  static_assert(kWarpN <= kGroup || kWarpN % kGroup == 0,
                "warp columns within scale blocks");
  static_assert(kQuantRows % 8 == 0, "eight rows per quantizer iteration");

  extern __shared__ std::uint8_t smem_raw[];
  // 128-byte TMA swizzle repeats every 1024 bytes: align the ring to 1024.
  const std::uint32_t raw = smem_addr(smem_raw);
  const std::uint32_t ring = (raw + 1023u) & ~1023u;
  float *const scale_ring =
      reinterpret_cast<float *>(smem_raw + (ring - raw) + STAGES * kStage);
  const std::uint32_t full_bar =
      ring + STAGES * kStage + (kFused ? STAGES * BM * 4 : 0);
  const std::uint32_t empty_bar = full_bar + STAGES * 8;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int groups = p.k / kGroup;
  const int tiles = p.tiles_m * p.tiles_n;

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) {
      // Fused: every quantizer lane arrives, plus the weight TMA's expect_tx
      // arrival.
      mbar_init(full_bar + 8 * s, kFused ? K::kQuantWarps * 32 + 1 : 1);
      mbar_init(empty_bar + 8 * s, kMma);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp >= kMma) {
    // Producers: the weight tiles by TMA from one lane and, fused, the
    // activation tiles quantized by the producer warps, STAGES groups ahead of
    // the MMA warps.
    const int producer = warp - kMma;
    const bool tma_lane = producer == 0 && lane == 0;
    if (!kFused && lane != 0)
      return;
    if (tma_lane) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(
                       reinterpret_cast<std::uint64_t>(&b_map))
                   : "memory");
      if (!kFused)
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(
                         reinterpret_cast<std::uint64_t>(&a_map))
                     : "memory");
    }
    int slot = 0;
    std::uint32_t phase = 1; // empty slots: the first wait on each passes
    // This CTA's (tile, group) steps in order; fused, the activation slice of
    // the next step is loaded (into registers) before the current one is
    // quantized.
    const int my_tiles =
        blockIdx.x < tiles ? (tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
    const int steps = my_tiles * groups;
    auto step_coords = [&](int step, int &tm, int &tn, int &g) {
      tile_coords(p, GROUP_M, blockIdx.x + (step / groups) * gridDim.x, tm, tn);
      g = step % groups;
    };
    auto fill = [&](int step) {
      int tm, tn, g;
      step_coords(step, tm, tn, g);
      // Coordinates before the wait: the load issues as soon as the slot frees
      // (ptxas otherwise sinks this integer math behind the wait, ~15% at 2
      // stages).
      asm volatile("" : "+r"(tm), "+r"(tn), "+r"(g));
      mbar_wait(empty_bar + 8 * slot, phase);
      const std::uint32_t full = full_bar + 8 * slot;
      const std::uint32_t a_tile = ring + slot * kStage;
      if (tma_lane) {
        mbar_expect_tx(full, kFused ? K::kBTile : kStage);
        tma_load_2d(a_tile + kATile, &b_map, full, g * kGroup, tn * BN);
        if (!kFused)
          tma_load_2d(a_tile, &a_map, full, g * kGroup, tm * BM);
      }
      return full;
    };
    auto advance = [&] {
      if (++slot == STAGES) {
        slot = 0;
        phase ^= 1;
      }
    };
    if constexpr (kFused) {
      // Each iteration's registers are refilled with the next step's slice as
      // soon as they are quantized, so the next group's loads are in flight
      // while this group is processed.
      constexpr int kIters = kQuantRows / 8;
      const int first = producer * kQuantRows + (lane >> 2);
      uint4 raw[kIters][4];
      auto load = [&](int step, int it) {
        int tm, tn, g;
        step_coords(step, tm, tn, g);
        load_slice<In>(p, tm * BM + first + it * 8, g, lane, raw[it]);
      };
      if (steps > 0) {
#pragma unroll
        for (int it = 0; it < kIters; ++it)
          load(0, it);
      }
      for (int step = 0; step < steps; ++step) {
        const std::uint32_t full = fill(step);
        const std::uint32_t a_tile = ring + slot * kStage;
        float *scales = scale_ring + slot * BM;
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
          quantize_slice<In>(raw[it], first + it * 8, a_tile, scales, lane);
          if (step + 1 < steps)
            load(step + 1, it);
        }
        mbar_arrive(full);
        advance();
      }
    } else {
      for (int step = 0; step < steps; ++step) {
        fill(step);
        advance();
      }
    }
    return;
  }

  // MMA warps.
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;
  const int quad =
      lane >> 3; // ldmatrix: lanes 8i..8i+7 address the rows of matrix i
  const int quad_row = lane & 7;
  int slot = 0;
  // A slot is released after the wait for the next one: by then every MMA of
  // the slot was issued, so its ldmatrix reads completed before the producers
  // may overwrite it.
  int held = -1;
  std::uint32_t phase = 0;
  Out *out = static_cast<Out *>(p.out);

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    int tm, tn;
    tile_coords(p, GROUP_M, t, tm, tn);
    const int m0 = tm * BM;
    const int n0 = tn * BN;
    // Scale rows of this thread: per m16 tile, rows lane / 4 and lane / 4 + 8
    // (clamped to the last row: rows past M are zero in the slot and never
    // stored).
    const int row0 = warp_m * kWarpM + (lane >> 2);
    const float *a_scale_row[kMT][2];
    if constexpr (!kFused) {
#pragma unroll
      for (int i = 0; i < kMT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = min(m0 + row0 + i * 16 + h * 8, p.m - 1);
          a_scale_row[i][h] =
              p.a_scale + static_cast<std::int64_t>(row) * p.sam;
        }
    }
    const float *b_scale_col =
        p.b_scale +
        static_cast<std::int64_t>((n0 + warp_n * kWarpN) / kGroup) * p.sbn;

    float acc[kMT][kNT][4];
#pragma unroll
    for (int i = 0; i < kMT; ++i)
#pragma unroll
      for (int j = 0; j < kNT; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c)
          acc[i][j][c] = 0.f;

    for (int g = 0; g < groups; ++g) {
      float sa[kMT][2], sb[kScaleBlocks];
#pragma unroll
      for (int j = 0; j < kScaleBlocks; ++j)
        sb[j] = __ldg(b_scale_col + static_cast<std::int64_t>(j) * p.sbn +
                      static_cast<std::int64_t>(g) * p.sbg);
      if constexpr (!kFused) {
#pragma unroll
        for (int i = 0; i < kMT; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            sa[i][h] =
                __ldg(a_scale_row[i][h] + static_cast<std::int64_t>(g) * p.sag);
      }

      mbar_wait(full_bar + 8 * slot, phase);
      if (held >= 0) {
        __syncwarp();
        if (lane == 0)
          mbar_arrive(empty_bar + 8 * held);
      }
      held = slot;
      if constexpr (kFused) {
        const float *scales = scale_ring + slot * BM;
#pragma unroll
        for (int i = 0; i < kMT; ++i)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            sa[i][h] = scales[row0 + i * 16 + h * 8];
      }
      const std::uint32_t a_tile = ring + slot * kStage;
      const std::uint32_t b_tile = a_tile + kATile;

      float part[kMT][kNT][4];
#pragma unroll
      for (int ks = 0; ks < kGroup / 32; ++ks) {
        std::uint32_t a_frag[kMT][4];
        std::uint32_t b_frag[kNT][2];
#pragma unroll
        for (int i = 0; i < kMT; ++i) {
          const int row = warp_m * kWarpM + i * 16 + (quad & 1) * 8 + quad_row;
          ldmatrix_x4(a_frag[i],
                      a_tile + tile_offset(row, 2 * ks + (quad >> 1)));
        }
#pragma unroll
        for (int j = 0; j < kNT; j += 2) {
          const int row = warp_n * kWarpN + j * 8 + (quad >> 1) * 8 + quad_row;
          std::uint32_t r[4];
          ldmatrix_x4(r, b_tile + tile_offset(row, 2 * ks + (quad & 1)));
          b_frag[j][0] = r[0];
          b_frag[j][1] = r[1];
          b_frag[j + 1][0] = r[2];
          b_frag[j + 1][1] = r[3];
        }
#pragma unroll
        for (int i = 0; i < kMT; ++i)
#pragma unroll
          for (int j = 0; j < kNT; ++j) {
            if (ks == 0)
              mma_e4m3_zero(part[i][j], a_frag[i], b_frag[j][0], b_frag[j][1]);
            else
              mma_e4m3(part[i][j], a_frag[i], b_frag[j][0], b_frag[j][1]);
          }
      }
      if (++slot == STAGES) {
        slot = 0;
        phase ^= 1;
      }

      // acc += part * (a_scale * w_scale).
#pragma unroll
      for (int i = 0; i < kMT; ++i)
#pragma unroll
        for (int j = 0; j < kNT; ++j) {
          const float b = sb[(j * 8) / kGroup];
          const float r0 = sa[i][0] * b;
          const float r1 = sa[i][1] * b;
          acc[i][j][0] = fmaf(part[i][j][0], r0, acc[i][j][0]);
          acc[i][j][1] = fmaf(part[i][j][1], r0, acc[i][j][1]);
          acc[i][j][2] = fmaf(part[i][j][2], r1, acc[i][j][2]);
          acc[i][j][3] = fmaf(part[i][j][3], r1, acc[i][j][3]);
        }
    }

#pragma unroll
    for (int i = 0; i < kMT; ++i) {
      const int row = m0 + warp_m * kWarpM + i * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < kNT; ++j) {
        const int col = n0 + warp_n * kWarpN + j * 8 + (lane & 3) * 2;
        if (row < p.m)
          store_pair(out + static_cast<std::int64_t>(row) * p.ldo + col,
                     acc[i][j][0], acc[i][j][1]);
        if (row + 8 < p.m)
          store_pair(out + static_cast<std::int64_t>(row + 8) * p.ldo + col,
                     acc[i][j][2], acc[i][j][3]);
      }
    }
  }
}

// A 2-D uint8 tensor map of `rows` x `cols` (row stride `stride` bytes) read in
// boxes of `box_rows` x 128 bytes with the 128-byte swizzle; rows past `rows`
// read as zeros.
inline CUtensorMap make_map(const void *base, std::int64_t rows,
                            std::int64_t cols, std::int64_t stride,
                            int box_rows) {
  using Encode = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                              void *, const cuuint64_t *, const cuuint64_t *,
                              const cuuint32_t *, const cuuint32_t *,
                              CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static const Encode encode = [] {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    // Ask for the 12.0 ABI (where cuTensorMapEncodeTiled appeared; unchanged
    // since): the unversioned query asks for the compiling runtime's version,
    // which an older driver (e.g. CUDA 13.0 under a 13.1 toolkit) rejects with
    // cudaErrorInvalidValue.
#if CUDART_VERSION >= 12050
    host::CUDA_CHECK(::cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &status));
#else
    host::CUDA_CHECK(::cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn,
                                               cudaEnableDefault, &status));
#endif
    host::RuntimeCheck(
        fn != nullptr && status == cudaDriverEntryPointSuccess,
        "block-FP8 sm12x GEMM: cuTensorMapEncodeTiled is unavailable");
    return reinterpret_cast<Encode>(fn);
  }();
  CUtensorMap map;
  const cuuint64_t dims[2] = {static_cast<cuuint64_t>(cols),
                              static_cast<cuuint64_t>(rows)};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(stride)};
  const cuuint32_t box[2] = {static_cast<cuuint32_t>(kGroup),
                             static_cast<cuuint32_t>(box_rows)};
  const cuuint32_t element_strides[2] = {1, 1};
  const CUresult result =
      encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(base),
             dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  host::RuntimeCheck(result == CUDA_SUCCESS,
                     "block-FP8 sm12x GEMM: cuTensorMapEncodeTiled failed (",
                     static_cast<int>(result), ")");
  return map;
}

template <int BM, int BN, int WARPS_M, int WARPS_N, int STAGES, int GROUP_M>
struct BlockFP8GemmSm12x {
  template <typename In, typename Out>
  static void launch(const CUtensorMap &a_map, const CUtensorMap &b_map,
                     const GemmParams &params, std::int64_t sms,
                     DLDevice device) {
    using K = Kernel<BM, BN, WARPS_M, WARPS_N, STAGES, GROUP_M, In, Out>;
    constexpr auto kernel = block_fp8_gemm_kernel<BM, BN, WARPS_M, WARPS_N,
                                                  STAGES, GROUP_M, In, Out>;
    const std::int64_t tiles =
        static_cast<std::int64_t>(params.tiles_m) * params.tiles_n;
    const unsigned blocks = static_cast<unsigned>(tiles < sms ? tiles : sms);
    host::set_smem_once<kernel>(K::kSmem);
    host::LaunchKernel(blocks, K::kThreads, device, K::kSmem)(kernel, a_map,
                                                              b_map, params);
  }

  // Checks of the weight, its scale grid and the output, shared by both entry
  // points.
  static GemmParams params(const tvm::ffi::TensorView b,
                           const tvm::ffi::TensorView b_scale,
                           const tvm::ffi::TensorView out, std::int64_t m,
                           std::int64_t k, DLDevice &device,
                           DLDataType &out_dtype) {
    using namespace host;
    auto N = SymbolicSize{"N"};
    auto NB = SymbolicSize{"NB"};
    auto G = SymbolicSize{"G"};
    auto LDB = SymbolicSize{"ldb"};
    auto LDO = SymbolicSize{"ldo"};
    auto SBN = SymbolicSize{"sbn"};
    auto SBG = SymbolicSize{"sbg"};
    auto device_ = SymbolicDevice{};
    auto out_dtype_ = SymbolicDType{};
    TensorMatcher({N, k})
        .with_strides({LDB, 1})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(b);
    TensorMatcher({NB, G})
        .with_strides({SBN, SBG})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device_)
        .verify(b_scale);
    TensorMatcher({m, N})
        .with_strides({LDO, 1})
        .with_dtype(out_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(out);
    const auto n = N.unwrap();
    RuntimeCheck(n % BN == 0 && k % kGroup == 0, "block-FP8 sm12x GEMM: N % ",
                 BN, " and K % 128");
    RuntimeCheck(G.unwrap() == k / kGroup && NB.unwrap() == n / kGroup,
                 "block-FP8 sm12x GEMM: scale grid");
    RuntimeCheck(LDB.unwrap() % 16 == 0 && LDO.unwrap() % 2 == 0,
                 "block-FP8 sm12x GEMM: row alignment");
    out_dtype = out_dtype_.unwrap();
    RuntimeCheck(out_dtype.bits == 16 && (out_dtype.code == kDLBfloat ||
                                          out_dtype.code == kDLFloat),
                 "block-FP8 sm12x GEMM: BF16 or FP16 output");
    device = device_.unwrap();
    return GemmParams{
        .x = nullptr,
        .a_scale = nullptr,
        .b_scale = static_cast<const float *>(b_scale.data_ptr()),
        .out = out.data_ptr(),
        .ldx = 0,
        .ldo = LDO.unwrap(),
        .sam = 0,
        .sag = 0,
        .sbn = SBN.unwrap(),
        .sbg = SBG.unwrap(),
        .m = static_cast<int>(m),
        .n = static_cast<int>(n),
        .k = static_cast<int>(k),
        .tiles_m = static_cast<int>((m + BM - 1) / BM),
        .tiles_n = static_cast<int>(n / BN),
    };
  }

  // out = q . w^T with caller-quantized E4M3 q [M, K] (as bytes) and FP32
  // scales [M, K / 128].
  static void run(const tvm::ffi::TensorView a,
                  const tvm::ffi::TensorView a_scale,
                  const tvm::ffi::TensorView b,
                  const tvm::ffi::TensorView b_scale,
                  const tvm::ffi::TensorView out, std::int64_t sms) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto G = SymbolicSize{"G"};
    auto LDA = SymbolicSize{"lda"};
    auto SAM = SymbolicSize{"sam"};
    auto SAG = SymbolicSize{"sag"};
    TensorMatcher({M, K})
        .with_strides({LDA, 1})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>()
        .verify(a);
    TensorMatcher({M, G})
        .with_strides({SAM, SAG})
        .with_dtype<float>()
        .with_device<kDLCUDA>()
        .verify(a_scale);
    DLDevice device;
    DLDataType out_dtype;
    auto params_ =
        params(b, b_scale, out, M.unwrap(), K.unwrap(), device, out_dtype);
    RuntimeCheck(G.unwrap() == K.unwrap() / kGroup && LDA.unwrap() % 16 == 0,
                 "block-FP8 sm12x GEMM: activation layout");
    if (params_.m == 0)
      return;
    params_.a_scale = static_cast<const float *>(a_scale.data_ptr());
    params_.sam = SAM.unwrap();
    params_.sag = SAG.unwrap();
    const auto a_map =
        make_map(a.data_ptr(), params_.m, params_.k, LDA.unwrap(), BM);
    const auto b_map =
        make_map(b.data_ptr(), params_.n, params_.k, b.stride(0), BN);
    if (out_dtype.code == kDLBfloat)
      launch<std::uint8_t, __nv_bfloat16>(a_map, b_map, params_, sms, device);
    else
      launch<std::uint8_t, __half>(a_map, b_map, params_, sms, device);
  }

  // out = quantize(x) . w^T with BF16/FP16 x [M, K] quantized in the kernel.
  static void run_fused(const tvm::ffi::TensorView x,
                        const tvm::ffi::TensorView b,
                        const tvm::ffi::TensorView b_scale,
                        const tvm::ffi::TensorView out, std::int64_t sms) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto LDX = SymbolicSize{"ldx"};
    auto x_dtype_ = SymbolicDType{};
    TensorMatcher({M, K})
        .with_strides({LDX, 1})
        .with_dtype(x_dtype_)
        .with_device<kDLCUDA>()
        .verify(x);
    DLDevice device;
    DLDataType out_dtype;
    auto params_ =
        params(b, b_scale, out, M.unwrap(), K.unwrap(), device, out_dtype);
    const auto x_dtype = x_dtype_.unwrap();
    RuntimeCheck(x_dtype.bits == 16 &&
                     (x_dtype.code == kDLBfloat || x_dtype.code == kDLFloat),
                 "block-FP8 sm12x GEMM: BF16 or FP16 activations");
    RuntimeCheck(LDX.unwrap() % 8 == 0 &&
                     reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0,
                 "block-FP8 sm12x GEMM: 16-byte aligned activation rows");
    if (params_.m == 0)
      return;
    params_.x = x.data_ptr();
    params_.ldx = LDX.unwrap();
    const auto b_map =
        make_map(b.data_ptr(), params_.n, params_.k, b.stride(0), BN);
    const bool bf16_in = x_dtype.code == kDLBfloat,
               bf16_out = out_dtype.code == kDLBfloat;
    if (bf16_in && bf16_out)
      launch<__nv_bfloat16, __nv_bfloat16>(b_map, b_map, params_, sms, device);
    else if (bf16_in)
      launch<__nv_bfloat16, __half>(b_map, b_map, params_, sms, device);
    else if (bf16_out)
      launch<__half, __nv_bfloat16>(b_map, b_map, params_, sms, device);
    else
      launch<__half, __half>(b_map, b_map, params_, sms, device);
  }
};

} // namespace
