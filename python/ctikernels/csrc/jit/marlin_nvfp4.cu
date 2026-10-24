// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (C) Marlin.2024 Elias Frantar
// Modified by Neural Magic
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from SGLang 0.5.22 baa9befa381511cd4d56f83980951e5a45fffc6d
// (`python/sglang/kernels/jit/csrc/gemm/marlin/`); see THIRD_PARTY_NOTICES.

// Upstream: marlin.cuh, marlin_template.h, dequant.h and gptq_marlin.cuh,
// themselves adapted from https://github.com/IST-DASLab/marlin.
//
// NVFP4 W4A16 GEMM: out[m, n] = round(alpha * sum_k x[m, k] * w[n, k]), where
// w = e2m1(code) * e4m3(block scale), one scale per 16 k. Changes from the
// source: only E2M1 weights with group-16 E4M3 scales (no act-order, zero
// points or integer/FP8 weights). The global scale is the caller's FP32 alpha,
// applied to the FP32 accumulator with one rounding (the source rounds to
// BF16/FP16 first and multiplies by a 16-bit global). The split-K reduction
// always runs in FP32 in slice order (no atomics), so results are identical
// run to run. The BF16 scale decode adds the exponent bias instead of placing
// bits, which is exact for every E4M3 scale (the source's decode loses scales
// below 2^-6). The weight repack reads the checkpoint layout directly.
//
// Exactness: codes decode to x * 2^-126 (BF16) / x * 2^-14 (FP16) and scales
// to s * 2^119 / s * 2^7, so every dequantized weight is w * 2^-7, exact (at
// most 6 significant bits, inside the normal range); the epilogue multiplies
// by alpha * 2^7. Tensor-core products of exact operands are exact; only the
// FP32 accumulation order differs from a reference GEMM.

#include <ctiengine/tensor.h>
#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

namespace marlin {

using device::div_ceil;

constexpr int kDefaultThreads = 256;
constexpr int kPipeStages = 4;
constexpr int kMinThreadN = 64;
constexpr int kMinThreadK = 64;
constexpr int kMaxThreadN = 256;
constexpr int kMaxThreadMBlocks = 4;
// Exponent offset of the dequantized weights (see the header comment).
constexpr float kWeightDomain = 128.0f;

template <typename T, int n> struct Vec {
  T elems[n];
  __device__ T &operator[](int i) { return elems[i]; }
};
using I4 = Vec<int, 4>;

template <typename scalar_t> struct Types;

template <> struct Types<half> {
  using scalar_t2 = half2;
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<half2, 1>;
  static __device__ half2 num2num2(half x) { return __half2half2(x); }
  static __device__ half2 pack(float a, float b) {
    return __halves2half2(__float2half_rn(a), __float2half_rn(b));
  }
};

template <> struct Types<nv_bfloat16> {
  using scalar_t2 = nv_bfloat162;
  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
  using FragC = Vec<float, 4>;
  using FragS = Vec<nv_bfloat162, 1>;
  static __device__ nv_bfloat162 num2num2(nv_bfloat16 x) {
    return __bfloat162bfloat162(x);
  }
  static __device__ nv_bfloat162 pack(float a, float b) {
    return __halves2bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
  }
};

__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                      bool pred = true) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n"
               "   .reg .pred p;\n"
               "   setp.ne.b32 p, %0, 0;\n"
               "   @p cp.async.cg.shared.global [%1], [%2], 16;\n"
               "}\n" ::"r"((int)pred),
               "r"(smem), "l"(glob_ptr));
}

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem),
               "l"(glob_ptr));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n> __device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// m16n8k16 tensor-core MMA, FP32 accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename Types<scalar_t>::FragA &a_frag,
                           const typename Types<scalar_t>::FragB &frag_b,
                           typename Types<scalar_t>::FragC &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  float *c = reinterpret_cast<float *>(&frag_c);
  if constexpr (std::is_same_v<scalar_t, half>) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                   "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
}

// The same MMA with A and B swapped (m <= 8 tiles compute the transpose).
template <typename scalar_t>
__device__ inline void mma_trans(const typename Types<scalar_t>::FragA &a_frag,
                                 const typename Types<scalar_t>::FragB &frag_b,
                                 const typename Types<scalar_t>::FragB &frag_b2,
                                 typename Types<scalar_t>::FragC &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
  const uint32_t *b2 = reinterpret_cast<const uint32_t *>(&frag_b2);
  float *c = reinterpret_cast<float *>(&frag_c);
  if constexpr (std::is_same_v<scalar_t, half>) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]),
                   "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]),
                   "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  }
}

template <int count, typename scalar_t>
__device__ inline void ldsm(typename Types<scalar_t>::FragA &frag_a,
                            const void *smem_ptr) {
  uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else {
    static_assert(count == 2);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  }
}

// Eight E2M1 codes (interleaved by the repack) to four BF16/FP16 pairs, placed
// into the exponent/mantissa bits without rebiasing: x * 2^-126 / x * 2^-14.
template <typename scalar_t2>
__device__ inline void dequant_fp4(uint32_t q, scalar_t2 *frag_b) {
  constexpr int kShift = std::is_same_v<scalar_t2, half2> ? 3 : 6;
  uint32_t out1 = (q & 0x80008000u) | ((q & 0x70007000u) >> kShift);
  q <<= 4;
  uint32_t out2 = (q & 0x80008000u) | ((q & 0x70007000u) >> kShift);
  frag_b[1] = *reinterpret_cast<const scalar_t2 *>(&out1);
  frag_b[0] = *reinterpret_cast<const scalar_t2 *>(&out2);
}

// Four scale bytes (the top byte of FP16(s * 2^7) without its sign) to two
// pairs: FP16 s * 2^7 by placement; BF16 s * 2^119 by adding 224 to the
// exponent (the byte's 5 exponent bits are FP16-biased).
template <typename scalar_t2>
__device__ inline void dequant_scales(uint32_t q, scalar_t2 *frag_s) {
  uint32_t out1, out2;
  if constexpr (std::is_same_v<scalar_t2, half2>) {
    out1 = (q & 0xFF00FF00u) >> 1;
    out2 = ((q << 8) & 0xFF00FF00u) >> 1;
  } else {
    out1 = ((q & 0xFF00FF00u) >> 4) + 0x70007000u;
    out2 = (((q << 8) & 0xFF00FF00u) >> 4) + 0x70007000u;
  }
  frag_s[1] = *reinterpret_cast<const scalar_t2 *>(&out1);
  frag_s[0] = *reinterpret_cast<const scalar_t2 *>(&out2);
}

template <typename scalar_t>
__device__ inline void scale(typename Types<scalar_t>::FragB &frag_b,
                             typename Types<scalar_t>::FragS &frag_s, int i) {
  auto s = Types<scalar_t>::num2num2(reinterpret_cast<scalar_t *>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// Row scales: four scale bytes (rows 2t, 2t + 1, 2t + 8, 2t + 9 of one
// 16-column group) applied to both column fragments of that group.
template <typename scalar_t>
__device__ inline void scale_rows(typename Types<scalar_t>::FragB &frag_b0,
                                  typename Types<scalar_t>::FragB &frag_b1,
                                  uint32_t q) {
  using scalar_t2 = typename Types<scalar_t>::scalar_t2;
  uint32_t lo, hi;
  if constexpr (std::is_same_v<scalar_t, half>) {
    lo = ((q & 0xFFu) << 7) | ((q & 0xFF00u) << 15);
    hi = (((q >> 16) & 0xFFu) << 7) | ((q >> 24) << 23);
  } else {
    lo = (((q & 0xFFu) << 4) | ((q & 0xFF00u) << 12)) + 0x70007000u;
    hi = ((((q >> 16) & 0xFFu) << 4) | ((q >> 24) << 20)) + 0x70007000u;
  }
  const scalar_t2 s_lo = *reinterpret_cast<const scalar_t2 *>(&lo);
  const scalar_t2 s_hi = *reinterpret_cast<const scalar_t2 *>(&hi);
  frag_b0[0] = __hmul2(frag_b0[0], s_lo);
  frag_b0[1] = __hmul2(frag_b0[1], s_hi);
  frag_b1[0] = __hmul2(frag_b1[0], s_lo);
  frag_b1[1] = __hmul2(frag_b1[1], s_hi);
}

// Wait until the lock reaches `count`, then hold it for this threadblock.
__device__ inline void barrier_acquire(int *lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release the lock to the next slice block; the last one resets it to 0.
__device__ inline void barrier_release(int *lock, bool reset) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" ::"l"(lock),
                 "r"(1));
  }
}

// kRowScales: one scale per (k row, 16 n columns) instead of per (16 k rows, n
// column); the input gradient of a checkpoint weight, whose blocks run along n.
template <typename scalar_t, int threads, int thread_m_blocks,
          int thread_n_blocks, int thread_k_blocks, bool m_block_size_8,
          int stages, bool kRowScales>
__global__ void gemm_kernel(const int4 *__restrict__ A,
                            const int4 *__restrict__ B, int4 *__restrict__ C,
                            int4 *__restrict__ C_tmp,
                            const int4 *__restrict__ scales_ptr,
                            const float *__restrict__ alpha_ptr, int prob_m,
                            int prob_n, int prob_k, int lda, int *locks) {
  // Each threadblock processes one "stripe" of B of (roughly) equal size,
  // possibly spanning several column slices of width 16 * thread_n_blocks;
  // e.g. with 3x3 tiles and 5 blocks:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // Blocks sharing a slice reduce through C_tmp in slice order.
  using T = Types<scalar_t>;
  using scalar_t2 = typename T::scalar_t2;
  using FragA = typename T::FragA;
  using FragB = typename T::FragB;
  using FragC = typename T::FragC;
  using FragS = typename T::FragS;

  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  constexpr int pack_factor = 8;
  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  static_assert(thread_k_blocks > 1, "one scale group per 16 k");

  const float out_scale = alpha_ptr[0] * kWeightDomain;

  // Larger GEMMs run several batch-64 problems side by side.
  int parallel = 1;
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);

  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;     // threadblock tiles in the current slice
  int slice_count = 0; // active threadblocks in the current slice
  int slice_idx;       // index of this threadblock in the slice, bottom to top

  int locks_off = 0;

  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * lda / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    slice_col = slice_col_par % n_tiles;
  }
  if (parallel * n_tiles >= gridDim.x) {
    // At most gridDim.x conflicting tiles.
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }

  auto init_slice = [&]() {
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (parallel * n_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1)
        locks_off++;
    } else {
      locks_off++;
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * lda / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
    }
  };
  init_slice();

  // A strides (int4 units).
  int a_gl_stride = lda / 8;
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B strides.
  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale strides: one byte per scale, one group per 16 k.
  int s_gl_stride = prob_n / 16;
  constexpr int s_sh_stride = thread_n_blocks;
  constexpr int s_tb_groups = thread_k_blocks;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  constexpr int k_iter_size = 16 * thread_k_blocks / b_sh_wr_iters;

  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd =
      b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_wr = threadIdx.x;
  auto b_sh_rd = threadIdx.x;

  int s_gl_rd = s_gl_stride * (thread_k_blocks * slice_row) +
                s_sh_stride * slice_col + threadIdx.x;
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;
  int s_sh_rd =
      8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;

  // Threads past the tile or batch do not read.
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // XOR layout keeping A tile writes and fragment reads bank-conflict free.
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] =
          transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  const int4 *B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  extern __shared__ int4 sh[];
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4 *sh_b = sh;
  int4 *sh_red = sh;
  int4 *sh_s = sh_b + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_s_size = stages * s_sh_stage;
  static_assert(thread_m_blocks * 16 * thread_n_blocks * 16 / 8 <=
                stages * b_sh_stage);
  int4 *sh_a = sh_s + sh_s_size;

  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float *>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and scale tiles into pipeline slot
  // `pipe`.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }
      int4 *sh_s_stage = sh_s + s_sh_stage * pipe;
      for (int i = 0; i < s_tb_groups; i++) {
        if (s_sh_wr_pred)
          cp_async4(&sh_s_stage[i * s_sh_stride + s_sh_wr],
                    &scales_ptr[s_gl_rd]);
        s_gl_rd += s_gl_stride;
      }
    }
    // Fence even while winding down so the wait below stays correct.
    cp_async_fence();
  };

  // Only stages - 2 fetches are in flight: the slot being read must be fully
  // consumed before it is refilled.
  auto wait_for_stage = [&]() {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&](int k, int pipe) {
    int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, scalar_t>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4 *>(
        &sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    int warp_row = (threadIdx.x / 32) / (thread_n_blocks / 4);
    int group = (warp_row * 16 + k_iter_size * (k % b_sh_wr_iters)) / 16;
    int4 *sh_s_stage = sh_s + s_sh_stage * pipe;
    if constexpr (kRowScales) {
      // 16 bytes per (k subtile, 16-column group), row r at byte r; this
      // thread's rows 2t, 2t + 1, 2t + 8, 2t + 9 for the warp's 4 groups.
      const int t = threadIdx.x % 4;
      const int warp_n = (threadIdx.x / 32) % (thread_n_blocks / 4);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t *words = reinterpret_cast<const uint32_t *>(
            &sh_s_stage[group * s_sh_stride + 4 * warp_n + j]);
        const uint32_t lo = (words[t / 2] >> (16 * (t % 2))) & 0xFFFFu;
        const uint32_t hi = (words[2 + t / 2] >> (16 * (t % 2))) & 0xFFFFu;
        reinterpret_cast<uint32_t *>(&frag_s[k % 2])[j] = lo | (hi << 16);
      }
    } else {
      reinterpret_cast<int2 *>(&frag_s[k % 2])[0] = reinterpret_cast<int2 *>(
          sh_s_stage)[s_sh_rd + group * (2 * s_sh_stride)];
    }
  };

  auto matmul = [&](int k) {
    int k2 = k % 2;
    if constexpr (!kRowScales) {
      uint32_t s_quant_0 = reinterpret_cast<uint32_t *>(frag_s[k2])[0];
      uint32_t s_quant_1 = reinterpret_cast<uint32_t *>(frag_s[k2])[1];
      dequant_scales(s_quant_0, reinterpret_cast<scalar_t2 *>(&frag_s[k2]));
      dequant_scales(s_quant_1, reinterpret_cast<scalar_t2 *>(&frag_s[k2]) + 2);
    }
    // m is the inner loop to overlap dequantization with the MMAs.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      uint32_t b_quant_1 = frag_b_quant[k2][j];
      uint32_t b_quant_0 = b_quant_1 << 8;
      dequant_fp4(b_quant_0, reinterpret_cast<scalar_t2 *>(&frag_b0));
      dequant_fp4(b_quant_1, reinterpret_cast<scalar_t2 *>(&frag_b1));
      if constexpr (kRowScales) {
        scale_rows<scalar_t>(frag_b0, frag_b1,
                             reinterpret_cast<uint32_t *>(frag_s[k2])[j]);
      } else {
        scale<scalar_t>(frag_b0, frag_s[k2][j], 0);
        scale<scalar_t>(frag_b1, frag_s[k2][j], 1);
      }
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<scalar_t>(frag_a[k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
        } else {
          mma<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
          mma<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
        }
      }
    }
  };

  // Warps splitting k inside the block reduce their partial sums in shared
  // memory (logarithmic tree).
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) +
                      (threadIdx.x % b_sh_stride);
#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float *c_rd = reinterpret_cast<float *>(
                    &sh_red[red_sh_delta * j + red_sh_rd]);
                float *c_wr = reinterpret_cast<float *>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] =
                  reinterpret_cast<int4 *>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i += (m_block_size_8 ? 2 : 1)) {
            float *c_rd = reinterpret_cast<float *>(
                &sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + i][j] +=
                  c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Blocks sharing a column slice add their FP32 partials through C_tmp, one
  // after another in slice order.
  auto global_reduce = [&](bool first, bool last) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;
    int c_cur_offset = locks_off * c_size;
    if (threadIdx.x >= active_threads)
      return;
    if (!first) {
      float *frag_c_ptr = reinterpret_cast<float *>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];
        float *sh_c_ptr = reinterpret_cast<float *>(&sh_red[threadIdx.x]);
#pragma unroll
        for (int f = 0; f < 4; f++)
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
      }
    }
    if (!last) {
      int4 *frag_c_ptr = reinterpret_cast<int4 *>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1))
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
    }
  };

  // Scale, round once and write the slice result through shared memory for
  // coalesced global stores.
  auto write_result = [&]() {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += 32 * (threadIdx.x / 32);
    }
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&](int idx, float c0, float c1) {
      scalar_t2 res = T::pack(c0 * out_scale, c1 * out_scale);
      if constexpr (m_block_size_8) {
        ((scalar_t *)sh_red)[idx] = res.x;
        ((scalar_t *)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((scalar_t2 *)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                  frag_c[i][j][0][1]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                  frag_c[i][j][0][3]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                  frag_c[i][j][1][1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                  frag_c[i][j][1][3]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  auto start_pipes = [&]() {
#pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters)
    start_pipes();

  // Main loop, unrolled over the fetch and register pipelines so every shared
  // memory access is static; both have even length, so each iteration starts
  // at index 0.
  while (slice_iters) {
#pragma unroll
    for (int pipe = 0; pipe < stages;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      thread_block_reduce();
      if (slice_count > 1) {
        barrier_acquire(&locks[locks_off], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[locks_off], last);
      }
      if (last)
        write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
#pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// One thread per output int32: eight 4-bit codes of a 16 x 64 (k x n) tile in
// the tensor-core fragment order the kernel reads (Marlin's FP4 repack, here
// reading the checkpoint bytes directly: [N, K / 2] with blocks along k, or
// kTransposed [K, N / 2] with blocks along n). Codes whose scale is zero become
// zero (the BF16 scale decode never yields zero).
template <bool kTransposed>
__global__ void repack_weight_kernel(const uint8_t *__restrict__ data,
                                     const uint8_t *__restrict__ scale,
                                     uint32_t *__restrict__ out, int n, int k) {
  const int64_t o = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t row_ints = 2 * int64_t(n);
  if (o >= row_ints * (k / 16))
    return;
  const int kt = int(o / row_ints);
  const int rem = int(o % row_ints);
  const int t = rem % 128;
  const int th = t / 4, warp = t % 4;
  const int col = (rem / 128) * 64 + warp * 16 + th / 4;
  const int row0 = kt * 16 + (th % 4) * 2;
  constexpr int kOffsets[4] = {0, 1, 8, 9};
  uint32_t vals[8];
#pragma unroll
  for (int half = 0; half < 2; half++) {
    const int c = col + 8 * half;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int r = row0 + kOffsets[i];
      uint8_t byte;
      bool live, high;
      if constexpr (kTransposed) {
        byte = data[int64_t(r) * (n / 2) + c / 2];
        live = scale[int64_t(r) * (n / 16) + c / 16] != 0;
        high = c % 2;
      } else {
        byte = data[int64_t(c) * (k / 2) + r / 2];
        live = scale[int64_t(c) * (k / 16) + kt] != 0;
        high = r % 2;
      }
      vals[4 * half + i] = live ? (high ? byte >> 4 : byte & 15) : 0;
    }
  }
  constexpr int kPack[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t res = 0;
#pragma unroll
  for (int i = 0; i < 8; i++)
    res |= vals[kPack[i]] << (i * 4);
  out[o] = res;
}

// Scales [N, K / 16] E4M3 -> [K / 16, N] bytes in the kernel's read order: the
// top byte of FP16(s * 2^7) without its sign, columns permuted within each
// group of 64.
__global__ void repack_scale_kernel(const __nv_fp8_e4m3 *__restrict__ scale,
                                    uint8_t *__restrict__ out, int n, int k) {
  const int64_t o = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= int64_t(n) * (k / 16))
    return;
  const int g = int(o / n);
  const int col = int(o % n);
  const int q = col % 64;
  constexpr int kPair[4] = {0, 2, 1, 3};
  const int p = 4 * (q / 4) + kPair[q % 4];
  const int src = (col / 64) * 64 + p / 8 + 8 * (p % 8);
  const float s = float(scale[int64_t(src) * (k / 16) + g]);
  const uint16_t bits = __half_as_ushort(__float2half_rn(s * 128.0f));
  out[o] = uint8_t((bits >> 7) & 0xFF);
}

// Row scales [K, N / 16] E4M3 -> [K / 16, N / 16, 16] bytes (16 k rows of one
// 16-column group contiguous), same byte encoding as above.
__global__ void repack_row_scale_kernel(const __nv_fp8_e4m3 *__restrict__ scale,
                                        uint8_t *__restrict__ out, int n,
                                        int k) {
  const int64_t o = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (o >= int64_t(n) * (k / 16))
    return;
  const int r = int(o % 16);
  const int64_t group = o / 16;
  const int j = int(group % (n / 16));
  const int kt = int(group / (n / 16));
  const float s = float(scale[int64_t(kt * 16 + r) * (n / 16) + j]);
  const uint16_t bits = __half_as_ushort(__float2half_rn(s * 128.0f));
  out[o] = uint8_t((bits >> 7) & 0xFF);
}

struct ThreadConfig {
  int thread_k;
  int thread_n;
  int num_threads;
};

constexpr ThreadConfig kSmallBatchConfigs[] = {
    {128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
constexpr ThreadConfig kLargeBatchConfigs[] = {
    {64, 256, 256}, {64, 128, 128}, {128, 64, 128}};

inline int shared_bytes(ThreadConfig c, int thread_m_blocks) {
  const int tb_m = thread_m_blocks * 16;
  const int sh_a = kPipeStages * tb_m * c.thread_k * 2;
  const int sh_b = kPipeStages * (c.thread_k * c.thread_n / 8) * 4;
  const int sh_red = tb_m * (c.thread_n + 8);
  const int sh_s =
      host::div_ceil(c.thread_k, 16) * c.thread_n * 2 * kPipeStages;
  return std::max(sh_b, sh_red) + sh_a + sh_s;
}

inline bool valid(ThreadConfig c, int thread_m_blocks, int n, int k,
                  int max_shared) {
  return k % c.thread_k == 0 && n % c.thread_n == 0 &&
         shared_bytes(c, thread_m_blocks) <= max_shared;
}

using KernelFn = void (*)(const int4 *, const int4 *, int4 *, int4 *,
                          const int4 *, const float *, int, int, int, int,
                          int *);

#define CTI_MARLIN_IF(M_BLOCKS, N_BLOCKS, K_BLOCKS, M8, THREADS)               \
  if (thread_m_blocks == M_BLOCKS && thread_n_blocks == N_BLOCKS &&            \
      thread_k_blocks == K_BLOCKS && m_block_size_8 == M8 &&                   \
      num_threads == THREADS)                                                  \
    return gemm_kernel<scalar_t, THREADS, M_BLOCKS, N_BLOCKS, K_BLOCKS, M8,    \
                       kPipeStages, kRowScales>;

#define CTI_MARLIN_M1(N_BLOCKS, K_BLOCKS, THREADS)                             \
  CTI_MARLIN_IF(1, N_BLOCKS, K_BLOCKS, true, THREADS)                          \
  CTI_MARLIN_IF(1, N_BLOCKS, K_BLOCKS, false, THREADS)

#define CTI_MARLIN_M234(N_BLOCKS, K_BLOCKS, THREADS)                           \
  CTI_MARLIN_IF(2, N_BLOCKS, K_BLOCKS, false, THREADS)                         \
  CTI_MARLIN_IF(3, N_BLOCKS, K_BLOCKS, false, THREADS)                         \
  CTI_MARLIN_IF(4, N_BLOCKS, K_BLOCKS, false, THREADS)

template <typename scalar_t, bool kRowScales>
KernelFn select_kernel(int thread_m_blocks, int thread_n_blocks,
                       int thread_k_blocks, bool m_block_size_8,
                       int num_threads) {
  CTI_MARLIN_M1(8, 8, 256)
  CTI_MARLIN_M1(8, 4, 128)
  CTI_MARLIN_M1(4, 8, 128)
  CTI_MARLIN_M234(16, 4, 256)
  CTI_MARLIN_M234(8, 4, 128)
  CTI_MARLIN_M234(4, 8, 128)
  return nullptr;
}

#undef CTI_MARLIN_M234
#undef CTI_MARLIN_M1
#undef CTI_MARLIN_IF

} // namespace marlin

template <typename scalar_t> struct MarlinNVFP4 {
  // x [M, K] (unit inner stride, row stride a multiple of 8), weight/scales
  // from `repack`, alpha FP32 [1], out [M, N] contiguous; c_tmp FP32 of at
  // least SMs * 64 * 256 and locks int32 [SMs] that are zero (and left zero).
  // `gemm_rows` takes the row-scale layout of `repack_rows`.
  static void gemm(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                   tvm::ffi::TensorView scales, tvm::ffi::TensorView alpha,
                   tvm::ffi::TensorView out, tvm::ffi::TensorView c_tmp,
                   tvm::ffi::TensorView locks) {
    run<false>(x, weight, scales, alpha, out, c_tmp, locks);
  }
  static void gemm_rows(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                        tvm::ffi::TensorView scales, tvm::ffi::TensorView alpha,
                        tvm::ffi::TensorView out, tvm::ffi::TensorView c_tmp,
                        tvm::ffi::TensorView locks) {
    run<true>(x, weight, scales, alpha, out, c_tmp, locks);
  }

  template <bool kRowScales>
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView weight,
                  tvm::ffi::TensorView scales, tvm::ffi::TensorView alpha,
                  tvm::ffi::TensorView out, tvm::ffi::TensorView c_tmp,
                  tvm::ffi::TensorView locks) {
    using namespace host;
    const int m = static_cast<int>(x.size(0));
    const int k = static_cast<int>(x.size(1));
    const int n = static_cast<int>(out.size(1));
    int lda = static_cast<int>(x.stride(0));
    if (m == 0)
      return;
    const DLDevice device = x.device();
    const cudaStream_t stream = LaunchKernel::resolve_device(device);
    int sms = 0, max_shared = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                      device.device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared,
                                      cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      device.device_id));
    RuntimeCheck(locks.size(0) >= sms, "marlin: locks hold fewer than ", sms,
                 " entries");
    RuntimeCheck(c_tmp.size(0) >= int64_t(sms) * 64 * marlin::kMaxThreadN,
                 "marlin: c_tmp is too small");

    auto a_ptr = static_cast<const int4 *>(x.data_ptr());
    auto c_ptr = static_cast<int4 *>(out.data_ptr());
    // Rows run in chunks of up to 64 * max_par so the stripes stay balanced.
    const int max_par = n <= 4096 ? 16 * 8 : 16;
    int max_m_blocks = marlin::kMaxThreadMBlocks;
    int rest = m;
    while (rest) {
      int par = std::min(rest / (max_m_blocks * 16), max_par);
      int rows = par > 0 ? par * max_m_blocks * 16 : rest;
      int thread_m_blocks = std::min(div_ceil(rows, 16), max_m_blocks);
      bool m8 = rows <= 8;
      const marlin::ThreadConfig *configs = thread_m_blocks > 1
                                                ? marlin::kLargeBatchConfigs
                                                : marlin::kSmallBatchConfigs;
      marlin::KernelFn kernel = nullptr;
      marlin::ThreadConfig chosen{};
      for (int i = 0; i < 3 && kernel == nullptr; i++) {
        if (!marlin::valid(configs[i], thread_m_blocks, n, k, max_shared))
          continue;
        chosen = configs[i];
        kernel = marlin::select_kernel<scalar_t, kRowScales>(
            thread_m_blocks, chosen.thread_n / 16, chosen.thread_k / 16, m8,
            chosen.num_threads);
      }
      if (kernel == nullptr) {
        RuntimeCheck(max_m_blocks > 1, "marlin: no kernel for M=", m, " N=", n,
                     " K=", k);
        max_m_blocks--;
        continue;
      }
      CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
      LaunchKernel(sms, chosen.num_threads, stream, max_shared)(
          kernel, a_ptr, static_cast<const int4 *>(weight.data_ptr()), c_ptr,
          static_cast<int4 *>(c_tmp.data_ptr()),
          static_cast<const int4 *>(scales.data_ptr()),
          static_cast<const float *>(alpha.data_ptr()), rows, n, k, lda,
          static_cast<int *>(locks.data_ptr()));
      a_ptr += int64_t(rows) * (lda / 8);
      c_ptr += int64_t(rows) * (n / 8);
      rest -= rows;
    }
  }
};

// data uint8 [N, K / 2], scale E4M3 [N, K / 16] (both contiguous) -> weight
// int32 [K / 16, 2 N] and scales uint8 [K / 16, N].
inline void marlin_nvfp4_repack(tvm::ffi::TensorView data,
                                tvm::ffi::TensorView scale,
                                tvm::ffi::TensorView weight,
                                tvm::ffi::TensorView scales) {
  using namespace host;
  const int n = static_cast<int>(data.size(0));
  const int k = static_cast<int>(data.size(1)) * 2;
  const cudaStream_t stream = LaunchKernel::resolve_device(data.device());
  constexpr int kThreads = 256;
  const int64_t words = int64_t(k / 16) * 2 * n;
  LaunchKernel(div_ceil(words, kThreads), kThreads,
               stream)(marlin::repack_weight_kernel<false>,
                       static_cast<const uint8_t *>(data.data_ptr()),
                       static_cast<const uint8_t *>(scale.data_ptr()),
                       static_cast<uint32_t *>(weight.data_ptr()), n, k);
  const int64_t bytes = int64_t(k / 16) * n;
  LaunchKernel(div_ceil(bytes, kThreads), kThreads,
               stream)(marlin::repack_scale_kernel,
                       static_cast<const __nv_fp8_e4m3 *>(scale.data_ptr()),
                       static_cast<uint8_t *>(scales.data_ptr()), n, k);
}

// The input-gradient layout of the same checkpoint weight: data uint8 [K, N /
// 2], scale E4M3 [K, N / 16] (blocks along n) -> weight int32 [K / 16, 2 N] and
// row scales uint8 [K / 16, N].
inline void marlin_nvfp4_repack_rows(tvm::ffi::TensorView data,
                                     tvm::ffi::TensorView scale,
                                     tvm::ffi::TensorView weight,
                                     tvm::ffi::TensorView scales) {
  using namespace host;
  const int k = static_cast<int>(data.size(0));
  const int n = static_cast<int>(data.size(1)) * 2;
  const cudaStream_t stream = LaunchKernel::resolve_device(data.device());
  constexpr int kThreads = 256;
  const int64_t words = int64_t(k / 16) * 2 * n;
  LaunchKernel(div_ceil(words, kThreads), kThreads,
               stream)(marlin::repack_weight_kernel<true>,
                       static_cast<const uint8_t *>(data.data_ptr()),
                       static_cast<const uint8_t *>(scale.data_ptr()),
                       static_cast<uint32_t *>(weight.data_ptr()), n, k);
  const int64_t bytes = int64_t(k / 16) * n;
  LaunchKernel(div_ceil(bytes, kThreads), kThreads,
               stream)(marlin::repack_row_scale_kernel,
                       static_cast<const __nv_fp8_e4m3 *>(scale.data_ptr()),
                       static_cast<uint8_t *>(scales.data_ptr()), n, k);
}

} // namespace
