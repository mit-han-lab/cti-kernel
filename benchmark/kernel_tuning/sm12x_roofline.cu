// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Reference: CUTLASS/CuTe warp-specialized TMA (BSD-3-Clause); public PTX ISA.
// Original pipeline implementation; design references only, no adapted code.
// Microbenchmarks of a compute capability 12.x GPU for the block-FP8 GEMM
// roofline (driver: sm12x_roofline.py): mma.sync throughput and latency, DRAM
// and L2 streaming, shared-memory banks, cp.async and TMA (cp.async.bulk)
// fills.

#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

#define DEVI __device__ __forceinline__

DEVI void mma_e4m3_f32(float (&d)[4], const uint32_t (&a)[4],
                       const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}
DEVI void mma_e4m3_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                       const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, "
               "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}
DEVI void mma_bf16_f32(float (&d)[4], const uint32_t (&a)[4],
                       const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}
DEVI void mma_f16_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                      const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
               "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

// KIND 0: e4m3->f32, 1: e4m3->f16, 2: bf16->f32, 3: f16->f16. ILP independent
// accumulators per warp.
template <int KIND, int ILP>
__global__ void mma_loop(float *out, int iters, long long *cycles) {
  uint32_t a[4], b[2];
  for (int i = 0; i < 4; ++i)
    a[i] = 0x38383838u ^ (threadIdx.x * 7 + i);
  for (int i = 0; i < 2; ++i)
    b[i] = 0x30303030u ^ (threadIdx.x * 3 + i);
  float df[ILP][4];
  uint32_t dh[ILP][2];
  for (int j = 0; j < ILP; ++j) {
    for (int i = 0; i < 4; ++i)
      df[j][i] = 0.f;
    dh[j][0] = dh[j][1] = 0;
  }
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      if (KIND == 0)
        mma_e4m3_f32(df[j], a, b);
      else if (KIND == 1)
        mma_e4m3_f16(dh[j], a, b);
      else if (KIND == 2)
        mma_bf16_f32(df[j], a, b);
      else
        mma_f16_f16(dh[j], a, b);
    }
  }
  long long t1 = clock64();
  float s = 0.f;
  for (int j = 0; j < ILP; ++j)
    s += df[j][0] + df[j][1] + df[j][2] + df[j][3] +
         __uint_as_float(dh[j][0] ^ dh[j][1]);
  if (s == 12345.f)
    out[0] = s;
  if (threadIdx.x == 0 && blockIdx.x == 0)
    cycles[0] = t1 - t0;
}

// MMA: returns {seconds, cycles of block 0 warp 0}.
std::vector<double> mma_bench(int kind, int ilp, int blocks, int warps,
                              int iters) {
  auto out =
      torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto cyc =
      torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto launch = [&]() {
#define L(K, I)                                                                \
  if (kind == K && ilp == I)                                                   \
  mma_loop<K, I><<<blocks, warps * 32>>>(out.data_ptr<float>(), iters,         \
                                         (long long *)cyc.data_ptr<int64_t>())
    L(0, 1);
    L(0, 2);
    L(0, 4);
    L(0, 8);
    L(1, 1);
    L(1, 2);
    L(1, 4);
    L(1, 8);
    L(2, 1);
    L(2, 2);
    L(2, 4);
    L(2, 8);
    L(3, 1);
    L(3, 4);
    L(3, 8);
#undef L
  };
  launch();
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  launch();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return {ms * 1e-3, (double)cyc.cpu().item<int64_t>()};
}

// DRAM read: 16-byte loads, grid-stride.
__global__ void read_kernel(const int4 *__restrict__ p, long long n, int *out) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    int4 v = __ldcs(p + i); // streaming
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x1234567)
    out[0] = 1;
}
__global__ void read_kernel_ca(const int4 *__restrict__ p, long long n,
                               int *out) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x) {
    int4 v = p[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x1234567)
    out[0] = 1;
}
__global__ void copy_kernel(const int4 *__restrict__ p, int4 *__restrict__ q,
                            long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x)
    q[i] = p[i];
}
__global__ void write_kernel(int4 *__restrict__ q, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
       i += (long long)gridDim.x * blockDim.x)
    q[i] = make_int4(i, 0, 0, 0);
}

// mode 0 read(cs), 1 read(ca), 2 copy, 3 write. Bytes of src; returns seconds
// (median-free: best of reps).
double mem_bench(torch::Tensor src, torch::Tensor dst, int mode, int blocks,
                 int threads, int reps) {
  long long n = src.numel() * src.element_size() / 16;
  auto flag =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto go = [&]() {
    if (mode == 0)
      read_kernel<<<blocks, threads>>>((const int4 *)src.data_ptr(), n,
                                       flag.data_ptr<int>());
    else if (mode == 1)
      read_kernel_ca<<<blocks, threads>>>((const int4 *)src.data_ptr(), n,
                                          flag.data_ptr<int>());
    else if (mode == 2)
      copy_kernel<<<blocks, threads>>>((const int4 *)src.data_ptr(),
                                       (int4 *)dst.data_ptr(), n);
    else
      write_kernel<<<blocks, threads>>>((int4 *)dst.data_ptr(), n);
  };
  go();
  cudaDeviceSynchronize();
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(e0);
    go();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, (double)ms * 1e-3);
  }
  return best;
}

// Shared memory: each thread reads 32-bit words at word index (lane * STRIDE) %
// 8192, or 128-bit at lane*4 (+swizzle).
template <int STRIDE, bool V4>
__global__ void smem_kernel(int iters, int *out, long long *cycles) {
  __shared__ __align__(16) uint32_t s[8192];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x)
    s[i] = i * 2654435761u;
  __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t acc = 0;
  long long t0 = clock64();
  if (V4) {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(s) +
                    ((warp * 512 + lane * 16 * STRIDE) % 32768);
    for (int it = 0; it < iters; ++it) {
      uint32_t x, y, z, w;
      uint32_t addr = base ^ ((it & 7) << 9);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                   : "r"(addr));
      acc ^= x ^ y ^ z ^ w;
    }
  } else {
    uint32_t base = (uint32_t)__cvta_generic_to_shared(s) +
                    4 * ((warp * 64 + lane * STRIDE) % 8192);
    for (int it = 0; it < iters; ++it) {
      uint32_t x;
      uint32_t addr = base ^ ((it & 7) << 9);
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr));
      acc ^= x;
    }
  }
  long long t1 = clock64();
  if (acc == 0x7654321)
    out[0] = acc;
  if (threadIdx.x == 0 && blockIdx.x == 0)
    cycles[0] = t1 - t0;
}
// ldmatrix x4 (the MMA operand path): each lane gives a 16-byte row address;
// rows conflict-free.
__global__ void ldmatrix_kernel(int iters, int *out, long long *cycles) {
  __shared__ __align__(16) uint32_t s[8192];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x)
    s[i] = i;
  __syncthreads();
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  uint32_t base = (uint32_t)__cvta_generic_to_shared(s) +
                  ((warp * 512 + lane * 16) % 32768);
  uint32_t acc = 0;
  long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
    uint32_t a, b, c, d;
    uint32_t addr = base ^ ((it & 7) << 9);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "r"(addr));
    acc ^= a ^ b ^ c ^ d;
  }
  long long t1 = clock64();
  if (acc == 0x7654321)
    out[0] = acc;
  if (threadIdx.x == 0 && blockIdx.x == 0)
    cycles[0] = t1 - t0;
}

std::vector<double> smem_bench(int stride, bool v4, int ldm, int blocks,
                               int threads, int iters) {
  auto flag =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto cyc =
      torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto go = [&]() {
    int *f = flag.data_ptr<int>();
    long long *c = (long long *)cyc.data_ptr<int64_t>();
    if (ldm) {
      ldmatrix_kernel<<<blocks, threads>>>(iters, f, c);
      return;
    }
#define S(T)                                                                   \
  if (stride == T) {                                                           \
    if (v4)                                                                    \
      smem_kernel<T, true><<<blocks, threads>>>(iters, f, c);                  \
    else                                                                       \
      smem_kernel<T, false><<<blocks, threads>>>(iters, f, c);                 \
  }
    S(1) S(2) S(4) S(8) S(16) S(32)
#undef S
  };
  go();
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  go();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  return {ms * 1e-3, (double)cyc.cpu().item<int64_t>()};
}

// cp.async: each block streams its slice of a DRAM buffer into a ring of STAGES
// 16-KiB smem tiles.
__global__ void cpasync_kernel(const char *__restrict__ src, long long bytes,
                               int stages, int *out) {
  extern __shared__ __align__(128) char ring[];
  const int TILE = 16384;
  long long tiles = bytes / TILE;
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(ring);
  int issued = 0;
  uint32_t acc = 0;
  for (long long t = blockIdx.x; t < tiles; t += gridDim.x) {
    int slot = issued % stages;
    const char *g = src + t * TILE;
    for (int off = threadIdx.x * 16; off < TILE; off += blockDim.x * 16)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       sbase + slot * TILE + off),
                   "l"(g + off));
    asm volatile("cp.async.commit_group;");
    ++issued;
    if (issued >= stages) {
      asm volatile("cp.async.wait_group %0;" ::"n"(2));
      __syncthreads();
      acc ^= *(const uint32_t *)(ring + ((issued - stages) % stages) * TILE +
                                 threadIdx.x * 4);
      __syncthreads();
    }
  }
  asm volatile("cp.async.wait_all;");
  if (acc == 0x7654321)
    out[0] = acc;
}

// TMA 1-D bulk copies (cp.async.bulk) completing on an mbarrier; one elected
// thread issues.
__global__ void bulk_kernel(const char *__restrict__ src, long long bytes,
                            int stages, int *out) {
  extern __shared__ __align__(128) char ring[];
  __shared__ __align__(8) uint64_t bar[8];
  const int TILE = 16384;
  long long tiles = bytes / TILE;
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(ring);
  if (threadIdx.x == 0)
    for (int i = 0; i < stages; ++i)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
          (uint32_t)__cvta_generic_to_shared(&bar[i])));
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();
  uint32_t phase[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t acc = 0;
  long long mine = (tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
  // prologue
  for (int i = 0; i < stages && i < mine; ++i) {
    if (threadIdx.x == 0) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[i]);
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),
          "r"(TILE));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx:"
                   ":bytes [%0], [%1], %2, [%3];" ::"r"(sbase + i * TILE),
                   "l"(src + (blockIdx.x + (long long)i * gridDim.x) * TILE),
                   "r"(TILE), "r"(b)
                   : "memory");
    }
  }
  for (long long j = 0; j < mine; ++j) {
    int slot = j % stages;
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[slot]);
    uint32_t done = 0;
    while (!done) {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 "
                   "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done)
                   : "r"(b), "r"(phase[slot]));
    }
    phase[slot] ^= 1;
    acc ^= *(const uint32_t *)(ring + slot * TILE + threadIdx.x * 4);
    __syncthreads();
    long long nxt = j + stages;
    if (threadIdx.x == 0 && nxt < mine) {
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),
          "r"(TILE));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx:"
                   ":bytes [%0], [%1], %2, [%3];" ::"r"(sbase + slot * TILE),
                   "l"(src + (blockIdx.x + nxt * gridDim.x) * TILE), "r"(TILE),
                   "r"(b)
                   : "memory");
    }
  }
  if (acc == 0x7654321)
    out[0] = acc;
}

double async_bench(torch::Tensor src, int tma, int blocks, int threads,
                   int stages, int reps) {
  long long bytes = src.numel() * src.element_size();
  auto flag =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  int smem = stages * 16384;
  if (tma)
    cudaFuncSetAttribute(bulk_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  else
    cudaFuncSetAttribute(cpasync_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto go = [&]() {
    if (tma)
      bulk_kernel<<<blocks, threads, smem>>>(
          (const char *)src.data_ptr(), bytes, stages, flag.data_ptr<int>());
    else
      cpasync_kernel<<<blocks, threads, smem>>>(
          (const char *)src.data_ptr(), bytes, stages, flag.data_ptr<int>());
  };
  go();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    return -1.0;
  double best = 1e9;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(e0);
    go();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, (double)ms * 1e-3);
  }
  return best;
}

// L2-resident streaming: tiles of a small buffer re-read `laps` times
// (cp.async, 4 stages; or TMA bulk).
__global__ void l2_cpasync_kernel(const char *__restrict__ src,
                                  long long tiles_buf, long long tiles,
                                  int stages, int *out) {
  extern __shared__ __align__(128) char ring[];
  const int TILE = 16384;
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(ring);
  int issued = 0;
  uint32_t acc = 0;
  for (long long t = blockIdx.x; t < tiles; t += gridDim.x) {
    int slot = issued % stages;
    const char *g = src + (t % tiles_buf) * TILE;
    for (int off = threadIdx.x * 16; off < TILE; off += blockDim.x * 16)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       sbase + slot * TILE + off),
                   "l"(g + off));
    asm volatile("cp.async.commit_group;");
    ++issued;
    if (issued >= stages) {
      asm volatile("cp.async.wait_group %0;" ::"n"(2));
      __syncthreads();
      acc ^= *(const uint32_t *)(ring + ((issued - stages) % stages) * TILE +
                                 threadIdx.x * 4);
      __syncthreads();
    }
  }
  asm volatile("cp.async.wait_all;");
  if (acc == 0x7654321)
    out[0] = acc;
}
__global__ void l2_bulk_kernel(const char *__restrict__ src,
                               long long tiles_buf, long long tiles, int stages,
                               int *out) {
  extern __shared__ __align__(128) char ring[];
  __shared__ __align__(8) uint64_t bar[8];
  const int TILE = 16384;
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(ring);
  if (threadIdx.x == 0)
    for (int i = 0; i < stages; ++i)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(
          (uint32_t)__cvta_generic_to_shared(&bar[i])));
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();
  uint32_t phase[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t acc = 0;
  long long mine = (tiles - blockIdx.x + gridDim.x - 1) / gridDim.x;
  for (int i = 0; i < stages && i < mine; ++i) {
    if (threadIdx.x == 0) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[i]);
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),
          "r"(TILE));
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1], %2, [%3];" ::"r"(sbase + i * TILE),
          "l"(src +
              ((blockIdx.x + (long long)i * gridDim.x) % tiles_buf) * TILE),
          "r"(TILE), "r"(b)
          : "memory");
    }
  }
  for (long long j = 0; j < mine; ++j) {
    int slot = j % stages;
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&bar[slot]);
    uint32_t done = 0;
    while (!done) {
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 "
                   "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done)
                   : "r"(b), "r"(phase[slot]));
    }
    phase[slot] ^= 1;
    acc ^= *(const uint32_t *)(ring + slot * TILE + threadIdx.x * 4);
    __syncthreads();
    long long nxt = j + stages;
    if (threadIdx.x == 0 && nxt < mine) {
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),
          "r"(TILE));
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
          "[%0], [%1], %2, [%3];" ::"r"(sbase + slot * TILE),
          "l"(src + ((blockIdx.x + nxt * gridDim.x) % tiles_buf) * TILE),
          "r"(TILE), "r"(b)
          : "memory");
    }
  }
  if (acc == 0x7654321)
    out[0] = acc;
}
double l2_bench(torch::Tensor src, int tma, int laps, int blocks, int threads,
                int stages) {
  long long tiles_buf = src.numel() / 16384, tiles = tiles_buf * laps;
  auto flag =
      torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  int smem = stages * 16384;
  cudaFuncSetAttribute(l2_bulk_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(l2_cpasync_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  auto go = [&]() {
    if (tma)
      l2_bulk_kernel<<<blocks, threads, smem>>>((const char *)src.data_ptr(),
                                                tiles_buf, tiles, stages,
                                                flag.data_ptr<int>());
    else
      l2_cpasync_kernel<<<blocks, threads, smem>>>((const char *)src.data_ptr(),
                                                   tiles_buf, tiles, stages,
                                                   flag.data_ptr<int>());
  };
  go();
  if (cudaDeviceSynchronize() != cudaSuccess)
    return -1;
  double best = 1e9;
  for (int r = 0; r < 3; ++r) {
    cudaEventRecord(e0);
    go();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    best = std::min(best, (double)ms * 1e-3);
  }
  return (double)tiles * 16384 / best;
}

int occupancy(int which, int threads, int smem) {
  int n = 0;
  if (which == 0)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, cpasync_kernel, threads,
                                                  smem);
  return n;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mma_bench", &mma_bench);
  m.def("mem_bench", &mem_bench);
  m.def("smem_bench", &smem_bench);
  m.def("async_bench", &async_bench);
  m.def("occupancy", &occupancy);
  m.def("l2_bench", &l2_bench);
}
