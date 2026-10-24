// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

// DeepSeek-V4 inverse rotary of the attention output, in place: o [T, H, 512]
// BF16 (token/head strides free, unit channel stride); channels 448..511 rotate
// back in adjacent pairs by the angles of positions[t] (int64) from an FP32
// table [P, 64] of 32 cosines then 32 sines. One warp per (token, group of
// heads), one pair per lane. Groups share the angle loads; they pay off only
// where latency, not bandwidth, bounds a long chunk (sm120 sweeps: 4 heads at
// >= 4096 (token, head) pairs, 2 at >= 1024; GB10 and others: 1).

#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kRopeThreads = 128;

struct RopeParams {
  __nv_bfloat162 *o;
  int64_t stride_t, stride_h; // in BF16 pairs
  const float *table;
  const int64_t *positions;
  int tokens, heads;
  int group, groups; // heads a warp; warps a token
};

__global__ void __launch_bounds__(kRopeThreads)
    dsv4_inverse_rope_kernel(const __grid_constant__ RopeParams p) {
  device::PDL::wait<true>(); // o comes from the attention kernel
  const int work = (blockIdx.x * kRopeThreads + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (work >= p.tokens * p.groups)
    return;
  const int t = work / p.groups, first = work % p.groups * p.group;
  const int last = min(first + p.group, p.heads);
  const float *angles = p.table + p.positions[t] * 64;
  const float c = angles[lane], s = angles[32 + lane];
  __nv_bfloat162 *row = p.o + t * p.stride_t + 224 + lane;
#pragma unroll 4
  for (int h = first; h < last; ++h) {
    __nv_bfloat162 *pair = row + h * p.stride_h;
    const float2 x = __bfloat1622float2(*pair);
    *pair = __floats2bfloat162_rn(x.x * c + x.y * s, x.y * c - x.x * s);
  }
  device::PDL::launch<true>();
}

// Compute capability 12.0 (RTX 5090 / RTX PRO 6000), cached per device.
bool sm120(int device) {
  static int cached[64] = {}; // 0 unknown, 1 yes, 2 no
  if (device < 0 || device >= 64)
    return false;
  if (cached[device] == 0) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cached[device] = major == 12 && minor == 0 ? 1 : 2;
  }
  return cached[device] == 1;
}

} // namespace

struct Dsv4InverseRope {
  static void run(const tvm::ffi::TensorView o,
                  const tvm::ffi::TensorView table,
                  const tvm::ffi::TensorView positions) {
    using host::RuntimeCheck;
    RuntimeCheck(o.ndim() == 3 && o.size(2) == 512 && o.stride(2) == 1 &&
                     o.dtype().code == kDLBfloat && o.dtype().bits == 16 &&
                     o.stride(0) % 2 == 0 && o.stride(1) % 2 == 0,
                 "DSV4 inverse rope: BF16 o [T, H, 512], even strides");
    RuntimeCheck(table.dtype().code == kDLFloat && table.dtype().bits == 32 &&
                     positions.dtype().code == kDLInt &&
                     positions.dtype().bits == 64 &&
                     positions.size(0) == o.size(0),
                 "DSV4 inverse rope: FP32 table, int64 positions");
    const int tokens = static_cast<int>(o.size(0));
    const int heads = static_cast<int>(o.size(1));
    if (tokens == 0)
      return;
    const int work = tokens * heads;
    const int group = !sm120(o.device().device_id) ? 1
                      : work >= 4096               ? 4
                      : work >= 1024               ? 2
                                                   : 1;
    const int groups = (heads + group - 1) / group;
    const RopeParams params{static_cast<__nv_bfloat162 *>(o.data_ptr()),
                            o.stride(0) / 2,
                            o.stride(1) / 2,
                            static_cast<const float *>(table.data_ptr()),
                            static_cast<const int64_t *>(positions.data_ptr()),
                            tokens,
                            heads,
                            group,
                            groups};
    const auto device = o.device();
    const auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
    const int warps = tokens * groups;
    const int blocks = (warps * 32 + kRopeThreads - 1) / kRopeThreads;
    host::LaunchKernel(blocks, kRopeThreads, stream)
        .with_attr(true)(dsv4_inverse_rope_kernel, params);
    RuntimeCheck(cudaGetLastError() == cudaSuccess,
                 "DSV4 inverse rope: launch");
  }
};
