// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
// (`python/minisgl/kernel/csrc/jit/store.cu`); see THIRD_PARTY_NOTICES.
// design ideas from SGLang 72730d6
// `python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh` (row splitting,
// occupancy-sized blocks and early PDL trigger), and from vLLM 0.31.0's
// `reshape_and_cache_flash` kernel as compiled in its wheel (per-thread
// K-then-V copy, used for mid-sized batches); independent implementation.

#include <ctiengine/tensor.h>
#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>
#include <ctiengine/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t kv_cache_stride;
  std::size_t kv_input_stride;
  std::size_t length;
};

inline constexpr auto row_splits(std::size_t bytes) -> std::size_t {
  // One vector per lane for each of K and V, including narrower small rows.
  const auto unit = device::warp::details::resolve_unit_size(bytes);
  return unit ? bytes / (device::kWarpThreads * unit) : 0;
}

template <typename Package, std::integral T>
__always_inline __device__ void
store_pair_if(T pos, std::size_t stride, std::size_t byte_offset, void *k_cache,
              void *v_cache, const Package &key, const Package &value) {
  // Form no destination pointer for a skipped row.
  if (pos >= 0) {
    const auto dst_k = static_cast<Package *>(
        device::pointer::offset(k_cache, pos * stride, byte_offset));
    const auto dst_v = static_cast<Package *>(
        device::pointer::offset(v_cache, pos * stride, byte_offset));
    *dst_k = key;
    *dst_v = value;
  }
}

template <typename Package, std::integral T>
__always_inline __device__ void store_if(T pos, std::size_t stride,
                                         std::size_t byte_offset, void *cache,
                                         const Package &value) {
  // Form no destination pointer for a skipped row.
  if (pos >= 0)
    *static_cast<Package *>(
        device::pointer::offset(cache, pos * stride, byte_offset)) = value;
}

template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, bool kKThenV, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const __grid_constant__ StoreKernelParams params) {
  using namespace device;

  constexpr auto kSplits = row_splits(kElementSize);
  static_assert(kSplits > 0,
                "row size must be a positive multiple of 128 bytes");
  constexpr auto kSplitBytes = kElementSize / kSplits;
  using Package = warp::details::mem_package_t<
      kSplitBytes, warp::details::resolve_unit_size(kSplitBytes)>;
  static_assert(kSplitBytes == sizeof(Package) * kWarpThreads);

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride,
               kv_input_stride, length] = params;
  const auto warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const auto row_id = warp_id / kSplits;
  const auto split_offset = (warp_id % kSplits) * kSplitBytes;
  if (row_id >= length)
    return;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  PDL::wait<kUsePDL>();
#endif

  if constexpr (kKThenV) {
    // Mid-sized batches: copy K, then V, each thread holding one vector at a
    // time (the per-thread order of vLLM's reshape_and_cache_flash). Faster
    // here than holding K and V together; slower on small batches. Loads stay
    // coherent: under PDL the producer of K/V may still be running.
    const auto pos = static_cast<const T *>(indices)[row_id];
    const auto src_k = static_cast<const Package *>(
        pointer::offset(k, row_id * kv_input_stride, split_offset));
    const auto src_v = static_cast<const Package *>(
        pointer::offset(v, row_id * kv_input_stride, split_offset));
    const auto key = src_k[threadIdx.x];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    // The guarded K store before the V load keeps ptxas from hoisting the V
    // load above it.
    const auto byte_offset = split_offset + threadIdx.x * sizeof(Package);
    store_if(pos, kv_cache_stride, byte_offset, k_cache, key);
    const auto value = src_v[threadIdx.x];
    store_if(pos, kv_cache_stride, byte_offset, v_cache, value);
    return;
  }

  const auto pos = static_cast<const T *>(indices)[row_id];
  const auto src_k = static_cast<const Package *>(
      pointer::offset(k, row_id * kv_input_stride, split_offset));
  const auto src_v = static_cast<const Package *>(
      pointer::offset(v, row_id * kv_input_stride, split_offset));
  const auto key = src_k[threadIdx.x];
  const auto value = src_v[threadIdx.x];
  // The trigger fires whatever this kernel's own launch attribute says: the
  // consumer's PDL attribute decides the overlap. A PDL consumer must execute
  // griddepcontrol.wait before it reads the cache; the wait covers these
  // stores. The "memory" clobber keeps K/V loads ahead of the sign guard
  // (ptxas sinks them below it otherwise). Before SM90 the instruction does not
  // exist; a bare compiler barrier stands in (correct, not tuned there).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#else
  asm volatile("" ::: "memory");
#endif
  const auto byte_offset = split_offset + threadIdx.x * sizeof(Package);
  store_pair_if(pos, kv_cache_stride, byte_offset, k_cache, v_cache, key,
                value);
}

// Size bands (in warps, i.e. 512-byte row pieces) measured on the RTX 5090,
// RTX PRO 6000 and GB10 against SGLang's and vLLM's store kernels.
inline constexpr std::size_t kSmallBatchWarps = 1024;
inline constexpr std::size_t kKThenVMinWarps = 8192;
inline constexpr std::size_t kKThenVMaxWarps = 32768;

inline auto device_attribute(::cudaDeviceAttr attr, int device_id) -> int {
  // Cached per device; querying an attribute is legal during graph capture.
  static std::array<std::array<std::atomic<int>, 64>, 2> cache{};
  const auto slot = attr == ::cudaDevAttrMultiProcessorCount ? 0 : 1;
  auto &entry = cache[slot].at(static_cast<std::size_t>(device_id));
  auto value = entry.load(std::memory_order_relaxed);
  if (value == 0) {
    host::CUDA_CHECK(::cudaDeviceGetAttribute(&value, attr, device_id));
    entry.store(value, std::memory_order_relaxed);
  }
  return value;
}

inline auto sm_count(int device_id) -> std::size_t {
  return static_cast<std::size_t>(
      device_attribute(::cudaDevAttrMultiProcessorCount, device_id));
}

// Programmatic dependent launch needs SM90+; older devices launch plainly.
inline auto pdl_supported(int device_id) -> bool {
  return device_attribute(::cudaDevAttrComputeCapabilityMajor, device_id) >= 9;
}

template <std::size_t element_size, // depends on data type and embedding dim
          std::size_t num_threads = 256,   // maximum threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = true>
struct StoreKernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v) {
    using namespace host;
    auto D = SymbolicSize{"D"}; // element size
    auto L = SymbolicSize{"L"}; // length
    auto X = SymbolicSize{"X"}; // stride kv cache
    auto Y = SymbolicSize{"Y"}; // stride kv input
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, D}) //
        .with_strides({X, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k_cache)
        .verify(v_cache);
    TensorMatcher({L, D}) //
        .with_strides({Y, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(k)
        .verify(v);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype_.unwrap());
    RuntimeCheck(element_size == dtype_size * D.unwrap());

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_stride = X.unwrap() * dtype_size;
    const auto kv_input_stride = Y.unwrap() * dtype_size;

    const auto params = StoreKernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kv_cache_stride = kv_cache_stride,
        .kv_input_stride = kv_input_stride,
        .length = length,
    };

    if (length == 0)
      return;
    constexpr auto kMaxWarps = num_threads / 32;
    static_assert(num_threads == 32 || num_threads == 64 ||
                  num_threads == 128 || num_threads == 256);
    const auto total_warps = length * row_splits(element_size);
    const auto sms = sm_count(device.device_id);
    unsigned block_warps = 1;
    while (block_warps < kMaxWarps && total_warps > sms * block_warps)
      block_warps *= 2;
    // Small batches of rows >= 2 KiB: one row per block spreads the rows over
    // more SMs (2-warp blocks for 1 KiB rows measured slower).
    constexpr auto kRowWarps = static_cast<unsigned>(row_splits(element_size));
    if (total_warps <= kSmallBatchWarps && element_size >= 2048 &&
        kRowWarps <= kMaxWarps && block_warps > kRowWarps)
      block_warps = kRowWarps;
    const auto num_blocks = div_ceil(total_warps, block_warps);
    const bool k_then_v =
        total_warps >= kKThenVMinWarps && total_warps <= kKThenVMaxWarps;
    const auto kernel =
        k_then_v
            ? (use_int32 ? store_kv_cache<num_threads, max_concurrency, use_pdl,
                                          element_size, true, int32_t>
                         : store_kv_cache<num_threads, max_concurrency, use_pdl,
                                          element_size, true, int64_t>)
            : (use_int32 ? store_kv_cache<num_threads, max_concurrency, use_pdl,
                                          element_size, false, int32_t>
                         : store_kv_cache<num_threads, max_concurrency, use_pdl,
                                          element_size, false, int64_t>);
    LaunchKernel(num_blocks, {32, block_warps}, device)
        .with_attr(use_pdl && pdl_supported(device.device_id))(kernel, params);
  }
};

} // namespace
