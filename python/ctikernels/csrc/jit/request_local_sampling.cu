// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2024 FlashInfer team
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from FlashInfer fa381e5162d9b6ab5479b77e5b6d287b732d108e
// (`include/flashinfer/sampling.cuh`); see THIRD_PARTY_NOTICES.

// Each request uses its own seed/offset stream with subsequence zero;
// the tensor entry point validates inputs before batched dispatch.

#include <ctiengine/tensor.h>
#include <ctiengine/utils.cuh>
#include <flashinfer/sampling.cuh>
#include <tvm/ffi/container/tensor.h>

namespace {

using namespace flashinfer::sampling;

template <uint32_t Threads, uint32_t VecSize>
__global__ void sample_requests(const float *probs, const int64_t *rng,
                                int32_t *output, uint32_t vocab_size,
                                uint32_t batch_size) {
  const uint32_t request = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  // Request position, never its current batch index, selects the random draw.
  curand_init(rng[request], 0, rng[batch_size + request], &state);
  using Storage = SamplingTempStorage<Threads, SCAN_ALGO, REDUCE_ALGO>;
  __shared__ Storage storage;
  if (tx == 0) {
    storage.sampled_id = vocab_size;
    storage.last_valid_id = -1;
  }
  __syncthreads();

  flashinfer::vec_t<float, VecSize> values;
  float aggregate = 0;
  const float uniform = curand_uniform(&state);
#pragma unroll 2
  for (uint32_t i = 0; i < flashinfer::ceil_div(vocab_size, Threads * VecSize);
       ++i) {
    values.fill(0);
    if ((i * Threads + tx) * VecSize < vocab_size) {
      values.cast_load(probs + static_cast<size_t>(request) * vocab_size +
                       (i * Threads + tx) * VecSize);
    }
    DeviceSamplingFromProb<VecSize, Threads, SCAN_ALGO, REDUCE_ALGO, true>(
        i, vocab_size, [](float x) { return x > 0; }, uniform, values,
        aggregate, &storage);
    if (aggregate > uniform) {
      break;
    }
  }
  if (tx == 0) {
    // Match the provider's rounding-tail fallback (empty support yields 0).
    output[request] = storage.sampled_id < vocab_size
                          ? storage.sampled_id
                          : max(storage.last_valid_id, 0);
  }
}

void launch_request_sampling(tvm::ffi::TensorView probs,
                             tvm::ffi::TensorView rng,
                             tvm::ffi::TensorView output) {
  using namespace host;
  SymbolicSize batch, vocab;
  SymbolicDevice device;
  TensorMatcher({batch, vocab})
      .with_device<kDLCUDA>(device)
      .with_dtype<float>()
      .verify(probs);
  TensorMatcher({2, batch})
      .with_device<kDLCUDA>(device)
      .with_dtype<int64_t>()
      .verify(rng);
  TensorMatcher({batch})
      .with_device<kDLCUDA>(device)
      .with_dtype<int32_t>()
      .verify(output);
  RuntimeCheck(batch.unwrap() > 0 && vocab.unwrap() > 0);
  RuntimeCheck(vocab.unwrap() <= INT32_MAX);

  const auto batch_size = static_cast<uint32_t>(batch.unwrap());
  const auto vocab_size = static_cast<uint32_t>(vocab.unwrap());
  const auto vec_size = std::gcd(4u, vocab_size);
  const auto capability = flashinfer::GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(capability, Threads, {
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VecSize, {
      LaunchKernel(batch_size, Threads, device.unwrap())(
          sample_requests<Threads, VecSize>,
          static_cast<const float *>(probs.data_ptr()),
          static_cast<const int64_t *>(rng.data_ptr()),
          static_cast<int32_t *>(output.data_ptr()), vocab_size, batch_size);
    });
  });
}

} // namespace
