// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// tvm-ffi entry of FlashMLA's sm100 sparse prefill (vendored under flashmla/,
// MIT): DeepSeek-V4 extend over BF16 rows.

// out[s, h] = softmax over the selected rows (key = value, d = 512, one KV
// head) with one FP32 sink per head joining the denominator; 64 query heads,
// BF16 q/out; -1 indices are skipped.

#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "flashmla/sm100/prefill/sparse/fwd/head64/phase1.cuh"
#include "flashmla/sm100/prefill/sparse/fwd/head64/phase1.h"

#include <cstdint>
#include <limits>

namespace {

int as_int(std::int64_t value) {
  host::RuntimeCheck(value <= std::numeric_limits<int>::max(),
                     "DSV4 sparse prefill: stride exceeds int32");
  return static_cast<int>(value);
}

bool is(const tvm::ffi::TensorView t, std::uint8_t code, std::uint8_t bits) {
  return t.dtype().code == code && t.dtype().bits == bits;
}

} // namespace

struct Dsv4FlashMla {
  // Prefill: q [S, 64, 512] BF16 over kv [N, 512] BF16 rows; max_logits / lse
  // are FP32 [S, 64].
  static void
  prefill(const tvm::ffi::TensorView q, const tvm::ffi::TensorView kv,
          const tvm::ffi::TensorView indices,
          const tvm::ffi::TensorView topk_length,
          const tvm::ffi::TensorView sinks, const tvm::ffi::TensorView out,
          const tvm::ffi::TensorView max_logits, const tvm::ffi::TensorView lse,
          double sm_scale, std::int64_t num_sms) {
    using host::RuntimeCheck;
    RuntimeCheck(q.ndim() == 3 && kv.ndim() == 2 && indices.ndim() == 2,
                 "DSV4 sparse prefill: q [S, 64, 512], kv [N, 512], "
                 "indices [S, K]");
    const auto s_q = q.size(0), topk = indices.size(1);
    RuntimeCheck(q.size(1) == 64 && q.size(2) == 512 && kv.size(1) == 512,
                 "DSV4 sparse prefill: 64 heads of 512 channels");
    RuntimeCheck(is(q, kDLBfloat, 16) && is(kv, kDLBfloat, 16) &&
                     is(out, kDLBfloat, 16) && is(indices, kDLInt, 32) &&
                     is(topk_length, kDLInt, 32) && is(sinks, kDLFloat, 32) &&
                     is(max_logits, kDLFloat, 32) && is(lse, kDLFloat, 32),
                 "DSV4 sparse prefill: dtypes");
    RuntimeCheck(q.stride(2) == 1 && kv.stride(1) == 1 &&
                     indices.stride(1) == 1 && out.is_contiguous() &&
                     max_logits.is_contiguous() && lse.is_contiguous() &&
                     topk_length.is_contiguous() && sinks.is_contiguous(),
                 "DSV4 sparse prefill: layouts");
    RuntimeCheck(indices.size(0) == s_q && topk_length.size(0) == s_q &&
                     sinks.size(0) == 64 && out.size(0) == s_q,
                 "DSV4 sparse prefill: shapes");
    if (s_q == 0)
      return;
    const auto device = q.device();
    SparseAttnFwdParams params = {
        static_cast<int>(s_q),
        as_int(kv.size(0)),
        64,
        1,
        512,
        512,
        as_int(topk),
        static_cast<float>(sm_scale),
        static_cast<float>(sm_scale * 1.44269504),
        static_cast<cutlass::bfloat16_t *>(q.data_ptr()),
        static_cast<cutlass::bfloat16_t *>(kv.data_ptr()),
        static_cast<int *>(indices.data_ptr()),
        static_cast<float *>(sinks.data_ptr()),
        static_cast<int *>(topk_length.data_ptr()),
        as_int(q.stride(0)),
        as_int(q.stride(1)),
        as_int(kv.stride(0)),
        as_int(kv.stride(0)),
        as_int(indices.stride(0)),
        as_int(indices.stride(0)),
        static_cast<cutlass::bfloat16_t *>(out.data_ptr()),
        static_cast<float *>(max_logits.data_ptr()),
        static_cast<float *>(lse.data_ptr()),
        static_cast<int>(num_sms),
        static_cast<cudaStream_t>(
            ::TVMFFIEnvGetStream(device.device_type, device.device_id)),
    };
    sm100::fwd::head64::run_fwd_phase1_kernel<512>(params);
  }
};
