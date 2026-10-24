// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <cublasLt.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace {
using host::RuntimeCheck;
using tvm::ffi::TensorView;
constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

void check(cublasStatus_t status) {
  RuntimeCheck(status == CUBLAS_STATUS_SUCCESS,
               "alpha_gemm cuBLASLt error: ", static_cast<int>(status));
}

void *address(const TensorView &t) {
  return static_cast<char *>(t.data_ptr()) + t.byte_offset();
}

uint32_t alignment(const TensorView &t) {
  auto ptr = reinterpret_cast<uintptr_t>(address(t));
  uint32_t result = 1;
  while (result < 256 && ptr % (2 * result) == 0) {
    result *= 2;
  }
  return result;
}

cudaDataType_t dtype(const TensorView &t) {
  auto d = t.dtype();
  RuntimeCheck(d.lanes == 1, "alpha_gemm requires scalar element types");
  if (d.code == kDLBfloat && d.bits == 16)
    return CUDA_R_16BF;
  if (d.code == kDLFloat && d.bits == 16)
    return CUDA_R_16F;
  RuntimeCheck(d.code == kDLFloat && d.bits == 32,
               "alpha_gemm requires BF16, FP16 or FP32 matrices");
  return CUDA_R_32F;
}

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t left = nullptr, right = nullptr, output = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  ~Plan() {
    if (preference)
      cublasLtMatmulPreferenceDestroy(preference);
    if (output)
      cublasLtMatrixLayoutDestroy(output);
    if (right)
      cublasLtMatrixLayoutDestroy(right);
    if (left)
      cublasLtMatrixLayoutDestroy(left);
    if (op)
      cublasLtMatmulDescDestroy(op);
  }
};

// Include pointer alignment classes: a saved-logits slice can be unaligned even
// when another call has identical shape/strides. Descriptors never retain
// pointers.
using Key = std::array<int64_t, 9>;
struct DeviceState {
  cublasLtHandle_t handle = nullptr;
  std::map<Key, std::unique_ptr<Plan>> plans;
  DeviceState() { check(cublasLtCreate(&handle)); }
  ~DeviceState() { cublasLtDestroy(handle); }
};

void alpha_gemm(TensorView a, TensorView b, TensorView alpha, TensorView zero,
                TensorView out, TensorView workspace) {
  RuntimeCheck(a.device().device_type == kDLCUDA,
               "alpha_gemm requires CUDA tensors");
  for (const auto &t : {b, alpha, zero, out, workspace}) {
    RuntimeCheck(t.device().device_type == kDLCUDA &&
                     t.device().device_id == a.device().device_id,
                 "alpha_gemm tensors must share a CUDA device");
  }
  RuntimeCheck(a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2,
               "alpha_gemm matrices must be two-dimensional");
  const auto k = a.size(0), m = a.size(1), n = b.size(1);
  RuntimeCheck(b.size(0) == k && out.size(0) == m && out.size(1) == n,
               "alpha_gemm requires A[K,M], B[K,N], out[M,N]");
  const auto type = dtype(a);
  RuntimeCheck(dtype(b) == type && dtype(out) == type,
               "alpha_gemm matrix dtypes must match");
  RuntimeCheck(a.stride(1) == 1 && a.stride(0) >= m && b.stride(1) == 1 &&
                   b.stride(0) >= n && out.is_contiguous(),
               "alpha_gemm requires unit column strides and contiguous output");
  for (const auto &t : {alpha, zero}) {
    RuntimeCheck(t.ndim() <= 1 && t.numel() == 1 && dtype(t) == CUDA_R_32F,
                 "alpha_gemm alpha/beta must be scalar FP32 tensors");
  }
  RuntimeCheck(
      workspace.dtype().code == kDLUInt && workspace.dtype().bits == 8 &&
          workspace.dtype().lanes == 1 && workspace.is_contiguous() &&
          workspace.numel() >= kWorkspaceBytes && alignment(workspace) == 256,
      "alpha_gemm needs at least 32 MiB aligned contiguous uint8 workspace");
  int current = -1;
  host::CUDA_CHECK(cudaGetDevice(&current));
  RuntimeCheck(current == a.device().device_id,
               "alpha_gemm current CUDA device must match its tensors");
  const auto stream = host::LaunchKernel::resolve_device(a.device());
  if (m == 0 || n == 0)
    return;
  if (k == 0) {
    host::CUDA_CHECK(cudaMemsetAsync(
        address(out), 0, out.numel() * out.dtype().bits / 8, stream));
    return;
  }

  static std::mutex mutex;
  static std::map<int, std::unique_ptr<DeviceState>> devices;
  std::lock_guard<std::mutex> lock(mutex);
  auto &state = devices[current];
  if (!state)
    state = std::make_unique<DeviceState>();
  const Key key{k,
                m,
                n,
                a.stride(0),
                b.stride(0),
                static_cast<int64_t>(type),
                alignment(b),
                alignment(a),
                alignment(out)};
  auto it = state->plans.find(key);
  if (it == state->plans.end()) {
    auto plan = std::make_unique<Plan>();
    check(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const auto pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    check(cublasLtMatmulDescSetAttribute(plan->op,
                                         CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                         &pointer_mode, sizeof(pointer_mode)));
    // Row-major out[M,N] is column-major out^T[N,M]. B[K,N] is
    // column-major B^T[N,K] (ld=B row stride), used without transpose.
    // A[K,M] is column-major A^T[M,K] (ld=A row stride), transposed.
    // Thus out^T = alpha * B^T @ A, with no packing or FP32 output buffer.
    const auto transpose = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &transpose, sizeof(transpose)));
    check(cublasLtMatrixLayoutCreate(&plan->left, type, n, k, b.stride(0)));
    check(cublasLtMatrixLayoutCreate(&plan->right, type, m, k, a.stride(0)));
    check(cublasLtMatrixLayoutCreate(&plan->output, type, n, m, n));
    check(cublasLtMatmulPreferenceCreate(&plan->preference));
    check(cublasLtMatmulPreferenceSetAttribute(
        plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kWorkspaceBytes, sizeof(kWorkspaceBytes)));
    // Forbid split-K reductions in the output dtype: accumulation must be FP32.
    const uint32_t reductions =
        CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    check(cublasLtMatmulPreferenceSetAttribute(
        plan->preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        &reductions, sizeof(reductions)));
    const std::array attrs{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                           CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    const std::array<uint32_t, 4> alignments{alignment(b), alignment(a),
                                             alignment(out), alignment(out)};
    for (size_t i = 0; i < attrs.size(); ++i) {
      check(cublasLtMatmulPreferenceSetAttribute(
          plan->preference, attrs[i], &alignments[i], sizeof(uint32_t)));
    }
    int count = 0;
    check(cublasLtMatmulAlgoGetHeuristic(
        state->handle, plan->op, plan->left, plan->right, plan->output,
        plan->output, plan->preference, 1, &plan->heuristic, &count));
    RuntimeCheck(count == 1, "alpha_gemm has no supported cuBLASLt algorithm");
    check(plan->heuristic.state);
    it = state->plans.emplace(key, std::move(plan)).first;
  }
  const auto &plan = *it->second;
  check(cublasLtMatmul(state->handle, plan.op, address(alpha), address(b),
                       plan.left, address(a), plan.right, address(zero),
                       address(out), plan.output, address(out), plan.output,
                       &plan.heuristic.algo, address(workspace),
                       kWorkspaceBytes, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(alpha_gemm, &alpha_gemm);
} // namespace
