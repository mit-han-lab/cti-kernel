// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
// (`python/minisgl/kernel/csrc/src/pynccl.cu`); see THIRD_PARTY_NOTICES.

#include <ctiengine/nccl227.h>
#include <ctiengine/tensor.h>
#include <ctiengine/utils.cuh>
#include <ctiengine/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

using NCCLIDList = tvm::ffi::Array<char>;

auto NCCL_CHECK(::ncclResult_t result) -> void {
  if (result != ::ncclSuccess) {
    host::RuntimeCheck(false, ::ncclGetErrorString(result));
  }
}

auto get_uid(const NCCLIDList &wrapper) -> ncclUniqueId {
  host::RuntimeCheck(wrapper.size() == NCCL_UNIQUE_ID_BYTES,
                     "Invalid NCCL ID wrapper size");
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}

auto create_uid() -> NCCLIDList {
  ncclUniqueId id;
  NCCL_CHECK(::ncclGetUniqueId(&id));
  return NCCLIDList(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

const auto kNCCLReduceOPMap = std::unordered_map<std::string_view, ncclRedOp_t>{
    {"sum", ::ncclSum}, {"prod", ::ncclProd}, {"max", ::ncclMax},
    {"min", ::ncclMin}, {"avg", ::ncclAvg},
};

struct DLDataTypeHash {
  auto operator()(const DLDataType &dtype) const noexcept -> std::size_t {
    return std::bit_cast<std::uint32_t>(dtype);
  }
};

template <typename = void>
auto operator==(const DLDataType &a, const DLDataType &b) -> bool {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

const auto kNCCLDtypeMap =
    std::unordered_map<DLDataType, ncclDataType_t, DLDataTypeHash>{
        {{DLDataTypeCode::kDLInt, 8, 1}, ncclInt8},
        {{DLDataTypeCode::kDLUInt, 8, 1}, ncclUint8},
        {{DLDataTypeCode::kDLInt, 32, 1}, ncclInt32},
        {{DLDataTypeCode::kDLInt, 64, 1}, ncclInt64},
        {{DLDataTypeCode::kDLFloat, 16, 1}, ncclFloat16},
        {{DLDataTypeCode::kDLFloat, 32, 1}, ncclFloat32},
        {{DLDataTypeCode::kDLFloat, 64, 1}, ncclFloat64},
        {{DLDataTypeCode::kDLBfloat, 16, 1}, ncclBfloat16},
        {{DLDataTypeCode::kDLFloat8_e4m3fn, 8, 1}, ncclFloat8e4m3},
        {{DLDataTypeCode::kDLFloat8_e5m2, 8, 1}, ncclFloat8e5m2},
    };

using std::shared_ptr;

template <typename T> using shared_obj = shared_ptr<std::remove_pointer_t<T>>;
template <auto Fn>
inline constexpr auto template_fn =
    [](auto &&...args) { return Fn(std::forward<decltype(args)>(args)...); };

// The oldest runtime that provides every entry point this file calls
// (symmetric windows arrived in 2.27); the header is 2.27.1.
constexpr int kMinNCCLVersion = NCCL_VERSION_CODE;
constexpr int kNextNCCLMajorVersion = 30000;

auto runtime_version() -> int {
  int version = 0;
  NCCL_CHECK(::ncclGetVersion(&version));
  return version;
}

auto check_runtime_version() -> void {
  const auto version = runtime_version();
  host::RuntimeCheck(version >= kMinNCCLVersion &&
                         version < kNextNCCLMajorVersion,
                     "NCCL runtime version ", version,
                     " is not supported: need >= ", kMinNCCLVersion,
                     " (the 2.27.1 header) and < ", kNextNCCLMajorVersion);
}

struct NCCLWrapper : public tvm::ffi::Object {
public:
  NCCLWrapper(int rank, int world_size, const size_t max_bytes, NCCLIDList uid,
              int min_capability, std::string mode, std::string reason,
              bool test_null_window)
      : m_rank(rank), m_world_size(world_size),
        m_max_bytes(mode == "symmetric" ? max_bytes : 0),
        m_min_capability(min_capability), m_window_mode(std::move(mode)),
        m_window_reason(std::move(reason)) {
    using namespace host;
    check_runtime_version();
    RuntimeCheck(world_size > 0 && rank >= 0 && rank < world_size,
                 "Invalid NCCL rank ", rank, " for world size ", world_size);
    RuntimeCheck(m_window_mode == "symmetric" || m_window_mode == "in_place",
                 "Invalid window mode");
    RuntimeCheck(m_window_mode != "symmetric" || max_bytes > 0,
                 "Symmetric window must have positive size");
    ncclUniqueId id = get_uid(uid);
    ncclComm_t comm;
    NCCL_CHECK(::ncclCommInitRank(&comm, m_world_size, id, m_rank));
    m_comm = {comm, template_fn<::ncclCommDestroy>};

    int count = 0, user_rank = 0;
    NCCL_CHECK(::ncclCommCount(comm, &count));
    NCCL_CHECK(::ncclCommUserRank(comm, &user_rank));
    NCCL_CHECK(::ncclCommCuDevice(comm, &m_device));
    RuntimeCheck(count == m_world_size && user_rank == m_rank,
                 "NCCL communicator disagrees with rank/world size");

    // NCCL rejects an empty allocation: max_bytes == 0 means no window, so
    // every all-reduce runs in place on the caller's tensor.
    if (m_max_bytes == 0) {
      return;
    }
    void *buf;
    NCCL_CHECK(::ncclMemAlloc(&buf, max_bytes));
    m_sym_mem = {buf, template_fn<::ncclMemFree>};

    // test_null_window stands in for a null registration result (tests only).
    ncclWindow_t win = nullptr;
    if (!test_null_window) {
      NCCL_CHECK(::ncclCommWindowRegister(comm, buf, max_bytes, &win,
                                          NCCL_WIN_COLL_SYMMETRIC));
    }
    // Registration is the support probe: NCCL returns success with a null
    // window when symmetric memory is unsupported on this communicator. Then
    // the staging copies buy nothing, so drop the buffer and run in place.
    if (win == nullptr) {
      m_sym_mem.reset();
      m_max_bytes = 0;
      m_window_mode = "in_place";
      m_window_reason += "; NCCL window registration returned no window "
                         "(symmetric memory unsupported)";
      return;
    }
    m_win = {win, [comm = m_comm](ncclWindow_t w) {
               return NCCL_CHECK(::ncclCommWindowDeregister(comm.get(), w));
             }};
  }

  auto all_reduce(tvm::ffi::TensorView t, std::string op) const -> void {
    using namespace host;
    check_tensor(t, "Input");
    const auto size_dim = static_cast<size_t>(t.numel());
    const auto dtype = nccl_dtype(t);
    const auto elem_bytes = static_cast<size_t>(t.dtype().bits / 8);
    const auto size_bytes = size_dim * elem_bytes;
    const auto data_addr = address(t);
    const auto data_ptr = reinterpret_cast<void *>(data_addr);
    const auto reduce_op = reduction_op(t, op);
    if (size_dim == 0) {
      return;
    }
    if (size_bytes <= m_max_bytes) {
      const auto buf_addr = reinterpret_cast<std::uintptr_t>(m_sym_mem.get());
      const bool overlaps = data_addr < buf_addr + size_bytes &&
                            buf_addr < data_addr + size_bytes;
      RuntimeCheck(!overlaps || data_addr == buf_addr,
                   "Input overlaps the staging buffer at a different address");
    }
    // Validate before resolving the stream or enqueueing staging copies/NCCL.
    const auto stream = LaunchKernel::resolve_device(t.device());

    if (size_bytes <= m_max_bytes && size_bytes > 0) { // use internal buffer
      // ncclMemAlloc's CUDA VMM allocation uses the recommended granularity
      // (NCCL User Buffer Registration); its base is at least 256-byte aligned,
      // covering 4-byte FP32/int32 and 8-byte int64. Counts remain in elements;
      // only the staging copies use size_bytes.
      const auto buf_ptr = m_sym_mem.get();
      const auto need_memcpy = (buf_ptr != data_ptr);
      if (need_memcpy) {
        CUDA_CHECK(::cudaMemcpyAsync(buf_ptr, data_ptr, size_bytes,
                                     ::cudaMemcpyDeviceToDevice, stream));
      }
      NCCL_CHECK(::ncclAllReduce(
          /*sendbuff=*/buf_ptr,
          /*recvbuff=*/buf_ptr,
          /*count=*/size_dim,
          /*datatype=*/dtype,
          /*op=*/reduce_op,
          /*comm=*/m_comm.get(),
          /*stream=*/stream));
      if (need_memcpy) {
        CUDA_CHECK(::cudaMemcpyAsync(data_ptr, buf_ptr, size_bytes,
                                     ::cudaMemcpyDeviceToDevice, stream));
      }
    } else {
      NCCL_CHECK(::ncclAllReduce(
          /*sendbuff=*/data_ptr,
          /*recvbuff=*/data_ptr,
          /*count=*/size_dim,
          /*datatype=*/dtype,
          /*op=*/reduce_op,
          /*comm=*/m_comm.get(),
          /*stream=*/stream));
    }
  }

  // Gathers `src` from every rank into `dst` in rank order. It is a byte
  // copy, so callers may reinterpret padded bytes as a 16-bit dtype. Every
  // check runs before NCCL is called, so a rejected call enqueues nothing.
  auto all_gather(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) const
      -> void {
    using namespace host;
    check_tensor(src, "Source");
    check_tensor(dst, "Destination");
    RuntimeCheck(dst.dtype() == src.dtype(),
                 "Destination dtype must match the source dtype");
    const auto dtype = nccl_dtype(src);
    const auto count = static_cast<size_t>(src.numel());
    RuntimeCheck(count <= std::numeric_limits<size_t>::max() / m_world_size &&
                     static_cast<size_t>(dst.numel()) ==
                         count * static_cast<size_t>(m_world_size),
                 "Destination must hold exactly world_size (", m_world_size,
                 ") x ", count, " source elements, got ", dst.numel());
    check_pair_overlap(
        dst, src, "Source overlaps the destination outside this rank's slice");
    if (count == 0) {
      return;
    }
    const auto src_addr = address(src);
    const auto dst_addr = address(dst);
    const auto stream = LaunchKernel::resolve_device(src.device());
    NCCL_CHECK(::ncclAllGather(
        /*sendbuff=*/reinterpret_cast<const void *>(src_addr),
        /*recvbuff=*/reinterpret_cast<void *>(dst_addr),
        /*sendcount=*/count,
        /*datatype=*/copy_dtype(dtype),
        /*comm=*/m_comm.get(),
        /*stream=*/stream));
  }

  auto broadcast(tvm::ffi::TensorView t, int root) const -> void {
    using namespace host;
    check_tensor(t, "Input");
    const auto dtype = nccl_dtype(t);
    RuntimeCheck(root >= 0 && root < m_world_size,
                 "Broadcast root must be in [0, world_size)");
    if (t.numel() == 0) {
      return;
    }
    const auto ptr = reinterpret_cast<void *>(address(t));
    const auto stream = LaunchKernel::resolve_device(t.device());
    NCCL_CHECK(::ncclBroadcast(ptr, ptr, static_cast<size_t>(t.numel()),
                               copy_dtype(dtype), root, m_comm.get(), stream));
  }

  auto reduce_scatter(tvm::ffi::TensorView dst, tvm::ffi::TensorView src,
                      std::string op) const -> void {
    using namespace host;
    check_tensor(src, "Source");
    check_tensor(dst, "Destination");
    RuntimeCheck(dst.dtype() == src.dtype(),
                 "Destination dtype must match the source dtype");
    const auto dtype = nccl_dtype(src);
    const auto reduce_op = reduction_op(src, op);
    const auto count = static_cast<size_t>(dst.numel());
    RuntimeCheck(count <= std::numeric_limits<size_t>::max() / m_world_size &&
                     static_cast<size_t>(src.numel()) == count * m_world_size,
                 "Source must hold exactly world_size x destination elements");
    check_pair_overlap(
        src, dst, "Destination overlaps the source outside this rank's slice");
    if (count == 0) {
      return;
    }
    const auto stream = LaunchKernel::resolve_device(src.device());
    NCCL_CHECK(::ncclReduceScatter(reinterpret_cast<const void *>(address(src)),
                                   reinterpret_cast<void *>(address(dst)),
                                   count, dtype, reduce_op, m_comm.get(),
                                   stream));
  }

  auto window_mode() const -> std::string { return m_window_mode; }
  auto window_reason() const -> std::string { return m_window_reason; }
  auto window_bytes() const -> size_t { return m_max_bytes; }

  auto get_buffer() const -> void * { return m_sym_mem.get(); }

  auto device() const -> int { return m_device; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ctiengine.NCCLWrapper", NCCLWrapper,
                                    tvm::ffi::Object);

private:
  auto check_tensor(const tvm::ffi::TensorView &t, const char *name) const
      -> void {
    using namespace host;
    RuntimeCheck(t.device().device_type == kDLCUDA, name,
                 " tensor must be on a CUDA device");
    RuntimeCheck(t.device().device_id == m_device, name,
                 " tensor is on cuda:", t.device().device_id,
                 " but the NCCL communicator is on cuda:", m_device);
    RuntimeCheck(t.is_contiguous(), name, " tensor must be contiguous");
    nccl_dtype(t);
    const auto elem_bytes = static_cast<size_t>(t.dtype().bits / 8);
    RuntimeCheck(t.numel() >= 0 &&
                     static_cast<size_t>(t.numel()) <=
                         std::numeric_limits<size_t>::max() / elem_bytes,
                 name, " tensor size overflows");
    RuntimeCheck(address(t) % elem_bytes == 0, name, " must be ", elem_bytes,
                 "-byte aligned");
    RuntimeCheck(address(t) <= std::numeric_limits<std::uintptr_t>::max() -
                                   static_cast<size_t>(t.numel()) * elem_bytes,
                 name, " tensor address range overflows");
  }

  static auto nccl_dtype(const tvm::ffi::TensorView &t) -> ncclDataType_t {
    const auto it = kNCCLDtypeMap.find(t.dtype());
    host::RuntimeCheck(
        it != kNCCLDtypeMap.end(),
        "NCCL supported dtypes: int8, uint8, int32, int64, float16, "
        "float32, float64, bfloat16, float8_e4m3fn, float8_e5m2");
    return it->second;
  }

  auto reduction_op(const tvm::ffi::TensorView &t, const std::string &op) const
      -> ncclRedOp_t {
    const auto it = kNCCLReduceOPMap.find(op);
    host::RuntimeCheck(it != kNCCLReduceOPMap.end(), "Unknown reduce op: ", op);
    const auto code = t.dtype().code;
    host::RuntimeCheck(
        it->second != ncclAvg || (code != kDLInt && code != kDLUInt),
        "avg is not supported for int8/uint8/int32/int64; use sum and divide");
    host::RuntimeCheck(
        !is_fp8(nccl_dtype(t)) || m_min_capability >= 90,
        "FP8 reductions require every rank's compute capability >= 9.0");
    return it->second;
  }

  static auto is_fp8(ncclDataType_t dtype) -> bool {
    return dtype == ncclFloat8e4m3 || dtype == ncclFloat8e5m2;
  }

  static auto copy_dtype(ncclDataType_t dtype) -> ncclDataType_t {
    // NCCL 2.27 applies its sm90 guard even to FP8 copies. Both FP8 formats
    // occupy one byte, so transporting them as uint8 avoids that old guard.
    return is_fp8(dtype) ? ncclUint8 : dtype;
  }

  auto check_pair_overlap(const tvm::ffi::TensorView &large,
                          const tvm::ffi::TensorView &small,
                          const char *message) const -> void {
    const auto small_bytes =
        static_cast<size_t>(small.numel()) * (small.dtype().bits / 8);
    const auto large_bytes =
        static_cast<size_t>(large.numel()) * (large.dtype().bits / 8);
    const auto lo = address(large), hi = address(small);
    const bool overlaps = small_bytes > 0 && large_bytes > 0 &&
                          lo < hi + small_bytes && hi < lo + large_bytes;
    host::RuntimeCheck(!overlaps || hi == lo + m_rank * small_bytes, message);
  }

  static auto address(const tvm::ffi::TensorView &t) -> std::uintptr_t {
    return reinterpret_cast<std::uintptr_t>(t.data_ptr()) +
           static_cast<std::uintptr_t>(t.byte_offset());
  }

  int m_rank;
  int m_world_size;
  size_t m_max_bytes;
  int m_min_capability;
  std::string m_window_mode;
  std::string m_window_reason;
  int m_device = -1;
  shared_obj<ncclComm_t> m_comm;
  shared_ptr<void> m_sym_mem;
  shared_obj<ncclWindow_t> m_win;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<NCCLWrapper>()
      .def(refl::init<int, int, size_t, NCCLIDList, int, std::string,
                      std::string, bool>(),
           "__init__")
      .def("all_reduce", &NCCLWrapper::all_reduce)
      .def("all_gather", &NCCLWrapper::all_gather)
      .def("broadcast", &NCCLWrapper::broadcast)
      .def("reduce_scatter", &NCCLWrapper::reduce_scatter)
      .def("_window_mode", &NCCLWrapper::window_mode)
      .def("_window_reason", &NCCLWrapper::window_reason)
      .def("window_bytes", &NCCLWrapper::window_bytes)
      .def("get_buffer", &NCCLWrapper::get_buffer)
      .def("device", &NCCLWrapper::device);
}

auto device_pci_bus_id(int device) -> std::string {
  char bus_id[32] = {};
  using namespace host;
  CUDA_CHECK(::cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device));
  return bus_id;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(device_pci_bus_id, &device_pci_bus_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(create_nccl_uid, &create_uid);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nccl_runtime_version, &runtime_version);

} // namespace
