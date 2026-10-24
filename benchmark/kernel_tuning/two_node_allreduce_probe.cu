// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
/*
 * Measurement probe only, not shipped: ctikernels never imports or builds it.
 * Two-node GB10 BF16 all-reduce latency probe; CUDA 13, C++17.
 * Build (NCCL_HOME = an NCCL install):
 *   nvcc -O3 -std=c++17 -arch=sm_121a two_node_allreduce_probe.cu -o probe
 *     -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl -libverbs -lpthread
 * Launch one process per node with mpirun -np 2; both get identical options:
 *   ./probe --root-ip <rank0-ip> [--port 29571] [--ibdev rocep1s0f1] [--gid 3]
 *         [--sizes 2048,4096,8192,16384,24576,32768,65536,131072,262144,
 *                  524288,1048576] [--iters 2000] [--graph-ops 43] [--ctas 1]
 *         [--main-cpu N] [--proxy-cpu 19] [--rank 0|1] [--help]
 * Without --main-cpu the process runs on the X925 cores (5-9, 15-19) minus the
 * proxy core; NCCL threads inherit that mask. Pinning it to one core starves
 * the NCCL proxy thread (ms-scale NCCL steps).
 * OMPI_COMM_WORLD_RANK overrides --rank. CUDA device 0 is used on each node.
 * Sizes are bytes, positive multiples of 16, less than 2^24-16. iters must
 * be >= graph-ops; graph timing uses floor(iters/graph-ops)*graph-ops steps.
 * Fixed: 200 warmups/method/mode, 50 checks/method/size, seven paired rounds.
 * JSON and the human table are emitted on rank 0; checks include both ranks.
 * pingpong size_bytes=0 denotes a 16-byte control probe, reported as half RTT.
 * Assumptions: coherent mapped host memory, host-native atomics, RoCE v2 at
 * the selected GID, exclusive GPU use. No device allocation is NIC-registered.
 * proxy1 intentionally relies on ordered placement within one RDMA WRITE;
 * every checked iteration validates that assumption, not a portable guarantee.
 */

#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <limits>
#include <nccl.h>
#include <netinet/tcp.h>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using bf16 = __nv_bfloat16;
using u64 = unsigned long long;
static_assert(sizeof(u64) == 8 && sizeof(bf16) == 2, "wire sizes");
constexpr u64 SEQ_LIMIT = (1ULL << 40);
constexpr size_t ALIGN = 2 * 1024 * 1024;
constexpr int WARMUP = 200, CHECKS = 50, ROUNDS = 7;

[[noreturn]] static void die(const char *file, int line, const char *what) {
  std::fprintf(stderr, "%s:%d: %s\n", file, line, what);
  std::exit(1);
}
#define REQUIRE(x, msg)                                                        \
  do {                                                                         \
    if (!(x))                                                                  \
      die(__FILE__, __LINE__, msg);                                            \
  } while (0)
#define CUDA(x)                                                                \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess)                                                     \
      die(__FILE__, __LINE__, cudaGetErrorString(e_));                         \
  } while (0)
#define NCCL(x)                                                                \
  do {                                                                         \
    ncclResult_t e_ = (x);                                                     \
    if (e_ != ncclSuccess)                                                     \
      die(__FILE__, __LINE__, ncclGetErrorString(e_));                         \
  } while (0)
#define IB(x)                                                                  \
  do {                                                                         \
    int e_ = (x);                                                              \
    if (e_ != 0)                                                               \
      die(__FILE__, __LINE__, #x " failed");                                   \
  } while (0)
#define SYS(x)                                                                 \
  do {                                                                         \
    if ((x) < 0)                                                               \
      die(__FILE__, __LINE__, std::strerror(errno));                           \
  } while (0)

// CLI and affinity. Never dump environment variables or arbitrary argument
// values.
struct Options {
  int rank = -1, port = 29571, gid = 3, iters = 2000, graph_ops = 43;
  int ctas = 1, main_cpu = -1, proxy_cpu = 19;
  std::string root, ibdev = "rocep1s0f1";
  std::vector<size_t> sizes{2048,  4096,   8192,   16384,  24576,  32768,
                            65536, 131072, 262144, 524288, 1048576};
};
static long number(const char *s) {
  char *end = nullptr;
  errno = 0;
  long v = std::strtol(s, &end, 10);
  REQUIRE(errno == 0 && end != s && *end == '\0' && v >= 0 &&
              v <= std::numeric_limits<int>::max(),
          "invalid numeric option");
  return v;
}
static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--help") {
      std::puts("uar --root-ip IPv4 [--rank 0|1] [--port 29571] "
                "[--ibdev rocep1s0f1] [--gid 3] [--sizes bytes,...] "
                "[--iters 2000] [--graph-ops 43] [--ctas 1] "
                "[--main-cpu N] [--proxy-cpu 19]");
      std::exit(0);
    }
    REQUIRE(i + 1 < argc, "missing option value");
    const char *v = argv[++i];
    if (k == "--root-ip")
      o.root = v;
    else if (k == "--ibdev")
      o.ibdev = v;
    else if (k == "--rank")
      o.rank = int(number(v));
    else if (k == "--port")
      o.port = int(number(v));
    else if (k == "--gid")
      o.gid = int(number(v));
    else if (k == "--iters")
      o.iters = int(number(v));
    else if (k == "--graph-ops")
      o.graph_ops = int(number(v));
    else if (k == "--ctas")
      o.ctas = int(number(v));
    else if (k == "--main-cpu")
      o.main_cpu = int(number(v));
    else if (k == "--proxy-cpu")
      o.proxy_cpu = int(number(v));
    else if (k == "--sizes") {
      o.sizes.clear();
      std::stringstream ss(v);
      std::string item;
      REQUIRE(*v && v[std::strlen(v) - 1] != ',', "invalid sizes list");
      while (std::getline(ss, item, ','))
        o.sizes.push_back(size_t(number(item.c_str())));
    } else
      die(__FILE__, __LINE__, "unknown option");
  }
  if (const char *v = std::getenv("OMPI_COMM_WORLD_RANK"))
    o.rank = int(number(v));
  if (const char *v = std::getenv("OMPI_COMM_WORLD_SIZE"))
    REQUIRE(number(v) == 2, "exactly two MPI processes required");
  REQUIRE(o.rank == 0 || o.rank == 1, "rank must be 0 or 1");
  REQUIRE(o.rank == 0 || !o.root.empty(), "rank 1 requires --root-ip");
  REQUIRE(o.port > 0 && o.port <= 65535, "invalid TCP port");
  REQUIRE(o.ctas > 0 && o.graph_ops > 0 && o.iters >= o.graph_ops,
          "invalid iteration/grid count");
  REQUIRE(o.main_cpu < CPU_SETSIZE && o.proxy_cpu < CPU_SETSIZE &&
              o.main_cpu != o.proxy_cpu && o.proxy_cpu >= 0,
          "invalid or overlapping affinity CPUs");
  REQUIRE(!o.sizes.empty(), "empty sizes list");
  for (size_t b : o.sizes)
    REQUIRE(b > 0 && b % 16 == 0 && b < (1u << 24) - 16,
            "sizes must be 16-byte multiples below 2^24-16");
  // Includes eager, graph, both warmups, check passes, and pingpong; no wrap.
  u64 upper = u64(o.sizes.size()) * 2 *
                  (CHECKS + 2 * WARMUP + u64(ROUNDS) * 2 * o.iters) +
              WARMUP + u64(ROUNDS) * o.iters;
  REQUIRE(upper < SEQ_LIMIT - 1, "run exceeds 40-bit sequence space");
  return o;
}
// X925 cores of GB10 (5-9, 15-19) minus the proxy core; library threads (NCCL
// proxy) inherit it.
static void pin_big_except(int skip) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : {5, 6, 7, 8, 9, 15, 16, 17, 18, 19})
    if (c != skip)
      CPU_SET(c, &set);
  REQUIRE(sched_setaffinity(0, sizeof(set), &set) == 0,
          "sched_setaffinity failed");
}
static void pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  SYS(sched_setaffinity(0, sizeof(set), &set));
}

// TCP bootstrap: exact transfers, rank-ordered exchange, and a tiny barrier.
struct Bootstrap {
  int fd = -1, rank;
  explicit Bootstrap(const Options &o) : rank(o.rank) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(o.port));
    if (rank == 0) {
      int listener = ::socket(AF_INET, SOCK_STREAM, 0);
      SYS(listener);
      int one = 1;
      SYS(setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)));
      a.sin_addr.s_addr = htonl(INADDR_ANY);
      SYS(bind(listener, reinterpret_cast<sockaddr *>(&a), sizeof(a)));
      SYS(listen(listener, 1));
      do {
        fd = accept(listener, nullptr, nullptr);
      } while (fd < 0 && errno == EINTR);
      SYS(fd);
      SYS(close(listener));
    } else {
      REQUIRE(inet_pton(AF_INET, o.root.c_str(), &a.sin_addr) == 1,
              "root-ip must be IPv4");
      for (int attempt = 0; attempt < 600; ++attempt) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        SYS(fd);
        if (connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0)
          break;
        SYS(close(fd));
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      REQUIRE(fd >= 0, "TCP bootstrap connection failed");
    }
    int one = 1;
    SYS(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)));
  }
  void send(const void *p, size_t n) {
    auto b = static_cast<const char *>(p);
    while (n) {
      ssize_t k = ::send(fd, b, n, MSG_NOSIGNAL);
      if (k < 0 && errno == EINTR)
        continue;
      REQUIRE(k > 0, "TCP send failed");
      b += k;
      n -= size_t(k);
    }
  }
  void recv(void *p, size_t n) {
    auto b = static_cast<char *>(p);
    while (n) {
      ssize_t k = ::recv(fd, b, n, 0);
      if (k < 0 && errno == EINTR)
        continue;
      REQUIRE(k > 0, "TCP receive failed/peer disconnected");
      b += k;
      n -= size_t(k);
    }
  }
  void exchange(const void *local, void *peer, size_t n) {
    if (rank == 0) {
      send(local, n);
      recv(peer, n);
    } else {
      recv(peer, n);
      send(local, n);
    }
  }
  void barrier() {
    uint8_t local = 0x5a, peer = 0;
    exchange(&local, &peer, 1);
    REQUIRE(peer == local, "TCP barrier mismatch");
  }
  void check_config(const Options &o) {
    std::ostringstream s;
    s << "uar-v1 " << o.iters << ' ' << o.graph_ops << ' ' << o.ctas;
    for (size_t b : o.sizes)
      s << ' ' << b;
    std::string local = s.str();
    uint32_t len = htonl(uint32_t(local.size())), other = 0;
    exchange(&len, &other, sizeof(len));
    REQUIRE(ntohl(other) == local.size(), "rank configurations differ");
    std::string peer(local.size(), '\0');
    exchange(local.data(), peer.data(), local.size());
    REQUIRE(peer == local, "rank configurations differ");
  }
  ~Bootstrap() {
    if (fd >= 0)
      SYS(close(fd));
  }
};

// One mapped/registered host allocation: send[2], recv[2], three flag lines.
struct Region {
  char *send[2];
  char *recv[2];
  u64 *sflag[2];
  u64 *rflag[2];
  u64 *seq_ctr;
  unsigned *arrived;
  unsigned *finished;
  int one_write;
};
struct Memory {
  char *host = nullptr;
  char *device = nullptr;
  size_t slot, bytes, flag_offset;
  Region dev{};
  explicit Memory(size_t max_size) {
    slot = (max_size + 64 + 127) & ~size_t(127);
    flag_offset = 4 * slot;
    bytes = (flag_offset + 4 * 128 + ALIGN - 1) & ~(ALIGN - 1);
    void *p = nullptr;
    REQUIRE(posix_memalign(&p, ALIGN, bytes) == 0, "posix_memalign failed");
    host = static_cast<char *>(p);
    std::memset(host, 0, bytes);
    CUDA(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
    void *d = nullptr;
    CUDA(cudaHostGetDevicePointer(&d, host, 0));
    device = static_cast<char *>(d);
    for (int i = 0; i < 2; ++i) {
      dev.send[i] = device + i * slot;
      dev.recv[i] = device + (i + 2) * slot;
      dev.rflag[i] =
          reinterpret_cast<u64 *>(device + flag_offset + (i + 1) * 128);
    }
    // One doorbell per slot: a rank's step k+1 may publish before its proxy saw
    // step k.
    dev.sflag[0] = reinterpret_cast<u64 *>(device + flag_offset);
    dev.sflag[1] = reinterpret_cast<u64 *>(device + flag_offset + 3 * 128);
    CUDA(cudaMalloc(&dev.seq_ctr, sizeof(u64)));
    CUDA(cudaMalloc(&dev.arrived, sizeof(unsigned)));
    CUDA(cudaMalloc(&dev.finished, sizeof(unsigned)));
    u64 one = 1;
    CUDA(cudaMemcpy(dev.seq_ctr, &one, sizeof(one), cudaMemcpyHostToDevice));
    CUDA(cudaMemset(dev.arrived, 0, sizeof(unsigned)));
    CUDA(cudaMemset(dev.finished, 0, sizeof(unsigned)));
  }
  ~Memory() {
    CUDA(cudaFree(dev.finished));
    CUDA(cudaFree(dev.arrived));
    CUDA(cudaFree(dev.seq_ctr));
    CUDA(cudaHostUnregister(host));
    std::free(host);
  }
};

// Fixed-width network-order endpoint record, including receive-region base.
struct Endpoint {
  uint32_t qpn, psn, rkey, mtu;
  uint32_t addr_hi, addr_lo;
  uint8_t gid[16];
};
static_assert(sizeof(Endpoint) == 40, "endpoint layout");
struct Transport {
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_mr *mr = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  u64 remote_base = 0;
  uint32_t remote_key = 0;
  Transport(const Options &o, Memory &m, Bootstrap &tcp) {
    int count = 0;
    ibv_device **list = ibv_get_device_list(&count);
    REQUIRE(list != nullptr, "ibv_get_device_list failed");
    for (int i = 0; i < count; ++i) {
      const char *name = ibv_get_device_name(list[i]);
      REQUIRE(name != nullptr, "ibv_get_device_name failed");
      if (o.ibdev == name) {
        ctx = ibv_open_device(list[i]);
        break;
      }
    }
    ibv_free_device_list(list);
    REQUIRE(ctx != nullptr, "ibdev not found or open failed");
    ibv_port_attr port{};
    IB(ibv_query_port(ctx, 1, &port));
    REQUIRE(port.state == IBV_PORT_ACTIVE &&
                port.link_layer == IBV_LINK_LAYER_ETHERNET,
            "port 1 must be active Ethernet/RoCE");
    REQUIRE(o.gid < port.gid_tbl_len && o.gid <= 255, "GID index out of range");
    ibv_gid gid{};
    IB(ibv_query_gid(ctx, 1, o.gid, &gid));
    bool nonzero = false;
    for (uint8_t b : gid.raw)
      nonzero |= b != 0;
    REQUIRE(nonzero, "selected GID is empty");
    pd = ibv_alloc_pd(ctx);
    REQUIRE(pd, "ibv_alloc_pd failed");
    // No RELAXED_ORDERING: proxy1 depends on payload-before-tail placement.
    mr = ibv_reg_mr(pd, m.host, m.bytes,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    REQUIRE(mr, "ibv_reg_mr failed (check memlock)");
    cq = ibv_create_cq(ctx, 256, nullptr, nullptr, 0);
    REQUIRE(cq, "ibv_create_cq failed");
    ibv_qp_init_attr init{};
    init.send_cq = cq;
    init.recv_cq = cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 256;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    init.cap.max_inline_data = 64;
    qp = ibv_create_qp(pd, &init);
    REQUIRE(qp, "ibv_create_qp failed");
    REQUIRE(init.cap.max_send_wr >= 256 && init.cap.max_inline_data >= 64,
            "insufficient QP capacity");
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    IB(ibv_modify_qp(qp, &a,
                     IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                         IBV_QP_ACCESS_FLAGS));
    uint32_t psn = (0x13579bu + uint32_t(o.rank) * 0x10203u) & 0xffffffu;
    u64 base = reinterpret_cast<uintptr_t>(m.host + 2 * m.slot);
    Endpoint local{htonl(qp->qp_num),
                   htonl(psn),
                   htonl(mr->rkey),
                   htonl(uint32_t(port.active_mtu)),
                   htonl(uint32_t(base >> 32)),
                   htonl(uint32_t(base)),
                   {}};
    std::memcpy(local.gid, gid.raw, 16);
    Endpoint peer{};
    tcp.exchange(&local, &peer, sizeof(peer));
    remote_base = (u64(ntohl(peer.addr_hi)) << 32) | ntohl(peer.addr_lo);
    remote_key = ntohl(peer.rkey);
    uint32_t mtu = std::min(uint32_t(port.active_mtu), ntohl(peer.mtu));
    REQUIRE(mtu >= IBV_MTU_256 && mtu <= IBV_MTU_4096, "invalid peer MTU");
    a = {};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = static_cast<ibv_mtu>(mtu);
    a.dest_qp_num = ntohl(peer.qpn);
    a.rq_psn = ntohl(peer.psn);
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = 1;
    std::memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
    a.ah_attr.grh.sgid_index = uint8_t(o.gid);
    a.ah_attr.grh.hop_limit = 64;
    IB(ibv_modify_qp(qp, &a,
                     IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
    a = {};
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = psn;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.max_rd_atomic = 1;
    IB(ibv_modify_qp(qp, &a,
                     IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                         IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                         IBV_QP_MAX_QP_RD_ATOMIC));
    tcp.barrier();
  }
  ~Transport() {
    IB(ibv_destroy_qp(qp));
    IB(ibv_destroy_cq(cq));
    IB(ibv_dereg_mr(mr));
    IB(ibv_dealloc_pd(pd));
    IB(ibv_close_device(ctx));
  }
};

// GPU <-> CPU doorbell and GPU <-> NIC polling, all in coherent host memory.
__device__ __forceinline__ u64 acquire_sys(const u64 *p) {
  u64 v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void release_sys(u64 *p, u64 v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}
__device__ __forceinline__ uint4 load_peer(const void *p) {
  uint4 v;
  asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ bool last_cta(unsigned *p) {
  cuda::atomic_ref<unsigned, cuda::thread_scope_device> a(*p);
  bool last = a.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1;
  if (last)
    a.store(0, cuda::memory_order_release);
  return last;
}
__device__ __forceinline__ unsigned sum_pair(unsigned local, unsigned peer,
                                             int rank) {
  unsigned result = 0;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    unsigned short a =
        static_cast<unsigned short>((rank == 0 ? local : peer) >> (16 * j));
    unsigned short b =
        static_cast<unsigned short>((rank == 0 ? peer : local) >> (16 * j));
    float s = __fadd_rn(__bfloat162float(__ushort_as_bfloat16(a)),
                        __bfloat162float(__ushort_as_bfloat16(b)));
    result |= unsigned(__bfloat16_as_ushort(__float2bfloat16_rn(s)))
              << (16 * j);
  }
  return result;
}
// Same entry point for eager and graph. No captured, host-valued sequence.
__global__ void ar_step(const bf16 *in, bf16 *out, int n, int rank, Region r) {
  __shared__ u64 seq;
  if (threadIdx.x == 0)
    seq = *r.seq_ctr;
  __syncthreads();
  int slot = int(seq & 1), bytes = n * 2;
  int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = int(gridDim.x * blockDim.x);
  const uint4 *src = reinterpret_cast<const uint4 *>(in);
  uint4 *send = reinterpret_cast<uint4 *>(r.send[slot]);
  for (int i = tid; i < n / 8; i += stride)
    send[i] = src[i];
  // Every writer fences, then its CTA leader joins an acquire/release chain.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (last_cta(r.arrived)) {
      if (r.one_write) {
        u64 *tail = reinterpret_cast<u64 *>(r.send[slot] + bytes);
        tail[1] = ~seq;
        release_sys(tail,
                    seq); // trailing 16 bytes written after all payload stores
      }
      __threadfence_system();
      release_sys(r.sflag[slot], (seq << 24) | u64(bytes));
    }
    const u64 *flag = r.one_write
                          ? reinterpret_cast<u64 *>(r.recv[slot] + bytes)
                          : r.rflag[slot];
    while (acquire_sys(flag) != seq) {
    }
  }
  __syncthreads();
  uint4 *dst = reinterpret_cast<uint4 *>(out);
  for (int i = tid; i < n / 8; i += stride) {
    uint4 a = src[i], b = load_peer(r.recv[slot] + size_t(i) * 16), c;
    c.x = sum_pair(a.x, b.x, rank);
    c.y = sum_pair(a.y, b.y, rank);
    c.z = sum_pair(a.z, b.z, rank);
    c.w = sum_pair(a.w, b.w, rank);
    dst[i] = c;
  }
  // Each CTA must finish its reads before any future kernel advances seq.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && last_cta(r.finished))
    *r.seq_ctr = seq + 1;
}
__global__ void ping_step(Region r, int rank) {
  u64 seq = *r.seq_ctr;
  int slot = int(seq & 1);
  if (rank == 1)
    while (acquire_sys(r.rflag[slot]) != seq) {
    }
  release_sys(r.sflag[slot], (seq << 24) | 16ULL);
  if (rank == 0)
    while (acquire_sys(r.rflag[slot]) != seq) {
    }
  *r.seq_ctr = seq + 1;
}

// CPU proxy: one post_send call per step (a two-WR chain for proxy2).
// Signal its final WR every 32 posts: at most 64 WRs outstanding, below 256.
struct Proxy {
  Transport &t;
  Memory &m;
  int cpu;
  std::atomic<bool> stop{false};
  std::atomic<int> one_write{0};
  std::atomic<u64> handled{0};
  std::thread worker;
  u64 posts = 0;
  Proxy(Transport &transport, Memory &memory, int core)
      : t(transport), m(memory), cpu(core) {
    worker = std::thread([this] { run(); });
  }
  void completion(u64 expected) {
    ibv_wc wc{};
    int n;
    do {
      n = ibv_poll_cq(t.cq, 1, &wc);
      REQUIRE(n >= 0, "ibv_poll_cq failed");
    } while (n == 0);
    if (wc.status != IBV_WC_SUCCESS) {
      std::fprintf(stderr, "%s:%d: CQ status=%d vendor_err=%u wr_id=%llu\n",
                   __FILE__, __LINE__, int(wc.status), wc.vendor_err,
                   (u64)wc.wr_id);
      std::exit(1);
    }
    REQUIRE(wc.wr_id == expected, "unexpected CQ completion");
  }
  void post_step(u64 seq, uint32_t bytes, bool one) {
    int slot = int(seq & 1), last = one ? 0 : 1;
    ibv_sge sg[2]{};
    ibv_send_wr wr[2]{};
    sg[0].addr = reinterpret_cast<uintptr_t>(m.host + slot * m.slot);
    sg[0].length = bytes + (one ? 16 : 0);
    sg[0].lkey = t.mr->lkey;
    sg[1].addr = reinterpret_cast<uintptr_t>(&seq);
    sg[1].length = sizeof(seq);
    // INLINE copies this stack value during post_send; no MR/lkey is needed.
    for (int i = 0; i <= last; ++i) {
      wr[i].opcode = IBV_WR_RDMA_WRITE;
      wr[i].sg_list = &sg[i];
      wr[i].num_sge = 1;
      wr[i].wr.rdma.rkey = t.remote_key;
    }
    wr[0].wr.rdma.remote_addr = t.remote_base + slot * m.slot;
    if (!one) {
      wr[0].next = &wr[1];
      wr[1].send_flags = IBV_SEND_INLINE;
      // remote_base is recv[0], not the beginning of the allocation.
      wr[1].wr.rdma.remote_addr = t.remote_base + 2 * m.slot + (slot + 1) * 128;
    }
    wr[last].wr_id = ++posts;
    bool signal = posts % 32 == 0;
    if (signal)
      wr[last].send_flags |= IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    IB(ibv_post_send(t.qp, wr, &bad));
    if (signal)
      completion(posts);
  }
  void run() {
    pin(cpu);
    u64 seen = 0;
    volatile const u64 *flags[2] = {
        reinterpret_cast<volatile const u64 *>(m.host + m.flag_offset),
        reinterpret_cast<volatile const u64 *>(m.host + m.flag_offset +
                                               3 * 128)};
    while (!stop.load(std::memory_order_acquire)) {
      u64 doorbell = *flags[(seen + 1) & 1], seq = doorbell >> 24;
      if (seq != seen + 1)
        continue; // slot still holds step seen-1 (or 0)
      std::atomic_thread_fence(std::memory_order_acquire);
      REQUIRE(seq == seen + 1 && seq < SEQ_LIMIT,
              "proxy sequence skipped or wrapped");
      uint32_t bytes = uint32_t(doorbell & 0xffffffULL);
      REQUIRE(bytes && bytes + 16 <= m.slot, "invalid doorbell size");
      bool one = one_write.load(std::memory_order_acquire) != 0;
      post_step(seq, bytes, one);
      seen = seq;
      handled.store(seen, std::memory_order_release);
    }
    // Shutdown-only CQE retires any final unsignaled WRs before MR/QP teardown.
    if (posts % 32) {
      ibv_send_wr wr{}, *bad = nullptr;
      wr.wr_id = ++posts;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.wr.rdma.remote_addr = t.remote_base;
      wr.wr.rdma.rkey = t.remote_key;
      IB(ibv_post_send(t.qp, &wr, &bad));
      completion(posts);
    }
  }
  void wait(u64 seq) {
    while (handled.load(std::memory_order_acquire) < seq) {
    }
  }
  void shutdown() {
    stop.store(true, std::memory_order_release);
    if (worker.joinable())
      worker.join();
  }
  ~Proxy() {
    if (worker.joinable())
      shutdown();
  }
};

// Identical host/device PRNG; host reference rounds finite FP32 to BF16, RNE.
__host__ __device__ static uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__host__ __device__ static float sample(int rank, int step, int i) {
  uint32_t h = mix(uint32_t(i) ^ (uint32_t(step) + 1) * 0x9e3779b9u ^
                   (uint32_t(rank) + 1) * 0x85ebca6bu);
  return (int(h % 65537u) - 32768) * (1.0f / 8192.0f);
}
__global__ void fill(bf16 *p, int n, int rank, int step) {
  for (int i = int(blockIdx.x * blockDim.x + threadIdx.x); i < n;
       i += int(gridDim.x * blockDim.x))
    p[i] = __float2bfloat16_rn(sample(rank, step, i));
}
static uint16_t to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static float from_bf16(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
static uint16_t reference(int step, int i) {
  float rank0 = from_bf16(to_bf16(sample(0, step, i)));
  float rank1 = from_bf16(to_bf16(sample(1, step, i)));
  return to_bf16(rank0 + rank1);
}

enum Method { Nccl = 0, Proxy2 = 1, Proxy1 = 2 };
static const char *names[] = {"nccl", "proxy2", "proxy1"};
struct Result {
  size_t bytes;
  std::string method, mode;
  double median, minimum;
  u64 steps;
};
static Result result(size_t bytes, const char *method, const char *mode,
                     std::vector<double> v, u64 steps) {
  std::sort(v.begin(), v.end());
  return {bytes, method, mode, v[v.size() / 2], v.front(), steps};
}

// Benchmark driver. Host next_seq is an audit only; device sequence drives all
// launches.
struct Bench {
  const Options &o;
  Bootstrap &tcp;
  Memory &mem;
  Proxy &proxy;
  ncclComm_t comm{};
  cudaStream_t stream{};
  cudaEvent_t start{}, end{};
  bf16 *in = nullptr, *out = nullptr;
  u64 next_seq = 1;
  bool check_failed = false;
  std::vector<Result> results;
  Bench(const Options &opts, Bootstrap &b, Memory &m, Proxy &p, ncclUniqueId id)
      : o(opts), tcp(b), mem(m), proxy(p) {
    NCCL(ncclCommInitRank(&comm, 2, id, o.rank));
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&end));
    size_t max_size = *std::max_element(o.sizes.begin(), o.sizes.end());
    CUDA(cudaMalloc(&in, max_size));
    CUDA(cudaMalloc(&out, max_size));
    int per_sm = 0, sms = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ar_step, 512,
                                                       0));
    CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    REQUIRE(o.ctas <= per_sm * sms,
            "--ctas exceeds resident grid capacity (spin-barrier deadlock)");
  }
  void launch(Method method, int n) {
    if (method == Nccl)
      NCCL(ncclAllReduce(in, out, size_t(n), ncclBfloat16, ncclSum, comm,
                         stream));
    else {
      Region r = mem.dev;
      r.one_write = method == Proxy1;
      ar_step<<<o.ctas, 512, 0, stream>>>(in, out, n, o.rank, r);
      CUDA(cudaGetLastError());
    }
  }
  void account(Method method, u64 steps) {
    if (method != Nccl)
      next_seq += steps;
  }
  void settle() {
    CUDA(cudaStreamSynchronize(stream));
    proxy.wait(next_seq - 1);
    u64 actual = 0;
    CUDA(cudaMemcpy(&actual, mem.dev.seq_ctr, sizeof(actual),
                    cudaMemcpyDeviceToHost));
    REQUIRE(actual == next_seq, "host/device sequence audit failed");
  }
  void prepare(Method method) {
    settle();
    proxy.one_write.store(method == Proxy1, std::memory_order_release);
    tcp.barrier();
  }
  void checks(size_t bytes) {
    int n = int(bytes / 2);
    std::vector<uint16_t> baseline(size_t(CHECKS) * n), got(n);
    for (Method method : {Nccl, Proxy2, Proxy1}) {
      prepare(method);
      u64 bad_ref = 0, bad_nccl = 0, bad_peer = 0;
      int first_step = -1, first_index = -1;
      std::vector<uint16_t> peer(n);
      for (int step = 0; step < CHECKS; ++step) {
        fill<<<std::min(256, (n + 255) / 256), 256, 0, stream>>>(in, n, o.rank,
                                                                 step);
        CUDA(cudaGetLastError());
        launch(method, n);
        account(method, 1);
        settle();
        CUDA(cudaMemcpy(got.data(), out, bytes, cudaMemcpyDeviceToHost));
        // Exact peer comparison also checks bitwise equality across ranks.
        tcp.exchange(got.data(), peer.data(), bytes);
        for (int i = 0; i < n; ++i) {
          size_t j = size_t(step) * n + i;
          bool ref_bad = got[i] != reference(step, i);
          bool nccl_bad = method != Nccl && got[i] != baseline[j];
          bool peer_bad = got[i] != peer[i];
          bad_ref += ref_bad;
          bad_nccl += nccl_bad;
          bad_peer += peer_bad;
          if ((ref_bad || nccl_bad || peer_bad) && first_step < 0) {
            first_step = step;
            first_index = i;
          }
          if (method == Nccl)
            baseline[j] = got[i];
        }
      }
      bool ok = bad_ref == 0 && bad_nccl == 0 && bad_peer == 0;
      check_failed |= !ok;
      // Serialize check lines so two MPI stdout streams remain readable.
      for (int rank = 0; rank < 2; ++rank) {
        tcp.barrier();
        if (rank == o.rank) {
          std::printf("CHECK %s rank=%d size_bytes=%zu method=%s steps=%d "
                      "ref_mismatches=%llu "
                      "nccl_mismatches=%llu peer_mismatches=%llu first_step=%d "
                      "first_index=%d\n",
                      ok ? "ok" : "FAIL", o.rank, bytes, names[method], CHECKS,
                      bad_ref, bad_nccl, bad_peer, first_step, first_index);
          std::fflush(stdout);
        }
      }
      tcp.barrier();
    }
  }
  double timed(Method method, int n, cudaGraphExec_t graph) {
    prepare(method); // includes TCP barrier immediately before each timed block
    int repeats = graph ? o.iters / o.graph_ops : o.iters;
    u64 steps = graph ? u64(repeats) * o.graph_ops : u64(repeats);
    CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      if (graph)
        CUDA(cudaGraphLaunch(graph, stream));
      else
        launch(method, n);
    }
    CUDA(cudaEventRecord(end, stream));
    CUDA(cudaEventSynchronize(end));
    float ms = 0;
    CUDA(cudaEventElapsedTime(&ms, start, end));
    account(method, steps);
    settle();
    return double(ms) * 1000.0 / double(steps);
  }
  void size(size_t bytes) {
    // A smaller new size can put its tail inside a previous payload. Clear
    // that location while both ranks are idle, before any new proxy steps.
    settle();
    tcp.barrier();
    for (int slot = 0; slot < 2; ++slot)
      std::memset(mem.host + (slot + 2) * mem.slot + bytes, 0, 16);
    std::atomic_thread_fence(std::memory_order_release);
    tcp.barrier();
    checks(bytes);
    int n = int(bytes / 2);
    std::array<cudaGraphExec_t, 3> graphs{};
    for (Method method : {Nccl, Proxy2, Proxy1}) {
      prepare(method);
      fill<<<std::min(256, (n + 255) / 256), 256, 0, stream>>>(in, n, o.rank,
                                                               0);
      CUDA(cudaGetLastError());
      for (int i = 0; i < WARMUP; ++i)
        launch(method, n);
      account(method, WARMUP);
      settle();
      tcp.barrier();
      cudaGraph_t graph{};
      CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < o.graph_ops; ++i)
        launch(method, n);
      CUDA(cudaStreamEndCapture(stream, &graph));
      CUDA(cudaGraphInstantiate(&graphs[method], graph, nullptr, nullptr, 0));
      CUDA(cudaGraphDestroy(graph));
      // Exactly 200 graph warmup steps: full graph replays plus eager
      // remainder.
      for (int i = 0; i < WARMUP / o.graph_ops; ++i)
        CUDA(cudaGraphLaunch(graphs[method], stream));
      for (int i = 0; i < WARMUP % o.graph_ops; ++i)
        launch(method, n);
      account(method, WARMUP);
      settle();
      // Upload even if graph_ops > warmup, keeping first-upload cost outside
      // timing.
      CUDA(cudaGraphUpload(graphs[method], stream));
      CUDA(cudaStreamSynchronize(stream));
    }
    std::array<std::array<std::vector<double>, 2>, 3> times;
    std::mt19937 rng(0x554152u);
    for (int round = 0; round < ROUNDS; ++round) {
      std::array<Method, 3> order{Nccl, Proxy2, Proxy1};
      std::shuffle(order.begin(), order.end(), rng);
      for (Method method : order) {
        times[method][0].push_back(timed(method, n, nullptr));
        times[method][1].push_back(timed(method, n, graphs[method]));
      }
    }
    for (Method method : {Nccl, Proxy2, Proxy1}) {
      results.push_back(
          result(bytes, names[method], "eager", times[method][0], o.iters));
      results.push_back(result(bytes, names[method], "graph", times[method][1],
                               u64(o.iters / o.graph_ops) * o.graph_ops));
      CUDA(cudaGraphExecDestroy(graphs[method]));
    }
  }
  void pingpong() {
    prepare(Proxy2);
    for (int i = 0; i < WARMUP; ++i) {
      ping_step<<<1, 1, 0, stream>>>(mem.dev, o.rank);
      CUDA(cudaGetLastError());
    }
    next_seq += WARMUP;
    settle();
    std::vector<double> times;
    for (int round = 0; round < ROUNDS; ++round) {
      prepare(Proxy2);
      CUDA(cudaEventRecord(start, stream));
      for (int i = 0; i < o.iters; ++i) {
        ping_step<<<1, 1, 0, stream>>>(mem.dev, o.rank);
        CUDA(cudaGetLastError());
      }
      CUDA(cudaEventRecord(end, stream));
      CUDA(cudaEventSynchronize(end));
      float ms = 0;
      CUDA(cudaEventElapsedTime(&ms, start, end));
      next_seq += o.iters;
      settle();
      times.push_back(double(ms) * 1000.0 / o.iters / 2.0);
    }
    results.push_back(result(0, "pingpong", "eager", times, o.iters));
  }
  void report() {
    if (o.rank != 0)
      return;
    for (const auto &r : results)
      std::printf("{\"size_bytes\":%zu,\"method\":\"%s\",\"mode\":\"%s\","
                  "\"median_us\":%.6f,"
                  "\"min_us\":%.6f,\"rounds\":%d,\"steps_per_round\":%llu}\n",
                  r.bytes, r.method.c_str(), r.mode.c_str(), r.median,
                  r.minimum, ROUNDS, r.steps);
    std::puts("size_bytes method mode median_us min_us");
    for (const auto &r : results)
      std::printf("%10zu %-8s %-5s %10.3f %10.3f\n", r.bytes, r.method.c_str(),
                  r.mode.c_str(), r.median, r.minimum);
    std::fflush(stdout);
  }
  ~Bench() {
    settle();
    NCCL(ncclCommDestroy(comm));
    CUDA(cudaFree(out));
    CUDA(cudaFree(in));
    CUDA(cudaEventDestroy(end));
    CUDA(cudaEventDestroy(start));
    CUDA(cudaStreamDestroy(stream));
  }
};

int main(int argc, char **argv) {
  Options o = parse(argc, argv);
  if (o.main_cpu >= 0)
    pin(o.main_cpu);
  else
    pin_big_except(o.proxy_cpu);
  CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
  CUDA(cudaSetDevice(0));
  int mapped = 0, atomics = 0, integrated = 0;
  CUDA(cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, 0));
  CUDA(cudaDeviceGetAttribute(&atomics, cudaDevAttrHostNativeAtomicSupported,
                              0));
  CUDA(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, 0));
  REQUIRE(mapped && atomics && integrated,
          "requires integrated GPU with mapped host memory and native host "
          "atomics");
  Bootstrap tcp(o);
  tcp.check_config(o);
  ncclUniqueId id{};
  if (o.rank == 0) {
    NCCL(ncclGetUniqueId(&id));
    tcp.send(&id, sizeof(id));
  } else
    tcp.recv(&id, sizeof(id));
  Memory memory(*std::max_element(o.sizes.begin(), o.sizes.end()));
  Transport transport(o, memory, tcp);
  Proxy proxy(transport, memory, o.proxy_cpu);
  int failed = 0;
  {
    Bench bench(o, tcp, memory, proxy, id);
    for (size_t bytes : o.sizes)
      bench.size(bytes);
    bench.pingpong();
    bench.report();
    bench.settle();
    tcp.barrier();
    uint8_t local = bench.check_failed, peer = 0;
    tcp.exchange(&local, &peer, 1);
    failed = local || peer;
  }
  proxy.shutdown();
  tcp.barrier();
  return failed ? 1 : 0;
}
