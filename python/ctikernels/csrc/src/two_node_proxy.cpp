// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
// Protocol follows our two_node_allreduce_probe.cu proxy2 experiment.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if __has_include(<infiniband/verbs.h>) && !defined(CTI_PROXY_NO_VERBS)
#define CTI_PROXY_VERBS 1
#include <arpa/inet.h>
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#define CTI_PROXY_VERBS 0
#endif

namespace {
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Bytes;
using tvm::ffi::Map;
using tvm::ffi::String;
using u64 = uint64_t;

void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

#if CTI_PROXY_VERBS
// Own names deliberately avoid verbs.h's function-like macros. Only post_send
// and poll_cq use the header's context-ops inlines. No linked verbs symbols.
struct Verbs {
  void *library = nullptr;
  std::string error;
  ibv_device **(*devices)(int *) = nullptr;
  void (*free_devices)(ibv_device **) = nullptr;
  const char *(*device_name)(ibv_device *) = nullptr;
  ibv_context *(*open)(ibv_device *) = nullptr;
  int (*close)(ibv_context *) = nullptr;
  // The exported legacy query_port ABI fills the stable prefix through
  // link_layer; newer extended fields are unused and remain zero-initialized.
  int (*port)(ibv_context *, uint8_t, _compat_ibv_port_attr *) = nullptr;
  int (*gid)(ibv_context *, uint8_t, int, ibv_gid *) = nullptr;
  ibv_pd *(*alloc_pd)(ibv_context *) = nullptr;
  int (*dealloc_pd)(ibv_pd *) = nullptr;
  ibv_mr *(*reg)(ibv_pd *, void *, size_t, int) = nullptr;
  int (*dereg)(ibv_mr *) = nullptr;
  ibv_cq *(*create_cq)(ibv_context *, int, void *, ibv_comp_channel *,
                       int) = nullptr;
  int (*destroy_cq)(ibv_cq *) = nullptr;
  ibv_qp *(*create_qp)(ibv_pd *, ibv_qp_init_attr *) = nullptr;
  int (*destroy_qp)(ibv_qp *) = nullptr;
  int (*modify)(ibv_qp *, ibv_qp_attr *, int) = nullptr;

  template <typename T> void bind(T &target, const char *name) {
    target = reinterpret_cast<T>(dlsym(library, name));
    if (!target && error.empty())
      error = std::string("libibverbs missing entry point: ") + name;
  }
  Verbs() {
    library = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      error = "cannot load libibverbs.so.1";
      return;
    }
    bind(devices, "ibv_get_device_list");
    bind(free_devices, "ibv_free_device_list");
    bind(device_name, "ibv_get_device_name");
    bind(open, "ibv_open_device");
    bind(close, "ibv_close_device");
    bind(port, "ibv_query_port");
    bind(gid, "ibv_query_gid");
    bind(alloc_pd, "ibv_alloc_pd");
    bind(dealloc_pd, "ibv_dealloc_pd");
    bind(reg, "ibv_reg_mr");
    bind(dereg, "ibv_dereg_mr");
    bind(create_cq, "ibv_create_cq");
    bind(destroy_cq, "ibv_destroy_cq");
    bind(create_qp, "ibv_create_qp");
    bind(destroy_qp, "ibv_destroy_qp");
    bind(modify, "ibv_modify_qp");
  }
  // Keep the library alive through all FFI object destructors.
};
Verbs &verbs() {
  static Verbs v;
  return v;
}
u64 now_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return u64(t.tv_sec) * 1000000000 + t.tv_nsec;
}
struct Endpoint {
  uint32_t qpn, psn, rkey, mtu, high, low;
  uint8_t gid[16];
};
static_assert(sizeof(Endpoint) == 40);
#endif

std::string available() {
#if CTI_PROXY_VERBS
  return verbs().error;
#else
  return "built without infiniband/verbs.h";
#endif
}

class TwoNodeProxy : public tvm::ffi::Object {
public:
  TwoNodeProxy() { require(available().empty(), available().c_str()); }
  ~TwoNodeProxy() { shutdown(); }
#if CTI_PROXY_VERBS
  void init(std::string device, int gid_index, int64_t slot_bytes) {
    require(!host && !closed, "proxy already initialized or closed");
    require(slot_bytes > 0 && slot_bytes < (1 << 24), "invalid slot size");
    require(gid_index >= 0 && gid_index <= 255, "invalid GID index");
    v = &verbs();
    try {
      int count = 0;
      auto list = v->devices(&count);
      require(list != nullptr, "device enumeration failed");
      for (int i = 0; i < count; ++i) {
        const char *name = v->device_name(list[i]);
        if (name && device == name) {
          ctx = v->open(list[i]);
          break;
        }
      }
      v->free_devices(list);
      require(ctx != nullptr, "RDMA device unavailable");
      ibv_port_attr p{};
      require(v->port(ctx, 1, reinterpret_cast<_compat_ibv_port_attr *>(&p)) ==
                  0,
              "port query failed");
      require(p.state == IBV_PORT_ACTIVE &&
                  p.link_layer == IBV_LINK_LAYER_ETHERNET,
              "port 1 must be ACTIVE Ethernet");
      require(gid_index < p.gid_tbl_len, "GID index out of range");
      ibv_gid g{};
      require(v->gid(ctx, 1, gid_index, &g) == 0, "GID query failed");
      require(std::any_of(g.raw, g.raw + 16, [](uint8_t b) { return b != 0; }),
              "selected GID is empty");
      gid_idx = gid_index;
      capacity = slot_bytes;
      slot = (slot_bytes + 127) & ~size_t(127);
      constexpr size_t alignment = 2 << 20;
      region_size = (4 * slot + 8 * 128 + alignment - 1) & ~(alignment - 1);
      void *allocation = nullptr;
      require(posix_memalign(&allocation, alignment, region_size) == 0,
              "host allocation failed");
      host = static_cast<char *>(allocation);
      std::memset(host, 0, region_size);
      pd = v->alloc_pd(ctx);
      require(pd != nullptr, "PD allocation failed");
      mr = v->reg(pd, host, region_size,
                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      require(mr != nullptr, "MR registration failed (check memlock)");
      cq = v->create_cq(ctx, 256, nullptr, nullptr, 0);
      require(cq != nullptr, "CQ creation failed");
      ibv_qp_init_attr q{};
      q.send_cq = q.recv_cq = cq;
      q.qp_type = IBV_QPT_RC;
      q.cap.max_send_wr = 256;
      q.cap.max_recv_wr = 1;
      q.cap.max_send_sge = q.cap.max_recv_sge = 1;
      q.cap.max_inline_data = 64;
      qp = v->create_qp(pd, &q);
      require(qp != nullptr, "QP creation failed");
      require(q.cap.max_send_wr >= 256 && q.cap.max_inline_data >= 64,
              "insufficient QP capacity");
      ibv_qp_attr a{};
      a.qp_state = IBV_QPS_INIT;
      a.port_num = 1;
      a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
      require(v->modify(qp, &a,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                            IBV_QP_ACCESS_FLAGS) == 0,
              "QP INIT failed");
      psn = uint32_t(now_ns()) & 0xffffff;
      u64 base = reinterpret_cast<uintptr_t>(host + 2 * slot);
      endpoint = {htonl(qp->qp_num),
                  htonl(psn),
                  htonl(mr->rkey),
                  htonl(p.active_mtu),
                  htonl(uint32_t(base >> 32)),
                  htonl(uint32_t(base)),
                  {}};
      std::memcpy(endpoint.gid, g.raw, 16);
    } catch (...) {
      shutdown();
      throw;
    }
  }
  int64_t host_address() const { return reinterpret_cast<intptr_t>(host); }
  int64_t region_bytes() const { return region_size; }
  Array<int64_t> layout() const {
    return {0,
            int64_t(slot),
            int64_t(2 * slot),
            int64_t(3 * slot),
            int64_t(4 * slot),
            int64_t(4 * slot + 128),
            int64_t(4 * slot + 256),
            int64_t(4 * slot + 384),
            int64_t(4 * slot + 512),
            int64_t(slot)};
  }
  Bytes local_endpoint() const {
    require(qp && !closed, "proxy not initialized");
    return Bytes(reinterpret_cast<const char *>(&endpoint), sizeof(endpoint));
  }
  void connect(Bytes bytes) {
    require(qp && !connected && !closed, "invalid connect state");
    require(bytes.size() == sizeof(Endpoint), "endpoint must contain 40 bytes");
    Endpoint peer{};
    std::memcpy(&peer, bytes.data(), sizeof(peer));
    auto mtu = ntohl(peer.mtu);
    require(mtu >= IBV_MTU_256 && mtu <= IBV_MTU_4096, "invalid peer MTU");
    require(ntohl(peer.qpn) > 0 && ntohl(peer.qpn) <= 0xffffff &&
                ntohl(peer.psn) <= 0xffffff,
            "invalid peer QPN/PSN");
    remote = (u64(ntohl(peer.high)) << 32) | ntohl(peer.low);
    key = ntohl(peer.rkey);
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = static_cast<ibv_mtu>(std::min(mtu, ntohl(endpoint.mtu)));
    a.dest_qp_num = ntohl(peer.qpn);
    a.rq_psn = ntohl(peer.psn);
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = 1;
    std::memcpy(a.ah_attr.grh.dgid.raw, peer.gid, 16);
    a.ah_attr.grh.sgid_index = uint8_t(gid_idx);
    a.ah_attr.grh.hop_limit = 64;
    require(v->modify(qp, &a,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) ==
                0,
            "QP RTR failed");
    a = {};
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = psn;
    a.timeout = 14;
    a.retry_cnt = a.rnr_retry = 7;
    a.max_rd_atomic = 1;
    require(v->modify(qp, &a,
                      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT |
                          IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_MAX_QP_RD_ATOMIC) == 0,
            "QP RTS failed");
    connected = true;
  }
  void start(int cpu, int hot_us, int warm_us, int cold_poll_us) {
    require(connected && !started && !closed, "invalid start state");
    require(cpu >= -1 && cpu < CPU_SETSIZE, "invalid proxy CPU");
    require(hot_us >= 0 && warm_us >= 0 && cold_poll_us > 0, "invalid backoff");
    hot_ns = u64(hot_us) * 1000;
    warm_ns = u64(warm_us) * 1000;
    cold_ns = u64(cold_poll_us) * 1000;
    start_ns = now_ns();
    last_activity.store(start_ns);
    // Bounded ready-or-error handshake: worker startup failures (e.g. a CPU
    // outside the process cpuset) raise here, before the caller's agreement.
    auto ready = std::make_shared<std::promise<std::string>>();
    auto result = ready->get_future();
    worker = std::thread([this, cpu, ready] { run(cpu, *ready); });
    if (result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      stopping.store(true, std::memory_order_release);
      wake();
      throw std::runtime_error("proxy worker did not report ready within 5 s");
    }
    auto error = result.get();
    if (!error.empty()) {
      worker.join();
      throw std::runtime_error("proxy worker startup failed: " + error);
    }
    started = true;
  }
  void arm(int64_t hold_us) {
    require(started && !closed, "proxy not running");
    require(hold_us <= int64_t(std::numeric_limits<u64>::max() / 2000),
            "hold too large");
    auto now = now_ns();
    last_activity.store(now, std::memory_order_release);
    armed_until.store(hold_us < 0 ? std::numeric_limits<u64>::max()
                                  : now + u64(hold_us) * 1000,
                      std::memory_order_release);
    state.store(0, std::memory_order_release);
    wake();
  }
  void disarm() {
    armed_until.store(0, std::memory_order_release);
    wake();
  }
  Map<String, Any> stats() {
    std::lock_guard<std::mutex> lock(error_mutex);
    return {{"steps", int64_t(handled.load())},
            {"posts", int64_t(posts.load())},
            {"cpu_ns", int64_t(cpu_ns.load())},
            {"wall_ns",
             int64_t(started ? (closed ? end_ns : now_ns()) - start_ns : 0)},
            {"futex_sleeps", int64_t(sleeps.load())},
            {"futex_wakeups", int64_t(wakeups.load())},
            {"state", String(states[state.load()])},
            { "error",
              String(error) }};
  }
  bool wait_handled(int64_t seq, int timeout_ms) {
    require(seq >= 0 && timeout_ms >= 0, "invalid wait");
    auto deadline = now_ns() + u64(timeout_ms) * 1000000;
    while (handled.load(std::memory_order_acquire) < u64(seq)) {
      if (failed.load() || now_ns() >= deadline || closed)
        return false;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
  }
  void shutdown() {
    if (closed)
      return;
    stopping.store(true, std::memory_order_release);
    wake();
    if (worker.joinable())
      worker.join();
    // run() drains before returning. On error, destroying the QP retires its
    // remaining work before the MR is deregistered or its storage freed.
    if (qp)
      v->destroy_qp(qp);
    if (cq)
      v->destroy_cq(cq);
    if (mr)
      v->dereg(mr);
    if (pd)
      v->dealloc_pd(pd);
    if (ctx)
      v->close(ctx);
    qp = nullptr;
    cq = nullptr;
    mr = nullptr;
    pd = nullptr;
    ctx = nullptr;
    std::free(host);
    host = nullptr;
    end_ns = now_ns();
    state.store(3);
    closed = true;
  }

private:
  u64 *word(size_t line) {
    return reinterpret_cast<u64 *>(host + 4 * slot + line * 128);
  }
  void wake() {
    futex_word.fetch_add(1, std::memory_order_release);
    syscall(SYS_futex, &futex_word, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
  }
  void sample_cpu(clockid_t clock) {
    timespec t{};
    if (clock_gettime(clock, &t) == 0)
      cpu_ns.store(u64(t.tv_sec) * 1000000000 + t.tv_nsec,
                   std::memory_order_relaxed);
  }
  void fail(const char *message) {
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      error = message;
    }
    failed.store(true, std::memory_order_release);
    __atomic_store_n(word(4), u64(1), __ATOMIC_RELEASE);
  }
  void completion(u64 expected) {
    const auto deadline = now_ns() + 5000000000ULL;
    ibv_wc wc{};
    unsigned polls = 0;
    while (true) {
      int n = ibv_poll_cq(cq, 1, &wc);
      require(n >= 0, "CQ poll failed");
      if (n) {
        require(wc.status == IBV_WC_SUCCESS, "CQ work completion error");
        require(wc.wr_id == expected, "unexpected CQ completion");
        return;
      }
      if ((++polls & 255) == 0)
        require(now_ns() < deadline, "CQ completion timeout");
    }
  }
  void check_cq() {
    ibv_wc wc{};
    int n = ibv_poll_cq(cq, 1, &wc);
    require(n >= 0, "CQ poll failed");
    // All successful signaled CQEs are consumed synchronously. An unsolicited
    // CQE is an error, including a flushed unsignaled WR.
    require(n == 0, "asynchronous CQ error");
  }
  void post(u64 seq, uint32_t bytes) {
    int parity = seq & 1;
    ibv_sge sg[2]{};
    ibv_send_wr wr[2]{};
    sg[0] = {reinterpret_cast<uintptr_t>(host + parity * slot), bytes,
             mr->lkey};
    sg[1] = {reinterpret_cast<uintptr_t>(&seq), 8, 0};
    for (int i = 0; i < 2; ++i) {
      wr[i].opcode = IBV_WR_RDMA_WRITE;
      wr[i].sg_list = &sg[i];
      wr[i].num_sge = 1;
      wr[i].wr.rdma.rkey = key;
    }
    wr[0].next = &wr[1];
    wr[0].wr.rdma.remote_addr = remote + parity * slot;
    wr[1].wr.rdma.remote_addr = remote + 2 * slot + (2 + parity) * 128;
    wr[1].send_flags = IBV_SEND_INLINE;
    u64 count = posts.fetch_add(1) + 1;
    wr[1].wr_id = count;
    if (count % 32 == 0)
      wr[1].send_flags |= IBV_SEND_SIGNALED;
    ibv_send_wr *bad = nullptr;
    require(ibv_post_send(qp, wr, &bad) == 0, "RDMA post failed");
    if (count % 32 == 0)
      completion(count);
  }
  void drain() {
    if (posts.load() % 32 == 0)
      return;
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = posts.fetch_add(1) + 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = remote;
    wr.wr.rdma.rkey = key;
    require(ibv_post_send(qp, &wr, &bad) == 0, "drain post failed");
    completion(wr.wr_id);
  }
  void run(int cpu, std::promise<std::string> &ready) noexcept {
    clockid_t clock = CLOCK_THREAD_CPUTIME_ID;
    pthread_getcpuclockid(pthread_self(), &clock);
    if (cpu >= 0) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(cpu, &mask);
      if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
        state.store(3);
        ready.set_value("proxy CPU affinity failed");
        return;
      }
    }
    ready.set_value("");
    try {
      u64 seen = 0;
      unsigned polls = 0;
      while (!stopping.load(std::memory_order_acquire)) {
        // Snapshot before checking work: arm between this check and FUTEX_WAIT
        // changes the word, making the wait return EAGAIN instead of losing it.
        uint32_t ticket = futex_word.load(std::memory_order_acquire);
        u64 bell = __atomic_load_n(word((seen + 1) & 1), __ATOMIC_ACQUIRE);
        u64 seq = bell >> 24;
        require(seq <= seen + 1 && seq < (1ULL << 39),
                "doorbell sequence skipped or wrapped");
        if (seq == seen + 1) {
          uint32_t bytes = bell & 0xffffff;
          require(bytes && bytes <= capacity && bytes % 16 == 0,
                  "invalid doorbell size");
          post(seq, bytes);
          handled.store(++seen, std::memory_order_release);
          last_activity.store(now_ns(), std::memory_order_release);
          state.store(0);
          polls = 0;
          continue;
        }
        if (state.load(std::memory_order_relaxed) == 0 && (++polls & 255))
          continue;
        auto now = now_ns();
        sample_cpu(clock);
        check_cq();
        auto last = last_activity.load(std::memory_order_acquire);
        auto armed = armed_until.load(std::memory_order_acquire);
        if (now < armed || now - std::min(now, last) < hot_ns) {
          state.store(0);
        } else if (now - std::min(now, last) < hot_ns + warm_ns) {
          state.store(1);
          sched_yield();
        } else {
          state.store(2);
          timespec timeout{time_t(cold_ns / 1000000000),
                           long(cold_ns % 1000000000)};
          sleeps.fetch_add(1);
          auto result = syscall(SYS_futex, &futex_word, FUTEX_WAIT_PRIVATE,
                                ticket, &timeout, nullptr, 0);
          if (result == 0)
            wakeups.fetch_add(1);
        }
      }
      drain();
    } catch (const std::exception &e) {
      fail(e.what());
    } catch (...) {
      fail("unknown proxy error");
    }
    sample_cpu(clock);
    state.store(3);
  }
  Verbs *v = nullptr;
  ibv_context *ctx = nullptr;
  ibv_pd *pd = nullptr;
  ibv_mr *mr = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  char *host = nullptr;
  size_t slot = 0, capacity = 0, region_size = 0;
  Endpoint endpoint{};
  int gid_idx = 0;
  uint32_t psn = 0, key = 0;
  u64 remote = 0, start_ns = 0, end_ns = 0, hot_ns = 0, warm_ns = 0,
      cold_ns = 0;
  bool connected = false, started = false, closed = false;
  std::atomic<bool> stopping{false}, failed{false};
  std::atomic<u64> handled{0}, posts{0}, cpu_ns{0}, sleeps{0}, wakeups{0};
  std::atomic<u64> armed_until{0}, last_activity{0};
  alignas(4) std::atomic<uint32_t> futex_word{0};
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  static_assert(sizeof(std::atomic<uint32_t>) == 4);
  std::atomic<int> state{3};
  static constexpr const char *states[] = {"HOT", "WARM", "COLD", "STOPPED"};
  std::mutex error_mutex;
  std::string error;
  std::thread worker;

public:
#else
  void init(std::string, int, int64_t) { require(false, available().c_str()); }
  int64_t host_address() const { return 0; }
  int64_t region_bytes() const { return 0; }
  Array<int64_t> layout() const { return {}; }
  Bytes local_endpoint() const { return Bytes(); }
  void connect(Bytes) { require(false, available().c_str()); }
  void start(int, int, int, int) { require(false, available().c_str()); }
  void arm(int64_t) { require(false, available().c_str()); }
  void disarm() {}
  Map<String, Any> stats() { return {{"error", String(available())}}; }
  bool wait_handled(int64_t, int) { return false; }
  void shutdown() {}
#endif
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ctiengine.TwoNodeProxy", TwoNodeProxy,
                                    tvm::ffi::Object);
};
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TwoNodeProxy>()
      .def(refl::init<>(), "__init__")
      .def("init", &TwoNodeProxy::init)
      .def("host_address", &TwoNodeProxy::host_address)
      .def("region_bytes", &TwoNodeProxy::region_bytes)
      .def("layout", &TwoNodeProxy::layout)
      .def("local_endpoint", &TwoNodeProxy::local_endpoint)
      .def("connect", &TwoNodeProxy::connect)
      .def("start", &TwoNodeProxy::start)
      .def("arm", &TwoNodeProxy::arm)
      .def("disarm", &TwoNodeProxy::disarm)
      .def("stats", &TwoNodeProxy::stats)
      .def("wait_handled", &TwoNodeProxy::wait_handled)
      .def("shutdown", &TwoNodeProxy::shutdown);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(two_node_proxy_available, &available);
} // namespace
