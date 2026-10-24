# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""B1 all-reduce and all-gather: two processes, one integrated sm_12x GPU per host.

Construct collectively outside capture after torch.cuda.set_device. Bootstrap
uses only the supplied torch.distributed.Store: prefix/{rank}/ep and /config,
plus the prefix/opened and /started agreements. Use a fresh prefix per instance.
One 2-MiB-aligned host allocation is NIC-registered and cudaHostRegister'ed
Portable|Mapped; callers' tensors stay in device memory. libibverbs is optional
and loaded at runtime; its development header is needed to build the provider.

Each call launches once on the current torch stream. All ranks issue identical
ops/sizes/dtypes in identical order, eager or replayed. Both ops share the
device sequence. Serialize calls on one
stream (or explicitly order streams); concurrent instances need distinct
prefixes. A skipped call leaves the peer spinning like any collective. A local
proxy error sets an abort word that ends local GPU spins; it cannot notify a
peer whose link failed. check() synchronizes and is for tests/step boundaries.

Provider selection is a two-rank agreement through the store (prefix/opened
and prefix/started): each rank publishes "ok" or its reason, and both use B1
only if both are ok. Otherwise (no libibverbs, a failed init, a peer failure)
both close their local resources and route calls to the op's fallback, with
active False and reason naming the failing rank. Construction raises RuntimeError
if neither fallback is provided; an inactive op without its fallback also raises.
A peer that never answers raises TimeoutError after
timeout_s on the waiting rank.

Graph capture uses a device sequence counter; prepare(sizes, dtypes) first.
Per-call transport choice reads only facts identical on both ranks by
contract: byte count, dtype, shapes (and, in prefer_* helpers, capture state).
Unsupported calls by those facts go to fallback(x, out), which must return its
result (for example wrap pynccl's in-place all_reduce and return the input),
or raise ValueError on both ranks. Local layout never switches transport:
misaligned, non-contiguous or partially overlapping operands are staged
through preallocated 16 B-aligned device buffers (graph-safe stream-ordered
copies); exact aliasing runs in place. A tensor off the provider device raises.
all_gather(x, out) copies bytes into rank-major output shaped (2, *x.shape),
allowing x to be exactly out[rank]. prepare_all_gather(sizes) compiles before
capture; gather_fallback(x, out) handles inactive or unsupported gathers.

The adaptive proxy occupies one big core while HOT, yields while WARM, and
uses timed futex sleeps while COLD. arm() at the engine's step boundary avoids
the cold polling delay; disarm() releases an indefinite arm. NCCL stays for
>=512 KiB all-reduces, with NCCL_MAX_NCHANNELS=2 and big-core affinity.

close() is local and idempotent: call after a barrier when no peer call is
outstanding. It synchronizes the device, cudaHostUnregisters, then stops/joins
the proxy, drains WRs and frees the region. Do not close during capture.
"""

from __future__ import annotations

import torch

from ._two_node import discovery, proxy

DISPATCH_MAX_BYTES = 256 * 1024
# Measured on the GB10 pair against NCCL 2.30.7 (1/2/4 channels, LL/Simple):
# eager B1 wins at 4-128 KiB per rank; NCCL in a CUDA graph costs ~0.5 ms per
# replay, so captured gathers use B1 through the largest measured size.
GATHER_EAGER_MIN_BYTES = 4 * 1024
GATHER_EAGER_MAX_BYTES = 128 * 1024
GATHER_GRAPH_MAX_BYTES = 512 * 1024
_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def prefer_two_node(nbytes, dtype) -> bool:
    return dtype in _DTYPES and 0 < nbytes <= DISPATCH_MAX_BYTES and nbytes % 16 == 0


def prefer_two_node_all_gather(nbytes, capturing=None) -> bool:
    """Per-rank bytes; capturing defaults to the current stream's capture state."""
    if capturing is None:
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    if nbytes <= 0 or nbytes % 16:
        return False
    if capturing:
        return nbytes <= GATHER_GRAPH_MAX_BYTES
    return GATHER_EAGER_MIN_BYTES <= nbytes <= GATHER_EAGER_MAX_BYTES


def _cuda_check(result):
    error = result[0] if isinstance(result, tuple) else result
    if int(error) != 0:
        raise RuntimeError(f"CUDA runtime error {error}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def _runtime():
    # PyTorch's cudart exposes host registration but not every device-attribute
    # or pointer-mapping entry point. Use cuda-python for those missing methods.
    from cuda.bindings import runtime

    return runtime


def support(device=None) -> str | None:
    """None when usable, otherwise an availability reason; never raises."""
    try:
        if not torch.cuda.is_available():
            return "two-node all-reduce requires CUDA"
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if device.type != "cuda":
            return "two-node all-reduce requires a CUDA device"
        index = torch.cuda.current_device() if device.index is None else device.index
        if torch.cuda.get_device_capability(index)[0] != 12:
            return "two-node all-reduce requires compute capability (12, x)"
        runtime = _runtime()
        for name in ("Integrated", "CanMapHostMemory", "HostNativeAtomicSupported"):
            attribute = getattr(runtime.cudaDeviceAttr, f"cudaDevAttr{name}")
            if not _cuda_check(runtime.cudaDeviceGetAttribute(attribute, index)):
                return f"CUDA device requires {name}"
        import cutlass  # noqa: F401

        reason = proxy.available()
        if reason:
            return reason
        if not discovery.roce_devices():
            return "no ACTIVE RoCE v2 IPv4-mapped GID discovered"
        return None
    except Exception as error:  # noqa: BLE001 - optional capability query
        return f"two-node all-reduce unavailable ({type(error).__name__})"


def _dispatch_reason(x, out, *, max_bytes, gather) -> str | None:
    """Transport choice from rank-symmetric facts only (bytes, dtype, shapes).

    Both ranks issue identical sizes/dtypes/shapes by contract, so both reach
    the same answer. Local layout (alignment, contiguity, overlap) never
    selects the transport; _launch stages it instead.
    """
    if not gather and x.dtype not in _DTYPES:
        return "x dtype must be bf16, fp16 or fp32"
    nbytes = x.numel() * x.element_size()
    if nbytes <= 0 or nbytes % 16:
        return "x nbytes must be a positive multiple of 16"
    if nbytes > max_bytes:
        return "x nbytes exceeds max_bytes"
    if out is not None:
        if gather and (tuple(out.shape) != (2, *x.shape) or out.dtype != x.dtype):
            return "out must have shape (2, *x.shape) and the same dtype as x"
        if not gather and (tuple(out.shape) != tuple(x.shape) or out.dtype != x.dtype):
            return "out must have the same shape and dtype as x"
    return None


def _device_error(x, out, device) -> str | None:
    """A tensor off the provider device is a caller bug, not a transport choice."""
    if x.device != device:
        return "x must be on the provider device"
    if out is not None and out.device != device:
        return "out must be on the provider device"
    return None


def _staging(x, out, *, nbytes, rank, gather) -> tuple[bool, bool]:
    """Local (stage_x, stage_out): which operands go through aligned buffers."""
    stage_out = out is not None and (not out.is_contiguous() or out.data_ptr() % 16 != 0)
    stage_x = not x.is_contiguous() or x.data_ptr() % 16 != 0
    if out is not None and not stage_out and not stage_x:
        a, b = x.data_ptr(), out.data_ptr()
        alias = b + rank * nbytes if gather else b
        span = 2 * nbytes if gather else nbytes
        # Exact aliasing is served in place; any other overlap reads a copy.
        stage_x = a != alias and a < b + span and b < a + nbytes
    return stage_x, stage_out


def grid_policy(nbytes, sms):
    """Small, sweepable policy: <=64 KiB per CTA, at most one CTA per SM."""
    return min(sms, max(1, (nbytes + 65535) // 65536)), 512


class TwoNodeAllReduce:
    """Two-rank all-reduce and byte all-gather sharing one device sequence.

    Both ranks must issue the same op sequence, including graph replays.
    """

    def __init__(
        self,
        *,
        rank,
        store,
        prefix="two-node-ar",
        max_bytes=256 * 1024,
        ibdev=None,
        gid_index=None,
        proxy_cpu=None,
        hot_us=2000,
        warm_us=10000,
        cold_poll_us=500,
        fallback=None,
        gather_fallback=None,
        device=None,
        timeout_s=60,
    ):
        if rank not in (0, 1):
            raise ValueError("rank must be 0 or 1 (world size is exactly 2)")
        if max_bytes <= 0 or max_bytes >= (1 << 24) or max_bytes % 16:
            raise ValueError("max_bytes must be a positive multiple of 16 below 2**24")
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct TwoNodeAllReduce outside CUDA graph capture")
        self.rank, self.max_bytes, self.fallback = rank, max_bytes, fallback
        self.gather_fallback = gather_fallback
        self.compiled = {}
        self._proxy = None
        self._registered = False
        self._closed = False
        self.active = False
        self.reason = ""
        try:
            # Both ranks agree twice (opened, then connected+started), so a local
            # failure or a peer failure puts both ranks on the same provider.
            reason = self._attempt(self._open, device, ibdev, gid_index)
            reason = proxy.agree(store, f"{prefix}/opened", rank, reason, timeout_s)
            if not reason:
                reason = self._attempt(
                    self._connect,
                    store,
                    prefix,
                    proxy_cpu,
                    (hot_us, warm_us, cold_poll_us),
                    timeout_s,
                )
                reason = proxy.agree(store, f"{prefix}/started", rank, reason, timeout_s)
        except BaseException:
            self.close()
            raise
        if reason:
            self._release()
            self.reason = reason
            if fallback is None and gather_fallback is None:
                raise RuntimeError(f"two-node all-reduce unavailable: {reason}")
            return
        self.active = True

    @staticmethod
    def _attempt(step, *args):
        try:
            step(*args)
            return ""
        except Exception as error:  # noqa: BLE001 - reported to both ranks
            return f"{type(error).__name__}: {error}"

    def _open(self, device, ibdev, gid_index):
        reason = support(device)
        if reason:
            raise RuntimeError(reason)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        candidates = [
            pair
            for pair in discovery.roce_devices()
            if (ibdev is None or pair[0] == ibdev) and (gid_index is None or pair[1] == gid_index)
        ]
        if not candidates:
            raise ValueError("no matching ACTIVE RoCE v2 IPv4-mapped GID")
        ibdev, gid_index = candidates[0]
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._proxy = proxy.create(ibdev, gid_index, self.max_bytes)
        self.host = int(self._proxy.host_address())
        self.layout = tuple(int(v) for v in self._proxy.layout())
        with torch.cuda.device(self.device):
            _cuda_check(
                torch.cuda.cudart().cudaHostRegister(self.host, int(self._proxy.region_bytes()), 3)
            )
            self._registered = True
            self.base = int(_cuda_check(_runtime().cudaHostGetDevicePointer(self.host, 0)))
            # Four Int64 words keep all accesses naturally aligned.
            self.words = torch.tensor([1, 0, 0, 0], dtype=torch.int64, device=self.device)
            # Input slot [0, max_bytes) and output slot [max_bytes, 3 * max_bytes).
            self._stage = torch.empty(3 * self.max_bytes, dtype=torch.uint8, device=self.device)
            torch.cuda.current_stream(self.device).synchronize()

    def _connect(self, store, prefix, proxy_cpu, backoff, timeout_s):
        proxy.bootstrap(self._proxy, store, prefix, self.rank, self.max_bytes, timeout_s)
        cpu = discovery.proxy_core() if proxy_cpu is None else proxy_cpu
        self._proxy.start(cpu, *backoff)

    def _kernel(self, dtype, ctas, threads, *, gather=False):
        key = gather, None if gather else dtype, ctas, threads
        if key not in self.compiled:
            if torch.cuda.is_current_stream_capturing():
                prepare = "prepare_all_gather()" if gather else "prepare()"
                raise RuntimeError(f"call {prepare} for this size/dtype before capture")
            from ._two_node.cute.kernels import compile_step

            with torch.cuda.device(self.device):
                self.compiled[key] = compile_step(dtype, ctas, threads, gather=gather)
        return self.compiled[key]

    def prepare(self, nbytes_list, dtypes=(torch.bfloat16,)):
        if self._closed:
            raise RuntimeError("provider is closed")
        if not self.active:
            return
        for nbytes in nbytes_list:
            if nbytes <= 0 or nbytes > self.max_bytes or nbytes % 16:
                raise ValueError("prepare sizes must be positive multiples of 16 <= max_bytes")
            for dtype in dtypes:
                if dtype not in _DTYPES:
                    raise ValueError("prepare dtype must be bf16, fp16 or fp32")
                self._kernel(dtype, *grid_policy(nbytes, self.sms))

    def prepare_all_gather(self, nbytes_list):
        if self._closed:
            raise RuntimeError("provider is closed")
        if not self.active:
            return
        for nbytes in nbytes_list:
            if nbytes <= 0 or nbytes > self.max_bytes or nbytes % 16:
                raise ValueError("prepare sizes must be positive multiples of 16 <= max_bytes")
            self._kernel(None, *grid_policy(nbytes, self.sms), gather=True)

    def __call__(self, x, out=None):
        if self._closed:
            raise RuntimeError("provider is closed")
        if not self.active:
            if self.fallback is None:
                raise RuntimeError(f"two-node all-reduce unavailable: {self.reason}")
            return self.fallback(x, out)
        reason = _dispatch_reason(x, out, max_bytes=self.max_bytes, gather=False)
        if reason:
            if self.fallback is not None:
                return self.fallback(x, out)
            raise ValueError(reason)
        return self._launch(x, out, gather=False)

    def all_gather(self, x, out=None):
        if self._closed:
            raise RuntimeError("provider is closed")
        if not self.active:
            if self.gather_fallback is None:
                raise RuntimeError(f"two-node all-gather unavailable: {self.reason}")
            return self.gather_fallback(x, out)
        reason = _dispatch_reason(x, out, max_bytes=self.max_bytes, gather=True)
        if reason:
            if self.gather_fallback is not None:
                return self.gather_fallback(x, out)
            raise ValueError(reason)
        return self._launch(x, out, gather=True)

    def _launch(self, x, out, *, gather):
        error = _device_error(x, out, self.device)
        if error:
            raise ValueError(error)
        nbytes = x.numel() * x.element_size()
        stage_x, stage_out = _staging(x, out, nbytes=nbytes, rank=self.rank, gather=gather)
        shape = (2, *x.shape) if gather else x.shape
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        # Preallocated 16 B-aligned buffers: graph-safe, stream-ordered copies.
        src = x
        if stage_x:
            src = self._stage[:nbytes].view(x.dtype).view(x.shape)
            src.copy_(x)
        dst = out
        if stage_out:
            span = 2 * nbytes if gather else nbytes
            base = self.max_bytes
            dst = self._stage[base : base + span].view(x.dtype).view(shape)
        fn = self._kernel(
            None if gather else x.dtype, *grid_policy(nbytes, self.sms), gather=gather
        )
        with torch.cuda.device(self.device):
            fn(
                src.data_ptr(),
                dst.data_ptr(),
                self.words,
                self.base,
                self.layout[-1],
                nbytes // 16,
                self.rank,
            )
        if stage_out:
            out.copy_(dst)
        return out

    def arm(self, hold_us=-1):
        if self.active:
            self._proxy.arm(hold_us)

    def disarm(self):
        if self.active:
            self._proxy.disarm()

    def stats(self):
        if not self.active:
            return {"error": "", "fallback": self.reason}
        return dict(self._proxy.stats().items())

    def check(self):
        if not self.active:
            return
        # Calls may have run on a non-default stream. item() alone only orders
        # its copy on the current stream, so first wait for this whole device.
        torch.cuda.synchronize(self.device)
        error = self.stats()["error"]
        if error:
            raise RuntimeError(error)
        if int(self.words[3].item()):
            raise RuntimeError("two-node GPU step aborted")

    def close(self):
        """Call after a barrier when no peer call is outstanding; unregister before free."""
        if self._closed:
            return
        self._release()
        self._closed = True

    def _release(self):
        """Free local B1 resources; idempotent, leaves fallback routing usable."""
        if self._registered:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                _cuda_check(torch.cuda.cudart().cudaHostUnregister(self.host))
            self._registered = False
        if self._proxy is not None:
            self._proxy.shutdown()
            self._proxy = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
