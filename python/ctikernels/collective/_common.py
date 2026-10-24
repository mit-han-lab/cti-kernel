# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The exchange of one tensor- or expert-parallel group on one node, shared by inference and
training: every reduction across the group's ranks (all-reduce, all-reduce + residual add +
RMSNorm, and the reduce after a GEMM whose epilogue published its tiles) goes through it.

Contract:

* Built collectively once per group, after ``torch.cuda.set_device`` and outside graph capture
  and memory-saver regions; ``cpu_group`` is a gloo group of the same ranks. Capacity is
  ``max_tokens`` rows of ``max_hidden`` columns per call: the fused reduce raises above it,
  ``all_reduce`` and ``add_rms_norm`` slice larger calls into complete exchanges.
* All ranks issue the same calls with the same shapes in the same order on one stream, eager
  or replayed in CUDA graphs. Each call is two launches on the current stream (the publishing
  producer, then the reduce), no host synchronization, no NCCL. A zero-row call does nothing.
* Every rank gets bitwise-identical results: partials are summed in FP32 in rank order (after
  the residual, when given) and rounded once. With a norm, residual = h and x = h * rstd * g
  (g = 1 + w zero-centered, else w), rstd from the unrounded h.
* Transports (``transport``): ``peer`` pushes into the peers' device memory (NVLink, or PCIe
  peer-to-peer) when every pair of the group's GPUs has peer access; ``host`` pushes into
  pinned host mailboxes shared by the ranks; ``nccl`` (devices outside cc 10.0-12.1, or
  requested) is the communicator's all-reduce plus ``norm.inference.add_rms_norm_``;
  ``local`` is a group of one.
* ``abort()`` sets a word every rank's spinning kernels poll: they stop, skip their stores and
  set this rank's error word, which ``check()`` raises. ``timeout_s`` > 0 also bounds each spin.
* ``close()`` is local: synchronize, unmap, free. Call it on every rank after the last call any
  peer may issue.

Fused GEMM epilogues: a producer kernel takes ``epilogue`` (``None`` unless the transport
publishes), calls ``ctikernels.collective.triton.exchange.publish`` on each output tile and
stores its own partial as usual; the caller then calls ``reduce(partial, tile=...)``.
"""

from __future__ import annotations

import importlib.util
import mmap
import os
import secrets
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist

from ctikernels.norm.inference import add_rms_norm_

# cudaHostRegisterPortable | cudaHostRegisterMapped
_HOST_REGISTER_FLAGS = 3
_PAGE = 4096
# Control page: the abort word, then one error word per rank, a cache line apart.
_ERROR_STRIDE = 128
# Smallest producer tile the flags are sized for (rows x columns), besides one-row tiles.
_MIN_TILE = 16
# Calls of at most this many elements per rank use the self-validating word protocol
# (8-byte words of two values and the call tag).
LL_MAX_ELEMS = 65536
# Word-protocol calls of up to this many rows split each row over several reduce programs
# (shorter host polls), at least this many columns each, within the device's SMs.
SPLIT_MAX_ROWS = 16
SPLIT_MIN_COLUMNS = 512
SPLIT_MAX_PARTS = 16
_TRANSPORTS = ("auto", "peer", "host", "nccl", "local")


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _cudart():
    """cuda-python's runtime bindings (imported on first use: CPU installs lack them)."""
    from cuda.bindings import runtime

    return runtime


def _kernels():
    """The Triton exchange kernels (imported on first use: CPU installs lack Triton)."""
    from .triton import exchange

    return exchange


def _check(result):
    error = result[0] if isinstance(result, tuple) else result
    if int(error) != 0:
        raise RuntimeError(f"CUDA runtime error {error}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


class SharedHostMemory:
    """Pinned host memory mapped by every rank (a /dev/shm file unlinked as soon as all ranks
    have mapped it) and registered with CUDA so kernels address it through its device alias.
    Each rank owns its mapping and registration; ``close()`` releases both."""

    def __init__(self, nbytes, rank, group, device):
        cudart = _cudart()
        self.device = device
        nbytes = _round_up(nbytes, mmap.PAGESIZE)
        name = None
        if rank == 0:
            name = f"/dev/shm/cti-exchange-{os.getpid()}-{secrets.token_hex(8)}"
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, nbytes)
        names = [name]
        root = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast_object_list(names, src=root, group=group)
        if rank != 0:
            fd = os.open(names[0], os.O_RDWR)
        try:
            self.map = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        dist.barrier(group=group)
        if rank == 0:
            os.unlink(name)
        self.tensor = torch.frombuffer(self.map, dtype=torch.uint8)
        self.host = self.tensor.data_ptr()
        _check(cudart.cudaHostRegister(self.host, nbytes, _HOST_REGISTER_FLAGS))
        self.registered = True
        self.base = int(_check(cudart.cudaHostGetDevicePointer(self.host, 0)))
        self.nbytes = nbytes
        self.used = 0

    def take(self, nbytes):
        """Device address of the next ``nbytes``, 4 KiB aligned."""
        start = _round_up(self.used, _PAGE)
        self.used = start + nbytes
        assert self.used <= self.nbytes
        return self.base + start

    def close(self):
        """Unregister and unmap; idempotent. Waits for the device's outstanding work first."""
        cudart = _cudart()
        if self.tensor is None:
            return
        torch.cuda.synchronize(self.device)
        if self.registered:
            self.registered = False
            _check(cudart.cudaHostUnregister(self.host))
        self.tensor = None
        self.map.close()


class _PeerMemory:
    """This rank's device inbox (cudaMalloc, outside torch's allocator) and the peers' inboxes
    opened through CUDA IPC. ``open`` is collective and returns None on every rank unless every
    rank mapped every peer (no peer access, or a peer's GPU not visible here)."""

    @classmethod
    def open(cls, nbytes, rank, world, group):
        cudart = _cudart()
        self = cls()
        self.local = int(_check(cudart.cudaMalloc(nbytes)))
        _check(cudart.cudaMemset(self.local, 0, nbytes))
        _check(cudart.cudaDeviceSynchronize())
        handle = _check(cudart.cudaIpcGetMemHandle(self.local))
        handles = [None] * world
        dist.all_gather_object(handles, bytes(handle.reserved), group=group)
        self.opened, self.bases = [], []
        ok = True
        for r, raw in enumerate(handles):
            if r == rank:
                self.bases.append(self.local)
                continue
            peer = cudart.cudaIpcMemHandle_t()
            peer.reserved = raw
            error, address = cudart.cudaIpcOpenMemHandle(
                peer, cudart.cudaIpcMemLazyEnablePeerAccess
            )
            if int(error) != 0:
                ok = False
                break
            self.opened.append(int(address))
            self.bases.append(int(address))
        vote = torch.tensor([int(ok)], dtype=torch.int32)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        if not vote.item():
            self.close()
            return None
        return self

    def close(self):
        cudart = _cudart()
        for address in self.opened:
            _check(cudart.cudaIpcCloseMemHandle(address))
        self.opened = []
        if self.local:
            _check(cudart.cudaFree(self.local))
            self.local = 0


def _triton_capable(props) -> bool:
    """The validated interval of the Triton exchange: cc 10.0-12.1 with Triton installed."""
    if props is None or not (10, 0) <= (props.major, props.minor) <= (12, 1):
        return False
    return all(importlib.util.find_spec(name) for name in ("triton", "cuda.bindings"))


def _one_host(world, group) -> bool:
    """Whether every rank of the group runs on this host (all ranks agree)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot = f.read().strip()
    except OSError:
        boot = ""
    hosts = [None] * world
    dist.all_gather_object(hosts, (socket.gethostname(), boot), group=group)
    return all(h == hosts[0] for h in hosts)


@dataclass(frozen=True)
class Epilogue:
    """What a producer kernel passes to ``publish``: the descriptor tensor (int64, device) and
    the constexprs ``RANK`` and ``WORLD``."""

    desc: torch.Tensor
    rank: int
    world: int


class Exchange:
    """One rank of a group's exchange; construction is collective (see the module docstring)."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        cpu_group,
        device,
        max_tokens: int,
        max_hidden: int,
        communicator=None,
        dtype: torch.dtype = torch.bfloat16,
        transport: str = "auto",
        timeout_s: float = 0.0,
        nccl_above: int | str | None = "auto",
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be within a positive world_size")
        if max_tokens < 1 or max_hidden < 1:
            raise ValueError("max_tokens and max_hidden must be positive")
        if transport not in _TRANSPORTS:
            raise ValueError(f"unknown exchange transport: {transport!r}")
        if dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("the exchange carries bfloat16 or float16 partials")
        self.rank, self.world = rank, world_size
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_tokens, self.max_hidden, self.dtype = max_tokens, max_hidden, dtype
        props = None
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
        self.communicator = communicator
        self.host_bytes = self.device_bytes = 0
        self._host = self._peer = None
        self._closed = False
        one_host = transport in ("auto", "peer", "host") and world_size > 1
        one_host = not one_host or _one_host(world_size, cpu_group)
        if transport in ("peer", "host") and not one_host:
            raise ValueError(f"{transport} transport needs every rank on one host")
        if transport == "auto":
            if world_size == 1:
                transport = "local"
            elif not one_host or not _triton_capable(props):
                transport = "nccl"
            else:
                transport = "peer or host"
        if transport == "local" and world_size != 1:
            raise ValueError("local transport requires world_size == 1")
        if transport == "nccl" and communicator is None:
            raise ValueError("nccl transport requires a communicator")
        self.transport, self.nccl_above = transport, None
        if transport in ("peer", "host", "peer or host"):
            self._sms = props.multi_processor_count
            with torch.cuda.device(self.device):
                self._build(cpu_group, float(timeout_s), nccl_above)

    def _build(self, group, timeout_s, nccl_above):
        world, elsize = self.world, torch.empty((), dtype=self.dtype).element_size()
        cells = self.max_tokens * self.max_hidden
        slot_bytes = _round_up(max(cells * elsize, min(cells, LL_MAX_ELEMS) * 4), _PAGE)
        tiles = -(-self.max_tokens // _MIN_TILE) * -(-self.max_hidden // _MIN_TILE)
        flag_cap = _round_up(max(self.max_tokens, tiles), 1024)
        inbox_bytes = 2 * (world - 1) * slot_bytes
        receiver_bytes = inbox_bytes + _round_up((world - 1) * flag_cap * 4, _PAGE)
        control = _round_up(_ERROR_STRIDE * (world + 1), _PAGE)
        if self.transport != "host":
            # Peer memory where every rank maps every peer (NVLink, PCIe P2P), else mailboxes.
            self._peer = _PeerMemory.open(receiver_bytes, self.rank, world, group)
            if self._peer is None and self.transport == "peer":
                raise RuntimeError("peer transport: a rank could not map its peers' memory")
            self.transport = "host" if self._peer is None else "peer"
        if nccl_above == "auto":
            # RTX PRO 6000 pair, BF16 rows of 4096: host mailboxes beat NCCL through 16 rows
            # (64K elements per rank), NCCL wins from 64 rows.
            nccl_above = LL_MAX_ELEMS if self.transport == "host" else None
        self.nccl_above = nccl_above if self.communicator is not None else None
        host_receivers = world if self.transport == "host" else 0
        self._host = SharedHostMemory(
            control + host_receivers * receiver_bytes, self.rank, group, self.device
        )
        self._control = self._host.take(control)
        if self.transport == "host":
            bases = [self._host.take(receiver_bytes) for _ in range(world)]
            self.host_bytes = self._host.nbytes
        else:
            bases = self._peer.bases
            self.host_bytes, self.device_bytes = self._host.nbytes, receiver_bytes
        self._counters = torch.tensor([1, 0], dtype=torch.int32, device=self.device)
        # Split rows' partial sums of squares, each marked with its call tag once written.
        self._sums = torch.zeros(SPLIT_MAX_ROWS * SPLIT_MAX_PARTS, device=self.device)
        self._marks = torch.zeros_like(self._sums, dtype=torch.int32)
        self.flag_cap = flag_cap
        error = self._control + _ERROR_STRIDE * (1 + self.rank)
        fixed = [
            self._counters.data_ptr(),
            self._counters.data_ptr() + 4,
            self._control,
            error,
            slot_bytes,
            flag_cap,
            int(timeout_s * 1e9),
            2**62 if self.nccl_above is None else self.nccl_above,
            LL_MAX_ELEMS,
            self.max_tokens,
            self.max_hidden,
        ]
        flags = [base + inbox_bytes for base in bases]
        self._desc = torch.tensor(fixed + bases + flags, dtype=torch.int64, device=self.device)
        self.device_bytes += sum(
            t.nbytes for t in (self._desc, self._counters, self._sums, self._marks)
        )
        dist.barrier(group=group)

    # ------------------------------------------------------------------ calls
    @property
    def epilogue(self) -> Epilogue | None:
        """The producer's hook arguments, or None when this transport does not publish."""
        if self.transport not in ("peer", "host"):
            return None
        return Epilogue(self._desc, self.rank, self.world)

    def _parts(self, m, n) -> int:
        """Reduce programs per row: the same on every rank (same shapes, same devices)."""
        if m * n > LL_MAX_ELEMS or m > SPLIT_MAX_ROWS:
            return 1
        parts = 1
        while (
            parts < SPLIT_MAX_PARTS
            and n / (2 * parts) >= SPLIT_MIN_COLUMNS
            and m * 2 * parts <= self._sms
        ):
            parts *= 2
        return parts

    def _via_nccl(self, numel) -> bool:
        """Whether a call of ``numel`` elements per rank goes through the communicator (or is
        local); every rank decides alike. Producers skip publishing such calls."""
        if self.transport in ("nccl", "local"):
            return True
        return self.nccl_above is not None and numel > self.nccl_above

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Exchange used after close()")

    def _validate(self, *tensors):
        for t in tensors:
            if t is not None and t.device != self.device:
                raise ValueError("all tensors must be on the exchange's device")

    def reduce(self, partial, *, tile, out=None, residual=None, norm=None):
        """Finish a fused call: ``partial`` [M, N] is this rank's partial, stored by the producer
        that published it with tiles ``tile = (TILE_M, TILE_N)``. Writes the rank-ordered sum
        (plus ``residual``) to ``out`` (default ``partial``); with ``norm = (weight, eps,
        zero_centered)`` writes x to ``out`` and h to ``residual`` and returns both."""
        self._require_open()
        out = partial if out is None else out
        self._validate(partial, out, residual)
        if partial.ndim != 2 or out.shape != partial.shape:
            raise ValueError("partial and out must have the same shape [M, N]")
        if residual is not None and residual.shape != partial.shape:
            raise ValueError("residual must have the shape of partial")
        if norm is not None and residual is None:
            raise ValueError("a fused norm needs the residual")
        M, N = partial.shape
        if M == 0:
            return (out, residual) if norm is not None else out
        if self._via_nccl(M * N):
            return self._unfused(partial, out, residual, norm)
        tile_m, tile_n = tile
        try:
            if M > self.max_tokens:
                raise ValueError("call exceeds the exchange's max_tokens")
            if tile_m < 1 or tile_n < 1 or -(-M // tile_m) * -(-N // tile_n) > self.flag_cap:
                raise ValueError(f"tile {tile} has more tiles than the exchange's flags")
            self._check_rows(N, partial, out, residual)
        except ValueError:
            # A producer may have published already: every rank rejects alike (the checks see
            # only shapes, strides and dtypes) and moves to the next call tag, so no stale
            # word or flag of this call can validate later.
            with torch.cuda.device(self.device):
                _kernels().launch_skip(self._desc)
            raise
        weight, eps, zero_centered = norm if norm is not None else (None, 0.0, False)
        with torch.cuda.device(self.device):
            _kernels().launch_reduce(
                self._desc,
                partial,
                out,
                residual,
                weight,
                eps,
                tile=tile,
                norm=norm is not None,
                zero_centered=zero_centered,
                rank=self.rank,
                world=self.world,
                pdl=True,
                parts=self._parts(M, N),
                sums=self._sums,
                marks=self._marks,
            )
        return (out, residual) if norm is not None else out

    def _unfused(self, partial, out, residual, norm):
        if self.transport != "local":
            self.communicator.all_reduce(partial, "sum")
        if norm is not None:
            weight, eps, zero_centered = norm
            x, residual = add_rms_norm_(partial, residual, weight, eps, zero_centered=zero_centered)
            if out is not partial:
                out.copy_(x)
            return out, residual
        if residual is not None:
            torch.add(residual, partial, out=out)
        elif out is not partial:
            out.copy_(partial)
        return out

    def _check_rows(self, n, *tensors):
        """The published layout's rules, checked before anything is sent (``publish`` itself
        skips calls beyond the capacity, so a raising call leaves every rank in step)."""
        rows = [t for t in tensors if t is not None]
        if n > self.max_hidden:
            raise ValueError("call exceeds the exchange's max_hidden")
        if n % 8 or any(t.stride(1) != 1 or t.stride(0) % 8 for t in rows):
            raise ValueError(
                "the exchange needs N divisible by 8, unit inner strides and row strides "
                "divisible by 8"
            )
        if rows[0].dtype != self.dtype:
            raise ValueError("partial must have the exchange dtype")

    def _sliced(self, x, residual, norm):
        with torch.cuda.device(self.device):
            for start in range(0, x.shape[0], self.max_tokens):
                rows = slice(start, start + self.max_tokens)
                part = x[rows]
                _kernels().launch_publish_rows(self._desc, part, rank=self.rank, world=self.world)
                self.reduce(
                    part,
                    tile=(1, part.shape[1]),
                    residual=None if residual is None else residual[rows],
                    norm=norm,
                )

    def all_reduce(self, x):
        """Sum ``x`` [M, N] over the group in place, in rank order; returns ``x``."""
        self._require_open()
        self._validate(x)
        if x.ndim != 2:
            raise ValueError("x must be [M, N]")
        if x.shape[0] == 0:
            return x
        if self._via_nccl(x.numel()):
            return self._unfused(x, x, None, None)
        self._check_rows(x.shape[1], x)
        self._sliced(x, None, None)
        return x

    def add_rms_norm(self, x, residual, weight, eps, zero_centered=True):
        """All-reduce the partial ``x``, add ``residual`` and normalize, in place: returns
        ``(x, residual)`` with residual = h and x = RMSNorm(h)."""
        self._require_open()
        self._validate(x, residual, weight)
        if x.ndim != 2 or residual.shape != x.shape or weight.shape != (x.shape[1],):
            raise ValueError("x and residual must be [M, N] and weight [N]")
        norm = (weight, float(eps), bool(zero_centered))
        if x.shape[0] == 0:
            return x, residual
        if self._via_nccl(x.numel()):
            return self._unfused(x, x, residual, norm)
        self._check_rows(x.shape[1], x, residual)
        self._sliced(x, residual, norm)
        return x, residual

    def prepare(self, token_counts) -> None:
        """Validate the call sizes a capture will use. Kernels compile on the first eager call
        of each width, which the engine's warm-up issues before capture."""
        self._require_open()
        if any(count < 0 for count in token_counts):
            raise ValueError("token counts must be nonnegative")

    # ------------------------------------------------------------------ lifecycle
    def _word(self, offset) -> tuple[mmap.mmap, int]:
        return self._host.map, offset - self._host.base

    def abort(self) -> None:
        """Stop every rank's spinning kernels (they report through ``check``)."""
        if self._host is not None and self._host.tensor is not None:
            mapping, offset = self._word(self._control)
            mapping[offset : offset + 4] = (1).to_bytes(4, "little")

    def check(self) -> None:
        """Raise if a call on this rank stopped on an abort or timeout (synchronizes)."""
        if self._host is None or self._host.tensor is None:
            return
        torch.cuda.synchronize(self.device)
        mapping, offset = self._word(self._control + _ERROR_STRIDE * (1 + self.rank))
        if int.from_bytes(mapping[offset : offset + 4], "little"):
            raise RuntimeError("exchange call stopped: a rank aborted or the spin timed out")

    def close(self) -> None:
        """Synchronize and release local resources; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._host is not None:
            torch.cuda.synchronize(self.device)
            if self._peer is not None:
                with torch.cuda.device(self.device):
                    self._peer.close()
            self._host.close()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
