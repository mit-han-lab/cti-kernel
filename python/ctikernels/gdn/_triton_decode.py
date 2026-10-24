# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own gdn-decode Gluon tree d9ce6c3f (``solution.py``).

"""Decode on every CUDA device of compute capability 8.9 or newer: the fused Gluon kernel of
``_triton_decode_kernels``, one launch per call.

:func:`load` returns the implementation for every valid Geometry on such a device, else None
(the operation keeps fla-core). Configuration depends only on the batch size, the geometry and
the device's SM count, never on device identity or tensor values, so a call reads nothing on the
host. Qualified on device: RTX 5090 and RTX PRO 6000 (sm120), GB10 (sm121) and GB300 (sm103);
sm100 runs the same code, compiled ahead of time in CI, unverified on device.

**Configuration** (:func:`_config`, measured on the qualified cards; the tuning notes are in the
d9ce6c3f tree): per call ``heads = batch * Hv`` against the SM count picks the state rows per
CTA (32, 64, 128, or 96 for the three-CTA split of a head) and warps, and whether finisher CTAs
compute y. From ``WIDE_SMS`` SMs the wide rules apply (measured on the 188-SM RTX PRO 6000:
whole-head CTAs earlier, 64/4 flag halves in two bands, no finishers on 64-row tiles) with the
self-cleaning scratch; below it the default rules (measured at 170 and 48 SMs) with the epoch
scratch. ``batch_invariant`` fixes whole-head CTAs for every batch size, so each row's results
are bitwise the same in any batch.

**Scratch.** The exchange's words belong to the state pool: a pool's scratch is made on the
first call that needs it (int64, zeroed on that call's stream by our kernel ahead of the call),
kept while the pool lives and shared by the operations and streams that use the pool; the first
call on another stream waits for the maker stream once (asynchronously). Sizes per slot, R = Hv
/ Hk: flag words ``Hk (7 R + 3)`` (epochs) or ``Hk 16 R`` (self-cleaning), o words ``Hv 144``
(finisher calls, one set with warp sums and one without) with their counters ``Hk (4 R + 1)``
(epochs): at (16, 32) 2.1 KiB (4 KiB self-cleaning) of flags and up to twice 37 KiB of o words
per slot, at most 3.7% of the slot's 2 MiB of state. A pool first seen under
graph capture gets scratch private to that graph call, zeroed inside the graph before each
replay's call (the epoch words of the call's slots, or all of a self-cleaning scratch indexed by
batch row), so no word keeps what the allocator left.

**Scratch lifetime.** A pool's shared scratch lives as long as the pool's storage (not one
tensor object: views and detached aliases keep it), and every scratch a CUDA graph's kernels
address lives as long as that graph and the executables made from it: the capture retains a
CUDA user object in the graph whose destructor (``sem_post``, no Python) marks the graph gone;
the next scratch made, or the next pool released, drops what the gone graphs held.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from dataclasses import dataclass

import torch

from ._contract import CONV, Geometry, K, V, check_pools, check_projected, check_tensor


def check_decode(
    geometry,
    device,
    projected,
    conv_weight,
    A_log,
    dt_bias,
    norm_weight,
    slot_indices,
    state,
    conv_state,
) -> int:
    """#25's local contract; prefill/training and other providers keep their checks."""
    batch = check_projected(geometry, device, projected)
    check_tensor("conv_weight", conv_weight, torch.bfloat16, (geometry.conv_dim, CONV), device)
    for name, value, shape, dtypes in (
        ("A_log", A_log, (geometry.v_heads,), (torch.float32, torch.bfloat16, torch.float16)),
        ("dt_bias", dt_bias, (geometry.v_heads,), (torch.float32, torch.bfloat16, torch.float16)),
        ("norm_weight", norm_weight, (V,), (torch.float32, torch.bfloat16, torch.float16)),
        ("slot_indices", slot_indices, (batch,), (torch.int32, torch.int64)),
    ):
        if isinstance(value, torch.Tensor) and value.dtype not in dtypes:
            raise TypeError(f"{name} must have one of {dtypes}, not {value.dtype}")
        check_tensor(name, value, getattr(value, "dtype", dtypes[0]), shape, device)
    check_pools(geometry, device, state, conv_state)
    return batch


# The oldest compute capability served (8.9-12.1); programmatic dependent launch from (9, 0).
MIN_CAPABILITY = (8, 9)
PDL_CAPABILITY = (9, 0)
# SM counts from which the wide rules (and the self-cleaning scratch) apply; measured on the
# 188-SM RTX PRO 6000, against the default rules measured at 170 (RTX 5090) and 48 (GB10).
WIDE_SMS = 180
# SM counts up to which whole heads never pay: at 48 (GB10) 64/8 flags are 0.7-9.3% faster at
# 4-6 heads per SM (two hosts) and within 1% beyond.
SMALL_SMS = 64


def load(
    geometry: Geometry, device: torch.device, *, batch_invariant: bool = False
) -> TritonDecode | None:
    """The fused decode on any CUDA device of compute capability 8.9 or newer, or None (or
    where Triton's Gluon is unavailable)."""
    if not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_capability(device) < MIN_CAPABILITY:
        return None
    try:
        from . import _triton_decode_kernels  # noqa: F401
    except ImportError:
        return None
    return TritonDecode(device, batch_invariant=batch_invariant)


def _config(batch: int, hv: int, hk: int, sms: int) -> tuple[int, int, bool]:
    """(state rows per CTA, warps, finishers); 96 is the three-CTA split (64 + 32 + 32 rows).

    The rules depend only on the batch, the geometry and the SM count. Default rules (below
    WIDE_SMS SMs; measured at 170 and 48), heads = batch * Hv, R = Hv / Hk:
    - R = 3, three CTAs per head + one finisher per key head fit one per SM and four would not:
      the split;
    - 4 heads <= SMs: quarters (32/4) with finishers for R <= 4;
    - heads <= SMs: the split where 2 heads overflow the SMs by 4-13% (R = 2..4), else halves
      (64/4) with finishers for R = 3, 4 while 3 heads <= SMs, flags otherwise;
    - beyond: 64/4 with finishers while the grid fits three CTAs per SM (R <= 4; and for R = 3,
      Hk >= 8 while 2 heads <= 3 SMs), 64/4 flags while 2 heads <= 3 SMs or 5 SMs < 2 heads <=
      6 SMs, whole heads (128/8) from 2.15 heads per SM above SMALL_SMS SMs, else 64/8 flags.
    Wide rules (WIDE_SMS SMs and up; measured at 188): quarters with finishers for R <= 4 while 4 heads <= SMs; up to one head per SM whole
    heads once 2 heads > SMs + 4, else 64/4 flags; beyond, 64/4 flags for R = 2 while 2 heads
    <= 3 SMs or 5 SMs < 2 heads <= 6 SMs (P4, P5), else 64/8 flags."""
    heads = batch * hv
    r = hv // hk
    wide = sms >= WIDE_SMS
    if not wide and r == 3 and 3 * heads + batch * hk <= sms < 4 * heads + batch * hk:
        return 96, 4, True
    if 4 * heads <= sms:
        return 32, 4, r <= 4
    if heads <= sms:
        if wide and 2 * heads > sms + 4:
            return 128, 8, False
        if not wide and 2 <= r <= 4 and 104 * sms < 200 * heads <= 113 * sms:
            return 96, 4, True
        return 64, 4, not wide and 3 <= r <= 4 and 3 * heads <= sms
    if wide:
        if r == 2 and (2 * heads <= 3 * sms or 5 * sms < 2 * heads <= 6 * sms):
            return 64, 4, False
        return 64, 8, False
    if r <= 4 and 2 * heads + batch * hk <= 3 * sms:
        return 64, 4, True
    if r == 3 and hk >= 8 and 2 * heads <= 3 * sms:
        return 64, 4, True
    if r <= 4 and (2 * heads <= 3 * sms or 5 * sms < 2 * heads <= 6 * sms):
        return 64, 4, False
    if 20 * heads >= 43 * sms and sms > SMALL_SMS:
        return 128, 8, False
    return 64, 8, False


class _GraphHolds:
    """Scratch held for the CUDA graphs whose kernels address it, until each graph and every
    executable made from it is destroyed.

    The first hold under a capture creates a CUDA user object, its data a POSIX semaphore and
    its destructor ``sem_post``, and moves it into the graph being captured; :meth:`drain` (on
    the host thread, under the pools' lock) drops the scratch of every graph whose semaphore
    was posted. The destructor runs on a driver thread and touches no Python state."""

    def __init__(self) -> None:
        self._driver = None
        # capture id -> (semaphore, scratch held for that graph)
        self._graphs: dict[int, tuple[ctypes.Array, list[torch.Tensor]]] = {}

    def _bind(self):
        if self._driver is None:
            cuda, libc = ctypes.CDLL("libcuda.so.1"), ctypes.CDLL("libc.so.6", use_errno=True)
            p = ctypes.c_void_p
            info = cuda.cuStreamGetCaptureInfo_v2
            info.argtypes = [p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
            info.argtypes += [ctypes.POINTER(p), p, p]
            create = cuda.cuUserObjectCreate
            create.argtypes = [ctypes.POINTER(p), p, p, ctypes.c_uint, ctypes.c_uint]
            retain = cuda.cuGraphRetainUserObject
            retain.argtypes = [p, p, ctypes.c_uint, ctypes.c_uint]
            libc.sem_init.argtypes = [p, ctypes.c_int, ctypes.c_uint]
            libc.sem_trywait.argtypes = [p]
            libc.sem_destroy.argtypes = [p]
            post = ctypes.cast(libc.sem_post, p)
            self._driver = (info, create, retain, libc, post)
        return self._driver

    @staticmethod
    def _check(result: int, call: str) -> None:
        if result != 0:
            raise RuntimeError(f"{call} failed with CUresult {result}")

    def hold(self, scratch: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Keep ``scratch`` while the graph ``stream`` is capturing into lives."""
        info, create, retain, libc, post = self._bind()
        status, capture, graph = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p()
        self._check(
            info(stream.cuda_stream, status, capture, graph, None, None),
            "cuStreamGetCaptureInfo",
        )
        if status.value != 1:  # CU_STREAM_CAPTURE_STATUS_ACTIVE
            raise RuntimeError("the decode is captured on a stream whose capture is not active")
        held = self._graphs.get(capture.value)
        if held is None:
            # 64 bytes covers glibc's sem_t (32 on 64-bit Linux).
            semaphore = ctypes.create_string_buffer(64)
            if libc.sem_init(semaphore, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), "sem_init failed")
            user_object = ctypes.c_void_p()
            # CU_USER_OBJECT_NO_DESTRUCTOR_SYNC; the graph takes the only reference
            # (CU_GRAPH_USER_OBJECT_MOVE).
            self._check(
                create(ctypes.byref(user_object), ctypes.addressof(semaphore), post, 1, 1),
                "cuUserObjectCreate",
            )
            self._check(retain(graph, user_object, 1, 1), "cuGraphRetainUserObject")
            held = self._graphs[capture.value] = (semaphore, [])
        if not any(scratch is kept for kept in held[1]):
            held[1].append(scratch)

    def drain(self) -> None:
        """Drop the scratch of every destroyed graph."""
        if not self._graphs:
            return
        libc = self._bind()[3]
        for capture, (semaphore, _) in list(self._graphs.items()):
            if libc.sem_trywait(semaphore) == 0:
                libc.sem_destroy(semaphore)
                del self._graphs[capture]


class _Pools:
    """Scratch per (state pool, kind, slot dtype): shared scratch made on an eager call, kept while the
    pool's storage lives, and graph-private scratch made under capture; any scratch a capture
    uses is also kept while that graph lives."""

    def __init__(self) -> None:
        # Reentrant: a pool's finalizer may run (garbage collection) while the lock is held.
        self._lock = threading.RLock()
        self._shared: dict[tuple, tuple[torch.Tensor, torch.cuda.Stream, set]] = {}
        self._graphs = _GraphHolds()

    @staticmethod
    def _key(state: torch.Tensor, per_slot: int, kind: str) -> tuple:
        return (state.device, state.data_ptr(), tuple(state.shape), per_slot, kind)

    def _forget(self, key: tuple) -> None:
        with self._lock:
            self._shared.pop(key, None)
            self._graphs.drain()

    def reset(self, state: torch.Tensor | None = None) -> None:
        """Re-zero the shared scratch of ``state``'s pool (every pool when None) on the current
        stream, after the streams that used it: the state every scratch starts in (epoch words
        and their counters at tag 0, self-cleaning words 0). Eager only (the operation checks)."""
        with self._lock:
            for key, (scratch, _, streams) in list(self._shared.items()):
                if state is not None and key[:3] != (
                    state.device,
                    state.data_ptr(),
                    tuple(state.shape),
                ):
                    continue
                with torch.cuda.device(scratch.device):
                    stream = torch.cuda.current_stream(scratch.device)
                    for other in streams:
                        stream.wait_stream(other)
                    scratch.zero_()
                # Other streams now wait for this one before their next use.
                self._shared[key] = (scratch, stream, {stream})

    def _held(self, scratch: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            self._graphs.hold(scratch, stream)
        return scratch

    def shared(self, state, slot_indices, per_slot: int, kind: str) -> torch.Tensor:
        """The pool's zeroed scratch of ``per_slot`` words per slot; under capture of a pool no
        eager call has seen, a graph-private one whose words of the call's slots the graph zeroes
        before each replay's call."""
        from ._triton_decode_kernels import zero_scratch_kernel

        # Each slot pointer dtype must warm both zeroing variants before capture.
        key = (*self._key(state, per_slot, kind), slot_indices.dtype)
        capturing = torch.cuda.is_current_stream_capturing()
        stream = torch.cuda.current_stream(state.device)
        with self._lock:
            held = self._shared.get(key)
            if held is not None:
                scratch, maker, streams = held
                if not capturing and stream not in streams:
                    stream.wait_stream(maker)
                    streams.add(stream)
                if not capturing:
                    scratch.record_stream(stream)
                return self._held(scratch, stream)
            self._graphs.drain()
            words = state.shape[0] * per_slot
            # Kept with the pool: never an inference tensor of the call that made it, so
            # reset_scratch may zero it outside inference mode.
            with torch.inference_mode(False):
                scratch = torch.empty(words, dtype=torch.int64, device=state.device)
            per_slot_zero = (slot_indices.shape[0],), (scratch, slot_indices, per_slot)
            if capturing:
                grid, args = per_slot_zero
                zero_scratch_kernel[grid](*args, PER_SLOT=True, BLOCK=_pow2(per_slot))
                return self._held(scratch, stream)
            block = 1024
            zero_scratch_kernel[(-(-words // block),)](
                scratch, slot_indices, words, PER_SLOT=False, BLOCK=block
            )
            # Compiles the per-slot zeroing a later capture on a new pool launches (the slots'
            # words are zero already).
            grid, args = per_slot_zero
            zero_scratch_kernel[grid](*args, PER_SLOT=True, BLOCK=_pow2(per_slot))
            # The pool is its storage: views and detached aliases of the state keep it.
            weakref.finalize(state.untyped_storage(), self._forget, key)
            self._shared[key] = (scratch, stream, {stream})
            return scratch

    def self_cleaning(self, state, slot_indices, per_slot: int, kind: str):
        """The self-cleaning scratch and whether it is indexed by batch row: under capture of a
        pool no eager call has seen, a scratch private to the graph call with ``per_slot`` words
        per batch row, all zeroed in the graph before each replay's call; else the pool's
        shared scratch (zeroed once, left zeroed by every call)."""
        from ._triton_decode_kernels import zero_scratch_kernel

        key = (*self._key(state, per_slot, kind), slot_indices.dtype)
        with self._lock:
            known = key in self._shared
        if not torch.cuda.is_current_stream_capturing() or known:
            # (Making it compiles the whole-scratch zeroing a later capture launches.)
            return self.shared(state, slot_indices, per_slot, kind), False
        words = slot_indices.shape[0] * per_slot
        with self._lock:
            self._graphs.drain()
            scratch = torch.empty(words, dtype=torch.int64, device=state.device)
            self._held(scratch, torch.cuda.current_stream(state.device))
        block = 1024
        zero_scratch_kernel[(-(-words // block),)](
            scratch, slot_indices, words, PER_SLOT=False, BLOCK=block
        )
        return scratch, True


def _pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


_POOLS = _Pools()


def decode_plan(
    geometry: Geometry,
    batch: int,
    row_stride: int,
    packed: bool,
    *,
    sms: int,
    pdl: bool = True,
    batch_invariant: bool = False,
) -> "_Plan":
    """The launch of a batch size: grid, compile-time arguments and scratch. ``pdl`` launches
    with programmatic dependent launch (compute capability 9.0 and up); ``batch_invariant``
    fixes whole-head CTAs (128 rows, 8 warps) whatever the batch size, so each row's results do
    not depend on the batch it runs in."""
    from triton.experimental.gluon import language as gl

    from ._triton_decode_kernels import O_WORDS, counter_words, flag_words, sc_flag_words

    hk, hv = geometry.k_heads, geometry.v_heads
    r = hv // hk
    wide = sms >= WIDE_SMS
    bv, warps, fin = (128, 8, False) if batch_invariant else _config(batch, hv, hk, sms)
    # Finishers have at most four value-head lanes; never truncate R to RP.
    assert not fin or r <= 4, "decode finishers require R <= 4"
    three = bv == 96
    if three:
        bv = 32
    nv = 3 if three else V // bv
    n_main = batch * hv * nv
    if n_main + (batch * hk if fin else 0) > 2**31 - 1:
        raise ValueError("decode CTA count exceeds CUDA grid.x limit (2^31-1)")
    resident = batch * hv <= sms
    wsum = fin and bv == 32 and warps == 4 and 2 <= r <= 4
    kind = "o" if wsum else "on"
    if fin and wide:
        # The finisher, the o words' only reader, zeroes them; flag_ptr is unused.
        scratch = (("self_cleaning", hv * O_WORDS, kind),)
    elif fin:
        # The o set (with warp sums or without) and its counters, made together.
        scratch = (
            ("shared", hk * counter_words(r), kind + "ctr"),
            ("shared", hv * O_WORDS, kind),
        )
    elif wide:
        scratch = (("self_cleaning", hk * sc_flag_words(r), "sc"),)
    else:
        scratch = (("shared", hk * flag_words(r), "flags"),)
    rows = max(1, min(8, bv // warps))
    # Split poll: 64-row flag tiles in a single wave.
    spl = 64 if bv == 64 and not fin and resident else 0
    single_wave_flags = bv == 64 and warps == 4 and not fin and n_main <= sms and not wide
    options = {}
    # Batch-invariant codegen must not depend on the batch size either.
    if resident and not batch_invariant:
        # The whole grid is resident: let ptxas spend registers on issuing loads early.
        options["maxnreg"] = 255
    elif fin and bv == 64:
        # Three 128-thread CTAs per SM, so the finisher grid stays resident.
        options["maxnreg"] = 168
    arguments = dict(
        SP0=row_stride,
        HK=hk,
        HV=hv,
        BV=bv,
        NV=nv,
        G=1 << max(0, (r * nv - 1).bit_length()),
        PACKED=packed,
        TILE=gl.BlockedLayout([1, 4], [rows, 32 // rows], [warps, 1], [1, 0]),
        VEC=gl.BlockedLayout([max(1, K // (32 * warps))], [32], [warps], [0]),
        GRP=gl.BlockedLayout([1], [32], [warps], [0]),
        FIN=fin,
        # The finisher's [RP, 128] tile of o words: one warp per value head when R > warps
        # / 2, else all warps across each row.
        RP=warps if fin and 2 * r > warps else _pow2(r),
        FTILE=gl.BlockedLayout([1, 1], [1, 32], [warps, 1], [1, 0])
        if fin and 2 * r > warps
        else gl.BlockedLayout([1, max(1, K // (32 * warps))], [1, 32], [1, warps], [1, 0]),
        SPL=spl,
        WSUM=wsum,
        STILE=gl.BlockedLayout([1, 8], [2, 16], [warps, 1], [1, 0]),
        VSH=fin and bv == 32,
        A3=three,
        NCOL=single_wave_flags,
        BFLY=fin
        and not three
        and bv == 32
        and warps == 4
        and r == 2
        and n_main + batch * hk <= sms
        and not wide,
        OSQ=single_wave_flags,
        SC=wide,
        PWZ=wide and spl > 0 and r == 2 and hk & (hk - 1) == 0,
        OCC=batch if not wide and bv == 64 and warps == 4 and not fin and n_main > 2 * sms else 0,
        num_warps=warps,
        PDL=pdl,
        launch_pdl=pdl,
        **options,
    )
    return _Plan((n_main + (batch * hk if fin else 0),), n_main, scratch, arguments)


class TritonDecode:
    """``_contract.DecodeKernel`` for one device; :meth:`variant` names what its first call at a
    batch size compiles, so the capture guard can require it eagerly."""

    provider = "triton"

    def __init__(self, device: torch.device, *, batch_invariant: bool = False) -> None:
        self.batch_invariant = batch_invariant
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._pdl = torch.cuda.get_device_capability(device) >= PDL_CAPABILITY
        self._plans: dict[tuple, _Plan] = {}

    def variant(self, projected, *tensors: torch.Tensor) -> tuple:
        """What the compiled kernel depends on besides the batch size, given the call's tensors
        after ``projected``: the row stride of ``projected``, the conv taps' (the first tensor's)
        8-byte alignment and every pointer's dtype and 16-byte alignment."""
        return (
            projected.stride(0),
            tensors[0].data_ptr() % 8 == 0,
            tuple(tensor.dtype for tensor in (projected, *tensors)),
            *(tensor.data_ptr() % 16 == 0 for tensor in (projected, *tensors)),
        )

    def reset_scratch(self, state: torch.Tensor | None = None) -> None:
        """Re-zero the exchange scratch of ``state``'s pool (every pool when None); eager only."""
        _POOLS.reset(state)

    def _plan(self, geometry: Geometry, batch: int, row_stride: int, packed: bool) -> _Plan:
        """The launch of a batch size: grid, compile-time arguments and scratch."""
        return decode_plan(
            geometry,
            batch,
            row_stride,
            packed,
            sms=self._sms,
            pdl=self._pdl,
            batch_invariant=self.batch_invariant,
        )

    def __call__(
        self,
        geometry: Geometry,
        projected: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        conv_state: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from ._triton_decode_kernels import gdn_decode_kernel

        batch = projected.shape[0]
        packed = conv_weight.data_ptr() % 8 == 0
        dtypes = tuple(
            t.dtype
            for t in (
                projected,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
                slot_indices,
                state,
                conv_state,
            )
        )
        key = (batch, projected.stride(0), packed, dtypes)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = self._plan(geometry, batch, projected.stride(0), packed)
        rowidx = False
        words = []
        for how, per_slot, kind in plan.scratch:
            if how == "shared":
                words.append(_POOLS.shared(state, slot_indices, per_slot, kind))
            else:
                scratch, rowidx = _POOLS.self_cleaning(state, slot_indices, per_slot, kind)
                words.append(scratch)
        flags, owords = words[0], words[-1]
        taps = conv_weight.view(torch.int64) if packed else conv_weight
        y = out
        if y is None:
            y = torch.empty(
                batch, geometry.value_width, dtype=torch.bfloat16, device=projected.device
            )
        gdn_decode_kernel[plan.grid](
            projected,
            taps,
            A_log,
            dt_bias,
            norm_weight,
            slot_indices,
            state,
            conv_state,
            y,
            flags,
            owords,
            plan.n_main,
            ROWIDX=rowidx,
            **plan.arguments,
        )
        return y


@dataclass(frozen=True)
class _Plan:
    grid: tuple[int]
    n_main: int
    # (how, words per slot, kind) of each scratch: flags first, o words last.
    scratch: tuple[tuple[str, int, str], ...]
    arguments: dict


__all__ = ["MIN_CAPABILITY", "TritonDecode", "load"]
