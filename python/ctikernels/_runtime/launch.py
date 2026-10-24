# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Kernel launch helpers shared by ops: priming each kernel once at warm-up, and eager Triton
launches without ``JITFunction.run``'s per-call argument binding (:class:`FastLaunch`).

CUDA loads a kernel lazily at its first launch (CUDA_MODULE_LOADING=LAZY, the default; CuTe
DSL's TVM-FFI functions also load their library inside that first call). That load waits for
all queued device work, so a kernel a warm-up only compiled stalls the host for a whole queue
drain at its first serving call. Warm-ups call :func:`prime` with each kernel they compile or
select: it runs ``launch(*args, **kwargs)`` (one small call on scratch) the first time it
sees ``kernel`` in this process, eagerly, and never again. ``kernel`` is the compiled object, or a value key
naming a library route (such as a cuBLAS shape bucket) whose kernel is not visible.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Hashable

import torch

__all__ = ["Bound", "CallPlans", "FastLaunch", "WarmRegistry", "call_class", "eager_on", "prime"]

_lock = threading.Lock()
_primed: set[Hashable] = set()  # holds each kernel, so identity keys are never reused


def prime(kernel: Hashable, launch: Callable[..., object], /, *args, **kwargs) -> bool:
    """Run ``launch(*args, **kwargs)`` once per ``kernel`` per process; returns whether it
    ran now. Raises
    RuntimeError inside a CUDA graph capture (prime at warm-up, before capture)."""
    if kernel in _primed:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prime runs at warm-up, outside CUDA graph capture")
    with _lock:
        if kernel in _primed:
            return False
        launch(*args, **kwargs)
        _primed.add(kernel)
    return True


class WarmRegistry:
    """Warm-up classes whose process-wide state (compiles, module loads, autotune or tactic picks)
    one instance already exercised: a later instance of the same class may skip those launches,
    never its per-instance work (workspace reservations, scratch, caches)."""

    def __init__(self):
        self._seen: set = set()

    def __contains__(self, key) -> bool:
        return key in self._seen

    def add(self, key) -> None:
        self._seen.add(key)


_OPTIONS = ("num_warps", "num_stages", "num_ctas", "maxnreg", "launch_pdl")
_MAX_BOUND = 256  # call sites kept bound per kernel; the map restarts beyond


class FastLaunch:
    """``launch(grid, *args, **kwargs)`` is ``fn[grid](*args, **kwargs)`` for a tuple ``grid``,
    at a fraction of the host time.

    ``JITFunction.run`` binds and specializes every argument in Python on each call (~20 us per
    launch). A FastLaunch keys each call by Triton's own specialization of its arguments
    (``native_specialize_impl``, which ``run`` uses), the constexpr values, the launch options
    and the device. A key's first call goes through ``run`` (compiling as usual), keeps the
    compiled kernel and primes it; later calls launch that kernel directly, with the bits
    ``run`` gives. Registered Triton launch hooks route every call through ``run``.
    """

    def __init__(self, fn):
        from triton._C.libtriton import native_specialize_impl
        from triton.runtime.driver import driver

        self._specialize_impl, self._driver = native_specialize_impl, driver
        self._fn = fn
        parameters = list(inspect.signature(fn.fn).parameters.values())
        self._names = [p.name for p in parameters]
        self._defaults = {p.name: p.default for p in parameters if p.default is not p.empty}
        self._params = [
            (p.is_constexpr, not p.do_not_specialize, not p.do_not_specialize_on_alignment)
            for p in fn.params
        ]
        self._kernels = {}
        self._bound = {}  # (grid, non-tensor args with tensor dtypes, kwargs) -> Bound

    def _key(self, args, kwargs):
        options = tuple((name, kwargs.pop(name)) for name in _OPTIONS if name in kwargs)
        values = list(args)
        for name in self._names[len(args) :]:
            values.append(kwargs[name] if name in kwargs else self._defaults[name])
        device = torch.cuda.current_device()
        key = [device, options]
        append = key.append
        for value, (constexpr, specialize, align) in zip(values, self._params, strict=True):
            if constexpr:
                append(value)
            elif type(value) is torch.Tensor or isinstance(value, torch.Tensor):
                # Triton's pointer specialization: element type and 16-byte alignment.
                append((value.dtype, align and value.data_ptr() % 16 == 0))
            elif type(value) is int:
                # Triton's integer specialization: 32/64-bit, and (unless disabled) 1 / 16 | v.
                wide = not -(2**31) <= value < 2**31
                append((wide, value == 1, value % 16 == 0) if specialize else (wide,))
            else:
                backend = self._fn.device_caches[device][3]
                append(self._specialize_impl(backend, value, False, specialize, align))
        return tuple(key), values, options, device

    def _kernel(self, grid, args, kwargs, options, key):
        """The compiled kernel of ``key``, launching through ``run`` (compiling) if new."""
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = self._fn[grid](*args, **kwargs, **dict(options))
            self._kernels[key] = kernel
            if not torch.cuda.is_current_stream_capturing():
                prime(kernel, kernel._init_handles)
            return kernel, True
        return kernel, False

    def __call__(self, grid, *args, **kwargs):
        site = (
            grid,
            tuple(a.dtype if isinstance(a, torch.Tensor) else a for a in args),
            tuple(kwargs.items()),
        )
        bound = self._bound.get(site)
        if bound is not None:
            bound(*args)
            return
        if len(self._bound) >= _MAX_BOUND:
            self._bound.clear()
        self._bound[site] = self.bind(grid, *args, **kwargs)

    def bind(self, grid, *args, **kwargs) -> Bound:
        """Launch ``fn[grid](*args, **kwargs)`` and return a :class:`Bound` relaunching the same
        compiled kernel and grid for the same non-tensor arguments."""
        given = dict(kwargs)
        key, values, options, device = self._key(args, kwargs)
        kernel = None
        if _hooked():
            self._fn[grid](*args, **kwargs, **dict(options))
        else:
            kernel, launched = self._kernel(grid, args, kwargs, options, key)
            if not launched:
                _run(kernel, (*grid, 1, 1), self._driver, device, values)
        return Bound(self, kernel, grid, args, values[len(args) :], given, device)


def _run(kernel, grid, driver, device, values):
    kernel.run(
        grid[0], grid[1], grid[2], driver.active.get_current_stream(device), kernel.function,
        kernel.packed_metadata, None, None, None, *values,
    )  # fmt: skip


def _hooked():
    from triton import knobs

    return bool(knobs.runtime.launch_enter_hook.calls or knobs.runtime.launch_exit_hook.calls)


class Bound:
    """One kernel launch with fixed non-tensor arguments, grid and device (made by
    :meth:`FastLaunch.bind`): ``bound(*args)`` relaunches it with the same positional
    arguments, tensors replaced. A tensor whose 16-byte alignment differs from the binding's
    (another Triton specialization), another current device or registered launch hooks take
    the full specialization path instead. A steady relaunch is one call of the compiled
    kernel's launcher with the handles, grid and stream getter resolved at binding."""

    __slots__ = (
        "_launcher",
        "_kernel",
        "_given",
        "_kwargs",
        "_device",
        "_aligned",
        "_hooks",
        "_launch",
    )

    def __init__(self, launcher, kernel, grid, args, tail, kwargs, device):
        from triton import knobs

        self._launcher, self._kernel, self._given = launcher, kernel, grid
        self._kwargs, self._device = kwargs, device
        self._hooks = knobs.runtime
        self._aligned = tuple(
            (i, a.data_ptr() % 16 == 0) for i, a in enumerate(args) if isinstance(a, torch.Tensor)
        )
        self._launch = None
        if kernel is not None:
            run, function, metadata = kernel.run, kernel.function, kernel.packed_metadata
            stream = launcher._driver.active.get_current_stream
            x, y, z = (*grid, 1, 1)[:3]
            tail = tuple(tail)

            def launch(*args):
                run(x, y, z, stream(device), function, metadata, None, None, None, *args, *tail)

            self._launch = launch

    def __call__(self, *args):
        hooks = self._hooks
        steady = (
            self._launch is not None
            and not (hooks.launch_enter_hook.calls or hooks.launch_exit_hook.calls)
            and torch.cuda.current_device() == self._device
        )
        if steady:
            for i, aligned in self._aligned:
                if (args[i].data_ptr() % 16 == 0) != aligned:
                    steady = False
                    break
        if steady:
            self._launch(*args)
        else:
            self._launcher.bind(self._given, *args, **self._kwargs)


def eager_on(index: int) -> bool:
    """Whether work enqueued now runs eagerly (no stream capture) on CUDA device ``index``,
    the current device. ``is_initialized`` rather than ``is_available``: callers hold CUDA tensors,
    and ``is_available`` queries NVML and the environment on every call."""
    return (
        torch.cuda.is_initialized()
        and torch.cuda.current_device() == index
        and not torch.cuda.is_current_stream_capturing()
    )


def call_class(t):
    """The validation and launch class of an operand (None stays None): everything checks and
    kernel specializations read (shape, strides, dtype, device, 16-byte alignment)."""
    return None if t is None else (t.shape, t.stride(), t.dtype, t.get_device(), t.data_ptr() % 16)


class CallPlans:
    """Launches remembered per call class after a validated eager call, so that later calls of
    the class skip validation and dispatch (bounded: the map restarts beyond ``limit``)."""

    def __init__(self, limit: int = 64):
        self._plans: dict = {}
        self._limit = limit

    def get(self, key):
        return self._plans.get(key)

    def put(self, key, plan) -> None:
        if len(self._plans) >= self._limit:
            self._plans.clear()
        self._plans[key] = plan
