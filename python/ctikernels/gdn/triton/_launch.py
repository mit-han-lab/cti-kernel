# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Launch Triton kernels through their compiled handles.

``Launcher()(fn, grid, *args, **constexprs)`` is ``fn[grid](*args, **constexprs)`` with Triton's
argument binder run once per specialization: each tensor's dtype and 16-byte alignment, each
integer's width and, unless ``fn`` does not specialize on it, its equality to 1 and
divisibility by 16, and the constexprs (num_warps and num_stages included). Later calls with
the same key go straight to the compiled kernel's launcher, about 10 us less host time per
launch. While ``capturing`` is set (a CUDA graph capture, as the operation's guard found), a
new key raises RuntimeError: it would compile inside the capture.
Pinned to Triton 3.7's CompiledKernel.run."""

from __future__ import annotations

import torch
from triton import knobs
from triton.runtime import driver


def _key(arg, specialized: bool):
    if isinstance(arg, torch.Tensor):
        return arg.dtype, arg.data_ptr() & 15 == 0
    if isinstance(arg, bool) or not isinstance(arg, int):
        return type(arg)
    wide = not -(2**31) <= arg < 2**31
    return (arg == 1, arg & 15 == 0, wide) if specialized else wide


class Launcher:
    """Launches kernels and owns their compiled handles, by specialization key."""

    def __init__(self) -> None:
        self._kernels: dict = {}
        self._specialized: dict = {}
        self.capturing = False

    def __len__(self) -> int:
        return len(self._kernels)

    def __call__(self, fn, grid: tuple[int, int, int], *args, **constexprs) -> None:
        specialized = self._specialized.get(fn)
        if specialized is None:
            skip = set(fn.do_not_specialize)
            names = [n for n in fn.arg_names if n not in constexprs]
            specialized = self._specialized[fn] = tuple(
                n not in skip and i not in skip for i, n in enumerate(names)
            )
        device = driver.active.get_current_device()  # handles are loaded per device
        key = (device, fn, tuple(constexprs.items()), tuple(map(_key, args, specialized)))
        entry = self._kernels.get(key)
        if entry is None:
            if self.capturing:
                raise RuntimeError(
                    f"{fn.__name__}: run this call once eagerly before capturing it (its "
                    "pointer alignments or integer classes never ran, so it would compile)"
                )
            kernel = fn[grid](*args, **constexprs)
            names = fn.arg_names
            slots = sorted((names.index(n), v) for n, v in constexprs.items() if n in names)
            self._kernels[key] = (kernel, slots)
            return
        kernel, slots = entry
        full = list(args)
        for index, value in slots:
            full.insert(index, value)
        stream = driver.active.get_current_stream(device)
        kernel.run(
            grid[0], grid[1], grid[2], stream, kernel.function, kernel.packed_metadata,
            kernel.launch_metadata(grid, stream, *full),
            knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *full,
        )  # fmt: skip


__all__ = ["Launcher"]
