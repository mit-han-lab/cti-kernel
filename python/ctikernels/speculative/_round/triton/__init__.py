# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Triton fixed_query_addresses provider. Triton is imported on first use only."""

import sys

import torch

from ctikernels._runtime.launch import prime

# One config for every shape: BLOCK columns per work item, SCAN rows per prefix step.
BLOCK, SCAN, WARPS = 1024, 256, 4
_INT32_MAX = 2**31 - 1


def compile_count():
    """Binaries in this process's Triton JIT cache for the addresses kernel."""
    kernels = sys.modules.get(__name__ + ".kernels")
    if kernels is None:
        return 0
    return sum(len(cache[0]) for cache in kernels.fixed_query_addresses.device_caches.values())


def describe(q, pdl):
    return f"RoundVariant('fixed_query_addresses') at block_size={q}, use_pdl={pdl}"


class Addresses:
    def __init__(self, device):
        if torch.cuda.get_device_capability(device) < (8, 0):
            raise RuntimeError("fixed_query_addresses requires a CUDA device with cc >= 8.0")
        self.device = device
        # Cap on persistent programs: a host static of the device, never of B or M.
        self.programs = 4 * torch.cuda.get_device_properties(device).multi_processor_count
        # Private test-only mutation (no PDL wait); excluded from warm_up.
        self._wait = True
        # (Q, PDL) declared through warm_up; capture admits only these.
        self._declared = set()

    @staticmethod
    def _options(q, pdl, wait):
        return {
            "Q": q,
            "BLOCK": BLOCK,
            "SCAN": SCAN,
            "WAIT": pdl and wait,
            "num_warps": WARPS,
            "launch_pdl": pdl,
        }

    def warm_up(self, q, pdl):
        from .kernels import fixed_query_addresses as kernel

        with torch.cuda.device(self.device):
            # Pointers are dtypes (no allocation). Integers are typed by value
            # only: warm the int32 and the int64 table stride.
            for stride in (1, _INT32_MAX + 1):
                compiled = kernel.warmup(
                    torch.int32,
                    torch.int64,
                    torch.bool,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    1,
                    1,
                    0,
                    1,
                    stride,
                    1,
                    grid=(1,),
                    **self._options(q, pdl, True),
                )
                # Load the module now: CUDA's lazy load at a first launch drains the device.
                prime(compiled, compiled._init_handles)
        self._declared.add((q, pdl))

    def __call__(self, t):
        from .kernels import fixed_query_addresses as kernel

        q, m, pdl = t["block_size"], t["max_seq_len"], t["use_pdl"]
        batch = t["request_indices"].numel()
        with torch.cuda.device(self.device):
            if (
                self._wait
                and (q, pdl) not in self._declared
                and torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    f"speculative: {describe(q, pdl)} was not declared "
                    "to speculative.warm_up before capture"
                )
            tiles = -(-(m + q) // BLOCK)
            cache, indices = t["cache_index_table"], t["kv_indices"]
            # An empty view may point past its allocation, which Triton's
            # pointer check rejects. With capacity 0 nothing is written to it.
            out = indices if indices.numel() else t["cache_write_indices"]
            kernel[(min(batch * tiles, self.programs),)](
                t["request_indices"],
                t["start_frontiers"],
                t["round_done"],
                cache,
                t["cache_write_indices"],
                t["kv_indptr"],
                out,
                batch,
                m,
                t["guard_start"],
                # CSR offsets never exceed int32: a larger capacity is equivalent.
                min(indices.numel(), _INT32_MAX),
                cache.stride(0),
                tiles,
                **self._options(q, pdl, self._wait),
            )
