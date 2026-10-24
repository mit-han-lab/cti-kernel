# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Ported from cti's CuTe RTX PRO 6000 pair solution (accepted tree 5d6a2ce7); protocol unchanged.
"""All-reduce + residual add + RMSNorm (gain ``1 + w`` or ``w``) for tensor parallelism on
GPUs without peer access (RTX PRO 6000 over PCIe): one CuTe DSL kernel per call exchanges the
partial sums through pinned host memory shared by the ranks and normalizes in the same pass.

``HostExchange(...)`` is built once per rank; calling it with ``(x, residual, weight, eps)``
updates ``x`` and ``residual`` in place and returns them. See kernels.py for the protocol.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from ..._common import SharedHostMemory

KIB = 1024
# Rows per launch (sizes the mailboxes); larger calls run as several complete exchanges.
ROWS_CAP = 8192
# Batch sizes the engine captures CUDA graphs for: a batch between two of them runs padded
# to the next one, so it takes that size's schedule; larger batches run at their exact size.
GRAPH_BUCKETS = (*range(1, 9), *range(10, 33, 2), *range(40, 65, 4))
FLAG, PC = "flag", "pc"
# Bytes per direction and runs of the link calibration.
CALIB_BYTES = 16 << 20
# Six runs 20 ms apart avoid transient contention on a shared upstream link.
# Single samples measured 28.6 and 31.9 of 35.0 bytes/ns; three consecutive runs
# once all read low on an RTX PRO 6000 pair.
CALIB_RUNS = 6
CALIB_GAP_S = 0.02
# Implausible readings fall back to the slower shared-upstream-link rate.
CALIB_PLAUSIBLE = (5.0, 60.0)
CALIB_FALLBACK = 20.0
# Uploads can queue ahead of polls/downloads. Pace when uploads would delay them
# by about two link round trips: row r leaves at r * pace ns. Calibration uses
# 94 + 94 CTAs, below the 85 + 85-CTA both-ways rate used in earlier RTX 5090
# measurements: fastest 8 KiB pace 360 ns at 21.8 bytes/ns (factor 1.04).
# An RTX PRO 6000 pair with independent upstream links reads 35.07 against 42.3,
# putting that optimum near 1.25 in calibrated units. Factor 1.3 beat 1.04 by
# 5-8% on every paced bucket of both widths; 1.2-1.5 were within about 2%,
# while 1.8 and 2.5 were slower from 100 rows. Pacing paid from 50 KiB per rank
# in the earlier measurements, equivalent to 2.35 us of link time.
PACE_FACTOR = 1.3
PACE_FROM_NS = 2350
# Independent upstream links measured 35.0-35.2 bytes/ns; shared links about half.
OWN_LINKS_RATE = 25.0
# Five runs in three orders at fixed rate: pacing below the usual threshold
# improved 8 KiB bucket 10 by 4.6%, and 10 KiB buckets 5/7 by 6.4%/3.9%.
# Neighbouring buckets 5-8 / 6 and 8 were slower (short-exchange fence epochs).
OWN_PACED = frozenset({(10, False), (5, True), (7, True)})
# 8 warps improved 10 KiB bucket 10 by 2.4% and bucket 7 by 3.5% (six runs).
# 16 warps improved 8 KiB buckets 30/32 by 1.3%/1.5%.
OWN_WARPS = {(7, True): 8, (10, True): 8, (30, False): 16, (32, False): 16}
# Shared upstream links: factor 1.3 was faster through 64 rows, but 1.2-4.2%
# slower at 96-129 rows than the original factor 1.04 with 8 warps.
SHARED_FAST_PACE_ROWS = 64
SHARED_LONG_PACE_FACTOR = 1.04
# RTX 5090 producer/consumer sweeps: <=8 KiB rows use 16 producers, 4 warps
# below 3 MiB and 8 above. Wider rows use 12 producers of 4 warps through
# 8 MiB, then 16 of 8 warps. Two consumers per producer.
PC_SMALL_BYTES = 3 << 20
# One CTA per SM publishes R rows behind one fence/progress word/poll.
# R=2 beat producer/consumer by 19-29% at 189-376 rows; R=3-8 by 20-27%
# (8 KiB) / 5-18% (10 KiB) through 1504 rows; R=9-24 by 17.5-19% / 2.2-5%
# through 4512; R=25-44 by 12-18% / 1.4-2.7% through 8192 (one run; the
# reverse-order repeat was incomplete). This covers every slice on 188 SMs.
# Producer/consumer serves larger R on devices with fewer SMs.
FLAG_MAX_ROWS_PER_CTA = 44
PC_WIDE_BYTES = 8 << 20


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _fake_tensors(hidden):
    import cutlass.cute as cute
    from cutlass import BFloat16, Int64

    rows = cute.runtime.make_fake_tensor(
        BFloat16, (cute.sym_int(), hidden), stride=(cute.sym_int(), 1), assumed_align=16
    )
    rows2 = cute.runtime.make_fake_tensor(
        BFloat16, (cute.sym_int(), hidden), stride=(cute.sym_int(), 1), assumed_align=16
    )
    weight = cute.runtime.make_fake_tensor(BFloat16, (hidden,), stride=(1,), assumed_align=16)
    counters = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,), assumed_align=8)
    return [rows, rows2, weight, counters]


def schedule(tokens, hidden, *, slots, link_rate) -> tuple:
    """The schedule, from the workload's axes and the calibrated link rate only: (FLAG,
    warps, pace ns, rows per CTA) or (PC, warps, pace ns, producers, consumers per
    producer). A batch between CUDA-graph buckets takes its bucket's schedule (the engine
    runs it padded to that bucket); larger batches their own size's."""
    padded = next((bucket for bucket in GRAPH_BUCKETS if bucket >= tokens), tokens)
    row = hidden * 2
    nbytes = padded * row
    rows_per_cta = -(-padded // slots)
    own_links = link_rate >= OWN_LINKS_RATE
    bucket = (padded, row > 8 * KIB) if padded <= GRAPH_BUCKETS[-1] else None
    if rows_per_cta <= FLAG_MAX_ROWS_PER_CTA:
        factor = PACE_FACTOR
        shared_long = not own_links and rows_per_cta == 1 and padded > SHARED_FAST_PACE_ROWS
        if shared_long:
            factor = SHARED_LONG_PACE_FACTOR
        pace = 0
        if nbytes > link_rate * PACE_FROM_NS or (own_links and bucket in OWN_PACED):
            pace = round(row / (factor * link_rate))
        # 16 warps (one or two 16-byte vectors per thread: the peer row arrives in fewer
        # rounds of loads), except paced rows of up to 8 KiB, where 16 warps measured
        # within +-2% of 8; unpaced rows of both sizes measured 2.8-10.8%
        # faster, paced 10 KiB rows 1.2-3.5%.
        warps = 8 if pace and row <= 8 * KIB else 16
        if own_links and bucket in OWN_WARPS:
            warps = OWN_WARPS[bucket]
        if shared_long:
            warps = 8
        return (FLAG, warps, pace, rows_per_cta)
    if row > 8 * KIB:
        return (PC, 4, 0, 12, 2) if nbytes <= PC_WIDE_BYTES else (PC, 8, 0, 16, 2)
    return (PC, 4, 0, 16, 2) if nbytes < PC_SMALL_BYTES else (PC, 8, 0, 16, 2)


class _KernelCompiler:
    def __init__(self, hidden, slots, world, rank, rows_cap, link_rate, zero_centered=True):
        self.hidden, self.slots, self.world, self.rank = hidden, slots, world, rank
        self.zero_centered = zero_centered
        self.rows_cap, self.link_rate = rows_cap, link_rate
        self.row_stride = _round_up(hidden * 2, 128)
        self.par_stride = rows_cap * self.row_stride
        self.rank_stride = 2 * self.par_stride
        self.compiled = {}

    def prepare(self, token_counts, hidden=None):
        """Compile every launch slice without allocating exchange buffers or launching."""
        hidden = self.hidden if hidden is None else hidden
        if hidden <= 0 or hidden % 8 or hidden > self.hidden:
            raise ValueError("hidden must be a positive multiple of 8 within the configured width")
        for tokens in token_counts:
            if tokens < 0:
                raise ValueError("token counts must be nonnegative")
            for start in range(0, tokens, self.rows_cap):
                count = min(self.rows_cap, tokens - start)
                self._kernel(self.config(count, hidden), hidden)

    # ------------------------------------------------------------------ schedule
    def config(self, tokens, hidden):
        return schedule(tokens, hidden, slots=self.slots, link_rate=self.link_rate)

    # ------------------------------------------------------------------ kernels
    def _kernel(self, cfg, hidden):
        import cutlass.cute as cute
        from cutlass import Float32, Int32, Int64

        from . import kernels as _kernels

        kind, warps, pace = cfg[0], cfg[1], cfg[2]
        rows_per_cta = cfg[3] if kind == FLAG else 1
        key = (kind, hidden, warps, pace > 0, rows_per_cta, self.zero_centered)
        fn = self.compiled.get(key)
        if fn is not None:
            return fn
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        layout = (
            self.world,
            self.rank,
            self.slots,
            self.rank_stride,
            self.par_stride,
            self.row_stride,
            pace > 0,
        )
        options = "--enable-tvm-ffi"
        if kind == FLAG:
            op = _kernels.FlagKernel(
                hidden, 32 * warps, *layout, R=rows_per_cta, zero_centered=self.zero_centered
            )
            fn = cute.compile(
                op,
                *_fake_tensors(hidden),
                Int64(0),
                Int64(0),
                Int32(1),
                Int32(1),
                Float32(0.0),
                Int64(0),
                stream,
                options=options,
            )
        else:
            op = _kernels.PCKernel(hidden, 32 * warps, *layout, zero_centered=self.zero_centered)
            lprog = cute.runtime.make_fake_tensor(
                Int64, (cute.sym_int(),), stride=(1,), assumed_align=8
            )
            fn = cute.compile(
                op,
                *_fake_tensors(hidden),
                lprog,
                Int64(0),
                Int64(0),
                Int32(1),
                Int32(1),
                Int32(1),
                Float32(0.0),
                Int64(0),
                stream,
                options=options,
            )
        self.compiled[key] = fn
        return fn


def compile_kernels(
    hidden,
    token_counts,
    slots,
    world,
    rank,
    *,
    rows_cap=ROWS_CAP,
    link_rate=CALIB_FALLBACK,
    zero_centered=True,
):
    """Compile exchange variants on the current device, without memory registration or launches.

    The returned cache retains the compiled functions. Both this helper and
    :meth:`HostExchange.prepare` use the same compiler and slice schedules.
    """
    compiler = _KernelCompiler(hidden, slots, world, rank, rows_cap, link_rate, zero_centered)
    compiler.prepare(token_counts)
    return compiler.compiled


class HostExchange(_KernelCompiler):
    """One rank of a shared host exchange; construction is collective."""

    def __init__(
        self, *, rank, world_size, cpu_group, device, hidden, max_tokens, zero_centered=True
    ):
        from . import kernels as _kernels

        self.zero_centered = zero_centered
        self.rank = rank
        self.world = world_size
        self.device = torch.device(device)
        group = cpu_group
        props = torch.cuda.get_device_properties(self.device)
        # At most one CTA per SM, so every CTA of a call is resident at once (CTAs wait on
        # their peers' CTAs).
        self.slots = props.multi_processor_count
        self.hidden = max(8, hidden)
        self.row_stride = _round_up(self.hidden * 2, 128)
        self.rows_cap = max(self.slots, min(ROWS_CAP, max(1, max_tokens or ROWS_CAP)))
        self.rank_stride = 2 * self.rows_cap * self.row_stride
        self.par_stride = self.rows_cap * self.row_stride
        prog_bytes = self.world * self.slots * _kernels.PROG_BYTES
        flag_bytes = self.world * _kernels.PROG_BYTES
        nbytes = self.world * self.rank_stride + prog_bytes + flag_bytes + 3 * 4096
        with torch.cuda.device(self.device):
            self.host = SharedHostMemory(nbytes, self.rank, group, self.device)
            self.box = self.host.take(self.world * self.rank_stride)
            self.prog = self.host.take(prog_bytes)
            self.flags = self.host.take(flag_bytes)
            # The call counter (one slot per SM, every slot advances once per call) and the
            # producers' local progress words (producer/consumer schedule).
            self.counters = torch.zeros(self.slots, dtype=torch.int64, device=self.device)
            self.lprog = torch.zeros(self.slots, dtype=torch.int64, device=self.device)
        dist.barrier(group=group)
        self.compiled = {}
        self.link_rate = self._calibrate(group)

    def _calibrate(self, group):
        """The host path's rate (bytes per ns, per direction) with every rank uploading and
        downloading at once, as the exchange runs: the best of CALIB_RUNS runs, the slowest
        rank's (so every rank paces alike). The pace of the uploads follows it: on the RTX PRO
        6000 pair, GPUs that share an upstream link move about half what pairs on their
        own links do."""
        import cutlass.cute as cute
        from cutlass import Int32, Int64

        from . import kernels as _kernels

        grid = 2 * (self.slots // 2)
        unit = 256 * 16
        chunk = min(CALIB_BYTES, self.rank_stride) // (grid // 2) // unit * unit
        op = _kernels.CalibKernel(256, self.world, self.rank)
        stamps = cute.runtime.make_fake_tensor(Int64, (8,), stride=(1,), assumed_align=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = cute.compile(
            op,
            stamps,
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )
        peer = (self.rank + 1) % self.world
        # This rank's own readings (bytes per ns, None for a run whose handshake failed),
        # kept for inspection; its best; and the slowest rank's best, which the pace uses.
        self.calib_samples = []
        with torch.cuda.device(self.device):
            # Slots: 0 earliest start, 1 latest end, 3 the handshake's go word (it keeps the
            # last run's tag: CTAs wait for go >= this run's tag), 4 a guard against dead-code
            # elimination, 5 this run's failure flag (the kernel only ever sets it).
            times = torch.zeros(8, dtype=torch.int64, device=self.device)
            for run in range(1, CALIB_RUNS + 1):
                times[0] = 2**62
                times[1] = 0
                times[5] = 0
                time.sleep(CALIB_GAP_S)
                dist.barrier(group=group)
                fn(
                    times,
                    self.box + self.rank * self.rank_stride,
                    self.box + peer * self.rank_stride,
                    self.flags,
                    chunk,
                    run,
                    grid,
                )
                start, end, _, _, _, failed = times[:6].tolist()
                self.calib_samples.append(
                    None if failed else (grid // 2) * chunk / max(1, end - start)
                )
            del times
        self.calib_local = max((v for v in self.calib_samples if v is not None), default=0.0)
        rate = torch.tensor([self.calib_local], dtype=torch.float64)
        dist.all_reduce(rate, op=dist.ReduceOp.MIN, group=group)
        self.calibrated = float(rate.item())
        if not CALIB_PLAUSIBLE[0] <= self.calibrated <= CALIB_PLAUSIBLE[1]:
            # A failed or implausible reading: pace for the slower, shared host path.
            return CALIB_FALLBACK
        return self.calibrated

    def _launch(self, x, residual, weight, eps, tokens, hidden):
        cfg = self.config(tokens, hidden)
        fn = self._kernel(cfg, hidden)
        pace = int(cfg[2])
        if cfg[0] == FLAG:
            fn(
                x,
                residual,
                weight,
                self.counters,
                self.box,
                self.prog,
                tokens,
                -(-tokens // cfg[3]),
                eps,
                pace,
            )
        else:
            producers, per = cfg[3], cfg[4]
            producers = max(1, min(producers, tokens, self.slots // 2))
            per = max(1, min(per, -(-tokens // producers), (self.slots - producers) // producers))
            grid = producers * (per + 1)
            fn(
                x,
                residual,
                weight,
                self.counters,
                self.lprog,
                self.box,
                self.prog,
                tokens,
                producers,
                grid,
                eps,
                pace,
            )

    def close(self):
        """Release what the instance owns locally (no process group needed):
        wait for outstanding calls, then unregister and unmap the host memory. Idempotent."""
        if self.host is None:
            return
        torch.cuda.synchronize(self.device)
        self.counters = self.lprog = None
        host, self.host = self.host, None
        host.close()

    def __call__(self, x, residual, weight, eps):
        if self.host is None:
            raise RuntimeError("HostExchange used after close()")
        tokens, hidden = x.shape
        if tokens == 0:
            return x, residual
        if hidden % 8 or hidden > self.hidden:
            raise ValueError("hidden must be a multiple of 8, at most the configured width")
        for name, t in (("x", x), ("residual", residual)):
            if t.stride(1) != 1 or t.stride(0) % 8 or t.data_ptr() % 16:
                raise ValueError(
                    f"{name} needs unit inner stride, a row stride divisible by 8 elements "
                    "and a 16-byte-aligned base (the exchange runs on the SM path only)"
                )
        if not weight.is_contiguous() or weight.data_ptr() % 16:
            raise ValueError("weight must be contiguous with a 16-byte-aligned base")
        eps = float(eps)
        for start in range(0, tokens, self.rows_cap):
            stop = min(tokens, start + self.rows_cap)
            self._launch(x[start:stop], residual[start:stop], weight, eps, stop - start, hidden)
        return x, residual
