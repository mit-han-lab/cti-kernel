# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Opt-in host-visible GEMM breadcrumbs; no CUDA/DSL imports on the CPU path."""

from functools import lru_cache

FIELDS = (
    "cta",
    "tid",
    "site",
    "fc",
    "stage",
    "phase",
    "iteration",
    "expert",
    "ntile",
    "tx",
    "cycle",
    "timeout",
)
SITES = {
    1: "barrier_init",
    2: "schedule_clear",
    3: "schedule_publish",
    4: "tma_wait",
    5: "operands_consumed",
    6: "fc1_staged",
    7: "fc2_staged",
    8: "epilogue_consumed",
    9: "schedule_done",
    10: "operand_empty",
    11: "work_full",
    12: "work_empty",
}
SLOTS = 289  # Progress in slot zero; each thread owns a separate timeout slot.
BUDGET_NS = 2_000_000_000


def allocate_records(grid):
    import torch

    records = torch.empty((grid, SLOTS, len(FIELDS)), dtype=torch.int32, pin_memory=True)
    records.zero_()
    return records


def decode_records(records):
    """Read CPU pinned storage even after CUDA has entered its sticky error state."""
    for block in records.tolist():
        for values in block:
            if values[2]:
                record = dict(zip(FIELDS, values, strict=True))
                record["site_name"] = SITES.get(record["site"], "unknown")
                yield record


class WatchedLaunch:
    def __init__(self, compiled, records):
        self.compiled = compiled
        self.watchdog_buffer = records

    def __call__(self, *args):
        pointer = None if self.watchdog_buffer is None else self.watchdog_buffer.data_ptr()
        return self.compiled(*args, pointer)


@lru_cache(maxsize=1)
def device_watchdog():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32, Uint64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    @cute.jit
    def timer():
        return Uint64(
            llvm.inline_asm(
                T.i64(),
                [],
                "mov.u64 $0, %globaltimer;",
                "=l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def store(pointer, value):
        llvm.inline_asm(
            None,
            [Int64(pointer).ir_value(), Int32(value).ir_value()],
            "st.volatile.global.u32 [$0], $1;",
            "l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def stamp(
        pointer,
        enabled: cutlass.Constexpr,
        cta,
        tid,
        site: cutlass.Constexpr,
        fc: cutlass.Constexpr,
        stage,
        phase,
        iteration,
        expert,
        ntile,
        tx: cutlass.Constexpr,
        cycle,
        timeout: cutlass.Constexpr = False,
    ):
        if cutlass.const_expr(enabled):
            slot = Int32(0)
            if cutlass.const_expr(timeout):
                slot = tid + 1
            if (tid == 0) | timeout:
                address = pointer + Int64((cta * SLOTS + slot) * len(FIELDS) * 4)
                store(address + 8, 0)
                values = (
                    cta,
                    tid,
                    site,
                    fc,
                    stage,
                    phase,
                    iteration,
                    expert,
                    ntile,
                    tx,
                    cycle,
                    int(timeout),
                )
                for field in cutlass.range_constexpr(len(FIELDS)):
                    if cutlass.const_expr(field != 2):
                        store(address + field * 4, values[field])
                # Publish fields before the site marker, and drain host writes
                # before trap. Timeout slots cannot be overwritten by progress.
                llvm.inline_asm(
                    None,
                    [],
                    "membar.sys;",
                    "~{memory}",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
                store(address + 8, site)
                llvm.inline_asm(
                    None,
                    [],
                    "membar.sys;",
                    "~{memory}",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )

    @cute.jit
    def wait_stage(
        barrier,
        phase,
        pointer,
        enabled: cutlass.Constexpr,
        cta,
        tid,
        fc: cutlass.Constexpr,
        stage,
        iteration,
        expert,
        ntile,
        tx: cutlass.Constexpr,
        cycle,
        site: cutlass.Constexpr = 4,
    ):
        if cutlass.const_expr(enabled):
            stamp(
                pointer,
                enabled,
                cta,
                tid,
                site,
                fc,
                stage,
                phase,
                iteration,
                expert,
                ntile,
                tx,
                cycle,
            )
            start = timer()
            ready = Int32(0)
            elapsed = Uint64(0)
            while (ready == 0) & (elapsed < BUDGET_NS):
                ready = Int32(
                    llvm.inline_asm(
                        T.i32(),
                        [Uint32(barrier.toint()).ir_value(), Int32(phase).ir_value()],
                        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2, 1000; selp.u32 $0, 1, 0, p; }",
                        "=r,r,r,~{memory}",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                elapsed = timer() - start
            if ready == 0:
                stamp(
                    pointer,
                    enabled,
                    cta,
                    tid,
                    site,
                    fc,
                    stage,
                    phase,
                    iteration,
                    expert,
                    ntile,
                    tx,
                    cycle,
                    True,
                )
                llvm.inline_asm(
                    None,
                    [],
                    "trap;",
                    "",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
        else:
            cute.arch.mbarrier_wait(barrier, phase)

    @cute.jit
    def cta_barrier(
        pointer,
        enabled: cutlass.Constexpr,
        cta,
        tid,
        site: cutlass.Constexpr,
        fc: cutlass.Constexpr,
        stage,
        phase,
        iteration,
        expert,
        ntile,
        tx: cutlass.Constexpr,
        math_only: cutlass.Constexpr = False,
    ):
        stamp(pointer, enabled, cta, tid, site, fc, stage, phase, iteration, expert, ntile, tx, -1)
        if cutlass.const_expr(math_only):
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
        else:
            cute.arch.barrier()

    return stamp, wait_stage, cta_barrier
