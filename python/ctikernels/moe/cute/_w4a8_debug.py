# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Explicit probe specializations; never consulted by tactic selection."""

from functools import lru_cache

from ._w4a8_watchdog import WatchedLaunch

DEBUG_MODES = ("load_only", "mma_only", "no_epilogue", "fragments_only")


def validate_debug(mode, stamps):
    if mode is not None and mode not in DEBUG_MODES:
        raise ValueError("unsupported W4A8 debug mode")
    if not isinstance(stamps, bool):
        raise TypeError("debug_stamps must be bool")


class DebugLaunch(WatchedLaunch):
    def __init__(self, compiled, records, sink):
        super().__init__(compiled, records)
        self.debug_sink = sink

    def __call__(self, *args):
        return super().__call__(*args, self.debug_sink)


@lru_cache(maxsize=1)
def device_debug():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T

    @cute.jit
    def timer():
        # Global timestamps and elapsed waits use nanoseconds, not SM clocks.
        return Int64(
            llvm.inline_asm(
                T.i64(),
                [],
                "mov.u64 $0, %globaltimer;",
                "=l,~{memory}",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @cute.jit
    def fold_fragment(fragment, value):
        # Consume every word. A real store below keeps the dependency alive
        # through ptxas as well as LLVM; empty inline asm only protects LLVM.
        words = cute.recast_tensor(fragment, Uint32)
        for i in cutlass.range_constexpr(cute.size(words)):
            value += words[i]
        return value

    @cute.jit
    def store_sink(pointer, value):
        llvm.inline_asm(
            None,
            [Int64(pointer.toint()).ir_value(), Uint32(value).ir_value()],
            "st.volatile.global.u32 [$0], $1;",
            "l,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    return timer, fold_fragment, store_sink
