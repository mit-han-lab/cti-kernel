# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Debug-only waits. Tags and progress encoding are documented in CP2.md."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T


@cute.jit
def progress(words, tag, blk):
    if cute.arch.lane_idx() == 0:
        ptr = words + cute.arch.warp_idx()
        value = cutlass.Int32((blk + 1) * 256 + tag)
        llvm.inline_asm(
            None,
            [ptr.toint().ir_value(), value.ir_value()],
            "st.volatile.shared.u32 [$0], $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def read_progress(words, warp: cutlass.Constexpr):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [(words + warp).toint().ir_value()],
            "ld.volatile.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def wait(barrier, parity, words, tag, blk, stage):
    progress(words, tag, blk)
    # The explicit suspend hint bounds each individual try_wait as well. The
    # globaltimer (nanoseconds) budget is per wait, independent of GPU clocks.
    ready = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [barrier.toint().ir_value(), cutlass.Int32(parity).ir_value()],
            "{ .reg .pred done, expired; .reg .b64 start, now, elapsed; "
            "mov.u64 start, %globaltimer; "
            "GDN_POLL: mbarrier.try_wait.parity.shared::cta.b64 done, [$1], $2, 0x100; "
            "@done bra GDN_DONE; mov.u64 now, %globaltimer; "
            "sub.u64 elapsed, now, start; setp.ge.u64 expired, elapsed, 2000000000; "
            "@!expired bra GDN_POLL; "
            "GDN_DONE: selp.u32 $0, 1, 0, done; }",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    # All callers are full warps. A uniform failure decision guarantees lane
    # zero prints even if another lane crosses the time limit just before it.
    if cute.arch.vote_any_sync(ready == 0):
        if cute.arch.lane_idx() == 0:
            cute.printf(
                "GDN_WATCHDOG tag=%d blk=%d warp=%d lane=%d stage=%d parity=%d bx=%d "
                "progress=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n",
                tag,
                blk,
                cute.arch.warp_idx(),
                cute.arch.lane_idx(),
                stage,
                parity,
                cute.arch.block_idx()[0],
                read_progress(words, 0),
                read_progress(words, 1),
                read_progress(words, 2),
                read_progress(words, 3),
                read_progress(words, 4),
                read_progress(words, 5),
                read_progress(words, 6),
                read_progress(words, 7),
                read_progress(words, 8),
                read_progress(words, 9),
                read_progress(words, 10),
                read_progress(words, 11),
                end="",
            )
        # Do not let another lane trap before lane zero finishes printf.
        cute.arch.sync_warp()
        llvm.inline_asm(
            None,
            [],
            "trap;",
            "",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
