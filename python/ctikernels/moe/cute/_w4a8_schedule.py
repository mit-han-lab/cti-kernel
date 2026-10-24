# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team
# See THIRD_PARTY_NOTICES; upstream 0b46115d238d29ea2e92c6b9b2cb077d40fbc352.
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from FlashInfer 0.7.1 (fused_moe/cute_dsl/blackwell_sm12x/
# _moe_utils/moe_scheduler.py).
# Local modifications: any local expert count (32-expert lane windows), producer-owned work
# records, and dependency wait before any device route metadata is read.

"""Static persistent distribution of a device-dependent grouped tile list."""

from functools import lru_cache

ORDERS = ("expert_mtile_ntile", "expert_ntile_mtile")


def schedule_model(offsets, rows, channels, grid, order="expert_mtile_ntile", *, bn=128):
    """CPU ownership model; entries are (expert, routed row start, N tile, end)."""
    if (
        grid <= 0
        or channels <= 0
        or bn not in (64, 128)
        or channels % bn
        or rows not in (8, 32, 64, 128)
    ):
        raise ValueError("invalid schedule geometry")
    if order not in ORDERS:
        raise ValueError("unsupported schedule order")
    if offsets[0] != 0 or any(a > b for a, b in zip(offsets, offsets[1:], strict=False)):
        raise ValueError("offsets must be a nondecreasing exclusive prefix")
    tiles = [
        (e, row, n, end)
        for e, (start, end) in enumerate(zip(offsets, offsets[1:], strict=False))
        for row in range(start, end, rows)
        for n in range(channels // bn)
    ]
    if order == "expert_ntile_mtile":
        tiles.sort(key=lambda tile: (tile[0], tile[2], tile[1]))
    return [tiles[cta::grid] for cta in range(grid)]


@lru_cache(maxsize=1)
def device_schedule():
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32

    @cute.jit
    def publish(
        offsets,
        scale_offsets,
        work,
        record,
        linear,
        lane,
        rows: cutlass.Constexpr,
        ntiles: cutlass.Constexpr,
        order: cutlass.Constexpr,
        experts: cutlass.Constexpr,
    ):
        # The producer owns this record after waiting on work_empty. Its warp
        # gathers metadata, then publishes it with a release mbarrier arrival.
        if lane == 0:
            work[record, 5] = Int32(0)
        cute.arch.sync_warp()
        target = linear // ntiles
        if cutlass.const_expr(order == "expert_ntile_mtile"):
            target = linear
        prefix = Int32(0)
        # One lane per local expert, in 32-expert windows; lanes past E own empty experts.
        for window in cutlass.range_constexpr((experts + 31) // 32):
            expert = window * 32 + lane
            start = offsets[experts]
            end = start
            if expert < experts:
                start = offsets[expert]
                end = offsets[expert + 1]
            mtiles = (end - start + rows - 1) // rows
            count = mtiles
            if cutlass.const_expr(order == "expert_ntile_mtile"):
                count *= ntiles
            inclusive = count
            for bit in cutlass.range_constexpr(5):
                distance = 1 << bit
                previous = cute.arch.shuffle_sync_up(inclusive, distance, mask_and_clamp=0)
                if lane >= distance:
                    inclusive += previous
            first = prefix + inclusive - count
            if (target >= first) & (target < prefix + inclusive):
                local = (target - first) * rows
                ntile = linear % ntiles
                if cutlass.const_expr(order == "expert_ntile_mtile"):
                    # Only the owning lane decodes; empty experts never divide
                    # by zero. Adjacent CTAs reuse this expert's weight box.
                    local = ((target - first) % mtiles) * rows
                    ntile = (target - first) // mtiles
                work[record, 0] = expert
                work[record, 1] = start + local
                work[record, 2] = scale_offsets[expert] + local
                work[record, 3] = ntile
                work[record, 4] = end
                work[record, 5] = Int32(1)
            prefix += cute.arch.shuffle_sync(inclusive, 31)
        cute.arch.sync_warp()

    return publish
