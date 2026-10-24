# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Static pool bounds for CuTe's narrow inner-loop addressing."""


def wide_pool_offsets(slots, hv, window, width, checkpoint_stride=0, history_stride=0):
    """Promote the slot before either pool stride multiplication can overflow.

    K2 uses Int64 source/target slots unconditionally; Step uses this predicate.
    Inner head/tile/column offsets remain bounded by one slot's extent.
    """
    if checkpoint_stride == 0 and history_stride == 0:
        return slots * max(hv * 16384, (window + 3) * width) >= 2**31
    return slots * max(hv * 16384, (window + 3) * width, checkpoint_stride, history_stride) >= 2**31
