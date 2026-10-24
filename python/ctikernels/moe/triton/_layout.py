# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Port of cti's own frozen MoE-block dX tree p2c (2026-10-11); expert-parallel masks
# and the shared expert's own intermediate size added here.
"""Expert-sorted, block-padded slot layout (built by ``_slot_layout.build_layout``)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SlotLayout:
    sorted_slot: torch.Tensor  # [npad] int32, slot index per padded row, -1 for padding
    block_expert: torch.Tensor  # [npad // block_m] int32, expert per block, > E past the end
    inv: torch.Tensor  # [P] int32, padded row of each slot (-1 for remote/masked slots)
    block_m: int

    @property
    def num_blocks(self) -> int:
        return self.block_expert.numel()
