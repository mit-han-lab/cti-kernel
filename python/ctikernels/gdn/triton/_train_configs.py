# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Fixed launch configurations of the Triton training kernels, tuned offline on the RTX PRO
6000 (benchmark/kernel_tuning/gdn_training_triton_sweep.py); no runtime autotuning. Each entry
gives a kernel's block sizes and Triton launch options. One table serves every compute
capability the provider covers until a device class measures better with its own."""

from __future__ import annotations

CONFIGS: dict[str, dict[str, int]] = {
    "features": {"BT": 32, "num_warps": 4},
    "solve": {"BK": 128, "BW": 32, "num_warps": 2},
    "states": {"BV": 32, "num_warps": 4, "num_stages": 2},
    "gated_norm": {"BT": 16, "num_warps": 8},
    "state_backward": {"BV": 32, "num_warps": 4, "num_stages": 2},
    "dqkw": {"BK": 32, "BV": 64, "num_warps": 4, "num_stages": 2},
    "wy_backward": {"BD": 32, "num_warps": 4, "num_stages": 1},
    "conv_backward": {"BT": 32, "num_warps": 4},
}


def configs(capability: tuple[int, int]) -> dict[str, dict[str, int]]:
    """Every kernel's configuration on a device of ``capability`` (one table for all)."""
    return CONFIGS


__all__ = ["CONFIGS", "configs"]
