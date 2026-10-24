# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Gated DeltaNet (Qwen3.5 linear attention) operations between the frozen in_proj and
out_proj GEMMs: decode and chunked prefill over slot-indexed state pools
(:mod:`ctikernels.gdn.inference`), and the varlen training forward with its activation
backward (:mod:`ctikernels.gdn.training`). Base weights are frozen: a weight that requires
grad raises ValueError before any device work, and no weight gradient exists.

Importing this package loads no provider; an operation binds its provider when constructed.
"""

from .inference import GatedDeltaNetDecode, GatedDeltaNetPrefill
from .replay import GatedDeltaNetReplay, ReplayLayers, ReplayPending, ReplayState, replay_sizes
from .training import GatedDeltaNetTraining

__all__ = [
    "GatedDeltaNetDecode",
    "GatedDeltaNetPrefill",
    "GatedDeltaNetTraining",
    "GatedDeltaNetReplay",
    "ReplayLayers",
    "ReplayPending",
    "ReplayState",
    "replay_sizes",
]
