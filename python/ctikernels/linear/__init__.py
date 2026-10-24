# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Linear tensor-operation implementations."""

from ._autotune import autotune, autotune_entries
from ._lora_segmented import (
    LoRAChunkTable,
    last_config,
    lora_delta_factors_,
    lora_segment_metadata,
    lora_segmented_delta_,
    lora_segmented_projection,
)
from .inference import (
    MAX_M,
    Linear,
    gated_input,
    release_cublas_workspaces,
    release_scratch,
    scratch_bytes,
)

__all__ = [
    "MAX_M",
    "Linear",
    "LoRAChunkTable",
    "autotune",
    "autotune_entries",
    "gated_input",
    "last_config",
    "lora_delta_factors_",
    "lora_segment_metadata",
    "lora_segmented_delta_",
    "lora_segmented_projection",
    "release_cublas_workspaces",
    "release_scratch",
    "scratch_bytes",
]
