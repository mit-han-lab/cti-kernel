# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""MoE LoRA device facts, live scales and the compile lock (no CuTe import)."""

import importlib.metadata
import threading
from functools import lru_cache

import torch

LOCK = threading.RLock()
_SCALES = {}


@lru_cache(maxsize=1)
def available():
    try:
        return importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.0"
    except importlib.metadata.PackageNotFoundError:
        return False


@lru_cache(maxsize=None)
def device_info(device):
    prop = torch.cuda.get_device_properties(device)
    return (prop.major, prop.minor), prop.multi_processor_count


def scale_tensor(scaling, device):
    if isinstance(scaling, torch.Tensor):
        if scaling.dtype != torch.float32 or scaling.numel() != 1 or scaling.device != device:
            raise ValueError("scaling must be one FP32 element on the input device")
        if not scaling.is_contiguous():
            raise ValueError("scaling must be contiguous")
        return scaling.reshape(1)
    if type(scaling) is not float:
        raise ValueError("scaling must be a Python float or a one-element FP32 tensor")
    key = (device, scaling)
    with LOCK:
        value = _SCALES.get(key)
        if value is None:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up each float scaling value before CUDA graph capture")
            # Process-global: never an inference tensor of the first caller (training saves it).
            with torch.inference_mode(False):
                value = torch.tensor([scaling], dtype=torch.float32, device=device)
            _SCALES[key] = value
    return value


def grid_capacity(sorted_ids, pairs, block_m):
    """Trim the aligned id buffer that sizes the launch grids.

    Each non-empty expert pads at most ``block_m - 1`` rows and there are at most
    ``pairs`` non-empty experts, so the padded count never exceeds
    ``pairs * block_m``. Beyond that, CTAs only exit: with fewer pairs than
    experts (decode batches) the full buffer launches up to twice the needed grid.
    """
    return sorted_ids[: min(sorted_ids.numel(), pairs * block_m)]


_STREAMS = {}


def overlap_stream(device):
    """One LoRA side stream per device (both providers), initialized during eager warm-up."""
    with LOCK:
        stream = _STREAMS.get(device)
        if stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up MoE LoRA overlap before CUDA graph capture")
            stream = torch.cuda.Stream(device=device)
            _STREAMS[device] = stream
    return stream
