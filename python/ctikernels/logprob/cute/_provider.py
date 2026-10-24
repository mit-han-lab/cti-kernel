# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Per-device dispatch and grow-only reduction workspace."""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache

import torch

from ctikernels._runtime.workspace import GraphWorkspace


@lru_cache(maxsize=1)
def unavailable_reason() -> str | None:
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
        if version != "4.8.0":
            return f"CuTe requires nvidia-cutlass-dsl 4.8.0; found {version}"
        from . import _kernel  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        return f"CuTe kernel import failed ({type(error).__name__})"
    return None


def available() -> bool:
    return unavailable_reason() is None


@lru_cache(maxsize=64)
def inverse_temperature(temperature: float) -> float:
    """FP32 1/T exactly as PyTorch CUDA forms it for ``logits / T`` with a Python scalar.

    BinaryDivTrueKernel takes the reciprocal of the double scalar in double, casts it
    to FP32, and multiplies by it. The cast runs on the CPU whatever torch's default
    device is, so a new temperature never syncs the GPU or breaks graph capture.
    """
    return torch.tensor(1.0 / temperature, dtype=torch.float32, device="cpu").item()


def choose_splits(rows: int, vocab: int, sm_count: int) -> int:
    """Two CTAs/SM for small batches; bound register storage for large rows."""
    if rows == 0:
        return 0
    # Bucket rows to powers of two: few compiled variants as the token count varies.
    rows = 1 << (rows - 1).bit_length()
    occupancy = (2 * sm_count + rows - 1) // rows
    minimum = (vocab + 7 + 16383) // 16384
    useful = max(1, (vocab + 1023) // 1024)
    return max(minimum, min(occupancy, useful))


def _workspace_counts(max_rows: int, vocab: int, sm_count: int) -> tuple[int, int]:
    """Partials and counters covering every row count up to ``max_rows``."""
    partials = max(r * choose_splits(r, vocab, sm_count) * 2 for r in range(1, max_rows + 1))
    return partials, max_rows


def compile_kernel(dtype, vocab, splits, per_row_temperature, capability):
    import cutlass
    import cutlass.cute as cute
    from cutlass.runtime import make_fake_stream

    from ._kernel import Kernel

    scalar = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    fake = cute.runtime.make_fake_tensor
    rows = cute.sym_int64()
    return cute.compile(
        Kernel(vocab, splits, scalar.width, per_row_temperature),
        fake(scalar, (rows, vocab), (cute.sym_int64(), 1), assumed_align=scalar.width // 8),
        fake(cutlass.Int64, (rows,), (cute.sym_int64(),), assumed_align=8),
        fake(cutlass.Float32, (rows,), (1,), assumed_align=4),
        fake(cutlass.Float32, (cute.sym_int64(),), (1,), assumed_align=4),
        fake(cutlass.Int32, (cute.sym_int64(),), (1,), assumed_align=4),
        fake(cutlass.Float32, (rows,), (cute.sym_int64(),), assumed_align=4),
        cutlass.Float32(1.0),
        cutlass.Int64(-100),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=(
            f"--enable-tvm-ffi --gpu-arch sm_{capability[0]}{capability[1]} "
            "--ptxas-options=--fmad=false"
        ),
    )


class Provider:
    name = "cute"

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Cache hardware properties once, when this device's provider is constructed.
        properties = torch.cuda.get_device_properties(device)
        self.sm_count = properties.multi_processor_count
        self.capability = (properties.major, properties.minor)
        self._compiled = {}
        self._workspace = None
        self._retired_workspaces = []

    def _allocate(self, counts):
        return (
            torch.empty(counts[0], dtype=torch.float32, device=self.device),
            torch.zeros(counts[1], dtype=torch.int32, device=self.device),
        )

    def __call__(self, logits, targets, output, temperature, ignore_index=-100):
        rows, vocab = logits.shape
        if rows == 0:
            return
        splits = choose_splits(rows, vocab, self.sm_count)
        per_row_temperature = isinstance(temperature, torch.Tensor)
        key = (logits.dtype, vocab, splits, per_row_temperature)
        with torch.cuda.device(self.device):
            compiled = self._compiled.get(key)
            capacity = rows * splits * 2
            group = GraphWorkspace.active()
            if group is not None:
                # A graph workspace sizes one shared slot for every row count up front.
                slot = ("logprob", self.device, torch.float32)
                group.reserve(slot, (vocab, self.sm_count), _workspace_counts, self._allocate)
                workspace = group.take(slot, rows, (capacity, rows))
                grow = False
            else:
                workspace = self._workspace
                grow = (
                    workspace is None
                    or workspace[0].numel() < capacity
                    or workspace[1].numel() < rows
                )
            if grow or compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm selected logprobs shapes before CUDA graph capture")
                if compiled is None:
                    compiled = compile_kernel(*key, self.capability)
                    self._compiled[key] = compiled
                if grow:
                    if workspace is not None:
                        self._retired_workspaces.append(workspace)
                    workspace = self._allocate((capacity, rows))
                    self._workspace = workspace
            compiled(
                logits,
                targets,
                output,
                *workspace,
                temperature if per_row_temperature else output,
                1.0 if per_row_temperature else inverse_temperature(temperature),
                ignore_index,
            )
