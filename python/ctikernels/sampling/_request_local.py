# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ctikernels.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module


@cache
def _sampling_module() -> Module:
    import flashinfer

    return load_jit(
        "request_local_sampling",
        flashinfer.__version__,
        cuda_files=["request_local_sampling.cu"],
        cuda_wrappers=[("launch", "launch_request_sampling")],
        extra_include_paths=[str(Path(flashinfer.__file__).parent / "data/include")],
    )


# Fixed-point probability mass: one unit is 2^-62, as K1's top-p (``reference.fixed_mass``).
_MASS_UNIT = 2.0**62


def _sorted_masses(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Descending order, smaller ID first among ties, and the exact floor(p * 2^62)."""
    values, order = torch.sort(probabilities, dim=-1, descending=True, stable=True)
    return order, (values.double() * _MASS_UNIT).long()


def _renormalize(probabilities, order, masses, kept_sorted):
    """Zero all but the sorted prefix ``kept_sorted``; divide it by its retained mass.

    The integer sum is exact and order-free. It is rounded to FP32 once and scaled
    by 2^-62 exactly; each kept p is then divided with FP32 div.rn.
    """
    scale = torch.where(kept_sorted, masses, 0).sum(-1, keepdim=True).float() / _MASS_UNIT
    kept = torch.empty_like(kept_sorted).scatter_(1, order, kept_sorted)
    return torch.where(kept, probabilities / scale, 0)


def filter_requests(
    probabilities: torch.Tensor, top_k: torch.Tensor | None, top_p: torch.Tensor | None
) -> torch.Tensor:
    """Top-k, then top-p, of FP32 [B, V] probability rows; inputs are not modified.

    K1's filter law on a probability input. Top-k keeps exactly k tokens, ties to the
    smaller ID. Top-p keeps the descending prefix (smaller ID first) whose preceding
    fixed-point mass is below ceil(p * 2^62). Each filter divides what it keeps by the
    exact retained mass; k >= V and p == 1 are identities that keep every bit.
    Admission: nonnegative normalized rows, k >= 1 and p in (0, 1]. Each step is a
    stable sort, an integer sum or elementwise FP32 arithmetic, so results are
    bitwise run-to-run identical, and CPU and CUDA agree.
    """
    vocab = probabilities.shape[1]
    if top_k is not None:
        order, masses = _sorted_masses(probabilities)
        ranks = torch.arange(vocab, device=probabilities.device)
        kept = _renormalize(probabilities, order, masses, ranks < top_k[:, None])
        probabilities = torch.where(top_k[:, None] >= vocab, probabilities, kept)
    if top_p is not None:
        order, masses = _sorted_masses(probabilities)
        threshold = torch.ceil(top_p.double() * _MASS_UNIT).long()[:, None]
        kept = _renormalize(probabilities, order, masses, masses.cumsum(-1) - masses < threshold)
        probabilities = torch.where(top_p[:, None] == 1, probabilities, kept)
    return probabilities


def sample_requests(
    probabilities: torch.Tensor, seeds: tuple[int, ...], positions: tuple[int, ...]
) -> torch.Tensor:
    module = _sampling_module()
    # Each call owns its staging storage. PyTorch tracks asynchronous copy lifetime;
    # no persistent host buffer can be overwritten by the next overlapping batch.
    rng = torch.tensor((seeds, positions), dtype=torch.int64, pin_memory=True).to(
        probabilities.device, non_blocking=True
    )
    output = torch.empty(probabilities.shape[0], device=probabilities.device, dtype=torch.int32)
    module.launch(probabilities, rng, output)
    return output
