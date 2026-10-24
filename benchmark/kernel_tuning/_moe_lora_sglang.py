# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Execute SGLang's unmodified MoE LoRA Triton op (``fused_moe_lora_kernel.py``) for one adapter.

Loaded by path from an SGLang source tree (72730d6 = 0.5.22 for this file); only the tensor-parallel
collectives (unused, fail closed) and the PDL device checks (SGLang's own definitions) are shimmed.
The classic runner's call is reproduced: BLOCK_SIZE_M 64 alignment, rank-aware shrink N / expand K,
4 warps, 2 stages. SGLang builds that alignment with its JIT ``moe_lora_align_block_size``; for a
single adapter its layout equals ``moe_align_block_size`` at the same block size, which is used here.
Run in a dedicated process: the module is registered process-wide.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

KERNEL = "sglang/kernels/ops/lora/moe/fused_moe_lora_kernel.py"


def _unavailable(*args, **kwargs):
    raise NotImplementedError("tensor-parallel MoE LoRA is off the benchmark path")


def load(python_dir):
    """Return SGLang's ``_fused_moe_lora`` from ``python_dir`` (the tree holding ``sglang/``)."""
    import torch

    path = Path(python_dir) / KERNEL
    if not path.is_file():
        raise FileNotFoundError(path)
    distributed = ModuleType("sglang.srt.distributed")
    distributed.tensor_model_parallel_all_gather = _unavailable
    distributed.tensor_model_parallel_all_reduce = _unavailable
    common = ModuleType("sglang.srt.utils.common")
    cuda = torch.version.cuda or "0"
    common.is_sm90_supported = lambda device=None: (
        torch.cuda.get_device_capability(device)[0] == 9 and cuda >= "12.3"
    )
    common.is_blackwell_supported = lambda device=None: (
        torch.cuda.get_device_capability(device)[0] in (10, 12) and cuda >= "12.8"
    )

    def no_registry(**kwargs):
        # The module falls back to its plain functions on AttributeError.
        raise AttributeError("no torch custom-op registration in the benchmark")

    common.direct_register_custom_op = no_registry
    for name in ("sglang", "sglang.srt", "sglang.srt.utils"):
        sys.modules.setdefault(name, ModuleType(name))
    sys.modules[distributed.__name__] = distributed
    sys.modules[common.__name__] = common
    spec = importlib.util.spec_from_file_location("sglang_fused_moe_lora_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fused_moe_lora


def _block_config(rank):
    # SGLang lora_moe_runners._get_moe_lora_block_config at 72730d6.
    rank_pow2 = 1 << (rank - 1).bit_length()
    return min(64, rank_pow2), max(16, min(64, rank_pow2))


class SGLangMoELoRA:
    """One adapter's gate/up and down LoRA deltas, as SGLang's classic runner issues them."""

    def __init__(self, fused, factors, experts, top_k, device, scaling):
        import torch

        gate_a, gate_b, up_a, up_b, down_a, down_b = factors
        # SGLang stacks adapters [max_loras, E, ...] and folds the scaling into B at load.
        self.a13 = [gate_a[None], up_a[None]]
        self.b13 = [(gate_b * scaling)[None], (up_b * scaling)[None]]
        self.a2, self.b2 = [down_a[None]], [(down_b * scaling)[None]]
        self.rank = gate_a.shape[1]
        self.fused, self.experts, self.top_k = fused, experts, top_k
        self.lora_ids = torch.zeros(1, dtype=torch.int32, device=device)
        self.enabled = torch.ones(1, dtype=torch.int32, device=device)

    def alignment(self, ids):
        from ctikernels.moe_impl import moe_align_block_size

        sorted_ids, expert_ids, padded = moe_align_block_size(ids, 64, self.experts)
        return sorted_ids[None], expert_ids[None], padded

    def _call(self, out, x, a, b, weights, align, mul_routed_weight):
        shrink_n, expand_k = _block_config(self.rank)
        self.fused(
            out, x, a, b, weights, *align, self.rank, self.top_k, self.lora_ids, self.enabled,
            64, shrink_n, 64, 8, 4, 2, 1, 64, 64, expand_k, 8, 4, 2, 1,
            mul_routed_weight=mul_routed_weight,
        )  # fmt: skip

    def __call__(self, cache1, cache2, cache3, x, ids, weights):
        """cache1 [T, top_k, 2I] += gate/up delta; cache3 [T, top_k, H] += weighted down delta."""
        align = self.alignment(ids)
        self._call(cache1, x, self.a13, self.b13, weights, align, False)
        self._call(cache3, cache2, self.a2, self.b2, weights, align, True)
