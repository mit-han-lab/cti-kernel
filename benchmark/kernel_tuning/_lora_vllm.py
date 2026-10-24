# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Load unmodified vLLM 0.31.0 LoRA wheel modules as benchmark comparators.

Adapted from our wheel-loader experiment. Only package initializers, platform
queries and custom-op registration are shimmed; metadata, pointer preparation,
config selection and numerical kernels execute the wheel's own code. Imports
are process-wide, so use a dedicated benchmark process.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import types
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.kernel_tuning._prologue_market import _namespace, ensure_vllm

OPS = "vllm.lora.ops.triton_ops"


def _stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


class _Platform:
    @staticmethod
    def is_cuda():
        return torch.cuda.is_available() and torch.version.cuda is not None

    @staticmethod
    def has_device_capability(capability):
        return _Platform.is_cuda() and torch.cuda.get_device_capability() >= divmod(capability, 10)


def _register(op_name, op_func, **kwargs):
    # Registration metadata only; the exported op is the exact wheel function.
    setattr(torch.ops.vllm, op_name, op_func)


def load_vllm(cache):
    """Download/cache the pinned wheel outside the repo; never load serving code."""
    import triton
    import triton.language as tl

    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise RuntimeError("vLLM comparator requires a process without existing vLLM imports")
    cache = Path(cache)
    _, info = ensure_vllm(cache)
    root = cache / "lora-0.31.0"
    hashes = {}
    with zipfile.ZipFile(cache / info["wheel"]) as archive:
        for member in archive.namelist():
            path = Path(member)
            if path.is_absolute() or ".." in path.parts:
                continue
            if not (
                member.startswith("vllm/lora/ops/triton_ops/")
                and member.endswith((".py", ".json"))
                or member in ("vllm/utils/torch_utils.py", "vllm/utils/math_utils.py")
            ):
                continue
            data = archive.read(member)
            destination = root / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_bytes(data)
            temporary.replace(destination)
            hashes[member] = hashlib.sha256(data).hexdigest()
    for name in ("vllm", "vllm.lora", "vllm.lora.ops", OPS, "vllm.utils", "vllm.triton_utils"):
        _namespace(name, root)
    sys.modules["vllm.triton_utils"].__dict__.update(triton=triton, tl=tl)
    logger = logging.getLogger("lora_vllm_comparator")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    for name in ("info_once", "warning_once", "debug_once"):
        setattr(logger, name, lambda *a, **kw: None)
    _stub("vllm.logger", init_logger=lambda name: logger)
    _stub("vllm.platforms", current_platform=_Platform())
    _stub("vllm.utils.platform_utils", is_pin_memory_available=torch.cuda.is_available)
    _stub(
        "vllm.envs",
        VLLM_BATCH_INVARIANT=False,
        VLLM_TUNED_CONFIG_FOLDER=None,
        VLLM_LORA_DISABLE_PDL=False,
        VLLM_LORA_ENABLE_DUAL_STREAM=False,
        VLLM_USE_LAYERNAME=True,
    )
    utilities = importlib.import_module("vllm.utils.torch_utils")
    utilities.direct_register_custom_op = _register
    shrink = importlib.import_module(f"{OPS}.lora_shrink_op").lora_shrink
    expand = importlib.import_module(f"{OPS}.lora_expand_op").lora_expand
    metadata = importlib.import_module(f"{OPS}.lora_kernel_metadata").LoRAKernelMeta
    return types.SimpleNamespace(
        shrink=shrink,
        expand=expand,
        metadata=metadata,
        info={
            "version": "0.31.0",
            "wheel": info["wheel"],
            "source_sha256": hashes,
            "entrypoints": [f"{OPS}.lora_shrink", f"{OPS}.lora_expand"],
            "registration": "direct calls to wheel functions",
            "off_path_shims": [
                "package initializers",
                "logger",
                "platform queries",
                "env defaults",
            ],
            "specialize_active_lora": True,
            "scaling": "B pre-scaled once as in LoRALayerWeights.optimize; shrink scale=1",
        },
    )


class PunicaLinear:
    """One adapter, real LoRAKernelMeta, FP32 scratch and add_inputs=True.

    Calls the same shrink/expand pair as PunicaWrapperGPU.add_lora_linear.
    Routing preparation occurs once, outside the timed projection, as in decode.
    """

    base_provider = "torch"

    def __init__(self, runtime, x, weight, factors, scaling):
        self.runtime, self.x, self.weight = runtime, x, weight
        self.a = tuple(a.unsqueeze(0) for a in factors[0::2])
        self.b = tuple((b * scaling).unsqueeze(0) for b in factors[1::2])
        self.meta = runtime.metadata.make(1, x.shape[0], x.device)
        self.meta.prepare_tensors(torch.zeros(x.shape[0], dtype=torch.int32, device=x.device))

    def base(self):
        return F.linear(self.x, self.weight)

    def __call__(self):
        output = self.base()
        scratch = torch.empty(
            (len(self.a), self.x.shape[0], self.a[0].shape[-2]),
            dtype=torch.float32,
            device=self.x.device,
        )
        routing = self.meta.meta_args(self.x.shape[0], specialize_active_lora=True)
        self.runtime.shrink(self.x, self.a, scratch, *routing, 1.0)
        self.runtime.expand(scratch, self.b, output, *routing, offset_start=0, add_inputs=True)
        return output
