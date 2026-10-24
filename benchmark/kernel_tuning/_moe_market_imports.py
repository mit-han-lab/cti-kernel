# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Import isolation for the standalone, BF16 SGLang block benchmark.

All measured kernels and the runtime context are imported from the supplied
package. Small ordinary Python dependencies (including msgspec) must be installed.
"""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path


def _unused(*args, **kwargs):
    raise RuntimeError("An off-path server hook was unexpectedly called")


def load_sglang(path):
    root = Path(path) / "sglang"
    if not (root / "version.py").is_file():
        raise ValueError("--sglang-path must contain the sglang package")
    if "sglang" in sys.modules and root.resolve() not in {
        Path(p).resolve() for p in sys.modules["sglang"].__path__
    }:
        raise ValueError("SGLang was already imported from a different package directory")
    shims = []
    if "sglang" not in sys.modules:
        # The public client initializer imports Transformers and HTTP clients;
        # those are unrelated to a tensor-only block. Keep real submodule paths.
        package = types.ModuleType("sglang")
        package.__path__ = [str(root)]
        package.__spec__ = importlib.machinery.ModuleSpec("sglang", None, is_package=True)
        sys.modules["sglang"] = package
        shims.append("sglang public client initializer")
    name = "sglang.srt.layers.deep_gemm_wrapper"
    if name not in sys.modules:
        # The unused DeepGEMM backend imports the full model executor and
        # Transformers at import time. An empty module fails if accidentally
        # called; the measured Triton BF16 path never accesses this backend.
        module = types.ModuleType(name)
        module.__path__ = [str(root / "srt/layers/deep_gemm_wrapper")]
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        sys.modules[name] = module
        shims.append("unused DeepGEMM backend loader")
    # This KV-cache hook eagerly imports every Transformers model config. The
    # real resolution pipeline returns for model_path='dummy' before calling it.
    # Likewise CP/TP group sharing only runs after the dummy-model return.
    for suffix, function in (
        ("kv_shard_hook", "handle_kv_cache_sharding"),
        ("cp_tp_group_sharing_hook", "resolve_cp_tp_group_sharing"),
    ):
        name = "sglang.srt.arg_groups." + suffix
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            setattr(module, function, _unused)
            sys.modules[name] = module
            shims.append("unused " + suffix)
    version = importlib.import_module("sglang.version").__version__
    if version != "0.5.22":
        raise ValueError("SGLang 0.5.22 is required")
    modules = {
        "fused": "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
        "config": "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config",
        "gate": "sglang.kernels.ops.moe.moe_fused_gate",
        "activation": "sglang.kernels.ops.activation.activation",
        "elementwise": "sglang.kernels.ops.elementwise.elementwise",
        "runtime": "sglang.srt.runtime_context",
        "server": "sglang.srt.server_args",
    }
    result = {key: importlib.import_module(value) for key, value in modules.items()}
    result.update(version=version, shims=shims)
    return types.SimpleNamespace(**result)
