# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Our original comparator import glue, reused from gdn_replayssm_compare.py.

External implementations are imported from user-supplied read-only checkouts,
never vendored. Design references: official ReplaySSM at the full revision below,
and SGLang 72730d6. Projection preparation occurs INSIDE the timed callable.
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

REPLAY_COMMIT = "73a9592e9d8a2268620e8883a82f5cec7442ed36"
OFFICIAL_FILE = "vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py"
CONFIG_FILE = "vllm/model_executor/layers/mamba/ops/replayssm_config.py"
SGLANG_FILE = "sglang/kernels/ops/attention/fla/fused_recurrent_linear_replayssm.py"


def load_file(name, path):
    """Execute the external file in memory; never write bytecode into its tree."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Triton resolves JIT functions by module name.
    try:
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@contextmanager
def temporary_modules(exports):
    """Only the two official import dependencies are shimmed, during leaf loading."""
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in exports}
    try:
        for name, values in exports.items():
            module = types.ModuleType(name)
            module.__dict__.update(values)
            sys.modules[name] = module
        yield
    finally:
        for name, old in saved.items():
            if old is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


@lru_cache(maxsize=None)
def load_official(root):
    import torch
    import triton
    import triton.language as tl

    root = Path(root).expanduser().resolve()

    def family(capability, device_id=0):
        # Exact upstream family predicate: 100 means SM10x, NOT all Blackwell.
        # Thus 5090 / PRO 6000 (SM120) / GB10 (SM121) use the non-SM10x config.
        return torch.cuda.get_device_capability(device_id)[0] == capability // 10

    with temporary_modules(
        {
            "vllm.triton_utils": {"triton": triton, "tl": tl},
            "vllm.platforms": {
                "current_platform": types.SimpleNamespace(is_device_capability_family=family)
            },
        }
    ):
        config = load_file("_cti_official_replay_config", root / CONFIG_FILE)
        with temporary_modules(
            {
                "vllm.model_executor.layers.mamba.ops.replayssm_config": {
                    "get_replayssm_config": config.get_replayssm_config
                }
            }
        ):
            module = load_file("_cti_official_replay_decode", root / OFFICIAL_FILE)
    return module.fused_recurrent_gated_delta_rule_replayssm


@lru_cache(maxsize=None)
def load_sglang_replay(root):
    return load_file(
        "_cti_sglang_replay_decode", Path(root).expanduser().resolve() / SGLANG_FILE
    ).fused_recurrent_gdn_replayssm_decode


def load_boundary(root, hk, hv):
    # The serving-independent loader uses Python's import machinery. Suppress
    # its bytecode writes too: external checkouts are read-only inputs.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        from benchmark.kernel_tuning.gdn_delayed_decode import SGLangDecode

        return SGLangDecode(root, hk, hv)
    finally:
        sys.dont_write_bytecode = previous


class External:
    def __init__(
        self, provider, hk, hv, window, weights, state, history, sglang_root, replayssm_root
    ):
        import torch

        self.provider, self.window = provider, window
        self.op = load_boundary(sglang_root, hk, hv)
        self.weights = weights
        self.state = state.clone()
        self.history = history.transpose(1, 2).contiguous()
        self.rings = {}
        self.leaf = None
        if provider != "sglang_default":
            self.leaf = (
                load_official(replayssm_root)
                if provider == "official"
                else load_sglang_replay(sglang_root)
            )
            slots = state.shape[0]
            dtype = torch.float16 if provider == "official" else torch.float32
            self.rings = {
                "d_cache": torch.zeros(slots, hv, window, 128, dtype=dtype, device="cuda"),
                "k_cache": torch.zeros(slots, hk, window, 128, dtype=dtype, device="cuda"),
                "g_cache": torch.zeros(slots, hv, window, dtype=torch.float32, device="cuda"),
            }

    def call(self, x, slots, phase):
        import torch

        boundary = object.__new__(type(self.op))
        boundary.__dict__.update(self.op.__dict__)
        if self.leaf is not None:
            cursor = torch.full_like(slots, phase)
            flush = torch.full(
                slots.shape,
                phase == self.window - 1,
                dtype=torch.int8 if self.provider == "official" else torch.int32,
                device=slots.device,
            )

            def recurrent(**kwargs):
                extra = {"is_flush" if self.provider == "official" else "force_flush": flush}
                return self.leaf(**kwargs, **self.rings, write_pos=cursor, **extra)

            boundary.recurrent = recurrent
        return lambda: boundary(
            boundary.prepare_projected(x), *self.weights, slots, self.state, self.history
        )
