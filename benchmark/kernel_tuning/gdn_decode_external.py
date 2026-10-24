# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Read-only default CUDA decode leaves; no serving package initializers.

Loader plumbing follows the replay external comparator. SGLang's boundary
follows revision 72730d6.
Only helper/platform/unused decorator shims are supplied; kernel code, math
helpers and launch heuristics execute directly from the supplied checkouts.
"""

import hashlib
import os
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar

from benchmark.kernel_tuning.gdn_replay_external import load_file, temporary_modules


# Local stub for serving-only helpers.
def _unused(*args, **kwargs):
    raise RuntimeError("Serving compile-only helper must not run in this benchmark")


class _WarmupBase(Generic[TypeVar("CompileKey")]):
    pass


VBASE = "vllm/third_party/flash_linear_attention/ops/"
SBASE = "sglang/kernels/ops/attention/"
VLEAVES = {
    "op": VBASE + "op.py",
    "conv": "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "recurrent": VBASE + "fused_recurrent.py",
    "norm": VBASE + "layernorm_guard.py",
}
SLEAVES = {
    "op": SBASE + "fla/op.py",
    "conv": SBASE + "triton_gdn_fused_proj.py",
    "recurrent": SBASE + "fla/fused_recurrent.py",
    "norm": SBASE + "fla/layernorm_gated.py",
}


def _load(root, prefix, paths, exports, op_name):
    if os.environ.get("FLA_USE_FAST_OPS", "0") != "0":
        raise ValueError("Default decode comparison requires FLA_USE_FAST_OPS disabled")
    modules = {}
    hashes = {}
    with temporary_modules(exports):
        path = root / paths["op"]
        hashes[paths["op"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        modules["op"] = load_file(prefix + ".op", path)
        # SGLang uses an absolute import; vLLM uses the private relative name.
        with temporary_modules({op_name: vars(modules["op"])}):
            for name, relative in paths.items():
                if name == "op":
                    continue
                path = root / relative
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                modules[name] = load_file(prefix + "." + name, path)
    return modules, hashes


@lru_cache(maxsize=None)
def load_vllm(root, sm=80):
    """Import-checkable on CPU: SM is supplied, never queried during loading."""
    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    root = Path(root).expanduser().resolve()
    prefix = "_cti_decode_vllm_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    exports = {
        "vllm.triton_utils": {"triton": triton, "tl": tl, "tldevice": tldevice},
        "vllm.platforms": {
            "current_platform": SimpleNamespace(
                is_cpu=lambda: False, is_arch_support_pdl=lambda: sm >= 90
            )
        },
        "vllm.v1.attention.backends.utils": {"NULL_BLOCK_ID": 0, "PAD_SLOT_ID": -1},
        "vllm.utils.math_utils": {
            "cdiv": triton.cdiv,
            "next_power_of_2": triton.next_power_of_2,
        },
        "vllm.utils.platform_utils": {
            "num_compute_units": lambda device: (
                torch.cuda.get_device_properties(device).multi_processor_count
            )
        },
        "vllm.model_executor.warmup.jit_warmup": {"VllmJitKernel": _WarmupBase},
        "vllm.model_executor.warmup.jit_warmup_triton_helper": {
            "TritonPointerInputVariant": object,
            "TritonWarmupTensor": object,
            "triton_scalar_specialization_rep": _unused,
        },
        prefix + ".utils": {
            "input_guard": lambda fn: fn,  # Unused by the selected forward wrappers.
            "is_gather_supported": hasattr(tl, "gather"),
        },
    }
    modules, hashes = _load(root, prefix, VLEAVES, exports, prefix + ".op")
    return SimpleNamespace(
        conv=modules["conv"].causal_conv1d_update,
        recurrent=modules["recurrent"].fused_recurrent_gated_delta_rule_packed_decode,
        norm=modules["norm"].layer_norm_fwd,
        metadata={
            "revision": "0.31.0 (user-supplied tree)",
            "source_sha256": hashes,
            "shims": sorted(exports),
            "dispatch": "CUDA fused-decode enabled, packed recurrence, non-spec Qwen3.5",
            "conv_layout": "SD storage [slot,3,channel]; transposed view to conv",
            "state_layout": "[slot,Hv,V,K] FP32",
            "norm_before_gate": True,
            "gate_reshape_copy": "B > 1",
        },
    )


@lru_cache(maxsize=None)
def load_sglang(root, sm=80):
    import torch
    import triton
    import triton.language as tl

    root = Path(root).expanduser().resolve()
    prefix = "_cti_decode_sglang_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    exports = {
        "sglang.kernels.jit.utils": {"is_arch_support_pdl": lambda: sm >= 90},
        "sglang.srt.utils": {
            "get_bool_env_var": lambda name, default="false": str(default).lower() in ("true", "1"),
            "is_hip": lambda: False,
            "is_cpu": lambda: False,
            "is_npu": lambda: False,
            "cpu_has_amx_support": lambda: False,
            "cdiv": triton.cdiv,
            "next_power_of_2": triton.next_power_of_2,
            "device_context": lambda device: (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            ),
        },
        "sglang.srt.batch_invariant_ops": {"is_batch_invariant_mode_enabled": lambda: False},
        "sglang.srt.model_executor.cuda_graph_config": {
            "Backend": SimpleNamespace(TC_PIECEWISE="tc_piecewise"),
            "Phase": SimpleNamespace(PREFILL="prefill"),
            "check_cuda_graph_backend": lambda *args: False,
        },
        "sglang.kernels.ops.attention.fla.utils": {
            "input_guard": lambda fn: fn,  # Not used by packed decode.
            "is_gather_supported": hasattr(tl, "gather"),
        },
    }
    modules, hashes = _load(root, prefix, SLEAVES, exports, "sglang.kernels.ops.attention.fla.op")
    return SimpleNamespace(
        conv=modules["conv"].fused_qkvzba_causal_conv1d_update_contiguous,
        recurrent=modules["recurrent"].fused_recurrent_gated_delta_rule_packed_decode,
        norm=modules["norm"].rms_norm_gated,
        metadata={
            "revision": "72730d6 (user-supplied tree)",
            "source_sha256": hashes,
            "shims": sorted(exports),
            "dispatch": "CUDA default Triton Qwen3.5 decode; batch invariance disabled",
            "conv_layout": "contiguous [slot,channel,3]",
            "state_layout": "[slot,Hv,V,K] FP32",
            "norm_before_gate": True,
        },
    )


class ExternalDecode:
    """Projection preparation and cache conversion belong outside captured calls."""

    def __init__(self, provider, root, sm, hk, hv, weights, state, history):
        self.provider, self.hk, self.hv = provider, hk, hv
        self.dc = (2 * hk + hv) * 128
        self.weights = weights
        self.leaves = (load_vllm if provider == "vllm" else load_sglang)(root, sm)
        self.state = state.clone()
        self.history = (
            history.transpose(1, 2).contiguous() if provider == "sglang" else history.clone()
        )

    def canonical_history(self):
        return self.history.transpose(1, 2) if self.provider == "sglang" else self.history

    def prepare(self, x):
        import torch

        # Separate contiguous GEMM outputs, prepared once per distinct input step.
        # clone also isolates B=1, where contiguous() alone can alias the packed row.
        split = self.dc + self.hv * 128
        return (
            x[:, :split].clone(memory_format=torch.contiguous_format),
            x[:, split:].clone(memory_format=torch.contiguous_format),
        )

    def step(self, projected, slots):
        import torch

        qkvz, ba = projected[:2]
        cw, al, bias, nw = self.weights
        batch = qkvz.shape[0]
        if self.provider == "sglang":
            qkv, z, b, a = self.leaves.conv(
                qkvz,
                ba,
                self.history,
                cw,
                None,
                slots,
                qkv_dim=self.dc,
                v_dim=self.hv * 128,
                num_v_heads=self.hv,
                head_v_dim=128,
                activation="silu",
            )
            out = qkv.new_empty(batch, 1, self.hv, 128)
        else:
            # Default fused-norm-enabled forward_cuda; plain decode does not use
            # the speculative fused CUDA recurrence even when the _C op exists.
            out = torch.zeros((batch, self.hv, 128), dtype=qkvz.dtype, device=qkvz.device)
            qkv, z = qkvz.split((self.dc, self.hv * 128), dim=-1)
            z = z.reshape(batch, self.hv, 128)
            b, a = ba.chunk(2, dim=-1)
            b, a = b.contiguous(), a.contiguous()
            qkv = self.leaves.conv(
                qkv,
                self.history.transpose(1, 2),
                cw,
                None,
                "silu",
                conv_state_indices=slots,
                validate_data=False,
            )
            out = out.unsqueeze(1)
        self.leaves.recurrent(
            mixed_qkv=qkv,
            a=a,
            b=b,
            A_log=al,
            dt_bias=bias,
            scale=128**-0.5,
            initial_state=self.state,
            out=out,
            ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
            **(projected[2] if len(projected) == 3 else {}),
        )
        x = out.reshape(-1, 128)
        # vLLM's _rms_norm_gated_cuda does this reshape inside its boundary.
        # For B>1, the gate's qkvz row gaps require a real device copy.
        z = z.reshape(-1, 128)
        kwargs = {
            "x": x,
            "weight": nw,
            "bias": None,
            "eps": 1e-6,
            "z": z,
            "group_size": 128,
            "norm_before_gate": True,
            "is_rms_norm": True,
        }
        if self.provider == "vllm":
            self.leaves.norm(**kwargs, out=x, activation="silu")
            return out.reshape(batch, self.hv * 128)
        return self.leaves.norm(**kwargs).reshape(batch, self.hv * 128)


class UnavailableComparator(RuntimeError):
    """A known unavailable external route, with a safe receipt reason."""


class SGLangReplayDecode(ExternalDecode):
    """SGLang's standalone ReplaySSM leaf with the ordinary conv/norm boundary.

    Cursors and flush controls are caller-owned and prepared outside capture.
    Each graph spans complete windows for every state group.
    """

    def __init__(self, root, sm, hk, hv, weights, state, history, window=8):
        import torch

        from benchmark.kernel_tuning.gdn_replay_external import SGLANG_FILE, load_sglang_replay

        path = Path(root).expanduser() / SGLANG_FILE
        if not path.is_file():
            raise UnavailableComparator("SGLang checkout has no standalone ReplaySSM leaf")
        super().__init__("sglang", root, sm, hk, hv, weights, state, history)
        self.window = window
        self.rings = {
            "d_cache": torch.zeros(
                state.shape[0], hv, window, 128, dtype=torch.float32, device=state.device
            ),
            "k_cache": torch.zeros(
                state.shape[0], hk, window, 128, dtype=torch.float32, device=state.device
            ),
            "g_cache": torch.zeros(
                state.shape[0], hv, window, dtype=torch.float32, device=state.device
            ),
        }
        metadata = self.leaves.metadata | {
            "dispatch": "standalone SGLang ReplaySSM with default conv/norm",
            "window": window,
            "source_sha256": self.leaves.metadata["source_sha256"]
            | {SGLANG_FILE: hashlib.sha256(path.read_bytes()).hexdigest()},
        }
        self.leaves = SimpleNamespace(
            **(vars(self.leaves) | {"recurrent": load_sglang_replay(root), "metadata": metadata})
        )

    def prepare(self, x, phase=0, *, force_flush=False):
        import torch

        return (
            *super().prepare(x),
            {
                **self.rings,
                "write_pos": torch.full((x.shape[0],), phase, dtype=torch.int32, device=x.device),
                "force_flush": torch.full(
                    (x.shape[0],), int(force_flush), dtype=torch.int32, device=x.device
                ),
            },
        )

    def reset(self, initial, history):
        self.state.copy_(initial)
        self.canonical_history().copy_(history)
        for ring in self.rings.values():
            ring.zero_()
