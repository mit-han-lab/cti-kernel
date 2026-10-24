# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Read-only leaf loaders for the explicitly selected CUDA/Triton verify paths.

No serving package is imported. Shims supply CUDA platform predicates, math
helpers, and disabled serving-only modes. vLLM's missing warmup framework is
represented by a generic base and compile-only placeholders: the external
LayerNormFwdKernel.__call__ still launches its original kernel unmodified.
The input_guard shim is unused by our direct layer_norm_fwd call. SGLang uses
ordinary (non batch-invariant, non piecewise-prefill) norm dispatch and PDL on
SM90+. These choices are part of the benchmark configuration, not auto detection
of a running engine. No kernel implementation is shimmed or copied.
"""

import ast
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar

from benchmark.kernel_tuning.gdn_replay_external import load_file, temporary_modules

VCORE = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
VCONV = "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
VFUSED = "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py"
VNORM = "vllm/third_party/flash_linear_attention/ops/layernorm_guard.py"
SCORE = "sglang/srt/layers/attention/linear/gdn_backend.py"
STRITON = "sglang/srt/layers/attention/linear/kernels/gdn_triton.py"
SCONV = "sglang/kernels/ops/mamba/causal_conv1d_triton.py"
SFUSED = "sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py"
SNORM = "sglang/kernels/ops/attention/fla/layernorm_gated.py"
SSCATTER = "sglang/kernels/ops/mamba/mamba_state_scatter_triton.py"
SPTR = "sglang/kernels/ops/memory/ptr_table.py"
SPROJ = "sglang/kernels/ops/attention/triton_gdn_fused_proj.py"

DISPATCH = {
    "ours": [
        "benchmark/kernel_tuning/gdn_replay_common.py:Replay.__init__",
        "python/ctikernels/gdn/replay/_api.py:GatedDeltaNetReplay.begin",
        "python/ctikernels/gdn/replay/_api.py:GatedDeltaNetReplay.verify",
        "python/ctikernels/gdn/replay/_api.py:GatedDeltaNetReplay.commit",
        "python/ctikernels/gdn/replay/_api.py:GatedDeltaNetReplay.end",
        "python/ctikernels/gdn/replay/cute/_phase_launch.py:run",
    ],
    "vllm": [
        f"{VCORE}:QwenGatedDeltaNetAttention._forward_core",
        f"{VCORE}:QwenGatedDeltaNetAttention.rearrange_mixed_qkv",
        f"{VCONV}:causal_conv1d_update",
        f"{VFUSED}:fused_sigmoid_gating_delta_rule_update",
        f"{VNORM}:layer_norm_fwd",
    ],
    "sglang": [
        "sglang/srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet.forward",
        f"{SPROJ}:fused_qkvzba_split_reshape_cat_contiguous",
        f"{SCORE}:GDNAttnBackend.forward_extend",
        f"{STRITON}:TritonGDNKernel.target_verify",
        f"{SCONV}:causal_conv1d_update",
        f"{SFUSED}:fused_sigmoid_gating_delta_rule_update",
        f"{SNORM}:_layer_norm_fwd",
        "sglang/srt/layers/attention/hybrid_linear_attn_backend.py:HybridLinearAttnBackend.update_mamba_state_after_mtp_verify",
        f"{SSCATTER}:scatter_mamba_states_after_mtp_verify",
        f"{SSCATTER}:fused_mamba_state_scatter_with_mask",
        f"{SSCATTER}:fused_conv_window_scatter_multi",
    ],
}


def _unused(*args, **kwargs):
    raise RuntimeError("Serving compile-only helper must not run in this benchmark")


def load_vllm_rearrange(root):
    """Load the original tensor-only method without importing the serving stack.

    Compile its AST directly from the supplied checkout, in memory. No source
    is copied into the repository and no third-party bytecode is written.
    """
    import torch

    path = Path(root).expanduser().resolve() / VCORE
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenGatedDeltaNetAttention"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "rearrange_mixed_qkv"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class _WarmupBase(Generic[TypeVar("CompileKey")]):
    pass


@lru_cache(maxsize=None)
def load_vllm(root, sm):
    import torch
    import triton
    import triton.language as tl

    root = Path(root).expanduser().resolve()
    platform = SimpleNamespace(is_cpu=lambda: False, is_arch_support_pdl=lambda: sm >= 90)
    exports = {
        "vllm.triton_utils": {"triton": triton, "tl": tl},
        "vllm.platforms": {"current_platform": platform},
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
        "_cti_verify_vllm.utils": {"input_guard": lambda fn: fn},
    }
    with temporary_modules(exports):
        conv = load_file("_cti_verify_vllm.conv", root / VCONV)
        recurrent = load_file("_cti_verify_vllm.recurrent", root / VFUSED)
        norm = load_file("_cti_verify_vllm.norm", root / VNORM)
    return SimpleNamespace(
        rearrange=load_vllm_rearrange(root),
        conv=conv.causal_conv1d_update,
        recurrent=recurrent.fused_sigmoid_gating_delta_rule_update,
        norm=norm.layer_norm_fwd,
    )


@lru_cache(maxsize=None)
def load_sglang(root, sm):
    import torch
    import triton

    root = Path(root).expanduser().resolve()
    exports = {
        "sglang.kernels.jit.utils": {"is_arch_support_pdl": lambda: sm >= 90},
        "sglang.srt.utils": {
            "get_device_sm": lambda: sm,
            "is_cuda": lambda: True,
            "is_hip": lambda: False,
            "get_bool_env_var": lambda *args: False,
            "is_gfx95_supported": lambda: False,
            "is_sm90_supported": lambda: sm == 90,
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
    }
    ptr = load_file("_cti_verify_sglang.ptr", root / SPTR)
    exports["sglang.kernels.ops.memory.ptr_table"] = {"make_ptr_table": ptr.make_ptr_table}
    with temporary_modules(exports):
        conv = load_file("_cti_verify_sglang.conv", root / SCONV)
        recurrent = load_file("_cti_verify_sglang.recurrent", root / SFUSED)
        norm = load_file("_cti_verify_sglang.norm", root / SNORM)
        scatter = load_file("_cti_verify_sglang.scatter", root / SSCATTER)
        projection = load_file("_cti_verify_sglang.projection", root / SPROJ)
    return SimpleNamespace(
        conv=conv.causal_conv1d_update,
        recurrent=recurrent.fused_sigmoid_gating_delta_rule_update,
        norm=norm._layer_norm_fwd,
        scatter=scatter.scatter_mamba_states_after_mtp_verify,
        unpack=projection.fused_qkvzba_split_reshape_cat_contiguous,
    )
