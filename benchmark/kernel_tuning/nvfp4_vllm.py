# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""vLLM 0.31.0 ModelOpt comparators, using installed methods and CUDA operators.

Construction consumes raw checkpoint tensors and prepares weights outside timing.
Merged gate/up globals must be equal: vLLM collapses unequal globals to their max
without compensating the slots. No framework imports occur until construction.
"""

from __future__ import annotations

import importlib.metadata as md
from types import SimpleNamespace

VERSION = "0.31.0"
FLASHINFER_VERSION = "0.7.0.post1"
_LOADED = None
_CONFIG = None


def load():
    global _LOADED
    if _LOADED is None:
        for package, expected in (("vllm", VERSION), ("flashinfer-python", FLASHINFER_VERSION)):
            if md.version(package) != expected:
                raise RuntimeError(f"vLLM parity requires {package}=={expected}")
        import torch
        from vllm import _custom_ops
        from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
            FlashInferCutlassNvFp4LinearKernel,
        )
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
            build_linear_method,
        )

        _single_process_groups()
        _LOADED = SimpleNamespace(
            torch=torch,
            ops=_custom_ops,
            config=ModelOptNvFp4Config,
            build=build_linear_method,
            kernel=FlashInferCutlassNvFp4LinearKernel,
        )
    return _LOADED


def _single_process_groups():
    # Parameter construction reads the tensor-parallel rank: form TP=PP=1 groups
    # over gloo in this process (no NCCL communicator, no server).
    import tempfile

    import torch
    from vllm.distributed import parallel_state as ps

    if ps.model_parallel_is_initialized():
        return
    # Default engine config for the life of the process: group setup and custom ops
    # read it (an unset config is an error in vLLM 0.31).
    from vllm.config import VllmConfig, set_current_vllm_config

    global _CONFIG
    _CONFIG = set_current_vllm_config(VllmConfig())
    _CONFIG.__enter__()
    if not torch.distributed.is_initialized():
        handle, path = tempfile.mkstemp(prefix="nvfp4-vllm-pg-")
        import os

        os.close(handle)
        os.unlink(path)
        ps.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + path,
            backend="gloo",
        )
    ps.initialize_model_parallel(1, 1, backend="gloo")


class VLLMLinear:
    """BF16 input quantization and GEMM through the real ModelOpt method."""

    def __init__(self, weights, input_scale):
        self.api = load()
        torch = self.api.torch
        for weight in weights[1:]:
            torch.testing.assert_close(
                weight.global_scale,
                weights[0].global_scale,
                rtol=0,
                atol=0,
                msg="vLLM comparator requires equal gate/up global scales",
            )
        widths = [weight.data.shape[0] for weight in weights]
        k = weights[0].data.shape[1] * 2
        cfg = self.api.config(
            quant_method="NVFP4", is_checkpoint_nvfp4_serialized=True, group_size=16
        )
        projection = "gate_up_proj" if len(weights) == 2 else "down_proj"
        self.method = self.api.build(cfg, "NVFP4", f"model.layers.0.mlp.{projection}")
        self.layer = torch.nn.Module()
        with torch.device(input_scale.device), torch.no_grad():
            self.method.create_weights(self.layer, k, widths, k, sum(widths), torch.bfloat16)
            assert type(self.method.kernel) is self.api.kernel, (
                "Expected FlashInferCutlassNvFp4LinearKernel, selected "
                + type(self.method.kernel).__name__
            )
            self.layer.weight.copy_(torch.cat([weight.data for weight in weights]))
            self.layer.weight_scale.copy_(torch.cat([weight.scale for weight in weights]))
            self.layer.weight_scale_2.copy_(
                torch.cat([weight.global_scale.reshape(1) for weight in weights])
            )
            self.layer.input_scale.copy_(input_scale.expand(len(widths)))
            self.method.process_weights_after_loading(self.layer)
        self._probed = set()

    def __call__(self, x):
        return self.method.apply(self.layer, x, bias=None)

    def probe(self, x):
        result = self(x)
        self._probed.add(x.shape[0])
        return result

    def metadata(self, rows):
        assert rows in self._probed, "Probe every M before graph timing"
        return {
            "vllm_version": md.version("vllm"),
            "flashinfer_version": md.version("flashinfer-python"),
            "kernel_class": type(self.method.kernel).__name__,
            "chosen_backend": "flashinfer-cutlass",
            "boundary": "BF16 activation -> input quantization -> BF16 GEMM output",
            "weight_global_scales": "equal-per-slot",
            "autotuning": "default-context (no explicit autotune)",
        }


class VLLMGated(VLLMLinear):
    """Gate/up GEMM plus the default compiled-serving activation/quant operator.

    This invokes the compiler replacement directly, without compiling a serving
    model. Shared experts use the same boundary. ``eager`` labels the separate
    activation and quantizer calls and their different rounding explicitly.
    """

    def __init__(self, gate, up, input_scale, output_scale, *, shared=False, eager=False):
        super().__init__((gate, up), input_scale)
        self.shared = shared
        self.eager = eager
        self.output_scale_inv = output_scale.reciprocal()

    def __call__(self, x):
        torch = self.api.torch
        gate_up = super().__call__(x)
        m, twice_n = gate_up.shape
        n = twice_n // 2
        if self.eager:
            activated = torch.empty((m, n), dtype=gate_up.dtype, device=gate_up.device)
            torch.ops._C.silu_and_mul(activated, gate_up)
            return self.api.ops.scaled_fp4_quant(
                activated, self.output_scale_inv, True, backend="flashinfer-cutlass"
            )
        packed, scales = self.api.ops.create_fp4_output_tensors(m, n, gate_up.device, True)
        torch.ops._C.silu_and_mul_nvfp4_quant(packed, scales, gate_up, self.output_scale_inv)
        return packed, scales.view(torch.float8_e4m3fn)

    def metadata(self, rows):
        return {
            **super().metadata(rows),
            "boundary": "BF16 activation -> NVFP4 + E4M3 scales",
            "vllm_gate_up_dtype": "bfloat16",
            "activation_path": "eager-separate" if self.eager else "compiled-serving-fused-op",
            "shared_expert": self.shared,
        }
