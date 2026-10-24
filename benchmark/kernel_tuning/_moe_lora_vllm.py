# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Execute unmodified vLLM 0.31.0 wheel modules, without a serving installation.

Only serving infrastructure and unused quantization backends are shimmed. The
experts, config selection, alignment wrappers, activations, Punica, and Triton
kernels are real wheel code. Run in a dedicated process: imports are process-wide.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import os
import re
import sys
import types
import zipfile
from pathlib import Path

from benchmark.kernel_tuning._prologue_market import _namespace, ensure_vllm

MOE = "vllm.model_executor.layers.fused_moe"
LORA = "vllm.lora.ops.triton_ops"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unavailable(*args, **kwargs):
    """Fail closed if an off-path shim is accidentally executed."""
    raise NotImplementedError("Off-path vLLM dependency reached")


class TorchPlatform:
    """Hardware answers come from PyTorch, including on CPU import probes."""

    simple_compile_backend = "inductor"

    @staticmethod
    def is_cuda():
        import torch

        return torch.cuda.is_available() and torch.version.cuda is not None

    is_cuda_alike = is_cuda

    @staticmethod
    def is_rocm():
        return False

    is_xpu = is_rocm
    is_fp8_fnuz = is_rocm

    @staticmethod
    def get_device_capability(device_id=None):
        import torch

        return torch.cuda.get_device_capability(device_id) if torch.cuda.is_available() else None

    @classmethod
    def has_device_capability(cls, capability, device_id=None):
        requested = divmod(capability, 10) if isinstance(capability, int) else capability
        actual = cls.get_device_capability(device_id)
        return actual is not None and actual >= requested

    @classmethod
    def is_device_capability(cls, capability, device_id=None):
        requested = divmod(capability, 10) if isinstance(capability, int) else capability
        return cls.get_device_capability(device_id) == requested

    @staticmethod
    def fp8_dtype():
        import torch

        return torch.float8_e4m3fn

    @staticmethod
    def num_compute_units(device_id=None):
        import torch

        return torch.cuda.get_device_properties(device_id).multi_processor_count

    @staticmethod
    def get_device_name(device_id=None):
        import torch

        return torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else "CPU"

    @staticmethod
    def import_kernels():
        # Libraries are loaded explicitly from the wheel by load_libraries().
        return None


def direct_register_custom_op(op_name, op_func, **kwargs):
    """Registration-only shim: retain and call the exact wheel function."""
    import torch

    setattr(torch.ops.vllm, op_name, op_func)


def _stub(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    module.__benchmark_shim__ = True
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def _install_shims(root):
    import torch
    import triton
    import triton.language as tl

    packages = (
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        MOE,
        f"{MOE}.experts",
        f"{MOE}.runner",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.utils",
        "vllm.model_executor.models",
        "vllm.lora",
        "vllm.lora.layers",
        "vllm.lora.ops",
        LORA,
        "vllm.lora.punica_wrapper",
        "vllm.utils",
        "vllm.triton_utils",
        "vllm.compilation",
        "vllm.v1",
        "vllm.v1.worker",
    )
    for name in packages:
        _namespace(name, root)
    sys.modules[MOE].get_config = lambda: None  # No serving override_config context.
    sys.modules["vllm.triton_utils"].__dict__.update(triton=triton, tl=tl, HAS_TRITON=True)
    logger = logging.getLogger("moe_lora_comparator")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    for name in ("info_once", "warning_once", "debug_once"):
        setattr(logger, name, lambda *a, **kw: None)
    _stub("vllm.logger", init_logger=lambda name: logger)
    _stub("vllm.platforms", current_platform=TorchPlatform())
    _stub(
        "vllm.envs",
        VLLM_BATCH_INVARIANT=False,
        VLLM_TUNED_CONFIG_FOLDER=None,
        VLLM_LORA_DISABLE_PDL=False,
        VLLM_LORA_ENABLE_DUAL_STREAM=False,
        VLLM_TRITON_USE_TD=None,
        VLLM_ROCM_FP8_MFMA_PAGE_ATTN=False,
        VLLM_USE_LAYERNAME=True,
    )
    sys.modules["vllm.config"].__dict__.update(
        ParallelConfig=type("ParallelConfig", (), {}),
        SchedulerConfig=types.SimpleNamespace(DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP=8192),
    )
    _stub("vllm.config.kernel", MoEBackend=str)
    distributed = dict.fromkeys(
        (
            "get_dp_group",
            "get_pcp_group",
            "get_tensor_model_parallel_rank",
            "tensor_model_parallel_all_gather",
            "tensor_model_parallel_all_reduce",
            "get_ep_group",
            "get_tp_group",
        ),
        unavailable,
    )
    sys.modules["vllm.distributed"].__dict__.update(distributed)
    _stub("vllm.distributed.parallel_state", **distributed)
    _stub(
        "vllm.utils.platform_utils",
        get_device_name_as_file_name=lambda: re.sub(
            r"[\s/]+", "_", TorchPlatform.get_device_name()
        ),
        is_pin_memory_available=torch.cuda.is_available,
    )
    _stub(
        "vllm.utils.mem_utils",
        get_max_shared_memory_bytes=lambda gpu=0: (
            torch.cuda.get_device_properties(gpu).shared_memory_per_block_optin
        ),
    )
    _stub("vllm.utils.import_utils", get_triton_kernels_version=lambda: None)
    _stub("vllm.utils.flashinfer", flashinfer_quant_nvfp4_8x4_sf_layout=unavailable)
    _stub("vllm.utils.gpu_sync_debug", gpu_sync_allowed=contextlib.nullcontext)
    _stub("vllm.lora.utils", get_captured_lora_counts=unavailable)
    _stub(
        "vllm.compilation.breakable_cudagraph",
        BreakableCUDAGraphCapture=types.SimpleNamespace(is_active=lambda: False),
    )
    _stub(
        "vllm._aiter_ops",
        rocm_aiter_ops=types.SimpleNamespace(
            is_fused_moe_enabled=lambda: False, is_fusion_moe_shared_experts_enabled=lambda: False
        ),
    )
    _stub("vllm.model_executor.models.utils", PPMissingLayer=type("PPMissingLayer", (), {}))
    _stub(
        f"{MOE}.runner.shared_experts",
        SharedExperts=type("SharedExperts", (), {}),
        SharedExpertsOrder=type("SharedExpertsOrder", (), {}),
    )
    _stub(
        "vllm.v1.worker.ubatching",
        **dict.fromkeys(
            ("dbo_enabled", "dbo_maybe_run_recv_hook", "dbo_register_recv_hook", "dbo_yield"),
            unavailable,
        ),
    )
    _stub("vllm.v1.worker.workspace", current_workspace_manager=unavailable)
    # These functions are unreachable for unquantized BF16. Never replace any
    # function used by the measured path with a numerical stand-in.
    quant = {
        "config_utils": ["is_shared_expert_quant_fse_compatible"],
        "fp8_utils": ["per_token_group_quant_fp8", "is_deep_gemm_e8m0_used"],
        "int8_utils": ["per_token_group_quant_int8", "per_token_quant_int8"],
        "mxfp4_utils": ["quant_dequant_mxfp4", "xpu_mxfp4_quantize"],
        "mxfp6_utils": ["quant_dequant_mxfp6"],
        "mxfp8_utils": ["mxfp8_e4m3_quantize", "xpu_mxfp8_quantize"],
        "nvfp4_emulation_utils": ["ref_nvfp4_quant_dequant"],
        "w8a8_utils": ["per_tensor_dequantize"],
    }
    for suffix, names in quant.items():
        _stub(
            f"vllm.model_executor.layers.quantization.utils.{suffix}",
            **dict.fromkeys(names, unavailable),
        )


class WheelRuntime:
    """A wheel-backed module loader and provenance recorder."""

    def __init__(self, cache: Path, *, load_ops=True):
        _, info = ensure_vllm(cache)
        self.wheel = cache / info["wheel"]
        digest = file_sha256(self.wheel)
        self.root = cache / digest[:16]
        self.hashes = {}
        self.libraries = []
        self.library_hashes = {}
        with zipfile.ZipFile(self.wheel) as archive:
            for member in archive.namelist():
                path = Path(member)
                if path.is_absolute() or ".." in path.parts or path.parts[0] != "vllm":
                    continue
                source = member.endswith((".py", ".json"))
                library = member in (
                    "vllm/_C_stable_libtorch.abi3.so",
                    "vllm/_moe_C_stable_libtorch.abi3.so",
                )
                if not (source or library):
                    continue
                data = archive.read(member)
                destination = self.root / member
                destination.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not destination.is_file()
                    or hashlib.sha256(destination.read_bytes()).digest()
                    != hashlib.sha256(data).digest()
                ):
                    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                if member.endswith(".py"):
                    self.hashes[member] = hashlib.sha256(data).hexdigest()
                if library:
                    self.libraries.append(destination)
                    self.library_hashes[path.name] = hashlib.sha256(data).hexdigest()
        self.info = {
            "version": "0.31.0",
            "wheel": self.wheel.name,
            "wheel_sha256": digest,
            "scaling": "vLLM B factors pre-scaled; ours receives live scaling",
            "entrypoint": "TritonExperts.apply",
            "shared": "PunicaWrapperGPU.add_lora_linear (gate_up + down deltas)",
        }
        _install_shims(self.root)
        torch_utils = importlib.import_module("vllm.utils.torch_utils")
        torch_utils.direct_register_custom_op = direct_register_custom_op
        if load_ops:
            self.load_libraries()
        self.config = self.module(f"{MOE}.config")
        self.base = self.module(f"{MOE}.fused_moe")
        layers = sys.modules["vllm.lora.layers"]
        layers.LoRAMapping = self.module("vllm.lora.layers.utils").LoRAMapping
        exports = {
            "fused_moe_lora": "fused_moe_lora_op",
            "lora_shrink": "lora_shrink_op",
            "lora_expand": "lora_expand_op",
            "LoRAKernelMeta": "lora_kernel_metadata",
        }
        for name, suffix in exports.items():
            setattr(sys.modules[LORA], name, getattr(self.module(f"{LORA}.{suffix}"), name))
        self.punica = self.module("vllm.lora.punica_wrapper.punica_gpu")
        self.experts = self.module(f"{MOE}.experts.triton_moe")
        self.context = self.module(f"{MOE}.experts.lora_context")

    def module(self, name):
        return importlib.import_module(name)

    def load_libraries(self):
        import torch

        for path in sorted(self.libraries):
            torch.ops.load_library(str(path))
        for namespace, name in (
            ("_C", "silu_and_mul"),
            ("_moe_C", "moe_sum"),
            ("_moe_C", "moe_align_block_size"),
            ("_moe_C", "moe_lora_align_block_size"),
        ):
            getattr(getattr(torch.ops, namespace), name)

    def provenance(self):
        sources = {}
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if (
                name.startswith("vllm.")
                and filename
                and getattr(module.__spec__, "loader", None) is not None
            ):
                path = Path(filename)
                if path.is_relative_to(self.root) and path.is_file():
                    member = path.relative_to(self.root).as_posix()
                    if member in self.hashes:
                        sources[member] = self.hashes[member]
        shims = sorted(
            name
            for name, module in tuple(sys.modules.items())
            if name.startswith("vllm.") and getattr(module, "__benchmark_shim__", False)
        )
        return dict(
            self.info,
            source_sha256=sources,
            library_sha256=self.library_hashes,
            off_path_shims=shims,
            registration="direct calls to wheel functions",
            config_override=None,
            environment="vLLM defaults; dual-stream selected via context",
        )


def path_flags(tokens, rank, ep, *, experts=256, top_k=8):
    """Report Punica's exact BF16, one-adapter dispatch predicates."""
    naive = ep == 1 and tokens * top_k * 8 <= experts // ep
    small = naive and rank <= 64 and tokens * top_k * rank <= 1024
    return {"naive": naive, "small_batch": small, "one_shot": not small}


class SharedLayer:
    """Dense shared-expert deltas through the wheel's unmodified Punica path.

    add_lora_linear allocates an FP32 [slices, T, rank] buffer on each call;
    shrink clears it, then expand adds the slices at their cumulative offsets.
    Metadata and pre-scaled B weights are prepared outside the timed calls.
    """

    def __init__(self, runtime, x, factors, scaling):
        import torch

        a13, b13, a2, b2 = factors
        self.wrapper = object.__new__(runtime.punica.PunicaWrapperGPU)
        self.wrapper.lora_config = types.SimpleNamespace(specialize_active_lora=True)
        self.wrapper.token_mapping_meta = runtime.module(
            f"{LORA}.lora_kernel_metadata"
        ).LoRAKernelMeta.make(1, x.shape[0], x.device)
        self.wrapper.token_mapping_meta.prepare_tensors(
            torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)
        )
        self.a13 = tuple(a.unsqueeze(0) for a in a13.unbind(0))
        self.b13 = tuple((b * scaling).unsqueeze(0) for b in b13.unbind(0))
        self.a2, self.b2 = (a2.unsqueeze(0),), ((b2 * scaling).unsqueeze(0),)
        self.slices13, self.slices2 = (b13.shape[1],) * 2, (b2.shape[0],)

    def delta(self, cache, out, x, hidden):
        self.wrapper.add_lora_linear(
            cache, x, self.a13, self.b13, 1.0, self.slices13, add_inputs=True
        )
        self.wrapper.add_lora_linear(
            out, hidden, self.a2, self.b2, 1.0, self.slices2, add_inputs=True
        )
        return cache, out


class RoutedLayer:
    """Real TritonExperts with explicit workspaces and a single real adapter.

    EP measures the local partial output, without communication on either side.
    Each invocation still includes base and LoRA alignment inside apply().
    """

    def __init__(
        self,
        runtime,
        x,
        w13,
        w2,
        ids,
        weights,
        factors,
        scaling,
        ep,
        *,
        model,
        ep_rank=0,
        dual=False,
        base=False,
        fp8_scales=None,
    ):
        """``fp8_scales`` = (w13_scale, w2_scale): E4M3 weights with 128x128 block scales,
        vLLM's block-FP8 W8A8 Triton path (activations quantized 1x128 per token)."""
        import torch

        self.runtime = runtime
        self.x, self.w13, self.w2 = x, w13, w2
        self.ids, self.weights = ids, weights
        self.t, self.h = x.shape
        self.k = ids.shape[1]
        self.e, self.twice_i, _ = w13.shape
        if (self.h, self.twice_i, self.e, self.k) != (
            model.hidden,
            2 * model.intermediate,
            model.experts // ep,
            model.top_k,
        ):
            raise ValueError("routed tensors do not match the model spec")
        self.activation = runtime.module(f"{MOE}.activation").MoEActivation.SILU
        parallel = runtime.config.FusedMoEParallelConfig.make_no_parallel()
        parallel.ep_size, parallel.use_ep = ep, ep != 1
        config = runtime.config.FusedMoEConfig(
            num_experts=model.experts,
            experts_per_token=self.k,
            hidden_dim=self.h,
            intermediate_size=self.twice_i // 2,
            num_local_experts=self.e,
            num_logical_experts=model.experts,
            activation=self.activation,
            device=x.device,
            routing_method=runtime.config.RoutingMethodType.Renormalize,
            moe_parallel_config=parallel,
            in_dtype=x.dtype,
            is_lora_enabled=not base,
        )
        quant = runtime.config.FUSED_MOE_UNQUANTIZED_CONFIG
        if fp8_scales is not None:
            quant = runtime.config.fp8_w8a8_moe_quant_config(
                w1_scale=fp8_scales[0], w2_scale=fp8_scales[1], block_shape=[128, 128]
            )
        self.expert = runtime.experts.TritonExperts(config, quant)
        self.expert_map = None
        if ep != 1:
            self.expert_map = torch.full((self.e * ep,), -1, dtype=torch.int32, device=x.device)
            start = ep_rank * self.e
            self.expert_map[start : start + self.e] = torch.arange(
                self.e, dtype=torch.int32, device=x.device
            )
        self.base_config = runtime.base.try_get_optimal_moe_config(
            w13.shape,
            w2.shape,
            self.k,
            None if fp8_scales is None else "fp8_w8a8",
            self.t,
            None if fp8_scales is None else [128, 128],
        )
        shapes = self.expert.workspace_shapes(
            self.t, self.twice_i, self.h, self.k, self.e * ep, self.e, None, self.activation
        )
        self.workspace13, self.workspace2, self.output = (
            torch.empty(shape, device=x.device, dtype=x.dtype) for shape in shapes
        )
        self.ctx = None
        if not base:
            ga, gb, ua, ub, da, db = factors
            wrapper = object.__new__(runtime.punica.PunicaWrapperGPU)
            wrapper.lora_config = types.SimpleNamespace(specialize_active_lora=True)
            wrapper.token_mapping_meta = runtime.module(
                f"{LORA}.lora_kernel_metadata"
            ).LoRAKernelMeta.make(1, self.t, x.device)
            wrapper.token_mapping_meta.prepare_tensors(
                torch.zeros(self.t, dtype=torch.int32, device=x.device)
            )
            # Punica does not mutate factor tensors. Multiplication occurs once,
            # as in vLLM's weight loader; it is not part of the timed call.
            self.ctx = runtime.context.MoELoRAContext(
                w13_lora_a_stacked=(ga.unsqueeze(0), ua.unsqueeze(0)),
                w13_lora_b_stacked=((gb * scaling).unsqueeze(0), (ub * scaling).unsqueeze(0)),
                w2_lora_a_stacked=(da.unsqueeze(0),),
                w2_lora_b_stacked=((db * scaling).unsqueeze(0),),
                adapter_enabled=torch.ones(1, dtype=torch.int32, device=x.device),
                max_loras=1,
                top_k=self.k,
                w13_num_slices=2,
                fully_sharded=False,
                tp_rank=0,
                tp_size=1,
                local_num_experts=self.e,
                punica_wrapper=wrapper,
                use_tuned_config=False,
                aux_stream=torch.cuda.Stream() if dual else None,
                events=tuple(torch.cuda.Event() for _ in range(4)) if dual else None,
            )
            self.expert.set_lora_context(self.ctx)

    def __call__(self, x=None, ids=None, weights=None):
        if x is not None:
            self.x, self.ids, self.weights = x, ids, weights
        self.expert.apply(
            self.output,
            self.x,
            self.w13,
            self.w2,
            self.weights,
            self.ids,
            self.activation,
            self.e * self.expert.moe_config.moe_parallel_config.ep_size,
            self.expert_map,
            None,
            None,
            self.workspace13,
            self.workspace2,
            None,
            False,
        )
        return self.output

    def delta(self, cache1, cache2, cache3, x=None, ids=None, weights=None):
        """LoRA-only calls, including real alignment; cache2 is fixed input."""
        if x is not None:
            self.x, self.ids, self.weights = x, ids, weights
        sorted_ids, expert_ids, padded, mapping = self.expert.apply_w13_lora(
            self.ctx,
            y=cache1,
            x=self.x,
            topk_ids=self.ids,
            topk_weights=self.weights,
            expert_map=self.expert_map,
            w1=self.w13,
            w2=self.w2,
            num_tokens=self.t,
            top_k_num=self.k,
        )
        self.expert.apply_w2_lora(
            self.ctx,
            y=cache3.view(self.t, self.k, self.h),
            x=cache2,
            topk_weights=self.weights,
            sorted_token_ids_lora=sorted_ids,
            expert_ids_lora=expert_ids,
            num_tokens_post_padded_lora=padded,
            token_lora_mapping=mapping,
            num_tokens=self.t,
            w1=self.w13,
            w2=self.w2,
            top_k_num=self.k,
        )
