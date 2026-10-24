# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Runtime-only adapters for vLLM 0.31.0 and SGLang 72730d6.

No upstream implementation is vendored. Reuse the round benchmark's AST loader.
Warmup registration is a no-op: actual warmup is performed before graph capture.
Dispatch configuration AND argument/stride binding come from upstream. Container
classes and platform/logging queries replace application infrastructure only.
No penalties, grammar, min-p, logprobs, watermarking, or synthetic acceptance.
"""

import ast
import contextlib
import dataclasses
import functools
import hashlib
import importlib
import importlib.machinery
import inspect
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

from benchmark.dflash_round.comparators import _definition_names, load_definitions

REQUIRED = (
    "vllm/model_executor/warmup/jit_warmup_triton_helper.py",
    "vllm/v1/sample/ops/topk_topp_triton.py",
    "vllm/v1/sample/ops/topk_topp_sampler.py",
    "vllm/v1/sample/sampler.py",
    "vllm/v1/sample/rejection_sampler.py",
    "vllm/v1/attention/backends/flashinfer.py",
    "sglang/srt/speculative/dflash_utils.py",
    "sglang/kernels/ops/speculative/reject_sampling.py",
    "sglang/kernels/ops/speculative/dflash.py",
)
V2_FILES = tuple(
    "vllm/v1/worker/gpu/" + path
    for path in (
        "spec_decode/rejection_sampler.py",
        "spec_decode/rejection_sampler_utils.py",
        "sample/sampler.py",
        "sample/states.py",
        "sample/gumbel.py",
        "sample/min_p.py",
        "sample/thinking_budget.py",
        "sample/watermark.py",
        "sample/output.py",
        "sample/logits_processor/interface.py",
        "input_batch.py",
    )
) + ("vllm/v1/watermarking/prfs/philox.py",)


SGLANG_JIT_FILES = (
    "sglang/kernels/jit/__init__.py",
    "sglang/kernels/jit/csrc/speculative/sampling.cuh",
    "sglang/kernels/jit/include/sgl_kernel/bits.h",
    "sglang/kernels/jit/include/sgl_kernel/runtime.cuh",
    "sglang/kernels/jit/include/sgl_kernel/source_location.h",
    "sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh",
    "sglang/kernels/jit/include/sgl_kernel/tensor.h",
    "sglang/kernels/jit/include/sgl_kernel/utils.cuh",
    "sglang/kernels/jit/include/sgl_kernel/utils.h",
    "sglang/kernels/jit/utils/__init__.py",
    "sglang/kernels/jit/utils/arch.py",
    "sglang/kernels/jit/utils/common.py",
    "sglang/kernels/jit/utils/compile/__init__.py",
    "sglang/kernels/jit/utils/compile/cache.py",
    "sglang/kernels/jit/utils/compile/cpp_args.py",
    "sglang/kernels/jit/utils/compile/loader.py",
    "sglang/kernels/jit/utils/compile/ninja.py",
    "sglang/kernels/jit/utils/compile/paths.py",
    "sglang/kernels/jit/utils/compile/spec.py",
    "sglang/kernels/jit/utils/compile/toolchain.py",
    "sglang/kernels/jit/utils/deps.py",
    "sglang/kernels/kernel_api_logging.py",
    "sglang/kernels/ops/speculative/sampling.py",
    "sglang/srt/environ.py",
    "sglang/srt/speculative/dflash_worker_v2.py",
    "sglang/srt/utils/common.py",
)


def validate_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest["projects"] != {"vllm": "0.31.0", "sglang": "72730d6"}:
        raise ValueError("Expected vLLM 0.31.0 / SGLang 72730d6 manifest")
    for relative in REQUIRED + V2_FILES + SGLANG_JIT_FILES:
        data = (root / relative).read_bytes()
        entry = manifest["files"][relative]
        if entry != {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}:
            raise ValueError(f"Bundle hash/size mismatch: {relative}")
    # This establishes file identity within the supplied bundle, not authenticity
    # against a release tarball; the study documents the supplied source snapshot.
    return hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()


@functools.cache
def load_sglang_sampling(root):
    """Import the pinned production CUDA JIT entry without server initializers.

    Namespace parents persist for lazy JIT imports. Refuse another SGLang root
    rather than accidentally mixing pinned and installed implementations.
    Only two intact platform helpers are extracted from srt.utils.common: its
    unrelated server/video/HTTP imports are unnecessary for this CUDA build.
    No sampling or compiler implementation is stubbed.
    """
    root = Path(root).resolve()
    package_root = root / "sglang"
    existing = sys.modules.get("sglang")
    if existing is not None and getattr(existing, "_dflash_bundle_root", None) != package_root:
        raise RuntimeError("SGLang already imported from outside this comparator bundle")
    cache_dir = os.environ.get("SGLANG_JIT_CACHE_DIR")
    if not cache_dir:
        cache_dir = tempfile.mkdtemp(prefix="dflash-sglang-jit-")
        os.environ["SGLANG_JIT_CACHE_DIR"] = cache_dir
    Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
    for name in (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.speculative",
        "sglang.srt",
        "sglang.srt.utils",
    ):
        if name in sys.modules:
            continue
        directory = root.joinpath(*name.split("."))
        module = ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        module._dflash_bundle_root = package_root
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    common = load_definitions(
        package_root / "srt/utils/common.py",
        ("is_cuda", "get_cuda_version"),
        module_name="sglang.srt.utils.common",
        globals={"lru_cache": functools.lru_cache},
    )
    sys.modules["sglang.srt.utils"].common = common
    return importlib.import_module("sglang.kernels.ops.speculative.sampling")


def load_sglang_verify(root, probability_builder, tree=None):
    """Intact DFlash chain/RNG/gather and commit helpers, with server plumbing removed.

    The graph-pool borrowing context becomes a null context: benchmark graph
    capture owns its allocation pool. For verify-only, probability_builder
    returns prebuilt p. Pipeline passes the upstream probability builder.
    """
    root = Path(root)
    if tree is None:
        tree = load_sglang_sampling(root).tree_speculative_sampling_target_only
    verify = load_definitions(
        root / "sglang/srt/speculative/dflash_utils.py",
        ("_get_or_create_chain_verify_buffers", "compute_dflash_sampling_correct_drafts_and_bonus"),
        module_name="_k1_sglang_production_verify",
        globals={
            "_DFLASH_CHAIN_VERIFY_BUFFERS": {},
            "_DFLASH_SAMPLING_VERIFY_AVAILABLE": True,
            "borrow_graph_pool": lambda **kw: contextlib.nullcontext(),
            "build_speculative_verify_target_probs": probability_builder,
            "tree_speculative_sampling_target_only": tree,
        },
    )
    commit = load_definitions(
        root / "sglang/srt/speculative/dflash_worker_v2.py",
        ("_commit_accept",),
        module_name="_k1_sglang_production_commit",
    )
    return SimpleNamespace(
        verify=verify.compute_dflash_sampling_correct_drafts_and_bonus,
        commit=commit._commit_accept,
    )


class Capability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self):
        return f"{self.major}.{self.minor}"


def load_v2(root, apply_top_k_top_p, device="cuda"):
    """Load intact V2 dispatch, with a setup-only host/device buffer adapter.

    The server's UVA allocator is replaced by explicit CPU mirrors and device
    tensors. Admission/staged copies happen before timing. Optional processors
    are disabled; no sampling, filtering, RNG, or count kernel is substituted.
    """
    import numpy as np
    import torch
    from triton.language import math as tl_math
    from triton.language.extra import libdevice

    def source(key, names=None, **dependencies):
        relative = next(path for path in V2_FILES if path.endswith("/" + key + ".py"))
        path = Path(root) / relative
        if names is None:
            names = [
                name
                for node in ast.parse(path.read_text()).body
                for name in _definition_names(node)
            ]
        return load_definitions(
            path, names, module_name=f"_k1_sampling_v2_{key}", globals=dependencies
        )

    class HostDeviceBuffer:
        def __init__(self, size, *, dtype):
            self.cpu = torch.empty(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self.gpu = torch.empty(size, dtype=dtype, device=device)

        def copy_to_uva(self):
            self.gpu.copy_(self.cpu)

    gumbel = source("gumbel")
    # Keep even the inactive watermark dependency upstream, so Triton resolves
    # real JIT functions when scanning the rejection kernel's dependencies.
    constants = (
        "_UINT32_MASK",
        "_PHILOX_M0",
        "_PHILOX_M1",
        "_PHILOX_W0",
        "_PHILOX_W1",
        "_CONTEXT_DOMAIN",
        "_TOKEN_DOMAIN",
    )
    philox = source("philox", constants)
    watermark = source(
        "watermark",
        HAS_TRITON=True,
        tl_math=tl_math,
        gumbel_noised_argmax=gumbel.gumbel_noised_argmax,
        **{name + "_VALUE": getattr(philox, name) for name in constants},
    )
    min_p = source("min_p")
    states = source(
        "states",
        np=np,
        UvaBackedTensor=HostDeviceBuffer,
        apply_temperature=gumbel.apply_temperature,
        apply_min_p=min_p.apply_min_p,
        apply_top_k_top_p=apply_top_k_top_p,
    )
    thinking = source("thinking_budget", ("ThinkingBudgetState",), np=np)
    context = source("interface", ("LogitsContext",), dataclass=dataclasses.dataclass)
    output = source("output", ("SamplerOutput",), dataclass=dataclasses.dataclass)
    sampler = source("sampler", ("Sampler",), np=np, LogitsContext=context.LogitsContext)
    batch = source(
        "input_batch", ("_get_num_sampled_and_rejected_kernel", "get_num_sampled_and_rejected")
    )
    utils = source(
        "rejection_sampler_utils",
        tldevice=libdevice,
        gumbel_block_argmax=gumbel.gumbel_block_argmax,
        tl_rand32=gumbel.tl_rand32,
        philox_gumbel_block_argmax=watermark.philox_gumbel_block_argmax,
    )
    rejection = source(
        "rejection_sampler",
        np=np,
        NO_LOGPROBS=states.NO_LOGPROBS,
        PROCESSED_LOGPROBS_MODES=("processed_logits", "processed_logprobs"),
        rejection_sample=utils.rejection_sample,
        get_num_sampled_and_rejected=batch.get_num_sampled_and_rejected,
        SamplerOutput=output.SamplerOutput,
    )
    return SimpleNamespace(
        Sampler=sampler.Sampler,
        SamplingStates=states.SamplingStates,
        ThinkingBudgetState=thinking.ThinkingBudgetState,
        RejectionSampler=rejection.RejectionSampler,
        # Exposed for CPU dispatch-contract tests with recording launchers.
        rejection=rejection,
        utils=utils,
        batch=batch,
        gumbel=gumbel,
    )


def load(root):
    import torch
    import triton

    root = Path(root)
    logger = logging.getLogger("dflash.comparator")
    log = SimpleNamespace(
        info_once=logger.info, warning_once=logger.warning, debug_once=logger.debug
    )
    # vLLM's logger accepts scope=; the adapter drops this logging-only keyword.
    for name in ("info_once", "warning_once", "debug_once"):
        fn = getattr(log, name)
        setattr(log, name, lambda *a, _fn=fn, **kw: _fn(*a))
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_cpu=lambda: False,
        is_cuda_alike=lambda: True,
        num_compute_units=lambda index: (
            torch.cuda.get_device_properties(index).multi_processor_count
        ),
        get_device_capability=lambda: Capability(*torch.cuda.get_device_capability()),
    )
    modules = {}

    def source(key, relative, names=None, **dependencies):
        path = root / relative
        if names is None:
            names = [
                name
                for node in ast.parse(path.read_text()).body
                for name in _definition_names(node)
            ]
        module = load_definitions(
            path, names, module_name=f"_k1_sampling_{key}", globals=dependencies
        )
        modules[key] = module
        return module

    binding = source(
        "binding",
        REQUIRED[0],
        (
            "_LaunchBindingPlan",
            "_launch_binding_plan",
            "VllmTritonJitKernel._prepare_launch_kwargs",
        ),
        dataclass=dataclasses.dataclass,
        cache=functools.cache,
    )

    def dispatcher(*, kernel, warmup_inputs):
        # The only warmup shim: bypass vLLM's server registration/cache. Both
        # dispatch and argument binding still execute the pinned upstream code.
        def decorate(dispatch):
            signature = inspect.signature(dispatch)
            owner = SimpleNamespace(_kernel_arg_names=tuple(kernel.arg_names))

            @functools.wraps(dispatch)
            def launch(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                grid, config = dispatch(*args, **kwargs)
                config = binding._prepare_launch_kwargs(
                    owner, tuple(bound.arguments), tuple(bound.arguments.values()), config
                )
                return kernel[grid](**config)

            launch.register_warmup = lambda **kw: None
            return launch

        return decorate

    common = {
        "triton_kernel_dispatcher_with_warmup": dispatcher,
        "current_platform": platform,
        "gpu_sync_allowed": contextlib.nullcontext,
        "num_compute_units": platform.num_compute_units,
        "next_power_of_2": triton.next_power_of_2,
    }
    trunc = source("trunc", REQUIRED[1], **common)
    backend = source(
        "backend",
        REQUIRED[5],
        ("FlashInferBackend.supports_compute_capability",),
        DeviceCapability=Capability,
    )
    backend_type = type(
        "FlashInferBackend",
        (),
        {"supports_compute_capability": classmethod(backend.supports_compute_capability)},
    )
    envs = SimpleNamespace(
        VLLM_USE_FLASHINFER_SAMPLER=os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1") != "0",
        is_set=lambda name: name in os.environ,
    )
    top = source(
        "top",
        REQUIRED[2],
        (
            "_flashinfer_jit_unsupported_reason",
            "flashinfer_sampler_supported",
            "TopKTopPSampler",
            "apply_top_k_top_p",
            "empty_exponential_noise_like",
            "sample_with_exponential_noise",
            "random_sample",
            "flashinfer_sample",
        ),
        nn=torch.nn,
        envs=envs,
        logger=log,
        current_platform=platform,
        HAS_TRITON=True,
        PROCESSED_LOGPROBS_MODES=("processed_logits", "processed_logprobs"),
        apply_top_k_top_p_triton=trunc.apply_top_k_top_p_triton,
        register_top_k_top_p_warmups=lambda: None,
    )
    # Only this setup-time capability query needs a vLLM import. Restore any
    # preexisting module immediately; do not shadow the installed server package.
    key = "vllm.v1.attention.backends.flashinfer"
    shim = ModuleType(key)
    shim.FlashInferBackend = backend_type
    previous = sys.modules.get(key)
    sys.modules[key] = shim
    try:
        flashinfer_enabled = top.flashinfer_sampler_supported()
    finally:
        if previous is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = previous
    top.flashinfer_sampler_supported = lambda: flashinfer_enabled
    sampler = source(
        "sampler",
        REQUIRED[3],
        ("_SAMPLING_EPS", "Sampler"),
        nn=torch.nn,
        TopKTopPSampler=top.TopKTopPSampler,
        PIN_MEMORY=False,
        SamplerOutput=SimpleNamespace,
        LogprobsTensors=lambda *args: SimpleNamespace(logprobs=args[1]),
    )
    reject = source(
        "reject",
        REQUIRED[4],
        [
            name
            for node in ast.parse((root / REQUIRED[4]).read_text()).body
            for name in _definition_names(node)
            if name != "logger"
        ],
        **common,
        nn=torch.nn,
        replace=dataclasses.replace,
        PROCESSED_LOGPROBS_MODES=("processed_logits", "processed_logprobs"),
        apply_top_k_top_p=top.apply_top_k_top_p,
        SamplerOutput=SimpleNamespace,
    )
    # Optional FlashInfer for the SGLang KP probability builder, independent of
    # vLLM's hardware eligibility policy. No sgl_kernel dependency is required.
    try:
        from flashinfer.sampling import top_p_renorm_probs
    except (ImportError, OSError):
        top_p_renorm_probs = None
    sgl = source(
        "sgl_probs",
        REQUIRED[6],
        ("build_speculative_verify_target_probs",),
        F=torch.nn.functional,
        _dflash_top_p_renorm_prob=top_p_renorm_probs,
    )
    from benchmark.dflash_sampling.bench import load_reject_sampling

    chain = load_reject_sampling((str(root),))
    greedy = source(
        "sgl_greedy",
        REQUIRED[8],
        (
            "_dflash_accept_bonus_contig_kernel",
            "_pick_num_warps",
            "_is_row_major_contiguous_2d",
            "_compute_dflash_accept_bonus_triton_unchecked",
        ),
    )
    return SimpleNamespace(
        v2=load_v2(root, top.apply_top_k_top_p),
        sampler=sampler.Sampler(),
        reject=reject,
        top=top,
        sgl=sgl,
        sgl_verify_factory=lambda builder=None: load_sglang_verify(
            root, builder if builder is not None else sgl.build_speculative_verify_target_probs
        ),
        chain=chain,
        greedy=greedy,
        flashinfer_enabled=flashinfer_enabled,
        sgl_kp_available=top_p_renorm_probs is not None,
    )
