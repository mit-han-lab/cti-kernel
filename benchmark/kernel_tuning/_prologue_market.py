# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# ruff: noqa: BLE001
# External dependency failures become explicit NA records without raw diagnostics.

"""Runtime market adapters. No third-party kernel implementation is vendored.

Dispatch audit: SGLang 72730d6 qwen3.forward_prepare_native -> apply_qk_norm
-> fused_inplace_qknorm; RotaryEmbedding.forward_cuda -> non-fused JIT RoPE;
MHATokenToKVPool._store_kv_layer -> _set_kv_buffer_impl -> JIT store_cache.
vLLM 0.31.0 Qwen3Attention.forward uses native RMSNorm and RotaryEmbedding
under Inductor (custom_ops=none). QK-norm/RoPE fusion defaults off, and
fuse_rope_kvcache is ROCm-only. FlashAttentionImpl.do_kv_cache_update calls
reshape_and_cache_flash behind opaque unified_kv_cache_update: visible as a
custom op to the outer graph, but not fused into the norm/RoPE Inductor region.
This benchmark's norm/RoPE is an independent surrogate under generic Inductor,
not vLLM's compiled graph; only the wheel's KV-write operation is real vLLM.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import importlib.machinery
import os
import platform
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import torch

_REGISTRATIONS = []
SGLANG_REVISION = "72730d6360a5894b167500d5d1ad82c34c6bb116"
SGLANG_BLOBS = {
    "sglang/srt/mem_cache/memory_pool.py": "5cc72238dc95956dc5919c0fb8ea6ac85c5c3972",
    "sglang/kernels/jit/csrc/elementwise/kvcache.cuh": "b17168466159666db0a6848d886e6590536c9c68",
    "sglang/kernels/jit/csrc/elementwise/pos_enc.cuh": "13880865f4d03482031b74ee463751cb9cbebd23",
    "sglang/kernels/jit/csrc/elementwise/qknorm.cuh": "b2d862d561f2ba113a0b9dd5fd7c6a3263480b23",
    "sglang/kernels/jit/csrc/elementwise/rope.cuh": "8ec1c256460ac08a0202028a5b922e6cb5fc7bb3",
    "sglang/kernels/ops/attention/rope.py": "a5f6be632f53cf210616dff8346943867a0c580d",
    "sglang/kernels/ops/kvcache/kvcache.py": "bb9814361b4c09fcdcf0c83fdfc2b35265242be6",
    "sglang/kernels/ops/layernorm/norm.py": "c3a8e948f32a9916ba4835aff0d845426168eb99",
}
VLLM_NORM_ROPE = (
    "surrogate: independent native expression of vLLM 0.31.0 RMSNorm.forward_native + "
    "RotaryEmbedding (NeoX) under generic Inductor; not vLLM's compiled graph"
)


def _register_metadata(op_name, op_func, mutates_args, fake_impl=None, target_lib=None):
    """Serving registration shim; eager wrappers still invoke the real function."""
    if target_lib is not None:
        raise RuntimeError("Unexpected custom-op library")
    definition = torch.library.custom_op(f"sglang::{op_name}", op_func, mutates_args=mutates_args)
    if fake_impl is not None:
        definition.register_fake(fake_impl)
    _REGISTRATIONS.append(definition)


@contextlib.contextmanager
def quiet_external():
    """Discard dependency/compiler diagnostics; never persist raw logs or URLs."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(os.devnull, "w") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, fd in zip((1, 2), saved, strict=True):
                os.dup2(fd, target)
                os.close(fd)


def _namespace(name: str, root: Path):
    """Bypass only serving package initializers, retaining real submodules."""
    location = root.joinpath(*name.split("."))
    if name in sys.modules:
        existing = sys.modules[name]
        if str(location) not in getattr(existing, "__path__", []):
            raise RuntimeError("SGLang revision already imported")
        return
    package = types.ModuleType(name)
    package.__path__ = [str(location)]
    package.__package__ = name
    package.__file__ = str(location / "__init__.py")
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[name] = package


def sglang_modules(root: Path):
    root = root.resolve()
    if not (root / "sglang/kernels/ops/layernorm/norm.py").is_file():
        raise FileNotFoundError("SGLang kernel tree missing")
    sys.path.insert(0, str(root))
    # The top-level text-generation API and utils/__init__ import serving-only
    # dependencies. No kernel, JIT, decorator, or compiler module is stubbed.
    for name in ("sglang", "sglang.srt", "sglang.srt.utils"):
        _namespace(name, root)
    common_name = "sglang.srt.utils.common"
    if common_name not in sys.modules:
        common = types.ModuleType(common_name)
        # Only a hardware predicate and op registration metadata are needed;
        # the real custom_op decorator calls the real op_func in eager mode.
        common.is_cuda = lambda: torch.cuda.is_available() and torch.version.cuda is not None
        common.direct_register_custom_op = _register_metadata
        common.__spec__ = importlib.machinery.ModuleSpec(common_name, loader=None)
        sys.modules[common_name] = common
    modules = tuple(
        importlib.import_module(f"sglang.kernels.ops.{suffix}")
        for suffix in ("layernorm.norm", "attention.rope", "kvcache.kvcache")
    )
    for module in modules:
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("SGLang kernel imported from another tree")
    hashes, mismatched = {}, []
    for filename, expected in SGLANG_BLOBS.items():
        try:
            data = (root / filename).read_bytes()
            actual = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        except OSError:
            actual = None
        hashes[Path(filename).name] = actual
        if actual != expected:
            mismatched.append(Path(filename).name)
    metadata = {
        "revision": SGLANG_REVISION if not mismatched else "unverified",
        "revision_verified": not mismatched,
        "revision_mismatches": mismatched,
        "entrypoints": [
            "fused_inplace_qknorm",
            "apply_rope_with_cos_sin_cache_inplace",
            "store_cache",
        ],
        "git_blob_sha1": hashes,
        "off_path_stubs": [
            "sglang",
            "sglang.srt",
            "sglang.srt.utils",
            "sglang.srt.utils.common.is_cuda",
            "sglang.srt.utils.common.direct_register_custom_op (metadata only)",
        ],
        "fused_rope_kv": False,
        "reserved_slot": 0,
    }
    revision_file = root.parent / "REVISION"
    if revision_file.is_file():
        metadata["revision_file_matches"] = revision_file.read_bytes().startswith(
            SGLANG_REVISION.encode("ascii")
        )
    return modules, metadata


def build_sglang(modules):
    """Compile/load only, with an explicit sm_120a target if no GPU is visible."""
    from sglang.kernels.jit.utils import override_jit_cuda_arch

    norm, rope, cache = modules
    context = (
        contextlib.nullcontext() if torch.cuda.is_available() else override_jit_cuda_arch(12, 0)
    )
    with context:
        norm._jit_qknorm_module(128, torch.bfloat16)
        rope._jit_fused_rope_module(True, 128, torch.bfloat16, torch.bfloat16)
        cache._jit_kvcache_module(2048, 2048, 0)
    return {
        "status": "built",
        "launched": False,
        "target": "device" if torch.cuda.is_available() else "sm_120a",
    }


def ensure_vllm(cache: Path):
    """Extract the cache-op library and hash Python sources in memory, without vendoring."""
    cache.mkdir(parents=True, exist_ok=True)
    arch = platform.machine()
    wheels = sorted(cache.glob(f"vllm-0.31.0-*{arch}.whl"))
    if not wheels:
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--disable-pip-version-check",
            "--no-input",
            "-d",
            str(cache),
            "vllm==0.31.0",
        ]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900
        )
        if result.returncode:
            raise RuntimeError("vLLM wheel download failed")
        wheels = sorted(cache.glob(f"vllm-0.31.0-*{arch}.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one matching vLLM wheel")
    wheel = wheels[0]
    libraries = []
    with zipfile.ZipFile(wheel) as archive:
        sources = ["vllm/model_executor/layers/layernorm.py"]
        rotary = []
        for name in sorted(archive.namelist()):
            if name.startswith("vllm/model_executor/layers/rotary_embedding/") and name.endswith(
                ".py"
            ):
                tree = ast.parse(archive.read(name))
                if any(
                    isinstance(node, ast.ClassDef) and node.name == "RotaryEmbedding"
                    for node in tree.body
                ):
                    rotary.append(name)
        if not rotary:
            rotary = ["vllm/model_executor/layers/rotary_embedding.py"]
        if len(rotary) != 1:
            raise RuntimeError("Expected one RotaryEmbedding base module")
        sources.extend(rotary)
        source_sha256 = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in sources}
        members = [
            n
            for n in archive.namelist()
            if Path(n).parent.as_posix() == "vllm"
            and Path(n).name.startswith("_C")
            and n.endswith(".so")
        ]
        # Prefer the stable libtorch ABI; avoid loading unrelated registries.
        members.sort(key=lambda n: ("stable_libtorch" not in n, n))
        for member in members:
            with archive.open(member) as src:
                data = src.read()
            if b"reshape_and_cache_flash" not in data:
                continue
            destination = cache / Path(member).name
            if not destination.is_file() or destination.stat().st_size != len(data):
                temporary = destination.with_suffix(f".tmp-{os.getpid()}")
                temporary.write_bytes(data)
                temporary.replace(destination)
            libraries.append((destination, hashlib.sha256(data).hexdigest()))
    if not libraries:
        raise RuntimeError("Cache-op symbol absent from wheel")
    info = {
        "version": "0.31.0",
        "wheel": wheel.name,
        "architecture": arch,
        "symbol_check": "reshape_and_cache_flash present",
        "libraries": [{"name": p.name, "sha256": h} for p, h in libraries],
        "source_sha256": source_sha256,
        "norm_rope": VLLM_NORM_ROPE,
    }
    return libraries, info


def load_vllm_cache(cache: Path):
    info = {"version": "0.31.0", "vllm_kv_write": "stand-in", "norm_rope": VLLM_NORM_ROPE}
    try:
        libraries, extracted = ensure_vllm(cache)
        info.update(extracted)
    except Exception as error:
        info.update(failure_phase="download/extract", error_type=type(error).__name__)
        return None, info
    for library, _ in libraries:
        try:
            torch.ops.load_library(str(library))
            operation = torch.ops._C_cache_ops.reshape_and_cache_flash
            info.update(vllm_kv_write="reshape_and_cache_flash", loaded_library=library.name)
            return operation, info
        except Exception as error:
            info.update(failure_phase="load", error_type=type(error).__name__)
    return None, info


def native_vllm(qkv, qw, kw, table, positions):
    """Independent norm/RoPE surrogate, including vLLM's native BF16 rounds.

    RMSNorm's native IR casts normalized x to weight.dtype before multiplying.
    RotaryEmbedding matches its table to query.dtype once during warmup. The
    BF16 products below intentionally retain native pointwise rounding semantics;
    Generic Inductor optimizes these expressions outside vLLM's serving graph.
    """
    q, k, _ = qkv.view(-1, 48, 128).split((32, 8, 8), dim=1)
    normalized = []
    for heads, weight in ((q, qw), (k, kw)):
        fp = heads.float()
        inverse = (fp.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        normalized.append((fp * inverse).to(weight.dtype) * weight)
    coeff = table[positions].unsqueeze(1)
    c, s = coeff[..., :64], coeff[..., 64:]
    result = []
    for heads in normalized:
        left, right = heads[..., :64], heads[..., 64:]
        result.append(torch.cat((left * c - right * s, right * c + left * s), dim=-1))
    return result[0], result[1]


def compile_vllm():
    # Independent code objects avoid static/dynamic specialization cache sharing.
    static_fn = types.FunctionType(native_vllm.__code__.replace(co_name="vllm_static"), globals())
    dynamic_fn = types.FunctionType(native_vllm.__code__.replace(co_name="vllm_dynamic"), globals())
    return {
        "vllm_static": torch.compile(
            static_fn, backend="inductor", mode="default", dynamic=False, fullgraph=True
        ),
        "vllm_dynamic": torch.compile(
            dynamic_fn, backend="inductor", mode="default", dynamic=None, fullgraph=True
        ),
    }


SGLANG_ENGINE_BLOBS = {
    "sglang/srt/models/qwen3_5.py": "def6a402d5b4013502bc109dad70e6b0c37ba534",
    "sglang/srt/layers/layernorm.py": "6068e8f2095f09b0841913fec7fe07a88b7020ef",
    "sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py": (
        "3648bd557ce780bbb46db4d836627424df241ab6"
    ),
}


def safe_failure(error, phase):
    """Identify the exact failing source location without exposing exception text."""
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append({"file": Path(code.co_filename).name, "line": traceback.tb_lineno})
        traceback = traceback.tb_next
    result = {"phase": phase, "exception": type(error).__name__, "frames": frames}
    if isinstance(error, ModuleNotFoundError):
        # Module names are identifiers, never arbitrary diagnostics or URLs.
        name = error.name
        if name and all(part.isidentifier() for part in name.split(".")):
            result["missing_module"] = name
    return result


def sglang_engine_modules(root, *, gated):
    modules, info = sglang_modules(root)
    for name, expected in SGLANG_ENGINE_BLOBS.items():
        data = (root / name).read_bytes()
        actual = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        info["git_blob_sha1"][name] = actual
        if actual != expected:
            info["revision_mismatches"].append(name)
    info["revision_verified"] = not info["revision_mismatches"]
    if not info["revision_verified"]:
        raise RuntimeError("SGLang engine source revision mismatch")
    if gated:
        extra = importlib.import_module("sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate")
        norm = extra.fused_qk_gemma_rmsnorm_rope_gate
        info["entrypoints"] = ["fused_qk_gemma_rmsnorm_rope_gate", "store_cache"]
    else:
        # qwen3_5._apply_qk_norm -> GemmaRMSNorm._forward_impl on CUDA.
        from sgl_kernel import gemma_rmsnorm

        norm = gemma_rmsnorm
        info["entrypoints"] = ["gemma_rmsnorm (Q and K)", *info["entrypoints"][1:]]
    return (*modules, norm), info


def vllm_source_modules(wheel):
    """Zipimport unchanged wheel sources; only off-path serving imports are shimmed.

    Native IR is selected through its public configuration APIs. This exposes the
    real native body to generic Inductor, corresponding to vLLM's native IR lowering.
    No norm/rotary computation or kernel decorator is replaced by a stub.
    """
    from collections import Counter

    root = Path(wheel).resolve()
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.rotary_embedding",
        "vllm.model_executor.determinism",
        "vllm.ir",
        "vllm.ir.ops",
        "vllm.utils",
    ):
        _namespace(name, root)
    stubs = []

    def stub(name, **attributes):
        if name in sys.modules:
            raise RuntimeError("vLLM dependency already imported")
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, module)
        stubs.append(name)
        return module

    class SilentLogger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    config = types.SimpleNamespace(
        custom_ops=["none"], enabled_custom_ops=Counter(), disabled_custom_ops=Counter()
    )

    def unreachable(*args, **kwargs):
        raise RuntimeError("Off-path serving shim invoked")

    stub("vllm.config", get_cached_compilation_config=lambda: config)
    stub("vllm.config.compilation", CompilationMode=types.SimpleNamespace(NONE=0))
    stub("vllm.logger", init_logger=lambda name: SilentLogger())
    stub("vllm.logging_utils", lazy=lambda function: None, tensors_str_no_data=unreachable)
    stub("vllm.model_executor.utils", maybe_disable_graph_partition=unreachable)
    stub("vllm.platforms", current_platform=types.SimpleNamespace(is_cpu=lambda: True))
    stub("vllm.envs", VLLM_BATCH_INVARIANT=False)
    stub("vllm.kernels")  # Other IR providers are not used by native Inductor lowering.
    stub(
        "vllm.model_executor.determinism.batch_invariant",
        rms_norm_batch_invariant=unreachable,
    )
    stub(
        "vllm._aiter_ops",
        rocm_aiter_ops=types.SimpleNamespace(is_triton_rotary_embed_enabled=lambda: False),
    )

    # Only common.py's unused FlashInfer registration; fail if its body is called.
    def register_unused(op_name, op_func, mutates_args):
        if op_name != "flashinfer_rotary_embedding":
            raise RuntimeError("Unexpected off-path op registration")

    stub("vllm.utils.torch_utils", direct_register_custom_op=register_unused)
    ir_op = importlib.import_module("vllm.ir.op")
    ir_norm = importlib.import_module("vllm.ir.ops.layernorm")
    ir_norm.rms_norm.set_default(["native"])
    ir_op.set_default_torch_wrap(False)
    sys.modules["vllm.ir"].ops = sys.modules["vllm.ir.ops"]
    sys.modules["vllm.ir.ops"].rms_norm = ir_norm.rms_norm
    norm = importlib.import_module("vllm.model_executor.layers.layernorm")
    rope = importlib.import_module("vllm.model_executor.layers.rotary_embedding.base")
    imported = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("vllm.") and getattr(module, "__file__", None)
    }
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in (
            "vllm/model_executor/models/qwen3.py",
            "vllm/model_executor/models/qwen3_next.py",
            "vllm/model_executor/models/qwen3_5.py",
            "vllm/model_executor/layers/attention/attention.py",
            "vllm/model_executor/layers/rotary_embedding/mrope.py",
            "vllm/v1/attention/backends/flash_attn.py",
            "vllm/utils/torch_utils.py",
            "vllm/_custom_ops.py",
            "vllm/config/compilation.py",
            "vllm/config/vllm.py",
            "vllm/platforms/cuda.py",
        ):
            hashes[member] = hashlib.sha256(archive.read(member)).hexdigest()
        for module in imported.values():
            filename = str(module.__file__)
            if filename.startswith(str(root) + "/") and filename.endswith(".py"):
                member = filename[len(str(root)) + 1 :]
                hashes[member] = hashlib.sha256(archive.read(member)).hexdigest()
    return (norm, rope), {
        "norm_rope": "real wheel forward_native + real native IR under Inductor",
        "off_path_stubs": stubs,
        "source_sha256": hashes,
        "ir": "native; set_default_torch_wrap(False), set_default(['native'])",
    }


def vllm_fused_module():
    """Import the wheel's default Qwen3.5 gated kernel, using real Triton."""
    import triton
    import triton.language as tl

    # Bypass vLLM's unrelated descriptor/gluon compatibility helpers.
    name = "vllm.triton_utils"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module.triton, module.tl = triton, tl
        sys.modules[name] = module
    return importlib.import_module("vllm.model_executor.layers.fused_qk_norm_rope")
