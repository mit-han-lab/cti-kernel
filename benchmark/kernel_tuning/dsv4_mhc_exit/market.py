# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Small, lazy adapters. Use in a standalone benchmark, not a framework server."""

import importlib
import importlib.util
import os
import re
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from .reference import exit_torch_bf16

NAMES = ("torch", "torch_compile", "vllm", "sglang", "flashinfer", "cti")


class Unavailable(RuntimeError):
    pass


def failure_reason(exc, stage):
    # Third-party exception text can contain environment values or absolute paths.
    # Only our own fixed diagnostics and exception *types* enter reports.
    if isinstance(exc, Unavailable):
        return str(exc)
    if isinstance(exc, ModuleNotFoundError):
        dependency = (exc.name or "").split(".")[0]
        if dependency in {
            "tilelang",
            "triton",
            "sgl_kernel",
            "flashinfer",
            "deep_gemm",
            "cutlass",
            "tvm_ffi",
        }:
            return f"{stage}: missing {dependency}"
    return f"{stage} failed ({type(exc).__name__}); check runtime dependencies"


def _flashinfer_norm_backend(api, device):
    # flashinfer/norm/__init__.py:60-101,188-198 (0.7.1).
    return (
        "FlashInfer JIT CUDA" if api._use_cuda_norm(torch.device(device)) else "FlashInfer CuTe DSL"
    )


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _packages(root, names):
    """Bypass package initializers; all numerical code is loaded from the given tree."""
    for name in names:
        if name in sys.modules:
            continue
        path = root.joinpath(*name.split(".")[1:])
        if not path.is_dir():
            raise Unavailable("source root lacks a required package directory")
        module = _module(name, __path__=[str(path)])
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)


def _vllm(root, T):
    if root is None:
        raise Unavailable("supply --vllm-root (directory containing vllm or vllm itself)")
    root = Path(root)
    if (root / "vllm").is_dir():
        root /= "vllm"
    _packages(
        root,
        (
            "vllm",
            "vllm.utils",
            "vllm.model_executor",
            "vllm.model_executor.kernels",
            "vllm.model_executor.kernels.mhc",
            "vllm.model_executor.warmup",
            "vllm.third_party",
        ),
    )
    # Only platform queries/registration are stubbed, never a numerical kernel.
    # vllm/platforms/cuda.py:755; tilelang.py:5-6; tilelang_utils/__init__.py:11-12.
    _module(
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_cuda=lambda: True,
            is_rocm=lambda: False,
            is_arch_support_pdl=lambda: torch.cuda.get_device_capability()[0] >= 9,
        ),
    )
    _module("vllm.utils.torch_utils", direct_register_custom_op=lambda **kwargs: None)
    _module(
        "vllm.utils.import_utils",
        has_tilelang=lambda: importlib.util.find_spec("tilelang") is not None,
    )
    if T > 32:
        # Match optional external-then-vendored selection, without loading the server.
        # vllm/utils/import_utils.py:500; utils/deep_gemm.py:274-294,931-953.
        name = (
            "deep_gemm" if importlib.util.find_spec("deep_gemm") else "vllm.third_party.deep_gemm"
        )
        present = importlib.util.find_spec(name) is not None
        enabled = os.environ.get("VLLM_USE_DEEP_GEMM", "1") == "1" and present
        dg = None
        if enabled:
            try:
                dg = importlib.import_module(name)
            except ImportError:
                if name != "deep_gemm":
                    raise
                dg = importlib.import_module("vllm.third_party.deep_gemm")
        if dg is not None:
            if hasattr(dg, "set_pdl"):
                dg.set_pdl(True)
            gemm = dg.tf32_hc_prenorm_gemm
        else:

            def gemm(*args):
                raise Unavailable("DeepGEMM is unavailable")

        _module(
            "vllm.utils.deep_gemm",
            is_deep_gemm_supported=lambda: enabled,
            tf32_hc_prenorm_gemm=gemm,
        )
    return importlib.import_module("vllm.model_executor.kernels.mhc.tilelang")


def _sglang(root):
    if root is None:
        raise Unavailable("supply --sglang-python (directory containing sglang)")
    root = Path(root) / "sglang"
    _packages(
        root,
        (
            "sglang",
            "sglang.kernels",
            "sglang.kernels.ops",
            "sglang.kernels.ops.layernorm",
            "sglang.kernels.jit",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.utils",
            "sglang.srt.layers.attention",
            "sglang.srt.layers.attention.dsa",
            "sglang.srt.layers.utils",
            "sglang.srt.distributed",
            "sglang.srt.distributed.device_communicators",
        ),
    )
    # mhc.py:12-22 imports runtime policy for other paths. This adapter fixes the
    # default CUDA, no-CP, nonsymmetric standalone path; it loads real kernels.
    _module("sglang.kernels.jit.utils", is_arch_support_pdl=lambda: True)
    _module(
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        use_symmetric_memory=lambda *args, **kwargs: nullcontext(),
    )
    _module("sglang.srt.layers.attention.dsa.utils", is_dsa_prefill_cp_interleave=lambda: False)
    _module("sglang.srt.layers.dp_attention", is_allocation_symmetric=lambda: False)
    _module("sglang.srt.layers.utils.common", strict_contiguous=lambda x: x.contiguous())
    _module(
        "sglang.srt.runtime_context",
        get_parallel=lambda: SimpleNamespace(tp_group=None),
        get_platform=lambda: SimpleNamespace(is_blackwell=True, is_sm100=False, is_sm90=False),
    )
    sys.modules["sglang.srt.utils"].is_hip = lambda: False
    sys.modules["sglang.srt.utils"].is_gfx95_supported = lambda: False
    _module("sglang.srt.utils.common", is_gfx1250_supported=lambda: False)
    flags = {
        key: SimpleNamespace(get=lambda value=value: value)
        for key, value in (
            ("SGLANG_USE_AITER", False),
            ("SGLANG_OPT_DEEPGEMM_HC_PRENORM", True),
            ("SGLANG_OPT_USE_TILELANG_MHC_PRE", True),
            ("SGLANG_OPT_USE_TILELANG_MHC_POST", True),
        )
    }
    _module("sglang.srt.environ", envs=SimpleNamespace(**flags))
    return importlib.import_module("sglang.kernels.ops.layernorm.mhc")


@dataclass
class Provider:
    name: str
    vllm_root: Path | None = None
    sglang_python: Path | None = None
    status: str = "unprepared"
    metadata: dict = field(default_factory=dict)

    def prepare(self, cfg, weights, T, device):
        """Return callable(x, R, post, comb), or None with an unavailable status.

        The callable returns (residual, post, comb, out). Output buffers may be
        reused by the next invocation; consume correctness results immediately.
        """
        try:
            if self.name != "torch" and torch.device(device).type != "cuda":
                raise Unavailable("requires CUDA; CPU preparation does not import GPU libraries")
            call = self._prepare(cfg, weights, T, device)
            self.status = "available"
            return call
        except Exception as exc:  # noqa: BLE001 - optional providers report unavailable
            self.status = "unavailable: " + failure_reason(exc, "prepare")
            return None

    def _prepare(self, cfg, w, T, device):
        if self.name in ("torch", "torch_compile"):

            def call(x, R, post, comb):
                result = exit_torch_bf16(cfg, w, x, R, post, comb)
                return tuple(result[key] for key in ("residual", "post", "comb", "out"))

            self.metadata = {"kernels_per_exit": None, "backend": self.name}
            if self.name == "torch_compile":
                if not hasattr(torch, "compile"):
                    raise Unavailable("torch.compile is not installed")
                return torch.compile(call, fullgraph=True)
            return call
        if self.name == "cti":
            from ctikernels.mhc import MhcExit, support

            reason = support(device, hidden=cfg.hidden, hc_mult=cfg.mult)
            if reason is not None:
                raise Unavailable(reason)
            op = MhcExit(
                device,
                hidden=cfg.hidden,
                rms_eps=cfg.rms_eps,
                hc_eps=cfg.hc_eps,
                norm_eps=cfg.norm_eps,
                post_mult=cfg.post_mult,
                sinkhorn_iters=cfg.sinkhorn_iters,
            )
            op.prepare([T])
            outputs = {
                "residual_out": torch.empty(
                    (T, 4, cfg.hidden), device=device, dtype=torch.bfloat16
                ),
                "post_out": torch.empty((T, 4), device=device, dtype=torch.float32),
                "comb_out": torch.empty((T, 4, 4), device=device, dtype=torch.float32),
                "out": torch.empty((T, cfg.hidden), device=device, dtype=torch.bfloat16),
            }
            self.metadata = {"kernels_per_exit": 1, "backend": "CuTe DSL", "norm_fused": True}

            def call(x, R, post, comb):
                return op(x, R, post, comb, **w, **outputs)

            # The closure owns the instance/workspace for the captured graph's lifetime.
            return call
        if cfg.hidden != 4096 or cfg.mult != 4:
            raise Unavailable("market adapters cover H=4096 and mult=4 only")
        if torch.cuda.get_device_capability(device)[0] != 12:
            raise Unavailable("these dispatch adapters require sm_12x")
        if self.name == "vllm":
            api = _vllm(self.vllm_root, T)
            self.metadata = {
                "source_version": "0.31.0",
                "kernels_per_exit": 2 if T <= 32 else 3,
                "effective_tile_n": 2 if T < 16 else 6 if T <= 32 else None,
                "effective_splits": 8 if T <= 32 else None,
                "norm_fused": True,
            }

            def call(x, R, post, comb):
                # vllm/models/deepseek_v4/nvidia/model.py:1366-1418.
                r, p, c, out = api.mhc_fused_post_pre_tilelang(
                    x,
                    R,
                    post,
                    comb,
                    w["fn"],
                    w["scale"],
                    w["base"],
                    cfg.rms_eps,
                    cfg.hc_eps,
                    cfg.hc_eps,
                    cfg.post_mult,
                    cfg.sinkhorn_iters,
                    n_splits=1,
                    tile_n=1,
                    norm_weight=w["norm_weight"],
                    norm_eps=cfg.norm_eps,
                )
                return r, p.reshape(T, cfg.mult), c, out

            return call
        if self.name == "sglang":
            if cfg.post_mult != 2:
                raise Unavailable("SGLang Sinkhorn fixes post_mult=2")
            api = _sglang(self.sglang_python)
            rmsnorm = importlib.import_module("sgl_kernel").rmsnorm
            elementwise = importlib.import_module("sgl_kernel.elementwise")
            norm_backend = "sgl_kernel AOT CUDA"
            if getattr(elementwise, "_has_flashinfer", False):
                norm_backend = _flashinfer_norm_backend(elementwise._flashinfer_norm, device)
            norm_out = torch.empty((T, cfg.hidden), dtype=torch.bfloat16, device=device)
            self.metadata = {
                "source_version": "72730d6",
                "kernels_per_exit": 5,
                "backend": "TileLang post + Triton stats/Sinkhorn/combine + separate RMSNorm",
                "norm_backend": norm_backend,
                "norm_fused": False,
            }

            def call(x, R, post, comb):
                # sglang/srt/models/deepseek_v4_mhc.py:259-367,394-445,452-539.
                r = api.mhc_post(x, R, post, comb)
                pre, p, c = api.hc_mix_stats_sinkhorn(
                    r.flatten(1),
                    w["fn"],
                    w["scale"],
                    w["base"],
                    cfg.mult,
                    cfg.sinkhorn_iters,
                    cfg.rms_eps,
                    cfg.hc_eps,
                )
                y = api.hc_combine(r.flatten(1), pre, cfg.mult, torch.bfloat16)
                rmsnorm(y, w["norm_weight"], cfg.norm_eps, out=norm_out)
                return r, p, c, norm_out

            return call
        if self.name == "flashinfer":
            package = importlib.import_module("flashinfer")
            version = package.__version__
            self.metadata = {
                "installed_version": version
                if re.fullmatch(r"\d+\.\d+\.\d+(?:\.post\d+|rc\d+)?", version)
                else "unrecognized"
            }
            if version != "0.7.1":
                raise Unavailable("requires FlashInfer 0.7.1 (installed version differs)")
            api = importlib.import_module("flashinfer.mhc")
            norm_api = importlib.import_module("flashinfer.norm")
            norm = norm_api.rmsnorm
            dot = torch.empty((T, 24), dtype=torch.float32, device=device)
            out = torch.empty((T, cfg.hidden), dtype=torch.bfloat16, device=device)
            self.metadata = {
                "version": version,
                "kernels_per_exit": 5,
                "backend": "CUDA post + torch cast/GEMM + CUDA pre + separate RMSNorm",
                "norm_backend": _flashinfer_norm_backend(norm_api, device),
                "norm_fused": False,
            }

            def call(x, R, post, comb):
                # flashinfer/mhc.py:76,334: prenorm computes sqrsum, not projection or final norm.
                r = api.mhc_post(x, R, post, comb)
                torch.mm(r.flatten(1).float(), w["fn"].T, out=dot)
                p, c, y = api.mhc_pre_big_fuse_with_prenorm(
                    dot,
                    r,
                    w["scale"],
                    w["base"],
                    rms_eps=cfg.rms_eps,
                    mhc_pre_eps=cfg.hc_eps,
                    mhc_sinkhorn_eps=cfg.hc_eps,
                    mhc_post_mult_value=cfg.post_mult,
                    sinkhorn_repeat=cfg.sinkhorn_iters,
                )
                norm(y, w["norm_weight"], cfg.norm_eps, out=out)
                return r, p.reshape(T, cfg.mult), c, out

            return call
        raise Unavailable("unknown provider")
