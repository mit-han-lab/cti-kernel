# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Prebuild every FlashInfer JIT module ctikernels can select on a device class.

    python -m ctikernels._runtime.prewarm --device-class sm120 [--jobs N] [--cubins]
    ctikernels.prewarm_flashinfer(device=None, modules="selected")  # at engine startup

A cold FlashInfer cache builds some modules on first call for 14-20 minutes; run this once per
image or node instead. Modules land in FlashInfer's cache for the class's architecture
(``FLASHINFER_CUDA_ARCH_LIST``), the directory a node of that class reads at run time, so the
build host needs no GPU. Idempotent: built modules are skipped. ``--cubins`` also downloads
FlashInfer's prebuilt TRT-LLM gen cubins, which sm100-family providers otherwise fetch on first
call.
"""

import argparse
import importlib
import json
import os
import sys
import time

DEVICE_CLASSES = {"sm100": "10.0", "sm103": "10.3", "sm120": "12.0", "sm121": "12.1"}
SM100, SM120 = ("sm100", "sm103"), ("sm120", "sm121")

# (provider, device classes, FlashInfer JIT generator, its arguments). Provider owners keep this
# current. Not listed: attention prefill/decode modules (keyed by head dim and dtype: they build
# at the engine's own warm-up) and CuTe DSL kernels (compiled per shape at warm-up).
ALL = SM100 + SM120
_GEMM = "flashinfer.jit.gemm.core:"
MODULES = (
    ("moe W4A8 experts (CUTLASS)", SM120, "flashinfer.jit.fused_moe:gen_cutlass_fused_moe_sm120_module", {}),
    ("moe W4A8 experts (TRT-LLM gen)", SM100, "flashinfer.jit.fused_moe:gen_trtllm_gen_fused_moe_sm100_module", {}),
    ("linear MXFP8 quantize", ALL, "flashinfer.jit.fp8_quantization:gen_mxfp8_quantization_sm100_module", {}),
    ("linear grouped block-FP8 GEMM", SM100, _GEMM + "gen_gemm_sm100_module", {}),
    ("linear block-FP8 / static FP8 GEMM (CUTLASS groupwise)", SM120, _GEMM + "gen_gemm_sm120_module", {}),
    ("linear static FP8 GEMM (CUTLASS)", SM100, _GEMM + "gen_gemm_sm100_module_cutlass_fp8", {}),
    ("linear FP8 bmm (cuBLAS)", ALL, _GEMM + "gen_gemm_module", {}),
    ("linear NVFP4 GEMM (CUTLASS)", SM120, _GEMM + "gen_gemm_sm120_module_cutlass_fp4", {}),
    ("attention DSV4 top-k page table", ALL, "flashinfer.jit.topk:gen_topk_module", {}),
    ("attention DSV4 sparse decode (TRT-LLM gen)", SM100, "flashinfer.jit.attention:gen_trtllm_gen_fmha_module", {}),
    ("attention DSV4 sparse decode (sparse MLA)", SM120, "flashinfer.jit.mla:gen_sparse_mla_sm120_module", {}),
    ("activation SiLU and mul", ALL, "flashinfer.jit.activation:gen_act_and_mul_module", {"act_func_name": "silu"}),
    ("activation GELU and mul", ALL, "flashinfer.jit.activation:gen_act_and_mul_module", {"act_func_name": "gelu"}),
    ("activation GELU-tanh and mul", ALL, "flashinfer.jit.activation:gen_act_and_mul_module", {"act_func_name": "gelu_tanh"}),
    ("norm RMSNorm", ALL, "flashinfer.jit.norm:gen_norm_module", {}),
    ("rotary", ALL, "flashinfer.jit.rope:gen_rope_module", {}),
    ("sampling", ALL, "flashinfer.jit.sampling:gen_sampling_module", {}),
)  # fmt: skip

_GIB_PER_JOB = 24  # gemm_sm100 ran out of 160 GiB at one nvcc job per core (16), not at 4


def default_jobs():
    """nvcc jobs bounded by cores and by available memory."""
    available = 0
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) // (1 << 20)
    except OSError:
        pass
    return max(1, min(os.cpu_count() or 1, available // _GIB_PER_JOB or 1))


def prewarm(device_class, *, jobs=None, cubins=False):
    """Build the class's modules for its arch (build hosts need no GPU); one row per module."""
    arch = DEVICE_CLASSES[device_class]
    if os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", arch).rstrip("af") != arch:
        raise ValueError(f"FLASHINFER_CUDA_ARCH_LIST is set to another arch than {arch}")
    if "flashinfer" in sys.modules:
        raise RuntimeError(
            "prewarm must run before FlashInfer is imported (its cache dir is fixed then)"
        )
    return _build(device_class, None, jobs, cubins)


def prewarm_flashinfer(device=None, modules="selected"):
    """Startup entry: build, in this process, the FlashInfer modules ctikernels can select on
    ``device`` (default: the current CUDA device), into the cache this process reads. A warm
    cache builds nothing. ``modules``: "selected" for every provider of the device's class, or
    an iterable of provider names from ``MODULES``. Returns one row per module."""
    import torch

    capability = torch.cuda.get_device_capability(device)
    device_class = {(10, 0): "sm100", (10, 3): "sm103", (12, 0): "sm120", (12, 1): "sm121"}.get(
        capability
    )
    if device_class is None:
        return []
    names = None if modules == "selected" else set(modules)
    return _build(device_class, names, None, False)


def _build(device_class, names, jobs, cubins):
    os.environ.setdefault("MAX_JOBS", str(jobs or default_jobs()))
    os.environ.setdefault("FLASHINFER_NVCC_THREADS", "1")
    unknown = (names or set()) - {what for what, *_ in MODULES}
    if unknown:
        raise ValueError(f"unknown FlashInfer prewarm providers: {sorted(unknown)}")
    report = []
    for what, classes, target, kwargs in MODULES:
        if device_class not in classes or (names is not None and what not in names):
            continue
        module, generator = target.split(":")
        spec = getattr(importlib.import_module(module), generator)(**kwargs)
        start, built = time.monotonic(), not spec.is_compiled
        if built:
            spec.build(need_lock=True)
        report.append(
            {
                "module": spec.name,
                "for": what,
                "built": built,
                "seconds": round(time.monotonic() - start, 1),
            }
        )
    if cubins:
        from flashinfer.artifacts import download_artifacts

        start = time.monotonic()
        download_artifacts()
        report.append(
            {
                "module": "cubins",
                "for": "TRT-LLM gen",
                "built": True,
                "seconds": round(time.monotonic() - start, 1),
            }
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device-class", required=True, choices=sorted(DEVICE_CLASSES))
    parser.add_argument("--jobs", type=int, help="nvcc jobs (default: by cores and memory)")
    parser.add_argument("--cubins", action="store_true", help="also download TRT-LLM gen cubins")
    args = parser.parse_args(argv)
    for row in prewarm(args.device_class, jobs=args.jobs, cubins=args.cubins):
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
