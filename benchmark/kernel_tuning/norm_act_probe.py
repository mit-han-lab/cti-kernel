# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Disposable profiling interpreter: python -m ...norm_act_probe --spec JSON.

Only this module activates Kineto. Specs contain construction parameters, never
live tensors or IPC handles. A worker's inherited PYTHONPATH preserves its FI
overlay; CUDA_VISIBLE_DEVICES and the caller's logical device are preserved too.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from benchmark.kernel_tuning.norm_act_vllm import MarketError


def encode_spec(spec):
    allowed = {
        "kind",
        "op",
        "m",
        "width",
        "dtype",
        "device",
        "market_root",
        "sglang_root",
        "site",
        "mode",
        "configs",
        "compiled_configs",
        "aux_configs",
        "aux",
        "provider",
        "config",
        "static_weight",
        "candidate",
        "fused",
        "residual_dtype",
    }
    if set(spec) - allowed or spec.get("kind") not in {
        "vllm",
        "provider",
        "sglang_silu",
        "worker",
        "sgl_fp32",
        "sgl_fp32c",
    }:
        raise ValueError("invalid probe spec")
    if spec.get("dtype") not in {"bf16", "fp16"} or min(spec["m"], spec["width"]) <= 0:
        raise ValueError("invalid probe shape/dtype")
    return json.dumps(
        spec,
        default=lambda value: str(value) if isinstance(value, Path) else asdict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def run_probe(spec):
    """Blocking handoff: the caller has no GPU work pending while the probe runs."""
    torch.cuda.synchronize()
    encoded = encode_spec(spec | {"device": torch.cuda.current_device()})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.kernel_tuning.norm_act_probe", "--spec", encoded],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=env,
    )
    # Never echo native diagnostics, command lines, or environment values.
    if result.returncode:
        raise MarketError("kernel probe failed")
    try:
        reply = json.loads(result.stdout)
        for key in ("kernels", "kernels_even"):
            if key not in reply and key == "kernels_even":
                continue
            if not isinstance(reply[key], list) or not reply[key]:
                raise ValueError("missing kernels")
            for kernel in reply[key]:
                if not isinstance(kernel["name"], str):
                    raise ValueError("invalid kernel name")
                for field in ("grid", "block"):
                    values = kernel[field]
                    if (
                        not isinstance(values, list)
                        or len(values) != 3
                        or any(type(v) is not int or v <= 0 for v in values)
                    ):
                        raise ValueError("invalid launch geometry")
        return reply
    except (KeyError, TypeError, ValueError):
        raise MarketError("invalid kernel probe receipt") from None


def provider_spec(fn, *, op, m, width, dtype, residual_dtype="same"):
    """Describe a warmed local provider, including explicit tuning overrides."""
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall

    fused = isinstance(fn, SumNormCall) and fn.fused
    if isinstance(fn, SumNormCall):
        fn = fn.norm
    from benchmark.kernel_tuning.norm_act_norm import SglangFP32Call

    if isinstance(fn, SglangFP32Call):
        return {
            "kind": "sgl_fp32c" if fn.compiled else "sgl_fp32",
            "op": op,
            "m": m,
            "width": width,
            "dtype": dtype,
            "residual_dtype": "fp32",
        }
    handle = getattr(fn, "op", fn)
    return {
        "kind": "provider",
        "op": op,
        "m": m,
        "width": width,
        "dtype": dtype,
        "provider": handle._override or "auto",
        "residual_dtype": residual_dtype,
        "config": handle.config,
        "static_weight": getattr(handle, "static_weight", True),
        "fused": fused,
    }


def profile_kernels(fn, x, out):
    """One eager call; compilation and warmup happen before activating Kineto."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        fn(x, out)
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory(prefix="norm-act-trace-") as directory:
        path = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    return [
        {
            "name": event["name"],
            "grid": event.get("args", {}).get("grid"),
            "block": event.get("args", {}).get("block"),
        }
        for event in events
        if event.get("cat") == "kernel"
    ]


def fresh_tensors(spec):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[spec["dtype"]]
    m, width = spec["m"], spec["width"]
    x = torch.randn((m, width * (2 if spec["op"] == "silu_mul" else 1)), device="cuda", dtype=dtype)
    if spec["op"] == "silu_mul":
        return x, torch.empty((m, width), device="cuda", dtype=dtype)
    residual_dtype = torch.float32 if spec.get("residual_dtype") == "fp32" else dtype
    tensors = (
        x,
        torch.randn_like(x, dtype=residual_dtype),
        torch.randn(width, device="cuda", dtype=dtype),
    )
    if spec["op"].endswith("_sum"):
        tensors += (
            torch.empty((m, 3 * width), device="cuda", dtype=residual_dtype)[:, width : 2 * width],
        )
    return tensors, None


def build_call(spec, inputs):
    from benchmark.kernel_tuning.norm_act_norm import NormCall
    from benchmark.kernel_tuning.norm_act_paired import SumNormCall, load_sglang, vllm_native
    from benchmark.kernel_tuning.norm_act_vllm import (
        COMPILE_OPTIONS,
        VllmSite,
        compiled_configs,
        equality_receipt,
        fragments,
        load_shipped,
        site_file,
    )

    kind, zero = spec["kind"], spec["op"].startswith("gemma")
    if kind in ("sgl_fp32", "sgl_fp32c"):
        from benchmark.kernel_tuning.norm_act_norm import SglangFP32Call

        call = SglangFP32Call(zero, compiled=kind == "sgl_fp32c")
        return SumNormCall(call) if spec["op"].endswith("_sum") else call
    if kind == "sglang_silu":
        return load_sglang(Path(spec["sglang_root"]), inputs.dtype)[0]
    if kind == "provider":
        if spec["op"] == "silu_mul":
            from ctikernels.activation._dispatch import Config
            from ctikernels.activation.inference import SiluAndMul

            config = Config(**spec["config"]) if spec.get("config") else None
            return SiluAndMul(provider=spec["provider"], config=config)
        from ctikernels.norm._dispatch import Config

        config = Config(**spec["config"]) if spec.get("config") else None
        call = NormCall(zero, spec["provider"], config, static_weight=spec["static_weight"])
        return (
            SumNormCall(call, fused=spec.get("fused", False))
            if spec["op"].endswith("_sum")
            else call
        )
    if kind == "worker":
        if spec["candidate"] == "cute_remote":
            return build_call(
                spec | {"kind": "provider", "provider": "cute", "fused": True}, inputs
            )
        from benchmark.kernel_tuning.norm_act_market import SglangNormCall, load_sglang_norm

        wrapper, direct, _ = load_sglang_norm(Path(spec["market_root"]))
        module = {"sglang": wrapper, "fi071_direct": direct}[spec["candidate"]]
        return SglangNormCall(
            getattr(module, "gemma_fused_add_rmsnorm" if zero else "fused_add_rmsnorm")
        )
    site = VllmSite.__new__(VllmSite)
    site.site, site.compiled, site.kernel, site.aux_kernel = spec["site"], None, None, None
    root = Path(spec["market_root"])
    h1 = torch.randn_like(inputs[0]) if site.site in ("b", "d2") else None
    path = site_file(root, site.site, zero, spec["width"])
    if spec["mode"] == "compiled":
        from torch._inductor.utils import run_and_get_code

        native = vllm_native if site.site == "silu" else fragments(root, zero)[site.site]
        for tensor in inputs if isinstance(inputs, tuple) else (inputs,):
            if tensor.ndim > 1:
                torch._dynamo.mark_dynamic(tensor, 0)
        if h1 is not None:
            torch._dynamo.mark_dynamic(h1, 0)
        compiled = torch.compile(native, fullgraph=True, options=COMPILE_OPTIONS)

        proven = False

        def pinned(*values):
            nonlocal proven
            with compiled_configs(spec["compiled_configs"]):
                if not proven:
                    result, sources = run_and_get_code(compiled, *values)
                    if not equality_receipt(site.site, sources, path)["equal"]:
                        raise MarketError("compiled probe kernel was not proven equal")
                    proven = True
                    return result
                return compiled(*values)

        site.compiled = pinned
    else:
        if len(spec["configs"]) != 1:
            raise MarketError("probe requires the parent's selected vLLM config")
        site.kernel, _ = load_shipped(path, configs=spec["configs"])
        if spec.get("aux"):
            if len(spec["aux_configs"]) != 1:
                raise MarketError("probe requires the parent's selected aux config")
            site.aux_kernel, _ = load_shipped(
                path, aux_stride=3 * spec["width"], configs=spec["aux_configs"]
            )
    if site.site == "silu":
        return lambda x, out: site(x)

    def call(tensors, unused):
        x, residual, weight = tensors[:3]
        values = (x, h1, residual, weight) if h1 is not None else (x, residual, weight)
        return site(*values, **({"aux_output": tensors[3]} if spec.get("aux") else {}))

    return call


def probe(spec):
    encode_spec(spec)
    torch.cuda.set_device(spec.get("device", 0))
    inputs, out = fresh_tensors(spec)
    fn = build_call(spec, inputs)
    fn(inputs, out)
    summed = spec["op"].endswith("_sum") and spec["kind"] != "vllm"
    # Even D2 layers run the plain norm: a fused sum wrapper's base call takes 3 tensors.
    even = getattr(fn, "norm", fn)
    if summed:
        even(inputs[:3], out)
    torch.cuda.synchronize()
    result = {"kernels": profile_kernels(fn, inputs, out)}
    if summed:
        result["kernels_even"] = profile_kernels(even, inputs[:3], out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    # Native compiler/import output must not corrupt the JSON protocol.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    with open(os.devnull, "w") as quiet:
        os.dup2(quiet.fileno(), sys.stdout.fileno())
        os.dup2(quiet.fileno(), sys.stderr.fileno())
        try:
            with contextlib.redirect_stdout(quiet), torch.inference_mode():
                result = probe(json.loads(args.spec))
        except Exception:  # noqa: BLE001 -- native errors may contain environment values
            return 1
    output.write(json.dumps(result, allow_nan=False) + "\n")
    output.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
