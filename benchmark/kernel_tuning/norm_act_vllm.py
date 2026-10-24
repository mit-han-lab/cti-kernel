# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""vLLM 0.31.0 market kernels, loaded from --market-root (never vendored).

--vllm-mode shipped (default) launches the actual cache modules. compiled builds
fragments from the supplied IR and requires source equality; auto falls back to
shipped on any failed proof. Failed proofs never qualify compiled kernels.
"""

import ast
import contextlib
import difflib
import hashlib
import importlib.util
import json
import linecache
import re
import sys
import types
from unittest.mock import patch

import torch

COMPILE_OPTIONS = {
    "enable_auto_functionalized_v2": False,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}
# from the vLLM 0.31.0 nsys traces (Qwen3-8B and Qwen3.5-9B DFlash), RTX 5090. Qwen3.5's
# site A (`..._to_copy_add_fused_add_rms_norm_0`) runs 8 warps; every other site 16 warps.
TRACE_GEOMETRY = {
    "a": {"block": [512, 1, 1]},
    "b": {"block": [512, 1, 1]},
    "silu": {"block": [128, 1, 1], "grids": {16: [192, 1, 1], 64: [768, 1, 1]}},
}
TRACE_GEOMETRY_ZERO = {**TRACE_GEOMETRY, "a": {"block": [256, 1, 1]}}


class MarketError(RuntimeError):
    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt or {}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Execute by file path without writing __pycache__ into the shipped tree.
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


class IROp:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_input_generator(self, fn):
        return fn

    def override_tolerance(self, *args, **kwargs):
        return None


def load_ir(root):
    # A private stub hierarchy resolves ..op without importing vLLM or its extensions.
    prefix = "_cti_market_vllm_ir"
    for name in (prefix, prefix + ".ops"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    op = types.ModuleType(prefix + ".op")
    op.register_op = lambda fn=None, **kw: IROp(fn) if fn is not None else IROp
    sys.modules[op.__name__] = op
    return load_file(prefix + ".ops.layernorm", root / "vllm-ir/layernorm.py")


def extract_kernels(source):
    """Extract JIT functions, including strings embedded in Inductor Python wrappers."""
    tree = ast.parse(source)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("triton_"):
            if any(ast.unparse(d) == "triton.jit" for d in node.decorator_list):
                result.append(ast.unparse(node))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "@triton.jit" in node.value:
                result.extend(extract_kernels(node.value))
    return result


def normalize_kernel(source):
    """Ignore host metadata/names, never arithmetic, casts, loads, stores or tiling."""
    kernels = extract_kernels(source)
    if len(kernels) != 1:
        raise ValueError(f"expected one JIT kernel, got {len(kernels)}")
    fn = ast.parse(kernels[0]).body[0]
    fn.name = "KERNEL"
    for decorator in fn.decorator_list:
        if isinstance(decorator, ast.Call):
            decorator.keywords = [
                kw
                for kw in decorator.keywords
                if kw.arg not in ("triton_meta", "inductor_meta", "filename")
            ]
    names = {}
    counters = {}

    def rename(match):
        value = match.group(0)
        if value not in names:
            family = re.sub(r"\d+$", "", value)
            index = counters.get(family, 0)
            counters[family] = index + 1
            names[value] = f"{family}{index}"
        return names[value]

    return re.sub(r"\b(?:_?tmp|buf|in_ptr|out_ptr|in_out_ptr)\d+\b", rename, ast.unparse(fn))


def kernel_signature(source):
    """Pointer dtypes are ABI, even though host metadata is excluded from text proof."""
    fn = ast.parse(extract_kernels(source)[0]).body[0]
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "triton_meta" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if isinstance(key, ast.Constant) and key.value == "signature":
                        signature = ast.literal_eval(value)
                        return [signature.get(arg.arg) for arg in fn.args.args]
    return None


def equality_receipt(site, sources, path):
    kernels = [k for source in sources for k in extract_kernels(source)]
    shipped = path.read_text()
    expected = normalize_kernel(shipped)
    actual = normalize_kernel(kernels[0]) if len(kernels) == 1 else f"kernel_count={len(kernels)}"
    expected_abi = kernel_signature(shipped)
    actual_abi = kernel_signature(kernels[0]) if len(kernels) == 1 else None
    equal = actual == expected and expected_abi == actual_abi
    diff = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="shipped",
            tofile="compiled",
            lineterm="",
        )
    )
    if expected_abi != actual_abi:
        diff = [f"shipped ABI: {expected_abi}", f"compiled ABI: {actual_abi}"] + diff
    return {
        "site": site,
        "vllm_file": str(path),
        "equal": equal,
        "abi_equal": expected_abi == actual_abi,
        "diff": diff[:40] if not equal else [],
    }


def site_file(root, site, zero, width=4096):
    if width == 2048 and not zero and site in ("a", "b"):
        directory = root / "vllm-kernels/qwen3moe-h2048"
        pattern = "*fused_add_rms_norm_moe_forward_0.*.py" if site == "a" else "*.cw4iw2rh*.py"
    elif (width == 4096 and site in ("a", "b", "d2")) or (width == 12288 and site == "silu"):
        directory = root / "vllm-kernels" / ("qwen35" if zero else "qwen3")
        pattern = {
            "a": "triton_red_fused_*fused_add_rms_norm_0.*.py",
            "b": "*.c7zxzi*.py" if zero else "*.ctzv25*.py",
            "d2": "*.cxy7p5q7*.py" if zero else "*.ciqqjvb7*.py",
            "silu": "triton_poi_fused_mul_silu_slice_1.*.py",
        }[site]
    else:
        raise MarketError(f"no vLLM 0.31.0 capture for width={width}, zero={zero}, site={site}")
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise MarketError(f"expected one shipped {site} kernel in {directory}, got {len(paths)}")
    return paths[0]


def optional_sites(build):
    """Omit only unsupported captures; failed proofs and broken fixtures remain errors."""
    try:
        return build(), None
    except MarketError as error:
        if not str(error).startswith("no vLLM 0.31.0 capture for "):
            raise
        return None, {"omitted": True, "reason": str(error)}


class RuntimeMetadata(ast.NodeTransformer):
    """Patch host device/cache metadata only; leave every JIT instruction intact."""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "DeviceProperties":
            return ast.copy_location(ast.Name(id="_current_device", ctx=ast.Load()), node)
        node = self.generic_visit(node)
        for kw in node.keywords:
            if kw.arg == "inductor_meta":
                kw.value = ast.Call(
                    func=ast.Name(id="_runtime_meta", ctx=ast.Load()), args=[kw.value], keywords=[]
                )
        return node


def specialize_aux_stride(source, stride):
    """Adapt only the shipped D2 aux store address, keeping all math intact."""
    tree = ast.parse(source)
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func) == "tl.store"
        and node.args
        and isinstance(node.args[0], ast.BinOp)
        and isinstance(node.args[0].left, ast.Name)
        and node.args[0].left.id == "out_ptr3"
    ]
    if len(stores) != 1:
        raise MarketError("D2 must have exactly one aux store")
    address = stores[0].args[0]
    if ast.unparse(address) != "out_ptr3 + (r0_1 + 4096 * x0)":
        raise MarketError("unexpected shipped D2 aux address")
    address.right.right.left = ast.Constant(value=stride)
    return ast.unparse(ast.fix_missing_locations(tree))


def load_shipped(path, *, aux_stride=None, configs=None):
    import triton
    from torch._inductor.runtime.hints import DeviceProperties

    props = DeviceProperties.create(torch.device("cuda", torch.cuda.current_device()))
    best_path = path.with_suffix(".best_config")
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    pinned = best is not None and "RTX 5090" in torch.cuda.get_device_name()

    def runtime_meta(meta):
        return meta | {
            "force_disable_caches": True,
            "autotune_local_cache": False,
            "autotune_remote_cache": False,
            **({"dynamic_scale_rblock": False} if pinned or configs is not None else {}),
            **({"coordinate_descent_tuning": False} if configs is not None else {}),
        }

    name = "_cti_vllm_kernel_" + hashlib.sha256(f"{path}:{aux_stride}".encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module._current_device = props
    module._runtime_meta = runtime_meta
    sys.modules[name] = module
    source = path.read_text()
    filename = str(path)
    if aux_stride is not None:
        source = specialize_aux_stride(source, aux_stride)
        # Triton uses inspect to recover the JIT definition. Register the layout
        # specialization in memory only; never write vendor source to the repo.
        filename = f"<cti-vllm-{name}-aux-stride-{aux_stride}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    tree = ast.fix_missing_locations(RuntimeMetadata().visit(ast.parse(source)))
    exec(compile(tree, filename, "exec"), module.__dict__)
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(names) != 1:
        raise MarketError("shipped module must contain exactly one kernel")
    kernel = getattr(module, names[0])
    if pinned:
        kernel.configs = [
            triton.Config(
                {key: value for key, value in best.items() if key.endswith("BLOCK")},
                num_warps=best["num_warps"],
                num_stages=best["num_stages"],
            )
        ]
    if configs is not None:
        kernel.configs = [
            triton.Config(
                {k: v for k, v in config.items() if k not in ("num_warps", "num_stages")},
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            for config in configs
        ]
    return kernel, best if pinned else None


def chosen_configs(kernel):
    return [
        launcher.config.kwargs
        | {
            "num_warps": launcher.config.num_warps,
            "num_stages": launcher.config.num_stages,
        }
        for launcher in kernel.launchers
    ]


@contextlib.contextmanager
def compiled_configs(expected=None):
    """Record warmed Inductor launchers, or constrain a probe to those launchers.

    This wraps dispatch only during eager setup. It never instruments timing or
    activates Kineto. Refuse a probe if its compiler cannot supply the selection.
    """
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    observed = {}
    original = CachingAutotuner.run

    def run(kernel, *args, **kwargs):
        name = kernel.fn.__name__
        if expected is not None:
            if name not in expected or len(expected[name]) != 1:
                raise MarketError("compiled probe kernel differs from parent")
            if not kernel.launchers:
                kernel.precompile()
            matches = [
                launcher
                for launcher, config in zip(kernel.launchers, chosen_configs(kernel), strict=True)
                if config == expected[name][0]
            ]
            if not matches:
                raise MarketError("compiled probe cannot reproduce parent config")
            kernel.launchers = matches[:1]
            # Newer Inductor versions cache the selected launcher's fast path.
            if hasattr(kernel, "_cached_launcher"):
                kernel._cached_launcher = None
            kernel.inductor_meta["coordinate_descent_tuning"] = False
        result = original(kernel, *args, **kwargs)
        observed[name] = chosen_configs(kernel)
        return result

    with patch.object(CachingAutotuner, "run", run):
        yield observed
    if not observed or (expected is not None and observed != expected):
        raise MarketError("compiled probe launch configs were not proven equal")


def geometry_receipt(site, m, kernels, device, zero=False, *, width=4096, configs=None):
    if width == 2048:
        config = configs[0] if configs is not None and len(configs) == 1 else {}
        xblock, warps = config.get("XBLOCK"), config.get("num_warps")
        grid = [(m + xblock - 1) // xblock, 1, 1] if xblock else None
        block = [32 * warps, 1, 1] if warps else None
        matches = bool(
            grid
            and block
            and len(kernels) == 1
            and kernels[0].get("grid") == grid
            and kernels[0].get("block") == block
        )
        return {
            "basis": "autotuned-launcher",
            "kernels": kernels,
            "expected_grid": grid,
            "expected_block": block,
            "checked": True,
            "matches": matches,
            "valid": matches,
            "warning": None,
        }
    expected = (TRACE_GEOMETRY_ZERO if zero else TRACE_GEOMETRY)[site]
    grid = [m, 1, 1] if site != "silu" else expected["grids"].get(m)
    on_5090 = "RTX 5090" in device
    valid = (
        len(kernels) == 1
        and kernels[0].get("grid") is not None
        and kernels[0].get("block") is not None
    )
    matches = (
        valid
        and kernels[0]["block"] == expected["block"]
        and (grid is None or kernels[0]["grid"] == grid)
    )
    return {
        "basis": "trace",
        "kernels": kernels,
        "expected_grid": grid,
        "expected_block": expected["block"],
        "checked": on_5090,
        "matches": bool(matches) if on_5090 else None,
        "warning": None if on_5090 else "Trace geometry is RTX 5090 only; recorded on this device",
        "valid": bool(valid and (matches or not on_5090)),
    }


def fragments(root, zero):
    op = load_ir(root).fused_add_rms_norm.fn

    def weight(w):
        # vLLM 0.31.0 model_executor/layers/layernorm.py:159-168,
        # GemmaRMSNorm.forward_native passes FP32 weight + 1 to the same IR op.
        return w.float() + 1.0 if zero else w

    def a(h, residual, w):
        return op(h, residual, weight(w), 1e-6)[0]

    def b(h2, h1, residual, w):
        s1 = (h1.float() + residual.float()).to(h1.dtype)
        return op(h2, s1, weight(w), 1e-6)

    def b_aux(h2, h1, residual, w):
        s1 = (h1.float() + residual.float()).to(h1.dtype)
        normalized, new_residual = op(h2, s1, weight(w), 1e-6)
        return normalized, new_residual, h2 + s1

    return {"a": a, "b": b, "d2": b_aux}


class VllmSite:
    def __init__(self, args, site, example, zero=False):
        from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard
        from benchmark.kernel_tuning.norm_act_paired import vllm_native
        from benchmark.kernel_tuning.norm_act_probe import run_probe

        install_profiler_guard()

        self.site = site
        width = example[0].shape[-1] // (2 if site == "silu" else 1)
        self.path = site_file(args.market_root, site, zero, width)
        self.compiled = None
        self.kernel = None
        self.metadata = {
            "site": site,
            "model": "Qwen3-30B-A3B" if width == 2048 else ("Qwen3.5-9B" if zero else "Qwen3-8B"),
            "fixture": self.path.parent.name,
            "vllm_file": str(self.path),
            "source_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }
        if width == 2048:
            self.metadata["provenance_sha256"] = hashlib.sha256(
                (self.path.parent / "PROVENANCE.txt").read_bytes()
            ).hexdigest()
        if site == "d2":
            self.metadata["dispatch"] = (
                "DFlash capturing layer B: h2, h1, residual, weight -> normalized, new_residual, aux"
            )
            self.metadata["compiled_aux_packing"] = (
                "original contiguous three-output IR graph plus timed copy into aux slice"
            )
        self.metadata.update(
            {
                "wrapper_sha256": hashlib.sha256(
                    (self.path.parent / "layer_graph.wrapper.txt").read_bytes()
                ).hexdigest(),
                "ir_sha256": hashlib.sha256(
                    (args.market_root / "vllm-ir/layernorm.py").read_bytes()
                ).hexdigest(),
                "compile_options": COMPILE_OPTIONS,
                "vllm_config": {
                    "custom_ops": ["none"],
                    "ir_op_priority": {"fused_add_rms_norm": ["native"]},
                    "compile_range": [1, 8192],
                },
            }
        )
        mode = args.vllm_mode
        proof = None
        if mode in ("compiled", "auto"):
            from torch._inductor.utils import run_and_get_code

            native = vllm_native if site == "silu" else fragments(args.market_root, zero)[site]
            for tensor in example:
                if tensor.ndim > 1:
                    torch._dynamo.mark_dynamic(tensor, 0)
            try:
                compiled = torch.compile(native, fullgraph=True, options=COMPILE_OPTIONS)
                _, sources = run_and_get_code(compiled, *example)
                proof = equality_receipt(site, sources, self.path)
                if proof["equal"]:
                    self.compiled = compiled
            except Exception as error:  # noqa: BLE001 -- compiler failures become failed proofs
                proof = {
                    "site": site,
                    "vllm_file": str(self.path),
                    "equal": False,
                    "diff": [],
                    "error_type": type(error).__name__,
                }
            self.metadata["proof"] = proof
            if self.compiled is None and mode == "compiled":
                raise MarketError("compiled vLLM kernel was not proven equal", self.metadata)
        if self.compiled is None:
            if example[0].dtype != torch.bfloat16:
                raise MarketError("shipped kernels require BF16", self.metadata)
            self.kernel, pinned = load_shipped(self.path)
            self.aux_kernel = None
            if site == "d2":
                self.aux_kernel, _ = load_shipped(self.path, aux_stride=3 * width)
                adapted = specialize_aux_stride(self.path.read_text(), 3 * width)
                self.metadata["aux_layout"] = {
                    "original_row_stride": width,
                    "slice_row_stride": 3 * width,
                    "adaptation": "only out_ptr3 row stride; in-memory JIT specialization",
                    "original_jit_equal": False,
                    "adapted_source_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
                }
            self.metadata.update(
                mode="shipped",
                pinned_config=pinned,
                best_config=(
                    json.loads(self.path.with_suffix(".best_config").read_text())
                    if self.path.with_suffix(".best_config").exists()
                    else None
                ),
                autotuned=pinned is None,
                proof={
                    "site": site,
                    "vllm_file": str(self.path),
                    "equal": True,
                    "diff": [],
                    "basis": "launch original shipped JIT body",
                },
            )
            if proof is not None:
                self.metadata["compiled_proof"] = proof
            if site == "d2":
                self.metadata["proof"]["scope"] = (
                    "original contiguous kernel; timed slice specialization described in aux_layout"
                )
        else:
            self.metadata["mode"] = "compiled"
        if self.compiled is not None:
            with compiled_configs() as selected:
                self(*example)
            self.metadata["chosen_configs"] = selected
        else:
            self(*example)
        probe_spec = {
            "kind": "vllm",
            "site": site,
            "mode": self.metadata["mode"],
            "market_root": args.market_root,
            "op": "silu_mul" if site == "silu" else ("gemma_add_norm" if zero else "add_norm"),
            "m": example[0].shape[0],
            "width": example[0].shape[-1] // (2 if site == "silu" else 1),
            "dtype": {torch.bfloat16: "bf16", torch.float16: "fp16"}[example[0].dtype],
        }
        if self.kernel is not None:
            self.metadata["chosen_configs"] = chosen_configs(self.kernel)
            probe_spec["configs"] = self.metadata["chosen_configs"]
        else:
            probe_spec["compiled_configs"] = self.metadata["chosen_configs"]
        kernels = run_probe(probe_spec)["kernels"]
        geometry = geometry_receipt(
            "b" if site == "d2" else site,
            example[0].shape[0],
            kernels,
            torch.cuda.get_device_name(),
            zero,
            width=width,
            configs=(
                self.metadata["chosen_configs"]
                if self.kernel is not None
                else [c for values in self.metadata["chosen_configs"].values() for c in values]
            ),
        )
        self.metadata.update(geometry=geometry)
        if site == "d2" and self.kernel is not None:
            aux = torch.empty(
                (example[0].shape[0], 3 * example[0].shape[1]),
                device=example[0].device,
                dtype=example[0].dtype,
            )[:, example[0].shape[1] : 2 * example[0].shape[1]]
            self(*example, aux_output=aux)
            aux_configs = chosen_configs(self.aux_kernel)
            aux_kernels = run_probe(
                probe_spec
                | {
                    "op": probe_spec["op"] + "_sum",
                    "aux": True,
                    "aux_configs": aux_configs,
                }
            )["kernels"]
            aux_geometry = geometry_receipt(
                "b", example[0].shape[0], aux_kernels, torch.cuda.get_device_name(), zero
            )
            self.metadata["aux_layout"]["geometry"] = aux_geometry
            self.metadata["aux_layout"]["chosen_configs"] = aux_configs
            if not aux_geometry["valid"]:
                raise MarketError("vLLM D2 aux geometry proof failed", self.metadata)
        if not geometry["valid"]:
            raise MarketError("vLLM geometry proof failed", self.metadata)

    def __call__(self, *tensors, output=None, residual_output=None, aux_output=None):
        if self.compiled is not None:
            result = self.compiled(*tensors)
            if self.site == "d2" and aux_output is not None:
                aux_output.copy_(result[2])
                return *result[:2], aux_output
            return result
        x = tensors[0]
        out = (
            output
            if output is not None
            else (
                torch.empty_like(x)
                if self.site != "silu"
                else torch.empty_like(
                    x[..., : x.shape[-1] // 2], memory_format=torch.contiguous_format
                )
            )
        )
        stream = torch.cuda.current_stream().cuda_stream
        if self.site == "silu":
            self.kernel.run(x, out, out.numel(), stream=stream)
            return out
        if self.site == "a":
            outputs = (out, torch.empty_like(x)) if x.shape[1] == 2048 else (out,)
            self.kernel.run(*tensors, *outputs, x.shape[0], x.shape[1], stream=stream)
            return out
        residual = residual_output if residual_output is not None else torch.empty_like(x)
        if self.site == "d2":
            aux = aux_output if aux_output is not None else torch.empty_like(x)
            if aux.shape != x.shape or aux.dtype != x.dtype:
                raise MarketError("D2 aux output must match input shape and dtype")
            kernel = self.kernel
            if aux.stride() == (3 * x.shape[1], 1):
                kernel = self.aux_kernel
            elif not aux.is_contiguous():
                raise MarketError("D2 aux output must be a column slice of [M, 3*H]")
            # Capturing-layer vLLM cache wrapper call order evidence:
            # triton_red_fused_add_fused_add_rms_norm_2.run(
            #     buf5, buf0, arg4_1, arg7_1, buf6, buf8, buf15, s72, 4096)
            # h2, h1, old residual, weight -> new residual, normalized, aux.
            kernel.run(*tensors, residual, out, aux, x.shape[0], x.shape[1], stream=stream)
            return out, residual, aux
        # wrapper .run: h2, h1, old residual, weight, new residual, normalized, M, H.
        self.kernel.run(*tensors, residual, out, x.shape[0], x.shape[1], stream=stream)
        return out, residual


class VllmNorm:
    """capture_call uses the layer index, independent of warm/check call counts."""

    def __init__(self, a, b, pools, site=None):
        self.a, self.b, self.pools, self.site = a, b, pools, site

    def __call__(self, tensors, unused=None):
        return self.a(*tensors[:3])

    def capture_call(self, index, tensors, unused=None):
        if self.site == "b" or (self.site is None and index % 2):
            x, residual, w = tensors[:3]
            return self.b(x, self.pools.previous_hidden(index), residual, w)
        return self.a(*tensors[:3])
