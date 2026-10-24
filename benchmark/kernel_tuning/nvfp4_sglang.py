# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Runtime comparator for SGLang 72730d6; no vendored implementation.

Import the unmodified ModelOpt layer, backend selector, qwen3x NVFP4 GEMM probe/kernel, Qwen dense
activation+quant helper and shared-expert activation. Server/model dependencies outside
this boundary are import-only sentinels: calling one fails. Runtime configuration is a
single-device, TP=1 fixture with fp4 backend 'auto'. Actual dispatch remains SGLang's.

SGLang rounds gate_up GEMM output to BF16 before SiLU*up+quant; our fused path does not;
gate 2 checks that it is no worse vs FP64. Shared experts use a separate BF16 activation
and down-input quantizer. Both gated timings end at packed NVFP4 + E4M3 scales.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

REVISION = "72730d6"
# These files and the JIT include closure are shipped unmodified, outside the package.
REAL_MODULES = (
    "sglang.srt.environ",
    "sglang.srt.utils.common",
    "sglang.srt.utils.custom_op",
    "sglang.srt.utils.patch_torch",
    "sglang.srt.layers.utils",
    "sglang.srt.layers.utils.common",
    "sglang.srt.layers.parameter",
    "sglang.srt.layers.modelopt_utils",
    "sglang.srt.layers.quantization.base_config",
    "sglang.srt.layers.quantization.fp4_utils",
    "sglang.srt.layers.quantization.modelopt_quant",
    "sglang.srt.layers.activation",
    "sglang.srt.models.qwen2_moe",
    "sglang.kernels.jit",
    "sglang.kernels.spec",
    "sglang.kernels.registry",
    "sglang.kernels.selector",
    "sglang.kernels.fused_op",
    "sglang.kernels.kernel_api_logging",
    "sglang.kernels.ops.gemm",
    "sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm",
    "sglang.kernels.kda_kernels.qwen3x_nvfp4_gemm_sm120",
    "sglang.kernels.ops.activation.activation",
)
JIT_UTILS = "sglang/kernels/jit/utils"
_STUBS = set()


def _source(root, name):
    path = root.joinpath(*name.split("."))
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"


def source_files(root):
    """Minimal runtime source list, also used by run.py without importing torch."""
    root = Path(root)
    files = {_source(root, name) for name in REAL_MODULES}
    files.update((root / JIT_UTILS).rglob("*.py"))
    # Original activation and its transitive local include dependencies only.
    pending = [root / "sglang/kernels/jit/csrc/elementwise/activation.cuh"]
    include = root / "sglang/kernels/jit/include"
    import re

    while pending:
        path = pending.pop()
        if path in files:
            continue
        files.add(path)
        for target in re.findall(r'#include\s*[<"](sgl_kernel/[^>"]+)[>"]', path.read_text()):
            pending.append(include / target)
        for target in re.findall(r'#include\s*"([^"\n]+)"', path.read_text()):
            relative = path.parent / target
            if relative.is_file():
                pending.append(relative)
    files.add(root / "pyproject.toml")
    return sorted(files)


class _Unavailable:
    """Imported only for annotations/class bases of unrelated server components."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Off-path SGLang dependency reached the measured boundary")

    @classmethod
    def __class_getitem__(cls, _):
        return cls


def _namespace(root, name):
    if name in sys.modules:
        return sys.modules[name]
    if "." in name:
        parent, leaf = name.rsplit(".", 1)
        owner = _namespace(root, parent)
    else:
        owner, leaf = None, name
    module = types.ModuleType(name)
    module.__path__ = [str(root.joinpath(*name.split(".")))]
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = module
    if owner is not None:
        setattr(owner, leaf, module)
    return module


def _stub(root, name, **symbols):
    module = _namespace(root, name)
    module.__dict__.update(symbols)
    _STUBS.add(name)
    return module


def _prepare_imports(root):
    """Stub only explicitly excluded dependency modules, never real kernel modules."""
    real = set(REAL_MODULES)
    for path in (root / JIT_UTILS).rglob("*.py"):
        parts = path.relative_to(root).with_suffix("").parts
        real.add(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    # Namespace containers avoid server startup side effects from broad __init__ exports.
    for name in sorted(real, key=lambda n: n.count(".")):
        parent = name.rpartition(".")[0]
        if parent and parent not in real:
            _namespace(root, parent)
    for name in sorted(real):
        for node in ast.walk(ast.parse(_source(root, name).read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            dep = node.module
            if not dep.startswith("sglang.") or dep in real:
                continue
            # srt.utils is a lazy facade over the real common module below.
            if dep == "sglang.srt.utils":
                continue
            module = _stub(root, dep)
            for alias in node.names:
                if alias.name != "*":
                    module.__dict__.setdefault(alias.name, _Unavailable)
    # The server context is configuration, not a replacement GEMM/quantizer/activation.
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    platform = types.SimpleNamespace(
        is_sm100=cc[0] == 10, is_sm120=cc[0] == 12, is_blackwell=cc[0] in (10, 11, 12)
    )
    config = types.SimpleNamespace(kernel=types.SimpleNamespace(fp4_gemm_runner_backend="auto"))
    _stub(
        root,
        "sglang.srt.runtime_context",
        get_platform=lambda: platform,
        get_exec=lambda: config,
        publish_role=lambda: None,
    )
    _stub(
        root,
        "sglang.srt.platforms",
        current_platform=types.SimpleNamespace(is_out_of_tree=lambda: False),
    )
    facade = _namespace(root, "sglang.srt.utils")
    facade.__getattr__ = lambda name: getattr(
        importlib.import_module("sglang.srt.utils.common"), name
    )
    # common.py imports real Starlette/Requests and their pinned dependency closure.
    # run.py supplies these even on minimal runners; partial installs must fail here.
    # Not used by the measured layer or aligned shared activation; fail if reached.
    for name in ("sgl_kernel", "transformers"):
        if name not in sys.modules:
            mod = _stub(root, name)
            mod.__getattr__ = lambda name: _Unavailable if not name.startswith("__") else None

    # Stubbing an excluded child (e.g. gemm.cutedsl_dual_gemm) recursively creates
    # its parents. A real package must execute its __init__.py on import, not remain
    # an empty namespace. Defer removing these temporary parents until all children
    # are prepared, so importing the real package cannot reach an unstubbed child.
    assert not real.intersection(_STUBS)
    for name in real:
        module = sys.modules.get(name)
        if module is not None and module.__spec__.loader is None:
            del sys.modules[name]
            parent, _, leaf = name.rpartition(".")
            owner = sys.modules.get(parent)
            if owner is not None and getattr(owner, leaf, None) is module:
                delattr(owner, leaf)


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "missing"


_LOADED = None


def load(sglang_python=None):
    global torch, _LOADED
    if _LOADED is not None:
        return _LOADED
    import torch

    root = Path(sglang_python or os.environ.get("CTI_SGLANG_PYTHON", "")).resolve()
    if not (root / "sglang/srt/layers/quantization/modelopt_quant.py").is_file():
        raise RuntimeError("Set CTI_SGLANG_PYTHON or --sglang-python to pinned 72730d6/python")
    manifest = Path(__file__).with_name("nvfp4_sglang_manifest.json")
    if not manifest.is_file():
        raise RuntimeError("Pinned source manifest missing; use run.py's verified payload")
    spec = json.loads(manifest.read_text())
    assert spec["revision"] == REVISION
    for name, digest in spec["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    if any(n.startswith("sglang.") for n in sys.modules):
        raise RuntimeError("Import this isolated comparator before other SGLang modules")
    _prepare_imports(root)
    fp4 = importlib.import_module("sglang.srt.layers.quantization.fp4_utils")
    fp4.initialize_fp4_gemm_config()  # auto resolution, as in SGLang server initialization
    modelopt = importlib.import_module("sglang.srt.layers.quantization.modelopt_quant")
    qwen = importlib.import_module("sglang.srt.models.qwen2_moe")
    activation = importlib.import_module("sglang.srt.layers.activation")
    gemm = importlib.import_module("sglang.kernels.ops.gemm")
    import flashinfer

    # Fail at load, before allocating benchmark weights, if isolation masks a real
    # module. The manifest verifies bytes; this verifies Python actually loaded them.
    for name in REAL_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            assert getattr(module, "__file__", None) == str(_source(root, name)), name
    assert callable(gemm.try_qwen3x_nvfp4_gemm)
    assert callable(modelopt.flashinfer_fp4_gemm)
    assert callable(qwen.Qwen2MoeMLP._silu_fp4_quant_fused)
    for name in ("fp4_quantize", "mm_fp4", "silu_and_mul_scaled_nvfp4_experts_quantize"):
        assert callable(getattr(flashinfer, name)), name
    assert modelopt.enable_flashinfer_fp4_gemm and fp4.fp4_quantize is not None
    _LOADED = types.SimpleNamespace(
        root=root, fp4=fp4, modelopt=modelopt, qwen=qwen, activation=activation, gemm=gemm
    )
    return _LOADED


class SGLangLinear:
    """A real ModelOptFp4LinearMethod with original packed ModelOpt weight operands."""

    def __init__(self, weights, input_scale, *, sglang_python=None):
        self.api = load(sglang_python)
        cfg = self.api.modelopt.ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16, use_per_token_activation=False
        )
        self.method = self.api.modelopt.ModelOptFp4LinearMethod(cfg)
        assert self.method.quant_mode == "w4a4"
        self.layer = torch.nn.Module()
        widths = [w.data.shape[0] for w in weights]
        k = weights[0].data.shape[1] * 2
        # SGLang takes max(weight_scale_2) across partitions. Require the exact common
        # global scale used by our benchmark; never requantize either path's weights.
        for w in weights[1:]:
            torch.testing.assert_close(w.global_scale, weights[0].global_scale, rtol=0, atol=0)
        with torch.device(input_scale.device), torch.no_grad():
            self.method.create_weights(self.layer, k, widths, k, sum(widths), torch.bfloat16)
            self.layer.weight.copy_(torch.cat([w.data for w in weights]))
            self.layer.weight_scale.copy_(torch.cat([w.scale for w in weights]))
            self.layer.weight_scale_2.copy_(torch.cat([w.global_scale.reshape(1) for w in weights]))
            self.layer.input_scale.copy_(input_scale.expand(len(widths)))
            self.method.process_weights_after_loading(self.layer)
        self._dispatch = {}

    def __call__(self, x):
        return self.method.apply(self.layer, x)

    def probe(self, x):
        """Observe the actual branch during an untimed call; restore all wrappers afterward."""
        events = []
        gemm = self.api.gemm
        modelopt = self.api.modelopt
        # 72730d6 fp4_gemm imports the GEMM facade attribute inside its body on
        # every call. mm_fp4, in contrast, is imported once into modelopt_quant.
        # Wrap those exact bindings, not the qwen3x GEMM implementation or flashinfer.mm_fp4.
        original_qwen3x = gemm.try_qwen3x_nvfp4_gemm
        original_fi = modelopt.flashinfer_fp4_gemm

        def qwen3x(*args, **kwargs):
            result = original_qwen3x(*args, **kwargs)
            events.append(("qwen3x-probe", result is not None))
            return result

        def fi(*args, **kwargs):
            result = original_fi(*args, **kwargs)
            events.append(("flashinfer", kwargs["backend"]))
            return result

        gemm.try_qwen3x_nvfp4_gemm = qwen3x
        modelopt.flashinfer_fp4_gemm = fi
        try:
            result = self(x)
        finally:
            gemm.try_qwen3x_nvfp4_gemm = original_qwen3x
            modelopt.flashinfer_fp4_gemm = original_fi
        assert events and events[0][0] == "qwen3x-probe", events
        if events[0][1]:
            assert len(events) == 1, events
            backend = "sglang-cute-sm120"
        else:
            assert len(events) == 2 and events[1][0] == "flashinfer", events
            backend = "flashinfer-" + events[1][1]
            assert events[1][1] != "auto", "SGLang auto must resolve before timing"
        self._dispatch[x.shape[0]] = backend
        return result

    def metadata(self, rows):
        assert rows in self._dispatch, "Probe every M before graph timing"
        return {
            "sglang_revision": REVISION,
            "flashinfer_version": _version("flashinfer-python"),
            "cutlass_dsl_version": _version("nvidia-cutlass-dsl"),
            "chosen_backend": self._dispatch[rows],
            "qwen3x_probe_taken": self._dispatch[rows] == "sglang-cute-sm120",
            "code_paths": [
                "sglang/srt/layers/quantization/modelopt_quant.py:ModelOptFp4LinearMethod.apply",
                "sglang/srt/layers/quantization/modelopt_quant.py:fp4_gemm",
                "sglang/srt/layers/quantization/fp4_utils.py:initialize_fp4_gemm_config",
                "sglang/srt/layers/quantization/fp4_utils.py:_flashinfer_fp4_quantize_impl",
                "sglang/kernels/ops/gemm/__init__.py:try_qwen3x_nvfp4_gemm",
                "sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm.py:try_qwen3x_nvfp4_gemm",
                (
                    "sglang/kernels/kda_kernels/qwen3x_nvfp4_gemm_sm120.py:_run_qwen3x_nvfp4_gemm"
                    if self._dispatch[rows] == "sglang-cute-sm120"
                    else "flashinfer/gemm/gemm_base.py:mm_fp4"
                ),
            ],
            "import_stubs": sorted(_STUBS),
            "boundary": "BF16 activation -> input quantization -> BF16 GEMM output",
        }


class SGLangGated(SGLangLinear):
    def __init__(self, gate, up, input_scale, output_scale, *, shared=False, sglang_python=None):
        super().__init__((gate, up), input_scale, sglang_python=sglang_python)
        self.shared = shared
        # Only the original helper's documented state is needed; no model/server startup.
        self.mlp = self.api.qwen.Qwen2MoeMLP.__new__(self.api.qwen.Qwen2MoeMLP)
        torch.nn.Module.__init__(self.mlp)
        self.mlp.down_proj = torch.nn.Module()
        self.mlp.down_proj.input_scale_inv = output_scale.reciprocal()
        self.mlp._down_input_scale_inv_1d = None
        self.mlp._masked_m_cache = {}

    def __call__(self, x):
        gate_up = super().__call__(x)
        if self.shared:
            activated = self.api.activation.silu_and_mul(gate_up)
            return self.api.fp4.fp4_quantize(activated, self.mlp.down_proj.input_scale_inv)
        return self.mlp._silu_fp4_quant_fused(gate_up)

    def metadata(self, rows):
        record = super().metadata(rows)
        record["boundary"] = "BF16 activation -> NVFP4 + E4M3 scales"
        record["sglang_gate_up_dtype"] = "bfloat16"
        record["activation_path"] = "separate-shared" if self.shared else "dense-fused-quant"
        record["code_paths"] += (
            [
                "sglang/srt/layers/activation.py:silu_and_mul",
                "sglang/kernels/ops/activation/activation.py:silu_and_mul",
                "sglang/srt/layers/quantization/fp4_utils.py:_flashinfer_fp4_quantize_impl",
            ]
            if self.shared
            else [
                "sglang/srt/models/qwen2_moe.py:Qwen2MoeMLP._silu_fp4_quant_fused",
                "flashinfer/activation.py:silu_and_mul_scaled_nvfp4_experts_quantize",
            ]
        )
        return record
