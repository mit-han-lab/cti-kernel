# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Optional CPU import regression against the external, pinned SGLang source tree."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from ...support.child import run_child


def test_sglang_leaf_import_without_serving_dependencies():
    if any(importlib.util.find_spec(name) is None for name in ("torch", "triton", "einops")):
        pytest.skip("requires an existing torch/triton/einops environment")
    result = run_child(
        [
            sys.executable,
            "-c",
            r'''
import os
import sys
from pathlib import Path

blocked = """
orjson psutil zmq fastapi uvicorn pydantic requests aiohttp uvloop setproctitle
msgspec pybase64 sentencepiece transformers huggingface_hub safetensors IPython
pynvml cuda sgl_kernel flashinfer vllm decord PIL outlines xgrammar openai tiktoken
prometheus_client partial_json_parser compressed_tensors modelscope blobfile
llguidance soundfile numba scipy gguf torchao ray
""".split()
for name in blocked:
    sys.modules[name] = None

import torch
import triton

def forbidden(*args, **kwargs):
    raise AssertionError("GPU initialization, kernel launch or JIT build during CPU test")

torch.cuda._lazy_init = forbidden
triton.runtime.JITFunction.run = forbidden
triton.compiler.compile = forbidden

from benchmark.kernel_tuning.gdn_decode_external import SLEAVES, load_sglang

root_value = os.environ.get("SGLANG_ROOT")
if not root_value:
    sys.exit(77)
root = Path(root_value).resolve()
if not (root / "sglang").is_dir():
    sys.exit(77)
leaves = load_sglang(str(root), 120)
assert all(callable(fn) for fn in (leaves.conv, leaves.recurrent, leaves.norm))
assert load_sglang(str(root), 120) is leaves
assert set(leaves.metadata["source_sha256"]) == set(SLEAVES.values())
assert all(len(digest) == 64 for digest in leaves.metadata["source_sha256"].values())
assert all(sys.modules[name] is None for name in blocked)
assert "sglang.srt.utils" in leaves.metadata["shims"]
# Shims are scoped to loading and do not leak a fake serving package.
assert "sglang.srt.utils" not in sys.modules
assert not torch.cuda.is_initialized()
print("SGLang CPU leaf import passed (serving dependencies blocked)")
''',
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=Path(__file__).resolve().parents[3],
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("set SGLANG_ROOT to the pinned SGLang python source directory")
    assert result.returncode == 0, result.stdout + result.stderr
