# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU regressions for the isolated comparator; no SGLang implementation vendored."""

import importlib
import sys
from types import SimpleNamespace

from benchmark.kernel_tuning import nvfp4_sglang as comparator


def test_excluded_child_does_not_hide_real_parent(tmp_path, monkeypatch):
    package = tmp_path / "sglang/kernels/ops/gemm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("loaded_from_source = True\n")
    consumer = tmp_path / "sglang/consumer.py"
    consumer.write_text("from sglang.kernels.ops.gemm.excluded import unused\n")
    monkeypatch.setattr(comparator, "REAL_MODULES", ("sglang.kernels.ops.gemm", "sglang.consumer"))
    monkeypatch.setattr(comparator, "_STUBS", set())
    monkeypatch.setattr(
        comparator,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        raising=False,
    )
    # These imports are intentionally process-local in production. Isolate the fixture.
    before = set(sys.modules)
    try:
        comparator._prepare_imports(tmp_path)
        gemm = importlib.import_module("sglang.kernels.ops.gemm")
        assert gemm.loaded_from_source
        assert gemm.__file__ == str(package / "__init__.py")
        importlib.import_module("sglang.consumer")
    finally:
        for name in set(sys.modules) - before:
            if name.split(".")[0] in ("sglang", "sgl_kernel", "transformers"):
                del sys.modules[name]
