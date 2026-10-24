# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Optional reference implementations (vLLM, SGLang, FlashInfer), loaded only if installed.

A benchmark asks for a market by name and gets the module or the reason it is missing; tables
print ``n/a`` with that reason instead of failing. Nothing here is vendored.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

# market -> module holding its kernels
MODULES = {
    "vllm": "vllm._custom_ops",
    "sglang": "sgl_kernel",
    "flashinfer": "flashinfer",
}


@dataclass(frozen=True)
class Market:
    name: str
    module: ModuleType | None
    reason: str = ""

    @property
    def version(self) -> str:
        root = importlib.import_module(self.module.__name__.split(".")[0]) if self.module else None
        return getattr(root, "__version__", "?") if root else "n/a"


def load(name: str) -> Market:
    try:
        return Market(name, importlib.import_module(MODULES[name]))
    except Exception as error:  # noqa: BLE001 - any import failure makes the market unavailable
        return Market(name, None, f"{type(error).__name__}: {error}")
