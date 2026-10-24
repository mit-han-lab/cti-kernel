# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Op test directories (tests/<op>/) in environments without the GPU stack.

``test_api.py`` is CPU-only; every other test module there is GPU-only and may import Triton or
CuTe at module level. Where Triton is missing (the CPU job), those modules are not collected;
the CPU job also deselects ``-m gpu``, so its collected ids do not depend on this hook.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_TIERS = {"benchmarks", "ci", "gpu", "modules", "multi_gpu", "multi_node", "quality", "support"}
_GPU_STACK = importlib.util.find_spec("triton") is not None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    if _GPU_STACK or collection_path.suffix != ".py" or collection_path.name == "test_api.py":
        return None
    parent = collection_path.parent
    if parent.parent == Path(__file__).parent and parent.name not in _TIERS:
        return collection_path.name.startswith("test_") or None
    return None
