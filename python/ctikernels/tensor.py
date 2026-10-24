# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 sgl-project
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
# Adapted from Mini-SGLang 9a91cfafe754aa85daee49998176275667eb58f2
# (`python/minisgl/kernel/tensor.py`); see THIRD_PARTY_NOTICES.

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_test_tensor_module() -> Module:
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    return _load_test_tensor_module().test(x, y)


def weak_tensor_alias() -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the provider's weak-reference tensor alias used to keep CUDA graph
    inputs addressable without owning their storage."""
    try:
        from sgl_kernel import weak_ref_tensor
    except (ImportError, OSError) as error:
        raise RuntimeError("piecewise CUDA graphs require sgl_kernel.weak_ref_tensor") from error
    if not callable(weak_ref_tensor):
        raise RuntimeError("piecewise CUDA graphs require callable sgl_kernel.weak_ref_tensor")
    return weak_ref_tensor
