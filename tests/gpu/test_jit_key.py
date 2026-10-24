# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Built tvm-ffi modules are reused across install paths and future mtimes (JIT-KEY)."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import time

import pytest

from ..support.child import run_child

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "python" / "ctikernels"

# Child process: import ctikernels from one tree copy, count builds, run each module once.
_CHILD = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import torch
from ctikernels import utils
built = []
build = utils._build_module
def counted(kind, name, build_dir, **kwargs):
    built.append(name)
    return build(kind, name, build_dir, **kwargs)
utils._build_module = counted
from ctikernels import radix
from ctikernels.index import indexing
from ctikernels.sampling._request_local import sample_requests
from ctikernels.store import store_cache
assert utils.__file__.startswith(sys.argv[1])
start = time.perf_counter()
radix.warm_up()
a = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
assert radix.fast_compare_key(a, torch.tensor([1, 2, 9], dtype=torch.int64)) == 2
weights = torch.randn((11, 256), device="cuda", dtype=torch.bfloat16)
indices = torch.tensor([10, 0, 5, 5, 1], device="cuda", dtype=torch.int32)
assert torch.equal(indexing(weights, indices), weights.index_select(0, indices.long()))
k_cache = torch.zeros((8, 2, 64), device="cuda", dtype=torch.bfloat16)
v_cache = torch.zeros_like(k_cache)
k = torch.randn((3, 128), device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
slots = torch.tensor([6, -1, 2], device="cuda", dtype=torch.int32)
store_cache(k_cache, v_cache, slots, k, v)
assert torch.equal(k_cache[6].view(-1), k[0]) and torch.equal(v_cache[2].view(-1), v[2])
assert int(k_cache[[0, 1, 3, 4, 5, 7]].abs().sum()) == 0
probabilities = torch.zeros((3, 1000), device="cuda", dtype=torch.float32)
probabilities[torch.arange(3), torch.tensor([7, 999, 0])] = 1.0
sampled = sample_requests(probabilities, (1, 2, 3), (0, 0, 0))
assert sampled.tolist() == [7, 999, 0]
torch.cuda.synchronize()
print(json.dumps({"built": built, "seconds": time.perf_counter() - start}))
"""


def _run(python_dir: pathlib.Path, cache: pathlib.Path) -> dict:
    env = {**os.environ, "TVM_FFI_CACHE_DIR": str(cache)}
    env.pop("CTIKERNELS_JIT_REBUILD", None)
    result = run_child([sys.executable, "-c", _CHILD, str(python_dir)], env=env, timeout=1800)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_modules_load_from_moved_tree_with_future_mtimes_without_rebuild(tmp_path):
    pytest.importorskip("flashinfer")
    trees = []
    for name in ("a", "b"):
        python_dir = tmp_path / name / "python"
        shutil.copytree(
            _PACKAGE, python_dir / "ctikernels", ignore=shutil.ignore_patterns("__pycache__")
        )
        trees.append(python_dir)
    future = time.time() + 10 * 365 * 24 * 3600
    for path in trees[1].rglob("*"):
        os.utime(path, (future, future))
    cache = tmp_path / "cache"

    first = _run(trees[0], cache)
    assert len(first["built"]) == 4, first
    moved = _run(trees[1], cache)
    assert moved["built"] == [], moved
    again = _run(trees[0], cache)
    assert again["built"] == [], again
    print(
        f"JIT-KEY load+run: cold {first['seconds']:.2f}s, moved+future {moved['seconds']:.2f}s, warm {again['seconds']:.2f}s"
    )
