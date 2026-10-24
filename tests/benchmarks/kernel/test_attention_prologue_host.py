# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks for the eager host benchmark's timing and checksum protocol."""

import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmark.kernel_tuning import attention_prologue_host as host

from ...support.child import run_child


def test_import_and_help_need_no_torch_numpy_or_cutlass():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'cutlass'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from benchmark.kernel_tuning.attention_prologue_host import main
main(['--help'])
"""
    result = run_child([sys.executable, "-c", code], timeout=600, check=True)
    for option in ("--cache", "--masked", "--geometry", "--steps", "--json"):
        assert option in result.stdout


def test_summary_and_positive_steps():
    assert host.summary(list(range(100))) == {"n": 100, "median": 49.5, "p10": 10, "p90": 90}
    with pytest.raises(SystemExit):
        host.main(["--steps", "0"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_checksum_hashes_logical_tensor_bytes(dtype):
    tensor = torch.arange(12).to(dtype).reshape(3, 4)[:, :2]
    digest = hashlib.sha256()
    host.hash_tensor(digest, tensor)
    expected = bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())
    assert digest.hexdigest() == hashlib.sha256(expected).hexdigest()


def test_real_model_geometries():
    assert host.GEOMETRIES == {
        "qwen3": (32, 8, 128, 128, False),
        "qwen35": (16, 4, 256, 64, True),
    }


def test_step_shares_metadata_and_times_only_the_op(monkeypatch):
    events, calls = [], []
    times = iter(range(1000))

    def clock():
        events.append("clock")
        return next(times) * 1000

    class Pool:
        def squeeze(self, dim):
            assert dim == 1
            events.append("squeeze")
            return object()

    class Source:
        def __getitem__(self, key):
            events.append("row")
            return self

        def clone(self):
            events.append("clone")
            return object()

    class Operation:
        masked = True

        def __call__(self, *args):
            calls.append(args)
            events.append("call")
            return object()

    metadata = []

    def step_metadata(device, tokens, step, masked):
        assert masked
        pair = (object(), object())
        metadata.append((tokens, pair))
        return pair

    monkeypatch.setattr(host, "step_metadata", step_metadata)
    monkeypatch.setattr(host.time, "perf_counter_ns", clock)
    monkeypatch.setattr(
        host,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: events.append("sync"))),
        raising=False,
    )
    layers = [(object(), object(), Pool(), Pool(), Source()) for _ in range(41)]
    samples = host.run_step(Operation(), layers, object(), "cuda", 0)
    assert samples == [1] * 41
    assert events == ["row", "clone", "squeeze", "squeeze", "clock", "call", "clock"] * 41 + [
        "sync"
    ]
    assert [entry[0] for entry in metadata] == [101, 64]
    for index, call in enumerate(calls):
        positions, indices = metadata[int(index >= 36)][1]
        assert call[4] is positions and call[7] is indices


def test_checksum_sequence_is_fixed_and_covers_every_output_and_cache(monkeypatch):
    seen, steps = [], []

    def run_step(op, layers, table, device, step, *, outputs):
        steps.append(step)
        outputs.extend((f"q{step}-0", f"q{step}-1"))

    monkeypatch.setattr(host, "run_step", run_step)
    monkeypatch.setattr(host, "hash_tensor", lambda digest, tensor: seen.append(tensor))
    layers = [(None, None, "k0", "v0", None), (None, None, "k1", "v1", None)]
    host.checksum(None, layers, None, None)
    assert steps == list(range(6))
    assert seen == [f"q{step}-{layer}" for step in range(6) for layer in range(2)] + [
        "k0",
        "v0",
        "k1",
        "v1",
    ]
