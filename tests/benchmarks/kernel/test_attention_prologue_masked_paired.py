# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU checks of the masked-store pairing script; no CUDA workloads."""

import math
import sys
from types import SimpleNamespace

import pytest

from benchmark.kernel_tuning import attention_prologue_masked_paired as paired

from ...support.child import run_child


def test_interval_is_the_geometric_mean_with_a_log_t_interval():
    ratio, low, high = paired._interval([0.5, 2.0, 1.0, 1.0])
    assert ratio == pytest.approx(1.0)
    half = 1.96 * math.log(2) * math.sqrt(2 / 3) / 2
    assert (low, high) == pytest.approx((math.exp(-half), math.exp(half)))


def test_importing_the_script_imports_no_cutlass():
    code = (
        "import sys, benchmark.kernel_tuning.attention_prologue_masked_paired; "
        "print('cutlass' in sys.modules)"
    )
    out = run_child([sys.executable, "-c", code], timeout=600, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("baseline", [None, "baseline.py"])
def test_candidates_select_masking_and_adapt_legacy_keys(monkeypatch, baseline):
    loaded, instances = [], []
    current = SimpleNamespace(Kernel=lambda *key: ("current", key))

    def module(path, name):
        loaded.append((path, name))
        return SimpleNamespace(Kernel=lambda *key: ("legacy", key))

    class Launcher:
        def __init__(self, device, kernels, variant):
            self.kernels = kernels
            self.variant = variant
            self.compiled = {}
            instances.append(self)

        def __call__(self, *args, masked=False, **kwargs):
            return args, masked, kwargs

    monkeypatch.setattr(paired, "_module", module)
    monkeypatch.setattr(paired._prologue, "Launcher", Launcher)
    variant = paired.GEOMETRIES["hd256"][2]
    candidates = paired._launchers("cpu", current, variant, baseline)
    assert list(candidates) == ["base", "base2", "masked"]
    assert len(instances) == 3
    assert len({id(launcher.compiled) for launcher in instances}) == 3
    assert all(launcher.variant == variant for launcher in instances)
    for name, candidate in candidates.items():
        assert candidate("inputs", cache="e4m3") == (
            ("inputs",),
            name == "masked",
            {"cache": "e4m3"},
        )
    key = (16, 4, 1, 1, 1, 4, False, *variant, "e4m3", False)
    expected = ("legacy", key[:-1]) if baseline else ("current", key)
    assert candidates["base"].kernels.Kernel(*key) == expected
    assert candidates["base2"].kernels.Kernel(*key) == expected
    masked_key = (*key[:-1], True)
    assert instances[2].kernels.Kernel(*masked_key) == ("current", masked_key)
    assert loaded == (
        [(baseline, "prologue_base"), (baseline, "prologue_base2")] if baseline else []
    )
