# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import pytest


@pytest.fixture(autouse=True)
def norm_act_timing_process(request, monkeypatch):
    """Model a guarded timing interpreter without changing other benchmark tests."""
    if not request.module.__name__.split(".")[-1].startswith("test_norm_act"):
        return
    import torch

    from benchmark.kernel_tuning.norm_act_guard import install_profiler_guard

    for owner, name in (
        (torch.profiler.profile, "__enter__"),
        (torch.profiler.profile, "start"),
        (torch.autograd.profiler.profile, "__enter__"),
        (torch.profiler._KinetoProfile, "prepare_trace"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    install_profiler_guard()
