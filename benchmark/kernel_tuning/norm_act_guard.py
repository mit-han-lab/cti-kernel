# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Lifetime profiler exclusion for timing interpreters (never installed in probes)."""

import torch


def _forbid_profile(*args, **kwargs):
    raise RuntimeError("profiling is forbidden in a norm/activation timing process; use a probe")


def install_profiler_guard():
    """Install before provider setup, including protection against explicit start()."""
    if torch.autograd.profiler._is_profiler_enabled:
        raise RuntimeError("profiler already active in timing process")
    torch.profiler.profile.__enter__ = _forbid_profile
    torch.profiler.profile.start = _forbid_profile
    torch.autograd.profiler.profile.__enter__ = _forbid_profile
    # Also cover the low-level Kineto API and explicit start/prepare_trace calls.
    torch.profiler._KinetoProfile.prepare_trace = _forbid_profile


def profiler_clean():
    if (
        torch.profiler.profile.__enter__ is not _forbid_profile
        or torch.profiler.profile.start is not _forbid_profile
        or torch.autograd.profiler.profile.__enter__ is not _forbid_profile
        or torch.profiler._KinetoProfile.prepare_trace is not _forbid_profile
        or torch.autograd.profiler._is_profiler_enabled
    ):
        raise RuntimeError("timing process profiler guard is not intact")
    return True
