# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Shared launch and kernel-entry switch for K1 PDL A/B compilation."""

# Every consumer waits before touching memory, so dependents can trigger early.
PDL = True
