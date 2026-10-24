# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Lazy, independent CuTe residual RMSNorm provider."""

from ._provider import available, get_compiled

__all__ = ["available", "get_compiled"]
