# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CuTe DSL small-M provider; the DSL is imported only when compiling a supported call."""

from ._provider import available, get_compiled, mtiles_for
from .variants import TABLES, VARIANTS, smallm_bf16_linear

__all__ = ["TABLES", "VARIANTS", "available", "get_compiled", "mtiles_for", "smallm_bf16_linear"]
