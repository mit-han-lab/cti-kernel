# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Print every architecture-guard count in ledger.toml form.

Run from the repository root: ``python -m tests.quality.architecture.measure``. Use it to lower
ledger entries after fixing violations; entries may only go down.
"""

from __future__ import annotations

from . import test_layers, test_layout, test_owners, test_public_api, test_selection

GUARDS = {
    "layers": test_layers,
    "layout": test_layout,
    "owners": test_owners,
    "selection": test_selection,
    "public_api": test_public_api,
}


def main() -> None:
    for guard, module in GUARDS.items():
        print(f"[{guard}]")
        for key, value in module.measure().items():
            print(f"{key} = {value}")
        print()


if __name__ == "__main__":
    main()
