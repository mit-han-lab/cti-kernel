# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Runtime qualification shared by the norm and activation inference providers."""

import importlib.metadata
import os
from functools import lru_cache

ALLOWED = ("0.7.0.post1",)


@lru_cache(maxsize=1)
def installed_version():
    try:
        return importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@lru_cache(maxsize=1)
def importable():
    try:
        importlib.import_module("flashinfer")
    except Exception:  # noqa: BLE001 -- optional provider import boundary
        return False
    return True


def unusable_reason():
    if not importable():
        return "FlashInfer is not importable"
    if installed_version() not in ALLOWED and os.environ.get("CTI_FLASHINFER_ALLOW_ANY") != "1":
        return f"FlashInfer version {installed_version()} is not the pin {ALLOWED[0]}"
    return None


def allowed():
    return unusable_reason() is None


def require_allowed():
    if not allowed():
        raise RuntimeError(
            f"{unusable_reason()}; allowed versions: {', '.join(ALLOWED)}. "
            "Set CTI_FLASHINFER_ALLOW_ANY=1 to explicitly bypass version qualification."
        )
