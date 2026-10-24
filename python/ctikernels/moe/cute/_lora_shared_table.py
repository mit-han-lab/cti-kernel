# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Measured shared-projection choices, with a metadata-only fallback."""

SHARED_MODE_OVERRIDE = None
# BEGIN GENERATED SHARED TABLE
SHARED_TABLE = {
    "gb10": {
        (1, 512, 2048, 8): (
            (16, "dense", (16, 1, 16, 64, 4)),
            (32, "dense", (16, 1, 32, 64, 4)),
            (64, "dense", (16, 1, 16, 64, 4)),
            (128, "dense", (16, 1, 64, 64, 4)),
            (256, "dense", (16, 1, 32, 64, 4)),
            (512, "dense", (16, 1, 64, 64, 3)),
            (1024, "dense", (32, 1, 64, 64, 4)),
            (2048, "dense", (32, 1, 64, 64, 3)),
            (4096, "dense", (32, 1, 64, 128, 2)),
            (None, "library", None),
        ),
        (1, 512, 2048, 16): (
            (16, "dense", (16, 1, 16, 64, 2)),
            (32, "dense", (16, 1, 16, 64, 3)),
            (64, "dense", (16, 1, 16, 64, 4)),
            (128, "dense", (16, 1, 32, 128, 3)),
            (512, "dense", (16, 1, 64, 64, 4)),
            (1024, "dense", (32, 1, 64, 64, 3)),
            (2048, "dense", (64, 8, 64, 64, 2)),
            (4096, "dense", (32, 1, 64, 128, 2)),
            (None, "library", None),
        ),
        (1, 512, 2048, 32): (
            (16, "dense", (16, 4, 16, 64, 2)),
            (32, "dense", (16, 8, 16, 128, 2)),
            (64, "dense", (16, 8, 16, 64, 2)),
            (128, "dense", (32, 8, 16, 64, 2)),
            (256, "dense", (16, 8, 32, 64, 2)),
            (512, "dense", (16, 1, 32, 64, 3)),
            (1024, "dense", (32, 1, 64, 64, 4)),
            (2048, "dense", (64, 4, 64, 64, 2)),
            (4096, "dense", (16, 1, 64, 128, 4)),
            (None, "library", None),
        ),
        (1, 512, 2048, 64): (
            (64, "dense", (16, 8, 16, 64, 2)),
            (128, "dense", (16, 4, 32, 64, 2)),
            (256, "dense", (32, 4, 64, 64, 3)),
            (2048, "library", None),
            (4096, "dense", (16, 1, 64, 128, 4)),
            (None, "library", None),
        ),
        (1, 512, 2048, 128): (
            (16, "dense", (16, 8, 16, 64, 2)),
            (32, "dense", (16, 8, 16, 128, 2)),
            (64, "dense", (16, 8, 32, 128, 2)),
            (128, "dense", (16, 4, 32, 64, 3)),
            (None, "library", None),
        ),
        (2, 2048, 512, 8): (
            (32, "dense", (16, 16, 16, 64, 2)),
            (64, "dense", (16, 16, 16, 128, 2)),
            (128, "dense", (16, 16, 16, 64, 2)),
            (256, "dense", (32, 16, 32, 64, 2)),
            (512, "dense", (16, 1, 32, 64, 4)),
            (1024, "dense", (64, 6, 64, 64, 4)),
            (None, "cute", None),
        ),
        (2, 2048, 512, 16): (
            (16, "dense", (16, 32, 16, 64, 2)),
            (32, "dense", (16, 16, 16, 128, 2)),
            (64, "dense", (32, 16, 32, 64, 3)),
            (128, "dense", (64, 16, 32, 64, 3)),
            (256, "dense", (64, 16, 16, 64, 2)),
            (512, "dense", (64, 16, 64, 64, 2)),
            (1024, "dense", (64, 8, 64, 128, 2)),
            (None, "cute", None),
        ),
        (2, 2048, 512, 32): (
            (32, "dense", (16, 16, 16, 64, 2)),
            (64, "dense", (16, 32, 16, 64, 2)),
            (128, "dense", (64, 16, 32, 64, 2)),
            (256, "dense", (64, 8, 32, 64, 2)),
            (512, "dense", (64, 8, 64, 64, 2)),
            (1024, "dense", (64, 4, 64, 128, 2)),
            (2048, "library", None),
            (None, "cute", None),
        ),
        (2, 2048, 512, 64): (
            (32, "dense", (16, 16, 16, 64, 2)),
            (64, "dense", (32, 16, 16, 64, 2)),
            (128, "dense", (32, 8, 64, 64, 2)),
            (256, "dense", (64, 8, 64, 128, 2)),
            (512, "dense", (64, 4, 64, 64, 2)),
            (None, "library", None),
        ),
        (2, 2048, 512, 128): (
            (32, "dense", (16, 16, 16, 64, 2)),
            (64, "dense", (16, 8, 32, 64, 2)),
            (128, "dense", (32, 8, 64, 64, 2)),
            (None, "library", None),
        ),
    },
    "rtx_5090": {
        (1, 512, 2048, 8): (
            (32, "cute", None),
            (64, "dense", (16, 1, 16, 128, 4)),
            (256, "dense", (16, 1, 16, 128, 2)),
            (512, "dense", (16, 1, 16, 64, 3)),
            (1024, "dense", (16, 1, 32, 64, 2)),
            (2048, "dense", (32, 1, 64, 64, 3)),
            (None, "dense", (64, 1, 64, 64, 2)),
        ),
        (1, 512, 2048, 16): (
            (16, "cute", None),
            (32, "dense", (16, 1, 16, 64, 4)),
            (256, "dense", (16, 1, 16, 64, 3)),
            (512, "dense", (16, 1, 64, 64, 3)),
            (2048, "dense", (16, 1, 64, 64, 2)),
            (None, "dense", (32, 1, 64, 64, 2)),
        ),
        (1, 512, 2048, 32): (
            (16, "cute", None),
            (32, "dense", (16, 8, 16, 128, 2)),
            (64, "dense", (16, 8, 16, 64, 2)),
            (128, "dense", (16, 8, 16, 128, 2)),
            (256, "dense", (16, 8, 32, 64, 2)),
            (512, "dense", (16, 4, 64, 64, 2)),
            (1024, "dense", (32, 4, 32, 64, 2)),
            (2048, "dense", (64, 4, 64, 64, 2)),
            (None, "library", None),
        ),
        (1, 512, 2048, 64): (
            (32, "dense", (16, 4, 16, 64, 2)),
            (128, "dense", (16, 8, 16, 128, 2)),
            (256, "dense", (16, 8, 32, 64, 2)),
            (512, "dense", (32, 8, 64, 128, 2)),
            (1024, "dense", (32, 4, 64, 64, 2)),
            (2048, "dense", (64, 4, 64, 64, 3)),
            (None, "library", None),
        ),
        (1, 512, 2048, 128): (
            (32, "dense", (16, 8, 16, 64, 2)),
            (64, "dense", (16, 8, 16, 128, 2)),
            (256, "dense", (16, 8, 32, 64, 2)),
            (512, "dense", (32, 8, 64, 64, 2)),
            (None, "library", None),
        ),
        (2, 2048, 512, 8): (
            (16, "cute", None),
            (32, "dense", (16, 16, 16, 128, 2)),
            (64, "dense", (16, 16, 16, 128, 3)),
            (128, "dense", (16, 16, 16, 128, 2)),
            (256, "dense", (16, 16, 16, 64, 2)),
            (1024, "dense", (32, 16, 16, 64, 2)),
            (2048, "dense", (64, 4, 32, 64, 2)),
            (None, "dense", (64, 4, 64, 64, 2)),
        ),
        (2, 2048, 512, 16): (
            (16, "cute", None),
            (32, "dense", (32, 32, 16, 64, 2)),
            (64, "dense", (16, 32, 16, 64, 2)),
            (128, "dense", (32, 32, 16, 64, 2)),
            (256, "dense", (16, 16, 16, 128, 2)),
            (512, "dense", (32, 8, 32, 128, 2)),
            (1024, "dense", (64, 16, 32, 64, 2)),
            (2048, "dense", (64, 8, 64, 64, 2)),
            (None, "dense", (64, 4, 64, 64, 2)),
        ),
        (2, 2048, 512, 32): (
            (32, "dense", (16, 32, 16, 64, 2)),
            (64, "dense", (16, 16, 16, 64, 2)),
            (128, "dense", (16, 16, 16, 128, 2)),
            (256, "dense", (32, 16, 16, 64, 2)),
            (512, "dense", (64, 16, 32, 64, 2)),
            (2048, "dense", (64, 8, 64, 64, 2)),
            (None, "dense", (64, 4, 32, 128, 2)),
        ),
        (2, 2048, 512, 64): (
            (32, "dense", (16, 32, 16, 64, 2)),
            (128, "dense", (16, 16, 16, 64, 3)),
            (256, "dense", (32, 16, 16, 64, 2)),
            (512, "dense", (32, 8, 64, 64, 2)),
            (1024, "dense", (32, 4, 32, 64, 2)),
            (2048, "dense", (64, 4, 64, 128, 2)),
            (None, "library", None),
        ),
        (2, 2048, 512, 128): (
            (32, "dense", (16, 32, 16, 64, 2)),
            (64, "dense", (16, 16, 16, 64, 2)),
            (128, "dense", (16, 16, 32, 64, 2)),
            (256, "dense", (32, 16, 32, 64, 2)),
            (512, "dense", (32, 8, 64, 64, 2)),
            (None, "library", None),
        ),
    },
}
# END GENERATED SHARED TABLE
_MODES = ("cute", "split", "library", "dense")
_FALLBACK = {"rtx_pro_6000": "rtx_5090"}


def device_class(capability, sms):
    if tuple(capability) == (12, 1):
        return "gb10"
    if tuple(capability) == (12, 0):
        return "rtx_pro_6000" if sms >= 180 else "rtx_5090"
    return None


def shared_choice(rows, rank, k, n, slices, capability, sms):
    if SHARED_MODE_OVERRIDE is not None:
        if SHARED_MODE_OVERRIDE not in _MODES:
            raise ValueError(f"shared mode must be one of {_MODES}")
        return SHARED_MODE_OVERRIDE, None
    device = device_class(capability, sms)
    # The RTX PRO 6000 shares the RTX 5090's sm_120 table where it has no entry.
    table = SHARED_TABLE.get(device) or SHARED_TABLE.get(_FALLBACK.get(device), {})
    key = (slices, k, n, rank)
    if key not in table:
        # Resource guards also hold for smaller ranks (covered by CPU constructors).
        key = min(
            (key for key in table if key[:3] == (slices, k, n) and key[3] >= rank),
            key=lambda key: key[3],
            default=None,
        )
    for max_rows, mode, config in table.get(key, ()):
        if max_rows is None or rows <= max_rows:
            return mode, config
    # Untabulated shapes: decode rows take one CuTe launch; library GEMMs measured
    # fastest for rank 128 and very large batches; dense kernels otherwise.
    if rows <= 16:
        return "cute", None
    return ("library" if rank >= 128 or rows >= 4096 else "dense"), None
