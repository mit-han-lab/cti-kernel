# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Head-local launch policies; GPU tuning must precede qualification.

The table may carry per-geometry rows; None matches any local head geometry.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    value_tile: int = 128
    warps: int = 8
    prefetch: bool = False
    lanes: int = 4
    sections: bool = False
    snapshot_controls: bool = False
    speculative_ring: bool = True
    immediate: bool = False
    provider: str = "cute"
    trusted: bool = True
    flush_always: bool = False

    # Transaction phases (verify/commit/...): warps for full-head CTAs and for
    # Q > warps (one staging chunk), and whether more full-head CTAs than SMs
    # split into 64-row tiles with two co-resident CTAs per SM.
    phase_warps: int = 16
    # Launch bound (minimum resident CTAs per SM) of this specialization.
    min_blocks: int = 1
    split_oversubscribed: bool = True

    def __post_init__(self):
        if self.phase_warps not in (4, 8, 16) or self.min_blocks not in (1, 2, 3):
            raise ValueError("invalid phase warps or min_blocks")
        if not isinstance(self.trusted, bool):
            raise ValueError("trusted must be bool")
        if not isinstance(self.flush_always, bool):
            raise ValueError("flush_always must be bool")
        if self.flush_always and (self.immediate or self.provider == "product"):
            raise ValueError("flush_always requires delayed CuTe")
        if not isinstance(self.immediate, bool):
            raise ValueError("immediate must be bool")
        if self.provider not in ("cute", "product"):
            raise ValueError("provider must be cute or product")
        if self.provider == "product" and not self.immediate:
            raise ValueError("product requires immediate=True")
        if self.value_tile not in (32, 64, 128):
            raise ValueError("value_tile must be 32, 64 or 128")
        if self.warps not in (4, 8, 16) or self.lanes not in (4, 8):
            raise ValueError("CuTe replay requires 4/8 warps and 4/8 lanes per row")
        if self.value_tile < 128 and self.lanes != 4:
            raise ValueError("split value tiles require 4 lanes per row")
        if self.value_tile < 128 and self.snapshot_controls:
            raise ValueError("snapshot_controls requires value_tile=128")
        if not isinstance(self.speculative_ring, bool):
            raise ValueError("speculative_ring must be bool")

    def shared_bytes(self, window, *, speculative=None):
        """16-byte-aligned allocations in Step, including optional state staging."""
        if self.immediate:
            tile, cols = self.value_tile, 256 + self.value_tile
            sizes = [
                16,
                16 if self.sections else 0,
                4 * cols * 2,
                (256 + 2 * tile) * 4,
                32,
                8,
                512,
                cols * 8,
                tile * 2,
            ]
            return sum((size + 15) // 16 * 16 for size in sizes)
        spec = self.speculative_ring if speculative is None else speculative
        tile, cols = self.value_tile, 256 + self.value_tile
        sizes = [
            16,
            tile * 128 * 4 if window <= 8 else (16 if self.sections else 0),
            (window + 4) * cols * 2,
            window * (256 + 2 * tile) * 4,
            window * 8 * 4,
            window * 2 * 4,
            128 * 4,
            (window + 3) * 2 * 4 if spec else 0,
            cols * 4 * 2,
            tile * 2,
        ]
        return sum((size + 15) // 16 * 16 for size in sizes)

    def stages_ring(self, window):
        return (
            not self.immediate and self.speculative_ring and self.shared_bytes(window) <= 99 * 1024
        )


# Ceilings: B=1, B=2, B=3..4, B=5..8, B>8 (also empty batches).
BATCH_BUCKETS = (1, 2, 4, 8, None)
_PRODUCT = {"provider": "product", "immediate": True}
_TILES = {1: 32, 2: 32, 4: 64, 8: 128, None: 128}
# Measured policy: a CuTe
# specialization enters only where it beats #25 by more than eps with all fixed costs
# counted. Product entries keep the immediate tile so explicit modes stay tuned.
POLICY = {
    **{
        (device, None, bucket): {
            **_PRODUCT,
            "value_tile": _TILES[bucket],
            "speculative_ring": False,
        }
        for device in ("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000")
        for bucket in BATCH_BUCKETS
    },
    # GB10: delayed beats #25 at every measured B (1-16), immediate does not.
    **{
        ("NVIDIA GB10", None, bucket): {
            "immediate": False,
            "value_tile": 128,
            "speculative_ring": True,
        }
        for bucket in BATCH_BUCKETS
    },
}


def select(name, batch=0, *, mode="auto", config=None, geometry=None, window=8):
    """Replay selection point. A product entry routes step to #25.

    Explicit Config fields override the table; explicit mode forces the CuTe
    specialization (immediate or delayed) even where the table routes to #25.
    """
    if mode not in ("auto", "immediate", "delayed"):
        raise ValueError("mode must be auto, immediate or delayed")
    if config is None:
        bucket = next((n for n in BATCH_BUCKETS if n is not None and 0 < batch <= n), None)
        fields = next(
            (
                v
                for geom in ((geometry, None) if geometry is not None else (None,))
                for (prefix, g, b), v in POLICY.items()
                if g == geom and b == bucket and (name or "").startswith(prefix)
            ),
            {},
        )
        config = replace(select_phases(name), **fields)
        if window > 16 and not config.immediate:
            config = replace(config, provider="product", immediate=True)
    if mode != "auto":
        config = replace(config, immediate=mode == "immediate", provider="cute", flush_always=False)
    return config


def decode_route(name, batch, *, mode="auto", config=None, geometry=None, window=8):
    """Replay routing for one (device, geometry, batch): "product", "immediate" or "delayed"."""
    ringless = all_immediate(name, mode=mode, config=config, geometry=geometry, window=window)
    config = select(name, batch, mode=mode, config=config, geometry=geometry, window=window)
    if config.provider == "product":
        # #25 cannot drain a ring left by another batch bucket.
        return "product" if ringless else "immediate"
    return "immediate" if config.immediate else "delayed"


def all_immediate(name, *, mode="auto", config=None, geometry=None, window=8):
    return all(
        select(name, b, mode=mode, config=config, geometry=geometry, window=window).immediate
        for b in (1, 2, 4, 8, 9)
    )


DISPATCH = {
    "NVIDIA GeForce RTX 5090": Config(),
    "NVIDIA RTX PRO 6000": Config(),
    # DRAM-bound at B >= 2: 64-row splits re-read the prologue for no gain.
    "NVIDIA GB10": Config(split_oversubscribed=False),
}


def select_phases(name):
    return next(
        (cfg for prefix, cfg in DISPATCH.items() if (name or "").startswith(prefix)), Config()
    )


# Measured policies keyed by device-name prefix: (batch bucket, Q, window) entries
# that run the delayed commit; every other key (including B >= 5 and window 16,
# which were not measured) stays eager. Buckets are 1, 2, 3, 4 and 5 (B >= 5).
# Rule (R3c receipts): same-process eager/delayed A/B over K-round graphs, 20 pairs;
# delayed when the eager/delayed median ratio is >= 1 for both (16,32) and (16,48)
# (GB10: (16,32) only) and > 1 + eps for at least one; rows whose fold period is
# one round (delayed does the same work) never qualify.
_DELAYED = {
    "NVIDIA GeForce RTX 5090": (
        (1, 4, 4),
        (1, 4, 8),
        (1, 5, 4),
        (1, 5, 8),
        (1, 9, 8),
        (2, 4, 4),
        (2, 4, 8),
        (2, 5, 4),
        (2, 5, 8),
        (2, 6, 4),
        (2, 8, 8),
        (2, 9, 8),
        (3, 4, 4),
        (3, 4, 8),
        (3, 5, 4),
        (3, 5, 8),
        (3, 6, 4),
        (3, 6, 8),
        (3, 8, 8),
        (3, 9, 8),
        (4, 4, 4),
        (4, 4, 8),
        (4, 5, 4),
        (4, 5, 8),
        (4, 6, 4),
        (4, 6, 8),
        (4, 8, 8),
        (4, 9, 8),
    ),
    "NVIDIA RTX PRO 6000": (
        (1, 4, 4),
        (1, 4, 8),
        (1, 5, 4),
        (1, 5, 8),
        (1, 9, 8),
        (2, 4, 4),
        (2, 4, 8),
        (2, 5, 4),
        (2, 5, 8),
        (2, 6, 4),
        (2, 9, 8),
        (3, 4, 4),
        (3, 4, 8),
        (3, 5, 4),
        (3, 5, 8),
        (3, 6, 4),
        (3, 6, 8),
        (3, 8, 8),
        (3, 9, 8),
        (4, 4, 4),
        (4, 4, 8),
        (4, 5, 4),
        (4, 5, 8),
        (4, 6, 4),
        (4, 8, 8),
        (4, 9, 8),
    ),
    "NVIDIA GB10": (
        (1, 4, 4),
        (1, 4, 8),
        (1, 5, 4),
        (1, 5, 8),
        (1, 6, 4),
        (1, 6, 8),
        (1, 8, 8),
        (1, 9, 8),
        (2, 4, 4),
        (2, 4, 8),
        (2, 5, 4),
        (2, 5, 8),
        (2, 6, 4),
        (2, 6, 8),
        (2, 8, 8),
        (2, 9, 8),
        (3, 4, 4),
        (3, 4, 8),
        (3, 5, 4),
        (3, 5, 8),
        (3, 6, 4),
        (3, 6, 8),
        (3, 8, 8),
        (3, 9, 8),
        (4, 4, 4),
        (4, 4, 8),
        (4, 5, 4),
        (4, 5, 8),
        (4, 6, 4),
        (4, 6, 8),
        (4, 8, 8),
        (4, 9, 8),
    ),
}
COMMIT_POLICY = {prefix: dict.fromkeys(_DELAYED.get(prefix, ()), "delayed") for prefix in DISPATCH}
# Per-geometry rows (device, (Hk, Hv)) replace the shared table for that local head
# geometry: TP2 per-rank geometries (8,16) (Qwen3.5-9B) and (8,24) (Qwen3.8-27B).
# Rule, measured at that geometry alone with eager and delayed on the same ring
# storage: delayed when the eager/delayed median ratio is > 1 + eps and the fold
# period exceeds one round. Windows 4 and 8, Q 4/5/6/8/9/12/16, B 1-4.
_DELAYED_GEOMETRY = {
    ("NVIDIA GB10", (8, 16)): (
        (2, 4, 8),
        (2, 5, 8),
        (3, 4, 4),
        (3, 4, 8),
        (3, 5, 4),
        (3, 5, 8),
        (3, 6, 4),
        (4, 4, 4),
        (4, 4, 8),
        (4, 5, 4),
        (4, 5, 8),
        (4, 6, 4),
        (4, 6, 8),
        (4, 8, 8),
        (4, 9, 8),
    ),
    ("NVIDIA GB10", (8, 24)): (
        (2, 4, 8),
        (2, 5, 8),
        (2, 6, 8),
        (3, 4, 4),
        (3, 4, 8),
        (3, 5, 4),
        (3, 5, 8),
        (3, 6, 4),
        (3, 6, 8),
        (3, 9, 8),
        (4, 4, 4),
        (4, 4, 8),
        (4, 5, 4),
        (4, 5, 8),
        (4, 6, 4),
        (4, 6, 8),
    ),
    ("NVIDIA GeForce RTX 5090", (8, 16)): (),
    ("NVIDIA RTX PRO 6000", (8, 16)): (),
    ("NVIDIA RTX PRO 6000", (8, 24)): (),
    ("NVIDIA GeForce RTX 5090", (8, 24)): (
        (4, 4, 8),
        (4, 5, 8),
    ),
}
COMMIT_POLICY_GEOMETRY = {
    key: dict.fromkeys(rows, "delayed") for key, rows in _DELAYED_GEOMETRY.items()
}


def commit_mode(name, batch, q, window, geometry=None):
    """Return the qualified policy, defaulting to eager for every missing key."""
    bucket = min(max(batch, 1), 5)
    policy = None
    if geometry is not None:
        policy = next(
            (
                p
                for (prefix, g), p in COMMIT_POLICY_GEOMETRY.items()
                if g == tuple(geometry) and (name or "").startswith(prefix)
            ),
            None,
        )
    if policy is None:
        policy = next(
            (p for prefix, p in COMMIT_POLICY.items() if (name or "").startswith(prefix)), {}
        )
    mode = (
        policy(bucket, q, window) if callable(policy) else policy.get((bucket, q, window), "eager")
    )
    if mode not in ("eager", "delayed"):
        raise ValueError("commit policy must return eager or delayed")
    return mode
