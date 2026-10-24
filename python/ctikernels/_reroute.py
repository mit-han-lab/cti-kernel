# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Visible provider reroutes, shared by every op.

An op that cannot run its normal provider on some input and runs another one instead calls
``reroute(op, chosen, reason)`` and stores the result as its ``.provider``. The result compares
equal to the provider name and carries the reason. The first reroute per (op, reason) logs one
warning. With ``CTI_ENGINE_CHECKS`` set (any value except "", "0", "false", "no", "off"), a
reroute raises ``RerouteError`` instead, so qualification runs catch every silent slow path.
"""

import logging
import os
from threading import Lock

__all__ = ["Provider", "RerouteError", "engine_checks", "reroute"]

_LOG = logging.getLogger(__name__)
_LOCK = Lock()
_WARNED: set[tuple[str, str]] = set()
_OFF = ("", "0", "false", "no", "off")


class RerouteError(RuntimeError):
    """A reroute happened while CTI_ENGINE_CHECKS is on."""


class Provider(str):
    """A provider name; ``reason`` is why the op was rerouted to it, else None."""

    reason: str | None

    def __new__(cls, name: str, reason: str | None = None) -> "Provider":
        result = super().__new__(cls, name)
        result.reason = reason
        return result

    def __repr__(self) -> str:
        base = str.__repr__(self)
        return base if self.reason is None else f"{base} (rerouted: {self.reason})"

    def __reduce__(self):
        return Provider, (str(self), self.reason)


def engine_checks() -> bool:
    """Whether CTI_ENGINE_CHECKS is on (read on every call, so tests and engines may toggle it)."""
    return os.environ.get("CTI_ENGINE_CHECKS", "").strip().lower() not in _OFF


def reroute(op: str, chosen: str, reason: str) -> Provider:
    """Record that ``op`` runs ``chosen`` because of ``reason``.

    Raises RerouteError under CTI_ENGINE_CHECKS; otherwise logs once per (op, reason) and
    returns the provider to store as ``.provider``. Call it before any launch.
    """
    if not reason:
        raise ValueError(f"{op}: a reroute needs a reason")
    if engine_checks():
        raise RerouteError(f"{op}: CTI_ENGINE_CHECKS forbids the reroute to {chosen}: {reason}")
    key = (op, reason)
    with _LOCK:
        first = key not in _WARNED
        _WARNED.add(key)
    if first:
        _LOG.warning("%s rerouted to %s: %s", op, chosen, reason)
    return Provider(chosen, reason)
