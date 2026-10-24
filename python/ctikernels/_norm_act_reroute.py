# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Stand-in to be replaced by ctikernels._reroute once it lands (same interface)."""

import logging
import os

_LOG = logging.getLogger(__name__)
_WARNED = set()


class Provider(str):
    """A provider name carrying the reason for a reroute, if any."""

    def __new__(cls, chosen, reason=None):
        result = super().__new__(cls, chosen)
        result.reason = reason
        return result


class RerouteError(RuntimeError):
    """Engine checks forbid a provider reroute."""


def engine_checks():
    return os.environ.get("CTI_ENGINE_CHECKS", "").lower() not in ("", "0", "false", "no", "off")


def reroute(op, chosen, reason):
    if engine_checks():
        raise RerouteError(f"{op} rerouted to {chosen}: {reason}")
    key = (op, reason)
    if key not in _WARNED:
        _WARNED.add(key)
        _LOG.warning("%s rerouted to %s: %s", op, chosen, reason)
    return Provider(chosen, reason)
