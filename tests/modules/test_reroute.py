# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import logging
import pickle

import pytest

from ctikernels import _reroute


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(_reroute, "_WARNED", set())


def test_provider_is_its_name_and_keeps_the_reason():
    plain = _reroute.Provider("cute-fp8")
    assert plain == "cute-fp8" and plain.reason is None and repr(plain) == "'cute-fp8'"
    moved = _reroute.reroute("BlockFP8Linear", "triton-fp8", "N % 128 != 0")
    assert moved == "triton-fp8" and isinstance(moved, str)
    assert moved.reason == "N % 128 != 0"
    assert "N % 128 != 0" in repr(moved)
    back = pickle.loads(pickle.dumps(moved))
    assert back == moved and back.reason == moved.reason


def test_logs_once_per_op_and_reason(caplog):
    caplog.set_level(logging.WARNING, logger=_reroute.__name__)
    for _ in range(3):
        _reroute.reroute("Op", "fallback", "reason a")
    _reroute.reroute("Op", "fallback", "reason b")
    _reroute.reroute("Other", "fallback", "reason a")
    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "Op rerouted to fallback: reason a",
        "Op rerouted to fallback: reason b",
        "Other rerouted to fallback: reason a",
    ]


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "strict"])
def test_engine_checks_raise(monkeypatch, value, caplog):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", value)
    assert _reroute.engine_checks()
    with pytest.raises(
        _reroute.RerouteError, match="Op: CTI_ENGINE_CHECKS forbids the reroute to fb: why"
    ):
        _reroute.reroute("Op", "fb", "why")
    assert not caplog.records


@pytest.mark.parametrize("value", ["", "0", "false", "No", " off "])
def test_engine_checks_off(monkeypatch, value):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", value)
    assert not _reroute.engine_checks()
    assert _reroute.reroute("Op", "fb", "why").reason == "why"


def test_reason_is_required():
    with pytest.raises(ValueError):
        _reroute.reroute("Op", "fb", "")
