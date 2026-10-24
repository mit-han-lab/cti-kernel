# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

import logging

import pytest

from ctikernels import _norm_act_reroute as route


def test_provider_is_a_string_with_reason():
    provider = route.Provider("triton", "unavailable")
    assert provider == "triton" and hash(provider) == hash("triton")
    assert provider.reason == "unavailable"
    assert route.Provider("cute").reason is None


def test_warning_once_per_operation_and_reason(monkeypatch, caplog):
    monkeypatch.delenv("CTI_ENGINE_CHECKS", raising=False)
    monkeypatch.setattr(route, "_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=route.__name__):
        for _ in range(2):
            assert route.reroute("rms_norm", "triton", "unavailable").reason == "unavailable"
        route.reroute("rms_norm", "cute", "unavailable")
        route.reroute("rms_norm_", "triton", "unavailable")
        route.reroute("rms_norm", "triton", "wrong version")
    assert len(caplog.records) == 3


@pytest.mark.parametrize(
    "value", ["", "0", "false", "no", "off", "FALSE", "OFF", "1", "yes", "anything"]
)
def test_checks_read_environment_on_every_call(monkeypatch, value):
    monkeypatch.setenv("CTI_ENGINE_CHECKS", value)
    enabled = value.lower() not in ("", "0", "false", "no", "off")
    assert route.engine_checks() == enabled
    if enabled:
        with pytest.raises(route.RerouteError, match="rms_norm.*triton.*unavailable"):
            route.reroute("rms_norm", "triton", "unavailable")
    else:
        assert route.reroute("rms_norm", "triton", "unavailable") == "triton"
