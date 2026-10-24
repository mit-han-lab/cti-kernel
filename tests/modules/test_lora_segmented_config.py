# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""CPU-only tuning override checks."""

import importlib.util
import sys

import pytest

from ctikernels.linear import _lora_segmented as impl
from ctikernels.linear import last_config


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, (None, None, None, None)),
        ("auto,auto,auto", (None, None, None, None)),
        (" 16, 32,256 ", (16, 32, 256, None)),
        ("128,16,64", (128, 16, 64, None)),
        ("7,auto,128", (7, None, 128, None)),
        ("auto,16,128,32", (None, 16, 128, 32)),
        ((2, 32, 256, 64), (2, 32, 256, 64)),
        ("1,auto,64", (1, None, 64, None)),
        ((8, 16, 128, None), (8, 16, 128, None)),
        ([None, "auto", 256], (None, None, 256, None)),
    ],
)
def test_parse_config(value, expected):
    assert impl._parse_config(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1,16",
        "1,16,64,128",
        "1,16,64,0",
        "1,16,64,32,16",
        (1, 16, 64, True),
        "0,16,64",
        "129,16,64",
        "-1,16,64",
        "1,8,64",
        "1,64,128",
        "1,16,32",
        "1,16,512",
        "1.0,16,64",
        "AUTO,16,64",
        "auto,,64",
        (True, 16, 64),
        (1.0, 16, 64),
        (1, 16, False),
        ("1", 16, 64),
        4,
        {},
    ],
)
def test_invalid_config(value):
    with pytest.raises(ValueError, match="config"):
        impl._parse_config(value)


def _fresh_module():
    spec = importlib.util.spec_from_file_location("config_probe", impl.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config_probe"] = module  # dataclasses resolve their module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules["config_probe"]
    return module


def test_environment_read_once(monkeypatch):
    monkeypatch.setenv("CTI_LORA_SEG_CONFIG", "16,auto,256")
    module = _fresh_module()
    assert module._ENV_CONFIG == (16, None, 256, None)
    monkeypatch.setenv("CTI_LORA_SEG_CONFIG", "1,16,64")
    assert module._ENV_CONFIG == (16, None, 256, None)
    monkeypatch.setenv("CTI_LORA_SEG_CONFIG", "0,16,64")
    with pytest.raises(ValueError, match="splits"):
        _fresh_module()
    monkeypatch.delenv("CTI_LORA_SEG_CONFIG")
    assert _fresh_module()._ENV_CONFIG == (None, None, None, None)


@pytest.mark.parametrize("api", ["lora_segmented_delta_", "lora_delta_factors_"])
def test_public_api_validates_config_before_device_work(api):
    kwargs = (
        {
            "lora_a": None,
            "lora_b": None,
            "seg_indptr": None,
            "weight_indices": None,
            "ranks": None,
            "scalings": None,
        }
        if api == "lora_segmented_delta_"
        else {"factors": (), "scaling": 1}
    )
    with pytest.raises(ValueError, match="splits"):
        getattr(impl, api)(None, None, **kwargs, output_sizes=(16,), config="129,16,64")
    assert last_config() is None
