# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Stand-ins for the provider of ctikernels.gdn on the CPU: fla-core's modules (in sys.modules,
or as an installed distribution on disk for a fresh interpreter) and the package's Triton
modules."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from ctikernels.gdn import _fla

# fla-core's modules that read a configuration variable when imported.
DISPATCH = "fla.ops.backends"
FAST_OPS = "fla.ops.utils.op"
# Their variables, and their stand-in sources: the dispatch module's flag, and the op module
# reduced to exp, which triton.jit keeps as .fn and the variable binds to libdevice's fast_expf.
CONFIGURATION = {
    DISPATCH: (
        _fla.DISPATCH_VARIABLE,
        f"""import os

_DISPATCH_DISABLED = os.environ.get({_fla.DISPATCH_VARIABLE!r}) == "1"
""",
    ),
    FAST_OPS: (
        _fla.FAST_OPS_VARIABLE,
        f"""import os


class JITFunction:
    def __init__(self, fn):
        self.fn = fn


if os.environ.get({_fla.FAST_OPS_VARIABLE!r}, "0") == "1":
    def exp(x):
        return tldevice.fast_expf(x)
else:
    def exp(x):
        return tl.exp(x)
exp = JITFunction(exp)
""",
    ),
}

# Every fla-core module the provider imports, with the entry points it binds from it.
ENTRIES = {
    "fla.modules.conv.triton.ops": ("causal_conv1d_update",),
    "fla.modules.fused_norm_gate": ("layer_norm_gated_fwd", "rms_norm_gated"),
    "fla.ops.common.chunk_delta_h": (
        "chunk_gated_delta_rule_bwd_dhu",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    ),
    "fla.ops.common.chunk_o": ("chunk_bwd_dqkwg", "chunk_bwd_dv_local", "chunk_fwd_o"),
    "fla.ops.common.gate": ("fused_beta_sigmoid_bwd", "fused_beta_sigmoid_fwd"),
    "fla.ops.gated_delta_rule": ("fused_recurrent_gated_delta_rule",),
    "fla.ops.gated_delta_rule.chunk_fwd": ("chunk_gated_delta_rule_fwd_intra",),
    "fla.ops.gated_delta_rule.gate": ("gdn_gate_chunk_cumsum",),
    "fla.ops.gated_delta_rule.wy_fast": ("prepare_wy_repr_bwd", "recompute_w_u_fwd"),
    "fla.ops.utils.cumsum": ("chunk_local_cumsum",),
}

# fla-core modules the tests' FLA references import besides the provider's.
REFERENCE_ENTRIES = {"fla.ops.gated_delta_rule.chunk": ("chunk_gated_delta_rule_fwd",)}

# The FlaCore attribute each entry point is bound to.
BINDINGS = {
    "causal_conv1d_update": "_conv_update",
    "fused_recurrent_gated_delta_rule": "_recurrent",
    "gdn_gate_chunk_cumsum": "_gate_cumsum",
    "chunk_gated_delta_rule_fwd_intra": "_intra",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "_state_kernel",
    "chunk_fwd_o": "_output",
    "fused_beta_sigmoid_fwd": "_beta_sigmoid",
    "rms_norm_gated": "_gated_norm",
    "layer_norm_gated_fwd": "_gated_norm_fwd",
    "recompute_w_u_fwd": "_recompute_w_u",
    "chunk_gated_delta_rule_fwd_h": "_chunk_h",
    "chunk_bwd_dv_local": "_dv_local",
    "chunk_gated_delta_rule_bwd_dhu": "_dhu",
    "chunk_bwd_dqkwg": "_dqkwg",
    "prepare_wy_repr_bwd": "_wy_bwd",
    "fused_beta_sigmoid_bwd": "_beta_sigmoid_bwd",
    "chunk_local_cumsum": "_local_cumsum",
}


def version(installed: str | None):
    """A stand-in for importlib.metadata.version that knows fla-core only."""

    def lookup(name: str) -> str:
        if name != _fla.DISTRIBUTION:
            raise AssertionError(name)
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    return lookup


class NoFla(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla" or fullname.startswith("fla."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def configuration_module(name: str, value: str) -> ModuleType:
    """The stand-in of fla-core's module ``name`` as imported with its variable at ``value``."""
    variable, source = CONFIGURATION[name]
    module = ModuleType(name)
    with mock.patch.dict(os.environ, {variable: value}):
        exec(source, module.__dict__)
    return module


class ConfigurationModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stand-ins of fla-core's configuration modules ``names`` that, like fla-core's, read
    their variable when imported."""

    def __init__(self, names) -> None:
        self.names = frozenset(names)

    def find_spec(self, fullname, path=None, target=None):
        return importlib.util.spec_from_loader(fullname, self) if fullname in self.names else None

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        exec(CONFIGURATION[module.__name__][1], module.__dict__)


def fake_fla(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dispatch_disabled: bool | None,
    fast_ops: bool | None = False,
) -> dict[str, object]:
    """Stand-in fla-core modules in sys.modules; returns the entry points by name. A None
    setting leaves its module unloaded, to read its variable when imported."""
    entries: dict[str, dict[str, object]] = {
        module: {name: object() for name in names}
        for module, names in {**ENTRIES, **REFERENCE_ENTRIES}.items()
    }
    settings = {DISPATCH: dispatch_disabled, FAST_OPS: fast_ops}
    absent = [name for name, setting in settings.items() if setting is None]
    for name in absent:
        # Absent now; the entry its import adds goes (or the real one returns) afterwards.
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    if absent:
        monkeypatch.setattr(sys, "meta_path", [ConfigurationModules(absent), *sys.meta_path])
    modules: dict[str, ModuleType] = {}
    for dotted in [*entries, *settings]:
        parts = dotted.split(".")
        for depth in range(1, len(parts) + 1):
            name = ".".join(parts[:depth])
            if name in modules:
                continue
            if name in settings:
                if settings[name] is None:
                    continue  # imported later, through ConfigurationModules
                modules[name] = configuration_module(name, "1" if settings[name] else "0")
            else:
                modules[name] = ModuleType(name)
                modules[name].__path__ = []
            if depth > 1:
                setattr(modules[".".join(parts[: depth - 1])], parts[depth - 1], modules[name])
    for dotted, attributes in entries.items():
        for attribute, value in attributes.items():
            setattr(modules[dotted], attribute, value)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(importlib.metadata, "version", version(_fla.VERSION))
    return {name: entries[module][name] for module, names in ENTRIES.items() for name in names}


def fake_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-ins for the package's Triton modules, which the provider loads."""
    stand_ins = {
        "_rows": {"gather_rows": "gather", "write_rows": "write"},
        "_fla_glue": {"prepare": "prepare", "commit": "commit"},
        "triton._features": {"features": "features"},
        "triton._launch": {"Launcher": object},
        "triton._intra": {"intra": "intra"},
        "triton._state": {"state": "state"},
        "triton._output": {"output": "output"},
        "triton._norm": {"gated_norm": "prefill norm"},
        "_gated_norm": {"gated_rms_norm_backward": "gated norm backward"},
        "_fla_training": {
            "sequence_index": "sequence index",
            "conv_forward": "conv forward",
            "conv_backward": "conv backward",
            "gate_gradient": "gate gradient",
        },
    }
    for leaf, attributes in stand_ins.items():
        module = ModuleType(f"ctikernels.gdn.{leaf}")
        for name, value in attributes.items():
            setattr(module, name, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        package, _, name = module.__name__.rpartition(".")
        monkeypatch.setattr(importlib.import_module(package), name, module, raising=False)


def write_distribution(root: Path) -> None:
    """fla-core 0.5.2 installed under ``root`` for a fresh interpreter: entry points that fail
    when called, and the two configuration modules reading their variables at import."""
    info = root / f"fla_core-{_fla.VERSION}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {_fla.DISTRIBUTION}\nVersion: {_fla.VERSION}\n"
    )
    sources: dict[str, list[str]] = {}
    for dotted, names in ENTRIES.items():
        parts = dotted.split(".")
        for depth in range(1, len(parts) + 1):
            sources.setdefault(".".join(parts[:depth]), [])
        sources[dotted].extend(
            f"def {name}(*args, **kwargs):\n    raise AssertionError({name!r})\n" for name in names
        )
    for dotted, (_, source) in CONFIGURATION.items():
        parts = dotted.split(".")
        for depth in range(1, len(parts)):
            sources.setdefault(".".join(parts[:depth]), [])
        sources[dotted] = [source]
    for dotted, lines in sources.items():
        package = root.joinpath(*dotted.split("."))
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("".join(lines))


__all__ = [
    "BINDINGS",
    "CONFIGURATION",
    "DISPATCH",
    "ENTRIES",
    "FAST_OPS",
    "ConfigurationModules",
    "NoFla",
    "configuration_module",
    "fake_fla",
    "fake_kernels",
    "version",
    "write_distribution",
]
