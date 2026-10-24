# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""ctikernels.gdn on the CPU: geometry, provider binding and its fla-core configuration, the
frozen-weight rule, inference contract checks and the device-built varlen metadata.

Decode and prefill bind fla-core only off the fused Triton kernels' device types and
geometries, so the provider tests stand them on a device type without those; the dispatch tests
at the end stand them on each accepted device type."""

from __future__ import annotations

import importlib.metadata
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from ctikernels.gdn import (
    GatedDeltaNetDecode,
    GatedDeltaNetPrefill,
    GatedDeltaNetTraining,
    _contract,
    _cute,
    _fla,
    _triton_decode,
    _triton_prefill,
    inference,
)

from ..support.child import run_child
from ._gdn_provider import (
    BINDINGS,
    DISPATCH,
    FAST_OPS,
    NoFla,
    fake_fla,
    fake_kernels,
    version,
    write_distribution,
)

K = V = 128
WEIGHTS = ("conv_weight", "A_log", "dt_bias", "norm_weight")
# Operations that bind fla-core; decode and prefill do so here, on a device type without the
# fused kernels below compute capability 8.9.
CONSTRUCTED = (GatedDeltaNetDecode, GatedDeltaNetPrefill, GatedDeltaNetTraining)
_DECODE_LOAD = _triton_decode.load
_PREFILL_DEVICE_TYPE = _triton_prefill.device_type
# _fla.describe adds the product name and compute capability when CUDA is available.
_CUDA0 = r"cuda:0(?: \([^)]*\))?"


@pytest.fixture(autouse=True)
def _unbound_provider(monkeypatch):
    """Every test starts without a bound provider and restores the process's afterwards, with
    cuda:0 of a device type the fused kernels do not serve (whatever GPU the host has); these
    tests hold the fla-core training provider (test_gdn_cute.py holds the CuTe one)."""
    # These provider stand-ins model Triton/FLA; CuTe policy has its own CPU suite.
    monkeypatch.setenv("CTI_GDN_PREFILL_CUTE", "0")
    monkeypatch.setattr(_fla, "_provider", None)
    monkeypatch.setattr(_triton_decode, "load", lambda *arguments, **options: None)
    monkeypatch.setattr(_triton_prefill, "device_type", lambda device: None)
    monkeypatch.setattr(_cute, "training", lambda device: None)
    # Below compute capability 8.9, prefill and training bind fla-core too.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA Test GPU")


@pytest.mark.parametrize(
    ("heads", "conv_dim", "proj_width"),
    [
        ((16, 32), 8192, 12352),
        ((16, 48), 10240, 16480),
        ((8, 16), 4096, 6176),
        ((2, 6), 1280, 2060),
    ],
)
def test_geometry_widths_follow_the_merged_projection(heads, conv_dim, proj_width) -> None:
    geometry = _contract.Geometry(*heads)
    assert (geometry.conv_dim, geometry.proj_width) == (conv_dim, proj_width)


@pytest.mark.parametrize("heads", [(0, 32), (16, 0), (16, 24), (-1, 2), (True, 2), (2.0, 4)])
def test_operations_reject_impossible_head_counts_at_construction(heads, monkeypatch) -> None:
    loads = []
    monkeypatch.setattr(_fla, "load", lambda *arguments: loads.append(arguments))
    for operation in CONSTRUCTED:
        with pytest.raises(ValueError):
            operation(num_k_heads=heads[0], num_v_heads=heads[1], device="cuda:0")
    assert loads == []


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_operations_run_on_cuda_devices_only(operation, monkeypatch) -> None:
    monkeypatch.setattr(_fla, "load", lambda *arguments: pytest.fail("provider bound"))
    with pytest.raises(ValueError, match=f"{operation.__name__} runs on a CUDA device, not cpu"):
        operation(num_k_heads=16, num_v_heads=32, device="cpu")


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_a_missing_provider_fails_construction_naming_the_operation_and_device(
    operation, monkeypatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", version(None))
    with pytest.raises(RuntimeError) as raised:
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0")
    message = str(raised.value)
    assert message.startswith(f"{operation.__name__} on cuda:0")
    assert "fla-core==0.5.2" in message and "cti-kernels[gdn]" in message
    assert "not installed" in message


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_another_provider_version_fails_construction(operation, monkeypatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", version("0.4.2"))
    with pytest.raises(RuntimeError, match=r"needs fla-core==0\.5\.2.*found fla-core 0\.4\.2"):
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0")


def _fast_exp(module) -> bool:
    """Whether fla-core's op module (or its stand-in) was imported with fast exp and log."""
    return "fast_expf" in module.exp.fn.__code__.co_names


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_the_provider_is_imported_with_dispatch_and_fast_ops_pinned_off(
    operation, monkeypatch
) -> None:
    """fla-core's backend dispatch and fast exp and log are turned off before its first import
    (inherited values are overridden); a failing import still fails construction."""
    monkeypatch.setattr(importlib.metadata, "version", version(_fla.VERSION))
    monkeypatch.setenv(_fla.DISPATCH_VARIABLE, "0")
    monkeypatch.setenv(_fla.FAST_OPS_VARIABLE, "1")
    for name in [name for name in sys.modules if name == "fla" or name.startswith("fla.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [NoFla(), *sys.meta_path])
    with pytest.raises(RuntimeError, match="needs fla-core==0.5.2.*importing it failed"):
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0")
    assert os.environ[_fla.DISPATCH_VARIABLE] == "1"
    assert os.environ[_fla.FAST_OPS_VARIABLE] == "0"


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_a_provider_imported_with_dispatch_on_fails_construction(operation, monkeypatch) -> None:
    fake_fla(monkeypatch, dispatch_disabled=False)
    with pytest.raises(
        RuntimeError,
        match=rf"{operation.__name__} on {_CUDA0}: fla-core's {re.escape(DISPATCH)} was imported .* "
        rf"dispatch on; export {_fla.DISPATCH_VARIABLE}=1",
    ):
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0")


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_a_provider_imported_with_fast_ops_on_fails_construction(operation, monkeypatch) -> None:
    fake_fla(monkeypatch, dispatch_disabled=True, fast_ops=True)
    assert _fast_exp(sys.modules[FAST_OPS])
    with pytest.raises(
        RuntimeError,
        match=rf"{operation.__name__} on {_CUDA0}: fla-core's {re.escape(FAST_OPS)} was imported .* "
        rf"fast exp and log on; export {_fla.FAST_OPS_VARIABLE}=0",
    ):
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0")


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_both_variables_are_pinned_when_only_the_top_level_package_was_imported(
    operation, monkeypatch
) -> None:
    """With only the top-level fla package imported (transformers' kernel lookup), construction
    still pins dispatch and fast ops off before their modules read the variables."""
    monkeypatch.setenv(_fla.DISPATCH_VARIABLE, "0")
    monkeypatch.setenv(_fla.FAST_OPS_VARIABLE, "1")
    fake_fla(monkeypatch, dispatch_disabled=None, fast_ops=None)
    fake_kernels(monkeypatch)
    assert "fla" in sys.modules and DISPATCH not in sys.modules and FAST_OPS not in sys.modules
    operation(num_k_heads=16, num_v_heads=32, device="cuda:0")
    assert os.environ[_fla.DISPATCH_VARIABLE] == "1"
    assert os.environ[_fla.FAST_OPS_VARIABLE] == "0"
    assert sys.modules[DISPATCH]._DISPATCH_DISABLED is True
    assert not _fast_exp(sys.modules[FAST_OPS])


def test_the_provider_binds_once_per_process_to_fla_cores_entry_points(monkeypatch) -> None:
    entries = fake_fla(monkeypatch, dispatch_disabled=True)
    fake_kernels(monkeypatch)
    decode = GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:0")
    prefill = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=48, device=0)
    training = GatedDeltaNetTraining(num_k_heads=8, num_v_heads=24, device="cuda:0")
    provider = _fla.load("unused", torch.device("cuda:0"))
    assert decode._kernel._kernel == provider.decode and prefill._kernel._kernel == provider.prefill
    assert training._kernel is provider
    from ctikernels.gdn import _fla_fixed, _l2norm

    assert provider._l2norm is _l2norm.l2norm_forward
    assert provider._l2norm_bwd is _l2norm.l2norm_backward
    assert set(BINDINGS) == set(entries)
    assert provider._training_chunk is _fla_fixed.chunk_gated_delta_rule_fwd
    fixed = {
        "layer_norm_gated_fwd",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_bwd_dv_local",
        "chunk_gated_delta_rule_bwd_dhu",
        "chunk_bwd_dqkwg",
        "prepare_wy_repr_bwd",
        "chunk_local_cumsum",
    }
    for entry, attribute in BINDINGS.items():
        expected = getattr(_fla_fixed, entry) if entry in fixed else entries[entry]
        assert getattr(provider, attribute) is expected, entry
    assert (provider._gather, provider._write) == ("gather", "write")
    assert provider._gated_norm_bwd == "gated norm backward"
    training_parts = (
        provider._sequence_index,
        provider._conv_fwd,
        provider._conv_bwd,
        provider._gate_gradient,
    )
    assert training_parts == ("sequence index", "conv forward", "conv backward", "gate gradient")
    assert prefill.device == torch.device("cuda", 0)


ORDERS = {
    "inference first": (GatedDeltaNetDecode, GatedDeltaNetPrefill, GatedDeltaNetTraining),
    "training first": (GatedDeltaNetTraining, GatedDeltaNetPrefill, GatedDeltaNetDecode),
}


@pytest.mark.parametrize("order", list(ORDERS))
def test_operations_constructed_in_either_order_share_one_configured_provider(
    order, monkeypatch
) -> None:
    """Whichever operation is constructed first imports fla-core under the one configuration;
    the others bind the same provider and never import or configure again."""
    monkeypatch.setenv(_fla.DISPATCH_VARIABLE, "0")
    monkeypatch.setenv(_fla.FAST_OPS_VARIABLE, "1")
    fake_fla(monkeypatch, dispatch_disabled=None, fast_ops=None)
    fake_kernels(monkeypatch)
    imports = []
    load = _fla._import
    monkeypatch.setattr(
        _fla, "_import", lambda *arguments: imports.append(arguments) or load(*arguments)
    )
    built = [
        operation(num_k_heads=16, num_v_heads=32, device="cuda:0") for operation in ORDERS[order]
    ]
    assert imports == [(ORDERS[order][0].__name__, torch.device("cuda", 0))]
    assert sys.modules[DISPATCH]._DISPATCH_DISABLED is True
    assert not _fast_exp(sys.modules[FAST_OPS])
    provider = _fla._provider
    kernels = {type(op): op._kernel for op in built}
    assert kernels[GatedDeltaNetTraining] is provider
    assert kernels[GatedDeltaNetDecode]._kernel == provider.decode
    assert kernels[GatedDeltaNetPrefill]._kernel == provider.prefill


# fla-core variables the one configuration leaves as the process set them (values that differ
# from fla-core's defaults; FLA_INTRACARD_CP unset).
LEFT_ALONE = {
    "FLA_DISABLE_TENSOR_CACHE": "1",
    "FLA_TENSOR_CACHE_SIZE": "1",
    "FLA_CACHE_RESULTS": "0",
    "FLA_CACHE_MODE": "strict",
}


@pytest.mark.parametrize("operation", CONSTRUCTED)
def test_fla_cores_other_variables_are_left_as_the_process_set_them(operation, monkeypatch) -> None:
    """Only dispatch and fast ops are pinned: a process that set fla-core's cache variables
    binds, and its settings stay."""
    for name, value in LEFT_ALONE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("FLA_INTRACARD_CP", raising=False)
    fake_fla(monkeypatch, dispatch_disabled=None, fast_ops=None)
    fake_kernels(monkeypatch)
    operation(num_k_heads=16, num_v_heads=32, device="cuda:0")
    assert {name: os.environ[name] for name in LEFT_ALONE} == LEFT_ALONE
    assert "FLA_INTRACARD_CP" not in os.environ


CONSTRUCTION = """
import importlib, os, sys
from types import ModuleType

import torch

import ctikernels.gdn
from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetPrefill, GatedDeltaNetTraining, _fla

# A device below compute capability 8.9: prefill and training bind fla-core there.
torch.cuda.get_device_capability = lambda device=None: (8, 6)
torch.cuda.get_device_name = lambda device=None: "NVIDIA Test GPU"

# The package's Triton modules: not loaded on the CPU.
for leaf, names in {{"_rows": ("gather_rows", "write_rows"),
                     "_fla_glue": ("prepare", "commit"),
                     "triton._features": ("features",),
                     "triton._launch": ("Launcher",),
                     "triton._norm": ("gated_norm",),
                     "_gated_norm": ("gated_rms_norm_backward",),
                     "_fla_training": ("sequence_index", "conv_forward", "conv_backward",
                                       "gate_gradient")}}.items():
    module = ModuleType("ctikernels.gdn." + leaf)
    for name in names:
        setattr(module, name, name)
    sys.modules[module.__name__] = module
    package, _, name = module.__name__.rpartition(".")
    setattr(importlib.import_module(package), name, module)
sys.modules["ctikernels.gdn.triton._launch"].Launcher = object
assert not any(name == "fla" or name.startswith("fla.") for name in sys.modules)
operations = {{op.__name__: op for op in
              (GatedDeltaNetDecode, GatedDeltaNetPrefill, GatedDeltaNetTraining)}}
built = [operations[name](num_k_heads=16, num_v_heads=32, device="cuda:0")
         for name in {order!r}]
import fla.ops.backends
import fla.ops.utils.op
assert fla.ops.backends._DISPATCH_DISABLED is True
assert "fast_expf" not in fla.ops.utils.op.exp.fn.__code__.co_names
assert os.environ["FLA_DISABLE_BACKEND_DISPATCH"] == "1"
assert os.environ["FLA_USE_FAST_OPS"] == "0"
provider = _fla._provider
for op in built:
    kernel = op._kernel
    assert kernel is provider or kernel.__self__ is provider, op
print("bound", *(type(op).__name__ for op in built))
"""


@pytest.mark.parametrize("order", list(ORDERS))
def test_a_fresh_interpreter_binds_every_operation_in_either_order(order, tmp_path) -> None:
    """In a new process whose stand-in fla-core inherits wrong variables, the first construction
    configures it and every operation binds the same provider, in either order."""
    write_distribution(tmp_path)
    names = [operation.__name__ for operation in ORDERS[order]]
    package = Path(_fla.__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(package))),
        _fla.DISPATCH_VARIABLE: "0",
        _fla.FAST_OPS_VARIABLE: "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = run_child(
        [sys.executable, "-c", CONSTRUCTION.format(order=names)],
        env=environment,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["bound", *names]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *arguments):
        self.calls.append(arguments)
        geometry, projected = arguments[:2]
        return torch.zeros(projected.shape[0], geometry.value_width, device=projected.device)


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    """Operations bound on the meta device to a recording kernel, resolved once each."""
    kernel, resolutions = _Recorder(), []

    def resolve(geometry, device, **options):
        resolutions.append((geometry, device))
        return kernel

    monkeypatch.setattr(inference, "cuda_device", lambda operation, device: torch.device("meta"))
    monkeypatch.setattr(inference, "_decode_kernel", resolve)
    monkeypatch.setattr(inference, "_prefill_kernel", resolve)
    kernel.resolutions = resolutions
    return kernel


def _meta(shape, dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


def _decode_arguments(k_heads=16, v_heads=32, batch=4, slots=10) -> dict[str, torch.Tensor]:
    geometry = _contract.Geometry(k_heads, v_heads)
    return {
        "projected": _meta((batch, geometry.proj_width), torch.bfloat16),
        "conv_weight": _meta((geometry.conv_dim, 4), torch.bfloat16),
        "A_log": _meta((v_heads,), torch.float32),
        "dt_bias": _meta((v_heads,), torch.float32),
        "norm_weight": _meta((V,), torch.float32),
        "slot_indices": _meta((batch,), torch.int32),
        "state": _meta((slots, v_heads, V, K), torch.float32),
        "conv_state": _meta((slots, 3, geometry.conv_dim), torch.bfloat16),
    }


def _prefill_arguments(k_heads=16, v_heads=32, tokens=40, sequences=3, slots=10):
    arguments = _decode_arguments(k_heads, v_heads, tokens, slots)
    del arguments["slot_indices"]
    state, conv_state = arguments.pop("state"), arguments.pop("conv_state")
    return {
        **arguments,
        "cu_seqlens": _meta((sequences + 1,), torch.int32),
        "slot_indices": _meta((sequences,), torch.int32),
        "has_initial_state": _meta((sequences,), torch.bool),
        "state": state,
        "conv_state": conv_state,
    }


OPERATIONS = {
    "decode": (GatedDeltaNetDecode, _decode_arguments),
    "prefill": (GatedDeltaNetPrefill, _prefill_arguments),
}


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_calls_reach_the_kernel_resolved_at_construction_with_checked_inputs(
    operation, recorder
) -> None:
    cls, make = OPERATIONS[operation]
    op = cls(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = make()
    for _ in range(3):
        y = op(*arguments.values())
        assert y.shape == (arguments["projected"].shape[0], 32 * V)
    assert recorder.resolutions == [(_contract.Geometry(16, 32), torch.device("meta"))]
    assert len(recorder.calls) == 3
    for call in recorder.calls:
        assert call[0] == _contract.Geometry(16, 32)
        assert all(sent is given for sent, given in zip(call[1:], arguments.values(), strict=True))


def test_an_empty_decode_batch_launches_nothing(recorder) -> None:
    op = GatedDeltaNetDecode(num_k_heads=16, num_v_heads=48, device="cuda:0")
    y = op(*_decode_arguments(16, 48, batch=0).values())
    assert y.shape == (0, 48 * V) and y.dtype == torch.bfloat16
    assert recorder.calls == []


@pytest.mark.parametrize("operation", list(OPERATIONS))
@pytest.mark.parametrize("weight", WEIGHTS)
@pytest.mark.parametrize("grad_mode", [True, False])
def test_a_weight_that_requires_grad_is_rejected_before_any_other_check(
    operation, weight, grad_mode, recorder
) -> None:
    cls, make = OPERATIONS[operation]
    op = cls(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = make()
    # Every other argument is wrong too: the frozen rule must be the first check.
    broken = dict.fromkeys(arguments)
    broken[weight] = torch.zeros(1, requires_grad=True)
    with torch.set_grad_enabled(grad_mode):
        with pytest.raises(ValueError, match=f"{cls.__name__}: {weight} is a frozen base weight"):
            op(*broken.values())
        arguments[weight] = arguments[weight].clone().requires_grad_(True)
        with pytest.raises(ValueError, match=f"{weight} is a frozen base weight"):
            op(*arguments.values())
    assert recorder.calls == []


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_inference_operations_refuse_autograd(operation, recorder) -> None:
    cls, make = OPERATIONS[operation]
    op = cls(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = make()
    arguments["projected"] = arguments["projected"].clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="does not support autograd"):
        op(*arguments.values())
    with torch.no_grad():
        op(*arguments.values())
    for pool in ("state", "conv_state"):
        broken = {**make(), pool: make()[pool].clone().requires_grad_(True)}
        with torch.no_grad(), pytest.raises(RuntimeError, match="cannot update a pool"):
            op(*broken.values())
    assert len(recorder.calls) == 1


def _strided(shape, dtype) -> torch.Tensor:
    return _meta((shape[1], shape[0]), dtype).t()


DECODE_VIOLATIONS = {
    "projected dtype": ("projected", lambda: _meta((4, 12352), torch.float16), TypeError),
    "projected width": ("projected", lambda: _meta((4, 12351), torch.bfloat16), ValueError),
    "projected rank": ("projected", lambda: _meta((4, 1, 12352), torch.bfloat16), ValueError),
    "projected column stride": ("projected", lambda: _strided((4, 12352), torch.bfloat16), ValueError),
    "projected not a tensor": ("projected", lambda: [0.0], TypeError),
    "conv_weight dtype": ("conv_weight", lambda: _meta((8192, 4), torch.float32), TypeError),
    "conv_weight checkpoint shape": ("conv_weight", lambda: _meta((8192, 1, 4), torch.bfloat16), ValueError),
    "conv_weight strided": ("conv_weight", lambda: _strided((8192, 4), torch.bfloat16), ValueError),
    "A_log BF16": ("A_log", lambda: _meta((32,), torch.bfloat16), TypeError),
    "A_log heads": ("A_log", lambda: _meta((16,), torch.float32), ValueError),
    "dt_bias BF16": ("dt_bias", lambda: _meta((32,), torch.bfloat16), TypeError),
    "norm_weight width": ("norm_weight", lambda: _meta((256,), torch.float32), ValueError),
    "slot_indices dtype": ("slot_indices", lambda: _meta((4,), torch.int64), TypeError),
    "slot_indices count": ("slot_indices", lambda: _meta((5,), torch.int32), ValueError),
    "state dtype": ("state", lambda: _meta((10, 32, V, K), torch.bfloat16), TypeError),
    "state key-first layout": ("state", lambda: _meta((10, 32, V, K), torch.float32).transpose(2, 3), ValueError),
    "state heads": ("state", lambda: _meta((10, 48, V, K), torch.float32), ValueError),
    "conv_state slots": ("conv_state", lambda: _meta((9, 3, 8192), torch.bfloat16), ValueError),
    "conv_state channel-major": ("conv_state", lambda: _meta((10, 8192, 3), torch.bfloat16), ValueError),
    "conv_state dtype": ("conv_state", lambda: _meta((10, 3, 8192), torch.float32), TypeError),
    "device": ("slot_indices", lambda: torch.zeros(4, dtype=torch.int32), ValueError),
}  # fmt: skip


@pytest.mark.parametrize("case", list(DECODE_VIOLATIONS))
def test_decode_checks_its_contract_before_any_kernel(case, recorder) -> None:
    name, value, error = DECODE_VIOLATIONS[case]
    op = GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = {**_decode_arguments(), name: value()}
    with pytest.raises(error):
        op(*arguments.values())
    assert recorder.calls == []


PREFILL_VIOLATIONS = {
    "cu_seqlens dtype": ("cu_seqlens", lambda: _meta((4,), torch.int64), TypeError),
    "cu_seqlens rank": ("cu_seqlens", lambda: _meta((4, 1), torch.int32), ValueError),
    "cu_seqlens without a sequence": ("cu_seqlens", lambda: _meta((1,), torch.int32), ValueError),
    "slot_indices count": ("slot_indices", lambda: _meta((4,), torch.int32), ValueError),
    "has_initial_state dtype": ("has_initial_state", lambda: _meta((3,), torch.int32), TypeError),
    "has_initial_state count": ("has_initial_state", lambda: _meta((2,), torch.bool), ValueError),
    "projected width": ("projected", lambda: _meta((40, 12351), torch.bfloat16), ValueError),
    "state heads": ("state", lambda: _meta((10, 16, V, K), torch.float32), ValueError),
}  # fmt: skip


@pytest.mark.parametrize("case", list(PREFILL_VIOLATIONS))
def test_prefill_checks_its_contract_before_any_kernel(case, recorder) -> None:
    name, value, error = PREFILL_VIOLATIONS[case]
    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = {**_prefill_arguments(), name: value()}
    with pytest.raises(error):
        op(*arguments.values())
    assert recorder.calls == []


@pytest.mark.parametrize(
    ("make_out", "error", "message"),
    [
        (lambda: [], TypeError, "out must be a tensor"),
        (lambda: _meta((40, 4096), torch.float32), TypeError, "out must be torch.bfloat16"),
        (lambda: _meta((39, 4096), torch.bfloat16), ValueError, "out must be"),
        (lambda: _meta((40, 4095), torch.bfloat16), ValueError, "out must be"),
        (lambda: _meta((40, 32, 128), torch.bfloat16), ValueError, "out must be"),
        (lambda: _strided((40, 4096), torch.bfloat16), ValueError, "out must be contiguous"),
        (
            lambda: _meta((40, 4097), torch.bfloat16)[:, :4096],
            ValueError,
            "out must be contiguous",
        ),
        (
            lambda: _meta((1, 4096), torch.bfloat16).expand(40, -1),
            ValueError,
            "out must be contiguous",
        ),
        (lambda: torch.empty(40, 4096, dtype=torch.bfloat16), ValueError, "out must be on"),
        (
            lambda: _meta((40, 4096), torch.bfloat16).requires_grad_(),
            ValueError,
            "out must not require gradients",
        ),
    ],
)
def test_prefill_rejects_invalid_out_before_dispatch(make_out, error, message, recorder):
    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
    with pytest.raises(error, match=message):
        op(*_prefill_arguments().values(), out=make_out())
    assert recorder.calls == []
    assert not op._guard._eager


@pytest.mark.parametrize("provider", ["cute", "triton", "fla-core"])
def test_prefill_returns_supplied_out_and_only_fallbacks_copy(provider, monkeypatch):
    monkeypatch.setattr(inference, "cuda_device", lambda *args: torch.device("cpu"))
    calls = []

    def kernel(geometry, projected, *args, out=None, capture_rows=None):
        assert capture_rows is None
        calls.append(out)
        if provider == "cute":
            assert out is output
            return out.fill_(3)
        assert out is None
        return torch.full((projected.shape[0], geometry.value_width), 3, dtype=torch.bfloat16)

    kernel.provider = provider
    dispatch = kernel
    if provider == "cute":
        # Exercise the real Python dispatch wrapper's out= pass-through too.
        dispatch = object.__new__(_triton_prefill.TritonPrefill)
        dispatch._cute = kernel
        dispatch.selected_variant = lambda *args, **kwargs: "cute"

        def no_copy(*args, **kwargs):
            pytest.fail("CuTe out= must not copy the output")

        monkeypatch.setattr(torch.Tensor, "copy_", no_copy)
    monkeypatch.setattr(inference, "_prefill_kernel", lambda *args, **options: dispatch)
    op = GatedDeltaNetPrefill(num_k_heads=1, num_v_heads=2, device="cuda:0")
    arguments = {
        name: torch.empty(t.shape, dtype=t.dtype)
        for name, t in _prefill_arguments(1, 2, tokens=4, sequences=1, slots=1).items()
    }
    # A contiguous slice with an offset is legal; adjacent caller storage is untouched.
    storage = torch.full((4 * 256 + 2,), -1, dtype=torch.bfloat16)
    output = storage[1:-1].view(4, 256)
    assert op(*arguments.values(), out=output) is output
    assert len(calls) == 1
    assert torch.all(output == 3)
    assert storage[0] == storage[-1] == -1


def test_projected_may_be_a_row_strided_view(recorder) -> None:
    op = GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:0")
    arguments = _decode_arguments()
    arguments["projected"] = _meta((4, 12352 + 64), torch.bfloat16)[:, :12352]
    op(*arguments.values())
    overlapping = torch.empty(4, 12352, dtype=torch.bfloat16, device="meta").as_strided(
        (4, 12352), (100, 1)
    )
    with pytest.raises(ValueError, match="non-overlapping rows"):
        op(*{**arguments, "projected": overlapping}.values())
    assert len(recorder.calls) == 1


def _host_chunk_table(lengths: list[int]) -> list[tuple[int, int]]:
    return [(n, c) for n, length in enumerate(lengths) for c in range(-(-length // 64))]


@pytest.mark.parametrize(
    "lengths",
    [[1], [64], [65], [1, 63, 64, 65, 127, 128, 129], [1] * 9, [300, 1, 1], [5, 200], [64] * 4],
)
def test_the_device_chunk_table_lists_every_chunk_then_only_masked_entries(lengths) -> None:
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    tokens = sum(lengths)
    table = _fla._chunk_table(cu, tokens)
    capacity = (tokens + 63 * len(lengths)) // 64
    assert table.dtype == torch.int32 and table.shape == (capacity, 2)
    chunks = _host_chunk_table(lengths)
    assert [tuple(row) for row in table[: len(chunks)].tolist()] == chunks
    # The rest name the last sequence at a chunk past its end: every row of it is masked.
    for sequence, chunk in table[len(chunks) :].tolist():
        assert sequence == len(lengths) - 1 and chunk * 64 >= lengths[-1]


@pytest.mark.parametrize(
    ("capability", "refused"),
    [
        ((10, 3), set()),  # prefill and training both bind our Triton providers there
        ((12, 0), set()),
        ((9, 0), set()),
        ((10, 0), set()),
    ],
)
def test_operations_the_provider_computes_wrongly_on_a_device_fail_construction_there(
    monkeypatch, capability, refused
) -> None:
    fake_fla(monkeypatch, dispatch_disabled=True)
    fake_kernels(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA Test GPU")
    # Our Triton training provider (cc 10.0-12.1) stands in: Triton is absent on the CPU.
    triton_training = ModuleType("ctikernels.gdn.triton.training")
    triton_training.Training = lambda capability: SimpleNamespace(provider="triton")
    monkeypatch.setitem(sys.modules, "ctikernels.gdn.triton.training", triton_training)
    for operation in CONSTRUCTED:
        if operation.__name__ in refused:
            sm = f"sm_{capability[0]}{capability[1]}"
            with pytest.raises(
                RuntimeError,
                match=rf"{operation.__name__} on cuda:0 \(NVIDIA Test GPU, {sm}\) is not supported by fla-core 0\.5\.2",
            ):
                operation(num_k_heads=16, num_v_heads=32, device="cuda:0")
        else:
            operation(num_k_heads=16, num_v_heads=32, device="cuda:0")


# One accepted (device name, compute capability) per fused-prefill device type.
ACCEPTED_DEVICES = {
    _triton_prefill.RTX_5090: ("NVIDIA GeForce RTX 5090", (12, 0)),
    _triton_prefill.RTX_PRO_6000: ("NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0)),
    _triton_prefill.GB10: ("NVIDIA GB10", (12, 1)),
}
# The fused decode's range: every compute capability from 8.9.
DECODE_CAPABILITIES = [(8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)]


def _on_device(monkeypatch, name: str, capability: tuple[int, int]) -> None:
    """cuda:0 is ``name`` at ``capability``, with fla-core absent and the fused kernels' Triton
    modules stand-ins."""
    monkeypatch.setattr(_triton_decode, "load", _DECODE_LOAD)
    monkeypatch.setattr(_triton_prefill, "device_type", _PREFILL_DEVICE_TYPE)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=48),
    )
    kernels = ModuleType("ctikernels.gdn._triton_decode_kernels")
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    for leaf, entry, value in (
        ("_launch", "Launcher", object),
        ("_features", "features", None),
        ("_intra", "intra", None),
        ("_state", "state", None),
        ("_output", "output", None),
    ):
        kernels = ModuleType(f"ctikernels.gdn.triton.{leaf}")
        setattr(kernels, entry, value)
        monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    monkeypatch.setattr(importlib.metadata, "version", version(None))


@pytest.mark.parametrize("capability", DECODE_CAPABILITIES)
@pytest.mark.parametrize("heads", [(8, 16), (8, 24), (16, 32), (16, 48), (2, 6)])
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_decode_from_compute_capability_8_9_runs_the_fused_kernel_without_fla_core(
    capability, heads, batch_invariant, monkeypatch
) -> None:
    """Decode binds the fused kernel on any device of compute capability 8.9 or newer, whatever
    its name, and never needs fla-core; without CuTe, prefill binds our Triton prefill there."""
    _on_device(monkeypatch, "NVIDIA Test GPU", capability)
    monkeypatch.setattr(_fla, "load", lambda *arguments: pytest.fail("fla-core bound"))
    op = GatedDeltaNetDecode(
        num_k_heads=heads[0],
        num_v_heads=heads[1],
        device="cuda:0",
        batch_invariant=batch_invariant,
    )
    assert op.provider == "triton"
    assert isinstance(op._kernel, _triton_decode.TritonDecode)
    assert op._kernel._pdl == (capability >= (9, 0))
    assert op._kernel.batch_invariant == batch_invariant
    op = GatedDeltaNetPrefill(num_k_heads=heads[0], num_v_heads=heads[1], device="cuda:0")
    assert op.provider == "triton" and op.supports_state_capture


@pytest.mark.parametrize("capability", [(8, 0), (8, 6)])
def test_decode_below_compute_capability_8_9_needs_fla_core(capability, monkeypatch) -> None:
    _on_device(monkeypatch, "NVIDIA Test GPU", capability)
    with pytest.raises(RuntimeError, match="GatedDeltaNetDecode on cuda:0.*not installed"):
        GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:0")
    with pytest.raises(ValueError, match="batch_invariant needs the Triton decode"):
        GatedDeltaNetDecode(num_k_heads=16, num_v_heads=32, device="cuda:0", batch_invariant=True)


@pytest.mark.parametrize(
    ("device", "provider"),
    [
        (("NVIDIA GeForce RTX 5080", (12, 0)), "triton"),
        (("NVIDIA B200", (10, 0)), "triton"),
        (("NVIDIA GB300", (10, 3)), "triton"),
        (("NVIDIA H100 80GB HBM3", (9, 0)), "triton"),
        (("NVIDIA L40S", (8, 9)), "triton"),
        (("NVIDIA A100-SXM4-80GB", (8, 0)), "fla-core"),
    ],
)
def test_prefill_binds_triton_from_compute_capability_8_9_and_fla_core_below(
    device, provider, monkeypatch
) -> None:
    _on_device(monkeypatch, *device)
    if provider == "triton":
        monkeypatch.setattr(_fla, "load", lambda *arguments: pytest.fail("fla-core bound"))
        op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
        assert op.provider == "triton"
    else:
        with pytest.raises(RuntimeError, match="GatedDeltaNetPrefill on cuda:0.*not installed"):
            GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (lambda: [], TypeError),
        (lambda: _meta((3,), torch.int64), TypeError),
        (lambda: _meta((3, 1), torch.int32), ValueError),
        (lambda: _meta((2,), torch.int32), ValueError),
        (lambda: _meta((6,), torch.int32)[::2], ValueError),
        (lambda: torch.empty(3, dtype=torch.int32), ValueError),
    ],
)
def test_prefill_capture_rows_checked_before_launch(rows, error, recorder):
    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
    with pytest.raises(error):
        op(*_prefill_arguments().values(), capture_rows=rows())
    assert recorder.calls == []
    assert not op._guard._eager


def test_prefill_fallback_rejects_captures_before_launch(recorder):
    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
    recorder.provider = "fla-core"
    assert not op.supports_state_capture
    with pytest.raises(NotImplementedError, match="capture_rows needs the cute or triton provider"):
        op(*_prefill_arguments().values(), capture_rows=_meta((3,), torch.int32))
    assert recorder.calls == []
    assert not op._guard._eager


def test_prefill_frozen_weight_precedes_capture_checks(recorder):
    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda:0")
    args = _prefill_arguments()
    args["conv_weight"].requires_grad_()
    with pytest.raises(ValueError, match="frozen base weight"):
        op(*args.values(), capture_rows=[])
    assert recorder.calls == []


def test_prefill_dispatch_passes_capture_rows_unchanged(monkeypatch):
    monkeypatch.setattr(inference, "cuda_device", lambda *args: torch.device("cpu"))
    rows = torch.zeros(1, dtype=torch.int32)
    calls = []

    def kernel(geometry, projected, *args, out=None, capture_rows=None):
        assert capture_rows is rows
        calls.append(1)
        return torch.empty((projected.shape[0], geometry.value_width), dtype=torch.bfloat16)

    dispatch = object.__new__(_triton_prefill.TritonPrefill)
    dispatch._cute = kernel
    dispatch.provider = "cute"
    dispatch.selected_variant = lambda *args, **kwargs: "cute"
    monkeypatch.setattr(inference, "_prefill_kernel", lambda *args, **options: dispatch)
    op = GatedDeltaNetPrefill(num_k_heads=1, num_v_heads=2, device="cuda:0")
    assert op.supports_state_capture and op.CAPTURE_GRID == 64
    args = [
        torch.empty(t.shape, dtype=t.dtype)
        for t in _prefill_arguments(1, 2, tokens=4, sequences=1, slots=1).values()
    ]
    op(*args, capture_rows=rows)
    assert calls == [1]
