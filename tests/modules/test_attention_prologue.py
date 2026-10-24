# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Prologue on the CPU: where its CuTe kernel resolves (nowhere but a qualified device with the
pinned nvidia-cutlass-dsl, imported only then, once per device and variant), its launch table and the checks
that run before anything launches."""

from __future__ import annotations

import ast
import importlib.abc
import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ctikernels.attention
from ctikernels.attention import _prologue
from ctikernels.attention.inference import Prologue

from ..support.child import run_child

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "python"
# nvidia-cutlass-dsl's modules; its .pth imports only the small nvidia_cutlass_dsl wrapper at
# interpreter start, so "cutlass" (the DSL) is what must stay unimported.
CUTLASS = ("cutlass", "nvidia_cutlass_dsl")
DEVICE = torch.device("cuda", 0)
RTX_5090 = "NVIDIA GeForce RTX 5090"
L2_5090 = 96 << 20
VARIANTS = [(256, 64, True), (128, 128, False)]


class NoCutlass(importlib.abc.MetaPathFinder):
    """nvidia-cutlass-dsl as absent, recording each import of it."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in CUTLASS:
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def _installed(version: str | None):
    def lookup(name: str) -> str:
        assert name == _prologue.DISTRIBUTION, name
        if version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version

    return lookup


def _never(*args, **kwargs):
    raise AssertionError("looked up")


def _hardware(monkeypatch, name: str, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)


@pytest.fixture
def blocked(monkeypatch: pytest.MonkeyPatch) -> NoCutlass:
    """An RTX 5090 stand-in resolved afresh, without nvidia-cutlass-dsl."""
    finder = NoCutlass()
    for name in [name for name in sys.modules if name.split(".")[0] in CUTLASS]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "ctikernels.attention._prologue_cute", raising=False)
    monkeypatch.delattr(ctikernels.attention, "_prologue_cute", raising=False)
    monkeypatch.setattr(_prologue, "_launchers", {})
    _hardware(monkeypatch, RTX_5090, (12, 0))
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.delenv(_prologue.VARIABLE, raising=False)
    return finder


def test_off_cuda_it_is_none():
    for device in (torch.device("cpu"), torch.device("meta")):
        assert _prologue.load(device) is None


def test_cache_addresses_and_stores_are_guarded_on_masked_paths():
    """Masked K computes cache addresses (``k_addresses``) and stores only under ``slot >= 0``;
    masked V guards the shared store_v helper at its call."""
    tree = ast.parse((SOURCE_ROOT / "ctikernels/attention/_prologue_cute.py").read_text())
    names = {"finish_k_masked", "store_v", "copy_v", "norm_rope", "norm_rope_128", "k_addresses"}
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(methods) == names

    def paths(node, guards=()):
        """Track only true-branch ancestors; an else does not inherit its if's condition."""
        yield node, guards
        if isinstance(node, ast.If):
            yield from paths(node.test, guards)
            for child in node.body:
                yield from paths(child, (*guards, ast.unparse(node.test)))
            for child in node.orelse:
                yield from paths(child, guards)
        else:
            for child in ast.iter_child_nodes(node):
                yield from paths(child, guards)

    def masked_body(method, condition):
        branches = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.If) and ast.unparse(node.test) == condition
        ]
        assert len(branches) == 1
        return ast.Module(body=branches[0].body, type_ignores=[])

    def check_stores(node, guards=()):
        stores = targets = 0
        for child, ancestors in paths(node, guards):
            store = isinstance(child, ast.Call) and ast.unparse(child.func) in {
                "_store",
                "_store_if",
                "_store_bytes",
                "_st_at",
            }
            target = (
                isinstance(child, ast.Name)
                and child.id == "target"
                and isinstance(child.ctx, ast.Store)
            ) or (isinstance(child, ast.Call) and ast.unparse(child.func) == "self.k_addresses")
            if store or target:
                assert "slot >= 0" in ancestors, (child.lineno, ast.unparse(child))
                stores += store
                targets += target
        assert stores and targets

    masked_k = ast.unparse(ast.parse("const_expr(self.masked and not in_place)", mode="eval"))
    check_stores(methods["finish_k_masked"])
    check_stores(masked_body(methods["norm_rope_128"], masked_k))

    # store_v also serves the unmasked kernel. Follow its masked call with the caller's
    # actual branch ancestors instead of requiring a second guard in the helper itself.
    masked_v = masked_body(methods["copy_v"], "const_expr(self.masked)")
    calls = [
        (node, guards)
        for node, guards in paths(masked_v)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.store_v"
    ]
    assert calls
    for call, guards in calls:
        assert "slot >= 0" in guards, call.lineno
        check_stores(methods["store_v"], guards)

    # Every cache-address computation of the masked K path runs under slot >= 0.
    calls = [
        (node, guards)
        for node, guards in paths(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.k_addresses"
    ]
    assert len(calls) == 4
    for call, guards in calls:
        assert "slot >= 0" in guards, call.lineno
    slot_terms = [
        node
        for node in ast.walk(methods["k_addresses"])
        if isinstance(node, ast.Name) and node.id == "slot"
    ]
    assert slot_terms

    calls = [
        (node, guards)
        for node, guards in paths(methods["norm_rope"])
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self.finish_k_masked"
    ]
    assert calls
    for call, guards in calls:
        assert masked_k in guards, call.lineno


def test_docstring_says_negative_indices_skip_k_and_v_writes():
    tree = ast.parse((SOURCE_ROOT / "ctikernels/attention/_prologue.py").read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Prologue"
    )
    doc = " ".join(ast.get_docstring(cls).split())
    assert (
        "negative when ``masked``: a negative index (padding) skips that token's K and V cache writes"
        in doc
    )
    assert "Without ``masked`` negative indices are not allowed" in doc
    assert "pad with a sink slot" in doc
    assert "its Q is still normalized and rotated" in doc


@pytest.mark.parametrize("masked", [None, 0, 1, 0.0, 1.0, "True", [], {}])
def test_masked_requires_bool_before_device_handling(monkeypatch, masked):
    monkeypatch.setattr(_prologue, "load", _never)
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    with pytest.raises(TypeError, match="masked must be a bool"):
        Prologue(num_q_heads=8, num_kv_heads=2, eps=1e-6, device="cpu", masked=masked)


@pytest.mark.parametrize("masked", [False, True])
def test_masked_bool_is_accepted_without_device_access(monkeypatch, masked):
    monkeypatch.setattr(_prologue, "load", lambda device, variant: None)
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    op = Prologue(num_q_heads=8, num_kv_heads=2, eps=1e-6, device=DEVICE, masked=masked)
    assert op.masked is masked


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cache", [None, "e4m3"])
def test_launcher_keys_distinguish_masked_from_default(variant, cache):
    launcher = object.__new__(_prologue.Launcher)
    launcher._l2_bytes = L2_5090
    launcher._variant, launcher._axes = variant, None
    default = launcher.keys(512, 16, 4, cache=cache)
    masked = launcher.keys(512, 16, 4, cache=cache, masked=True)
    assert default and masked
    assert all(key[-2] is False for key in default)
    assert all(key[-2] is True for key in masked)
    assert {key[:-2]: count for key, count in default.items()} == {
        key[:-2]: count for key, count in masked.items()
    }


@pytest.mark.parametrize("version", [None, "4.6.0.dev0", "4.8.1"])
def test_without_the_pinned_version_it_is_none_and_imports_nothing(blocked, monkeypatch, version):
    monkeypatch.setattr(importlib.metadata, "version", _installed(version))
    monkeypatch.setattr(torch.cuda, "get_device_name", _never)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _never)
    assert _prologue.load(DEVICE) is None
    assert blocked.attempts == []


@pytest.mark.parametrize(
    "hardware",
    [
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", (12, 0)),
        ("NVIDIA H100 80GB HBM3", (9, 0)),
        ("NVIDIA GeForce RTX 4090", (8, 9)),
    ],
)
def test_on_other_device_types_it_is_none_and_imports_nothing(blocked, monkeypatch, hardware):
    _hardware(monkeypatch, *hardware)
    monkeypatch.setattr(importlib.metadata, "version", _installed(_prologue.VERSION))
    assert _prologue.load(DEVICE) is None
    assert blocked.attempts == []


def test_the_variable_turns_it_off_before_any_lookup(blocked, monkeypatch):
    monkeypatch.setenv(_prologue.VARIABLE, "0")
    _hardware(monkeypatch, _never, _never)
    monkeypatch.setattr(torch.cuda, "get_device_name", _never)
    monkeypatch.setattr(importlib.metadata, "version", _never)
    assert _prologue.load(DEVICE) is None


def test_a_pinned_install_that_does_not_import_warns_once_and_is_none(blocked, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _installed(_prologue.VERSION))
    with pytest.warns(RuntimeWarning, match="does not import"):
        assert _prologue.load(DEVICE) is None
    assert "cutlass" not in sys.modules
    monkeypatch.setattr(importlib.metadata, "version", _never)
    blocked.attempts.clear()
    assert _prologue.load(DEVICE) is None
    assert blocked.attempts == []


@pytest.mark.parametrize("heads", [(0, 4), (16, 0)])
def test_invalid_head_counts_raise(heads):
    with pytest.raises(ValueError, match="positive"):
        Prologue(num_q_heads=heads[0], num_kv_heads=heads[1], eps=1e-6, device=DEVICE)


@pytest.mark.parametrize(
    "dtypes",
    [
        {"dtype": torch.float16},
        {"dtype": torch.float32},
        {"dtype": torch.float8_e4m3fn},
        {"dtype": torch.float8_e5m2},
        {"dtype": torch.uint8},
        {"cache_dtype": torch.float32},
        {"cache_dtype": torch.uint8},
        {"cache_dtype": "bfloat16"},
        {"cache_dtype": torch.float16},
        {"dtype": "bfloat16"},
    ],
)
def test_unsupported_dtypes_raise_at_construction(monkeypatch, dtypes):
    monkeypatch.setattr(_prologue, "load", _never)
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    with pytest.raises(TypeError, match="dtype"):
        Prologue(num_q_heads=8, num_kv_heads=2, eps=1e-6, device=DEVICE, **dtypes)


def test_bf16_dtypes_are_accepted(monkeypatch):
    monkeypatch.setattr(_prologue, "load", lambda device, variant: None)
    op = Prologue(
        num_q_heads=8, num_kv_heads=2, eps=1e-6, device=DEVICE,
        dtype=torch.bfloat16, cache_dtype=torch.bfloat16,
    )  # fmt: skip
    assert op.provider == "compose"


def test_a_cpu_device_raises():
    with pytest.raises(ValueError, match="CUDA"):
        Prologue(num_q_heads=16, num_kv_heads=4, eps=1e-6, device="cpu")


def test_provider_is_compose_without_the_extra_or_device_access(blocked, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", _installed(None))
    for name in (
        "current_device",
        "get_device_name",
        "get_device_capability",
        "get_device_properties",
    ):
        monkeypatch.setattr(torch.cuda, name, _never)
    monkeypatch.setattr(_prologue, "_compose", _never)
    op = Prologue(num_q_heads=16, num_kv_heads=4, eps=1e-6, device=DEVICE)
    monkeypatch.setattr(_prologue, "load", _never)
    assert op.provider == "compose"
    assert blocked.attempts == []
    with pytest.raises(AttributeError):
        op.provider = "cute"


@pytest.mark.parametrize(
    ("variant", "hardware"),
    [
        (variant, hardware)
        for variant, devices in _prologue.DEVICES.items()
        for hardware in sorted(devices)
    ],
)
def test_provider_is_cute_without_compiling_or_launching(blocked, monkeypatch, variant, hardware):
    empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *args, **kwargs: empty(*args, **dict(kwargs, device="cpu"))
    )
    _hardware(monkeypatch, *hardware)
    monkeypatch.setattr(importlib.metadata, "version", _installed(_prologue.VERSION))
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=L2_5090)
    )
    monkeypatch.setattr(
        ctikernels.attention, "_prologue_cute", SimpleNamespace(Kernel=_never), raising=False
    )
    monkeypatch.setattr(_prologue.Launcher, "__call__", _never)
    monkeypatch.setattr(_prologue, "_compile", _never)
    op = Prologue(
        num_q_heads=16,
        num_kv_heads=4,
        eps=1e-6,
        device=DEVICE,
        head_dim=variant[0],
        rotary_dim=variant[1],
        zero_centered=variant[2],
    )
    monkeypatch.setattr(_prologue, "load", _never)
    assert op.provider == "cute"
    arch = "sm_121a" if hardware[1] == (12, 1) else "sm_120a"
    assert op._kernel._options == f"--enable-tvm-ffi --gpu-arch {arch}"
    assert op._kernel._variant == variant
    assert blocked.attempts == []
    with pytest.raises(AttributeError):
        op.provider = "compose"


@pytest.mark.parametrize("trainable", ["q_weight", "k_weight", "cos_sin_cache", "qkv", "k_cache"])
def test_gradients_raise_before_anything_else(monkeypatch, trainable):
    monkeypatch.setattr(_prologue, "load", lambda device, variant: None)
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    # An integer device asks torch to resolve an accelerator, even with load stubbed.
    # The explicit descriptor needs no accelerator; the inputs below stay on CPU.
    op = Prologue(num_q_heads=2, num_kv_heads=1, eps=1e-6, device=DEVICE)
    monkeypatch.setattr(_prologue, "_compose", _never)
    tensors = {
        "qkv": torch.zeros(3, 4 * 256),
        "q_weight": torch.zeros(256),
        "k_weight": torch.zeros(256),
        "cos_sin_cache": torch.zeros(8, 64),
        "positions": torch.zeros(3, dtype=torch.int64),
        "k_cache": torch.zeros(8, 1, 256),
        "v_cache": torch.zeros(8, 1, 256),
        "cache_indices": torch.zeros(3, dtype=torch.int32),
    }
    tensors[trainable].requires_grad_()
    with pytest.raises(ValueError, match="grad"):
        op(**tensors)


@pytest.mark.parametrize(
    ("tokens", "q_heads", "kv_heads", "expected"),
    [
        (1, 16, 4, (1, 1, 1, 4, False)),
        (2, 16, 4, (1, 1, 1, 2, False)),
        (9, 16, 4, (1, 1, 1, 4, False)),
        (48, 12, 2, (1, 1, 1, 8, False)),
        (256, 8, 2, (2, 2, 2, 4, False)),
        (200, 24, 4, (4, 2, 2, 8, False)),
        (8192, 16, 4, (2, 2, 2, 4, True)),
    ],
)
def test_launch_configurations(tokens, q_heads, kv_heads, expected):
    assert _prologue.config(tokens, q_heads, kv_heads, L2_5090) == expected


def test_importing_the_operation_never_imports_cutlass(tmp_path: Path):
    code = """
import sys

import torch

from ctikernels.attention import _prologue
from ctikernels.attention.inference import Prologue  # noqa: F401

assert _prologue.load(torch.device("cpu")) is None
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    run_child((sys.executable, "-c", code), env=environment, cwd=tmp_path, timeout=600, check=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_properties_are_read_only(blocked, monkeypatch, variant):
    monkeypatch.setattr(importlib.metadata, "version", _installed(None))
    kwargs = dict(zip(("head_dim", "rotary_dim", "zero_centered"), variant, strict=True))
    op = Prologue(num_q_heads=32, num_kv_heads=8, eps=1e-6, device=DEVICE, **kwargs)
    for name, value in kwargs.items():
        assert getattr(op, name) == value
        with pytest.raises(AttributeError):
            setattr(op, name, value)
    default = Prologue(num_q_heads=16, num_kv_heads=4, eps=1e-6, device=DEVICE)
    assert (default.head_dim, default.rotary_dim, default.zero_centered) == VARIANTS[0]


@pytest.mark.parametrize(
    "variant",
    [
        (128, 128, True),
        (128, 64, False),
        (256, 64, False),
        (256, 128, True),
        (64, 64, False),
        (128, 0, False),
    ],
)
def test_unsupported_variants_raise_before_device_lookup(monkeypatch, variant):
    monkeypatch.setattr(_prologue, "load", _never)
    monkeypatch.setattr(torch.cuda, "current_device", _never)
    with pytest.raises(ValueError, match="unsupported"):
        Prologue(
            num_q_heads=32,
            num_kv_heads=8,
            eps=1e-6,
            device="cuda",
            head_dim=variant[0],
            rotary_dim=variant[1],
            zero_centered=variant[2],
        )


def test_gb10_resolution_is_cached_per_variant(blocked, monkeypatch):
    _hardware(monkeypatch, "NVIDIA GB10", (12, 1))
    monkeypatch.setattr(importlib.metadata, "version", _installed(_prologue.VERSION))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=24 << 20)
    )
    monkeypatch.setattr(
        ctikernels.attention, "_prologue_cute", SimpleNamespace(Kernel=_never), raising=False
    )
    launchers = [_prologue.load(DEVICE, variant) for variant in VARIANTS]
    assert all(launcher is not None for launcher in launchers)
    monkeypatch.setattr(importlib.metadata, "version", _never)
    assert [_prologue.load(DEVICE, variant) for variant in VARIANTS] == launchers
    assert blocked.attempts == []


def _cpu_operation(
    monkeypatch, variant, cache_dtype=torch.bfloat16, kernel=None, position_axes=None
):
    ones = torch.ones
    monkeypatch.setattr(
        torch, "ones", lambda *args, **kwargs: ones(*args, **dict(kwargs, device="cpu"))
    )
    empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: empty(
            *args, **(dict(kwargs, device="cpu") if kwargs.get("device") == DEVICE else kwargs)
        ),
    )
    monkeypatch.setattr(_prologue, "load", lambda device, variant, axes=None: kernel)
    op = Prologue(
        num_q_heads=8,
        num_kv_heads=2,
        eps=1e-6,
        device=DEVICE,
        head_dim=variant[0],
        rotary_dim=variant[1],
        zero_centered=variant[2],
        cache_dtype=cache_dtype,
        position_axes=position_axes,
    )
    # Exercise metadata checks without CUDA; the public constructor still requires CUDA.
    op.device = torch.device("cpu")
    # Match the CPU descriptor's index without bypassing the per-call device check.
    monkeypatch.setattr(
        torch.Tensor,
        "get_device",
        lambda tensor: None if tensor.device.type == "cpu" else -1,
    )
    return op


def _cpu_inputs(variant):
    hd, rd, _ = variant
    return {
        "qkv": torch.randn(3, 12 * hd, dtype=torch.bfloat16),
        "q_weight": torch.ones(hd, dtype=torch.bfloat16),
        "k_weight": torch.ones(hd, dtype=torch.bfloat16),
        "cos_sin_cache": torch.randn(8, rd),
        "positions": torch.tensor([0, 3, 7], dtype=torch.int64),
        "k_cache": torch.zeros(8, 2, hd, dtype=torch.bfloat16),
        "v_cache": torch.zeros(8, 2, hd, dtype=torch.bfloat16),
        "cache_indices": torch.tensor([1, 4, 6], dtype=torch.int32),
    }


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "name", ["qkv", "q_weight", "k_weight", "cos_sin_cache", "k_cache", "v_cache"]
)
def test_shapes_use_the_variant(monkeypatch, variant, name):
    op, inputs = _cpu_operation(monkeypatch, variant), _cpu_inputs(variant)
    op._check(**inputs)
    monkeypatch.setattr(_prologue, "_compose", _never)
    # Substitute the other variant's width, including valid alignment and contiguous slots.
    other = _cpu_inputs(VARIANTS[1] if variant == VARIANTS[0] else VARIANTS[0])
    inputs[name] = other[name]
    with pytest.raises(ValueError, match=name):
        op(**inputs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_compose_passes_variant_and_returns_q_view(monkeypatch, variant):
    op, inputs = _cpu_operation(monkeypatch, variant), _cpu_inputs(variant)
    calls = []

    def norm(x, weight, eps, *, zero_centered):
        calls.append(("norm", x.shape, zero_centered))
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        x.copy_((normalized * (weight.float() + int(zero_centered))).bfloat16())

    def rotary(q, k, table, positions, *, head_dim, position_axes=None):
        calls.append(("rotary", head_dim))
        for tensor in (q, k):
            x = tensor.view(3, -1, head_dim)
            half = table.shape[-1] // 2
            cos, sin = table[positions][:, None].chunk(2, -1)
            lo, hi = x[..., :half].float(), x[..., half : 2 * half].float()
            x[..., :half] = (lo * cos - hi * sin).bfloat16()
            x[..., half : 2 * half] = (hi * cos + lo * sin).bfloat16()

    def store(k_cache, v_cache, indices, k, v):
        calls.append(("store",))
        k_cache[indices.long()] = k.reshape(3, 2, op.head_dim)
        v_cache[indices.long()] = v.reshape(3, 2, op.head_dim)

    for module, functions in (
        ("ctikernels.norm.inference", {"rms_norm_": norm}),
        ("ctikernels.rotary.inference", {"apply_rotary_": rotary}),
        ("ctikernels.store", {"store_cache": store}),
    ):
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**functions))
    original = inputs["qkv"].clone()
    q = op(**inputs)
    assert calls == [
        ("norm", (3, 8, op.head_dim), op.zero_centered),
        ("norm", (3, 2, op.head_dim), op.zero_centered),
        ("rotary", op.head_dim),
        ("store",),
    ]
    assert q.shape == (3, 8, op.head_dim)
    assert q.data_ptr() == inputs["qkv"].data_ptr()
    assert not torch.equal(q.flatten(1), original[:, : 8 * op.head_dim])
    assert torch.equal(
        inputs["v_cache"][inputs["cache_indices"].long()],
        original[:, 10 * op.head_dim :].reshape(3, 2, op.head_dim),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_launcher_configuration_override(monkeypatch, variant):
    _hardware(monkeypatch, RTX_5090, (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=L2_5090)
    )
    launcher = _prologue.Launcher(DEVICE, SimpleNamespace(Kernel=_never), variant)
    keys, launches = [], []

    def compile_key(key):
        keys.append(key)
        return lambda *args: launches.append(args)

    monkeypatch.setattr(launcher, "_compile", compile_key)
    inputs = _cpu_inputs(variant)
    override = (4, 2, 1, 2, True)
    launcher(*inputs.values(), 1e-6, 8, 2, config=override)
    assert keys == [(8, 2, *override, *variant, None, False, None)]
    assert launches[0][-3:] == (3, 2, 8)


@pytest.mark.parametrize(
    "head_dim, expected", [(256, (4, 2, 2, 4, True)), (128, (4, 2, 2, 8, False))]
)
def test_streaming_uses_variant_bytes(head_dim, expected):
    assert _prologue.config(1000, 16, 4, L2_5090, head_dim) == expected


FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_fp8_constructor_and_default_scales(monkeypatch, dtype):
    op = _cpu_operation(monkeypatch, VARIANTS[0], dtype)
    assert op.cache_dtype == dtype
    assert op.provider == "compose"
    k, v = op._scales(None, None)
    assert k.dtype == v.dtype == torch.float32
    assert k.shape == v.shape == (1,)
    assert k.item() == v.item() == 1
    explicit = torch.tensor([[0.0478]])
    assert op._scales(explicit, None)[0].data_ptr() == explicit.data_ptr()


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
@pytest.mark.parametrize(
    "kind,error",
    [
        ("scalar", TypeError),
        ("dtype", TypeError),
        ("empty", TypeError),
        ("many", TypeError),
        ("device", ValueError),
        ("grad", ValueError),
        ("alignment", ValueError),
    ],
)
def test_fp8_scale_validation(monkeypatch, dtype, name, kind, error):
    op = _cpu_operation(monkeypatch, VARIANTS[0], dtype)
    scales = {
        "scalar": 1.0,
        "dtype": torch.tensor([1.0], dtype=torch.bfloat16),
        "empty": torch.empty(0),
        "many": torch.empty(2),
        "device": torch.empty(1, device="meta"),
        "grad": torch.tensor([1.0], requires_grad=True),
        "alignment": torch.frombuffer(bytearray(5), dtype=torch.float32, offset=1),
    }
    inputs = _cpu_inputs(VARIANTS[0])
    for cache in ("k_cache", "v_cache"):
        inputs[cache] = inputs[cache].to(dtype)
    monkeypatch.setattr(_prologue, "_compose", _never)
    with pytest.raises(error, match=name):
        op(**inputs, **{name: scales[kind]})


@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
def test_bf16_cache_rejects_scales(monkeypatch, name):
    op = _cpu_operation(monkeypatch, VARIANTS[0])
    monkeypatch.setattr(_prologue, "_compose", _never)
    with pytest.raises(ValueError, match="FP8"):
        op(**_cpu_inputs(VARIANTS[0]), **{name: torch.tensor([1.0])})


@pytest.mark.parametrize("dtype", [torch.bfloat16, *FP8_DTYPES])
@pytest.mark.parametrize("name", ["k_cache", "v_cache"])
def test_cache_dtype_must_match_constructor(monkeypatch, dtype, name):
    op = _cpu_operation(monkeypatch, VARIANTS[0], dtype)
    inputs = _cpu_inputs(VARIANTS[0])
    for cache in ("k_cache", "v_cache"):
        inputs[cache] = inputs[cache].to(dtype)
    op._check(**inputs)
    inputs[name] = inputs[name].to(
        torch.float8_e5m2 if dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
    )
    with pytest.raises(TypeError, match=name):
        op._check(**inputs)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("padding,accepted", [(0, True), (8, False), (16, True), (2**31, False)])
def test_fp8_slot_stride_is_sixteen_bytes(monkeypatch, dtype, padding, accepted):
    op = _cpu_operation(monkeypatch, VARIANTS[0], dtype)
    inputs = _cpu_inputs(VARIANTS[0])
    # One slot permits testing the 32-bit stride bound without a huge allocation.
    for name in ("k_cache", "v_cache"):
        inputs[name] = torch.empty_strided((1, 2, 256), (512 + padding, 256, 1), dtype=dtype)
    if accepted:
        op._check(**inputs)
    else:
        with pytest.raises(ValueError, match="slot stride"):
            op._check(**inputs)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_quantize_saturates_preserves_nan_and_rounds_ties_to_even(dtype):
    info = torch.finfo(dtype)
    step = info.eps
    subnormal = info.tiny * step
    values = torch.tensor(
        [
            -float("inf"),
            -2 * info.max,
            -info.max,
            info.max,
            2 * info.max,
            float("inf"),
            float("nan"),
            1 + step / 2,
            1 + 3 * step / 2,
            -1 - step / 2,
            -1 - 3 * step / 2,
            subnormal / 2,
            1.5 * subnormal,
        ]
    )
    result = _prologue._quantize(values * 2, torch.tensor([2.0]), dtype)
    assert result.dtype == torch.uint8
    assert result.shape == values.shape
    decoded = result.view(dtype).float()
    assert torch.equal(decoded[:6], torch.tensor([-info.max] * 3 + [info.max] * 3))
    assert torch.isnan(decoded[6])
    assert torch.equal(
        decoded[7:], torch.tensor([1, 1 + 2 * step, -1, -1 - 2 * step, 0, 2 * subnormal])
    )


def _count_full(monkeypatch, op):
    calls = []
    full = op._check_full

    def checked(*args, **kwargs):
        calls.append(1)
        return full(*args, **kwargs)

    monkeypatch.setattr(op, "_check_full", checked)
    return calls


def test_bound_layers_skip_full_validation_and_new_cache_views(monkeypatch):
    op = _cpu_operation(monkeypatch, VARIANTS[0])
    layers = [_cpu_inputs(VARIANTS[0]) for _ in range(3)]
    for inputs in layers[1:]:
        for name in ("positions", "cache_indices", "cos_sin_cache"):
            inputs[name] = layers[0][name]
    pools = [[inputs[name].unsqueeze(1) for name in ("k_cache", "v_cache")] for inputs in layers]
    calls = _count_full(monkeypatch, op)
    for index, inputs in enumerate(layers):
        assert op._check(**inputs) == 3
        assert len(calls) == index + 1
    for _ in range(2):
        for inputs, pair in zip(layers, pools, strict=True):
            for name, pool in zip(("k_cache", "v_cache"), pair, strict=True):
                inputs[name] = pool.squeeze(1)
            assert op._check(**inputs) == 3
    assert len(calls) == 3


@pytest.mark.parametrize("name", ["positions", "cache_indices"])
def test_new_step_object_triggers_exactly_one_full_check(monkeypatch, name):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    calls = _count_full(monkeypatch, op)
    op._check(**inputs)
    assert len(calls) == 1
    original = inputs[name]
    inputs[name] = original.detach()
    assert inputs[name] is not original and inputs[name].data_ptr() == original.data_ptr()
    op._check(**inputs)
    assert len(calls) == 2
    op._check(**inputs)
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["q_weight", "k_weight", "cos_sin_cache", "k_cache", "v_cache"])
@pytest.mark.parametrize("change", ["shape", "dtype", "stride", "pointer", "object"])
def test_bound_mutations_raise_the_original_error(monkeypatch, name, change):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    calls = _count_full(monkeypatch, op)
    tensor = inputs[name]
    if change == "shape":
        bad = tensor[..., :-1]
    elif change == "dtype":
        bad = tensor.to(torch.float64)
    elif change == "stride":
        # Change only strides: identity, pointer, shape and dtype stay bound.
        strides = list(tensor.stride())
        strides[-1] = 0
        bad = tensor.as_strided(tensor.shape, strides)
    elif change == "object":
        bad = tensor[..., :-1]
    else:
        # Preserve shape/strides but move the same object's data to an unaligned pointer.
        bad = torch.empty(tensor.numel() + 1, dtype=tensor.dtype)[1:].view(tensor.shape)
    if change == "object":
        inputs[name] = bad
    else:
        tensor.data = bad
    with pytest.raises((TypeError, ValueError)) as expected:
        op._check_full(**inputs)
    with pytest.raises(type(expected.value)) as actual:
        op._check(**inputs)
    assert str(actual.value) == str(expected.value)
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["q_weight", "k_weight", "cos_sin_cache"])
@pytest.mark.parametrize("change", ["object", "pointer"])
def test_valid_rebinding_runs_full_check(monkeypatch, name, change):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    calls = _count_full(monkeypatch, op)
    op._check(**inputs)
    if change == "object":
        inputs[name] = inputs[name].detach()
    else:
        inputs[name].data = inputs[name].clone()
    op._check(**inputs)
    op._check(**inputs)
    assert len(calls) == 2


def test_bound_table_outer_stride_change_preserves_original_error(monkeypatch):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    table = inputs["cos_sin_cache"]
    table.as_strided_(table.shape, (0, 1))
    with pytest.raises(ValueError, match="cos_sin_cache must be contiguous"):
        op._check_full(**inputs)
    with pytest.raises(ValueError, match="cos_sin_cache must be contiguous"):
        op._check(**inputs)


@pytest.mark.parametrize("tokens", [0, 1])
@pytest.mark.parametrize("name", ["positions", "cache_indices", "cos_sin_cache"])
def test_small_bound_tensors_reject_unit_stride_violation(monkeypatch, tokens, name):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    for key in ("qkv", "positions", "cache_indices", "cos_sin_cache"):
        inputs[key] = inputs[key][:tokens]
    op._check(**inputs)
    tensor = inputs[name]
    tensor.as_strided_(tensor.shape, (*tensor.stride()[:-1], 0))
    with pytest.raises(ValueError, match=f"{name} must be aligned with unit inner stride"):
        op._check_full(**inputs)
    with pytest.raises(ValueError, match=f"{name} must be aligned with unit inner stride"):
        op._check(**inputs)


@pytest.mark.parametrize(
    "name", ["qkv", "q_weight", "k_weight", "cos_sin_cache", "k_cache", "v_cache"]
)
def test_bound_grad_toggle_still_raises(monkeypatch, name):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    inputs[name].requires_grad_()
    with pytest.raises(ValueError) as expected:
        op._check_full(**inputs)
    with pytest.raises(ValueError) as actual:
        op._check(**inputs)
    assert str(actual.value) == str(expected.value)


def test_bound_caches_are_bounded_and_do_not_own_tensors(monkeypatch):
    import weakref

    op = _cpu_operation(monkeypatch, VARIANTS[0])
    monkeypatch.setattr(_prologue, "_VALIDATED_LIMIT", 3)
    layers = [_cpu_inputs(VARIANTS[0]) for _ in range(4)]
    refs = [weakref.ref(tensor) for inputs in layers for tensor in inputs.values()]
    calls = _count_full(monkeypatch, op)
    for index in range(3):
        op._check(**layers[index])
        assert set(op._bound) == {
            (name, id(layers[index][name])) for name in ("q_weight", "k_weight", "cos_sin_cache")
        }
        assert len(op._bound_caches) == index + 1
    op._check(**layers[3])
    assert len(op._bound) == 3 and len(op._bound_caches) == 1
    assert len(calls) == 4
    op._check(**layers[0])  # Evicted tensors need validation again.
    assert len(calls) == 5
    del layers
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("name", ["positions", "cache_indices"])
@pytest.mark.parametrize("change", ["dtype", "shape", "stride", "pointer", "device"])
def test_step_tensor_mutation_is_checked_on_every_call(monkeypatch, name, change):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    tensor = inputs[name]
    if change == "dtype":
        tensor.data = tensor.float()
    elif change == "shape":
        tensor.resize_(2)
    elif change == "stride":
        tensor.as_strided_(tensor.shape, (0,))
    elif change == "pointer":
        tensor.data = torch.frombuffer(
            bytearray(3 * tensor.element_size() + 1), dtype=tensor.dtype, offset=1
        )
    else:
        inputs[name] = tensor.to("meta")
    with pytest.raises((TypeError, ValueError)) as expected:
        op._check_full(**inputs)
    with pytest.raises(type(expected.value)) as actual:
        op._check(**inputs)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
def test_bound_scale_rechecks_mutation_and_grad(monkeypatch, name):
    op = _cpu_operation(monkeypatch, VARIANTS[0], torch.float8_e4m3fn)
    scale = torch.ones(1)
    calls = []
    full = op._check_scale

    def checked(*args):
        calls.append(1)
        return full(*args)

    monkeypatch.setattr(op, "_check_scale", checked)
    for _ in range(2):
        op._scales(**{name: scale, "v_scale" if name == "k_scale" else "k_scale": None})
    assert len(calls) == 1
    scale.requires_grad_()
    with pytest.raises(ValueError, match=f"{name} must not require grad"):
        op._scales(**{name: scale, "v_scale" if name == "k_scale" else "k_scale": None})
    scale.requires_grad_(False)
    scale.data = scale.double()
    with pytest.raises(TypeError, match=f"{name} must be a one-element torch.float32 tensor"):
        op._scales(**{name: scale, "v_scale" if name == "k_scale" else "k_scale": None})
    assert len(calls) == 3


@pytest.mark.parametrize("tokens", [0, 1, 3])
@pytest.mark.parametrize("offset", [0, 1])
def test_return_view_metadata_is_identical(monkeypatch, tokens, offset):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    rows = torch.empty(5, 13 * op.head_dim, dtype=torch.bfloat16)
    inputs["qkv"] = rows[offset : offset + tokens, : 12 * op.head_dim]
    inputs["positions"] = inputs["positions"][:tokens]
    inputs["cache_indices"] = inputs["cache_indices"][:tokens]
    monkeypatch.setattr(_prologue, "_compose", lambda *args: None)
    q = op(**inputs)
    expected = inputs["qkv"][:, : 8 * op.head_dim].view(tokens, 8, op.head_dim)
    assert q.shape == expected.shape
    assert q.stride() == expected.stride()
    assert q.storage_offset() == expected.storage_offset()
    assert q._base is expected._base
    assert q.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()


def _fake_launcher(monkeypatch):
    _hardware(monkeypatch, RTX_5090, (12, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(L2_cache_size=L2_5090)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    compiles, launches, loads = [], [], []

    def compile_kernel(kernel, *args):
        compiles.append(kernel)
        return lambda *args: launches.append(args)

    monkeypatch.setattr(_prologue, "_compile", compile_kernel)
    monkeypatch.setattr(_prologue.Launcher, "_load", lambda self, key, compiled: loads.append(key))
    launcher = _prologue.Launcher(DEVICE, SimpleNamespace(Kernel=lambda *key: key))
    return launcher, compiles, launches, loads


def test_warm_up_fills_plans_and_hits_skip_config_and_capture_checks(monkeypatch):
    launcher, compiles, _, loads = _fake_launcher(monkeypatch)
    count = launcher.warm_up(128, 32, 8)
    assert count == len(compiles) > 1
    assert loads == compiles  # each compiled kernel is launched (loaded) once at compile
    assert set(launcher._plans[(32, 8, None, False)]) == set(range(1, 129))
    for tokens in range(1, 129):
        config = _prologue.config(tokens, 32, 8, L2_5090)
        q, k, v, warps, _ = config
        counts = [
            -(-tokens * (heads // rows) // warps) for heads, rows in ((32, q), (8, k), (8, v))
        ]
        assert launcher.plan(tokens, 32, 8)[1:] == (*counts[:2], sum(counts))
    monkeypatch.setattr(_prologue, "_default_config", _never)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", _never)
    assert launcher.warm_up(128, 32, 8) == 0
    assert launcher.plan(64, 32, 8) is launcher.plan(64, 32, 8)


def test_capture_compile_miss_is_rejected_before_compilation(monkeypatch):
    launcher, compiles, _, _ = _fake_launcher(monkeypatch)
    launcher.warm_up(1, 32, 8)
    before = len(launcher._compiled)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm_up.*eagerly"):
        launcher.plan(128, 32, 8)
    assert len(compiles) == len(launcher._compiled) == before
    assert 128 not in launcher._plans[(32, 8, None, False)]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    launcher.plan(128, 32, 8)
    assert len(compiles) == before + 1


def test_bf16_launch_reuses_unread_scales(monkeypatch):
    launcher, _, launches, _ = _fake_launcher(monkeypatch)
    op = _cpu_operation(monkeypatch, VARIANTS[0], kernel=launcher)
    inputs = _cpu_inputs(VARIANTS[0])
    op(**inputs)
    assert op._plans[3] is launcher.plan(3, 8, 2)
    monkeypatch.setattr(launcher, "plan", _never)
    monkeypatch.setattr(torch, "empty", _never)
    op(**inputs)
    for index in (8, 9):
        assert launches[0][index] is launches[1][index] is op._unread_scales[index - 8]
        assert launches[0][index].shape == (1,)
        assert launches[0][index].dtype == torch.float32


def test_op_plan_table_clears_at_limit_and_reuses_launcher_plans(monkeypatch):
    launcher, compiles, _, _ = _fake_launcher(monkeypatch)
    op = _cpu_operation(monkeypatch, VARIANTS[0], kernel=launcher)
    monkeypatch.setattr(_prologue, "_VALIDATED_LIMIT", 2)
    plans = [op._plan(tokens) for tokens in (1, 2)]
    assert set(op._plans) == {1, 2}
    op._plan(3)
    assert set(op._plans) == {3}
    before = len(compiles)
    monkeypatch.setattr(_prologue, "_default_config", _never)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", _never)
    assert op._plan(1) is plans[0]
    assert set(op._plans) == {1, 3}
    assert len(compiles) == before


@pytest.mark.parametrize("name", ["q_weight", "k_weight", "cos_sin_cache", "k_cache", "v_cache"])
def test_unsupported_bound_layout_keeps_original_dtype_error(monkeypatch, name):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    inputs[name] = inputs[name].double().to_sparse()
    with pytest.raises(TypeError) as expected:
        op._check_full(**inputs)
    with pytest.raises(TypeError) as actual:
        op._check(**inputs)
    assert str(actual.value) == str(expected.value)


def test_capture_guard_applies_to_explicit_config_overrides(monkeypatch):
    launcher, compiles, _, _ = _fake_launcher(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm_up.*eagerly"):
        launcher(*_cpu_inputs(VARIANTS[0]).values(), 1e-6, 8, 2, config=(1, 1, 1, 4, False))
    assert not compiles and not launcher._compiled


def test_fp8_launch_preserves_byte_views_and_real_scales(monkeypatch):
    op, inputs = (
        _cpu_operation(monkeypatch, VARIANTS[0], torch.float8_e4m3fn),
        _cpu_inputs(VARIANTS[0]),
    )
    for name in ("k_cache", "v_cache"):
        inputs[name] = inputs[name].to(torch.float8_e4m3fn)
    launcher, _, launches, _ = _fake_launcher(monkeypatch)
    op._kernel = launcher
    scales = (torch.tensor([[0.5]]), torch.tensor(1.5))
    op(**inputs, k_scale=scales[0], v_scale=scales[1])
    for index, name in ((6, "k_cache"), (7, "v_cache")):
        passed = launches[0][index]
        assert passed.dtype == torch.uint8
        assert passed.data_ptr() == inputs[name].data_ptr()
        assert passed.shape == inputs[name].shape and passed.stride() == inputs[name].stride()
    for passed, original in zip(launches[0][8:10], scales, strict=True):
        assert passed.shape == (1,) and passed.data_ptr() == original.data_ptr()


@pytest.mark.parametrize(
    "change",
    [
        "dtype",
        "shape",
        "scalar",
        "rank3",
        "width",
        "device",
        "pointer",
        "stride",
        "row_stride",
        "unaligned_row_stride",
    ],
)
def test_qkv_still_gets_full_checks_after_binding(monkeypatch, change):
    op, inputs = _cpu_operation(monkeypatch, VARIANTS[0]), _cpu_inputs(VARIANTS[0])
    op._check(**inputs)
    qkv = inputs["qkv"]
    if change == "dtype":
        inputs["qkv"] = qkv.float()
    elif change == "shape":
        inputs["qkv"] = qkv.flatten()
    elif change == "scalar":
        inputs["qkv"] = qkv[0, 0]
    elif change == "rank3":
        inputs["qkv"] = qkv.unsqueeze(0)
    elif change == "width":
        inputs["qkv"] = qkv[:, :-8]
    elif change == "device":
        inputs["qkv"] = qkv.to("meta")
    elif change == "pointer":
        inputs["qkv"] = torch.empty(qkv.numel() + 1, dtype=qkv.dtype)[1:].view(qkv.shape)
    elif change == "stride":
        inputs["qkv"] = torch.stack((qkv, qkv), -1)[..., 0]
    else:
        # A single row makes an oversized row stride safe to construct on CPU.
        row_stride = 2**31 if change == "row_stride" else qkv.shape[1] + 1
        inputs["qkv"] = qkv[:1].as_strided((1, qkv.shape[1]), (row_stride, 1))
        inputs["positions"] = inputs["positions"][:1]
        inputs["cache_indices"] = inputs["cache_indices"][:1]
    with pytest.raises((TypeError, ValueError)) as expected:
        op._check_full(**inputs)
    with pytest.raises(type(expected.value)) as actual:
        op._check(**inputs)
    assert str(actual.value) == str(expected.value)


def test_scale_cache_is_bounded_and_weak(monkeypatch):
    import weakref

    op = _cpu_operation(monkeypatch, VARIANTS[0], torch.float8_e4m3fn)
    monkeypatch.setattr(_prologue, "_VALIDATED_LIMIT", 2)
    scales = [torch.ones(1) for _ in range(3)]
    for scale in scales[:2]:
        op._scales(scale, None)
    assert len(op._bound_scales) == 2
    scale = scales[-1]
    op._scales(scale, None)
    assert len(op._bound_scales) == 1
    ref = weakref.ref(scale)
    del scale, scales
    assert ref() is None
    inputs = _cpu_inputs(VARIANTS[0])
    for name in ("k_cache", "v_cache"):
        inputs[name] = inputs[name].to(torch.float8_e4m3fn)
    op._check(**inputs)
    op._check(**inputs)
    refs = [weakref.ref(inputs[name]) for name in ("positions", "cache_indices")]
    assert op._bound_rows[0]() is inputs["positions"]
    assert op._bound_rows[1]() is inputs["cache_indices"]
    del inputs
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_bound_tensor_in_another_role_takes_the_full_check(monkeypatch, cache_dtype):
    """A tensor bound in one role offered in another runs that role's full check and raises
    its original error (validation is cached per role, not per tensor)."""
    op = _cpu_operation(monkeypatch, VARIANTS[0], cache_dtype)
    fresh = _cpu_operation(monkeypatch, VARIANTS[0], cache_dtype)
    inputs = _cpu_inputs(VARIANTS[0])
    for name in ("k_cache", "v_cache"):
        inputs[name] = inputs[name].to(cache_dtype)
    scales = {}
    if cache_dtype != torch.bfloat16:
        scales = {"k_scale": torch.tensor([0.7]), "v_scale": torch.tensor([1.3])}
    op._check(**inputs)
    if scales:
        op._scales(*scales.values())
    calls = _count_full(monkeypatch, op)
    check_scale = op._check_scale
    monkeypatch.setattr(op, "_check_scale", lambda *a: (calls.append(1), check_scale(*a))[1])
    bound = {key: value for key, value in {**inputs, **scales}.items() if key != "qkv"}
    for source, tensor in bound.items():
        for role in bound:
            if role == source:
                continue
            offered = {**inputs, **scales}
            offered[role] = tensor
            k_scale, v_scale = offered.pop("k_scale", None), offered.pop("v_scale", None)
            expected = None
            try:
                fresh._check_full(**offered)
                fresh._scales(k_scale, v_scale)
            except (TypeError, ValueError) as error:
                expected = error
            before = len(calls)
            if expected is None:
                op._check(**offered)
                op._scales(k_scale, v_scale)
            else:
                with pytest.raises(type(expected)) as actual:
                    op._check(**offered)
                    op._scales(k_scale, v_scale)
                assert str(actual.value) == str(expected), (source, role)
            assert len(calls) > before, (source, role)
            op._check(**inputs)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("bad", ["short", "axis"])
def test_position_axes_give_each_frequency_an_axis(monkeypatch, variant, bad):
    frequencies = variant[1] // 2
    axes = (0,) * (frequencies - 1) if bad == "short" else (0,) * (frequencies - 1) + (3,)
    with pytest.raises(ValueError, match="position_axes"):
        _cpu_operation(monkeypatch, variant, position_axes=axes)


@pytest.mark.parametrize("variant", VARIANTS)
def test_multi_axis_positions_shapes(monkeypatch, variant):
    """A multi-axis op takes [T] and contiguous [3, T]; a 1-D op takes [T] only."""
    axes = tuple(f % 3 for f in range(variant[1] // 2))
    op, inputs = _cpu_operation(monkeypatch, variant, position_axes=axes), _cpu_inputs(variant)
    one = inputs["positions"]
    for positions in (one, torch.stack([one, one + 1, one + 2])):
        op._check(**dict(inputs, positions=positions))
    for positions in (torch.stack([one, one]), torch.stack([one] * 3).T.contiguous().T):
        with pytest.raises(ValueError):
            op._check_full(**dict(inputs, positions=positions))
    plain = _cpu_operation(monkeypatch, variant)
    with pytest.raises(ValueError, match="wrong shape"):
        plain._check_full(**dict(inputs, positions=torch.stack([one] * 3)))
