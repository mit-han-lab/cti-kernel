# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU tests of base dispatch; no Triton or CUDA execution."""

import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from ..support.child import run_child


@pytest.fixture
def implementation(monkeypatch):
    triton = ModuleType("triton")
    triton.jit = lambda function: function
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", ModuleType("triton.language"))
    name = "ctikernels.linear._lora_inference"
    # import_module also replaces the parent's attribute. Restore that reference
    # with sys.modules so later tests patch the same module the harness imports.
    parent = importlib.import_module("ctikernels.linear")
    monkeypatch.setattr(
        parent, "_lora_inference", getattr(parent, "_lora_inference", None), raising=False
    )
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def make_plan(module, dtype):
    from ctikernels.linear._lora_policy import _SMALL_BATCH_POLICY

    factors = (torch.zeros(16, 32, dtype=dtype), torch.zeros(48, 16, dtype=dtype))
    return module.SmallBatchLoRAInferencePlan(
        factors,
        input_size=32,
        output_sizes=(48,),
        scaling=2.0,
        policy=_SMALL_BATCH_POLICY,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "x_strided", "weight_strided", "bias"])
def test_base_selection_and_bias(implementation, monkeypatch, dtype, layout):
    calls = []

    class Linear:
        provider = "cute"

        def __call__(self, x, weight):
            calls.append((x, weight))
            return F.linear(x, weight)

    monkeypatch.setattr(implementation, "Linear", Linear)
    plan = make_plan(implementation, dtype)
    x = torch.randn(3, 32, dtype=dtype)
    weight = torch.randn(48, 32, dtype=dtype)
    if layout == "x_strided":
        x = torch.randn(3, 64, dtype=dtype)[:, ::2]
    if layout == "weight_strided":
        weight = torch.randn(48, 64, dtype=dtype)[:, ::2]
    bias = torch.randn(48, dtype=dtype) if layout == "bias" else None
    assert plan.can_project(x, weight, bias, plan._factors, scaling=2.0)
    expected = F.linear(x, weight, bias)
    actual = plan.base(x, weight, bias)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    use_linear = dtype == torch.bfloat16 and layout == "contiguous"
    assert bool(calls) == use_linear
    assert plan.base_provider == ("cute" if use_linear else "torch")


def test_replaced_factor_storage_rebinds_plan(implementation):
    plan = make_plan(implementation, torch.float16)
    factor = torch.nn.Parameter(plan._factors[0].clone())
    plan._factors = (factor, plan._factors[1])
    plan._bind_factors()
    assert plan._delta_factors[0].data_ptr() == factor.data_ptr()
    assert not plan._delta_factors[0].requires_grad
    pointer = factor.data_ptr()
    with torch.no_grad():
        factor.add_(1.0)  # in-place updates keep the binding
    plan._check_storage()
    assert plan._delta_factors[0].data_ptr() == pointer
    assert plan._a_ptrs.tolist()[0] == pointer
    factor.data = torch.ones_like(factor)
    plan._check_storage()
    assert plan._delta_factors[0].data_ptr() == factor.data_ptr() != pointer
    assert plan._a_ptrs.tolist()[0] == factor.data_ptr()
    assert torch.equal(plan._delta_factors[0], torch.ones_like(factor))


@pytest.mark.parametrize(
    "replacement",
    [
        lambda f: torch.zeros(f.shape[0] + 1, f.shape[1], dtype=f.dtype),
        lambda f: torch.zeros(f.shape[1], f.shape[0], dtype=f.dtype).t(),
        lambda f: torch.zeros(f.shape, dtype=torch.float32),
    ],
)
def test_replaced_factor_storage_with_new_layout_raises(implementation, replacement):
    plan = make_plan(implementation, torch.float16)
    factor = torch.nn.Parameter(plan._factors[0].clone())
    plan._factors = (factor, plan._factors[1])
    plan._bind_factors()
    factor.data = replacement(factor)
    with pytest.raises(ValueError, match="prepare a new inference plan"):
        plan._check_storage()


def test_plan_is_deterministic_by_default(implementation):
    plan = make_plan(implementation, torch.float16)
    assert plan.deterministic
    assert plan._select_provider(3) == "segmented"


def test_capture_miss_is_not_silently_changed_to_torch(implementation, monkeypatch):
    from ctikernels.linear.inference import _capture_miss

    class Linear:
        def __call__(self, x, weight):
            raise _capture_miss(48, 32, 4)

    monkeypatch.setattr(implementation, "Linear", Linear)
    plan = make_plan(implementation, torch.bfloat16)
    with pytest.raises(RuntimeError, match="call the op eagerly"):
        plan.base(
            torch.zeros(3, 32, dtype=torch.bfloat16), torch.zeros(48, 32, dtype=torch.bfloat16)
        )


def test_real_linear_base_on_cpu(implementation):
    plan = make_plan(implementation, torch.bfloat16)
    x, weight = torch.randn(3, 32).bfloat16(), torch.randn(48, 32).bfloat16()
    torch.testing.assert_close(plan.base(x, weight), F.linear(x, weight), atol=0, rtol=0)
    assert plan.base_provider == "cublas"


@pytest.mark.parametrize("tokens,uses_plan", [(3, True), (65, True), (1024, True)])
@torch.inference_mode()
def test_benchmark_production_uses_public_dispatch(implementation, monkeypatch, tokens, uses_plan):
    from benchmark.kernel_tuning.lora import ProjectionCase
    from benchmark.kernel_tuning.lora_sglang import _calls
    from ctikernels.linear import lora

    plan = make_plan(implementation, torch.bfloat16)
    monkeypatch.setattr(lora, "try_prepare_lora_inference_plan", lambda *a, **kw: plan)
    visits = []

    def project(x, weight, bias):
        visits.append("project")
        return plan.base(x, weight, bias)

    monkeypatch.setattr(plan, "project", project)
    x, weight = torch.ones(tokens, 32).bfloat16(), torch.ones(48, 32).bfloat16()
    operands = [(x, weight, None, None, plan._factors)]
    call = _calls("ours_prod", operands, ProjectionCase("single", 32, (48,)), None, {}, None, None)[
        0
    ]
    torch.testing.assert_close(call(), F.linear(x, weight), atol=0, rtol=0)
    assert bool(visits) == uses_plan
    assert (call.plan is plan) == uses_plan
    assert call.base_provider == ("cublas" if uses_plan else "torch")


def test_forced_fbase_ablation_never_calls_linear(implementation, monkeypatch):
    from benchmark.kernel_tuning.lora import ProjectionCase
    from benchmark.kernel_tuning.lora_sglang import _calls

    class Linear:
        def __call__(self, *args):
            pytest.fail("F.linear ablation must not enter the Linear dispatcher")

    monkeypatch.setattr(implementation, "Linear", Linear)
    plan = make_plan(implementation, torch.bfloat16)
    x, weight = torch.ones(3, 32).bfloat16(), torch.ones(48, 32).bfloat16()
    operands = [(x, weight, None, None, plan._factors)]
    call = _calls(
        "ours_plan_fbase", operands, ProjectionCase("single", 32, (48,)), None, {}, None, None
    )[0]
    torch.testing.assert_close(call.base(), F.linear(x, weight), atol=0, rtol=0)
    assert call.base_provider == "torch"


@pytest.mark.parametrize("layout", ["contiguous", "strided", "batched"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_project_uses_segmented_delta_after_base(implementation, monkeypatch, layout, dtype):
    from ctikernels.linear import _lora_segmented
    from ctikernels.linear.lora import lora_projection

    plan = make_plan(implementation, dtype)
    x = torch.randn(65, 32, dtype=dtype)
    if layout == "strided":
        x = torch.randn(65, 64, dtype=dtype)[:, ::2]
    elif layout == "batched":
        x = x.reshape(5, 13, 32)
    weight = torch.randn(48, 32, dtype=dtype)
    bias = torch.randn(48, dtype=dtype)
    expected = F.linear(x, weight, bias)
    seen = []

    def delta(output, delta_x, factors, *, scaling, output_sizes, row_parallel):
        torch.testing.assert_close(output, expected.reshape(-1, 48), rtol=0, atol=0)
        assert delta_x.is_contiguous()
        torch.testing.assert_close(delta_x, x.reshape(-1, 32), rtol=0, atol=0)
        assert all(a is b for a, b in zip(factors, plan._factors, strict=True))
        assert type(scaling) is float and scaling == 2.0
        assert output_sizes == (48,)
        seen.append("segmented")
        output.add_(1)

    monkeypatch.setattr(_lora_segmented, "lora_delta_factors_", delta)
    result = lora_projection(x, weight, bias, plan._factors, scaling=2.0, inference_plan=plan)
    assert seen == ["segmented"]
    torch.testing.assert_close(result, expected + 1, rtol=0, atol=0)
    assert result.shape == expected.shape
    assert plan.provider == "triton-lora-v1"
    assert plan.base_provider == "torch"


@torch.inference_mode()
def test_last_provider_changes_with_table_row(implementation, monkeypatch):
    from ctikernels.linear import _lora_policy, _lora_segmented

    plan = make_plan(implementation, torch.float16)
    plan._policy = _lora_policy.SmallBatchLoRAPolicy("rtx5090", True)
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append("triton")

    monkeypatch.setattr(implementation, "_shrink_kernel", Kernel())
    monkeypatch.setattr(implementation, "_expand_kernel", Kernel())
    monkeypatch.setattr(sys.modules["triton"], "cdiv", lambda a, b: (a + b - 1) // b, raising=False)
    monkeypatch.setattr(
        _lora_segmented, "lora_delta_factors_", lambda *a, **kw: seen.append("segmented")
    )
    monkeypatch.setitem(_lora_policy.PROVIDER_TABLE, ("rtx5090", 1, 16, 3, 32), "triton")
    weight = torch.zeros(48, 32, dtype=torch.float16)
    # Deterministic (default) plans never take the FP-atomic Triton row.
    plan.project(torch.zeros(3, 32, dtype=torch.float16), weight, None)
    assert plan.provider == "triton-lora-v1"
    assert seen == ["segmented"]
    plan._deterministic = False
    plan.project(torch.zeros(3, 32, dtype=torch.float16), weight, None)
    assert plan.provider == "triton-small-batch-v1"
    plan.project(torch.zeros(65, 32, dtype=torch.float16), weight, None)
    assert plan.provider == "triton-lora-v1"
    assert seen == ["segmented", "triton", "triton", "segmented"]


def test_trainable_factors_use_live_detached_aliases(implementation):
    factors = (torch.zeros(16, 32, requires_grad=True), torch.zeros(48, 16, requires_grad=True))
    plan = implementation.SmallBatchLoRAInferencePlan(
        factors, input_size=32, output_sizes=(48,), scaling=2.0
    )
    assert plan._factors is factors
    for original, alias in zip(factors, plan._delta_factors, strict=True):
        assert not alias.requires_grad
        assert original.data_ptr() == alias.data_ptr()
        with torch.no_grad():
            original.add_(1)
        torch.testing.assert_close(alias, torch.ones_like(alias))


def test_delta_capture_miss_propagates(implementation, monkeypatch):
    from ctikernels.linear import _lora_segmented

    def cold_delta(*args, **kwargs):
        raise RuntimeError("new factor metadata during capture; warm it eagerly first")

    monkeypatch.setattr(_lora_segmented, "lora_delta_factors_", cold_delta)
    plan = make_plan(implementation, torch.float16)
    with pytest.raises(RuntimeError, match="warm it eagerly first"):
        plan.project(
            torch.zeros(65, 32, dtype=torch.float16), torch.zeros(48, 32, dtype=torch.float16), None
        )
    assert plan.provider == "uninitialized"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
@pytest.mark.parametrize("contiguous", [True, False])
def test_public_preparation_on_unnamed_device(
    implementation, monkeypatch, dtype, capability, contiguous
):
    from types import SimpleNamespace

    from ctikernels.linear.lora import try_prepare_lora_inference_plan

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "unlisted device")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=99),
    )
    seen = []

    def prepare(factors, **kwargs):
        seen.append(kwargs)
        return "prepared"

    monkeypatch.setattr(implementation, "SmallBatchLoRAInferencePlan", prepare)
    factors = tuple(
        SimpleNamespace(
            device=torch.device("cuda:0"),
            dtype=dtype,
            ndim=2,
            shape=shape,
            is_contiguous=lambda: contiguous,
        )
        for shape in [(512, 13), (7, 512), (512, 13), (3, 512)]
    )
    plan = try_prepare_lora_inference_plan(
        factors, input_size=13, output_sizes=(7, 3), scaling=0.37
    )
    if contiguous:
        assert plan == "prepared"
        assert seen[0]["policy"].device_class == "other"
        assert not seen[0]["policy"].triton_available
    else:
        assert plan is None
        assert not seen


@pytest.mark.parametrize("variant", ["ours_triton", "ours_plan", "ours_plan_fbase"])
def test_benchmark_forces_frozen_triton(implementation, variant):
    from benchmark.kernel_tuning.lora import ProjectionCase
    from benchmark.kernel_tuning.lora_sglang import _calls

    plan = make_plan(implementation, torch.float16)
    x = torch.zeros(3, 32, dtype=torch.float16)
    weight = torch.zeros(48, 32, dtype=torch.float16)
    call = _calls(
        variant,
        [(x, weight, None, None, plan._factors)],
        ProjectionCase("test", 32, (48,)),
        None,
        {},
        None,
        None,
    )[0]
    assert call.plan._select_provider(3) == "triton"


def test_public_cpu_import_does_not_load_cuda_compilers():

    result = run_child(
        [
            sys.executable,
            "-c",
            "import sys; import ctikernels.linear.lora; "
            "assert not any(n.split('.')[0] in ('cutlass', 'triton') for n in sys.modules)",
        ],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
