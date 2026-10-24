# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only MoE contracts; no optional GPU packages or NumPy required."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ctikernels.moe import Experts, MoEBlock  # noqa: E402
from ctikernels.moe._configs import TUNED, lookup  # noqa: E402
from ctikernels.moe._routing_check import classify_topk  # noqa: E402
from ctikernels.moe.triton_block import admits as discrete_admits  # noqa: E402
from ctikernels.moe.triton_block_gb10 import admits as integrated_admits  # noqa: E402


@pytest.mark.parametrize("kind,limit", [("rtx5090", 64), ("pro6000", 64), ("gb10", 128)])
def test_block_policy_boundaries(kind, limit):
    op = MoEBlock()
    assert op.decision(1, kind) == "triton_block"
    assert op.decision(limit, kind) == "triton_block"
    assert op.decision(limit + 1, kind) == "composed"
    assert MoEBlock("composed").decision(1, kind) == "composed"
    assert MoEBlock("triton_block").decision(512, kind) == "triton_block"
    assert op.provider == "auto"  # inspection does not mutate the last-call report


def test_unknown_policy_and_invalid_arguments():
    assert MoEBlock().decision(1, "unknown") == "composed"
    assert MoEBlock().decision(1, "cpu") == "composed"
    for factory in (Experts, MoEBlock):
        with pytest.raises(ValueError, match="provider"):
            factory("invalid")
    with pytest.raises(ValueError, match="nonnegative"):
        MoEBlock().decision(-1, "unknown")


@pytest.mark.parametrize("kind", ["rtx5090", "pro6000"])
@pytest.mark.parametrize("experts", [64, 128, 256])
@pytest.mark.parametrize("phase", ["up", "down"])
def test_config_exact_nearest_and_copy(kind, experts, phase):
    table = TUNED[kind][(experts, 512)][phase]
    for key, value in table.items():
        assert lookup(kind, experts, 512, key, phase) == value
    for rows in (0, 3, 5, 100, 511, 100000):
        nearest = min(table, key=lambda key: (abs(key - rows), key))
        assert lookup(kind, experts, 512, rows, phase) == table[nearest]
    copy = lookup(kind, experts, 512, 1, phase)
    copy["BLOCK_SIZE_M"] = -1
    assert table[1]["BLOCK_SIZE_M"] > 0


def test_missing_config_and_distinct_projections():
    assert lookup("gb10", 128, 512, 8, "up") is None
    assert lookup(None, 256, 512, 8, "up") is None
    assert lookup("rtx5090", 256, 128, 8, "up") is None
    assert lookup("rtx5090", 128, 512, 1, "up") != lookup("rtx5090", 128, 512, 1, "down")
    with pytest.raises(ValueError, match="phase"):
        lookup("gb10", 256, 512, 1, "bad")


def _shape(**overrides):
    return dict(
        rows=64,
        hidden=2048,
        experts=256,
        inter=512,
        local_experts=256,
        shared_inter=512,
        dtype=torch.bfloat16,
        **overrides,
    )


@pytest.mark.parametrize("local", [64, 128, 256])
@pytest.mark.parametrize("si", [128, 256, 512])
def test_shared_slices_admitted(local, si):
    shape = _shape()
    shape.update(local_experts=local, shared_inter=si)
    assert discrete_admits(device_class="rtx5090", **shape)
    assert discrete_admits(device_class="pro6000", **shape)
    assert integrated_admits(device_class="gb10", **shape)
    assert not discrete_admits(device_class="gb10", **shape)
    assert not integrated_admits(device_class="pro6000", **shape)


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden", 1024),
        ("experts", 128),
        ("inter", 256),
        ("local_experts", 32),
        ("shared_inter", 64),
        ("dtype", torch.float16),
        ("contiguous", False),
        ("rows", -1),
        ("rows", 1025),
    ],
)
def test_block_admission_rejects_unsupported(key, value):
    shape = _shape()
    shape[key] = value
    assert not discrete_admits(device_class="rtx5090", **shape)
    assert not integrated_admits(device_class="gb10", **shape)


def test_explicit_provider_row_caps():
    shape = _shape()
    for kind, admits, maximum in (
        ("rtx5090", discrete_admits, 256),
        ("pro6000", discrete_admits, 256),
        ("gb10", integrated_admits, 1024),
    ):
        shape["rows"] = maximum
        assert admits(device_class=kind, **shape)
        shape["rows"] += 1
        assert not admits(device_class=kind, **shape)


def _inputs():
    return [
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((2, 32), (16, 32), (16, 16, 32), (16, 32, 8), (16, 32), (32, 8), (1, 32))
    ]


@pytest.mark.parametrize("weight", range(1, 7))
@pytest.mark.parametrize("provider", ["auto", "triton_block", "composed"])
def test_frozen_before_any_cuda(monkeypatch, weight, provider):
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA queried before frozen-weight rejection")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "device", forbidden)
    args = _inputs()
    args[weight].requires_grad_(True)
    with torch.no_grad(), pytest.raises(ValueError, match="frozen"):
        MoEBlock(provider)(*args)


@pytest.mark.parametrize(
    "case,message",
    [
        ("cpu", "CUDA"),
        ("dtype", "BF16"),
        ("x_shape", "expected"),
        ("down_shape", "down weight"),
        ("shared_shape", "shared_gate_up"),
        ("map", "expert_map"),
        ("valid", "valid_tokens"),
        ("out", "out must"),
        ("local", "expert_map is required"),
        ("stride", "contiguous"),
        ("x_grad", "inference only"),
    ],
)
def test_block_validation(case, message):
    args, kwargs = _inputs(), {}
    if case == "dtype":
        args[0] = args[0].float()
    elif case == "x_shape":
        args[0] = args[0].flatten()
    elif case == "down_shape":
        args[3] = args[3][:, :, :4]
    elif case == "shared_shape":
        args[4] = args[4][:8]
    elif case == "map":
        kwargs["expert_map"] = torch.arange(16, dtype=torch.int64)
    elif case == "valid":
        kwargs["valid_tokens"] = torch.ones(2, dtype=torch.int32)
    elif case == "out":
        kwargs["out"] = torch.empty(1)
    elif case == "local":
        args[2], args[3] = args[2][:8], args[3][:8]
    elif case == "stride":
        args[0] = torch.empty(32, 2, dtype=torch.bfloat16).T
    elif case == "x_grad":
        args[0].requires_grad_(True)
    with pytest.raises(ValueError, match=message):
        MoEBlock()(*args, **kwargs)


def test_experts_capacity_and_capture_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=1))
    capture = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)
    op, x = Experts(), torch.empty(1, dtype=torch.bfloat16)
    first = op._intermediates(x, 8, 2, 32, 64)
    smaller = op._intermediates(x, 2, 2, 32, 64)
    assert [a.data_ptr() for a in first] == [a.data_ptr() for a in smaller]
    capture = True
    assert op._intermediates(x, 4, 2, 32, 64)[0].data_ptr() == first[0].data_ptr()
    grown = op._intermediates(x, 16, 2, 32, 64)
    assert grown[0].data_ptr() != first[0].data_ptr()
    assert next(iter(op._buffers.values()))[0].data_ptr() == first[0].data_ptr()
    # Storage a capture used outlives the eager growth that supersedes it.
    pinned = {b.data_ptr() for b in op._graph_storage.values()}
    assert {a.data_ptr() for a in first} <= pinned
    assert grown[0].data_ptr() not in pinned
    capture = False
    grown = op._intermediates(x, 16, 2, 32, 64)
    assert next(iter(op._buffers.values()))[0].data_ptr() == grown[0].data_ptr()
    assert next(iter(op._buffers.values()))[1].dtype == torch.bfloat16  # unweighted FC2 rows
    assert {a.data_ptr() for a in first} <= {b.data_ptr() for b in op._graph_storage.values()}
    other = Experts()._intermediates(x, 16, 2, 32, 64)
    assert other[0].data_ptr() != grown[0].data_ptr()


def test_routing_check_ties_and_mismatches():
    """Exact top-k passes; a swap across a near-tie is a tie; a far swap is a mismatch."""
    router = torch.zeros((16, 4), dtype=torch.bfloat16)
    x = torch.ones((1, 4), dtype=torch.bfloat16)
    scores = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # reference top-4 are experts 0-3
    for expert, score in enumerate(scores):
        router[expert, 0] = score
    router[4, 0] = 5.0 - 2.0**-6  # 5.0 and 4.984 are within the BF16-logit epsilon
    exact = torch.tensor([[3, 1, 0, 2]])
    assert classify_topk(x, router, exact) == (0, 0)
    assert classify_topk(x, router, torch.tensor([[0, 1, 2, 4]])) == (1, 0)
    assert classify_topk(x, router, torch.tensor([[0, 1, 2, 7]])) == (0, 1)
    assert classify_topk(x, router, torch.tensor([[0, 1, 2, 7]]), valid=0) == (0, 0)
    with pytest.raises(AssertionError, match="repeats"):
        classify_topk(x, router, torch.tensor([[0, 1, 1, 2]]))
    with pytest.raises(AssertionError, match="out-of-range"):
        classify_topk(x, router, torch.tensor([[0, 1, 2, -1]]))


def test_qwen3_tables_cover_all_ep_shapes_and_buckets():
    required = {
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        12,
        16,
        24,
        32,
        48,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    }
    kinds = {kind for kind, shapes in TUNED.items() if any(i == 768 for _, i in shapes)}
    if not kinds:
        pytest.skip("Qwen3 I=768 tables have not been installed yet")
    for kind in kinds:
        for local in (128, 64, 32):
            assert (local, 768) in TUNED[kind], (kind, local)
            for phase in ("up", "down"):
                table = TUNED[kind][local, 768][phase]
                assert required <= table.keys(), (kind, local, phase, required - table.keys())
                assert all(table[m] for m in required)
