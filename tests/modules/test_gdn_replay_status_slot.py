# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Device-selected invocation slots, including pitched capacity prefixes."""

import inspect
from dataclasses import fields, replace

import pytest
import torch

from ctikernels.gdn import ReplayPending

from ._replay_case import case, clone, graph_buffers, projected


def bytes_equal(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def all_equal(a, b):
    for f in fields(a):
        bytes_equal(getattr(a, f.name), getattr(b, f.name))


def stacked(op, batch, depth, *, layers=2, pad=0):
    """``pad`` extra BF16 elements per slot pitch (odd pad misaligns slots >= 1)."""
    controls = graph_buffers(op, batch=batch + 2, depth=depth, layers=layers)
    for d in range(depth):
        controls.status(d).sequence.fill_(d + 1)
        op.begin(status=controls.status(d))
    shape = (depth, batch + 2, op.verify_length, op.geometry.conv_dim + 2 * op.geometry.v_heads)
    pitch = shape[1] * shape[2] * shape[3] + pad
    raw = torch.full((depth * pitch,), 0.125, dtype=torch.bfloat16, device=op.device).as_strided(
        shape, (pitch, shape[2] * shape[3], shape[3], 1)
    )[:, :batch]
    pending = controls.slotted_pending(raw, layer=layers - 1)
    return controls, pending


def verify_args(op, state, pending, weights, slots, pos, status):
    b, q = slots.numel(), op.verify_length
    return {
        "projected": projected(op, b, length=q),
        "conv_weight": weights[0],
        "A_log": weights[1],
        "dt_bias": weights[2],
        "norm_weight": weights[3],
        "slot_indices": slots,
        "positions": pos[:, None] + torch.arange(q, device=op.device),
        "state": state,
        "pending": pending,
        "out": torch.full(
            (b, q, op.geometry.value_width), 7, dtype=torch.bfloat16, device=op.device
        ),
        "status": status,
    }


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("q", [8, 16])
def test_slotted_round_is_bitwise_static(depth, batch, q):
    op, state, _, weights, slots, pos, _ = case(batch=batch, q=q)
    reference = clone(state)
    controls, pending = stacked(op, batch, depth)
    other, expected_pending = stacked(op, batch, depth)
    selector = torch.zeros(1, dtype=torch.int32)
    valid = torch.ones(depth, 2 * batch + 2, dtype=torch.bool)[:, : 2 * batch : 2]
    expected_valid = valid.clone()
    arrivals = torch.zeros(depth, batch, dtype=torch.int32)
    for d in range(depth):
        selector.fill_(d)
        untouched = (
            pending.raw.clone(),
            pending.metadata.clone(),
            controls.buffers["records"].clone(),
        )
        for m in (0, q // 2, q):
            kwargs = verify_args(
                op,
                state,
                pending,
                weights,
                slots,
                pos,
                controls.slotted_status(1, batch_size=batch),
            )
            expected = dict(
                kwargs,
                state=reference,
                pending=other.pending(expected_pending.raw[d], slot=d, layer=1),
                status=other.status(d, 1, batch_size=batch),
                out=kwargs["out"].clone(),
            )
            op.verify(**kwargs, status_slot=selector)
            op.verify(**expected)
            bytes_equal(kwargs["out"], expected["out"])
            args = {
                "slot_indices": slots,
                "accepted_lengths": torch.full((batch,), m, dtype=torch.int32),
                "snapshot_rows": slots + state.metadata.shape[0],
                "snapshot_offsets": torch.arange(batch, dtype=torch.int64).clamp(max=m),
            }
            op.commit(
                *weights,
                **args,
                state=state,
                pending=pending,
                status=kwargs["status"],
                status_slot=selector,
                snapshot_valid=valid,
                workspace={"arrivals": arrivals},
            )
            op.commit(
                *weights,
                **args,
                state=reference,
                pending=expected["pending"],
                status=expected["status"],
                snapshot_valid=expected_valid[d],
            )
            pos.add_(m)
            # Reject a fresh verification, not an already committed transaction.
            kwargs["positions"] = pos[:, None] + torch.arange(q)
            expected["positions"] = kwargs["positions"]
            op.verify(**kwargs, status_slot=selector)
            op.verify(**expected)
            op.reject(
                slot_indices=slots,
                state=state,
                pending=pending,
                status=kwargs["status"],
                status_slot=selector,
            )
            op.reject(
                slot_indices=slots,
                state=reference,
                pending=expected["pending"],
                status=expected["status"],
            )
            all_equal(state, reference)
            all_equal(pending, expected_pending)
            bytes_equal(valid, expected_valid)
            assert not arrivals.any()
            for name in ("records", "validity", "sequence", "poison", "pending_metadata"):
                bytes_equal(controls.buffers[name], other.buffers[name])
            assert not controls.buffers["poison"].any()
        for unselected in set(range(depth)) - {d}:
            bytes_equal(pending.raw[unselected], untouched[0][unselected])
            bytes_equal(pending.metadata[unselected], untouched[1][unselected])
            bytes_equal(controls.buffers["records"][:, unselected], untouched[2][:, unselected])


@pytest.mark.parametrize("value", [-1, 2, 2**31 - 1])
@pytest.mark.parametrize("mode", ["verify", "commit", "reject"])
def test_out_of_range_only_poisons(value, mode):
    op, state, _, weights, slots, pos, _ = case(batch=1)
    controls, pending = stacked(op, 1, 2)
    kwargs = verify_args(
        op, state, pending, weights, slots, pos, controls.slotted_status(1, batch_size=1)
    )
    before = clone(state), clone(pending), kwargs["out"].clone()
    records, validity = controls.buffers["records"].clone(), controls.buffers["validity"].clone()
    selector = torch.tensor([value], dtype=torch.int32)
    valid = torch.ones(2, 1, dtype=torch.bool)
    if mode == "verify":
        op.verify(**kwargs, status_slot=selector)
    else:
        args = {
            "slot_indices": slots,
            "state": state,
            "pending": pending,
            "status": kwargs["status"],
            "status_slot": selector,
        }
        if mode == "commit":
            op.commit(
                *weights,
                **args,
                accepted_lengths=torch.zeros(1, dtype=torch.int32),
                snapshot_rows=slots + state.metadata.shape[0],
                snapshot_offsets=torch.zeros(1, dtype=torch.int64),
                snapshot_valid=valid,
            )
        else:
            op.reject(**args)
    all_equal(state, before[0])
    all_equal(pending, before[1])
    bytes_equal(kwargs["out"], before[2])
    bytes_equal(controls.buffers["records"], records)
    bytes_equal(controls.buffers["validity"], validity)
    assert controls.buffers["poison"].item() != 0
    assert valid.all()


@pytest.mark.parametrize(
    "bad",
    ["dtype", "shape", "device", "overlap", "status", "pending", "metadata", "depth", "missing"],
)
def test_selector_host_validation(bad):
    op, state, _, weights, slots, pos, _ = case(batch=1)
    controls, pending = stacked(op, 1, 2)
    kwargs = verify_args(
        op, state, pending, weights, slots, pos, controls.slotted_status(1, batch_size=1)
    )
    selector = torch.zeros(1, dtype=torch.int32)
    if bad == "dtype":
        selector = selector.long()
    elif bad == "shape":
        selector = selector.reshape(())
    elif bad == "device":
        selector = torch.empty(1, dtype=torch.int32, device="meta")
    elif bad == "overlap":
        selector = slots
    elif bad == "status":
        kwargs["status"] = controls.status(0, 1)
    elif bad == "pending":
        kwargs["pending"] = controls.pending(pending.raw[0], slot=0, layer=1)
    elif bad == "metadata":
        kwargs["pending"] = ReplayPending(pending.raw, pending.metadata[0])
    elif bad == "depth":
        kwargs["pending"] = ReplayPending(pending.raw[:1], pending.metadata[:1])
    elif bad == "missing":
        selector = None
    with pytest.raises((TypeError, ValueError)):
        op.verify(**kwargs, status_slot=selector)


@pytest.mark.parametrize("name", ["snapshot_valid", "arrivals"])
def test_commit_requires_stacked_auxiliary_buffers(name):
    op, state, _, weights, slots, _, _ = case(batch=1)
    controls, pending = stacked(op, 1, 2)
    args = {
        "snapshot_rows": slots + state.metadata.shape[0],
        "snapshot_offsets": torch.zeros(1, dtype=torch.int64),
        "snapshot_valid": torch.zeros(2, 1, dtype=torch.bool),
        "workspace": {"arrivals": torch.zeros(2, 1, dtype=torch.int32)},
    }
    if name == "snapshot_valid":
        args[name] = args[name][0]
    else:
        args["workspace"][name] = args["workspace"][name][0]
    with pytest.raises(ValueError):
        op.commit(
            *weights,
            slot_indices=slots,
            accepted_lengths=slots,
            state=state,
            pending=pending,
            status=controls.slotted_status(1, batch_size=1),
            status_slot=slots.clone(),
            **args,
        )


def test_warm_signature_and_trusted_memo_include_selector():
    op, state, _, weights, slots, pos, _ = case(batch=1)
    op._host_trusted = True
    controls, pending = stacked(op, 1, 2)
    kwargs = verify_args(
        op, state, pending, weights, slots, pos, controls.slotted_status(1, batch_size=1)
    )
    op._warm.clear()
    op.verify(**kwargs, status_slot=torch.zeros(1, dtype=torch.int32))
    warm = set(op._warm)
    op.verify(
        **dict(
            kwargs,
            pending=controls.pending(pending.raw[0], slot=0, layer=1),
            status=controls.status(0, 1),
        )
    )
    assert len(op._warm) == 2 and warm < op._warm
    # A cached successful layout must not hide a new alias or mixed form.
    with pytest.raises(ValueError, match="overlapping"):
        op.verify(**kwargs, status_slot=slots)
    with pytest.raises(ValueError, match="status_slot"):
        op.verify(
            **dict(kwargs, status=replace(kwargs["status"], slotted=False)),
            status_slot=slots.clone(),
        )


@pytest.mark.parametrize("checks", ["strict", "trusted"])
@pytest.mark.parametrize("mode", ["verify", "commit", "reject"])
def test_cute_slot_binding_and_cache(checks, mode, monkeypatch):
    from types import SimpleNamespace

    from ctikernels.gdn.replay.cute import Config, _phase_launch

    op, state, _, weights, slots, pos, _ = case(batch=3)
    controls, pending = stacked(op, 3, 3)
    op.checks, op.cute_config = checks, Config()
    status = controls.slotted_status(1, batch_size=3) if checks == "strict" else None
    kwargs = verify_args(op, state, pending, weights, slots, pos, status)
    selector = torch.zeros(1, dtype=torch.int32)
    arrivals = torch.zeros(3, 5, dtype=torch.int32)[:, :3]
    valid = torch.ones(3, 5, dtype=torch.bool)[:, :3]
    kwargs.update(
        status_slot=selector,
        arrivals=arrivals,
        snapshot_valid=valid,
        snapshot_rows=slots.clone(),
        snapshot_offsets=pos.clone(),
        accepted_lengths=slots.clone(),
    )
    for name in ("conv_weight", "A_log", "dt_bias", "norm_weight"):
        kwargs.pop(name)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170, name="test device"),
    )
    compiled, launched = [], []

    def compile_stub(*args, **kw):
        compiled.append(kw)
        return lambda *buffers: launched.append(buffers)

    monkeypatch.setattr(_phase_launch, "compile_phases", compile_stub)
    for d in (0, 1, 2):
        selector.fill_(d)
        _phase_launch.run(op, mode, weights=weights, **kwargs)
    assert len(compiled) == 1
    assert compiled[0]["slot_layout"] == (
        3,
        16 if checks == "strict" else 0,
        1 if checks == "strict" else 0,
        pending.raw.stride(0),
        pending.metadata.stride(0),
        valid.stride(0),
        arrivals.stride(0),
    )
    assert compiled[0]["layout"].valid == 1
    for args in launched:
        assert len(args) == 25 and args[-1] is selector
        for i, tensor in ((13, pending.raw), (14, pending.metadata), (18, valid), (23, arrivals)):
            assert args[i].data_ptr() == tensor.data_ptr()
            assert args[i].numel() >= (tensor.shape[0] - 1) * tensor.stride(0) + tensor[0].numel()


@pytest.mark.parametrize(
    "kernel,mode",
    [
        ("guard", "verify"),
        ("propagate", "commit"),
        ("validate", "verify"),
        ("validate", "commit"),
        ("validate", "reject"),
        ("arithmetic", "verify"),
        ("arithmetic", "commit"),
        ("publish", "verify"),
        ("publish", "commit"),
        ("publish", "reject"),
    ],
)
def test_triton_slot_offsets_compile_as_int64(kernel, mode, tmp_path, monkeypatch):
    """Offline compilation verifies real IR without requiring a CUDA driver."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from ctikernels.gdn.replay._layout import Layout
    from ctikernels.gdn.replay.triton import _kernels, _status

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    fn = getattr(_status if kernel in ("guard", "propagate") else _kernels, kernel)
    from dataclasses import astuple

    constants = {
        "R": 4,
        "S": 4,
        "L": 8,
        "Q": 8,
        "PB": 1,
        "HK": 1,
        "HV": 4,
        "XS": 1032,
        "MODE": mode,
        "CAPTURE": True,
        "INPLACE": False,
        "BLOCK": 1024,
        "LT": astuple(
            Layout(
                4 * 16384,
                3 * 768,
                4 * 16384,
                3 * 768,
                positions=8,
                token=1288,
                out_batch=4096,
                out_token=512,
            )
        ),
        "RINGLESS": False,
        "HAS_SLOT": True,
        "SLOT_DEPTH": 3,
    }
    pointers = {
        "REC": "*u8",
        "SEQ": "*i64",
        "POISON": "*i32",
        "SLOT": "*i32",
        "CODE": "*i32",
        "INDICES": "*i32",
        "POS": "*i64",
        "M": "*i64",
        "E": "*i64",
        "SM": "*i64",
        "PM": "*i64",
        "ACCEPT": "*i32",
        "SR": "*i32",
        "OFF": "*i64",
        "STATUS": "*i32",
        "C": "*fp32",
        "RING": "*bf16",
        "X": "*bf16",
        "PR": "*bf16",
        "CW": "*bf16",
        "AL": "*fp32",
        "BIAS": "*fp32",
        "NW": "*fp32",
        "OUT": "*bf16",
        "CANON": "*fp32",
        "H": "*bf16",
        "VALID": "*i1",
        "HISTORY": "*bf16",
        "PRENORM": "*fp32",
    }
    # Constexprs not set above (BV, CHUNKS) compile at their defaults.
    defaults = inspect.signature(fn.fn).parameters
    constants = {
        name: constants.get(name, defaults[name].default)
        for i, name in enumerate(fn.arg_names)
        if i in fn.constexprs
    }
    signature = {name: pointers.get(name, "i64") for name in fn.arg_names if name not in constants}
    result = triton.compile(
        ASTSource(fn, signature, constexprs=constants),
        target=GPUTarget("cuda", 80, 32),
        options={"num_warps": 8, "enable_fp_fusion": False},
    )
    ir = result.asm["ttir"]
    assert "arith.extsi" in ir and "i32 to i64" in ir
    assert any("arith.muli" in line and ": i64" in line for line in ir.splitlines())
    for name in ("SD_STATUS", "SD_PR", "SD_PM", "SD_VALID", "RD", "SQD"):
        if name in fn.arg_names:
            assert name in fn.do_not_specialize


@pytest.mark.parametrize("field", ["records", "validity", "sequence", "poison"])
def test_malformed_stacked_status(field):
    op, state, _, weights, slots, pos, _ = case(batch=1)
    controls, pending = stacked(op, 1, 2)
    status = controls.slotted_status(1, batch_size=1)
    status = replace(status, **{field: getattr(status, field).reshape(-1)[:1].reshape(())})
    kwargs = verify_args(op, state, pending, weights, slots, pos, status)
    with pytest.raises(ValueError):
        op.verify(**kwargs, status_slot=slots.clone())


@pytest.mark.parametrize("pad", [0, 1, 7, 8])
def test_vector_publication_requires_aligned_slot_pitch(pad):
    from ctikernels.gdn.replay.cute import _phase_launch

    op, *_ = case(batch=1)
    _, pending = stacked(op, 1, 2, pad=pad)
    vec = _phase_launch.vector_publication(op, pending)
    assert pending.raw.data_ptr() % 16 == 0
    if _phase_launch.VECTOR_PUBLISH and op.geometry.v_heads % 4 == 0:
        assert vec == (pad % 8 == 0)
    else:
        assert not vec


def test_snapshot_valid_alias_requires_matching_strides():
    # validity [J=2,D=3,B=1]: same pointer and shape as layer 0's [D,B] view but
    # stride 2 would let slot 1 write invocation 2's validity.
    op, state, _, weights, slots, _, _ = case(batch=1)
    controls = graph_buffers(op, batch=1, depth=3, layers=2)
    for d in range(3):
        controls.status(d).sequence.fill_(d + 1)
        op.begin(status=controls.status(d))
    status = controls.slotted_status(0, batch_size=1)
    assert tuple(status.validity.shape) == (2, 3, 1)
    raw = torch.zeros(
        (3, 1, op.verify_length, op.geometry.conv_dim + 2 * op.geometry.v_heads),
        dtype=torch.bfloat16,
        device=op.device,
    )
    pending = controls.slotted_pending(raw, layer=0)
    good = status.valid
    bad = good.as_strided((3, 1), (2, 1))
    assert bad.data_ptr() == good.data_ptr() and bad.shape == good.shape
    common = {
        "slot_indices": slots,
        "accepted_lengths": torch.zeros_like(slots),
        "state": state,
        "pending": pending,
        "status": status,
        "status_slot": torch.ones(1, dtype=torch.int32, device=op.device),
        "snapshot_rows": slots + state.metadata.shape[0],
        "snapshot_offsets": torch.zeros(1, dtype=torch.int64, device=op.device),
    }
    if op._phase_arrivals("commit"):
        common["workspace"] = {"arrivals": torch.zeros(3, 1, dtype=torch.int32, device=op.device)}
    with pytest.raises(ValueError, match="overlapping"):
        op.commit(*weights, snapshot_valid=bad, **common)
    op.commit(*weights, snapshot_valid=good, **common)
