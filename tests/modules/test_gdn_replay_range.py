# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""RP2 host range, allocation and unmeasured-window routing contracts."""

from types import SimpleNamespace

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay, replay_sizes
from ctikernels.gdn.replay import graph_control_sizes
from ctikernels.gdn.replay.cute import Config, all_immediate, decode_route, effective_config, select
from ctikernels.gdn.replay.cute._config import commit_mode
from ctikernels.gdn.replay.cute._phase_launch import phase_launch
from ctikernels.gdn.replay.torch_reference import raw_row, token

from ._replay_case import case, clone, projected, same, verify


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("q", [1, 33, 64, 100])
@pytest.mark.parametrize("window", [2, 32, 64, 128])
@pytest.mark.parametrize("provider", ["torch_reference", "triton", "cute"])
def test_range_constructor(q, window, provider):
    op = GatedDeltaNetReplay(
        num_k_heads=1,
        num_v_heads=4,
        verify_length=q,
        window=window,
        device="cpu" if provider == "torch_reference" else "cuda:0",
        provider=provider,
        **({"cute_config": Config()} if provider == "cute" else {}),
    )
    assert op.capabilities["verify_lengths"] == (1, None)
    assert op.capabilities["windows"] == "powers of two >= 2"
    assert op.state_spec(num_slots=2, num_snapshot_rows=1)["ring"].shape == (2, window + 3, 776)
    assert op.workspace_spec("verify", batch_size=3)["pending.raw"].shape == (3, q, 776)
    assert graph_control_sizes(batch_size=3, verify_length=q)["verify_positions"].shape == (2, 3, q)


@pytest.mark.parametrize("window", [0, 3, 12, -2, True, 2.0])
def test_range_invalid_window(window):
    with pytest.raises(ValueError, match="power of two"):
        replay_sizes(num_k_heads=1, num_v_heads=4, window=window)


@pytest.mark.parametrize("q", [0, -1, True, 1.0])
def test_range_invalid_q(q):
    with pytest.raises(ValueError, match="integer >= 1"):
        replay_sizes(num_k_heads=1, num_v_heads=4, verify_length=q)
    with pytest.raises(ValueError, match="integer >= 1"):
        graph_control_sizes(batch_size=1, verify_length=q)


def test_range_sizes():
    hk, hv, b, q, window = 16, 48, 3, 64, 32
    width = (2 * hk + hv) * 128 + 2 * hv
    op = GatedDeltaNetReplay(
        num_k_heads=hk, num_v_heads=hv, device="cpu", window=window, verify_length=q
    )
    sizes = replay_sizes(
        num_k_heads=hk, num_v_heads=hv, window=window, verify_length=q, num_slots=5, batch_size=b
    )
    state = op.state_spec(num_slots=5, num_snapshot_rows=1)
    work = op.workspace_spec("verify", batch_size=b)
    assert sizes["ring"] == state["ring"]
    assert sizes["ring"].payload_bytes == 5 * (window + 3) * width * 2
    assert op.bytes_per_slot()["ring"] == (window + 3) * width * 2
    assert (
        sizes["pending.raw"].payload_bytes == work["pending.raw"].payload_bytes == b * q * width * 2
    )
    assert work["out"].payload_bytes == b * q * hv * 128 * 2
    assert (
        graph_control_sizes(batch_size=b, verify_length=q)["verify_positions"].payload_bytes
        == 2 * b * q * 8
    )
    # Host descriptions do not allocate, including offsets beyond signed int32.
    huge = replay_sizes(num_k_heads=hk, num_v_heads=hv, verify_length=2**31, window=2**31)
    assert huge["pending.raw"].payload_bytes == 2**31 * width * 2
    assert huge["ring"].shape == (1, 2**31 + 3, width)


def test_range_layout_span_cross_int32():
    from ctikernels.gdn.replay._layout import physical, span

    stride, batch = 2**30, 3
    assert (torch.tensor(2, dtype=torch.int32) * stride).item() == -(2**31)
    backing = torch.empty(2 * stride + 2, dtype=torch.int32, device="meta")
    for offset in (0, 1):
        control = backing.as_strided((batch,), (stride,), offset)
        assert span(control) == 2**31 + 1
        assert physical(control).shape == (2**31 + 1,)
        assert physical(control).storage_offset() == offset
    width = (2 * 16 + 2 * 32) * 128 + 2 * 32
    x = torch.empty_strided((batch, width), (stride, 1), dtype=torch.bfloat16, device="meta")
    assert span(x) == 2**31 + width
    assert physical(x).shape == (2**31 + width,)


@pytest.mark.parametrize("name", ["NVIDIA GB10", "NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000"])
@pytest.mark.parametrize("window", [32, 64, 128])
def test_range_routes_and_ring(name, window, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: name)
    op = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda:0",
        provider="cute",
        window=window,
        verify_length=64,
    )
    assert all_immediate(name, window=window)
    assert (
        op.bytes_per_slot()["ring"]
        == op.state_spec(num_slots=3, num_snapshot_rows=1)["ring"].payload_bytes
        == 0
    )
    for batch in (1, 3, 8, 16):
        assert select(name, batch, window=window).provider == "product"
        assert decode_route(name, batch, window=window) == op.decode_route(batch) == "product"
        assert effective_config(op, batch).provider == "product"
        assert decode_route(name, batch, mode="delayed", window=window) == "delayed"
        assert commit_mode(name, batch, 64, window) == "eager"
        assert phase_launch(batch, 32, 48, 100, Config())[0] in (32, 64, 128)
    delayed = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        device="cuda:0",
        provider="cute",
        window=window,
        commit_mode="delayed",
    )
    assert delayed.bytes_per_slot()["ring"] > 0
    assert effective_config(delayed, 3).flush_always
    assert decode_route(name, 1, window=8) == ("delayed" if name == "NVIDIA GB10" else "product")


@pytest.mark.parametrize("checks,flush", [("strict", False), ("trusted", False), ("trusted", True)])
def test_range_large_step_uses_chunked_phases(checks, flush, monkeypatch):
    from ctikernels.gdn.replay import cute
    from ctikernels.gdn.replay.cute import _phase_launch

    op, state, _, weights, slots, positions, status = case(window=64)
    op.cute_config = op._step_config = Config(trusted=checks == "trusted", flush_always=flush)
    op.checks = checks
    calls = []
    monkeypatch.setattr(
        _phase_launch, "compile_phases", lambda *a, **kw: calls.append(a) or (lambda *args: None)
    )
    monkeypatch.setattr(
        cute, "compile_step", lambda *a, **kw: pytest.fail("resident Step compiled")
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170, name="test device"),
    )
    cute.run(
        op,
        "step",
        state=state,
        status=status,
        weights=weights,
        projected=projected(op, 2),
        slot_indices=slots,
        positions=positions,
        out=torch.empty(2, 512, dtype=torch.bfloat16),
        arrivals=torch.zeros(2, dtype=torch.int32),
    )
    assert calls[0][10] == "delayed_step"
    assert calls[0][11].flush_always == flush
    assert not calls[0][11].immediate


@pytest.mark.parametrize("q,window", [(1, 2), (33, 32), (64, 64), (100, 128)])
def test_range_reference_verify_commit(q, window):
    op, state, pending, weights, slots, pos, status = case(q=q, window=window, batch=1)
    before = clone(state)
    x, out = verify(op, state, pending, weights, slots, pos, status)
    same(state, before)
    matrix, history = (
        state.checkpoint[0].to(op.reference_dtype).clone(),
        state.conv_history[0].clone(),
    )
    prefix = None
    for t in range(q):
        matrix, history, y = token(
            op.geometry,
            weights,
            matrix,
            history,
            raw_row(op.geometry, x[0, t]),
            x[0, t, op.geometry.conv_dim : op.geometry.conv_dim + op.geometry.value_width],
        )
        assert torch.equal(out[0, t], y.to(torch.bfloat16))
        if t + 1 == max(1, q // 2):
            prefix = (matrix.clone(), history.clone())
    op.commit(
        *weights,
        slot_indices=slots,
        accepted_lengths=torch.tensor([max(1, q // 2)], dtype=torch.int32),
        state=state,
        pending=pending,
        status=status,
    )
    assert torch.equal(state.checkpoint[0], prefix[0].to(state.checkpoint.dtype))
    assert torch.equal(state.conv_history[0], prefix[1])
    assert status.code.item() == 0


def test_range_benchmark_anchor_and_arguments():
    from benchmark.kernel_tuning.gdn_k2_round import parse_args
    from benchmark.kernel_tuning.gdn_verify_compare import fold_period

    args = parse_args(["--q", "1", "64", "100", "--window", "128"])
    assert args.q == [1, 64, 100] and args.window == 128
    assert fold_period("delayed", 1, 32) == 32
    assert fold_period("eager", 1, 32) == 1


class SupportPair:
    """Compare the supplied ABI with exactly equal FP32 weights/native controls."""

    def __init__(
        self,
        variant,
        *,
        device="cpu",
        provider="torch_reference",
        hk=1,
        hv=4,
        batch=3,
        q=4,
        mode="delayed",
        checks="strict",
        trusted_side=False,
    ):
        """``trusted_side``: side 1 is a trusted op called without status (same provider)."""
        from dataclasses import replace

        _, state, pending, weights, slots, _, self.status = case(
            device=device, hk=hk, hv=hv, batch=batch, q=q, window=4
        )
        self.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device=device,
            provider=provider,
            mode=mode if provider == "cute" else "auto",
            checks=checks,
            window=4,
            verify_length=q,
            commit_mode="eager",
        )
        self.trusted_op = None
        if trusted_side:
            self.trusted_op = GatedDeltaNetReplay(
                num_k_heads=hk, num_v_heads=hv, device=device, provider=provider,
                checks="trusted", window=4, verify_length=q, commit_mode="eager",
            )  # fmt: skip
        if self.op._ringless:
            state = replace(state, ring=state.ring[:, :0].clone())
        self.states = [state, clone(state)]
        self.pending = [pending, clone(pending)]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(variant, torch.float32)
        dtypes = (
            (torch.bfloat16, torch.float16, torch.float32) if variant == "mixed" else (dtype,) * 3
        )
        low = (weights[0], *(w.to(t) for w, t in zip(weights[1:], dtypes, strict=True)))
        self.weights = [(low[0], *(w.float() for w in low[1:])), low]
        self.grad = variant == "grad_storage"
        if self.grad:
            for obj in (self.states[1], self.pending[1]):
                for name in ("checkpoint", "conv_history", "ring", "raw"):
                    if hasattr(obj, name):
                        getattr(obj, name).requires_grad_()
        self.integer_variant = variant == "controls"
        self.slots = slots
        self.pos = torch.full((batch,), 7, dtype=torch.int64, device=device)
        self.batch, self.q = batch, q
        self.x = projected(self.op, batch)
        self.vx = projected(self.op, batch, length=q)
        self.outputs = [
            torch.zeros(batch, hv * 128, dtype=torch.bfloat16, device=device) for _ in range(2)
        ]
        self.voutputs = [
            torch.zeros(batch, q, hv * 128, dtype=torch.bfloat16, device=device) for _ in range(2)
        ]
        self.valid = [torch.zeros(batch, dtype=torch.bool, device=device) for _ in range(2)]
        self.rows = torch.arange(batch, dtype=torch.int32, device=device) + state.metadata.shape[0]
        self.accepted = torch.full((batch,), q // 2, dtype=torch.int32, device=device)
        self.offsets = torch.ones(batch, dtype=torch.int64, device=device)
        self.workspaces = {}
        for phase in ("step", "commit", "flush", "export_snapshot"):
            self.workspaces[phase] = [
                {
                    name: torch.full(
                        spec.shape, spec.initial_value or 0, dtype=spec.dtype, device=device
                    )
                    for name, spec in self.op.workspace_spec(phase, batch_size=batch).items()
                    if name in ("arrivals", "controls", "sections")
                }
                for _ in range(2)
            ]
        if self.grad:
            self.x.requires_grad_()
            self.vx.requires_grad_()
            self.outputs[1].requires_grad_()
            self.voutputs[1].requires_grad_()

    def control(self, value, side):
        if side and self.integer_variant:
            return value.to(torch.int64 if value.dtype == torch.int32 else torch.int32)
        return value

    def call(self, phase, side):
        trusted = side and self.trusted_op is not None
        op, i = (self.trusted_op if trusted else self.op), side
        kw = {"state": self.states[i], "status": None if trusted else self.status}
        if phase in ("export_snapshot", "import_snapshot"):
            pairs = torch.stack((self.slots, self.rows), dim=1)
            if phase == "import_snapshot":
                pairs = pairs.flip(1).contiguous()
            kw["pairs"] = self.control(pairs, i)
        else:
            kw["slot_indices"] = self.control(self.slots, i)
        if phase in self.workspaces:
            kw["workspace"] = self.workspaces[phase][i]
        if phase in ("verify", "commit", "reject", "export_snapshot"):
            kw["pending"] = self.pending[i]
        if phase in ("commit", "export_snapshot"):
            kw["accepted_lengths"] = self.control(self.accepted, i)
            kw["snapshot_valid"] = self.valid[i]
        if phase == "commit":
            kw["snapshot_rows"] = self.control(self.rows, i)
            kw["snapshot_offsets"] = self.control(self.offsets, i)
        if phase == "export_snapshot":
            kw["offsets"] = self.control(self.offsets, i)
        if phase == "initialize":
            kw["next_positions"] = self.control(self.pos, i)
        if phase == "handoff":
            kw.update(
                canonical_state=torch.full_like(self.states[i].checkpoint[: self.batch], 0.001),
                history=self.vx[:, :3, : op.geometry.conv_dim].contiguous(),
                next_positions=self.control(self.pos, i),
            )
        args = ()
        if phase in ("step", "verify", "commit", "flush", "export_snapshot"):
            args = self.weights[i]
        if phase in ("step", "verify"):
            verify_phase = phase == "verify"
            x = self.vx if verify_phase else self.x
            args = (x.detach() if self.grad and not i else x, *args)
            positions = (
                self.pos[:, None] + torch.arange(self.q, device=op.device)
                if verify_phase
                else self.pos
            )
            kw.update(
                positions=self.control(positions, i),
                out=(self.voutputs if verify_phase else self.outputs)[i],
            )
        getattr(op, phase)(*args, **kw)

    def compare(self):
        from dataclasses import fields

        # Check every byte, including unused ring/pending storage, within a provider.
        for objects in (self.states, self.pending):
            for f in fields(objects[0]):
                assert torch.equal(
                    getattr(objects[0], f.name).view(torch.uint8),
                    getattr(objects[1], f.name).view(torch.uint8),
                ), f.name
        for tensors in (self.outputs, self.voutputs, self.valid):
            assert torch.equal(tensors[0].view(torch.uint8), tensors[1].view(torch.uint8))
        assert self.status.code.item() == self.status.poison.item() == 0

    def round(self):
        for phase in ("initialize", "handoff"):
            for i in range(2):
                self.call(phase, i)
            self.compare()
        if self.batch > 1:
            self.slots[-1] = -1
        for _ in range(5):
            for i in range(2):
                self.call("step", i)
            self.compare()
            self.pos.add_(1)
        for phase in ("verify", "export_snapshot", "commit", "flush", "import_snapshot"):
            for i in range(2):
                self.call(phase, i)
            self.compare()
        self.pos.add_(self.offsets)
        for phase in ("verify", "reject"):
            for i in range(2):
                self.call(phase, i)
            self.compare()


@pytest.mark.parametrize("variant", ["bf16", "fp16", "mixed", "controls", "grad_storage"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_support_range_reference(variant, context):
    # Create grad-marked leaves outside the disabled-grad context.
    pair = SupportPair(variant)
    with context():
        pair.round()


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("context", [torch.enable_grad, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize(
    "phase", ["step", "bridge_step", "verify", "commit", "flush", "export_snapshot", "begin"]
)
def test_support_range_grad_weights_raise_before_launch(index, context, phase, monkeypatch):
    op, _, _, weights, *_ = case(batch=1)
    weights = list(weights)
    # Parameter construction must precede entering disabled grad mode.
    weights[index] = torch.nn.Parameter(weights[index])
    monkeypatch.setattr(op, "_check_state", lambda *_: pytest.fail("validation reached storage"))
    monkeypatch.setattr(op, "_check_status", lambda *_: pytest.fail("validation reached status"))
    op._backends["torch_reference"] = SimpleNamespace(
        run=lambda *a, **kw: pytest.fail("launch reached")
    )
    monkeypatch.setattr(op, "_lifecycle", lambda *a, **kw: pytest.fail("lifecycle reached"))
    args, kw = weights, {"state": None, "status": None, "slot_indices": None}
    if phase in ("step", "bridge_step", "verify"):
        args = (None, *weights)
        kw.update(positions=None, out=None)
    if phase in ("verify", "commit"):
        kw["pending"] = None
    if phase == "commit":
        kw["accepted_lengths"] = None
    if phase == "export_snapshot":
        del kw["slot_indices"]
        kw.update(pairs=None, snapshot_valid=None)
    if phase == "begin":
        args, kw = (), {"status": None, "layer_weights": (weights,)}
    with context(), pytest.raises(ValueError, match="grad"):
        getattr(op, phase)(*args, **kw)


@pytest.mark.parametrize("target", ["projected", "checkpoint", "conv_history", "ring", "pending"])
def test_support_range_grad_inputs_raise(target):
    op, state, pending, weights, slots, pos, status = case(batch=1)
    x = projected(op, 1, length=op.verify_length)
    value = (
        x
        if target == "projected"
        else pending.raw
        if target == "pending"
        else getattr(state, target)
    )
    value.requires_grad_()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="autograd"):
        op.verify(
            x,
            *weights,
            slot_indices=slots,
            positions=pos[:, None] + torch.arange(op.verify_length),
            state=state,
            pending=pending,
            out=torch.empty(1, op.verify_length, 512, dtype=torch.bfloat16),
            status=status,
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
@pytest.mark.parametrize("index", [1, 2, 3])
def test_support_range_invalid_weight_dtype(dtype, index):
    op, state, _, weights, slots, pos, status = case(batch=1)
    weights = list(weights)
    weights[index] = weights[index].to(dtype)
    with pytest.raises(TypeError, match="FP32, BF16 or FP16"):
        op.step(
            projected(op, 1),
            *weights,
            slot_indices=slots,
            positions=pos,
            state=state,
            out=torch.empty(1, 512, dtype=torch.bfloat16),
            status=status,
        )


@pytest.mark.parametrize(
    "name",
    [
        "slot_indices",
        "pairs",
        "accepted_lengths",
        "snapshot_rows",
        "positions",
        "next_positions",
        "snapshot_offsets",
        "offsets",
    ],
)
def test_support_range_integer_host_validation(name):
    op, *_ = case(batch=1)
    for dtype in (torch.int32, torch.int64):
        value = torch.zeros(1, dtype=dtype)
        op._tensor(name, value, torch.int32, (1,))
    for dtype in (torch.int16, torch.float32, torch.bool):
        with pytest.raises(TypeError, match="int32 or int64"):
            op._tensor(name, torch.zeros(1, dtype=dtype), torch.int32, (1,))
    with pytest.raises(TypeError, match="torch.int64"):
        op._tensor("metadata", torch.zeros(1, dtype=torch.int32), torch.int64, (1,))


@pytest.mark.parametrize("phase", ["step", "verify", "handoff_layers", "ring_empty"])
def test_support_range_cute_dtype_cache(phase, monkeypatch):
    from ctikernels.gdn.replay import cute
    from ctikernels.gdn.replay.cute import _handoff_launch, _phase_launch

    pair = SupportPair("mixed", batch=1)
    op, state = pair.op, pair.states[0]
    op.cute_config = op._step_config = Config(trusted=False)
    op.checks = "strict"
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=170, name="test device"),
    )

    def compile_stub(*args, **kw):
        calls.append(kw.get("tensor_types", args[-1]))
        return lambda *buffers: None

    for module, name in (
        (cute, "compile_step"),
        (_phase_launch, "compile_phases"),
        (_handoff_launch, "compile_handoff"),
        (cute, "compile_ring_empty"),
    ):
        monkeypatch.setattr(module, name, compile_stub)
    layers = op.bind_layers(
        states=(state,),
        table=torch.empty(1, 8, dtype=torch.int64),
        canonical_states=(state.checkpoint[:1],),
        histories=(state.conv_history[:1],),
    )
    flag = torch.empty(1, dtype=torch.bool)

    def run(wide):
        slots = pair.slots.long() if wide else pair.slots
        pos = pair.pos.int() if wide else pair.pos
        if phase == "handoff_layers":
            _handoff_launch.run(op, layers, pos, slots, pair.status)
        elif phase == "ring_empty":
            cute.ring_empty(op, state, slots, flag)
        else:
            cute.run(
                op,
                phase,
                state=state,
                status=pair.status,
                weights=pair.weights[int(wide)],
                slot_indices=slots,
                positions=pos,
                projected=pair.vx if phase == "verify" else pair.x,
                pending=pair.pending[0],
                out=pair.voutputs[0] if phase == "verify" else pair.outputs[0],
                complete=False,
                advance_positions=False,
                arrivals=torch.zeros(1, dtype=torch.int32),
            )

    run(False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm CuTe specialization"):
        run(True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    run(True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    run(False)
    run(True)
    assert len(calls) == 2 and calls[0] != calls[1]
    if phase in ("step", "verify"):
        types = dict(calls[1])
        assert tuple(types[i] for i in (2, 3, 4, 5, 6)) == (
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.int64,
            torch.int32,
        )
    if phase == "handoff_layers":
        assert dict(calls[1])[1] == torch.int32 and dict(calls[1])[2] == torch.int64


@pytest.mark.parametrize("variant", ["mixed", "controls", "grad_storage"])
def test_support_range_product_without_workspace(variant, monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace

    from ctikernels.gdn.replay import cute

    pair = SupportPair(variant, batch=1)
    op = pair.op
    op.cute_config = op._step_config = Config(provider="product", immediate=True)
    op.checks, op._ringless = "trusted", True
    op.provider = dict(op.provider, step="cute")
    pair.states[1] = replace(pair.states[1], ring=pair.states[1].ring[:, :0])
    calls = []
    op._backends["cute"] = SimpleNamespace(run=lambda *a, **kw: pytest.fail("CuTe step fallback"))
    op._product_decode = lambda *a, **kw: calls.append(a)
    stream = SimpleNamespace(
        record_event=lambda: None, wait_event=lambda *a: None, wait_stream=lambda *a: None
    )
    op._side_stream = stream
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *a: nullcontext())
    advances = []
    monkeypatch.setattr(cute, "advance_positions", lambda *a: advances.append(a))
    pair.workspaces["step"][1] = {}
    with torch.no_grad():
        pair.call("step", 1)
    assert len(calls) == len(advances) == 1
    assert all(a is b for a, b in zip(calls[0][1:5], pair.weights[1], strict=True))
    assert calls[0][5].dtype == (torch.int64 if variant == "controls" else torch.int32)


class LayoutPair:
    """Same-provider dense/pitched runs, with guards around every borrowed view."""

    def __init__(
        self,
        *,
        device="cpu",
        provider="torch_reference",
        hk=1,
        hv=4,
        batch=3,
        q=4,
        mode="delayed",
        checks="strict",
        trusted_side=False,
    ):
        """``trusted_side``: side 1 is a trusted op called without status (same provider)."""
        from dataclasses import replace

        base = SupportPair(
            "layout",
            device=device,
            provider=provider,
            hk=hk,
            hv=hv,
            batch=batch,
            q=q,
            mode=mode,
            checks=checks,
        )
        self.__dict__.update(base.__dict__)
        self.guards = []
        self.states[1] = replace(
            self.states[1],
            checkpoint=self.pool(self.states[1].checkpoint, 3),
            conv_history=self.pool(self.states[1].conv_history, 2),
        )
        self.xs = [self.x, self.features(self.x)]
        self.vxs = [self.vx, self.features(self.vx)]
        self.outputs[1] = self.features(self.outputs[1])
        self.voutputs[1] = self.features(self.voutputs[1])
        self.valid[1] = self.control_view(self.valid[1], 3)
        self.controls = {}
        for name, value, stride in (
            ("slots", self.slots, 2),
            ("pos", self.pos, 3),
            ("accepted", self.accepted, 3),
            ("rows", self.rows, 2),
            ("offsets", self.offsets, 3),
        ):
            self.controls[name] = [value, self.control_view(value, stride)]
        positions = self.pos[:, None] + torch.arange(q, device=device)
        pairs = torch.stack((self.slots, self.rows), 1)
        imports = pairs.flip(1)
        for name, value in (("positions", positions), ("pairs", pairs), ("imports", imports)):
            strides = (q + 5, 1) if name == "positions" else (7, 3)
            self.controls[name] = [value, self.borrow(value, strides)]
        canonical = torch.full_like(self.states[0].checkpoint[:batch], 0.001)
        history = self.vx[:, :3, : self.op.geometry.conv_dim].clone()
        self.canonical = [canonical, self.pool(canonical, 2)]
        self.histories = [history, self.pool(history, 3)]

    def pool(self, value, layers):
        # Exercise the relaxed 16-byte base requirement, independently of 256B.
        offset = 16 // value.element_size()
        allocation = torch.full(
            (value.numel() * layers + offset,), -17, dtype=value.dtype, device=value.device
        )
        backing = allocation[offset:].view(value.shape[0], layers, *value.shape[1:])
        self.guards.append((allocation[:offset], -17))
        view = backing[:, 1]
        view.copy_(value)
        self.guards.extend((backing[:, i], -17) for i in range(layers) if i != 1)
        return view

    def borrow(self, value, strides):
        span = 1 + sum((n - 1) * s for n, s in zip(value.shape, strides, strict=True))
        sentinel = True if value.dtype == torch.bool else -17
        backing = torch.full((span + 2,), sentinel, dtype=value.dtype, device=value.device)
        view = backing.as_strided(value.shape, strides, 1)
        view.copy_(value)
        mask = torch.ones(backing.shape, dtype=torch.bool, device=value.device)
        mask.as_strided(value.shape, strides, 1).fill_(False)
        self.guards.append((backing, sentinel, mask))
        return view

    def features(self, value):
        width = value.shape[-1]
        strides = (
            (width + 5, 1)
            if value.ndim == 2
            else ((value.shape[1] + 1) * (width + 5), width + 5, 1)
        )
        return self.borrow(value, strides)

    def control_view(self, value, stride):
        return self.borrow(value, (stride,))

    def update(self, name, value):
        for target in self.controls[name]:
            target.copy_(value)

    def call(self, phase, side):
        trusted = side and self.trusted_op is not None
        op, i = (self.trusted_op if trusted else self.op), side
        kw = {"state": self.states[i], "status": None if trusted else self.status}
        c = lambda name: self.controls[name][i]
        if phase in ("export_snapshot", "import_snapshot"):
            kw["pairs"] = c("pairs" if phase == "export_snapshot" else "imports")
        else:
            kw["slot_indices"] = c("slots")
        if phase in self.workspaces:
            kw["workspace"] = self.workspaces[phase][i]
        if phase in ("verify", "commit", "reject", "export_snapshot"):
            kw["pending"] = self.pending[i]
        if phase in ("commit", "export_snapshot"):
            kw.update(accepted_lengths=c("accepted"), snapshot_valid=self.valid[i])
        if phase == "commit":
            kw.update(snapshot_rows=c("rows"), snapshot_offsets=c("offsets"))
        if phase == "export_snapshot":
            kw["offsets"] = c("offsets")
        if phase in ("handoff", "initialize"):
            kw["next_positions"] = c("pos")
        if phase == "handoff":
            kw.update(canonical_state=self.canonical[i], history=self.histories[i])
        args = (
            self.weights[i]
            if phase in ("step", "verify", "commit", "flush", "export_snapshot")
            else ()
        )
        if phase in ("step", "verify"):
            verifying = phase == "verify"
            args = ((self.vxs if verifying else self.xs)[i], *args)
            kw.update(
                positions=c("positions" if verifying else "pos"),
                out=(self.voutputs if verifying else self.outputs)[i],
            )
        getattr(op, phase)(*args, **kw)

    def compare(self):
        SupportPair.compare(self)
        for guard in self.guards:
            value, sentinel = guard[:2]
            if len(guard) == 3:
                value = value[guard[2]]
            assert torch.all(value == sentinel), "borrowed view touched padding/neighbour"

    def both(self, phase):
        for side in range(2):
            self.call(phase, side)
        self.compare()

    def round(self):
        self.both("initialize")
        self.both("handoff")
        for _ in range(5):
            self.both("step")
            self.update("pos", self.controls["pos"][0] + 1)
        self.update(
            "positions",
            self.controls["pos"][0][:, None] + torch.arange(self.q, device=self.op.device),
        )
        for phase in ("verify", "export_snapshot", "commit", "flush", "import_snapshot"):
            self.both(phase)
        self.update("pos", self.controls["pos"][0] + self.offsets)
        self.update(
            "positions",
            self.controls["pos"][0][:, None] + torch.arange(self.q, device=self.op.device),
        )
        self.both("verify")
        self.both("reject")


@pytest.mark.parametrize("batch,q", [(1, 4), (3, 8)])
def test_layout_reference_bitwise(batch, q):
    LayoutPair(batch=batch, q=q).round()


def test_layout_exact_alias_and_layer_binding():
    from dataclasses import replace

    from ctikernels.gdn.replay._layout import overlaps, physical, span

    from ._replay_case import graph_buffers

    pair = LayoutPair(batch=3)
    op, template = pair.op, pair.states[0]
    cp = torch.zeros(template.checkpoint.shape[0], 3, *template.checkpoint.shape[1:])
    hist = torch.zeros(
        template.conv_history.shape[0], 3, *template.conv_history.shape[1:], dtype=torch.bfloat16
    )
    states = [
        replace(clone(template), checkpoint=cp[:, j], conv_history=hist[:, j]) for j in range(3)
    ]
    assert not overlaps(cp[:, 0], cp[:, 1])
    assert overlaps(cp[:, 0], cp[1:, 0])
    assert physical(cp[:, 1]).data_ptr() == cp[:, 1].data_ptr()
    assert physical(cp[:, 1]).stride() == (1,)
    assert span(cp[:, 1]) > cp[:, 1].numel()
    table = torch.empty(3, 8, dtype=torch.int64)
    binding = op.bind_layers(
        states=states,
        table=table,
        canonical_states=tuple(pair.canonical[1] + i for i in range(3)),
        histories=tuple(pair.histories[1] + i for i in range(3)),
    )
    status = graph_buffers(op, batch=3, layers=3).status(0)
    status.sequence.fill_(1)
    op.begin(status=status)
    op.handoff_layers(
        binding,
        next_positions=pair.controls["pos"][1],
        slot_indices=pair.controls["slots"][1],
        status=status,
    )
    for i, state in enumerate(states):
        assert torch.equal(state.checkpoint[:3], pair.canonical[1] + i)
        assert torch.equal(state.conv_history[:3], pair.histories[1] + i)
    with pytest.raises(ValueError, match="overlapping layer"):
        op.bind_layers(states=(states[0], states[0]), table=torch.empty(2, 8, dtype=torch.int64))
    controls = torch.arange(30, dtype=torch.int32)
    assert not overlaps(controls[::3], controls[1::3])
    assert overlaps(controls[::3], controls[3::3])
    assert overlaps(controls[:1].expand(7), controls[:1])


@pytest.mark.parametrize(
    "bad",
    [
        "checkpoint_inner",
        "history_inner",
        "pool_pitch",
        "output_rows",
        "tokens",
        "positions_q",
        "internal_ring",
    ],
)
def test_layout_rejects_outside_slice(bad):
    from dataclasses import replace

    pair = LayoutPair()
    op, state = pair.op, pair.states[1]
    if bad in ("checkpoint_inner", "history_inner", "internal_ring"):
        name = {
            "checkpoint_inner": "checkpoint",
            "history_inner": "conv_history",
            "internal_ring": "ring",
        }[bad]
        value = getattr(state, name)
        if bad == "checkpoint_inner":
            value = value.transpose(-1, -2)
        else:
            backing = torch.zeros(*value.shape, 2, dtype=value.dtype)
            value = backing[..., 0]
        with pytest.raises(ValueError):
            op._check_state(replace(state, **{name: value}))
    elif bad == "pool_pitch":
        value = state.conv_history
        pitch = value[0].numel() + 1
        backing = torch.zeros(value.shape[0] * pitch, dtype=value.dtype)
        view = backing.as_strided(value.shape, (pitch, value.shape[-1], 1))
        with pytest.raises(ValueError, match="16-byte slot pitch"):
            op._check_state(replace(state, conv_history=view))
    else:
        kw = {
            "slot_indices": pair.slots,
            "positions": pair.controls["positions"][0],
            "state": state,
            "pending": pair.pending[1],
            "out": pair.voutputs[1],
            "status": pair.status,
        }
        x = pair.vxs[1]
        if bad == "output_rows":
            kw["out"] = kw["out"].as_strided(kw["out"].shape, (0, 0, 1))
        elif bad == "tokens":
            x = x.as_strided(x.shape, (x.stride(0), x.shape[-1] - 1, 1))
        else:
            kw["positions"] = torch.empty(3, pair.q * 2, dtype=torch.int64)[:, ::2]
        with pytest.raises(ValueError):
            op.verify(x, *pair.weights[0], **kw)


@pytest.mark.parametrize("phase", ["step", "verify"])
def test_layout_cute_cache_and_physical_spans(phase, monkeypatch):
    from ctikernels.gdn.replay import cute
    from ctikernels.gdn.replay._layout import span
    from ctikernels.gdn.replay.cute import _phase_launch

    pair = LayoutPair()
    op = pair.op
    op.cute_config = op._step_config = Config(trusted=False)
    op.provider = dict(op.provider, **{phase: "cute"})
    op._backends["cute"] = cute
    pair.workspaces["step"] = [{"arrivals": torch.zeros(1, dtype=torch.int32)} for _ in range(2)]
    compiled, launches = [], []

    def compile_stub(*args, **kw):
        compiled.append(kw["layout"])
        return lambda *buffers: launches.append(buffers)

    monkeypatch.setattr(cute, "compile_step", compile_stub)
    monkeypatch.setattr(_phase_launch, "compile_phases", compile_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="test device", multi_processor_count=170),
    )
    pair.call(phase, 0)
    pair.call(phase, 1)
    assert len(compiled) == 2
    dense, pitched = compiled
    assert dense.x_alignment == 16 and pitched.x_alignment == 2
    assert pitched.checkpoint == pair.states[1].checkpoint.stride(0)
    assert pitched.history == pair.states[1].conv_history.stride(0)
    assert pitched.indices == 2
    assert pitched.positions == (pair.q + 5 if phase == "verify" else 3)
    buffers = launches[-1]
    x = (pair.vxs if phase == "verify" else pair.xs)[1]
    out = (pair.voutputs if phase == "verify" else pair.outputs)[1]
    for index, tensor in (
        (0, x),
        (5, pair.controls["slots"][1]),
        (7, pair.states[1].checkpoint),
        (8, pair.states[1].conv_history),
        (19 if phase == "verify" else 16, out),
    ):
        assert buffers[index].data_ptr() == tensor.data_ptr()
        assert buffers[index].shape == (span(tensor),)
        assert buffers[index].stride() == (1,)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    pair.call(phase, 0)
    pair.call(phase, 1)
    assert len(compiled) == 2
    # A new alignment class cannot accidentally reuse a warmed specialization.
    values = pair.vxs if phase == "verify" else pair.xs
    backing = torch.empty(values[1].numel() + 2, dtype=torch.bfloat16)
    values[1] = backing[2:].view(values[1].shape)
    with pytest.raises(RuntimeError, match="warm CuTe specialization"):
        pair.call(phase, 1)


@pytest.mark.parametrize(
    "field", ["checkpoint", "conv_history", "slots", "out", "out_base", "projected"]
)
def test_layout_product_host_dispatch(field):
    from dataclasses import replace

    pair = LayoutPair(batch=3)
    op = pair.op
    op.cute_config = op._step_config = Config(provider="product", immediate=True)
    op.checks, op._ringless = "trusted", True
    op.provider = dict(op.provider, step="cute")
    pair.states[0] = replace(pair.states[0], ring=pair.states[0].ring[:, :0])
    if field in ("checkpoint", "conv_history"):
        pair.states[0] = replace(pair.states[0], **{field: getattr(pair.states[1], field)})
    elif field == "slots":
        pair.controls["slots"][0] = pair.controls["slots"][1]
    elif field == "out":
        pair.outputs[0] = pair.outputs[1]
    elif field == "out_base":
        backing = torch.zeros(pair.outputs[0].numel() + 1, dtype=torch.bfloat16)
        pair.outputs[0] = backing[1:].view_as(pair.outputs[0])
    else:
        pair.xs[0] = pair.xs[1]
    pair.workspaces["step"][0] = {}
    calls = []
    op._backends["cute"] = SimpleNamespace(run=lambda *args, **kw: calls.append(kw))
    pair.call("step", 0)
    assert calls[0]["product_fallback"] and calls[0]["arrivals"] is None
    assert op._product_decode is None


def test_layout_broadcast_controls_and_minimum_pool_pitch():
    from dataclasses import replace

    pair = LayoutPair()
    # Read-only positions/counts may repeat one value with stride zero.
    for name in ("pos", "accepted", "offsets"):
        value = pair.controls[name][0]
        pair.controls[name][1] = value[:1].expand_as(value)
    pair.both("initialize")
    pair.both("handoff")
    pair.both("verify")
    pair.both("commit")
    state = pair.states[1]
    for name in ("checkpoint", "conv_history"):
        value = getattr(state, name)
        row = value[0].numel()
        pitch = row + 16 // value.element_size()
        backing = torch.full((value.shape[0] * pitch,), -17, dtype=value.dtype)
        view = backing.as_strided(value.shape, (pitch, *value.stride()[1:]))
        view.copy_(value)
        state = replace(state, **{name: view})
        pair.guards.append((backing.view(value.shape[0], pitch)[:, row:], -17))
    pair.states[1] = state
    pair.controls["pos"] = [p + pair.accepted for p in pair.controls["pos"]]
    pair.both("step")
