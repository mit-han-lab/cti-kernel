# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Qualification plumbing; all allocations here belong to the harness, not the op."""

import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import torch

from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetReplay

ROOT = Path(__file__).resolve().parents[2]
PIN = "a2fcb65a09742dc2adbe6a39a951d7cf178e32f4"
REFERENCE_SHA = "ac912500d3d90c9a19841e7fe1c544f10e26bf6055bf5f63d671818c6ae26dc3"
# Product arithmetic is pinned at PIN. These RC adapter hashes additionally include
# borrowed out and RP3's decode-local dtype range (BF16/FP16 scalar weights widened
# to FP32 on load, int64 slots, dtype-keyed plans); FP32-weight arithmetic is unchanged.
# They identify sources, not new GPU qualification receipts.
PRODUCT_HASHES = {
    "python/ctikernels/gdn/_triton_decode.py": "825d9ffd1808f9996e7692d393d57d95c5041d59ed7af441fa6f83a1f26f6b56",
    "python/ctikernels/gdn/_triton_decode_kernels.py": "00138b2d4e905be34e9e81c4be0de897c90e186f2b299410be9d6f3555048576",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(row):
    def safe(x):
        if isinstance(x, float) and not math.isfinite(x):
            return "infinity" if x > 0 else "-infinity" if x < 0 else "nan"
        if isinstance(x, dict):
            return {k: safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        if isinstance(x, torch.dtype):
            return str(x)
        return x

    print(json.dumps(safe(row), allow_nan=False), flush=True)


def manifest():
    paths = [
        "tests/gpu/_gdn_reference.py",
        "python/ctikernels/gdn/_triton_decode.py",
        "python/ctikernels/gdn/_triton_decode_kernels.py",
    ]
    paths += [
        str(p.relative_to(ROOT)) for p in (ROOT / "python/ctikernels/gdn/replay").rglob("*.py")
    ]
    paths += [
        str(p.relative_to(ROOT)) for p in (ROOT / "benchmark/kernel_tuning").glob("gdn_replay*.py")
    ]
    hashes = {p: digest(ROOT / p) for p in sorted(paths)}
    baseline_match = hashes[paths[0]] == REFERENCE_SHA
    baseline_match &= all(hashes[p] == expected for p, expected in PRODUCT_HASHES.items())
    versions = {}
    for package in ("torch", "triton", "nvidia-cutlass-dsl", "cuda-bindings"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    driver = None
    if props is not None:
        try:
            from cuda.bindings.driver import cuDriverGetVersion

            status, value = cuDriverGetVersion()
            driver = int(value) if int(status) == 0 else None
        except ImportError:
            driver = None
    return {
        "source_hashes": hashes,
        "product_pin": PIN,
        "baseline_pin_matches": bool(baseline_match),
        "versions": versions,
        "cuda": torch.version.cuda,
        "driver": driver,
        "device": None if props is None else props.name,
        "gpu_id": None if props is None else str(getattr(props, "uuid", "unavailable")),
        "precision": "default",
        "derived_cache": False,
        "output_checkpoint": False,
    }


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise BudgetExpired("time budget reached; remaining matrix is unqualified")


def allocate(specs, device):
    return {
        name: torch.full(s.shape, s.initial_value or 0, dtype=s.dtype, device=device)
        for name, s in specs.items()
    }


class Replay:
    def __init__(
        self,
        provider,
        hk,
        hv,
        b,
        window,
        q,
        weights,
        initial,
        history,
        positions,
        *,
        num_slots=None,
        cute_config=None,
        checks="trusted",
        mode="auto",
    ):
        self.device = torch.device("cpu" if provider == "torch_reference" else "cuda")
        self.op = GatedDeltaNetReplay(
            num_k_heads=hk,
            num_v_heads=hv,
            device=self.device,
            window=window,
            verify_length=q,
            provider=provider,
            reference_dtype=torch.float32,
            cute_config=cute_config,
            checks=checks,
            mode=mode,
        )
        self.weights = tuple(w.to(self.device) for w in weights)
        self.b, self.hv = b, hv
        self.r = num_slots or b
        self.state = self.op.create_state(
            buffers=allocate(self.op.state_spec(num_slots=self.r, num_snapshot_rows=b), self.device)
        )
        self.controls = self.op.bind_graph_buffers(
            buffers=allocate(self.op.graph_buffer_spec(batch_size=b), self.device)
        )
        self.status = self.controls.status(0)
        self.status.sequence.fill_(1)
        self.op.begin(status=self.status, layer_weights=(self.weights,))
        self.slots = torch.arange(b, dtype=torch.int32, device=self.device)
        self.pos = positions.to(self.device).clone()
        self.expected_cursor = torch.zeros(b, dtype=torch.int64, device=self.device)
        self.expected_n = torch.zeros_like(self.expected_cursor)
        self.out = torch.empty(b, hv * 128, dtype=torch.bfloat16, device=self.device)
        ws = allocate(self.op.workspace_spec("step", batch_size=b), self.device)
        self.workspace = {k: v for k, v in ws.items() if k != "out"}
        ps = allocate(self.op.workspace_spec("verify", batch_size=b), self.device)
        self.pending = self.op.create_pending(
            raw=ps["pending.raw"], metadata=ps["pending.metadata"]
        )
        self.verify_out = ps["out"]
        self.phase_workspace = allocate(
            {
                k: v
                for k, v in self.op.workspace_spec("commit", batch_size=b).items()
                if k == "arrivals"
            },
            self.device,
        )
        self.pairs = torch.stack((self.slots, self.slots + self.r), -1).contiguous()
        self.valid = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.op.handoff(
            canonical_state=initial.to(self.device),
            history=history.to(self.device),
            next_positions=self.pos,
            slot_indices=self.slots,
            state=self.state,
            status=self.status,
        )

    def step(self, x):
        self.op.step(
            x.to(self.device),
            *self.weights,
            slot_indices=self.slots,
            positions=self.pos,
            state=self.state,
            status=self.status,
            out=self.out,
            workspace=self.workspace,
        )
        active = (self.slots >= 0).to(torch.int64)
        from ctikernels.gdn.replay.cute import effective_config

        config = effective_config(self.op, self.b) if self.op.cute_config is not None else None
        if config is None or not config.immediate:
            self.expected_cursor = (self.expected_cursor + active) % (self.op.window + 3)
            if config is not None and config.flush_always:
                self.expected_n.zero_()
            else:
                self.expected_n = (self.expected_n + active) % self.op.window
        self.pos.add_(1)
        return self.out

    def verify(self, x):
        q = self.op.verify_length
        # A singleton-B transpose can be "contiguous" yet retain stride(0)=P,
        # smaller than Q*P. Re-view through a flat row to canonicalize that stride.
        x = x.to(self.device).contiguous().view(self.b, -1).view(self.b, q, -1)
        self.op.verify(
            x,
            *self.weights,
            slot_indices=self.slots,
            positions=self.pos[:, None] + torch.arange(q, device=self.device),
            state=self.state,
            pending=self.pending,
            out=self.verify_out,
            status=self.status,
        )
        return self.verify_out

    def resolve(self, accepted, reject=False, offsets=None):
        if reject:
            self.op.reject(
                slot_indices=self.slots, state=self.state, pending=self.pending, status=self.status
            )
        else:
            accepted = accepted.to(self.device)
            controls = {}
            branch = None
            if offsets is not None:
                offsets = offsets.to(self.device)
                branch = type(self.state)(
                    **{f.name: getattr(self.state, f.name).clone() for f in fields(self.state)}
                )
                pending = type(self.pending)(
                    self.pending.raw.clone(), self.pending.metadata.clone()
                )
                self.op.commit(
                    *self.weights,
                    slot_indices=self.slots,
                    accepted_lengths=offsets.to(torch.int32),
                    state=branch,
                    pending=pending,
                    status=self.status,
                    workspace=self.phase_workspace,
                )
                controls = {
                    "snapshot_rows": torch.arange(self.b, dtype=torch.int32, device=self.device)
                    + self.r,
                    "snapshot_offsets": offsets,
                    "snapshot_valid": self.valid,
                }
            self.op.commit(
                *self.weights,
                slot_indices=self.slots,
                accepted_lengths=accepted,
                state=self.state,
                pending=self.pending,
                status=self.status,
                workspace=self.phase_workspace,
                **controls,
            )
            if branch is not None:
                if not bool(self.valid.all()):
                    raise RuntimeError("accepted-offset valid flag not set")
                if not torch.equal(
                    self.state.checkpoint[self.r :], branch.checkpoint[self.slots.long()]
                ):
                    raise RuntimeError("accepted-offset state differs from commit(k)")
                if not torch.equal(
                    self.state.conv_history[self.r :], branch.conv_history[self.slots.long()]
                ):
                    raise RuntimeError("accepted-offset history differs from commit(k)")
            self.pos.add_(accepted)
            self.expected_n.zero_()
            self.expected_cursor.zero_()

    def materialize(self):
        self.pairs[:, 0].copy_(self.slots)
        self.op.export_snapshot(
            *self.weights,
            pairs=self.pairs,
            state=self.state,
            status=self.status,
            workspace=self.phase_workspace,
            snapshot_valid=self.valid,
        )
        return self.state.checkpoint[self.r :]

    def flush(self):
        self.expected_n.zero_()
        self.op.flush(
            *self.weights,
            slot_indices=self.slots,
            state=self.state,
            status=self.status,
            workspace=self.phase_workspace,
        )

    def check(self):
        if self.status.code.cpu().item() or self.status.poison.cpu().item():
            raise RuntimeError("replay status/poison failed")


class Product:
    def __init__(self, hk, hv, weights, initial, history):
        self.op = GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda")
        if "triton" not in self.op.provider.lower():
            raise RuntimeError("product did not select frozen Triton")
        self.weights = weights
        self.state, self.history = initial.clone(), history.clone()
        self.slots = torch.arange(initial.shape[0], dtype=torch.int32, device="cuda")

    def step(self, x):
        return self.op(x, *self.weights, self.slots, self.state, self.history)

    def launch_config(self):
        return [
            {
                # (batch, row stride, packed, pointer dtypes)
                "key": [v if isinstance(v, (int, bool)) else str(v) for v in key],
                "grid": list(plan.grid),
                "arguments": {
                    k: v if isinstance(v, (int, float, bool, str)) else str(v)
                    for k, v in plan.arguments.items()
                },
            }
            for key, plan in self.op._kernel._plans.items()
        ]

    def verify(self, x, accepted):
        saved, saved_h = self.state.clone(), self.history.clone()
        committed, committed_h = saved.clone(), saved_h.clone()
        ys = []
        for t in range(x.shape[1]):
            ys.append(self.step(x[:, t].contiguous()))
            mask = accepted > t
            indices = self.slots.long()
            committed.index_copy_(
                0,
                indices,
                torch.where(
                    mask[:, None, None, None],
                    self.state.index_select(0, indices),
                    committed.index_select(0, indices),
                ),
            )
            committed_h.index_copy_(
                0,
                indices,
                torch.where(
                    mask[:, None, None],
                    self.history.index_select(0, indices),
                    committed_h.index_select(0, indices),
                ),
            )
        self.state.copy_(committed)
        self.history.copy_(committed_h)
        return torch.stack(ys, 1)


class Oracle:
    """Batched FP64 transcription of the pinned _Sequence, exact supplied inputs.

    State uses [B,Hv,K,V], matching the pinned accumulation orientation. No
    reduced-precision intermediates. Oracle work is never in timed serving rows.
    ``graph=True`` uses reusable capture buffers; on CPU the same buffer plumbing
    executes eagerly so warm-up, replacement and verify semantics can be tested.
    """

    def __init__(self, hk, weights, state, history, *, graph=False):
        self.hk = hk
        self.weights = tuple(w.double() for w in weights)
        self.state = state.double().transpose(-1, -2).contiguous()
        self.history = history.clone()
        self.graph = graph
        self._captured = None

    def step(self, x):
        if not self.graph:
            return self._step(x)
        if self._captured is None:
            # Capture only the independent FP64 oracle. All rows/heads remain
            # batched; product replicas retain their original launch geometry.
            # canonical() is a view of FP64 state. Double + transpose + contiguous
            # in Oracle.__init__ can return that same storage. Warm-up must never
            # mutate the live trajectory before it is copied into capture buffers.
            work = Oracle(self.hk, self.weights, self.canonical().clone(), self.history)
            gx, gs, gh = x.clone(), work.state, work.history

            def body():
                work.state, work.history = gs, gh
                y = work._step(gx)
                gs.copy_(work.state)
                gh.copy_(work.history)
                return y

            if x.is_cuda:
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    body()
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    gy = body()
                replay = graph.replay
            else:
                # Exercise the same warm-up and static-buffer lifecycle without CUDA.
                body()
                gy = body()

                def replay():
                    gy.copy_(body())

            self._captured = replay, gx, gs, gh, gy
        replay, gx, gs, gh, gy = self._captured
        gx.copy_(x)
        gs.copy_(self.state)
        gh.copy_(self.history)
        replay()
        self.state, self.history = gs, gh
        # verify retains every hypothetical output until the whole block ends.
        return gy.clone()

    def _step(self, x):
        cw, al, bias, nw = self.weights
        hv, dc = al.numel(), cw.shape[0]
        window = torch.cat((self.history, x[:, None, :dc]), 1).double()
        c = (window.transpose(-1, -2) * cw).sum(-1)
        c = c * torch.sigmoid(c)
        q, k, v = c.split((self.hk * 128, self.hk * 128, hv * 128), -1)
        q, k = q.reshape(-1, self.hk, 128), k.reshape(-1, self.hk, 128)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128**-0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q, k = (a.repeat_interleave(hv // self.hk, 1) for a in (q, k))
        z, b, a = x[:, dc:].double().split((hv * 128, hv, hv), -1)
        alpha = torch.exp(-al.exp() * torch.logaddexp(a + bias, torch.zeros_like(a)))
        s = alpha[..., None, None] * self.state
        delta = torch.sigmoid(b)[..., None] * (
            v.reshape(-1, hv, 128) - torch.einsum("bhkv,bhk->bhv", s, k)
        )
        self.state = s + k[..., None] * delta[..., None, :]
        o = torch.einsum("bhkv,bhk->bhv", self.state, q)
        y = nw * o * torch.rsqrt((o * o).mean(-1, keepdim=True) + 1e-6)
        z = z.reshape(-1, hv, 128)
        y *= z * torch.sigmoid(z)
        self.history = window[:, 1:].to(x.dtype)
        return y

    def canonical(self):
        return self.state.transpose(-1, -2)

    def verify(self, x, accepted):
        committed, history = self.state.clone(), self.history.clone()
        ys = []
        for t in range(x.shape[1]):
            ys.append(self.step(x[:, t]))
            mask = accepted > t
            committed.copy_(torch.where(mask[:, None, None, None], self.state, committed))
            history.copy_(torch.where(mask[:, None, None], self.history, history))
        self.state, self.history = committed, history
        return torch.stack(ys, 1)
