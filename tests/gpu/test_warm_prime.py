# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""After an op's warm_up, its first call at any warmed row count does not wait for queued
device work: CUDA loads a kernel at its first launch, which drains the device, so warm-ups
launch every kernel they compile once (``_runtime.launch``)."""

import pytest
import torch

ROWS = (1, 3, 8, 17, 64, 100, 257, 512)


def _queue_work():
    busy = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    for _ in range(64):
        busy.matmul(busy)
    queued = torch.cuda.Event()
    queued.record()
    return queued


def _norm():
    from ctikernels.norm import inference as norm

    norm.warm_up_add_rms_norm(512, hidden=4096, dtype=torch.bfloat16, device="cuda",
                              zero_centered=True)  # fmt: skip
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
    residual, weight = torch.randn_like(x), torch.zeros(4096, dtype=x.dtype, device="cuda")
    return lambda m: norm.add_rms_norm_(x[:m], residual[:m], weight, 1e-6, zero_centered=True)


def _activation():
    from ctikernels.activation import inference as activation

    activation.warm_up_silu_and_mul(512, width=12288, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(512, 2 * 12288, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(512, 12288, dtype=x.dtype, device="cuda")
    return lambda m: activation.silu_and_mul(x[:m], out[:m])


def _linear():
    from ctikernels.linear.inference import Linear

    op = Linear()
    weight = (torch.randn(4096, 4096, device="cuda") * 0.02).bfloat16()
    op.warm_up(weight, 512)
    x = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
    return lambda m: op(x[:m], weight)


def _gdn_prefill():
    from ctikernels.gdn.inference import GatedDeltaNetPrefill

    from . import test_gdn as gdn

    op = GatedDeltaNetPrefill(num_k_heads=16, num_v_heads=32, device="cuda")
    op.warm_up(512, max_sequences=2)
    weights = gdn._weights(16, 32, seed=6)
    projected = gdn._projected(512, 16, 32, seed=7)
    state, conv_state = gdn._pools(16, 32, seed=8)
    cases = {m: gdn._prefill_case([m], [3], [True]) for m in ROWS}

    def call(m):
        with torch.inference_mode():
            op(projected[:m], *weights.args(), *cases[m], state, conv_state)

    return call


def _moe():
    from ctikernels.moe.inference import MoEBlock

    experts, hidden, inter = 256, 2048, 512

    def weight(*shape):
        return (torch.randn(*shape, device="cuda") * 0.02).bfloat16()

    weights = (weight(experts, hidden), weight(experts, 2 * inter, hidden),
               weight(experts, hidden, inter), weight(2 * inter, hidden), weight(hidden, inter),
               weight(1, hidden))  # fmt: skip
    op = MoEBlock()
    op.warm_up(512, *weights)
    x = torch.randn(512, hidden, dtype=torch.bfloat16, device="cuda")
    return lambda m: op(x[:m], *weights)


@pytest.mark.parametrize("make", [_norm, _activation, _linear, _gdn_prefill, _moe])
def test_warmed_first_calls_do_not_wait_for_queued_work(make):
    call = make()
    for m in ROWS:
        queued = _queue_work()
        call(m)
        assert not queued.query(), m
        torch.cuda.synchronize()


def _triton_entries():
    import sys

    from triton.runtime.jit import JITFunction

    functions = {
        id(value): value
        for name, module in list(sys.modules.items())
        if name.startswith("ctikernels")
        for value in vars(module).values()
        if isinstance(value, JITFunction)
    }
    return sum(len(c[0]) for f in functions.values() for c in f.device_caches.values())


def test_moe_warm_rows_cover_every_row_count():
    """warm_up primes a few row counts per kernel variant, not every count: afterwards no
    row count up to the bound compiles (and so first-launches) a new Triton kernel."""
    from ctikernels.moe.inference import MoEBlock

    experts, hidden, inter = 256, 2048, 512

    def weight(*shape):
        return (torch.randn(*shape, device="cuda") * 0.02).bfloat16()

    weights = (weight(experts, hidden), weight(experts, 2 * inter, hidden),
               weight(experts, hidden, inter), weight(2 * inter, hidden), weight(hidden, inter),
               weight(1, hidden))  # fmt: skip
    op = MoEBlock()
    op.warm_up(1024, *weights)
    assert len(op.warm_rows(1024, torch.device("cuda"), experts, 2 * inter)) < 1024 // 4
    entries = _triton_entries()
    x = torch.zeros(1024, hidden, dtype=torch.bfloat16, device="cuda")
    for rows in range(1, 1025):
        op(x[:rows], *weights)
    assert _triton_entries() == entries
