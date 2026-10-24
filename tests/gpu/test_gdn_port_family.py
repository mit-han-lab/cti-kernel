# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""The GDN family with the port's prefill at long lengths: chained port prefill then decode
follows the FP64 reference over the whole sequence, and the training forward equals stateless
port prefill, within tests/gpu/test_gdn.py's tolerances. The port is forced so both RTX 5090
and RTX PRO 6000 run it whatever their provider tables choose."""

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetDecode, GatedDeltaNetPrefill, GatedDeltaNetTraining

from . import _gdn_reference as reference
from .test_gdn import (
    STATE_TOLERANCE,
    Y_TOLERANCE,
    _assert_close,
    _pools,
    _prefill_case,
    _projected,
    _same,
    _weights,
)
from .test_gdn_port_product import _reason

_REASON = _reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")
GEOMETRIES = [(16, 32), (16, 48)]


@pytest.fixture(autouse=True)
def _port(monkeypatch):
    monkeypatch.setenv("CTI_GDN_PREFILL_PORT", "1")


@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("chunks", [[2048, 1024, 64], [512, 4096]])
def test_chained_port_prefill_then_decode(hk, hv, chunks):
    prefill = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert prefill.selected_variant(chunks[0], 1) == "port"
    decode_steps = 4
    total = sum(chunks) + decode_steps
    weights = _weights(hk, hv, seed=11)
    projected = _projected(total, hk, hv, seed=12)
    state, conv_state = _pools(hk, hv, seed=13)
    slot = 5
    cu, slots, initial = _prefill_case([total], [slot], [False])
    expected_y, expected_state, expected_conv = reference.prefill(
        projected, weights, hk, cu, slots, initial, state, conv_state
    )
    decode = GatedDeltaNetDecode(num_k_heads=hk, num_v_heads=hv, device="cuda")
    outputs, start = [], 0
    with torch.inference_mode():
        for index, length in enumerate(chunks):
            cu, slots, initial = _prefill_case([length], [slot], [index > 0])
            rows = projected[start : start + length]
            outputs.append(prefill(rows, *weights.args(), cu, slots, initial, state, conv_state))
            start += length
        slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
        for _ in range(decode_steps):
            row = projected[start : start + 1]
            outputs.append(decode(row, *weights.args(), slots, state, conv_state))
            start += 1
    _assert_close(torch.cat(outputs), expected_y, **Y_TOLERANCE)
    _assert_close(state[slot], expected_state[slot], **STATE_TOLERANCE)
    assert _same(conv_state[slot], expected_conv[slot])


@pytest.mark.parametrize(("hk", "hv"), GEOMETRIES)
@pytest.mark.parametrize("lengths", [[512, 2048], [4096]])
def test_training_forward_matches_stateless_port_prefill(hk, hv, lengths):
    training = GatedDeltaNetTraining(num_k_heads=hk, num_v_heads=hv, device="cuda")
    prefill = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device="cuda")
    assert prefill.selected_variant(sum(lengths), len(lengths)) == "port"
    weights = _weights(hk, hv, seed=21)
    projected = _projected(sum(lengths), hk, hv, seed=22)
    cu, _, _ = _prefill_case(lengths, [0] * len(lengths), [False] * len(lengths))
    count = len(lengths)
    state = torch.zeros(1, hv, 128, 128, device="cuda")
    conv_state = torch.zeros(1, 3, (2 * hk + hv) * 128, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        y = training(projected, *weights.args(), cu, max_sequence_length=max(lengths))
        expected = prefill(
            projected,
            *weights.args(),
            cu,
            torch.full((count,), -1, dtype=torch.int32, device="cuda"),
            torch.zeros(count, dtype=torch.bool, device="cuda"),
            state,
            conv_state,
        )
    torch.cuda.synchronize()
    # tests/gpu/test_gdn_training.py's bound where prefill runs our own kernels.
    _assert_close(y, expected, rms=1e-2, peak=2e-2)
