# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""All-layer handoff parity, graph replay, and the R3d host latency gate."""

from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from statistics import median
from time import perf_counter

import pytest
import torch

from ctikernels.gdn import GatedDeltaNetReplay

from ..modules._handoff_case import (
    assert_layer_prefix,
    assert_layer_status,
    compare,
    error_case,
    per_layer,
    setup,
    snapshot,
)
from ._exhaustive import ci_cases


def _available():
    try:
        return torch.cuda.is_available() and version("nvidia-cutlass-dsl") == "4.8.0"
    except PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="CUDA and cutlass DSL 4.8.0 required")


@pytest.mark.parametrize(
    "window,inplace,layers,hk,hv",
    ci_cases(
        [4, 8],
        [True, False],
        [1, 4],
        [(16, 32), (16, 48), (8, 16), (8, 24), (1, 1)],
        core=[
            (8, True, 1, 1, 1),
            (8, True, 1, 8, 16),
            (8, False, 4, 1, 1),
            (8, False, 1, 8, 16),
            (4, False, 4, 8, 24),
        ],
    ),
)
@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_bitwise(hk, hv, layers, inplace, window, provider):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider,
        device="cuda",
        hk=hk,
        hv=hv,
        layers=layers,
        inplace=inplace,
        window=window,
    )
    assert op.provider["handoff_layers"] == provider
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    compare(states, binding, statuses)


@pytest.mark.parametrize(
    "error",
    [
        "position",
        "slot",
        "negative_slot",
        "mapping",
        "padding",
        "poison",
        "zero_sequence",
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_strict_errors(error, inplace, provider):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=8, hv=16, inplace=inplace
    )
    error_case(error, statuses, slots, positions)
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    compare(states, binding, statuses)


@pytest.mark.parametrize("error", ["sequence", "done", "code"])
@pytest.mark.parametrize("layer", [1, 2, 3])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_strict_layer_failure_prefix(error, layer, inplace, provider):
    """A failure at record ``layer`` (the last is 3) keeps exactly layers < layer."""
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=8, hv=16, inplace=inplace
    )
    before = snapshot(binding.states)
    error_case(error, statuses, slots, positions, layer=layer)
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    compare(states, binding, statuses)
    assert_layer_prefix(before, binding, layer)
    assert_layer_status(statuses, error, layer)


@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_detects_swapped_table_rows(provider):
    """Mutant check: swapping pointer-table rows 1 and 2 must not survive the suite."""
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=8, hv=16, inplace=True
    )
    binding.table[[1, 2]] = binding.table[[2, 1]].clone()
    error_case("sequence", statuses, slots, positions, layer=2)
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])
    with pytest.raises(AssertionError):
        compare(states, binding, statuses)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_trusted(inplace, provider):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=16, hv=32, inplace=inplace
    )
    trusted = GatedDeltaNetReplay(
        num_k_heads=16,
        num_v_heads=32,
        window=op.window,
        device="cuda",
        provider=provider,
        checks="trusted",
    )
    binding = trusted.bind_layers(
        states=binding.states,
        table=binding.table,
        canonical_states=binding.canonical_states,
        histories=binding.histories,
    )
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    trusted.handoff_layers(binding, next_positions=positions, slot_indices=slots)
    compare(states, binding)
    # Supplying status still selects strict, including error propagation.
    positions[0] = -1
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    trusted.handoff_layers(
        binding, next_positions=positions, slot_indices=slots, status=statuses[1]
    )
    compare(states, binding, statuses)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_graph(inplace, strict, provider):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=16, hv=32, inplace=inplace
    )
    if not strict:
        op = GatedDeltaNetReplay(
            num_k_heads=16,
            num_v_heads=32,
            window=op.window,
            device="cuda",
            provider=provider,
            checks="trusted",
        )
        binding = op.bind_layers(
            states=binding.states,
            table=binding.table,
            canonical_states=binding.canonical_states,
            histories=binding.histories,
        )
    status = statuses[1] if strict else None
    op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=status)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=status)
    # Restore the exact starting state after warming/capture, then change fixed
    # input buffers. This checks the captured table and dynamic row controls.
    for source, target in zip(states, binding.states, strict=True):
        for field in fields(source):
            getattr(target, field.name).copy_(getattr(source, field.name))
    positions.add_(13)
    slots[0] = -1
    slots[-1] = slots.numel() - 1 if inplace else 0
    graph.replay()
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    compare(states, binding, statuses if strict else None)


@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_capture_requires_warm_binding(provider):
    op, _, binding, _, _, statuses, slots, positions = setup(provider=provider, device="cuda")
    # Warm the specialization with another table; this binding must still warm.
    other = op.bind_layers(
        states=binding.states,
        table=torch.empty_like(binding.table),
        canonical_states=binding.canonical_states,
        histories=binding.histories,
    )
    op.handoff_layers(other, next_positions=positions, slot_indices=slots, status=statuses[1])
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="warm"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op.handoff_layers(
                binding, next_positions=positions, slot_indices=slots, status=statuses[1]
            )


@pytest.mark.parametrize("provider", ["cute", "triton"])
def test_handoff_layers_host_gate(provider):
    op, states, binding, canonical, histories, statuses, slots, positions = setup(
        provider=provider, device="cuda", hk=16, hv=32, layers=24
    )
    fused = lambda: op.handoff_layers(
        binding, next_positions=positions, slot_indices=slots, status=statuses[1]
    )
    loop = lambda: per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    samples = {}
    for name, call in (("per_layer", loop), ("fused", fused)):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(200):
            start = perf_counter()
            call()
            elapsed.append((perf_counter() - start) * 1e6)
        torch.cuda.synchronize()
        samples[name] = median(elapsed)
    print(f"handoff host us: per_layer={samples['per_layer']:.1f}, fused={samples['fused']:.1f}")
    assert samples["fused"] <= 500


@pytest.mark.bitwise
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_handoff_layers_mixed_ring_capacity(inplace, captured):
    from ..modules._handoff_case import mixed_setup

    op, states, binding, canonical, histories, statuses, slots, positions = mixed_setup(
        device="cuda", inplace=inplace
    )

    def fused():
        op.handoff_layers(binding, next_positions=positions, slot_indices=slots, status=statuses[1])

    fused()
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused()
    for source, target in zip(states, binding.states, strict=True):
        for field in fields(source):
            getattr(target, field.name).copy_(getattr(source, field.name))
    positions.add_(11)
    if captured:
        graph.replay()
    else:
        fused()
    per_layer(op, states, canonical, histories, statuses[0], slots, positions)
    compare(states, binding, statuses)
