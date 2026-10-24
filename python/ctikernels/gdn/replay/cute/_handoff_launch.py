# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Cached, allocation-free host adapter for the all-layer handoff."""

from functools import lru_cache
from importlib.metadata import version

from .._layout import physical


def handoff_chunks(hk, hv, copies_checkpoint):
    vectors = 3 * ((2 * hk + hv) * 128 + 2 * hv) // 8
    if copies_checkpoint:
        vectors += hv * 128 * 128 // 4
    return max(1, (vectors + 1023) // 1024)


@lru_cache(maxsize=256)
def compile_handoff(
    hk,
    hv,
    window,
    batch,
    layers,
    slots,
    snapshots,
    record_stride,
    strict,
    chunks,
    arch,
    ringless=False,
    tensor_types=(),
    pitches=(),
    index_stride=1,
    position_stride=1,
):
    if version("nvidia-cutlass-dsl") != "4.8.0":
        raise RuntimeError("replay CuTe requires nvidia-cutlass-dsl[cu13]==4.8.0")
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint8
    from cutlass.runtime import make_fake_stream

    from . import specialize_types
    from ._handoff import Handoff

    dtypes = specialize_types((Int64, Int64, Int32, Uint8, Int64, Int32), tensor_types)
    args = [
        cute.runtime.make_fake_tensor(dtype, (cute.sym_int64(),), (1,), assumed_align=alignment)
        for dtype, alignment in zip(dtypes, (16, 4, 4, 16, 4, 4), strict=True)
    ]
    return cute.compile(
        Handoff(
            hk,
            hv,
            window,
            batch,
            layers,
            slots,
            snapshots,
            record_stride,
            strict,
            chunks,
            ringless,
            pitches,
            index_stride,
            position_stride,
        ),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a --opt-level=3",
    )


def run(op, layers, positions, indices, status):
    import torch

    strict = status is not None
    j = len(layers.states)
    stride = status.records.stride(0) if strict else 16
    chunks = handoff_chunks(op.geometry.k_heads, op.geometry.v_heads, layers.copies_checkpoint)
    key = (
        "handoff_layers",
        op.geometry.k_heads,
        op.geometry.v_heads,
        op.window,
        layers.batch_size,
        j,
        layers.num_slots,
        layers.num_snapshot_rows,
        stride,
        strict,
        chunks,
    )
    # Elide ring code for all-ringless tables; mixed tables use each row's capacity flag.
    ringless = all(state.ring.shape[1] == 0 for state in layers.states)
    base_key = key
    if ringless:
        key += (True,)
    tensor_types = (
        (1, positions.dtype),
        (2, indices.dtype),
        (4, status.sequence.dtype if strict else positions.dtype),
        (5, status.poison.dtype if strict else indices.dtype),
    )
    pitches = tuple(
        (s.checkpoint.stride(0), s.conv_history.stride(0), c.stride(0), h.stride(0))
        for s, c, h in zip(layers.states, layers.canonical_states, layers.histories, strict=True)
    )
    key += (tensor_types, pitches, indices.stride(0), positions.stride(0))
    launcher = op._cute_launchers.get(key)
    if launcher is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm CuTe specialization before capture")
        cc = torch.cuda.get_device_capability(op.device)
        launcher = compile_handoff(
            *base_key[1:],
            f"{cc[0]}{cc[1]}",
            tensor_types=tensor_types,
            pitches=pitches,
            index_stride=indices.stride(0),
            position_stride=positions.stride(0),
            **({"ringless": True} if ringless else {}),
        )
        op._cute_launchers[key] = launcher
    # A strided graph-ring slice has holes between records. Retain that span;
    # flattening via reshape would allocate and lose its physical row stride.
    records = (
        status.records.as_strided(((j - 1) * stride + 16,), (1,))
        if strict
        else layers.table_flat.view(torch.uint8)
    )
    position_view = positions if positions.stride(0) == 1 else physical(positions)
    index_view = indices if indices.stride(0) == 1 else physical(indices)
    launcher(
        layers.table_flat,
        position_view,
        index_view,
        records,
        status.sequence if strict else position_view,
        status.poison if strict else index_view,
    )
