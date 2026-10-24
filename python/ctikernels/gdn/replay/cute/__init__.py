# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CuTe product recurrence with portable non-arithmetic lifecycle phases."""

from dataclasses import replace
from functools import lru_cache
from importlib.metadata import version

from .._layout import layout as describe_layout
from .._layout import physical
from ._config import Config, all_immediate, commit_mode, decode_route, select


def specialize_types(dtypes, overrides):
    """Match fake ABI types to borrowed tensors, including unused placeholders."""
    import cutlass

    names = {
        "torch.float32": "Float32",
        "torch.float16": "Float16",
        "torch.bfloat16": "BFloat16",
        "torch.int32": "Int32",
        "torch.int64": "Int64",
        "torch.uint8": "Uint8",
        "torch.bool": "Boolean",
    }
    result = list(dtypes)
    for index, dtype in overrides:
        result[index] = getattr(cutlass, names[str(dtype)])
    return tuple(result)


def arithmetic_plan(hk, hv, batch, stride, arch, sms=None):
    from ..._contract import Geometry
    from ..._triton_decode import decode_plan

    sms = sms if sms is not None else (48 if str(arch) == "121" else 170)
    plan = decode_plan(Geometry(hk, hv), max(1, batch), stride, True, sms=sms).arguments
    return tuple(plan[k] for k in ("BV", "num_warps", "FIN", "WSUM", "BFLY", "A3", "NCOL"))


@lru_cache(maxsize=128)
def compile_step(
    hk,
    hv,
    window,
    batch,
    stride,
    config,
    arch,
    slots=1,
    complete=False,
    advance=False,
    sms=None,
    tensor_types=(),
    layout=None,
):
    if version("nvidia-cutlass-dsl") != "4.8.0":
        raise RuntimeError("replay CuTe requires nvidia-cutlass-dsl[cu13]==4.8.0")
    import cutlass.cute as cute
    from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8
    from cutlass.runtime import make_fake_stream

    from ._kernel import Step

    # Only launch metadata is consulted. No CUDA discovery in offline compilation.
    arithmetic = arithmetic_plan(hk, hv, batch, stride, arch, sms)

    dtypes = (
        BFloat16,
        BFloat16,
        Float32,
        Float32,
        Float32,
        Int32,
        Int64,
        Float32,
        BFloat16,
        BFloat16,
        Int64,
        Int64,
        Uint8,
        Int64,
        Int32,
        Boolean,
        BFloat16,
        Int32,
        Int64,
        Int64,
    )
    alignments = (
        layout.x_alignment if layout else 16,
        16,
        16,
        16,
        16,
        4,
        4,
        layout.checkpoint_alignment if layout else 256,
        layout.history_alignment if layout else 256,
        256,
        256,
        256,
        16,
        8,
        4,
        1,
        layout.out_alignment if layout else 16,
        4,
        8,
        8,
    )
    args = [
        cute.runtime.make_fake_tensor(t, (cute.sym_int64(),), (1,), assumed_align=a)
        for t, a in zip(specialize_types(dtypes, tensor_types), alignments, strict=True)
    ]
    return cute.compile(
        Step(hk, hv, window, batch, stride, config, slots, complete, advance, arithmetic, layout),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a --opt-level=3",
    )


def effective_config(op, batch):
    name = getattr(op, "_cute_device_name", None)
    config = select(
        name,
        batch,
        mode=getattr(op, "mode", "auto"),
        config=None if name is not None else op.cute_config,
        geometry=(op.geometry.k_heads, op.geometry.v_heads),
        window=op.window,
    )
    config = replace(config, trusted=getattr(op, "checks", "trusted") == "trusted")
    if config.provider == "product" and not config.trusted:
        # Strict checks never take #25: the guarded immediate step follows its recipe.
        config = replace(config, provider="cute")
    if not getattr(op, "_ringless", True) and config.immediate:
        config = replace(config, immediate=False, flush_always=True, provider="cute")
    return config


def prepare(op, x, state, complete, advance, tensor_types=(), layout=None):
    import torch

    config = effective_config(op, x.shape[0])
    if state.ring.shape[1] and config.immediate and config.trusted:
        # A borrowed full allocation can contain R3c rows even on a ringless op.
        # Strict immediate rejects pending rows instead, so it keeps the ring intact.
        config = replace(config, immediate=False, flush_always=True, provider="cute")
    if config.provider == "product":
        raise ValueError("product route must dispatch through replay.step")
    key = (
        x.shape[0],
        x.stride(0),
        state.metadata.shape[0],
        complete,
        advance,
        config,
        tensor_types,
        layout,
    )
    if key not in op._cute_launchers:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm CuTe specialization before capture")
        cc = torch.cuda.get_device_capability(op.device)
        props = torch.cuda.get_device_properties(op.device)
        op._cute_launchers[key] = compile_step(
            op.geometry.k_heads,
            op.geometry.v_heads,
            op.window,
            *key[:2],
            config,
            f"{cc[0]}{cc[1]}",
            *key[2:5],
            sms=props.multi_processor_count,
            tensor_types=tensor_types,
            layout=layout,
        )
    return op._cute_launchers[key]


def run(op, mode, *, state, status, weights=None, **kw):
    if mode in ("verify", "commit", "reject", "flush", "bridge_step", "export_snapshot"):
        from ._phase_launch import run as phase_run

        return phase_run(op, mode, state=state, status=status, weights=weights, **kw)
    if mode != "step":
        from ..triton import run as portable

        portable(op, mode, state=state, status=status, weights=weights, **kw)
        return
    if kw.get("product_fallback"):
        from ._phase_launch import run as phase_run

        # Ordered compute/publication needs no counters, preserving #25's
        # workspace-free call form. This uses the same product arithmetic plan.
        phase_run(op, "bridge_step", state=state, status=status, weights=weights, fused=False, **kw)
        return
    x = kw["projected"]
    if op.window > 32:
        config = effective_config(op, x.shape[0])
        if state.ring.shape[1] and config.immediate and config.trusted:
            config = replace(config, immediate=False, flush_always=True, provider="cute")
        if not config.immediate:
            # Phases already stages at most warps tokens, independent of the window.
            # The dedicated mode retains Step's ring append and exact flush boundary.
            from ._phase_launch import run as phase_run

            phase_run(
                op,
                "delayed_step",
                state=state,
                status=status,
                weights=weights,
                step_config=config,
                **kw,
            )
            return
    tensor_types = tuple((i + 1, w.dtype) for i, w in enumerate(weights)) + (
        (5, kw["slot_indices"].dtype),
        (6, kw["positions"].dtype),
    )
    layout = describe_layout(op.geometry, state, kw)
    fn = prepare(
        op,
        x,
        state,
        kw["complete"],
        kw["advance_positions"],
        tensor_types,
        layout,
    )
    xf = x.as_strided((max(0, (x.shape[0] - 1) * x.stride(0) + x.shape[1]),), (1,))
    # Unused ABI arguments borrow typed views; trusted Step never dereferences
    # status/epoch controls. This also permits status=None without allocations.
    if op.cute_config.trusted:
        import torch

        record = state.metadata.view(torch.uint8).view(-1)
        sequence = state.pool_epoch
        poison = kw["arrivals"]
        valid = state.metadata.view(torch.bool).view(-1)
    else:
        record, sequence, poison = status.record, status.sequence, status.poison
        valid = status.validity[status.layer]
    fn(
        xf,
        *(w.view(-1) for w in weights),
        kw["slot_indices"] if layout.indices == 1 else physical(kw["slot_indices"]),
        kw["positions"] if layout.positions == 1 else physical(kw["positions"]),
        state.checkpoint.view(-1)
        if layout.checkpoint == op.geometry.v_heads * 16384
        else physical(state.checkpoint),
        state.conv_history.view(-1)
        if layout.history == 3 * op.geometry.conv_dim
        else physical(state.conv_history),
        state.ring.view(-1),
        state.metadata.view(-1),
        state.pool_epoch,
        record,
        sequence,
        poison,
        valid,
        kw["out"].view(-1) if layout.out_batch == op.geometry.value_width else physical(kw["out"]),
        kw["arrivals"] if not op.cute_config.snapshot_controls else poison,
        kw["sections"].view(-1) if op.cute_config.sections else sequence,
        kw["controls"] if op.cute_config.snapshot_controls else sequence,
    )
    return None


@lru_cache(maxsize=128)
def compile_advance_positions(batch, slots, arch, index_dtype=None, index_stride=1):
    if version("nvidia-cutlass-dsl") != "4.8.0":
        raise RuntimeError("replay CuTe requires nvidia-cutlass-dsl[cu13]==4.8.0")
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.runtime import make_fake_stream

    from ._kernel import AdvancePositions

    args = [
        cute.runtime.make_fake_tensor(t, (cute.sym_int64(),), (1,), assumed_align=a)
        for t, a in (
            (Int64, 8),
            (specialize_types((Int32,), ((0, index_dtype),))[0] if index_dtype else Int32, 4),
        )
    ]
    return cute.compile(
        AdvancePositions(batch, slots, index_stride),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a --opt-level=3",
    )


def advance_positions(op, state, slots):
    """Product-route metadata: checkpoint and committed positions advance by one."""
    import torch

    key = (
        "advance_positions",
        slots.shape[0],
        state.metadata.shape[0],
        slots.dtype,
        slots.stride(0),
    )
    if key not in op._cute_launchers:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm CuTe specialization before capture")
        cc = torch.cuda.get_device_capability(op.device)
        op._cute_launchers[key] = compile_advance_positions(
            *key[1:3], f"{cc[0]}{cc[1]}", slots.dtype, index_stride=slots.stride(0)
        )
    if slots.shape[0]:
        op._cute_launchers[key](
            state.metadata.view(-1), slots if slots.stride(0) == 1 else physical(slots)
        )


__all__ = [
    "Config",
    "all_immediate",
    "decode_route",
    "select",
    "commit_mode",
    "fused_publish_enabled",
    "handoff_layers",
    "advance_positions",
    "ring_empty",
]


@lru_cache(maxsize=128)
def compile_ring_empty(batch, slots, arch, index_dtype=None, index_stride=1):
    if version("nvidia-cutlass-dsl") != "4.8.0":
        raise RuntimeError("replay CuTe requires nvidia-cutlass-dsl[cu13]==4.8.0")
    import cutlass.cute as cute
    from cutlass import Boolean, Int32, Int64
    from cutlass.runtime import make_fake_stream

    from ._kernel import RingEmpty

    args = [
        cute.runtime.make_fake_tensor(t, (cute.sym_int64(),), (1,), assumed_align=a)
        for t, a in (
            (Int64, 8),
            (specialize_types((Int32,), ((0, index_dtype),))[0] if index_dtype else Int32, 4),
            (Boolean, 1),
        )
    ]
    return cute.compile(
        RingEmpty(batch, slots, index_stride),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a --opt-level=3",
    )


def ring_empty(op, state, slots, flag):
    import torch

    key = ("ring_empty", slots.shape[0], state.metadata.shape[0], slots.dtype, slots.stride(0))
    if key not in op._cute_launchers:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm CuTe specialization before capture")
        cc = torch.cuda.get_device_capability(op.device)
        op._cute_launchers[key] = compile_ring_empty(
            *key[1:3], f"{cc[0]}{cc[1]}", slots.dtype, index_stride=slots.stride(0)
        )
    op._cute_launchers[key](
        state.metadata.view(-1), slots if slots.stride(0) == 1 else physical(slots), flag
    )


def fused_publish_enabled():
    from ._phase_launch import FUSED_PUBLISH

    return FUSED_PUBLISH


def handoff_layers(op, layers, next_positions, slot_indices, status):
    from ._handoff_launch import run

    run(op, layers, next_positions, slot_indices, status)
