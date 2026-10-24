# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Lazy, allocation-free adapter for ordered CuTe replay phases."""

import dataclasses
from functools import lru_cache
from importlib.metadata import version

from .._layout import layout as describe_layout
from .._layout import physical
from ._config import Config

# Benchmark switch for the scalar publication baseline; production keeps it on.
VECTOR_PUBLISH = True
# Benchmark override; None selects by batch/head occupancy on the device.
PHASE_TILE = None
# Programmatic dependent launch; included in both specialization caches.
PDL = True
# A/B baseline: trusted compute + separate publication when disabled.
FUSED_PUBLISH = True
# Benchmark overrides for compute CTAs: warps per CTA and the launch bound
# (minimum resident CTAs per SM); None keeps the policy below.
PHASE_WARPS = None
PHASE_MIN_BLOCKS = None


def device_sms(op):
    """The SM count of the op's device, queried once per op."""
    import torch

    cached = op.__dict__.get("_device_sms")
    if cached is None:
        props = torch.cuda.get_device_properties(op.device)
        cached = op._device_sms = props.multi_processor_count
    return cached


def switches():
    return VECTOR_PUBLISH, PHASE_TILE, PDL, FUSED_PUBLISH, PHASE_WARPS, PHASE_MIN_BLOCKS


def per_call(kw):
    values = {
        name: value
        for name, value in kw.items()
        if name not in ("state", "status", "pending") and value is not None
    }
    pending = kw.get("pending")
    if pending is not None:
        values.update({"pending.raw": pending.raw, "pending.metadata": pending.metadata})
    return values


@dataclasses.dataclass(frozen=True)
class PhasePlan:
    """Same launcher and exact view expressions as run, with borrowed inputs unbound.

    Dynamic argument slots contain None: retaining a plan never pins an activation,
    output or pending allocation. Only persistent views live in the template.
    """

    launcher: object
    args: tuple
    recipe: tuple

    def launch(self, per_call):
        args = list(self.args)
        views = {}
        for index, name, kind in self.recipe:
            if name in views:
                args[index] = views[name]
                continue
            value = per_call[name]
            if kind == "view":
                value = value.view(-1)
            elif kind == "physical":
                value = physical(value)
            elif kind != "identity":
                value = value.as_strided((kind,), (1,))
            args[index] = value
            views[name] = value
        self.launcher(*args)


def phase_tile(batch, hv, sms):
    """Use the smallest value tile whose head CTAs fit within one SM wave."""
    for tile in (32, 64):
        if batch * hv * (128 // tile) <= sms:
            return tile
    return 128


def phase_launch(batch, hv, sms, q, config):
    """(tile, warps, min_blocks) for transaction phases. Measured (R3b): when
    full-head CTAs are needed, 16 warps share one prologue and stage up to 16
    tokens per chunk; with more head CTAs than SMs, 64-row tiles at two CTAs per
    SM beat a second wave on discrete GPUs. All choices are bitwise equal."""
    tile = phase_tile(batch, hv, sms)
    if tile == 128 and batch * hv > sms and config.split_oversubscribed:
        return 64, config.warps, 2
    if tile == 128 or q > config.warps:
        return tile, config.phase_warps, config.min_blocks
    return tile, config.warps, config.min_blocks


@lru_cache(maxsize=256)
def compile_phases(
    hk,
    hv,
    window,
    q,
    batch,
    stride,
    slots,
    ns,
    pb,
    capture,
    mode,
    config,
    arch,
    sms,
    vec=False,
    tile=None,
    checks="strict",
    pdl=True,
    fused=True,
    delayed=False,
    arithmetic_batch=None,
    tensor_types=(),
    layout=None,
    slot_layout=(),
):
    tile = phase_tile(batch, hv, sms) if tile is None else tile
    if tile not in (32, 64, 128):
        raise ValueError("phase tile must be 32, 64 or 128")
    if version("nvidia-cutlass-dsl") != "4.8.0":
        raise RuntimeError("replay CuTe requires nvidia-cutlass-dsl[cu13]==4.8.0")
    import cutlass.cute as cute
    from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8
    from cutlass.runtime import make_fake_stream

    from . import arithmetic_plan, specialize_types
    from ._phases import Phases

    # Product decode sees one dense projection per batch row, even for verify.
    # Neither the [B,Q,P] batch stride nor an absent input's zero stride belongs
    # in the product arithmetic dispatch key for materialization.
    projection_width = (2 * hk + 2 * hv) * 128 + 2 * hv
    product_stride = stride if mode in ("bridge_step", "delayed_step") else projection_width
    # Pair count still controls validation and the grid; only the recipe uses B.
    plan_batch = batch if arithmetic_batch is None else arithmetic_batch
    arithmetic = arithmetic_plan(hk, hv, plan_batch, product_stride, arch, sms)
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
        Int64,
        BFloat16,
        Int64,
        Int32,
        Int32,
        Int64,
        Boolean,
        BFloat16,
        Uint8,
        Int64,
        Int32,
        Int32,
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
        256,
        2,
        8,
        4,
        4,
        4,
        1,
        layout.out_alignment if layout else 16,
        16,
        8,
        4,
        # arrivals: scalar atomics only; absent phases borrow the int32 indices.
        4,
    )
    args = [
        cute.runtime.make_fake_tensor(t, (cute.sym_int64(),), (1,), assumed_align=a)
        for t, a in zip(specialize_types(dtypes, tensor_types), alignments, strict=True)
    ]
    return cute.compile(
        Phases(
            hk,
            hv,
            window,
            batch,
            stride,
            config,
            slots,
            arithmetic,
            mode,
            q,
            ns,
            pb,
            capture,
            vec,
            tile,
            checks == "trusted",
            pdl,
            fused,
            delayed,
            layout,
            **({"slot_layout": slot_layout} if slot_layout else {}),
        ),
        *args,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        *(
            [cute.runtime.make_fake_tensor(Int32, (1,), (1,), assumed_align=4)]
            if slot_layout
            else []
        ),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a --opt-level=3",
    )


def vector_publication(op, pending):
    """16-byte raw-row publication: x/ring/history rows are 16-byte aligned once
    Hv % 4 == 0; a packed per-layer pending view must be checked, and a stacked
    [D,B,Q,W] view also needs a 16-byte slot pitch so every selected slot is aligned."""
    if not VECTOR_PUBLISH or op.geometry.v_heads % 4:
        return False
    if pending is None:
        return True
    raw = pending.raw
    pitch = raw.stride(0) * raw.element_size() if raw.ndim == 4 else 0
    return raw.data_ptr() % 16 == 0 and pitch % 16 == 0


def plan(op, mode, *, state, status, weights=None, **kw):
    """Prepare the original specialization and ABI without launching or retaining inputs."""
    import torch

    indices = kw["pairs"] if mode == "export_snapshot" else kw["slot_indices"]
    x, pending = kw.get("projected"), kw.get("pending")
    batch = indices.shape[0]
    stride = x.stride(0) if x is not None else 0
    selector = kw.get("status_slot")
    pb = pending.raw.shape[int(selector is not None)] if pending is not None else 0
    capture = kw.get("snapshot_rows") is not None
    fused = kw.pop("fused", FUSED_PUBLISH)
    delayed = (
        mode == "commit"
        and op.checks == "trusted"
        and fused
        and op.commit_mode_for(batch) == "delayed"
    )
    layout = describe_layout(op.geometry, state, kw)
    vec = vector_publication(op, pending) and layout.x_alignment == 16
    # Launcher-only stubs carry no CuTe config; the defaults define the policy.
    config = kw.pop("step_config", op.cute_config)
    policy = config if config is not None else Config()
    tile, warps, min_blocks = PHASE_TILE, policy.warps, policy.min_blocks
    if tile is None:
        tile, warps, min_blocks = phase_launch(
            batch,
            op.geometry.v_heads,
            device_sms(op),
            op.verify_length,
            policy,
        )
    if tile not in (32, 64, 128):
        raise ValueError("phase tile must be 32, 64 or 128")
    warps = warps if PHASE_WARPS is None else PHASE_WARPS
    min_blocks = min_blocks if PHASE_MIN_BLOCKS is None else PHASE_MIN_BLOCKS
    if config is not None and (warps, min_blocks) != (config.warps, config.min_blocks):
        config = dataclasses.replace(config, warps=warps, min_blocks=min_blocks)
    if config is not None:
        # Mixed bucket policies need phases capable of materializing delayed slots.
        # Product-routed steps still run their transaction phases here.
        config = dataclasses.replace(
            config,
            immediate=state.ring.shape[1] == 0,
            provider="cute",
            flush_always=mode == "delayed_step" and config.flush_always,
            snapshot_controls=False,
        )
    key = (
        mode,
        batch,
        stride,
        state.metadata.shape[0],
        state.snapshot_metadata.shape[0],
        pb,
        capture,
        config,
        vec,
        tile,
        op.checks,
        PDL,
        fused,
        delayed,
    )
    if kw.get("arithmetic_batch") is not None:
        key += (kw["arithmetic_batch"],)
    # Typed borrowed placeholders are removed by specialization. Views create no
    # device storage, including projected inputs with padding between batch rows.
    xf = state.ring.view(-1)
    if x is not None:
        if layout.token == op.geometry.proj_width:
            row_width = op.geometry.proj_width * (op.verify_length if mode == "verify" else 1)
            xf = x.as_strided((max(0, (batch - 1) * stride + row_width),), (1,))
        else:
            xf = physical(x)

    def flat(value):
        # Preserve the original launch views for dense arguments, including
        # typed placeholders; only pitched arguments need their physical span.
        return value.view(-1) if value.is_contiguous() else physical(value)

    def optional(name, fallback):
        value = kw.get(name)
        return fallback if value is None else value

    # Trusted specializations never dereference status arguments. Borrow typed
    # views of existing storage; no status or scratch allocation is needed.
    trusted = op.checks == "trusted"
    slot_layout = ()
    if selector is not None:
        slot_layout = (
            pending.raw.shape[0],
            0 if trusted else status.records.stride(1),
            0 if trusted else status.sequence.stride(0),
            pending.raw.stride(0),
            pending.metadata.stride(0),
            kw["snapshot_valid"].stride(0) if capture else 0,
            kw["arrivals"].stride(0) if kw.get("arrivals") is not None else 0,
        )
    args = (
        xf,
        *(
            flat(w)
            for w in (weights or (state.ring, state.checkpoint, state.checkpoint, state.checkpoint))
        ),
        flat(indices),
        flat(kw.get("positions", state.pool_epoch)),
        flat(state.checkpoint),
        flat(state.conv_history),
        state.ring.view(-1),
        state.metadata.view(-1),
        state.pool_epoch,
        state.snapshot_metadata.view(-1),
        flat(pending.raw) if pending is not None else state.ring.view(-1),
        flat(pending.metadata) if pending is not None else state.metadata.view(-1),
        flat(optional("accepted_lengths", indices)),
        flat(kw.get("snapshot_rows") if capture else indices),
        flat(
            kw.get("offsets")
            if mode == "export_snapshot" and pending is not None
            else kw.get("snapshot_offsets")
            if capture
            else state.pool_epoch
        ),
        flat(
            optional(
                "snapshot_valid",
                state.ring.view(torch.bool) if trusted else status.validity[status.layer],
            )
        ),
        flat(kw.get("out", state.ring)),
        state.ring.view(torch.uint8).view(-1) if trusted else flat(status.record),
        state.pool_epoch if trusted else status.sequence,
        flat(indices) if trusted else status.poison,
        flat(optional("arrivals", indices)),
    )
    tensor_types = tuple((i, arg.dtype) for i, arg in enumerate(args))
    key += (tensor_types, layout)
    if slot_layout:
        key += (slot_layout,)
        args += (selector,)
    if key not in op._cute_launchers:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm CuTe specialization before capture")
        cc = torch.cuda.get_device_capability(op.device)
        sms = device_sms(op)
        op._cute_launchers[key] = compile_phases(
            op.geometry.k_heads,
            op.geometry.v_heads,
            op.window,
            op.verify_length,
            batch,
            stride,
            *key[3:7],
            mode,
            config,
            f"{cc[0]}{cc[1]}",
            sms,
            vec,
            tile,
            op.checks,
            PDL,
            fused,
            delayed,
            tensor_types=tensor_types,
            layout=layout,
            **({"slot_layout": slot_layout} if slot_layout else {}),
            **(
                {"arithmetic_batch": kw["arithmetic_batch"]}
                if kw.get("arithmetic_batch") is not None
                else {}
            ),
        )
    recipe = []

    def bind(index, name, kind=None):
        value = sources.get(name)
        if value is not None:
            recipe.append(
                (
                    index,
                    name,
                    ("view" if value.is_contiguous() else "physical") if kind is None else kind,
                )
            )

    sources = per_call(kw)
    bind(
        0,
        "projected",
        xf.numel() if x is not None and layout.token == op.geometry.proj_width else "physical",
    )
    index_name = "pairs" if mode == "export_snapshot" else "slot_indices"
    bind(5, index_name)
    bind(6, "positions")
    bind(13, "pending.raw")
    bind(14, "pending.metadata")
    bind(15, "accepted_lengths" if kw.get("accepted_lengths") is not None else index_name)
    bind(16, "snapshot_rows" if capture else index_name)
    bind(
        17,
        "offsets"
        if mode == "export_snapshot" and pending is not None
        else "snapshot_offsets"
        if capture
        else "absent",
    )
    bind(18, "snapshot_valid")
    bind(19, "out")
    if trusted:
        bind(22, index_name)
    bind(23, "arrivals" if kw.get("arrivals") is not None else index_name)
    bind(24, "status_slot", "identity")
    template = list(args)
    for index, _, _ in recipe:
        template[index] = None
    return PhasePlan(op._cute_launchers[key], tuple(template), tuple(recipe))


def run(op, mode, *, state, status, weights=None, **kw):
    prepared = plan(op, mode, state=state, status=status, weights=weights, **kw)
    prepared.launch(per_call(kw))
    return prepared
