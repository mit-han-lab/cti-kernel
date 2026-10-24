# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Portable replay API."""

from collections import OrderedDict
from dataclasses import fields, replace
from importlib import import_module
from types import MappingProxyType

import torch

from .._contract import CONV, Geometry, V, check_tensor
from ._host import HostPlan, descriptor
from ._layout import dense_rows, feature_rows, overlapping_pairs, overlaps
from ._status import ReplayGraphBuffers, ReplayStatus, graph_control_sizes, read_status_copy
from ._storage import ReplayLayers, ReplayPending, ReplayState, regions, replay_sizes

HOST_FAST_PATH = True


def require_frozen(operation, weights):
    """Base weights are frozen: requires_grad raises in every grad mode (no_grad and
    inference_mode included), before any launch."""
    for name, weight in weights.items():
        if isinstance(weight, torch.Tensor) and weight.requires_grad:
            raise ValueError(
                f"{operation}: {name} is a frozen base weight and must not require grad"
            )


def check_weights(geometry, device, conv_weight, A_log, dt_bias, norm_weight):
    """Replay's weight range; other GDN operations retain their shared contract."""
    check_tensor("conv_weight", conv_weight, torch.bfloat16, (geometry.conv_dim, CONV), device)
    for name, weight, shape in (
        ("A_log", A_log, (geometry.v_heads,)),
        ("dt_bias", dt_bias, (geometry.v_heads,)),
        ("norm_weight", norm_weight, (V,)),
    ):
        if isinstance(weight, torch.Tensor) and weight.dtype not in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            raise TypeError(f"{name} must be FP32, BF16 or FP16")
        check_tensor(name, weight, getattr(weight, "dtype", torch.float32), shape, device)


_INTEGER_CONTROLS = frozenset(
    (
        "slot_indices",
        "pairs",
        "accepted_lengths",
        "snapshot_rows",
        "positions",
        "next_positions",
        "snapshot_offsets",
        "offsets",
    )
)

# Devices where the CuTe transaction phases are proven bitwise equal to the
# product decode recipe. Other devices default to the portable provider.
# GB10 qualified 2026-10-15: product-bitwise 88/88, trusted K2 475/475.
_CUTE_DEVICES = ("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000", "NVIDIA GB10")


def _cute_qualified(device):
    from importlib.metadata import PackageNotFoundError, version

    try:
        if version("nvidia-cutlass-dsl") != "4.8.0":
            return False
    except PackageNotFoundError:
        return False
    return torch.cuda.get_device_name(device).startswith(_CUTE_DEVICES)


def _cute_handoff_available():
    """Copy/control work has no product-arithmetic device allowlist."""
    from importlib.metadata import PackageNotFoundError, version
    from importlib.util import find_spec

    try:
        return version("nvidia-cutlass-dsl") == "4.8.0" and find_spec("cutlass") is not None
    except (PackageNotFoundError, ImportError):
        return False


_PHASES = (
    "step",
    "flush",
    "verify",
    "commit",
    "reject",
    "export_snapshot",
    "import_snapshot",
    "initialize",
    "handoff",
    "handoff_layers",
    "release",
    "reset",
    "begin",
    "end",
    "bridge_step",
)


_overlap = overlaps


def _tensors(obj):
    return [getattr(obj, f.name) for f in fields(obj)]


def _identity(named):
    return tuple(
        (
            name,
            t.data_ptr(),
            tuple(t.shape),
            t.stride(),
            t.dtype,
            t.untyped_storage().data_ptr(),
            t.device,
            t.requires_grad,
        )
        for name, t in named
    )


def _fields(obj):
    return [(f.name, getattr(obj, f.name)) for f in fields(obj)]


class GatedDeltaNetReplay:
    """Frozen-weight replay state ops (§6), all outputs/storage borrowed.

    CUDA uses CuTe or portable Triton; CPU uses the torch semantic reference. ``reference_dtype``
    selects FP32/FP64 arithmetic on CPU only. FP64 checkpoint storage is permitted only
    by that reference for an unrounded oracle; production storage is always FP32.
    K=V=128, width-four convolution, positive grouped local heads are supported.
    Warm each call shape eagerly before capture. No device allocations or host reads
    occur in the CUDA provider. CuTe strict precision selects immediate writeback;
    cross-batch/chunk precision remains unqualified.
    Export/import run outside capture, on the current stream. Trusted CuTe flush
    is graph-capturable and borrows phase arrivals like commit.
    CuTe K2 phases default to checks="trusted": status is optional and no
    begin/end is needed. checks="strict" retains status/poison validation.
    Trusted CuTe step also needs no begin/end and ignores a supplied status.
    Portable providers and lifecycle operations retain their status contract.
    checks="trusted" caches host validation by exact buffer identity on every provider
    (portable providers still report and keep strict status semantics); new layouts
    are checked on first use. checks="strict" validates every invocation.
    The engine uploads each pair table once; operations never perform H2D copies.
    Snapshot indices are absolute rows [R,R+S), never live rows. The caller guarantees
    unique write destinations, serializes lifecycle changes, and pins each verified
    slot and its weights until commit/reject. Disjoint invocations borrow disjoint
    pending/output buffers. Snapshot valid bits are payload flags, not publication
    permission; the engine ledger excludes rejected or failed provisional exports.

    commit_mode=None or "eager" keeps the eager commit (the default; the canonical
    checkpoint is always current). "auto" selects per batch bucket from the measured
    device policy (cute._config.COMMIT_POLICY); "delayed" forces delayed commit.
    Opt in to "auto"/"delayed" only after every direct checkpoint consumer flushes.
    Delayed commit requires trusted CuTe and can leave n > 0 buffered rows.
    Call flush(slots) before direct checkpoint consumers: product decode,
    prefill/extend, row-copy snapshots, or handoff to another provider. This op's
    verify/commit/reject/export_snapshot/step/bridge_step handle n > 0 themselves.
    Handoff/import/initialize/release/reset reset the ring as before.
    """

    def __init__(
        self,
        *,
        num_k_heads: int,
        num_v_heads: int,
        device,
        window: int = 8,
        verify_length: int = 16,
        precision: str = "default",
        reference_dtype: torch.dtype = torch.float64,
        provider: str | None = None,
        cute_config=None,
        checks: str = "trusted",
        commit_mode: str | None = None,
        mode: str = "auto",
    ):
        if checks not in ("trusted", "strict"):
            raise ValueError("checks must be trusted or strict")
        if commit_mode not in (None, "eager", "delayed", "auto"):
            raise ValueError("commit_mode must be None, eager, delayed or auto")
        commit_mode = commit_mode or "eager"
        self.geometry = Geometry(num_k_heads, num_v_heads)
        replay_sizes(
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            window=window,
            verify_length=verify_length,
        )
        self.device = (
            torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("replay supports CPU reference or CUDA Triton")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if precision not in ("default", "strict"):
            raise ValueError("unknown replay precision tier")
        if reference_dtype not in (torch.float32, torch.float64):
            raise ValueError("reference_dtype must be FP32 or FP64")
        self.window, self.verify_length = window, verify_length
        self.reference_dtype = reference_dtype
        auto = provider is None and self.device.type == "cuda" and _cute_qualified(self.device)
        if auto:
            provider = "cute"
        provider = provider or ("triton" if self.device.type == "cuda" else "torch_reference")
        if provider not in ("triton", "cute", "torch_reference"):
            raise ValueError("unknown replay provider")
        if (provider == "torch_reference") != (self.device.type == "cpu"):
            raise ValueError("torch_reference requires CPU; triton/cute require CUDA")
        if cute_config is not None and provider != "cute":
            raise ValueError("cute_config requires provider='cute'")
        self.checks = checks if provider != "torch_reference" else "strict"
        if commit_mode == "delayed" and not (provider == "cute" and self.checks == "trusted"):
            raise ValueError("delayed commit requires provider='cute' and checks='trusted'")
        self._commit_mode = commit_mode
        self._commit_modes = {}
        self._device_name = ""
        if mode not in ("auto", "immediate", "delayed"):
            raise ValueError("mode must be auto, immediate or delayed")
        if provider != "cute" and (mode != "auto" or precision == "strict"):
            raise ValueError("strict replay precision / explicit mode requires provider='cute'")
        if precision == "strict":
            if mode == "delayed" or (
                mode == "auto"
                and cute_config is not None
                and not getattr(cute_config, "immediate", False)
            ):
                raise ValueError("strict precision requires immediate")
            mode = "immediate"
        if checks not in ("trusted", "strict"):
            raise ValueError("checks must be trusted or strict")
        self.checks = checks if provider != "torch_reference" else "strict"
        # Requested trust also caches host validation for portable providers.
        self._host_trusted = checks == "trusted"
        self.mode = mode
        self._ringless = False
        self.cute_config = None
        self._cute_device_name = None
        if provider == "cute":
            from .cute import Config, all_immediate, select
            from .cute import commit_mode as select_commit_mode

            if cute_config is None:
                self._cute_device_name = torch.cuda.get_device_name(self.device)
            if cute_config is not None and not isinstance(cute_config, Config):
                raise ValueError("cute_config must be replay.cute.Config")
            self.cute_config = replace(
                select(
                    self._cute_device_name,
                    mode=mode,
                    config=cute_config,
                    geometry=(num_k_heads, num_v_heads),
                    window=window,
                ),
                trusted=checks == "trusted",
            )
            if self.cute_config.snapshot_controls and checks != "strict":
                raise ValueError("snapshot_controls requires checks='strict'")
            self._device_name = self._cute_device_name
            commit_ring = self.checks == "trusted" and commit_mode == "delayed"
            if self.checks == "trusted" and commit_mode == "auto":
                if self._device_name is None:
                    self._device_name = torch.cuda.get_device_name(self.device)
                commit_ring = any(
                    select_commit_mode(
                        self._device_name,
                        b,
                        verify_length,
                        window,
                        geometry=(num_k_heads, num_v_heads),
                    )
                    == "delayed"
                    for b in (1, 2, 3, 4, 5)
                )
            self._ringless = not commit_ring and all_immediate(
                self._cute_device_name,
                mode=mode,
                config=cute_config,
                geometry=(num_k_heads, num_v_heads),
                window=window,
            )
        phases = dict.fromkeys(_PHASES, "triton" if provider == "cute" else provider)
        if provider == "cute":
            for phase in (
                "step",
                "verify",
                "commit",
                "reject",
                "flush",
                "bridge_step",
                "export_snapshot",
            ):
                phases[phase] = "cute"
            if auto:
                # Default dispatch keeps the portable step (no borrowed step workspace).
                phases["step"] = "triton"
                self._ringless = False
        phases["handoff_layers"] = (
            "cute" if provider == "cute" and _cute_handoff_available() else phases["handoff"]
        )
        self.provider = MappingProxyType(phases)
        self._step_config = self.cute_config if phases["step"] == "cute" else None
        self.capabilities = MappingProxyType(
            {
                "phases": _PHASES,
                "checks": self.checks,
                "capture": tuple(
                    p
                    for p in _PHASES
                    if p not in ("export_snapshot", "import_snapshot", "flush")
                    or (p == "flush" and phases[p] == "cute" and self.checks == "trusted")
                ),
                "commit_modes": ("eager", "delayed")
                if provider == "cute" and self.checks == "trusted"
                else ("eager",),
                "windows": "powers of two >= 2",
                "verify_lengths": (1, None),
                "geometries": ((num_k_heads, num_v_heads),),
                "head_dims": (128, 128),
                "conv_width": 4,
                "memory_version": "r2-k2",
                "precision_tiers": ("default", "strict") if provider == "cute" else ("default",),
                "strict_qualified": False,
                "snapshot_offsets": True,
                "ring_empty": provider == "cute",
                "scratch_bytes": (None if self._step_config is not None else 0),
                "step_scratch_bytes_per_row": (
                    (
                        56 * num_v_heads * self._step_config.sections
                        + 16 * self._step_config.snapshot_controls
                        + 4 * self._step_config.trusted
                    )
                    if self._step_config is not None
                    else 0
                ),
                "step_scratch_fixed_bytes": (
                    (
                        8
                        if self._step_config.snapshot_controls
                        else 0
                        if self._step_config.trusted
                        else 4
                    )
                    + (56 if self._step_config.sections else 0)
                    if self._step_config is not None
                    else 0
                ),
                "phase_providers": self.provider,
                "qualified": False,
                "strict_chunk_cache_phase": False,
            }
        )
        self._backends = {}
        self._warm = set()
        self._validated = OrderedDict()
        self._plans = OrderedDict()
        self._cute_launchers = {}
        self._product_decode = None
        self._side_stream = None

    def commit_mode_for(self, batch_size: int) -> str:
        """Effective commit mode for this batch/graph key (no device reads).

        Unfused publication and strict/portable paths always use eager commit.
        Device-policy choices are pinned on first query for each batch size.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if self.provider["commit"] != "cute" or self.checks != "trusted":
            return "eager"
        from .cute import commit_mode, fused_publish_enabled

        if not fused_publish_enabled():
            return "eager"
        if batch_size not in self._commit_modes:
            mode = self._commit_mode
            if mode == "auto":
                if self._device_name is None:
                    self._device_name = torch.cuda.get_device_name(self.device)
                mode = commit_mode(
                    self._device_name,
                    batch_size,
                    self.verify_length,
                    self.window,
                    geometry=(self.geometry.k_heads, self.geometry.v_heads),
                )
            if self._ringless and mode == "delayed":
                raise ValueError("commit policy changed after ring sizing; construct a new op")
            self._commit_modes[batch_size] = mode
        return self._commit_modes[batch_size]

    def state_spec(self, *, num_slots: int, num_snapshot_rows: int):
        """Persistent regions; zero metadata/epoch or reset before first use (§8)."""
        return {
            k: v
            for k, v in replay_sizes(
                num_k_heads=self.geometry.k_heads,
                num_v_heads=self.geometry.v_heads,
                window=self.window,
                verify_length=self.verify_length,
                num_slots=num_slots,
                num_snapshot_rows=num_snapshot_rows,
                batch_size=0,
                immediate=self._ringless,
            ).items()
            if not k.startswith("pending.")
        }

    def _phase_arrivals(self, phase):
        return (
            self.checks == "trusted"
            and self.provider[phase] == "cute"
            and phase in ("commit", "bridge_step", "export_snapshot", "flush")
        )

    def bytes_per_slot(self):
        """Immutable payload bytes for one live slot, excluding region alignment."""
        g = self.geometry
        sizes = {
            "checkpoint": g.v_heads * 128 * 128 * 4,
            "conv_history": 3 * g.conv_dim * 2,
            "ring": 0 if self._ringless else (self.window + 3) * (g.conv_dim + 2 * g.v_heads) * 2,
            "metadata": 8 * 8,
        }
        return MappingProxyType({**sizes, "total": sum(sizes.values())})

    def bytes_per_snapshot_row(self):
        """Immutable payload bytes for one canonical snapshot row."""
        live = self.bytes_per_slot()
        sizes = {k: live[k] for k in ("checkpoint", "conv_history")}
        sizes["snapshot_metadata"] = 4 * 8
        return MappingProxyType({**sizes, "total": sum(sizes.values())})

    def ring_empty(self, state, slot_indices, out_flag):
        """Write a device bool without synchronization; -1 rows are ignored.

        Invalid slot indices return false. Warm this shape before graph capture.
        Checks n==0 and checkpoint position==next position, not slot validity.
        """
        if self.cute_config is None:
            raise ValueError("ring_empty requires provider='cute'")
        self._check_state(state)
        b = slot_indices.shape[0]
        self._tensor("slot_indices", slot_indices, torch.int32, (b,), alignment=4)
        self._tensor("out_flag", out_flag, torch.bool, (1,), alignment=1)
        self._check_disjoint(
            "ring_empty",
            [out_flag],
            [*_tensors(state), slot_indices],
            "out_flag must not overlap state or slots",
        )
        from .cute import ring_empty

        ring_empty(self, state, slot_indices, out_flag)

    def decode_route(self, batch_size: int) -> str:
        """Measured per-(device, geometry, batch) decode route: "product", "immediate" or "delayed".

        "product" means step calls #25 internally for this batch. Mixed policies route
        product entries to "immediate" to drain buffered tokens through the flush bridge.
        checks="strict" routes product entries to the guarded CuTe immediate step, which
        follows #25's arithmetic and honours status/poison. Pure host policy.
        """
        if self.cute_config is None:
            raise ValueError("decode_route requires provider='cute'")
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a non-negative int")
        from .cute import decode_route

        route = decode_route(
            self._cute_device_name,
            batch_size,
            mode=self.mode,
            config=None if self._cute_device_name is not None else self.cute_config,
            geometry=(self.geometry.k_heads, self.geometry.v_heads),
            window=self.window,
        )
        return (
            "immediate"
            if route == "product" and (self.checks == "strict" or not self._ringless)
            else route
        )

    def workspace_spec(self, phase: str, *, batch_size: int):
        """§8 borrowed output/pending regions and provider-specific arithmetic scratch.

        Status is invocation-lifetime, shared across layers. Pending lives through
        resolution. Output lives through its consumer. Status/control rings are sized separately by graph_buffer_spec.
        Trusted CuTe commit/bridge/export/flush borrow arrivals: int32[max(1,B)],
        4*max(1,B) payload bytes, 256-byte aligned. The engine zeros it once;
        each active row's last CTA resets it. Sequential phases may share it,
        including graph replays; concurrent invocations need separate storage.
        The unfused A/B launcher uses the same workspace contract.
        """
        if phase not in _PHASES or type(batch_size) is not int or batch_size < 0:
            raise ValueError("unsupported phase or batch size")
        g, b, q = self.geometry, batch_size, self.verify_length
        entries = []
        if phase in ("verify", "commit", "reject"):
            entries += [
                ("pending.raw", (b, q, g.conv_dim + 2 * g.v_heads), torch.bfloat16, None),
                ("pending.metadata", (b, 8), torch.int64, 0),
            ]
        if phase in ("verify", "step", "bridge_step"):
            shape = (b, q, g.value_width) if phase == "verify" else (b, g.value_width)
            entries.append(("out", shape, torch.bfloat16, None))
        if phase == "step" and self._step_config is not None and self.decode_route(b) != "product":
            if self.cute_config.snapshot_controls:
                entries.append(("controls", (2 * b + 1,), torch.int64, None))
            else:
                entries.append(
                    ("arrivals", (max(1, b) if self.cute_config.trusted else 1,), torch.int32, 0)
                )
            if self.cute_config.sections:
                entries.append(("sections", (b * g.v_heads + 1, 7), torch.int64, None))
        if self._phase_arrivals(phase):
            entries.append(("arrivals", (max(1, b),), torch.int32, 0))
        return regions(entries)

    def create_state(self, *, buffers, workspace=None):
        """Bind six borrowed tensors from ``state_spec``; no allocation (§3).

        Metadata/epoch start at zero unless reset before use; payload may be undefined.
        No persistent workspace is needed; ``workspace`` must be empty or None.
        """
        if workspace:
            raise ValueError("replay uses no persistent workspace")
        state = ReplayState(**buffers)
        self._check_state(state)
        return state

    def create_pending(self, *, raw, metadata):
        """Bind Q-bounded raw records and eight int64 descriptor fields (§5)."""
        pending = ReplayPending(raw, metadata)
        self._check_pending(pending)
        return pending

    def _tensor(self, name, tensor, dtype, shape, *, alignment=8, contiguous=True):
        if name in _INTEGER_CONTROLS and isinstance(tensor, torch.Tensor):
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError(f"{name} must be int32 or int64")
            dtype = tensor.dtype
            alignment = min(alignment, tensor.element_size())
        external_pool = name in ("checkpoint", "conv_history", "canonical_state", "history")
        control = name in _INTEGER_CONTROLS or name == "snapshot_valid"
        check_tensor(
            name,
            tensor,
            dtype,
            shape,
            self.device,
            contiguous=contiguous and not external_pool and not control,
        )
        if external_pool:
            dense_rows(name, tensor)
            alignment = 16
        if control:
            if any(s < 0 for s in tensor.stride()):
                raise ValueError(f"{name} requires non-negative strides")
            if name == "positions" and tensor.ndim == 2 and tensor.stride(1) != 1:
                raise ValueError("verify positions require a dense Q axis")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise RuntimeError("replay does not support autograd through inputs or storage")
        if self.device.type == "cuda" and tensor.numel() and tensor.data_ptr() % alignment:
            raise ValueError(f"{name} requires {alignment}-byte base alignment")

    def _check_state(self, state):
        key = self._validation_key("state", _fields(state))
        if self._cached(key):
            return
        r, s = state.metadata.shape[0], state.snapshot_metadata.shape[0]
        for name, spec in self.state_spec(num_slots=r, num_snapshot_rows=s).items():
            x = getattr(state, name)
            dtype = spec.dtype
            if name == "checkpoint" and self.device.type == "cpu" and x.dtype == torch.float64:
                dtype = torch.float64
            shape = spec.shape
            # An immediate op may borrow a full replay allocation after materialization.
            if (
                name == "ring"
                and self._ringless
                and x.shape
                == (r, self.window + 3, self.geometry.conv_dim + 2 * self.geometry.v_heads)
            ):
                shape = tuple(x.shape)
            self._tensor(name, x, dtype, shape, alignment=256)
        xs = _tensors(state)
        if next(overlapping_pairs(xs, overlap=_overlap), None) is not None:
            raise ValueError("state regions must not overlap")
        self._remember(key)

    def _check_pending(self, pending, depth=None):
        key = self._validation_key("pending", _fields(pending), depth)
        if self._cached(key):
            return
        if pending.raw.ndim != (3 if depth is None else 4):
            raise ValueError("pending must match the status_slot form")
        b, g = pending.raw.shape[0 if depth is None else 1], self.geometry
        prefix = () if depth is None else (depth,)
        self._tensor(
            "pending.raw",
            pending.raw,
            torch.bfloat16,
            prefix + (b, self.verify_length, g.conv_dim + 2 * g.v_heads),
            alignment=2,
            contiguous=depth is None,
        )
        # G1 packs [D,B,...] densely: a slot view has natural alignment, while
        # the owning allocation/region has the 256-byte base alignment.
        self._tensor(
            "pending.metadata",
            pending.metadata,
            torch.int64,
            prefix + (b, 8),
            alignment=8,
            contiguous=depth is None,
        )
        if depth is not None:
            for name, tensor in (
                ("pending.raw", pending.raw),
                ("pending.metadata", pending.metadata),
            ):
                self._check_slot_rows(name, tensor)
        if self.device.type == "cuda" and any(
            t.untyped_storage().data_ptr() % 256 for t in (pending.raw, pending.metadata)
        ):
            raise ValueError("pending backing allocations require 256-byte alignment")
        if _overlap(pending.raw, pending.metadata):
            raise ValueError("pending regions must not overlap")
        self._remember(key)

    def _validation_key(self, kind, named, *extra):
        if not self._host_trusted:
            return None
        return kind, _identity(named), torch.is_grad_enabled(), extra

    def _cached(self, key):
        if key is None or key not in self._validated:
            return False
        self._validated.move_to_end(key)
        return True

    def _remember(self, key):
        if key is not None:
            self._validated[key] = None
            self._validated.move_to_end(key)
            if len(self._validated) > 4096:
                self._validated.popitem(last=False)

    def _check_disjoint(self, mode, left, right, message):
        if not left or not right:
            return
        tensors = left + right
        key = self._validation_key(mode, list(enumerate(tensors)), len(left))
        if not self._cached(key):
            pairs = overlapping_pairs(
                tensors, overlap=_overlap, exclude=lambda i, j: not i < len(left) <= j
            )
            if next(pairs, None) is not None:
                raise ValueError(message)
            self._remember(key)

    @staticmethod
    def _named_tensors(state, status, weights, kw, status_optional, pending):
        named = (
            []
            if status_optional
            else [
                ("status." + name, getattr(status, name))
                for name in ("records", "validity", "sequence", "poison")
            ]
        )
        if state is not None:
            named += [(f.name, getattr(state, f.name)) for f in fields(state)]
        if pending is not None:
            named += [("pending." + f.name, getattr(pending, f.name)) for f in fields(pending)]
        for i, block in enumerate(kw.get("pending_blocks", ())):
            named += [
                (f"pending_blocks.{i}.{f.name}", getattr(block, f.name)) for f in fields(block)
            ]
        named += [(k, v) for k, v in kw.items() if isinstance(v, torch.Tensor)]
        named += [(f"weight{i}", w) for i, w in enumerate(weights or ())]
        return named

    def _validate_overlaps(self, mode, state, status, weights, kw, status_optional, pending=None):
        named = self._named_tensors(state, status, weights, kw, status_optional, pending)
        key = self._validation_key(
            "overlaps",
            named,
            mode,
            status_optional,
            kw.get("advance_positions", False),
            (status.layer, status.batch_size) if not status_optional else None,
        )
        if not self._cached(key):
            self._check_overlaps(mode, status, weights, kw, named)
            self._remember(key)
        return named

    @staticmethod
    def _check_overlaps(mode, status, weights, kw, named):
        # Handoff permits exact identity; the device also checks identity row mapping.
        readonly = {
            name
            for name, value in kw.items()
            if isinstance(value, torch.Tensor)
            and name not in ("out", "snapshot_valid", "sections", "arrivals", "controls")
        }
        readonly.discard("status_slot")
        readonly.update(f"weight{i}" for i in range(len(weights or ())))
        if kw.get("advance_positions"):
            readonly.discard("positions")
        first = None
        for i, j in overlapping_pairs(
            [t for _, t in named],
            overlap=_overlap,
            exclude=lambda i, j: named[i][0] in readonly and named[j][0] in readonly,
        ):
            (an, a), (bn, other) = named[i], named[j]
            if (an, bn) == ("status.validity", "snapshot_valid") and (
                other.data_ptr() == status.valid.data_ptr()
                and other.shape == status.valid.shape
                and other.stride() == status.valid.stride()
            ):
                continue
            allowed = mode == "handoff" and (an, bn) in (
                ("checkpoint", "canonical_state"),
                ("conv_history", "history"),
            )
            if not (allowed and a.data_ptr() == other.data_ptr() and a.stride() == other.stride()):
                first = min(first, (i, j)) if first is not None else (i, j)
        if first is not None:
            raise ValueError(f"overlapping buffers: {named[first[0]][0]}, {named[first[1]][0]}")

    def _fast_phase(self, mode, weights, kw):
        """Reuse an eager validated plan with the identical launcher and view ABI."""
        # Frozen weights precede eligibility, descriptor reads and all launches.
        if any(isinstance(w, torch.Tensor) and w.requires_grad for w in weights):
            require_frozen(
                type(self).__name__,
                dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
            )
        if not (
            HOST_FAST_PATH
            and self._host_trusted
            and self.checks == "trusted"
            and self.provider[mode] == "cute"
            and self.device.type == "cuda"
            and kw["status"] is None
            and kw.get("status_slot") is None
            and not torch.is_grad_enabled()
        ):
            return False
        from .cute import _phase_launch

        state, pending = kw["state"], kw["pending"]
        if not isinstance(state, ReplayState) or not isinstance(pending, ReplayPending):
            return False
        per_call = _phase_launch.per_call(kw)
        if not all(isinstance(t, torch.Tensor) for t in (*weights, *per_call.values())):
            return False
        slots = kw["slot_indices"]
        if not isinstance(slots, torch.Tensor) or slots.ndim != 1:
            return False
        # Backing allocation alignment is independent of the view alignment class.
        if any(t.untyped_storage().data_ptr() % 256 for t in (pending.raw, pending.metadata)):
            return False
        pointers = tuple(t.data_ptr() for t in per_call.values())
        key = (
            mode,
            id(state),
            *(id(w) for w in weights),
            tuple(
                (name, descriptor(t), pointer % 256)
                for (name, t), pointer in zip(per_call.items(), pointers, strict=True)
            ),
            _phase_launch.switches(),
            id(self.cute_config),
            self.commit_mode_for(slots.shape[0]) if mode == "commit" else None,
        )
        entry = self._plans.get(key)
        if entry is not None:
            if not entry.unchanged():
                del self._plans[key]
                return False
            if entry.addresses_valid(pointers):
                self._plans.move_to_end(key)
                # The recipe repeats the slow path's exact view expressions; both
                # paths call the same compiled function with the same tensor ABI.
                entry.plan.launch(per_call)
                return True
        capturing = torch.cuda.is_current_stream_capturing()
        prepared = self._run(mode, weights=weights, **kw)
        if not capturing and prepared is not None:
            self._plans[key] = HostPlan(
                state, weights, prepared, per_call, pointers, self.cute_config
            )
            self._plans.move_to_end(key)
            if len(self._plans) > 4096:
                self._plans.popitem(last=False)
        return True

    def _run(self, mode, *, state=None, status, weights=None, **kw):
        # This MUST precede all other checks and every provider/device call.
        if weights is not None:
            names = ("conv_weight", "A_log", "dt_bias", "norm_weight")
            require_frozen(type(self).__name__, dict(zip(names, weights, strict=True)))
        status_optional = self.checks == "trusted" and self.provider[mode] != "torch_reference"
        status_slot = kw.get("status_slot")
        slotted = status_slot is not None
        depth = None
        if slotted:
            if mode not in ("verify", "commit", "reject"):
                raise ValueError("status_slot requires verify, commit or reject")
            self._tensor("status_slot", status_slot, torch.int32, (1,), alignment=4)
            if status_slot.data_ptr() % 4:
                raise ValueError("status_slot requires 4-byte alignment")
            pending = kw.get("pending")
            if pending is None or pending.raw.ndim != 4 or pending.raw.shape[0] < 1:
                raise ValueError("status_slot requires stacked pending [D,B,Q,W]")
            depth = pending.raw.shape[0]
        if not status_optional or (slotted and status is not None):
            self._check_status(status, depth=depth)
        elif isinstance(status, ReplayStatus) and status.slotted:
            raise ValueError("slotted status requires status_slot")
        if state is not None:
            self._check_state(state)
        g, q = self.geometry, self.verify_length
        if weights is not None:
            check_weights(g, self.device, *weights)
            for w in weights:
                if w.data_ptr() % 16:
                    raise ValueError("weights require 16-byte alignment")
        slots, pairs = kw.get("slot_indices"), kw.get("pairs")
        b = slots.shape[0] if slots is not None else pairs.shape[0] if pairs is not None else 0
        if slots is not None:
            self._tensor("slot_indices", slots, torch.int32, (b,), alignment=4)
        if pairs is not None:
            self._tensor("pairs", pairs, torch.int32, (b, 2), alignment=4)
        pending = kw.get("pending")
        if pending is not None:
            self._check_pending(pending, depth=depth)
            if mode != "export_snapshot" and pending.raw.shape[int(slotted)] != b:
                raise ValueError("pending batch must match slots")
        for name, dtype in (
            ("positions", torch.int64),
            ("next_positions", torch.int64),
            ("accepted_lengths", torch.int32),
            ("snapshot_rows", torch.int32),
            ("snapshot_offsets", torch.int64),
            ("offsets", torch.int64),
            ("snapshot_valid", torch.bool),
        ):
            x = kw.get(name)
            if x is not None:
                n = (
                    pending.raw.shape[0]
                    if name == "accepted_lengths" and mode == "export_snapshot"
                    else b
                )
                shape = (b, q) if name == "positions" and mode == "verify" else (n,)
                if name == "snapshot_valid" and slotted:
                    shape = (depth, b)
                self._tensor(
                    name,
                    x,
                    dtype,
                    shape,
                    alignment=x.element_size(),
                    contiguous=not (slotted and name == "snapshot_valid"),
                )
                if name == "snapshot_valid" and slotted:
                    if x.stride(1) < 1 or x.stride(0) < max(0, (b - 1) * x.stride(1) + 1):
                        raise ValueError("snapshot_valid requires nonoverlapping slots and rows")
        x = kw.get("projected")
        if x is not None:
            shape = (b, q, g.proj_width) if mode == "verify" else (b, g.proj_width)
            self._tensor("projected", x, torch.bfloat16, shape, alignment=2, contiguous=False)
            feature_rows("projected", x)
        out = kw.get("out")
        if out is not None:
            shape = (b, q, g.value_width) if mode == "verify" else (b, g.value_width)
            self._tensor("out", out, torch.bfloat16, shape, alignment=2, contiguous=False)
            feature_rows("out", out)
        if mode == "step" and self._step_config is not None and not kw.get("product_fallback"):
            if self.cute_config.snapshot_controls:
                if kw.get("controls") is None:
                    raise ValueError("CuTe snapshot mode requires workspace['controls']")
                self._tensor("controls", kw["controls"], torch.int64, (2 * b + 1,), alignment=256)
            else:
                if kw.get("arrivals") is None:
                    raise ValueError("CuTe step requires borrowed workspace['arrivals']")
                self._tensor(
                    "arrivals",
                    kw["arrivals"],
                    torch.int32,
                    (max(1, b) if self.cute_config.trusted else 1,),
                    alignment=256,
                )
            if self.cute_config.sections:
                if kw.get("sections") is None:
                    raise ValueError("instrumented CuTe step requires workspace['sections']")
                self._tensor(
                    "sections", kw["sections"], torch.int64, (b * g.v_heads + 1, 7), alignment=256
                )
        if self._phase_arrivals(mode):
            if kw.get("arrivals") is None:
                raise ValueError("trusted CuTe phase requires borrowed workspace['arrivals']")
            # Scalar atomics only: rows of graph_data_spec's [D, max(1,B)] buffer
            # are 4-byte aligned.
            self._tensor(
                "arrivals",
                kw["arrivals"],
                torch.int32,
                (depth, max(1, b)) if slotted else (max(1, b),),
                alignment=4,
                contiguous=not slotted,
            )
            if slotted:
                self._check_slot_rows("arrivals", kw["arrivals"])
        elif kw.get("arrivals") is not None and not slotted and kw["arrivals"].ndim != 1:
            raise ValueError("stacked arrivals require status_slot")
        elif kw.get("arrivals") is not None and slotted:
            self._tensor(
                "arrivals",
                kw["arrivals"],
                torch.int32,
                (depth, max(1, b)),
                alignment=4,
                contiguous=False,
            )
            self._check_slot_rows("arrivals", kw["arrivals"])
        if mode == "handoff":
            self._tensor(
                "canonical_state",
                kw["canonical_state"],
                state.checkpoint.dtype,
                (b, g.v_heads, 128, 128),
                alignment=16,
            )
            self._tensor("history", kw["history"], torch.bfloat16, (b, 3, g.conv_dim), alignment=16)
        named = self._validate_overlaps(
            mode,
            state,
            status,
            weights,
            kw,
            status_optional and not (slotted and status is not None),
            pending,
        )
        # Match the provider's alignment classes, including the stronger original
        # dense-pool assumptions when available. Phase arrivals bind at 4 bytes.
        signature = (
            mode,
            self.checks,
            self.commit_mode_for(b) if mode == "commit" else None,
            tuple(
                (
                    name,
                    tuple(t.shape),
                    tuple(t.stride()),
                    t.dtype,
                    t.data_ptr()
                    % (
                        4
                        if name == "arrivals"
                        else 256
                        if name in ("checkpoint", "conv_history")
                        else 16
                    ),
                )
                for name, t in named
            ),
            kw.get("weight_revision"),
            kw.get("complete", False),
            kw.get("advance_positions", False),
            mode == "handoff"
            and (
                kw["canonical_state"].data_ptr() == state.checkpoint.data_ptr()
                or kw["history"].data_ptr() == state.conv_history.data_ptr()
            ),
        )
        capturing = self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if capturing and mode not in self.capabilities["capture"]:
            raise RuntimeError(f"{mode} must run outside CUDA graphs")
        if capturing and signature not in self._warm:
            raise RuntimeError("warm this replay call shape eagerly before capture")
        result = self._backend(self.provider[mode]).run(
            self, mode, state=state, status=status, weights=weights, **kw
        )
        if not capturing:
            self._warm.add(signature)
        return result

    def _backend(self, provider):
        if provider not in self._backends:
            if provider == "triton" and torch.cuda.get_device_capability(self.device) < (8, 9):
                raise RuntimeError(
                    "the portable Triton replay requires a CUDA device with cc >= 8.9"
                )
            self._backends[provider] = import_module(f".{provider}", __package__)
        return self._backends[provider]

    def graph_buffer_spec(self, *, batch_size, depth=2, num_layers=1):
        """Shared fixed controls, exact §3 payload and separately aligned regions."""
        return graph_control_sizes(
            batch_size=batch_size,
            verify_length=self.verify_length,
            depth=depth,
            num_layers=num_layers,
        )

    def graph_data_spec(self, *, batch_size, depth=2, num_layers=1):
        """Additional fixed capture payloads, beyond graph_buffer_spec and state_spec.

        Includes both position-independent input/output shapes. Sequential graph
        variants may share these addresses. Pending is retained for each layer/slot;
        register-only providers report no additional global arithmetic workspace.
        phase_arrivals is int32[D, max(1,B)] (4*D*max(1,B) bytes), initialized once
        to zero. Row d belongs to in-flight round slot d: bind phase_arrivals[d] as
        workspace['arrivals'] for that round's trusted commit/bridge/export/flush.
        Each fused launch's last arriver resets its counters before the grid ends,
        so phases, layers and graph variants issued in order on one stream may also
        share a row; rows exist so concurrent streams never share one. It is
        distinct from decode step scratch. For commit(status_slot=...), pass
        the whole [D,max(1,B)] tensor instead of a host-int row view.
        """
        self.graph_buffer_spec(batch_size=batch_size, depth=depth, num_layers=num_layers)
        b, d, j, q, g = batch_size, depth, num_layers, self.verify_length, self.geometry
        entries = [
            ("projected", (j, d, b, g.proj_width), torch.bfloat16, None),
            ("verify_projected", (j, d, b, q, g.proj_width), torch.bfloat16, None),
            ("out", (j, d, b, g.value_width), torch.bfloat16, None),
            ("verify_out", (j, d, b, q, g.value_width), torch.bfloat16, None),
            ("pending.raw", (j, d, b, q, g.conv_dim + 2 * g.v_heads), torch.bfloat16, None),
        ]
        if self._step_config is not None:
            if self.cute_config.snapshot_controls:
                entries.append(("controls", (2 * b + 1,), torch.int64, None))
            else:
                entries.append(
                    ("arrivals", (max(1, b) if self.cute_config.trusted else 1,), torch.int32, 0)
                )
            if self.cute_config.sections:
                entries.append(("sections", (b * g.v_heads + 1, 7), torch.int64, None))
        if any(self._phase_arrivals(phase) for phase in _PHASES):
            entries.append(("phase_arrivals", (d, max(1, b)), torch.int32, 0))
        return regions(entries)

    def bind_graph_buffers(self, *, buffers):
        """Bind engine allocations without copying, initializing or allocating tensors.

        One binding is shared by all graph variants. Use prefix views for smaller B.
        The engine initializes the descriptors' initial values before first use.
        """
        d, b = buffers["positions"].shape
        j = buffers["records"].shape[0]
        specs = self.graph_buffer_spec(batch_size=b, depth=d, num_layers=j)
        if buffers.keys() != specs.keys():
            raise ValueError("graph buffers must match graph_buffer_spec")
        for name, spec in specs.items():
            self._tensor(name, buffers[name], spec.dtype, spec.shape, alignment=256)
        values = list(buffers.values())
        if next(overlapping_pairs(values, overlap=_overlap), None) is not None:
            raise ValueError("graph buffer regions must not overlap")
        return ReplayGraphBuffers(MappingProxyType(dict(buffers)))

    @staticmethod
    def _check_slot_rows(name, tensor):
        """Dense per-slot payload, allowing unused capacity between slots."""
        width = 1
        for n, stride in reversed(list(zip(tensor.shape[1:], tensor.stride()[1:], strict=True))):
            if stride != width:
                raise ValueError(f"{name} requires dense per-slot rows")
            width *= n
        if tensor.stride(0) < width:
            raise ValueError(f"{name} requires nonoverlapping slots")

    def _check_status(self, status, depth=None):
        if not isinstance(status, ReplayStatus):
            raise TypeError("status must be a bound ReplayStatus, not a scalar error word")
        if status.slotted != (depth is not None):
            raise ValueError("status must match the status_slot form")
        key = self._validation_key(
            "status",
            [
                (name, getattr(status, name))
                for name in ("records", "validity", "sequence", "poison")
            ],
            status.layer,
            status.batch_size,
            depth,
        )
        if self._cached(key):
            return
        if depth is not None:
            if status.records.ndim != 3 or status.validity.ndim != 3:
                raise ValueError("slotted status requires records/validity [J,D,...]")
            j = status.records.shape[0]
            if not j or not 0 <= status.layer < j:
                raise ValueError("invalid slotted status")
            b = status.validity.shape[-1]
            if b < 1 or (status.batch_size is not None and not 0 < status.batch_size <= b):
                raise ValueError("invalid status row count")
            self._tensor(
                "status.records",
                status.records,
                torch.uint8,
                (j, depth, 16),
                alignment=16,
                contiguous=False,
            )
            self._tensor(
                "status.validity",
                status.validity,
                torch.bool,
                (j, depth, b),
                alignment=1,
                contiguous=False,
            )
            for name, t in (("records", status.records), ("validity", status.validity)):
                if (
                    t.stride(2) != 1
                    or t.stride(1) < t.shape[2]
                    or t.stride(0) < (depth - 1) * t.stride(1) + t.shape[2]
                ):
                    raise ValueError(f"invalid status {name} strides")
            if status.records.stride(1) % 16 or status.records.stride(0) % 16:
                raise ValueError("invalid status record alignment")
            self._tensor("status.sequence", status.sequence, torch.int64, (depth,))
            self._tensor("status.poison", status.poison, torch.int32, (1,), alignment=4)
            self._remember(key)
            return
        j = status.records.shape[0]
        if not j or not 0 <= status.layer < j or not status.validity.shape[1]:
            raise ValueError("invalid status layer or row count")
        self._tensor(
            "status.records", status.records, torch.uint8, (j, 16), alignment=16, contiguous=False
        )
        self._tensor(
            "status.validity",
            status.validity,
            torch.bool,
            (j, status.validity.shape[1]),
            alignment=1,
            contiguous=False,
        )
        self._tensor("status.sequence", status.sequence, torch.int64, (1,))
        self._tensor("status.poison", status.poison, torch.int32, (1,), alignment=4)
        if (
            status.records.stride(-1) != 1
            or status.records.stride(0) < 16
            or status.records.stride(0) % 16
            or status.validity.stride(-1) != 1
            or status.validity.stride(0) < status.validity.shape[1]
        ):
            raise ValueError("invalid status record strides")
        self._remember(key)

    def _lifecycle(self, mode, status, expected=0):
        self._check_status(status)
        if self.device.type == "cpu":
            from .torch_reference._status import lifecycle

            lifecycle(status, mode, expected)
        else:
            from .triton._status import lifecycle

            lifecycle[(1,)](
                status.records,
                status.validity,
                status.sequence,
                status.poison,
                status.records.stride(0),
                status.validity.stride(0),
                status.records.shape[0],
                status.validity.shape[1],
                mode,
                expected,
            )

    def begin(self, *, status, sequence=None, layer_weights=()):
        """Check the free modulo slot on device; never overwrite unsettled records.

        Preflight ALL participating layer weights here before any invocation launch.
        Graph replay permission must also be revoked by the engine on weight changes.
        ``sequence`` may name only the handle's fixed input (no capture rebinding).
        """
        for weights in layer_weights:
            require_frozen(
                type(self).__name__,
                dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
            )
        self._check_status(status)
        if sequence is not None and (
            sequence.data_ptr() != status.sequence.data_ptr()
            or sequence.shape != status.sequence.shape
            or sequence.dtype != torch.int64
        ):
            raise ValueError("sequence must be the bound fixed-address int64 input")
        self._lifecycle("begin", status)

    def end(self, *, status):
        """Finalize every layer, including failed records, on the execution stream."""
        self._lifecycle("end", status)

    @staticmethod
    def read_status_copy(records, validity, *, expected_sequence):
        return read_status_copy(records, validity, expected_sequence=expected_sequence)

    def settle(self, *, status, result):
        """Enqueue clear of only this settled slot; never clear poison or wait.

        Caller has joined existing completion, all ranks and KV in submission order.
        Failed but matching completed records may be reclaimed during drain. A stale
        result cannot clear a new record: the clear checks the sequence on device.
        """
        if not result.complete or len(result.codes) != status.records.shape[0]:
            raise ValueError("cannot settle a zero, stale or incomplete status record")
        self._lifecycle("clear", status, result.expected_sequence)
        return result.ok

    def recover_reset(self, *, states, status, pending_blocks=(), weight_revision=0, drained=False):
        """Quiescent recovery: invalidate all layers/snapshots/pending, then clear poison.

        ``drained=True`` asserts that the engine stopped admission/publication, drained
        ALL dependent streams/ranks and settled every outstanding result. This method
        introduces no synchronization. Start a separate recovery invocation afterwards
        and reopen admission only after all ranks complete successfully.
        """
        self._plans.clear()
        if not drained:
            raise ValueError("recovery requires engine drain and settlement")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("recovery must run outside capture")
        self._check_status(status)
        states, pending_blocks = tuple(states), tuple(pending_blocks)
        if len(states) != status.records.shape[0]:
            raise ValueError("recovery must invalidate every participating layer")
        if type(weight_revision) is not int or not 0 <= weight_revision < 2**63:
            raise ValueError("weight_revision must be a nonnegative int64")
        for state in states:
            self._check_state(state)
        for p in pending_blocks:
            # Q8 and Q16 transactions may share the pool; recovery clears both.
            self._tensor("pending.metadata", p.metadata, torch.int64, (p.raw.shape[0], 8))
        if self.device.type == "cpu":
            for state in states:
                state.pool_epoch.add_(1)
                state.metadata[:, :5] = 0
                state.metadata[:, 5] += 1
                state.metadata[:, 6] = weight_revision
                state.metadata[:, 7] = state.pool_epoch[0]
                state.snapshot_metadata.zero_()
            for p in pending_blocks:
                p.metadata.zero_()
            status.poison.zero_()
        else:
            import triton

            from .triton import _status as control

            for state in states:
                r, ns = state.metadata.shape[0], state.snapshot_metadata.shape[0]
                control.invalidate[(max(1, triton.cdiv(max(r * 8, ns * 4), 256)),)](
                    state.metadata,
                    state.snapshot_metadata,
                    state.metadata,
                    state.pool_epoch,
                    r,
                    ns,
                    0,
                    weight_revision,
                    256,
                )
                control.recovery_finish[(1,)](state.pool_epoch)
            # No state writes can be admitted between invalidation and this clear.
            for p in pending_blocks:
                p.metadata.zero_()
            control.clear_poison[(1,)](status.poison)

    def step(
        self,
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        slot_indices,
        positions,
        state,
        out,
        status=None,
        workspace=None,
    ):
        """Decode [B,P] at int32/int64 next positions; flush exactly at L (§4).

        Product routing (trusted only) borrows out, runs #25 and advances each active
        slot's replay metadata on a parallel graph branch. The engine owns positions.
        Trusted CuTe needs no begin/end and ignores a supplied status. Strict calls
        use explicit begin/end. CuTe borrows workspace_spec("step") scratch.
        Default arrivals starts at zero; snapshot_controls uses controls scratch.
        Optional sections scratch is only required by the instrumented specialization.
        Scratch must be private to concurrent invocations; serial graphs may reuse it.
        """
        require_frozen(
            type(self).__name__,
            dict(
                zip(
                    ("conv_weight", "A_log", "dt_bias", "norm_weight"),
                    (conv_weight, A_log, dt_bias, norm_weight),
                    strict=True,
                )
            ),
        )
        product = (
            self._step_config is not None
            and self.decode_route(projected.shape[0]) == "product"
            and (state is None or state.ring.shape[1] == 0)
        )
        # #25 accepts the extended dtypes and disabled-grad inputs, but still needs
        # dense pools/output and aligned projected rows. Positions are not read:
        # the metadata kernel increments the committed int64 position directly.
        product_fallback = product and (
            not state.checkpoint.is_contiguous()
            or not state.conv_history.is_contiguous()
            or not out.is_contiguous()
            or out.data_ptr() % 16 != 0
            or slot_indices.stride(0) != 1
            or projected.data_ptr() % 16 != 0
            or projected.stride(0) * projected.element_size() % 16 != 0
            or positions.dtype not in (torch.int32, torch.int64)
        )
        if product and not product_fallback:
            self._check_state(state)
            self._tensor(
                "out",
                out,
                torch.bfloat16,
                (projected.shape[0], self.geometry.value_width),
                alignment=16,
            )
            b = projected.shape[0]
            self._tensor("slot_indices", slot_indices, torch.int32, (b,), alignment=4)
            self._validate_overlaps(
                "step",
                state,
                None,
                (conv_weight, A_log, dt_bias, norm_weight),
                {
                    "projected": projected,
                    "out": out,
                    "positions": positions,
                    "slot_indices": slot_indices,
                },
                True,
            )
            if self._product_decode is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm this replay call shape eagerly before capture")
                from ..inference import GatedDeltaNetDecode

                self._product_decode = GatedDeltaNetDecode(
                    num_k_heads=self.geometry.k_heads,
                    num_v_heads=self.geometry.v_heads,
                    device=self.device,
                )
            from .cute import advance_positions as advance

            # Metadata advance is independent of #25 (which never reads metadata), so it
            # runs on a branch forked before #25 and joins before the next consumer of
            # this stream. It is enqueued only after #25 accepted its arguments.
            current = torch.cuda.current_stream(self.device)
            if self._side_stream is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm this replay call shape eagerly before capture")
                self._side_stream = torch.cuda.Stream(self.device)
            fork = current.record_event()
            self._product_decode(
                projected,
                conv_weight,
                A_log,
                dt_bias,
                norm_weight,
                slot_indices,
                state.checkpoint,
                state.conv_history,
                out=out,
                _replay=True,
            )
            self._side_stream.wait_event(fork)
            with torch.cuda.stream(self._side_stream):
                advance(self, state, slot_indices)
            current.wait_stream(self._side_stream)
            return
        self._run(
            "step",
            weights=(conv_weight, A_log, dt_bias, norm_weight),
            projected=projected,
            slot_indices=slot_indices,
            positions=positions,
            state=state,
            out=out,
            status=status,
            sections=None if workspace is None else workspace.get("sections"),
            arrivals=None if workspace is None else workspace.get("arrivals"),
            controls=None if workspace is None else workspace.get("controls"),
            complete=False,
            advance_positions=False,
            product_fallback=product_fallback,
        )

    def bridge_step(
        self,
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        slot_indices,
        positions,
        state,
        out,
        status=None,
        workspace=None,
    ):
        """Borrowed canonical bridge: validate → materialize → compute → handoff.

        The canonical pool itself is the working state. Trusted CuTe borrows
        per-row arrivals (workspace_spec). Strict mode gates on the same poison/code.
        Positions advance on device. No allocating legacy __call__ is used.
        """
        weights = (conv_weight, A_log, dt_bias, norm_weight)
        require_frozen(
            type(self).__name__,
            dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
        )
        if workspace and not self._phase_arrivals("bridge_step"):
            raise ValueError("canonical bridge uses no extra workspace")
        self._run(
            "bridge_step",
            weights=weights,
            projected=projected,
            slot_indices=slot_indices,
            positions=positions,
            state=state,
            out=out,
            status=status,
            **({"arrivals": workspace.get("arrivals")} if workspace is not None else {}),
        )

    def verify(
        self,
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        slot_indices,
        positions,
        state,
        pending,
        out,
        status=None,
        status_slot=None,
    ):
        """Read-only live state; Q outputs and independent pending records (§5).

        status_slot is an int32[1] device selector for stacked status/pending.
        Invalid selectors write only shared poison in strict mode; trusted is a no-op.
        """
        weights = (conv_weight, A_log, dt_bias, norm_weight)
        kw = {
            "projected": projected,
            "slot_indices": slot_indices,
            "positions": positions,
            "state": state,
            "pending": pending,
            "out": out,
            "status": status,
            "status_slot": status_slot,
        }
        if not self._fast_phase("verify", weights, kw):
            self._run("verify", weights=weights, **kw)

    def commit(
        self,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        slot_indices,
        accepted_lengths,
        state,
        pending,
        status=None,
        status_slot=None,
        workspace=None,
        snapshot_rows=None,
        snapshot_offsets=None,
        snapshot_valid=None,
    ):
        """Accept m=int32/int64 [B] on device, optionally capture prefix k (§§5,7).

        Delayed mode appends exact raw rows while n+m < window and no capture
        occurs; otherwise it folds n+m into the checkpoint. Flush before reading
        the checkpoint directly; this op's replay-aware methods need no flush.

        k>m leaves snapshot payload untouched and returns snapshot_valid=False.
        Valid bits indicate captured payload only; engine publication awaits completion.
        """
        weights = (conv_weight, A_log, dt_bias, norm_weight)
        if any(isinstance(w, torch.Tensor) and w.requires_grad for w in weights):
            require_frozen(
                type(self).__name__,
                dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
            )
        if any(
            x is not None for x in (snapshot_rows, snapshot_offsets, snapshot_valid)
        ) and not all(x is not None for x in (snapshot_rows, snapshot_offsets, snapshot_valid)):
            raise ValueError("capture requires rows, offsets and valid buffers together")
        kw = dict(
            slot_indices=slot_indices,
            accepted_lengths=accepted_lengths,
            state=state,
            pending=pending,
            status=status,
            status_slot=status_slot,
            **({"arrivals": workspace.get("arrivals")} if workspace is not None else {}),
            snapshot_rows=snapshot_rows,
            snapshot_offsets=snapshot_offsets,
            snapshot_valid=snapshot_valid,
        )
        if not self._fast_phase("commit", weights, kw):
            self._run("commit", weights=weights, **kw)

    def reject(self, *, slot_indices, state, pending, status=None, status_slot=None):
        """Discard only the transaction; live checkpoint/ring/history unchanged (§5)."""
        self._run(
            "reject",
            slot_indices=slot_indices,
            state=state,
            pending=pending,
            status=status,
            status_slot=status_slot,
        )

    def flush(
        self,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        slot_indices,
        state,
        status=None,
        workspace=None,
    ):
        """Canonicalize buffered rows before direct checkpoint consumers (§4).

        Retains cursor rotation. Graph-capturable in trusted CuTe mode with
        borrowed arrivals; accepts -1 padded slots without host synchronization.
        """
        weights = (conv_weight, A_log, dt_bias, norm_weight)
        require_frozen(
            type(self).__name__,
            dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
        )
        if self._ringless and (state is None or state.ring.shape[1] == 0):
            return
        self._run(
            "flush",
            weights=weights,
            slot_indices=slot_indices,
            state=state,
            status=status,
            **({"arrivals": workspace.get("arrivals")} if workspace is not None else {}),
        )

    def export_snapshot(
        self,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        *,
        pairs,
        state,
        status=None,
        workspace=None,
        snapshot_valid,
        pending=None,
        accepted_lengths=None,
        offsets=None,
        arithmetic_batch=None,
    ):
        """Nonmutating batched live→snapshot materialization, outside graphs (§7).

        arithmetic_batch selects the CuTe recipe of the replayed graph's decode B;
        None uses the pair count. Portable providers accept and ignore it.
        Bitwise written-back equivalence requires all buffered tokens at that B.
        Optional pending is searched by source slot, permitting repeated sources.
        Accepted lengths are per pending row; offsets/valid are per pair.
        The engine must invalidate provisional exports after reject/failed completion.
        """
        weights = (conv_weight, A_log, dt_bias, norm_weight)
        require_frozen(
            type(self).__name__,
            dict(zip(("conv_weight", "A_log", "dt_bias", "norm_weight"), weights, strict=True)),
        )
        if arithmetic_batch is not None and (
            type(arithmetic_batch) is not int or arithmetic_batch <= 0
        ):
            raise ValueError("arithmetic_batch must be a positive integer or None")
        if (pending is None) != (accepted_lengths is None) or (pending is None) != (
            offsets is None
        ):
            raise ValueError(
                "pending export requires pending, accepted_lengths and offsets together"
            )
        self._run(
            "export_snapshot",
            weights=weights,
            pairs=pairs,
            state=state,
            status=status,
            **({"arrivals": workspace.get("arrivals")} if workspace is not None else {}),
            snapshot_valid=snapshot_valid,
            pending=pending,
            accepted_lengths=accepted_lengths,
            offsets=offsets,
            arithmetic_batch=arithmetic_batch,
        )

    def import_snapshot(self, *, pairs, state, status):
        """Batched snapshot→live copy, revision/epoch validation and reseeding (§7)."""
        self._run("import_snapshot", pairs=pairs, state=state, status=status)

    def initialize(self, *, slot_indices, next_positions, state, status):
        """Fresh zero state/history, increment generation, READY at supplied boundary (§4)."""
        self._run(
            "initialize",
            slot_indices=slot_indices,
            next_positions=next_positions,
            state=state,
            status=status,
        )

    def handoff(self, *, canonical_state, history, next_positions, slot_indices, state, status):
        """Import FP32 prefill results and three BF16 pre-conv rows at int64 p (§7).

        Input rows correspond to batch rows. Exact in-place prefix views require
        slot_indices[b]==b for active rows. Caller zero-left-pads short prompt history.
        """
        self._run(
            "handoff",
            canonical_state=canonical_state,
            history=history,
            next_positions=next_positions,
            slot_indices=slot_indices,
            state=state,
            status=status,
        )

    def layer_table_spec(self, *, num_layers):
        """Engine-owned pointer table; bind once outside capture."""
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        return regions([("layer_table", (num_layers, 8), torch.int64, None)])

    def bind_layers(self, *, states, table, canonical_states=None, histories=None):
        """Validate all layers once and upload their pointers in one copy.

        Exact prefix identities with a layer's own checkpoint/history are allowed.
        All other regions, including the table, must be disjoint. Keep the binding
        and its tensors alive and unchanged until every captured use has finished.
        """
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind_layers must run outside capture")
        states = tuple(states)
        j = len(states)
        spec = self.layer_table_spec(num_layers=j)["layer_table"]
        self._tensor("layer_table", table, spec.dtype, spec.shape, alignment=16)
        canonical_states = (
            tuple(s.checkpoint for s in states)
            if canonical_states is None
            else tuple(canonical_states)
        )
        histories = tuple(s.conv_history for s in states) if histories is None else tuple(histories)
        if len(canonical_states) != j or len(histories) != j:
            raise ValueError("one canonical state and history required per layer")
        b = canonical_states[0].shape[0]
        r, ns = states[0].metadata.shape[0], states[0].snapshot_metadata.shape[0]
        intervals, rows = [], []

        def region(tensor):
            if tensor.numel():
                start = tensor.data_ptr()
                if start % 16:
                    raise ValueError("layer regions require 16-byte alignment")
                intervals.append(tensor)

        region(table)
        copies = False
        for state, canonical, history in zip(states, canonical_states, histories, strict=True):
            self._check_state(state)
            if (state.metadata.shape[0], state.snapshot_metadata.shape[0]) != (r, ns):
                raise ValueError("layers must share slot and snapshot row counts")
            self._tensor(
                "canonical_state",
                canonical,
                state.checkpoint.dtype,
                (b, self.geometry.v_heads, 128, 128),
                alignment=16,
            )
            self._tensor(
                "history", history, torch.bfloat16, (b, 3, self.geometry.conv_dim), alignment=16
            )
            for tensor in _tensors(state):
                region(tensor)
            for source, target in ((canonical, state.checkpoint), (history, state.conv_history)):
                identity = (
                    source.data_ptr() == target.data_ptr() and source.stride() == target.stride()
                )
                if not identity:
                    region(source)
                elif source.numel() > target.numel():
                    raise ValueError("identity input exceeds its layer region")
            inplace = (
                canonical.data_ptr() == state.checkpoint.data_ptr()
                or history.data_ptr() == state.conv_history.data_ptr()
            )
            copies |= canonical.data_ptr() != state.checkpoint.data_ptr()
            rows.append(
                [
                    state.checkpoint.data_ptr(),
                    state.conv_history.data_ptr(),
                    state.ring.data_ptr(),
                    state.metadata.data_ptr(),
                    state.pool_epoch.data_ptr(),
                    canonical.data_ptr(),
                    history.data_ptr(),
                    int(inplace) | (int(state.ring.shape[1] == 0) << 1),
                ]
            )
        # Fold exact own-layer identities, then compare occupied bytes, including holes.
        if next(overlapping_pairs(intervals, overlap=_overlap), None) is not None:
            raise ValueError("overlapping layer regions")
        table.copy_(torch.tensor(rows, dtype=torch.int64, device="cpu"))
        return ReplayLayers(
            self,
            (self.device, self.geometry, self.window),
            states,
            canonical_states,
            histories,
            table,
            table.view(-1),
            b,
            r,
            ns,
            copies,
        )

    def handoff_layers(self, layers, *, next_positions, slot_indices, status=None):
        """Handoff every bound layer; constant host validation on the CuTe path.

        Status selects strict semantics even on a trusted op. Without status the
        trusted contract requires valid, uniquely mapped rows. Dynamic controls
        must be disjoint from bound storage; their addresses may stay fixed in graphs.
        """
        if (
            not isinstance(layers, ReplayLayers)
            or layers.owner is not self
            or layers.configuration != (self.device, self.geometry, self.window)
        ):
            raise ValueError("foreign or incompatible layer binding")
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (
                *layers.canonical_states,
                *layers.histories,
                *(t for state in layers.states for t in _tensors(state)),
            )
        ):
            raise RuntimeError("replay does not support autograd through inputs or storage")
        b, j = layers.batch_size, len(layers.states)
        self._tensor("slot_indices", slot_indices, torch.int32, (b,), alignment=4)
        self._tensor("next_positions", next_positions, torch.int64, (b,))
        strict = status is not None
        if strict:
            self._check_status(status)
            if status.records.shape[0] != j:
                raise ValueError("status must contain one record per bound layer")
        elif self.checks != "trusted":
            raise ValueError("status is required unless checks='trusted'")
        signature = (
            "handoff_layers",
            b,
            j,
            strict,
            layers.table.data_ptr(),
            status.records.data_ptr() % 16 if strict else 0,
            status.records.stride(0) if strict else 0,
            slot_indices.dtype,
            next_positions.dtype,
            slot_indices.stride(),
            next_positions.stride(),
            tuple(
                (s.checkpoint.stride(), s.conv_history.stride(), c.stride(), h.stride())
                for s, c, h in zip(
                    layers.states, layers.canonical_states, layers.histories, strict=True
                )
            ),
        )
        capturing = self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if capturing and signature not in self._warm:
            raise RuntimeError("warm this replay call shape eagerly before capture")
        if self.provider["handoff_layers"] in ("cute", "triton"):
            backend = self._backend(self.provider["handoff_layers"])
            backend.handoff_layers(self, layers, next_positions, slot_indices, status)
        else:
            for index, state in enumerate(layers.states):
                self.handoff(
                    canonical_state=layers.canonical_states[index],
                    history=layers.histories[index],
                    next_positions=next_positions,
                    slot_indices=slot_indices,
                    state=state,
                    status=status.for_layer(index) if strict else None,
                )
        self._warm.add(signature)

    def release(self, *, slot_indices, state, status):
        """Invalidate slots and advance generation without flushing payload (§7)."""
        self._run("release", slot_indices=slot_indices, state=state, status=status)

    def reset(self, *, state, weight_revision: int, status, pending_blocks=(), workspaces=()):
        """O(R+S+B) metadata reset, epoch increment, no payload replay/zeroing (§7).

        Pass every resident pending block to clear transaction metadata. Epoch mismatch
        also invalidates any omitted block. Caller drains prior launches before reset.
        Pool contents may be undefined; status must be valid and begun. Pass borrowed
        step/graph workspace dicts or arrivals tensors to zero publication counters
        on the current stream before metadata reset; otherwise zero them externally.
        Payload remains undefined until initialize, handoff or import_snapshot.
        """
        self._plans.clear()
        if type(weight_revision) is not int or not 0 <= weight_revision < 2**63:
            raise ValueError("weight_revision must be a nonnegative int64 host integer")
        pending_blocks = tuple(pending_blocks)
        for pending in pending_blocks:
            self._check_pending(pending)
        self._check_state(state)
        self._check_status(status)
        arrivals = []
        for workspace in workspaces:
            if isinstance(workspace, dict) and "phase_arrivals" in workspace:
                counters = workspace["phase_arrivals"]
                if not isinstance(counters, torch.Tensor) or counters.ndim != 2:
                    raise ValueError("phase_arrivals must be a two-dimensional tensor")
                depth, batch = counters.shape
                if depth < 1 or batch < 1:
                    raise ValueError("phase_arrivals dimensions must be positive")
                self._tensor("phase_arrivals", counters, torch.int32, (depth, batch), alignment=256)
                arrivals.append(counters)
            tensor = workspace.get("arrivals") if isinstance(workspace, dict) else workspace
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or not tensor.numel():
                raise ValueError("workspace arrivals must be a nonempty one-dimensional tensor")
            batch = (
                workspace["out"].shape[-2]
                if isinstance(workspace, dict) and "out" in workspace
                else tensor.shape[0]
            )
            spec = self.workspace_spec("step", batch_size=batch).get("arrivals")
            if spec is None:
                spec = self.graph_data_spec(batch_size=batch).get("arrivals")
            if spec is None:
                raise ValueError("this configuration has no arrivals workspace")
            self._tensor("arrivals", tensor, spec.dtype, spec.shape, alignment=256)
            arrivals.append(tensor)
        protected = [*_tensors(state), *(_ for p in pending_blocks for _ in _tensors(p))]
        protected += [status.records, status.validity, status.sequence, status.poison]
        self._check_disjoint(
            "reset_counters",
            arrivals,
            protected,
            "workspace counters must not overlap state, pending or status",
        )
        for tensor in arrivals:
            tensor.zero_()
        self._run(
            "reset",
            state=state,
            weight_revision=weight_revision,
            pending_blocks=pending_blocks,
            status=status,
        )
