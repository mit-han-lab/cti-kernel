# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Fixed-address invocation controls and host-only completion decoding (§§3–4)."""

from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from ._storage import ReplayPending, _valid_q, regions


def graph_control_sizes(*, batch_size, verify_length=16, depth=2, num_layers=1):
    """Exact control payloads; each named allocation also reports ABI alignment.

    Host mirrors cost ``num_layers * depth * (16 + batch_size)`` bytes. Raw
    pending, projections, outputs and provider scratch are additional allocations.
    ``depth`` must cover the engine's maximum submitted-but-unsettled count.
    """
    if any(type(n) is not int or n < 1 for n in (batch_size, depth, num_layers)):
        raise ValueError("batch_size, depth and num_layers must be positive integers")
    if not _valid_q(verify_length):
        raise ValueError("verify_length must be an integer >= 1")
    d, b, q, j = depth, batch_size, verify_length, num_layers
    specs = regions(
        [
            ("positions", (d, b), torch.int64, 0),
            ("verify_positions", (d, b, q), torch.int64, 0),
            ("slot_indices", (d, b), torch.int32, -1),
            ("accepted_lengths", (d, b), torch.int32, 0),
            ("snapshot_offsets", (d, b), torch.int64, 0),
            ("snapshot_rows", (d, b), torch.int32, -1),
            ("pending_metadata", (j, d, b, 8), torch.int64, 0),
            ("validity", (j, d, b), torch.bool, 0),
            ("records", (j, d, 16), torch.uint8, 0),
            ("sequence", (d,), torch.int64, 0),
            ("poison", (1,), torch.int32, 0),
        ]
    )
    return {
        name: replace(spec, lifetime="pool" if name == "poison" else "invocation")
        for name, spec in specs.items()
    }


@dataclass(frozen=True)
class ReplayStatus:
    """One invocation's layer record and whole-invocation lifecycle views.

    Construct via ``bind_graph_buffers().status(slot, layer)`` before capture.
    Begin/end/clear cover ALL layers, independent of the selected arithmetic layer.
    The sequence input must stay unchanged through completion and settlement.
    """

    records: torch.Tensor  # [J,16], or slotted [J,D,16]
    validity: torch.Tensor  # [J,B], or slotted [J,D,B]
    sequence: torch.Tensor  # [1], or slotted [D]
    poison: torch.Tensor  # [1], shared by all layers and specializations
    layer: int = 0
    batch_size: int | None = None
    slotted: bool = False

    @property
    def record(self):
        return self.records[self.layer]

    @property
    def code(self):
        return self.record[..., 8:12].view(torch.int32)

    @property
    def valid(self):
        return self.validity[self.layer, ..., : self.batch_size]

    def for_layer(self, layer):
        if not 0 <= layer < self.records.shape[0]:
            raise ValueError("invalid layer")
        return ReplayStatus(
            self.records,
            self.validity,
            self.sequence,
            self.poison,
            layer,
            self.batch_size,
            self.slotted,
        )

    def copy_to_host(self):
        """Synchronous TEST convenience. Serving uses its existing async D2H."""
        return self.records.cpu().clone(), self.validity.cpu().clone()


@dataclass(frozen=True)
class ReplayResult:
    expected_sequence: int
    sequences: tuple[int, ...]
    codes: tuple[int, ...]
    completions: tuple[int, ...]
    validity: tuple[tuple[bool, ...], ...]

    @property
    def complete(self):
        return self.expected_sequence > 0 and all(
            s == self.expected_sequence and c == 1
            for s, c in zip(self.sequences, self.completions, strict=True)
        )

    @property
    def ok(self):
        return self.complete and not any(self.codes)


def read_status_copy(records, validity, *, expected_sequence):
    """Read only a host copy the engine already completed via async D2H.

    No wait, CUDA access or copy is performed here. Local OK is necessary but
    insufficient for publication: the engine joins all ranks and KV completion
    and settles FIFO. A failed/stale validity bit never grants publication.
    """
    if (
        records.device.type != "cpu"
        or validity.device.type != "cpu"
        or records.dtype != torch.uint8
        or records.ndim != 2
        or records.shape[1] != 16
        or validity.dtype != torch.bool
        or validity.ndim != 2
        or validity.shape[0] != records.shape[0]
        or not records.shape[0]
    ):
        raise ValueError("expected host uint8 [layers,16] and bool [layers,rows] copies")
    if type(expected_sequence) is not int or not 0 < expected_sequence < 2**63:
        raise ValueError("expected sequence must be a positive int64")
    return ReplayResult(
        expected_sequence,
        tuple(int(row[:8].view(torch.int64)[0]) for row in records),
        tuple(int(row[8:12].view(torch.int32)[0]) for row in records),
        tuple(int(row[12:].view(torch.int32)[0]) for row in records),
        tuple(tuple(row.tolist()) for row in validity),
    )


@dataclass(frozen=True)
class ReplayGraphBuffers:
    buffers: MappingProxyType

    def status(self, slot, layer=0, *, batch_size=None):
        b = self.buffers
        if not 0 <= slot < b["sequence"].numel():
            raise ValueError("invalid invocation slot")
        if batch_size is None:
            batch_size = b["positions"].shape[1]
        if not 0 < batch_size <= b["positions"].shape[1]:
            raise ValueError("invalid row capacity")
        return ReplayStatus(
            b["records"][:, slot],
            b["validity"][:, slot],
            b["sequence"][slot : slot + 1],
            b["poison"],
            batch_size=batch_size,
        ).for_layer(layer)

    def pending(self, raw, *, slot, layer=0):
        return ReplayPending(raw, self.buffers["pending_metadata"][layer, slot, : raw.shape[0]])

    def slotted_status(self, layer=0, *, batch_size=None):
        """All invocation slots; select one on device with ``status_slot``."""
        b = self.buffers
        capacity = b["positions"].shape[1]
        batch_size = capacity if batch_size is None else batch_size
        if type(batch_size) is not int or not 0 < batch_size <= capacity:
            raise ValueError("invalid row capacity")
        return ReplayStatus(
            b["records"],
            b["validity"][:, :, :batch_size],
            b["sequence"],
            b["poison"],
            batch_size=batch_size,
            slotted=True,
        ).for_layer(layer)

    def slotted_pending(self, raw, *, layer=0):
        """Borrow raw [D,B,Q,W] and the matching layer's [D,B,8] metadata."""
        if raw.ndim != 4 or raw.shape[0] != self.buffers["sequence"].numel():
            raise ValueError("slotted pending requires raw [D,B,Q,W]")
        if not 0 <= layer < self.buffers["pending_metadata"].shape[0]:
            raise ValueError("invalid layer")
        if raw.shape[1] > self.buffers["pending_metadata"].shape[2]:
            raise ValueError("invalid pending row capacity")
        return ReplayPending(raw, self.buffers["pending_metadata"][layer, :, : raw.shape[1]])
