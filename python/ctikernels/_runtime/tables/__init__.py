# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Offline-tuned launch configs: per-op, per-device-class tables and a pure lookup.

An op declares a :class:`TunedOp`: its config space, its bucket dimensions and a
default config rule. ``benchmark/kernel_tuning/tune`` measures the space on each
device class and writes ``data/<op>@<device class>.json``. At run time
:meth:`TunedOp.config` reads that table once and looks the call's bucket up:
the exact bucket, else the nearest smaller tuned bucket along the first ordered
dimension (the others must match), else the op's default. No timing, no
autotune: the same shape on the same device class always gets the same config.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("data")
FORMAT = 1


@dataclass(frozen=True)
class Dim:
    """A bucket dimension: ``edges`` (ascending inclusive upper limits) make it ordered,
    values above the last edge fall in the last bucket; no edges keep values exact."""

    name: str
    edges: tuple[int, ...] | None = None

    def bucket(self, value):
        if self.edges is None:
            return list(value) if isinstance(value, tuple) else value
        if type(value) is not int or value < 0:
            raise ValueError(f"{self.name} must be a non-negative int")
        return next((edge for edge in self.edges if value <= edge), self.edges[-1])


def pow2_edges(low: int, high: int) -> tuple[int, ...]:
    """Power-of-two edges low, 2*low, ..., high."""
    edges = [low]
    while edges[-1] < high:
        edges.append(edges[-1] * 2)
    return tuple(edges)


@dataclass(frozen=True)
class DeviceClass:
    """Table key of a device, from its hardware alone."""

    major: int
    minor: int
    sms: int

    @property
    def name(self) -> str:
        return f"sm{self.major}{self.minor}-{self.sms}"


@lru_cache(maxsize=None)
def device_class(device) -> DeviceClass:
    import torch

    props = torch.cuda.get_device_properties(device)
    return DeviceClass(props.major, props.minor, props.multi_processor_count)


class TunedOp:
    """One op's config space, bucket function and per-device-class tables.

    ``space`` maps each config parameter to its allowed values. ``dims`` are the
    shape keywords :meth:`bucket` takes; the first ordered one is the inheritance
    axis. ``default(shape, device_class)`` returns the untuned config for a shape
    (keywords as in ``dims``); it must lie in ``space``. ``valid(config, shape)`` states
    shape-dependent limits (e.g. a tile no larger than the rank bucket); ``shape`` is the
    bucketed shape. ``version`` changes whenever the kernel or space changes in a way that
    invalidates measured tables.
    """

    def __init__(
        self,
        name: str,
        *,
        version: int,
        space: Mapping[str, tuple],
        dims: tuple[Dim, ...],
        default: Callable[[dict, DeviceClass], dict],
        valid: Callable[[dict, dict], bool] = lambda config, shape: True,
    ):
        self.name = name
        self.version = version
        self.space = {key: tuple(values) for key, values in space.items()}
        self.dims = tuple(dims)
        self.axis = next((i for i, dim in enumerate(self.dims) if dim.edges is not None), None)
        self._default = default
        self._valid = valid
        self._tables: dict[str, dict] = {}  # device class name -> rows
        self._lock = threading.Lock()

    def bucket(self, **shape) -> tuple:
        if shape.keys() != {dim.name for dim in self.dims}:
            raise ValueError(f"{self.name} shape keys must be {[d.name for d in self.dims]}")
        return tuple(_freeze(dim.bucket(shape[dim.name])) for dim in self.dims)

    def default(self, device_class: DeviceClass, **shape) -> dict:
        return self.check(self._default(shape, device_class), shape)

    def valid(self, config: Mapping, shape: Mapping) -> bool:
        """Whether ``config`` obeys the op's limits for ``shape`` (bucketed first)."""
        names = [dim.name for dim in self.dims]
        bucketed = dict(zip(names, _thaw(self.bucket(**shape)), strict=True))
        return bool(self._valid(dict(config), bucketed))

    def check(self, config: Mapping, shape: Mapping) -> dict:
        """``config`` as a plain dict, or ValueError if it leaves the declared space or
        breaks the op's limits for ``shape``."""
        if config.keys() != self.space.keys():
            raise ValueError(f"{self.name} config keys must be {sorted(self.space)}")
        for key, value in config.items():
            if value not in self.space[key]:
                raise ValueError(f"{self.name} {key}={value!r} not in {self.space[key]}")
        if not self.valid(config, shape):
            raise ValueError(f"{self.name} config {dict(config)} invalid for {dict(shape)}")
        return dict(config)

    def config(self, device, **shape) -> dict:
        """The tuned config for ``shape`` on ``device`` (a dict to pass to the launcher)."""
        return self.lookup(device_class(device), **shape)

    def lookup(self, device_class: DeviceClass, **shape) -> dict:
        key = self.bucket(**shape)
        rows = self.table(device_class)
        if key in rows:
            return dict(rows[key])
        if self.axis is not None:
            below = [
                row
                for row in rows
                if row[self.axis] < key[self.axis]
                and _drop(row, self.axis) == _drop(key, self.axis)
            ]
            if below:
                config = rows[max(below, key=lambda row: row[self.axis])]
                if self.valid(config, shape):
                    return dict(config)
        return self.default(device_class, **shape)

    def configs(self, device, shapes: Iterable[Mapping]) -> list[dict]:
        """Distinct configs of the declared ``shapes``, in first-seen order (for warm-up)."""
        seen = []
        for shape in shapes:
            config = self.config(device, **shape)
            if config not in seen:
                seen.append(config)
        return seen

    def table(self, device_class: DeviceClass) -> dict[tuple, dict]:
        with self._lock:
            if device_class.name not in self._tables:
                path = table_path(self.name, device_class.name)
                self._tables[device_class.name] = self._load(path)
            return self._tables[device_class.name]

    def _load(self, path: Path) -> dict[tuple, dict]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        if (data.get("format"), data.get("op"), data.get("version")) != (
            FORMAT,
            self.name,
            self.version,
        ):
            raise ValueError(f"{path.name}: stale table for {self.name} v{self.version}; retune")
        names = [dim.name for dim in self.dims]
        if data["dims"] != names or data["params"] != sorted(self.space):
            raise ValueError(f"{path.name}: dims/params differ from {self.name}; retune")
        rows = {}
        for bucket, values in data["rows"]:
            shape = dict(zip(names, _thaw(bucket), strict=True))
            key = self.bucket(**shape)
            if key in rows:
                raise ValueError(f"{path.name}: duplicate bucket {bucket}")
            rows[key] = self.check(dict(zip(data["params"], values, strict=True)), shape)
        return rows


def table_path(op: str, device_class: str) -> Path:
    return DATA / f"{op}@{device_class}.json"


def encode(op: TunedOp, rows: Mapping[tuple, Mapping]) -> str:
    """Table file text for ``rows`` (bucket -> config)."""
    params = sorted(op.space)
    head = {
        "format": FORMAT,
        "op": op.name,
        "version": op.version,
        "dims": [dim.name for dim in op.dims],
        "params": params,
    }
    return dumps(head, [[_thaw(b), [c[p] for p in params]] for b, c in rows.items()])


def dumps(head: Mapping, rows: Iterable[list]) -> str:
    """Table file text: ``head`` then ``rows`` ([bucket, values]) sorted, one per line."""
    lines = sorted(json.dumps(row, separators=(",", ":")) for row in rows)
    if not lines:
        return json.dumps({**head, "rows": []}) + "\n"
    body = ",\n    ".join(lines)
    return json.dumps(dict(head))[:-1] + f', "rows": [\n    {body}\n]}}\n'


def _drop(key, index):
    return key[:index] + key[index + 1 :]


def _freeze(value):
    return tuple(_freeze(item) for item in value) if isinstance(value, list) else value


def _thaw(value):
    return [_thaw(item) for item in value] if isinstance(value, (tuple, list)) else value
