# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""Deterministic discovery; roots are injectable for CPU-only testing."""

from ipaddress import IPv6Address
from pathlib import Path


def roce_devices(root="/sys/class/infiniband") -> list[tuple[str, int]]:
    """ACTIVE port-1 devices with a RoCE v2 IPv4-mapped GID, in name/index order."""
    found = []
    for device in sorted(Path(root).glob("*")):
        port = device / "ports/1"
        try:
            if port.joinpath("state").read_text().strip().split(":")[0] != "4":
                continue
            indices = sorted(
                (p for p in port.joinpath("gids").glob("*") if p.name.isdecimal()),
                key=lambda p: int(p.name),
            )
            for gid in indices:
                try:
                    kind = (port / "gid_attrs/types" / gid.name).read_text().strip()
                    if kind == "RoCE v2" and IPv6Address(gid.read_text().strip()).ipv4_mapped:
                        found.append((device.name, int(gid.name)))
                except (OSError, ValueError):
                    continue
        except OSError:
            continue
    return found


def big_cores(root="/sys/devices/system/cpu") -> list[int]:
    """Online cores of the fastest tier: by maximum frequency, else by capacity.

    Capacity differs slightly within one tier (GB10's X925 cores read 997-1024), so a core
    counts when it is within 10% of the highest value."""
    cpus = []
    for path in Path(root).glob("cpu[0-9]*"):
        if not path.name[3:].isdecimal():
            continue
        try:
            if (path / "online").exists() and (path / "online").read_text().strip() == "0":
                continue
        except OSError:
            continue
        cpus.append((int(path.name[3:]), path))
    for field in ("cpufreq/cpuinfo_max_freq", "cpu_capacity"):
        scores = {}
        for cpu, path in cpus:
            try:
                value = int((path / field).read_text().strip())
                if value > 0:
                    scores[cpu] = value
            except (OSError, ValueError):
                continue
        if scores:
            highest = max(scores.values())
            return sorted(cpu for cpu, value in scores.items() if value * 10 >= highest * 9)
    return sorted(cpu for cpu, _ in cpus)


def proxy_core(root="/sys/devices/system/cpu") -> int:
    """Highest-numbered big core; -1 when sysfs provides no CPU information."""
    return max(big_cores(root), default=-1)
