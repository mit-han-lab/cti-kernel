# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors
"""CPU-only empirical rooflines."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEVICES = ("rtx5090", "rtx-pro-6000", "gb10", "gb300")  # gb300: memory and launch only
CEILINGS_DIR = Path(__file__).with_name("ceilings")
MMA_KINDS = (
    "bf16_m16n8k16",
    "e4m3_m16n8k32",
    "mxf8f6f4_e4m3_m16n8k32",
    "mxf8f6f4_e2m1_e4m3_m16n8k32",
    "mxf8f6f4_e4m3_e2m1_m16n8k32",
    "mxf4nvf4_m16n8k64",
    "mxf4_m16n8k64",
)
DRAM_CATEGORIES = (
    "weights",
    "scales",
    "activations",
    "kv_state",
    "outputs",
    "intermediates",
    "splitk",
)


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required(table: Mapping, key: str):
    value = table
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing ceiling key: {key}")
        value = value[part]
    return value


def load_ceilings(source: str | Mapping, *, directory: Path | None = None) -> dict:
    """Validate present rates; allow explicitly declared gaps in measured builder tables."""
    if isinstance(source, str):
        if source not in DEVICES:
            raise ValueError(f"Unknown device {source!r}; expected one of {DEVICES}")
        path = (CEILINGS_DIR if directory is None else Path(directory)) / f"{source}.json"
        with path.open(encoding="utf-8") as stream:
            table = json.load(stream)
    elif isinstance(source, Mapping):
        table = dict(source)
    else:
        raise ValueError("ceilings must be a device name or a mapping")
    _integer(_required(table, "sm_count"), "sm_count")
    missing = table.get("missing_measurements", [])
    if not isinstance(missing, list) or any(not isinstance(k, str) for k in missing):
        raise ValueError("missing_measurements must be a list of ceiling keys")

    def check(key, *, positive=False):
        if key in missing or any(key.startswith(k + ".") for k in missing):
            # Declaring a present value unavailable is inconsistent and must not hide bad data.
            try:
                _required(table, key)
            except ValueError:
                return
            raise ValueError(f"Unavailable ceiling is present: {key}")
        _number(_required(table, key), key, positive=positive)

    for key in ("dram_read_gbs", "dram_write_gbs", "dram_copy_gbs", "l2_read_gbs"):
        check(f"memory.{key}", positive=True)
    for kind in MMA_KINDS:
        for rate in ("tflops_gpu", "tflops_per_sm"):
            key = f"tensor_core.{kind}.{rate}"
            check(key, positive=True)
    for cost in ("graph_node_us", "graph_node_pdl_us", "eager_launch_us"):
        key = f"fixed_costs.{cost}"
        check(key)
    _validate_curve(_required(table, "memory.small_transfer"), "memory.small_transfer")
    directions = table.get("memory", {}).get("small_transfer_by_direction", {})
    if not isinstance(directions, Mapping):
        raise ValueError("memory.small_transfer_by_direction must be a mapping")
    for direction, curve in directions.items():
        _validate_curve(curve, f"memory.small_transfer_by_direction.{direction}")
    return table


def _validate_curve(curve, name):
    if not isinstance(curve, list):
        raise ValueError(f"{name} must be a list")
    sizes = []
    for index, point in enumerate(curve):
        for key in ("bytes", "gbs", "time_us"):
            point_name = f"{name}[{index}].{key}"
            if not isinstance(point, Mapping) or key not in point:
                raise ValueError(f"Missing ceiling key: {point_name}")
            _number(point[key], point_name, positive=True)
        sizes.append(point["bytes"])
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"{name} bytes must be unique")


@dataclass
class Kernel:
    name: str
    dram_bytes: Mapping[str, float] = field(default_factory=dict)
    l2_bytes: float = 0
    flops: Mapping[str, float] = field(default_factory=dict)
    executed_flops: Mapping[str, float] | None = None
    n_ctas: int = 1
    cta_capacity: int | None = None
    cache_state: str = "cold"
    pdl: bool = False
    # Subsets of dram_bytes served by L2 instead; l2_bytes counts additional traffic.
    l2_resident_bytes: Mapping[str, float] = field(default_factory=dict)
    dram_mode: str = "read"
    # Calibrated tensor service times at the chosen residency, NOT FLOP counts.
    cta_costs_us: Sequence[float] | None = None
    cta_threads: int | None = None
    cta_smem_bytes: int | None = None
    cta_regs_per_thread: int | None = None

    def __post_init__(self):
        if self.cache_state not in ("cold", "warm", "app"):
            raise ValueError("cache_state must be cold, warm or app")
        if self.dram_mode not in ("read", "write", "copy"):
            raise ValueError("dram_mode must be read, write or copy")
        _integer(self.n_ctas, "n_ctas")
        if self.cta_capacity is not None:
            _integer(self.cta_capacity, "cta_capacity")
        if self.cta_threads is not None:
            _integer(self.cta_threads, "cta_threads")
        if self.cta_smem_bytes is not None:
            if (
                isinstance(self.cta_smem_bytes, bool)
                or not isinstance(self.cta_smem_bytes, int)
                or self.cta_smem_bytes < 0
            ):
                raise ValueError("cta_smem_bytes must be a nonnegative integer")
        if self.cta_regs_per_thread is not None:
            _integer(self.cta_regs_per_thread, "cta_regs_per_thread")
        _number(self.l2_bytes, "l2_bytes")
        for category, size in self.dram_bytes.items():
            if category not in DRAM_CATEGORIES:
                raise ValueError(f"Unknown DRAM category: {category}")
            _number(size, f"dram_bytes.{category}")
        for category, size in self.l2_resident_bytes.items():
            _number(size, f"l2_resident_bytes.{category}")
            if category not in self.dram_bytes or size > self.dram_bytes[category]:
                raise ValueError(f"l2_resident_bytes.{category} must be a subset of dram_bytes")
        if self.cache_state == "cold" and any(self.l2_resident_bytes.values()):
            raise ValueError("cold cache_state cannot specify L2-resident DRAM bytes")
        for label, counts in (("flops", self.flops), ("executed_flops", self.executed_flops)):
            for kind, count in (counts or {}).items():
                if kind not in MMA_KINDS:
                    raise ValueError(f"Unknown MMA kind in {label}: {kind}")
                _number(count, f"{label}.{kind}")
        if self.executed_flops is not None:
            for kind, count in self.flops.items():
                if self.executed_flops.get(kind, 0) < count:
                    raise ValueError(f"executed_flops.{kind} must cover useful flops.{kind}")
        if self.cta_costs_us is not None:
            if len(self.cta_costs_us) != self.n_ctas:
                raise ValueError("cta_costs_us must contain one cost per CTA")
            for cost in self.cta_costs_us:
                _number(cost, "cta_costs_us")


@dataclass
class Call:
    kernels: Sequence[Kernel]
    mode: str = "serial"

    def __post_init__(self):
        if self.mode not in ("serial", "pdl_chain"):
            raise ValueError("mode must be serial or pdl_chain")


def lpt_slot_loads(cta_costs_us: Sequence[float], capacity: int) -> list[float]:
    """Longest-processing-time greedy assignment; max(loads) is the makespan.

    Slots represent SMs at residency=1, or calibrated concurrent CTA slots otherwise.
    This separate scheduling model is a heuristic, not an optimal scheduling bound.
    """
    _integer(capacity, "capacity")
    costs = [_number(cost, "cta_costs_us") for cost in cta_costs_us]
    loads = [0.0] * capacity
    for cost in sorted(costs, reverse=True):
        heapq.heapreplace(loads, loads[0] + cost)
    return sorted(loads, reverse=True)


@dataclass
class KernelBreakdown:
    name: str
    t_ref_us: float
    terms_us: dict[str, float]
    limiting_resource: str
    bound_type: str
    dram_bytes: float
    l2_bytes: float
    tensor_by_kind_us: dict[str, float]
    raw_floor_us: float
    overlapped_floor_us: float
    cta_capacity: int | None
    waves: int
    wave_efficiency: float
    slot_loads_us: list[float] | None
    cta_capacity_evidence: dict | None


@dataclass
class Utilization:
    ratio: float
    above_reference: bool


@dataclass
class Result:
    t_ref_us: float
    per_kernel: list[KernelBreakdown]
    limiting_resource: str
    bound_type: str
    assumptions: list[str]
    estimate_type: str

    def utilization(self, t_measured_us: float) -> Utilization:
        ratio = self.t_ref_us / _number(t_measured_us, "t_measured_us", positive=True)
        return Utilization(ratio, ratio > 1)

    def to_dict(self, t_measured_us: float | None = None) -> dict:
        data = asdict(self)
        if t_measured_us is not None:
            data["utilization"] = asdict(self.utilization(t_measured_us))
        return data


def _dram_time(size: float, memory: Mapping, model: str, direction: str) -> float:
    if size == 0:
        return 0.0
    rate = _required({"memory": memory}, f"memory.dram_{direction}_gbs")
    large = size / (rate * 1e3)
    if model == "large":
        return large
    key = f"memory.small_transfer_by_direction.{direction}"
    curve = sorted(_required({"memory": memory}, key), key=lambda point: point["bytes"])
    if not curve:
        raise ValueError(f"{key} must be nonempty for bw_model='curve'")
    if size > curve[-1]["bytes"]:
        return large
    if size <= curve[0]["bytes"]:
        return curve[0]["time_us"]  # Unmeasured sub-minimum latency: clamp, do not extrapolate.
    for left, right in zip(curve, curve[1:], strict=False):
        if size <= right["bytes"]:
            fraction = math.log(size / left["bytes"]) / math.log(right["bytes"] / left["bytes"])
            return math.exp(
                math.log(left["time_us"]) + fraction * math.log(right["time_us"] / left["time_us"])
            )
    raise AssertionError("unreachable curve interval")


def _bound(resource: str) -> str:
    return {"dram": "bandwidth", "l2": "bandwidth", "tensor": "compute", "floor": "floor"}[resource]


def _cta_capacity(kernel: Kernel, table: Mapping, residency: int | None):
    if kernel.cta_capacity is not None:
        return kernel.cta_capacity, {
            "source": "caller-calibrated",
            "cta_capacity": kernel.cta_capacity,
        }
    if residency is not None:
        capacity = table["sm_count"] * residency
        return capacity, {
            "source": "uncalibrated caller residency",
            "residency": residency,
            "cta_capacity": capacity,
        }
    key = "scheduling.cta_capacity"
    entries = _required(table, key)
    if not isinstance(entries, list):
        raise ValueError(f"{key} must be a list")
    if kernel.cta_threads is None or kernel.cta_smem_bytes is None:
        raise ValueError(
            f"{key}: provide cta_threads and cta_smem_bytes, or an explicit capacity/residency override"
        )
    shape = {"threads": kernel.cta_threads, "smem_bytes": kernel.cta_smem_bytes}
    if kernel.cta_regs_per_thread is not None:
        shape["regs_per_thread"] = kernel.cta_regs_per_thread
    candidates = []
    for entry in entries:
        for key_name in (
            "threads",
            "regs_per_thread",
            "cta_capacity",
            "theoretical_residency_per_sm",
            "measured_max_concurrent",
            "q_saturation",
        ):
            _integer(_required(entry, key_name), f"{key}.{key_name}")
        smem = _required(entry, "smem_bytes")
        if isinstance(smem, bool) or not isinstance(smem, int) or smem < 0:
            raise ValueError(f"{key}.smem_bytes must be a nonnegative integer")
        if all(entry[k] >= v for k, v in shape.items()):
            candidates.append(entry)
    # Pareto-minimal dominating shapes; a larger resource request is not a closer match.
    minimal = [
        entry
        for entry in candidates
        if not any(
            all(other[k] <= entry[k] for k in shape) and any(other[k] < entry[k] for k in shape)
            for other in candidates
        )
    ]
    if not minimal:
        raise ValueError(f"{key}: no calibrated shape dominates {shape}")
    chosen = min(
        minimal,
        key=lambda entry: (
            entry["cta_capacity"],
            entry["threads"],
            entry["smem_bytes"],
            entry["regs_per_thread"],
        ),
    )
    return chosen["cta_capacity"], {"source": "calibrated table", **chosen}


def estimate(
    call: Call,
    ceilings: str | Mapping,
    *,
    estimate_type: str = "useful",
    bw_model: str = "curve",
    residency: int | None = None,
) -> Result:
    """Estimate a call using useful work, or executed work and CTA scheduling.

    All mma.sync forms share one tensor pipeline: sum F/C, never max across kinds.
    Graph-node cost is a floor inside max: measured bandwidth/compute times already
    include launch, so adding the floor would count that launch cost twice.
    """
    if estimate_type not in ("useful", "implementation"):
        raise ValueError("estimate_type must be useful or implementation")
    if bw_model not in ("large", "curve"):
        raise ValueError("bw_model must be large or curve")
    call.__post_init__()
    if residency is not None:
        _integer(residency, "residency")
    table = load_ceilings(ceilings)
    assumptions = [
        "Empirical reference, not a physical bound; rates and floors may be an optimistic "
        "device envelope from different configurations. Preserve paired measurement metadata.",
        "Decimal GB/s and TFLOP/s; times in microseconds; FMA counts as two FLOPs.",
        "All mma.sync kinds share a tensor pipeline; their F/C times are summed.",
        "Graph-node floor is inside max, not added: bandwidth and compute ceilings already "
        "include launch; the independently measured node cost is a minimum service time.",
        "Only tensor, DRAM, L2 and node floor are modeled; scalar work, synchronization, "
        "shared memory and communication require separate measurements.",
        "Overall resource receives the largest sum of per-kernel limiting contributions; "
        "ties prefer floor, DRAM, L2, then tensor.",
    ]
    if table.get("missing_measurements"):
        assumptions.append("Partial measured table: unused unavailable resources are omitted.")
    if bw_model == "curve":
        assumptions.append(
            "Small-transfer time uses log(bytes)-log(time) interpolation through the largest "
            "sample; positive sizes below the first sample clamp to its time. Above the last "
            "sample use sustained bandwidth; the boundary can be discontinuous. "
            "The curve and sustained rate both match the kernel's DRAM direction."
        )
    else:
        assumptions.append(
            "DRAM uses the selected sustained read/write/copy bandwidth at all sizes."
        )
    if estimate_type == "useful":
        assumptions.append("Useful FLOPs only; no padding, wave penalty or LPT scheduling.")
    else:
        assumptions.append(
            "Q uses caller-calibrated capacity, explicit uncalibrated residency, or the smallest "
            "capacity among minimal dominating calibrated CTA shapes. Equal CTA waves affect tensor time only."
        )
    if call.mode == "pdl_chain":
        assumptions.append(
            "PDL chain: sum per-kernel maxima using the measured PDL per-node slope for PDL "
            "kernels. The slope already includes overlap; never subtract predecessor time."
        )
    else:
        assumptions.append("Strictly serial call: sum the per-kernel maxima.")
    breakdown = []
    contributions = dict.fromkeys(("floor", "dram", "l2", "tensor"), 0.0)
    for kernel in call.kernels:
        # Validate again because callers may have edited a dataclass or its mappings.
        kernel.__post_init__()
        resident = sum(kernel.l2_resident_bytes.values())
        dram = sum(kernel.dram_bytes.values()) - resident
        l2 = kernel.l2_bytes + resident
        assumptions.append(
            f"{kernel.name}: cache={kernel.cache_state}, {resident:g} DRAM-category bytes "
            f"explicitly moved to L2; l2_bytes is additional traffic; DRAM pattern={kernel.dram_mode}."
        )
        capacity = None
        capacity_evidence = None
        waves = 1
        efficiency = 1.0
        counts = kernel.flops
        slot_loads = None
        if estimate_type == "implementation":
            capacity, capacity_evidence = _cta_capacity(kernel, table, residency)
            assumptions.append(
                f"{kernel.name}: CTA capacity evidence: {json.dumps(capacity_evidence, sort_keys=True)}"
            )
            if kernel.executed_flops is None:
                assumptions.append(f"{kernel.name}: executed_flops absent; assuming useful FLOPs.")
            else:
                counts = kernel.executed_flops
            if kernel.cta_costs_us is None:
                waves = (kernel.n_ctas + capacity - 1) // capacity
                efficiency = kernel.n_ctas / (capacity * waves)
            else:
                slot_loads = lpt_slot_loads(kernel.cta_costs_us, capacity)
                assumptions.append(
                    f"{kernel.name}: LPT max slot load replaces tensor F/C and waves; per-CTA "
                    "costs must include executed work and be calibrated at this capacity. "
                    "Slots are SMs only at residency=1; persistent scheduling is approximated."
                )
        tensor_by_kind = {
            kind: count / (_required(table, f"tensor_core.{kind}.tflops_gpu") * 1e6)
            for kind, count in counts.items()
            if count > 0
        }
        tensor_time = sum(tensor_by_kind.values()) / efficiency
        if slot_loads is not None:
            tensor_time = max(slot_loads)
        floor_key = "graph_node_pdl_us" if kernel.pdl else "graph_node_us"
        raw_floor = _required(table, f"fixed_costs.{floor_key}")
        floor = raw_floor
        terms = {
            "floor": floor,
            "dram": _dram_time(dram, table["memory"], bw_model, kernel.dram_mode),
            "l2": l2 / (_required(table, "memory.l2_read_gbs") * 1e3) if l2 else 0.0,
            "tensor": tensor_time,
        }
        resource = max(terms, key=terms.get)
        elapsed = terms[resource]
        contributions[resource] += elapsed
        breakdown.append(
            KernelBreakdown(
                kernel.name,
                elapsed,
                terms,
                resource,
                _bound(resource),
                dram,
                l2,
                tensor_by_kind,
                raw_floor,
                raw_floor - floor,
                capacity,
                waves,
                efficiency,
                slot_loads,
                capacity_evidence,
            )
        )
    resource = max(contributions, key=contributions.get)
    return Result(
        math.fsum(row.t_ref_us for row in breakdown),
        breakdown,
        resource,
        _bound(resource),
        assumptions,
        estimate_type,
    )


def estimate_both(call: Call, ceilings: str | Mapping, **kwargs) -> dict[str, Result]:
    """Return the useful-work roofline and implementation estimate independently."""
    return {
        kind: estimate(call, ceilings, estimate_type=kind, **kwargs)
        for kind in ("useful", "implementation")
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICES, required=True)
    parser.add_argument("--dram-bytes", type=float, default=0, help="bytes per kernel")
    parser.add_argument("--flops", action="append", default=[], help="MMA_KIND=FLOPS per kernel")
    parser.add_argument("--executed-flops", action="append", default=[])
    parser.add_argument("--kernels", type=int, default=1, help="identical serial kernels")
    parser.add_argument("--n-ctas", type=int, default=1)
    parser.add_argument("--cta-capacity", type=int)
    parser.add_argument("--residency", type=int, help="explicit uncalibrated residency override")
    parser.add_argument("--cta-threads", type=int)
    parser.add_argument("--cta-smem-bytes", type=int)
    parser.add_argument("--cta-regs-per-thread", type=int)
    parser.add_argument("--bw-model", choices=("large", "curve"), default="curve")
    parser.add_argument("--mode", choices=("serial", "pdl_chain"), default="serial")
    parser.add_argument("--pdl", action="store_true")
    parser.add_argument("--measured-us", type=float)
    args = parser.parse_args(argv)

    def counts(items):
        result = {}
        for item in items:
            kind, count = item.split("=", 1)
            result[kind] = result.get(kind, 0) + float(count)
        return result

    try:
        _integer(args.kernels, "kernels")
        call = Call(
            [
                Kernel(
                    f"kernel_{i}",
                    dram_bytes={"weights": args.dram_bytes},
                    flops=counts(args.flops),
                    executed_flops=counts(args.executed_flops) if args.executed_flops else None,
                    n_ctas=args.n_ctas,
                    cta_capacity=args.cta_capacity,
                    cta_threads=args.cta_threads,
                    cta_smem_bytes=args.cta_smem_bytes,
                    cta_regs_per_thread=args.cta_regs_per_thread,
                    pdl=args.pdl,
                )
                for i in range(args.kernels)
            ],
            mode=args.mode,
        )
        results = estimate_both(call, args.device, bw_model=args.bw_model, residency=args.residency)
        print(json.dumps({k: v.to_dict(args.measured_us) for k, v in results.items()}, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
