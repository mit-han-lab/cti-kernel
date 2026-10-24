# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""The fused GDN decode across device types on shared inputs.

``python -m tests.gpu.xdevice_gdn_decode dump OUT.pt`` on an accepted device saves, per
geometry and batch size, the launch configuration, y and the new state of the first four
touched slots. Inputs come from integer hashing, exact on any device and host architecture
(CPU ``torch.randn`` differs between x86 and aarch64 hosts).
``... compare A.pt B.pt`` (CPU only) requires bitwise equality where the two devices launched
the same configuration and otherwise agreement within twice the tier-1 bound (each side is
within one BF16 ULP of FP64). pytest collects no tests from this module.
"""

import argparse
import sys

import torch

BATCHES = (1, 2, 3, 4, 8, 16, 64, 256)
GEOMETRIES = ((8, 16), (8, 24), (16, 32), (16, 48))
TIER1_ULP = 1.0
K = V = 128


def _uniform(shape, seed: int, low: float, high: float, device, dtype=torch.float32):
    """Values on a 2^-15 grid in [low, high) from an integer hash of the index and ``seed``."""
    n = 1
    for size in shape:
        n *= size
    x = torch.arange(n, dtype=torch.int64, device=device)
    for _ in range(2):
        x = (x * 1103515245 + 12345 + seed * 2654435761) % 2147483648
        x = x ^ (x >> 13)
    unit = (x % 32768).to(torch.float32) / 32768
    return (low + (high - low) * unit).view(shape).to(dtype)


def _fingerprint(*tensors: torch.Tensor) -> int:
    """An exact integer sum of the tensors' bits."""
    total = 0
    for t in tensors:
        bits = t.contiguous().view(
            {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        )
        total += int(bits.long().sum())
    return total


def _case(k_heads: int, v_heads: int, batch: int, device):
    from .test_gdn import _widths

    conv_dim, width = _widths(k_heads, v_heads)
    seed = 1000 * batch + 10 * k_heads + v_heads
    weights = (
        _uniform((conv_dim, 4), seed, -0.8, 0.8, device, torch.bfloat16),
        _uniform((v_heads,), seed + 1, -0.7, 2.1, device),
        _uniform((v_heads,), seed + 2, -2.0, 2.0, device),
        _uniform((V,), seed + 3, 0.85, 1.15, device),
    )
    slots = batch + 4
    projected = _uniform((batch, width), seed + 4, -2.0, 2.0, device, torch.bfloat16)
    order = _uniform((slots - 1,), seed + 5, 0.0, 1.0, device).argsort(stable=True) + 1
    slot_indices = order[:batch].to(torch.int32)
    state = _uniform((slots, v_heads, V, K), seed + 6, -0.4, 0.4, device)
    conv_state = _uniform((slots, 3, conv_dim), seed + 7, -2.0, 2.0, device, torch.bfloat16)
    return weights, projected, slot_indices, state, conv_state


def dump(path: str) -> None:
    from ctikernels.gdn import GatedDeltaNetDecode, _triton_decode
    from ctikernels.gdn._contract import Geometry

    from .test_gdn_triton import _launch_traits

    device = torch.device("cuda")
    traits = _launch_traits(device)
    rows = {}
    for k_heads, v_heads in GEOMETRIES:
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device=device)
        assert op.provider == "triton", torch.cuda.get_device_name()
        for batch in BATCHES:
            weights, projected, slot_indices, state, conv_state = _case(
                k_heads, v_heads, batch, device
            )
            # Before the call: it updates the pools in place.
            inputs = _fingerprint(projected, slot_indices, state, conv_state, *weights)
            with torch.inference_mode():
                y = op(projected, *weights, slot_indices, state, conv_state)
            plan = _triton_decode.decode_plan(
                Geometry(k_heads, v_heads), batch, projected.stride(0), True, **traits
            )
            touched = slot_indices[:4].long()
            rows[(k_heads, v_heads, batch)] = {
                "inputs": inputs,
                "config": repr(sorted((k, str(v)) for k, v in plan.arguments.items())),
                "grid": plan.grid,
                "y": y.cpu(),
                "state": state[touched].cpu(),
            }
    torch.save({"device": torch.cuda.get_device_name(device), "rows": rows}, path)
    print(f"{path}: {len(rows)} rows on {torch.cuda.get_device_name(device)}")


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest |a - b| in BF16 ULP of max(|a|, |b|, row RMS / 8)."""
    a, b = a.double(), b.double()
    rms = b.pow(2).mean(-1, keepdim=True).sqrt()
    scale = torch.maximum(torch.maximum(a.abs(), b.abs()), rms / 8)
    _, exponent = torch.frexp(scale)
    ulp = torch.pow(2.0, (exponent - 8).double()).clamp_min(2.0**-133)
    return float(((a - b).abs() / ulp).max())


def compare(first: str, second: str) -> int:
    a, b = torch.load(first), torch.load(second)
    failures = same = 0
    for key, left in a["rows"].items():
        right = b["rows"][key]
        if left["inputs"] != right["inputs"]:
            raise SystemExit(f"{key}: the two dumps' inputs differ")
        if left["config"] == right["config"]:
            same += 1
            ok = all(torch.equal(left[t].view(-1), right[t].view(-1)) for t in ("y", "state"))
            how = "bitwise"
        else:
            worst = max(_max_ulp(left[t].float(), right[t].float()) for t in ("y", "state"))
            ok = worst <= 2 * TIER1_ULP
            how = f"{worst:.2f} ULP"
        failures += not ok
        print(f"{key}: {'ok' if ok else 'FAIL'} ({how})")
    print(f"{a['device']} vs {b['device']}: {same} same-config rows, {failures} failures")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dump").add_argument("path")
    pair = sub.add_parser("compare")
    pair.add_argument("first")
    pair.add_argument("second")
    args = parser.parse_args()
    if args.command == "dump":
        dump(args.path)
        return 0
    return compare(args.first, args.second)


if __name__ == "__main__":
    sys.exit(main())
