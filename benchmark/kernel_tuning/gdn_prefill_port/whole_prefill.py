# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GDN prefill benchmark: product (whatever GatedDeltaNetPrefill resolves), product fla-core,
external reference kernels (g305, gb10) and the market chains (SGLang, vLLM), CUDA-graph replay
of one call, L2 flushed before each replay, median of REPS.

usage: whole_prefill.py [--providers a,b,...] [--geoms 16x32,16x48] [--set small|long|all]
                        [--reps 20] [--check]
Prints one JSON line per (case, provider). Port is imported only when requested.
Port correctness against v1 is always checked before timing, including pool writes.
The companion whole_parity.py preserves the source paired ABAB protocol (rotating
order), CUDA graphs, 256 MiB L2 flush outside events, eps from 20 self pairs and
verdicts. This single-provider driver retains the source median-replay protocol.
No CUDA activity at import; diagnostics never echo argv, env or exception text.
"""

import argparse
import json
import math
import os
import sys
import time

import torch

BUNDLE = os.environ.get("GDNPP_BUNDLE", "")
K = V = 128

SMALL = [  # (lengths, initial flags) - small batch first
    ([16], [False]),
    ([64], [False]),
    ([128], [True]),
    ([256], [False]),
    ([512], [False]),
    ([64] * 2, [True, False]),
    ([128] * 4, [True, False, True, False]),
    ([32, 96, 200, 184], [True, True, False, False]),
]
LONG = [
    ([1024], [False]),
    ([2048], [False]),
    ([4096], [False]),
    ([8192], [False]),
    ([4096] * 2, [False, True]),
    ([2048] * 4, [False] * 4),
    ([1024] * 8, [True, False] * 4),
    ([6000, 1500, 500, 192], [False] * 4),
    ([16384], [False]),
]


class QuietParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "Invalid arguments; use --help for supported options.\n")


class PortPrefillUnavailableError(RuntimeError):
    """The port's lazy entry point is not available in this checkout."""


PORT_UNAVAILABLE = (
    "PortPrefill is unavailable; implement ctikernels.gdn.chunk_ws.prefill.PortPrefill."
)


def parse_port_options(value):
    """Parse distinct identifier=value options as bool, decimal int, or string."""
    options = {}
    if not value.strip():
        return options
    for item in value.split(","):
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if (
            not sep
            or not key.isidentifier()
            or not raw
            or key in options
            or key in {"num_k_heads", "num_v_heads", "device"}
        ):
            raise argparse.ArgumentTypeError("Invalid port options")
        if raw.lower() in {"true", "false"}:
            parsed = raw.lower() == "true"
        else:
            try:
                parsed = int(raw)
            except ValueError:
                parsed = raw
        options[key] = parsed
    return options


def diff_metrics(actual, reference):
    """Chunked max absolute and RMS(error)/RMS(reference), with JSON-safe nonfinites."""
    if actual.shape != reference.shape:
        return {"max_abs": None, "rms_relative": None, "shape_match": False}
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    maximum = error_sq = reference_sq = 0.0
    for start in range(0, actual.numel(), 1 << 20):
        a = actual[start : start + (1 << 20)].float()
        b = reference[start : start + (1 << 20)].float()
        d = a - b
        peak = d.abs().max().item()
        if not math.isfinite(peak):
            return {"max_abs": None, "rms_relative": None, "finite": False}
        maximum = max(maximum, peak)
        error_sq += d.double().square().sum().item()
        reference_sq += b.double().square().sum().item()
    relative = (
        math.sqrt(error_sq / reference_sq) if reference_sq else (0.0 if not error_sq else None)
    )
    return {"max_abs": maximum, "rms_relative": relative}


def check_port_product(port_call, port_args, product_call, product_args):
    """One fresh eager call per side; restore both pools even on failure."""
    pools = (port_args[-2:], product_args[-2:])
    originals = [tuple(p.clone() for p in side) for side in pools]
    try:
        actual = port_call(*port_args)
        reference = product_call(*product_args)
        slots = port_args[6].long()
        written = [diff_metrics(a[slots], b[slots]) for a, b in zip(*pools, strict=False)]
        untouched = {}
        for label, side, before in zip(("port", "product"), pools, originals, strict=False):
            mask = torch.ones(side[0].shape[0], dtype=torch.bool, device=side[0].device)
            mask[slots] = False
            untouched[label] = {
                name: torch.equal(pool[mask], original[mask])
                for name, pool, original in zip(("state", "conv_state"), side, before, strict=False)
            }
        return {
            "output": diff_metrics(actual, reference),
            "output_bf16": actual.dtype == torch.bfloat16,
            "state": written[0],
            "conv_state": written[1],
            "untouched_rows": untouched,
            "untouched_rows_ok": all(ok for side in untouched.values() for ok in side.values()),
        }
    finally:
        for side, before in zip(pools, originals, strict=False):
            for pool, original in zip(side, before, strict=False):
                pool.copy_(original)


def _case(hk, hv, lengths, initial, seed, device):
    g = torch.Generator().manual_seed(seed)
    dc = 2 * hk * K + hv * V
    p = dc + hv * V + 2 * hv
    t = sum(lengths)
    n = len(lengths)
    slots = n + 2
    projected = torch.randn(t, p, generator=g).to(torch.bfloat16).to(device)
    conv_weight = (0.5 * torch.randn(dc, 4, generator=g)).to(torch.bfloat16).to(device)
    A_log = (torch.rand(hv, generator=g) * 2.8 - 0.7).to(device)
    dt_bias = (-2 + 4 * torch.rand(hv, generator=g)).to(device)
    norm_weight = (1 + 0.1 * torch.randn(V, generator=g)).to(device)
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device
    )
    slot_indices = torch.arange(1, n + 1, dtype=torch.int32, device=device)
    has_init = torch.tensor(initial, dtype=torch.bool, device=device)
    state = (0.2 * torch.randn(slots, hv, V, K, generator=g)).to(device)
    conv_state = torch.randn(slots, 3, dc, generator=g).to(torch.bfloat16).to(device)
    return (
        projected,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        cu,
        slot_indices,
        has_init,
        state,
        conv_state,
    )


def _provider(name, hk, hv, device, *, port_options=None):
    """(prepare(args) -> call args, call(*args) -> y, label)."""
    ident = lambda a: a
    if name == "port":
        try:
            from ctikernels.gdn.chunk_ws.prefill import PortPrefill
        except ImportError:
            raise PortPrefillUnavailableError(PORT_UNAVAILABLE) from None
        op = PortPrefill(num_k_heads=hk, num_v_heads=hv, device=device, **(port_options or {}))
        return ident, op, "port"

    from ctikernels.gdn import GatedDeltaNetPrefill, _fla
    from ctikernels.gdn._contract import Geometry

    if name == "product":
        op = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=device)
        label = op.provider if hasattr(op, "provider") else "fla-core"
        # TP2 (Hk=8) v1 is not CuTe on this branch (rc-0.1.13 ships it); measure what it is.
        sm120 = torch.cuda.get_device_capability(device) == (12, 0) and hk != 8
        if sm120 and label != "cute":
            raise RuntimeError("Product v1 requires provider cute on sm120")

        def product_call(*a):
            if sm120 and hasattr(op, "selected_provider"):
                if op.selected_provider(a[0].shape[0], a[6].numel()) != "cute":
                    raise RuntimeError("Product v1 requires row provider cute on sm120")
            return op(*a)

        return ident, product_call, label
    if name == "fla":
        kernel = _fla.load("GatedDeltaNetPrefill", device).prefill
        geo = Geometry(hk, hv)
        return ident, lambda *a: kernel(geo, *a), "fla-core"
    if name in ("g305", "gb10"):
        path = os.path.join(BUNDLE, "cells", name)
        for mod in (
            "solution",
            "gdn_ws",
            "gdn_pipe",
            "gdn_c32",
            "gdn_c64",
            "gdn_cl",
            "gdn_gluon",
            "gdn_prefill",
        ):
            sys.modules.pop(mod, None)
        sys.path.insert(0, path)
        import importlib

        sol = importlib.import_module("solution")
        sys.path.remove(path)
        run = sol.run
        return ident, lambda *a: run(*a)[0], name
    if name in ("sglang", "sglang72", "sglang72chunk", "vllm", "vllm31", "flachunk"):
        path = os.path.join(BUNDLE, "market")
        if path not in sys.path:
            sys.path.insert(0, path)
        import importlib

        module = {"flachunk": "fla_chunk", "sglang72chunk": "sglang72_chunk"}.get(
            name, name + "_triton"
        )
        mod = importlib.import_module(module)
        if name == "flachunk":
            return (lambda a: mod.prepare(*a)), (lambda *a: mod.run(*a)[0]), name
        return (lambda a: mod.prepare(*a)), (lambda *a: mod.run(*a)), name
    raise ValueError(name)


def _time(call, args, reps, flush):
    call(*args)
    call(*args)
    torch.cuda.synchronize()
    graph = None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            call(*args)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(*args)
        torch.cuda.synchronize()
        run = graph.replay
        mode = "graph"
    except Exception as e:  # noqa: BLE001
        print("# graph capture failed:", type(e).__name__, file=sys.stderr)
        torch.cuda.synchronize()
        run = lambda: call(*args)
        mode = "eager"
    times = []
    for _ in range(reps):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    del graph
    times.sort()
    return times[len(times) // 2], times[0], mode


def main():
    ap = QuietParser(prog="whole_prefill.py", description=__doc__)
    ap.add_argument("--providers", default="product,sglang72,vllm31")
    ap.add_argument("--port-options", type=parse_port_options, default={}, metavar="k=v,k=v")
    ap.add_argument("--geoms", default="16x32,16x48")
    ap.add_argument("--set", choices=("small", "long", "all"), default="all")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="", help="comma list of case indices")
    args = ap.parse_args()
    cases = list(enumerate({"small": SMALL, "long": LONG, "all": SMALL + LONG}[args.set]))
    providers = args.providers.split(",")
    try:
        if args.reps < 1 or any(
            p
            not in {
                "port",
                "product",
                "fla",
                "g305",
                "gb10",
                "sglang",
                "sglang72",
                "sglang72chunk",
                "vllm",
                "vllm31",
                "flachunk",
            }
            for p in providers
        ):
            raise ValueError
        geoms = [tuple(map(int, geom.split("x"))) for geom in args.geoms.split(",")]
        if any(len(g) != 2 or min(g) < 1 or g[1] % g[0] for g in geoms):
            raise ValueError
        if args.only:
            selected = list(map(int, args.only.split(",")))
            if any(i < 0 or i >= len(cases) for i in selected):
                raise ValueError
            cases = [cases[i] for i in selected]
    except ValueError:
        ap.error("Invalid workload selection")
    if not torch.cuda.is_available():
        print("# CUDA GPU required; no benchmark executed", flush=True)
        return 1
    device = torch.device("cuda")
    name = torch.cuda.get_device_name()
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    failed = False
    for hk, hv in geoms:
        geom = f"{hk}x{hv}"
        provs = {}
        for p in providers:
            try:
                provs[p] = _provider(p, hk, hv, device, port_options=args.port_options)
            except Exception as e:  # noqa: BLE001
                print(
                    json.dumps({"error": type(e).__name__}),
                    flush=True,
                )
                if isinstance(e, PortPrefillUnavailableError):
                    print("# " + PORT_UNAVAILABLE, flush=True)
                return 1
        for ci, (lengths, initial) in cases:
            ref_y = None
            for p, (prepare, call, label) in provs.items():
                base = _case(hk, hv, lengths, initial, 1234 + ci, device)
                rec = {
                    "device": name,
                    "geom": geom,
                    "case": ci,
                    "T": sum(lengths),
                    "N": len(lengths),
                    "lengths": lengths if len(lengths) <= 8 else None,
                    "provider": p,
                    "label": label,
                }
                try:
                    prepared = prepare(base)
                    if p == "port":
                        product_prepare, product_call, _ = _provider("product", hk, hv, device)
                        fresh = _case(hk, hv, lengths, initial, 1234 + ci, device)
                        rec["port_vs_product"] = check_port_product(
                            call, prepared, product_call, product_prepare(fresh)
                        )
                        del fresh, product_call, product_prepare
                    if args.check:
                        fresh = _case(hk, hv, lengths, initial, 1234 + ci, device)
                        check_args = prepare(fresh)
                        y = call(*check_args)
                        if p == "sglang72chunk":
                            from sglang72_chunk import finish

                            y = finish(y, check_args)[0]
                        y = y.float()
                        torch.cuda.synchronize()
                        if ref_y is None:
                            ref_y = y
                        else:
                            d = (y - ref_y).abs()
                            rec["max_abs_vs_first"] = float(d.max())
                            rec["rms_vs_first"] = float(d.pow(2).mean().sqrt())
                        rec["finite"] = bool(torch.isfinite(y).all())
                        del fresh, y, check_args
                    med, best, mode = _time(call, prepared, args.reps, flush)
                    rec.update(us=round(med, 2), us_min=round(best, 2), mode=mode)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = type(e).__name__
                    failed = True
                print(json.dumps(rec), flush=True)
                del base
                torch.cuda.empty_cache()
    return int(failed)


if __name__ == "__main__":
    t0 = time.time()
    try:
        status = main()
    except Exception as exc:  # noqa: BLE001 - sanitize diagnostics
        print(f"# benchmark stopped ({type(exc).__name__})", flush=True)
        status = 1
    print(f"# elapsed {time.time() - t0:.0f}s", file=sys.stderr)
    raise SystemExit(status)
