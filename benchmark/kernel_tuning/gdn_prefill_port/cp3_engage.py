# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Diagnose the CP path on the frozen precision inputs: per-segment differences and workspace use."""

import sys
from pathlib import Path

import torch

from ctikernels.gdn.chunk_ws import _cp
from ctikernels.gdn.chunk_ws.prefill import PortPrefill, compile_key

reference_root = Path(__file__).resolve().parents[3]
if not (reference_root / "tests/gpu/_gdn_reference.py").is_file():
    reference_root = Path.cwd()
sys.path.insert(0, str(reference_root))
from tests.gpu.test_gdn_cute_prefill import _precision_inputs  # noqa: E402


def run(op, x, weights, meta, pools):
    p = [q.clone() for q in pools]
    y = op(x, *weights.args(), *meta, *p)
    torch.cuda.synchronize()
    return y.float(), p


@torch.inference_mode()
def main():
    device = torch.device("cuda", torch.cuda.current_device())
    for hk, hv, lengths, segs in ((16, 48, [512], (2, 3)), (16, 32, [4096], (4,))):
        weights, x, meta, base, pools, written = _precision_inputs(hk, hv, lengths, "ordinary", 0)
        t = x.shape[0]
        ref = PortPrefill(num_k_heads=hk, num_v_heads=hv, device=device, cp=False)
        y0, p0 = run(ref, x, weights, meta, pools)
        for s in segs:
            op = PortPrefill(num_k_heads=hk, num_v_heads=hv, device=device, cp_segments=s)
            j, seg = _cp.cp_plan(t, hv, op.num_sms, s)
            y1, p1 = run(op, x, weights, meta, pools)
            per = [(y1[a : a + seg] - y0[a : a + seg]).abs().max().item() for a in range(0, t, seg)]
            ds = (p1[0][written] - p0[0][written]).abs().max().item()
            w = op.cp_workspace
            norms = [[round(w[plane, k].norm().item(), 4) for k in range(j)] for plane in range(3)]
            print(
                "SEG",
                hv,
                lengths,
                "J",
                j,
                "L",
                seg,
                "y diff per segment",
                per,
                "state diff",
                ds,
                "W norms",
                norms,
                flush=True,
            )
            # Sequential reference states at segment starts (non-CP prefix runs).
            for k in range(1, j):
                cu = torch.tensor([0, k * seg], dtype=torch.int32, device=device)
                pk = [q.clone() for q in pools]
                ref(x[: k * seg], *weights.args(), cu, meta[1], meta[2], *pk)
                torch.cuda.synchronize()
                slot = written[0]
                diff = (w[0, k] - pk[0][slot]).abs().max().item()
                print(
                    "  start",
                    k,
                    "W0 vs sequential max diff",
                    diff,
                    "seq norm",
                    round(pk[0][slot].norm().item(), 4),
                    flush=True,
                )
            # Perturb W[0,1] and relaunch only C: segment 1's y must change.
            key = compile_key(op.hk, op.hv, x.stride(0), op.port_options, "seg")
            chunk, _ = op.compiled[key]
            p2 = [q.clone() for q in pools]
            out = torch.empty_like(y1, dtype=torch.bfloat16)
            w[0, 1] += 1.0
            cu0, slots0, init0 = meta
            chunk(
                x,
                *weights.args(),
                cu0,
                slots0,
                init0.view(torch.int8),
                *p2,
                out,
                op.tensormaps,
                w,
                seg,
                j,
            )
            torch.cuda.synchronize()
            per2 = [
                (out.float()[a : a + seg] - y1[a : a + seg]).abs().max().item()
                for a in range(0, t, seg)
            ]
            print("  perturbed W0[1]: y change per segment", per2, flush=True)


main()
