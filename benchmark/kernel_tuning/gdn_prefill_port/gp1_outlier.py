# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""GP1 precision outlier probe: where the port's largest y/state errors sit vs fla-core and v1.

For each case (Hk,Hv,kind,lengths,seed index) runs fla-core, v1 and the port on the
frozen-gate inputs and prints, per provider, the max |error| vs FP64 and its location
(token, value head, column), the FP64 value there, |z| gate and the per-token RMS of the FP64
pre-norm path proxy (RMS of y over the head's columns).
"""

import json
import os
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
if not (root / "tests/gpu/_gdn_reference.py").is_file():
    root = Path.cwd()
sys.path.insert(0, str(root))
os.environ["CTI_GDN_PREFILL_CUTE"] = "1"
os.environ["CTI_GDN_PREFILL_PORT"] = "0"

from ctikernels.gdn import GatedDeltaNetPrefill  # noqa: E402
from ctikernels.gdn._contract import Geometry  # noqa: E402
from ctikernels.gdn.chunk_ws.prefill import PortPrefill  # noqa: E402
from tests.gpu import _gdn_reference as reference  # noqa: E402
from tests.gpu._gdn_prefill_cases import _fla_prefill  # noqa: E402
from tests.gpu.test_gdn_cute_prefill import _precision_inputs  # noqa: E402

CASES = [
    (8, 16, "ordinary", [64], 2),
    (8, 16, "ordinary", [1, 2, 61], 2),
    (8, 24, "heavy", [1024], 4),
    (8, 24, "ordinary", [1, 2, 61], 1),
    (16, 32, "ordinary", [1, 2, 61], None),
]


def where(err, shape_hv):
    flat = int(err.argmax())
    tok, col = divmod(flat, err.shape[1])
    return tok, col // 128, col % 128


def main():
    dev = torch.device("cuda")
    out = []
    for hk, hv, kind, lengths, seed_pick in CASES:
        v1 = GatedDeltaNetPrefill(num_k_heads=hk, num_v_heads=hv, device=dev)
        port = PortPrefill(hk, hv, dev, cp=False)
        seeds = range(5) if seed_pick is None else [seed_pick]
        for seed in seeds:
            weights, x, meta, base, pools, written = _precision_inputs(hk, hv, lengths, kind, seed)
            exact = reference.prefill(x, weights, hk, *meta, *base)
            ey, es = exact[0].double(), exact[1].double()
            rows = {}
            with torch.inference_mode():
                fp = [p.clone() for p in base]
                rows["fla"] = (_fla_prefill(Geometry(hk, hv), x, *weights.args(), *meta, *fp), fp)
                vp = [p.clone() for p in pools]
                rows["v1"] = (v1(x, *weights.args(), *meta, *vp), vp)
                pp = [p.clone() for p in pools]
                rows["port"] = (port(x, *weights.args(), *meta, *pp), pp)
            torch.cuda.synchronize()
            for name, (y, p) in rows.items():
                err = (y.double() - ey).abs()
                tok, head, col = where(err, hv)
                serr = (p[0][written].double() - es[written]).abs()
                rec = {
                    "case": [hk, hv, kind, lengths, seed],
                    "provider": name,
                    "y_max": float(err.max()),
                    "y_rms": float(err.square().mean().sqrt()),
                    "y_at": [tok, head, col],
                    "y_ref": float(ey[tok, head * 128 + col]),
                    "y_head_rms_ref": float(
                        ey[tok, head * 128 : (head + 1) * 128].square().mean().sqrt()
                    ),
                    "y_got": float(y[tok, head * 128 + col]),
                    "state_max": float(serr.max()),
                    "state_rms": float(serr.square().mean().sqrt()),
                    "state_at": [int(i) for i in torch.unravel_index(serr.argmax(), serr.shape)],
                }
                print("OUTLIER " + json.dumps(rec), flush=True)
                out.append(rec)


if __name__ == "__main__":
    main()
