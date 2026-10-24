# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Bounded compute-sanitizer driver for the fused GDN decode.

Run ``compute-sanitizer --tool memcheck python -m tests.gpu.sanitize_gdn_decode`` on an
accepted device. Each launch configuration class of the device runs one eager call on a warm
pool and one CUDA graph replay on a fresh pool, with a padding row; finisher classes only where
the grid fits one CTA per SM (instrumentation must not break the whole-grid residency finishers
rely on; flag configurations need only a resident prefix).
No GPU work happens at import, and pytest collects no tests from this module.
"""

import torch

from ctikernels.gdn import GatedDeltaNetDecode, _triton_decode
from ctikernels.gdn._contract import Geometry

from .test_gdn import _pools, _projected, _weights
from .test_gdn_triton import _config_classes, _launch_traits, _slots


def main() -> None:
    device = torch.device("cuda")
    traits = _launch_traits(device)
    sms = traits["sms"]
    runs = skipped = 0
    for k_heads, v_heads, batch in _config_classes().values():
        plan = _triton_decode.decode_plan(Geometry(k_heads, v_heads), batch, 1, True, **traits)
        if plan.arguments["FIN"] and plan.grid[0] > sms:
            skipped += 1
            continue
        slots = batch + 4
        weights = _weights(k_heads, v_heads, seed=2000)
        op = GatedDeltaNetDecode(num_k_heads=k_heads, num_v_heads=v_heads, device=device)
        projected = _projected(batch, k_heads, v_heads, seed=2001)
        slot_indices = _slots(batch, slots, seed=2002, padding=[batch - 1] if batch > 1 else [])
        with torch.inference_mode():
            op(projected, *weights.args(), slot_indices, *_pools(k_heads, v_heads, 2003, slots))
            pools = _pools(k_heads, v_heads, 2004, slots)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                op(projected, *weights.args(), slot_indices, *pools)
            graph.replay()
        torch.cuda.synchronize()
        runs += 1
    print(
        f"sanitize_gdn_decode: {runs} classes run, {skipped} finisher classes beyond one wave skipped"
    )


if __name__ == "__main__":
    main()
