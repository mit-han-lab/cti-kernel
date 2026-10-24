# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

# Two-GPU qualification

Requires two sm_120/sm_121 CUDA GPUs, the `cute` extra, and the NCCL extension.
The target qualification device is an RTX PRO 6000 pair. Run explicitly:

```sh
pytest tests/multi_gpu -s
```

This directory is outside the default testpaths and the single-GPU suite.
It skips when fewer than two CUDA devices are available. One spawned process
pair runs the shape sweep, padded-stride and sliced inputs, layout rejection,
graph/eager interleaving, and NCCL fallback. Residuals must match the FP32
rank-order reference bitwise. Both outputs must pass the measured floor: their
maximum and RMS errors to an FP64 oracle may not exceed those of the market
composition (NCCL all-reduce, BF16 residual add, RMSNorm) on the same inputs.
Each case also reports the BF16 ULP distance to the FP32 reference. Both ranks
must produce identical hashes. Peer call ordering
is collective: interruption or a rank that skips a call can leave a GPU spinning.
