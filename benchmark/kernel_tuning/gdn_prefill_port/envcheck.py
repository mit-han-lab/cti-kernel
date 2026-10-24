# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 cti-engine contributors

"""Print the node's GPU, clocks and the FlashInfer / CuTe DSL versions the harness will use."""

import importlib.metadata as md
import json

import torch

out = {"gpu": torch.cuda.get_device_name(), "cc": torch.cuda.get_device_capability()}
for name in ("flashinfer-python", "nvidia-cutlass-dsl", "cuda-python", "cuda-core", "torch"):
    try:
        out[name] = md.version(name)
    except md.PackageNotFoundError:
        out[name] = None
import flashinfer  # noqa: E402
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm120  # noqa: E402,F401

out["flashinfer_file"] = flashinfer.__version__
print("ENVCHECK " + json.dumps(out), flush=True)
