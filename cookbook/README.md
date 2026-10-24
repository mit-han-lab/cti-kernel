# Cookbook

Run these commands from the repository root. See the
[development notes](../docs/development.md) for contributing and packaging.

## Install and test

This repository is an independent build root. In a compatible environment, run
the following from its root, with `setuptools>=77` and `wheel` installed
(required by `--no-build-isolation`):

```bash
python -m pip install --no-build-isolation -e '.[dev,cute]'
python -m pytest -q tests/modules tests/benchmarks tests/quality tests/ci tests/norm -m "not gpu"
```

The `cute` extra installs CuTe DSL for the CuTe kernels; drop it to run only the
portable paths. Training rotary and packed/varlen attention additionally need Dao
FlashAttention built for the target GPU after Torch is installed. Gated DeltaNet
(`ctikernels.gdn`) uses its own kernels on compute capability 8.9 and newer; the
GPU suite also compares against fla-core, the pure-Python `gdn` extra pinned in
`pyproject.toml` and the provider below 8.9. Install both before the GPU suite:

```bash
python -m pip install --no-build-isolation 'flash-attn==2.8.3.post1'
python -m pip install --no-deps 'fla-core==0.5.2'
python -m pytest -q tests/gpu tests/norm
```

Consumers qualify native device coverage with their selected kernel package and
provider versions. A successful package import does not qualify an unavailable
provider or GPU architecture. The GPU suite requires the installed numerical
providers and a CUDA device; CPU contract tests do not substitute for it.

## Request-local filtered sampling

```python
import torch
from ctikernels.sampling import RequestLocalCategorical

probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device="cuda")
operation = RequestLocalCategorical()
tokens = operation.sample(
    probabilities,
    seeds=(42,),
    positions=(0,),
    top_k=torch.tensor([3], dtype=torch.int32, device="cuda"),
    top_p=torch.tensor([0.75], device="cuda"),
)
```

Only candidates 2 and 3 remain after top-k-then-top-p filtering. Increment the
response position for the next draw; replay the same seed/position and probability
row to reproduce a draw. This does not guarantee model-level batch invariance.

## Profile an operation

Kernel tuning stays next to this package, independently of engine benchmarks:

```bash
python -m benchmark.kernel_tuning.lora --help
python -m benchmark.kernel_tuning.sampling_pdl --help
```

These tools compare implementations on the current GPU; they do not update the
runtime selection policy automatically. LoRA receipts include the measured shapes,
source fingerprints and correctness checks. Commit a reviewed candidate before
collecting reproducible performance evidence.
