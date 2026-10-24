# cti-kernels

PyTorch GPU operations for continuous training and inference. The package owns
numerical implementations and provider wrappers; it does not own request
scheduling or model lifecycle, and it does not depend on the engine.

## Modules

Everything is under `ctikernels`:

| Module | Contents |
| --- | --- |
| `attention` | Paged and fixed-query attention, packed/varlen attention, DeepSeek-V4 sparse attention, DFlash draft attention |
| `kvcache` | Row copies across state pools (prefix-hit state restore) |
| `gdn` | Gated DeltaNet decode, chunked prefill and training forward/backward |
| `linear` | Dense layers: block-FP8, static FP8 and NVFP4 GEMMs, with LoRA |
| `mlp` | Gated MLP, including training |
| `moe` | MoE experts (BF16, block-FP8, W4A8), routing, MoE-LoRA and the MoE-block training op |
| `activation`, `norm`, `rotary`, `embedding` | Gated activations; RMS norm and add + LayerNorm; rotary embedding; vision position-embedding resample |
| `sampling`, `logprob` | Sampling, logit steering and penalties, top-k by id, selected and top log-probabilities |
| `collective` | All-reduce + add + RMSNorm |
| `dsv4_indexer`, `mhc` | DeepSeek-V4 KV compressor and indexer; mHC |
| `speculative` | DFlash round bookkeeping and the DFlash2 block convolution |

Top-level modules add `store` (KV-cache stores), `radix`, `index`, `pynccl`, and
`cross_entropy` and `vocab_parallel` (chunked and vocabulary-parallel cross-entropy).

## Version and dependencies

This tree is cti-kernels 0.0.1. It requires Python 3.10 or newer and
pins (`pyproject.toml`):

| Dependency | Version |
| --- | --- |
| torch | 2.13.0 |
| triton | >=3.7.1,<3.8 |
| flashinfer-python | 0.7.0.post1 |
| sglang-kernel | 0.4.6.post1 |
| liger-kernel | 0.8.0 |
| apache-tvm-ffi | >=0.1.4 |
| quack-kernels | 0.6.5 |

Extras:

- `cute`: nvidia-cutlass-dsl[cu13] 4.8.0 with its CUDA 13 libraries. The base dependencies already install
  CuTe DSL, but the CuTe DSL kernels require exactly this version.
- `gdn`: fla-core 0.5.2, the Gated DeltaNet fallback provider.
- `varlen`: flash-attn 2.8.3.post1, for packed/varlen attention and training rotary. Use the prebuilt
  wheels below, or build it for the target GPU after Torch is installed.
- `dev`: test, lint and build tools.

## Devices

Releases are gated on the RTX 5090 and RTX PRO 6000 Server Edition (sm_120) and
on GB10 (sm_121). The RTX PRO 6000 Workstation and Max-Q editions are not gated:
every op runs on them, and ops whose CuTe or Gluon variants are validated only on
the Server Edition use their generic Triton or FlashInfer paths. GB300 (sm_103)
paths exist but were not re-run for this release. Other CUDA devices use the
portable Triton paths, with fla-core for Gated DeltaNet below compute capability
8.9; they are not qualified. A successful import does not qualify a device or
provider.

## Install

Wheels are published as GitHub releases of
[mit-han-lab/cti-kernel](https://github.com/mit-han-lab/cti-kernel/releases), not on a
package index. Download the wheel, check it against the SHA256 listed on the
release page, then install it:

```bash
VERSION=0.0.1
WHEEL=cti_kernels-$VERSION-py3-none-any.whl
curl -LO "https://github.com/mit-han-lab/cti-kernel/releases/download/v$VERSION/$WHEEL"
echo "<sha256 from the release page>  $WHEEL" | sha256sum --check --strict -
python -m pip install "./$WHEEL[cute,gdn]"
```

### flash-attn

The release also carries prebuilt flash-attn 2.8.3.post1 wheels for the `varlen` extra:
`flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl` and
`flash_attn-2.8.3.post1-cp312-cp312-linux_aarch64.whl`. They are built for Python 3.12,
Torch 2.13.0 and CUDA 13.0, and contain sm_120 code only, so they run on RTX 5090,
RTX PRO 6000 and GB10 and on no other GPU. The x86_64 wheel was tested on RTX PRO 6000
(varlen forward and backward, BF16 and FP16, causal and non-causal), and the aarch64 wheel
on GB10 (varlen). For other setups, build flash-attn from source with
`--no-build-isolation` after Torch is installed.

```bash
FA=flash_attn-2.8.3.post1-cp312-cp312-linux_$(uname -m).whl
curl -LO "https://github.com/mit-han-lab/cti-kernel/releases/download/v$VERSION/$FA"
echo "<sha256 from the release page>  $FA" | sha256sum --check --strict -
python -m pip install "./$FA"
```

To build from source and run the tests, see the [cookbook](cookbook/README.md).

## Usage

Each module exposes operation objects or functions with documented contracts, for
example `ctikernels.sampling.RequestLocalCategorical` or
`ctikernels.gdn.GatedDeltaNetPrefill`. Most ops compile on first use: call their
`warm_up` or `prewarm` before CUDA graph capture. The cookbook has a runnable
sampling example and the profiling commands.

## Documentation

- [Test and benchmark standard](docs/testing.md)
- [Development and release](docs/development.md)

Kernel releases are versioned independently. Each engine release selects a
reviewed kernel version.

## Acknowledgments

Many kernels in this package were developed and tuned with [KDA](https://github.com/NVlabs/kda).

## License

MIT for cti-kernels' own code; files adapted from third-party projects keep their
upstream licenses — see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Each file's
`SPDX-License-Identifier` line names the license that applies to it.

Original code and modifications are copyright 2026 cti-engine contributors.
The package adapts code or designs from:

- Mini-SGLang and SGLang, including SGLang's tuned fused-MoE configs;
- Marlin (IST-DASLab; Elias Frantar; modified by Neural Magic), received through SGLang;
- vLLM, the Triton tutorials, Liger Kernel, Flash Linear Attention and FlashMLA;
- FlashInfer, including the 0.7.1 W4A8 MoE port;
- QuACK and NVIDIA CUTLASS;
- Unsloth and NVIDIA NCCL.

It credits Efficient Cross Entropy for the chunked cross-entropy idea (via Liger)
and follows Unsloth Zoo's MoE training numerics as a reference. Adapted files keep
their upstream copyright lines, for example the NCCL header generated from NCCL's
`nccl.h.in`, which keeps BSD-3-Clause.

Dependencies keep their own licenses; importing them does not make them vendored
sources. The `cute` extra installs NVIDIA's CuTe DSL (`nvidia-cutlass-dsl`), which
is distributed under NVIDIA's own license terms.
